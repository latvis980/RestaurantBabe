# telegram_bot.py - Updated with location button functionality
import telebot
from telebot import types  # Add this import for inline keyboards
import logging
import time
import threading
import json
import asyncio
from threading import Event
from typing import Dict, List, Any, Optional, Tuple
from utils.voice_handler import VoiceMessageHandler


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from utils.orchestrator_manager import get_orchestrator

from utils.telegram_location_handler import TelegramLocationHandler, LocationData
from location.location_analyzer import LocationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize bot
bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize AI conversation handler
conversation_ai = ChatOpenAI(
    model=config.OPENAI_MODEL,
    temperature=0.3
)

# Simple conversation history storage
user_conversations = {}

# CANCEL COMMAND FUNCTIONALITY
# Track active searches and their cancellation events
active_searches = {}  # user_id -> {"thread": thread_object, "cancel_event": Event, "chat_id": chat_id}

location_handler = TelegramLocationHandler()
location_analyzer = None  # Will be initialized in main()

# Track users waiting for location input - UPDATED to include button tracking
users_awaiting_location = {}  # user_id -> {"query": str, "timestamp": float, "has_button": bool}

voice_handler = None  # Will be initialized in main()

def create_cancel_event(user_id, chat_id):
    """Create a cancellation event for a user's search"""
    cancel_event = Event()
    active_searches[user_id] = {
        "cancel_event": cancel_event,
        "chat_id": chat_id,
        "start_time": time.time()
    }
    logger.info(f"Created cancel event for user {user_id}")
    return cancel_event

def cleanup_search(user_id):
    """Clean up search tracking for a user"""
    if user_id in active_searches:
        del active_searches[user_id]
        logger.info(f"Cleaned up search tracking for user {user_id}")

def is_search_cancelled(user_id):
    """Check if search has been cancelled for this user"""
    if user_id in active_searches:
        return active_searches[user_id]["cancel_event"].is_set()
    return False

def create_location_button():
    """Create the 'Send my coordinates' button"""
    keyboard = types.ReplyKeyboardMarkup(
        one_time_keyboard=True,
        resize_keyboard=True,
        row_width=1
    )

    # Create location button
    location_button = types.KeyboardButton(
        "📍 Send my coordinates",
        request_location=True
    )

    keyboard.add(location_button)
    return keyboard

def remove_location_button():
    """Remove the location button (return to normal keyboard)"""
    return types.ReplyKeyboardRemove()

# Welcome message (unchanged)
WELCOME_MESSAGE = (
    "🍸 Hello! I'm an AI assistant, Restaurant Babe, but friend call me Babe. I know all about the most delicious and trendy restaurants, cafes, bakeries, bars, and coffee shops around the world.\n\n"
    "Tell me what you are looking for, like <i>best specialty coffee places in Berlin</i>. Or I can just search for good places around you.\n\n"

    "I will check with my restaurant critic friends and provide the best recommendations. This might take a couple of minutes because I search very carefully and thoroughly verify the results. But there won't be any random places in my list.\n\n"
    "I understand voice messages, too!\n\n"
    "💡 <b>Tip:</b> If you change your mind while I'm searching, just type /cancel to stop the current search.\n\n"
    "Shall we begin?"
)

# AI Conversation Prompt (unchanged)
CONVERSATION_PROMPT = """
You are an expert AI assistant for restaurant recommendations worldwide. Your name is Restaurant Babe, or simply Babe. You are very friendly, helpful, and enthusiastic about food and dining. You are helping a user find the best restaurants, cafes, bars, and other dining spots based on their preferences and location.

CONVERSATION HISTORY:
{{conversation_history}}

CURRENT USER MESSAGE: {{user_message}}

TASK: Analyze the conversation and decide what to do next. You need TWO pieces of information:
1. LOCATION (city/neighborhood/area)  
2. DINING PREFERENCE (cuisine type, restaurant style, or specific request)

DECISION RULES:
- If you have BOTH location AND dining preference → Action: "SEARCH"
- If missing one or both pieces → Action: "CLARIFY"
- If completely off-topic → Action: "REDIRECT"

RESPONSE FORMAT (JSON only):
{{
    "action": "SEARCH" | "CLARIFY" | "REDIRECT",
    "search_query": "complete restaurant search query (only if action is SEARCH)",
    "bot_response": "what to say to the user",
    "reasoning": "brief explanation"
}}

EXAMPLES:
User: "ramen in tokyo" → {{"action": "SEARCH", "search_query": "best ramen restaurants in Tokyo", "bot_response": "Perfect! Let me find the best ramen places in Tokyo for you."}}

User: "I want something romantic" → {{"action": "CLARIFY", "bot_response": "Romantic sounds wonderful! Which city would you like me to search for romantic restaurants?"}}

User: "How's the weather?" → {{"action": "REDIRECT", "bot_response": "I specialize in restaurant recommendations! Could you tell me what city you're interested in and what type of dining you're looking for?"}}
"""

# Conversation history functions (unchanged)
def add_to_conversation(user_id, message, is_user=True):
    """Add message to user's conversation history"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    user_conversations[user_id].append({
        "role": "user" if is_user else "assistant",
        "message": message,
        "timestamp": time.time()
    })

    # Keep only last 10 messages to avoid too much context
    if len(user_conversations[user_id]) > 10:
        user_conversations[user_id] = user_conversations[user_id][-10:]

def format_conversation_history(user_id):
    """Format conversation history for AI prompt"""
    if user_id not in user_conversations:
        return "No previous conversation."

    formatted = []
    for msg in user_conversations[user_id]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['message']}")

    return "\n".join(formatted)

# Bot command handlers
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Send welcome message"""
    user_id = message.from_user.id

    # Clear conversation history for fresh start
    if user_id in user_conversations:
        del user_conversations[user_id]

    # Cancel any active search for this user
    if user_id in active_searches:
        active_searches[user_id]["cancel_event"].set()
        cleanup_search(user_id)

    # Clear any waiting location state
    if user_id in users_awaiting_location:
        del users_awaiting_location[user_id]

    bot.reply_to(message, WELCOME_MESSAGE, parse_mode='HTML', reply_markup=remove_location_button())

@bot.message_handler(commands=['cancel'])
def handle_cancel(message):
    """Handle /cancel command to stop current search"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"Cancel command received from user {user_id}")

    # Check if user has an active search
    if user_id not in active_searches:
        bot.reply_to(
            message,
            "🤷‍♀️ I'm not currently searching for anything for you. Feel free to ask me about restaurants!",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )
        return

    # Cancel the search
    search_info = active_searches[user_id]
    search_info["cancel_event"].set()

    # Calculate how long the search was running
    search_duration = round(time.time() - search_info["start_time"], 1)

    # Send cancellation confirmation
    bot.reply_to(
        message,
        f"✋ Search cancelled! I was searching for {search_duration} seconds.\n\n"
        "What else would you like me to help you find?",
        parse_mode='HTML',
        reply_markup=remove_location_button()
    )

    # Clean up
    cleanup_search(user_id)
    if user_id in users_awaiting_location:
        del users_awaiting_location[user_id]

    # Add cancellation to conversation history
    add_to_conversation(user_id, "Search cancelled by user", is_user=False)

    logger.info(f"Successfully cancelled search for user {user_id} after {search_duration}s")

@bot.message_handler(content_types=['location'])
def handle_location_message(message):
    """Handle GPS location messages from users"""
    try:
        user_id = message.from_user.id

        # Extract GPS coordinates
        latitude = message.location.latitude
        longitude = message.location.longitude

        logger.info(f"Received GPS location from user {user_id}: {latitude}, {longitude}")

        # Validate coordinates
        if not location_handler.validate_gps_coordinates(latitude, longitude):
            bot.reply_to(
                message,
                "❌ The location coordinates seem invalid. Could you try sending your location again or describe where you are?",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
            return

        # Process GPS location
        location_data = location_handler.process_gps_location(latitude, longitude)

        # Check if user was waiting for location input
        if user_id in users_awaiting_location:
            # Get the original query
            awaiting_data = users_awaiting_location[user_id]
            original_query = awaiting_data["query"]

            # Remove from waiting list
            del users_awaiting_location[user_id]

            # Confirm location received and remove button
            bot.reply_to(
                message,
                f"📍 <b>Perfect! I received your location.</b>\n\n"
                f"🔍 Now searching for: <i>{original_query}</i>\n\n"
                "⏱ This might take a minute while I find the best nearby places...",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

            # Start location-based search
            threading.Thread(
                target=perform_location_search,
                args=(original_query, location_data, message.chat.id, user_id),
                daemon=True
            ).start()

        else:
            # User sent location without context - ask what they're looking for
            bot.reply_to(
                message,
                f"📍 <b>Great! I have your location.</b>\n\n"
                f"📌 <i>{location_handler.format_location_summary(location_data)}</i>\n\n"
                "What type of restaurants or bars are you looking for nearby?\n\n"
                "<i>Examples: \"natural wine bars\", \"good coffee\", \"romantic dinner\"</i>",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

            # Store location for next message
            user_conversations[user_id] = user_conversations.get(user_id, [])
            user_conversations[user_id].append({
                "role": "system",
                "message": f"User shared GPS location: {latitude}, {longitude}",
                "timestamp": time.time(),
                "location_data": location_data
            })

    except Exception as e:
        logger.error(f"Error handling location message: {e}")
        bot.reply_to(
            message,
            "😔 Sorry, I had trouble processing your location. Could you try again or describe where you are?",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )

# PRESERVED TEST COMMANDS - These are critical for debugging!

@bot.message_handler(commands=['test_scrape'])
def handle_test_scrape(message):
    """Handle /test_scrape command - PRESERVED FOR DEBUGGING"""
    user_id = message.from_user.id
    admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

    # Check if user is admin
    if not admin_chat_id or str(user_id) != str(admin_chat_id):
        bot.reply_to(message, "❌ This command is only available to administrators.")
        return

    # Parse command
    command_text = message.text.strip()

    if len(command_text.split(None, 1)) < 2:
        help_text = (
            "🧪 <b>Scraping Process Test</b>\n\n"
            "<b>Usage:</b>\n"
            "<code>/test_scrape [restaurant query]</code>\n\n"
            "<b>Examples:</b>\n"
            "<code>/test_scrape best brunch in Lisbon</code>\n"
            "<code>/test_scrape romantic restaurants Paris</code>\n"
            "<code>/test_scrape family pizza Rome</code>\n\n"
            "This runs the complete scraping process and shows:\n"
            "• Which search results are found\n"
            "• What gets scraped successfully\n"
            "• Exact content that goes to list_analyzer\n"
            "• Scraping method statistics\n\n"
            "📄 Results are saved to a detailed file."
        )
        bot.reply_to(message, help_text, parse_mode='HTML')
        return

    # Extract query
    restaurant_query = command_text.split(None, 1)[1].strip()

    if not restaurant_query:
        bot.reply_to(message, "❌ Please provide a restaurant query to test.")
        return

    # Send confirmation
    bot.reply_to(
        message,
        f"🧪 <b>Starting scraping test...</b>\n\n"
        f"📝 Query: <code>{restaurant_query}</code>\n\n"
        "⏱ Please wait 1-2 minutes...",
        parse_mode='HTML'
    )

    # Run test in background
    def run_scrape_test():
        try:
            from scrape_test import ScrapeTest
            # Use singleton orchestrator
            scrape_tester = ScrapeTest(config, get_orchestrator())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results_path = loop.run_until_complete(
                scrape_tester.test_scraping_process(restaurant_query, bot)
            )

            loop.close()
            logger.info(f"Scraping test completed: {results_path}")
        except Exception as e:
            logger.error(f"Error in scraping test: {e}")
            try:
                bot.send_message(admin_chat_id, f"❌ Scraping test failed: {str(e)}")
            except:
                pass

    threading.Thread(target=run_scrape_test, daemon=True).start()

@bot.message_handler(commands=['test_search'])
def handle_test_search(message):
    """Handle /test_search command - PRESERVED FOR DEBUGGING"""
    user_id = message.from_user.id
    admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

    # Check if user is admin
    if not admin_chat_id or str(user_id) != str(admin_chat_id):
        bot.reply_to(message, "❌ This command is only available to administrators.")
        return

    # Parse command
    command_text = message.text.strip()

    if len(command_text.split(None, 1)) < 2:
        help_text = (
            "🔍 <b>Search Filtering Test</b>\n\n"
            "<b>Usage:</b>\n"
            "<code>/test_search [restaurant query]</code>\n\n"
            "<b>Examples:</b>\n"
            "<code>/test_search best brunch in Lisbon</code>\n"
            "<code>/test_search romantic restaurants Paris</code>\n"
            "<code>/test_search family pizza Rome</code>\n\n"
            "This shows:\n"
            "• What search URLs are found\n"
            "• Which URLs pass filtering\n"
            "• AI evaluation scores\n"
            "• Final URLs sent to scraper\n\n"
            "📄 Results include detailed filtering analysis."
        )
        bot.reply_to(message, help_text, parse_mode='HTML')
        return

    restaurant_query = command_text.split(None, 1)[1].strip()
    if not restaurant_query:
        bot.reply_to(message, "❌ Please provide a restaurant query to test.")
        return

    # Send confirmation
    bot.reply_to(
        message,
        f"🔍 <b>Starting search test...</b>\n\n"
        f"📝 Query: <code>{restaurant_query}</code>\n\n"
        "⏱ Please wait 1-2 minutes...",
        parse_mode='HTML'
    )

    # Run test in background
    def run_search_test():
        try:
            from search_test import SearchTest
            # Use singleton orchestrator
            search_tester = SearchTest(config, get_orchestrator())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results_path = loop.run_until_complete(
                search_tester.test_search_process(restaurant_query, bot)
            )

            loop.close()
            logger.info(f"Search test completed: {results_path}")
        except Exception as e:
            logger.error(f"Error in search test: {e}")
            try:
                bot.send_message(admin_chat_id, f"❌ Search test failed: {str(e)}")
            except:
                pass

    threading.Thread(target=run_search_test, daemon=True).start()

@bot.message_handler(commands=['test_wscrape'])
def handle_test_wscrape(message):
    """Handle /test_wscrape command - Web scraping only test (bypasses database)"""
    user_id = message.from_user.id
    admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

    # Check if user is admin
    if not admin_chat_id or str(user_id) != str(admin_chat_id):
        bot.reply_to(message, "❌ This command is only available to administrators.")
        return

    # Parse command
    command_text = message.text.strip()

    if len(command_text.split(None, 1)) < 2:
        help_text = (
            "🌐 <b>Web Scraping Test (Database Bypassed)</b>\n\n"
            "<b>Usage:</b>\n"
            "<code>/test_wscrape [restaurant query]</code>\n\n"
            "<b>Examples:</b>\n"
            "<code>/test_wscrape best brunch in Lisbon</code>\n"
            "<code>/test_wscrape romantic restaurants Paris</code>\n"
            "<code>/test_wscrape family pizza Rome</code>\n\n"
            "This test <b>BYPASSES the database entirely</b> and shows:\n"
            "• Query analysis and search term generation\n"
            "• Web search results and filtering\n"
            "• Complete scraping process with full content\n"
            "• Scraping method performance analysis\n"
            "• Content ready for editor agent\n\n"
            "🎯 <b>Perfect for debugging web scraping issues!</b>\n\n"
            "📄 Results include detailed analysis with full scraped content."
        )
        bot.reply_to(message, help_text, parse_mode='HTML')
        return

    # Extract query
    restaurant_query = command_text.split(None, 1)[1].strip()

    if not restaurant_query:
        bot.reply_to(message, "❌ Please provide a restaurant query to test.")
        return

    # Send confirmation
    bot.reply_to(
        message,
        f"🌐 <b>Starting web scraping test...</b>\n\n"
        f"📝 Query: <code>{restaurant_query}</code>\n"
        f"🗃️ Database: <b>BYPASSED</b> (forced web search)\n\n"
        "⏱ Please wait 2-3 minutes for complete web analysis...",
        parse_mode='HTML'
    )

    # Run test in background
    def run_web_scrape_test():
        try:
            from web_scrape_test import WebScrapeTest
            # Use singleton orchestrator
            web_scrape_tester = WebScrapeTest(config, get_orchestrator())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results_path = loop.run_until_complete(
                web_scrape_tester.test_web_scraping_only(restaurant_query, bot)
            )

            loop.close()
            logger.info(f"Web scraping test completed: {results_path}")
        except Exception as e:
            logger.error(f"Error in web scraping test: {e}")
            try:
                bot.send_message(admin_chat_id, f"❌ Web scraping test failed: {str(e)}")
            except:
                pass

    threading.Thread(target=run_web_scrape_test, daemon=True).start()

# Add this to your telegram_bot.py file after the existing test commands

@bot.message_handler(commands=['test_editor'])
def handle_test_editor(message):
    """Handle /test_editor command - EDITOR PIPELINE TEST"""
    user_id = message.from_user.id
    admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

    # Check if user is admin
    if not admin_chat_id or str(user_id) != str(admin_chat_id):
        bot.reply_to(message, "❌ This command is only available to administrators.")
        return

    # Parse command
    command_text = message.text.strip()

    if len(command_text.split(None, 1)) < 2:
        help_text = (
            "✏️ <b>Editor Pipeline Test</b>\n\n"
            "<b>Usage:</b>\n"
            "<code>/test_editor [restaurant query]</code>\n\n"
            "<b>Examples:</b>\n"
            "<code>/test_editor greek tavernas in Athens</code>\n"
            "<code>/test_editor best brunch in Lisbon</code>\n"
            "<code>/test_editor romantic restaurants Paris</code>\n\n"
            "This runs the complete editor pipeline:\n"
            "• Web search → scraping → text cleaning\n"
            "• Editor processing with transparency\n"
            "• Generates 2 files: scraped content + edited results\n"
            "• Shows WHY you get 5 vs more restaurants\n\n"
            "🎯 <b>Perfect for debugging editor decisions!</b>"
        )
        bot.reply_to(message, help_text, parse_mode='HTML')
        return

    # Extract query
    restaurant_query = command_text.split(None, 1)[1].strip()

    if not restaurant_query:
        bot.reply_to(message, "❌ Please provide a restaurant query to test.")
        return

    # Send confirmation
    bot.reply_to(
        message,
        f"✏️ <b>Starting editor pipeline test...</b>\n\n"
        f"📝 Query: <code>{restaurant_query}</code>\n\n"
        "⏱ Please wait 2-3 minutes for complete analysis...",
        parse_mode='HTML'
    )

    # Run test in background
    def run_editor_test():
        try:
            from editor_test import EditorTest
            editor_tester = EditorTest(config, get_orchestrator())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results_path = loop.run_until_complete(
                editor_tester.test_complete_editor_pipeline(restaurant_query, bot)
            )

            loop.close()
            logger.info(f"Editor test completed: {results_path}")
        except Exception as e:
            logger.error(f"Error in editor test: {e}")
            try:
                bot.send_message(admin_chat_id, f"❌ Editor test failed: {str(e)}")
            except:
                pass

    threading.Thread(target=run_editor_test, daemon=True).start()

def perform_restaurant_search(search_query, chat_id, user_id):
    """Perform restaurant search using orchestrator with cancellation support"""
    cancel_event = None
    processing_msg = None

    try:
        # Create cancellation event for this search
        cancel_event = create_cancel_event(user_id, chat_id)

        # Send processing message WITH VIDEO
        try:
            with open("media/searching.mp4", "rb") as video:
                processing_msg = bot.send_video(
                    chat_id,
                    video,
                    caption="🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!\n\n💡 Type /cancel if you want to stop the search.",
                    parse_mode='HTML',
                    reply_markup=remove_location_button()
                )
        except FileNotFoundError:
            # Fallback to text message if video file doesn't exist
            logger.warning("Video file not found, sending text message instead")
            processing_msg = bot.send_message(
                chat_id,
                "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!\n\n💡 Type /cancel if you want to stop the search.",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
        except Exception as e:
            # Fallback to text message if video sending fails
            logger.warning(f"Failed to send video: {e}, sending text message instead")
            processing_msg = bot.send_message(
                chat_id,
                "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!\n\n💡 Type /cancel if you want to stop the search.",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

        # ... rest of the function remains the same
        logger.info(f"Started restaurant search for user {user_id}: {search_query}")

        # Check for cancellation before starting the actual search
        if is_search_cancelled(user_id):
            logger.info(f"Search cancelled before processing for user {user_id}")
            return

        # Get orchestrator using singleton pattern
        orchestrator_instance = get_orchestrator()
        result = orchestrator_instance.process_query(search_query)

        # Check if cancelled after processing
        if is_search_cancelled(user_id):
            logger.info(f"Search was cancelled during processing for user {user_id}")
            try:
                if processing_msg:
                    bot.delete_message(chat_id, processing_msg.message_id)
            except:
                pass
            return

        # Get the pre-formatted text from orchestrator
        telegram_text = result.get('telegram_formatted_text', 'Sorry, no recommendations found.')

        # Delete processing message
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        # Final cancellation check before sending results
        if is_search_cancelled(user_id):
            logger.info(f"Search cancelled before sending results for user {user_id}")
            return

        # Send the results directly (already formatted by TelegramFormatter)
        bot.send_message(
            chat_id,
            telegram_text,
            parse_mode='HTML',
            disable_web_page_preview=True,
            reply_markup=remove_location_button()
        )

        logger.info(f"Successfully sent restaurant recommendations to user {user_id}")
        add_to_conversation(user_id, "Restaurant recommendations delivered!", is_user=False)

    except Exception as e:
        logger.error(f"Error in restaurant search process: {e}")
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        if not is_search_cancelled(user_id):
            bot.send_message(
                chat_id,
                "😔 Sorry, I encountered an error while searching for restaurants. Please try again with a different query!",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
    finally:
        cleanup_search(user_id)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all text messages with location-aware conversation management"""
    try:
        user_id = message.from_user.id
        user_message = message.text.strip()

        logger.info(f"Received message from user {user_id}: {user_message}")

        # Check if user has an active search
        if user_id in active_searches:
            bot.reply_to(
                message,
                "⏳ Oh, darling! I'm on the phone with my friend, we are discussing the list of restaurants for you. I'll be with you in a minute, or just type /cancel to stop the search.",
                parse_mode='HTML'
            )
            return

        # NEW: Check if user typed a location while waiting for location
        if user_id in users_awaiting_location:
            # User typed their location instead of using the button
            awaiting_data = users_awaiting_location[user_id]
            original_query = awaiting_data["query"]

            # Remove from waiting list and button
            del users_awaiting_location[user_id]

            # Process text-based location
            location_data = location_handler.extract_location_from_text(user_message)

            if location_data.confidence > 0.3:  # Reasonable confidence in location extraction
                # Confirm location and start search
                bot.reply_to(
                    message,
                    f"📍 <b>Perfect! I understand you're looking in: {location_data.description}</b>\n\n"
                    f"🔍 Now searching for: <i>{original_query}</i>\n\n"
                    "⏱ This might take a minute while I find the best places...",
                    parse_mode='HTML',
                    reply_markup=remove_location_button()
                )

                # Combine original query with location
                full_query = f"{original_query} in {location_data.description}"

                # Start location search
                threading.Thread(
                    target=perform_location_search,
                    args=(full_query, location_data, message.chat.id, user_id),
                    daemon=True
                ).start()
                return
            else:
                # Couldn't understand location, ask again
                bot.reply_to(
                    message,
                    "🤔 I'm having trouble understanding that location. Could you be more specific?\n\n"
                    "Examples: \"Downtown\", \"Near Central Park\", \"Chinatown\", \"Rua da Rosa\"\n\n"
                    "Or use the button below to send your exact coordinates:",
                    parse_mode='HTML',
                    reply_markup=create_location_button()
                )
                return

        # Add user message to conversation history
        add_to_conversation(user_id, user_message, is_user=True)

        # Send typing indicator
        bot.send_chat_action(message.chat.id, 'typing')

        # STEP 1: Check if user has shared location in recent conversation
        recent_location = get_recent_location_from_conversation(user_id)

        # STEP 2: Analyze message for location intent
        location_analysis = location_analyzer.analyze_message(user_message)
        search_type = location_analyzer.determine_search_type(location_analysis)

        logger.debug(f"Location analysis for user {user_id}: {search_type}")

        # STEP 3: Route based on analysis
        if search_type == "location_search":
            # User has specific location in message - proceed with location search
            handle_location_search_request(message, location_analysis)

        elif search_type == "request_location":
            # User wants location search but needs to specify location
            handle_location_request(message, location_analysis)

        elif search_type == "general_search":
            # Use existing general search pipeline
            handle_general_search(message, location_analysis)

        else:
            # Need clarification or off-topic
            handle_clarification_needed(message, location_analysis)

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.reply_to(
            message, 
            "I'm having trouble understanding right now. Could you try asking again about restaurants in a specific city?",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )

@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages using Whisper transcription"""
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id

        logger.info(f"🎤 Received voice message from user {user_id}")

        # Check if user has an active search
        if user_id in active_searches:
            bot.reply_to(
                message,
                "⏳ Oh, darling! I'm on the phone with my friend, we are discussing the list of restaurants for you. I'll be with you in a minute, or just type /cancel to stop the search.",
                parse_mode='HTML'
            )
            return

        # Check if user was waiting for location input
        if user_id in users_awaiting_location:
            awaiting_data = users_awaiting_location[user_id]
            original_query = awaiting_data["query"]

            # Remove from waiting list and button
            del users_awaiting_location[user_id]

            # Send immediate acknowledgment
            bot.reply_to(
                message,
                "🎤 <b>Got your location!</b>\n\n⏱ Processing your voice message and searching...",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

            # Process voice message in background
            threading.Thread(
                target=process_voice_for_location,
                args=(message, original_query, user_id, chat_id),
                daemon=True
            ).start()
            return

        # Send immediate acknowledgment for voice processing
        processing_voice_msg = bot.reply_to(
            message,
            "🎤 <b>Processing your voice message...</b>\n\n<i>Just a sec, processing...</i>",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )

        # Process voice message in background thread
        threading.Thread(
            target=process_voice_message,
            args=(message, user_id, chat_id, processing_voice_msg.message_id),
            daemon=True
        ).start()

    except Exception as e:
        logger.error(f"❌ Error handling voice message: {e}")
        bot.reply_to(
            message,
            "😔 Sorry, I had trouble processing your voice message. Could you try again or send a text message?",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )


def get_recent_location_from_conversation(user_id):
    """Check if user shared location in recent conversation"""
    if user_id not in user_conversations:
        return None

    # Look for recent location data (within last 10 messages)
    recent_messages = user_conversations[user_id][-10:]

    for msg in reversed(recent_messages):
        if msg.get("location_data"):
            # Check if location is recent (within last 30 minutes)
            if time.time() - msg["timestamp"] < 1800:  # 30 minutes
                return msg["location_data"]

    return None

def handle_location_search_request(message, location_analysis):
    """Handle requests with specific location mentioned"""
    user_id = message.from_user.id
    location_detected = location_analysis.get("location_detected")
    cuisine_preference = location_analysis.get("cuisine_preference", "restaurants")

    # Confirm and start search
    response_text = f"🔍 <b>Perfect! Searching for {cuisine_preference} in {location_detected}.</b>\n\n"
    response_text += "⏱ This might take a minute while I find the best places and verify them with reputable sources..."

    bot.reply_to(message, response_text, parse_mode='HTML', reply_markup=remove_location_button())
    add_to_conversation(user_id, response_text, is_user=False)

    # Create location data from text
    location_data = location_handler.extract_location_from_text(message.text)

    # Start location search
    threading.Thread(
        target=perform_location_search,
        args=(message.text, location_data, message.chat.id, user_id),
        daemon=True
    ).start()

def handle_location_request(message, location_analysis):
    """Handle requests that need location specification - NOW WITH BUTTON"""
    user_id = message.from_user.id
    cuisine_preference = location_analysis.get("cuisine_preference", "restaurants")

    # Determine query type for customized location request
    query_type = "general"
    if "wine" in cuisine_preference.lower():
        query_type = "wine"
    elif "coffee" in cuisine_preference.lower():
        query_type = "coffee"
    elif any(word in cuisine_preference.lower() for word in ["fine", "romantic", "fancy"]):
        query_type = "fine_dining"

    # Create the location request message
    if query_type == "wine":
        emoji = "🍷"
        context = "wine bars and natural wine spots"
    elif query_type == "coffee":
        emoji = "☕"
        context = "coffee shops and cafes"
    elif query_type == "fine_dining":
        emoji = "🍽️"
        context = "fine dining restaurants"
    else:
        emoji = "📍"
        context = "restaurants and bars"

    location_request_msg = (
        f"{emoji} <b>Perfect! I'd love to help you find great {context} near you.</b>\n\n"
        "To give you the best recommendations, I need to know where you are:\n\n"
        "🗺️ <b>Option 1:</b> Tell me your neighborhood, street, or nearby landmark\n"
        "📍 <b>Option 2:</b> Use the button below to send your exact coordinates\n\n"
        "<i>Examples: \"I'm in Chinatown\", \"Near Times Square\", \"On Rua da Rosa in Lisbon\"</i>\n\n"
        "💡 <b>Don't worry:</b> I only use your location to find nearby places. I don't store it."
    )

    bot.reply_to(
        message, 
        location_request_msg, 
        parse_mode='HTML',
        reply_markup=create_location_button()
    )
    add_to_conversation(user_id, location_request_msg, is_user=False)

    # Mark user as awaiting location WITH button tracking
    users_awaiting_location[user_id] = {
        "query": cuisine_preference,
        "timestamp": time.time(),
        "has_button": True
    }

def handle_general_search(message, location_analysis):
    """Handle general restaurant searches (existing pipeline)"""
    user_id = message.from_user.id

    # Use existing conversation AI logic
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", CONVERSATION_PROMPT),
        ("human", "Conversation history:\n{conversation_history}\n\nCurrent message: {user_message}")
    ])

    conversation_chain = conversation_prompt | conversation_ai
    response = conversation_chain.invoke({
        "conversation_history": format_conversation_history(user_id),
        "user_message": message.text
    })

    # Parse AI response (existing logic)
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        ai_decision = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI decision: {e}")
        ai_decision = {
            "action": "CLARIFY",
            "bot_response": "I'd love to help you find restaurants! Could you tell me what city you're interested in and what type of dining you're looking for?"
        }

    # Handle existing search logic
    action = ai_decision.get("action")
    bot_response = ai_decision.get("bot_response", "How can I help you find restaurants?")

    if action == "SEARCH":
        search_query = ai_decision.get("search_query")
        if search_query:
            add_to_conversation(user_id, bot_response, is_user=False)
            bot.reply_to(message, bot_response, parse_mode='HTML', reply_markup=remove_location_button())

            # Use existing search pipeline
            threading.Thread(
                target=perform_restaurant_search,
                args=(search_query, message.chat.id, user_id),
                daemon=True
            ).start()
        else:
            add_to_conversation(user_id, bot_response, is_user=False)
            bot.reply_to(message, bot_response, parse_mode='HTML', reply_markup=remove_location_button())
    else:
        add_to_conversation(user_id, bot_response, is_user=False)
        bot.reply_to(message, bot_response, parse_mode='HTML', reply_markup=remove_location_button())

def handle_clarification_needed(message, location_analysis):
    """Handle messages that need clarification"""
    user_id = message.from_user.id
    suggested_response = location_analysis.get("suggested_response", 
        "I specialize in restaurant recommendations! What type of dining are you looking for and in which city?")

    bot.reply_to(message, suggested_response, parse_mode='HTML', reply_markup=remove_location_button())
    add_to_conversation(user_id, suggested_response, is_user=False)

def perform_location_search(query, location_data, chat_id, user_id):
    """
    Perform location-based restaurant search using the location orchestrator
    """
    processing_msg = None

    try:
        # Create cancel event for this search
        cancel_event = create_cancel_event(user_id, chat_id)

        # Function to check cancellation
        def is_cancelled():
            return is_search_cancelled(user_id)

        # Send processing message WITH VIDEO
        try:
            with open("media/searching.mp4", "rb") as video:
                processing_msg = bot.send_video(
                    chat_id,
                    video,
                    caption="🔍 <b>Searching for nearby restaurants...</b>\n\n"
                            "<i>This may take a couple of minutes as I'll have to ask around to see what professional foodies have to say about these places'</i>\n\n"
                            "💡 Type /cancel to stop the search",
                    parse_mode='HTML',
                    reply_markup=remove_location_button()
                )
        except FileNotFoundError:
            # Fallback to text message if video file doesn't exist
            logger.warning("Video file not found, sending text message instead")
            processing_msg = bot.send_message(
                chat_id,
                "🔍 <b>Searching for nearby restaurants...</b>\n\n"
                "<i>This might take a couple of minutes as I'll check with my critic friends what they think about those places</i>\n\n"
                "💡 Type /cancel to stop the search",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
        except Exception as e:
            # Fallback to text message if video sending fails
            logger.warning(f"Failed to send video: {e}, sending text message instead")
            processing_msg = bot.send_message(
                chat_id,
                "🔍 <b>Searching for nearby restaurants...</b>\n\n"
                 "<i>This might take a couple of minutes as I'll check with my critic friends what they think about those places</i>\n\n"
                "💡 Type /cancel to stop the search",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

        logger.info(f"🎯 Started location search for user {user_id}: {query}")

        # Check for early cancellation
        if is_cancelled():
            logger.info(f"Location search cancelled before processing for user {user_id}")
            return

        # Initialize location orchestrator
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        # Run the location search pipeline
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                location_orchestrator.process_location_query(
                    query=query,
                    location_data=location_data,
                    cancel_check_fn=is_cancelled
                )
            )
        finally:
            loop.close()

        # Check if cancelled during processing
        if is_cancelled():
            logger.info(f"Location search was cancelled during processing for user {user_id}")
            try:
                if processing_msg:
                    bot.delete_message(chat_id, processing_msg.message_id)
            except:
                pass
            return

        # Delete processing message
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        # Check if search was successful
        if not result.get('success', False):
            error_text = result.get('telegram_formatted_text', 'Sorry, no results found.')
            bot.send_message(chat_id, error_text, parse_mode='HTML', reply_markup=remove_location_button())
            add_to_conversation(user_id, error_text, is_user=False)
            return

        # Get the formatted response
        response_text = result.get('telegram_formatted_text', '')
        results_count = result.get('results_count', 0)
        search_method = result.get('search_method', 'unknown')
        processing_time = result.get('processing_time', 0)

        # Final cancellation check before sending results
        if is_cancelled():
            logger.info(f"Location search cancelled before sending results for user {user_id}")
            return

        # Send the results
        bot.send_message(
            chat_id,
            response_text,
            parse_mode='HTML',
            disable_web_page_preview=True,
            reply_markup=remove_location_button()
        )

        # Send a follow-up with search stats (optional)
        stats_msg = (
            f"✅ <b>Search completed!</b>\n"
            f"📊 Found {results_count} results using {search_method}\n"
            f"⏱ Processed in {processing_time:.1f} seconds"
        )

        bot.send_message(chat_id, stats_msg, parse_mode='HTML', reply_markup=remove_location_button())

        logger.info(f"✅ Successfully sent location search results to user {user_id}")

        # Add to conversation history
        add_to_conversation(user_id, f"Location search completed - {results_count} results found!", is_user=False)

    except Exception as e:
        logger.error(f"❌ Error in location search process: {e}")

        # Delete processing message if it exists
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        # Only send error message if search wasn't cancelled
        if not is_search_cancelled(user_id):
            error_msg = (
                "😔 <b>Sorry, I encountered an error while searching for nearby restaurants.</b>\n\n"
                "This could be due to:\n"
                "• Location services temporarily unavailable\n"
                "• No restaurants found in your area\n"
                "• Network connectivity issues\n\n"
                "Please try again with a different location or search term!"
            )
            bot.send_message(chat_id, error_msg, parse_mode='HTML', reply_markup=remove_location_button())

    finally:
        # Always clean up the search tracking
        cleanup_search(user_id)


# In telegram_bot.py, replace the process_voice_message function with this updated version:

def process_voice_message(message, user_id, chat_id, processing_msg_id):
    """Process voice message in background thread"""
    try:
        # Step 1: Transcribe voice message
        transcribed_text = voice_handler.process_voice_message(bot, message.voice)

        if not transcribed_text:
            # Delete processing message and send error
            try:
                bot.delete_message(chat_id, processing_msg_id)
            except:
                pass

            bot.send_message(
                chat_id,
                "😔 <b>Sorry, I couldn't understand your voice message.</b>\n\n"
                "This could be due to:\n"
                "• Background noise\n"
                "• Very quiet recording\n"
                "• Connection issues\n\n"
                "Could you try recording again or send a text message?",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
            return

        logger.info(f"✅ Voice transcribed for user {user_id}: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}'")

        # Step 2: Add transcribed text to conversation history
        add_to_conversation(user_id, transcribed_text, is_user=True)

        # Step 3: Delete processing message (no confirmation message sent)
        try:
            bot.delete_message(chat_id, processing_msg_id)
        except:
            pass

        # Step 4: Process transcribed text directly - this will generate the AI response
        process_transcribed_text(transcribed_text, user_id, chat_id)

    except Exception as e:
        logger.error(f"❌ Error processing voice message: {e}")

        # Delete processing message
        try:
            bot.delete_message(chat_id, processing_msg_id)
        except:
            pass

        bot.send_message(
            chat_id,
            "😔 Sorry, I encountered an error processing your voice message. Could you try again or send a text message?",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )

def process_transcribed_text(transcribed_text, user_id, chat_id):
    """Process transcribed voice message text directly"""
    try:
        logger.debug(f"🎤 Processing transcribed text for user {user_id}: '{transcribed_text[:50]}...'")

        # STEP 1: Check if user has shared location in recent conversation
        recent_location = get_recent_location_from_conversation(user_id)

        # STEP 2: Analyze message for location intent
        location_analysis = location_analyzer.analyze_message(transcribed_text)
        search_type = location_analyzer.determine_search_type(location_analysis)

        logger.debug(f"🎤 Voice message analysis for user {user_id}: {search_type}")

        # STEP 3: Route based on analysis
        if search_type == "location_search":
            # User has specific location in voice message
            handle_voice_location_search(transcribed_text, location_analysis, user_id, chat_id)

        elif search_type == "request_location":
            # For voice messages, handle as location request
            handle_voice_location_request(transcribed_text, location_analysis, user_id, chat_id)

        elif search_type == "general_search":
            # Use existing general search pipeline
            handle_voice_general_search(transcribed_text, user_id, chat_id)

        else:
            # Need clarification
            handle_voice_clarification(transcribed_text, location_analysis, user_id, chat_id)

    except Exception as e:
        logger.error(f"❌ Error processing transcribed text: {e}")
        bot.send_message(
            chat_id,
            "😔 I had trouble understanding your voice request. Could you try again or be more specific about what restaurants you're looking for?",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )

def handle_voice_location_search(transcribed_text, location_analysis, user_id, chat_id):
    """Handle voice requests with specific location mentioned"""
    location_detected = location_analysis.get("location_detected")
    cuisine_preference = location_analysis.get("cuisine_preference", "restaurants")

    # Confirm and start search
    response_text = f"🔍 <b>Perfect! Searching for {cuisine_preference} in {location_detected}.</b>\n\n"
    response_text += "⏱ This might take a minute while I find the best places and verify them with reputable sources..."

    bot.send_message(chat_id, response_text, parse_mode='HTML', reply_markup=remove_location_button())
    add_to_conversation(user_id, response_text, is_user=False)

    # Create location data from text
    location_data = location_handler.extract_location_from_text(transcribed_text)

    # Start location search
    threading.Thread(
        target=perform_location_search,
        args=(transcribed_text, location_data, chat_id, user_id),
        daemon=True
    ).start()

def handle_voice_location_request(transcribed_text, location_analysis, user_id, chat_id):
    """Handle voice requests that need location specification"""
    cuisine_preference = location_analysis.get("cuisine_preference", "restaurants")

    # Determine query type for customized location request
    query_type = "general"
    if "wine" in cuisine_preference.lower():
        query_type = "wine"
    elif "coffee" in cuisine_preference.lower():
        query_type = "coffee"
    elif any(word in cuisine_preference.lower() for word in ["fine", "romantic", "fancy"]):
        query_type = "fine_dining"

    # Create the location request message
    if query_type == "wine":
        emoji = "🍷"
        context = "wine bars and natural wine spots"
    elif query_type == "coffee":
        emoji = "☕"
        context = "coffee shops and cafes"
    elif query_type == "fine_dining":
        emoji = "🍽️"
        context = "fine dining restaurants"
    else:
        emoji = "📍"
        context = "restaurants and bars"

    location_request_msg = (
        f"{emoji} <b>Perfect! I'd love to help you find great {context} near you.</b>\n\n"
        "To give you the best recommendations, I need to know where you are:\n\n"
        "🗺️ <b>Option 1:</b> Tell me your neighborhood, street, or nearby landmark\n"
        "📍 <b>Option 2:</b> Use the button below to send your exact coordinates\n\n"
        "<i>Examples: \"I'm in Chinatown\", \"Near Times Square\", \"On Rua da Rosa in Lisbon\"</i>\n\n"
        "💡 <b>Don't worry:</b> I only use your location to find nearby places. I don't store it."
    )

    bot.send_message(
        chat_id, 
        location_request_msg, 
        parse_mode='HTML',
        reply_markup=create_location_button()
    )
    add_to_conversation(user_id, location_request_msg, is_user=False)

    # Mark user as awaiting location WITH button tracking
    users_awaiting_location[user_id] = {
        "query": cuisine_preference,
        "timestamp": time.time(),
        "has_button": True
    }

def handle_voice_general_search(transcribed_text, user_id, chat_id):
    """Handle general restaurant searches from voice (existing pipeline)"""
    # Use existing conversation AI logic
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", CONVERSATION_PROMPT),
        ("human", "Conversation history:\n{conversation_history}\n\nCurrent message: {user_message}")
    ])

    conversation_chain = conversation_prompt | conversation_ai
    response = conversation_chain.invoke({
        "conversation_history": format_conversation_history(user_id),
        "user_message": transcribed_text
    })

    # Parse AI response (existing logic)
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        ai_decision = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI decision: {e}")
        ai_decision = {
            "action": "CLARIFY",
            "bot_response": "I'd love to help you find restaurants! Could you tell me what city you're interested in and what type of dining you're looking for?"
        }

    # Handle existing search logic
    action = ai_decision.get("action")
    bot_response = ai_decision.get("bot_response", "How can I help you find restaurants?")

    if action == "SEARCH":
        search_query = ai_decision.get("search_query")
        if search_query:
            add_to_conversation(user_id, bot_response, is_user=False)
            bot.send_message(chat_id, bot_response, parse_mode='HTML', reply_markup=remove_location_button())

            # Use existing search pipeline
            threading.Thread(
                target=perform_restaurant_search,
                args=(search_query, chat_id, user_id),
                daemon=True
            ).start()
        else:
            add_to_conversation(user_id, bot_response, is_user=False)
            bot.send_message(chat_id, bot_response, parse_mode='HTML', reply_markup=remove_location_button())
    else:
        add_to_conversation(user_id, bot_response, is_user=False)
        bot.send_message(chat_id, bot_response, parse_mode='HTML', reply_markup=remove_location_button())

def handle_voice_clarification(transcribed_text, location_analysis, user_id, chat_id):
    """Handle voice messages that need clarification"""
    suggested_response = location_analysis.get("suggested_response", 
        "I specialize in restaurant recommendations! What type of dining are you looking for and in which city?")

    bot.send_message(chat_id, suggested_response, parse_mode='HTML', reply_markup=remove_location_button())
    add_to_conversation(user_id, suggested_response, is_user=False)

# REMOVE the process_transcribed_message function entirely - it's replaced by process_transcribed_text

def process_voice_for_location(message, original_query, user_id, chat_id):
    """Process voice message when user was providing location"""
    try:
        # Transcribe the voice message
        transcribed_text = voice_handler.process_voice_message(bot, message.voice)

        if not transcribed_text:
            bot.send_message(
                chat_id,
                "😔 I couldn't understand your voice message for the location. Could you try again or type your location?",
                parse_mode='HTML',
                reply_markup=create_location_button()
            )
            # Re-add to waiting list
            users_awaiting_location[user_id] = {
                "query": original_query,
                "timestamp": time.time(),
                "has_button": True
            }
            return

        logger.info(f"🎤📍 Voice location transcribed for user {user_id}: '{transcribed_text}'")

        # Process location from transcribed text
        location_data = location_handler.extract_location_from_text(transcribed_text)

        if location_data.confidence > 0.3:
            # Good location understanding
            bot.send_message(
                chat_id,
                f"📍 <b>Perfect! I understand you're looking in: {location_data.description}</b>\n\n"
                f"🔍 Now searching for: <i>{original_query}</i>\n\n"
                "⏱ This might take a minute while I find the best places...",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

            # Combine query with location
            full_query = f"{original_query} in {location_data.description}"

            # Start location search
            threading.Thread(
                target=perform_location_search,
                args=(full_query, location_data, chat_id, user_id),
                daemon=True
            ).start()

        else:
            # Couldn't understand location
            bot.send_message(
                chat_id,
                "🤔 I'm having trouble understanding that location from your voice message. Could you try again?\n\n"
                "Examples: \"Downtown\", \"Near Central Park\", \"Chinatown\", \"Rua da Rosa\"\n\n"
                "Or use the button below to send your exact coordinates:",
                parse_mode='HTML',
                reply_markup=create_location_button()
            )

            # Re-add to waiting list
            users_awaiting_location[user_id] = {
                "query": original_query,
                "timestamp": time.time(),
                "has_button": True
            }

    except Exception as e:
        logger.error(f"❌ Error processing voice location: {e}")
        bot.send_message(
            chat_id,
            "😔 Sorry, I had trouble processing your voice message. Could you try typing your location or using the button?",
            parse_mode='HTML',
            reply_markup=create_location_button()
        )

def main():
    """Main function to start the bot with location and voice support"""
    global location_analyzer, voice_handler

    logger.info("Starting Restaurant Babe Telegram Bot with Location & Voice Support...")

    # Verify bot token works
    try:
        bot_info = bot.get_me()
        logger.info(f"Bot started successfully: @{bot_info.username}")
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return

    # Initialize location analyzer
    try:
        location_analyzer = LocationAnalyzer(config)
        logger.info("✅ Location Analyzer initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Location Analyzer: {e}")
        return

    # Initialize voice handler
    try:
        voice_handler = VoiceMessageHandler()
        logger.info("✅ Voice Message Handler initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Voice Handler: {e}")
        return

    # Verify orchestrator is available
    try:
        orchestrator_instance = get_orchestrator()
        logger.info("✅ Orchestrator singleton confirmed available")
        logger.info("🎯 Admin commands available: /test_scrape, /test_search")
        logger.info("🛑 Cancel command available: /cancel")
        logger.info("📍 Location support: GPS pins + text descriptions + location button")
        logger.info("🎤 Voice support: Whisper transcription + live recognition flow")
    except RuntimeError as e:
        logger.error(f"❌ Orchestrator not initialized: {e}")
        logger.error("Make sure main.py calls setup_orchestrator() before starting the bot")
        return

    # Start polling with error handling
    while True:
        try:
            logger.info("Starting bot polling with location button and voice support...")
            bot.infinity_polling(
                timeout=10, 
                long_polling_timeout=5,
                restart_on_change=False,
                none_stop=True
            )
        except telebot.apihelper.ApiTelegramException as e:
            if "409" in str(e):
                logger.error("Another bot instance is running! Stopping this instance.")
                break
            else:
                logger.error(f"Telegram API error: {e}")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error in bot polling: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()