# telegram_bot.py - Updated with location button functionality
import telebot
import logging
import time
import threading
import json
import asyncio
from threading import Event
from typing import Dict, List, Any, Optional, Tuple
from telebot.types import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from utils.orchestrator_manager import get_orchestrator

from utils.telegram_location_handler import TelegramLocationHandler, LocationData
from agents.location_analyzer import LocationAnalyzer

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

# Track users waiting for location input
users_awaiting_location = {}  # user_id -> {"query": str, "timestamp": float}

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

def create_location_keyboard():
    """Create a keyboard with location button"""
    keyboard = ReplyKeyboardMarkup(
        row_width=1,
        resize_keyboard=True,
        one_time_keyboard=True  # This makes the keyboard disappear after use
    )

    # Create location button
    location_button = KeyboardButton(
        text="📍 Share My Location",
        request_location=True
    )

    keyboard.add(location_button)

    # Add option to type location instead
    keyboard.add(KeyboardButton("✏️ I'll type my location"))

    return keyboard

def remove_keyboard():
    """Remove the keyboard"""
    return ReplyKeyboardRemove()

# Welcome message (unchanged)
WELCOME_MESSAGE = (
    "🍸 Hello! I'm an AI assistant Restaurant Babe, and I know all about the most delicious and trendy restaurants, cafes, bakeries, bars, and coffee shops around the world.\n\n"
    "Tell me what you are looking for. For example:\n"
    "<i>What new restaurants have recently opened in Lisbon?</i>\n"
    "<i>Local residents' favorite cevicherias in Lima</i>\n"
    "<i>Where can I find the most delicious plov in Tashkent?</i>\n"
    "<i>Recommend places with brunch and specialty coffee in Barcelona.</i>\n"
    "<i>Best cocktail bars in Paris's Marais district</i>\n\n"
    "I will check with my restaurant critic friends and provide the best recommendations. This might take a couple of minutes because I search very carefully and thoroughly verify the results.\n\n"
    "💡 <b>Pro tip:</b> I can find restaurants near you! Just ask something like \"restaurants near me\" or \"good coffee nearby\" and I'll show you a location button."
)

CONVERSATION_PROMPT = """You are an AI assistant specializing in restaurant recommendations. 

Your primary task is to:
1. Understand what the user is looking for (cuisine type, dining style, location, etc.)
2. Decide if you can provide a restaurant search or need more information
3. Format your response as JSON with specific structure

Response format:
{{
    "action": "SEARCH|CLARIFY|LOCATION_NEEDED",
    "search_query": "query for restaurant search (only if action is SEARCH)",
    "bot_response": "message to send to user",
    "location_required": true/false
}}

Actions:
- SEARCH: When you have enough info to search for restaurants
- CLARIFY: When you need more details about cuisine, style, or preferences  
- LOCATION_NEEDED: When user wants nearby restaurants but no location provided

Guidelines:
- Be conversational and helpful
- For location-based requests without location, set action to "LOCATION_NEEDED"
- Always try to extract useful search query even from partial information
- Focus on restaurant-related requests only

Conversation history is provided for context."""

def add_to_conversation(user_id, message, is_user=True):
    """Add message to conversation history"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    role = "user" if is_user else "assistant"
    user_conversations[user_id].append({
        "role": role,
        "message": message,
        "timestamp": time.time()
    })

    # Keep only last 20 messages
    if len(user_conversations[user_id]) > 20:
        user_conversations[user_id] = user_conversations[user_id][-20:]

def format_conversation_history(user_id):
    """Format conversation history for AI"""
    if user_id not in user_conversations:
        return ""

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

    # Clear waiting location state
    if user_id in users_awaiting_location:
        del users_awaiting_location[user_id]

    bot.reply_to(message, WELCOME_MESSAGE, parse_mode='HTML', reply_markup=remove_keyboard())

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
            reply_markup=remove_keyboard()
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
        reply_markup=remove_keyboard()
    )

    # Clean up
    cleanup_search(user_id)

    # Clear waiting location state
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
                reply_markup=remove_keyboard()
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

            # Confirm location received and remove keyboard
            bot.reply_to(
                message,
                f"📍 <b>Perfect! I received your location.</b>\n\n"
                f"🔍 Now searching for: <i>{original_query}</i>\n\n"
                "⏱ This might take a minute while I find the best nearby places...",
                parse_mode='HTML',
                reply_markup=remove_keyboard()  # Remove the location keyboard
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
                reply_markup=remove_keyboard()  # Remove the location keyboard
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
            reply_markup=remove_keyboard()
        )

@bot.message_handler(func=lambda message: message.text == "✏️ I'll type my location")
def handle_type_location_choice(message):
    """Handle when user chooses to type their location"""
    user_id = message.from_user.id

    # Check if user was waiting for location
    if user_id in users_awaiting_location:
        awaiting_data = users_awaiting_location[user_id]
        original_query = awaiting_data["query"]

        bot.reply_to(
            message,
            f"✏️ <b>No problem!</b> Just tell me where you are.\n\n"
            f"🔍 Looking for: <i>{original_query}</i>\n\n"
            "You can say something like:\n"
            "• \"I'm in Chinatown\"\n"
            "• \"Near Times Square\"\n"
            "• \"On Rua da Rosa in Lisbon\"\n"
            "• \"123 Main Street, Downtown\"",
            parse_mode='HTML',
            reply_markup=remove_keyboard()  # Remove the location keyboard
        )
    else:
        bot.reply_to(
            message,
            "✏️ Please tell me where you are and what type of restaurants you're looking for!\n\n"
            "<i>Example: \"Italian restaurants in Manhattan\" or \"I'm in Brooklyn, looking for good brunch\"</i>",
            parse_mode='HTML',
            reply_markup=remove_keyboard()
        )

@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    """Handle all text messages"""
    try:
        user_id = message.from_user.id
        user_message = message.text.strip()

        # Skip if message is empty
        if not user_message:
            return

        # Check if search is already in progress
        if user_id in active_searches:
            bot.reply_to(
                message,
                "🔍 <b>I'm still working on your previous request!</b>\n\n"
                "Please wait for the results or type /cancel to stop the search.",
                parse_mode='HTML'
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
            reply_markup=remove_keyboard()
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

    bot.reply_to(message, response_text, parse_mode='HTML', reply_markup=remove_keyboard())
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
    """Handle requests that need location specification - SHOW LOCATION BUTTON HERE"""
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

    # Create customized message with location button
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
        "👇 <b>Choose an option below:</b>"
    )

    # Send message with location keyboard
    bot.reply_to(
        message, 
        location_request_msg, 
        parse_mode='HTML',
        reply_markup=create_location_keyboard()  # Show the location button!
    )

    add_to_conversation(user_id, location_request_msg, is_user=False)

    # Mark user as awaiting location
    users_awaiting_location[user_id] = {
        "query": cuisine_preference,
        "timestamp": time.time()
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

    action = ai_decision.get("action", "CLARIFY")
    bot_response = ai_decision.get("bot_response", "Could you tell me more about what you're looking for?")

    if action == "SEARCH":
        search_query = ai_decision.get("search_query", message.text)

        # Send confirmation message
        confirmation_msg = f"🔍 <b>Great question!</b>\n\n{bot_response}\n\n⏱ Let me search for the best recommendations..."
        bot.reply_to(message, confirmation_msg, parse_mode='HTML', reply_markup=remove_keyboard())
        add_to_conversation(user_id, confirmation_msg, is_user=False)

        # Start search in background
        threading.Thread(
            target=perform_general_search,
            args=(search_query, message.chat.id, user_id),
            daemon=True
        ).start()

    elif action == "LOCATION_NEEDED":
        # Show location button for nearby searches
        handle_location_request(message, location_analysis)

    else:
        # CLARIFY - just respond normally
        bot.reply_to(message, bot_response, parse_mode='HTML', reply_markup=remove_keyboard())
        add_to_conversation(user_id, bot_response, is_user=False)

def handle_clarification_needed(message, location_analysis):
    """Handle unclear requests"""
    user_id = message.from_user.id

    suggested_response = (
        "🤔 I'd love to help you find amazing restaurants! Could you be a bit more specific?\n\n"
        "<b>For example:</b>\n"
        "• \"Best sushi in Tokyo\"\n"
        "• \"Romantic restaurants near me\"\n"
        "• \"Good coffee shops in Barcelona\"\n"
        "• \"Family-friendly pizza places in Rome\"\n\n"
        "What type of dining are you looking for and in which city?"
    )

    bot.reply_to(message, suggested_response, parse_mode='HTML', reply_markup=remove_keyboard())
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

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 <b>Searching for nearby restaurants...</b>\n\n"
            "📍 Getting your precise location\n"
            "🗄️ Checking our restaurant database\n" 
            "📰 Searching with reputable food sources\n\n"
            "<i>This takes 1-2 minutes for thorough verification</i>\n\n"
            "💡 Type /cancel to stop the search",
            parse_mode='HTML'
        )

        logger.info(f"🎯 Started location search for user {user_id}: {query}")

        # Check for early cancellation
        if is_cancelled():
            logger.info(f"Location search cancelled before processing for user {user_id}")
            return

        # Initialize location orchestrator
        from agents.location_orchestrator import LocationOrchestrator
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
            bot.send_message(chat_id, error_text, parse_mode='HTML')
            add_to_conversation(user_id, error_text, is_user=False)
            return

        # Send successful results
        results_text = result.get('telegram_formatted_text', 'Found some great places for you!')

        # Split long messages if needed
        if len(results_text) > 4000:
            parts = [results_text[i:i+4000] for i in range(0, len(results_text), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    bot.send_message(chat_id, part, parse_mode='HTML')
                else:
                    bot.send_message(chat_id, f"<b>Continued...</b>\n\n{part}", parse_mode='HTML')
        else:
            bot.send_message(chat_id, results_text, parse_mode='HTML')

        # Add to conversation history
        add_to_conversation(user_id, "Location search completed successfully", is_user=False)

        logger.info(f"✅ Location search completed successfully for user {user_id}")

    except Exception as e:
        logger.error(f"Error in location search for user {user_id}: {e}")

        # Clean up processing message
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        # Send error message
        error_msg = (
            "😔 <b>Sorry, I encountered an issue while searching.</b>\n\n"
            "Could you try asking again with a different query?"
        )
        bot.send_message(chat_id, error_msg, parse_mode='HTML')

    finally:
        # Always clean up the search tracking
        cleanup_search(user_id)

def perform_general_search(query, chat_id, user_id):
    """Perform general restaurant search (existing functionality)"""
    processing_msg = None

    try:
        # Create cancel event for this search
        cancel_event = create_cancel_event(user_id, chat_id)

        # Function to check cancellation
        def is_cancelled():
            return is_search_cancelled(user_id)

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 <b>Searching for restaurant recommendations...</b>\n\n"
            "📰 Consulting reputable food sources\n"
            "🧠 Analyzing expert reviews\n"
            "✅ Verifying recommendations\n\n"
            "<i>This takes 1-2 minutes for thorough research</i>\n\n"
            "💡 Type /cancel to stop the search",
            parse_mode='HTML'
        )

        logger.info(f"🎯 Started general search for user {user_id}: {query}")

        # Check for early cancellation
        if is_cancelled():
            logger.info(f"General search cancelled before processing for user {user_id}")
            return

        # Get orchestrator and run search
        orchestrator = get_orchestrator()

        # Run the search pipeline
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                orchestrator.process_restaurant_query(
                    query=query,
                    cancel_check_fn=is_cancelled
                )
            )
        finally:
            loop.close()

        # Check if cancelled during processing
        if is_cancelled():
            logger.info(f"General search was cancelled during processing for user {user_id}")
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

        # Send results
        if result and result.get('success', False):
            results_text = result.get('telegram_formatted_text', 'Found some great recommendations!')

            # Split long messages if needed
            if len(results_text) > 4000:
                parts = [results_text[i:i+4000] for i in range(0, len(results_text), 4000)]
                for i, part in enumerate(parts):
                    if i == 0:
                        bot.send_message(chat_id, part, parse_mode='HTML')
                    else:
                        bot.send_message(chat_id, f"<b>Continued...</b>\n\n{part}", parse_mode='HTML')
            else:
                bot.send_message(chat_id, results_text, parse_mode='HTML')

            add_to_conversation(user_id, "General search completed successfully", is_user=False)
            logger.info(f"✅ General search completed successfully for user {user_id}")
        else:
            error_text = result.get('telegram_formatted_text', 'Sorry, no results found.') if result else 'Sorry, no results found.'
            bot.send_message(chat_id, error_text, parse_mode='HTML')
            add_to_conversation(user_id, error_text, is_user=False)

    except Exception as e:
        logger.error(f"Error in general search for user {user_id}: {e}")

        # Clean up processing message
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        # Send error message
        error_msg = (
            "😔 <b>Sorry, I encountered an issue while searching.</b>\n\n"
            "Could you try asking again with a different query?"
        )
        bot.send_message(chat_id, error_msg, parse_mode='HTML')

    finally:
        # Always clean up the search tracking
        cleanup_search(user_id)

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

    # Send confirmation and run test
    bot.reply_to(
        message,
        f"🔍 <b>Starting search filtering test...</b>\n\n"
        f"📝 Query: <code>{restaurant_query}</code>\n\n"
        "⏱ Please wait...",
        parse_mode='HTML'
    )

    def run_search_test():
        try:
            from search_test import SearchTest
            search_tester = SearchTest(config, get_orchestrator())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results_path = loop.run_until_complete(
                search_tester.test_search_filtering(restaurant_query, bot)
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

# UPDATED MAIN FUNCTION (add location_analyzer initialization)
def main():
    """Main function to start the bot with location support"""
    global location_analyzer

    logger.info("Starting Restaurant Babe Telegram Bot with Location Button Support...")

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

    # Verify orchestrator is available
    try:
        orchestrator_instance = get_orchestrator()
        logger.info("✅ Orchestrator singleton confirmed available")
        logger.info("🎯 Admin commands available: /test_scrape, /test_search")
        logger.info("🛑 Cancel command available: /cancel")
        logger.info("📍 Location support: GPS pins + location button + text descriptions")
        logger.info("⌨️ Location button: Shows when user asks for nearby restaurants")
    except RuntimeError as e:
        logger.error(f"❌ Orchestrator not initialized: {e}")
        logger.error("Make sure main.py calls setup_orchestrator() before starting the bot")
        return

    # Start polling with error handling
    while True:
        try:
            logger.info("Starting bot polling with location button support...")
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