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

from location.telegram_location_handler import TelegramLocationHandler, LocationData
from location.location_analyzer import LocationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize bot with type safety
if not config.TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in config")
bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize AI conversation handler with type safety
if not config.OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL not found in config")
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

# NEW: Track pending location searches with user choice
pending_location_choices = {}  # user_id -> {"query": str, "location_data": LocationData, "database_results": List, "timestamp": float}

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

def create_choice_buttons():
    """Create inline buttons for user choice after database results"""
    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(
        types.InlineKeyboardButton("✅ These look good", callback_data="accept_results"),
        types.InlineKeyboardButton("🔍 Find more options", callback_data="search_more")
    )
    return keyboard

def remove_location_button():
    """Remove the location button (return to normal keyboard)"""
    return types.ReplyKeyboardRemove()

@bot.callback_query_handler(func=lambda call: call.data in ["accept_results", "search_more"])
def handle_user_choice(call):
    """Handle user choice after database results are shown"""
    try:
        user_id = call.from_user.id
        chat_id = call.message.chat.id
        choice = call.data

        # Check if user has pending choice
        if user_id not in pending_location_choices:
            bot.answer_callback_query(call.id, "This search has expired. Please start a new search.")
            return

        pending_data = pending_location_choices[user_id]
        query = pending_data["query"]
        location_data = pending_data["location_data"]

        # Answer the callback to remove loading state
        bot.answer_callback_query(call.id)

        # Edit the message to remove buttons
        try:
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
        except:
            pass

        if choice == "accept_results":
            # User accepts database results
            bot.send_message(
                chat_id,
                "✅ <b>Great choice!</b> Hope you enjoy these recommendations!\n\n"
                "💡 Feel free to ask for more recommendations anytime.",
                parse_mode='HTML'
            )
            logger.info(f"User {user_id} accepted database results")

        elif choice == "search_more":
            # User wants more options - start Google Maps search
            bot.send_message(
                chat_id,
                "🔍 <b>Searching for more options...</b>\n\n"
                "⏱ I'll check Google Maps for additional restaurants nearby.",
                parse_mode='HTML'
            )

            # Start Google Maps search
            threading.Thread(
                target=perform_google_maps_only_search,
                args=(query, location_data, chat_id, user_id),
                daemon=True
            ).start()

        # Clean up pending choice
        del pending_location_choices[user_id]

    except Exception as e:
        logger.error(f"Error handling user choice: {e}")
        bot.answer_callback_query(call.id, "Sorry, there was an error processing your choice.")


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
    user_id = message.from_user.id

    # Initialize conversation for new users
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    bot.reply_to(message, WELCOME_MESSAGE, parse_mode='HTML', reply_markup=remove_location_button())
    add_to_conversation(user_id, WELCOME_MESSAGE, is_user=False)

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

    # Clean up immediately
    cleanup_search(user_id)

    # Clean up any pending location choices or waiting states
    if user_id in pending_location_choices:
        del pending_location_choices[user_id]
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
        # ADD NULL CHECK for location_analyzer
        if location_analyzer is None:
            logger.warning("Location analyzer not initialized, using general search")
            handle_general_search(message, {})
            return

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
    SIMPLIFIED: Perform location-based restaurant search with user choice logic
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
                            "<i>This may take a couple of minutes as I'll check what professional foodies have to say about these places</i>\n\n"
                            "💡 Type /cancel to stop the search",
                    parse_mode='HTML',
                    reply_markup=remove_location_button()
                )
        except FileNotFoundError:
            # Fallback to text message if video file doesn't exist
            processing_msg = bot.send_message(
                chat_id,
                "🔍 <b>Searching for nearby restaurants...</b>\n\n"
                "<i>This may take a couple of minutes as I'll check what professional foodies have to say about these places</i>\n\n"
                "💡 Type /cancel to stop the search",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

        # Import location orchestrator
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        # Run location search
        logger.info(f"Starting location search for user {user_id}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            location_orchestrator.process_location_query(query, location_data, is_cancelled)
        )

        loop.close()

        # Delete processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except:
                pass

        # Check if search was cancelled
        if is_cancelled():
            logger.info(f"Location search cancelled for user {user_id}")
            return

        # Handle errors
        if not result.get('success', False):
            error_message = result.get('error', 'Unknown error occurred')

            if result.get('cancelled'):
                return

            bot.send_message(
                chat_id,
                f"😔 {error_message}\n\nTry a different search or let me know if you need help!",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
            return

        # SIMPLIFIED RESULT HANDLING
        source = result.get('source', 'unknown')

        if source == "database_with_choice":
            # Database results that need user choice
            handle_database_results_with_choice(result, query, location_data, chat_id, user_id)

        else:
            # Any other results - orchestrator should provide formatted message
            formatted_message = result.get('formatted_message', 'Found restaurants!')

            bot.send_message(
                chat_id,
                formatted_message,
                parse_mode='HTML',
                reply_markup=remove_location_button(),
                disable_web_page_preview=True
            )

            logger.info(f"✅ Location search completed for user {user_id}")
            add_to_conversation(user_id, f"Found restaurants for: {query}", is_user=False)

    except Exception as e:
        logger.error(f"Error in location search process: {e}")
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

def handle_database_results_with_choice(result, query, location_data, chat_id, user_id):
    """
    NEW: Handle database results and offer user choice
    FIXED: Handle the correct data structure from orchestrator
    """
    try:
        # The orchestrator returns results as a list, not a dict
        restaurants = result.get('results', [])
        restaurant_count = result.get('restaurant_count', len(restaurants))

        # Use the location formatter to create the message
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        # Format the results properly
        formatted_results = location_orchestrator.formatter.format_database_results(
            restaurants=restaurants,
            query=query,
            location_description=location_data.description or "your location",
            offer_more_search=True
        )

        message_text = formatted_results.get('message', 'Found some restaurants from my personal notes!')

        # Store pending choice data
        pending_location_choices[user_id] = {
            "query": query,
            "location_data": location_data,
            "database_results": restaurants,
            "timestamp": time.time()
        }

        # Send results with choice buttons
        bot.send_message(
            chat_id,
            message_text + "\n\n" + 
            "👆 <b>What would you like to do?</b>",
            parse_mode='HTML',
            reply_markup=create_choice_buttons(),
            disable_web_page_preview=True
        )

        logger.info(f"✅ Database results sent for user {user_id} with choice buttons")
        add_to_conversation(user_id, f"Found {restaurant_count} restaurants from database", is_user=False)

    except Exception as e:
        logger.error(f"❌ Error handling database results with choice: {e}")

        # Fallback: send results without choice buttons
        try:
            restaurants = result.get('results', [])
            if restaurants:
                # Simple fallback formatting
                restaurant_names = [r.get('name', 'Unknown') for r in restaurants[:5]]
                fallback_message = f"📝 Found {len(restaurants)} restaurants:\n\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(restaurant_names)])
            else:
                fallback_message = "Found restaurants from my personal notes!"

            bot.send_message(
                chat_id,
                fallback_message,
                parse_mode='HTML',
                reply_markup=remove_location_button(),
                disable_web_page_preview=True
            )

            logger.info(f"✅ Fallback results sent for user {user_id}")
            add_to_conversation(user_id, "Found restaurants from database", is_user=False)

        except Exception as fallback_error:
            logger.error(f"❌ Error in fallback handling: {fallback_error}")
            bot.send_message(
                chat_id,
                "😔 Found restaurants but had trouble displaying them. Please try your search again.",
                parse_mode='HTML'
            )
        
def perform_google_maps_only_search(query, location_data, chat_id, user_id):
    """
    NEW: Perform Google Maps search only (after user chooses "more options")
    CLEAN VERSION - no geocoding logic, let location orchestrator handle it
    """
    try:
        # Create cancel event for this search
        cancel_event = create_cancel_event(user_id, chat_id)

        # Function to check cancellation
        def is_cancelled():
            return is_search_cancelled(user_id)

        # Import location orchestrator
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        # Let the orchestrator handle coordinate resolution
        logger.info(f"Starting Google Maps only search for user {user_id}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Call the orchestrator's Google Maps search flow directly
        result = loop.run_until_complete(
            location_orchestrator.process_location_query(query, location_data, is_cancelled)
        )

        loop.close()

        # Check if search was cancelled
        if is_cancelled():
            logger.info(f"Google Maps search cancelled for user {user_id}")
            return

        # Simple result handling - orchestrator should return formatted message
        if result.get('success', False):
            formatted_message = result.get('formatted_message', f"Found {len(result.get('results', []))} restaurants!")

            bot.send_message(
                chat_id,
                formatted_message,
                parse_mode='HTML',
                reply_markup=remove_location_button(),
                disable_web_page_preview=True
            )

            logger.info(f"✅ Google Maps results sent for user {user_id}")
            add_to_conversation(user_id, "Found additional restaurants via Google Maps", is_user=False)
        else:
            bot.send_message(
                chat_id,
                f"😔 {result.get('error', 'No additional restaurants found')}\n\n"
                "The recommendations from my notes might be your best options in this area!",
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"Error in Google Maps only search: {e}")
        bot.send_message(
            chat_id,
            "😔 Sorry, I encountered an error searching for additional options.",
            parse_mode='HTML'
        )
    finally:
        cleanup_search(user_id)


def handle_location_results(result, chat_id, user_id):
    """Handle ANY location results - already formatted by orchestrator"""
    message = result.get('formatted_message', 'Results found!')

    bot.send_message(
        chat_id,
        message,
        parse_mode='HTML',
        reply_markup=remove_location_button(),
        disable_web_page_preview=True
    )

def process_voice_message(message, user_id, chat_id, processing_msg_id):
    """Process voice message in background thread"""
    try:
        # Add this check
        if voice_handler is None:
            logger.error("Voice handler not initialized")
            bot.send_message(
                chat_id,
                "😔 Voice processing is not available right now. Please send a text message.",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )
            return
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
        # ADD NULL CHECK for location_analyzer
        if location_analyzer is None:
            logger.warning("Location analyzer not initialized, using general search")
            handle_voice_general_search(transcribed_text, user_id, chat_id)
            return

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