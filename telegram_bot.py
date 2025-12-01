# telegram_bot.py - AI-FIRST ROUTING
"""
Telegram Bot with AI-First Message Routing

KEY ARCHITECTURE:
- ALL messages go to AI Chat Layer first - NO EXCEPTIONS
- AI decides whether message is a location answer, new query, or conversation
- Location button state managed by AI Chat Layer (single source of truth)
- Telegram bot is a pure I/O layer - no routing logic

Features:
- AI Chat Layer for intelligent conversation flow
- AI-generated search messages with videos
- Location button for "near me" queries
- Voice message support
- Memory and conversation context
"""

import telebot
import asyncio
import logging
import time
import os
import re
import tempfile
from typing import Optional, List, Tuple
from threading import Event
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import the enhanced unified agent
from langgraph_orchestrator import create_unified_restaurant_agent
from utils.voice_handler import VoiceMessageHandler
from utils.database import initialize_database
from location.geocoding import initialize_geocoding_service
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize bot
if not config.TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is required in config")
bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize database FIRST
initialize_database(config)
initialize_geocoding_service(config)

# Initialize enhanced unified agent with AI Chat Layer
unified_agent = create_unified_restaurant_agent(config)

# Initialize AI for search message generation
ai_message_generator = None

# Initialize voice handler
voice_handler = VoiceMessageHandler() if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY else None

# Cancellation tracking
active_searches = {}  # user_id -> Event

# Welcome message
WELCOME_MESSAGE = (
    "🍸 Hello! I'm Restaurant Babe. I know all about the most delicious restaurants worldwide.\n\n"
    "Tell me what you're looking for, like <i>best ramen in Tokyo</i>, or share your location for nearby recommendations.\n\n"
    "I remember our conversations and your preferences, so feel free to just say things like \"more options\" or \"different cuisine\"!\n\n"
    "💡 <b>Tip:</b> Type /cancel to stop a search.\n\n"
    "What are you hungry for?"
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fix_telegram_html(text: str) -> str:
    """Fix HTML formatting for Telegram - preserve HTML tags for proper formatting"""
    if not text:
        return text

    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', text)

    open_tags = []
    result = []
    tag_pattern = r'<(/?)(\w+)(?:[^>]*)>'
    last_pos = 0

    for match in re.finditer(tag_pattern, text):
        result.append(text[last_pos:match.start()])

        is_closing = bool(match.group(1))
        tag_name = match.group(2).lower()

        if is_closing:
            if open_tags and open_tags[-1] == tag_name:
                open_tags.pop()
                result.append(match.group(0))
        else:
            if tag_name in ['b', 'i', 'a', 'code', 'pre']:
                open_tags.append(tag_name)
            result.append(match.group(0))

        last_pos = match.end()

    result.append(text[last_pos:])

    for tag in reversed(open_tags):
        result.append(f'</{tag}>')

    return ''.join(result)

def create_cancel_event(user_id: int) -> Event:
    """Create cancellation event for search"""
    cancel_event = Event()
    active_searches[user_id] = cancel_event
    return cancel_event

def cleanup_search(user_id: int):
    """Clean up search tracking"""
    active_searches.pop(user_id, None)

def is_search_cancelled(user_id: int) -> bool:
    """Check if search is cancelled"""
    event = active_searches.get(user_id)
    return event.is_set() if event else False

def create_location_button():
    """Create reply keyboard with location sharing button"""
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    location_button = telebot.types.KeyboardButton("📍 Share My Location", request_location=True)
    markup.add(location_button)
    markup.add(telebot.types.KeyboardButton("❌ Cancel"))
    return markup

def remove_location_button():
    """Remove reply keyboard"""
    return telebot.types.ReplyKeyboardRemove()

def split_message_for_telegram(message: str, max_length: int = 4096) -> List[str]:
    """Split long messages for Telegram's character limit"""
    if len(message) <= max_length:
        return [message]

    chunks = []
    current_chunk = ""

    paragraphs = message.split('\n\n')
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_length:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            sentences = chunk.split('. ')
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_length:
                    if current_chunk:
                        current_chunk += '. ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    current_chunk = sentence

            if current_chunk:
                final_chunks.append(current_chunk)

    return final_chunks

# ============================================================================
# CORE MESSAGE PROCESSING - AI FIRST!
# ============================================================================

async def process_user_message(
    user_id: int,
    chat_id: int, 
    message_text: str,
    gps_coordinates: Optional[Tuple[float, float]] = None,
    message_type: str = "text"
) -> None:
    """
    Process ANY user message through AI Chat Layer.

    This is the SINGLE ENTRY POINT for all message processing.
    AI decides everything: is it a location answer? new query? conversation?
    """
    try:
        # Show typing indicator
        bot.send_chat_action(chat_id, 'typing')

        logger.info(f"🎯 Processing message for user {user_id}: '{message_text[:50]}...' (gps={gps_coordinates is not None})")

        # Call unified agent - AI makes ALL decisions
        result = await unified_agent.process_message(
            query=message_text,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=f"telegram_{user_id}_{int(time.time())}",
            telegram_bot=bot,
            chat_id=chat_id
        )

        # Check if location button is needed
        if result.get("needs_location_button"):
            logger.info(f"🔘 AI requests location button for user {user_id}")

            # Get the AI response to show with the button
            ai_response = result.get("ai_response", "")

            # Build location request message
            location_msg = ai_response or "📍 Please share your location or tell me your neighborhood."

            # Add instructions if not already included
            if "Option" not in location_msg and "share" not in location_msg.lower():
                location_msg += (
                    "\n\n🗺️ <b>Option 1:</b> Tell me your neighborhood, street, or nearby landmark\n"
                    "📍 <b>Option 2:</b> Use the button below to send your exact coordinates\n\n"
                    "<i>Examples: \"I'm in Chinatown\", \"Near Times Square\", \"Alfama in Lisbon\"</i>"
                )

            bot.send_message(
                chat_id,
                location_msg,
                parse_mode='HTML',
                reply_markup=create_location_button()
            )
            return

        # Handle normal responses
        search_triggered = result.get("search_triggered", False)
        action_taken = result.get("action_taken", "unknown")

        # Send AI response
        ai_response = result.get("ai_response") or result.get("formatted_message")

        if ai_response:
            # For search results, add slight delay after confirmation message
            if search_triggered:
                await asyncio.sleep(2)

            if len(ai_response) > 4000:
                chunks = split_message_for_telegram(ai_response)
                for chunk in chunks:
                    bot.send_message(
                        chat_id,
                        fix_telegram_html(chunk),
                        parse_mode='HTML',
                        disable_web_page_preview=True,
                        reply_markup=remove_location_button()  # Always clear location button
                    )
            else:
                bot.send_message(
                    chat_id,
                    fix_telegram_html(ai_response),
                    parse_mode='HTML',
                    disable_web_page_preview=True,
                    reply_markup=remove_location_button()  # Always clear location button
                )
        elif not search_triggered:
            bot.send_message(
                chat_id,
                "I'm here to help you find amazing restaurants! What are you looking for?",
                parse_mode='HTML',
                reply_markup=remove_location_button()
            )

        # Log success
        processing_time = result.get("processing_time", 0)
        reasoning = result.get("reasoning", "No reasoning provided")

        if search_triggered:
            search_result = result.get("search_result", {})
            if search_result:
                restaurants_count = search_result.get("restaurant_count", 0)
                if restaurants_count == 0 and search_result.get("final_restaurants"):
                    restaurants_count = len(search_result["final_restaurants"])
            else:
                restaurants_count = result.get("restaurant_count", 0)

            logger.info(f"✅ Search completed in {processing_time}s - Found {restaurants_count} restaurants")
        else:
            logger.info(f"✅ Conversation continued in {processing_time}s - Action: {action_taken}")
            logger.info(f"🤖 AI Reasoning: {reasoning}")

    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        logger.error(error_msg)

        bot.send_message(
            chat_id,
            "I'm having a bit of trouble right now. Could you try asking again in a moment?",
            parse_mode='HTML',
            reply_markup=remove_location_button()
        )

# ============================================================================
# TELEGRAM BOT HANDLERS
# ============================================================================

@bot.message_handler(commands=['start'])
def handle_start(message):
    """Handle /start command"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"📱 New user {user_id} started conversation")

    # Clear any pending state for fresh start
    if hasattr(unified_agent, 'ai_chat_layer'):
        unified_agent.ai_chat_layer.clear_pending_gps(user_id)

    bot.send_message(
        chat_id,
        WELCOME_MESSAGE,
        parse_mode='HTML',
        reply_markup=remove_location_button()
    )

@bot.message_handler(commands=['cancel'])
def handle_cancel(message):
    """Handle /cancel command"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Cancel any active search
    event = active_searches.get(user_id)
    if event:
        event.set()
        cleanup_search(user_id)

    # Clear pending GPS state in AI Chat Layer
    if hasattr(unified_agent, 'ai_chat_layer'):
        unified_agent.ai_chat_layer.clear_pending_gps(user_id)

    bot.send_message(
        chat_id,
        "🛑 Cancelled! What else can I help you find?",
        parse_mode='HTML',
        reply_markup=remove_location_button()
    )

@bot.message_handler(func=lambda message: message.text == "❌ Cancel")
def handle_location_cancel_button(message):
    """Handle location cancel button press"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Clear pending GPS state in AI Chat Layer
    if hasattr(unified_agent, 'ai_chat_layer'):
        unified_agent.ai_chat_layer.clear_pending_gps(user_id)

    bot.send_message(
        chat_id,
        "No problem! What would you like to know about restaurants?",
        parse_mode='HTML',
        reply_markup=remove_location_button()
    )

@bot.message_handler(content_types=['location'])
def handle_location_message(message):
    """Handle GPS location messages - send to AI with coordinates"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    latitude = message.location.latitude
    longitude = message.location.longitude

    logger.info(f"📍 GPS location from user {user_id}: {latitude}, {longitude}")

    # Get pending cuisine from AI Chat Layer (if any)
    pending_cuisine = None
    if hasattr(unified_agent, 'ai_chat_layer'):
        pending_state = unified_agent.ai_chat_layer.get_pending_gps_state(user_id)
        pending_cuisine = pending_state.get('pending_gps_cuisine')

    # Build message for AI
    if pending_cuisine:
        # User was asked for location - tell AI they provided GPS
        message_for_ai = f"[User shared GPS location for: {pending_cuisine}]"
    else:
        # Unsolicited GPS - ask what they want
        message_for_ai = "[User shared GPS location]"

    # Remove location button immediately
    bot.send_message(
        chat_id,
        "📍 Got your location!",
        parse_mode='HTML',
        reply_markup=remove_location_button()
    )

    # Process through AI with GPS coordinates
    asyncio.run(process_user_message(
        user_id=user_id,
        chat_id=chat_id,
        message_text=message_for_ai,
        gps_coordinates=(latitude, longitude),
        message_type="location"
    ))

@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    """
    Handle ALL text messages - route to AI first, no exceptions.

    AI Chat Layer decides if this is:
    - A location answer (when GPS was pending)
    - A new complete query
    - A follow-up request
    - General conversation
    """
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_text = message.text

    logger.info(f"📝 Text message from user {user_id}: '{message_text[:50]}...'")

    # Check for active search
    if user_id in active_searches:
        bot.send_message(
            chat_id,
            "⏳ I'm currently searching for you. Please wait or type /cancel to stop.",
            parse_mode='HTML'
        )
        return

    # ALL messages go to AI - no bypasses!
    # AI Chat Layer knows about pending GPS state and will handle appropriately
    asyncio.run(process_user_message(
        user_id=user_id,
        chat_id=chat_id,
        message_text=message_text,
        message_type="text"
    ))

@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages"""
    if not voice_handler:
        bot.reply_to(message, "Voice messages are not supported in this configuration.")
        return

    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"🎙️ Voice message from user {user_id}")

    try:
        # Get file info and download voice message
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Save to temporary file for transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_file:
            temp_file.write(downloaded_file)
            temp_file_path = temp_file.name

        try:
            # Transcribe voice message
            transcription = voice_handler.transcribe_voice_message(temp_file_path)

            if transcription:
                logger.info(f"🎙️ Transcribed: '{transcription[:50]}...'")

                # Process transcribed text through AI Chat Layer
                asyncio.run(process_user_message(
                    user_id=user_id,
                    chat_id=chat_id,
                    message_text=transcription,
                    message_type="voice"
                ))
            else:
                bot.send_message(
                    chat_id,
                    "I couldn't understand your voice message. Could you try typing instead?",
                    parse_mode='HTML'
                )
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        bot.send_message(
            chat_id,
            "I had trouble processing your voice message. Could you try typing instead?",
            parse_mode='HTML'
        )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Start the Telegram bot with all features"""
    global ai_message_generator

    logger.info("🤖 Starting Telegram bot with AI-FIRST routing")
    logger.info("✅ All messages routed through AI Chat Layer")
    logger.info("📍 Location state managed by AI (single source of truth)")
    logger.info("🚫 No message bypasses - AI decides everything")

    try:
        # Initialize AI message generator
        try:
            ai_message_generator = ChatOpenAI(
                model=getattr(config, 'AI_MESSAGE_MODEL', 'gpt-4o-mini'),
                temperature=0.7,
                max_tokens=200,
                api_key=config.OPENAI_API_KEY
            )
            logger.info("✅ AI message generator initialized")
        except Exception as e:
            logger.warning(f"⚠️ AI message generator failed to initialize: {e}. Using fallback messages.")
            ai_message_generator = None

        logger.info("🎬 Videos: City searches use media/searching.mp4")
        logger.info("📍 Videos: Location searches use media/vicinity_search.mp4")

        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"❌ Bot polling error: {e}")
        raise

if __name__ == "__main__":
    main()