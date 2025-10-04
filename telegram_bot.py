# telegram_bot_clean.py
"""
Clean Telegram Bot - ONLY Messaging, NO Business Logic

This bot is a thin wrapper that:
1. Receives Telegram messages
2. Sends them to the unified LangGraph agent  
3. Displays agent responses
4. Handles Telegram-specific UI elements

ALL business logic is handled by the unified agent.
"""

import telebot
import asyncio
import logging
import os
from typing import Optional, Dict, Any
from threading import Event

# Import ONLY the unified agent - no orchestrators!
from agents.unified_restaurant_agent import create_unified_restaurant_agent
from utils.voice_handler import VoiceMessageHandler
from utils.database import initialize_database
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize bot
if not config.TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is required in config")
bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize database FIRST
initialize_database(config)

# Initialize unified agent (single source of truth)
unified_agent = create_unified_restaurant_agent(config)

# Initialize voice handler for transcription only
voice_handler = VoiceMessageHandler() if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY else None

# Simple cancellation tracking (Telegram-specific concern)
active_searches = {}  # user_id -> Event

# Welcome message (UI concern - stays in telegram bot)
WELCOME_MESSAGE = (
    "🍸 Hello! I'm Restaurant Babe. I know all about the most delicious restaurants worldwide.\n\n"
    "Tell me what you're looking for, like <i>best ramen in Tokyo</i>, or share your location for nearby recommendations.\n\n"
    "💡 <b>Tip:</b> Type /cancel to stop a search.\n\n"
    "What are you hungry for?"
)


# ============================================================================
# CORE MESSAGE PROCESSING (Delegates to Unified Agent)
# ============================================================================

async def process_user_message(
    user_id: int,
    chat_id: int, 
    message_text: str,
    gps_coordinates: Optional[tuple] = None,
    message_type: str = "text"
) -> None:
    """
    CORE FUNCTION: Process any user message through unified agent

    This is the ONLY place where business logic decisions are made,
    and they're all delegated to the unified agent.
    """
    processing_msg = None

    try:
        # 1. TELEGRAM CONCERN: Show processing message
        processing_msg = show_processing_message(chat_id, message_text, gps_coordinates)

        # 2. TELEGRAM CONCERN: Set up cancellation
        cancel_event = setup_cancellation(user_id, chat_id)

        # 3. BUSINESS LOGIC: Delegate everything to unified agent
        result = await unified_agent.search_restaurants(
            query=message_text,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=f"user_{user_id}"
        )

        # 4. TELEGRAM CONCERN: Clean up processing message
        cleanup_processing_message(chat_id, processing_msg)

        # 5. TELEGRAM CONCERN: Check cancellation
        if is_cancelled(user_id):
            return

        # 6. TELEGRAM CONCERN: Display agent response
        await display_agent_response(chat_id, user_id, result)

    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")
        cleanup_processing_message(chat_id, processing_msg)
        send_error_message(chat_id, "I encountered an error. Please try again.")

    finally:
        cleanup_cancellation(user_id)


async def display_agent_response(chat_id: int, user_id: int, result: Dict[str, Any]) -> None:
    """
    TELEGRAM CONCERN: Display unified agent response

    Handles different response types from the agent.
    """
    if result.get("success"):
        # Agent succeeded - display results
        formatted_message = result.get("formatted_message", "Found some great restaurants!")

        bot.send_message(
            chat_id,
            formatted_message,
            parse_mode='HTML',
            disable_web_page_preview=True
        )

        # Handle human-in-the-loop decisions (Telegram UI concern)
        if result.get("human_decision_pending"):
            await handle_human_decision_ui(chat_id, user_id, result)

        logger.info(f"✅ Sent results to user {user_id}: {len(result.get('final_restaurants', []))} restaurants")

    else:
        # Agent failed - display error
        error_message = result.get("error_message", "Search failed")
        send_error_message(chat_id, f"😔 {error_message}")


async def handle_human_decision_ui(chat_id: int, user_id: int, result: Dict[str, Any]) -> None:
    """
    TELEGRAM CONCERN: Show human-in-the-loop UI

    The decision logic is in the agent, this just shows the UI.
    """
    decision_message = result.get("human_decision_message", "Found some restaurants. Search for more?")

    # Create Telegram inline keyboard
    keyboard = telebot.types.InlineKeyboardMarkup()
    keyboard.row(
        telebot.types.InlineKeyboardButton(
            "🔍 Yes, find more options", 
            callback_data=f"decision_accept_{user_id}"
        )
    )
    keyboard.row(
        telebot.types.InlineKeyboardButton(
            "✅ These are perfect", 
            callback_data=f"decision_skip_{user_id}"
        )
    )

    bot.send_message(
        chat_id,
        f"<b>{decision_message}</b>\n\nI can search for additional options in the same area.",
        reply_markup=keyboard,
        parse_mode='HTML'
    )


# ============================================================================
# TELEGRAM MESSAGE HANDLERS (Pure Messaging)
# ============================================================================

@bot.message_handler(commands=['start'])
def handle_start(message):
    """Handle /start command - Pure UI"""
    bot.send_message(
        message.chat.id,
        WELCOME_MESSAGE,
        parse_mode='HTML'
    )


@bot.message_handler(commands=['cancel'])
def handle_cancel(message):
    """Handle /cancel command - Telegram-specific cancellation"""
    user_id = message.from_user.id

    if user_id in active_searches:
        active_searches[user_id].set()
        bot.send_message(message.chat.id, "🛑 Search cancelled.")
        logger.info(f"🛑 User {user_id} cancelled search")
    else:
        bot.send_message(message.chat.id, "No active search to cancel.")


@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    """Handle text messages - Delegate to unified agent"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    text = message.text.strip()

    if not text or text.startswith('/'):
        return

    logger.info(f"📝 Text message from user {user_id}: '{text[:50]}...'")

    # Delegate everything to unified agent
    asyncio.run(process_user_message(user_id, chat_id, text))


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages - Transcribe then delegate"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    if not voice_handler:
        bot.send_message(chat_id, "😔 Voice messages are not available. Please send text.")
        return

    # Show transcription progress
    processing_msg = bot.send_message(chat_id, "🎙️ Transcribing your message...")

    try:
        # TRANSCRIPTION ONLY - no business logic
        transcribed_text = voice_handler.process_voice_message(bot, message.voice)

        # Clean up transcription message
        bot.delete_message(chat_id, processing_msg.message_id)

        if transcribed_text:
            logger.info(f"🎙️ Voice transcribed for user {user_id}: '{transcribed_text[:50]}...'")

            # Delegate transcribed text to unified agent
            asyncio.run(process_user_message(user_id, chat_id, transcribed_text, message_type="voice"))
        else:
            bot.send_message(chat_id, "😔 Couldn't understand your voice message. Please try again.")

    except Exception as e:
        logger.error(f"❌ Voice transcription error: {e}")
        bot.delete_message(chat_id, processing_msg.message_id)
        bot.send_message(chat_id, "😔 Error processing voice message.")


@bot.message_handler(content_types=['location'])
def handle_location_message(message):
    """Handle location messages - Extract coordinates and delegate"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Extract GPS coordinates (Telegram-specific)
    latitude = message.location.latitude
    longitude = message.location.longitude
    gps_coordinates = (latitude, longitude)

    logger.info(f"📍 Location from user {user_id}: {latitude:.4f}, {longitude:.4f}")

    # Delegate to unified agent with coordinates
    asyncio.run(process_user_message(
        user_id=user_id,
        chat_id=chat_id,
        message_text="restaurants near me",  # Default query for location
        gps_coordinates=gps_coordinates
    ))


# ============================================================================
# HUMAN-IN-THE-LOOP CALLBACK HANDLER (Telegram UI)
# ============================================================================

@bot.callback_query_handler(func=lambda call: call.data.startswith("decision_"))
def handle_decision_callback(call):
    """Handle human decision callbacks - Pure UI delegation"""
    try:
        parts = call.data.split("_")
        action = parts[1]  # "accept" or "skip"  
        user_id = int(parts[2])

        # Validate user
        if call.from_user.id != user_id:
            bot.answer_callback_query(call.id, "❌ Invalid user")
            return

        # Show processing
        bot.edit_message_text(
            "🔄 Processing your choice...",
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML'
        )

        # Delegate decision to unified agent
        asyncio.run(continue_agent_workflow(
            user_id=user_id,
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            decision=action
        ))

    except Exception as e:
        logger.error(f"❌ Error handling decision callback: {e}")
        bot.answer_callback_query(call.id, "❌ Error occurred")


async def continue_agent_workflow(user_id: int, chat_id: int, message_id: int, decision: str):
    """Continue unified agent workflow after human decision"""
    try:
        # BUSINESS LOGIC: Delegate to unified agent
        result = unified_agent.handle_human_decision(
            thread_id=f"user_{user_id}",
            decision=decision
        )

        # TELEGRAM CONCERN: Update UI with agent response
        if result.get("success"):
            formatted_message = result["formatted_message"]

            bot.edit_message_text(
                formatted_message,
                chat_id,
                message_id,
                parse_mode='HTML'
            )

            logger.info(f"✅ Decision {decision} completed for user {user_id}")

        else:
            error_message = result.get("error_message", "Decision processing failed")
            bot.edit_message_text(
                f"😔 {error_message}",
                chat_id,
                message_id,
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"❌ Error continuing workflow: {e}")
        bot.edit_message_text(
            "😔 Something went wrong. Please try a new search.",
            chat_id,
            message_id,
            parse_mode='HTML'
        )


# ============================================================================
# TELEGRAM-SPECIFIC UTILITY FUNCTIONS
# ============================================================================

def show_processing_message(chat_id: int, message_text: str, gps_coordinates: Optional[tuple]) -> Optional[telebot.types.Message]:
    """Show processing message - Pure Telegram UI"""
    try:
        if gps_coordinates:
            text = "🔍 <b>Searching for restaurants near your location...</b>"
        else:
            text = f"🔍 <b>Searching for {message_text[:30]}...</b>"

        return bot.send_message(chat_id, text, parse_mode='HTML')
    except Exception:
        return None


def cleanup_processing_message(chat_id: int, processing_msg: Optional[telebot.types.Message]) -> None:
    """Clean up processing message - Pure Telegram UI"""
    if processing_msg:
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except Exception:
            pass


def send_error_message(chat_id: int, message: str) -> None:
    """Send error message - Pure Telegram UI"""
    bot.send_message(chat_id, message, parse_mode='HTML')


def setup_cancellation(user_id: int, chat_id: int) -> Event:
    """Set up cancellation tracking - Telegram-specific"""
    cancel_event = Event()
    active_searches[user_id] = cancel_event
    return cancel_event


def is_cancelled(user_id: int) -> bool:
    """Check if search is cancelled - Telegram-specific"""
    if user_id in active_searches:
        return active_searches[user_id].is_set()
    return False


def cleanup_cancellation(user_id: int) -> None:
    """Clean up cancellation tracking - Telegram-specific"""
    active_searches.pop(user_id, None)


# ============================================================================
# BOT STARTUP
# ============================================================================

def main():
    """Start the clean telegram bot"""
    logger.info("🚀 Starting Clean Telegram Bot (Business Logic in LangGraph)")

    try:
        # Test unified agent initialization
        logger.info("🧪 Testing unified agent...")
        test_agent = create_unified_restaurant_agent(config)
        logger.info("✅ Unified agent ready")

        # Start bot
        logger.info("🤖 Starting Telegram bot polling...")
        bot.infinity_polling(timeout=10, long_polling_timeout=5)

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot startup error: {e}")
        raise


if __name__ == "__main__":
    main()
