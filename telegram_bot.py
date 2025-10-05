# telegram_bot.py
"""
UPDATED Telegram Bot - Enhanced AI Chat Integration

This version uses the enhanced AI Chat Layer that:
- Manages conversation flow intelligently
- Collects information before triggering search
- Provides natural conversation responses
- Only routes to search pipeline when ready

KEY IMPROVEMENT: Fixes the flow so messages go through AI Chat Layer first,
not directly to the query analyzer.
"""

import telebot
import asyncio
import logging
import time
from typing import Optional, List
from threading import Event

# Import the enhanced unified agent
from agents.unified_restaurant_agent import create_unified_restaurant_agent
from utils.voice_handler import VoiceMessageHandler
from utils.database import initialize_database
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

# Initialize enhanced unified agent with AI Chat Layer
unified_agent = create_unified_restaurant_agent(config)

# Initialize voice handler for transcription only
voice_handler = VoiceMessageHandler() if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY else None

# Simple cancellation tracking (Telegram-specific concern)
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
# CORE MESSAGE PROCESSING (Enhanced AI Chat Layer)
# ============================================================================

async def process_user_message(
    user_id: int,
    chat_id: int, 
    message_text: str,
    gps_coordinates: Optional[tuple] = None,
    message_type: str = "text"
) -> None:
    """
    ENHANCED ENTRY POINT: Process any user message through AI Chat Layer

    The AI Chat Layer now decides whether to:
    1. Continue conversation to collect more info
    2. Trigger city-wide search when ready  
    3. Trigger location-based search when ready
    4. Handle follow-up requests
    """
    processing_msg = None

    try:
        # 1. TELEGRAM CONCERN: Show contextual processing message
        processing_msg = show_contextual_processing_message(chat_id, message_text)

        # 2. AI CHAT LAYER: All intelligence happens here
        logger.info(f"🎯 Processing message for user {user_id}: '{message_text[:50]}...'")

        result = await unified_agent.restaurant_search_with_memory(
            query=message_text,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=f"telegram_{user_id}_{int(time.time())}"
        )

        # 3. TELEGRAM CONCERN: Clean up processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.id)
            except Exception:
                pass  # Message might already be deleted

        # 4. TELEGRAM CONCERN: Send the AI response
        ai_response = result.get("ai_response") or result.get("formatted_message")
        search_triggered = result.get("search_triggered", False)
        action_taken = result.get("action_taken", "unknown")

        if ai_response:
            # Handle long messages
            if len(ai_response) > 4000:
                chunks = split_message_for_telegram(ai_response)
                for chunk in chunks:
                    bot.send_message(
                        chat_id,
                        fix_telegram_html(chunk),
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    )
            else:
                bot.send_message(
                    chat_id,
                    fix_telegram_html(ai_response),
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
        else:
            # Minimal fallback - should rarely happen
            bot.send_message(
                chat_id,
                "I'm here to help you find amazing restaurants! What are you looking for?",
                parse_mode='HTML'
            )

        # 5. Log success
        processing_time = result.get("processing_time", 0)
        reasoning = result.get("reasoning", "No reasoning provided")

        if search_triggered:
            restaurants_count = len(result.get("final_restaurants", []))
            logger.info(f"✅ Search triggered and completed in {processing_time}s - Found {restaurants_count} restaurants")
        else:
            logger.info(f"✅ Conversation continued in {processing_time}s - Action: {action_taken}")
            logger.info(f"🤖 AI Reasoning: {reasoning}")

    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        logger.error(error_msg)

        # Clean up processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.id)
            except Exception:
                pass

        # Send error response
        bot.send_message(
            chat_id,
            "I'm having a bit of trouble right now. Could you try asking again in a moment?",
            parse_mode='HTML'
        )


# ============================================================================
# TELEGRAM MESSAGE HANDLERS (Entry Points)
# ============================================================================

@bot.message_handler(commands=['start'])
def handle_start(message):
    """Handle /start command"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"📱 New user {user_id} started conversation")

    bot.send_message(
        chat_id,
        WELCOME_MESSAGE,
        parse_mode='HTML'
    )


@bot.message_handler(commands=['cancel'])
def handle_cancel(message):
    """Handle /cancel command"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Cancel any active searches
    if user_id in active_searches:
        active_searches[user_id].set()
        del active_searches[user_id]
        bot.send_message(chat_id, "🛑 Search cancelled.", parse_mode='HTML')
    else:
        bot.send_message(chat_id, "No active search to cancel.", parse_mode='HTML')


@bot.message_handler(commands=['reset'])
def handle_reset(message):
    """Handle /reset command - clear conversation history"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    try:
        # Clear AI Chat Layer session
        if hasattr(unified_agent, 'ai_chat_layer'):
            unified_agent.ai_chat_layer.clear_session(user_id)

        bot.send_message(
            chat_id, 
            "🔄 Conversation reset! Let's start fresh. What restaurants are you looking for?",
            parse_mode='HTML'
        )
        logger.info(f"🔄 Reset conversation for user {user_id}")

    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        bot.send_message(chat_id, "Reset completed!", parse_mode='HTML')


@bot.message_handler(content_types=['location'])
def handle_location(message):
    """Handle GPS location sharing"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    if message.location:
        coordinates = (message.location.latitude, message.location.longitude)
        logger.info(f"📍 GPS location from user {user_id}: {coordinates}")

        # Process with location context
        asyncio.run(process_user_message(
            user_id, chat_id, 
            "Find restaurants near my current location",
            gps_coordinates=coordinates,
            message_type="location"
        ))
    else:
        bot.send_message(chat_id, "I couldn't get your location. Could you try again?", parse_mode='HTML')


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"🎤 Voice message from user {user_id}")

    if not voice_handler:
        bot.send_message(chat_id, "Voice processing is not available. Please send a text message.", parse_mode='HTML')
        return

    # Show processing message
    processing_msg = bot.send_message(chat_id, "🎤 <b>Processing your voice message...</b>", parse_mode='HTML')

    try:
        # Transcribe voice
        transcribed_text = voice_handler.process_voice_message(bot, message.voice)

        # Clean up processing message
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except Exception:
            pass

        if transcribed_text:
            logger.info(f"✅ Voice transcribed: '{transcribed_text[:100]}...'")
            # Process as text message
            asyncio.run(process_user_message(user_id, chat_id, transcribed_text, message_type="voice"))
        else:
            bot.send_message(chat_id, "I couldn't understand your voice message. Could you try again?", parse_mode='HTML')

    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except Exception:
            pass
        bot.send_message(chat_id, "Error processing voice message. Please try again.", parse_mode='HTML')


@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    """Handle all text messages - MAIN ENTRY POINT"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_text = message.text

    logger.info(f"📝 Text message from user {user_id}: '{message_text[:50]}...'")

    # Delegate to Enhanced AI Chat Layer
    asyncio.run(process_user_message(user_id, chat_id, message_text, message_type="text"))


# ============================================================================
# TELEGRAM UTILITY FUNCTIONS
# ============================================================================

def show_contextual_processing_message(chat_id: int, message_text: str) -> Optional[telebot.types.Message]:
    """Show contextual processing message based on user input"""
    try:
        message_lower = message_text.lower()

        if any(word in message_lower for word in ['help', 'start', 'hello', 'hi']):
            return bot.send_message(chat_id, "👋 One moment...")
        elif any(word in message_lower for word in ['near', 'nearby', 'around', 'location']):
            return bot.send_message(chat_id, "📍 Looking for restaurants nearby...")
        elif any(word in message_lower for word in ['restaurant', 'food', 'eat', 'dining', 'lunch', 'dinner']):
            return bot.send_message(chat_id, "🔍 Searching for restaurants...")
        else:
            return bot.send_message(chat_id, "🤔 Let me think about that...")

    except Exception as e:
        logger.error(f"Error showing processing message: {e}")
        return None


def fix_telegram_html(text: str) -> str:
    """Fix HTML formatting for Telegram"""
    if not text:
        return ""

    # Escape HTML entities first
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')

    # Allow specific HTML tags
    allowed_tags = ['b', 'i', 'u', 'code', 'pre', 'a']
    for tag in allowed_tags:
        text = text.replace(f'&lt;{tag}&gt;', f'<{tag}>')
        text = text.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
        # Handle tags with attributes (like <a href="...">)
        text = text.replace(f'&lt;{tag} ', f'<{tag} ')

    return text


def split_message_for_telegram(message: str, max_length: int = 4000) -> List[str]:
    """Split long messages into Telegram-compatible chunks"""
    if len(message) <= max_length:
        return [message]

    chunks = []
    current_chunk = ""

    # Split by paragraphs first
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
            else:
                # Paragraph too long, split by sentences
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_length:
                        if current_chunk:
                            current_chunk += '. ' + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Start the Telegram bot with Enhanced AI Chat Layer integration"""
    logger.info("🤖 Starting Telegram bot with Enhanced AI Chat Layer integration")
    logger.info("✅ Enhanced unified agent with AI Chat Layer initialized")
    logger.info("🎯 All messages processed through AI Chat Layer for intelligent conversation flow")
    logger.info("🔧 Fixed: No more direct routing to query analyzer - conversation managed intelligently")

    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"❌ Bot polling error: {e}")
        raise


if __name__ == "__main__":
    main()