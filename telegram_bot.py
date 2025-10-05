# telegram_bot.py
"""
Memory-Enhanced Telegram Bot for Restaurant Recommendations

This bot integrates with the AI Chat Layer and Memory System to provide:
1. Intelligent conversation routing (chat vs search)
2. Memory-aware responses
3. Natural conversation flow
4. Context preservation across sessions

NO MORE AUTOMATED SEARCH MESSAGES - All responses are now contextual and intelligent.
"""

import telebot
import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from threading import Event
import time

# Import the memory-enhanced unified agent
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

# Initialize memory-enhanced unified agent (single source of truth)
unified_agent = create_unified_restaurant_agent(config)

# Initialize voice handler for transcription only
voice_handler = VoiceMessageHandler() if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY else None

# Simple cancellation tracking (Telegram-specific concern)
active_searches = {}  # user_id -> Event

# Welcome message (UI concern - stays in telegram bot)
WELCOME_MESSAGE = (
    "🍸 Hello! I'm Restaurant Babe. I know all about the most delicious restaurants worldwide.\n\n"
    "Tell me what you're looking for, like <i>best ramen in Tokyo</i>, or share your location for nearby recommendations.\n\n"
    "I remember our conversations and your preferences, so feel free to just say things like \"more options\" or \"different cuisine\"!\n\n"
    "💡 <b>Tip:</b> Type /cancel to stop a search.\n\n"
    "What are you hungry for?"
)


# ============================================================================
# CORE MESSAGE PROCESSING (Delegates to Memory-Enhanced Agent)
# ============================================================================

async def process_user_message(
    user_id: int,
    chat_id: int, 
    message_text: str,
    gps_coordinates: Optional[tuple] = None,
    message_type: str = "text"
) -> None:
    """
    CORE FUNCTION: Process any user message through memory-enhanced agent

    This now provides intelligent, contextual responses instead of 
    automated search messages.
    """
    processing_msg = None

    try:
        # 1. TELEGRAM CONCERN: Show initial processing (brief)
        processing_msg = show_brief_processing_message(chat_id, message_text)

        # 2. DELEGATE TO MEMORY-ENHANCED AGENT: All intelligence happens here
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

        if ai_response:
            # Split long messages for Telegram
            if len(ai_response) > 4000:
                # Send in chunks
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
            # Fallback response
            bot.send_message(
                chat_id,
                "I'm here to help you find amazing restaurants! What are you looking for?",
                parse_mode='HTML'
            )

        # 5. Log success
        processing_time = result.get("processing_time", 0)
        logger.info(f"✅ Message processed successfully in {processing_time}s")

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
# TELEGRAM MESSAGE HANDLERS
# ============================================================================

@bot.message_handler(commands=['start'])
def handle_start_command(message):
    """Handle /start command"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"📱 New user {user_id} started conversation")

    bot.send_message(
        chat_id,
        fix_telegram_html(WELCOME_MESSAGE),
        parse_mode='HTML'
    )


@bot.message_handler(commands=['cancel'])
def handle_cancel_command(message):
    """Handle /cancel command to stop searches"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Remove from active searches if present
    if user_id in active_searches:
        try:
            active_searches[user_id].set()  # Signal cancellation
            del active_searches[user_id]
            bot.send_message(chat_id, "🛑 Search cancelled.")
        except Exception as e:
            logger.error(f"Error cancelling search: {e}")
            bot.send_message(chat_id, "✅ No active search to cancel.")
    else:
        bot.send_message(chat_id, "✅ No active search to cancel.")


@bot.message_handler(commands=['memory'])
def handle_memory_command(message):
    """Handle /memory command to show user preferences"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    async def get_memory():
        try:
            memory_summary = await unified_agent.get_user_memory_summary(user_id)

            if memory_summary and memory_summary.get("memory_summary"):
                prefs = memory_summary["memory_summary"]

                response = "🧠 <b>What I remember about your food preferences:</b>\n\n"

                if prefs.get("cuisine_preferences"):
                    response += f"🍽️ <b>Cuisines:</b> {', '.join(prefs['cuisine_preferences'])}\n"

                if prefs.get("dietary_restrictions"):
                    response += f"🥗 <b>Dietary:</b> {', '.join(prefs['dietary_restrictions'])}\n"

                if prefs.get("price_preferences"):
                    response += f"💰 <b>Budget:</b> {', '.join(prefs['price_preferences'])}\n"

                if prefs.get("favorite_cities"):
                    response += f"🌆 <b>Cities searched:</b> {', '.join(prefs['favorite_cities'])}\n"

                if prefs.get("conversation_style"):
                    response += f"💬 <b>Communication style:</b> {prefs['conversation_style']}\n"

                response += "\n💡 I use this info to give you better recommendations!"

            else:
                response = "🧠 I'm just getting to know your preferences! Tell me what restaurants you're looking for to help me learn."

            bot.send_message(
                chat_id,
                fix_telegram_html(response),
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Error getting user memory: {e}")
            bot.send_message(
                chat_id,
                "🧠 I'm having trouble accessing your preferences right now. Try asking for restaurants!",
                parse_mode='HTML'
            )

    # Run async function
    asyncio.run(get_memory())


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

    # Delegate to memory-enhanced agent with coordinates
    asyncio.run(process_user_message(
        user_id=user_id,
        chat_id=chat_id,
        message_text="restaurants near me",  # Default query for location
        gps_coordinates=gps_coordinates
    ))


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages - TRANSCRIPTION ONLY"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"🎙️ Voice message from user {user_id}")

    # Check if voice handler is available
    if not voice_handler:
        bot.send_message(
            chat_id,
            "🔇 Sorry, voice message processing isn't available right now. Please send text.",
            parse_mode='HTML'
        )
        return

    # Show transcription progress
    processing_msg = bot.send_message(chat_id, "🎙️ Transcribing your message...")

    try:
        # TRANSCRIPTION ONLY - no business logic
        transcribed_text = voice_handler.process_voice_message(bot, message.voice)

        # Clean up transcription message
        if processing_msg:
            bot.delete_message(chat_id, processing_msg.id)

        if transcribed_text:
            logger.info(f"🎙️ Voice transcribed for user {user_id}: '{transcribed_text[:50]}...'")

            # Delegate transcribed text to memory-enhanced agent
            asyncio.run(process_user_message(user_id, chat_id, transcribed_text, message_type="voice"))
        else:
            bot.send_message(chat_id, "😔 Couldn't understand your voice message. Please try again.")

    except Exception as e:
        logger.error(f"❌ Voice transcription error: {e}")
        if processing_msg:
            bot.delete_message(chat_id, processing_msg.id)
        bot.send_message(chat_id, "😔 Error processing voice message.")


@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    """Handle all text messages - Main entry point"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_text = message.text

    logger.info(f"📝 Text message from user {user_id}: '{message_text[:50]}...'")

    # Delegate to memory-enhanced agent
    asyncio.run(process_user_message(user_id, chat_id, message_text, message_type="text"))


# ============================================================================
# HUMAN-IN-THE-LOOP CALLBACK HANDLER (Preserved from original)
# ============================================================================

@bot.callback_query_handler(func=lambda call: call.data.startswith("decision_"))
def handle_decision_callback(call):
    """Handle human-in-the-loop decisions (for location enhancement)"""
    try:
        user_id = call.from_user.id
        decision = call.data.replace("decision_", "")  # "accept" or "skip"

        logger.info(f"🤔 Human decision from user {user_id}: {decision}")

        # Answer the callback to remove loading state
        bot.answer_callback_query(call.id, f"Decision: {decision}")

        # Generate thread ID for this decision
        thread_id = f"telegram_{user_id}_{int(time.time())}"

        # Handle the decision through unified agent
        async def handle_async_decision():
            try:
                # FIXED: handle_human_decision is synchronous, not async
                result = unified_agent.handle_human_decision(thread_id, decision)

                if result.get("success"):
                    # Send the final results
                    final_message = result.get("formatted_message") or result.get("ai_response")
                    if final_message:
                        bot.send_message(
                            call.message.chat.id,
                            fix_telegram_html(final_message),
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                else:
                    bot.send_message(
                        call.message.chat.id,
                        "I had some trouble processing that decision. Could you try asking again?",
                        parse_mode='HTML'
                    )

            except Exception as e:
                logger.error(f"Error handling human decision: {e}")
                bot.send_message(
                    call.message.chat.id,
                    "Sorry, I encountered an error. Please try your request again.",
                    parse_mode='HTML'
                )

        # Run async function
        asyncio.run(handle_async_decision())

    except Exception as e:
        logger.error(f"Error in decision callback: {e}")
        bot.send_message(
            call.message.chat.id,
            "Something went wrong. Please try again.",
            parse_mode='HTML'
        )


# ============================================================================
# TELEGRAM UTILITY FUNCTIONS
# ============================================================================

def show_brief_processing_message(chat_id: int, message_text: str) -> Optional[telebot.types.Message]:
    """Show a brief, contextual processing message"""
    try:
        # Contextual processing messages based on message content
        if any(word in message_text.lower() for word in ['help', 'start', 'hello', 'hi']):
            return bot.send_message(chat_id, "👋 One moment...")
        elif 'location' in message_text.lower() or 'near' in message_text.lower():
            return bot.send_message(chat_id, "📍 Looking for restaurants nearby...")
        elif any(word in message_text.lower() for word in ['restaurant', 'food', 'eat', 'dining']):
            return bot.send_message(chat_id, "🔍 Searching for restaurants...")
        else:
            return bot.send_message(chat_id, "🤔 Let me think...")
    except Exception as e:
        logger.error(f"Error showing processing message: {e}")
        return None


def fix_telegram_html(text: str) -> str:
    """Fix HTML formatting for Telegram"""
    if not text:
        return ""

    # Telegram HTML fixes
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')

    # But allow these specific HTML tags
    text = text.replace('&lt;b&gt;', '<b>')
    text = text.replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>')
    text = text.replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;u&gt;', '<u>')
    text = text.replace('&lt;/u&gt;', '</u>')
    text = text.replace('&lt;code&gt;', '<code>')
    text = text.replace('&lt;/code&gt;', '</code>')
    text = text.replace('&lt;pre&gt;', '<pre>')
    text = text.replace('&lt;/pre&gt;', '</pre>')

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
                # Paragraph is too long, split by sentences
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
    """Start the Telegram bot"""
    logger.info("🤖 Starting Telegram bot with AI Chat Layer integration")
    logger.info("✅ Memory-enhanced unified agent initialized")
    logger.info("🎯 All messages will be processed through AI Chat Layer")

    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"❌ Bot polling error: {e}")
        raise


if __name__ == "__main__":
    main()