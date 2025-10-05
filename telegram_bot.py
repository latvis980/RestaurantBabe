# telegram_bot.py - UPDATED: AI-Generated Search Videos with Existing AI Chat Layer
"""
Telegram Bot with Enhanced Search Messages

IMPORTANT: This version works with the EXISTING AI Chat Layer in unified_restaurant_agent.py
- Keeps the existing AI chat architecture 
- ONLY adds AI-generated search messages with videos when searches are triggered
- Removes automated "let me think about that" messages
- Replaces with typing indicator for non-search interactions
"""

import telebot
import asyncio
import logging
import time
import os
from typing import Optional, List
from threading import Event
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import the enhanced unified agent (with existing AI Chat Layer)
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

# Initialize enhanced unified agent with EXISTING AI Chat Layer
unified_agent = create_unified_restaurant_agent(config)

# Initialize AI for search message generation
ai_message_generator = None

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
# AI MESSAGE GENERATION FOR SEARCH VIDEOS
# ============================================================================

def generate_search_message(search_query: str, search_type: str = "city_wide") -> str:
    """
    Generate AI-powered search message for Restaurant Babe

    Args:
        search_query: The user's restaurant search query
        search_type: Either "city_wide" or "location_based"

    Returns:
        AI-generated message string
    """
    global ai_message_generator

    try:
        if ai_message_generator is None:
            logger.warning("AI message generator not initialized, using fallback")
            return get_fallback_search_message(search_type)

        # Create context-aware prompt for Restaurant Babe
        if search_type == "city_wide":
            context = "searching across an entire city for the best restaurants"
            action_desc = "checking my curated collection, consulting my foodie network, and reviewing recent press coverage"
        else:  # location_based
            context = "searching for restaurants in the specific area you mentioned"
            action_desc = "checking what's in that vicinity, calling my local contacts, and reviewing neighborhood guides"

        prompt = f"""You are Restaurant Babe, a sophisticated AI restaurant expert. You're about to start {context} based on this query: "{search_query}"

Generate a brief, engaging message (2-3 lines max) that:
1. Confirms you're starting the search
2. Mentions it might take a minute 
3. References that you're {action_desc}
4. Keep the tone warm, professional, and enthusiastic like a knowledgeable foodie friend

Use HTML formatting with <b> for emphasis. Don't use emojis in the text (they'll be added to the video caption).

Examples of the style:
- "Perfect! I'm searching for amazing [cuisine] spots in [location]. This might take a minute while I check my notes and reach out to my foodie contacts."
- "Great choice! Let me find you the best restaurants in that area. I'm consulting my curated guides and checking with local food critics."

Generate the message now:"""

        # Generate the message
        response = ai_message_generator.invoke([HumanMessage(content=prompt)])
        message = response.content.strip()

        # Clean up any unwanted characters
        message = message.replace('"', '').replace('*', '').strip()

        logger.info(f"✅ Generated AI search message: {message[:50]}...")
        return message

    except Exception as e:
        logger.error(f"❌ Error generating AI search message: {e}")
        return get_fallback_search_message(search_type)


def get_fallback_search_message(search_type: str) -> str:
    """Fallback messages when AI generation fails"""
    if search_type == "city_wide":
        return ("<b>Perfect! I'm searching for the best restaurants for you.</b>\n\n"
                "This might take a minute while I check my curated collection and consult with my foodie network.")
    else:  # location_based
        return ("<b>Great! I'm searching for amazing restaurants in that area.</b>\n\n"
                "Give me a moment to check my local guides and reach out to my contacts in the vicinity.")


def send_search_message_with_video(chat_id: int, search_query: str, search_type: str) -> Optional[telebot.types.Message]:
    """
    Send search message with appropriate video

    Args:
        chat_id: Telegram chat ID
        search_query: User's search query
        search_type: "city_wide" or "location_based"

    Returns:
        Message object or None if failed
    """
    try:
        # Generate AI message
        ai_message = generate_search_message(search_query, search_type)

        # Choose appropriate video
        if search_type == "city_wide":
            video_path = 'media/searching.mp4'
            fallback_emoji = "🔍"
        else:  # location_based
            video_path = 'media/vicinity_search.mp4'
            fallback_emoji = "📍"

        # Try to send with video first
        try:
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video:
                    return bot.send_video(
                        chat_id,
                        video,
                        caption=f"{fallback_emoji} {ai_message}",
                        parse_mode='HTML'
                    )
            else:
                logger.warning(f"Video file not found: {video_path}")
                raise FileNotFoundError("Video not available")

        except Exception as video_error:
            logger.warning(f"Could not send video: {video_error}")
            # Fallback to text message with emoji
            return bot.send_message(
                chat_id,
                f"{fallback_emoji} {ai_message}",
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"Error sending search message: {e}")
        # Ultimate fallback
        return bot.send_message(
            chat_id,
            "🔍 <b>Searching for restaurants...</b>\n\nThis might take a moment.",
            parse_mode='HTML'
        )


# ============================================================================
# CORE MESSAGE PROCESSING (Enhanced AI Chat Layer Integration)
# ============================================================================

async def process_user_message(
    user_id: int,
    chat_id: int, 
    message_text: str,
    gps_coordinates: Optional[tuple] = None,
    message_type: str = "text"
) -> None:
    """
    ENHANCED ENTRY POINT: Process any user message through EXISTING AI Chat Layer

    The EXISTING AI Chat Layer decides whether to:
    1. Continue conversation to collect more info
    2. Trigger city-wide search when ready  
    3. Trigger location-based search when ready
    4. Handle follow-up requests

    THIS VERSION: Only adds video messages when searches are triggered
    """
    processing_msg = None

    try:
        # 1. UPDATED: Show typing indicator instead of contextual processing message
        bot.send_chat_action(chat_id, 'typing')

        # 2. AI CHAT LAYER: All intelligence happens here (EXISTING SYSTEM)
        logger.info(f"🎯 Processing message for user {user_id}: '{message_text[:50]}...'")

        result = await unified_agent.restaurant_search_with_memory(
            query=message_text,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=f"telegram_{user_id}_{int(time.time())}"
        )

        # 3. UPDATED: Handle search-triggered results with videos
        search_triggered = result.get("search_triggered", False)
        action_taken = result.get("action_taken", "unknown")

        if search_triggered:
            # SEARCH WAS TRIGGERED - Send video based on search type

            # Determine search type from the conversation context or action
            conversation_context = result.get("conversation_context", message_text)

            # Simple heuristic to determine search type
            search_type = determine_search_type(conversation_context, action_taken)

            # Send AI-generated video message
            processing_msg = send_search_message_with_video(chat_id, conversation_context, search_type)

            # Give the search a moment to start, then clean up the video message
            await asyncio.sleep(1)

            if processing_msg:
                try:
                    bot.delete_message(chat_id, processing_msg.message_id)
                except Exception:
                    pass  # Message might already be deleted

        # 4. TELEGRAM CONCERN: Send the AI response (as before)
        ai_response = result.get("ai_response") or result.get("formatted_message")

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
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass

        # Send error response
        bot.send_message(
            chat_id,
            "I'm having a bit of trouble right now. Could you try asking again in a moment?",
            parse_mode='HTML'
        )


def determine_search_type(conversation_context: str, action_taken: str) -> str:
    """
    Determine if this is a city-wide or location-based search

    Args:
        conversation_context: The accumulated conversation context
        action_taken: The action taken by AI chat layer

    Returns:
        "city_wide" or "location_based"
    """
    context_lower = conversation_context.lower()

    # Location-based indicators
    location_indicators = [
        'near me', 'nearby', 'around here', 'in this area', 'close to', 
        'vicinity', 'neighborhood', 'local', 'walking distance'
    ]

    # City-wide indicators
    city_indicators = [
        'best in', 'top restaurants', 'finest', 'must-visit', 'famous',
        'recommended in', 'popular in', 'well-known'
    ]

    # Check for location-based search
    if any(indicator in context_lower for indicator in location_indicators):
        return "location_based"

    # Check for city-wide search 
    if any(indicator in context_lower for indicator in city_indicators):
        return "city_wide"

    # Check action type
    if "location" in action_taken.lower():
        return "location_based"
    elif "city" in action_taken.lower():
        return "city_wide"

    # Default to city_wide for general searches
    return "city_wide"


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
def handle_gps_location(message):
    """Handle GPS location sharing"""
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id

        latitude = message.location.latitude
        longitude = message.location.longitude

        logger.info(f"📍 Received GPS location from user {user_id}: ({latitude:.4f}, {longitude:.4f})")

        # Process through AI Chat Layer with GPS coordinates
        asyncio.run(process_user_message(
            user_id, chat_id, 
            "restaurants near my current location",
            gps_coordinates=(latitude, longitude),
            message_type="location"
        ))

    except Exception as e:
        logger.error(f"Error handling GPS location: {e}")
        bot.reply_to(
            message,
            "😔 I had trouble processing your location. Could you try again?",
            parse_mode='HTML')


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages - Convert to text and process"""
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id

        logger.info(f"🎤 Received voice message from user {user_id}")

        # Check if voice handler is available
        if voice_handler is None:
            bot.reply_to(
                message,
                "😔 Voice processing is not available right now. Please send a text message.",
                parse_mode='HTML')
            return

        # Show typing indicator
        bot.send_chat_action(chat_id, 'typing')

        # Transcribe voice message
        transcribed_text = voice_handler.process_voice_message(bot, message.voice)

        if not transcribed_text:
            bot.send_message(
                chat_id,
                "😔 I couldn't understand your voice message. Could you try again or send a text message?",
                parse_mode='HTML')
            return

        logger.info(f"✅ Voice transcribed for user {user_id}: '{transcribed_text[:100]}...'")

        # Process transcribed text through AI Chat Layer
        asyncio.run(process_user_message(user_id, chat_id, transcribed_text, message_type="voice"))

    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        bot.reply_to(
            message,
            "😔 Sorry, I had trouble processing your voice message. Could you try again?",
            parse_mode='HTML')


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
    global ai_message_generator

    logger.info("🤖 Starting Telegram bot with Enhanced AI Chat Layer integration")
    logger.info("✅ Enhanced unified agent with AI Chat Layer initialized")
    logger.info("🎯 All messages processed through AI Chat Layer for intelligent conversation flow")
    logger.info("🔧 UPDATED: AI-generated search messages with videos")

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