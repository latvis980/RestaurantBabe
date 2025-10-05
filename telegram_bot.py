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
# UTILITY FUNCTIONS
# ============================================================================

def fix_telegram_html(text: str) -> str:
    """Fix HTML formatting for Telegram"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def create_cancel_event(user_id: int, chat_id: int) -> Event:
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

def determine_search_type(conversation_context: str, action_taken: str) -> str:
    """
    Determine search type based on conversation context and AI action

    Args:
        conversation_context: The accumulated conversation context
        action_taken: The action taken by AI Chat Layer

    Returns:
        "city_wide" or "location_based"
    """
    try:
        # First check the action taken by AI Chat Layer
        if action_taken == "trigger_city_search":
            return "city_wide"
        elif action_taken == "trigger_location_search":
            return "location_based"

        # Fallback: analyze conversation context for location indicators
        context_lower = conversation_context.lower()

        # Location-based indicators
        location_indicators = [
            "around", "near", "close to", "in the area", "nearby", 
            "chiado", "bairro alto", "príncipe real", "cais do sodré",
            "here", "vicinity", "my location", "current location"
        ]

        # City-wide indicators  
        city_indicators = [
            "best", "top", "favorite", "recommended", "must try",
            "in lisbon", "in porto", "in madrid", "in barcelona",
            "city", "downtown", "famous"
        ]

        # Check for location-based search
        if any(indicator in context_lower for indicator in location_indicators):
            logger.info(f"🗺️ Detected location-based search from context: {context_lower[:100]}")
            return "location_based"

        # Check for city-wide search  
        if any(indicator in context_lower for indicator in city_indicators):
            logger.info(f"🏙️ Detected city-wide search from context: {context_lower[:100]}")
            return "city_wide"

        # Default fallback based on length and specificity
        if len(context_lower.split()) > 8:
            # Longer, more specific queries tend to be location-based
            return "location_based"
        else:
            # Shorter queries tend to be city-wide
            return "city_wide"

    except Exception as e:
        logger.warning(f"Error determining search type: {e}")
        return "city_wide"  # Safe fallback

def generate_search_message(search_query: str, search_type: str) -> str:
    """Generate AI-powered search message"""
    global ai_message_generator

    if not ai_message_generator:
        # Fallback to static messages
        if search_type == "city_wide":
            return ("<b>I'm searching for the best restaurants for you.</b>\n\n"
                    "This might take a minute while I check my curated collection and consult with my foodie network.")
        else:  # location_based
            return ("<b>Great! I'm searching for amazing restaurants in that area.</b>\n\n"
                    "Give me a moment to check my local guides and reach out to my contacts in the vicinity.")

    try:
        # Create AI prompt for generating search message
        prompt = f"""Generate a short, enthusiastic search message for a restaurant bot. 

Search type: {search_type}
User query: {search_query}

Requirements:
- 1-2 sentences max
- Enthusiastic but not over the top
- Mention that it might take a moment
- Use HTML formatting with <b> tags
- Match the personality of a knowledgeable foodie assistant

Example:
<b>Perfect! I'm searching for amazing sushi spots in Chiado.</b>

Let me check my local network and curated recommendations."""

        response = ai_message_generator.invoke([HumanMessage(content=prompt)])
        ai_message = response.content.strip()

        # Ensure proper HTML formatting
        if not ai_message.startswith('<b>'):
            ai_message = f"<b>{ai_message}</b>"

        return ai_message

    except Exception as e:
        logger.warning(f"AI message generation failed: {e}")
        # Fallback to static messages
        if search_type == "city_wide":
            return ("<b>I'm searching for the best restaurants for you.</b>\n\n"
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


def split_message_for_telegram(message: str, max_length: int = 4096) -> List[str]:
    """Split long messages for Telegram's character limit"""
    if len(message) <= max_length:
        return [message]

    chunks = []
    current_chunk = ""

    # Try to split by paragraphs first
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

    # If any chunk is still too long, split by sentences
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
# TELEGRAM BOT HANDLERS
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

    # Cancel any active search
    event = active_searches.get(user_id)
    if event:
        event.set()
        cleanup_search(user_id)
        bot.send_message(
            chat_id,
            "🛑 Search cancelled! What else can I help you find?",
            parse_mode='HTML'
        )
    else:
        bot.send_message(
            chat_id,
            "No active search to cancel. What are you looking for?",
            parse_mode='HTML'
        )

@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    """Handle text messages"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_text = message.text

    logger.info(f"📝 Text message from user {user_id}: '{message_text[:50]}...'")

    # Process through enhanced AI Chat Layer
    asyncio.run(process_user_message(
        user_id=user_id,
        chat_id=chat_id,
        message_text=message_text,
        message_type="text"
    ))

@bot.message_handler(content_types=['location'])
def handle_location_message(message):
    """Handle location messages"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    latitude = message.location.latitude
    longitude = message.location.longitude

    logger.info(f"📍 Location message from user {user_id}: {latitude}, {longitude}")

    # Process location with AI Chat Layer
    asyncio.run(process_user_message(
        user_id=user_id,
        chat_id=chat_id,
        message_text="[USER SHARED LOCATION]",
        gps_coordinates=(latitude, longitude),
        message_type="location"
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

        # Transcribe voice message
        transcription = voice_handler.transcribe_voice(downloaded_file)

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