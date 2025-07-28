# telegram_bot.py - Updated with /cancel command functionality
import telebot
import logging
import time
import threading
import json
import asyncio
from threading import Event

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from utils.orchestrator_manager import get_orchestrator

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

# Welcome message (unchanged)
WELCOME_MESSAGE = (
    "🍸 Hello! I'm an AI assistant Restaurant Babe, and I know all about the most delicious and trendy restaurants, cafes, bakeries, bars, and coffee shops around the world.\n\n"
    "Tell me what you are looking for. For example:\n"
    "<i>What new restaurants have recently opened in Lisbon?</i>\n"
    "<i>Local residents' favorite cevicherias in Lima</i>\n"
    "<i>Where can I find the most delicious plov in Tashkent?</i>\n"
    "<i>Recommend places with brunch and specialty coffee in Barcelona.</i>\n"
    "<i>Best cocktail bars in Paris's Marais district</i>\n\n"
    "I will check with my restaurant critic friends and provide the best recommendations. This might take a couple of minutes because I search very carefully and thoroughly verify the results. But there won't be any random places in my list.\n\n"
    "💡 <b>Tip:</b> If you change your mind while I'm searching, just type /cancel to stop the current search.\n\n"
    "Shall we begin?"
)

# AI Conversation Prompt (unchanged)
CONVERSATION_PROMPT = """
You are Restaurant Babe, an expert AI assistant for restaurant recommendations worldwide. 

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

    bot.reply_to(message, WELCOME_MESSAGE, parse_mode='HTML')

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
            parse_mode='HTML'
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
        parse_mode='HTML'
    )

    # Clean up
    cleanup_search(user_id)

    # Add cancellation to conversation history
    add_to_conversation(user_id, "Search cancelled by user", is_user=False)

    logger.info(f"Successfully cancelled search for user {user_id} after {search_duration}s")

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

def perform_restaurant_search(search_query, chat_id, user_id):
    """Perform restaurant search using orchestrator with cancellation support"""
    cancel_event = None
    processing_msg = None

    try:
        # Create cancellation event for this search
        cancel_event = create_cancel_event(user_id, chat_id)

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!\n\n"
            "💡 Type /cancel if you want to stop the search.",
            parse_mode='HTML'
        )

        logger.info(f"Started restaurant search for user {user_id}: {search_query}")

        # Check for cancellation before starting the actual search
        if is_search_cancelled(user_id):
            logger.info(f"Search cancelled before processing for user {user_id}")
            return

        # Get orchestrator using singleton pattern
        orchestrator_instance = get_orchestrator()

        # The orchestrator doesn't support cancellation directly, but we can check periodically
        # For now, we'll run the search and check cancellation afterward
        # In a future update, you could modify the orchestrator to accept a cancel_event

        result = orchestrator_instance.process_query(search_query)

        # Check if cancelled after processing
        if is_search_cancelled(user_id):
            logger.info(f"Search was cancelled during processing for user {user_id}")
            # Delete processing message if search was cancelled
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
            disable_web_page_preview=True
        )

        logger.info(f"Successfully sent restaurant recommendations to user {user_id}")

        # Add completion to conversation history
        add_to_conversation(user_id, "Restaurant recommendations delivered!", is_user=False)

    except Exception as e:
        logger.error(f"Error in restaurant search process: {e}")

        # Delete processing message if it exists
        try:
            if processing_msg:
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        # Only send error message if search wasn't cancelled
        if not is_search_cancelled(user_id):
            bot.send_message(
                chat_id,
                "😔 Sorry, I encountered an error while searching for restaurants. Please try again with a different query!",
                parse_mode='HTML'
            )
    finally:
        # Always clean up the search tracking
        cleanup_search(user_id)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all text messages with AI conversation management"""
    try:
        user_id = message.from_user.id
        user_message = message.text.strip()

        logger.info(f"Received message from user {user_id}: {user_message}")

        # Check if user has an active search
        if user_id in active_searches:
            bot.reply_to(
                message,
                "⏳ I'm currently searching for restaurants for you! Please wait for the results or type /cancel to stop the search.",
                parse_mode='HTML'
            )
            return

        # Add user message to conversation history
        add_to_conversation(user_id, user_message, is_user=True)

        # Send typing indicator
        bot.send_chat_action(message.chat.id, 'typing')

        # Create conversation analysis prompt
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", CONVERSATION_PROMPT),
            ("human", "Conversation history:\n{conversation_history}\n\nCurrent message: {user_message}")
        ])

        # Get AI decision
        conversation_chain = conversation_prompt | conversation_ai
        response = conversation_chain.invoke({
            "conversation_history": format_conversation_history(user_id),
            "user_message": user_message
        })

        # Parse AI response
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            ai_decision = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI decision: {e}")
            # Fallback response
            ai_decision = {
                "action": "CLARIFY",
                "bot_response": "I'd love to help you find restaurants! Could you tell me what city you're interested in and what type of dining you're looking for?"
            }

        action = ai_decision.get("action")
        bot_response = ai_decision.get("bot_response", "How can I help you find restaurants?")

        if action == "SEARCH":
            search_query = ai_decision.get("search_query")
            if search_query:
                # Add bot response to conversation before search
                add_to_conversation(user_id, bot_response, is_user=False)

                # Send the bot response
                bot.reply_to(message, bot_response, parse_mode='HTML')

                # Perform the search in background
                threading.Thread(
                    target=perform_restaurant_search,
                    args=(search_query, message.chat.id, user_id),
                    daemon=True
                ).start()
            else:
                # Fallback if no search query
                add_to_conversation(user_id, bot_response, is_user=False)
                bot.reply_to(message, bot_response, parse_mode='HTML')
        else:
            # CLARIFY or REDIRECT - just send the bot response
            add_to_conversation(user_id, bot_response, is_user=False)
            bot.reply_to(message, bot_response, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.reply_to(
            message, 
            "I'm having trouble understanding right now. Could you try asking again about restaurants in a specific city?",
            parse_mode='HTML'
        )

def main():
    """Main function to start the bot"""
    logger.info("Starting Restaurant Babe Telegram Bot...")

    # Verify bot token works
    try:
        bot_info = bot.get_me()
        logger.info(f"Bot started successfully: @{bot_info.username}")
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return

    # Verify orchestrator is available
    try:
        orchestrator_instance = get_orchestrator()
        logger.info("✅ Orchestrator singleton confirmed available")
        logger.info("🎯 Admin commands available: /test_scrape, /test_search")
        logger.info("🛑 Cancel command available: /cancel")
    except RuntimeError as e:
        logger.error(f"❌ Orchestrator not initialized: {e}")
        logger.error("Make sure main.py calls setup_orchestrator() before starting the bot")
        return

    # Start polling with error handling
    while True:
        try:
            logger.info("Starting bot polling...")
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
            logger.error(f"Unexpected bot error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()