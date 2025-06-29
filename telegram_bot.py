# telegram_bot.py - AI-Powered Restaurant Bot
import telebot
import logging
import time
import threading
from telebot import types
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from main import setup_orchestrator
import json

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
    model=config.OPENAI_MODEL,  # GPT-4o as specified
    temperature=0.3
)

# Simple conversation history storage (last 5 messages per user)
user_conversations = {}

# Welcome message
WELCOME_MESSAGE = (
    """🍸 Hello! I'm an AI assistant Restaurant Babe, and I know all about the most delicious and trendy restaurants, cafes, bakeries, bars, and coffee shops around the world.\n\n"""
    """Tell me what you are looking for. For example:\n"""
    """<i>What new restaurants have recently opened in Lisbon?</i>\n"""
    """<i>Local residents' favorite cevicherias in Lima</i>\n"""
    """<i>Where can I find the most delicious plov in Tashkent?</i>\n"""
    """<i>Recommend places with brunch and specialty coffee in Barcelona.</i>\n"""
    """<i>Best cocktail bars in Paris's Marais district</i>\n\n"""
    """I will check with my restaurant critic friends and provide the best recommendations. This might take a couple of minutes because I search very carefully and thoroughly verify the results. But there won't be any random places in my list.\n"""
    """Shall we begin?"""
)

# AI Conversation Prompt
CONVERSATION_PROMPT = """
You are Restaurant Babe, an expert AI assistant for restaurant recommendations worldwide. You help users find amazing restaurants, cafes, bars, bakeries, and coffee shops.

CONVERSATION HISTORY (last few messages):
{conversation_history}

CURRENT USER MESSAGE: {user_message}

YOUR TASK:
Analyze the conversation and decide what to do next. You need TWO pieces of information to search:
1. LOCATION (city/neighborhood/area)
2. DINING PREFERENCE (cuisine type, restaurant style, or specific request like "brunch", "cocktails", "romantic dinner")

DECISION RULES:
- If you have BOTH location AND dining preference → Action: "SEARCH"
- If missing one or both pieces → Action: "CLARIFY" 
- If completely off-topic → Action: "REDIRECT"

RESPONSE FORMAT:
Return JSON only:
{{
    "action": "SEARCH" | "CLARIFY" | "REDIRECT",
    "search_query": "complete restaurant search query (only if action is SEARCH)",
    "bot_response": "what to say to the user",
    "reasoning": "brief explanation of your decision"
}}

EXAMPLES:

User: "ramen in tokyo"
→ {{"action": "SEARCH", "search_query": "best ramen restaurants in Tokyo", "bot_response": "Perfect! Let me find the best ramen places in Tokyo for you.", "reasoning": "Have both location and preference"}}

User: "I want something romantic"
→ {{"action": "CLARIFY", "bot_response": "Romantic sounds wonderful! Which city or area would you like me to search for romantic restaurants?", "reasoning": "Missing location information"}}

User: "How's the weather?"
→ {{"action": "REDIRECT", "bot_response": "I specialize in restaurant recommendations! Tell me what type of food or dining experience you're looking for, and in which city.", "reasoning": "Off-topic question"}}
"""

def add_to_conversation(user_id, message, is_user=True):
    """Add message to user conversation history (keeps last 5 messages)"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    role = "User" if is_user else "Bot"
    user_conversations[user_id].append(f"{role}: {message}")

    # Keep only last 5 messages
    if len(user_conversations[user_id]) > 5:
        user_conversations[user_id] = user_conversations[user_id][-5:]

def format_conversation_history(user_id):
    """Format conversation history for AI prompt"""
    if user_id not in user_conversations:
        return "No previous conversation"

    history = user_conversations[user_id]
    return "\n".join(history)

# Initialize orchestrator (will be set up when first needed)
orchestrator = None

def get_orchestrator():
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = setup_orchestrator()
    return orchestrator

# COMMAND HANDLERS MUST BE DEFINED FIRST

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Handle /start and /help commands"""
    try:
        user_id = message.from_user.id

        # Clear conversation history on start
        if user_id in user_conversations:
            del user_conversations[user_id]

        bot.reply_to(
            message, 
            WELCOME_MESSAGE, 
            parse_mode='HTML'
        )
        logger.info(f"Sent welcome message to user {user_id}")
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")
        bot.reply_to(message, "Hello! I'm Restaurant Babe, ready to help you find amazing restaurants!")

# ADMIN COMMAND: /test_scrape - DEFINED DIRECTLY HERE
@bot.message_handler(commands=['test_scrape'])
def handle_test_scrape(message):
    """Handle /test_scrape command - scraping process test"""

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
        f"🧪 <b>Starting scraping process test...</b>\n\n"
        f"📝 Query: <code>{restaurant_query}</code>\n\n"
        "This will run the complete pipeline:\n"
        "1️⃣ Query analysis\n"
        "2️⃣ Web search\n"
        "3️⃣ Intelligent scraping\n"
        "4️⃣ Content analysis\n\n"
        "⏱ Please wait 2-3 minutes...",
        parse_mode='HTML'
    )

    # Run test in background
    def run_test():
        try:
            # Import and run the scraping test
            from scrape_test import ScrapeTest
            import asyncio

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
                bot.send_message(
                    admin_chat_id,
                    f"❌ Scraping test failed for '{restaurant_query}': {str(e)}"
                )
            except:
                pass

    thread = threading.Thread(target=run_test, daemon=True)
    thread.start()

# NOTE: /test_search command will be added via function call in main()

def perform_restaurant_search(search_query, chat_id, user_id):
    """Perform restaurant search using orchestrator"""
    try:
        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!",
            parse_mode='HTML'
        )

        # Get orchestrator and perform search
        orchestrator_instance = get_orchestrator()

        # This is where the actual search happens
        result = orchestrator_instance.process_query(search_query)

        # Format for Telegram (ensure proper formatting)
        telegram_text = result.get('telegram_formatted_text', 'Sorry, no recommendations found.')

        # Delete the processing message
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass  # Don't worry if we can't delete it

        # Send the results
        bot.send_message(
            chat_id,
            telegram_text,
            parse_mode='HTML',
            disable_web_page_preview=True
        )

        logger.info(f"Successfully sent restaurant recommendations to user {user_id}")

        # Add search completion to conversation history
        add_to_conversation(user_id, "Restaurant recommendations delivered!", is_user=False)

    except Exception as e:
        logger.error(f"Error in restaurant search process: {e}")
        try:
            # Delete processing message if it exists
            if 'processing_msg' in locals():
                bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass

        bot.send_message(
            chat_id,
            "😔 Sorry, I encountered an error while searching for restaurants. Please try again with a different query!",
            parse_mode='HTML'
        )

# IMPORTANT: This must be the LAST message handler (catch-all for non-command messages)
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all text messages with AI conversation management"""
    try:
        user_id = message.from_user.id
        user_message = message.text.strip()

        logger.info(f"Received message from user {user_id}: {user_message}")

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
            logger.error(f"Raw content: {content}")
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

    # Initialize the orchestrator
    orchestrator_instance = get_orchestrator()

    # Add admin test commands
    try:
        from search_test import add_search_test_command
        add_search_test_command(bot, config, orchestrator_instance)
        logger.info("Added /test_search command")
    except ImportError as e:
        logger.error(f"Failed to add search test command: {e}")

    # Log that admin commands are available
    logger.info("Admin commands available: /test_scrape, /test_search")

    # Start polling with better error handling
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
                logger.error("Another bot instance is running. Waiting 30 seconds before retry...")
                time.sleep(30)
                continue
            else:
                logger.error(f"Telegram API error: {e}")
                break
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Restarting in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    main()