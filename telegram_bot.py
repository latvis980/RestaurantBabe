# telegram_bot.py - AI-Powered Restaurant Bot - FIXED VERSION
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

# BASIC COMMAND HANDLERS - Define these FIRST

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

# FUNCTION TO ADD EXTERNAL COMMANDS
def setup_admin_commands():
    """Setup admin commands from external files"""
    commands_added = []

    # Add scrape test command
    try:
        from scrape_test import add_scrape_test_command
        add_scrape_test_command(bot, config, get_orchestrator())
        commands_added.append("/test_scrape")
        logger.info("✅ Added /test_scrape command from scrape_test.py")
    except ImportError as e:
        logger.error(f"❌ Failed to add scrape test command: {e}")
        logger.error("Make sure scrape_test.py exists and has no syntax errors")

    # Add search test command
    try:
        from search_test import add_search_test_command
        add_search_test_command(bot, config, get_orchestrator())
        commands_added.append("/test_search")
        logger.info("✅ Added /test_search command from search_test.py")
    except ImportError as e:
        logger.error(f"❌ Failed to add search test command: {e}")
        logger.error("Make sure search_test.py exists and has no syntax errors")

    return commands_added

# IMPORTANT: This must be the LAST message handler (catch-all for non-command messages)
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all text messages with AI conversation management"""

    # IMPORTANT: Skip if this is a command that should be handled by external handlers
    if message.text.startswith('/test_'):
        logger.warning(f"Command {message.text.split()[0]} not recognized - check if external command loaded properly")
        bot.reply_to(message, "❌ Command not recognized. Available admin commands: /test_scrape, /test_search")
        return

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

    # Setup external admin commands AFTER defining all message handlers
    logger.info("🔧 Setting up admin commands...")
    commands_added = setup_admin_commands()

    # Log available commands
    if commands_added:
        logger.info(f"🎯 Admin commands available: {', '.join(commands_added)}")
    else:
        logger.warning("⚠️ No admin test commands were loaded")

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