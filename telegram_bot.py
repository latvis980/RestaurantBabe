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
→ {{"action": "SEARCH", "search_query": "best ramen restaurants in Tokyo", "bot_response": "Perfect! I'll search for the best ramen restaurants in Tokyo. Let me check with my critic friends - this will take a couple of minutes.", "reasoning": "Have both location (Tokyo) and preference (ramen)"}}

User: "I want Italian food"
→ {{"action": "CLARIFY", "search_query": "", "bot_response": "Great choice! Italian cuisine is amazing. Which city are you looking for Italian restaurants in?", "reasoning": "Have preference (Italian) but missing location"}}

User: "Paris"
→ {{"action": "CLARIFY", "search_query": "", "bot_response": "Ah, Paris! Wonderful city for dining. What type of restaurants are you interested in? Perhaps traditional French, modern bistros, international cuisine, or something specific?", "reasoning": "Have location (Paris) but missing dining preference"}}

User: "what's the weather like?"
→ {{"action": "REDIRECT", "search_query": "", "bot_response": "I'm your restaurant expert! I can help you find amazing places to eat and drink around the world. What city are you interested in dining in?", "reasoning": "Off-topic, not about restaurants"}}

CONVERSATION STYLE:
- Be warm, enthusiastic, and knowledgeable about food
- Reference previous messages naturally 
- Once you have location + preference, always choose "SEARCH"
- Keep responses concise but friendly
- Show expertise about restaurants and dining
"""

def get_conversation_history(user_id):
    """Get recent conversation history for user"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    return user_conversations[user_id]

def add_to_conversation(user_id, message, is_user=True):
    """Add message to conversation history"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    # Keep only last 8 messages (4 exchanges)
    if len(user_conversations[user_id]) >= 8:
        user_conversations[user_id] = user_conversations[user_id][-6:]

    sender = "User" if is_user else "Bot"
    user_conversations[user_id].append(f"{sender}: {message}")

def format_conversation_history(user_id):
    """Format conversation history for AI prompt"""
    history = get_conversation_history(user_id)
    if not history:
        return "No previous conversation."
    return "\n".join(history)

# Initialize orchestrator (will be set up when first needed)
orchestrator = None

def get_orchestrator():
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = setup_orchestrator()
    return orchestrator

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
        search_query = ai_decision.get("search_query", "")
        reasoning = ai_decision.get("reasoning", "")

        logger.info(f"AI Decision - Action: {action}, Reasoning: {reasoning}")

        # Add bot response to conversation history
        add_to_conversation(user_id, bot_response, is_user=False)

        if action == "SEARCH" and search_query:
            # Send immediate response
            bot.reply_to(message, bot_response, parse_mode='HTML')

            # Process restaurant search in background
            threading.Thread(
                target=process_restaurant_search,
                args=(message, search_query, user_id),
                daemon=True
            ).start()

        else:
            # Send clarification or redirect response
            bot.reply_to(message, bot_response, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.reply_to(
            message, 
            "Sorry, I encountered an error. Please try asking about restaurants in a specific city!"
        )

def process_restaurant_search(message, search_query, user_id):
    """Process restaurant search request in background"""
    try:
        chat_id = message.chat.id

        logger.info(f"Starting restaurant search for user {user_id}: {search_query}")

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!",
            parse_mode='HTML'
        )

        # Process the query through orchestrator
        orch = get_orchestrator()
        result = orch.process_query(search_query)

        # Get the formatted response
        telegram_text = result.get("telegram_text", "Sorry, I couldn't find any restaurants for your request.")

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

def main():
    """Main function to start the bot"""
    logger.info("Starting AI-Powered Restaurant Babe Telegram Bot...")

    # Verify bot token works
    try:
        bot_info = bot.get_me()
        logger.info(f"Bot started successfully: @{bot_info.username}")
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
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
                logger.error("Another bot instance is running. Waiting 30 seconds...")
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