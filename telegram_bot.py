# telegram_bot.py
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

# Initialize OpenAI for request analysis
request_analyzer = ChatOpenAI(
    model=config.OPENAI_MODEL,
    temperature=0.3
)

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

# Request analysis prompt
REQUEST_ANALYSIS_PROMPT = """
You are an AI assistant that helps users with restaurant search requests. Your job is to analyze incoming messages and decide what to do.

ANALYZE the user's message and respond with a JSON object containing:
{
    "action": "process" | "ask_clarification" | "remind_purpose",
    "response": "your response message to the user",
    "reasoning": "brief explanation of your decision"
}

ACTIONS:
1. "process" - The request has enough information (restaurant type + location). Ready to search.
2. "ask_clarification" - The request is restaurant-related but missing key info (usually location or specific preferences)
3. "remind_purpose" - The request is not restaurant-related or too narrow for our app

GUIDELINES:
- For "process": Must have both restaurant type/preference AND location
- For "ask_clarification": Restaurant-related but missing location or being too vague ("restaurants in general")
- For "remind_purpose": Off-topic, single restaurant queries, or non-food related

EXAMPLES:

User: "restaurants in Paris" 
→ {"action": "ask_clarification", "response": "I'd love to help you find great restaurants in Paris! Could you tell me what type of cuisine or dining experience you're looking for? For example, are you interested in fine dining, casual bistros, specific cuisines like French or international, or perhaps places with a particular atmosphere?"}

User: "best ramen in Tokyo"
→ {"action": "process", "response": "Perfect! I'll search for the best ramen restaurants in Tokyo. Let me check with my critic friends - this will take a couple of minutes."}

User: "what's the weather like?"
→ {"action": "remind_purpose", "response": "I'm Restaurant Babe, your specialized assistant for finding amazing restaurants, cafes, bars, and food spots around the world! I can't help with weather, but I'd love to recommend some great places to dine. What city are you in or planning to visit?"}

User: "tell me about Le Bernardin"
→ {"action": "remind_purpose", "response": "I specialize in discovering multiple restaurant options rather than providing details about specific establishments. Instead, I can help you find the best seafood restaurants in New York or fine dining options in your preferred city. What type of dining experience are you looking for and where?"}

Always be friendly, enthusiastic about food, and encourage the user to be more specific when needed.
"""

# Create the analysis chain
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", REQUEST_ANALYSIS_PROMPT),
    ("human", "{user_message}")
])
analysis_chain = analysis_prompt | request_analyzer

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
        bot.reply_to(
            message, 
            WELCOME_MESSAGE, 
            parse_mode='HTML'
        )
        logger.info(f"Sent welcome message to user {message.from_user.id}")
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")
        bot.reply_to(message, "Hello! I'm Restaurant Babe, ready to help you find amazing restaurants!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all text messages"""
    try:
        user_id = message.from_user.id
        user_message = message.text.strip()

        logger.info(f"Received message from user {user_id}: {user_message}")

        # Send typing indicator
        bot.send_chat_action(message.chat.id, 'typing')

        # Analyze the request using OpenAI
        response = analysis_chain.invoke({"user_message": user_message})

        # Parse the response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            analysis_result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis result: {e}")
            # Fallback to a generic response
            analysis_result = {
                "action": "ask_clarification",
                "response": "I'd love to help you find restaurants! Could you tell me what city you're interested in and what type of dining experience you're looking for?"
            }

        action = analysis_result.get("action")
        response_text = analysis_result.get("response", "I'm not sure how to help with that. Could you ask about restaurants in a specific city?")

        logger.info(f"Analysis result for user {user_id}: action={action}")

        if action == "process":
            # Send immediate response
            bot.reply_to(message, response_text, parse_mode='HTML')

            # Process the restaurant search in a separate thread
            threading.Thread(
                target=process_restaurant_search,
                args=(message, user_message),
                daemon=True
            ).start()

        else:
            # Send clarification or reminder
            bot.reply_to(message, response_text, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.reply_to(
            message, 
            "Sorry, I encountered an error. Please try asking about restaurants in a specific city!"
        )

def process_restaurant_search(message, user_query):
    """Process restaurant search request in background"""
    try:
        chat_id = message.chat.id
        user_id = message.from_user.id

        logger.info(f"Starting restaurant search for user {user_id}: {user_query}")

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!",
            parse_mode='HTML'
        )

        # Process the query through your existing orchestrator
        orch = get_orchestrator()
        result = orch.process_query(user_query, user_id=user_id)  

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
    logger.info("Starting Restaurant Babe Telegram Bot...")

    # Verify bot token works
    try:
        bot_info = bot.get_me()
        logger.info(f"Bot started successfully: @{bot_info.username}")
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return

    # Start polling
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot polling error: {e}")

if __name__ == "__main__":
    main()