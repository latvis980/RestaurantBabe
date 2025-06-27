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

# Session storage for user contexts
user_sessions = {}

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

# Request analysis prompt with session awareness
REQUEST_ANALYSIS_PROMPT = """
You are an AI assistant for restaurant search. Analyze the user's message and conversation context.

CONVERSATION CONTEXT:
- Previous location: {previous_location}
- Previous cuisine: {previous_cuisine}
- Previous dining style: {previous_dining_style}
- Previous preferences: {previous_preferences}

DECISION RULES:
1. If we have BOTH location AND any restaurant preference (cuisine/style) → "process" 
2. If missing location but have restaurant preference → ask for location only
3. If have location but missing restaurant preference → ask for cuisine/preference only
4. If completely off-topic → "remind_purpose"

Return JSON:
{{
    "action": "process" | "ask_clarification" | "remind_purpose",
    "response": "your response to user",
    "session_updates": {{"location": "", "cuisine_type": "", "dining_style": "", "other_preferences": ""}}
}}

EXAMPLES:

Context: location="", cuisine="ramen", style="", preferences=""
User: "Lisbon"
→ {{"action": "process", "response": "Perfect! I'll search for the best ramen restaurants in Lisbon. Let me check with my critic friends - this will take a couple of minutes.", "session_updates": {{"location": "Lisbon"}}}}

Context: location="Lisbon", cuisine="", style="", preferences=""
User: "ramen restaurants"  
→ {{"action": "process", "response": "Excellent! I'll find the best ramen restaurants in Lisbon for you. This will take a few minutes.", "session_updates": {{"cuisine_type": "ramen"}}}}

Context: location="Lisbon", cuisine="ramen", style="", preferences=""
User: "actually make it traditional Portuguese instead"
→ {{"action": "process", "response": "Great choice! I'll search for traditional Portuguese restaurants in Lisbon instead.", "session_updates": {{"cuisine_type": "traditional Portuguese"}}}}

Be conversational and reference what they already told you. Once you have location + any food preference, always choose "process".
"""

def get_user_session(user_id):
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "location": "",
            "cuisine_type": "",
            "dining_style": "",
            "other_preferences": "",
            "last_activity": time.time()
        }
    return user_sessions[user_id]

def update_user_session(user_id, updates):
    """Update user session with new information"""
    session = get_user_session(user_id)
    for key, value in updates.items():
        if value and value.strip():  # Only update if value is not empty
            session[key] = value.strip()
    session["last_activity"] = time.time()
    return session

def build_query_from_session(session):
    """Build a complete query from session data"""
    parts = []

    if session.get("cuisine_type"):
        parts.append(session["cuisine_type"])
    if session.get("dining_style"):
        parts.append(session["dining_style"])
    if session.get("other_preferences"):
        parts.append(session["other_preferences"])

    if parts and session.get("location"):
        query = f"{' '.join(parts)} restaurants in {session['location']}"
        return query

    return None

def session_has_enough_info(session):
    """Check if session has enough info to process query"""
    has_location = bool(session.get("location", "").strip())
    has_preferences = any([
        session.get("cuisine_type", "").strip(),
        session.get("dining_style", "").strip(), 
        session.get("other_preferences", "").strip()
    ])
    return has_location and has_preferences

# Create the analysis chain with session context
def create_analysis_chain():
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", REQUEST_ANALYSIS_PROMPT),
        ("human", "User message: {user_message}\n\nPrevious context:\nLocation: {previous_location}\nCuisine: {previous_cuisine}\nDining style: {previous_dining_style}\nOther preferences: {previous_preferences}")
    ])
    return analysis_prompt | request_analyzer

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
        # Clear user session on start
        user_id = message.from_user.id
        if user_id in user_sessions:
            del user_sessions[user_id]

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
    """Handle all text messages with session awareness"""
    try:
        user_id = message.from_user.id
        user_message = message.text.strip()

        logger.info(f"Received message from user {user_id}: {user_message}")

        # Get user session
        session = get_user_session(user_id)

        # Send typing indicator
        bot.send_chat_action(message.chat.id, 'typing')

        # Create analysis chain with session context
        analysis_chain = create_analysis_chain()

        # Analyze the request using OpenAI with session context
        response = analysis_chain.invoke({
            "user_message": user_message,
            "previous_location": session.get("location", ""),
            "previous_preferences": session.get("other_preferences", ""),
            "previous_cuisine": session.get("cuisine_type", ""),
            "previous_dining_style": session.get("dining_style", "")
        })

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
        session_updates = analysis_result.get("session_updates", {})

        # Update session with new information
        if session_updates:
            update_user_session(user_id, session_updates)
            session = get_user_session(user_id)  # Get updated session

        logger.info(f"Analysis result for user {user_id}: action={action}")
        logger.info(f"Session state: {session}")
        logger.info(f"Has enough info: {session_has_enough_info(session)}")

        if action == "process":
            # Build complete query from session
            complete_query = build_query_from_session(session)
            logger.info(f"Built query: {complete_query}")

            if complete_query:
                # Send immediate response
                bot.reply_to(message, response_text, parse_mode='HTML')

                # Process the restaurant search in a separate thread
                threading.Thread(
                    target=process_restaurant_search,
                    args=(message, complete_query, user_id),
                    daemon=True
                ).start()
            else:
                # Fallback if query building fails
                logger.error(f"Failed to build query from session: {session}")
                bot.reply_to(message, "I have some information but need a bit more. What type of restaurants are you looking for?", parse_mode='HTML')

        else:
            # Send clarification or reminder
            bot.reply_to(message, response_text, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.reply_to(
            message, 
            "Sorry, I encountered an error. Please try asking about restaurants in a specific city!"
        )

def process_restaurant_search(message, complete_query, user_id):
    """Process restaurant search request in background"""
    try:
        chat_id = message.chat.id

        logger.info(f"Starting restaurant search for user {user_id}: {complete_query}")

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 Searching for the best recommendations... This may take a few minutes as I consult with my critic friends!",
            parse_mode='HTML'
        )

        # Process the query through your existing orchestrator
        orch = get_orchestrator()
        result = orch.process_query(complete_query)

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

        # Clear user session after successful search
        if user_id in user_sessions:
            del user_sessions[user_id]

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
                time.sleep(30)  # Wait longer for other instance to stop
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