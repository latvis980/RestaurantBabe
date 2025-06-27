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
You are an AI assistant that helps users with restaurant search requests. Your job is to analyze incoming messages and decide what to do, taking into account the conversation history.

ANALYZE the user's message and conversation context, then respond with a JSON object containing:
{{
    "action": "process" | "ask_clarification" | "remind_purpose" | "update_session",
    "response": "your response message to the user",
    "reasoning": "brief explanation of your decision",
    "session_updates": {{"location": "city_name", "cuisine_type": "cuisine", "dining_style": "style", "other_preferences": "preferences"}}
}}

ACTIONS:
1. "process" - Have both location AND restaurant preferences. Ready to search.
2. "ask_clarification" - Missing either location or restaurant type, ask for the missing piece
3. "remind_purpose" - Off-topic or asking about specific single restaurants  
4. "update_session" - User provided new info, update session and ask for remaining details

CONVERSATION CONTEXT:
- Previous location mentioned: {previous_location}
- Previous preferences mentioned: {previous_preferences}
- Previous cuisine type: {previous_cuisine}
- Previous dining style: {previous_dining_style}

GUIDELINES:
- If user already provided location in conversation, don't ask for it again
- If user already provided restaurant preferences, build on them rather than ignoring them
- Only ask for ONE missing piece at a time
- Be conversational and reference what they already told you
- For "process": Must have BOTH location AND some restaurant preference (cuisine, style, type, etc.)

EXAMPLES:

Context: location="", preferences="", cuisine="", style=""
User: "ramen restaurants"
→ {{"action": "ask_clarification", "response": "Great choice! Ramen is amazing. Which city are you looking for ramen restaurants in?", "session_updates": {{"cuisine_type": "ramen"}}}}

Context: location="Lisbon", preferences="", cuisine="", style=""  
User: "ramen restaurants"
→ {{"action": "process", "response": "Perfect! I'll search for the best ramen restaurants in Lisbon. Let me check with my critic friends - this will take a couple of minutes.", "session_updates": {{"cuisine_type": "ramen"}}}}

Context: location="", preferences="", cuisine="", style=""
User: "restaurants in Paris"
→ {{"action": "ask_clarification", "response": "Paris has incredible dining! What type of cuisine or dining experience are you looking for? For example, traditional French bistros, fine dining, specific cuisines, or casual spots?", "session_updates": {{"location": "Paris"}}}}

Context: location="Paris", preferences="", cuisine="", style=""
User: "something romantic"
→ {{"action": "process", "response": "Wonderful! I'll find romantic restaurants in Paris for you. This will take a few minutes while I consult the best sources.", "session_updates": {{"dining_style": "romantic"}}}}

Context: location="Tokyo", preferences="local favorites", cuisine="", style=""
User: "actually, make it Kyoto instead"
→ {{"action": "update_session", "response": "Got it! So you're looking for local favorite restaurants in Kyoto. What type of cuisine or dining experience interests you most?", "session_updates": {{"location": "Kyoto"}}}}

Always be friendly and reference what the user has already shared with you.
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
        ("human", "{user_message}")
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

        logger.info(f"Analysis result for user {user_id}: action={action}, session={session}")

        if action == "process" or (action == "update_session" and session_has_enough_info(session)):
            # Build complete query from session
            complete_query = build_query_from_session(session)

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
        result = orch.process_query(complete_query, user_id=user_id)

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