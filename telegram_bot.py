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

# Initialize orchestrator (will be set up when first needed)
orchestrator = None

def get_orchestrator():
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = setup_orchestrator()
    return orchestrator

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
2. TYPE/STYLE (cuisine, restaurant type, specific dish, occasion, etc.)

RESPONSE RULES:
- If you have BOTH location and type/style from current or recent messages → ACTION: "SEARCH"
- If you're missing either location or type/style → ACTION: "CLARIFY"
- If user is just greeting/chatting → ACTION: "CHAT"
- If query is too vague → ACTION: "CLARIFY"

CLARIFY EXAMPLES:
- "What city are you interested in?"
- "What type of cuisine or dining experience are you looking for?"
- "Are you looking for a specific dish or type of restaurant?"

SEARCH EXAMPLES (when you have both location and type):
- User says "best sushi in Lisbon" → you have location (Lisbon) and type (sushi) → SEARCH
- User says "romantic dinner" but no location → CLARIFY for location
- User says "restaurants in Madrid" but no type → CLARIFY for type

Respond in JSON format:
{
  "action": "SEARCH|CLARIFY|CHAT",
  "bot_response": "Your friendly response here",
  "search_query": "only if action is SEARCH - the exact query to search for restaurants"
}

Be warm, enthusiastic, and helpful. Keep responses concise but engaging.
"""

def add_to_conversation(user_id: int, message: str, is_user: bool = True):
    """Add message to user's conversation history"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    user_conversations[user_id].append({
        "message": message,
        "is_user": is_user,
        "timestamp": time.time()
    })

    # Keep only last 5 messages
    user_conversations[user_id] = user_conversations[user_id][-5:]

def format_conversation_history(user_id: int) -> str:
    """Format conversation history for AI prompt"""
    if user_id not in user_conversations:
        return "No previous conversation."

    history_lines = []
    for msg in user_conversations[user_id]:
        speaker = "User" if msg["is_user"] else "Restaurant Babe"
        history_lines.append(f"{speaker}: {msg['message']}")

    return "\n".join(history_lines)

# Start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Send welcome message"""
    logger.info(f"New user started bot: {message.from_user.id}")

    # Clear conversation history for fresh start
    user_conversations.pop(message.from_user.id, None)

    bot.send_message(
        message.chat.id,
        WELCOME_MESSAGE,
        parse_mode='HTML'
    )

    add_to_conversation(message.from_user.id, "Started conversation", is_user=False)

def perform_restaurant_search(user_id: int, chat_id: int, search_query: str):
    """Perform restaurant search in background thread"""
    try:
        logger.info(f"Starting restaurant search for user {user_id}: {search_query}")

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            "🔍 Searching for the best restaurants...\nThis might take a moment while I check with restaurant critics and food experts.",
            parse_mode='HTML'
        )

        # Get orchestrator and run search
        orch = get_orchestrator()

        # Call orchestrator with proper parameters
        result = orch.process_query(search_query)

        logger.info(f"Search completed for user {user_id}")

        # Extract results
        recommendations = result.get('final_recommendations', {})
        html_result = result.get('telegram_html', '')

        # Check if we got any restaurants
        main_list = recommendations.get('main_list', [])
        if not main_list or len(main_list) == 0:
            # Log detailed debug info
            logger.warning(f"No restaurants found for query: {search_query}")
            logger.warning(f"Full result keys: {list(result.keys())}")
            logger.warning(f"Recommendations structure: {recommendations}")

            bot.edit_message_text(
                "😔 I couldn't find any restaurants matching your search. This could be because:\n\n"
                "• The location might not have many online reviews\n"
                "• Try a more general search (e.g., 'restaurants in [city]')\n"
                "• Some specific cuisines might be limited in certain areas\n\n"
                "Would you like to try a different search?",
                chat_id=chat_id,
                message_id=processing_msg.message_id,
                parse_mode='HTML'
            )
            return

        # Delete the processing message
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except:
            pass  # Don't worry if we can't delete it

        # Send the results
        if html_result:
            telegram_text = html_result
        else:
            # Fallback formatting if no HTML result
            telegram_text = f"🍽️ <b>Restaurant Recommendations</b>\n\n"
            for i, restaurant in enumerate(main_list[:5], 1):
                name = restaurant.get('name', 'Unknown Restaurant')
                address = restaurant.get('address', 'Address not available')
                description = restaurant.get('description', 'No description available')
                telegram_text += f"<b>{i}. {name}</b>\n"
                telegram_text += f"📍 {address}\n"
                telegram_text += f"{description}\n\n"

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

# ADMIN COMMANDS - These will be processed BEFORE the general message handler
@bot.message_handler(commands=['test_scrape'])
def handle_test_scrape_command(message):
    """Handle /test_scrape command - ADMIN ONLY"""
    # Check if user is admin
    admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)
    if not admin_chat_id or str(message.chat.id) != str(admin_chat_id):
        bot.reply_to(message, "❌ This command is only available for admins.")
        return

    # Extract the search query from the command
    command_parts = message.text.split(' ', 1)
    if len(command_parts) < 2:
        bot.reply_to(message, 
            "❌ Please provide a search query.\n"
            "Format: `/test_scrape your search query here`\n"
            "Example: `/test_scrape best ramen restaurants in Tokyo`", 
            parse_mode='Markdown')
        return

    search_query = command_parts[1].strip()

    # Send initial response
    status_message = bot.reply_to(message, 
        f"🔍 **Testing scraping process**\n"
        f"Query: `{search_query}`\n"
        f"Starting search and scraping...", 
        parse_mode='Markdown')

    def run_test_scrape():
        """Run test scrape in background thread"""
        try:
            from langchain_core.tracers.context import tracing_v2_enabled
            import tempfile
            from datetime import datetime

            # Run the scraping process
            with tracing_v2_enabled(project_name="restaurant-recommender-test"):
                logger.info(f"Starting test scraping for query: {search_query}")

                # Update status
                bot.edit_message_text(
                    f"🔍 **Testing scraping process**\n"
                    f"Query: `{search_query}`\n"
                    f"⏳ Searching for sources...", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                # Get orchestrator and run search
                orch = get_orchestrator()
                search_agent = orch.search_agent
                search_results = search_agent.search(search_query)

                # Update status
                bot.edit_message_text(
                    f"🔍 **Testing scraping process**\n"
                    f"Query: `{search_query}`\n"
                    f"✅ Found {len(search_results.get('results', []))} sources\n"
                    f"⏳ Scraping content...", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                # Run scraping
                scraper = orch.scraper
                scraping_results = scraper.scrape_search_results(search_results.get('results', []))

                # Prepare dump data
                dump_data = {
                    'metadata': {
                        'query': search_query,
                        'timestamp': datetime.now().isoformat(),
                        'admin_chat_id': message.chat.id,
                        'total_sources_found': len(search_results.get('results', [])),
                        'successfully_scraped': len([r for r in scraping_results if r.get('scraped_content')]),
                        'failed_scrapes': len([r for r in scraping_results if not r.get('scraped_content')])
                    },
                    'raw_search_results': search_results,
                    'scraped_data': scraping_results
                }

                # Update final status
                successful_scrapes = len([r for r in scraping_results if r.get('scraped_content')])
                bot.edit_message_text(
                    f"🔍 **Testing scraping process**\n"
                    f"Query: `{search_query}`\n"
                    f"✅ Found {len(search_results.get('results', []))} sources\n"
                    f"✅ Successfully scraped: {successful_scrapes}\n"
                    f"📄 Preparing dump files...", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                # Create temporary file for the dump
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"scraping_test_{timestamp}.json"

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as json_file:
                    json.dump(dump_data, json_file, indent=2, ensure_ascii=False)
                    json_temp_path = json_file.name

                # Send the file
                with open(json_temp_path, 'rb') as json_file:
                    bot.send_document(
                        message.chat.id, 
                        json_file, 
                        caption=f"📄 **Complete scraping test results**\n"
                                f"Query: `{search_query}`\n"
                                f"Results: {successful_scrapes}/{len(search_results.get('results', []))} successful",
                        parse_mode='Markdown',
                        visible_file_name=json_filename
                    )

                # Clean up temporary file
                import os
                os.unlink(json_temp_path)

                # Final success message
                bot.edit_message_text(
                    f"✅ **Scraping test completed!**\n"
                    f"Query: `{search_query}`\n"
                    f"📊 Results: {successful_scrapes}/{len(search_results.get('results', []))} successful\n"
                    f"📄 File sent above ⬆️", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                logger.info(f"Test scraping completed successfully for query: {search_query}")

        except Exception as e:
            logger.error(f"Error in test scraping: {e}")
            bot.edit_message_text(
                f"❌ **Scraping test failed**\n"
                f"Query: `{search_query}`\n"
                f"Error: `{str(e)}`", 
                chat_id=status_message.chat.id,
                message_id=status_message.message_id,
                parse_mode='Markdown')

    # Run in background thread
    thread = threading.Thread(target=run_test_scrape, daemon=True)
    thread.start()

@bot.message_handler(commands=['debug_query'])
def handle_debug_query(message):
    """Handle /debug_query command - ADMIN ONLY"""
    user_id = message.from_user.id
    admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

    # Check if user is admin
    if not admin_chat_id or str(user_id) != str(admin_chat_id):
        bot.reply_to(message, "❌ This command is only available to administrators.")
        return

    # Parse command arguments
    command_text = message.text

    # Extract query after the command
    if len(command_text.split(None, 1)) < 2:
        help_text = (
            "🧠 <b>Intelligent Scraper Pipeline Debug Command</b>\n\n"
            "<b>Usage:</b>\n"
            "<code>/debug_query [your restaurant query]</code>\n\n"
            "<b>Examples:</b>\n"
            "<code>/debug_query best cevicherias in Lima</code>\n"
            "<code>/debug_query romantic restaurants in Paris</code>\n"
            "<code>/debug_query family-friendly pizza in Rome</code>\n\n"
            "This will run the complete intelligent scraper pipeline up to the list_analyzer stage "
            "and show you exactly what content gets passed to the AI for analysis.\n\n"
            "📊 <b>Features:</b>\n"
            "• Shows AI strategy analysis for each URL\n"
            "• Displays cost savings vs Firecrawl-only approach\n"
            "• Reports scraping method distribution\n"
            "• Shows domain intelligence cache\n"
            "• Tracks strategy effectiveness"
        )
        bot.reply_to(message, help_text, parse_mode='HTML')
        return

    # Extract the query
    user_query = command_text.split(None, 1)[1].strip()

    if not user_query:
        bot.reply_to(message, "❌ Please provide a restaurant query to debug.")
        return

    # Send confirmation and start debug
    bot.reply_to(
        message, 
        f"🧠 Starting intelligent scraper pipeline debug for query:\n<code>{user_query}</code>\n\n"
        "This will run the complete search and intelligent scraping pipeline. "
        "You'll receive a detailed report showing:\n\n"
        "• 🤖 AI analysis decisions for each URL\n"
        "• 📊 Strategy distribution and cost savings\n"
        "• 🎯 Cache hits and domain learning\n"
        "• 📝 Exact content passed to list_analyzer\n\n"
        "⏱ This may take 2-3 minutes...",
        parse_mode='HTML'
    )

    # Run debug in background thread
    def run_debug():
        try:
            # Import the debug handler here to avoid circular imports
            from debug_query_command import DebugQueryCommand

            # Create debug handler
            debug_handler = DebugQueryCommand(config, get_orchestrator())

            # Run the async debug
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            debug_path = loop.run_until_complete(
                debug_handler.debug_query_pipeline(user_query, bot)
            )

            loop.close()

            logger.info(f"Intelligent scraper query debug completed. Report saved to: {debug_path}")

        except Exception as e:
            logger.error(f"Error in intelligent scraper query debug: {e}")
            try:
                bot.send_message(
                    admin_chat_id,
                    f"❌ Intelligent scraper debug failed for '{user_query}': {str(e)}"
                )
            except:
                pass

    thread = threading.Thread(target=run_debug, daemon=True)
    thread.start()

# GENERAL MESSAGE HANDLER - This should be LAST to catch all non-command messages
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

        # Add bot response to conversation history
        add_to_conversation(user_id, bot_response, is_user=False)

        if action == "SEARCH":
            # Send the bot response first
            bot.send_message(message.chat.id, bot_response, parse_mode='HTML')

            # Then start restaurant search in background
            search_query = ai_decision.get("search_query", user_message)

            # Run search in background thread to avoid blocking
            search_thread = threading.Thread(
                target=perform_restaurant_search,
                args=(user_id, message.chat.id, search_query),
                daemon=True
            )
            search_thread.start()

        else:
            # Just send the response for CLARIFY or CHAT actions
            bot.send_message(message.chat.id, bot_response, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.send_message(
            message.chat.id,
            "😔 Sorry, I encountered an error. Please try again!",
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