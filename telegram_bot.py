# telegram_bot.py
import telebot
import logging
import asyncio
from telebot.async_telebot import AsyncTeleBot
import time
import traceback
import os

import config
from agents.langchain_orchestrator import LangChainOrchestrator
from utils.async_utils import sync_to_async, wait_for_pending_tasks, track_async_task
from utils.database import initialize_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database
initialize_db(config)

# Initialize the bot
bot = AsyncTeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize the orchestrator
orchestrator = None

# Store user states for multi-step interactions
user_states = {}

# Dictionary to store standing preferences for users
user_preferences = {}

@bot.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    """Send a welcome message and help information"""
    help_text = (
        "🍽️ *Restaurant Recommendation Bot* 🍽️\n\n"
        "I can help you find great restaurants based on your preferences! Just tell me what you're looking for.\n\n"
        "*Examples:*\n"
        "- Where to eat in Paris with great wine selection?\n"
        "- Looking for affordable Italian restaurants in Rome\n"
        "- Best places for seafood in Tokyo\n\n"
        "*Commands:*\n"
        "/start or /help - Show this help message\n"
        "/settings - Manage your preferences\n"
        "/clear - Clear your current preferences\n"
        "/fetch_test URL - Test URL fetching functionality\n"
        "/scrape_test URL - Test URL scraping functionality\n\n"
        "Made with ❤️ by Restaurant Recommender"
    )
    await bot.send_message(message.chat.id, help_text, parse_mode="Markdown")

@bot.message_handler(commands=['settings'])
async def settings_command(message):
    """Manage user settings and preferences"""
    # Welcome message for settings
    welcome_text = (
        "⚙️ *Settings* ⚙️\n\n"
        "Here you can set your standing preferences for restaurant recommendations.\n\n"
        "Your current preferences are:\n"
    )

    # Get user preferences or initialize empty
    user_id = str(message.from_user.id)
    prefs = user_preferences.get(user_id, [])

    # Format preferences or show default message
    if prefs:
        pref_text = "\n".join([f"- {pref}" for pref in prefs])
        welcome_text += pref_text
    else:
        welcome_text += "_No preferences set_"

    # Instructions to add or remove
    welcome_text += (
        "\n\nTo add a preference, send:\n"
        "`/add preference_name`\n\n"
        "To remove a preference, send:\n"
        "`/remove preference_name`\n\n"
        "Examples of good preferences: 'vegetarian', 'romantic', 'outdoor seating', etc."
    )

    await bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['add'])
async def add_preference(message):
    """Add a user preference"""
    user_id = str(message.from_user.id)

    # Get the preference from the message
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await bot.send_message(message.chat.id, "Please specify a preference to add. Example: `/add vegetarian`", parse_mode="Markdown")
        return

    preference = parts[1].lower().strip()

    # Initialize user preferences if not exist
    if user_id not in user_preferences:
        user_preferences[user_id] = []

    # Add preference if not already present
    if preference not in user_preferences[user_id]:
        user_preferences[user_id].append(preference)
        await bot.send_message(message.chat.id, f"Added preference: *{preference}*", parse_mode="Markdown")
    else:
        await bot.send_message(message.chat.id, f"Preference *{preference}* already exists", parse_mode="Markdown")

@bot.message_handler(commands=['remove'])
async def remove_preference(message):
    """Remove a user preference"""
    user_id = str(message.from_user.id)

    # Get the preference from the message
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await bot.send_message(message.chat.id, "Please specify a preference to remove. Example: `/remove vegetarian`", parse_mode="Markdown")
        return

    preference = parts[1].lower().strip()

    # Check if user has preferences and remove if exists
    if user_id in user_preferences and preference in user_preferences[user_id]:
        user_preferences[user_id].remove(preference)
        await bot.send_message(message.chat.id, f"Removed preference: *{preference}*", parse_mode="Markdown")
    else:
        await bot.send_message(message.chat.id, f"Preference *{preference}* not found", parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
async def clear_preferences(message):
    """Clear all user preferences"""
    user_id = str(message.from_user.id)

    if user_id in user_preferences:
        user_preferences[user_id] = []

    await bot.send_message(message.chat.id, "All your preferences have been cleared.", parse_mode="Markdown")

@bot.message_handler(commands=['fetch_test'])
async def fetch_test_command(message):
    """Test URL fetching functionality"""
    # Get URL from message
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await bot.send_message(message.chat.id, "Please provide a URL to test. Example: `/fetch_test https://example.com`", parse_mode="Markdown")
        return

    url = parts[1].strip()

    # Send processing message
    processing_msg = await bot.send_message(message.chat.id, f"Testing URL fetch for: {url}...")

    try:
        # Initialize WebScraper
        from agents.scraper import WebScraper
        scraper = WebScraper(config)

        # Run the fetch test
        result = await scraper.fetch_url(url)

        # Format result message
        if 'error' in result:
            result_msg = f"❌ Error fetching URL: {result.get('error')}"
        else:
            result_msg = (
                f"✅ Successfully fetched URL: {url}\n"
                f"Status code: {result.get('status_code')}\n"
                f"Content length: {result.get('content_length')} characters\n\n"
                f"Preview:\n{result.get('content_preview', '')[:200]}..."
            )

        await bot.edit_message_text(result_msg, message.chat.id, processing_msg.message_id)
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in fetch_test: {error_details}")
        await bot.edit_message_text(f"❌ Error: {str(e)}", message.chat.id, processing_msg.message_id)

@bot.message_handler(commands=['scrape_test'])
async def scrape_test_command(message):
    """Test URL scraping functionality"""
    # Get URL from message
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await bot.send_message(message.chat.id, "Please provide a URL to test. Example: `/scrape_test https://example.com`", parse_mode="Markdown")
        return

    url = parts[1].strip()

    # Send processing message
    processing_msg = await bot.send_message(message.chat.id, f"Testing URL scraping for: {url}...\nThis may take a moment.")

    try:
        # Initialize WebScraper
        from agents.scraper import WebScraper
        scraper = WebScraper(config)

        # Run the scrape test
        result = await scraper.scrape_url(url)

        # Format result message
        if result.get('error'):
            result_msg = f"❌ Error scraping URL: {result.get('error')}"
        else:
            # Get a preview of scraped content
            content_preview = result.get('scraped_content', '')[:300]
            content_length = len(result.get('scraped_content', ''))

            result_msg = (
                f"✅ Successfully scraped URL: {url}\n"
                f"Status code: {result.get('status_code')}\n"
                f"Quality score: {result.get('quality_score', 0):.2f}\n"
                f"Content length: {content_length} characters\n\n"
                f"Preview:\n{content_preview}..."
            )

        await bot.edit_message_text(result_msg, message.chat.id, processing_msg.message_id)
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in scrape_test: {error_details}")
        await bot.edit_message_text(f"❌ Error: {str(e)}", message.chat.id, processing_msg.message_id)

@bot.message_handler(func=lambda message: True)
async def handle_message(message):
    """Handle all other messages as restaurant recommendation requests"""
    global orchestrator

    # Initialize orchestrator if not already done
    if orchestrator is None:
        orchestrator = LangChainOrchestrator(config)

    # Get user ID
    user_id = str(message.from_user.id)

    # Get user's standing preferences
    standing_prefs = user_preferences.get(user_id, [])

    # Send typing status
    await bot.send_chat_action(message.chat.id, 'typing')

    # Send initial response
    initial_response = await bot.send_message(
        message.chat.id,
        "🔍 Searching for restaurant recommendations...\nThis may take a minute or two.",
        parse_mode="HTML"
    )

    try:
        # Process the query using orchestrator
        results = await sync_to_async(orchestrator.process_query)(message.text, standing_prefs)

        # Extract telegram formatted text
        telegram_text = results.get('telegram_text', "<b>Извините, не удалось найти рестораны по вашему запросу.</b>")

        # Update original message with results
        await bot.edit_message_text(
            telegram_text,
            message.chat.id,
            initial_response.message_id,
            parse_mode="HTML"
        )

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error processing recommendation: {error_details}")

        # Update message with error
        await bot.edit_message_text(
            "<b>Извините, произошла ошибка при обработке вашего запроса.</b>\n\nПожалуйста, попробуйте позже или измените запрос.",
            message.chat.id,
            initial_response.message_id,
            parse_mode="HTML"
        )

def main():
    """Start the Telegram bot"""
    logger.info("Starting restaurant recommendation Telegram bot")

    # Start the bot polling in threaded mode
    from telebot.util import threaded

    @threaded
    def bot_polling():
        logger.info("Bot polling started")
        asyncio.run(bot.polling(none_stop=True, interval=1))

    # Start the bot
    bot_polling()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle graceful shutdown
        logger.info("Bot shutting down...")
        asyncio.run(wait_for_pending_tasks())
        logger.info("Bot stopped")

if __name__ == "__main__":
    main()