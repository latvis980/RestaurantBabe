# telegram_bot.py
import telebot
import logging
import time
import traceback
import asyncio
from agents.langchain_orchestrator import LangChainOrchestrator
import os
import config
from langchain_core.tracers.langchain import wait_for_all_tracers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize bot
bot = telebot.TeleBot(os.environ["TELEGRAM_BOT_TOKEN"])

# Initialize orchestrator
orchestrator = LangChainOrchestrator(config)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Handle start and help commands"""
    bot.reply_to(message, 
                "Привет! Я ИИ-ассистент про прозвищу Restaurant Babe и я умею находить самые вкусные, самые модные, самые классные рестораны, кафе, пекарни, бары и кофейни по всему миру. Напишите, что вы ищете, например: \n\n'Модные места для бранча в Лиссабоне с необычными блюдами'\n\n Я наведу справки у своих знакомых ресторанных экспертов, пролистаю колонки гастрономических критиков — и выдам лучшие рекомендации. \n\n Это может занять пару минут, потому что ищу я очень внимательно и тщательно проверяю результаты. Но никаких случайных мест в моем списке не будет. \n\nНачнем? Напишите свой первый запрос!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all other messages - pure passthrough with no formatting"""
    try:
        user_query = message.text
        # Send typing status
        bot.send_chat_action(message.chat.id, 'typing')
        # Acknowledge receipt of the message
        initial_reply = bot.reply_to(message, "Я ищу для вас рестораны. Это может занять несколько минут...")

        # Process the query
        logger.info(f"Processing query from user {message.from_user.id}: {user_query}")
        start_time = time.time()

        try:
            # Call the orchestrator to process the query
            result = orchestrator.process_query(user_query)

            end_time = time.time()
            logger.info(f"Query processed in {end_time - start_time:.2f} seconds")

            # Delete the "processing" message to avoid cluttering the chat
            try:
                bot.delete_message(message.chat.id, initial_reply.message_id)
            except Exception as e:
                logger.warning(f"Could not delete initial message: {e}")

            # Get the pre-formatted HTML text
            if isinstance(result, dict) and "telegram_text" in result:
                formatted_text = result["telegram_text"]
            elif isinstance(result, str):
                formatted_text = result
            else:
                # We shouldn't get here if orchestrator is working properly
                formatted_text = "Sorry, couldn't format restaurant recommendations properly."

            # Ensure the message isn't too long for Telegram
            if len(formatted_text) > 4000:
                formatted_text = formatted_text[:3997] + "..."

            # Send the response
            bot.send_message(
                message.chat.id, 
                formatted_text,
                parse_mode='HTML'
            )

        except Exception as process_error:
            logger.error(f"Error processing query: {process_error}")
            logger.error(traceback.format_exc())
            bot.reply_to(message, "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.")

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        bot.reply_to(message, "Извините, произошла ошибка. Пожалуйста, попробуйте еще раз.")
    finally:
        # Ensure all traces are submitted
        wait_for_all_tracers()

        # Also wait for our async tasks
        from utils.async_utils import wait_for_pending_tasks
        try:
            # Get the event loop or create a new one
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in a running loop, so create a task that will run later
                    asyncio.create_task(wait_for_pending_tasks())
                else:
                    # We have a loop but it's not running
                    loop.run_until_complete(wait_for_pending_tasks())
            except RuntimeError:
                # No running loop, create a new one
                asyncio.run(wait_for_pending_tasks())
        except Exception as e2:
            logger.warning(f"Task cleanup failed: {e2}")

def shutdown():
    """Clean shutdown function for asyncio resources"""
    logger.info("Shutting down and cleaning up resources...")
    from utils.async_utils import wait_for_pending_tasks
    try:
        asyncio.run(wait_for_pending_tasks())
    except RuntimeError as e:
        logger.warning(f"Could not run wait_for_pending_tasks during shutdown: {e}")
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(wait_for_pending_tasks())
        except Exception as e2:
            logger.warning(f"Alternative shutdown cleanup also failed: {e2}")

def main():
    """Main function to start the bot"""
    logger.info("Starting Telegram Bot")

    # Register the shutdown function
    import atexit
    atexit.register(shutdown)

    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Error in bot polling: {e}", exc_info=True)
    finally:
        # Make sure all traces are submitted before exiting
        wait_for_all_tracers()
        # Final cleanup attempt
        shutdown()

if __name__ == '__main__':
    main()