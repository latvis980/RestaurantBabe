# telegram_bot.py
import telebot
from telebot import types
import logging
import time
import traceback
from agents.langchain_orchestrator import LangChainOrchestrator
import config
from langchain_core.tracers.langchain import wait_for_all_tracers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize bot
bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize orchestrator
orchestrator = LangChainOrchestrator(config)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Handle start and help commands"""
    bot.reply_to(message, 
                "Привет! Я помогу найти лучшие рестораны. Просто напишите, что вы ищете, например: 'Хочу найти потрясающие бранч-места в Лиссабоне с необычными блюдами'")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all other messages"""
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

            # Check if result is valid
            if not result or not isinstance(result, dict):
                raise ValueError(f"Invalid result format: {type(result)}")

            # Format the response for Telegram
            response = format_telegram_response(result)

            # Delete the "processing" message to avoid cluttering the chat
            try:
                bot.delete_message(message.chat.id, initial_reply.message_id)
            except Exception as e:
                logger.warning(f"Could not delete initial message: {e}")

            # Send the response
            bot.send_message(
                message.chat.id, 
                response,
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

def format_telegram_response(result):
    """Format the result for Telegram HTML message"""
    try:
        response = "<b>🍽️ РЕКОМЕНДУЕМЫЕ РЕСТОРАНЫ:</b>\n\n"

        # Add recommended restaurants
        recommended = result.get("recommended", [])
        if recommended:
            for i, restaurant in enumerate(recommended, 1):
                response += format_restaurant(restaurant, i)
        else:
            response += "К сожалению, рекомендуемые рестораны не найдены.\n\n"

        # Add hidden gems
        response += "\n\n<b>💎 ДЛЯ СВОИХ:</b>\n\n"
        hidden_gems = result.get("hidden_gems", [])
        if hidden_gems:
            for i, restaurant in enumerate(hidden_gems, 1):
                response += format_restaurant(restaurant, i)
        else:
            response += "К сожалению, скрытые жемчужины не найдены.\n\n"

        # Add footer
        response += "\n\n<i>Рекомендации составлены на основе анализа экспертных источников.</i>"

        # Ensure response isn't too long for Telegram
        if len(response) > 4000:
            response = response[:3997] + "..."

        return response
    except Exception as e:
        logger.error(f"Error formatting Telegram response: {e}", exc_info=True)
        return "Извините, произошла ошибка при форматировании ответа."

def format_restaurant(restaurant, index):
    """Format a single restaurant for Telegram HTML message"""
    try:
        response = f"<b>{index}. {restaurant.get('name', 'Ресторан')}</b>\n"

        # Add address
        if restaurant.get('address'):
            response += f"📍 {restaurant.get('address')}\n"

        # Add description
        if restaurant.get('description'):
            response += f"{restaurant.get('description')}\n"

        # Add price range
        if restaurant.get('price_range'):
            response += f"💰 {restaurant.get('price_range')}\n"
        elif restaurant.get('price_indication'):
            response += f"💰 {restaurant.get('price_indication')}\n"

        # Add recommended dishes
        if restaurant.get('recommended_dishes'):
            dishes = restaurant.get('recommended_dishes')
            if isinstance(dishes, list):
                dishes_str = ", ".join(dishes)
            else:
                dishes_str = dishes
            response += f"👨‍🍳 Рекомендуемые блюда: {dishes_str}\n"

        # Add sources
        if restaurant.get('sources'):
            sources = restaurant.get('sources')
            if isinstance(sources, list):
                sources_str = ", ".join(sources)
            else:
                sources_str = sources
            response += f"📝 Рекомендовано: {sources_str}\n"

        # Add reservations if required
        if restaurant.get('reservations_required'):
            response += "⚠️ Рекомендуется бронирование\n"

        # Add Instagram if available
        if restaurant.get('instagram'):
            response += f"📸 {restaurant.get('instagram')}\n"

        # Add hours if available
        if restaurant.get('hours'):
            response += f"🕒 {restaurant.get('hours')}\n"

        response += "\n"
        return response
    except Exception as e:
        logger.error(f"Error formatting restaurant info: {e}")
        return f"<b>{index}. {restaurant.get('name', 'Ресторан')}</b>\n" + \
               "Извините, дополнительная информация недоступна.\n\n"

def main():
    """Main function to start the bot"""
    logger.info("Starting Telegram Bot")
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Error in bot polling: {e}", exc_info=True)
    finally:
        # Make sure all traces are submitted before exiting
        wait_for_all_tracers()

if __name__ == '__main__':
    main()