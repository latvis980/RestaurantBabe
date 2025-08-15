# telegram_bot.py - REFACTORED with Centralized AI Handler
"""
Telegram Bot with Centralized Conversation Handler

Key Changes:
- Single entry point for all message processing
- Centralized AI decision making
- Clean separation between message handling and search execution
- Scalable architecture for new query types
"""

import telebot
from telebot import types
import logging
import time
import threading
import asyncio
from threading import Event
from typing import Dict, List, Any, Optional, Tuple

# Import the centralized handler
from utils.conversation_handler import CentralizedConversationHandler, ConversationState
from utils.voice_handler import VoiceMessageHandler

import config
from utils.orchestrator_manager import get_orchestrator
from location.telegram_location_handler import TelegramLocationHandler, LocationData
from location.location_analyzer import LocationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize bot
if not config.TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in config")
bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Initialize centralized conversation handler
conversation_handler = None  # Will be initialized in main()

# Initialize other components
location_handler = TelegramLocationHandler(config)
location_analyzer = None  # Will be initialized in main()
voice_handler = None  # Will be initialized in main()

# Track active searches for cancellation
active_searches = {
}  # user_id -> {"cancel_event": Event, "chat_id": int, "start_time": float}

# Track users awaiting location input
users_awaiting_location = {}  # user_id -> {"query": str, "timestamp": float}

# Welcome message
WELCOME_MESSAGE = (
    "🍸 Hello! I'm Restaurant Babe, but friends call me Babe. I know all about the most delicious and trendy restaurants, cafes, bakeries, bars, and coffee shops around the world.\n\n"
    "Tell me what you're looking for, like <i>best specialty coffee places in Berlin</i>, or I can search for good places around you.\n\n"
    "I'll check with my restaurant critic friends and provide the best recommendations. This might take a couple of minutes because I search very carefully.\n\n"
    "I understand voice messages too!\n\n"
    "💡 <b>Tip:</b> Type /cancel anytime to stop a search.\n\n"
    "What are you hungry for?")

# ============ UTILITY FUNCTIONS ============


def create_cancel_event(user_id, chat_id):
    """Create a cancellation event for a user's search"""
    cancel_event = Event()
    active_searches[user_id] = {
        "cancel_event": cancel_event,
        "chat_id": chat_id,
        "start_time": time.time()
    }
    logger.info(f"Created cancel event for user {user_id}")
    return cancel_event


def cleanup_search(user_id):
    """Clean up search tracking for a user"""
    if user_id in active_searches:
        del active_searches[user_id]
        logger.info(f"Cleaned up search tracking for user {user_id}")


def is_search_cancelled(user_id):
    """Check if search has been cancelled for this user"""
    if user_id in active_searches:
        return active_searches[user_id]["cancel_event"].is_set()
    return False


def create_location_button():
    """Create reply keyboard with one-click location sharing button"""
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True,
                                       resize_keyboard=True)
    location_button = types.KeyboardButton("📍 Share My Location",
                                           request_location=True)
    markup.add(location_button)
    markup.add(types.KeyboardButton("❌ Cancel"))
    return markup


def remove_location_button():
    """Remove reply keyboard"""
    return types.ReplyKeyboardRemove()


@bot.message_handler(func=lambda message: message.text == "❌ Cancel")
def handle_location_cancel(message):
    """Handle location sharing cancellation"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Remove from awaiting location if applicable
    if user_id in users_awaiting_location:
        del users_awaiting_location[user_id]

    # Remove keyboard
    bot.send_message(
        chat_id,
        "No problem! You can just tell me your neighborhood or area name instead.\n\n"
        "For example: <i>\"I'm in downtown\", \"Near Central Park\", \"Chinatown area\"</i>",
        parse_mode='HTML',
        reply_markup=remove_location_button())


# ============ CORE MESSAGE PROCESSING ============


def process_text_message(message_text: str,
                         user_id: int,
                         chat_id: int,
                         is_voice: bool = False):
    """
    Step 1: Process any text message (original text or transcribed voice)
    This is the single entry point for all conversation processing
    """
    try:
        # Check if conversation handler is initialized
        if conversation_handler is None:
            logger.error("Conversation handler not initialized")
            bot.send_message(
                chat_id,
                "😔 I'm having trouble initializing. Please try again in a moment.",
                parse_mode='HTML')
            return

        # Check if user has active search
        if user_id in active_searches:
            bot.send_message(
                chat_id,
                "⏳ I'm currently searching for restaurants for you. Please wait or type /cancel to stop the search.",
                parse_mode='HTML')
            return

        # Check if user was awaiting location
        if user_id in users_awaiting_location:
            handle_location_input(message_text, user_id, chat_id)
            return

        if conversation_handler and conversation_handler.get_user_state(user_id) == ConversationState.AWAITING_LOCATION_CLARIFICATION:
            # Handle clarification response directly
            result = conversation_handler.handle_location_clarification(user_id, message_text)
            execute_action(result, user_id, chat_id)
            return

        # Send typing indicator
        bot.send_chat_action(chat_id, 'typing')

        # Step 2: Process with centralized AI handler
        result = conversation_handler.process_message(
            message_text=message_text,
            user_id=user_id,
            chat_id=chat_id,
            is_voice=is_voice)

        # Step 3: Execute the determined action
        execute_action(result, user_id, chat_id)

    except Exception as e:
        logger.error(f"Error processing message for user {user_id}: {e}")
        bot.send_message(
            chat_id,
            "😔 I had trouble understanding that. Could you tell me what restaurants you're looking for?",
            parse_mode='HTML')


def execute_action(result: Dict[str, Any], user_id: int, chat_id: int):
    """
    Execute the action determined by the centralized AI handler
    """
    action = result.get("action")
    action_data = result.get("action_data", {})
    bot_response = result.get("bot_response", "")

    # Always send bot response if provided
    if result.get("response_needed", True) and bot_response:
        bot.send_message(chat_id, bot_response, parse_mode='HTML')

    # Execute specific actions
    if action == "LAUNCH_CITY_SEARCH":
        # City-wide restaurant search using existing LangChain orchestrator
        search_query = action_data.get("search_query")
        threading.Thread(target=perform_city_search,
                         args=(search_query, chat_id, user_id),
                         daemon=True).start()

    elif action == "REQUEST_USER_LOCATION":
        # Request user's physical location
        request_user_location(user_id, chat_id,
                              action_data.get("context", "restaurants"))

    elif action == "LAUNCH_LOCATION_SEARCH":
        # Geographic location search
        search_query = action_data.get("search_query")
        threading.Thread(target=perform_location_search,
                         args=(search_query, user_id, chat_id),
                         daemon=True).start()

    elif action == "SEND_LOCATION_CLARIFICATION":
        # Send clarification request (bot_response already sent above)
        analysis_result = action_data.get("analysis_result", {})

        # Store context for clarification
        if conversation_handler:
            conversation_handler.store_ambiguous_location_context(user_id, {
                "query": analysis_result.get("original_message", ""),
                "location_detected": analysis_result.get("location_detected", ""),
                "ambiguity_reason": analysis_result.get("ambiguity_reason", "")
            })

    elif action == "PROCESS_LOCATION_CLARIFICATION":
        # Process user's clarification
        clarification_text = action_data.get("clarification_text", "")

        if conversation_handler:
            clarification_result = conversation_handler.handle_location_clarification(
                user_id, clarification_text)

            if clarification_result.get("action") == "SEARCH_LOCATION":
                search_query = clarification_result.get("search_query", "")
                threading.Thread(target=perform_location_search,
                               args=(search_query, user_id, chat_id),
                               daemon=True).start()

    elif action == "LAUNCH_GOOGLE_MAPS_SEARCH":
        # FIXED: Google Maps search for more options in same location
        search_type = action_data.get("search_type")
        if search_type == "google_maps_more":
            # Get stored location context
            if conversation_handler is None:
                bot.send_message(chat_id, "😔 I'm having trouble with the conversation system. Please try again.", parse_mode='HTML')
                return

            location_context = conversation_handler.get_location_search_context(user_id)
            if not location_context:
                bot.send_message(chat_id, "😔 I don't have the location context. Please search again.", parse_mode='HTML')
                return

            # Use the dedicated "more results" method
            original_query = location_context.get("query", "restaurants")
            location_data = location_context.get("location_data")
            location_description = location_context.get("location_description", "the area")

            # Extract coordinates
            coordinates = (location_data.latitude, location_data.longitude) if location_data else None
            if not coordinates:
                bot.send_message(chat_id, "😔 Could not get coordinates. Please try again.", parse_mode='HTML')
                return

            # Call orchestrator's "more results" method directly
            threading.Thread(target=call_orchestrator_more_results,
                           args=(original_query, coordinates, location_description, user_id, chat_id),
                           daemon=True).start()
        else:
            logger.warning(f"Unknown Google Maps search type: {search_type}")

    elif action == "LAUNCH_WEB_SEARCH":
        # General question web search (planned feature)
        search_query = action_data.get("search_query")
        bot.send_message(
            chat_id,
            "🔍 Web search feature coming soon! For now, I specialize in restaurant recommendations.",
            parse_mode='HTML')

    elif action in ["SEND_REDIRECT", "SEND_CLARIFICATION", "ERROR"]:
        # Bot response already sent above
        pass

    else:
        logger.warning(f"Unknown action: {action}")


def call_orchestrator_more_results(query: str, coordinates: tuple, location_desc: str, user_id: int, chat_id: int):
    """
    Call orchestrator's more results method for Google Maps search
    This is a clean wrapper that delegates to the location orchestrator
    """
    processing_msg = None
    try:
        logger.info(f"🔍 Starting 'more results' search for user {user_id}: '{query}' in {location_desc}")

        # Add to active searches for cancellation tracking
        create_cancel_event(user_id, chat_id)

        # Send processing message
        processing_msg = bot.send_message(
            chat_id,
            f"🔍 <b>Searching for more {query} options in {location_desc}...</b>",
            parse_mode='HTML'
        )

        # Create location orchestrator
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        def cancel_check():
            return is_search_cancelled(user_id)

        # Create async loop and call orchestrator's more results method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            location_orchestrator.process_more_results_query(
                query=query,
                coordinates=coordinates,
                location_desc=location_desc,
                cancel_check_fn=cancel_check
            )
        )

        loop.close()

        # Clean up processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass

        if is_search_cancelled(user_id):
            return

        # Handle results using the same logic as perform_location_search
        if result.get("success"):
            # Check if this requires media verification
            if result.get("requires_verification"):
                # Handle Google Maps with verification flow
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                loop.run_until_complete(
                    handle_google_maps_with_verification(
                        chat_id=chat_id,
                        user_id=user_id,
                        orchestrator_result=result,
                        original_query=query,
                        location_description=location_desc
                    )
                )

                loop.close()
            else:
                # Direct results without verification
                formatted_message = result.get("location_formatted_results", 
                    f"Found {result.get('restaurant_count', 0)} more restaurants!")

                bot.send_message(
                    chat_id,
                    formatted_message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )

            logger.info(f"✅ 'More results' search completed for user {user_id}: {result.get('restaurant_count', 0)} restaurants")
        else:
            # Handle error case
            error_message = result.get("location_formatted_results", "😔 Couldn't find more restaurants in that area.")
            bot.send_message(
                chat_id,
                error_message,
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"❌ Error in 'more results' search for user {user_id}: {e}")

        # Clean up processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass

        bot.send_message(
            chat_id,
            "😔 I had trouble searching for more restaurants. Please try again.",
            parse_mode='HTML'
        )
    finally:
        # Always clean up search tracking
        cleanup_search(user_id)

# ============ SEARCH EXECUTION FUNCTIONS ============


def perform_city_search(search_query: str, chat_id: int, user_id: int):
    """Execute city-wide restaurant search using existing orchestrator"""
    try:
        cancel_event = create_cancel_event(user_id, chat_id)

        # Send processing message
        # Send processing message with video
        try:
            with open('media/searching.mp4', 'rb') as video:
                processing_msg = bot.send_video(
                    chat_id,
                    video,
                    caption="🔍 <b>Searching for the best restaurants...</b>\n\n⏱ This might take a minute while I check with my sources.",
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.warning(f"Could not send video: {e}")
            # Fallback to text message
            processing_msg = bot.send_message(
                chat_id,
                "🔍 <b>Searching for the best restaurants...</b>\n\n⏱ This might take a minute while I check with my sources.",
                parse_mode='HTML')

        # Get orchestrator instance
        orchestrator = get_orchestrator()

        # FIXED: Use the orchestrator.search() method (not .chain.invoke())
        result = orchestrator.process_query(search_query)

        # Clean up processing message
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except Exception:
            pass

        if is_search_cancelled(user_id):
            return

        # FIXED: Send results using the correct key name
        if result and result.get("langchain_formatted_results"):
            formatted_message = result["langchain_formatted_results"]

            # Debug logging to see what we got
            logger.info(f"🔍 Result keys: {list(result.keys())}")
            logger.info(f"📱 Message length: {len(formatted_message)} characters")

            bot.send_message(
                chat_id,
                formatted_message,
                parse_mode='HTML',
                disable_web_page_preview=True)
            logger.info(f"✅ City search results sent for user {user_id}")
        else:
            # Log the actual result structure for debugging
            logger.warning(f"❌ Search failed or no langchain_formatted_results. Result keys: {list(result.keys()) if result else 'None'}")
            logger.warning(f"❌ Result content: {result}")

            bot.send_message(
                chat_id,
                "😔 I couldn't find any restaurants matching your criteria. Could you try a different search?",
                parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error in city search: {e}")
        # Clean up processing message
        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except Exception:
            pass
        bot.send_message(
            chat_id,
            "😔 I encountered an error while searching. Please try again!",
            parse_mode='HTML')
    finally:
        cleanup_search(user_id)

def perform_location_search(search_query: str, user_id: int, chat_id: int):
    """
    Execute location-based search using the actual location orchestrator
    CLEAN VERSION: Orchestrator handles ALL verification internally
    """
    processing_msg = None
    try:
        cancel_event = create_cancel_event(user_id, chat_id)

        # Send processing message with video
        try:
            with open('media/searching.mp4', 'rb') as video:
                processing_msg = bot.send_video(
                    chat_id,
                    video,
                    caption="📍 <b>Searching for restaurants in that area...</b>\n\n⏱ Checking my curated collection and finding the best places nearby.",
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.warning(f"Could not send video: {e}")
            processing_msg = bot.send_message(
                chat_id,
                "📍 <b>Searching for restaurants in that area...</b>\n\n⏱ Checking my curated collection and finding the best places nearby.",
                parse_mode='HTML')

        # Extract location from search query
        location_data = location_handler.extract_location_from_text(search_query)

        if location_data.confidence < 0.3:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass
            bot.send_message(chat_id, "😔 I couldn't understand the location. Could you be more specific about where you want to search?", parse_mode='HTML')
            return

        # Store location context for follow-up searches
        if conversation_handler is not None:
            conversation_handler.store_location_search_context(
                user_id=user_id,
                query=search_query,
                location_data=location_data,
                location_description=location_data.description or "the area"
            )

        # Create location orchestrator and cancel check function
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        def cancel_check():
            return is_search_cancelled(user_id)

        # Create async loop for location processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Process location query with orchestrator - ORCHESTRATOR HANDLES EVERYTHING
        result = loop.run_until_complete(
            location_orchestrator.process_location_query(
                query=search_query,
                location_data=location_data,
                cancel_check_fn=cancel_check
            )
        )

        loop.close()

        # Clean up processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass

        if is_search_cancelled(user_id):
            return

        # CLEAN: Handle ALL result types the same way - no special verification logic
        if result.get("success"):
            formatted_message = result.get("location_formatted_results", 
                f"Found {result.get('restaurant_count', 0)} restaurants!")

            # Handle database choice option
            if result.get("source") == "database_with_choice" and result.get("offer_more_results"):
                if conversation_handler is not None:
                    conversation_handler.set_user_state(user_id, ConversationState.RESULTS_SHOWN)
                location_desc = location_data.description or "the area"
                formatted_message += f"\n\n💬 <b>Want more options?</b> Just ask me to find more restaurants in {location_desc}!"

            bot.send_message(
                chat_id,
                formatted_message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )

            logger.info(f"✅ Location search results sent for user {user_id}: {result.get('restaurant_count', 0)} restaurants")

        else:
            error_message = result.get("error", "I couldn't find restaurants in that area.")
            bot.send_message(
                chat_id,
                f"😔 {error_message}\n\nTry a different location or be more specific about the area?",
                parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error in location search: {e}")
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass
        bot.send_message(chat_id, "😔 I encountered an error while searching. Please try again!", parse_mode='HTML')
    finally:
        cleanup_search(user_id)


def request_user_location(user_id: int, chat_id: int, context: str):
    """Request user's physical location with button"""
    location_msg = (
        f"📍 <b>I'd love to help you find great {context} near you!</b>\n\n"
        "To give you the best recommendations, I need to know where you are:\n\n"
        "🗺️ <b>Option 1:</b> Tell me your neighborhood, street, or nearby landmark\n"
        "📍 <b>Option 2:</b> Use the button below to send your exact coordinates\n\n"
        "<i>Examples: \"I'm in Chinatown\", \"Near Times Square\", \"On Rua da Rosa in Lisbon\"</i>\n\n"
        "💡 <b>Don't worry:</b> I only use your location to find nearby places. I don't store it."
    )

    bot.send_message(chat_id,
                     location_msg,
                     parse_mode='HTML',
                     reply_markup=create_location_button())

    # Mark user as awaiting location
    users_awaiting_location[user_id] = {
        "query": context,
        "timestamp": time.time()
    }

async def handle_google_maps_with_verification(chat_id: int, user_id: int, orchestrator_result: Dict[str, Any], original_query: str, location_description: str):
    """
    Handle Google Maps results that require media verification
    Two-step process: intermediate message + verification + final results
    FIXED: Correct parameters and function calls for current architecture
    """
    try:
        # Step 1: Send intermediate message (don't mention Google Maps)
        intermediate_message = orchestrator_result.get("formatted_message", 
            "Found some restaurants in the vicinity, let me check what local media and international guides have to say about them.")

        processing_msg = bot.send_message(
            chat_id,
            intermediate_message,
            parse_mode='HTML'
        )

        # Step 2: Complete media verification
        venues = orchestrator_result.get("venues_for_verification", [])
        coordinates = orchestrator_result.get("coordinates")
        query = orchestrator_result.get("query", original_query)

        # Create location orchestrator instance
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        # Define cancel check function
        def cancel_check():
            return is_search_cancelled(user_id)

        # Run media verification using location_orchestrator
        final_result = await location_orchestrator.complete_media_verification(
            venues=venues,
            query=query,
            coordinates=coordinates,
            location_desc=location_description,
            cancel_check_fn=cancel_check
        )

        # Clean up processing message
        if processing_msg:
            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass

        if is_search_cancelled(user_id):
            return

        # Step 3: Send final verified results
        if final_result.get("success") and final_result.get("results"):
            formatted_message = final_result.get("location_formatted_results", 
                f"Found {len(final_result.get('results', []))} verified restaurants!")

            bot.send_message(
                chat_id,
                formatted_message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )

            logger.info(f"✅ Google Maps with verification completed for user {user_id}: {len(final_result.get('results', []))} venues")
        else:
            error_message = final_result.get("location_formatted_results", "😔 No suitable restaurants found after verification.")
            bot.send_message(
                chat_id,
                error_message,
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"❌ Error in Google Maps verification flow: {e}")
        bot.send_message(
            chat_id,
            "😔 Had trouble verifying restaurants. Please try again.",
            parse_mode='HTML'
        )

def handle_location_input(location_text: str, user_id: int, chat_id: int):
    """Handle location input from user who was awaiting location"""
    try:
        # Get original context
        awaiting_data = users_awaiting_location.get(user_id, {})
        context = awaiting_data.get("query", "restaurants")

        # Remove from awaiting list
        if user_id in users_awaiting_location:
            del users_awaiting_location[user_id]

        # Extract location data
        location_data = location_handler.extract_location_from_text(
            location_text)

        if location_data.confidence > 0.3:
            # Good location - start proximity search
            bot.send_message(
                chat_id,
                f"📍 <b>Perfect! Searching for {context} near {location_data.description}...</b>",
                parse_mode='HTML',
                reply_markup=remove_location_button())

            # Store location context for potential follow-up searches
            if conversation_handler is not None:
                conversation_handler.store_location_search_context(
                    user_id=user_id,
                    query=context,
                    location_data=location_data,
                    location_description=location_data.description
                    or "searched area")

            # Use location orchestrator for proximity search
            from location.location_orchestrator import LocationOrchestrator
            location_orchestrator = LocationOrchestrator(config)

            def is_cancelled():
                return False  # No cancellation for location input processing

            # Process with location orchestrator
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                location_orchestrator.process_location_query(
                    query=context,
                    location_data=location_data,
                    cancel_check_fn=is_cancelled))

            loop.close()

            # Handle results
            if result.get("success"):
                formatted_message = result.get(
                    "location_formatted_results",
                    f"Found restaurants near {location_data.description}!")

                # Check if this was a database result with choice option
                if result.get(
                        "source") == "database_with_choice" and result.get(
                            "offer_more_results"):
                    if conversation_handler is not None:
                        conversation_handler.set_user_state(
                            user_id, ConversationState.RESULTS_SHOWN)
                    location_desc = location_data.description or "the area"
                    formatted_message += f"\n\n💬 <b>Want more options?</b> Just ask me to find more restaurants in {location_desc}!"

                bot.send_message(chat_id,
                                 formatted_message,
                                 parse_mode='HTML',
                                 reply_markup=remove_location_button(),
                                 disable_web_page_preview=True)
            else:
                location_desc = location_data.description or "that location"
                bot.send_message(
                    chat_id,
                    f"😔 I couldn't find restaurants near {location_desc}. Try a different location?",
                    parse_mode='HTML')

        else:
            # Poor location understanding
            bot.send_message(
                chat_id,
                "😔 I couldn't understand that location. Could you be more specific?\n\n"
                "Examples: \"Downtown\", \"Near Central Park\", \"Chinatown\", \"Rua da Rosa\"",
                parse_mode='HTML',
                reply_markup=create_location_button())
            # Re-add to awaiting list
            users_awaiting_location[user_id] = awaiting_data

    except Exception as e:
        logger.error(f"Error handling location input: {e}")
        bot.send_message(
            chat_id,
            "😔 I had trouble processing that location. Could you try again?",
            parse_mode='HTML')


# ============ BOT COMMAND HANDLERS ============


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Handle /start and /help commands"""
    user_id = message.from_user.id

    # Initialize user in conversation handler
    if conversation_handler is not None:
        conversation_handler.set_user_state(user_id, ConversationState.IDLE)

    bot.reply_to(message, WELCOME_MESSAGE, parse_mode='HTML')


@bot.message_handler(commands=['cancel'])
def handle_cancel(message):
    """Handle /cancel command to stop current search"""
    user_id = message.from_user.id
    chat_id = message.chat.id

    logger.info(f"Cancel command received from user {user_id}")

    # Check if user has an active search
    if user_id not in active_searches:
        bot.reply_to(
            message,
            "🤷‍♀️ I'm not currently searching for anything. What restaurants are you looking for?",
            parse_mode='HTML')
        return

    # Cancel the search
    search_info = active_searches[user_id]
    search_info["cancel_event"].set()

    # Calculate search duration
    search_duration = round(time.time() - search_info["start_time"], 1)

    # Send cancellation confirmation
    bot.reply_to(
        message,
        f"✋ Search cancelled! I was searching for {search_duration} seconds.\n\n"
        "What else would you like me to help you find?",
        parse_mode='HTML')

    # Clean up
    cleanup_search(user_id)
    if user_id in users_awaiting_location:
        del users_awaiting_location[user_id]

    # Reset user state
    if conversation_handler is not None:
        conversation_handler.set_user_state(user_id, ConversationState.IDLE)

    logger.info(
        f"Successfully cancelled search for user {user_id} after {search_duration}s"
    )


@bot.message_handler(content_types=['location'])
def handle_gps_location(message):
    """Handle GPS location messages from users"""
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id

        # Check if user was awaiting location
        if user_id not in users_awaiting_location:
            bot.reply_to(
                message,
                "📍 Got your location! What kind of restaurants are you looking for nearby?",
                parse_mode='HTML')
            return

        # Get awaiting data
        awaiting_data = users_awaiting_location[user_id]
        context = awaiting_data.get("query", "restaurants")

        # Remove from awaiting list
        del users_awaiting_location[user_id]

        # Extract GPS coordinates
        latitude = message.location.latitude
        longitude = message.location.longitude

        logger.info(
            f"Received GPS location from user {user_id}: {latitude}, {longitude}"
        )

        # Create location data
        location_data = LocationData(
            latitude=latitude,
            longitude=longitude,
            description=f"GPS: {latitude:.4f}, {longitude:.4f}",
            location_type="gps",
            confidence=1.0)

        bot.send_message(
            chat_id,
            f"📍 <b>Perfect! Searching for {context} near your location...</b>",
            parse_mode='HTML',
            reply_markup=remove_location_button())

        # Store location context for potential follow-up searches
        if conversation_handler is not None:
            conversation_handler.store_location_search_context(
                user_id=user_id,
                query=context,
                location_data=location_data,
                location_description=f"GPS: {latitude:.4f}, {longitude:.4f}")

        # Use location orchestrator for GPS proximity search
        from location.location_orchestrator import LocationOrchestrator
        location_orchestrator = LocationOrchestrator(config)

        def is_cancelled():
            return False  # No cancellation for GPS location processing

        # Process with location orchestrator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            location_orchestrator.process_location_query(
                query=context,
                location_data=location_data,
                cancel_check_fn=is_cancelled))

        loop.close()

        # Handle results
        if result.get("success"):
            formatted_message = result.get(
                "location_formatted_results", f"Found restaurants near your location!")

            # Check if this was a database result with choice option
            if result.get("source") == "database_with_choice" and result.get(
                    "offer_more_results"):
                if conversation_handler is not None:
                    conversation_handler.set_user_state(
                        user_id, ConversationState.RESULTS_SHOWN)
                formatted_message += f"\n\n💬 <b>Want more options?</b> Just ask me to find more restaurants around here!"

            bot.send_message(chat_id,
                             formatted_message,
                             parse_mode='HTML',
                             reply_markup=remove_location_button(),
                             disable_web_page_preview=True)
        else:
            bot.send_message(
                chat_id,
                "😔 I couldn't find restaurants near your location. Try expanding the search area or be more specific about cuisine type?",
                parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error handling GPS location: {e}")
        bot.reply_to(
            message,
            "😔 I had trouble processing your location. Could you try again?",
            parse_mode='HTML')


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages - Step 1: Convert to text"""
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id

        logger.info(f"🎤 Received voice message from user {user_id}")

        # Check if voice handler is available
        if voice_handler is None:
            bot.reply_to(
                message,
                "😔 Voice processing is not available right now. Please send a text message.",
                parse_mode='HTML')
            return

        # Send processing message
        processing_msg = bot.reply_to(
            message,
            "🎤 <b>Processing your voice message...</b>",
            parse_mode='HTML')

        # Process voice in background
        threading.Thread(target=process_voice_in_background,
                         args=(message, user_id, chat_id,
                               processing_msg.message_id),
                         daemon=True).start()

    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        bot.reply_to(
            message,
            "😔 Sorry, I had trouble processing your voice message. Could you try again?",
            parse_mode='HTML')


def process_voice_in_background(message, user_id: int, chat_id: int,
                                processing_msg_id: int):
    """Background processing of voice message"""
    try:
        # Check if voice handler is available
        if voice_handler is None:
            bot.send_message(
                chat_id,
                "😔 Voice processing is not available right now. Please send a text message.",
                parse_mode='HTML')
            return

        # Step 1: Transcribe voice message
        transcribed_text = voice_handler.process_voice_message(
            bot, message.voice)

        # Clean up processing message
        try:
            bot.delete_message(chat_id, processing_msg_id)
        except Exception:
            pass

        if not transcribed_text:
            bot.send_message(
                chat_id,
                "😔 I couldn't understand your voice message. Could you try again or send a text message?",
                parse_mode='HTML')
            return

        logger.info(
            f"✅ Voice transcribed for user {user_id}: '{transcribed_text[:100]}...'"
        )

        # Step 2: Process transcribed text using the same pipeline as text messages
        process_text_message(transcribed_text, user_id, chat_id, is_voice=True)

    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        # Clean up processing message
        try:
            bot.delete_message(chat_id, processing_msg_id)
        except Exception:
            pass
        bot.send_message(
            chat_id,
            "😔 Sorry, I encountered an error processing your voice message.",
            parse_mode='HTML')


@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    """Handle all text messages - Main entry point"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_text = message.text

    logger.info(
        f"📝 Received text message from user {user_id}: '{message_text[:50]}...'"
    )

    # Process text message using centralized handler
    process_text_message(message_text, user_id, chat_id, is_voice=False)


@bot.callback_query_handler(func=lambda call: call.data == "share_location")
def handle_location_button(call):
    """Handle location sharing button press"""
    bot.answer_callback_query(
        call.id,
        "Please share your location using Telegram's location feature.")

    # Send instructions for sharing location
    bot.send_message(call.message.chat.id,
                     "📍 <b>To share your location:</b>\n\n"
                     "1. Tap the attachment button (📎)\n"
                     "2. Select 'Location'\n"
                     "3. Choose 'Send My Current Location'\n\n"
                     "Or just tell me your neighborhood/area name!",
                     parse_mode='HTML')


# ============ INITIALIZATION ============


def main():
    """Initialize and start the bot"""
    global conversation_handler, location_analyzer, voice_handler

    try:
        # Initialize conversation handler
        conversation_handler = CentralizedConversationHandler(config)
        logger.info("✅ Centralized conversation handler initialized")

        # Initialize location analyzer
        location_analyzer = LocationAnalyzer(config)
        logger.info("✅ Location analyzer initialized")

        # Initialize voice handler
        voice_handler = VoiceMessageHandler()
        logger.info("✅ Voice handler initialized")

        # Start bot
        logger.info("🤖 Starting Telegram bot...")
        bot.infinity_polling(timeout=10, long_polling_timeout=5)

    except Exception as e:
        logger.error(f"❌ Error starting bot: {e}")
        raise


if __name__ == "__main__":
    main()
