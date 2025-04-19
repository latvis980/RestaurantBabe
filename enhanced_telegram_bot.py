"""
Enhanced Telegram Bot for the Restaurant Recommendation App.

This module provides a flexible, conversational Telegram bot interface
that can understand natural language queries and context for restaurant recommendations.
Added support for voice messages using Whisper API for transcription.
"""
import asyncio
import logging
import re
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import json
from telegram.constants import ParseMode

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
    CallbackQueryHandler
)
from telegram.constants import ParseMode
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import config
from whisper_transcriber import WhisperTranscriber

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("enhanced_telegram_bot")

class EnhancedRestaurantBot:
    """
    Enhanced Telegram bot with natural language understanding capabilities
    and voice message handling for providing restaurant recommendations.
    """

    def __init__(self, recommender: Any):
        """
        Initialize the Enhanced Telegram bot with a restaurant recommender.

        Args:
            recommender: The restaurant recommendation orchestrator
        """
        self.token = config.TELEGRAM_BOT_TOKEN
        self.recommender = recommender
        self.application = None

        # Initialize transcriber for voice messages
        self.transcriber = WhisperTranscriber()

        # Initialize the OpenAI model for intent recognition
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=0.3,  # Lower temperature for more deterministic responses
        )

        # Create the intent recognition chain
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", config.INTENT_RECOGNITION_PROMPT),
            ("human", "{user_message}")
        ])

        # JSON output parser for structured intent data
        self.json_parser = JsonOutputParser()

        # Chain for intent recognition
        self.intent_chain = self.intent_prompt | self.llm | self.json_parser

        # Create conversation memory storage
        self.user_contexts = {}  # Store conversation context by user_id

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Start the conversation with a welcome message.
        """
        user = update.effective_user
        user_id = user.id

        # Initialize user context
        self.user_contexts[user_id] = {
            "last_restaurants": [],
            "last_query": None,
            "last_location": None,
            "conversation_history": []
        }

        welcome_message = (
            f"Hello {user.first_name}! \n\nI'm an AI restaurant expert and I can help you find a great spot to eat and drink.\n\n"
            f"I only use trusted sources like world-famous guides, local critics' publications and reputable foodies' blogs. No Tripadvisor, ever, promise!\n\n"
            f"<b>\Simply ask me something like:\</b>\n"
            f"• <i>\"Tell me where to find the best dim sum in Hong Kong 🥟\"</i>\n"
            f"• <i>\"Any ideas for romantic dinner in Paris? 🥂🦪\"</i>\n"
            f"• <i>\"Recommend some hidden cocktail bars in Manhattan 🍸\"</i>\n"
            f"• <i>\"I'm in Lisbon and looking for a brunch spot 🥐🍳\"</i>\n\n"
            f"You can also send me voice messages, and I'll transcribe and process your request!\n\n"
            f"What are you looking for today?"
        )
        await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Display help information.
        """
        help_text = (
            "🍽️ *Restaurant Recommendation Bot Help* 🍽️\n\n"
            "I can help you find great restaurants, bars, and cafes from trusted sources!\n\n"
            "*Commands:*\n"
            "/start - Start a new conversation\n"
            "/help - Show this help message\n\n"
            "*How to use:*\n"
            "• Ask naturally, for example:\n"
            "  - \"Where can I find good coffee in Berlin?\"\n"
            "  - \"Best Thai food in San Francisco\"\n" 
            "  - \"Affordable breakfast places in Barcelona\"\n\n"
            "• You can also send voice messages for your requests\n"
            "• You can ask follow-up questions about the results\n"
            "• You can request more details about a specific restaurant\n\n"
            "I always search reputable sources like Michelin Guide, 50 Best, Timeout, and professional food critics - never crowd-sourced review sites!"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Main handler for text messages that leverages the LangChain orchestrator.
        """
        user_message = update.message.text
        user_id = update.effective_user.id

        # Ensure user context exists and update conversation history
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "last_restaurants": [],
                "last_query": None,
                "last_location": None,
                "conversation_history": []
            }

        # Add message to conversation history
        self.user_contexts[user_id]["conversation_history"].append(user_message)
        if len(self.user_contexts[user_id]["conversation_history"]) > 5:
            self.user_contexts[user_id]["conversation_history"].pop(0)

        # Process the text message
        await self._process_user_input(update, context, user_message)

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handler for voice messages.
        """
        user_id = update.effective_user.id
        voice = update.message.voice

        # Ensure user context exists
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "last_restaurants": [],
                "last_query": None,
                "last_location": None,
                "conversation_history": []
            }

        # Send "transcribing" action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Let the user know we're processing their voice message
        status_message = await update.message.reply_text(
            "🎙️ Transcribing your voice message... Just a moment."
        )

        try:
            # Get file info
            file = await context.bot.get_file(voice.file_id)

            # Download and transcribe the voice message
            file_path = await self.transcriber.download_voice_file(file.file_path, self.token)
            transcription = await self.transcriber.transcribe(file_path)

            if not transcription:
                await status_message.edit_text(
                    "Sorry, I couldn't transcribe your voice message. Could you try again or type your request?"
                )
                return

            # Update the status message to show the transcription
            await status_message.edit_text(
                f"🎙️ I heard: \"{transcription}\"\n\nProcessing your request..."
            )

            # Add transcription to conversation history
            self.user_contexts[user_id]["conversation_history"].append(transcription)
            if len(self.user_contexts[user_id]["conversation_history"]) > 5:
                self.user_contexts[user_id]["conversation_history"].pop(0)

            # Process the transcribed text
            await self._process_user_input(update, context, transcription, status_message)

        except Exception as e:
            logger.error(f"Error handling voice message: {e}", exc_info=True)
            await status_message.edit_text(
                "Sorry, I encountered an error processing your voice message. Could you try again or type your request?"
            )

    async def _process_user_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                user_input: str, status_message = None) -> None:
        """
        Process user input from either text or voice messages.

        Args:
            update: The update from Telegram
            context: The context from Telegram
            user_input: The text to process (either from text message or transcribed voice)
            status_message: Optional message object to update instead of sending new messages
        """
        user_id = update.effective_user.id

        # Send "typing" action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            # Identify user intent using LLM
            intent_data = await self._identify_intent(user_input, user_id)
            logger.info(f"Identified intent: {intent_data}")

            # For restaurant_search, use the orchestrator directly
            if intent_data["intent"] == "restaurant_search":
                if status_message:
                    await status_message.edit_text(
                        "🔍 Searching for recommendations... This may take a moment."
                    )
                    searching_message = status_message
                else:
                    searching_message = await update.message.reply_text(
                        "🔍 Searching for recommendations... This may take a moment."
                    )

                # Extract information from intent_data
                query = intent_data["query"]
                location = intent_data["location"] or ""
                cuisine = intent_data["cuisine"] or ""
                language = intent_data.get("language", "English")

                # Store the query and location in context for future reference
                self.user_contexts[user_id]["last_query"] = query
                self.user_contexts[user_id]["last_location"] = location

                # Use the orchestrator directly
                result = await self.recommender.get_recommendation_async(
                    query=query,
                    location=location,
                    cuisine=cuisine,
                    language=language
                )

                # Delete the "searching" message if we created a new one
                if not status_message:
                    await context.bot.delete_message(
                        chat_id=update.effective_chat.id,
                        message_id=searching_message.message_id
                    )

                # Store restaurants in context for follow-up questions
                try:
                    if "compiled_results" in result and result["compiled_results"]:
                        # Try to parse if it's a string
                        if isinstance(result["compiled_results"], str):
                            import re
                            import json
                            json_match = re.search(r'\[.*\]', result["compiled_results"], re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                restaurants = json.loads(json_str)
                                self.user_contexts[user_id]["last_restaurants"] = restaurants
                        # If already a list
                        elif isinstance(result["compiled_results"], list):
                            self.user_contexts[user_id]["last_restaurants"] = result["compiled_results"]
                except Exception as parse_error:
                    logger.error(f"Error parsing restaurant results: {parse_error}")
                    self.user_contexts[user_id]["last_restaurants"] = result.get("restaurant_results", [])

                # Get the formatted response
                response_text = result.get("formatted_response", "Sorry, no recommendations found.")

                # Update the status message or send a new message with the results
                if status_message:
                    # Split if too long
                    if len(response_text) > 4000:
                        # First, update the status message with the first chunk
                        chunks = self._split_message(response_text)
                        await status_message.edit_text(
                            chunks[0],
                            parse_mode=ParseMode.HTML
                        )
                        # Then send the rest as new messages
                        for chunk in chunks[1:]:
                            await context.bot.send_message(
                                chat_id=update.effective_chat.id,
                                text=chunk,
                                parse_mode=ParseMode.HTML
                            )
                    else:
                        await status_message.edit_text(
                            response_text,
                            parse_mode=ParseMode.HTML
                        )
                else:
                    # Split if too long
                    if len(response_text) > 4000:
                        chunks = self._split_message(response_text)
                        for chunk in chunks:
                            await context.bot.send_message(
                                chat_id=update.effective_chat.id,
                                text=chunk,
                                parse_mode=ParseMode.HTML
                            )
                    else:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=response_text,
                            parse_mode=ParseMode.HTML
                        )

            # For follow-up questions about restaurants
            elif intent_data["intent"] == "followup_question" or intent_data["intent"] == "more_info":
                await self._handle_followup_with_orchestrator(update, context, intent_data, status_message)

            # For conversational queries, use the LLM directly
            elif intent_data["intent"] == "conversation":
                await self._handle_conversation(update, context, intent_data, status_message)

            # For help requests
            elif intent_data["intent"] == "help":
                if status_message:
                    await status_message.edit_text(
                        "Let me help you with that..."
                    )
                await self.help_command(update, context)

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            error_message = "Sorry, I encountered an error processing your request. Please try again."

            if status_message:
                await status_message.edit_text(error_message)
            else:
                await update.message.reply_text(error_message)

    async def _handle_followup_with_orchestrator(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                              intent_data: Dict[str, Any], status_message = None) -> None:
        """
        Handle follow-up questions using the orchestrator.
        """
        user_id = update.effective_user.id
        user_context = self.user_contexts.get(user_id, {})
        query = intent_data["query"]
        restaurant_name = intent_data.get("restaurant_name")
        location = user_context.get("last_location", "")

        # Send typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Create a specific search query
        if restaurant_name and restaurant_name != "null":
            search_query = f"Information about {restaurant_name}"
            if location:
                search_query += f" in {location}"
            search_query += f": {query}"
        else:
            # Use the general query for context
            search_query = query
            if user_context.get("last_query"):
                search_query = f"Follow-up about {user_context.get('last_query')}: {query}"

        # Inform user search is in progress
        if status_message:
            await status_message.edit_text(
                "🔍 Finding that information for you... Just a moment."
            )
            searching_message = status_message
        else:
            searching_message = await update.message.reply_text(
                "🔍 Finding that information for you... Just a moment."
            )

        try:
            # Get language from intent data or detect it
            language = intent_data.get("language")
            if not language:
                language = await self._detect_language(query)

            # Use the orchestrator to get a response
            result = await self.recommender.get_recommendation_async(
                query=search_query,
                location=location or "",
                cuisine="",
                language=language
            )

            # Delete the "searching" message if we created a new one
            if not status_message:
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=searching_message.message_id
                )

            # Get the formatted response
            response = result.get("formatted_response", f"Sorry, I couldn't find specific information about that.")

            # Split if too long
            if len(response) > 4000:
                chunks = self._split_message(response)

                if status_message:
                    # Update the status message with the first chunk
                    await status_message.edit_text(
                        chunks[0],
                        parse_mode=ParseMode.HTML
                    )
                    # Send the rest as new messages
                    for chunk in chunks[1:]:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk,
                            parse_mode=ParseMode.HTML
                        )
                else:
                    for chunk in chunks:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk,
                            parse_mode=ParseMode.HTML
                        )
            else:
                if status_message:
                    await status_message.edit_text(
                        response,
                        parse_mode=ParseMode.HTML
                    )
                else:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=response,
                        parse_mode=ParseMode.HTML
                    )

        except Exception as e:
            logger.error(f"Error during follow-up: {e}", exc_info=True)
            error_message = f"Sorry, I couldn't find the specific information you requested."

            if status_message:
                await status_message.edit_text(error_message)
            else:
                # Delete the "searching" message if it exists
                if 'searching_message' in locals():
                    await context.bot.delete_message(
                        chat_id=update.effective_chat.id,
                        message_id=searching_message.message_id
                    )

                await update.message.reply_text(error_message)

    async def _identify_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """
        Identify the intent behind a user message using OpenAI.
        """
        try:
            # Use the intent chain from initialization
            intent_data = await self.intent_chain.ainvoke({"user_message": user_message})
            logger.info(f"Identified intent: {intent_data}")
            return intent_data
        except Exception as e:
            logger.error(f"Error in intent recognition: {e}")
            # Simple fallback - don't overthink it
            return {
                "intent": "restaurant_search",
                "location": None,
                "cuisine": None,
                "restaurant_name": None,
                "query": user_message,
                "language": "English"
            }

    async def _handle_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                 intent_data: Dict[str, Any], status_message = None) -> None:
        """
        Handle conversational messages that don't require a search using OpenAI.
        """
        query = intent_data["query"]
        user_id = update.effective_user.id

        # Get conversation history for context
        history = self.user_contexts.get(user_id, {}).get("conversation_history", [])

        # Get language from intent data if available
        language = intent_data.get("language", "English")

        try:
            # Use conversation prompt from config
            conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", config.CONVERSATION_HANDLER_PROMPT),
                ("human", f"User message: {query}\nUser message language: {language}\nRecent conversation history: {history[-3:] if len(history) > 0 else 'None'}")
            ])

            conversation_chain = conversation_prompt | self.llm | StrOutputParser()
            response = await conversation_chain.ainvoke({})

            if status_message:
                await status_message.edit_text(response)
            else:
                await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            # Fallback to simple responses if AI generation fails

            # Check for simple patterns as fallback
            if any(word in query.lower() for word in ["thank", "thanks", "thx"]):
                response = "You're welcome! Feel free to ask me for any restaurant recommendations."
            elif any(word in query.lower() for word in ["hi", "hello", "hey", "greetings"]):
                response = "Hello! How can I help you find great restaurants today?"
            elif any(word in query.lower() for word in ["bye", "goodbye", "farewell"]):
                response = "Goodbye! Feel free to come back when you need restaurant recommendations."
            else:
                response = "I'm specialized in finding restaurant recommendations from trusted sources. " \
                         "How can I help you discover great places to eat or drink today?"

            if status_message:
                await status_message.edit_text(response)
            else:
                await update.message.reply_text(response)

    async def _detect_language(self, text: str) -> str:
        """
        AI-based language detection.

        Args:
            text: The text to detect language from

        Returns:
            Detected language name
        """
        try:
            # Use the LLM to detect language
            lang_prompt = ChatPromptTemplate.from_messages([
                ("system", config.LANGUAGE_DETECTION_PROMPT),
                ("human", f"Detect the language of this text: '{text}'")
            ])

            lang_chain = lang_prompt | self.llm | StrOutputParser()
            language = await lang_chain.ainvoke({})

            # Clean up and normalize the response
            language = language.strip().split('\n')[0].strip()
            if language.lower() == "the language is" or language.lower().startswith("the language"):
                language = language.split("is")[-1].strip()

            return language
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            # Default to English if detection fails
            return "English"

    def _split_message(self, message: str, max_length: int = 4000) -> List[str]:
        """
        Split a long message into multiple chunks for Telegram API limits.

        Args:
            message: The message to split
            max_length: Maximum length per chunk

        Returns:
            List of message chunks
        """
        if len(message) <= max_length:
            return [message]

        chunks = []
        current_chunk = ""

        # Split by paragraphs first
        paragraphs = message.split("\n\n")

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def error_handler(self, update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle errors in the Telegram bot.

        Args:
            update: The update that caused the error (may be None)
            context: The context with error information
        """
        logger.error(f"Update {update} caused error: {context.error}")

        # Safely access update attributes
        chat_id = None
        if update and update.effective_message:
            chat_id = update.effective_message.chat_id

        if chat_id:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Sorry, something went wrong. Please try again or type /start for help."
            )

    def setup_handlers(self) -> None:
        """Set up the command and message handlers."""
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))

        # Add message handler for all text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Add handler for voice messages
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))

        # Add error handler
        self.application.add_error_handler(self.error_handler)

    def run_polling(self) -> None:
        """Start the bot in polling mode."""
        # Create application
        self.application = Application.builder().token(self.token).build()

        # Set up handlers
        self.setup_handlers()

        # Start polling
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

    async def run_webhook(self, webhook_url: str, webhook_path: str = "/webhook", port: int = 8443) -> None:
        """
        Start the bot in webhook mode.

        Args:
            webhook_url: The URL for the webhook
            webhook_path: The path for the webhook
            port: The port to listen on
        """
        # Create application
        self.application = Application.builder().token(self.token).build()

        # Set up handlers
        self.setup_handlers()

        # Start webhook
        webhook_url = webhook_url.rstrip("/")
        webhook_full_url = f"{webhook_url}{webhook_path}"

        # Log webhook information
        logger.info(f"Setting webhook to: {webhook_full_url}")

        await self.application.bot.set_webhook(url=webhook_full_url)

        # Start webhook server
        await self.application.start()

        # Log successful start
        logger.info(f"Webhook server started at port {port}")

        # Keep the webhook server running
        update_queue = asyncio.Queue()
        await self.application.updater.start_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=webhook_path,
            webhook_url=webhook_full_url
        )