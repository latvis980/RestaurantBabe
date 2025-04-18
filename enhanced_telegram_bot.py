"""
Enhanced Telegram Bot for the Restaurant Recommendation App.

This module provides a flexible, conversational Telegram bot interface
that can understand natural language queries and context for restaurant recommendations.
"""
import asyncio
import logging
import re
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import json

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

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("enhanced_telegram_bot")

class EnhancedRestaurantBot:
    """
    Enhanced Telegram bot with natural language understanding capabilities
    for providing restaurant recommendations.
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

        # Initialize the OpenAI model for intent recognition
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=0.3,  # Lower temperature for more deterministic responses
        )

        # Create the intent recognition chain
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_intent_system_prompt()),
            ("human", "{user_message}")
        ])

        # JSON output parser for structured intent data
        self.json_parser = JsonOutputParser()

        # Chain for intent recognition
        self.intent_chain = self.intent_prompt | self.llm | self.json_parser

        # Create conversation memory storage
        self.user_contexts = {}  # Store conversation context by user_id

    def _get_intent_system_prompt(self) -> str:
        """Get the system prompt for intent recognition."""
        return """
        You are an advanced AI assistant that helps identify user intents for a restaurant recommendation bot. You can understand any language and recognize what the user is asking for.

        Analyze the user's message and categorize it into one of these intents:
        1. "restaurant_search" - User is looking for restaurant/bar/cafe recommendations. Extract location and cuisine/food type if present.
        2. "followup_question" - User is asking a specific question about previously mentioned venues.
        3. "conversation" - User is making small talk or saying something that doesn't require a search.
        4. "more_info" - User wants more details about a specific venue mentioned earlier.
        5. "help" - User is asking how to use the bot or what it can do.

        Return your analysis as a JSON object with the following fields:
        - intent: Either "restaurant_search", "followup_question", "conversation", "more_info", or "help"
        - location: Extracted location or null if not found
        - cuisine: Extracted cuisine or food/drink type or null if not found
        - restaurant_name: Specific venue name if mentioned or null
        - query: The core query, simplified for search if needed
        - language: The detected language of the query

        For restaurant_search, capture all relevant details like type of venue, price range, atmosphere, and special requirements.
        """

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
            f"👋 Hello {user.first_name}! I'm your restaurant recommendation assistant.\n\n"
            f"I can help you find great places to eat and drink, drawing from trusted sources "
            f"like Michelin Guide, Condé Nast, Food & Wine, and reputable food critics.\n\n"
            f"Simply ask me something like:\n"
            f"• \"Best sushi restaurants in Tokyo\"\n"
            f"• \"Romantic dinner spots in Paris\"\n"
            f"• \"Hidden cocktail bars in Manhattan\"\n\n"
            f"What are you looking for today?"
        )
        await update.message.reply_text(welcome_message)

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
            "• You can ask follow-up questions about the results\n"
            "• You can request more details about a specific restaurant\n\n"
            "I always search reputable sources like Michelin Guide, Condé Nast, and professional food critics - never crowd-sourced review sites!"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Main handler for user messages. Identifies intent and routes to appropriate handler.
        """
        user_message = update.message.text
        user_id = update.effective_user.id

        # Ensure user context exists
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "last_restaurants": [],
                "last_query": None,
                "last_location": None,
                "conversation_history": []
            }

        # Add message to conversation history (limited to last 5 messages)
        self.user_contexts[user_id]["conversation_history"].append(user_message)
        if len(self.user_contexts[user_id]["conversation_history"]) > 5:
            self.user_contexts[user_id]["conversation_history"].pop(0)

        # Send "typing" action to show the bot is processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            # Identify user intent using LLM
            intent_data = await self._identify_intent(user_message, user_id)

            # Log the identified intent
            logger.info(f"Identified intent: {intent_data}")

            # Route to appropriate handler based on intent
            if intent_data["intent"] == "restaurant_search":
                await self._handle_restaurant_search(update, context, intent_data)
            elif intent_data["intent"] == "followup_question":
                await self._handle_followup_question(update, context, intent_data)
            elif intent_data["intent"] == "more_info":
                await self._handle_more_info_request(update, context, intent_data)
            elif intent_data["intent"] == "help":
                await self.help_command(update, context)
            else:  # conversation intent
                await self._handle_conversation(update, context, intent_data)

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your request. Please try again."
            )

    async def _identify_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """
        Identify the intent behind a user message using OpenAI.

        Args:
            user_message: The user's message
            user_id: The user's ID for context

        Returns:
            Dictionary with intent classification and extracted entities
        """
        # Get conversation context
        user_context = self.user_contexts.get(user_id, {})

        # Add conversation context to help with intent recognition
        context_message = ""
        if user_context:
            if user_context.get("last_restaurants"):
                restaurant_names = [r.get("name", "") for r in user_context.get("last_restaurants", []) 
                                  if isinstance(r, dict) and "name" in r]
                if restaurant_names:
                    context_message += f"Recently mentioned restaurants: {', '.join(restaurant_names[:5])}\n"

            if user_context.get("last_query"):
                context_message += f"Last search query: {user_context.get('last_query')}\n"

            if user_context.get("last_location"):
                context_message += f"Last mentioned location: {user_context.get('last_location')}\n"

            if user_context.get("conversation_history"):
                recent_history = user_context.get("conversation_history", [])[-3:]
                if recent_history:
                    context_message += f"Recent conversation: {' | '.join(recent_history)}\n"

        try:
            # Create a customized prompt with context
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_intent_system_prompt()),
                ("human", f"User message: {user_message}\n\nConversation context:\n{context_message}")
            ])

            # Create a new chain with the context-aware prompt
            context_aware_chain = intent_prompt | self.llm | self.json_parser

            # Use the context-aware intent recognition chain
            intent_result = await context_aware_chain.ainvoke({})

            # Log the detected intent
            logger.info(f"AI detected intent: {intent_result.get('intent')}, language: {intent_result.get('language', 'unknown')}")

            # If successful, return the parsed intent
            return intent_result

        except Exception as e:
            logger.error(f"Error in intent recognition: {e}")
            # Fallback to simple pattern matching
            if re.search(r"(restaurant|food|eat|dining|dinner|lunch|breakfast|brunch|cafe|bar|drink)", 
                        user_message.lower()):
                return {
                    "intent": "restaurant_search",
                    "location": None,
                    "cuisine": None,
                    "restaurant_name": None,
                    "query": user_message,
                    "language": "English"  # Default fallback language
                }
            return {
                "intent": "conversation",
                "location": None,
                "cuisine": None,
                "restaurant_name": None,
                "query": user_message,
                "language": "English"  # Default fallback language
            }

    async def _handle_restaurant_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                      intent_data: Dict[str, Any]) -> None:
        """
        Handle a restaurant search request.

        Args:
            update: The update from Telegram
            context: The context from Telegram
            intent_data: Intent classification data
        """
        user_id = update.effective_user.id
        query = intent_data["query"]
        location = intent_data["location"]
        cuisine = intent_data["cuisine"]

        # Update user context
        self.user_contexts[user_id]["last_query"] = query
        self.user_contexts[user_id]["last_location"] = location

        # Send searching message
        searching_message = await update.message.reply_text(
            "🔍 Searching for recommendations... This may take a moment."
        )

        try:
            # Detect language using AI
            language = await self._detect_language(query)

            # Log the search
            logger.info(f"Searching for: query='{query}', location='{location}', cuisine='{cuisine}', detected language='{language}'")

            # Get recommendation using the recommender
            result = await self.recommender.get_recommendation_async(
                query=query,
                location=location or "",
                cuisine=cuisine or "",
                language=language
            )

            # Store the restaurant results in context for follow-up questions
            try:
                # Try to extract restaurant names from the compiled results if available
                if "compiled_results" in result and result["compiled_results"]:
                    # Try to parse the compiled results if they're a string (JSON)
                    if isinstance(result["compiled_results"], str):
                        import re
                        json_match = re.search(r'\[.*\]', result["compiled_results"], re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            restaurants = json.loads(json_str)
                            self.user_contexts[user_id]["last_restaurants"] = restaurants
                    # If compiled_results is already a list
                    elif isinstance(result["compiled_results"], list):
                        self.user_contexts[user_id]["last_restaurants"] = result["compiled_results"]
            except Exception as parse_error:
                logger.error(f"Error parsing restaurant results: {parse_error}")
                # Store the raw search results as fallback
                self.user_contexts[user_id]["last_restaurants"] = result.get("restaurant_results", [])

            # Delete the "searching" message
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=searching_message.message_id
            )

            # Get the formatted response
            response = result.get("formatted_response", "Sorry, no recommendations found.")

            # Split response into chunks if it's too long
            if len(response) > 4000:
                chunks = self._split_message(response)
                for chunk in chunks:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN
                    )
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=response,
                    parse_mode=ParseMode.MARKDOWN
                )

        except Exception as e:
            logger.error(f"Error during search: {e}")
            # Delete the "searching" message
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=searching_message.message_id
            )

            await update.message.reply_text(
                "Sorry, I encountered an error during the search. Please try again."
            )

    async def _handle_followup_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                      intent_data: Dict[str, Any]) -> None:
        """
        Handle follow-up questions about previously recommended restaurants.

        Args:
            update: The update from Telegram
            context: The context from Telegram
            intent_data: Intent classification data
        """
        user_id = update.effective_user.id
        user_context = self.user_contexts.get(user_id, {})
        query = intent_data["query"]
        restaurant_name = intent_data["restaurant_name"]

        # If we have a specific restaurant mentioned
        if restaurant_name and restaurant_name != "null":
            # Send typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )

            # Get last location for context
            location = user_context.get("last_location", "")

            # Create a specific query for this restaurant
            search_query = f"Information about {restaurant_name}"
            if location:
                search_query += f" in {location}"
            search_query += f": {query}"

            try:
                # Get language from intent data if available, otherwise detect it
                language = intent_data.get("language")
                if not language:
                    language = await self._detect_language(query)

                # Use Perplexity for verified information
                result = await self.recommender.get_recommendation_async(
                    query=search_query,
                    location=location or "",
                    cuisine="",
                    language=language
                )

                # Get the formatted response
                response = result.get("formatted_response", 
                                    f"Sorry, I couldn't find specific information about {restaurant_name}.")

                await update.message.reply_text(
                    response,
                    parse_mode=ParseMode.MARKDOWN
                )

            except Exception as e:
                logger.error(f"Error during follow-up search: {e}")
                await update.message.reply_text(
                    f"Sorry, I couldn't find the specific information about {restaurant_name} that you requested."
                )
        else:
            # General follow-up about previous results
            await self._handle_perplexity_search(update, context, query)

    async def _handle_more_info_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                      intent_data: Dict[str, Any]) -> None:
        """
        Handle requests for more details about a specific restaurant.

        Args:
            update: The update from Telegram
            context: The context from Telegram
            intent_data: Intent classification data
        """
        user_id = update.effective_user.id
        user_context = self.user_contexts.get(user_id, {})
        restaurant_name = intent_data["restaurant_name"]

        if not restaurant_name or restaurant_name == "null":
            await update.message.reply_text(
                "Could you please specify which restaurant you'd like more information about?"
            )
            return

        # Send typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Get last location for context
        location = user_context.get("last_location", "")

        try:
            # Get language from intent data if available, otherwise detect it
            language = intent_data.get("language")
            if not language:
                language = await self._detect_language(update.message.text)

            # Check if we have the restaurant in our context
            found_restaurant = None
            for restaurant in user_context.get("last_restaurants", []):
                if isinstance(restaurant, dict) and restaurant.get("name", "").lower() == restaurant_name.lower():
                    found_restaurant = restaurant
                    break

            if found_restaurant:
                # Use the restaurant data we already have
                detail_response = self._format_restaurant_details(found_restaurant)
                await update.message.reply_text(
                    detail_response,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # If not found in context, search for it
                search_query = f"Detailed information about {restaurant_name}"
                if location:
                    search_query += f" in {location}"

                # Use Perplexity for verified information
                result = await self.recommender.get_recommendation_async(
                    query=search_query,
                    location=location or "",
                    cuisine="",
                    language=language
                )

                # Get the formatted response
                response = result.get("formatted_response", 
                                    f"Sorry, I couldn't find specific information about {restaurant_name}.")

                await update.message.reply_text(
                    response,
                    parse_mode=ParseMode.MARKDOWN
                )

        except Exception as e:
            logger.error(f"Error during more info request: {e}")
            await update.message.reply_text(
                f"Sorry, I couldn't find more information about {restaurant_name}."
            )

    def _format_restaurant_details(self, restaurant: Dict[str, Any]) -> str:
        """
        Format detailed restaurant information from a restaurant object.

        Args:
            restaurant: Restaurant data dictionary

        Returns:
            Formatted restaurant details
        """
        # Extract available fields
        name = restaurant.get("name", "Unknown restaurant")
        description = restaurant.get("description", "No description available")
        address = restaurant.get("address", "Address not available")
        price_range = restaurant.get("price_range", "Price information not available")
        website = restaurant.get("website", "")

        # Handle recommended dishes
        dishes = restaurant.get("recommended_dishes", restaurant.get("signature_dishes", []))
        if isinstance(dishes, str):
            dishes_text = dishes
        elif isinstance(dishes, list) and dishes:
            dishes_text = ", ".join(dishes)
        else:
            dishes_text = "Not specified"

        # Handle opening hours
        hours = restaurant.get("opening_hours", "Hours not available")

        # Build the response
        response = f"*{name}*\n\n"
        response += f"*Description:* {description}\n\n"
        response += f"*Address:* {address}\n\n"
        response += f"*Price Range:* {price_range}\n\n"
        response += f"*Recommended Dishes:* {dishes_text}\n\n"
        response += f"*Opening Hours:* {hours}\n\n"

        if website:
            response += f"*Website:* {website}\n\n"

        return response

    async def _handle_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                 intent_data: Dict[str, Any]) -> None:
        """
        Handle conversational messages that don't require a search using OpenAI.

        Args:
            update: The update from Telegram
            context: The context from Telegram
            intent_data: Intent classification data
        """
        query = intent_data["query"]
        user_id = update.effective_user.id

        # Get conversation history for context
        history = self.user_contexts.get(user_id, {}).get("conversation_history", [])

        # Get language from intent data if available
        language = intent_data.get("language")

        try:
            # Use OpenAI to generate a contextual response
            conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly restaurant recommendation assistant. 
                You should respond conversationally to the user's message in their original language.
                Keep responses concise and focused on helping users find great places to eat.
                If the user is asking for restaurant recommendations, politely redirect them to be more specific about what they're looking for.
                Don't make up information about restaurants - you should only provide verified information from searches.
                """),
                ("human", f"User message: {query}\nUser message language: {language or 'unknown'}\nRecent conversation history: {history[-3:] if len(history) > 0 else 'None'}")
            ])

            conversation_chain = conversation_prompt | self.llm | StrOutputParser()
            response = await conversation_chain.ainvoke({})

            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            # Fallback to simple responses if AI generation fails

            # Check for simple patterns as fallback
            if any(word in query.lower() for word in ["thank", "thanks", "thx"]):
                await update.message.reply_text(
                    "You're welcome! Feel free to ask me for any restaurant recommendations."
                )
            elif any(word in query.lower() for word in ["hi", "hello", "hey", "greetings"]):
                await update.message.reply_text(
                    "Hello! How can I help you find great restaurants today?"
                )
            elif any(word in query.lower() for word in ["bye", "goodbye", "farewell"]):
                await update.message.reply_text(
                    "Goodbye! Feel free to come back when you need restaurant recommendations."
                )
            else:
                await update.message.reply_text(
                    "I'm specialized in finding restaurant recommendations from trusted sources. "
                    "How can I help you discover great places to eat or drink today?"
                )

    async def _handle_perplexity_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, language: str = None) -> None:
        """
        Handle a general search using Perplexity for verified information.

        Args:
            update: The update from Telegram
            context: The context from Telegram
            query: The search query
            language: Detected language (or None to detect automatically)
        """
        user_id = update.effective_user.id
        user_context = self.user_contexts.get(user_id, {})
        location = user_context.get("last_location", "")

        # Send typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        searching_message = await update.message.reply_text(
            "🔍 Searching for that information... This may take a moment."
        )

        try:
            # Detect language if not provided
            if not language:
                language = await self._detect_language(query)

            # Use Perplexity for verified information
            result = await self.recommender.get_recommendation_async(
                query=query,
                location=location or "",
                cuisine="",
                language=language
            )

            # Delete the "searching" message
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=searching_message.message_id
            )

            # Get the formatted response
            response = result.get("formatted_response", "Sorry, I couldn't find that information.")

            # Split response into chunks if it's too long
            if len(response) > 4000:
                chunks = self._split_message(response)
                for chunk in chunks:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN
                    )
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=response,
                    parse_mode=ParseMode.MARKDOWN
                )

        except Exception as e:
            logger.error(f"Error during Perplexity search: {e}")
            # Delete the "searching" message
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=searching_message.message_id
            )

            await update.message.reply_text(
                "Sorry, I couldn't find the information you requested. Could you try asking in a different way?"
            )

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
                ("system", "You are a language detection specialist. Identify the language of the given text and respond with only the language name in English (e.g., 'English', 'French', 'Spanish', 'German', etc.)."),
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