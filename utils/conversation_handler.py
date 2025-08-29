# utils/conversation_handler.py - UPDATED with AI-powered destination detection
"""
ENHANCED: Conversation Handler with AI-Powered Destination Change Detection

Key Enhancement:
- Replaced regex-based destination detection with intelligent AI analysis
- More accurate detection of when users change destinations
- Better handling of edge cases and ambiguous queries
"""

import json
import logging
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from langchain.schema import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from location.location_analyzer import LocationAnalyzer
from utils.ai_destination_detector import AIDestinationChangeDetector

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the bot can handle"""
    RESTAURANT_REQUEST = "restaurant_request"
    GENERAL_QUESTION = "general_question" 
    UNRELATED = "unrelated"
    LOCATION_BASED_AMBIGUOUS = "location_based_ambiguous"

class RequestType(Enum):
    """Types of restaurant requests"""
    CITY_WIDE = "city_wide"
    LOCATION_BASED_NEARBY = "location_based_nearby"
    LOCATION_BASED_GEOGRAPHIC = "location_based_geographic"
    FOLLOW_UP = "follow_up"
    GOOGLE_MAPS_MORE = "google_maps_more"  # NEW: Request more options via Google Maps

class ConversationState(Enum):
    """Conversation states"""
    IDLE = "idle"
    AWAITING_LOCATION = "awaiting_location"
    SEARCHING = "searching"
    AWAITING_CHOICE = "awaiting_choice"
    RESULTS_SHOWN = "results_shown"
    AWAITING_LOCATION_CLARIFICATION = "awaiting_location_clarification"  # for ambiguous locations


class CentralizedConversationHandler:
    """
    Central AI conversation handler that analyzes all messages and routes
    to appropriate flows with proper state management.
    """

    def __init__(self, config):
        self.config = config
        self.conversation_ai = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Initialize location analyzer for ambiguity detection
        self.location_analyzer = LocationAnalyzer(config)

        # ENHANCED: Initialize AI-powered destination change detector
        self.destination_detector = AIDestinationChangeDetector(config)

        # State tracking
        self.user_states: Dict[int, ConversationState] = {}
        self.user_conversations: Dict[int, List[Dict[str, Any]]] = {}
        self.user_contexts: Dict[int, Dict[str, Any]] = {}

        # Location context storage with AI-powered change detection
        self.location_search_context: Dict[int, Dict[str, Any]] = {}
        self.ambiguous_location_context: Dict[int, Dict[str, Any]] = {}

        # Build AI prompts
        self._build_prompts()

        logger.info("✅ Centralized Conversation Handler initialized with AI destination detection")

    def store_ambiguous_location_context(self, user_id: int, context: Dict[str, Any]):
        """Store context for ambiguous location clarification"""
        self.ambiguous_location_context[user_id] = {
            "original_query": context.get("query", ""),
            "location_detected": context.get("location_detected", ""),
            "ambiguity_reason": context.get("ambiguity_reason", ""),
            "timestamp": time.time()
        }
        self.user_states[user_id] = ConversationState.AWAITING_LOCATION_CLARIFICATION

    def handle_location_clarification(self, user_id: int, clarification_text: str) -> Dict[str, Any]:
        """
        Handle user's clarification of ambiguous location

        Args:
            user_id: Telegram user ID
            clarification_text: User's clarification (e.g., "Massachusetts", "the one in UK")

        Returns:
            Dict with processed clarification and next action
        """
        try:
            # Get stored context
            context = self.ambiguous_location_context.get(user_id)
            if not context:
                return {
                    "action": "CLARIFY",
                    "bot_response": "I'm not sure what location you're referring to. Could you please search again?",
                    "needs_clarification": True
                }

            # Clear the stored context
            del self.ambiguous_location_context[user_id]
            self.user_states[user_id] = ConversationState.IDLE

            # Combine original location with clarification
            original_location = context["location_detected"]
            combined_location = f"{original_location}, {clarification_text}"
            original_query = context["original_query"]

            logger.info(f"Location clarified: '{original_location}' + '{clarification_text}' = '{combined_location}'")

            return {
                "action": "SEARCH_LOCATION",
                "bot_response": f"Perfect! Searching for {original_query} in {combined_location}...",
                "search_query": f"{original_query} in {combined_location}",
                "clarified_location": combined_location,
                "needs_clarification": False
            }

        except Exception as e:
            logger.error(f"Error handling location clarification: {e}")
            return {
                "action": "CLARIFY", 
                "bot_response": "Sorry, I had trouble understanding that. Could you search again?",
                "needs_clarification": True
            }

    def handle_ambiguous_location(self, user_id: int, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ambiguous location detection by letting AI generate clarification message

        Args:
            user_id: Telegram user ID
            analysis_result: Full analysis result from LocationAnalyzer

        Returns:
            Dict with AI-generated clarification response
        """
        try:
            # Store context for clarification
            self.store_ambiguous_location_context(user_id, {
                "query": analysis_result.get("original_message", ""),
                "location_detected": analysis_result.get("location_detected", ""),
                "ambiguity_reason": analysis_result.get("ambiguity_reason", "")
            })

            # Return the action that will be used by telegram bot
            return {
                "action": "REQUEST_LOCATION_CLARIFICATION",
                "bot_response": analysis_result.get("suggested_response", "Could you be more specific about the location?"),
                "needs_clarification": True,
                "analysis_result": analysis_result
            }

        except Exception as e:
            logger.error(f"Error handling ambiguous location: {e}")
            return {
                "action": "CLARIFY",
                "bot_response": "Could you be more specific about the location you're looking for?",
                "needs_clarification": True
            }

    async def _detect_destination_change_ai(self, user_id: int, new_message: str) -> bool:
        """
        ENHANCED: AI-powered detection of destination changes

        Returns True if destination has changed and context should be cleared
        """
        try:
            # Get current location context
            current_context = self.location_search_context.get(user_id)
            if not current_context:
                logger.info(f"No location context found for user {user_id} - no change to detect")
                return False

            # Use AI to analyze destination change
            detection_result = self.destination_detector.detect_destination_change(
                current_message=new_message,
                stored_context=current_context
            )

            destination_changed = detection_result.get("destination_changed", False)
            confidence = detection_result.get("confidence", 0.0)
            reasoning = detection_result.get("reasoning", "No reasoning provided")

            if destination_changed:
                old_location = detection_result.get("old_location", "unknown")
                new_location = detection_result.get("new_location", "unknown")
                logger.info(f"🤖 AI detected destination change for user {user_id}:")
                logger.info(f"   From: {old_location} → To: {new_location}")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Reasoning: {reasoning}")
            else:
                logger.info(f"🤖 AI determined same destination for user {user_id}: {reasoning}")

            return destination_changed

        except Exception as e:
            logger.error(f"❌ Error in AI destination change detection for user {user_id}: {e}")
            # Conservative fallback: don't clear context if AI fails
            return False

    def _detect_destination_change(self, user_id: int, new_message: str) -> bool:
        """
        Synchronous wrapper for AI destination change detection
        """
        import asyncio
        try:
            # Try to run the async AI detection
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._detect_destination_change_ai(user_id, new_message))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self._detect_destination_change_ai(user_id, new_message))
        except Exception as e:
            logger.error(f"❌ Failed to run AI destination detection: {e}")
            # Fallback to conservative behavior
            return False
    
    def _build_prompts(self):
        """Build all AI prompts for different analysis stages"""

        # Main conversation analysis prompt
        self.main_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_main_system_prompt()),
            ("human", """
CONVERSATION HISTORY:
{conversation_history}

CURRENT USER MESSAGE: {user_message}

USER STATE: {user_state}

LOCATION CONTEXT: {location_context}
""")
        ])

        # Chain for main conversation handling
        self.main_chain = self.main_prompt | self.conversation_ai

    def _get_main_system_prompt(self) -> str:
        """Enhanced system prompt with ambiguity handling"""
        return """
    You are an AI conversation analyzer for a restaurant recommendation bot with location ambiguity handling.

    ANALYZE each message and determine the appropriate response and action.

    USER STATES:
    - idle: Normal conversation 
    - awaiting_location: User needs to provide location for "near me" searches
    - awaiting_location_clarification: User needs to clarify ambiguous location
    - results_shown: Results were just shown, user can ask for more
    - searching: Currently searching (handled separately)

    QUERY TYPES:
    1. RESTAURANT_REQUEST - User wants restaurant recommendations
       a. city_wide: "best sushi in Tokyo" → SEARCH_CITY
       b. location_based_nearby: "pizza near me" → REQUEST_LOCATION  
       c. location_based_geographic: "bars in SoHo" → SEARCH_LOCATION
       d. follow_up: "show more" when results_shown → GOOGLE_MAPS_MORE
       e. location_based_ambiguous: "restaurants in Springfield" → REQUEST_LOCATION_CLARIFICATION


    2. GENERAL_QUESTION - Questions about food/restaurants/chefs but not requesting recommendations
       Examples: "How many Michelin restaurants are in Rome?", "Who is Gordon Ramsay?"
       → WEB_SEARCH

    3. UNRELATED - Nothing to do with restaurants/food/cuisine
       → REDIRECT

    4. LOCATION_CLARIFICATION - User responding to ambiguous location request
   Examples: "Massachusetts" when user_state = "awaiting_location_clarification"
   → HANDLE_LOCATION_CLARIFICATION

    SPECIAL HANDLING:
    - If user_state is "awaiting_location_clarification", treat ANY response as location clarification
    - When user_state is "results_shown" and user asks for more options, use GOOGLE_MAPS_MORE action
    - When location is ambiguous, generate smart clarification questions based on the specific ambiguity

    AMBIGUOUS LOCATION EXAMPLES:
    - Springfield (multiple US cities): Ask which state
    - Cambridge (UK vs Massachusetts): Give specific options
    - Generic ambiguity: Ask for city/state/country clarification

    RESPONSE FORMAT (JSON only):
    {{
        "query_type": "restaurant_request" | "general_question" | "unrelated" | "location_clarification" | "location_ambiguous",
        "request_type": "city_wide" | "location_based_nearby" | "location_based_geographic" | "follow_up" | null,
        "action": "SEARCH_CITY" | "REQUEST_LOCATION" | "SEARCH_LOCATION" | "GOOGLE_MAPS_MORE" | "WEB_SEARCH" | "CLARIFY" | "REDIRECT" | "HANDLE_LOCATION_CLARIFICATION" | "REQUEST_LOCATION_CLARIFICATION",
        "bot_response": "what to say to the user (conversational, friendly)",
        "search_query": "search query if action requires search",
        "needs_clarification": true|false,
        "missing_info": ["city", "cuisine", "location"],
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation of decision"
    }}

    EXAMPLES:

    User: "Massachusetts" (when user_state = "awaiting_location_clarification")
    → {{"query_type": "location_clarification", "action": "HANDLE_LOCATION_CLARIFICATION", "bot_response": "Perfect! Searching in Massachusetts...", "needs_clarification": false, "confidence": 0.9}}

    "best steakhouses in Dubai" → {{"action": "SEARCH_CITY", "search_query": "best steakhouses in Dubai"}}
    "restaurants in Paris" → {{"action": "SEARCH_CITY", "search_query": "restaurants in Paris"}}
    "bars in London" → {{"action": "SEARCH_CITY", "search_query": "bars in London"}}

    "restaurants in SoHo" → {{"action": "SEARCH_LOCATION", "search_query": "restaurants in SoHo"}}
    "bars near me" → {{"action": "REQUEST_LOCATION", "search_query": "bars"}}
    "restaurants in Chinatown" → {{"action": "SEARCH_LOCATION", "search_query": "restaurants in Chinatown"}}

    User: "show me more" (when user_state = "results_shown")
    → {{"query_type": "restaurant_request", "request_type": "follow_up", "action": "GOOGLE_MAPS_MORE", "bot_response": "Great! I'll find more restaurants in that area for you.", "needs_clarification": false, "confidence": 0.9}}

    For ambiguous locations, create smart clarification messages like:
    - "I found multiple places called Springfield. Which state did you mean?"
    - "Which Cambridge - the one in England or Massachusetts?"
    - "Could you be more specific about which {{location}} you meant? (city, state, or country)"
    """

    def process_message(
        self, 
        message_text: str, 
        user_id: int, 
        chat_id: int,
        is_voice: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point: Process any message (text or transcribed voice)

        Returns:
        {
            "response_needed": bool,
            "bot_response": str,
            "action": str,
            "action_data": dict,
            "new_state": ConversationState
        }
        """
        try:
            # Step 1: Add message to conversation history
            if self._detect_destination_change(user_id, message_text):
                logger.info(f"🤖 AI detected destination change - clearing location context for user {user_id}")
                self.clear_location_search_context(user_id)
            
            self._add_to_conversation(user_id, message_text, is_user=True)

            # Step 2: Get current state and context
            current_state = self.user_states.get(user_id, ConversationState.IDLE)
            conversation_history = self._format_conversation_history(user_id)
            location_context = self._format_location_context(user_id)

            # Step 3: Analyze message with AI
            ai_response = self.main_chain.invoke({
                "conversation_history": conversation_history,
                "user_message": message_text,
                "user_state": current_state.value,
                "location_context": location_context
            })

            # Step 4: Parse AI response
            analysis = self._parse_ai_response(ai_response.content)

            # Step 5: Determine action and new state
            action_result = self._determine_action(analysis, user_id, current_state)

            # Step 6: Update user state
            new_state = action_result.get("new_state", current_state)
            self.user_states[user_id] = new_state

            # Step 7: Add bot response to conversation
            if action_result.get("bot_response"):
                self._add_to_conversation(user_id, action_result["bot_response"], is_user=False)

            logger.info(f"Processed message for user {user_id}: {analysis.get('action')} -> {new_state.value}")

            return action_result

        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}")
            return {
                "response_needed": True,
                "bot_response": "I had trouble understanding that. Could you tell me what restaurants you're looking for?",
                "action": "ERROR",
                "action_data": {},
                "new_state": ConversationState.IDLE
            }

    def _parse_ai_response(self, response_content) -> Dict[str, Any]:
        """Parse AI response from JSON"""
        try:
            # Handle different response content types
            if hasattr(response_content, 'content'):
                content = response_content.content
            else:
                content = str(response_content)

            # Handle code block formatting
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            analysis = json.loads(content)

            # Validate required fields
            required_fields = ["query_type", "action", "bot_response"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = self._get_default_value(field)

            return analysis

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing AI response: {e}")
            return {
                "query_type": "restaurant_request",
                "action": "CLARIFY",
                "bot_response": "I'm not sure what you're looking for. Could you tell me what kind of restaurant you'd like to find?",
                "needs_clarification": True,
                "confidence": 0.0
            }

    def _determine_action(
        self, 
        analysis: Dict[str, Any], 
        user_id: int, 
        current_state: ConversationState
    ) -> Dict[str, Any]:
        """Determine what action to take based on AI analysis"""

        action = analysis.get("action", "CLARIFY")
        bot_response = analysis.get("bot_response", "How can I help you with restaurants?")

        # Base result structure
        result = {
            "response_needed": True,
            "bot_response": bot_response,
            "action": action,
            "action_data": {},
            "new_state": current_state
        }

        # Route based on action
        if action == "SEARCH_CITY":
            # City-wide restaurant search
            result.update({
                "action": "LAUNCH_CITY_SEARCH",
                "action_data": {
                    "search_query": analysis.get("search_query"),
                    "request_type": "city_wide"
                },
                "new_state": ConversationState.SEARCHING
            })

        elif action == "REQUEST_LOCATION":
            # Need user's physical location
            result.update({
                "action": "REQUEST_USER_LOCATION",
                "action_data": {
                    "context": analysis.get("search_query", "restaurants")
                },
                "new_state": ConversationState.AWAITING_LOCATION
            })

        elif action == "SEARCH_LOCATION":
            # Geographic location search
            result.update({
                "action": "LAUNCH_LOCATION_SEARCH", 
                "action_data": {
                    "search_query": analysis.get("search_query"),
                    "request_type": "location_based_geographic"
                },
                "new_state": ConversationState.SEARCHING
            })

        elif action == "GOOGLE_MAPS_MORE":
            # Google Maps search for more options in same location
            result.update({
                "action": "LAUNCH_GOOGLE_MAPS_SEARCH",
                "action_data": {
                    "search_type": "google_maps_more"
                },
                "new_state": ConversationState.SEARCHING
            })
        elif action == "WEB_SEARCH":
            # General question - web search (planned)
            result.update({
                "action": "LAUNCH_WEB_SEARCH",
                "action_data": {
                    "search_query": analysis.get("search_query")
                },
                "new_state": ConversationState.IDLE
            })

        elif action == "REDIRECT":
            # Unrelated query
            result.update({
                "action": "SEND_REDIRECT",
                "action_data": {},
                "new_state": ConversationState.IDLE
            })

        elif action == "REQUEST_LOCATION_CLARIFICATION":
            # Location is ambiguous - ask for clarification
            result.update({
                "action": "SEND_LOCATION_CLARIFICATION",
                "action_data": {
                    "analysis_result": analysis.get("analysis_result", {})
                },
                "new_state": ConversationState.AWAITING_LOCATION_CLARIFICATION
            })

        elif action == "HANDLE_LOCATION_CLARIFICATION":
            # User provided clarification for ambiguous location
            result.update({
                "action": "PROCESS_LOCATION_CLARIFICATION",
                "action_data": {
                    "clarification_text": analysis.get("search_query", "")
                },
                "new_state": ConversationState.SEARCHING
            })

        else:
            # CLARIFY or unknown
            result.update({
                "action": "SEND_CLARIFICATION",
                "action_data": {},
                "new_state": ConversationState.IDLE
            })

        

        return result

    def _add_to_conversation(self, user_id: int, message: str, is_user: bool = True):
        """Add message to user's conversation history"""
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []

        self.user_conversations[user_id].append({
            "role": "user" if is_user else "assistant",
            "message": message,
            "timestamp": time.time()
        })

        # Keep only last 10 messages
        if len(self.user_conversations[user_id]) > 10:
            self.user_conversations[user_id] = self.user_conversations[user_id][-10:]

    def _format_conversation_history(self, user_id: int) -> str:
        """Format conversation history for AI prompt"""
        if user_id not in self.user_conversations:
            return "No previous conversation."

        formatted = []
        for msg in self.user_conversations[user_id]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['message']}")

        return "\n".join(formatted)

    def _format_location_context(self, user_id: int) -> str:
        """Format location search context for AI prompt"""
        if user_id not in self.location_search_context:
            return "No recent location searches."

        context = self.location_search_context[user_id]
        time_ago = time.time() - context.get("last_search_time", 0)

        if time_ago > 1800:  # 30 minutes
            return "No recent location searches."

        return f"Recent location search: '{context.get('query', '')}' at {context.get('location_description', 'unknown location')} ({int(time_ago/60)} minutes ago)"

    def store_location_search_context(
        self, 
        user_id: int, 
        query: str, 
        location_data: Any, 
        location_description: str,
        coordinates: Optional[Tuple[float, float]] = None
    ) -> None:
        """FIXED: Simplified coordinate storage"""
        try:
            # Extract coordinates with single priority order
            final_coordinates = None

            # Priority 1: Explicit coordinates parameter (from orchestrator result)
            if coordinates and len(coordinates) == 2:
                try:
                    final_coordinates = (float(coordinates[0]), float(coordinates[1]))
                    logger.info(f"✅ Using explicit coordinates: {final_coordinates}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"❌ Invalid explicit coordinates: {coordinates} - {e}")

            # Priority 2: From location_data object (from geocoding)
            if not final_coordinates and location_data:
                if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                    if location_data.latitude is not None and location_data.longitude is not None:
                        try:
                            final_coordinates = (float(location_data.latitude), float(location_data.longitude))
                            logger.info(f"✅ Using location_data coordinates: {final_coordinates}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"❌ Invalid location_data coordinates: {e}")

            # Validate coordinates
            if final_coordinates:
                from location.location_utils import LocationUtils
                if not LocationUtils.validate_coordinates(final_coordinates[0], final_coordinates[1]):
                    logger.error(f"❌ Coordinates failed validation: {final_coordinates}")
                    final_coordinates = None

            # Store SIMPLE context - remove all the redundant storage
            context_data = {
                "query": query,
                "location_data": location_data,
                "location_description": location_description,
                "coordinates": final_coordinates,  # Single source of truth
                "last_search_time": time.time()
            }

            self.location_search_context[user_id] = context_data

            # Log storage result
            if final_coordinates:
                logger.info(f"✅ Stored coordinates for user {user_id}: {final_coordinates}")
            else:
                logger.warning(f"⚠️ No coordinates stored for user {user_id}")

        except Exception as e:
            logger.error(f"❌ Error storing location context: {e}")

    def get_location_search_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """FIXED: Simplified coordinate retrieval"""
        try:
            context = self.location_search_context.get(user_id)
            if not context:
                logger.info(f"No location context found for user {user_id}")
                return None

            # Check if context is still valid (within 30 minutes)
            time_ago = time.time() - context.get("last_search_time", 0)
            if time_ago > 1800:  # 30 minutes
                logger.info(f"Location context expired for user {user_id}")
                del self.location_search_context[user_id]
                return None

            # Log retrieval
            coordinates = context.get("coordinates")
            if coordinates:
                logger.info(f"✅ Retrieved coordinates for user {user_id}: {coordinates}")
            else:
                logger.warning(f"⚠️ No coordinates in context for user {user_id}")

            return context

        except Exception as e:
            logger.error(f"❌ Error retrieving location context for user {user_id}: {e}")
            return None

    def clear_location_search_context(self, user_id: int) -> None:
        """Clear stored location context for a user - KEEP EXISTING METHOD"""
        try:
            if user_id in self.location_search_context:
                del self.location_search_context[user_id]
                logger.info(f"✅ Cleared location context for user {user_id}")
            else:
                logger.info(f"No location context to clear for user {user_id}")
        except Exception as e:
            logger.error(f"❌ Error clearing location context for user {user_id}: {e}")

    def debug_coordinate_storage(self, user_id: int) -> Dict[str, Any]:
        """Debug coordinate storage for a user"""
        try:
            # FIXED: Use existing attribute name
            context = self.location_search_context.get(user_id, {})

            debug_info = {
                "has_context": user_id in self.location_search_context,
                "context_keys": list(context.keys()),
                "coordinates": context.get("coordinates"),
                "lat": context.get("lat"),
                "lng": context.get("lng"),
                "lat_lng_tuple": context.get("lat_lng_tuple"),
                "coordinate_string": context.get("coordinate_string"),
                "coordinate_source": context.get("coordinate_source"),
                "location_data_type": type(context.get("location_data")).__name__ if context.get("location_data") else None,
                "last_search_time": context.get("last_search_time")
            }

            logger.info(f"🔍 COORDINATE STORAGE DEBUG for user {user_id}:")
            for key, value in debug_info.items():
                logger.info(f"   {key}: {value}")

            return debug_info

        except Exception as e:
            logger.error(f"❌ Error in coordinate debug for user {user_id}: {e}")
            return {"error": str(e)}
    
    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing fields"""
        defaults = {
            "query_type": "restaurant_request",
            "action": "CLARIFY",
            "bot_response": "How can I help you find restaurants?",
            "needs_clarification": True,
            "confidence": 0.0
        }
        return defaults.get(field, "")

    def validate_ai_destination_detection(self) -> bool:
        """
        Validate the AI destination detection system
        Returns True if validation passes
        """
        return self.destination_detector.validate_detection_logic()

    # State management methods
    def set_user_state(self, user_id: int, state: ConversationState):
        """Set user's conversation state"""
        self.user_states[user_id] = state
        logger.debug(f"Set user {user_id} state to {state.value}")

    def get_user_state(self, user_id: int) -> ConversationState:
        """Get user's current conversation state"""
        return self.user_states.get(user_id, ConversationState.IDLE)

    def clear_user_data(self, user_id: int):
        """Clear all data for a user"""
        if user_id in self.user_states:
            del self.user_states[user_id]
        if user_id in self.user_conversations:
            del self.user_conversations[user_id]
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]
        if user_id in self.location_search_context:
            del self.location_search_context[user_id]
        if user_id in self.ambiguous_location_context:
            del self.ambiguous_location_context[user_id]
        logger.debug(f"Cleared all data for user {user_id}")