# utils/conversation_handler.py
"""
Centralized AI Conversation Handler

Handles all conversational AI logic for the telegram bot with intelligent routing
between different search flows and query types.

FIXED: Corrected syntax errors, type issues, and brace problems from previous version.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
        """Initialize with enhanced state tracking for location clarification"""
        self.config = config

        # Initialize AI model
        self.conversation_ai = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )

        # User state tracking with location context
        self.user_states = {}  # user_id -> ConversationState
        self.user_contexts = {}  # user_id -> context dict
        self.location_search_context = {}  # user_id -> location search context
        self.ambiguous_location_context = {}  # user_id -> clarification context
        self.user_conversations = {}


        # Create conversation analysis prompt
        self._build_prompts()

        logger.info("✅ Centralized Conversation Handler initialized with ambiguity support")

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

    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response from JSON"""
        try:
            # Handle code block formatting
            content = response_content.strip()
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

    # conversation_handler.py - COORDINATE STORAGE FIX

    # Update the store_location_search_context method to ensure coordinates are properly preserved:

    def store_location_search_context(
        self, 
        user_id: int, 
        query: str, 
        location_data: Any, 
        location_description: str,
        coordinates: Optional[Tuple[float, float]] = None  # NEW: Explicit coordinate parameter
    ) -> None:
        """
        Store location search context for follow-up queries
        FIXED: Enhanced coordinate storage to prevent loss
        """
        try:
            # Extract coordinates from multiple sources with priority order
            final_coordinates = None
            coordinate_source = "none"

            # Priority 1: Explicit coordinates parameter (most reliable)
            if coordinates and len(coordinates) == 2:
                try:
                    final_coordinates = (float(coordinates[0]), float(coordinates[1]))
                    coordinate_source = "explicit_parameter"
                    logger.info(f"✅ Using explicit coordinates: {final_coordinates[0]:.6f}, {final_coordinates[1]:.6f}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"❌ Invalid explicit coordinates: {coordinates} - {e}")

            # Priority 2: From location_data object
            if not final_coordinates and location_data:
                if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                    if location_data.latitude is not None and location_data.longitude is not None:
                        try:
                            final_coordinates = (float(location_data.latitude), float(location_data.longitude))
                            coordinate_source = "location_data_object"
                            logger.info(f"✅ Extracted from location_data: {final_coordinates[0]:.6f}, {final_coordinates[1]:.6f}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"❌ Invalid location_data coordinates: lat={location_data.latitude}, lng={location_data.longitude} - {e}")

            # Validate extracted coordinates
            if final_coordinates:
                from location.location_utils import LocationUtils
                if not LocationUtils.validate_coordinates(final_coordinates[0], final_coordinates[1]):
                    logger.error(f"❌ Coordinates failed validation: {final_coordinates}")
                    final_coordinates = None
                    coordinate_source = "validation_failed"

            # Store comprehensive context with redundant coordinate storage
            context_data = {
                "query": query,
                "location_data": location_data,
                "location_description": location_description,
                "coordinates": final_coordinates,  # Primary coordinate storage
                "coordinate_source": coordinate_source,
                "timestamp": time.time(),

                # REDUNDANT STORAGE: Multiple coordinate representations for reliability
                "lat": final_coordinates[0] if final_coordinates else None,
                "lng": final_coordinates[1] if final_coordinates else None,
                "lat_lng_tuple": final_coordinates,
                "coordinate_string": f"{final_coordinates[0]:.6f},{final_coordinates[1]:.6f}" if final_coordinates else None,

                # Backup location information
                "location_backup": {
                    "description": location_description,
                    "type": getattr(location_data, 'location_type', 'unknown') if location_data else 'unknown',
                    "confidence": getattr(location_data, 'confidence', 0.0) if location_data else 0.0
                }
            }

            # Store in user location context
            if user_id not in self.user_location_context:
                self.user_location_context[user_id] = {}

            self.user_location_context[user_id].update(context_data)

            # Log successful storage
            logger.info(f"✅ LOCATION CONTEXT STORED for user {user_id}:")
            logger.info(f"   Query: {query}")
            logger.info(f"   Location: {location_description}")
            if final_coordinates:
                logger.info(f"   Coordinates: {final_coordinates[0]:.6f}, {final_coordinates[1]:.6f} (from {coordinate_source})")
            else:
                logger.warning(f"   No coordinates stored - source: {coordinate_source}")

            # Additional validation log
            logger.info(f"   Context keys stored: {list(context_data.keys())}")

        except Exception as e:
            logger.error(f"❌ Error storing location context: {e}")
            logger.error(f"   user_id: {user_id}, query: {query}")
            logger.error(f"   location_data: {location_data}")
            logger.error(f"   coordinates: {coordinates}")

    def get_location_search_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get stored location search context
        FIXED: Enhanced coordinate retrieval with multiple fallbacks
        """
        try:
            context = self.user_location_context.get(user_id)
            if not context:
                logger.info(f"No location context found for user {user_id}")
                return None

            # Validate and repair coordinates if needed
            coordinates = context.get("coordinates")

            # Try multiple coordinate sources if primary is missing
            if not coordinates:
                logger.warning(f"Primary coordinates missing, trying fallbacks for user {user_id}")

                # Fallback 1: Individual lat/lng fields
                lat = context.get("lat")
                lng = context.get("lng")
                if lat is not None and lng is not None:
                    coordinates = (lat, lng)
                    context["coordinates"] = coordinates  # Repair the context
                    logger.info(f"✅ Repaired coordinates from lat/lng: {lat:.6f}, {lng:.6f}")

                # Fallback 2: Tuple field
                elif context.get("lat_lng_tuple"):
                    coordinates = context["lat_lng_tuple"]
                    context["coordinates"] = coordinates  # Repair the context
                    logger.info(f"✅ Repaired coordinates from tuple: {coordinates[0]:.6f}, {coordinates[1]:.6f}")

                # Fallback 3: Extract from location_data
                elif context.get("location_data"):
                    location_data = context["location_data"]
                    if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                        if location_data.latitude is not None and location_data.longitude is not None:
                            coordinates = (location_data.latitude, location_data.longitude)
                            context["coordinates"] = coordinates  # Repair the context
                            logger.info(f"✅ Repaired coordinates from location_data: {coordinates[0]:.6f}, {coordinates[1]:.6f}")

            # Final validation
            if coordinates:
                from location.location_utils import LocationUtils
                if not LocationUtils.validate_coordinates(coordinates[0], coordinates[1]):
                    logger.error(f"❌ Stored coordinates are invalid: {coordinates}")
                    context["coordinates"] = None
                    coordinates = None

            # Log context retrieval
            logger.info(f"📍 LOCATION CONTEXT RETRIEVED for user {user_id}:")
            logger.info(f"   Query: {context.get('query', 'unknown')}")
            logger.info(f"   Location: {context.get('location_description', 'unknown')}")
            if coordinates:
                logger.info(f"   Coordinates: {coordinates[0]:.6f}, {coordinates[1]:.6f}")
            else:
                logger.warning(f"   No valid coordinates available")

            return context

        except Exception as e:
            logger.error(f"❌ Error retrieving location context for user {user_id}: {e}")
            return None

    # ADDITIONAL METHOD: Clear location context when coordinates become unreliable
    def clear_location_context(self, user_id: int) -> None:
        """Clear stored location context for a user"""
        try:
            if user_id in self.user_location_context:
                del self.user_location_context[user_id]
                logger.info(f"✅ Cleared location context for user {user_id}")
            else:
                logger.info(f"No location context to clear for user {user_id}")
        except Exception as e:
            logger.error(f"❌ Error clearing location context for user {user_id}: {e}")

    # DIAGNOSTIC METHOD: Debug coordinate storage
    def debug_coordinate_storage(self, user_id: int) -> Dict[str, Any]:
        """Debug coordinate storage for a user"""
        try:
            context = self.user_location_context.get(user_id, {})

            debug_info = {
                "has_context": user_id in self.user_location_context,
                "context_keys": list(context.keys()),
                "coordinates": context.get("coordinates"),
                "lat": context.get("lat"),
                "lng": context.get("lng"),
                "lat_lng_tuple": context.get("lat_lng_tuple"),
                "coordinate_string": context.get("coordinate_string"),
                "coordinate_source": context.get("coordinate_source"),
                "location_data_type": type(context.get("location_data")).__name__ if context.get("location_data") else None,
                "timestamp": context.get("timestamp")
            }

            logger.info(f"🔍 COORDINATE STORAGE DEBUG for user {user_id}:")
            for key, value in debug_info.items():
                logger.info(f"   {key}: {value}")

            return debug_info

        except Exception as e:
            logger.error(f"❌ Error in coordinate debug for user {user_id}: {e}")
            return {"error": str(e)}

    def get_location_search_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get stored location search context for Google Maps follow-up"""
        context = self.location_search_context.get(user_id)
        if context:
            # Check if context is still valid (within 30 minutes)
            time_ago = time.time() - context.get("last_search_time", 0)
            if time_ago <= 1800:  # 30 minutes
                return context
            else:
                # Context expired, remove it
                del self.location_search_context[user_id]
        return None

    def clear_location_search_context(self, user_id: int):
        """Clear location search context for a user"""
        if user_id in self.location_search_context:
            del self.location_search_context[user_id]

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