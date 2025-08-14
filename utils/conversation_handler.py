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
    RESULTS_SHOWN = "results_shown"  # NEW: Location results shown, can ask for more

class CentralizedConversationHandler:
    """
    Central AI conversation handler that analyzes all messages and routes
    to appropriate flows with proper state management.
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI models
        self.conversation_ai = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )

        # User states and conversation history
        self.user_states = {}  # user_id -> ConversationState
        self.user_conversations = {}  # user_id -> [messages]
        self.user_context = {}  # user_id -> context info

        # NEW: Track location search context for follow-up Google Maps searches
        self.location_search_context = {}  # user_id -> {"query": str, "location_data": LocationData, "last_search_time": float}

        # Build prompts
        self._build_prompts()

        logger.info("✅ Centralized Conversation Handler initialized")

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
        """Get the main system prompt for conversation analysis"""
        return """You are Restaurant Babe (or simply Babe), a friendly AI assistant specializing in restaurant recommendations worldwide. You are enthusiastic about food and dining.

CORE TASK: Analyze user messages and classify them into query types, then provide appropriate responses.

QUERY CLASSIFICATION:

1. RESTAURANT REQUEST - User wants restaurant recommendations
   Sub-types:
   a) CITY_WIDE: Need city + cuisine type → use LangChain orchestrator
      Examples: "sushi in Tokyo", "best pizza in Rome"

   b) LOCATION_BASED_NEARBY: "Near me", "nearby", "within walking distance"
      → Request user's physical location

   c) LOCATION_BASED_GEOGRAPHIC: Specific neighborhood/street/landmark within city
      Examples: "restaurants in SoHo", "good food on Broadway", "near Times Square"
      → Use location-based search

   d) FOLLOW_UP: After previous results, asking for more/different options
      Examples: "show me more", "something cheaper", "any vegetarian options?"

2. GENERAL_QUESTION - Questions about food/restaurants/chefs but not requesting recommendations
   Examples: "How many Michelin restaurants are in Rome?", "Who is Gordon Ramsay?", "What is neo-bistro?"
   → Trigger web search (not implemented yet)

3. UNRELATED - Nothing to do with restaurants/food/cuisine
   → Politely redirect to restaurant focus

RESPONSE REQUIREMENTS:
- Be conversational and friendly, not robotic
- Handle state transitions smoothly 
- Maintain context from conversation history
- Ask clarifying questions naturally when needed
- For cuisine-only queries, always include "restaurants" or "places" in search_query

RESPONSE FORMAT (JSON only):
{{
    "query_type": "restaurant_request" | "general_question" | "unrelated",
    "request_type": "city_wide" | "location_based_nearby" | "location_based_geographic" | "follow_up" | null,
    "action": "SEARCH_CITY" | "REQUEST_LOCATION" | "SEARCH_LOCATION" | "WEB_SEARCH" | "CLARIFY" | "REDIRECT",
    "bot_response": "what to say to the user (conversational, friendly)",
    "search_query": "search query if action requires search",
    "needs_clarification": true|false,
    "missing_info": ["city", "cuisine", "location"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of decision"
}}

EXAMPLES:

User: "best sushi in Tokyo"
→ {{"query_type": "restaurant_request", "request_type": "city_wide", "action": "SEARCH_CITY", "search_query": "best sushi restaurants in Tokyo", "bot_response": "Perfect! Let me find the best sushi places in Tokyo for you.", "needs_clarification": false, "missing_info": [], "confidence": 0.95}}

User: "pizza"
→ {{"query_type": "restaurant_request", "request_type": "location_based_nearby", "action": "REQUEST_LOCATION", "search_query": "pizza restaurants", "bot_response": "I'd love to help you find great pizza places nearby! Could you share your location or tell me what neighborhood you're in?", "needs_clarification": false, "missing_info": ["location"], "confidence": 0.9}}

User: "restaurants near me"  
→ {{"query_type": "restaurant_request", "request_type": "location_based_nearby", "action": "REQUEST_LOCATION", "search_query": "restaurants", "bot_response": "I'd love to help you find great restaurants nearby! Could you share your location or tell me what neighborhood you're in?", "needs_clarification": false, "missing_info": ["location"], "confidence": 0.9}}

User: "sushi"
→ {{"query_type": "restaurant_request", "request_type": "location_based_nearby", "action": "REQUEST_LOCATION", "search_query": "sushi restaurants", "bot_response": "I'd love to help you find amazing sushi places nearby! Could you share your location or tell me what neighborhood you're in?", "needs_clarification": false, "missing_info": ["location"], "confidence": 0.9}}

User: "pizza close to Times Square"
→ {{"query_type": "restaurant_request", "request_type": "location_based_geographic", "action": "SEARCH_LOCATION", "search_query": "pizza restaurants near Times Square", "bot_response": "Great choice! Let me find the best pizza places near Times Square for you.", "needs_clarification": false, "missing_info": [], "confidence": 0.9}}

User: "best pizza in NYC"  
→ {{"query_type": "restaurant_request", "request_type": "city_wide", "action": "SEARCH_CITY", "search_query": "best pizza restaurants in NYC", "bot_response": "Perfect! Let me find the best pizza places in NYC for you.", "needs_clarification": false, "missing_info": [], "confidence": 0.95}}

User: "good Italian food in SoHo"
→ {{"query_type": "restaurant_request", "request_type": "location_based_geographic", "action": "SEARCH_LOCATION", "search_query": "Italian restaurants in SoHo", "bot_response": "Great choice! Let me find the best Italian restaurants in SoHo for you.", "needs_clarification": false, "missing_info": [], "confidence": 0.9}}

User: "How many Michelin stars does Gordon Ramsay have?"
→ {{"query_type": "general_question", "request_type": null, "action": "WEB_SEARCH", "search_query": "Gordon Ramsay Michelin stars", "bot_response": "Let me look that up for you!", "needs_clarification": false, "missing_info": [], "confidence": 0.85}}

User: "What's the weather like?"
→ {{"query_type": "unrelated", "request_type": null, "action": "REDIRECT", "bot_response": "I specialize in restaurant recommendations! What kind of dining experience are you looking for?", "needs_clarification": false, "missing_info": [], "confidence": 0.9}}

CONVERSATION FLOW:
- Maintain natural conversation flow
- Remember context from previous messages  
- Handle incomplete requests gracefully
- Guide users toward successful searches
- Be enthusiastic about food and dining"""

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

    def store_location_search_context(self, user_id: int, query: str, location_data, location_description: str):
        """Store context from a location search for follow-up Google Maps searches"""
        self.location_search_context[user_id] = {
            "query": query,
            "location_data": location_data,
            "location_description": location_description,
            "last_search_time": time.time()
        }
        logger.debug(f"Stored location context for user {user_id}: {location_description}")

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
        if user_id in self.user_context:
            del self.user_context[user_id]
        if user_id in self.location_search_context:
            del self.location_search_context[user_id]
        logger.debug(f"Cleared all data for user {user_id}")