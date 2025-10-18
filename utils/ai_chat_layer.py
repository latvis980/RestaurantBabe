# utils/ai_chat_layer.py
"""
ENHANCED CHAT LAYER V2: AI Chat Layer with Location Context Enrichment

NEW FEATURE:
- Enriches partial locations with stored city context
- Example: User searched "restaurants in Lisbon" yesterday
          Today: "coffee in Lapa" → enriched to "Lapa, Lisbon"

CRITICAL IMPORT NOTE:
This file must be copied to: utils/ai_chat_layer.py
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.handoff_protocol import (
    HandoffMessage, SearchContext, SearchType, HandoffCommand,
    create_search_handoff, create_conversation_handoff, create_resume_handoff
)

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Internal conversation states"""
    GREETING = "greeting"
    COLLECTING_CUISINE = "collecting_cuisine"
    COLLECTING_LOCATION = "collecting_location"
    COLLECTING_PREFERENCES = "collecting_preferences"
    READY_TO_SEARCH = "ready_to_search"
    SHOWING_RESULTS = "showing_results"
    FOLLOW_UP = "follow_up"


class AIChatLayer:
    """
    ENHANCED V2: Supervisor agent with location context enrichment

    NEW FEATURE:
    - Tracks last_searched_city for each user
    - Enriches partial locations (neighborhoods) with stored city context
    - Example: "Lapa" becomes "Lapa, Lisbon" if user recently searched Lisbon

    Responsibilities:
    - Manage conversation flow
    - Collect destination + cuisine
    - Enrich locations with context
    - Decide WHEN to search
    - Detect destination changes
    - Return structured HandoffMessage
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )

        # User sessions
        self.user_sessions: Dict[int, Dict[str, Any]] = {}

        # Build prompts
        self._build_prompts()

        logger.info("✅ AI Chat Layer V2 with Context Enrichment initialized")

    def _build_prompts(self):
        """Build AI prompts with state tracking"""

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a conversation manager for a restaurant bot with Dialog State Tracking.

    Your job is to ACCUMULATE information across multiple turns and decide when you have enough to search.

    CRITICAL: You must also determine the SEARCH MODE based on user intent.

    SEARCH MODES:
    1. GPS_REQUIRED - User wants restaurants near their physical location
       - Examples: "near me", "nearby", "around me", "close to me", "in my area", "where I am"
       - Requires: GPS coordinates
       - Action: If no GPS provided, respond with needs_gps=true

    2. FOLLOW_UP_MORE_RESULTS - User asks for more after seeing initial results
       - Examples: "show more", "let's find more", "more options", "other places", "find more"
       - Context: User recently got database results, wants Google Maps enhancement
       - Requires: Nothing - graph state has coordinates
       - Action: Set search_mode="follow_up_more_results", is_complete=true, trigger resume

    3. CITY_SEARCH - User specifies a city/destination
       - Examples: "in Tokyo", "Paris restaurants", "best sushi in NYC"
       - Requires: City name
       - Action: Normal destination collection

    4. NEIGHBORHOOD_SEARCH - User specifies neighborhood without city
       - Examples: "in SoHo", "Chinatown restaurants", "bars in Lapa"
       - Requires: Neighborhood + city (may need clarification)
       - Action: Collect or enrich with city context

    DIALOG STATE TRACKING:
    - You maintain an accumulated_state that builds across ALL turns
    - Each turn, you CRUD (Create/Read/Update/Delete) the state
    - You decide when state is "complete" enough to trigger search

    STATE STRUCTURE:
    {{
        "location_parts": ["Lapa", "Lisbon"],
        "destination": "Lapa, Lisbon",
        "cuisine": "specialty coffee",
        "requirements": ["specialty"],
        "preferences": {{}},
        "search_mode": "gps_required" | "city_search" | "neighborhood_search",
        "needs_gps": true | false,
        "is_complete": true | false
    }}

    CRITICAL RULES FOR GPS_REQUIRED MODE:
    1. ALWAYS set search_mode="gps_required" when user indicates "near me" / "nearby" intent
    2. ALWAYS set needs_gps=true if search_mode="gps_required" and no GPS coordinates provided
    3. NEVER ask about stored cities when search_mode="gps_required"
    4. IGNORE stored city context completely for GPS_REQUIRED mode
    5. Even if you know user searched "Valencia" yesterday, if they say "near me" today, set needs_gps=true

    EXAMPLES:

    Turn 1: "specialty coffee near me"
    → {{
        "state_update": {{
            "cuisine": "specialty coffee",
            "search_mode": "gps_required",
            "needs_gps": true,
            "is_complete": false
        }},
        "action": "request_gps",
        "response_text": "I'd love to help you find great specialty coffee near you!",
        "reasoning": "User wants nearby results. Need GPS coordinates, not city name."
    }}

    Turn 1: "restaurants around me"
    → {{
        "state_update": {{
            "cuisine": "restaurants",
            "search_mode": "gps_required",
            "needs_gps": true,
            "is_complete": false
        }},
        "action": "request_gps",
        "response_text": "I'd love to help you find great restaurants near you!",
        "reasoning": "User wants nearby results. Need GPS coordinates."
    }}

    After showing database results, user says: "Let's find more"
    → {{
        "state_update": {{
            "search_mode": "follow_up_more_results",
            "is_complete": true
        }},
        "action": "trigger_search",
        "response_text": "Perfect! I'll find more restaurants for you...",
        "reasoning": "User wants more results. Resume graph execution with accept decision."
    }}

    After showing database results, user says: "show me other options"
    → {{
        "state_update": {{
            "search_mode": "follow_up_more_results",
            "is_complete": true
        }},
        "action": "trigger_search",
        "response_text": "Great! Searching for more options...",
        "reasoning": "Follow-up request for more results. Resume with Google Maps search."
    }}

    Turn 1: "best ramen in Tokyo"
    → {{
        "state_update": {{
            "location_parts": ["Tokyo"],
            "destination": "Tokyo",
            "cuisine": "ramen",
            "requirements": ["best"],
            "search_mode": "city_search",
            "needs_gps": false,
            "is_complete": true
        }},
        "action": "trigger_search",
        "response_text": "Great! Searching for the best ramen in Tokyo.",
        "reasoning": "Complete city search info. No GPS needed."
    }}

    Turn 1: "coffee in Lapa"
    → {{
        "state_update": {{
            "location_parts": ["Lapa"],
            "cuisine": "coffee",
            "search_mode": "neighborhood_search",
            "needs_gps": false,
            "is_complete": false
        }},
        "action": "collect_info",
        "response_text": "I can help with that! Which city is Lapa in?",
        "reasoning": "Neighborhood specified but need city. Can enrich with stored city context if available."
    }}

    STORED CONTEXT USAGE:
    - ONLY use stored city context for "neighborhood_search" mode
    - NEVER use stored city context for "gps_required" mode
    - For "city_search" mode, city is already specified

    USER HAS STORED CITY "Valencia":
    Turn 1: "coffee near me"
    → {{
        "search_mode": "gps_required",
        "needs_gps": true,
        "action": "request_gps"
    }}
    WRONG: "Are you in Valencia?"
    CORRECT: Request GPS immediately

    Turn 1: "coffee in Lapa" 
    → {{
        "search_mode": "neighborhood_search",
        "location_parts": ["Lapa"],
        "action": "collect_info",
        "response_text": "I can help! Is Lapa in Lisbon?"
    }}
    CORRECT: Can use stored context to enrich

    RESPONSE FORMAT (JSON only):
    {{
        "action": "chat_response" | "collect_info" | "request_gps" | "trigger_search",
        "response_text": "what to say",
        "state_update": {{
            "location_parts": [...],
            "destination": "combined location" | null,
            "cuisine": "cuisine" | null,
            "requirements": [...],
            "preferences": {{}},
            "search_mode": "gps_required" | "city_search" | "neighborhood_search" | null,
            "needs_gps": true | false,
            "is_complete": true | false
        }},
        "reasoning": "explain decision and search mode detection"
    }}

    RULES:
    1. ALWAYS detect search_mode first based on user intent
    2. ALWAYS include state_update with ALL fields
    3. For GPS_REQUIRED: NEVER use stored city, ALWAYS request GPS if not provided
    4. For NEIGHBORHOOD_SEARCH: CAN use stored city to enrich
    5. For CITY_SEARCH: City already specified in query
    6. Set is_complete=true ONLY when you have enough info for the detected search mode
    """),
        ("human", """CONVERSATION HISTORY:
    {conversation_history}

    CURRENT ACCUMULATED STATE:
    {accumulated_state}

    STORED CONTEXT:
    - Current destination: {current_destination}
    - Current cuisine: {current_cuisine}
    - Last searched city: {last_searched_city}
    - Conversation state: {conversation_state}

    GPS COORDINATES PROVIDED: {has_gps}

    USER MESSAGE: {user_message}

    Analyze the message, detect search mode, and determine action.""")
        ])

        self.conversation_chain = self.conversation_prompt | self.llm


    async def process_message(
        self,
        user_id: int,
        user_message: str,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None
    ) -> HandoffMessage:
        """
        Process user message with AI-detected search mode

        Returns structured HandoffMessage based on AI decision
        """
        try:
            # Get or create session
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            session = self._get_or_create_session(user_id, thread_id)

            # Store GPS if provided
            if gps_coordinates:
                session['gps_coordinates'] = gps_coordinates

            # Add user message to history
            session['conversation_history'].append({
                'role': 'user',
                'message': user_message,
                'timestamp': time.time()
            })

            # Get current state
            current_state = session.get('state', ConversationState.GREETING)
            current_state_value = current_state.value if isinstance(current_state, ConversationState) else str(current_state)

            accumulated_state = session.get('accumulated_state', {})

            # Prepare prompt variables - INCLUDE GPS STATUS
            prompt_vars = {
                'conversation_history': self._format_conversation_context(session),
                'accumulated_state': json.dumps(accumulated_state, indent=2),
                'current_destination': session.get('current_destination') or 'None',
                'current_cuisine': session.get('current_cuisine') or 'None',
                'last_searched_city': session.get('last_searched_city') or 'None',
                'conversation_state': current_state_value,
                'has_gps': 'Yes' if gps_coordinates else 'No',  # NEW: Tell AI if we have GPS
                'user_message': user_message
            }

            # Get AI decision
            response = await self.llm.ainvoke(
                self.conversation_prompt.format_messages(**prompt_vars)
            )

            # Parse response
            if hasattr(response, 'content'):
                response_content = str(response.content)
            else:
                response_content = str(response)

            try:
                decision = self._parse_ai_response(response_content)
            except json.JSONDecodeError:
                logger.error(f"JSON parse error: {response_content}")
                return create_conversation_handoff(
                    response="How can I help you find restaurants?",
                    reasoning="JSON parse error - using fallback"
                )

            logger.info(f"🤖 AI Decision: {decision.get('action')} - Search mode: {decision.get('state_update', {}).get('search_mode')} - {decision.get('reasoning')}")

            # UPDATE ACCUMULATED STATE
            state_update = decision.get('state_update', {})
            if state_update:
                accumulated_state.update(state_update)
                session['accumulated_state'] = accumulated_state
                logger.info(f"📊 Updated state: {accumulated_state}")

            # Extract info from accumulated state
            destination = accumulated_state.get('destination') or session.get('current_destination')
            cuisine = accumulated_state.get('cuisine') or session.get('current_cuisine')
            is_complete = accumulated_state.get('is_complete', False)
            requirements = accumulated_state.get('requirements', [])
            preferences = accumulated_state.get('preferences', {})
            search_mode = accumulated_state.get('search_mode')
            needs_gps = accumulated_state.get('needs_gps', False)

            # Update session
            session['current_destination'] = destination
            session['current_cuisine'] = cuisine

            # Update state
            new_state_str = decision.get('new_state', 'greeting')
            try:
                session['state'] = ConversationState(new_state_str)
            except ValueError:
                session['state'] = ConversationState.GREETING

            # Add assistant response to history
            if decision.get('response_text'):
                session['conversation_history'].append({
                    'role': 'assistant',
                    'message': decision['response_text'],
                    'timestamp': time.time()
                })

            # Route by action
            action = decision.get('action')

            # Handle GPS request
            if action == 'request_gps' or needs_gps:
                logger.info("🎯 AI detected GPS-required search mode - requesting location button")
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    reasoning=decision.get('reasoning', 'GPS coordinates required'),
                    conversation_response=decision.get('response_text', 'I\'d love to help you find restaurants near you!')
                )

            if action in ['chat_response', 'collect_info']:
                # Continue conversation
                return create_conversation_handoff(
                    response=decision.get('response_text', "How can I help?"),
                    reasoning=decision.get('reasoning', '')
                )

            elif action == 'trigger_search' and is_complete:
                # ====================================================================
                # NEW: Check if this is a "more results" request - RESUME graph
                # ====================================================================
                # When resuming
                if search_mode == 'follow_up_more_results':
                    # Get the ORIGINAL search thread_id
                    original_thread_id = session.get('last_search_thread_id')

                    if not original_thread_id:
                        return create_conversation_handoff(
                            response="I couldn't find your previous search...",
                            reasoning="Missing original thread_id"
                        )

                    return create_resume_handoff(
                        thread_id=original_thread_id,  # Use ORIGINAL, not current!
                        decision="accept"
                    )

                # ====================================================================
                # Regular NEW SEARCH - track search time
                # ====================================================================
                session['last_search_time'] = time.time()
                session['last_search_thread_id'] = thread_id

                # Extract city from destination if this is a new city search
                if destination and (search_mode == 'city_search' or search_mode == 'neighborhood_search'):
                    city = self._extract_city_from_destination(destination)
                    if city:
                        session['last_searched_city'] = city
                        logger.info(f"💾 Stored city context for user {user_id}: {city}")

                # Determine search type hint based on AI-detected mode
                if search_mode == 'gps_required':
                    search_type_hint = SearchType.LOCATION_SEARCH
                else:
                    search_type_hint = SearchType.CITY_SEARCH

                # Create search handoff
                return create_search_handoff(
                    destination=destination or "unknown",
                    cuisine=cuisine,
                    search_type=search_type_hint,
                    user_query=user_message,
                    user_id=user_id,
                    thread_id=thread_id,
                    gps_coordinates=gps_coordinates,
                    requirements=requirements,
                    preferences=preferences,
                    clear_previous=False,
                    is_new_destination=False,
                    reasoning=decision.get('reasoning', '')
                )

            else:
                # Unknown action - fallback
                return create_conversation_handoff(
                    response=decision.get('response_text', "Let me help you find restaurants."),
                    reasoning=f"Unknown action: {action}"
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return create_conversation_handoff(
                response="I'd be happy to help you find restaurants. What are you looking for?",
                reasoning=f"Error: {str(e)}"
            )


    def _extract_city_from_destination(self, destination: str) -> Optional[str]:
        """
        Extract city name from destination string for context storage

        Examples:
        "Lapa, Lisbon" → "Lisbon"
        "SoHo, New York" → "New York"
        "Paris" → "Paris"
        "Rua Augusta, Lisbon" → "Lisbon"
        """
        if not destination:
            return None

        # Split by comma and take the last part (usually the city)
        parts = [p.strip() for p in destination.split(',')]

        # If multiple parts, last one is usually city
        if len(parts) > 1:
            return parts[-1]

        # If single part, it might be a city itself
        return parts[0]

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for AI"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

        # Format last 10 messages
        recent = history[-10:]
        formatted = []
        for msg in recent:
            role = msg.get('role', 'user')
            message = msg.get('message', '')
            formatted.append(f"{role.capitalize()}: {message}")

        return "\n".join(formatted)

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response (handles markdown code blocks)"""
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)

    def _get_or_create_session(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get or create user session with proper state tracking"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'thread_id': thread_id,
                'created_at': time.time(),
                'state': ConversationState.GREETING,
                'conversation_history': [],

                # ACCUMULATED STATE - builds across turns
                'accumulated_state': {
                    'location_parts': [],  # ["Lapa"] → ["Lapa", "Lisbon"] → "Lapa, Lisbon"
                    'destination': None,   # Final combined destination
                    'cuisine': None,
                    'requirements': [],
                    'preferences': {},
                    'is_complete': False   # AI decides this
                },

                'current_destination': None,
                'current_cuisine': None,
                'last_searched_city': None,
                'gps_coordinates': None,
                'last_search_time': None,
                'last_search_thread_id': None
            }
        return self.user_sessions[user_id]

    def _update_collected_info(self, session: Dict[str, Any], destination: Optional[str], 
                               cuisine: Optional[str], requirements: List[str], 
                               preferences: Dict[str, Any]):
        """Update collected information in session"""
        collected = session.get('collected_info', {})

        if destination:
            collected['location'] = destination
        if cuisine:
            collected['cuisine'] = cuisine
        if requirements:
            collected['requirements'] = requirements
        if preferences:
            collected['preferences'] = preferences

        session['collected_info'] = collected

    def clear_session(self, user_id: int):
        """Clear user session but keep last_searched_city for enrichment"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            last_city = session.get('last_searched_city')  # Preserve this!

            # Clear everything else
            session['current_destination'] = None
            session['current_cuisine'] = None
            session['state'] = ConversationState.GREETING
            session['collected_info'] = {
                'location': None,
                'cuisine': None,
                'requirements': [],
                'preferences': {}
            }
            session['last_search_time'] = None

            # Restore city context
            session['last_searched_city'] = last_city

            logger.info(f"🧹 Cleared session for user {user_id} (kept city context: {last_city})")

    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current session info"""
        return self.user_sessions.get(user_id)