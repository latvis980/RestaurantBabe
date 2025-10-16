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
    create_search_handoff, create_conversation_handoff
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

    DIALOG STATE TRACKING:
    - You maintain an accumulated_state that builds across ALL turns
    - Each turn, you CRUD (Create/Read/Update/Delete) the state
    - You decide when state is "complete" enough to trigger search

    STATE STRUCTURE:
    {{
        "location_parts": ["Lapa", "Lisbon"],  // Accumulates location fragments
        "destination": "Lapa, Lisbon",         // Combined when complete
        "cuisine": "specialty coffee",
        "requirements": ["specialty"],
        "preferences": {{}},
        "is_complete": true                    // You decide this!
    }}

    EXAMPLES OF STATE ACCUMULATION:

    Turn 1: "specialty coffee in Lapa"
    → {{
        "state_update": {{
            "location_parts": ["Lapa"],
            "cuisine": "specialty coffee",
            "is_complete": false
        }},
        "action": "collect_info",
        "response_text": "I can help with that! Could you tell me which city Lapa is in?",
        "reasoning": "Need city to complete location. Added 'Lapa' to location_parts."
    }}

    Turn 2: "Lisbon"
    → {{
        "state_update": {{
            "location_parts": ["Lapa", "Lisbon"],  // ACCUMULATED!
            "destination": "Lapa, Lisbon",         // COMBINED!
            "cuisine": "specialty coffee",         // PRESERVED!
            "is_complete": true                    // READY!
        }},
        "action": "trigger_search",
        "response_text": "Perfect! I'll find specialty coffee in Lapa, Lisbon.",
        "reasoning": "User provided city. Combined with neighborhood. State complete."
    }}

    ALTERNATIVE: One-shot query
    Turn 1: "best ramen in Tokyo"
    → {{
        "state_update": {{
            "location_parts": ["Tokyo"],
            "destination": "Tokyo",
            "cuisine": "ramen",
            "requirements": ["best"],
            "is_complete": true
        }},
        "action": "trigger_search",
        "response_text": "Great! Searching for the best ramen in Tokyo.",
        "reasoning": "Complete info in first message. State immediately complete."
    }}

    RESPONSE FORMAT:
    {{
        "action": "chat_response" | "collect_info" | "trigger_search",
        "response_text": "what to say",
        "state_update": {{
            "location_parts": [...],  // ADD new location fragments
            "destination": "combined location" | null,
            "cuisine": "cuisine" | null,
            "requirements": [...],
            "preferences": {{}},
            "is_complete": true | false
        }},
        "reasoning": "explain state changes"
    }}

    RULES:
    1. ALWAYS include state_update with ALL fields (even if unchanged)
    2. ADD to location_parts, don't replace (accumulation!)
    3. Set is_complete=true ONLY when you have enough to search
    4. Can be ready in 1 turn OR 10 turns - YOU decide based on state
    5. Don't ask for info already in accumulated_state
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

    USER MESSAGE: {user_message}

    Analyze the message and determine action.""")
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
        Process user message with context enrichment

        Returns structured HandoffMessage with enriched location
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

            # Prepare prompt variables
            prompt_vars = {
                'conversation_history': self._format_conversation_context(session),
                'accumulated_state': json.dumps(accumulated_state, indent=2),
                'current_destination': session.get('current_destination') or 'None',
                'current_cuisine': session.get('current_cuisine') or 'None',
                'last_searched_city': session.get('last_searched_city') or 'None',
                'conversation_state': current_state_value,
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

            logger.info(f"🤖 AI Decision: {decision.get('action')} - {decision.get('reasoning')}")

            # UPDATE ACCUMULATED STATE
            state_update = decision.get('state_update', {})
            if state_update:
                accumulated_state.update(state_update)
                session['accumulated_state'] = accumulated_state
                logger.info(f"📊 Updated state: {accumulated_state}")

            # Extract info from accumulated state (not decision)
            destination = accumulated_state.get('destination') or session.get('current_destination')
            cuisine = accumulated_state.get('cuisine') or session.get('current_cuisine')
            is_complete = accumulated_state.get('is_complete', False)
            requirements = accumulated_state.get('requirements', [])
            preferences = accumulated_state.get('preferences', {})
            is_new_destination = decision.get('is_new_destination', False)

            # Update session
            session['current_destination'] = destination
            session['current_cuisine'] = cuisine

            # ENHANCED: Extract and store city for context enrichment
            if destination and is_new_destination:
                # Extract city from destination for future enrichment
                city = self._extract_city_from_destination(destination)
                if city:
                    session['last_searched_city'] = city
                    logger.info(f"💾 Stored city context for user {user_id}: {city}")

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

            # Update collected info
            self._update_collected_info(session, destination, cuisine, requirements, preferences)

            # Route by action
            action = decision.get('action')

            if action in ['chat_response', 'collect_info']:
                # Continue conversation
                return create_conversation_handoff(
                    response=decision.get('response_text', "How can I help?"),
                    reasoning=decision.get('reasoning', '')
                )

            elif action == 'trigger_search' and is_complete:
                # Track search time
                session['last_search_time'] = time.time()

                # Determine search type hint
                search_type_hint = (
                    SearchType.LOCATION_SEARCH if gps_coordinates
                    else SearchType.CITY_SEARCH
                )

                # Create search handoff with enriched location
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
                    clear_previous=is_new_destination,
                    is_new_destination=is_new_destination,
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
                'last_search_time': None
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