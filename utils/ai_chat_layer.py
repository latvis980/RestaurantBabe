# utils/ai_chat_layer.py
"""
AI Chat Layer with Context-Aware Parameter Management

ARCHITECTURE:
- Conversation history provides implicit context (last 10 messages)
- Active context tracks current search parameters (destination, cuisine, radius, etc.)
- AI explicitly decides: CONTINUE, MODIFY, or NEW context
- AI provides ALL parameters on every turn (no blind accumulation)
- Balance: Remember what we're discussing BUT allow easy parameter changes

Key Features:
- Memory context integration (preferences, past restaurants, patterns)
- FIXED: Search mode detection (GPS vs city vs neighborhood) - comma = neighborhood search
- Context-aware parameter tracking with explicit change detection
- AI-driven parameter modification (no hardcoded keyword matching)
- Pending GPS state management
- Personalized responses based on user history
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.handoff_protocol import (
    HandoffMessage, SearchContext, SearchType, HandoffCommand
)
from location.geocoding import geocode_location

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Internal conversation states"""
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    READY_TO_SEARCH = "ready_to_search"
    SHOWING_RESULTS = "showing_results"
    FOLLOW_UP = "follow_up"


class ContextDecisionType(Enum):
    """AI's decision about context handling"""
    CONTINUE = "CONTINUE"  # Same search context
    MODIFY = "MODIFY"      # Update specific parameters
    NEW = "NEW"            # Fresh context


class AIChatLayer:
    """
    AI Chat Layer with Context-Aware Parameter Management

    Core Responsibilities:
    - Track active search context (what we're currently discussing)
    - Let AI decide to CONTINUE/MODIFY/NEW context
    - Provide conversation history for implicit context
    - Detect search modes (GPS vs city vs neighborhood)
    - Handle ambiguous locations
    - Manage pending GPS state
    - Use memory for personalized responses
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI
        self.llm = ChatOpenAI(
            model=getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )

        # User sessions (in-memory for current conversation)
        self.user_sessions: Dict[int, Dict[str, Any]] = {}

        # Location contexts (for 30-minute expiry tracking)
        self.location_contexts: Dict[int, Dict[str, Any]] = {}

        # Build prompts
        self._build_conversation_prompt()

        logger.info("✅ AI Chat Layer initialized with context-aware parameter management")

    def _build_conversation_prompt(self):
        """Build the main conversation prompt with context-aware parameter tracking"""

        system_prompt = """You are an AI conversation manager for a restaurant recommendation bot.

Your job is to analyze user messages and decide what to do based on the current conversation context.

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION CONTEXT (for reference - provides implicit understanding)
═══════════════════════════════════════════════════════════════════════════════

RECENT CONVERSATION (last 10 messages):
{conversation_history}

USER PREFERENCES FROM MEMORY:
{memory_context}

ACTIVE SEARCH CONTEXT (what we're currently discussing):
- Destination: {active_destination}
- Cuisine: {active_cuisine}
- Search radius: {active_radius}km
- Requirements: {active_requirements}
- Established: {context_age}
- Searches performed: {search_count}

LAST SEARCH RESULTS (shown {time_ago}):
- Restaurants shown: {shown_restaurants}

STORED LOCATION (if user shared location previously):
{stored_location}

PENDING GPS STATE:
- Waiting for GPS: {pending_gps}
- For cuisine: {pending_gps_cuisine}

═══════════════════════════════════════════════════════════════════════════════
CURRENT USER MESSAGE
═══════════════════════════════════════════════════════════════════════════════

"{user_message}"

GPS Coordinates provided now: {has_gps}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

1. **DETERMINE CONTEXT DECISION** (how to handle parameters):

   **CONTINUE**: User is continuing with the same search context
   - Keep ALL active context parameters unchanged
   - User says: "show more", "any other options", "what else"

   **MODIFY**: User changed ONE OR MORE specific parameters
   - Update changed parameters, keep others from active context
   - User says: "actually I want pizza", "closer to me", "I meant lunch not brunch"

   **NEW**: Completely new search or topic
   - Fresh parameter extraction, ignore old active context
   - User says: "best sushi in Tokyo", "now show me bars in Paris"

2. **EXTRACT ALL PARAMETERS** (you MUST provide all parameters):

   Even if CONTINUING, you must provide all parameters (copy from active context).

   - **destination**: Extract from message OR infer from active context OR infer from conversation
     * If user mentions BOTH neighborhood AND city, format as "Neighborhood, City"
     * If user mentions ONLY city, format as "City"
     * The COMMA is critical for distinguishing city-wide vs neighborhood searches

   - **cuisine**: Extract from message OR infer from active context OR infer from conversation

   - **search_radius_km**: Extract if mentioned OR use active context OR default 1.5
     * "within 5 min walk" → ~0.4km
     * "within 10 min walk" → ~0.8km
     * "nearby" / "walking distance" → 1.5km (default)
     * "closer" (when modifying) → reduce by 50%

   - **requirements**: Extract new ones OR keep from active context

   - **preferences**: Any additional filters (price, atmosphere, etc.)

3. **DETERMINE ACTION**:

   - **execute_search**: User wants restaurant recommendations
   - **chat_response**: Need more info, casual chat, or clarification
   - **request_gps**: Need user's physical location (for "near me" queries)

4. **SEARCH MODE DETECTION** (for execute_search):

   **CITY_SEARCH**: Searching an entire city with NO specific neighborhood/area
   - Destination is ONLY a city name (no neighborhood, no landmark)
   - User wants city-wide recommendations
   - Uses web scraping

   **LOCATION_SEARCH**: Searching a specific area within a city
   - Destination includes neighborhood + city (comma-separated)
   - Destination is a landmark or specific area
   - GPS coordinates are provided
   - Searches database first, then Google Maps if needed

   **CRITICAL LOGIC**:
   - If destination format is "Neighborhood, City" → LOCATION_SEARCH
   - If destination format is "City" alone → CITY_SEARCH
   - If destination has a comma → usually LOCATION_SEARCH
   - If GPS coordinates provided → LOCATION_SEARCH
   - "near [landmark]" → LOCATION_SEARCH

5. **GPS HANDLING**:

   If user says "near me", "around me", "close to me" WITHOUT providing GPS:
   - Action: request_gps
   - Extract cuisine from message
   - Set needs_gps: true

   If GPS coordinates are provided ({has_gps} = Yes):
   - Action: execute_search
   - Search mode: LOCATION_SEARCH
   - Use provided coordinates

6. **AMBIGUITY HANDLING**:

   If location is ambiguous (Springfield, Cambridge, etc.) and you can't determine from context:
   - Ask for clarification
   - Suggest known options if available

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only)
═══════════════════════════════════════════════════════════════════════════════

{{
  "action": "execute_search" | "chat_response" | "request_gps",

  "context_decision": {{
    "type": "CONTINUE" | "MODIFY" | "NEW",
    "reasoning": "brief explanation of decision"
  }},

  "parameters": {{
    "destination": "city or neighborhood name (REQUIRED)",
    "cuisine": "cuisine type or null (REQUIRED)",
    "search_mode": "CITY_SEARCH" | "LOCATION_SEARCH",
    "search_radius_km": 1.5,
    "requirements": ["requirement1", "requirement2"],
    "preferences": {{}},
    "modifications": {{
      // Only if context_decision.type = MODIFY
      "cuisine": {{"from": "ramen", "to": "sushi", "reason": "user explicitly changed"}},
      "radius": {{"from": 1.5, "to": 0.7, "reason": "user wants closer"}}
    }}
  }},

  "response_text": "your response to the user",
  "reasoning": "internal reasoning for your decision",
  "needs_gps": false,

  "state_update": {{
    // Only include if you want to clear pending GPS
    "clear_pending_gps": false
  }}
}}

IMPORTANT RULES:
- ALWAYS provide ALL parameters (destination, cuisine, search_mode, search_radius_km)
- Use active context to fill in parameters user didn't explicitly mention
- Be smart about implicit continuation ("show more" means same parameters)
- Be smart about modifications ("closer" means reduce radius, keep other params)
- Don't ask unnecessary questions if context is clear
- Use memory context to personalize responses
- PAY ATTENTION to destination format: comma means neighborhood search, no comma means city search
"""

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Analyze the message above and provide your decision in JSON format.")
        ])

    # ============================================================================
    # CORE MESSAGE PROCESSING
    # ============================================================================

    async def process_message(
        self,
        user_id: int,
        user_message: str,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> HandoffMessage:
        """
        Process user message with context-aware parameter management.

        This is the SINGLE DECISION POINT for all user messages.
        """
        try:
            # Get or create session
            session = self._get_or_create_session(user_id, thread_id or f"chat_{user_id}")

            # Get active context
            active_context = session.get('active_context', {})

            # Get pending GPS state
            pending_gps_state = session.get('pending_gps', {})
            pending_gps = pending_gps_state.get('active', False)
            pending_gps_cuisine = pending_gps_state.get('cuisine')

            # Get stored location context
            stored_location = session.get('stored_location', {})
            stored_location_text = "None"
            if stored_location:
                loc = stored_location.get('name', '')
                coords = stored_location.get('coordinates')
                age_min = (time.time() - stored_location.get('stored_at', 0)) / 60
                if age_min < 30:  # Only show if less than 30 min old
                    if coords:
                        stored_location_text = f"{loc} ({coords[0]:.4f}, {coords[1]:.4f}) - {int(age_min)} min ago"
                    else:
                        stored_location_text = f"{loc} - {int(age_min)} min ago"

            # Add user message to history
            self.add_message(user_id, 'user', user_message)

            # Get last search info
            last_search = session.get('last_search', {})
            last_search_timestamp = last_search.get('timestamp', 0)
            time_since_search = time.time() - last_search_timestamp
            time_ago_text = self._format_time_ago(time_since_search) if last_search_timestamp else "None"
            shown_restaurants = last_search.get('shown_restaurants', [])

            # Format memory context
            memory_context_text = self._format_memory_context(user_context)

            # Prepare prompt variables
            prompt_vars = {
                'conversation_history': self._format_conversation_context(session),
                'memory_context': memory_context_text,
                'active_destination': active_context.get('destination') or 'None',
                'active_cuisine': active_context.get('cuisine') or 'None',
                'active_radius': active_context.get('search_radius_km', 1.5),
                'active_requirements': ', '.join(active_context.get('requirements', [])) or 'None',
                'context_age': self._format_time_ago(time.time() - active_context.get('established_at', time.time())) if active_context else 'None',
                'search_count': active_context.get('search_count', 0),
                'time_ago': time_ago_text,
                'shown_restaurants': ', '.join(shown_restaurants[:10]) if shown_restaurants else 'None',
                'stored_location': stored_location_text,
                'pending_gps': 'Yes' if pending_gps else 'No',
                'pending_gps_cuisine': pending_gps_cuisine or 'None',
                'user_message': user_message,
                'has_gps': 'Yes' if gps_coordinates else 'No'
            }

            # Get AI decision
            response = await self.llm.ainvoke(
                self.conversation_prompt.format_messages(**prompt_vars)
            )

            # Parse response - handle AIMessage type
            response_content = response.content if hasattr(response, 'content') else str(response)
            decision = self._parse_ai_response(response_content)

            # Extract decision fields
            action = decision.get('action', 'chat_response')
            context_decision = decision.get('context_decision', {})
            context_type = context_decision.get('type', 'NEW')
            parameters = decision.get('parameters', {})
            response_text = decision.get('response_text', '')
            reasoning = decision.get('reasoning', '')
            needs_gps = decision.get('needs_gps', False)
            state_update = decision.get('state_update', {})

            # Handle clear_pending_gps signal
            if state_update.get('clear_pending_gps'):
                self.clear_pending_gps(user_id)

            logger.info(f"🤖 AI Decision: action={action}, context={context_type}, destination={parameters.get('destination')}, cuisine={parameters.get('cuisine')}, mode={parameters.get('search_mode')}")

            # ================================================================
            # UPDATE ACTIVE CONTEXT based on context decision
            # ================================================================

            if action == 'execute_search':
                if context_type == 'NEW':
                    # Fresh context - replace everything
                    session['active_context'] = {
                        'destination': parameters.get('destination'),
                        'cuisine': parameters.get('cuisine'),
                        'search_radius_km': parameters.get('search_radius_km', 1.5),
                        'requirements': parameters.get('requirements', []),
                        'preferences': parameters.get('preferences', {}),
                        'established_at': time.time(),
                        'last_modified': time.time(),
                        'search_count': 0
                    }
                    logger.info(f"🆕 NEW context: {parameters.get('destination')}, {parameters.get('cuisine')}")

                elif context_type == 'MODIFY':
                    # Update specific parameters, keep others
                    if not active_context:
                        active_context = {}

                    active_context.update({
                        'destination': parameters.get('destination'),
                        'cuisine': parameters.get('cuisine'),
                        'search_radius_km': parameters.get('search_radius_km', 1.5),
                        'requirements': parameters.get('requirements', active_context.get('requirements', [])),
                        'preferences': parameters.get('preferences', active_context.get('preferences', {})),
                        'last_modified': time.time()
                    })
                    session['active_context'] = active_context

                    modifications = parameters.get('modifications', {})
                    logger.info(f"✏️ MODIFIED context: {modifications}")

                elif context_type == 'CONTINUE':
                    # Increment search count, keep everything else
                    if active_context:
                        active_context['search_count'] = active_context.get('search_count', 0) + 1
                        session['active_context'] = active_context
                    logger.info(f"➡️ CONTINUING context: {active_context.get('destination')}, {active_context.get('cuisine')}")

            # ================================================================
            # RETURN APPROPRIATE HANDOFF
            # ================================================================

            # Handle GPS request
            if action == 'request_gps':
                # Set pending GPS state
                cuisine = parameters.get('cuisine')
                if cuisine:
                    self.set_pending_gps(user_id, cuisine)

                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=response_text or "Please share your location to find nearby restaurants.",
                    reasoning=reasoning,
                    needs_gps=True
                )

            # Handle chat response
            if action == 'chat_response':
                # Add assistant message to history
                self.add_message(user_id, 'assistant', response_text)

                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=response_text or "How can I help you find restaurants?",
                    reasoning=reasoning,
                    needs_gps=needs_gps
                )

            # Handle execute_search
            if action == 'execute_search':
                # Determine search type
                search_mode = parameters.get('search_mode', 'CITY_SEARCH')

                if search_mode == 'CITY_SEARCH':
                    search_type = SearchType.CITY_SEARCH
                else:
                    search_type = SearchType.LOCATION_SEARCH

                # Build search context
                search_context = SearchContext(
                    destination=parameters.get('destination', ''),
                    cuisine=parameters.get('cuisine'),
                    search_type=search_type,
                    gps_coordinates=gps_coordinates,
                    search_radius_km=parameters.get('search_radius_km', 1.5),
                    requirements=parameters.get('requirements', []),
                    preferences=parameters.get('preferences', {}),
                    user_query=user_message,
                    is_follow_up=(context_type == 'CONTINUE'),
                    exclude_restaurants=shown_restaurants if context_type == 'CONTINUE' else [],
                    user_id=user_id,
                    thread_id=thread_id or f"chat_{user_id}",
                    supervisor_instructions=f"Context decision: {context_type}. {context_decision.get('reasoning', '')}"
                )

                return HandoffMessage(
                    command=HandoffCommand.EXECUTE_SEARCH,
                    search_context=search_context,
                    reasoning=reasoning
                )

            # Fallback
            return HandoffMessage(
                command=HandoffCommand.CONTINUE_CONVERSATION,
                conversation_response="How can I help you find restaurants?",
                reasoning="Unknown action",
                needs_gps=False
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return HandoffMessage(
                command=HandoffCommand.CONTINUE_CONVERSATION,
                conversation_response="I'd be happy to help you find restaurants. What are you looking for?",
                reasoning=f"Error: {str(e)}",
                needs_gps=False
            )

    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================

    def _get_or_create_session(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get or create user session with active context tracking"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'thread_id': thread_id,
                'created_at': time.time(),
                'conversation_history': [],

                # Active context (what we're currently discussing)
                'active_context': {},

                # Last search (for "more results" and exclusions)
                'last_search': {},

                # Location storage
                'stored_location': {},

                # GPS state
                'pending_gps': {
                    'active': False,
                    'cuisine': None,
                    'timestamp': None
                }
            }

            logger.info(f"🆕 Created new session for user {user_id}")

        return self.user_sessions[user_id]

    def update_last_search_context(
        self,
        user_id: int,
        search_type: str,
        cuisine: Optional[str] = None,
        destination: Optional[str] = None,
        coordinates: Optional[Tuple[float, float]] = None,
        search_radius_km: Optional[float] = None,
        restaurants: Optional[List[Dict]] = None
    ) -> None:
        """Update last search metadata after successful search"""
        session = self.user_sessions.get(user_id)
        if not session:
            return

        # Extract restaurant names for exclusion
        shown_restaurants = []
        if restaurants:
            for r in restaurants[:20]:  # Keep last 20
                name = r.get('name') or r.get('restaurant_name') or r.get('title', '')
                if name:
                    shown_restaurants.append(name.lower().strip())

        # Update last search
        session['last_search'] = {
            'timestamp': time.time(),
            'type': search_type,
            'parameters': {
                'cuisine': cuisine,
                'destination': destination,
                'coordinates': coordinates,
                'search_radius_km': search_radius_km
            },
            'shown_restaurants': shown_restaurants
        }

        logger.info(f"✅ Updated last search for user {user_id}: {search_type}, {cuisine}, {destination}")

    def get_last_search_context(self, user_id: int) -> Dict[str, Any]:
        """Get last search context for follow-up requests"""
        session = self.user_sessions.get(user_id)
        if not session:
            return {}

        last_search = session.get('last_search', {})
        if not last_search:
            return {}

        # Return in format expected by orchestrator
        params = last_search.get('parameters', {})
        return {
            'search_type': last_search.get('type'),
            'cuisine': params.get('cuisine'),
            'destination': params.get('destination'),
            'coordinates': params.get('coordinates'),
            'search_radius_km': params.get('search_radius_km'),
            'shown_restaurants': last_search.get('shown_restaurants', []),
            'timestamp': last_search.get('timestamp')
        }

    # ============================================================================
    # MESSAGE HANDLING
    # ============================================================================

    def add_message(self, user_id: int, role: str, content: str) -> None:
        """Add a message to conversation history (in-memory only)"""
        session = self._get_or_create_session(user_id, f"msg_{user_id}")

        session['conversation_history'].append({
            'role': role,
            'message': content,
            'timestamp': time.time()
        })

        # Keep only last 10 messages in memory
        if len(session['conversation_history']) > 10:
            session['conversation_history'] = session['conversation_history'][-10:]

    def add_search_results(self, user_id: int, formatted_results: str, search_context: Optional[Dict] = None) -> None:
        """Add search results to conversation history"""
        self.add_message(user_id, 'assistant', formatted_results)

    # ============================================================================
    # GPS STATE MANAGEMENT
    # ============================================================================

    def set_pending_gps(self, user_id: int, cuisine: str):
        """Set pending GPS state - user needs to provide location"""
        session = self.user_sessions.get(user_id)
        if not session:
            session = self._get_or_create_session(user_id, f"gps_{user_id}")

        session['pending_gps'] = {
            'active': True,
            'cuisine': cuisine,
            'timestamp': time.time()
        }

        logger.info(f"📍 Set pending GPS for user {user_id}: {cuisine}")

    def clear_pending_gps(self, user_id: int):
        """Clear pending GPS state"""
        session = self.user_sessions.get(user_id)
        if session:
            session['pending_gps'] = {
                'active': False,
                'cuisine': None,
                'timestamp': None
            }
            logger.info(f"✅ Cleared pending GPS for user {user_id}")

    def get_pending_gps_state(self, user_id: int) -> Dict[str, Any]:
        """Get pending GPS state"""
        session = self.user_sessions.get(user_id)
        if not session:
            return {'pending_gps': False}

        pending = session.get('pending_gps', {})

        # Check if expired (30 minutes)
        if pending.get('active') and pending.get('timestamp'):
            age = time.time() - pending['timestamp']
            if age > 1800:  # 30 minutes
                self.clear_pending_gps(user_id)
                return {'pending_gps': False}

        return {
            'pending_gps': pending.get('active', False),
            'pending_gps_cuisine': pending.get('cuisine')
        }

    # ============================================================================
    # LOCATION CONTEXT
    # ============================================================================

    def store_location_context(
        self,
        user_id: int,
        location: str,
        coordinates: Optional[Tuple[float, float]] = None
    ) -> None:
        """Store location context with 30-minute expiry"""
        session = self.user_sessions.get(user_id)
        if not session:
            session = self._get_or_create_session(user_id, f"loc_{user_id}")

        session['stored_location'] = {
            'name': location,
            'coordinates': coordinates,
            'stored_at': time.time()
        }

        logger.info(f"📍 Stored location for user {user_id}: {location}")

    # ============================================================================
    # FORMATTING HELPERS
    # ============================================================================

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for prompt"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-10:]:
            role = msg.get('role', 'user').upper()
            content = msg.get('message', '')

            # Truncate very long messages
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"

            formatted.append(f"[{role}]: {content}")

        return "\n\n".join(formatted)

    def _format_memory_context(self, user_context: Optional[Dict[str, Any]]) -> str:
        """Format user memory context for AI prompt"""
        if not user_context:
            return "No memory available for this user (new user)."

        parts = []

        # Preferences
        prefs = user_context.get("preferences")
        if prefs:
            if hasattr(prefs, 'preferred_cuisines') and prefs.preferred_cuisines:
                parts.append(f"Preferred cuisines: {', '.join(prefs.preferred_cuisines)}")
            if hasattr(prefs, 'dietary_restrictions') and prefs.dietary_restrictions:
                parts.append(f"Dietary restrictions: {', '.join(prefs.dietary_restrictions)}")
            if hasattr(prefs, 'budget_range') and prefs.budget_range:
                parts.append(f"Budget preference: {prefs.budget_range}")

        # Restaurant history
        history = user_context.get("restaurant_history", [])
        if history:
            recent = history[:10]
            restaurant_names = [r.restaurant_name if hasattr(r, 'restaurant_name') else r.get('restaurant_name', 'Unknown') 
                              for r in recent]
            parts.append(f"Recently recommended (AVOID repeating): {', '.join(restaurant_names)}")

        if not parts:
            return "User has no stored preferences yet."

        return "\n".join(parts)

    def _format_time_ago(self, seconds: float) -> str:
        """Format time ago in human-readable format"""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} min ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days > 1 else ''} ago"

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

    # ============================================================================
    # SESSION CLEANUP
    # ============================================================================

    def clear_session(self, user_id: int):
        """Clear user session but keep location context"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            location_context = session.get('stored_location')

            # Clear active context and history
            session['active_context'] = {}
            session['conversation_history'] = []
            session['last_search'] = {}
            session['pending_gps'] = {'active': False, 'cuisine': None, 'timestamp': None}

            if location_context:
                logger.info(f"🧹 Cleared session for user {user_id} (kept location context)")
            else:
                logger.info(f"🧹 Cleared session for user {user_id}")

    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current session info"""
        return self.user_sessions.get(user_id)