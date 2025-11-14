# utils/ai_chat_layer.py
"""
OPTIMIZED AI Chat Layer - Single Source of Truth for Conversation Management

Key Features:
- Clearer search mode detection ("around me" vs "around [place]")
- Integrated ambiguity handling
- Context enrichment for neighborhoods
- Explicit needs_gps flag in handoffs
- No duplication with LocationAnalyzer
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
from location.geocoding import geocode_location

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
    OPTIMIZED: Single source of truth for all conversation decisions
    
    Responsibilities:
    - Detect search modes (GPS vs city vs neighborhood)
    - Handle ambiguous locations
    - Enrich with context
    - Decide when to search
    - Return structured handoffs with explicit flags
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

        logger.info("✅ AI Chat Layer initialized (optimized)")

    def _build_prompts(self):
        """Build AI prompts with state tracking and ambiguity detection"""

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the conversation manager for a restaurant bot. Your job:
1. Accumulate info across conversation turns
2. Detect search mode based on user intent  
3. Decide when you have enough info to search
4. Handle ambiguous locations naturally

═══════════════════════════════════════════════════════════════
SEARCH MODES
═══════════════════════════════════════════════════════════════

GPS_REQUIRED - User wants restaurants near their current location
├─ Triggers: "near me", "nearby", "around me", "close to me", "in my area", 
│           "where I am", "walking distance", "5 mins away"
├─ Requires: GPS coordinates
└─ Action: If no GPS → set needs_gps=true

CITY_SEARCH - City or popular tourist area (use web search)
├─ Examples: "in Tokyo", "Paris restaurants", "Manhattan", "Brooklyn", 
│           "Marais", "SoHo", "Shibuya", "Tuscany"
├─ Use for: Well-known areas that have online guides/articles
├─ Query pattern: "best [cuisine] in [area]"
└─ Action: Web scraping search, no GPS needed

COORDINATES_SEARCH - Specific street/landmark/less-known location
├─ Examples: "around Viale delle Egadi", "near Pantheon", "close to Via delle Palme",
│           "on Rua Augusta", "by the Colosseum", "around [specific street name]"
├─ Use for: Specific addresses, local landmarks, non-tourist spots
├─ Pattern: "around/near/close to [specific place]" + has city context
├─ Requires: Specific location + city (use stored city if available)
└─ Action: Coordinates-based Google Maps search

TOURIST_AREA_CLARIFICATION - Could be either city or coordinates
├─ Examples: "Chinatown" (NYC or SF?), "Lapa" (Lisbon or Rio?)
├─ Action: Ask user which city they mean
└─ Then decide: if major tourist area → CITY_SEARCH, if specific → COORDINATES_SEARCH

═══════════════════════════════════════════════════════════════
AMBIGUITY HANDLING
═══════════════════════════════════════════════════════════════

Detect ambiguous locations and ask natural clarifying questions:

AMBIGUOUS: "restaurants in Springfield"
→ is_ambiguous=true, needs_clarification=true
→ "There are several Springfields! Which state did you mean? (e.g., Massachusetts, Illinois, Missouri)"

AMBIGUOUS: "coffee in Cambridge"
→ is_ambiguous=true, needs_clarification=true
→ "Cambridge in England or Cambridge, Massachusetts?"

CLEAR: "restaurants in Lapa, Lisbon" → NOT ambiguous
CLEAR: "around Viale delle Egadi in Rome" → NOT ambiguous, this is NEIGHBORHOOD_SEARCH

═══════════════════════════════════════════════════════════════
CONTEXT ENRICHMENT
═══════════════════════════════════════════════════════════════

Use stored city context ONLY for NEIGHBORHOOD_SEARCH:

User previously searched "Lisbon", new query: "coffee in Lapa"
→ Enrich to "Lapa, Lisbon", search_mode="neighborhood_search"

User previously searched "Valencia", new query: "coffee near me"
→ DO NOT use Valencia, request GPS, search_mode="gps_required"

═══════════════════════════════════════════════════════════════
STATE STRUCTURE
═══════════════════════════════════════════════════════════════

{{
    "destination": "full location string" | null,
    "cuisine": "what they want" | null,
    "search_mode": "gps_required|city_search|coordinates_search|follow_up" | null,
    "needs_gps": true | false,
    "is_ambiguous": true | false,
    "needs_clarification": true | false,
    "is_complete": true | false
}}

═══════════════════════════════════════════════════════════════
DECISION RULES
═══════════════════════════════════════════════════════════════

1. GPS_REQUIRED: NEVER use stored city, ALWAYS request GPS if not provided
2. NEIGHBORHOOD_SEARCH: CAN use stored city, check if ambiguous
3. CITY_SEARCH: City specified, check if ambiguous
4. Set is_complete=true ONLY when: have enough info AND NOT ambiguous

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

"specialty coffee near me"
→ {{
    "search_mode": "gps_required",
    "cuisine": "specialty coffee",
    "needs_gps": true,
    "is_complete": false,
    "action": "request_gps",
    "response_text": "I'd love to help you find specialty coffee near you!",
    "reasoning": "GPS required mode, need coordinates"
}}

"Find good places around Viale delle Egadi in Rome"
→ {{
    "search_mode": "coordinates_search",
    "destination": "Viale delle Egadi, Rome",
    "cuisine": "restaurants",
    "needs_gps": false,
    "is_ambiguous": false,
    "is_complete": true,
    "action": "trigger_search",
    "response_text": "Searching for restaurants around Viale delle Egadi in Rome!",
    "reasoning": "Specific street name = coordinates search using Google Maps"
}}

"best pizza in Marais" (with stored city: "Paris")
→ {{
    "search_mode": "city_search",
    "destination": "Marais, Paris",
    "cuisine": "pizza",
    "needs_gps": false,
    "is_ambiguous": false,
    "is_complete": true,
    "action": "trigger_search",
    "response_text": "Searching for the best pizza in Marais!",
    "reasoning": "Marais is a popular tourist area, likely has online guides"
}}

"restaurants near Pantheon" (with stored city: "Rome")
→ {{
    "search_mode": "coordinates_search",
    "destination": "Pantheon, Rome",
    "cuisine": "restaurants",
    "needs_gps": false,
    "is_ambiguous": false,
    "is_complete": true,
    "action": "trigger_search",
    "response_text": "Finding restaurants near the Pantheon!",
    "reasoning": "Specific landmark = coordinates search for precise location"
}}

"best ramen in Tokyo"
→ {{
    "search_mode": "city_search",
    "destination": "Tokyo",
    "cuisine": "ramen",
    "needs_gps": false,
    "is_ambiguous": false,
    "is_complete": true,
    "action": "trigger_search",
    "response_text": "Searching for the best ramen in Tokyo!",
    "reasoning": "Clear city search"
}}

"restaurants in Springfield"
→ {{
    "search_mode": "city_search",
    "destination": "Springfield",
    "cuisine": "restaurants",
    "is_ambiguous": true,
    "needs_clarification": true,
    "is_complete": false,
    "action": "collect_info",
    "response_text": "There are several Springfields! Which state did you mean? (Massachusetts, Illinois, Missouri, etc.)",
    "reasoning": "Ambiguous city name, need clarification"
}}

"coffee in Lapa" (with stored city: "Lisbon")
→ {{
    "search_mode": "neighborhood_search",
    "destination": "Lapa, Lisbon",
    "cuisine": "coffee",
    "needs_gps": false,
    "is_ambiguous": false,
    "is_complete": true,
    "action": "trigger_search",
    "response_text": "Searching for coffee in Lapa, Lisbon!",
    "reasoning": "Neighborhood enriched with stored city context"
}}

"coffee in Lapa" (NO stored city)
→ {{
    "search_mode": "neighborhood_search",
    "destination": "Lapa",
    "cuisine": "coffee",
    "is_ambiguous": true,
    "needs_clarification": true,
    "is_complete": false,
    "action": "collect_info",
    "response_text": "I found several neighborhoods called Lapa! Did you mean Lapa in Lisbon, Lapa in São Paulo, or somewhere else?",
    "reasoning": "Neighborhood without city - could be multiple places"
}}

User says "Let's find more" (after seeing results)
→ {{
    "search_mode": "follow_up",
    "is_complete": true,
    "action": "trigger_search",
    "response_text": "Perfect! Finding more options for you...",
    "reasoning": "Follow-up request, resume graph"
}}

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT (JSON ONLY)
═══════════════════════════════════════════════════════════════

{{
    "action": "request_gps" | "collect_info" | "trigger_search",
    "response_text": "what to say to user",
    "state_update": {{
        "destination": "full location" | null,
        "cuisine": "what they want" | null,
        "search_mode": "gps_required|city_search|neighborhood_search|follow_up" | null,
        "needs_gps": true | false,
        "is_ambiguous": true | false,
        "needs_clarification": true | false,
        "is_complete": true | false
    }},
    "reasoning": "brief explanation"
}}
"""),
            ("human", """CONVERSATION HISTORY:
{conversation_history}

CURRENT STATE:
{accumulated_state}

STORED CONTEXT:
- Last searched city: {last_searched_city}

GPS COORDINATES: {has_gps}

USER MESSAGE: {user_message}

Detect search mode and decide action.""")
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
        Process user message with AI-detected search mode and early geocoding

        Returns structured HandoffMessage based on AI decision
        """
        try:
            # Get or create session
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            session = self._get_or_create_session(user_id, thread_id)

            # CRITICAL FIX: Store GPS if provided AND use it for this turn
            current_gps = gps_coordinates
            if gps_coordinates:
                session['gps_coordinates'] = gps_coordinates
                logger.info(f"📍 Stored GPS coordinates: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}")
            elif session.get('gps_coordinates'):
                # Use previously stored GPS if available
                current_gps = session['gps_coordinates']
                logger.info(f"📍 Using stored GPS coordinates: {current_gps[0]:.4f}, {current_gps[1]:.4f}")

            # Add user message to history
            session['conversation_history'].append({
                'role': 'user',
                'message': user_message,
                'timestamp': time.time()
            })

            # Get current state
            current_state = session.get('state', ConversationState.GREETING)
            accumulated_state = session.get('accumulated_state', {})

            # Prepare prompt variables
            prompt_vars = {
                'conversation_history': self._format_conversation_context(session),
                'accumulated_state': json.dumps(accumulated_state, indent=2),
                'last_searched_city': session.get('last_searched_city') or 'None',
                'has_gps': 'Yes' if current_gps else 'No',
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
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response="How can I help you find restaurants?",
                    reasoning="JSON parse error - using fallback",
                    needs_gps=False
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
            is_ambiguous = accumulated_state.get('is_ambiguous', False)
            needs_clarification = accumulated_state.get('needs_clarification', False)

            # Update session
            session['current_destination'] = destination
            session['current_cuisine'] = cuisine

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
                    conversation_response=decision.get('response_text', 'I\'d love to help you find restaurants near you!'),
                    needs_gps=True
                )

            # Handle clarification requests (ambiguous locations)
            if action in ['chat_response', 'collect_info'] or needs_clarification:
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=decision.get('response_text', "How can I help?"),
                    reasoning=decision.get('reasoning', ''),
                    needs_gps=False
                )

            # Handle search trigger
            if action == 'trigger_search' and is_complete:
                # Check if follow-up request
                if search_mode == 'follow_up_more_results':
                    original_thread_id = session.get('last_search_thread_id')

                    if not original_thread_id:
                        return HandoffMessage(
                            command=HandoffCommand.CONTINUE_CONVERSATION,
                            conversation_response="I couldn't find your previous search. Could you tell me what you're looking for?",
                            reasoning="Missing original thread_id",
                            needs_gps=False
                        )

                    return create_resume_handoff(
                        thread_id=original_thread_id,
                        decision="accept"
                    )

                # Regular new search
                session['last_search_time'] = time.time()
                session['last_search_thread_id'] = thread_id

                # Extract city from destination for context storage
                if destination and (search_mode == 'city_search' or search_mode == 'coordinates_search'):
                    city = self._extract_city_from_destination(destination)
                    if city:
                        session['last_searched_city'] = city
                        logger.info(f"💾 Stored city context for user {user_id}: {city}")

                # ================================================================
                # EARLY GEOCODING FOR COORDINATES_SEARCH MODE
                # ================================================================
                geocoded_coordinates = None

                if search_mode == 'coordinates_search' and destination and not current_gps:
                    logger.info(f"🌍 Early geocoding for coordinates_search: '{destination}'")

                    try:
                        # Attempt to geocode the destination
                        geocoded_coordinates = geocode_location(destination)

                        if geocoded_coordinates:
                            logger.info(f"✅ Successfully geocoded '{destination}' → {geocoded_coordinates[0]:.4f}, {geocoded_coordinates[1]:.4f}")
                            # Store in session for future use
                            session['last_geocoded_location'] = {
                                'address': destination,
                                'coordinates': geocoded_coordinates,
                                'timestamp': time.time()
                            }
                        else:
                            # Geocoding failed - ask for clarification
                            logger.warning(f"❌ Failed to geocode '{destination}'")

                            clarification_message = (
                                f"I couldn't find '{destination}' on the map. 🗺️\n\n"
                                f"Could you help me out? Try one of these:\n\n"
                                f"• <b>Be more specific</b>: Add the city name\n"
                                f"  Example: \"Viale delle Egadi, <i>Rome</i>\"\n\n"
                                f"• <b>Use a landmark</b>: Reference something nearby\n"
                                f"  Example: \"Near the Colosseum\"\n\n"
                                f"• <b>Share your GPS</b>: Use the button below for exact location"
                            )

                            return HandoffMessage(
                                command=HandoffCommand.CONTINUE_CONVERSATION,
                                conversation_response=clarification_message,
                                reasoning=f"Geocoding failed for: {destination}",
                                needs_gps=True  # Offer GPS button as alternative
                            )

                    except Exception as geocoding_error:
                        logger.error(f"❌ Geocoding error for '{destination}': {geocoding_error}")

                        # Graceful fallback - ask user for help
                        return HandoffMessage(
                            command=HandoffCommand.CONTINUE_CONVERSATION,
                            conversation_response=(
                                f"I'm having trouble finding '{destination}' on the map. "
                                f"Could you provide more details or share your GPS location?"
                            ),
                            reasoning=f"Geocoding exception: {str(geocoding_error)}",
                            needs_gps=True
                        )

                # Determine which coordinates to use
                final_coordinates = current_gps or geocoded_coordinates

                if search_mode in ['coordinates_search', 'gps_required'] and final_coordinates:
                    logger.info(f"📍 Using coordinates for search: {final_coordinates[0]:.4f}, {final_coordinates[1]:.4f}")
                    logger.info(f"   Source: {'GPS' if current_gps else 'Geocoded'}")

                # Determine search type hint based on search mode
                if search_mode == 'gps_required' or search_mode == 'coordinates_search':
                    search_type_hint = SearchType.LOCATION_SEARCH
                else:
                    search_type_hint = SearchType.CITY_SEARCH

                # Create search handoff with coordinates (GPS or geocoded)
                logger.info(f"🔍 Creating search handoff:")
                logger.info(f"   Mode: {search_mode}")
                logger.info(f"   Type: {search_type_hint.value}")
                logger.info(f"   Destination: {destination}")
                logger.info(f"   Coordinates: {final_coordinates}")

                return create_search_handoff(
                    destination=destination or "unknown",
                    cuisine=cuisine,
                    search_type=search_type_hint,
                    user_query=user_message,
                    user_id=user_id,
                    thread_id=thread_id,
                    gps_coordinates=final_coordinates,  # GPS or geocoded coordinates
                    requirements=requirements,
                    preferences=preferences,
                    clear_previous=False,
                    is_new_destination=False,
                    reasoning=decision.get('reasoning', '')
                )

            # Unknown action - fallback
            return HandoffMessage(
                command=HandoffCommand.CONTINUE_CONVERSATION,
                conversation_response=decision.get('response_text', "Let me help you find restaurants."),
                reasoning=f"Unknown action: {action}",
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

    def _extract_city_from_destination(self, destination: str) -> Optional[str]:
        """
        Extract city name from destination string for context storage

        Examples:
        "Lapa, Lisbon" → "Lisbon"
        "SoHo, New York" → "New York"
        "Paris" → "Paris"
        """
        if not destination:
            return None

        parts = [p.strip() for p in destination.split(',')]

        if len(parts) > 1:
            return parts[-1]

        return parts[0]

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for AI"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

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
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'thread_id': thread_id,
                'created_at': time.time(),
                'state': ConversationState.GREETING,
                'conversation_history': [],
                'accumulated_state': {},
                'current_destination': None,
                'current_cuisine': None,
                'last_searched_city': None,
                'gps_coordinates': None,
                'last_search_time': None,
                'last_search_thread_id': None
            }
        return self.user_sessions[user_id]

    def clear_session(self, user_id: int):
        """Clear user session but keep last_searched_city"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            last_city = session.get('last_searched_city')

            # Clear everything else
            session['current_destination'] = None
            session['current_cuisine'] = None
            session['state'] = ConversationState.GREETING
            session['accumulated_state'] = {}
            session['last_search_time'] = None

            # Restore city context
            session['last_searched_city'] = last_city

            logger.info(f"🧹 Cleared session for user {user_id} (kept city context: {last_city})")

    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current session info"""
        return self.user_sessions.get(user_id)
