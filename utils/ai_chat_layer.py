# utils/ai_chat_layer.py
"""
AI Chat Layer with Memory Integration - Single Source of Truth for Conversation Management

Key Features:
- Memory context integration (preferences, past restaurants, patterns)
- Search mode detection ("around me" vs "around [place]")
- Integrated ambiguity handling
- Context enrichment for neighborhoods
- Explicit needs_gps flag in handoffs
- Personalized responses based on user history
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
    AI Chat Layer with Memory Integration

    Responsibilities:
    - Detect search modes (GPS vs city vs neighborhood)
    - Handle ambiguous locations
    - Enrich with context
    - Decide when to search
    - Use memory context for personalized responses
    - Return structured handoffs with explicit flags
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

        # Location context storage (for follow-up searches)
        self.location_contexts: Dict[int, Dict[str, Any]] = {}

        # Initialize Supabase memory store for persistent conversation history
        self.memory_store = None
        try:
            from utils.supabase_memory_system import create_supabase_memory_store
            self.memory_store = create_supabase_memory_store(config)
            logger.info("✅ AI Chat Layer connected to persistent memory store")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize memory store: {e}")

        # Build prompts
        self._build_prompts()

        logger.info("✅ AI Chat Layer initialized with memory support")

    """
    NEW AI CHAT LAYER PROMPTS - Clean Rewrite

    Key Changes:
    1. Simplified prompt structure - removed redundant sections
    2. Clearer action definitions with explicit conditions
    3. Better GPS flow handling - when GPS is received, TRIGGER SEARCH immediately
    4. State accumulation works across conversation turns
    5. Removed unused variables

    CRITICAL FIX: When has_gps=Yes AND cuisine is known, trigger search immediately.
    The previous prompt kept asking for GPS even when it was already available.

    TO APPLY: Replace the `_build_prompts` method in utils/ai_chat_layer.py
    """

    def _build_prompts(self):
        """Build AI prompts - CLEAN REWRITE for proper state accumulation"""

        from langchain_core.prompts import ChatPromptTemplate

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the conversation manager for a restaurant recommendation bot.

    ## YOUR ROLE
    Analyze user messages, accumulate search parameters across turns, and decide when to trigger a search.

    ## REQUIRED PARAMETERS FOR SEARCH
    A search can only be triggered when you have BOTH:
    1. **CUISINE/TYPE**: What kind of food (e.g., "sushi", "Italian", "natural wine", "brunch")
    2. **LOCATION**: One of these:
       - GPS coordinates (when has_gps=Yes)
       - City name (e.g., "Tokyo", "Paris")  
       - Neighborhood/landmark that can be geocoded (e.g., "SoHo NYC", "near Eiffel Tower")

    ## OPTIONAL PARAMETERS
    - Requirements: Additional preferences (e.g., "romantic", "outdoor seating", "budget-friendly")

    ## ACTIONS (choose exactly one)

    **request_gps**: User wants nearby search but no GPS yet
    - Use when: "near me", "nearby", "close by", "around here" AND has_gps=No
    - NEVER use when: has_gps=Yes (you already have location!)

    **collect_info**: Need more information before searching
    - Use when: Missing cuisine OR missing location
    - Ask naturally, don't interrogate

    **trigger_search**: Ready to search - have both cuisine AND location
    - Use when: You have cuisine AND (has_gps=Yes OR city_name OR geocodable_location)
    - Set is_complete=true
    - Set appropriate search_mode:
      - "gps_search" if using GPS coordinates (user shared location)
      - "city_search" if searching entire city by name (e.g., "Tokyo", "Paris") - uses web scraping
      - "coordinates_search" if neighborhood/area needs geocoding first (e.g., "Alfama", "SoHo") - will be geocoded then proximity search
      - "follow_up_more_results" if user wants more options from previous search

    IMPORTANT: 
    - city_search = web scraping for best restaurants in a CITY
    - coordinates_search = geocode neighborhood/landmark, then proximity search around those coordinates

    **chat_response**: Answer questions, greet, or redirect
    - Use for: greetings, questions about shown results, off-topic queries

    ## CRITICAL RULES

    1. **WHEN GPS IS AVAILABLE (has_gps=Yes)**:
       - If cuisine is known → IMMEDIATELY trigger_search with search_mode="gps_search"
       - If cuisine unknown → collect_info to ask what food they want
       - NEVER request_gps when has_gps=Yes

    2. **ACCUMULATE STATE ACROSS TURNS**:
       - If user said "Italian" before and now shares GPS → cuisine is still "Italian"
       - Read current_cuisine and current_destination - these are already extracted!

    3. **DON'T RE-ASK FOR INFO YOU ALREADY HAVE**:
       - If current_cuisine is set, don't ask "what cuisine?"
       - If current_destination is set, don't ask "where?"
       - If has_gps=Yes, don't ask for location

    4. **PERSONALIZATION FROM MEMORY**:
       - Reference user's past preferences when relevant
       - Avoid recommending restaurants from their history

    ## RESPONSE FORMAT (JSON only, no markdown)

    {{
        "action": "request_gps" | "collect_info" | "trigger_search" | "chat_response",
        "response_text": "Your message to the user",
        "state_update": {{
            "cuisine": "extracted cuisine type or null (e.g., 'natural wine', 'sushi', 'Italian')",
            "destination": "city/neighborhood/landmark or null (e.g., 'Tokyo', 'Alfama, Lisbon')", 
            "search_mode": "gps_search|city_search|coordinates_search|follow_up_more_results|null",
            "needs_gps": true | false,
            "is_complete": true | false,
            "requirements": ["outdoor", "romantic", "budget-friendly", "good wine list", etc.],
            "raw_query": "the original user message for context preservation"
        }},
        "reasoning": "Brief explanation of your decision"
    }}

    IMPORTANT FIELDS:
    - cuisine: The TYPE of food/drink (natural wine, pizza, sushi, brunch) - NOT the full query
    - destination: The LOCATION only (Tokyo, Alfama, SoHo NYC) - NOT the full query  
    - requirements: Additional preferences extracted from the query
    - raw_query: Store the original user message to preserve context

    ## EXAMPLES

    **Example 1: User shares GPS after asking for nearby restaurants**
    Current state: cuisine="natural wine", has_gps=Yes
    User message: "[Location shared]"
    → {{
        "action": "trigger_search",
        "response_text": "Perfect! Let me find natural wine bars near you! 🍷",
        "state_update": {{"cuisine": "natural wine", "search_mode": "gps_search", "is_complete": true}},
        "reasoning": "Have cuisine from previous turn and GPS now available"
    }}

    **Example 2: User wants nearby food, no GPS yet**
    Current state: cuisine=None, has_gps=No
    User message: "Find good pizza near me"
    → {{
        "action": "request_gps",
        "response_text": "I'd love to find great pizza nearby! 📍 Please share your location.",
        "state_update": {{"cuisine": "pizza", "search_mode": "gps_search", "needs_gps": true, "raw_query": "Find good pizza near me"}},
        "reasoning": "Extracted cuisine=pizza, need GPS for 'near me' search"
    }}

    **Example 3: City search with complete info**
    Current state: cuisine=None, has_gps=No
    User message: "Best ramen in Tokyo"
    → {{
        "action": "trigger_search",
        "response_text": "Finding the best ramen in Tokyo for you! 🍜",
        "state_update": {{"cuisine": "ramen", "destination": "Tokyo", "search_mode": "city_search", "is_complete": true, "raw_query": "Best ramen in Tokyo"}},
        "reasoning": "Complete query - cuisine=ramen, city=Tokyo"
    }}

    **Example 4: Neighborhood search**
    Current state: cuisine=None, has_gps=No  
    User message: "Wine bars in Alfama, Lisbon"
    → {{
        "action": "trigger_search",
        "response_text": "Searching for wine bars in Alfama! 🍷",
        "state_update": {{"cuisine": "wine bars", "destination": "Alfama, Lisbon", "search_mode": "coordinates_search", "is_complete": true, "raw_query": "Wine bars in Alfama, Lisbon"}},
        "reasoning": "Alfama is a neighborhood - needs geocoding"
    }}

    **Example 5: Missing cuisine, have GPS**
    Current state: cuisine=None, has_gps=Yes
    User message: "[User just shared location without prior context]"
    → {{
        "action": "collect_info",
        "response_text": "Got your location! What kind of food are you in the mood for?",
        "state_update": {{"needs_gps": false}},
        "reasoning": "Have GPS but need cuisine preference"
    }}

    **Example 6: Follow-up request**
    Current state: cuisine="Italian", has_gps=Yes (results were shown)
    User message: "Show me more options"
    → {{
        "action": "trigger_search",
        "response_text": "Let me find more Italian restaurants for you!",
        "state_update": {{"search_mode": "follow_up_more_results", "is_complete": true, "raw_query": "Show me more options"}},
        "reasoning": "User wants more results in same area"
    }}

    **Example 7: Query with requirements**
    Current state: cuisine=None, has_gps=No
    User message: "Romantic Italian restaurant with outdoor seating in Paris"
    → {{
        "action": "trigger_search",
        "response_text": "Finding romantic Italian spots with outdoor seating in Paris! 🇫🇷",
        "state_update": {{"cuisine": "Italian", "destination": "Paris", "search_mode": "city_search", "is_complete": true, "requirements": ["romantic", "outdoor seating"], "raw_query": "Romantic Italian restaurant with outdoor seating in Paris"}},
        "reasoning": "Complete query with cuisine, destination, and requirements"
    }}

    **Example 8: Off-topic question**  
    Current state: cuisine=None, has_gps=No
    User message: "What's the weather like?"
    → {{
        "action": "chat_response",
        "response_text": "I'm specialized in restaurant recommendations! But I can definitely help you find a great place to eat. What cuisine are you in the mood for?",
        "state_update": {{}},
        "reasoning": "Off-topic - redirect to food while being friendly"
    }}
    """),
            ("human", """## CURRENT STATE

    **Memory Context (user's history):**
    {memory_context}

    **Accumulated Parameters:**
    - Cuisine: {current_cuisine}
    - Destination: {current_destination}  
    - GPS Available: {has_gps}
    - Stored Location Context: {stored_location}

    **Optional parameters**
    - Additional requirements: {requirements}: athmosphere, budget, dietary resctrictions, etc.

    **Recent Conversation:**
    {conversation_history}

    **Current Message:**
    {user_message}

    DECISION RULES:
    1. If has_gps=Yes AND current_cuisine is NOT "None" → trigger_search with search_mode="gps_search"
    2. If has_gps=Yes AND current_cuisine="None" → collect_info to ask for cuisine
    3. If has_gps=No AND user wants "nearby" → request_gps
    4. Don't re-ask for info you already have

    Respond with valid JSON only, no markdown code blocks.""")
        ])

        self.conversation_chain = self.conversation_prompt | self.llm


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
            if hasattr(prefs, 'preferred_cities') and prefs.preferred_cities:
                parts.append(f"Frequently searched cities: {', '.join(prefs.preferred_cities)}")
            if hasattr(prefs, 'dietary_restrictions') and prefs.dietary_restrictions:
                parts.append(f"Dietary restrictions: {', '.join(prefs.dietary_restrictions)}")
            if hasattr(prefs, 'budget_range') and prefs.budget_range:
                parts.append(f"Budget preference: {prefs.budget_range}")

        # Restaurant history
        history = user_context.get("restaurant_history", [])
        if history:
            recent = history[:10]  # Last 10 recommendations
            restaurant_names = [r.restaurant_name if hasattr(r, 'restaurant_name') else r.get('restaurant_name', 'Unknown') 
                              for r in recent]
            parts.append(f"Recently recommended restaurants (AVOID REPEATING): {', '.join(restaurant_names)}")

            # Get cities from history
            cities = list(set([r.city if hasattr(r, 'city') else r.get('city', '') for r in recent if r]))
            if cities:
                parts.append(f"Cities searched before: {', '.join(cities)}")

        # Conversation patterns
        patterns = user_context.get("conversation_patterns")
        if patterns:
            style = patterns.user_communication_style if hasattr(patterns, 'user_communication_style') else patterns.get('user_communication_style', 'casual')
            parts.append(f"Communication style: {style}")

        if not parts:
            return "User has no stored preferences yet (relatively new user)."

        return "\n".join(parts)

    def add_message(self, user_id: int, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Primary: In-memory session (fast)
        Backup: Supabase (survives restarts)

        Call this for:
        - User messages (role="user")
        - Bot responses (role="assistant") 
        - Search results sent to user (role="assistant")

        This ensures the AI sees everything shown in Telegram.
        """
        session = self._get_or_create_session(user_id, f"msg_{user_id}")

        session['conversation_history'].append({
            'role': role,
            'message': content,
            'timestamp': time.time()
        })

        # Keep only last 10 messages in memory
        if len(session['conversation_history']) > 10:
            session['conversation_history'] = session['conversation_history'][-10:]

        # Backup to Supabase (non-blocking, survives restarts)
        self._save_message_async(user_id, role, content)

        logger.debug(f"📝 Added {role} message to history for user {user_id} ({len(content)} chars)")

    def add_search_results(self, user_id: int, formatted_results: str, search_context: Optional[Dict] = None) -> None:
        """
        Add search results to conversation history.

        Call this after sending results to the user in Telegram.
        The formatted_results should be the exact text sent to the user.
        """
        self.add_message(user_id, 'assistant', formatted_results)

        if search_context:
            session = self._get_or_create_session(user_id, f"msg_{user_id}")
            session['last_search_context'] = {
                **search_context,
                'timestamp': time.time()
            }

        logger.info(f"📋 Added search results to conversation history for user {user_id}")

    async def process_message(
        self,
        user_id: int,
        user_message: str,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None  # NEW: Memory context from supervisor
    ) -> HandoffMessage:
        """
        Process user message with memory context for personalized responses.

        Args:
            user_id: Telegram user ID
            user_message: The user's message
            gps_coordinates: Optional GPS coordinates if provided
            thread_id: Thread ID for session tracking
            user_context: Memory context from supervisor (preferences, history, patterns)

        Returns:
            HandoffMessage with command and context
        """
        try:
            # Get or create session
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            session = self._get_or_create_session(user_id, thread_id)
            # Load history from DB if app just restarted (non-blocking after first call)
            await self._ensure_history_loaded(user_id)

            # Handle GPS coordinates - with 30-minute expiry
            current_gps = gps_coordinates
            if gps_coordinates:
                session['gps_coordinates'] = gps_coordinates
                session['gps_timestamp'] = time.time()  # Track when GPS was received
                logger.info(f"📍 Received GPS coordinates: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}")
            elif session.get('gps_coordinates'):
                # Check if stored GPS is still fresh (< 30 minutes)
                gps_age = time.time() - session.get('gps_timestamp', 0)
                if gps_age < 1800:  # 30 minutes
                    current_gps = session['gps_coordinates']
                    logger.info(f"📍 Using stored GPS: {current_gps[0]:.4f}, {current_gps[1]:.4f} ({gps_age/60:.0f} min old)")
                else:
                    # GPS expired - clear it
                    logger.info(f"⏰ Stored GPS expired ({gps_age/60:.0f} min old), clearing")
                    del session['gps_coordinates']
                    if 'gps_timestamp' in session:
                        del session['gps_timestamp']

            # Check for stored location context
            stored_location = self.get_location_context(user_id)
            stored_location_text = 'None'
            if stored_location:
                loc = stored_location['location']
                coords = stored_location.get('coordinates')
                age_min = (time.time() - stored_location['stored_at']) / 60
                if coords:
                    stored_location_text = f"{loc} ({coords[0]:.4f}, {coords[1]:.4f}) - {age_min:.0f} min ago"
                else:
                    stored_location_text = f"{loc} - {age_min:.0f} min ago"

            # Add user message to in-memory history (also backs up to Supabase)
            self.add_message(user_id, 'user', user_message)

            # Get accumulated state
            accumulated_state = session.get('accumulated_state', {})

            # Format memory context for AI
            memory_context_text = self._format_memory_context(user_context)

            # Log memory context usage
            if user_context and user_context.get("preferences"):
                logger.info(f"🧠 Using memory context for user {user_id}")
            else:
                logger.info(f"🆕 No memory context for user {user_id} (new user)")

            # Get current cuisine and destination from accumulated state
            current_cuisine = accumulated_state.get('cuisine') or session.get('current_cuisine') or 'None'
            current_destination = accumulated_state.get('destination') or session.get('current_destination') or 'None'

            # Prepare prompt variables - include ALL context the AI needs
            # Get requirements from accumulated state or user preferences
            current_requirements = accumulated_state.get('requirements', [])
            if not current_requirements and user_context:
                # Pull from memory if available
                prefs = user_context.get("preferences")
                if prefs:
                    if hasattr(prefs, 'dietary_restrictions') and prefs.dietary_restrictions:
                        current_requirements.extend(prefs.dietary_restrictions)
                    if hasattr(prefs, 'budget_range') and prefs.budget_range:
                        current_requirements.append(f"budget: {prefs.budget_range}")

            requirements_text = ', '.join(current_requirements) if current_requirements else 'None'

            prompt_vars = {
                'memory_context': memory_context_text,
                'conversation_history': self._format_conversation_context(session),
                'current_cuisine': current_cuisine,
                'current_destination': current_destination,
                'stored_location': stored_location_text,
                'has_gps': 'Yes' if current_gps else 'No',
                'user_message': user_message,
                'requirements': requirements_text
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
                    reasoning="JSON parse error",
                    needs_gps=False
                )

            # Extract decision fields
            action = decision.get('action', 'chat_response')
            state_update = decision.get('state_update', {})
            response_text = decision.get('response_text', '')
            reasoning = decision.get('reasoning', '')

            # Update session state
            if state_update:
                accumulated_state.update({k: v for k, v in state_update.items() if v is not None})
                session['accumulated_state'] = accumulated_state

            # Extract key fields
            search_mode = state_update.get('search_mode', '')
            destination = state_update.get('destination', '')
            cuisine = state_update.get('cuisine', '')
            requirements = state_update.get('requirements', [])
            preferences = state_update.get('preferences', {})
            is_complete = state_update.get('is_complete', False)
            needs_clarification = state_update.get('needs_clarification', False)
            needs_gps = state_update.get('needs_gps', False)

            # Store cuisine for location button flow
            if cuisine:
                session['current_cuisine'] = cuisine

            # Add assistant response to history
            session['conversation_history'].append({
                'role': 'assistant',
                'message': response_text,
                'timestamp': time.time()
            })

            logger.info(f"🤖 AI decision: action={action}, mode={search_mode}, complete={is_complete}")
 
            # ================================================================
            # RETURN APPROPRIATE HANDOFF
            # ================================================================

            # Handle GPS request
            if action == 'request_gps' or (search_mode == 'gps_required' and not current_gps):
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=response_text or "I need your location to find nearby restaurants.",
                    reasoning=reasoning,
                    needs_gps=True
                )

            # Handle clarification requests
            if action in ['chat_response', 'collect_info'] or needs_clarification:
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=response_text or "How can I help?",
                    reasoning=reasoning,
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
                            conversation_response="I couldn't find your previous search. What are you looking for?",
                            reasoning="Missing original thread_id",
                            needs_gps=False
                        )
                    return create_resume_handoff(
                        thread_id=original_thread_id,
                        decision="yes"
                    )

                # Store for follow-up
                session['last_search_time'] = time.time()
                session['last_search_thread_id'] = thread_id

                # Store location context
                if destination:
                    self.store_location_context(
                        user_id=user_id,
                        location=destination,
                        coordinates=current_gps,
                        search_type=search_mode
                    )

                # Handle geocoding for coordinates_search
                geocoded_coordinates = None
                if search_mode == 'coordinates_search' and destination and not current_gps:
                    try:
                        geocoded_coordinates = geocode_location(destination)
                        if geocoded_coordinates:
                            logger.info(f"📍 Geocoded '{destination}' to {geocoded_coordinates}")
                    except Exception as geocoding_error:
                        logger.warning(f"⚠️ Geocoding failed for '{destination}': {geocoding_error}")
                        return HandoffMessage(
                            command=HandoffCommand.CONTINUE_CONVERSATION,
                            conversation_response=f"I couldn't find the exact location for '{destination}'. Could you be more specific?",
                            reasoning=f"Geocoding failed: {str(geocoding_error)}",
                            needs_gps=True
                        )

                # Determine final coordinates
                final_coordinates = current_gps or geocoded_coordinates

                # Determine search type
                if search_mode in ['gps_required', 'coordinates_search']:
                    search_type_hint = SearchType.LOCATION_SEARCH
                else:
                    search_type_hint = SearchType.CITY_SEARCH

                logger.info(f"🔍 Creating search handoff: mode={search_mode}, type={search_type_hint.value}")

                return create_search_handoff(
                    destination=destination or "unknown",
                    cuisine=cuisine,
                    search_type=search_type_hint,
                    user_query=user_message,
                    user_id=user_id,
                    thread_id=thread_id,
                    gps_coordinates=final_coordinates,
                    requirements=requirements,
                    preferences=preferences,
                    clear_previous=False,
                    is_new_destination=False,
                    reasoning=reasoning
                )

            # Unknown action - fallback
            return HandoffMessage(
                command=HandoffCommand.CONTINUE_CONVERSATION,
                conversation_response=response_text or "Let me help you find restaurants.",
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

    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================

    def _get_or_create_session(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get or create user session, loading persistent history if available"""
        if user_id not in self.user_sessions:
            # Create new session
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'thread_id': thread_id,
                'created_at': time.time(),
                'state': ConversationState.GREETING,
                'conversation_history': [],
                'accumulated_state': {},
                'current_destination': None,
                'current_cuisine': None,
                'gps_coordinates': None,
            
                'last_search_time': None,
                'last_search_thread_id': None,
                'current_location': None,
                    'history_loaded': False  # Track if we loaded from DB
            }

            # Load persistent conversation history from Supabase
            if self.memory_store:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule loading for later if we're in async context
                        asyncio.create_task(self._load_persistent_history(user_id))
                    else:
                        loop.run_until_complete(self._load_persistent_history(user_id))
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(self._load_persistent_history(user_id))
                except Exception as e:
                    logger.warning(f"Could not load persistent history: {e}")

        return self.user_sessions[user_id]

    async def _load_persistent_history(self, user_id: int) -> None:
        """Load conversation history from Supabase into session"""
        try:
            if self.memory_store:
                history = await self.memory_store.get_conversation_history(user_id, limit=10)
                if history and user_id in self.user_sessions:
                    self.user_sessions[user_id]['conversation_history'] = history
                    logger.info(f"📜 Loaded {len(history)} messages from history for user {user_id}")
        except Exception as e:
            logger.error(f"Error loading persistent history: {e}")

    async def _ensure_history_loaded(self, user_id: int) -> None:
        """Load history from Supabase if not already loaded (only happens after restart)"""
        session = self.user_sessions.get(user_id)
        if not session or session.get('history_loaded'):
            return

        # Only load if memory is empty (app just restarted)
        if not session['conversation_history'] and self.memory_store:
            try:
                history = await self.memory_store.get_conversation_history(user_id, limit=10)
                if history:
                    session['conversation_history'] = history
                    logger.info(f"📜 Restored {len(history)} messages from DB for user {user_id}")
            except Exception as e:
                logger.warning(f"Could not load history from DB: {e}")

        session['history_loaded'] = True

    def _save_message_async(self, user_id: int, role: str, message: str) -> None:
        """Save message to Supabase in background (non-blocking fallback storage)"""
        if not self.memory_store:
            return

        import asyncio

        # Store reference to avoid None check issues
        memory_store = self.memory_store

        async def _save():
            try:
                await memory_store.add_conversation_message(user_id, role, message)
                logger.debug(f"💾 Backed up {role} message to Supabase for user {user_id}")
            except Exception as e:
                logger.warning(f"Background save failed: {e}")

        # Schedule the coroutine to actually run
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_save())
            else:
                loop.run_until_complete(_save())
        except RuntimeError:
            # No event loop - try creating one
            try:
                asyncio.run(_save())
            except Exception as e:
                logger.debug(f"Could not schedule background save: {e}")

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for AI - includes everything shown in Telegram"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-10:]:  # Last 10 messages
            role = msg.get('role', 'user').upper()
            content = msg.get('message', '')

            # Truncate very long content (like search results) but keep enough for context
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"

            formatted.append(f"[{role}]: {content}")

        return "\n\n".join(formatted)

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
    # LOCATION CONTEXT MANAGEMENT
    # ============================================================================

    def store_location_context(
        self,
        user_id: int,
        location: str,
        coordinates: Optional[Tuple[float, float]] = None,
        search_type: str = "city_search"
    ) -> None:
        """
        Store location context with automatic 30-minute expiry

        This is the SINGLE SOURCE OF TRUTH for current location.
        All layers check this method for valid location context.

        Args:
            user_id: User ID
            location: Location name/description (e.g., "Paris", "Tokyo", "Berlin")
            coordinates: Optional GPS coordinates (lat, lng)
            search_type: 'city_search' or 'coordinates_search'
        """
        session = self.user_sessions.get(user_id)
        if not session:
            # Create minimal session if doesn't exist
            session = {
                'user_id': user_id,
                'created_at': time.time()
            }
            self.user_sessions[user_id] = session

        # Store location context with timestamp in session
        session['current_location'] = {
            'location': location,
            'coordinates': coordinates,
            'search_type': search_type,
            'stored_at': time.time()
        }

        # Also store in separate dict for compatibility
        self.location_contexts[user_id] = session['current_location']

        coord_str = f" ({coordinates[0]:.4f}, {coordinates[1]:.4f})" if coordinates else ""
        logger.info(f"📍 Stored location for user {user_id}: {location}{coord_str} (expires in 30 min)")

    def get_location_context(self, user_id: int, max_age_minutes: int = 30) -> Optional[Dict[str, Any]]:
        """
        Get current location context if still valid (< 30 minutes)

        This is the SINGLE CHECK for location validity across all layers.
        Returns None if expired or not found.

        Returns:
            Dict with keys: location, coordinates, search_type, stored_at
            or None if expired/not found
        """
        session = self.user_sessions.get(user_id)
        if not session:
            # Check separate storage as fallback
            context = self.location_contexts.get(user_id)
            if context:
                age_minutes = (time.time() - context['stored_at']) / 60
                if age_minutes > max_age_minutes:
                    del self.location_contexts[user_id]
                    return None
                return context
            return None

        location_ctx = session.get('current_location')
        if not location_ctx:
            return None

        # Check if expired (default 30 minutes = 1800 seconds)
        stored_at = location_ctx.get('stored_at', 0)
        age_seconds = time.time() - stored_at
        max_age_seconds = max_age_minutes * 60

        if age_seconds > max_age_seconds:
            age_minutes = age_seconds / 60
            logger.info(f"⏰ Location context expired for user {user_id} ({age_minutes:.1f} min old)")
            # Clean up expired location
            del session['current_location']
            return None

        age_minutes = age_seconds / 60
        logger.info(f"✅ Location context valid for user {user_id}: {location_ctx['location']} ({age_minutes:.1f} min old)")
        return location_ctx

    def clear_location_context(self, user_id: int) -> None:
        """
        Clear stored location context (e.g., when destination changes)

        Call this when:
        - AI detects destination change
        - User explicitly changes location
        - Manual reset needed
        """
        session = self.user_sessions.get(user_id)
        if session and 'current_location' in session:
            old_location = session['current_location'].get('location', 'unknown')
            del session['current_location']
            logger.info(f"🗑️ Cleared location context for user {user_id} (was: {old_location})")
        else:
            logger.debug(f"ℹ️ No location context to clear for user {user_id}")

        # Also clear from separate storage if exists
        if user_id in self.location_contexts:
            del self.location_contexts[user_id]

    def get_location_age_minutes(self, user_id: int) -> Optional[float]:
        """
        Get age of stored location in minutes

        Returns:
            Age in minutes, or None if no location stored
        """
        session = self.user_sessions.get(user_id)
        if not session:
            return None

        location_ctx = session.get('current_location')
        if not location_ctx:
            return None

        stored_at = location_ctx.get('stored_at', 0)
        age_seconds = time.time() - stored_at
        return age_seconds / 60

    def clear_session(self, user_id: int):
        """Clear user session but keep location context"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            # Keep the location context (has 30-min expiry)
            location_context = session.get('current_location')

            # Clear everything else
            session['current_destination'] = None
            session['current_cuisine'] = None
            session['state'] = ConversationState.GREETING
            session['accumulated_state'] = {}
            session['last_search_time'] = None
            session['conversation_history'] = []

            if location_context:
                loc_name = location_context.get('location', 'unknown')
                logger.info(f"🧹 Cleared session for user {user_id} (kept location context: {loc_name})")
            else:
                logger.info(f"🧹 Cleared session for user {user_id}")

    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current session info"""
        return self.user_sessions.get(user_id)

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