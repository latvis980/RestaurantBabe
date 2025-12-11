# utils/ai_chat_layer.py
"""
AI Chat Layer with Context-Aware Parameter Management

ARCHITECTURE:
- Conversation history provides implicit context (last 10 messages)
- Active context tracks current search parameters (destination, cuisine, radius, etc.)
- Conversation summary provides compressed understanding of user's evolving intent
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
- NEW: Conversation summary for better intent tracking
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
    - Maintain conversation summary for intent tracking
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
        self._build_summary_prompt()

        logger.info("✅ AI Chat Layer initialized with context-aware parameter management")

    def _build_summary_prompt(self):
        """Build prompt for updating conversation summary"""
        
        summary_prompt = """You are a conversation summarizer for a restaurant recommendation bot.

Your task is to update the conversation summary based on new messages.

CURRENT SUMMARY:
{current_summary}

NEW MESSAGES:
{new_messages}

RULES FOR UPDATING SUMMARY:
1. Preserve key search intent information:
   - WHAT they're looking for (cuisine type, dining style, specific dishes)
   - WHERE they want to go (city, neighborhood, specific location, "near me")
   - WHY/preferences (atmosphere, price range, occasion, dietary needs)
   - FEEDBACK on results shown (liked/disliked, what was wrong)

2. Track evolution of user intent:
   - If user refines request, note what changed and what stayed the same
   - If user expresses dissatisfaction, note what criteria wasn't met
   - If user asks for "more", "different", "similar", note the comparison basis

3. Keep it concise but informative (2-4 sentences max)

4. Format: "User is looking for [WHAT] in/around [WHERE]. [Additional context about preferences/refinements/feedback]"

EXAMPLES:
- "User is looking for wine bars around Prado Museum in Madrid. After seeing Michelin-starred options, they want something more casual."
- "User wants Italian restaurants in SoHo, NYC. They prefer trendy spots with good pasta and moderate prices. Previous suggestions were too expensive."
- "User is looking for brunch spots near their current GPS location in Brooklyn. They want somewhere with outdoor seating."

Return ONLY the updated summary, nothing else."""

        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", summary_prompt),
            ("human", "Update the summary based on the new messages above.")
        ])

    def _build_conversation_prompt(self):
        """Build the main conversation prompt with context-aware parameter tracking"""

        system_prompt = """You are an AI conversation manager for a restaurant recommendation bot.

Your job is to analyze user messages and decide what to do based on the FULL conversation context.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: CONTEXT PRESERVATION RULES
═══════════════════════════════════════════════════════════════════════════════

**BEFORE making any decision, you MUST reason through these questions:**

1. **What is the user's CURRENT search context?** (from active context + conversation history)
   - Location/area they're interested in
   - Type of place they're looking for
   - Any specific requirements or preferences

2. **What is the user EXPLICITLY changing?**
   - Only consider parameters the user DIRECTLY mentions in their new message
   - "More casual" = changing STYLE, NOT location or cuisine type
   - "Closer" = changing DISTANCE, NOT cuisine or style
   - "Something cheaper" = changing PRICE, NOT location or cuisine

3. **What should STAY THE SAME?**
   - If user doesn't mention location → KEEP the current location/coordinates
   - If user doesn't mention cuisine type → KEEP the current cuisine type
   - If user doesn't mention area → KEEP the current area
   - NEVER reset parameters that weren't explicitly changed

**MODIFICATION EXAMPLES (learn from these):**

| User said | Previous context | What to MODIFY | What to KEEP |
|-----------|------------------|----------------|--------------|
| "More casual" | Wine bars, Prado Museum | Style/atmosphere | Location (Prado), Type (wine bars) |
| "Closer to me" | Italian, 2km radius | Radius (reduce) | Cuisine (Italian), Location |
| "Actually, pizza" | Sushi, Tokyo | Cuisine (pizza) | Location (Tokyo) |
| "In Shibuya instead" | Ramen, Tokyo | Location (Shibuya) | Cuisine (ramen), Style |
| "Something cheaper" | Fine dining, Paris | Price range | Location (Paris), cuisine |
| "Show me more" | Coffee shops, Brooklyn | Nothing | Everything (CONTINUE) |

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION CONTEXT
═══════════════════════════════════════════════════════════════════════════════

**CONVERSATION SUMMARY (compressed understanding of user's evolving intent):**
{conversation_summary}

**RECENT CONVERSATION (last 10 messages for detail):**
{conversation_history}

**USER PREFERENCES FROM MEMORY:**
{memory_context}

**ACTIVE SEARCH CONTEXT (current search parameters):**
- Destination: {active_destination}
- Cuisine/Type: {active_cuisine}
- Search radius: {active_radius}km
- Requirements: {active_requirements}
- Established: {context_age}
- Searches performed: {search_count}

**LAST SEARCH RESULTS (shown {time_ago}):**
- Restaurants shown: {shown_restaurants}

**STORED COORDINATES (from previous location search, valid for 30 min):**
{last_search_coordinates}

**STORED LOCATION:**
{stored_location}

**PENDING GPS STATE:**
- Waiting for GPS: {pending_gps}
- For cuisine: {pending_gps_cuisine}

═══════════════════════════════════════════════════════════════════════════════
CURRENT USER MESSAGE
═══════════════════════════════════════════════════════════════════════════════

"{user_message}"

GPS Coordinates provided now: {has_gps}

═══════════════════════════════════════════════════════════════════════════════
YOUR DECISION PROCESS (follow this step by step)
═══════════════════════════════════════════════════════════════════════════════

**STEP 1: Analyze what the user is asking**
- Is this a new search request?
- Is this a modification of current search?
- Is this a request for more of the same?
- Is this a general question or off-topic?

**STEP 2: If modification or continuation, identify what's preserved**
Think carefully: The user said "{user_message}"
- Did they mention a NEW location? If NO → use {active_destination} and coordinates from last search
- Did they mention a NEW cuisine type? If NO → use {active_cuisine}
- Did they mention a NEW area/neighborhood? If NO → use stored location/coordinates
- Did they ask for "more", "different style", "more casual", etc.? → This is MODIFY, NOT new search

**STEP 3: Determine context decision type**

**CONTINUE**: User wants MORE of the same (exact same parameters)
- Triggers: "show more", "any other options", "what else", "more like these"
- Action: Keep ALL parameters identical, just get more results
- CRITICAL: Preserve coordinates/location from last search

**MODIFY**: User wants to REFINE current search (change 1-2 parameters)
- Triggers: "more casual", "closer", "cheaper", "different cuisine", "in [new area]"
- Action: Change ONLY what user explicitly mentioned, PRESERVE everything else
- CRITICAL: If user says "more casual" but NOT a new location → keep same location/coordinates
- CRITICAL: If user says "closer" but NOT a new cuisine → keep same cuisine

**NEW**: Completely different search topic
- Triggers: New city + new cuisine, unrelated question, explicit restart
- Action: Fresh parameters, clear old context
- Only use NEW when user is clearly starting over

**STEP 4: Determine search mode**
- CITY_SEARCH: City-wide search (just city name, no neighborhood)
- LOCATION_SEARCH: Specific area search - USE THIS when:
  * User mentioned neighborhood/landmark ("near Prado Museum", "in SoHo")
  * User has stored coordinates from previous search
  * User provided GPS
  * Previous search was location-based and user is refining

**STEP 5: Handle coordinates properly**
- If MODIFY or CONTINUE and previous search had coordinates → REUSE those coordinates
- If user mentions new location → geocode will handle it
- If user says "near me" without GPS → request_gps
- NEVER lose coordinates on a MODIFY request

═══════════════════════════════════════════════════════════════════════════════
SEARCH MODES (critical for correct routing)
═══════════════════════════════════════════════════════════════════════════════

**CITY_SEARCH** (city-wide, uses database + web scraping):
- "Best sushi in Tokyo"
- "Wine bars in Madrid"
- Format destination as: "City" (no comma)

**LOCATION_SEARCH** (nearby area, uses database + Google Maps):
- "Coffee near Prado Museum"
- "Restaurants around me" (with GPS)
- "Pizza in SoHo, New York"
- Format destination as: "Neighborhood, City" OR use stored coordinates

**GPS LOGIC:**
| User says | Location mentioned? | Action |
|-----------|---------------------|--------|
| "Find wine bars near Prado Museum" | YES (Prado Museum) | execute_search, LOCATION_SEARCH |
| "Find food near me" | NO | request_gps |
| "More casual" (after location search) | NO but have coords | execute_search, LOCATION_SEARCH with stored coords |

═══════════════════════════════════════════════════════════════════════════════
GPS HANDLING (CRITICAL - understand the system flow)
═══════════════════════════════════════════════════════════════════════════════

**HOW THE SYSTEM WORKS:**
- If you provide a destination (landmark, neighborhood, street, etc.) → System will GEOCODE it automatically
- Geocoding converts "Prado Museum, Madrid" → GPS coordinates behind the scenes
- You do NOT need user's GPS when they mention a place name!

**REQUEST GPS (action: request_gps) ONLY when:**
- User says "near me", "around me", "close to me", "nearby" with NO place name
- There is ZERO location information in the message
- Examples: "Find pizza near me", "What's good nearby?", "Restaurants close by"

**USE execute_search WITH LOCATION_SEARCH when:**
- User mentions ANY place name (landmark, neighborhood, street, district, area)
- Even with words like "around", "near", "close to" - if there's a PLACE NAME, search!
- The destination field will be geocoded automatically - you don't need GPS!

**GPS EXAMPLES:**

| User Message | Action | Why |
|--------------|--------|-----|
| "Wine bars around Prado Museum in Madrid" | execute_search | Has location: "Prado Museum, Madrid" → will be geocoded |
| "Restaurants near Times Square" | execute_search | Has location: "Times Square, New York" → will be geocoded |
| "Cafes in Shibuya" | execute_search | Has location: "Shibuya, Tokyo" → will be geocoded |
| "Pizza near me" | request_gps | NO location mentioned, need user's GPS |
| "What's good nearby?" | request_gps | NO location mentioned, need user's GPS |
| "Find food close to me" | request_gps | NO location mentioned, need user's GPS |

**DECISION RULE:**
- Can you extract a place name from the message? → execute_search (geocoding handles it)
- No place name at all, just "near me/nearby"? → request_gps

**GEOGRAPHICAL AWARENESS:**
- If user mentions a destination that requires clarification (the name is not widely known), naturally ask to confirm
- If the location is ambiguous (e.g., "Springfield"), ask for clarification
- If the user mentions only a neighborhood/street/landmark without city, naturally ask to confirm the city

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only)
═══════════════════════════════════════════════════════════════════════════════

{{
  "reasoning_steps": {{
    "user_intent": "What is the user trying to do?",
    "explicit_changes": "What parameters did user EXPLICITLY mention changing?",
    "preserved_params": "What parameters should stay the same from context?",
    "location_handling": "How am I handling location/coordinates?"
  }},

  "action": "execute_search" | "chat_response" | "request_gps",

  "context_decision": {{
    "type": "CONTINUE" | "MODIFY" | "NEW",
    "reasoning": "Why this decision - what's being preserved vs changed"
  }},

  "parameters": {{
    "destination": "city OR 'Neighborhood, City' - PRESERVE from context unless explicitly changed",
    "cuisine": "type of place - PRESERVE from context unless explicitly changed",
    "search_mode": "CITY_SEARCH" | "LOCATION_SEARCH",
    "search_radius_km": 1.5,
    "requirements": ["list of requirements - combine existing + new"],
    "preferences": {{}},
    "modifications": {{
      "changed_field": {{"from": "old", "to": "new", "reason": "user said X"}}
    }}
  }},

  "response_text": "your response to the user",
  "reasoning": "internal reasoning for your decision",
  "needs_gps": false,
  "summary_update": "Brief note for updating conversation summary",

  "state_update": {{
    "clear_pending_gps": false
  }}
}}

═══════════════════════════════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════════════════════════════

1. **NEVER lose location context on modifications**
   - "More casual" after "wine bars near Prado" → Keep "Prado Museum, Madrid"
   - "Closer options" after area search → Keep same area, reduce radius
   - "Different cuisine" after location search → Keep location, change cuisine

2. **ALWAYS fill in parameters from context**
   - If user doesn't specify destination → use {active_destination}
   - If user doesn't specify cuisine → use {active_cuisine}
   - If MODIFY/CONTINUE and we have stored coordinates → use LOCATION_SEARCH

3. **Qualitative modifiers = MODIFY, not NEW**
   - "more casual", "more upscale", "cheaper", "closer", "different vibe"
   - These refine the search, they don't start a new one
   - Preserve location and basic type unless explicitly changed

4. **Use LOCATION_SEARCH when we have coordinates**
   - Previous search stored coordinates → can reuse for MODIFY/CONTINUE
   - User mentioned specific place → geocode and search nearby
   - User provided GPS → use for nearby search

5. **Comma in destination = neighborhood search**
   - "SoHo, New York" → LOCATION_SEARCH
   - "Madrid" → CITY_SEARCH
"""

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Analyze the message above and provide your decision in JSON format.")
        ])

    # ============================================================================
    # CONVERSATION SUMMARY MANAGEMENT
    # ============================================================================

    async def _update_conversation_summary(
        self,
        user_id: int,
        new_messages: List[Dict[str, str]],
        current_summary: str
    ) -> str:
        """Update conversation summary with new messages"""
        try:
            if not new_messages:
                return current_summary

            # Format new messages
            messages_text = "\n".join([
                f"[{msg.get('role', 'user').upper()}]: {msg.get('message', '')}"
                for msg in new_messages
            ])

            prompt_vars = {
                'current_summary': current_summary or "No previous summary.",
                'new_messages': messages_text
            }

            response = await self.llm.ainvoke(
                self.summary_prompt.format_messages(**prompt_vars)
            )

            new_summary = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"📝 Updated conversation summary for user {user_id}: {new_summary[:100]}...")
            return new_summary.strip()

        except Exception as e:
            logger.error(f"Error updating conversation summary: {e}")
            return current_summary or ""

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

            # Get stored coordinates from last search (for LOCATION_SEARCH continuations)
            last_search_coords_text = "None"
            last_search_params = last_search.get('parameters', {})
            last_search_coords = last_search_params.get('coordinates')
            if last_search_coords and last_search_timestamp:
                age_min = (time.time() - last_search_timestamp) / 60
                if age_min < 30:  # 30-minute expiry
                    last_search_coords_text = f"({last_search_coords[0]:.4f}, {last_search_coords[1]:.4f}) - {int(age_min)} min ago"

            # Get or update conversation summary
            conversation_summary = session.get('conversation_summary', '')
            
            # Update summary periodically (every 3 messages or on significant changes)
            messages_since_summary = session.get('messages_since_summary', 0)
            if messages_since_summary >= 2 or not conversation_summary:
                recent_messages = session.get('conversation_history', [])[-3:]
                conversation_summary = await self._update_conversation_summary(
                    user_id, recent_messages, conversation_summary
                )
                session['conversation_summary'] = conversation_summary
                session['messages_since_summary'] = 0
            else:
                session['messages_since_summary'] = messages_since_summary + 1

            # Format memory context
            memory_context_text = self._format_memory_context(user_context)

            # Prepare prompt variables
            prompt_vars = {
                'conversation_summary': conversation_summary or "No summary yet - this is the start of conversation.",
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
                'last_search_coordinates': last_search_coords_text,
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
            # Ensure it's a string for parsing
            if not isinstance(response_content, str):
                response_content = str(response_content)
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
            reasoning_steps = decision.get('reasoning_steps', {})

            # Log detailed reasoning
            logger.info(f"🤖 AI Decision: action={action}, context={context_type}")
            logger.info(f"   Reasoning steps: {reasoning_steps}")
            logger.info(f"   Parameters: destination={parameters.get('destination')}, cuisine={parameters.get('cuisine')}, mode={parameters.get('search_mode')}")

            # Handle clear_pending_gps signal
            if state_update.get('clear_pending_gps'):
                self.clear_pending_gps(user_id)

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

                    # IMPORTANT: For MODIFY, preserve coordinates from last search if not explicitly changing location
                    preserved_destination = parameters.get('destination') or active_context.get('destination')
                    
                    active_context.update({
                        'destination': preserved_destination,
                        'cuisine': parameters.get('cuisine') or active_context.get('cuisine'),
                        'search_radius_km': parameters.get('search_radius_km', active_context.get('search_radius_km', 1.5)),
                        'requirements': parameters.get('requirements', active_context.get('requirements', [])),
                        'preferences': parameters.get('preferences', active_context.get('preferences', {})),
                        'last_modified': time.time()
                    })
                    session['active_context'] = active_context

                    modifications = parameters.get('modifications', {})
                    logger.info(f"✏️ MODIFIED context: {modifications}")
                    logger.info(f"   Preserved destination: {preserved_destination}")
                    logger.info(f"   Preserved cuisine: {active_context.get('cuisine')}")

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

                # IMPORTANT: For MODIFY/CONTINUE, try to preserve coordinates from last search
                effective_gps = gps_coordinates
                if not effective_gps and context_type in ['MODIFY', 'CONTINUE']:
                    # Check if we have stored coordinates from last location search
                    if last_search_coords and (time.time() - last_search_timestamp) < 1800:  # 30 min
                        effective_gps = tuple(last_search_coords) if isinstance(last_search_coords, list) else last_search_coords
                        logger.info(f"📍 Reusing coordinates from last search: {effective_gps}")

                # Get effective destination (preserve from context if MODIFY/CONTINUE)
                effective_destination = parameters.get('destination', '')
                if not effective_destination and context_type in ['MODIFY', 'CONTINUE']:
                    effective_destination = active_context.get('destination', '')
                    logger.info(f"📍 Preserved destination from context: {effective_destination}")

                # Get effective cuisine
                effective_cuisine = parameters.get('cuisine')
                if not effective_cuisine and context_type in ['MODIFY', 'CONTINUE']:
                    effective_cuisine = active_context.get('cuisine')
                    logger.info(f"🍽️ Preserved cuisine from context: {effective_cuisine}")

                # Build search context
                search_context = SearchContext(
                    destination=effective_destination,
                    cuisine=effective_cuisine,
                    search_type=search_type,
                    gps_coordinates=effective_gps,
                    search_radius_km=parameters.get('search_radius_km', 1.5),
                    requirements=parameters.get('requirements', []),
                    preferences=parameters.get('preferences', {}),
                    user_query=user_message,
                    is_follow_up=(context_type in ['CONTINUE', 'MODIFY']),
                    exclude_restaurants=shown_restaurants if context_type == 'CONTINUE' else [],
                    user_id=user_id,
                    thread_id=thread_id or f"chat_{user_id}",
                    supervisor_instructions=f"Context: {context_type}. {context_decision.get('reasoning', '')}. User wants: {reasoning_steps.get('user_intent', '')}"
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

                # Conversation summary (compressed understanding)
                'conversation_summary': '',
                'messages_since_summary': 0,

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
            for r in restaurants:
                name = r.get('name') if isinstance(r, dict) else getattr(r, 'name', None)
                if name:
                    shown_restaurants.append(name)

        session['last_search'] = {
            'search_type': search_type,
            'cuisine': cuisine,
            'destination': destination,
            'parameters': {
                'coordinates': coordinates,
                'search_radius_km': search_radius_km
            },
            'shown_restaurants': shown_restaurants,
            'timestamp': time.time()
        }

        # Also update active context if this is a location search with coordinates
        if coordinates and session.get('active_context'):
            session['active_context']['last_coordinates'] = coordinates

        logger.info(f"📝 Updated last search context for user {user_id}: {search_type}, {destination}, {len(shown_restaurants)} restaurants")

    def get_last_search_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get last search context for follow-up requests"""
        session = self.user_sessions.get(user_id)
        if not session:
            return None
        return session.get('last_search')

    # ============================================================================
    # MESSAGE HISTORY
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
        prefs = user_context.get('preferences', {})
        if prefs:
            if prefs.get('preferred_cuisines'):
                parts.append(f"Favorite cuisines: {', '.join(prefs['preferred_cuisines'])}")
            if prefs.get('dietary_restrictions'):
                parts.append(f"Dietary restrictions: {', '.join(prefs['dietary_restrictions'])}")
            if prefs.get('price_preference'):
                parts.append(f"Price preference: {prefs['price_preference']}")
            if prefs.get('atmosphere_preference'):
                parts.append(f"Atmosphere preference: {prefs['atmosphere_preference']}")

        # Recent restaurants
        history = user_context.get('restaurant_history', [])
        if history:
            recent = history[-5:]
            restaurant_names = [r.get('name', 'Unknown') for r in recent if isinstance(r, dict)]
            if restaurant_names:
                parts.append(f"Recent recommendations: {', '.join(restaurant_names)}")

        if parts:
            return "\n".join(parts)
        return "User has interacted before but no specific preferences recorded."

    def _format_time_ago(self, seconds: float) -> str:
        """Format time difference as human-readable string"""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days > 1 else ''} ago"

    # ============================================================================
    # RESPONSE PARSING
    # ============================================================================

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI JSON response with robust error handling"""
        try:
            # Clean the response
            cleaned = response_text.strip()

            # Remove markdown code blocks if present
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]

            cleaned = cleaned.strip()

            # Try to parse JSON
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")

            # Return safe fallback
            return {
                'action': 'chat_response',
                'context_decision': {'type': 'NEW', 'reasoning': 'Parse error'},
                'parameters': {},
                'response_text': "I'd be happy to help you find restaurants. What are you looking for?",
                'reasoning': f"JSON parse error: {str(e)}"
            }
