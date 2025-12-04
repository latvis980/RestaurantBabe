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
- PENDING GPS STATE MANAGEMENT (new) - AI controls when to show/clear location button
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
    - MANAGE PENDING GPS STATE - single source of truth
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

        logger.info("✅ AI Chat Layer initialized with memory support and pending GPS management")

    def _build_prompts(self):
        """Build AI prompts with pending GPS awareness"""

        from langchain_core.prompts import ChatPromptTemplate

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the conversation manager for a restaurant recommendation bot.

    ## YOUR ROLE
    Analyze user messages, accumulate search parameters across turns, and decide when to trigger a search.
    You are the SINGLE SOURCE OF TRUTH for conversation state - you decide everything.

    ## REQUIRED PARAMETERS FOR SEARCH
    A search can only be triggered when you have BOTH:
    1. **CUISINE/TYPE**: What kind of food (e.g., "sushi", "Italian", "natural wine", "brunch")
    2. **LOCATION**: One of these:
       - GPS coordinates (when has_gps=Yes)
       - City name (e.g., "Tokyo", "Paris")  
       - Neighborhood/landmark that can be geocoded (e.g., "SoHo NYC", "near Eiffel Tower")

    ## CRITICAL: DESTINATION EXTRACTION RULES

    **RULE 1: CITY is the PRIMARY identifier, neighborhoods are QUALIFIERS**
    When extracting destination, the CITY is always the anchor. Neighborhoods/districts are added as qualifiers.

    **RULE 2: Format destinations as "Neighborhood, City" for geocoding**
    Many neighborhood names exist in multiple cities (Ari in Bangkok AND Italy, SoHo in NYC AND London).
    The geocoder needs city context to find the right place.

    **DESTINATION FORMAT EXAMPLES:**
    - "restaurants in Ari neighbourhood in Bangkok" → destination: "Ari, Bangkok"
    - "coffee in SoHo, New York" → destination: "SoHo, New York"
    - "bars in Chinatown, San Francisco" → destination: "Chinatown, San Francisco"
    - "pizza in Alfama, Lisbon" → destination: "Alfama, Lisbon"
    - "brunch in Shibuya, Tokyo" → destination: "Shibuya, Tokyo"
    - "wine bars in Le Marais, Paris" → destination: "Le Marais, Paris"

    **WRONG vs CORRECT:**
    - ❌ "specialty coffee in Ari neighbourhood in Bangkok" → destination: "Ari" 
    - ✅ "specialty coffee in Ari neighbourhood in Bangkok" → destination: "Ari, Bangkok"
    - ❌ "restaurants in Chinatown, NYC" → destination: "Chinatown"
    - ✅ "restaurants in Chinatown, NYC" → destination: "Chinatown, New York"

    **RULE 3: City-only queries stay as city**
    - "best sushi in Tokyo" → destination: "Tokyo" (no neighborhood mentioned)
    - "restaurants in Bangkok" → destination: "Bangkok" (no neighborhood mentioned)

    ## PENDING GPS STATE (CRITICAL!)
    
    When pending_gps=Yes, the user was previously asked for their location (location button is shown).
    The pending_gps_cuisine tells you what they were originally looking for.
    
    **RULES FOR PENDING GPS:**
    
    1. **User provides LOCATION ANSWER** (neighborhood, street, area name):
       Examples: "Chinatown", "near Times Square", "Alfama", "downtown", "Lapa"
       → Use pending_gps_cuisine + this location
       → trigger_search with search_mode="coordinates_search"
       → Set clear_pending_gps=true
    
    2. **User provides NEW COMPLETE QUERY** (has BOTH cuisine AND city):
       Examples: "find bakeries in Bangkok", "sushi in Tokyo", "pizza in Rome"
       → IGNORE pending GPS state, this is a fresh search
       → trigger_search with the new query
       → Set clear_pending_gps=true
    
    3. **User provides GPS coordinates** (has_gps=Yes):
       → Use pending_gps_cuisine + GPS
       → trigger_search with search_mode="gps_search"
       → Set clear_pending_gps=true
    
    4. **User says something off-topic or wants to cancel**:
       Examples: "never mind", "cancel", "what's the weather", "hello"
       → Clear pending state, respond appropriately
       → Set clear_pending_gps=true
    
    5. **User provides ONLY cuisine** (no location):
       Examples: "actually, I want pizza", "sushi instead"
       → Update pending_gps_cuisine to new cuisine
       → Keep asking for location (pending_gps stays true)
       → Set clear_pending_gps=false

    ## ACTIONS (choose exactly one)

    **request_gps**: User wants nearby search but no GPS yet
    - Use when: "near me", "nearby", "close by", "around here" AND has_gps=No AND pending_gps=No
    - NEVER use when: has_gps=Yes (you already have location!)
    - NEVER use when: pending_gps=Yes (already waiting for location!)

    **collect_info**: Need more information before searching
    - Use when: Missing cuisine OR missing location (and not a "near me" request)
    - Ask naturally, don't interrogate

    **trigger_search**: Ready to search - have both cuisine AND location
    - Use when: You have cuisine AND (has_gps=Yes OR city_name OR geocodable_location)
    - Set is_complete=true
    - Set appropriate search_mode:
      - "gps_search" if using GPS coordinates
      - "city_search" if searching entire city by name (e.g., "Tokyo", "Paris")
      - "coordinates_search" if neighborhood/area needs geocoding (e.g., "Alfama", "SoHo")
      - "google_maps_more" if user wants MORE options after a LOCATION-based search (uses Google Maps)
      - "follow_up_more_results" if user wants more options after a CITY-based search
    
    **IMPORTANT - "MORE" REQUESTS:**
    - Check last_search_type to decide which mode to use!
    - If last_search_type="location_search" → use "google_maps_more"
    - If last_search_type="city_search" → use "follow_up_more_results"

    **chat_response**: Answer questions, greet, or redirect
    - Use for: greetings, questions about shown results, off-topic queries

    ## RESPONSE FORMAT (JSON only, no markdown)

    {{
        "action": "request_gps" | "collect_info" | "trigger_search" | "chat_response",
        "response_text": "Your message to the user",
        "state_update": {{
            "cuisine": "extracted cuisine type or null",
            "destination": "city/neighborhood/landmark or null", 
            "search_mode": "gps_search|city_search|coordinates_search|follow_up_more_results|null",
            "needs_gps": true | false,
            "is_complete": true | false,
            "requirements": ["outdoor", "romantic", etc.],
            "raw_query": "original user message",
            "clear_pending_gps": true | false
        }},
        "reasoning": "Brief explanation of your decision"
    }}

    ## EXAMPLES

    **Example 1: User provides location when GPS was pending**
    Pending GPS: Yes, Cuisine: "coffee"
    User message: "I'm in Chinatown"
    → {{
        "action": "trigger_search",
        "response_text": "Finding great coffee spots in Chinatown! ☕",
        "state_update": {{"cuisine": "coffee", "destination": "Chinatown", "search_mode": "coordinates_search", "is_complete": true, "clear_pending_gps": true}},
        "reasoning": "User answered location question - combine with pending cuisine"
    }}

    **Example 2: User starts NEW query while GPS was pending**
    Pending GPS: Yes, Cuisine: "coffee"
    User message: "find best bakeries in Bangkok"
    → {{
        "action": "trigger_search",
        "response_text": "Finding the best bakeries in Bangkok! 🥐",
        "state_update": {{"cuisine": "bakeries", "destination": "Bangkok", "search_mode": "city_search", "is_complete": true, "clear_pending_gps": true}},
        "reasoning": "Complete NEW query - ignore pending GPS, start fresh city search"
    }}

    **Example 3: User shares GPS after being asked**
    Pending GPS: Yes, Cuisine: "pizza", has_gps: Yes
    User message: "[Location shared]"
    → {{
        "action": "trigger_search",
        "response_text": "Perfect! Finding pizza near you! 🍕",
        "state_update": {{"cuisine": "pizza", "search_mode": "gps_search", "is_complete": true, "clear_pending_gps": true}},
        "reasoning": "GPS received - search with pending cuisine"
    }}

    **Example 4: User wants nearby food (no pending GPS yet)**
    Pending GPS: No, has_gps: No
    User message: "Find good pizza near me"
    → {{
        "action": "request_gps",
        "response_text": "I'd love to find pizza nearby! 📍 Please share your location or tell me your neighborhood.",
        "state_update": {{"cuisine": "pizza", "needs_gps": true, "clear_pending_gps": false}},
        "reasoning": "Need location for 'near me' - request GPS"
    }}

    **Example 5: Complete city query (no GPS needed)**
    Pending GPS: No, has_gps: No
    User message: "Best ramen in Tokyo"
    → {{
        "action": "trigger_search",
        "response_text": "Finding the best ramen in Tokyo! 🍜",
        "state_update": {{"cuisine": "ramen", "destination": "Tokyo", "search_mode": "city_search", "is_complete": true, "clear_pending_gps": false}},
        "reasoning": "Complete query - cuisine + city"
    }}

    **Example 6: User cancels/changes mind**
    Pending GPS: Yes, Cuisine: "coffee"
    User message: "never mind" or "cancel"
    → {{
        "action": "chat_response",
        "response_text": "No problem! What else can I help you find?",
        "state_update": {{"clear_pending_gps": true}},
        "reasoning": "User cancelled - clear pending state"
    }}

    **Example 7: User changes cuisine while GPS pending**
    Pending GPS: Yes, Cuisine: "coffee"
    User message: "actually pizza"
    → {{
        "action": "chat_response",
        "response_text": "Pizza sounds great! 🍕 Where are you located?",
        "state_update": {{"cuisine": "pizza", "needs_gps": true, "clear_pending_gps": false}},
        "reasoning": "Updated cuisine, still need location"
    }}

    **Example 8: User wants MORE after location search**
    Last Search Type: location_search, Last Cuisine: "Thai", Last Destination: "Mandarin Oriental"
    User message: "show me more options"
    → {{
        "action": "trigger_search",
        "response_text": "Let me search Google Maps for more Thai restaurants near Mandarin Oriental! 🗺️",
        "state_update": {{"cuisine": "Thai", "destination": "Mandarin Oriental", "search_mode": "google_maps_more", "is_complete": true, "clear_pending_gps": false}},
        "reasoning": "More requested after location search - MUST use google_maps_more to search Google Maps directly (database already exhausted)"
    }}

    **Example: Neighborhood + City query**
    User message: "I'm looking for specialty coffee places in Ari neighbourhood in Bangkok"
    → {{
        "action": "trigger_search",
        "response_text": "Finding specialty coffee spots in Ari, Bangkok! ☕",
        "state_update": {{"cuisine": "specialty coffee", "destination": "Ari, Bangkok", "search_mode": "coordinates_search", "is_complete": true, "clear_pending_gps": true}},
        "reasoning": "Complete query with neighborhood + city - destination formatted as 'Neighborhood, City' for accurate geocoding"
    }}

    **Example 9: User wants MORE after city search**
    Last Search Type: city_search, Last Cuisine: "ramen", Last Destination: "Tokyo"
    User message: "any other options?"
    → {{
        "action": "trigger_search",
        "response_text": "Let me find more ramen spots in Tokyo! 🍜",
        "state_update": {{"cuisine": "ramen", "destination": "Tokyo", "search_mode": "follow_up_more_results", "is_complete": true, "clear_pending_gps": false}},
        "reasoning": "More requested after city search - use follow-up mode"
    }}

    **CRITICAL: FOLLOW-UP REQUESTS WITH USER MODIFICATIONS**

    When user asks for "more" but includes modifications (different cuisine, closer, specific requirements),
    you MUST generate supervisor_instructions to guide downstream agents.

    **Example 10: User wants more but with modifications**
    Last Search Type: location_search, Last Cuisine: "brunch", Last Destination: "near me"
    Already Shown Restaurants: Cafe Luna, The Brunch House, Morning Glory
    User message: "show me more, but lunch not brunch and somewhere closer"
    → {{
        "action": "trigger_search",
        "response_text": "Looking for lunch spots closer to you! 🍽️",
        "state_update": {{
            "cuisine": "lunch", 
            "destination": "near me", 
            "search_mode": "google_maps_more", 
            "is_complete": true, 
            "clear_pending_gps": false,
            "modified_query": "lunch restaurants",
            "supervisor_instructions": "User previously searched for BRUNCH but now specifically wants LUNCH (not brunch). They also want places CLOSER to their location - previous results may have been too far. EXCLUDE already shown: Cafe Luna, The Brunch House, Morning Glory. Prioritize proximity and lunch-appropriate venues.",
            "exclude_restaurants": ["Cafe Luna", "The Brunch House", "Morning Glory"],
            "is_follow_up": true
        }},
        "reasoning": "Follow-up with modifications - generate supervisor_instructions for downstream filtering"
    }}

    **Example 11: Simple "more" without modifications**
    Last Search Type: location_search, Last Cuisine: "Italian", Last Destination: "SoHo"
    Already Shown Restaurants: Emilio's, Pasta Palace, Trattoria Roma
    User message: "any other options?"
    → {{
        "action": "trigger_search",
        "response_text": "Finding more Italian spots in SoHo! 🍝",
        "state_update": {{
            "cuisine": "Italian", 
            "destination": "SoHo", 
            "search_mode": "google_maps_more", 
            "is_complete": true, 
            "clear_pending_gps": false,
            "supervisor_instructions": "User wants MORE Italian restaurants in SoHo. EXCLUDE already shown: Emilio's, Pasta Palace, Trattoria Roma. Same criteria as before, just different results.",
            "exclude_restaurants": ["Emilio's", "Pasta Palace", "Trattoria Roma"],
            "is_follow_up": true
        }},
        "reasoning": "Simple follow-up - exclude previously shown restaurants"
    }}

    **Example 12: User wants more with specific new requirement**
    Last Search Type: location_search, Last Cuisine: "restaurants", Last Destination: "Williamsburg"
    Already Shown Restaurants: The Commodore, Lilia, Maison Premiere
    User message: "more options but with outdoor seating"
    → {{
        "action": "trigger_search",
        "response_text": "Looking for places with outdoor seating in Williamsburg! 🌞",
        "state_update": {{
            "cuisine": "restaurants", 
            "destination": "Williamsburg", 
            "search_mode": "google_maps_more", 
            "is_complete": true,
            "requirements": ["outdoor seating"],
            "supervisor_instructions": "User wants MORE restaurants but specifically with OUTDOOR SEATING. Previous results didn't emphasize this. EXCLUDE: The Commodore, Lilia, Maison Premiere. Prioritize venues that mention patios, terraces, gardens, or outdoor dining.",
            "exclude_restaurants": ["The Commodore", "Lilia", "Maison Premiere"],
            "is_follow_up": true
        }},
        "reasoning": "Follow-up with new requirement - add to supervisor_instructions"
    }}

    ## SUPERVISOR_INSTRUCTIONS RULES

    When generating supervisor_instructions for follow-up requests:
    1. **Always mention what changed** - what the user wants differently from before
    2. **Always list exclusions** - restaurants already shown that should be filtered out  
    3. **Include the original context** - so downstream agents understand the full picture
    4. **Be specific about priorities** - closer, cheaper, outdoor, etc.
    5. **Use natural language** - downstream AI will interpret this, not code

    For INITIAL searches (not follow-ups), supervisor_instructions should be null or omitted.
    
    """),
            ("human", """## CURRENT STATE

    **Pending GPS Request:**
    - GPS Button Shown: {pending_gps}
    - Pending Cuisine: {pending_gps_cuisine}

    **Last Search Context (for "more" requests):**
    - Last Search Type: {last_search_type}
    - Last Cuisine: {last_search_cuisine}
    - Last Destination: {last_search_destination}
    - Already Shown Restaurants: {last_shown_restaurants}

    **Memory Context (user's history):**
    {memory_context}

    **Accumulated Parameters:**
    - Cuisine: {current_cuisine}
    - Destination: {current_destination}  
    - GPS Available: {has_gps}
    - Stored Location Context: {stored_location}

    **Optional parameters:**
    - Additional requirements: {requirements}

    **Recent Conversation:**
    {conversation_history}

    **Current Message:**
    {user_message}

    Respond with valid JSON only, no markdown code blocks.""")
        ])

        self.conversation_chain = self.conversation_prompt | self.llm

    # ============================================================================
    # PENDING GPS STATE MANAGEMENT
    # ============================================================================

    def set_pending_gps(self, user_id: int, cuisine: str) -> None:
        """
        Set pending GPS state when location button is shown.
        Called by Telegram bot when showing location button.
        """
        session = self._get_or_create_session(user_id, f"pending_{user_id}")
        session['pending_gps'] = True
        session['pending_gps_cuisine'] = cuisine
        session['pending_gps_timestamp'] = time.time()
        logger.info(f"📍 Set pending GPS for user {user_id}: cuisine='{cuisine}'")

    def clear_pending_gps(self, user_id: int) -> None:
        """
        Clear pending GPS state.
        Called when user provides location, completes search, or cancels.
        """
        session = self.user_sessions.get(user_id)
        if session:
            was_pending = session.get('pending_gps', False)
            session['pending_gps'] = False
            session['pending_gps_cuisine'] = None
            session['pending_gps_timestamp'] = None
            if was_pending:
                logger.info(f"🗑️ Cleared pending GPS for user {user_id}")

    def get_pending_gps_state(self, user_id: int) -> Dict[str, Any]:
        """
        Get current pending GPS state for a user.
        Returns dict with pending_gps, pending_gps_cuisine, age_minutes
        """
        session = self.user_sessions.get(user_id)
        if not session:
            return {'pending_gps': False, 'pending_gps_cuisine': None, 'age_minutes': None}

        pending = session.get('pending_gps', False)
        cuisine = session.get('pending_gps_cuisine')
        timestamp = session.get('pending_gps_timestamp')

        # Check expiry (15 minutes)
        if pending and timestamp:
            age_minutes = (time.time() - timestamp) / 60
            if age_minutes > 15:
                logger.info(f"⏰ Pending GPS expired for user {user_id} ({age_minutes:.1f} min)")
                self.clear_pending_gps(user_id)
                return {'pending_gps': False, 'pending_gps_cuisine': None, 'age_minutes': None}
            return {'pending_gps': True, 'pending_gps_cuisine': cuisine, 'age_minutes': age_minutes}

        return {'pending_gps': pending, 'pending_gps_cuisine': cuisine, 'age_minutes': None}

    def is_pending_gps(self, user_id: int) -> bool:
        """Quick check if GPS is pending for user"""
        return self.get_pending_gps_state(user_id)['pending_gps']

    # ============================================================================
    # SHOWN RESTAURANTS TRACKING (for duplicate detection)
    # ============================================================================

    def store_shown_restaurants(self, user_id: int, restaurants: List[Dict[str, Any]], search_type: str, cuisine: str = None, destination: str = None) -> None:
        """
        Store the restaurants that were just shown to the user.
        Used to detect duplicates on "more" requests.
        
        Args:
            user_id: User ID
            restaurants: List of restaurant dicts (must have 'name' key)
            search_type: 'city_search' or 'location_search'
            cuisine: The cuisine that was searched
            destination: The destination that was searched
        """
        session = self._get_or_create_session(user_id, f"track_{user_id}")
        
        # Extract restaurant names for comparison
        restaurant_names = []
        for r in restaurants:
            name = r.get('name') or r.get('restaurant_name') or r.get('title', '')
            if name:
                restaurant_names.append(name.lower().strip())
        
        session['last_shown_restaurants'] = restaurant_names
        session['last_search_type'] = search_type
        session['last_search_cuisine'] = cuisine
        session['last_search_destination'] = destination
        session['last_search_time'] = time.time()
        
        logger.info(f"📋 Stored {len(restaurant_names)} shown restaurants for user {user_id} (type={search_type})")

    def check_for_duplicates(self, user_id: int, new_restaurants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if new results are duplicates of what was already shown.
        
        Returns:
            {
                'has_duplicates': bool,
                'duplicate_count': int,
                'new_count': int,
                'duplicate_percentage': float,
                'all_duplicates': bool,  # True if ALL results are duplicates
                'new_restaurants': list  # Only the non-duplicate restaurants
            }
        """
        session = self.user_sessions.get(user_id)
        if not session:
            return {'has_duplicates': False, 'duplicate_count': 0, 'new_count': len(new_restaurants), 
                    'duplicate_percentage': 0.0, 'all_duplicates': False, 'new_restaurants': new_restaurants}
        
        shown = set(session.get('last_shown_restaurants', []))
        if not shown:
            return {'has_duplicates': False, 'duplicate_count': 0, 'new_count': len(new_restaurants),
                    'duplicate_percentage': 0.0, 'all_duplicates': False, 'new_restaurants': new_restaurants}
        
        duplicates = 0
        new_restaurants_filtered = []
        
        for r in new_restaurants:
            name = r.get('name') or r.get('restaurant_name') or r.get('title', '')
            if name and name.lower().strip() in shown:
                duplicates += 1
            else:
                new_restaurants_filtered.append(r)
        
        total = len(new_restaurants)
        duplicate_pct = (duplicates / total * 100) if total > 0 else 0
        all_dupes = duplicates == total and total > 0
        
        result = {
            'has_duplicates': duplicates > 0,
            'duplicate_count': duplicates,
            'new_count': len(new_restaurants_filtered),
            'duplicate_percentage': duplicate_pct,
            'all_duplicates': all_dupes,
            'new_restaurants': new_restaurants_filtered
        }
        
        if duplicates > 0:
            logger.info(f"🔄 Duplicate check for user {user_id}: {duplicates}/{total} duplicates ({duplicate_pct:.0f}%)")
        
        return result

    def get_last_search_context(self, user_id: int) -> Dict[str, Any]:
        """Get the context from the last search for follow-up requests"""
        session = self.user_sessions.get(user_id)
        if not session:
            return {}
        
        return {
            'search_type': session.get('last_search_type'),
            'cuisine': session.get('last_search_cuisine'),
            'destination': session.get('last_search_destination'),
            'shown_restaurants': session.get('last_shown_restaurants', []),
            'search_time': session.get('last_search_time')
        }

    def update_last_search_context(
        self,
        user_id: int,
        search_type: str,  # 'location_search' or 'city_search'
        cuisine: Optional[str] = None,
        destination: Optional[str] = None,
        restaurants: Optional[List[Dict[str, Any]]] = None,
        coordinates: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Update session with last search context for follow-up "more" requests.

        Called by LangGraphSupervisor after successful search completion.
        """
        session = self.user_sessions.get(user_id)
        if not session:
            session = self._get_or_create_session(user_id, f"search_{user_id}")

        # Update search tracking fields
        session['last_search_type'] = search_type
        session['last_search_time'] = time.time()

        if cuisine:
            session['last_search_cuisine'] = cuisine

        if destination:
            session['last_search_destination'] = destination

        if coordinates:
            session['gps_coordinates'] = coordinates
            session['gps_timestamp'] = time.time()

        # Store shown restaurants for duplicate detection
        if restaurants:
            restaurant_names = []
            for r in restaurants[:20]:  # Keep last 20
                name = r.get('name') or r.get('restaurant_name') or r.get('title', '')
                if name:
                    restaurant_names.append(name.lower().strip())
            session['last_shown_restaurants'] = restaurant_names

        logger.info(f"✅ Updated last search context for user {user_id}: "
                    f"type={search_type}, cuisine={cuisine}, destination={destination}")

    # ============================================================================
    # MEMORY CONTEXT FORMATTING
    # ============================================================================

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

    # ============================================================================
    # MESSAGE HANDLING
    # ============================================================================

    def add_message(self, user_id: int, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Primary: In-memory session (fast)
        Backup: Supabase (survives restarts)
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
        user_context: Optional[Dict[str, Any]] = None
    ) -> HandoffMessage:
        """
        Process user message with full context awareness including pending GPS state.

        This is the SINGLE DECISION POINT for all user messages.
        """
        try:
            # Get or create session
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            session = self._get_or_create_session(user_id, thread_id)
            await self._ensure_history_loaded(user_id)

            # Handle GPS coordinates - with 30-minute expiry
            current_gps = gps_coordinates
            if gps_coordinates:
                session['gps_coordinates'] = gps_coordinates
                session['gps_timestamp'] = time.time()
                logger.info(f"📍 Received GPS coordinates: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}")
            elif session.get('gps_coordinates'):
                gps_age = time.time() - session.get('gps_timestamp', 0)
                if gps_age < 1800:  # 30 minutes
                    current_gps = session['gps_coordinates']
                    logger.info(f"📍 Using stored GPS: {current_gps[0]:.4f}, {current_gps[1]:.4f} ({gps_age/60:.0f} min old)")
                else:
                    logger.info(f"⏰ Stored GPS expired ({gps_age/60:.0f} min old), clearing")
                    del session['gps_coordinates']
                    if 'gps_timestamp' in session:
                        del session['gps_timestamp']

            # Get pending GPS state
            pending_state = self.get_pending_gps_state(user_id)
            pending_gps = pending_state['pending_gps']
            pending_gps_cuisine = pending_state['pending_gps_cuisine']

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

            # Add user message to history
            self.add_message(user_id, 'user', user_message)

            # Get accumulated state
            accumulated_state = session.get('accumulated_state', {})

            # Format memory context for AI
            memory_context_text = self._format_memory_context(user_context)

            # Get current cuisine and destination
            current_cuisine = accumulated_state.get('cuisine') or session.get('current_cuisine') or 'None'
            current_destination = accumulated_state.get('destination') or session.get('current_destination') or 'None'

            # Get requirements
            current_requirements = accumulated_state.get('requirements', [])
            if not current_requirements and user_context:
                prefs = user_context.get("preferences")
                if prefs:
                    if hasattr(prefs, 'dietary_restrictions') and prefs.dietary_restrictions:
                        current_requirements.extend(prefs.dietary_restrictions)
                    if hasattr(prefs, 'budget_range') and prefs.budget_range:
                        current_requirements.append(f"budget: {prefs.budget_range}")

            requirements_text = ', '.join(current_requirements) if current_requirements else 'None'

            # Prepare prompt variables
            prompt_vars = {
                'pending_gps': 'Yes' if pending_gps else 'No',
                'pending_gps_cuisine': pending_gps_cuisine or 'None',
                'last_search_type': session.get('last_search_type') or 'None',
                'last_search_cuisine': session.get('last_search_cuisine') or 'None',
                'last_search_destination': session.get('last_search_destination') or 'None',
                'memory_context': memory_context_text,
                'conversation_history': self._format_conversation_context(session),
                'current_cuisine': current_cuisine,
                'current_destination': current_destination,
                'stored_location': stored_location_text,
                'has_gps': 'Yes' if current_gps else 'No',
                'user_message': user_message,
                'requirements': requirements_text, 
                'last_shown_restaurants': ', '.join(session.get('last_shown_restaurants', [])) or 'None'
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

            # Handle clear_pending_gps signal
            clear_pending = state_update.get('clear_pending_gps', False)
            if clear_pending:
                self.clear_pending_gps(user_id)

            # Update session state
            if state_update:
                # Filter out control signals from accumulated state
                state_for_accumulation = {k: v for k, v in state_update.items() 
                                         if v is not None and k not in ['clear_pending_gps', 'needs_gps', 'is_complete']}
                accumulated_state.update(state_for_accumulation)
                session['accumulated_state'] = accumulated_state

            # Extract key fields
            search_mode = state_update.get('search_mode', '')
            destination = state_update.get('destination', '')
            cuisine = state_update.get('cuisine', '')
            requirements = state_update.get('requirements', [])
            preferences = state_update.get('preferences', {})
            is_complete = state_update.get('is_complete', False)
            needs_gps = state_update.get('needs_gps', False)

            # Store cuisine for follow-up
            if cuisine:
                session['current_cuisine'] = cuisine

            # Add assistant response to history
            session['conversation_history'].append({
                'role': 'assistant',
                'message': response_text,
                'timestamp': time.time()
            })

            logger.info(f"🤖 AI decision: action={action}, mode={search_mode}, complete={is_complete}, clear_pending={clear_pending}")

            # ================================================================
            # RETURN APPROPRIATE HANDOFF
            # ================================================================

            # Handle GPS request - set pending state
            if action == 'request_gps':
                # Set pending GPS state with cuisine
                extracted_cuisine = cuisine or pending_gps_cuisine or current_cuisine
                if extracted_cuisine and extracted_cuisine != 'None':
                    self.set_pending_gps(user_id, extracted_cuisine)
                
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=response_text or "I need your location to find nearby restaurants.",
                    reasoning=reasoning,
                    needs_gps=True
                )

            # Handle clarification requests
            if action in ['chat_response', 'collect_info']:
                return HandoffMessage(
                    command=HandoffCommand.CONTINUE_CONVERSATION,
                    conversation_response=response_text or "How can I help?",
                    reasoning=reasoning,
                    needs_gps=needs_gps
                )

            # Handle search trigger
            if action == 'trigger_search' and is_complete:
                # Check if follow-up request for city search
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

                # Handle Google Maps "more" for location searches
                if search_mode == 'google_maps_more':
                    # Use last search context
                    last_cuisine = session.get('last_search_cuisine') or cuisine
                    last_destination = session.get('last_search_destination') or destination
                    
                    # Get stored GPS or location context
                    stored_gps = current_gps or session.get('gps_coordinates')
                    if not stored_gps:
                        # Try to get from location context
                        loc_ctx = self.get_location_context(user_id)
                        if loc_ctx and loc_ctx.get('coordinates'):
                            stored_gps = loc_ctx['coordinates']
                    
                    if not stored_gps and last_destination:
                        # Geocode the last destination
                        try:
                            stored_gps = geocode_location(last_destination)
                        except Exception as e:
                            logger.warning(f"Could not geocode {last_destination}: {e}")
                    
                    logger.info(f"🗺️ Google Maps MORE: cuisine={last_cuisine}, dest={last_destination}, gps={stored_gps is not None}")
                    
                    return create_search_handoff(
                        destination=last_destination or "nearby",
                        cuisine=last_cuisine,
                        search_type=SearchType.LOCATION_MAPS_SEARCH,  # Use maps-only type
                        user_query=f"more {last_cuisine} near {last_destination}",
                        user_id=user_id,
                        thread_id=thread_id,
                        gps_coordinates=stored_gps,
                        requirements=requirements,
                        preferences={},  # Type is explicit now
                        clear_previous=False,
                        is_new_destination=False,
                        reasoning=f"Maps-only search for more options: {reasoning}"
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
                if search_mode in ['gps_search', 'coordinates_search']:
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
        """Get or create user session with pending GPS fields"""
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
                'gps_coordinates': None,
                'gps_timestamp': None,
                'last_search_time': None,
                'last_search_thread_id': None,
                'current_location': None,
                'history_loaded': False,
                # Pending GPS state
                'pending_gps': False,
                'pending_gps_cuisine': None,
                'pending_gps_timestamp': None,
                # Last search tracking (for "more" requests and duplicate detection)
                'last_search_type': None,  # 'city_search' or 'location_search'
                'last_search_cuisine': None,
                'last_search_destination': None,
                'last_shown_restaurants': [],  # List of restaurant names shown
            }

            # Load persistent conversation history from Supabase
            if self.memory_store:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._load_persistent_history(user_id))
                    else:
                        loop.run_until_complete(self._load_persistent_history(user_id))
                except RuntimeError:
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
        """Load history from Supabase if not already loaded"""
        session = self.user_sessions.get(user_id)
        if not session or session.get('history_loaded'):
            return

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
        """Save message to Supabase in background"""
        if not self.memory_store:
            return

        import asyncio

        memory_store = self.memory_store

        async def _save():
            try:
                await memory_store.add_conversation_message(user_id, role, message)
                logger.debug(f"💾 Backed up {role} message to Supabase for user {user_id}")
            except Exception as e:
                logger.warning(f"Background save failed: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_save())
            else:
                loop.run_until_complete(_save())
        except RuntimeError:
            try:
                asyncio.run(_save())
            except Exception as e:
                logger.debug(f"Could not schedule background save: {e}")

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for AI"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-10:]:
            role = msg.get('role', 'user').upper()
            content = msg.get('message', '')

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
        """Store location context with automatic 30-minute expiry"""
        session = self.user_sessions.get(user_id)
        if not session:
            session = {
                'user_id': user_id,
                'created_at': time.time()
            }
            self.user_sessions[user_id] = session

        session['current_location'] = {
            'location': location,
            'coordinates': coordinates,
            'search_type': search_type,
            'stored_at': time.time()
        }

        self.location_contexts[user_id] = session['current_location']

        coord_str = f" ({coordinates[0]:.4f}, {coordinates[1]:.4f})" if coordinates else ""
        logger.info(f"📍 Stored location for user {user_id}: {location}{coord_str} (expires in 30 min)")

    def get_location_context(self, user_id: int, max_age_minutes: int = 30) -> Optional[Dict[str, Any]]:
        """Get current location context if still valid"""
        session = self.user_sessions.get(user_id)
        if not session:
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

        stored_at = location_ctx.get('stored_at', 0)
        age_seconds = time.time() - stored_at
        max_age_seconds = max_age_minutes * 60

        if age_seconds > max_age_seconds:
            age_minutes = age_seconds / 60
            logger.info(f"⏰ Location context expired for user {user_id} ({age_minutes:.1f} min old)")
            del session['current_location']
            return None

        return location_ctx

    def clear_location_context(self, user_id: int) -> None:
        """Clear stored location context"""
        session = self.user_sessions.get(user_id)
        if session and 'current_location' in session:
            old_location = session['current_location'].get('location', 'unknown')
            del session['current_location']
            logger.info(f"🗑️ Cleared location context for user {user_id} (was: {old_location})")

        if user_id in self.location_contexts:
            del self.location_contexts[user_id]

    def get_location_age_minutes(self, user_id: int) -> Optional[float]:
        """Get age of stored location in minutes"""
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
            location_context = session.get('current_location')

            session['current_destination'] = None
            session['current_cuisine'] = None
            session['state'] = ConversationState.GREETING
            session['accumulated_state'] = {}
            session['last_search_time'] = None
            session['conversation_history'] = []
            # Also clear pending GPS
            session['pending_gps'] = False
            session['pending_gps_cuisine'] = None
            session['pending_gps_timestamp'] = None

            if location_context:
                loc_name = location_context.get('location', 'unknown')
                logger.info(f"🧹 Cleared session for user {user_id} (kept location context: {loc_name})")
            else:
                logger.info(f"🧹 Cleared session for user {user_id}")

    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current session info"""
        return self.user_sessions.get(user_id)

    def _extract_city_from_destination(self, destination: str) -> Optional[str]:
        """Extract city name from destination string"""
        if not destination:
            return None

        parts = [p.strip() for p in destination.split(',')]

        if len(parts) > 1:
            return parts[-1]

        return parts[0]
