# utils/ai_memory_system.py
"""
AI Memory System for Restaurant Bot with Supabase Support

This module implements a comprehensive memory system that can use either:
- InMemoryStore (temporary, for testing)
- Supabase PostgreSQL (persistent, for production)

Manages three types of memory:
1. Short-term: Current conversation context (thread-scoped)
2. Long-term: User preferences and history (cross-thread)
3. Session: Active search state and temporary data
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.store.memory import InMemoryStore
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories we store"""
    SEMANTIC = "semantic"      # Facts about user preferences
    EPISODIC = "episodic"      # Past experiences/recommendations  
    PROCEDURAL = "procedural"  # Learned conversation patterns


class ConversationState(Enum):
    """Current conversation states"""
    IDLE = "idle"
    SEARCHING = "searching"
    PRESENTING_RESULTS = "presenting_results" 
    GATHERING_FEEDBACK = "gathering_feedback"
    CASUAL_CHAT = "casual_chat"


@dataclass
class UserPreferences:
    """Semantic memory: User's dining preferences"""
    preferred_cities: List[str]
    preferred_cuisines: List[str]
    dietary_restrictions: List[str]
    budget_range: str  # "budget", "mid-range", "upscale"
    preferred_ambiance: List[str]  # "casual", "romantic", "family-friendly"
    meal_times: List[str]  # "breakfast", "lunch", "dinner", "late-night"
    group_size_typical: str  # "solo", "couple", "small-group", "large-group"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        return cls(**data)


@dataclass
class RestaurantMemory:
    """Memory of a recommended restaurant"""
    restaurant_name: str
    city: str
    cuisine: str
    recommended_date: str
    user_feedback: Optional[str]  # "liked", "disliked", "visited", None
    rating_given: Optional[float]
    notes: Optional[str]
    source: str  # "database", "web_search", "google_maps"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RestaurantMemory':
        return cls(**data)


@dataclass
class ConversationPattern:
    """Procedural memory: Learned conversation patterns"""
    user_communication_style: str  # "formal", "casual", "brief", "detailed"
    preferred_response_length: str  # "short", "medium", "detailed"
    likes_follow_up_questions: bool
    prefers_immediate_results: bool
    timezone: Optional[str]
    typical_search_times: List[str]  # ["morning", "evening"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationPattern':
        return cls(**data)


class AIMemorySystem:
    """
    Comprehensive memory system with Supabase support

    Manages three types of memory:
    1. Short-term: Current conversation context (thread-scoped)
    2. Long-term: User preferences and history (cross-thread)
    3. Session: Active search state and temporary data
    """

    def __init__(self, config):
        self.config = config

        # Determine which memory store to use
        memory_store_type = getattr(config, 'MEMORY_STORE_TYPE', 'in_memory')

        if memory_store_type == 'supabase':
            # Use Supabase for persistent storage
            from utils.supabase_memory_system import create_supabase_memory_store
            self.memory_store = create_supabase_memory_store(config)
            self.is_persistent = True
            logger.info("✅ AI Memory System initialized with Supabase backend")
        elif memory_store_type == 'postgresql':
            # Use PostgreSQL (Railway) for persistent storage
            from utils.supabase_memory_system import create_supabase_memory_store
            self.memory_store = create_supabase_memory_store(config)
            self.is_persistent = True
            logger.info("✅ AI Memory System initialized with PostgreSQL backend")
        else:
            # Use InMemoryStore (default)
            from langgraph.store.memory import InMemoryStore
            self.memory_store = InMemoryStore()
            self.is_persistent = False
            logger.info("✅ AI Memory System initialized with InMemory backend")

    # =====================================================================
    # USER NAMESPACE MANAGEMENT
    # =====================================================================

    def _get_user_namespace(self, user_id: int) -> tuple[str, ...]:
        """Get namespace for user's long-term memory"""
        return (f"user_{user_id}",)

    def _get_session_namespace(self, user_id: int, thread_id: str) -> tuple[str, ...]:
        """Get namespace for session-specific memory"""
        return (f"session_{user_id}_{thread_id}",)

    # =====================================================================
    # SEMANTIC MEMORY (User Preferences)
    # =====================================================================

    async def get_user_preferences(self, user_id: int) -> UserPreferences:
        """Get user's dining preferences"""
        try:
            if self.is_persistent:
                # Supabase/PostgreSQL backend
                return await self.memory_store.get_user_preferences(user_id)
            else:
                # InMemory backend (original code)
                namespace = self._get_user_namespace(user_id)
                stored_item = await self.memory_store.aget(namespace, "preferences")

                if stored_item:
                    preferences_data = stored_item.value
                    return UserPreferences.from_dict(preferences_data)
                else:
                    return UserPreferences(
                        preferred_cities=[],
                        preferred_cuisines=[],
                        dietary_restrictions=[],
                        budget_range="mid-range",
                        preferred_ambiance=[],
                        meal_times=[],
                        group_size_typical="couple"
                    )

        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {e}")
            return UserPreferences(
                preferred_cities=[],
                preferred_cuisines=[],
                dietary_restrictions=[],
                budget_range="mid-range", 
                preferred_ambiance=[],
                meal_times=[],
                group_size_typical="couple"
            )

    async def update_user_preferences(
        self, 
        user_id: int, 
        preferences: UserPreferences
    ) -> bool:
        """Update user's dining preferences"""
        try:
            if self.is_persistent:
                # Supabase/PostgreSQL backend
                return await self.memory_store.update_user_preferences(user_id, preferences)
            else:
                # InMemory backend (original code)
                namespace = self._get_user_namespace(user_id)
                await self.memory_store.aput(
                    namespace,
                    "preferences",
                    preferences.to_dict()
                )
                logger.info(f"Updated preferences for user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {e}")
            return False

    async def learn_preferences_from_message(
        self, 
        user_id: int, 
        message: str,
        current_city: Optional[str] = None
    ) -> bool:
        """
        Extract and learn preferences from user's message
        This would typically use an LLM to extract structured information
        """
        try:
            current_prefs = await self.get_user_preferences(user_id)
            updated = False

            # Simple keyword-based learning (in production, use LLM)
            message_lower = message.lower()

            # Learn cuisine preferences
            cuisines = ["italian", "chinese", "japanese", "mexican", "indian", 
                       "french", "thai", "korean", "mediterranean", "american",
                       "ceviche", "sushi", "pizza", "ramen", "tacos"]

            for cuisine in cuisines:
                if cuisine in message_lower and cuisine not in current_prefs.preferred_cuisines:
                    current_prefs.preferred_cuisines.append(cuisine)
                    updated = True

            # Learn dietary restrictions
            restrictions = ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto"]
            for restriction in restrictions:
                if restriction in message_lower and restriction not in current_prefs.dietary_restrictions:
                    current_prefs.dietary_restrictions.append(restriction)
                    updated = True

            # Learn city preferences
            if current_city and current_city not in current_prefs.preferred_cities:
                current_prefs.preferred_cities.append(current_city)
                updated = True

            # Learn budget preferences
            if any(word in message_lower for word in ["cheap", "budget", "affordable"]):
                current_prefs.budget_range = "budget"
                updated = True
            elif any(word in message_lower for word in ["expensive", "upscale", "fine dining"]):
                current_prefs.budget_range = "upscale"
                updated = True

            if updated:
                await self.update_user_preferences(user_id, current_prefs)
                logger.info(f"Learned new preferences for user {user_id} from message")

            return updated

        except Exception as e:
            logger.error(f"Error learning preferences from message: {e}")
            return False

    # =====================================================================
    # EPISODIC MEMORY (Restaurant History)
    # =====================================================================

    async def add_restaurant_memory(
        self, 
        user_id: int, 
        restaurant_memory: RestaurantMemory
    ) -> bool:
        """Add a restaurant to user's memory"""
        try:
            if self.is_persistent:
                # Supabase/PostgreSQL backend
                return await self.memory_store.add_restaurant_memory(user_id, restaurant_memory)
            else:
                # InMemory backend (original code)
                namespace = self._get_user_namespace(user_id)
                restaurants = await self.get_restaurant_history(user_id)
                restaurants.append(restaurant_memory)

                # Keep only last 100 restaurant memories to prevent bloat
                if len(restaurants) > 100:
                    restaurants = restaurants[-100:]

                # Store updated list
                restaurant_data = {
                    "restaurants": [r.to_dict() for r in restaurants]
                }
                await self.memory_store.aput(
                    namespace,
                    "restaurant_history", 
                    restaurant_data
                )

                logger.info(f"Added restaurant memory for user {user_id}: {restaurant_memory.restaurant_name}")
                return True

        except Exception as e:
            logger.error(f"Error adding restaurant memory: {e}")
            return False

    async def get_restaurant_history(self, user_id: int, city: Optional[str] = None) -> List[RestaurantMemory]:
        """Get user's restaurant recommendation history"""
        try:
            if self.is_persistent:
                # Supabase/PostgreSQL backend
                return await self.memory_store.get_restaurant_history(user_id, city)
            else:
                # InMemory backend (original code)
                namespace = self._get_user_namespace(user_id)
                stored_item = await self.memory_store.aget(namespace, "restaurant_history")

                if stored_item:
                    restaurant_data = stored_item.value
                    restaurant_dicts = restaurant_data.get("restaurants", [])
                    all_restaurants = [RestaurantMemory.from_dict(r) for r in restaurant_dicts]

                    # Filter by city if specified
                    if city:
                        return [r for r in all_restaurants if r.city.lower() == city.lower()]
                    return all_restaurants
                else:
                    return []

        except Exception as e:
            logger.error(f"Error getting restaurant history for {user_id}: {e}")
            return []

    async def get_restaurants_for_city(self, user_id: int, city: str) -> List[RestaurantMemory]:
        """Get restaurants recommended for a specific city"""
        try:
            all_restaurants = await self.get_restaurant_history(user_id)
            return [r for r in all_restaurants if r.city.lower() == city.lower()]

        except Exception as e:
            logger.error(f"Error getting restaurants for city {city}: {e}")
            return []

    # =====================================================================
    # PROCEDURAL MEMORY (Conversation Patterns)
    # =====================================================================

    async def get_conversation_patterns(self, user_id: int) -> ConversationPattern:
        """Get user's conversation patterns"""
        try:
            if self.is_persistent:
                # Supabase/PostgreSQL backend
                return await self.memory_store.get_conversation_patterns(user_id)
            else:
                # InMemory backend (original code)
                namespace = self._get_user_namespace(user_id)
                stored_item = await self.memory_store.aget(namespace, "conversation_patterns")

                if stored_item:
                    pattern_data = stored_item.value
                    return ConversationPattern.from_dict(pattern_data)
                else:
                    return ConversationPattern(
                        user_communication_style="casual",
                        preferred_response_length="medium",
                        likes_follow_up_questions=True,
                        prefers_immediate_results=False,
                        timezone=None,
                        typical_search_times=[]
                    )

        except Exception as e:
            logger.error(f"Error getting conversation patterns for {user_id}: {e}")
            return ConversationPattern(
                user_communication_style="casual",
                preferred_response_length="medium",
                likes_follow_up_questions=True,
                prefers_immediate_results=False,
                timezone=None,
                typical_search_times=[]
            )

    async def learn_conversation_patterns(
        self, 
        user_id: int,
        message: str,
        response_time: Optional[float] = None
    ) -> bool:
        """Learn user's conversation patterns"""
        try:
            patterns = await self.get_conversation_patterns(user_id)

            # Analyze message characteristics
            # Learn communication style
            if len(message.split()) > 20:
                patterns.user_communication_style = "detailed"
            elif len(message.split()) < 5:
                patterns.user_communication_style = "brief"

            # Learn if user likes follow-up questions
            if "?" in message:
                patterns.likes_follow_up_questions = True

            # Store updated patterns
            namespace = self._get_user_namespace(user_id)
            await self.memory_store.aput(
                namespace,
                "conversation_patterns",
                patterns.to_dict()
            )

            return True

        except Exception as e:
            logger.error(f"Error learning conversation patterns: {e}")
            return False

    # =====================================================================
    # SESSION MEMORY (Temporary State)
    # =====================================================================

    async def set_session_state(
        self, 
        user_id: int, 
        thread_id: str, 
        state: ConversationState,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set current conversation state"""
        try:
            namespace = self._get_session_namespace(user_id, thread_id)

            session_data = {
                "state": state.value,
                "context": context or {},
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            await self.memory_store.aput(namespace, "session", session_data)
            return True

        except Exception as e:
            logger.error(f"Error setting session state: {e}")
            return False

    async def get_session_state(
        self, 
        user_id: int, 
        thread_id: str
    ) -> Tuple[ConversationState, Dict[str, Any]]:
        """Get current conversation state"""
        try:
            namespace = self._get_session_namespace(user_id, thread_id)

            stored_item = await self.memory_store.aget(namespace, "session")

            if stored_item:
                session_data = stored_item.value
                state = ConversationState(session_data.get("state", "idle"))
                context = session_data.get("context", {})
                return state, context
            else:
                return ConversationState.IDLE, {}

        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            return ConversationState.IDLE, {}

    async def set_current_city(
        self, 
        user_id: int, 
        thread_id: str, 
        city: str
    ) -> bool:
        """Set current search city for this session"""
        try:
            state, context = await self.get_session_state(user_id, thread_id)
            context["current_city"] = city
            return await self.set_session_state(user_id, thread_id, state, context)

        except Exception as e:
            logger.error(f"Error setting current city: {e}")
            return False

    async def get_current_city(
        self, 
        user_id: int, 
        thread_id: str
    ) -> Optional[str]:
        """Get current search city for this session"""
        try:
            _, context = await self.get_session_state(user_id, thread_id)
            return context.get("current_city")

        except Exception as e:
            logger.error(f"Error getting current city: {e}")
            return None

    # =====================================================================
    # MEMORY RETRIEVAL AND CONTEXT
    # =====================================================================

    async def get_user_context(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get comprehensive user context for AI decision making"""
        try:
            # Get all types of memory
            preferences = await self.get_user_preferences(user_id)
            restaurant_history = await self.get_restaurant_history(user_id)
            session_state, session_context = await self.get_session_state(user_id, thread_id)
            conversation_patterns = await self.get_conversation_patterns(user_id)
            current_city = await self.get_current_city(user_id, thread_id)

            # Build comprehensive context
            context = {
                "user_id": user_id,
                "thread_id": thread_id,
                "current_city": current_city,
                "session_state": session_state.value,
                "session_context": session_context,
                "preferences": preferences.to_dict(),
                "restaurant_history": [r.to_dict() for r in restaurant_history],
                "conversation_patterns": patterns.to_dict()
            }

            return context

        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}

    # =====================================================================
    # FILTERING AND RECOMMENDATIONS
    # =====================================================================

    async def filter_already_recommended(
        self, 
        user_id: int,
        restaurants: List[Dict[str, Any]],
        city: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter out restaurants already recommended to user"""
        try:
            # Get user's restaurant history
            history = await self.get_restaurant_history(user_id, city)

            if not history:
                return restaurants

            # Create set of restaurant names user has seen
            seen_restaurants = {r.restaurant_name.lower() for r in history}

            # Filter out already recommended restaurants
            filtered_restaurants = [
                r for r in restaurants 
                if r.get('name', '').lower() not in seen_restaurants
            ]

            logger.info(f"Filtered {len(restaurants) - len(filtered_restaurants)} already recommended restaurants")
            return filtered_restaurants

        except Exception as e:
            logger.error(f"Error filtering already recommended restaurants: {e}")
            return restaurants  # Return original list if filtering fails

    async def cleanup_old_sessions(self, days: int = 30) -> bool:
        """Clean up old session data (for maintenance)"""
        try:
            if self.is_persistent:
                # Call cleanup on persistent store
                return await self.memory_store.cleanup_expired_sessions()
            else:
                # For InMemoryStore, no cleanup needed
                logger.info(f"Cleanup initiated for sessions older than {days} days")
                return True

        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return False

    async def get_memory_stats(self, user_id: int) -> Dict[str, Any]:
        """Get memory statistics for debugging"""
        try:
            preferences = await self.get_user_preferences(user_id)
            restaurant_history = await self.get_restaurant_history(user_id)
            conversation_patterns = await self.get_conversation_patterns(user_id)

            return {
                "user_id": user_id,
                "restaurant_count": len(restaurant_history),
                "cities_visited": len(preferences.preferred_cities),
                "cuisines_tried": len(preferences.preferred_cuisines),
                "dietary_restrictions": len(preferences.dietary_restrictions),
                "communication_style": conversation_patterns.user_communication_style,
                "memory_created": True,
                "backend": "supabase" if self.is_persistent else "in_memory"
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"user_id": user_id, "error": str(e)}


# =====================================================================
# FACTORY FUNCTION
# =====================================================================

def create_ai_memory_system(config) -> AIMemorySystem:
    """Factory function to create AI memory system"""
    return AIMemorySystem(config)