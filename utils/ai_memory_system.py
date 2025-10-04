# utils/ai_memory_system.py
"""
AI Memory System for Restaurant Bot

This module implements a comprehensive memory system using LangGraph's memory store
to provide both short-term (conversation) and long-term (cross-session) memory
for intelligent restaurant recommendations and chat behavior.

Memory Types:
- Semantic: User preferences, dietary restrictions, facts
- Episodic: Past restaurant recommendations, search history
- Procedural: Learned conversation patterns, response styles
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
    Comprehensive memory system for the restaurant bot using LangGraph memory store

    Manages three types of memory:
    1. Short-term: Current conversation context (thread-scoped)
    2. Long-term: User preferences and history (cross-thread)
    3. Session: Active search state and temporary data
    """

    def __init__(self, config):
        self.config = config

        # Initialize LangGraph memory store
        # In production, this would be a persistent store like PostgreSQL or MongoDB
        self.memory_store = InMemoryStore()

        logger.info("✅ AI Memory System initialized")

    # =====================================================================
    # USER NAMESPACE MANAGEMENT
    # =====================================================================

    def _get_user_namespace(self, user_id: int) -> str:
        """Get namespace for user's long-term memory"""
        return f"user_{user_id}"

    def _get_session_namespace(self, user_id: int, thread_id: str) -> str:
        """Get namespace for session-specific memory"""
        return f"session_{user_id}_{thread_id}"

    # =====================================================================
    # SEMANTIC MEMORY (User Preferences)
    # =====================================================================

    async def get_user_preferences(self, user_id: int) -> UserPreferences:
        """Get user's dining preferences"""
        try:
            namespace = self._get_user_namespace(user_id)

            # Try to get existing preferences
            stored_items = await self.memory_store.aget(namespace, "preferences")

            if stored_items:
                preferences_data = stored_items[0].value
                return UserPreferences.from_dict(preferences_data)
            else:
                # Return default preferences
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
            namespace = self._get_user_namespace(user_id)

            # Store preferences
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
            namespace = self._get_user_namespace(user_id)

            # Get existing restaurant memories
            restaurants = await self.get_restaurant_history(user_id)

            # Add new restaurant
            restaurants.append(restaurant_memory)

            # Keep only last 100 restaurant memories to prevent bloat
            if len(restaurants) > 100:
                restaurants = restaurants[-100:]

            # Store updated list
            restaurant_dicts = [r.to_dict() for r in restaurants]
            await self.memory_store.aput(
                namespace,
                "restaurant_history", 
                restaurant_dicts
            )

            logger.info(f"Added restaurant memory for user {user_id}: {restaurant_memory.restaurant_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding restaurant memory: {e}")
            return False

    async def get_restaurant_history(self, user_id: int) -> List[RestaurantMemory]:
        """Get user's restaurant history"""
        try:
            namespace = self._get_user_namespace(user_id)

            stored_items = await self.memory_store.aget(namespace, "restaurant_history")

            if stored_items:
                restaurant_dicts = stored_items[0].value
                return [RestaurantMemory.from_dict(r) for r in restaurant_dicts]
            else:
                return []

        except Exception as e:
            logger.error(f"Error getting restaurant history: {e}")
            return []

    async def get_restaurants_for_city(self, user_id: int, city: str) -> List[RestaurantMemory]:
        """Get restaurants user has been recommended in a specific city"""
        try:
            all_restaurants = await self.get_restaurant_history(user_id)
            return [r for r in all_restaurants if r.city.lower() == city.lower()]
        except Exception as e:
            logger.error(f"Error getting restaurants for city {city}: {e}")
            return []

    async def has_restaurant_been_recommended(self, user_id: int, restaurant_name: str, city: str) -> bool:
        """Check if a restaurant has already been recommended to user"""
        try:
            city_restaurants = await self.get_restaurants_for_city(user_id, city)
            return any(
                r.restaurant_name.lower() == restaurant_name.lower() 
                for r in city_restaurants
            )
        except Exception as e:
            logger.error(f"Error checking restaurant recommendation history: {e}")
            return False

    # =====================================================================
    # SESSION MEMORY (Current Conversation)
    # =====================================================================

    async def set_session_state(
        self, 
        user_id: int, 
        thread_id: str, 
        state: ConversationState,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set current session state and context"""
        try:
            namespace = self._get_session_namespace(user_id, thread_id)

            session_data = {
                "state": state.value,
                "context": context or {},
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            await self.memory_store.aput(namespace, "session_state", session_data)
            return True

        except Exception as e:
            logger.error(f"Error setting session state: {e}")
            return False

    async def get_session_state(self, user_id: int, thread_id: str) -> Tuple[ConversationState, Dict[str, Any]]:
        """Get current session state and context"""
        try:
            namespace = self._get_session_namespace(user_id, thread_id)

            stored_items = await self.memory_store.aget(namespace, "session_state")

            if stored_items:
                session_data = stored_items[0].value
                state = ConversationState(session_data["state"])
                context = session_data.get("context", {})
                return state, context
            else:
                return ConversationState.IDLE, {}

        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            return ConversationState.IDLE, {}

    async def set_current_city(self, user_id: int, thread_id: str, city: str) -> bool:
        """Set user's current city for this session"""
        try:
            state, context = await self.get_session_state(user_id, thread_id)
            context["current_city"] = city
            return await self.set_session_state(user_id, thread_id, state, context)
        except Exception as e:
            logger.error(f"Error setting current city: {e}")
            return False

    async def get_current_city(self, user_id: int, thread_id: str) -> Optional[str]:
        """Get user's current city for this session"""
        try:
            _, context = await self.get_session_state(user_id, thread_id)
            return context.get("current_city")
        except Exception as e:
            logger.error(f"Error getting current city: {e}")
            return None

    # =====================================================================
    # CONVERSATION PATTERNS (Procedural Memory)
    # =====================================================================

    async def get_conversation_patterns(self, user_id: int) -> ConversationPattern:
        """Get user's conversation patterns"""
        try:
            namespace = self._get_user_namespace(user_id)

            stored_items = await self.memory_store.aget(namespace, "conversation_patterns")

            if stored_items:
                pattern_data = stored_items[0].value
                return ConversationPattern.from_dict(pattern_data)
            else:
                # Return default patterns
                return ConversationPattern(
                    user_communication_style="casual",
                    preferred_response_length="medium",
                    likes_follow_up_questions=True,
                    prefers_immediate_results=True,
                    timezone=None,
                    typical_search_times=[]
                )

        except Exception as e:
            logger.error(f"Error getting conversation patterns: {e}")
            return ConversationPattern(
                user_communication_style="casual",
                preferred_response_length="medium",
                likes_follow_up_questions=True,
                prefers_immediate_results=True,
                timezone=None,
                typical_search_times=[]
            )

    async def learn_conversation_patterns(
        self, 
        user_id: int, 
        message: str, 
        response_time: float
    ) -> bool:
        """Learn from user's conversation patterns"""
        try:
            patterns = await self.get_conversation_patterns(user_id)

            # Simple pattern learning (in production, use more sophisticated analysis)
            if len(message) < 20:
                patterns.preferred_response_length = "short"
            elif len(message) > 100:
                patterns.preferred_response_length = "detailed"

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
                "conversation_patterns": conversation_patterns.to_dict(),
                "restaurant_count": len(restaurant_history),
                "recent_restaurants": [
                    r.to_dict() for r in restaurant_history[-5:]  # Last 5 restaurants
                ]
            }

            # Add city-specific context if current city is set
            if current_city:
                city_restaurants = await self.get_restaurants_for_city(user_id, current_city)
                context["city_restaurant_count"] = len(city_restaurants)
                context["city_recent_restaurants"] = [
                    r.to_dict() for r in city_restaurants[-3:]  # Last 3 for this city
                ]

            return context

        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                "user_id": user_id,
                "thread_id": thread_id,
                "error": str(e)
            }

    # =====================================================================
    # MEMORY CLEANUP AND MAINTENANCE
    # =====================================================================

    async def cleanup_old_sessions(self, user_id: int, days_old: int = 7) -> bool:
        """Clean up old session data"""
        try:
            # This would typically clean up sessions older than X days
            # For now, we'll just log the intent
            logger.info(f"Would clean up sessions older than {days_old} days for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return False


# =====================================================================
# FACTORY FUNCTION
# =====================================================================

def create_ai_memory_system(config) -> AIMemorySystem:
    """Factory function to create AI memory system"""
    return AIMemorySystem(config)