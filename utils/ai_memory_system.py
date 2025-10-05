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

# Try PostgreSQL first, fallback to in-memory
try:
    from langgraph.store.postgres import PostgresStore
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

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

        # Initialize LangGraph memory store with PostgreSQL support
        if (POSTGRES_AVAILABLE and 
            hasattr(config, 'DATABASE_URL') and 
            config.DATABASE_URL and 
            getattr(config, 'MEMORY_STORE_TYPE', 'in_memory') == 'postgresql'):

            try:
                self.memory_store = PostgresStore(
                    connection_string=config.DATABASE_URL,
                    schema="memory_store"
                )
                logger.info("✅ Using PostgreSQL memory store for persistence")
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL memory store: {e}")
                logger.info("Falling back to in-memory store")
                self.memory_store = InMemoryStore()
        else:
            # Use in-memory store for development or when PostgreSQL not available
            self.memory_store = InMemoryStore()
            logger.info("✅ Using in-memory store (data won't persist between restarts)")

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

            await self.memory_store.aput(
                namespace, 
                "preferences", 
                preferences.to_dict()
            )

            logger.info(f"Updated preferences for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False

    # =====================================================================
    # EPISODIC MEMORY (Restaurant History)
    # =====================================================================

    async def save_restaurant_memory(
        self, 
        user_id: int, 
        restaurant_memory: RestaurantMemory
    ) -> bool:
        """Save a restaurant recommendation to episodic memory"""
        try:
            namespace = self._get_user_namespace(user_id)

            # Get existing memories
            stored_items = await self.memory_store.aget(namespace, "restaurant_memories")
            memories = []

            if stored_items:
                memories = stored_items[0].value

            # Add new memory
            memories.append(restaurant_memory.to_dict())

            # Keep only the most recent N memories
            max_memories = getattr(self.config, 'MAX_RESTAURANT_MEMORIES', 100)
            if len(memories) > max_memories:
                memories = memories[-max_memories:]

            await self.memory_store.aput(namespace, "restaurant_memories", memories)

            logger.info(f"Saved restaurant memory for user {user_id}: {restaurant_memory.restaurant_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving restaurant memory: {e}")
            return False

    async def get_restaurant_memories(
        self, 
        user_id: int, 
        city: Optional[str] = None,
        limit: int = 10
    ) -> List[RestaurantMemory]:
        """Get user's restaurant recommendation history"""
        try:
            namespace = self._get_user_namespace(user_id)

            stored_items = await self.memory_store.aget(namespace, "restaurant_memories")

            if not stored_items:
                return []

            memories = stored_items[0].value
            restaurant_memories = [RestaurantMemory.from_dict(m) for m in memories]

            # Filter by city if specified
            if city:
                restaurant_memories = [
                    m for m in restaurant_memories 
                    if m.city.lower() == city.lower()
                ]

            # Return most recent memories first
            restaurant_memories.sort(key=lambda x: x.recommended_date, reverse=True)

            return restaurant_memories[:limit]

        except Exception as e:
            logger.error(f"Error getting restaurant memories: {e}")
            return []

    # =====================================================================
    # PROCEDURAL MEMORY (Conversation Patterns)
    # =====================================================================

    async def learn_conversation_pattern(
        self, 
        user_id: int, 
        pattern: ConversationPattern
    ) -> bool:
        """Learn and update user's conversation patterns"""
        try:
            namespace = self._get_user_namespace(user_id)

            await self.memory_store.aput(
                namespace, 
                "conversation_pattern", 
                pattern.to_dict()
            )

            logger.info(f"Updated conversation pattern for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error learning conversation pattern: {e}")
            return False

    async def get_conversation_pattern(self, user_id: int) -> ConversationPattern:
        """Get user's learned conversation patterns"""
        try:
            namespace = self._get_user_namespace(user_id)

            stored_items = await self.memory_store.aget(namespace, "conversation_pattern")

            if stored_items:
                pattern_data = stored_items[0].value
                return ConversationPattern.from_dict(pattern_data)
            else:
                # Return default pattern
                return ConversationPattern(
                    user_communication_style="casual",
                    preferred_response_length="medium",
                    likes_follow_up_questions=True,
                    prefers_immediate_results=False,
                    timezone=None,
                    typical_search_times=[]
                )

        except Exception as e:
            logger.error(f"Error getting conversation pattern: {e}")
            return ConversationPattern(
                user_communication_style="casual",
                preferred_response_length="medium",
                likes_follow_up_questions=True,
                prefers_immediate_results=False,
                timezone=None,
                typical_search_times=[]
            )

    # =====================================================================
    # SESSION MEMORY (Current Conversation State)
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

    # =====================================================================
    # COMPREHENSIVE USER CONTEXT
    # =====================================================================

    async def get_user_context(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get comprehensive user context for AI decision making"""
        try:
            # Get all memory components
            preferences = await self.get_user_preferences(user_id)
            session_state, session_context = await self.get_session_state(user_id, thread_id)
            conversation_pattern = await self.get_conversation_pattern(user_id)

            # Get current city from session
            current_city = session_context.get("current_city")
            recent_restaurants = await self.get_restaurant_memories(user_id, current_city, limit=5)

            return {
                "user_id": user_id,
                "thread_id": thread_id,
                "current_city": current_city,
                "session_state": session_state.value,
                "session_context": session_context,
                "preferences": preferences.to_dict(),
                "conversation_pattern": conversation_pattern.to_dict(),
                "recent_restaurants": [r.to_dict() for r in recent_restaurants],
                "total_restaurant_memories": len(await self.get_restaurant_memories(user_id))
            }

        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                "user_id": user_id,
                "thread_id": thread_id,
                "current_city": None,
                "session_state": "idle",
                "session_context": {},
                "preferences": {},
                "conversation_pattern": {},
                "recent_restaurants": [],
                "total_restaurant_memories": 0
            }

    # =====================================================================
    # AI LEARNING FUNCTIONS
    # =====================================================================

    async def learn_preferences_from_message(
        self, 
        user_id: int, 
        message: str, 
        current_city: Optional[str] = None
    ) -> bool:
        """Extract and learn preferences from user messages"""
        try:
            # Get current preferences
            preferences = await self.get_user_preferences(user_id)

            # Simple keyword-based learning (could be enhanced with NLP)
            message_lower = message.lower()

            # Learn cuisine preferences
            cuisines = ["italian", "japanese", "chinese", "thai", "indian", "french", "mexican", "korean"]
            for cuisine in cuisines:
                if cuisine in message_lower and cuisine not in preferences.preferred_cuisines:
                    preferences.preferred_cuisines.append(cuisine)

            # Learn dietary restrictions
            restrictions = ["vegetarian", "vegan", "gluten-free", "halal", "kosher"]
            for restriction in restrictions:
                if restriction in message_lower and restriction not in preferences.dietary_restrictions:
                    preferences.dietary_restrictions.append(restriction)

            # Learn cities
            if current_city and current_city not in preferences.preferred_cities:
                preferences.preferred_cities.append(current_city)

            # Update preferences if we learned something
            return await self.update_user_preferences(user_id, preferences)

        except Exception as e:
            logger.error(f"Error learning from message: {e}")
            return False

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
    # MEMORY CLEANUP AND MAINTENANCE
    # =====================================================================

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up old session data"""
        # This would be implemented for production cleanup
        # For now, return 0 as placeholder
        return 0

    async def get_memory_stats(self, user_id: int) -> Dict[str, Any]:
        """Get memory usage statistics for a user"""
        try:
            preferences = await self.get_user_preferences(user_id)
            restaurants = await self.get_restaurant_memories(user_id)

            return {
                "preferred_cities_count": len(preferences.preferred_cities),
                "preferred_cuisines_count": len(preferences.preferred_cuisines),
                "dietary_restrictions_count": len(preferences.dietary_restrictions),
                "total_restaurant_memories": len(restaurants),
                "has_conversation_pattern": True  # We always return a pattern
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}