# utils/supabase_memory_system.py
"""
Supabase REST API Memory System for Restaurant Bot

Uses Supabase REST API (not direct PostgreSQL) for persistent storage of:
- User preferences (semantic memory)
- Restaurant history (episodic memory)
- Conversation patterns (procedural memory)
- Session data (short-term memory)

NO psycopg2 - Uses supabase-py client instead!
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Import data classes from original memory system
from utils.ai_memory_system import (
    UserPreferences, 
    RestaurantMemory, 
    ConversationPattern, 
    MemoryType, 
    ConversationState
)


class SupabaseMemoryStore:
    """
    Supabase REST API-based persistent memory store

    This uses Supabase's REST API instead of direct PostgreSQL connections,
    making it compatible with Railway and other platforms.
    """

    def __init__(self, config):
        """Initialize Supabase client"""
        self.config = config

        try:
            from supabase import create_client, Client

            self.supabase: Client = create_client(
                config.SUPABASE_URL,
                config.SUPABASE_KEY
            )
            logger.info("✅ Supabase Memory Store initialized (REST API)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase Memory Store: {e}")
            raise

    # =====================================================================
    # USER PREFERENCES (Semantic Memory)
    # =====================================================================

    async def get_user_preferences(self, user_id: int) -> UserPreferences:
        """Get user preferences from Supabase"""
        try:
            result = self.supabase.table('user_preferences')\
                .select('*')\
                .eq('user_id', user_id)\
                .maybe_single()\
                .execute()

            if result.data:
                return UserPreferences(
                    preferred_cities=result.data.get('preferred_cities', []),
                    preferred_cuisines=result.data.get('preferred_cuisines', []),
                    dietary_restrictions=result.data.get('dietary_restrictions', []),
                    budget_range=result.data.get('budget_range', 'mid-range'),
                    preferred_ambiance=result.data.get('preferred_ambiance', []),
                    meal_times=result.data.get('meal_times', []),
                    group_size_typical=result.data.get('group_size_typical', 'couple')
                )
            else:
                # Return defaults
                return UserPreferences(
                    preferred_cities=[],
                    preferred_cuisines=[],
                    dietary_restrictions=[],
                    budget_range='mid-range',
                    preferred_ambiance=[],
                    meal_times=[],
                    group_size_typical='couple'
                )

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            # Return defaults on error
            return UserPreferences(
                preferred_cities=[],
                preferred_cuisines=[],
                dietary_restrictions=[],
                budget_range='mid-range',
                preferred_ambiance=[],
                meal_times=[],
                group_size_typical='couple'
            )

    async def update_user_preferences(
        self, 
        user_id: int, 
        preferences: UserPreferences
    ) -> bool:
        """Update user preferences in Supabase"""
        try:
            data = {
                'user_id': user_id,
                'preferred_cities': preferences.preferred_cities,
                'preferred_cuisines': preferences.preferred_cuisines,
                'dietary_restrictions': preferences.dietary_restrictions,
                'budget_range': preferences.budget_range,
                'preferred_ambiance': preferences.preferred_ambiance,
                'meal_times': preferences.meal_times,
                'group_size_typical': preferences.group_size_typical,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            # Upsert (insert or update)
            self.supabase.table('user_preferences')\
                .upsert(data, on_conflict='user_id')\
                .execute()

            logger.info(f"✅ Updated preferences for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False

    # =====================================================================
    # RESTAURANT HISTORY (Episodic Memory)
    # =====================================================================

    async def get_restaurant_history(
        self, 
        user_id: int,
        city: Optional[str] = None,
        limit: int = 100
    ) -> List[RestaurantMemory]:
        """Get user's restaurant recommendation history"""
        try:
            query = self.supabase.table('restaurant_history')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('recommended_date', desc=True)\
                .limit(limit)

            if city:
                query = query.eq('city', city)

            result = query.execute()

            # Convert to RestaurantMemory objects
            return [
                RestaurantMemory(
                    restaurant_name=row['restaurant_name'],
                    city=row['city'],
                    cuisine=row.get('cuisine', ''),
                    recommended_date=row['recommended_date'] if row['recommended_date'] else '',
                    user_feedback=row.get('user_feedback'),
                    rating_given=float(row['rating_given']) if row.get('rating_given') else None,
                    notes=row.get('notes'),
                    source=row.get('source', '')
                )
                for row in result.data
            ]

        except Exception as e:
            logger.error(f"Error getting restaurant history: {e}")
            return []

    async def add_restaurant_memory(
        self, 
        user_id: int, 
        restaurant_memory: RestaurantMemory
    ) -> bool:
        """Add a restaurant to user's history"""
        try:
            data = {
                'user_id': user_id,
                'restaurant_name': restaurant_memory.restaurant_name,
                'city': restaurant_memory.city,
                'cuisine': restaurant_memory.cuisine,
                'user_feedback': restaurant_memory.user_feedback,
                'rating_given': restaurant_memory.rating_given,
                'notes': restaurant_memory.notes,
                'source': restaurant_memory.source,
                'recommended_date': datetime.now(timezone.utc).isoformat()
            }

            self.supabase.table('restaurant_history').insert(data).execute()

            logger.info(f"✅ Added restaurant memory: {restaurant_memory.restaurant_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding restaurant memory: {e}")
            return False

    # =====================================================================
    # CONVERSATION PATTERNS (Procedural Memory)
    # =====================================================================

    async def get_conversation_patterns(self, user_id: int) -> ConversationPattern:
        """Get user's conversation patterns"""
        try:
            result = self.supabase.table('conversation_patterns')\
                .select('*')\
                .eq('user_id', user_id)\
                .maybe_single()\
                .execute()

            if result.data:
                return ConversationPattern(
                    user_communication_style=result.data.get('user_communication_style', 'casual'),
                    preferred_response_length=result.data.get('preferred_response_length', 'medium'),
                    likes_follow_up_questions=result.data.get('likes_follow_up_questions', True),
                    prefers_immediate_results=result.data.get('prefers_immediate_results', True),
                    timezone=result.data.get('timezone'),
                    typical_search_times=result.data.get('typical_search_times', [])
                )
            else:
                # Return defaults
                return ConversationPattern(
                    user_communication_style='casual',
                    preferred_response_length='medium',
                    likes_follow_up_questions=True,
                    prefers_immediate_results=True,
                    timezone=None,
                    typical_search_times=[]
                )

        except Exception as e:
            logger.error(f"Error getting conversation patterns: {e}")
            return ConversationPattern(
                user_communication_style='casual',
                preferred_response_length='medium',
                likes_follow_up_questions=True,
                prefers_immediate_results=True,
                timezone=None,
                typical_search_times=[]
            )

    async def update_conversation_patterns(
        self, 
        user_id: int, 
        patterns: ConversationPattern
    ) -> bool:
        """Update user's conversation patterns"""
        try:
            data = {
                'user_id': user_id,
                'user_communication_style': patterns.user_communication_style,
                'preferred_response_length': patterns.preferred_response_length,
                'likes_follow_up_questions': patterns.likes_follow_up_questions,
                'prefers_immediate_results': patterns.prefers_immediate_results,
                'timezone': patterns.timezone,
                'typical_search_times': patterns.typical_search_times,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            # Upsert
            self.supabase.table('conversation_patterns')\
                .upsert(data, on_conflict='user_id')\
                .execute()

            logger.info(f"✅ Updated conversation patterns for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating conversation patterns: {e}")
            return False

    # =====================================================================
    # SESSION MEMORY (Short-term/Temporary)
    # =====================================================================

    async def get_session_data(
        self, 
        user_id: int, 
        thread_id: str
    ) -> Dict[str, Any]:
        """Get session-specific temporary data"""
        try:
            result = self.supabase.table('session_memory')\
                .select('session_data, expires_at')\
                .eq('user_id', user_id)\
                .eq('thread_id', thread_id)\
                .maybe_single()\
                .execute()

            if result.data:
                # Check if expired
                expires_at = result.data.get('expires_at')
                if expires_at:
                    expires_dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    if expires_dt < datetime.now(timezone.utc):
                        # Expired - delete and return empty
                        await self._delete_session(user_id, thread_id)
                        return {}

                return result.data.get('session_data', {})
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return {}

    async def update_session_data(
        self, 
        user_id: int, 
        thread_id: str,
        session_data: Dict[str, Any],
        expire_hours: int = 24
    ) -> bool:
        """Update session-specific temporary data"""
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expire_hours)

            data = {
                'user_id': user_id,
                'thread_id': thread_id,
                'session_data': session_data,
                'expires_at': expires_at.isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            # Upsert
            self.supabase.table('session_memory')\
                .upsert(data, on_conflict='user_id,thread_id')\
                .execute()

            return True

        except Exception as e:
            logger.error(f"Error updating session data: {e}")
            return False

    async def _delete_session(self, user_id: int, thread_id: str) -> bool:
        """Delete a session"""
        try:
            self.supabase.table('session_memory')\
                .delete()\
                .eq('user_id', user_id)\
                .eq('thread_id', thread_id)\
                .execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    # =====================================================================
    # CONVERSATION HISTORY (Persistent Chat Dialog)
    # =====================================================================

    async def get_conversation_history(
        self, 
        user_id: int, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get user's recent conversation history from Supabase.
        Returns the last N messages as a list of {role, message, timestamp} dicts.
        """
        try:
            result = self.supabase.table('conversation_history')\
                .select('role, message, created_at')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()

            if result.data:
                # Reverse to get chronological order (oldest first)
                messages = list(reversed(result.data))
                return [
                    {
                        'role': row['role'],
                        'message': row['message'],
                        'timestamp': row['created_at']
                    }
                    for row in messages
                ]
            return []

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def add_conversation_message(
        self, 
        user_id: int, 
        role: str, 
        message: str
    ) -> bool:
        """
        Add a message to user's conversation history.
        Also trims old messages to keep only the last 10.
        """
        try:
            # Insert new message
            data = {
                'user_id': user_id,
                'role': role,
                'message': message
            }
            self.supabase.table('conversation_history').insert(data).execute()

            # Trim old messages - keep only last 10
            await self._trim_conversation_history(user_id, keep_last=10)

            logger.debug(f"💬 Added {role} message for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding conversation message: {e}")
            return False

    async def _trim_conversation_history(self, user_id: int, keep_last: int = 10) -> None:
        """Remove old messages, keeping only the most recent ones"""
        try:
            # Get IDs of messages to keep
            keep_result = self.supabase.table('conversation_history')\
                .select('id')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(keep_last)\
                .execute()

            if keep_result.data and len(keep_result.data) >= keep_last:
                keep_ids = [row['id'] for row in keep_result.data]

                # Delete messages NOT in the keep list
                # Using a subquery approach: delete where id < minimum kept id
                min_keep_id = min(keep_ids)

                self.supabase.table('conversation_history')\
                    .delete()\
                    .eq('user_id', user_id)\
                    .lt('id', min_keep_id)\
                    .execute()

        except Exception as e:
            logger.error(f"Error trimming conversation history: {e}")

    async def clear_conversation_history(self, user_id: int) -> bool:
        """Clear all conversation history for a user"""
        try:
            self.supabase.table('conversation_history')\
                .delete()\
                .eq('user_id', user_id)\
                .execute()

            logger.info(f"🗑️ Cleared conversation history for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            return False

    # =====================================================================
    # CLEANUP & MAINTENANCE
    # =====================================================================

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired session data"""
        try:
            # Delete sessions where expires_at < now
            now = datetime.now(timezone.utc).isoformat()

            result = self.supabase.table('session_memory')\
                .delete()\
                .lt('expires_at', now)\
                .execute()

            deleted_count = len(result.data) if result.data else 0

            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} expired sessions")

            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0

    # =====================================================================
    # LANGGRAPH STORE COMPATIBILITY METHODS
    # =====================================================================

    async def aput(self, namespace: tuple, key: str, value: Dict[str, Any]) -> None:
        """
        LangGraph Store compatibility: Store data in namespace

        This method provides compatibility with LangGraph's BaseStore interface
        """
        try:
            # Convert namespace tuple to string
            namespace_str = "_".join(str(x) for x in namespace)

            # Store in a generic key-value table
            data = {
                'namespace': namespace_str,
                'key': key,
                'value': value,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            self.supabase.table('memory_store')\
                .upsert(data, on_conflict='namespace,key')\
                .execute()

        except Exception as e:
            logger.error(f"Error in aput: {e}")

    async def aget(self, namespace: tuple, key: str) -> Optional[Dict[str, Any]]:
        """
        LangGraph Store compatibility: Get data from namespace
        """
        try:
            namespace_str = "_".join(str(x) for x in namespace)

            result = self.supabase.table('memory_store')\
                .select('value')\
                .eq('namespace', namespace_str)\
                .eq('key', key)\
                .maybe_single()\
                .execute()

            if result.data:
                return result.data.get('value')
            return None

        except Exception as e:
            logger.error(f"Error in aget: {e}")
            return None

    async def adelete(self, namespace: tuple, key: str) -> None:
        """
        LangGraph Store compatibility: Delete data from namespace
        """
        try:
            namespace_str = "_".join(str(x) for x in namespace)

            self.supabase.table('memory_store')\
                .delete()\
                .eq('namespace', namespace_str)\
                .eq('key', key)\
                .execute()

        except Exception as e:
            logger.error(f"Error in adelete: {e}")


# =====================================================================
# FACTORY FUNCTION
# =====================================================================

def create_supabase_memory_store(config) -> SupabaseMemoryStore:
    """Factory function to create Supabase memory store using REST API"""
    return SupabaseMemoryStore(config)