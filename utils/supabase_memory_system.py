# utils/supabase_memory_system.py
"""
PostgreSQL-based Memory System for Restaurant Bot

Replaces InMemoryStore with persistent PostgreSQL storage for:
- User preferences (semantic memory)
- Restaurant history (episodic memory)
- Conversation patterns (procedural memory)
- Session data (short-term memory)
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

# Import data classes from original memory system
from utils.ai_memory_system import (
    UserPreferences,
    RestaurantMemory,
    ConversationPattern,
    MemoryType,
    ConversationState
)


class PostgresMemoryStore:
    """
    PostgreSQL-based persistent memory store

    This replaces LangGraph's InMemoryStore with actual database persistence
    """

    def __init__(self, config):
        """Initialize PostgreSQL connection"""
        self.config = config

        # Use Supabase PostgreSQL connection
        # Supabase provides this in: Settings > Database > Connection string
        self.database_url = os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL')

        if not self.database_url:
            # Fallback: Use Supabase URL and key to construct
            # Get from: Settings > API > Project URL and Connection Pooling
            supabase_url = os.getenv('SUPABASE_URL', '')
            # Extract host from URL (e.g., abcdefg.supabase.co)
            host = supabase_url.replace('https://', '').replace('http://', '')

            self.database_url = f"postgresql://postgres.{host.split('.')[0]}:{os.getenv('SUPABASE_DB_PASSWORD')}@{host}:5432/postgres"


    def _get_connection(self):
        """Get a connection from the pool"""
        return self.pool.getconn()

    def _return_connection(self, conn):
        """Return a connection to the pool"""
        self.pool.putconn(conn)

    # =====================================================================
    # USER PREFERENCES (Semantic Memory)
    # =====================================================================

    async def get_user_preferences(self, user_id: int) -> UserPreferences:
        """Get user preferences from database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                "SELECT * FROM user_preferences WHERE user_id = %s",
                (user_id,)
            )

            result = cursor.fetchone()
            cursor.close()

            if result:
                # Convert database record to UserPreferences object
                return UserPreferences(
                    preferred_cities=result['preferred_cities'] or [],
                    preferred_cuisines=result['preferred_cuisines'] or [],
                    dietary_restrictions=result['dietary_restrictions'] or [],
                    budget_range=result['budget_range'] or 'mid-range',
                    preferred_ambiance=result['preferred_ambiance'] or [],
                    meal_times=result['meal_times'] or [],
                    group_size_typical=result['group_size_typical'] or 'couple'
                )
            else:
                # Return default preferences
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
            # Return default on error
            return UserPreferences(
                preferred_cities=[],
                preferred_cuisines=[],
                dietary_restrictions=[],
                budget_range='mid-range',
                preferred_ambiance=[],
                meal_times=[],
                group_size_typical='couple'
            )
        finally:
            if conn:
                self._return_connection(conn)

    async def update_user_preferences(
        self, 
        user_id: int, 
        preferences: UserPreferences
    ) -> bool:
        """Update user preferences in database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Use INSERT ... ON CONFLICT to update or insert
            cursor.execute("""
                INSERT INTO user_preferences (
                    user_id, preferred_cities, preferred_cuisines, 
                    dietary_restrictions, budget_range, preferred_ambiance,
                    meal_times, group_size_typical
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    preferred_cities = EXCLUDED.preferred_cities,
                    preferred_cuisines = EXCLUDED.preferred_cuisines,
                    dietary_restrictions = EXCLUDED.dietary_restrictions,
                    budget_range = EXCLUDED.budget_range,
                    preferred_ambiance = EXCLUDED.preferred_ambiance,
                    meal_times = EXCLUDED.meal_times,
                    group_size_typical = EXCLUDED.group_size_typical,
                    updated_at = NOW()
            """, (
                user_id,
                preferences.preferred_cities,
                preferences.preferred_cuisines,
                preferences.dietary_restrictions,
                preferences.budget_range,
                preferences.preferred_ambiance,
                preferences.meal_times,
                preferences.group_size_typical
            ))

            conn.commit()
            cursor.close()

            logger.info(f"✅ Updated preferences for user {user_id}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error updating user preferences: {e}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

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
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if city:
                cursor.execute("""
                    SELECT * FROM restaurant_history 
                    WHERE user_id = %s AND city = %s
                    ORDER BY recommended_date DESC
                    LIMIT %s
                """, (user_id, city, limit))
            else:
                cursor.execute("""
                    SELECT * FROM restaurant_history 
                    WHERE user_id = %s
                    ORDER BY recommended_date DESC
                    LIMIT %s
                """, (user_id, limit))

            results = cursor.fetchall()
            cursor.close()

            # Convert to RestaurantMemory objects
            return [
                RestaurantMemory(
                    restaurant_name=row['restaurant_name'],
                    city=row['city'],
                    cuisine=row['cuisine'] or '',
                    recommended_date=row['recommended_date'].isoformat() if row['recommended_date'] else '',
                    user_feedback=row['user_feedback'],
                    rating_given=float(row['rating_given']) if row['rating_given'] else None,
                    notes=row['notes'],
                    source=row['source'] or ''
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error getting restaurant history: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)

    async def add_restaurant_memory(
        self, 
        user_id: int, 
        restaurant_memory: RestaurantMemory
    ) -> bool:
        """Add a restaurant to user's history"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO restaurant_history (
                    user_id, restaurant_name, city, cuisine,
                    user_feedback, rating_given, notes, source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                restaurant_memory.restaurant_name,
                restaurant_memory.city,
                restaurant_memory.cuisine,
                restaurant_memory.user_feedback,
                restaurant_memory.rating_given,
                restaurant_memory.notes,
                restaurant_memory.source
            ))

            conn.commit()
            cursor.close()

            logger.info(f"✅ Added restaurant memory: {restaurant_memory.restaurant_name}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error adding restaurant memory: {e}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

    # =====================================================================
    # CONVERSATION PATTERNS (Procedural Memory)
    # =====================================================================

    async def get_conversation_patterns(self, user_id: int) -> ConversationPattern:
        """Get user's conversation patterns"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                "SELECT * FROM conversation_patterns WHERE user_id = %s",
                (user_id,)
            )

            result = cursor.fetchone()
            cursor.close()

            if result:
                return ConversationPattern(
                    user_communication_style=result['user_communication_style'],
                    preferred_response_length=result['preferred_response_length'],
                    likes_follow_up_questions=result['likes_follow_up_questions'],
                    prefers_immediate_results=result['prefers_immediate_results'],
                    timezone=result['timezone'],
                    typical_search_times=result['typical_search_times'] or []
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
        finally:
            if conn:
                self._return_connection(conn)

    async def update_conversation_patterns(
        self, 
        user_id: int, 
        patterns: ConversationPattern
    ) -> bool:
        """Update user's conversation patterns"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO conversation_patterns (
                    user_id, user_communication_style, preferred_response_length,
                    likes_follow_up_questions, prefers_immediate_results,
                    timezone, typical_search_times
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    user_communication_style = EXCLUDED.user_communication_style,
                    preferred_response_length = EXCLUDED.preferred_response_length,
                    likes_follow_up_questions = EXCLUDED.likes_follow_up_questions,
                    prefers_immediate_results = EXCLUDED.prefers_immediate_results,
                    timezone = EXCLUDED.timezone,
                    typical_search_times = EXCLUDED.typical_search_times,
                    updated_at = NOW()
            """, (
                user_id,
                patterns.user_communication_style,
                patterns.preferred_response_length,
                patterns.likes_follow_up_questions,
                patterns.prefers_immediate_results,
                patterns.timezone,
                patterns.typical_search_times
            ))

            conn.commit()
            cursor.close()

            logger.info(f"✅ Updated conversation patterns for user {user_id}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error updating conversation patterns: {e}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

    # =====================================================================
    # SESSION MEMORY (Short-term/Temporary)
    # =====================================================================

    async def get_session_data(
        self, 
        user_id: int, 
        thread_id: str
    ) -> Dict[str, Any]:
        """Get session-specific temporary data"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT session_data FROM session_memory 
                WHERE user_id = %s AND thread_id = %s
                AND (expires_at IS NULL OR expires_at > NOW())
            """, (user_id, thread_id))

            result = cursor.fetchone()
            cursor.close()

            if result:
                return result['session_data'] or {}
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return {}
        finally:
            if conn:
                self._return_connection(conn)

    async def update_session_data(
        self, 
        user_id: int, 
        thread_id: str,
        session_data: Dict[str, Any],
        expire_hours: int = 24
    ) -> bool:
        """Update session-specific temporary data"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            expires_at = datetime.now(timezone.utc) + timedelta(hours=expire_hours)

            cursor.execute("""
                INSERT INTO session_memory (
                    user_id, thread_id, session_data, expires_at
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, thread_id) DO UPDATE SET
                    session_data = EXCLUDED.session_data,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = NOW()
            """, (
                user_id,
                thread_id,
                Json(session_data),
                expires_at
            ))

            conn.commit()
            cursor.close()

            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error updating session data: {e}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

    # =====================================================================
    # CLEANUP & MAINTENANCE
    # =====================================================================

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired session data"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT cleanup_expired_sessions()")
            deleted_count = cursor.fetchone()[0]

            conn.commit()
            cursor.close()

            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} expired sessions")

            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
        finally:
            if conn:
                self._return_connection(conn)

    def close(self):
        """Close all connections in the pool"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.info("✅ Closed PostgreSQL connection pool")


# =====================================================================
# FACTORY FUNCTION
# =====================================================================

def create_supabase_memory_store(config) -> PostgresMemoryStore:
    """Factory function to create Supabase memory store"""
    return PostgresMemoryStore(config)