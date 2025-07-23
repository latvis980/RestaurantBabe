# utils/database.py - Supabase adapter (replaces your old PostgreSQL setup)
import logging
from typing import Dict, Any, Optional, List
from .supabase_manager import SupabaseManager

logger = logging.getLogger(__name__)

# Global instance
_supabase_manager = None

def initialize_db(config):
    """Initialize Supabase connection (replaces old PostgreSQL initialization)"""
    global _supabase_manager

    if _supabase_manager is not None:
        return  # Already initialized

    try:
        _supabase_manager = SupabaseManager(config)

        # Create vector search function if it doesn't exist
        _supabase_manager.create_vector_search_function()

        logger.info("Supabase database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing Supabase: {e}")
        raise

def get_supabase_manager() -> SupabaseManager:
    """Get the global Supabase manager instance"""
    if _supabase_manager is None:
        raise RuntimeError("Supabase not initialized. Call initialize_db() first.")
    return _supabase_manager

# ============ LEGACY COMPATIBILITY FUNCTIONS ============
# These maintain compatibility with your existing code

def save_data(table_name: str, data_dict: Dict[str, Any], config) -> Optional[str]:
    """Legacy function for saving data - now routes to appropriate Supabase method"""
    try:
        manager = get_supabase_manager()

        # Route to appropriate method based on old table names
        if table_name == "user_preferences":
            user_id = data_dict.get('user_id') or data_dict.get('id')
            if user_id:
                success = manager.save_user_preferences(user_id, data_dict)
                return user_id if success else None

        elif table_name == "searches":
            # Cache search results
            query = data_dict.get('query', '')
            if query:
                success = manager.cache_search_results(query, data_dict)
                return data_dict.get('id') if success else None

        elif table_name == "processes":
            # For process logging, we could store in user preferences or create a separate method
            logger.info(f"Process logged: {data_dict.get('query', 'unknown')}")
            return data_dict.get('id')

        # For unknown tables, log warning
        logger.warning(f"Unknown table name in save_data: {table_name}")
        return None

    except Exception as e:
        logger.error(f"Error in legacy save_data: {e}")
        return None

def find_data(table_name: str, query: Dict[str, Any], config) -> Optional[Dict[str, Any]]:
    """Legacy function for finding data - now routes to appropriate Supabase method"""
    try:
        manager = get_supabase_manager()

        if table_name == "user_preferences":
            user_id = query.get('user_id')
            if user_id:
                preferences = manager.get_user_preferences(user_id)
                return preferences

        elif table_name == "searches":
            # Try to get cached results
            search_query = query.get('query', '')
            if search_query:
                cached = manager.get_cached_results(search_query)
                return cached

        # For unknown queries, return None
        logger.warning(f"Unknown table name in find_data: {table_name}")
        return None

    except Exception as e:
        logger.error(f"Error in legacy find_data: {e}")
        return None

# ============ NEW DOMAIN INTELLIGENCE FUNCTIONS ============
# Enhanced versions of your existing domain intelligence

def save_domain_intelligence(domain: str, intelligence_data: Dict[str, Any], config) -> bool:
    """Save domain intelligence data"""
    try:
        manager = get_supabase_manager()
        return manager.save_domain_intelligence(domain, intelligence_data)
    except Exception as e:
        logger.error(f"Error saving domain intelligence: {e}")
        return False

def get_domain_intelligence(domain: str, config) -> Optional[Dict[str, Any]]:
    """Get domain intelligence data"""
    try:
        manager = get_supabase_manager()
        return manager.get_domain_intelligence(domain)
    except Exception as e:
        logger.error(f"Error getting domain intelligence: {e}")
        return None

def update_domain_success(domain: str, success: bool, restaurants_found: int, config):
    """Update domain success metrics"""
    try:
        manager = get_supabase_manager()
        manager.update_domain_success(domain, success, restaurants_found)
    except Exception as e:
        logger.error(f"Error updating domain success: {e}")

def get_trusted_domains(config, min_confidence: float = None) -> List[str]:
    """Get list of trusted domains"""
    try:
        manager = get_supabase_manager()
        return manager.get_trusted_domains(min_confidence)
    except Exception as e:
        logger.error(f"Error getting trusted domains: {e}")
        return []

# ============ NEW RAG FUNCTIONS ============

def save_scraped_content(source_url: str, content: str, restaurant_mentions: Optional[List[str]] = None) -> bool:
    """Save scraped content for RAG"""
    try:
        manager = get_supabase_manager()
        return manager.save_scraped_content(source_url, content, restaurant_mentions)
    except Exception as e:
        logger.error(f"Error saving scraped content: {e}")
        return False

def search_similar_content(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar content using vector search"""
    try:
        manager = get_supabase_manager()
        return manager.search_similar_content(query, limit)
    except Exception as e:
        logger.error(f"Error searching similar content: {e}")
        return []

def save_restaurant_data(restaurant_data: Dict[str, Any]) -> Optional[str]:
    """Save restaurant data"""
    try:
        manager = get_supabase_manager()
        return manager.save_restaurant(restaurant_data)
    except Exception as e:
        logger.error(f"Error saving restaurant data: {e}")
        return None