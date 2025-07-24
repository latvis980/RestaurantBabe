# utils/database.py - Clean Supabase-only interface
import logging
from typing import Dict, Any, Optional, List
from .supabase_manager import SupabaseManager

logger = logging.getLogger(__name__)

# Global instance
_supabase_manager = None

def initialize_db(config):
    """Initialize Supabase connection"""
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

# ============ DOMAIN INTELLIGENCE FUNCTIONS ============
# Direct delegation to SupabaseManager methods

def save_domain_intelligence(domain: str, intelligence_data: Dict[str, Any], config=None) -> bool:
    """Save domain intelligence data"""
    try:
        manager = get_supabase_manager()
        return manager.save_domain_intelligence(domain, intelligence_data)
    except Exception as e:
        logger.error(f"Error saving domain intelligence: {e}")
        return False

def get_domain_intelligence(domain: str, config=None) -> Optional[Dict[str, Any]]:
    """Get domain intelligence data"""
    try:
        manager = get_supabase_manager()
        return manager.get_domain_intelligence(domain)
    except Exception as e:
        logger.error(f"Error getting domain intelligence: {e}")
        return None

def update_domain_success(domain: str, success: bool, restaurants_found: int = 0, config=None):
    """Update domain success metrics"""
    try:
        manager = get_supabase_manager()
        manager.update_domain_success(domain, success, restaurants_found)
    except Exception as e:
        logger.error(f"Error updating domain success: {e}")

def get_trusted_domains(config=None, min_confidence: float = None) -> List[str]:
    """Get list of trusted domains"""
    try:
        manager = get_supabase_manager()
        return manager.get_trusted_domains(min_confidence)
    except Exception as e:
        logger.error(f"Error getting trusted domains: {e}")
        return []

def load_all_domain_intelligence(config=None) -> List[Dict[str, Any]]:
    """Load all domain intelligence records"""
    try:
        manager = get_supabase_manager()
        return manager.load_all_domain_intelligence()
    except Exception as e:
        logger.error(f"Error loading domain intelligence: {e}")
        return []

def cleanup_old_domain_intelligence(config=None, days_old: int = 90, min_confidence: float = 0.3) -> int:
    """Clean up old domain intelligence records"""
    try:
        manager = get_supabase_manager()
        return manager.cleanup_old_domain_intelligence(days_old, min_confidence)
    except Exception as e:
        logger.error(f"Error cleaning up domain intelligence: {e}")
        return 0

def get_domain_intelligence_stats(config=None) -> Dict[str, Any]:
    """Get domain intelligence statistics"""
    try:
        manager = get_supabase_manager()
        return manager.get_domain_intelligence_stats()
    except Exception as e:
        logger.error(f"Error getting domain intelligence stats: {e}")
        return {}

def export_domain_intelligence(config=None, file_path: str = None) -> str:
    """Export domain intelligence to JSON"""
    try:
        manager = get_supabase_manager()
        return manager.export_domain_intelligence(file_path)
    except Exception as e:
        logger.error(f"Error exporting domain intelligence: {e}")
        raise

def import_domain_intelligence(config=None, file_path: str = None) -> int:
    """Import domain intelligence from JSON"""
    try:
        manager = get_supabase_manager()
        return manager.import_domain_intelligence(file_path)
    except Exception as e:
        logger.error(f"Error importing domain intelligence: {e}")
        raise

# ============ USER & SEARCH FUNCTIONS ============

def save_user_preferences(user_id: str, preferences: Dict[str, Any], config=None) -> bool:
    """Save user preferences"""
    try:
        manager = get_supabase_manager()
        return manager.save_user_preferences(user_id, preferences)
    except Exception as e:
        logger.error(f"Error saving user preferences: {e}")
        return False

def get_user_preferences(user_id: str, config=None) -> Optional[Dict[str, Any]]:
    """Get user preferences"""
    try:
        manager = get_supabase_manager()
        return manager.get_user_preferences(user_id)
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        return None

def add_to_search_history(user_id: str, query: str, results_count: int = 0, config=None):
    """Add search to user's history"""
    try:
        manager = get_supabase_manager()
        manager.add_to_search_history(user_id, query, results_count)
    except Exception as e:
        logger.error(f"Error adding to search history: {e}")

def cache_search_results(query: str, results: Dict[str, Any], config=None) -> bool:
    """Cache search results"""
    try:
        manager = get_supabase_manager()
        return manager.cache_search_results(query, results)
    except Exception as e:
        logger.error(f"Error caching search results: {e}")
        return False

def get_cached_results(query: str, config=None) -> Optional[Dict[str, Any]]:
    """Get cached search results"""
    try:
        manager = get_supabase_manager()
        return manager.get_cached_results(query)
    except Exception as e:
        logger.error(f"Error getting cached results: {e}")
        return None

# ============ RAG FUNCTIONS ============

def save_scraped_content(source_url: str, content: str, restaurant_mentions: Optional[List[str]] = None, source_domain: str = None, config=None) -> bool:
    """
    Save scraped content for RAG with domain source attribution - SIMPLE VERSION
    """
    try:
        manager = get_supabase_manager()
        return manager.save_scraped_content(source_url, content, restaurant_mentions, source_domain)
    except Exception as e:
        logger.error(f"Error saving scraped content: {e}")
        return False

def search_similar_content(query: str, limit: int = 10, config=None) -> List[Dict[str, Any]]:
    """Search for similar content using vector search"""
    try:
        manager = get_supabase_manager()
        return manager.search_similar_content(query, limit)
    except Exception as e:
        logger.error(f"Error searching similar content: {e}")
        return []

def save_restaurant_data(restaurant_data: Dict[str, Any], config=None) -> Optional[str]:
    """Save restaurant data"""
    try:
        manager = get_supabase_manager()
        return manager.save_restaurant(restaurant_data)
    except Exception as e:
        logger.error(f"Error saving restaurant data: {e}")
        return None

# ============ LEGACY COMPATIBILITY ============

def save_data(table_name: str, data_dict: Dict[str, Any], config=None) -> Optional[str]:
    """Legacy function - routes to appropriate Supabase method"""
    try:
        if table_name == "user_preferences":
            user_id = data_dict.get('user_id') or data_dict.get('id')
            if user_id:
                success = save_user_preferences(user_id, data_dict)
                return user_id if success else None

        elif table_name == "searches":
            query = data_dict.get('query', '')
            if query:
                success = cache_search_results(query, data_dict)
                return data_dict.get('id') if success else None

        elif table_name == "processes":
            # Log process data and cache the results
            logger.info(f"Process logged: {data_dict.get('query', 'unknown')}")

            # If it's a restaurant query, cache it
            if 'query' in data_dict:
                cache_search_results(data_dict['query'], data_dict)

            return data_dict.get('id')

        logger.warning(f"Unknown table name in save_data: {table_name}")
        return None

    except Exception as e:
        logger.error(f"Error in legacy save_data: {e}")
        return None

def find_data(table_name: str, query: Dict[str, Any], config=None) -> Optional[Dict[str, Any]]:
    """Legacy function - routes to appropriate Supabase method"""
    try:
        if table_name == "user_preferences":
            user_id = query.get('user_id')
            if user_id:
                return get_user_preferences(user_id)

        elif table_name == "searches":
            search_query = query.get('query', '')
            if search_query:
                return get_cached_results(search_query)

        logger.warning(f"Unknown table name in find_data: {table_name}")
        return None

    except Exception as e:
        logger.error(f"Error in legacy find_data: {e}")
        return None