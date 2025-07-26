# utils/database.py - SIMPLIFIED: Direct Supabase interface (no wrapper)
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from supabase import create_client, Client
import numpy as np

logger = logging.getLogger(__name__)

class Database:
    """
    Direct Supabase database interface for the AI-powered restaurant bot.
    Simplified, single-purpose database class with clear naming.
    """

    def __init__(self, config):
        """Initialize direct Supabase connection"""
        self.config = config

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("✅ Database (Supabase) initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise

        # Initialize geocoder (optional)
        try:
            from geopy.geocoders import Nominatim
            self.geocoder = Nominatim(user_agent="ai-restaurant-bot")
            logger.info("✅ Geocoder initialized")
        except Exception as e:
            logger.warning(f"⚠️ Geocoder initialization failed: {e}")
            self.geocoder = None

    # ============ RESTAURANT METHODS ============

    def save_restaurant(self, restaurant_data: Dict[str, Any]) -> Optional[str]:
        """Save restaurant with new simplified schema"""
        try:
            name = restaurant_data.get('name', '').strip()
            city = restaurant_data.get('city', '').strip()

            if not name or not city:
                logger.warning(f"Missing required fields: name='{name}', city='{city}'")
                return None

            # Check if restaurant already exists (by name and city)
            existing = self._find_existing_restaurant(name, city)

            if existing:
                # Update existing restaurant
                restaurant_id = existing['id']

                # Combine descriptions
                new_description = restaurant_data.get('raw_description', '')
                if new_description:
                    combined_description = existing['raw_description'] + "\n\n--- NEW MENTION ---\n\n" + new_description
                else:
                    combined_description = existing['raw_description']

                # Combine and deduplicate cuisine tags
                existing_tags = existing.get('cuisine_tags', [])
                new_tags = restaurant_data.get('cuisine_tags', [])
                combined_tags = list(set(existing_tags + new_tags))

                # Combine and deduplicate sources
                existing_sources = existing.get('sources', [])
                new_sources = restaurant_data.get('sources', [])
                combined_sources = list(set(existing_sources + new_sources))

                update_data = {
                    'raw_description': combined_description,
                    'cuisine_tags': combined_tags,
                    'sources': combined_sources,
                    'mention_count': existing['mention_count'] + 1,
                    'last_updated': datetime.now().isoformat(),
                    # Update address if we have a new one and existing is null
                    'address': restaurant_data.get('address') if existing['address'] is None else existing['address']
                }

                self.supabase.table('restaurants')\
                    .update(update_data)\
                    .eq('id', restaurant_id)\
                    .execute()

                logger.info(f"🔄 Updated existing restaurant: {name}")
                return str(restaurant_id)

            else:
                # Insert new restaurant
                insert_data = {
                    'name': name,
                    'raw_description': restaurant_data.get('raw_description', ''),
                    'address': restaurant_data.get('address'),
                    'city': city,
                    'country': restaurant_data.get('country', ''),
                    'cuisine_tags': restaurant_data.get('cuisine_tags', []),
                    'sources': restaurant_data.get('sources', []),
                    'mention_count': 1,
                    'first_added': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }

                # Add coordinates if available
                if restaurant_data.get('coordinates'):
                    coords = restaurant_data['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        insert_data['coordinates'] = f"POINT({coords[1]} {coords[0]})"  # lon, lat for PostGIS

                result = self.supabase.table('restaurants').insert(insert_data).execute()

                if result.data:
                    restaurant_id = result.data[0]['id']
                    logger.info(f"➕ Inserted new restaurant: {name}")
                    return str(restaurant_id)
                else:
                    logger.error(f"Failed to insert restaurant: {name}")
                    return None

        except Exception as e:
            logger.error(f"Error saving restaurant: {e}")
            return None

    def _find_existing_restaurant(self, name: str, city: str) -> Optional[Dict]:
        """Find existing restaurant by name and city"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('name', name)\
                .eq('city', city)\
                .execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error finding existing restaurant: {e}")
            return None

    def get_restaurants_by_city(self, city: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all restaurants for a city, ordered by mention count"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('city', city)\
                .order('mention_count', desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error getting restaurants for {city}: {e}")
            return []

    def search_restaurants_by_cuisine(self, city: str, cuisine_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Search restaurants by city and cuisine tags using array overlap"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('city', city)\
                .overlaps('cuisine_tags', cuisine_tags)\
                .order('mention_count', desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error searching restaurants by cuisine: {e}")
            return []

    def update_restaurant_geodata(self, restaurant_id: int, address: str, coordinates: Tuple[float, float]):
        """Update restaurant with address and coordinates"""
        try:
            update_data = {
                'address': address,
                'coordinates': f"POINT({coordinates[1]} {coordinates[0]})",  # lon, lat for PostGIS
                'last_updated': datetime.now().isoformat()
            }

            self.supabase.table('restaurants')\
                .update(update_data)\
                .eq('id', restaurant_id)\
                .execute()

            logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id}")

        except Exception as e:
            logger.error(f"Error updating geodata: {e}")

    def get_restaurants_by_preference_tags(self, city: str, preference_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Get restaurants that match any of the preference tags"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('city', city)\
                .overlaps('cuisine_tags', preference_tags)\
                .order('mention_count', desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error getting restaurants by preference tags: {e}")
            return []

    # ============ DOMAIN INTELLIGENCE METHODS ============

    def save_domain_intelligence(self, domain: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save domain intelligence data"""
        try:
            # Check if domain already exists
            existing = self.get_domain_intelligence(domain)

            if existing:
                # Update existing domain
                self.supabase.table('domain_intelligence')\
                    .update(intelligence_data)\
                    .eq('domain', domain)\
                    .execute()
            else:
                # Insert new domain
                data = {'domain': domain, **intelligence_data}
                self.supabase.table('domain_intelligence').insert(data).execute()

            logger.debug(f"💾 Saved domain intelligence for {domain}")
            return True

        except Exception as e:
            logger.error(f"Error saving domain intelligence for {domain}: {e}")
            return False

    def get_domain_intelligence(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain intelligence data"""
        try:
            result = self.supabase.table('domain_intelligence').select('*').eq('domain', domain).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting domain intelligence for {domain}: {e}")
            return None

    def update_domain_success(self, domain: str, success: bool, restaurants_found: int = 0):
        """Update domain success/failure counts"""
        try:
            current = self.get_domain_intelligence(domain)
            if not current:
                logger.warning(f"No domain intelligence found for {domain}")
                return

            # Update counts
            if success:
                new_success = current['success_count'] + 1
                new_failure = current['failure_count']
                update_data = {
                    'success_count': new_success,
                    'total_restaurants_found': current['total_restaurants_found'] + restaurants_found,
                    'last_successful_scrape': datetime.now(timezone.utc).isoformat()
                }
            else:
                new_success = current['success_count'] 
                new_failure = current['failure_count'] + 1
                update_data = {
                    'failure_count': new_failure
                }

                # Block domain if too many failures
                if new_failure >= getattr(self.config, 'DOMAIN_FAILURE_LIMIT', 10):
                    update_data.update({
                        'is_blocked': True,
                        'blocked_at': datetime.now(timezone.utc).isoformat()
                    })

            # Calculate new confidence
            total_attempts = new_success + new_failure
            if total_attempts > 0:
                update_data['confidence'] = new_success / total_attempts

            update_data['last_updated_at'] = datetime.now(timezone.utc).isoformat()

            # Update database
            self.supabase.table('domain_intelligence').update(update_data).eq('domain', domain).execute()
            logger.info(f"Updated domain intelligence for {domain}: success={success}")

        except Exception as e:
            logger.error(f"Error updating domain success for {domain}: {e}")

    def get_trusted_domains(self, min_confidence: float = None) -> List[str]:
        """Get list of trusted domains based on success rate"""
        try:
            min_conf = min_confidence or getattr(self.config, 'DOMAIN_SUCCESS_THRESHOLD', 0.7)

            result = self.supabase.table('domain_intelligence')\
                .select('domain')\
                .gte('confidence', min_conf)\
                .eq('is_blocked', False)\
                .execute()

            return [row['domain'] for row in result.data]

        except Exception as e:
            logger.error(f"Error getting trusted domains: {e}")
            return []

    def load_all_domain_intelligence(self) -> List[Dict[str, Any]]:
        """Load all domain intelligence records"""
        try:
            result = self.supabase.table('domain_intelligence').select('*').execute()
            return result.data
        except Exception as e:
            logger.error(f"Error loading domain intelligence: {e}")
            return []

    # ============ STATISTICS AND MONITORING ============

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            stats = {}

            # Count restaurants
            restaurants_count = self.supabase.table('restaurants').select('id', count='exact').execute()
            stats['total_restaurants'] = restaurants_count.count

            # Count domains
            domains_count = self.supabase.table('domain_intelligence').select('domain', count='exact').execute()
            stats['total_domains'] = domains_count.count

            # Get top cities
            cities_result = self.supabase.table('restaurants')\
                .select('city', count='exact')\
                .group_by('city')\
                .order('count', desc=True)\
                .limit(10)\
                .execute()

            stats['top_cities'] = cities_result.data

            # Coordinate coverage
            with_coords = self.supabase.table('restaurants')\
                .select('id', count='exact')\
                .not_.is_('coordinates', 'null')\
                .execute()

            stats['coordinate_coverage'] = {
                'with_coordinates': with_coords.count,
                'without_coordinates': restaurants_count.count - with_coords.count,
                'coverage_percentage': round((with_coords.count / restaurants_count.count * 100), 1) if restaurants_count.count > 0 else 0
            }

            logger.info(f"📊 Database stats: {stats['total_restaurants']} restaurants, {stats['total_domains']} domains")
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_restaurants': 0,
                'total_domains': 0,
                'top_cities': [],
                'coordinate_coverage': {'coverage_percentage': 0}
            }

    # ============ SIMPLE CACHE (OPTIONAL) ============

    def cache_search_results(self, query: str, results: Dict[str, Any]) -> bool:
        """Simple in-memory cache (placeholder for future implementation)"""
        # For now, just log that we would cache
        logger.debug(f"Would cache search results for: {query}")
        return True

    def get_cached_results(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search results (placeholder)"""
        # For now, always return None to force fresh searches
        return None

    # ============ GEOCODING HELPER ============

    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to get coordinates"""
        try:
            if not self.geocoder:
                logger.warning("Geocoder not available")
                return None

            location = self.geocoder.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None

        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
            return None

# ============ GLOBAL DATABASE INSTANCE ============

_database = None

def initialize_database(config):
    """Initialize global database instance"""
    global _database

    if _database is not None:
        return  # Already initialized

    try:
        _database = Database(config)

        # Log connection success with stats
        stats = _database.get_database_stats()
        logger.info(f"✅ Database initialized: {stats['total_restaurants']} restaurants, {stats['coordinate_coverage']['coverage_percentage']}% have coordinates")

    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        raise

def get_database() -> Database:
    """Get the global database instance"""
    if _database is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _database

# ============ CONVENIENCE FUNCTIONS ============
# These maintain compatibility with existing code

def save_restaurant_data(restaurant_data: Dict[str, Any]) -> Optional[str]:
    """Save restaurant data"""
    return get_database().save_restaurant(restaurant_data)

def get_restaurants_by_city(city: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get restaurants for a city"""
    return get_database().get_restaurants_by_city(city, limit)

def search_restaurants_by_cuisine(city: str, cuisine_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Search restaurants by cuisine tags"""
    return get_database().search_restaurants_by_cuisine(city, cuisine_tags, limit)

def save_domain_intelligence(domain: str, intelligence_data: Dict[str, Any]) -> bool:
    """Save domain intelligence data"""
    return get_database().save_domain_intelligence(domain, intelligence_data)

def get_domain_intelligence(domain: str) -> Optional[Dict[str, Any]]:
    """Get domain intelligence data"""
    return get_database().get_domain_intelligence(domain)

def update_domain_success(domain: str, success: bool, restaurants_found: int = 0):
    """Update domain success metrics"""
    get_database().update_domain_success(domain, success, restaurants_found)

def get_trusted_domains(min_confidence: float = None) -> List[str]:
    """Get list of trusted domains"""
    return get_database().get_trusted_domains(min_confidence)

def load_all_domain_intelligence() -> List[Dict[str, Any]]:
    """Load all domain intelligence records"""
    return get_database().load_all_domain_intelligence()

def cache_search_results(query: str, results: Dict[str, Any]) -> bool:
    """Cache search results"""
    return get_database().cache_search_results(query, results)

def get_cached_results(query: str) -> Optional[Dict[str, Any]]:
    """Get cached search results"""
    return get_database().get_cached_results(query)

# ============ LEGACY COMPATIBILITY STUBS ============
# These prevent errors but don't do anything with the simplified schema

def add_to_search_history(user_id: str, query: str, results_count: int = 0):
    """Legacy user history tracking (disabled in simplified schema)"""
    logger.debug(f"Search history tracking disabled: {user_id} searched for {query}")

def save_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """Legacy user preferences (disabled in simplified schema)"""
    logger.debug(f"User preferences disabled: {user_id}")
    return True

def get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    """Legacy user preferences (disabled in simplified schema)"""
    return None

# Deprecated RAG functions (return safe defaults)
def save_scraped_content(source_url: str, content: str, restaurant_mentions: Optional[List[str]] = None, source_domain: str = None) -> bool:
    """DEPRECATED: RAG disabled in simplified schema"""
    return True

def search_similar_content(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """DEPRECATED: Vector search disabled in simplified schema"""
    return []

# ============ BACKWARDS COMPATIBILITY FOR MAIN.PY ============

# For main.py to work without changes
def initialize_db(config):
    """Alias for initialize_database for main.py compatibility"""
    initialize_database(config)

def get_supabase_manager():
    """Alias for get_database for backwards compatibility"""
    return get_database()