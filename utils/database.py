# utils/database.py - CLEAN VERSION: Restaurant database operations only
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
    Clean Supabase database interface for restaurant operations.
    Focus: Restaurant CRUD operations, source quality tracking, and basic database management.
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

    # ======== SOURCE QUALITY METHODS ========

    def store_source_quality(self, destination: str, url: str, score: float) -> bool:
        """
        Store source quality information in the database

        Args:
            destination: The destination/city from the query
            url: The cleaned URL (main domain)
            score: Quality score from AI evaluation (0.0-1.0)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if this destination+url combination already exists
            existing = self.supabase.table('source_quality')\
                .select('*')\
                .eq('destination', destination)\
                .eq('url', url)\
                .execute()

            if existing.data:
                # Update existing record with new score (keep the highest score)
                current_score = existing.data[0].get('score', 0.0)
                new_score = max(current_score, score)

                self.supabase.table('source_quality')\
                    .update({
                        'score': new_score,
                        'last_updated': datetime.now(timezone.utc).isoformat(),
                        'mention_count': existing.data[0].get('mention_count', 0) + 1
                    })\
                    .eq('id', existing.data[0]['id'])\
                    .execute()

                logger.debug(f"📊 Updated source quality: {url} for {destination} (score: {new_score})")
            else:
                # Insert new record
                self.supabase.table('source_quality').insert({
                    'destination': destination,
                    'url': url,
                    'score': score,
                    'mention_count': 1,
                    'first_added': datetime.now(timezone.utc).isoformat(),
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }).execute()

                logger.debug(f"➕ Added new source quality: {url} for {destination} (score: {score})")

            return True

        except Exception as e:
            logger.error(f"❌ Error storing source quality: {e}")
            return False

    # ============ RESTAURANT METHODS ============

    def save_restaurant(self, restaurant_data: Dict[str, Any]) -> Optional[str]:
        """Save restaurant with simplified coordinate handling"""
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
                    combined_description = existing.get('raw_description', '') + "\n\n--- NEW MENTION ---\n\n" + new_description
                else:
                    combined_description = existing.get('raw_description', '')

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
                    'mention_count': existing.get('mention_count', 1) + 1,
                    'last_updated': datetime.now().isoformat(),
                    # Update address if we have a new one and existing is null
                    'address': restaurant_data.get('address') if existing.get('address') is None else existing.get('address'),
                    # Update country if we have a new one and existing is null
                    'country': restaurant_data.get('country') if existing.get('country') is None else existing.get('country')
                }

                # Handle coordinates safely
                if restaurant_data.get('coordinates'):
                    coords = restaurant_data['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        lat, lng = coords
                        update_data['latitude'] = lat
                        update_data['longitude'] = lng

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

                # Handle coordinates safely during insert
                if restaurant_data.get('coordinates'):
                    coords = restaurant_data['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        lat, lng = coords
                        insert_data['latitude'] = lat
                        insert_data['longitude'] = lng

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

    def get_restaurants_by_coordinates(self, center: Tuple[float, float], radius_km: float, limit: int = 20) -> List[Dict[str, Any]]:
        """Get restaurants within a geographic radius (requires PostGIS)"""
        try:
            # This requires PostGIS extensions in Supabase
            # For now, return empty list as a placeholder
            logger.warning("Geographic search not implemented - requires PostGIS")
            return []

        except Exception as e:
            logger.error(f"Error getting restaurants by coordinates: {e}")
            return []

    def update_restaurant_geodata(self, restaurant_id: int, address: str, coordinates: Tuple[float, float]):
        """Update restaurant with address and coordinates"""
        try:
            lat, lng = coordinates

            # Update the restaurant record
            update_data = {
                'address': address,
                'latitude': lat,
                'longitude': lng,
                'last_updated': datetime.now().isoformat()
            }

            self.supabase.table('restaurants')\
                .update(update_data)\
                .eq('id', restaurant_id)\
                .execute()

            logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id} with coords ({lat}, {lng})")

        except Exception as e:
            logger.error(f"Error updating geodata: {e}")

    def delete_closed_restaurants(self, business_status: str = None) -> Dict[str, Any]:
        """
        Delete restaurants that are permanently or temporarily closed from the database

        Args:
            business_status: Specific status to delete ('CLOSED_PERMANENTLY', 'CLOSED_TEMPORARILY') 
                            If None, deletes both types of closed restaurants

        Returns:
            Dictionary with deletion statistics
        """
        try:
            logger.info("🗑️ Starting closed restaurant deletion process")

            # Step 1: Find restaurants to delete based on business status
            if business_status:
                # Delete specific closure type
                query = self.supabase.table('restaurants')\
                    .select('id, name, city, business_status')\
                    .eq('business_status', business_status)
            else:
                # Delete all closed restaurants (both temporarily and permanently)
                query = self.supabase.table('restaurants')\
                    .select('id, name, city, business_status')\
                    .in_('business_status', ['CLOSED_PERMANENTLY', 'CLOSED_TEMPORARILY'])

            result = query.execute()
            restaurants_to_delete = result.data or []

            if not restaurants_to_delete:
                logger.info("✅ No closed restaurants found to delete")
                return {
                    'deleted_count': 0,
                    'deleted_restaurants': [],
                    'message': 'No closed restaurants found'
                }

            # Step 2: Log what we're about to delete
            logger.info(f"Found {len(restaurants_to_delete)} closed restaurants to delete:")
            deleted_restaurants = []

            for restaurant in restaurants_to_delete:
                restaurant_info = {
                    'id': restaurant['id'],
                    'name': restaurant['name'],
                    'city': restaurant['city'],
                    'business_status': restaurant.get('business_status', 'Unknown')
                }
                deleted_restaurants.append(restaurant_info)
                logger.info(f"  - {restaurant['name']} in {restaurant['city']} ({restaurant.get('business_status', 'Unknown')})")

            # Step 3: Delete the restaurants
            restaurant_ids = [r['id'] for r in restaurants_to_delete]

            delete_result = self.supabase.table('restaurants')\
                .delete()\
                .in_('id', restaurant_ids)\
                .execute()

            deleted_count = len(delete_result.data) if delete_result.data else 0

            logger.info(f"✅ Successfully deleted {deleted_count} closed restaurants from database")

            return {
                'deleted_count': deleted_count,
                'deleted_restaurants': deleted_restaurants,
                'deletion_completed_at': datetime.now(timezone.utc).isoformat(),
                'message': f'Successfully deleted {deleted_count} closed restaurants'
            }

        except Exception as e:
            logger.error(f"❌ Error deleting closed restaurants: {e}")
            return {
                'deleted_count': 0,
                'deleted_restaurants': [],
                'error': str(e),
                'message': 'Failed to delete closed restaurants'
            }

    def delete_restaurants_by_ids(self, restaurant_ids: List[int]) -> Dict[str, Any]:
        """
        Delete specific restaurants by their IDs

        Args:
            restaurant_ids: List of restaurant IDs to delete

        Returns:
            Dictionary with deletion statistics
        """
        try:
            if not restaurant_ids:
                return {
                    'deleted_count': 0,
                    'message': 'No restaurant IDs provided'
                }

            logger.info(f"🗑️ Deleting {len(restaurant_ids)} restaurants by ID")

            # First get the restaurant details for logging
            restaurants_query = self.supabase.table('restaurants')\
                .select('id, name, city')\
                .in_('id', restaurant_ids)\
                .execute()

            restaurants_info = restaurants_query.data or []

            # Log what we're deleting
            for restaurant in restaurants_info:
                logger.info(f"  - Deleting: {restaurant['name']} in {restaurant['city']} (ID: {restaurant['id']})")

            # Delete the restaurants
            delete_result = self.supabase.table('restaurants')\
                .delete()\
                .in_('id', restaurant_ids)\
                .execute()

            deleted_count = len(delete_result.data) if delete_result.data else 0

            logger.info(f"✅ Successfully deleted {deleted_count} restaurants by ID")

            return {
                'deleted_count': deleted_count,
                'deleted_restaurants': restaurants_info,
                'deletion_completed_at': datetime.now(timezone.utc).isoformat(),
                'message': f'Successfully deleted {deleted_count} restaurants'
            }

        except Exception as e:
            logger.error(f"❌ Error deleting restaurants by ID: {e}")
            return {
                'deleted_count': 0,
                'error': str(e),
                'message': 'Failed to delete restaurants by ID'
            }

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

    # ============ STATISTICS AND MONITORING ============

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            stats = {}

            # Count restaurants
            restaurants_count = self.supabase.table('restaurants').select('id', count='exact').execute()
            stats['total_restaurants'] = restaurants_count.count

            # Get cities manually (no group_by in current Supabase client)
            try:
                cities_result = self.supabase.table('restaurants')\
                    .select('city')\
                    .execute()

                # Count cities manually
                city_counts = {}
                for row in cities_result.data:
                    city = row.get('city', 'Unknown')
                    city_counts[city] = city_counts.get(city, 0) + 1

                # Sort by count and take top 10
                top_cities = sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                stats['top_cities'] = [{'city': city, 'count': count} for city, count in top_cities]

            except Exception as e:
                logger.warning(f"Could not get city stats: {e}")
                stats['top_cities'] = []

            # Coordinate coverage
            with_coords = self.supabase.table('restaurants')\
                .select('id', count='exact')\
                .not_.is_('latitude', 'null')\
                .execute()

            stats['coordinate_coverage'] = {
                'with_coordinates': with_coords.count,
                'without_coordinates': restaurants_count.count - with_coords.count,
                'coverage_percentage': round((with_coords.count / restaurants_count.count * 100), 1) if restaurants_count.count > 0 else 0
            }

            logger.info(f"📊 Database stats: {stats['total_restaurants']} restaurants, {stats['coordinate_coverage']['coverage_percentage']}% have coordinates")
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_restaurants': 0,
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

def cache_search_results(query: str, results: Dict[str, Any]) -> bool:
    """Cache search results"""
    return get_database().cache_search_results(query, results)

def get_cached_results(query: str) -> Optional[Dict[str, Any]]:
    """Get cached search results"""
    return get_database().get_cached_results(query)

# ============ DOMAIN INTELLIGENCE COMPATIBILITY STUBS ============
# These maintain compatibility with existing code that expects domain intelligence
# in database.py but redirect to the dedicated domain intelligence module

def save_domain_intelligence(domain: str, intelligence_data: Dict[str, Any]) -> bool:
    """Save domain intelligence data - redirects to domain intelligence module"""
    try:
        from utils.database_domain_intelligence import save_domain_intelligence_from_scraper
        return save_domain_intelligence_from_scraper(domain, intelligence_data)
    except ImportError:
        logger.warning("Domain intelligence module not available")
        return False

def get_domain_intelligence(domain: str) -> Optional[Dict[str, Any]]:
    """Get domain intelligence data - redirects to domain intelligence module"""
    try:
        from utils.database_domain_intelligence import get_domain_intelligence_manager
        manager = get_domain_intelligence_manager()
        return manager.get_domain_intelligence(domain)
    except ImportError:
        logger.warning("Domain intelligence module not available")
        return None

def update_domain_success(domain: str, success: bool, restaurants_found: int = 0):
    """Update domain success metrics - redirects to domain intelligence module"""
    try:
        from utils.database_domain_intelligence import get_domain_intelligence_manager
        manager = get_domain_intelligence_manager()
        # Construct fake URL for compatibility
        fake_url = f"https://{domain}/"
        manager.save_scrape_result(fake_url, "enhanced_http", success, restaurants_found=restaurants_found)
    except ImportError:
        logger.warning("Domain intelligence module not available")

def get_trusted_domains(min_confidence: float = None) -> List[str]:
    """Get list of trusted domains - redirects to domain intelligence module"""
    try:
        from utils.database_domain_intelligence import get_domain_intelligence_manager
        manager = get_domain_intelligence_manager()
        return manager.get_trusted_domains(min_confidence or 0.7)
    except ImportError:
        logger.warning("Domain intelligence module not available")
        return []

def load_all_domain_intelligence() -> List[Dict[str, Any]]:
    """Load all domain intelligence records - redirects to domain intelligence module"""
    try:
        from utils.database_domain_intelligence import get_domain_intelligence_manager
        manager = get_domain_intelligence_manager()
        return manager.load_all_domain_intelligence()
    except ImportError:
        logger.warning("Domain intelligence module not available")
        return []

# ============ BACKWARDS COMPATIBILITY FOR MAIN.PY ============

# For main.py to work without changes
def initialize_db(config):
    """Alias for initialize_database for main.py compatibility"""
    initialize_database(config)

def get_supabase_manager():
    """Alias for get_database for backwards compatibility"""
    return get_database()