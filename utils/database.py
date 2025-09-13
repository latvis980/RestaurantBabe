# utils/database.py - CLEANED: Direct Supabase interface with all type errors fixed
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from supabase import create_client, Client

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
        """Save restaurant with new simplified schema - SAFER coordinate approach"""
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
                        try:
                            lat, lng = float(coords[0]), float(coords[1])
                            update_data['latitude'] = lat
                            update_data['longitude'] = lng
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid coordinates in restaurant data: {coords}, error: {e}")

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
                        try:
                            lat, lng = float(coords[0]), float(coords[1])
                            insert_data['latitude'] = lat
                            insert_data['longitude'] = lng
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid coordinates in restaurant data: {coords}, error: {e}")

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
        """Update restaurant with address and coordinates - SAFER approach"""
        try:
            lat, lng = coordinates

            # Update both address and coordinates in one call
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

    # utils/database.py - Coordinate-based search using PostGIS spatial functions

    def get_restaurants_by_coordinates(self, center: Tuple[float, float], radius_km: float, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get restaurants within radius of coordinates using PostGIS spatial search

        NOTE: This method should ONLY be called by location/database_search.py
        All location-based logic should remain in the /location/ folder.

        Args:
            center: Tuple of (latitude, longitude) for center point
            radius_km: Search radius in kilometers
            limit: Maximum number of restaurants to return

        Returns:
            List of restaurants with distances, sorted by distance
        """
        try:
            center_lat, center_lng = center
            logger.info(f"📍 Searching restaurants within {radius_km}km of ({center_lat}, {center_lng})")

            # Use PostGIS function for efficient spatial search
            result = self.supabase.rpc('search_restaurants_by_coordinates', {
                'center_lat': center_lat,
                'center_lng': center_lng,
                'radius_km': radius_km,
                'result_limit': limit
            }).execute()

            restaurants = result.data or []
            logger.info(f"✅ PostGIS search found {len(restaurants)} restaurants within {radius_km}km")
            return restaurants

        except Exception as e:
            logger.error(f"❌ Error in PostGIS coordinate search: {e}")
            return []

    # ============ STATISTICS AND MONITORING ============

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring - FIXED VERSION"""
        try:
            stats = {}

            # Count restaurants - Fixed count method
            restaurants_result = self.supabase.table('restaurants').select('id').execute()
            stats['total_restaurants'] = len(restaurants_result.data) if restaurants_result.data else 0

            # Set domains to 0 since we don't have domain intelligence anymore
            stats['total_domains'] = 0

            # Get cities using a simple approach
            try:
                cities_result = self.supabase.table('restaurants')\
                    .select('city')\
                    .execute()

                # Count cities manually
                city_counts = {}
                for row in cities_result.data or []:
                    city = row.get('city', 'Unknown')
                    city_counts[city] = city_counts.get(city, 0) + 1

                # Sort by count and take top 10
                top_cities = sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                stats['top_cities'] = [{'city': city, 'count': count} for city, count in top_cities]

            except Exception as e:
                logger.warning(f"Could not get city stats: {e}")
                stats['top_cities'] = []

            # Coordinate coverage - Check both coordinate columns
            with_coords_result = self.supabase.table('restaurants')\
                .select('id')\
                .or_('coordinates.not.is.null,and(latitude.not.is.null,longitude.not.is.null)')\
                .execute()

            with_coords_count = len(with_coords_result.data) if with_coords_result.data else 0
            total_count = stats['total_restaurants']

            stats['coordinate_coverage'] = {
                'with_coordinates': with_coords_count,
                'without_coordinates': total_count - with_coords_count,
                'coverage_percentage': round((with_coords_count / total_count * 100), 1) if total_count > 0 else 0
            }

            logger.info(f"📊 Database stats: {stats['total_restaurants']} restaurants, {stats['coordinate_coverage']['coverage_percentage']}% have coordinates")
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
        """Geocode an address to get coordinates - SIMPLIFIED and SAFE"""
        try:
            if not self.geocoder or not address:
                logger.warning("Geocoder not available or empty address")
                return None

            # Simple, safe geocoding - remove timeout parameter to avoid type issues
            location = self.geocoder.geocode(address)

            # Handle the result safely
            if location:
                # Extract coordinates with explicit type checking
                lat = getattr(location, 'latitude', None)
                lng = getattr(location, 'longitude', None)

                if lat is not None and lng is not None:
                    try:
                        return (float(lat), float(lng))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid coordinate values: lat={lat}, lng={lng}")
                        return None

            logger.debug(f"No geocoding result for: {address}")
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

def cache_search_results(query: str, results: Dict[str, Any]) -> bool:
    """Cache search results"""
    return get_database().cache_search_results(query, results)

def get_cached_results(query: str) -> Optional[Dict[str, Any]]:
    """Get cached search results"""
    return get_database().get_cached_results(query)

# ============ CONVENIENCE FUNCTION FOR SOURCE QUALITY ============

def store_source_quality_data(destination: str, url: str, score: float) -> bool:
    """Store source quality data - convenience function"""
    if not destination or not url:
        logger.warning(f"Invalid parameters for source quality: destination='{destination}', url='{url}'")
        return False
    return get_database().store_source_quality(destination, url, score)


# ============ BACKWARDS COMPATIBILITY FOR MAIN.PY ============

# For main.py to work without changes
def initialize_db(config):
    """Alias for initialize_database for main.py compatibility"""
    initialize_database(config)

def get_supabase_manager():
    """Alias for get_database for backwards compatibility"""
    return get_database()