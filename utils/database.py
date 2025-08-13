# utils/database.py - SIMPLIFIED: Direct Supabase interface (no wrapper)
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
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

    # Also add this convenience function at the bottom of utils/database.py

    def store_source_quality_data(destination: str, url: str, score: float) -> bool:
        """Store source quality data - convenience function"""
        return get_database().store_source_quality(destination, url, score)
    
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

    def _geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to get coordinates using the same approach as the older implementation"""
        try:
            if not self.geocoder:
                return None

            location = self.geocoder.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
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

            # First update the address
            update_data = {
                'address': address,
                'last_updated': datetime.now().isoformat()
            }

            self.supabase.table('restaurants')\
                .update(update_data)\
                .eq('id', restaurant_id)\
                .execute()

            # Use RPC function for coordinates (safer than raw SQL)
            try:
                # Try using RPC first (if the function exists in your Supabase)
                self.supabase.rpc('update_restaurant_coordinates', {
                    'restaurant_id': restaurant_id,
                    'lat': lat,
                    'lng': lng
                }).execute()
                logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id} with coords ({lat}, {lng}) via RPC")
            except Exception as rpc_error:
                # Fallback: use the latitude/longitude columns approach like your old implementation
                logger.warning(f"RPC failed, using fallback coordinate storage: {rpc_error}")

                # Update with separate lat/lng columns (like your old working implementation)
                coord_update = {
                    'latitude': lat,
                    'longitude': lng
                }

                self.supabase.table('restaurants')\
                    .update(coord_update)\
                    .eq('id', restaurant_id)\
                    .execute()

                logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id} with coords ({lat}, {lng}) via lat/lng columns")

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
            
    def get_restaurants_by_coordinates(self, center: Tuple[float, float], radius_km: float, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get restaurants within radius of coordinates using PostGIS or fallback method

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

            # Try PostGIS search first (now with the correct RPC function)
            try:
                result = self.supabase.rpc('search_restaurants_by_coordinates', {
                    'center_lat': center_lat,
                    'center_lng': center_lng,
                    'radius_km': radius_km,
                    'result_limit': limit
                }).execute()

                restaurants = result.data or []
                logger.info(f"✅ PostGIS search found {len(restaurants)} restaurants")
                return restaurants

            except Exception as postgis_error:
                logger.warning(f"PostGIS search failed: {postgis_error}")
                logger.info("🔄 Falling back to manual distance calculation...")

                # Fallback: Get all restaurants with coordinates and filter manually
                result = self.supabase.table('restaurants')\
                    .select('id, name, address, city, country, latitude, longitude, place_id, cuisine_tags, mention_count')\
                    .not_.is_('latitude', 'null')\
                    .not_.is_('longitude', 'null')\
                    .neq('latitude', 0)\
                    .neq('longitude', 0)\
                    .execute()

                all_restaurants = result.data or []

                # Filter by distance using LocationUtils
                from location.location_utils import LocationUtils
                restaurants_with_distance = []

                for restaurant in all_restaurants:
                    try:
                        rest_lat = float(restaurant['latitude'])
                        rest_lng = float(restaurant['longitude'])

                        # Calculate distance
                        distance_km = LocationUtils.calculate_distance(
                            (center_lat, center_lng), (rest_lat, rest_lng)
                        )

                        # Only include if within radius
                        if distance_km <= radius_km:
                            restaurant['distance_km'] = round(distance_km, 2)
                            restaurants_with_distance.append(restaurant)

                    except (ValueError, TypeError) as e:
                        logger.debug(f"Invalid coordinates for restaurant {restaurant.get('name', 'Unknown')}: {e}")
                        continue

                # Sort by distance and apply limit
                restaurants_with_distance.sort(key=lambda x: x['distance_km'])
                results = restaurants_with_distance[:limit]

                logger.info(f"✅ Fallback search found {len(results)} restaurants")
                return results

        except Exception as e:
            logger.error(f"❌ Error in coordinate search: {e}")
            return []
        
            
    # ============ STATISTICS AND MONITORING ============

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring - FIXED VERSION"""
        try:
            stats = {}

            # Count restaurants
            restaurants_count = self.supabase.table('restaurants').select('id', count='exact').execute()
            stats['total_restaurants'] = restaurants_count.count

            # Count domains
            domains_count = self.supabase.table('domain_intelligence').select('domain', count='exact').execute()
            stats['total_domains'] = domains_count.count

            # FIXED: Remove group_by which doesn't exist in current Supabase client
            # Get cities using a simple approach
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
                .not_.is_('coordinates', 'null')\
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