# location/database_service.py
"""
Location-based database service for coordinate searches

This service handles coordinate-based queries to the main app's database.
Used by location orchestrator for proximity searches.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from utils.database import get_database

logger = logging.getLogger(__name__)

class LocationDatabaseService:
    """
    Service for location-based database operations
    """

    def __init__(self, config):
        self.config = config
        self.default_radius_km = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        logger.info("✅ Location Database Service initialized")

    def get_restaurants_by_proximity(
        self, 
        coordinates: Tuple[float, float], 
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """
        Get restaurants within proximity of coordinates

        Args:
            coordinates: (latitude, longitude) tuple
            radius_km: Search radius in kilometers

        Returns:
            List of restaurant dictionaries with distance info
        """
        try:
            latitude, longitude = coordinates
            logger.info(f"🗃️ Searching database for restaurants within {radius_km}km of {latitude:.4f}, {longitude:.4f}")

            # Use your existing database interface
            from utils.database import get_database
            db = get_database()

            # Query restaurants with coordinates within the radius
            # This assumes you have a method to get restaurants by proximity
            # Adjust the method name based on your actual database interface

            try:
                # Try the method if it exists
                restaurants = db.get_restaurants_by_coordinates(
                    center=(latitude, longitude),
                    radius_km=radius_km,
                    limit=50
                )
            except Exception as e:
                logger.error(f"Error calling get_restaurants_by_coordinates: {e}")
                restaurants = []

            logger.info(f"📊 Found {len(restaurants)} restaurants within {radius_km}km")
            return restaurants

        except Exception as e:
            logger.error(f"❌ Error getting restaurants by proximity: {e}")
            return []

    def _filter_by_distance(
        self, 
        restaurants: List[Dict[str, Any]], 
        center: Tuple[float, float], 
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """Filter restaurants by distance from center point"""
        try:
            from utils.location_utils import LocationUtils

            filtered = []
            center_lat, center_lng = center

            for restaurant in restaurants:
                lat = restaurant.get('latitude')
                lng = restaurant.get('longitude')

                if lat and lng:
                    distance = LocationUtils.calculate_distance((center_lat, center_lng), (lat, lng))
                    if distance <= radius_km:
                        restaurant['distance_km'] = distance
                        filtered.append(restaurant)

            # Sort by distance
            filtered.sort(key=lambda x: x.get('distance_km', float('inf')))
            return filtered

        except Exception as e:
            logger.error(f"❌ Error filtering by distance: {e}")
            return restaurants

    def get_restaurants_by_coordinates(
        self, 
        center: Tuple[float, float], 
        radius_km: Optional[float] = None,  # Add Optional here
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get restaurants within radius of coordinates using PostGIS

        Args:
            center: Tuple of (latitude, longitude) for center point
            radius_km: Search radius in kilometers (default from config)
            limit: Maximum number of restaurants to return

        Returns:
            List of restaurants with distances, sorted by distance
        """
        try:
            if radius_km is None:
                radius_km = self.default_radius_km

            center_lat, center_lng = center
            logger.info(f"📍 Searching restaurants within {radius_km}km of ({center_lat}, {center_lng})")

            # Get database connection
            db = get_database()

            # Try PostGIS search first
            try:
                # Use Supabase RPC function for PostGIS search
                result = db.supabase.rpc('search_restaurants_by_coordinates', {
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

                return self._fallback_coordinate_search(center, radius_km, limit)

        except Exception as e:
            logger.error(f"❌ Error in coordinate search: {e}")
            return []

    def _fallback_coordinate_search(
        self, 
        center: Tuple[float, float], 
        radius_km: Optional[float], 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fallback method using manual distance calculation
        """
        try:
            if radius_km is None:
                radius_km = self.default_radius_km

            center_lat, center_lng = center
            db = get_database()

            # Get all restaurants with coordinates
            result = db.supabase.table('restaurants')\
                .select('id, name, address, city, country, latitude, longitude, place_id, cuisine_tags, mention_count')\
                .not_.is_('latitude', 'null')\
                .not_.is_('longitude', 'null')\
                .neq('latitude', 0)\
                .neq('longitude', 0)\
                .execute()

            restaurants = result.data or []

            # Calculate distances using LocationUtils
            from utils.location_utils import LocationUtils
            restaurants_with_distance = []

            for restaurant in restaurants:
                try:
                    rest_lat = float(restaurant['latitude'])
                    rest_lng = float(restaurant['longitude'])

                    # Use LocationUtils to calculate distance
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

            # Sort by distance
            restaurants_with_distance.sort(key=lambda x: x['distance_km'])

            # Apply limit
            results = restaurants_with_distance[:limit]

            logger.info(f"✅ Fallback search found {len(results)} restaurants")
            return results

        except Exception as e:
            logger.error(f"❌ Error in fallback coordinate search: {e}")
            return []

    def search_restaurants_nearby(
        self, 
        center: Tuple[float, float], 
        radius_km: Optional[float] = None,  # Add Optional here
        limit: int = 20
    ) -> List[str]:
        """
        Simple interface returning restaurant names with distances

        Args:
            center: Tuple of (latitude, longitude) 
            radius_km: Search radius in kilometers
            limit: Maximum number of restaurants

        Returns:
            List of strings: "Restaurant Name (1.2km)"
        """
        try:
            restaurants = self.get_restaurants_by_coordinates(center, radius_km, limit)

            # Format as simple name + distance strings
            results = []
            for restaurant in restaurants:
                name = restaurant.get('name', 'Unknown')
                distance = restaurant.get('distance_km', 0)
                results.append(f"{name} ({distance}km)")

            return results

        except Exception as e:
            logger.error(f"❌ Error in nearby search: {e}")
            return []

    def get_database_summary_for_location(
        self, 
        center: Tuple[float, float], 
        radius_km: Optional[float] = None  # Add Optional here
    ) -> Dict[str, Any]:
        """
        Get summary statistics for restaurants near coordinates

        Returns:
            Dict with count, cuisine types, distance stats
        """
        try:
            restaurants = self.get_restaurants_by_coordinates(center, radius_km, limit=100)

            if not restaurants:
                return {
                    'total_count': 0,
                    'message': 'No restaurants found in database for this location'
                }

            # Calculate statistics
            distances = [r.get('distance_km', 0) for r in restaurants]
            cuisine_tags = []

            for restaurant in restaurants:
                tags = restaurant.get('cuisine_tags', [])
                if tags:
                    cuisine_tags.extend(tags)

            # Count cuisine types
            cuisine_counts = {}
            for cuisine in cuisine_tags:
                cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

            return {
                'total_count': len(restaurants),
                'closest_distance_km': min(distances) if distances else 0,
                'farthest_distance_km': max(distances) if distances else 0,
                'avg_distance_km': round(sum(distances) / len(distances), 2) if distances else 0,
                'top_cuisines': sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'sample_restaurants': [
                    f"{r.get('name', 'Unknown')} ({r.get('distance_km', 0)}km)"
                    for r in restaurants[:3]
                ]
            }

        except Exception as e:
            logger.error(f"❌ Error getting location summary: {e}")
            return {'total_count': 0, 'error': str(e)}