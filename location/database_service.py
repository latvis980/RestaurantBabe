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

    def get_restaurants_by_coordinates(
        self, 
        center: Tuple[float, float], 
        radius_km: float = None, 
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
        radius_km: float, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fallback method using manual distance calculation
        """
        try:
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

            # Calculate distances using Haversine formula
            restaurants_with_distance = []

            for restaurant in restaurants:
                try:
                    rest_lat = float(restaurant['latitude'])
                    rest_lng = float(restaurant['longitude'])

                    # Haversine formula to calculate distance
                    distance_km = self._calculate_distance(
                        center_lat, center_lng, rest_lat, rest_lng
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

    def _calculate_distance(
        self, 
        lat1: float, lng1: float, 
        lat2: float, lng2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers

        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)

        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlng/2)**2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance_km = R * c

        return distance_km

    def search_restaurants_nearby(
        self, 
        center: Tuple[float, float], 
        radius_km: float = None, 
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
        radius_km: float = None
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