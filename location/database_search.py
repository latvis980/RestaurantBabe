# location/database_search.py
"""
Database Search for Location-Based Queries - STEP 1

Renamed from database_service.py for clean architecture.
Handles coordinate-based queries to the main app's database with PostGIS.

This implements Step 1 of the location search flow:
- Proximity search using coordinates (PostGIS, 3 km radius)
- Extract all database results by proximity 
- Extract cuisine_tags and descriptions
- Compile results for further analysis in Step 2
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from utils.database import get_database
from location.location_utils import LocationUtils

logger = logging.getLogger(__name__)

class LocationDatabaseService:
    """
    Service for location-based database operations - STEP 1
    """

    def __init__(self, config):
        self.config = config
        self.default_radius_km = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        logger.info("✅ Location Database Search Service initialized (Step 1)")

    def search_by_proximity(
        self, 
        coordinates: Tuple[float, float], 
        radius_km: Optional[float] = None,
        extract_descriptions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        STEP 1: Database proximity search using coordinates (PostGIS, 3 km radius)

        Extract all database results by proximity, then extract cuisine_tags 
        and descriptions for further analysis in Step 2.

        Args:
            coordinates: (latitude, longitude) tuple
            radius_km: Search radius in kilometers (defaults to config setting)
            extract_descriptions: Whether to include full description data

        Returns:
            List of restaurant dictionaries with distance info and descriptions
        """
        try:
            if radius_km is None:
                radius_km = self.default_radius_km

            latitude, longitude = coordinates
            logger.info(f"🗃️ STEP 1: Database proximity search within {radius_km}km of {latitude:.4f}, {longitude:.4f}")

            # Use existing database interface
            db = get_database()

            # Try PostGIS-enabled proximity search first
            restaurants = self._search_with_postgis(db, coordinates, radius_km)

            # If PostGIS fails, fallback to manual distance calculation
            if not restaurants:
                logger.info("PostGIS search failed, falling back to manual distance calculation")
                restaurants = self._search_with_manual_distance(db, coordinates, radius_km)

            logger.info(f"📊 STEP 1 COMPLETE: Found {len(restaurants)} restaurants within {radius_km}km")

            # Add cuisine tags and description extraction for Step 2
            if extract_descriptions:
                restaurants = self._extract_cuisine_and_descriptions(restaurants)

            return restaurants

        except Exception as e:
            logger.error(f"❌ Error in Step 1 database proximity search: {e}")
            return []

    def _search_with_postgis(
        self, 
        db, 
        coordinates: Tuple[float, float], 
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """
        Search using PostGIS spatial functions (preferred method)
        """
        try:
            latitude, longitude = coordinates

            # Try PostGIS spatial query
            restaurants = db.get_restaurants_by_coordinates(
                center=(latitude, longitude),
                radius_km=radius_km,
                limit=50
            )

            logger.info(f"PostGIS search returned {len(restaurants)} restaurants")
            return restaurants

        except Exception as e:
            logger.warning(f"PostGIS search failed: {e}")
            return []

    def _search_with_manual_distance(
        self, 
        db, 
        coordinates: Tuple[float, float], 
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """
        Fallback: Manual distance calculation for all restaurants with coordinates
        FIX: Removed the erroneous .execute() call on SyncSelectRequestBuilder
        """
        try:
            latitude, longitude = coordinates

            # Get all restaurants with coordinates from the database
            # FIX: This was missing .execute() call - supabase query builder needs execute()
            result = db.supabase.table('restaurants')\
                .select('*')\
                .not_('latitude', 'is', None)\
                .not_('longitude', 'is', None)\
                .limit(200)\
                .execute()  # FIX: Added the missing .execute() call

            all_restaurants = result.data or []
            logger.info(f"Retrieved {len(all_restaurants)} restaurants with coordinates for manual filtering")

            # Filter by distance manually
            restaurants_with_distance = []
            center_lat, center_lng = coordinates

            for restaurant in all_restaurants:
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

            logger.info(f"Manual distance filtering found {len(restaurants_with_distance)} restaurants")
            return restaurants_with_distance

        except Exception as e:
            logger.error(f"❌ Error in manual distance search: {e}")
            return []

    def _extract_cuisine_and_descriptions(
        self, 
        restaurants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract cuisine_tags and descriptions for Step 2 analysis

        This compiles the data needed for AI filtering in Step 2.
        """
        try:
            enriched_restaurants = []

            for restaurant in restaurants:
                # Ensure we have the required fields for Step 2
                enriched_restaurant = restaurant.copy()

                # Extract cuisine tags (ensure it's a list)
                cuisine_tags = restaurant.get('cuisine_tags', [])
                if isinstance(cuisine_tags, str):
                    # Handle case where cuisine_tags might be a string
                    cuisine_tags = [tag.strip() for tag in cuisine_tags.split(',') if tag.strip()]

                enriched_restaurant['cuisine_tags'] = cuisine_tags

                # Ensure description fields are available
                description = restaurant.get('raw_description', '') or restaurant.get('description', '')
                enriched_restaurant['description'] = description
                enriched_restaurant['raw_description'] = description

                # Add summary info for Step 2 filtering
                enriched_restaurant['has_description'] = bool(description and len(description) > 10)
                enriched_restaurant['cuisine_count'] = len(cuisine_tags)

                enriched_restaurants.append(enriched_restaurant)

            logger.info(f"📋 Extracted cuisine and description data for {len(enriched_restaurants)} restaurants")

            # Log summary statistics for Step 2
            with_descriptions = sum(1 for r in enriched_restaurants if r['has_description'])
            total_cuisines = sum(r['cuisine_count'] for r in enriched_restaurants)

            logger.info(f"📊 Step 1 Summary: {with_descriptions}/{len(enriched_restaurants)} have descriptions, {total_cuisines} total cuisine tags")

            return enriched_restaurants

        except Exception as e:
            logger.error(f"❌ Error extracting cuisine and descriptions: {e}")
            return restaurants

    def get_restaurants_by_proximity(
        self, 
        coordinates: Tuple[float, float], 
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """
        BACKWARD COMPATIBILITY: Maintain the old method name

        This ensures existing code still works while we transition.
        """
        return self.search_by_proximity(coordinates, radius_km, extract_descriptions=True)

    def get_database_summary_for_location(
        self, 
        coordinates: Tuple[float, float], 
        radius_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for restaurants near coordinates

        Returns:
            Dict with count, cuisine types, distance stats for debugging
        """
        try:
            restaurants = self.search_by_proximity(coordinates, radius_km, extract_descriptions=False)

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