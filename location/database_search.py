# location/database_search.py
"""
Database Search for Location-Based Queries - STEP 1

Renamed from database_service.py for clean architecture.
Handles coordinate-based queries to the main app's database with PostGIS.

This implements Step 1 of the location search flow:
- Proximity search using coordinates (PostGIS, 3 km radius)
- Extract all database results by proximity 
- Extract cuisine_tags, descriptions, and sources
- Compile results for further analysis in Step 2
"""

import logging
import math
import json
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

        Extract all database results by proximity, then extract cuisine_tags,
        descriptions, and sources for further analysis in Step 2.

        Args:
            coordinates: (latitude, longitude) tuple
            radius_km: Search radius in kilometers (defaults to config setting)
            extract_descriptions: Whether to include full description data

        Returns:
            List of restaurant dictionaries with distance info, descriptions, and sources
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

            # Add cuisine tags, description, and sources extraction for Step 2
            if extract_descriptions:
                restaurants = self._extract_cuisine_descriptions_and_sources(restaurants)

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
        """Fallback: Manual distance calculation for all restaurants with coordinates.

        The Supabase query builder must be executed with ``.execute()`` rather than
        being invoked directly. This retrieves the raw restaurant data before the
        manual distance calculations are applied.
        """
        try:
            latitude, longitude = coordinates

            # Get all restaurants with coordinates from the database. Calling
            # ``.execute()`` runs the query builder and returns the result object.
            result = db.supabase.table('restaurants')\
                .select('*')\
                .not_('latitude', 'is', None)\
                .not_('longitude', 'is', None)\
                .limit(200)\
                .execute() 

            all_restaurants = result.data or []
            logger.info(f"Retrieved {len(all_restaurants)} restaurants with coordinates for manual filtering")

            # Filter by distance
            nearby_restaurants = []

            for restaurant in all_restaurants:
                try:
                    rest_lat = float(restaurant.get('latitude', 0))
                    rest_lon = float(restaurant.get('longitude', 0))

                    if rest_lat == 0 or rest_lon == 0:
                        continue

                    # Calculate distance using haversine formula
                    distance = LocationUtils.calculate_distance(
                        (latitude, longitude),
                        (rest_lat, rest_lon)
                    )

                    if distance <= radius_km:
                        restaurant['distance_km'] = round(distance, 2)
                        nearby_restaurants.append(restaurant)

                except (ValueError, TypeError) as e:
                    logger.debug(f"Invalid coordinates for restaurant {restaurant.get('id')}: {e}")
                    continue

            # Sort by distance
            nearby_restaurants.sort(key=lambda x: x.get('distance_km', float('inf')))

            logger.info(f"Manual distance filtering: {len(nearby_restaurants)} restaurants within {radius_km}km")
            return nearby_restaurants

        except Exception as e:
            logger.error(f"❌ Error in manual distance calculation: {e}")
            return []

    def _extract_cuisine_descriptions_and_sources(self, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced: Extract cuisine tags, descriptions, AND sources for Step 2 processing

        This ensures all necessary data is available for AI filtering and description editing
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

                # Extract and parse sources - convert from database text to domain list
                sources = self._parse_sources_field(restaurant)
                enriched_restaurant['sources'] = sources
                enriched_restaurant['sources_domains'] = self._extract_domains_from_sources(sources)

                # Add summary info for Step 2 filtering
                enriched_restaurant['has_description'] = bool(description and len(description) > 10)
                enriched_restaurant['cuisine_count'] = len(cuisine_tags)
                enriched_restaurant['sources_count'] = len(sources)

                enriched_restaurants.append(enriched_restaurant)

            logger.info(f"📋 Extracted cuisine, description, and sources data for {len(enriched_restaurants)} restaurants")

            # Log summary statistics for Step 2
            with_descriptions = sum(1 for r in enriched_restaurants if r['has_description'])
            with_sources = sum(1 for r in enriched_restaurants if r['sources_count'] > 0)
            total_cuisines = sum(r['cuisine_count'] for r in enriched_restaurants)

            logger.info(f"📊 Step 1 Summary: {with_descriptions}/{len(enriched_restaurants)} have descriptions")
            logger.info(f"📊 Step 1 Summary: {with_sources}/{len(enriched_restaurants)} have sources")
            logger.info(f"📊 Step 1 Summary: {total_cuisines} total cuisine tags")

            return enriched_restaurants

        except Exception as e:
            logger.error(f"❌ Error extracting cuisine, descriptions, and sources: {e}")
            return restaurants

    def _parse_sources_field(self, restaurant: Dict[str, Any]) -> List[str]:
        """
        Parse the sources field from database TEXT to actual list

        The sources column in Supabase is stored as TEXT but contains JSON-like strings
        We need to convert these to actual Python lists
        """
        try:
            sources_raw = restaurant.get('sources', [])

            # If it's already a list, return as-is
            if isinstance(sources_raw, list):
                return sources_raw

            # If it's a string that looks like a JSON array, parse it
            if isinstance(sources_raw, str) and sources_raw.strip():
                try:
                    # Try JSON parsing first
                    sources_list = json.loads(sources_raw)
                    if isinstance(sources_list, list):
                        return sources_list
                    else:
                        # If JSON parsing returns non-list, wrap in list
                        return [str(sources_list)]
                except json.JSONDecodeError:
                    try:
                        # Try ast.literal_eval as fallback
                        import ast
                        sources_list = ast.literal_eval(sources_raw)
                        if isinstance(sources_list, list):
                            return sources_list
                        else:
                            return [str(sources_list)]
                    except (ValueError, SyntaxError):
                        # If all parsing fails, treat as single source
                        return [sources_raw]
            else:
                # Empty or None sources
                return []

        except Exception as e:
            logger.error(f"Error parsing sources for restaurant {restaurant.get('name', 'Unknown')}: {e}")
            return []

    def _extract_domains_from_sources(self, sources: List[str]) -> List[str]:
        """
        Extract just the domain names from full URLs in sources

        Args:
            sources: List of full URLs

        Returns:
            List of domain names (e.g., ['timeout.com', 'eater.com'])
        """
        try:
            domains = []

            for source in sources:
                if not source or not isinstance(source, str):
                    continue

                # Extract domain from URL
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(source)
                    domain = parsed_url.netloc.lower()

                    # Remove 'www.' prefix if present
                    if domain.startswith('www.'):
                        domain = domain[4:]

                    if domain and domain not in domains:
                        domains.append(domain)

                except Exception as e:
                    logger.debug(f"Could not parse URL {source}: {e}")
                    continue

            return domains

        except Exception as e:
            logger.error(f"Error extracting domains from sources: {e}")
            return []

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