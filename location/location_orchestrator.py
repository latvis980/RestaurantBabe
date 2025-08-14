# location/location_orchestrator.py
"""
Location Search Orchestrator - FIXED VERSION

Business Logic:
1. User enters location (text or pin)
2. Database search
3. Database results analysis:
   a. if more than 0 results - send to user + offer choice (accept or request more)
   b. if 0 results - start google maps search directly
4. Return formatted results (location service has its own formatter)

FIXES:
- Added missing formatted_message key to all returns
- Improved coordinate handling and geocoding fallback
- Fixed database result formatting flow
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import json

# UPDATED IMPORTS - Using new file structure
from location.location_utils import LocationUtils, LocationPoint
from location.telegram_location_handler import LocationData
from location.google_maps_search import GoogleMapsSearchAgent, VenueResult
from location.media_verification import MediaVerificationAgent
from location.database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_telegram_formatter import LocationTelegramFormatter

# Import AI components for restaurant filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Location search orchestrator with proper user choice flow
    Implements the correct business logic as specified
    """

    def __init__(self, config):
        self.config = config

        # Initialize location-specific services
        self.database_service = LocationDatabaseService(config)  # Step 1
        self.filter_evaluator = LocationFilterEvaluator(config)  # Step 2
        self.google_maps_agent = GoogleMapsSearchAgent(config)   # Step 3
        self.media_verifier = MediaVerificationAgent(config)     # Step 4 & 5
        self.formatter = LocationTelegramFormatter(config)       # Unified formatter

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = getattr(config, 'MIN_DB_MATCHES_REQUIRED', 3)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Location Orchestrator initialized with user choice flow")

    # ============ MAIN PROCESSING METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query with CORRECT business logic:

        1. User enters location (text or pin) - already handled
        2. Database search
        3. Database results analysis:
           a. if more than 0 results - send to user + offer choice
           b. if 0 results - start google maps search directly
        4. Return formatted results
        """
        start_time = time.time()

        try:
            # Get coordinates from location data
            coordinates = self._get_coordinates(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            location_desc = location_data.description or f"{coordinates[0]:.4f}, {coordinates[1]:.4f}"

            logger.info(f"🎯 Processing location query: '{query}' at {location_desc}")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # Step 1: Database search
            logger.info("🗄️ Step 1: Database search")
            db_results = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=self.db_search_radius,
                extract_descriptions=True
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # Step 2: Filter and evaluate database results (MISSING STEP - NOW ADDED)
            db_restaurant_count = len(db_results)

            if db_restaurant_count > 0:
                logger.info(f"🔍 Step 2: Filtering {db_restaurant_count} database results by query relevance")

                # Add distance information first
                restaurants_with_distance = self._add_distance_info(
                    db_results, coordinates
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                # STEP 2: Filter results using AI to match user query
                filter_result = self.filter_evaluator.filter_and_evaluate(
                    restaurants=restaurants_with_distance,
                    query=query,
                    location_description=location_desc
                )

                filtered_restaurants = filter_result.get("filtered_restaurants", [])
                filtered_count = len(filtered_restaurants)

                logger.info(f"🎯 Step 2 Complete: {filtered_count}/{db_restaurant_count} restaurants match query")

                # Step 3: Database results analysis (CORRECTED)
                if filtered_count > 0:
                    logger.info(f"✅ Found {filtered_count} relevant database results - sending to user with choice")

                    # Format filtered database results
                    formatted_results = self.formatter.format_database_results(
                        restaurants=filtered_restaurants,
                        query=query,
                        location_description=location_desc,
                        offer_more_search=True
                    )

                    processing_time = time.time() - start_time

                    return {
                        "success": True,
                        "results": filtered_restaurants,
                        "source": "database_with_choice",
                        "processing_time": processing_time,
                        "restaurant_count": filtered_count,
                        "total_found": db_restaurant_count,
                        "coordinates": coordinates,
                        "location_description": location_desc,
                        "offer_more_results": True,
                        "filter_reasoning": filter_result.get("reasoning", ""),
                        # FIX: Add the missing formatted_message key
                        "formatted_message": formatted_results.get("message", f"Found {filtered_count} relevant restaurants from my notes!")
                    }
                else:
                    logger.info(f"📍 No relevant matches in {db_restaurant_count} database results - starting Google Maps search")
                    return await self._search_google_maps_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

            else:
                logger.info("📍 No database results found - starting Google Maps search directly")
                return await self._search_google_maps_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

        except Exception as e:
            logger.error(f"❌ Error in location query processing: {e}")
            return self._create_error_response(f"Search failed: {str(e)}")

    # ============ GOOGLE MAPS SEARCH FLOW ============

    async def _search_google_maps_flow(
        self, 
        query: str, 
        coordinates: Tuple[float, float], 
        location_desc: str,
        cancel_check_fn=None,
        start_time=None
    ) -> Dict[str, Any]:
        """Google Maps search flow when database has no results"""
        if start_time is None:
            start_time = time.time()

        try:
            # Step 3: Google Maps venue search
            logger.info("🗺️ Step 3: Google Maps venue search")
            venues = await self._search_google_maps_venues(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not venues:
                return self._create_error_response("No restaurants found in Google Maps")

            # Step 4 & 5: Media verification (optional)
            if hasattr(self.config, 'ENABLE_MEDIA_VERIFICATION') and self.config.ENABLE_MEDIA_VERIFICATION:
                logger.info("📸 Steps 4 & 5: Media verification and filtering")

                verified_venues = await self._verify_and_filter_venues(
                    venues, query, coordinates, cancel_check_fn
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                final_venues = verified_venues
            else:
                # Skip media verification - use venues directly
                final_venues = venues[:self.max_venues_to_verify]

            if not final_venues:
                return self._create_error_response("No suitable restaurants found after verification")

            # Format results using unified formatter
            formatted_results = self.formatter.format_google_maps_results(
                venues=final_venues,
                query=query,
                location_description=location_desc
            )

            processing_time = time.time() - start_time
            logger.info(f"✅ Google Maps search completed in {processing_time:.1f}s")

            return {
                "success": True,
                "results": final_venues,
                "source": "google_maps_only",
                "processing_time": processing_time,
                "restaurant_count": len(final_venues),
                "coordinates": coordinates,
                "location_description": location_desc,
                # FIX: Add the missing formatted_message key
                "formatted_message": formatted_results.get("message", f"Found {len(final_venues)} restaurants via Google Maps!")
            }

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search flow: {e}")
            return self._create_error_response(f"Google Maps search failed: {str(e)}")

    # ============ HELPER METHODS ============

    def _get_coordinates(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Extract or geocode coordinates from location data - IMPROVED GEOCODING"""
        try:
            # If we have GPS coordinates, use them directly
            if location_data.latitude and location_data.longitude:
                return (location_data.latitude, location_data.longitude)

            # If we have a description, try to geocode it
            if location_data.description:
                logger.info(f"🌍 Geocoding location: {location_data.description}")

                # FIX: Improve geocoding with better Google Maps queries
                try:
                    if hasattr(self.config, 'GOOGLE_MAPS_API_KEY') and self.config.GOOGLE_MAPS_API_KEY:
                        import googlemaps
                        gmaps = googlemaps.Client(key=self.config.GOOGLE_MAPS_API_KEY)

                        # FIX: Add location context for better geocoding
                        # If the query doesn't include a city/country, add context
                        description = location_data.description
                        if not any(city in description.lower() for city in ['lisbon', 'lisboa', 'portugal']):
                            # Add Lisbon context if it seems to be a local area
                            if any(area in description.lower() for area in ['cais', 'sodre', 'chiado', 'bairro', 'rua']):
                                description = f"{description}, Lisbon, Portugal"

                        geocode_result = gmaps.geocode(description)

                        if geocode_result:
                            location = geocode_result[0]['geometry']['location']
                            coordinates = (location['lat'], location['lng'])
                            logger.info(f"✅ Google Maps geocoded to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                            return coordinates
                        else:
                            logger.warning(f"❌ Google Maps geocoding returned no results for: {description}")
                except Exception as e:
                    logger.warning(f"Google Maps geocoding failed: {e}")

                # Fallback: Try database geocoding if available
                try:
                    from utils.database import get_database
                    db = get_database()

                    if hasattr(db, 'geocode_address'):
                        coordinates = db.geocode_address(location_data.description)
                        if coordinates:
                            logger.info(f"✅ Database geocoded to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                            return coordinates
                        else:
                            logger.warning(f"❌ Database geocoding failed: {location_data.description}")
                except Exception as e:
                    logger.warning(f"Database geocoding failed: {e}")

            logger.error("❌ Could not determine coordinates from location data")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None


    def _add_distance_info(self, restaurants: List[Dict[str, Any]], coordinates: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Add distance information to restaurant results"""
        try:
            from location.location_utils import LocationUtils

            for restaurant in restaurants:
                if restaurant.get('latitude') and restaurant.get('longitude'):
                    restaurant_coords = (float(restaurant['latitude']), float(restaurant['longitude']))
                    distance_km = LocationUtils.calculate_distance(coordinates, restaurant_coords)

                    restaurant['distance_km'] = distance_km
                    restaurant['distance_text'] = LocationUtils.format_distance(distance_km)
                else:
                    restaurant['distance_km'] = float('inf')
                    restaurant['distance_text'] = "Distance unknown"

            # Sort by distance
            restaurants.sort(key=lambda x: x.get('distance_km', float('inf')))
            return restaurants

        except Exception as e:
            logger.error(f"❌ Error adding distance info: {e}")
            return restaurants

    async def _search_google_maps_venues(self, coordinates: Tuple[float, float], query: str, cancel_check_fn=None) -> List[VenueResult]:
        """Search Google Maps for venues - FIX: Remove cancel_check_fn parameter"""
        try:
            # FIX: Don't pass cancel_check_fn to search_venues - it doesn't accept this parameter
            venues = await self.google_maps_agent.search_venues(
                coordinates=coordinates,
                query=query
                # Removed: cancel_check_fn=cancel_check_fn  # This parameter doesn't exist
            )

            # Handle cancellation manually if needed
            if cancel_check_fn and cancel_check_fn():
                return []

            return venues
        except Exception as e:
            logger.error(f"❌ Error searching Google Maps venues: {e}")
            return []

    async def _verify_and_filter_venues(self, venues: List[VenueResult], query: str, coordinates: Tuple[float, float], cancel_check_fn=None) -> List[VenueResult]:
        """Verify venues through media sources"""
        try:
            verified_venues = await self.media_verifier.verify_venues(
                venues=venues,
                query=query,
                coordinates=coordinates,
                cancel_check_fn=cancel_check_fn
            )
            return verified_venues
        except Exception as e:
            logger.error(f"❌ Error verifying venues: {e}")
            return venues  # Return unverified venues as fallback

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "restaurant_count": 0,
            "formatted_message": f"😔 {error_message}"
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create standardized cancellation response"""
        return {
            "success": False,
            "cancelled": True,
            "results": [],
            "restaurant_count": 0,
            "formatted_message": "Search was cancelled."
        }