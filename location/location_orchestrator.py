# location/location_orchestrator.py
"""
Location Search Orchestrator - UPDATED with proper user choice flow

Business Logic:
1. User enters location (text or pin)
2. Database search
3. Database results analysis:
   a. if more than 0 results - send to user + offer choice (accept or request more)
   b. if 0 results - start google maps search directly
4. Return formatted results (location service has its own formatter)
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
        try:
            start_time = time.time()
            logger.info(f"🎯 Processing location query: '{query}'")

            # STEP 0: Get coordinates
            coordinates = self._get_coordinates(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates
            location_desc = location_data.description or f"GPS location ({lat:.4f}, {lng:.4f})"
            logger.info(f"📍 Coordinates: {lat:.4f}, {lng:.4f}")

            # STEP 1: Database proximity search
            logger.info("🗃️ Step 1: Database proximity search")
            nearby_restaurants = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=self.db_search_radius,
                extract_descriptions=True
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 2: AI-based analysis and filtering
            logger.info("🧠 Step 2: AI analysis and filtering")
            filter_result = self.filter_evaluator.filter_and_evaluate(
                restaurants=nearby_restaurants,
                query=query,
                location_description=location_desc
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # BUSINESS LOGIC IMPLEMENTATION:
            filtered_restaurants = filter_result.get("filtered_restaurants", [])

            if len(filtered_restaurants) > 0:
                # CASE 3a: More than 0 results - send to user with choice option
                logger.info(f"✅ Found {len(filtered_restaurants)} database matches - sending with choice")

                # Add distance info and format results
                restaurants_with_distance = self._add_distance_info(filtered_restaurants, coordinates)

                formatted_results = self.formatter.format_database_results(
                    restaurants=restaurants_with_distance,
                    query=query,
                    location_description=location_desc,
                    offer_more_search=True  # Always offer choice
                )

                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "results": formatted_results,
                    "source": "database_with_choice",  # Key indicator for telegram bot
                    "processing_time": processing_time,
                    "restaurant_count": len(filtered_restaurants),
                    "coordinates": coordinates,
                    "location_description": location_desc,
                    "needs_user_response": True  # Telegram bot will show choice buttons
                }

            else:
                # CASE 3b: 0 results - start Google Maps search directly
                logger.info("🔍 No database results found - launching Google Maps search")
                return await self._google_maps_search_flow(
                    coordinates, query, location_desc, start_time, cancel_check_fn
                )

        except Exception as e:
            logger.error(f"❌ Error in location query processing: {e}")
            return self._create_error_response(f"Search error: {str(e)}")

    # ============ GOOGLE MAPS SEARCH FLOW ============

    async def _google_maps_search_flow(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        location_desc: str,
        start_time: float,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Google Maps search flow for when database has no results OR user wants more
        """
        try:
            logger.info("🗺️ Step 3: Google Maps venue search")

            # Search Google Maps for venues
            venues = await self._search_google_maps_venues(
                coordinates, query, cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not venues:
                return self._create_error_response("No restaurants found in this area")

            logger.info(f"🏪 Found {len(venues)} venues from Google Maps")

            # Steps 4 & 5: Media verification and filtering (if enabled)
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
                "results": formatted_results.get("venues", []),
                "source": "google_maps_only",
                "processing_time": processing_time,
                "restaurant_count": len(final_venues),
                "coordinates": coordinates,
                "location_description": location_desc
            }

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search flow: {e}")
            return self._create_error_response(f"Google Maps search failed: {str(e)}")

    # ============ HELPER METHODS ============

    def _get_coordinates(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Extract or geocode coordinates from location data"""
        try:
            # If we have GPS coordinates, use them directly
            if location_data.latitude and location_data.longitude:
                return (location_data.latitude, location_data.longitude)

            # If we have a description, geocode it
            if location_data.description:
                logger.info(f"🌍 Geocoding location: {location_data.description}")

                from location.location_utils import LocationUtils
                location_utils = LocationUtils(self.config)
                coordinates = location_utils.geocode_location_description(location_data.description)

                if coordinates:
                    logger.info(f"✅ Geocoded to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                    return coordinates
                else:
                    logger.warning(f"❌ Failed to geocode: {location_data.description}")

            return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None

    def _add_distance_info(self, restaurants: List[Dict[str, Any]], coordinates: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Add distance information to restaurants"""
        try:
            from location.location_utils import LocationUtils

            for restaurant in restaurants:
                if restaurant.get('latitude') and restaurant.get('longitude'):
                    distance = LocationUtils.calculate_distance(
                        coordinates[0], coordinates[1],
                        restaurant['latitude'], restaurant['longitude']
                    )
                    restaurant['distance_km'] = round(distance, 2)
                else:
                    restaurant['distance_km'] = None

            # Sort by distance
            restaurants.sort(key=lambda x: x.get('distance_km') or float('inf'))
            return restaurants

        except Exception as e:
            logger.error(f"❌ Error adding distance info: {e}")
            return restaurants

    async def _search_google_maps_venues(
        self, 
        coordinates: Tuple[float, float], 
        query: str,
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues"""
        try:
            venues = self.google_maps_agent.search_nearby_venues(
                coordinates=coordinates,
                query=query,
                radius_meters=self.db_search_radius * 1000  # Convert km to meters
            )

            # Filter and sort venues
            filtered_venues = []
            for venue in venues:
                if cancel_check_fn and cancel_check_fn():
                    break

                # Basic filtering
                if venue.rating and venue.rating >= 3.5:  # Minimum rating
                    filtered_venues.append(venue)

            # Sort by rating and review count
            filtered_venues.sort(key=lambda v: (v.rating or 0, v.user_ratings_total or 0), reverse=True)

            return filtered_venues[:self.max_venues_to_verify]

        except Exception as e:
            logger.error(f"❌ Error searching Google Maps venues: {e}")
            return []

    async def _verify_and_filter_venues(
        self, 
        venues: List[VenueResult], 
        query: str,
        coordinates: Tuple[float, float],
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Verify and filter venues using media verification"""
        try:
            verified_venues = []

            for venue in venues:
                if cancel_check_fn and cancel_check_fn():
                    break

                # Media verification
                verification_result = await self.media_verifier.verify_venue(
                    venue=venue,
                    query=query,
                    coordinates=coordinates
                )

                if verification_result.get('is_relevant', False):
                    verified_venues.append(venue)

                # Respect rate limits
                await asyncio.sleep(0.5)

            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error in venue verification: {e}")
            # Fallback: return unverified venues
            return venues

    # ============ RESPONSE HELPERS ============

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "source": "error"
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create standardized cancellation response"""
        return {
            "success": False,
            "cancelled": True,
            "error": "Search was cancelled",
            "results": [],
            "source": "cancelled"
        }