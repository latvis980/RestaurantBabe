# location/location_orchestrator.py
"""
Location Search Orchestrator - UPDATED with Media Verification

UPDATED: Fixed Google Maps flow with proper media verification message and implementation
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

    # ============ GOOGLE MAPS SEARCH FLOW - UPDATED ============

    async def _search_google_maps_flow(
        self, 
        query: str, 
        coordinates: Tuple[float, float], 
        location_desc: str,
        cancel_check_fn=None,
        start_time=None
    ) -> Dict[str, Any]:
        """Google Maps search flow with UPDATED media verification message"""
        if start_time is None:
            start_time = time.time()

        try:
            # Step 3: Google Maps venue search
            logger.info("🗺️ Step 3: Google Maps venue search")
            venues = await self._search_google_maps_venues(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not venues:
                return self._create_error_response("No restaurants found in the area")

            # UPDATED: Return intermediate message for media verification
            return {
                "success": True,
                "results": venues,
                "source": "google_maps_with_verification",
                "processing_time": time.time() - start_time,
                "restaurant_count": len(venues),
                "coordinates": coordinates,
                "location_description": location_desc,
                "requires_verification": True,
                # UPDATED MESSAGE: Don't mention Google Maps, just say "found some restaurants"
                "formatted_message": f"Found some restaurants in the vicinity, let me check what local media and international guides have to say about them.",
                "query": query,
                "venues_for_verification": venues
            }

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search flow: {e}")
            return self._create_error_response(f"Google Maps search failed: {str(e)}")

    async def complete_media_verification(
        self,
        venues: List[VenueResult],
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Complete the media verification flow (Steps 4 & 5)
        Called after the initial "checking media" message
        """
        start_time = time.time()

        try:
            logger.info("📸 Steps 4 & 5: Media verification and filtering")

            verified_venues = await self._verify_and_filter_venues(
                venues, query, coordinates, cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            final_venues = verified_venues

            if not final_venues:
                return self._create_error_response("No suitable restaurants found after verification")

            # Format results using unified formatter
            formatted_results = self.formatter.format_google_maps_results(
                venues=final_venues,
                query=query,
                location_description=location_desc
            )

            processing_time = time.time() - start_time
            logger.info(f"✅ Google Maps search with verification completed in {processing_time:.1f}s")

            return {
                "success": True,
                "results": final_venues,
                "source": "google_maps_verified",
                "processing_time": processing_time,
                "restaurant_count": len(final_venues),
                "coordinates": coordinates,
                "location_description": location_desc,
                "formatted_message": formatted_results.get("message", f"Found {len(final_venues)} restaurants after verification!")
            }

        except Exception as e:
            logger.error(f"❌ Error in media verification: {e}")
            return self._create_error_response(f"Media verification failed: {str(e)}")

    # ============ HELPER METHODS ============

    async def _search_google_maps_venues(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues"""
        try:
            venues = await self.google_maps_agent.search_venues(
                coordinates=coordinates,
                query=query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"🗺️ Found {len(venues)} venues from Google Maps")
            return venues

        except Exception as e:
            logger.error(f"❌ Error searching Google Maps: {e}")
            return []

    async def _verify_and_filter_venues(
        self, 
        venues: List[VenueResult], 
        query: str, 
        coordinates: Tuple[float, float],
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Verify venues through media sources and filter by reputation"""
        try:
            verified_venues = await self.media_verifier.verify_venues(
                venues=venues,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"📸 Media verification completed: {len(verified_venues)} venues verified")
            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error in venue verification: {e}")
            # Fallback: convert venues to dict format without verification
            return self._convert_venues_to_fallback(venues)

    def _convert_venues_to_fallback(self, venues: List[VenueResult]) -> List[Dict[str, Any]]:
        """Convert VenueResult objects to dictionary format (fallback)"""
        return [
            {
                'name': venue.name,
                'address': venue.address,
                'latitude': venue.latitude,
                'longitude': venue.longitude,
                'distance_km': venue.distance_km,
                'rating': venue.rating,
                'place_id': venue.place_id,
                'description': f"Restaurant in {self._extract_city_from_address(venue.address)}",
                'sources': [],
                'media_verified': False,
                'google_maps_url': venue.google_maps_url
            }
            for venue in venues[:self.max_venues_to_verify]
        ]

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from venue address"""
        try:
            if not address:
                return "Unknown"

            parts = [part.strip() for part in address.split(',')]
            if len(parts) >= 3:
                return parts[1]
            elif len(parts) == 2:
                return parts[0]
            else:
                return parts[0] if parts else "Unknown"
        except Exception:
            return "Unknown"

    def _get_coordinates(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Extract coordinates from location data"""
        try:
            if location_data.latitude and location_data.longitude:
                return (location_data.latitude, location_data.longitude)

            # Try geocoding if we have a description
            if location_data.description:
                logger.info(f"🌍 Geocoding location: {location_data.description}")
                coordinates = LocationUtils.geocode_location(location_data.description)
                if coordinates:
                    return coordinates

            return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None

    def _add_distance_info(self, restaurants: List[Dict[str, Any]], coordinates: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Add distance information to restaurants"""
        try:
            restaurants_with_distance = []

            for restaurant in restaurants:
                # Calculate distance if restaurant has coordinates
                restaurant_lat = restaurant.get('latitude')
                restaurant_lng = restaurant.get('longitude')

                if restaurant_lat and restaurant_lng:
                    distance_km = LocationUtils.calculate_distance(
                        coordinates[0], coordinates[1],
                        restaurant_lat, restaurant_lng
                    )
                    restaurant['distance_km'] = distance_km
                    restaurant['distance_text'] = LocationUtils.format_distance(distance_km)
                else:
                    restaurant['distance_km'] = None
                    restaurant['distance_text'] = "Distance unknown"

                restaurants_with_distance.append(restaurant)

            return restaurants_with_distance

        except Exception as e:
            logger.error(f"❌ Error adding distance info: {e}")
            return restaurants

    # ============ RESPONSE HELPERS ============

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": message,
            "results": [],
            "restaurant_count": 0,
            "formatted_message": f"😔 {message}"
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create standardized cancellation response"""
        return {
            "success": False,
            "cancelled": True,
            "results": [],
            "restaurant_count": 0,
            "formatted_message": "Search cancelled by user"
        }