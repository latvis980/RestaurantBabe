# location/location_orchestrator.py
"""
Location Search Orchestrator - COMPLETE CLEAN VERSION

Uses enhanced media verification system when database results < 2 restaurants.
All legacy Google Maps search code removed to prevent VenueResult conflicts.

FIXED METHOD NAMES:
- database_service.search_by_proximity() ✅ (not search_nearby_restaurants)
- filter_evaluator.filter_and_evaluate() ✅ (not filter_and_rank_restaurants)
- LocationData has .latitude and .longitude attributes, not .coordinates ✅
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import json

# Core location utilities
from location.location_utils import LocationUtils, LocationPoint
from location.telegram_location_handler import LocationData

# Database and filtering
from location.database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_telegram_formatter import LocationTelegramFormatter

# Enhanced verification system
from location.enhanced_media_verification import EnhancedMediaVerificationAgent
from location.location_text_editor import LocationTextEditor

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Location search orchestrator with enhanced media verification
    Uses new enhanced system when database results < 2 restaurants
    """

    def __init__(self, config):
        self.config = config

        # Initialize location-specific services
        self.database_service = LocationDatabaseService(config)  # Step 1
        self.filter_evaluator = LocationFilterEvaluator(config)  # Step 2

        # Enhanced verification system
        self.enhanced_verifier = EnhancedMediaVerificationAgent(config)  # Steps 1-6
        self.text_editor = LocationTextEditor(config)  # Description generation

        self.formatter = LocationTelegramFormatter(config)  # Unified formatter

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = 2  # Trigger enhanced search when < 2 results
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Location Orchestrator initialized with Enhanced Media Verification")

    # ============ MAIN PROCESSING METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query with enhanced verification trigger

        Logic:
        1. Database search
        2. If < 2 relevant results → Enhanced verification flow
        3. If >= 2 results → Traditional database flow
        """
        start_time = time.time()

        try:
            logger.info(f"🎯 Processing location query: '{query}'")

            # Step 1: Get coordinates from location data
            coordinates = self._extract_coordinates(location_data)
            location_desc = location_data.description or "your location"

            if not coordinates:
                return self._create_error_response("Unable to determine location coordinates")

            # Step 2: Database search - FIXED METHOD NAME (not async)
            db_restaurants = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=self.db_search_radius,
                extract_descriptions=True  # Get cuisine_tags and descriptions
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            db_restaurant_count = len(db_restaurants)
            logger.info(f"📊 Database search found {db_restaurant_count} restaurants")

            if db_restaurant_count > 0:
                # Filter database results - FIXED METHOD NAME (not async)
                filter_result = self.filter_evaluator.filter_and_evaluate(
                    restaurants=db_restaurants,
                    query=query,
                    location_description=location_desc
                )

                filtered_restaurants = filter_result.get("filtered_restaurants", [])
                filtered_count = len(filtered_restaurants)
                logger.info(f"🔍 After filtering: {filtered_count} relevant restaurants")

                # Check if we have enough results (>= 2)
                if filtered_count >= self.min_db_matches:
                    # Traditional database flow - sufficient results
                    logger.info(f"✅ Sufficient database results ({filtered_count}), using traditional flow")

                    formatted_results = self.formatter.format_database_results(
                        restaurants=filtered_restaurants,
                        query=query,
                        location_description=location_desc
                    )

                    return {
                        "success": True,
                        "results": filtered_restaurants,
                        "source": "database",
                        "processing_time": time.time() - start_time,
                        "restaurant_count": filtered_count,
                        "coordinates": coordinates,
                        "location_description": location_desc,
                        "location_formatted_results": formatted_results.get("message", f"Found {filtered_count} relevant restaurants!")
                    }
                else:
                    # Enhanced verification flow - insufficient database results
                    logger.info(f"⚡ Insufficient database results ({filtered_count} < {self.min_db_matches}), starting enhanced verification flow")
                    return await self._enhanced_verification_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

            else:
                # No database results - enhanced verification flow
                logger.info("⚡ No database results - starting enhanced verification flow")
                return await self._enhanced_verification_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

        except Exception as e:
            logger.error(f"❌ Error in location query processing: {e}")
            return self._create_error_response(f"Search failed: {str(e)}")

    # ============ ENHANCED VERIFICATION FLOW ============

    async def _enhanced_verification_flow(
        self,
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None,
        start_time=None
    ) -> Dict[str, Any]:
        """
        Enhanced verification flow using the new system

        This replaces the old Google Maps + media verification with the new integrated system
        """
        if start_time is None:
            start_time = time.time()

        try:
            logger.info("🚀 Starting Enhanced Verification Flow")

            # Use enhanced media verification agent
            # This handles Steps 1-6 of the new process:
            # 1. Enhanced Google Maps search
            # 2. AI review analysis
            # 4. Tavily media search  
            # 5. AI media source analysis
            # 6. Professional content scraping
            enhanced_venues = await self.enhanced_verifier.verify_and_enhance_venues(
                coordinates=coordinates,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not enhanced_venues:
                return self._create_error_response("No restaurants found matching your criteria after thorough search")

            logger.info(f"🔍 Enhanced verification found {len(enhanced_venues)} quality venues")

            # Use location text editor to create professional descriptions
            restaurant_results = await self.text_editor.create_professional_descriptions(enhanced_venues)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # Format final results
            final_formatted = self.text_editor.format_final_results(
                results=restaurant_results,
                user_coordinates=coordinates
            )

            processing_time = time.time() - start_time
            logger.info(f"✅ Enhanced verification flow completed in {processing_time:.1f}s")

            return {
                "success": True,
                "results": restaurant_results,
                "source": "enhanced_verification",
                "processing_time": processing_time,
                "restaurant_count": len(restaurant_results),
                "coordinates": coordinates,
                "location_description": location_desc,
                "location_formatted_results": final_formatted.get("message", f"Found {len(restaurant_results)} excellent restaurants!")
            }

        except Exception as e:
            logger.error(f"❌ Error in enhanced verification flow: {e}")
            return self._create_error_response(f"Enhanced verification failed: {str(e)}")

    # ============ LEGACY COMPATIBILITY METHODS ============

    async def complete_media_verification(
        self,
        venues: List[Any],  # Accept any venue type for compatibility
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        LEGACY METHOD: Maintained for backward compatibility with existing telegram bot code

        This method is still called by the telegram bot for the old flow.
        We redirect it to use the new enhanced system and ignore the venues parameter.
        """
        logger.info("🔄 Legacy complete_media_verification called - redirecting to enhanced flow")

        # Convert to new enhanced flow (ignore the old venues parameter)
        return await self._enhanced_verification_flow(
            query=query,
            coordinates=coordinates,
            location_desc=location_desc,
            cancel_check_fn=cancel_check_fn
        )

    async def process_more_results_query(
        self,
        query: str,
        coordinates: Tuple[float, float], 
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process "more results" query - now uses enhanced verification
        """
        logger.info(f"🔍 Processing 'more results' query with enhanced system: '{query}' at {location_desc}")

        # Validate coordinates
        if not coordinates or len(coordinates) != 2:
            logger.error(f"❌ Invalid coordinates provided: {coordinates}")
            return self._create_error_response("Invalid coordinates for search")

        try:
            latitude, longitude = float(coordinates[0]), float(coordinates[1])
            if not LocationUtils.validate_coordinates(latitude, longitude):
                return self._create_error_response("Invalid coordinate values")

            # Use enhanced verification flow for "more results"
            return await self._enhanced_verification_flow(
                query=query,
                coordinates=(latitude, longitude),
                location_desc=location_desc,
                cancel_check_fn=cancel_check_fn
            )

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Coordinate conversion error: {e}")
            return self._create_error_response("Invalid coordinate format")

    # ============ HELPER METHODS ============

    def _extract_coordinates(self, location_data: LocationData) -> Optional[Tuple[float, float]]:
        """Extract coordinates from location data - FIXED ATTRIBUTE ACCESS"""
        try:
            # FIXED: LocationData has .latitude and .longitude attributes, not .coordinates
            if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                if location_data.latitude is not None and location_data.longitude is not None:
                    lat, lng = location_data.latitude, location_data.longitude
                    if LocationUtils.validate_coordinates(lat, lng):
                        return (lat, lng)

            # If location_data doesn't have coordinates, try to geocode description
            if hasattr(location_data, 'description') and location_data.description:
                logger.info(f"🌍 Geocoding location: {location_data.description}")

                # Use database geocoding if available
                try:
                    from utils.database import get_database
                    db = get_database()
                    if hasattr(db, 'geocode_address'):
                        coordinates = db.geocode_address(location_data.description)
                        if coordinates:
                            logger.info(f"✅ Geocoded to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                            return coordinates
                        else:
                            logger.warning(f"❌ Failed to geocode: {location_data.description}")
                    else:
                        logger.warning("❌ Geocoding not available in database")
                except Exception as e:
                    logger.error(f"❌ Geocoding error: {e}")

            return None
        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
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
                        coordinates,  # (lat, lng) tuple
                        (restaurant_lat, restaurant_lng)  # (lat, lng) tuple
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

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "restaurant_count": 0,
            "location_formatted_results": f"😔 {error_message}"
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create standardized cancelled response"""
        return {
            "success": False,
            "cancelled": True,
            "results": [],
            "restaurant_count": 0,
            "location_formatted_results": "🚫 Search cancelled"
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline"""
        return {
            'database_service': True,
            'enhanced_verifier': True,
            'text_editor': True,
            'min_db_matches_trigger': self.min_db_matches,
            'enhanced_verification_stats': self.enhanced_verifier.get_verification_stats()
        }