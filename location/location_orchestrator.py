# location/location_orchestrator.py
"""
Location Search Orchestrator - COMPLETE CLEAN VERSION WITH AI DESCRIPTION EDITOR

Uses enhanced media verification system when database results < 2 restaurants.
All legacy Google Maps search code removed to prevent VenueResult conflicts.

ENHANCED FLOW:
1. Database search (extract raw_descriptions, sources)
2. AI filter restaurants 
3. AI edit descriptions (NEW STEP)
4. Format and send to Telegram

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
from location.location_ai_editor import LocationAIEditor
from location.location_map_search import LocationMapSearchAgent
from location.enhanced_media_verification import LocationMediaVerificationAgent

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Location search orchestrator with enhanced media verification AND AI description editing
    Uses new enhanced system when database results < 2 restaurants

    Enhanced flow:
    1. Database search → 2. Filter evaluation → 3. AI description editing → 4. Telegram format
    """

    def __init__(self, config):
        self.config = config

        # Initialize location-specific services
        self.database_service = LocationDatabaseService(config)  # Step 1
        self.filter_evaluator = LocationFilterEvaluator(config)  # Step 2

        # AI Description Editor for database results
        self.description_editor = LocationAIEditor(config)

        # NEW: Separate agents for enhanced verification flow
        from location.location_map_search import LocationMapSearchAgent
        from location.enhanced_media_verification import LocationMediaVerificationAgent
        from location.location_ai_editor import LocationAIEditor

        self.map_search_agent = LocationMapSearchAgent(config)
        self.media_verification_agent = LocationMediaVerificationAgent(config)
        self.ai_editor = LocationAIEditor(config)

        self.formatter = LocationTelegramFormatter(config)  # Unified formatter

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = 2  # Trigger enhanced search when < 2 results
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Location Orchestrator initialized with Separate Agent Architecture")


    # ============ MAIN PROCESSING METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query with dual flow architecture

        Flow Logic:
        1. Database search
        2. Filter evaluation  
        3. AI description editing
        4. If < 2 relevant results → Enhanced verification flow (separate agents)
        5. Format for Telegram
        """
        start_time = time.time()

        try:
            logger.info(f"Processing location query: '{query}'")

            # Step 1: Get coordinates from location data
            coordinates = self._extract_coordinates(location_data)
            location_desc = location_data.description or "your location"

            if not coordinates:
                return self._create_error_response("Unable to determine location coordinates")

            # Step 2: Database search with raw descriptions and sources
            logger.info("Step 1: Database proximity search...")
            db_restaurants = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=self.db_search_radius,
                extract_descriptions=True
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            db_restaurant_count = len(db_restaurants)
            logger.info(f"Database search found {db_restaurant_count} restaurants")

            if db_restaurant_count > 0:
                # Step 3: Filter database results
                logger.info("Step 2: AI filtering evaluation...")
                filter_result = self.filter_evaluator.filter_and_evaluate(
                    restaurants=db_restaurants,
                    query=query,
                    location_description=location_desc
                )

                filtered_restaurants = filter_result.get("filtered_restaurants", [])
                filtered_count = len(filtered_restaurants)
                logger.info(f"After filtering: {filtered_count} relevant restaurants")

                # Step 4: AI Description Editing for database results
                if filtered_restaurants:
                    logger.info("Step 3: AI description editing...")
                    edited_restaurants = self.description_editor.edit_descriptions(
                        filtered_restaurants=filtered_restaurants,
                        user_query=query,
                        location_description=location_desc
                    )

                    logger.info(f"AI edited {len(edited_restaurants)} restaurant descriptions")
                else:
                    edited_restaurants = []

                # Check if we have enough results
                if len(edited_restaurants) >= self.min_db_matches:
                    # Sufficient database results - use database flow
                    logger.info(f"Sufficient database results ({len(edited_restaurants)}), using database flow")

                    # Format for Telegram using AI-edited descriptions
                    formatted_results = self.description_editor.create_telegram_formatted_results(
                        edited_restaurants=edited_restaurants,
                        user_query=query,
                        location_description=location_desc
                    )

                    return {
                        "success": True,
                        "results": edited_restaurants,
                        "source": "database_ai_enhanced", 
                        "processing_time": time.time() - start_time,
                        "restaurant_count": len(edited_restaurants),
                        "coordinates": coordinates,
                        "location_description": location_desc,
                        "location_formatted_results": formatted_results.get("message", f"Found {len(edited_restaurants)} relevant restaurants!"),
                        "ai_edited": True
                    }
                else:
                    # Insufficient database results - use enhanced flow
                    logger.info(f"Insufficient database results ({len(edited_restaurants)} < {self.min_db_matches}), starting enhanced verification flow")
                    return await self._enhanced_verification_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

            else:
                # No database results - use enhanced flow
                logger.info("No database results - starting enhanced verification flow")
                return await self._enhanced_verification_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

        except Exception as e:
            logger.error(f"Error in location query processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        Enhanced verification flow using separate agent architecture

        New Flow:
        1. LocationMapSearchAgent -> Google Maps/Places search with AI analysis
        2. LocationMediaVerificationAgent -> Professional media coverage search
        3. LocationAIEditor -> Combine results into professional descriptions
        4. Format for Telegram display
        """
        if start_time is None:
            start_time = time.time()

        try:
            logger.info("Starting Enhanced Verification Flow with Separate Agents")

            # Step 1: Google Maps/Places search with AI-powered query analysis
            logger.info("Step 1: AI-powered Google Maps search")
            map_search_results = await self.map_search_agent.search_venues_with_ai_analysis(
                coordinates=coordinates,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not map_search_results:
                return self._create_error_response("No restaurants found matching your criteria in Google Maps search")

            logger.info(f"Step 1: Found {len(map_search_results)} venues from Google Maps search")

            # Step 2: Media verification for professional coverage
            logger.info("Step 2: Professional media verification")
            media_verification_results = await self.media_verification_agent.verify_venues_media_coverage(
                venues=map_search_results,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"Step 2: Completed media verification for {len(media_verification_results)} venues")

            # Step 3: AI description generation combining both sources
            logger.info("Step 3: AI-powered description generation")
            restaurant_descriptions = await self.ai_editor.create_professional_descriptions(
                map_search_results=map_search_results,
                media_verification_results=media_verification_results,
                user_query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not restaurant_descriptions:
                return self._create_error_response("Failed to generate restaurant descriptions")

            logger.info(f"Step 3: Generated {len(restaurant_descriptions)} professional descriptions")

            # Step 4: Format final results for Telegram
            logger.info("Step 4: Formatting results for display")
            final_formatted = self.ai_editor.format_final_results(
                descriptions=restaurant_descriptions,
                user_coordinates=coordinates
            )

            processing_time = time.time() - start_time
            logger.info(f"Enhanced verification flow completed in {processing_time:.1f}s")

            # Return comprehensive results
            return {
                "success": True,
                "results": restaurant_descriptions,
                "source": "enhanced_verification_separate_agents",
                "processing_time": processing_time,
                "restaurant_count": len(restaurant_descriptions),
                "coordinates": coordinates,
                "location_description": location_desc,
                "location_formatted_results": final_formatted.get("message", f"Found {len(restaurant_descriptions)} excellent restaurants!"),
                "has_media_coverage": final_formatted.get("has_media_coverage", False),
                "search_stats": {
                    "google_maps_results": len(map_search_results),
                    "media_verified_venues": len(media_verification_results),
                    "final_descriptions": len(restaurant_descriptions),
                    "venues_with_media_coverage": sum(1 for desc in restaurant_descriptions if desc.has_media_coverage)
                }
            }

        except Exception as e:
            logger.error(f"Error in enhanced verification flow: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(f"Enhanced verification failed: {str(e)}")

    async def process_more_results_query(
        self,
        query: str,
        coordinates: Tuple[float, float], 
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process "more results" query - uses enhanced verification flow
        """
        logger.info(f"Processing 'more results' query with enhanced system: '{query}' at {location_desc}")

        # Validate coordinates
        if not coordinates or len(coordinates) != 2:
            logger.error(f"Invalid coordinates provided: {coordinates}")
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
            logger.error(f"Coordinate conversion error: {e}")
            return self._create_error_response("Invalid coordinate format")

    # ============ LEGACY COMPATIBILITY METHODS ============

    async def complete_media_verification(
        self,
        venues: List[Any],
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        LEGACY METHOD: Maintained for backward compatibility with existing telegram bot code
        Redirects to enhanced verification flow using separate agents
        """
        logger.info("Legacy complete_media_verification called - redirecting to enhanced flow")

        return await self._enhanced_verification_flow(
            query=query,
            coordinates=coordinates,
            location_desc=location_desc,
            cancel_check_fn=cancel_check_fn
        )

    # ============ HELPER METHODS ============

    def _extract_coordinates(self, location_data: LocationData) -> Optional[Tuple[float, float]]:
        """Extract coordinates from location data"""
        try:
            if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                if location_data.latitude is not None and location_data.longitude is not None:
                    lat, lng = location_data.latitude, location_data.longitude
                    if LocationUtils.validate_coordinates(lat, lng):
                        return (lat, lng)

            # If location_data doesn't have coordinates, try to geocode description
            if hasattr(location_data, 'description') and location_data.description:
                logger.info(f"Geocoding location: {location_data.description}")

                try:
                    from utils.database import get_database
                    db = get_database()
                    if hasattr(db, 'geocode_address'):
                        coordinates = db.geocode_address(location_data.description)
                        if coordinates:
                            logger.info(f"Geocoded to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                            return coordinates
                        else:
                            logger.warning(f"Failed to geocode: {location_data.description}")
                    else:
                        logger.warning("Geocoding not available in database")
                except Exception as e:
                    logger.error(f"Geocoding error: {e}")

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

    def _create_no_results_response(self, query: str, location_desc: str, processing_time: float) -> Dict[str, Any]:
        """Create response when no results are found"""
        return {
            "success": False,
            "no_results": True,
            "processing_time": processing_time,
            "location_formatted_results": f"Sorry, I couldn't find any restaurants matching '{query}' near {location_desc}. Try broadening your search or checking a different area."
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline"""
        return {
            'database_service': True,
            'enhanced_verifier': True,
            'text_editor': True,
            'description_editor': True,  # NEW
            'min_db_matches_trigger': self.min_db_matches,
            'enhanced_verification_stats': self.enhanced_verifier.get_verification_stats()
        }

    # ============ DEBUGGING METHODS ============

    async def debug_location_search(
        self, 
        query: str, 
        location_data: LocationData
    ) -> Dict[str, Any]:
        """
        Debug method to show detailed pipeline information
        """
        debug_info = {
            "query": query,
            "location_data": {
                "latitude": getattr(location_data, 'latitude', None),
                "longitude": getattr(location_data, 'longitude', None),
                "description": getattr(location_data, 'description', None)
            },
            "pipeline_steps": {}
        }

        try:
            # Step 1: Database search
            coordinates = self._extract_coordinates(location_data)
            if coordinates:
                db_restaurants = self.database_service.search_by_proximity(coordinates, extract_descriptions=True)
                debug_info["pipeline_steps"]["database_search"] = {
                    "restaurant_count": len(db_restaurants),
                    "with_descriptions": sum(1 for r in db_restaurants if r.get('raw_description')),
                    "with_sources": sum(1 for r in db_restaurants if r.get('sources_domains')),
                    "sample_restaurants": [r.get('name', 'Unknown') for r in db_restaurants[:3]]
                }

                # Step 2: Filtering
                if db_restaurants:
                    filter_result = self.filter_evaluator.filter_and_evaluate(
                        restaurants=db_restaurants,
                        query=query,
                        location_description=location_data.description or "your location"
                    )
                    filtered_restaurants = filter_result.get("filtered_restaurants", [])
                    debug_info["pipeline_steps"]["filtering"] = {
                        "filtered_count": len(filtered_restaurants),
                        "filter_reasoning": filter_result.get("reasoning", "No reasoning provided")
                    }

                    # Step 3: AI Description Editing
                    if filtered_restaurants:
                        edited_restaurants = self.description_editor.edit_descriptions(
                            filtered_restaurants=filtered_restaurants,
                            user_query=query,
                            location_description=location_data.description or "your location"
                        )
                        debug_info["pipeline_steps"]["ai_description_editing"] = {
                            "edited_count": len(edited_restaurants),
                            "sample_descriptions": [
                                {
                                    "name": r.get('name', 'Unknown'),
                                    "description": r.get('description', '')[:100] + "..."
                                }
                                for r in edited_restaurants[:2]
                            ]
                        }

            return debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            return debug_info