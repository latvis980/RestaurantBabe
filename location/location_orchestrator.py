# location/location_orchestrator.py
"""
Location Search Orchestrator - FIXED IMPORT AND NAMING ISSUES

Uses enhanced media verification system when database results < 2 restaurants.
All legacy Google Maps search code removed to prevent VenueResult conflicts.

FIXED ISSUES:
- Changed import from 'enhanced_media_verification' to 'location_media_verification'
- Removed references to non-existent 'enhanced_verifier' attribute
- Fixed all import paths and class references
- Cleaned up duplicate LocationAIEditor imports

ENHANCED FLOW:
1. Database search (extract raw_descriptions, sources)
2. AI filter restaurants 
3. AI edit descriptions (NEW STEP)
4. Format and send to Telegram
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

# AI Editor and separate agents - FIXED IMPORTS
from location.location_ai_editor import LocationAIEditor
from location.location_map_search import LocationMapSearchAgent
from location.location_media_verification import LocationMediaVerificationAgent  # FIXED: was enhanced_media_verification

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

        # NEW: Separate agents for enhanced verification flow - FIXED: removed duplicate imports
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
        FIXED: Geocode location_data BEFORE extracting coordinates

        Flow Logic:
        1. Geocode location if needed
        2. Database search
        3. Filter evaluation  
        4. AI description editing
        5. If < 2 relevant results → Enhanced verification flow (separate agents)
        6. Format for Telegram
        """
        start_time = time.time()

        try:
            # STEP 0: Geocode location_data if coordinates are missing
            if (location_data.latitude is None or location_data.longitude is None) and location_data.description:
                logger.info(f"🌍 Geocoding location description: {location_data.description}")

                try:
                    geocoded_coords = LocationUtils.geocode_location(location_data.description)
                    if geocoded_coords:
                        location_data.latitude = geocoded_coords[0]
                        location_data.longitude = geocoded_coords[1]
                        logger.info(f"✅ Successfully geocoded: {geocoded_coords[0]:.4f}, {geocoded_coords[1]:.4f}")
                    else:
                        logger.error(f"❌ Failed to geocode location: {location_data.description}")
                        return self._create_error_response(f"Could not find location: {location_data.description}")
                except Exception as e:
                    logger.error(f"❌ Geocoding error: {e}")
                    return self._create_error_response(f"Error finding location: {location_data.description}")

            # STEP 1: Extract coordinates (should work now)
            coordinates = self._extract_coordinates(location_data)
            if not coordinates:
                return self._create_error_response("Could not extract valid coordinates from location data")

            latitude, longitude = coordinates
            location_desc = getattr(location_data, 'description', f"GPS: {latitude:.4f}, {longitude:.4f}")

            logger.info(f"Processing location query: '{query}' at {location_desc}")

            # Step 2: Database proximity search
            logger.info(f"Step 2: Database search within {self.db_search_radius}km")
            db_restaurants = await self.database_service.search_by_proximity(
                coordinates=(latitude, longitude),
                radius_km=self.db_search_radius,
                max_results=self.max_venues_to_verify
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"Step 2: Found {len(db_restaurants)} restaurants in database")

            if db_restaurants:
                # Step 3: AI filter and evaluate relevance
                logger.info(f"Step 3: AI filtering {len(db_restaurants)} database results")
                filtered_restaurants = await self.filter_evaluator.filter_restaurants(
                    restaurants=db_restaurants,
                    query=query,
                    coordinates=(latitude, longitude)
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                logger.info(f"Step 3: {len(filtered_restaurants)} restaurants passed AI filtering")

                # Step 4: AI edit descriptions for database results
                if filtered_restaurants and self.description_editor:
                    logger.info(f"Step 4: AI editing descriptions for {len(filtered_restaurants)} results")
                    edited_restaurants = await self.description_editor.edit_restaurant_descriptions(
                        restaurants=filtered_restaurants,
                        query=query,
                        coordinates=(latitude, longitude)
                    )

                    if cancel_check_fn and cancel_check_fn():
                        return self._create_cancelled_response()

                    logger.info(f"Step 4: AI editing completed")
                    filtered_restaurants = edited_restaurants

                # Step 5: Check if we have enough results
                if len(filtered_restaurants) >= self.min_db_matches:
                    # Format and return database results
                    formatted_results = self.formatter.format_for_telegram(
                        restaurants=filtered_restaurants,
                        query=query,
                        location_desc=location_desc,
                        source="database"
                    )

                    processing_time = round(time.time() - start_time, 2)
                    logger.info(f"✅ Database flow completed in {processing_time}s - {len(filtered_restaurants)} results")

                    return {
                        "success": True,
                        "results": filtered_restaurants,
                        "location_formatted_results": formatted_results,
                        "restaurant_count": len(filtered_restaurants),
                        "source": "database",
                        "processing_time": processing_time,
                        "coordinates": (latitude, longitude)  # Include coordinates in response
                    }

            # Step 6: Enhanced verification flow (< 2 database results)
            logger.info(f"Step 6: Triggering enhanced verification flow (found {len(db_restaurants if db_restaurants else [])} database results)")

            enhanced_result = await self._enhanced_verification_flow(
                query=query,
                coordinates=(latitude, longitude),
                location_desc=location_desc,
                cancel_check_fn=cancel_check_fn
            )

            # Add coordinates to the response
            if enhanced_result.get("success"):
                enhanced_result["coordinates"] = (latitude, longitude)

            processing_time = round(time.time() - start_time, 2)
            logger.info(f"✅ Enhanced flow completed in {processing_time}s")

            if "processing_time" not in enhanced_result:
                enhanced_result["processing_time"] = processing_time

            return enhanced_result

        except Exception as e:
            logger.error(f"❌ Error in process_location_query: {e}")
            return self._create_error_response(f"Search error: {str(e)}")

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

            logger.warning(f"Invalid coordinates in location_data: lat={getattr(location_data, 'latitude', None)}, lng={getattr(location_data, 'longitude', None)}")
            return None

        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            return None

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "source": "error",
            "location_formatted_results": f"😔 {error_message}",
            "restaurant_count": 0
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create standardized cancellation response"""
        return {
            "success": False,
            "cancelled": True,
            "results": [],
            "source": "cancelled",
            "location_formatted_results": "🔄 Search was cancelled.",
            "restaurant_count": 0
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline"""
        return {
            'database_service': True,
            'enhanced_verifier': self.media_verification_agent is not None,
            'text_editor': self.description_editor is not None,
            'description_editor': self.ai_editor is not None,
            'min_db_matches_trigger': self.min_db_matches,
            'enhanced_verification_stats': self.media_verification_agent.get_verification_stats() if self.media_verification_agent else None
        }