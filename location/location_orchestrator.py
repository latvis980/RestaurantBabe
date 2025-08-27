import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import json

# UPDATED IMPORTS - Using enhanced verification system
from location.location_utils import LocationUtils, LocationPoint
from location.telegram_location_handler import LocationData
from location.database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_telegram_formatter import LocationTelegramFormatter

# NEW ENHANCED SYSTEM IMPORTS
from location.enhanced_media_verification import EnhancedMediaVerificationAgent
from location.location_text_editor import LocationTextEditor

# Import AI components for restaurant filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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

        # NEW: Initialize enhanced verification system
        self.enhanced_verifier = EnhancedMediaVerificationAgent(config)  # Steps 1-6
        self.text_editor = LocationTextEditor(config)  # Description generation

        self.formatter = LocationTelegramFormatter(config)  # Unified formatter

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = 2  # UPDATED: Trigger enhanced search when < 2 results
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
        UPDATED: Process location query with enhanced verification trigger

        NEW LOGIC:
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

            # Step 2: Database search (existing logic)
            db_restaurants = await self.database_service.search_nearby_restaurants(
                coordinates=coordinates,
                query=query,
                radius_km=self.db_search_radius
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            db_restaurant_count = len(db_restaurants)
            logger.info(f"📊 Database search found {db_restaurant_count} restaurants")

            if db_restaurant_count > 0:
                # Filter database results
                filtered_restaurants = await self.filter_evaluator.filter_and_rank_restaurants(
                    restaurants=db_restaurants,
                    query=query,
                    coordinates=coordinates
                )

                filtered_count = len(filtered_restaurants)
                logger.info(f"🔍 After filtering: {filtered_count} relevant restaurants")

                # NEW LOGIC: Check if we have enough results (>= 2)
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
                    # NEW: Enhanced verification flow - insufficient database results
                    logger.info(f"⚡ Insufficient database results ({filtered_count} < {self.min_db_matches}), starting enhanced verification flow")
                    return await self._enhanced_verification_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

            else:
                # NEW: No database results - enhanced verification flow
                logger.info("⚡ No database results - starting enhanced verification flow")
                return await self._enhanced_verification_flow(query, coordinates, location_desc, cancel_check_fn, start_time)

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

    async def _enhanced_verification_flow(
        self,
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None,
        start_time=None
    ) -> Dict[str, Any]:
        """
        NEW: Enhanced verification flow using the new system

        This replaces the old Google Maps + media verification with the new integrated system
        """
        if start_time is None:
            start_time = time.time()

        try:
            logger.info("🚀 Starting Enhanced Verification Flow")

            # NEW: Use enhanced media verification agent
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

            # NEW: Use location text editor to create professional descriptions
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


    # ============ HELPER METHODS ============

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
            # FIXED: Pass coordinates for distance calculation in formatting
            verified_venues = await self.media_verifier.verify_venues(
                venues=venues,
                query=query,
                coordinates=coordinates,  # ADDED: coordinates parameter
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"📸 Media verification completed: {len(verified_venues)} venues verified")
            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error in venue verification: {e}")
            # Fallback: convert venues to dict format without verification
            return self._convert_venues_to_fallback(venues, coordinates)  # ADDED: coordinates parameter


    def _convert_venues_to_fallback(self, venues: List[VenueResult], coordinates: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Convert VenueResult objects to fallback format with distance calculation"""
        try:
            fallback_venues = []

            for venue in venues:
                # Calculate distance for formatting
                distance_km = venue.distance_km
                if coordinates and venue.latitude and venue.longitude:
                    distance_km = LocationUtils.calculate_distance(
                        coordinates, 
                        (venue.latitude, venue.longitude)
                    )

                fallback_venues.append({
                    'name': venue.name,
                    'address': venue.address,
                    'latitude': venue.latitude,
                    'longitude': venue.longitude,
                    'distance_km': distance_km,
                    'distance_text': LocationUtils.format_distance(distance_km) if distance_km else "Distance unknown",
                    'rating': venue.rating,
                    'place_id': venue.place_id,
                    'description': f"Restaurant near {venue.address}",
                    'sources': [],
                    'media_verified': False,
                    'google_maps_url': venue.google_maps_url
                })

            return fallback_venues

        except Exception as e:
            logger.error(f"❌ Error in fallback conversion: {e}")
            return []

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
        """Add distance information to restaurants - FIXED method call"""
        try:
            restaurants_with_distance = []

            for restaurant in restaurants:
                # Calculate distance if restaurant has coordinates
                restaurant_lat = restaurant.get('latitude')
                restaurant_lng = restaurant.get('longitude')

                if restaurant_lat and restaurant_lng:
                    # FIXED: Pass coordinates as tuples, not separate arguments
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