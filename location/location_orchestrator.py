# location/location_orchestrator.py
"""
Location Search Orchestrator - CLEAN VERSION with updated imports

Updated for Phase 1.2:
- Fixed imports to use renamed files
- Consolidated overlapping functionality
- Clean import structure
- Removed duplicate methods
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import json

# UPDATED IMPORTS - Using new file structure
from location.location_utils import LocationUtils, LocationPoint
from location.telegram_location_handler import LocationData
from location.google_maps_search import GoogleMapsSearchAgent, VenueResult  # Will create this
from location.media_verification import MediaVerificationAgent  # Will create this
from location.database_search import LocationDatabaseService  # Renamed from database_service
from location.filter_evaluator import LocationFilterEvaluator
from location.location_telegram_formatter import LocationTelegramFormatter  # Will create this

# Import AI components for restaurant filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Location search orchestrator with clean architecture
    Steps 0-5 as specified in requirements
    """

    def __init__(self, config):
        self.config = config

        # Initialize location-specific services (renamed/clean versions)
        self.database_service = LocationDatabaseService(config)  # Step 1
        self.filter_evaluator = LocationFilterEvaluator(config)  # Step 2
        self.google_maps_agent = GoogleMapsSearchAgent(config)   # Step 3
        self.media_verifier = MediaVerificationAgent(config)     # Step 4 & 5
        self.formatter = LocationTelegramFormatter(config)       # Unified formatter

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = getattr(config, 'MIN_DB_MATCHES_REQUIRED', 3)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Location Orchestrator initialized with clean architecture")

    # ============ MAIN PROCESSING METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Clean location query processing following Steps 0-5

        Step 0: User coordinates (already handled)
        Step 1: Database proximity search  
        Step 2: AI-based analysis and filtering
        Step 3: Google Maps search (if needed)
        Step 4 & 5: Media verification and filtering
        """
        try:
            start_time = time.time()
            logger.info(f"🎯 Processing location query: '{query}'")

            # STEP 0: Get coordinates (already implemented, works fine)
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
                extract_descriptions=True  # Get cuisine_tags and descriptions
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 2: AI-based analysis and filtering
            logger.info("🧠 Step 2: AI analysis and filtering")
            filter_result = self.filter_evaluator.analyze_and_filter(
                restaurants=nearby_restaurants,
                query=query,
                location_description=location_desc
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # DECISION LOGIC: Check if we have good database results
            filtered_restaurants = filter_result.get("filtered_restaurants", [])
            should_send_immediately = filter_result.get("send_immediately", False)

            if should_send_immediately and len(filtered_restaurants) > 0:
                # STEP 2a: Send database results with user choice
                logger.info(f"✅ Found {len(filtered_restaurants)} good database matches")

                # Add distance info and format results
                restaurants_with_distance = self._add_distance_info(filtered_restaurants, coordinates)

                formatted_results = self.formatter.format_database_results(
                    restaurants=restaurants_with_distance,
                    query=query,
                    location_description=location_desc,
                    offer_more_search=True  # Ask if user wants more results
                )

                processing_time = time.time() - start_time
                return {
                    "results": formatted_results,
                    "source": "database_with_choice",
                    "processing_time": processing_time,
                    "restaurant_count": len(filtered_restaurants),
                    "coordinates": coordinates,
                    "location_description": location_desc,
                    "needs_user_response": True  # Indicate user choice needed
                }

            else:
                # STEP 2b: No perfect matches, launch Google Maps search
                logger.info("🔍 No sufficient database results, launching Google Maps search")
                return await self._google_maps_search_flow(
                    coordinates, query, location_desc, start_time, cancel_check_fn
                )

        except Exception as e:
            logger.error(f"❌ Error in location search: {e}")
            return self._create_error_response(f"Search error: {str(e)}")

    async def _google_maps_search_flow(
        self,
        coordinates: Tuple[float, float],
        query: str,
        location_desc: str,
        start_time: float,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Handle Steps 3-5: Google Maps search + media verification
        """
        try:
            # STEP 3: Google Maps search
            logger.info("🗺️ Step 3: Google Maps search")
            venues = await self.google_maps_agent.search_venues(
                coordinates=coordinates,
                query=query,
                radius_km=self.db_search_radius,
                max_results=self.max_venues_to_verify
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 4 & 5: Media verification  
            logger.info("📰 Steps 4-5: Media verification")
            verified_venues = await self.media_verifier.verify_venues(
                venues=venues,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # Add distance information
            venues_with_distance = self._add_distance_info(verified_venues, coordinates)

            # Format final results
            formatted_results = self.formatter.format_google_maps_results(
                restaurants=venues_with_distance,
                query=query,
                location_description=location_desc
            )

            processing_time = time.time() - start_time
            logger.info(f"⚡ Google Maps search completed in {processing_time:.2f}s")

            return {
                "results": formatted_results,
                "source": "google_maps_verified",
                "processing_time": processing_time,
                "restaurant_count": len(verified_venues),
                "coordinates": coordinates,
                "location_description": location_desc
            }

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search flow: {e}")
            return self._create_error_response(f"Google Maps search error: {str(e)}")

    # ============ UTILITY METHODS ============

    def _get_coordinates(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates from location data"""
        try:
            if location_data.location_type == "gps":
                lat = location_data.latitude
                lng = location_data.longitude

                if lat is not None and lng is not None:
                    return (float(lat), float(lng))
                else:
                    logger.error("GPS location data has None coordinates")
                    return None

            elif location_data.location_type == "description" and location_data.description:
                logger.info(f"🗺️ Geocoding location: {location_data.description}")

                # Use database geocoding method
                from utils.database import get_database
                db = get_database()

                coordinates = db.geocode_address(location_data.description)
                if coordinates:
                    return coordinates

                logger.warning("No geocoding fallback available")

            return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None

    def _add_distance_info(
        self, 
        restaurants: List[Dict[str, Any]], 
        coordinates: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """Add distance information to each restaurant"""
        try:
            user_lat, user_lng = coordinates
            restaurants_with_distance = []

            for restaurant in restaurants:
                restaurant_copy = restaurant.copy()

                # Calculate distance if restaurant has coordinates
                r_lat = restaurant.get('latitude')
                r_lng = restaurant.get('longitude')

                if r_lat and r_lng:
                    try:
                        distance_km = LocationUtils.calculate_distance(
                            (user_lat, user_lng), (float(r_lat), float(r_lng))
                        )
                        restaurant_copy['distance_km'] = round(distance_km, 2)
                        restaurant_copy['distance_text'] = LocationUtils.format_distance(distance_km)
                    except Exception as e:
                        logger.warning(f"Could not calculate distance for {restaurant.get('name', 'Unknown')}: {e}")
                        restaurant_copy['distance_km'] = None
                        restaurant_copy['distance_text'] = "Distance unknown"
                else:
                    restaurant_copy['distance_km'] = None
                    restaurant_copy['distance_text'] = "Distance unknown"

                restaurants_with_distance.append(restaurant_copy)

            # Sort by distance (closest first)
            restaurants_with_distance.sort(
                key=lambda x: x.get('distance_km', float('inf'))
            )

            return restaurants_with_distance

        except Exception as e:
            logger.error(f"❌ Error adding distance info: {e}")
            return restaurants

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "results": {"main_list": [], "search_info": {"error": error_message}},
            "source": "error",
            "processing_time": 0,
            "restaurant_count": 0,
            "error": error_message
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create cancelled response"""
        return {
            "results": {"main_list": [], "search_info": {"cancelled": True}},
            "source": "cancelled",
            "processing_time": 0,
            "restaurant_count": 0,
            "cancelled": True
        }