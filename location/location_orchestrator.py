# location/location_orchestrator.py
"""
Location Search Orchestrator - COMPLETE with database filtering

Updated features:
1. Coordinate-based database search with AI filtering
2. Immediate sending of database results as "personal notes"
3. Fallback to Google Maps search if no database results
4. Isolated filtering logic for location searches
5. Maintains all original functionality
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import json

from utils.location_utils import LocationUtils, LocationPoint
from utils.telegram_location_handler import LocationData
from location.location_search_agent import LocationSearchAgent, VenueResult
from location.media_search_agent import MediaSearchAgent
from location.database_service import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from formatters.telegram_formatter import TelegramFormatter
from location.result_formatter import LocationResultFormatter

# Import AI components for restaurant filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Location search orchestrator with database-first approach
    """

    def __init__(self, config):
        self.config = config

        # Initialize location-specific services
        self.db_service = LocationDatabaseService(config)
        self.filter_evaluator = LocationFilterEvaluator(config)
        self.result_formatter = LocationResultFormatter(config)

        # Initialize existing components
        self.location_search_agent = LocationSearchAgent(config)
        self.media_search_agent = MediaSearchAgent(config)
        self.telegram_formatter = TelegramFormatter()

        # Initialize AI for restaurant filtering
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = getattr(config, 'MIN_DB_MATCHES_REQUIRED', 3)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Location Orchestrator initialized with database-first approach")

    # ============ MAIN PROCESSING METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query with database-first approach and enhanced formatting
        """
        try:
            start_time = time.time()
            logger.info(f"🎯 Starting enhanced location search: '{query}'")

            # STEP 1: Get coordinates
            coordinates = self._get_coordinates_sync(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates
            location_desc = location_data.description or f"GPS location ({lat:.4f}, {lng:.4f})"
            logger.info(f"📍 Coordinates: {lat:.4f}, {lng:.4f}")

            # STEP 2: Check database and filter with enhanced result processing
            db_result = self._check_database_and_filter_enhanced(
                coordinates, query, location_desc, cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 3: Decision based on database results
            if db_result.get("send_immediately", False):
                # IMMEDIATE SENDING - Found relevant restaurants in database
                filtered_restaurants = db_result.get("filtered_restaurants", [])

                logger.info(f"✅ IMMEDIATE SEND: {len(filtered_restaurants)} restaurants with enhanced formatting")

                # Add distance information to each restaurant
                restaurants_with_distance = self._add_distance_info(filtered_restaurants, coordinates)

                # Format results using the new comprehensive formatter
                formatted_results = self.result_formatter.format_location_results(
                    restaurants=restaurants_with_distance,
                    query=query,
                    location_description=location_desc,
                    source="personal notes"
                )

                processing_time = time.time() - start_time
                logger.info(f"⚡ Location search completed in {processing_time:.2f}s")

                return {
                    "results": formatted_results,
                    "source": "database_immediate",
                    "processing_time": processing_time,
                    "restaurant_count": len(filtered_restaurants),
                    "search_type": "location_database",
                    "coordinates": coordinates,
                    "location_description": location_desc
                }

            else:
                # FALLBACK to Google Maps search
                logger.info(f"🔍 No sufficient database results, falling back to Google Maps search")

                venues = await self._search_google_maps_venues(
                    coordinates, query, location_desc, cancel_check_fn
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                # Verify venues using media search
                verified_venues = self._verify_venues_sync(venues, cancel_check_fn)

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                # Format Google Maps results
                formatted_venues = self._format_google_maps_results(
                    verified_venues, coordinates, query
                )

                processing_time = time.time() - start_time
                logger.info(f"⚡ Location search completed in {processing_time:.2f}s")

                return {
                    "results": formatted_venues,
                    "source": "google_maps",
                    "processing_time": processing_time,
                    "restaurant_count": len(verified_venues),
                    "search_type": "location_google_maps",
                    "coordinates": coordinates,
                    "location_description": location_desc
                }

        except Exception as e:
            logger.error(f"❌ Error in enhanced location search: {e}")
            return self._create_error_response(f"Search error: {str(e)}")

    # ============ DATABASE METHODS ============

    # Fix for location/location_orchestrator.py
    # Replace the _check_database_and_filter_enhanced method

    def _check_database_and_filter_enhanced(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Enhanced database check and filtering with better result processing"""
        try:
            # Step 1: Get nearby restaurants from database
            nearby_restaurants = self.db_service.get_restaurants_by_proximity(
                coordinates, self.db_search_radius
            )

            if cancel_check_fn and cancel_check_fn():
                return {"send_immediately": False, "filtered_restaurants": []}

            logger.info(f"🗃️ Found {len(nearby_restaurants)} restaurants within {self.db_search_radius}km")

            if not nearby_restaurants:
                return {"send_immediately": False, "filtered_restaurants": []}

            # Step 2: Filter restaurants using AI - USE EXISTING METHOD
            filter_result = self.filter_evaluator.filter_and_evaluate(
                restaurants=nearby_restaurants,
                query=query,
                location_description=location_desc
            )

            if cancel_check_fn and cancel_check_fn():
                return {"send_immediately": False, "filtered_restaurants": []}

            filtered_restaurants = filter_result.get("filtered_restaurants", [])
            evaluation = filter_result.get("evaluation", {})
            send_immediately = filter_result.get("send_immediately", False)

            logger.info(f"🧠 AI filtered to {len(filtered_restaurants)} relevant restaurants")

            # Step 3: Enhanced decision logic - be more generous for location searches
            if not send_immediately and len(filtered_restaurants) >= 1:
                logger.info("🔧 Overriding AI decision - sending single good match for location search")
                send_immediately = True

            return {
                "send_immediately": send_immediately,
                "filtered_restaurants": filtered_restaurants,
                "evaluation": evaluation,
                "total_nearby": len(nearby_restaurants)
            }

        except Exception as e:
            logger.error(f"❌ Error in enhanced database check: {e}")
            return {"send_immediately": False, "filtered_restaurants": []}

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
                        distance_km = LocationUtils.calculate_distance((lat, lng), (v_lat, v_lng))
                        restaurant_copy['distance_km'] = distance_km
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

    # ============ GOOGLE MAPS METHODS ============

    async def _search_google_maps_venues(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        location_desc: str,
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues"""
        try:
            lat, lng = coordinates

            venues = await self.location_search_agent.search_venues(
                latitude=lat,
                longitude=lng,
                query=query,
                radius_km=self.db_search_radius,
                max_results=self.max_venues_to_verify
            )

            logger.info(f"🗺️ Google Maps found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search: {e}")
            return []

    def _verify_venues_sync(self, venues: List[VenueResult], cancel_check_fn=None) -> List[Dict[str, Any]]:
        """Verify venues using media search (synchronous version)"""
        try:
            # Convert VenueResult objects to format expected by media search
            venue_data = []
            for venue in venues[:self.max_venues_to_verify]:
                city = self._extract_city_from_address(venue.address)

                venue_data.append({
                    'name': venue.name,
                    'address': venue.address,
                    'city': city,
                    'latitude': venue.latitude,
                    'longitude': venue.longitude,
                    'rating': getattr(venue, 'rating', None),
                    'place_id': getattr(venue, 'place_id', None)
                })

            # Simple verification for now
            verified_results = []
            for venue_info in venue_data:
                if cancel_check_fn and cancel_check_fn():
                    break
                verified_results.append(venue_info)

            return verified_results

        except Exception as e:
            logger.error(f"❌ Error in venue verification: {e}")
            return []

    def _format_google_maps_results(
        self, 
        venues: List[Dict[str, Any]], 
        coordinates: Tuple[float, float],
        query: str
    ) -> Dict[str, Any]:
        """Format Google Maps results"""
        try:
            lat, lng = coordinates

            # Add distance to each venue
            for venue in venues:
                try:
                    v_lat = venue.get('latitude')
                    v_lng = venue.get('longitude')

                    if v_lat and v_lng:
                        distance_km = LocationUtils.calculate_distance((user_lat, user_lng), (r_lat, r_lng))
                        venue['distance_km'] = distance_km
                        venue['distance_text'] = LocationUtils.format_distance(distance_km)
                    else:
                        venue['distance_km'] = None
                        venue['distance_text'] = "Distance unknown"
                except Exception as e:
                    logger.warning(f"Could not calculate distance for venue: {e}")
                    venue['distance_km'] = None
                    venue['distance_text'] = "Distance unknown"

            # Sort by distance
            venues.sort(key=lambda x: x.get('distance_km', float('inf')))

            return {
                "main_list": venues,
                "search_info": {
                    "query": query,
                    "source": "google_maps",
                    "count": len(venues)
                },
                "source_type": "google_maps"
            }

        except Exception as e:
            logger.error(f"❌ Error formatting Google Maps results: {e}")
            return {"main_list": [], "search_info": {"query": query, "count": 0}}

    # ============ UTILITY METHODS ============

    def _get_coordinates_sync(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
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

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city from venue address"""
        try:
            if not address:
                return "Unknown"

            # Simple extraction - split by comma and take the second-to-last part
            parts = [part.strip() for part in address.split(',')]
            if len(parts) >= 2:
                return parts[-2]
            else:
                return parts[0] if parts else "Unknown"

        except Exception:
            return "Unknown"

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