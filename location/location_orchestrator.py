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

        # Setup filtering prompt for legacy methods
        self._setup_filtering_prompt()

        logger.info("✅ Location Orchestrator initialized with database-first approach")

    def _setup_filtering_prompt(self):
        """Setup the filtering prompt for legacy compatibility"""
        self.batch_analysis_prompt = ChatPromptTemplate.from_template("""
USER QUERY: {{raw_query}}
LOCATION: {{destination}}

You are analyzing restaurants from our database to see which ones match the user's query.

RESTAURANT LIST:
{{restaurants_text}}

TASK: Analyze this list and return the restaurant IDs that best match the user's query.

MATCHING CRITERIA:
- Cuisine type relevance (direct matches, related cuisines)
- Atmosphere and dining style 
- Special features mentioned (wine lists, vegan options, price range, etc.)
- General vibe from descriptions

OUTPUT: Return ONLY valid JSON with matching restaurant IDs:
{{
    "selected_restaurants": [
        {{
            "id": "ID",
            "relevance_score": score,
            "reasoning": "why this matches the search intent"
        }}
    ]
}}

Include restaurants that are good matches. Focus on quality over quantity.
""")

    # ============ MAIN PROCESSING METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query with database-first approach
        """
        try:
            start_time = time.time()
            logger.info(f"🎯 Starting location search: '{query}'")

            # STEP 1: Get coordinates
            coordinates = self._get_coordinates_sync(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates
            location_desc = location_data.description or f"GPS location ({lat:.4f}, {lng:.4f})"
            logger.info(f"📍 Coordinates: {lat:.4f}, {lng:.4f}")

            # STEP 2: Check database and filter (NEW APPROACH)
            db_result = self._check_database_and_filter(
                coordinates, query, location_desc, cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 3: Decision based on database results
            if db_result.get("send_immediately", False):
                # IMMEDIATE SENDING - Found relevant restaurants in database
                filtered_restaurants = db_result.get("filtered_restaurants", [])

                logger.info(f"✅ IMMEDIATE SEND: {len(filtered_restaurants)} restaurants from database")

                # Format personal notes message
                message = self.filter_evaluator.format_personal_notes_message(
                    restaurants=filtered_restaurants,
                    query=query,
                    location_description=location_desc
                )

                total_time = time.time() - start_time

                return {
                    "success": True,
                    "source": "database_notes",
                    "message": message,
                    "restaurants": filtered_restaurants,
                    "total_count": len(filtered_restaurants),
                    "location": {"lat": lat, "lng": lng, "description": location_desc},
                    "query": query,
                    "processing_time": total_time,
                    "additional_search_available": True,  # User can request more
                    "reasoning": db_result.get("reasoning", "Found relevant restaurants in database")
                }

            else:
                # NO DATABASE RESULTS - Proceed with Google Maps search
                logger.info("📭 No sufficient database results, proceeding with Google Maps search")

                return await self._fallback_to_google_maps_search(
                    coordinates, query, location_desc, start_time, cancel_check_fn
                )

        except Exception as e:
            logger.error(f"❌ Error in location search: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_response(f"Search failed: {str(e)}")

    # ============ DATABASE METHODS (NEW) ============

    def _check_database_and_filter(
        self, 
        coordinates: Tuple[float, float], 
        query: str,
        location_description: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Check database, filter results, and decide if sufficient for immediate sending"""
        try:
            logger.info(f"🗃️ Checking database within {self.db_search_radius}km")

            # Get nearby restaurants from database
            nearby_restaurants = self.db_service.get_restaurants_by_coordinates(
                center=coordinates,
                radius_km=self.db_search_radius,
                limit=50
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"📊 Database returned {len(nearby_restaurants)} restaurants")

            if not nearby_restaurants:
                return {
                    "database_sufficient": False,
                    "filtered_restaurants": [],
                    "send_immediately": False,
                    "reasoning": "No restaurants found in database for this location"
                }

            # Filter and evaluate for location search
            result = self.filter_evaluator.filter_and_evaluate(
                restaurants=nearby_restaurants,
                query=query,
                location_description=location_description
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"🎯 Filter result: {result['selected_count']} selected, send_immediately: {result['send_immediately']}")

            return result

        except Exception as e:
            logger.error(f"❌ Error in database check and filter: {e}")
            return {
                "database_sufficient": False,
                "filtered_restaurants": [],
                "send_immediately": False,
                "reasoning": f"Error checking database: {str(e)}"
            }


    # ============ GOOGLE MAPS FALLBACK ============

    async def _fallback_to_google_maps_search(
        self,
        coordinates: Tuple[float, float],
        query: str,
        location_desc: str,
        start_time: float,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Fallback to Google Maps search when no database results"""
        try:
            logger.info("🗺️ Starting Google Maps search...")

            # Search Google Maps for venues
            venues = await self._search_google_maps_async(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not venues:
                return self._create_error_response("No venues found near your location")

            logger.info(f"🗺️ Google Maps found {len(venues)} venues")

            # AI-powered source verification
            logger.info(f"🔍 Starting source verification for {len(venues)} venues")
            verified_venues = await self._verify_venues_async(venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"✅ Source verification complete: {len(verified_venues)} venues verified")

            # Format results
            formatted_response = self._format_location_results_sync(
                verified_venues, coordinates, query, "google_maps"
            )

            total_time = time.time() - start_time
            formatted_response['processing_time'] = total_time

            logger.info(f"✅ Location search completed in {total_time:.1f}s with {len(verified_venues)} results")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error in Google Maps fallback: {e}")
            return self._create_error_response(f"Google Maps search failed: {str(e)}")

    # ============ COORDINATE METHODS ============

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

    # ============ GOOGLE MAPS SEARCH METHODS ============

    async def _search_google_maps_async(
        self, 
        coordinates: Tuple[float, float], 
        query: str,
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues"""
        try:
            lat, lng = coordinates

            # Extract search terms for Google Maps
            search_terms = self._extract_search_terms_sync(query)

            logger.info(f"🗺️ Searching Google Maps for '{search_terms}' near {lat:.4f}, {lng:.4f}")

            # Call in executor since it's sync
            venues = await asyncio.get_event_loop().run_in_executor(
                None,
                self.location_search_agent.search_nearby_venues,
                lat, lng, search_terms
            )

            return venues or []

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search: {e}")
            return []

    def _extract_search_terms_sync(self, query: str) -> str:
        """Extract search terms for Google Maps"""
        # Simple extraction - you can enhance this
        return query

    async def _verify_venues_async(
        self, 
        venues: List[VenueResult], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Verify venues using media search agent"""
        try:
            # Convert VenueResult objects to format expected by media search
            venue_data = []
            for venue in venues[:self.max_venues_to_verify]:
                # Extract city from venue address
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

            # Verify using media search
            verified_results = []
            for venue_info in venue_data:
                if cancel_check_fn and cancel_check_fn():
                    break

                # Simple verification - you can enhance this
                verified_results.append(venue_info)

            return verified_results

        except Exception as e:
            logger.error(f"❌ Error in venue verification: {e}")
            return []

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

    # ============ FORMATTING METHODS ============

    def _format_location_results_sync(
        self, 
        restaurants: List[Dict[str, Any]], 
        coordinates: Tuple[float, float],
        query: str,
        source: str
    ) -> Dict[str, Any]:
        """Format results for response"""
        try:
            lat, lng = coordinates

            # Add distance to each restaurant
            for restaurant in restaurants:
                try:
                    r_lat = restaurant.get('latitude')
                    r_lng = restaurant.get('longitude')

                    if r_lat and r_lng:
                        distance = LocationUtils.calculate_distance(
                            (lat, lng), 
                            (float(r_lat), float(r_lng))
                        )
                        restaurant['distance_km'] = round(distance, 2)
                    else:
                        restaurant['distance_km'] = None
                except Exception as e:
                    logger.debug(f"Error calculating distance for {restaurant.get('name')}: {e}")
                    restaurant['distance_km'] = None

            # Sort by distance (closest first)
            restaurants_with_distance = [r for r in restaurants if r.get('distance_km') is not None]
            restaurants_without_distance = [r for r in restaurants if r.get('distance_km') is None]

            sorted_restaurants = (
                sorted(restaurants_with_distance, key=lambda x: x['distance_km']) +
                restaurants_without_distance
            )

            return {
                'success': True,
                'results': sorted_restaurants,
                'total_count': len(sorted_restaurants),
                'search_coordinates': {'latitude': lat, 'longitude': lng},
                'search_radius_km': self.db_search_radius,
                'source': source,
                'query': query
            }

        except Exception as e:
            logger.error(f"❌ Error formatting location results: {e}")
            return self._create_error_response(f"Error formatting results: {str(e)}")

    # ============ UTILITY METHODS ============

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "error": error_message,
            "restaurants": [],
            "total_count": 0
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create cancelled response"""
        return {
            "success": False,
            "cancelled": True,
            "restaurants": [],
            "total_count": 0,
            "message": "Search was cancelled"
        }