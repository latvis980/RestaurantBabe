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

                # Format personal notes message using LocationFilterEvaluator
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
                # NO DATABASE RESULTS - Send "asking around" message and proceed with Google Maps
                logger.info("📭 No sufficient database results, sending 'asking around' message and searching Google Maps")

                return await self._send_asking_around_and_search_maps(
                    coordinates, query, location_desc, start_time, cancel_check_fn
                )

        except Exception as e:
            logger.error(f"❌ Error in location search: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_response(f"Search failed: {str(e)}")

    # ============ DATABASE METHODS ============

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
            nearby_restaurants = self._get_nearby_restaurants_sync(coordinates, cancel_check_fn)

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

            # Use LocationFilterEvaluator for database-specific logic
            result = self.filter_evaluator.filter_and_evaluate(
                restaurants=nearby_restaurants,
                query=query,
                location_description=location_description
            )

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"🎯 Database filter result: {result['selected_count']} selected, send_immediately: {result['send_immediately']}")

            return result

        except Exception as e:
            logger.error(f"❌ Error in database check and filter: {e}")
            return {
                "database_sufficient": False,
                "filtered_restaurants": [],
                "send_immediately": False,
                "reasoning": f"Error checking database: {str(e)}"
            }

    # ============ RESTAURANT FILTERING (UNIFIED) ============

    def _get_nearby_restaurants_sync(
        self, 
        coordinates: Tuple[float, float], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Get nearby restaurants from database"""
        try:
            # Use the new database service for coordinate-based search
            nearby_restaurants = self.db_service.get_restaurants_by_coordinates(
                center=coordinates,
                radius_km=self.db_search_radius,
                limit=50
            )

            logger.info(f"📊 Database query returned {len(nearby_restaurants)} restaurants")

            if nearby_restaurants:
                logger.info(f"📝 Sample restaurants found:")
                for i, restaurant in enumerate(nearby_restaurants[:3]):
                    name = restaurant.get('name', 'Unknown')
                    distance = restaurant.get('distance_km', 'Unknown')
                    cuisine = restaurant.get('cuisine_tags', 'No cuisine info')
                    logger.info(f"  {i+1}. {name} ({distance}km) - {cuisine}")

            return nearby_restaurants

        except Exception as e:
            logger.error(f"❌ Error getting nearby restaurants: {e}")
            return []

    def _filter_restaurants_sync(
        self, 
        restaurants: List[Dict[str, Any]], 
        query: str, 
        location_desc: str,
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Filter restaurants using AI - STRICTER filtering for Google Maps results"""
        try:
            if not restaurants:
                return []

            # Create restaurant text for AI analysis
            restaurants_text = ""
            for i, restaurant in enumerate(restaurants):
                name = restaurant.get('name', 'Unknown')
                cuisine = restaurant.get('cuisine_tags', [])
                cuisine_str = ', '.join(cuisine) if cuisine else 'No cuisine info'
                description = restaurant.get('raw_description', 'No description')[:200]
                distance = restaurant.get('distance_km', 'Unknown')
                rating = restaurant.get('rating', 'No rating')

                restaurants_text += f"{i+1}. ID: {restaurant.get('id', i)} | {name}"
                if distance != 'Unknown':
                    restaurants_text += f" ({distance}km)"
                if rating != 'No rating':
                    restaurants_text += f" - ⭐{rating}"
                restaurants_text += f"\n   Cuisine: {cuisine_str}\n"
                restaurants_text += f"   Description: {description}...\n\n"

            # Enhanced prompt for Google Maps results (stricter quality filtering)
            enhanced_prompt = ChatPromptTemplate.from_template("""
USER QUERY: {{raw_query}}
LOCATION: {{destination}}

You are analyzing restaurants from Google Maps to find the BEST matches for this location-based query.

RESTAURANT LIST:
{{restaurants_text}}

TASK: Select only HIGH-QUALITY restaurants that strongly match the user's query intent.

STRICTER CRITERIA for Google Maps results:
- Must have strong cuisine type relevance (not just loosely related)
- Good ratings (prefer 4.0+ if available)
- Quality descriptions that show it's a good establishment
- Special features that match the query (natural wine, good coffee, etc.)

Be MORE SELECTIVE than for database results. Only include restaurants you'd confidently recommend.

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

Focus on quality over quantity. Only include clearly good matches.
""")

            # Use enhanced prompt for Google Maps results
            response = self.ai_model.invoke(
                enhanced_prompt.format(
                    raw_query=query,
                    destination=location_desc,
                    restaurants_text=restaurants_text
                )
            )

            # Parse AI response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                ai_result = json.loads(content)
                selected_data = ai_result.get("selected_restaurants", [])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse AI filtering response: {content}")
                return []

            # Map selected IDs back to full restaurant objects
            restaurant_lookup = {str(r.get('id', i)): r for i, r in enumerate(restaurants)}
            selected_restaurants = []

            for selection in selected_data:
                restaurant_id = str(selection.get('id', ''))
                if restaurant_id in restaurant_lookup:
                    restaurant = restaurant_lookup[restaurant_id].copy()
                    restaurant['_relevance_score'] = selection.get('relevance_score', 0)
                    restaurant['_match_reasoning'] = selection.get('reasoning', '')
                    selected_restaurants.append(restaurant)

            logger.info(f"🎯 Strict AI filtering selected {len(selected_restaurants)} from {len(restaurants)} Google Maps results")
            return selected_restaurants

        except Exception as e:
            logger.error(f"❌ Error in AI filtering: {e}")
            return []

    def _get_full_restaurant_details_sync(
        self, 
        restaurants: List[Dict[str, Any]], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Get full restaurant details from database (LEGACY METHOD)"""
        try:
            # The restaurants from get_restaurants_by_coordinates already have full details
            logger.info(f"📋 Restaurant data already contains full details for {len(restaurants)} restaurants")
            return restaurants

        except Exception as e:
            logger.error(f"❌ Error getting full restaurant details: {e}")
            return []

    # ============ GOOGLE MAPS SEARCH WITH "ASKING AROUND" MESSAGE ============

    async def _send_asking_around_and_search_maps(
        self,
        coordinates: Tuple[float, float],
        query: str,
        location_desc: str,
        start_time: float,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Send 'asking around' message, then search Google Maps with quality filtering"""
        try:
            # Return "asking around" message immediately, then trigger background search
            asking_around_message = (
                f"🤔 I don't see many good matches in my notes for {location_desc}.\n\n"
                f"💬 <b>Let me ask around for you...</b>\n\n"
                f"⏱ This might take a minute while I search for the best {query} nearby."
            )

            logger.info("📱 Sending 'asking around' message, starting background Google Maps search")

            # This will be sent immediately to user
            return {
                "success": True,
                "source": "asking_around",
                "message": asking_around_message,
                "restaurants": [],
                "total_count": 0,
                "location": {"lat": coordinates[0], "lng": coordinates[1], "description": location_desc},
                "query": query,
                "processing_time": time.time() - start_time,
                "background_search_triggered": True,  # Flag for telegram_bot to trigger background search
                "search_params": {
                    "coordinates": coordinates,
                    "query": query,
                    "location_desc": location_desc,
                    "start_time": start_time
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in asking around flow: {e}")
            return self._create_error_response(f"Error starting search: {str(e)}")

    async def _background_google_maps_search(
        self,
        coordinates: Tuple[float, float],
        query: str,
        location_desc: str,
        start_time: float,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Background Google Maps search with quality filtering for location searches"""
        try:
            logger.info("🗺️ Starting background Google Maps search...")

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

            # Quality filtering for Google Maps results (stricter than database)
            if verified_venues:
                filtered_venues = self._filter_restaurants_sync(
                    verified_venues, query, location_desc, cancel_check_fn
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                logger.info(f"🎯 Google Maps quality filtering: {len(filtered_venues)} from {len(verified_venues)} venues")
            else:
                filtered_venues = []

            # Format results
            formatted_response = self._format_location_results_sync(
                filtered_venues, coordinates, query, "google_maps_verified"
            )

            total_time = time.time() - start_time
            formatted_response['processing_time'] = total_time

            logger.info(f"✅ Background location search completed in {total_time:.1f}s with {len(filtered_venues)} results")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error in background Google Maps search: {e}")
            return self._create_error_response(f"Background search failed: {str(e)}")

    # ============ LEGACY GOOGLE MAPS FALLBACK (for compatibility) ============

    async def _fallback_to_google_maps_search(
        self,
        coordinates: Tuple[float, float],
        query: str,
        location_desc: str,
        start_time: float,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Legacy fallback method - now calls background search"""
        return await self._background_google_maps_search(
            coordinates, query, location_desc, start_time, cancel_check_fn
        )

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

    def _format_personal_notes_message(
        self, 
        restaurants: List[Dict[str, Any]],
        query: str,
        location_description: str = "your location"
    ) -> str:
        """Format the 'personal notes' message for immediate sending"""
        try:
            if not restaurants:
                return "🤔 I don't have any restaurants from my notes for this location."

            count = len(restaurants)

            # Header message
            header = f"📝 <b>Here are {count} restaurants from my notes near {location_description}:</b>\n\n"

            # List restaurants
            restaurant_list = ""
            for i, restaurant in enumerate(restaurants[:8]):  # Limit to 8 for readability
                name = restaurant.get('name', 'Unknown')
                distance = restaurant.get('distance_km', '?')
                cuisine = restaurant.get('cuisine_tags', [])
                cuisine_str = ', '.join(cuisine[:2]) if cuisine else ''  # Max 2 cuisine tags

                restaurant_list += f"🍽 <b>{name}</b> ({distance}km)"
                if cuisine_str:
                    restaurant_list += f" - <i>{cuisine_str}</i>"
                restaurant_list += "\n"

            # Footer message
            footer = (
                f"\n💡 <b>These are from my personal notes.</b> "
                f"Let me know if you want me to make some calls and search for more good addresses!"
            )

            return header + restaurant_list + footer

        except Exception as e:
            logger.error(f"❌ Error formatting message: {e}")
            return "📝 Found some restaurants from my notes, but had trouble formatting the list."

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create cancelled response"""
        return {
            "success": False,
            "cancelled": True,
            "restaurants": [],
            "total_count": 0,
            "message": "Search was cancelled"
        }