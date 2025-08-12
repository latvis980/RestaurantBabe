# location/location_orchestrator.py
"""
Location Search Orchestrator - FIXED ALL TYPE ERRORS

Fixed issues:
1. Used actual database methods that exist
2. Fixed return types and coordinate validation  
3. Fixed async/sync consistency
4. Fixed response.content type handling
5. Fixed search_nearby_venues return type
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
from formatters.telegram_formatter import TelegramFormatter
from location.database_service import LocationDatabaseService

# Import AI components for restaurant filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Location search orchestrator with all type errors fixed
    """

    def __init__(self, config):
        self.config = config

        # Initialize existing components
        self.location_search_agent = LocationSearchAgent(config)
        # Initialize media search agent (replaces source mapping)
        self.media_search_agent = MediaSearchAgent(config)
        self.telegram_formatter = TelegramFormatter()
        self.db_service = LocationDatabaseService(config)

        # Initialize AI for restaurant filtering
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 2.0)
        self.min_db_matches = getattr(config, 'MIN_DB_MATCHES_REQUIRED', 3)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        # Setup filtering prompt
        self._setup_filtering_prompt()

        logger.info("✅ Location Orchestrator initialized")

    def _setup_filtering_prompt(self):
        """Setup the filtering prompt"""
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
- Quality indicators from descriptions

SCORING:
- Perfect match (9-10): Direct cuisine match + special features match
- High relevance (7-8): Strong cuisine or feature match
- Moderate relevance (5-6): Some connection to query
- Low relevance (3-4): Weak connection
- Not relevant (0-2): No meaningful connection

Return ONLY valid JSON with the top matches:
{{
    "selected_restaurants": [
        {{
            "id": "restaurant_id",
            "name": "restaurant_name", 
            "relevance_score": 8,
            "reasoning": "why this restaurant matches"
        }}
    ],
    "total_analyzed": number_of_restaurants_analyzed,
    "query_analysis": "brief analysis of what user is looking for"
}}

IMPORTANT: Only include restaurants with score 5 or higher. Prioritize quality over quantity.
""")

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process a location-based restaurant query
        """
        try:
            start_time = time.time()
            logger.info(f"🎯 Starting location search: '{query}'")

            # STEP 1: Get GPS coordinates (SYNC)
            coordinates = self._get_coordinates_sync(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates
            logger.info(f"📍 Coordinates: {lat:.4f}, {lng:.4f}")

            # STEP 2: Check database for nearby restaurants (SYNC)  
            logger.info(f"🗃️ STEP 2: Checking database within {self.db_search_radius}km")
            nearby_restaurants = self._get_nearby_restaurants_sync(coordinates, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"📊 Database returned {len(nearby_restaurants)} restaurants")

            if nearby_restaurants:
                logger.info(f"✅ Found {len(nearby_restaurants)} nearby restaurants in database")

                # STEP 3: AI filtering (SYNC)
                logger.info(f"🧠 STEP 3: AI filtering restaurants for query: '{query}'")
                filtered_restaurants = self._filter_restaurants_sync(
                    nearby_restaurants, query, location_data.description or "GPS location", cancel_check_fn
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                logger.info(f"🎯 AI filtering result: {len(filtered_restaurants)} restaurants matched")

                if filtered_restaurants:
                    logger.info(f"✅ AI filtering found {len(filtered_restaurants)} matching restaurants")

                    # STEP 4: Get full details (SYNC)
                    logger.info(f"📋 STEP 4: Getting full details for matched restaurants")
                    detailed_restaurants = self._get_full_restaurant_details_sync(filtered_restaurants, cancel_check_fn)

                    if cancel_check_fn and cancel_check_fn():
                        return self._create_cancelled_response()

                    logger.info(f"📄 Retrieved details for {len(detailed_restaurants)} restaurants")

                    # STEP 5: Format and return (SYNC)
                    logger.info(f"📝 STEP 5: Formatting database results")
                    formatted_response = self._format_location_results_sync(
                        detailed_restaurants, coordinates, query, "database_filtered"
                    )

                    total_time = time.time() - start_time
                    formatted_response['processing_time'] = total_time

                    logger.info(f"✅ Database location search completed in {total_time:.1f}s with {len(detailed_restaurants)} filtered results")
                    return formatted_response
                else:
                    logger.info(f"❌ AI filtering found no matches from {len(nearby_restaurants)} database restaurants")
            else:
                logger.info("📭 No restaurants found in database within radius")

            # STEP 6: Fallback to Google Maps + verification (ASYNC)
            logger.info("🌐 STEP 6: No database matches found, proceeding with Google Maps + source verification")

            venues = await self._search_google_maps_async(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not venues:
                return self._create_error_response("No venues found near your location")

            logger.info(f"🗺️ Google Maps found {len(venues)} venues")

            # AI-powered source verification (ASYNC)
            logger.info(f"🔍 Starting source verification for {len(venues)} venues")
            verified_venues = await self._verify_venues_async(venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            logger.info(f"✅ Source verification complete: {len(verified_venues)} venues verified")

            # Format results (SYNC)
            formatted_response = self._format_location_results_sync(
                verified_venues, coordinates, query, "google_maps"
            )

            total_time = time.time() - start_time
            formatted_response['processing_time'] = total_time

            logger.info(f"✅ Location search completed in {total_time:.1f}s with {len(verified_venues)} results")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error in location search pipeline: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_response(f"Search failed: {str(e)}")

    # ============ SYNC METHODS (Database, AI, Formatting) ============

    def _get_coordinates_sync(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates from location data (SYNC) - FIXED TYPES"""
        try:
            if location_data.location_type == "gps":
                # FIXED: Validate coordinates before returning
                lat = location_data.latitude
                lng = location_data.longitude

                if lat is not None and lng is not None:
                    return (float(lat), float(lng))
                else:
                    logger.error("GPS location data has None coordinates")
                    return None

            elif location_data.location_type == "description" and location_data.description:
                logger.info(f"🗺️ Geocoding location: {location_data.description}")

                # Use database geocoding method (SYNC)
                from utils.database import get_database
                db = get_database()

                coordinates = db.geocode_address(location_data.description)
                if coordinates:
                    return coordinates

                # Note: LocationUtils doesn't have geocode_location method
                logger.warning("No geocoding fallback available in LocationUtils")

            return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None

    def _get_nearby_restaurants_sync(
        self, 
        coordinates: Tuple[float, float], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Get nearby restaurants using LocationDatabaseService"""
        try:
            lat, lng = coordinates
            logger.info(f"🗃️ Querying database for restaurants within {self.db_search_radius}km of {lat:.4f}, {lng:.4f}")

            # Use the dedicated location database service
            nearby_restaurants = self.db_service.get_restaurants_by_coordinates(
                center=(lat, lng),
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
        """Filter restaurants using AI (SYNC) - FIXED RESPONSE HANDLING"""
        try:
            if not restaurants:
                return []

            # Create restaurant text for AI analysis
            restaurants_text = ""
            for i, restaurant in enumerate(restaurants):
                name = restaurant.get('name', 'Unknown')
                cuisine = restaurant.get('cuisine_tags', 'No cuisine info')
                description = restaurant.get('raw_description', 'No description')
                mention_count = restaurant.get('mention_count', 0)

                restaurants_text += f"{i+1}. ID: {restaurant.get('id')}\n"
                restaurants_text += f"   Name: {name}\n"
                restaurants_text += f"   Cuisine: {cuisine}\n"
                restaurants_text += f"   Description: {description[:200]}{'...' if len(description) > 200 else ''}\n"
                restaurants_text += f"   Mentions: {mention_count}\n\n"

            # AI analysis (SYNC)
            response = self.ai_model.invoke(
                self.batch_analysis_prompt.format(
                    raw_query=query,
                    destination=location_desc,
                    restaurants_text=restaurants_text
                )
            )

            # FIXED: Handle response.content properly (it's a string, not a list)
            content = response.content
            if isinstance(content, str):
                content = content.strip()
            else:
                logger.error(f"Unexpected response type: {type(content)}")
                content = str(content).strip()

            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse AI response
            import json
            analysis = json.loads(content)

            selected_restaurants = analysis.get('selected_restaurants', [])

            # Map selected IDs back to restaurant objects
            selected_ids = {str(r.get('id')) for r in selected_restaurants}
            filtered = [r for r in restaurants if str(r.get('id')) in selected_ids]

            logger.info(f"🎯 AI selected {len(filtered)} restaurants from {len(restaurants)} candidates")

            return filtered

        except Exception as e:
            logger.error(f"❌ Error in AI restaurant filtering: {e}")
            # Return top 5 by mention count as fallback
            sorted_restaurants = sorted(restaurants, key=lambda x: x.get('mention_count', 0), reverse=True)
            return sorted_restaurants[:5]

    def _get_full_restaurant_details_sync(
        self, 
        restaurants: List[Dict[str, Any]], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Get full restaurant details from database (SYNC) - USING ACTUAL METHODS"""
        try:
            # FIXED: The restaurants from get_restaurants_by_coordinates already have full details
            # No need to fetch them again
            logger.info(f"📋 Restaurant data already contains full details for {len(restaurants)} restaurants")
            return restaurants

        except Exception as e:
            logger.error(f"❌ Error getting full restaurant details: {e}")
            return []

    def _format_location_results_sync(
        self, 
        restaurants: List[Dict[str, Any]], 
        coordinates: Tuple[float, float],
        query: str,
        source: str
    ) -> Dict[str, Any]:
        """Format results for response (SYNC)"""
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

    # ============ ASYNC METHODS (Google Maps, Web Search) ============

    async def _search_google_maps_async(
        self, 
        coordinates: Tuple[float, float], 
        query: str,
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues (ASYNC) - FIXED RETURN TYPE"""
        try:
            lat, lng = coordinates

            # Extract search terms for Google Maps
            search_terms = self._extract_search_terms_sync(query)

            logger.info(f"🗺️ Searching Google Maps for '{search_terms}' near {lat:.4f}, {lng:.4f}")

            # FIXED: search_nearby_venues returns List[VenueResult], not awaitable
            # Call it in executor since it's sync
            venues = await asyncio.get_event_loop().run_in_executor(
                None,
                self.location_search_agent.search_nearby_venues,
                lat, lng, search_terms
            )

            return venues or []

        except Exception as e:
            logger.error(f"❌ Error in Google Maps search: {e}")
            return []

    async def _verify_venues_async(
        self, 
        venues: List[VenueResult], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Verify venues using media search agent (ASYNC) - UPDATED FOR MEDIA SEARCH"""
        try:
            # Convert VenueResult objects to format expected by media search
            venue_data = []
            for venue in venues[:self.max_venues_to_verify]:
                # Extract city from venue address
                city = self._extract_city_from_address(venue.address)

                venue_data.append({
                    'name': venue.name,
                    'city': city,
                    'type': self._determine_venue_type_from_types(venue.types)
                })

            logger.info(f"🔍 Starting media search for {len(venue_data)} venues")

            # Batch media search
            media_results = await self.media_search_agent.batch_search_venues(venue_data)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Merge media results with venue data
            verified_venues = []
            for i, venue in enumerate(venues[:self.max_venues_to_verify]):
                media_result = media_results[i] if i < len(media_results) else None

                # Convert VenueResult to dict and add media verification
                venue_dict = {
                    'name': venue.name,
                    'address': venue.address,
                    'latitude': venue.latitude,
                    'longitude': venue.longitude,
                    'rating': venue.rating,
                    'user_ratings_total': venue.user_ratings_total,
                    'price_level': venue.price_level,
                    'types': venue.types,
                    'google_maps_url': venue.google_maps_url,
                    'source': 'google_maps'
                }

                # Add media coverage data
                if media_result and media_result.get('has_coverage'):
                    venue_dict['media_coverage'] = media_result.get('media_coverage', {})
                    venue_dict['verified'] = True
                    venue_dict['coverage_confidence'] = media_result.get('media_coverage', {}).get('confidence', 0.0)
                    venue_dict['reputable_sources'] = media_result.get('media_coverage', {}).get('reputable_sources', [])
                else:
                    venue_dict['media_coverage'] = {'has_media_coverage': False}
                    venue_dict['verified'] = False
                    venue_dict['coverage_confidence'] = 0.0
                    venue_dict['reputable_sources'] = []

                verified_venues.append(venue_dict)

            # Log summary
            coverage_summary = self.media_search_agent.get_coverage_summary(media_results)
            logger.info(f"📊 Media coverage summary: {coverage_summary['venues_with_coverage']}/{coverage_summary['total_venues_searched']} venues have professional coverage")

            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error in media verification: {e}")
            return []

    # ============ UTILITY METHODS ============

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city from venue address"""
        try:
            # Simple approach: get the last part before country/postal code
            # Address format is usually: "Street, Neighborhood, City, Country"
            parts = address.split(',')
            if len(parts) >= 2:
                # Take second-to-last part as likely city
                city = parts[-2].strip()
                # Remove numbers/postal codes
                import re
                city = re.sub(r'\d+', '', city).strip()
                return city if city else "Unknown"
            return "Unknown"
        except:
            return "Unknown"

    def _determine_venue_type_from_types(self, types: List[str]) -> str:
        """Convert Google Places types to venue type"""
        if not types:
            return "restaurant"

        type_mapping = {
            'bar': 'bar',
            'night_club': 'bar', 
            'cafe': 'cafe',
            'bakery': 'bakery',
            'meal_takeaway': 'restaurant',
            'meal_delivery': 'restaurant',
            'restaurant': 'restaurant'
        }

        for gtype in types:
            if gtype in type_mapping:
                return type_mapping[gtype]

        return "restaurant"

    def _extract_search_terms_sync(self, query: str) -> str:
        """Extract search terms from query (SYNC)"""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            extraction_prompt = ChatPromptTemplate.from_template("""
Extract the cuisine type or restaurant category from this location-based query. If the description is too vague, return a general term like "restaurant". 

Query: {query}

Return ONLY the cuisine/restaurant type (2-4 words max), nothing else.
""")

            # Use AI to extract search terms (SYNC)
            response = self.ai_model.invoke(extraction_prompt.format(query=query))

            # FIXED: Handle response properly
            if hasattr(response, 'content'):
                extracted = response.content.strip().lower() if isinstance(response.content, str) else str(response.content).strip().lower()
            else:
                extracted = str(response).strip().lower()

            # Clean up the response
            if len(extracted) > 50:  # Too long, fallback
                extracted = "restaurant"

            logger.info(f"🔍 Extracted search terms: '{extracted}' from query: '{query}'")
            return extracted

        except Exception as e:
            logger.error(f"❌ Error extracting search terms: {e}")
            # Simple fallback
            if any(word in query.lower() for word in ['coffee', 'cafe']):
                return "cafe"
            elif any(word in query.lower() for word in ['bar', 'wine', 'cocktail']):
                return "bar"
            elif any(word in query.lower() for word in ['pizza']):
                return "pizza"
            else:
                return "restaurant"

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'success': False,
            'error': message,
            'results': [],
            'total_count': 0
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create cancelled response"""
        return {
            'success': False,
            'cancelled': True,
            'message': 'Search was cancelled',
            'results': [],
            'total_count': 0
        }