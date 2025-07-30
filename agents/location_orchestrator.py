# agents/location_orchestrator.py
"""
Location Search Orchestrator - MODIFIED VERSION

Updated to implement the same filtering algorithm as database_search_agent.py:
1. Get nearby restaurants from database (IDs + names + tags)
2. Create temp file with restaurant data  
3. Analyze in single API call using the same prompt
4. If no matches, do additional location-based search
5. Extract full details for matched restaurants
6. Format and send to client
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import json
import tempfile
import os

from utils.location_utils import LocationUtils, LocationPoint
from utils.telegram_location_handler import LocationData
from agents.location_search_agent import LocationSearchAgent, VenueResult
from agents.source_mapping_agent import SourceMappingAgent
from formatters.telegram_formatter import TelegramFormatter

# Import AI components for restaurant filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Orchestrates the complete location-based restaurant search pipeline
    WITH OPTIMIZED FILTERING using same algorithm as database_search_agent.py
    """

    def __init__(self, config):
        self.config = config

        # Initialize existing components
        self.location_search_agent = LocationSearchAgent(config)
        self.source_mapping_agent = SourceMappingAgent(config)
        self.telegram_formatter = TelegramFormatter()

        # NEW: Initialize AI for restaurant filtering (same as database_search_agent.py)
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,  # Use GPT-4o as requested
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 2.0)
        self.min_db_matches = getattr(config, 'MIN_DB_MATCHES_REQUIRED', 3)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        # NEW: Setup filtering prompt (same as database_search_agent.py)
        self._setup_filtering_prompt()

        logger.info("✅ Location Orchestrator initialized with AI filtering")

    def _setup_filtering_prompt(self):
        """Setup the same filtering prompt as database_search_agent.py"""

        # Same batch analysis prompt from database_search_agent.py
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
        MODIFIED: Now includes AI filtering step before returning results

        Args:
            query: User's search query (e.g. "italian restaurants")
            location_data: Location information (GPS coordinates or description)
            cancel_check_fn: Function to check if operation should be cancelled
        """
        try:
            start_time = time.time()
            logger.info(f"🎯 Starting MODIFIED location search: '{query}'")

            # STEP 1: Get GPS coordinates
            coordinates = await self._get_coordinates(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates
            logger.info(f"📍 Coordinates: {lat:.4f}, {lng:.4f}")

            # STEP 2: Get nearby restaurants from database (IDs + names + tags only)
            nearby_restaurants = await self._get_nearby_restaurants_basic_data(coordinates, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if nearby_restaurants:
                logger.info(f"🗃️ Found {len(nearby_restaurants)} nearby restaurants in database")

                # STEP 3: NEW - Analyze restaurants using AI (same algorithm as database_search_agent.py)
                filtered_restaurants = await self._filter_restaurants_with_ai(
                    nearby_restaurants, query, location_data.description or "GPS location", cancel_check_fn
                )

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                if filtered_restaurants:
                    logger.info(f"✅ AI filtering found {len(filtered_restaurants)} matching restaurants")

                    # STEP 4: Extract full details for matched restaurants
                    detailed_restaurants = await self._get_full_restaurant_details(filtered_restaurants, cancel_check_fn)

                    if cancel_check_fn and cancel_check_fn():
                        return self._create_cancelled_response()

                    # STEP 5: Format and return results
                    formatted_response = await self._format_location_results(
                        detailed_restaurants, coordinates, query, "database_filtered", cancel_check_fn
                    )

                    total_time = time.time() - start_time
                    formatted_response['processing_time'] = total_time

                    logger.info(f"✅ Database location search completed in {total_time:.1f}s with {len(detailed_restaurants)} filtered results")
                    return formatted_response

            # STEP 6: No database matches - do additional location-based search (existing implementation)
            logger.info("🌐 No database matches found, proceeding with location-based search")

            # Search Google Maps for venues near coordinates
            venues = await self._search_google_maps(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            if not venues:
                return self._create_error_response("No venues found near your location")

            # AI-powered source mapping and verification
            verified_venues = await self._verify_venues_with_sources(venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # Format results for Telegram
            formatted_response = await self._format_location_results(
                verified_venues, coordinates, query, "google_maps", cancel_check_fn
            )

            # Add timing information
            total_time = time.time() - start_time
            formatted_response['processing_time'] = total_time

            logger.info(f"✅ Location search completed in {total_time:.1f}s with {len(verified_venues)} results")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error in location search pipeline: {e}")
            return self._create_error_response(f"Search failed: {str(e)}")

    async def _get_nearby_restaurants_basic_data(
        self, 
        coordinates: Tuple[float, float], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """
        NEW METHOD: Get nearby restaurants with basic data only (IDs, names, tags)
        Same as step 1 in the desired flow
        """
        try:
            from utils.database import get_database
            db = get_database()

            lat, lng = coordinates
            logger.info(f"🗃️ Checking database for restaurants within {self.db_search_radius}km")

            # Get nearby restaurants with basic data only
            nearby_restaurants = db.get_restaurants_by_proximity(
                latitude=lat,
                longitude=lng,
                radius_km=self.db_search_radius,
                limit=50,  # Reasonable limit for processing
                fields=['id', 'name', 'cuisine_tags', 'mention_count', 'raw_description']  # Basic fields only
            )

            logger.info(f"📊 Found {len(nearby_restaurants)} restaurants in database within {self.db_search_radius}km")
            return nearby_restaurants

        except Exception as e:
            logger.error(f"❌ Error getting nearby restaurants: {e}")
            return []

    async def _filter_restaurants_with_ai(
        self, 
        restaurants: List[Dict[str, Any]], 
        raw_query: str, 
        destination: str,
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """
        NEW METHOD: Filter restaurants using AI analysis (same algorithm as database_search_agent.py)
        Steps 2-3 in the desired flow: create temp file, analyze in single API call
        """
        try:
            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"🧠 Filtering {len(restaurants)} restaurants with AI (single API call)")

            # Step 2: Gather list into temp file (in memory for efficiency)
            restaurants_text = self._compile_restaurants_for_analysis(restaurants)

            # Step 3: Analyze in single API call (same prompt as database_search_agent.py)
            chain = self.batch_analysis_prompt | self.ai_model

            response = chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "restaurants_text": restaurants_text
            })

            # Parse the response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            analysis_result = json.loads(content)

            # Map selected restaurant IDs back to restaurant data
            selected_restaurants = self._map_selected_restaurants(
                analysis_result.get("selected_restaurants", []),
                restaurants
            )

            logger.info(f"✅ AI filtering complete: {len(selected_restaurants)} restaurants selected")
            return selected_restaurants

        except Exception as e:
            logger.error(f"❌ Error in AI restaurant filtering: {e}")
            # Return empty list to trigger location-based search
            return []

    def _compile_restaurants_for_analysis(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Compile restaurant data for AI analysis (same format as database_search_agent.py)
        """
        compiled_text = []

        for restaurant in restaurants:
            restaurant_id = restaurant.get('id', 'unknown')
            name = restaurant.get('name', 'Unknown')
            cuisine_tags = ', '.join(restaurant.get('cuisine_tags', []))
            description = restaurant.get('raw_description', '')[:300]  # Truncate to save tokens
            mention_count = restaurant.get('mention_count', 1)

            # Format for analysis (same format as database_search_agent.py)
            restaurant_entry = f"ID: {restaurant_id} | {name} | Tags: {cuisine_tags} | Mentions: {mention_count} | Desc: {description}"
            compiled_text.append(restaurant_entry)

        return "\n".join(compiled_text)

    def _map_selected_restaurants(
        self, 
        selected_data: List[Dict[str, Any]], 
        all_restaurants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map AI-selected restaurant IDs back to restaurant data (same as database_search_agent.py)
        """
        selected_restaurants = []

        # Create lookup dict for fast access
        restaurant_lookup = {str(r.get('id')): r for r in all_restaurants}

        for selection in selected_data:
            restaurant_id = str(selection.get('id', ''))

            if restaurant_id in restaurant_lookup:
                restaurant = restaurant_lookup[restaurant_id].copy()

                # Add AI analysis metadata
                restaurant['_ai_relevance_score'] = selection.get('relevance_score', 0)
                restaurant['_ai_reasoning'] = selection.get('reasoning', '')

                selected_restaurants.append(restaurant)
            else:
                logger.warning(f"⚠️ AI selected restaurant ID {restaurant_id} not found in original list")

        return selected_restaurants

    async def _get_full_restaurant_details(
        self, 
        filtered_restaurants: List[Dict[str, Any]], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """
        NEW METHOD: Extract full details for matched restaurants (step 5 in desired flow)
        """
        try:
            if cancel_check_fn and cancel_check_fn():
                return []

            from utils.database import get_database
            db = get_database()

            detailed_restaurants = []
            restaurant_ids = [str(r.get('id')) for r in filtered_restaurants]

            logger.info(f"📋 Getting full details for {len(restaurant_ids)} restaurants")

            # Get full restaurant details using IDs
            for restaurant_id in restaurant_ids:
                if cancel_check_fn and cancel_check_fn():
                    break

                try:
                    # Get full restaurant data including descriptions, addresses, sources
                    full_restaurant = db.get_restaurant_by_id(restaurant_id)
                    if full_restaurant:
                        # Preserve AI analysis metadata
                        filtered_restaurant = next(
                            (r for r in filtered_restaurants if str(r.get('id')) == restaurant_id), 
                            {}
                        )
                        full_restaurant['_ai_relevance_score'] = filtered_restaurant.get('_ai_relevance_score', 0)
                        full_restaurant['_ai_reasoning'] = filtered_restaurant.get('_ai_reasoning', '')

                        detailed_restaurants.append(full_restaurant)

                except Exception as e:
                    logger.error(f"❌ Error getting details for restaurant {restaurant_id}: {e}")
                    continue

            logger.info(f"✅ Retrieved full details for {len(detailed_restaurants)} restaurants")
            return detailed_restaurants

        except Exception as e:
            logger.error(f"❌ Error getting full restaurant details: {e}")
            return []

    # Keep all existing methods unchanged
    async def _get_coordinates(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates from location data"""
        try:
            if location_data.location_type == "gps":
                # Direct GPS coordinates
                return (location_data.latitude, location_data.longitude)

            elif location_data.location_type == "description" and location_data.description:
                # Geocode text description
                logger.info(f"🗺️ Geocoding location: {location_data.description}")

                # Use existing geocoding infrastructure if available
                from utils.database import get_database
                db = get_database()

                # Try to get coordinates for the location description
                coordinates = db.geocode_location(location_data.description)
                if coordinates:
                    return coordinates

                # Fallback to location utilities
                location_utils = LocationUtils()
                location_point = location_utils.geocode_location(location_data.description)

                if location_point:
                    return (location_point.latitude, location_point.longitude)

            return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None

    async def _search_google_maps(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues near coordinates"""
        try:
            logger.info(f"🗺️ Searching Google Maps for venues")

            lat, lng = coordinates

            # Determine venue type from query
            venue_type = self.location_search_agent.determine_venue_type(query)

            # Search for nearby venues
            venues = self.location_search_agent.search_nearby_venues(
                latitude=lat,
                longitude=lng,
                query=query,
                venue_type=venue_type
            )

            logger.info(f"🏪 Google Maps found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"❌ Error searching Google Maps: {e}")
            return []

    async def _verify_venues_with_sources(
        self, 
        venues: List[VenueResult], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Use AI to map sources and verify venue information"""
        try:
            logger.info(f"🔍 Starting AI-powered source verification for {len(venues)} venues")

            verified_venues = []

            for i, venue in enumerate(venues[:self.max_venues_to_verify]):
                if cancel_check_fn and cancel_check_fn():
                    break

                logger.debug(f"📰 Verifying venue {i+1}/{len(venues[:self.max_venues_to_verify])}: {venue.name}")

                # Use source mapping agent to find and verify sources
                source_result = await self.source_mapping_agent.map_venue_sources(venue)

                if source_result and source_result.get('sources'):
                    verified_venues.append({
                        'name': venue.name,
                        'address': venue.address,
                        'rating': venue.rating,
                        'price_level': venue.price_level,
                        'place_id': venue.place_id,
                        'location': {'lat': venue.latitude, 'lng': venue.longitude},
                        'sources': source_result['sources'],
                        'verification_confidence': source_result.get('confidence', 0.5)
                    })

            logger.info(f"✅ Venue verification complete: {len(verified_venues)} venues verified")
            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error verifying venues: {e}")
            return []

    async def _format_location_results(
        self, 
        results: List[Dict[str, Any]], 
        coordinates: Tuple[float, float], 
        query: str, 
        search_method: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Format results for Telegram"""
        try:
            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates

            # Create location info
            location_info = {
                'coordinates': {'lat': lat, 'lng': lng},
                'search_radius_km': self.db_search_radius,
                'search_method': search_method
            }

            # Format using Telegram formatter
            formatted_text = self.telegram_formatter.format_location_results(
                results, location_info, query
            )

            return {
                'success': True,
                'telegram_formatted_text': formatted_text,
                'results_count': len(results),
                'location_info': location_info,
                'search_method': search_method
            }

        except Exception as e:
            logger.error(f"❌ Error formatting location results: {e}")
            return self._create_error_response(f"Error formatting results: {str(e)}")

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'success': False,
            'telegram_formatted_text': f"❌ {message}",
            'error': message
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create cancelled response"""
        return {
            'success': False,
            'telegram_formatted_text': "🛑 Search was cancelled",
            'cancelled': True
        }