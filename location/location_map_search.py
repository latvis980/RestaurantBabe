# location/location_map_search.py
"""
Google Maps/Places Search Agent - Classic Integration Only

Uses classic Google Maps library for text-based restaurant search with
AI-optimized query generation and API key rotation support.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import googlemaps
from langsmith import traceable
from location.location_utils import LocationUtils
from formatters.google_links import build_google_maps_url


logger = logging.getLogger(__name__)


# FIXED: OpenAI import with proper v1.0+ syntax
try:
    from openai import OpenAI
    HAS_OPENAI = True
    logger.info("✅ OpenAI v1.0+ imported successfully")
except ImportError:
    logger.warning("⚠️  OpenAI not available")
    OpenAI = None
    HAS_OPENAI = False

@dataclass
class VenueSearchResult:
    """Structure for Google Maps search results"""
    place_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float
    business_status: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    google_reviews: List[Dict] = field(default_factory=list)
    search_source: str = "googlemaps"
    google_maps_url: str = ""

    def __post_init__(self):
        if not self.google_maps_url and self.place_id:
            self.google_maps_url = build_google_maps_url(self.place_id, self.name)

class LocationMapSearchAgent:
    """
    Google Maps search agent using classic integration

    Provides restaurant search using Google Maps library with text queries,
    AI query optimization, and automatic API key rotation.
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.rating_threshold = float(getattr(config, 'RATING_THRESHOLD', 4.3))
        self.search_radius_km = float(getattr(config, 'SEARCH_RADIUS_KM', 2.0))
        self.max_venues_to_search = int(getattr(config, 'MAX_VENUES_TO_SEARCH', 20))

        # OpenAI configuration - FIXED: proper v1.0+ client initialization
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.openai_client: Optional[Any] = None

        if HAS_OPENAI and OpenAI is not None:
            openai_key = getattr(config, 'OPENAI_API_KEY', None)
            if openai_key:
                try:
                    self.openai_client = OpenAI(api_key=openai_key)
                    logger.info("✅ OpenAI client initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                    self.openai_client = None

        # Initialize client
        self.gmaps: Optional[googlemaps.Client] = None
        self.api_usage = {"gmaps": 0}

        self._initialize_clients()

        # Comprehensive Google Places API place types for restaurants/food
        self.RESTAURANT_PLACE_TYPES = [
            # General food & dining
            "restaurant",
            "food", 
            "meal_takeaway",
            "meal_delivery",
            "establishment",

            # Specific restaurant types
            "american_restaurant",
            "bakery", 
            "bar",
            "barbecue_restaurant",
            "brazilian_restaurant",
            "breakfast_restaurant", 
            "brunch_restaurant",
            "cafe",
            "chinese_restaurant",
            "coffee_shop",
            "fast_food_restaurant",
            "french_restaurant",
            "greek_restaurant",
            "hamburger_restaurant",
            "ice_cream_shop",
            "indian_restaurant",
            "indonesian_restaurant",
            "italian_restaurant",
            "japanese_restaurant",
            "korean_restaurant",
            "lebanese_restaurant",
            "mediterranean_restaurant",
            "mexican_restaurant",
            "middle_eastern_restaurant",
            "pizza_restaurant",
            "ramen_restaurant",
            "sandwich_shop",
            "seafood_restaurant",
            "spanish_restaurant",
            "steak_house",
            "sushi_restaurant",
            "thai_restaurant",
            "turkish_restaurant",
            "vegan_restaurant",
            "vegetarian_restaurant",
            "vietnamese_restaurant",

            # Drinking establishments
            "night_club",
            "pub",
            "wine_bar",
            "cocktail_bar",

            # Food-related services
            "grocery_store",
            "supermarket",
            "convenience_store",
            "liquor_store",
            "food_delivery",
            "catering_service"
        ]

        logger.info("✅ LocationMapSearchAgent initialized:")
        logger.info(f"   - Rating threshold: {self.rating_threshold}")
        logger.info(f"   - Search radius: {self.search_radius_km}km") 
        logger.info(f"   - Has GoogleMaps client: {self.gmaps is not None}")

    def _initialize_clients(self):
        """Initialize Google Maps client with key rotation support"""
        # GoogleMaps library - uses regular API key
        api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)
        if api_key:
            try:
                self.gmaps = googlemaps.Client(key=api_key)
                logger.info("✅ GoogleMaps client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GoogleMaps client: {e}")

        # Secondary GoogleMaps client for rotation
        api_key_2 = getattr(self.config, 'GOOGLE_MAPS_API_KEY2', None)
        if api_key_2:
            try:
                self.gmaps_secondary = googlemaps.Client(key=api_key_2)
                self.has_secondary_gmaps = True
                logger.info("✅ Secondary GoogleMaps client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize secondary GoogleMaps client: {e}")
                self.gmaps_secondary = None
                self.has_secondary_gmaps = False
        else:
            self.gmaps_secondary = None
            self.has_secondary_gmaps = False





    def _log_coordinates(self, latitude: float, longitude: float, context: str):
        """Log coordinates for debugging"""
        logger.info(f"🌍 {context}: {latitude:.6f}, {longitude:.6f}")


    def _convert_gmaps_result(
        self, 
        place: Dict[str, Any], 
        search_lat: float, 
        search_lng: float
    ) -> Optional[VenueSearchResult]:
        """Convert GoogleMaps result with FIXED type handling"""
        try:
            # FIXED: Safe dictionary access
            geometry = place.get('geometry', {})
            location = geometry.get('location', {}) if geometry else {}
            venue_lat = location.get('lat') if location else None
            venue_lng = location.get('lng') if location else None

            if venue_lat is None or venue_lng is None:
                logger.warning(f"⚠️  No location data for place: {place.get('name', 'Unknown')}")
                return None

            distance_km = LocationUtils.calculate_distance(
                (search_lat, search_lng), (venue_lat, venue_lng)
            )

            return VenueSearchResult(
                place_id=place.get('place_id', ''),
                name=place.get('name', 'Unknown'),
                address=place.get('formatted_address', ''),
                latitude=venue_lat,
                longitude=venue_lng,
                distance_km=distance_km,
                business_status=place.get('business_status', 'OPERATIONAL'),
                rating=place.get('rating'),
                user_ratings_total=place.get('user_ratings_total'),
                search_source="googlemaps"
            )

        except Exception as e:
            logger.error(f"❌ Error converting GoogleMaps result: {e}")
            return None

    @traceable(run_type="chain", name="map_search")
    async def search_venues_with_ai_analysis(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[VenueSearchResult]:
        """
        MAIN SEARCH METHOD: Uses classic Google Maps text search with AI-optimized queries
        
        Uses AI to convert user queries into optimized text search terms for Google Maps,
        enabling effective text-based restaurant discovery with key rotation support.

        Args:
            coordinates: (latitude, longitude) tuple
            query: User search query 
            cancel_check_fn: Optional cancellation check function

        Returns:
            List of VenueSearchResult objects
        """
        try:
            logger.info(f"🎯 Starting AI-guided search for '{query}'")
            latitude, longitude = coordinates

            # Check for cancellation
            if cancel_check_fn and cancel_check_fn():
                logger.info("🚫 Search cancelled by user")
                return []

            self._log_coordinates(latitude, longitude, "INPUT coordinates")

            # NEW: Get both text search query and place types from AI analysis
            search_analysis = await self._analyze_query_for_search(query)
            place_types = search_analysis["place_types"]
            text_search_query = search_analysis["text_search_query"]

            logger.info(f"🤖 Search strategy: text='{text_search_query}', types={place_types}")

            venues = []

            # Use GoogleMaps with optimized text search query
            logger.info(f"🔍 Searching with GoogleMaps using text query: '{text_search_query}'")
            venues = await self._googlemaps_search_with_rotation(latitude, longitude, text_search_query)
            if cancel_check_fn and cancel_check_fn():
                return []
            # Ensure results are within search radius
            distance_filtered_venues = [
                v for v in venues
                if LocationUtils.is_within_radius(
                    (latitude, longitude), (v.latitude, v.longitude), self.search_radius_km
                )
            ]

            if not distance_filtered_venues:
                logger.info("📵 No venues found within search radius")
                return []

            # Apply rating filter
            filtered_venues = [
                v for v in distance_filtered_venues
                if v.rating and v.rating >= self.rating_threshold
            ]

            # Sort and limit
            filtered_venues.sort(key=lambda x: (x.rating or 0, -x.distance_km), reverse=True)
            final_venues = filtered_venues[:self.max_venues_to_search]

            # Final logging
            logger.info(f"🎯 Search completed: {len(final_venues)} venues")
            if final_venues:
                for i, venue in enumerate(final_venues[:5], 1):
                    logger.info(f"   {i}. {venue.name}: {venue.rating}⭐ ({venue.distance_km:.1f}km)")

            return final_venues

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return []

    async def _googlemaps_search_with_rotation(
        self, 
        latitude: float, 
        longitude: float, 
        query: str
    ) -> List[VenueSearchResult]:
        """GoogleMaps library search with API key rotation support"""

        # Try primary GoogleMaps client first
        venues = await self._googlemaps_search_internal(latitude, longitude, query, self.gmaps, "primary")

        # If no results and we have secondary client, try that
        if not venues and self.has_secondary_gmaps and self.gmaps_secondary:
            logger.info("🔄 Trying secondary GoogleMaps API key...")
            venues = await self._googlemaps_search_internal(latitude, longitude, query, self.gmaps_secondary, "secondary")

        return venues

    @traceable(run_type="tool", name="googlemaps_search")
    async def _googlemaps_search_internal(
        self, 
        latitude: float, 
        longitude: float, 
        query: str,
        gmaps_client,
        key_name: str
    ) -> List[VenueSearchResult]:
        """
        Internal GoogleMaps search method with ENHANCED DEBUG LOGGING
        """
        venues = []

        if not gmaps_client:
            logger.warning(f"⚠️  GoogleMaps {key_name} client not available")
            return venues

        # Initialize final_query to ensure it's always available in exception handler
        final_query = query
        
        try:
            self._log_coordinates(latitude, longitude, f"GoogleMaps library search ({key_name})")
            self.api_usage["gmaps"] += 1

            location = f"{latitude},{longitude}"
            radius_m = int(self.search_radius_km * 1000)

            # ENHANCED: Smarter query construction with detailed logging
            search_terms = query.lower()
            has_venue_type = any(term in search_terms for term in [
                'restaurant', 'bar', 'cafe', 'coffee', 'bakery', 'bistro', 
                'steakhouse', 'pizzeria', 'sushi', 'pub', 'tavern'
            ])

            if has_venue_type:
                final_query = query
            else:
                final_query = f"{query} restaurant"

            # 🆕 ENHANCED DEBUG LOGGING - Multiple log levels for visibility
            logger.info(f"🔍 GoogleMaps Search Debug ({key_name}):")
            logger.info(f"   📥 Original user query: '{query}'")
            logger.info(f"   🎯 Final search query: '{final_query}'")
            logger.info(f"   📍 Location: {latitude:.4f}, {longitude:.4f}")
            logger.info(f"   📏 Radius: {radius_m}m ({self.search_radius_km}km)")
            logger.info(f"   🏷️  Has venue type: {has_venue_type}")

            # Also log at DEBUG level for detailed debugging
            logger.debug("🔍 GoogleMaps detailed params:")
            logger.debug(f"   query='{final_query}'")
            logger.debug(f"   location='{location}'")
            logger.debug(f"   radius={radius_m}")

            # Execute the search
            response = gmaps_client.places(
                query=final_query,
                location=location,
                radius=radius_m,
            )

            results = response.get('results', []) if response else []

            # 🆕 ENHANCED RESULTS LOGGING
            logger.info(f"✅ GoogleMaps ({key_name}) Results:")
            logger.info(f"   📊 Total results: {len(results)}")

            if results:
                logger.info("   🏆 Top 3 results:")
                for i, place in enumerate(results[:3], 1):
                    name = place.get('name', 'Unknown')
                    rating = place.get('rating', 'No rating')
                    logger.info(f"     {i}. {name} ({rating}⭐)")

            # Convert results
            for place in results:
                try:
                    venue = self._convert_gmaps_result(place, latitude, longitude)
                    if venue and venue.distance_km <= self.search_radius_km:
                        venues.append(venue)
                    elif venue:
                        logger.debug(
                        f"Skipping {venue.name} at {venue.distance_km:.2f}km – outside radius"
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Error converting GoogleMaps result: {e}")

        except Exception as e:
            logger.error(f"❌ GoogleMaps search failed ({key_name}): {e}")
            logger.error(f"   Query was: '{final_query}'")
            logger.error(f"   Location was: {latitude:.4f}, {longitude:.4f}")

        logger.info(f"🎯 GoogleMaps search completed: {len(venues)} venues ({key_name})")
        return venues

    async def _analyze_query_for_search(self, query: str) -> Dict[str, Any]:
        """
        FIXED: AI analysis that converts user query to effective text search query + place types

        Returns both optimized text search query and relevant place types for hybrid approach
        """
        try:
            # FIXED: Type guard for OpenAI v1.0+ client
            if not self.openai_client:
                logger.info("🤖 No OpenAI client available, using default analysis")
                return self._get_default_search_analysis(query)

            # Create prompt for comprehensive query analysis
            place_types_str = ", ".join(self.RESTAURANT_PLACE_TYPES)

            prompt = f"""
    Convert this user restaurant query into an optimized Google Maps text search query and relevant place types.

    User Query: "{{query}}"

    Your task:
    1. Create an optimized text search query that will find relevant restaurants
    2. Select 2-3 most relevant Google Places API place types as backup filters

    TEXT SEARCH QUERY GUIDELINES:
    - Extract the key intent and convert to effective search terms
    - The length of the search query should be 1-3 words
    - Discard location information if included — neighborhoods/streets are handled by coordinates
    - For dishes/food: include the dish name + cuisine type if relevant
    - For atmosphere/features: include descriptive terms
    - For specific needs: focus on the main requirement
    - Keep it concise but specific (1-3 words typically)

    EXAMPLES:
    - "my kids want pasta, where to go" → "pasta italian restaurant"
    - "I'm looking for some fancy cocktails" → "mixology cocktails bar"
    - "somewhere nice for a date" → "romantic restaurant"
    - "best ramen in the area" → "ramen japanese"
    - "I want to have lunch and I'm vegan" → "vegan restaurant"
    - "restaurant with good wine list" → "restaurant wine list"
    - "italian restaurants" → "italian restaurant"
    - "coffee and breakfast" → "coffee breakfast cafe"
    - "where to grab a bite after the club?" → "late night food"
    - "Want to have a drink outside" → "outdoor seating bar"

    Available place types: {place_types_str}

    Return ONLY valid JSON:
    {{{{
        "text_search_query": "optimized search terms",
        "place_types": ["type1", "type2", "type3"],
        "search_intent": "brief description of what user wants"
    }}}}

    Make the text_search_query specific enough to find relevant places but not so narrow it excludes good options.
    """

            # FIXED: Insert the actual query into the prompt properly
            formatted_prompt = prompt.format(query=query)

            # FIXED: OpenAI v1.0+ API call syntax with better error handling
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.1,
                max_tokens=150
            )

            # FIXED: Better response validation and error handling
            if not response or not response.choices or len(response.choices) == 0:
                logger.warning("⚠️  OpenAI returned empty response")
                return self._get_default_search_analysis(query)

            result = response.choices[0].message.content

            # FIXED: Handle None response content
            if not result:
                logger.warning("⚠️  OpenAI returned None content")
                return self._get_default_search_analysis(query)

            result = result.strip()

            # FIXED: Handle empty response
            if not result:
                logger.warning("⚠️  OpenAI returned empty content after strip")
                return self._get_default_search_analysis(query)

            # FIXED: Clean up markdown formatting if present
            if result.startswith('```json'):
                result = result[7:]  # Remove ```json
                if result.endswith('```'):
                    result = result[:-3]  # Remove trailing ```
            elif result.startswith('```'):
                result = result[3:]  # Remove ```
                if result.endswith('```'):
                    result = result[:-3]  # Remove trailing ```

            result = result.strip()

            # FIXED: Final check for empty content
            if not result:
                logger.warning("⚠️  Empty content after cleaning markdown")
                return self._get_default_search_analysis(query)

            # FIXED: Better JSON parsing with error handling
            try:
                analysis = json.loads(result)
            except json.JSONDecodeError as json_error:
                logger.warning(f"⚠️  AI returned invalid JSON: {json_error}")
                logger.debug(f"   Raw response: '{result}'")
                return self._get_default_search_analysis(query)

            # FIXED: Validate the analysis structure
            if not isinstance(analysis, dict):
                logger.warning(f"⚠️  AI returned non-dict: {type(analysis)}")
                return self._get_default_search_analysis(query)

            # FIXED: Extract and validate fields with proper fallbacks
            text_search_query = analysis.get("text_search_query")
            if not text_search_query or not isinstance(text_search_query, str):
                logger.warning("⚠️  No valid text_search_query in AI response")
                text_search_query = query  # Use original query as fallback

            place_types = analysis.get("place_types", [])
            if not isinstance(place_types, list):
                place_types = []

            # Validate place types are in our list
            valid_place_types = [t for t in place_types if t in self.RESTAURANT_PLACE_TYPES]

            if len(valid_place_types) < 2:
                # Add fallbacks if AI didn't return enough valid types
                fallbacks = ["restaurant", "food", "meal_takeaway"]
                for fallback in fallbacks:
                    if fallback not in valid_place_types and fallback in self.RESTAURANT_PLACE_TYPES:
                        valid_place_types.append(fallback)
                    if len(valid_place_types) >= 5:
                        break

            result_analysis = {
                "text_search_query": text_search_query,
                "place_types": valid_place_types[:5],
                "search_intent": analysis.get("search_intent", "restaurant search")
            }

            logger.info(f"🤖 AI query analysis for '{query}':")
            logger.info(f"   Text search: '{result_analysis['text_search_query']}'")
            logger.info(f"   Place types: {result_analysis['place_types']}")
            return result_analysis

        except Exception as e:
            logger.warning(f"⚠️  AI query analysis failed: {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
            return self._get_default_search_analysis(query)

    def _get_default_search_analysis(self, query: str) -> Dict[str, Any]:
        # Handle None query
        if not query:
            return {
                "text_search_query": "restaurant", 
                "place_types": ["restaurant", "food", "meal_takeaway"],
                "search_intent": "general restaurant search"
            }
        # Initialize text_query with a default value to prevent UnboundLocalError
        text_query = "restaurant"  # <-- ADD THIS LINE

        try:
            # Use a simpler, more constrained AI prompt for fallback
            simple_prompt = f"""
    Convert this query to a simple Google Maps search: "{query}"

    Rules:
    - Extract main food/cuisine type if mentioned
    - Add "restaurant" if not present
    - Keep it 1-3 words max
    - Remove location info

    Examples:
    "best sushi" → "sushi restaurant"
    "coffee and pastries" → "coffee cafe"
    "romantic dinner" → "romantic restaurant"
    "where to eat" → "restaurant"

    Return only the search terms, nothing else:
    """

            # Try a simple completion call with very low temperature
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model for fallback
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.0,
                    max_tokens=20
                )

                if response and response.choices and response.choices[0].message.content:
                    ai_query = response.choices[0].message.content.strip()

                    # Clean up the response
                    ai_query = ai_query.replace('"', '').replace("'", '').strip()

                    if ai_query and len(ai_query.split()) <= 4:
                        # Determine place types based on the AI result
                        place_types = ["restaurant", "food"]

                        ai_lower = ai_query.lower()
                        if "coffee" in ai_lower or "cafe" in ai_lower:
                            place_types = ["cafe", "coffee_shop", "restaurant"]
                        elif "bar" in ai_lower or "drink" in ai_lower:
                            place_types = ["bar", "restaurant", "establishment"]
                        elif any(word in ai_lower for word in ["breakfast", "brunch"]):
                            place_types = ["breakfast_restaurant", "cafe", "restaurant"]
                        else:
                            place_types = ["restaurant", "food", "meal_takeaway"]

                        return {
                            "text_search_query": ai_query,
                            "place_types": place_types,
                            "search_intent": f"Search for {ai_query}"
                        }

            else:
                text_query = "restaurant"

            # Use default place type mapping based on the original query
            place_types = self._get_default_place_types(text_query)
            
            
            return {
                "text_search_query": text_query,
                "place_types": place_types,
                "search_intent": f"Search for {text_query}"
            }

        except Exception as e:
            logger.warning(f"⚠️  Even fallback analysis failed: {e}")
            # Ultimate fallback
            return {
                "text_search_query": "restaurant",
                "place_types": ["restaurant", "food", "meal_takeaway"],
                "search_intent": "general restaurant search"
            }

    def _get_default_place_types(self, query: str) -> List[str]:
        """Get default place types based on simple query analysis - FIXED"""
        query_lower = query.lower()

        # Cuisine-specific defaults
        if any(word in query_lower for word in ["italian", "pizza", "pasta"]):
            return ["italian_restaurant", "pizza_restaurant", "restaurant", "food", "meal_takeaway"]
        elif any(word in query_lower for word in ["chinese", "asian"]):
            return ["chinese_restaurant", "restaurant", "food", "meal_takeaway", "establishment"]
        elif any(word in query_lower for word in ["japanese", "sushi", "ramen"]):
            return ["japanese_restaurant", "sushi_restaurant", "ramen_restaurant", "restaurant", "food"]
        elif any(word in query_lower for word in ["indian", "curry"]):
            return ["indian_restaurant", "restaurant", "food", "meal_takeaway", "establishment"]
        elif any(word in query_lower for word in ["mexican", "taco", "burrito"]):
            return ["mexican_restaurant", "restaurant", "food", "meal_takeaway", "establishment"]
        elif any(word in query_lower for word in ["french", "bistro"]):
            return ["french_restaurant", "restaurant", "food", "establishment", "meal_takeaway"]
        elif any(word in query_lower for word in ["thai"]):
            return ["thai_restaurant", "restaurant", "food", "meal_takeaway", "establishment"]
        elif any(word in query_lower for word in ["korean", "bbq", "barbecue"]):
            return ["korean_restaurant", "barbecue_restaurant", "restaurant", "food", "establishment"]
        elif any(word in query_lower for word in ["mediterranean", "greek"]):
            return ["mediterranean_restaurant", "greek_restaurant", "restaurant", "food", "establishment"]
        elif any(word in query_lower for word in ["vietnamese", "pho"]):
            return ["vietnamese_restaurant", "restaurant", "food", "meal_takeaway", "establishment"]

        # Service type defaults
        elif any(word in query_lower for word in ["coffee", "cafe", "espresso"]):
            return ["coffee_shop", "cafe", "bakery", "breakfast_restaurant", "food"]
        elif any(word in query_lower for word in ["bakery", "bread", "pastry"]):
            return ["bakery", "cafe", "breakfast_restaurant", "food", "establishment"]
        elif any(word in query_lower for word in ["bar", "pub", "drinks"]):
            return ["bar", "pub", "restaurant", "night_club", "establishment"]
        elif any(word in query_lower for word in ["fast food", "quick", "drive"]):
            return ["fast_food_restaurant", "hamburger_restaurant", "restaurant", "meal_takeaway", "food"]
        elif any(word in query_lower for word in ["delivery", "takeaway", "takeout"]):
            return ["meal_delivery", "meal_takeaway", "restaurant", "food", "establishment"]
        elif any(word in query_lower for word in ["breakfast", "brunch"]):
            return ["breakfast_restaurant", "brunch_restaurant", "cafe", "restaurant", "food"]
        elif any(word in query_lower for word in ["steak", "steakhouse", "beef"]):
            return ["steak_house", "american_restaurant", "restaurant", "food", "establishment"]
        elif any(word in query_lower for word in ["seafood", "fish", "lobster"]):
            return ["seafood_restaurant", "restaurant", "food", "establishment", "meal_takeaway"]
        elif any(word in query_lower for word in ["vegetarian", "vegan", "plant"]):
            return ["vegetarian_restaurant", "vegan_restaurant", "restaurant", "food", "establishment"]
        elif any(word in query_lower for word in ["ice cream", "gelato", "dessert"]):
            return ["ice_cream_shop", "bakery", "cafe", "restaurant", "food"]

        # Default fallback
        logger.info(f"🤖 Using default place types for query: '{query}'")
        return ["restaurant", "food", "meal_takeaway", "establishment", "cafe"]

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'has_googlemaps_client': self.gmaps is not None,
            'has_secondary_googlemaps': getattr(self, 'has_secondary_gmaps', False),
            'rating_threshold': self.rating_threshold,
            'search_radius_km': self.search_radius_km,
            'api_usage': self.api_usage.copy(),
            'place_types_count': len(self.RESTAURANT_PLACE_TYPES)
        }