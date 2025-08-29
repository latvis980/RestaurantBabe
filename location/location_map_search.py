# location/location_map_search.py
"""
Google Maps/Places Search Agent - ALL TYPE ERRORS FIXED

Fixed all type checking errors by adding proper type guards,
handling None cases, and using conditional imports correctly.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import googlemaps
from location.location_utils import LocationUtils

logger = logging.getLogger(__name__)

# FIXED: Proper conditional imports with type checking
try:
    from google.oauth2 import service_account
    from google.maps import places_v1
    from google.type import latlng_pb2
    HAS_PLACES_API = True
    logger.info("✅ Google Places API v1 imports successful")
except ImportError as e:
    logger.warning(f"⚠️  Google Places API v1 not available: {e}")
    # Create placeholder types to avoid None errors
    service_account = None
    places_v1 = None
    latlng_pb2 = None
    HAS_PLACES_API = False

# FIXED: OpenAI import with proper error handling
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    logger.warning("⚠️  OpenAI not available")
    openai = None
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
    search_source: str = "places_api"
    google_maps_url: str = ""

    def __post_init__(self):
        if not self.google_maps_url and self.place_id:
            self.google_maps_url = f"https://maps.google.com/maps/place/?q=place_id:{self.place_id}"

class LocationMapSearchAgent:
    """
    Google Maps/Places search agent with ALL TYPE ERRORS FIXED

    Proper type guards and None handling for all imports and client operations.
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.rating_threshold = float(getattr(config, 'RATING_THRESHOLD', 4.3))
        self.search_radius_km = float(getattr(config, 'SEARCH_RADIUS_KM', 2.0))
        self.max_venues_to_search = int(getattr(config, 'MAX_VENUES_TO_SEARCH', 20))

        # OpenAI configuration - FIXED: proper None handling
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        if HAS_OPENAI and openai is not None:
            openai_key = getattr(config, 'OPENAI_API_KEY', None)
            if openai_key:
                openai.api_key = openai_key

        # Initialize clients - FIXED: proper typing
        self.places_client: Optional[Any] = None
        self.gmaps: Optional[googlemaps.Client] = None
        self.api_usage = {"places": 0, "gmaps": 0}

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
        logger.info(f"   - Has Places API v1: {self.places_client is not None}")
        logger.info(f"   - Has GoogleMaps fallback: {self.gmaps is not None}")

    def _initialize_clients(self):
        """Initialize Google Maps clients with FIXED type handling"""
        # GoogleMaps library (always reliable)
        api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)
        if api_key:
            try:
                self.gmaps = googlemaps.Client(key=api_key)
                logger.info("✅ GoogleMaps client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GoogleMaps client: {e}")

        # Places API v1 client (only if imports are available) - FIXED: type guards
        if HAS_PLACES_API and service_account is not None and places_v1 is not None:
            try:
                # FIXED: Check file path exists
                creds_path = getattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
                if creds_path and os.path.exists(creds_path):
                    credentials = service_account.Credentials.from_service_account_file(
                        creds_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )

                    # FIXED: Use correct client initialization with type guard
                    self.places_client = places_v1.PlacesClient(credentials=credentials)
                    logger.info("✅ Places API v1 client initialized")

                elif hasattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS_JSON'):
                    # Handle JSON credentials from environment - FIXED: type guard
                    creds_json = getattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS_JSON', None)
                    if creds_json:
                        try:
                            creds_info = json.loads(creds_json)
                            credentials = service_account.Credentials.from_service_account_info(
                                creds_info,
                                scopes=['https://www.googleapis.com/auth/cloud-platform']
                            )
                            self.places_client = places_v1.PlacesClient(credentials=credentials)
                            logger.info("✅ Places API v1 client initialized from JSON")
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Invalid JSON in credentials: {e}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize Places API v1: {e}")
                logger.info("🔧 Will use GoogleMaps library as fallback")
        else:
            logger.info("⚠️  Places API v1 imports not available - using GoogleMaps only")

    def _log_coordinates(self, latitude: float, longitude: float, context: str):
        """Log coordinates for debugging"""
        logger.info(f"🌍 {context}: {latitude:.6f}, {longitude:.6f}")

    async def _places_api_search(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[VenueSearchResult]:
        """Execute Places API v1 search with FIXED type handling"""
        venues = []

        # FIXED: Type guards for all None checks
        if (not self.places_client or 
            not HAS_PLACES_API or 
            places_v1 is None or 
            latlng_pb2 is None):
            logger.info("⚠️  Places API v1 not available, skipping")
            return venues

        try:
            self._log_coordinates(latitude, longitude, "Places API v1 search")
            self.api_usage["places"] += 1

            # FIXED: Create request using type guards
            center = latlng_pb2.LatLng(latitude=latitude, longitude=longitude)
            radius_m = int(self.search_radius_km * 1000)

            # FIXED: Create the search request with proper type handling
            request = places_v1.SearchNearbyRequest(
                location_restriction=places_v1.SearchNearbyRequest.LocationRestriction(
                    circle=places_v1.Circle(
                        center=center,
                        radius=radius_m
                    )
                ),
                included_types=place_types[:5],
                max_result_count=min(20, self.max_venues_to_search),
                language_code="en",
                rank_preference=places_v1.SearchNearbyRequest.RankPreference.POPULARITY
            )

            # FIXED: Execute with proper type handling
            response = self.places_client.search_nearby(
                request=request,
                metadata=[
                    ("x-goog-fieldmask", 
                     "places.id,places.displayName,places.formattedAddress,places.location," +
                     "places.rating,places.userRatingCount,places.businessStatus")
                ]
            )

            # FIXED: Check response with proper type handling
            if hasattr(response, 'places') and response.places:
                logger.info(f"✅ Places API v1 returned {len(response.places)} results")

                for place in response.places:
                    try:
                        venue = self._convert_places_result(place, latitude, longitude)
                        if venue:
                            venues.append(venue)
                    except Exception as e:
                        logger.warning(f"⚠️  Error converting Places result: {e}")

        except Exception as e:
            logger.error(f"❌ Places API v1 search failed: {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")

        logger.info(f"🎯 Places API v1 search completed: {len(venues)} venues")
        return venues

    def _convert_places_result(
        self, 
        place: Any, 
        search_lat: float, 
        search_lng: float
    ) -> Optional[VenueSearchResult]:
        """Convert Places API v1 result with FIXED safe attribute access"""
        try:
            # FIXED: Safe attribute extraction with proper None handling
            place_id = getattr(place, 'id', '') or ''

            display_name = getattr(place, 'display_name', None)
            if display_name and hasattr(display_name, 'text'):
                name = display_name.text or "Unknown"
            else:
                name = "Unknown"

            address = getattr(place, 'formatted_address', '') or ''

            location = getattr(place, 'location', None)
            if not location:
                logger.warning(f"⚠️  No location for place: {name}")
                return None

            venue_lat = getattr(location, 'latitude', None)
            venue_lng = getattr(location, 'longitude', None)

            if venue_lat is None or venue_lng is None:
                logger.warning(f"⚠️  Invalid coordinates for place: {name}")
                return None

            # Calculate distance
            distance_km = LocationUtils.calculate_distance(
                (search_lat, search_lng), (venue_lat, venue_lng)
            )

            # FIXED: Extract other fields safely
            rating = getattr(place, 'rating', None)
            user_ratings_total = getattr(place, 'user_rating_count', None)

            business_status_obj = getattr(place, 'business_status', None)
            if business_status_obj and hasattr(business_status_obj, 'name'):
                business_status = business_status_obj.name
            else:
                business_status = "OPERATIONAL"

            return VenueSearchResult(
                place_id=place_id,
                name=name,
                address=address,
                latitude=venue_lat,
                longitude=venue_lng,
                distance_km=distance_km,
                business_status=business_status,
                rating=rating,
                user_ratings_total=user_ratings_total,
                search_source="places_api_v1"
            )

        except Exception as e:
            logger.error(f"❌ Error converting Places result: {e}")
            return None

    async def _googlemaps_search(
        self, 
        latitude: float, 
        longitude: float, 
        query: str
    ) -> List[VenueSearchResult]:
        """GoogleMaps library search with FIXED type handling"""
        venues = []

        # FIXED: Type guard for gmaps client
        if not self.gmaps:
            logger.warning("⚠️  GoogleMaps client not available")
            return venues

        try:
            self._log_coordinates(latitude, longitude, "GoogleMaps library search")
            self.api_usage["gmaps"] += 1

            location = f"{latitude},{longitude}"
            radius_m = int(self.search_radius_km * 1000)

            # FIXED: Text search with proper error handling
            response = self.gmaps.places(
                query=f"{query} restaurant",
                location=location,
                radius=radius_m,
            )

            results = response.get('results', []) if response else []
            logger.info(f"✅ GoogleMaps returned {len(results)} results")

            for place in results:
                try:
                    venue = self._convert_gmaps_result(place, latitude, longitude)
                    if venue:
                        venues.append(venue)
                except Exception as e:
                    logger.warning(f"⚠️  Error converting GoogleMaps result: {e}")

        except Exception as e:
            logger.error(f"❌ GoogleMaps search failed: {e}")

        logger.info(f"🎯 GoogleMaps search completed: {len(venues)} venues")
        return venues

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

    async def search_venues_with_ai_analysis(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[VenueSearchResult]:
        """
        MAIN SEARCH METHOD: Uses Places API v1 with GoogleMaps fallback

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

            # Get place types for search
            place_types = await self._analyze_query_for_place_types(query)
            if not place_types:
                place_types = ["restaurant", "food", "meal_takeaway"]

            venues = []

            # Try Places API v1 first
            if self.places_client:
                venues = await self._places_api_search(latitude, longitude, place_types)
                if cancel_check_fn and cancel_check_fn():
                    return []

            # Fallback to GoogleMaps if needed
            if not venues and self.gmaps:
                logger.info("🔄 Falling back to GoogleMaps library")
                venues = await self._googlemaps_search(latitude, longitude, query)
                if cancel_check_fn and cancel_check_fn():
                    return []

            # Apply rating filter
            filtered_venues = [
                v for v in venues 
                if v.rating and v.rating >= self.rating_threshold
            ]

            # Sort and limit
            filtered_venues.sort(key=lambda x: (x.rating or 0, -x.distance_km), reverse=True)
            final_venues = filtered_venues[:self.max_venues_to_search]

            # Final logging
            logger.info(f"🎯 Search completed: {len(final_venues)} venues")
            if final_venues:
                for i, venue in enumerate(final_venues[:5], 1):
                    logger.info(f"   {i}. {venue.name}: {venue.latitude:.6f}, {venue.longitude:.6f}")

            return final_venues

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return []

    async def _analyze_query_for_place_types(self, query: str) -> List[str]:
        """AI analysis for place types using comprehensive restaurant type list - FIXED"""
        try:
            # FIXED: Type guard for OpenAI
            if not HAS_OPENAI or openai is None or not hasattr(openai, 'api_key') or not openai.api_key:
                logger.info("🤖 No OpenAI available, using default place types")
                return self._get_default_place_types(query)

            # Create prompt with full place type list
            place_types_str = ", ".join(self.RESTAURANT_PLACE_TYPES)

            prompt = f"""
            Analyze this restaurant search query and return the 5 most relevant Google Places API place types.

            Query: "{query}"

            Available place types: {place_types_str}

            Instructions:
            - Choose 5 most relevant types based on the query
            - Order by relevance (most relevant first)  
            - Always include "restaurant" unless query is very specific (like "coffee shop")
            - Consider cuisine type, dining style, and service type
            - Return ONLY a JSON array, no explanation

            Examples:
            - "Italian restaurants" → ["italian_restaurant", "restaurant", "food", "establishment", "meal_takeaway"]
            - "coffee shops near me" → ["coffee_shop", "cafe", "bakery", "breakfast_restaurant", "food"]
            - "sushi delivery" → ["sushi_restaurant", "japanese_restaurant", "meal_delivery", "restaurant", "food"]
            - "best steakhouse" → ["steak_house", "american_restaurant", "restaurant", "food", "establishment"]
            """

            # FIXED: OpenAI API call with proper error handling
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )

            if response and response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                place_types = json.loads(result)

                # Validate that returned types are in our list
                valid_types = [t for t in place_types if t in self.RESTAURANT_PLACE_TYPES]

                if len(valid_types) < 3:
                    # Add fallbacks if AI didn't return enough valid types
                    fallbacks = ["restaurant", "food", "meal_takeaway"]
                    for fallback in fallbacks:
                        if fallback not in valid_types:
                            valid_types.append(fallback)
                        if len(valid_types) >= 5:
                            break

                logger.info(f"🤖 AI selected place types for '{query}': {valid_types[:5]}")
                return valid_types[:5]
            else:
                logger.warning("⚠️  OpenAI returned empty response")
                return self._get_default_place_types(query)

        except Exception as e:
            logger.warning(f"⚠️  AI query analysis failed: {e}")
            return self._get_default_place_types(query)

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
        """Get search statistics - FIXED return type"""
        return {
            'has_places_api_v1': self.places_client is not None,
            'has_googlemaps_fallback': self.gmaps is not None,
            'rating_threshold': self.rating_threshold,
            'search_radius_km': self.search_radius_km,
            'api_usage': self.api_usage.copy(),
            'place_types_count': len(self.RESTAURANT_PLACE_TYPES)
        }