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
from urllib.parse import quote

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
    search_source: str = "places_api"
    google_maps_url: str = ""

    def __post_init__(self):
        if not self.google_maps_url and self.place_id and self.name:
            # Use 2025 universal format for mobile/desktop compatibility
            from urllib.parse import quote
            encoded_name = quote(self.name.strip(), safe='')
            self.google_maps_url = f"https://www.google.com/maps/search/?api=1&query={encoded_name}&query_place_id={self.place_id}"
        elif not self.google_maps_url and self.place_id:
            # Fallback if no name available
            self.google_maps_url = f"https://www.google.com/maps/search/?api=1&query=restaurant&query_place_id={self.place_id}"

class LocationMapSearchAgent:
    """
    Google Maps/Places search agent

    Proper type guards and None handling for all imports and client operations.
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
        """Initialize Google Maps clients with FIXED credential rotation support"""
        # GoogleMaps library (always reliable) - uses regular API key
        api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)
        if api_key:
            try:
                self.gmaps = googlemaps.Client(key=api_key)
                logger.info("✅ GoogleMaps client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GoogleMaps client: {e}")

        # NEW: Add secondary GoogleMaps client for rotation
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

        # Places API v1 client with credential rotation - FIXED implementation
        if HAS_PLACES_API and service_account is not None and places_v1 is not None:
            self.places_client = self._initialize_places_client_with_rotation()
        else:
            logger.info("⚠️  Places API v1 imports not available - using GoogleMaps only")

    def _initialize_places_client_with_rotation(self):
        """Initialize Places API client with credential rotation support"""

        # Try PRIMARY credentials first
        primary_creds = getattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS_JSON_PRIMARY', None)
        secondary_creds = getattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS_JSON_SECONDARY', None)

        # Track which credentials we're using
        self.active_places_credentials = 'primary'

        for creds_name, creds_data in [('primary', primary_creds), ('secondary', secondary_creds)]:
            if not creds_data:
                logger.info(f"⚠️  No {creds_name} credentials found")
                continue

            try:
                # Check if it's a file path or JSON string
                if os.path.exists(creds_data):
                    # File path
                    credentials = service_account.Credentials.from_service_account_file(
                        creds_data,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    logger.info(f"✅ Loaded {creds_name} credentials from file")
                else:
                    # JSON string
                    try:
                        creds_info = json.loads(creds_data)
                        credentials = service_account.Credentials.from_service_account_info(
                            creds_info,
                            scopes=['https://www.googleapis.com/auth/cloud-platform']
                        )
                        logger.info(f"✅ Loaded {creds_name} credentials from JSON")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON in {creds_name} credentials: {e}")
                        continue

                # Try to create the client
                client = places_v1.PlacesClient(credentials=credentials)

                # Test the client with a simple request to verify it works
                # (You could add a test request here if needed)

                self.active_places_credentials = creds_name
                logger.info(f"✅ Places API v1 client initialized with {creds_name} credentials")

                # Store secondary credentials for rotation if this is primary
                if creds_name == 'primary' and secondary_creds:
                    self.secondary_places_credentials = secondary_creds

                return client

            except Exception as e:
                logger.error(f"❌ Failed to initialize Places API v1 with {creds_name} credentials: {e}")
                continue

        logger.error("❌ Failed to initialize Places API v1 with any available credentials")
        return None

    def _rotate_places_credentials(self):
        """Rotate to secondary Places API credentials when primary fails"""
        if not hasattr(self, 'secondary_places_credentials') or not self.secondary_places_credentials:
            logger.warning("⚠️  No secondary Places API credentials available for rotation")
            return False

        try:
            logger.info("🔄 Rotating to secondary Places API credentials...")

            # Load secondary credentials
            try:
                if os.path.exists(self.secondary_places_credentials):
                    credentials = service_account.Credentials.from_service_account_file(
                        self.secondary_places_credentials,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                else:
                    creds_info = json.loads(self.secondary_places_credentials)
                    credentials = service_account.Credentials.from_service_account_info(
                        creds_info,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
            except Exception as e:
                logger.error(f"❌ Failed to load secondary credentials: {e}")
                return False

            # Create new client with secondary credentials
            self.places_client = places_v1.PlacesClient(credentials=credentials)
            self.active_places_credentials = 'secondary'

            logger.info("✅ Successfully rotated to secondary Places API credentials")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to rotate to secondary Places API credentials: {e}")
            return False

    async def _places_api_search_with_rotation(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[VenueSearchResult]:
        """Execute Places API v1 search with automatic credential rotation on failure"""

        # Try the search with current credentials
        venues = await self._places_api_search_internal(latitude, longitude, place_types)

        # If search failed and we haven't tried secondary credentials yet
        if (not venues and 
            self.active_places_credentials == 'primary' and 
            hasattr(self, 'secondary_places_credentials')):

            logger.info("🔄 Primary credentials may be exhausted, trying secondary...")

            if self._rotate_places_credentials():
                # Retry search with secondary credentials
                venues = await self._places_api_search_internal(latitude, longitude, place_types)

        return venues

    async def _places_api_search_internal(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[VenueSearchResult]:
        """Internal Places API search method (original _places_api_search logic)"""
        venues = []

        # FIXED: Type guards for all None checks
        if (not self.places_client or 
            not HAS_PLACES_API or 
            places_v1 is None or 
            latlng_pb2 is None):
            logger.info("⚠️  Places API v1 not available, skipping")
            return venues

        try:
            self._log_coordinates(latitude, longitude, f"Places API v1 search ({self.active_places_credentials})")
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
                logger.info(f"✅ Places API v1 returned {len(response.places)} results ({self.active_places_credentials})")

                for place in response.places:
                    try:
                        venue = self._convert_places_result(place, latitude, longitude)
                        if venue:
                            venues.append(venue)
                    except Exception as e:
                        logger.warning(f"⚠️  Error converting Places result: {e}")

        except Exception as e:
            logger.error(f"❌ Places API v1 search failed ({self.active_places_credentials}): {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")

        logger.info(f"🎯 Places API v1 search completed: {len(venues)} venues ({self.active_places_credentials})")
        return venues

    def _log_coordinates(self, latitude: float, longitude: float, context: str):
        """Log coordinates for debugging"""
        logger.info(f"🌍 {context}: {latitude:.6f}, {longitude:.6f}")

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
        MAIN SEARCH METHOD: Uses Places API v1 with credential rotation and GoogleMaps fallback
        UPDATED: Now uses AI-generated text search queries for better specificity

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

            # Try Places API v1 first with automatic credential rotation
            if self.places_client:
                venues = await self._places_api_search_with_rotation(latitude, longitude, place_types)
                if cancel_check_fn and cancel_check_fn():
                    return []

            # UPDATED: Fallback to GoogleMaps using optimized text search query
            if not venues:
                logger.info(f"🔄 Falling back to GoogleMaps with text query: '{text_search_query}'")
                venues = await self._googlemaps_search_with_rotation(latitude, longitude, text_search_query)
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
                    if venue:
                        venues.append(venue)
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
        NEW: AI analysis that converts user query to effective text search query + place types

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

            User Query: "{query}"

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
            {{
                "text_search_query": "optimized search terms",
                "place_types": ["type1", "type2", "type3"],
                "search_intent": "brief description of what user wants"
            }}

            Make the text_search_query specific enough to find relevant places but not so narrow it excludes good options.
            """

            # FIXED: OpenAI v1.0+ API call syntax
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )

            if response and response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                analysis = json.loads(result)

                # Validate place types are in our list
                valid_place_types = [t for t in analysis.get("place_types", []) if t in self.RESTAURANT_PLACE_TYPES]

                if len(valid_place_types) < 2:
                    # Add fallbacks if AI didn't return enough valid types
                    fallbacks = ["restaurant", "food", "meal_takeaway"]
                    for fallback in fallbacks:
                        if fallback not in valid_place_types:
                            valid_place_types.append(fallback)
                        if len(valid_place_types) >= 5:
                            break

                result_analysis = {
                    "text_search_query": analysis.get("text_search_query", query),
                    "place_types": valid_place_types[:5],
                    "search_intent": analysis.get("search_intent", "restaurant search")
                }

                logger.info(f"🤖 AI query analysis for '{query}':")
                logger.info(f"   Text search: '{result_analysis['text_search_query']}'")
                logger.info(f"   Place types: {result_analysis['place_types']}")
                return result_analysis
            else:
                logger.warning("⚠️  OpenAI returned empty response")
                return self._get_default_search_analysis(query)

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  AI returned invalid JSON: {e}")
            return self._get_default_search_analysis(query)
        except Exception as e:
            logger.warning(f"⚠️  AI query analysis failed: {e}")
            return self._get_default_search_analysis(query)

    def _get_default_search_analysis(self, query: str) -> Dict[str, Any]:
        """
        Get default search analysis based on simple query analysis
        Returns both text search query and place types
        """
        query_lower = query.lower()

        # Simple keyword-based analysis
        if any(word in query_lower for word in ["pasta", "italian"]):
            return {
                "text_search_query": "italian pasta restaurant",
                "place_types": ["italian_restaurant", "restaurant", "food"],
                "search_intent": "Italian cuisine"
            }
        elif any(word in query_lower for word in ["coffee", "cafe", "espresso"]):
            return {
                "text_search_query": "coffee cafe",
                "place_types": ["coffee_shop", "cafe", "breakfast_restaurant"],
                "search_intent": "Coffee/cafe"
            }
        elif any(word in query_lower for word in ["sushi", "japanese", "ramen"]):
            return {
                "text_search_query": "japanese sushi restaurant",
                "place_types": ["japanese_restaurant", "sushi_restaurant", "restaurant"],
                "search_intent": "Japanese cuisine"
            }
        elif any(word in query_lower for word in ["cocktail", "bar", "drinks"]):
            return {
                "text_search_query": "cocktail bar",
                "place_types": ["bar", "restaurant", "establishment"],
                "search_intent": "Cocktails/bar"
            }
        elif any(word in query_lower for word in ["romantic", "anniversary", "date"]):
            return {
                "text_search_query": "romantic restaurant",
                "place_types": ["restaurant", "food", "establishment"],
                "search_intent": "Romantic dining"
            }
        elif any(word in query_lower for word in ["vegan", "vegetarian", "plant"]):
            return {
                "text_search_query": "vegan vegetarian restaurant",
                "place_types": ["vegetarian_restaurant", "vegan_restaurant", "restaurant"],
                "search_intent": "Vegan/vegetarian"
            }
        elif any(word in query_lower for word in ["wine", "sommelier"]):
            return {
                "text_search_query": "wine restaurant bar",
                "place_types": ["bar", "restaurant", "establishment"],
                "search_intent": "Wine focused"
            }
        elif any(word in query_lower for word in ["breakfast", "brunch"]):
            return {
                "text_search_query": "breakfast brunch",
                "place_types": ["breakfast_restaurant", "brunch_restaurant", "cafe"],
                "search_intent": "Breakfast/brunch"
            }
        elif any(word in query_lower for word in ["kids", "family", "children"]):
            return {
                "text_search_query": "family restaurant",
                "place_types": ["restaurant", "family_restaurant", "food"],
                "search_intent": "Family dining"
            }
        else:
            # Generic restaurant search
            return {
                "text_search_query": "restaurant",
                "place_types": ["restaurant", "food", "meal_takeaway"],
                "search_intent": "General restaurant search"
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
        """Get search statistics - FIXED return type"""
        return {
            'has_places_api_v1': self.places_client is not None,
            'has_googlemaps_fallback': self.gmaps is not None,
            'rating_threshold': self.rating_threshold,
            'search_radius_km': self.search_radius_km,
            'api_usage': self.api_usage.copy(),
            'place_types_count': len(self.RESTAURANT_PLACE_TYPES)
        }