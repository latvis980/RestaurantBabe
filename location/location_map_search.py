# location/location_map_search.py
"""
Location Google Maps/Places Search Agent

Handles all Google Maps and Places API search functionality with:
- AI-powered query analysis for optimal search strategy
- Google Places API (New) with fallback to googlemaps library
- Dual credential support with automatic load balancing
- Comprehensive error handling and logging
- Compatible with existing location orchestrator

This file consolidates all Google search logic from the enhanced media verification system.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import googlemaps
from google.oauth2 import service_account
from google.maps import places_v1
from location.location_utils import LocationUtils
import openai

logger = logging.getLogger(__name__)

try:
    from google.type import latlng_pb2
except ImportError:
    logger.warning("google.type.latlng_pb2 not available - some features may not work")
    latlng_pb2 = None

@dataclass
class VenueSearchResult:
    """Structure for Google Maps search results"""
    # Basic Google Maps data
    place_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float

    # Enhanced Google data
    business_status: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    google_reviews: List[Dict] = field(default_factory=list)

    # Search metadata
    search_source: str = "places_api"  # "places_api" or "googlemaps"
    google_maps_url: str = ""

    def __post_init__(self):
        if not self.google_maps_url and self.place_id:
            self.google_maps_url = f"https://maps.google.com/maps/place/?q=place_id:{self.place_id}"

class LocationMapSearchAgent:
    """
    Google Maps/Places search agent with AI-powered query analysis

    Features:
    - Uses Google Places API (New) as primary search method
    - Falls back to googlemaps library when Places API fails
    - AI analysis determines optimal search strategy per query
    - Dual credential support with automatic load balancing
    - Comprehensive error handling and retry logic
    """

    def __init__(self, config):
        self.config = config

        # Initialize configuration attributes with defaults
        self.rating_threshold = getattr(config, 'ENHANCED_RATING_THRESHOLD', 4.3)
        self.max_venues_to_search = getattr(config, 'MAX_VENUES_TO_VERIFY', 8)
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.search_radius_km = getattr(config, 'LOCATION_SEARCH_RADIUS_KM', 2.0)

        # Initialize OpenAI client for AI query analysis
        self.openai_client = openai.OpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        # Initialize Google Maps client for text search fallback
        api_key = getattr(config, 'GOOGLE_MAPS_API_KEY2', None) or getattr(config, 'GOOGLE_MAPS_API_KEY', None)
        if api_key:
            self.gmaps = googlemaps.Client(key=api_key)
            logger.info("Google Maps client initialized for text search fallback")
        else:
            self.gmaps = None
            logger.warning("No Google Maps API key found - text search fallback disabled")

        # Load Google service account credentials for Places API (New)
        self.places_client_primary = self._initialize_places_client('primary')
        self.places_client_secondary = self._initialize_places_client('secondary')

        # Determine if we have dual credentials
        self.has_dual_credentials = (self.places_client_primary is not None and 
                                     self.places_client_secondary is not None)

        if not self.places_client_primary:
            raise ValueError("No valid Google Places API credentials found")

        # API usage tracking for load balancing
        self.api_usage = {'primary': 0, 'secondary': 0}

        logger.info("Location Map Search Agent initialized with AI Query Analysis")
        if self.has_dual_credentials:
            logger.info("Dual credentials mode enabled - automatic load balancing")

    def _initialize_places_client(self, client_type: str):
        """Initialize Places API client with proper credentials"""
        try:
            if client_type == 'primary':
                env_key = 'GOOGLE_APPLICATION_CREDENTIALS_JSON_PRIMARY'
            elif client_type == 'secondary':
                env_key = 'GOOGLE_APPLICATION_CREDENTIALS_JSON_SECONDARY'
            else:
                raise ValueError(f"Invalid client type: {client_type}")

            credentials = self._load_credentials_from_env(env_key, client_type)
            if not credentials:
                return None

            client = places_v1.PlacesClient(credentials=credentials)
            logger.info(f"{client_type.capitalize()} Places API client initialized")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize {client_type} Places API client: {e}")
            return None

    def _load_credentials_from_env(self, env_key: str, key_type: str):
        """Load service account credentials from environment variable"""
        try:
            creds_json_str = getattr(self.config, env_key, None)
            if not creds_json_str:
                logger.info(f"{key_type} credentials not found in environment")
                return None

            # Parse JSON string to dict
            credentials_info = json.loads(creds_json_str)

            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )

            logger.info(f"{key_type} credentials loaded from environment")
            return credentials

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {key_type} credentials: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {key_type} credentials: {e}")
            return None

    def _get_places_client(self):
        """Get the appropriate Places client with load balancing"""
        if not self.has_dual_credentials or not self.places_client_secondary:
            return self.places_client_primary, 'primary'

        # Simple round-robin load balancing
        if self.api_usage['primary'] <= self.api_usage['secondary']:
            return self.places_client_primary, 'primary'
        else:
            return self.places_client_secondary, 'secondary'

    async def search_venues_with_ai_analysis(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[VenueSearchResult]:
        """
        MAIN METHOD: Search venues with AI-powered query analysis

        Args:
            coordinates: (latitude, longitude) tuple
            query: User search query (e.g., "cocktail bars", "sushi restaurants")
            cancel_check_fn: Optional cancellation check function

        Returns:
            List of VenueSearchResult objects
        """
        try:
            logger.info(f"Starting AI-guided venue search for: '{query}'")

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 1: AI-powered query analysis
            logger.info("AI analyzing query and determining search strategy")
            search_strategy = await self._analyze_query_for_search_strategy(query)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"AI Strategy: {search_strategy['approach']} for '{search_strategy['primary_intent']}' using types {search_strategy['place_types']}")

            # Step 2: Execute search based on AI-determined strategy
            venues = await self._execute_ai_guided_search(coordinates, query, search_strategy, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"AI-guided search found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"Error in AI-guided venue search: {e}")
            return []

    async def _analyze_query_for_search_strategy(self, query: str) -> Dict[str, Any]:
        """AI-powered query analysis using Google's official place types"""
        try:
            # Google's official place types for food and drink establishments
            google_place_types = [
                "bar", "wine_bar", "restaurant", "cafe", "coffee_shop", "bakery",
                "fast_food_restaurant", "fine_dining_restaurant", "pizza_restaurant",
                "chinese_restaurant", "italian_restaurant", "japanese_restaurant",
                "sushi_restaurant", "mexican_restaurant", "thai_restaurant",
                "indian_restaurant", "french_restaurant", "american_restaurant",
                "seafood_restaurant", "steakhouse", "vegetarian_restaurant",
                "vegan_restaurant", "brunch_restaurant", "breakfast_restaurant"
            ]

            prompt = f"""Analyze this query and determine the optimal Google Maps search strategy.

USER QUERY: "{query}"

AVAILABLE GOOGLE PLACE TYPES: {google_place_types}

Return JSON only with this exact structure:
{{
  "primary_intent": "brief description of what user wants",
  "place_types": ["exact_google_type1", "exact_google_type2"], 
  "search_keywords": ["keyword1", "keyword2"],
  "approach": "both",
  "use_text_search": true,
  "use_places_api": true,
  "confidence": 0.9
}}

Rules:
- Only use place_types from the provided list
- Limit to max 3 place_types and 3 search_keywords
- Set approach to "both" unless query is very specific
- Higher confidence (0.8+) for clear queries like "cocktail bars"
- Lower confidence (0.5-0.7) for vague queries like "good food"
"""

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.1
                ),
                timeout=15.0
            )

            if not response or not response.choices:
                raise Exception("Empty AI response")

            analysis_text = response.choices[0].message.content
            if not analysis_text:
                raise Exception("Empty content from AI")

            analysis_text = analysis_text.strip()
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text.replace('```json', '').replace('```', '').strip()

            return json.loads(analysis_text)

        except Exception as e:
            logger.error(f"AI query analysis failed: {e}")
            # Intelligent fallback based on query content
            return self._create_fallback_strategy(query)

    def _create_fallback_strategy(self, query: str) -> Dict[str, Any]:
        """Create fallback search strategy when AI analysis fails"""
        query_lower = query.lower()

        # Cocktail bars
        if any(term in query_lower for term in ['cocktail', 'bar', 'drinks']):
            return {
                'primary_intent': 'cocktail bars and drinking establishments',
                'place_types': ['bar', 'wine_bar'],
                'search_keywords': ['cocktail bar', 'bar'],
                'approach': 'both',
                'use_text_search': True,
                'use_places_api': True,
                'confidence': 0.7
            }

        # Coffee shops
        elif any(term in query_lower for term in ['coffee', 'cafe', 'espresso']):
            return {
                'primary_intent': 'coffee shops and cafes',
                'place_types': ['cafe', 'coffee_shop'],
                'search_keywords': ['coffee', 'cafe'],
                'approach': 'both',
                'use_text_search': True,
                'use_places_api': True,
                'confidence': 0.8
            }

        # Sushi
        elif any(term in query_lower for term in ['sushi', 'japanese']):
            return {
                'primary_intent': 'Japanese and sushi restaurants',
                'place_types': ['sushi_restaurant', 'japanese_restaurant'],
                'search_keywords': ['sushi', 'japanese restaurant'],
                'approach': 'both',
                'use_text_search': True,
                'use_places_api': True,
                'confidence': 0.8
            }

        # Default restaurant search
        else:
            return {
                'primary_intent': 'general restaurants',
                'place_types': ['restaurant'],
                'search_keywords': [query],
                'approach': 'both', 
                'use_text_search': True,
                'use_places_api': True,
                'confidence': 0.5
            }

    async def _execute_ai_guided_search(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        search_strategy: Dict[str, Any],
        cancel_check_fn=None
    ) -> List[VenueSearchResult]:
        """Execute search based on AI-determined strategy"""
        try:
            latitude, longitude = coordinates
            venues = []

            # Execute searches based on AI strategy
            if search_strategy['use_text_search'] and self.gmaps:
                text_venues = await self._text_search_venues(
                    latitude, longitude, search_strategy['search_keywords']
                )
                venues.extend(text_venues)

            if cancel_check_fn and cancel_check_fn():
                return []

            if search_strategy['use_places_api']:
                places_venues = await self._places_api_search_venues(
                    latitude, longitude, search_strategy['place_types']
                )
                venues.extend(places_venues)

            # Remove duplicates by place_id
            unique_venues = {}
            for venue in venues:
                if venue.place_id not in unique_venues:
                    unique_venues[venue.place_id] = venue

            final_venues = list(unique_venues.values())

            # Sort by rating and distance
            final_venues.sort(key=lambda x: (x.rating or 0, -x.distance_km), reverse=True)

            return final_venues[:self.max_venues_to_search]

        except Exception as e:
            logger.error(f"Error in AI-guided search execution: {e}")
            return []

    async def _text_search_venues(
        self, 
        latitude: float, 
        longitude: float, 
        search_keywords: List[str]
    ) -> List[VenueSearchResult]:
        """Text search using googlemaps library"""
        try:
            if not self.gmaps:
                logger.warning("Google Maps client not available for text search")
                return []

            venues = []
            location = f"{latitude},{longitude}"
            radius_m = int(self.search_radius_km * 1000)

            for keyword in search_keywords[:3]:  # Limit to top 3 keywords
                try:
                    search_query = f"{keyword} near {latitude},{longitude}"
                    logger.info(f"Text search: {search_query}")

                    response = self.gmaps.places(
                        query=search_query,
                        location=location,
                        radius=radius_m,
                    )

                    results = response.get('results', [])
                    logger.info(f"Text search for '{keyword}' returned {len(results)} results")

                    for place in results:
                        try:
                            venue = await self._convert_gmaps_result(place, latitude, longitude)
                            if venue:
                                venues.append(venue)
                        except Exception as e:
                            logger.warning(f"Error converting text search result: {e}")
                            continue

                except Exception as e:
                    logger.warning(f"Text search failed for keyword '{keyword}': {e}")
                    continue

            logger.info(f"Text search completed, found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []

    async def _places_api_search_venues(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[VenueSearchResult]:
        """Places API search using determined place types"""
        try:
            client, key_name = self._get_places_client()
            self.api_usage[key_name] += 1

            venues = []
            radius_m = int(self.search_radius_km * 1000)

            for place_type in place_types[:3]:  # Limit to top 3 types
                try:
                    logger.info(f"Places API search for type: {place_type}")

                    if not latlng_pb2:
                        logger.warning("latlng_pb2 not available, falling back to text search")
                        continue

                    # Create search request
                    center_point = latlng_pb2.LatLng(latitude=latitude, longitude=longitude)
                    circle_area = places_v1.types.Circle(center=center_point, radius=radius_m)
                    location_restriction = places_v1.SearchNearbyRequest.LocationRestriction(circle=circle_area)

                    request = places_v1.SearchNearbyRequest(
                        location_restriction=location_restriction,
                        included_types=[place_type],
                        max_result_count=10,
                        language_code="en"
                    )

                    metadata = [
                        ("x-goog-fieldmask", 
                         "places.id,places.displayName,places.formattedAddress,places.location," +
                         "places.rating,places.userRatingCount,places.businessStatus,places.reviews")
                    ]

                    response = client.search_nearby(request=request, metadata=metadata)

                    if hasattr(response, 'places'):
                        logger.info(f"Places API for '{place_type}' returned {len(response.places)} results")

                        for place in response.places:
                            try:
                                venue = await self._convert_places_result(place, latitude, longitude)
                                if venue:
                                    venues.append(venue)
                            except Exception as e:
                                logger.warning(f"Error converting Places API result: {e}")
                                continue

                except Exception as e:
                    logger.warning(f"Places API search failed for type '{place_type}': {e}")
                    continue

            logger.info(f"Places API search completed, found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"Error in Places API search: {e}")
            return []

    async def _convert_gmaps_result(self, place: Dict, user_lat: float, user_lng: float) -> Optional[VenueSearchResult]:
        """Convert Google Maps API result to VenueSearchResult"""
        try:
            place_id = place.get('place_id')
            if not place_id:
                return None

            name = place.get('name', 'Unknown')
            address = place.get('formatted_address', 'Unknown address')

            geometry = place.get('geometry', {})
            location = geometry.get('location', {})
            place_lat = location.get('lat')
            place_lng = location.get('lng')

            if place_lat is None or place_lng is None:
                return None

            distance_km = LocationUtils.calculate_distance(
                (user_lat, user_lng), (place_lat, place_lng)
            )

            return VenueSearchResult(
                place_id=place_id,
                name=name,
                address=address,
                latitude=place_lat,
                longitude=place_lng,
                distance_km=distance_km,
                business_status=place.get('business_status', 'OPERATIONAL'),
                rating=place.get('rating'),
                user_ratings_total=place.get('user_ratings_total', 0),
                google_reviews=[],
                search_source="googlemaps"
            )

        except Exception as e:
            logger.warning(f"Error converting gmaps result: {e}")
            return None

    async def _convert_places_result(self, place, user_lat: float, user_lng: float) -> Optional[VenueSearchResult]:
        """Convert Places API result to VenueSearchResult"""
        try:
            place_id = place.id if hasattr(place, 'id') else None
            if not place_id:
                return None

            name = place.display_name.text if hasattr(place, 'display_name') and place.display_name else "Unknown"
            address = place.formatted_address if hasattr(place, 'formatted_address') else "Unknown address"

            if hasattr(place, 'location') and place.location:
                place_lat = place.location.latitude
                place_lng = place.location.longitude
            else:
                return None

            distance_km = LocationUtils.calculate_distance(
                (user_lat, user_lng), (place_lat, place_lng)
            )

            # Extract reviews if available
            google_reviews = []
            if hasattr(place, 'reviews') and place.reviews:
                for review in place.reviews[:3]:
                    review_data = {
                        'rating': review.rating if hasattr(review, 'rating') else 0,
                        'text': review.text.text if hasattr(review, 'text') and hasattr(review.text, 'text') else "",
                        'time': review.publish_time if hasattr(review, 'publish_time') else None
                    }
                    google_reviews.append(review_data)

            # Get business status safely
            business_status = "OPERATIONAL"  # Default
            if hasattr(place, 'business_status'):
                business_status = str(place.business_status)

            return VenueSearchResult(
                place_id=place_id,
                name=name,
                address=address,
                latitude=place_lat,
                longitude=place_lng,
                distance_km=distance_km,
                business_status=business_status,
                rating=place.rating if hasattr(place, 'rating') else None,
                user_ratings_total=place.user_rating_count if hasattr(place, 'user_rating_count') else 0,
                google_reviews=google_reviews,
                search_source="places_api"
            )

        except Exception as e:
            logger.warning(f"Error converting Places API result: {e}")
            return None

    async def search_venues_basic(
        self, 
        coordinates: Tuple[float, float], 
        query: str,
        radius_km: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> List[VenueSearchResult]:
        """
        Basic venue search without AI analysis (for compatibility)

        Args:
            coordinates: (latitude, longitude) tuple
            query: Search query
            radius_km: Search radius in kilometers
            max_results: Maximum number of results

        Returns:
            List of VenueSearchResult objects
        """
        try:
            latitude, longitude = coordinates
            search_radius = radius_km or self.search_radius_km
            max_venues = max_results or self.max_venues_to_search

            logger.info(f"Basic venue search for '{query}' near {latitude:.4f}, {longitude:.4f}")

            venues = []

            # Try text search first if available
            if self.gmaps:
                location = f"{latitude},{longitude}"
                radius_m = int(search_radius * 1000)

                try:
                    response = self.gmaps.places(
                        query=query,
                        location=location,
                        radius=radius_m,
                    )

                    results = response.get('results', [])
                    logger.info(f"Basic search returned {len(results)} results")

                    for place in results:
                        try:
                            venue = await self._convert_gmaps_result(place, latitude, longitude)
                            if venue:
                                venues.append(venue)
                        except Exception as e:
                            logger.warning(f"Error converting basic search result: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Basic search failed: {e}")

            # Sort by rating and distance, limit results
            venues.sort(key=lambda x: (x.rating or 0, -x.distance_km), reverse=True)
            return venues[:max_venues]

        except Exception as e:
            logger.error(f"Error in basic venue search: {e}")
            return []

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search configuration"""
        return {
            'has_places_api_v1': self.places_client_primary is not None,
            'has_googlemaps_fallback': self.gmaps is not None,
            'dual_credentials': self.has_dual_credentials,
            'rating_threshold': self.rating_threshold,
            'search_radius_km': self.search_radius_km,
            'max_venues': self.max_venues_to_search,
            'ai_model': self.openai_model,
            'api_usage': self.api_usage.copy()
        }