"""
Enhanced Media Verification Agent - QUERY-AWARE SEARCH VERSION

FIXED ISSUES:
1. Now passes actual query to search methods
2. Adds query-specific type filtering for cocktail bars, wine bars, etc.
3. Uses text search with proper keywords instead of generic nearby search
4. Maintains all existing functionality with improved search targeting
"""

import logging
import asyncio
import json
import aiohttp
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import googlemaps
from google.oauth2 import service_account
from google.maps import places_v1
# Import existing utilities and models
from location.location_utils import LocationUtils

# Direct OpenAI API instead of LangChain
import openai

logger = logging.getLogger(__name__)

try:
    from google.type import latlng_pb2
except ImportError:
    logger.warning("google.type.latlng_pb2 not available - some features may not work")
    latlng_pb2 = None

@dataclass
class EnhancedVenueData:
    """Structure for enhanced venue data with all verification info"""
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

    # AI analysis results
    review_quality_score: float = 0.0
    selected_for_verification: bool = False

    # Media verification data
    media_search_results: List[Dict] = field(default_factory=list)
    professional_sources: List[Dict] = field(default_factory=list)
    scraped_content: List[Dict] = field(default_factory=list)

    # Final data for text editor
    has_professional_coverage: bool = False
    combined_review_data: Dict = field(default_factory=dict)

class EnhancedMediaVerificationAgent:
    """
    Enhanced media verification agent - AI-POWERED QUERY ANALYSIS VERSION

    Key improvements:
    - Uses AI to analyze queries and determine search strategy
    - Maps queries to Google's official place types intelligently
    - No hardcoded venue type mappings - fully scalable
    - Better relevance and accuracy for all query types
    """

    def __init__(self, config):
        self.config = config

        # Initialize configuration attributes with defaults
        self.rating_threshold = getattr(config, 'ENHANCED_RATING_THRESHOLD', 4.3)
        self.max_venues_to_verify = getattr(config, 'MAX_VENUES_TO_VERIFY', 5)
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        # Initialize Tavily API
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY')

        # Initialize Google Maps client for text search
        api_key = getattr(config, 'GOOGLE_MAPS_API_KEY2', None) or getattr(config, 'GOOGLE_MAPS_API_KEY', None)
        if api_key:
            self.gmaps = googlemaps.Client(key=api_key)
        else:
            self.gmaps = None

        # Load Google service account credentials for Places API
        self.places_client_primary = self._initialize_places_client('primary')
        self.places_client_secondary = self._initialize_places_client('secondary')

        # Determine if we have dual credentials
        self.has_dual_credentials = (self.places_client_primary is not None and 
                                     self.places_client_secondary is not None)

        if not self.places_client_primary:
            raise ValueError("No valid Google Places API credentials found")

        # API usage tracking
        self.api_usage = {'primary': 0, 'secondary': 0}

        logger.info("✅ Enhanced Media Verification Agent initialized with AI Query Analysis")
        if self.has_dual_credentials:
            logger.info("🔄 Dual credentials mode enabled - automatic load balancing")

    def _load_credentials_from_env(self, env_key: str, key_type: str):
        """Load service account credentials from environment variable"""
        try:
            creds_json_str = getattr(self.config, env_key, None)
            if not creds_json_str:
                logger.info(f"ℹ️ {key_type} credentials not found in environment")
                return None

            # Parse JSON string to dict
            credentials_info = json.loads(creds_json_str)

            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )

            logger.info(f"✅ {key_type} credentials loaded from environment")
            return credentials

        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {key_type} credentials: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading {key_type} credentials: {e}")
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

    async def search_nearby_enhanced(self, latitude: float, longitude: float, radius_meters: int = 1000):
        """Search using the appropriate client with rotation - FIXED API CALLS"""
        client, key_name = self._get_places_client()

        logger.info(f"🔍 Using {key_name} Places API client for search")

        # FIXED: Create search request with CORRECT API structure based on official docs
        try:
            # Import LatLng from the correct location
            from google.type import latlng_pb2

            # Create the LatLng object for the center
            center_point = latlng_pb2.LatLng(latitude=latitude, longitude=longitude)

            # Create the Circle object
            circle_area = places_v1.types.Circle(
                center=center_point,
                radius=radius_meters
            )

            # Add the circle to the location restriction
            location_restriction = places_v1.SearchNearbyRequest.LocationRestriction(
                circle=circle_area
            )

            request = places_v1.SearchNearbyRequest(
                location_restriction=location_restriction,
                included_types=["restaurant"],
                max_result_count=10,
                language_code="en"
            )
        except Exception as api_error:
            logger.error(f"❌ API structure creation failed: {api_error}")
            # Final fallback - use the googlemaps library instead
            return await self._fallback_to_googlemaps(latitude, longitude, radius_meters)

        try:
            # IMPORTANT: Set field mask in metadata (required for new Places API)
            metadata = [
                ("x-goog-fieldmask", 
                 "places.id,places.displayName,places.formattedAddress,places.location," +
                 "places.rating,places.userRatingCount,places.businessStatus,places.reviews")
            ]

            response = client.search_nearby(request=request, metadata=metadata)

            # Update usage counter
            self.api_usage[key_name] += 1

            logger.info(f"✅ Places API search completed using {key_name} client (usage: {self.api_usage[key_name]})")
            return response

        except Exception as e:
            logger.error(f"❌ Places API search failed with {key_name} client: {e}")

            # Try the other client if dual mode is available
            if self.has_dual_credentials:
                other_client = self.places_client_secondary if key_name == 'primary' else self.places_client_primary
                other_key = 'secondary' if key_name == 'primary' else 'primary'

                try:
                    logger.info(f"🔄 Retrying with {other_key} client")
                    response = other_client.search_nearby(request=request, metadata=metadata)
                    self.api_usage[other_key] += 1
                    return response
                except Exception as retry_error:
                    logger.error(f"❌ Both clients failed. Last error: {retry_error}")

            # Final fallback to googlemaps library
            logger.info("🔄 Falling back to googlemaps library")
            return await self._fallback_to_googlemaps(latitude, longitude, radius_meters)

    async def _analyze_query_for_search_strategy(self, query: str) -> Dict[str, Any]:
        """AI-powered query analysis using Google's official place types"""
        try:
            # Google's official place types
            google_place_types = [
                "bar", "wine_bar", "restaurant", "cafe", "coffee_shop", "bakery",
                "fast_food_restaurant", "fine_dining_restaurant", "pizza_restaurant",
                "chinese_restaurant", "italian_restaurant", "japanese_restaurant",
                "sushi_restaurant", "mexican_restaurant", "thai_restaurant"
            ]

            prompt = f"""Analyze this query and determine search strategy using Google's official place types.

    USER QUERY: "{query}"

    GOOGLE PLACE TYPES: {google_place_types}

    Return JSON only:
    {{
      "primary_intent": "brief description",
      "place_types": ["exact_google_type1", "exact_google_type2"], 
      "search_keywords": ["keyword1", "keyword2"],
      "approach": "both",
      "use_text_search": true,
      "use_places_api": true,
      "confidence": 0.9
    }}"""

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

            # Add null checks
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
            # Fallback for cocktail bars
            if any(term in query.lower() for term in ['cocktail', 'bar']):
                return {
                    'primary_intent': 'cocktail bars',
                    'place_types': ['bar'],
                    'search_keywords': ['cocktail bar', 'bar'],
                    'approach': 'both',
                    'use_text_search': True,
                    'use_places_api': True,
                    'confidence': 0.6
                }
            else:
                return {
                    'primary_intent': 'restaurants',
                    'place_types': ['restaurant'],
                    'search_keywords': [query],
                    'approach': 'both', 
                    'use_text_search': True,
                    'use_places_api': True,
                    'confidence': 0.5
                }
    
    async def _fallback_to_googlemaps(self, latitude: float, longitude: float, radius_meters: int):
        """Fallback to use the standard googlemaps library if Places API v1 fails"""
        try:
            import googlemaps

            # Use the same API key configuration as your GoogleMapsSearchAgent
            api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY2', None) or getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)

            if not api_key:
                logger.error("❌ No Google Maps API key found for fallback")
                return None

            gmaps = googlemaps.Client(key=api_key)

            # Use nearby search with the standard library
            response = gmaps.places_nearby(
                location=(latitude, longitude),
                radius=radius_meters,
                type='restaurant',
                language='en'
            )

            logger.info("✅ Fallback search completed using googlemaps library")

            # Convert the response to match the expected format
            class FallbackResponse:
                def __init__(self, results):
                    self.places = [self._convert_result(place) for place in results]

                def _convert_result(self, place):
                    # Convert googlemaps result to match Places API v1 format
                    class FallbackPlace:
                        def __init__(self, place_data):
                            self.id = place_data.get('place_id')

                            # Create display_name object
                            class DisplayName:
                                def __init__(self, text):
                                    self.text = text
                            self.display_name = DisplayName(place_data.get('name', 'Unknown'))

                            self.formatted_address = place_data.get('formatted_address', place_data.get('vicinity', ''))

                            # Create location object
                            geometry = place_data.get('geometry', {})
                            location_data = geometry.get('location', {})
                            class Location:
                                def __init__(self, lat, lng):
                                    self.latitude = lat
                                    self.longitude = lng
                            self.location = Location(location_data.get('lat'), location_data.get('lng'))

                            self.rating = place_data.get('rating')
                            self.user_rating_count = place_data.get('user_ratings_total')

                            # Business status
                            class BusinessStatus:
                                def __init__(self):
                                    self.name = 'OPERATIONAL'
                            self.business_status = BusinessStatus()

                            # Empty reviews for fallback
                            self.reviews = []

                    return FallbackPlace(place)

            return FallbackResponse(response.get('results', []))

        except Exception as fallback_error:
            logger.error(f"❌ Fallback to googlemaps library also failed: {fallback_error}")
            return None

    async def verify_and_enhance_venues(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        MAIN METHOD: Enhanced verification flow with AI-powered query analysis
        """
        try:
            logger.info("🚀 Starting Enhanced Media Verification Flow with AI Query Analysis")

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 1: AI-powered query analysis
            logger.info("🤖 Step 1: AI analyzing query and determining search strategy")
            search_strategy = await self._analyze_query_for_search_strategy(query)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"🎯 AI Strategy: {search_strategy['approach']} for '{search_strategy['primary_intent']}' using types {search_strategy['place_types']}")

            # Step 2: AI-guided Google Maps search
            logger.info("🔍 Step 2: AI-guided Google Maps search with reviews")
            venues_data = await self._ai_guided_google_search(coordinates, query, search_strategy, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"📍 Step 2: Found {len(venues_data)} venues from AI-guided search")

            if not venues_data:
                logger.warning("❌ No venues found from AI-guided search")
                return []

            # Step 3: AI venue selection with query context
            logger.info("🤖 Step 3: AI venue selection with query relevance analysis")
            selected_venues = await self._analyze_and_select_venues_with_query_context(
                venues_data, query, search_strategy, cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"✅ Step 3: Selected {len(selected_venues)} venues for verification")

            # Step 4: Media searches for selected venues
            logger.info("🔍 Step 4: Tavily media searches")
            await self._tavily_search_venues(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("🔍 Step 4: Completed media searches")

            # Step 5: AI analysis of media sources
            logger.info("📰 Step 5: AI analysis of media sources")
            await self._analyze_media_sources(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("📰 Step 5: Completed media source analysis")

            # Step 6: Prepare combined data for text editor
            self._prepare_combined_data(selected_venues)

            logger.info(f"✅ AI-powered enhanced verification completed for {len(selected_venues)} venues")
            return selected_venues

        except Exception as e:
            logger.error(f"❌ Error in AI-powered enhanced media verification: {e}")
            return []

    async def _ai_guided_google_search(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        search_strategy: Dict[str, Any],
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        Step 2: AI-guided Google Maps search using the determined strategy
        """
        try:
            latitude, longitude = coordinates

            logger.info(f"🔍 Performing AI-guided search for: '{query}'")
            logger.info(f"🎯 Strategy: {search_strategy['approach']} with types {search_strategy['place_types']}")

            venues_data = []

            # Execute search based on AI-determined strategy
            if search_strategy['use_text_search']:
                text_venues = await self._ai_text_search(
                    latitude, longitude, search_strategy['search_keywords']
                )
                venues_data.extend(text_venues)

            if search_strategy['use_places_api']:
                places_venues = await self._ai_places_search(
                    latitude, longitude, search_strategy['place_types']
                )
                venues_data.extend(places_venues)

            # Remove duplicates by place_id
            unique_venues = {}
            for venue in venues_data:
                if venue.place_id not in unique_venues:
                    unique_venues[venue.place_id] = venue

            final_venues = list(unique_venues.values())
            logger.info(f"🔍 AI-guided search found {len(final_venues)} unique venues")

            return final_venues

        except Exception as e:
            logger.error(f"❌ Error in AI-guided Google search: {e}")
            return []

    async def _ai_text_search(
        self, 
        latitude: float, 
        longitude: float, 
        search_keywords: List[str]
    ) -> List[EnhancedVenueData]:
        """
        AI-guided text search using determined keywords
        """
        try:
            if not self.gmaps:
                logger.warning("No Google Maps client available for text search")
                return []

            venues_data = []
            location = f"{latitude},{longitude}"

            # Try each AI-determined keyword
            for keyword in search_keywords[:3]:  # Limit to top 3 keywords
                try:
                    search_query = f"{keyword} near {latitude},{longitude}"
                    logger.info(f"🔍 AI text search: {search_query}")

                    response = self.gmaps.places(
                        query=search_query,
                        location=location,
                        radius=2000,  # 2km radius
                    )

                    results = response.get('results', [])
                    logger.info(f"📍 Text search for '{keyword}' returned {len(results)} results")

                    for place in results:
                        try:
                            venue = await self._convert_gmaps_result_to_venue_data(place, latitude, longitude)
                            if venue:
                                venues_data.append(venue)
                        except Exception as e:
                            logger.warning(f"Error converting text search result: {e}")
                            continue

                except Exception as e:
                    logger.warning(f"Text search failed for keyword '{keyword}': {e}")
                    continue

            logger.info(f"✅ AI text search completed, found {len(venues_data)} venues")
            return venues_data

        except Exception as e:
            logger.error(f"❌ Error in AI text search: {e}")
            return []

    async def _ai_places_search(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[EnhancedVenueData]:
        """
        AI-guided Places API search using determined place types
        """
        try:
            client, key_name = self._get_places_client()

            venues_data = []

            # Use AI-determined place types
            for place_type in place_types[:3]:  # Limit to top 3 types
                try:
                    logger.info(f"🔍 AI Places API search for type: {place_type}")

                    # Create search request for this specific type
                    from google.type import latlng_pb2

                    center_point = latlng_pb2.LatLng(latitude=latitude, longitude=longitude)
                    circle_area = places_v1.types.Circle(center=center_point, radius=2000)
                    location_restriction = places_v1.SearchNearbyRequest.LocationRestriction(circle=circle_area)

                    request = places_v1.SearchNearbyRequest(
                        location_restriction=location_restriction,
                        included_types=[place_type],  # Use AI-determined type
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
                        logger.info(f"📍 Places API for '{place_type}' returned {len(response.places)} results")

                        for place in response.places:
                            try:
                                venue = await self._convert_places_result_to_venue_data(place, latitude, longitude)
                                if venue:
                                    venues_data.append(venue)
                            except Exception as e:
                                logger.warning(f"Error converting Places API result: {e}")
                                continue

                except Exception as e:
                    logger.warning(f"Places API search failed for type '{place_type}': {e}")
                    continue

            logger.info(f"✅ AI Places API search completed, found {len(venues_data)} venues")
            return venues_data

        except Exception as e:
            logger.error(f"❌ Error in AI Places API search: {e}")
            return []

    async def _convert_gmaps_result_to_venue_data(self, place: Dict, user_lat: float, user_lng: float) -> Optional[EnhancedVenueData]:
        """Convert Google Maps API result to EnhancedVenueData"""
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

            return EnhancedVenueData(
                place_id=place_id,
                name=name,
                address=address,
                latitude=place_lat,
                longitude=place_lng,
                distance_km=distance_km,
                business_status=place.get('business_status', 'OPERATIONAL'),
                rating=place.get('rating'),
                user_ratings_total=place.get('user_ratings_total', 0),
                google_reviews=[]
            )

        except Exception as e:
            logger.warning(f"Error converting gmaps result: {e}")
            return None

    async def _convert_places_result_to_venue_data(self, place, user_lat: float, user_lng: float) -> Optional[EnhancedVenueData]:
        """Convert Places API result to EnhancedVenueData"""
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

            google_reviews = []
            if hasattr(place, 'reviews') and place.reviews:
                for review in place.reviews[:3]:
                    review_data = {
                        'rating': review.rating if hasattr(review, 'rating') else 0,
                        'text': review.text.text if hasattr(review, 'text') and hasattr(review.text, 'text') else "",
                        'time': review.publish_time if hasattr(review, 'publish_time') else None
                    }
                    google_reviews.append(review_data)

            return EnhancedVenueData(
                place_id=place_id,
                name=name,
                address=address,
                latitude=place_lat,
                longitude=place_lng,
                distance_km=distance_km,
                business_status=place.business_status if hasattr(place, 'business_status') else "OPERATIONAL",
                rating=place.rating if hasattr(place, 'rating') else None,
                user_ratings_total=place.user_rating_count if hasattr(place, 'user_rating_count') else 0,
                google_reviews=google_reviews
            )

        except Exception as e:
            logger.warning(f"Error converting Places API result: {e}")
            return None

    async def _analyze_and_select_venues_with_query_context(
        self, 
        venues_data: List[EnhancedVenueData], 
        query: str,
        search_strategy: Dict[str, Any],
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        Step 3: Enhanced AI analysis with full query context and search strategy
        """
        try:
            if not venues_data:
                logger.warning("No venues to analyze")
                return []

            # Prepare data for AI analysis with query context
            restaurant_data = []
            for venue in venues_data:
                venue_info = {
                    'place_id': venue.place_id,
                    'name': venue.name,
                    'rating': venue.rating,
                    'review_count': venue.user_ratings_total,
                    'reviews': venue.google_reviews[:3],  # Send top 3 reviews for analysis
                    'distance': venue.distance_km
                }
                restaurant_data.append(venue_info)

            # Enhanced prompt with full query context and search strategy
            prompt = f"""You are a food and venue expert analyzing Google search results to select the best matches for a specific user query.

    USER QUERY: "{query}"
    SEARCH INTENT: "{search_strategy['primary_intent']}"
    TARGET PLACE TYPES: {search_strategy['place_types']}
    AI CONFIDENCE: {search_strategy['confidence']}

    IMPORTANT: You must respond with valid JSON only. No additional text or explanation.

    Your task:
    1. Analyze if each venue matches the user's specific query and intent
    2. Rate venues based on BOTH relevance to query AND review quality
    3. Prioritize venues that clearly match what the user is looking for
    4. Consider venue type, menu items mentioned in reviews, and atmosphere

    Scoring criteria:
    - RELEVANCE (50%): Does this venue match the user's query? (cocktail bar for "cocktail bars")
    - QUALITY (30%): Review quality, detail, enthusiasm
    - SPECIFICITY (20%): Specific mentions of relevant items (cocktails, dishes, atmosphere)

    Rate each venue 0-10 based on combined score. Select venues scoring 7.0+.

    VENUES TO ANALYZE:
    {json.dumps(restaurant_data, indent=2)}

    Return only this JSON format:
    [
    {{
    "place_id": "venue_place_id",
    "selected": true/false,
    "overall_score": 8.5,
    "relevance_score": 9.0,
    "quality_score": 8.0,
    "reasoning": "High-quality cocktail bar with detailed reviews mentioning craft cocktails and mixology"
    }}
    ]"""

            try:
                # Send to OpenAI with timeout
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0.1
                    ),
                    timeout=30.0
                )

                # Add null checks
                if not response or not response.choices:
                    raise Exception("Empty AI response")

                analysis_text = response.choices[0].message.content
                if not analysis_text:
                    raise Exception("Empty content from AI")

                # Parse response
                analysis_text = analysis_text.strip()

                # Clean response and parse JSON
                if analysis_text.startswith('```json'):
                    analysis_text = analysis_text.replace('```json', '').replace('```', '').strip()

                analysis_results = json.loads(analysis_text)
                logger.info(f"🤖 AI analyzed {len(analysis_results)} venues with full query context")

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI analysis JSON: {e}")
                # Fallback: select top rated venues
                venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
                return venues_data[:3]

            except Exception as e:
                logger.error(f"❌ AI analysis failed: {e}")
                # Fallback: select top rated venues
                venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
                return venues_data[:3]

            # Update venues with AI analysis results
            selected_venues = []
            for analysis in analysis_results:
                place_id = analysis.get('place_id')

                # Find corresponding venue
                venue = next((v for v in venues_data if v.place_id == place_id), None)
                if venue and analysis.get('selected', False):
                    venue.review_quality_score = analysis.get('overall_score', 0)
                    venue.selected_for_verification = True
                    selected_venues.append(venue)

                    relevance = analysis.get('relevance_score', 0)
                    quality = analysis.get('quality_score', 0)
                    logger.debug(f"Selected {venue.name} (relevance: {relevance:.1f}, quality: {quality:.1f}, overall: {venue.review_quality_score:.1f})")

            # Sort by overall score and limit results
            selected_venues.sort(key=lambda x: x.review_quality_score, reverse=True)
            final_selection = selected_venues[:self.max_venues_to_verify]

            logger.info(f"✅ AI query-aware selection: {len(final_selection)} venues from {len(venues_data)} candidates")
            return final_selection

        except asyncio.TimeoutError:
            logger.error("❌ Venue analysis timed out")
            # Fallback: select top rated venues
            venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
            return venues_data[:3]
        except Exception as e:
            logger.error(f"❌ Error in venue analysis: {e}")
            # Fallback: select top rated venues
            venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
            return venues_data[:3]

    def _initialize_places_client(self, client_type: str):
        """Initialize Google Places API client"""
        try:
            if client_type == 'primary':
                creds_key = 'GOOGLE_APPLICATION_CREDENTIALS_JSON_PRIMARY'
            else:
                creds_key = 'GOOGLE_APPLICATION_CREDENTIALS_JSON_SECONDARY'

            credentials = self._load_credentials_from_env(creds_key, client_type)
            if not credentials:
                if client_type == 'primary':
                    logger.error("Primary credentials required but not found")
                    return None
                else:
                    logger.info("Secondary credentials not available")
                    return None

            try:
                client = places_v1.PlacesClient(credentials=credentials)
                logger.info(f"{client_type.title()} Places API client initialized")
                return client
            except Exception as e:
                logger.error(f"Failed to create {client_type} Places client: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to initialize {client_type} Places client: {e}")
            return None

    async def _tavily_search_venues(self, venues: List[EnhancedVenueData], cancel_check_fn=None):
        """
        Step 4: Tavily media search for each selected venue
        """
        if not self.tavily_api_key:
            logger.warning("⚠️ Tavily API key not available - skipping media search")
            return

        try:
            async with aiohttp.ClientSession() as session:
                for venue in venues:
                    if cancel_check_fn and cancel_check_fn():
                        break

                    # Create search queries
                    city = self._extract_city_from_address(venue.address)
                    search_queries = [
                        f"{venue.name} {city} restaurant review",
                        f"{venue.name} {city} food guide",
                        f"{venue.name} restaurant michelin",
                    ]

                    venue.media_search_results = []

                    for query in search_queries:
                        try:
                            tavily_payload = {
                                "api_key": self.tavily_api_key,
                                "query": query,
                                "search_depth": "basic",
                                "include_answer": False,
                                "include_images": False,
                                "include_raw_content": False,
                                "max_results": 5
                            }

                            async with session.post(
                                "https://api.tavily.com/search",
                                json=tavily_payload,
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    search_results = result.get('results', [])
                                    venue.media_search_results.extend(search_results)

                        except Exception as e:
                            logger.debug(f"Tavily search error for {venue.name}: {e}")
                            continue

                    logger.debug(f"{venue.name}: Found {len(venue.media_search_results)} media results")

        except Exception as e:
            logger.error(f"❌ Error in Tavily media search: {e}")

    async def _analyze_media_sources(self, venues: List[EnhancedVenueData], cancel_check_fn=None):
        """
        Step 5: AI analysis of media sources to identify professional guides - WITH JSON FIX
        """
        for venue in venues:
            if cancel_check_fn and cancel_check_fn():
                break

            if not venue.media_search_results:
                venue.professional_sources = []
                venue.has_professional_coverage = False
                continue

            try:
                # Enhanced prompt with clearer JSON requirements
                prompt = f"""You are a media analyst identifying professional restaurant guides and publications.

    IMPORTANT: Respond with valid JSON only. No additional text.

    IDENTIFY PROFESSIONAL SOURCES:
    - Food & travel magazines (Conde Nast, Forbes Travel, Food & Wine, etc.)
    - Local newspapers and magazines (Time Out, local papers)
    - Professional food critics and established food blogs
    - Tourism boards and official city guides
    - Restaurant award guides (Michelin, World's 50 Best, etc.)

    IGNORE:
    - TripAdvisor, Yelp, user review sites
    - Social media posts
    - Generic listicles
    - Travel booking sites
    - Personal blogs without professional credentials

    RESPOND ONLY WITH THIS JSON FORMAT:
    {{
      "professional_sources": [
        {{
          "url": "https://example.com/article",
          "title": "article title",
          "description": "description",
          "source_type": "food_magazine",
          "credibility_score": 9.0,
          "worth_scraping": true
        }}
      ],
      "total_results": 15,
      "professional_count": 3
    }}

    Media search results for {venue.name}:
    {json.dumps(venue.media_search_results[:10], indent=2)}"""  # Limit to 10 results to avoid token limits

                logger.debug(f"🔍 Analyzing media sources for {venue.name}")

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=1500
                    ),
                    timeout=20  # 20 second timeout
                )

                # Use the safe parsing method
                media_analysis = self._safe_parse_openai_response(
                    response, 
                    fallback_data={"professional_sources": [], "professional_count": 0}, 
                    context=f"media analysis for {venue.name}"
                )

                professional_sources = media_analysis.get('professional_sources', [])
                venue.professional_sources = professional_sources
                venue.has_professional_coverage = len(professional_sources) > 0

                logger.debug(f"✅ {venue.name}: Found {len(professional_sources)} professional sources")

            except asyncio.TimeoutError:
                logger.error(f"❌ Media analysis timed out for {venue.name}")
                venue.professional_sources = []
                venue.has_professional_coverage = False
            except Exception as e:
                logger.error(f"❌ Error analyzing media sources for {venue.name}: {e}")
                venue.professional_sources = []
                venue.has_professional_coverage = False

    async def _scrape_professional_content(self, venues: List[EnhancedVenueData], cancel_check_fn=None):
        """
        Step 6: Smart scraping of professional content
        For now, we'll prepare the URLs and mark for future scraping
        """
        for venue in venues:
            if cancel_check_fn and cancel_check_fn():
                break

            if not venue.professional_sources:
                continue

            # For now, just prepare the scraping targets
            # In production, this would call smart scraper API
            scraping_targets = []

            for source in venue.professional_sources[:3]:  # Limit to top 3 sources
                if source.get('worth_scraping', False):
                    scraping_targets.append({
                        'url': source['url'],
                        'title': source['title'],
                        'source_type': source['source_type'],
                        'content': f"[PLACEHOLDER] Content from {source['title']} - professional review of {venue.name}"
                        # TODO: Replace with actual smart scraper call
                    })

            venue.scraped_content = scraping_targets

            if scraping_targets:
                logger.debug(f"{venue.name}: Prepared {len(scraping_targets)} sources for scraping")

    def _prepare_combined_data(self, venues: List[EnhancedVenueData]):
        """Step 6: Prepare combined data for text editor"""
        for venue in venues:
            # Combine Google reviews and scraped professional content
            combined_data = {
                'google_reviews': venue.google_reviews,
                'professional_content': venue.scraped_content,
                'has_media_coverage': venue.has_professional_coverage,
                'media_sources': [s.get('title', 'Unknown') for s in venue.professional_sources],
                'quality_indicators': {
                    'review_quality_score': venue.review_quality_score,
                    'google_rating': venue.rating,
                    'review_count': venue.user_ratings_total,
                    'professional_mentions': len(venue.professional_sources)
                }
            }

            venue.combined_review_data = combined_data

    def _safe_parse_openai_response(self, response, fallback_data=None, context=""):
        """
        Safely parse OpenAI response with detailed error handling and logging

        Args:
            response: OpenAI response object
            fallback_data: Default data to return if parsing fails
            context: Context string for logging (e.g., "venue analysis", "media analysis")

        Returns:
            Parsed JSON data or fallback_data
        """
        try:
            if not response or not hasattr(response, 'choices') or not response.choices:
                logger.error(f"❌ Empty or invalid OpenAI response for {context}")
                return fallback_data or {}

            response_text = response.choices[0].message.content

            # Log the raw response for debugging
            logger.debug(f"🔍 Raw OpenAI response for {context}: {response_text[:200]}...")

            if not response_text or response_text.strip() == "":
                logger.error(f"❌ Empty response text from OpenAI for {context}")
                return fallback_data or {}

            # Clean up the response text
            response_text = response_text.strip()

            # Handle cases where the model returns text before the JSON
            if response_text.startswith('```json'):
                # Extract JSON from code block
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
            elif not response_text.startswith('{'):
                # Look for JSON in the response
                start_idx = response_text.find('{')
                if start_idx != -1:
                    response_text = response_text[start_idx:]

            # Try to parse the JSON
            try:
                parsed_data = json.loads(response_text)
                logger.debug(f"✅ Successfully parsed JSON for {context}")
                return parsed_data
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error for {context}: {e}")
                logger.error(f"Problematic text: {response_text}")
                return fallback_data or {}

        except Exception as e:
            logger.error(f"❌ Unexpected error parsing OpenAI response for {context}: {e}")
            return fallback_data or {}

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from address"""
        try:
            if not address:
                return "Unknown"

            parts = [part.strip() for part in address.split(',')]

            if len(parts) >= 2:
                return parts[1]  # Usually the city
            else:
                return parts[0] if parts else "Unknown"

        except Exception:
            return "Unknown"

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get statistics about the verification process"""
        return {
            'has_places_api_v1': True,
            'has_tavily_api': self.tavily_api_key is not None,
            'rating_threshold': self.rating_threshold,
            'max_venues': self.max_venues_to_verify,
            'ai_model': self.openai_model,
            'uses_langchain': False,
            'api_version': 'places_v1'
        }