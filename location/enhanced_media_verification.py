"""
Enhanced Media Verification Agent - PLACES API (NEW) ONLY VERSION

FIXED ISSUES:
1. Completely migrated to Places API (New) - no more mixing old/new methods
2. Removed deprecated maxResultCount, using pageSize instead
3. Uses only places_v1 client for all search operations
4. Consistent error handling and response processing
5. Future-proofed for Google's API changes
"""

import logging
import asyncio
import json
import aiohttp
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
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
    Enhanced media verification agent - PLACES API (NEW) ONLY VERSION

    Key improvements:
    - Uses ONLY Places API (New) for all search operations
    - No more mixing of old and new Google API methods
    - Future-proofed against Google's API changes
    - Consistent response handling across all searches
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

        # REMOVED: Old googlemaps.Client - using only Places API (New) now

        # Load Google service account credentials for Places API (New)
        self.places_client_primary = self._initialize_places_client('primary')
        self.places_client_secondary = self._initialize_places_client('secondary')

        # Determine if we have dual credentials
        self.has_dual_credentials = (self.places_client_primary is not None and 
                                     self.places_client_secondary is not None)

        if not self.places_client_primary:
            raise ValueError("No valid Google Places API credentials found")

        # API usage tracking
        self.api_usage = {'primary': 0, 'secondary': 0}

        logger.info("✅ Enhanced Media Verification Agent initialized with Places API (New) ONLY")
        if self.has_dual_credentials:
            logger.info("🔄 Dual credentials mode enabled - automatic load balancing")

    def _initialize_places_client(self, client_type: str):
        """Initialize Places API client with proper credentials"""
        try:
            if client_type == 'primary':
                env_key = 'GOOGLE_PLACES_SERVICE_ACCOUNT_JSON'
            elif client_type == 'secondary':
                env_key = 'GOOGLE_PLACES_SERVICE_ACCOUNT_JSON2'
            else:
                raise ValueError(f"Invalid client type: {client_type}")

            credentials = self._load_credentials_from_env(env_key, client_type)
            if not credentials:
                return None

            client = places_v1.PlacesClient(credentials=credentials)
            logger.info(f"✅ {client_type.capitalize()} Places API client initialized")
            return client

        except Exception as e:
            logger.error(f"❌ Failed to initialize {client_type} Places API client: {e}")
            return None

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

    async def verify_and_enhance_venues(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        MAIN METHOD: Enhanced verification flow with AI-powered query analysis
        Compatible with existing location orchestrator
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
            venues_data = await self._ai_guided_places_api_search(coordinates, query, search_strategy, cancel_check_fn)

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
            logger.error(f"❌ Error in Enhanced Media Verification Flow: {e}")
            return []

    async def enhanced_media_verification(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        Main entry point for enhanced media verification - PLACES API (NEW) ONLY
        This is an alias for verify_and_enhance_venues for compatibility
        """
        return await self.verify_and_enhance_venues(coordinates, query, cancel_check_fn)

    async def _ai_guided_places_api_search(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        search_strategy: Dict[str, Any],
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        AI-guided search using ONLY Places API (New) - both text and nearby search
        """
        try:
            latitude, longitude = coordinates

            logger.info(f"🔍 Performing AI-guided Places API (New) search for: '{query}'")
            logger.info(f"🎯 Strategy: {search_strategy['approach']} with types {search_strategy['place_types']}")

            venues_data = []

            # Execute text search using Places API (New) if needed
            if search_strategy['use_text_search']:
                text_venues = await self._places_api_text_search(
                    latitude, longitude, search_strategy['search_keywords']
                )
                venues_data.extend(text_venues)

            # Execute nearby search using Places API (New) if needed
            if search_strategy['use_places_api']:
                nearby_venues = await self._places_api_nearby_search(
                    latitude, longitude, search_strategy['place_types']
                )
                venues_data.extend(nearby_venues)

            # Remove duplicates by place_id
            unique_venues = {}
            for venue in venues_data:
                if venue.place_id not in unique_venues:
                    unique_venues[venue.place_id] = venue

            final_venues = list(unique_venues.values())
            logger.info(f"🔍 AI-guided Places API search found {len(final_venues)} unique venues")

            return final_venues

        except Exception as e:
            logger.error(f"❌ Error in AI-guided Places API search: {e}")
            return []

    async def _places_api_text_search(
        self, 
        latitude: float, 
        longitude: float, 
        search_keywords: List[str]
    ) -> List[EnhancedVenueData]:
        """
        Text search using Places API (New) ONLY
        """
        try:
            client, key_name = self._get_places_client()
            self.api_usage[key_name] += 1

            venues_data = []

            # Try each AI-determined keyword using Places API (New) Text Search
            for keyword in search_keywords[:3]:  # Limit to top 3 keywords
                try:
                    search_query = f"{keyword} near {latitude},{longitude}"
                    logger.info(f"🔍 Places API (New) text search: {search_query}")

                    # Create Text Search request
                    request = places_v1.SearchTextRequest(
                        text_query=search_query,
                        page_size=10,  # FIXED: Using pageSize instead of deprecated maxResultCount
                        language_code="en"
                    )

                    # Set field mask for the data we need
                    metadata = [
                        ("x-goog-fieldmask", 
                         "places.id,places.displayName,places.formattedAddress,places.location," +
                         "places.rating,places.userRatingCount,places.businessStatus,places.reviews")
                    ]

                    response = client.search_text(request=request, metadata=metadata)

                    if hasattr(response, 'places'):
                        logger.info(f"📍 Text search for '{keyword}' returned {len(response.places)} results")

                        for place in response.places:
                            try:
                                venue = await self._convert_places_result_to_venue_data(place, latitude, longitude)
                                if venue:
                                    venues_data.append(venue)
                            except Exception as e:
                                logger.warning(f"Error converting text search result: {e}")
                                continue

                except Exception as e:
                    logger.warning(f"Places API text search failed for keyword '{keyword}': {e}")
                    continue

            logger.info(f"✅ Places API (New) text search completed, found {len(venues_data)} venues")
            return venues_data

        except Exception as e:
            logger.error(f"❌ Error in Places API text search: {e}")
            return []

    async def _places_api_nearby_search(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[EnhancedVenueData]:
        """
        Nearby search using Places API (New) ONLY
        """
        try:
            client, key_name = self._get_places_client()
            self.api_usage[key_name] += 1

            venues_data = []

            # Use AI-determined place types
            for place_type in place_types[:3]:  # Limit to top 3 types
                try:
                    logger.info(f"🔍 Places API (New) nearby search for type: {place_type}")

                    # Create search request for this specific type
                    from google.type import latlng_pb2

                    center_point = latlng_pb2.LatLng(latitude=latitude, longitude=longitude)
                    circle_area = places_v1.types.Circle(center=center_point, radius=2000)
                    location_restriction = places_v1.SearchNearbyRequest.LocationRestriction(circle=circle_area)

                    request = places_v1.SearchNearbyRequest(
                        location_restriction=location_restriction,
                        included_types=[place_type],  # Use AI-determined type
                        max_result_count=10,  # This is still valid for nearby search
                        language_code="en"
                    )

                    metadata = [
                        ("x-goog-fieldmask", 
                         "places.id,places.displayName,places.formattedAddress,places.location," +
                         "places.rating,places.userRatingCount,places.businessStatus,places.reviews")
                    ]

                    response = client.search_nearby(request=request, metadata=metadata)

                    if hasattr(response, 'places'):
                        logger.info(f"📍 Nearby search for '{place_type}' returned {len(response.places)} results")

                        for place in response.places:
                            try:
                                venue = await self._convert_places_result_to_venue_data(place, latitude, longitude)
                                if venue:
                                    venues_data.append(venue)
                            except Exception as e:
                                logger.warning(f"Error converting nearby search result: {e}")
                                continue

                except Exception as e:
                    logger.warning(f"Places API nearby search failed for type '{place_type}': {e}")
                    continue

            logger.info(f"✅ Places API (New) nearby search completed, found {len(venues_data)} venues")
            return venues_data

        except Exception as e:
            logger.error(f"❌ Error in Places API nearby search: {e}")
            return []

    async def _convert_places_result_to_venue_data(
        self, 
        place, 
        search_lat: float, 
        search_lon: float
    ) -> Optional[EnhancedVenueData]:
        """
        Convert Places API (New) result to EnhancedVenueData - UNIFIED METHOD
        """
        try:
            # Extract place ID
            place_id = getattr(place, 'id', None)
            if not place_id:
                logger.warning("Place missing ID, skipping")
                return None

            # Extract name
            display_name = getattr(place, 'display_name', None)
            name = display_name.text if display_name and hasattr(display_name, 'text') else "Unknown"

            # Extract address
            formatted_address = getattr(place, 'formatted_address', 'Address not available')

            # Extract location
            location = getattr(place, 'location', None)
            if not location:
                logger.warning(f"Place {name} missing location, skipping")
                return None

            latitude = location.latitude
            longitude = location.longitude

            # Calculate distance
            distance_km = LocationUtils.haversine_distance(
                search_lat, search_lon, latitude, longitude
            )

            # Extract business status
            business_status = getattr(place, 'business_status', 'UNKNOWN')
            if hasattr(business_status, 'name'):
                business_status = business_status.name
            else:
                business_status = str(business_status)

            # Extract rating info
            rating = getattr(place, 'rating', None)
            user_ratings_total = getattr(place, 'user_rating_count', None)

            # Extract reviews if available
            google_reviews = []
            reviews = getattr(place, 'reviews', [])
            for review in reviews[:5]:  # Limit to first 5 reviews
                try:
                    review_data = {
                        'author_name': getattr(review.author_attribution, 'display_name', 'Anonymous') if hasattr(review, 'author_attribution') else 'Anonymous',
                        'rating': getattr(review, 'rating', None),
                        'text': getattr(review.text, 'text', '') if hasattr(review, 'text') else '',
                        'time': getattr(review, 'publish_time', None)
                    }
                    google_reviews.append(review_data)
                except Exception as e:
                    logger.warning(f"Error processing review: {e}")
                    continue

            # Create venue data object
            venue_data = EnhancedVenueData(
                place_id=place_id,
                name=name,
                address=formatted_address,
                latitude=latitude,
                longitude=longitude,
                distance_km=distance_km,
                business_status=business_status,
                rating=rating,
                user_ratings_total=user_ratings_total,
                google_reviews=google_reviews
            )

            return venue_data

        except Exception as e:
            logger.error(f"❌ Error converting Places API result to venue data: {e}")
            return None

    async def _analyze_query_for_search_strategy(self, query: str) -> Dict[str, Any]:
        """AI-powered query analysis using Google's official place types - PLACES API (NEW) VERSION"""
        try:
            # Google's official place types for restaurants and food venues
            google_place_types = [
                "bar", "wine_bar", "restaurant", "cafe", "coffee_shop", "bakery",
                "fast_food_restaurant", "fine_dining_restaurant", "pizza_restaurant",
                "chinese_restaurant", "italian_restaurant", "japanese_restaurant",
                "sushi_restaurant", "mexican_restaurant", "thai_restaurant",
                "indian_restaurant", "french_restaurant", "seafood_restaurant",
                "steakhouse", "vegetarian_restaurant", "vegan_restaurant",
                "breakfast_restaurant", "brunch_restaurant", "night_club",
                "cocktail_bar", "sports_bar", "pub", "tapas_restaurant",
                "ice_cream_shop", "food_truck", "meal_takeaway", "meal_delivery"
            ]

            prompt = f"""Analyze this query and determine search strategy using Google's official place types.

QUERY: "{query}"

GOOGLE PLACE TYPES: {google_place_types}

Instructions:
1. Identify the PRIMARY INTENT (what type of venue/experience the user wants)
2. Select 3-5 most relevant Google place types from the official list
3. Generate 3-5 effective search keywords for text search
4. Determine confidence level (1-10) in your analysis
5. Choose the best search approach

SEARCH APPROACHES:
- "text_search_primary": Use when query has specific names, cuisines, or descriptive terms
- "nearby_search_primary": Use when query is general ("restaurants", "bars", "food")
- "hybrid": Use both for comprehensive results (recommended for most queries)

Respond with JSON only:
{{
  "primary_intent": "brief description of what user wants",
  "place_types": ["type1", "type2", "type3"],
  "search_keywords": ["keyword1", "keyword2", "keyword3"],
  "approach": "text_search_primary|nearby_search_primary|hybrid",
  "use_text_search": true/false,
  "use_places_api": true/false,
  "confidence": 8,
  "reasoning": "Brief explanation"
}}"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing restaurant/venue search queries and mapping them to Google's official place types. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    strategy = json.loads(json_match.group())

                    # Validate required fields
                    required_fields = ['primary_intent', 'place_types', 'search_keywords', 'approach', 'use_text_search', 'use_places_api', 'confidence']
                    if not all(field in strategy for field in required_fields):
                        raise ValueError("Missing required fields in AI response")

                    # Ensure place_types are from the official list
                    valid_types = [pt for pt in strategy['place_types'] if pt in google_place_types]
                    if not valid_types:
                        # Fallback to restaurant if no valid types
                        valid_types = ["restaurant", "bar", "cafe"]
                    strategy['place_types'] = valid_types[:5]  # Limit to 5 types

                    logger.info(f"🤖 AI Query Analysis: {strategy.get('primary_intent', 'Unknown intent')}")
                    logger.info(f"🎯 Selected types: {strategy['place_types']}")

                    return strategy
                else:
                    raise ValueError("No JSON found in response")

            except Exception as parse_error:
                logger.warning(f"AI response parsing failed: {parse_error}")
                raise parse_error

        except Exception as e:
            logger.warning(f"AI query analysis failed: {e}, using fallback strategy")

            # Smart fallback based on query keywords
            query_lower = query.lower()

            # Try to detect intent from keywords
            if any(word in query_lower for word in ['cocktail', 'bar', 'drink', 'beer', 'wine']):
                place_types = ["bar", "cocktail_bar", "wine_bar", "restaurant"]
                primary_intent = "bars and drinking establishments"
            elif any(word in query_lower for word in ['coffee', 'cafe', 'espresso']):
                place_types = ["cafe", "coffee_shop", "bakery"]
                primary_intent = "coffee shops and cafes"  
            elif any(word in query_lower for word in ['pizza', 'italian']):
                place_types = ["pizza_restaurant", "italian_restaurant", "restaurant"]
                primary_intent = "pizza and Italian restaurants"
            elif any(word in query_lower for word in ['chinese', 'asian']):
                place_types = ["chinese_restaurant", "restaurant"]
                primary_intent = "Chinese and Asian restaurants"
            elif any(word in query_lower for word in ['sushi', 'japanese']):
                place_types = ["sushi_restaurant", "japanese_restaurant", "restaurant"]
                primary_intent = "sushi and Japanese restaurants"
            elif any(word in query_lower for word in ['mexican', 'taco']):
                place_types = ["mexican_restaurant", "restaurant"]
                primary_intent = "Mexican restaurants"
            elif any(word in query_lower for word in ['bakery', 'bread', 'pastry']):
                place_types = ["bakery", "cafe", "restaurant"]
                primary_intent = "bakeries and pastries"
            else:
                place_types = ["restaurant", "bar", "cafe"]
                primary_intent = "restaurants and food venues"

            return {
                "primary_intent": primary_intent,
                "place_types": place_types,
                "search_keywords": [query, f"{query} restaurant", f"{query} food"],
                "approach": "hybrid",
                "use_text_search": True,
                "use_places_api": True,
                "confidence": 5,
                "reasoning": f"Fallback strategy - detected intent: {primary_intent}"
            }

    # ... (rest of the methods remain the same as they don't involve Google Maps API calls)
    # Include: _ai_select_venues_for_verification, _tavily_search_venues, 
    # _analyze_media_sources, _prepare_combined_data, etc.

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

Rate each venue 0-10 based on combined score. Select venues scoring {self.rating_threshold}+.

VENUES: {json.dumps(restaurant_data, default=str)}

Respond with JSON:
{{
  "selected_venues": [
    {{
      "place_id": "venue_id",
      "quality_score": 8.5,
      "selection_reason": "Brief reason for selection"
    }}
  ],
  "analysis_summary": "Brief summary of selection process"
}}"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert food critic and venue analyst. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            content = response.choices[0].message.content.strip()

            # Parse AI response
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    ai_analysis = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in AI response")

                # Apply AI selection to venues
                selected_venues = []
                selected_place_ids = {v['place_id'] for v in ai_analysis.get('selected_venues', [])}

                for venue in venues_data:
                    if venue.place_id in selected_place_ids:
                        # Find the AI analysis for this venue
                        venue_analysis = next((v for v in ai_analysis['selected_venues'] if v['place_id'] == venue.place_id), None)
                        if venue_analysis:
                            venue.review_quality_score = venue_analysis.get('quality_score', 0.0)
                            venue.selected_for_verification = True
                            selected_venues.append(venue)

                # Limit to max venues
                selected_venues = selected_venues[:self.max_venues_to_verify]

                logger.info(f"🤖 AI selected {len(selected_venues)} venues: {ai_analysis.get('analysis_summary', 'No summary')}")
                return selected_venues

            except Exception as parse_error:
                logger.warning(f"AI venue analysis parsing failed: {parse_error}")
                # Fallback to simple rating-based selection
                return self._fallback_venue_selection(venues_data)

        except Exception as e:
            logger.error(f"❌ Error in AI venue analysis: {e}")
            # Fallback to simple rating-based selection
            return self._fallback_venue_selection(venues_data)

    def _fallback_venue_selection(self, venues_data: List[EnhancedVenueData]) -> List[EnhancedVenueData]:
        """Fallback venue selection based on rating and review count"""
        try:
            # Sort by rating and review count
            sorted_venues = sorted(
                venues_data,
                key=lambda x: (x.rating or 0, x.user_ratings_total or 0),
                reverse=True
            )

            # Select top venues that meet rating threshold
            selected_venues = []
            for venue in sorted_venues:
                if venue.rating and venue.rating >= self.rating_threshold:
                    venue.review_quality_score = venue.rating
                    venue.selected_for_verification = True
                    selected_venues.append(venue)

                    if len(selected_venues) >= self.max_venues_to_verify:
                        break

            logger.info(f"📊 Fallback selection: {len(selected_venues)} venues with rating >= {self.rating_threshold}")
            return selected_venues

        except Exception as e:
            logger.error(f"❌ Even fallback venue selection failed: {e}")
            return venues_data[:self.max_venues_to_verify]  # Just return first few venues

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
        Step 5: AI analysis of media sources to identify professional guides
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

EXCLUDE:
- User review sites (Yelp, TripAdvisor, Google Reviews)
- Social media posts (Instagram, Facebook)
- General business directories
- Personal blogs without credibility

Analyze these search results for "{venue.name}":
{json.dumps(venue.media_search_results[:10], default=str)}

Response format:
{{
  "professional_sources": [
    {{
      "url": "source_url",
      "title": "source_title", 
      "source_type": "food_magazine|local_newspaper|tourism_guide|award_guide|food_blog",
      "credibility_score": 8.5,
      "worth_scraping": true,
      "reason": "Brief explanation"
    }}
  ],
  "has_professional_coverage": true,
  "summary": "Brief analysis summary"
}}"""

                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert media analyst. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    timeout=15
                )

                content = response.choices[0].message.content.strip()

                # Parse JSON response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())

                        # Filter professional sources by credibility score
                        professional_sources = [
                            source for source in analysis.get('professional_sources', [])
                            if source.get('credibility_score', 0) >= getattr(self.config, 'PROFESSIONAL_SOURCE_MIN_SCORE', 7.0)
                        ]

                        venue.professional_sources = professional_sources
                        venue.has_professional_coverage = len(professional_sources) > 0

                        logger.debug(f"{venue.name}: Found {len(professional_sources)} professional sources")
                    else:
                        raise ValueError("No JSON found in response")

                except Exception as parse_error:
                    logger.warning(f"Media analysis parsing failed for {venue.name}: {parse_error}")
                    venue.professional_sources = []
                    venue.has_professional_coverage = False

            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Media analysis timed out for {venue.name}")
                venue.professional_sources = []
                venue.has_professional_coverage = False
            except Exception as e:
                logger.error(f"❌ Error analyzing media sources for {venue.name}: {e}")
                venue.professional_sources = []
                venue.has_professional_coverage = False

    def _prepare_combined_data(self, venues: List[EnhancedVenueData]):
        """Step 6: Prepare combined data for text editor"""
        for venue in venues:
            # Combine Google reviews and scraped professional content
            combined_data = {
                'google_reviews': venue.google_reviews,
                'professional_sources': venue.professional_sources,
                'scraped_content': venue.scraped_content,
                'has_media_coverage': venue.has_professional_coverage,
                'review_quality_score': venue.review_quality_score
            }

            # Add summary information for text editor
            combined_data['venue_summary'] = {
                'name': venue.name,
                'address': venue.address,
                'rating': venue.rating,
                'review_count': venue.user_ratings_total,
                'distance_km': venue.distance_km,
                'professional_coverage': venue.has_professional_coverage
            }

            venue.combined_review_data = combined_data
            logger.debug(f"Prepared combined data for {venue.name}")

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from address for search queries"""
        try:
            # Simple extraction - split by comma and get likely city part
            parts = address.split(',')
            if len(parts) >= 2:
                # Usually city is the second-to-last part before country/postal
                city_part = parts[-2].strip()
                # Remove postal codes and numbers
                import re
                city = re.sub(r'\d+', '', city_part).strip()
                return city if city else "restaurant"
            return "restaurant"
        except:
            return "restaurant"