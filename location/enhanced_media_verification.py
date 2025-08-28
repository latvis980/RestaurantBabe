# location/enhanced_media_verification.py - FIXED VERSION FOR API COMPATIBILITY
"""
Enhanced Media Verification Agent - NEW PLACES API VERSION - FIXED API CALLS

FIXED ISSUES:
1. Fixed LocationRestriction API call - use correct import structure
2. Updated to use proper Circle and LatLng from places_v1.types
3. Added proper error handling for API version differences
4. Maintained all existing functionality with corrected API calls
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
    Enhanced media verification agent - NEW PLACES API VERSION - FIXED
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
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY', None)
        if not self.tavily_api_key:
            logger.warning("⚠️ TAVILY_API_KEY not found - media verification will be limited")

        # Initialize dual Places clients using environment variables
        self.places_client_primary = None
        self.places_client_secondary = None
        self.has_dual_credentials = False

        try:
            # Initialize primary client
            primary_creds = self._get_credentials_from_env('PRIMARY')
            if primary_creds:
                self.places_client_primary = places_v1.PlacesClient(credentials=primary_creds)
                logger.info("✅ Primary Google Places API v1 client initialized")

            # Initialize secondary client
            secondary_creds = self._get_credentials_from_env('SECONDARY')
            if secondary_creds:
                self.places_client_secondary = places_v1.PlacesClient(credentials=secondary_creds)
                self.has_dual_credentials = True
                logger.info("✅ Secondary Google Places API v1 client initialized - dual mode enabled")

            # Fallback to default ADC if no environment variables
            if not self.places_client_primary:
                self.places_client_primary = places_v1.PlacesClient()
                logger.info("⚠️ Using default ADC for Google Places API v1")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Places API v1 clients: {e}")
            raise

        # Track usage for rotation (matching your existing pattern)
        self.api_usage = {
            'primary': 0,
            'secondary': 0
        }

    def _get_credentials_from_env(self, key_type: str):
        """Get credentials from Railway environment variables"""
        try:
            env_var = f'GOOGLE_APPLICATION_CREDENTIALS_JSON_{key_type}'
            creds_json_str = os.environ.get(env_var)

            if not creds_json_str:
                logger.warning(f"No {key_type} credentials found in environment")
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

    async def _fallback_to_googlemaps(self, latitude: float, longitude: float, radius_meters: int):
        """Fallback to use the standard googlemaps library if Places API v1 fails"""
        try:
            import googlemaps

            # Use the same API key configuration as your GoogleMapsSearchAgent
            api_key = getattr(self.config, 'GOOGLE_MAPS_KEY2', None) or getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)

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

            logger.info(f"✅ Fallback search completed using googlemaps library")

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
        Main entry point for enhanced media verification flow

        Steps:
        1. Enhanced Google Maps search with reviews
        2. AI-powered venue selection based on review quality
        3. Tavily media searches for selected venues
        4. AI analysis of media sources to identify professional guides
        5. Combined data preparation for text editor
        """
        try:
            logger.info("🚀 Starting Enhanced Media Verification Flow")

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 1: Enhanced Google search with reviews
            logger.info("🔍 Step 1: Enhanced Google Maps search with reviews")
            venues_data = await self._enhanced_google_search(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"📍 Step 1: Found {len(venues_data)} venues from Google Maps")

            if not venues_data:
                logger.warning("❌ No venues found from Google Maps search")
                return []

            # Step 2: AI venue selection
            logger.info("🤖 Step 2: AI-powered venue selection based on reviews")
            selected_venues = await self._analyze_and_select_venues(venues_data, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"✅ Step 2: Selected {len(selected_venues)} venues for verification")

            # Step 3: Media searches for selected venues
            logger.info("🔍 Step 3: Tavily media searches")
            await self._tavily_search_venues(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("🔍 Step 4: Completed media searches")

            # Step 4: AI analysis of media sources
            logger.info("📰 Step 5: AI analysis of media sources")
            await self._analyze_media_sources(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("📰 Step 5: Completed media source analysis")

            # Step 5: Prepare combined data for text editor
            self._prepare_combined_data(selected_venues)

            logger.info(f"✅ Enhanced verification completed for {len(selected_venues)} venues")
            return selected_venues

        except Exception as e:
            logger.error(f"❌ Error in enhanced media verification: {e}")
            return []

    async def _enhanced_google_search(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        Step 1: Enhanced Google Maps search using NEW Places API v1 - FIXED
        """
        try:
            latitude, longitude = coordinates

            # Use the fixed search method
            response = await self.search_nearby_enhanced(latitude, longitude, 2000)

            if not response or not hasattr(response, 'places'):
                logger.warning("No response or places from enhanced search")
                return []

            venues_data = []

            for place in response.places:
                if cancel_check_fn and cancel_check_fn():
                    break

                try:
                    # CORRECTED: Access place attributes properly for new API
                    place_id = place.id if hasattr(place, 'id') else None
                    name = place.display_name.text if hasattr(place, 'display_name') and place.display_name else "Unknown"
                    address = place.formatted_address if hasattr(place, 'formatted_address') else "Unknown address"

                    # Extract location
                    if hasattr(place, 'location') and place.location:
                        place_lat = place.location.latitude
                        place_lng = place.location.longitude
                    else:
                        continue

                    # Calculate distance
                    distance_km = LocationUtils.calculate_distance(
                        coordinates, (place_lat, place_lng)
                    )

                    # Extract reviews - CORRECTED for new API
                    google_reviews = []
                    if hasattr(place, 'reviews') and place.reviews:
                        for review in place.reviews[:5]:  # Limit to 5 reviews
                            google_reviews.append({
                                'author_name': review.author_attribution.display_name if hasattr(review, 'author_attribution') and review.author_attribution else '',
                                'rating': review.rating if hasattr(review, 'rating') else None,
                                'text': review.text.text if hasattr(review, 'text') and review.text else '',
                                'relative_time_description': review.relative_publish_time_description if hasattr(review, 'relative_publish_time_description') else ''
                            })

                    # Extract rating and business status - CORRECTED
                    rating = place.rating if hasattr(place, 'rating') else None
                    user_ratings_total = place.user_rating_count if hasattr(place, 'user_rating_count') else None

                    # Business status handling
                    if hasattr(place, 'business_status'):
                        business_status = place.business_status.name if hasattr(place.business_status, 'name') else 'OPERATIONAL'
                    else:
                        business_status = 'OPERATIONAL'

                    # Create enhanced venue data
                    venue_data = EnhancedVenueData(
                        place_id=place_id,
                        name=name,
                        address=address,
                        latitude=place_lat,
                        longitude=place_lng,
                        distance_km=distance_km,
                        business_status=business_status,
                        rating=rating,
                        user_ratings_total=user_ratings_total,
                        google_reviews=google_reviews
                    )

                    venues_data.append(venue_data)

                except Exception as e:
                    logger.warning(f"Error processing place: {e}")
                    continue

            return venues_data

        except Exception as e:
            logger.error(f"❌ Error in enhanced Google search: {e}")
            return []

    async def _analyze_and_select_venues(
        self, 
        venues_data: List[EnhancedVenueData], 
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        Step 2: AI analysis to select venues with the best reviews - WITH JSON FIX
        """
        try:
            if not venues_data:
                logger.warning("No venues to analyze")
                return []

            # Prepare data for AI analysis
            restaurant_data = []
            for venue in venues_data:
                venue_info = {
                    'place_id': venue.place_id,
                    'name': venue.name,
                    'rating': venue.rating,
                    'review_count': venue.user_ratings_total,
                    'reviews': venue.google_reviews[:3]  # Send top 3 reviews for analysis
                }
                restaurant_data.append(venue_info)

            # Enhanced prompt with clearer JSON format requirements
            prompt = f"""You are a food expert analyzing Google Reviews to identify the best restaurants.

    IMPORTANT: You must respond with valid JSON only. No additional text or explanation.

    Look for reviews that are:
    - DETAILED and descriptive (not just "great place!")
    - WARM and emotional (genuine enthusiasm)
    - SPECIFIC about dishes, cocktails, or menu items
    - Show personal experience and genuine appreciation

    Rate each restaurant from 0-10 based on review quality. Select the top restaurants with scores 7.0 and above.

    RESPOND ONLY WITH THIS JSON FORMAT:
    {{
      "analysis": [
        {{
          "place_id": "place_id_here", 
          "name": "restaurant_name",
          "quality_score": 8.5,
          "selected": true,
          "reasoning": "Brief explanation"
        }}
      ]
    }}

    Restaurants to analyze:
    {json.dumps(restaurant_data, indent=2)}"""

            logger.debug(f"🤖 Sending venue analysis request for {len(restaurant_data)} venues")

            # Make OpenAI API call with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2000
                ),
                timeout=30  # 30 second timeout
            )

            # Use the safe parsing method
            analysis_data = self._safe_parse_openai_response(
                response, 
                fallback_data={"analysis": []}, 
                context="venue analysis"
            )

            analysis_results = analysis_data.get('analysis', [])

            if not analysis_results:
                logger.warning("No analysis results from AI, using rating-based fallback")
                # Fallback: select top rated venues
                venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
                return venues_data[:self.max_venues_to_verify]

            # Update venues with AI analysis results
            selected_venues = []
            for analysis in analysis_results:
                place_id = analysis.get('place_id')

                # Find corresponding venue
                venue = next((v for v in venues_data if v.place_id == place_id), None)
                if venue and analysis.get('selected', False):
                    venue.review_quality_score = analysis.get('quality_score', 0)
                    venue.selected_for_verification = True
                    selected_venues.append(venue)
                    logger.debug(f"Selected {venue.name} with quality score {venue.review_quality_score}")

            # Sort by quality score and limit results
            selected_venues.sort(key=lambda x: x.review_quality_score, reverse=True)
            final_selection = selected_venues[:self.max_venues_to_verify]

            logger.info(f"✅ AI selected {len(final_selection)} venues from {len(venues_data)} candidates")
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