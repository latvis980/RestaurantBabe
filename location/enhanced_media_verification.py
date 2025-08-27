# location/enhanced_media_verification.py
"""
Enhanced Media Verification Agent - NEW PLACES API VERSION

FIXED ISSUES:
1. Using NEW Google Places API v1 (google-maps-places library)
2. Correct method names: search_nearby() and get_place()
3. Removed LangChain dependencies - location flow doesn't use LangChain
4. Direct OpenAI API calls instead of LangChain
5. Fixed aiohttp timeout usage

This agent implements the new enhanced location-based restaurant verification flow:
1. Google Maps search with enhanced fields (business_status, rating, reviews)
2. AI-powered review analysis to select best restaurants
3. Tavily media search for professional coverage
4. AI analysis of media sources to identify professional guides
5. Smart scraping of professional content
6. Combined data preparation for text editor
"""

import logging
import asyncio
import json
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

# Import existing utilities and models
from location.location_utils import LocationUtils

# Direct OpenAI API instead of LangChain
import openai

# NEW Google Maps Places API v1
from google.maps import places_v1

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
    Enhanced media verification agent - NEW PLACES API VERSION
    """

    def __init__(self, config):
        self.config = config

        # Initialize NEW Google Maps Places API v1 client
        try:
            self.places_client = places_v1.PlacesClient()
            logger.info("✅ Google Maps Places API v1 client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Places API v1 client: {e}")
            raise

        # Initialize Tavily API
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY', None)
        if not self.tavily_api_key:
            logger.warning("⚠️ TAVILY_API_KEY not found - media verification will be limited")

        # Initialize OpenAI client directly (no LangChain)
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')

        # Configuration
        self.rating_threshold = getattr(config, 'ENHANCED_RATING_THRESHOLD', 4.5)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Enhanced Media Verification Agent initialized (NEW Places API)")

    async def verify_and_enhance_venues(
        self,
        coordinates: Tuple[float, float],
        query: str,
        cancel_check_fn=None
    ) -> List[EnhancedVenueData]:
        """
        Main verification flow following the new enhanced process
        """
        try:
            logger.info("🔍 Starting enhanced media verification flow")

            # Step 1: Enhanced Google Maps search
            venues_data = await self._enhanced_google_search(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            if not venues_data:
                logger.info("No venues found in Google Maps search")
                return []

            logger.info(f"📍 Step 1: Found {len(venues_data)} venues from Google Maps")

            # Step 2: AI analysis of reviews - select best restaurants
            selected_venues = await self._analyze_and_select_venues(venues_data, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            if not selected_venues:
                logger.info("No venues selected after review analysis")
                return []

            logger.info(f"🤖 Step 2: Selected {len(selected_venues)} venues after AI review analysis")

            # Step 4: Tavily search for each selected venue
            await self._tavily_search_venues(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("🔍 Step 4: Completed Tavily media searches")

            # Step 5: AI analysis of media sources
            await self._analyze_media_sources(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("🎯 Step 5: Completed media source analysis")

            # Step 6: Smart scraping of professional sources
            await self._scrape_professional_content(selected_venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info("📰 Step 6: Completed professional content scraping")

            # Prepare combined data for text editor
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
        Step 1: Enhanced Google Maps search using NEW Places API v1
        """
        try:
            latitude, longitude = coordinates

            # Create search request using NEW Places API v1
            request = places_v1.SearchNearbyRequest(
                location_restriction=places_v1.LocationRestriction(
                    circle=places_v1.Circle(
                        center=places_v1.LatLng(latitude=latitude, longitude=longitude),
                        radius=2000  # 2km radius
                    )
                ),
                included_types=["restaurant"],
                language_code="en",
                max_result_count=self.max_venues_to_verify,
                # Request fields we need
                # field_mask is handled automatically by the client
            )

            # Make the search request
            response = await asyncio.to_thread(self.places_client.search_nearby, request=request)

            venues_data = []

            for place in response.places:
                if cancel_check_fn and cancel_check_fn():
                    break

                try:
                    # Extract basic information
                    place_id = place.id
                    name = place.display_name.text if place.display_name else "Unknown"
                    address = place.formatted_address if place.formatted_address else "Unknown address"

                    # Extract location
                    if place.location:
                        place_lat = place.location.latitude
                        place_lng = place.location.longitude
                    else:
                        continue

                    # Calculate distance
                    distance_km = LocationUtils.calculate_distance(
                        coordinates, (place_lat, place_lng)
                    )

                    # Extract reviews
                    google_reviews = []
                    if hasattr(place, 'reviews') and place.reviews:
                        for review in place.reviews[:5]:  # Limit to 5 reviews
                            google_reviews.append({
                                'author_name': review.author_attribution.display_name if review.author_attribution else '',
                                'rating': review.rating if hasattr(review, 'rating') else None,
                                'text': review.text.text if review.text else '',
                                'relative_time_description': review.relative_publish_time_description if hasattr(review, 'relative_publish_time_description') else ''
                            })

                    # Extract rating and business status
                    rating = place.rating if hasattr(place, 'rating') else None
                    user_ratings_total = place.user_rating_count if hasattr(place, 'user_rating_count') else None
                    business_status = place.business_status.name if hasattr(place, 'business_status') else 'OPERATIONAL'

                    # Create enhanced venue data
                    venue_data = EnhancedVenueData(
                        place_id=place_id,
                        name=name,
                        address=address,
                        latitude=place_lat,
                        longitude=place_lng,
                        distance_km=round(distance_km, 2) if distance_km else 0.0,
                        business_status=business_status,
                        rating=rating,
                        user_ratings_total=user_ratings_total,
                        google_reviews=google_reviews
                    )

                    # Filter out closed restaurants and low ratings
                    if (business_status not in ['CLOSED_PERMANENTLY', 'CLOSED_TEMPORARILY'] and
                        (not rating or rating >= self.rating_threshold)):
                        venues_data.append(venue_data)

                except Exception as e:
                    logger.warning(f"Error processing place {place.id if hasattr(place, 'id') else 'unknown'}: {e}")
                    continue

            # Sort by rating and return
            venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
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
        Step 2: AI analysis to select venues with the best reviews
        Direct OpenAI API calls instead of LangChain
        """
        try:
            # Prepare data for AI analysis
            restaurant_data = []
            for venue in venues_data:
                restaurant_data.append({
                    'place_id': venue.place_id,
                    'name': venue.name,
                    'rating': venue.rating,
                    'review_count': venue.user_ratings_total,
                    'reviews': venue.google_reviews[:3]  # Send top 3 reviews for analysis
                })

            # Direct OpenAI API call
            prompt = f"""You are a food expert analyzing Google Reviews to identify the best restaurants.

Look for reviews that are:
- DETAILED and descriptive (not just "great place!")
- WARM and emotional (genuine enthusiasm)
- SPECIFIC about dishes, cocktails, or menu items
- Show personal experience and genuine appreciation

Rate each restaurant from 0-10 based on review quality. Prioritize restaurants where reviewers:
- Name specific dishes or drinks
- Describe flavors, atmosphere, service details
- Show genuine emotional connection to the experience
- Provide context about why they loved it

Return JSON format:
{{
  "analysis": [
    {{
      "place_id": "place_id_here", 
      "name": "restaurant_name",
      "quality_score": 8.5,
      "selected": true,
      "reasoning": "Reviews mention specific dishes like 'amazing truffle pasta' and 'perfectly crafted negronis'. Multiple reviewers describe warm atmosphere and exceptional service."
    }}
  ]
}}

Analyze these restaurants and their Google Reviews:

{json.dumps(restaurant_data, indent=2)}"""

            # Make direct OpenAI API call
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            # Handle OpenAI response properly
            try:
                response_text = response.choices[0].message.content
                analysis_data = json.loads(response_text)
                analysis_results = analysis_data.get('analysis', [])

            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"❌ Error in venue analysis: {e}")
                logger.debug(f"Response: {response_text}")
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
                    venue.review_quality_score = analysis.get('quality_score', 0)
                    venue.selected_for_verification = True
                    selected_venues.append(venue)

            # Sort by quality score
            selected_venues.sort(key=lambda x: x.review_quality_score, reverse=True)

            return selected_venues[:5]  # Limit to top 5 quality venues

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
        Step 5: AI analysis of media sources to identify professional guides
        Direct OpenAI API calls instead of LangChain
        """
        for venue in venues:
            if cancel_check_fn and cancel_check_fn():
                break

            if not venue.media_search_results:
                venue.professional_sources = []
                venue.has_professional_coverage = False
                continue

            try:
                # Direct OpenAI API call
                prompt = f"""You are a media analyst identifying professional restaurant guides and publications.

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

Return JSON:
{{
  "professional_sources": [
    {{
      "url": "https://example.com/article",
      "title": "article title",
      "description": "description",
      "source_type": "food_magazine|local_newspaper|tourism_guide|food_critic",
      "credibility_score": 9.0,
      "worth_scraping": true
    }}
  ],
  "total_results": 15,
  "professional_count": 3
}}

Analyze these Tavily search results for restaurant media coverage:

{json.dumps(venue.media_search_results, indent=2)}"""

                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )

                # Handle OpenAI response properly
                try:
                    response_text = response.choices[0].message.content
                    media_analysis = json.loads(response_text)
                    professional_sources = media_analysis.get('professional_sources', [])

                except (json.JSONDecodeError, AttributeError) as e:
                    logger.error(f"❌ Error parsing media analysis for {venue.name}: {e}")
                    professional_sources = []

                venue.professional_sources = professional_sources
                venue.has_professional_coverage = len(professional_sources) > 0

                logger.debug(f"{venue.name}: Found {len(professional_sources)} professional sources")

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
        """
        Prepare combined review and media data for the text editor
        """
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