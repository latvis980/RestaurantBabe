# location/enhanced_media_verification.py
"""
Enhanced Media Verification Agent

This agent implements the new enhanced location-based restaurant verification flow:
1. Google Maps search with enhanced fields (business_status, rating, reviews)
2. AI-powered review analysis to select best restaurants
3. Tavily media search for professional coverage
4. AI analysis of media sources to identify professional guides
5. Smart scraping of professional content
6. Combined data preparation for text editor

Uses Places API (New) with correct field names for 2025.
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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
    Enhanced media verification agent that follows the new restaurant editor workflow
    """

    def __init__(self, config):
        self.config = config

        # Initialize Google Maps client (reuse existing setup)
        api_key = getattr(config, 'GOOGLE_MAPS_KEY2', None) or getattr(config, 'GOOGLE_MAPS_API_KEY', None)
        if not api_key:
            raise ValueError("Google Maps API key required")

        import googlemaps
        self.gmaps = googlemaps.Client(key=api_key)

        # Initialize Tavily API
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY', None)
        if not self.tavily_api_key:
            logger.warning("⚠️ TAVILY_API_KEY not found - media verification will be limited")

        # Initialize AI models for analysis
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )

        # Configuration
        self.rating_threshold = getattr(config, 'ENHANCED_RATING_THRESHOLD', 4.5)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        # Setup AI prompts
        self._setup_prompts()

        logger.info("✅ Enhanced Media Verification Agent initialized")

    def _setup_prompts(self):
        """Setup AI prompts for review analysis and media evaluation"""

        # Step 2: Review quality analysis prompt
        self.review_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a food expert analyzing Google Reviews to identify the best restaurants.

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
}}"""),
            ("human", "Analyze these restaurants and their Google Reviews:\n\n{restaurant_data}")
        ])

        # Step 5: Media source analysis prompt
        self.media_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a media analyst identifying professional restaurant guides and publications.

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
}}"""),
            ("human", "Analyze these Tavily search results for restaurant media coverage:\n\n{search_results}")
        ])

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
        Step 1: Enhanced Google Maps search with business_status, rating, and reviews
        Uses Places API (New) field names
        """
        try:
            latitude, longitude = coordinates

            # Use text search for better query handling
            search_results = self.gmaps.places_nearby(
                location=(latitude, longitude),
                radius=2000,  # 2km radius
                type='restaurant',
                keyword=query
            )

            venues_data = []

            for place in search_results.get('results', []):
                if cancel_check_fn and cancel_check_fn():
                    break

                # Get enhanced place details using Places API (New) fields
                place_id = place.get('place_id')
                if not place_id:
                    continue

                # Fetch detailed place data with enhanced fields
                try:
                    # Use correct field names for Places API
                    details = self.gmaps.place(
                        place_id=place_id,
                        fields=[
                            'name', 'formatted_address', 'geometry', 
                            'business_status', 'rating', 'user_ratings_total',
                            'reviews'
                        ]
                    )

                    place_details = details.get('result', {})

                    # Extract location
                    geometry = place_details.get('geometry', {})
                    location = geometry.get('location', {})
                    place_lat = location.get('lat')
                    place_lng = location.get('lng')

                    if not place_lat or not place_lng:
                        continue

                    # Calculate distance
                    distance_km = LocationUtils.calculate_distance(
                        coordinates, (place_lat, place_lng)
                    )

                    # Extract reviews (up to 5 from Google API)
                    google_reviews = place_details.get('reviews', [])
                    processed_reviews = []

                    for review in google_reviews:
                        processed_reviews.append({
                            'author_name': review.get('author_name', ''),
                            'rating': review.get('rating'),
                            'text': review.get('text', ''),
                            'time': review.get('time'),
                            'relative_time_description': review.get('relative_time_description', '')
                        })

                    # Create enhanced venue data
                    venue_data = EnhancedVenueData(
                        place_id=place_id,
                        name=place_details.get('name', ''),
                        address=place_details.get('formatted_address', ''),
                        latitude=place_lat,
                        longitude=place_lng,
                        distance_km=round(distance_km, 2) if distance_km else 0.0,
                        business_status=place_details.get('business_status', 'OPERATIONAL'),
                        rating=place_details.get('rating'),
                        user_ratings_total=place_details.get('user_ratings_total'),
                        google_reviews=processed_reviews
                    )

                    # Filter out closed restaurants and low ratings
                    if (venue_data.business_status != 'CLOSED_PERMANENTLY' and 
                        venue_data.business_status != 'CLOSED_TEMPORARILY' and
                        (not venue_data.rating or venue_data.rating >= self.rating_threshold)):
                        venues_data.append(venue_data)

                except Exception as e:
                    logger.warning(f"Error fetching details for place {place_id}: {e}")
                    continue

            # Sort by rating and limit results
            venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
            return venues_data[:self.max_venues_to_verify]

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

            # Get AI analysis
            response = await self.review_analysis_prompt.ainvoke({
                'restaurant_data': json.dumps(restaurant_data, indent=2)
            })

            # Parse AI response
            try:
                analysis_data = json.loads(response.content)
                analysis_results = analysis_data.get('analysis', [])
            except json.JSONDecodeError:
                logger.error("Failed to parse AI analysis response")
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

            logger.info(f"AI selected {len(selected_venues)} venues based on review quality")
            return selected_venues

        except Exception as e:
            logger.error(f"❌ Error in venue analysis: {e}")
            # Fallback: return top rated venues
            venues_data.sort(key=lambda x: x.rating or 0, reverse=True)
            return venues_data[:3]

    async def _tavily_search_venues(
        self, 
        venues: List[EnhancedVenueData], 
        cancel_check_fn=None
    ):
        """
        Step 4: Tavily search for each venue
        """
        if not self.tavily_api_key:
            logger.warning("Skipping Tavily search - no API key")
            return

        for venue in venues:
            if cancel_check_fn and cancel_check_fn():
                break

            try:
                # Extract city from address for search
                city = self._extract_city_from_address(venue.address)
                search_query = f"{venue.name} {city} restaurant"

                # Perform Tavily search
                search_results = await self._perform_tavily_search(search_query)
                venue.media_search_results = search_results

                logger.debug(f"Tavily search for {venue.name}: {len(search_results)} results")

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error in Tavily search for {venue.name}: {e}")
                venue.media_search_results = []

    async def _perform_tavily_search(self, query: str) -> List[Dict]:
        """Perform Tavily API search"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_images": False,
                    "include_raw_content": False,
                    "max_results": 10,
                    "exclude_domains": [
                        "tripadvisor.com", "yelp.com", "facebook.com", 
                        "instagram.com", "wanderlog.com", "youtube.com"
                    ]
                }

                async with session.post(
                    "https://api.tavily.com/search", 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
                    else:
                        logger.warning(f"Tavily search failed: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    async def _analyze_media_sources(
        self, 
        venues: List[EnhancedVenueData], 
        cancel_check_fn=None
    ):
        """
        Step 5: AI analysis of media sources to identify professional content
        """
        for venue in venues:
            if cancel_check_fn and cancel_check_fn():
                break

            if not venue.media_search_results:
                continue

            try:
                # Get AI analysis of search results
                response = await self.media_analysis_prompt.ainvoke({
                    'search_results': json.dumps(venue.media_search_results, indent=2)
                })

                # Parse response
                try:
                    analysis_data = json.loads(response.content)
                    professional_sources = analysis_data.get('professional_sources', [])
                    venue.professional_sources = professional_sources

                    if professional_sources:
                        venue.has_professional_coverage = True
                        logger.debug(f"{venue.name}: Found {len(professional_sources)} professional sources")

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse media analysis for {venue.name}")
                    venue.professional_sources = []

            except Exception as e:
                logger.warning(f"Error analyzing media sources for {venue.name}: {e}")
                venue.professional_sources = []

    async def _scrape_professional_content(
        self, 
        venues: List[EnhancedVenueData], 
        cancel_check_fn=None
    ):
        """
        Step 6: Smart scraping of professional sources
        TODO: Integrate with smart scraper API (Thunderbit, Browse AI, etc.)
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
            'has_google_maps_api': True,
            'has_tavily_api': self.tavily_api_key is not None,
            'rating_threshold': self.rating_threshold,
            'max_venues': self.max_venues_to_verify,
            'ai_model': self.config.OPENAI_MODEL
        }