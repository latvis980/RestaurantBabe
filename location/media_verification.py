# location/media_verification.py
"""
Media Verification Agent - STEPS 4 & 5 - UPDATED

UPDATED: Implements proper Tavily search with AI filtering copied from search_agent.py
Constructs searches like: venue name + city + media
Filters by reputable sources using AI evaluation
Extracts descriptions from trusted sources
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from location.google_maps_search import VenueResult
from location.location_utils import LocationUtils

# Import AI components for source quality evaluation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class VerifiedVenue:
    """Structure for media-verified venue results"""
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float
    rating: Optional[float] = None
    place_id: Optional[str] = None
    description: str = ""
    media_sources: Optional[List[str]] = None
    google_maps_url: str = ""

    def __post_init__(self):
        if self.media_sources is None:
            self.media_sources = []

class MediaVerificationAgent:
    """
    STEPS 4-5: Media verification and description extraction

    UPDATED: Uses Tavily search with AI filtering for source quality
    """

    def __init__(self, config):
        self.config = config

        # Initialize Tavily API for media verification
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY', None)
        if not self.tavily_api_key:
            logger.warning("⚠️ TAVILY_API_KEY not found - media verification will be limited")

        # Initialize AI for source quality evaluation (COPIED FROM search_agent.py)
        self.eval_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Source quality evaluation prompt (COPIED FROM search_agent.py)
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze a web page's content and rate it using the criteria below.

        PRIORITIZE THESE SOURCES (score 0.8-1.0):
        - Established food and travel publications (e.g., Conde Nast Traveler, Forbes Travel, Food & Wine, Bon Appétit, etc.)
        - Local newspapers and magazines (like Expresso.pt, Le Monde, El Pais, Time Out)
        - Professional food critics and culinary experts
        - Reputable local food blogs (Katie Parla for Italy, 2Foodtrippers, David Leibovitz for Paris, etc.)
        - Local tourism boards and official regional and city guides
        - Restaurant guides and gastronomic awards (Michelin, The World's 50 Best, World of Mouth)

        VALID CONTENT (score 0.6-0.8):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris", "Best artisanal pizza in Rome", etc.)
        - Local expertise and knowledge
        - Detailed professional descriptions of restaurants/cuisine

        AVOID THESE SOURCES (score 0.0-0.5):
        - TripAdvisor and similar review aggregation sites
        - Social media content on Facebook and Instagram without professional curation
        - ANY Wanderlog content
        - Single restaurant reviews (with exception of professional media)
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites like Booking.com, Agoda, Expedia, Jalan, etc.
        - Websites of irrelevant businesses: real estate agencies, rental companies, tour booking sites, etc.
        - Video content (YouTube, TikTok, etc.)

        SCORING CRITERIA:
        - Multiple restaurants mentioned (essential, with the only exception of single restaurant reviews in professional media)
        - Professional curation or expertise evident
        - Local expertise and knowledge
        - Detailed professional descriptions of restaurants/cuisine

        FORMAT:
        Respond with a JSON object containing:
        {{
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "passed_filter": true/false,
          "reasoning": "brief explanation of your decision"
        }}
        """

        # Set up evaluation prompt
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("human", "URL: {{url}}\nTitle: {{title}}\nDescription: {{description}}\nContent Preview: {{content_preview}}")
        ])

        # Create evaluation chain
        self.eval_chain = self.eval_prompt | self.eval_model

        # Media verification settings
        self.min_media_mentions = getattr(config, 'MIN_MEDIA_MENTIONS_REQUIRED', 1)
        self.min_quality_score = 0.7  # High quality threshold for sources

        logger.info("✅ Media Verification Agent initialized with Tavily search and AI filtering")

    async def verify_venues(
        self,
        venues: List[VenueResult],
        query: str,
        coordinates: Optional[Tuple[float, float]] = None,  # ADDED: for distance calculation
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """
        STEPS 4-5: Verify venues in media and extract descriptions
        UPDATED: Added coordinates parameter for distance calculation in formatting

        Args:
            venues: List of VenueResult objects from Google Maps
            query: Original search query
            coordinates: Optional user coordinates for distance calculation
            cancel_check_fn: Function to check if operation should be cancelled

        Returns:
            List of verified venue dictionaries with media descriptions and distances
        """
        try:
            logger.info(f"🔍 Starting media verification for {len(venues)} venues")

            verified_venues = []

            for venue in venues:
                if cancel_check_fn and cancel_check_fn():
                    break

                # Step 4: Verify venue in media
                media_verification = await self._verify_venue_in_media(venue, query)

                if media_verification['is_verified']:
                    # Step 5: Extract description from trusted sources
                    description = await self._extract_venue_description(
                        venue, media_verification['sources']
                    )

                    # Create verified venue record with distance calculation
                    verified_venue = self._create_verified_venue(
                        venue, media_verification, description, coordinates
                    )
                    verified_venues.append(verified_venue)

            logger.info(f"✅ Media verification completed: {len(verified_venues)}/{len(venues)} venues verified")

            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error in venue verification: {e}")
            # Fallback: return venues without verification
            return self._convert_venues_to_dict(venues, coordinates)

    async def _verify_venue_in_media(
        self, 
        venue: VenueResult, 
        query: str
    ) -> Dict[str, Any]:
        """
        STEP 4: Verify venue in trusted media sources using Tavily search

        Constructs searches like: restaurant + city + media
        """
        try:
            # Extract city from venue address
            city = self._extract_city_from_address(venue.address)

            # Construct media search query (as specified)
            search_query = f"{venue.name} {city} media"

            logger.info(f"🔍 Media search: '{search_query}'")

            # Perform Tavily search
            search_results = await self._perform_tavily_search(search_query)

            # Filter by AI evaluation for source quality
            trusted_sources = await self._filter_by_ai_evaluation(search_results)

            verification_result = {
                'is_verified': len(trusted_sources) >= self.min_media_mentions,
                'sources': trusted_sources,
                'search_query': search_query,
                'total_results': len(search_results)
            }

            return verification_result

        except Exception as e:
            logger.error(f"❌ Error verifying {venue.name} in media: {e}")
            return {'is_verified': False, 'sources': [], 'search_query': '', 'total_results': 0}

    async def _perform_tavily_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform Tavily search for media coverage
        """
        try:
            if not self.tavily_api_key:
                logger.warning("⚠️ No Tavily API key - skipping search")
                return []

            import aiohttp

            # Tavily API endpoint
            url = "https://api.tavily.com/search"

            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": 10,
                "include_domains": [],
                "exclude_domains": [
                    "tripadvisor.com", 
                    "tripadvisor.pt", 
                    "yelp.com", 
                    "facebook.com", 
                    "instagram.com",
                    "wanderlog.com",
                    "youtube.com",
                    "tiktok.com"
                ]
            }

            timeout = aiohttp.ClientTimeout(total=30)  # Create ClientTimeout object
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:  # Remove timeout from here
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        logger.info(f"📊 Tavily search returned {len(results)} results")
                        return results
                    else:
                        logger.error(f"❌ Tavily API error: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"❌ Error in Tavily search: {e}")
            return []

    async def _filter_by_ai_evaluation(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter search results using AI evaluation for source quality
        COPIED LOGIC FROM search_agent.py
        """
        try:
            trusted_sources = []

            for result in search_results:
                url = result.get('url', '')
                title = result.get('title', '')
                content_preview = result.get('content', '')[:500]  # First 500 chars

                # Evaluate source quality using AI
                try:
                    evaluation_input = {
                        "url": url,
                        "title": title,
                        "description": result.get('snippet', ''),
                        "content_preview": content_preview
                    }

                    response = await self.eval_chain.ainvoke(evaluation_input)

                    # FIXED: Proper type handling for response content
                    evaluation_text = response.content
                    if not isinstance(evaluation_text, str):
                        # Handle case where content might not be a string
                        evaluation_text = str(evaluation_text)

                    # Parse AI evaluation
                    if "```json" in evaluation_text:
                        evaluation_text = evaluation_text.split("```json")[1].split("```")[0]
                    elif "```" in evaluation_text:
                        parts = evaluation_text.split("```")
                        if len(parts) >= 3:
                            evaluation_text = parts[1]

                    evaluation = json.loads(evaluation_text.strip())

                    quality_score = evaluation.get('content_quality', 0.0)
                    passed_filter = evaluation.get('passed_filter', False)

                    # Include only high-quality sources
                    if passed_filter and quality_score >= self.min_quality_score:
                        trusted_sources.append({
                            'url': url,
                            'domain': self._extract_domain(url),
                            'title': title,
                            'snippet': result.get('content', ''),
                            'quality_score': quality_score,
                            'evaluation': evaluation
                        })

                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Could not parse AI evaluation for {url}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error evaluating {url}: {e}")
                    continue

            logger.info(f"✅ AI filtering: {len(trusted_sources)}/{len(search_results)} sources passed quality check")
            return trusted_sources

        except Exception as e:
            logger.error(f"❌ Error in AI evaluation: {e}")
            return []

    async def _extract_venue_description(
        self, 
        venue: VenueResult, 
        trusted_sources: List[Dict[str, Any]]
    ) -> str:
        """
        STEP 5: Extract brief restaurant description from trusted media sources

        Gets descriptions from professional guides and media only.
        """
        try:
            if not trusted_sources:
                return f"Restaurant in {self._extract_city_from_address(venue.address)}"

            # Take the best description from highest quality source
            best_source = max(trusted_sources, key=lambda x: x.get('quality_score', 0))
            snippet = best_source.get('snippet', '')

            if snippet and len(snippet) > 20:
                # Clean and format the description (1-2 sentences)
                description = self._clean_description(snippet)
                return description

            # Fallback description with source attribution
            source_domain = best_source.get('domain', 'professional guides')
            city = self._extract_city_from_address(venue.address)
            return f"Restaurant recommended by {source_domain} in {city}"

        except Exception as e:
            logger.error(f"❌ Error extracting description for {venue.name}: {e}")
            return f"Restaurant in {self._extract_city_from_address(venue.address)}"

    def _clean_description(self, snippet: str) -> str:
        """
        Clean and format description from media snippet (1-2 sentences)
        """
        try:
            # Remove common snippet artifacts
            description = snippet.strip()

            # Remove date references and other clutter
            import re
            description = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', description)
            description = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b', '', description)

            # Limit to 1-2 sentences (up to 200 chars)
            if len(description) > 200:
                description = description[:200]
                # Find the last complete sentence
                last_period = description.rfind('.')
                if last_period > 100:
                    description = description[:last_period + 1]

            return description.strip()

        except Exception:
            return snippet[:200] if snippet else ""

    def _create_verified_venue(
        self, 
        venue: VenueResult, 
        media_verification: Dict[str, Any], 
        description: str,
        user_coordinates: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Create verified venue dictionary with all required fields
        UPDATED: Added distance calculation for formatting
        """
        sources = media_verification.get('sources', [])
        source_names = [source.get('domain', 'Unknown') for source in sources]

        # Calculate distance if user coordinates provided
        distance_km = venue.distance_km  # Use existing if available
        distance_text = LocationUtils.format_distance(distance_km) if distance_km else "Distance unknown"

        if user_coordinates and venue.latitude and venue.longitude:
            # Recalculate distance with user coordinates
            distance_km = LocationUtils.calculate_distance(
                user_coordinates, 
                (venue.latitude, venue.longitude)
            )
            distance_text = LocationUtils.format_distance(distance_km)

        return {
            'name': venue.name,
            'address': venue.address,
            'latitude': venue.latitude,
            'longitude': venue.longitude,
            'distance_km': distance_km,
            'distance_text': distance_text,
            'rating': venue.rating,
            'place_id': venue.place_id,
            'description': description,
            'sources': source_names,
            'media_verified': True,
            'media_sources_count': len(sources),
            'google_maps_url': venue.google_maps_url,
            'verification_query': media_verification.get('search_query', '')
        }

    def _convert_venues_to_dict(
        self, 
        venues: List[VenueResult], 
        user_coordinates: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert VenueResult objects to dictionaries (fallback when verification fails)
        UPDATED: Added distance calculation
        """
        result_venues = []

        for venue in venues:
            # Calculate distance if coordinates provided
            distance_km = venue.distance_km
            distance_text = LocationUtils.format_distance(distance_km) if distance_km else "Distance unknown"

            if user_coordinates and venue.latitude and venue.longitude:
                distance_km = LocationUtils.calculate_distance(
                    user_coordinates, 
                    (venue.latitude, venue.longitude)
                )
                distance_text = LocationUtils.format_distance(distance_km)

            result_venues.append({
                'name': venue.name,
                'address': venue.address,
                'latitude': venue.latitude,
                'longitude': venue.longitude,
                'distance_km': distance_km,
                'distance_text': distance_text,
                'rating': venue.rating,
                'place_id': venue.place_id,
                'description': f"Restaurant in {self._extract_city_from_address(venue.address)}",
                'sources': [],
                'media_verified': False,
                'google_maps_url': venue.google_maps_url
            })

        return result_venues


    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from venue address"""
        try:
            if not address:
                return "Unknown"

            # Simple extraction - split by comma and take appropriate part
            parts = [part.strip() for part in address.split(',')]

            if len(parts) >= 3:
                # Usually: street, city, state/country
                return parts[1]
            elif len(parts) == 2:
                return parts[0]
            else:
                return parts[0] if parts else "Unknown"

        except Exception:
            return "Unknown"

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain

        except Exception:
            return url.lower()

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get statistics about media verification process"""
        return {
            'has_tavily_api': self.tavily_api_key is not None,
            'min_media_mentions': self.min_media_mentions,
            'min_quality_score': self.min_quality_score,
            'evaluation_model': self.config.OPENAI_MODEL
        }