# location/location_media_verification.py
"""
Location Media Verification Agent

Handles media search and verification functionality including:
- Tavily API searches for professional restaurant coverage
- AI analysis of media sources for credibility and relevance
- Professional content identification and preparation for scraping
- Media-based venue quality assessment and selection

This agent works with venue data from the map search agent to enhance
venue descriptions with professional media coverage.
"""

import logging
import asyncio
import json
import aiohttp
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import openai

logger = logging.getLogger(__name__)

@dataclass
class MediaVerificationResult:
    """Results from media verification process"""
    venue_id: str
    venue_name: str

    # Media search results
    media_search_results: List[Dict] = field(default_factory=list)
    professional_sources: List[Dict] = field(default_factory=list)
    scraped_content: List[Dict] = field(default_factory=list)

    # Analysis results
    has_professional_coverage: bool = False
    media_coverage_score: float = 0.0
    credibility_assessment: Dict = field(default_factory=dict)

    # Combined data for text generation
    combined_media_data: Dict = field(default_factory=dict)

class LocationMediaVerificationAgent:
    """
    Media verification agent for restaurant venues

    Core responsibilities:
    1. Search for professional media coverage using Tavily API
    2. AI analysis of media sources for credibility and relevance
    3. Identify high-quality professional sources worth scraping
    4. Prepare combined media data for description generation
    5. Quality scoring based on media coverage
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.max_venues_to_verify = getattr(config, 'MAX_VENUES_TO_VERIFY', 5)
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.media_search_timeout = getattr(config, 'MEDIA_SEARCH_TIMEOUT', 30.0)
        self.tavily_max_results = getattr(config, 'TAVILY_SEARCH_MAX_RESULTS', 10)
        self.max_professional_sources = getattr(config, 'MAX_PROFESSIONAL_SOURCES', 3)
        self.professional_min_score = getattr(config, 'PROFESSIONAL_SOURCE_MIN_SCORE', 7.0)

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        # Initialize Tavily API
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY')
        if not self.tavily_api_key:
            logger.warning("Tavily API key not found - media searches will be disabled")

        logger.info("Location Media Verification Agent initialized")

    async def verify_venues_media_coverage(
        self,
        venues: List[Any],  # VenueSearchResult or compatible objects
        query: str,
        cancel_check_fn=None
    ) -> List[MediaVerificationResult]:
        """
        Main method: Verify media coverage for a list of venues

        Args:
            venues: List of venue objects (from map search)
            query: Original user query for context
            cancel_check_fn: Optional cancellation check function

        Returns:
            List of MediaVerificationResult objects
        """
        try:
            logger.info(f"Starting media verification for {len(venues)} venues")

            if not self.tavily_api_key:
                logger.warning("Tavily API not available - returning venues without media verification")
                return self._create_basic_results(venues)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 1: Perform media searches
            logger.info("Step 1: Performing Tavily media searches")
            media_results = await self._search_media_coverage(venues, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: AI analysis of media sources
            logger.info("Step 2: AI analysis of media sources")
            await self._analyze_media_sources(media_results, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 3: Quality assessment and source selection
            logger.info("Step 3: Quality assessment and professional source selection")
            await self._assess_media_quality(media_results, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 4: Prepare content for scraping (placeholder for now)
            logger.info("Step 4: Preparing professional content for scraping")
            await self._prepare_scraping_targets(media_results, cancel_check_fn)

            # Step 5: Combine all media data
            self._prepare_combined_media_data(media_results)

            logger.info(f"Media verification completed for {len(media_results)} venues")
            return media_results

        except Exception as e:
            logger.error(f"Error in media verification: {e}")
            return self._create_basic_results(venues)

    async def _search_media_coverage(
        self,
        venues: List[Any],
        cancel_check_fn=None
    ) -> List[MediaVerificationResult]:
        """Step 1: Search for media coverage using Tavily API"""
        media_results = []

        try:
            async with aiohttp.ClientSession() as session:
                for venue in venues:
                    if cancel_check_fn and cancel_check_fn():
                        break

                    # Create media verification result
                    venue_id = getattr(venue, 'place_id', str(venue.name))
                    venue_name = getattr(venue, 'name', 'Unknown')
                    venue_address = getattr(venue, 'address', '')

                    media_result = MediaVerificationResult(
                        venue_id=venue_id,
                        venue_name=venue_name
                    )

                    # Extract city from address for better search context
                    city = self._extract_city_from_address(venue_address)

                    # Create targeted search queries
                    search_queries = self._create_media_search_queries(venue_name, city)

                    # Execute searches
                    all_search_results = []
                    for query in search_queries:
                        try:
                            search_results = await self._execute_tavily_search(session, query)
                            all_search_results.extend(search_results)
                        except Exception as e:
                            logger.debug(f"Tavily search error for {venue_name} with query '{query}': {e}")
                            continue

                    media_result.media_search_results = all_search_results
                    media_results.append(media_result)

                    logger.debug(f"{venue_name}: Found {len(all_search_results)} media search results")

            return media_results

        except Exception as e:
            logger.error(f"Error in media coverage search: {e}")
            return []

    def _create_media_search_queries(self, venue_name: str, city: str) -> List[str]:
        """Create targeted search queries for media coverage"""
        queries = [
            f'"{venue_name}" {city} restaurant review',
            f'"{venue_name}" {city} food guide michelin',
            f'"{venue_name}" {city} best restaurants',
            f'"{venue_name}" restaurant critic review',
        ]

        # Remove queries with empty city
        if not city or city.lower() in ['unknown', 'n/a']:
            queries = [q.replace(f' {city}', '') for q in queries]

        return queries[:3]  # Limit to 3 queries per venue

    async def _execute_tavily_search(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Execute a single Tavily search with error handling"""
        try:
            tavily_payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": self.tavily_max_results
            }

            async with session.post(
                "https://api.tavily.com/search",
                json=tavily_payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('results', [])
                else:
                    logger.debug(f"Tavily search failed with status {response.status} for query: {query}")
                    return []

        except Exception as e:
            logger.debug(f"Tavily search exception for query '{query}': {e}")
            return []

    async def _analyze_media_sources(
        self,
        media_results: List[MediaVerificationResult],
        cancel_check_fn=None
    ):
        """Step 2: AI analysis of media sources to identify professional coverage"""
        for result in media_results:
            if cancel_check_fn and cancel_check_fn():
                break

            if not result.media_search_results:
                result.professional_sources = []
                result.has_professional_coverage = False
                result.credibility_assessment = {'analysis': 'No media results found'}
                continue

            try:
                logger.debug(f"Analyzing media sources for {result.venue_name}")

                # Create analysis prompt
                prompt = self._create_media_analysis_prompt(result.venue_name, result.media_search_results)

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model=self.openai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert media analyst specializing in identifying credible restaurant and food publications. Always respond with valid JSON only."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        temperature=0.2,
                        max_tokens=1500
                    ),
                    timeout=20
                )

                # Parse response safely
                analysis_data = self._safe_parse_media_analysis(response, result.venue_name)

                result.professional_sources = analysis_data.get('professional_sources', [])
                result.has_professional_coverage = len(result.professional_sources) > 0
                result.credibility_assessment = analysis_data.get('credibility_assessment', {})

                logger.debug(f"{result.venue_name}: Found {len(result.professional_sources)} professional sources")

            except asyncio.TimeoutError:
                logger.error(f"Media analysis timed out for {result.venue_name}")
                result.professional_sources = []
                result.has_professional_coverage = False
                result.credibility_assessment = {'analysis': 'Analysis timed out'}
            except Exception as e:
                logger.error(f"Error analyzing media sources for {result.venue_name}: {e}")
                result.professional_sources = []
                result.has_professional_coverage = False
                result.credibility_assessment = {'analysis': f'Analysis failed: {str(e)}'}

    def _create_media_analysis_prompt(self, venue_name: str, search_results: List[Dict]) -> str:
        """Create prompt for AI media analysis"""
        return f"""Analyze media search results to identify professional restaurant coverage.

VENUE: "{venue_name}"

IDENTIFY AS PROFESSIONAL:
- Food & travel magazines (Food & Wine, Conde Nast Traveler, etc.)
- Local newspapers and established magazines (Time Out, local papers)
- Professional food critics and established food blogs
- Official tourism guides and city guides
- Restaurant award guides (Michelin, James Beard, World's 50 Best, etc.)
- Established culinary websites (Eater, Serious Eats, etc.)

EXCLUDE:
- User review platforms (Yelp, TripAdvisor, OpenTable reviews)
- Social media posts and personal accounts
- Generic business directories
- Unestablished personal blogs
- Aggregate review sites

SEARCH RESULTS:
{json.dumps(search_results[:10], indent=2)}

Respond with valid JSON only:
{{
  "professional_sources": [
    {{
      "url": "source_url",
      "title": "article_title",
      "source_name": "publication_name",
      "source_type": "food_magazine|local_newspaper|tourism_guide|award_guide|culinary_website",
      "credibility_score": 8.5,
      "worth_scraping": true,
      "relevance_score": 9.0,
      "reason": "Brief explanation of why this is professional coverage"
    }}
  ],
  "credibility_assessment": {{
    "overall_media_score": 7.5,
    "has_major_publication": true,
    "has_award_mention": false,
    "coverage_quality": "good|excellent|limited",
    "summary": "Brief assessment of overall media coverage quality"
  }}
}}"""

    def _safe_parse_media_analysis(self, response, venue_name: str) -> Dict[str, Any]:
        """Safely parse AI media analysis response"""
        try:
            if not response or not response.choices:
                logger.error(f"Empty AI response for media analysis of {venue_name}")
                return self._get_empty_analysis()

            response_text = response.choices[0].message.content
            if not response_text:
                logger.error(f"Empty response content for {venue_name}")
                return self._get_empty_analysis()

            response_text = response_text.strip()

            # Clean JSON markers
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            # Find JSON in response
            if not response_text.startswith('{'):
                start_idx = response_text.find('{')
                if start_idx != -1:
                    response_text = response_text[start_idx:]

            parsed_data = json.loads(response_text)
            logger.debug(f"Successfully parsed media analysis for {venue_name}")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {venue_name}: {e}")
            logger.debug(f"Problematic response: {response_text[:500]}")
            return self._get_empty_analysis()
        except Exception as e:
            logger.error(f"Unexpected error parsing media analysis for {venue_name}: {e}")
            return self._get_empty_analysis()

    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Get default empty analysis structure"""
        return {
            'professional_sources': [],
            'credibility_assessment': {
                'overall_media_score': 0.0,
                'has_major_publication': False,
                'has_award_mention': False,
                'coverage_quality': 'limited',
                'summary': 'No professional media coverage found'
            }
        }

    async def _assess_media_quality(
        self,
        media_results: List[MediaVerificationResult],
        original_query: str,
        cancel_check_fn=None
    ):
        """Step 3: Quality assessment and professional source filtering"""
        for result in media_results:
            if cancel_check_fn and cancel_check_fn():
                break

            # Calculate media coverage score
            credibility = result.credibility_assessment
            base_score = credibility.get('overall_media_score', 0.0)

            # Boost score for major publications or awards
            if credibility.get('has_major_publication', False):
                base_score += 1.0
            if credibility.get('has_award_mention', False):
                base_score += 1.5

            # Filter professional sources by minimum score
            high_quality_sources = [
                source for source in result.professional_sources
                if source.get('credibility_score', 0) >= self.professional_min_score
            ]

            # Sort by credibility and relevance, limit count
            high_quality_sources.sort(
                key=lambda x: (x.get('credibility_score', 0) + x.get('relevance_score', 0)) / 2,
                reverse=True
            )

            result.professional_sources = high_quality_sources[:self.max_professional_sources]
            result.media_coverage_score = min(base_score, 10.0)  # Cap at 10
            result.has_professional_coverage = len(result.professional_sources) > 0

            logger.debug(f"{result.venue_name}: Media score {result.media_coverage_score:.1f}, {len(result.professional_sources)} quality sources")

    async def _prepare_scraping_targets(
        self,
        media_results: List[MediaVerificationResult],
        cancel_check_fn=None
    ):
        """Step 4: Prepare professional sources for content scraping"""
        for result in media_results:
            if cancel_check_fn and cancel_check_fn():
                break

            scraping_targets = []

            for source in result.professional_sources:
                if source.get('worth_scraping', False):
                    scraping_target = {
                        'url': source['url'],
                        'title': source['title'],
                        'source_name': source.get('source_name', 'Unknown'),
                        'source_type': source.get('source_type', 'unknown'),
                        'credibility_score': source.get('credibility_score', 0),
                        'content': f"[PLACEHOLDER] Professional coverage from {source.get('source_name', 'source')} - {source['title']}"
                        # TODO: Replace with actual smart scraper integration
                    }
                    scraping_targets.append(scraping_target)

            result.scraped_content = scraping_targets

            if scraping_targets:
                logger.debug(f"{result.venue_name}: Prepared {len(scraping_targets)} sources for scraping")

    def _prepare_combined_media_data(self, media_results: List[MediaVerificationResult]):
        """Step 5: Combine all media data for description generation"""
        for result in media_results:
            combined_data = {
                'has_media_coverage': result.has_professional_coverage,
                'media_coverage_score': result.media_coverage_score,
                'professional_sources': [
                    {
                        'title': source['title'],
                        'source_name': source.get('source_name', 'Unknown'),
                        'source_type': source.get('source_type', 'unknown'),
                        'credibility_score': source.get('credibility_score', 0)
                    }
                    for source in result.professional_sources
                ],
                'scraped_content': result.scraped_content,
                'media_mentions': [source.get('source_name', 'Unknown') for source in result.professional_sources],
                'credibility_indicators': {
                    'has_major_publication': result.credibility_assessment.get('has_major_publication', False),
                    'has_award_mention': result.credibility_assessment.get('has_award_mention', False),
                    'coverage_quality': result.credibility_assessment.get('coverage_quality', 'limited'),
                    'overall_assessment': result.credibility_assessment.get('summary', 'No assessment available')
                }
            }

            result.combined_media_data = combined_data

    def _create_basic_results(self, venues: List[Any]) -> List[MediaVerificationResult]:
        """Create basic results when media verification is disabled"""
        results = []
        for venue in venues:
            venue_id = getattr(venue, 'place_id', str(venue.name))
            venue_name = getattr(venue, 'name', 'Unknown')

            result = MediaVerificationResult(
                venue_id=venue_id,
                venue_name=venue_name,
                has_professional_coverage=False,
                media_coverage_score=0.0,
                combined_media_data={'has_media_coverage': False}
            )
            results.append(result)

        return results

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
        """Get statistics about media verification configuration"""
        return {
            'has_tavily_api': self.tavily_api_key is not None,
            'max_venues_to_verify': self.max_venues_to_verify,
            'tavily_max_results': self.tavily_max_results,
            'max_professional_sources': self.max_professional_sources,
            'professional_min_score': self.professional_min_score,
            'ai_model': self.openai_model,
            'media_search_timeout': self.media_search_timeout
        }