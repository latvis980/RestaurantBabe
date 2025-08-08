# agents/media_search_agent.py
"""
Media Search Agent

Searches for professional media coverage of venues using a separate Brave API key.
Focuses on reputable sources and filters out user reviews/directories.
"""

import logging
import asyncio
import requests
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class MediaSearchAgent:
    """
    Searches for professional media coverage of venues
    """

    def __init__(self, config):
        self.config = config

        # Separate API key for media searches
        self.media_api_key = getattr(config, 'BRAVE_MEDIA_API_KEY', None) or config.BRAVE_API_KEY
        self.brave_base_url = "https://api.search.brave.com/res/v1/web/search"

        # AI for evaluating search results
        self.ai = ChatOpenAI(
            model=config.SEARCH_EVALUATION_MODEL or "gpt-4o-mini",
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )

        # Media evaluation prompt
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_evaluation_prompt()),
            ("human", "VENUE: {{venue_name}}\nCITY: {{city}}\nSEARCH RESULTS:\n{{search_results}}")
        ])

        self.evaluation_chain = self.evaluation_prompt | self.ai

        logger.info("✅ Media Search Agent initialized")

    def _get_evaluation_prompt(self) -> str:
        """Prompt for evaluating media coverage"""
        return """
You analyze search results to find PROFESSIONAL media coverage of restaurants/venues.

REPUTABLE SOURCES (ACCEPT):
- Food magazines: Bon Appétit, Food & Wine, Saveur
- Local newspapers and magazines with food sections
- Professional food critics and journalists
- Established food blogs with editorial standards
- Travel publications: Condé Nast Traveler, Travel + Leisure
- City magazines: TimeOut, Eater, local publications
- Restaurant guides: Michelin, Zagat, local guides

EXCLUDE (REJECT):
- TripAdvisor, Yelp, Google Reviews
- OpenTable, Resy, booking platforms
- Social media posts (Instagram, Facebook, TikTok)
- Delivery apps (Uber Eats, DoorDash)
- User review sites
- Directory listings
- The venue's own website

ANALYSIS:
Examine each search result. Extract only mentions from reputable sources that provide meaningful coverage (reviews, recommendations, features).

RESPONSE FORMAT (JSON):
{{
    "has_media_coverage": true/false,
    "reputable_sources": [
        {{
            "source_name": "publication name",
            "article_title": "article title",
            "mention_type": "review/feature/recommendation/list",
            "credibility": "high/medium",
            "excerpt": "brief quote about the venue"
        }}
    ],
    "total_sources_found": number,
    "confidence": 0.1-1.0,
    "reasoning": "brief explanation"
}}

Be strict - only include genuine professional coverage.
"""

    async def search_venue_media_coverage(
        self, 
        venue_name: str, 
        city: str,
        venue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for professional media coverage of a single venue

        Args:
            venue_name: Name of the venue
            city: City where venue is located
            venue_type: Optional type hint (restaurant, bar, etc.)

        Returns:
            Dict with media coverage analysis
        """
        try:
            # Form search query: venue + city + media keywords
            query = f'"{venue_name}" {city} review restaurant'

            logger.debug(f"🔍 Searching media coverage for: {venue_name} in {city}")

            # Perform web search
            search_results = await self._perform_media_search(query)

            if not search_results:
                return self._create_no_coverage_result(venue_name)

            # AI evaluation of results
            evaluation = await self._evaluate_media_results(venue_name, city, search_results)

            return {
                'venue_name': venue_name,
                'city': city,
                'search_query': query,
                'media_coverage': evaluation,
                'has_coverage': evaluation.get('has_media_coverage', False)
            }

        except Exception as e:
            logger.error(f"❌ Error searching media coverage for {venue_name}: {e}")
            return self._create_error_result(venue_name, str(e))

    async def batch_search_venues(self, venues: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Search media coverage for multiple venues in parallel

        Args:
            venues: List of dicts with keys: name, city, type (optional)

        Returns:
            List of media coverage results
        """
        try:
            logger.info(f"🔍 Starting batch media search for {len(venues)} venues")

            # Create tasks for parallel execution
            tasks = []
            for venue in venues:
                name = venue.get('name', '')
                city = venue.get('city', '')
                venue_type = venue.get('type')

                if name and city:
                    task = self.search_venue_media_coverage(name, city, venue_type)
                    tasks.append(task)

            # Execute all searches in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    venue_name = venues[i].get('name', 'Unknown')
                    logger.error(f"❌ Media search failed for {venue_name}: {result}")
                    valid_results.append(self._create_error_result(venue_name, str(result)))
                else:
                    valid_results.append(result)

            # Log summary
            covered_venues = sum(1 for r in valid_results if r.get('has_coverage', False))
            logger.info(f"✅ Media search complete: {covered_venues}/{len(valid_results)} venues have coverage")

            return valid_results

        except Exception as e:
            logger.error(f"❌ Error in batch media search: {e}")
            return []

    async def _perform_media_search(self, query: str) -> Optional[str]:
        """
        Perform web search using Brave API (media-specific key)
        """
        try:
            if not self.media_api_key:
                logger.warning("No media API key configured")
                return None

            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.media_api_key
            }

            params = {
                "q": query,
                "count": 10,  # More results for better coverage detection
                "freshness": "6months",  # Recent coverage
                "text_decorations": False,
                "search_lang": "en"
            }

            # Execute request in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.brave_base_url, headers=headers, params=params, timeout=30)
            )

            if response.status_code != 200:
                logger.error(f"Brave Media API error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            results = data.get('web', {}).get('results', [])

            if not results:
                return None

            # Format results for AI analysis
            formatted_results = ""
            for i, result in enumerate(results[:10]):
                title = result.get('title', '')
                url = result.get('url', '')
                description = result.get('description', '')

                formatted_results += f"{i+1}. Title: {title}\n"
                formatted_results += f"   URL: {url}\n"
                formatted_results += f"   Description: {description}\n\n"

            return formatted_results.strip()

        except Exception as e:
            logger.error(f"❌ Media search request failed for '{query}': {e}")
            return None

    async def _evaluate_media_results(
        self, 
        venue_name: str, 
        city: str, 
        search_results: str
    ) -> Dict[str, Any]:
        """
        Use AI to evaluate search results for reputable media coverage
        """
        try:
            # Execute AI evaluation in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.evaluation_chain.invoke({
                    "venue_name": venue_name,
                    "city": city,
                    "search_results": search_results
                })
            )

            content = response.content.strip()

            # Clean JSON from markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse AI response
            import json
            evaluation = json.loads(content)

            logger.debug(f"📊 Media evaluation for {venue_name}: {evaluation.get('has_media_coverage', False)}")
            return evaluation

        except Exception as e:
            logger.error(f"❌ AI evaluation failed for {venue_name}: {e}")
            return {
                "has_media_coverage": False,
                "reputable_sources": [],
                "total_sources_found": 0,
                "confidence": 0.0,
                "reasoning": f"Evaluation error: {str(e)}"
            }

    def _create_no_coverage_result(self, venue_name: str) -> Dict[str, Any]:
        """Create result for venue with no search results"""
        return {
            'venue_name': venue_name,
            'media_coverage': {
                "has_media_coverage": False,
                "reputable_sources": [],
                "total_sources_found": 0,
                "confidence": 1.0,
                "reasoning": "No search results found"
            },
            'has_coverage': False
        }

    def _create_error_result(self, venue_name: str, error: str) -> Dict[str, Any]:
        """Create result for venue with search error"""
        return {
            'venue_name': venue_name,
            'media_coverage': {
                "has_media_coverage": False,
                "reputable_sources": [],
                "total_sources_found": 0,
                "confidence": 0.0,
                "reasoning": f"Search error: {error}"
            },
            'has_coverage': False,
            'error': error
        }

    def get_coverage_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for media coverage results
        """
        total_venues = len(results)
        covered_venues = sum(1 for r in results if r.get('has_coverage', False))

        high_confidence = sum(1 for r in results 
                            if r.get('media_coverage', {}).get('confidence', 0) > 0.7)

        total_sources = sum(r.get('media_coverage', {}).get('total_sources_found', 0) 
                           for r in results)

        return {
            'total_venues_searched': total_venues,
            'venues_with_coverage': covered_venues,
            'coverage_percentage': round((covered_venues / total_venues * 100), 1) if total_venues > 0 else 0,
            'high_confidence_matches': high_confidence,
            'total_reputable_sources_found': total_sources
        }