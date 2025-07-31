# agents/source_mapping_agent.py
"""
AI Source Mapping Agent with Web Search Verification

Intelligently determines which reputable sources to search for each venue,
performs web searches, and uses AI to verify mentions in professional publications.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class SourceMappingAgent:
    """
    Uses AI to determine relevant authoritative sources and verify venue mentions
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for source mapping
        self.ai = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )

        # Initialize search evaluation model (cost-optimized)
        self.search_ai = ChatOpenAI(
            model=config.SEARCH_EVALUATION_MODEL or "gpt-4o-mini",
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )

        # Source mapping prompt
        self.mapping_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_source_mapping_prompt()),
            ("human", "Venue: {{venue_name}}\nType: {{venue_type}}\nLocation: {{location}}\nDescription: {{description}}")
        ])

        # Web content evaluation prompt
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_evaluation_prompt()),
            ("human", "VENUE: {{venue_name}}\nSEARCH QUERY: {{search_query}}\nWEB CONTENT:\n{{web_content}}")
        ])

        # Create chains
        self.mapping_chain = self.mapping_prompt | self.ai
        self.evaluation_chain = self.evaluation_prompt | self.search_ai

        # Initialize web search (assuming BraveSearchAgent exists)
        try:
            from agents.search_agent import BraveSearchAgent
            self.search_agent = BraveSearchAgent(config)
            logger.info("✅ Web search integration enabled")
        except ImportError:
            logger.warning("⚠️ Web search agent not available - will use fallback")
            self.search_agent = None

        logger.info("✅ AI Source Mapping Agent with verification initialized")

    def _get_source_mapping_prompt(self) -> str:
        """System prompt for source mapping"""
        return """
You are an expert food and beverage researcher who knows which authoritative sources are most credible for different types of venues.

Your job is to suggest 1-2 authoritative sources that would most likely have reviewed or mentioned a specific venue.

VENUE ANALYSIS:
- Analyze the venue name, type, and location
- Consider the venue's likely market positioning (casual, upscale, specialty, etc.)
- Think about which publications would cover this type of establishment

SOURCE CATEGORIES & EXAMPLES:
- Fine Dining: Michelin Guide, World's 50 Best, James Beard, Zagat, local food critics
- Natural Wine: Raisin, The Wine List, Punch Magazine, Natural Wine Company, Glou Glou
- Specialty Coffee: Sprudge, Perfect Daily Grind, Coffee Review, Standart Magazine
- Cocktails/Bars: World's 50 Best Bars, Punch Magazine, Difford's Guide, Imbibe Magazine
- Local/Casual: TimeOut, Eater, Conde Nast Traveler, local food bloggers, city magazines
- Bakeries: Local food media, specialty baking publications
- International: Local Michelin guides, regional food publications

LOCATION CONSIDERATIONS:
- Major cities (NYC, Paris, London): Likely covered by international publications
- Smaller cities: Focus on local food media and regional publications
- Food capitals: Higher chance of specialty publication coverage

RESPONSE FORMAT (JSON only):
{{
    "primary_source": "most likely authoritative source",
    "secondary_source": "backup authoritative source",
    "reasoning": "why these sources were chosen",
    "search_terms": ["term1", "term2"],
    "confidence": 0.1-1.0
}}

GUIDELINES:
- Be realistic about venue prominence vs source likelihood
- Consider local vs international publications based on location
- Higher confidence for well-known venues in major cities
- Lower confidence for small/local venues
- Always suggest search terms that combine venue name + source
"""

    def _get_evaluation_prompt(self) -> str:
        """System prompt for evaluating web search results"""
        return """
You are an expert at analyzing web content to determine if restaurants/venues are mentioned in reputable sources.

Your task is to evaluate whether the web content contains meaningful coverage of the venue from professional sources.

POSITIVE INDICATORS (venue is mentioned professionally):
- Articles in established food publications, local newspapers, magazines
- Reviews by professional food critics or journalists
- Mentions in curated restaurant guides or lists
- Coverage in travel publications with editorial standards
- Local media restaurant coverage with professional writing
- Food blogger reviews with clear expertise and following

NEGATIVE INDICATORS (not professional coverage):
- Social media posts (Instagram, Facebook, TikTok)
- Booking platforms (OpenTable, Resy, TheFork)
- Review aggregators (TripAdvisor, Yelp, Google Reviews)
- Delivery platforms (Uber Eats, DoorDash, Grubhub)
- Generic directory listings
- User-generated content without editorial oversight
- The venue's own website or promotional content

ANALYSIS CRITERIA:
1. Source credibility - Is this from a reputable publication?
2. Editorial oversight - Does this have professional editorial standards?
3. Content quality - Is this substantive coverage vs just a listing?
4. Context relevance - Is the venue meaningfully discussed?

RESPONSE FORMAT (JSON only):
{{
    "has_professional_mention": true/false,
    "source_type": "description of the source",
    "mention_quality": "brief description of how venue is mentioned",
    "confidence": 0.1-1.0,
    "reasoning": "explanation of decision"
}}

Be strict - only return true for genuine professional coverage, not just any web mention.
"""

    async def map_venue_sources(self, venue) -> Dict[str, Any]:
        """
        Main method: Map sources for a venue and verify with web search

        Args:
            venue: VenueResult object with name, types, address etc.

        Returns:
            Dict with verification results and found sources
        """
        try:
            logger.debug(f"🔍 Starting source verification for: {venue.name}")

            # Extract venue information
            venue_name = venue.name
            venue_type = self._determine_venue_type(venue)
            location = self._extract_location(venue)

            # Step 1: AI source mapping
            source_mapping = await self._get_source_mapping(venue_name, venue_type, location)

            if not source_mapping:
                return self._create_failed_result(venue_name, "Source mapping failed")

            # Step 2: Generate search queries
            search_queries = self._generate_search_queries(venue_name, source_mapping)

            # Step 3: Perform web searches and AI evaluation
            verification_result = await self._verify_with_web_search(venue_name, search_queries)

            # Step 4: Combine results
            result = {
                'venue_name': venue_name,
                'venue_type': venue_type,
                'location': location,
                'source_mapping': source_mapping,
                'verification': verification_result,
                'sources': verification_result.get('professional_sources', []),
                'verified': verification_result.get('has_professional_mentions', False)
            }

            logger.debug(f"✅ Verification complete for {venue_name}: {result['verified']}")
            return result

        except Exception as e:
            logger.error(f"❌ Error in venue source mapping: {e}")
            return self._create_failed_result(venue.name, str(e))

    async def _get_source_mapping(self, venue_name: str, venue_type: str, location: str) -> Optional[Dict[str, Any]]:
        """Use AI to map appropriate sources for this venue"""
        try:
            description = f"{venue_type} in {location}"

            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.mapping_chain.invoke({
                    "venue_name": venue_name,
                    "venue_type": venue_type,
                    "location": location,
                    "description": description
                })
            )

            content = response.content.strip()

            # Clean JSON from markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            source_mapping = json.loads(content)

            # Add venue info
            source_mapping.update({
                "venue_name": venue_name,
                "venue_type": venue_type,
                "location": location
            })

            return source_mapping

        except Exception as e:
            logger.error(f"AI source mapping failed: {e}")
            return self._fallback_source_mapping(venue_name, venue_type, location)

    def _generate_search_queries(self, venue_name: str, source_mapping: Dict[str, Any]) -> List[str]:
        """Generate search queries using the mapped sources"""
        queries = []

        # Use AI-suggested search terms if available
        suggested_terms = source_mapping.get('search_terms', [])
        if suggested_terms:
            queries.extend(suggested_terms)

        # Add source-based queries
        primary_source = source_mapping.get('primary_source', '')
        secondary_source = source_mapping.get('secondary_source', '')

        if primary_source:
            queries.append(f"{venue_name} {primary_source}")
        if secondary_source:
            queries.append(f"{venue_name} {secondary_source}")

        # Add general professional coverage queries
        queries.extend([
            f"{venue_name} review",
            f"{venue_name} food critic",
            f"{venue_name} restaurant guide"
        ])

        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen:
                unique_queries.append(query)
                seen.add(query)

        return unique_queries[:5]  # Limit to 5 searches per venue

    async def _verify_with_web_search(self, venue_name: str, search_queries: List[str]) -> Dict[str, Any]:
        """Perform web searches and AI evaluation of results"""
        try:
            professional_sources = []
            has_professional_mentions = False
            total_searches = len(search_queries)

            logger.debug(f"🔍 Performing {total_searches} verification searches for {venue_name}")

            for i, query in enumerate(search_queries):
                try:
                    logger.debug(f"📊 Search {i+1}/{total_searches}: '{query}'")

                    # Perform web search
                    search_results = await self._perform_web_search(query)

                    if not search_results:
                        continue

                    # AI evaluation of results
                    evaluation = await self._evaluate_search_results(venue_name, query, search_results)

                    if evaluation.get('has_professional_mention', False):
                        has_professional_mentions = True
                        professional_sources.append({
                            'search_query': query,
                            'source_type': evaluation.get('source_type', 'Unknown'),
                            'mention_quality': evaluation.get('mention_quality', ''),
                            'confidence': evaluation.get('confidence', 0.5)
                        })

                        logger.debug(f"✅ Found professional mention via '{query}'")

                except Exception as e:
                    logger.error(f"Error in search verification for '{query}': {e}")
                    continue

            return {
                'has_professional_mentions': has_professional_mentions,
                'professional_sources': professional_sources,
                'searches_performed': total_searches,
                'sources_found': len(professional_sources)
            }

        except Exception as e:
            logger.error(f"Web search verification failed: {e}")
            return {
                'has_professional_mentions': False,
                'professional_sources': [],
                'searches_performed': 0,
                'sources_found': 0,
                'error': str(e)
            }

    async def _perform_web_search(self, query: str) -> Optional[str]:
        """Perform web search and return combined content"""
        try:
            if not self.search_agent:
                return None

            # Use existing search agent
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.search_agent.search_web(query)
            )

            if not search_results:
                return None

            # Combine snippets from search results
            combined_content = ""
            for result in search_results[:3]:  # Top 3 results
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                url = result.get('url', '')

                combined_content += f"Title: {title}\n"
                combined_content += f"URL: {url}\n" 
                combined_content += f"Content: {snippet}\n\n"

            return combined_content.strip()

        except Exception as e:
            logger.error(f"Web search failed for '{query}': {e}")
            return None

    async def _evaluate_search_results(self, venue_name: str, query: str, content: str) -> Dict[str, Any]:
        """Use AI to evaluate if search results contain professional mentions"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.evaluation_chain.invoke({
                    "venue_name": venue_name,
                    "search_query": query,
                    "web_content": content
                })
            )

            content_text = response.content.strip()

            # Clean JSON from markdown
            if "```json" in content_text:
                content_text = content_text.split("```json")[1].split("```")[0].strip()
            elif "```" in content_text:
                content_text = content_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content_text)
            return evaluation

        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return {
                'has_professional_mention': False,
                'source_type': 'Unknown',
                'mention_quality': '',
                'confidence': 0.0,
                'reasoning': f'Evaluation error: {str(e)}'
            }

    def _determine_venue_type(self, venue) -> str:
        """Extract venue type from VenueResult"""
        if hasattr(venue, 'types') and venue.types:
            # Google Places types to readable types
            type_mapping = {
                'restaurant': 'restaurant',
                'bar': 'bar',
                'cafe': 'cafe',
                'bakery': 'bakery',
                'meal_takeaway': 'takeaway',
                'night_club': 'nightclub'
            }

            for google_type in venue.types:
                if google_type in type_mapping:
                    return type_mapping[google_type]

        return 'restaurant'  # Default

    def _extract_location(self, venue) -> str:
        """Extract location from VenueResult"""
        if hasattr(venue, 'address') and venue.address:
            # Extract city/area from address
            address_parts = venue.address.split(',')
            if len(address_parts) >= 2:
                return address_parts[-2].strip()  # Usually city
            return address_parts[0].strip()
        return "Unknown"

    def _fallback_source_mapping(self, venue_name: str, venue_type: str, location: str) -> Dict[str, Any]:
        """Fallback source mapping using simple heuristics"""
        venue_type_lower = venue_type.lower()

        if any(term in venue_type_lower for term in ["fine", "upscale"]):
            primary_source = "michelin guide"
            secondary_source = "timeout"
        elif "wine" in venue_type_lower:
            primary_source = "raisin"
            secondary_source = "punch magazine"
        elif "coffee" in venue_type_lower:
            primary_source = "sprudge" 
            secondary_source = "timeout"
        elif "bar" in venue_type_lower:
            primary_source = "punch magazine"
            secondary_source = "worlds 50 best bars"
        else:
            primary_source = "timeout"
            secondary_source = "eater"

        return {
            "venue_name": venue_name,
            "venue_type": venue_type,
            "location": location,
            "primary_source": primary_source,
            "secondary_source": secondary_source,
            "reasoning": f"Fallback mapping for {venue_type} in {location}",
            "search_terms": [
                f"{venue_name} {primary_source}",
                f"{venue_name} {secondary_source}"
            ],
            "confidence": 0.5
        }

    def _create_failed_result(self, venue_name: str, error: str) -> Dict[str, Any]:
        """Create a failed verification result"""
        return {
            'venue_name': venue_name,
            'verification': {
                'has_professional_mentions': False,
                'professional_sources': [],
                'error': error
            },
            'sources': [],
            'verified': False
        }

    # Batch processing methods for efficiency
    async def batch_verify_venues(self, venues: List) -> List[Dict[str, Any]]:
        """Verify multiple venues in parallel"""
        tasks = [self.map_venue_sources(venue) for venue in venues]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        verified_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Venue verification failed: {result}")
                continue
            verified_results.append(result)

        return verified_results