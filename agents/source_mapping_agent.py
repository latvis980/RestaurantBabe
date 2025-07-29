# agents/source_mapping_agent.py
"""
AI Source Mapping Agent

Intelligently determines which reputable sources to search for each venue
based on venue type, location, and characteristics. No hardcoding - pure AI decision making.
"""

import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class SourceMappingAgent:
    """
    Uses AI to determine the most relevant authoritative sources 
    for verifying restaurant/venue information
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for source mapping
        self.ai = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,  # Consistent but allowing some creativity
            api_key=config.OPENAI_API_KEY
        )

        # Source mapping prompt
        self.mapping_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_source_mapping_prompt()),
            ("human", "Venue: {venue_name}\nType: {venue_type}\nLocation: {location}\nDescription: {description}")
        ])

        # Create the mapping chain
        self.mapping_chain = self.mapping_prompt | self.ai

        # Source guidelines (not hardcoded rules, just suggestions for AI)
        self.source_guidelines = getattr(config, 'SOURCE_MAPPING_GUIDELINES', {
            "fine_dining": ["michelin", "worlds 50 best", "james beard", "local food critics"],
            "natural_wine": ["raisin", "wine list", "punch magazine", "natural wine company"],
            "coffee": ["sprudge", "perfect daily grind", "coffee review", "specialty coffee"],
            "cocktails": ["worlds 50 best bars", "punch magazine", "difford's guide", "imbibe"],
            "general": ["timeout", "eater", "conde nast traveler", "local food media"]
        })

        logger.info("✅ AI Source Mapping Agent initialized")

    def _get_source_mapping_prompt(self) -> str:
        """Get the system prompt for source mapping"""
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

EXAMPLES:

Input: "Le Bernardin, Fine Dining, New York"
{{
    "primary_source": "michelin guide",
    "secondary_source": "james beard",
    "reasoning": "Le Bernardin is a renowned fine dining establishment in NYC, definitely covered by Michelin and James Beard",
    "search_terms": ["Le Bernardin michelin", "Le Bernardin james beard"],
    "confidence": 0.95
}}

Input: "Blue Bottle Coffee, Coffee Shop, San Francisco"
{{
    "primary_source": "sprudge",
    "secondary_source": "perfect daily grind", 
    "reasoning": "Blue Bottle is a major specialty coffee chain, likely covered by specialty coffee publications",
    "search_terms": ["Blue Bottle Coffee sprudge", "Blue Bottle Coffee perfect daily grind"],
    "confidence": 0.85
}}

Input: "Local Pizza Joint, Pizza, Small Town"
{{
    "primary_source": "local food blog",
    "secondary_source": "yelp elite reviews",
    "reasoning": "Small town pizza place unlikely to be in major publications, focus on local sources",
    "search_terms": ["Local Pizza Joint food blog", "Local Pizza Joint review"],
    "confidence": 0.6
}}

GUIDELINES:
- Be realistic about venue prominence vs source likelihood
- Consider local vs international publications based on location
- Higher confidence for well-known venues in major cities
- Lower confidence for small/local venues
- Always suggest search terms that combine venue name + source
"""

    def map_sources_for_venue(
        self, 
        venue_name: str, 
        venue_type: str, 
        location: str,
        venue_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use AI to determine the best sources for a specific venue

        Args:
            venue_name: Name of the restaurant/venue
            venue_type: Type of venue (restaurant, bar, cafe, etc.)
            location: Location/city of the venue
            venue_description: Optional additional context

        Returns:
            Dict with source recommendations and search terms
        """
        try:
            logger.debug(f"🔍 Mapping sources for: {venue_name} ({venue_type}) in {location}")

            # Prepare description
            description = venue_description or f"{venue_type} in {location}"

            # Get AI recommendation
            response = self.mapping_chain.invoke({
                "venue_name": venue_name,
                "venue_type": venue_type, 
                "location": location,
                "description": description
            })

            # Parse AI response
            content = response.content.strip()

            # Clean up JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON response
            import json
            source_mapping = json.loads(content)

            # Add venue info for reference
            source_mapping["venue_name"] = venue_name
            source_mapping["venue_type"] = venue_type
            source_mapping["location"] = location

            logger.debug(f"✅ Source mapping: {source_mapping['primary_source']}, {source_mapping['secondary_source']}")
            return source_mapping

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse source mapping JSON: {e}")
            logger.error(f"Raw content: {content}")
            return self._fallback_source_mapping(venue_name, venue_type, location)

        except Exception as e:
            logger.error(f"Error in source mapping: {e}")
            return self._fallback_source_mapping(venue_name, venue_type, location)

    def _fallback_source_mapping(self, venue_name: str, venue_type: str, location: str) -> Dict[str, Any]:
        """
        Fallback source mapping using simple heuristics
        """
        venue_type_lower = venue_type.lower()
        location_lower = location.lower()

        # Determine primary source based on venue type
        if any(term in venue_type_lower for term in ["fine", "michelin", "upscale"]):
            primary_source = "michelin guide"
            secondary_source = "timeout"
        elif any(term in venue_type_lower for term in ["wine", "sommelier"]):
            primary_source = "raisin"
            secondary_source = "punch magazine"
        elif any(term in venue_type_lower for term in ["coffee", "cafe", "espresso"]):
            primary_source = "sprudge"
            secondary_source = "timeout"
        elif any(term in venue_type_lower for term in ["cocktail", "mixology", "speakeasy"]):
            primary_source = "punch magazine"
            secondary_source = "worlds 50 best bars"
        elif any(term in venue_type_lower for term in ["bar", "pub"]):
            primary_source = "timeout"
            secondary_source = "eater"
        else:
            primary_source = "timeout"
            secondary_source = "eater"

        # Adjust for major cities
        major_cities = ["new york", "paris", "london", "tokyo", "san francisco", "los angeles"]
        is_major_city = any(city in location_lower for city in major_cities)

        confidence = 0.7 if is_major_city else 0.5

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
            "confidence": confidence
        }

    def batch_map_sources(self, venues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map sources for multiple venues efficiently

        Args:
            venues: List of venue dictionaries

        Returns:
            List of venues with source mapping added
        """
        results = []

        for venue in venues:
            venue_name = venue.get('name', '')
            venue_type = venue.get('venue_type', 'restaurant')
            location = venue.get('location', venue.get('address', ''))

            # Get source mapping
            source_mapping = self.map_sources_for_venue(
                venue_name, venue_type, location
            )

            # Add source mapping to venue data
            venue_with_sources = venue.copy()
            venue_with_sources['source_mapping'] = source_mapping

            results.append(venue_with_sources)

        logger.info(f"✅ Mapped sources for {len(results)} venues")
        return results

    def get_search_queries_for_venue(self, venue_name: str, source_mapping: Dict[str, Any]) -> List[str]:
        """
        Generate specific search queries for a venue based on source mapping

        Args:
            venue_name: Name of the venue
            source_mapping: Source mapping result

        Returns:
            List of search query strings
        """
        queries = []

        primary_source = source_mapping.get('primary_source', '')
        secondary_source = source_mapping.get('secondary_source', '')

        # Use AI-suggested search terms if available
        search_terms = source_mapping.get('search_terms', [])
        if search_terms:
            queries.extend(search_terms)
        else:
            # Fallback: create basic search terms
            if primary_source:
                queries.append(f"{venue_name} {primary_source}")
            if secondary_source:
                queries.append(f"{venue_name} {secondary_source}")

        # Add a general review search as backup
        queries.append(f"{venue_name} review")

        logger.debug(f"Generated search queries for {venue_name}: {queries}")
        return queries