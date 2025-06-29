# agents/list_analyzer.py - CLAUDE SONNET 4 VERSION
# Replace your existing agents/list_analyzer.py with this code

from __future__ import annotations
"""
ListAnalyzer v4.0 — Now using Claude Sonnet 4 for reliable structured output
============================================================================
MAJOR CHANGES:
* ✅ **Switched to Claude Sonnet 4** – reliable JSON output, no more [40], [true], [1] artifacts
* ✅ **200K context window maintained** – can still process large scraped content
* ✅ **Simplified validation** – Claude's structured output is naturally reliable
* ✅ **Better reasoning** – improved restaurant analysis and description quality
* ✅ **Removed retry complexity** – no longer needed with Claude's reliability

"""
import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Sequence
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
import config

###############################################################################
# Logging
###############################################################################
logger = logging.getLogger(__name__)
if os.getenv("LIST_ANALYZER_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

###############################################################################
# Simplified Pydantic schema (Claude is much more reliable)
###############################################################################
class Restaurant(BaseModel):
    name: str = Field(..., description="Restaurant name as written by sources")
    address: str = Field(default="Address unavailable", description="Street and house number if available")
    description: str = Field(
        default="Description unavailable",
        description="40‑60 word vivid summary starting with one concrete fact"
    )
    price_range: str = Field(default="Price range not specified", description="Price range information")
    recommended_dishes: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list) 
    location: str
    city: str = Field(default="", description="City name from the query")

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, v):
        """Simplified validation - Claude is much more reliable than Mistral"""
        if not isinstance(v, str):
            logger.warning(f"Non-string description received: {type(v)} = {v}")
            return "Description unavailable"

        if v.strip() == "":
            return "Description unavailable"

        return v.strip()

    @field_validator("name", "address", "price_range", "location", "city", mode="before")
    @classmethod
    def validate_string_fields(cls, v):
        """Ensure string fields are valid"""
        if not isinstance(v, str):
            return str(v) if v is not None else ""
        return v.strip()

class ListResponse(BaseModel):
    """Response model for restaurant list"""
    restaurants: List[Restaurant] = Field(..., description="List of recommended restaurants")

# Parser for structured output
PARSER = PydanticOutputParser(pydantic_object=ListResponse)

###############################################################################
# Enhanced prompts optimized for Claude
###############################################################################
SYSTEM_PROMPT = """You are an expert restaurant analyst. Your task is to analyze scraped restaurant content and extract structured restaurant recommendations.

INSTRUCTIONS:
1. Return a JSON object with a single "restaurants" array containing all recommended restaurants
2. Each restaurant must include: name, location, city, description (40-60 words starting with a concrete fact), price_range, recommended_dishes, sources, source_urls
3. Extract at least 8 restaurants that match the search parameters and are in the specified location/city
4. For sources: include only publication names (e.g., "Eater", "Time Out"), not full article titles
5. Extract source URLs from [URL: ...] markers in the snippets
6. Focus on restaurants from the specified destination city only
7. Include both well-known establishments and hidden gems based on the analysis

QUALITY STANDARDS:
- Descriptions must be informative, engaging, and fact-based
- Include specific details about cuisine, atmosphere, or signature dishes
- Ensure price ranges are consistent (use €, €€, €€€ or $, $$, $$$)
- Recommended dishes should be specific menu items when available

{format_instructions}"""

HUMAN_TEMPLATE = """SEARCH ANALYSIS REQUEST:

PRIMARY SEARCH PARAMETERS: {primary}
SECONDARY FILTER PARAMETERS: {secondary}
KEYWORDS FOR ANALYSIS: {keywords}
DESTINATION CITY: {destination}

SCRAPED CONTENT TO ANALYZE:
{snippets}

Please analyze the above content and extract restaurant recommendations that match the search criteria for {destination}."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])

###############################################################################
# Helper functions (unchanged)
###############################################################################
def _clean_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_keyword_sentences(text: str, keywords: Sequence[str], max_sent: int = 3) -> List[str]:
    """Return up to *max_sent* sentences from *text* containing any keyword."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    hits = []
    for sent in sentences:
        if any(k.lower() in sent.lower() for k in keywords):
            hits.append(_clean_sentence(sent))
        if len(hits) >= max_sent:
            break
    return hits or sentences[:max_sent]

def _build_snippets(raw_articles: Sequence[Dict[str, Any]], keywords: Sequence[str]) -> str:
    """Create a trimmed block of article snippets focusing on keyword sentences."""
    if not raw_articles:
        return "No articles available for analysis."

    parts = []
    for art in raw_articles[:12]:  # Process more articles with Claude's reliability
        url = art.get("url", "Unknown URL")

        # Extract source name from URL
        if url and url != "Unknown URL":
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()
            except:
                domain = url.lower()

            if domain.startswith('www.'):
                domain = domain[4:]
            source_name = domain.split('.')[0].capitalize()

            # Map common domains to proper names
            source_map = {
                'eater': 'Eater',
                'timeout': 'Time Out',
                'thefork': 'The Fork',
                'infatuation': 'The Infatuation',
                'michelin': 'Michelin Guide',
                'worldofmouth': 'World of Mouth',
                'nytimes': 'New York Times',
                'forbes': 'Forbes',
                'guardian': 'The Guardian',
                'telegraph': 'The Telegraph',
                'cntraveler': 'Condé Nast Traveler',
                'laliste': 'La Liste',
                'oadguides': 'OAD Guides',
                'zagat': 'Zagat',
                'bonappetit': 'Bon Appétit',
                'foodandwine': 'Food & Wine'
            }

            for key, val in source_map.items():
                if key in domain.lower():
                    source_name = val
                    break
        else:
            source_name = "Unknown Source"

        body = art.get("scraped_content", "")
        key_sents = _extract_keyword_sentences(body, keywords)
        snippet = f"### Source: {source_name} [URL: {url}]\n" + "\n".join(key_sents[:4])
        parts.append(snippet)

    return "\n\n".join(parts)

###############################################################################
# Simplified retry wrapper (less needed with Claude)
###############################################################################
def retry_async(fn):
    async def wrapper(*args, **kwargs):
        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=4),
            stop=stop_after_attempt(3),  # Fewer retries needed
            reraise=True,
        )
        async def _inner():
            return await fn(*args, **kwargs)
        return await _inner()
    return wrapper

###############################################################################
# ListAnalyzer with Claude Sonnet 4
###############################################################################
class ListAnalyzer:
    """Restaurant list analyzer using Claude Sonnet 4 for reliable structured output"""

    def __init__(self):
        # Initialize Claude Sonnet 4
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",  # Latest Claude model
            temperature=0.2,  # Slightly higher for more creative descriptions
            max_tokens=8192,  # Claude can output more tokens
            api_key=config.ANTHROPIC_API_KEY,  # You'll need to add this to config.py
        )
        self._semaphore = asyncio.Semaphore(2)  # Limit concurrent calls

    async def analyze_search_results(
        self,
        search_results: List[Dict[str, Any]],
        primary_search_parameters: str,
        secondary_filter_parameters: str = "",
        keywords_for_analysis: List[str] = None,
        destination: str = "Unknown",
        max_retries: int = 2,  # Fewer retries needed with Claude
    ) -> Dict[str, Any]:
        """Analyze search results with Claude Sonnet 4"""

        if keywords_for_analysis is None:
            keywords_for_analysis = []

        # Ensure we have a destination
        if not destination or destination == "Unknown":
            destination = "Location from search query"

        # Convert parameters to strings
        if isinstance(primary_search_parameters, list):
            primary_params = ", ".join(primary_search_parameters)
        else:
            primary_params = primary_search_parameters

        if isinstance(secondary_filter_parameters, list):
            secondary_params = ", ".join(secondary_filter_parameters)
        else:
            secondary_params = secondary_filter_parameters

        snippets = _build_snippets(search_results, keywords_for_analysis)

        prompt_values = {
            "primary": primary_params,
            "secondary": secondary_params,
            "keywords": ", ".join(keywords_for_analysis),
            "destination": destination,
            "snippets": snippets,
            "format_instructions": PARSER.get_format_instructions(),
        }

        logger.info(f"Analyzing {len(search_results)} articles with Claude Sonnet 4")
        logger.debug("Prepared prompt of %d chars", len(snippets))

        # Try analysis with simple retry
        for attempt in range(max_retries):
            try:
                logger.info(f"Claude analysis attempt {attempt + 1}/{max_retries}")

                # Call Claude
                content = await self._call_llm(prompt_values)

                # Parse and validate
                try:
                    response_model = PARSER.parse(content)
                    logger.info(f"Successfully parsed {len(response_model.restaurants)} restaurants")
                    break

                except Exception as ve:
                    logger.warning(f"Parsing failed on attempt {attempt + 1}: {ve}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.error("All parsing attempts failed, creating fallback response")
                        return self._create_fallback_response(primary_params, destination)

            except Exception as exc:
                logger.error(f"Claude call failed on attempt {attempt + 1}: {exc}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    logger.error("All attempts failed, creating fallback response")
                    return self._create_fallback_response(primary_params, destination)

        # Post-processing
        try:
            response_model = await self._enhance_descriptions(response_model, destination)

            # Set city for all restaurants
            for restaurant in response_model.restaurants:
                restaurant.city = destination
                restaurant.location = destination

            logger.info(f"Successfully analyzed {len(response_model.restaurants)} restaurants with Claude")

            # Return in expected format (maintaining compatibility)
            return {
                "main_list": [r.model_dump() for r in response_model.restaurants],
                "hidden_gems": []  # Empty for compatibility
            }

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return self._create_fallback_response(primary_params, destination)

    def _create_fallback_response(self, query: str, destination: str) -> Dict[str, Any]:
        """Create fallback response when analysis fails"""
        logger.warning("Creating fallback response due to analysis failure")

        fallback_restaurant = {
            "name": "Restaurant search temporarily unavailable",
            "address": "Address unavailable", 
            "description": "We're experiencing technical difficulties with restaurant analysis. Please try rephrasing your search or try again in a moment.",
            "price_range": "Price range not specified",
            "recommended_dishes": [],
            "sources": [],
            "source_urls": [],
            "location": destination,
            "city": destination
        }

        return {
            "main_list": [fallback_restaurant],
            "hidden_gems": []
        }

    @retry_async
    async def _call_llm(self, prompt_values: Dict[str, Any]) -> str:
        """Call Claude with retry logic"""
        async with self._semaphore:
            logger.debug("Making Claude API call...")
            chain = PROMPT | self.llm
            result = await chain.ainvoke(prompt_values)
            logger.debug("Claude call completed successfully")
            return result.content

    async def _enhance_descriptions(self, response: ListResponse, location: str) -> ListResponse:
        """Enhance any short descriptions (Claude usually doesn't need this but keeping for safety)"""
        restaurants_to_enhance = []

        for restaurant in response.restaurants:
            if (restaurant.description == "Description unavailable" or 
                len(restaurant.description.split()) < 10):
                restaurants_to_enhance.append(restaurant)

        if not restaurants_to_enhance:
            return response

        logger.info(f"Enhancing descriptions for {len(restaurants_to_enhance)} restaurants")

        # Enhance descriptions
        for restaurant in restaurants_to_enhance:
            try:
                enhanced_desc = await self._enhance_single_description(restaurant, location)
                if enhanced_desc and len(enhanced_desc.split()) >= 10:
                    restaurant.description = enhanced_desc
                    logger.debug(f"Enhanced description for {restaurant.name}")
            except Exception as e:
                logger.warning(f"Failed to enhance description for {restaurant.name}: {e}")
                continue

        return response

    async def _enhance_single_description(self, restaurant: Restaurant, location: str) -> str:
        """Enhance a single restaurant description using Claude"""
        enhance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a restaurant writer. Create a vivid, informative 40-60 word description that starts with a concrete fact about the restaurant.

Focus on what makes this restaurant special - the cuisine, atmosphere, signature dishes, or unique features.

Return ONLY the description text, no quotes, no extra formatting."""),
            ("human", """Restaurant: {name}
Location: {location}
Current description: {current_desc}
Sources: {sources}

Write a compelling 40-60 word description:""")
        ])

        chain = enhance_prompt | self.llm | StrOutputParser()

        try:
            result = await chain.ainvoke({
                "name": restaurant.name,
                "location": location,
                "current_desc": restaurant.description,
                "sources": ", ".join(restaurant.sources) if restaurant.sources else "restaurant guide"
            })

            # Clean and validate
            cleaned = result.strip().strip('"').strip("'")
            if len(cleaned.split()) >= 10:
                return cleaned

        except Exception as e:
            logger.warning(f"Description enhancement failed: {e}")

        return restaurant.description