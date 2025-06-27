# agents/scraper.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from firecrawl import FirecrawlApp
from langchain_core.tracers.context import tracing_v2_enabled
from utils.debug_utils import dump_chain_state
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class RestaurantExtractionResult:
    """Result from restaurant extraction"""
    restaurants: List[Dict[str, Any]]
    source_url: str
    source_name: str
    extraction_method: str
    success: bool
    error_message: Optional[str] = None

class RestaurantSchema(BaseModel):
    """Pydantic schema for restaurant extraction"""
    name: str = Field(description="Restaurant name exactly as written on the website")
    description: str = Field(description="Full description with all key details like cuisine, atmosphere, specialties")
    address: str = Field(default="", description="Street address if available")
    city: str = Field(default="", description="City name")
    price_range: str = Field(default="", description="Price range (€, €€, €€€ or similar)")
    cuisine_type: str = Field(default="", description="Type of cuisine")
    recommended_dishes: List[str] = Field(default_factory=list, description="Signature dishes mentioned")
    phone: str = Field(default="", description="Phone number if available")
    website: str = Field(default="", description="Restaurant website if mentioned")
    rating: str = Field(default="", description="Rating or review score if mentioned")
    opening_hours: str = Field(default="", description="Opening hours if available")

class RestaurantListSchema(BaseModel):
    """Schema for the complete restaurant list extraction"""
    restaurants: List[RestaurantSchema] = Field(description="List of all restaurants found on the page")
    total_count: int = Field(description="Total number of restaurants extracted")
    source_publication: str = Field(description="Name of the publication/guide (e.g., Time Out, Eater)")

class FirecrawlWebScraper:
    """
    AI-powered web scraper using Firecrawl for restaurant content extraction.
    Designed to extract comprehensive restaurant data from complex, JavaScript-heavy pages.
    """

    def __init__(self, config):
        self.config = config

        # Initialize Firecrawl client
        if not hasattr(config, 'FIRECRAWL_API_KEY') or not config.FIRECRAWL_API_KEY:
            raise ValueError("FIRECRAWL_API_KEY is required in config")

        self.firecrawl = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

        # Rate limiting and retry settings
        self.max_retries = 3
        self.retry_delay = 2
        self.max_concurrent = 2  # Can handle 2 concurrent since we have pre-filtered quality URLs
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Statistics tracking
        self.stats = {
            "total_scraped": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_restaurants_found": 0,
            "credits_used": 0
        }

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for scraping search results.

        Args:
            search_results: List of search results from BraveSearchAgent

        Returns:
            List of enriched results with scraped restaurant data
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            logger.info(f"Starting Firecrawl scraping for {len(search_results)} URLs")

            # Create async tasks for all URLs
            tasks = []
            for result in search_results:
                task = self._scrape_single_result(result)
                tasks.append(task)

            # Execute all scraping tasks concurrently
            scraped_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            enriched_results = []
            for original_result, scraped_data in zip(search_results, scraped_results):
                if isinstance(scraped_data, Exception):
                    logger.error(f"Error scraping {original_result.get('url', 'unknown')}: {scraped_data}")
                    # Keep original result but mark as failed
                    original_result["scraping_failed"] = True
                    original_result["scraping_error"] = str(scraped_data)
                    enriched_results.append(original_result)
                else:
                    enriched_results.append(scraped_data)

            # Log final statistics
            self._log_scraping_stats()

            dump_chain_state("firecrawl_scraping_complete", {
                "input_count": len(search_results),
                "output_count": len(enriched_results),
                "stats": self.stats
            })

            return enriched_results

    async def _scrape_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape a single URL and extract restaurant data.

        Args:
            result: Search result dictionary with URL and metadata

        Returns:
            Enriched result with scraped restaurant data
        """
        async with self.semaphore:  # Rate limiting
            url = result.get("url", "")
            if not url:
                logger.warning("No URL found in result")
                return result

            logger.info(f"Scraping URL: {url}")

            # Add small delay between requests to respect free tier limits
            await asyncio.sleep(7)  # 10 requests per minute = 6+ seconds between requests

            try:
                # Extract restaurant data using Firecrawl's AI extraction
                extraction_result = await self._extract_restaurants_with_retry(url)

                if extraction_result.success:
                    self.stats["successful_extractions"] += 1
                    self.stats["total_restaurants_found"] += len(extraction_result.restaurants)

                    # Enrich the original result with scraped data
                    enriched_result = result.copy()
                    enriched_result.update({
                        "scraped_content": self._format_restaurants_for_analyzer(extraction_result.restaurants),
                        "restaurants_data": extraction_result.restaurants,
                        "source_info": {
                            "name": extraction_result.source_name,
                            "url": url,
                            "extraction_method": extraction_result.extraction_method
                        },
                        "scraping_success": True,
                        "restaurant_count": len(extraction_result.restaurants)
                    })

                    logger.info(f"Successfully extracted {len(extraction_result.restaurants)} restaurants from {url}")
                    return enriched_result

                else:
                    self.stats["failed_extractions"] += 1
                    logger.error(f"Failed to extract restaurants from {url}: {extraction_result.error_message}")

                    # Return original result with failure info
                    result["scraping_failed"] = True
                    result["scraping_error"] = extraction_result.error_message
                    return result

            except Exception as e:
                self.stats["failed_extractions"] += 1
                logger.error(f"Exception while scraping {url}: {e}")

                result["scraping_failed"] = True
                result["scraping_error"] = str(e)
                return result

            finally:
                self.stats["total_scraped"] += 1

    async def _extract_restaurants_with_retry(self, url: str) -> RestaurantExtractionResult:
        """
        Extract restaurants with retry logic and multiple extraction methods.

        Args:
            url: URL to scrape

        Returns:
            RestaurantExtractionResult with extracted data
        """

        for attempt in range(self.max_retries):
            try:
                # Method 1: Try AI extraction with schema first (most precise)
                try:
                    result = await self._extract_with_ai_schema(url)
                    if result.success and result.restaurants:
                        return result
                except Exception as e:
                    logger.warning(f"AI schema extraction failed for {url} (attempt {attempt + 1}): {e}")

                # Method 2: Fallback to prompt-based extraction
                try:
                    result = await self._extract_with_prompt(url)
                    if result.success and result.restaurants:
                        return result
                except Exception as e:
                    logger.warning(f"Prompt extraction failed for {url} (attempt {attempt + 1}): {e}")

                # Method 3: Last resort - basic scrape + GPT processing
                try:
                    result = await self._extract_with_basic_scrape(url)
                    if result.success and result.restaurants:
                        return result
                except Exception as e:
                    logger.warning(f"Basic scrape failed for {url} (attempt {attempt + 1}): {e}")

                # If we reach here, all methods failed for this attempt
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.error(f"Unexpected error during extraction attempt {attempt + 1} for {url}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        # All attempts failed
        return RestaurantExtractionResult(
            restaurants=[],
            source_url=url,
            source_name=self._extract_source_name(url),
            extraction_method="failed",
            success=False,
            error_message="All extraction methods failed after retries"
        )

    async def _extract_with_ai_schema(self, url: str) -> RestaurantExtractionResult:
        """
        Extract using Firecrawl's AI extraction with Pydantic schema.
        This is the most precise method.
        """
        logger.debug(f"Attempting AI schema extraction for {url}")

        # Use Firecrawl's extract endpoint with schema
        extract_result = self.firecrawl.extract(
            urls=[url],  # Note: urls is now a list
            schema=RestaurantListSchema.model_json_schema(),
            prompt="""
            Extract ALL restaurants mentioned on this page. For each restaurant, get:
            - Exact name as written
            - Complete description with all details (cuisine, atmosphere, specialties)
            - Full address if available
            - City name
            - Price indicators (€, €€, €€€, $ symbols, price ranges)
            - Cuisine type
            - Any recommended dishes or specialties mentioned
            - Contact info if available
            - Hours if mentioned
            - Ratings or review scores

            Also identify the source publication name (Time Out, Eater, Michelin, etc.).

            Be thorough - extract every restaurant mentioned, even if briefly.
            """
        )

        if extract_result and extract_result.get("success"):
            extracted_data = extract_result.get("data", {})

            # Convert to our format
            restaurants = []
            if "restaurants" in extracted_data:
                for rest_data in extracted_data["restaurants"]:
                    restaurant = {
                        "name": rest_data.get("name", ""),
                        "description": rest_data.get("description", ""),
                        "address": rest_data.get("address", ""),
                        "city": rest_data.get("city", ""),
                        "price_range": rest_data.get("price_range", ""),
                        "cuisine_type": rest_data.get("cuisine_type", ""),
                        "recommended_dishes": rest_data.get("recommended_dishes", []),
                        "phone": rest_data.get("phone", ""),
                        "website": rest_data.get("website", ""),
                        "rating": rest_data.get("rating", ""),
                        "opening_hours": rest_data.get("opening_hours", ""),
                        "source_url": url
                    }
                    restaurants.append(restaurant)

            source_name = extracted_data.get("source_publication", self._extract_source_name(url))
            self.stats["credits_used"] += 50  # AI extraction costs 50 credits

            return RestaurantExtractionResult(
                restaurants=restaurants,
                source_url=url,
                source_name=source_name,
                extraction_method="ai_schema",
                success=True
            )

        raise Exception("AI schema extraction returned no valid data")

    async def _extract_with_prompt(self, url: str) -> RestaurantExtractionResult:
        """
        Extract using Firecrawl's AI extraction with natural language prompt.
        Fallback method when schema extraction fails.
        """
        logger.debug(f"Attempting prompt-based extraction for {url}")

        extract_result = self.firecrawl.extract(
            urls=[url],  # Note: urls is now a list
            prompt="""
            Find and extract ALL restaurants mentioned on this page. Return as JSON:

            {
              "restaurants": [
                {
                  "name": "Restaurant Name",
                  "description": "Full description with cuisine, atmosphere, and specialties",
                  "address": "Street address",
                  "city": "City name", 
                  "price_range": "Price indicators (€, €€, etc.)",
                  "cuisine_type": "Cuisine type",
                  "recommended_dishes": ["dish1", "dish2"],
                  "source_url": "current_page_url"
                }
              ],
              "source_publication": "Publication name (Time Out, Eater, etc.)"
            }

            Extract every restaurant mentioned, even if information is incomplete.
            """
        )

        if extract_result and extract_result.get("success"):
            extracted_data = extract_result.get("data", {})

            try:
                # The data structure may be different
                restaurants = extracted_data.get("restaurants", [])
                source_name = extracted_data.get("source_publication", self._extract_source_name(url))

                # Ensure all restaurants have source_url
                for restaurant in restaurants:
                    restaurant["source_url"] = url

                self.stats["credits_used"] += 50

                return RestaurantExtractionResult(
                    restaurants=restaurants,
                    source_url=url,
                    source_name=source_name,
                    extraction_method="ai_prompt",
                    success=True
                )

            except Exception as e:
                raise Exception(f"Failed to parse data from prompt extraction: {e}")

        raise Exception("Prompt-based extraction returned no valid data")

    async def _extract_with_basic_scrape(self, url: str) -> RestaurantExtractionResult:
        """
        Last resort: Basic scrape + manual processing.
        Uses regular scrape endpoint and processes content.
        """
        logger.debug(f"Attempting basic scrape for {url}")

        # Use basic scrape endpoint
        scrape_result = self.firecrawl.scrape_url(
            url,
            formats=["markdown"]
        )

        if scrape_result and scrape_result.get("success"):
            data = scrape_result.get("data", {})
            markdown_content = data.get("markdown", "")

            if markdown_content:
                # Process with local GPT-4o to extract restaurants
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate

                model = ChatOpenAI(model=self.config.OPENAI_MODEL, temperature=0.2)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", """
                    You are an expert at extracting restaurant information from web content.
                    Extract ALL restaurants mentioned in the content.

                    Return JSON format:
                    {
                      "restaurants": [
                        {
                          "name": "Restaurant Name",
                          "description": "Description with key details",
                          "address": "Address if available",
                          "city": "City",
                          "price_range": "Price range",
                          "cuisine_type": "Cuisine type",
                          "recommended_dishes": ["dish1", "dish2"]
                        }
                      ],
                      "source_publication": "Publication name"
                    }
                    """),
                    ("human", "Content from {url}:\n\n{content}")
                ])

                chain = prompt | model
                response = await chain.ainvoke({
                    "url": url,
                    "content": markdown_content[:4000]  # Limit content size
                })

                try:
                    # Parse response
                    content = response.content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]

                    result_data = json.loads(content.strip())
                    restaurants = result_data.get("restaurants", [])
                    source_name = result_data.get("source_publication", self._extract_source_name(url))

                    # Add source URL to each restaurant
                    for restaurant in restaurants:
                        restaurant["source_url"] = url

                    self.stats["credits_used"] += 1  # Basic scrape costs 1 credit

                    return RestaurantExtractionResult(
                        restaurants=restaurants,
                        source_url=url,
                        source_name=source_name,
                        extraction_method="basic_scrape",
                        success=True
                    )

                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse GPT response: {e}")

        raise Exception("Basic scrape returned no content")

    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Map known domains to proper names
            domain_mapping = {
                'timeout.com': 'Time Out',
                'eater.com': 'Eater',
                'thefork.com': 'The Fork',
                'infatuation.com': 'The Infatuation',
                'guide.michelin.com': 'Michelin Guide',
                'michelin.com': 'Michelin Guide',
                'worldofmouth.app': 'World of Mouth',
                'nytimes.com': 'New York Times',
                'forbes.com': 'Forbes',
                'guardian.co.uk': 'The Guardian',
                'telegraph.co.uk': 'The Telegraph',
                'cntraveler.com': 'Condé Nast Traveler',
                'laliste.com': 'La Liste',
                'oadguides.com': 'OAD Guides',
                '50best.com': "World's 50 Best",
                'theworlds50best.com': "World's 50 Best",
                'zagat.com': 'Zagat',
                'opentable.com': 'OpenTable',
                'saveur.com': 'Saveur',
                'foodandwine.com': 'Food & Wine'
            }

            # Check for exact matches first
            if domain in domain_mapping:
                return domain_mapping[domain]

            # Check for partial matches
            for key, value in domain_mapping.items():
                if key in domain:
                    return value

            # Extract main domain name and capitalize
            main_domain = domain.split('.')[0]
            return main_domain.replace('-', ' ').replace('_', ' ').title()

        except Exception:
            return "Unknown Source"

    def _format_restaurants_for_analyzer(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Format extracted restaurant data for the list analyzer.
        Creates a clean, structured text that Mistral can process.
        """
        if not restaurants:
            return ""

        formatted_parts = []
        for i, restaurant in enumerate(restaurants, 1):
            parts = [f"Restaurant {i}: {restaurant.get('name', 'Unknown')}"]

            if restaurant.get('description'):
                parts.append(f"Description: {restaurant['description']}")

            if restaurant.get('address'):
                parts.append(f"Address: {restaurant['address']}")

            if restaurant.get('city'):
                parts.append(f"City: {restaurant['city']}")

            if restaurant.get('cuisine_type'):
                parts.append(f"Cuisine: {restaurant['cuisine_type']}")

            if restaurant.get('price_range'):
                parts.append(f"Price Range: {restaurant['price_range']}")

            if restaurant.get('recommended_dishes'):
                dishes = ", ".join(restaurant['recommended_dishes'])
                parts.append(f"Recommended Dishes: {dishes}")

            if restaurant.get('rating'):
                parts.append(f"Rating: {restaurant['rating']}")

            formatted_parts.append("\n".join(parts))

        return "\n\n".join(formatted_parts)

    def _log_scraping_stats(self):
        """Log scraping statistics"""
        logger.info(f"Scraping Statistics:")
        logger.info(f"  Total URLs scraped: {self.stats['total_scraped']}")
        logger.info(f"  Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"  Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"  Total restaurants found: {self.stats['total_restaurants_found']}")
        logger.info(f"  Estimated credits used: {self.stats['credits_used']}")

        success_rate = (self.stats['successful_extractions'] / max(self.stats['total_scraped'], 1)) * 100
        logger.info(f"  Success rate: {success_rate:.1f}%")

    def get_stats(self) -> Dict[str, Any]:
        """Get current scraping statistics"""
        return self.stats.copy()

# Legacy compatibility wrapper for existing code
class WebScraper:
    """
    Legacy wrapper to maintain compatibility with existing code.
    Redirects to FirecrawlWebScraper.
    """

    def __init__(self, config):
        self.scraper = FirecrawlWebScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.scraper.scrape_search_results(search_results)

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Alternative method name for compatibility"""
        return await self.scraper.scrape_search_results(search_results)