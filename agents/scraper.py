# agents/scraper.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from firecrawl import FirecrawlApp, AsyncFirecrawlApp
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
    AI-powered web scraper using Firecrawl SDK v2.0+ for restaurant content extraction.
    Updated for 2025 with named parameters, typed responses, and async extract API.
    """

    def __init__(self, config):
        self.config = config

        # Initialize Firecrawl client
        if not hasattr(config, 'FIRECRAWL_API_KEY') or not config.FIRECRAWL_API_KEY:
            raise ValueError("FIRECRAWL_API_KEY is required in config")

        # Use both sync and async clients
        self.firecrawl = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)
        self.async_firecrawl = AsyncFirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

        # Rate limiting and retry settings - updated for 2025 limits
        self.max_retries = 3
        self.retry_delay = 2
        self.max_concurrent = 3  # Conservative for free tier
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
            logger.info(f"Starting Firecrawl v2.0 scraping for {len(search_results)} URLs")

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

            dump_chain_state("firecrawl_v2_scraping_complete", {
                "input_count": len(search_results),
                "output_count": len(enriched_results),
                "stats": self.stats
            })

            return enriched_results

    async def _scrape_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape a single URL and extract restaurant data using Firecrawl v2.0 patterns.

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

            logger.info(f"Scraping URL with v2.0 SDK: {url}")

            # Rate limiting for 2025 API limits
            await asyncio.sleep(2)  # More reasonable for current limits

            try:
                # Extract restaurant data using Firecrawl's v2.0 extraction
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
        Extract restaurants with retry logic using Firecrawl v2.0 patterns.
        """

        for attempt in range(self.max_retries):
            try:
                # Method 1: Try new v2.0 scrape with extract (most reliable)
                try:
                    result = await self._extract_with_v2_scrape_extract(url)
                    if result.success and result.restaurants:
                        return result
                except Exception as e:
                    logger.warning(f"V2.0 scrape extract failed for {url} (attempt {attempt + 1}): {e}")

                # Method 2: Basic scrape + GPT processing (fallback)
                try:
                    result = await self._extract_with_v2_basic_scrape(url)
                    if result.success and result.restaurants:
                        return result
                except Exception as e:
                    logger.warning(f"V2.0 basic scrape failed for {url} (attempt {attempt + 1}): {e}")

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

    async def _extract_with_v2_scrape_extract(self, url: str) -> RestaurantExtractionResult:
        """
        Extract using Firecrawl v2.0 scrape_url with named parameters and typed responses.
        """
        logger.debug(f"Attempting v2.0 scrape with extract for {url}")

        try:
            # Use v2.0 scrape_url with named parameters
            scrape_result = await self.async_firecrawl.scrape_url(
                url=url,  # Named parameter
                formats=['markdown', 'extract'],
                extract={
                    'prompt': """
                    Find and extract ALL restaurants mentioned on this page. Return as JSON:

                    {{
                      "restaurants": [
                        {{
                          "name": "Restaurant Name",
                          "description": "Full description with cuisine, atmosphere, and specialties",
                          "address": "Street address",
                          "city": "City name", 
                          "price_range": "Price indicators (€, €€, etc.)",
                          "cuisine_type": "Cuisine type",
                          "recommended_dishes": ["dish1", "dish2"],
                          "source_url": "current_page_url"
                        }}
                      ],
                      "source_publication": "Publication name (Time Out, Eater, etc.)"
                    }}

                    Extract every restaurant mentioned, even if information is incomplete.
                    """,
                    'schema': RestaurantListSchema.model_json_schema()
                }
            )

            # Handle v2.0 typed response object
            extracted_data = {}

            # Try to access data in different ways based on v2.0 response structure
            if hasattr(scrape_result, 'data') and scrape_result.data:
                data = scrape_result.data
                if isinstance(data, dict):
                    extracted_data = data.get('extract', {})
                else:
                    # data might be the extract result directly
                    extracted_data = data if isinstance(data, dict) else {}
            elif hasattr(scrape_result, 'extract'):
                extracted_data = scrape_result.extract
            elif hasattr(scrape_result, 'success') and scrape_result.success:
                # Try to convert object to dict
                result_dict = dict(scrape_result) if hasattr(scrape_result, '__dict__') else {}
                extracted_data = result_dict.get('extract', result_dict.get('data', {}))
            else:
                raise Exception("No data returned from v2.0 scrape")

            # Extract restaurants from the response
            restaurants = extracted_data.get("restaurants", []) if isinstance(extracted_data, dict) else []
            source_name = extracted_data.get("source_publication", self._extract_source_name(url)) if isinstance(extracted_data, dict) else self._extract_source_name(url)

            # Ensure all restaurants have source_url
            for restaurant in restaurants:
                restaurant["source_url"] = url

            self.stats["credits_used"] += 10  # Scrape with extract cost

            return RestaurantExtractionResult(
                restaurants=restaurants,
                source_url=url,
                source_name=source_name,
                extraction_method="v2_scrape_extract",
                success=True
            )

        except Exception as e:
            raise Exception(f"V2.0 scrape extract failed: {str(e)}")

    async def _extract_with_v2_basic_scrape(self, url: str) -> RestaurantExtractionResult:
        """
        Last resort: Basic scrape + GPT processing using v2.0 patterns.
        """
        logger.debug(f"Attempting v2.0 basic scrape for {url}")

        try:
            # Use v2.0 basic scrape endpoint with named parameters
            scrape_result = await self.async_firecrawl.scrape_url(
                url=url,  # Named parameter
                formats=["markdown"]
            )

            # Handle v2.0 typed response object for markdown
            markdown_content = None

            if hasattr(scrape_result, 'data') and scrape_result.data:
                data = scrape_result.data
                if isinstance(data, dict):
                    markdown_content = data.get("markdown", "")
                else:
                    # data might be the markdown directly
                    markdown_content = data if isinstance(data, str) else ""
            elif hasattr(scrape_result, 'markdown'):
                markdown_content = scrape_result.markdown
            elif hasattr(scrape_result, 'success') and scrape_result.success:
                # Try to convert object to dict
                result_dict = dict(scrape_result) if hasattr(scrape_result, '__dict__') else {}
                markdown_content = result_dict.get('markdown', result_dict.get('data', {}).get('markdown', ''))
            else:
                raise Exception("No markdown content returned from v2.0 basic scrape")

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
                    {{
                      "restaurants": [
                        {{
                          "name": "Restaurant Name",
                          "description": "Description with key details",
                          "address": "Address if available",
                          "city": "City",
                          "price_range": "Price range",
                          "cuisine_type": "Cuisine type",
                          "recommended_dishes": ["dish1", "dish2"]
                        }}
                      ],
                      "source_publication": "Publication name"
                    }}
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

                    self.stats["credits_used"] += 1  # Basic scrape cost

                    return RestaurantExtractionResult(
                        restaurants=restaurants,
                        source_url=url,
                        source_name=source_name,
                        extraction_method="v2_basic_scrape",
                        success=True
                    )

                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse GPT response: {e}")
            else:
                raise Exception("No markdown content returned")

        except Exception as e:
            raise Exception(f"V2.0 basic scrape failed: {str(e)}")

    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL - updated with more 2025 domains"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Updated mapping with more domains for 2025
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
                'guardian.com': 'The Guardian',
                'telegraph.co.uk': 'The Telegraph',
                'cntraveler.com': 'Condé Nast Traveler',
                'laliste.com': 'La Liste',
                'oadguides.com': 'OAD Guides',
                '50best.com': "World's 50 Best",
                'theworlds50best.com': "World's 50 Best",
                'zagat.com': 'Zagat',
                'opentable.com': 'OpenTable',
                'saveur.com': 'Saveur',
                'foodandwine.com': 'Food & Wine',
                'thrillist.com': 'Thrillist',
                'delish.com': 'Delish',
                'bonappetit.com': 'Bon Appétit',
                'epicurious.com': 'Epicurious',
                'seriouseats.com': 'Serious Eats',
                'resy.com': 'Resy',
                'tasting-table.com': 'Tasting Table'
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
        logger.info(f"Firecrawl v2.0 Scraping Statistics:")
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