# agents/firecrawl_scraper.py - FIXED VERSION
"""
FIXED: Firecrawl Web Scraper with Anti-Bot Protection Handling

Issues Fixed:
1. Added missing scrape_url() method (was only scrape_search_results())
2. Enhanced anti-bot detection and handling for 403 errors
3. Improved timeout handling for sites like timeoutdubai.com
4. Better error reporting and fallback mechanisms
"""

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
    FIXED: AI-powered web scraper using Firecrawl SDK v2.0+ for restaurant content extraction.
    Enhanced with anti-bot protection handling and missing methods.
    """

    def __init__(self, config):
        self.config = config

        # Initialize Firecrawl client
        if not hasattr(config, 'FIRECRAWL_API_KEY') or not config.FIRECRAWL_API_KEY:
            raise ValueError("FIRECRAWL_API_KEY is required in config")

        # Use both sync and async clients
        self.firecrawl = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)
        self.async_firecrawl = AsyncFirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

        # Enhanced rate limiting and retry settings for anti-bot protection
        self.max_retries = 3
        self.retry_delay = 3  # Increased delay for anti-bot sites
        self.max_concurrent = 2  # Reduced concurrency to avoid triggering bot detection
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Enhanced timeout settings for problematic sites
        self.default_timeout = 30
        self.problematic_site_timeout = 60  # Longer timeout for sites with anti-bot protection

        # Known problematic domains that have strong anti-bot protection
        self.problematic_domains = [
            'timeout.com', 'timeoutdubai.com', 'timeoutlondon.com',
            'eater.com', 'thrillist.com', 'bonappetit.com',
            'foodandwine.com', 'zagat.com', 'yelp.com'
        ]

        # Statistics tracking
        self.stats = {
            "total_scraped": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_restaurants_found": 0,
            "credits_used": 0,
            "anti_bot_detected": 0,
            "timeouts": 0
        }

    def _is_problematic_domain(self, url: str) -> bool:
        """Check if URL is from a domain known for anti-bot protection"""
        return any(domain in url.lower() for domain in self.problematic_domains)

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        FIXED: Single URL scraping method (was missing!)
        This method was being called by smart_scraper.py but didn't exist.

        Args:
            url: URL to scrape

        Returns:
            Dictionary with scraping results
        """
        logger.info(f"🔥 Firecrawl scraping single URL: {url}")

        is_problematic = self._is_problematic_domain(url)
        if is_problematic:
            logger.warning(f"⚠️ Detected problematic domain for {url} - using enhanced anti-bot settings")

        try:
            extraction_result = await self._extract_restaurants_with_retry(url)

            if extraction_result.success:
                return {
                    "success": True,
                    "content": self._format_restaurants_for_analyzer(extraction_result.restaurants),
                    "restaurants_found": [r.get("name", "") for r in extraction_result.restaurants],
                    "extraction_method": extraction_result.extraction_method,
                    "source_name": extraction_result.source_name
                }
            else:
                return {
                    "success": False,
                    "content": "",
                    "restaurants_found": [],
                    "error": extraction_result.error_message,
                    "extraction_method": "failed"
                }

        except Exception as e:
            logger.error(f"❌ Firecrawl scraping failed for {url}: {e}")
            return {
                "success": False,
                "content": "",
                "restaurants_found": [],
                "error": str(e),
                "extraction_method": "failed"
            }

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for scraping search results.
        Automatically detects specialized URLs and routes them appropriately to save Firecrawl credits.
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            logger.info(f"Starting intelligent scraping for {len(search_results)} URLs")

            # Separate URLs based on whether we have specialized handlers
            specialized_urls = []
            regular_urls = []

            # Import specialized scraper to check which URLs it can handle
            try:
                from agents.specialized_scraper import EaterTimeoutSpecializedScraper

                # Create a temporary instance to check handlers
                temp_scraper = EaterTimeoutSpecializedScraper(self.config)

                for result in search_results:
                    url = result.get('url', '')

                    # Check if ANY specialized handler can process this URL
                    if temp_scraper._find_handler(url):
                        specialized_urls.append(result)
                        logger.info(f"Routing to specialized handler: {url}")
                    else:
                        regular_urls.append(result)

            except ImportError:
                logger.warning("Specialized scraper not available, using Firecrawl for all URLs")
                regular_urls = search_results

            enriched_results = []

            # Process specialized URLs with RSS/sitemap approach (NO FIRECRAWL CREDITS USED)
            if specialized_urls:
                logger.info(f"💡 Processing {len(specialized_urls)} URLs with specialized handlers (saving Firecrawl credits)")

                try:
                    from agents.specialized_scraper import EaterTimeoutSpecializedScraper

                    async with EaterTimeoutSpecializedScraper(self.config) as specialized_scraper:
                        specialized_results = await specialized_scraper.process_specialized_urls(specialized_urls)
                        enriched_results.extend(specialized_results)

                        # Log specialized scraper stats
                        specialized_scraper._log_stats()

                except Exception as e:
                    logger.error(f"Error in specialized scraping: {e}")
                    # Fall back to regular scraping for these URLs
                    logger.warning(f"Falling back to Firecrawl for {len(specialized_urls)} URLs due to specialized scraper error")
                    regular_urls.extend(specialized_urls)

            # Process regular URLs with standard Firecrawl approach (USES FIRECRAWL CREDITS)
            if regular_urls:
                logger.info(f"🔥 Processing {len(regular_urls)} URLs with Firecrawl (using credits)")

                # Create async tasks for regular URLs
                tasks = []
                for result in regular_urls:
                    task = self._scrape_single_result(result)
                    tasks.append(task)

                # Execute all scraping tasks concurrently
                scraped_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and handle exceptions
                for original_result, scraped_data in zip(regular_urls, scraped_results):
                    if isinstance(scraped_data, Exception):
                        logger.error(f"Error scraping {original_result.get('url', 'unknown')}: {scraped_data}")
                        # Keep original result but mark as failed
                        original_result["scraping_failed"] = True
                        original_result["scraping_error"] = str(scraped_data)
                        enriched_results.append(original_result)
                    else:
                        enriched_results.append(scraped_data)

            # Log final statistics with credit usage info
            self._log_scraping_stats()

            dump_chain_state("intelligent_scraping_complete", {
                "input_count": len(search_results),
                "specialized_count": len(specialized_urls),
                "firecrawl_count": len(regular_urls),
                "output_count": len(enriched_results),
                "firecrawl_credits_saved": len(specialized_urls) * 10,  # Estimated credits saved
                "stats": self.stats
            })

            logger.info(f"💰 Credit optimization: {len(specialized_urls)} URLs processed without Firecrawl, saving ~{len(specialized_urls) * 10} credits")

            return enriched_results

    async def _scrape_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Scrape a single URL with better anti-bot protection handling.
        """
        async with self.semaphore:  # Rate limiting
            url = result.get("url", "")
            if not url:
                logger.warning("No URL found in result")
                return result

            logger.info(f"Scraping URL with v2.0 SDK: {url}")

            # Check if this is a problematic domain and adjust settings
            is_problematic = self._is_problematic_domain(url)

            if is_problematic:
                logger.warning(f"⚠️ Problematic domain detected: {url} - using enhanced anti-bot settings")
                self.stats["anti_bot_detected"] += 1

            # Use longer delays for problematic sites
            delay = 8 if is_problematic else 2
            await asyncio.sleep(delay)

            try:
                # Extract restaurant data using Firecrawl's v2.0 extraction
                extraction_result = await self._extract_restaurants_with_retry(url)

                if extraction_result.success and extraction_result.restaurants:
                    # Success - format for the pipeline
                    formatted_content = self._format_restaurants_for_analyzer(extraction_result.restaurants)

                    result.update({
                        "scraped_content": formatted_content,
                        "scraped_title": f"{extraction_result.source_name} Restaurant Guide",
                        "scraping_success": True,
                        "scraping_method": "firecrawl_v2",
                        "extraction_method": extraction_result.extraction_method,
                        "restaurants_found": [r.get("name", "") for r in extraction_result.restaurants],
                        "source_info": {
                            "name": extraction_result.source_name,
                            "url": url,
                            "extraction_method": extraction_result.extraction_method
                        }
                    })

                    self.stats["successful_extractions"] += 1
                    self.stats["total_restaurants_found"] += len(extraction_result.restaurants)

                else:
                    # Failed extraction
                    error_msg = extraction_result.error_message or "No restaurants found"
                    logger.warning(f"⚠️ No restaurants extracted from {url}: {error_msg}")

                    result.update({
                        "scraping_failed": True,
                        "scraping_error": error_msg,
                        "scraping_method": "firecrawl_v2",
                        "extraction_method": "failed",
                        "is_problematic_site": is_problematic
                    })

                    self.stats["failed_extractions"] += 1

                return result

            except Exception as e:
                logger.error(f"❌ Scraping error for {url}: {e}")

                # Determine error type for better handling
                error_type = "unknown"
                if "403" in str(e) or "forbidden" in str(e).lower():
                    error_type = "anti_bot_protection"
                elif "timeout" in str(e).lower():
                    error_type = "timeout"
                    self.stats["timeouts"] += 1

                result.update({
                    "scraping_failed": True,
                    "scraping_error": str(e),
                    "error_type": error_type,
                    "is_problematic_site": is_problematic,
                    "scraping_method": "firecrawl_v2"
                })

                self.stats["failed_extractions"] += 1
                return result

            finally:
                self.stats["total_scraped"] += 1

    async def _extract_restaurants_with_retry(self, url: str) -> RestaurantExtractionResult:
        """
        ENHANCED: Extract restaurants with enhanced retry logic for anti-bot protection.
        """
        is_problematic = self._is_problematic_domain(url)

        # Adjust retry settings based on domain type
        max_retries = 2 if is_problematic else self.max_retries
        retry_delay = 8 if is_problematic else self.retry_delay

        for attempt in range(max_retries):
            try:
                # Method 1: Try new v2.0 scrape with extract (most reliable)
                try:
                    result = await self._extract_with_v2_scrape_extract(url, is_problematic)
                    if result.success and result.restaurants:
                        return result
                except Exception as e:
                    logger.warning(f"V2.0 scrape extract failed for {url} (attempt {attempt + 1}): {e}")

                # For problematic sites, skip basic scrape to save time and avoid more bot detection
                if not is_problematic:
                    # Method 2: Basic scrape + GPT processing (fallback)
                    try:
                        result = await self._extract_with_v2_basic_scrape(url, is_problematic)
                        if result.success and result.restaurants:
                            return result
                    except Exception as e:
                        logger.warning(f"V2.0 basic scrape failed for {url} (attempt {attempt + 1}): {e}")

                # If we reach here, all methods failed for this attempt
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"⏳ Waiting {wait_time}s before retry {attempt + 2} for {url}")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Unexpected error during extraction attempt {attempt + 1} for {url}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        # All attempts failed
        error_msg = "All extraction methods failed after retries"
        if is_problematic:
            error_msg += " (site has strong anti-bot protection - Firecrawl may not be able to bypass)"

        return RestaurantExtractionResult(
            restaurants=[],
            source_url=url,
            source_name=self._extract_source_name(url),
            extraction_method="failed",
            success=False,
            error_message=error_msg
        )

    async def _extract_with_v2_scrape_extract(self, url: str, is_problematic: bool = False) -> RestaurantExtractionResult:
        """
        ENHANCED: Extract using Firecrawl v2.0 scrape_url with enhanced anti-bot settings.
        """
        # Use longer timeout for problematic sites
        timeout_seconds = self.problematic_site_timeout if is_problematic else self.default_timeout

        logger.debug(f"Attempting v2.0 scrape extract for {url} (timeout: {timeout_seconds}s)")

        try:
            # Enhanced Firecrawl options for anti-bot protection
            extract_options = {
                'schema': RestaurantListSchema.model_json_schema(),
                'systemPrompt': """You are an expert at extracting restaurant information from food guides and review sites.

                Extract ALL restaurants mentioned in the content with complete details.

                Return a JSON object with this structure:
                {{
                  "restaurants": [
                    {{
                      "name": "Restaurant Name",
                      "description": "Complete description including cuisine, atmosphere, specialties",
                      "address": "Street address if available",
                      "city": "City name",
                      "price_range": "Price indicator (€, €€, €€€)",
                      "cuisine_type": "Type of cuisine",
                      "recommended_dishes": ["dish1", "dish2"],
                      "source_url": "current_page_url"
                    }}
                  ],
                  "source_publication": "Publication name (Time Out, Eater, etc.)"
                }}

                Extract every restaurant mentioned, even if information is incomplete.
                """
            }

            # Enhanced scraping options for anti-bot protection
            scrape_options = {
                'formats': ['extract'],
                'extract': extract_options,
                'timeout': timeout_seconds * 1000,  # Convert to milliseconds
            }

            # Add anti-bot protection options for problematic sites
            if is_problematic:
                scrape_options.update({
                    'waitFor': 3000,  # Wait 3 seconds for content to load
                    'screenshot': False,  # Disable screenshot to be less detectable
                })

            # Use v2.0 scrape endpoint with enhanced options
            scrape_result = await self.async_firecrawl.scrape_url(
                url=url,
                **scrape_options
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

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout_seconds}s - site may have strong anti-bot protection")
        except Exception as e:
            error_msg = f"V2.0 scrape extract failed: {str(e)}"
            if "403" in str(e) or "forbidden" in str(e).lower():
                error_msg += " (Anti-bot protection detected - site blocking Firecrawl)"
            raise Exception(error_msg)

    async def _extract_with_v2_basic_scrape(self, url: str, is_problematic: bool = False) -> RestaurantExtractionResult:
        """
        ENHANCED: Last resort: Basic scrape + GPT processing with anti-bot handling.
        """
        timeout_seconds = self.problematic_site_timeout if is_problematic else self.default_timeout

        logger.debug(f"Attempting v2.0 basic scrape for {url}")

        try:
            # Enhanced basic scrape options
            scrape_options = {
                'formats': ['markdown'],
                'timeout': timeout_seconds * 1000,
            }

            # Add anti-bot protection for problematic sites
            if is_problematic:
                scrape_options.update({
                    'waitFor': 2000,  # Wait for content
                    'screenshot': False,
                })

            # Use v2.0 basic scrape endpoint with named parameters
            scrape_result = await self.async_firecrawl.scrape_url(
                url=url,
                **scrape_options
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

                    Return a JSON array of restaurants with this structure:
                    {{
                      "restaurants": [
                        {{
                          "name": "Restaurant Name",
                          "description": "Complete description",
                          "address": "Address if available",
                          "cuisine_type": "Cuisine type",
                          "recommended_dishes": ["dish1", "dish2"],
                          "source_url": "{url}"
                        }}
                      ]
                    }}
                    """),
                    ("human", "Extract restaurants from this content:\n\n{{content}}")
                ])

                chain = prompt | model
                response = await chain.ainvoke({{
                    "content": markdown_content[:8000],  # Limit content size
                    "url": url
                }})

                # Parse GPT response
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]

                try:
                    parsed_data = json.loads(content.strip())
                    restaurants = parsed_data.get("restaurants", [])

                    self.stats["credits_used"] += 5  # Basic scrape + GPT cost

                    return RestaurantExtractionResult(
                        restaurants=restaurants,
                        source_url=url,
                        source_name=self._extract_source_name(url),
                        extraction_method="v2_basic_scrape_gpt",
                        success=True
                    )
                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse GPT response: {e}")

            else:
                raise Exception("No content returned from basic scrape")

        except asyncio.TimeoutError:
            raise Exception(f"Basic scrape timeout after {timeout_seconds}s")
        except Exception as e:
            error_msg = f"V2.0 basic scrape failed: {str(e)}"
            if "403" in str(e) or "forbidden" in str(e).lower():
                error_msg += " (Anti-bot protection active)"
            raise Exception(error_msg)

    def _extract_source_name(self, url: str) -> str:
        """Extract readable source name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Extract main domain name
            main_domain = domain.split('.')[0]
            return main_domain.replace('-', ' ').replace('_', ' ').title()

        except Exception:
            return "Unknown Source"

    def _format_restaurants_for_analyzer(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Format extracted restaurant data for the list analyzer.
        Creates a clean, structured text that the AI can process.
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
        """Enhanced logging with anti-bot protection stats"""
        logger.info(f"🔥 Firecrawl v2.0 Scraping Statistics:")
        logger.info(f"  Total URLs scraped: {self.stats['total_scraped']}")
        logger.info(f"  Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"  Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"  Total restaurants found: {self.stats['total_restaurants_found']}")
        logger.info(f"  Estimated credits used: {self.stats['credits_used']}")
        logger.info(f"  Anti-bot protection detected: {self.stats['anti_bot_detected']}")
        logger.info(f"  Timeouts: {self.stats['timeouts']}")

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