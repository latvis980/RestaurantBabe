# agents/enhanced_scraper.py - Improved WebScraper with Scrapy-inspired practices
import asyncio
import logging
import time
import re
import json
import os
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import platform

# Set proper event loop policy for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Create semaphore for concurrency control
SEM = asyncio.Semaphore(3)

# Check for Playwright availability
PLAYWRIGHT_ENABLED = True
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    logger = logging.getLogger("restaurant-recommender.enhanced_scraper")
    logger.info("Playwright successfully imported")
except ImportError as e:
    PLAYWRIGHT_ENABLED = False
    logger = logging.getLogger("restaurant-recommender.enhanced_scraper")
    logger.error(f"Failed to import Playwright: {e}, falling back to HTTP methods")

from readability import Document
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled

from utils.async_utils import track_async_task
from utils.debug_utils import dump_chain_state

logger = logging.getLogger("restaurant-recommender.enhanced_scraper")

class ContentExtractor:
    """
    Advanced content extraction using CSS selectors and content patterns
    inspired by Scrapy's extraction methods
    """

    # Restaurant-specific selectors prioritized by likelihood of containing restaurant data
    RESTAURANT_SELECTORS = [
        # Main content areas
        'main', 'article', '.content', '.post', '.article-content', '.entry-content',

        # Restaurant-specific classes (common patterns from food sites)
        '.restaurant', '.venue', '.place', '.listing', '.restaurant-item',
        '.restaurant-card', '.venue-card', '.place-card',

        # List containers
        '.restaurants-list', '.venue-list', '.places-list', '.directory',

        # Content sections
        '.recommendations', '.guide', '.review', '.reviews'
    ]

    # Selectors for restaurant names (in order of priority)
    NAME_SELECTORS = [
        'h1', 'h2', 'h3', 'h4',  # Headers most likely to contain names
        '.restaurant-name', '.venue-name', '.place-name', '.name',
        '.title', '.restaurant-title', '.venue-title',
        'strong', 'b',  # Bold text often indicates names
        '.restaurant-link a', '.venue-link a',  # Links to restaurant pages
        'a[href*="restaurant"]', 'a[href*="venue"]'  # Links with restaurant/venue in URL
    ]

    # Selectors for addresses
    ADDRESS_SELECTORS = [
        'address', '.address', '.location', '.venue-address',
        '.restaurant-address', '.street-address', '.addr',
        '.contact-info .address', '.location-info'
    ]

    # Selectors for descriptions/content
    DESCRIPTION_SELECTORS = [
        'p', '.description', '.review-text', '.excerpt',
        '.restaurant-description', '.venue-description',
        '.summary', '.intro', '.content-text'
    ]

    # Selectors for lists (restaurant listings)
    LIST_SELECTORS = [
        'ul li', 'ol li', '.list-item', '.restaurant-item',
        '.venue-item', '.place-item', '.directory-item'
    ]

    @staticmethod
    def extract_structured_content(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract structured content using CSS selectors
        Returns organized data similar to Scrapy items
        """
        extracted = {
            "url": url,
            "title": "",
            "restaurant_names": [],
            "addresses": [],
            "descriptions": [],
            "list_items": [],
            "main_content": "",
            "structured_sections": []
        }

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            extracted["title"] = title_tag.get_text().strip()

        # Find main content area using priority selectors
        main_content_element = None
        for selector in ContentExtractor.RESTAURANT_SELECTORS:
            element = soup.select_one(selector)
            if element:
                main_content_element = element
                break

        # Fallback to body if no main content found
        if not main_content_element:
            main_content_element = soup.find('body') or soup

        # Extract restaurant names
        for selector in ContentExtractor.NAME_SELECTORS:
            elements = main_content_element.select(selector)
            for element in elements:
                text = element.get_text().strip()
                # Filter for likely restaurant names (not too long, not too short)
                if text and 2 < len(text) < 80 and not text.lower().startswith(('http', 'www')):
                    # Avoid duplicates
                    if text not in [item["text"] for item in extracted["restaurant_names"]]:
                        extracted["restaurant_names"].append({
                            "text": text,
                            "selector": selector,
                            "context": element.parent.name if element.parent else "unknown"
                        })

        # Extract addresses
        for selector in ContentExtractor.ADDRESS_SELECTORS:
            elements = main_content_element.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and 5 < len(text) < 200:
                    extracted["addresses"].append({
                        "text": text,
                        "selector": selector
                    })

        # Extract descriptions/paragraphs
        for selector in ContentExtractor.DESCRIPTION_SELECTORS:
            elements = main_content_element.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and 10 < len(text) < 1000:  # Reasonable length for descriptions
                    extracted["descriptions"].append({
                        "text": text,
                        "selector": selector,
                        "length": len(text)
                    })

        # Extract list items (often contain restaurant listings)
        for selector in ContentExtractor.LIST_SELECTORS:
            elements = main_content_element.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and 5 < len(text) < 500:
                    extracted["list_items"].append({
                        "text": text,
                        "selector": selector
                    })

        # Extract structured sections (divs with restaurant-related classes)
        restaurant_sections = main_content_element.select('[class*="restaurant"], [class*="venue"], [class*="place"]')
        for section in restaurant_sections:
            section_data = {
                "class": section.get('class', []),
                "content": section.get_text().strip()[:500],  # Limit length
                "subsections": []
            }

            # Look for nested structure within this section
            names = section.select('h1, h2, h3, h4, strong, b')
            for name in names[:3]:  # Limit to first 3 to avoid noise
                name_text = name.get_text().strip()
                if name_text and len(name_text) < 100:
                    section_data["subsections"].append({
                        "type": "name",
                        "text": name_text,
                        "tag": name.name
                    })

            if section_data["content"]:
                extracted["structured_sections"].append(section_data)

        # Create main content text
        extracted["main_content"] = main_content_element.get_text(separator='\n').strip()

        return extracted

class RetryMiddleware:
    """
    Retry logic for failed requests with exponential backoff
    """

    def __init__(self, max_retries=3, base_delay=1, max_delay=10):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def fetch_with_retry(self, fetch_func, *args, **kwargs):
        """Execute fetch function with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await fetch_func(*args, **kwargs)
                if not result.get("error"):
                    return result
                last_exception = Exception(result["error"])
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

            if attempt < self.max_retries:
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # All attempts failed
        return {
            "error": f"All {self.max_retries + 1} attempts failed. Last error: {str(last_exception)}",
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "content_length": 0
        }

class EnhancedWebScraper:
    """
    Enhanced WebScraper with Scrapy-inspired improvements:
    - Better content extraction using CSS selectors
    - Retry middleware with exponential backoff
    - Structured data extraction
    - Improved content processing pipeline
    """

    def __init__(self, config):
        self.config = config
        self.content_extractor = ContentExtractor()
        self.retry_middleware = RetryMiddleware(
            max_retries=getattr(config, 'SCRAPER_MAX_RETRIES', 3),
            base_delay=getattr(config, 'SCRAPER_BASE_DELAY', 1),
            max_delay=getattr(config, 'SCRAPER_MAX_DELAY', 10)
        )

        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            default_headers={
                "Connection": "close"
            }
        )

        self._concurrency = getattr(config, "SCRAPER_CONCURRENCY", 3)

        # Content evaluation prompt
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze if a web page contains a curated list of restaurants or restaurant recommendations.

        VALID CONTENT (score > 0.7):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris")
        - Collections of restaurants in professional restaurant guides
        - Food critic reviews covering multiple restaurants
        - Articles in reputable media discussing various dining options in an area

        NOT VALID CONTENT (score < 0.3):
        - Official website of a single restaurant
        - Collections of restaurants in booking and delivery websites like Uber Eats, The Fork, Glovo, etc.
        - Wanderlog content
        - Individual restaurant menus
        - Single restaurant reviews
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites

        SCORING CRITERIA:
        - Multiple restaurants mentioned (essential)
        - Professional curation or expertise evident
        - Detailed descriptions of restaurants/cuisine
        - Location information for multiple restaurants
        - Price or quality indications for multiple venues

        FORMAT:
        Respond with a JSON object containing:
        {{
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "reasoning": "brief explanation of your evaluation"
        }}
        """

        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("human", "URL: {url}\n\nPage Title: {title}\n\nContent Preview:\n{preview}")
        ])
        self.eval_chain = self.eval_prompt | self.model

        self.browser_config = {
            "headless": True,
            "timeout": 30_000,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "args": ["--disable-dev-shm-usage"]
        }

        self.successful_urls, self.failed_urls = [], []
        self.invalid_content_urls = []

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point: Enhanced filtering and scraping with better content extraction
        """
        logger.info(f"Enhanced scraper processing {len(search_results)} search results")

        # Create a local semaphore for this run
        local_sem = asyncio.Semaphore(self._concurrency)

        MAX_URLS_TO_PROCESS = 15

        # Limit the number of URLs we process
        if len(search_results) > MAX_URLS_TO_PROCESS:
            logger.info(f"Limiting processing from {len(search_results)} to {MAX_URLS_TO_PROCESS} URLs")
            search_results = search_results[:MAX_URLS_TO_PROCESS]

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            start_time = time.time()
            logger.info(f"Starting enhanced processing of {len(search_results)} search results")

            # Reset trackers
            self.successful_urls = []
            self.failed_urls = []
            self.invalid_content_urls = []

            # Process results in batches
            enriched_results = []
            batch_size = 3

            # Initialize browser for Playwright if available
            if PLAYWRIGHT_ENABLED:
                try:
                    logger.info("Initializing Playwright browser for enhanced processing")
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=["--disable-dev-shm-usage"]
                        )

                        for i in range(0, len(search_results), batch_size):
                            batch = search_results[i:i+batch_size]

                            if i > 0:
                                await asyncio.sleep(2)  # Respectful delay

                            batch_tasks = [
                                self._process_search_result_enhanced(result, browser, local_sem) 
                                for result in batch
                            ]
                            batch_results = await asyncio.gather(*batch_tasks)

                            valid_results = [r for r in batch_results if r is not None]
                            enriched_results.extend(valid_results)

                            await asyncio.sleep(1)

                        await browser.close()

                except Exception as e:
                    logger.error(f"Error with Playwright: {e}. Falling back to HTTP methods.")
                    # Fallback to HTTP processing
                    enriched_results = await self._process_with_http_fallback(search_results, local_sem, batch_size)
            else:
                # HTTP-only processing
                logger.info("Enhanced scraper using HTTP methods")
                enriched_results = await self._process_with_http_fallback(search_results, local_sem, batch_size)

            elapsed = time.time() - start_time

            # Log comprehensive statistics
            logger.info(f"Enhanced scraping completed in {elapsed:.2f} seconds")
            logger.info(f"Total results: {len(search_results)}")
            logger.info(f"Successfully scraped: {len(self.successful_urls)}")
            logger.info(f"Failed to scrape: {len(self.failed_urls)}")
            logger.info(f"Filtered by content evaluator: {len(self.invalid_content_urls)}")
            logger.info(f"Final enriched results: {len(enriched_results)}")

            dump_chain_state("enhanced_scraper_results", {
                "total_results": len(search_results),
                "successful_urls": self.successful_urls,
                "failed_urls": self.failed_urls,
                "invalid_content_urls": self.invalid_content_urls,
                "final_count": len(enriched_results)
            })

            return enriched_results

    async def _process_with_http_fallback(self, search_results, local_sem, batch_size):
        """Process results using HTTP methods only"""
        enriched_results = []

        for i in range(0, len(search_results), batch_size):
            batch = search_results[i:i+batch_size]
            batch_tasks = [
                self._process_search_result_enhanced(result, None, local_sem) 
                for result in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)

            valid_results = [r for r in batch_results if r is not None]
            enriched_results.extend(valid_results)

            await asyncio.sleep(1)

        return enriched_results

    async def _process_search_result_enhanced(self, result: Dict[str, Any], browser=None, sem=None) -> Optional[Dict[str, Any]]:
        """
        Enhanced processing of a single search result
        """
        semaphore = sem if sem is not None else SEM

        async with semaphore:
            url = result.get("url")
            if not url:
                return None

            try:
                # Extract domain info
                source_domain = self._extract_domain(url)
                result["source_domain"] = source_domain

                # Fetch content with retry logic
                if PLAYWRIGHT_ENABLED and browser is not None:
                    fetch_result = await self.retry_middleware.fetch_with_retry(
                        self._fetch_with_playwright_enhanced, url, browser
                    )
                else:
                    fetch_result = await self.retry_middleware.fetch_with_retry(
                        self._fetch_with_http_enhanced, url
                    )

                if fetch_result.get("error"):
                    logger.warning(f"Failed to fetch URL after retries: {url}, Error: {fetch_result['error']}")
                    self.failed_urls.append(url)
                    return None

                # Enhanced AI evaluation with structured data
                evaluation = await self._evaluate_content_enhanced(
                    url,
                    fetch_result.get("title", ""),
                    fetch_result.get("content_preview", ""),
                    fetch_result.get("structured_data", {})
                )

                if not (evaluation.get("is_restaurant_list") and evaluation.get("content_quality", 0) > 0.5):
                    logger.info(f"Filtered URL by enhanced content evaluation: {url}")
                    self.invalid_content_urls.append(url)
                    return None

                # Enhanced content processing
                processed_content = await self._process_content_enhanced(
                    fetch_result.get("html", ""),
                    fetch_result.get("structured_data", {})
                )

                # Enhanced source information
                source_info = self._extract_source_info_enhanced(
                    url, 
                    fetch_result.get("title", ""), 
                    source_domain,
                    result.get("favicon", ""),
                    fetch_result.get("structured_data", {})
                )

                # Build enriched result
                enriched_result = {
                    **result,
                    "scraped_title": fetch_result.get("title", ""),
                    "scraped_content": processed_content,
                    "content_length": len(processed_content),
                    "quality_score": evaluation.get("content_quality", 0.0),
                    "restaurant_count": evaluation.get("restaurant_count", 0),
                    "source_info": source_info,
                    "structured_data": fetch_result.get("structured_data", {}),
                    "timestamp": time.time(),
                }

                self.successful_urls.append(url)
                logger.info(f"Successfully processed with enhanced scraper: {url}")
                return enriched_result

            except Exception as e:
                logger.error(f"Error in enhanced processing for URL {url}: {str(e)}")
                self.failed_urls.append(url)
                return None

    async def _fetch_with_playwright_enhanced(self, url: str, browser) -> Dict[str, Any]:
        """Enhanced Playwright fetching with better content extraction"""
        logger.info(f"Enhanced Playwright fetch: {url}")
        result = {
            "url": url,
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "structured_data": {},
            "error": None,
            "content_length": 0
        }

        try:
            context = await browser.new_context(
                user_agent=self.browser_config['user_agent'],
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="Europe/London",
                has_touch=False
            )

            page = await context.new_page()
            page.set_default_timeout(self.browser_config['timeout'])

            response = await page.goto(url, wait_until="domcontentloaded", timeout=self.browser_config['timeout'])
            result["status_code"] = response.status if response else 0

            # Wait for content to load
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeoutError:
                pass

            # Get basic content
            result["html"] = await page.content()
            result["title"] = await page.title()

            # Enhanced structured extraction using JavaScript
            structured_data = await page.evaluate('''() => {
                const data = {
                    restaurantNames: [],
                    addresses: [],
                    descriptions: [],
                    listItems: [],
                    sections: []
                };

                // Enhanced restaurant name extraction
                const nameSelectors = ['h1', 'h2', 'h3', 'h4', 'strong', 'b', '.restaurant-name', '.venue-name'];
                nameSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        const text = el.innerText?.trim();
                        if (text && text.length > 2 && text.length < 80) {
                            data.restaurantNames.push({
                                text: text,
                                selector: selector,
                                context: el.parentElement?.tagName || 'unknown'
                            });
                        }
                    });
                });

                // Address extraction
                document.querySelectorAll('address, .address, .location').forEach(el => {
                    const text = el.innerText?.trim();
                    if (text && text.length > 5 && text.length < 200) {
                        data.addresses.push({
                            text: text,
                            element: el.tagName.toLowerCase()
                        });
                    }
                });

                // Description extraction
                document.querySelectorAll('p, .description, .summary').forEach(el => {
                    const text = el.innerText?.trim();
                    if (text && text.length > 10 && text.length < 1000) {
                        data.descriptions.push({
                            text: text,
                            length: text.length
                        });
                    }
                });

                // List items
                document.querySelectorAll('li').forEach(el => {
                    const text = el.innerText?.trim();
                    if (text && text.length > 5 && text.length < 500) {
                        data.listItems.push({
                            text: text,
                            parent: el.parentElement?.tagName || 'unknown'
                        });
                    }
                });

                // Structured sections
                document.querySelectorAll('[class*="restaurant"], [class*="venue"], [class*="place"]').forEach(el => {
                    const content = el.innerText?.trim();
                    if (content && content.length > 10) {
                        data.sections.push({
                            classes: Array.from(el.classList),
                            content: content.substring(0, 500),
                            hasLinks: el.querySelectorAll('a').length > 0
                        });
                    }
                });

                return data;
            }''')

            result["structured_data"] = structured_data
            result["content_preview"] = json.dumps(structured_data)[:2000]
            result["content_length"] = len(result["html"])

            await context.close()

        except PlaywrightTimeoutError:
            result["error"] = "Timeout error in enhanced fetch"
        except Exception as e:
            result["error"] = f"Enhanced fetch error: {str(e)}"

        return result

    async def _fetch_with_http_enhanced(self, url: str) -> Dict[str, Any]:
        """Enhanced HTTP fetching with structured extraction"""
        logger.info(f"Enhanced HTTP fetch: {url}")
        result = {
            "url": url,
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "structured_data": {},
            "error": None,
            "content_length": 0
        }

        try:
            # Use aiohttp if available
            try:
                import aiohttp

                headers = {
                    "User-Agent": self.browser_config['user_agent'],
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "close"
                }

                timeout = aiohttp.ClientTimeout(total=30)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers) as response:
                        result["status_code"] = response.status

                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            result["html"] = html
                            result["title"] = soup.title.text if soup.title else ""

                            # Enhanced structured extraction
                            structured_data = self.content_extractor.extract_structured_content(soup, url)
                            result["structured_data"] = structured_data
                            result["content_preview"] = json.dumps({
                                "restaurant_names": structured_data["restaurant_names"][:5],
                                "descriptions_count": len(structured_data["descriptions"]),
                                "list_items_count": len(structured_data["list_items"])
                            })

                            result["content_length"] = len(html)
                            return result
                        else:
                            result["error"] = f"HTTP error: {response.status}"
                            return result

            except ImportError:
                # Fallback to requests
                import requests

                headers = {
                    "User-Agent": self.browser_config['user_agent'],
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Connection": "close"
                }

                response = requests.get(url, headers=headers, timeout=15)
                result["status_code"] = response.status_code

                if response.status_code == 200:
                    html = response.text
                    soup = BeautifulSoup(html, 'html.parser')

                    result["html"] = html
                    result["title"] = soup.title.text if soup.title else ""

                    # Enhanced structured extraction
                    structured_data = self.content_extractor.extract_structured_content(soup, url)
                    result["structured_data"] = structured_data
                    result["content_preview"] = json.dumps({
                        "restaurant_names": structured_data["restaurant_names"][:5],
                        "descriptions_count": len(structured_data["descriptions"]),
                        "list_items_count": len(structured_data["list_items"])
                    })

                    result["content_length"] = len(html)
                else:
                    result["error"] = f"HTTP error: {response.status_code}"

        except Exception as e:
            result["error"] = f"Enhanced HTTP fetch error: {str(e)}"

        return result

    async def _evaluate_content_enhanced(self, url: str, title: str, preview: str, structured_data: Dict) -> Dict[str, Any]:
        """Enhanced content evaluation using structured data"""
        try:
            # Quick structured data check
            restaurant_count = len(structured_data.get("restaurant_names", []))
            descriptions_count = len(structured_data.get("descriptions", []))

            # If we have good structured data, use it for evaluation
            if restaurant_count > 1 and descriptions_count > 2:
                evaluation_text = f"""
                Structured data analysis:
                - Restaurant names found: {restaurant_count}
                - Descriptions found: {descriptions_count}
                - List items: {len(structured_data.get("list_items", []))}
                - Sections: {len(structured_data.get("structured_sections", []))}

                Sample names: {[item["text"] for item in structured_data.get("restaurant_names", [])[:3]]}
                """
            else:
                evaluation_text = f"Title: {title}\nPreview: {preview[:1000]}"

            response = await self.eval_chain.ainvoke({
                "url": url,
                "title": title,
                "preview": evaluation_text
            })

            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content.strip())

            # Enhance evaluation with structured data insights
            if restaurant_count > 3:
                evaluation["restaurant_count"] = max(evaluation.get("restaurant_count", 0), restaurant_count)
                evaluation["content_quality"] = max(evaluation.get("content_quality", 0), 0.8)

            if "content_quality" not in evaluation:
                evaluation["content_quality"] = 0.8 if evaluation.get("is_restaurant_list", False) else 0.2

            return evaluation

        except Exception as e:
            logger.error(f"Enhanced evaluation error for {url}: {str(e)}")
            return {
                "is_restaurant_list": False,
                "restaurant_count": 0,
                "content_quality": 0.0,
                "reasoning": f"Enhanced evaluation error: {str(e)}"
            }

    async def _process_content_enhanced(self, html: str, structured_data: Dict) -> str:
        """Enhanced content processing using structured data"""
        paragraphs = []

        # Use structured data if available
        if structured_data:
            # Add title
            title = structured_data.get("title", "")
            if title:
                paragraphs.append(f"TITLE: {title}")

            # Add restaurant names section
            restaurant_names = structured_data.get("restaurant_names", [])
            if restaurant_names:
                paragraphs.append("RESTAURANT NAMES:")
                for name_data in restaurant_names[:10]:  # Limit to avoid noise
                    text = name_data.get("text", "")
                    selector = name_data.get("selector", "unknown")
                    context = name_data.get("context", "")
                    paragraphs.append(f"  {selector.upper()}: {text} (context: {context})")
                paragraphs.append("")

            # Add addresses
            addresses = structured_data.get("addresses", [])
            if addresses:
                paragraphs.append("ADDRESSES:")
                for addr_data in addresses[:5]:  # Limit addresses
                    text = addr_data.get("text", "")
                    paragraphs.append(f"  ADDRESS: {text}")
                paragraphs.append("")

            # Add structured sections
            sections = structured_data.get("structured_sections", [])
            if sections:
                paragraphs.append("RESTAURANT SECTIONS:")
                for section in sections[:5]:  # Limit sections
                    classes = section.get("class", [])
                    content = section.get("content", "")
                    paragraphs.append(f"  SECTION ({', '.join(classes)}): {content[:200]}...")
                paragraphs.append("")

            # Add descriptions
            descriptions = structured_data.get("descriptions", [])
            if descriptions:
                paragraphs.append("CONTENT DESCRIPTIONS:")
                for desc_data in descriptions[:8]:  # Limit descriptions
                    text = desc_data.get("text", "")
                    selector = desc_data.get("selector", "paragraph")
                    length = desc_data.get("length", 0)
                    if length > 50:  # Only include substantial descriptions
                        paragraphs.append(f"  {selector.upper()}: {text}")
                paragraphs.append("")

            # Add list items
            list_items = structured_data.get("list_items", [])
            if list_items:
                paragraphs.append("LIST ITEMS:")
                for item_data in list_items[:10]:  # Limit list items
                    text = item_data.get("text", "")
                    selector = item_data.get("selector", "list_item")
                    paragraphs.append(f"  {selector.upper()}: {text}")
                paragraphs.append("")

        # Fallback to HTML processing if no structured data
        if not paragraphs and html:
            try:
                # Use readability for main content
                doc = Document(html)
                readable_html = doc.summary()
                soup = BeautifulSoup(readable_html, 'html.parser')

                # Extract using CSS selectors
                structured_extraction = self.content_extractor.extract_structured_content(soup, "")

                # Process the fallback extraction
                if structured_extraction.get("restaurant_names"):
                    paragraphs.append("RESTAURANT NAMES (from HTML):")
                    for name_data in structured_extraction["restaurant_names"][:8]:
                        text = name_data.get("text", "")
                        selector = name_data.get("selector", "unknown")
                        paragraphs.append(f"  {selector.upper()}: {text}")
                    paragraphs.append("")

                # Add main content paragraphs
                for p in soup.find_all('p'):
                    text = p.get_text().strip()
                    if text and 20 < len(text) < 800:  # Reasonable paragraph length
                        paragraphs.append(f"PARAGRAPH: {text}")

                # Add list items
                for li in soup.find_all('li'):
                    text = li.get_text().strip()
                    if text and 10 < len(text) < 300:
                        paragraphs.append(f"LIST_ITEM: {text}")

            except Exception as e:
                logger.error(f"Error in fallback HTML processing: {str(e)}")
                # Last resort - simple text extraction
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    simple_text = soup.get_text(separator='\n\n')
                    paragraphs.append(f"SIMPLE_EXTRACTION: {simple_text[:2000]}...")

        # Join all paragraphs
        processed_text = "\n\n".join(p for p in paragraphs if p)

        # Clean up extra whitespace
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
        processed_text = re.sub(r' {2,}', ' ', processed_text)

        return processed_text

    def _extract_source_info_enhanced(self, url: str, title: str, domain: str, favicon: str = "", structured_data: Dict = None) -> Dict[str, Any]:
        """Enhanced source information extraction"""
        source_type = "Website"

        # Enhanced source type detection
        domain_lower = domain.lower()
        title_lower = title.lower() if title else ""

        # Check structured data for additional context
        restaurant_count = 0
        if structured_data:
            restaurant_count = len(structured_data.get("restaurant_names", []))

        # Determine source type based on multiple factors
        if any(kw in domain_lower for kw in ["guide", "michelin", "zagat"]):
            source_type = "Restaurant Guide"
        elif any(kw in domain_lower for kw in ["news", "times", "post", "magazine", "telegraph", "guardian"]):
            source_type = "News Publication"
        elif any(kw in domain_lower for kw in ["blog", "food", "critic", "review", "eater"]):
            source_type = "Food Blog"
        elif any(kw in domain_lower for kw in ["timeout", "travel", "visit"]):
            source_type = "Travel Guide"
        elif restaurant_count > 5:
            source_type = "Restaurant Directory"

        # Enhanced source name extraction
        source_name = domain.split('.')[0].capitalize()
        if domain.startswith("www."):
            source_name = domain[4:].split('.')[0].capitalize()

        # Enhanced mapping for known sources
        source_mapping = {
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
            'foodandwine': 'Food & Wine',
            'bonappetit': 'Bon Appétit',
            'saveur': 'Saveur'
        }

        for key, val in source_mapping.items():
            if key in domain_lower:
                source_name = val
                break

        # Try to extract better name from title
        if title and '|' in title:
            title_parts = title.split('|')
            if len(title_parts) > 1:
                possible_name = title_parts[-1].strip()
                if 3 < len(possible_name) < 30:
                    source_name = possible_name

        return {
            "name": source_name,
            "domain": domain,
            "type": source_type,
            "favicon": favicon,
            "restaurant_count": restaurant_count,
            "credibility_score": self._calculate_credibility_score(domain, source_type, restaurant_count)
        }

    def _calculate_credibility_score(self, domain: str, source_type: str, restaurant_count: int) -> float:
        """Calculate a credibility score for the source"""
        score = 0.5  # Base score

        # Bonus for recognized domains
        high_credibility_domains = [
            'michelin', 'eater', 'nytimes', 'timeout', 'theguardian', 
            'telegraph', 'foodandwine', 'bonappetit', 'saveur', 'worldofmouth'
        ]

        if any(domain_part in domain.lower() for domain_part in high_credibility_domains):
            score += 0.3

        # Bonus for source type
        type_bonuses = {
            "Restaurant Guide": 0.3,
            "News Publication": 0.2,
            "Food Blog": 0.1,
            "Travel Guide": 0.15
        }
        score += type_bonuses.get(source_type, 0)

        # Bonus for restaurant count (indicates comprehensive coverage)
        if restaurant_count > 5:
            score += 0.1
        elif restaurant_count > 10:
            score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url

    # Legacy compatibility methods
    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Legacy API method for backward compatibility"""
        logger.info("Using legacy API - consider migrating to filter_and_scrape_results")
        return await self.filter_and_scrape_results(search_results)

    async def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch a URL for testing purposes"""
        if PLAYWRIGHT_ENABLED:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        headless=True, 
                        args=["--disable-dev-shm-usage"]
                    )
                    result = await self._fetch_with_playwright_enhanced(url, browser)
                    await browser.close()
                    return result
            except Exception as e:
                logger.error(f"Error using Playwright for testing: {e}")
                return await self._fetch_with_http_enhanced(url)
        else:
            return await self._fetch_with_http_enhanced(url)