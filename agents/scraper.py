# agents/scraper.py
import httpx
import asyncio
import time
import json
import re
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_core.tracers.context import tracing_v2_enabled
from diskcache import Cache
import tiktoken
from readability import Document
from contextlib import asynccontextmanager, suppress
from utils.source_validator import validate_source

# Set up logging
logger = logging.getLogger(__name__)

# Initialize cache
cache = Cache("./http_cache")

class WebScraper:
    def __init__(self, config):
        self.config = config
        self.timeout = 15  # seconds
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        ]

        # Default headers
        self.default_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "DNT": "1"
        }

        # Excluded domains (in addition to config)
        self.excluded_domains = set(config.EXCLUDED_RESTAURANT_SOURCES + [
            "pinterest.com", 
            "twitter.com", 
            "facebook.com", 
            "instagram.com",
            "youtube.com"
        ])

    @asynccontextmanager
    async def get_client(self):
        """Context manager for httpx client with appropriate settings"""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                http2=True,
            ) as client:
                yield client
        except Exception as e:
            logger.error(f"Error creating httpx client: {e}")
            # Create a basic client as fallback
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    async def scrape_url(self, url, max_retries=2):
        """
        Scrape content from a URL with retries and caching

        Args:
            url (str): URL to scrape
            max_retries (int): Maximum number of retry attempts

        Returns:
            dict: Scraped content and metadata
        """
        # Normalize URL
        url = url.strip()

        # Check cache first
        cache_key = f"url:{url}"
        cached_response = cache.get(cache_key)
        if cached_response:
            return cached_response

        # Parse domain to check exclusions
        domain = urlparse(url).netloc
        if any(excluded in domain for excluded in self.excluded_domains):
            return {
                "url": url,
                "html": "",
                "scraped_content": "",
                "status_code": None,
                "error": "Excluded domain",
                "quality_score": 0.0
            }

        # Select random user agent
        import random
        headers = self.default_headers.copy()
        headers["User-Agent"] = random.choice(self.user_agents)

        error = None
        status_code = None
        html = ""

        for attempt in range(max_retries):
            try:
                async with self.get_client() as client:
                    response = await client.get(url, headers=headers, timeout=self.timeout)
                    status_code = response.status_code

                    if status_code == 200:
                        html = response.text
                        break
                    else:
                        logger.warning(f"HTTP error {status_code} for {url}")
                        error = f"HTTP error {status_code}"
                        # Add delay between retries
                        await asyncio.sleep(1)
            except httpx.TimeoutException:
                error = "Timeout"
                logger.warning(f"Timeout for {url}")
                await asyncio.sleep(1)
            except Exception as e:
                error = str(e)
                logger.error(f"Error fetching {url}: {e}")
                await asyncio.sleep(1)

        # Extract and clean content
        scraped_content = ""
        quality_score = 0.0

        if html:
            try:
                # Use readability to extract main content
                doc = Document(html)
                title = doc.title()
                content = doc.summary()

                # Parse with BeautifulSoup to clean further
                soup = BeautifulSoup(content, "lxml")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                # Get text and clean it
                text = soup.get_text(separator=" ", strip=True)

                # Clean up whitespace
                scraped_content = re.sub(r'\s+', ' ', text).strip()

                # Calculate quality score based on content length and restaurant indicators
                content_length = len(scraped_content)
                if self.tokenizer:
                    tokens = len(self.tokenizer.encode(scraped_content))
                else:
                    tokens = content_length // 4  # Rough estimate

                has_restaurant_indicators = any(word in scraped_content.lower() 
                                               for word in ["restaurant", "dining", "chef", 
                                                           "menu", "dish", "food"])

                quality_score = min(1.0, (content_length / 5000) * 0.7 + (1 if has_restaurant_indicators else 0) * 0.3)

                # Add structured data if available
                structured_data = self._extract_structured_data(html)
                if structured_data:
                    scraped_content += "\n\nSTRUCTURED DATA: " + json.dumps(structured_data)

            except Exception as e:
                logger.error(f"Error parsing content from {url}: {e}")
                error = f"Parsing error: {str(e)}"

        # Prepare result
        result = {
            "url": url,
            "html": html[:10000] if html else "",  # Limit stored HTML to avoid huge objects
            "scraped_content": scraped_content,
            "status_code": status_code,
            "error": error,
            "quality_score": quality_score,
            "scraped_at": time.time()
        }

        # Store in cache (only success cases)
        if scraped_content and not error:
            cache.set(cache_key, result, expire=86400 * 7)  # Cache for 7 days

        return result

    def _extract_structured_data(self, html):
        """Extract structured data (JSON-LD, schema.org) from HTML"""
        structured_data = {}

        try:
            soup = BeautifulSoup(html, "lxml")

            # Look for JSON-LD
            for script in soup.find_all("script", type="application/ld+json"):
                with suppress(Exception):
                    data = json.loads(script.string)
                    if data:
                        if isinstance(data, list):
                            for item in data:
                                if item.get("@type") in ["Restaurant", "LocalBusiness", "FoodEstablishment"]:
                                    structured_data = item
                                    break
                        elif data.get("@type") in ["Restaurant", "LocalBusiness", "FoodEstablishment"]:
                            structured_data = data
                            break
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")

        return structured_data

    async def scrape_search_results(self, search_results):
        """
        Scrape content from search results

        Args:
            search_results (list): List of search result dictionaries

        Returns:
            list: Search results with scraped content
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Validate and filter search results
            filtered_results = await self.filter_and_scrape_results(search_results)
            return filtered_results

    async def filter_and_scrape_results(self, search_results):
        """
        Filter and scrape search results with domain validation

        Args:
            search_results (list): List of search result dictionaries

        Returns:
            list: Filtered and scraped search results
        """
        filtered_results = []

        # First pass: quick filter before scraping
        pre_filtered = []
        seen_urls = set()

        for result in search_results:
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            # Check domain against exclusion list
            domain = urlparse(url).netloc
            if any(excluded in domain for excluded in self.excluded_domains):
                continue

            # Check title and description for relevance
            title = result.get("title", "").lower()
            description = result.get("description", "").lower()

            # Skip obvious non-restaurant results
            if any(word in title for word in ["login", "sign up", "privacy policy", "terms of service"]):
                continue

            pre_filtered.append(result)

        # Second pass: validate domains and gather scraping tasks
        scraping_tasks = []
        domain_validations = {}

        # First, run all domain validations (with a limit to avoid overloading)
        tasks = []
        domains_to_validate = set()

        for result in pre_filtered:
            url = result.get("url", "")
            domain = urlparse(url).netloc
            domains_to_validate.add(domain)

        # Limit to 10 concurrent domain validations
        semaphore = asyncio.Semaphore(10)

        async def validate_with_semaphore(domain):
            async with semaphore:
                try:
                    return domain, await validate_source(domain, self.config)
                except Exception as e:
                    logger.error(f"Error validating domain {domain}: {e}")
                    return domain, True  # Default to accepting on error

        # Run all domain validations concurrently
        validation_tasks = [validate_with_semaphore(domain) for domain in domains_to_validate]
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Process validation results
        for result in validation_results:
            if isinstance(result, Exception):
                logger.error(f"Validation error: {result}")
                continue

            domain, is_valid = result
            domain_validations[domain] = is_valid

        # Now create scraping tasks for valid domains
        for result in pre_filtered:
            url = result.get("url", "")
            domain = urlparse(url).netloc

            # Skip invalid domains
            if domain in domain_validations and not domain_validations[domain]:
                continue

            # Set reputation on result
            result["domain_reputation"] = 1.0 if domain_validations.get(domain, True) else 0.0

            # Add to scraping tasks
            scraping_tasks.append((result, self.scrape_url(url)))

        # Execute scraping tasks with a limit
        scrape_semaphore = asyncio.Semaphore(5)

        async def process_scrape_task(task_tuple):
            result, scrape_task = task_tuple
            async with scrape_semaphore:
                try:
                    scraped = await scrape_task
                    if scraped.get("scraped_content") and scraped.get("quality_score", 0) > 0.2:
                        return {**result, **scraped}
                    return None
                except Exception as e:
                    logger.error(f"Error scraping {result.get('url')}: {e}")
                    return None

        # Process all scraping tasks safely
        processing_tasks = [process_scrape_task(task) for task in scraping_tasks]
        processing_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

        # Filter out errors and None values
        for result in processing_results:
            if result is not None and not isinstance(result, Exception):
                filtered_results.append(result)

        # Sort by quality score
        filtered_results.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        return filtered_results

    async def fetch_url(self, url):
        """
        Simple method to fetch a URL directly (for testing purposes)

        Args:
            url (str): URL to fetch

        Returns:
            dict: Response data
        """
        try:
            async with self.get_client() as client:
                headers = self.default_headers.copy()
                headers["User-Agent"] = self.user_agents[0]

                response = await client.get(url, headers=headers, timeout=self.timeout)

                return {
                    "url": url,
                    "status_code": response.status_code,
                    "content_length": len(response.text),
                    "content_preview": response.text[:500] + "..." if response.text else ""
                }
        except Exception as e:
            return {
                "url": url,
                "error": str(e)
            }