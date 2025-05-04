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
        tasks = []
        validated_domains = set()

        for result in pre_filtered:
            url = result.get("url", "")
            domain = urlparse(url).netloc

            # Only validate domain once
            if domain not in validated_domains:
                reputation_score = await validate_source(domain, self.config)
                validated_domains.add(domain)
                result["domain_reputation"] = 1.0 if reputation_score else 0.0

                # Skip low reputation sources
                if not reputation_score:
                    continue

            # Create scraping task
            tasks.append(self.scrape_url(url))

        # Execute scraping tasks concurrently (with rate limiting)
        # Use semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(5)

        async def scrape_with_semaphore(url_task):
            async with semaphore:
                return await url_task

        scraped_results = await asyncio.gather(*[scrape_with_semaphore(task) for task in tasks])

        # Merge scraped content with original results
        for i, result in enumerate(pre_filtered[:len(scraped_results)]):
            if i < len(scraped_results):  # Safety check
                scraped = scraped_results[i]

                # Only include results with actual content
                if scraped.get("scraped_content") and scraped.get("quality_score", 0) > 0.2:
                    merged_result = {**result, **scraped}
                    filtered_results.append(merged_result)

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