# agents/scraper.py
import httpx
import asyncio
import time
import json
import re
import logging
import random
from collections import deque
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

# Constants
CACHE_EXPIRE_SECONDS = 60 * 60 * 24 * 7   # 7 days
MIN_TOKEN_KEEP = 50                       # Lower threshold
DOMAINS_ALWAYS_KEEP = ("michelin.", "resy.", "worlds50", "theinfatuation.", "guide.", "culinarybackstreets.")
RESTAURANT_KEYWORDS = ("restaurant", "chef", "menu", "dish", "food", "dining", "eat", "cuisine", "bistro", "cafe", "bar")

class WebScraper:
    def __init__(self, config):
        self.config = config
        self.timeout = 15  # seconds
        self.max_html_cache = 100_000  # Increased from 10,000 to 100,000 bytes

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
            "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1 Mobile/15E148 Safari/604.1",
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

        # Use proxy for retries if configured
        self.use_proxy = getattr(config, 'USE_PROXY_FOR_RETRY', False)
        self.proxy_url = getattr(config, 'PROXY_URL', None)

    @asynccontextmanager
    async def get_client(self, use_proxy=False):
        """Context manager for httpx client with appropriate settings"""
        try:
            kwargs = {
                "timeout": self.timeout,
                "follow_redirects": True,
                "http2": True,
            }

            # Add proxy if configured and requested
            if use_proxy and self.proxy_url:
                kwargs["proxies"] = {"all://": self.proxy_url}

            async with httpx.AsyncClient(**kwargs) as client:
                yield client
        except Exception as e:
            logger.error(f"Error creating httpx client: {e}")
            # Create a basic client as fallback
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    async def _quick_preview(self, client, url):
        """Return the first 300 chars of visible text (for ranking before full scrape)."""
        try:
            headers = self.default_headers.copy()
            headers["User-Agent"] = random.choice(self.user_agents)

            r = await client.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(r.text, "lxml")
            visible = soup.get_text(" ", strip=True)
            return visible[:300]
        except Exception as e:
            logger.error(f"Error in quick preview for {url}: {e}")
            return ""

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
        domain = urlparse(url).netloc.lower()
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
        headers = self.default_headers.copy()
        headers["User-Agent"] = random.choice(self.user_agents)

        error = None
        status_code = None
        html = ""
        backoff = 1

        # First try without proxy
        async with self.get_client() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(url, headers=headers, timeout=self.timeout)
                    status_code = response.status_code

                    if status_code == 200:
                        html = response.text
                        break
                    else:
                        logger.warning(f"HTTP error {status_code} for {url}")
                        error = f"HTTP error {status_code}"
                except httpx.TimeoutException:
                    error = "Timeout"
                    logger.warning(f"Timeout for {url}")
                except Exception as e:
                    error = str(e)
                    logger.error(f"Error fetching {url}: {e}")

                # If we still need to retry, use exponential backoff
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2  # exponential backoff

        # If failed with regular client, try with proxy if configured
        if not html and self.use_proxy and self.proxy_url:
            logger.info(f"Retrying {url} with proxy")
            async with self.get_client(use_proxy=True) as proxy_client:
                try:
                    response = await proxy_client.get(url, headers=headers, timeout=self.timeout)
                    status_code = response.status_code

                    if status_code == 200:
                        html = response.text
                        error = None
                    else:
                        error = f"HTTP error {status_code} (proxy)"
                except Exception as e:
                    error = f"Proxy error: {str(e)}"

        # Extract and clean content
        scraped_content = ""
        quality_score = 0.0

        if html:
            try:
                # Extract text using our improved method
                scraped_content = self._extract_article_text(html)

                # Score the content
                quality_score = self._score(scraped_content, domain)

            except Exception as e:
                logger.error(f"Error parsing content from {url}: {e}")
                error = f"Parsing error: {str(e)}"

        # Prepare result
        result = {
            "url": url,
            "html": html[:self.max_html_cache] if html else "",  # Limit stored HTML size
            "scraped_content": scraped_content,
            "status_code": status_code,
            "error": error,
            "quality_score": quality_score,
            "scraped_at": time.time()
        }

        # Store in cache (only success cases)
        if scraped_content and not error:
            cache.set(cache_key, result, expire=CACHE_EXPIRE_SECONDS)

        return result

    def _extract_article_text(self, html):
        """
        Extract and clean article text with improved listicle handling
        """
        try:
            # Use readability to locate the main content area
            doc = Document(html)
            node = BeautifulSoup(doc.summary(), "lxml")  # Get the core candidate

            # Walk siblings up to a reasonable limit, greedy merge if they
            # contain restaurant words to avoid listicle truncation
            queue = deque(node.find_all(recursive=False))
            text_parts = []

            seen_elements = set()  # Track elements we've already processed

            while queue and len(queue) < 500:  # Safety limit
                el = queue.popleft()

                # Skip if already processed
                if id(el) in seen_elements:
                    continue
                seen_elements.add(id(el))

                # Skip script and style tags
                if el.name in ("script", "style"):
                    continue

                # Process the content
                txt = el.get_text(" ", strip=True)
                if txt:
                    # If it contains restaurant keywords or is part of a list, keep it
                    if any(kw in txt.lower() for kw in RESTAURANT_KEYWORDS) or el.name == "li":
                        text_parts.append(txt)

                        # Check next sibling to pick up lists and related content
                        if el.next_sibling:
                            queue.append(el.next_sibling)

                        # For lists, add all list items
                        if el.name in ("ul", "ol"):
                            for li in el.find_all("li", recursive=False):
                                if id(li) not in seen_elements:
                                    queue.append(li)

                        # For div/section containers, check their children
                        if el.name in ("div", "section", "article"):
                            for child in el.find_all(recursive=False):
                                if id(child) not in seen_elements:
                                    queue.append(child)

            # Clean and join text
            cleaned = re.sub(r"\s+", " ", " ".join(text_parts)).strip()

            # Add structured data if available
            structured_data = self._extract_structured_data(html)
            if structured_data:
                cleaned += "\n\nSTRUCTURED DATA: " + json.dumps(structured_data)

            return cleaned

        except Exception as e:
            logger.error(f"Error in _extract_article_text: {e}")
            # Try a simpler fallback method
            try:
                soup = BeautifulSoup(html, "lxml")
                # Just remove script and style tags
                for tag in soup(["script", "style"]):
                    tag.decompose()

                text = soup.get_text(" ", strip=True)
                return re.sub(r"\s+", " ", text).strip()
            except:
                return ""

    def _score(self, text, domain):
        """
        Score the relevance and quality of the extracted content
        """
        if not text:
            return 0.0

        # Calculate tokens
        tokens = len(self.tokenizer.encode(text)) if self.tokenizer else len(text) // 4

        # Check for priority domains
        is_priority_domain = any(d in domain for d in DOMAINS_ALWAYS_KEEP)

        # Filter out short content unless it's from a priority domain
        if tokens < MIN_TOKEN_KEEP and not is_priority_domain:
            return 0.0

        # Check for restaurant-related keywords
        has_keywords = any(kw in text.lower() for kw in RESTAURANT_KEYWORDS)

        # Add a small boost for priority domains
        domain_boost = 0.2 if is_priority_domain else 0.0

        # Calculate score based on length and restaurant relevance
        base_score = min(1.0, (tokens / 7_500) * 0.6 + (1 if has_keywords else 0) * 0.4)

        # Add domain boost but cap at 1.0
        return min(1.0, base_score + domain_boost)

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

        if not pre_filtered:
            return []

        # Process domain validations first
        domain_validations = {}
        validation_tasks = []

        # Get unique domains
        unique_domains = {urlparse(result.get("url", "")).netloc for result in pre_filtered}

        # Create validation tasks
        validation_sem = asyncio.Semaphore(10)  # Limit concurrent validations

        async def validate_domain_with_semaphore(domain):
            async with validation_sem:
                try:
                    is_valid = await validate_source(domain, self.config)
                    return domain, is_valid
                except Exception as e:
                    logger.error(f"Error validating domain {domain}: {e}")
                    # Default to accepting on error
                    return domain, True

        for domain in unique_domains:
            validation_tasks.append(validate_domain_with_semaphore(domain))

        # Run validations and store results
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        for result in validation_results:
            if isinstance(result, tuple) and len(result) == 2:
                domain, is_valid = result
                domain_validations[domain] = is_valid

        # Now create scraping tasks for valid domains
        scraping_tasks = []
        scrape_sem = asyncio.Semaphore(5)  # Limit concurrent scrapes

        async def scrape_with_semaphore(result):
            url = result.get("url", "")
            domain = urlparse(url).netloc

            # Skip invalid domains
            if domain in domain_validations and not domain_validations[domain]:
                return None

            # Set domain reputation
            result["domain_reputation"] = 1.0 if domain_validations.get(domain, True) else 0.0

            # Scrape with semaphore
            async with scrape_sem:
                try:
                    scraped = await self.scrape_url(url)
                    if scraped.get("scraped_content") and scraped.get("quality_score", 0) > 0.2:
                        return {**result, **scraped}
                    return None
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    return None

        # Create tasks for all results
        tasks = [scrape_with_semaphore(result) for result in pre_filtered]

        # Process all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and None values
        for result in results:
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