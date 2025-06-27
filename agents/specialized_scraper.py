# agents/specialized_scraper.py
import asyncio
import logging
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional, Callable
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SpecializedExtractionResult:
    """Result from specialized extraction"""
    articles: List[Dict[str, Any]]
    source_url: str
    source_name: str
    extraction_method: str
    success: bool
    error_message: Optional[str] = None

class SpecializedHandler(ABC):
    """Abstract base class for specialized handlers"""

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this handler can process the given URL"""
        pass

    @abstractmethod
    async def process_url(self, result: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process the URL and return enriched result"""
        pass

    @abstractmethod
    def get_handler_name(self) -> str:
        """Get the name of this handler"""
        pass

class EaterHandler(SpecializedHandler):
    """Handler for Eater.com using RSS feeds"""

    def can_handle(self, url: str) -> bool:
        return 'eater.com' in url.lower()

    def get_handler_name(self) -> str:
        return "Eater RSS Handler"

    async def process_url(self, result: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process Eater URLs using RSS feeds"""
        url = result.get('url', '')

        try:
            # Extract the Eater subdomain (e.g., ny.eater.com, london.eater.com)
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Try to find the appropriate RSS feed
            rss_url = f"https://{domain}/rss/index.xml"

            # If it's just eater.com, try common city variants
            if domain == 'eater.com':
                # Extract city from URL path if possible
                path_parts = parsed_url.path.strip('/').split('/')
                if path_parts and len(path_parts[0]) == 2:  # City code like 'ny', 'la'
                    rss_url = f"https://{path_parts[0]}.eater.com/rss/index.xml"
                else:
                    # Default to main Eater RSS
                    rss_url = "https://www.eater.com/rss/index.xml"

            logger.info(f"Attempting to fetch Eater RSS: {rss_url}")

            # Fetch and parse RSS feed
            extraction_result = await self._fetch_and_parse_rss(rss_url, url, session)

            if extraction_result.success:
                # Enrich the original result
                enriched_result = result.copy()
                enriched_result.update({
                    "scraped_content": self._format_articles_for_analyzer(extraction_result.articles),
                    "articles_data": extraction_result.articles,
                    "source_info": {
                        "name": extraction_result.source_name,
                        "url": url,
                        "rss_url": rss_url,
                        "extraction_method": extraction_result.extraction_method
                    },
                    "scraping_success": True,
                    "article_count": len(extraction_result.articles),
                    "specialized_scraping": True,
                    "handler_used": self.get_handler_name()
                })

                logger.info(f"Successfully extracted {len(extraction_result.articles)} articles from Eater RSS")
                return enriched_result
            else:
                raise Exception(extraction_result.error_message)

        except Exception as e:
            logger.error(f"Failed to process Eater URL {url}: {e}")

            result["scraping_failed"] = True
            result["scraping_error"] = f"Eater RSS processing failed: {str(e)}"
            result["specialized_scraping"] = True
            result["handler_used"] = self.get_handler_name()
            return result

    async def _fetch_and_parse_rss(self, rss_url: str, original_url: str, session: aiohttp.ClientSession) -> SpecializedExtractionResult:
        """Fetch and parse RSS feed to extract restaurant articles"""
        try:
            async with session.get(rss_url) as response:
                if response.status != 200:
                    raise Exception(f"RSS fetch failed with status {response.status}")

                rss_content = await response.text()

            # Parse RSS XML
            root = ET.fromstring(rss_content)

            # Find all items in the RSS feed
            items = root.findall('.//item')
            articles = []

            # Get recent articles (last 30 days) that might be restaurant-related
            cutoff_date = datetime.now() - timedelta(days=30)

            for item in items:
                try:
                    title = item.find('title').text if item.find('title') is not None else ""
                    link = item.find('link').text if item.find('link') is not None else ""
                    description = item.find('description').text if item.find('description') is not None else ""
                    pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""

                    # Check if this article is restaurant-related
                    if self._is_restaurant_related(title, description):
                        article = {
                            "title": title,
                            "url": link,
                            "description": description,
                            "published_date": pub_date,
                            "source_url": original_url
                        }
                        articles.append(article)

                        # Limit to prevent too many results
                        if len(articles) >= 20:
                            break

                except Exception as e:
                    logger.warning(f"Error parsing RSS item: {e}")
                    continue

            return SpecializedExtractionResult(
                articles=articles,
                source_url=original_url,
                source_name="Eater",
                extraction_method="rss_feed",
                success=True
            )

        except Exception as e:
            return SpecializedExtractionResult(
                articles=[],
                source_url=original_url,
                source_name="Eater",
                extraction_method="rss_feed",
                success=False,
                error_message=str(e)
            )

    def _is_restaurant_related(self, title: str, description: str) -> bool:
        """Check if an article is restaurant-related"""
        text = (title + " " + description).lower()

        restaurant_keywords = [
            'restaurant', 'cafe', 'bar', 'bistro', 'diner', 'eatery',
            'food', 'dining', 'menu', 'chef', 'kitchen', 'cuisine',
            'eat', 'drink', 'cocktail', 'wine', 'beer', 'coffee',
            'pizza', 'burger', 'sushi', 'italian', 'french', 'chinese',
            'best places', 'where to eat', 'new opening', 'takeaway',
            'delivery', 'brunch', 'lunch', 'dinner', 'breakfast'
        ]

        return any(keyword in text for keyword in restaurant_keywords)

    def _format_articles_for_analyzer(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles for the list analyzer"""
        if not articles:
            return ""

        formatted_parts = []
        for i, article in enumerate(articles, 1):
            parts = [f"Article {i}: {article.get('title', 'Unknown')}"]

            if article.get('description'):
                parts.append(f"Description: {article['description']}")

            if article.get('url'):
                parts.append(f"URL: {article['url']}")

            if article.get('published_date'):
                parts.append(f"Published: {article['published_date']}")

            formatted_parts.append("\n".join(parts))

        return "\n\n".join(formatted_parts)

class TimeoutHandler(SpecializedHandler):
    """Handler for Timeout.com using sitemaps and RSS"""

    def can_handle(self, url: str) -> bool:
        return 'timeout.com' in url.lower()

    def get_handler_name(self) -> str:
        return "Timeout Sitemap Handler"

    async def process_url(self, result: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process Timeout URLs using sitemaps and RSS when available"""
        url = result.get('url', '')

        try:
            # Extract the Timeout domain (e.g., timeout.com, timeout.pt)
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Try different approaches for Timeout
            sitemap_url = f"https://{domain}/sitemap.xml"

            logger.info(f"Attempting to fetch Timeout sitemap: {sitemap_url}")

            # First try sitemap approach
            extraction_result = await self._fetch_and_parse_sitemap(sitemap_url, url, session)

            if extraction_result.success:
                # Enrich the original result
                enriched_result = result.copy()
                enriched_result.update({
                    "scraped_content": self._format_articles_for_analyzer(extraction_result.articles),
                    "articles_data": extraction_result.articles,
                    "source_info": {
                        "name": extraction_result.source_name,
                        "url": url,
                        "sitemap_url": sitemap_url,
                        "extraction_method": extraction_result.extraction_method
                    },
                    "scraping_success": True,
                    "article_count": len(extraction_result.articles),
                    "specialized_scraping": True,
                    "handler_used": self.get_handler_name()
                })

                logger.info(f"Successfully extracted {len(extraction_result.articles)} articles from Timeout sitemap")
                return enriched_result
            else:
                raise Exception(extraction_result.error_message)

        except Exception as e:
            logger.error(f"Failed to process Timeout URL {url}: {e}")

            result["scraping_failed"] = True
            result["scraping_error"] = f"Timeout sitemap processing failed: {str(e)}"
            result["specialized_scraping"] = True
            result["handler_used"] = self.get_handler_name()
            return result

    async def _fetch_and_parse_sitemap(self, sitemap_url: str, original_url: str, session: aiohttp.ClientSession) -> SpecializedExtractionResult:
        """Fetch and parse sitemap to extract restaurant articles"""
        try:
            async with session.get(sitemap_url) as response:
                if response.status != 200:
                    raise Exception(f"Sitemap fetch failed with status {response.status}")

                sitemap_content = await response.text()

            # Parse sitemap XML
            root = ET.fromstring(sitemap_content)

            # Find URLs in the sitemap
            urls = []

            # Handle different sitemap formats
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None:
                    urls.append(loc.text)

            # Filter for restaurant-related URLs
            restaurant_urls = []
            for url in urls:
                if self._is_restaurant_url(url):
                    restaurant_urls.append(url)

                    # Limit to prevent too many results
                    if len(restaurant_urls) >= 15:
                        break

            # Convert URLs to article format
            articles = []
            for url in restaurant_urls:
                # Extract title from URL
                title = self._extract_title_from_url(url)

                article = {
                    "title": title,
                    "url": url,
                    "description": f"Restaurant article from Timeout: {title}",
                    "published_date": "",
                    "source_url": original_url
                }
                articles.append(article)

            return SpecializedExtractionResult(
                articles=articles,
                source_url=original_url,
                source_name="Time Out",
                extraction_method="sitemap",
                success=True
            )

        except Exception as e:
            return SpecializedExtractionResult(
                articles=[],
                source_url=original_url,
                source_name="Time Out",
                extraction_method="sitemap",
                success=False,
                error_message=str(e)
            )

    def _is_restaurant_url(self, url: str) -> bool:
        """Check if a URL is likely restaurant-related"""
        url_lower = url.lower()

        restaurant_url_patterns = [
            '/restaurant', '/dining', '/food', '/eat', '/cafe',
            '/bar', '/drink', '/menu', '/chef', '/cuisine',
            '/places-to-eat', '/best-restaurants', '/where-to-eat'
        ]

        return any(pattern in url_lower for pattern in restaurant_url_patterns)

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL"""
        try:
            path = urlparse(url).path
            # Get the last part of the path and clean it up
            title_part = path.split('/')[-1] or path.split('/')[-2]

            # Clean up the title
            title = title_part.replace('-', ' ').replace('_', ' ')
            title = ' '.join(word.capitalize() for word in title.split())

            return title if title else "Restaurant Article"
        except:
            return "Restaurant Article"

    def _format_articles_for_analyzer(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles for the list analyzer"""
        if not articles:
            return ""

        formatted_parts = []
        for i, article in enumerate(articles, 1):
            parts = [f"Article {i}: {article.get('title', 'Unknown')}"]

            if article.get('description'):
                parts.append(f"Description: {article['description']}")

            if article.get('url'):
                parts.append(f"URL: {article['url']}")

            if article.get('published_date'):
                parts.append(f"Published: {article['published_date']}")

            formatted_parts.append("\n".join(parts))

        return "\n\n".join(formatted_parts)

class BonAppetitHandler(SpecializedHandler):
    """Handler for Bon Appétit - RSS based"""

    def can_handle(self, url: str) -> bool:
        return 'bonappetit.com' in url.lower()

    def get_handler_name(self) -> str:
        return "Bon Appétit RSS Handler"

    async def process_url(self, result: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process Bon Appétit URLs using RSS feeds"""
        # TODO: Implement RSS parsing for Bon Appétit
        # RSS URL: https://www.bonappetit.com/feed/rss
        logger.info("Bon Appétit handler - TODO: Implement RSS parsing")

        result["scraping_failed"] = True
        result["scraping_error"] = "Bon Appétit handler not yet implemented"
        result["specialized_scraping"] = True
        result["handler_used"] = self.get_handler_name()
        return result

class FoodAndWineHandler(SpecializedHandler):
    """Handler for Food & Wine - RSS based"""

    def can_handle(self, url: str) -> bool:
        return 'foodandwine.com' in url.lower()

    def get_handler_name(self) -> str:
        return "Food & Wine RSS Handler"

    async def process_url(self, result: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process Food & Wine URLs using RSS feeds"""
        # TODO: Implement RSS parsing for Food & Wine
        # RSS URL: https://www.foodandwine.com/rss
        logger.info("Food & Wine handler - TODO: Implement RSS parsing")

        result["scraping_failed"] = True
        result["scraping_error"] = "Food & Wine handler not yet implemented"
        result["specialized_scraping"] = True
        result["handler_used"] = self.get_handler_name()
        return result

class EaterTimeoutSpecializedScraper:
    """
    Extensible specialized scraper that uses different handlers for different sites.

    To add a new site:
    1. Create a new Handler class that extends SpecializedHandler
    2. Implement can_handle(), process_url(), and get_handler_name() methods
    3. Add the handler to the handlers list in __init__
    """

    def __init__(self, config):
        self.config = config
        self.session = None

        # Initialize all handlers - ADD NEW HANDLERS HERE
        self.handlers: List[SpecializedHandler] = [
            EaterHandler(),
            TimeoutHandler(),
            BonAppetitHandler(),  # TODO: Implement
            FoodAndWineHandler(),  # TODO: Implement
            # Add more handlers here as you implement them
        ]

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_articles_found": 0,
            "handlers_used": {}  # Track which handlers were used
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def process_specialized_urls(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process URLs using specialized handlers.

        Args:
            search_results: List of search results that may be handled by specialized handlers

        Returns:
            List of enriched results with article data
        """
        if not self.session:
            async with self:
                return await self._process_urls_internal(search_results)
        else:
            return await self._process_urls_internal(search_results)

    async def _process_urls_internal(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Internal processing method"""
        specialized_results = []

        for result in search_results:
            url = result.get('url', '')
            if not url:
                specialized_results.append(result)
                continue

            # Find the appropriate handler for this URL
            handler = self._find_handler(url)

            if handler:
                logger.info(f"Processing {url} with {handler.get_handler_name()}")

                try:
                    processed_result = await handler.process_url(result, self.session)

                    # Track handler usage
                    handler_name = handler.get_handler_name()
                    self.stats["handlers_used"][handler_name] = self.stats["handlers_used"].get(handler_name, 0) + 1

                    # Update stats
                    if processed_result.get("scraping_success"):
                        self.stats["successful_extractions"] += 1
                        self.stats["total_articles_found"] += processed_result.get("article_count", 0)
                    else:
                        self.stats["failed_extractions"] += 1

                    specialized_results.append(processed_result)

                except Exception as e:
                    logger.error(f"Error in {handler.get_handler_name()} for {url}: {e}")
                    result["scraping_failed"] = True
                    result["scraping_error"] = f"Handler error: {str(e)}"
                    result["specialized_scraping"] = True
                    result["handler_used"] = handler.get_handler_name()
                    specialized_results.append(result)
                    self.stats["failed_extractions"] += 1
            else:
                # No specialized handler found, return as-is
                specialized_results.append(result)

            self.stats["total_processed"] += 1

        return specialized_results

    def _find_handler(self, url: str) -> Optional[SpecializedHandler]:
        """Find the appropriate handler for a given URL"""
        for handler in self.handlers:
            if handler.can_handle(url):
                return handler
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.stats.copy()

    def _log_stats(self):
        """Log processing statistics"""
        logger.info(f"Specialized Scraping Statistics:")
        logger.info(f"  Total URLs processed: {self.stats['total_processed']}")
        logger.info(f"  Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"  Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"  Total articles found: {self.stats['total_articles_found']}")

        if self.stats['handlers_used']:
            logger.info(f"  Handlers used:")
            for handler_name, count in self.stats['handlers_used'].items():
                logger.info(f"    {handler_name}: {count} URLs")

        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful_extractions'] / self.stats['total_processed']) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")