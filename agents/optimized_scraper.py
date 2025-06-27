# agents/optimized_scraper.py
import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from dataclasses import dataclass
from enum import Enum

# Import your existing scrapers
from agents.scraper import FirecrawlWebScraper
from agents.specialized_scraper import EaterTimeoutSpecializedScraper

# For the hybrid scraper components
import httpx
from bs4 import BeautifulSoup
from readability import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class ScrapeComplexity(Enum):
    """Website complexity levels for scraper selection"""
    SIMPLE_HTML = "simple_html"           # Static HTML, minimal JS
    MODERATE_JS = "moderate_js"           # Some JS but content in HTML  
    HEAVY_JS = "heavy_js"                 # Content loaded via JS/AJAX
    SPECIALIZED = "specialized"           # Has specialized handler
    UNKNOWN = "unknown"                   # Needs analysis

@dataclass
class ScrapeStrategy:
    """Strategy for scraping a specific URL"""
    complexity: ScrapeComplexity
    scraper_type: str
    estimated_cost: int  # In "firecrawl credit units"
    confidence: float    # 0.0-1.0

class OptimizedHybridScraper:
    """
    Intelligent hybrid scraper that routes URLs to the most cost-effective method:

    1. Specialized handlers (FREE) - RSS/sitemaps for Eater, Timeout, etc.
    2. Simple HTTP + BeautifulSoup (VERY CHEAP) - Static HTML sites
    3. Enhanced HTTP with readability (CHEAP) - Moderate complexity sites  
    4. Firecrawl (EXPENSIVE) - Only for heavy JS sites that absolutely need it
    """

    def __init__(self, config):
        self.config = config

        # Initialize all scraper components
        self.firecrawl_scraper = FirecrawlWebScraper(config)
        self.specialized_scraper = None  # Will initialize as needed

        # For strategy analysis
        self.analyzer = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1
        )

        # Website categorization patterns
        self.simple_html_patterns = [
            # News sites with good HTML structure
            r'.*guardian\.com.*', r'.*telegraph\.co\.uk.*', r'.*bbc\.com.*',
            r'.*nytimes\.com.*', r'.*washingtonpost\.com.*',
            # Food blogs and simple sites
            r'.*serious-?eats\.com.*', r'.*food52\.com.*',
            # Government and institutional sites
            r'.*\.gov.*', r'.*\.edu.*', r'.*wikipedia\.org.*',
            # Simple review/guide sites
            r'.*zagat\.com.*', r'.*michelin\.com.*'
        ]

        self.heavy_js_patterns = [
            # Known problematic sites
            r'.*timeout\.com.*', r'.*eater\.com.*',  # But these have specialized handlers
            r'.*thrillist\.com.*', r'.*bonappetit\.com.*',
            r'.*foodandwine\.com.*', r'.*epicurious\.com.*',
            # Social and dynamic sites
            r'.*instagram\.com.*', r'.*facebook\.com.*', r'.*twitter\.com.*',
            # SPA frameworks
            r'.*angular\..*', r'.*react\..*', r'.*vue\..*'
        ]

        # Cost tracking
        self.stats = {
            "total_processed": 0,
            "firecrawl_used": 0,
            "specialized_used": 0,
            "simple_http_used": 0,
            "enhanced_http_used": 0,
            "total_cost_saved": 0,  # Estimated firecrawl credits saved
            "processing_time": 0.0
        }

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point - intelligently route URLs to optimal scrapers

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of enriched results with scraped content
        """
        start_time = time.time()
        logger.info(f"🔍 Starting intelligent scraping for {len(search_results)} URLs")

        # Step 1: Analyze and categorize all URLs
        categorization_tasks = [
            self._analyze_url_strategy(result) 
            for result in search_results
        ]
        strategies = await asyncio.gather(*categorization_tasks)

        # Step 2: Group URLs by scraping strategy
        url_groups = self._group_by_strategy(search_results, strategies)

        # Step 3: Log the strategy distribution
        self._log_strategy_distribution(url_groups)

        # Step 4: Process each group with optimal scraper
        all_results = []

        # Process specialized URLs first (FREE)
        if url_groups.get(ScrapeComplexity.SPECIALIZED):
            logger.info(f"📡 Processing {len(url_groups[ScrapeComplexity.SPECIALIZED])} URLs with specialized handlers (FREE)")
            specialized_results = await self._process_specialized_urls(
                url_groups[ScrapeComplexity.SPECIALIZED]
            )
            all_results.extend(specialized_results)
            self.stats["specialized_used"] += len(url_groups[ScrapeComplexity.SPECIALIZED])

        # Process simple HTML URLs (VERY CHEAP)
        if url_groups.get(ScrapeComplexity.SIMPLE_HTML):
            logger.info(f"🌐 Processing {len(url_groups[ScrapeComplexity.SIMPLE_HTML])} URLs with simple HTTP (VERY CHEAP)")
            simple_results = await self._process_simple_http_urls(
                url_groups[ScrapeComplexity.SIMPLE_HTML]
            )
            all_results.extend(simple_results)
            self.stats["simple_http_used"] += len(url_groups[ScrapeComplexity.SIMPLE_HTML])

        # Process moderate JS URLs (CHEAP)
        if url_groups.get(ScrapeComplexity.MODERATE_JS):
            logger.info(f"📄 Processing {len(url_groups[ScrapeComplexity.MODERATE_JS])} URLs with enhanced HTTP (CHEAP)")
            enhanced_results = await self._process_enhanced_http_urls(
                url_groups[ScrapeComplexity.MODERATE_JS]
            )
            all_results.extend(enhanced_results)
            self.stats["enhanced_http_used"] += len(url_groups[ScrapeComplexity.MODERATE_JS])

        # Process heavy JS URLs with Firecrawl (EXPENSIVE) - only as last resort
        if url_groups.get(ScrapeComplexity.HEAVY_JS):
            logger.warning(f"🔥 Processing {len(url_groups[ScrapeComplexity.HEAVY_JS])} URLs with Firecrawl (EXPENSIVE)")
            firecrawl_results = await self._process_firecrawl_urls(
                url_groups[ScrapeComplexity.HEAVY_JS]
            )
            all_results.extend(firecrawl_results)
            self.stats["firecrawl_used"] += len(url_groups[ScrapeComplexity.HEAVY_JS])

        # Process unknown URLs with fallback strategy
        if url_groups.get(ScrapeComplexity.UNKNOWN):
            logger.info(f"❓ Processing {len(url_groups[ScrapeComplexity.UNKNOWN])} unknown URLs with fallback")
            unknown_results = await self._process_unknown_urls(
                url_groups[ScrapeComplexity.UNKNOWN]
            )
            all_results.extend(unknown_results)

        # Update final stats
        self.stats["total_processed"] = len(search_results)
        self.stats["processing_time"] = time.time() - start_time

        # Calculate cost savings
        total_urls = len(search_results)
        firecrawl_cost_if_all = total_urls * 10  # 10 credits per URL with Firecrawl
        actual_firecrawl_cost = self.stats["firecrawl_used"] * 10
        self.stats["total_cost_saved"] = firecrawl_cost_if_all - actual_firecrawl_cost

        self._log_final_stats()

        return all_results

    async def _analyze_url_strategy(self, result: Dict[str, Any]) -> ScrapeStrategy:
        """Analyze a URL to determine the optimal scraping strategy"""
        url = result.get("url", "")
        title = result.get("title", "")
        description = result.get("description", "")

        # Check for specialized handlers first
        if await self._has_specialized_handler(url):
            return ScrapeStrategy(
                complexity=ScrapeComplexity.SPECIALIZED,
                scraper_type="specialized",
                estimated_cost=0,
                confidence=0.9
            )

        # Check simple HTML patterns
        if self._matches_patterns(url, self.simple_html_patterns):
            return ScrapeStrategy(
                complexity=ScrapeComplexity.SIMPLE_HTML,
                scraper_type="simple_http",
                estimated_cost=0.1,  # Minimal cost for HTTP request
                confidence=0.8
            )

        # Check heavy JS patterns
        if self._matches_patterns(url, self.heavy_js_patterns):
            return ScrapeStrategy(
                complexity=ScrapeComplexity.HEAVY_JS,
                scraper_type="firecrawl",
                estimated_cost=10,
                confidence=0.8
            )

        # For unknown URLs, use AI analysis with quick HTTP probe
        return await self._ai_analyze_url_complexity(url, title, description)

    async def _ai_analyze_url_complexity(self, url: str, title: str, description: str) -> ScrapeStrategy:
        """Use AI to analyze URL complexity when patterns don't match"""

        # Quick HTTP probe to check initial content
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })

                if response.status_code == 200:
                    html_content = response.text[:3000]  # First 3k chars

                    # AI analysis prompt
                    analysis_prompt = ChatPromptTemplate.from_messages([
                        ("system", """
                        Analyze this website to determine scraping complexity:

                        SIMPLE_HTML: Static content, minimal JavaScript, content in initial HTML
                        - News sites, blogs, government sites, simple restaurant pages

                        MODERATE_JS: Some JavaScript but main content in HTML with readability
                        - Magazine sites, food blogs with some interactive elements

                        HEAVY_JS: Content loaded dynamically via JavaScript/AJAX  
                        - Single Page Apps, heavy media sites, complex restaurant platforms

                        Return JSON: {{"complexity": "SIMPLE_HTML|MODERATE_JS|HEAVY_JS", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
                        """),
                        ("human", """
                        URL: {url}
                        Title: {title}
                        Description: {description}

                        HTML Preview (first 3000 chars):
                        {html_preview}
                        """)
                    ])

                    chain = analysis_prompt | self.analyzer
                    response = await chain.ainvoke({
                        "url": url,
                        "title": title,
                        "description": description,
                        "html_preview": html_content
                    })

                    # Parse AI response
                    content = response.content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]

                    analysis = json.loads(content.strip())
                    complexity_str = analysis.get("complexity", "MODERATE_JS")
                    confidence = analysis.get("confidence", 0.5)

                    # Map to enum
                    complexity_map = {
                        "SIMPLE_HTML": ScrapeComplexity.SIMPLE_HTML,
                        "MODERATE_JS": ScrapeComplexity.MODERATE_JS,
                        "HEAVY_JS": ScrapeComplexity.HEAVY_JS
                    }

                    complexity = complexity_map.get(complexity_str, ScrapeComplexity.MODERATE_JS)

                    # Determine scraper type and cost
                    if complexity == ScrapeComplexity.SIMPLE_HTML:
                        scraper_type, cost = "simple_http", 0.1
                    elif complexity == ScrapeComplexity.MODERATE_JS:
                        scraper_type, cost = "enhanced_http", 0.5
                    else:
                        scraper_type, cost = "firecrawl", 10

                    return ScrapeStrategy(
                        complexity=complexity,
                        scraper_type=scraper_type,
                        estimated_cost=cost,
                        confidence=confidence
                    )

        except Exception as e:
            logger.warning(f"Failed to analyze URL {url}: {e}")

        # Default fallback
        return ScrapeStrategy(
            complexity=ScrapeComplexity.UNKNOWN,
            scraper_type="enhanced_http",
            estimated_cost=0.5,
            confidence=0.3
        )

    async def _has_specialized_handler(self, url: str) -> bool:
        """Check if URL can be handled by specialized scraper"""
        try:
            if not self.specialized_scraper:
                self.specialized_scraper = EaterTimeoutSpecializedScraper(self.config)

            return self.specialized_scraper._find_handler(url) is not None
        except Exception:
            return False

    def _matches_patterns(self, url: str, patterns: List[str]) -> bool:
        """Check if URL matches any of the given regex patterns"""
        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in patterns)

    def _group_by_strategy(self, search_results: List[Dict], strategies: List[ScrapeStrategy]) -> Dict:
        """Group URLs by their optimal scraping strategy"""
        groups = {}

        for result, strategy in zip(search_results, strategies):
            if strategy.complexity not in groups:
                groups[strategy.complexity] = []

            # Add strategy info to result
            result["scrape_strategy"] = strategy
            groups[strategy.complexity].append(result)

        return groups

    def _log_strategy_distribution(self, url_groups: Dict):
        """Log how URLs were distributed across strategies"""
        logger.info("📊 URL Distribution by Scraping Strategy:")

        total_cost = 0
        for complexity, urls in url_groups.items():
            count = len(urls)
            if count > 0:
                avg_cost = sum(url["scrape_strategy"].estimated_cost for url in urls) / count
                strategy_cost = sum(url["scrape_strategy"].estimated_cost for url in urls)
                total_cost += strategy_cost

                logger.info(f"  {complexity.value}: {count} URLs (avg cost: {avg_cost:.1f}, total: {strategy_cost:.1f})")

        logger.info(f"  💰 Estimated total cost: {total_cost:.1f} Firecrawl credits")

    async def _process_specialized_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using specialized handlers (RSS, sitemaps, etc.)"""
        if not self.specialized_scraper:
            self.specialized_scraper = EaterTimeoutSpecializedScraper(self.config)

        async with self.specialized_scraper:
            return await self.specialized_scraper.process_specialized_urls(urls)

    async def _process_simple_http_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using simple HTTP + BeautifulSoup"""
        results = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def process_single_simple(result):
            async with semaphore:
                url = result.get("url", "")
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })

                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Remove unwanted elements
                            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                                tag.decompose()

                            # Extract clean text
                            title = soup.title.text.strip() if soup.title else ""

                            # Get main content
                            main_content = (soup.find('main') or 
                                          soup.find('article') or 
                                          soup.find(class_='content') or 
                                          soup.body)

                            if main_content:
                                content_text = main_content.get_text(separator='\n\n', strip=True)
                            else:
                                content_text = soup.get_text(separator='\n\n', strip=True)

                            # Extract restaurant-related info using patterns
                            restaurants = self._extract_restaurant_names_simple(content_text)

                            enhanced_result = result.copy()
                            enhanced_result.update({
                                "scraped_content": content_text[:3000],  # Limit size
                                "scraped_title": title,
                                "restaurants_found": restaurants,
                                "scraping_method": "simple_http",
                                "scraping_success": True,
                                "source_info": {
                                    "name": self._extract_source_name(url),
                                    "url": url,
                                    "extraction_method": "simple_http"
                                }
                            })

                            return enhanced_result

                except Exception as e:
                    logger.warning(f"Simple HTTP scraping failed for {url}: {e}")

                # Return original result if scraping failed
                result["scraping_failed"] = True
                result["scraping_method"] = "simple_http"
                return result

        # Process all URLs concurrently
        tasks = [process_single_simple(result) for result in urls]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r]

    async def _process_enhanced_http_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using HTTP + Readability + AI extraction"""
        results = []
        semaphore = asyncio.Semaphore(3)  # Lower concurrency for enhanced processing

        async def process_single_enhanced(result):
            async with semaphore:
                url = result.get("url", "")
                try:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        response = await client.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })

                        if response.status_code == 200:
                            # Use readability to extract main content
                            doc = Document(response.text)
                            readable_html = doc.summary()
                            title = doc.title()

                            # Parse with BeautifulSoup
                            soup = BeautifulSoup(readable_html, 'html.parser')
                            content_text = soup.get_text(separator='\n\n', strip=True)

                            # Use AI to extract restaurants from content
                            restaurants = await self._ai_extract_restaurants(content_text[:2000])

                            enhanced_result = result.copy()
                            enhanced_result.update({
                                "scraped_content": content_text[:4000],
                                "scraped_title": title or "",
                                "restaurants_found": restaurants,
                                "scraping_method": "enhanced_http",
                                "scraping_success": True,
                                "source_info": {
                                    "name": self._extract_source_name(url),
                                    "url": url,
                                    "extraction_method": "enhanced_http_readability"
                                }
                            })

                            return enhanced_result

                except Exception as e:
                    logger.warning(f"Enhanced HTTP scraping failed for {url}: {e}")

                result["scraping_failed"] = True
                result["scraping_method"] = "enhanced_http"
                return result

        tasks = [process_single_enhanced(result) for result in urls]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r]

    async def _process_firecrawl_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using Firecrawl (expensive, last resort)"""
        logger.warning(f"🚨 Using expensive Firecrawl for {len(urls)} URLs")
        return await self.firecrawl_scraper.scrape_search_results(urls)

    async def _process_unknown_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process unknown URLs with fallback strategy"""
        # Try enhanced HTTP first, fall back to Firecrawl if needed
        enhanced_results = await self._process_enhanced_http_urls(urls)

        # Check which ones failed and might need Firecrawl
        failed_urls = [r for r in enhanced_results if r.get("scraping_failed")]
        successful_urls = [r for r in enhanced_results if not r.get("scraping_failed")]

        if failed_urls:
            logger.info(f"🔄 Retrying {len(failed_urls)} failed URLs with Firecrawl")
            firecrawl_results = await self._process_firecrawl_urls(failed_urls)
            successful_urls.extend(firecrawl_results)
            self.stats["firecrawl_used"] += len(failed_urls)

        return successful_urls

    def _extract_restaurant_names_simple(self, text: str) -> List[str]:
        """Extract potential restaurant names using simple patterns"""
        # Look for common restaurant name patterns
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Two capitalized words
            r'\b[A-Z][a-z]+\'s\b',           # Possessive names
            r'\bCafe [A-Z][a-z]+\b',         # Cafe + name
            r'\bRestaurant [A-Z][a-z]+\b',   # Restaurant + name
        ]

        restaurants = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            restaurants.extend(matches)

        # Remove duplicates and filter
        unique_restaurants = list(set(restaurants))
        return [r for r in unique_restaurants if len(r) > 3 and len(r) < 50][:10]

    async def _ai_extract_restaurants(self, content: str) -> List[str]:
        """Use AI to extract restaurant names from content"""
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract restaurant, cafe, bar, and food venue names from the text. Return as JSON array: [\"Restaurant 1\", \"Restaurant 2\"]"),
                ("human", "Content:\n{content}")
            ])

            chain = extraction_prompt | self.analyzer
            response = await chain.ainvoke({"content": content})

            # Parse response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            restaurants = json.loads(content.strip())
            return restaurants[:15] if isinstance(restaurants, list) else []

        except Exception as e:
            logger.warning(f"AI extraction failed: {e}")
            return []

    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # Simple domain to name mapping
            domain_map = {
                'timeout.com': 'Time Out',
                'eater.com': 'Eater',
                'theguardian.com': 'The Guardian',
                'nytimes.com': 'New York Times',
                'seriouseats.com': 'Serious Eats',
                'food52.com': 'Food52',
            }

            return domain_map.get(domain, domain.split('.')[0].title())

        except Exception:
            return "Unknown Source"

    def _log_final_stats(self):
        """Log final scraping statistics with cost analysis"""
        stats = self.stats

        logger.info("📈 HYBRID SCRAPING RESULTS:")
        logger.info(f"  Total URLs processed: {stats['total_processed']}")
        logger.info(f"  Specialized handlers: {stats['specialized_used']} (FREE)")
        logger.info(f"  Simple HTTP: {stats['simple_http_used']} (VERY CHEAP)")  
        logger.info(f"  Enhanced HTTP: {stats['enhanced_http_used']} (CHEAP)")
        logger.info(f"  Firecrawl: {stats['firecrawl_used']} (EXPENSIVE)")
        logger.info(f"  💰 Cost saved: ~{stats['total_cost_saved']} Firecrawl credits")
        logger.info(f"  ⏱️ Processing time: {stats['processing_time']:.2f}s")

        if stats['total_processed'] > 0:
            firecrawl_percentage = (stats['firecrawl_used'] / stats['total_processed']) * 100
            logger.info(f"  📊 Firecrawl usage: {firecrawl_percentage:.1f}% of total URLs")

    def get_stats(self) -> Dict[str, Any]:
        """Get current scraping statistics"""
        return self.stats.copy()


# Legacy compatibility wrapper
class WebScraper:
    """Compatibility wrapper for existing code"""

    def __init__(self, config):
        self.hybrid_scraper = OptimizedHybridScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.hybrid_scraper.scrape_search_results(search_results)

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.hybrid_scraper.scrape_search_results(search_results)

    def get_stats(self) -> Dict[str, Any]:
        return self.hybrid_scraper.get_stats()