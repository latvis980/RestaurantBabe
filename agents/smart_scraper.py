# agents/smart_scraper.py
"""
Smart Restaurant Scraper System

Combines intelligent URL classification with optimized content sectioning.
Uses the best parts of your previous implementations in a clean, efficient system.

Key features:
- Smart URL classification (RSS/Simple HTTP/Firecrawl)
- DeepSeek-powered content sectioning
- Specialized handlers for complex sites
- Cost optimization through intelligent routing
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

# Import specialized components
from agents.specialized_scraper import EaterTimeoutSpecializedScraper
from agents.firecrawl_scraper import FirecrawlWebScraper
from agents.content_sectioning_agent import ContentSectioningAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class ScrapingStrategy(Enum):
    """Scraping strategies in order of cost efficiency"""
    SPECIALIZED = "specialized"      # RSS/Sitemap - FREE
    SIMPLE_HTTP = "simple_http"      # Basic HTTP + BeautifulSoup - ~0.1 credits
    ENHANCED_HTTP = "enhanced_http"  # HTTP + Readability + AI - ~0.5 credits
    FIRECRAWL = "firecrawl"         # Firecrawl for JS-heavy sites - ~10 credits

@dataclass
class ScrapeAnalysis:
    """Analysis result for a URL"""
    strategy: ScrapingStrategy
    confidence: float
    reasoning: str
    estimated_cost: float
    content_indicators: Dict[str, Any]

@dataclass
class ScrapingResult:
    """Result from scraping a single URL"""
    url: str
    success: bool
    strategy_used: ScrapingStrategy
    raw_content: str
    sectioned_content: str
    restaurants_found: List[str]
    source_info: Dict[str, Any]
    sectioning_stats: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class SmartRestaurantScraper:
    """
    Intelligent restaurant scraper that optimizes costs and quality.

    Flow:
    1. Analyze URLs to determine optimal scraping strategy
    2. Apply specialized handlers where possible (free)
    3. Use appropriate HTTP/Firecrawl strategy for others
    4. Section content using DeepSeek for optimal AI analysis
    5. Extract restaurant information
    """

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.analyzer = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Lazy initialization of heavy components
        self._specialized_scraper = None
        self._firecrawl_scraper = None
        self._content_sectioner = None

        # Analysis cache to avoid re-analyzing same domains
        self._domain_cache = {}

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "total_cost_estimate": 0.0,
            "cost_saved_vs_all_firecrawl": 0.0,
            "sectioning_time": 0.0,
            "avg_content_improvement": 0.0,
            "cache_hits": 0
        }

        # URL analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert at analyzing restaurant websites for optimal scraping strategies.

CLASSIFICATION RULES:

🟢 SIMPLE_HTTP (cheapest - use basic HTTP scraping):
- Static content sites where restaurant info is in HTML
- News sites, simple blogs, government tourism sites
- Sites with clear content structure visible in HTML source
- Examples: cntraveller.com articles, simple food blogs

🟡 ENHANCED_HTTP (moderate cost - use Readability + AI):
- Modern magazine sites with some JavaScript
- Sites where content is in HTML but needs cleanup
- Professional food publications with moderate complexity
- Examples: timeout.com, some food magazines

🔴 FIRECRAWL (expensive - only when necessary):
- Heavy JavaScript sites where content is dynamically loaded
- Single Page Applications (SPAs)
- Sites with anti-bot protection or complex frameworks
- Examples: some interactive restaurant platforms

ANALYSIS FACTORS:
1. Restaurant content indicators in preview
2. JavaScript complexity and framework usage  
3. Site type and known behavior patterns
4. Content-to-navigation ratio

Return JSON:
{{
    "strategy": "SIMPLE_HTTP|ENHANCED_HTTP|FIRECRAWL",
    "confidence": 0.0-1.0,
    "reasoning": "Why this strategy will work for restaurant extraction",
    "estimated_restaurant_count": 0-50,
    "javascript_dependency": "low|medium|high",
    "content_quality_score": 0.0-1.0
}}
            """),
            ("human", """
Analyze this restaurant website:

URL: {url}
Domain: {domain}
Title: {title}
Content Preview: {content_preview}

HTTP Response Analysis:
- Status: {status_code}
- Response Time: {response_time:.2f}s
- Content Length: {content_length}
- Restaurant Keywords Found: {restaurant_keywords}
- Has List Structure: {has_lists}
- JavaScript Usage: {script_analysis}
            """)
        ])

    @property
    def specialized_scraper(self):
        """Lazy initialization of specialized scraper"""
        if self._specialized_scraper is None:
            self._specialized_scraper = EaterTimeoutSpecializedScraper(self.config)
        return self._specialized_scraper

    @property
    def firecrawl_scraper(self):
        """Lazy initialization of Firecrawl scraper"""
        if self._firecrawl_scraper is None:
            self._firecrawl_scraper = FirecrawlWebScraper(self.config)
        return self._firecrawl_scraper

    @property
    def content_sectioner(self):
        """Lazy initialization of content sectioner"""
        if self._content_sectioner is None:
            self._content_sectioner = ContentSectioningAgent(self.config)
        return self._content_sectioner

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point - intelligently scrape search results.

        Args:
            search_results: List of search results from BraveSearchAgent

        Returns:
            List of enriched results with scraped restaurant data
        """
        start_time = time.time()

        logger.info(f"🧠 Smart scraping pipeline: {len(search_results)} URLs")

        # Step 1: Analyze and classify URLs
        classified_urls = await self._classify_urls(search_results)

        # Step 2: Process each strategy group
        all_results = []

        for strategy, urls in classified_urls.items():
            if not urls:
                continue

            logger.info(f"🔄 Processing {len(urls)} URLs with {strategy.value}")

            if strategy == ScrapingStrategy.SPECIALIZED:
                results = await self._process_specialized(urls)
            elif strategy == ScrapingStrategy.SIMPLE_HTTP:
                results = await self._process_simple_http(urls)
            elif strategy == ScrapingStrategy.ENHANCED_HTTP:
                results = await self._process_enhanced_http(urls)
            elif strategy == ScrapingStrategy.FIRECRAWL:
                results = await self._process_firecrawl(urls)

            all_results.extend(results)
            self.stats["strategy_breakdown"][strategy.value] += len(urls)

        # Step 3: Update statistics
        self.stats["total_processed"] += len(search_results)
        processing_time = time.time() - start_time
        self._calculate_cost_savings()

        logger.info(f"✅ Smart scraping completed in {processing_time:.2f}s")
        self._log_performance_summary()

        return all_results

    async def _classify_urls(self, search_results: List[Dict[str, Any]]) -> Dict[ScrapingStrategy, List[Dict[str, Any]]]:
        """Classify URLs by optimal scraping strategy"""

        classified = {strategy: [] for strategy in ScrapingStrategy}

        for result in search_results:
            url = result.get("url", "")
            if not url:
                continue

            # Check for specialized handlers first (free option)
            if self._is_specialized_url(url):
                classified[ScrapingStrategy.SPECIALIZED].append(result)
                continue

            # Check domain cache
            domain = self._extract_domain(url)
            cache_key = f"domain_{domain}"

            if cache_key in self._domain_cache:
                strategy = self._domain_cache[cache_key]
                classified[strategy].append(result)
                self.stats["cache_hits"] += 1
                continue

            # Analyze URL to determine strategy
            try:
                analysis = await self._analyze_url_strategy(result)
                strategy = analysis.strategy

                # Cache the result for this domain
                self._domain_cache[cache_key] = strategy

                classified[strategy].append(result)
                result["scrape_analysis"] = analysis

            except Exception as e:
                logger.warning(f"URL analysis failed for {url}: {e}")
                # Safe fallback to enhanced HTTP
                classified[ScrapingStrategy.ENHANCED_HTTP].append(result)

        return classified

    def _is_specialized_url(self, url: str) -> bool:
        """Check if URL can be handled by specialized scraper (RSS/sitemap)"""
        specialized_domains = [
            'eater.com', 'timeout.com', 'bonappetit.com', 'foodandwine.com'
        ]
        url_lower = url.lower()
        return any(domain in url_lower for domain in specialized_domains)

    async def _analyze_url_strategy(self, result: Dict[str, Any]) -> ScrapeAnalysis:
        """Analyze a URL to determine optimal scraping strategy"""

        url = result.get("url", "")
        domain = self._extract_domain(url)

        # Quick HTTP probe to analyze content
        probe_data = await self._probe_url_content(url)

        if not probe_data:
            # Fallback analysis based on URL patterns
            return self._fallback_analysis(url, domain)

        # Use AI analysis for better classification
        try:
            chain = self.analysis_prompt | self.analyzer
            response = await chain.ainvoke({
                "url": url,
                "domain": domain,
                "title": probe_data.get("title", ""),
                "content_preview": probe_data.get("content_preview", "")[:500],
                "status_code": probe_data.get("status_code", 0),
                "response_time": probe_data.get("response_time", 0),
                "content_length": probe_data.get("content_length", 0),
                "restaurant_keywords": probe_data.get("restaurant_keywords", 0),
                "has_lists": probe_data.get("has_lists", False),
                "script_analysis": probe_data.get("script_analysis", "unknown")
            })

            # Parse AI response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            ai_result = json.loads(content.strip())

            # Convert to strategy enum
            strategy_map = {
                "SIMPLE_HTTP": ScrapingStrategy.SIMPLE_HTTP,
                "ENHANCED_HTTP": ScrapingStrategy.ENHANCED_HTTP,
                "FIRECRAWL": ScrapingStrategy.FIRECRAWL
            }

            strategy = strategy_map.get(ai_result["strategy"], ScrapingStrategy.ENHANCED_HTTP)

            # Override if restaurant indicators are too low for complex strategies
            if (probe_data.get("restaurant_keywords", 0) < 2 and 
                strategy != ScrapingStrategy.FIRECRAWL):
                strategy = ScrapingStrategy.FIRECRAWL
                reasoning = f"Override: Low restaurant content detected, using Firecrawl. {ai_result.get('reasoning', '')}"
            else:
                reasoning = ai_result.get("reasoning", "AI analysis completed")

            return ScrapeAnalysis(
                strategy=strategy,
                confidence=ai_result.get("confidence", 0.7),
                reasoning=reasoning,
                estimated_cost=self._get_strategy_cost(strategy),
                content_indicators=probe_data
            )

        except Exception as e:
            logger.warning(f"AI analysis failed for {url}: {e}")
            return self._fallback_analysis(url, domain)

    async def _probe_url_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Quick probe of URL to analyze content structure"""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                start_time = time.time()
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                response_time = time.time() - start_time

                if response.status_code != 200:
                    return None

                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text()

                # Analyze content
                restaurant_keywords = [
                    'restaurant', 'menu', 'cuisine', 'chef', 'dining', 'food',
                    'dish', 'bistro', 'cafe', 'bar', 'eatery', 'price', '€'
                ]

                keyword_count = sum(1 for kw in restaurant_keywords 
                                  if kw.lower() in text_content.lower())

                # Check for list structures
                lists = soup.find_all(['ol', 'ul', 'div'], class_=lambda x: x and 'list' in str(x).lower())
                has_lists = len(lists) > 1

                # Analyze JavaScript usage
                scripts = soup.find_all('script')
                script_analysis = f"{len(scripts)} scripts found"
                if any('react' in str(script).lower() for script in scripts[:5]):
                    script_analysis += ", React detected"
                if any('angular' in str(script).lower() for script in scripts[:5]):
                    script_analysis += ", Angular detected"

                return {
                    "title": soup.title.text.strip() if soup.title else "",
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content_length": len(response.text),
                    "content_preview": text_content[:1000],
                    "restaurant_keywords": keyword_count,
                    "has_lists": has_lists,
                    "script_analysis": script_analysis
                }

        except Exception as e:
            logger.warning(f"URL probe failed for {url}: {e}")
            return None

    def _fallback_analysis(self, url: str, domain: str) -> ScrapeAnalysis:
        """Fallback analysis when probe fails"""

        # Simple domain-based heuristics
        simple_patterns = ['cntraveller.com', 'lisbonlux.com', 'gov.']
        heavy_js_patterns = ['timeout.com', 'eater.com']

        if any(pattern in domain for pattern in simple_patterns):
            strategy = ScrapingStrategy.SIMPLE_HTTP
            confidence = 0.7
        elif any(pattern in domain for pattern in heavy_js_patterns):
            strategy = ScrapingStrategy.FIRECRAWL
            confidence = 0.8
        else:
            strategy = ScrapingStrategy.ENHANCED_HTTP
            confidence = 0.5

        return ScrapeAnalysis(
            strategy=strategy,
            confidence=confidence,
            reasoning=f"Fallback analysis for {domain}",
            estimated_cost=self._get_strategy_cost(strategy),
            content_indicators={}
        )

    async def _process_specialized(self, urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process URLs using specialized handlers (RSS/sitemap - FREE)"""

        logger.info(f"🆓 Processing {len(urls)} URLs with specialized handlers (NO COST)")

        async with self.specialized_scraper:
            results = await self.specialized_scraper.process_specialized_urls(urls)

        # Apply content sectioning to specialized results
        for result in results:
            if result.get("scraping_success") and result.get("scraped_content"):
                await self._apply_content_sectioning(result, "specialized")

        return results

    async def _process_simple_http(self, urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process URLs using simple HTTP scraping"""

        results = []
        semaphore = asyncio.Semaphore(8)  # Higher concurrency for simple requests

        async def scrape_single_simple(result):
            async with semaphore:
                url = result.get("url", "")

                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })

                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Clean content extraction
                            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                                tag.decompose()

                            # Find main content
                            main_content = (
                                soup.find('main') or 
                                soup.find('article') or 
                                soup.find('div', class_=lambda x: x and 'content' in str(x).lower())
                            )

                            if main_content:
                                content = main_content.get_text(separator='\n\n', strip=True)
                            else:
                                content = soup.get_text(separator='\n\n', strip=True)

                            # Store raw content
                            result["scraped_content"] = content
                            result["scraping_success"] = True
                            result["scraping_method"] = "simple_http"

                            # Apply content sectioning
                            await self._apply_content_sectioning(result, "simple_http")

                            return result

                except Exception as e:
                    logger.warning(f"Simple HTTP scraping failed for {url}: {e}")

                result["scraping_failed"] = True
                result["scraping_method"] = "simple_http"
                return result

        tasks = [scrape_single_simple(result) for result in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _process_enhanced_http(self, urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process URLs using enhanced HTTP with Readability"""

        results = []
        semaphore = asyncio.Semaphore(5)  # Moderate concurrency

        async def scrape_single_enhanced(result):
            async with semaphore:
                url = result.get("url", "")

                try:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        response = await client.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })

                        if response.status_code == 200:
                            # Use readability for content extraction
                            from readability import Document
                            doc = Document(response.text)

                            readable_html = doc.summary()
                            title = doc.title()

                            # Parse cleaned HTML
                            soup = BeautifulSoup(readable_html, 'html.parser')
                            content = soup.get_text(separator='\n\n', strip=True)

                            # Store results
                            result["scraped_content"] = content
                            result["scraped_title"] = title
                            result["scraping_success"] = True
                            result["scraping_method"] = "enhanced_http"

                            # Apply content sectioning
                            await self._apply_content_sectioning(result, "enhanced_http")

                            return result

                except Exception as e:
                    logger.warning(f"Enhanced HTTP scraping failed for {url}: {e}")

                result["scraping_failed"] = True
                result["scraping_method"] = "enhanced_http"
                return result

        tasks = [scrape_single_enhanced(result) for result in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _process_firecrawl(self, urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process URLs using expensive Firecrawl"""

        logger.warning(f"💸 Using Firecrawl for {len(urls)} URLs (~{len(urls) * 10} credits)")

        # Use existing Firecrawl scraper
        results = await self.firecrawl_scraper.scrape_search_results(urls)

        # Apply content sectioning to Firecrawl results
        for result in results:
            if result.get("scraping_success") and result.get("scraped_content"):
                await self._apply_content_sectioning(result, "firecrawl")

        return results

    async def _apply_content_sectioning(self, result: Dict[str, Any], method: str):
        """Apply DeepSeek-powered content sectioning to optimize for AI analysis"""

        content = result.get("scraped_content", "")
        if not content or len(content) < 500:
            return  # Skip sectioning for short content

        url = result.get("url", "")

        try:
            sectioning_start = time.time()

            # Apply content sectioning
            sectioning_result = await self.content_sectioner.process_content(
                content=content,
                url=url,
                source_method=method
            )

            sectioning_time = time.time() - sectioning_start
            self.stats["sectioning_time"] += sectioning_time

            # Update result with sectioned content
            result["scraped_content"] = sectioning_result.optimized_content
            result["sectioning_result"] = {
                "original_length": sectioning_result.original_length,
                "optimized_length": sectioning_result.optimized_length,
                "improvement_ratio": sectioning_result.optimized_length / max(sectioning_result.original_length, 1),
                "restaurant_density": sectioning_result.restaurants_density,
                "confidence": sectioning_result.confidence,
                "method": sectioning_result.sectioning_method
            }

            # Track improvement
            if sectioning_result.original_length > 0:
                improvement = sectioning_result.optimized_length / sectioning_result.original_length
                current_avg = self.stats["avg_content_improvement"]
                processed = self.stats["total_processed"]
                self.stats["avg_content_improvement"] = (current_avg * processed + improvement) / (processed + 1)

            logger.debug(f"📊 Content sectioned: {sectioning_result.original_length} → {sectioning_result.optimized_length} chars")

        except Exception as e:
            logger.warning(f"Content sectioning failed for {url}: {e}")
            # Keep original content if sectioning fails

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return domain.replace('www.', '') if domain.startswith('www.') else domain
        except:
            return url

    def _get_strategy_cost(self, strategy: ScrapingStrategy) -> float:
        """Get estimated cost for a strategy"""
        costs = {
            ScrapingStrategy.SPECIALIZED: 0.0,
            ScrapingStrategy.SIMPLE_HTTP: 0.1,
            ScrapingStrategy.ENHANCED_HTTP: 0.5,
            ScrapingStrategy.FIRECRAWL: 10.0
        }
        return costs.get(strategy, 1.0)

    def _calculate_cost_savings(self):
        """Calculate cost savings vs using Firecrawl for everything"""
        total_urls = self.stats["total_processed"]
        if total_urls == 0:
            return

        # Calculate actual costs
        actual_cost = sum(
            count * self._get_strategy_cost(ScrapingStrategy(strategy))
            for strategy, count in self.stats["strategy_breakdown"].items()
        )

        # Calculate what it would cost with all Firecrawl
        all_firecrawl_cost = total_urls * 10.0

        self.stats["total_cost_estimate"] = actual_cost
        self.stats["cost_saved_vs_all_firecrawl"] = all_firecrawl_cost - actual_cost

    def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        stats = self.stats

        logger.info("🚀 SMART SCRAPING PERFORMANCE SUMMARY:")
        logger.info(f"  📊 Total URLs: {stats['total_processed']}")

        for strategy, count in stats["strategy_breakdown"].items():
            if count > 0:
                cost = self._get_strategy_cost(ScrapingStrategy(strategy))
                emoji = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}
                logger.info(f"  {emoji.get(strategy, '📌')} {strategy.title()}: {count} URLs (~{count * cost:.1f} credits)")

        logger.info(f"  💰 Total cost estimate: {stats['total_cost_estimate']:.1f} credits")
        logger.info(f"  💾 Cost saved vs all-Firecrawl: {stats['cost_saved_vs_all_firecrawl']:.1f} credits")

        if stats["sectioning_time"] > 0:
            logger.info(f"  🧠 Content sectioning: {stats['sectioning_time']:.1f}s")
            logger.info(f"  📈 Avg content improvement: {stats['avg_content_improvement']:.1f}x")

        if stats["cache_hits"] > 0:
            logger.info(f"  🎯 Domain cache hits: {stats['cache_hits']}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics"""
        return {
            **self.stats,
            "domain_cache_size": len(self._domain_cache),
            "sectioning_stats": self.content_sectioner.get_stats() if self._content_sectioner else {}
        }

    def clear_cache(self):
        """Clear domain analysis cache"""
        self._domain_cache.clear()
        logger.info("🗑️ Domain cache cleared")


# Legacy compatibility wrapper
class WebScraper:
    """
    Drop-in replacement for existing WebScraper class.
    Maintains full API compatibility.
    """

    def __init__(self, config):
        self.scraper = SmartRestaurantScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main scraping method"""
        return await self.scraper.scrape_search_results(search_results)

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Alternative method name for compatibility"""
        return await self.scraper.scrape_search_results(search_results)

    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        return self.scraper.get_stats()