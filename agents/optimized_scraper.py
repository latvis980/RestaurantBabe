# agents/optimized_scraper.py - Updated with Persistent Domain Intelligence
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
from agents.firecrawl_scraper import FirecrawlWebScraper
from agents.specialized_scraper import EaterTimeoutSpecializedScraper

# Import domain intelligence utilities
from utils.domain_intelligence import (
    initialize_domain_intelligence_db,
    save_domain_intelligence,
    load_all_domain_intelligence,
    get_domain_intelligence,
    cleanup_old_domain_intelligence,
    get_domain_intelligence_stats
)

# For the intelligent scraper components
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
    reasoning: str       # Why this strategy was chosen

class IntelligentAdaptiveScraper:
    """
    Intelligent scraper that uses AI to analyze each URL dynamically.
    Now with persistent domain intelligence that learns and improves over time.

    Features:
    1. Quick HTTP probe to get initial content
    2. AI analysis of HTML structure and content
    3. Dynamic routing to optimal scraper
    4. Persistent learning from successes/failures
    5. Database-backed domain intelligence cache
    """

    def __init__(self, config):
        self.config = config

        # Initialize database
        initialize_domain_intelligence_db(config)

        # Initialize all scraper components
        self.firecrawl_scraper = FirecrawlWebScraper(config)
        self.specialized_scraper = None  # Will initialize as needed

        # AI analyzer with enhanced prompts
        self.analyzer = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1  # Low temperature for consistent analysis
        )

        # Domain intelligence - now backed by database
        self.domain_intelligence = {}  # In-memory cache for speed
        self._intelligence_loaded = False

        # Cost tracking
        self.stats = {
            "total_processed": 0,
            "firecrawl_used": 0,
            "specialized_used": 0,
            "simple_http_used": 0,
            "enhanced_http_used": 0,
            "total_cost_saved": 0,
            "processing_time": 0.0,
            "ai_analysis_calls": 0,
            "cache_hits": 0,
            "database_cache_hits": 0,
            "strategy_overrides": 0,
            "domains_learned_this_session": 0,
            "intelligence_saves": 0
        }

    async def _ensure_domain_intelligence_loaded(self):
        """Load domain intelligence from database if not already loaded"""
        if self._intelligence_loaded:
            return

        try:
            logger.info("Loading domain intelligence from database...")

            # Load all domain intelligence from database
            domain_records = load_all_domain_intelligence(self.config)

            # Convert to in-memory format
            for record in domain_records:
                domain = record['domain']
                self.domain_intelligence[domain] = {
                    'complexity': record['complexity'],
                    'scraper_type': record['scraper_type'],
                    'cost': record['cost'],
                    'confidence': record['confidence'],
                    'reasoning': record['reasoning'],
                    'success_count': record.get('success_count', 0),
                    'failure_count': record.get('failure_count', 0),
                    'timestamp': record.get('last_updated', time.time()),
                    'total_restaurants_found': record.get('total_restaurants_found', 0),
                    'avg_content_length': record.get('avg_content_length', 0),
                    'metadata': record.get('metadata', {})
                }

            self._intelligence_loaded = True
            logger.info(f"Loaded intelligence for {len(self.domain_intelligence)} domains from database")

            # Log some stats about loaded intelligence
            high_confidence = sum(1 for info in self.domain_intelligence.values() if info['confidence'] > 0.8)
            logger.info(f"High-confidence domains: {high_confidence}/{len(self.domain_intelligence)}")

        except Exception as e:
            logger.error(f"Failed to load domain intelligence from database: {e}")
            self._intelligence_loaded = True  # Don't keep trying

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point - intelligently analyze each URL and route to optimal scraper
        """
        start_time = time.time()
        logger.info(f"🧠 Starting intelligent analysis for {len(search_results)} URLs")

        # Ensure domain intelligence is loaded from database
        await self._ensure_domain_intelligence_loaded()

        # Step 1: Analyze all URLs with AI-powered strategy detection
        analysis_tasks = [
            self._intelligent_url_analysis(result) 
            for result in search_results
        ]
        strategies = await asyncio.gather(*analysis_tasks)

        # Step 2: Group URLs by strategy and log distribution
        url_groups = self._group_by_strategy(search_results, strategies)
        self._log_strategy_distribution(url_groups)

        # Step 3: Process each group with optimal scraper
        all_results = await self._process_url_groups(url_groups)

        # Step 4: Learn from results and save to database
        await self._update_domain_intelligence(all_results)

        # Step 5: Periodic cleanup of old intelligence
        await self._periodic_cleanup()

        # Update final stats
        self.stats["total_processed"] = len(search_results)
        self.stats["processing_time"] = time.time() - start_time
        self._calculate_cost_savings()
        self._log_final_stats()

        return all_results

    async def _intelligent_url_analysis(self, result: Dict[str, Any]) -> ScrapeStrategy:
        """
        Intelligent analysis of URL using persistent domain intelligence and AI
        """
        url = result.get("url", "")
        title = result.get("title", "")
        description = result.get("description", "")

        # Step 1: Check persistent domain intelligence cache
        domain = self._extract_domain(url)
        if domain in self.domain_intelligence:
            cached_strategy = self.domain_intelligence[domain]

            # Use cached strategy if confidence is high and recent
            age_hours = (time.time() - cached_strategy['timestamp']) / 3600
            confidence_threshold = 0.8 if age_hours < 24 else 0.9  # Higher threshold for older cache

            if cached_strategy['confidence'] > confidence_threshold:
                self.stats["cache_hits"] += 1
                self.stats["database_cache_hits"] += 1
                logger.debug(f"🎯 Using cached strategy for {domain}: {cached_strategy['complexity']} (confidence: {cached_strategy['confidence']:.2f})")
                return ScrapeStrategy(
                    complexity=ScrapeComplexity(cached_strategy['complexity']),
                    scraper_type=cached_strategy['scraper_type'],
                    estimated_cost=cached_strategy['cost'],
                    confidence=cached_strategy['confidence'],
                    reasoning=f"Cached from database: {cached_strategy['reasoning']} (used {cached_strategy['success_count']} times)"
                )

        # Step 2: Check for specialized handlers
        if await self._has_specialized_handler(url):
            strategy = ScrapeStrategy(
                complexity=ScrapeComplexity.SPECIALIZED,
                scraper_type="specialized",
                estimated_cost=0,
                confidence=0.95,
                reasoning="Has dedicated specialized handler"
            )

            # Save this knowledge for future use
            await self._save_strategy_to_intelligence(domain, strategy, is_new=True)
            return strategy

        # Step 3: AI-powered analysis with quick HTTP probe
        return await self._ai_powered_analysis(url, title, description, domain)

    async def _save_strategy_to_intelligence(self, domain: str, strategy: ScrapeStrategy, is_new: bool = False, 
                                           success_info: Dict[str, Any] = None):
        """Save strategy to persistent domain intelligence"""

        # Get existing intelligence or create new
        existing = self.domain_intelligence.get(domain, {})

        intelligence_data = {
            'complexity': strategy.complexity.value,
            'scraper_type': strategy.scraper_type,
            'cost': strategy.estimated_cost,
            'confidence': strategy.confidence,
            'reasoning': strategy.reasoning,
            'success_count': existing.get('success_count', 0),
            'failure_count': existing.get('failure_count', 0),
            'total_restaurants_found': existing.get('total_restaurants_found', 0),
            'avg_content_length': existing.get('avg_content_length', 0),
            'timestamp': time.time(),
            'metadata': existing.get('metadata', {})
        }

        # Update with success information if provided
        if success_info:
            intelligence_data['was_successful'] = success_info.get('success', False)
            if success_info.get('success'):
                intelligence_data['success_count'] += 1
                intelligence_data['total_restaurants_found'] += success_info.get('restaurants_found', 0)
                intelligence_data['avg_content_length'] = int(
                    (intelligence_data['avg_content_length'] + success_info.get('content_length', 0)) / 2
                )
            else:
                intelligence_data['failure_count'] += 1

        # Update in-memory cache
        self.domain_intelligence[domain] = intelligence_data

        # Save to database asynchronously (don't block on database)
        try:
            success = save_domain_intelligence(domain, intelligence_data, self.config)
            if success:
                self.stats["intelligence_saves"] += 1
                if is_new:
                    self.stats["domains_learned_this_session"] += 1
                logger.debug(f"💾 Saved intelligence for {domain} to database")
        except Exception as e:
            logger.warning(f"Failed to save domain intelligence for {domain}: {e}")

    async def _ai_powered_analysis(self, url: str, title: str, description: str, domain: str) -> ScrapeStrategy:
        """
        Use AI to analyze website characteristics and determine optimal scraping strategy
        Results are automatically saved to persistent storage
        """
        self.stats["ai_analysis_calls"] += 1

        # Quick HTTP probe to get website characteristics
        html_preview = ""
        response_time = 0
        status_code = 0

        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                response_time = time.time() - start_time
                status_code = response.status_code

                if response.status_code == 200:
                    html_content = response.text
                    html_preview = html_content[:4000]  # First 4k chars for analysis
                else:
                    # If we can't fetch, default to enhanced method
                    strategy = ScrapeStrategy(
                        complexity=ScrapeComplexity.MODERATE_JS,
                        scraper_type="enhanced_http",
                        estimated_cost=0.5,
                        confidence=0.4,
                        reasoning=f"HTTP error {response.status_code}, using safe fallback"
                    )
                    await self._save_strategy_to_intelligence(domain, strategy, is_new=True)
                    return strategy

        except Exception as e:
            logger.warning(f"Failed to probe {url}: {e}")
            strategy = ScrapeStrategy(
                complexity=ScrapeComplexity.MODERATE_JS,
                scraper_type="enhanced_http", 
                estimated_cost=0.5,
                confidence=0.3,
                reasoning=f"Network error during probe: {str(e)}"
            )
            await self._save_strategy_to_intelligence(domain, strategy, is_new=True)
            return strategy

        # AI analysis with comprehensive context
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert web scraping analyst. Analyze websites to determine the optimal scraping strategy.

            ANALYSIS CRITERIA:

            🟢 SIMPLE_HTML (use basic HTTP + BeautifulSoup):
            - Static HTML sites where main content is in initial response
            - News sites, blogs, government sites, simple business pages
            - Minimal JavaScript, content visible in HTML source
            - Fast loading, server-side rendered content

            🟡 MODERATE_JS (use HTTP + Readability + AI extraction):
            - Sites with some JavaScript but main content extractable
            - Magazine-style sites, modern blogs with interactive elements
            - Content mostly in HTML but may need cleanup/enhancement
            - Moderate complexity, some client-side rendering

            🔴 HEAVY_JS (use expensive Firecrawl - last resort):
            - Single Page Applications (SPAs) where content is JavaScript-generated
            - Sites where content only appears after JS execution
            - Complex interactive platforms, heavy frameworks (React/Angular/Vue)
            - Anti-bot protection, complex dynamic loading

            DECISION FACTORS:
            1. HTML content analysis - Is meaningful content in the initial response?
            2. JavaScript complexity - How much JS is required for content?
            3. Site type identification - News vs SPA vs blog vs platform
            4. Performance indicators - Response time, content size
            5. Technical stack detection - Framework usage, rendering method

            COST AWARENESS:
            - SIMPLE_HTML: ~0.1 credits (very cheap)
            - MODERATE_JS: ~0.5 credits (cheap)  
            - HEAVY_JS: ~10 credits (expensive - avoid unless necessary)

            OUTPUT FORMAT:
            {{
              "complexity": "SIMPLE_HTML|MODERATE_JS|HEAVY_JS",
              "confidence": 0.0-1.0,
              "reasoning": "detailed analysis of why this complexity was chosen",
              "content_in_html": true/false,
              "javascript_dependency": "low|medium|high",
              "site_type": "news|blog|magazine|platform|spa|ecommerce|other",
              "estimated_restaurant_extractability": 0.0-1.0
            }}
            """),
            ("human", """
            Analyze this website for optimal scraping strategy:

            URL: {url}
            Domain: {domain}
            Title: {title}
            Description: {description}

            HTTP Response:
            - Status Code: {status_code}
            - Response Time: {response_time:.2f}s
            - Content Length: {content_length} chars

            HTML Analysis:
            - Has <script> tags: {has_scripts}
            - Script count: {script_count}
            - Has framework indicators: {has_frameworks}
            - Content preview: {content_preview}

            Technical Indicators:
            - Meta viewport: {has_viewport}
            - JSON-LD structured data: {has_structured_data}
            - Social media meta tags: {has_social_meta}
            """)
        ])

        # Analyze HTML structure
        soup = BeautifulSoup(html_preview, 'html.parser')

        # Extract technical indicators
        scripts = soup.find_all('script')
        has_scripts = len(scripts) > 0
        script_count = len(scripts)

        # Check for SPA/framework indicators
        framework_indicators = ['react', 'angular', 'vue', 'ember', 'backbone']
        has_frameworks = any(indicator in html_preview.lower() for indicator in framework_indicators)

        # Check for structured data
        has_structured_data = bool(soup.find('script', type='application/ld+json'))

        # Check for responsive design (often indicates modern sites)
        has_viewport = bool(soup.find('meta', attrs={'name': 'viewport'}))

        # Check for social media meta tags
        has_social_meta = bool(soup.find('meta', attrs={'property': 'og:title'}) or 
                              soup.find('meta', attrs={'name': 'twitter:title'}))

        # Get clean content preview
        content_preview = soup.get_text(separator=' ', strip=True)[:500]

        try:
            chain = analysis_prompt | self.analyzer
            response = await chain.ainvoke({
                "url": url,
                "domain": domain,
                "title": title,
                "description": description,
                "status_code": status_code,
                "response_time": response_time,
                "content_length": len(html_preview),
                "has_scripts": has_scripts,
                "script_count": script_count,
                "has_frameworks": has_frameworks,
                "content_preview": content_preview,
                "has_viewport": has_viewport,
                "has_structured_data": has_structured_data,
                "has_social_meta": has_social_meta
            })

            # Parse AI response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            analysis = json.loads(content.strip())

            # Map AI response to strategy
            complexity_str = analysis.get("complexity", "MODERATE_JS")
            confidence = analysis.get("confidence", 0.5)
            reasoning = analysis.get("reasoning", "AI analysis")

            # Additional confidence adjustment based on technical factors
            if analysis.get("content_in_html", True) and script_count < 5:
                if complexity_str == "HEAVY_JS":
                    # Override AI if content is clearly in HTML
                    complexity_str = "MODERATE_JS"
                    confidence = max(0.7, confidence)
                    reasoning += " [Override: Content found in HTML]"
                    self.stats["strategy_overrides"] += 1

            # Map to enum and determine scraper
            complexity_map = {
                "SIMPLE_HTML": (ScrapeComplexity.SIMPLE_HTML, "simple_http", 0.1),
                "MODERATE_JS": (ScrapeComplexity.MODERATE_JS, "enhanced_http", 0.5),
                "HEAVY_JS": (ScrapeComplexity.HEAVY_JS, "firecrawl", 10)
            }

            complexity, scraper_type, cost = complexity_map.get(
                complexity_str, 
                (ScrapeComplexity.MODERATE_JS, "enhanced_http", 0.5)
            )

            strategy = ScrapeStrategy(
                complexity=complexity,
                scraper_type=scraper_type,
                estimated_cost=cost,
                confidence=confidence,
                reasoning=reasoning
            )

            # Save this analysis to persistent storage for future use
            await self._save_strategy_to_intelligence(domain, strategy, is_new=True)

            return strategy

        except Exception as e:
            logger.warning(f"AI analysis failed for {url}: {e}")
            # Safe fallback
            strategy = ScrapeStrategy(
                complexity=ScrapeComplexity.MODERATE_JS,
                scraper_type="enhanced_http",
                estimated_cost=0.5,
                confidence=0.4,
                reasoning=f"AI analysis failed: {str(e)}, using safe fallback"
            )
            await self._save_strategy_to_intelligence(domain, strategy, is_new=True)
            return strategy

    async def _has_specialized_handler(self, url: str) -> bool:
        """Check if URL can be handled by specialized scraper"""
        try:
            if not self.specialized_scraper:
                self.specialized_scraper = EaterTimeoutSpecializedScraper(self.config)

            return self.specialized_scraper._find_handler(url) is not None
        except Exception:
            return False

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url

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
        logger.info("🧠 Intelligent URL Analysis Results:")

        total_cost = 0
        for complexity, urls in url_groups.items():
            count = len(urls)
            if count > 0:
                avg_cost = sum(url["scrape_strategy"].estimated_cost for url in urls) / count
                strategy_cost = sum(url["scrape_strategy"].estimated_cost for url in urls)
                total_cost += strategy_cost
                avg_confidence = sum(url["scrape_strategy"].confidence for url in urls) / count

                logger.info(f"  {complexity.value}: {count} URLs (avg cost: {avg_cost:.1f}, confidence: {avg_confidence:.2f})")

        logger.info(f"  💰 Estimated total cost: {total_cost:.1f} credits")
        logger.info(f"  🎯 Cache hits: {self.stats['cache_hits']}")
        logger.info(f"  💾 Database cache hits: {self.stats['database_cache_hits']}")
        logger.info(f"  🔄 AI analysis calls: {self.stats['ai_analysis_calls']}")

    async def _process_url_groups(self, url_groups: Dict) -> List[Dict]:
        """Process each group with the appropriate scraper"""
        all_results = []

        # Process in order of preference (cheapest first)
        processing_order = [
            ScrapeComplexity.SPECIALIZED,
            ScrapeComplexity.SIMPLE_HTML, 
            ScrapeComplexity.MODERATE_JS,
            ScrapeComplexity.HEAVY_JS,
            ScrapeComplexity.UNKNOWN
        ]

        for complexity in processing_order:
            if complexity not in url_groups:
                continue

            urls = url_groups[complexity]
            if not urls:
                continue

            logger.info(f"🔄 Processing {len(urls)} URLs with {complexity.value}")

            if complexity == ScrapeComplexity.SPECIALIZED:
                results = await self._process_specialized_urls(urls)
                self.stats["specialized_used"] += len(urls)

            elif complexity == ScrapeComplexity.SIMPLE_HTML:
                results = await self._process_simple_http_urls(urls)
                self.stats["simple_http_used"] += len(urls)

            elif complexity == ScrapeComplexity.MODERATE_JS:
                results = await self._process_enhanced_http_urls(urls)
                self.stats["enhanced_http_used"] += len(urls)

            elif complexity == ScrapeComplexity.HEAVY_JS:
                results = await self._process_firecrawl_urls(urls)
                self.stats["firecrawl_used"] += len(urls)

            else:  # UNKNOWN
                results = await self._process_unknown_urls(urls)

            all_results.extend(results)

        return all_results

    async def _process_specialized_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using specialized handlers"""
        if not self.specialized_scraper:
            self.specialized_scraper = EaterTimeoutSpecializedScraper(self.config)

        async with self.specialized_scraper:
            return await self.specialized_scraper.process_specialized_urls(urls)

    async def _process_simple_http_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using simple HTTP + BeautifulSoup"""
        results = []
        semaphore = asyncio.Semaphore(8)  # Higher concurrency for simple requests

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

                            # Smart content extraction
                            title = soup.title.text.strip() if soup.title else ""

                            # Remove unwanted elements
                            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                                tag.decompose()

                            # Get main content intelligently
                            main_content = (soup.find('main') or 
                                          soup.find('article') or 
                                          soup.find(class_=re.compile(r'content|article|post', re.I)) or
                                          soup.find('div', class_=re.compile(r'story|text|body', re.I)) or
                                          soup.body)

                            if main_content:
                                content_text = main_content.get_text(separator='\n\n', strip=True)
                            else:
                                content_text = soup.get_text(separator='\n\n', strip=True)

                            # AI extraction of restaurants
                            restaurants = await self._extract_restaurants_ai(content_text[:3000])

                            enhanced_result = result.copy()
                            enhanced_result.update({
                                "scraped_content": content_text[:4000],
                                "scraped_title": title,
                                "restaurants_found": restaurants,
                                "scraping_method": "simple_http",
                                "scraping_success": len(restaurants) > 0,
                                "source_info": {
                                    "name": self._extract_source_name(url),
                                    "url": url,
                                    "extraction_method": "simple_http"
                                }
                            })

                            return enhanced_result

                except Exception as e:
                    logger.warning(f"Simple HTTP scraping failed for {url}: {e}")

                # Return failed result
                result["scraping_failed"] = True
                result["scraping_method"] = "simple_http"
                return result

        tasks = [process_single_simple(result) for result in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _process_enhanced_http_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using HTTP + Readability + AI extraction"""
        results = []
        semaphore = asyncio.Semaphore(5)  # Moderate concurrency

        async def process_single_enhanced(result):
            async with semaphore:
                url = result.get("url", "")
                try:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        response = await client.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })

                        if response.status_code == 200:
                            # Use readability for content extraction
                            doc = Document(response.text)
                            readable_html = doc.summary()
                            title = doc.title()

                            # Parse and clean
                            soup = BeautifulSoup(readable_html, 'html.parser')
                            content_text = soup.get_text(separator='\n\n', strip=True)

                            # AI extraction
                            restaurants = await self._extract_restaurants_ai(content_text[:4000])

                            enhanced_result = result.copy()
                            enhanced_result.update({
                                "scraped_content": content_text[:5000],
                                "scraped_title": title or "",
                                "restaurants_found": restaurants,
                                "scraping_method": "enhanced_http",
                                "scraping_success": len(restaurants) > 0,
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
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _process_firecrawl_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process URLs using expensive Firecrawl"""
        logger.warning(f"💸 Using expensive Firecrawl for {len(urls)} URLs")
        for url_info in urls:
            logger.warning(f"  🔥 Firecrawl: {url_info.get('url', 'unknown')} - {url_info.get('scrape_strategy', {}).get('reasoning', 'no reason')}")

        return await self.firecrawl_scraper.scrape_search_results(urls)

    async def _process_unknown_urls(self, urls: List[Dict]) -> List[Dict]:
        """Process unknown URLs with adaptive fallback"""
        # Try enhanced HTTP first, fall back to Firecrawl if needed
        enhanced_results = await self._process_enhanced_http_urls(urls)

        # Check success rates and retry failures with Firecrawl if needed
        failed_urls = []
        successful_urls = []

        for result in enhanced_results:
            if (result.get("scraping_failed") or 
                len(result.get("restaurants_found", [])) == 0):
                failed_urls.append(result)
            else:
                successful_urls.append(result)

        if failed_urls and len(failed_urls) / len(urls) > 0.5:  # More than 50% failed
            logger.info(f"🔄 High failure rate, retrying {len(failed_urls)} URLs with Firecrawl")
            firecrawl_results = await self._process_firecrawl_urls(failed_urls)
            successful_urls.extend(firecrawl_results)
            self.stats["firecrawl_used"] += len(failed_urls)
        else:
            successful_urls.extend(failed_urls)  # Keep the failed attempts

        return successful_urls

    async def _extract_restaurants_ai(self, content: str) -> List[str]:
        """AI-powered restaurant extraction"""
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                Extract restaurant, cafe, bar, bistro, and food venue names from the content.
                Look for:
                - Restaurant names (proper nouns)
                - Cafe and coffee shop names  
                - Bar and pub names
                - Food venue names
                - Chef-owned establishments

                Return ONLY the names as a JSON array: ["Restaurant 1", "Restaurant 2"]
                Maximum 20 names. Focus on quality over quantity.
                """),
                ("human", "Content:\n{content}")
            ])

            chain = extraction_prompt | self.analyzer
            response = await chain.ainvoke({"content": content})

            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            restaurants = json.loads(content.strip())
            return restaurants[:20] if isinstance(restaurants, list) else []

        except Exception as e:
            logger.warning(f"AI restaurant extraction failed: {e}")
            return []

    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # Convert domain to readable name
            parts = domain.split('.')
            if len(parts) >= 2:
                name = parts[0].replace('-', ' ').replace('_', ' ').title()
                return name

            return domain.title()

        except Exception:
            return "Unknown Source"

    async def _update_domain_intelligence(self, results: List[Dict]):
        """Learn from scraping results and update persistent domain intelligence"""
        for result in results:
            domain = self._extract_domain(result.get("url", ""))
            scraping_success = result.get("scraping_success", False)
            restaurants_found = len(result.get("restaurants_found", []))
            method = result.get("scraping_method", "")
            content_length = len(result.get("scraped_content", ""))

            if domain in self.domain_intelligence:
                # Update existing intelligence with success/failure data
                success_info = {
                    'success': scraping_success,
                    'restaurants_found': restaurants_found,
                    'content_length': content_length
                }

                # Create a dummy strategy to trigger the update
                current_intelligence = self.domain_intelligence[domain]
                strategy = ScrapeStrategy(
                    complexity=ScrapeComplexity(current_intelligence['complexity']),
                    scraper_type=current_intelligence['scraper_type'],
                    estimated_cost=current_intelligence['cost'],
                    confidence=current_intelligence['confidence'],
                    reasoning=current_intelligence['reasoning']
                )

                await self._save_strategy_to_intelligence(domain, strategy, is_new=False, success_info=success_info)

                logger.debug(f"📚 Updated intelligence for {domain}: confidence={current_intelligence['confidence']:.2f}")

    async def _periodic_cleanup(self):
        """Periodically clean up old domain intelligence"""
        # Only run cleanup occasionally to avoid database overhead
        import random
        if random.random() < 0.1:  # 10% chance per scraping session
            try:
                deleted_count = cleanup_old_domain_intelligence(self.config, days_old=90, min_confidence=0.3)
                if deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {deleted_count} old domain intelligence records")
            except Exception as e:
                logger.warning(f"Error during domain intelligence cleanup: {e}")

    def _calculate_cost_savings(self):
        """Calculate cost savings vs all-Firecrawl approach"""
        total_urls = self.stats["total_processed"]
        if total_urls > 0:
            firecrawl_cost_if_all = total_urls * 10
            actual_firecrawl_cost = self.stats["firecrawl_used"] * 10
            other_costs = (self.stats["simple_http_used"] * 0.1 + 
                          self.stats["enhanced_http_used"] * 0.5)

            total_actual_cost = actual_firecrawl_cost + other_costs
            self.stats["total_cost_saved"] = firecrawl_cost_if_all - total_actual_cost

    def _log_final_stats(self):
        """Log comprehensive final statistics with persistent intelligence info"""
        stats = self.stats

        logger.info("🧠 INTELLIGENT ADAPTIVE SCRAPING RESULTS:")
        logger.info(f"  📊 URLs processed: {stats['total_processed']}")
        logger.info(f"  🆓 Specialized: {stats['specialized_used']} (FREE)")
        logger.info(f"  🟢 Simple HTTP: {stats['simple_http_used']} (VERY CHEAP)")
        logger.info(f"  🟡 Enhanced HTTP: {stats['enhanced_http_used']} (CHEAP)")
        logger.info(f"  🔥 Firecrawl: {stats['firecrawl_used']} (EXPENSIVE)")
        logger.info(f"  💰 Cost saved: ~{stats['total_cost_saved']:.1f} Firecrawl credits")
        logger.info(f"  ⏱️ Processing time: {stats['processing_time']:.2f}s")
        logger.info(f"  🎯 Cache hits: {stats['cache_hits']}")
        logger.info(f"  💾 Database cache hits: {stats['database_cache_hits']}")
        logger.info(f"  🤖 AI analysis calls: {stats['ai_analysis_calls']}")
        logger.info(f"  🔄 Strategy overrides: {stats['strategy_overrides']}")
        logger.info(f"  🧠 Domains learned this session: {stats['domains_learned_this_session']}")
        logger.info(f"  💾 Intelligence saves: {stats['intelligence_saves']}")

        if stats['total_processed'] > 0:
            firecrawl_percentage = (stats['firecrawl_used'] / stats['total_processed']) * 100
            cache_hit_rate = (stats['cache_hits'] / stats['total_processed']) * 100
            db_cache_rate = (stats['database_cache_hits'] / stats['total_processed']) * 100
            logger.info(f"  📊 Firecrawl usage: {firecrawl_percentage:.1f}% of total URLs")
            logger.info(f"  📚 Total cache hit rate: {cache_hit_rate:.1f}%")
            logger.info(f"  💾 Database cache hit rate: {db_cache_rate:.1f}%")

            # Learning effectiveness
            domains_learned = len(self.domain_intelligence)
            logger.info(f"  🧠 Total domains in intelligence: {domains_learned}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics including persistent intelligence"""
        base_stats = {
            **self.stats,
            "domains_learned": len(self.domain_intelligence),
            "learning_cache_sample": dict(list(self.domain_intelligence.items())[:5])
        }

        # Add database stats if available
        try:
            db_stats = get_domain_intelligence_stats(self.config)
            base_stats["database_intelligence_stats"] = db_stats
        except Exception as e:
            logger.warning(f"Could not get database intelligence stats: {e}")

        return base_stats

    def get_domain_intelligence(self) -> Dict[str, Any]:
        """Get the current domain intelligence cache"""
        return self.domain_intelligence.copy()

    def get_database_intelligence_stats(self) -> Dict[str, Any]:
        """Get statistics about the persistent domain intelligence"""
        try:
            return get_domain_intelligence_stats(self.config)
        except Exception as e:
            logger.error(f"Error getting database intelligence stats: {e}")
            return {}

    def clear_domain_cache(self):
        """Clear the in-memory domain intelligence cache (useful for testing)"""
        self.domain_intelligence.clear()
        logger.info("🧠 In-memory domain intelligence cache cleared")

    async def export_domain_intelligence(self, file_path: str = None) -> str:
        """Export domain intelligence to file for backup"""
        try:
            from utils.domain_intelligence import export_domain_intelligence
            return export_domain_intelligence(self.config, file_path)
        except Exception as e:
            logger.error(f"Error exporting domain intelligence: {e}")
            raise


# Legacy compatibility wrapper  
class WebScraper:
    """
    Drop-in replacement for existing WebScraper that uses intelligent analysis with persistent learning
    """

    def __init__(self, config):
        self.intelligent_scraper = IntelligentAdaptiveScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.intelligent_scraper.scrape_search_results(search_results)

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.intelligent_scraper.scrape_search_results(search_results)

    def get_stats(self) -> Dict[str, Any]:
        return self.intelligent_scraper.get_stats()

    def get_domain_intelligence(self) -> Dict[str, Any]:
        return self.intelligent_scraper.get_domain_intelligence()

    def get_database_intelligence_stats(self) -> Dict[str, Any]:
        return self.intelligent_scraper.get_database_intelligence_stats()

    def clear_domain_cache(self):
        return self.intelligent_scraper.clear_domain_cache()

    async def export_domain_intelligence(self, file_path: str = None) -> str:
        return await self.intelligent_scraper.export_domain_intelligence(file_path)