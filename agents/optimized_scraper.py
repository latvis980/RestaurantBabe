# agents/optimized_scraper_enhanced.py
"""
Enhanced version of the optimized scraper with intelligent content sectioning.
This version integrates the ContentSectioningAgent into the existing scraper pipeline.
"""

import asyncio
import logging
import time
import httpx
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from enum import Enum
from dataclasses import dataclass

from bs4 import BeautifulSoup
from readability import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import existing components
from agents.firecrawl_scraper import FirecrawlWebScraper
from agents.specialized_scraper import EaterTimeoutSpecializedScraper
from agents.content_sectioning_agent import ContentSectioningAgent
# Removed database utils dependency for now

logger = logging.getLogger(__name__)

class ScrapeComplexity(Enum):
    SPECIALIZED = "specialized"
    SIMPLE_HTML = "simple_html"
    MODERATE_JS = "moderate_js"
    HEAVY_JS = "heavy_js"
    UNKNOWN = "unknown"

@dataclass
class ScrapeStrategy:
    complexity: ScrapeComplexity
    confidence: float
    reasoning: str
    estimated_cost: float

class EnhancedOptimizedScraper:
    """
    Enhanced intelligent scraper with AI-powered content sectioning.
    Integrates seamlessly with existing architecture while adding smart content optimization.
    """

    def __init__(self, config):
        self.config = config

        # Initialize existing components
        self.analyzer = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        self.firecrawl_scraper = FirecrawlWebScraper(config)
        self.specialized_scraper = None  # Lazy initialization

        # NEW: Initialize content sectioning agent
        self.content_sectioner = ContentSectioningAgent(config)

        # Enhanced statistics tracking
        self.stats = {
            "total_processed": 0,
            "specialized_used": 0,
            "simple_http_used": 0,
            "enhanced_http_used": 0,
            "firecrawl_used": 0,
            "total_cost_saved": 0.0,
            "processing_time": 0.0,
            "ai_analysis_calls": 0,
            "cache_hits": 0,
            "database_cache_hits": 0,
            "strategy_overrides": 0,
            # NEW: Content sectioning stats
            "content_sectioning_used": 0,
            "average_content_improvement": 0.0,
            "sectioning_time": 0.0
        }

        # Strategy analysis prompt (existing)
        self.strategy_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze websites to determine the optimal scraping strategy.

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

    async def scrape_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """
        Enhanced main scraping method with intelligent content sectioning.
        """
        start_time = time.time()
        self.stats["total_processed"] += len(search_results)

        logger.info(f"🧠 Enhanced intelligent scraping: {len(search_results)} URLs")

        # Group URLs by complexity using existing logic
        complexity_groups = await self._analyze_and_group_urls(search_results)

        all_results = []

        # Process each complexity group
        for complexity, urls in complexity_groups.items():
            if not urls:
                continue

            logger.info(f"🔄 Processing {len(urls)} URLs with {complexity.value}")

            if complexity == ScrapeComplexity.SPECIALIZED:
                results = await self._process_specialized_urls(urls)
                self.stats["specialized_used"] += len(urls)

            elif complexity == ScrapeComplexity.SIMPLE_HTML:
                results = await self._process_simple_http_urls_enhanced(urls)  # Enhanced version
                self.stats["simple_http_used"] += len(urls)

            elif complexity == ScrapeComplexity.MODERATE_JS:
                results = await self._process_enhanced_http_urls_enhanced(urls)  # Enhanced version
                self.stats["enhanced_http_used"] += len(urls)

            elif complexity == ScrapeComplexity.HEAVY_JS:
                results = await self._process_firecrawl_urls_enhanced(urls)  # Enhanced version
                self.stats["firecrawl_used"] += len(urls)

            else:  # UNKNOWN
                results = await self._process_unknown_urls_enhanced(urls)  # Enhanced version

            all_results.extend(results)

        # Calculate final stats
        self.stats["processing_time"] = time.time() - start_time
        self._calculate_cost_savings()
        self._log_enhanced_stats()

        return all_results

    async def _process_simple_http_urls_enhanced(self, urls: List[Dict]) -> List[Dict]:
        """Enhanced simple HTTP processing with content sectioning"""
        results = []
        semaphore = asyncio.Semaphore(8)

        async def process_single_simple_enhanced(result):
            async with semaphore:
                url = result.get("url", "")
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })

                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Smart content extraction (existing logic)
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

                            # NEW: Apply intelligent content sectioning instead of hard truncation
                            sectioning_result = await self._apply_content_sectioning(
                                content_text, url, "simple_http"
                            )

                            # AI extraction of restaurants using optimized content
                            restaurants = await self._extract_restaurants_ai(sectioning_result.optimized_content)

                            enhanced_result = result.copy()
                            enhanced_result.update({
                                "scraped_content": sectioning_result.optimized_content,
                                "scraped_title": title,
                                "restaurants_found": restaurants,
                                "scraping_method": "simple_http",
                                "scraping_success": len(restaurants) > 0,
                                "source_info": {
                                    "name": self._extract_source_name(url),
                                    "url": url,
                                    "extraction_method": "simple_http_enhanced"
                                },
                                # NEW: Add sectioning metadata
                                "sectioning_result": {
                                    "original_length": sectioning_result.original_length,
                                    "optimized_length": sectioning_result.optimized_length,
                                    "sections_identified": sectioning_result.sections_identified,
                                    "restaurant_density": sectioning_result.restaurants_density,
                                    "sectioning_method": sectioning_result.sectioning_method,
                                    "confidence": sectioning_result.confidence
                                }
                            })

                            return enhanced_result

                except Exception as e:
                    logger.warning(f"Enhanced simple HTTP scraping failed for {url}: {e}")

                # Return failed result
                result["scraping_failed"] = True
                result["scraping_method"] = "simple_http"
                return result

        tasks = [process_single_simple_enhanced(result) for result in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _process_enhanced_http_urls_enhanced(self, urls: List[Dict]) -> List[Dict]:
        """Enhanced HTTP + Readability processing with content sectioning"""
        results = []
        semaphore = asyncio.Semaphore(5)

        async def process_single_enhanced_enhanced(result):
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

                            # NEW: Apply intelligent content sectioning
                            sectioning_result = await self._apply_content_sectioning(
                                content_text, url, "enhanced_http"
                            )

                            # AI extraction using optimized content
                            restaurants = await self._extract_restaurants_ai(sectioning_result.optimized_content)

                            enhanced_result = result.copy()
                            enhanced_result.update({
                                "scraped_content": sectioning_result.optimized_content,
                                "scraped_title": title or "",
                                "restaurants_found": restaurants,
                                "scraping_method": "enhanced_http",
                                "scraping_success": len(restaurants) > 0,
                                "source_info": {
                                    "name": self._extract_source_name(url),
                                    "url": url,
                                    "extraction_method": "enhanced_http_readability_sectioned"
                                },
                                "sectioning_result": {
                                    "original_length": sectioning_result.original_length,
                                    "optimized_length": sectioning_result.optimized_length,
                                    "sections_identified": sectioning_result.sections_identified,
                                    "restaurant_density": sectioning_result.restaurants_density,
                                    "sectioning_method": sectioning_result.sectioning_method,
                                    "confidence": sectioning_result.confidence
                                }
                            })

                            return enhanced_result

                except Exception as e:
                    logger.warning(f"Enhanced HTTP scraping failed for {url}: {e}")

                result["scraping_failed"] = True
                result["scraping_method"] = "enhanced_http"
                return result

        tasks = [process_single_enhanced_enhanced(result) for result in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _process_firecrawl_urls_enhanced(self, urls: List[Dict]) -> List[Dict]:
        """Enhanced Firecrawl processing with content sectioning"""
        logger.warning(f"💸 Using expensive Firecrawl for {len(urls)} URLs with enhanced processing")

        # Use existing Firecrawl scraper but enhance the results
        firecrawl_results = await self.firecrawl_scraper.scrape_search_results(urls)

        enhanced_results = []
        for result in firecrawl_results:
            if result.get("scraping_success") and result.get("scraped_content"):
                try:
                    # Apply content sectioning to Firecrawl content
                    sectioning_result = await self._apply_content_sectioning(
                        result["scraped_content"], 
                        result.get("url", ""), 
                        "firecrawl"
                    )

                    # Update result with sectioned content
                    result["scraped_content"] = sectioning_result.optimized_content
                    result["sectioning_result"] = {
                        "original_length": sectioning_result.original_length,
                        "optimized_length": sectioning_result.optimized_length,
                        "sections_identified": sectioning_result.sections_identified,
                        "restaurant_density": sectioning_result.restaurants_density,
                        "sectioning_method": sectioning_result.sectioning_method,
                        "confidence": sectioning_result.confidence
                    }

                    # Re-extract restaurants with optimized content
                    if hasattr(result.get("source_info", {}), "extraction_method"):
                        result["source_info"]["extraction_method"] += "_sectioned"

                except Exception as e:
                    logger.warning(f"Content sectioning failed for Firecrawl result: {e}")
                    # Keep original Firecrawl result if sectioning fails

            enhanced_results.append(result)

        return enhanced_results

    async def _process_unknown_urls_enhanced(self, urls: List[Dict]) -> List[Dict]:
        """Enhanced unknown URL processing with adaptive fallback"""
        # Try enhanced HTTP first, fall back to Firecrawl if needed
        enhanced_results = await self._process_enhanced_http_urls_enhanced(urls)

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
            logger.info(f"🔄 High failure rate, retrying {len(failed_urls)} URLs with enhanced Firecrawl")
            firecrawl_results = await self._process_firecrawl_urls_enhanced(failed_urls)
            successful_urls.extend(firecrawl_results)
            self.stats["firecrawl_used"] += len(failed_urls)
        else:
            successful_urls.extend(failed_urls)  # Keep the failed attempts

        return successful_urls

    async def _apply_content_sectioning(
        self, 
        content: str, 
        url: str, 
        scraping_method: str
    ) -> 'SectioningResult':
        """
        Apply intelligent content sectioning to optimize restaurant extraction.
        This is the core enhancement that replaces hard truncation.
        """
        sectioning_start = time.time()

        try:
            # Determine target limit based on scraping method and content quality
            target_limit = self._calculate_dynamic_target_limit(content, scraping_method, url)

            # Apply content sectioning
            sectioning_result = await self.content_sectioner.section_content(
                content=content,
                target_limit=target_limit,
                min_restaurant_density=0.05  # Minimum 1 restaurant per 200 characters
            )

            # Update stats
            self.stats["content_sectioning_used"] += 1
            sectioning_time = time.time() - sectioning_start
            self.stats["sectioning_time"] += sectioning_time

            # Track improvement
            if sectioning_result.original_length > 0:
                improvement_ratio = sectioning_result.restaurants_density / max(
                    sectioning_result.original_length / len(sectioning_result.optimized_content), 0.1
                )
                self.stats["average_content_improvement"] = (
                    (self.stats["average_content_improvement"] * (self.stats["content_sectioning_used"] - 1) + improvement_ratio)
                    / self.stats["content_sectioning_used"]
                )

            logger.info(f"📊 Content sectioned in {sectioning_time:.2f}s: "
                       f"{sectioning_result.original_length} → {sectioning_result.optimized_length} chars "
                       f"({sectioning_result.sectioning_method})")

            return sectioning_result

        except Exception as e:
            logger.error(f"Content sectioning failed for {url}: {e}")
            # Fallback to smart truncation
            sectioning_time = time.time() - sectioning_start
            self.stats["sectioning_time"] += sectioning_time

            from agents.content_sectioning_agent import SectioningResult
            return SectioningResult(
                optimized_content=content[:6000],  # Conservative fallback
                original_length=len(content),
                optimized_length=min(len(content), 6000),
                sections_identified=["fallback"],
                restaurants_density=0.0,
                sectioning_method="fallback_truncation",
                confidence=0.3
            )

    def _calculate_dynamic_target_limit(self, content: str, scraping_method: str, url: str) -> int:
        """Calculate dynamic target limit based on content quality and source"""

        base_limits = {
            "simple_http": 6000,      # Increased from 4000
            "enhanced_http": 7000,    # Increased from 5000  
            "firecrawl": 8000,        # Increased from 4000 (maximize expensive Firecrawl value)
            "specialized": 10000      # High limit for specialized sources
        }

        base_limit = base_limits.get(scraping_method, 6000)

        # Adjust based on source quality
        domain = urlparse(url).netloc.lower()

        # High-quality sources get more space
        premium_domains = [
            'timeout.com', 'eater.com', 'cntraveller.com', 'theinfatuation.com',
            'bonappetit.com', 'foodandwine.com', 'saveur.com'
        ]

        if any(premium in domain for premium in premium_domains):
            base_limit = int(base_limit * 1.3)  # 30% more for premium sources

        # Adjust based on content characteristics
        content_length = len(content)

        if content_length > 20000:  # Very long articles
            base_limit = int(base_limit * 1.4)  # Allow more space for comprehensive guides
        elif content_length < 2000:  # Short articles
            base_limit = min(base_limit, content_length)  # Don't exceed original length

        return base_limit

    # Keep existing methods unchanged
    async def _analyze_and_group_urls(self, search_results: List[Dict]) -> Dict[ScrapeComplexity, List[Dict]]:
        """Existing URL analysis and grouping logic (unchanged)"""
        # This method stays exactly the same as your original
        complexity_groups = {complexity: [] for complexity in ScrapeComplexity}

        for result in search_results:
            url = result.get("url", "")
            if not url:
                continue

            # Check for specialized handlers first
            if self._is_specialized_url(url):
                complexity_groups[ScrapeComplexity.SPECIALIZED].append(result)
                continue

            # For others, analyze complexity (existing logic)
            try:
                strategy = await self._analyze_url_complexity(result)
                complexity_groups[strategy.complexity].append(result)
                result["scrape_strategy"] = strategy
            except Exception as e:
                logger.warning(f"Strategy analysis failed for {url}: {e}")
                complexity_groups[ScrapeComplexity.UNKNOWN].append(result)

        return complexity_groups

    def _is_specialized_url(self, url: str) -> bool:
        """Check if URL can be handled by specialized scraper"""
        specialized_domains = ['timeout.com', 'eater.com']
        return any(domain in url.lower() for domain in specialized_domains)

    # Enhanced AI Analysis System for Restaurant Sites

    async def _analyze_url_complexity(self, result: Dict) -> ScrapeStrategy:
        """
        Enhanced URL complexity analysis that focuses on content quality, not just technical indicators.
        Specifically designed for restaurant guide websites.
        """
        url = result.get("url", "")
        domain = self._extract_domain(url)

        try:
            # First, do a quick HTTP probe to analyze the actual content
            probe_result = await self._probe_url_content(url)

            if not probe_result:
                return ScrapeStrategy(
                    complexity=ScrapeComplexity.MODERATE_JS,
                    scraper_type="enhanced_http",
                    estimated_cost=0.5,
                    confidence=0.4,
                    reasoning=f"Failed to probe {domain}, using safe fallback"
                )

            # Analyze both technical indicators AND content quality
            analysis_data = {
                "url": url,
                "domain": domain,
                "title": probe_result.get("title", ""),
                "description": result.get("description", ""),
                "status_code": probe_result.get("status_code", 0),
                "response_time": probe_result.get("response_time", 0),
                "content_length": probe_result.get("content_length", 0),
                "has_scripts": probe_result.get("has_scripts", False),
                "script_count": probe_result.get("script_count", 0),
                "has_frameworks": probe_result.get("has_frameworks", False),
                "content_preview": probe_result.get("content_preview", ""),
                "has_viewport": probe_result.get("has_viewport", False),
                "has_structured_data": probe_result.get("has_structured_data", False),
                "has_social_meta": probe_result.get("has_social_meta", False),
                # NEW: Content quality indicators
                "restaurant_indicators_found": probe_result.get("restaurant_indicators", 0),
                "list_structure_detected": probe_result.get("has_list_structure", False),
                "navigation_heavy": probe_result.get("navigation_heavy", False),
                "content_to_navigation_ratio": probe_result.get("content_ratio", 0.0)
            }

            # Use enhanced AI prompt that considers content quality
            chain = self.enhanced_analysis_prompt | self.analyzer
            response = await chain.ainvoke(analysis_data)

            # Parse AI response
            result_text = response.content
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]

            analysis = json.loads(result_text.strip())

            # Convert to ScrapeStrategy
            complexity_map = {
                "SIMPLE_HTML": ScrapeComplexity.SIMPLE_HTML,
                "MODERATE_JS": ScrapeComplexity.MODERATE_JS, 
                "HEAVY_JS": ScrapeComplexity.HEAVY_JS
            }

            complexity = complexity_map.get(analysis.get("complexity", "MODERATE_JS"), ScrapeComplexity.MODERATE_JS)

            # Override low restaurant count sites to use Firecrawl
            if (probe_result.get("restaurant_indicators", 0) < 3 and 
                complexity != ScrapeComplexity.HEAVY_JS and
                "restaurant" in url.lower()):

                complexity = ScrapeComplexity.HEAVY_JS
                reasoning = f"OVERRIDE: {analysis.get('reasoning', '')} However, only {probe_result.get('restaurant_indicators', 0)} restaurant indicators found in initial probe, suggesting content may be dynamically loaded. Upgrading to Firecrawl."
            else:
                reasoning = analysis.get("reasoning", "AI analysis completed")

            scraper_type_map = {
                ScrapeComplexity.SIMPLE_HTML: "simple_http",
                ScrapeComplexity.MODERATE_JS: "enhanced_http",
                ScrapeComplexity.HEAVY_JS: "firecrawl"
            }

            cost_map = {
                ScrapeComplexity.SIMPLE_HTML: 0.1,
                ScrapeComplexity.MODERATE_JS: 0.5,
                ScrapeComplexity.HEAVY_JS: 10.0
            }

            strategy = ScrapeStrategy(
                complexity=complexity,
                scraper_type=scraper_type_map[complexity],
                estimated_cost=cost_map[complexity],
                confidence=analysis.get("confidence", 0.7),
                reasoning=reasoning
            )

            # Save this analysis to persistent storage
            await self._save_strategy_to_intelligence(domain, strategy, is_new=True)

            self.stats["ai_analysis_calls"] += 1

            return strategy

        except Exception as e:
            logger.warning(f"Enhanced AI analysis failed for {url}: {e}")
            # Safe fallback
            return ScrapeStrategy(
                complexity=ScrapeComplexity.MODERATE_JS,
                scraper_type="enhanced_http", 
                estimated_cost=0.5,
                confidence=0.4,
                reasoning=f"Enhanced AI analysis failed: {str(e)}, using safe fallback"
            )

    async def _probe_url_content(self, url: str) -> Optional[Dict]:
        """
        Probe URL to analyze actual content quality, not just technical indicators.
        This is key for detecting restaurant sites that load content dynamically.
        """
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

                # Extract basic info
                title = soup.title.text.strip() if soup.title else ""

                # Technical indicators
                scripts = soup.find_all('script')
                has_scripts = len(scripts) > 0
                script_count = len(scripts)

                # Check for frameworks
                html_text = response.text.lower()
                frameworks = ['react', 'angular', 'vue', 'ember', 'backbone']
                has_frameworks = any(fw in html_text for fw in frameworks)

                # Meta indicators
                has_viewport = bool(soup.find('meta', attrs={'name': 'viewport'}))
                has_structured_data = bool(soup.find('script', attrs={'type': 'application/ld+json'}))
                has_social_meta = bool(soup.find('meta', attrs={'property': 'og:title'}))

                # NEW: Content quality analysis
                text_content = soup.get_text(separator=' ', strip=True)
                content_preview = text_content[:500]

                # Count restaurant indicators in the visible content
                restaurant_keywords = [
                    'restaurant', 'menu', 'cuisine', 'chef', 'dining', 'food', 
                    'dish', 'meal', 'bistro', 'cafe', 'bar', 'eatery',
                    'address:', 'price:', '$', '€', 'rating', 'review'
                ]

                restaurant_indicators = sum(1 for keyword in restaurant_keywords 
                                          if keyword.lower() in text_content.lower())

                # Detect list structures (important for restaurant guides)
                list_elements = soup.find_all(['ol', 'ul']) + soup.find_all(class_=re.compile(r'list|item|entry'))
                has_list_structure = len(list_elements) > 2

                # Check if page is navigation-heavy (bad sign for content extraction)
                nav_elements = soup.find_all(['nav', 'header', 'footer']) + soup.find_all(class_=re.compile(r'nav|menu|header|footer'))
                nav_text_length = sum(len(elem.get_text(strip=True)) for elem in nav_elements)
                content_text_length = len(text_content)

                navigation_heavy = nav_text_length > (content_text_length * 0.3) if content_text_length > 0 else True
                content_ratio = (content_text_length - nav_text_length) / max(content_text_length, 1)

                return {
                    "title": title,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content_length": len(response.text),
                    "has_scripts": has_scripts,
                    "script_count": script_count,
                    "has_frameworks": has_frameworks,
                    "content_preview": content_preview,
                    "has_viewport": has_viewport,
                    "has_structured_data": has_structured_data,
                    "has_social_meta": has_social_meta,
                    # NEW: Content quality metrics
                    "restaurant_indicators": restaurant_indicators,
                    "has_list_structure": has_list_structure,
                    "navigation_heavy": navigation_heavy,
                    "content_ratio": content_ratio
                }

        except Exception as e:
            logger.warning(f"URL probe failed for {url}: {e}")
            return None

    # Enhanced AI prompt that considers content quality
    @property
    def enhanced_analysis_prompt(self):
        """Enhanced analysis prompt that focuses on content quality for restaurant sites"""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at analyzing restaurant guide websites for optimal scraping strategies.
            Your goal is to determine if a site can extract restaurant information effectively.

            CRITICAL ANALYSIS FACTORS:

            🎯 CONTENT QUALITY INDICATORS (MOST IMPORTANT):
            - Restaurant indicators found: How many restaurant-related terms appear?
            - List structure detected: Are restaurants organized in lists/grids?
            - Content-to-navigation ratio: Is the page mostly content or navigation?
            - Content preview: Does the preview show actual restaurant information?

            🔧 TECHNICAL INDICATORS (SECONDARY):
            - JavaScript complexity and framework usage
            - Response time and content length
            - Meta tags and structured data

            CLASSIFICATION RULES:

            🟢 SIMPLE_HTML: Use ONLY if:
            - Restaurant indicators > 5 AND
            - List structure detected AND
            - Content preview shows restaurant names/details AND
            - Low JavaScript dependency

            🟡 MODERATE_JS: Use if:
            - Some restaurant indicators (3-5) AND
            - Mixed content quality AND
            - Moderate JavaScript but content visible in HTML

            🔴 HEAVY_JS: Use if:
            - Few restaurant indicators (< 3) OR
            - Navigation-heavy page OR
            - Content preview lacks restaurant details OR
            - Heavy JavaScript frameworks detected

            RESTAURANT SITE RED FLAGS:
            - Content preview is mostly intro text without restaurant names
            - High navigation-to-content ratio
            - Few restaurant indicators despite being a restaurant guide
            - Dynamic loading indicators

            OUTPUT FORMAT:
            {{
              "complexity": "SIMPLE_HTML|MODERATE_JS|HEAVY_JS",
              "confidence": 0.0-1.0,
              "reasoning": "Focus on WHY this choice will succeed/fail for restaurant extraction",
              "content_in_html": true/false,
              "javascript_dependency": "low|medium|high",
              "site_type": "restaurant_guide|single_review|magazine|platform|other",
              "estimated_restaurant_extractability": 0.0-1.0
            }}
            """),
            ("human", """
            Analyze this restaurant website for optimal scraping strategy:

            URL: {url}
            Domain: {domain}
            Title: {title}
            Description: {description}

            TECHNICAL METRICS:
            - Status: {status_code}
            - Response Time: {response_time:.2f}s
            - Content Length: {content_length} chars
            - Scripts: {script_count} (has frameworks: {has_frameworks})
            - Meta: viewport={has_viewport}, structured_data={has_structured_data}, social={has_social_meta}

            CONTENT QUALITY METRICS:
            - Restaurant indicators found: {restaurant_indicators_found}
            - List structure detected: {list_structure_detected}
            - Navigation heavy: {navigation_heavy}
            - Content ratio: {content_to_navigation_ratio:.2f}

            CONTENT PREVIEW:
            {content_preview}

            Focus your analysis on: Will this approach successfully extract restaurant information?
            """)
        ])

    async def _process_specialized_urls(self, urls: List[Dict]) -> List[Dict]:
        """Existing specialized URL processing (unchanged)"""
        if not self.specialized_scraper:
            self.specialized_scraper = EaterTimeoutSpecializedScraper(self.config)

        async with self.specialized_scraper:
            return await self.specialized_scraper.process_specialized_urls(urls)

    async def _extract_restaurants_ai(self, content: str) -> List[str]:
        """Existing AI restaurant extraction (unchanged)"""
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

            import json
            restaurants = json.loads(content.strip())
            return restaurants[:20] if isinstance(restaurants, list) else []

        except Exception as e:
            logger.warning(f"AI restaurant extraction failed: {e}")
            return []

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

    async def _save_strategy_to_intelligence(self, domain: str, strategy: ScrapeStrategy, is_new: bool = False):
        """Save strategy to intelligence (placeholder)"""
        logger.info(f"📝 Domain analysis: {domain} → {strategy.complexity.value}")
        pass

    def _extract_source_name(self, url: str) -> str:
        """Existing source name extraction (unchanged)"""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # Convert domain to readable name
            parts = domain.split('.')
            if len(parts) >= 2:
                name = parts[0].replace('-', ' ').title()
                return name
            return domain
        except:
            return "Unknown Source"

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

    def _log_enhanced_stats(self):
        """Enhanced logging with content sectioning statistics"""
        stats = self.stats

        logger.info("🚀 ENHANCED INTELLIGENT SCRAPING RESULTS:")
        logger.info(f"  📊 URLs processed: {stats['total_processed']}")
        logger.info(f"  🆓 Specialized: {stats['specialized_used']} (FREE)")
        logger.info(f"  🟢 Simple HTTP: {stats['simple_http_used']} (Enhanced)")
        logger.info(f"  🟡 Enhanced HTTP: {stats['enhanced_http_used']} (Enhanced)")
        logger.info(f"  🔥 Firecrawl: {stats['firecrawl_used']} (Enhanced)")
        logger.info(f"  💰 Cost saved: ~{stats['total_cost_saved']:.1f} Firecrawl credits")
        logger.info(f"  ⏱️ Processing time: {stats['processing_time']:.2f}s")

        # NEW: Content sectioning stats
        if stats["content_sectioning_used"] > 0:
            logger.info("🧠 CONTENT SECTIONING ENHANCEMENT:")
            logger.info(f"  📝 Content sectioned: {stats['content_sectioning_used']} articles")
            logger.info(f"  ⚡ Sectioning time: {stats['sectioning_time']:.2f}s")
            logger.info(f"  📈 Average improvement: {stats['average_content_improvement']:.1f}x")

    def get_stats(self) -> Dict:
        """Get comprehensive scraping statistics"""
        return self.stats.copy()

    def get_sectioning_stats(self) -> Dict:
        """Get content sectioning specific statistics"""
        return self.content_sectioner.get_stats()

    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_processed": 0,
            "specialized_used": 0,
            "simple_http_used": 0,
            "enhanced_http_used": 0,
            "firecrawl_used": 0,
            "total_cost_saved": 0.0,
            "processing_time": 0.0,
            "ai_analysis_calls": 0,
            "cache_hits": 0,
            "database_cache_hits": 0,
            "strategy_overrides": 0,
            "content_sectioning_used": 0,
            "average_content_improvement": 0.0,
            "sectioning_time": 0.0
        }
        self.content_sectioner.reset_stats()


# Enhanced WebScraper wrapper with content sectioning
class WebScraper:
    """
    Drop-in replacement for existing WebScraper that uses enhanced scraping with content sectioning.
    Maintains full compatibility with existing code.
    """

    def __init__(self, config):
        self.enhanced_scraper = EnhancedOptimizedScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main scraping method with enhanced content sectioning"""
        return await self.enhanced_scraper.scrape_search_results(search_results)

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Alternative method name for compatibility"""
        return await self.enhanced_scraper.scrape_search_results(search_results)

    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        return self.enhanced_scraper.get_stats()

    def get_sectioning_stats(self) -> Dict[str, Any]:
        """Get content sectioning specific statistics"""
        return self.enhanced_scraper.get_sectioning_stats()

    # Legacy compatibility methods (if they exist in your original)
    def get_domain_intelligence(self) -> Dict[str, Any]:
        """Legacy compatibility - return empty dict"""
        return {}

    def get_database_intelligence_stats(self) -> Dict[str, Any]:
        """Legacy compatibility - return empty dict"""
        return {}

    def clear_domain_cache(self):
        """Legacy compatibility - no-op"""
        pass

    async def export_domain_intelligence(self, file_path: str = None) -> str:
        """Legacy compatibility - return empty string"""
        return ""


