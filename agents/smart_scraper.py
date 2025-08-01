# agents/smart_scraper.py - SIMPLIFIED VERSION
"""
Simplified AI-Powered Smart Scraper with Domain Learning

Key simplifications:
- Minimal hardcoding (only examples in AI prompt)
- Simple domain intelligence table (domain, strategy, cost, success stats)
- No keyword counting (pages already pre-filtered for restaurant content)
- Learn over time to reduce AI calls
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
from readability import Document

from agents.specialized_scraper import EaterTimeoutSpecializedScraper
from agents.firecrawl_scraper import FirecrawlWebScraper
from agents.content_sectioning_agent import ContentSectioningAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.database import get_database

logger = logging.getLogger(__name__)

class ScrapingStrategy(Enum):
    """Scraping strategies aligned with database"""
    SPECIALIZED = "specialized"      # FREE - RSS/Sitemap
    SIMPLE_HTTP = "simple_http"      # 0.1 credits
    ENHANCED_HTTP = "enhanced_http"  # 0.5 credits  
    FIRECRAWL = "firecrawl"         # 10.0 credits

class SmartRestaurantScraper:
    """
    AI-Powered Smart Scraper with Simple Domain Learning

    Flow:
    1. Check specialized handlers first (free)
    2. Check domain intelligence cache (learned strategies)
    3. If no cache, use AI to classify strategy
    4. After scraping, update domain intelligence
    5. Over time, most domains become cached and skip AI
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()

        # AI analyzer for new domains only
        self.analyzer = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Lazy initialization
        self._specialized_scraper = None
        self._firecrawl_scraper = None
        self._content_sectioner = None

        # Simple stats tracking
        self.stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "ai_analysis_calls": 0,
            "domain_cache_hits": 0,
            "new_domains_learned": 0,
            "total_cost_estimate": 0.0,
            "cost_saved_vs_all_firecrawl": 0.0
        }

        # Simple AI prompt - no hardcoded domains, minimal complexity
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert at analyzing websites to determine the optimal scraping strategy.

Since these pages are already filtered for restaurant content, focus only on TECHNICAL requirements:

🟢 SIMPLE_HTTP (0.1 credits):
- Static HTML sites where content loads immediately
- Simple blogs, basic news sites
- Content clearly visible in initial HTML response

🟡 ENHANCED_HTTP (0.5 credits):  
- Modern magazine sites with some JavaScript
- Content in HTML but needs cleaning/extraction
- Professional publications with moderate complexity

🔴 FIRECRAWL (10.0 credits):
- Heavy JavaScript where content loads dynamically  
- React/Vue/Angular single-page applications
- Anti-bot protection or complex authentication
- Interactive sites requiring browser rendering

Examples (guidance only, analyze each site individually):
- Simple food blogs, government tourism → SIMPLE_HTTP
- Professional food magazines → ENHANCED_HTTP
- Complex interactive platforms, anti-bot sites → FIRECRAWL

Return only:
{{
    "strategy": "SIMPLE_HTTP|ENHANCED_HTTP|FIRECRAWL",
    "confidence": 0.0-1.0,
    "reasoning": "Brief technical explanation"
}}
            """),
            ("human", """
Analyze this URL for scraping strategy:

URL: {url}
Domain: {domain}
Title: {title}
Content preview: {content_preview}
JavaScript indicators: {js_indicators}

Focus on technical complexity, not content keywords.
            """)
        ])

    @property 
    def specialized_scraper(self):
        if self._specialized_scraper is None:
            self._specialized_scraper = EaterTimeoutSpecializedScraper(self.config)
        return self._specialized_scraper

    @property
    def firecrawl_scraper(self):
        if self._firecrawl_scraper is None:
            self._firecrawl_scraper = FirecrawlWebScraper(self.config)
        return self._firecrawl_scraper

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main entry point for smart scraping with domain learning"""

        start_time = time.time()
        logger.info(f"🧠 Smart scraping with domain learning: {len(search_results)} URLs")

        # Step 1: Classify URLs using cache + AI for new domains
        classified_urls = await self._classify_urls_smartly(search_results)

        # Step 2: Process each strategy group  
        all_results = []
        for strategy, urls in classified_urls.items():
            if not urls:
                continue

            logger.info(f"🔄 Processing {len(urls)} URLs with {strategy.value}")

            # Process URLs based on strategy
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

        # Step 3: Update domain intelligence based on results
        await self._update_domain_intelligence(all_results)

        # Step 4: Calculate final stats
        self.stats["total_processed"] += len(search_results)
        self._calculate_costs()

        processing_time = time.time() - start_time
        logger.info(f"✅ Smart scraping completed in {processing_time:.2f}s")
        self._log_performance()

        return all_results

    async def _classify_urls_smartly(self, search_results: List[Dict[str, Any]]) -> Dict[ScrapingStrategy, List[Dict[str, Any]]]:
        """Classify URLs using domain intelligence cache + AI for new domains"""

        classified = {strategy: [] for strategy in ScrapingStrategy}

        for result in search_results:
            url = result.get("url", "")
            if not url:
                continue

            domain = self._extract_domain(url)

            # Step 1: Check specialized handlers first (always free)
            if self._is_specialized_url(url):
                classified[ScrapingStrategy.SPECIALIZED].append(result)
                result["classification_source"] = "specialized"
                logger.debug(f"🎯 Specialized: {domain}")
                continue

            # Step 2: Check domain intelligence cache
            cached_strategy = await self._get_cached_strategy(domain)

            if cached_strategy:
                classified[cached_strategy].append(result)
                result["classification_source"] = "domain_cache"
                self.stats["domain_cache_hits"] += 1
                logger.debug(f"💾 Cache hit: {domain} → {cached_strategy.value}")
                continue

            # Step 3: AI analysis for new domains
            try:
                strategy = await self._analyze_with_ai(result)
                classified[strategy].append(result)
                result["classification_source"] = "ai_analysis"
                result["ai_strategy"] = strategy.value
                self.stats["ai_analysis_calls"] += 1
                logger.debug(f"🤖 AI analysis: {domain} → {strategy.value}")

            except Exception as e:
                logger.warning(f"AI analysis failed for {url}: {e}")
                # Safe fallback
                classified[ScrapingStrategy.ENHANCED_HTTP].append(result)
                result["classification_source"] = "fallback"

        return classified

    def _is_specialized_url(self, url: str) -> bool:
        """Check if URL can be handled by specialized scrapers"""
        try:
            return bool(self.specialized_scraper._find_handler(url))
        except:
            return False

    async def _get_cached_strategy(self, domain: str) -> Optional[ScrapingStrategy]:
        """Get cached strategy for domain if reliable enough"""

        try:
            # Simple SQL query to get domain strategy
            result = self.database.supabase.table('domain_intelligence')\
                .select('strategy, confidence, total_attempts')\
                .eq('domain', domain)\
                .execute()

            if result.data:
                data = result.data[0]
                confidence = data.get('confidence', 0)
                total_attempts = data.get('total_attempts', 0)

                # Only use cache if we have enough data and good confidence
                if total_attempts >= 2 and confidence >= 0.6:
                    strategy_name = data.get('strategy', '')
                    try:
                        return ScrapingStrategy(strategy_name)
                    except ValueError:
                        return None

            return None

        except Exception as e:
            logger.debug(f"Error getting cached strategy for {domain}: {e}")
            return None

    async def _analyze_with_ai(self, result: Dict[str, Any]) -> ScrapingStrategy:
        """Use AI to analyze URL for new domains"""

        url = result.get("url", "")
        domain = self._extract_domain(url)

        # Quick probe for technical indicators
        probe_data = await self._quick_probe(url)

        try:
            chain = self.analysis_prompt | self.analyzer
            response = await chain.ainvoke({
                "url": url,
                "domain": domain,
                "title": probe_data.get("title", "")[:100],
                "content_preview": probe_data.get("content_preview", "")[:300],
                "js_indicators": probe_data.get("js_indicators", "")
            })

            # Parse AI response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            ai_result = json.loads(content.strip())

            strategy_map = {
                "SIMPLE_HTTP": ScrapingStrategy.SIMPLE_HTTP,
                "ENHANCED_HTTP": ScrapingStrategy.ENHANCED_HTTP,
                "FIRECRAWL": ScrapingStrategy.FIRECRAWL
            }

            strategy = strategy_map.get(ai_result["strategy"], ScrapingStrategy.ENHANCED_HTTP)

            # Store AI reasoning for learning
            result["ai_reasoning"] = ai_result.get("reasoning", "")
            result["ai_confidence"] = ai_result.get("confidence", 0.7)

            return strategy

        except Exception as e:
            logger.warning(f"AI analysis parsing failed for {url}: {e}")
            return ScrapingStrategy.ENHANCED_HTTP

    async def _quick_probe(self, url: str) -> Dict[str, Any]:
        """Quick technical probe (no keyword counting)"""

        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RestaurantBot/1.0)"
                })

                if response.status_code != 200:
                    return {}

                soup = BeautifulSoup(response.text, 'html.parser')

                # Basic technical indicators
                scripts = soup.find_all('script')
                js_indicators = f"{len(scripts)} scripts"

                # Framework detection
                page_text = response.text.lower()
                if 'react' in page_text: js_indicators += ", React"
                if 'vue' in page_text: js_indicators += ", Vue" 
                if 'angular' in page_text: js_indicators += ", Angular"

                return {
                    "title": soup.title.text.strip() if soup.title else "",
                    "content_preview": soup.get_text()[:500],
                    "js_indicators": js_indicators
                }

        except Exception as e:
            logger.debug(f"Quick probe failed for {url}: {e}")
            return {}

    async def _update_domain_intelligence(self, results: List[Dict[str, Any]]):
        """Update domain intelligence after scraping"""

        for result in results:
            # Skip specialized URLs (they don't need learning)
            if result.get("classification_source") == "specialized":
                continue

            url = result.get("url", "")
            domain = self._extract_domain(url)
            success = result.get("scraping_success", False)
            strategy_used = result.get("scraping_method", "unknown")

            # Map strategy names to our enum values
            strategy_mapping = {
                "simple_http": "simple_http",
                "simple_http_deepseek": "simple_http", 
                "enhanced_http": "enhanced_http",
                "enhanced_http_deepseek": "enhanced_http",
                "firecrawl": "firecrawl",
                "v2_scrape_extract": "firecrawl",
                "v2_basic_scrape": "firecrawl"
            }

            clean_strategy = strategy_mapping.get(strategy_used, "enhanced_http")

            try:
                # Use the simple SQL function to update stats
                self.database.supabase.rpc('update_domain_stats', {
                    'p_domain': domain,
                    'p_strategy': clean_strategy,
                    'p_success': success
                }).execute()

                logger.debug(f"🧠 Updated domain stats: {domain} ({clean_strategy}, success={success})")

                # Track new domains learned
                if result.get("classification_source") == "ai_analysis":
                    self.stats["new_domains_learned"] += 1

            except Exception as e:
                logger.warning(f"Failed to update domain intelligence for {domain}: {e}")

    async def _process_specialized(self, urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process with specialized handlers (FREE)"""
        logger.info(f"🆓 Specialized processing: {len(urls)} URLs (NO COST)")

        async with self.specialized_scraper:
            results = await self.specialized_scraper.process_specialized_urls(urls)

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
        """Process with Firecrawl (10.0 credits each)"""
        logger.warning(f"💸 Firecrawl processing: {len(urls)} URLs (~{len(urls) * 10} credits)")

        return await self.firecrawl_scraper.scrape_search_results(urls)

    def _extract_source_name(self, url: str) -> str:
        """Extract readable source name from URL"""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # Known source mappings
            source_mapping = {
                'timeout.com': 'Time Out',
                'eater.com': 'Eater',
                'cntraveler.com': 'Condé Nast Traveler',
                'guide.michelin.com': 'Michelin Guide',
                'foodandwine.com': 'Food & Wine',
                'bonappetit.com': 'Bon Appétit',
                'theinfatuation.com': 'The Infatuation',
                'zagat.com': 'Zagat',
                'sortiraparis.com': 'Sortir à Paris',
                'secretdeparis.com': 'Secret de Paris',
                'myparisianlife.com': 'My Parisian Life'
            }

            for domain_part, source_name in source_mapping.items():
                if domain_part in domain:
                    return source_name

            # Fallback to cleaned domain name
            return domain.replace('.com', '').replace('.fr', '').title()

        except Exception as e:
            logger.warning(f"Failed to extract source name from {url}: {e}")
            return "Unknown Source"

    async def _apply_content_sectioning(self, content: str, url: str, source_method: str):
        """
        Apply content sectioning if available, otherwise use simple truncation.
        """
        try:
            if hasattr(self, 'content_sectioner'):
                # Use DeepSeek-powered content sectioning for 90% speed improvement
                sectioning_result = await self.content_sectioner.process_content(
                    content, url, source_method
                )
                return sectioning_result
            else:
                # Fallback to simple truncation if content sectioner not available
                from agents.content_sectioning_agent import SectioningResult
                max_length = 6000  # Default limit
                truncated_content = content[:max_length] if len(content) > max_length else content

                return SectioningResult(
                    optimized_content=truncated_content,
                    original_length=len(content),
                    optimized_length=len(truncated_content),
                    sections_identified=["simple_truncation"],
                    restaurants_density=0.0,
                    sectioning_method="simple_truncation",
                    confidence=0.5
                )
        except Exception as e:
            logger.error(f"Content sectioning failed for {url}: {e}")
            # Simple fallback
            max_length = 6000
            truncated_content = content[:max_length] if len(content) > max_length else content

            # Create a minimal result object
            class SimpleResult:
                def __init__(self, content):
                    self.optimized_content = content

            return SimpleResult(truncated_content)

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return domain.replace('www.', '') if domain.startswith('www.') else domain
        except:
            return url

    def _calculate_costs(self):
        """Calculate cost statistics"""
        costs = {
            "specialized": 0.0,
            "simple_http": 0.1, 
            "enhanced_http": 0.5,
            "firecrawl": 10.0
        }

        total_cost = sum(
            count * costs.get(strategy, 0)
            for strategy, count in self.stats["strategy_breakdown"].items()
        )

        total_urls = self.stats["total_processed"]
        all_firecrawl_cost = total_urls * 10.0

        self.stats["total_cost_estimate"] = total_cost
        self.stats["cost_saved_vs_all_firecrawl"] = all_firecrawl_cost - total_cost

    def _log_performance(self):
        """Log performance summary"""
        stats = self.stats

        logger.info("🧠 SMART SCRAPING PERFORMANCE:")
        logger.info(f"  📊 Total URLs: {stats['total_processed']}")
        logger.info(f"  🤖 AI calls: {stats['ai_analysis_calls']}")
        logger.info(f"  💾 Cache hits: {stats['domain_cache_hits']}")
        logger.info(f"  📚 New domains learned: {stats['new_domains_learned']}")

        for strategy, count in stats["strategy_breakdown"].items():
            if count > 0:
                cost_map = {"specialized": 0, "simple_http": 0.1, "enhanced_http": 0.5, "firecrawl": 10}
                total_cost = count * cost_map.get(strategy, 0)
                emoji = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}
                logger.info(f"  {emoji.get(strategy, '📌')} {strategy}: {count} URLs (~{total_cost:.1f} credits)")

        logger.info(f"  💰 Total cost: {stats['total_cost_estimate']:.1f} credits")
        logger.info(f"  💾 Saved vs all-Firecrawl: {stats['cost_saved_vs_all_firecrawl']:.1f} credits")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            **self.stats,
            "cache_efficiency": {
                "hit_rate": self.stats["domain_cache_hits"] / max(self.stats["total_processed"], 1),
                "ai_usage_rate": self.stats["ai_analysis_calls"] / max(self.stats["total_processed"], 1)
            }
        }

# Legacy compatibility wrapper
class WebScraper:
    def __init__(self, config):
        self.scraper = SmartRestaurantScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.scraper.scrape_search_results(search_results)

    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.scraper.scrape_search_results(search_results)

    def get_stats(self) -> Dict[str, Any]:
        return self.scraper.get_stats()