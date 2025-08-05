# agents/smart_scraper.py - FIXED VERSION
"""
FIXED: Smart Scraper with Corrected LangChain Prompt Template

Issue Fixed: The analysis_prompt had unescaped curly braces in the JSON example,
which LangChain was interpreting as template variables instead of literal JSON.
All JSON examples in prompts now use double curly braces {{}} for escaping.
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
    FIXED: Proper LangChain prompt template escaping

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

        # FIXED: Proper LangChain prompt template with escaped JSON
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
        You are an expert web crawling strategist. Your task is to recommend the optimal scraping strategy based **purely on technical complexity** of the target page, NOT its content topic or genre.

        Available strategies:

        🟢 SIMPLE_HTTP (0.1 credits):
        - Static HTML pages where main content is fully available in the initial server response
        - Minimal or no JavaScript required to render content
        - No complex lazy-loading or client-side API calls

        🟡 ENHANCED_HTTP (0.5 credits):  
        - Pages with moderate client-side scripting
        - Content mostly in HTML but requires cleaning or minor JavaScript execution
        - Typical modern news or magazine sites using CMS platforms with some dynamic elements

        🔴 FIRECRAWL (10.0 credits):
        - Heavy JavaScript rendering (React, Vue, Angular SPA)
        - Content not visible in raw HTML (requires headless browser to load)
        - Infinite scroll, API-driven loading, or anti-bot protections (CAPTCHA, bot detection, auth walls)
        - Sites with significant client-side hydration or interactive UI

        ---

        Analyze these **technical markers**:
        - Presence and count of `<script>` tags, inline JS
        - Detected frameworks (React, Vue, Angular)
        - Whether text content is present in initial HTML or only minimal placeholders
        - Lazy-loading indicators (`data-src`, infinite scroll scripts)
        - Complex or obfuscated DOM structure
        - Any signs of bot-blocking or login walls

        Output strictly in JSON format. Use this EXACT structure:

        {{
            "strategy": "SIMPLE_HTTP|ENHANCED_HTTP|FIRECRAWL",
            "confidence": 0.0-1.0,
            "reasoning": "Technical explanation of why this strategy is required"
        }}
            """),
            ("human", """
        Analyze this URL and return the scraping strategy based on technical complexity only:

        URL: {{url}}
        Domain: {{domain}}
        Title: {{title}}
        Content preview: {{content_preview}}
        JavaScript indicators: {{js_indicators}}

        Focus on technical factors (rendering, JS execution, complexity), ignore content topic or keywords.
            """)
        ])

        logger.info("✅ FIXED SmartRestaurantScraper initialized with corrected prompt")

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

    def get_stats(self) -> Dict[str, Any]:
        """Get current scraping statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "ai_analysis_calls": 0,
            "domain_cache_hits": 0,
            "new_domains_learned": 0,
            "total_cost_estimate": 0.0,
            "cost_saved_vs_all_firecrawl": 0.0
        }

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point: Scrape search results using intelligent strategy selection
        """
        if not search_results:
            return []

        start_time = time.time()

        # Reset stats for this batch
        batch_stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "ai_analysis_calls": 0,
            "domain_cache_hits": 0,
            "new_domains_learned": 0,
            "total_cost_estimate": 0.0,
            "cost_saved_vs_all_firecrawl": 0.0
        }

        logger.info(f"🤖 Smart scraper starting for {len(search_results)} URLs")

        try:
            # Step 1: Classify all URLs by strategy
            classified_results = await self._classify_all_urls(search_results, batch_stats)

            # Step 2: Execute scraping with the determined strategies
            enriched_results = await self._execute_scraping(classified_results, batch_stats)

            # Step 3: Update domain intelligence
            await self._update_domain_intelligence(enriched_results)

            # Update global stats
            self._merge_stats(batch_stats)

            processing_time = time.time() - start_time

            logger.info(f"✅ Smart scraper completed in {processing_time:.1f}s")
            logger.info(f"💰 Cost estimate: {batch_stats['total_cost_estimate']:.1f} credits")
            logger.info(f"💾 Cost saved: {batch_stats['cost_saved_vs_all_firecrawl']:.1f} credits")

            return enriched_results

        except Exception as e:
            logger.error(f"❌ Smart scraper error: {e}")
            return []

    async def _classify_all_urls(self, search_results: List[Dict[str, Any]], stats: Dict) -> List[Dict[str, Any]]:
        """Classify all URLs by optimal scraping strategy"""

        classified_results = []

        for result in search_results:
            url = result.get("url", "")
            domain = self._extract_domain(url)

            # Step 1: Check specialized handlers first (free)
            if await self._is_specialized_url(url):
                result["scrape_strategy"] = ScrapingStrategy.SPECIALIZED
                result["classification_source"] = "specialized"
                classified_results.append(result)
                continue

            # Step 2: Check domain cache
            cached_strategy = await self._get_cached_strategy(domain)
            if cached_strategy:
                result["scrape_strategy"] = cached_strategy
                result["classification_source"] = "cache"
                stats["domain_cache_hits"] += 1
                classified_results.append(result)
                continue

            # Step 3: Use AI to classify (new domains only)
            ai_strategy = await self._analyze_with_ai(result)
            result["scrape_strategy"] = ai_strategy
            result["classification_source"] = "ai"
            stats["ai_analysis_calls"] += 1
            stats["new_domains_learned"] += 1

            classified_results.append(result)

        return classified_results

    async def _execute_scraping(self, classified_results: List[Dict[str, Any]], stats: Dict) -> List[Dict[str, Any]]:
        """Execute scraping using the determined strategies"""

        enriched_results = []

        # Group by strategy for efficient batch processing
        strategy_groups = {}
        for result in classified_results:
            strategy = result.get("scrape_strategy", ScrapingStrategy.ENHANCED_HTTP)
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(result)

        # Process each strategy group
        for strategy, results in strategy_groups.items():
            logger.info(f"📊 Processing {len(results)} URLs with {strategy.value}")

            if strategy == ScrapingStrategy.SPECIALIZED:
                scraped = await self._scrape_specialized(results)
            elif strategy == ScrapingStrategy.SIMPLE_HTTP:
                scraped = await self._scrape_simple_http(results)
            elif strategy == ScrapingStrategy.ENHANCED_HTTP:
                scraped = await self._scrape_enhanced_http(results)
            elif strategy == ScrapingStrategy.FIRECRAWL:
                scraped = await self._scrape_firecrawl(results)
            else:
                scraped = results  # Fallback

            enriched_results.extend(scraped)

            # Update stats
            stats["strategy_breakdown"][strategy.value] += len(results)
            stats["total_processed"] += len(results)

            # Cost calculation
            cost_per_url = {
                ScrapingStrategy.SPECIALIZED: 0.0,
                ScrapingStrategy.SIMPLE_HTTP: 0.1,
                ScrapingStrategy.ENHANCED_HTTP: 0.5,
                ScrapingStrategy.FIRECRAWL: 10.0
            }

            cost = len(results) * cost_per_url.get(strategy, 0.5)
            stats["total_cost_estimate"] += cost

        # Calculate cost savings (vs all Firecrawl)
        total_urls = stats["total_processed"]
        all_firecrawl_cost = total_urls * 10.0
        stats["cost_saved_vs_all_firecrawl"] = all_firecrawl_cost - stats["total_cost_estimate"]

        return enriched_results

    async def _is_specialized_url(self, url: str) -> bool:
        """Check if URL can be handled by specialized scrapers"""
        try:
            return await self.specialized_scraper.can_handle(url)
        except Exception as e:
            logger.debug(f"Error checking specialized handler for {url}: {e}")
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

    async def _scrape_specialized(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape using specialized handlers (RSS/sitemap)"""
        enriched_results = []

        for result in results:
            try:
                scraped_data = await self.specialized_scraper.scrape(result["url"])
                result.update({
                    "scraping_method": "specialized",
                    "scraping_success": len(scraped_data.get("restaurants_found", [])) > 0,
                    "scraped_content": scraped_data.get("content", ""),
                    "restaurants_found": scraped_data.get("restaurants_found", [])
                })
                enriched_results.append(result)

            except Exception as e:
                logger.warning(f"Specialized scraping failed for {result['url']}: {e}")
                result.update({
                    "scraping_method": "specialized",
                    "scraping_success": False,
                    "scraped_content": "",
                    "restaurants_found": []
                })
                enriched_results.append(result)

        return enriched_results

    async def _scrape_simple_http(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape using simple HTTP requests"""
        enriched_results = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for result in results:
                try:
                    response = await client.get(result["url"], headers={
                        "User-Agent": "Mozilla/5.0 (compatible; RestaurantBot/1.0)"
                    })

                    if response.status_code == 200:
                        # Use readability to extract clean content
                        doc = Document(response.text)
                        clean_content = doc.summary()

                        result.update({
                            "scraping_method": "simple_http",
                            "scraping_success": len(clean_content) > 200,
                            "scraped_content": clean_content[:6000],  # Limit content size
                            "restaurants_found": []  # Will be populated by content sectioner
                        })
                    else:
                        result.update({
                            "scraping_method": "simple_http",
                            "scraping_success": False,
                            "scraped_content": "",
                            "restaurants_found": []
                        })

                    enriched_results.append(result)

                except Exception as e:
                    logger.warning(f"Simple HTTP scraping failed for {result['url']}: {e}")
                    result.update({
                        "scraping_method": "simple_http",
                        "scraping_success": False,
                        "scraped_content": "",
                        "restaurants_found": []
                    })
                    enriched_results.append(result)

        return enriched_results

    async def _scrape_enhanced_http(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape using enhanced HTTP with content sectioning"""
        # First do simple HTTP scraping
        enriched_results = await self._scrape_simple_http(results)

        # Then enhance with content sectioning for successful scrapes
        for result in enriched_results:
            if result.get("scraping_success") and result.get("scraped_content"):
                try:
                    # Use content sectioner to extract restaurants
                    sectioned_content = await self.content_sectioner.section_content(
                        result["scraped_content"],
                        result["url"]
                    )

                    result.update({
                        "scraping_method": "enhanced_http",
                        "restaurants_found": sectioned_content.get("restaurants_found", []),
                        "scraped_content": sectioned_content.get("content", result["scraped_content"])
                    })

                except Exception as e:
                    logger.warning(f"Content sectioning failed for {result['url']}: {e}")
                    result["scraping_method"] = "enhanced_http"

        return enriched_results

    async def _scrape_firecrawl(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape using Firecrawl for JavaScript-heavy sites"""
        enriched_results = []

        for result in results:
            try:
                scraped_data = await self.firecrawl_scraper.scrape_url(result["url"])

                result.update({
                    "scraping_method": "firecrawl",
                    "scraping_success": scraped_data.get("success", False),
                    "scraped_content": scraped_data.get("content", ""),
                    "restaurants_found": scraped_data.get("restaurants_found", [])
                })
                enriched_results.append(result)

            except Exception as e:
                logger.warning(f"Firecrawl scraping failed for {result['url']}: {e}")
                result.update({
                    "scraping_method": "firecrawl",
                    "scraping_success": False,
                    "scraped_content": "",
                    "restaurants_found": []
                })
                enriched_results.append(result)

        return enriched_results

    async def _update_domain_intelligence(self, results: List[Dict[str, Any]]):
        """Update domain intelligence after scraping"""

        for result in results:
            # Skip specialized URLs (they don't need learning)
            if result.get("classification_source") == "specialized":
                continue

            url = result.get("url", "")
            domain = self._extract_domain(url)
            success = result.get("scraping_success", False)
            strategy = result.get("scrape_strategy")

            if not domain or not strategy:
                continue

            try:
                # Get current domain data
                current_data = self.database.supabase.table('domain_intelligence')\
                    .select('*')\
                    .eq('domain', domain)\
                    .execute()

                if current_data.data:
                    # Update existing record
                    existing = current_data.data[0]
                    total_attempts = existing.get('total_attempts', 0) + 1
                    successful_attempts = existing.get('successful_attempts', 0) + (1 if success else 0)
                    confidence = successful_attempts / total_attempts if total_attempts > 0 else 0

                    self.database.supabase.table('domain_intelligence')\
                        .update({
                            'total_attempts': total_attempts,
                            'successful_attempts': successful_attempts,
                            'confidence': confidence,
                            'last_used': 'now()',
                            'strategy': strategy.value
                        })\
                        .eq('domain', domain)\
                        .execute()
                else:
                    # Create new record
                    self.database.supabase.table('domain_intelligence')\
                        .insert({
                            'domain': domain,
                            'strategy': strategy.value,
                            'total_attempts': 1,
                            'successful_attempts': 1 if success else 0,
                            'confidence': 1.0 if success else 0.0,
                            'last_used': 'now()'
                        })\
                        .execute()

            except Exception as e:
                logger.debug(f"Error updating domain intelligence for {domain}: {e}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except:
            return ""

    def _merge_stats(self, batch_stats: Dict):
        """Merge batch stats into global stats"""
        for key, value in batch_stats.items():
            if key == "strategy_breakdown":
                for strategy, count in value.items():
                    self.stats[key][strategy] += count
            elif isinstance(value, (int, float)):
                self.stats[key] += value