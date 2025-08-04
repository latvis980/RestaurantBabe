# agents/smart_scraper.py - COMPLETELY REWORKED VERSION
"""
REWORKED Smart Scraper System with Enhanced Logic Flow

NEW FEATURES:
1. ✅ AI determines if page needs simple HTTP, enhanced HTTP, or Firecrawl
2. ✅ Enhanced HTTP scraping with sectioning as default 
3. ✅ Firecrawl only as last resort with single attempt
4. ✅ Domain intelligence saved AFTER successful scrape
5. ✅ Domain intelligence checked BEFORE AI filtering
6. ✅ Sectioning only for simple and enhanced scraping
7. ✅ Latest Firecrawl API implementation

FLOW:
1. Check domain intelligence table first
2. If domain exists, use cached strategy directly
3. If new domain, use AI to classify strategy needed
4. Try strategies in order: simple → enhanced → firecrawl (single attempt)
5. Save domain intelligence AFTER successful scrape
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
from utils.database_domain_intelligence import get_domain_intelligence_manager
from agents.firecrawl_scraper import FirecrawlWebScraper
from agents.content_sectioning_agent import ContentSectioningAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.database import get_database

logger = logging.getLogger(__name__)

manager = get_domain_intelligence_manager()

class ScrapingStrategy(Enum):
    """Scraping strategies aligned with database"""
    SPECIALIZED = "specialized"      # FREE - RSS/Sitemap
    SIMPLE_HTTP = "simple_http"      # 0.1 credits
    ENHANCED_HTTP = "enhanced_http"  # 0.5 credits  
    FIRECRAWL = "firecrawl"         # 10.0 credits

class SmartRestaurantScraper:
    """
    REWORKED AI-Powered Smart Scraper with Enhanced Domain Intelligence

    NEW FLOW:
    1. Check domain intelligence table first (bypass AI if domain known)
    2. Use AI only for new domains to classify strategy
    3. Try enhanced HTTP with sectioning as default approach
    4. Use Firecrawl only as last resort with single attempt
    5. Save domain intelligence AFTER successful scrape
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

        # Enhanced stats tracking
        self.stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "ai_analysis_calls": 0,
            "domain_cache_hits": 0,
            "new_domains_learned": 0,
            "total_cost_estimate": 0.0,
            "cost_saved_vs_all_firecrawl": 0.0,
            "sectioning_calls": 0,
            "firecrawl_attempts": 0,
            "firecrawl_success_rate": 0.0
        }

        # UPDATED: AI prompt for strategy classification
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert web scraping strategist. Analyze the URL and determine the optimal scraping strategy based on technical complexity.

STRATEGIES:
- simple_http: Static HTML pages, basic content sites, simple blogs
- enhanced_http: Moderate complexity sites that may need readability processing
- firecrawl: JavaScript-heavy sites, SPAs, sites with anti-bot protection

Focus on TECHNICAL indicators, not content type. Return ONLY the strategy name.

EXAMPLES:
- blog.example.com/restaurant-guide → simple_http
- timeout.com/restaurants → enhanced_http  
- resy.com/cities → firecrawl
- medium.com/@foodie/best-restaurants → enhanced_http
            """),
            ("human", "Analyze this URL for scraping strategy: {{url}}")
        ])

        # HTTP client for simple scraping
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
        )

    @property
    def specialized_scraper(self):
        """Lazy load specialized scraper"""
        if self._specialized_scraper is None:
            self._specialized_scraper = EaterTimeoutSpecializedScraper(self.config)
        return self._specialized_scraper

    @property
    def firecrawl_scraper(self):
        """Lazy load firecrawl scraper"""
        if self._firecrawl_scraper is None:
            self._firecrawl_scraper = FirecrawlWebScraper(self.config)
        return self._firecrawl_scraper

    @property
    def content_sectioner(self):
        """Lazy load content sectioner"""
        if self._content_sectioner is None:
            self._content_sectioner = ContentSectioningAgent(self.config)
        return self._content_sectioner

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        REWORKED: Main entry point with new domain intelligence flow
        """
        if not search_results:
            return []

        logger.info(f"🎯 Smart scraper processing {len(search_results)} URLs with NEW LOGIC")

        enriched_results = []

        for result in search_results:
            start_time = time.time()
            url = result.get("url", "")

            if not url:
                enriched_results.append(result)
                continue

            # STEP 1: Check specialized handlers first (FREE)
            if await self._try_specialized_scraping(result):
                enriched_results.append(result)
                self.stats["strategy_breakdown"]["specialized"] += 1
                continue

            # STEP 2: Check domain intelligence cache
            domain = self._extract_domain(url)
            cached_strategy = await self._get_cached_strategy(domain)

            if cached_strategy:
                logger.info(f"🧠 Using cached strategy for {domain}: {cached_strategy}")
                self.stats["domain_cache_hits"] += 1

                # Use cached strategy directly
                success = await self._execute_strategy(result, cached_strategy)
                if success:
                    enriched_results.append(result)
                    self.stats["strategy_breakdown"][cached_strategy] += 1
                    continue

            # STEP 3: New domain - use AI to classify strategy
            strategy = await self._classify_strategy_with_ai(url)
            self.stats["ai_analysis_calls"] += 1

            # STEP 4: Execute strategy with fallback chain
            success = await self._execute_strategy_with_fallback(result, strategy)

            if success:
                # STEP 5: Save domain intelligence AFTER successful scrape
                await self._save_domain_intelligence(domain, result.get("scraping_method", strategy))
                self.stats["new_domains_learned"] += 1

            enriched_results.append(result)
            self.stats["total_processed"] += 1

            # Update cost estimates
            self._update_cost_estimates(result.get("scraping_method", "failed"))

            processing_time = time.time() - start_time
            logger.info(f"⚡ Processed {url} in {processing_time:.1f}s using {result.get('scraping_method', 'failed')}")

        # Log final statistics
        self._log_final_stats()
        return enriched_results

    async def _try_specialized_scraping(self, result: Dict[str, Any]) -> bool:
        """Try specialized scraping first (RSS/Sitemap - FREE)"""
        url = result.get("url", "")

        try:
            # Check if this URL can be handled by specialized scraper
            if await self.specialized_scraper.can_handle_url(url):
                specialized_results = await self.specialized_scraper.process_single_url(result)

                if specialized_results.get("scraping_success"):
                    result.update(specialized_results)
                    result["scraping_method"] = "specialized"
                    return True

        except Exception as e:
            logger.debug(f"Specialized scraping failed for {url}: {e}")

        return False

    async def _get_cached_strategy(self, domain: str) -> Optional[str]:
        """Get cached strategy from domain intelligence table"""
        try:
            from utils.database_domain_intelligence import get_domain_intelligence_manager

            manager = get_domain_intelligence_manager()
            url = f"https://{domain}/"

            cached_strategy = manager.get_cached_strategy(url)

            if cached_strategy:
                logger.debug(f"🧠 Found cached strategy for {domain}: {cached_strategy}")

            return cached_strategy

        except Exception as e:
            logger.debug(f"Error fetching cached strategy for {domain}: {e}")
            return None

    async def _classify_strategy_with_ai(self, url: str) -> str:
        """Use AI to classify strategy for new domains"""
        try:
            logger.info(f"🤖 AI analyzing new domain: {self._extract_domain(url)}")

            formatted_prompt = self.analysis_prompt.format(url=url)
            response = await self.analyzer.ainvoke(formatted_prompt)

            strategy = response.content.strip().lower()

            # Validate strategy
            valid_strategies = [s.value for s in ScrapingStrategy if s != ScrapingStrategy.SPECIALIZED]
            if strategy not in valid_strategies:
                logger.warning(f"Invalid AI strategy '{strategy}', defaulting to enhanced_http")
                strategy = "enhanced_http"

            logger.info(f"🎯 AI classified {url} as: {strategy}")
            return strategy

        except Exception as e:
            logger.error(f"AI strategy classification failed for {url}: {e}")
            return "enhanced_http"  # Safe default

    async def _execute_strategy_with_fallback(self, result: Dict[str, Any], initial_strategy: str) -> bool:
        """Execute strategy with fallback chain: simple → enhanced → firecrawl"""

        # Define fallback chain
        if initial_strategy == "simple_http":
            strategy_chain = ["simple_http", "enhanced_http", "firecrawl"]
        elif initial_strategy == "enhanced_http":
            strategy_chain = ["enhanced_http", "firecrawl"]
        else:  # firecrawl
            strategy_chain = ["firecrawl"]

        for strategy in strategy_chain:
            logger.info(f"🔄 Trying {strategy} for {result.get('url', '')}")

            success = await self._execute_strategy(result, strategy)
            if success:
                self.stats["strategy_breakdown"][strategy] += 1
                return True

            # For Firecrawl, only try once
            if strategy == "firecrawl":
                logger.warning(f"❌ Firecrawl failed for {result.get('url', '')} - no more attempts")
                break

        return False

    async def _execute_strategy(self, result: Dict[str, Any], strategy: str) -> bool:
        """Execute specific scraping strategy"""
        url = result.get("url", "")

        try:
            if strategy == "simple_http":
                return await self._scrape_simple_http(result)
            elif strategy == "enhanced_http":
                return await self._scrape_enhanced_http(result)
            elif strategy == "firecrawl":
                return await self._scrape_firecrawl(result)
            else:
                logger.error(f"Unknown strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Strategy {strategy} failed for {url}: {e}")
            return False

    async def _scrape_simple_http(self, result: Dict[str, Any]) -> bool:
        """Simple HTTP scraping with sectioning"""
        url = result.get("url", "")

        try:
            # Basic HTTP request
            response = await self.http_client.get(url)
            response.raise_for_status()

            # Basic HTML parsing
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            content = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in content.splitlines())
            content = '\n'.join(line for line in lines if line)

            if len(content) < 500:  # Too little content
                return False

            # Apply content sectioning for restaurant extraction
            sectioned_result = await self._apply_sectioning(content, url)

            result.update({
                "scraping_method": "simple_http",
                "scraping_success": True,
                "scraped_content": sectioned_result.get("content", content),
                "restaurants_found": sectioned_result.get("restaurants_found", [])
            })

            self.stats["sectioning_calls"] += 1
            return True

        except Exception as e:
            logger.debug(f"Simple HTTP scraping failed for {url}: {e}")
            return False

    async def _scrape_enhanced_http(self, result: Dict[str, Any]) -> bool:
        """Enhanced HTTP scraping with readability and sectioning"""
        url = result.get("url", "")

        try:
            # HTTP request
            response = await self.http_client.get(url)
            response.raise_for_status()

            # Use readability for better content extraction
            doc = Document(response.text)
            title = doc.title()
            content = doc.summary()

            # Parse with BeautifulSoup for cleaning
            soup = BeautifulSoup(content, 'html.parser')

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
                element.decompose()

            clean_content = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in clean_content.splitlines())
            clean_content = '\n'.join(line for line in lines if line)

            if len(clean_content) < 500:  # Too little content
                return False

            # Apply content sectioning for restaurant extraction
            sectioned_result = await self._apply_sectioning(clean_content, url)

            result.update({
                "scraping_method": "enhanced_http",
                "scraping_success": True,
                "scraped_content": sectioned_result.get("content", clean_content),
                "scraped_title": title,
                "restaurants_found": sectioned_result.get("restaurants_found", [])
            })

            self.stats["sectioning_calls"] += 1
            return True

        except Exception as e:
            logger.debug(f"Enhanced HTTP scraping failed for {url}: {e}")
            return False

    async def _scrape_firecrawl(self, result: Dict[str, Any]) -> bool:
        """Firecrawl scraping - single attempt only"""
        url = result.get("url", "")

        try:
            logger.warning(f"🔥 Using expensive Firecrawl for {url} (single attempt)")
            self.stats["firecrawl_attempts"] += 1

            # Use Firecrawl scraper (already has latest API implementation)
            scraped_data = await self.firecrawl_scraper.scrape_url(url)

            if scraped_data.get("success", False):
                result.update({
                    "scraping_method": "firecrawl",
                    "scraping_success": True,
                    "scraped_content": scraped_data.get("content", ""),
                    "restaurants_found": scraped_data.get("restaurants_found", [])
                })
                return True
            else:
                logger.error(f"❌ Firecrawl failed for {url}: {scraped_data.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"❌ Firecrawl exception for {url}: {e}")
            return False

    async def _apply_sectioning(self, content: str, url: str) -> Dict[str, Any]:
        """Apply content sectioning to extract restaurant information"""
        try:
            sectioned_result = await self.content_sectioner.process_content(content, url)

            return {
                "content": sectioned_result.optimized_content or content,
                "restaurants_found": getattr(sectioned_result, 'restaurants_density', 0) > 0.5
            }

        except Exception as e:
            logger.debug(f"Content sectioning failed for {url}: {e}")
            return {"content": content, "restaurants_found": []}

    async def _save_domain_intelligence(self, domain: str, strategy: str):
        """Save domain intelligence AFTER successful scrape"""
        try:
            intelligence_data = {
                'strategy': strategy,
                'success_count': 1,
                'total_attempts': 1,
                'confidence': 0.8,  # High confidence after successful scrape
                'cost_per_scrape': self._get_strategy_cost(strategy),
                'updated_at': time.time()
            }

            self.database.save_domain_intelligence(domain, intelligence_data)
            logger.info(f"💾 Saved domain intelligence: {domain} → {strategy}")

        except Exception as e:
            logger.error(f"Failed to save domain intelligence for {domain}: {e}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return url

    def _get_strategy_cost(self, strategy: str) -> float:
        """Get cost estimate for strategy"""
        cost_map = {
            "specialized": 0.0,
            "simple_http": 0.1,
            "enhanced_http": 0.5,
            "firecrawl": 10.0
        }
        return cost_map.get(strategy, 0.5)

    def _update_cost_estimates(self, strategy: str):
        """Update cost tracking"""
        cost = self._get_strategy_cost(strategy)
        self.stats["total_cost_estimate"] += cost

        # Calculate savings vs all-Firecrawl
        firecrawl_cost = 10.0
        savings = firecrawl_cost - cost
        self.stats["cost_saved_vs_all_firecrawl"] += max(0, savings)

    def _log_final_stats(self):
        """Log comprehensive final statistics"""
        logger.info("=" * 60)
        logger.info("🎯 SMART SCRAPER FINAL STATISTICS")
        logger.info("=" * 60)

        total = self.stats["total_processed"]
        if total > 0:
            logger.info(f"📊 Total URLs Processed: {total}")
            logger.info(f"🧠 AI Analysis Calls: {self.stats['ai_analysis_calls']}")
            logger.info(f"💾 Domain Cache Hits: {self.stats['domain_cache_hits']}")
            logger.info(f"📚 New Domains Learned: {self.stats['new_domains_learned']}")

            logger.info("\n📈 STRATEGY BREAKDOWN:")
            for strategy, count in self.stats["strategy_breakdown"].items():
                if count > 0:
                    emoji = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}
                    percentage = (count / total) * 100
                    logger.info(f"   {emoji.get(strategy, '📌')} {strategy.upper()}: {count} ({percentage:.1f}%)")

            logger.info(f"\n💰 COST ANALYSIS:")
            logger.info(f"   Actual Cost: {self.stats['total_cost_estimate']:.1f} credits")
            logger.info(f"   All-Firecrawl Cost: {total * 10:.1f} credits")
            logger.info(f"   Cost Saved: {self.stats['cost_saved_vs_all_firecrawl']:.1f} credits")

            if total > 0:
                efficiency = (self.stats['cost_saved_vs_all_firecrawl'] / (total * 10)) * 100
                logger.info(f"   Cost Efficiency: {efficiency:.1f}%")

            logger.info(f"\n🔧 TECHNICAL STATS:")
            logger.info(f"   Sectioning Calls: {self.stats['sectioning_calls']}")
            logger.info(f"   Firecrawl Attempts: {self.stats['firecrawl_attempts']}")

            if self.stats['firecrawl_attempts'] > 0:
                firecrawl_success = self.stats["strategy_breakdown"]["firecrawl"]
                success_rate = (firecrawl_success / self.stats['firecrawl_attempts']) * 100
                logger.info(f"   Firecrawl Success Rate: {success_rate:.1f}%")

        logger.info("=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.http_client.aclose()
            if self._specialized_scraper:
                await self._specialized_scraper.cleanup()
            if self._firecrawl_scraper:
                await self._firecrawl_scraper.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")