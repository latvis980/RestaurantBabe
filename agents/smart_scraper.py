# agents/smart_scraper.py - FIXED BUSINESS LOGIC VERSION
"""
Smart Scraper System with CORRECTED Business Logic

CORRECTED 4-STEP PROCESS:
Step 1: Using AI, divide URLs into two groups: simple_http | requiring more sophisticated methods
Step 2: Process simple_http, if success → save to domain intelligence marked simple_http. If failed → move all URLs to enhanced_http group
Step 3: Process ALL URLs that made it to this step using enhanced_http with sectioning. Successful ones saved to domain intelligence marked enhanced_http. Failed ones go to Firecrawl
Step 4: Firecrawl processes the remaining ones, one attempt, no retry. If success, save to domain_intelligence marked as firecrawl

FIXES:
- ✅ AI prompt fixed to return only strategy names
- ✅ Proper 4-step fallback logic implemented
- ✅ Domain intelligence saving after each successful step
- ✅ Batch processing by strategy groups
- ✅ Single Firecrawl attempt with no retries
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Set
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

class ScrapingStrategy(Enum):
    """Scraping strategies aligned with database"""
    SPECIALIZED = "specialized"      # FREE - RSS/Sitemap
    SIMPLE_HTTP = "simple_http"      # 0.1 credits
    ENHANCED_HTTP = "enhanced_http"  # 0.5 credits  
    FIRECRAWL = "firecrawl"         # 10.0 credits

class SmartRestaurantScraper:
    """
    FIXED AI-Powered Smart Scraper with Corrected 4-Step Business Logic
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()

        # Initialize domain intelligence manager
        try:
            self.domain_manager = get_domain_intelligence_manager()
        except Exception as e:
            logger.warning(f"Domain intelligence manager not available: {e}")
            self.domain_manager = None

        # AI analyzer for URL classification
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
            "firecrawl_success_rate": 0.0,
            "step_breakdown": {
                "step1_ai_classification": 0,
                "step2_simple_http_attempts": 0,
                "step2_simple_http_successes": 0,
                "step3_enhanced_http_attempts": 0,
                "step3_enhanced_http_successes": 0,
                "step4_firecrawl_attempts": 0,
                "step4_firecrawl_successes": 0
            }
        }

        # FIXED: AI prompt for strategy classification that returns ONLY strategy names
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a web scraping strategy classifier. Analyze the URL and determine if it needs simple HTTP scraping or more sophisticated methods.

OUTPUT RULES:
- Return EXACTLY ONE of these words: simple_http, enhanced_http
- simple_http: Basic static content sites, blogs, news articles
- enhanced_http: Complex sites with dynamic content, heavy JavaScript, or anti-bot measures

TECHNICAL INDICATORS:
- simple_http: .com/blog/, wordpress sites, static news sites, basic restaurant sites
- enhanced_http: .js frameworks evident, SPA sites, timeout.com, eater.com, complex restaurant platforms

Return ONLY the strategy name, nothing else.
            """),
            ("human", "Classify scraping strategy for: {{url}}")
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
        CORRECTED: Main entry point implementing proper 4-step business logic
        """
        if not search_results:
            return []

        logger.info(f"🎯 Smart scraper processing {len(search_results)} URLs with CORRECTED 4-STEP LOGIC")

        # First handle specialized URLs (RSS/sitemap) - these bypass the 4-step process
        enriched_results = []
        remaining_urls = []

        for result in search_results:
            if await self._try_specialized_scraping(result):
                enriched_results.append(result)
                self.stats["strategy_breakdown"]["specialized"] += 1
            else:
                remaining_urls.append(result)

        if not remaining_urls:
            self._log_final_stats()
            return enriched_results

        # Now apply the 4-step process to remaining URLs
        processed_results = await self._apply_four_step_process(remaining_urls)
        enriched_results.extend(processed_results)

        self._log_final_stats()
        return enriched_results

    async def _apply_four_step_process(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CORRECTED: Apply the exact 4-step business logic you specified
        """

        # STEP 1: Using AI, divide URLs into two groups: simple_http | requiring more sophisticated methods
        logger.info("📋 STEP 1: AI classifying URLs into simple_http vs enhanced_http groups")
        simple_http_urls = []
        enhanced_http_urls = []

        for result in search_results:
            url = result.get("url", "")
            if not url:
                continue

            # Check domain intelligence cache first
            domain = self._extract_domain(url)
            cached_strategy = await self._get_cached_strategy(domain)

            if cached_strategy:
                self.stats["domain_cache_hits"] += 1
                if cached_strategy == "simple_http":
                    simple_http_urls.append(result)
                else:
                    enhanced_http_urls.append(result)
            else:
                # Use AI to classify
                ai_strategy = await self._classify_strategy_with_ai(url)
                self.stats["ai_analysis_calls"] += 1
                self.stats["step_breakdown"]["step1_ai_classification"] += 1

                if ai_strategy == "simple_http":
                    simple_http_urls.append(result)
                else:
                    enhanced_http_urls.append(result)

        logger.info(f"📊 Step 1 Results: {len(simple_http_urls)} simple_http, {len(enhanced_http_urls)} enhanced_http")

        # STEP 2: Process simple_http, if success → save to domain intelligence. If failed → move to enhanced_http group
        logger.info("🔧 STEP 2: Processing simple_http URLs")
        step2_results = []
        failed_simple_urls = []

        for result in simple_http_urls:
            self.stats["step_breakdown"]["step2_simple_http_attempts"] += 1
            success = await self._scrape_simple_http(result)

            if success:
                self.stats["step_breakdown"]["step2_simple_http_successes"] += 1
                self.stats["strategy_breakdown"]["simple_http"] += 1
                step2_results.append(result)

                # Save to domain intelligence marked as simple_http
                domain = self._extract_domain(result.get("url", ""))
                await self._save_domain_intelligence(domain, "simple_http")
            else:
                # Failed simple_http URLs move to enhanced_http group
                failed_simple_urls.append(result)
                logger.info(f"⬆️ Moving failed simple_http URL to enhanced_http: {result.get('url', '')}")

        # Add failed simple URLs to enhanced_http group
        enhanced_http_urls.extend(failed_simple_urls)

        logger.info(f"📊 Step 2 Results: {len(step2_results)} succeeded, {len(failed_simple_urls)} moved to enhanced_http")

        # STEP 3: Process ALL URLs in enhanced_http group using sectioning. Success → save as enhanced_http. Failed → go to Firecrawl
        logger.info("⚡ STEP 3: Processing enhanced_http URLs with sectioning")
        step3_results = []
        failed_enhanced_urls = []

        for result in enhanced_http_urls:
            self.stats["step_breakdown"]["step3_enhanced_http_attempts"] += 1
            success = await self._scrape_enhanced_http(result)

            if success:
                self.stats["step_breakdown"]["step3_enhanced_http_successes"] += 1
                self.stats["strategy_breakdown"]["enhanced_http"] += 1
                step3_results.append(result)

                # Save to domain intelligence marked as enhanced_http
                domain = self._extract_domain(result.get("url", ""))
                await self._save_domain_intelligence(domain, "enhanced_http")
            else:
                # Failed enhanced_http URLs go to Firecrawl
                failed_enhanced_urls.append(result)
                logger.info(f"🔥 Moving failed enhanced_http URL to Firecrawl: {result.get('url', '')}")

        logger.info(f"📊 Step 3 Results: {len(step3_results)} succeeded, {len(failed_enhanced_urls)} going to Firecrawl")

        # STEP 4: Firecrawl processes the remaining ones, one attempt, no retry. Success → save as firecrawl
        logger.info("🔥 STEP 4: Processing remaining URLs with Firecrawl (single attempt)")
        step4_results = []

        for result in failed_enhanced_urls:
            self.stats["step_breakdown"]["step4_firecrawl_attempts"] += 1
            self.stats["firecrawl_attempts"] += 1
            success = await self._scrape_firecrawl(result)

            if success:
                self.stats["step_breakdown"]["step4_firecrawl_successes"] += 1
                self.stats["strategy_breakdown"]["firecrawl"] += 1
                step4_results.append(result)

                # Save to domain intelligence marked as firecrawl
                domain = self._extract_domain(result.get("url", ""))
                await self._save_domain_intelligence(domain, "firecrawl")
            else:
                # Final failure - add result anyway with failure markers
                result.update({
                    "scraping_method": "failed",
                    "scraping_success": False,
                    "scraped_content": "",
                    "restaurants_found": []
                })
                step4_results.append(result)

        logger.info(f"📊 Step 4 Results: {self.stats['step_breakdown']['step4_firecrawl_successes']} succeeded")

        # Update Firecrawl success rate
        if self.stats["firecrawl_attempts"] > 0:
            self.stats["firecrawl_success_rate"] = (self.stats["step_breakdown"]["step4_firecrawl_successes"] / self.stats["firecrawl_attempts"]) * 100

        # Combine all results
        all_results = step2_results + step3_results + step4_results

        # Update total stats
        self.stats["total_processed"] += len(search_results)
        self._update_cost_estimates()

        return all_results

    async def _try_specialized_scraping(self, result: Dict[str, Any]) -> bool:
        """Try specialized scraping (RSS/sitemap) first - bypasses 4-step process"""
        url = result.get("url", "")
        domain = self._extract_domain(url)

        # Check if domain supports specialized scraping
        specialized_domains = ["timeout.com", "eater.com", "bonappetit.com", "foodandwine.com"]

        if not any(spec_domain in domain for spec_domain in specialized_domains):
            return False

        try:
            scraped_data = await self.specialized_scraper.scrape_url(url)

            if scraped_data.get("success", False):
                result.update({
                    "scraping_method": "specialized",
                    "scraping_success": True,
                    "scraped_content": scraped_data.get("content", ""),
                    "restaurants_found": scraped_data.get("restaurants_found", [])
                })
                return True

        except Exception as e:
            logger.debug(f"Specialized scraping failed for {url}: {e}")

        return False

    async def _get_cached_strategy(self, domain: str) -> Optional[str]:
        """Get cached strategy from domain intelligence table"""
        if not self.domain_manager:
            return None

        try:
            # The domain manager expects a full URL, so we construct one
            url = f"https://{domain}/"
            cached_strategy = self.domain_manager.get_cached_strategy(url)

            if cached_strategy:
                logger.debug(f"🧠 Found cached strategy for {domain}: {cached_strategy}")
                self.stats["domain_cache_hits"] += 1

            return cached_strategy

        except Exception as e:
            logger.debug(f"Error fetching cached strategy for {domain}: {e}")
            return None

    async def _classify_strategy_with_ai(self, url: str) -> str:
        """FIXED: Use AI to classify strategy for new domains - returns only simple_http or enhanced_http"""
        try:
            domain = self._extract_domain(url)  # Only use for logging
            logger.info(f"🤖 AI analyzing new domain: {domain}")

            # Use the corrected prompt
            messages = self.analysis_prompt.format_messages(url=url)
            response = await self.analyzer.ainvoke(messages)

            strategy = response.content.strip().lower()

            # Validate strategy - only allow simple_http or enhanced_http from AI
            if strategy not in ["simple_http", "enhanced_http"]:
                logger.warning(f"AI returned invalid strategy '{strategy}' for {url}, defaulting to enhanced_http")
                strategy = "enhanced_http"

            logger.info(f"🎯 AI classified {url} as: {strategy}")
            return strategy

        except Exception as e:
            logger.error(f"AI strategy classification failed for {url}: {e}")
            return "enhanced_http"  # Safe default

    async def _scrape_simple_http(self, result: Dict[str, Any]) -> bool:
        """Simple HTTP scraping with basic content extraction"""
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
                logger.debug(f"Simple HTTP failed for {url}: insufficient content ({len(content)} chars)")
                return False

            # Apply content sectioning for restaurant extraction
            sectioned_result = await self._apply_sectioning(content, url)

            # Keep original result structure - editor expects specific format
            result.update({
                "scraping_method": "simple_http",
                "scraping_success": True,
                "scraped_content": sectioned_result.get("content", content),
                "restaurants_found": sectioned_result.get("restaurants_found", [])
            })

            logger.info(f"✅ Simple HTTP success for {url}")
            return True

        except Exception as e:
            logger.debug(f"Simple HTTP failed for {url}: {e}")
            return False

    async def _scrape_enhanced_http(self, result: Dict[str, Any]) -> bool:
        """Enhanced HTTP scraping with readability processing and sectioning"""
        url = result.get("url", "")

        try:
            # Enhanced HTTP request with better error handling
            response = await self.http_client.get(url)
            response.raise_for_status()

            # Use readability to extract main content
            doc = Document(response.text)
            content = doc.summary()

            # Convert to text
            soup = BeautifulSoup(content, 'html.parser')
            content_text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in content_text.splitlines())
            content_text = '\n'.join(line for line in lines if line)

            if len(content_text) < 800:  # Higher threshold for enhanced
                logger.debug(f"Enhanced HTTP failed for {url}: insufficient content ({len(content_text)} chars)")
                return False

            # Apply content sectioning for restaurant extraction
            sectioned_result = await self._apply_sectioning(content_text, url)

            # Keep original result structure - editor expects specific format
            result.update({
                "scraping_method": "enhanced_http",
                "scraping_success": True,
                "scraped_content": sectioned_result.get("content", content_text),
                "restaurants_found": sectioned_result.get("restaurants_found", [])
            })

            logger.info(f"✅ Enhanced HTTP success for {url}")
            return True

        except Exception as e:
            logger.debug(f"Enhanced HTTP failed for {url}: {e}")
            return False

    async def _scrape_firecrawl(self, result: Dict[str, Any]) -> bool:
        """Firecrawl scraping - single attempt, no retry"""
        url = result.get("url", "")

        try:
            logger.info(f"🔥 Firecrawl attempt for {url} (single attempt, no retry)")

            # Use Firecrawl scraper (single attempt)
            scraped_data = await self.firecrawl_scraper.scrape_url(url)

            if scraped_data.get("success", False):
                # Keep original result structure - editor expects specific format
                result.update({
                    "scraping_method": "firecrawl",
                    "scraping_success": True,
                    "scraped_content": scraped_data.get("content", ""),
                    "restaurants_found": scraped_data.get("restaurants_found", [])
                })
                logger.info(f"✅ Firecrawl success for {url}")
                return True
            else:
                logger.warning(f"❌ Firecrawl failed for {url}: {scraped_data.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"❌ Firecrawl exception for {url}: {e}")
            return False

    async def _apply_sectioning(self, content: str, url: str) -> Dict[str, Any]:
        """Apply content sectioning to extract restaurant information"""
        try:
            self.stats["sectioning_calls"] += 1
            sectioned_result = await self.content_sectioner.process_content(content, url)

            return {
                "content": sectioned_result.optimized_content or content,
                "restaurants_found": getattr(sectioned_result, 'restaurants_found', [])
            }

        except Exception as e:
            logger.debug(f"Content sectioning failed for {url}: {e}")
            return {"content": content, "restaurants_found": []}

    async def _save_domain_intelligence(self, domain: str, strategy: str):
        """Save domain intelligence AFTER successful scrape"""
        if not self.domain_manager:
            return

        try:
            # Build intelligence data
            intelligence_data = {
                'strategy': strategy,
                'success_count': 1,
                'total_attempts': 1,
                'confidence': 0.8,  # High confidence after successful scrape
                'cost_per_scrape': self._get_strategy_cost(strategy),
                'updated_at': time.time()
            }

            # Use the domain manager's save method (check if it exists)
            if hasattr(self.domain_manager, 'save_scrape_result'):
                url = f"https://{domain}/"
                self.domain_manager.save_scrape_result(
                    url=url,
                    scraping_method=strategy,
                    scraping_success=True,
                    **intelligence_data
                )
                logger.info(f"💾 Saved domain intelligence: {domain} → {strategy}")
                self.stats["new_domains_learned"] += 1
            else:
                logger.debug(f"Domain manager doesn't support saving intelligence for {domain}")

        except Exception as e:
            logger.error(f"Failed to save domain intelligence for {domain}: {e}")

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL - ONLY used internally for domain intelligence

        NOTE: This is NOT used for sources pipeline - editor handles that.
        This is only for domain intelligence operations within the smart scraper.
        """
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

    def _update_cost_estimates(self):
        """Update cost tracking based on strategy breakdown"""
        total_cost = 0.0
        for strategy, count in self.stats["strategy_breakdown"].items():
            cost_per_url = self._get_strategy_cost(strategy)
            total_cost += count * cost_per_url

        self.stats["total_cost_estimate"] = total_cost

        # Calculate savings vs all-Firecrawl
        total_urls = sum(self.stats["strategy_breakdown"].values())
        all_firecrawl_cost = total_urls * 10.0
        self.stats["cost_saved_vs_all_firecrawl"] = all_firecrawl_cost - total_cost

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

            # Step breakdown
            logger.info("\n📋 4-STEP PROCESS BREAKDOWN:")
            logger.info(f"   Step 1 - AI Classifications: {self.stats['step_breakdown']['step1_ai_classification']}")
            logger.info(f"   Step 2 - Simple HTTP: {self.stats['step_breakdown']['step2_simple_http_successes']}/{self.stats['step_breakdown']['step2_simple_http_attempts']} success")
            logger.info(f"   Step 3 - Enhanced HTTP: {self.stats['step_breakdown']['step3_enhanced_http_successes']}/{self.stats['step_breakdown']['step3_enhanced_http_attempts']} success")
            logger.info(f"   Step 4 - Firecrawl: {self.stats['step_breakdown']['step4_firecrawl_successes']}/{self.stats['step_breakdown']['step4_firecrawl_attempts']} success")

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

        logger.info("=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            **self.stats,
            "cost_efficiency_percentage": (
                (self.stats['cost_saved_vs_all_firecrawl'] / 
                 max(self.stats['total_processed'] * 10, 1)) * 100
            ) if self.stats['total_processed'] > 0 else 0
        }