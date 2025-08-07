# agents/smart_scraper.py
"""
Simplified Smart Scraper - Human Mimic Only
Uses Human Mimic for all scraping, no filtering or specialized handlers
"""

import asyncio
import logging
import time
from typing import Dict, List, Any
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from utils.database import get_database

logger = logging.getLogger(__name__)

class SmartRestaurantScraper:
    """
    Simplified Smart Scraper using Human Mimic for all URLs

    Strategy: Human Mimic for everything (2.0 credits per URL)
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()
        self.max_concurrent = 2
        self.browser = None
        self.contexts = []

        # Initialize Text Cleaner Agent
        from agents.text_cleaner_agent import TextCleanerAgent
        self._text_cleaner = TextCleanerAgent(config, model_override='deepseek')

        # Simplified stats tracking
        self.stats = {
            "total_processed": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_cost_estimate": 0.0,
            "total_processing_time": 0.0,
            "strategy_breakdown": {"human_mimic": 0}
        }

        logger.info("✅ SmartRestaurantScraper initialized with Human Mimic only")

    @property
    def content_sectioner(self):
        """REMOVED - No longer needed since Human Mimic gets full page content"""
        return None

    async def start(self):
        """Initialize browser and contexts"""
        if self.browser:
            return

        try:
            self.playwright = await async_playwright().start()

            # Launch browser with optimized settings for Railway
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--single-process',
                    '--no-zygote',
                    '--disable-extensions',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding'
                ]
            )

            # Create multiple contexts for concurrent scraping
            self.contexts = []
            for i in range(self.max_concurrent):
                context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                self.contexts.append(context)

            logger.info(f"🎭 Human mimic scraper started with {len(self.contexts)} concurrent contexts")

        except Exception as e:
            logger.error(f"Failed to start human mimic scraper: {e}")
            raise

    async def stop(self):
        """Clean up browser and contexts"""
        try:
            if self.contexts:
                for context in self.contexts:
                    await context.close()
                self.contexts = []

            if self.browser:
                await self.browser.close()
                self.browser = None

            if hasattr(self, 'playwright'):
                await self.playwright.stop()

            logger.info("🎭 Human mimic scraper stopped")

        except Exception as e:
            logger.error(f"Error stopping human mimic scraper: {e}")

    async def _get_domain_timeout(self, domain: str) -> int:
        """Get optimal timeout for domain based on history"""
        try:
            result = self.database.table('domain_intelligence').select('timeout_needed').eq('domain', domain).execute()
            if result.data and len(result.data) > 0:
                return max(10, min(30, result.data[0].get('timeout_needed', 15)))
        except Exception as e:
            logger.debug(f"Could not get cached timeout for {domain}: {e}")

        return 15  # Default timeout

    async def _save_domain_timeout(self, domain: str, timeout_used: int, success: bool):
        """Save optimal timeout for domain"""
        try:
            new_timeout = timeout_used
            if not success and timeout_used < 30:
                new_timeout = min(30, timeout_used + 5)  # Increase timeout for next time
            elif success and timeout_used > 10:
                new_timeout = max(10, timeout_used - 2)  # Optimize timeout down

            self.database.table('domain_intelligence').upsert({
                'domain': domain,
                'timeout_needed': new_timeout,
                'last_success': success,
                'learned_at': time.time(),
                'source': 'human_mimic_scraper'
            }).execute()
        except Exception as e:
            logger.debug(f"Could not save timeout for {domain}: {e}")

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape a single URL with semaphore control"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    async def _scrape_single_url(self, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape a single URL using human mimicking with intelligent timeouts"""
        start_time = time.time()
        domain = urlparse(url).netloc
        timeout = await self._get_domain_timeout(domain)

        try:
            context = self.contexts[context_index]
            page = await context.new_page()

            try:
                # Navigate to page with domain-specific timeout
                logger.debug(f"🎭 Loading {url} (timeout: {timeout}s)")
                await page.goto(url, wait_until='domcontentloaded', timeout=timeout * 1000)

                # Wait for any dynamic content to load
                await page.wait_for_timeout(2000)

                # Human mimic: Select all content like a human would (Cmd+A)
                await page.keyboard.press('Control+a')  # Use Ctrl+A on Linux/Windows
                await page.wait_for_timeout(500)  # Brief pause like human

                # Get the selected text content
                content = await page.evaluate("""
                    () => {
                        const selection = window.getSelection();
                        return selection.toString();
                    }
                """)

                # Fallback to body text if selection is empty
                if not content or len(content.strip()) < 100:
                    content = await page.evaluate("""
                        () => {
                            // Remove script and style elements
                            const scripts = document.querySelectorAll('script, style, nav, header, footer');
                            scripts.forEach(el => el.remove());

                            // Get clean body text
                            return document.body ? document.body.innerText : '';
                        }
                    """)

                load_time = round(time.time() - start_time, 2)
                char_count = len(content) if content else 0

                # Save successful timeout
                await self._save_domain_timeout(domain, timeout, True)

                logger.debug(f"✅ Human mimic scraped {url}: {char_count} chars in {load_time}s")

                return {
                    'content': content,
                    'success': True,
                    'load_time': load_time,
                    'char_count': char_count,
                    'timeout_used': timeout
                }

            finally:
                await page.close()

        except Exception as e:
            load_time = round(time.time() - start_time, 2)

            # Save failed timeout (increase for next time)
            await self._save_domain_timeout(domain, timeout, False)

            logger.warning(f"❌ Human mimic failed for {url}: {e}")

            return {
                'content': "",
                'success': False,
                'load_time': load_time,
                'char_count': 0,
                'error': str(e),
                'timeout_used': timeout
            }

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main scraping method - Human Mimic for all URLs
        """
        if not search_results:
            return []

        if not self.browser:
            await self.start()

        urls = [result.get('url') for result in search_results if result.get('url')]
        logger.info(f"🎭 Human mimic scraping {len(urls)} URLs with {self.max_concurrent} concurrent contexts")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Scrape all URLs concurrently
        scrape_tasks = [
            self._scrape_url_with_semaphore(semaphore, url, i % len(self.contexts))
            for i, url in enumerate(urls)
        ]

        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        # Merge scraping results back with search results
        enriched_results = []
        for search_result, scrape_result in zip(search_results, scrape_results):
            enriched = search_result.copy()

            if isinstance(scrape_result, Exception):
                logger.error(f"Exception scraping {enriched.get('url')}: {scrape_result}")
                enriched.update({
                    'scraped_content': "",
                    'scraping_success': False,
                    'scraping_method': 'human_mimic',
                    'scraping_error': str(scrape_result),
                    'load_time': 0.0
                })
                self.stats["failed_scrapes"] += 1
            else:
                enriched.update({
                    'scraped_content': scrape_result['content'],
                    'scraping_success': scrape_result['success'],
                    'scraping_method': 'human_mimic',
                    'scraping_error': scrape_result.get('error'),
                    'load_time': scrape_result['load_time'],
                    'char_count': scrape_result['char_count']
                })

                if scrape_result['success']:
                    self.stats["successful_scrapes"] += 1
                else:
                    self.stats["failed_scrapes"] += 1

            self.stats["total_processed"] += 1
            self.stats["strategy_breakdown"]["human_mimic"] += 1
            self.stats["total_cost_estimate"] += 2.0  # 2.0 credits per URL
            enriched_results.append(enriched)

        # Apply text cleaning to successful scrapes (skip content sectioning)
        await self._apply_text_cleaning(enriched_results)

        logger.info(f"✅ Smart scraper complete: {len(enriched_results)} results processed")
        self._log_stats()

        return enriched_results

    async def _apply_text_cleaning(self, enriched_results: List[Dict[str, Any]]):
        """Apply text cleaning only - no content sectioning needed"""
        successful_results = [r for r in enriched_results if r.get('scraping_success') and r.get('scraped_content')]

        if not successful_results:
            return

        logger.info(f"🧹 Applying text cleaning to {len(successful_results)} successful scrapes")

        for result in successful_results:
            try:
                # Clean content 
                content_to_clean = result.get('scraped_content', '')
                if content_to_clean:
                    cleaned_content = await self._text_cleaner.clean_text(content_to_clean)
                    result['cleaned_content'] = cleaned_content

                    # Mark as ready for editor processing
                    result['ready_for_editor'] = True

            except Exception as e:
                logger.warning(f"Text cleaning failed for {result.get('url')}: {e}")
                result['ready_for_editor'] = False

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main scraping method - Human Mimic for all URLs
        """
        if not search_results:
            return []

        if not self.browser:
            await self.start()

        urls = [result.get('url') for result in search_results if result.get('url')]
        logger.info(f"🎭 Human mimic scraping {len(urls)} URLs with {self.max_concurrent} concurrent contexts")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Scrape all URLs concurrently
        scrape_tasks = [
            self._scrape_url_with_semaphore(semaphore, url, i % len(self.contexts))
            for i, url in enumerate(urls)
        ]

        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        # Merge scraping results back with search results
        enriched_results = []
        for search_result, scrape_result in zip(search_results, scrape_results):
            enriched = search_result.copy()

            if isinstance(scrape_result, Exception):
                logger.error(f"Exception scraping {enriched.get('url')}: {scrape_result}")
                enriched.update({
                    'scraped_content': "",
                    'scraping_success': False,
                    'scraping_method': 'human_mimic',
                    'scraping_error': str(scrape_result),
                    'load_time': 0.0
                })
                self.stats["failed_scrapes"] += 1
            else:
                enriched.update({
                    'scraped_content': scrape_result['content'],
                    'scraping_success': scrape_result['success'],
                    'scraping_method': 'human_mimic',
                    'scraping_error': scrape_result.get('error'),
                    'load_time': scrape_result['load_time'],
                    'char_count': scrape_result['char_count']
                })

                if scrape_result['success']:
                    self.stats["successful_scrapes"] += 1
                else:
                    self.stats["failed_scrapes"] += 1

            self.stats["total_processed"] += 1
            self.stats["strategy_breakdown"]["human_mimic"] += 1
            self.stats["total_cost_estimate"] += 2.0  # 2.0 credits per URL
            enriched_results.append(enriched)

        # Apply content sectioning to successful scrapes
        await self._apply_content_sectioning(enriched_results)

        logger.info(f"✅ Smart scraper complete: {len(enriched_results)} results processed")
        self._log_stats()

        return enriched_results

    async def _apply_content_sectioning(self, enriched_results: List[Dict[str, Any]]):
        """Apply content sectioning to successful scrapes"""
        successful_results = [r for r in enriched_results if r.get('scraping_success') and r.get('scraped_content')]

        if not successful_results:
            return

        logger.info(f"📄 Applying content sectioning to {len(successful_results)} successful scrapes")

        for result in successful_results:
            try:
                # Clean content first
                content_to_clean = result.get('scraped_content', '')
                if content_to_clean:
                    cleaned_content = await self._text_cleaner.clean_text(content_to_clean)
                    result['cleaned_content'] = cleaned_content

                # Use cleaned content if available, otherwise original
                content_to_process = result.get('cleaned_content') or result.get('scraped_content')

                # Extract restaurants from content
                if hasattr(self, '_content_sectioner') and self._content_sectioner:
                    sectioned_content = await self._content_sectioner.section_content(content_to_process)
                    result['restaurants_found'] = sectioned_content.get('restaurants', [])
                else:
                    # Fallback: mark for editor processing
                    result['restaurants_found'] = []
                    result['needs_editor_processing'] = True

            except Exception as e:
                logger.warning(f"Content sectioning failed for {result.get('url')}: {e}")
                result['restaurants_found'] = []

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.stats,
            "success_rate": (self.stats["successful_scrapes"] / max(self.stats["total_processed"], 1)) * 100
        }

    def get_domain_intelligence(self) -> Dict[str, Any]:
        """Get domain intelligence from cache"""
        try:
            result = self.database.table('domain_intelligence').select('*').execute()
            return {row['domain']: row for row in result.data} if result.data else {}
        except Exception as e:
            logger.debug(f"Could not get domain intelligence: {e}")
            return {}

    def _log_stats(self):
        """Log processing statistics"""
        if self.stats["total_processed"] > 0:
            logger.info("📊 SMART SCRAPER STATISTICS:")
            logger.info(f"   🎭 Human Mimic: {self.stats['strategy_breakdown']['human_mimic']} URLs")
            logger.info(f"   ✅ Success Rate: {self.stats['successful_scrapes']}/{self.stats['total_processed']} ({(self.stats['successful_scrapes']/self.stats['total_processed']*100):.1f}%)")
            logger.info(f"   💰 Cost Estimate: {self.stats['total_cost_estimate']:.1f} credits")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup browser"""
        await self.stop()

    # =============================================================================
    # HUMAN MIMIC SCRAPER METHODS
    # =============================================================================

    async def _get_domain_timeout(self, domain: str) -> int:
        """Get optimal timeout for domain based on history"""
        try:
            result = self.database.table('domain_intelligence').select('timeout_needed').eq('domain', domain).execute()
            if result.data and len(result.data) > 0:
                return max(10, min(30, result.data[0].get('timeout_needed', 15)))
        except Exception as e:
            logger.debug(f"Could not get cached timeout for {domain}: {e}")

        return 15  # Default timeout

    async def _save_domain_timeout(self, domain: str, timeout_used: int, success: bool):
        """Save optimal timeout for domain"""
        try:
            new_timeout = timeout_used
            if not success and timeout_used < 30:
                new_timeout = min(30, timeout_used + 5)  # Increase timeout for next time
            elif success and timeout_used > 10:
                new_timeout = max(10, timeout_used - 2)  # Optimize timeout down

            self.database.table('domain_intelligence').upsert({
                'domain': domain,
                'timeout_needed': new_timeout,
                'last_success': success,
                'learned_at': time.time(),
                'source': 'human_mimic_scraper'
            }).execute()
        except Exception as e:
            logger.debug(f"Could not save timeout for {domain}: {e}")

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape a single URL with semaphore control"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    async def _scrape_single_url(self, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape a single URL using human mimicking with intelligent timeouts"""
        start_time = time.time()
        domain = urlparse(url).netloc
        timeout = await self._get_domain_timeout(domain)

        try:
            context = self.contexts[context_index]
            page = await context.new_page()

            try:
                # Navigate to page with domain-specific timeout
                logger.debug(f"🎭 Loading {url} (timeout: {timeout}s)")
                await page.goto(url, wait_until='domcontentloaded', timeout=timeout * 1000)

                # Wait for any dynamic content to load
                await page.wait_for_timeout(2000)

                # Human mimic: Select all content like a human would (Cmd+A)
                await page.keyboard.press('Control+a')  # Use Ctrl+A on Linux/Windows
                await page.wait_for_timeout(500)  # Brief pause like human

                # Get the selected text content
                content = await page.evaluate("""
                    () => {
                        const selection = window.getSelection();
                        return selection.toString();
                    }
                """)

                # Fallback to body text if selection is empty
                if not content or len(content.strip()) < 100:
                    content = await page.evaluate("""
                        () => {
                            // Remove script and style elements
                            const scripts = document.querySelectorAll('script, style, nav, header, footer');
                            scripts.forEach(el => el.remove());

                            // Get clean body text
                            return document.body ? document.body.innerText : '';
                        }
                    """)

                load_time = round(time.time() - start_time, 2)
                char_count = len(content) if content else 0

                # Save successful timeout
                await self._save_domain_timeout(domain, timeout, True)

                self.stats["total_processing_time"] += load_time

                logger.debug(f"✅ Human mimic scraped {url}: {char_count} chars in {load_time}s")

                return {
                    'content': content,
                    'success': True,
                    'load_time': load_time,
                    'char_count': char_count,
                    'timeout_used': timeout
                }

            finally:
                await page.close()

        except Exception as e:
            load_time = round(time.time() - start_time, 2)

            # Save failed timeout (increase for next time)
            await self._save_domain_timeout(domain, timeout, False)

            self.stats["total_processing_time"] += load_time

            logger.warning(f"❌ Human mimic failed for {url}: {e}")

            return {
                'content': "",
                'success': False,
                'load_time': load_time,
                'char_count': 0,
                'error': str(e),
                'timeout_used': timeout
            }