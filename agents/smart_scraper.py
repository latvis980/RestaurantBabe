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

    async def start(self):
        """Initialize browser and contexts"""
        if self.browser:
            return

        try:
            self.playwright = await async_playwright().start()

            # Launch browser with optimized settings for Railway + ad blocking
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
                    '--disable-renderer-backgrounding',
                    # Ad blocking related flags
                    '--disable-default-apps',
                    '--disable-plugins',
                    '--disable-sync',
                    '--no-default-browser-check',
                    '--no-first-run',
                    # Performance optimizations
                    '--memory-pressure-off',
                    '--max_old_space_size=4096'
                ]
            )

            # Create multiple contexts for concurrent scraping
            self.contexts = []
            for i in range(self.max_concurrent):
                context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                    },
                    # Block unnecessary permissions that can slow things down
                    permissions=[],
                    geolocation=None,
                    ignore_https_errors=True
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

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape a single URL with semaphore control"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    async def _scrape_single_url(self, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape a single URL with progressive timeout (no reload) and ad blocking"""
        start_time = time.time()
        domain = urlparse(url).netloc

        # Progressive timeout attempts (no reload between attempts)
        timeout_attempts = [15000, 30000, 45000]  # 15s, 30s, 45s

        try:
            context = self.contexts[context_index]
            page = await context.new_page()

            try:
                # Configure ad blocking for faster loading
                await self._configure_page_with_adblock(page)

                load_success = False
                final_timeout = timeout_attempts[0]

                # Progressive timeout strategy - NO PAGE RELOAD
                for attempt, timeout in enumerate(timeout_attempts):
                    try:
                        logger.debug(f"🎭 Attempt {attempt + 1}: Loading {url} (timeout: {timeout/1000}s)")

                        if attempt == 0:
                            # First attempt: load page
                            await page.goto(url, wait_until='domcontentloaded', timeout=timeout)
                        else:
                            # Subsequent attempts: just wait longer (no reload)
                            await page.wait_for_load_state('domcontentloaded', timeout=timeout)

                        load_success = True
                        final_timeout = timeout
                        break

                    except Exception as timeout_error:
                        if attempt < len(timeout_attempts) - 1:
                            logger.debug(f"⏱️ Timeout attempt {attempt + 1} failed, trying longer timeout...")
                            continue
                        else:
                            raise timeout_error

                if not load_success:
                    raise Exception("All timeout attempts failed")

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

                self.stats["total_processing_time"] += load_time

                logger.debug(f"✅ Human mimic scraped {url}: {char_count} chars in {load_time}s")

                return {
                    'content': content,
                    'success': True,
                    'load_time': load_time,
                    'char_count': char_count,
                    'final_timeout': final_timeout
                }

            finally:
                await page.close()

        except Exception as e:
            load_time = round(time.time() - start_time, 2)
            self.stats["total_processing_time"] += load_time

            logger.warning(f"❌ Human mimic failed for {url}: {e}")

            return {
                'content': "",
                'success': False,
                'load_time': load_time,
                'char_count': 0,
                'error': str(e)
            }

    async def _configure_page_with_adblock(self, page: Page):
        """
        Configure page with ad blocking for dramatically faster loading
        Can improve load times by 20-40%
        """
        # Block resource types that slow down loading
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in [
                'image',      # Images (biggest bandwidth hog)
                'media',      # Videos/audio  
                'font',       # Web fonts
                'stylesheet', # CSS (may break layout but speeds loading)
                'beacon',     # Analytics beacons
                'csp_report', # Security reports
                'object',     # Flash/embed objects
                'texttrack'   # Video subtitles
            ] else route.continue_()
        ))

        # Block ad/tracking domains
        ad_domains = [
            'doubleclick.net',
            'googleadservices.com', 
            'googlesyndication.com',
            'facebook.com/tr',
            'analytics.google.com',
            'googletagmanager.com',
            'google-analytics.com',
            'hotjar.com',
            'crazyegg.com',
            'mouseflow.com',
            'fullstory.com',
            'mixpanel.com',
            'segment.com',
            'amplitude.com'
        ]

        await page.route("**/*", lambda route: (
            route.abort() if any(ad_domain in route.request.url for ad_domain in ad_domains)
            else route.continue_()
        ))

        # Set faster headers
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
        })

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main scraping method - Human Mimic for all URLs
        """
        if not search_results:
            return []

        if not self.browser:
            await self.start()

        # Filter out None URLs
        urls = [result.get('url') for result in search_results if result.get('url')]
        logger.info(f"🎭 Human mimic scraping {len(urls)} URLs with {self.max_concurrent} concurrent contexts")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Scrape all URLs concurrently
        scrape_tasks = [
            self._scrape_url_with_semaphore(semaphore, url, i % len(self.contexts))
            for i, url in enumerate(urls) if url  # Ensure url is not None
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
            elif isinstance(scrape_result, dict):
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
            else:
                # Handle unexpected result type
                logger.error(f"Unexpected scrape result type: {type(scrape_result)}")
                enriched.update({
                    'scraped_content': "",
                    'scraping_success': False,
                    'scraping_method': 'human_mimic',
                    'scraping_error': "Unexpected result type",
                    'load_time': 0.0
                })
                self.stats["failed_scrapes"] += 1

            self.stats["total_processed"] += 1
            self.stats["strategy_breakdown"]["human_mimic"] += 1
            self.stats["total_cost_estimate"] += 2.0  # 2.0 credits per URL
            enriched_results.append(enriched)

        # Apply text cleaning to successful scrapes
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
                # Clean content using the correct method name
                content_to_clean = result.get('scraped_content', '')
                if content_to_clean:
                    cleaned_content = self._text_cleaner.clean_scraped_results([result])
                    result['cleaned_content'] = cleaned_content

                    # Mark as ready for editor processing
                    result['ready_for_editor'] = True

            except Exception as e:
                logger.warning(f"Text cleaning failed for {result.get('url')}: {e}")
                result['ready_for_editor'] = False

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.stats,
            "success_rate": (self.stats["successful_scrapes"] / max(self.stats["total_processed"], 1)) * 100
        }

    def get_domain_intelligence(self) -> Dict[str, Any]:
        """REMOVED - No longer needed since we eliminated filtering"""
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