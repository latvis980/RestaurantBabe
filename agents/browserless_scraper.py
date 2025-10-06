# agents/optimized_browserless_scraper.py
"""
OPTIMIZED: Enhanced Restaurant Scraper with Persistent Browser Sessions
KEY OPTIMIZATIONS:
- Browser session persistence across multiple URLs
- Connection pooling with 2-3 concurrent browsers
- Reduced Railway Browserless connection overhead
- Smart session management with reconnection
"""

import asyncio
import logging
import time
import re
import os
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
from utils.database import get_database

logger = logging.getLogger(__name__)


class OptimizedBrowserlessRestaurantScraper:
    """
    OPTIMIZED: Enhanced Restaurant Scraper with persistent Railway Browserless sessions
    - Maintains browser sessions across multiple URLs
    - Uses connection pooling for better performance
    - Reduces connection overhead significantly
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()

        # OPTIMIZATION: Persistent browser pool
        self.browser_pool_size = 3  # 2-3 browsers for rotation
        self.browser_pool: List[Browser] = []
        self.browser_contexts: List[BrowserContext] = []
        self.current_browser_index = 0

        # Session management
        self.playwright = None
        self.session_active = False
        self._session_lock = asyncio.Lock()

        # Keep operation tracking
        self._active_operations = 0
        self._operation_lock = asyncio.Lock()

        # Railway Browserless configuration
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        self.browserless_endpoint_private = os.getenv('BROWSER_PLAYWRIGHT_ENDPOINT_PRIVATE')
        self.browserless_endpoint_public = os.getenv('BROWSER_PLAYWRIGHT_ENDPOINT')
        self.browserless_token = os.getenv('BROWSER_TOKEN')

        # Choose the best endpoint
        if self.browserless_endpoint_private:
            self.browserless_endpoint = self.browserless_endpoint_private
            logger.info(f"🚀 Railway Browserless private endpoint: {self.browserless_endpoint}")
        elif self.browserless_endpoint_public:
            self.browserless_endpoint = self.browserless_endpoint_public
            logger.info(f"🚀 Railway Browserless public endpoint: {self.browserless_endpoint}")
        else:
            self.browserless_endpoint = None
            logger.info("🔧 Local Playwright mode")

        # OPTIMIZATION: Timeouts reduced for faster cycling
        self.default_timeout = 15000  # 15 seconds default
        self.slow_timeout = 30000     # 30 seconds for slow sites
        self.browser_launch_timeout = 20000  # 20 seconds for browser launch

        # Domain-specific timeouts
        self.domain_timeouts = {
            'guide.michelin.com': 25000,
            'timeout.com': 20000,
            'eater.com': 20000,
            'yelp.com': 18000,
            'opentable.com': 18000
        }

        # OPTIMIZATION: Reduced wait times for faster processing
        self.load_wait_time = 1.5      # Reduced from 3.0s
        self.interaction_delay = 0.3   # Reduced from 0.5s

        # Enhanced stats tracking
        self.stats = {
            "total_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "avg_scrape_time": 0.0,
            "total_processing_time": 0.0,
            "browser_reuses": 0,
            "session_reconnects": 0,
            "pool_efficiency": 0.0,
            "railway_browserless": bool(self.browserless_endpoint),
            "endpoint_used": self.browserless_endpoint or "local_playwright",
            "browser_pool_size": self.browser_pool_size,
            "connection_savings_estimate": 0.0
        }

        logger.info("🎯 OPTIMIZED Restaurant Scraper initialized")
        logger.info(f"🔄 Browser Pool Size: {self.browser_pool_size}")
        logger.info(f"🚂 Railway Browserless: {'✓' if self.browserless_endpoint else '✗'}")

    async def scrape_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """
        OPTIMIZED MAIN PIPELINE: Uses persistent browser sessions
        """
        if not search_results:
            logger.warning("⚠️ No search results to scrape")
            return []

        logger.info(f"🤖 OPTIMIZED SCRAPING: Processing {len(search_results)} URLs with persistent browsers")

        # Initialize browser pool once
        await self._initialize_browser_pool()

        try:
            # OPTIMIZATION: Use browser pool with smart cycling
            semaphore = asyncio.Semaphore(self.browser_pool_size)
            tasks = []

            for i, result in enumerate(search_results):
                browser_index = i % self.browser_pool_size
                task = self._scrape_with_persistent_browser(semaphore, result, browser_index)
                tasks.append(task)

            # Execute all scraping tasks
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful_results = []
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Task {i} failed: {result}")
                    original_result = search_results[i]
                    fallback_result = original_result.copy()
                    fallback_result.update({
                        "content": "",
                        "scraping_success": False,
                        "scraping_failed": True,
                        "error": str(result)
                    })
                    successful_results.append(fallback_result)
                else:
                    successful_results.append(result)

            # Update stats
            self.stats["total_scraped"] += len(search_results)
            successful_count = sum(1 for r in successful_results if r.get("scraping_success", False))
            self.stats["successful_scrapes"] += successful_count
            self.stats["failed_scrapes"] += len(search_results) - successful_count

            # Calculate efficiency gains
            browser_reuses = len(search_results) - self.browser_pool_size
            self.stats["browser_reuses"] += max(0, browser_reuses)
            self.stats["connection_savings_estimate"] += max(0, browser_reuses) * 2.5  # ~2.5s per saved connection

            success_rate = (successful_count / len(search_results)) * 100 if search_results else 0
            logger.info(f"🎯 OPTIMIZED SCRAPING COMPLETE: {successful_count}/{len(search_results)} successful ({success_rate:.1f}%)")
            logger.info(f"🔄 Browser Reuses: {self.stats['browser_reuses']}, Time Saved: ~{self.stats['connection_savings_estimate']:.1f}s")

            return successful_results

        finally:
            # Keep browsers alive for next batch - don't close here
            pass

    async def _initialize_browser_pool(self):
        """
        OPTIMIZATION: Initialize persistent browser pool
        """
        if self.session_active:
            return  # Already initialized

        async with self._session_lock:
            if self.session_active:
                return

            logger.info("🚀 Initializing browser pool...")

            # Initialize Playwright once
            self.playwright = await async_playwright().start()

            # Create browser pool
            for i in range(self.browser_pool_size):
                try:
                    browser = await self._create_single_browser(f"pool-{i}")
                    if browser:
                        self.browser_pool.append(browser)
                        # Create a context for each browser
                        context = await browser.new_context()
                        self.browser_contexts.append(context)
                        logger.info(f"✅ Browser {i+1}/{self.browser_pool_size} ready")
                    else:
                        logger.warning(f"⚠️ Failed to create browser {i+1}")
                except Exception as e:
                    logger.error(f"❌ Failed to create browser {i+1}: {e}")

            if len(self.browser_pool) > 0:
                self.session_active = True
                logger.info(f"🎯 Browser pool initialized: {len(self.browser_pool)}/{self.browser_pool_size} browsers ready")
            else:
                logger.error("❌ Failed to initialize any browsers in pool")
                raise Exception("Browser pool initialization failed")

    async def _create_single_browser(self, browser_id: str) -> Optional[Browser]:
        """
        OPTIMIZATION: Create a single persistent browser connection
        """
        try:
            browser_options = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-background-networking",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-ipc-flooding-protection",
                    # OPTIMIZATION: Additional performance flags
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",  # Block images for speed
                    "--disable-javascript-harmony-shipping",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-field-trial-config"
                ]
            }

            # Use Railway Browserless endpoint or local
            if self.browserless_endpoint:
                try:
                    connect_url = self.browserless_endpoint
                    if self.browserless_token and '?token=' not in connect_url:
                        separator = '&' if '?' in connect_url else '?'
                        connect_url = f"{connect_url}{separator}token={self.browserless_token}"

                    browser = await self.playwright.chromium.connect(
                        connect_url,
                        timeout=self.browser_launch_timeout
                    )
                    logger.info(f"🚂 Browser {browser_id} connected to Railway Browserless")
                    return browser
                except Exception as browserless_error:
                    logger.warning(f"⚠️ Railway Browserless connection failed for {browser_id}: {browserless_error}")
                    logger.info(f"🔄 Falling back to local for {browser_id}...")
                    return await self.playwright.webkit.launch(**browser_options)
            else:
                return await self.playwright.webkit.launch(**browser_options)

        except Exception as e:
            logger.error(f"❌ Failed to create browser {browser_id}: {e}")
            return None

    async def _scrape_with_persistent_browser(self, semaphore: asyncio.Semaphore, result: Dict, browser_index: int) -> Dict:
        """
        OPTIMIZATION: Scrape using persistent browser from pool
        """
        async with semaphore:
            return await self._scrape_single_url_optimized(result, browser_index)

    async def _scrape_single_url_optimized(self, result: Dict, browser_index: int) -> Dict:
        """
        OPTIMIZED: Scrape individual URL using persistent browser
        """
        url = result.get("url", "")
        if not url:
            logger.warning("⚠️ No URL provided")
            return result

        start_time = time.time()
        original_result = result.copy()

        # Select browser from pool
        if browser_index >= len(self.browser_pool):
            logger.warning(f"⚠️ Browser index {browser_index} out of range, using browser 0")
            browser_index = 0

        browser = self.browser_pool[browser_index]
        context = self.browser_contexts[browser_index]

        try:
            logger.info(f"🎯 Scraping with browser-{browser_index}: {url}")

            # Create new page in existing context
            page = await context.new_page()

            try:
                # Configure page for optimization
                await self._configure_page_optimized(page)

                # Get domain-specific timeout
                domain = urlparse(url).netloc.lower()
                timeout = self.domain_timeouts.get(domain, self.default_timeout)

                # Navigate to page
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                await asyncio.sleep(self.load_wait_time)

                # Extract content
                content = await self._extract_structured_content_optimized(page)

                if content and len(content.strip()) > 100:
                    content = self._clean_restaurant_content(content)
                    processing_time = time.time() - start_time

                    logger.info(f"✅ SUCCESS with browser-{browser_index}: {url} in {processing_time:.2f}s ({len(content)} chars)")

                    result = original_result.copy()
                    result.update({
                        "content": content,
                        "scraping_success": True,
                        "scraping_failed": False,
                        "processing_time": processing_time,
                        "content_length": len(content),
                        "browser_id": browser_index,
                        "extraction_method": "optimized_persistent"
                    })
                    return result
                else:
                    processing_time = time.time() - start_time
                    logger.warning(f"⚠️ LOW CONTENT with browser-{browser_index}: {url}")
                    result = original_result.copy()
                    result.update({
                        "content": "",
                        "scraping_success": False,
                        "scraping_failed": True,
                        "error": "Content too short or empty",
                        "processing_time": processing_time,
                        "browser_id": browser_index
                    })
                    return result

            finally:
                # OPTIMIZATION: Close page but keep browser/context alive
                try:
                    await page.close()
                except Exception as e:
                    logger.debug(f"Page close error (non-critical): {e}")

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ FAILED with browser-{browser_index}: {url} in {processing_time:.2f}s - {e}")

            result = original_result.copy()
            result.update({
                "scraping_success": False,
                "scraping_failed": True,
                "error": str(e),
                "processing_time": processing_time,
                "browser_id": browser_index
            })
            return result

    async def _configure_page_optimized(self, page: Page):
        """
        OPTIMIZED: Faster page configuration
        """
        try:
            # Minimal headers for speed
            await page.set_extra_http_headers({
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache"
            })

            # Block resources aggressively for speed
            await page.route("**/*", self._block_resources_aggressively)

            # Minimal script injection
            await page.add_init_script("""
                // Disable animations completely
                document.addEventListener('DOMContentLoaded', function() {
                    const style = document.createElement('style');
                    style.textContent = '*, *::before, *::after { animation: none !important; transition: none !important; }';
                    document.head.appendChild(style);
                });

                // Disable popups
                window.alert = window.confirm = window.prompt = () => {};
            """)

        except Exception as e:
            logger.warning(f"⚠️ Page configuration partially failed: {e}")

    async def _block_resources_aggressively(self, route):
        """
        OPTIMIZED: Aggressive resource blocking for maximum speed
        """
        resource_type = route.request.resource_type
        url = route.request.url

        # Block everything except essential HTML and scripts
        if resource_type in ['image', 'media', 'font', 'stylesheet', 'websocket', 'manifest']:
            await route.abort()
        elif any(domain in url for domain in [
            'google-analytics', 'googletagmanager', 'facebook.com', 
            'doubleclick', 'adsystem', 'amazon-adsystem', 'googlesyndication',
            'twitter.com', 'instagram.com', 'youtube.com', 'vimeo.com'
        ]):
            await route.abort()
        else:
            await route.continue_()

    async def _extract_structured_content_optimized(self, page: Page) -> str:
        """
        OPTIMIZED: Faster content extraction
        """
        try:
            # Quick dismiss of overlays
            await self._quick_dismiss_overlays(page)

            # Simplified content extraction
            content_data = await page.evaluate("""
                () => {
                    // Fast extraction focusing on main content only
                    const selectors = [
                        'main', 'article', '[role="main"]', '.main-content', '.content'
                    ];

                    let bestContent = '';

                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            // Remove unwanted elements
                            const unwanted = element.querySelectorAll('script, style, nav, header, footer, aside, .ad, .ads');
                            unwanted.forEach(el => el.remove());

                            const text = element.innerText || element.textContent || '';
                            if (text.length > bestContent.length) {
                                bestContent = text;
                            }
                        }
                    }

                    // Fallback to body if nothing found
                    if (!bestContent || bestContent.length < 200) {
                        const bodyText = document.body.innerText || document.body.textContent || '';
                        bestContent = bodyText.substring(0, 5000); // Limit to 5000 chars
                    }

                    return bestContent.trim();
                }
            """)

            return content_data if content_data and len(content_data.strip()) > 50 else ""

        except Exception as e:
            logger.warning(f"⚠️ Optimized extraction failed: {e}")
            try:
                fallback_content = await page.inner_text('body')
                return fallback_content[:3000] if fallback_content else ""
            except:
                return ""

    async def _quick_dismiss_overlays(self, page: Page):
        """
        OPTIMIZED: Quick overlay dismissal
        """
        overlay_selectors = [
            '.cookie-consent button',
            '.modal-close',
            '[aria-label="close"]',
            'button:contains("Accept")',
            'button:contains("Close")'
        ]

        for selector in overlay_selectors[:2]:  # Only try first 2 for speed
            try:
                await page.click(selector, timeout=1000)
                break
            except:
                continue

    def _clean_restaurant_content(self, content: str) -> str:
        """
        OPTIMIZED: Faster content cleaning
        """
        if not content:
            return ""

        # Quick cleanup
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'(cookie|privacy|newsletter|follow us)', '', content, flags=re.IGNORECASE)

        return content.strip()

    def get_stats(self) -> Dict[str, Any]:
        """
        OPTIMIZED: Return optimization statistics
        """
        stats = self.stats.copy()
        if self.stats["total_scraped"] > 0:
            stats["pool_efficiency"] = (self.stats["browser_reuses"] / self.stats["total_scraped"]) * 100
        return stats

    def print_stats(self):
        """
        OPTIMIZED: Print optimization statistics
        """
        if self.stats["total_scraped"] > 0:
            success_rate = (self.stats["successful_scrapes"] / self.stats["total_scraped"]) * 100
            pool_efficiency = (self.stats["browser_reuses"] / self.stats["total_scraped"]) * 100

            logger.info("=" * 60)
            logger.info("🚀 OPTIMIZED RESTAURANT SCRAPER STATISTICS")
            logger.info("=" * 60)
            logger.info(f"   📊 Success Rate: {success_rate:.1f}% ({self.stats['successful_scrapes']}/{self.stats['total_scraped']})")
            logger.info(f"   🔄 Browser Pool Size: {self.browser_pool_size}")
            logger.info(f"   ⚡ Browser Reuses: {self.stats['browser_reuses']}")
            logger.info(f"   🎯 Pool Efficiency: {pool_efficiency:.1f}%")
            logger.info(f"   ⏱️ Time Saved: ~{self.stats['connection_savings_estimate']:.1f}s")
            logger.info(f"   🚂 Railway Browserless: {'✓' if self.browserless_endpoint else '✗'}")
            logger.info("=" * 60)

    async def close(self):
        """
        OPTIMIZED: Clean shutdown of browser pool
        """
        logger.info("🛑 Shutting down optimized scraper...")

        # Close all contexts
        for context in self.browser_contexts:
            try:
                await context.close()
            except Exception as e:
                logger.debug(f"Context close error: {e}")

        # Close all browsers
        for browser in self.browser_pool:
            try:
                await browser.close()
            except Exception as e:
                logger.debug(f"Browser close error: {e}")

        # Stop playwright
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                logger.debug(f"Playwright stop error: {e}")

        self.session_active = False
        self.print_stats()
        logger.info("✅ Optimized scraper shutdown complete")

    # Legacy compatibility methods
    async def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """Legacy compatibility method"""
        search_results = [{"url": url, "title": "", "snippet": ""} for url in urls]
        return await self.scrape_search_results(search_results)