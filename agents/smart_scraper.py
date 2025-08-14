# agents/smart_scraper.py
"""
UPDATED: WebKit Smart Scraper - Memory Optimized
Uses WebKit browser for 50% less memory usage than Chromium
Fallback to Firefox if WebKit fails
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from utils.database import get_database

logger = logging.getLogger(__name__)

class SmartRestaurantScraper:
    """
    Memory-optimized Smart Scraper using WebKit with Firefox fallback
    Strategy: Human Mimic for everything (2.0 credits per URL)
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()
        self.max_concurrent = 1  
        self.browser = None
        self.contexts = []
        self.browser_type = "webkit"  # NEW: Primary browser choice

        # Progressive timeout strategy
        self.default_timeout = 30000  # 30 seconds default
        self.slow_timeout = 60000     # 60 seconds for slow sites
        self.domain_timeouts = {
            # Known slow domains
            'guide.michelin.com': 60000,
            'timeout.com': 45000,
        }

        # Human-like timing
        self.load_wait_time = 3.0      # Human reading time after load
        self.interaction_delay = 0.5    # Delay between actions
        self.retry_delay = 2.0          # Delay between retries

        # Simplified stats tracking
        self.stats = {
            "total_processed": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_cost_estimate": 0.0,
            "total_processing_time": 0.0,
            "strategy_breakdown": {"human_mimic": 0},
            "avg_load_time": 0.0,
            "total_load_time": 0.0,
            "concurrent_peak": 0,
            "browser_used": "webkit"  # NEW: Track which browser was used
        }

        logger.info("✅ Smart Restaurant Scraper initialized with WebKit (memory optimized)")

    async def start(self):
        """Initialize Playwright and browser contexts with WebKit (memory optimized)"""
        if self.browser:
            return  # Already started

        logger.info("🎭 Starting Memory-Optimized Browser (WebKit primary, Firefox fallback)...")

        self.playwright = await async_playwright().start()

        # Try WebKit first (lightest option)
        try:
            await self._launch_webkit()
            self.stats["browser_used"] = "webkit"
            logger.info("✅ WebKit browser launched successfully")
        except Exception as webkit_error:
            logger.warning(f"⚠️ WebKit failed: {webkit_error}")
            try:
                await self._launch_firefox()
                self.stats["browser_used"] = "firefox"
                logger.info("✅ Firefox fallback launched successfully")
            except Exception as firefox_error:
                logger.warning(f"⚠️ Firefox failed: {firefox_error}")
                await self._launch_chromium_minimal()
                self.stats["browser_used"] = "chromium_minimal"
                logger.info("✅ Minimal Chromium launched as last resort")

        # Create optimized contexts
        await self._create_optimized_contexts()

    async def _launch_webkit(self):
        """Launch WebKit browser (lightest memory footprint)"""
        self.browser = await self.playwright.webkit.launch(
            headless=True,
            # WebKit has fewer args than Chromium
            args=[
                '--disable-web-security',  # May help with some sites
                '--disable-features=VizDisplayCompositor',  # Memory saver
            ]
        )
        self.browser_type = "webkit"

    async def _launch_firefox(self):
        """Launch Firefox browser (medium memory footprint)"""
        self.browser = await self.playwright.firefox.launch(
            headless=True,
            # Firefox-specific memory optimizations
            args=[
                '--memory-pressure-off',
                '--no-remote',
            ]
        )
        self.browser_type = "firefox"

    async def _launch_chromium_minimal(self):
        """Launch minimal Chromium as last resort"""
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-networking',
                '--disable-default-apps',
                '--disable-extensions',
                '--disable-plugins',
                '--memory-pressure-off',
                '--max_old_space_size=512',  # REDUCED: From 4096 to 512MB
                '--single-process',  # KEY: Single process mode for memory savings
                '--disable-features=VizDisplayCompositor',
            ]
        )
        self.browser_type = "chromium_minimal"

    async def _create_optimized_contexts(self):
        """Create memory-optimized browser contexts"""
        for i in range(self.max_concurrent):
            context = await self.browser.new_context(
                viewport={'width': 1366, 'height': 768},  # REDUCED: Smaller viewport
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                # Block unnecessary permissions that can slow things down
                permissions=[],
                geolocation=None,
                ignore_https_errors=True,
                # NEW: Additional memory optimizations
                java_script_enabled=True,  # Keep JS for dynamic content
                bypass_csp=True,  # May help with some sites
            )

            self.contexts.append(context)

        logger.info(f"✅ {len(self.contexts)} browser contexts ready with {self.browser_type}")

    async def _configure_page_with_adblock(self, page: Page):
        """Configure page with lightweight ad blocking for memory savings"""
        # 1. Block images and media (major memory savings)
        await page.route("**/*.{png,jpg,jpeg,gif,svg,webp,ico,css,woff,woff2}", lambda route: route.abort())

        # 2. Block ad/tracking domains (lightweight but effective)
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

        # 3. Set faster user agent (optional)
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
        })

    async def stop(self):
        """Clean up all browser resources"""
        try:
            # Close all contexts
            for context in self.contexts:
                if context:
                    await context.close()
            self.contexts.clear()

            # Close browser
            if self.browser:
                await self.browser.close()
                self.browser = None

            # Stop Playwright
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None

            logger.info(f"🎭 {self.browser_type} scraper stopped")

        except Exception as e:
            logger.error(f"Error stopping {self.browser_type} scraper: {e}")

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape URL with concurrency control using specific context"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    async def _scrape_single_url(self, url: str, context_index: int = 0) -> Dict[str, Any]:
        """
        Scrape a single URL - Simple human mimic: select all, copy, save as RTF
        UPDATED: Enhanced error handling and memory cleanup
        """
        domain = urlparse(url).netloc
        initial_timeout = self.default_timeout

        start_time = time.time()
        page = None
        final_timeout = initial_timeout

        try:
            logger.info(f"🎭 Context-{context_index} scraping: {url} (timeout: {initial_timeout/1000}s, browser: {self.browser_type})")

            # Get the appropriate context
            context = self.contexts[context_index % len(self.contexts)]
            page = await context.new_page()

            # Configure page for optimal performance (includes ad blocking now)
            await self._configure_page_with_adblock(page)

            # Smart timeout strategy
            timeouts_to_try = [initial_timeout]
            if domain in self.domain_timeouts:
                timeouts_to_try = [self.domain_timeouts[domain]]
            elif domain in ['guide.michelin.com', 'timeout.com', 'zagat.com']:
                timeouts_to_try = [self.slow_timeout]

            load_success = False

            for timeout_attempt, timeout_ms in enumerate(timeouts_to_try, 1):
                try:
                    final_timeout = timeout_ms
                    page.set_default_timeout(timeout_ms)
                    page.set_default_navigation_timeout(timeout_ms)

                    logger.info(f"🌐 Attempt {timeout_attempt}: Loading {domain} (timeout: {timeout_ms/1000}s)")

                    await page.goto(url, wait_until='domcontentloaded')
                    load_success = True
                    logger.info(f"✅ Page loaded successfully on attempt {timeout_attempt}")
                    break

                except Exception as timeout_error:
                    logger.warning(f"⚠️ Timeout attempt {timeout_attempt} failed for {url}: {timeout_error}")
                    if timeout_attempt == len(timeouts_to_try):
                        logger.error(f"❌ All timeout attempts failed for {url}")
                        raise

            if not load_success:
                raise Exception("Page failed to load after all timeout attempts")

            # Human-like behavior: wait and read the page
            await asyncio.sleep(self.load_wait_time)

            # KEYBOARD SELECTION: Use Ctrl+A for broader compatibility
            await asyncio.sleep(self.interaction_delay)

            # Use different key combinations based on browser
            if self.browser_type == "webkit":
                await page.keyboard.press('Meta+a')  # Mac-style for WebKit
            else:
                await page.keyboard.press('Control+a')  # Standard for Firefox/Chromium

            await asyncio.sleep(self.interaction_delay)

            # SIMPLIFIED RTF - Keep only essential formatting for AI readability
            rtf_content = await page.evaluate("""
                () => {
                    const selection = window.getSelection();
                    if (selection.rangeCount > 0) {
                        const range = selection.getRangeAt(0);
                        const div = document.createElement('div');
                        div.appendChild(range.cloneContents());

                        // Clean up the content - keep structure, remove noise
                        const elements = div.querySelectorAll('*');
                        elements.forEach(el => {
                            // Remove hidden elements and noise
                            const style = window.getComputedStyle ? window.getComputedStyle(el) : el.style;
                            if (style && (
                                style.display === 'none' || 
                                style.visibility === 'hidden' ||
                                el.tagName === 'SCRIPT' ||
                                el.tagName === 'STYLE' ||
                                el.tagName === 'NOSCRIPT'
                            )) {
                                el.remove();
                            }
                        });

                        return div.textContent || div.innerText || '';
                    }
                    return document.body.textContent || document.body.innerText || '';
                }
            """)

            if not rtf_content or len(rtf_content.strip()) < 50:
                raise Exception("No substantial content found after selection")

            # Clean the content
            cleaned_content = self._clean_scraped_text(rtf_content)

            processing_time = time.time() - start_time
            self.stats["total_load_time"] += processing_time
            self.stats["successful_scrapes"] += 1
            self.stats["total_processed"] += 1

            logger.info(f"✅ Successfully scraped {url} ({len(cleaned_content)} chars, {processing_time:.2f}s, {self.browser_type})")

            return {
                "success": True,
                "url": url,
                "content": cleaned_content,
                "processing_time": processing_time,
                "content_length": len(cleaned_content),
                "browser_used": self.browser_type,
                "strategy": "human_mimic"
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_scrapes"] += 1
            self.stats["total_processed"] += 1

            logger.error(f"❌ Failed to scrape {url}: {e} (took {processing_time:.2f}s, {self.browser_type})")

            return {
                "success": False,
                "url": url,
                "error": str(e),
                "processing_time": processing_time,
                "browser_used": self.browser_type,
                "strategy": "human_mimic"
            }

        finally:
            # IMPORTANT: Always close page to free memory
            if page:
                try:
                    await page.close()
                except:
                    pass  # Ignore close errors

    def _clean_scraped_text(self, text: str) -> str:
        """
        Clean the scraped text to match what human would see and copy
        Optimized for restaurant content extraction
        """
        if not text:
            return ""

        # Remove excessive whitespace while preserving structure
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        cleaned = re.sub(r' {2,}', ' ', cleaned)  # Multiple spaces -> single space
        cleaned = cleaned.strip()

        # Remove common navigation/footer noise (restaurant site specific)
        noise_patterns = [
            r'Cookie Policy.*?(?=\n|$)',
            r'Privacy Policy.*?(?=\n|$)', 
            r'Terms.*?Service.*?(?=\n|$)',
            r'Subscribe.*?newsletter.*?(?=\n|$)',
            r'Follow us.*?(?=\n|$)',
            r'Download.*?app.*?(?=\n|$)',
            r'Sign up.*?alerts.*?(?=\n|$)',
            r'Advertisement\n',
            r'Skip to.*?content',
            r'Accept.*?cookies.*?(?=\n|$)',
        ]

        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Restaurant content optimization
        # Preserve important structural elements
        cleaned = re.sub(r'(\$\d+)', r' \1 ', cleaned)  # Space around prices
        cleaned = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', cleaned)  # Fix time ranges

        return cleaned.strip()

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point - Process search results with concurrent human mimicking
        UPDATED: Single concurrent scraping for memory optimization
        """
        if not search_results:
            return []

        if not self.browser:
            await self.start()

        urls = [result.get('url') for result in search_results if result.get('url')]
        logger.info(f"🎭 Memory-optimized scraping {len(urls)} URLs with {self.max_concurrent} concurrent contexts ({self.browser_type})")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Scrape all URLs concurrently - filter None URLs properly
        valid_urls = [(i, url) for i, url in enumerate(urls) if url is not None]
        scrape_tasks = [
            self._scrape_url_with_semaphore(semaphore, url, i % len(self.contexts))
            for i, url in valid_urls
        ]

        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        # Merge scraping results back with search results
        enriched_results = []
        for search_result, scrape_result in zip(search_results, scrape_results):
            enriched = search_result.copy()

            if isinstance(scrape_result, Exception):
                logger.error(f"Exception scraping {enriched.get('url')}: {scrape_result}")
                enriched.update({
                    "success": False,
                    "error": str(scrape_result),
                    "content": "",
                    "browser_used": self.browser_type,
                    "strategy": "human_mimic"
                })
                self.stats["failed_scrapes"] += 1
            else:
                enriched.update(scrape_result)

            self.stats["strategy_breakdown"]["human_mimic"] += 1
            self.stats["total_cost_estimate"] += 2.0  # 2.0 credits per URL
            enriched_results.append(enriched)

        successful = sum(1 for r in scrape_results if isinstance(r, dict) and r.get('success'))
        logger.info(f"✅ Memory-optimized batch complete: {successful}/{len(urls)} successful ({self.browser_type})")

        return enriched_results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics"""
        return {
            **self.stats,
            "success_rate": (self.stats["successful_scrapes"] / max(self.stats["total_processed"], 1)) * 100,
            "concurrent_contexts": len(self.contexts),
            "memory_optimized": True,
            "browser_used": self.browser_type
        }

    def _log_stats(self):
        """Log processing statistics"""
        if self.stats["total_processed"] > 0:
            logger.info("📊 MEMORY-OPTIMIZED SCRAPER STATISTICS:")
            logger.info(f"   🎭 Human Mimic: {self.stats['strategy_breakdown']['human_mimic']} URLs")
            logger.info(f"   🌐 Browser: {self.browser_type}")
            logger.info(f"   ✅ Success Rate: {self.stats['successful_scrapes']}/{self.stats['total_processed']} ({(self.stats['successful_scrapes']/self.stats['total_processed']*100):.1f}%)")
            logger.info(f"   💰 Cost Estimate: {self.stats['total_cost_estimate']:.1f} credits")
            logger.info(f"   🧠 Memory Optimized: {self.max_concurrent} concurrent context(s)")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup browser"""
        await self.stop()