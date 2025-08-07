# agents/smart_scraper.py
"""
Simplified Smart Scraper - Human Mimic Only
Uses Human Mimic for all scraping, no filtering or specialized handlers
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
    Simplified Smart Scraper using Human Mimic for all URLs

    Strategy: Human Mimic for everything (2.0 credits per URL)
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()
        self.max_concurrent = 2
        self.browser = None
        self.contexts = []

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
            "strategy_breakdown": {"human_mimic": 0},
            "avg_load_time": 0.0,
            "total_load_time": 0.0,
            "concurrent_peak": 0
        }

        logger.info("✅ SmartRestaurantScraper initialized with Human Mimic only")

    async def start(self):
        """Initialize Playwright and browser contexts for production with ad blocking"""
        if self.browser:
            return  # Already started

        logger.info("🎭 Starting Production Human Mimic Browser with Ad Blocking...")

        self.playwright = await async_playwright().start()

        # Launch browser with production + ad blocking optimized settings
        self.browser = await self.playwright.chromium.launch(
            headless=True,  # Always headless in production
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',  # Important for Railway
                '--disable-web-security',   # May help with some sites
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-background-networking',
                '--disable-ipc-flooding-protection',
                # Ad blocking related flags
                '--disable-default-apps',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-sync',
                '--no-default-browser-check',
                '--no-first-run',
                # Performance optimizations
                '--memory-pressure-off',
                '--max_old_space_size=4096'
            ]
        )

        # Create optimized contexts with ad blocking
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

        logger.info(f"✅ {len(self.contexts)} browser contexts ready with ad blocking")

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

            logger.info("🎭 Human mimic scraper stopped")

        except Exception as e:
            logger.error(f"Error stopping human mimic scraper: {e}")

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

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape URL with concurrency control using specific context"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    async def _scrape_single_url(self, url: str, context_index: int = 0) -> Dict[str, Any]:
        """
        Scrape a single URL with NO page reload on timeout + ad blocking
        """
        domain = urlparse(url).netloc
        initial_timeout = self.default_timeout

        start_time = time.time()
        page = None
        final_timeout = initial_timeout  # Initialize final_timeout

        try:
            logger.info(f"🎭 Context-{context_index} scraping: {url} (timeout: {initial_timeout/1000}s)")

            # Get the appropriate context
            context = self.contexts[context_index % len(self.contexts)]
            page = await context.new_page()

            # Configure page for optimal performance (includes ad blocking now)
            await self._configure_page_with_adblock(page)

            # Smart timeout strategy - NO PAGE RELOAD
            timeout_attempts = [
                initial_timeout,                    # First try: learned/default timeout
                max(initial_timeout * 1.5, 45000), # Second try: 50% longer or 45s minimum
                60000                               # Final try: 60s maximum
            ]

            load_success = False
            final_timeout = initial_timeout

            for attempt, timeout in enumerate(timeout_attempts):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}/{len(timeout_attempts)}: {timeout/1000}s timeout")

                    # Try to load page with current timeout
                    await page.goto(url, wait_until='networkidle', timeout=timeout)

                    # If we get here, page loaded successfully
                    load_success = True
                    final_timeout = timeout

                    break  # Success - exit timeout attempts

                except Exception as e:
                    if "timeout" in str(e).lower() and attempt < len(timeout_attempts) - 1:
                        # Timeout occurred, but we have more attempts
                        logger.info(f"⏱️ Timeout at {timeout/1000}s, trying {timeout_attempts[attempt + 1]/1000}s...")
                        continue  # Try next timeout WITHOUT reloading page
                    else:
                        # Either not a timeout, or we've exhausted all attempts
                        if attempt >= len(timeout_attempts) - 1:
                            logger.error(f"❌ All timeout attempts failed for {url}")
                        raise

            if not load_success:
                raise Exception("Page failed to load after all timeout attempts")

            # Human-like behavior: wait and read the page
            await asyncio.sleep(self.load_wait_time)

            # The "dumb but modern" magic: Select All + Extract
            await asyncio.sleep(self.interaction_delay)
            await page.keyboard.press('Meta+a')  # Cmd+A (works on most systems)
            await asyncio.sleep(self.interaction_delay)

            # Get selected text (human clipboard equivalent)
            selected_text = await page.evaluate("""
                () => {
                    const selection = window.getSelection();
                    if (selection.rangeCount > 0) {
                        return selection.toString();
                    }
                    // Fallback: get all visible text
                    return document.body.innerText || document.body.textContent || '';
                }
            """)

            # Clean the scraped content
            cleaned_content = self._clean_scraped_text(selected_text)
            load_time = time.time() - start_time

            # Update stats
            self.stats["total_scraped"] = self.stats.get("total_scraped", 0) + 1
            self.stats["successful_scrapes"] += 1
            self.stats["total_load_time"] = self.stats.get("total_load_time", 0) + load_time
            self.stats["avg_load_time"] = self.stats["total_load_time"] / self.stats.get("total_scraped", 1)

            logger.info(f"✅ Context-{context_index} scraped {len(cleaned_content)} chars in {load_time:.2f}s (final timeout: {final_timeout/1000}s)")

            return {
                "success": True,
                "content": cleaned_content,
                "url": url,
                "load_time": load_time,
                "char_count": len(cleaned_content),
                "method": "human_mimic",
                "context_used": context_index,
                "timeout_used": final_timeout,
                "error": None
            }

        except Exception as e:
            load_time = time.time() - start_time
            error_msg = str(e)

            self.stats["total_scraped"] = self.stats.get("total_scraped", 0) + 1
            self.stats["failed_scrapes"] += 1

            logger.error(f"❌ Context-{context_index} failed scraping {url}: {error_msg}")

            return {
                "success": False,
                "content": "",
                "url": url,
                "load_time": load_time,
                "char_count": 0,
                "method": "human_mimic",
                "context_used": context_index,
                "timeout_used": final_timeout,
                "error": error_msg
            }

        finally:
            if page:
                await page.close()

    async def _configure_page_with_adblock(self, page: Page):
        """
        Configure page with ad blocking for dramatically faster loading
        This can improve load times by 20-40% according to research
        """

        # 1. Block resource types that slow down loading
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in [
                'image',      # Images (biggest bandwidth hog)
                'media',      # Videos/audio  
                'font',       # Web fonts
                'stylesheet', # CSS (optional - may break layout but speeds up loading)
                'beacon',     # Analytics beacons
                'csp_report', # Security reports
                'object',     # Flash/embed objects
                'texttrack'   # Video subtitles
            ] else route.continue_()
        ))

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

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point - Process search results with concurrent human mimicking
        """
        if not search_results:
            return []

        if not self.browser:
            await self.start()

        urls = [result.get('url') for result in search_results if result.get('url')]
        logger.info(f"🎭 Human mimic scraping {len(urls)} URLs with {self.max_concurrent} concurrent contexts")

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
                    'scraped_content': "",
                    'scraping_success': False,
                    'scraping_method': 'human_mimic',
                    'scraping_error': str(scrape_result),
                    'load_time': 0.0
                })
            elif isinstance(scrape_result, dict):
                enriched.update({
                    'scraped_content': scrape_result['content'],
                    'scraping_success': scrape_result['success'],
                    'scraping_method': 'human_mimic',
                    'scraping_error': scrape_result.get('error'),
                    'load_time': scrape_result['load_time'],
                    'char_count': scrape_result['char_count']
                })
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

            # Update stats
            if enriched.get('scraping_success'):
                self.stats["successful_scrapes"] += 1
            else:
                self.stats["failed_scrapes"] += 1

            self.stats["total_processed"] += 1
            self.stats["strategy_breakdown"]["human_mimic"] += 1
            self.stats["total_cost_estimate"] += 2.0  # 2.0 credits per URL
            enriched_results.append(enriched)

        # Apply text cleaning to successful scrapes
        await self._apply_text_cleaning(enriched_results)

        successful = sum(1 for r in scrape_results if isinstance(r, dict) and r.get('success'))
        logger.info(f"✅ Human mimic batch complete: {successful}/{len(urls)} successful")

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
        """Get comprehensive scraping statistics"""
        return {
            **self.stats,
            "success_rate": (self.stats["successful_scrapes"] / max(self.stats["total_processed"], 1)) * 100,
            "concurrent_contexts": len(self.contexts)
        }

    def _log_stats(self):
        """Log processing statistics"""
        if self.stats["total_processed"] > 0:
            logger.info("📊 SMART SCRAPER STATISTICS:")
            logger.info(f"   🎭 Human Mimic: {self.stats['strategy_breakdown']['human_mimic']} URLs")
            logger.info(f"   ✅ Success Rate: {self.stats['successful_scrapes']}/{self.stats['total_processed']} ({(self.stats['successful_scrapes']/self.stats['total_processed']*100):.1f}%)")
            logger.info(f"   💰 Cost Estimate: {self.stats['total_cost_estimate']:.1f} credits")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup browser"""
        await self.stop()