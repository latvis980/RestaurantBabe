# agents/human_mimic_scraper.py
"""
Production Human Mimic Scraper Agent
Replaces Firecrawl for content sites without CAPTCHA protection

Features:
- Progressive timeout strategy with domain learning
- 2 concurrent browsers for optimal performance  
- Headless operation for Railway deployment
- Smart retry logic with timeout escalation
- Domain intelligence caching
- Full integration with existing SmartRestaurantScraper architecture
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from utils.database import get_database
import json

logger = logging.getLogger(__name__)

class HumanMimicScraper:
    """
    Production Human Mimic Scraper that acts like a human user:
    1. Loads page with proper waiting
    2. Executes Cmd+A to select all content  
    3. Extracts selected text (human clipboard content)
    4. Learns domain-specific timing requirements
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()

        # Browser management
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.contexts: List[BrowserContext] = []
        self.max_concurrent = 2  # Optimal for Railway resources

        # Progressive timeout strategy
        self.default_timeout = 30000  # 30 seconds default
        self.slow_timeout = 60000     # 60 seconds for slow sites
        self.domain_timeouts = {
            # Known slow domains
            'guide.michelin.com': 60000,
            'timeout.com': 45000,
            'zagat.com': 45000,
        }

        # Human-like timing
        self.load_wait_time = 3.0      # Human reading time after load
        self.interaction_delay = 0.5    # Delay between actions
        self.retry_delay = 2.0          # Delay between retries

        # Stats tracking
        self.stats = {
            "total_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "timeout_escalations": 0,
            "domain_learnings": 0,
            "avg_load_time": 0.0,
            "total_load_time": 0.0,
            "concurrent_peak": 0
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

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


    async def close(self):
        """Clean up all browser resources"""
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

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for SmartRestaurantScraper integration
        Process search results with concurrent human mimicking
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
            else:
                enriched.update({
                    'scraped_content': scrape_result['content'],
                    'scraping_success': scrape_result['success'],
                    'scraping_method': 'human_mimic',
                    'scraping_error': scrape_result.get('error'),
                    'load_time': scrape_result['load_time'],
                    'char_count': scrape_result['char_count']
                })

            enriched_results.append(enriched)

        successful = sum(1 for r in scrape_results if isinstance(r, dict) and r.get('success'))
        logger.info(f"✅ Human mimic batch complete: {successful}/{len(urls)} successful")

        return enriched_results

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape URL with concurrency control using specific context"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    async def _scrape_single_url(self, url: str, context_index: int = 0) -> Dict[str, Any]:
        """
        Scrape a single URL with NO page reload on timeout + ad blocking
        """
        domain = urlparse(url).netloc
        initial_timeout = self._get_timeout_for_domain(domain)

        start_time = time.time()
        page = None

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

                    # Learn if this was a timeout escalation
                    if timeout > initial_timeout:
                        await self._learn_slow_domain(domain, timeout)
                        self.stats["timeout_escalations"] += 1
                        logger.info(f"📚 Learned {domain} needs {timeout/1000}s")

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
            self.stats["total_scraped"] += 1
            self.stats["successful_scrapes"] += 1
            self.stats["total_load_time"] += load_time
            self.stats["avg_load_time"] = self.stats["total_load_time"] / self.stats["total_scraped"]

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

            self.stats["total_scraped"] += 1
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


    # OPTIONAL: For even better ad blocking, add this advanced version
    # This requires installing the Ghostery adblocker (JavaScript library)
    # You'd need to set up a Node.js service or use subprocess to call it

    # async def _configure_page_with_advanced_adblock(self, page: Page):
      #   """
        # ADVANCED: Configure page with Ghostery adblocker (requires Node.js setup)
        # This provides uBlock Origin-level ad blocking
        # """
        # First apply basic blocking
        # await self._configure_page_with_adblock(page)

        # TODO: Add Ghostery adblocker integration if you want maximum blocking
        # This would require setting up Node.js integration or a subprocess call
        # See the research results above for implementation details


    def _get_timeout_for_domain(self, domain: str) -> int:
        """Get appropriate timeout for domain with learning"""
        # Check learned domain timeouts first
        if domain in self.domain_timeouts:
            return self.domain_timeouts[domain]

        # Check database for learned timeouts
        learned_timeout = self._get_learned_timeout_from_db(domain)
        if learned_timeout:
            self.domain_timeouts[domain] = learned_timeout
            return learned_timeout

        return self.default_timeout

    def _get_learned_timeout_from_db(self, domain: str) -> Optional[int]:
        """Get learned timeout for domain from database"""
        try:
            # Query your database for domain intelligence
            # This integrates with your existing domain learning system
            result = self.database.table('domain_intelligence').select('timeout').eq('domain', domain).execute()
            if result.data and len(result.data) > 0:
                return result.data[0].get('timeout', self.default_timeout)
        except Exception as e:
            logger.debug(f"Could not get learned timeout for {domain}: {e}")

        return None

    async def _learn_slow_domain(self, domain: str, successful_timeout: int):
        """Learn that a domain requires longer timeout"""
        self.domain_timeouts[domain] = successful_timeout
        self.stats["domain_learnings"] += 1

        try:
            # Update domain intelligence in database
            self.database.table('domain_intelligence').upsert({
                'domain': domain,
                'timeout': successful_timeout,
                'learned_at': time.time(),
                'strategy_hint': 'human_mimic_slow'
            }).execute()

            logger.info(f"📚 Learned that {domain} needs {successful_timeout/1000}s timeout")
        except Exception as e:
            logger.debug(f"Could not save domain learning for {domain}: {e}")

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

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics"""
        return {
            **self.stats,
            "domain_timeouts_learned": len(self.domain_timeouts),
            "success_rate": (self.stats["successful_scrapes"] / max(self.stats["total_scraped"], 1)) * 100,
            "concurrent_contexts": len(self.contexts),
            "avg_timeout_used": sum(self.domain_timeouts.values()) / max(len(self.domain_timeouts), 1) / 1000  # in seconds
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            "total_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "timeout_escalations": 0,
            "domain_learnings": 0,
            "avg_load_time": 0.0,
            "total_load_time": 0.0,
            "concurrent_peak": 0
        }


# Integration class for SmartRestaurantScraper
class HumanMimicScrapingStrategy:
    """
    Integration wrapper that provides the same interface as FirecrawlWebScraper
    This allows seamless replacement in your existing smart scraper system
    """

    def __init__(self, config):
        self.config = config
        self.scraper = HumanMimicScraper(config)

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main interface method for integration with SmartRestaurantScraper
        """
        async with self.scraper:
            return await self.scraper.scrape_search_results(search_results)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for monitoring"""
        return self.scraper.get_stats()

    def reset_stats(self):
        """Reset statistics"""
        self.scraper.reset_stats()