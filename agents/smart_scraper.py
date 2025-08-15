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
        self.browser: Optional[Browser] = None
        self.context: List[BrowserContext] = []
        self.browser_type = "webkit"  # NEW: Primary browser choice
        self.playwright = None

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
        """Initialize Playwright and browser context with WebKit (memory optimized)"""
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

        # Create optimized context
        await self._create_optimized_context()

    async def _configure_page_with_adblock(self, page: Page):
        """
        ENHANCED: Ultra-aggressive resource blocking for text-only scraping
        Blocks everything except HTML and essential JavaScript while preserving text structure
        """
        # 1. COMPREHENSIVE MEDIA BLOCKING - Block ALL visual/audio content
        await page.route("**/*.{png,jpg,jpeg,gif,svg,webp,ico,bmp,tiff,avif,heic,heif}", lambda route: route.abort())
        await page.route("**/*.{woff,woff2,ttf,otf,eot}", lambda route: route.abort())  # Fonts
        await page.route("**/*.{css}", lambda route: route.abort())  # Stylesheets
        await page.route("**/*.{mp4,avi,mov,wmv,flv,webm,m4v,mp3,wav,aac,ogg,flac}", lambda route: route.abort())  # Media
        await page.route("**/*.{pdf,doc,docx,xls,xlsx,ppt,pptx}", lambda route: route.abort())  # Documents

        # 2. ENHANCED AD/TRACKING DOMAIN BLOCKING
        blocked_domains = [
            # Major ad networks
            'doubleclick.net', 'googleadservices.com', 'googlesyndication.com',
            'adsystem.amazon.com', 'amazon-adsystem.com', 'facebook.com/tr',
            'googletagmanager.com', 'google-analytics.com', 'googleanalytics.com',
            # Analytics & tracking
            'hotjar.com', 'crazyegg.com', 'mouseflow.com', 'fullstory.com',
            'mixpanel.com', 'segment.com', 'amplitude.com', 'chartbeat.com',
            'quantserve.com', 'scorecardresearch.com', 'omtrdc.net',
            # Social media widgets  
            'twitter.com/widgets', 'instagram.com/embed', 'youtube.com/embed',
            'facebook.com/plugins', 'linkedin.com/embed',
            # CDNs serving primarily media/ads
            'imgur.com', 'giphy.com', 'tenor.com', 'cloudinary.com', 'imagekit.io',
            # Comment systems
            'disqus.com', 'disquscdn.com', 'livefyre.com'
        ]

        await page.route("**/*", lambda route: (
            route.abort() if any(domain in route.request.url for domain in blocked_domains)
            else route.continue_()
        ))

        # 3. RESOURCE TYPE BLOCKING - Block by request type  
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in [
                'image', 'media', 'font', 'stylesheet', 'manifest', 'other'
            ] else route.continue_()
        ))

        # 4. TEXT-ONLY HEADERS
        await page.set_extra_http_headers({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Cache-Control': 'no-cache',
            'DNT': '1'  # Do Not Track
        })

        # 5. DISABLE IMAGES AT BROWSER LEVEL
        await page.add_init_script("""
            // Disable image loading completely
            Object.defineProperty(HTMLImageElement.prototype, 'src', {
                set: function() { /* do nothing */ },
                get: function() { return ''; }
            });

            // Disable background images
            const originalSetProperty = CSSStyleDeclaration.prototype.setProperty;
            CSSStyleDeclaration.prototype.setProperty = function(property, value, priority) {
                if (property === 'background-image' || property === 'background') {
                    return;
                }
                return originalSetProperty.call(this, property, value, priority);
            };

            // Speed up animations for faster text loading
            document.addEventListener('DOMContentLoaded', function() {
                const style = document.createElement('style');
                style.textContent = `
                    *, *::before, *::after {
                        animation-duration: 0.01ms !important;
                        animation-delay: -0.01ms !important;
                        transition-duration: 0.01ms !important;
                        transition-delay: -0.01ms !important;
                    }
                `;
                document.head.appendChild(style);
            });
        """)

    async def _launch_webkit(self):
        """WebKit launch with ONLY WebKit-compatible arguments"""
        self.browser = await self.playwright.webkit.launch(
            headless=True
            # NO ARGUMENTS - WebKit is very picky about args
        )
        self.browser_type = "webkit"

    async def _launch_firefox(self):
        """Enhanced Firefox launch optimized for text extraction"""
        self.browser = await self.playwright.firefox.launch(
            headless=True,
            firefox_user_prefs={
                # DISABLE IMAGES AND MEDIA
                'permissions.default.image': 2,          # Block images
                'media.autoplay.enabled': False,         # No autoplay
                # PERFORMANCE OPTIMIZATIONS  
                'browser.cache.disk.enable': False,      # No disk cache
                'browser.cache.memory.enable': True,     # Memory cache only
                'network.dns.disableIPv6': True,         # IPv4 only (faster)
                # PRIVACY (faster loading)
                'privacy.trackingprotection.enabled': True,   # Block trackers
                'dom.webnotifications.enabled': False,        # No notifications
                'geo.enabled': False,                          # No geolocation
            },
            args=[
                '--memory-pressure-off',
                '--no-remote'
            ]
        )
        self.browser_type = "firefox"

    async def _launch_chromium_minimal(self):
        """Enhanced Chromium with aggressive text-only optimizations"""
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                # EXISTING ARGS
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-networking',
                '--disable-default-apps',
                '--disable-extensions',
                '--disable-plugins',
                '--memory-pressure-off',
                '--max_old_space_size=256',  # REDUCED: From 512 to 256MB
                '--single-process',
                '--disable-features=VizDisplayCompositor',

                # NEW TEXT-ONLY OPTIMIZATIONS
                '--blink-settings=imagesEnabled=false',      # Disable images at Blink level
                '--disable-images',                          # Additional image blocking
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI,BlinkGenPropertyTrees',
                '--disable-audio-output',                    # No audio needed
                '--mute-audio',                             # Mute everything
                '--disable-ipc-flooding-protection',        # Faster IPC
                '--disable-domain-reliability',
            ]
        )
        self.browser_type = "chromium_minimal"

    async def _create_optimized_context(self):
        """Enhanced context creation with text-only optimizations"""
        if not self.browser:
            raise RuntimeError("Browser not initialized")

        for i in range(self.max_concurrent):
            context = await self.browser.new_context(
                viewport={'width': 1024, 'height': 600},  # REDUCED: Smaller viewport for less rendering
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 TextExtractor/1.0',
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Cache-Control': 'no-cache',
                    'DNT': '1',  # Do Not Track
                },
                permissions=[],
                geolocation=None,
                ignore_https_errors=True,
                java_script_enabled=True,  # Keep JS for dynamic content
                bypass_csp=True,
            )

            self.context.append(context)

        logger.info(f"✅ {len(self.context)} text-optimized context ready ({self.browser_type})")

    async def stop(self):
        """Clean up all browser resources"""
        try:
            # Close all contexts
            for context in self.context:
                if context:
                    await context.close()
            self.context.clear()

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
        page: Optional[Page] = None
        final_timeout = initial_timeout

        try:
            logger.info(f"🎭 Context-{context_index} scraping: {url} (timeout: {initial_timeout/1000}s, browser: {self.browser_type})")

            # Get the appropriate context
            context = self.context[context_index % len(self.context)]
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
        logger.info(f"🎭 Memory-optimized scraping {len(urls)} URLs with {self.max_concurrent} concurrent context ({self.browser_type})")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Scrape all URLs concurrently - filter None URLs properly
        valid_urls = [(i, url) for i, url in enumerate(urls) if url is not None]
        scrape_tasks = [
            self._scrape_url_with_semaphore(semaphore, url, i % len(self.context))
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
            "concurrent_context": len(self.context),
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