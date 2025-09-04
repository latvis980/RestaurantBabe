# agents/smart_scraper.py - FIXED VERSION
"""
FIXED: Smart Scraper with Proper Session Management
- Fixed race condition between cleanup and active operations  
- Added operation locks to prevent premature browser closure
- Improved Railway resource management
- Fixed context lifecycle management
"""

import asyncio
import logging
import time
import re
import os
import threading
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
from utils.database import get_database

logger = logging.getLogger(__name__)


class SmartRestaurantScraper:
    def __init__(self, config):
        self.config = config
        self.database = get_database()
        self.max_concurrent = 2  
        self.browser: Optional[Browser] = None
        self.context: List[BrowserContext] = []
        self.browser_type = "webkit"
        self.playwright = None

        # Keep operation tracking for concurrency safety
        self._active_operations = 0
        self._operation_lock = asyncio.Lock()
        self._browser_lock = asyncio.Lock()

        # Railway environment detection
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None

        # Progressive timeout strategy
        self.default_timeout = 30000  # 30 seconds default
        self.slow_timeout = 60000     # 60 seconds for slow sites
        self.browser_launch_timeout = 30000  # 30 seconds for browser launch

        # Domain-specific timeouts
        self.domain_timeouts = {
            'guide.michelin.com': 60000,
            'timeout.com': 45000,
            'opentable.com': 45000,
            'yelp.com': 40000,
        }

        # Human-like timing
        self.load_wait_time = 3.0      # Human reading time after load
        self.interaction_delay = 0.5   # Delay between actions
        self.retry_delay = 2.0         # Delay between retries

        # Enhanced stats tracking
        self.stats = {
            "total_processed": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "browser_launch_failures": 0,
            "timeout_failures": 0,
            "content_failures": 0,
            "selection_method_stats": {
                "keyboard_success": 0,
                "manual_success": 0, 
                "smart_fallback_success": 0,
                "emergency_fallback": 0
            },
            "total_cost_estimate": 0.0,
            "total_processing_time": 0.0,
            "strategy_breakdown": {"human_mimic": 0},
            "avg_load_time": 0.0,
            "total_load_time": 0.0,
            "concurrent_peak": 0,
            "browser_used": "webkit",
            "memory_optimized": True,
            "session_managed": True,
            "browser_starts": 0,
            "browser_stops": 0,
        }

        logger.info(f"✅ FIXED Smart Restaurant Scraper initialized - Railway mode: {self.is_railway}")

    async def _start_operation(self):
        """Mark the start of an active operation - prevents cleanup"""
        if not hasattr(self, '_active_operations'):
            self._active_operations = 0
        if not hasattr(self, '_operation_lock'):
            self._operation_lock = asyncio.Lock()

        async with self._operation_lock:
            self._active_operations += 1
            self.last_activity = time.time()
            logger.debug(f"🔄 Started operation (active: {self._active_operations})")

    async def _end_operation(self):
        """Mark the end of an active operation"""
        if not hasattr(self, '_active_operations'):
            self._active_operations = 0
        if not hasattr(self, '_operation_lock'):
            self._operation_lock = asyncio.Lock()

        async with self._operation_lock:
            self._active_operations = max(0, self._active_operations - 1)
            self.last_activity = time.time()
            logger.debug(f"✅ Ended operation (active: {self._active_operations})")

    async def _ensure_browser_session(self):
        """SIMPLIFIED: Just ensure browser is running"""
        async with self._browser_lock:
            if not self.browser:
                await self._start_browser_session()

    async def start(self):
        """Start browser once and keep it open"""
        await self._ensure_browser_session()

    async def _start_browser_session(self):
        """Start browser session - keep existing logic"""
        if self.browser:
            return  # Already running

        logger.info("🚀 Starting browser session (will stay open)...")
        start_time = time.time()

        try:
            self.playwright = await asyncio.wait_for(
                async_playwright().start(),
                timeout=10.0
            )

            # Your existing browser launch logic
            if self.is_railway:
                logger.info("🚂 Railway environment detected")
                await self._launch_railway_optimized_browser()
            else:
                logger.info("💻 Local environment")
                await self._launch_with_fallbacks()

            # Create contexts
            await asyncio.wait_for(
                self._create_optimized_context(), 
                timeout=15.0
            )

            self.stats["browser_starts"] += 1
            startup_time = time.time() - start_time
            logger.info(f"✅ Browser session ready and will stay open ({self.browser_type}, {startup_time:.1f}s)")

        except Exception as e:
            logger.error(f"❌ Failed to start browser: {e}")
            self.stats["browser_launch_failures"] += 1
            raise

    async def _launch_railway_optimized_browser(self):
        """FIXED: Railway-specific browser launch with better resource management"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")

        try:
            # Try Chromium first with FIXED Railway-optimized settings
            logger.info("🎭 Launching FIXED Railway-optimized Chromium...")
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--disable-background-networking',
                    '--disable-default-apps',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-sync',
                    '--disable-translate',
                    '--disable-web-security',
                    '--disable-features=TranslateUI,BlinkGenPropertyTrees',
                    '--memory-pressure-off',
                    '--max_old_space_size=300',  # FIXED: Increased from 200MB to 300MB
                    '--single-process',
                    '--no-zygote',
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-audio-output',
                    '--mute-audio',
                    '--blink-settings=imagesEnabled=false',
                    '--disable-images',
                    '--disable-javascript-harmony-shipping',
                    '--disable-features=VizDisplayCompositor,VizHitTestSurfaceLayer',
                    '--run-all-compositor-stages-before-draw',
                    '--disable-threaded-compositing',
                    '--disable-checker-imaging',
                    # FIXED: Added stability flags
                    '--disable-crash-reporter',
                    '--no-crash-upload',
                    '--disable-logging',
                ]
            )
            self.browser_type = "chromium_railway"
            self.stats["browser_used"] = "chromium_railway"
            logger.info("✅ FIXED Railway-optimized Chromium launched")

        except Exception as chromium_error:
            logger.warning(f"⚠️ Railway Chromium failed: {chromium_error}")
            # Fallback to Firefox for Railway
            try:
                logger.info("🦊 Falling back to FIXED Railway-optimized Firefox...")
                self.browser = await self.playwright.firefox.launch(
                    headless=True,
                    firefox_user_prefs={
                        'permissions.default.image': 2,
                        'media.autoplay.enabled': False,
                        'browser.cache.disk.enable': False,
                        'browser.cache.memory.enable': True,
                        'network.dns.disableIPv6': True,
                        'privacy.trackingprotection.enabled': True,
                        'dom.webnotifications.enabled': False,
                        'geo.enabled': False,
                    },
                    args=[
                        '--memory-pressure-off',
                        '--no-remote',
                        '--safe-mode'
                    ]
                )
                self.browser_type = "firefox_railway"
                self.stats["browser_used"] = "firefox_railway"
                logger.info("✅ FIXED Railway-optimized Firefox launched")
            except Exception as firefox_error:
                logger.error(f"❌ All Railway browsers failed. Chromium: {chromium_error}, Firefox: {firefox_error}")
                raise Exception(f"Browser launch failed on Railway: {firefox_error}")

    async def _launch_with_fallbacks(self):
        """Enhanced fallback sequence for local development"""
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

    async def _launch_webkit(self):
        """WebKit launch with ONLY WebKit-compatible arguments"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
        self.browser = await self.playwright.webkit.launch(headless=True)
        self.browser_type = "webkit"

    async def _launch_firefox(self):
        """Enhanced Firefox launch optimized for text extraction"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
        self.browser = await self.playwright.firefox.launch(
            headless=True,
            firefox_user_prefs={
                'permissions.default.image': 2,
                'media.autoplay.enabled': False,
                'browser.cache.disk.enable': False,
                'browser.cache.memory.enable': True,
                'network.dns.disableIPv6': True,
                'privacy.trackingprotection.enabled': True,
                'dom.webnotifications.enabled': False,
                'geo.enabled': False,
            },
            args=['--memory-pressure-off', '--no-remote']
        )
        self.browser_type = "firefox"

    async def _launch_chromium_minimal(self):
        """Enhanced Chromium with aggressive text-only optimizations"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
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
                '--disable-sync',
                '--disable-translate',
                '--disable-web-security',
                '--memory-pressure-off',
                '--single-process',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--disable-audio-output',
                '--mute-audio',
                '--blink-settings=imagesEnabled=false',
                '--disable-images',
            ]
        )
        self.browser_type = "chromium_minimal"

    async def _create_optimized_context(self):
        """FIXED: Create browser contexts with better lifecycle management"""
        if not self.browser:
            raise RuntimeError("Browser not initialized")

        # FIXED: Clear old contexts first
        for context in self.context:
            try:
                await context.close()
            except Exception as e:
                logger.debug(f"Context cleanup error (non-critical): {e}")
        self.context.clear()

        # Create fresh contexts
        for i in range(self.max_concurrent):
            try:
                context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    ignore_https_errors=True,
                    java_script_enabled=True,
                    bypass_csp=True
                )
                self.context.append(context)
                logger.debug(f"✅ Context {i+1} created")
            except Exception as e:
                logger.error(f"❌ Failed to create context {i+1}: {e}")
                raise
                
    async def _configure_page_with_adblock(self, page: Page):
        """
        ENHANCED: Optimized resource blocking that preserves content extraction
        Key change: Allow CSS for proper content visibility detection
        """
        try:
            # 1. Block media files (keep this for memory savings)
            await page.route("**/*.{png,jpg,jpeg,gif,svg,webp,ico,bmp,tiff,avif,heic,heif}", lambda route: route.abort())
            await page.route("**/*.{mp4,avi,mov,wmv,flv,webm,m4v,mp3,wav,aac,ogg,flac}", lambda route: route.abort())

            # 2. Block fonts (keep this for memory savings)
            await page.route("**/*.{woff,woff2,ttf,otf,eot}", lambda route: route.abort())

            # 3. IMPORTANT: Don't block CSS completely as it affects content visibility detection
            # REMOVED: await page.route("**/*.{css}", lambda route: route.abort())

            # 4. Block major ad and tracking domains (enhanced list)
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

            # 5. Block by resource type (but allow stylesheet for content detection)
            await page.route("**/*", lambda route: (
                route.abort() if route.request.resource_type in [
                    'image', 'media', 'font', 'manifest'  # Removed 'stylesheet' and 'other'
                ] else route.continue_()
            ))

            # 6. SET HEADERS FOR ENHANCED PERFORMANCE
            await page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Cache-Control': 'no-cache',
                'DNT': '1'
            })

            # 7. Enhanced script optimizations (but keep CSS functionality)
            await page.add_init_script("""
                // Disable image loading but keep CSS
                Object.defineProperty(HTMLImageElement.prototype, 'src', {
                    set: function() { /* blocked */ },
                    get: function() { return ''; }
                });

                // Speed up animations
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

                // Disable notifications and popups
                window.Notification = undefined;
                window.alert = () => {};
                window.confirm = () => true;
                window.prompt = () => '';
            """)

        except Exception as e:
            logger.warning(f"⚠️ Page configuration partially failed: {e}")

    async def _dismiss_overlays(self, page: Page):
        """Dismiss common overlays and popups"""
        overlay_selectors = [
            '[data-testid="cookie-banner"] button',
            '.cookie-consent button',
            '.gdpr-banner button', 
            '[class*="cookie"] button[class*="accept"]',
            '[class*="modal"] button[class*="close"]',
            '.modal-close',
            '[aria-label*="Close"]',
            '[aria-label*="close"]'
        ]

        for selector in overlay_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await asyncio.sleep(0.3)
                    logger.debug(f"📴 Dismissed overlay: {selector}")
                    break
            except Exception:
                # Overlay element not clickable or doesn't exist, try next selector
                continue

    async def _light_scroll(self, page: Page):
        """Light scrolling to trigger lazy-loaded content"""
        try:
            # Get page height
            page_height = await page.evaluate('document.body.scrollHeight')
            viewport_height = await page.evaluate('window.innerHeight')

            if page_height <= viewport_height:
                return  # No need to scroll

            # Light scrolling
            scroll_steps = min(3, max(1, page_height // viewport_height))
            scroll_step = page_height // scroll_steps

            for i in range(scroll_steps):
                scroll_to = scroll_step * (i + 1)
                await page.evaluate(f'window.scrollTo(0, {scroll_to})')
                await asyncio.sleep(0.2)

            # Scroll back to top
            await page.evaluate('window.scrollTo(0, 0)')
            await asyncio.sleep(0.2)

            logger.debug(f"🖱️ Completed scrolling ({scroll_steps} steps)")

        except Exception as e:
            logger.debug(f"Scrolling error (non-critical): {e}")

    async def _scrape_single_url(self, url: str, context_index: int = 0) -> Dict[str, Any]:
        """FIXED: Single URL scraping with operation tracking"""
        await self._start_operation()  # FIXED: Track active operation

        try:
            await self._ensure_browser_session()  # Ensure session is active

            domain = urlparse(url).netloc
            start_time = time.time()
            page: Optional[Page] = None

            try:
                logger.info(f"🎭 FIXED scraping: {url} (timeout: {self.default_timeout/1000}s, browser: {self.browser_type})")

                # FIXED: Better context validation
                if not self.context:
                    raise Exception("No browser context available")

                context = self.context[context_index % len(self.context)]

                # FIXED: Validate context before creating page
                try:
                    # Test if context is still valid
                    context.pages
                except Exception as context_error:
                    logger.warning(f"⚠️ Context invalid, recreating: {context_error}")
                    await self._create_optimized_context()
                    context = self.context[context_index % len(self.context)]

                page = await context.new_page()

                # Configure page with timeout
                await asyncio.wait_for(
                    self._configure_page_with_adblock(page), 
                    timeout=10.0
                )

                # Navigate with domain-specific timeout
                timeout = self.domain_timeouts.get(domain, self.default_timeout)
                logger.debug(f"🌐 Navigating to {url} with {timeout/1000}s timeout")

                await page.goto(
                    url, 
                    wait_until='domcontentloaded',
                    timeout=timeout
                )

                # Human-like behavior
                await asyncio.sleep(self.load_wait_time)
                await self._dismiss_overlays(page)
                await self._light_scroll(page)

                # FIXED: Better content extraction with validation
                content = await self._extract_content_enhanced_multiStrategy(page)


                if not content or len(content.strip()) < 100:
                    self.stats["content_failures"] += 1
                    logger.warning(f"⚠️ Insufficient content from {url} ({len(content)} chars)")
                    return {
                        "success": False,
                        "url": url,
                        "error": "Insufficient content extracted",
                        "content_length": len(content) if content else 0,
                        "browser_used": self.browser_type,
                        "strategy": "human_mimic"
                    }

                # Get page title
                try:
                    title = await page.title()
                except:
                    title = ""

                processing_time = time.time() - start_time
                self.stats["total_load_time"] += processing_time
                self.stats["successful_scrapes"] += 1
                self.stats["total_processed"] += 1

                logger.info(f"✅ FIXED scraping success: {url} ({len(content)} chars, {processing_time:.2f}s)")

                return {
                    "success": True,
                    "url": url,
                    "title": title,
                    "content": content,
                    "processing_time": processing_time,
                    "content_length": len(content),
                    "browser_used": self.browser_type,
                    "strategy": "human_mimic"
                }

            except Exception as e:
                processing_time = time.time() - start_time
                self.stats["failed_scrapes"] += 1
                self.stats["total_processed"] += 1

                error_type = type(e).__name__
                logger.error(f"❌ FIXED scraping failed: {url}")
                logger.error(f"   Error: {e}")
                logger.error(f"   Type: {error_type}")
                logger.error(f"   Time: {processing_time:.2f}s")
                logger.error(f"   Browser: {self.browser_type}")

                return {
                    "success": False,
                    "url": url,
                    "error": str(e),
                    "error_type": error_type,
                    "processing_time": processing_time,
                    "browser_used": self.browser_type,
                    "strategy": "human_mimic"
                }

            finally:
                # FIXED: Proper page cleanup
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass  # Page might already be closed

        finally:
            await self._end_operation()  # FIXED: Always end operation tracking


    async def _extract_content_enhanced_multiStrategy(self, page: Page) -> str:
        """
        ENHANCED: Multi-strategy content extraction with comprehensive logging
        Combines the best of both implementations with enhanced fallback
        """
        page_url = page.url
        strategy_start_time = time.time()

        logger.info("🎯 STARTING MULTI-STRATEGY CONTENT EXTRACTION")
        logger.info(f"🎯 Target URL: {page_url}")
        logger.info(f"🎯 Browser: {self.browser_type}")

        try:
            # Strategy 1: Enhanced keyboard selection with detailed debugging
            logger.info("🎹 === STRATEGY 1: KEYBOARD SELECTION ===")
            strategy1_start = time.time()
            content = await self._try_keyboard_selection(page)
            strategy1_time = time.time() - strategy1_start

            if content and len(content.strip()) > 100:
                self.stats["selection_method_stats"]["keyboard_success"] += 1
                total_time = time.time() - strategy_start_time
                logger.info(f"✅ STRATEGY 1 SUCCESS: Keyboard selection in {strategy1_time:.2f}s")
                logger.info(f"✅ Total extraction time: {total_time:.2f}s")
                logger.info(f"✅ Final content length: {len(content)} characters")
                return self._clean_scraped_text(content)
            else:
                logger.warning(f"❌ STRATEGY 1 FAILED: Keyboard selection insufficient ({len(content) if content else 0} chars) in {strategy1_time:.2f}s")

            # Strategy 2: Manual range selection on main content
            logger.info("🔧 === STRATEGY 2: MANUAL RANGE SELECTION ===")
            strategy2_start = time.time()
            content = await self._try_manual_selection(page)
            strategy2_time = time.time() - strategy2_start

            if content and len(content.strip()) > 100:
                self.stats["selection_method_stats"]["manual_success"] += 1
                total_time = time.time() - strategy_start_time
                logger.info(f"✅ STRATEGY 2 SUCCESS: Manual selection in {strategy2_time:.2f}s")
                logger.info(f"✅ Total extraction time: {total_time:.2f}s")
                logger.info(f"✅ Final content length: {len(content)} characters")
                return self._clean_scraped_text(content)
            else:
                logger.warning(f"❌ STRATEGY 2 FAILED: Manual selection insufficient ({len(content) if content else 0} chars) in {strategy2_time:.2f}s")

            # Strategy 3: Smart content area extraction
            logger.info("🆘 === STRATEGY 3: SMART CONTENT EXTRACTION ===")
            strategy3_start = time.time()
            content = await self._extract_content_smart_fallback(page)
            strategy3_time = time.time() - strategy3_start

            if content and len(content.strip()) > 100:
                self.stats["selection_method_stats"]["smart_fallback_success"] += 1
                total_time = time.time() - strategy_start_time
                logger.info(f"✅ STRATEGY 3 SUCCESS: Smart fallback in {strategy3_time:.2f}s")
                logger.info(f"✅ Total extraction time: {total_time:.2f}s")  
                logger.info(f"✅ Final content length: {len(content)} characters")
                return self._clean_scraped_text(content)
            else:
                logger.warning(f"❌ STRATEGY 3 FAILED: Smart fallback insufficient ({len(content) if content else 0} chars) in {strategy3_time:.2f}s")

            # Strategy 4: Emergency fallback with strict limits
            logger.info("🚨 === STRATEGY 4: EMERGENCY EXTRACTION ===")
            strategy4_start = time.time()
            content = await self._emergency_content_extraction(page)
            strategy4_time = time.time() - strategy4_start

            if content:
                self.stats["selection_method_stats"]["emergency_fallback"] += 1
                total_time = time.time() - strategy_start_time
                logger.warning(f"⚠️ STRATEGY 4 SUCCESS: Emergency fallback used in {strategy4_time:.2f}s")
                logger.warning(f"⚠️ Total extraction time: {total_time:.2f}s")
                logger.warning(f"⚠️ Final content length: {len(content)} characters")
                return self._clean_scraped_text(content)
            else:
                logger.error(f"❌ STRATEGY 4 FAILED: Emergency extraction returned no content in {strategy4_time:.2f}s")

            # Complete failure
            total_time = time.time() - strategy_start_time
            logger.error(f"❌ ALL STRATEGIES FAILED for {page_url}")
            logger.error(f"❌ Total time spent: {total_time:.2f}s")
            logger.error(f"❌ Browser: {self.browser_type}")
            return ""

        except Exception as e:
            total_time = time.time() - strategy_start_time
            logger.error(f"❌ MULTI-STRATEGY EXTRACTION CRITICAL ERROR: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Time elapsed: {total_time:.2f}s")
            logger.error(f"❌ URL: {page_url}")
            return ""

    async def _try_keyboard_selection(self, page: Page) -> str:
        """
        Enhanced keyboard selection with detailed debugging and comprehensive logging
        """
        try:
            # Wait for page to be fully ready for selection
            await asyncio.sleep(self.interaction_delay)

            # Enhanced browser and page info for debugging
            browser_info = await page.evaluate("navigator.userAgent")
            page_url = page.url
            page_title = await page.title() or "No title"

            logger.info(f"🔤 KEYBOARD SELECTION ATTEMPT - URL: {page_url[:80]}...")
            logger.info(f"🔤 Page title: {page_title[:60]}...")
            logger.info(f"🔤 Browser: {browser_info[:100]}...")

            # Check page readiness
            try:
                page_ready_info = await page.evaluate("""
                    () => {
                        return {
                            readyState: document.readyState,
                            bodyExists: !!document.body,
                            bodyChildCount: document.body ? document.body.children.length : 0,
                            totalTextLength: document.body ? document.body.textContent.length : 0,
                            hasSelection: !!window.getSelection
                        };
                    }
                """)
                logger.info(f"🔤 Page readiness: {page_ready_info}")
            except Exception as e:
                logger.warning(f"🔤 Could not check page readiness: {e}")

            # Try different keyboard selection strategies with enhanced logging
            selection_strategies = [
                ("Meta+a", "Mac/WebKit-style (Cmd+A)"),
                ("Control+a", "Windows/Linux-style (Ctrl+A)"),
            ]

            for strategy_idx, (key_combination, description) in enumerate(selection_strategies, 1):
                try:
                    logger.info(f"🔤 STRATEGY {strategy_idx}: Attempting {description}: {key_combination}")

                    # Clear any existing selection first
                    await page.evaluate("window.getSelection().removeAllRanges()")
                    await asyncio.sleep(0.1)

                    # Enhanced focus attempt with logging
                    focus_success = await page.evaluate("""
                        () => {
                            try {
                                document.body.focus();
                                return {
                                    success: true,
                                    activeElement: document.activeElement ? document.activeElement.tagName : 'none',
                                    bodyFocusable: document.body.tabIndex !== undefined
                                };
                            } catch (e) {
                                return { success: false, error: e.message };
                            }
                        }
                    """)
                    logger.debug(f"🔤 Focus attempt: {focus_success}")
                    await asyncio.sleep(0.1)

                    # Perform keyboard selection with timing
                    selection_start = time.time()
                    await page.keyboard.press(key_combination)
                    await asyncio.sleep(self.interaction_delay)
                    selection_time = time.time() - selection_start

                    # Enhanced selection analysis
                    selection_info = await page.evaluate("""
                        () => {
                            const selection = window.getSelection();
                            const selectedText = selection.toString();

                            return {
                                rangeCount: selection.rangeCount,
                                selectedLength: selectedText.length,
                                selectedPreview: selectedText.substring(0, 200) + (selectedText.length > 200 ? '...' : ''),
                                anchorNode: selection.anchorNode ? selection.anchorNode.nodeName : 'none',
                                focusNode: selection.focusNode ? selection.focusNode.nodeName : 'none',
                                isCollapsed: selection.isCollapsed,
                                rangeStartContainer: selection.rangeCount > 0 ? 
                                    selection.getRangeAt(0).startContainer.nodeName : 'none',
                                rangeEndContainer: selection.rangeCount > 0 ? 
                                    selection.getRangeAt(0).endContainer.nodeName : 'none'
                            };
                        }
                    """)

                    logger.info(f"🔤 STRATEGY {strategy_idx} RESULTS:")
                    logger.info(f"     Selection time: {selection_time:.3f}s")
                    logger.info(f"     Range count: {selection_info['rangeCount']}")
                    logger.info(f"     Selected length: {selection_info['selectedLength']} chars")
                    logger.info(f"     Is collapsed: {selection_info['isCollapsed']}")
                    logger.info(f"     Anchor node: {selection_info['anchorNode']}")
                    logger.info(f"     Focus node: {selection_info['focusNode']}")

                    if selection_info['selectedLength'] > 0:
                        logger.info(f"     Content preview: {selection_info['selectedPreview'][:100]}...")

                    # Extract the selected content with detailed logging
                    content_extraction_start = time.time()
                    content = await page.evaluate("""
                        () => {
                            const selection = window.getSelection();
                            if (selection.rangeCount > 0 && selection.toString().length > 0) {
                                const range = selection.getRangeAt(0);
                                const div = document.createElement('div');
                                div.appendChild(range.cloneContents());

                                // Count elements before cleanup
                                const elementsBeforeCleanup = div.querySelectorAll('*').length;

                                // Remove non-content elements but keep the content structure
                                const elementsToRemove = div.querySelectorAll(
                                    'script, style, noscript'
                                );
                                elementsToRemove.forEach(el => el.remove());

                                const elementsAfterCleanup = div.querySelectorAll('*').length;
                                const text = div.textContent || div.innerText || '';

                                return {
                                    content: text,
                                    elementsBeforeCleanup,
                                    elementsAfterCleanup,
                                    elementsRemoved: elementsBeforeCleanup - elementsAfterCleanup
                                };
                            }
                            return { content: null, elementsBeforeCleanup: 0, elementsAfterCleanup: 0, elementsRemoved: 0 };
                        }
                    """)
                    content_extraction_time = time.time() - content_extraction_start

                    logger.info(f"🔤 Content extraction completed in {content_extraction_time:.3f}s")
                    logger.info(f"     Elements before cleanup: {content['elementsBeforeCleanup']}")
                    logger.info(f"     Elements removed: {content['elementsRemoved']}")
                    logger.info(f"     Final content length: {len(content['content']) if content['content'] else 0} chars")

                    # Validate that we got actual content
                    if content['content'] and len(content['content'].strip()) > 100:
                        logger.info(f"✅ STRATEGY {strategy_idx} SUCCESS: {description}")
                        logger.info(f"     Final content: {len(content['content'])} characters extracted")
                        logger.info(f"     Content starts with: {content['content'][:150]}...")
                        return content['content']
                    else:
                        content_length = len(content['content']) if content['content'] else 0
                        logger.warning(f"❌ STRATEGY {strategy_idx} FAILED: {description}")
                        logger.warning(f"     Insufficient content: {content_length} chars (need >100)")
                        if content['content']:
                            logger.warning(f"     What we got: '{content['content'][:100]}'")

                except Exception as e:
                    logger.error(f"❌ STRATEGY {strategy_idx} EXCEPTION: {description}")
                    logger.error(f"     Error: {e}")
                    logger.error(f"     Error type: {type(e).__name__}")
                    continue

            # If we get here, all strategies failed
            logger.error(f"❌ ALL KEYBOARD STRATEGIES FAILED for {page_url}")
            logger.error(f"     Tried {len(selection_strategies)} different key combinations")
            logger.error(f"     Browser: {self.browser_type}")
            return ""

        except Exception as e:
            logger.error(f"❌ KEYBOARD SELECTION CRITICAL ERROR: {e}")
            logger.error(f"     Error type: {type(e).__name__}")
            logger.error(f"     Page URL: {page.url if page else 'unknown'}")
            return ""

    async def _try_manual_selection(self, page: Page) -> str:
        """
        Manual selection approach when keyboard selection fails
        Simulates what a human would manually select
        """
        try:
            content = await page.evaluate("""
                () => {
                    // Create a range that selects the main visible content
                    const range = document.createRange();

                    // Try to find the main content container
                    const mainElements = [
                        document.querySelector('main'),
                        document.querySelector('article'),
                        document.querySelector('[role="main"]'),
                        document.querySelector('.content'),
                        document.querySelector('#content'),
                        document.body
                    ].filter(el => el !== null);

                    if (mainElements.length === 0) return null;

                    const mainElement = mainElements[0];

                    // Select all content within the main element
                    range.selectNodeContents(mainElement);

                    // Apply the selection
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);

                    // Extract the selected content
                    const div = document.createElement('div');
                    div.appendChild(range.cloneContents());

                    // Remove non-content elements
                    const elementsToRemove = div.querySelectorAll(
                        'script, style, noscript, nav, footer, header, ' +
                        '.navigation, .sidebar, .cookie, .popup, .modal'
                    );
                    elementsToRemove.forEach(el => el.remove());

                    return div.textContent || div.innerText || '';
                }
            """)

            return content if content and len(content.strip()) > 100 else ""

        except Exception as e:
            logger.debug(f"Manual selection error: {e}")
            return ""

    async def _extract_content_smart_fallback(self, page: Page) -> str:
        """
        SMART FALLBACK: Extract main content areas instead of entire page
        """
        try:
            content = await page.evaluate("""
                () => {
                    // Strategy 1: Look for main content containers
                    const mainSelectors = [
                        'main',
                        'article', 
                        '[role="main"]',
                        '.main-content',
                        '.content',
                        '.article-content',
                        '.post-content',
                        '#content',
                        '#main'
                    ];

                    for (const selector of mainSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            // Clone to avoid modifying original page
                            const clone = element.cloneNode(true);
                            const cloneElementsToRemove = clone.querySelectorAll(
                                'nav, footer, header, aside, .sidebar, .navigation, ' +
                                '.cookie, .popup, .modal, .advertisement, ' +
                                '[role="navigation"], [role="banner"], [role="contentinfo"], ' +
                                'script, style, noscript'
                            );
                            cloneElementsToRemove.forEach(el => el.remove());

                            const text = clone.textContent || clone.innerText || '';

                            // Validate: should be substantial but not the entire page
                            if (text.length > 500 && text.length < 20000) {
                                return text;
                            }
                        }
                    }

                    // Strategy 2: Get content paragraphs (common for articles/restaurants)
                    const paragraphs = Array.from(document.querySelectorAll('p'))
                        .map(p => p.textContent || p.innerText || '')
                        .filter(text => text.length > 50) // Filter out short paragraphs
                        .join('\\n\\n');

                    if (paragraphs.length > 500 && paragraphs.length < 15000) {
                        return paragraphs;
                    }

                    return null; // Let emergency method handle the rest
                }
            """)

            return content if content else ""

        except Exception as e:
            logger.debug(f"Smart fallback error: {e}")
            return ""

    async def _emergency_content_extraction(self, page: Page) -> str:
        """
        Emergency fallback with strict content limits
        """
        try:
            # Last resort: get text from body but with strict limits
            emergency_content = await page.evaluate("""
                () => {
                    const bodyText = document.body.textContent || document.body.innerText || '';

                    // If body text is suspiciously long (> 30k), it's probably the full page
                    // Return first 10k characters to avoid processing the entire page
                    if (bodyText.length > 30000) {
                        console.warn('Body text too long, truncating to avoid full page content');
                        return bodyText.substring(0, 10000) + '\\n\\n[Content truncated - emergency extraction mode]';
                    }

                    return bodyText;
                }
            """)

            return emergency_content if emergency_content else ""

        except Exception as e:
            logger.debug(f"Emergency extraction error: {e}")
            return ""

    def _clean_scraped_text(self, text: str) -> str:
        """
        Enhanced text cleaning for restaurant content extraction
        """
        if not text:
            return ""

        # Remove excessive whitespace while preserving structure
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        cleaned = re.sub(r' {2,}', ' ', cleaned)  # Multiple spaces -> single space
        cleaned = cleaned.strip()

        # Remove common navigation/footer noise (enhanced patterns)
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
            r'Back to top.*?(?=\n|$)',
        ]

        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Remove isolated navigation words
        nav_words = [
            'Menu', 'Home', 'About', 'Contact', 'Login', 'Sign up', 'Register',
            'Search', 'Categories', 'Tags', 'Archive', 'Subscribe'
        ]

        for word in nav_words:
            cleaned = re.sub(f'\n{word}\n', '\n', cleaned)
            cleaned = re.sub(f'^{word}\n', '', cleaned)
            cleaned = re.sub(f'\n{word}$', '', cleaned)

        # Restaurant content optimization
        # Preserve important structural elements
        cleaned = re.sub(r'(\$\d+)', r' \1 ', cleaned)  # Space around prices
        cleaned = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', cleaned)  # Fix time ranges

        # Final cleanup
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    async def stop(self):
        """ONLY place where browser gets closed"""
        logger.info("🛑 Stopping scraper and closing browser...")

        # Wait for any active operations
        max_wait = 30
        wait_start = time.time()

        while self._active_operations > 0 and (time.time() - wait_start) < max_wait:
            logger.info(f"⏳ Waiting for {self._active_operations} operations to complete...")
            await asyncio.sleep(1)

        # Close browser
        await self._stop_browser_session()
        logger.info("🛑 Browser closed and scraper stopped")

    async def _stop_browser_session(self):
        """Close browser and free resources"""
        if not self.browser:
            return

        try:
            # Close contexts
            for context in self.context:
                try:
                    await context.close()
                except Exception:
                    pass
            self.context.clear()

            # Close browser
            if self.browser:
                await self.browser.close()
                self.browser = None

            # Stop Playwright
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None

            self.stats["browser_stops"] += 1
            logger.info(f"🎭 Browser session closed ({self.browser_type})")

        except Exception as e:
            logger.error(f"Error stopping browser: {e}")

    async def force_cleanup(self):
        """Force immediate cleanup (for testing/debugging)"""
        await self._stop_browser_session()
        
    def _log_enhanced_stats(self):
        """Enhanced stats logging with detailed breakdown"""
        if self.stats["total_processed"] > 0:
            logger.info("📊 ENHANCED SMART SCRAPER STATISTICS:")
            logger.info(f"   🌐 Browser: {self.browser_type}")
            logger.info(f"   ✅ Success: {self.stats['successful_scrapes']}/{self.stats['total_processed']} ({(self.stats['successful_scrapes']/self.stats['total_processed']*100):.1f}%)")
            logger.info(f"   ❌ Failures: {self.stats['failed_scrapes']} ({self.stats['timeout_failures']} timeouts, {self.stats['content_failures']} content)")
            logger.info(f"   🚀 Browser Sessions: {self.stats['browser_starts']} starts, {self.stats['browser_stops']} stops")
            logger.info("   🎯 Content Extraction Methods:")
            for method, count in self.stats["selection_method_stats"].items():
                if count > 0:
                    logger.info(f"      {method}: {count}")
            logger.info(f"   💰 Cost Estimate: {self.stats['total_cost_estimate']:.1f} credits")
            logger.info(f"   ⚡ Avg Load Time: {(self.stats['total_load_time']/max(self.stats['total_processed'], 1)):.2f}s")
            logger.info(f"   🧠 Session Managed: {self.stats['session_managed']}")

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FIXED: Process search results with proper URL validation
        """
        if not search_results:
            return []

        logger.info(f"🎯 Starting batch of {len(search_results)} URLs - browser will stay open")

        # Ensure browser is ready ONCE
        await self._ensure_browser_session()

        # FIXED: Filter out results without valid URLs
        valid_urls = []
        valid_results = []

        for result in search_results:
            url = result.get("url")
            if url and isinstance(url, str) and url.strip():  # FIXED: Proper URL validation
                valid_urls.append(url.strip())
                valid_results.append(result)
            else:
                logger.warning(f"⚠️ Skipping result with invalid URL: {url}")

        if not valid_urls:
            logger.warning("⚠️ No valid URLs found in search results")
            return search_results  # Return original results with no scraping data

        logger.info(f"📊 Processing {len(valid_urls)} valid URLs out of {len(search_results)} total")

        # Concurrent scraping with persistent browser
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._scrape_url_with_semaphore(semaphore, url, i % len(self.context))
            for i, url in enumerate(valid_urls)
        ]

        # Wait for all scraping to complete
        scraping_results = await asyncio.gather(*tasks, return_exceptions=True)

        # FIXED: Process results with proper mapping
        enriched_results = []
        scraping_index = 0

        for original_result in search_results:
            url = original_result.get("url")

            if url and isinstance(url, str) and url.strip():
                # This result had a valid URL, so it was scraped
                scraping_result = scraping_results[scraping_index]
                scraping_index += 1

                # FIXED: Always convert Exception to dict
                if isinstance(scraping_result, Exception):
                    logger.error(f"❌ Exception for URL {url}: {scraping_result}")
                    scraping_result = {
                        "success": False,
                        "url": url,
                        "error": str(scraping_result)
                    }
            else:
                # This result had no valid URL, create a failure result
                scraping_result = {
                    "success": False,
                    "url": url if url else "no_url_provided",
                    "error": "No valid URL provided"
                }

            # FIXED: Now scraping_result is guaranteed to be a dict
            if isinstance(scraping_result, dict):  # Extra safety check
                enriched_result = {**original_result, **scraping_result}
            else:
                # Fallback if something unexpected happens
                enriched_result = {
                    **original_result,
                    "success": False,
                    "error": "Unexpected scraping result type"
                }

            enriched_results.append(enriched_result)

        successful = sum(1 for r in scraping_results if isinstance(r, dict) and r.get('success'))
        logger.info(f"✅ Batch complete: {successful}/{len(valid_urls)} successful - browser staying open")

        return enriched_results

    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, context_index: int) -> Dict[str, Any]:
        """Scrape URL with concurrency control using specific context"""
        async with semaphore:
            return await self._scrape_single_url(url, context_index)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics with enhanced metrics"""
        return {
            **self.stats,
            "success_rate": (self.stats["successful_scrapes"] / max(self.stats["total_processed"], 1)) * 100,
            "concurrent_context": len(self.context),
            "session_active": self.browser is not None,
            "session_timeout_seconds": 0,  # FIXED: No longer using session timeout
            "last_activity": None,  # FIXED: No longer tracking last activity
            "extraction_method_breakdown": self.stats["selection_method_stats"]
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory-specific statistics"""
        return {
            'browser_active': self.browser is not None,
            'browser_type': self.browser_type if self.browser else None,
            'session_timeout': 0,  # FIXED: No session timeout
            'memory_optimized': True,
            'session_managed': False,  # FIXED: Session management disabled
            'resource_blocking_enabled': True,
            'css_blocking_disabled': True,
            'estimated_memory_idle': "~0MB (browser closed on stop)",
            'estimated_memory_active': f"~150-200MB ({self.browser_type})",
            'browser_lifecycle': {
                'starts': self.stats['browser_starts'],
                'stops': self.stats['browser_stops'],
                'current_session_active': self.browser is not None
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup browser"""
        await self.stop()