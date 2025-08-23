# agents/smart_scraper.py
"""
ENHANCED: Smart Scraper with Multi-Strategy Content Extraction and Session Management
- Combines Railway optimizations with robust content extraction
- Keeps browser open with session management for better performance
- Multi-fallback content extraction (keyboard + manual + smart)
- Enhanced debugging and error handling
- Preserves ALL existing functionality
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
    """
    Enhanced Smart Scraper with session management and multi-strategy content extraction
    Strategy: Human Mimic for everything (2.0 credits per URL)
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()
        self.max_concurrent = 1  
        self.browser: Optional[Browser] = None
        self.context: List[BrowserContext] = []
        self.browser_type = "webkit"  # Primary browser choice
        self.playwright = None

        # SESSION MANAGEMENT (from alternative implementation)
        self.session_timeout = 300  # 5 minutes of inactivity before closing
        self.last_activity = None
        self.cleanup_timer = None
        self._cleanup_scheduled = False

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

        # Enhanced stats tracking with failure reasons
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
            # Session management stats
            "browser_starts": 0,
            "browser_stops": 0,
        }

        logger.info(f"✅ Enhanced Smart Restaurant Scraper initialized - Railway mode: {self.is_railway}")

    def _schedule_cleanup(self):
        """Schedule browser cleanup after inactivity timeout"""
        # Cancel existing timer
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
            self._cleanup_scheduled = False

        # Use background thread approach instead of mixing threading.Timer with asyncio
        def cleanup_check():
            """Check if cleanup is needed and schedule it properly"""
            if self.last_activity and (time.time() - self.last_activity) >= self.session_timeout:
                # Set a flag that will be checked by the next async operation
                self._cleanup_scheduled = True
                logger.info(f"🧹 Cleanup scheduled - browser session inactive for {self.session_timeout}s")

        # Schedule cleanup check
        self.cleanup_timer = threading.Timer(self.session_timeout, cleanup_check)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()

    async def _check_and_cleanup_if_needed(self):
        """Check if cleanup is scheduled and execute it in async context"""
        if self._cleanup_scheduled:
            self._cleanup_scheduled = False
            await self._cleanup_inactive_session()

    async def _cleanup_inactive_session(self):
        """Clean up browser session if inactive"""
        if self.last_activity and (time.time() - self.last_activity) >= self.session_timeout:
            logger.info(f"🧹 Closing inactive browser session ({self.session_timeout}s timeout)")
            await self._stop_browser_session()

    async def _ensure_browser_session(self):
        """Ensure browser session is active, start if needed"""
            # Check for scheduled cleanup first
        await self._check_and_cleanup_if_needed()

        if not self.browser:
            await self._start_browser_session()

            # Update activity and reset cleanup timer
        self.last_activity = time.time()
        self._schedule_cleanup()

    async def start(self):
        """Initialize Playwright and browser context (with session management)"""
        await self._ensure_browser_session()

    async def _start_browser_session(self):
        """Start browser session with enhanced optimizations"""
        if self.browser:
            return  # Already running

        logger.info("🚀 Starting enhanced browser session...")
        start_time = time.time()

        try:
            self.playwright = await asyncio.wait_for(
                async_playwright().start(),
                timeout=10.0
            )

            # Enhanced browser launch sequence with Railway detection
            if self.is_railway:
                logger.info("🚂 Railway environment detected - using optimized settings")
                await self._launch_railway_optimized_browser()
            else:
                logger.info("💻 Local environment - using WebKit with fallbacks")
                await self._launch_with_fallbacks()

            # Create optimized context with timeout
            await asyncio.wait_for(
                self._create_optimized_context(), 
                timeout=15.0
            )

            self.stats["browser_starts"] += 1
            startup_time = time.time() - start_time
            logger.info(f"🎭 Enhanced browser session ready ({self.browser_type}, {startup_time:.1f}s)")

        except asyncio.TimeoutError:
            logger.error("❌ Browser startup timeout - this might be a Railway resource issue")
            self.stats["browser_launch_failures"] += 1
            raise Exception("Browser startup timeout")
        except Exception as e:
            logger.error(f"❌ Failed to start browser: {e}")
            self.stats["browser_launch_failures"] += 1
            raise

    async def _launch_railway_optimized_browser(self):
        """Railway-specific browser launch with aggressive optimizations"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")

        try:
            # Try Chromium first with Railway-optimized settings
            logger.info("🎭 Launching Railway-optimized Chromium...")
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
                    '--max_old_space_size=200',  # Very conservative memory limit
                    '--single-process',
                    '--no-zygote',  # Important for containers
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
                ]
            )
            self.browser_type = "chromium_railway"
            self.stats["browser_used"] = "chromium_railway"
            logger.info("✅ Railway-optimized Chromium launched")

        except Exception as chromium_error:
            logger.warning(f"⚠️ Railway Chromium failed: {chromium_error}")
            # Fallback to Firefox for Railway
            try:
                logger.info("🦊 Falling back to Railway-optimized Firefox...")
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
                logger.info("✅ Railway-optimized Firefox launched")
            except Exception as firefox_error:
                logger.error(f"❌ All Railway browsers failed. Chromium: {chromium_error}, Firefox: {firefox_error}")
                raise Exception(f"Browser launch failed on Railway: {firefox_error}")

    async def _launch_with_fallbacks(self):
        """Enhanced fallback sequence for local development"""
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

    async def _launch_webkit(self):
        """WebKit launch with ONLY WebKit-compatible arguments (simplified)"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
        self.browser = await self.playwright.webkit.launch(
            headless=True
            # NO ARGUMENTS - WebKit is very picky about args
        )
        self.browser_type = "webkit"

    async def _launch_firefox(self):
        """Enhanced Firefox launch optimized for text extraction"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
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
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
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
                '--mute-audio',                              # Mute everything
                '--disable-ipc-flooding-protection',        # Faster IPC
                '--disable-domain-reliability',
            ]
        )
        self.browser_type = "chromium_minimal"

    async def _create_optimized_context(self):
        """Create browser context with enhanced optimizations"""
        try:
            for i in range(self.max_concurrent):
                if not self.browser:
                    raise RuntimeError("Browser not initialized")
                context = await self.browser.new_context(
                    viewport={'width': 1024, 'height': 600},  # Smaller viewport
                    user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 TextExtractor/1.0',
                    ignore_https_errors=True,
                    bypass_csp=True,
                    # Enhanced optimizations
                    java_script_enabled=True,  # Keep for dynamic content
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate',
                        'Cache-Control': 'no-cache',
                        'DNT': '1'  # Do Not Track
                    },
                    permissions=[],
                    geolocation=None,
                )

                # Set optimized timeouts
                context.set_default_timeout(self.default_timeout)
                context.set_default_navigation_timeout(self.default_timeout)

                self.context.append(context)

            logger.info(f"✅ {len(self.context)} enhanced contexts created ({self.browser_type})")

        except Exception as e:
            logger.error(f"❌ Failed to create browser context: {e}")
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
        """
        ENHANCED: Single URL scraping with multi-strategy content extraction
        """
        await self._ensure_browser_session()  # Ensure session is active

        domain = urlparse(url).netloc
        start_time = time.time()
        page: Optional[Page] = None

        try:
            logger.info(f"🎭 Enhanced scraping: {url} (timeout: {self.default_timeout/1000}s, browser: {self.browser_type})")

            # Get context with error handling
            if not self.context:
                raise Exception("No browser context available")

            context = self.context[context_index % len(self.context)]
            page = await context.new_page()

            # Configure page with timeout
            await asyncio.wait_for(
                self._configure_page_with_adblock(page), 
                timeout=5.0
            )

            # Smart timeout strategy
            timeout_ms = self.domain_timeouts.get(domain, self.default_timeout)
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)

            # Navigate with retry logic
            load_success = False
            last_error = None

            for attempt in range(2):  # Maximum 2 attempts
                try:
                    logger.debug(f"🌐 Attempt {attempt + 1}: Loading {domain}")

                    # Use domcontentloaded for faster loading
                    await page.goto(url, wait_until='domcontentloaded', timeout=timeout_ms)
                    load_success = True
                    logger.debug(f"✅ Page loaded successfully on attempt {attempt + 1}")
                    break

                except PlaywrightTimeoutError as te:
                    last_error = f"Timeout after {timeout_ms/1000}s"
                    logger.warning(f"⏰ Attempt {attempt + 1} timed out: {te}")
                    if attempt == 0:
                        await asyncio.sleep(1.0)  # Brief pause before retry
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    if attempt == 0:
                        await asyncio.sleep(1.0)

            if not load_success:
                self.stats["timeout_failures"] += 1
                raise Exception(f"Failed to load after 2 attempts: {last_error}")

            # Human-like behavior: wait and read the page
            await asyncio.sleep(self.load_wait_time)

            # Dismiss overlays
            await self._dismiss_overlays(page)

            # Light scrolling to trigger lazy content
            await self._light_scroll(page)

            # ENHANCED: Multi-strategy content extraction with detailed logging
            logger.info("🎯 Starting enhanced multi-strategy content extraction")
            content = await self._extract_content_enhanced_multiStrategy(page)
            logger.info(f"🎯 Content extraction completed: {len(content)} characters")

            if not content or len(content.strip()) < 100:
                self.stats["content_failures"] += 1
                logger.warning(f"⚠️ Insufficient content extracted from {url} ({len(content)} chars)")
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

            logger.info(f"✅ Enhanced scraping success: {url} ({len(content)} chars, {processing_time:.2f}s)")

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

            # Enhanced error logging
            error_type = type(e).__name__
            logger.error(f"❌ Enhanced scraping failed: {url}")
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
            # Critical: Always close page to free memory
            if page:
                try:
                    await page.close()
                    logger.debug(f"🗑️ Page closed for {url}")
                except Exception as e:
                    logger.warning(f"⚠️ Error closing page: {e}")

    async def _extract_content_enhanced_multiStrategy(self, page: Page) -> str:
        """
        ENHANCED: Multi-strategy content extraction with debugging
        Combines the best of both implementations with enhanced fallback
        """
        try:
            logger.info("🎹 Starting enhanced keyboard-based content selection")

            # Strategy 1: Enhanced keyboard selection with detailed debugging
            content = await self._try_keyboard_selection(page)
            if content and len(content.strip()) > 100:
                self.stats["selection_method_stats"]["keyboard_success"] += 1
                logger.info(f"✅ Keyboard selection successful: {len(content)} chars")
                return self._clean_scraped_text(content)

            # Strategy 2: Manual range selection on main content
            logger.warning("🔧 Keyboard selection insufficient, trying manual range selection")
            content = await self._try_manual_selection(page)
            if content and len(content.strip()) > 100:
                self.stats["selection_method_stats"]["manual_success"] += 1
                logger.info(f"✅ Manual selection successful: {len(content)} chars")
                return self._clean_scraped_text(content)

            # Strategy 3: Smart content area extraction
            logger.warning("🆘 Manual selection failed, trying smart content extraction")
            content = await self._extract_content_smart_fallback(page)
            if content and len(content.strip()) > 100:
                self.stats["selection_method_stats"]["smart_fallback_success"] += 1
                logger.info(f"✅ Smart fallback successful: {len(content)} chars")
                return self._clean_scraped_text(content)

            # Strategy 4: Emergency fallback with strict limits
            logger.error("🚨 All primary methods failed, using emergency extraction")
            content = await self._emergency_content_extraction(page)
            if content:
                self.stats["selection_method_stats"]["emergency_fallback"] += 1
                logger.warning(f"⚠️ Emergency fallback used: {len(content)} chars")
                return self._clean_scraped_text(content)

            logger.error("❌ All content extraction strategies failed")
            return ""

        except Exception as e:
            logger.error(f"❌ Multi-strategy content extraction error: {e}")
            return ""

    async def _try_keyboard_selection(self, page: Page) -> str:
        """
        Enhanced keyboard selection with detailed debugging
        """
        try:
            # Wait for page to be fully ready for selection
            await asyncio.sleep(self.interaction_delay)

            # Browser info for debugging
            browser_info = await page.evaluate("navigator.userAgent")
            logger.debug(f"🔍 Browser: {browser_info[:100]}...")

            # Try different keyboard selection strategies with proper debugging
            selection_strategies = [
                ("Meta+a", "Mac/WebKit-style (Cmd+A)"),
                ("Control+a", "Windows/Linux-style (Ctrl+A)"),
            ]

            for key_combination, description in selection_strategies:
                try:
                    logger.debug(f"🔤 Attempting {description}: {key_combination}")

                    # Clear any existing selection first
                    await page.evaluate("window.getSelection().removeAllRanges()")
                    await asyncio.sleep(0.1)

                    # Focus on the body to ensure keyboard events work
                    await page.evaluate("document.body.focus()")
                    await asyncio.sleep(0.1)

                    # Perform keyboard selection
                    await page.keyboard.press(key_combination)
                    await asyncio.sleep(self.interaction_delay)

                    # Debug: Check what was actually selected
                    selection_info = await page.evaluate("""
                        () => {
                            const selection = window.getSelection();
                            return {
                                rangeCount: selection.rangeCount,
                                selectedText: selection.toString().substring(0, 200) + '...',
                                selectedLength: selection.toString().length,
                                anchorNode: selection.anchorNode ? selection.anchorNode.nodeName : 'none',
                                focusNode: selection.focusNode ? selection.focusNode.nodeName : 'none'
                            };
                        }
                    """)

                    logger.debug(f"🔍 Selection debug: {selection_info}")

                    # Extract the selected content
                    content = await page.evaluate("""
                        () => {
                            const selection = window.getSelection();
                            if (selection.rangeCount > 0 && selection.toString().length > 0) {
                                const range = selection.getRangeAt(0);
                                const div = document.createElement('div');
                                div.appendChild(range.cloneContents());

                                // Remove non-content elements but keep the content structure
                                const elementsToRemove = div.querySelectorAll(
                                    'script, style, noscript'
                                );
                                elementsToRemove.forEach(el => el.remove());

                                const text = div.textContent || div.innerText || '';
                                return text;
                            }
                            return null;
                        }
                    """)

                    # Validate that we got actual content
                    if content and len(content.strip()) > 100:
                        logger.debug(f"✅ {description} selection successful: {len(content)} chars")
                        return content
                    else:
                        logger.debug(f"❌ {description} selection returned insufficient content: {len(content) if content else 0} chars")

                except Exception as e:
                    logger.debug(f"❌ {description} selection error: {e}")
                    continue

            return ""

        except Exception as e:
            logger.debug(f"Keyboard selection error: {e}")
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

    async def _stop_browser_session(self):
        """Stop browser session and free memory"""
        if not self.browser:
            return

        try:
            # Close context first
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
            logger.info(f"🎭 Enhanced browser session closed ({self.browser_type})")

        except Exception as e:
            logger.error(f"Error stopping browser session: {e}")

        finally:
            # Cancel cleanup timer
            if self.cleanup_timer:
                self.cleanup_timer.cancel()
                self.cleanup_timer = None
                self._cleanup_scheduled = False

    async def force_cleanup(self):
        """Force immediate cleanup (for testing/debugging)"""
        await self._stop_browser_session()

    async def stop(self):
        """Clean up all resources"""
        logger.info("🛑 Stopping Enhanced Smart Restaurant Scraper...")

        # Cancel cleanup timer
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
            self.cleanup_timer = None

        # Stop browser session
        await self._stop_browser_session()

        # Log final stats
        self._log_enhanced_stats()
        logger.info("🛑 Enhanced Smart Restaurant Scraper stopped")

    def _log_enhanced_stats(self):
        """Enhanced stats logging with detailed breakdown"""
        if self.stats["total_processed"] > 0:
            logger.info("📊 ENHANCED SMART SCRAPER STATISTICS:")
            logger.info(f"   🌐 Browser: {self.browser_type}")
            logger.info(f"   ✅ Success: {self.stats['successful_scrapes']}/{self.stats['total_processed']} ({(self.stats['successful_scrapes']/self.stats['total_processed']*100):.1f}%)")
            logger.info(f"   ❌ Failures: {self.stats['failed_scrapes']} ({self.stats['timeout_failures']} timeouts, {self.stats['content_failures']} content)")
            logger.info(f"   🚀 Browser Sessions: {self.stats['browser_starts']} starts, {self.stats['browser_stops']} stops")
            logger.info(f"   🎯 Content Extraction Methods:")
            for method, count in self.stats["selection_method_stats"].items():
                if count > 0:
                    logger.info(f"      {method}: {count}")
            logger.info(f"   💰 Cost Estimate: {self.stats['total_cost_estimate']:.1f} credits")
            logger.info(f"   ⚡ Avg Load Time: {(self.stats['total_load_time']/max(self.stats['total_processed'], 1)):.2f}s")
            logger.info(f"   🧠 Session Managed: {self.stats['session_managed']}")

    # PRESERVED: Keep all original methods for compatibility
    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point - Process search results with enhanced concurrent scraping
        PRESERVED: Original interface with enhanced functionality
        """
        if not search_results:
            return []

        # Ensure browser session is active
        await self._ensure_browser_session()

        urls = [result.get('url') for result in search_results if result.get('url')]
        logger.info(f"🎭 Enhanced scraping {len(urls)} URLs with {self.max_concurrent} concurrent context ({self.browser_type})")

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
            elif isinstance(scrape_result, dict):
                # Explicit type check: scrape_result is Dict[str, Any]
                enriched.update(scrape_result)
            else:
                # Fallback for unexpected types
                logger.error(f"Unexpected scrape result type: {type(scrape_result)}")
                enriched.update({
                    "success": False,
                    "error": "Unexpected result type",
                    "content": "",
                    "browser_used": self.browser_type,
                    "strategy": "human_mimic"
                })

            self.stats["strategy_breakdown"]["human_mimic"] += 1
            self.stats["total_cost_estimate"] += 2.0  # 2.0 credits per URL
            enriched_results.append(enriched)

        successful = sum(1 for r in scrape_results if isinstance(r, dict) and r.get('success'))
        logger.info(f"✅ Enhanced batch complete: {successful}/{len(urls)} successful ({self.browser_type})")

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
            "session_timeout_seconds": self.session_timeout,
            "last_activity": self.last_activity,
            "extraction_method_breakdown": self.stats["selection_method_stats"]
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory-specific statistics"""
        return {
            'browser_active': self.browser is not None,
            'browser_type': self.browser_type if self.browser else None,
            'session_timeout': self.session_timeout,
            'memory_optimized': True,
            'session_managed': True,
            'resource_blocking_enabled': True,
            'css_blocking_disabled': True,  # New: CSS is allowed for content detection
            'estimated_memory_idle': "~0MB (browser closed after timeout)",
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