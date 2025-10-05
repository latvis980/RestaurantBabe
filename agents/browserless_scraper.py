# agents/browserless_scraper.py
"""
FIXED: Enhanced Restaurant Scraper with Railway Browserless Integration
COMPATIBLE WITH: Railway Browserless v1.41 (Playwright v1.41.0)

KEY FIXES:
- FIXED: Playwright version alignment (1.41.0 client -> 1.41.0 server)
- ENHANCED: Better error handling for version mismatches
- IMPROVED: Railway Browserless connection with token authentication
- MAINTAINS: Full pipeline integration compatibility

INTEGRATION POINTS:
- orchestrator.scraper.scrape_search_results() - PRESERVED
- Text cleaner integration - PRESERVED  
- Supabase storage integration - PRESERVED
- Stats tracking and cost optimization - ENHANCED
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


class BrowserlessRestaurantScraper:
    """
    FIXED: Enhanced Restaurant Scraper with Railway Browserless v1.41 compatibility
    Drop-in replacement for SmartRestaurantScraper with version-aligned Browserless optimizations
    """

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

        # FIXED: Railway environment detection and endpoint configuration
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None

        # ENHANCED: Support both private and public Railway Browserless endpoints
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
            logger.info("🔧 Local Playwright mode (no Railway Browserless endpoint)")

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

        # Enhanced stats tracking
        self.stats = {
            "total_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "avg_scrape_time": 0.0,
            "total_processing_time": 0.0,

            # Smart scraper compatibility
            "specialized_used": 0,
            "simple_http_used": 0,
            "enhanced_http_used": 0,
            "firecrawl_used": 0,
            "total_cost_estimate": 0.0,
            "cost_saved_vs_all_firecrawl": 0.0,
            "strategy_breakdown": {
                "specialized": 0,
                "simple_http": 0,
                "enhanced_http": 0,
                "firecrawl": 0
            },

            # FIXED: Railway Browserless v1.41 compatibility
            "railway_browserless": bool(self.browserless_endpoint),
            "endpoint_used": self.browserless_endpoint or "local_playwright",
            "playwright_version": "1.41.0",  # Track for debugging
            "structure_preserving_success": 0,
            "structure_quality": {
                "headings_preserved": 0,
                "paragraphs_preserved": 0,
                "avg_heading_count": 0.0,
                "avg_paragraph_count": 0.0
            }
        }

        logger.info("🎯 FIXED Enhanced Restaurant Scraper initialized")
        logger.info("✨ NEW: Structure-preserving extraction enabled")
        logger.info(f"🚂 Railway Browserless v1.41: {'✓' if self.browserless_endpoint else '✗'}")

    async def scrape_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """
        MAIN PIPELINE METHOD: Process multiple URLs from search results
        PRESERVES: Full compatibility with LangChainOrchestrator integration
        ENHANCES: Structure-preserving content extraction with v1.41 compatibility
        """
        if not search_results:
            logger.warning("⚠️ No search results to scrape")
            return []

        logger.info(f"🤖 FIXED SCRAPING PIPELINE: Processing {len(search_results)} URLs")
        logger.info(f"🚂 Using Railway Browserless v1.41: {'✓' if self.browserless_endpoint else '✗'}")

        async with self._operation_lock:
            self._active_operations += 1

        try:
            # Process results with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = []

            for result in search_results:
                task = self._scrape_with_semaphore(semaphore, result)
                tasks.append(task)

            # Execute all scraping tasks
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            successful_results = []
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Task {i} failed: {result}")
                    # Create fallback result
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

            success_rate = (successful_count / len(search_results)) * 100 if search_results else 0
            logger.info(f"🎯 FIXED SCRAPING COMPLETE: {successful_count}/{len(search_results)} successful ({success_rate:.1f}%)")

            return successful_results

        finally:
            async with self._operation_lock:
                self._active_operations -= 1

    async def _scrape_with_semaphore(self, semaphore: asyncio.Semaphore, result: Dict) -> Dict:
        """
        ENHANCED: Individual URL scraping with Railway Browserless v1.41 compatibility
        """
        async with semaphore:
            return await self._scrape_single_url(result)

    async def _scrape_single_url(self, result: Dict) -> Dict:
        """
        CORE METHOD: Scrape individual URL with structure-preserving extraction
        FIXED: Railway Browserless v1.41 connection handling
        """
        url = result.get("url", "")
        if not url:
            logger.warning("⚠️ No URL provided")
            return result

        start_time = time.time()
        original_result = result.copy()

        try:
            logger.info(f"🎯 Scraping: {url}")

            # FIXED: Structure-preserving extraction with v1.41 compatibility
            content = await self._extract_content_structure_preserving(url)

            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time

            if content and len(content.strip()) > 100:
                self.stats["successful_scrapes"] += 1
                self.stats["structure_preserving_success"] += 1

                logger.info(f"✅ SCRAPING SUCCESS: {url} in {processing_time:.2f}s ({len(content)} chars)")

                result = original_result.copy()
                result.update({
                    "content": content,
                    "scraping_success": True,
                    "scraping_failed": False,
                    "processing_time": processing_time,
                    "content_length": len(content),
                    "extraction_method": "structure_preserving_v1.41"
                })
                return result
            else:
                logger.warning(f"⚠️ LOW CONTENT: {url} in {processing_time:.2f}s (content too short)")
                result = original_result.copy()
                result.update({
                    "content": "",
                    "scraping_success": False,
                    "scraping_failed": True,
                    "error": "Content too short or empty",
                    "processing_time": processing_time
                })
                return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_scrapes"] += 1

            logger.error(f"❌ SCRAPING FAILED: {url} in {processing_time:.2f}s - {e}")

            result = original_result.copy()
            result.update({
                "scraping_success": False,
                "scraping_failed": True,
                "error": str(e),
                "processing_time": processing_time
            })
            return result

    async def _extract_content_structure_preserving(self, url: str) -> str:
        """
        FIXED: Structure-preserving content extraction using Railway Browserless v1.41
        CORE INNOVATION: Maintains headings and paragraphs with version compatibility
        """
        playwright = None
        browser = None
        page = None

        try:
            # Get domain-specific timeout
            domain = urlparse(url).netloc.lower()
            timeout = self.domain_timeouts.get(domain, self.default_timeout)

            logger.info(f"🔍 Structure-preserving extraction v1.41: {url}")
            logger.info(f"⏱️ Timeout: {timeout}ms for domain: {domain}")

            # Initialize Playwright
            playwright = await async_playwright().start()

            # FIXED: Browser launch configuration for v1.41 compatibility
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
                    "--disable-ipc-flooding-protection"
                ]
            }

            # FIXED: Use Railway Browserless endpoint with proper token handling
            if self.browserless_endpoint:
                try:
                    # ENHANCED: Add token to URL if available
                    connect_url = self.browserless_endpoint
                    if self.browserless_token and '?token=' not in connect_url:
                        separator = '&' if '?' in connect_url else '?'
                        connect_url = f"{connect_url}{separator}token={self.browserless_token}"

                    browser = await playwright.chromium.connect(
                        connect_url,
                        timeout=self.browser_launch_timeout
                    )
                    logger.info("🚂 Connected to Railway Browserless v1.41")
                except Exception as browserless_error:
                    logger.warning(f"⚠️ Railway Browserless connection failed: {browserless_error}")
                    logger.info("🔄 Falling back to local Playwright...")
                    browser = await playwright.webkit.launch(**browser_options)
            else:
                browser = await playwright.webkit.launch(**browser_options)
                logger.info("🔧 Launched local Playwright browser")

            # Create page with Railway Browserless configuration
            page = await browser.new_page()

            # Configure page with Railway Browserless optimizations
            await self._configure_page_railway_optimized(page)

            # Navigate to page
            logger.info(f"🌐 Navigating to: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

            # Human-like wait time
            await asyncio.sleep(self.load_wait_time)

            # CORE ENHANCEMENT: Structure-preserving content extraction
            content = await self._extract_structured_content(page)

            if content and len(content.strip()) > 100:
                # Clean and optimize content
                content = self._clean_restaurant_content(content)
                logger.info(f"✅ Structure-preserving extraction successful: {len(content)} chars")
                return content
            else:
                logger.warning(f"⚠️ Structure extraction yielded insufficient content")
                return ""

        except Exception as e:
            # ENHANCED: Better error handling for version mismatches
            error_msg = str(e)
            if "version mismatch" in error_msg.lower() or "428" in error_msg:
                logger.error(f"❌ PLAYWRIGHT VERSION MISMATCH DETECTED: {e}")
                logger.error("🔧 Ensure Playwright client version 1.41.0 matches Railway Browserless server v1.41")
            else:
                logger.error(f"❌ Structure-preserving extraction failed: {e}")
            return ""

        finally:
            # Clean up resources in reverse order
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=5.0)
                except Exception as e:
                    logger.debug(f"Page close error (non-critical): {e}")

            if browser:
                try:
                    await asyncio.wait_for(browser.close(), timeout=5.0)
                except Exception as e:
                    logger.debug(f"Browser close error (non-critical): {e}")

            if playwright:
                try:
                    await playwright.stop()
                except Exception as e:
                    logger.debug(f"Playwright stop error (non-critical): {e}")

    async def _configure_page_railway_optimized(self, page: Page):
        """
        RAILWAY BROWSERLESS OPTIMIZED: Page configuration for v1.41 compatibility
        Optimized for Railway Browserless service performance
        """
        try:
            # Railway Browserless optimized headers
            await page.set_extra_http_headers({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            })

            # Block unnecessary resources for faster Railway Browserless performance
            await page.route("**/*", self._block_resources)

            # Railway Browserless optimized script injection
            await page.add_init_script("""
                // Disable image loading for speed
                Object.defineProperty(HTMLImageElement.prototype, 'src', {
                    set: function() { /* blocked */ },
                    get: function() { return ''; }
                });

                // Speed up animations for Railway Browserless
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

                // Disable popups for cleaner extraction
                window.Notification = undefined;
                window.alert = () => {};
                window.confirm = () => true;
                window.prompt = () => '';
            """)

        except Exception as e:
            logger.warning(f"⚠️ Page configuration partially failed: {e}")

    async def _block_resources(self, route):
        """
        RAILWAY BROWSERLESS OPTIMIZATION: Block unnecessary resources
        Speeds up loading and reduces Railway Browserless resource usage
        """
        resource_type = route.request.resource_type
        url = route.request.url

        # Block resource types that slow down scraping
        if resource_type in ['image', 'media', 'font', 'stylesheet']:
            await route.abort()
        # Block known tracking and ad domains
        elif any(domain in url for domain in [
            'google-analytics', 'googletagmanager', 'facebook.com', 
            'doubleclick', 'adsystem', 'amazon-adsystem', 'googlesyndication'
        ]):
            await route.abort()
        else:
            await route.continue_()

    async def _extract_structured_content(self, page: Page) -> str:
        """
        ENHANCED: Extract content while preserving structure (headings, paragraphs)
        COMPLETE: Includes full extraction logic from original implementation
        """
        try:
            # Wait for content to load
            await page.wait_for_load_state("domcontentloaded")

            # Dismiss any overlays first
            await self._dismiss_overlays(page)

            # Light scroll to trigger lazy loading
            await self._light_scroll(page)

            logger.info("🗏️ STRUCTURE-PRESERVING EXTRACTION v1.41")

            # ENHANCED: Complete structured content extraction script
            content_data = await page.evaluate("""
                () => {
                    // Function to convert HTML element to structured text
                    function htmlToStructuredText(element, level = 0) {
                        if (!element) return '';

                        let result = '';
                        const indent = '  '.repeat(level);

                        // Handle text nodes
                        if (element.nodeType === Node.TEXT_NODE) {
                            const text = element.textContent.trim();
                            return text ? text + ' ' : '';
                        }

                        // Handle element nodes
                        if (element.nodeType === Node.ELEMENT_NODE) {
                            const tagName = element.tagName.toLowerCase();

                            switch (tagName) {
                                case 'h1':
                                case 'h2':
                                case 'h3':
                                case 'h4':
                                case 'h5':
                                case 'h6':
                                    const headingText = element.textContent.trim();
                                    if (headingText) {
                                        result += '\\n## ' + headingText + '\\n\\n';
                                    }
                                    break;

                                case 'p':
                                case 'div':
                                case 'span':
                                case 'section':
                                case 'article':
                                    // For paragraphs and content containers
                                    let containerText = '';
                                    for (let child of element.childNodes) {
                                        containerText += htmlToStructuredText(child, level + 1);
                                    }
                                    containerText = containerText.trim();
                                    if (containerText && containerText.length > 10) {
                                        result += containerText;
                                        if (!containerText.endsWith('\\n')) {
                                            result += '\\n\\n';
                                        }
                                    }
                                    break;

                                case 'br':
                                    result += '\\n';
                                    break;

                                case 'li':
                                    const listText = element.textContent.trim();
                                    if (listText) {
                                        result += '• ' + listText + '\\n';
                                    }
                                    break;

                                case 'strong':
                                case 'b':
                                    const strongText = element.textContent.trim();
                                    if (strongText) {
                                        result += '**' + strongText + '**';
                                    }
                                    break;

                                case 'em':
                                case 'i':
                                    const emText = element.textContent.trim();
                                    if (emText) {
                                        result += '*' + emText + '*';
                                    }
                                    break;

                                default:
                                    // For other elements, process children
                                    for (let child of element.childNodes) {
                                        result += htmlToStructuredText(child, level + 1);
                                    }
                                    break;
                            }
                        }

                        return result;
                    }

                    // Strategy 1: Try to find main content containers first
                    const mainSelectors = [
                        'main',
                        'article', 
                        '[role="main"]',
                        '.main-content',
                        '.content',
                        '.article-content',
                        '.post-content',
                        '#content',
                        '#main',
                        '.entry-content'
                    ];

                    let bestContent = '';
                    let bestScore = 0;
                    let selectedElement = null;

                    for (const selector of mainSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            // Remove unwanted elements from a clone
                            const clone = element.cloneNode(true);

                            // Remove navigation, sidebar, ads, etc.
                            const unwanted = clone.querySelectorAll(
                                'script, style, nav, header, footer, aside, .nav, .navbar, .menu, ' +
                                '.advertisement, .ad, .ads, .sidebar, .cookie, .popup, .modal, ' +
                                '.social, .share, .comment, .related, .newsletter, .subscribe, ' +
                                '.breadcrumb, .pagination, .tags, .meta, .author, .date'
                            );
                            unwanted.forEach(el => el.remove());

                            const content = htmlToStructuredText(clone);
                            const score = content.length;

                            if (score > bestScore) {
                                bestScore = score;
                                bestContent = content;
                                selectedElement = selector;
                            }
                        }
                    }

                    // Strategy 2: If no main container found, extract from body
                    if (!bestContent || bestContent.length < 200) {
                        // Remove unwanted elements from body
                        const bodyClone = document.body.cloneNode(true);
                        const unwanted = bodyClone.querySelectorAll(
                            'script, style, nav, header, footer, aside, .nav, .navbar, .menu, ' +
                            '.advertisement, .ad, .ads, .sidebar, .cookie, .popup, .modal, ' +
                            '.social, .share, .comment, .related, .newsletter, .subscribe, ' +
                            '.breadcrumb, .pagination, .tags, .meta, .author, .date'
                        );
                        unwanted.forEach(el => el.remove());

                        const bodyContent = htmlToStructuredText(bodyClone);
                        if (bodyContent.length > bestContent.length) {
                            bestContent = bodyContent;
                            selectedElement = 'body (filtered)';
                        }
                    }

                    // Strategy 3: Fallback to specific content selectors
                    if (!bestContent || bestContent.length < 100) {
                        const contentSelectors = [
                            'h1, h2, h3, h4, h5, h6, p, div.text, .description, .summary, .content-text',
                            'p, div[class*="content"], div[class*="text"], div[class*="description"]',
                            '[class*="restaurant"], [class*="review"], [class*="listing"]'
                        ];

                        for (const selectorGroup of contentSelectors) {
                            const elements = document.querySelectorAll(selectorGroup);
                            let fallbackContent = '';

                            for (const element of elements) {
                                const text = element.textContent?.trim();
                                if (text && text.length > 20) {
                                    if (element.tagName.match(/H[1-6]/)) {
                                        fallbackContent += '\\n## ' + text + '\\n\\n';
                                    } else {
                                        fallbackContent += text + '\\n\\n';
                                    }
                                }
                            }

                            if (fallbackContent.length > bestContent.length) {
                                bestContent = fallbackContent;
                                selectedElement = 'fallback: ' + selectorGroup.substring(0, 50);
                            }
                        }
                    }

                    return {
                        content: bestContent.trim(),
                        selector: selectedElement,
                        score: bestScore,
                        strategies_tried: ['main_containers', 'body_filtered', 'content_selectors']
                    };
                }
            """)

            extracted_content = content_data.get('content', '')
            selector_used = content_data.get('selector', 'unknown')

            if extracted_content and len(extracted_content.strip()) > 50:
                logger.info(f"✅ Structure extraction successful: {len(extracted_content)} chars using '{selector_used}'")

                # Track structure quality
                self._calculate_structure_quality(extracted_content)

                return extracted_content
            else:
                logger.warning(f"⚠️ Structure extraction yielded insufficient content from '{selector_used}'")
                # Fallback to simpler extraction
                fallback_content = await page.inner_text('body')
                return fallback_content[:5000] if fallback_content else ""

        except Exception as e:
            logger.warning(f"⚠️ Structured extraction failed: {e}, falling back to basic")
            try:
                fallback_content = await page.inner_text('body')
                return fallback_content[:5000] if fallback_content else ""
            except:
                return ""

    async def _dismiss_overlays(self, page: Page):
        """
        ENHANCED: Dismiss overlays and popups with better error handling
        """
        overlay_selectors = [
            '[data-testid="gdpr-banner"] button',
            '.cookie-consent button',
            '.cookie-banner button:contains("Accept")',
            '.modal-close',
            '.popup-close',
            '[aria-label="close"]',
            '.close-button',
            'button:contains("Close")',
            'button:contains("Accept")',
            'button:contains("Agree")',
        ]

        for selector in overlay_selectors:
            try:
                await page.click(selector, timeout=2000)
                await asyncio.sleep(0.5)
                logger.debug(f"✅ Dismissed overlay: {selector}")
                break
            except:
                continue

    def _clean_restaurant_content(self, content: str) -> str:
        """
        ENHANCED: Clean and optimize restaurant content
        """
        if not content:
            return ""

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)

        # Remove common noise patterns
        noise_patterns = [
            r'cookies?\s+policy',
            r'privacy\s+policy', 
            r'terms\s+of\s+service',
            r'newsletter\s+signup',
            r'follow\s+us',
            r'share\s+this',
            r'related\s+articles?',
        ]

        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        return content.strip()

    def get_stats(self) -> Dict[str, Any]:
        """
        ENHANCED: Return comprehensive scraping statistics with v1.41 compatibility info
        """
        return self.stats.copy()

    def print_stats(self):
        """
        ENHANCED: Print comprehensive scraping statistics with v1.41 compatibility info
        """
        if self.stats["total_scraped"] > 0:
            # Calculate average scrape time
            if self.stats["successful_scrapes"] > 0:
                self.stats["avg_scrape_time"] = self.stats["total_processing_time"] / self.stats["successful_scrapes"]

            success_rate = (self.stats["successful_scrapes"] / self.stats["total_scraped"]) * 100

            logger.info("=" * 60)
            logger.info("🎯 ENHANCED RESTAURANT SCRAPER STATISTICS")
            logger.info("=" * 60)
            logger.info(f"   📊 Success Rate: {success_rate:.1f}% ({self.stats['successful_scrapes']}/{self.stats['total_scraped']})")
            logger.info(f"   ⚡ Avg Scrape Time: {self.stats['avg_scrape_time']:.2f}s")
            logger.info(f"   🚂 Railway Browserless: {'✓' if self.browserless_endpoint else '✗'}")
            logger.info(f"   🎭 Playwright Version: {self.stats['playwright_version']}")
            logger.info(f"   🗏️ Structure Preservation:")
            logger.info(f"      Successful extractions: {self.stats['structure_preserving_success']}")
            logger.info(f"      Avg headings per page: {self.stats['structure_quality']['avg_heading_count']:.1f}")
            logger.info(f"      Avg paragraphs per page: {self.stats['structure_quality']['avg_paragraph_count']:.1f}")

            if self.stats["cost_saved_vs_all_firecrawl"] > 0:
                logger.info(f"   💰 Estimated cost savings: {self.stats['cost_saved_vs_all_firecrawl']:.2f} credits")

            logger.info("=" * 60)

    # ============ COMPATIBILITY METHODS ============

    async def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """
        LEGACY COMPATIBILITY: Legacy method for direct URL scraping
        Converts URL list to search results format for pipeline compatibility
        """
        logger.info(f"🔄 Legacy scrape_urls called with {len(urls)} URLs")
        search_results = [{"url": url, "title": "", "snippet": ""} for url in urls]
        return await self.scrape_search_results(search_results)

    async def _light_scroll(self, page: Page):
        """
        ENHANCED: Light scrolling to trigger lazy loading
        """
        try:
            # Get page height for smart scrolling
            page_height = await page.evaluate("document.body.scrollHeight")
            viewport_height = await page.evaluate("window.innerHeight")

            if page_height > viewport_height:
                # Scroll to middle, then back to top
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(self.interaction_delay)
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(self.interaction_delay)

        except Exception as e:
            logger.debug(f"Light scroll failed (non-critical): {e}")

    def _calculate_structure_quality(self, content: str):
        """
        ENHANCED: Calculate and track structure quality metrics
        """
        if not content:
            return

        # Count headings and paragraphs
        heading_count = len(re.findall(r'\n## .+\n', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip() and not p.startswith('## ')])

        # Update running averages
        total_extractions = self.stats['structure_preserving_success']
        if total_extractions > 0:
            current_avg_headings = self.stats['structure_quality']['avg_heading_count']
            current_avg_paragraphs = self.stats['structure_quality']['avg_paragraph_count']

            # Update running averages
            self.stats['structure_quality']['avg_heading_count'] = (
                (current_avg_headings * (total_extractions - 1) + heading_count) / total_extractions
            )
            self.stats['structure_quality']['avg_paragraph_count'] = (
                (current_avg_paragraphs * (total_extractions - 1) + paragraph_count) / total_extractions
            )
        else:
            self.stats['structure_quality']['avg_heading_count'] = heading_count
            self.stats['structure_quality']['avg_paragraph_count'] = paragraph_count

        # Track preservation success
        self.stats['structure_quality']['headings_preserved'] += heading_count
        self.stats['structure_quality']['paragraphs_preserved'] += paragraph_count

    async def close(self):
        """
        ENHANCED: Clean shutdown with better resource management
        """
        logger.info("🛑 Shutting down Enhanced Restaurant Scraper...")

        # Wait for active operations to complete
        max_wait = 30  # seconds
        wait_time = 0
        while self._active_operations > 0 and wait_time < max_wait:
            logger.info(f"⏳ Waiting for {self._active_operations} active operations...")
            await asyncio.sleep(1)
            wait_time += 1

        # Force cleanup if operations are still running
        if self._active_operations > 0:
            logger.warning(f"⚠️ Force closing with {self._active_operations} operations still active")

        # Print final stats
        if self.stats["total_scraped"] > 0:
            self.print_stats()

        logger.info("✅ Enhanced Restaurant Scraper shutdown complete")