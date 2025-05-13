# agents/scraper.py - Optimized for Microsoft Playwright container
import asyncio, logging, time, re, json, os
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

# Create semaphore for concurrency control
SEM = asyncio.Semaphore(3)  # Adjust based on Railway plan RAM

# We'll assume Playwright is available in the Microsoft container
PLAYWRIGHT_ENABLED = True

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    logger = logging.getLogger("restaurant-recommender.scraper")
    logger.info("Playwright successfully imported")
except ImportError as e:
    PLAYWRIGHT_ENABLED = False
    logger = logging.getLogger("restaurant-recommender.scraper")
    logger.error(f"Failed to import Playwright: {e}, falling back to HTTP methods")

from readability import Document
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled

from utils.async_utils import track_async_task
from utils.debug_utils import dump_chain_state

logger = logging.getLogger("restaurant-recommender.scraper")

class WebScraper:
    # In WebScraper class

    def __init__(self, config):
        self.config = config

        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
        )

        self._concurrency = getattr(config, "SCRAPER_CONCURRENCY", 3)
        self._semaphore = None  # This will be created per event loop

        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze if a web page contains a curated list of restaurants or restaurant recommendations.

        VALID CONTENT (score > 0.7):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris")
        - Collections of restaurants in professional restaurant guides
        - Food critic reviews covering multiple restaurants
        - Articles in reputable media discussing various dining options in an area

        NOT VALID CONTENT (score < 0.3):
        - Official website of a single restaurant
        - Collections of restaurants in booking and delivery websites like Uber Eats, The Fork, Glovo, etc.
        - Wanderlog content
        - Individual restaurant menus
        - Single restaurant reviews
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites

        SCORING CRITERIA:
        - Multiple restaurants mentioned (essential)
        - Professional curation or expertise evident
        - Detailed descriptions of restaurants/cuisine
        - Location information for multiple restaurants
        - Price or quality indications for multiple venues

        FORMAT:
        Respond with a JSON object containing:
        {{
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "reasoning": "brief explanation of your evaluation"
        }}
        """

        self.eval_prompt = ChatPromptTemplate.from_messages(
            [("system", self.eval_system_prompt),
             ("human", "URL: {url}\n\nPage Title: {title}\n\nContent Preview:\n{preview}")]
        )
        self.eval_chain = self.eval_prompt | self.model

        self.browser_config = {
            "headless": True,
            "timeout": 30_000,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "args": ["--disable-dev-shm-usage"]  # Prevents /dev/shm crashes
        }

        self.successful_urls, self.failed_urls = [], []

    # -------------------------------------------------
    # Internal helpers
    async def _process_search_result(self, result: Dict[str, Any], browser=None, sem=None) -> Optional[Dict[str, Any]]:
        """
        Process a single search result through the full pipeline

        Args:
            result: Search result dictionary with URL
            browser: Optional browser instance for Playwright
            sem: Optional semaphore to use for concurrency control

        Returns:
            Enriched result dictionary or None if processing failed
        """
        # Use the passed semaphore if available, otherwise default to the global one
        semaphore = sem if sem is not None else SEM

        async with semaphore:  # Use the variable 'semaphore' here, not SEM
            # Rest of the method stays the same
            url = result.get("url")
            if not url:
                return None

            try:
                # 1. domain reputation
                source_domain = self._extract_domain(url)
                result["source_domain"] = source_domain

                # 2. fetch html - either with Playwright or HTTP methods
                if PLAYWRIGHT_ENABLED and browser is not None:
                    fetch_result = await self._fetch_with_playwright(url, browser)
                else:
                    fetch_result = await self._fetch_with_http(url)

                if fetch_result.get("error"):
                    logger.warning(f"Failed to fetch URL: {url}, Error: {fetch_result['error']}")
                    self.failed_urls.append(url)
                    return None

                # 3. AI evaluation
                evaluation = await self._evaluate_content(
                    url,
                    fetch_result.get("title", ""),
                    fetch_result.get("content_preview", ""),
                )
                if not (
                    evaluation.get("is_restaurant_list")
                    and evaluation.get("content_quality", 0) > 0.5
                ):
                    logger.info(f"Filtered URL by content evaluation: {url}")
                    self.invalid_content_urls.append(url)
                    return None

                # 4. text extraction
                processed_content = await self._process_content(
                    fetch_result.get("html", ""), 
                    fetch_result.get("structured_content")
                )

                # Extract and format source information
                source_info = self._extract_source_info(
                    url, 
                    fetch_result.get("title", ""), 
                    source_domain,
                    result.get("favicon", "")
                )

                # 5. pack result
                enriched_result = {
                    **result,
                    "scraped_title": fetch_result.get("title", ""),
                    "scraped_content": processed_content,
                    "content_length": len(processed_content),
                    "source_reputation": source_validation.get("reputation_score", 0.5),
                    "quality_score": evaluation.get("content_quality", 0.0),
                    "restaurant_count": evaluation.get("restaurant_count", 0),
                    "source_info": source_info,
                    "timestamp": time.time(),
                }
                self.successful_urls.append(url)
                logger.info(f"Successfully processed: {url}")
                return enriched_result

            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                self.failed_urls.append(url)
                return None


    # -------------------------------------------------
    # Public orchestrator
    async def filter_and_scrape_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: Filter search results and scrape valid content

        Args:
            search_results: List of search result dictionaries with URLs

        Returns:
            List of enriched results with scraped content
        """
        logger.info(f"PLAYWRIGHT_ENABLED: {PLAYWRIGHT_ENABLED}")
        if PLAYWRIGHT_ENABLED:
            try:
                # Test Playwright availability
                async with async_playwright() as p:
                    browser_version = await p.chromium.executable_path
                    logger.info(f"Playwright browser executable: {browser_version}")
                    logger.info("Playwright is available and functioning")
            except Exception as e:
                logger.error(f"Playwright test failed: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Will fall back to HTTP methods for all scraping")

        # Create a local semaphore for this run
        local_sem = asyncio.Semaphore(3)

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            start_time = time.time()
            logger.info(f"Starting to process {len(search_results)} search results")

            # Reset trackers
            self.successful_urls = []
            self.failed_urls = []
            self.invalid_content_urls = []

            # Process results in batches to avoid overwhelming the system
            enriched_results = []
            batch_size = 5

            # Initialize a single browser instance for the whole batch if Playwright is enabled
            if PLAYWRIGHT_ENABLED:
                try:
                    logger.info("Initializing Playwright browser for batch processing")
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=["--disable-dev-shm-usage"]  # Prevents /dev/shm crashes
                        )

                        # Process batches with the shared browser instance
                        for i in range(0, len(search_results), batch_size):
                            batch = search_results[i:i+batch_size]
                            # Pass the local semaphore to _process_search_result
                            batch_tasks = [self._process_search_result(result, browser, local_sem) for result in batch]
                            batch_results = await asyncio.gather(*batch_tasks)

                            # Filter out None results (failed processing)
                            valid_results = [r for r in batch_results if r is not None]
                            enriched_results.extend(valid_results)

                            # Small delay between batches to be respectful to servers
                            await asyncio.sleep(1)

                        # Close the browser when done
                        await browser.close()
                except Exception as e:
                    logger.error(f"Error initializing Playwright: {e}. Falling back to HTTP methods.")
                    # Fall back to HTTP methods
                    for i in range(0, len(search_results), batch_size):
                        batch = search_results[i:i+batch_size]
                        batch_tasks = [self._process_search_result(result) for result in batch]
                        batch_results = await asyncio.gather(*batch_tasks)

                        valid_results = [r for r in batch_results if r is not None]
                        enriched_results.extend(valid_results)

                        await asyncio.sleep(1)
            else:
                # Playwright disabled, use HTTP methods instead
                logger.info("Playwright is disabled, using HTTP methods")
                for i in range(0, len(search_results), batch_size):
                    batch = search_results[i:i+batch_size]
                    batch_tasks = [self._process_search_result(result) for result in batch]
                    batch_results = await asyncio.gather(*batch_tasks)

                    valid_results = [r for r in batch_results if r is not None]
                    enriched_results.extend(valid_results)

                    await asyncio.sleep(1)

            elapsed = time.time() - start_time

            # Log comprehensive statistics
            logger.info(f"Scraping completed in {elapsed:.2f} seconds")
            logger.info(f"Total results: {len(search_results)}")
            logger.info(f"Successfully scraped: {len(self.successful_urls)}")
            logger.info(f"Failed to scrape: {len(self.failed_urls)}")
            logger.info(f"Filtered by content evaluator: {len(self.invalid_content_urls)}")
            logger.info(f"Final enriched results: {len(enriched_results)}")

            # Dump state for debugging
            dump_chain_state("scraper_results", {
                "total_results": len(search_results),
                "successful_urls": self.successful_urls,
                "failed_urls": self.failed_urls,
                "invalid_content_urls": self.invalid_content_urls,
                "final_count": len(enriched_results)
            })

            return enriched_results



    async def _fetch_with_playwright(self, url: str, browser) -> Dict[str, Any]:
        """
        Fetch a URL using Playwright with an existing browser instance

        Args:
            url: URL to fetch
            browser: Browser instance

        Returns:
            Dictionary with HTML content and metadata
        """
        logger.info(f"Fetching URL with Playwright: {url}")
        result = {
            "url": url,
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "error": None,
            "content_length": 0
        }

        try:
            # Create context with additional options
            context = await browser.new_context(
                user_agent=self.browser_config['user_agent'],
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="Europe/London",
                has_touch=False
            )

            # Set up page with improved error handling
            page = await context.new_page()
            page.set_default_timeout(self.browser_config['timeout'])

            # Configure navigation and wait options
            navigation_options = {
                "wait_until": "domcontentloaded",
                "timeout": self.browser_config['timeout'],
                "referer": "https://www.google.com/",
            }

            # Navigate to the URL with better options
            response = await page.goto(url, **navigation_options)
            result["status_code"] = response.status if response else 0

            # Improved content loading - wait for network idle
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeoutError:
                # Continue anyway, many pages never reach network idle
                pass

            # Get page content with better techniques
            result["html"] = await page.content()
            result["title"] = await page.title()

            # Get a content preview using JavaScript evaluation
            try:
                content_text = await page.evaluate('''() => {
                    // Get visible text only
                    function getVisibleText(node) {
                        if (node.nodeType === Node.TEXT_NODE) {
                            return node.textContent || "";
                        }

                        const style = window.getComputedStyle(node);
                        if (style && (style.display === "none" || style.visibility === "hidden")) {
                            return "";
                        }

                        let text = "";
                        for (let child of node.childNodes) {
                            if (child.nodeType === Node.TEXT_NODE) {
                                text += child.textContent || "";
                            } else if (child.nodeType === Node.ELEMENT_NODE) {
                                text += getVisibleText(child);
                            }
                        }
                        return text;
                    }

                    // Use main content area if available
                    const mainContent = document.querySelector('main') || 
                                      document.querySelector('article') || 
                                      document.querySelector('.content') || 
                                      document.body;

                    return getVisibleText(mainContent).trim().substring(0, 2000);
                }''')
                result["content_preview"] = content_text
            except Exception as e:
                # Fallback to simpler extraction
                content_text = await page.evaluate('''() => {
                    // Get ALL visible text with special attention to potential restaurant names
                    const extractedItems = [];

                    // Specifically look for restaurant names in common patterns
                    function findRestaurantNames() {
                        // Restaurant names are often in headers, strong tags, or styled links
                        const potentialNameElements = [];

                        // Collect all headings (h1-h6)
                        document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(el => {
                            const text = el.innerText.trim();
                            if (text) potentialNameElements.push({
                                text: text,
                                type: 'HEADING',
                                tag: el.tagName.toLowerCase()
                            });
                        });

                        // Collect all bold/strong elements (often used for restaurant names)
                        document.querySelectorAll('strong, b').forEach(el => {
                            const text = el.innerText.trim();
                            if (text && text.length < 100) potentialNameElements.push({
                                text: text,
                                type: 'STRONG',
                                tag: el.tagName.toLowerCase()
                            });
                        });

                        // Collect all links (a) with styling that might be restaurant names
                        document.querySelectorAll('a').forEach(el => {
                            const style = window.getComputedStyle(el);
                            if (style.display !== 'none' && style.visibility !== 'hidden') {
                                const text = el.innerText.trim();
                                if (text && text.length < 100) potentialNameElements.push({
                                    text: text,
                                    type: 'LINK',
                                    tag: 'a',
                                    href: el.href
                                });
                            }
                        });

                        // Collect divs and spans with special styling (font-weight, etc.)
                        document.querySelectorAll('div, span').forEach(el => {
                            const style = window.getComputedStyle(el);
                            if (style.fontWeight >= 600 || style.fontSize > '16px') {
                                const text = el.innerText.trim();
                                if (text && text.length < 100 && text.length > 2) potentialNameElements.push({
                                    text: text,
                                    type: 'STYLED',
                                    tag: el.tagName.toLowerCase()
                                });
                            }
                        });

                        return potentialNameElements;
                    }

                    // Main text extraction - get all visible text
                    function getAllVisibleText() {
                        function isVisible(el) {
                            const style = window.getComputedStyle(el);
                            return !(style.display === 'none' || style.visibility === 'hidden');
                        }

                        // Use main content area if available
                        const mainContent = document.querySelector('main') || 
                                           document.querySelector('article') || 
                                           document.querySelector('.content') || 
                                           document.body;

                        const textItems = [];

                        // Get all paragraphs
                        mainContent.querySelectorAll('p').forEach(el => {
                            if (isVisible(el)) {
                                const text = el.innerText.trim();
                                if (text) textItems.push({
                                    text: text,
                                    type: 'PARAGRAPH'
                                });
                            }
                        });

                        // Get all list items
                        mainContent.querySelectorAll('li').forEach(el => {
                            if (isVisible(el)) {
                                const text = el.innerText.trim();
                                if (text) textItems.push({
                                    text: text,
                                    type: 'LIST_ITEM'
                                });
                            }
                        });

                        return textItems;
                    }

                    const restaurantNames = findRestaurantNames();
                    const mainText = getAllVisibleText();

                    return JSON.stringify({
                        restaurantNames: restaurantNames,
                        mainText: mainText
                    });
                }''')

                # Store the structured data
                try:
                    structured_content = json.loads(content_text)
                    result["restaurant_names"] = structured_content.get("restaurantNames", [])
                    result["structured_content"] = structured_content
                    result["content_preview"] = json.dumps(structured_content.get("restaurantNames", []))[:2000]
                except Exception as e:
                    logger.error(f"Error parsing structured content: {e}")
                    # Fallback to simple content preview
                    result["content_preview"] = content_text[:2000] if content_text else ""

            # Extract content length for logging
            result["content_length"] = len(result["html"])

            # Clean close
            await context.close()

        except PlaywrightTimeoutError:
            result["error"] = "Timeout error"
            logger.warning(f"Timeout fetching URL: {url}")
        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"Error fetching URL: {url}, Error: {str(e)}")

        return result

    async def _fetch_with_http(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL using HTTP libraries (requests/aiohttp)

        Args:
            url: URL to fetch

        Returns:
            Dictionary with HTML content and metadata
        """
        logger.info(f"Fetching URL with HTTP: {url}")
        result = {
            "url": url,
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "error": None,
            "content_length": 0
        }

        try:
            # Try to use aiohttp first (async)
            try:
                import aiohttp

                headers = {
                    "User-Agent": self.browser_config['user_agent'],
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Accept-Language": "en-US,en;q=0.9",
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=15) as response:
                        result["status_code"] = response.status

                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            result["html"] = html
                            result["title"] = soup.title.text if soup.title else ""

                            # Extract content preview
                            main_content = soup.find('main') or soup.find('article') or soup.find(class_='content') or soup.body
                            if main_content:
                                preview_text = main_content.get_text(separator=' ', strip=True)
                                result["content_preview"] = preview_text[:2000]
                            else:
                                result["content_preview"] = soup.get_text(separator=' ', strip=True)[:2000]

                            result["content_length"] = len(html)
                            return result
                        else:
                            result["error"] = f"HTTP error: {response.status}"
                            return result

            except ImportError:
                # Fallback to requests if aiohttp is not available
                logger.info("aiohttp not available, falling back to requests")
                pass
            except Exception as e:
                logger.warning(f"aiohttp fetch failed: {str(e)}, falling back to requests")
                pass

            # Fallback to requests (synchronous)
            import requests

            headers = {
                "User-Agent": self.browser_config['user_agent'],
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
            }

            response = requests.get(url, headers=headers, timeout=15)
            result["status_code"] = response.status_code

            if response.status_code == 200:
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')

                result["html"] = html
                result["title"] = soup.title.text if soup.title else ""

                # Extract content preview
                main_content = soup.find('main') or soup.find('article') or soup.find(class_='content') or soup.body
                if main_content:
                    preview_text = main_content.get_text(separator=' ', strip=True)
                    result["content_preview"] = preview_text[:2000]
                else:
                    result["content_preview"] = soup.get_text(separator=' ', strip=True)[:2000]

                result["content_length"] = len(html)
            else:
                result["error"] = f"HTTP error: {response.status_code}"

        except Exception as e:
            result["error"] = f"Error fetching URL: {str(e)}"
            logger.warning(f"Error fetching URL: {url}, Error: {str(e)}")

        return result

    async def _evaluate_content(self, url: str, title: str, preview: str) -> Dict[str, Any]:
        """
        Evaluate if the content is a restaurant list using AI

        Args:
            url: URL of the page
            title: Page title
            preview: Content preview

        Returns:
            Evaluation result with scores
        """
        try:
            # Basic keyword check to avoid LLM calls for obviously irrelevant content
            restaurant_keywords = ["restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish"]
            if not any(kw in title.lower() or kw in preview.lower() for kw in restaurant_keywords):
                logger.info(f"URL filtered by basic keyword check: {url} - No restaurant keywords found")
                return {
                    "is_restaurant_list": False,
                    "restaurant_count": 0,
                    "content_quality": 0.0,
                    "reasoning": "No restaurant-related keywords found"
                }

            response = await self.eval_chain.ainvoke({
                "url": url,
                "title": title,
                "preview": preview[:1500]  # Trim to avoid token limits
            })

            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content.strip())

            # Ensure content_quality is in the response
            if "content_quality" not in evaluation:
                evaluation["content_quality"] = 0.8 if evaluation.get("is_restaurant_list", False) else 0.2

            # Get validity status based on current threshold
            is_restaurant_list = evaluation.get("is_restaurant_list", False)
            content_quality = evaluation.get("content_quality", 0.0)
            reasoning = evaluation.get("reasoning", "No reasoning provided")
            current_threshold = 0.5

            # Log detailed evaluation results
            logger.info(f"Content evaluation for {url}:")
            logger.info(f"  - Title: {title[:50]}...")
            logger.info(f"  - Is Restaurant List: {is_restaurant_list}")
            logger.info(f"  - Quality Score: {content_quality}")
            logger.info(f"  - Current Threshold: {current_threshold}")
            logger.info(f"  - Pass Filter: {is_restaurant_list and content_quality > current_threshold}")
            logger.info(f"  - Reasoning: {reasoning}")

            # If it would pass with a lower threshold but not the current one, log that specifically
            if is_restaurant_list and 0.3 <= content_quality <= current_threshold:
                logger.info(f"  - NOTE: This content would pass with a lower threshold of 0.3")

            # Return the evaluation
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating content for {url}: {str(e)}")
            # Return a conservative default (invalid content)
            return {
                "is_restaurant_list": False,
                "restaurant_count": 0,
                "content_quality": 0.0,
                "source_type": "unknown",
                "reasoning": "Error in evaluation"
            }

    async def _process_content(self, html: str, extracted_content=None) -> str:
        """
        Process HTML content using readability and cleaning with enhanced extraction of restaurant names

        Args:
            html: Raw HTML content
            extracted_content: Optional structured content extracted directly with Playwright

        Returns:
            Cleaned and processed text content with preserved restaurant names
        """
        # Create a collection for all paragraphs/content elements
        paragraphs = []
        restaurant_names = set()  # For deduplication of restaurant names

        # 1. HANDLE PLAYWRIGHT DIRECT EXTRACTION (if available)
        if extracted_content and isinstance(extracted_content, dict):
            logger.info("Using Playwright-extracted structured content")

            # Process restaurant names from Playwright extraction
            playwright_names = extracted_content.get("restaurantNames", [])
            if playwright_names:
                name_paragraphs = []
                for item in playwright_names:
                    if isinstance(item, dict):
                        name_type = item.get("type", "UNKNOWN")
                        name_text = item.get("text", "").strip()
                        if name_text and 2 < len(name_text) < 100:
                            name_paragraphs.append(f"RESTAURANT_NAME [{name_type}]: {name_text}")
                            restaurant_names.add(name_text)

                if name_paragraphs:
                    paragraphs.append("LIKELY RESTAURANT NAMES:")
                    paragraphs.extend(name_paragraphs)
                    paragraphs.append("")  # Empty line separator

            # Process main text content from Playwright
            playwright_text = extracted_content.get("mainText", [])
            if playwright_text:
                for item in playwright_text:
                    if isinstance(item, dict):
                        text_type = item.get("type", "TEXT")
                        text = item.get("text", "").strip()
                        if text:
                            paragraphs.append(f"{text_type}: {text}")

        # 2. HTML PROCESSING WITH READABILITY & BEAUTIFULSOUP
        if not html:
            if paragraphs:
                return "\n\n".join(paragraphs)
            return ""

        try:
            # First extract title from HTML (before readability possibly removes it)
            soup_full = BeautifulSoup(html, 'html.parser')
            page_title = soup_full.title.text.strip() if soup_full.title else ""

            # Add page title if not already included
            if page_title and not any(page_title in p for p in paragraphs):
                paragraphs.insert(0, f"TITLE: {page_title}")

            # Use readability to extract main content
            doc = Document(html)
            readable_html = doc.summary()
            doc_title = doc.title()

            # Add readability title if different from page title
            if doc_title and doc_title != page_title:
                paragraphs.insert(1, f"ARTICLE_TITLE: {doc_title}")

            # Parse the readability-extracted HTML with BeautifulSoup
            soup = BeautifulSoup(readable_html, 'html.parser')

            # RESTAURANT NAME EXTRACTION - Look specifically for patterns that typically contain restaurant names

            # 1. Extract names from links (common for restaurant lists)
            link_names = []
            for link in soup.find_all('a'):
                link_text = link.get_text().strip()
                # Restaurant names are typically short, non-sentence-like text
                if (link_text and 2 < len(link_text) < 60 and 
                        link_text.count(' ') < 8 and
                        not link_text.lower().startswith(('http', 'www', 'click', 'read', 'more'))):
                    if link_text not in restaurant_names:
                        link_names.append(f"LINK: {link_text}")
                        restaurant_names.add(link_text)

            # Only add link section if we found potential restaurant names
            if link_names:
                paragraphs.append("POTENTIAL RESTAURANT NAMES FROM LINKS:")
                paragraphs.extend(link_names)
                paragraphs.append("")  # Separator

            # 2. Extract names from headings
            heading_texts = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_text = heading.get_text().strip()
                if heading_text and heading_text not in restaurant_names:
                    heading_tag = heading.name
                    heading_texts.append(f"HEADING_{heading_tag.upper()}: {heading_text}")
                    restaurant_names.add(heading_text)
                    # Remove processed heading to avoid duplication
                    heading.extract()

            if heading_texts:
                paragraphs.append("HEADINGS (OFTEN RESTAURANT NAMES):")
                paragraphs.extend(heading_texts)
                paragraphs.append("")  # Separator

            # 3. Extract names from styled elements (bold, strong, etc.)
            styled_texts = []
            for element in soup.find_all(['strong', 'b', 'em', 'i', 'mark']):
                styled_text = element.get_text().strip()
                if (styled_text and len(styled_text) < 60 and
                        styled_text not in restaurant_names and
                        not any(styled_text.lower().startswith(word) for word in ['http', 'www'])):
                    styled_texts.append(f"STYLED: {styled_text}")
                    restaurant_names.add(styled_text)

            if styled_texts:
                paragraphs.append("STYLED TEXT (POTENTIAL RESTAURANT NAMES):")
                paragraphs.extend(styled_texts)
                paragraphs.append("")  # Separator

            # 4. Process lists (often contain restaurant info)
            for list_elem in soup.find_all(['ul', 'ol']):
                list_items = []
                for item in list_elem.find_all('li'):
                    item_text = item.get_text().strip()
                    if item_text:
                        # Try to extract restaurant names from list items that look like they contain restaurant names
                        if len(item_text) < 200:  # Not too long
                            list_items.append(f"LIST_ITEM: {item_text}")
                        else:
                            # For longer list items, still include but truncate for readability
                            list_items.append(f"LIST_ITEM: {item_text[:200]}...")

                if list_items:
                    paragraphs.append("LIST ITEMS:")
                    paragraphs.extend(list_items)
                    paragraphs.append("")  # Separator

                # Remove processed list to avoid duplication
                list_elem.extract()

            # 5. Process regular paragraphs
            para_texts = []
            for para in soup.find_all('p'):
                text = para.get_text().strip()
                if text:
                    # Keep paragraphs at a reasonable length
                    if len(text) > 500:
                        para_texts.append(f"PARAGRAPH: {text[:500]}...")
                    else:
                        para_texts.append(f"PARAGRAPH: {text}")
                # Remove processed paragraph to avoid duplication
                para.extract()

            if para_texts:
                paragraphs.append("CONTENT PARAGRAPHS:")
                paragraphs.extend(para_texts)
                paragraphs.append("")  # Separator

            # 6. Get any remaining text not captured above
            remaining_text = soup.get_text().strip()
            if remaining_text:
                # Split by newlines and filter empty strings
                lines = [line.strip() for line in remaining_text.split('\n') if line.strip()]
                if lines:
                    paragraphs.append("ADDITIONAL CONTENT:")
                    for line in lines:
                        # Avoid overly long lines
                        if len(line) > 500:
                            paragraphs.append(f"{line[:500]}...")
                        else:
                            paragraphs.append(line)

            # 7. Process divs with class attributes that might indicate restaurant content
            restaurant_div_keywords = ['restaurant', 'venue', 'place', 'location', 'listing', 'item']
            for div in soup_full.find_all('div', class_=True):
                div_class = ' '.join(div.get('class', []))
                if any(keyword in div_class.lower() for keyword in restaurant_div_keywords):
                    div_text = div.get_text(strip=True)
                    # Limit to reasonable length for restaurant section
                    if 10 < len(div_text) < 500:  
                        # Process nested elements for better structure
                        div_content = []

                        # Check for restaurant name indicators
                        for name_tag in div.find_all(['h3', 'h4', 'strong', 'b']):
                            name_text = name_tag.get_text(strip=True)
                            if name_text and name_text not in restaurant_names:
                                div_content.append(f"NAME: {name_text}")
                                restaurant_names.add(name_text)

                        # Check for address indicators
                        for addr_tag in div.find_all(['address', 'p']):
                            addr_text = addr_tag.get_text(strip=True)
                            if addr_text and len(addr_text) < 200:
                                div_content.append(f"DETAIL: {addr_text}")

                        if div_content:
                            if "RESTAURANT SECTIONS:" not in paragraphs:
                                paragraphs.append("RESTAURANT SECTIONS:")
                            paragraphs.extend(div_content)
                            paragraphs.append("---")  # Section separator

            # Join all paragraphs with appropriate spacing
            processed_text = "\n\n".join(p for p in paragraphs if p)

            # Clean up extra whitespace
            processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
            processed_text = re.sub(r' {2,}', ' ', processed_text)

            return processed_text

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            # Return what we've been able to gather so far, or a simple extraction as fallback
            if paragraphs:
                return "\n\n".join(paragraphs)

            # Last resort fallback - simple text extraction
            try:
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator='\n\n')
            except:
                return "Failed to extract content from HTML"

    def _extract_source_info(self, url: str, title: str, domain: str, favicon: str = "") -> Dict[str, Any]:
        """Extract and format source information"""
        source_type = "Website"

        # Try to determine source type based on domain and title
        if any(kw in domain for kw in ["guide", "michelin"]):
            source_type = "Restaurant Guide"
        elif any(kw in domain for kw in ["news", "times", "post", "magazine"]):
            source_type = "News Publication"
        elif any(kw in domain for kw in ["blog", "food", "critic", "review"]):
            source_type = "Food Blog"

        # Extract source name from domain
        source_name = domain.split('.')[0].capitalize()
        if domain.startswith("www."):
            source_name = domain[4:].split('.')[0].capitalize()

        # If title contains the source name in a better format, use that
        if title and len(title) > 3:
            title_parts = title.split('|')
            if len(title_parts) > 1:
                possible_name = title_parts[-1].strip()
                if 3 < len(possible_name) < 25:  # Reasonable length for a source name
                    source_name = possible_name

        return {
            "name": source_name,
            "domain": domain,
            "type": source_type,
            "favicon": favicon
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url

    # -------------------------------------------------
    # Legacy API compatibility for backward compatibility
    # -------------------------------------------------
    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Legacy API method for backward compatibility

        Args:
            search_results: List of search result dictionaries with URLs

        Returns:
            List of enriched results with scraped content
        """
        logger.info("Using legacy API scrape_search_results - consider migrating to filter_and_scrape_results")
        return await self.filter_and_scrape_results(search_results)

    # Public method for testing URL fetching separately
    async def fetch_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL for testing purposes

        Args:
            url: URL to fetch

        Returns:
            Dictionary with content and status information
        """
        # Use the appropriate fetch method based on availability
        if PLAYWRIGHT_ENABLED:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        headless=True, 
                        args=["--disable-dev-shm-usage"]
                    )
                    result = await self._fetch_with_playwright(url, browser)
                    await browser.close()
                    return result
            except Exception as e:
                logger.error(f"Error using Playwright for testing: {e}, falling back to HTTP")
                return await self._fetch_with_http(url)
        else:
            return await self._fetch_with_http(url)