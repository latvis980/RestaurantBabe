# agents/scraper.py - Enhanced Scraper with Load More and Better List Extraction
import re
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from readability import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled

from utils.source_validator import validate_source
from utils.async_utils import track_async_task
from utils.debug_utils import dump_chain_state

# Configure logger
logger = logging.getLogger("restaurant-recommender.scraper")

# We'll assume Playwright is available in the container
PLAYWRIGHT_ENABLED = True

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    logger.info("Playwright successfully imported")
except ImportError as e:
    PLAYWRIGHT_ENABLED = False
    logger.error(f"Failed to import Playwright: {e}, falling back to HTTP methods")

class WebScraper:
    def __init__(self, config):
        self.config = config

        # Initialize model for content evaluation
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
        )

        # Semaphore for concurrency control
        self._concurrency = getattr(config, "SCRAPER_CONCURRENCY", 3)
        self._semaphore = None

        # Curated list evaluation prompt
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
        - Wanderlog posts
        - Individual restaurant menus
        - Single restaurant reviews
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation

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

        # Restaurant extraction prompt
        self.extraction_prompt = """
        You are an expert travel journalist assistant. Your job is to extract the names of restaurants or cafés 
        and their descriptions from the following text. If available, extract any information about:
        - Address
        - Price range
        - Recommended dishes
        - Chef
        - Atmosphere
        - Reservations requirements
        - Instagram handle

        IMPORTANT: Only include actual food and beverage places. Do not include intros, general tips, ads, 
        or unrelated content. For each restaurant, provide as much of the requested information as available.

        Format each restaurant name in square brackets like [Restaurant Name] for easy identification.
        """

        self.extract_prompt = ChatPromptTemplate.from_messages(
            [("system", self.extraction_prompt),
             ("human", "Extract restaurant information from this text:\n\n{content}")]
        )
        self.extract_chain = self.extract_prompt | self.model

        # Browser configuration
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

        # Tracking variables
        self.successful_urls = []
        self.failed_urls = []
        self.filtered_urls = []
        self.invalid_content_urls = []

    # New method for handling Load More buttons
    async def _handle_load_more_buttons(self, page, max_clicks=5):
        """Handle 'Load More' buttons to expand content"""
        clicks = 0
        while clicks < max_clicks:
            try:
                # Look for common "Load More" button patterns
                load_more_selectors = [
                    'button:has-text("Load more")',
                    'button:has-text("Show more")',
                    'button:has-text("Load more cafés")',
                    'a:has-text("Load more")',
                    'a:has-text("Show more")',
                    '.load-more',
                    '#load-more',
                    '[class*="load-more"]',
                    '[id*="load-more"]',
                    '[class*="show-more"]',
                    'button[data-action*="load"]'
                ]

                button_found = False
                for selector in load_more_selectors:
                    try:
                        button = await page.query_selector(selector)
                        if button and await button.is_visible():
                            # Scroll to button to ensure it's in view
                            await button.scroll_into_view_if_needed()
                            await asyncio.sleep(1)  # Wait for scroll

                            # Click the button
                            await button.click()
                            await asyncio.sleep(2)  # Wait for content to load

                            button_found = True
                            clicks += 1
                            logger.info(f"Clicked '{selector}' button (click {clicks})")
                            break
                    except Exception:
                        continue

                if not button_found:
                    break  # No more buttons found

            except Exception as e:
                logger.warning(f"Error clicking load more button: {e}")
                break

        return clicks

    # New method for extracting structured restaurant lists
    async def _extract_restaurant_list(self, page):
        """Extract structured restaurant/cafe lists from pages"""
        try:
            # Get all potential restaurant entries
            restaurant_data = await page.evaluate('''() => {
                const restaurants = [];

                // Common selectors for restaurant listings
                const selectors = [
                    // Cards/grid items (European Coffee Trip uses these)
                    '[data-element_type="widget"] a[href*="/cafe/"]',
                    '[data-element_type="widget"] a[href*="/restaurant/"]',
                    '.elementor-widget-container a[href*="/cafe/"]',
                    '.cafe-card, .restaurant-card, [class*="card"], [class*="item"]',
                    // List items with links
                    'li a[href*="/cafe/"], li a[href*="/restaurant/"]',
                    // Named sections with links
                    'h2 + a, h3 + a, h4 + a',
                    // Sections or articles
                    'article, section[class*="restaurant"], section[class*="cafe"]'
                ];

                for (const selector of selectors) {
                    const elements = document.querySelectorAll(selector);

                    for (const element of elements) {
                        let name = '';
                        let url = '';
                        let description = '';

                        // Extract name from various possible locations
                        const nameElements = [
                            element.querySelector('h2, h3, h4, h5, h6'),
                            element.querySelector('[class*="title"], [class*="name"]'),
                            element.querySelector('a'),
                            element  // The element itself might be the link
                        ];

                        for (const nameEl of nameElements) {
                            if (nameEl && nameEl.textContent.trim()) {
                                name = nameEl.textContent.trim();
                                // Clean up the name (remove awards, dates, etc.)
                                name = name.replace(/NEW|2024 WINNER|\\d{4}|⭐|★/g, '').trim();
                                if (name) break;
                            }
                        }

                        // Extract URL
                        const linkEl = element.querySelector('a') || element;
                        if (linkEl && linkEl.href) {
                            url = linkEl.href;
                        }

                        // Extract description
                        const descElements = [
                            element.querySelector('p, .description, [class*="desc"]'),
                            element.nextElementSibling
                        ];

                        for (const descEl of descElements) {
                            if (descEl && descEl.textContent.trim()) {
                                description = descEl.textContent.trim();
                                break;
                            }
                        }

                        // Only add if we have a name and it looks like a restaurant/cafe
                        if (name && (
                            name.match(/café|cafe|coffee|restaurant|bistro|bar|brasserie/i) ||
                            url.match(/cafe|restaurant|coffee/i) ||
                            description.match(/café|cafe|coffee|restaurant|dining/i) ||
                            name.length > 2  // Catch names without keywords
                        )) {
                            restaurants.push({
                                name: name,
                                url: url,
                                description: description,
                                type: 'cafe'
                            });
                        }
                    }
                }

                // Deduplicate by name
                const unique = [];
                const seen = new Set();

                for (const restaurant of restaurants) {
                    const key = restaurant.name.toLowerCase().trim();
                    if (!seen.has(key) && restaurant.name.length > 2) {
                        seen.add(key);
                        unique.push(restaurant);
                    }
                }

                return unique;
            }''')

            logger.info(f"Extracted {len(restaurant_data)} restaurants from structured list")
            for restaurant in restaurant_data[:5]:  # Log first 5
                logger.info(f"  - [{restaurant['name']}]")

            return restaurant_data
        except Exception as e:
            logger.error(f"Error extracting restaurant list: {e}")
            return []

    # Main entry point
    async def filter_and_scrape_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point: Filter search results and scrape valid content
        """
        # Create a local semaphore for this run
        local_sem = asyncio.Semaphore(self._concurrency)

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            start_time = time.time()
            logger.info(f"Starting to process {len(search_results)} search results")

            # Reset trackers
            self.successful_urls = []
            self.failed_urls = []
            self.filtered_urls = []
            self.invalid_content_urls = []

            # Process results in batches
            enriched_results = []
            batch_size = 5

            # Initialize Playwright browser if available
            if PLAYWRIGHT_ENABLED:
                try:
                    logger.info("Initializing Playwright browser for batch processing")
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=["--disable-dev-shm-usage"]
                        )

                        # Process batches with shared browser
                        for i in range(0, len(search_results), batch_size):
                            batch = search_results[i:i+batch_size]
                            batch_tasks = [self._process_search_result(result, browser, local_sem) for result in batch]
                            batch_results = await asyncio.gather(*batch_tasks)

                            # Filter out None results
                            valid_results = [r for r in batch_results if r is not None]
                            enriched_results.extend(valid_results)

                            # Be respectful to servers
                            await asyncio.sleep(1)

                        await browser.close()
                except Exception as e:
                    logger.error(f"Error initializing Playwright: {e}. Falling back to HTTP methods.")
                    # Fall back to HTTP methods
                    for i in range(0, len(search_results), batch_size):
                        batch = search_results[i:i+batch_size]
                        batch_tasks = [self._process_search_result(result, None, local_sem) for result in batch]
                        batch_results = await asyncio.gather(*batch_tasks)

                        valid_results = [r for r in batch_results if r is not None]
                        enriched_results.extend(valid_results)

                        await asyncio.sleep(1)
            else:
                # Playwright disabled, use HTTP methods
                logger.info("Playwright is disabled, using HTTP methods")
                for i in range(0, len(search_results), batch_size):
                    batch = search_results[i:i+batch_size]
                    batch_tasks = [self._process_search_result(result, None, local_sem) for result in batch]
                    batch_results = await asyncio.gather(*batch_tasks)

                    valid_results = [r for r in batch_results if r is not None]
                    enriched_results.extend(valid_results)

                    await asyncio.sleep(1)

            elapsed = time.time() - start_time

            # Log statistics
            logger.info(f"Scraping completed in {elapsed:.2f} seconds")
            logger.info(f"Total results: {len(search_results)}")
            logger.info(f"Successfully scraped: {len(self.successful_urls)}")
            logger.info(f"Failed to scrape: {len(self.failed_urls)}")
            logger.info(f"Filtered by source validator: {len(self.filtered_urls)}")
            logger.info(f"Filtered by content evaluator: {len(self.invalid_content_urls)}")
            logger.info(f"Final enriched results: {len(enriched_results)}")

            # Debug state
            dump_chain_state("scraper_results", {
                "total_results": len(search_results),
                "successful_urls": len(self.successful_urls),
                "failed_urls": len(self.failed_urls),
                "filtered_urls": len(self.filtered_urls),
                "invalid_content_urls": len(self.invalid_content_urls),
                "final_count": len(enriched_results)
            })

            return enriched_results

    # Processing a single search result
    async def _process_search_result(self, result: Dict[str, Any], browser=None, sem=None) -> Optional[Dict[str, Any]]:
        """Process a single search result through the full pipeline"""
        semaphore = sem if sem is not None else asyncio.Semaphore(self._concurrency)

        async with semaphore:
            url = result.get("url")
            if not url:
                return None

            try:
                # 1. Source validation
                source_domain = self._extract_domain(url)
                result["source_domain"] = source_domain

                source_validation = validate_source(source_domain, self.config)
                if not source_validation["is_valid"]:
                    logger.info(f"Filtered URL by source validation: {url}")
                    self.filtered_urls.append(url)
                    return None

                # 2. Fetch content
                if PLAYWRIGHT_ENABLED and browser is not None:
                    fetch_result = await self._fetch_with_playwright(url, browser)
                else:
                    fetch_result = await self._fetch_with_http(url)

                if fetch_result.get("error"):
                    logger.warning(f"Failed to fetch URL: {url}, Error: {fetch_result['error']}")
                    self.failed_urls.append(url)
                    return None

                # 3. Evaluate if it's a restaurant list
                evaluation = await self._evaluate_content(
                    url,
                    fetch_result.get("title", ""),
                    fetch_result.get("content_preview", ""),
                    fetch_result
                )

                if not (
                    evaluation.get("is_restaurant_list")
                    and evaluation.get("content_quality", 0) > 0.5
                ):
                    logger.info(f"Filtered URL by content evaluation: {url}")
                    self.invalid_content_urls.append(url)
                    return None

                # 4. Process and extract content
                processed_content = await self._process_content(fetch_result.get("html", ""))

                # 5. Extract restaurant information
                extracted_info = await self._extract_restaurant_info(
                    processed_content, 
                    fetch_result.get("restaurant_list")
                )

                # 6. Format source info
                source_info = self._extract_source_info(
                    url, 
                    fetch_result.get("title", ""), 
                    source_domain,
                    result.get("favicon", "")
                )

                # 7. Create enriched result
                enriched_result = {
                    **result,
                    "scraped_title": fetch_result.get("title", ""),
                    "scraped_content": processed_content,
                    "extracted_restaurants": extracted_info,
                    "content_length": len(processed_content),
                    "source_reputation": source_validation.get("reputation_score", 0.5),
                    "quality_score": evaluation.get("content_quality", 0.0),
                    "restaurant_count": evaluation.get("restaurant_count", len(extracted_info)),
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

    # Enhanced Fetch with Playwright
    async def _fetch_with_playwright(self, url: str, browser) -> Dict[str, Any]:
        """Fetch a URL using Playwright with an existing browser instance"""
        logger.info(f"Fetching URL with Playwright: {url}")
        result = {
            "url": url,
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "error": None,
            "content_length": 0,
            "likely_restaurant_list": False,
            "restaurant_list": []
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

            # Set up page
            page = await context.new_page()
            page.set_default_timeout(self.browser_config['timeout'])

            # Navigation options
            navigation_options = {
                "wait_until": "domcontentloaded",
                "timeout": self.browser_config['timeout'],
                "referer": "https://www.google.com/",
            }

            # Navigate to URL
            response = await page.goto(url, **navigation_options)
            result["status_code"] = response.status if response else 0

            # Wait for network idle
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeoutError:
                # Continue anyway
                pass

            # Handle "Load More" buttons to expand content
            try:
                load_more_clicks = await self._handle_load_more_buttons(page)
                if load_more_clicks > 0:
                    logger.info(f"Clicked 'Load More' {load_more_clicks} times")
                    # Wait for final content to settle
                    await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception as e:
                logger.warning(f"Error handling load more buttons: {e}")

            # Get page content
            result["html"] = await page.content()
            result["title"] = await page.title()

            # Extract structured restaurant list
            try:
                restaurant_list = await self._extract_restaurant_list(page)
                if restaurant_list and len(restaurant_list) > 2:
                    # Format as a clear list
                    preview_parts = ["RESTAURANT LIST:"]
                    for i, restaurant in enumerate(restaurant_list[:10], 1):
                        preview_parts.append(f"{i}. [{restaurant['name']}]")
                        if restaurant.get('description'):
                            preview_parts.append(f"   {restaurant['description'][:100]}...")

                    result["content_preview"] = "\n".join(preview_parts)
                    result["likely_restaurant_list"] = True
                    result["restaurant_list"] = restaurant_list
                else:
                    # Fall back to existing content extraction
                    content_text = await page.evaluate('''() => {
                        // Get visible text only
                        function getVisibleText(node) {
                            if (node.nodeType === Node.TEXT_NODE) {
                                return node.textContent || "";
                            }

                            // Skip invisible elements
                            try {
                                const style = window.getComputedStyle(node);
                                if (style && (style.display === "none" || style.visibility === "hidden" || style.opacity === "0")) {
                                    return "";
                                }
                            } catch (e) {
                                // Ignore errors with getComputedStyle
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

                        // Get main content
                        const mainContent = document.querySelector('main') || 
                                          document.querySelector('article') || 
                                          document.querySelector('.content') || 
                                          document.body;

                        return getVisibleText(mainContent).trim().substring(0, 4000);
                    }''')
                    result["content_preview"] = content_text
            except Exception as e:
                logger.warning(f"Error with restaurant list extraction: {e}")
                # Final fallback extraction
                content_text = await page.evaluate('document.body.innerText')
                result["content_preview"] = content_text[:4000] if content_text else ""

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

    # Enhanced HTTP fetching
    async def _fetch_with_http(self, url: str) -> Dict[str, Any]:
        """Fetch a URL using HTTP libraries"""
        logger.info(f"Fetching URL with HTTP: {url}")
        result = {
            "url": url,
            "status_code": None,
            "html": "",
            "title": "",
            "content_preview": "",
            "error": None,
            "content_length": 0,
            "likely_restaurant_list": False,
            "restaurant_list": []
        }

        try:
            # Use httpx for async HTTP
            headers = {
                "User-Agent": self.browser_config['user_agent'],
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=15)
                result["status_code"] = response.status_code

                if response.status_code == 200:
                    html = response.text
                    soup = BeautifulSoup(html, 'html.parser')

                    result["html"] = html
                    result["title"] = soup.title.text if soup.title else ""

                    # IMPROVED: Extract content preview focused on restaurant content
                    # First look for restaurant entries specifically
                    restaurant_entries = []

                    # Look for numbered entries (like "1. Restaurant Name")
                    numbered_headings = soup.find_all(["h2", "h3", "h4"], 
                                                      string=lambda s: s and re.match(r'^\d+\.|\d+\s+', s))
                    restaurant_entries.extend(numbered_headings)

                    # Look for restaurant-related headings
                    restaurant_headings = soup.find_all(["h2", "h3", "h4"], 
                                                     string=lambda s: s and re.search(r'restaurant|café|cafe|dining|bistro|eatery', s, re.I))
                    restaurant_entries.extend(restaurant_headings)

                    # Look for elements with restaurant-related classes/IDs
                    restaurant_classes = soup.select('.restaurant, .cafe, .listing, .place, [id*="restaurant"], [id*="place"], [class*="listing-item"]')
                    restaurant_entries.extend(restaurant_classes)

                    # Try to find a list container that likely contains restaurant entries
                    list_containers = soup.select('div > ul, div > ol, .list, .listings, [class*="list"]')
                    for container in list_containers:
                        # Check if this list looks like a restaurant list
                        list_items = container.find_all('li')
                        restaurant_like_items = [li for li in list_items if 
                                                re.search(r'restaurant|café|cafe|bistro|address|\d+\s+[a-zA-Z]+\s+st|\$|€', 
                                                          li.get_text().lower())]
                        if len(restaurant_like_items) >= 2:
                            restaurant_entries.append(container)

                    # If we found restaurant entries, use them for the preview
                    preview_text = ""
                    if restaurant_entries:
                        # Take up to 5 restaurant entries
                        for i, entry in enumerate(restaurant_entries[:5]):
                            # Get the parent element to include context
                            container = entry.parent or entry

                            # For headings, include content until next heading of same level
                            if entry.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                                heading_level = entry.name
                                content = f"[{entry.get_text()}]\n"

                                # Get content until next heading of same level
                                next_elem = entry.find_next()
                                while next_elem and next_elem.name != heading_level:
                                    if next_elem.name not in ["script", "style", "meta"]:
                                        content += next_elem.get_text() + "\n"
                                    next_elem = next_elem.find_next()

                                preview_text += content + "\n----------\n\n"
                            else:
                                # For other elements, just get their text
                                preview_text += container.get_text(separator=' ', strip=True) + "\n\n----------\n\n"

                        result["content_preview"] = preview_text[:4000]
                        result["likely_restaurant_list"] = True
                    else:
                        # Fallback to main content
                        main_content = soup.find('main') or soup.find('article') or soup.find(class_='content') or soup.body
                        if main_content:
                            preview_text = main_content.get_text(separator=' ', strip=True)
                            result["content_preview"] = preview_text[:4000]
                        else:
                            result["content_preview"] = soup.get_text(separator=' ', strip=True)[:4000]

                    # Check if the preview text contains restaurant list indicators
                    if re.search(r'\b(\d+\.|No\.\s*\d+|Top\s*\d+)\s+', result["content_preview"]) or \
                       re.search(r'(restaurant|café|bistro|eatery|dining).*?(address|located|price|\$|€)', 
                                 result["content_preview"], re.IGNORECASE):
                        result["likely_restaurant_list"] = True

                    result["content_length"] = len(html)
                else:
                    result["error"] = f"HTTP error: {response.status_code}"
        except Exception as e:
            result["error"] = f"Error fetching URL: {str(e)}"
            logger.warning(f"Error fetching URL: {url}, Error: {str(e)}")

        return result

    # Fixed Evaluate content quality
    async def _evaluate_content(self, url: str, title: str, preview: str, fetch_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate if the content is a restaurant list using AI"""
        try:
            # If we've already identified this as a restaurant list in a previous check or if the fetch_result indicates it's likely a restaurant list, skip the evaluation
            if fetch_result and fetch_result.get("likely_restaurant_list"):
                # If the preview doesn't have enough restaurant content, 
                # but we know it's a restaurant list, assign good scores
                logger.info(f"URL previously identified as restaurant list: {url}")
                return {
                    "is_restaurant_list": True,
                    "restaurant_count": len(fetch_result.get("restaurant_list", [])) or 5,  # Use actual count or conservative estimate
                    "content_quality": 0.8,
                    "reasoning": "Pre-identified as restaurant list based on content patterns and/or structured list extraction"
                }

            # Basic keyword check
            restaurant_keywords = ["restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish", "café", "cafe", "bistro"]
            if not any(kw in title.lower() or kw in preview.lower() for kw in restaurant_keywords):
                logger.info(f"URL filtered by basic keyword check: {url}")
                return {
                    "is_restaurant_list": False,
                    "restaurant_count": 0,
                    "content_quality": 0.0,
                    "reasoning": "No restaurant-related keywords found"
                }

            # Check for list format indicators - this is a fast pre-check before using the LLM
            list_indicators = [
                r'\b\d+\.\s+\w+',  # numbered items like "1. Restaurant"
                r'best\s+\d+',      # "best 10" or similar
                r'top\s+\d+',       # "top 10" or similar
                r'\b\d+\s+best',    # "10 best" or similar
            ]

            list_pattern = re.compile('|'.join(list_indicators), re.IGNORECASE)
            if list_pattern.search(title) or list_pattern.search(preview[:500]):
                # Strong indication this is a restaurant list - skip LLM evaluation
                        logger.info(f"URL identified as restaurant list by pattern matching: {url}")
                        return {
                            "is_restaurant_list": True,
                            "restaurant_count": 10,  # Reasonable estimate for list articles
                            "content_quality": 0.9,
                            "reasoning": "Identified as a curated list article by pattern matching"
                        }

                    # Using LLM for evaluation
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

                    import json
                    evaluation = json.loads(content.strip())

                    # Ensure content_quality is in response
                    if "content_quality" not in evaluation:
                        evaluation["content_quality"] = 0.8 if evaluation.get("is_restaurant_list", False) else 0.2

                    # Log evaluation results
                    logger.info(f"Content evaluation for {url}:")
                    logger.info(f"  - Is Restaurant List: {evaluation.get('is_restaurant_list')}")
                    logger.info(f"  - Quality Score: {evaluation.get('content_quality')}")
                    logger.info(f"  - Reasoning: {evaluation.get('reasoning')}")

                    return evaluation
                except Exception as e:
                    logger.error(f"Error evaluating content for {url}: {str(e)}")
                    # Default to invalid content
                    return {
                        "is_restaurant_list": False,
                        "restaurant_count": 0,
                        "content_quality": 0.0,
                        "reasoning": f"Error in evaluation: {str(e)}"
                    }

            # Process content
            async def _process_content(self, html: str) -> str:
                """Process HTML content using readability and cleaning"""
                if not html:
                    return ""

                try:
                    # Extract main content with readability
                    doc = Document(html)
                    readable_html = doc.summary()
                    title = doc.title()

                    # Parse with BeautifulSoup for cleaning
                    soup = BeautifulSoup(readable_html, 'html.parser')

                    # Extract text and preserve structure
                    paragraphs = []

                    # Process headings
                    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        paragraphs.append(f"HEADING: {heading.get_text().strip()}")
                        heading.extract()

                    # Process lists
                    for list_elem in soup.find_all(['ul', 'ol']):
                        list_items = []
                        for item in list_elem.find_all('li'):
                            item_text = item.get_text().strip()
                            if item_text:
                                list_items.append(f"- {item_text}")

                        if list_items:
                            paragraphs.append("\n".join(list_items))
                        list_elem.extract()

                    # Process regular paragraphs
                    for para in soup.find_all('p'):
                        text = para.get_text().strip()
                        if text:
                            paragraphs.append(text)

                    # Get remaining text
                    remaining_text = soup.get_text().strip()
                    if remaining_text:
                        lines = [line.strip() for line in remaining_text.split('\n') if line.strip()]
                        paragraphs.extend(lines)

                    # Add title at beginning
                    if title:
                        paragraphs.insert(0, f"TITLE: {title}")

                    # Join all paragraphs
                    processed_text = "\n\n".join(paragraphs)

                    # Clean up whitespace
                    processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
                    processed_text = re.sub(r' {2,}', ' ', processed_text)

                    return processed_text
                except Exception as e:
                    logger.error(f"Error processing content: {str(e)}")
                    # Fallback extraction
                    soup = BeautifulSoup(html, 'html.parser')
                    return soup.get_text(separator='\n\n')

            # Enhanced restaurant extraction
            async def _extract_restaurant_info(self, content: str, restaurant_list=None) -> List[Dict[str, Any]]:
                """Extract restaurant information from content using AI"""
                try:
                    # Use the structured list if available
                    if restaurant_list:
                        restaurants = []
                        for item in restaurant_list:
                            restaurant = {
                                "name": f"[{item['name']}]",  # Add square brackets for visibility
                                "description": item.get('description', ''),
                                "url": item.get('url', ''),
                                "type": item.get('type', 'restaurant')
                            }
                            restaurants.append(restaurant)
                        return restaurants

                    # Fall back to LLM extraction for unstructured content
                    response = await self.extract_chain.ainvoke({"content": content[:4000]})
                    extracted_text = response.content

                    # Process the extraction into structured data
                    restaurants = []
                    current_restaurant = None

                    for line in extracted_text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue

                        # Check if this is a new restaurant name
                        if line.startswith("Restaurant Name:") or line.startswith("Name:"):
                            # Save previous restaurant if exists
                            if current_restaurant and current_restaurant.get("name"):
                                restaurants.append(current_restaurant)

                            # Start new restaurant with square brackets
                            name = line.split(":", 1)[1].strip()
                            current_restaurant = {"name": f"[{name}]"}
                        elif current_restaurant is not None:
                            # Process other fields
                            if ":" in line:
                                key, value = line.split(":", 1)
                                key = key.strip().lower()
                                value = value.strip()

                                # Map to our standard fields
                                if "address" in key:
                                    current_restaurant["address"] = value
                                elif "price" in key:
                                    current_restaurant["price_range"] = value
                                elif "dish" in key or "recommend" in key or "signature" in key:
                                    current_restaurant.setdefault("recommended_dishes", []).append(value)
                                elif "chef" in key:
                                    current_restaurant["chef"] = value
                                elif "atmosphere" in key or "ambience" in key:
                                    current_restaurant["atmosphere"] = value
                                elif "reserv" in key:
                                    current_restaurant["reservations_required"] = "required" in value.lower() or "recommended" in value.lower()
                                elif "instagram" in key:
                                    current_restaurant["instagram"] = value
                                elif "description" in key:
                                    current_restaurant["description"] = value

                    # Add the last restaurant if exists
                    if current_restaurant and current_restaurant.get("name"):
                        restaurants.append(current_restaurant)

                    return restaurants
                except Exception as e:
                    logger.error(f"Error extracting restaurant info: {str(e)}")
                    return []

            # Extract source information
            def _extract_source_info(self, url: str, title: str, domain: str, favicon: str = "") -> Dict[str, Any]:
                """Extract and format source information"""
                source_type = "Website"

                # Determine source type
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

                # If title contains source name in better format, use that
                if title and len(title) > 3:
                    title_parts = title.split('|')
                    if len(title_parts) > 1:
                        possible_name = title_parts[-1].strip()
                        if 3 < len(possible_name) < 25:
                            source_name = possible_name

                return {
                    "name": source_name,
                    "domain": domain,
                    "type": source_type,
                    "favicon": favicon
                }

            # Extract domain
            def _extract_domain(self, url: str) -> str:
                """Extract domain from URL"""
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    return domain
                except Exception:
                    return url

            # Legacy API compatibility
            async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Legacy API method for backward compatibility"""
                logger.info("Using legacy API scrape_search_results")
                return await self.filter_and_scrape_results(search_results)

            # Public method for testing
            async def fetch_url(self, url: str) -> Dict[str, Any]:
                """Fetch a URL for testing purposes"""
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

