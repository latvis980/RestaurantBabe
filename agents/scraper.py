# agents/scraper.py - Hybrid Scraper with both HTTP and Playwright support
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
        - Wanderlog content
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
                extracted_info = await self._extract_restaurant_info(processed_content)

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

    # Fetch with Playwright
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
            "likely_restaurant_list": False
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

            # Get page content
            result["html"] = await page.content()
            result["title"] = await page.title()

            # Get content preview using JavaScript with improved restaurant detection
            try:
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

                    // Helper to check if a node likely contains a restaurant entry
                    function isLikelyRestaurantEntry(node) {
                        const text = node.textContent.toLowerCase();
                        const hasRestaurantKeywords = /restaurant|café|cafe|bistro|eatery|dining|bar|bakery|patisserie|brasserie/i.test(text);
                        const hasAddressPattern = /\\d+\\s+[a-zA-Z\\s]+\\s+(street|st|avenue|ave|boulevard|blvd|road|rd)/i.test(text);
                        const hasPriceInfo = /(\\$|€|£|¥)(\\$|€|£|¥)?(\\$|€|£|¥)?|\\s+\\d+\\s*(\\$|€|£|¥)/.test(text);

                        return hasRestaurantKeywords || hasAddressPattern || hasPriceInfo;
                    }

                    // Find the most likely parent container of restaurant listings
                    function findRestaurantListContainer() {
                        // Common container selectors for listings
                        const possibleContainers = [
                            ...document.querySelectorAll('main, .content, article, .article, .list, .listings, .places, .restaurants, [class*="list"], [class*="grid"]'),
                            ...document.querySelectorAll('div > ul, div > ol, section')
                        ];

                        // Score each container by how likely it is to contain restaurant listings
                        const containerScores = possibleContainers.map(container => {
                            let score = 0;

                            // Score based on children that look like restaurant entries
                            const children = Array.from(container.children);
                            const restaurantLikeChildren = children.filter(isLikelyRestaurantEntry);
                            score += restaurantLikeChildren.length * 5;

                            // Score based on heading patterns
                            const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
                            const restaurantLikeHeadings = Array.from(headings).filter(h => 
                                /^\\d+\\.|\\d+\\s+|top\\s+\\d+|best\\s+\\d+|restaurant|café|cafe|dining/i.test(h.textContent)
                            );
                            score += restaurantLikeHeadings.length * 10;

                            // Score based on numbered items
                            const numberedItems = container.querySelectorAll('ol > li, [class*="number"], [class*="item-"]');
                            score += numberedItems.length * 2;

                            // Score based on images (restaurants often have photos)
                            const images = container.querySelectorAll('img');
                            score += images.length * 0.5;

                            // Bonus for specific classes/IDs
                            if (/restaurant|cafe|listing|place|venue|spot|destination/i.test(container.className + container.id)) {
                                score += 20;
                            }

                            return { container, score };
                        });

                        // Sort by score descending
                        containerScores.sort((a, b) => b.score - a.score);

                        // Return the highest scoring container, if it has a reasonable score
                        return containerScores.length > 0 && containerScores[0].score > 5 
                               ? containerScores[0].container 
                               : null;
                    }

                    // Find potential restaurant entries
                    const restaurantContainer = findRestaurantListContainer();

                    if (restaurantContainer) {
                        // For a list container, extract individual entries
                        const listItems = restaurantContainer.querySelectorAll('li, article, .item, [class*="card"], [class*="list-item"]');
                        if (listItems && listItems.length > 0) {
                            // Get text from first few items
                            let entriesText = "";
                            const maxEntries = Math.min(5, listItems.length);

                            for (let i = 0; i < maxEntries; i++) {
                                entriesText += getVisibleText(listItems[i]) + "\\n\\n----------\\n\\n";
                            }

                            return entriesText.trim();
                        }

                        // If no distinct list items, get text from the container
                        return getVisibleText(restaurantContainer).substring(0, 4000);
                    }

                    // Alternative approach: find numbered headings (like "1. Restaurant Name")
                    const numberedHeadings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                        .filter(el => /^\\d+\\.|\\d+\\s+/.test(el.textContent.trim()))
                        .slice(0, 5);

                    if (numberedHeadings.length > 0) {
                        let headingsText = "";

                        for (const heading of numberedHeadings) {
                            // Get heading plus content until next heading
                            let content = heading.textContent + "\\n";
                            let nextEl = heading.nextElementSibling;

                            while (nextEl && nextEl.tagName.toLowerCase() !== heading.tagName.toLowerCase()) {
                                content += getVisibleText(nextEl) + "\\n";
                                nextEl = nextEl.nextElementSibling;
                                if (!nextEl) break;
                            }

                            headingsText += content + "\\n----------\\n\\n";
                        }

                        return headingsText.trim();
                    }

                    // Fallback: scan document for any section with restaurant-like content
                    const allParagraphs = document.querySelectorAll('p');
                    let restaurantParagraphs = "";
                    let count = 0;

                    for (const para of allParagraphs) {
                        if (isLikelyRestaurantEntry(para)) {
                            restaurantParagraphs += getVisibleText(para) + "\\n\\n";
                            count++;
                            if (count >= 5) break;
                        }
                    }

                    if (restaurantParagraphs) {
                        return restaurantParagraphs;
                    }

                    // Final fallback: just get main content
                    const mainContent = document.querySelector('main') || 
                                      document.querySelector('article') || 
                                      document.querySelector('.content') || 
                                      document.body;

                    return getVisibleText(mainContent).trim().substring(0, 4000);
                }''')
                result["content_preview"] = content_text

                # Set likely_restaurant_list flag if preview contains restaurant indicators
                if re.search(r'\b(\d+\.|No\.\s*\d+|Top\s*\d+)\s+', content_text) or \
                   re.search(r'(restaurant|café|bistro|eatery|dining).*?(address|located|price|\$|€)', content_text, re.IGNORECASE):
                    result["likely_restaurant_list"] = True

            except Exception as e:
                logger.warning(f"Error extracting content preview with JS: {e}")
                # Fallback extraction
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

    # Fetch with HTTP
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
            "likely_restaurant_list": False
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
                                content = entry.get_text() + "\n"

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

    # Evaluate content quality
    async def _evaluate_content(self, url: str, title: str, preview: str, fetch_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate if the content is a restaurant list using AI"""
        try:
            # If we've already identified this as a restaurant list in a previous check
            # or if the fetch_result indicates it's likely a restaurant list, skip the evaluation
            if fetch_result and fetch_result.get("likely_restaurant_list"):
                # If the preview doesn't have enough restaurant content, 
                # but we know it's a restaurant list, assign good scores
                logger.info(f"URL previously identified as restaurant list: {url}")
                return {
                    "is_restaurant_list": True,
                    "restaurant_count": 5,  # Conservative estimate
                    "content_quality": 0.8,
                    "reasoning": "Pre-identified as restaurant list based on content patterns"
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

    # Extract restaurant information
    async def _extract_restaurant_info(self, content: str) -> List[Dict[str, Any]]:
        """Extract restaurant information from content using AI"""
        try:
            # Use LLM to extract structured information
            response = await self.extract_chain.ainvoke({"content": content[:4000]})
            extracted_text = response.content

            # Process the extraction into structured data
            import json

            # Simple parsing to extract restaurant blocks
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

                    # Start new restaurant
                    current_restaurant = {"name": line.split(":", 1)[1].strip()}
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