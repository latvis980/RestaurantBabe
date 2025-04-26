import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import random
from langchain_core.tracers.context import tracing_v2_enabled
import asyncio
from utils.async_utils import track_async_task, sync_to_async


class WebScraper:
    def __init__(self, config):
        self.config = config
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]

    # agents/scraper.py (modification only to scrape_search_results method)
    def scrape_search_results(self, search_results, max_retries=2):
        """
        Scrape content from search results

        Args:
            search_results (list): List of search result dictionaries with URLs
            max_retries (int): Maximum number of retries for failed requests

        Returns:
            list: Enriched search results with scraped content
        """
        # Create a new event loop if none exists
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop
            return asyncio.run(self.filter_and_scrape_results(search_results, max_retries))

        # If we have a running loop
        if loop.is_running():
            # We're in a running event loop, schedule the task but don't wait for it
            enriched_results = []
            # Create a shallow copy to work with immediately
            for result in search_results:
                enriched_results.append(result.copy())

            # Schedule the actual scraping to happen asynchronously
            task = loop.create_task(self.filter_and_scrape_results(search_results, max_retries))
            _PENDING_TASKS.add(task)
            task.add_done_callback(lambda t: _PENDING_TASKS.discard(t))

            print("Warning: Returning potentially incomplete results due to async execution")
            return enriched_results
        else:
            # We have a loop but it's not running
            return loop.run_until_complete(self.filter_and_scrape_results(search_results, max_retries))

    # Modified filter_and_scrape_results method for WebScraper

    async def filter_and_scrape_results(self, search_results, max_retries=2):
        """
        Filter search results by source quality and scrape only reputable sources

        Args:
            search_results (list): List of search result dictionaries with URLs
            max_retries (int): Maximum number of retries for failed requests

        Returns:
            list: Enriched search results from reputable sources with scraped content
        """
        from utils.source_validator import check_source_reputation, evaluate_source_quality
        enriched_results = []

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # First batch process all URLs to check reputation
            # This allows us to filter out non-reputable sources before scraping
            reputation_checks = []

            for i, result in enumerate(search_results):
                url = result.get("url")
                if not url:
                    continue

                # Skip non-HTML content immediately
                if self._should_skip_url(url):
                    continue

                # Check global reputation status
                reputation_status = check_source_reputation(url, self.config)

                # Store the result and URL for processing
                reputation_checks.append((i, url, reputation_status))

            # Log how many results we're processing
            print(f"Processing {len(reputation_checks)} search results")

            # Process URLs with known reputations first
            for i, url, is_reputable in reputation_checks:
                # If we already know it's not reputable, skip it
                if is_reputable is False:
                    print(f"Skipping known non-reputable source: {url}")
                    continue

                # If we know it's reputable, process it
                if is_reputable is True:
                    print(f"Processing known reputable source: {url}")
                    try:
                        result = search_results[i]
                        # Get the HTML content
                        html_content = self._get_html(url, max_retries)
                        if not html_content:
                            continue

                        # Parse the HTML
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Extract main content as clean text
                        clean_text = self._extract_clean_text(soup)

                        # Get source information
                        domain = urlparse(url).netloc
                        title = result.get("title", "Unknown Title")

                        # Add source information as prefix to content
                        source_prefix = f"SOURCE: {domain}\nTITLE: {title}\nURL: {url}\n\nCONTENT:\n"
                        content_with_source = source_prefix + clean_text

                        # Add the scraped content to the result
                        result["scraped_content"] = content_with_source
                        result["source_domain"] = domain
                        result["is_reputable"] = True  # Mark as reputable for downstream processing
                        enriched_results.append(result)

                        # Be nice to servers
                        await track_async_task(asyncio.sleep(random.uniform(1.0, 2.0)))
                    except Exception as e:
                        print(f"Error scraping {url}: {e}")

            # Now process URLs with unknown reputations
            unknown_reputation_count = sum(1 for _, _, status in reputation_checks if status is None)
            if unknown_reputation_count > 0:
                print(f"Checking reputation for {unknown_reputation_count} sources with unknown reputation")

            for i, url, is_reputable in reputation_checks:
                # Only process URLs with unknown reputation
                if is_reputable is not None:
                    continue

                # We need to evaluate reputation
                try:
                    is_reputable = await evaluate_source_quality(url, self.config)

                    # If reputable, process it
                    if is_reputable:
                        print(f"New reputable source found: {url}")
                        result = search_results[i]

                        # Get the HTML content
                        html_content = self._get_html(url, max_retries)
                        if not html_content:
                            continue

                        # Parse the HTML
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Extract main content as clean text
                        clean_text = self._extract_clean_text(soup)

                        # Get source information
                        domain = urlparse(url).netloc
                        title = result.get("title", "Unknown Title")

                        # Add source information as prefix to content
                        source_prefix = f"SOURCE: {domain}\nTITLE: {title}\nURL: {url}\n\nCONTENT:\n"
                        content_with_source = source_prefix + clean_text

                        # Add the scraped content to the result
                        result["scraped_content"] = content_with_source
                        result["source_domain"] = domain
                        result["is_reputable"] = True  # Mark as reputable for downstream processing
                        enriched_results.append(result)

                        # Be nice to servers
                        await track_async_task(asyncio.sleep(random.uniform(1.0, 2.0)))
                    else:
                        print(f"Skipping non-reputable source: {url}")
                except Exception as e:
                    print(f"Error evaluating/scraping {url}: {e}")

        return enriched_results

    def _should_skip_url(self, url):
        """Check if URL should be skipped (non-HTML content)"""
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.mp3', '.mp4']
        return any(url.lower().endswith(ext) for ext in skip_extensions)

    def _get_html(self, url, max_retries=2):
        """Get HTML content from a URL with retries"""
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }

        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    return response.text
            except Exception as e:
                print(f"Attempt {retry_count+1}/{max_retries+1} failed for {url}: {e}")

            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(random.uniform(1.0, 3.0))  # Exponential backoff

        return None

    def _extract_clean_text(self, soup):
        """Extract clean text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script.extract()

        # Get the main content if available
        main_tags = ['article', 'main', '.content', '#content', '.post', '.article']
        main_content = None

        for tag in main_tags:
            main_elements = soup.select(tag)
            if main_elements:
                main_content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in main_elements])
                break

        # If no main content found, try to get all paragraph text
        if not main_content:
            paragraphs = soup.find_all('p')
            if paragraphs:
                main_content = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # If still no content, get all text from body
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)
            else:
                main_content = soup.get_text(separator=' ', strip=True)

        # Clean up the text
        lines = (line.strip() for line in main_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        main_content = ' '.join(chunk for chunk in chunks if chunk)

        return main_content