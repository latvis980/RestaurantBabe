import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import random
from langchain_core.tracers.context import tracing_v2_enabled
import asyncio


class WebScraper:
    def __init__(self, config):
        self.config = config
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]

    def scrape_search_results(self, search_results, max_retries=2):
        """
        Scrape content from search results

        Args:
            search_results (list): List of search result dictionaries with URLs
            max_retries (int): Maximum number of retries for failed requests

        Returns:
            list: Enriched search results with scraped content
        """
        # Convert synchronous function call to async
        return asyncio.run(self.filter_and_scrape_results(search_results, max_retries))

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
            for result in search_results:
                url = result.get("url")
                if not url:
                    continue

                # First check database for known reputation
                is_reputable = check_source_reputation(url, self.config)

                # If not in database, evaluate with AI
                if is_reputable is None:
                    is_reputable = await evaluate_source_quality(url, self.config)

                # Only proceed with reputable sources
                if is_reputable:
                    # Skip non-HTML content
                    if self._should_skip_url(url):
                        continue

                    try:
                        # Get the HTML content
                        html_content = self._get_html(url, max_retries)
                        if not html_content:
                            continue

                        # Parse the HTML
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Extract main content as clean text
                        clean_text = self._extract_clean_text(soup)

                        # Add the scraped content to the result
                        result["scraped_content"] = clean_text
                        result["source_domain"] = urlparse(url).netloc
                        result["is_reputable"] = True  # Mark as reputable for downstream processing
                        enriched_results.append(result)

                        # Be nice to servers
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    except Exception as e:
                        print(f"Error scraping {url}: {e}")
                else:
                    print(f"Skipping non-reputable source: {url}")

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