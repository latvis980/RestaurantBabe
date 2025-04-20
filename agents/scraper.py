import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import random
from langchain_core.tracers.context import tracing_v2_enabled

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
        enriched_results = []

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            for result in search_results:
                url = result.get("url")
                if not url:
                    continue

                # Skip non-HTML content
                if self._should_skip_url(url):
                    enriched_results.append(result)
                    continue

                try:
                    # Get the HTML content
                    html_content = self._get_html(url, max_retries)
                    if not html_content:
                        enriched_results.append(result)
                        continue

                    # Parse the HTML
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Extract main content
                    main_content = self._extract_main_content(soup)

                    # Extract restaurant information
                    restaurant_info = self._extract_restaurant_info(soup, url)

                    # Add the scraped content to the result
                    result["scraped_content"] = main_content
                    result["restaurant_info"] = restaurant_info
                    enriched_results.append(result)

                    # Be nice to servers
                    time.sleep(random.uniform(1.0, 2.0))

                except Exception as e:
                    print(f"Error scraping {url}: {e}")
                    # Add the original result without scraping
                    enriched_results.append(result)

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

    def _extract_main_content(self, soup):
        """Extract the main content from HTML"""
        # Try to find main content containers
        main_tags = ['article', 'main', '.content', '#content', '.post', '.article']

        for tag in main_tags:
            main_content = soup.select(tag)
            if main_content:
                # Join all text from these elements
                return ' '.join([elem.get_text(strip=True, separator=' ') for elem in main_content])

        # Fallback: get text from paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            return ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Last resort: get all text
        return soup.get_text(strip=True, separator=' ')

    def _extract_restaurant_info(self, soup, url):
        """Extract specific restaurant information from HTML"""
        restaurant_info = {}

        # Try to find restaurant name
        name_elements = soup.select('h1')
        if name_elements:
            restaurant_info['name'] = name_elements[0].get_text(strip=True)

        # Try to find address
        address_patterns = [
            'address',
            'location',
            '.address',
            '[itemprop="address"]',
            '.location-info'
        ]

        for pattern in address_patterns:
            address_elements = soup.select(pattern)
            if address_elements:
                restaurant_info['address'] = address_elements[0].get_text(strip=True)
                break

        # Try to find contact info (phones, etc.)
        contact_patterns = [
            '[itemprop="telephone"]',
            '.phone',
            '.tel'
        ]

        for pattern in contact_patterns:
            contact_elements = soup.select(pattern)
            if contact_elements:
                restaurant_info['contact'] = contact_elements[0].get_text(strip=True)
                break

        # Try to extract Instagram handle
        instagram_links = soup.select('a[href*="instagram.com"]')
        if instagram_links:
            instagram_url = instagram_links[0].get('href')
            if instagram_url:
                parts = instagram_url.split('instagram.com/')
                if len(parts) > 1:
                    handle = parts[1].split('?')[0].split('/')[0]
                    restaurant_info['instagram'] = f"instagram.com/{handle}"

        # Try to find opening hours
        hours_patterns = [
            '[itemprop="openingHours"]',
            '.hours',
            '.opening-hours',
            '.business-hours'
        ]

        for pattern in hours_patterns:
            hours_elements = soup.select(pattern)
            if hours_elements:
                restaurant_info['hours'] = hours_elements[0].get_text(strip=True)
                break

        # Domain of the source
        domain = urlparse(url).netloc
        restaurant_info['source_domain'] = domain

        return restaurant_info