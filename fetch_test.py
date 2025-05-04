# fetch_test.py
import asyncio
import sys
import json
import config
from agents.scraper import WebScraper

async def test_fetch(url):
    """Test the URL fetching functionality"""
    print(f"Testing fetch for URL: {url}")

    # Initialize WebScraper
    scraper = WebScraper(config)

    # Fetch the URL
    try:
        result = await scraper.fetch_url(url)
        print(f"\nFetch Result:")
        print(f"URL: {result.get('url')}")

        if 'error' in result:
            print(f"Error: {result.get('error')}")
        else:
            print(f"Status Code: {result.get('status_code')}")
            print(f"Content Length: {result.get('content_length')} characters")
            print(f"Content Preview: {result.get('content_preview')[:200]}...")

        return result
    except Exception as e:
        print(f"Error during fetch test: {e}")
        return {"url": url, "error": str(e)}

async def test_scrape(url):
    """Test the full scraping functionality"""
    print(f"Testing scrape for URL: {url}")

    # Initialize WebScraper
    scraper = WebScraper(config)

    # Scrape the URL
    try:
        result = await scraper.scrape_url(url)
        print(f"\nScrape Result:")
        print(f"URL: {result.get('url')}")

        if result.get('error'):
            print(f"Error: {result.get('error')}")
        else:
            print(f"Status Code: {result.get('status_code')}")
            print(f"Quality Score: {result.get('quality_score')}")
            content_preview = result.get('scraped_content', '')[:200]
            print(f"Content Preview: {content_preview}...")
            print(f"Content Length: {len(result.get('scraped_content', ''))}")

        return result
    except Exception as e:
        print(f"Error during scrape test: {e}")
        return {"url": url, "error": str(e)}

def main():
    """Run the fetch and scrape tests"""
    if len(sys.argv) < 2:
        print("Usage: python fetch_test.py <url>")
        return

    url = sys.argv[1]

    # Run both tests
    asyncio.run(test_fetch(url))
    print("\n" + "-"*50 + "\n")
    asyncio.run(test_scrape(url))

if __name__ == "__main__":
    main()