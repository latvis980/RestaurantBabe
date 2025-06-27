# Create a file called test_scrapers.py in your project root
import asyncio
import json
from agents.scraper import WebScraper
from agents.enhanced_scraper import EnhancedWebScraper
import config

async def compare_scrapers():
    # Test URLs (replace with actual restaurant guide URLs)
    test_urls = [
        {"url": "https://example-restaurant-guide.com/best-restaurants-paris"}
    ]

    # Initialize both scrapers
    default_scraper = WebScraper(config)
    enhanced_scraper = EnhancedWebScraper(config)

    print("Testing Default Scraper...")
    default_results = await default_scraper.filter_and_scrape_results(test_urls)

    print("Testing Enhanced Scraper...")
    enhanced_results = await enhanced_scraper.filter_and_scrape_results(test_urls)

    # Compare results
    print(f"\nDefault Scraper: {len(default_results)} results")
    print(f"Enhanced Scraper: {len(enhanced_results)} results")

    # Save results for comparison
    with open("default_results.json", "w") as f:
        json.dump(default_results, f, indent=2)

    with open("enhanced_results.json", "w") as f:
        json.dump(enhanced_results, f, indent=2)

    print("Results saved to files for comparison")

if __name__ == "__main__":
    asyncio.run(compare_scrapers())