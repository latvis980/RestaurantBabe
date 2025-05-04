# scraper_test.py
import sys
import json
import asyncio
import time
from pathlib import Path

# Import your existing components
import config
from agents.search_agent import BraveSearchAgent
from agents.scraper import WebScraper
from utils.debug_utils import dump_chain_state

async def test_scraper(query, output_file="scraped_results.json"):
    """
    Test the scraper with a search query and save the results to a file.

    Args:
        query (str): Search query to use
        output_file (str): Path to save the results
    """
    print(f"Testing scraper with query: {query}")

    # Initialize components
    search_agent = BraveSearchAgent(config)
    scraper = WebScraper(config)

    # Get search results
    print("Performing search...")
    search_queries = [query]
    search_results = search_agent.search(search_queries)
    print(f"Found {len(search_results)} search results")

    # Scrape the search results
    print("Scraping search results (this may take a while)...")
    start_time = time.time()
    enriched_results = await scraper.filter_and_scrape_results(search_results)
    elapsed = time.time() - start_time
    print(f"Scraping completed in {elapsed:.2f} seconds")

    # Save to file
    output_path = Path(output_file)

    # Create a simplified version for easier viewing
    simplified_results = []
    for result in enriched_results:
        # Create a copy without the HTML (too verbose)
        simplified = {k: v for k, v in result.items() if k != 'html'}

        # Truncate scraped_content for preview (full content saved in complete file)
        if 'scraped_content' in simplified:
            preview = simplified['scraped_content'][:500]
            simplified['scraped_content_preview'] = f"{preview}... (truncated, full in complete file)"

        simplified_results.append(simplified)

    # Save a complete version with all data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_results, f, ensure_ascii=False, indent=2)
    print(f"Complete results saved to {output_path}")

    # Save a simplified version for easier viewing
    simple_path = output_path.with_stem(f"{output_path.stem}_simple")
    with open(simple_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    print(f"Simplified results saved to {simple_path}")

    # Print some stats
    total_content_length = sum(len(r.get('scraped_content', '')) for r in enriched_results)
    avg_content_length = total_content_length / len(enriched_results) if enriched_results else 0

    print(f"\nScraper Stats:")
    print(f"- Total results: {len(enriched_results)}")
    print(f"- Total content scraped: {total_content_length} characters")
    print(f"- Average content per result: {avg_content_length:.1f} characters")
    print(f"- Results with quality score > 0.7: {sum(1 for r in enriched_results if r.get('quality_score', 0) > 0.7)}")

    return enriched_results

def main():
    """Run the scraper test with command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python scraper_test.py 'search query' [output_file]")
        return

    query = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "scraped_results.json"

    # Run the async test function
    asyncio.run(test_scraper(query, output_file))

if __name__ == "__main__":
    main()