# scrape_test.py - Fixed for proper BraveSearchAgent.search() method call
import asyncio
import time
import tempfile
import os
import threading
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ScrapeTest:
    """
    Simple test to see the complete scraping process:
    - What search results are found
    - Which ones get scraped successfully  
    - What content is actually scraped (showing FULL content)
    - What goes to editor_agent (not list_analyzer anymore)

    Updated for new pipeline: scrape → editor → follow_up → format
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator  # Receives the singleton instance
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize pipeline components - get from the orchestrator to ensure consistency
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper
        self.editor_agent = orchestrator.editor_agent  # Changed from list_analyzer to editor_agent

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run complete scraping process and dump results to file
        Shows FULL content of all scraped articles and what goes to editor

        Args:
            restaurant_query: The restaurant query to test (e.g., "best brunch in Lisbon")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing scraping process for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESTAURANT SCRAPING PROCESS TEST - NEW PIPELINE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Pipeline: scrape → editor → follow_up → format\n")
            f.write(f"Orchestrator: Singleton instance\n\n")

            try:
                # Step 1: Analyze Query
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Search Queries Generated: {len(query_analysis.get('search_queries', []))}\n")

                for i, query in enumerate(query_analysis.get('search_queries', []), 1):
                    f.write(f"  {i}. {query}\n")

                f.write(f"\nDestination: {query_analysis.get('destination', 'Unknown')}\n")
                f.write(f"Primary Parameters: {query_analysis.get('primary_search_parameters', [])}\n")
                f.write(f"Secondary Parameters: {query_analysis.get('secondary_filter_parameters', [])}\n\n")

                # Step 2: Search - FIXED TO USE CORRECT METHOD SIGNATURE
                f.write("STEP 2: WEB SEARCH\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                search_queries = query_analysis.get('search_queries', [])

                # FIXED: Call search method with correct parameters
                # The method signature is: search(queries, max_retries=3, retry_delay=2, enable_ai_filtering=True)
                search_results = self.search_agent.search(
                    search_queries,  # queries parameter
                    enable_ai_filtering=True  # explicitly pass enable_ai_filtering
                )
                search_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {search_time}s\n")
                f.write(f"Total URLs Found: {len(search_results)}\n")

                # Log all URLs for analysis
                f.write("\nSearch Results (First 10):\n")
                for i, result in enumerate(search_results[:10], 1):
                    f.write(f"  {i}. {result.get('title', 'No Title')}\n")
                    f.write(f"     URL: {result.get('url', 'No URL')}\n")
                    f.write(f"     Description: {result.get('description', 'No Description')[:100]}...\n")

                    # Show AI evaluation results if available
                    ai_eval = result.get('ai_evaluation', {})
                    if ai_eval:
                        f.write(f"     AI Score: {ai_eval.get('content_quality', 0):.2f}\n")
                        f.write(f"     AI Passed: {ai_eval.get('passed_filter', False)}\n")
                    f.write("\n")

                if not search_results:
                    f.write("❌ No search results found. Ending test.\n")
                    return filepath

                # Step 3: Scraping
                f.write("STEP 3: SCRAPING PROCESS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                scraped_results = await self.scraper.scrape_search_results(search_results)
                scraping_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {scraping_time}s\n")
                f.write(f"URLs sent to scraper: {len(search_results)}\n")
                f.write(f"Articles successfully scraped: {len(scraped_results)}\n\n")

                # Analyze scraping results
                successful_scrapes = 0
                failed_scrapes = 0
                total_content_length = 0

                f.write("SCRAPING RESULTS ANALYSIS:\n")
                for i, result in enumerate(scraped_results, 1):
                    url = result.get("url", "Unknown URL")
                    content = result.get("content", "")
                    content_length = len(content)

                    if content and content_length > 100:  # Minimum content threshold
                        successful_scrapes += 1
                        total_content_length += content_length
                        f.write(f"  {i}. ✅ {url}\n")
                        f.write(f"     Content Length: {content_length:,} chars\n")
                    else:
                        failed_scrapes += 1
                        f.write(f"  {i}. ❌ {url}\n")
                        f.write(f"     Status: Failed or insufficient content\n")

                f.write(f"\nScraping Summary:\n")
                f.write(f"  Successful: {successful_scrapes}\n")
                f.write(f"  Failed: {failed_scrapes}\n")
                f.write(f"  Total Content: {total_content_length:,} characters\n")
                f.write(f"  Average per successful scrape: {total_content_length // max(successful_scrapes, 1):,} chars\n")

                # Step 4: What goes to Editor Agent (instead of List Analyzer)
                f.write("\nSTEP 4: EDITOR AGENT INPUT\n")
                f.write("-" * 40 + "\n")

                # Prepare the exact input that would go to editor_agent
                editor_input = {
                    "scraped_results": scraped_results,  # Raw scraped content
                    "original_query": restaurant_query,
                    "destination": query_analysis.get("destination", "Unknown")
                }

                f.write(f"Articles sent to editor: {len(editor_input['scraped_results'])}\n")
                f.write(f"Original query: {editor_input['original_query']}\n")
                f.write(f"Destination: {editor_input['destination']}\n")
                f.write(f"Editor function: editor_agent.edit(scraped_results=..., original_query=..., destination=...)\n\n")

                # Step 5: Test Editor Processing (optional)
                f.write("STEP 5: EDITOR PROCESSING TEST\n")
                f.write("-" * 40 + "\n")

                try:
                    start_time = time.time()

                    # Call editor with scraped results
                    editor_output = self.editor_agent.edit(
                        scraped_results=scraped_results,
                        original_query=restaurant_query,
                        destination=query_analysis.get("destination", "Unknown")
                    )

                    editing_time = round(time.time() - start_time, 2)

                    f.write(f"Processing Time: {editing_time}s\n")

                    edited_results = editor_output.get("edited_results", {})
                    main_list = edited_results.get("main_list", [])
                    follow_up_queries = editor_output.get("follow_up_queries", [])

                    f.write(f"Restaurants extracted: {len(main_list)}\n")
                    f.write(f"Follow-up queries generated: {len(follow_up_queries)}\n")

                    # Show extracted restaurants
                    if main_list:
                        f.write("\nExtracted Restaurants:\n")
                        for i, restaurant in enumerate(main_list[:5], 1):  # Show first 5
                            f.write(f"  {i}. {restaurant.get('name', 'Unknown Name')}\n")
                            f.write(f"     Description: {restaurant.get('description', 'No description')[:100]}...\n")
                            f.write(f"     Source: {restaurant.get('source_url', 'Unknown source')}\n\n")

                    # Show follow-up queries
                    if follow_up_queries:
                        f.write("\nFollow-up Queries Generated:\n")
                        for i, query in enumerate(follow_up_queries, 1):
                            f.write(f"  {i}. {query}\n")

                except Exception as e:
                    f.write(f"❌ Error in editor processing: {str(e)}\n")

                # Final summary
                f.write("\n" + "=" * 80 + "\n")
                f.write("TEST SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Search queries generated: {len(query_analysis.get('search_queries', []))}\n")
                f.write(f"URLs found: {len(search_results)}\n")
                f.write(f"URLs scraped: {len(scraped_results)}\n")
                f.write(f"Successful scrapes: {successful_scrapes}\n")
                f.write(f"Total content: {total_content_length:,} characters\n")
                f.write(f"Restaurants extracted: {len(main_list) if 'main_list' in locals() else 'N/A'}\n")

            except Exception as e:
                f.write(f"\n❌ ERROR DURING TEST: {str(e)}\n")
                logger.error(f"Error during scraping test: {e}")

        # Send file to admin if bot is available
        if bot and self.admin_chat_id:
            try:
                with open(filepath, 'rb') as file:
                    bot.send_document(
                        self.admin_chat_id,
                        file,
                        caption=f"🧪 Scraping test results for: {restaurant_query}"
                    )
                logger.info("Successfully sent scraping test results to admin")
            except Exception as e:
                logger.error(f"Failed to send file to admin: {e}")

        return filepath