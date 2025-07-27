# scrape_test.py - Updated for new pipeline without list_analyzer
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

                # Step 2: Search
                f.write("STEP 2: WEB SEARCH\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                search_results = self.search_agent.search(query_analysis.get('search_queries', []))
                search_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {search_time}s\n")
                f.write(f"Total URLs Found: {len(search_results)}\n")

                # Log all URLs for analysis
                f.write("\nSearch Results (First 10):\n")
                for i, result in enumerate(search_results[:10], 1):
                    f.write(f"  {i}. {result.get('url', 'Unknown URL')}\n")
                    f.write(f"     Title: {result.get('title', 'No title')[:100]}...\n")

                if len(search_results) > 10:
                    f.write(f"  ... and {len(search_results) - 10} more results\n")

                f.write("\n")

                # Step 3: Scraping (The main focus)
                f.write("STEP 3: INTELLIGENT SCRAPING\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()

                # Use the orchestrator's scraper which has all the intelligent logic
                scraped_results = await self.scraper.scrape_search_results(search_results)

                scraping_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {scraping_time}s\n")
                f.write(f"Successfully Scraped: {len(scraped_results)}\n")

                # Analyze scraping results in detail
                total_content_length = 0
                successful_scrapes = 0
                failed_scrapes = 0

                f.write("\nScraping Analysis:\n")
                for i, result in enumerate(scraped_results, 1):
                    url = result.get('url', 'Unknown URL')
                    content = result.get('content', '')

                    if content and len(content.strip()) > 100:
                        successful_scrapes += 1
                        content_length = len(content)
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
                            f.write(f"  {i}. {restaurant.get('name', 'Unknown')}\n")
                            f.write(f"     Address: {restaurant.get('address', 'Unknown')}\n")
                            f.write(f"     Description: {restaurant.get('description', 'No description')[:100]}...\n")
                            f.write(f"     Sources: {restaurant.get('sources', [])}\n")

                        if len(main_list) > 5:
                            f.write(f"  ... and {len(main_list) - 5} more restaurants\n")

                    if follow_up_queries:
                        f.write(f"\nFollow-up Queries:\n")
                        for i, query in enumerate(follow_up_queries, 1):
                            f.write(f"  {i}. {query}\n")

                except Exception as e:
                    f.write(f"❌ Editor processing failed: {str(e)}\n")
                    editing_time = 0

                # Now, show the FULL content of all articles
                f.write("\n" + "=" * 80 + "\n")
                f.write("FULL CONTENT OF ALL SCRAPED ARTICLES\n")
                f.write("=" * 80 + "\n\n")

                for i, article in enumerate(editor_input['scraped_results'], 1):
                    f.write(f"\nARTICLE {i}:\n")
                    f.write(f"  URL: {article.get('url', 'Unknown')}\n")
                    f.write(f"  Title: {article.get('title', 'No title')}\n")
                    f.write(f"  Content length: {len(article.get('content', ''))}\n")
                    f.write(f"  FULL CONTENT:\n")
                    f.write("-" * 40 + "\n")

                    content = article.get('content', '')
                    if content:
                        f.write(content)
                    else:
                        f.write("  [No content available]")

                    f.write("\n" + "-" * 40 + "\n")

                # Get scraper statistics
                scraper_stats = self.scraper.get_stats()
                f.write(f"\nIntelligent Scraper Statistics:\n")
                for key, value in scraper_stats.items():
                    f.write(f"  {key}: {value}\n")

                # Overall timing
                total_time = analysis_time + search_time + scraping_time + (editing_time if 'editing_time' in locals() else 0)
                f.write(f"\nOVERALL TIMING:\n")
                f.write(f"  Query Analysis: {analysis_time}s\n")
                f.write(f"  Web Search: {search_time}s\n")
                f.write(f"  Intelligent Scraping: {scraping_time}s\n")
                if 'editing_time' in locals():
                    f.write(f"  Editor Processing: {editing_time}s\n")
                f.write(f"  Total: {total_time}s\n\n")

                f.write("=" * 80 + "\n")
                f.write("TEST COMPLETED SUCCESSFULLY\n")
                f.write(f"Pipeline: Query → Search → Scrape → Editor → Follow-up → Format\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during scraping test: {str(e)}\n")
                logger.error(f"Error during scraping test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, successful_scrapes if 'successful_scrapes' in locals() else 0)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, successful_count: int):
        """Send scraping test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🧪 <b>Scraping Process Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"✅ Successful scrapes: {successful_count}\n"
                f"🔧 Pipeline: scrape → editor → follow_up → format\n"
                f"🎯 Focus: Complete pipeline analysis (no list_analyzer)\n\n"
                f"{'✅ Content extracted successfully' if successful_count > 0 else '❌ No content extracted'}\n\n"
                f"📄 Detailed scraping analysis attached with FULL content of all articles."
            )

            bot.send_message(
                self.admin_chat_id,
                summary,
                parse_mode='HTML'
            )

            # Send the results file
            with open(file_path, 'rb') as f:
                bot.send_document(
                    self.admin_chat_id,
                    f,
                    caption=f"🧪 Scraping test results for: {query} (includes FULL content)"
                )

            logger.info("Successfully sent scraping test results to admin")

        except Exception as e:
            logger.error(f"Failed to send results to admin: {e}")


# Convenience function for backward compatibility
def add_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot
    This function is now deprecated since commands are handled directly in telegram_bot.py
    Keeping for backward compatibility.
    """
    logger.info("Note: scrape test commands are now handled directly in telegram_bot.py")
    pass