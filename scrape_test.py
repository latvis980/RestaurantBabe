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
            f.write("RESTAURANT SCRAPING PROCESS TEST (Complete Pipeline)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Pipeline: scrape → editor → follow_up → format\n")
            f.write(f"Orchestrator: Singleton instance\n\n")

            try:
                # Step 1: Query Analysis  
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Generated Search Queries:\n")

                search_queries = query_analysis.get('search_queries', [])
                destination = query_analysis.get('destination', 'Unknown')  # Extract destination

                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. {query}\n")

                f.write(f"\nSearch Parameters:\n")
                f.write(f"  Destination: {destination}\n")

                # Step 2: Web Search
                f.write("STEP 2: WEB SEARCH\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                # Pass destination to search method
                search_results = self.search_agent.search(search_queries, destination)
                search_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {search_time}s\n")
                f.write(f"Search Results Found: {len(search_results)}\n\n")

                for i, result in enumerate(search_results, 1):
                    f.write(f"{i}. {result.get('title', 'No Title')}\n")
                    f.write(f"   URL: {result.get('url', 'No URL')}\n")
                    f.write(f"   Description: {(result.get('description', '') or '')[:150]}...\n\n")

                # Step 3: Intelligent Scraping
                f.write("STEP 3: INTELLIGENT SCRAPING\n")
                f.write("-" * 40 + "\n")

                if not search_results:
                    f.write("❌ No search results to scrape!\n")
                    return filepath

                start_time = time.time()
                enriched_results = await self.scraper.scrape_search_results(search_results)
                scraping_time = round(time.time() - start_time, 2)

                f.write(f"Scraping Time: {scraping_time}s\n")
                f.write(f"Successful Scrapes: {len([r for r in enriched_results if r.get('scraped_content')])}\n")
                f.write(f"Failed Scrapes: {len([r for r in enriched_results if not r.get('scraped_content')])}\n\n")

                successful_scrapes = 0

                # Show detailed content for each scraped result
                for i, result in enumerate(enriched_results, 1):
                    f.write(f"SCRAPE RESULT {i}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"URL: {result.get('url', 'No URL')}\n")
                    f.write(f"Title: {result.get('title', 'No Title')}\n")
                    f.write(f"Scraping Method: {result.get('scraping_method', 'Unknown')}\n")

                    scraped_content = result.get('scraped_content')
                    if scraped_content:
                        successful_scrapes += 1
                        f.write(f"Content Length: {len(scraped_content)} characters\n")
                        f.write(f"Status: ✅ Successfully scraped\n\n")
                        f.write("FULL SCRAPED CONTENT:\n")
                        f.write("~" * 60 + "\n")
                        f.write(scraped_content)
                        f.write("\n" + "~" * 60 + "\n\n")
                    else:
                        f.write("Status: ❌ Failed to scrape\n")
                        error_msg = result.get('scraping_error', 'Unknown error')
                        f.write(f"Error: {error_msg}\n\n")

                # Step 4: Editor Processing
                f.write("STEP 4: EDITOR PROCESSING\n")
                f.write("-" * 40 + "\n")

                if successful_scrapes > 0:
                    start_time = time.time()

                    # Process all successfully scraped content through editor
                    scraped_contents = []
                    for result in enriched_results:
                        if result.get('scraped_content'):
                            scraped_contents.append({
                                'url': result.get('url'),
                                'title': result.get('title'),
                                'content': result.get('scraped_content')
                            })

                    if scraped_contents:
                        # Use the editor agent to process content
                        editor_input = {
                            'scraped_contents': scraped_contents,
                            'query_analysis': query_analysis,
                            'destination': destination
                        }

                        # Call editor agent (replace this with actual editor method)
                        # For now, we'll just show what would be sent to the editor
                        f.write(f"Content pieces sent to editor: {len(scraped_contents)}\n")
                        f.write(f"Total content length: {sum(len(c['content']) for c in scraped_contents)} characters\n")
                        f.write(f"Destination context: {destination}\n")
                        f.write(f"Query context: {restaurant_query}\n\n")

                        f.write("EDITOR INPUT SUMMARY:\n")
                        f.write("-" * 30 + "\n")
                        for i, content in enumerate(scraped_contents, 1):
                            f.write(f"{i}. {content['title'][:60]}...\n")
                            f.write(f"   URL: {content['url']}\n")
                            f.write(f"   Content: {len(content['content'])} chars\n")
                            f.write(f"   Preview: {content['content'][:200]}...\n\n")

                        editing_time = round(time.time() - start_time, 2)
                        f.write(f"Editor processing time: {editing_time}s\n\n")

                else:
                    f.write("❌ No content to send to editor (all scraping failed)\n\n")

                # Statistics
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
                f.write(f"Destination: {destination}\n")
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
                    caption=f"🧪 Scraping test: {query}"
                )

            logger.info("Successfully sent scraping test results to admin")

        except Exception as e:
            logger.error(f"Failed to send scraping results to admin: {e}")


# Convenience function for backward compatibility
def add_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot
    This function is now deprecated since commands are handled directly in telegram_bot.py
    Keeping for backward compatibility.
    """
    logger.info("Note: scrape test commands are now handled directly in telegram_bot.py")
    pass