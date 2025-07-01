# scrape_test.py - Updated to use orchestrator singleton
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
    - What content is actually scraped
    - What goes to list_analyzer

    Updated to use orchestrator singleton pattern
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator  # Now receives the singleton instance
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize pipeline components - get from the orchestrator to ensure consistency
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run complete scraping process and dump results to file

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
            f.write("RESTAURANT SCRAPING PROCESS TEST\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
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

                # Log first 10 URLs for analysis
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
                        # Write first few sentences instead of full content
                        preview = content[:300].replace('\n', ' ') + "..." if len(content) > 300 else content
                        f.write(f"     Preview: {preview}\n")
                    else:
                        failed_scrapes += 1
                        f.write(f"  {i}. ❌ {url}\n")
                        f.write(f"     Status: Failed or insufficient content\n")

                f.write(f"\nScraping Summary:\n")
                f.write(f"  Successful: {successful_scrapes}\n")
                f.write(f"  Failed: {failed_scrapes}\n")
                f.write(f"  Total Content: {total_content_length:,} characters\n")
                f.write(f"  Average per successful scrape: {total_content_length // max(successful_scrapes, 1):,} chars\n")

                # Step 4: What goes to List Analyzer
                f.write("\nSTEP 4: LIST ANALYZER INPUT\n")
                f.write("-" * 40 + "\n")

                # Prepare the exact input that would go to list_analyzer
                analyzer_input = {
                    "scraped_articles": scraped_results,
                    "keywords_for_analysis": query_analysis.get("primary_search_parameters", []),
                    "primary_search_parameters": query_analysis.get("primary_search_parameters", []),
                    "secondary_filter_parameters": query_analysis.get("secondary_filter_parameters", []),
                    "destination": query_analysis.get("destination", "Unknown")
                }

                f.write(f"Articles sent to analyzer: {len(analyzer_input['scraped_articles'])}\n")
                f.write(f"Keywords for analysis: {analyzer_input['keywords_for_analysis']}\n")
                f.write(f"Primary parameters: {analyzer_input['primary_search_parameters']}\n")
                f.write(f"Secondary parameters: {analyzer_input['secondary_filter_parameters']}\n")
                f.write(f"Destination: {analyzer_input['destination']}\n\n")

                # Show the actual content structure (only first 5 articles for brevity)
                f.write("Content Structure Analysis:\n")
                for i, article in enumerate(analyzer_input['scraped_articles'][:5], 1):
                    f.write(f"\nArticle {i}:\n")
                    f.write(f"  URL: {article.get('url', 'Unknown')}\n")
                    f.write(f"  Title: {article.get('title', 'No title')}\n")
                    f.write(f"  Content length: {len(article.get('content', ''))}\n")

                    content = article.get('content', '')
                    if content:
                        # Show just a preview instead of full content
                        sentences = content.split('. ')[:3]
                        preview = '. '.join(sentences)
                        f.write(f"  Content preview: {preview[:300]}...\n")

                if len(analyzer_input['scraped_articles']) > 5:
                    f.write(f"\n... and {len(analyzer_input['scraped_articles']) - 5} more articles\n")

                # Get scraper statistics
                scraper_stats = self.scraper.get_stats()
                f.write(f"\nIntelligent Scraper Statistics:\n")
                for key, value in scraper_stats.items():
                    f.write(f"  {key}: {value}\n")

                # Overall timing
                total_time = analysis_time + search_time + scraping_time
                f.write(f"\nOVERALL TIMING:\n")
                f.write(f"  Query Analysis: {analysis_time}s\n")
                f.write(f"  Web Search: {search_time}s\n")
                f.write(f"  Intelligent Scraping: {scraping_time}s\n")
                f.write(f"  Total: {total_time}s\n\n")

                f.write("=" * 80 + "\n")
                f.write("TEST COMPLETED SUCCESSFULLY\n")
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
                f"🔧 Orchestrator: Singleton instance\n"
                f"🎯 Focus: Complete pipeline analysis\n\n"
                f"{'✅ Content extracted successfully' if successful_count > 0 else '❌ No content extracted'}\n\n"
                f"📄 Detailed scraping analysis attached."
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
                    caption=f"🧪 Scraping test results for: {query}"
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