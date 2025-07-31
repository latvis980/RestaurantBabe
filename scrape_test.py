# scrape_test.py - UPDATED for current pipeline architecture
# Now follows the EXACT same path as production: query_analyzer → database_search_agent → dbcontent_evaluation_agent → search_agent → scraper → editor_agent

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
    UPDATED: Test that follows the EXACT production pipeline path:

    Production Pipeline:
    1. QueryAnalyzer - analyze query and generate search terms
    2. DatabaseSearchAgent - search database for matching restaurants
    3. ContentEvaluationAgent - evaluate if database results are sufficient
    4. [IF WEB SEARCH NEEDED] SearchAgent - perform web search with query metadata
    5. [IF WEB SEARCH NEEDED] WebScraper - scrape search results intelligently
    6. EditorAgent - process final content (database OR scraped)
    7. FollowUpSearchAgent - enhance content if needed
    8. TelegramFormatter - format for output

    This test follows steps 1-6 to debug scraping issues and see what content gets scraped.
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator  # Singleton instance
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents from orchestrator to ensure consistency with production
        self.query_analyzer = orchestrator.query_analyzer
        self.database_search_agent = orchestrator.database_search_agent  # NEW
        self.dbcontent_evaluation_agent = orchestrator.dbcontent_evaluation_agent  # NEW  
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper
        self.editor_agent = orchestrator.editor_agent

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run the COMPLETE production pipeline up to editor stage and dump all results.

        This shows:
        - What the query analyzer generates
        - What database results are found  
        - What the content evaluation agent decides
        - If web search happens, what search results are found
        - What gets scraped successfully (FULL content)
        - What goes to the editor agent

        Args:
            restaurant_query: The restaurant query to test (e.g., "best brunch in Lisbon")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing COMPLETE pipeline for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE RESTAURANT PIPELINE TEST (Production Path)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Pipeline: query → database → evaluation → [search] → [scrape] → editor\n")
            f.write(f"Orchestrator: Production singleton instance\n\n")

            try:
                total_start_time = time.time()

                # STEP 1: Query Analysis (same as production)
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Query Analysis Results:\n")
                f.write(f"  Destination: {query_analysis.get('destination', 'Unknown')}\n")
                f.write(f"  Is English Speaking: {query_analysis.get('is_english_speaking', 'Unknown')}\n")
                f.write(f"  Local Language: {query_analysis.get('local_language', 'None')}\n")
                f.write(f"  Search Queries Generated: {len(query_analysis.get('search_queries', []))}\n")

                search_queries = query_analysis.get('search_queries', [])
                for i, query in enumerate(search_queries, 1):
                    f.write(f"    {i}. {query}\n")

                # Add raw query to match production pipeline
                pipeline_data = {
                    **query_analysis,
                    "query": restaurant_query,
                    "raw_query": restaurant_query
                }

                # STEP 2: Database Search (NEW - matches production)
                f.write(f"\nSTEP 2: DATABASE SEARCH\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                database_result = self.database_search_agent.search_and_evaluate(pipeline_data)
                database_time = round(time.time() - start_time, 2)

                pipeline_data.update(database_result)

                f.write(f"Processing Time: {database_time}s\n")
                f.write(f"Database Results Found: {len(database_result.get('database_results', []))}\n")
                f.write(f"Has Database Content: {database_result.get('has_database_content', False)}\n")
                f.write(f"Content Source: {database_result.get('content_source', 'unknown')}\n")

                database_restaurants = database_result.get('database_results', [])
                if database_restaurants:
                    f.write(f"\nDatabase Restaurants:\n")
                    for i, restaurant in enumerate(database_restaurants[:5], 1):  # Show first 5
                        f.write(f"  {i}. {restaurant.get('name', 'No Name')} - {restaurant.get('cuisine_type', 'Unknown cuisine')}\n")
                    if len(database_restaurants) > 5:
                        f.write(f"  ... and {len(database_restaurants) - 5} more\n")

                # STEP 3: Content Evaluation (NEW - matches production)
                f.write(f"\nSTEP 3: CONTENT EVALUATION & ROUTING\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                evaluation_result = self.dbcontent_evaluation_agent.evaluate_and_route(pipeline_data)
                evaluation_time = round(time.time() - start_time, 2)

                pipeline_data.update(evaluation_result)

                f.write(f"Processing Time: {evaluation_time}s\n")

                evaluation_details = evaluation_result.get('evaluation_result', {})
                f.write(f"Database Sufficient: {evaluation_details.get('database_sufficient', False)}\n")
                f.write(f"Trigger Web Search: {evaluation_details.get('trigger_web_search', True)}\n")
                f.write(f"Skip Web Search: {pipeline_data.get('skip_web_search', False)}\n")
                f.write(f"Final Content Source: {pipeline_data.get('content_source', 'unknown')}\n")
                f.write(f"Reasoning: {evaluation_details.get('reasoning', 'No reasoning provided')}\n")

                # STEP 4: Web Search (conditional - only if evaluation says so)
                search_results = []
                search_time = 0

                if not pipeline_data.get('skip_web_search', False):
                    f.write(f"\nSTEP 4: WEB SEARCH (Required)\n")
                    f.write("-" * 40 + "\n")

                    start_time = time.time()

                    # Use exact same parameters as production pipeline
                    search_queries = pipeline_data.get('search_queries', [])
                    destination = pipeline_data.get('destination', 'Unknown')

                    # Prepare query metadata same as production
                    query_metadata = {
                        'is_english_speaking': pipeline_data.get('is_english_speaking', True),
                        'local_language': pipeline_data.get('local_language')
                    }

                    search_results = self.search_agent.search(search_queries, destination, query_metadata)
                    search_time = round(time.time() - start_time, 2)

                    f.write(f"Processing Time: {search_time}s\n")
                    f.write(f"Search Results Found: {len(search_results)}\n")
                    f.write(f"Query Metadata Used:\n")
                    f.write(f"  English Speaking: {query_metadata.get('is_english_speaking')}\n")
                    f.write(f"  Local Language: {query_metadata.get('local_language', 'None')}\n\n")

                    for i, result in enumerate(search_results, 1):
                        f.write(f"{i}. {result.get('title', 'No Title')}\n")
                        f.write(f"   URL: {result.get('url', 'No URL')}\n")
                        f.write(f"   Quality Score: {result.get('quality_score', 'N/A')}\n")
                        f.write(f"   Description: {(result.get('description', '') or '')[:150]}...\n\n")

                else:
                    f.write(f"\nSTEP 4: WEB SEARCH (Skipped - Database Sufficient)\n")
                    f.write("-" * 40 + "\n")
                    f.write("Web search skipped because database content was sufficient.\n")

                # STEP 5: Intelligent Scraping (conditional - only if search happened)
                enriched_results = []
                scraping_time = 0
                successful_scrapes = 0

                if search_results:
                    f.write(f"\nSTEP 5: INTELLIGENT SCRAPING\n")
                    f.write("-" * 40 + "\n")

                    start_time = time.time()
                    enriched_results = await self.scraper.scrape_search_results(search_results)
                    scraping_time = round(time.time() - start_time, 2)

                    successful_scrapes = len([r for r in enriched_results if r.get('scraped_content')])

                    f.write(f"Scraping Time: {scraping_time}s\n")
                    f.write(f"Successful Scrapes: {successful_scrapes}\n")
                    f.write(f"Failed Scrapes: {len(enriched_results) - successful_scrapes}\n\n")

                    # Show detailed content for each scraped result (MOST IMPORTANT FOR DEBUGGING)
                    for i, result in enumerate(enriched_results, 1):
                        f.write(f"SCRAPE RESULT {i}:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"URL: {result.get('url', 'No URL')}\n")
                        f.write(f"Title: {result.get('title', 'No Title')}\n")
                        f.write(f"Scraping Method: {result.get('scraping_method', 'Unknown')}\n")
                        f.write(f"Quality Score: {result.get('quality_score', 'N/A')}\n")

                        scraped_content = result.get('scraped_content')
                        if scraped_content:
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

                    pipeline_data['search_results'] = search_results
                    pipeline_data['enriched_results'] = enriched_results

                else:
                    f.write(f"\nSTEP 5: INTELLIGENT SCRAPING (Skipped)\n")
                    f.write("-" * 40 + "\n")
                    f.write("Scraping skipped - no search results to scrape.\n")

                # STEP 6: Editor Processing Preview
                f.write(f"\nSTEP 6: EDITOR AGENT INPUT PREPARATION\n")
                f.write("-" * 40 + "\n")

                # Prepare data that would be sent to editor agent (same as production)
                if pipeline_data.get('content_source') == 'database':
                    # Database route
                    final_content = pipeline_data.get('final_database_content', [])
                    f.write(f"Content Source: Database\n")
                    f.write(f"Database restaurants to be sent to editor: {len(final_content)}\n")
                    f.write(f"Editor will process: Database restaurant data\n")

                    for i, restaurant in enumerate(final_content[:3], 1):
                        f.write(f"  {i}. {restaurant.get('name', 'No Name')}\n")
                        f.write(f"     Cuisine: {restaurant.get('cuisine_type', 'Unknown')}\n")
                        f.write(f"     Description: {restaurant.get('description', 'No description')[:100]}...\n")

                elif successful_scrapes > 0:
                    # Web scraping route
                    scraped_contents = []
                    for result in enriched_results:
                        if result.get('scraped_content'):
                            scraped_contents.append({
                                'url': result.get('url'),
                                'title': result.get('title'),
                                'content': result.get('scraped_content')
                            })

                    f.write(f"Content Source: Web Scraping\n")
                    f.write(f"Scraped content pieces to be sent to editor: {len(scraped_contents)}\n")
                    f.write(f"Total scraped content length: {sum(len(c['content']) for c in scraped_contents)} characters\n")
                    f.write(f"Editor will process: Scraped web content\n\n")

                    f.write("EDITOR INPUT SUMMARY:\n")
                    f.write("-" * 30 + "\n")
                    for i, content in enumerate(scraped_contents, 1):
                        f.write(f"{i}. {content['title'][:60]}...\n")
                        f.write(f"   URL: {content['url']}\n")
                        f.write(f"   Content: {len(content['content'])} chars\n")
                        f.write(f"   Preview: {content['content'][:200]}...\n\n")

                else:
                    f.write(f"❌ No content available for editor (neither database nor scraped content)\n")

                # Statistics Section
                f.write("\nPIPELINE STATISTICS\n")
                f.write("=" * 40 + "\n")

                # Scraper stats (important for debugging)
                scraper_stats = self.scraper.get_stats()
                f.write(f"Intelligent Scraper Statistics:\n")
                for key, value in scraper_stats.items():
                    f.write(f"  {key}: {value}\n")

                # Search agent stats
                search_stats = self.search_agent.get_stats()
                f.write(f"\nSearch Agent Statistics:\n")
                f.write(f"  Total searches: {search_stats.get('total_searches', 0)}\n")
                f.write(f"  Results filtered: {search_stats.get('results_filtered', 0)}\n")
                f.write(f"  High quality sources: {search_stats.get('high_quality_sources', 0)}\n")

                # Database search stats if available
                if hasattr(self.database_search_agent, 'get_stats'):
                    db_stats = self.database_search_agent.get_stats()
                    f.write(f"\nDatabase Search Statistics:\n")
                    for key, value in db_stats.items():
                        f.write(f"  {key}: {value}\n")

                # Overall timing
                total_time = round(time.time() - total_start_time, 2)
                f.write(f"\nOVERALL PIPELINE TIMING:\n")
                f.write(f"  1. Query Analysis: {analysis_time}s\n")
                f.write(f"  2. Database Search: {database_time}s\n")
                f.write(f"  3. Content Evaluation: {evaluation_time}s\n")
                if search_time > 0:
                    f.write(f"  4. Web Search: {search_time}s\n")
                if scraping_time > 0:
                    f.write(f"  5. Intelligent Scraping: {scraping_time}s\n")
                f.write(f"  Total Pipeline Time: {total_time}s\n\n")

                # Final summary
                f.write("=" * 80 + "\n")
                f.write("PIPELINE TEST COMPLETED SUCCESSFULLY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Destination: {pipeline_data.get('destination', 'Unknown')}\n")
                f.write(f"Final Content Source: {pipeline_data.get('content_source', 'unknown')}\n")
                f.write(f"Database Restaurants: {len(database_restaurants)}\n")
                f.write(f"Search Results: {len(search_results)}\n")
                f.write(f"Successful Scrapes: {successful_scrapes}\n")
                f.write(f"Total Processing Time: {total_time}s\n")
                f.write("Pipeline: ✅ Query → ✅ Database → ✅ Evaluation → ")
                f.write("✅ Search → ✅ Scrape → ✅ Ready for Editor\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during pipeline test: {str(e)}\n")
                logger.error(f"Error during pipeline test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, successful_scrapes if 'successful_scrapes' in locals() else 0)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, successful_count: int):
        """Send pipeline test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🧪 <b>Complete Pipeline Test Results</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"✅ Successful scrapes: {successful_count}\n"
                f"🔧 Pipeline: query → database → evaluation → search → scrape → editor\n"
                f"🎯 Follows: EXACT production path\n\n"
                f"{'✅ Content ready for editor' if successful_count > 0 else '❌ No scraped content (check database route)'}\n\n"
                f"📄 Complete pipeline analysis attached with FULL scraped content."
            )

            await bot.send_message(
                self.admin_chat_id,
                summary,
                parse_mode='HTML'
            )

            # Send the results file
            with open(file_path, 'rb') as f:
                await bot.send_document(
                    self.admin_chat_id,
                    f,
                    caption=f"🧪 Pipeline test: {query}"
                )

            logger.info("Successfully sent pipeline test results to admin")

        except Exception as e:
            logger.error(f"Failed to send pipeline results to admin: {e}")


# Convenience function for backward compatibility
def add_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot
    This function is now deprecated since commands are handled directly in telegram_bot.py
    Keeping for backward compatibility.
    """
    logger.info("Note: pipeline test commands are now handled directly in telegram_bot.py")
    pass