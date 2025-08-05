# scrape_test.py - FIXED to show complete database details and AI reasoning

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
    FIXED: Test that shows complete database information and AI evaluator reasoning

    Fixes:
    1. Displays actual cuisine_tags (not non-existent cuisine_type)
    2. Shows raw_description from database
    3. Includes complete AI evaluator reasoning
    4. Shows full evaluation details
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator  # Singleton instance
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents from orchestrator to ensure consistency with production
        self.query_analyzer = orchestrator.query_analyzer
        self.database_search_agent = orchestrator.database_search_agent
        self.dbcontent_evaluation_agent = orchestrator.dbcontent_evaluation_agent  
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper
        self.editor_agent = orchestrator.editor_agent

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run the COMPLETE production pipeline with FIXED database display and AI reasoning
        """
        logger.info(f"Testing complete pipeline for: {restaurant_query}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_pipeline_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE RESTAURANT PIPELINE TEST (Production Path)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Pipeline: query → database → evaluation → [search] → [SmartRestaurantScraper] → editor\n")
            f.write(f"Orchestrator: Production singleton instance\n\n")

            try:
                total_start_time = time.time()

                # STEP 1: Query Analysis
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                destination = query_analysis.get('destination', 'Unknown')
                search_queries = query_analysis.get('search_queries', [])

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Query Analysis Results:\n")
                f.write(f"  Destination: {destination}\n")
                f.write(f"  Is English Speaking: {query_analysis.get('is_english_speaking', True)}\n")
                f.write(f"  Local Language: {query_analysis.get('local_language', 'None')}\n")
                f.write(f"  Search Queries Generated: {len(search_queries)}\n")

                for i, query in enumerate(search_queries, 1):
                    f.write(f"    {i}. {query}\n")

                pipeline_data = {
                    **query_analysis,
                    "query": restaurant_query,
                    "raw_query": restaurant_query
                }

                # STEP 2: Database Search (FIXED to show complete info)
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
                    f.write(f"\nDatabase Restaurants (FIXED - showing complete info):\n")
                    for i, restaurant in enumerate(database_restaurants[:5], 1):
                        # FIXED: Use correct field names from database
                        name = restaurant.get('name', 'No Name')
                        cuisine_tags = restaurant.get('cuisine_tags', [])  # FIXED: use cuisine_tags not cuisine_type
                        raw_description = restaurant.get('raw_description', '')  # FIXED: show actual description
                        mention_count = restaurant.get('mention_count', 0)
                        sources = restaurant.get('sources', [])

                        # Format cuisine tags nicely
                        cuisine_display = ', '.join(cuisine_tags) if cuisine_tags else 'Unknown cuisine'

                        # Show first 100 chars of description
                        description_preview = raw_description[:100] + "..." if len(raw_description) > 100 else raw_description or "No description"

                        f.write(f"  {i}. {name}\n")
                        f.write(f"     Cuisine: {cuisine_display}\n")
                        f.write(f"     Description: {description_preview}\n")
                        f.write(f"     Mentions: {mention_count}, Sources: {len(sources)}\n")

                    if len(database_restaurants) > 5:
                        f.write(f"  ... and {len(database_restaurants) - 5} more\n")

                # STEP 3: Content Evaluation (FIXED to show complete AI reasoning)
                f.write(f"\nSTEP 3: CONTENT EVALUATION & ROUTING\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                evaluation_result = self.dbcontent_evaluation_agent.evaluate_and_route(pipeline_data)
                evaluation_time = round(time.time() - start_time, 2)

                pipeline_data.update(evaluation_result)

                f.write(f"Processing Time: {evaluation_time}s\n")

                # FIXED: Show complete evaluation details from AI
                evaluation_details = evaluation_result.get('evaluation_result', {})
                f.write(f"Database Sufficient: {evaluation_details.get('database_sufficient', False)}\n")
                f.write(f"Trigger Web Search: {evaluation_details.get('trigger_web_search', True)}\n")
                f.write(f"Skip Web Search: {pipeline_data.get('skip_web_search', False)}\n")
                f.write(f"Final Content Source: {pipeline_data.get('content_source', 'unknown')}\n")

                # FIXED: Show actual AI reasoning instead of fallback message
                reasoning = evaluation_details.get('reasoning', 'No reasoning provided')
                quality_score = evaluation_details.get('quality_score', 'N/A')
                f.write(f"AI Reasoning: {reasoning}\n")
                f.write(f"Quality Score: {quality_score}\n")

                # FIXED: Show evaluation summary if available
                eval_summary = evaluation_details.get('evaluation_summary', {})
                if eval_summary:
                    f.write(f"Evaluation Summary: {eval_summary}\n")

                # STEP 4: Web Search (conditional)
                search_results = []
                search_time = 0

                if not pipeline_data.get('skip_web_search', False):
                    f.write(f"\nSTEP 4: WEB SEARCH (Required)\n")
                    f.write("-" * 40 + "\n")

                    start_time = time.time()

                    search_queries = pipeline_data.get('search_queries', [])
                    destination = pipeline_data.get('destination', 'Unknown')

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

                    for i, result in enumerate(search_results[:10], 1):
                        f.write(f"  {i}. {result.get('title', 'No Title')}\n")
                        f.write(f"     URL: {result.get('url', 'No URL')}\n")
                        f.write(f"     Quality Score: {result.get('quality_score', 'N/A')}\n")

                    pipeline_data['search_results'] = search_results

                else:
                    f.write(f"\nSTEP 4: WEB SEARCH (Skipped - Database Sufficient)\n")
                    f.write("-" * 40 + "\n")
                    f.write("Web search skipped because database content was sufficient.\n")

                # STEP 5: Smart Restaurant Scraper (conditional)
                enriched_results = []
                scraping_time = 0
                successful_scrapes = 0

                if search_results:
                    f.write(f"\nSTEP 5: SMART RESTAURANT SCRAPER\n")
                    f.write("-" * 40 + "\n")

                    start_time = time.time()

                    # Run intelligent scraping (same as production)
                    def run_scraping():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(
                                self.scraper.scrape_search_results(search_results)
                            )
                        finally:
                            loop.close()

                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        enriched_results = pool.submit(run_scraping).result()

                    scraping_time = round(time.time() - start_time, 2)

                    # Analyze scraping results
                    successful_scrapes = len([r for r in enriched_results if r.get('scraping_success')])
                    failed_scrapes = len([r for r in enriched_results if r.get('scraping_failed')])

                    f.write(f"Processing Time: {scraping_time}s\n")
                    f.write(f"URLs Processed: {len(enriched_results)}\n")
                    f.write(f"Successful Scrapes: {successful_scrapes}\n")
                    f.write(f"Failed Scrapes: {failed_scrapes}\n")
                    f.write(f"Success Rate: {round((successful_scrapes/max(len(enriched_results),1))*100, 1)}%\n\n")

                    # Show scraping details
                    f.write("DETAILED SCRAPING RESULTS:\n")
                    f.write("=" * 60 + "\n")

                    for i, result in enumerate(enriched_results, 1):
                        f.write(f"SCRAPE #{i}:\n")
                        f.write("-" * 50 + "\n")
                        f.write(f"URL: {result.get('url', 'No URL')}\n")
                        f.write(f"Title: {result.get('title', 'No Title')}\n")
                        f.write(f"Scraping Method: {result.get('scraping_method', 'Unknown')}\n")

                        scraped_content = result.get('scraped_content')
                        if scraped_content:
                            f.write(f"Status: ✅ SUCCESS\n")
                            f.write(f"Content Length: {len(scraped_content)} characters\n")
                        else:
                            f.write(f"Status: ❌ Failed to scrape\n")

                    pipeline_data['enriched_results'] = enriched_results

                else:
                    f.write(f"\nSTEP 5: SMART RESTAURANT SCRAPER (Skipped)\n")
                    f.write("-" * 40 + "\n")
                    f.write("SmartRestaurantScraper skipped - no search results to scrape.\n")

                # STEP 6: Editor Agent Input Preparation (FIXED to show complete database info)
                f.write(f"\nSTEP 6: EDITOR AGENT INPUT PREPARATION\n")
                f.write("-" * 40 + "\n")

                if pipeline_data.get('content_source') == 'database':
                    # Database route - FIXED to show complete restaurant info
                    final_content = pipeline_data.get('final_database_content', [])
                    f.write(f"Content Source: Database\n")
                    f.write(f"Database restaurants to be sent to editor: {len(final_content)}\n")
                    f.write(f"Editor will process: Database restaurant data\n\n")

                    f.write("COMPLETE DATABASE RESTAURANT DETAILS:\n")
                    f.write("-" * 50 + "\n")

                    for i, restaurant in enumerate(final_content[:3], 1):
                        name = restaurant.get('name', 'No Name')
                        cuisine_tags = restaurant.get('cuisine_tags', [])  # FIXED: correct field name
                        raw_description = restaurant.get('raw_description', '')  # FIXED: show actual description
                        address = restaurant.get('address', 'No address')
                        mention_count = restaurant.get('mention_count', 0)
                        sources = restaurant.get('sources', [])

                        f.write(f"  {i}. {name}\n")
                        f.write(f"     Cuisine Tags: {', '.join(cuisine_tags) if cuisine_tags else 'No cuisine tags'}\n")
                        f.write(f"     Address: {address}\n")
                        f.write(f"     Mentions: {mention_count}\n")
                        f.write(f"     Sources: {len(sources)} sources\n")
                        f.write(f"     Description: {raw_description[:200] + '...' if len(raw_description) > 200 else raw_description or 'No description'}\n\n")

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

                # PIPELINE STATISTICS
                f.write("PIPELINE STATISTICS\n")
                f.write("=" * 40 + "\n")

                # SmartRestaurantScraper stats
                scraper_stats = self.scraper.get_stats()
                f.write("SmartRestaurantScraper Statistics:\n")
                for key, value in scraper_stats.items():
                    f.write(f"  {key}: {value}\n")

                # Search agent stats
                search_stats = self.search_agent.get_stats()
                f.write("Search Agent Statistics:\n")
                for key, value in search_stats.items():
                    f.write(f"  {key}: {value}\n")

                # Database search stats
                db_stats = self.database_search_agent.get_stats()
                f.write("Database Search Statistics:\n")
                for key, value in db_stats.items():
                    f.write(f"  {key}: {value}\n")

                # Timing summary
                total_time = round(time.time() - total_start_time, 2)

                f.write("OVERALL PIPELINE TIMING:\n")
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
                f.write("✅ Search → ✅ SmartRestaurantScraper → ✅ Ready for Editor\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during pipeline test: {str(e)}\n")
                logger.error(f"Error during pipeline test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            self._send_results_to_admin(bot, filepath, restaurant_query, successful_scrapes if 'successful_scrapes' in locals() else 0)

        return filepath

    def _send_results_to_admin(self, bot, file_path: str, query: str, successful_count: int):
        """Send pipeline test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🧪 <b>Complete Pipeline Test Results (FIXED)</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"✅ Successful scrapes: {successful_count}\n"
                f"🔧 Pipeline: query → database → evaluation → search → scrape → editor\n"
                f"🎯 Follows: EXACT production path\n"
                f"🔧 FIXED: Shows complete database info + AI reasoning\n\n"
                f"{'✅ Content ready for editor' if successful_count > 0 else '❌ No scraped content (check database route)'}\n\n"
                f"📄 Complete pipeline analysis with FULL database details and AI reasoning."
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
                    caption=f"🧪 Pipeline test: {query}"
                )

            logger.info("Successfully sent pipeline test results to admin")

        except Exception as e:
            logger.error(f"Failed to send pipeline results to admin: {e}")


# Convenience function for telegram bot integration
def add_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot
    This function is kept for backward compatibility
    """
    logger.info("Note: scrape test commands are now handled directly in telegram_bot.py")