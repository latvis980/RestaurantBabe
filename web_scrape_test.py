# web_scrape_test.py - DEDICATED WEB SEARCH & SCRAPING TEST
# Bypasses database entirely - focuses on web search → scraping → content analysis

import asyncio
import time
import tempfile
import os
import threading
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class WebScrapeTest:
    """
    DEDICATED test for web search and scraping pipeline ONLY.

    This test:
    - Skips all database steps
    - Forces web search to happen
    - Shows detailed search results and filtering
    - Shows complete scraping process with full content
    - Perfect for debugging web scraping issues

    Pipeline tested: query_analyzer → search_agent → scraper → content analysis
    Command: /test_wscrape
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator  # Singleton instance
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get only the agents we need for web scraping test
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent  
        self.scraper = orchestrator.scraper
        self.editor_agent = orchestrator.editor_agent

    async def test_web_scraping_only(self, restaurant_query: str, bot=None) -> str:
        """
        Test ONLY the web search and scraping pipeline.

        Forces web search to happen regardless of database content.
        Shows complete search and scraping analysis.

        Args:
            restaurant_query: The restaurant query to test (e.g., "best brunch in Lisbon")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing WEB SCRAPING ONLY for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_scrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WEB SEARCH & SCRAPING TEST (Database Bypassed)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Focus: WEB SEARCH → SCRAPING → CONTENT ANALYSIS\n")
            f.write(f"Database: BYPASSED (forced web search)\n")
            f.write(f"Command: /test_wscrape\n\n")

            try:
                total_start_time = time.time()

                # STEP 1: Query Analysis (to get search terms)
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                destination = query_analysis.get('destination', 'Unknown')
                is_english_speaking = query_analysis.get('is_english_speaking', True)
                local_language = query_analysis.get('local_language', 'None')
                search_queries = query_analysis.get('search_queries', [])

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"ANALYSIS RESULTS:\n")
                f.write(f"  Destination: {destination}\n")
                f.write(f"  Is English Speaking: {is_english_speaking}\n")
                f.write(f"  Local Language: {local_language}\n")
                f.write(f"  Search Queries Generated: {len(search_queries)}\n\n")

                f.write("GENERATED SEARCH QUERIES:\n")
                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. {query}\n")

                if not search_queries:
                    f.write("❌ ERROR: No search queries generated!\n")
                    return filepath

                # STEP 2: WEB SEARCH (FORCED - bypass database)
                f.write(f"\nSTEP 2: WEB SEARCH (FORCED)\n")
                f.write("-" * 40 + "\n")
                f.write("🌐 BYPASSING DATABASE - FORCING WEB SEARCH\n\n")

                start_time = time.time()

                # Prepare query metadata exactly like production
                query_metadata = {
                    'is_english_speaking': is_english_speaking,
                    'local_language': local_language
                }

                f.write(f"Search Parameters:\n")
                f.write(f"  Destination: {destination}\n")
                f.write(f"  Query Metadata: {query_metadata}\n")
                f.write(f"  Search Queries: {search_queries}\n\n")

                # Execute search with production parameters
                search_results = self.search_agent.search(search_queries, destination, query_metadata)
                search_time = round(time.time() - start_time, 2)

                f.write(f"SEARCH RESULTS:\n")
                f.write(f"  Processing Time: {search_time}s\n")
                f.write(f"  Results Found: {len(search_results)}\n")
                f.write(f"  Search Method: BraveSearch + Filtering\n\n")

                if not search_results:
                    f.write("❌ NO SEARCH RESULTS FOUND!\n")
                    f.write("This could indicate:\n")
                    f.write("  - Network connectivity issues\n")
                    f.write("  - Search API problems\n")
                    f.write("  - Query generation issues\n")
                    f.write("  - Overly restrictive filtering\n")
                    return filepath

                # Show detailed search results
                f.write("DETAILED SEARCH RESULTS:\n")
                f.write("~" * 60 + "\n")
                for i, result in enumerate(search_results, 1):
                    f.write(f"{i}. SEARCH RESULT:\n")
                    f.write(f"   Title: {result.get('title', 'No Title')}\n")
                    f.write(f"   URL: {result.get('url', 'No URL')}\n")
                    f.write(f"   Quality Score: {result.get('quality_score', 'N/A')}\n")
                    f.write(f"   Source Type: {result.get('source_type', 'Unknown')}\n")

                    description = result.get('description', '')
                    if description:
                        f.write(f"   Description: {description[:200]}...\n")

                    # Show any filtering metadata
                    if 'filter_reason' in result:
                        f.write(f"   Filter Reason: {result['filter_reason']}\n")

                    f.write(f"\n")

                f.write("~" * 60 + "\n\n")

                # Get search agent statistics
                search_stats = self.search_agent.get_stats()
                f.write("SEARCH AGENT STATISTICS:\n")
                f.write("-" * 30 + "\n")
                for key, value in search_stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                # STEP 3: INTELLIGENT SCRAPING
                f.write("STEP 3: INTELLIGENT SCRAPING\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                enriched_results = await self.scraper.scrape_search_results(search_results)
                scraping_time = round(time.time() - start_time, 2)

                successful_scrapes = len([r for r in enriched_results if r.get('scraped_content')])
                failed_scrapes = len(enriched_results) - successful_scrapes

                f.write(f"SCRAPING SUMMARY:\n")
                f.write(f"  Processing Time: {scraping_time}s\n")
                f.write(f"  URLs Attempted: {len(enriched_results)}\n")
                f.write(f"  Successful Scrapes: {successful_scrapes}\n")
                f.write(f"  Failed Scrapes: {failed_scrapes}\n")
                f.write(f"  Success Rate: {round((successful_scrapes/max(len(enriched_results),1))*100, 1)}%\n\n")

                # Show detailed scraping results (MOST IMPORTANT FOR DEBUGGING)
                f.write("DETAILED SCRAPING RESULTS:\n")
                f.write("=" * 60 + "\n")

                total_content_length = 0
                content_by_method = {}

                for i, result in enumerate(enriched_results, 1):
                    f.write(f"SCRAPE #{i}:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"URL: {result.get('url', 'No URL')}\n")
                    f.write(f"Title: {result.get('title', 'No Title')}\n")
                    f.write(f"Original Quality Score: {result.get('quality_score', 'N/A')}\n")

                    scraping_method = result.get('scraping_method', 'Unknown')
                    f.write(f"Scraping Method: {scraping_method}\n")

                    # Track scraping methods
                    if scraping_method not in content_by_method:
                        content_by_method[scraping_method] = {'count': 0, 'success': 0, 'content_length': 0}
                    content_by_method[scraping_method]['count'] += 1

                    scraped_content = result.get('scraped_content')
                    if scraped_content:
                        content_length = len(scraped_content)
                        total_content_length += content_length
                        content_by_method[scraping_method]['success'] += 1
                        content_by_method[scraping_method]['content_length'] += content_length

                        f.write(f"Status: ✅ SUCCESS\n")
                        f.write(f"Content Length: {content_length} characters\n")
                        f.write(f"Processing Time: {result.get('scraping_time', 'N/A')}s\n")

                        # Show any additional metadata
                        if 'content_quality' in result:
                            f.write(f"Content Quality: {result['content_quality']}\n")
                        if 'restaurant_mentions' in result:
                            f.write(f"Restaurant Mentions: {result['restaurant_mentions']}\n")

                        f.write(f"\nFULL SCRAPED CONTENT:\n")
                        f.write("~" * 50 + "\n")
                        f.write(scraped_content)
                        f.write("\n" + "~" * 50 + "\n\n")

                    else:
                        f.write(f"Status: ❌ FAILED\n")
                        error_msg = result.get('scraping_error', 'Unknown error')
                        f.write(f"Error: {error_msg}\n")

                        # Show any retry information
                        if 'retry_count' in result:
                            f.write(f"Retries Attempted: {result['retry_count']}\n")
                        if 'fallback_attempted' in result:
                            f.write(f"Fallback Attempted: {result['fallback_attempted']}\n")

                        f.write(f"\n")

                # STEP 4: SCRAPING ANALYSIS
                f.write("STEP 4: SCRAPING ANALYSIS\n")
                f.write("-" * 40 + "\n")

                # Content by scraping method analysis
                f.write("CONTENT BY SCRAPING METHOD:\n")
                f.write("-" * 30 + "\n")
                for method, stats in content_by_method.items():
                    success_rate = round((stats['success'] / max(stats['count'], 1)) * 100, 1)
                    avg_content = round(stats['content_length'] / max(stats['success'], 1), 0) if stats['success'] > 0 else 0

                    f.write(f"{method.upper()}:\n")
                    f.write(f"  Attempts: {stats['count']}\n")
                    f.write(f"  Successes: {stats['success']}\n")
                    f.write(f"  Success Rate: {success_rate}%\n")
                    f.write(f"  Total Content: {stats['content_length']} chars\n")
                    f.write(f"  Avg Content/Success: {avg_content} chars\n\n")

                # Intelligent scraper statistics
                scraper_stats = self.scraper.get_stats()
                f.write("INTELLIGENT SCRAPER STATISTICS:\n")
                f.write("-" * 30 + "\n")
                for key, value in scraper_stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                # STEP 5: CONTENT PREPARATION FOR EDITOR
                f.write("STEP 5: CONTENT FOR EDITOR AGENT\n")
                f.write("-" * 40 + "\n")

                if successful_scrapes > 0:
                    # Prepare content exactly like production pipeline would
                    scraped_contents = []
                    for result in enriched_results:
                        if result.get('scraped_content'):
                            scraped_contents.append({
                                'url': result.get('url'),
                                'title': result.get('title'),
                                'content': result.get('scraped_content'),
                                'quality_score': result.get('quality_score'),
                                'scraping_method': result.get('scraping_method')
                            })

                    f.write(f"EDITOR INPUT READY:\n")
                    f.write(f"  Content Pieces: {len(scraped_contents)}\n")
                    f.write(f"  Total Content Length: {total_content_length} characters\n")
                    f.write(f"  Average Content Length: {round(total_content_length/len(scraped_contents), 0)} chars\n")
                    f.write(f"  Content Source: Web Scraping\n\n")

                    f.write("CONTENT SUMMARY FOR EDITOR:\n")
                    f.write("-" * 30 + "\n")
                    for i, content in enumerate(scraped_contents, 1):
                        f.write(f"{i}. {content['title'][:70]}...\n")
                        f.write(f"   URL: {content['url']}\n")
                        f.write(f"   Method: {content['scraping_method']}\n")
                        f.write(f"   Length: {len(content['content'])} chars\n")
                        f.write(f"   Quality: {content.get('quality_score', 'N/A')}\n")
                        f.write(f"   Preview: {content['content'][:150]}...\n\n")

                else:
                    f.write(f"❌ NO CONTENT AVAILABLE FOR EDITOR\n")
                    f.write("All scraping attempts failed. Possible issues:\n")
                    f.write("  - Website blocking/protection\n")
                    f.write("  - Network connectivity\n")
                    f.write("  - Scraping strategy selection\n")
                    f.write("  - Content parsing errors\n\n")

                # FINAL STATISTICS & TIMING
                total_time = round(time.time() - total_start_time, 2)

                f.write("FINAL STATISTICS\n")
                f.write("=" * 40 + "\n")
                f.write(f"Query Processing: {analysis_time}s\n")
                f.write(f"Web Search: {search_time}s\n")
                f.write(f"Intelligent Scraping: {scraping_time}s\n")
                f.write(f"Total Pipeline Time: {total_time}s\n\n")

                f.write(f"Search Results Found: {len(search_results)}\n")
                f.write(f"Scraping Success Rate: {round((successful_scrapes/max(len(search_results),1))*100, 1)}%\n")
                f.write(f"Total Content Scraped: {total_content_length} characters\n")
                f.write(f"Content Ready for Editor: {'YES' if successful_scrapes > 0 else 'NO'}\n\n")

                # Performance insights
                if successful_scrapes > 0:
                    chars_per_second = round(total_content_length / scraping_time, 0)
                    f.write(f"PERFORMANCE INSIGHTS:\n")
                    f.write(f"  Content scraped per second: {chars_per_second} chars/s\n")
                    f.write(f"  Average time per successful scrape: {round(scraping_time/successful_scrapes, 2)}s\n")

                # Final summary
                f.write("\n" + "=" * 80 + "\n")
                f.write("WEB SCRAPING TEST COMPLETED\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Search Success: {'✅' if len(search_results) > 0 else '❌'}\n")
                f.write(f"Scraping Success: {'✅' if successful_scrapes > 0 else '❌'}\n")
                f.write(f"Content for Editor: {'✅ READY' if successful_scrapes > 0 else '❌ NONE'}\n")
                f.write(f"Total Time: {total_time}s\n")
                f.write("Pipeline: ✅ Query → ✅ Search → ✅ Scrape → ✅ Analysis\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during web scraping test: {str(e)}\n")
                logger.error(f"Error during web scraping test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            self._send_results_to_admin(bot, filepath, restaurant_query, successful_scrapes if 'successful_scrapes' in locals() else 0)

        return filepath

    def _send_results_to_admin(self, bot, file_path: str, query: str, successful_count: int):
        """Send web scraping test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🌐 <b>Web Scraping Test Results</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"✅ Successful scrapes: {successful_count}\n"
                f"🔧 Pipeline: query → search → scrape (database bypassed)\n"
                f"🎯 Focus: Web search and scraping analysis\n\n"
                f"{'✅ Web content scraped successfully' if successful_count > 0 else '❌ No content scraped - check logs'}\n\n"
                f"📄 Complete web scraping analysis with FULL content attached."
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
                    caption=f"🌐 Web scrape test: {query}"
                )

            logger.info("Successfully sent web scraping test results to admin")

        except Exception as e:
            logger.error(f"Failed to send web scraping results to admin: {e}")


# Convenience function for telegram bot integration
def add_web_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_wscrape command to the Telegram bot
    This function is deprecated since commands are handled directly in telegram_bot.py
    Keeping for backward compatibility.
    """
    logger.info("Note: web scrape test commands are now handled directly in telegram_bot.py")
    pass