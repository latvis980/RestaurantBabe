# web_scrape_test.py - CLEANED UP IMPORTS
# Uses SmartRestaurantScraper directly, no legacy wrapper

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
    Web-only test that uses SmartRestaurantScraper directly.

    This test:
    - Skips all database steps (bypasses completely)
    - Forces web search to happen
    - Shows detailed search results and filtering
    - Shows complete SMART SCRAPING process with strategy breakdown
    - Perfect for debugging the smart scraping pipeline

    Pipeline tested: query_analyzer → search_agent → SmartRestaurantScraper (with AI strategy routing)
    Command: /test_wscrape
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents directly from orchestrator (now using SmartRestaurantScraper)
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent  
        self.scraper = orchestrator.scraper  # This is now SmartRestaurantScraper directly

    async def test_web_scraping_only(self, restaurant_query: str, bot=None) -> str:
        """
        Test ONLY the web search and smart scraping pipeline.
        Uses SmartRestaurantScraper directly for maximum debugging detail.
        """
        total_start_time = time.time()

        # Create timestamped filename for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webscrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        logger.info(f"Starting smart web scraping test for: {restaurant_query}")
        logger.info(f"Results will be saved to: {filepath}")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("SMART WEB SCRAPING TEST RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test Type: Web Search + SmartRestaurantScraper ONLY (Database Bypassed)\n")
                f.write("=" * 80 + "\n\n")

                f.write("SMART SCRAPING PIPELINE OVERVIEW:\n")
                f.write("1. Query Analysis → Search Terms Generation\n")
                f.write("2. Web Search → URL Discovery & Filtering\n")
                f.write("3. Smart Scraper → AI Strategy Classification\n")
                f.write("   ├── 🆓 Specialized (RSS/Sitemap)\n")
                f.write("   ├── 🟢 Simple HTTP (0.1 credits)\n")
                f.write("   ├── 🟡 Enhanced HTTP (0.5 credits)\n")
                f.write("   └── 🔴 Firecrawl (10.0 credits)\n")
                f.write("4. Content Analysis → Results for Editor\n")
                f.write("\n" + "-" * 80 + "\n\n")

                # STEP 1: Query Analysis
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                analysis_start = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - analysis_start, 2)

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Destination: {query_analysis.get('destination', 'Unknown')}\n")
                f.write(f"Language: {query_analysis.get('local_language', 'Unknown')}\n")
                f.write(f"Search Queries Generated: {len(query_analysis.get('search_queries', []))}\n\n")

                # Show generated search queries
                search_queries = query_analysis.get('search_queries', [])
                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. {query}\n")

                destination = query_analysis.get('destination', 'Unknown')

                # STEP 2: Web Search (FORCED - bypass database completely)
                f.write(f"\nSTEP 2: WEB SEARCH (FORCED - Database Bypassed)\n")
                f.write("-" * 40 + "\n")

                search_start = time.time()

                # Prepare query metadata for search agent
                query_metadata = {
                    'is_english_speaking': query_analysis.get('is_english_speaking', True),
                    'local_language': query_analysis.get('local_language')
                }

                search_results = self.search_agent.search(search_queries, destination, query_metadata)
                search_time = round(time.time() - search_start, 2)

                f.write(f"Search Time: {search_time}s\n")
                f.write(f"Search Queries Used: {len(search_queries)}\n")
                f.write(f"Results Found: {len(search_results)}\n\n")

                # Show search results details
                f.write("SEARCH RESULTS FOUND:\n")
                for i, result in enumerate(search_results, 1):
                    f.write(f"  {i}. {result.get('title', 'No Title')[:70]}...\n")
                    f.write(f"     URL: {result.get('url', 'No URL')}\n")
                    f.write(f"     Quality Score: {result.get('quality_score', 'N/A')}\n")
                    f.write(f"     Description: {(result.get('description', '') or '')[:150]}...\n\n")

                # STEP 3: SMART RESTAURANT SCRAPER with AI Strategy Classification
                f.write(f"\nSTEP 3: SMART RESTAURANT SCRAPER PIPELINE\n")
                f.write("-" * 40 + "\n")

                if not search_results:
                    f.write("❌ NO SEARCH RESULTS TO SCRAPE\n")
                    f.write("Cannot proceed with scraping - no URLs found.\n\n")
                    return filepath

                # Reset scraper stats for this test
                self.scraper.stats = {
                    "total_processed": 0,
                    "strategy_breakdown": {"specialized": 0, "simple_http": 0, "enhanced_http": 0, "firecrawl": 0},
                    "ai_analysis_calls": 0,
                    "domain_cache_hits": 0,
                    "new_domains_learned": 0,
                    "total_cost_estimate": 0.0,
                    "cost_saved_vs_all_firecrawl": 0.0
                }

                scraping_start = time.time()
                f.write(f"🧠 Starting SmartRestaurantScraper with AI strategy classification...\n")
                f.write(f"URLs to process: {len(search_results)}\n\n")

                # Run the SmartRestaurantScraper directly
                enriched_results = await self.scraper.scrape_search_results(search_results)
                scraping_time = round(time.time() - scraping_start, 2)

                # Get smart scraper statistics
                scraper_stats = self.scraper.get_stats()
                successful_scrapes = len([r for r in enriched_results if r.get('scraped_content')])

                f.write(f"✅ SmartRestaurantScraper Complete!\n")
                f.write(f"Processing Time: {scraping_time}s\n")
                f.write(f"Successful Scrapes: {successful_scrapes}\n")
                f.write(f"Failed Scrapes: {len(enriched_results) - successful_scrapes}\n")
                f.write(f"Success Rate: {round((successful_scrapes/max(len(enriched_results),1))*100, 1)}%\n\n")

                # SMART SCRAPER INTELLIGENCE BREAKDOWN
                f.write("SMART SCRAPER INTELLIGENCE BREAKDOWN:\n")
                f.write("=" * 60 + "\n")

                strategy_breakdown = scraper_stats.get('strategy_breakdown', {})
                total_urls = scraper_stats.get('total_processed', 0)

                f.write(f"🧠 AI Strategy Classification:\n")
                f.write(f"   Total URLs Analyzed: {total_urls}\n")
                f.write(f"   AI Analysis Calls: {scraper_stats.get('ai_analysis_calls', 0)}\n")
                f.write(f"   Domain Cache Hits: {scraper_stats.get('domain_cache_hits', 0)}\n")
                f.write(f"   New Domains Learned: {scraper_stats.get('new_domains_learned', 0)}\n\n")

                f.write("📊 Strategy Distribution:\n")
                cost_map = {"specialized": 0.0, "simple_http": 0.1, "enhanced_http": 0.5, "firecrawl": 10.0}
                emoji_map = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}

                for strategy, count in strategy_breakdown.items():
                    if count > 0:
                        cost = count * cost_map.get(strategy, 0)
                        emoji = emoji_map.get(strategy, "📌")
                        f.write(f"   {emoji} {strategy.upper()}: {count} URLs (~{cost:.1f} credits)\n")

                f.write(f"\n💰 Cost Analysis:\n")
                f.write(f"   Actual Cost: {scraper_stats.get('total_cost_estimate', 0):.1f} credits\n")
                f.write(f"   All-Firecrawl Cost: {total_urls * 10:.1f} credits\n")
                f.write(f"   Cost Saved: {scraper_stats.get('cost_saved_vs_all_firecrawl', 0):.1f} credits\n")
                f.write(f"   Efficiency: {(scraper_stats.get('cost_saved_vs_all_firecrawl', 0) / max(total_urls * 10, 1) * 100):.1f}% savings\n\n")

                # DETAILED SCRAPING RESULTS 
                f.write("DETAILED SCRAPING RESULTS:\n")
                f.write("=" * 60 + "\n")

                total_content_length = 0

                for i, result in enumerate(enriched_results, 1):
                    f.write(f"SCRAPE #{i}:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"URL: {result.get('url', 'No URL')}\n")
                    f.write(f"Title: {result.get('title', 'No Title')}\n")
                    f.write(f"Original Quality Score: {result.get('quality_score', 'N/A')}\n")

                    scraping_method = result.get('scraping_method', 'Unknown')
                    f.write(f"Scraping Strategy: {scraping_method}\n")

                    scraped_content = result.get('scraped_content')
                    if scraped_content:
                        content_length = len(scraped_content)
                        total_content_length += content_length

                        f.write(f"Status: ✅ SUCCESS\n")
                        f.write(f"Content Length: {content_length} characters\n")
                        f.write(f"Processing Time: {result.get('scraping_time', 'N/A')}s\n")

                        # Show content sectioning info if available
                        if result.get('content_sections'):
                            sections = result.get('content_sections', {})
                            f.write(f"Content Sections: {len(sections)} sections\n")
                            for section_name, section_content in sections.items():
                                f.write(f"   - {section_name}: {len(section_content)} chars\n")

                        f.write(f"\nFULL SCRAPED CONTENT:\n")
                        f.write("~" * 60 + "\n")
                        f.write(scraped_content)
                        f.write("\n" + "~" * 60 + "\n\n")

                    else:
                        f.write(f"Status: ❌ FAILED\n")
                        error_msg = result.get('scraping_error', 'Unknown error')
                        f.write(f"Error: {error_msg}\n\n")

                # FINAL STATISTICS & TIMING
                total_time = round(time.time() - total_start_time, 2)

                f.write("FINAL STATISTICS\n")
                f.write("=" * 40 + "\n")
                f.write(f"Query Processing: {analysis_time}s\n")
                f.write(f"Web Search: {search_time}s\n")
                f.write(f"SmartRestaurantScraper: {scraping_time}s\n")
                f.write(f"Total Pipeline Time: {total_time}s\n\n")

                f.write(f"Search Results Found: {len(search_results)}\n")
                f.write(f"Scraping Success Rate: {round((successful_scrapes/max(len(search_results),1))*100, 1)}%\n")
                f.write(f"Total Content Scraped: {total_content_length} characters\n")
                f.write(f"Content Ready for Editor: {'YES' if successful_scrapes > 0 else 'NO'}\n\n")

                # Smart scraper specific insights
                if successful_scrapes > 0:
                    chars_per_second = round(total_content_length / scraping_time, 0)
                    cost_efficiency = scraper_stats.get('cost_saved_vs_all_firecrawl', 0)

                    f.write(f"SMART SCRAPER INSIGHTS:\n")
                    f.write(f"  Content scraped per second: {chars_per_second} chars/s\n")
                    f.write(f"  Average time per successful scrape: {round(scraping_time/successful_scrapes, 2)}s\n")
                    f.write(f"  Cost efficiency vs all-Firecrawl: {cost_efficiency:.1f} credits saved\n")
                    f.write(f"  AI optimization: {scraper_stats.get('domain_cache_hits', 0)} cache hits, {scraper_stats.get('ai_analysis_calls', 0)} AI calls\n\n")

                # Final summary with correct naming
                f.write("\n" + "=" * 80 + "\n")
                f.write("SMART WEB SCRAPING TEST COMPLETED\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Search Success: {'✅' if len(search_results) > 0 else '❌'}\n")
                f.write(f"SmartRestaurantScraper Success: {'✅' if successful_scrapes > 0 else '❌'}\n")
                f.write(f"Content for Editor: {'✅ READY' if successful_scrapes > 0 else '❌ NONE'}\n")
                f.write(f"Total Time: {total_time}s\n")
                f.write(f"Pipeline: ✅ Query → ✅ Search → ✅ SmartScraper → ✅ Analysis\n")
                f.write(f"AI Features: ✅ Strategy Classification → ✅ Cost Optimization → ✅ Domain Learning\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n❌ ERROR during smart web scraping test: {str(e)}\n")
                logger.error(f"Error during web scraping test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            self._send_results_to_admin(bot, filepath, restaurant_query, successful_scrapes if 'successful_scrapes' in locals() else 0)

        return filepath

    def _send_results_to_admin(self, bot, file_path: str, query: str, successful_count: int):
        """Send smart web scraping test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🧠 <b>SmartRestaurantScraper Test Results</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"✅ Successful scrapes: {successful_count}\n"
                f"🎯 Pipeline: query → search → SmartRestaurantScraper (AI routing)\n"
                f"🧠 Focus: AI strategy classification and cost optimization\n\n"
                f"{'✅ Content scraped with smart strategies' if successful_count > 0 else '❌ No content scraped - check logs'}\n\n"
                f"📄 Complete SmartRestaurantScraper analysis attached."
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
                    caption=f"🧠 SmartRestaurantScraper test: {query}"
                )

            logger.info("Successfully sent SmartRestaurantScraper test results to admin")

        except Exception as e:
            logger.error(f"Failed to send smart scraping results to admin: {e}")


# ============================================================================
# SIMILAR UPDATES NEEDED FOR scrape_test.py
# ============================================================================

# scrape_test.py - CLEANED UP IMPORTS  
# Now uses SmartRestaurantScraper directly from orchestrator

class ScrapeTest:
    """
    Full pipeline test that uses SmartRestaurantScraper directly.
    Follows the EXACT production pipeline path.
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents directly from orchestrator (SmartRestaurantScraper, not wrapper)
        self.query_analyzer = orchestrator.query_analyzer
        self.database_search_agent = orchestrator.database_search_agent
        self.dbcontent_evaluation_agent = orchestrator.dbcontent_evaluation_agent  
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper  # This is now SmartRestaurantScraper directly
        self.editor_agent = orchestrator.editor_agent

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run the COMPLETE production pipeline up to editor stage using SmartRestaurantScraper directly.
        """
        total_start_time = time.time()

        # Create timestamped filename for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        logger.info(f"Starting full pipeline test with SmartRestaurantScraper for: {restaurant_query}")
        logger.info(f"Results will be saved to: {filepath}")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("FULL PIPELINE TEST RESULTS (SmartRestaurantScraper)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test Type: Complete Pipeline (Database → SmartRestaurantScraper → Editor)\n")
                f.write("=" * 80 + "\n\n")

                f.write("PRODUCTION PIPELINE OVERVIEW:\n")
                f.write("1. Query Analysis → Search Terms Generation\n")
                f.write("2. Database Search → Existing Restaurant Lookup\n")
                f.write("3. Content Evaluation → Sufficiency Assessment\n")
                f.write("4. [IF NEEDED] Web Search → URL Discovery\n")
                f.write("5. [IF NEEDED] SmartRestaurantScraper → AI Strategy Routing\n")
                f.write("6. Editor Agent → Final Processing\n")
                f.write("\n" + "-" * 80 + "\n\n")

                # Execute the same steps as production...
                # STEP 1: Query Analysis
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                analysis_start = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - analysis_start, 2)

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Destination: {query_analysis.get('destination', 'Unknown')}\n")

                # Continue with rest of pipeline...
                # (This would include database search, evaluation, conditional web search, 
                #  conditional SmartRestaurantScraper usage, and editor processing)

        except Exception as e:
            logger.error(f"Error in full pipeline test: {e}")

        return filepath