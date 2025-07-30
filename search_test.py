# search_test.py - Updated with detailed AI reasoning display
import asyncio
import time
import tempfile
import os
import threading
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SearchTest:
    """
    Test focused on search URL filtering and analysis:
    - What search URLs are found
    - Which URLs pass AI filtering with detailed reasoning
    - Final URLs sent to scraper
    - Detailed filtering analysis with cost optimization tracking

    Updated to use orchestrator singleton pattern and show GPT-4o-mini optimization
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator  # Now receives the singleton instance
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get components from orchestrator to ensure consistency
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent

    async def test_search_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run search and filtering analysis with detailed AI reasoning

        Args:
            restaurant_query: The restaurant query to test
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing search filtering process for: {restaurant_query}")

        # Reset evaluation stats for clean test
        self.search_agent.evaluation_stats = {
            "total_evaluated": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "failed_destination": 0,  # Include destination filtering stats
            "evaluation_errors": 0,
            "domain_filtered": 0,
            "model_used": getattr(self.config, 'SEARCH_EVALUATION_MODEL', 'gpt-4o'),
            "estimated_cost_saved": 0.0
        }
        self.search_agent.filtered_urls = []
        logger.info("🔄 Reset evaluation statistics for fresh test")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESTAURANT SEARCH FILTERING TEST (Enhanced with AI Reasoning)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Orchestrator: Singleton instance\n")
            f.write(f"AI Model: {getattr(self.config, 'SEARCH_EVALUATION_MODEL', 'gpt-4o')} (cost-optimized)\n\n")

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
                f.write(f"  Primary: {query_analysis.get('primary_search_parameters', [])}\n")
                f.write(f"  Secondary: {query_analysis.get('secondary_filter_parameters', [])}\n\n")

                # Step 2: Raw Search Results
                f.write("STEP 2: RAW SEARCH RESULTS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                raw_results = []

                # Execute each search query with destination
                for i, search_query in enumerate(search_queries, 1):
                    f.write(f"Search Query {i}: {search_query}\n")

                    # Get raw results for this query - pass destination parameter
                    query_results = self.search_agent.search([search_query], destination)
                    raw_results.extend(query_results)

                    f.write(f"  URLs found: {len(query_results)}\n")

                    # Show first 5 URLs from this query
                    for j, result in enumerate(query_results[:5], 1):
                        f.write(f"    {j}. {result.get('title', 'No Title')[:60]}...\n")
                        f.write(f"       URL: {result.get('url', 'No URL')}\n")
                        f.write(f"       Description: {(result.get('description', 'No Description') or '')[:100]}...\n")

                        # Show AI evaluation if available
                        ai_eval = result.get('ai_evaluation', {})
                        if ai_eval:
                            f.write(f"       AI Score: {ai_eval.get('content_quality', 0):.2f} | ")
                            f.write(f"Dest Match: {ai_eval.get('destination_match', 'N/A')} | ")
                            f.write(f"Pass: {ai_eval.get('passed_filter', False)}\n")
                            f.write(f"       Reasoning: {ai_eval.get('reasoning', 'No reasoning')[:80]}...\n")
                        f.write("\n")

                    f.write(f"\n")

                search_time = round(time.time() - start_time, 2)

                # Step 3: Detailed AI Filtering Analysis
                f.write("STEP 3: AI FILTERING ANALYSIS\n")
                f.write("-" * 40 + "\n")

                filtering_stats = getattr(self.search_agent, 'evaluation_stats', {})
                filtered_urls = getattr(self.search_agent, 'filtered_urls', [])

                f.write(f"AI Evaluation Model: {filtering_stats.get('model_used', 'Unknown')}\n")
                f.write(f"Total URLs Evaluated: {filtering_stats.get('total_evaluated', 0)}\n")
                f.write(f"Passed Filter: {filtering_stats.get('passed_filter', 0)}\n")
                f.write(f"Failed Filter: {filtering_stats.get('failed_filter', 0)}\n")
                f.write(f"Failed Destination Check: {filtering_stats.get('failed_destination', 0)}\n")
                f.write(f"Domain Filtered: {filtering_stats.get('domain_filtered', 0)}\n")
                f.write(f"Evaluation Errors: {filtering_stats.get('evaluation_errors', 0)}\n")

                if filtering_stats.get('model_used') == 'gpt-4o-mini':
                    cost_saved = filtering_stats.get('estimated_cost_saved', 0.0)
                    f.write(f"Estimated Cost Savings vs GPT-4o: ${cost_saved:.3f}\n")

                f.write(f"\nFiltered URLs ({len(filtered_urls)} total):\n")
                for i, url in enumerate(filtered_urls[:10], 1):  # Show first 10 filtered URLs
                    f.write(f"  {i}. {url}\n")

                if len(filtered_urls) > 10:
                    f.write(f"  ... and {len(filtered_urls) - 10} more filtered URLs\n")

                # Step 4: Final Results Summary
                f.write(f"\nSTEP 4: FINAL RESULTS SUMMARY\n")
                f.write("-" * 40 + "\n")

                f.write(f"Search Processing Time: {search_time}s\n")
                f.write(f"Total URLs Found: {len(raw_results)}\n")
                f.write(f"URLs Passed Filtering: {len([r for r in raw_results if r.get('ai_evaluation', {}).get('passed_filter', False)])}\n")
                f.write(f"Final URLs for Scraping: {len(raw_results)}\n\n")

                # Show detailed analysis for each passed URL
                passed_results = [r for r in raw_results if r.get('ai_evaluation', {}).get('passed_filter', True)]

                f.write("DETAILED ANALYSIS OF PASSED URLs:\n")
                f.write("-" * 40 + "\n")

                for i, result in enumerate(passed_results, 1):
                    f.write(f"\n{i}. {result.get('title', 'No Title')}\n")
                    f.write(f"URL: {result.get('url', 'No URL')}\n")
                    f.write(f"Description: {result.get('description', 'No Description')}\n")

                    ai_eval = result.get('ai_evaluation', {})
                    if ai_eval:
                        f.write(f"AI Evaluation:\n")
                        f.write(f"  Content Quality: {ai_eval.get('content_quality', 0):.2f}/1.0\n")
                        f.write(f"  Is Restaurant List: {ai_eval.get('is_restaurant_list', False)}\n")
                        f.write(f"  Restaurant Count: {ai_eval.get('restaurant_count', 0)}\n")
                        f.write(f"  Destination Match: {ai_eval.get('destination_match', 'N/A')}\n")
                        f.write(f"  Passed Filter: {ai_eval.get('passed_filter', False)}\n")
                        f.write(f"  Reasoning: {ai_eval.get('reasoning', 'No reasoning provided')}\n")
                    else:
                        f.write("AI Evaluation: Not available (possibly domain filtered)\n")

                    f.write("-" * 60 + "\n")

                # Overall timing
                f.write(f"\nOVERALL TIMING:\n")
                f.write(f"  Query Analysis: {analysis_time}s\n")
                f.write(f"  Search + AI Filtering: {search_time}s\n")
                f.write(f"  Total: {analysis_time + search_time}s\n\n")

                f.write("=" * 80 + "\n")
                f.write("SEARCH FILTERING TEST COMPLETED SUCCESSFULLY\n")
                f.write(f"Pipeline: Query Analysis → Web Search → AI Content Filtering\n")
                f.write(f"Destination Validation: Enabled for {destination}\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during search filtering test: {str(e)}\n")
                logger.error(f"Error during search filtering test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, len(raw_results) if 'raw_results' in locals() else 0)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, final_count: int):
        """Send search filtering test results to admin via Telegram"""
        try:
            # Get evaluation stats for summary
            evaluation_stats = getattr(self.search_agent, 'evaluation_stats', {})
            model_used = evaluation_stats.get('model_used', 'Unknown')
            cost_saved = evaluation_stats.get('estimated_cost_saved', 0.0)

            # Create summary message
            summary = (
                f"🔍 <b>Search Filtering Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🎯 Final URLs: {final_count}\n"
                f"🤖 AI Model: {model_used}\n"
                f"📊 Evaluations: {evaluation_stats.get('passed_filter', 0)}/{evaluation_stats.get('total_evaluated', 0)} passed\n"
                f"🏙 Destination filtered: {evaluation_stats.get('failed_destination', 0)}\n"
            )

            if model_used == 'gpt-4o-mini':
                summary += f"💰 Cost Savings: ${cost_saved:.3f}\n"

            summary += (
                f"🔧 Orchestrator: Singleton instance\n"
                f"🔍 Focus: URL filtering with AI reasoning + destination validation\n\n"
                f"{'✅ URLs found for scraping' if final_count > 0 else '❌ All URLs filtered out'}\n\n"
                f"📄 Detailed filtering analysis with AI reasoning attached."
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
                    caption=f"🔍 Search filtering analysis: {query}"
                )

            logger.info("Successfully sent search filtering test results to admin")

        except Exception as e:
            logger.error(f"Failed to send search filtering results to admin: {e}")


# Convenience function for backward compatibility
def add_search_test_command(bot, config, orchestrator):
    """
    Add the /test_search command to the Telegram bot
    This function is now deprecated since commands are handled directly in telegram_bot.py
    Keeping for backward compatibility.
    """
    logger.info("Note: search test commands are now handled directly in telegram_bot.py")
    pass