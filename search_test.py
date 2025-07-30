# search_test.py - Fixed for proper BraveSearchAgent.search() method call
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
                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. {query}\n")

                f.write(f"\nSearch Parameters:\n")
                f.write(f"  Destination: {query_analysis.get('destination', 'Unknown')}\n")
                f.write(f"  Primary: {query_analysis.get('primary_search_parameters', [])}\n")
                f.write(f"  Secondary: {query_analysis.get('secondary_filter_parameters', [])}\n\n")

                # Step 2: Raw Search Results - FIXED TO USE CORRECT METHOD SIGNATURE
                f.write("STEP 2: RAW SEARCH RESULTS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                raw_results = []

                # Execute each search query with FIXED method call
                for i, search_query in enumerate(search_queries, 1):
                    f.write(f"Search Query {i}: {search_query}\n")

                    # FIXED: Call search method with correct parameters
                    # The method signature is: search(queries, max_retries=3, retry_delay=2, enable_ai_filtering=True)
                    query_results = self.search_agent.search(
                        [search_query],  # queries parameter (must be a list)
                        enable_ai_filtering=True  # explicitly pass enable_ai_filtering
                    )
                    raw_results.extend(query_results)

                    f.write(f"  URLs found: {len(query_results)}\n")

                    # Show first 5 URLs from this query
                    for j, result in enumerate(query_results[:5], 1):
                        f.write(f"    {j}. {result.get('url', 'Unknown URL')}\n")
                        f.write(f"       Title: {result.get('title', 'No title')[:80]}...\n")

                    if len(query_results) > 5:
                        f.write(f"    ... and {len(query_results) - 5} more URLs\n")
                    f.write("\n")

                search_time = round(time.time() - start_time, 2)

                # Remove duplicates for analysis
                unique_urls = set()
                unique_results = []
                for result in raw_results:
                    url = result.get('url', '')
                    if url and url not in unique_urls:
                        unique_urls.add(url)
                        unique_results.append(result)

                f.write(f"Search Summary:\n")
                f.write(f"  Processing Time: {search_time}s\n")
                f.write(f"  Total URLs (with duplicates): {len(raw_results)}\n")
                f.write(f"  Unique URLs: {len(unique_results)}\n\n")

                # Step 3: AI Filtering Analysis
                f.write("STEP 3: AI FILTERING ANALYSIS\n")
                f.write("-" * 40 + "\n")

                # Check if search agent has evaluation stats
                evaluation_stats = getattr(self.search_agent, 'evaluation_stats', {})

                if evaluation_stats:
                    f.write("AI Evaluation Statistics:\n")
                    for key, value in evaluation_stats.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

                # Step 3.5: Detailed AI Evaluation Results
                f.write("STEP 3.5: DETAILED AI EVALUATION RESULTS\n")
                f.write("-" * 40 + "\n")

                passed_urls = []
                failed_urls = []

                for i, result in enumerate(unique_results, 1):
                    url = result.get('url', 'Unknown')
                    title = result.get('title', 'No title')
                    ai_eval = result.get('ai_evaluation', {})

                    f.write(f"\nURL {i}: {url}\n")
                    f.write(f"Title: {title[:100]}...\n")

                    if ai_eval:
                        f.write(f"✅ AI EVALUATION:\n")
                        f.write(f"   Passed Filter: {ai_eval.get('passed_filter', 'N/A')}\n")
                        f.write(f"   Is Restaurant List: {ai_eval.get('is_restaurant_list', 'N/A')}\n")
                        f.write(f"   Content Quality: {ai_eval.get('content_quality', 'N/A')}\n")
                        f.write(f"   Restaurant Count: {ai_eval.get('restaurant_count', 'N/A')}\n")
                        f.write(f"   Reasoning: {ai_eval.get('reasoning', 'No reasoning provided')}\n")

                        if ai_eval.get('passed_filter', False):
                            passed_urls.append(result)
                            f.write(f"   ✅ VERDICT: PASSED - Will be sent to scraper\n")
                        else:
                            failed_urls.append(result)
                            f.write(f"   ❌ VERDICT: FILTERED OUT\n")
                    else:
                        f.write(f"❌ NO AI EVALUATION (filtered out at domain level or error)\n")
                        failed_urls.append(result)

                # Step 4: Final Results Summary
                f.write("\nSTEP 4: FINAL FILTERING SUMMARY\n")
                f.write("-" * 40 + "\n")

                f.write(f"URLs that PASSED filtering (will be scraped): {len(passed_urls)}\n")
                if passed_urls:
                    for i, result in enumerate(passed_urls, 1):
                        ai_eval = result.get('ai_evaluation', {})
                        quality = ai_eval.get('content_quality', 0)
                        f.write(f"  {i}. Quality: {quality:.2f} - {result.get('url', 'Unknown')}\n")

                f.write(f"\nURLs that FAILED filtering: {len(failed_urls)}\n")
                if failed_urls:
                    for i, result in enumerate(failed_urls[:10], 1):  # Show first 10 only
                        ai_eval = result.get('ai_evaluation', {})
                        reason = ai_eval.get('reasoning', 'Domain filtered or error')[:60]
                        f.write(f"  {i}. Reason: {reason}... - {result.get('url', 'Unknown')}\n")
                    if len(failed_urls) > 10:
                        f.write(f"  ... and {len(failed_urls) - 10} more filtered URLs\n")

                # Step 5: Performance Analysis
                f.write("\nSTEP 5: PERFORMANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")

                model_used = evaluation_stats.get('model_used', 'Unknown')
                total_evaluated = evaluation_stats.get('total_evaluated', 0)
                passed_filter = evaluation_stats.get('passed_filter', 0)
                failed_filter = evaluation_stats.get('failed_filter', 0)
                domain_filtered = evaluation_stats.get('domain_filtered', 0)
                evaluation_errors = evaluation_stats.get('evaluation_errors', 0)

                f.write(f"AI Model Used: {model_used}\n")
                f.write(f"Total URLs evaluated by AI: {total_evaluated}\n")
                f.write(f"Passed AI filter: {passed_filter}\n")
                f.write(f"Failed AI filter: {failed_filter}\n")
                f.write(f"Domain filtered (pre-AI): {domain_filtered}\n")
                f.write(f"Evaluation errors: {evaluation_errors}\n")

                if total_evaluated > 0:
                    pass_rate = (passed_filter / total_evaluated) * 100
                    f.write(f"AI Filter Pass Rate: {pass_rate:.1f}%\n")

                # Cost analysis for GPT-4o-mini
                if model_used == 'gpt-4o-mini':
                    cost_saved = evaluation_stats.get('estimated_cost_saved', 0.0)
                    f.write(f"Estimated Cost Savings vs GPT-4o: ${cost_saved:.3f}\n")

                # Final summary
                f.write("\n" + "=" * 80 + "\n")
                f.write("SEARCH FILTERING TEST SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Search queries generated: {len(search_queries)}\n")
                f.write(f"Total URLs found: {len(unique_results)}\n")
                f.write(f"URLs passed filtering: {len(passed_urls)}\n")
                f.write(f"Filter effectiveness: {len(failed_urls)}/{len(unique_results)} filtered out\n")
                f.write(f"Ready for scraping: {'Yes' if passed_urls else 'No - all URLs filtered'}\n")

                # Send results to admin if available
                if bot and self.admin_chat_id:
                    await self._send_results_to_admin(bot, filepath, restaurant_query, len(passed_urls))

            except Exception as e:
                f.write(f"\n❌ ERROR DURING TEST: {str(e)}\n")
                logger.error(f"Error during search filtering test: {e}")

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
            )

            if model_used == 'gpt-4o-mini':
                summary += f"💰 Cost Savings: ${cost_saved:.3f}\n"

            summary += (
                f"🔧 Orchestrator: Singleton instance\n"
                f"🔍 Focus: URL filtering with AI reasoning\n\n"
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