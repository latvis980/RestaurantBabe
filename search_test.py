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

                # Step 2: Raw Search Results
                f.write("STEP 2: RAW SEARCH RESULTS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                raw_results = []

                # Execute each search query
                for i, search_query in enumerate(search_queries, 1):
                    f.write(f"Search Query {i}: {search_query}\n")

                    # Get raw results for this query
                    query_results = self.search_agent.search([search_query])
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
                        f.write(f"   Restaurant Count: {ai_eval.get('restaurant_count', 'N/A')}\n")
                        f.write(f"   Content Quality: {ai_eval.get('content_quality', 'N/A'):.2f}\n")
                        f.write(f"   Reasoning: {ai_eval.get('reasoning', 'No reasoning provided')}\n")
                    else:
                        f.write(f"❌ NO AI EVALUATION DATA (may have been filtered at domain level)\n")

                    f.write("-" * 60 + "\n")

                f.write("\n")

                # Analyze domain distribution
                domain_counts = {}
                filtered_urls = []

                for result in unique_results:
                    url = result.get('url', '')
                    if url:
                        # Extract domain
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(url).netloc.lower()
                            if domain.startswith('www.'):
                                domain = domain[4:]
                            domain_counts[domain] = domain_counts.get(domain, 0) + 1
                        except:
                            pass

                        # Check if this URL made it through filtering
                        ai_eval = result.get('ai_evaluation', {})
                        if ai_eval.get('passed_filter', False) or not ai_eval:
                            filtered_urls.append(result)

                f.write("Domain Analysis:\n")
                for domain, count in sorted(domain_counts.items()):
                    f.write(f"  {domain}: {count} URLs\n")

                f.write(f"\nFiltering Results:\n")
                f.write(f"  Before filtering: {len(unique_results)} URLs\n")
                f.write(f"  After AI filtering: {len(filtered_urls)} URLs\n")
                f.write(f"  Filtered out: {len(unique_results) - len(filtered_urls)} URLs\n\n")

                # Step 4: Final URLs for Scraping
                f.write("STEP 4: FINAL URLs FOR SCRAPING\n")
                f.write("-" * 40 + "\n")
                f.write("URLs that would be sent to scraper:\n")

                for i, result in enumerate(filtered_urls, 1):
                    url = result.get('url', 'Unknown URL')
                    title = result.get('title', 'No title')
                    f.write(f"  {i}. {url}\n")
                    f.write(f"     Title: {title[:100]}...\n")

                if len(filtered_urls) > 20:
                    f.write(f"  ... and {len(filtered_urls) - 20} more URLs\n")

                final_count = len(filtered_urls)

                # Cost Analysis Section
                f.write(f"\nCOST OPTIMIZATION ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                model_used = evaluation_stats.get('model_used', 'Unknown')
                cost_saved = evaluation_stats.get('estimated_cost_saved', 0.0)
                f.write(f"  AI Model Used: {model_used}\n")
                f.write(f"  Total Evaluations: {evaluation_stats.get('total_evaluated', 0)}\n")

                if model_used == 'gpt-4o-mini':
                    f.write(f"  💰 Estimated Cost Savings: ${cost_saved:.3f} vs GPT-4o\n")
                    f.write(f"  📊 Cost Reduction: ~95% vs GPT-4o\n")
                else:
                    f.write(f"  💡 Switch to gpt-4o-mini for 95% cost reduction\n")

                # Configuration Info
                f.write(f"\nCONFIGURATION INFO:\n")
                f.write(f"  Excluded domains: {getattr(self.config, 'EXCLUDED_RESTAURANT_SOURCES', [])}\n")
                f.write(f"  Search count limit: {getattr(self.config, 'BRAVE_SEARCH_COUNT', 15)}\n")
                f.write(f"  AI model: {getattr(self.config, 'SEARCH_EVALUATION_MODEL', getattr(self.config, 'OPENAI_MODEL', 'Unknown'))}\n")

                # Overall Summary
                f.write(f"\nOVERALL SUMMARY:\n")
                f.write(f"  Query analysis: {analysis_time}s\n")
                f.write(f"  Search execution: {search_time}s\n")
                f.write(f"  Total processing: {analysis_time + search_time}s\n")
                f.write(f"  Final URLs for scraping: {final_count}\n")
                f.write(f"  Success rate: {(final_count / max(len(unique_results), 1) * 100):.1f}%\n")
                f.write(f"  AI Filtering efficiency: {evaluation_stats.get('passed_filter', 0)}/{evaluation_stats.get('total_evaluated', 0)} passed\n\n")

                f.write("=" * 80 + "\n")
                f.write("SEARCH FILTERING TEST COMPLETED\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during search test: {str(e)}\n")
                logger.error(f"Error during search test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, final_count if 'final_count' in locals() else 0)

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