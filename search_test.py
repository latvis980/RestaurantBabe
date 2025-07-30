# search_test.py - Clean test following actual pipeline business logic
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
    Clean search test that follows the actual pipeline business logic:

    PIPELINE: Query Analysis → Web Search (with AI filtering) → Show Results

    This test ONLY tests the web search portion of the pipeline:
    - Query analysis to generate search queries + destination
    - Web search with AI filtering
    - Results analysis for destination relevance

    Does NOT involve:
    - Database search (different test)
    - Content evaluation (that's for database results)
    - Scraping (separate test)
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get components from orchestrator for consistency
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent

    async def test_web_search_pipeline(self, restaurant_query: str, bot=None) -> str:
        """
        Test the web search pipeline: Query Analysis → Web Search → AI Filtering → Analysis

        This simulates what happens when the ContentEvaluationAgent determines
        that database results are insufficient and web search is needed.
        """
        logger.info(f"Testing web search pipeline for: {restaurant_query}")

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

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_search_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WEB SEARCH PIPELINE TEST\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Pipeline: Query Analysis → Web Search → AI Filtering\n")
            f.write(f"AI Model: {getattr(self.config, 'SEARCH_EVALUATION_MODEL', 'gpt-4o')}\n")
            f.write(f"Purpose: Test web search when database is insufficient\n\n")

            try:
                # STEP 1: Query Analysis (same as actual pipeline)
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                # Extract key data
                destination = query_analysis.get('destination', 'Unknown')
                search_queries = query_analysis.get('search_queries', [])
                primary_params = query_analysis.get('primary_search_parameters', [])
                secondary_params = query_analysis.get('secondary_filter_parameters', [])

                f.write(f"Processing Time: {analysis_time}s\n")
                f.write(f"Destination Detected: {destination}\n")
                f.write(f"Primary Parameters: {primary_params}\n")
                f.write(f"Secondary Parameters: {secondary_params}\n\n")

                f.write(f"Generated Search Queries ({len(search_queries)}):\n")
                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. {query}\n")
                f.write("\n")

                if destination == "Unknown":
                    f.write("❌ No destination detected - web search would fail\n")
                    f.write("In actual pipeline, this would trigger fallback behavior\n")
                    return filepath

                if not search_queries:
                    f.write("❌ No search queries generated - analysis failed\n")
                    return filepath

                # STEP 2: Web Search with AI Filtering (current actual implementation)
                f.write("STEP 2: WEB SEARCH WITH AI FILTERING\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()

                # This is the actual call that happens in the pipeline currently
                # Note: destination filtering is not yet implemented in the search method
                search_results = self.search_agent.search(
                    search_queries,
                    enable_ai_filtering=True
                )

                search_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {search_time}s\n")
                f.write(f"URLs Found After AI Filtering: {len(search_results)}\n\n")

                # STEP 3: Filtering Analysis
                f.write("STEP 3: AI FILTERING ANALYSIS\n")
                f.write("-" * 40 + "\n")

                evaluation_stats = getattr(self.search_agent, 'evaluation_stats', {})

                f.write("AI Filtering Statistics:\n")
                f.write(f"  Total URLs Evaluated by AI: {evaluation_stats.get('total_evaluated', 0)}\n")
                f.write(f"  Passed AI Filter: {evaluation_stats.get('passed_filter', 0)}\n")
                f.write(f"  Failed AI Filter: {evaluation_stats.get('failed_filter', 0)}\n")
                f.write(f"  Domain Pre-filtered: {evaluation_stats.get('domain_filtered', 0)}\n")
                f.write(f"  Evaluation Errors: {evaluation_stats.get('evaluation_errors', 0)}\n\n")

                if evaluation_stats.get('total_evaluated', 0) > 0:
                    pass_rate = (evaluation_stats.get('passed_filter', 0) / evaluation_stats.get('total_evaluated', 1)) * 100
                    f.write(f"AI Filter Success Rate: {pass_rate:.1f}%\n\n")

                # STEP 4: Destination Relevance Analysis (manual check since not implemented yet)
                f.write("STEP 4: DESTINATION RELEVANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")

                destination_relevant = 0
                destination_irrelevant = 0
                unclear_relevance = 0

                f.write(f"Analyzing results for destination relevance to: {destination}\n")
                f.write("Note: Automatic destination filtering not yet implemented in search agent\n\n")

                for i, result in enumerate(search_results[:20], 1):  # Show first 20
                    url = result.get('url', 'Unknown')
                    title = result.get('title', 'No title')
                    description = result.get('description', 'No description')
                    ai_eval = result.get('ai_evaluation', {})

                    # Manual destination relevance check
                    combined_text = f"{title} {description}".lower()
                    destination_lower = destination.lower()

                    # Check if destination is mentioned
                    if destination_lower in combined_text:
                        relevance = "✅ RELEVANT"
                        destination_relevant += 1
                    elif any(city in combined_text for city in ['paris', 'london', 'new york', 'tokyo', 'madrid', 'rome', 'berlin', 'amsterdam'] if city != destination_lower):
                        relevance = "❌ WRONG DESTINATION"
                        destination_irrelevant += 1
                    else:
                        relevance = "❓ UNCLEAR"
                        unclear_relevance += 1

                    f.write(f"{i}. {relevance}\n")
                    f.write(f"   URL: {url}\n")
                    f.write(f"   Title: {title[:80]}...\n")

                    if ai_eval:
                        quality = ai_eval.get('content_quality', 0)
                        reasoning = ai_eval.get('reasoning', '')[:60]
                        f.write(f"   AI Quality: {quality:.2f} - {reasoning}...\n")

                    f.write("\n")

                # STEP 5: Summary and Recommendations
                f.write("STEP 5: PIPELINE TEST SUMMARY\n")
                f.write("-" * 40 + "\n")

                f.write(f"Original Query: {restaurant_query}\n")
                f.write(f"Detected Destination: {destination}\n")
                f.write(f"Search Queries Generated: {len(search_queries)}\n")
                f.write(f"Final URLs After AI Filters: {len(search_results)}\n\n")

                f.write("Destination Relevance Breakdown (Manual Analysis):\n")
                f.write(f"  ✅ Relevant to {destination}: {destination_relevant}\n")
                f.write(f"  ❌ Wrong destination: {destination_irrelevant}\n")
                f.write(f"  ❓ Unclear relevance: {unclear_relevance}\n\n")

                # Calculate effectiveness
                total_analyzed = destination_relevant + destination_irrelevant + unclear_relevance
                if total_analyzed > 0:
                    relevant_rate = (destination_relevant / total_analyzed) * 100
                    f.write(f"Destination Relevance Rate: {relevant_rate:.1f}%\n\n")

                # Pipeline recommendations
                f.write("RECOMMENDATIONS:\n")
                if destination_irrelevant > destination_relevant:
                    f.write("❗ ISSUE: More irrelevant destinations than relevant ones\n")
                    f.write("   → Implement destination filtering in BraveSearchAgent.search() method\n")
                    f.write("   → Add destination parameter to search method signature\n")
                    f.write("   → Filter results based on destination keywords\n\n")

                if len(search_results) >= 5 and destination_relevant >= 3:
                    f.write("✅ READY: Sufficient relevant URLs for scraping\n")
                elif len(search_results) > 0:
                    f.write("⚠️ LIMITED: May need query refinement or destination filtering\n")
                else:
                    f.write("❌ FAILED: No results - pipeline would fail\n")

                f.write(f"\nNext Pipeline Step: Scraping {len(search_results)} URLs\n")

                # Cost analysis
                model_used = evaluation_stats.get('model_used', 'Unknown')
                if model_used == 'gpt-4o-mini':
                    cost_saved = evaluation_stats.get('estimated_cost_saved', 0.0)
                    f.write(f"Cost Optimization: ${cost_saved:.3f} saved using {model_used}\n")

                # Implementation notes
                f.write("\nIMPLEMENTATION NOTES:\n")
                f.write("- ContentEvaluationAgent is working correctly (evaluates database results)\n")
                f.write("- BraveSearchAgent needs destination filtering implementation\n")
                f.write("- Current pipeline: Query → Database → Content Eval → Web Search → Scrape\n")
                f.write("- This test focuses only on the Web Search portion\n")

                # Send results to admin if available
                if bot and self.admin_chat_id:
                    await self._send_results_to_admin(bot, filepath, restaurant_query, len(search_results), destination)

            except Exception as e:
                f.write(f"\n❌ ERROR DURING WEB SEARCH TEST: {str(e)}\n")
                logger.error(f"Error during web search test: {e}")
                import traceback
                f.write(f"Full error: {traceback.format_exc()}\n")

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, result_count: int, destination: str):
        """Send web search test results to admin via Telegram"""
        try:
            evaluation_stats = getattr(self.search_agent, 'evaluation_stats', {})

            summary = (
                f"🔍 <b>Web Search Pipeline Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🎯 Destination: <code>{destination}</code>\n"
                f"📊 Final URLs: {result_count}\n"
                f"🤖 AI Model: {evaluation_stats.get('model_used', 'Unknown')}\n"
                f"✅ Passed Filter: {evaluation_stats.get('passed_filter', 0)}\n"
                f"❌ Failed Filter: {evaluation_stats.get('failed_filter', 0)}\n\n"
                f"🔧 Pipeline: Query Analysis → Web Search → AI Filtering\n"
                f"🎯 Focus: Testing web search component only\n\n"
                f"{'✅ Ready for scraping' if result_count >= 3 else '⚠️ Limited results'}\n\n"
                f"📄 Detailed analysis attached."
            )

            bot.send_message(self.admin_chat_id, summary, parse_mode='HTML')

            # Send the results file
            with open(file_path, 'rb') as f:
                bot.send_document(
                    self.admin_chat_id,
                    f,
                    caption=f"🔍 Web search pipeline test: {query}"
                )

            logger.info("Successfully sent web search test results to admin")

        except Exception as e:
            logger.error(f"Failed to send web search results to admin: {e}")


# Convenience function for backward compatibility
def add_search_test_command(bot, config, orchestrator):
    """Add the /test_search command to the Telegram bot"""
    logger.info("Note: search test commands are now handled directly in telegram_bot.py")
    pass