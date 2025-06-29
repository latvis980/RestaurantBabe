# search_test.py - Updated for simplified search agent
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
    Test the simplified search process:
    - What search results are found
    - Domain filtering decisions
    - AI filtering decisions
    - Final URLs for scraping
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize pipeline components
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)

    async def test_search_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run complete search process and dump results to file

        Args:
            restaurant_query: The restaurant query to test (e.g., "best brunch in Lisbon")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing simplified search process for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SIMPLIFIED RESTAURANT SEARCH PROCESS TEST\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n\n")

            try:
                # Step 1: Analyze Query
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                query_analysis = self.query_analyzer.analyze(restaurant_query)
                analysis_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {analysis_time}s\n")
                search_queries = query_analysis.get('search_queries', [])
                f.write(f"Search Queries Generated: {len(search_queries)}\n")

                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. {query}\n")

                f.write(f"Destination: {query_analysis.get('destination', 'Unknown')}\n")
                f.write(f"Keywords: {query_analysis.get('keywords_for_analysis', [])}\n\n")

                # Step 2: Execute Search with Simplified Filtering
                f.write("STEP 2: SIMPLIFIED SEARCH PROCESS\n")
                f.write("-" * 40 + "\n")

                # Reset search agent stats
                self.search_agent.evaluation_stats = {
                    "total_evaluated": 0,
                    "passed_filter": 0,
                    "failed_filter": 0,
                    "evaluation_errors": 0,
                    "domain_filtered": 0
                }
                self.search_agent.filtered_urls = []

                # Execute search
                start_time = time.time()
                search_results = self.search_agent.search(
                    queries=search_queries,
                    enable_ai_filtering=True
                )
                search_time = round(time.time() - start_time, 2)

                f.write(f"Total Search Time: {search_time}s\n")
                f.write(f"Final Results Count: {len(search_results)}\n\n")

                # Step 3: Detailed Analysis
                f.write("STEP 3: DETAILED FILTERING ANALYSIS\n")
                f.write("-" * 40 + "\n")

                stats = self.search_agent.evaluation_stats
                f.write(f"Domain Filtering:\n")
                f.write(f"  - URLs filtered by domain/video platforms: {stats['domain_filtered']}\n\n")

                f.write(f"AI Filtering:\n")
                f.write(f"  - Total URLs evaluated by AI: {stats['total_evaluated']}\n")
                f.write(f"  - Passed AI filter: {stats['passed_filter']}\n")
                f.write(f"  - Failed AI filter: {stats['failed_filter']}\n")
                f.write(f"  - Evaluation errors: {stats['evaluation_errors']}\n")

                if stats['total_evaluated'] > 0:
                    success_rate = (stats['passed_filter'] / stats['total_evaluated']) * 100
                    f.write(f"  - AI filtering success rate: {success_rate:.1f}%\n\n")

                # Step 4: Show Final URLs
                f.write("STEP 4: FINAL URLS FOR SCRAPING\n")
                f.write("-" * 40 + "\n")

                if search_results:
                    for i, result in enumerate(search_results, 1):
                        f.write(f"{i}. {result.get('title', 'N/A')}\n")
                        f.write(f"   URL: {result.get('url', 'N/A')}\n")

                        # Show AI evaluation details
                        ai_eval = result.get('ai_evaluation', {})
                        if ai_eval:
                            f.write(f"   AI Quality Score: {ai_eval.get('content_quality', 0):.2f}\n")
                            f.write(f"   Restaurant Count: {ai_eval.get('restaurant_count', 0)}\n")
                            f.write(f"   Reasoning: {ai_eval.get('reasoning', 'N/A')}\n")
                        f.write("\n")
                else:
                    f.write("No URLs passed the filtering process.\n\n")

                # Step 5: Show Recently Filtered URLs
                f.write("STEP 5: RECENTLY FILTERED URLS (DEBUG)\n")
                f.write("-" * 40 + "\n")

                filtered_urls = self.search_agent.filtered_urls[-10:]  # Last 10
                if filtered_urls:
                    for filtered in filtered_urls:
                        f.write(f"❌ {filtered.get('url', 'N/A')}\n")
                        f.write(f"   Reason: {filtered.get('reason', 'N/A')}\n\n")
                else:
                    f.write("No URLs were filtered out.\n\n")

                # Step 6: Configuration Info
                f.write("STEP 6: CURRENT CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Excluded Domains: {self.config.EXCLUDED_RESTAURANT_SOURCES}\n")
                f.write(f"Search Count per Query: {self.config.BRAVE_SEARCH_COUNT}\n")
                f.write(f"AI Model: {self.config.OPENAI_MODEL}\n")
                f.write(f"Video Platforms Blocked: {len(self.search_agent.video_platforms)}\n\n")

                # Step 7: Recommendations
                f.write("STEP 7: RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")

                if len(search_results) == 0:
                    f.write("⚠️  NO RESULTS - Possible Issues:\n")
                    f.write("   - AI filtering threshold too strict (currently 0.5)\n")
                    f.write("   - AI model not recognizing restaurant guides\n")
                    f.write("   - Domain filtering too aggressive\n")
                    f.write("   - Search queries not finding relevant content\n\n")
                elif len(search_results) < 5:
                    f.write("⚠️  LOW RESULTS - Consider:\n")
                    f.write("   - Lowering AI filtering threshold\n")
                    f.write("   - Increasing search count per query\n")
                    f.write("   - Adding more search query variations\n\n")
                else:
                    f.write("✅ GOOD RESULTS - System working properly\n\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during search test: {str(e)}\n")
                logger.error(f"Error during search test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, search_results)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, final_results: List[Dict]):
        """Send search test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🔍 <b>Simplified Search Test Completed</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🎯 Final URLs: {len(final_results)}\n"
                f"🤖 AI-only filtering active\n\n"
                f"📄 Detailed analysis attached."
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
                    caption=f"🔍 Simplified search test results for: {query}"
                )

            logger.info("Successfully sent search test results to admin")

        except Exception as e:
            logger.error(f"Failed to send search results to admin: {e}")


def add_search_test_command(bot, config, orchestrator):
    """
    Add the /test_search command to the Telegram bot
    Call this from telegram_bot.py main() function
    """

    search_tester = SearchTest(config, orchestrator)

    @bot.message_handler(commands=['test_search'])
    def handle_test_search(message):
        """Handle /test_search command"""

        user_id = message.from_user.id
        admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Check if user is admin
        if not admin_chat_id or str(user_id) != str(admin_chat_id):
            bot.reply_to(message, "❌ This command is only available to administrators.")
            return

        # Parse command
        command_text = message.text.strip()

        if len(command_text.split(None, 1)) < 2:
            help_text = (
                "🔍 <b>Simplified Search Process Test</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/test_search [restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/test_search best brunch in Lisbon</code>\n"
                "<code>/test_search romantic restaurants Paris</code>\n"
                "<code>/test_search family pizza Rome</code>\n\n"
                "This tests the simplified search process:\n"
                "• Raw Brave search results\n"
                "• Domain filtering (excluded sources + video platforms)\n"
                "• AI-only content filtering decisions\n"
                "• Final URLs for scraping\n"
                "• Configuration and recommendations\n\n"
                "📄 Results are saved to a detailed file.\n\n"
                "🤖 <b>Changes:</b> Removed keyword filtering, AI-only filtering now."
            )
            bot.reply_to(message, help_text, parse_mode='HTML')
            return

        # Extract query
        restaurant_query = command_text.split(None, 1)[1].strip()

        if not restaurant_query:
            bot.reply_to(message, "❌ Please provide a restaurant query to test.")
            return

        # Send confirmation
        bot.reply_to(
            message,
            f"🔍 <b>Starting simplified search test...</b>\n\n"
            f"📝 Query: <code>{restaurant_query}</code>\n\n"
            "New simplified process:\n"
            "1️⃣ Query analysis\n"
            "2️⃣ Domain filtering only\n"
            "3️⃣ AI-based content evaluation\n"
            "4️⃣ Results analysis\n\n"
            "⚡ Much faster - no keyword filtering!\n"
            "⏱ Please wait 1-2 minutes...",
            parse_mode='HTML'
        )

        # Run test in background
        def run_test():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                results_path = loop.run_until_complete(
                    search_tester.test_search_process(restaurant_query, bot)
                )

                loop.close()
                logger.info(f"Simplified search test completed: {results_path}")

            except Exception as e:
                logger.error(f"Error in simplified search test: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Simplified search test failed for '{restaurant_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_test, daemon=True)
        thread.start()