# search_test.py - FOCUSED on Brave URLs and filtering process only
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
    Focused test: Shows exactly what URLs Brave returns and what gets filtered out
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize only what we need for search and filtering
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)

    async def test_search_process(self, restaurant_query: str, bot=None) -> str:
        """
        Test ONLY the search and filtering process

        Args:
            restaurant_query: The restaurant query to test
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing search and filtering for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_filtering_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BRAVE SEARCH + FILTERING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n\n")

            try:
                # STEP 1: Query Analysis
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                query_analysis = self.query_analyzer.analyze(restaurant_query)
                search_queries = query_analysis.get('search_queries', [])

                f.write(f"Search Queries Generated: {len(search_queries)}\n")
                for i, query in enumerate(search_queries, 1):
                    f.write(f"  {i}. '{query}'\n")
                f.write(f"Destination: {query_analysis.get('destination', 'Unknown')}\n\n")

                # STEP 2: Raw Brave Search Results
                f.write("STEP 2: RAW BRAVE SEARCH RESULTS\n")
                f.write("-" * 40 + "\n")

                all_raw_results = []

                for query_idx, query in enumerate(search_queries, 1):
                    f.write(f"\nQuery {query_idx}: '{query}'\n")
                    f.write("-" * 20 + "\n")

                    # Execute raw Brave search
                    try:
                        raw_response = self.search_agent._execute_search(query)
                        brave_results = raw_response.get('web', {}).get('results', [])

                        f.write(f"Brave API returned: {len(brave_results)} results\n\n")

                        if brave_results:
                            f.write("RAW URLs from Brave:\n")
                            for i, result in enumerate(brave_results, 1):
                                url = result.get('url', 'No URL')
                                title = result.get('title', 'No title')
                                f.write(f"  {i:2d}. {url}\n")
                                f.write(f"      Title: {title[:80]}{'...' if len(title) > 80 else ''}\n")

                            all_raw_results.extend(brave_results)
                        else:
                            f.write("❌ No results returned by Brave API\n")

                    except Exception as e:
                        f.write(f"❌ Error calling Brave API: {e}\n")

                f.write(f"\nTOTAL RAW RESULTS: {len(all_raw_results)}\n\n")

                # STEP 3: Filtering Process Analysis
                f.write("STEP 3: FILTERING PROCESS ANALYSIS\n")
                f.write("-" * 40 + "\n")

                if all_raw_results:
                    # Reset search agent stats
                    self.search_agent.evaluation_stats = {
                        "total_evaluated": 0,
                        "passed_filter": 0,
                        "failed_filter": 0,
                        "evaluation_errors": 0,
                        "domain_filtered": 0
                    }
                    self.search_agent.filtered_urls = []

                    # Apply filtering step by step
                    f.write("FILTERING RESULTS:\n\n")

                    # Domain filtering
                    domain_passed = []
                    domain_failed = []

                    for result in all_raw_results:
                        url = result.get('url', '')
                        title = result.get('title', 'No title')

                        # Check domain exclusions
                        excluded_domain = self.search_agent._should_exclude_domain(url)
                        is_video = self.search_agent._is_video_platform(url)

                        if excluded_domain:
                            domain_failed.append({
                                'url': url,
                                'title': title,
                                'reason': 'Excluded domain'
                            })
                        elif is_video:
                            domain_failed.append({
                                'url': url,
                                'title': title,
                                'reason': 'Video/social platform'
                            })
                        else:
                            domain_passed.append(result)

                    f.write(f"DOMAIN FILTERING RESULTS:\n")
                    f.write(f"✅ Passed domain filter: {len(domain_passed)}\n")
                    f.write(f"❌ Failed domain filter: {len(domain_failed)}\n\n")

                    # Show what was filtered out by domain
                    if domain_failed:
                        f.write("DOMAIN FILTERED URLs:\n")
                        for i, failed in enumerate(domain_failed, 1):
                            f.write(f"  {i:2d}. ❌ {failed['reason']}\n")
                            f.write(f"      URL: {failed['url']}\n")
                            f.write(f"      Title: {failed['title'][:80]}{'...' if len(failed['title']) > 80 else ''}\n\n")

                    # AI filtering (if any URLs passed domain filtering)
                    if domain_passed:
                        f.write(f"AI FILTERING RESULTS:\n")
                        f.write(f"URLs sent to AI for evaluation: {len(domain_passed)}\n\n")

                        # Apply AI filtering
                        ai_filtered_results = asyncio.run(self.search_agent._apply_ai_filtering(domain_passed))

                        f.write(f"✅ Passed AI filter: {len(ai_filtered_results)}\n")
                        f.write(f"❌ Failed AI filter: {len(domain_passed) - len(ai_filtered_results)}\n\n")

                        # Show what passed AI filtering
                        if ai_filtered_results:
                            f.write("AI APPROVED URLs:\n")
                            for i, result in enumerate(ai_filtered_results, 1):
                                url = result.get('url', 'No URL')
                                title = result.get('title', 'No title')
                                ai_eval = result.get('ai_evaluation', {})
                                score = ai_eval.get('content_quality', 0)
                                reasoning = ai_eval.get('reasoning', 'No reasoning')

                                f.write(f"  {i:2d}. ✅ APPROVED (Score: {score:.2f})\n")
                                f.write(f"      URL: {url}\n")
                                f.write(f"      Title: {title[:80]}{'...' if len(title) > 80 else ''}\n")
                                f.write(f"      AI Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}\n\n")

                        # Show what was rejected by AI
                        ai_rejected = len(domain_passed) - len(ai_filtered_results)
                        if ai_rejected > 0:
                            f.write("AI REJECTED URLs:\n")
                            # Get recently filtered URLs from search agent
                            recent_filtered = self.search_agent.filtered_urls[-ai_rejected:] if hasattr(self.search_agent, 'filtered_urls') else []

                            for i, filtered in enumerate(recent_filtered, 1):
                                f.write(f"  {i:2d}. ❌ REJECTED\n")
                                f.write(f"      URL: {filtered.get('url', 'Unknown')}\n")
                                f.write(f"      Reason: {filtered.get('reason', 'Unknown')}\n\n")
                    else:
                        f.write("❌ NO URLs passed domain filtering - AI filtering skipped\n\n")

                # STEP 4: Summary and Analysis
                f.write("STEP 4: SUMMARY & ANALYSIS\n")
                f.write("-" * 40 + "\n")

                final_count = len(ai_filtered_results) if 'ai_filtered_results' in locals() else 0

                f.write(f"FILTERING PIPELINE SUMMARY:\n")
                f.write(f"Raw Brave results: {len(all_raw_results)}\n")
                f.write(f"After domain filtering: {len(domain_passed) if 'domain_passed' in locals() else 0}\n")
                f.write(f"Final URLs for scraping: {final_count}\n\n")

                if final_count == 0:
                    f.write("❌ PROBLEM IDENTIFIED:\n")
                    if len(all_raw_results) == 0:
                        f.write("- Brave API returned no results\n")
                        f.write("- Check API key, quotas, and search queries\n")
                    elif len(domain_passed) == 0:
                        f.write("- All results filtered out by domain filtering\n")
                        f.write("- Brave only returned TripAdvisor/Yelp/social media\n")
                        f.write("- Consider adjusting search queries or excluded domains\n")
                    else:
                        f.write("- Results passed domain filter but failed AI filtering\n")
                        f.write("- AI filtering threshold may be too strict (currently 0.5)\n")
                        f.write("- Check AI evaluation reasoning above\n")
                else:
                    f.write("✅ SUCCESS:\n")
                    f.write(f"- {final_count} URLs ready for scraping\n")
                    f.write("- Filtering process working correctly\n")

                f.write(f"\nCONFIGURATION:\n")
                f.write(f"Excluded domains: {getattr(self.config, 'EXCLUDED_RESTAURANT_SOURCES', [])}\n")
                f.write(f"Search count per query: {getattr(self.config, 'BRAVE_SEARCH_COUNT', 15)}\n")
                f.write(f"AI model: {getattr(self.config, 'OPENAI_MODEL', 'Unknown')}\n")

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
            # Create summary message
            summary = (
                f"🔍 <b>Search Filtering Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🎯 Final URLs: {final_count}\n"
                f"🔍 Focus: Brave URLs + Filtering analysis\n\n"
                f"{'✅ URLs found for scraping' if final_count > 0 else '❌ All URLs filtered out'}\n\n"
                f"📄 Detailed filtering analysis attached."
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


def add_search_test_command(bot, config, orchestrator):
    """
    Add the /test_search command to the Telegram bot
    Focused on search URLs and filtering analysis
    """

    search_tester = SearchTest(config, orchestrator)

    @bot.message_handler(commands=['test_search'])
    def handle_test_search(message):
        """Handle /test_search command - focused on URL filtering"""

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
                "🔍 <b>Search URL Filtering Test</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/test_search [restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/test_search best wine bars in rome</code>\n"
                "<code>/test_search romantic restaurants Paris</code>\n\n"
                "This shows exactly:\n"
                "• What URLs Brave returns\n"
                "• Which URLs get filtered out and why\n"
                "• Domain filtering results\n"
                "• AI filtering decisions\n\n"
                "📄 Results are saved to a detailed file.\n\n"
                "🎯 <b>Focus:</b> URL filtering analysis only."
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
            f"🔍 <b>Starting URL filtering analysis...</b>\n\n"
            f"📝 Query: <code>{restaurant_query}</code>\n\n"
            "Analyzing:\n"
            "1️⃣ Raw Brave search results\n"
            "2️⃣ Domain filtering process\n"
            "3️⃣ AI filtering decisions\n"
            "4️⃣ Final URLs for scraping\n\n"
            "🎯 <b>Focus:</b> Which URLs pass/fail filtering\n"
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
                logger.info(f"Search filtering test completed: {results_path}")

            except Exception as e:
                logger.error(f"Error in search filtering test: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Search filtering test failed for '{restaurant_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_test, daemon=True)
        thread.start()

    logger.info("Search filtering test command added to bot: /test_search")