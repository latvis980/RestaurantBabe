# search_test.py - Comprehensive search and filtering process tester
import asyncio
import time
import tempfile
import os
import threading
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SearchTest:
    """
    Comprehensive test to debug the complete search and filtering process:
    - What raw search results are returned by Brave
    - Which URLs pass domain filtering (excluded sources)
    - Which URLs pass AI/keyword filtering 
    - Detailed logging of filtering decisions
    - Final list of URLs that will be passed to scraper
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize components for manual testing
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)

    async def test_search_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run complete search and filtering process with detailed logging

        Args:
            restaurant_query: The restaurant query to test (e.g., "best brunch in Lisbon")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing search process for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESTAURANT SEARCH & FILTERING PROCESS TEST\n")
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
                f.write(f"Search Queries Generated: {len(query_analysis.get('search_queries', []))}\n\n")

                for i, query in enumerate(query_analysis.get('search_queries', []), 1):
                    f.write(f"  {i}. {query}\n")
                f.write("\n")

                # Step 2: Raw Search Results
                f.write("STEP 2: RAW BRAVE SEARCH RESULTS\n")
                f.write("-" * 40 + "\n")

                # Get raw results before any filtering
                raw_results = []
                search_queries = query_analysis.get('search_queries', [])

                for query in search_queries:
                    f.write(f"\nSearching for: '{query}'\n")
                    f.write("-" * 20 + "\n")

                    try:
                        # Execute raw search (this calls Brave API directly)
                        raw_search_data = self.search_agent._execute_search(query)

                        if raw_search_data and "web" in raw_search_data and "results" in raw_search_data["web"]:
                            query_results = raw_search_data["web"]["results"]
                            f.write(f"Raw results returned: {len(query_results)}\n")

                            for i, result in enumerate(query_results, 1):
                                url = result.get("url", "N/A")
                                title = result.get("title", "N/A")
                                f.write(f"  {i}. {title}\n")
                                f.write(f"     URL: {url}\n")
                                f.write(f"     Description: {result.get('description', 'N/A')[:100]}...\n\n")

                            raw_results.extend(query_results)
                        else:
                            f.write("No results returned from Brave API\n")

                    except Exception as e:
                        f.write(f"Error searching for '{query}': {str(e)}\n")

                f.write(f"\nTOTAL RAW RESULTS: {len(raw_results)}\n\n")

                # Step 3: Domain Filtering
                f.write("STEP 3: DOMAIN FILTERING\n")
                f.write("-" * 40 + "\n")
                f.write(f"Excluded domains: {self.config.EXCLUDED_RESTAURANT_SOURCES}\n")
                f.write(f"Video platforms excluded: {list(self.search_agent.video_platforms)}\n\n")

                domain_filtered_results = []
                domain_rejected_results = []

                for result in raw_results:
                    url = result.get("url", "")
                    title = result.get("title", "")

                    # Check excluded domains
                    excluded_domain = None
                    for excluded in self.config.EXCLUDED_RESTAURANT_SOURCES:
                        if excluded in url:
                            excluded_domain = excluded
                            break

                    # Check video platforms
                    is_video_platform = self.search_agent._is_video_platform(url)

                    if excluded_domain:
                        domain_rejected_results.append({
                            "url": url,
                            "title": title,
                            "reason": f"Excluded domain: {excluded_domain}"
                        })
                    elif is_video_platform:
                        domain_rejected_results.append({
                            "url": url,
                            "title": title,
                            "reason": "Video/social platform"
                        })
                    else:
                        domain_filtered_results.append(result)

                f.write(f"PASSED domain filtering: {len(domain_filtered_results)}\n")
                f.write(f"REJECTED by domain filtering: {len(domain_rejected_results)}\n\n")

                # List rejected URLs
                if domain_rejected_results:
                    f.write("REJECTED URLs:\n")
                    for rejected in domain_rejected_results:
                        f.write(f"  ❌ {rejected['title']}\n")
                        f.write(f"     URL: {rejected['url']}\n")
                        f.write(f"     Reason: {rejected['reason']}\n\n")

                # Step 4: Content/AI Filtering
                f.write("STEP 4: CONTENT/AI FILTERING\n")
                f.write("-" * 40 + "\n")

                ai_filtered_results = []
                ai_rejected_results = []

                # Reset search agent stats for this test
                self.search_agent.evaluation_stats = {
                    "total_evaluated": 0,
                    "passed_filter": 0,
                    "failed_filter": 0,
                    "evaluation_errors": 0,
                    "domain_filtered": 0
                }

                for result in domain_filtered_results:
                    url = result.get("url", "")
                    title = result.get("title", "")
                    description = result.get("description", "")

                    # Use the same evaluation method as search agent
                    evaluation = self.search_agent._basic_keyword_evaluation(url, title, description)

                    if evaluation.get("passed_filter", False):
                        result["ai_evaluation"] = evaluation
                        ai_filtered_results.append(result)
                        f.write(f"  ✅ PASSED: {title}\n")
                        f.write(f"     URL: {url}\n")
                        f.write(f"     Score: {evaluation.get('content_quality', 0):.2f}\n")
                        f.write(f"     Reasoning: {evaluation.get('reasoning', 'N/A')}\n\n")
                    else:
                        ai_rejected_results.append({
                            "url": url,
                            "title": title,
                            "evaluation": evaluation
                        })
                        f.write(f"  ❌ REJECTED: {title}\n")
                        f.write(f"     URL: {url}\n")
                        f.write(f"     Score: {evaluation.get('content_quality', 0):.2f}\n")
                        f.write(f"     Reasoning: {evaluation.get('reasoning', 'N/A')}\n\n")

                f.write(f"\nFINAL FILTERING RESULTS:\n")
                f.write(f"PASSED AI/content filtering: {len(ai_filtered_results)}\n")
                f.write(f"REJECTED by AI/content filtering: {len(ai_rejected_results)}\n\n")

                # Step 5: Final Summary
                f.write("STEP 5: FINAL SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Original query: {restaurant_query}\n")
                f.write(f"Search queries generated: {len(search_queries)}\n")
                f.write(f"Raw Brave results: {len(raw_results)}\n")
                f.write(f"After domain filtering: {len(domain_filtered_results)}\n")
                f.write(f"After AI filtering: {len(ai_filtered_results)}\n\n")

                f.write("FINAL URLs THAT WILL BE SCRAPED:\n")
                for i, result in enumerate(ai_filtered_results, 1):
                    f.write(f"  {i}. {result.get('title', 'N/A')}\n")
                    f.write(f"     URL: {result.get('url', 'N/A')}\n")
                    f.write(f"     Quality Score: {result.get('ai_evaluation', {}).get('content_quality', 0):.2f}\n\n")

                # Step 6: Filtering Statistics
                f.write("STEP 6: DETAILED FILTERING STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Domain exclusions:\n")
                for domain in self.config.EXCLUDED_RESTAURANT_SOURCES:
                    count = sum(1 for r in domain_rejected_results if domain in r.get('reason', ''))
                    f.write(f"  - {domain}: {count} URLs rejected\n")

                video_count = sum(1 for r in domain_rejected_results if 'Video/social' in r.get('reason', ''))
                f.write(f"  - Video/social platforms: {video_count} URLs rejected\n\n")

                f.write(f"AI/Content filtering statistics:\n")
                f.write(f"  - Total evaluated: {len(domain_filtered_results)}\n")
                f.write(f"  - Passed filter: {len(ai_filtered_results)}\n")
                f.write(f"  - Failed filter: {len(ai_rejected_results)}\n")
                f.write(f"  - Success rate: {len(ai_filtered_results) / max(len(domain_filtered_results), 1) * 100:.1f}%\n\n")

            except Exception as e:
                f.write(f"\n❌ ERROR during search test: {str(e)}\n")
                logger.error(f"Error during search test: {e}")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, ai_filtered_results)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, final_results: List[Dict]):
        """Send search test results to admin via Telegram"""
        try:
            # Create summary message
            summary = (
                f"🔍 <b>Search Test Completed</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🎯 Final URLs: {len(final_results)}\n\n"
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
                    caption=f"🔍 Search test results for: {query}"
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
                "🔍 <b>Search Process Test</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/test_search [restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/test_search best brunch in Lisbon</code>\n"
                "<code>/test_search romantic restaurants Paris</code>\n"
                "<code>/test_search family pizza Rome</code>\n\n"
                "This tests the complete search and filtering process:\n"
                "• Raw Brave search results\n"
                "• Domain filtering (excluded sources)\n"
                "• AI/content filtering decisions\n"
                "• Final URLs that pass to scraper\n"
                "• Detailed filtering statistics\n\n"
                "📄 Results are saved to a detailed file."
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
            f"🔍 <b>Starting search process test...</b>\n\n"
            f"📝 Query: <code>{restaurant_query}</code>\n\n"
            "This will analyze:\n"
            "1️⃣ Query analysis and search terms\n"
            "2️⃣ Raw Brave search results\n"
            "3️⃣ Domain filtering decisions\n"
            "4️⃣ AI/content filtering decisions\n"
            "5️⃣ Final URL list for scraping\n\n"
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
                logger.info(f"Search test completed: {results_path}")

            except Exception as e:
                logger.error(f"Error in search test: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Search test failed for '{restaurant_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_test, daemon=True)
        thread.start()

    logger.info("Search test command added to bot: /test_search")