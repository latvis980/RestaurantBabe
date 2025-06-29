# scrape_test.py - Simple scraping process tester
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
    Simple test to see the complete scraping process:
    - What search results are found
    - Which ones get scraped successfully  
    - What content is actually scraped
    - What goes to list_analyzer
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize pipeline components
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run complete scraping process and dump results to file

        Args:
            restaurant_query: The restaurant query to test (e.g., "best brunch in Lisbon")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing scraping process for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESTAURANT SCRAPING PROCESS TEST\n")
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
                f.write(f"Search Queries Generated: {len(query_analysis.get('search_queries', []))}\n")

                for i, query in enumerate(query_analysis.get('search_queries', []), 1):
                    f.write(f"  {i}. {query}\n")

                f.write(f"Keywords for Analysis: {query_analysis.get('keywords_for_analysis', [])}\n")
                f.write(f"Destination: {query_analysis.get('destination', 'Unknown')}\n\n")

                # Step 2: Web Search
                f.write("STEP 2: WEB SEARCH RESULTS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                search_results = self.search_agent.search(query_analysis.get('search_queries', []))
                search_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {search_time}s\n")
                f.write(f"URLs Found: {len(search_results)}\n\n")

                for i, result in enumerate(search_results, 1):
                    f.write(f"URL {i}: {result.get('url', 'Unknown')}\n")
                    f.write(f"Title: {result.get('title', 'No title')}\n")
                    f.write(f"Description: {result.get('description', 'No description')[:100]}...\n\n")

                # Step 3: Scraping Process
                f.write("STEP 3: SCRAPING PROCESS\n")
                f.write("-" * 40 + "\n")

                start_time = time.time()
                scraped_results = await self.scraper.scrape_search_results(search_results)
                scraping_time = round(time.time() - start_time, 2)

                f.write(f"Processing Time: {scraping_time}s\n")
                f.write(f"URLs Scraped: {len(scraped_results)}\n\n")

                # Scraping Statistics
                successful_scrapes = [r for r in scraped_results if r.get("scraping_success")]
                failed_scrapes = [r for r in scraped_results if not r.get("scraping_success")]

                f.write("SCRAPING SUMMARY:\n")
                f.write(f"✓ Successful: {len(successful_scrapes)}\n")
                f.write(f"✗ Failed: {len(failed_scrapes)}\n\n")

                # Method breakdown
                methods = {}
                for result in successful_scrapes:
                    method = result.get("scraping_method", "unknown")
                    methods[method] = methods.get(method, 0) + 1

                f.write("SCRAPING METHODS USED:\n")
                for method, count in methods.items():
                    f.write(f"  {method}: {count} URLs\n")
                f.write("\n")

                # Get scraper stats
                scraper_stats = self.scraper.get_stats()
                f.write("INTELLIGENT SCRAPER STATS:\n")
                f.write(f"  Total Processed: {scraper_stats.get('total_processed', 0)}\n")
                f.write(f"  Specialized Used: {scraper_stats.get('specialized_used', 0)} (FREE)\n")
                f.write(f"  Simple HTTP Used: {scraper_stats.get('simple_http_used', 0)}\n")
                f.write(f"  Enhanced HTTP Used: {scraper_stats.get('enhanced_http_used', 0)}\n")
                f.write(f"  Firecrawl Used: {scraper_stats.get('firecrawl_used', 0)}\n")
                f.write(f"  AI Analysis Calls: {scraper_stats.get('ai_analysis_calls', 0)}\n")
                f.write(f"  Credits Saved: {scraper_stats.get('total_cost_saved', 0)}\n\n")

                # Step 4: Content Analysis
                f.write("STEP 4: SCRAPED CONTENT ANALYSIS\n")
                f.write("-" * 40 + "\n")

                total_content = 0
                total_restaurants = 0

                for result in successful_scrapes:
                    content_length = len(result.get("scraped_content", ""))
                    restaurants_found = len(result.get("restaurants_found", []))
                    total_content += content_length
                    total_restaurants += restaurants_found

                f.write(f"Total Content Length: {total_content:,} characters\n")
                f.write(f"Total Restaurants Found: {total_restaurants}\n")
                f.write(f"Average Content per Source: {total_content // max(len(successful_scrapes), 1):,} chars\n\n")

                # Step 5: Individual Source Details
                f.write("STEP 5: DETAILED SCRAPING RESULTS\n")
                f.write("-" * 40 + "\n\n")

                for i, result in enumerate(scraped_results, 1):
                    url = result.get("url", "Unknown")
                    success = result.get("scraping_success", False)
                    method = result.get("scraping_method", "unknown")
                    content = result.get("scraped_content", "")
                    restaurants = result.get("restaurants_found", [])

                    f.write(f"SOURCE {i}: {'✓' if success else '✗'}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Method: {method}\n")

                    if success:
                        f.write(f"Content Length: {len(content):,} characters\n")
                        f.write(f"Restaurants Found: {len(restaurants)}\n")

                        if restaurants:
                            f.write("Restaurant Names:\n")
                            for rest in restaurants[:5]:  # Show first 5
                                f.write(f"  • {rest}\n")
                            if len(restaurants) > 5:
                                f.write(f"  ... and {len(restaurants) - 5} more\n")

                        # Show content preview
                        f.write("\nCONTENT PREVIEW:\n")
                        f.write("-" * 20 + "\n")
                        preview = content[:500] if content else "No content"
                        f.write(preview)
                        if len(content) > 500:
                            f.write("\n... [TRUNCATED] ...")
                        f.write("\n")
                    else:
                        error_msg = result.get("error", "Unknown error")
                        f.write(f"Error: {error_msg}\n")

                    f.write("\n" + "=" * 60 + "\n\n")

                # Step 6: What Goes to List Analyzer
                f.write("STEP 6: CONTENT FOR LIST_ANALYZER\n")
                f.write("-" * 40 + "\n")

                analyzer_input = {
                    "search_results": scraped_results,
                    "keywords_for_analysis": query_analysis.get("keywords_for_analysis", []),
                    "primary_search_parameters": query_analysis.get("primary_search_parameters", []),
                    "secondary_filter_parameters": query_analysis.get("secondary_filter_parameters", []),
                    "destination": query_analysis.get("destination", "Unknown")
                }

                f.write("This is EXACTLY what gets passed to the list_analyzer AI:\n\n")
                f.write(f"Keywords: {analyzer_input['keywords_for_analysis']}\n")
                f.write(f"Primary Parameters: {analyzer_input['primary_search_parameters']}\n")
                f.write(f"Secondary Parameters: {analyzer_input['secondary_filter_parameters']}\n")
                f.write(f"Destination: {analyzer_input['destination']}\n")
                f.write(f"Number of Sources: {len(analyzer_input['search_results'])}\n")
                f.write(f"Sources with Content: {len([r for r in analyzer_input['search_results'] if r.get('scraped_content')])}\n\n")

                # Show each source's content that goes to analyzer
                f.write("COMPLETE CONTENT FOR AI ANALYSIS:\n")
                f.write("=" * 50 + "\n\n")

                for i, result in enumerate(analyzer_input['search_results'], 1):
                    if result.get('scraped_content'):
                        f.write(f"--- SOURCE {i} CONTENT ---\n")
                        f.write(f"URL: {result.get('url', 'Unknown')}\n")
                        f.write(f"Method: {result.get('scraping_method', 'unknown')}\n")
                        f.write("Content:\n")
                        f.write(result['scraped_content'])
                        f.write("\n\n--- END SOURCE {i} ---\n\n")

                # Final Summary
                f.write("FINAL SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Pipeline Time: {analysis_time + search_time + scraping_time}s\n")
                f.write(f"Content Ready for AI: {'YES' if total_content > 0 else 'NO'}\n")
                f.write(f"Quality Score: {len(successful_scrapes)}/{len(search_results)} sources scraped successfully\n")

            except Exception as e:
                f.write(f"\nERROR OCCURRED:\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Error Type: {type(e).__name__}\n")
                import traceback
                f.write(f"Traceback:\n{traceback.format_exc()}\n")

        logger.info(f"Scraping test completed. Results saved to: {filepath}")

        # Send to admin if bot provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str):
        """Send the results file to admin"""
        if not self.admin_chat_id:
            return

        try:
            # Send summary message
            summary = (
                f"🧪 <b>Scraping Process Test Complete</b>\n\n"
                f"📝 <b>Query:</b> <code>{query}</code>\n\n"
                f"📄 Complete process details attached.\n"
                f"Shows exactly what gets scraped and passed to list_analyzer."
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
                    caption=f"🧪 Scraping test results for: {query}"
                )

            logger.info("Successfully sent scraping test results to admin")

        except Exception as e:
            logger.error(f"Failed to send results to admin: {e}")


def add_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot
    Call this from telegram_bot.py main() function
    """

    scrape_tester = ScrapeTest(config, orchestrator)

    @bot.message_handler(commands=['test_scrape'])
    def handle_test_scrape(message):
        """Handle /test_scrape command"""

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
                "🧪 <b>Scraping Process Test</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/test_scrape [restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/test_scrape best brunch in Lisbon</code>\n"
                "<code>/test_scrape romantic restaurants Paris</code>\n"
                "<code>/test_scrape family pizza Rome</code>\n\n"
                "This runs the complete scraping process and shows:\n"
                "• Which search results are found\n"
                "• What gets scraped successfully\n"
                "• Exact content that goes to list_analyzer\n"
                "• Scraping method statistics\n\n"
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
            f"🧪 <b>Starting scraping process test...</b>\n\n"
            f"📝 Query: <code>{restaurant_query}</code>\n\n"
            "This will run the complete pipeline:\n"
            "1️⃣ Query analysis\n"
            "2️⃣ Web search\n"
            "3️⃣ Intelligent scraping\n"
            "4️⃣ Content analysis\n\n"
            "⏱ Please wait 2-3 minutes...",
            parse_mode='HTML'
        )

        # Run test in background
        def run_test():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                results_path = loop.run_until_complete(
                    scrape_tester.test_scraping_process(restaurant_query, bot)
                )

                loop.close()
                logger.info(f"Scraping test completed: {results_path}")

            except Exception as e:
                logger.error(f"Error in scraping test: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Scraping test failed for '{restaurant_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_test, daemon=True)
        thread.start()

    logger.info("Scrape test command added to bot: /test_scrape")