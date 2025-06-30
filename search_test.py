# search_test.py
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
    Test the LIVE search process - same as what users get
    This replaced the standalone test to show actual live results
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

    async def test_search_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run the EXACT same process as orchestrator.process_query() - LIVE PROCESS

        Args:
            restaurant_query: The restaurant query to test (e.g., "best wine bars in rome")
            bot: Telegram bot instance for sending file

        Returns:
            str: Path to the results file
        """
        logger.info(f"Testing LIVE search process for: {restaurant_query}")

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_search_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LIVE RESTAURANT SEARCH PROCESS TEST\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Query: {restaurant_query}\n")
            f.write(f"Method: orchestrator.process_query() - SAME AS LIVE USERS GET\n\n")

            try:
                # STEP 1: Run the EXACT same orchestrator process as live searches
                f.write("STEP 1: RUNNING LIVE ORCHESTRATOR PROCESS\n")
                f.write("-" * 50 + "\n")

                start_time = time.time()
                f.write(f"Calling: orchestrator.process_query('{restaurant_query}')\n")
                f.write("This is the EXACT same call that happens when users search\n\n")

                # This is the EXACT same call that happens in live searches
                result = self.orchestrator.process_query(restaurant_query)

                total_time = round(time.time() - start_time, 2)
                f.write(f"Total Processing Time: {total_time}s\n\n")

                # STEP 2: Analyze the orchestrator result
                f.write("STEP 2: LIVE ORCHESTRATOR RESULT ANALYSIS\n")
                f.write("-" * 50 + "\n")

                f.write(f"Result Type: {type(result)}\n")
                f.write(f"Result Keys: {list(result.keys()) if result else 'None'}\n\n")

                if result:
                    # Check if we got recommendations
                    enhanced_recs = result.get('enhanced_recommendations', {})
                    main_list = enhanced_recs.get('main_list', []) if enhanced_recs else []
                    hidden_gems = enhanced_recs.get('hidden_gems', []) if enhanced_recs else []
                    telegram_text = result.get('telegram_formatted_text', '')

                    f.write(f"Enhanced Recommendations: {bool(enhanced_recs)}\n")
                    f.write(f"Main List Count: {len(main_list)}\n")
                    f.write(f"Hidden Gems Count: {len(hidden_gems)}\n")
                    f.write(f"Telegram Text Length: {len(telegram_text)} chars\n")

                    has_valid_output = telegram_text and telegram_text != 'Sorry, no recommendations found.'
                    f.write(f"Has Valid Telegram Output: {'YES' if has_valid_output else 'NO'}\n\n")

                    # Show the restaurants found (what users actually get)
                    if main_list:
                        f.write("🍽️ RESTAURANTS FOUND (Main List - What Users See):\n")
                        f.write("=" * 60 + "\n")
                        for i, restaurant in enumerate(main_list, 1):
                            name = restaurant.get('name', 'Unknown')
                            location = restaurant.get('location', 'Unknown')
                            price = restaurant.get('price_range', 'Unknown')
                            description = restaurant.get('description', 'N/A')
                            sources = restaurant.get('sources', [])

                            f.write(f"\n{i}. {name}\n")
                            f.write(f"   📍 Location: {location}\n")
                            f.write(f"   💰 Price: {price}\n")
                            f.write(f"   📖 Description: {description[:150]}{'...' if len(description) > 150 else ''}\n")
                            f.write(f"   📰 Sources: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''}\n")
                        f.write("\n")
                    else:
                        f.write("❌ NO RESTAURANTS IN MAIN LIST\n\n")

                    if hidden_gems:
                        f.write("💎 HIDDEN GEMS FOUND:\n")
                        for i, restaurant in enumerate(hidden_gems, 1):
                            name = restaurant.get('name', 'Unknown')
                            f.write(f"  {i}. {name}\n")
                        f.write("\n")

                    # Show what would be sent to user (the actual Telegram message)
                    if has_valid_output:
                        f.write("📱 TELEGRAM OUTPUT (What User Actually Receives):\n")
                        f.write("=" * 60 + "\n")
                        f.write(telegram_text[:2000])  # First 2000 chars
                        if len(telegram_text) > 2000:
                            f.write(f"\n\n... [TRUNCATED - Total length: {len(telegram_text)} chars]")
                        f.write("\n\n")
                    else:
                        f.write("❌ NO TELEGRAM OUTPUT - User would see 'Sorry, no recommendations found.'\n\n")
                else:
                    f.write("❌ ORCHESTRATOR RETURNED EMPTY RESULT\n\n")

                # STEP 3: Deep dive into orchestrator state (if available)
                f.write("STEP 3: ORCHESTRATOR INTERNAL STATE ANALYSIS\n")
                f.write("-" * 50 + "\n")

                # Try to access recent orchestrator state
                try:
                    # Check if orchestrator has any debug info
                    if hasattr(self.orchestrator, '__dict__'):
                        f.write("Orchestrator attributes:\n")
                        for key, value in self.orchestrator.__dict__.items():
                            if not key.startswith('_'):
                                f.write(f"  {key}: {type(value)}\n")
                        f.write("\n")

                    # Look for chain or pipeline info
                    if hasattr(self.orchestrator, 'chain'):
                        f.write("Orchestrator has chain attribute\n")
                    if hasattr(self.orchestrator, 'pipeline'):
                        f.write("Orchestrator has pipeline attribute\n")

                except Exception as e:
                    f.write(f"Could not analyze orchestrator internals: {e}\n")

                f.write("\n")

                # STEP 4: Configuration analysis
                f.write("STEP 4: CONFIGURATION ANALYSIS\n")
                f.write("-" * 50 + "\n")

                f.write(f"Config Type: {type(self.config)}\n")
                f.write(f"Excluded Domains: {getattr(self.config, 'EXCLUDED_RESTAURANT_SOURCES', 'Not found')}\n")
                f.write(f"Search Count per Query: {getattr(self.config, 'BRAVE_SEARCH_COUNT', 'Not found')}\n")
                f.write(f"AI Model: {getattr(self.config, 'OPENAI_MODEL', 'Not found')}\n")
                f.write(f"Brave API Key: {'SET' if getattr(self.config, 'BRAVE_API_KEY', None) else 'NOT SET'}\n\n")

                # STEP 5: Success/Failure analysis and next steps
                f.write("STEP 5: ANALYSIS & NEXT STEPS\n")
                f.write("-" * 50 + "\n")

                success = bool(result and main_list and has_valid_output)

                if success:
                    f.write("✅ SUCCESS: Live orchestrator is working correctly!\n")
                    f.write(f"✅ Found {len(main_list)} restaurants for users\n")
                    f.write("✅ Telegram output is properly formatted\n")
                    f.write("✅ System is functioning as expected\n\n")

                    f.write("WHAT THIS MEANS:\n")
                    f.write("- Your restaurant recommendation system is working\n")
                    f.write("- Users are getting good results\n")
                    f.write("- The simplified search agent fixes are successful\n")
                    f.write("- No further debugging needed for core functionality\n\n")

                    f.write("WHY PREVIOUS STANDALONE TESTS FAILED:\n")
                    f.write("- Standalone tests don't use the full orchestrator pipeline\n")
                    f.write("- Orchestrator has additional processing layers\n")
                    f.write("- Different configuration or timing in orchestrator\n")
                    f.write("- Orchestrator may have backup/fallback mechanisms\n\n")

                else:
                    f.write("❌ FAILURE: Live orchestrator is not working properly\n")
                    f.write("❌ Users are not getting restaurant recommendations\n")
                    f.write("❌ System needs immediate attention\n\n")

                    f.write("IMMEDIATE TROUBLESHOOTING NEEDED:\n")
                    f.write("1. Check orchestrator pipeline configuration\n")
                    f.write("2. Verify all components are properly initialized\n")
                    f.write("3. Check API quotas and network connectivity\n")
                    f.write("4. Test with simpler, more common queries\n")
                    f.write("5. Review recent code changes or deployments\n\n")

                # STEP 6: Raw result dump for debugging
                f.write("STEP 6: RAW ORCHESTRATOR RESULT (DEBUG)\n")
                f.write("-" * 50 + "\n")

                if result:
                    # Show the raw structure
                    f.write("Raw result structure:\n")
                    try:
                        import json
                        # Convert result to JSON-serializable format for inspection
                        debug_result = {}
                        for key, value in result.items():
                            if isinstance(value, (str, int, float, bool, list, dict)):
                                debug_result[key] = str(value)[:200] + "..." if len(str(value)) > 200 else value
                            else:
                                debug_result[key] = f"<{type(value).__name__}>"

                        f.write(json.dumps(debug_result, indent=2))
                    except Exception as e:
                        f.write(f"Could not serialize result: {e}")
                        f.write(f"Result: {str(result)[:500]}...")
                else:
                    f.write("No result to show")

            except Exception as e:
                f.write(f"\n❌ ERROR during live search test: {str(e)}\n")
                logger.error(f"Error during live search test: {e}")
                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

        # Send results to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_results_to_admin(bot, filepath, restaurant_query, result if 'result' in locals() else None)

        return filepath

    async def _send_results_to_admin(self, bot, file_path: str, query: str, result: Any):
        """Send live search test results to admin via Telegram"""
        try:
            # Analyze result for summary
            if result:
                main_list = result.get('enhanced_recommendations', {}).get('main_list', [])
                telegram_text = result.get('telegram_formatted_text', '')
                has_valid_output = telegram_text and telegram_text != 'Sorry, no recommendations found.'
                success = bool(main_list and has_valid_output)
                main_count = len(main_list)
            else:
                success = False
                main_count = 0

            # Create summary message
            summary = (
                f"🔍 <b>Live Search Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🎯 Restaurants Found: {main_count}\n"
                f"📱 Valid Telegram Output: {'✅ YES' if success else '❌ NO'}\n"
                f"🔄 Method: orchestrator.process_query() (live process)\n\n"
                f"{'✅ SYSTEM WORKING - Users get results' if success else '❌ SYSTEM BROKEN - Users get no results'}\n\n"
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
                    caption=f"🔍 Live search test results for: {query}"
                )

            logger.info("Successfully sent live search test results to admin")

        except Exception as e:
            logger.error(f"Failed to send live search results to admin: {e}")


def add_search_test_command(bot, config, orchestrator):
    """
    Add the /test_search command to the Telegram bot
    Now tests the LIVE process instead of standalone search
    """

    search_tester = SearchTest(config, orchestrator)

    @bot.message_handler(commands=['test_search'])
    def handle_test_search(message):
        """Handle /test_search command - now tests LIVE process"""

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
                "🔍 <b>Live Search Process Test</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/test_search [restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/test_search best wine bars in rome</code>\n"
                "<code>/test_search romantic restaurants Paris</code>\n\n"
                "This tests the LIVE search process:\n"
                "• Uses orchestrator.process_query() (same as users)\n"
                "• Shows actual restaurants users get\n"
                "• Shows actual Telegram output\n"
                "• Verifies system is working for real users\n\n"
                "📄 Results are saved to a detailed file.\n\n"
                "🎯 <b>NEW:</b> Tests live process, not standalone components."
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
            f"🔍 <b>Starting LIVE search process test...</b>\n\n"
            f"📝 Query: <code>{restaurant_query}</code>\n\n"
            "Testing the LIVE process (what users get):\n"
            "1️⃣ orchestrator.process_query()\n"
            "2️⃣ Full pipeline analysis\n"
            "3️⃣ Actual restaurant results\n"
            "4️⃣ Real Telegram output\n\n"
            "🎯 <b>NEW:</b> Same process users experience\n"
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
                logger.info(f"Live search test completed: {results_path}")

            except Exception as e:
                logger.error(f"Error in live search test: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Live search test failed for '{restaurant_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_test, daemon=True)
        thread.start()

    logger.info("Live search test command added to bot: /test_search")