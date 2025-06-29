# scraping_test_command.py
import asyncio
import json
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ScrapingTestCommand:
    """
    Admin command to test scraping quality on specific URLs
    Creates detailed reports that can be sent to admin Telegram group
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

        # Get admin chat ID from config
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize scraper components for direct testing
        from agents.scraper import FirecrawlWebScraper
        from agents.specialized_scraper import EaterTimeoutSpecializedScraper

        self.firecrawl_scraper = FirecrawlWebScraper(config)
        self.specialized_scraper = None  # Will be initialized when needed

    async def test_scraping_quality(self, test_urls: List[str], bot=None) -> str:
        """
        Test scraping quality on provided URLs and generate detailed report

        Args:
            test_urls: List of URLs to test scraping on
            bot: Telegram bot instance (optional, for sending files)

        Returns:
            str: Path to the generated report file
        """
        logger.info(f"Starting scraping quality test on {len(test_urls)} URLs")

        # Create test report
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_urls": test_urls,
            "results": [],
            "summary": {},
            "firecrawl_stats": {},
            "specialized_stats": {}
        }

        # Test each URL with both scrapers
        for i, url in enumerate(test_urls, 1):
            logger.info(f"Testing URL {i}/{len(test_urls)}: {url}")

            url_result = {
                "url": url,
                "test_order": i,
                "firecrawl_result": None,
                "specialized_result": None,
                "comparison": None
            }

            # Test with Firecrawl scraper
            try:
                firecrawl_result = await self._test_firecrawl_scraping(url)
                url_result["firecrawl_result"] = firecrawl_result
            except Exception as e:
                logger.error(f"Firecrawl test failed for {url}: {e}")
                url_result["firecrawl_result"] = {
                    "success": False,
                    "error": str(e),
                    "method": "firecrawl"
                }

            # Test with specialized scraper if applicable
            try:
                specialized_result = await self._test_specialized_scraping(url)
                url_result["specialized_result"] = specialized_result
            except Exception as e:
                logger.error(f"Specialized test failed for {url}: {e}")
                url_result["specialized_result"] = {
                    "success": False,
                    "error": str(e),
                    "method": "specialized"
                }

            # Compare results
            url_result["comparison"] = self._compare_scraping_results(
                url_result["firecrawl_result"], 
                url_result["specialized_result"]
            )

            test_results["results"].append(url_result)

            # Add delay between tests to be respectful
            await asyncio.sleep(2)

        # Generate summary statistics
        test_results["summary"] = self._generate_test_summary(test_results["results"])
        test_results["firecrawl_stats"] = self.firecrawl_scraper.get_stats()

        # Get specialized scraper stats if available
        if self.specialized_scraper:
            test_results["specialized_stats"] = self.specialized_scraper.get_stats()

        # Create and save report file
        report_file_path = await self._create_report_file(test_results)

        # Send to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_report_to_admin(bot, report_file_path, test_results["summary"])

        return report_file_path

    async def _test_firecrawl_scraping(self, url: str) -> Dict[str, Any]:
        """Test Firecrawl scraper on a single URL"""

        # Create a mock search result
        mock_result = {
            "url": url,
            "title": "Test URL",
            "description": "Testing scraping quality"
        }

        # Test the scraping
        start_time = time.time()
        scraped_result = await self.firecrawl_scraper._scrape_single_result(mock_result)
        end_time = time.time()

        # Extract key metrics
        return {
            "success": scraped_result.get("scraping_success", False),
            "method": "firecrawl",
            "processing_time": round(end_time - start_time, 2),
            "restaurant_count": scraped_result.get("restaurant_count", 0),
            "content_length": len(scraped_result.get("scraped_content", "")),
            "extraction_method": scraped_result.get("source_info", {}).get("extraction_method", "unknown"),
            "error": scraped_result.get("scraping_error"),
            "is_problematic_site": scraped_result.get("is_problematic_site", False),
            "restaurants_data": scraped_result.get("restaurants_data", []),
            "scraped_content_preview": scraped_result.get("scraped_content", "")[:500] + "..." if scraped_result.get("scraped_content") else ""
        }

    async def _test_specialized_scraping(self, url: str) -> Dict[str, Any]:
        """Test specialized scraper on a single URL"""

        # Initialize specialized scraper if needed
        if not self.specialized_scraper:
            from agents.specialized_scraper import EaterTimeoutSpecializedScraper
            self.specialized_scraper = EaterTimeoutSpecializedScraper(self.config)

        # Check if URL can be handled by specialized scraper
        handler = self.specialized_scraper._find_handler(url)
        if not handler:
            return {
                "success": False,
                "method": "specialized",
                "error": "No specialized handler available for this URL",
                "can_handle": False
            }

        # Create mock search result
        mock_result = {
            "url": url,
            "title": "Test URL",
            "description": "Testing specialized scraping"
        }

        # Test the scraping
        start_time = time.time()

        async with self.specialized_scraper as scraper:
            results = await scraper.process_specialized_urls([mock_result])

        end_time = time.time()

        if results and len(results) > 0:
            result = results[0]
            return {
                "success": result.get("scraping_success", False),
                "method": "specialized",
                "processing_time": round(end_time - start_time, 2),
                "article_count": result.get("article_count", 0),
                "content_length": len(result.get("scraped_content", "")),
                "handler_used": result.get("handler_used", "unknown"),
                "error": result.get("scraping_error"),
                "can_handle": True,
                "articles_data": result.get("articles_data", []),
                "scraped_content_preview": result.get("scraped_content", "")[:500] + "..." if result.get("scraped_content") else ""
            }
        else:
            return {
                "success": False,
                "method": "specialized",
                "error": "No results returned",
                "can_handle": True
            }

    def _compare_scraping_results(self, firecrawl_result: Dict, specialized_result: Dict) -> Dict[str, Any]:
        """Compare results from both scraping methods"""

        if not firecrawl_result or not specialized_result:
            return {
                "comparison_possible": False,
                "reason": "One or both results missing"
            }

        firecrawl_success = firecrawl_result.get("success", False)
        specialized_success = specialized_result.get("success", False)
        specialized_can_handle = specialized_result.get("can_handle", False)

        comparison = {
            "comparison_possible": True,
            "both_succeeded": firecrawl_success and specialized_success,
            "specialized_applicable": specialized_can_handle,
            "firecrawl_success": firecrawl_success,
            "specialized_success": specialized_success,
            "time_comparison": {},
            "content_comparison": {},
            "recommendation": ""
        }

        # Compare processing times
        if firecrawl_result.get("processing_time") and specialized_result.get("processing_time"):
            fc_time = firecrawl_result["processing_time"]
            spec_time = specialized_result["processing_time"]

            comparison["time_comparison"] = {
                "firecrawl_time": fc_time,
                "specialized_time": spec_time,
                "faster_method": "firecrawl" if fc_time < spec_time else "specialized",
                "time_difference": abs(fc_time - spec_time)
            }

        # Compare content quality
        fc_content_len = firecrawl_result.get("content_length", 0)
        spec_content_len = specialized_result.get("content_length", 0)

        fc_count = firecrawl_result.get("restaurant_count", 0)
        spec_count = specialized_result.get("article_count", 0)  # Note: different metrics

        comparison["content_comparison"] = {
            "firecrawl_content_length": fc_content_len,
            "specialized_content_length": spec_content_len,
            "firecrawl_restaurant_count": fc_count,
            "specialized_article_count": spec_count,
            "more_content": "firecrawl" if fc_content_len > spec_content_len else "specialized"
        }

        # Generate recommendation
        if not specialized_can_handle:
            comparison["recommendation"] = "Use Firecrawl (specialized handler not available)"
        elif specialized_success and not firecrawl_success:
            comparison["recommendation"] = "Use Specialized (Firecrawl failed)"
        elif firecrawl_success and not specialized_success:
            comparison["recommendation"] = "Use Firecrawl (Specialized failed)"
        elif both_succeeded := (firecrawl_success and specialized_success):
            if spec_time < fc_time and spec_content_len > fc_content_len * 0.7:
                comparison["recommendation"] = "Use Specialized (faster with good content)"
            elif fc_count > spec_count:
                comparison["recommendation"] = "Use Firecrawl (more restaurants found)"
            else:
                comparison["recommendation"] = "Use Specialized (saves Firecrawl credits)"
        else:
            comparison["recommendation"] = "Both methods failed"

        return comparison

    def _generate_test_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from test results"""

        total_tests = len(results)
        firecrawl_successes = sum(1 for r in results if r.get("firecrawl_result", {}).get("success", False))
        specialized_successes = sum(1 for r in results if r.get("specialized_result", {}).get("success", False))
        specialized_applicable = sum(1 for r in results if r.get("specialized_result", {}).get("can_handle", False))

        both_succeeded = sum(1 for r in results if (
            r.get("firecrawl_result", {}).get("success", False) and 
            r.get("specialized_result", {}).get("success", False)
        ))

        return {
            "total_urls_tested": total_tests,
            "firecrawl_success_rate": round(firecrawl_successes / total_tests * 100, 1) if total_tests > 0 else 0,
            "specialized_success_rate": round(specialized_successes / specialized_applicable * 100, 1) if specialized_applicable > 0 else 0,
            "specialized_applicable_rate": round(specialized_applicable / total_tests * 100, 1) if total_tests > 0 else 0,
            "both_methods_succeeded": both_succeeded,
            "firecrawl_only_succeeded": firecrawl_successes - both_succeeded,
            "specialized_only_succeeded": specialized_successes - both_succeeded,
            "total_failures": total_tests - max(firecrawl_successes, specialized_successes),
            "recommended_approach": self._get_overall_recommendation(results)
        }

    def _get_overall_recommendation(self, results: List[Dict]) -> str:
        """Get overall recommendation based on all test results"""

        recommendations = [r.get("comparison", {}).get("recommendation", "") for r in results]

        firecrawl_votes = sum(1 for rec in recommendations if "Firecrawl" in rec)
        specialized_votes = sum(1 for rec in recommendations if "Specialized" in rec)

        if specialized_votes > firecrawl_votes:
            return "Prioritize specialized scraping where available to save credits"
        elif firecrawl_votes > specialized_votes:
            return "Prioritize Firecrawl for better restaurant extraction"
        else:
            return "Mixed results - current hybrid approach is optimal"

    async def _create_report_file(self, test_results: Dict) -> str:
        """Create a detailed text report file"""

        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraping_test_report_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESTAURANT SCRAPING QUALITY TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Test Date: {test_results['test_timestamp']}\n")
            f.write(f"URLs Tested: {len(test_results['test_urls'])}\n\n")

            # Summary section
            summary = test_results['summary']
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total URLs Tested: {summary['total_urls_tested']}\n")
            f.write(f"Firecrawl Success Rate: {summary['firecrawl_success_rate']}%\n")
            f.write(f"Specialized Success Rate: {summary['specialized_success_rate']}%\n")
            f.write(f"Specialized Applicable: {summary['specialized_applicable_rate']}%\n")
            f.write(f"Both Methods Succeeded: {summary['both_methods_succeeded']}\n")
            f.write(f"Overall Recommendation: {summary['recommended_approach']}\n\n")

            # Detailed results for each URL
            f.write("DETAILED RESULTS\n")
            f.write("-" * 40 + "\n\n")

            for i, result in enumerate(test_results['results'], 1):
                f.write(f"URL {i}: {result['url']}\n")
                f.write("-" * 60 + "\n")

                # Firecrawl results
                fc_result = result.get('firecrawl_result', {})
                f.write(f"FIRECRAWL: {'✓' if fc_result.get('success') else '✗'}\n")
                if fc_result.get('success'):
                    f.write(f"  - Processing Time: {fc_result.get('processing_time', 0)}s\n")
                    f.write(f"  - Restaurants Found: {fc_result.get('restaurant_count', 0)}\n")
                    f.write(f"  - Content Length: {fc_result.get('content_length', 0)} chars\n")
                    f.write(f"  - Extraction Method: {fc_result.get('extraction_method', 'unknown')}\n")
                    if fc_result.get('is_problematic_site'):
                        f.write("  - ⚠️ Marked as problematic site\n")
                else:
                    f.write(f"  - Error: {fc_result.get('error', 'Unknown error')}\n")

                # Specialized results
                spec_result = result.get('specialized_result', {})
                f.write(f"SPECIALIZED: {'✓' if spec_result.get('success') else '✗'}\n")
                if spec_result.get('can_handle', False):
                    if spec_result.get('success'):
                        f.write(f"  - Processing Time: {spec_result.get('processing_time', 0)}s\n")
                        f.write(f"  - Articles Found: {spec_result.get('article_count', 0)}\n")
                        f.write(f"  - Content Length: {spec_result.get('content_length', 0)} chars\n")
                        f.write(f"  - Handler Used: {spec_result.get('handler_used', 'unknown')}\n")
                    else:
                        f.write(f"  - Error: {spec_result.get('error', 'Unknown error')}\n")
                else:
                    f.write("  - No specialized handler available\n")

                # Comparison
                comparison = result.get('comparison', {})
                if comparison.get('comparison_possible'):
                    f.write(f"RECOMMENDATION: {comparison.get('recommendation', 'None')}\n")

                f.write("\n")

            # Statistics section
            f.write("SCRAPER STATISTICS\n")
            f.write("-" * 40 + "\n")

            fc_stats = test_results.get('firecrawl_stats', {})
            if fc_stats:
                f.write("Firecrawl Stats:\n")
                f.write(f"  - Total Scraped: {fc_stats.get('total_scraped', 0)}\n")
                f.write(f"  - Successful: {fc_stats.get('successful_extractions', 0)}\n")
                f.write(f"  - Failed: {fc_stats.get('failed_extractions', 0)}\n")
                f.write(f"  - Credits Used: {fc_stats.get('credits_used', 0)}\n")
                f.write(f"  - Restaurants Found: {fc_stats.get('total_restaurants_found', 0)}\n\n")

            spec_stats = test_results.get('specialized_stats', {})
            if spec_stats:
                f.write("Specialized Stats:\n")
                f.write(f"  - Total Processed: {spec_stats.get('total_processed', 0)}\n")
                f.write(f"  - Successful: {spec_stats.get('successful_extractions', 0)}\n")
                f.write(f"  - Failed: {spec_stats.get('failed_extractions', 0)}\n")
                f.write(f"  - Articles Found: {spec_stats.get('total_articles_found', 0)}\n")

                handlers_used = spec_stats.get('handlers_used', {})
                if handlers_used:
                    f.write("  - Handlers Used:\n")
                    for handler, count in handlers_used.items():
                        f.write(f"    * {handler}: {count}\n")

        logger.info(f"Test report saved to: {filepath}")
        return filepath

    async def _send_report_to_admin(self, bot, report_file_path: str, summary: Dict):
        """Send the report file to admin Telegram group"""

        if not self.admin_chat_id:
            logger.warning("No admin chat ID configured, cannot send report")
            return

        try:
            # Send summary message first
            summary_text = (
                f"🧪 <b>Scraping Test Report</b>\n\n"
                f"📊 <b>Results Summary:</b>\n"
                f"• Total URLs: {summary['total_urls_tested']}\n"
                f"• Firecrawl Success: {summary['firecrawl_success_rate']}%\n"
                f"• Specialized Success: {summary['specialized_success_rate']}%\n"
                f"• Both Succeeded: {summary['both_methods_succeeded']}\n\n"
                f"💡 <b>Recommendation:</b>\n{summary['recommended_approach']}\n\n"
                f"📎 Detailed report attached below."
            )

            bot.send_message(
                self.admin_chat_id,
                summary_text,
                parse_mode='HTML'
            )

            # Send the report file
            with open(report_file_path, 'rb') as report_file:
                bot.send_document(
                    self.admin_chat_id,
                    report_file,
                    caption="📄 Detailed scraping test report"
                )

            logger.info("Successfully sent test report to admin")

        except Exception as e:
            logger.error(f"Failed to send report to admin: {e}")


# Function to integrate with telegram_bot.py
def add_test_scraping_command(bot, config, orchestrator):
    """
    Add the test scraping command to the Telegram bot
    Call this function from telegram_bot.py to enable the admin command
    """

    # Create the test command handler
    test_handler = ScrapingTestCommand(config, orchestrator)

    @bot.message_handler(commands=['test_scraping'])
    def handle_test_scraping(message):
        """Handle /test_scraping command"""

        user_id = message.from_user.id
        admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Check if user is admin
        if not admin_chat_id or str(user_id) != str(admin_chat_id):
            bot.reply_to(message, "❌ This command is only available to administrators.")
            return

        # Parse command arguments
        command_parts = message.text.split()

        if len(command_parts) < 2:
            help_text = (
                "🧪 <b>Scraping Test Command</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/test_scraping URL1 [URL2] [URL3] ...</code>\n\n"
                "<b>Example:</b>\n"
                "<code>/test_scraping https://ny.eater.com/maps/best-restaurants-nyc</code>\n\n"
                "This will test both Firecrawl and specialized scrapers on the provided URLs "
                "and send a detailed report."
            )
            bot.reply_to(message, help_text, parse_mode='HTML')
            return

        # Extract URLs from command
        test_urls = command_parts[1:]

        # Validate URLs
        valid_urls = []
        for url in test_urls:
            if url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                bot.reply_to(message, f"❌ Invalid URL: {url}")
                return

        if not valid_urls:
            bot.reply_to(message, "❌ No valid URLs provided.")
            return

        # Send confirmation and start test
        bot.reply_to(
            message, 
            f"🧪 Starting scraping test on {len(valid_urls)} URL(s)...\n"
            "This may take a few minutes. You'll receive a detailed report when complete.",
            parse_mode='HTML'
        )

        # Run test in background thread
        import threading

        def run_test():
            try:
                # Run the async test
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                report_path = loop.run_until_complete(
                    test_handler.test_scraping_quality(valid_urls, bot)
                )

                loop.close()

                logger.info(f"Scraping test completed. Report saved to: {report_path}")

            except Exception as e:
                logger.error(f"Error in scraping test: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Scraping test failed: {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_test, daemon=True)
        thread.start()

    logger.info("Test scraping command added to bot")