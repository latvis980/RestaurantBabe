# unified_scraping_analyzer.py
import asyncio
import json
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UnifiedScrapingAnalyzer:
    """
    Unified command that combines debug_query and scraping_test functionality
    Provides comprehensive scraping quality analysis for both queries and URLs
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

        # Get admin chat ID from config
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize all scraping components
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.scraper import FirecrawlWebScraper
        from agents.specialized_scraper import EaterTimeoutSpecializedScraper

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.intelligent_scraper = WebScraper(config)
        self.firecrawl_scraper = FirecrawlWebScraper(config)
        self.specialized_scraper = EaterTimeoutSpecializedScraper(config)

    async def analyze_scraping_quality(self, input_data: str, analysis_type: str = "auto", bot=None) -> str:
        """
        Unified analysis function that handles both queries and URLs

        Args:
            input_data: Either a restaurant query or URLs (comma-separated)
            analysis_type: "query", "urls", or "auto" (detect automatically)
            bot: Telegram bot instance for sending results

        Returns:
            str: Path to the generated analysis file
        """

        # Auto-detect input type if not specified
        if analysis_type == "auto":
            analysis_type = self._detect_input_type(input_data)

        logger.info(f"Starting unified scraping analysis - Type: {analysis_type}, Input: {input_data}")

        analysis_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "input_data": input_data,
            "analysis_type": analysis_type,
            "pipeline_stages": {},
            "scraping_comparison": {},
            "content_for_analyzer": {},
            "recommendations": {},
            "statistics": {}
        }

        try:
            if analysis_type == "query":
                await self._analyze_query_pipeline(input_data, analysis_data)
            elif analysis_type == "urls":
                await self._analyze_url_scraping(input_data, analysis_data)

            # Create comprehensive report
            report_path = await self._create_unified_report(analysis_data)

            # Send to admin if bot provided
            if bot and self.admin_chat_id:
                await self._send_unified_report(bot, report_path, analysis_data)

            return report_path

        except Exception as e:
            logger.error(f"Error in unified scraping analysis: {e}")
            analysis_data["error"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": __import__('traceback').format_exc()
            }

            # Still create report with error info
            report_path = await self._create_unified_report(analysis_data)
            return report_path

    def _detect_input_type(self, input_data: str) -> str:
        """Automatically detect if input is a query or URLs"""

        # Simple heuristic: if it contains http, treat as URLs
        if "http" in input_data.lower():
            return "urls"
        else:
            return "query"

    async def _analyze_query_pipeline(self, query: str, analysis_data: Dict):
        """Run complete pipeline analysis for a restaurant query"""

        logger.info("Running complete query pipeline analysis...")

        # Stage 1: Query Analysis
        start_time = time.time()
        query_analysis = self.query_analyzer.analyze(query)

        analysis_data["pipeline_stages"]["1_query_analysis"] = {
            "stage_name": "Query Analysis",
            "processing_time": round(time.time() - start_time, 2),
            "input": query,
            "output": query_analysis,
            "success": True
        }

        # Stage 2: Search
        start_time = time.time()
        search_queries = query_analysis.get("search_queries", [])
        search_results = self.search_agent.search(search_queries)

        analysis_data["pipeline_stages"]["2_search"] = {
            "stage_name": "Web Search",
            "processing_time": round(time.time() - start_time, 2),
            "input": search_queries,
            "output": {
                "total_results": len(search_results),
                "sample_urls": [r.get("url", "") for r in search_results[:5]]
            },
            "success": True
        }

        # Stage 3: Intelligent Scraping + URL Comparison
        start_time = time.time()

        # Get results from intelligent scraper
        intelligent_results = await self.intelligent_scraper.scrape_search_results(search_results)

        # Compare with individual scrapers on first 3 URLs for detailed analysis
        comparison_urls = [r.get("url") for r in search_results[:3] if r.get("url")]
        url_comparisons = []

        for url in comparison_urls:
            comparison = await self._compare_scrapers_on_url(url)
            url_comparisons.append(comparison)

        analysis_data["pipeline_stages"]["3_intelligent_scraping"] = {
            "stage_name": "Intelligent Scraping + Comparison",
            "processing_time": round(time.time() - start_time, 2),
            "intelligent_results": {
                "successful_scrapes": len([r for r in intelligent_results if r.get("scraping_success")]),
                "total_restaurants": sum(len(r.get("restaurants_found", [])) for r in intelligent_results),
                "scraper_stats": self.intelligent_scraper.get_stats()
            },
            "url_comparisons": url_comparisons,
            "success": True
        }

        # Stage 4: Prepare for list_analyzer
        analyzer_input = {
            "search_results": intelligent_results,
            "keywords_for_analysis": query_analysis.get("keywords_for_analysis", []),
            "primary_search_parameters": query_analysis.get("primary_search_parameters", []),
            "secondary_filter_parameters": query_analysis.get("secondary_filter_parameters", []),
            "destination": query_analysis.get("destination", "Unknown")
        }

        analysis_data["content_for_analyzer"] = {
            "total_content_length": sum(len(r.get("scraped_content", "")) for r in intelligent_results),
            "sources_with_content": len([r for r in intelligent_results if r.get("scraped_content")]),
            "analyzer_input": analyzer_input,
            "content_preview": self._create_content_preview(intelligent_results)
        }

    async def _analyze_url_scraping(self, urls_string: str, analysis_data: Dict):
        """Analyze scraping quality on specific URLs"""

        # Parse URLs
        urls = [url.strip() for url in urls_string.replace(",", " ").split() if url.strip().startswith("http")]

        logger.info(f"Analyzing scraping quality on {len(urls)} URLs...")

        url_results = []
        for i, url in enumerate(urls, 1):
            logger.info(f"Testing URL {i}/{len(urls)}: {url}")

            comparison = await self._compare_scrapers_on_url(url)
            url_results.append(comparison)

            # Add delay to be respectful
            await asyncio.sleep(2)

        analysis_data["scraping_comparison"] = {
            "urls_tested": len(urls),
            "url_results": url_results,
            "summary": self._generate_comparison_summary(url_results)
        }

    async def _compare_scrapers_on_url(self, url: str) -> Dict[str, Any]:
        """Compare all available scrapers on a single URL"""

        logger.info(f"Comparing scrapers on: {url}")

        comparison = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "results": {},
            "winner": None,
            "recommendation": ""
        }

        # Test Firecrawl scraper
        try:
            start_time = time.time()
            mock_result = {"url": url, "title": "Test URL", "description": "Testing"}
            firecrawl_result = await self.firecrawl_scraper._scrape_single_result(mock_result)

            comparison["results"]["firecrawl"] = {
                "success": firecrawl_result.get("scraping_success", False),
                "processing_time": round(time.time() - start_time, 2),
                "content_length": len(firecrawl_result.get("scraped_content", "")),
                "restaurants_found": len(firecrawl_result.get("restaurants_found", [])),
                "cost": "10 credits",
                "method": "AI-powered extraction"
            }
        except Exception as e:
            comparison["results"]["firecrawl"] = {
                "success": False,
                "error": str(e),
                "cost": "10 credits"
            }

        # Test specialized scraper
        try:
            start_time = time.time()
            specialized_result = await self.specialized_scraper.scrape_eater_timeout_content(url)

            comparison["results"]["specialized"] = {
                "success": specialized_result.get("success", False),
                "processing_time": round(time.time() - start_time, 2),
                "content_length": len(specialized_result.get("scraped_content", "")),
                "articles_found": len(specialized_result.get("articles_data", [])),
                "cost": "FREE",
                "method": "Specialized handler",
                "can_handle": specialized_result.get("can_handle", False)
            }
        except Exception as e:
            comparison["results"]["specialized"] = {
                "success": False,
                "error": str(e),
                "cost": "FREE",
                "can_handle": False
            }

        # Test intelligent scraper decision
        try:
            start_time = time.time()
            mock_search_results = [{"url": url, "title": "Test URL", "description": "Testing"}]
            intelligent_result = await self.intelligent_scraper.scrape_search_results(mock_search_results)

            if intelligent_result:
                result = intelligent_result[0]
                comparison["results"]["intelligent"] = {
                    "success": result.get("scraping_success", False),
                    "processing_time": round(time.time() - start_time, 2),
                    "content_length": len(result.get("scraped_content", "")),
                    "restaurants_found": len(result.get("restaurants_found", [])),
                    "method_chosen": result.get("scraping_method", "unknown"),
                    "ai_reasoning": result.get("strategy_info", {}).get("reasoning", "N/A"),
                    "estimated_cost": result.get("strategy_info", {}).get("estimated_cost", 0)
                }
        except Exception as e:
            comparison["results"]["intelligent"] = {
                "success": False,
                "error": str(e)
            }

        # Determine winner and recommendation
        comparison["winner"] = self._determine_best_scraper(comparison["results"])
        comparison["recommendation"] = self._generate_scraper_recommendation(comparison["results"])

        return comparison

    def _determine_best_scraper(self, results: Dict) -> str:
        """Determine which scraper performed best"""

        scores = {}

        for scraper, result in results.items():
            if not result.get("success", False):
                scores[scraper] = 0
                continue

            score = 0

            # Content length score (normalized)
            content_len = result.get("content_length", 0)
            score += min(content_len / 1000, 10)  # Max 10 points for content

            # Restaurant/article count score
            restaurants = result.get("restaurants_found", 0)
            articles = result.get("articles_found", 0)
            score += (restaurants + articles) * 2  # 2 points per item found

            # Speed bonus (faster = better)
            processing_time = result.get("processing_time", float('inf'))
            if processing_time < 5:
                score += 5 - processing_time

            # Cost efficiency bonus
            if result.get("cost") == "FREE":
                score += 5

            scores[scraper] = score

        return max(scores, key=scores.get) if scores else "none"

    def _generate_scraper_recommendation(self, results: Dict) -> str:
        """Generate human-readable recommendation"""

        firecrawl = results.get("firecrawl", {})
        specialized = results.get("specialized", {})
        intelligent = results.get("intelligent", {})

        if specialized.get("success") and specialized.get("can_handle"):
            return "Use specialized scraper - FREE and effective for this site"
        elif firecrawl.get("success") and not specialized.get("success"):
            return "Use Firecrawl - specialized scraper not available/failed"
        elif intelligent.get("success"):
            chosen_method = intelligent.get("method_chosen", "unknown")
            return f"Intelligent scraper chose: {chosen_method} - AI-optimized decision"
        else:
            return "All scrapers failed - may need manual review"

    def _generate_comparison_summary(self, url_results: List[Dict]) -> Dict:
        """Generate summary statistics from URL comparisons"""

        total_urls = len(url_results)

        scraper_successes = {
            "firecrawl": 0,
            "specialized": 0,
            "intelligent": 0
        }

        scraper_wins = {
            "firecrawl": 0,
            "specialized": 0,
            "intelligent": 0
        }

        for result in url_results:
            # Count successes
            for scraper in scraper_successes:
                if result.get("results", {}).get(scraper, {}).get("success", False):
                    scraper_successes[scraper] += 1

            # Count wins
            winner = result.get("winner", "none")
            if winner in scraper_wins:
                scraper_wins[winner] += 1

        return {
            "total_urls": total_urls,
            "success_rates": {
                scraper: round(count / total_urls * 100, 1) 
                for scraper, count in scraper_successes.items()
            },
            "win_rates": {
                scraper: round(count / total_urls * 100, 1) 
                for scraper, count in scraper_wins.items()
            },
            "overall_recommendation": self._get_overall_recommendation(url_results)
        }

    def _get_overall_recommendation(self, url_results: List[Dict]) -> str:
        """Get overall recommendation based on all results"""

        recommendations = [r.get("recommendation", "") for r in url_results]

        if not recommendations:
            return "No data available"

        # Count recommendation types
        specialized_votes = sum(1 for rec in recommendations if "specialized" in rec.lower())
        firecrawl_votes = sum(1 for rec in recommendations if "firecrawl" in rec.lower())
        intelligent_votes = sum(1 for rec in recommendations if "intelligent" in rec.lower())

        total_votes = len(recommendations)

        if specialized_votes / total_votes > 0.5:
            return "Prioritize specialized scrapers where available"
        elif firecrawl_votes / total_votes > 0.5:
            return "Prioritize Firecrawl for consistent results"
        elif intelligent_votes / total_votes > 0.4:
            return "Current intelligent scraper strategy is optimal"
        else:
            return "Mixed results - case-by-case optimization needed"

    def _create_content_preview(self, search_results: List[Dict]) -> List[Dict]:
        """Create preview of content that goes to list_analyzer"""

        preview = []

        for i, result in enumerate(search_results[:5], 1):  # Show first 5 sources
            content = result.get("scraped_content", "")

            preview.append({
                "source_number": i,
                "url": result.get("url", "Unknown"),
                "scraping_method": result.get("scraping_method", "unknown"),
                "content_length": len(content),
                "restaurants_found": len(result.get("restaurants_found", [])),
                "content_preview": content[:300] + "..." if len(content) > 300 else content,
                "success": result.get("scraping_success", False)
            })

        return preview

    async def _create_unified_report(self, analysis_data: Dict) -> str:
        """Create comprehensive unified report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unified_scraping_analysis_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("UNIFIED RESTAURANT SCRAPING QUALITY ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Analysis Date: {analysis_data['analysis_timestamp']}\n")
            f.write(f"Analysis Type: {analysis_data['analysis_type'].upper()}\n")
            f.write(f"Input Data: {analysis_data['input_data']}\n\n")

            # Handle errors
            if "error" in analysis_data:
                error = analysis_data["error"]
                f.write("ANALYSIS ERROR\n")
                f.write("=" * 50 + "\n")
                f.write(f"Error Type: {error['error_type']}\n")
                f.write(f"Error Message: {error['error_message']}\n")
                f.write("Traceback:\n")
                f.write(error.get('traceback', 'No traceback available'))
                f.write("\n\n")
                return filepath

            # Query Pipeline Analysis
            if analysis_data["analysis_type"] == "query":
                f.write("QUERY PIPELINE ANALYSIS\n")
                f.write("=" * 50 + "\n\n")

                for stage_id, stage_data in analysis_data.get("pipeline_stages", {}).items():
                    f.write(f"STAGE {stage_id}: {stage_data['stage_name']}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Processing Time: {stage_data['processing_time']}s\n")
                    f.write(f"Success: {'✓' if stage_data['success'] else '✗'}\n\n")

                # Content for analyzer section
                content_data = analysis_data.get("content_for_analyzer", {})
                f.write("CONTENT PREPARED FOR LIST_ANALYZER\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total Content Length: {content_data.get('total_content_length', 0)} characters\n")
                f.write(f"Sources with Content: {content_data.get('sources_with_content', 0)}\n\n")

                # Content preview
                content_preview = content_data.get("content_preview", [])
                for source in content_preview:
                    f.write(f"SOURCE {source['source_number']}: {source['url']}\n")
                    f.write(f"Method: {source['scraping_method']}\n")
                    f.write(f"Content: {source['content_length']} chars, {source['restaurants_found']} restaurants\n")
                    f.write(f"Preview: {source['content_preview']}\n\n")

            # URL Comparison Analysis
            if analysis_data["analysis_type"] == "urls":
                f.write("URL SCRAPING COMPARISON ANALYSIS\n")
                f.write("=" * 50 + "\n\n")

                comparison_data = analysis_data.get("scraping_comparison", {})
                summary = comparison_data.get("summary", {})

                f.write("SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"URLs Tested: {summary.get('total_urls', 0)}\n")
                f.write("Success Rates:\n")
                for scraper, rate in summary.get("success_rates", {}).items():
                    f.write(f"  {scraper.title()}: {rate}%\n")
                f.write("Win Rates:\n")
                for scraper, rate in summary.get("win_rates", {}).items():
                    f.write(f"  {scraper.title()}: {rate}%\n")
                f.write(f"\nOverall Recommendation: {summary.get('overall_recommendation', 'N/A')}\n\n")

                # Detailed results
                f.write("DETAILED URL RESULTS\n")
                f.write("-" * 30 + "\n\n")

                for i, result in enumerate(comparison_data.get("url_results", []), 1):
                    f.write(f"URL {i}: {result['url']}\n")
                    f.write(f"Winner: {result.get('winner', 'none').title()}\n")
                    f.write(f"Recommendation: {result.get('recommendation', 'N/A')}\n")

                    for scraper, data in result.get("results", {}).items():
                        f.write(f"\n{scraper.upper()}:\n")
                        if data.get("success"):
                            f.write(f"  ✓ Success\n")
                            f.write(f"  Time: {data.get('processing_time', 0)}s\n")
                            f.write(f"  Content: {data.get('content_length', 0)} chars\n")
                            f.write(f"  Items: {data.get('restaurants_found', data.get('articles_found', 0))}\n")
                            f.write(f"  Cost: {data.get('cost', 'Unknown')}\n")
                            if scraper == "intelligent":
                                f.write(f"  Method Chosen: {data.get('method_chosen', 'Unknown')}\n")
                                f.write(f"  AI Reasoning: {data.get('ai_reasoning', 'N/A')}\n")
                        else:
                            f.write(f"  ✗ Failed: {data.get('error', 'Unknown error')}\n")
                    f.write("\n" + "-" * 60 + "\n\n")

        logger.info(f"Unified analysis report saved to: {filepath}")
        return filepath

    async def _send_unified_report(self, bot, report_path: str, analysis_data: Dict):
        """Send unified report to admin"""

        if not self.admin_chat_id:
            logger.warning("No admin chat ID configured")
            return

        try:
            analysis_type = analysis_data["analysis_type"]
            input_data = analysis_data["input_data"]

            # Create summary message
            if analysis_type == "query":
                content_data = analysis_data.get("content_for_analyzer", {})
                summary_text = (
                    f"🔍 <b>Query Pipeline Analysis</b>\n\n"
                    f"📝 <b>Query:</b> <code>{input_data}</code>\n\n"
                    f"📊 <b>Results:</b>\n"
                    f"• Content Length: {content_data.get('total_content_length', 0)} chars\n"
                    f"• Sources: {content_data.get('sources_with_content', 0)}\n"
                    f"• Ready for Analyzer: {'✅' if content_data.get('total_content_length', 0) > 0 else '❌'}\n\n"
                    f"📎 Complete pipeline data attached."
                )
            else:
                comparison_data = analysis_data.get("scraping_comparison", {})
                summary = comparison_data.get("summary", {})
                summary_text = (
                    f"🧪 <b>URL Scraping Comparison</b>\n\n"
                    f"🔗 <b>URLs:</b> {summary.get('total_urls', 0)} tested\n\n"
                    f"📊 <b>Success Rates:</b>\n"
                )
                for scraper, rate in summary.get("success_rates", {}).items():
                    summary_text += f"• {scraper.title()}: {rate}%\n"
                summary_text += f"\n💡 <b>Recommendation:</b>\n{summary.get('overall_recommendation', 'N/A')}\n\n📎 Detailed comparison attached."

            bot.send_message(
                self.admin_chat_id,
                summary_text,
                parse_mode='HTML'
            )

            # Send report file
            with open(report_path, 'rb') as report_file:
                bot.send_document(
                    self.admin_chat_id,
                    report_file,
                    caption=f"📄 Unified Scraping Analysis Report"
                )

            logger.info("Successfully sent unified analysis report to admin")

        except Exception as e:
            logger.error(f"Failed to send unified report to admin: {e}")


# Integration function for telegram_bot.py
def add_unified_scraping_command(bot, config, orchestrator):
    """
    Add the unified scraping analysis command to the Telegram bot
    This replaces or supplements the individual debug_query and test_scraping commands
    """

    analyzer = UnifiedScrapingAnalyzer(config, orchestrator)

    @bot.message_handler(commands=['analyze_scraping'])
    def handle_analyze_scraping(message):
        """Handle /analyze_scraping command"""

        user_id = message.from_user.id
        admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Check if user is admin
        if not admin_chat_id or str(user_id) != str(admin_chat_id):
            bot.reply_to(message, "❌ This command is only available to administrators.")
            return

        # Parse command
        command_text = message.text

        if len(command_text.split(None, 1)) < 2:
            help_text = (
                "🔍 <b>Unified Scraping Quality Analyzer</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/analyze_scraping [query or URLs]</code>\n\n"
                "<b>Query Examples:</b>\n"
                "<code>/analyze_scraping best ramen in Tokyo</code>\n"
                "<code>/analyze_scraping romantic restaurants Paris</code>\n\n"
                "<b>URL Examples:</b>\n"
                "<code>/analyze_scraping https://ny.eater.com/maps/best-restaurants-nyc</code>\n"
                "<code>/analyze_scraping https://timeout.com/paris/restaurants, https://eater.com/london</code>\n\n"
                "🤖 <b>Auto-Detection:</b> The system automatically detects whether you're providing a restaurant query or URLs.\n\n"
                "📊 <b>For Queries:</b> Shows complete pipeline + content for list_analyzer\n"
                "🧪 <b>For URLs:</b> Compares all scraping methods + performance analysis"
            )
            bot.reply_to(message, help_text, parse_mode='HTML')
            return

        # Extract input data
        input_data = command_text.split(None, 1)[1].strip()

        if not input_data:
            bot.reply_to(message, "❌ Please provide a restaurant query or URLs to analyze.")
            return

        # Send confirmation message
        bot.reply_to(
            message,
            f"🔍 <b>Starting unified scraping analysis...</b>\n\n"
            f"📝 Input: <code>{input_data}</code>\n\n"
            "🤖 Auto-detecting input type and running comprehensive analysis.\n"
            "This will take 2-3 minutes and provide detailed insights about scraping quality.\n\n"
            "⏱ Please wait...",
            parse_mode='HTML'
        )

        # Run analysis in background
        import threading

        def run_analysis():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                report_path = loop.run_until_complete(
                    analyzer.analyze_scraping_quality(input_data, "auto", bot)
                )

                loop.close()
                logger.info(f"Unified scraping analysis completed: {report_path}")

            except Exception as e:
                logger.error(f"Error in unified scraping analysis: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Unified scraping analysis failed: {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

    logger.info("Unified scraping analysis command added to bot")
    logger.info("Available command: /analyze_scraping [query or URLs]")