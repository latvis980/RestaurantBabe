# debug_query_command.py
import asyncio
import json
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DebugQueryCommand:
    """
    Admin command to debug what content gets passed to the list_analyzer for any query
    Shows the complete pipeline output up to the list_analyzer stage
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

        # Get admin chat ID from config
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize components for manual pipeline execution
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.scraper import FirecrawlWebScraper

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = FirecrawlWebScraper(config)

    async def debug_query_pipeline(self, user_query: str, bot=None) -> str:
        """
        Run the pipeline up to list_analyzer and capture all intermediate data

        Args:
            user_query: The restaurant query to debug
            bot: Telegram bot instance (optional, for sending files)

        Returns:
            str: Path to the generated debug file
        """
        logger.info(f"Starting query pipeline debug for: {user_query}")

        debug_data = {
            "debug_timestamp": datetime.now().isoformat(),
            "original_query": user_query,
            "pipeline_stages": {},
            "final_analyzer_input": {},
            "statistics": {}
        }

        try:
            # Stage 1: Query Analysis
            logger.info("Stage 1: Analyzing query...")
            start_time = time.time()

            query_analysis = self.query_analyzer.analyze(user_query)

            debug_data["pipeline_stages"]["1_query_analysis"] = {
                "stage_name": "Query Analysis",
                "processing_time": round(time.time() - start_time, 2),
                "input": user_query,
                "output": query_analysis,
                "success": True
            }

            # Stage 2: Search
            logger.info("Stage 2: Performing web search...")
            start_time = time.time()

            search_queries = query_analysis.get("search_queries", [])
            if not search_queries:
                raise Exception("No search queries generated from query analysis")

            search_results = self.search_agent.search(search_queries)

            debug_data["pipeline_stages"]["2_search"] = {
                "stage_name": "Web Search",
                "processing_time": round(time.time() - start_time, 2),
                "input": {
                    "search_queries": search_queries,
                    "query_count": len(search_queries)
                },
                "output": {
                    "results_summary": [
                        {
                            "url": r.get("url", ""),
                            "title": r.get("title", ""),
                            "description": r.get("description", "")[:100] + "..." if r.get("description") else ""
                        } for r in search_results[:10]  # Show first 10 for brevity
                    ],
                    "total_results": len(search_results)
                },
                "success": True,
                "raw_results_count": len(search_results)
            }

            # Stage 3: Scraping
            logger.info("Stage 3: Scraping search results...")
            start_time = time.time()

            if not search_results:
                raise Exception("No search results to scrape")

            # Use the async scraper directly
            enriched_results = await self.scraper.scrape_search_results(search_results)

            # Analyze scraping results
            successful_scrapes = [r for r in enriched_results if r.get("scraping_success")]
            failed_scrapes = [r for r in enriched_results if r.get("scraping_failed")]

            debug_data["pipeline_stages"]["3_scraping"] = {
                "stage_name": "Content Scraping",
                "processing_time": round(time.time() - start_time, 2),
                "input": {
                    "urls_to_scrape": len(search_results),
                    "urls_sample": [r.get("url", "") for r in search_results[:5]]
                },
                "output": {
                    "successful_scrapes": len(successful_scrapes),
                    "failed_scrapes": len(failed_scrapes),
                    "total_restaurants_found": sum(r.get("restaurant_count", 0) for r in successful_scrapes),
                    "scraping_methods_used": list(set(
                        r.get("source_info", {}).get("extraction_method", "unknown") 
                        for r in successful_scrapes
                    )),
                    "successful_urls": [r.get("url", "") for r in successful_scrapes],
                    "failed_urls": [r.get("url", "") for r in failed_scrapes]
                },
                "success": True,
                "enriched_results_count": len(enriched_results)
            }

            # Stage 4: Prepare List Analyzer Input
            logger.info("Stage 4: Preparing input for list analyzer...")

            # This is exactly what gets passed to the list_analyzer
            analyzer_input = {
                "search_results": enriched_results,  # The actual parameter name used
                "keywords_for_analysis": query_analysis.get("keywords_for_analysis", []),
                "primary_search_parameters": query_analysis.get("primary_search_parameters", []),
                "secondary_filter_parameters": query_analysis.get("secondary_filter_parameters", []),
                "destination": query_analysis.get("destination", "Unknown")
            }

            debug_data["final_analyzer_input"] = analyzer_input

            # Create detailed content breakdown
            content_breakdown = self._analyze_scraped_content(enriched_results)

            debug_data["pipeline_stages"]["4_analyzer_prep"] = {
                "stage_name": "List Analyzer Input Preparation",
                "processing_time": 0.1,  # Minimal processing time
                "input": "Enriched scraping results",
                "output": {
                    "total_content_length": content_breakdown["total_content_length"],
                    "sources_with_content": content_breakdown["sources_with_content"],
                    "content_by_source": content_breakdown["content_by_source"],
                    "keywords_for_analysis": analyzer_input["keywords_for_analysis"],
                    "primary_parameters": analyzer_input["primary_search_parameters"],
                    "secondary_parameters": analyzer_input["secondary_filter_parameters"],
                    "destination": analyzer_input["destination"]
                },
                "success": True
            }

            # Statistics summary
            debug_data["statistics"] = {
                "total_pipeline_time": sum(
                    stage.get("processing_time", 0) 
                    for stage in debug_data["pipeline_stages"].values()
                ),
                "firecrawl_stats": self.scraper.get_stats(),
                "search_filtering_stats": getattr(self.search_agent, 'evaluation_stats', {}),
                "content_quality_summary": content_breakdown["quality_summary"]
            }

        except Exception as e:
            logger.error(f"Error in pipeline debug: {e}")

            # Add error information
            debug_data["pipeline_error"] = {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "failed_at_stage": len(debug_data["pipeline_stages"]) + 1
            }

        # Create and save debug file
        debug_file_path = await self._create_debug_file(debug_data)

        # Send to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_debug_to_admin(bot, debug_file_path, debug_data, user_query)

        return debug_file_path

    def _analyze_scraped_content(self, enriched_results: List[Dict]) -> Dict[str, Any]:
        """Analyze the scraped content that will go to list_analyzer"""

        analysis = {
            "total_content_length": 0,
            "sources_with_content": 0,
            "content_by_source": [],
            "quality_summary": {}
        }

        total_restaurants = 0
        successful_extractions = 0

        for result in enriched_results:
            scraped_content = result.get("scraped_content", "")
            source_info = result.get("source_info", {})
            url = result.get("url", "Unknown URL")

            if scraped_content:
                analysis["sources_with_content"] += 1
                analysis["total_content_length"] += len(scraped_content)

                # Content preview for this source
                content_info = {
                    "url": url,
                    "source_name": source_info.get("name", "Unknown Source"),
                    "content_length": len(scraped_content),
                    "restaurant_count": result.get("restaurant_count", 0),
                    "extraction_method": source_info.get("extraction_method", "unknown"),
                    "content_preview": scraped_content[:500] + "..." if len(scraped_content) > 500 else scraped_content,
                    "scraping_success": result.get("scraping_success", False)
                }

                analysis["content_by_source"].append(content_info)

                if result.get("scraping_success"):
                    successful_extractions += 1
                    total_restaurants += result.get("restaurant_count", 0)

        # Quality summary
        analysis["quality_summary"] = {
            "total_sources_processed": len(enriched_results),
            "sources_with_content": analysis["sources_with_content"],
            "successful_extraction_rate": round(successful_extractions / max(len(enriched_results), 1) * 100, 1),
            "total_restaurants_extracted": total_restaurants,
            "average_content_per_source": round(analysis["total_content_length"] / max(analysis["sources_with_content"], 1), 0),
            "content_ready_for_analyzer": analysis["total_content_length"] > 0
        }

        return analysis

    async def _create_debug_file(self, debug_data: Dict) -> str:
        """Create a detailed debug file showing the complete pipeline"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_debug_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("RESTAURANT QUERY PIPELINE DEBUG REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Debug Date: {debug_data['debug_timestamp']}\n")
            f.write(f"Original Query: {debug_data['original_query']}\n\n")

            # Pipeline stages
            f.write("PIPELINE EXECUTION STAGES\n")
            f.write("=" * 50 + "\n\n")

            for stage_id, stage_data in debug_data["pipeline_stages"].items():
                f.write(f"STAGE {stage_id}: {stage_data['stage_name']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Processing Time: {stage_data['processing_time']}s\n")
                f.write(f"Success: {'✓' if stage_data['success'] else '✗'}\n\n")

                # Input
                f.write("INPUT:\n")
                input_data = stage_data['input']
                if isinstance(input_data, str):
                    f.write(f"  {input_data}\n")
                elif isinstance(input_data, dict):
                    for key, value in input_data.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

                # Output summary
                f.write("OUTPUT SUMMARY:\n")
                output_data = stage_data['output']
                if isinstance(output_data, dict):
                    for key, value in output_data.items():
                        if isinstance(value, list) and len(value) > 3:
                            f.write(f"  {key}: [{len(value)} items]\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                f.write("\n\n")

            # Error information if any
            if "pipeline_error" in debug_data:
                error = debug_data["pipeline_error"]
                f.write("PIPELINE ERROR\n")
                f.write("=" * 50 + "\n")
                f.write(f"Error Type: {error['error_type']}\n")
                f.write(f"Error Message: {error['error_message']}\n")
                f.write(f"Failed at Stage: {error['failed_at_stage']}\n\n")

            # Final analyzer input (THE MOST IMPORTANT PART)
            f.write("FINAL INPUT TO LIST_ANALYZER\n")
            f.write("=" * 50 + "\n\n")

            analyzer_input = debug_data.get("final_analyzer_input", {})

            f.write("PARAMETERS:\n")
            f.write(f"Destination: {analyzer_input.get('destination', 'Not set')}\n")
            f.write(f"Keywords for Analysis: {analyzer_input.get('keywords_for_analysis', [])}\n")
            f.write(f"Primary Search Parameters: {analyzer_input.get('primary_search_parameters', [])}\n")
            f.write(f"Secondary Filter Parameters: {analyzer_input.get('secondary_filter_parameters', [])}\n\n")

            # Detailed content breakdown
            content_breakdown = debug_data["pipeline_stages"].get("4_analyzer_prep", {}).get("output", {})

            f.write("CONTENT ANALYSIS:\n")
            f.write(f"Total Content Length: {content_breakdown.get('total_content_length', 0)} characters\n")
            f.write(f"Sources with Content: {content_breakdown.get('sources_with_content', 0)}\n\n")

            # Content by source
            sources_content = content_breakdown.get("content_by_source", [])
            if sources_content:
                f.write("SCRAPED CONTENT BY SOURCE:\n")
                f.write("-" * 80 + "\n")

                for i, source in enumerate(sources_content, 1):
                    f.write(f"\nSOURCE {i}: {source['source_name']}\n")
                    f.write(f"URL: {source['url']}\n")
                    f.write(f"Content Length: {source['content_length']} characters\n")
                    f.write(f"Restaurants Found: {source['restaurant_count']}\n")
                    f.write(f"Extraction Method: {source['extraction_method']}\n")
                    f.write(f"Success: {'✓' if source['scraping_success'] else '✗'}\n")
                    f.write("Content Preview:\n")
                    f.write("-" * 40 + "\n")
                    f.write(source['content_preview'])
                    f.write("\n" + "-" * 40 + "\n")

            # Complete scraped content (what actually goes to analyzer)
            f.write("\n\nCOMPLETE SCRAPED CONTENT FOR ANALYZER\n")
            f.write("=" * 50 + "\n")
            f.write("This is the EXACT content that gets passed to the list_analyzer:\n\n")

            search_results = analyzer_input.get("search_results", [])
            for i, result in enumerate(search_results, 1):
                scraped_content = result.get("scraped_content", "")
                if scraped_content:
                    f.write(f"--- CONTENT SOURCE {i} ---\n")
                    f.write(f"URL: {result.get('url', 'Unknown')}\n")
                    f.write(f"Source: {result.get('source_info', {}).get('name', 'Unknown')}\n")
                    f.write("Content:\n")
                    f.write(scraped_content)
                    f.write("\n\n--- END SOURCE {i} ---\n\n")

            # Statistics
            f.write("\nSTATISTICS\n")
            f.write("=" * 50 + "\n")

            stats = debug_data.get("statistics", {})
            f.write(f"Total Pipeline Time: {stats.get('total_pipeline_time', 0)}s\n\n")

            # Firecrawl stats
            fc_stats = stats.get("firecrawl_stats", {})
            if fc_stats:
                f.write("Firecrawl Usage:\n")
                f.write(f"  Credits Used: {fc_stats.get('credits_used', 0)}\n")
                f.write(f"  URLs Scraped: {fc_stats.get('total_scraped', 0)}\n")
                f.write(f"  Successful Extractions: {fc_stats.get('successful_extractions', 0)}\n")
                f.write(f"  Total Restaurants Found: {fc_stats.get('total_restaurants_found', 0)}\n\n")

            # Content quality summary
            quality = stats.get("content_quality_summary", {})
            if quality:
                f.write("Content Quality:\n")
                for key, value in quality.items():
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

        logger.info(f"Debug report saved to: {filepath}")
        return filepath

    async def _send_debug_to_admin(self, bot, debug_file_path: str, debug_data: Dict, query: str):
        """Send the debug file to admin Telegram group"""

        if not self.admin_chat_id:
            logger.warning("No admin chat ID configured, cannot send debug report")
            return

        try:
            # Send summary message first
            stats = debug_data.get("statistics", {})
            quality = stats.get("content_quality_summary", {})

            summary_text = (
                f"🔍 <b>Query Pipeline Debug Report</b>\n\n"
                f"📝 <b>Query:</b> <code>{query}</code>\n\n"
                f"📊 <b>Results:</b>\n"
                f"• Sources Processed: {quality.get('total_sources_processed', 0)}\n"
                f"• Content Sources: {quality.get('sources_with_content', 0)}\n"
                f"• Extraction Rate: {quality.get('successful_extraction_rate', 0)}%\n"
                f"• Restaurants Found: {quality.get('total_restaurants_extracted', 0)}\n"
                f"• Total Content: {quality.get('average_content_per_source', 0)} chars/source\n"
                f"• Ready for Analyzer: {'✅' if quality.get('content_ready_for_analyzer') else '❌'}\n\n"
                f"💰 <b>Firecrawl Usage:</b>\n"
                f"• Credits Used: {stats.get('firecrawl_stats', {}).get('credits_used', 0)}\n\n"
                f"📎 Complete pipeline data attached below."
            )

            bot.send_message(
                self.admin_chat_id,
                summary_text,
                parse_mode='HTML'
            )

            # Send the debug file
            with open(debug_file_path, 'rb') as debug_file:
                bot.send_document(
                    self.admin_chat_id,
                    debug_file,
                    caption=f"🔍 Pipeline debug for: {query}"
                )

            logger.info("Successfully sent debug report to admin")

        except Exception as e:
            logger.error(f"Failed to send debug report to admin: {e}")


# Function to integrate with telegram_bot.py
def add_debug_query_command(bot, config, orchestrator):
    """
    Add the debug query command to the Telegram bot
    Call this function from telegram_bot.py to enable the admin command
    """

    # Create the debug command handler
    debug_handler = DebugQueryCommand(config, orchestrator)

    @bot.message_handler(commands=['debug_query'])
    def handle_debug_query(message):
        """Handle /debug_query command"""

        user_id = message.from_user.id
        admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Check if user is admin
        if not admin_chat_id or str(user_id) != str(admin_chat_id):
            bot.reply_to(message, "❌ This command is only available to administrators.")
            return

        # Parse command arguments
        command_text = message.text

        # Extract query after the command
        if len(command_text.split(None, 1)) < 2:
            help_text = (
                "🔍 <b>Query Pipeline Debug Command</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/debug_query [your restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/debug_query best cevicherias in Lima</code>\n"
                "<code>/debug_query romantic restaurants in Paris</code>\n"
                "<code>/debug_query family-friendly pizza in Rome</code>\n\n"
                "This will run the complete pipeline up to the list_analyzer stage "
                "and show you exactly what content gets passed to the AI for analysis."
            )
            bot.reply_to(message, help_text, parse_mode='HTML')
            return

        # Extract the query
        user_query = command_text.split(None, 1)[1].strip()

        if not user_query:
            bot.reply_to(message, "❌ Please provide a restaurant query to debug.")
            return

        # Send confirmation and start debug
        bot.reply_to(
            message, 
            f"🔍 Starting pipeline debug for query:\n<code>{user_query}</code>\n\n"
            "This will run the complete search and scraping pipeline. "
            "You'll receive a detailed report showing exactly what content "
            "gets passed to the list_analyzer.\n\n"
            "⏱ This may take 2-3 minutes...",
            parse_mode='HTML'
        )

        # Run debug in background thread
        import threading

        def run_debug():
            try:
                # Run the async debug
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                debug_path = loop.run_until_complete(
                    debug_handler.debug_query_pipeline(user_query, bot)
                )

                loop.close()

                logger.info(f"Query debug completed. Report saved to: {debug_path}")

            except Exception as e:
                logger.error(f"Error in query debug: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Query debug failed for '{user_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_debug, daemon=True)
        thread.start()

    logger.info("Debug query command added to bot")