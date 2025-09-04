# debug_query_command.py - Updated for Intelligent Scraper
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
    Updated for the new intelligent scraper system
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

        # Get admin chat ID from config
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Initialize components for manual pipeline execution
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.browserless_scraper import BrowserlessRestaurantScraper  # Updated import

        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = BrowserlessRestaurantScraper(config)  # Now uses intelligent scraper

    async def debug_query_pipeline(self, user_query: str, bot=None) -> str:
        """
        Run the pipeline up to list_analyzer and capture all intermediate data

        Args:
            user_query: The restaurant query to debug
            bot: Telegram bot instance (optional, for sending files)

        Returns:
            str: Path to the generated debug file
        """
        logger.info(f"Starting intelligent scraper pipeline debug for: {user_query}")

        debug_data = {
            "debug_timestamp": datetime.now().isoformat(),
            "original_query": user_query,
            "scraper_type": "intelligent_adaptive",
            "pipeline_stages": {},
            "final_analyzer_input": {},
            "intelligent_scraper_analysis": {},
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

            # Check if using AI filtering in search
            search_agent_stats = getattr(self.search_agent, 'evaluation_stats', {})

            search_results = self.search_agent.search(search_queries)

            debug_data["pipeline_stages"]["2_search"] = {
                "stage_name": "Web Search (with AI filtering)",
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
                            "description": r.get("description", "")[:100] + "..." if r.get("description") else "",
                            "ai_evaluation": r.get("ai_evaluation", {})  # Show AI filtering results
                        } for r in search_results[:10]  # Show first 10 for brevity
                    ],
                    "total_results": len(search_results),
                    "ai_filtering_stats": search_agent_stats
                },
                "success": True,
                "raw_results_count": len(search_results)
            }

            # Stage 3: Intelligent Scraping Analysis & Execution
            logger.info("Stage 3: Intelligent scraping analysis & execution...")
            start_time = time.time()

            if not search_results:
                raise Exception("No search results to scrape")

            # Clear cache for fresh analysis if needed
            # self.scraper.clear_domain_cache()

            # Use the intelligent scraper
            enriched_results = await self.scraper.scrape_search_results(search_results)

            # Get detailed intelligent scraper stats
            scraper_stats = self.scraper.get_stats()
            domain_intelligence = self.scraper.get_domain_intelligence()

            # Analyze scraping strategy distribution
            strategy_analysis = self._analyze_scraping_strategies(search_results, enriched_results)

            debug_data["intelligent_scraper_analysis"] = {
                "strategy_distribution": strategy_analysis,
                "domain_intelligence_cache": domain_intelligence,
                "ai_analysis_calls": scraper_stats.get("ai_analysis_calls", 0),
                "cache_hits": scraper_stats.get("cache_hits", 0),
                "strategy_overrides": scraper_stats.get("strategy_overrides", 0),
                "cost_savings": scraper_stats.get("total_cost_saved", 0)
            }

            # Analyze scraping results
            successful_scrapes = [r for r in enriched_results if r.get("scraping_success")]
            failed_scrapes = [r for r in enriched_results if r.get("scraping_failed")]

            debug_data["pipeline_stages"]["3_intelligent_scraping"] = {
                "stage_name": "Intelligent Adaptive Scraping",
                "processing_time": round(time.time() - start_time, 2),
                "input": {
                    "urls_to_scrape": len(search_results),
                    "urls_sample": [r.get("url", "") for r in search_results[:5]]
                },
                "output": {
                    "successful_scrapes": len(successful_scrapes),
                    "failed_scrapes": len(failed_scrapes),
                    "total_restaurants_found": sum(len(r.get("restaurants_found", [])) for r in successful_scrapes),
                    "scraping_methods_used": {
                        "specialized": scraper_stats.get("specialized_used", 0),
                        "simple_http": scraper_stats.get("simple_http_used", 0),
                        "enhanced_http": scraper_stats.get("enhanced_http_used", 0),
                        "firecrawl": scraper_stats.get("firecrawl_used", 0)
                    },
                    "strategy_distribution": strategy_analysis,
                    "successful_urls": [r.get("url", "") for r in successful_scrapes],
                    "failed_urls": [r.get("url", "") for r in failed_scrapes],
                    "cost_analysis": {
                        "total_cost_saved": scraper_stats.get("total_cost_saved", 0),
                        "firecrawl_usage_percentage": round((scraper_stats.get("firecrawl_used", 0) / max(len(search_results), 1)) * 100, 1)
                    }
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

            # Create detailed content breakdown with strategy info
            content_breakdown = self._analyze_scraped_content_with_strategies(enriched_results)

            debug_data["pipeline_stages"]["4_analyzer_prep"] = {
                "stage_name": "List Analyzer Input Preparation",
                "processing_time": 0.1,  # Minimal processing time
                "input": "Enriched scraping results from intelligent scraper",
                "output": {
                    "total_content_length": content_breakdown["total_content_length"],
                    "sources_with_content": content_breakdown["sources_with_content"],
                    "content_by_source": content_breakdown["content_by_source"],
                    "content_by_strategy": content_breakdown["content_by_strategy"],
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
                "intelligent_scraper_stats": scraper_stats,
                "search_filtering_stats": getattr(self.search_agent, 'evaluation_stats', {}),
                "content_quality_summary": content_breakdown["quality_summary"],
                "cost_efficiency": {
                    "firecrawl_credits_saved": scraper_stats.get("total_cost_saved", 0),
                    "total_urls_processed": scraper_stats.get("total_processed", 0),
                    "firecrawl_usage_rate": round((scraper_stats.get("firecrawl_used", 0) / max(scraper_stats.get("total_processed", 1), 1)) * 100, 1)
                }
            }

        except Exception as e:
            logger.error(f"Error in pipeline debug: {e}")
            import traceback

            # Add error information
            debug_data["pipeline_error"] = {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "failed_at_stage": len(debug_data["pipeline_stages"]) + 1,
                "traceback": traceback.format_exc()
            }

        # Create and save debug file
        debug_file_path = await self._create_debug_file(debug_data)

        # Send to admin if bot is provided
        if bot and self.admin_chat_id:
            await self._send_debug_to_admin(bot, debug_file_path, debug_data, user_query)

        return debug_file_path

    def _analyze_scraping_strategies(self, search_results: List[Dict], enriched_results: List[Dict]) -> Dict[str, Any]:
        """Analyze how URLs were distributed across scraping strategies"""

        strategy_distribution = {
            "specialized": [],
            "simple_http": [],
            "enhanced_http": [],
            "firecrawl": [],
            "failed": []
        }

        for result in enriched_results:
            url = result.get("url", "")
            method = result.get("scraping_method", "unknown")
            success = result.get("scraping_success", False)
            strategy = result.get("scrape_strategy")

            strategy_info = {
                "url": url,
                "method": method,
                "success": success,
                "restaurants_found": len(result.get("restaurants_found", [])),
                "content_length": len(result.get("scraped_content", "")),
            }

            if strategy:
                strategy_info.update({
                    "ai_reasoning": strategy.reasoning,
                    "confidence": strategy.confidence,
                    "estimated_cost": strategy.estimated_cost
                })

            if not success:
                strategy_distribution["failed"].append(strategy_info)
            elif method == "specialized":
                strategy_distribution["specialized"].append(strategy_info)
            elif method == "simple_http":
                strategy_distribution["simple_http"].append(strategy_info)
            elif method == "enhanced_http":
                strategy_distribution["enhanced_http"].append(strategy_info)
            elif "firecrawl" in method:
                strategy_distribution["firecrawl"].append(strategy_info)

        return strategy_distribution

    def _analyze_scraped_content_with_strategies(self, enriched_results: List[Dict]) -> Dict[str, Any]:
        """Analyze the scraped content with strategy information"""

        analysis = {
            "total_content_length": 0,
            "sources_with_content": 0,
            "content_by_source": [],
            "content_by_strategy": {
                "specialized": {"count": 0, "total_content": 0, "restaurants": 0},
                "simple_http": {"count": 0, "total_content": 0, "restaurants": 0},
                "enhanced_http": {"count": 0, "total_content": 0, "restaurants": 0},
                "firecrawl": {"count": 0, "total_content": 0, "restaurants": 0}
            },
            "quality_summary": {}
        }

        total_restaurants = 0
        successful_extractions = 0

        for result in enriched_results:
            scraped_content = result.get("scraped_content", "")
            source_info = result.get("source_info", {})
            url = result.get("url", "Unknown URL")
            method = result.get("scraping_method", "unknown")
            strategy = result.get("scrape_strategy")
            restaurants_found = result.get("restaurants_found", [])

            if scraped_content:
                analysis["sources_with_content"] += 1
                analysis["total_content_length"] += len(scraped_content)

                # Content preview for this source
                content_info = {
                    "url": url,
                    "source_name": source_info.get("name", "Unknown Source"),
                    "content_length": len(scraped_content),
                    "restaurant_count": len(restaurants_found),
                    "restaurants_found": restaurants_found,
                    "extraction_method": source_info.get("extraction_method", method),
                    "scraping_method": method,
                    "content_preview": scraped_content[:500] + "..." if len(scraped_content) > 500 else scraped_content,
                    "scraping_success": result.get("scraping_success", False),
                    "strategy_info": {
                        "reasoning": strategy.reasoning if strategy else "No strategy info",
                        "confidence": strategy.confidence if strategy else 0,
                        "estimated_cost": strategy.estimated_cost if strategy else 0
                    }
                }

                analysis["content_by_source"].append(content_info)

                # Track by strategy
                if method in analysis["content_by_strategy"]:
                    strategy_stats = analysis["content_by_strategy"][method]
                    strategy_stats["count"] += 1
                    strategy_stats["total_content"] += len(scraped_content)
                    strategy_stats["restaurants"] += len(restaurants_found)

                if result.get("scraping_success"):
                    successful_extractions += 1
                    total_restaurants += len(restaurants_found)

        # Quality summary with strategy breakdown
        analysis["quality_summary"] = {
            "total_sources_processed": len(enriched_results),
            "sources_with_content": analysis["sources_with_content"],
            "successful_extraction_rate": round(successful_extractions / max(len(enriched_results), 1) * 100, 1),
            "total_restaurants_extracted": total_restaurants,
            "average_content_per_source": round(analysis["total_content_length"] / max(analysis["sources_with_content"], 1), 0),
            "content_ready_for_analyzer": analysis["total_content_length"] > 0,
            "strategy_effectiveness": {
                method: {
                    "avg_restaurants_per_source": round(stats["restaurants"] / max(stats["count"], 1), 1),
                    "avg_content_per_source": round(stats["total_content"] / max(stats["count"], 1), 0)
                }
                for method, stats in analysis["content_by_strategy"].items()
                if stats["count"] > 0
            }
        }

        return analysis

    async def _create_debug_file(self, debug_data: Dict) -> str:
        """Create a detailed debug file showing the complete intelligent scraper pipeline"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligent_scraper_debug_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("INTELLIGENT RESTAURANT SCRAPER PIPELINE DEBUG REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Debug Date: {debug_data['debug_timestamp']}\n")
            f.write(f"Original Query: {debug_data['original_query']}\n")
            f.write(f"Scraper Type: {debug_data['scraper_type']}\n\n")

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
                        elif isinstance(value, dict) and len(str(value)) > 200:
                            f.write(f"  {key}: {{complex object}}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                f.write("\n\n")

            # Intelligent Scraper Analysis
            f.write("INTELLIGENT SCRAPER ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            scraper_analysis = debug_data.get("intelligent_scraper_analysis", {})

            f.write("STRATEGY DISTRIBUTION:\n")
            strategy_dist = scraper_analysis.get("strategy_distribution", {})
            for strategy, urls in strategy_dist.items():
                if urls:
                    f.write(f"  {strategy.upper()}: {len(urls)} URLs\n")
                    for url_info in urls[:3]:  # Show first 3 examples
                        f.write(f"    - {url_info['url'][:60]}... "
                               f"(Restaurants: {url_info['restaurants_found']}, "
                               f"Content: {url_info['content_length']} chars)\n")
                    if len(urls) > 3:
                        f.write(f"    ... and {len(urls) - 3} more\n")
                    f.write("\n")

            f.write(f"AI Analysis Calls: {scraper_analysis.get('ai_analysis_calls', 0)}\n")
            f.write(f"Cache Hits: {scraper_analysis.get('cache_hits', 0)}\n")
            f.write(f"Strategy Overrides: {scraper_analysis.get('strategy_overrides', 0)}\n")
            f.write(f"Cost Savings: {scraper_analysis.get('cost_savings', 0)} Firecrawl credits\n\n")

            # Domain Intelligence Cache
            domain_intel = scraper_analysis.get("domain_intelligence_cache", {})
            if domain_intel:
                f.write("LEARNED DOMAIN INTELLIGENCE:\n")
                for domain, info in list(domain_intel.items())[:10]:  # Show first 10
                    f.write(f"  {domain}: {info.get('complexity', 'unknown')} "
                           f"(confidence: {info.get('confidence', 0):.2f})\n")
                if len(domain_intel) > 10:
                    f.write(f"  ... and {len(domain_intel) - 10} more domains\n")
                f.write("\n")

            # Error information if any
            if "pipeline_error" in debug_data:
                error = debug_data["pipeline_error"]
                f.write("PIPELINE ERROR\n")
                f.write("=" * 50 + "\n")
                f.write(f"Error Type: {error['error_type']}\n")
                f.write(f"Error Message: {error['error_message']}\n")
                f.write(f"Failed at Stage: {error['failed_at_stage']}\n")
                f.write("Traceback:\n")
                f.write(error.get('traceback', 'No traceback available'))
                f.write("\n\n")

            # Final analyzer input (THE MOST IMPORTANT PART)
            f.write("FINAL INPUT TO LIST_ANALYZER\n")
            f.write("=" * 50 + "\n\n")

            analyzer_input = debug_data.get("final_analyzer_input", {})

            f.write("PARAMETERS:\n")
            f.write(f"Destination: {analyzer_input.get('destination', 'Not set')}\n")
            f.write(f"Keywords for Analysis: {analyzer_input.get('keywords_for_analysis', [])}\n")
            f.write(f"Primary Search Parameters: {analyzer_input.get('primary_search_parameters', [])}\n")
            f.write(f"Secondary Filter Parameters: {analyzer_input.get('secondary_filter_parameters', [])}\n\n")

            # Strategy-aware content breakdown
            content_breakdown = debug_data["pipeline_stages"].get("4_analyzer_prep", {}).get("output", {})

            f.write("CONTENT ANALYSIS:\n")
            f.write(f"Total Content Length: {content_breakdown.get('total_content_length', 0)} characters\n")
            f.write(f"Sources with Content: {content_breakdown.get('sources_with_content', 0)}\n\n")

            # Content by strategy
            content_by_strategy = content_breakdown.get("content_by_strategy", {})
            if content_by_strategy:
                f.write("CONTENT BY SCRAPING STRATEGY:\n")
                for strategy, stats in content_by_strategy.items():
                    if stats["count"] > 0:
                        f.write(f"  {strategy.upper()}:\n")
                        f.write(f"    Sources: {stats['count']}\n")
                        f.write(f"    Total Content: {stats['total_content']} characters\n")
                        f.write(f"    Restaurants Found: {stats['restaurants']}\n")
                        f.write(f"    Avg Content/Source: {stats['total_content'] // max(stats['count'], 1)}\n\n")

            # Content by source with strategy info
            sources_content = content_breakdown.get("content_by_source", [])
            if sources_content:
                f.write("SCRAPED CONTENT BY SOURCE (with Strategy Analysis):\n")
                f.write("-" * 80 + "\n")

                for i, source in enumerate(sources_content, 1):
                    f.write(f"\nSOURCE {i}: {source['source_name']}\n")
                    f.write(f"URL: {source['url']}\n")
                    f.write(f"Scraping Method: {source['scraping_method']}\n")
                    f.write(f"Content Length: {source['content_length']} characters\n")
                    f.write(f"Restaurants Found: {source['restaurant_count']}\n")

                    if source.get('restaurants_found'):
                        f.write(f"Restaurant Names: {', '.join(source['restaurants_found'][:5])}\n")
                        if len(source['restaurants_found']) > 5:
                            f.write(f"  ... and {len(source['restaurants_found']) - 5} more\n")

                    f.write(f"Success: {'✓' if source['scraping_success'] else '✗'}\n")

                    strategy_info = source.get('strategy_info', {})
                    f.write(f"AI Strategy Reasoning: {strategy_info.get('reasoning', 'N/A')}\n")
                    f.write(f"AI Confidence: {strategy_info.get('confidence', 0):.2f}\n")
                    f.write(f"Estimated Cost: {strategy_info.get('estimated_cost', 0)} credits\n")

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
                    f.write(f"--- CONTENT SOURCE {i} ({result.get('scraping_method', 'unknown')}) ---\n")
                    f.write(f"URL: {result.get('url', 'Unknown')}\n")
                    f.write(f"Source: {result.get('source_info', {}).get('name', 'Unknown')}\n")
                    f.write(f"Method: {result.get('scraping_method', 'unknown')}\n")
                    if result.get('restaurants_found'):
                        f.write(f"Restaurants Extracted: {', '.join(result['restaurants_found'][:5])}\n")
                    f.write("Content:\n")
                    f.write(scraped_content)
                    f.write("\n\n--- END SOURCE {i} ---\n\n")

            # Statistics with intelligent scraper focus
            f.write("\nSTATISTICS\n")
            f.write("=" * 50 + "\n")

            stats = debug_data.get("statistics", {})
            f.write(f"Total Pipeline Time: {stats.get('total_pipeline_time', 0)}s\n\n")

            # Intelligent scraper stats
            scraper_stats = stats.get("intelligent_scraper_stats", {})
            if scraper_stats:
                f.write("Intelligent Scraper Performance:\n")
                f.write(f"  Total URLs Processed: {scraper_stats.get('total_processed', 0)}\n")
                f.write(f"  Specialized Handler: {scraper_stats.get('specialized_used', 0)} URLs (FREE)\n")
                f.write(f"  Simple HTTP: {scraper_stats.get('simple_http_used', 0)} URLs (0.1 credits each)\n")
                f.write(f"  Enhanced HTTP: {scraper_stats.get('enhanced_http_used', 0)} URLs (0.5 credits each)\n")
                f.write(f"  Firecrawl: {scraper_stats.get('firecrawl_used', 0)} URLs (10 credits each)\n")
                f.write(f"  AI Analysis Calls: {scraper_stats.get('ai_analysis_calls', 0)}\n")
                f.write(f"  Cache Hits: {scraper_stats.get('cache_hits', 0)}\n")
                f.write(f"  Cost Saved: {scraper_stats.get('total_cost_saved', 0)} Firecrawl credits\n\n")

            # Cost efficiency
            cost_efficiency = stats.get("cost_efficiency", {})
            if cost_efficiency:
                f.write("Cost Efficiency Analysis:\n")
                f.write(f"  Firecrawl Usage Rate: {cost_efficiency.get('firecrawl_usage_rate', 0)}%\n")
                f.write(f"  Credits Saved vs All-Firecrawl: {cost_efficiency.get('firecrawl_credits_saved', 0)}\n\n")

            # Content quality summary
            quality = stats.get("content_quality_summary", {})
            if quality:
                f.write("Content Quality Summary:\n")
                for key, value in quality.items():
                    if key != "strategy_effectiveness":
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

                if "strategy_effectiveness" in quality:
                    f.write("  Strategy Effectiveness:\n")
                    for strategy, effectiveness in quality["strategy_effectiveness"].items():
                        f.write(f"    {strategy}: {effectiveness['avg_restaurants_per_source']:.1f} restaurants/source, "
                               f"{effectiveness['avg_content_per_source']:.0f} chars/source\n")

        logger.info(f"Intelligent scraper debug report saved to: {filepath}")
        return filepath

    async def _send_debug_to_admin(self, bot, debug_file_path: str, debug_data: Dict, query: str):
        """Send the debug file to admin Telegram group with intelligent scraper stats"""

        if not self.admin_chat_id:
            logger.warning("No admin chat ID configured, cannot send debug report")
            return

        try:
            # Send summary message with intelligent scraper stats
            stats = debug_data.get("statistics", {})
            quality = stats.get("content_quality_summary", {})
            scraper_stats = stats.get("intelligent_scraper_stats", {})
            cost_efficiency = stats.get("cost_efficiency", {})

            summary_text = (
                f"🧠 <b>Intelligent Scraper Debug Report</b>\n\n"
                f"📝 <b>Query:</b> <code>{query}</code>\n\n"
                f"📊 <b>Results:</b>\n"
                f"• Sources Processed: {quality.get('total_sources_processed', 0)}\n"
                f"• Content Sources: {quality.get('sources_with_content', 0)}\n"
                f"• Extraction Rate: {quality.get('successful_extraction_rate', 0)}%\n"
                f"• Restaurants Found: {quality.get('total_restaurants_extracted', 0)}\n"
                f"• Total Content: {quality.get('average_content_per_source', 0)} chars/source\n"
                f"• Ready for Analyzer: {'✅' if quality.get('content_ready_for_analyzer') else '❌'}\n\n"
                f"🧠 <b>Intelligent Scraper Performance:</b>\n"
                f"• 🆓 Specialized: {scraper_stats.get('specialized_used', 0)} URLs\n"
                f"• 🟢 Simple HTTP: {scraper_stats.get('simple_http_used', 0)} URLs\n"
                f"• 🟡 Enhanced HTTP: {scraper_stats.get('enhanced_http_used', 0)} URLs\n"
                f"• 🔥 Firecrawl: {scraper_stats.get('firecrawl_used', 0)} URLs\n"
                f"• 🤖 AI Analysis Calls: {scraper_stats.get('ai_analysis_calls', 0)}\n"
                f"• 🎯 Cache Hits: {scraper_stats.get('cache_hits', 0)}\n\n"
                f"💰 <b>Cost Efficiency:</b>\n"
                f"• Firecrawl Usage: {cost_efficiency.get('firecrawl_usage_rate', 0)}%\n"
                f"• Credits Saved: {cost_efficiency.get('firecrawl_credits_saved', 0)}\n\n"
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
                    caption=f"🧠 Intelligent scraper debug for: {query}"
                )

            logger.info("Successfully sent intelligent scraper debug report to admin")

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
                "🧠 <b>Intelligent Scraper Pipeline Debug Command</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/debug_query [your restaurant query]</code>\n\n"
                "<b>Examples:</b>\n"
                "<code>/debug_query best cevicherias in Lima</code>\n"
                "<code>/debug_query romantic restaurants in Paris</code>\n"
                "<code>/debug_query family-friendly pizza in Rome</code>\n\n"
                "This will run the complete intelligent scraper pipeline up to the list_analyzer stage "
                "and show you exactly what content gets passed to the AI for analysis.\n\n"
                "📊 <b>New Features:</b>\n"
                "• Shows AI strategy analysis for each URL\n"
                "• Displays cost savings vs Firecrawl-only approach\n"
                "• Reports scraping method distribution\n"
                "• Shows domain intelligence cache\n"
                "• Tracks strategy effectiveness"
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
            f"🧠 Starting intelligent scraper pipeline debug for query:\n<code>{user_query}</code>\n\n"
            "This will run the complete search and intelligent scraping pipeline. "
            "You'll receive a detailed report showing:\n\n"
            "• 🤖 AI analysis decisions for each URL\n"
            "• 📊 Strategy distribution and cost savings\n"
            "• 🎯 Cache hits and domain learning\n"
            "• 📝 Exact content passed to list_analyzer\n\n"
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

                logger.info(f"Intelligent scraper query debug completed. Report saved to: {debug_path}")

            except Exception as e:
                logger.error(f"Error in intelligent scraper query debug: {e}")
                try:
                    bot.send_message(
                        admin_chat_id,
                        f"❌ Intelligent scraper debug failed for '{user_query}': {str(e)}"
                    )
                except:
                    pass

        thread = threading.Thread(target=run_debug, daemon=True)
        thread.start()

    logger.info("Intelligent scraper debug query command added to bot")