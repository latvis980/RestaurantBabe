# agents/langchain_orchestrator.py
# CORRECTED VERSION - Uses proper file names and logical data flow

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures

from utils.database import save_data
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.telegram_formatter import TelegramFormatter

# Create logger
logger = logging.getLogger("restaurant-recommender.orchestrator")

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents with correct file names
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.list_analyzer import ListAnalyzer
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.list_analyzer = ListAnalyzer(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter()

        self.config = config

        # Build the pipeline steps
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the LangChain pipeline with clean step separation"""

        # Step 1: Analyze Query
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"]
            },
            name="analyze_query"
        )

        # Step 2: Search
        self.search = RunnableLambda(
            lambda x: {
                **x,
                "search_results": self.search_agent.search(x["search_queries"])
            },
            name="search"
        )

        # Step 3: Scrape
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )

        # Step 4: Analyze Results
        self.analyze_results = RunnableLambda(
            self._analyze_results_step,
            name="analyze_results"
        )

        # Step 5: Edit
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )

        # Step 6: Follow-up Search
        self.follow_up_search = RunnableLambda(
            self._follow_up_step,
            name="follow_up_search"
        )

        # Step 7: Format for Telegram
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )

        # Create the complete chain
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.search,
                self.scrape,
                self.analyze_results,
                self.edit,
                self.follow_up_search,
                self.format_output,
            ],
            last=RunnableLambda(lambda x: x),
            name="restaurant_recommendation_chain"
        )

    def _scrape_step(self, x):
        """Handle async scraping with proper event loop management"""
        search_results = x.get("search_results", [])
        logger.info(f"Scraping {len(search_results)} search results")

        def run_scraping():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.scraper.scrape_search_results(search_results)
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            enriched_results = pool.submit(run_scraping).result()

        # Log usage after scraping
        self._log_firecrawl_usage()

        logger.info(f"Scraping completed with {len(enriched_results)} enriched results")
        return {**x, "enriched_results": enriched_results}

    def _analyze_results_step(self, x):
        """Handle async result analysis"""
        def run_analysis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._analyze_results_async(x))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_analysis).result()

    async def _analyze_results_async(self, x):
        """Async analysis of search results"""
        try:
            dump_chain_state("pre_analyze_results", {
                "enriched_results_count": len(x.get("enriched_results", [])),
                "keywords": x.get("keywords_for_analysis", []),
                "destination": x.get("destination", "Unknown")
            })

            recommendations = await self.list_analyzer.analyze(
                search_results=x["enriched_results"],
                keywords_for_analysis=x.get("keywords_for_analysis", []),
                primary_search_parameters=x.get("primary_search_parameters", []),
                secondary_filter_parameters=x.get("secondary_filter_parameters", []),
                destination=x.get("destination")
            )

            # Standardize the recommendations structure
            standardized = self._standardize_recommendations(recommendations)

            return {**x, "recommendations": standardized}

        except Exception as e:
            logger.error(f"Error in analyze_results: {e}")
            dump_chain_state("analyze_results_error", x, error=e)
            return {**x, "recommendations": {"main_list": []}}

    def _standardize_recommendations(self, recommendations):
        """Convert recommendations to standard format"""
        if not isinstance(recommendations, dict):
            return {"main_list": []}

        all_restaurants = []

        # Get restaurants from main_list
        main_list = recommendations.get("main_list", [])
        if isinstance(main_list, list):
            all_restaurants.extend(main_list)

        # Get restaurants from hidden_gems and add to main list
        hidden_gems = recommendations.get("hidden_gems", [])
        if isinstance(hidden_gems, list):
            all_restaurants.extend(hidden_gems)

        # Handle legacy format
        if "recommended" in recommendations and not all_restaurants:
            recommended = recommendations.get("recommended", [])
            if isinstance(recommended, list):
                all_restaurants.extend(recommended)

        return {"main_list": all_restaurants}

    def _edit_step(self, x):
        """Edit step - processes scraped_results and returns edited_results"""
        try:
            dump_chain_state("pre_edit", {
                "available_keys": list(x.keys()),
                "enriched_results_count": len(x.get("enriched_results", [])),
                "query": x.get("query", "")
            })

            # Get the scraped results from previous step
            scraped_results = x.get("enriched_results", [])  # enriched_results = scraped_results
            original_query = x.get("query", "")
            destination = x.get("destination", "Unknown")

            if not scraped_results:
                logger.warning("No scraped results available for editing")
                return {
                    **x,
                    "edited_results": {"main_list": []},
                    "follow_up_queries": []
                }

            # Call the editor with scraped results
            edit_output = self.editor_agent.edit(
                scraped_results=scraped_results,
                original_query=original_query,
                destination=destination
            )

            dump_chain_state("post_edit", {
                "edit_output_keys": list(edit_output.keys() if edit_output else {}),
                "main_list_count": len(edit_output.get("edited_results", {}).get("main_list", []))
            })

            return {
                **x, 
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"Error in edit step: {e}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)

            # Return fallback response
            return {
                **x,
                "edited_results": {"main_list": []},
                "follow_up_queries": []
            }

    def _follow_up_step(self, x):
        """Follow-up search step - processes edited_results and returns enhanced_results"""
        try:
            dump_chain_state("pre_follow_up", {
                "edited_results_keys": list(x.get("edited_results", {}).keys()),
                "destination": x.get("destination", "Unknown")
            })

            edited_results = x.get("edited_results", {})
            follow_up_queries = x.get("follow_up_queries", [])

            if not edited_results.get("main_list"):
                logger.warning("No restaurants available for follow-up search")
                return {**x, "enhanced_results": {"main_list": []}}

            # Call follow-up search with edited results
            followup_output = self.follow_up_search_agent.perform_follow_up_searches(
                edited_results=edited_results,
                follow_up_queries=follow_up_queries,
                destination=x.get("destination", "Unknown"),
                secondary_filter_parameters=x.get("secondary_filter_parameters")
            )

            enhanced_results = followup_output.get("enhanced_results", {"main_list": []})

            dump_chain_state("post_follow_up", {
                "enhanced_count": len(enhanced_results.get("main_list", [])),
                "destination": x.get("destination", "Unknown")
            })

            return {**x, "enhanced_results": enhanced_results}

        except Exception as e:
            logger.error(f"Error in follow-up step: {e}")
            dump_chain_state("follow_up_error", x, error=e)
            return {**x, "enhanced_results": {"main_list": []}}

    def _format_step(self, x):
        """Format step - converts enhanced_results to telegram_formatted_text"""
        try:
            dump_chain_state("pre_format", {
                "enhanced_results_keys": list(x.get("enhanced_results", {}).keys()),
                "destination": x.get("destination", "Unknown")
            })

            enhanced_results = x.get("enhanced_results", {})
            main_list = enhanced_results.get("main_list", [])

            if not main_list:
                logger.warning("No restaurants to format for Telegram")
                return {
                    **x,
                    "telegram_formatted_text": "Sorry, no restaurant recommendations found for your query."
                }

            # Format for Telegram using the formatter
            telegram_text = self.telegram_formatter.format_restaurants(
                restaurants=main_list,
                destination=x.get("destination", "Unknown"),
                original_query=x.get("query", "")
            )

            dump_chain_state("post_format", {
                "telegram_text_length": len(telegram_text),
                "restaurant_count": len(main_list)
            })

            return {
                **x,
                "telegram_formatted_text": telegram_text,
                "final_results": enhanced_results  # Keep the enhanced results for any further processing
            }

        except Exception as e:
            logger.error(f"Error in format step: {e}")
            dump_chain_state("format_error", x, error=e)
            return {
                **x,
                "telegram_formatted_text": "Sorry, there was an error formatting the restaurant recommendations."
            }

    def _log_firecrawl_usage(self):
        """Log Firecrawl usage statistics"""
        try:
            stats = self.scraper.get_stats()
            logger.info("=" * 50)
            logger.info("FIRECRAWL USAGE REPORT")
            logger.info("=" * 50)
            logger.info(f"URLs scraped: {stats.get('total_scraped', 0)}")
            logger.info(f"Successful extractions: {stats.get('successful_extractions', 0)}")
            logger.info(f"Credits used: {stats.get('credits_used', 0)}")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error logging Firecrawl usage: {e}")

    @log_function_call
    def process_query(self, user_query: str, user_preferences: dict = None) -> dict:
        """
        Process a restaurant query through the complete pipeline.

        Args:
            user_query: The user's restaurant request
            user_preferences: Optional user preferences dict

        Returns:
            Dict with telegram_formatted_text and other results
        """

        # Generate trace ID for debugging
        trace_id = f"query_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                dump_chain_state("process_query_start", {"query": user_query, "trace_id": trace_id})

                # Prepare input data
                input_data = {
                    "query": user_query,
                    "user_preferences": user_preferences or {}
                }

                # Execute the chain
                result = self.chain.invoke(input_data)

                # Log completion
                dump_chain_state("process_query_complete", {
                    "result_keys": list(result.keys()),
                    "has_enhanced_results": "enhanced_results" in result,
                    "has_telegram_text": "telegram_formatted_text" in result,
                    "destination": result.get("destination", "Unknown")
                })

                # Final usage summary
                self._log_firecrawl_usage()

                # Save process record
                process_record = {
                    "query": user_query,
                    "destination": result.get("destination", "Unknown"),
                    "trace_id": trace_id,
                    "timestamp": time.time(),
                    "firecrawl_stats": self.scraper.get_stats()
                }

                save_data(self.config.DB_TABLE_PROCESSES, process_record, self.config)

                # Extract results with correct key names
                telegram_text = result.get("telegram_formatted_text", 
                                         "Sorry, no recommendations found.")

                enhanced_results = result.get("enhanced_results", {})
                main_list = enhanced_results.get("main_list", [])

                logger.info(f"Final result - Main list: {len(main_list)} restaurants for {result.get('destination', 'Unknown')}")

                # Return with correct key names that telegram_bot.py expects
                return {
                    "telegram_formatted_text": telegram_text,
                    "enhanced_results": enhanced_results,
                    "main_list": main_list,
                    "destination": result.get("destination"),
                    "firecrawl_stats": self.scraper.get_stats()
                }

            except Exception as e:
                logger.error(f"Error in chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query}, error=e)
                self._log_firecrawl_usage()

                return {
                    "main_list": [],
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",
                    "firecrawl_stats": self.scraper.get_stats()
                }