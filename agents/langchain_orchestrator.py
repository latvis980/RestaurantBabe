# agents/langchain_orchestrator.py
# FIXED VERSION - Now passes destination to follow-up search

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
        # Import agents
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

        # Step 7: Format for Telegram (SIMPLIFIED)
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
        """Edit step with error handling"""
        try:
            dump_chain_state("pre_edit", {
                "recommendations_keys": list(x.get("recommendations", {}).keys()),
                "query": x.get("query", "")
            })

            recommendations = x.get("recommendations", {})
            formatted_results = self.editor_agent.edit(recommendations, x["query"])

            dump_chain_state("post_edit", {
                "formatted_results_keys": list(formatted_results.keys() if formatted_results else {})
            })

            return {**x, "formatted_recommendations": formatted_results}

        except Exception as e:
            logger.error(f"Error in editor step: {e}")
            dump_chain_state("editor_error", x, error=e)
            return {
                **x,
                "formatted_recommendations": {
                    "formatted_recommendations": x.get("recommendations", {})
                }
            }

    def _follow_up_step(self, x):
        """Follow-up search step

        FIXED: Now passes the destination parameter to the follow-up search agent
        """
        try:
            dump_chain_state("pre_follow_up", {
                "formatted_recommendations_keys": list(x.get("formatted_recommendations", {}).keys()),
                "destination": x.get("destination", "Unknown")  # Log destination
            })

            formatted_recs = x.get("formatted_recommendations", {})

            # Extract the actual recommendations
            if "formatted_recommendations" in formatted_recs:
                actual_recs = formatted_recs.get("formatted_recommendations", {})
            else:
                actual_recs = formatted_recs

            # Get follow up queries and parameters
            follow_up_queries = formatted_recs.get("follow_up_queries", [])
            secondary_params = x.get("secondary_filter_parameters", [])

            # FIXED: Extract destination from the query analysis
            destination = x.get("destination", "Unknown")

            # Execute follow up search with destination parameter
            enhanced_recommendations = self.follow_up_search_agent.perform_follow_up_searches(
                actual_recs,
                follow_up_queries,
                destination,  # FIXED: Pass destination as third parameter
                secondary_params
            )

            dump_chain_state("post_follow_up", {
                "enhanced_recommendations_keys": list(enhanced_recommendations.keys() if enhanced_recommendations else {}),
                "destination_used": destination
            })

            return {**x, "enhanced_recommendations": enhanced_recommendations}

        except Exception as e:
            logger.error(f"Error in follow-up step: {e}")
            dump_chain_state("follow_up_error", x, error=e)
            return {
                **x,
                "enhanced_recommendations": {
                    "main_list": x.get("recommendations", {}).get("main_list", [])
                }
            }

    def _format_step(self, x):
        """Format the results for Telegram"""
        try:
            enhanced_recommendations = x.get("enhanced_recommendations", {})
            main_list = enhanced_recommendations.get("main_list", [])

            if not main_list:
                return {
                    **x,
                    "telegram_formatted_text": "Sorry, no restaurants found matching your criteria."
                }

            # Format for Telegram
            telegram_text = self.telegram_formatter.format(enhanced_recommendations)

            dump_chain_state("format_complete", {
                "restaurants_formatted": len(main_list),
                "text_length": len(telegram_text)
            })

            return {
                **x,
                "telegram_formatted_text": telegram_text
            }

        except Exception as e:
            logger.error(f"Error in format step: {e}")
            dump_chain_state("format_error", x, error=e)
            return {
                **x,
                "telegram_formatted_text": "Sorry, there was an error formatting the recommendations."
            }

    def _log_firecrawl_usage(self):
        """Log Firecrawl usage statistics"""
        try:
            stats = self.scraper.get_stats()
            logger.info(f"Firecrawl usage - Total calls: {stats.get('total_calls', 0)}, "
                       f"Credits used: {stats.get('credits_used', 0)}, "
                       f"Success rate: {stats.get('success_rate', 0):.1%}")
        except Exception as e:
            logger.warning(f"Error logging Firecrawl stats: {e}")

    def _extract_user_preferences(self, query):
        """Extract explicit preferences from the query"""
        # Simple extraction for now - can be enhanced later
        explicit_prefs = []
        clean_query = query

        # Extract dietary restrictions and preferences
        pref_indicators = [
            "vegetarian", "vegan", "halal", "kosher", "gluten-free", "organic",
            "farm-to-table", "local", "sustainable", "seafood", "steakhouse",
            "fine dining", "casual", "romantic", "family-friendly", "outdoor seating"
        ]

        for pref in pref_indicators:
            if pref in query.lower():
                explicit_prefs.append(pref)

        return clean_query, explicit_prefs

    def process_query(self, user_query: str, standing_prefs: list = None):
        """
        Process a user query using the LangChain sequence

        Args:
            user_query (str): The user's query about restaurant recommendations
            standing_prefs (list, optional): List of user's standing preferences

        Returns:
            dict: The formatted recommendations for Telegram
        """
        # Extract user preferences if included in the query
        clean_query, explicit_prefs = self._extract_user_preferences(user_query)
        user_preferences = list(set(explicit_prefs + (standing_prefs or [])))

        # Create trace ID for this request
        trace_id = f"restaurant_rec_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                logger.info(f"Processing query: {clean_query}")
                logger.info(f"User preferences: {user_preferences}")

                # Create input data
                input_data = {
                    "query": clean_query, 
                    "user_preferences": user_preferences
                }

                # Execute the chain
                result = self.chain.invoke(input_data)

                # Log completion
                dump_chain_state("process_query_complete", {
                    "result_keys": list(result.keys()),
                    "has_recommendations": "enhanced_recommendations" in result,
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

                # Extract results - FIXED KEY NAME!
                telegram_text = result.get("telegram_formatted_text", 
                                         "Sorry, no recommendations found.")

                enhanced_recommendations = result.get("enhanced_recommendations", {})
                main_list = enhanced_recommendations.get("main_list", [])

                logger.info(f"Final result - Main list: {len(main_list)} restaurants for {result.get('destination', 'Unknown')}")

                # FIXED: Return the correct key name that telegram_bot.py expects
                return {
                    "telegram_formatted_text": telegram_text,  # ← FIXED: Changed from "telegram_text"
                    "enhanced_recommendations": enhanced_recommendations,
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
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",  # ← FIXED: Changed from "telegram_text"
                    "firecrawl_stats": self.scraper.get_stats()
                }