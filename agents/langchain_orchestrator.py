# agents/langchain_orchestrator.py
# OPTIMIZED VERSION - Database branches + scraped content handling, no RAG/Supabase

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures

from utils.database import save_data, get_restaurants_by_city
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
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents (no list_analyzer - it's redundant)
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter()

        self.config = config

        # Build the pipeline steps
        self._build_pipeline()

    def _build_pipeline(self):
        """Build pipeline with database branches"""

        # Step 1: Analyze Query
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"]
            },
            name="analyze_query"
        )

        # Step 2: Check Database Coverage
        self.check_database = RunnableLambda(
            self._check_database_coverage,
            name="check_database"
        )

        # Step 3: Search (conditional - only if no database content)
        self.search = RunnableLambda(
            self._search_step,
            name="search"
        )

        # Step 4: Scrape (conditional - only if search happened)
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )

        # Step 5: Edit (handles both database restaurants and scraped content)
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
                self.check_database,
                self.search,
                self.scrape,
                self.edit,
                self.follow_up_search,
                self.format_output,
            ],
            last=RunnableLambda(lambda x: x),
            name="restaurant_recommendation_chain"
        )

    def _check_database_coverage(self, x):
        """Check if we have restaurants in database for this destination"""
        try:
            logger.info("🗃️ CHECKING DATABASE COVERAGE")

            destination = x.get("destination", "Unknown")

            if destination == "Unknown":
                logger.info("⚠️ No destination detected, will search web")
                return {**x, "has_database_content": False, "database_results": []}

            # Extract city from destination (simple parsing)
            city = destination
            if "," in destination:
                city = destination.split(",")[0].strip()

            logger.info(f"🔍 Checking database for: {city}")

            # Query database for existing restaurants
            database_restaurants = get_restaurants_by_city(city, self.config)

            # Decide if we have enough content (threshold: 3+ restaurants)
            if database_restaurants and len(database_restaurants) >= 3:
                logger.info(f"✅ Found {len(database_restaurants)} restaurants in database")
                logger.info("📊 Using DATABASE BRANCH - skipping web search")
                return {
                    **x,
                    "has_database_content": True,
                    "database_results": database_restaurants,
                    "content_source": "database"
                }
            else:
                logger.info(f"⚠️ Only {len(database_restaurants) if database_restaurants else 0} restaurants in database")
                logger.info("🌐 Using WEB SEARCH BRANCH")
                return {
                    **x,
                    "has_database_content": False,
                    "database_results": [],
                    "content_source": "web_search"
                }

        except Exception as e:
            logger.error(f"❌ Error checking database: {e}")
            return {
                **x,
                "has_database_content": False,
                "database_results": [],
                "content_source": "web_search"
            }

    def _search_step(self, x):
        """Search step - only runs if no database content"""
        try:
            # Check if we should skip search (database branch)
            if x.get("has_database_content", False):
                logger.info("⏭️ SKIPPING SEARCH - using database content")
                return {**x, "search_results": []}

            logger.info("🔍 RUNNING WEB SEARCH")

            search_queries = x.get("search_queries", [])

            if not search_queries:
                logger.warning("⚠️ No search queries available")
                return {**x, "search_results": []}

            search_results = self.search_agent.search(search_queries)

            logger.info(f"🌐 Found {len(search_results)} search results")

            return {**x, "search_results": search_results}

        except Exception as e:
            logger.error(f"❌ Error in search step: {e}")
            return {**x, "search_results": []}

    def _scrape_step(self, x):
        """Scrape step - only runs if search happened"""
        try:
            # Check if we should skip scraping (database branch)
            if x.get("has_database_content", False):
                logger.info("⏭️ SKIPPING SCRAPING - using database content")
                return {**x, "enriched_results": []}

            logger.info("🕷️ RUNNING WEB SCRAPING")

            search_results = x.get("search_results", [])

            if not search_results:
                logger.warning("⚠️ No search results to scrape")
                return {**x, "enriched_results": []}

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

            logger.info(f"✅ Scraped {len(enriched_results)} articles")
            return {**x, "enriched_results": enriched_results}

        except Exception as e:
            logger.error(f"❌ Error in scraping step: {e}")
            return {**x, "enriched_results": []}

    def _edit_step(self, x):
        """Edit step - handles both database restaurants and scraped content"""
        try:
            logger.info("✏️ ENTERING EDIT STEP")

            has_database_content = x.get("has_database_content", False)
            original_query = x.get("query", "")
            destination = x.get("destination", "Unknown")

            if has_database_content:
                # DATABASE BRANCH: Format existing restaurants
                logger.info("🗃️ Processing DATABASE restaurants")

                database_results = x.get("database_results", [])

                if not database_results:
                    logger.warning("⚠️ No database results to process")
                    return {
                        **x,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call editor with database restaurants
                edit_output = self.editor_agent.edit(
                    scraped_results=None,
                    database_restaurants=database_results,
                    original_query=original_query,
                    destination=destination
                )

                logger.info(f"✅ Formatted {len(edit_output.get('edited_results', {}).get('main_list', []))} database restaurants")

            else:
                # WEB SEARCH BRANCH: Process scraped content
                logger.info("🌐 Processing SCRAPED content")

                scraped_results = x.get("enriched_results", [])

                if not scraped_results:
                    logger.warning("⚠️ No scraped results to process")
                    return {
                        **x,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call editor with scraped content
                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=None,
                    original_query=original_query,
                    destination=destination
                )

                logger.info(f"✅ Processed {len(edit_output.get('edited_results', {}).get('main_list', []))} restaurants from scraped content")

            return {
                **x, 
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"❌ Error in edit step: {e}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)
            return {
                **x,
                "edited_results": {"main_list": []},
                "follow_up_queries": []
            }

    def _follow_up_step(self, x):
        """Follow-up search step - processes edited_results and returns enhanced_results"""
        try:
            logger.info("🔍 ENTERING FOLLOW-UP STEP")

            edited_results = x.get("edited_results", {})
            follow_up_queries = x.get("follow_up_queries", [])

            if not edited_results.get("main_list"):
                logger.warning("⚠️ No restaurants available for follow-up search")
                return {**x, "enhanced_results": {"main_list": []}}

            logger.info(f"🔍 Processing {len(edited_results['main_list'])} restaurants for follow-up")

            # Call follow-up search with edited results
            followup_output = self.follow_up_search_agent.perform_follow_up_searches(
                edited_results=edited_results,
                follow_up_queries=follow_up_queries,
                destination=x.get("destination", "Unknown"),
                secondary_filter_parameters=x.get("secondary_filter_parameters")
            )

            enhanced_results = followup_output.get("enhanced_results", {"main_list": []})

            logger.info(f"✅ Follow-up complete: {len(enhanced_results.get('main_list', []))} restaurants remain after filtering")

            return {**x, "enhanced_results": enhanced_results}

        except Exception as e:
            logger.error(f"❌ Error in follow-up step: {e}")
            dump_chain_state("follow_up_error", x, error=e)
            return {**x, "enhanced_results": {"main_list": []}}

    def _format_step(self, x):
        """Format step - converts enhanced_results to telegram_formatted_text"""
        try:
            logger.info("📱 ENTERING FORMAT STEP")

            enhanced_results = x.get("enhanced_results", {})
            main_list = enhanced_results.get("main_list", [])

            if not main_list:
                logger.warning("⚠️ No restaurants to format for Telegram")
                return {
                    **x,
                    "telegram_formatted_text": "Sorry, no restaurant recommendations found for your query."
                }

            logger.info(f"📱 Formatting {len(main_list)} restaurants for Telegram")

            # Format for Telegram using the formatter
            telegram_text = self.telegram_formatter.format_recommendations(
                enhanced_results  # Pass the entire enhanced_results dict
            )

            logger.info("✅ Telegram formatting complete")

            return {
                **x,
                "telegram_formatted_text": telegram_text,
                "final_results": enhanced_results
            }

        except Exception as e:
            logger.error(f"❌ Error in format step: {e}")
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
                logger.info(f"🚀 STARTING RECOMMENDATION PIPELINE")
                logger.info(f"Query: {user_query}")

                # Prepare input data
                input_data = {
                    "query": user_query,
                    "user_preferences": user_preferences or {}
                }

                # Execute the chain
                result = self.chain.invoke(input_data)

                # Log completion
                content_source = result.get("content_source", "unknown")
                logger.info("✅ PIPELINE COMPLETE")
                logger.info(f"📊 Content source: {content_source}")

                # Final usage summary (only if we used scraping)
                if content_source == "web_search":
                    self._log_firecrawl_usage()

                # Save process record
                process_record = {
                    "query": user_query,
                    "destination": result.get("destination", "Unknown"),
                    "content_source": content_source,
                    "trace_id": trace_id,
                    "timestamp": time.time(),
                    "firecrawl_stats": self.scraper.get_stats() if content_source == "web_search" else {}
                }

                save_data(self.config.DB_TABLE_PROCESSES, process_record, self.config)

                # Extract results with correct key names
                telegram_text = result.get("telegram_formatted_text", 
                                         "Sorry, no recommendations found.")

                enhanced_results = result.get("enhanced_results", {})
                main_list = enhanced_results.get("main_list", [])

                logger.info(f"📊 Final result: {len(main_list)} restaurants for {result.get('destination', 'Unknown')}")
                logger.info(f"📊 Source: {content_source}")

                # Return with correct key names that telegram_bot.py expects
                return {
                    "telegram_formatted_text": telegram_text,
                    "enhanced_results": enhanced_results,
                    "main_list": main_list,
                    "destination": result.get("destination"),
                    "content_source": content_source,
                    "firecrawl_stats": self.scraper.get_stats() if content_source == "web_search" else {}
                }

            except Exception as e:
                logger.error(f"❌ Error in chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query}, error=e)

                # Log usage even on error
                try:
                    self._log_firecrawl_usage()
                except:
                    pass

                return {
                    "main_list": [],
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",
                    "firecrawl_stats": self.scraper.get_stats()
                }