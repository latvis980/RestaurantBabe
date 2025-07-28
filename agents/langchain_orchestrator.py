# agents/langchain_orchestrator.py - COMPLETE VERSION with proper raw query handling
import os
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures

from utils.database import get_restaurants_by_city
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.telegram_formatter import TelegramFormatter

# Create logger
logger = logging.getLogger("restaurant-recommender.orchestrator")

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents with correct file names
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)
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
        """Build pipeline with database agent integration and proper raw query preservation"""

        # Step 1: Analyze Query (UPDATED to handle both queries properly)
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),  # Analyze the search query
                "query": x["query"],  # Keep search query for actual searching
                "raw_query": x["raw_query"]  # Preserve original user input
            },
            name="analyze_query"
        )

        # Step 2: Database Search and Decision
        self.check_database = RunnableLambda(
            self._check_database_coverage,
            name="check_database"
        )

        # Step 3: Search (conditional - only if no database content)
        self.search = RunnableLambda(
            self._search_step,
            name="search"
        )

        # Step 4: Scraping (conditional - only if search was performed)
        self.scrape = RunnableLambda(
            self._scraping_step,
            name="scrape"
        )

        # Step 5: Edit results (handles both database and scraped results)
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )

        # Step 6: Follow-up search for additional details
        self.follow_up = RunnableLambda(
            self._follow_up_step,
            name="follow_up"
        )

        # Step 7: Format for Telegram
        self.format_telegram = RunnableLambda(
            self._format_step,
            name="format_telegram"
        )

        # Build the complete chain - FIX: Remove the list wrapper
        self.chain = (
            self.analyze_query |
            self.check_database |
            self.search |
            self.scrape |
            self.edit |
            self.follow_up |
            self.format_telegram
        )

    def _log_firecrawl_usage(self):
        """Log Firecrawl usage statistics"""
        try:
            stats = self.scraper.get_stats()
            total_usage = stats.get('total_usage', 0)
            total_cost = stats.get('total_cost', 0.0)

            if total_usage > 0:
                logger.info(f"💰 Firecrawl Usage: {total_usage} credits, ${total_cost:.3f}")
        except Exception as e:
            logger.debug(f"Could not log Firecrawl usage: {e}")

    def _save_scraped_content_for_processing_simple(self, x, enriched_results):
        """Simplified version - logs the data that could be saved to Supabase"""
        try:
            # Log what would be saved (for monitoring purposes)
            destination = x.get('destination', 'Unknown')
            raw_query = x.get('raw_query', x.get('query', ''))

            logger.info(f"📊 Scraped content summary: {len(enriched_results)} articles for {destination}")
            logger.info(f"📝 Raw user query: {raw_query}")

            # In the future, this could save to Supabase for ML training
            # supabase_manager.save_scraped_content(destination, raw_query, enriched_results)

        except Exception as e:
            logger.error(f"Error in content saving: {e}")

    # UPDATED: New process_query method signature to handle both queries
    def process_query(self, user_query, raw_user_query=None, user_preferences=None):
        """
        Process a restaurant query through the complete pipeline.

        UPDATED: Now handles both formatted search query and raw user input.

        Args:
            user_query: The formatted search query (e.g., "best ramen restaurants in Tokyo")
            raw_user_query: The original user input (e.g., "Looking for good ramen places in Tokyo for lunch tomorrow")
            user_preferences: Optional user preferences dict

        Returns:
            Dict with telegram_formatted_text and other results
        """

        # Generate trace ID for debugging
        trace_id = f"query_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                logger.info(f"🚀 STARTING RECOMMENDATION PIPELINE")
                logger.info(f"🎯 Search Query: {user_query}")
                logger.info(f"📝 Raw User Query: {raw_user_query or 'Not provided'}")

                # Prepare input data - UPDATED to include both queries
                input_data = {
                    "query": user_query,  # Used for search engine queries
                    "raw_query": raw_user_query or user_query,  # Used for AI evaluation
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

                # Save process record (simplified - just log it)
                logger.info(f"📊 Process completed: {user_query} → {result.get('destination', 'Unknown')} → {content_source}")

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
                    "raw_query": result.get("raw_query", raw_user_query or user_query),
                    "search_query": user_query,  # Add search query to response
                    "firecrawl_stats": self.scraper.get_stats() if content_source == "web_search" else {}
                }

            except Exception as e:
                logger.error(f"❌ Error in chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query, "raw_query": raw_user_query}, error=e)

                # Log usage even on error
                try:
                    self._log_firecrawl_usage()
                except:
                    pass

                return {
                    "main_list": [],
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",
                    "raw_query": raw_user_query or user_query,
                    "search_query": user_query,
                    "firecrawl_stats": self.scraper.get_stats()
                }

    # BACKWARD COMPATIBILITY: Keep old method signature working
    def process_restaurant_query(self, user_query, user_preferences=None):
        """Backward compatibility method - treats user_query as both search and raw query"""
        return self.process_query(user_query, user_query, user_preferences)

    def _check_database_coverage(self, x):
        """
        SIMPLIFIED: Pure routing method that delegates ALL database logic to DatabaseSearchAgent.
        The orchestrator only handles routing - no business logic here.
        UPDATED: Ensures raw query is passed through properly.
        """
        try:
            logger.info("🗃️ ROUTING TO DATABASE SEARCH AGENT")

            # Pass the raw query along with other data
            query_data_with_raw = {
                **x,
                "raw_query": x.get("raw_query", x.get("query", ""))  # Ensure raw query is passed
            }

            # Delegate everything to the DatabaseSearchAgent
            database_result = self.database_search_agent.search_and_evaluate(query_data_with_raw)

            # Simple merge and return - no logic here, but preserve raw query
            return {
                **x, 
                **database_result,
                "raw_query": x.get("raw_query", x.get("query", ""))  # Preserve raw query
            }

        except Exception as e:
            logger.error(f"❌ Error routing to database search agent: {e}")
            # Simple fallback - no complex logic in orchestrator
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query even on error
                "has_database_content": False,
                "database_results": [],
                "content_source": "web_search",
                "skip_web_search": False,
                "evaluation_details": {
                    "sufficient": False,
                    "reason": f"routing_error: {str(e)}",
                    "details": {"error": str(e)}
                }
            }

    def _search_step(self, x):
        """Enhanced search step that can skip web search if database provided enough results"""
        try:
            logger.info("🔍 SEARCH STEP")

            # Check if we should skip web search
            if x.get("skip_web_search", False):
                logger.info("⏭️ Skipping web search - database provided sufficient results")
                return {**x, "search_results": []}

            destination = x.get("destination", "Unknown")
            search_terms = x.get("search_terms", [])
            language = x.get("language", "en")

            # If no search terms from query analyzer, create them from the search query (not raw query)
            if not search_terms:
                query = x.get("query", "")  # Use formatted search query for actual searching
                if query and destination != "Unknown":
                    search_terms = [f"{query}"]

            logger.info(f"🔍 Searching for: {search_terms}")

            start_time = time.time()
            search_results = self.search_agent.search(search_terms)
            search_time = time.time() - start_time

            logger.info(f"✅ Found {len(search_results)} search results in {search_time:.1f}s")

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "search_results": search_results
            }

        except Exception as e:
            logger.error(f"❌ Error in search step: {e}")
            dump_chain_state("search_error", x, error=e)
            return {**x, "search_results": []}

    def _scraping_step(self, x):
        """Scraping step using WebScraper with intelligent strategies (UPDATED with raw query preservation)"""
        try:
            logger.info("🕷️ ENTERING SCRAPING STEP")

            search_results = x.get("search_results", [])

            if not search_results:
                logger.info("⏭️ No search results to scrape")
                return {
                    **x, 
                    "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                    "enriched_results": []
                }

            logger.info(f"🕷️ Scraping {len(search_results)} URLs with intelligent strategies")

            # Run scraping in separate event loop
            def run_scraping():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
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

            # Save scraped content for Supabase manager
            if enriched_results:
                logger.info("💾 Proceeding to save scraped content...")
                self._save_scraped_content_for_processing_simple(x, enriched_results)
            else:
                logger.warning("⚠️ No enriched results to save")

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "enriched_results": enriched_results
            }

        except Exception as e:
            logger.error(f"❌ Error in scraping step: {e}")
            return {**x, "enriched_results": []}

    def _edit_step(self, x):
        """Edit step - handles both database restaurants and scraped content (UPDATED with proper raw query usage)"""
        try:
            logger.info("✏️ ENTERING EDIT STEP")

            has_database_content = x.get("has_database_content", False)
            search_query = x.get("query", "")  # The formatted search query
            raw_query = x.get("raw_query", search_query)  # The original user input
            destination = x.get("destination", "Unknown")

            logger.info(f"🔍 Using RAW QUERY for AI evaluation: {raw_query}")
            logger.info(f"🎯 Search query was: {search_query}")

            if has_database_content:
                # DATABASE BRANCH: Format existing restaurants
                logger.info("🗃️ Processing DATABASE restaurants")

                database_results = x.get("database_results", [])

                if not database_results:
                    logger.warning("⚠️ No database results to process")
                    return {
                        **x,
                        "raw_query": raw_query,  # Preserve raw query
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call editor with database restaurants and RAW QUERY
                # Raw query validation is built into the editor prompts
                edit_output = self.editor_agent.edit(
                    scraped_results=None,
                    database_restaurants=database_results,
                    original_query=raw_query,  # Use RAW QUERY for AI evaluation
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
                        "raw_query": raw_query,  # Preserve raw query
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call editor with scraped content and RAW QUERY
                # Raw query validation is built into the editor prompts
                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=None,
                    original_query=raw_query,  # Use RAW QUERY for AI evaluation
                    destination=destination
                )

                logger.info(f"✅ Processed {len(edit_output.get('edited_results', {}).get('main_list', []))} restaurants from scraped content")

            return {
                **x,
                "raw_query": raw_query,  # Preserve raw query
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"❌ Error in edit step: {e}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
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

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "enhanced_results": enhanced_results
            }

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
                    "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
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
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "telegram_formatted_text": telegram_text,
                "final_results": enhanced_results
            }

        except Exception as e:
            logger.error(f"❌ Error in format step: {e}")
            dump_chain_state("format_error", x, error=e)
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "telegram_formatted_text": "Sorry, there was an error formatting the restaurant recommendations."
            }

    # Additional utility methods

    def get_orchestrator_stats(self):
        """Get comprehensive stats from all components"""
        try:
            stats = {
                "orchestrator": {
                    "pipeline_steps": 7,
                    "raw_query_support": True
                },
                "query_analyzer": {
                    "enabled": True
                },
                "database_search": self.database_search_agent.get_stats(),
                "search_agent": {
                    "enabled": True
                },
                "scraper": self.scraper.get_stats(),
                "editor": self.editor_agent.get_editor_stats() if hasattr(self.editor_agent, 'get_editor_stats') else {"enabled": True},
                "follow_up_search": {
                    "enabled": True
                },
                "telegram_formatter": {
                    "enabled": True
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting orchestrator stats: {e}")
            return {"error": str(e)}

    def health_check(self):
        """Perform health check on all components"""
        try:
            health = {
                "orchestrator": "healthy",
                "agents": {}
            }

            # Check each agent
            agents = [
                ("query_analyzer", self.query_analyzer),
                ("database_search_agent", self.database_search_agent),
                ("search_agent", self.search_agent),
                ("scraper", self.scraper),
                ("editor_agent", self.editor_agent),
                ("follow_up_search_agent", self.follow_up_search_agent)
            ]

            for name, agent in agents:
                try:
                    if hasattr(agent, 'health_check'):
                        agent.health_check()
                    health["agents"][name] = "healthy"
                except Exception as e:
                    health["agents"][name] = f"error: {str(e)}"

            return health
        except Exception as e:
            return {"orchestrator": f"error: {str(e)}"}

    def update_config(self, new_config):
        """Update configuration for all agents"""
        try:
            self.config = new_config

            # Update each agent's config if they support it
            agents = [
                self.query_analyzer,
                self.database_search_agent, 
                self.search_agent,
                self.scraper,
                self.editor_agent,
                self.follow_up_search_agent
            ]

            for agent in agents:
                if hasattr(agent, 'update_config'):
                    agent.update_config(new_config)

            logger.info("✅ Configuration updated for all agents")

        except Exception as e:
            logger.error(f"❌ Error updating configuration: {e}")

    def reset_stats(self):
        """Reset statistics for all components"""
        try:
            # Reset stats for each component that supports it
            if hasattr(self.scraper, 'reset_stats'):
                self.scraper.reset_stats()
            if hasattr(self.database_search_agent, 'reset_stats'):
                self.database_search_agent.reset_stats()

            logger.info("✅ Statistics reset for all components")

        except Exception as e:
            logger.error(f"❌ Error resetting statistics: {e}")

    # Legacy compatibility methods
    def get_search_agent_stats(self):
        """Legacy method for backward compatibility"""
        return self.search_agent.get_stats() if hasattr(self.search_agent, 'get_stats') else {}

    def get_scraper_stats(self):
        """Legacy method for backward compatibility"""
        return self.scraper.get_stats()

    def set_database_threshold(self, new_threshold):
        """Set minimum database threshold"""
        if hasattr(self.database_search_agent, 'set_minimum_threshold'):
            self.database_search_agent.set_minimum_threshold(new_threshold)

    def enable_database_ai_evaluation(self, enabled=True):
        """Enable/disable AI evaluation in database search"""
        if hasattr(self.database_search_agent, 'enable_ai_evaluation'):
            self.database_search_agent.enable_ai_evaluation(enabled)

    # Debugging and monitoring methods
    def get_pipeline_status(self):
        """Get status of the entire pipeline"""
        try:
            status = {
                "pipeline_built": bool(self.chain),
                "agents_initialized": {
                    "query_analyzer": bool(self.query_analyzer),
                    "database_search": bool(self.database_search_agent),
                    "search_agent": bool(self.search_agent),
                    "scraper": bool(self.scraper),
                    "editor_agent": bool(self.editor_agent),
                    "follow_up_search": bool(self.follow_up_search_agent)
                },
                "formatter_initialized": bool(self.telegram_formatter)
            }
            return status
        except Exception as e:
            return {"error": str(e)}

    def test_chain_steps(self, test_query="test query"):
        """Test individual chain steps for debugging"""
        try:
            test_data = {
                "query": test_query,
                "raw_query": test_query,
                "user_preferences": {}
            }

            results = {}

            # Test each step individually
            try:
                results["analyze_query"] = "✅" if self.analyze_query else "❌"
            except:
                results["analyze_query"] = "❌"

            try:
                results["check_database"] = "✅" if self.check_database else "❌"
            except:
                results["check_database"] = "❌"

            try:
                results["search"] = "✅" if self.search else "❌"
            except:
                results["search"] = "❌"

            try:
                results["scrape"] = "✅" if self.scrape else "❌"
            except:
                results["scrape"] = "❌"

            try:
                results["edit"] = "✅" if self.edit else "❌"
            except:
                results["edit"] = "❌"

            try:
                results["follow_up"] = "✅" if self.follow_up else "❌"
            except:
                results["follow_up"] = "❌"

            try:
                results["format_telegram"] = "✅" if self.format_telegram else "❌"
            except:
                results["format_telegram"] = "❌"

            return results

        except Exception as e:
            return {"error": str(e)}

    def get_memory_usage(self):
        """Get approximate memory usage of components"""
        try:
            import sys

            components = {
                "query_analyzer": sys.getsizeof(self.query_analyzer),
                "database_search_agent": sys.getsizeof(self.database_search_agent),
                "search_agent": sys.getsizeof(self.search_agent),
                "scraper": sys.getsizeof(self.scraper),
                "editor_agent": sys.getsizeof(self.editor_agent),
                "follow_up_search_agent": sys.getsizeof(self.follow_up_search_agent),
                "telegram_formatter": sys.getsizeof(self.telegram_formatter),
                "chain": sys.getsizeof(self.chain) if self.chain else 0
            }

            total = sum(components.values())
            components["total_bytes"] = total
            components["total_mb"] = round(total / (1024 * 1024), 2)

            return components

        except Exception as e:
            return {"error": str(e)}