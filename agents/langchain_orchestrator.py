# agents/langchain_orchestrator.py
# UPDATED VERSION - Now uses SmartRestaurantScraper directly (no legacy wrapper)
# FIRECRAWL CLEANUP - Removed all Firecrawl remnants

import os
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import asyncio
import logging
import concurrent.futures
import threading
import requests
from datetime import datetime

from utils.database import get_restaurants_by_city
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.telegram_formatter import TelegramFormatter

# Create logger
logger = logging.getLogger("restaurant-recommender.orchestrator")

class LangChainOrchestrator:
    """
    Enhanced LangChain orchestrator with Smart Restaurant Scraper integration.

    Key improvements:
    - Better restaurant extraction through optimized content
    - Comprehensive monitoring and cost tracking
    - Drop-in compatibility with existing system

    """

    def __init__(self, config):
        # Import agents with correct names - NO MORE LEGACY WRAPPERS
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from agents.dbcontent_evaluation_agent import ContentEvaluationAgent
        from agents.search_agent import BraveSearchAgent
        from agents.smart_scraper import SmartRestaurantScraper  # DIRECT IMPORT
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent
        from agents.text_cleaner_agent import TextCleanerAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)
        self.dbcontent_evaluation_agent = ContentEvaluationAgent(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = SmartRestaurantScraper(config) 
        self._text_cleaner = TextCleanerAgent(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)
        self.dbcontent_evaluation_agent.set_brave_search_agent(self.search_agent)

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter(config)

        self.config = config

        from utils.supabase_storage import get_storage_manager
        self.storage_manager = get_storage_manager()

        if self.storage_manager:
            logger.info("✅ Orchestrator connected to storage manager")
        else:
            logger.warning("⚠️ Storage manager not available for orchestrator")

        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_processing_time": 0.0,
            "content_sources": {"database": 0, "web": 0},
            "scraper_stats": {},
            "cost_savings": 0.0
        }

        logger.info("✅ LangChain Orchestrator initialized with SmartRestaurantScraper")

        # Build the pipeline steps
        self._build_pipeline()

    def _build_pipeline(self):
        """Build pipeline with ContentEvaluationAgent integration"""

        # Step 1: Analyze Query
        # Step 1: Analyze Query
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"],
                "raw_query": x.get("raw_query", x["query"])  # Preserve original
            },
            name="analyze_query"
        )

        # Step 2: Database Search
        self.check_database = RunnableLambda(
            self._check_database_coverage,
            name="check_database"
        )

        # Step 3: Content Evaluation (NEW STEP)
        self.evaluate_content = RunnableLambda(
            self._evaluate_content_routing,
            name="evaluate_content"  
        )

        # Step 4: Search (conditional - now only if content evaluation says so)
        self.search = RunnableLambda(
            self._search_step,
            name="search"
        )

        # Step 5: Scrape (conditional - only if search happened)
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )

        # Step 6: Edit (receives optimized content from evaluation agent)
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )

        # Step 7: Follow-up Search
        self.follow_up_search = RunnableLambda(
            self._follow_up_step,
            name="follow_up_search"
        )

        # Step 8: Format for Telegram
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )

        # Create the complete chain
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.check_database,
                self.evaluate_content,  # NEW STEP
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
        """
        SIMPLIFIED: Pure routing method that delegates ALL database logic to DatabaseSearchAgent.
        The orchestrator only handles routing - no business logic here.
        """
        try:
            logger.info("🗃️ ROUTING TO DATABASE SEARCH AGENT")

            # Ensure raw_query flows to database search
            query_data_with_raw = {
                **x,
                "raw_query": x.get("raw_query", x.get("query", ""))
            }

            # Delegate everything to the DatabaseSearchAgent  
            database_result = self.database_search_agent.search_and_evaluate(query_data_with_raw)

            # Preserve raw_query through the pipeline
            result = {
                **x, 
                **database_result,
                "raw_query": x.get("raw_query", x.get("query", ""))
            }

            # PRESERVE destination from query analysis if database_result doesn't have it
            if not result.get("destination") or result.get("destination") == "Unknown":
                if x.get("destination") and x.get("destination") != "Unknown":
                    result["destination"] = x["destination"]
                    logger.info(f"🔧 Preserved destination from query analysis: {x['destination']}")

            return result

        except Exception as e:
            logger.error(f"❌ Error routing to database search agent: {e}")
            # Simple fallback - no complex logic in orchestrator
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query even on error
                "has_database_content": False,
                "database_restaurants": [],
                "content_source": "web_search",
                "skip_web_search": False,
                "evaluation_details": {
                    "sufficient": False,
                    "reason": f"routing_error: {str(e)}",
                    "details": {"error": str(e)}
                }
            }


    def _evaluate_content_routing(self, x):
        try:
            logger.info("🧠 ROUTING TO CONTENT EVALUATION AGENT")

            # Pure delegation - no business logic in orchestrator
            return self.dbcontent_evaluation_agent.evaluate_and_route(x)

        except Exception as e:
            logger.error(f"❌ Error routing to content evaluation agent: {e}")
            # Even error handling is minimal - just log and re-raise
            # Let the ContentEvaluationAgent handle error logic
            return self.dbcontent_evaluation_agent._handle_evaluation_error(x, e)

    def _search_step(self, x):
        """
        FIXED: Handle search queries properly and ensure destination flows correctly

        The issue was that destination could be lost when coming from content evaluation agent.
        This fix ensures destination is preserved from multiple sources and passed correctly to search agent.
        """
        try:
            logger.info("🔍 SEARCH STEP")

            # Check if we should skip web search
            if x.get("skip_web_search", False):
                logger.info("⏭️ Skipping web search - database provided sufficient results")
                return {**x, "search_results": []}

            # NEW: Check if content evaluation agent already provided search results
            if x.get("search_results") and len(x.get("search_results", [])) > 0:
                logger.info("⏭️ Skipping web search - content evaluation agent already provided results")
                return x  # Return unchanged - results already present

            # FIXED: Get destination from multiple possible sources
            destination = None

            # Try to get destination from various pipeline sources
            if x.get("destination"):
                destination = x.get("destination")
                logger.info(f"📍 Using destination from main pipeline: {destination}")
            elif x.get("query_analysis", {}).get("destination"):
                destination = x.get("query_analysis", {}).get("destination")
                logger.info(f"📍 Using destination from query_analysis: {destination}")
            elif x.get("evaluation_result", {}).get("destination"):
                destination = x.get("evaluation_result", {}).get("destination") 
                logger.info(f"📍 Using destination from evaluation_result: {destination}")
            else:
                destination = "Unknown"
                logger.warning("⚠️ No destination found in pipeline data")

            # FIXED: Get search queries from multiple possible sources
            search_queries = []

            # Try different query field names based on where the search was triggered
            if x.get("search_queries"):
                # From query analyzer (original flow)
                search_queries = x.get("search_queries", [])
                logger.info("📝 Using search_queries from query analyzer")
            elif x.get("english_queries") or x.get("local_queries"):
                # From query analyzer (enhanced format)
                english_queries = x.get("english_queries", [])
                local_queries = x.get("local_queries", [])
                search_queries = english_queries + local_queries
                logger.info(f"📝 Combining queries: {len(english_queries)} English + {len(local_queries)} local")
            else:
                # Fallback: generate from raw query
                raw_query = x.get("raw_query", x.get("query", ""))
                if raw_query and destination != "Unknown":
                    search_queries = [f"best restaurants {raw_query} {destination}"]
                    logger.info("📝 Generated fallback search query from raw query")

            # Validate we have queries and destination
            if not search_queries:
                logger.warning("❌ No search queries available")
                logger.warning(f"Available keys: {list(x.keys())}")
                # Don't return empty - try to generate from raw query one more time
                raw_query = x.get("raw_query", x.get("query", ""))
                if raw_query:
                    search_queries = [f"restaurants {raw_query}"]
                    logger.info(f"🔧 Emergency fallback: generated query from raw_query: {search_queries}")
                else:
                    return {**x, "search_results": []}

            if destination == "Unknown":
                logger.warning("❌ No destination available for search")
                logger.warning(f"Pipeline data keys: {list(x.keys())}")
                # Don't fail completely - try to extract destination from search queries
                for query in search_queries:
                    # Simple heuristic to extract potential city names
                    words = query.lower().split()
                    if 'in' in words:
                        in_index = words.index('in')
                        if in_index < len(words) - 1:
                            potential_destination = words[in_index + 1].title()
                            destination = potential_destination
                            logger.info(f"🔧 Extracted destination from query: {destination}")
                            break

                if destination == "Unknown":
                    logger.error("❌ Cannot proceed without destination")
                    return {**x, "search_results": []}

            # Prepare query metadata for search agent
            query_metadata = {
                'is_english_speaking': x.get('is_english_speaking', True),
                'local_language': x.get('local_language')
            }

            logger.info(f"🌐 Executing search with {len(search_queries)} queries:")
            for i, query in enumerate(search_queries, 1):
                logger.info(f"  {i}. {query}")
            logger.info(f"📍 Destination: {destination}")

            # FIXED: Call search agent with correct parameters
            search_results = self.search_agent.search(search_queries, destination, query_metadata)

            logger.info(f"✅ Search completed: {len(search_results)} results found")

            return {
                **x,
                "search_results": search_results,
                "destination": destination,  # Ensure destination flows forward
                "search_queries": search_queries  # Preserve search queries
            }

        except Exception as e:
            logger.error(f"❌ Error executing search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {**x, "search_results": []}

    # REPLACE the _scrape_step method in orchestrator - REMOVE individual text cleaning:

    def _scrape_step(self, x):
        """
        ENHANCED: Updated scrape step - text cleaning moved to save method
        """
        try:
            content_source = x.get("content_source", "unknown")

            # Check if we should skip scraping (database-only branch)
            if content_source == "database":
                logger.info("⏭️ SKIPPING SCRAPING - using database-only content")
                logger.info("⏭️ → NO FILES SENT TO SUPABASE MANAGER (using existing data)")
                return {**x, "scraped_results": []}

            # For hybrid and web_search, we need to scrape
            logger.info("🤖 SMART SCRAPING PIPELINE STARTING")

            if content_source == "hybrid":
                logger.info("🔄 → HYBRID MODE: Will combine with preserved database results")
            else:
                logger.info("🌐 → WEB-ONLY MODE: Fresh web search results")

            logger.info("🤖 → WILL SEND FILES TO SUPABASE MANAGER AFTER SCRAPING")

            search_results = x.get("search_results", [])

            if not search_results:
                logger.warning("⚠️ No search results to scrape")
                # For hybrid mode, this is OK - we still have database results
                if content_source == "hybrid":
                    logger.info("🔄 Hybrid mode: No web results to scrape, will use database results only")
                return {**x, "scraped_results": []}

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
                scraped_results = pool.submit(run_scraping).result()

            # ENHANCED: Log smart scraper statistics
            scraper_stats = self.scraper.get_stats()
            logger.info(f"🤖 Smart scraping complete:")
            logger.info(f"   💰 Cost estimate: {scraper_stats.get('total_cost_estimate', 0):.1f} credits")
            logger.info(f"   💾 Cost saved: {scraper_stats.get('cost_saved_vs_all_firecrawl', 0):.1f} credits")

            # Log strategy breakdown
            for strategy, count in scraper_stats.get("strategy_breakdown", {}).items():
                if count > 0:
                    emoji = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}
                    logger.info(f"   {emoji.get(strategy, '📌')} {strategy}: {count} URLs")

            logger.info(f"✅ Scraped {len(scraped_results)} articles")


            # Save scraped content for Supabase manager - text cleaning happens here
            if scraped_results:
                logger.info("💾 Proceeding to save scraped content...")
                self._save_scraped_content_for_processing(x, scraped_results)
            else:
                logger.warning("⚠️ No enriched results to save")

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),
                "scraped_results": scraped_results
            }

        except Exception as e:
            logger.error(f"❌ Error in enhanced scraping step: {e}")
            return {**x, "scraped_results": []}

    def _edit_step(self, x):
        """
        ENHANCED: Updated to handle new restaurant field names and hybrid mode

        Now handles:
        - database_restaurants_final: 4+ sufficient restaurants (database-only)
        - database_restaurants_hybrid: 1-3 restaurants for hybrid mode
        - scraped_results: web search results
        - Hybrid mode: combines database_restaurants_hybrid + scraped_results
        """
        try:
            logger.info("✏️ ENHANCED ROUTING TO EDITOR AGENT")

            content_source = x.get("content_source", "unknown")
            raw_query = x.get("raw_query", x.get("query", ""))
            # FIXED: Robust destination extraction from multiple sources
            destination = None

            # Try multiple sources to find destination
            if x.get("destination") and x.get("destination") != "Unknown":
                destination = x.get("destination")
                logger.info(f"📍 Using destination from main pipeline: {destination}")
            elif x.get("query_analysis", {}).get("destination"):
                destination = x.get("query_analysis", {}).get("destination")
                logger.info(f"📍 Using destination from query_analysis: {destination}")
            elif x.get("evaluation_result", {}).get("destination"):
                destination = x.get("evaluation_result", {}).get("destination") 
                logger.info(f"📍 Using destination from evaluation_result: {destination}")
            elif x.get("database_search_result", {}).get("destination"):
                destination = x.get("database_search_result", {}).get("destination")
                logger.info(f"📍 Using destination from database_search_result: {destination}")
            else:
                destination = "Unknown"
                logger.warning("⚠️ No destination found in edit step - this will cause follow-up search issues")
                logger.warning(f"Available pipeline keys: {list(x.keys())}")

            # NEW: Get enhanced restaurant field names
            database_restaurants_final = x.get("database_restaurants_final", [])
            database_restaurants_hybrid = x.get("database_restaurants_hybrid", [])
            scraped_results = x.get("scraped_results", [])

            logger.info(f"📊 Content analysis:")
            logger.info(f"   database_restaurants_final: {len(database_restaurants_final)}")
            logger.info(f"   database_restaurants_hybrid: {len(database_restaurants_hybrid)}")
            logger.info(f"   scraped_results: {len(scraped_results)}")
            logger.info(f"   content_source: {content_source}")

            if content_source == "database":
                # DATABASE-ONLY BRANCH: Use database_restaurants_final
                logger.info("🗃️ Processing DATABASE-ONLY content")

                if not database_restaurants_final:
                    logger.warning("⚠️ No final database results to process")
                    return {
                        **x,
                        "raw_query": raw_query,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Use database_restaurants_final for database-only route
                edit_output = self.editor_agent.edit(
                    scraped_results=None,
                    database_restaurants=database_restaurants_final,  # Use final results
                    raw_query=raw_query, 
                    destination=destination
                )

                logger.info(f"✅ Formatted {len(edit_output.get('edited_results', {}).get('main_list', []))} final database restaurants")

            elif content_source == "hybrid":
                # HYBRID BRANCH: Combine database_restaurants_hybrid + scraped_results
                logger.info("🔄 Processing HYBRID content (database + web)")

                if not database_restaurants_hybrid and not scraped_results:
                    logger.warning("⚠️ No hybrid content to process")
                    return {
                        **x,
                        "raw_query": raw_query,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Pass both database hybrid results AND scraped results
                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=database_restaurants_hybrid,  # Use hybrid results
                    raw_query=raw_query,
                    destination=destination
                )

                logger.info(f"✅ Processed hybrid content: {len(database_restaurants_hybrid)} database + {len(scraped_results)} scraped")

            elif content_source == "web_search":
                # WEB-ONLY BRANCH: Use only scraped_results
                logger.info("🌐 Processing WEB-ONLY content")

                if not scraped_results:
                    logger.warning("⚠️ No scraped results to process")
                    return {
                        **x,
                        "raw_query": raw_query,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Use only scraped results for web-only route
                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=None,  # No database content
                    raw_query=raw_query,
                    destination=destination
                )

                logger.info(f"✅ Processed {len(edit_output.get('edited_results', {}).get('main_list', []))} restaurants from web content")

            else:
                # FALLBACK: Try to process whatever we have
                logger.warning(f"⚠️ Unknown content_source: {content_source} - attempting fallback")

                # Check for legacy database_restaurants field
                legacy_database_restaurants = x.get("database_restaurants", [])

                if database_restaurants_final:
                    database_content = database_restaurants_final
                elif database_restaurants_hybrid:
                    database_content = database_restaurants_hybrid
                elif legacy_database_restaurants:
                    database_content = legacy_database_restaurants
                else:
                    database_content = None

                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results if scraped_results else None,
                    database_restaurants=database_content,
                    raw_query=raw_query,
                    destination=destination
                )

                logger.info("✅ Fallback processing completed")

            return {
                **x,
                "raw_query": raw_query,
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"❌ Error in enhanced edit step: {e}")
            logger.error(f"Available data keys: {list(x.keys())}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),
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
                destination=x.get("destination", "Unknown")
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
        """Format step - converts enhanced_results to langchain_formatted_results"""
        try:
            logger.info("📱 ENTERING FORMAT STEP")

            enhanced_results = x.get("enhanced_results", {})
            main_list = enhanced_results.get("main_list", [])

            if not main_list:
                logger.warning("⚠️ No restaurants to format for Telegram")
                return {
                    **x,
                    "langchain_formatted_results": "Sorry, no restaurant recommendations found for your query."
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
                "langchain_formatted_results": telegram_text,
                "final_results": enhanced_results
            }

        except Exception as e:
            logger.error(f"❌ Error in format step: {e}")
            dump_chain_state("format_error", x, error=e)
            return {
                **x,
                "langchain_formatted_results": "Sorry, there was an error formatting the restaurant recommendations."
            }

    def _save_scraped_content_for_processing(self, pipeline_data, scraped_results):
        """
        UPDATED: Upload existing TXT file from text cleaner to Supabase
        Text cleaner creates TXT file, we upload that file directly
        """
        try:
            logger.info("💾 Processing RTF through text cleaner and uploading TXT file to Supabase...")

            # Extract metadata (unchanged)
            destination_info = pipeline_data.get("destination", "Unknown")
            city = destination_info.split(",")[0].strip() if isinstance(destination_info, str) else "Unknown"
            country = destination_info.split(",")[1].strip() if isinstance(destination_info, str) and "," in destination_info else ""
            query = pipeline_data.get("raw_query", pipeline_data.get("query", ""))

            # NEW: Use text cleaner to process RTF and create TXT file
            if hasattr(self, '_text_cleaner') and self._text_cleaner:
                logger.info("🧹 Using text cleaner to convert RTF to TXT file...")

                # FIXED: Run async text cleaner in event loop
                def run_text_cleaner():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._text_cleaner.clean_scraped_results(scraped_results, query)
                        )
                    finally:
                        loop.close()

                # The cleaner processes all results and returns path to TXT file
                txt_file_path = run_text_cleaner()

                if not txt_file_path or not os.path.exists(txt_file_path):
                    logger.error("❌ Text cleaner failed to create TXT file")
                    return

                # Read the TXT file content
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    clean_content = f.read()

                logger.info(f"📊 TXT file created: {os.path.basename(txt_file_path)} ({len(clean_content)} chars)")

                # Prepare metadata for Supabase upload
                metadata = {
                    'city': city,
                    'country': country,
                    'query': query,
                    'scraped_at': datetime.now().isoformat(),
                    'source_count': len(scraped_results),
                    'content_length': len(clean_content),
                    'content_type': 'cleaned_restaurants',
                    'file_format': 'txt',  # TXT format
                    'processing_method': 'rtf_to_text',
                    'local_file': os.path.basename(txt_file_path)
                }

                # Background upload TXT file to Supabase
                def perform_background_uploads():
                    try:
                        if hasattr(self, 'storage_manager') and self.storage_manager:
                            logger.info("☁️ Uploading clean restaurant TXT to Supabase Storage...")

                            # Upload clean TXT content to Supabase
                            success, storage_path = self.storage_manager.upload_scraped_content(
                                clean_content,  # Clean text from file
                                metadata, 
                                file_type="txt"  # TXT files for Supabase
                            )

                            if success:
                                logger.info(f"✅ Clean restaurant TXT uploaded to: {storage_path}")
                            else:
                                logger.warning("⚠️ Failed to upload clean TXT to Supabase Storage")

                        else:
                            logger.warning("⚠️ No storage manager available - TXT file saved locally only")

                    except Exception as upload_error:
                        logger.error(f"❌ Error in TXT upload: {upload_error}")

                # Execute background upload
                from threading import Thread
                upload_thread = Thread(target=perform_background_uploads, daemon=True)
                upload_thread.start()

                logger.info("🚀 TXT background upload initiated")

            else:
                # Fallback: Save raw content as TXT
                logger.warning("⚠️ No text cleaner available - saving raw content as TXT fallback")

                # Compile raw content for fallback
                all_content = f"Query: {query}\nDestination: {city}, {country}\n\n"
                for idx, result in enumerate(scraped_results, 1):
                    content = result.get('content', '') or result.get('scraped_content', '')
                    if content:
                        all_content += f"SOURCE {idx}: {result.get('url', 'Unknown URL')}\n"
                        all_content += content + "\n\n"

                # Save as TXT fallback
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scraped_fallback_{city.replace(' ', '_')}_{timestamp}.txt"

                os.makedirs("scraped_content", exist_ok=True)
                file_path = os.path.join("scraped_content", filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(all_content)

                logger.info(f"💾 Fallback: Saved raw content to {file_path}")

        except Exception as e:
            logger.error(f"❌ Error in RTF-to-TXT processing: {e}")
            raise

    def _log_enhanced_usage(self):
        """Enhanced usage logging with smart scraper insights"""
        scraper_stats = self.scraper.get_stats()

        logger.info("=" * 60)
        logger.info("SMART SCRAPER USAGE REPORT")
        logger.info("=" * 60)
        logger.info(f"URLs processed: {scraper_stats.get('total_processed', 0)}")
        logger.info(f"Successful extractions: {scraper_stats.get('total_processed', 0) - scraper_stats.get('strategy_breakdown', {}).get('failed', 0)}")
        logger.info(f"Cost estimate: {scraper_stats.get('total_cost_estimate', 0):.1f} credits")
        logger.info(f"Cost saved vs all-Firecrawl: {scraper_stats.get('cost_saved_vs_all_firecrawl', 0):.1f} credits")

        # Strategy breakdown
        if scraper_stats.get("total_processed", 0) > 0:
            logger.info("STRATEGY BREAKDOWN:")
            for strategy, count in scraper_stats.get("strategy_breakdown", {}).items():
                if count > 0:
                    emoji = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}
                    logger.info(f"  {emoji.get(strategy, '📌')} {strategy}: {count} URLs")

        logger.info("=" * 60)

    def _update_enhanced_stats(self, result: dict, processing_time: float):
        """Update statistics with enhanced smart scraper data"""
        self.stats["total_queries"] += 1

        # Update processing time average
        current_avg = self.stats["avg_processing_time"]
        total_queries = self.stats["total_queries"]
        self.stats["avg_processing_time"] = (current_avg * (total_queries - 1) + processing_time) / total_queries

        # Track content source
        content_source = result.get("content_source", "unknown")
        if content_source in self.stats["content_sources"]:
            self.stats["content_sources"][content_source] += 1

        # Track success
        main_list = result.get("enhanced_results", {}).get("main_list", [])
        if main_list:
            self.stats["successful_queries"] += 1

        # Track smart scraper statistics
        scraper_stats = self.scraper.get_stats()
        self.stats["scraper_stats"] = scraper_stats
        self.stats["cost_savings"] += scraper_stats.get('cost_saved_vs_all_firecrawl', 0)

    @log_function_call
    def process_query(self, user_query: str) -> dict:
        """
        Process a restaurant query through the complete pipeline.

        Args:
            user_query: The user's restaurant request

        Returns:
            Dict with langchain_formatted_results and other results
        """
        start_time = time.time()  # FIXED: Add start_time for error handling

        # Generate trace ID for debugging
        trace_id = f"query_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                logger.info(f"🚀 STARTING RECOMMENDATION PIPELINE")
                logger.info(f"Query: {user_query}")

                # Prepare input data (UPDATED to include raw query from the start)
                input_data = {
                    "query": user_query,           # For processing/analysis
                    "raw_query": user_query       # PRESERVE original
                }

                # Execute the chain
                result = self.chain.invoke(input_data)

                # Log completion
                content_source = result.get("content_source", "unknown")
                logger.info("✅ PIPELINE COMPLETE")
                logger.info(f"📊 Content source: {content_source}")

                # CLEANED: Enhanced usage summary (no Firecrawl remnants)
                if content_source in ["web_search", "hybrid"]:
                    self._log_enhanced_usage()

                # Save process record (simplified - just log it)
                logger.info(f"📊 Process completed: {user_query} → {result.get('destination', 'Unknown')} → {content_source}")

                # Extract results with correct key names
                telegram_text = result.get("langchain_formatted_results", 
                                         "Sorry, no recommendations found.")

                enhanced_results = result.get("enhanced_results", {})
                main_list = enhanced_results.get("main_list", [])

                logger.info(f"📊 Final result: {len(main_list)} restaurants for {result.get('destination', 'Unknown')}")
                logger.info(f"📊 Source: {content_source}")

                processing_time = time.time() - start_time
                self._update_enhanced_stats(result, processing_time)

                # CLEANED: Return with smart scraper statistics
                return {
                    "langchain_formatted_results": telegram_text,
                    "enhanced_results": enhanced_results,
                    "main_list": main_list,
                    "destination": result.get("destination"),
                    "content_source": content_source,
                    "raw_query": result.get("raw_query", user_query),
                    # ENHANCED: Add smart scraper statistics
                    "smart_scraper_stats": self.scraper.get_stats(),
                    "cost_savings": self.scraper.get_stats().get('cost_saved_vs_all_firecrawl', 0),
                    "scraper_stats": self.scraper.get_stats()  # CLEANED: Updated field name
                }

            except Exception as e:
                logger.error(f"❌ Error in chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query}, error=e)

                # CLEANED: Enhanced usage logging (no Firecrawl calls)
                try:
                    self._log_enhanced_usage()
                except:
                    pass

                processing_time = time.time() - start_time
                # Create minimal result for stats tracking
                error_result = {
                    "content_source": "error",
                    "enhanced_results": {"main_list": []}
                }
                self._update_enhanced_stats(error_result, processing_time)

                return {
                    "main_list": [],
                    "langchain_formatted_results": "Sorry, there was an error processing your request.",
                    "raw_query": user_query,
                    "scraper_stats": self.scraper.get_stats()  # CLEANED: Updated field name
                }

    def get_enhanced_stats(self) -> dict:
        """Get comprehensive orchestrator and smart scraper statistics"""
        scraper_stats = self.scraper.get_stats()

        return {
            "orchestrator": {
                "total_queries": getattr(self, 'stats', {}).get("total_queries", 0),
                "successful_queries": getattr(self, 'stats', {}).get("successful_queries", 0),
                "content_sources": getattr(self, 'stats', {}).get("content_sources", {})
            },
            "smart_scraper": scraper_stats,
            "cost_analysis": {
                "total_cost_estimate": scraper_stats.get('total_cost_estimate', 0),
                "cost_saved_vs_all_firecrawl": scraper_stats.get('cost_saved_vs_all_firecrawl', 0),
                "efficiency_improvement": f"{scraper_stats.get('cost_saved_vs_all_firecrawl', 0) / max(scraper_stats.get('total_cost_estimate', 1), 1):.1%}"
            },
            "strategy_breakdown": scraper_stats.get("strategy_breakdown", {})
        }

    # CLEANED: Legacy compatibility updated to avoid confusion
    def get_scraper_stats(self) -> dict:
        """Get smart scraper stats (renamed from get_firecrawl_stats for clarity)"""
        return self.scraper.get_stats()