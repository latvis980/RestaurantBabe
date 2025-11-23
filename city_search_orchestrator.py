# city_search_orchestrator.py
"""
LangChain LCEL City Search Orchestrator

A clean, focused pipeline for city-wide restaurant searches using LangChain LCEL.
Follows the same pattern as location_orchestrator.py for consistency.

PIPELINE FLOW:
1. Query Analysis → Extract destination, cuisine, generate search queries
2. Database Search → Check existing restaurant database
3. Content Evaluation → Decide: database-only, hybrid, or web-search
4. Web Search → Execute Brave/Tavily search (conditional)
5. Scraping → Scrape search results (conditional)
6. Cleaning → Process scraped content through TextCleanerAgent (conditional)
7. Editing → AI-powered restaurant extraction and formatting
8. Follow-up → Address verification and rating filtering
9. Formatting → Generate Telegram-ready output

CONTENT MODES:
- database: Sufficient high-quality matches in database → skip web search
- hybrid: Some database matches + web search for more options
- web_search: No/poor database matches → full web search pipeline

CORRECTED METHOD NAMES (verified from project files):
✅ QueryAnalyzer.analyze(query) → returns dict with destination, search_queries, etc.
✅ DatabaseSearchAgent.search_and_evaluate(query_data) → returns database results
✅ ContentEvaluationAgent.evaluate_and_route(pipeline_data) → returns routing decision
✅ BraveSearchAgent.search(search_queries, destination, query_metadata) → returns search results
✅ BrowserlessRestaurantScraper.scrape_search_results(search_results) → async, returns scraped content
✅ TextCleanerAgent.process_scraped_results_individually(scraped_results, query) → async, returns file path
✅ EditorAgent.edit(scraped_results, database_restaurants, raw_query, destination, ...) → returns edited results
✅ FollowUpSearchAgent.perform_follow_up_searches(edited_results, follow_up_queries, destination) → returns enhanced results
✅ TelegramFormatter.format_recommendations(recommendations_data) → returns formatted text
"""

import logging
import asyncio
import time
import os
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableBranch
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import traceable
from langchain_core.tracers.langchain import wait_for_all_tracers

# Import agents (preserve their implementations)
from agents.query_analyzer import QueryAnalyzer
from agents.database_search_agent import DatabaseSearchAgent
from agents.dbcontent_evaluation_agent import ContentEvaluationAgent
from agents.search_agent import BraveSearchAgent
from agents.browserless_scraper import BrowserlessRestaurantScraper
from agents.text_cleaner_agent import TextCleanerAgent
from agents.editor_agent import EditorAgent
from agents.follow_up_search_agent import FollowUpSearchAgent

# Formatter
from formatters.telegram_formatter import TelegramFormatter

logger = logging.getLogger(__name__)


class CitySearchOrchestrator:
    """
    LangChain LCEL-based orchestrator for city-wide restaurant searches.
    
    This is a focused, deterministic pipeline that:
    - Uses LCEL RunnableSequence for clean step-by-step execution
    - Delegates ALL business logic to individual agents
    - Handles three content modes: database, hybrid, web_search
    - Provides full LangSmith tracing for debugging
    """

    def __init__(self, config):
        self.config = config

        # Initialize all agents
        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)
        self.content_evaluation_agent = ContentEvaluationAgent(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = BrowserlessRestaurantScraper(config)
        self.text_cleaner = TextCleanerAgent(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)
        self.telegram_formatter = TelegramFormatter(config)

        # Set up agent dependencies
        self.content_evaluation_agent.set_brave_search_agent(self.search_agent)

        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "content_sources": {"database": 0, "hybrid": 0, "web_search": 0},
            "avg_processing_time": 0.0
        }

        # Build the LCEL pipeline
        self._build_pipeline()

        logger.info("✅ CitySearchOrchestrator initialized with LangChain LCEL pipeline")

    def _build_pipeline(self):
        """Build the LangChain LCEL pipeline with traced steps"""

        # Step 1: Query Analysis
        self.analyze_query_chain = RunnableLambda(
            self._analyze_query_step_traced,
            name="analyze_query"
        )

        # Step 2: Database Search
        self.database_search_chain = RunnableLambda(
            self._database_search_step_traced,
            name="database_search"
        )

        # Step 3: Content Evaluation (decides routing)
        self.content_evaluation_chain = RunnableLambda(
            self._content_evaluation_step_traced,
            name="content_evaluation"
        )

        # Step 4: Web Search (conditional)
        self.web_search_chain = RunnableLambda(
            self._web_search_step_traced,
            name="web_search"
        )

        # Step 5: Scraping (conditional)
        self.scraping_chain = RunnableLambda(
            self._scraping_step_traced,
            name="scraping"
        )

        # Step 6: Cleaning (conditional)
        self.cleaning_chain = RunnableLambda(
            self._cleaning_step_traced,
            name="text_cleaning"
        )

        # Step 7: Editing
        self.editing_chain = RunnableLambda(
            self._editing_step_traced,
            name="editing"
        )

        # Step 8: Follow-up Search
        self.followup_chain = RunnableLambda(
            self._followup_step_traced,
            name="follow_up_search"
        )

        # Step 9: Formatting
        self.formatting_chain = RunnableLambda(
            self._formatting_step_traced,
            name="telegram_formatting"
        )

        # Build the complete pipeline using | operator
        self.pipeline = (
            self.analyze_query_chain |
            self.database_search_chain |
            self.content_evaluation_chain |
            self.web_search_chain |
            self.scraping_chain |
            self.cleaning_chain |
            self.editing_chain |
            self.followup_chain |
            self.formatting_chain
        )

        logger.info("✅ LCEL pipeline built with 9 traced steps")

    # ============================================================================
    # TRACED PIPELINE STEPS
    # ============================================================================

    @traceable(run_type="chain", name="query_analysis", metadata={"step": 1})
    async def _analyze_query_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Analyze the user query"""
        return await self._analyze_query_step(pipeline_input)

    @traceable(run_type="retriever", name="database_search", metadata={"step": 2})
    async def _database_search_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Search the database"""
        return await self._database_search_step(pipeline_input)

    @traceable(run_type="llm", name="content_evaluation", metadata={"step": 3})
    async def _content_evaluation_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Evaluate content and decide routing"""
        return await self._content_evaluation_step(pipeline_input)

    @traceable(run_type="tool", name="web_search", metadata={"step": 4})
    async def _web_search_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Execute web search (conditional)"""
        return await self._web_search_step(pipeline_input)

    @traceable(run_type="tool", name="scraping", metadata={"step": 5})
    async def _scraping_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Scrape search results (conditional)"""
        return await self._scraping_step(pipeline_input)

    @traceable(run_type="llm", name="text_cleaning", metadata={"step": 6})
    async def _cleaning_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Clean scraped content (conditional)"""
        return await self._cleaning_step(pipeline_input)

    @traceable(run_type="llm", name="editing", metadata={"step": 7})
    async def _editing_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Edit and extract restaurants"""
        return await self._editing_step(pipeline_input)

    @traceable(run_type="tool", name="follow_up_search", metadata={"step": 8})
    async def _followup_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Follow-up address verification"""
        return await self._followup_step(pipeline_input)

    @traceable(run_type="parser", name="telegram_formatting", metadata={"step": 9})
    async def _formatting_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 9: Format for Telegram output"""
        return await self._formatting_step(pipeline_input)

    # ============================================================================
    # PIPELINE STEP IMPLEMENTATIONS
    # ============================================================================

    async def _analyze_query_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Query Analysis
        
        Delegates to QueryAnalyzer.analyze() which extracts:
        - destination (city/location)
        - search_queries (English and local language)
        - language metadata
        """
        try:
            query = pipeline_input.get("query", "")
            cancel_check_fn = pipeline_input.get("cancel_check_fn")

            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            logger.info(f"🔍 Step 1: Analyzing query: '{query[:50]}...'")

            # Delegate to QueryAnalyzer (synchronous call)
            analysis_result = self.query_analyzer.analyze(query)

            logger.info(f"✅ Query analyzed: destination='{analysis_result.get('destination')}', "
                       f"{len(analysis_result.get('search_queries', []))} search queries")

            return {
                **pipeline_input,
                **analysis_result,
                "raw_query": query,  # Preserve original query
                "query": query,
                "step_completed": "analyze_query"
            }

        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            raise ValueError(f"Query analysis failed: {str(e)}")

    async def _database_search_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Database Search
        
        Delegates to DatabaseSearchAgent.search_and_evaluate() which:
        - Searches for restaurants in the destination
        - Returns matched restaurants with descriptions
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            logger.info("🗃️ Step 2: Searching database")

            # Delegate to DatabaseSearchAgent (synchronous call)
            database_result = self.database_search_agent.search_and_evaluate(pipeline_input)

            db_restaurants = database_result.get("database_restaurants", [])
            logger.info(f"✅ Database search complete: {len(db_restaurants)} restaurants found")

            # Preserve destination from analysis if database didn't return it
            if not database_result.get("destination") or database_result.get("destination") == "Unknown":
                if pipeline_input.get("destination") and pipeline_input.get("destination") != "Unknown":
                    database_result["destination"] = pipeline_input["destination"]

            return {
                **pipeline_input,
                **database_result,
                "step_completed": "database_search"
            }

        except Exception as e:
            logger.error(f"❌ Database search failed: {e}")
            # Return with empty database results, continue to web search
            return {
                **pipeline_input,
                "database_restaurants": [],
                "has_database_content": False,
                "content_source": "web_search",
                "skip_web_search": False,
                "step_completed": "database_search"
            }

    async def _content_evaluation_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Content Evaluation
        
        Delegates to ContentEvaluationAgent.evaluate_and_route() which decides:
        - database: Sufficient content, skip web search
        - hybrid: Some content, supplement with web search
        - web_search: Insufficient content, full web search
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            logger.info("🧠 Step 3: Evaluating content and deciding route")

            # Delegate to ContentEvaluationAgent (synchronous call)
            evaluation_result = self.content_evaluation_agent.evaluate_and_route(pipeline_input)

            content_source = evaluation_result.get("content_source", "web_search")
            skip_web_search = evaluation_result.get("skip_web_search", False)

            logger.info(f"✅ Content evaluation complete: source='{content_source}', skip_web={skip_web_search}")

            return {
                **pipeline_input,
                **evaluation_result,
                "step_completed": "content_evaluation"
            }

        except Exception as e:
            logger.error(f"❌ Content evaluation failed: {e}")
            # Default to web search on evaluation failure
            return {
                **pipeline_input,
                "content_source": "web_search",
                "skip_web_search": False,
                "database_restaurants_final": [],
                "database_restaurants_hybrid": [],
                "step_completed": "content_evaluation"
            }

    async def _web_search_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Web Search (Conditional)
        
        Only executes if:
        - skip_web_search is False
        - search_results not already provided by content evaluation
        
        Delegates to BraveSearchAgent.search()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            # Check if we should skip
            if pipeline_input.get("skip_web_search", False):
                logger.info("⏭️ Step 4: Skipping web search - database sufficient")
                return {**pipeline_input, "search_results": [], "step_completed": "web_search"}

            # Check if results already provided
            if pipeline_input.get("search_results") and len(pipeline_input.get("search_results", [])) > 0:
                logger.info("⏭️ Step 4: Skipping web search - results already provided")
                return {**pipeline_input, "step_completed": "web_search"}

            logger.info("🌐 Step 4: Executing web search")

            # Get search parameters
            search_queries = pipeline_input.get("search_queries", [])
            destination = pipeline_input.get("destination", "Unknown")
            query_metadata = {
                "is_english_speaking": pipeline_input.get("is_english_speaking", True),
                "local_language": pipeline_input.get("local_language")
            }

            # Fallback query generation if needed
            if not search_queries:
                raw_query = pipeline_input.get("raw_query", pipeline_input.get("query", ""))
                if raw_query and destination != "Unknown":
                    search_queries = [f"best restaurants {raw_query} {destination}"]
                    logger.info(f"📝 Generated fallback search query: {search_queries}")

            if not search_queries:
                logger.warning("❌ No search queries available")
                return {**pipeline_input, "search_results": [], "step_completed": "web_search"}

            logger.info(f"🔍 Searching with {len(search_queries)} queries for '{destination}'")

            # Delegate to BraveSearchAgent (synchronous call - handles its own async)
            search_results = self.search_agent.search(search_queries, destination, query_metadata)

            logger.info(f"✅ Web search complete: {len(search_results)} results")

            return {
                **pipeline_input,
                "search_results": search_results,
                "destination": destination,
                "step_completed": "web_search"
            }

        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return {**pipeline_input, "search_results": [], "step_completed": "web_search"}

    async def _scraping_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 5: Scraping (Conditional)
        
        Only executes if content_source is NOT 'database'
        Delegates to BrowserlessRestaurantScraper.scrape_search_results()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            content_source = pipeline_input.get("content_source", "unknown")

            # Skip for database-only
            if content_source == "database":
                logger.info("⏭️ Step 5: Skipping scraping - database-only content")
                return {**pipeline_input, "scraped_results": [], "step_completed": "scraping"}

            search_results = pipeline_input.get("search_results", [])
            if not search_results:
                logger.info("⏭️ Step 5: Skipping scraping - no search results")
                return {**pipeline_input, "scraped_results": [], "step_completed": "scraping"}

            logger.info(f"🤖 Step 5: Scraping {len(search_results)} URLs")

            # Execute async scraping in thread pool
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

            logger.info(f"✅ Scraping complete: {len(scraped_results)} articles scraped")

            # Log scraper stats
            scraper_stats = self.scraper.get_stats()
            logger.info(f"   💰 Cost estimate: {scraper_stats.get('total_cost_estimate', 0):.1f} credits")

            return {
                **pipeline_input,
                "scraped_results": scraped_results,
                "scraper_stats": scraper_stats,
                "step_completed": "scraping"
            }

        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            return {**pipeline_input, "scraped_results": [], "step_completed": "scraping"}

    async def _cleaning_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 6: Text Cleaning (Conditional)
        
        Only executes if there are scraped results
        Delegates to TextCleanerAgent.process_scraped_results_individually()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            content_source = pipeline_input.get("content_source", "unknown")

            # Skip for database-only
            if content_source == "database":
                logger.info("⏭️ Step 6: Skipping cleaning - database-only content")
                return {**pipeline_input, "step_completed": "text_cleaning"}

            scraped_results = pipeline_input.get("scraped_results", [])
            if not scraped_results:
                logger.info("⏭️ Step 6: Skipping cleaning - no scraped results")
                return {**pipeline_input, "step_completed": "text_cleaning"}

            logger.info(f"🧹 Step 6: Cleaning {len(scraped_results)} scraped results")

            query = pipeline_input.get("raw_query", pipeline_input.get("query", ""))

            # Execute async text cleaning in thread pool
            def run_text_cleaner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.text_cleaner.process_scraped_results_individually(scraped_results, query)
                    )
                finally:
                    loop.close()

            with concurrent.futures.ThreadPoolExecutor() as pool:
                cleaned_file_path = pool.submit(run_text_cleaner).result()

            if not cleaned_file_path or not os.path.exists(cleaned_file_path):
                logger.warning("⚠️ TextCleanerAgent didn't create file - using raw content")
                return {**pipeline_input, "step_completed": "text_cleaning"}

            # Read the combined cleaned content
            with open(cleaned_file_path, 'r', encoding='utf-8') as f:
                combined_cleaned_content = f.read()

            logger.info(f"✅ Cleaning complete: {len(combined_cleaned_content)} chars cleaned")

            # Add cleaned_content to each scraped result for EditorAgent
            updated_scraped_results = []
            for result in scraped_results:
                updated_result = result.copy()
                updated_result['cleaned_content'] = combined_cleaned_content
                updated_scraped_results.append(updated_result)

            return {
                **pipeline_input,
                "scraped_results": updated_scraped_results,
                "cleaned_file_path": cleaned_file_path,
                "step_completed": "text_cleaning"
            }

        except Exception as e:
            logger.error(f"❌ Text cleaning failed: {e}")
            return {**pipeline_input, "step_completed": "text_cleaning"}

    async def _editing_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 7: Editing
        
        Routes to appropriate EditorAgent method based on content_source:
        - database: Process database_restaurants_final
        - hybrid: Combine database_restaurants_hybrid + scraped_results
        - web_search: Process scraped_results only
        
        Delegates to EditorAgent.edit()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            content_source = pipeline_input.get("content_source", "unknown")
            raw_query = pipeline_input.get("raw_query", pipeline_input.get("query", ""))
            destination = pipeline_input.get("destination", "Unknown")
            cleaned_file_path = pipeline_input.get("cleaned_file_path")

            # Get restaurant data based on content source
            database_restaurants_final = pipeline_input.get("database_restaurants_final", [])
            database_restaurants_hybrid = pipeline_input.get("database_restaurants_hybrid", [])
            scraped_results = pipeline_input.get("scraped_results", [])

            logger.info(f"✏️ Step 7: Editing - mode='{content_source}'")
            logger.info(f"   database_final: {len(database_restaurants_final)}, "
                       f"database_hybrid: {len(database_restaurants_hybrid)}, "
                       f"scraped: {len(scraped_results)}")

            # Route to EditorAgent with appropriate parameters
            if content_source == "database":
                if not database_restaurants_final:
                    logger.warning("⚠️ No database restaurants to edit")
                    return {
                        **pipeline_input,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": [],
                        "step_completed": "editing"
                    }

                edit_output = self.editor_agent.edit(
                    scraped_results=None,
                    database_restaurants=database_restaurants_final,
                    raw_query=raw_query,
                    destination=destination,
                    processing_mode="database_only",
                    content_source=content_source
                )

            elif content_source == "hybrid":
                if not database_restaurants_hybrid and not scraped_results:
                    logger.warning("⚠️ No content to edit in hybrid mode")
                    return {
                        **pipeline_input,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": [],
                        "step_completed": "editing"
                    }

                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=database_restaurants_hybrid,
                    raw_query=raw_query,
                    destination=destination,
                    processing_mode="hybrid",
                    content_source=content_source,
                    cleaned_file_path=cleaned_file_path
                )

            else:  # web_search or unknown
                if not scraped_results:
                    logger.warning("⚠️ No scraped results to edit")
                    return {
                        **pipeline_input,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": [],
                        "step_completed": "editing"
                    }

                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=None,
                    raw_query=raw_query,
                    destination=destination,
                    processing_mode="web_only",
                    content_source=content_source,
                    cleaned_file_path=cleaned_file_path
                )

            edited_count = len(edit_output.get("edited_results", {}).get("main_list", []))
            logger.info(f"✅ Editing complete: {edited_count} restaurants extracted")

            return {
                **pipeline_input,
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", []),
                "step_completed": "editing"
            }

        except Exception as e:
            logger.error(f"❌ Editing failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                **pipeline_input,
                "edited_results": {"main_list": []},
                "follow_up_queries": [],
                "step_completed": "editing"
            }

    async def _followup_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 8: Follow-up Search
        
        Verifies addresses and filters restaurants based on:
        - Google Maps verification
        - Rating thresholds
        - Business status (open/closed)
        
        Delegates to FollowUpSearchAgent.perform_follow_up_searches()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            edited_results = pipeline_input.get("edited_results", {})
            follow_up_queries = pipeline_input.get("follow_up_queries", [])
            destination = pipeline_input.get("destination", "Unknown")

            main_list = edited_results.get("main_list", [])
            if not main_list:
                logger.warning("⚠️ Step 8: No restaurants for follow-up verification")
                return {
                    **pipeline_input,
                    "enhanced_results": {"main_list": []},
                    "step_completed": "follow_up_search"
                }

            logger.info(f"🔍 Step 8: Follow-up verification for {len(main_list)} restaurants")

            # Delegate to FollowUpSearchAgent (synchronous call)
            followup_output = self.follow_up_search_agent.perform_follow_up_searches(
                edited_results=edited_results,
                follow_up_queries=follow_up_queries,
                destination=destination
            )

            enhanced_results = followup_output.get("enhanced_results", {"main_list": []})
            final_count = len(enhanced_results.get("main_list", []))

            logger.info(f"✅ Follow-up complete: {final_count} restaurants passed verification")

            return {
                **pipeline_input,
                "enhanced_results": enhanced_results,
                "step_completed": "follow_up_search"
            }

        except Exception as e:
            logger.error(f"❌ Follow-up search failed: {e}")
            # Return edited results as enhanced results on failure
            return {
                **pipeline_input,
                "enhanced_results": pipeline_input.get("edited_results", {"main_list": []}),
                "step_completed": "follow_up_search"
            }

    async def _formatting_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 9: Telegram Formatting
        
        Converts enhanced_results to Telegram-ready formatted text
        Delegates to TelegramFormatter.format_recommendations()
        """
        try:
            enhanced_results = pipeline_input.get("enhanced_results", {})
            main_list = enhanced_results.get("main_list", [])

            if not main_list:
                logger.warning("⚠️ Step 9: No restaurants to format")
                return {
                    **pipeline_input,
                    "langchain_formatted_results": "Sorry, no restaurant recommendations found for your query.",
                    "success": False,
                    "step_completed": "telegram_formatting"
                }

            logger.info(f"📱 Step 9: Formatting {len(main_list)} restaurants for Telegram")

            # Delegate to TelegramFormatter (synchronous call)
            telegram_text = self.telegram_formatter.format_recommendations(enhanced_results)

            if not telegram_text or len(telegram_text.strip()) == 0:
                logger.error("❌ TelegramFormatter returned empty result")
                telegram_text = "Sorry, I found some restaurants but had trouble formatting them. Please try again."

            logger.info(f"✅ Formatting complete: {len(telegram_text)} chars")

            return {
                **pipeline_input,
                "langchain_formatted_results": telegram_text,
                "success": True,
                "step_completed": "telegram_formatting"
            }

        except Exception as e:
            logger.error(f"❌ Formatting failed: {e}")
            return {
                **pipeline_input,
                "langchain_formatted_results": "Sorry, there was an error formatting the results.",
                "success": False,
                "step_completed": "telegram_formatting"
            }

    # ============================================================================
    # PUBLIC API
    # ============================================================================

    @traceable(run_type="chain", name="city_search_pipeline", metadata={"pipeline_type": "langchain_lcel"})
    async def process_query_async(
        self,
        query: str,
        cancel_check_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a city search query through the complete LCEL pipeline (async version)
        
        Args:
            query: User's restaurant search query
            cancel_check_fn: Optional function to check if search should be cancelled
            
        Returns:
            Dict with langchain_formatted_results and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"🚀 Starting city search pipeline: '{query[:50]}...'")

            # Prepare pipeline input
            pipeline_input = {
                "query": query,
                "raw_query": query,
                "cancel_check_fn": cancel_check_fn,
                "start_time": start_time
            }

            # Execute the pipeline with tracing
            result = await self.pipeline.ainvoke(
                pipeline_input,
                config={
                    "run_name": f"city_search_{{query='{query[:30]}...'}}",
                    "metadata": {
                        "user_query": query,
                        "pipeline_version": "lcel_v1.0"
                    },
                    "tags": ["city_search", "lcel_pipeline"]
                }
            )

            # Add timing metadata
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            result["pipeline_type"] = "langchain_lcel"

            # Flush traces
            try:
                wait_for_all_tracers()
            except Exception as flush_error:
                logger.warning(f"⚠️ Failed to flush traces: {flush_error}")

            # Update statistics
            self._update_stats(result, processing_time)

            logger.info(f"✅ City search pipeline complete in {processing_time}s")

            return result

        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            processing_time = round(time.time() - start_time, 2)

            return {
                "success": False,
                "error": str(e),
                "langchain_formatted_results": "Sorry, there was an error processing your request.",
                "enhanced_results": {"main_list": []},
                "processing_time": processing_time,
                "pipeline_type": "langchain_lcel"
            }

    def process_query(self, query: str, cancel_check_fn: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process a city search query (synchronous wrapper)
        
        This is the main entry point for the Telegram bot and LangGraph supervisor.
        """
        start_time = time.time()

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # Run async pipeline in new event loop
                def run_async():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self.process_query_async(query, cancel_check_fn)
                        )
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(run_async).result()

                return result

            except Exception as e:
                logger.error(f"❌ Synchronous pipeline error: {e}")

                return {
                    "success": False,
                    "error": str(e),
                    "langchain_formatted_results": "Sorry, there was an error processing your request.",
                    "enhanced_results": {"main_list": []},
                    "processing_time": round(time.time() - start_time, 2),
                    "pipeline_type": "langchain_lcel"
                }

    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """Update orchestrator statistics"""
        self.stats["total_queries"] += 1

        # Update average processing time
        current_avg = self.stats["avg_processing_time"]
        total = self.stats["total_queries"]
        self.stats["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total

        # Track content source
        content_source = result.get("content_source", "unknown")
        if content_source in self.stats["content_sources"]:
            self.stats["content_sources"][content_source] += 1

        # Track success
        if result.get("success", False):
            self.stats["successful_queries"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "orchestrator": self.stats,
            "scraper": self.scraper.get_stats() if hasattr(self.scraper, 'get_stats') else {},
            "pipeline_type": "langchain_lcel"
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "pipeline_type": "langchain_lcel",
            "steps": [
                "analyze_query",
                "database_search",
                "content_evaluation",
                "web_search",
                "scraping",
                "text_cleaning",
                "editing",
                "follow_up_search",
                "telegram_formatting"
            ],
            "content_modes": ["database", "hybrid", "web_search"],
            "agents": [
                "QueryAnalyzer",
                "DatabaseSearchAgent",
                "ContentEvaluationAgent",
                "BraveSearchAgent",
                "BrowserlessRestaurantScraper",
                "TextCleanerAgent",
                "EditorAgent",
                "FollowUpSearchAgent",
                "TelegramFormatter"
            ],
            "tracing_enabled": True,
            "version": "1.0"
        }
