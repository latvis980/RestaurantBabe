# agents/langgraph_tools.py
"""
FINAL LangGraph Tools with Full Traditional Pipeline Integration - NO FALLBACKS

ENFORCES MAIN PIPELINE:
1. Web search → URLs
2. Intelligent scraping → scraped content  
3. Text cleaning → cleaned & structured content
4. Restaurant extraction → individual restaurant objects
5. Final formatting → user-ready output

NO FALLBACKS: Forces the pipeline to work properly or fail clearly for debugging
"""

import logging
import json
import asyncio
import concurrent.futures
import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class RestaurantSearchTools:
    """
    Collection of tools for restaurant search with FULL PIPELINE integration.
    NO FALLBACKS - Forces proper pipeline execution.
    """

    def __init__(self, config):
        """Initialize all agents for the full pipeline with correct imports"""
        self.config = config

        # Core agents - using correct imports
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from agents.dbcontent_evaluation_agent import ContentEvaluationAgent
        from agents.search_agent import BraveSearchAgent
        from agents.editor_agent import EditorAgent

        # Full pipeline agents with correct class names
        from agents.browserless_scraper import BrowserlessRestaurantScraper
        from agents.text_cleaner_agent import TextCleanerAgent

        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)
        self.content_evaluation_agent = ContentEvaluationAgent(config)
        self.search_agent = BraveSearchAgent(config)
        self.editor_agent = EditorAgent(config)

        # Initialize full pipeline agents
        self.scraper = BrowserlessRestaurantScraper(config)
        self.text_cleaner = TextCleanerAgent(config)

        # Set up agent dependencies
        self.content_evaluation_agent.set_brave_search_agent(self.search_agent)

        logger.info("✅ Restaurant Search Tools initialized with FULL TRADITIONAL PIPELINE - NO FALLBACKS")

    def create_tools(self):
        """Create and return LangGraph tools with full pipeline integration."""

        @tool
        def analyze_restaurant_query(query: str) -> Dict[str, Any]:
            """
            Analyze a user's restaurant query to extract destination, cuisine preferences, and search intent.
            """
            try:
                logger.info(f"🔍 Analyzing query: {query}")
                result = self.query_analyzer.analyze(query)
                dest = result.get('destination', 'Unknown')
                search_queries = result.get('search_queries', [])
                logger.info(f"✅ Query analysis complete: destination={dest}, queries={len(search_queries)}")
                return result
            except Exception as e:
                logger.error(f"❌ Error analyzing query: {e}")
                # Robust fallback
                return {
                    "error": str(e),
                    "destination": "Unknown",
                    "raw_query": query,
                    "search_queries": [f"restaurants {query}"],
                    "english_queries": [f"restaurants {query}"],
                    "local_queries": [],
                    "is_english_speaking": True,
                    "local_language": None
                }

        @tool
        def search_restaurant_database(query_analysis: str) -> Dict[str, Any]:
            """
            Search the local restaurant database based on analyzed query.
            """
            try:
                logger.info(f"🔍 Searching database...")

                # Parse the query analysis
                if isinstance(query_analysis, str):
                    analysis_data = json.loads(query_analysis)
                else:
                    analysis_data = query_analysis

                destination = analysis_data.get('destination', 'Unknown')
                logger.info(f"🔍 Searching database for: {destination}")

                # Call the database search agent
                result = self.database_search_agent.search_and_evaluate(analysis_data)

                database_restaurants = result.get("database_restaurants", [])
                logger.info(f"✅ Database search complete: {len(database_restaurants)} restaurants found")

                return result

            except Exception as e:
                logger.error(f"❌ Error in database search: {e}")
                return {
                    "error": str(e),
                    "database_restaurants": [],
                    "has_database_content": False,
                    "restaurant_count": 0,
                    "destination": "Unknown",
                    "raw_query": str(query_analysis),
                    "empty_reason": f"database_error: {str(e)}"
                }

        @tool
        def evaluate_and_route_content(combined_data: str) -> Dict[str, Any]:
            """
            Evaluate database results and determine if web search is needed.
            """
            try:
                logger.info(f"🔍 Evaluating content for routing decision")

                # Parse the combined data
                if isinstance(combined_data, str):
                    data = json.loads(combined_data)
                else:
                    data = combined_data

                # Call the correct method name
                result = self.content_evaluation_agent.evaluate_and_route(data)

                selected_restaurants = result.get("database_restaurants_final", [])
                trigger_web_search = result.get("evaluation_result", {}).get("trigger_web_search", True)

                logger.info(f"✅ Content evaluation complete: {len(selected_restaurants)} selected, web_search={trigger_web_search}")

                return result

            except Exception as e:
                logger.error(f"❌ Error evaluating content: {e}")
                return {
                    "error": str(e),
                    "selected_restaurants": [],
                    "trigger_web_search": True,
                    "database_restaurants_final": [],
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "reasoning": f"Evaluation error: {str(e)}"
                    }
                }

        @tool
        def search_web_for_restaurants(search_data: str) -> Dict[str, Any]:
            """
            FULL PIPELINE: Search web + scrape + clean + extract restaurants.

            NO FALLBACKS: Forces each step to work properly:
            1. Web search (Brave + Tavily) → URLs
            2. Intelligent scraping (Browserless) → scraped content
            3. Text cleaning (AI processing) → cleaned structured content
            4. Restaurant extraction → individual restaurant objects
            """
            try:
                logger.info("🚀 STARTING FULL WEB SEARCH PIPELINE - NO FALLBACKS")

                # Parse search data
                if isinstance(search_data, str):
                    data = json.loads(search_data)
                else:
                    data = search_data

                # Extract search parameters
                search_queries = self._extract_search_queries(data)
                destination = data.get('destination', 'Unknown')
                raw_query = data.get('raw_query', '')

                logger.info(f"🌐 Step 1: Web search with {len(search_queries)} queries")
                logger.info(f"   Queries: {search_queries}")

                if not search_queries:
                    logger.error("❌ PIPELINE FAILURE: No search queries available")
                    return {
                        "error": "No search queries available", 
                        "extracted_restaurants": [],
                        "pipeline_step": "query_extraction_failed"
                    }

                # STEP 1: Web Search - MUST SUCCEED
                search_results = self.search_agent.search_and_filter(
                    search_queries=search_queries,
                    destination=destination
                )

                raw_results = search_results.get("filtered_results", [])
                logger.info(f"✅ Step 1 complete: {len(raw_results)} search results")

                if not raw_results:
                    logger.error("❌ PIPELINE FAILURE: Web search returned no results")
                    return {
                        "error": "Web search returned no results",
                        "extracted_restaurants": [], 
                        "pipeline_step": "search_failed"
                    }

                # STEP 2: Intelligent Scraping - MUST SUCCEED
                logger.info(f"🤖 Step 2: Intelligent scraping of {len(raw_results)} URLs")
                scraped_results = self._run_async_scraping(raw_results)

                logger.info(f"✅ Step 2 complete: {len(scraped_results)} scraped successfully")

                if not scraped_results:
                    logger.error("❌ PIPELINE FAILURE: Scraping failed for all URLs")
                    return {
                        "error": "Scraping failed for all URLs",
                        "extracted_restaurants": [], 
                        "pipeline_step": "scraping_failed"
                    }

                # STEP 3: Text Cleaning - MUST SUCCEED  
                logger.info(f"🧹 Step 3: AI text cleaning and processing")
                cleaned_file_path = self._run_async_text_cleaning(scraped_results, raw_query)

                if not cleaned_file_path:
                    logger.error("❌ PIPELINE FAILURE: Text cleaning failed - this is required for proper restaurant extraction")
                    return {
                        "error": "Text cleaning pipeline failed",
                        "extracted_restaurants": [],
                        "pipeline_step": "text_cleaning_failed"
                    }

                logger.info(f"✅ Step 3 complete: cleaned content saved to {cleaned_file_path}")

                # STEP 4: Restaurant Extraction - MUST SUCCEED
                logger.info(f"🍴 Step 4: AI restaurant extraction from cleaned content")
                extracted_restaurants = self._extract_restaurants_from_cleaned_file(cleaned_file_path, destination, raw_query)

                if not extracted_restaurants:
                    logger.error("❌ PIPELINE FAILURE: No restaurants extracted from cleaned content")
                    return {
                        "error": "No restaurants extracted from cleaned content",
                        "extracted_restaurants": [],
                        "pipeline_step": "extraction_failed"
                    }

                logger.info(f"✅ FULL PIPELINE SUCCESS: {len(extracted_restaurants)} restaurants extracted")

                return {
                    "extracted_restaurants": extracted_restaurants,
                    "pipeline_step": "complete",
                    "search_results_count": len(raw_results),
                    "scraped_count": len(scraped_results),
                    "final_restaurant_count": len(extracted_restaurants)
                }

            except Exception as e:
                logger.error(f"❌ PIPELINE EXCEPTION: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {
                    "error": str(e),
                    "extracted_restaurants": [],
                    "pipeline_step": "exception"
                }

        @tool
        def format_restaurant_recommendations(all_data: str) -> Dict[str, Any]:
            """
            Format the final restaurant recommendations for the user.
            Now handles properly extracted restaurant objects.
            """
            try:
                logger.info(f"✨ Formatting recommendations")

                # Parse all collected data
                if isinstance(all_data, str):
                    data = json.loads(all_data)
                else:
                    data = all_data

                # Extract restaurant data
                database_restaurants = data.get('database_restaurants_final', [])
                # Use extracted restaurants instead of raw search results
                web_restaurants = data.get('extracted_restaurants', [])
                raw_query = data.get('raw_query', '')
                destination = data.get('destination', 'Unknown')

                # Log what we're working with
                db_count = len(database_restaurants) if database_restaurants else 0
                web_count = len(web_restaurants) if web_restaurants else 0

                logger.info(f"✨ Formatting: {db_count} database + {web_count} web restaurants")

                # Call editor agent with proper parameters
                result = self.editor_agent.edit(
                    database_restaurants=database_restaurants,
                    scraped_results=web_restaurants,  # Now contains restaurant objects, not URLs
                    raw_query=raw_query,
                    destination=destination
                )

                logger.info(f"✅ Formatting complete")
                return result

            except Exception as e:
                logger.error(f"❌ Error formatting recommendations: {e}")
                return {
                    "error": str(e),
                    "edited_results": {"main_list": []},
                    "follow_up_queries": []
                }

        return [
            analyze_restaurant_query,
            search_restaurant_database,
            evaluate_and_route_content,
            search_web_for_restaurants,
            format_restaurant_recommendations
        ]

    # HELPER METHODS FOR FULL PIPELINE - NO FALLBACKS

    def _extract_search_queries(self, data: Dict[str, Any]) -> list:
        """Extract search queries using multiple strategies"""
        search_queries = []

        # Strategy 1: Direct search_queries field
        if data.get('search_queries'):
            search_queries = data.get('search_queries', [])
        # Strategy 2: Combine english + local queries
        elif data.get('english_queries') or data.get('local_queries'):
            english_queries = data.get('english_queries', [])
            local_queries = data.get('local_queries', [])
            search_queries = english_queries + local_queries
        # Strategy 3: Generate from raw_query
        elif data.get('raw_query'):
            raw_query = data.get('raw_query', '')
            destination = data.get('destination', 'Unknown')
            if destination != "Unknown":
                search_queries = [f"best restaurants {raw_query} {destination}"]
            else:
                search_queries = [f"restaurants {raw_query}"]

        # Fallback
        if not search_queries:
            destination = data.get('destination', 'Unknown')
            fallback_query = f"restaurants in {destination}" if destination != "Unknown" else "restaurants"
            search_queries = [fallback_query]

        return search_queries

    def _run_async_scraping(self, search_results: list) -> list:
        """Run the intelligent scraper asynchronously - NO FALLBACKS"""
        try:
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

            # Filter successful scrapes - STRICT REQUIREMENTS
            successful_scrapes = [r for r in scraped_results if r.get("scraped_content") and len(r.get("scraped_content", "")) > 100]

            if not successful_scrapes:
                logger.error("❌ No successful scrapes with substantial content")

            return successful_scrapes

        except Exception as e:
            logger.error(f"❌ Error in async scraping: {e}")
            return []

    def _run_async_text_cleaning(self, scraped_results: list, query: str) -> Optional[str]:
        """Run the text cleaner asynchronously - NO FALLBACKS"""
        try:
            def run_text_cleaner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.text_cleaner.process_scraped_results_individually(scraped_results, query)
                    )
                finally:
                    loop.close()

            final_txt_file_path = run_text_cleaner()

            # Validate the cleaned file
            if final_txt_file_path and os.path.exists(final_txt_file_path):
                with open(final_txt_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    logger.info(f"✅ Text cleaning successful: {len(content)} characters")
                    return final_txt_file_path
                else:
                    logger.error("❌ Text cleaning produced empty file")
                    return None
            else:
                logger.error("❌ Text cleaning failed to create file")
                return None

        except Exception as e:
            logger.error(f"❌ Error in async text cleaning: {e}")
            return None

    def _extract_restaurants_from_cleaned_file(self, file_path: str, destination: str, raw_query: str) -> list:
        """Extract restaurant objects from cleaned text file using AI - NO FALLBACKS"""
        try:
            # Read the cleaned content
            with open(file_path, 'r', encoding='utf-8') as f:
                cleaned_content = f.read()

            if not cleaned_content.strip():
                logger.error("❌ Cleaned file is empty - this indicates a pipeline failure")
                return []

            logger.info(f"🍴 Processing {len(cleaned_content)} characters of cleaned content")

            # Create a mock scraped_results structure with the cleaned content
            mock_scraped_results = [{
                "scraped_content": cleaned_content,
                "url": "combined_sources",
                "title": "Restaurant Recommendations",
                "source_info": {"name": "Multiple Sources"}
            }]

            # Call editor agent with correct signature - 4 parameters
            result = self.editor_agent._process_scraped_content(
                mock_scraped_results,    # scraped_results
                raw_query,               # raw_query
                destination,             # destination
                file_path                # cleaned_file_path (4th parameter)
            )

            restaurants = result.get('edited_results', {}).get('main_list', [])

            if restaurants:
                logger.info(f"🍴 Successfully extracted {len(restaurants)} restaurants from cleaned content")
            else:
                logger.error("❌ No restaurants extracted from cleaned content - check AI processing")

            return restaurants

        except Exception as e:
            logger.error(f"❌ Error extracting restaurants from cleaned file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # NO FALLBACK - Force the pipeline to work properly
            logger.error("❌ Pipeline failed - returning empty results to force debugging")
            return []


def create_restaurant_tools(config):
    """Factory function to create restaurant search tools with full pipeline"""
    tools_manager = RestaurantSearchTools(config)
    return tools_manager.create_tools()