# agents/langgraph_tools.py
"""
FIXED LangGraph Tool Wrappers for Restaurant Recommendation Agents

Fixes the parameter passing issues and ensures web search gets proper queries.
Converts existing agents into LangGraph-compatible tools that can be used
by the LangGraph agent framework with proper state management.
"""

import logging
import json
from typing import Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class RestaurantSearchTools:
    """
    Collection of tools for restaurant search and recommendation.
    Wraps existing agents to work with LangGraph.
    """

    def __init__(self, config):
        """Initialize all agents that will be wrapped as tools"""
        self.config = config

        # Use the fixed query analyzer (now in the regular file)
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from agents.dbcontent_evaluation_agent import ContentEvaluationAgent
        from agents.search_agent import BraveSearchAgent
        from agents.editor_agent import EditorAgent

        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)
        self.content_evaluation_agent = ContentEvaluationAgent(config)
        self.search_agent = BraveSearchAgent(config)
        self.editor_agent = EditorAgent(config)

        self.content_evaluation_agent.set_brave_search_agent(self.search_agent)

        logger.info("✅ Restaurant Search Tools initialized with fixed query analyzer")

    def create_tools(self):
        """
        Create and return LangGraph tools.
        Returns a list of tool functions that can be used by the agent.
        """

        @tool
        def analyze_restaurant_query(query: str) -> Dict[str, Any]:
            """
            Analyze a user's restaurant query to extract destination, cuisine preferences, and search intent.

            Args:
                query: The user's restaurant search request (e.g., "best pizza in Rome")

            Returns:
                Dictionary with extracted information including destination, cuisine, search queries, etc.
            """
            try:
                logger.info(f"🔍 Analyzing query: {query}")
                result = self.query_analyzer.analyze(query)
                dest = result.get('destination', 'Unknown')
                cuisine = result.get('cuisine_type', 'Any')
                search_queries = result.get('search_queries', [])
                logger.info(f"✅ Query analysis complete: destination={dest}, queries={len(search_queries)}")
                logger.info(f"   Analysis result keys: {list(result.keys())}")
                logger.info(f"   Search queries: {search_queries}")
                return result
            except Exception as e:
                logger.error(f"❌ Error analyzing query: {e}")
                # Create a more robust fallback
                fallback_result = {
                    "error": str(e),
                    "destination": "Unknown",
                    "raw_query": query,
                    "search_queries": [f"restaurants {query}"],  # Always provide at least one query
                    "english_queries": [f"restaurants {query}"],
                    "local_queries": [],
                    "is_english_speaking": True,
                    "local_language": None,
                    "query_metadata": {
                        "is_english_speaking": True,
                        "local_language": None,
                        "english_query": f"restaurants {query}",
                        "local_query": None
                    }
                }
                logger.info(f"🔧 Using fallback result with search queries: {fallback_result['search_queries']}")
                return fallback_result

        @tool
        def search_restaurant_database(query_analysis: str) -> Dict[str, Any]:
            """
            Search the local restaurant database based on analyzed query.

            Args:
                query_analysis: JSON string containing the query analysis results

            Returns:
                Dictionary with database search results and metadata
            """
            try:
                logger.info(f"🔍 Searching database...")

                # Parse the query analysis (handle both string and dict)
                if isinstance(query_analysis, str):
                    analysis_data = json.loads(query_analysis)
                else:
                    analysis_data = query_analysis

                destination = analysis_data.get('destination', 'Unknown')
                logger.info(f"🔍 Searching database for: {destination}")

                # Call the database search agent
                result = self.database_search_agent.search_and_evaluate(analysis_data)

                database_restaurants = result.get("database_restaurants", [])
                has_content = result.get("has_database_content", False)

                logger.info(f"✅ Database search complete: {len(database_restaurants)} restaurants found, has_content={has_content}")
                logger.info(f"   Database result keys: {list(result.keys())}")

                return result

            except Exception as e:
                logger.error(f"❌ Error in database search: {e}")
                return {
                    "error": str(e),
                    "database_restaurants": [],
                    "has_database_content": False,
                    "restaurant_count": 0,
                    "destination": "Unknown",
                    "raw_query": query_analysis if isinstance(query_analysis, str) else str(query_analysis),
                    "empty_reason": f"database_error: {str(e)}"
                }

        @tool
        def evaluate_and_route_content(combined_data: str) -> Dict[str, Any]:
            """
            Evaluate database results and determine if web search is needed.

            Args:
                combined_data: JSON string containing query analysis + database results

            Returns:
                Dictionary with evaluation results and routing decisions
            """
            try:
                logger.info(f"🔍 Evaluating content for routing decision")

                # Parse the combined data
                if isinstance(combined_data, str):
                    data = json.loads(combined_data)
                else:
                    data = combined_data

                # FIXED: Call the correct method name
                result = self.content_evaluation_agent.evaluate_and_route(data)

                selected_restaurants = result.get("database_restaurants_final", [])
                trigger_web_search = result.get("evaluation_result", {}).get("trigger_web_search", True)

                logger.info(f"✅ Content evaluation complete: {len(selected_restaurants)} selected, web_search={trigger_web_search}")
                logger.info(f"   Evaluation result keys: {list(result.keys())}")

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
            Search the web for restaurant recommendations using multiple strategies.

            Args:
                search_data: JSON string containing search parameters and queries

            Returns:
                Dictionary with web search results
            """
            try:
                # Parse search data
                if isinstance(search_data, str):
                    data = json.loads(search_data)
                else:
                    data = search_data

                # ENHANCED: Multiple strategies to extract search queries
                search_queries = []
                destination = data.get('destination', 'Unknown')

                # Strategy 1: Direct search_queries field
                if data.get('search_queries'):
                    search_queries = data.get('search_queries', [])
                    logger.info(f"📝 Using direct search_queries: {search_queries}")

                # Strategy 2: Combine english_queries and local_queries
                elif data.get('english_queries') or data.get('local_queries'):
                    english_queries = data.get('english_queries', [])
                    local_queries = data.get('local_queries', [])
                    search_queries = english_queries + local_queries
                    logger.info(f"📝 Combining queries: {len(english_queries)} English + {len(local_queries)} local")

                # Strategy 3: Extract from query_metadata
                elif data.get('query_metadata'):
                    metadata = data.get('query_metadata', {})
                    english_query = metadata.get('english_query')
                    local_query = metadata.get('local_query')
                    if english_query:
                        search_queries.append(english_query)
                    if local_query:
                        search_queries.append(local_query)
                    logger.info(f"📝 Using query_metadata: {search_queries}")

                # Strategy 4: Generate from raw_query and destination
                elif data.get('raw_query'):
                    raw_query = data.get('raw_query', '')
                    destination = data.get('destination', 'Unknown')
                    if destination != "Unknown":
                        search_queries = [f"best restaurants {raw_query} {destination}"]
                    else:
                        search_queries = [f"restaurants {raw_query}"]
                    logger.info(f"📝 Generated from raw_query: {search_queries}")

                # Strategy 5: Last resort fallback
                if not search_queries:
                    logger.warning("⚠️ No search queries found, using fallback")
                    fallback_query = f"restaurants in {destination}" if destination != "Unknown" else "restaurants"
                    search_queries = [fallback_query]

                logger.info(f"🌐 Searching web with {len(search_queries)} queries")
                logger.info(f"   Queries: {search_queries}")
                logger.info(f"   Destination: {destination}")

                if not search_queries:
                    return {
                        "error": "No search queries available",
                        "filtered_results": []
                    }

                # Execute web search
                search_results = self.search_agent.search_and_filter(
                    search_queries=search_queries,
                    destination=destination
                )

                filtered_results = search_results.get("filtered_results", [])
                logger.info(f"✅ Web search complete: {len(filtered_results)} results")

                return search_results

            except Exception as e:
                logger.error(f"❌ Error in web search: {e}")
                return {
                    "error": str(e),
                    "filtered_results": [],
                    "search_results": [],
                    "high_quality_sources": []
                }

        @tool
        def format_restaurant_recommendations(all_data: str) -> Dict[str, Any]:
            """
            Format the final restaurant recommendations for the user.

            Args:
                all_data: JSON string containing all collected data from previous steps

            Returns:
                Dictionary with formatted recommendations ready for display
            """
            try:
                logger.info(f"✨ Formatting recommendations")

                # Parse all collected data
                if isinstance(all_data, str):
                    data = json.loads(all_data)
                else:
                    data = all_data

                # Extract the necessary data
                selected_restaurants = data.get('database_restaurants_final', [])
                web_results = data.get('filtered_results', [])
                raw_query = data.get('raw_query', '')
                destination = data.get('destination', 'Unknown')

                # Determine the mode based on available data
                db_count = len(selected_restaurants) if selected_restaurants else 0
                web_count = len(web_results) if web_results else 0

                if db_count > 0 and web_count > 0:
                    mode = "hybrid"
                elif db_count > 0:
                    mode = "database_only"
                elif web_count > 0:
                    mode = "web_only"
                else:
                    mode = "no_results"

                logger.info(f"✨ Formatting mode: {mode}, {db_count} selected, {web_count} web results")

                # Call the editor agent to format the results
                result = self.editor_agent.edit(
                    database_restaurants=selected_restaurants,
                    scraped_results=web_results,
                    raw_query=raw_query,
                    destination=destination
                )

                logger.info(f"✅ Formatting complete: formatted results")
                logger.info(f"   Formatted result keys: {list(result.keys())}")
                return result

            except Exception as e:
                logger.error(f"❌ Error formatting recommendations: {e}")
                return {
                    "error": str(e),
                    "edited_results": {"main_list": []},
                    "follow_up_queries": [],
                    "fallback_message": "I encountered an issue formatting the recommendations. Please try a different search."
                }

        return [
            analyze_restaurant_query,
            search_restaurant_database,
            evaluate_and_route_content,
            search_web_for_restaurants,
            format_restaurant_recommendations
        ]


def create_restaurant_tools(config):
    """Factory function to create restaurant search tools"""
    tools_manager = RestaurantSearchTools(config)
    return tools_manager.create_tools()