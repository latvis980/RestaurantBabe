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

        # Use the fixed query analyzer
        from agents.query_analyzer_fixed import QueryAnalyzer
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
                query_analysis: JSON string of the query analysis result from analyze_restaurant_query

            Returns:
                Dictionary with database search results including restaurants found
            """
            try:
                # FIXED: Better JSON parsing with fallback
                if isinstance(query_analysis, str):
                    try:
                        query_data = json.loads(query_analysis)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse query_analysis as JSON: {query_analysis}")
                        # Try to extract destination manually
                        if "destination" in query_analysis:
                            query_data = {"destination": "Unknown", "raw_query": query_analysis}
                        else:
                            raise ValueError("Invalid query analysis format")
                else:
                    query_data = query_analysis

                destination = query_data.get('destination', 'Unknown')
                logger.info(f"🔍 Searching database for: {destination}")
                result = self.database_search_agent.search_and_evaluate(query_data)
                count = result.get('restaurant_count', 0)
                has_content = result.get('has_database_content', False)
                logger.info(f"✅ Database search complete: {count} restaurants found, has_content={has_content}")
                logger.info(f"   Database result keys: {list(result.keys())}")
                return result
            except Exception as e:
                logger.error(f"❌ Error searching database: {e}")
                return {
                    "error": str(e),
                    "database_restaurants": [],
                    "has_database_content": False,
                    "restaurant_count": 0,
                    "destination": "Unknown",
                    "raw_query": str(query_analysis),
                    "search_flow": "error",
                    "empty_reason": f"database_error: {str(e)}"
                }

        @tool
        def evaluate_and_route_content(pipeline_data: str) -> Dict[str, Any]:
            """
            Evaluate database search results and determine if web search is needed.
            Also selects the best matching restaurants.

            Args:
                pipeline_data: JSON string containing query analysis and database results

            Returns:
                Dictionary with evaluation results and routing decision
            """
            try:
                # FIXED: Better JSON parsing with fallback
                if isinstance(pipeline_data, str):
                    try:
                        data = json.loads(pipeline_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse pipeline_data as JSON: {pipeline_data}")
                        # Create minimal data structure
                        data = {
                            "database_restaurants": [],
                            "has_database_content": False,
                            "restaurant_count": 0,
                            "destination": "Unknown",
                            "raw_query": pipeline_data
                        }
                else:
                    data = pipeline_data

                logger.info(f"🔍 Evaluating content for routing decision")
                result = self.content_evaluation_agent.evaluate(data)
                logger.info(f"✅ Content evaluation complete")
                logger.info(f"   Evaluation result keys: {list(result.keys())}")
                return result
            except Exception as e:
                logger.error(f"❌ Error evaluating content: {e}")
                return {
                    "error": str(e),
                    "selected_restaurants": [],
                    "needs_web_search": True,
                    "routing_decision": "hybrid_mode"
                }

        @tool
        def search_web_for_restaurants(search_data: str) -> Dict[str, Any]:
            """
            Search the web for restaurant information using Brave Search API.

            Args:
                search_data: JSON string containing search queries and destination

            Returns:
                Dictionary with web search results including URLs and content
            """
            try:
                # FIXED: Better JSON parsing and query extraction
                if isinstance(search_data, str):
                    try:
                        data = json.loads(search_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse search_data as JSON: {search_data}")
                        # Try to extract search queries from the raw string
                        # This is a fallback when JSON parsing fails
                        search_queries = [search_data.strip()]
                        destination = "Unknown"
                        query_metadata = {}
                        data = {
                            "search_queries": search_queries,
                            "destination": destination,
                            "query_metadata": query_metadata
                        }
                else:
                    data = search_data

                # FIXED: Extract search queries with multiple fallback strategies
                search_queries = []

                # Strategy 1: Direct search_queries field
                if data.get('search_queries'):
                    search_queries = data.get('search_queries', [])
                    logger.info(f"📝 Using search_queries field: {search_queries}")

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
                    logger.warning("❌ No search queries available, using fallback")
                    destination = data.get('destination', 'Unknown')
                    if destination != "Unknown":
                        search_queries = [f"best restaurants in {destination}"]
                    else:
                        search_queries = ["best restaurants"]
                    logger.info(f"📝 Fallback queries: {search_queries}")

                destination = data.get('destination', 'Unknown')
                query_metadata = data.get('query_metadata', {})

                logger.info(f"🌐 Searching web with {len(search_queries)} queries")
                logger.info(f"   Queries: {search_queries}")

                results = self.search_agent.search(search_queries, destination, query_metadata)
                logger.info(f"✅ Web search complete: {len(results)} results")

                return {
                    "filtered_results": results,
                    "search_performed": True,
                    "result_count": len(results),
                    "search_queries_used": search_queries
                }
            except Exception as e:
                logger.error(f"❌ Error searching web: {e}")
                return {
                    "error": str(e),
                    "filtered_results": [],
                    "search_performed": False,
                    "result_count": 0
                }

        @tool
        def format_restaurant_recommendations(recommendations_data: str) -> Dict[str, Any]:
            """
            Format restaurant recommendations into the final user-friendly output.

            Args:
                recommendations_data: JSON string containing selected restaurants and web results

            Returns:
                Dictionary with formatted recommendations ready for display
            """
            try:
                # FIXED: Better JSON parsing with fallback
                if isinstance(recommendations_data, str):
                    try:
                        data = json.loads(recommendations_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse recommendations_data as JSON: {recommendations_data}")
                        # Create minimal data structure
                        data = {
                            "selected_restaurants": [],
                            "filtered_results": [],
                            "raw_query": recommendations_data,
                            "destination": "Unknown"
                        }
                else:
                    data = recommendations_data

                logger.info(f"✨ Formatting recommendations")

                # Extract the necessary data
                selected_restaurants = data.get('selected_restaurants', [])
                web_results = data.get('filtered_results', [])
                raw_query = data.get('raw_query', '')
                destination = data.get('destination', 'Unknown')

                # Determine the mode based on available data
                if selected_restaurants and web_results:
                    mode = "hybrid"
                elif selected_restaurants:
                    mode = "database_only"
                elif web_results:
                    mode = "web_only"
                else:
                    mode = "no_results"

                logger.info(f"✨ Formatting mode: {mode}, {len(selected_restaurants)} selected, {len(web_results)} web results")

                # Call the editor agent to format the results
                result = self.editor_agent.edit(
                    selected_restaurants,
                    mode=mode,
                    query=raw_query,
                    destination=destination,
                    web_results=web_results
                )

                logger.info(f"✅ Formatting complete: formatted results")
                logger.info(f"   Formatted result keys: {list(result.keys())}")
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


def create_restaurant_tools(config):
    """Factory function to create restaurant search tools"""
    tools_manager = RestaurantSearchTools(config)
    return tools_manager.create_tools()