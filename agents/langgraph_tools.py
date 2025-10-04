"""
LangGraph Tool Wrappers for Restaurant Recommendation Agents

Converts existing agents into LangGraph-compatible tools that can be used
by the LangGraph agent framework with proper state management.
"""

import logging
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
        
        logger.info("✅ Restaurant Search Tools initialized")

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
                logger.info(f"✅ Query analysis complete: {result.get('destination', 'Unknown')}")
                return result
            except Exception as e:
                logger.error(f"❌ Error analyzing query: {e}")
                return {
                    "error": str(e),
                    "destination": "Unknown",
                    "raw_query": query
                }

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
                import json
                if isinstance(query_analysis, str):
                    query_data = json.loads(query_analysis)
                else:
                    query_data = query_analysis
                    
                logger.info(f"🔍 Searching database for: {query_data.get('destination', 'Unknown')}")
                result = self.database_search_agent.search_and_evaluate(query_data)
                logger.info(f"✅ Database search complete: {result.get('restaurant_count', 0)} restaurants found")
                return result
            except Exception as e:
                logger.error(f"❌ Error searching database: {e}")
                return {
                    "error": str(e),
                    "database_restaurants": [],
                    "has_database_content": False,
                    "restaurant_count": 0
                }

        @tool
        def evaluate_and_route_content(pipeline_data: str) -> Dict[str, Any]:
            """
            Evaluate database search results and determine if web search is needed.
            Also selects the best matching restaurants.
            
            Args:
                pipeline_data: JSON string containing query analysis and database search results
            
            Returns:
                Dictionary with evaluation results, selected restaurants, and routing decision
            """
            try:
                import json
                if isinstance(pipeline_data, str):
                    data = json.loads(pipeline_data)
                else:
                    data = pipeline_data
                    
                logger.info(f"🧠 Evaluating content for: {data.get('destination', 'Unknown')}")
                result = self.content_evaluation_agent.evaluate_and_route(data)
                logger.info(f"✅ Evaluation complete: DB sufficient={result.get('database_sufficient', False)}")
                return result
            except Exception as e:
                logger.error(f"❌ Error evaluating content: {e}")
                return {
                    "error": str(e),
                    "database_sufficient": False,
                    "trigger_web_search": True,
                    "selected_restaurants": []
                }

        @tool
        def search_web_for_restaurants(search_data: str) -> Dict[str, Any]:
            """
            Search the web using Brave Search API for restaurant recommendations.
            
            Args:
                search_data: JSON string containing search queries and destination
            
            Returns:
                Dictionary with web search results including URLs and content
            """
            try:
                import json
                if isinstance(search_data, str):
                    data = json.loads(search_data)
                else:
                    data = search_data
                    
                search_queries = data.get('search_queries', [])
                destination = data.get('destination', 'Unknown')
                query_metadata = data.get('query_metadata', {})
                
                logger.info(f"🌐 Searching web with {len(search_queries)} queries")
                results = self.search_agent.search(search_queries, destination, query_metadata)
                logger.info(f"✅ Web search complete: {len(results)} results")
                
                return {
                    "filtered_results": results,
                    "search_performed": True,
                    "result_count": len(results)
                }
            except Exception as e:
                logger.error(f"❌ Error searching web: {e}")
                return {
                    "error": str(e),
                    "filtered_results": [],
                    "search_performed": False
                }

        @tool
        def format_restaurant_recommendations(recommendations_data: str) -> Dict[str, Any]:
            """
            Format restaurant recommendations into the final user-friendly output.
            
            Args:
                recommendations_data: JSON string containing restaurant data to format
            
            Returns:
                Dictionary with formatted recommendations ready for display
            """
            try:
                import json
                if isinstance(recommendations_data, str):
                    data = json.loads(recommendations_data)
                else:
                    data = recommendations_data
                    
                logger.info(f"✨ Formatting recommendations")
                
                selected_restaurants = data.get('selected_restaurants', [])
                raw_query = data.get('raw_query', '')
                destination = data.get('destination', 'Unknown')
                scraped_results = data.get('scraped_results', [])
                processing_mode = data.get('processing_mode', 'database_only')
                
                formatted = self.editor_agent.edit(
                    database_restaurants=selected_restaurants,
                    scraped_results=scraped_results,
                    raw_query=raw_query,
                    destination=destination,
                    processing_mode=processing_mode
                )
                
                logger.info(f"✅ Formatting complete: {len(formatted.get('restaurants', []))} restaurants")
                return formatted
            except Exception as e:
                logger.error(f"❌ Error formatting recommendations: {e}")
                return {
                    "error": str(e),
                    "restaurants": []
                }

        return [
            analyze_restaurant_query,
            search_restaurant_database,
            evaluate_and_route_content,
            search_web_for_restaurants,
            format_restaurant_recommendations
        ]


def create_restaurant_tools(config):
    """
    Factory function to create restaurant search tools.
    
    Args:
        config: Application configuration object
    
    Returns:
        List of LangGraph tools for restaurant search
    """
    tools_manager = RestaurantSearchTools(config)
    return tools_manager.create_tools()
