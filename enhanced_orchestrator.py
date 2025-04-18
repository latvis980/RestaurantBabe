"""
Updated LangChain Orchestrator for the Restaurant Recommendation App.

This module connects the enhanced search agent (Perplexity), the new editor agent,
and the OpenAI formatting agent using LangChain.
"""
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client, traceable
import os
import json

import config
from openai_agent import RestaurantFormattingAgent
from editor_agent import RestaurantEditorAgent
from enhanced_perplexity_agent import EnhancedPerplexitySearchAgent

class EnhancedRestaurantRecommender:
    """
    Enhanced orchestrator that connects the search agent, editor agent, and formatting agent.
    """

    def __init__(
        self,
        langsmith_api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        enable_tracing: Optional[bool] = None
    ):
        """
        Initialize the enhanced restaurant recommender with LangChain components.

        Args:
            langsmith_api_key: LangSmith API key for tracing
            project_name: LangSmith project name
            enable_tracing: Whether to enable LangSmith tracing
        """
        # LangSmith configuration
        self.langsmith_api_key = langsmith_api_key or config.LANGSMITH_API_KEY
        self.project_name = project_name or config.LANGSMITH_PROJECT
        self.enable_tracing = enable_tracing if enable_tracing is not None else config.LANGSMITH_TRACING

        if self.enable_tracing and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.project_name

        # Initialize the LangSmith client (for custom tracing if needed)
        self.langsmith_client = Client(api_key=self.langsmith_api_key) if self.langsmith_api_key else None

        # Initialize search agent
        print(f"Using Enhanced Perplexity search provider with model: {config.PERPLEXITY_MODEL}")
        self.search_agent = EnhancedPerplexitySearchAgent(
            model=config.PERPLEXITY_MODEL,
            max_results=config.PERPLEXITY_MAX_RESULTS
        )

        # Initialize editor agent
        self.editor_agent = RestaurantEditorAgent()

        # Initialize formatting agent
        self.formatting_agent = RestaurantFormattingAgent()

        # Create the combined chain
        self._create_chain()

    def _create_chain(self):
        """Create the LangChain chain connecting search, editor, and formatting agents."""
        # Define the chain
        self.chain = RunnablePassthrough.assign(
            restaurant_results=lambda x: self.search_agent.search(
                query=x["query"],
                location=x.get("location", ""),
                cuisine=x.get("cuisine", "")
            )
        ) | RunnablePassthrough.assign(
            analyzed_results=lambda x: self.editor_agent.analyze(
                query=x["query"],
                restaurant_results=x["restaurant_results"],
                location=x.get("location", ""),
                cuisine=x.get("cuisine", "")
            )
        ) | RunnablePassthrough.assign(
            missing_info_queries=lambda x: self.editor_agent.identify_missing_info(
                analyzed_results=x["analyzed_results"]["analyzed_text"]
            )
        ) | RunnablePassthrough.assign(
            enriched_data=lambda x: self._perform_follow_up_searches(
                x["missing_info_queries"],
                x.get("location", "")
            )
        ) | RunnablePassthrough.assign(
            compiled_results=lambda x: self.editor_agent.compile(
                analyzed_results=x["analyzed_results"]["analyzed_text"],
                enriched_data=x["enriched_data"]
            )
        ) | RunnablePassthrough.assign(
            formatted_response=lambda x: self.formatting_agent.format(
                query=x["query"],
                restaurant_results=x["compiled_results"],
                language=x.get("language", "English")
            )
        )

    def _perform_follow_up_searches(self, missing_info_queries: List[Dict[str, Any]], location: str) -> Dict[str, Any]:
        """
        Perform follow-up searches for missing restaurant information.

        Args:
            missing_info_queries: List of queries for missing information
            location: The location context

        Returns:
            Dictionary of enriched data for each restaurant
        """
        enriched_data = {}

        # Limit to top 5 follow-up searches to keep response times reasonable
        for query_data in missing_info_queries[:5]:
            if not isinstance(query_data, dict):
                continue

            restaurant_name = query_data.get("restaurant_name")
            if not restaurant_name:
                continue

            # Log the follow-up search
            print(f"Performing follow-up search for: {restaurant_name}")

            # Use the search agent to find more details
            details = self.search_agent.follow_up_search(
                restaurant_name=restaurant_name,
                location=location or query_data.get("location", "")
            )

            if details:
                enriched_data[restaurant_name] = details

        return enriched_data

    @traceable(name="get_enhanced_restaurant_recommendation")
    def get_recommendation(self, 
                          query: str, 
                          location: str = "", 
                          cuisine: str = "", 
                          language: str = "English") -> Dict[str, Any]:
        """
        Get enhanced restaurant recommendations based on user query.

        Args:
            query: The user's restaurant request
            location: Optional location filter
            cuisine: Optional cuisine type filter
            language: The detected language of the user query

        Returns:
            Dictionary with search results, analyzed data, and formatted response
        """
        # Create input dictionary
        inputs = {
            "query": query,
            "location": location,
            "cuisine": cuisine,
            "language": language
        }

        # Use LangSmith tracing context if enabled
        if self.enable_tracing:
            with tracing_v2_enabled(project_name=self.project_name):
                result = self.chain.invoke(inputs)
        else:
            result = self.chain.invoke(inputs)

        return result

    async def get_recommendation_async(self, 
                                     query: str, 
                                     location: str = "", 
                                     cuisine: str = "", 
                                     language: str = "English") -> Dict[str, Any]:
        """
        Async version of get_recommendation.
        """
        # Create input dictionary
        inputs = {
            "query": query,
            "location": location,
            "cuisine": cuisine,
            "language": language
        }

        # Use LangSmith tracing context if enabled
        if self.enable_tracing:
            with tracing_v2_enabled(project_name=self.project_name):
                result = await self.chain.ainvoke(inputs)
        else:
            result = await self.chain.ainvoke(inputs)

        return result

# Example usage
if __name__ == "__main__":
    # For testing
    import asyncio

    # Verify required API keys
    try:
        config.validate_configuration()
    except ValueError as e:
        print(f"Configuration error: {e}")
        # For testing, continue with available keys

    # Initialize recommender
    recommender = EnhancedRestaurantRecommender()

    # Synchronous example
    print("Getting enhanced restaurant recommendation...")
    result = recommender.get_recommendation(
        query="romantic dinner",
        location="Paris",
        cuisine="French"
    )

    print("\nFormatted Response:")
    print(result["formatted_response"])