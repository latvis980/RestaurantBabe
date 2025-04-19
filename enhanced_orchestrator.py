"""
Updated LangChain Orchestrator for the Restaurant Recommendation App.

This module connects the enhanced search agent (Perplexity), the new editor agent,
and the OpenAI formatting agent using LangChain.
"""
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

        # Initialize search agent with configuration from config
        print(f"Using Enhanced Perplexity search provider with model: {config.PERPLEXITY_MODEL}")
        self.search_agent = EnhancedPerplexitySearchAgent(
            model=config.PERPLEXITY_MODEL,
            max_results=config.PERPLEXITY_MAX_RESULTS
        )

        # Initialize editor agent
        self.editor_agent = RestaurantEditorAgent()

        # Initialize formatting agent
        self.formatting_agent = RestaurantFormattingAgent()

        # Initialize OpenAI for conversation responses
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=0.7,
        )

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
        """
        enriched_data = {}

        # Limit to top 5 follow-up searches
        for query_data in missing_info_queries[:config.EDITOR_MAX_FOLLOWUPS]:
            if not isinstance(query_data, dict):
                continue

            restaurant_name = query_data.get("restaurant_name")
            if not restaurant_name:
                continue

            # Extract missing fields from the query data
            missing_fields = query_data.get("missing_fields", [])
            print(f"Performing follow-up search for: {restaurant_name}, missing fields: {missing_fields}")

            # Use the search agent with missing fields parameter
            details = self.search_agent.follow_up_search(
                restaurant_name=restaurant_name,
                location=location or query_data.get("location", ""),
                missing_fields=missing_fields  # Pass the missing fields
            )

            if details:
                enriched_data[restaurant_name] = details

        return enriched_data

    def get_conversation_response(self, query: str, language: str = "English") -> str:
        """
        Get a conversational response that explicitly avoids restaurant recommendations.

        Args:
            query: The user's message
            language: The detected language

        Returns:
            A friendly response that doesn't include restaurant recommendations
        """
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", config.CONVERSATION_HANDLER_PROMPT),
            ("human", f"User message: {query}\nUser message language: {language}")
        ])

        conversation_chain = conversation_prompt | self.llm | StrOutputParser()

        # Use LangSmith tracing if enabled
        if self.enable_tracing:
            with tracing_v2_enabled(project_name=self.project_name):
                response = conversation_chain.invoke({})
        else:
            response = conversation_chain.invoke({})

        return response

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
                result = await self.chain.ainvoke(inputs)
        else:
            result = await self.chain.ainvoke(inputs)

        return result

    async def get_conversation_response_async(self, query: str, language: str = "English") -> str:
        """
        Async version of get_conversation_response.

        Args:
            query: The user's message
            language: The detected language

        Returns:
            A friendly response that doesn't include restaurant recommendations
        """
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", config.CONVERSATION_HANDLER_PROMPT),
            ("human", f"User message: {query}\nUser message language: {language}")
        ])

        conversation_chain = conversation_prompt | self.llm | StrOutputParser()

        # Use LangSmith tracing if enabled
        if self.enable_tracing:
            with tracing_v2_enabled(project_name=self.project_name):
                response = await conversation_chain.ainvoke({})
        else:
            response = await conversation_chain.ainvoke({})

        return response

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

    # Conversation example
    print("\nGetting conversation response...")
    response = recommender.get_conversation_response("How are you today?")
    print(response)