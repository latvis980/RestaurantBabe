"""
OpenAI Formatting Agent for restaurant recommendations.

This module formats the raw search results into friendly,
engaging restaurant recommendations using OpenAI.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import config
from langsmith import traceable

class RestaurantFormattingAgent:
    """
    An agent that uses OpenAI to format raw restaurant search results
    with a friendly, engaging, and somewhat humorous tone.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the formatting agent with OpenAI.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: The OpenAI model to use
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Custom system prompt (uses default ToV if not provided)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or config.RESTAURANT_TOV_PROMPT

        # Initialize the OpenAI model
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
        )

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self._get_human_prompt_template())
        ])

        # Create the formatting chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _get_human_prompt_template(self) -> str:
        """
        Get the human prompt template for formatting restaurant results.

        Returns:
            Human prompt template string
        """
        return """
        USER QUERY: {query}
        USER LANGUAGE: {language}

        RESTAURANT SEARCH RESULTS:
        {restaurant_results}

        Please format these restaurant recommendations according to the specified tone and format. Be conversational and friendly. Make sure to detect the language of the query and respond in the same language.
        """

    @traceable(name="format_restaurant_results")
    def format(self, query: str, restaurant_results: List[Dict[str, Any]], language: str = "English") -> str:
        """
        Format restaurant search results into friendly recommendations.

        Args:
            query: The original user query
            restaurant_results: List of restaurant information dictionaries
            language: The detected language of the user query

        Returns:
            Formatted restaurant recommendations
        """
        # Convert restaurant results to a readable string format
        results_str = json.dumps(restaurant_results, indent=2)

        # Invoke the formatting chain
        formatted_response = self.chain.invoke({
            "query": query,
            "restaurant_results": results_str,
            "language": language
        })

        return formatted_response

# Example usage
if __name__ == "__main__":
    # For testing
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set")

    formatter = RestaurantFormattingAgent()

    # Mock restaurant results
    mock_results = [
        {
            "name": "Sushi Nakazawa",
            "description": "Acclaimed sushi restaurant offering an omakase experience with fish sourced from around the world.",
            "url": "https://guide.michelin.com/us/en/new-york-state/new-york/restaurant/sushi-nakazawa",
            "source": "guide.michelin.com",
            "score": 0.95
        },
        {
            "name": "Le Bernardin",
            "description": "Upscale French seafood restaurant with three Michelin stars, known for exquisite seafood preparations.",
            "url": "https://www.cntraveler.com/restaurants/new-york/le-bernardin",
            "source": "cntraveler.com",
            "score": 0.92
        }
    ]

    formatted = formatter.format("best seafood restaurants in NYC", mock_results)
    print(formatted)