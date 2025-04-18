"""
Restaurant Editor Agent using OpenAI.

This agent analyzes raw search results, enriches them with additional details,
and compiles comprehensive restaurant recommendations.
"""
import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import config

class RestaurantEditorAgent:
    """
    An agent that uses OpenAI to analyze, enrich, and compile restaurant search results
    into comprehensive recommendations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        temperature: float = None,  # Lower temperature for more reliable analysis
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the editor agent with OpenAI.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: The OpenAI model to use
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Custom system prompt (uses default if not provided)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.EDITOR_MODEL
        self.temperature = temperature if temperature is not None else config.EDITOR_TEMPERATURE

        # Use system prompt from config if not provided
        self.system_prompt = system_prompt or config.EDITOR_SYSTEM_PROMPT

        # Initialize the OpenAI model
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
        )

        # Create the prompt template for analysis
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self._get_analysis_prompt_template())
        ])

        # Create the prompt template for identifying missing information
        self.missing_info_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self._get_missing_info_template())
        ])

        # Create the formatting chain
        self.analysis_chain = self.analysis_prompt | self.llm | StrOutputParser()
        self.missing_info_chain = self.missing_info_prompt | self.llm | StrOutputParser()

    def _get_analysis_prompt_template(self) -> str:
        """
        Get the human prompt template for analyzing restaurant results.

        Returns:
            Human prompt template string
        """
        return """
        USER QUERY: {query}
        LOCATION: {location}
        CUISINE: {cuisine}

        RESTAURANT SEARCH RESULTS:
        {restaurant_results}

        Please analyze these restaurant search results and create comprehensive profiles for the most promising restaurants.
        For each restaurant, compile all available information from potentially multiple sources into a single detailed profile.
        Identify which restaurants have the strongest recommendations from reputable sources.

        Return a JSON array of the best restaurant recommendations with these fields for each:
        - name: Restaurant name
        - address: Full address if available
        - description: Detailed description (100+ words when possible) covering cuisine style, chef background, atmosphere
        - price_range: Price indicator ($/$$/$$$)
        - recommended_dishes: Array of 1-3 signature/recommended dishes
        - sources: Array of sources that recommended this restaurant
        - website: Restaurant website if available
        """

    def _get_missing_info_template(self) -> str:
        """
        Get the prompt template for identifying missing information.

        Returns:
            Human prompt template string
        """
        return """
        Based on the following restaurant recommendations, identify what critical information is missing that would make these recommendations more valuable.
        Create an array of follow-up search queries that would help find this missing information.

        RESTAURANT RECOMMENDATIONS:
        {analyzed_results}

        For each restaurant with incomplete information, create a specific query to find:
        1. Exact address (if missing)
        2. Price range (if missing)
        3. Signature dishes or menu highlights (if missing)
        4. Website or reservation information (if missing)

        Return a JSON array of follow-up queries, with each object having these fields:
        - restaurant_name: Name of the restaurant
        - location: Location context
        - missing_fields: Array of what information is missing
        - search_query: A specific query to find the missing information
        """

    def _get_compilation_template(self) -> str:
        """
        Get the prompt template for compiling final results.

        Returns:
            Human prompt template string
        """
        return """
        Compile the original restaurant information with the additional details from follow-up searches to create 
        complete, comprehensive restaurant profiles.

        ORIGINAL RESTAURANT INFORMATION:
        {analyzed_results}

        ADDITIONAL DETAILS FROM FOLLOW-UP SEARCHES:
        {enriched_data}

        Create a final JSON array of restaurant recommendations with all available information merged.
        For each restaurant, ensure you have the most complete:
        - name
        - street address
        - description (enhanced with any new details)
        - price_range
        - recommended_dishes (array)
        - opening_hours (if available)
        - reservation_info (if available, if not, omit)
        - website
        - recommended by (NEVER mention Tripadvisor, Yelp, or Google)
        """

    @traceable(name="analyze_restaurant_results")
    def analyze(self, query: str, restaurant_results: List[Dict[str, Any]], 
              location: str = "", cuisine: str = "") -> Dict[str, Any]:
        """
        Analyze restaurant search results to identify the most promising options.

        Args:
            query: The original user query
            restaurant_results: List of restaurant information dictionaries
            location: The location context
            cuisine: The cuisine type

        Returns:
            Dictionary with analyzed restaurant recommendations
        """
        # Convert restaurant results to a readable string format
        results_str = json.dumps(restaurant_results, indent=2)

        # Invoke the analysis chain
        analyzed_results = self.analysis_chain.invoke({
            "query": query,
            "restaurant_results": results_str,
            "location": location,
            "cuisine": cuisine
        })

        # Create a structure to return both the analyzed text and 
        # attempt to parse it as structured data
        result = {
            "analyzed_text": analyzed_results,
            "structured_data": None
        }

        # Try to parse the analyzed results as JSON
        try:
            # Look for JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', analyzed_results, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                result["structured_data"] = json.loads(json_str)
        except:
            # If parsing fails, keep structured_data as None
            pass

        return result

    @traceable(name="identify_missing_information")
    def identify_missing_info(self, analyzed_results: str) -> List[Dict[str, Any]]:
        """
        Identify what information is missing from the analyzed results.

        Args:
            analyzed_results: The analyzed restaurant recommendations

        Returns:
            List of follow-up queries to find missing information
        """
        # Invoke the missing info identification chain
        missing_info_response = self.missing_info_chain.invoke({
            "analyzed_results": analyzed_results
        })

        # Try to parse the response as JSON
        try:
            # Look for JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', missing_info_response, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return []
        except:
            # If parsing fails, return an empty list
            return []

    @traceable(name="compile_enriched_results")
    def compile(self, analyzed_results: str, enriched_data: Dict[str, Any]) -> str:
        """
        Compile the original analysis with enriched data into final recommendations.

        Args:
            analyzed_results: The initial analyzed restaurant recommendations
            enriched_data: Additional details gathered from follow-up searches

        Returns:
            Compiled restaurant recommendations
        """
        # Create the compilation prompt
        compilation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self._get_compilation_template())
        ])

        # Create the compilation chain
        compilation_chain = compilation_prompt | self.llm | StrOutputParser()

        # Convert enriched data to a string
        enriched_str = json.dumps(enriched_data, indent=2)

        # Invoke the compilation chain
        compiled_results = compilation_chain.invoke({
            "analyzed_results": analyzed_results,
            "enriched_data": enriched_str
        })

        return compiled_results

# Example usage
if __name__ == "__main__":
    # For testing
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set")

    editor = RestaurantEditorAgent()

    # Mock restaurant results
    mock_results = [
        {
            "name": "Sushi Nakazawa",
            "description": "Acclaimed sushi restaurant offering an omakase experience with fish sourced from around the world.",
            "url": "https://guide.michelin.com/us/en/new-york-state/new-york/restaurant/sushi-nakazawa",
            "source": "guide.michelin.com"
        },
        {
            "name": "Le Bernardin",
            "description": "Upscale French seafood restaurant with three Michelin stars, known for exquisite seafood preparations.",
            "url": "https://www.cntraveler.com/restaurants/new-york/le-bernardin",
            "source": "cntraveler.com"
        }
    ]

    # Test the analysis
    analysis_result = editor.analyze("best seafood restaurants in NYC", mock_results)
    print("Analysis Result:")
    print(analysis_result["analyzed_text"])

    # Test missing info identification
    missing_info = editor.identify_missing_info(analysis_result["analyzed_text"])
    print("\nMissing Information Queries:")
    for query in missing_info:
        print(f"- {query.get('restaurant_name')}: {query.get('search_query')}")

    # Mock enriched data
    mock_enriched = {
        "Sushi Nakazawa": {
            "address": "23 Commerce St",
            "price_range": "$$$",
            "signature_dishes": ["Fatty Tuna", "Sea Urchin", "Horsehair Crab"],
            "opening_hours": "5:00 PM - 10:30 PM, Closed Mondays"
        }
    }

    # Test compilation
    compiled = editor.compile(analysis_result["analyzed_text"], mock_enriched)
    print("\nCompiled Results:")
    print(compiled)