"""
Tavily Search Agent for finding restaurant recommendations from specific sources.
"""
import os
import requests
from typing import List, Dict, Any, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
import config

class RestaurantSearchAgent:
    """
    A search agent that uses Tavily to find restaurant recommendations
    from specific sources like Conde Nast, Michelin guides, etc.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        sources: Optional[List[str]] = None,
        search_depth: str = "basic",  # Changed from "moderate" to "basic"
        max_results: int = 5
    ):
        """
        Initialize the restaurant search agent with Tavily API.

        Args:
            api_key: Tavily API key (uses TAVILY_API_KEY env var if not provided)
            sources: List of domains to search
            search_depth: Search depth - "basic", "moderate", or "deep"
            max_results: Maximum number of results to return
        """
        self.api_key = api_key or config.TAVILY_API_KEY
        self.sources = sources or config.RESTAURANT_SOURCES
        self.search_depth = search_depth
        self.max_results = max_results

        # Set up Tavily search tool
        self.search_tool = TavilySearchResults(
            tavily_api_key=self.api_key,
            search_depth=self.search_depth,
            max_results=self.max_results,
            include_domains=self.sources,
        )

    def search(self, query: str, location: str = "", cuisine: str = "") -> List[Dict[str, Any]]:
        """
        Search for restaurant recommendations based on query and optional filters.

        Args:
            query: The search query for restaurants
            location: Optional location filter
            cuisine: Optional cuisine type filter

        Returns:
            List of restaurant information dictionaries
        """
        # Build a restaurant-specific search query
        search_query = f"restaurant {query}"
        if location:
            search_query += f" in {location}"
        if cuisine:
            search_query += f" {cuisine} cuisine"

        # Add more detailed debugging
        print(f"Searching Tavily for: {search_query}")
        print(f"Using these domains: {self.sources}")

        # Execute the search - try direct API first
        try:
            # Direct API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "query": search_query,
                "search_depth": self.search_depth,
                "max_results": self.max_results,
                "include_domains": self.sources,
            }

            print("Sending direct API request to Tavily")
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                print("Tavily returned a successful response")
                results = response.json().get("results", [])
                return self._process_results(results)
            else:
                print(f"Tavily API error: {response.status_code} - {response.text}")
                # Fall back to using the tool
                return self._fallback_search(search_query)

        except Exception as e:
            print(f"Error during direct Tavily API request: {e}")
            # Try the LangChain tool as fallback
            return self._fallback_search(search_query)

    def _fallback_search(self, search_query):
        """Fallback search method using the LangChain tool."""
        try:
            print("Trying fallback search with LangChain tool")
            results = self.search_tool.invoke(search_query)

            # Check what was returned
            print(f"Tavily search results type: {type(results)}")

            if isinstance(results, list):
                return self._process_results(results)
            elif isinstance(results, str):
                print(f"Tavily returned a string: {results}")
                # Create a minimal result when all else fails
                return [{
                    "name": "Search recommendation",
                    "description": f"We couldn't find specific recommendations for '{search_query}'. Please try a different query.",
                    "url": "",
                    "source": "system",
                    "score": 0,
                }]
            else:
                print(f"Tavily returned unexpected type: {type(results)}")
                return []
        except Exception as e:
            print(f"Fallback search error: {e}")
            return []

    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw Tavily search results into a structured format for restaurant information.

        Args:
            results: Raw search results from Tavily

        Returns:
            Structured restaurant information
        """
        processed_results = []

        for result in results:
            # First, ensure the result is a dictionary
            if not isinstance(result, dict):
                print(f"Skipping non-dictionary result: {result}")
                continue

            # Extract the source domain
            url = result.get("url", "")
            source_domain = next(
                (source for source in self.sources if source in url), 
                "unknown source"
            )

            # Create structured restaurant information
            restaurant_info = {
                "name": result.get("title", "Unknown Restaurant"),
                "description": result.get("content", "No description available"),
                "url": url,
                "source": source_domain,
                "score": result.get("score", 0),
            }

            processed_results.append(restaurant_info)

        return processed_results

# Example usage
if __name__ == "__main__":
    # For testing
    import os
    if "TAVILY_API_KEY" not in os.environ:
        print("Warning: TAVILY_API_KEY environment variable not set")

    search_agent = RestaurantSearchAgent()
    results = search_agent.search("best sushi", "New York")

    print("Search Results:")
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. {result['name']}")
        print(f"Source: {result['source']}")
        print(f"Description: {result['description'][:150]}...")