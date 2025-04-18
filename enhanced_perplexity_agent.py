"""
Enhanced Perplexity Search Agent for finding comprehensive restaurant and bar recommendations.

This module connects directly to the Perplexity API using their latest documentation
and leverages LangChain for proper integration.
"""
import os
import requests
from typing import List, Dict, Any, Optional
import json
import config
from langchain_perplexity import ChatPerplexity

class EnhancedPerplexitySearchAgent:
    """
    An improved search agent that uses Perplexity API to find detailed restaurant 
    and bar recommendations from multiple reputable sources.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,  # Increased from 5 to 10 for more comprehensive results
        model: str = "llama-3.1-sonar-small-128k-online"
    ):
        """
        Initialize the enhanced search agent with Perplexity API.

        Args:
            api_key: Perplexity API key (uses PERPLEXITY_API_KEY env var if not provided)
            max_results: Maximum number of results to return
            model: The Perplexity model to use
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY") or config.PERPLEXITY_API_KEY
        if not self.api_key:
            raise ValueError("Perplexity API key is required")

        self.max_results = max_results
        self.model = model
        self.api_url = "https://api.perplexity.ai/chat/completions"

        # Define trusted sources from config
        self.trusted_sources = config.RESTAURANT_SOURCES
        self.excluded_sources = config.EXCLUDED_RESTAURANT_SOURCES

        # Initialize LangChain integration with Perplexity
        os.environ["PPLX_API_KEY"] = self.api_key
        self.langchain_pplx = ChatPerplexity(
            model=self.model,
            temperature=0.2,
            pplx_api_key=self.api_key
        )

    def search(self, query: str, location: str = "", cuisine: str = "") -> List[Dict[str, Any]]:
        """
        Search for comprehensive recommendations based on query and optional filters.

        Args:
            query: The search query (e.g., "restaurants", "cocktail bars")
            location: Optional location filter
            cuisine: Optional cuisine/style type filter

        Returns:
            List of venue information dictionaries
        """
        # Detect if query is about restaurants, bars, or other venues
        query_type = self._detect_query_type(query)

        # Build a search query based on the detected type
        search_query = f"Find the best {query_type} {query}"
        if location:
            search_query += f" in {location}"
        if cuisine:
            search_query += f" {cuisine} style"

        # List recommended sources to search
        source_list = ", ".join(self.trusted_sources[:10])  # First 10 sources for brevity

        search_query += f". IMPORTANT: Search MULTIPLE sources including but not limited to {source_list} and other reputable guides. DO NOT include results from {', '.join(self.excluded_sources[:5])} or similar crowd-sourced review sites. For each place, return COMPLETE data including: name, FULL address, detailed description (50+ words), website URL (if available), source website, price range ($/$/$$), and specialties or signature offerings when possible. Return at least 8-10 places if available. Format results as a JSON array."

        try:
            # First approach: Use LangChain integration with Perplexity for better handling
            try:
                print(f"Using LangChain to search Perplexity for: {search_query}")

                # Create extra body parameters for Perplexity-specific features
                extra_params = {
                    "search_domain_filter": self.excluded_sources[:10],
                    "max_tokens": 4000,
                    "web_search_options": {"search_context_size": "high"},
                    "response_format": {"type": "json_object"}
                }

                # Invoke the LangChain Perplexity model
                system_message = f"You are a specialized researcher who gathers comprehensive information about {query_type} from multiple reputable sources. You search professional food guides, renowned critics, and respected local publications - never crowd-sourced review sites. For each place, provide complete information including name, full address, price range, detailed description, specialties, and the source of the information. Return results in JSON format as an array of objects with these fields: name, address, description, price_range, specialties, website, source. Sort results by relevance and quality."

                response = self.langchain_pplx.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": search_query}
                    ],
                    extra_body=extra_params
                )

                content = response.content
                print("LangChain Perplexity search successful")

                # Try to extract JSON from the response
                results = self._extract_json_from_response(content)

                # If we got valid results, process and return them
                if results and len(results) > 0:
                    return self._process_results(results)

                # If no valid results, fall back to direct API call
                print("No valid results from LangChain, falling back to direct API")

            except Exception as lc_error:
                print(f"LangChain Perplexity error: {lc_error}")
                print("Falling back to direct API call")

            # Direct API call as fallback
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Set up the payload for the Perplexity API request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a specialized researcher who gathers comprehensive information about {query_type} from multiple reputable sources. You search professional guides, renowned critics, and respected local publications - never crowd-sourced review sites. For each place, provide complete information including name, full address, price range, detailed description, specialties, and the source of the information. Return results in JSON format as an array of objects with these fields: name, address, description, price_range, specialties, website, source. Sort results by relevance and quality."
                    },
                    {
                        "role": "user",
                        "content": search_query
                    }
                ],
                "temperature": 0.2,  # Low temperature for factual responses
                "max_tokens": 4000,  # Increased for more comprehensive results
                "search_domain_filter": self.excluded_sources[:10],  # Exclude these domains
                "return_images": False,
                "return_related_questions": False,
                "response_format": {"type": "json_object"}
            }

            # Make the direct API request
            print(f"Sending direct search request to Perplexity for: {search_query}")
            response = requests.post(self.api_url, headers=headers, json=payload)

            # Handle the response
            if response.status_code != 200:
                print(f"Perplexity API error: {response.status_code} - {response.text}")
                return []

            response_data = response.json()
            print("Perplexity direct API response received successfully")

            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]

                # Try to extract JSON from the response
                results = self._extract_json_from_response(content)

                # Process and return the results
                return self._process_results(results)

            return []

        except Exception as e:
            print(f"Error during enhanced Perplexity search: {e}")
            return []

    def _detect_query_type(self, query: str) -> str:
        """
        Detect if the query is about restaurants, bars, or other venues.

        Args:
            query: The search query

        Returns:
            String indicating the query type ("restaurants", "bars", etc.)
        """
        query_lower = query.lower()

        if any(term in query_lower for term in ["bar", "pub", "cocktail", "drink", "speakeasy"]):
            return "bars"
        elif any(term in query_lower for term in ["cafe", "coffee", "tea", "bakery"]):
            return "cafes"
        else:
            return "restaurants"  # Default to restaurants

    def _extract_json_from_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON data from the Perplexity response text.

        Args:
            content: The text response from Perplexity

        Returns:
            List of restaurant dictionaries
        """
        try:
            # Look for JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try to find JSON with curly braces
                json_match = re.search(r'\{.*\}', content, re.DOTALL)

                if json_match:
                    json_str = json_match.group(0)
                    # Check if it's a single object or an array of objects
                    if '"name"' in json_str:
                        return [json.loads(json_str)]
                    else:
                        # Try wrapping in array brackets
                        try:
                            return json.loads("[" + json_str + "]")
                        except:
                            print("JSON parsing error with brackets")
                            return [{
                                "name": "Restaurant information",
                                "description": content,
                                "address": "",
                                "price_range": "",
                                "recommended_dish": "",
                                "website": "",
                                "source": "Perplexity AI"
                            }]
                else:
                    print("No JSON format found in response, creating simplified structure")
                    return [{
                        "name": "Restaurant information",
                        "description": content,
                        "address": "",
                        "price_range": "",
                        "recommended_dish": "",
                        "website": "",
                        "source": "Perplexity AI"
                    }]
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from Perplexity response: {e}")
            return [{
                "name": "Restaurant information",
                "description": content,
                "address": "",
                "price_range": "",
                "recommended_dish": "",
                "website": "",
                "source": "Perplexity AI"
            }]

    def follow_up_search(self, venue_name: str, location: str) -> Dict[str, Any]:
        """
        Perform a follow-up search to get more details about a specific venue.

        Args:
            venue_name: Name of the venue (restaurant/bar/cafe) to search for
            location: Location of the venue

        Returns:
            Dictionary with additional venue details
        """
        # Detect venue type (restaurant, bar, cafe)
        venue_type = self._detect_venue_type(venue_name)

        search_query = f"Find detailed information about {venue_type} '{venue_name}' in {location}. Include exact address, price range, signature offerings, opening hours, reservation details, and chef/owner information if available. Format as JSON."

        try:
            # First try using LangChain integration
            try:
                print(f"Using LangChain for follow-up search on: {venue_name}")

                # Create extra body parameters for Perplexity-specific features
                extra_params = {
                    "search_domain_filter": self.excluded_sources[:10],
                    "max_tokens": 4000,
                    "web_search_options": {"search_context_size": "high"},
                    "response_format": {}  # Empty object as per the latest Perplexity API documentation
                }

                # System prompt for follow-up details
                system_message = f"You are a detail specialist who finds specific information about {venue_type}s. Search for precise details and return structured data in JSON format with these fields: address, price_range, chef_or_owner, signature_offerings (array), opening_hours, reservation_info, website."

                # Invoke the LangChain Perplexity model
                response = self.langchain_pplx.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": search_query}
                    ],
                    extra_body=extra_params
                )

                content = response.content
                print("LangChain follow-up search successful")

                # Try to extract JSON from the response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(0)
                        result = json.loads(json_str)
                        if result and isinstance(result, dict) and len(result) > 1:
                            return result
                except Exception:
                    pass

                # If we're here, either extraction failed or result was invalid
                print("Could not extract valid JSON from LangChain response, trying direct API")

            except Exception as lc_error:
                print(f"LangChain follow-up error: {lc_error}")
                print("Falling back to direct API for follow-up search")

            # Fall back to direct API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Set up the payload for the Perplexity API request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are a specialized researcher who gathers comprehensive information about {query_type}s from multiple reputable sources. You search professional guides, renowned critics, and respected local publications - never crowd-sourced review sites. For each place, provide complete information including name, full address, price range, detailed description, specialties, and the source of the information. Return results in JSON format as an array of objects with these fields: name, address, description, price_range, specialties, website, source. Sort results by relevance and quality."
                    },
                    {
                        "role": "user",
                        "content": search_query
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 4000,
                "search_domain_filter": self.excluded_sources[:10],
                "return_images": False,
                "return_related_questions": False,
                "response_format": {}  # Empty object as per the latest Perplexity API documentation
            }

            # Make the API request
            print(f"Sending direct follow-up search to Perplexity for: {venue_name}")
            response = requests.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Follow-up search API error: {response.status_code} - {response.text}")
                return {}

            response_data = response.json()

            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]

                # Try to extract JSON from the response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(0)
                        return json.loads(json_str)
                    else:
                        # Return a basic structure with the full text
                        return {"details": content}
                except json.JSONDecodeError:
                    return {"details": content}

            return {}

        except Exception as e:
            print(f"Error during follow-up search: {e}")
            return {}

    def _detect_venue_type(self, venue_name: str) -> str:
        """
        Detect the type of venue based on its name.

        Args:
            venue_name: Name of the venue

        Returns:
            String indicating venue type (restaurant, bar, cafe)
        """
        venue_lower = venue_name.lower()

        if any(term in venue_lower for term in ["bar", "pub", "lounge", "tavern", "brewery", "speakeasy"]):
            return "bar"
        elif any(term in venue_lower for term in ["cafe", "coffee", "bakery", "patisserie", "tea house"]):
            return "cafe"
        else:
            return "restaurant"  # Default to restaurant

    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process search results into a structured format for restaurant information.

        Args:
            results: Raw search results from Perplexity

        Returns:
            Structured restaurant information
        """
        processed_results = []

        for result in results[:self.max_results]:  # Limit by max_results
            # Ensure the result is a dictionary
            if not isinstance(result, dict):
                print(f"Skipping non-dictionary result: {result}")
                continue

            # Create structured restaurant information
            restaurant_info = {
                "name": result.get("name", "Unknown Restaurant"),
                "description": result.get("description", "No description available"),
                "address": result.get("address", ""),
                "price_range": result.get("price_range", ""),
                "recommended_dish": result.get("recommended_dish", ""),
                "website": result.get("website", result.get("url", "")),
                "source": result.get("source", "Perplexity AI"),
                "score": 0.9,  # Default high score for Perplexity results
            }

            processed_results.append(restaurant_info)

        return processed_results

# Example usage
if __name__ == "__main__":
    # For testing
    if "PERPLEXITY_API_KEY" not in os.environ:
        print("Warning: PERPLEXITY_API_KEY environment variable not set")
        print("Please set your Perplexity API key to test")
    else:
        search_agent = EnhancedPerplexitySearchAgent()
        results = search_agent.search("best sushi", "Tokyo")

        print("\nSearch Results:")
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. {result['name']}")
            print(f"Address: {result.get('address', 'Not available')}")
            print(f"Price: {result.get('price_range', 'Not available')}")
            if result.get('recommended_dish'):
                print(f"Signature dish: {result['recommended_dish']}")
            print(f"Source: {result['source']}")
            print(f"Description: {result['description'][:150]}...")