"""
Enhanced Perplexity Search Agent with multilingual support for finding restaurant recommendations.

This module connects to the Perplexity API and supports searching in local languages
for more comprehensive global restaurant information.
"""
import os
import requests
from typing import List, Dict, Any, Optional
import json
import re
from langchain_perplexity import ChatPerplexity
from langchain_openai import ChatOpenAI
import config

class EnhancedPerplexitySearchAgent:
    """
    An improved search agent that uses Perplexity API to find detailed restaurant 
    and bar recommendations from multiple reputable sources, including local publications.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        max_results: int = 10,
        model: str = "llama-3.1-sonar-small-128k-online",
    ):
        """
        Initialize the enhanced search agent with Perplexity API.

        Args:
            api_key: Perplexity API key (uses PERPLEXITY_API_KEY env var if not provided)
            openai_api_key: OpenAI API key for language detection (uses OPENAI_API_KEY env var if not provided)
            max_results: Maximum number of results to return
            model: The Perplexity model to use
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY") or config.PERPLEXITY_API_KEY
        if not self.api_key:
            raise ValueError("Perplexity API key is required")

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for language detection")

        self.max_results = max_results
        self.model = model
        self.api_url = "https://api.perplexity.ai/chat/completions"

        # Define trusted and excluded sources from config
        self.trusted_sources = config.RESTAURANT_SOURCES  
        self.excluded_sources = config.EXCLUDED_RESTAURANT_SOURCES

        # Initialize OpenAI for language detection
        self.openai = ChatOpenAI(
            api_key=self.openai_api_key,
            model=config.OPENAI_MODEL,
            temperature=0.1  # Low temperature for accurate language detection
        )

        # Initialize LangChain integration with Perplexity
        os.environ["PPLX_API_KEY"] = self.api_key
        self.langchain_pplx = ChatPerplexity(
            model=self.model,
            temperature=0.2,
            pplx_api_key=self.api_key
        )

    def search(self, query: str, location: str = "", cuisine: str = "") -> List[Dict[str, Any]]:
        """
        Search for comprehensive recommendations based on query and optional filters,
        with support for local language sources.

        Args:
            query: The search query (e.g., "restaurants", "cocktail bars")
            location: Optional location filter
            cuisine: Optional cuisine/style type filter

        Returns:
            List of venue information dictionaries
        """
        # Detect the language of the query
        query_language = self._detect_language(query)

        # Detect venue type (restaurant, bar, cafe)
        query_type = self._detect_query_type(query)

        # Build a smart search query that handles local language searching
        search_query = self._build_search_query(query, location, cuisine, query_type, query_language)

        # Log what we're doing
        print(f"Performing search in detected language: {query_language}")

        # Execute the search
        results = self._perform_search(search_query)

        return results

    def _build_search_query(self, query: str, location: str, cuisine: str, query_type: str, 
                           query_language: str = "English") -> str:
        """
        Build a search query optimized for restaurant search results.

        Args:
            query: The user's search query
            location: Location filter
            cuisine: Cuisine filter
            query_type: Type of venue (restaurant, bar, etc.)
            query_language: The detected language of the query

        Returns:
            Formatted search query string
        """
        # Build the base query
        base_query = f"Find the best {query_type} {query}"
        if location:
            base_query += f" in {location}"
        if cuisine:
            base_query += f" {cuisine} style"

        # List some recommended sources to search as examples
        source_list = ", ".join(self.trusted_sources[:5])  # First 5 sources for brevity
        excluded_list = ", ".join(self.excluded_sources[:5])

        # Create a search query that leverages Perplexity's AI capabilities
        search_query = base_query + "\n\n"

        # Add specific instructions for local language searching
        if query_language != "English":
            search_query += f"IMPORTANT: I notice this query is in {query_language}. "
            search_query += f"Search for reputable LOCAL sources in {query_language} AND international sources in English. "
            search_query += f"Include local food blogs, newspapers, and magazines from {location if location else 'the relevant region'}. "
            search_query += f"Prioritize sources that locals would trust and use.\n\n"

        # Add general instructions for all searches
        search_query += f"- Search MULTIPLE reputable sources like {source_list} and other respected local guides and publications.\n"
        search_query += f"- DO NOT include results from {excluded_list} or any crowd-sourced review sites.\n"
        search_query += f"- For each place, return COMPLETE data including: name, full address, detailed description (50+ words), "
        search_query += f"website URL, source website, price range ($/$$/$$$), and signature offerings.\n"
        search_query += f"- Return at least 8-10 places if available. Format results as a JSON array.\n"

        if location:
            search_query += f"- For {location}, include information from local experts and publications whenever possible.\n"

        search_query += f"- If this search is for a non-English speaking location, include recommendations from respected LOCAL sources in the local language.\n"

        return search_query

    def _perform_search(self, search_query: str) -> List[Dict[str, Any]]:
        """
        Perform the actual search using Perplexity API.

        Args:
            search_query: The search query string

        Returns:
            List of search results
        """
        try:
            # First approach: Use LangChain integration with Perplexity for better handling
            try:
                print(f"Using LangChain to search Perplexity for: {search_query}")

                # Create extra body parameters for Perplexity-specific features
                extra_params = {
                    "search_domain_filter": self.excluded_sources[:10],
                    "max_tokens": 4000,
                    "web_search_options": {"search_context_size": "high"},
                    "response_format": {}  # Empty object per Perplexity API
                }

                # Invoke the LangChain Perplexity model with the search query
                # Using a minimal system prompt to let Perplexity's AI handle the details
                system_message = "You are a restaurant recommendation specialist who finds information from reputable sources. Return results as a JSON array. Each restaurant should include name, address, description, price_range, recommended_dishes (array), website, and source."

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
                        "content": "You are a restaurant recommendation specialist who finds information from reputable sources. Return results as a JSON array. Each restaurant should include name, address, description, price_range, recommended_dishes (array), website, and source."
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
                "return_related_questions": False
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

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the provided text using OpenAI.

        Args:
            text: The text to detect language from

        Returns:
            Detected language name (e.g., "English", "French", etc.)
        """
        if not text or len(text.strip()) == 0:
            return "English"

        try:
            # Simple language detection using OpenAI
            response = self.openai.invoke(
                [
                    {"role": "system", "content": "You are a language detection specialist. Identify the language of the given text and respond with only the language name in English (e.g., 'English', 'French', 'Spanish', 'German', etc.). Do not include any other information or explanation."},
                    {"role": "user", "content": f"Detect the language of this text: '{text}'"}
                ]
            )

            language = response.content.strip()
            # Clean up any extra text that might be included
            language = language.split("\n")[0].strip()
            # Handle cases where the model might include "The language is" or similar
            if language.lower().startswith("the language is"):
                language = language.split("is")[-1].strip()

            return language if language else "English"
        except Exception as e:
            print(f"Language detection error: {e}")
            return "English"

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

    def follow_up_search(self, restaurant_name: str, location: str = "") -> Dict[str, Any]:
        """
        Perform a follow-up search to get more details about a specific venue,
        with support for local language searches.

        Args:
            restaurant_name: Name of the venue (restaurant/bar/cafe) to search for
            location: Location of the venue

        Returns:
            Dictionary with additional venue details
        """
        # REMOVE: Detect venue type (restaurant, bar, cafe)
        # venue_type = self._detect_venue_type(restaurant_name)

        # Detect location language if provided
        location_language = self._detect_language(location) if location else "English"

        # Create search query with instructions for local language results if needed
        # CHANGE: use "restaurant" instead of venue_type variable
        search_query = f"Find detailed information about restaurant '{restaurant_name}'"
        if location:
            search_query += f" in {location}"

        # Add language instruction if it's a non-English location
        if location_language != "English":
            search_query += f". Search in both {location_language} AND English language sources for more authentic local information."

        search_query += ". Include exact address, price range, signature offerings, opening hours, reservation details, and chef/owner information if available. Format as JSON."

        try:
            # First try using LangChain integration
            try:
                print(f"Using LangChain for follow-up search on: {restaurant_name}")

                # Create extra body parameters for Perplexity-specific features
                extra_params = {
                    "search_domain_filter": self.excluded_sources[:10],
                    "max_tokens": 4000,
                    "web_search_options": {"search_context_size": "high"}
                }

                # System prompt for follow-up details
                # CHANGE: use "restaurants" instead of venue_type variable
                system_message = "You are a detail specialist who finds specific information about restaurants. Search for precise details and return structured data in JSON format with these fields: address, price_range, chef_or_owner, signature_offerings (array), opening_hours, reservation_info, website."

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
                        "content": "You are a detail specialist who finds specific information about food venues. Search for precise details and return structured data in JSON format with these fields: address, price_range, chef_or_owner, signature_offerings (array), opening_hours, reservation_info, website."
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
            print(f"Sending direct follow-up search to Perplexity for: {restaurant_name}")
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