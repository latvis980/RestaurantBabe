import requests
from langchain_core.tracers.context import tracing_v2_enabled
import json
import time
from utils.database import save_data


class BraveSearchAgent:
    def __init__(self, config):
        self.api_key = config.BRAVE_API_KEY
        self.search_count = config.BRAVE_SEARCH_COUNT
        self.excluded_domains = config.EXCLUDED_RESTAURANT_SOURCES
        self.config = config
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    #### In `search_agent.py`, add more debug output:

    ```python
    def search(self, queries, max_retries=3, retry_delay=2):
        """
        Perform searches with the given queries
        """
        all_results = []

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            for query in queries:
                retry_count = 0
                success = False

                while not success and retry_count < max_retries:
                    try:
                        # Add debug
                        print(f"[SearchAgent] Searching for: {query}")
                        results = self._execute_search(query)
                        print(f"[SearchAgent] Raw results count: {len(results.get('web', {}).get('results', []))}")

                        filtered_results = self._filter_results(results)
                        print(f"[SearchAgent] Filtered results count: {len(filtered_results)}")

                        all_results.extend(filtered_results)
                        success = True
                    except Exception as e:
                        print(f"Error in search for query '{query}': {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"Max retries reached for query '{query}'")

                    # Respect rate limits
                    time.sleep(1)

        # Add debug
        print(f"[SearchAgent] Total search results: {len(all_results)}")

        # Save results...
        return all_results

    def _execute_search(self, query):
        """Execute a single search query against Brave Search API"""
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": self.search_count,
            "freshness": "month"  # Get recent results
        }

        response = requests.get(
            self.base_url,
            headers=headers,
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Brave Search API error: {response.status_code}, {response.text}")

        return response.json()

    def _filter_results(self, search_results):
        """Filter search results to exclude unwanted domains"""
        if not search_results or "web" not in search_results or "results" not in search_results["web"]:
            return []

        filtered_results = []

        for result in search_results["web"]["results"]:
            # Skip results from excluded domains
            if not any(excluded in result.get("url", "") for excluded in self.excluded_domains):
                # Clean and extract the relevant information
                filtered_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("description", ""),
                    "language": result.get("language", "en"),
                    "favicon": result.get("favicon", "")
                }
                filtered_results.append(filtered_result)

        return filtered_results

    
    def follow_up_search(self, restaurant_name, location, additional_context=None):
        """
        Perform a follow-up search for a specific restaurant

        Args:
            restaurant_name (str): Name of the restaurant
            location (str): Location of the restaurant
            additional_context (str, optional): Additional search context

        Returns:
            dict: Search results specifically about this restaurant
        """
        # Create a specific query for this restaurant
        query = f"{restaurant_name} restaurant {location}"
        if additional_context:
            query += f" {additional_context}"

        # Search for this specific restaurant
        results = self._execute_search(query)
        filtered_results = self._filter_results(results)

        # Also check global guides
        global_guides_results = self._check_global_guides(restaurant_name, location)

        return {
            "direct_search": filtered_results,
            "global_guides": global_guides_results
        }

    def _check_global_guides(self, restaurant_name, location):
        """Check if the restaurant is mentioned in global guides"""
        global_guides = [
            "theworlds50best.com",
            "worldofmouth.app",
            "guide.michelin.com",
            "culinarybackstreets.com",
            "oadguides.com",
            "laliste.com"
        ]

        results = []

        for guide in global_guides:
            try:
                query = f"site:{guide} {restaurant_name} {location}"
                guide_results = self._execute_search(query)
                filtered_guide_results = self._filter_results(guide_results)

                if filtered_guide_results:
                    for result in filtered_guide_results:
                        result["guide"] = guide
                        results.append(result)

                # Respect rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error checking guide {guide}: {e}")

        return results