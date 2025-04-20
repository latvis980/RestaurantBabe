# agents/follow_up_search_agent.py
from langchain_core.tracers.context import tracing_v2_enabled
from agents.search_agent import BraveSearchAgent
from agents.scraper import WebScraper
import time

class FollowUpSearchAgent:
    def __init__(self, config):
        self.config = config
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)

    def perform_follow_up_searches(self, formatted_recommendations, follow_up_queries):
        """
        Perform follow-up searches for each restaurant to gather missing information

        Args:
            formatted_recommendations (dict): The formatted recommendations
            follow_up_queries (list): List of follow-up queries for each restaurant

        Returns:
            dict: Enhanced recommendations with additional information
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            enhanced_recommendations = {
                "recommended": [],
                "hidden_gems": []
            }

            # Process recommended restaurants
            for restaurant in formatted_recommendations.get("recommended", []):
                enhanced_restaurant = self._enhance_restaurant(restaurant, follow_up_queries)
                enhanced_recommendations["recommended"].append(enhanced_restaurant)

            # Process hidden gems
            for restaurant in formatted_recommendations.get("hidden_gems", []):
                enhanced_restaurant = self._enhance_restaurant(restaurant, follow_up_queries)
                enhanced_recommendations["hidden_gems"].append(enhanced_restaurant)

            return enhanced_recommendations

    def _enhance_restaurant(self, restaurant, follow_up_queries):
        """Enhance a single restaurant with additional information from follow-up searches"""
        restaurant_name = restaurant.get("name", "")
        restaurant_location = restaurant.get("address", "").split(",")[0] if restaurant.get("address") else ""

        # Find queries for this restaurant
        restaurant_queries = []
        for query_set in follow_up_queries:
            if query_set.get("restaurant_name") == restaurant_name:
                restaurant_queries = query_set.get("queries", [])
                break

        if not restaurant_queries:
            # No specific queries found, use default
            restaurant_queries = [
                f"{restaurant_name} restaurant {restaurant_location} hours prices",
                f"{restaurant_name} restaurant {restaurant_location} menu dishes",
                f"{restaurant_name} restaurant {restaurant_location} chef reservations"
            ]

        # Check for missing information
        missing_info = restaurant.get("missing_info", [])
        if missing_info:
            # Add specific queries for missing information
            for info in missing_info:
                restaurant_queries.append(f"{restaurant_name} restaurant {restaurant_location} {info}")

        # Check global guides specifically
        global_guide_info = self._check_global_guides(restaurant_name, restaurant_location)

        # Perform searches and gather information
        all_search_results = []
        for query in restaurant_queries:
            try:
                # Limit to 3 results per query to avoid excessive scraping
                results = self.search_agent._execute_search(query)
                filtered_results = self.search_agent._filter_results(results)[:3]

                # Scrape the results
                scraped_results = self.scraper.scrape_search_results(filtered_results)
                all_search_results.extend(scraped_results)

                # Be nice to servers
                time.sleep(1)
            except Exception as e:
                print(f"Error in follow-up search for {restaurant_name} with query '{query}': {e}")

        # Combine all results
        combined_results = all_search_results + global_guide_info

        # Add the enhanced information to the restaurant
        enhanced_restaurant = restaurant.copy()
        enhanced_restaurant["additional_info"] = {
            "follow_up_results": combined_results
        }

        return enhanced_restaurant

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
                guide_results = self.search_agent._execute_search(query)
                filtered_guide_results = self.search_agent._filter_results(guide_results)

                # Add guide information
                for result in filtered_guide_results:
                    result["guide"] = guide
                    results.append(result)

                # Respect rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error checking guide {guide}: {e}")

        return results