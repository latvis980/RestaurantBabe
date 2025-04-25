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

    def perform_follow_up_searches(self, formatted_recommendations, follow_up_queries, secondary_filter_parameters=None):
        """
        Perform follow-up searches for each restaurant to gather missing information

        Args:
            formatted_recommendations (dict): The formatted recommendations
            follow_up_queries (list): List of follow-up queries for each restaurant
            secondary_filter_parameters (list, optional): Secondary parameters from query analysis for targeted searches

        Returns:
            dict: Enhanced recommendations with additional information
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            enhanced_recommendations = {
                "main_list": [],
                "hidden_gems": []
            }

            # Process main_list restaurants
            for restaurant in formatted_recommendations.get("main_list", []):
                enhanced_restaurant = self._enhance_restaurant(restaurant, follow_up_queries, secondary_filter_parameters)
                enhanced_recommendations["main_list"].append(enhanced_restaurant)

            # Process hidden gems
            for restaurant in formatted_recommendations.get("hidden_gems", []):
                enhanced_restaurant = self._enhance_restaurant(restaurant, follow_up_queries, secondary_filter_parameters)
                enhanced_recommendations["hidden_gems"].append(enhanced_restaurant)

            return enhanced_recommendations

    def _enhance_restaurant(self, restaurant, follow_up_queries, secondary_filter_parameters=None):
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

        # Add secondary filter parameters for targeted searches
        if secondary_filter_parameters:
            for param in secondary_filter_parameters:
                restaurant_queries.append(f"{restaurant_name} restaurant {restaurant_location} {param}")

        # Execute searches one at a time to avoid overwhelming resources
        all_search_results = []
        for query in restaurant_queries[:3]:  # Limit to first 3 queries for performance
            try:
                # Limit to 2 results per query to avoid excessive scraping
                results = self.search_agent._execute_search(query)
                filtered_results = self.search_agent._filter_results(results)[:2]

                # Scrape the results synchronously
                try:
                    scraped_results = self.scraper.scrape_search_results(filtered_results)
                    all_search_results.extend(scraped_results)
                except Exception as scrape_error:
                    print(f"Error scraping results for {restaurant_name}: {scrape_error}")

                # Be nice to servers
                time.sleep(1)
            except Exception as e:
                print(f"Error in follow-up search for {restaurant_name} with query '{query}': {e}")

        # Check global guides specifically - limit to fewer guides
        selected_guides = ["guide.michelin.com", "theworlds50best.com"]
        global_guide_info = []
        for guide in selected_guides:
            try:
                query = f"site:{guide} {restaurant_name} {restaurant_location}"
                guide_results = self.search_agent._execute_search(query)
                filtered_guide_results = self.search_agent._filter_results(guide_results)[:1]

                # Add guide information
                for result in filtered_guide_results:
                    result["guide"] = guide
                    global_guide_info.append(result)

                # Respect rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error checking guide {guide}: {e}")

        # Add the enhanced information to the restaurant
        enhanced_restaurant = restaurant.copy()
        enhanced_restaurant["additional_info"] = {
            "follow_up_results": all_search_results + global_guide_info,
            "secondary_parameters_checked": secondary_filter_parameters if secondary_filter_parameters else []
        }

        return enhanced_restaurant

    def _extract_additional_details(self, search_results):
        """Try to extract additional details from search results"""
        additional_details = {}

        # Combine all scraped content
        combined_content = ""
        for result in search_results:
            if "scraped_content" in result:
                combined_content += result["scraped_content"] + "\n\n"

        # Very basic extraction of possibly better description (first 300-400 chars)
        if combined_content and len(combined_content) > 400:
            additional_details["description"] = combined_content[:400].rstrip() + "..."

        return additional_details

    def _extract_sources(self, source_results):
        """Extract source names from source-specific search results"""
        sources = []

        for result in source_results:
            # Extract source from domain
            domain = result.get("source_domain", "")

            if domain:
                # Simplify domain name to create a source name
                source_name = domain.replace("www.", "").split(".")[0]

                # Capitalize and format properly
                source_name = " ".join(word.capitalize() for word in source_name.split("-"))
                source_name = " ".join(word.capitalize() for word in source_name.split("_"))

                # Special case handling for common domains
                if "michelin" in domain:
                    source_name = "Michelin Guide"
                elif "foodandwine" in domain:
                    source_name = "Food & Wine"
                elif "eater" in domain:
                    source_name = "Eater"
                elif "zagat" in domain:
                    source_name = "Zagat"
                elif "infatuation" in domain:
                    source_name = "The Infatuation"
                elif "50best" in domain or "worlds50best" in domain:
                    source_name = "World's 50 Best"
                elif "wordofmouth" in domain or "worldofmouth" in domain:
                    source_name = "World of Mouth"

                # Add to sources if not already there and not banned
                if (source_name not in sources and
                    not any(banned in domain for banned in ["tripadvisor", "yelp", "google"])):
                    sources.append(source_name)

            # Check if there's a guide field
            if "guide" in result:
                guide = result.get("guide", "")
                if guide:
                    # Format guide name
                    guide_name = guide.replace("www.", "").split(".")[0]
                    guide_name = " ".join(word.capitalize() for word in guide_name.split("-"))
                    guide_name = " ".join(word.capitalize() for word in guide_name.split("_"))

                    # Special case handling
                    if "theworlds50best" in guide or "50best" in guide:
                        guide_name = "World's 50 Best"
                    elif "michelin" in guide:
                        guide_name = "Michelin Guide"
                    elif "worldofmouth" in guide or "wordofmouth" in guide:
                        guide_name = "World of Mouth"

                    # Add to sources if not already there and not banned
                    if (guide_name not in sources and 
                        not any(banned in guide.lower() for banned in ["tripadvisor", "yelp", "google"])):
                        sources.append(guide_name)

        return sources

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

    def _generate_basic_html(self, restaurant):
        """Generate basic HTML formatted string for a restaurant"""
        try:
            name = restaurant.get("name", "Restaurant")
            html = f"<b>{name}</b>\n"

            if "address" in restaurant:
                html += f"📍 {restaurant['address']}\n"

            if "description" in restaurant:
                html += f"{restaurant['description']}\n"

            if "recommended_by" in restaurant and restaurant["recommended_by"]:
                sources = restaurant["recommended_by"]
                if isinstance(sources, list):
                    sources_text = ", ".join(sources[:3])
                    html += f"<i>✅ Recommended by: {sources_text}</i>"
                else:
                    html += f"<i>✅ Recommended by: {sources}</i>"

            return html
        except Exception as e:
            print(f"Error generating basic HTML: {e}")
            return f"<b>{restaurant.get('name', 'Restaurant')}</b>"