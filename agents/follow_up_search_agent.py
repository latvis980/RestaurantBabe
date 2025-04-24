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
        Perform follow-up searches for each restaurant to gather additional information

        Args:
            formatted_recommendations (dict): The formatted recommendations
            follow_up_queries (list): List of follow-up queries for each restaurant
            secondary_filter_parameters (list, optional): Secondary parameters for targeted searches

        Returns:
            dict: Enhanced recommendations with additional information
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Debug the structure of the input
            print(f"Follow-up search received structure: {list(formatted_recommendations.keys() if isinstance(formatted_recommendations, dict) else [])}")

            # Ensure we have the correct structure
            if not formatted_recommendations:
                formatted_recommendations = {"main_list": [], "hidden_gems": []}

            if not isinstance(formatted_recommendations, dict):
                formatted_recommendations = {"main_list": [], "hidden_gems": []}

            # Handle both old "recommended" and new "main_list" structures
            if "main_list" not in formatted_recommendations and "recommended" in formatted_recommendations:
                formatted_recommendations["main_list"] = formatted_recommendations["recommended"]
                del formatted_recommendations["recommended"]
            elif "main_list" not in formatted_recommendations:
                formatted_recommendations["main_list"] = []

            if "hidden_gems" not in formatted_recommendations:
                formatted_recommendations["hidden_gems"] = []

            enhanced_recommendations = {
                "main_list": [],
                "hidden_gems": []
            }

            # Process main list restaurants
            for restaurant in formatted_recommendations.get("main_list", []):
                enhanced_restaurant = self._enhance_restaurant(restaurant, follow_up_queries)
                enhanced_recommendations["main_list"].append(enhanced_restaurant)

            # Process hidden gems
            for restaurant in formatted_recommendations.get("hidden_gems", []):
                enhanced_restaurant = self._enhance_restaurant(restaurant, follow_up_queries)
                enhanced_recommendations["hidden_gems"].append(enhanced_restaurant)

            return enhanced_recommendations

    def _enhance_restaurant(self, restaurant, follow_up_queries):
        """Enhance a single restaurant with additional information from follow-up searches"""
        restaurant_name = restaurant.get("name", "")
        restaurant_city = restaurant.get("city", "")
        restaurant_location = restaurant.get("address", "").split(",")[0] if restaurant.get("address") else ""

        # Find queries for this restaurant
        restaurant_queries = []
        for query_set in follow_up_queries:
            if query_set.get("restaurant_name") == restaurant_name:
                restaurant_queries = query_set.get("queries", [])
                break

        if not restaurant_queries:
            # No specific queries found, use default with source queries
            # Include city if available
            city_part = f" {restaurant_city}" if restaurant_city else ""

            restaurant_queries = [
                f"{restaurant_name} restaurant{city_part} hours",
                f"{restaurant_name} restaurant{city_part} menu",
                f"{restaurant_name} restaurant{city_part} reviews",
                # Add source-specific queries
                f"{restaurant_name} restaurant{city_part} michelin guide",
                f"{restaurant_name} restaurant{city_part} food critic",
                f"{restaurant_name} restaurant{city_part} culinary award"
            ]

        # Perform searches and gather information (use up to 3 queries including at least 1 source query)
        all_search_results = []
        source_results = []

        # Find source-related queries
        source_queries = [q for q in restaurant_queries if any(term in q.lower() for term in 
                         ["guide", "critic", "blog", "award", "chef", "magazine", "publication", "michelin", "50best", "word of mouth"])]

        # Make sure at least one source query is included
        search_queries = restaurant_queries[:2]  # Get first 2 regular queries
        if source_queries:
            search_queries.append(source_queries[0])  # Add 1 source query

        # Execute the selected queries
        for query in search_queries[:3]:  # Limit to 3 queries max
            try:
                # Limit to 2 results per query to avoid excessive scraping
                results = self.search_agent._execute_search(query)
                filtered_results = self.search_agent._filter_results(results)[:2]

                # Scrape the results
                scraped_results = self.scraper.scrape_search_results(filtered_results)

                # If this is a source query, add to source results
                if any(term in query.lower() for term in 
                      ["guide", "critic", "blog", "award", "chef", "magazine", "publication", "michelin", "50best", "word of mouth"]):
                    source_results.extend(scraped_results)

                all_search_results.extend(scraped_results)

                # Be nice to servers
                time.sleep(1)
            except Exception as e:
                print(f"Error in follow-up search for {restaurant_name} with query '{query}': {e}")

        # Also check global guides specifically
        global_guide_results = self._check_global_guides(restaurant_name, restaurant_city or restaurant_location)
        source_results.extend(global_guide_results)
        all_search_results.extend(global_guide_results)

        # Add any additional info we found
        enhanced_restaurant = restaurant.copy()
        if all_search_results:
            # Extract any additional details we might find
            additional_details = self._extract_additional_details(all_search_results)

            # Update the restaurant with additional information if found
            if "description" in additional_details and len(additional_details["description"]) > len(restaurant.get("description", "")):
                enhanced_restaurant["description"] = additional_details["description"]

            # Extract sources and update recommended_by field
            extracted_sources = self._extract_sources(source_results)
            if extracted_sources:
                # If we already have recommended_by, append new sources
                if "recommended_by" in enhanced_restaurant and enhanced_restaurant["recommended_by"]:
                    existing_sources = enhanced_restaurant["recommended_by"]
                    if isinstance(existing_sources, list):
                        # Add new sources, avoid duplicates
                        for source in extracted_sources:
                            if source not in existing_sources:
                                existing_sources.append(source)
                        enhanced_restaurant["recommended_by"] = existing_sources
                    else:
                        enhanced_restaurant["recommended_by"] = extracted_sources
                else:
                    enhanced_restaurant["recommended_by"] = extracted_sources

            # Store the full follow-up results
            enhanced_restaurant["additional_info"] = all_search_results

            # Update HTML formatting if needed
            if "html_formatted" in enhanced_restaurant:
                # We'll leave the existing formatting, but future improvements could update it with new info
                pass
            else:
                # Generate basic HTML formatting
                enhanced_restaurant["html_formatted"] = self._generate_basic_html(enhanced_restaurant)

        # Ensure we always have a recommended_by field
        if "recommended_by" not in enhanced_restaurant or not enhanced_restaurant["recommended_by"]:
            # Convert sources array to recommended_by if it exists
            if "sources" in enhanced_restaurant and enhanced_restaurant["sources"]:
                enhanced_restaurant["recommended_by"] = enhanced_restaurant["sources"]
            else:
                # Default fallback sources
                enhanced_restaurant["recommended_by"] = ["Local Food Guide", "Culinary Expert"]

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
                elif "foodand" in domain:
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
                # Include location in search if available
                location_part = f" {location}" if location else ""
                query = f"site:{guide} {restaurant_name}{location_part}"

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