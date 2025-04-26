# agents/follow_up_search_agent.py
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers import ConsoleCallbackHandler
from agents.search_agent import BraveSearchAgent
from agents.scraper import WebScraper
import time
from utils.debug_utils import dump_chain_state

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
            # Debug log the start of follow-up searches
            dump_chain_state("follow_up_search_start", {
                "restaurant_count": len(formatted_recommendations.get("main_list", [])) + 
                                    len(formatted_recommendations.get("hidden_gems", [])),
                "follow_up_queries_count": len(follow_up_queries),
                "secondary_parameters": secondary_filter_parameters
            })

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

            # Debug log completion of follow-up searches
            dump_chain_state("follow_up_search_complete", {
                "enhanced_main_list_count": len(enhanced_recommendations["main_list"]),
                "enhanced_hidden_gems_count": len(enhanced_recommendations["hidden_gems"])
            })

            return enhanced_recommendations

    # Modified _enhance_restaurant method for FollowUpSearchAgent

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
            # No specific queries found, use default for mandatory fields only
            default_queries = []

            # Check what mandatory information is missing
            address = restaurant.get("address", "")
            price_range = restaurant.get("price_range", "")
            recommended_dishes = restaurant.get("recommended_dishes", [])
            sources = restaurant.get("sources", [])

            # Only create queries for missing mandatory information
            if not address or address == "Address unavailable":
                default_queries.append(f"{restaurant_name} restaurant {restaurant_location} address location")

            if not price_range:
                default_queries.append(f"{restaurant_name} restaurant {restaurant_location} price range cost")

            if not recommended_dishes or len(recommended_dishes) < 2:
                default_queries.append(f"{restaurant_name} restaurant {restaurant_location} signature dishes menu specialties")

            if not sources or len(sources) < 2:
                default_queries.append(f"{restaurant_name} restaurant {restaurant_location} reviews recommended by")

            # Always check global guides
            default_queries.append(f"{restaurant_name} restaurant {restaurant_location} michelin guide awards")

            # Use these default queries (limited to 3)
            restaurant_queries = default_queries[:3]

            # Log that we're using default queries
            print(f"Using default queries for {restaurant_name}: {restaurant_queries}")

        # Check for missing information explicitly marked
        missing_info = restaurant.get("missing_info", [])
        if missing_info:
            # Add specific queries for missing information
            for info in missing_info:
                restaurant_queries.append(f"{restaurant_name} restaurant {restaurant_location} {info}")

        # Add secondary filter parameters for targeted searches
        if secondary_filter_parameters:
            for param in secondary_filter_parameters:
                restaurant_queries.append(f"{restaurant_name} restaurant {restaurant_location} {param}")

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
            "follow_up_results": combined_results,
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

        # Try to extract opening hours if mentioned
        if "hours" not in additional_details and ("opening hours" in combined_content.lower() or 
                                                 "open from" in combined_content.lower()):
            # This is a simplified extraction - in a real app, you'd use more sophisticated NLP
            low_content = combined_content.lower()
            hour_indicators = ["opening hours", "open from", "open:", "hours:", "we are open"]

            for indicator in hour_indicators:
                if indicator in low_content:
                    # Get text after the indicator
                    pos = low_content.find(indicator) + len(indicator)
                    hours_text = combined_content[pos:pos+100]  # Get about 100 chars after

                    # Try to find a sentence boundary to end
                    sentence_end = hours_text.find('.')
                    if sentence_end > 0:
                        hours_text = hours_text[:sentence_end+1]

                    additional_details["hours"] = hours_text.strip()
                    break

        # Try to extract price information
        price_indicators = ["price", "cost", "menu is", "dishes from", "dishes cost", "€", "$", "£"]
        if "price_range" not in additional_details:
            for indicator in price_indicators:
                if indicator in combined_content.lower():
                    pos = combined_content.lower().find(indicator)
                    price_text = combined_content[max(0, pos-20):pos+80]  # Get text around the indicator

                    # Simplify for now - in a real app, you'd use regex patterns
                    additional_details["price_info"] = price_text.strip()
                    break

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
                # Log each guide check explicitly
                dump_chain_state("checking_global_guide", {
                    "restaurant_name": restaurant_name,
                    "guide": guide
                })

                query = f"site:{guide} {restaurant_name} {location}"
                guide_results = self.search_agent._execute_search(query)
                filtered_guide_results = self.search_agent._filter_results(guide_results)

                # Log guide search results
                dump_chain_state("global_guide_results", {
                    "restaurant_name": restaurant_name,
                    "guide": guide,
                    "results_count": len(filtered_guide_results)
                })

                # Add guide information
                for result in filtered_guide_results:
                    result["guide"] = guide
                    results.append(result)

                # Respect rate limits
                time.sleep(1)
            except Exception as e:
                error_msg = f"Error checking guide {guide}: {e}"
                print(error_msg)
                # Log the error
                dump_chain_state("global_guide_error", {
                    "restaurant_name": restaurant_name,
                    "guide": guide,
                    "error": str(e)
                }, error=e)

        return results