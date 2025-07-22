# agents/simplified_follow_up_search_agent.py

import logging
import googlemaps
from typing import Dict, List, Any, Optional
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class SimplifiedFollowUpSearchAgent:
    """
    Simplified follow-up search agent that handles:
    1. Address verification using Google Maps API
    2. Rating filtering and restaurant rejection based on Google ratings
    """

    def __init__(self, config):
        self.config = config

        # Rating threshold for restaurant filtering
        self.min_acceptable_rating = getattr(config, 'MIN_ACCEPTABLE_RATING', 4.1)

        # Initialize Google Maps client
        api_key = getattr(config, 'GOOGLE_MAPS_API_KEY', None)
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY is required in config")

        self.gmaps = googlemaps.Client(key=api_key)

        # Fields we want from Google Places API (updated to include rating)
        self.place_fields = [
            "formatted_address",
            "geometry",
            "place_id",
            "url",
            "rating",
            "user_ratings_total"
        ]

    @log_function_call
    def perform_follow_up_searches(
        self,
        formatted_recommendations: Dict[str, List[Dict[str, Any]]],
        follow_up_queries: List[Dict[str, Any]] = None,  # Not used but kept for compatibility
        destination: str = "Unknown",
        secondary_filter_parameters: Optional[List[str]] = None  # Not used but kept for compatibility
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Verifies addresses and filters restaurants based on Google ratings.
        Restaurants below the minimum rating threshold are removed from the list.

        Args:
            formatted_recommendations: Restaurant data with main_list
            follow_up_queries: Ignored (kept for compatibility)
            destination: City from original query
            secondary_filter_parameters: Ignored (kept for compatibility)

        Returns:
            Dict with updated main_list containing only restaurants that pass rating filter
        """

        main_list = formatted_recommendations.get("main_list", [])

        logger.info(f"Starting address verification and rating filtering for {len(main_list)} restaurants in {destination}")
        logger.info(f"Minimum acceptable rating: {self.min_acceptable_rating}")

        # Process each restaurant
        verified_restaurants = []
        rejected_count = 0

        for restaurant in main_list:
            result = self._verify_and_filter_restaurant(restaurant, destination)

            if result is None:
                # Restaurant was rejected due to low rating
                rejected_count += 1
                logger.info(f"❌ Rejected: {restaurant.get('name', 'Unknown')} (low rating)")
            else:
                verified_restaurants.append(result)

        final_result = {"main_list": verified_restaurants}

        logger.info(f"✅ Address verification and filtering complete for {destination}")
        logger.info(f"   - Original count: {len(main_list)}")
        logger.info(f"   - Passed filter: {len(verified_restaurants)}")
        logger.info(f"   - Rejected: {rejected_count}")

        # Log overall statistics
        dump_chain_state("follow_up_search_complete", {
            "destination": destination,
            "original_count": len(main_list),
            "final_count": len(verified_restaurants),
            "rejected_count": rejected_count,
            "min_rating": self.min_acceptable_rating
        })

        return final_result

    def _verify_and_filter_restaurant(self, restaurant: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """
        Verify address and filter restaurant based on Google rating.

        Args:
            restaurant: Restaurant data
            destination: City from original query

        Returns:
            Updated restaurant data if it passes the rating filter, None if rejected
        """

        # Make a copy to avoid modifying the original
        updated_restaurant = restaurant.copy()

        restaurant_name = restaurant.get("name", "")
        if not restaurant_name:
            logger.warning("Restaurant missing name, cannot verify")
            return updated_restaurant

        logger.info(f"Verifying: {restaurant_name} in {destination}")

        # Search Google Maps for restaurant info
        maps_info = self._search_google_maps(restaurant_name, destination)

        if maps_info:
            # Check rating first - reject if below threshold
            rating = maps_info.get("rating")
            if rating is not None:
                rating = float(rating)
                updated_restaurant["rating"] = rating
                updated_restaurant["user_ratings_total"] = maps_info.get("user_ratings_total", 0)

                if rating < self.min_acceptable_rating:
                    # Log rejection details
                    dump_chain_state("restaurant_rejected_low_rating", {
                        "restaurant": restaurant_name,
                        "destination": destination,
                        "rating": rating,
                        "min_required": self.min_acceptable_rating,
                        "user_ratings_total": maps_info.get("user_ratings_total", 0)
                    })

                    logger.warning(f"❌ {restaurant_name} rejected: rating {rating} < {self.min_acceptable_rating}")
                    return None  # Reject this restaurant

                logger.info(f"✅ {restaurant_name} passed rating filter: {rating}")

            # Update address if it was requiring verification
            address = restaurant.get("address", "")
            if address == "Requires verification" and maps_info.get("formatted_address"):
                updated_restaurant["address"] = maps_info["formatted_address"]

                # Remove "address" from missing_info if present
                missing_info = updated_restaurant.get("missing_info", [])
                if "address" in missing_info:
                    missing_info = [info for info in missing_info if info != "address"]
                    updated_restaurant["missing_info"] = missing_info

                logger.info(f"✅ Address verified for {restaurant_name}: {maps_info['formatted_address']}")

                # Log successful address verification
                dump_chain_state("address_verified", {
                    "restaurant": restaurant_name,
                    "destination": destination,
                    "verified_address": maps_info["formatted_address"],
                    "rating": rating if rating else "N/A"
                })

            # Add Google Maps URL if available
            if maps_info.get("url"):
                updated_restaurant["google_maps_url"] = maps_info["url"]

        else:
            # No Maps data found - keep restaurant but log it
            logger.warning(f"⚠️ No Google Maps data found for {restaurant_name} in {destination}")
            dump_chain_state("google_maps_no_data", {
                "restaurant": restaurant_name,
                "destination": destination
            })

        return updated_restaurant

    def _search_google_maps(self, restaurant_name: str, city: str) -> Optional[Dict[str, Any]]:
        """
        Search Google Maps for restaurant info including address and rating.

        Args:
            restaurant_name: Name of the restaurant
            city: City where restaurant is located

        Returns:
            Dict with restaurant info or None if not found
        """

        try:
            # Create search query: restaurant name + city
            search_query = f"{restaurant_name} restaurant {city}"

            logger.debug(f"Google Maps search query: {search_query}")

            # Perform text search
            search_response = self.gmaps.places(query=search_query)

            results = search_response.get("results", [])
            if not results:
                logger.debug(f"No Google Maps results for: {search_query}")
                return None

            # Get the first result (most relevant)
            first_result = results[0]
            place_id = first_result.get("place_id")

            if not place_id:
                logger.debug(f"No place_id in first result for: {search_query}")
                return None

            # Get detailed place information including rating
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=self.place_fields
            )

            result_data = place_details.get("result", {})

            formatted_address = result_data.get("formatted_address")
            rating = result_data.get("rating")
            user_ratings_total = result_data.get("user_ratings_total")

            return {
                "formatted_address": formatted_address,
                "rating": rating,
                "user_ratings_total": user_ratings_total,
                "place_id": place_id,
                "url": result_data.get("url", f"https://maps.google.com/maps/place/?q=place_id:{place_id}"),
                "geometry": result_data.get("geometry", {})
            }

        except googlemaps.exceptions.ApiError as e:
            logger.error(f"Google Maps API error for {restaurant_name} in {city}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error searching Google Maps for {restaurant_name} in {city}: {e}")
            return None

# Compatibility wrapper to maintain existing interface
class FollowUpSearchAgent(SimplifiedFollowUpSearchAgent):
    """
    Compatibility wrapper for existing orchestrator code.
    Inherits from SimplifiedFollowUpSearchAgent to maintain the same interface.
    """
    pass