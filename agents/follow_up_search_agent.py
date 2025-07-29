# agents/follow_up_search_agent.py - Updated to check ALL addresses and filter closed restaurants

import logging
import googlemaps
from typing import Dict, List, Any, Optional
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class FollowUpSearchAgent:
    """
    Follow-up search agent that handles:
    1. Address verification for ALL restaurants using Google Maps API
    2. Rating filtering and restaurant rejection based on Google ratings
    3. Filtering out closed restaurants (temporarily or permanently)
    4. Saving coordinates back to database
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

        # Fields we want from Google Places API (updated to include business status)
        self.place_fields = [
            "formatted_address",
            "geometry",
            "place_id",
            "url",
            "rating",
            "user_ratings_total",
            "business_status",  # PRIMARY: To check if restaurant is closed (RECOMMENDED)
            "opening_hours"     # Additional info about operating hours
            # NOTE: permanently_closed is deprecated (May 2021) but still works
            # We use business_status as it's the current recommended approach
        ]

    @log_function_call
    def enhance(self, edited_results, follow_up_queries=None, destination="Unknown"):
        """
        Main method called by orchestrator - backward compatibility wrapper
        """
        return self.perform_follow_up_searches(
            edited_results=edited_results,
            follow_up_queries=follow_up_queries,
            destination=destination
        )

    @log_function_call
    def perform_follow_up_searches(
        self,
        edited_results: Dict[str, List[Dict[str, Any]]],
        follow_up_queries: List[Dict[str, Any]] = None,  # Not used but kept for compatibility
        destination: str = "Unknown",
        secondary_filter_parameters: Optional[List[str]] = None  # Not used but kept for compatibility
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Verifies addresses for ALL restaurants and filters based on:
        1. Google ratings (below minimum threshold)
        2. Business status (closed restaurants)
        """

        main_list = edited_results.get("main_list", [])

        logger.info(f"Starting comprehensive verification for {len(main_list)} restaurants in {destination}")
        logger.info(f"Minimum acceptable rating: {self.min_acceptable_rating}")

        # Process each restaurant
        verified_restaurants = []
        rejected_count = 0
        closed_count = 0
        coordinates_saved_count = 0

        for restaurant in main_list:
            result = self._verify_and_filter_restaurant(restaurant, destination)

            if result is None:
                # Restaurant was rejected due to low rating or closure
                rejected_count += 1
                logger.info(f"❌ Rejected: {restaurant.get('name', 'Unknown')}")
            elif result.get("is_closed", False):
                # Restaurant is closed
                closed_count += 1
                logger.info(f"🚫 Closed: {restaurant.get('name', 'Unknown')}")
            else:
                verified_restaurants.append(result)
                # Count if coordinates were added
                if result.get("coordinates_saved_to_db"):
                    coordinates_saved_count += 1

        final_result = {"enhanced_results": {"main_list": verified_restaurants}}

        logger.info(f"✅ Comprehensive verification complete for {destination}")
        logger.info(f"   - Original count: {len(main_list)}")
        logger.info(f"   - Passed filter: {len(verified_restaurants)}")
        logger.info(f"   - Rejected (low rating): {rejected_count}")
        logger.info(f"   - Rejected (closed): {closed_count}")
        logger.info(f"   - Coordinates saved to DB: {coordinates_saved_count}")

        # Log overall statistics
        dump_chain_state("follow_up_search_complete", {
            "destination": destination,
            "original_count": len(main_list),
            "final_count": len(verified_restaurants),
            "rejected_count": rejected_count,
            "closed_count": closed_count,
            "coordinates_saved_count": coordinates_saved_count,
            "min_rating": self.min_acceptable_rating
        })

        return final_result

    def _verify_and_filter_restaurant(self, restaurant: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """
        Verify address and filter restaurant based on Google rating and business status.
        NOW CHECKS ALL RESTAURANTS, not just those marked for verification.
        ALSO FILTERS OUT CLOSED RESTAURANTS.
        """
        # Make a copy to avoid modifying the original
        updated_restaurant = restaurant.copy()

        restaurant_name = restaurant.get("name", "")
        if not restaurant_name:
            logger.warning("Restaurant missing name, cannot verify")
            return updated_restaurant

        logger.info(f"Verifying: {restaurant_name} in {destination}")

        # Search Google Maps for restaurant info - FOR ALL RESTAURANTS
        maps_info = self._search_google_maps(restaurant_name, destination)

        coordinates_saved = False

        if maps_info:
            # Check if restaurant is closed first
            business_status = maps_info.get("business_status")

            # Filter out closed restaurants using current Google-recommended field
            if business_status in ["CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY"]:
                # Log closure details
                dump_chain_state("restaurant_rejected_closed", {
                    "restaurant": restaurant_name,
                    "destination": destination,
                    "business_status": business_status
                })

                logger.warning(f"🚫 {restaurant_name} rejected: {business_status}")
                updated_restaurant["is_closed"] = True
                updated_restaurant["closure_reason"] = business_status
                return None  # Reject closed restaurants

            # Check rating - reject if below threshold
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

            # UPDATE: Always update address information from Google Maps (not just verification cases)
            if maps_info.get("formatted_address"):
                old_address = restaurant.get("address", "")
                updated_restaurant["address"] = maps_info["formatted_address"]

                # Remove "address" from missing_info if present
                missing_info = updated_restaurant.get("missing_info", [])
                if "address" in missing_info:
                    missing_info = [info for info in missing_info if info != "address"]
                    updated_restaurant["missing_info"] = missing_info

                if old_address != maps_info["formatted_address"]:
                    logger.info(f"📍 Address updated for {restaurant_name}")
                    logger.debug(f"   Old: {old_address}")
                    logger.debug(f"   New: {maps_info['formatted_address']}")

            # Add coordinates for ALL restaurants
            geometry = maps_info.get("geometry", {})
            location = geometry.get("location", {})
            if location.get("lat") and location.get("lng"):
                updated_restaurant["latitude"] = location["lat"]
                updated_restaurant["longitude"] = location["lng"]
                updated_restaurant["coordinates"] = [location["lat"], location["lng"]]

                # Save coordinates back to database
                coordinates_saved = self._update_database_with_geodata(restaurant_name, destination, maps_info)

            # Add Google Maps URL if available
            if maps_info.get("url"):
                updated_restaurant["google_maps_url"] = maps_info["url"]

            # NEW: Add business status information
            if business_status:
                updated_restaurant["business_status"] = business_status

            # Log successful verification
            dump_chain_state("address_verified", {
                "restaurant": restaurant_name,
                "destination": destination,
                "verified_address": maps_info.get("formatted_address"),
                "coordinates": [location.get("lat"), location.get("lng")] if location.get("lat") else None,
                "rating": rating if rating else "N/A",
                "business_status": business_status,
                "saved_to_database": coordinates_saved
            })

        else:
            # No Maps data found - keep restaurant but log it
            logger.warning(f"⚠️ No Google Maps data found for {restaurant_name} in {destination}")
            dump_chain_state("google_maps_no_data", {
                "restaurant": restaurant_name,
                "destination": destination
            })

        # Add flag to track database updates
        updated_restaurant["coordinates_saved_to_db"] = coordinates_saved

        return updated_restaurant

    def _search_google_maps(self, restaurant_name: str, city: str) -> Optional[Dict[str, Any]]:
        """
        Search Google Maps for restaurant info including address, rating, and business status.
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

            # Get detailed place information including rating and business status
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=self.place_fields
            )

            result_data = place_details.get("result", {})

            formatted_address = result_data.get("formatted_address")
            rating = result_data.get("rating")
            user_ratings_total = result_data.get("user_ratings_total")
            business_status = result_data.get("business_status")

            return {
                "formatted_address": formatted_address,
                "rating": rating,
                "user_ratings_total": user_ratings_total,
                "business_status": business_status,
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

    def _update_database_with_geodata(self, restaurant_name: str, city: str, maps_info: Dict[str, Any]) -> bool:
        """Save address and coordinates back to the Supabase database"""
        try:
            # Use the database interface
            from utils.database import get_database

            # Get the database directly
            db = get_database()

            # Extract coordinates from Google Maps geometry
            geometry = maps_info.get("geometry", {})
            location = geometry.get("location", {})

            if location.get("lat") and location.get("lng"):
                coordinates = (float(location["lat"]), float(location["lng"]))
                address = maps_info.get("formatted_address", "")

                # Find the restaurant in database
                existing_restaurants = db.supabase.table('restaurants')\
                    .select('id, name')\
                    .eq('name', restaurant_name)\
                    .eq('city', city)\
                    .execute()

                if existing_restaurants.data:
                    restaurant_id = existing_restaurants.data[0]['id']

                    # Update with coordinates and address
                    db.update_restaurant_geodata(restaurant_id, address, coordinates)

                    logger.info(f"📍 Saved coordinates to database: {restaurant_name} at {coordinates}")
                    return True

                else:
                    logger.warning(f"Restaurant not found in database: {restaurant_name} in {city}")
                    return False

            else:
                logger.warning(f"No valid coordinates found for {restaurant_name}")
                return False

        except Exception as e:
            logger.error(f"Error updating database with geodata: {e}")
            return False