# agents/follow_up_search_agent.py - COMPLETE FIXED VERSION with address components

import logging
import googlemaps
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class FollowUpSearchAgent:
    """
    Enhanced follow-up search agent that handles:
    1. Address verification for ALL restaurants using Google Maps API
    2. Follow-up searches to check if restaurants have moved locations
    3. Rating filtering and restaurant rejection based on Google ratings
    4. Filtering out closed restaurants (temporarily or permanently) with auto-deletion
    5. Saving coordinates back to database
    6. FIXED: Proper address_components storage for street-only display
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
            "business_status",  # To check if restaurant is closed
            "opening_hours",    # Additional info about operating hours
            "address_components"  # For country extraction AND street-only display
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

    def perform_follow_up_searches(
        self,
        edited_results: Dict[str, List[Dict[str, Any]]],
        follow_up_queries: List[Dict[str, Any]] = None,  # Not used but kept for compatibility
        destination: str = "Unknown"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Performs follow-up searches for ALL restaurants and filters based on:
        1. Google ratings (below minimum threshold)
        2. Business status (closed restaurants)
        3. Location verification to check if restaurants have moved
        """

        main_list = edited_results.get("main_list", [])

        logger.info(f"Starting comprehensive verification for {len(main_list)} restaurants in {destination}")
        logger.info(f"Minimum acceptable rating: {self.min_acceptable_rating}")

        # Process each restaurant
        verified_restaurants = []
        rejected_count = 0
        closed_count = 0
        coordinates_saved_count = 0
        location_verified_count = 0

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
                # Count if location was verified
                if result.get("location_verified"):
                    location_verified_count += 1

        final_result = {"enhanced_results": {"main_list": verified_restaurants}}

        logger.info(f"✅ Follow-up searches and verification complete for {destination}")
        logger.info(f"   - Original count: {len(main_list)}")
        logger.info(f"   - Passed filter: {len(verified_restaurants)}")
        logger.info(f"   - Rejected (low rating): {rejected_count}")
        logger.info(f"   - Rejected (closed): {closed_count}")
        logger.info(f"   - Coordinates saved to DB: {coordinates_saved_count}")
        logger.info(f"   - Locations verified: {location_verified_count}")

        # Log overall statistics
        dump_chain_state("follow_up_search_complete", {
            "destination": destination,
            "original_count": len(main_list),
            "final_count": len(verified_restaurants),
            "rejected_count": rejected_count,
            "closed_count": closed_count,
            "coordinates_saved_count": coordinates_saved_count,
            "location_verified_count": location_verified_count,
            "min_rating": self.min_acceptable_rating
        })

        return final_result

    def _verify_and_filter_restaurant(self, restaurant: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced version that:
        1. Performs follow-up search for ALL restaurants using name + city format
        2. Checks if restaurants have moved to new locations
        3. Auto-deletes closed restaurants from database
        4. Filters by rating threshold
        5. STORES address_components for proper street-only formatting
        6. Saves coordinates back to database
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
        location_verified = False

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

                # Auto-delete from database when we find it's closed
                self._delete_closed_restaurant_from_database(restaurant_name, destination, business_status)

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
                    dump_chain_state("restaurant_rejected_rating", {
                        "restaurant": restaurant_name,
                        "destination": destination,
                        "rating": rating,
                        "threshold": self.min_acceptable_rating
                    })

                    logger.warning(f"❌ {restaurant_name} rejected: Rating {rating} below {self.min_acceptable_rating}")
                    return None

                logger.info(f"✅ {restaurant_name} passed rating filter: {rating}")

            
            # Update address information from Google Maps follow-up search
            formatted_address = maps_info.get("formatted_address")
            if formatted_address:
                old_address = restaurant.get("address", "")
                updated_restaurant["address"] = formatted_address
                location_verified = True

                # Remove "address" from missing_info if present
                missing_info = updated_restaurant.get("missing_info", [])
                if "address" in missing_info:
                    missing_info = [info for info in missing_info if info != "address"]
                    updated_restaurant["missing_info"] = missing_info

                if old_address != formatted_address:
                    logger.info(f"📍 Address updated for {restaurant_name}")
                    logger.debug(f"   Old: {old_address}")
                    logger.debug(f"   New: {formatted_address}")

                # Check if restaurant has potentially moved
                if old_address and old_address != formatted_address:
                    logger.info(f"🔄 Potential location change detected for {restaurant_name}")

            # Add coordinates for ALL restaurants
            geometry = maps_info.get("geometry", {})
            location = geometry.get("location", {})
            if location.get("lat") and location.get("lng"):
                updated_restaurant["latitude"] = location["lat"]
                updated_restaurant["longitude"] = location["lng"]
                updated_restaurant["coordinates"] = [location["lat"], location["lng"]]

                # Save coordinates and country back to database
                coordinates_saved = self._update_database_with_geodata(
                    restaurant_name, destination, maps_info
                )

            # CRITICAL: Store place_id and address_components for proper formatting
            place_id = maps_info.get("place_id")
            address_components = maps_info.get("address_components", [])

            if place_id:
                updated_restaurant["place_id"] = place_id
                logger.debug(f"✅ Stored place_id for {restaurant_name}: {place_id}")

            if address_components:
                updated_restaurant["address_components"] = address_components
                logger.debug(f"✅ Stored address_components for {restaurant_name}")

            # Add business status information
            if business_status:
                updated_restaurant["business_status"] = business_status

            # Update verification flags
            updated_restaurant.update({
                "verification_completed": True,
                "google_maps_verified": True,
                "google_place_id": maps_info.get("place_id", ""),
                "google_url": maps_info.get("url", "")
            })

            # Log successful verification
            dump_chain_state("follow_up_search_verified", {
                "restaurant": restaurant_name,
                "destination": destination,
                "verified_address": formatted_address,
                "coordinates": [location.get("lat"), location.get("lng")] if location.get("lat") else None,
                "rating": rating if rating else "N/A",
                "business_status": business_status,
                "saved_to_database": coordinates_saved,
                "location_verified": location_verified
            })

        else:
            # No Maps data found - keep restaurant but log it
            logger.warning(f"⚠️ No Google Maps data found for {restaurant_name} in {destination}")
            dump_chain_state("google_maps_no_data", {
                "restaurant": restaurant_name,
                "destination": destination
            })

            # Mark as unverified
            updated_restaurant.update({
                "verification_completed": True,
                "google_maps_verified": False,
                "coordinates_saved_to_db": False
            })

        # Add flags to track database updates
        updated_restaurant["coordinates_saved_to_db"] = coordinates_saved
        updated_restaurant["location_verified"] = location_verified

        return updated_restaurant


    def _delete_closed_restaurant_from_database(self, restaurant_name: str, city: str, business_status: str):
        """
        Delete a closed restaurant from the database

        Args:
            restaurant_name: Name of the restaurant to delete
            city: City where the restaurant is located  
            business_status: The closure status (CLOSED_PERMANENTLY or CLOSED_TEMPORARILY)
        """
        try:
            # Get the database interface
            from utils.database import get_database
            db = get_database()

            # Find the restaurant in the database
            existing_restaurants = db.supabase.table('restaurants')\
                .select('id, name, city')\
                .eq('name', restaurant_name)\
                .eq('city', city)\
                .execute()

            if existing_restaurants.data:
                restaurant_id = existing_restaurants.data[0]['id']
                restaurant_info = existing_restaurants.data[0]

                # Delete the restaurant
                delete_result = db.supabase.table('restaurants')\
                    .delete()\
                    .eq('id', restaurant_id)\
                    .execute()

                if delete_result.data:
                    logger.info(f"🗑️ AUTO-DELETED closed restaurant: {restaurant_name} in {city} (Status: {business_status})")

                    # Log the deletion for audit purposes
                    dump_chain_state("restaurant_auto_deleted", {
                        "restaurant_id": restaurant_id,
                        "restaurant_name": restaurant_name,
                        "city": city,
                        "business_status": business_status,
                        "deleted_at": datetime.now().isoformat()
                    })

                    return True
                else:
                    logger.warning(f"⚠️ Failed to delete {restaurant_name} from database")
                    return False
            else:
                logger.debug(f"Restaurant not found in database for deletion: {restaurant_name} in {city}")
                return False

        except Exception as e:
            logger.error(f"❌ Error auto-deleting closed restaurant {restaurant_name}: {e}")
            return False

    def _search_google_maps(self, restaurant_name: str, city: str) -> Optional[Dict[str, Any]]:
        """
        Search Google Maps for restaurant info including address, rating, business status, and address components.
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

            # Get detailed place information including rating, business status, and address components
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=self.place_fields
            )

            result_data = place_details.get("result", {})

            formatted_address = result_data.get("formatted_address")
            rating = result_data.get("rating")
            user_ratings_total = result_data.get("user_ratings_total")
            business_status = result_data.get("business_status")
            address_components = result_data.get("address_components", [])

            return {
                "formatted_address": formatted_address,
                "rating": rating,
                "user_ratings_total": user_ratings_total,
                "business_status": business_status,
                "place_id": place_id,
                "url": result_data.get("url", f"https://www.google.com/maps/place/?q=place_id:{place_id}"),
                "geometry": result_data.get("geometry", {}),
                "address_components": address_components
            }

        except googlemaps.exceptions.ApiError as e:
            logger.error(f"Google Maps API error for {restaurant_name} in {city}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error searching Google Maps for {restaurant_name} in {city}: {e}")
            return None

    def _update_database_with_geodata(self, restaurant_name: str, city: str, maps_info: Dict[str, Any]) -> bool:
        """
        Save address and coordinates back to the Supabase database

        Args:
            restaurant_name: Name of the restaurant
            city: City name
            maps_info: Google Maps information from follow-up search
        """
        try:
            # Use the database interface
            from utils.database import get_database
            db = get_database()

            # Extract coordinates from Google Maps geometry
            geometry = maps_info.get("geometry", {})
            location = geometry.get("location", {})

            if location.get("lat") and location.get("lng"):
                coordinates = (float(location["lat"]), float(location["lng"]))
                address = maps_info.get("formatted_address", "")

                # Find the restaurant in database
                existing_restaurants = db.supabase.table('restaurants')\
                    .select('id, name, country')\
                    .eq('name', restaurant_name)\
                    .eq('city', city)\
                    .execute()

                if existing_restaurants.data:
                    restaurant_id = existing_restaurants.data[0]['id']
                    current_country = existing_restaurants.data[0].get('country', '').strip()

                    # Prepare update data
                    update_data = {
                        'address': address,
                        'latitude': coordinates[0],
                        'longitude': coordinates[1],
                        'last_updated': datetime.now().isoformat()
                    }


                    # Update the restaurant
                    result = db.supabase.table('restaurants')\
                        .update(update_data)\
                        .eq('id', restaurant_id)\
                        .execute()

                    if result.data:
                        logger.info(f"📍 Updated database: {restaurant_name} at {coordinates}")
                        return True
                    else:
                        logger.error(f"❌ Failed to update restaurant in database: {restaurant_name}")
                        return False

                else:
                    logger.warning(f"Restaurant not found in database: {restaurant_name} in {city}")
                    return False

            else:
                logger.warning(f"No valid coordinates found for {restaurant_name}")
                return False

        except Exception as e:
            logger.error(f"Error updating database with geodata: {e}")
            return False