# agents/follow_up_search_agent.py - COMPLETE FIXED VERSION with address component and intelligent venue type detection

import logging
import googlemaps
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from langsmith import traceable
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.google_links import build_google_maps_url

logger = logging.getLogger(__name__)

class FollowUpSearchAgent:
    """
    Enhanced follow-up search agent that handles:
    1. Address verification for ALL restaurants using Google Maps API
    2. Country extraction from Google Maps formatted addresses 
    3. Rating filtering and restaurant rejection based on Google ratings
    4. Filtering out closed restaurants (temporarily or permanently) with auto-deletion
    5. Saving coordinates and corrected country data back to database
    6. FIXED: Proper address_components storage for street-only display
    7. NEW: Intelligent venue type detection from cuisine tags and descriptions
    """

    def __init__(self, config):
        self.config = config

        # Rating threshold for restaurant filtering
        self.min_acceptable_rating = getattr(config, 'MIN_ACCEPTABLE_RATING', 4.1)

        # Initialize Google Maps client
        api_key = getattr(config, 'GOOGLE_MAPS_API_KEY', None)
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY is required in config")

        api_key_2 = getattr(config, 'GOOGLE_MAPS_API_KEY2', None)
        if api_key_2:
            self.gmaps_secondary = googlemaps.Client(key=api_key_2)
            self.has_dual_keys = True
            logger.info("✅ Secondary Google Maps client initialized - dual key mode enabled")
        else:
            self.gmaps_secondary = None
            self.has_dual_keys = False
            logger.info("ℹ️ No secondary API key found - single key mode")

        # Track API usage for intelligent rotation
        self.api_usage = {
            'primary': 0,
            'secondary': 0
        }

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
            "address_component"  # FIXED: Singular form for Google Places API
        ]

    def _determine_venue_type(self, restaurant: Dict[str, Any]) -> str:
        """
        Intelligently determine venue type from restaurant data

        Analyzes cuisine_tags, description, and name to determine
        the most appropriate venue type for search query
        """
        # Get cuisine tags and description
        cuisine_tags = restaurant.get('cuisine_tags', [])
        description = restaurant.get('description', '').lower()
        name = restaurant.get('name', '').lower()

        # Convert cuisine tags to lowercase for matching
        tags_lower = [tag.lower() for tag in cuisine_tags]

        # Check for specific venue types in order of specificity
        venue_type_indicators = {
            'wine bar': ['wine-bar', 'wine bar', 'natural wine', 'wine focused'],
            'cocktail bar': ['cocktail', 'mixology', 'speakeasy', 'cocktail-bar'],
            'coffee shop': ['coffee', 'cafe', 'espresso', 'coffee-shop'],
            'bakery': ['bakery', 'patisserie', 'bread', 'pastry'],
            'bistro': ['bistro', 'brasserie'],
            'steakhouse': ['steakhouse', 'steak', 'grill'],
            'sushi bar': ['sushi', 'omakase', 'japanese'],
            'pizzeria': ['pizza', 'pizzeria'],
            'tapas bar': ['tapas', 'spanish bar'],
            'pub': ['pub', 'gastropub', 'public house'],
            'bar': ['bar', 'tavern', 'lounge'],
            'cafe': ['cafe', 'brunch', 'breakfast']
        }

        # Check each venue type
        for venue_type, indicators in venue_type_indicators.items():
            # Check in cuisine tags
            if any(indicator in tags_lower for indicator in indicators):
                logger.debug(f"Detected venue type '{venue_type}' from cuisine tags")
                return venue_type

            # Check in description
            if any(indicator in description for indicator in indicators):
                logger.debug(f"Detected venue type '{venue_type}' from description")
                return venue_type

            # Check in name
            if any(indicator in name for indicator in indicators):
                logger.debug(f"Detected venue type '{venue_type}' from name")
                return venue_type

        # Default fallback based on common patterns
        if any(tag in tags_lower for tag in ['street-food', 'food-truck', 'market']):
            return 'food'

        # Final default
        return 'restaurant'

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

    def _get_gmaps_client(self) -> tuple:
        """
        Get the appropriate Google Maps client based on usage
        Returns tuple of (client, key_name)
        """
        if not self.has_dual_keys:
            return self.gmaps, "primary"

        # Rotate between keys to balance usage
        if self.api_usage['primary'] <= self.api_usage['secondary']:
            return self.gmaps, "primary"
        else:
            return self.gmaps_secondary, "secondary"
    
    @log_function_call
    def perform_follow_up_searches(
        self,
            edited_results: Dict[str, List[Dict[str, Any]]],
            follow_up_queries: Optional[List[Dict[str, Any]]] = None,  # Changed to Optional
            destination: str = "Unknown"
        ) -> Dict[str, Any]:
        """
        Verifies addresses for ALL restaurants and filters based on:
        1. Google ratings (below minimum threshold)
        2. Business status (closed restaurants)
        3. ALSO extracts and corrects country information from Google Maps addresses
        """

        main_list = edited_results.get("main_list", [])

        logger.info(f"Starting comprehensive verification for {len(main_list)} restaurants in {destination}")
        logger.info(f"Minimum acceptable rating: {self.min_acceptable_rating}")

        # Process each restaurant
        verified_restaurants = []
        rejected_count = 0
        closed_count = 0
        coordinates_saved_count = 0
        country_extracted_count = 0

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
                # Count if country was extracted/corrected
                if result.get("country_extracted_from_address"):
                    country_extracted_count += 1

        final_result = {"enhanced_results": {"main_list": verified_restaurants}}

        logger.info(f"✅ Comprehensive verification complete for {destination}")
        logger.info(f"   - Original count: {len(main_list)}")
        logger.info(f"   - Passed filter: {len(verified_restaurants)}")
        logger.info(f"   - Rejected (low rating): {rejected_count}")
        logger.info(f"   - Rejected (closed): {closed_count}")
        logger.info(f"   - Coordinates saved to DB: {coordinates_saved_count}")
        logger.info(f"   - Countries extracted from addresses: {country_extracted_count}")

        # Log overall statistics
        dump_chain_state("follow_up_search_complete", {
            "destination": destination,
            "original_count": len(main_list),
            "final_count": len(verified_restaurants),
            "rejected_count": rejected_count,
            "closed_count": closed_count,
            "coordinates_saved_count": coordinates_saved_count,
            "country_extracted_count": country_extracted_count,
            "min_rating": self.min_acceptable_rating
        })


        return final_result

    def _verify_and_filter_restaurant(self, restaurant: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced version that:
        1. Verifies addresses for ALL restaurants (not just those marked for verification)
        2. Extracts country from Google Maps formatted addresses
        3. Auto-deletes closed restaurants from database
        4. Filters by rating threshold
        5. STORES address_components for proper street-only formatting
        6. Saves coordinates and country data back to database
        7. NEW: Uses intelligent venue type detection for better search accuracy
        """
        # Make a copy to avoid modifying the original
        updated_restaurant = restaurant.copy()

        # Initialize extracted_country to None
        extracted_country = None  # ADD THIS LINE
        coordinates_saved = False
        country_extracted = False

        restaurant_name = restaurant.get("name", "")
        if not restaurant_name:
            logger.warning("Restaurant missing name, cannot verify")
            return updated_restaurant

        logger.info(f"Verifying: {restaurant_name} in {destination}")

        # Search Google Maps for restaurant info with intelligent venue type
        maps_info = self._search_google_maps(restaurant_name, destination, restaurant)

        coordinates_saved = False
        country_extracted = False

        if maps_info:
            # FIXED: Extract variables from maps_info at the start
            place_id = maps_info.get("place_id")
            google_url = maps_info.get("url")
            address_components = maps_info.get("address_components", [])

            if place_id:
                updated_restaurant["place_id"] = place_id
                logger.debug(f"✅ Stored place_id: {place_id} for {restaurant_name}")

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

                # Auto-delete from database ONLY if permanently closed
                # Keep temporarily closed restaurants in database (they might reopen)
                if business_status == "CLOSED_PERMANENTLY":
                    self._delete_closed_restaurant_from_database(restaurant_name, destination, business_status)
                    logger.info(f"🗑️ Permanently closed restaurant will be deleted from database")
                else:
                    logger.info(f"📋 Temporarily closed restaurant kept in database (might reopen)")

                updated_restaurant["is_closed"] = True
                updated_restaurant["closure_reason"] = business_status
                return None  # Reject both types from user results

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

            # ENHANCED: Always update address information from Google Maps AND extract country
            formatted_address = maps_info.get("formatted_address")
            if formatted_address:
                old_address = restaurant.get("address", "")
                updated_restaurant["address"] = formatted_address

                # Extract country from the formatted address 
                extracted_country = self._extract_country_from_address(formatted_address)
                if extracted_country:
                    updated_restaurant["country"] = extracted_country
                    country_extracted = True
                    logger.info(f"🌍 Extracted country: {extracted_country} for {restaurant_name}")

                # Remove "address" from missing_info if present
                missing_info = updated_restaurant.get("missing_info", [])
                if "address" in missing_info:
                    missing_info = [info for info in missing_info if info != "address"]
                    updated_restaurant["missing_info"] = missing_info

                if old_address != formatted_address:
                    logger.info(f"📍 Address updated for {restaurant_name}")
                    logger.debug(f"   Old: {old_address}")
                    logger.debug(f"   New: {formatted_address}")

            # Add coordinates for ALL restaurants
            geometry = maps_info.get("geometry", {})
            location = geometry.get("location", {})
            if location.get("lat") and location.get("lng"):
                updated_restaurant["latitude"] = location["lat"]
                updated_restaurant["longitude"] = location["lng"]
                updated_restaurant["coordinates"] = [location["lat"], location["lng"]]

                # Save coordinates and country back to database
                coordinates_saved = self._update_database_with_geodata(
                    restaurant_name, destination, maps_info, extracted_country
                )

            # Add Google Maps URL using official place_id format or Google's URL
            if place_id:
                updated_restaurant["google_maps_url"] = build_google_maps_url(place_id, restaurant_name)
            elif google_url:

                updated_restaurant["google_maps_url"] = google_url

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
            dump_chain_state("address_verified", {
                "restaurant": restaurant_name,
                "destination": destination,
                "verified_address": formatted_address,
                "extracted_country": extracted_country,
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

            # Mark as unverified
            updated_restaurant.update({
                "verification_completed": True,
                "google_maps_verified": False,
                "coordinates_saved_to_db": False
            })

        # Add flags to track database updates
        updated_restaurant["coordinates_saved_to_db"] = coordinates_saved
        updated_restaurant["country_extracted_from_address"] = country_extracted

        return updated_restaurant

    def _extract_country_from_address(self, formatted_address: str) -> Optional[str]:
        """
        Extract country from Google Maps formatted address

        Google Maps addresses typically end with the country name.
        Examples:
        - "123 Main St, New York, NY 10001, USA" -> "USA"
        - "Rua Augusta 123, 1100-048 Lisboa, Portugal" -> "Portugal" 
        - "1-1-1 Shibuya, Tokyo 150-0002, Japan" -> "Japan"
        """
        try:
            if not formatted_address:
                return None

            # Split by commas and take the last part (usually country)
            parts = [part.strip() for part in formatted_address.split(',')]

            if not parts:
                return None

            # The last part is usually the country
            potential_country = parts[-1].strip()

            # Clean up common patterns
            # Remove postal codes (numbers at the end)
            potential_country = re.sub(r'\s+\d+.*$', '', potential_country)

            # Handle specific country name mappings/cleaning
            country_mappings = {
                'USA': 'United States',
                'US': 'United States', 
                'UK': 'United Kingdom',
                'UAE': 'United Arab Emirates'
            }

            # Clean the country name
            potential_country = potential_country.strip()

            # Apply mappings if needed
            if potential_country in country_mappings:
                potential_country = country_mappings[potential_country]

            # Basic validation - country should be alphabetic and reasonable length
            if (potential_country and 
                len(potential_country) >= 2 and 
                len(potential_country) <= 50 and
                re.match(r'^[a-zA-Z\s\-\.]+$', potential_country)):

                return potential_country

            return None

        except Exception as e:
            logger.error(f"Error extracting country from address '{formatted_address}': {e}")
            return None

    def _delete_closed_restaurant_from_database(self, restaurant_name: str, city: str, business_status: Optional[str] = None):
        """
        Delete a closed restaurant from the database

        Args:
            restaurant_name: Name of the restaurant to delete
            city: City where the restaurant is located  
            business_status: The closure status (CLOSED_PERMANENTLY or CLOSED_TEMPORARILY) or None
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
                # Remove unused variable: restaurant_info = existing_restaurants.data[0]

                # Delete the restaurant
                delete_result = db.supabase.table('restaurants')\
                    .delete()\
                    .eq('id', restaurant_id)\
                    .execute()

                if delete_result.data:
                    logger.info(f"🗑️ AUTO-DELETED closed restaurant: {restaurant_name} in {city} (Status: {business_status or 'Unknown'})")

                    # Log the deletion for audit purposes
                    dump_chain_state("restaurant_auto_deleted", {
                        "restaurant_id": restaurant_id,
                        "restaurant_name": restaurant_name,
                        "city": city,
                        "business_status": business_status or "Unknown",
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

    @traceable(run_type="tool", name="google_maps_follow_up")
    def _search_google_maps(self, restaurant_name: str, city: str, restaurant_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        OPTIMIZED: Search Google Maps for restaurant info with place_id optimization.
        
        For restaurants from database (with existing place_id):
        - Skip expensive text search (saves quota)
        - Go directly to cheap place details API
        
        For new restaurants (no place_id):
        - Fall back to expensive text search + place details
        """
        try:
            # Get the appropriate client
            gmaps_client, key_name = self._get_gmaps_client()

            # OPTIMIZATION: Check for existing place_id first
            existing_place_id = None
            if restaurant_data:
                existing_place_id = restaurant_data.get('place_id') or restaurant_data.get('google_maps_place_id')
            
            place_id = None
            
            if existing_place_id:
                # FAST PATH: Try using existing place_id first
                logger.info(f"💰 QUOTA SAVING: Attempting to use existing place_id for {restaurant_name}")
                
                try:
                    # Get detailed place information using existing place_id
                    place_details = gmaps_client.place(
                        place_id=existing_place_id,
                        fields=self.place_fields
                    )
                    
                    # Update usage counter for place details call
                    self.api_usage[key_name] += 1
                    
                    result_data = place_details.get("result", {})
                    
                    # Check if place details returned valid data
                    if result_data and result_data.get("place_id"):
                        place_id = existing_place_id
                        logger.info(f"✅ Successfully used existing place_id for {restaurant_name}")
                    else:
                        logger.warning(f"⚠️ Existing place_id invalid/stale for {restaurant_name}, falling back to text search")
                        
                except googlemaps.exceptions.ApiError as e:
                    if "NOT_FOUND" in str(e):
                        logger.warning(f"⚠️ Existing place_id not found for {restaurant_name}, falling back to text search")
                    else:
                        logger.error(f"Place details API error for existing place_id: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error using existing place_id: {e}")
            
            # FALLBACK PATH: If fast path failed or no existing place_id
            if not place_id:
                logger.info(f"💸 Using expensive text search for {restaurant_name}")
                
                # Determine venue type intelligently
                venue_type = self._determine_venue_type(restaurant_data)

                # Create search query with appropriate venue type
                search_query = f"{restaurant_name} {venue_type} {city}"
                logger.debug(f"Google Maps search query ({key_name} key): {search_query} [detected type: {venue_type}]")

                # Perform expensive text search
                search_response = gmaps_client.places(query=search_query)

                # Update usage counter for expensive text search
                self.api_usage[key_name] += 1

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

                # Get detailed place information using new place_id
                place_details = gmaps_client.place(
                    place_id=place_id,
                    fields=self.place_fields
                )
                
                # Update usage counter for place details call
                self.api_usage[key_name] += 1

                result_data = place_details.get("result", {})

            formatted_address = result_data.get("formatted_address")
            rating = result_data.get("rating")
            user_ratings_total = result_data.get("user_ratings_total")
            business_status = result_data.get("business_status")
            address_components = result_data.get("address_components", [])

            # Generate 2025 universal format URL
            if restaurant_name and restaurant_name.strip():
                    google_maps_url = build_google_maps_url(place_id, restaurant_name)
                
            else:
                    google_maps_url = build_google_maps_url(place_id)

            return {
                "formatted_address": formatted_address,
                "rating": rating,
                "user_ratings_total": user_ratings_total,
                "business_status": business_status,
                "place_id": place_id,
                "url": google_maps_url,
                "geometry": result_data.get("geometry", {}),
                "address_components": address_components
            }

        except googlemaps.exceptions.ApiError as e:
            logger.error(f"Google Maps API error for {restaurant_name} in {city}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error searching Google Maps for {restaurant_name} in {city}: {e}")
            return None

    def _update_database_with_geodata(self, restaurant_name: str, city: str, maps_info: Dict[str, Any], extracted_country: Optional[str] = None) -> bool:
        """
        Save address, coordinates, and country back to the Supabase database

        Args:
            restaurant_name: Name of the restaurant
            city: City name
            maps_info: Google Maps information
            extracted_country: Country extracted from formatted address (optional)
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

                    # Update country if we extracted one and current is missing/Unknown
                    if (extracted_country and 
                        (not current_country or current_country.lower() in ['unknown', '', 'null'])):
                        update_data['country'] = extracted_country
                        logger.info(f"🌍 Updating country: {restaurant_name} -> {extracted_country}")

                    # Update the restaurant
                    result = db.supabase.table('restaurants')\
                        .update(update_data)\
                        .eq('id', restaurant_id)\
                        .execute()

                    if result.data:
                        logger.info(f"📍 Updated database: {restaurant_name} at {coordinates}")
                        if extracted_country:
                            logger.info(f"🌍 Country set to: {extracted_country}")
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