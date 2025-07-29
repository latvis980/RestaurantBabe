# agents/location_orchestrator.py
"""
Location Search Orchestrator

Coordinates the complete location-based search pipeline:
1. Check database for nearby restaurants
2. Search Google Maps for venues
3. AI-powered source mapping
4. Web search and verification
5. Format results for Telegram
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time

from utils.location_utils import LocationUtils, LocationPoint
from utils.telegram_location_handler import LocationData
from agents.location_search_agent import LocationSearchAgent, VenueResult
from agents.source_mapping_agent import SourceMappingAgent
from formatters.telegram_formatter import TelegramFormatter

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Orchestrates the complete location-based restaurant search pipeline
    """

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.location_search_agent = LocationSearchAgent(config)
        self.source_mapping_agent = SourceMappingAgent(config)
        self.telegram_formatter = TelegramFormatter()

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 2.0)
        self.min_db_matches = getattr(config, 'MIN_DB_MATCHES_REQUIRED', 3)
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        logger.info("✅ Location Orchestrator initialized")

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process a location-based restaurant query

        Args:
            query: User's search query (e.g. "natural wine bars")
            location_data: Location information (GPS or text description)
            cancel_check_fn: Function to check if search should be cancelled

        Returns:
            Dict with search results formatted for Telegram
        """
        try:
            logger.info(f"🎯 Processing location query: '{query}' at {location_data.location_type}")

            start_time = time.time()

            # STEP 1: Get coordinates
            coordinates = await self._get_coordinates(location_data, cancel_check_fn)
            if not coordinates:
                return self._create_error_response("Could not determine location coordinates")

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            lat, lng = coordinates
            logger.info(f"📍 Search coordinates: {lat:.4f}, {lng:.4f}")

            # STEP 2: Check database for nearby restaurants
            db_results = await self._check_database_proximity(coordinates, query, cancel_check_fn)

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 3: If sufficient database results, use them; otherwise search Google Maps
            if len(db_results) >= self.min_db_matches:
                logger.info(f"✅ Found {len(db_results)} restaurants in database - using database results")
                final_results = db_results
                search_method = "database"
            else:
                logger.info(f"📍 Found only {len(db_results)} restaurants in database - searching Google Maps")

                # Search Google Maps for venues
                venues = await self._search_google_maps(coordinates, query, cancel_check_fn)

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                if not venues:
                    return self._create_error_response("No venues found near your location")

                # STEP 4: AI-powered source mapping and verification
                verified_venues = await self._verify_venues_with_sources(venues, cancel_check_fn)

                if cancel_check_fn and cancel_check_fn():
                    return self._create_cancelled_response()

                final_results = verified_venues
                search_method = "google_maps"

            if cancel_check_fn and cancel_check_fn():
                return self._create_cancelled_response()

            # STEP 5: Format results for Telegram
            formatted_response = await self._format_location_results(
                final_results, coordinates, query, search_method, cancel_check_fn
            )

            # Add timing information
            total_time = time.time() - start_time
            formatted_response['processing_time'] = total_time

            logger.info(f"✅ Location search completed in {total_time:.1f}s with {len(final_results)} results")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error in location search pipeline: {e}")
            return self._create_error_response(f"Search failed: {str(e)}")

    async def _get_coordinates(self, location_data: LocationData, cancel_check_fn=None) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates from location data"""
        try:
            if location_data.location_type == "gps":
                # Direct GPS coordinates
                return (location_data.latitude, location_data.longitude)

            elif location_data.location_type == "description" and location_data.description:
                # Geocode text description
                logger.info(f"🗺️ Geocoding location: {location_data.description}")

                # Use existing geocoding infrastructure if available
                from utils.database import get_database
                db = get_database()

                # Try to geocode the location description
                coordinates = db._geocode_address(location_data.description)
                if coordinates:
                    logger.info(f"✅ Geocoded '{location_data.description}' to {coordinates}")
                    return coordinates

                # Fallback: return None if geocoding fails
                logger.warning(f"⚠️ Failed to geocode location: {location_data.description}")
                return None

            else:
                logger.error(f"❌ Invalid location data: {location_data}")
                return None

        except Exception as e:
            logger.error(f"❌ Error getting coordinates: {e}")
            return None

    async def _check_database_proximity(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Check database for restaurants near the coordinates"""
        try:
            logger.info(f"🗄️ Checking database for restaurants within {self.db_search_radius}km")

            from utils.database import get_database
            db = get_database()

            # Get all restaurants from database (we'll filter by proximity)
            # Note: This is a simplified approach - in production you'd want spatial queries
            all_restaurants = db.supabase.table('restaurants')\
                .select('*')\
                .not_.is_('latitude', 'null')\
                .not_.is_('longitude', 'null')\
                .execute()

            restaurants = all_restaurants.data or []

            if cancel_check_fn and cancel_check_fn():
                return []

            # Filter by proximity using LocationUtils
            nearby_restaurants = LocationUtils.filter_by_proximity(
                center=coordinates,
                points=restaurants,
                radius_km=self.db_search_radius,
                lat_key="latitude",
                lon_key="longitude"
            )

            # TODO: Add query-specific filtering (cuisine type, etc.)
            # For now, return all nearby restaurants

            logger.info(f"📊 Found {len(nearby_restaurants)} restaurants in database within {self.db_search_radius}km")
            return nearby_restaurants

        except Exception as e:
            logger.error(f"❌ Error checking database proximity: {e}")
            return []

    async def _search_google_maps(
        self, 
        coordinates: Tuple[float, float], 
        query: str, 
        cancel_check_fn=None
    ) -> List[VenueResult]:
        """Search Google Maps for venues near coordinates"""
        try:
            logger.info(f"🗺️ Searching Google Maps for venues")

            lat, lng = coordinates

            # Determine venue type from query
            venue_type = self.location_search_agent.determine_venue_type(query)

            # Search for nearby venues
            venues = self.location_search_agent.search_nearby_venues(
                latitude=lat,
                longitude=lng,
                query=query,
                venue_type=venue_type
            )

            logger.info(f"🏪 Google Maps found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"❌ Error searching Google Maps: {e}")
            return []

    async def _verify_venues_with_sources(
        self, 
        venues: List[VenueResult], 
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """Use AI to map sources and verify venue information"""
        try:
            logger.info(f"🔍 Starting AI-powered source verification for {len(venues)} venues")

            verified_venues = []

            for i, venue in enumerate(venues[:self.max_venues_to_verify]):
                if cancel_check_fn and cancel_check_fn():
                    break

                logger.debug(f"📰 Verifying venue {i+1}/{len(venues)}: {venue.name}")

                # Get AI source mapping
                source_mapping = self.source_mapping_agent.map_sources_for_venue(
                    venue_name=venue.name,
                    venue_type=self._classify_venue_type(venue),
                    location=venue.address,
                    venue_description=f"Rating: {venue.rating}, Reviews: {venue.user_ratings_total}"
                )

                if cancel_check_fn and cancel_check_fn():
                    break

                # TODO: Implement actual web search and verification
                # For now, create a placeholder verified venue
                verified_venue = {
                    "name": venue.name,
                    "address": venue.address,
                    "google_maps_url": venue.google_maps_url,
                    "rating": venue.rating,
                    "user_ratings_total": venue.user_ratings_total,
                    "distance_km": venue.distance_km,
                    "latitude": venue.latitude,
                    "longitude": venue.longitude,
                    "price_level": venue.price_level,
                    "source_mapping": source_mapping,
                    "verification_status": "pending",  # Will be "verified" after web search
                    "reputable_sources": [],  # Will contain found sources
                    "description": f"Google Maps rating: {venue.rating}/5 ({venue.user_ratings_total} reviews)"
                }

                verified_venues.append(verified_venue)

            logger.info(f"✅ Verified {len(verified_venues)} venues with source mapping")
            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error verifying venues: {e}")
            return []

    def _classify_venue_type(self, venue: VenueResult) -> str:
        """Classify venue type from Google Places types"""
        types = [t.lower() for t in venue.types]

        if any(t in types for t in ["bar", "night_club"]):
            return "bar"
        elif any(t in types for t in ["cafe", "coffee"]):
            return "cafe"
        elif any(t in types for t in ["bakery"]):
            return "bakery"
        elif any(t in types for t in ["meal_takeaway", "meal_delivery"]):
            return "casual_dining"
        else:
            return "restaurant"

    async def _format_location_results(
        self, 
        results: List[Dict[str, Any]], 
        coordinates: Tuple[float, float],
        query: str,
        search_method: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Format results for Telegram display"""
        try:
            logger.info(f"📱 Formatting {len(results)} results for Telegram")

            if not results:
                return self._create_error_response("No restaurants found near your location")

            # Prepare data for formatter
            lat, lng = coordinates
            location_summary = f"{lat:.4f}, {lng:.4f}"

            # Create formatted text using existing TelegramFormatter patterns
            formatted_text = self._create_location_response_text(
                results, query, location_summary, search_method
            )

            return {
                "success": True,
                "telegram_formatted_text": formatted_text,
                "results_count": len(results),
                "search_method": search_method,
                "coordinates": coordinates,
                "query": query
            }

        except Exception as e:
            logger.error(f"❌ Error formatting results: {e}")
            return self._create_error_response("Failed to format results")

    def _create_location_response_text(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        location_summary: str,
        search_method: str
    ) -> str:
        """Create formatted text response for location search results"""

        # Header
        header = f"📍 <b>Found {len(results)} places for '{query}'</b>\n\n"

        if search_method == "database":
            method_info = "🗄️ <i>Results from our curated database</i>\n\n"
        else:
            method_info = "🗺️ <i>Results from Google Maps + verified sources</i>\n\n"

        # Format each result
        formatted_results = []

        for i, result in enumerate(results, 1):
            name = result.get('name', 'Unknown')
            distance = result.get('distance_km', 0)
            rating = result.get('rating')
            address = result.get('address', '')

            # Format distance
            distance_str = LocationUtils.format_distance(distance)

            # Format rating
            rating_str = f"⭐ {rating}/5" if rating else "No rating"

            # Create Google Maps link
            lat = result.get('latitude')
            lng = result.get('longitude')
            if lat and lng:
                maps_url = LocationUtils.generate_google_maps_url(lat, lng, name)
                address_link = f"<a href='{maps_url}'>{address}</a>"
            else:
                address_link = address

            # Add source information if available
            source_info = ""
            if 'source_mapping' in result:
                primary_source = result['source_mapping'].get('primary_source', '')
                if primary_source:
                    source_info = f"\n📰 <i>Verified by {primary_source}</i>"

            result_text = (
                f"<b>{i}. {name}</b> ({distance_str})\n"
                f"{rating_str}\n"
                f"📍 {address_link}"
                f"{source_info}\n"
            )

            formatted_results.append(result_text)

        # Footer
        footer = (
            f"\n💡 <i>Showing results within walking distance of your location</i>\n"
            f"📱 Tap addresses for directions in Google Maps"
        )

        return header + method_info + "\n".join(formatted_results) + footer

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "telegram_formatted_text": f"😔 {error_message}\n\nPlease try a different search or location.",
            "error": error_message
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create cancelled response"""
        return {
            "success": False,
            "telegram_formatted_text": "✋ Search was cancelled.",
            "cancelled": True
        }