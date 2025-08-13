# location/google_maps_search.py
"""
Google Maps Search Agent - STEP 3

Clean implementation for Google Maps Places API search.
This implements Step 3 of the location search flow:
- Search Google Maps for venues that match the query
- Clean, single-purpose Google Maps functionality
- Consistent result format
"""

import logging
import googlemaps
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

from location.location_utils import LocationUtils

logger = logging.getLogger(__name__)

@dataclass
class VenueResult:
    """Structure for venue search results"""
    name: str
    place_id: str
    address: str
    latitude: float
    longitude: float
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None
    types: List[str] = None
    distance_km: Optional[float] = None
    google_maps_url: str = ""

    def __post_init__(self):
        if self.types is None:
            self.types = []
        if not self.google_maps_url and self.place_id:
            self.google_maps_url = f"https://maps.google.com/maps/place/?q=place_id:{self.place_id}"

class GoogleMapsSearchAgent:
    """
    STEP 3: Clean Google Maps search implementation

    Searches for restaurants and venues near a specific location using Google Maps.
    Single responsibility: Google Maps API interaction only.
    """

    def __init__(self, config):
        self.config = config

        # Initialize Google Maps client
        # Priority: GOOGLE_MAPS_KEY2 > GOOGLE_MAPS_API_KEY
        api_key = getattr(config, 'GOOGLE_MAPS_KEY2', None) or getattr(config, 'GOOGLE_MAPS_API_KEY', None)

        if not api_key:
            raise ValueError("Google Maps API key not found. Set GOOGLE_MAPS_KEY2 or GOOGLE_MAPS_API_KEY")

        try:
            self.gmaps = googlemaps.Client(key=api_key)
            key_source = "GOOGLE_MAPS_KEY2" if getattr(config, 'GOOGLE_MAPS_KEY2', None) else "GOOGLE_MAPS_API_KEY"
            logger.info(f"✅ Google Maps Search Agent initialized (Step 3) using {key_source}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Maps client: {e}")
            raise

        # Search parameters from config
        self.search_radius = getattr(config, 'LOCATION_SEARCH_RADIUS_KM', 2.0) * 1000  # Convert to meters
        self.max_results = getattr(config, 'MAX_LOCATION_RESULTS', 8)
        self.search_timeout = getattr(config, 'LOCATION_SEARCH_TIMEOUT', 30.0)

        # Venue types from config
        self.venue_types = getattr(config, 'GOOGLE_PLACES_SEARCH_TYPES', [
            "restaurant", "bar", "cafe", "meal_takeaway", 
            "meal_delivery", "food", "bakery"
        ])

        logger.info(f"🔧 Search parameters: radius={self.search_radius/1000}km, max_results={self.max_results}")

    async def search_venues(
        self,
        coordinates: Tuple[float, float],
        query: str,
        radius_km: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> List[VenueResult]:
        """
        STEP 3: Search Google Maps for venues that match the query

        Args:
            coordinates: (latitude, longitude) tuple
            query: Search query (e.g., "wine bars", "sushi restaurants")
            radius_km: Search radius in kilometers
            max_results: Maximum number of results

        Returns:
            List of VenueResult objects
        """
        try:
            latitude, longitude = coordinates
            search_radius_m = (radius_km * 1000) if radius_km else self.search_radius
            max_results = max_results or self.max_results

            logger.info(f"🗺️ STEP 3: Google Maps search for '{query}' near {latitude:.4f}, {longitude:.4f}")
            logger.info(f"🔍 Search parameters: radius={search_radius_m/1000}km, max_results={max_results}")

            # Perform the search
            venues = await self._perform_places_search(
                latitude=latitude,
                longitude=longitude,
                query=query,
                radius_meters=search_radius_m,
                max_results=max_results
            )

            logger.info(f"✅ STEP 3 COMPLETE: Found {len(venues)} venues from Google Maps")
            return venues

        except Exception as e:
            logger.error(f"❌ Error in Step 3 Google Maps search: {e}")
            return []

    async def _perform_places_search(
        self,
        latitude: float,
        longitude: float,
        query: str,
        radius_meters: float,
        max_results: int
    ) -> List[VenueResult]:
        """
        Perform the actual Google Places API search
        """
        try:
            # Build search parameters
            location = f"{latitude},{longitude}"

            # Try text search first (more flexible for complex queries)
            search_results = self._text_search(query, location, radius_meters)

            if not search_results:
                # Fallback to nearby search with specific types
                search_results = self._nearby_search(location, radius_meters, query)

            # Convert to VenueResult objects
            venues = []
            user_location = (latitude, longitude)

            for place in search_results[:max_results]:
                try:
                    venue = self._convert_to_venue_result(place, user_location)
                    if venue:
                        venues.append(venue)
                except Exception as e:
                    logger.warning(f"Error converting place to venue: {e}")
                    continue

            logger.info(f"Converted {len(venues)} places to venue results")
            return venues

        except Exception as e:
            logger.error(f"❌ Error in places search: {e}")
            return []

    def _text_search(self, query: str, location: str, radius_meters: float) -> List[Dict[str, Any]]:
        """
        Perform Google Places text search
        """
        try:
            logger.info(f"🔍 Trying text search for: '{query}'")

            response = self.gmaps.places(
                query=query,
                location=location,
                radius=int(radius_meters),
                type='restaurant'  # Broad type filter
            )

            results = response.get('results', [])
            logger.info(f"Text search returned {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Text search failed: {e}")
            return []

    def _nearby_search(self, location: str, radius_meters: float, query: str) -> List[Dict[str, Any]]:
        """
        Fallback: Google Places nearby search with specific types
        """
        try:
            logger.info(f"🔍 Trying nearby search as fallback")

            # Use configured venue types
            all_results = []

            for venue_type in self.venue_types[:3]:  # Limit to first 3 types to avoid API limits
                try:
                    response = self.gmaps.places_nearby(
                        location=location,
                        radius=int(radius_meters),
                        type=venue_type,
                        keyword=query  # Use query as keyword
                    )

                    results = response.get('results', [])
                    all_results.extend(results)
                    logger.info(f"Nearby search ({venue_type}) returned {len(results)} results")

                except Exception as e:
                    logger.warning(f"Nearby search failed for type {venue_type}: {e}")
                    continue

            # Remove duplicates by place_id
            unique_results = {}
            for result in all_results:
                place_id = result.get('place_id')
                if place_id and place_id not in unique_results:
                    unique_results[place_id] = result

            results = list(unique_results.values())
            logger.info(f"Nearby search returned {len(results)} unique results")
            return results

        except Exception as e:
            logger.warning(f"Nearby search failed: {e}")
            return []

    def _convert_to_venue_result(
        self, 
        place: Dict[str, Any], 
        user_location: Tuple[float, float]
    ) -> Optional[VenueResult]:
        """
        Convert Google Places API result to VenueResult object
        """
        try:
            # Extract required fields
            place_id = place.get('place_id')
            name = place.get('name')

            if not place_id or not name:
                return None

            # Extract location
            geometry = place.get('geometry', {})
            location = geometry.get('location', {})
            latitude = location.get('lat')
            longitude = location.get('lng')

            if latitude is None or longitude is None:
                return None

            # Extract other fields
            address = place.get('formatted_address', place.get('vicinity', ''))
            rating = place.get('rating')
            user_ratings_total = place.get('user_ratings_total')
            price_level = place.get('price_level')
            types = place.get('types', [])

            # Calculate distance
            distance_km = LocationUtils.calculate_distance(
                user_location, (latitude, longitude)
            )

            return VenueResult(
                name=name,
                place_id=place_id,
                address=address,
                latitude=latitude,
                longitude=longitude,
                rating=rating,
                user_ratings_total=user_ratings_total,
                price_level=price_level,
                types=types,
                distance_km=round(distance_km, 2)
            )

        except Exception as e:
            logger.warning(f"Error converting place to venue result: {e}")
            return None

    def get_place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific place

        Useful for Steps 4-5 when we need more information for verification
        """
        try:
            response = self.gmaps.place(
                place_id=place_id,
                fields=['name', 'formatted_address', 'geometry', 'rating', 
                       'user_ratings_total', 'price_level', 'types', 'website',
                       'formatted_phone_number', 'opening_hours']
            )

            return response.get('result', {})

        except Exception as e:
            logger.error(f"❌ Error getting place details for {place_id}: {e}")
            return None