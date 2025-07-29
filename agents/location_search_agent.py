# agents/location_search_agent.py
"""
Location Search Agent

Uses Google Maps Places API to find restaurants and venues near a specific location.
Integrates with the existing database to check for known restaurants first.
"""

import logging
import googlemaps
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

from utils.location_utils import LocationUtils

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

class LocationSearchAgent:
    """
    Searches for restaurants and venues near a specific location using Google Maps
    """

    def __init__(self, config):
        self.config = config

        # Initialize Google Maps client
        api_key = getattr(config, 'GOOGLE_MAPS_KEY2', None) or getattr(config, 'GOOGLE_MAPS_API_KEY', None)
        if not api_key:
            raise ValueError("Google Maps API key not found. Set GOOGLE_MAPS_KEY2 or GOOGLE_MAPS_API_KEY")

        try:
            self.gmaps = googlemaps.Client(key=api_key)
            logger.info("✅ Google Maps client initialized for location search")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Maps client: {e}")
            raise

        # Search parameters
        self.search_radius = getattr(config, 'LOCATION_SEARCH_RADIUS_KM', 2.0) * 1000  # Convert to meters
        self.max_results = getattr(config, 'MAX_LOCATION_RESULTS', 8)
        self.search_timeout = getattr(config, 'LOCATION_SEARCH_TIMEOUT', 30.0)

        # Venue types to search for
        self.venue_types = getattr(config, 'GOOGLE_PLACES_SEARCH_TYPES', [
            "restaurant", "bar", "cafe", "meal_takeaway", 
            "meal_delivery", "food", "bakery"
        ])

        logger.info(f"📍 Location search configured: {self.search_radius/1000}km radius, max {self.max_results} results")

    def search_nearby_venues(
        self, 
        latitude: float, 
        longitude: float, 
        query: str,
        venue_type: Optional[str] = None
    ) -> List[VenueResult]:
        """
        Search for venues near a location

        Args:
            latitude: GPS latitude
            longitude: GPS longitude  
            query: Search query (e.g. "natural wine bars")
            venue_type: Optional specific venue type filter

        Returns:
            List[VenueResult]: Found venues sorted by relevance and rating
        """
        try:
            logger.info(f"🔍 Searching for '{query}' near {latitude:.4f}, {longitude:.4f}")

            # Validate coordinates
            if not LocationUtils.validate_coordinates(latitude, longitude):
                logger.error(f"Invalid coordinates: {latitude}, {longitude}")
                return []

            center_point = (latitude, longitude)
            venues = []

            # Method 1: Text search with location bias
            text_venues = self._text_search_nearby(center_point, query)
            venues.extend(text_venues)

            # Method 2: Nearby search by type (if specific venue type detected)
            if venue_type:
                type_venues = self._nearby_search_by_type(center_point, venue_type)
                venues.extend(type_venues)

            # Remove duplicates and filter by distance
            unique_venues = self._deduplicate_and_filter(venues, center_point)

            # Sort by relevance (rating, distance, review count)
            sorted_venues = self._sort_by_relevance(unique_venues)

            # Limit results
            final_venues = sorted_venues[:self.max_results]

            logger.info(f"✅ Found {len(final_venues)} venues near location")
            return final_venues

        except Exception as e:
            logger.error(f"❌ Error searching nearby venues: {e}")
            return []

    def _text_search_nearby(self, center_point: Tuple[float, float], query: str) -> List[VenueResult]:
        """Perform Google Places text search with location bias"""
        try:
            lat, lng = center_point
            location_bias = f"circle:{self.search_radius}@{lat},{lng}"

            logger.debug(f"Text search: '{query}' with location bias")

            response = self.gmaps.places(
                query=query,
                location=(lat, lng),
                radius=self.search_radius,
                type="restaurant|bar|cafe|food",
                language="en"
            )

            venues = []
            for result in response.get('results', []):
                venue = self._parse_place_result(result, center_point)
                if venue:
                    venues.append(venue)

            logger.debug(f"Text search found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []

    def _nearby_search_by_type(self, center_point: Tuple[float, float], venue_type: str) -> List[VenueResult]:
        """Perform Google Places nearby search by venue type"""
        try:
            lat, lng = center_point

            # Map venue types to Google Places types
            type_mapping = {
                "restaurant": "restaurant",
                "bar": "bar", 
                "cafe": "cafe",
                "coffee": "cafe",
                "wine": "bar",
                "cocktail": "bar",
                "bakery": "bakery"
            }

            google_type = type_mapping.get(venue_type.lower(), "restaurant")

            logger.debug(f"Nearby search: type '{google_type}' near {lat}, {lng}")

            response = self.gmaps.places_nearby(
                location=(lat, lng),
                radius=self.search_radius,
                type=google_type,
                language="en"
            )

            venues = []
            for result in response.get('results', []):
                venue = self._parse_place_result(result, center_point)
                if venue:
                    venues.append(venue)

            logger.debug(f"Nearby search found {len(venues)} venues")
            return venues

        except Exception as e:
            logger.error(f"Error in nearby search: {e}")
            return []

    def _parse_place_result(self, result: Dict[str, Any], center_point: Tuple[float, float]) -> Optional[VenueResult]:
        """Parse a Google Places API result into a VenueResult"""
        try:
            # Required fields
            name = result.get('name')
            place_id = result.get('place_id')
            geometry = result.get('geometry', {})
            location = geometry.get('location', {})

            if not all([name, place_id, location.get('lat'), location.get('lng')]):
                logger.debug(f"Skipping incomplete result: {name}")
                return None

            lat = float(location['lat'])
            lng = float(location['lng'])

            # Calculate distance from search center
            distance_km = LocationUtils.calculate_distance(center_point, (lat, lng))

            # Skip if too far (should be filtered by API but double-check)
            if distance_km > (self.search_radius / 1000):
                logger.debug(f"Skipping {name} - too far ({distance_km:.1f}km)")
                return None

            # Extract additional fields
            address = result.get('formatted_address', result.get('vicinity', ''))
            rating = result.get('rating')
            user_ratings_total = result.get('user_ratings_total')
            price_level = result.get('price_level')
            types = result.get('types', [])

            venue = VenueResult(
                name=name,
                place_id=place_id,
                address=address,
                latitude=lat,
                longitude=lng,
                rating=rating,
                user_ratings_total=user_ratings_total,
                price_level=price_level,
                types=types,
                distance_km=distance_km
            )

            logger.debug(f"Parsed venue: {name} ({distance_km:.2f}km, rating: {rating})")
            return venue

        except Exception as e:
            logger.error(f"Error parsing place result: {e}")
            return None

    def _deduplicate_and_filter(self, venues: List[VenueResult], center_point: Tuple[float, float]) -> List[VenueResult]:
        """Remove duplicates and apply quality filters"""
        # Remove duplicates by place_id
        seen_ids = set()
        unique_venues = []

        for venue in venues:
            if venue.place_id not in seen_ids:
                seen_ids.add(venue.place_id)
                unique_venues.append(venue)

        # Filter by quality criteria
        filtered_venues = []
        for venue in unique_venues:
            # Skip venues with very low ratings (if they have ratings)
            if venue.rating is not None and venue.rating < 3.0:
                logger.debug(f"Filtering out {venue.name} - low rating ({venue.rating})")
                continue

            # Skip venues with very few reviews (potential spam)
            if venue.user_ratings_total is not None and venue.user_ratings_total < 5:
                logger.debug(f"Filtering out {venue.name} - too few reviews ({venue.user_ratings_total})")
                continue

            filtered_venues.append(venue)

        logger.debug(f"After deduplication and filtering: {len(filtered_venues)} venues")
        return filtered_venues

    def _sort_by_relevance(self, venues: List[VenueResult]) -> List[VenueResult]:
        """Sort venues by relevance score (rating, distance, review count)"""
        def relevance_score(venue: VenueResult) -> float:
            score = 0.0

            # Rating component (0-5 scale, weight: 40%)
            if venue.rating:
                score += venue.rating * 0.4
            else:
                score += 3.0 * 0.4  # Default neutral rating

            # Distance component (closer is better, weight: 30%)
            if venue.distance_km:
                # Inverse distance score (max 2km radius)
                distance_score = max(0, (2.0 - venue.distance_km) / 2.0) * 1.5
                score += distance_score * 0.3

            # Review count component (more reviews = more reliable, weight: 30%)
            if venue.user_ratings_total:
                # Logarithmic scale for review count (max benefit at ~1000 reviews)
                import math
                review_score = min(1.5, math.log10(venue.user_ratings_total + 1) / 3)
                score += review_score * 0.3

            return score

        sorted_venues = sorted(venues, key=relevance_score, reverse=True)

        # Log top results for debugging
        for i, venue in enumerate(sorted_venues[:5]):
            score = relevance_score(venue)
            logger.debug(f"#{i+1}: {venue.name} (score: {score:.2f}, rating: {venue.rating}, distance: {venue.distance_km:.2f}km)")

        return sorted_venues

    def get_venue_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific venue"""
        try:
            logger.debug(f"Getting details for place_id: {place_id}")

            # Fields to retrieve
            fields = [
                'name', 'formatted_address', 'formatted_phone_number',
                'website', 'rating', 'user_ratings_total', 'price_level',
                'opening_hours', 'geometry', 'types', 'photos'
            ]

            response = self.gmaps.place(
                place_id=place_id,
                fields=fields,
                language="en"
            )

            return response.get('result', {})

        except Exception as e:
            logger.error(f"Error getting venue details for {place_id}: {e}")
            return None

    def determine_venue_type(self, query: str) -> Optional[str]:
        """Determine venue type from search query for targeted search"""
        query_lower = query.lower()

        type_keywords = {
            "coffee": ["coffee", "cafe", "espresso", "cappuccino", "latte"],
            "wine": ["wine", "vineyard", "sommelier", "natural wine", "wine bar"],
            "cocktail": ["cocktail", "mixology", "speakeasy", "gin", "whiskey", "bourbon"],
            "bar": ["bar", "pub", "tavern", "brewery", "beer"],
            "bakery": ["bakery", "pastry", "bread", "croissant", "patisserie"],
            "restaurant": ["restaurant", "dining", "cuisine", "food"]
        }

        for venue_type, keywords in type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.debug(f"Detected venue type '{venue_type}' from query: {query}")
                return venue_type

        logger.debug(f"No specific venue type detected from query: {query}")
        return None