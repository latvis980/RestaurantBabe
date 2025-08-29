# location/location_map_search.py - CORRECT Places API v1 imports for Replit

"""
Google Maps/Places Search Agent - FIXED imports for Replit

The issue was incorrect import paths. Places API v1 is stable in production,
this fixes the Replit development environment import issues.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import googlemaps
from location.location_utils import LocationUtils
import openai

logger = logging.getLogger(__name__)

# FIXED: Correct Places API v1 imports for Replit
try:
    from google.oauth2 import service_account
    # Correct import path based on the documentation
    from google.maps import places_v1
    from google.type import latlng_pb2
    HAS_PLACES_API = True
    logger.info("✅ Google Places API v1 imports successful")
except ImportError as e:
    logger.warning(f"⚠️  Google Places API v1 not available: {e}")
    logger.info("🔧 Try: pip install google-maps-places google-cloud-places")
    places_v1 = None
    latlng_pb2 = None
    service_account = None
    HAS_PLACES_API = False

@dataclass
class VenueSearchResult:
    """Structure for Google Maps search results"""
    place_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float
    business_status: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    google_reviews: List[Dict] = field(default_factory=list)
    search_source: str = "places_api"
    google_maps_url: str = ""

    def __post_init__(self):
        if not self.google_maps_url and self.place_id:
            self.google_maps_url = f"https://maps.google.com/maps/place/?q=place_id:{self.place_id}"

class LocationMapSearchAgent:
    """
    Google Maps/Places search agent with FIXED imports

    This version uses the correct import paths and should work in both
    Replit development and production environments.
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.rating_threshold = float(getattr(config, 'RATING_THRESHOLD', 4.3))
        self.search_radius_km = float(getattr(config, 'SEARCH_RADIUS_KM', 2.0))
        self.max_venues_to_search = int(getattr(config, 'MAX_VENUES_TO_SEARCH', 20))

        # OpenAI for query analysis
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        openai_key = getattr(config, 'OPENAI_API_KEY', None)
        if openai_key:
            openai.api_key = openai_key

        # Initialize clients
        self.places_client = None
        self.gmaps = None
        self.api_usage = {"places": 0, "gmaps": 0}

        self._initialize_clients()

        logger.info(f"✅ LocationMapSearchAgent initialized:")
        logger.info(f"   - Rating threshold: {self.rating_threshold}")
        logger.info(f"   - Search radius: {self.search_radius_km}km") 
        logger.info(f"   - Has Places API v1: {self.places_client is not None}")
        logger.info(f"   - Has GoogleMaps fallback: {self.gmaps is not None}")

    def _initialize_clients(self):
        """Initialize Google Maps clients with correct approach"""
        # GoogleMaps library (always reliable)
        api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)
        if api_key:
            try:
                self.gmaps = googlemaps.Client(key=api_key)
                logger.info("✅ GoogleMaps client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GoogleMaps client: {e}")

        # Places API v1 client (if available)
        if HAS_PLACES_API:
            try:
                # FIXED: Use correct service account initialization
                creds_path = getattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
                if creds_path and os.path.exists(creds_path):
                    credentials = service_account.Credentials.from_service_account_file(
                        creds_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )

                    # FIXED: Use correct client initialization
                    self.places_client = places_v1.PlacesClient(credentials=credentials)
                    logger.info("✅ Places API v1 client initialized")

                elif hasattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS_JSON'):
                    # Handle JSON credentials from environment
                    creds_json = getattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS_JSON', None)
                    if creds_json:
                        import json
                        creds_info = json.loads(creds_json)
                        credentials = service_account.Credentials.from_service_account_info(
                            creds_info,
                            scopes=['https://www.googleapis.com/auth/cloud-platform']
                        )
                        self.places_client = places_v1.PlacesClient(credentials=credentials)
                        logger.info("✅ Places API v1 client initialized from JSON")

            except Exception as e:
                logger.error(f"❌ Failed to initialize Places API v1: {e}")
                logger.info("🔧 Will use GoogleMaps library as fallback")

    def _log_coordinates(self, latitude: float, longitude: float, context: str):
        """Log coordinates for debugging"""
        logger.info(f"🌍 {context}: {latitude:.6f}, {longitude:.6f}")

    async def _places_api_search(
        self, 
        latitude: float, 
        longitude: float, 
        place_types: List[str]
    ) -> List[VenueSearchResult]:
        """Execute Places API v1 search with FIXED implementation"""
        venues = []

        if not self.places_client or not HAS_PLACES_API:
            logger.info("⚠️  Places API v1 not available, skipping")
            return venues

        try:
            self._log_coordinates(latitude, longitude, "Places API v1 search")
            self.api_usage["places"] += 1

            # FIXED: Create request using correct types
            center = latlng_pb2.LatLng(latitude=latitude, longitude=longitude)
            radius_m = int(self.search_radius_km * 1000)

            # Create the search request
            request = places_v1.SearchNearbyRequest(
                location_restriction=places_v1.SearchNearbyRequest.LocationRestriction(
                    circle=places_v1.Circle(
                        center=center,
                        radius=radius_m
                    )
                ),
                included_types=place_types[:5],
                max_result_count=min(20, self.max_venues_to_search),
                language_code="en",
                rank_preference=places_v1.SearchNearbyRequest.RankPreference.POPULARITY
            )

            # FIXED: Execute with correct field mask
            response = self.places_client.search_nearby(
                request=request,
                metadata=[
                    ("x-goog-fieldmask", 
                     "places.id,places.displayName,places.formattedAddress,places.location," +
                     "places.rating,places.userRatingCount,places.businessStatus")
                ]
            )

            if response.places:
                logger.info(f"✅ Places API v1 returned {len(response.places)} results")

                for place in response.places:
                    try:
                        venue = self._convert_places_result(place, latitude, longitude)
                        if venue:
                            venues.append(venue)
                    except Exception as e:
                        logger.warning(f"⚠️  Error converting Places result: {e}")

        except Exception as e:
            logger.error(f"❌ Places API v1 search failed: {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")

        logger.info(f"🎯 Places API v1 search completed: {len(venues)} venues")
        return venues

    def _convert_places_result(
        self, 
        place, 
        search_lat: float, 
        search_lng: float
    ) -> Optional[VenueSearchResult]:
        """Convert Places API v1 result with safe attribute access"""
        try:
            # Safe attribute extraction
            place_id = getattr(place, 'id', '')

            display_name = getattr(place, 'display_name', None)
            name = display_name.text if display_name and hasattr(display_name, 'text') else "Unknown"

            address = getattr(place, 'formatted_address', '')

            location = getattr(place, 'location', None)
            if not location:
                return None

            venue_lat = getattr(location, 'latitude', None)
            venue_lng = getattr(location, 'longitude', None)

            if venue_lat is None or venue_lng is None:
                return None

            # Calculate distance
            distance_km = LocationUtils.calculate_distance(
                (search_lat, search_lng), (venue_lat, venue_lng)
            )

            # Extract other fields safely
            rating = getattr(place, 'rating', None)
            user_ratings_total = getattr(place, 'user_rating_count', None)

            business_status_obj = getattr(place, 'business_status', None)
            business_status = business_status_obj.name if business_status_obj else "OPERATIONAL"

            return VenueSearchResult(
                place_id=place_id,
                name=name,
                address=address,
                latitude=venue_lat,
                longitude=venue_lng,
                distance_km=distance_km,
                business_status=business_status,
                rating=rating,
                user_ratings_total=user_ratings_total,
                search_source="places_api_v1"
            )

        except Exception as e:
            logger.error(f"❌ Error converting Places result: {e}")
            return None

    async def _googlemaps_search(
        self, 
        latitude: float, 
        longitude: float, 
        query: str
    ) -> List[VenueSearchResult]:
        """GoogleMaps library search as fallback"""
        venues = []

        if not self.gmaps:
            return venues

        try:
            self._log_coordinates(latitude, longitude, "GoogleMaps library search")
            self.api_usage["gmaps"] += 1

            location = f"{latitude},{longitude}"
            radius_m = int(self.search_radius_km * 1000)

            # Text search
            response = self.gmaps.places(
                query=f"{query} restaurant",
                location=location,
                radius=radius_m,
            )

            results = response.get('results', [])
            logger.info(f"✅ GoogleMaps returned {len(results)} results")

            for place in results:
                try:
                    venue = self._convert_gmaps_result(place, latitude, longitude)
                    if venue:
                        venues.append(venue)
                except Exception as e:
                    logger.warning(f"⚠️  Error converting GoogleMaps result: {e}")

        except Exception as e:
            logger.error(f"❌ GoogleMaps search failed: {e}")

        logger.info(f"🎯 GoogleMaps search completed: {len(venues)} venues")
        return venues

    def _convert_gmaps_result(
        self, 
        place: Dict, 
        search_lat: float, 
        search_lng: float
    ) -> Optional[VenueSearchResult]:
        """Convert GoogleMaps result"""
        try:
            location = place.get('geometry', {}).get('location', {})
            venue_lat = location.get('lat')
            venue_lng = location.get('lng')

            if venue_lat is None or venue_lng is None:
                return None

            distance_km = LocationUtils.calculate_distance(
                (search_lat, search_lng), (venue_lat, venue_lng)
            )

            return VenueSearchResult(
                place_id=place.get('place_id', ''),
                name=place.get('name', 'Unknown'),
                address=place.get('formatted_address', ''),
                latitude=venue_lat,
                longitude=venue_lng,
                distance_km=distance_km,
                business_status=place.get('business_status', 'OPERATIONAL'),
                rating=place.get('rating'),
                user_ratings_total=place.get('user_ratings_total'),
                search_source="googlemaps"
            )

        except Exception as e:
            logger.error(f"❌ Error converting GoogleMaps result: {e}")
            return None

    async def search_venues_ai_guided(
        self, 
        query: str, 
        coordinates: Tuple[float, float],
        max_results: int = 20
    ) -> List[VenueSearchResult]:
        """
        MAIN SEARCH METHOD: Uses Places API v1 with GoogleMaps fallback
        """
        try:
            latitude, longitude = coordinates

            logger.info(f"🎯 Starting search for '{query}'")
            self._log_coordinates(latitude, longitude, "INPUT coordinates")

            # Get place types for search
            place_types = await self._analyze_query_for_place_types(query)
            if not place_types:
                place_types = ["restaurant", "food", "meal_takeaway"]

            venues = []

            # Try Places API v1 first
            if self.places_client:
                venues = await self._places_api_search(latitude, longitude, place_types)

            # Fallback to GoogleMaps if needed
            if not venues and self.gmaps:
                logger.info("🔄 Falling back to GoogleMaps library")
                venues = await self._googlemaps_search(latitude, longitude, query)

            # Apply rating filter
            filtered_venues = [
                v for v in venues 
                if v.rating and v.rating >= self.rating_threshold
            ]

            # Sort and limit
            filtered_venues.sort(key=lambda x: (x.rating or 0, -x.distance_km), reverse=True)
            final_venues = filtered_venues[:max_results]

            # Final logging
            logger.info(f"🎯 Search completed: {len(final_venues)} venues")
            if final_venues:
                for i, venue in enumerate(final_venues[:5], 1):
                    logger.info(f"   {i}. {venue.name}: {venue.latitude:.6f}, {venue.longitude:.6f}")

            return final_venues

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    async def _analyze_query_for_place_types(self, query: str) -> List[str]:
        """AI analysis for place types"""
        try:
            if not openai.api_key:
                return ["restaurant", "food", "meal_takeaway"]

            prompt = f"""
            Query: "{query}"
            Return 3 Google Places API place types as JSON array.
            Choose from: restaurant, food, meal_takeaway, bakery, cafe, bar
            Example: ["restaurant", "food", "cafe"]
            """

            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )

            return json.loads(response.choices[0].message.content.strip())

        except Exception:
            return ["restaurant", "food", "meal_takeaway"]

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'has_places_api_v1': self.places_client is not None,
            'has_googlemaps_fallback': self.gmaps is not None,
            'rating_threshold': self.rating_threshold,
            'search_radius_km': self.search_radius_km,
            'api_usage': self.api_usage.copy()
        }