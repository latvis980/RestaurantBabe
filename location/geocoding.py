# location/geocoding.py
"""
Centralized Geocoding Service

Provides geocoding with multiple provider fallback:
1. Nominatim (free, rate-limited, OpenStreetMap)
2. Google Maps Geocoding API (paid fallback)

Used by:
- AI Chat Layer (early validation for text locations)
- LocationUtils (wrapper for other modules)
- Database operations (when updating restaurant coordinates)

Example usage:
    from location.geocoding import geocode_location

    coords = geocode_location("Viale delle Egadi, Rome")
    # Returns: (41.8902, 12.4922) or None if failed
"""

import logging
import time
from typing import Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests

logger = logging.getLogger(__name__)


class GeocodingService:
    """
    Multi-provider geocoding service with automatic fallback

    Provider priority:
    1. Nominatim (OpenStreetMap) - Free, 1 req/sec limit
    2. Google Maps Geocoding API - Paid, reliable fallback
    """

    def __init__(self, config):
        """
        Initialize geocoding service

        Args:
            config: Application config with GOOGLE_MAPS_API_KEY
        """
        self.config = config

        # Initialize Nominatim
        try:
            self.nominatim = Nominatim(
                user_agent="ai-restaurant-bot",
                timeout=5
            )
            logger.info("✅ Nominatim geocoder initialized")
        except Exception as e:
            logger.warning(f"⚠️ Nominatim initialization failed: {e}")
            self.nominatim = None

        # Track last Nominatim call for rate limiting
        self._last_nominatim_call = 0
        self._nominatim_min_interval = 1.0  # 1 second between calls

    def geocode_location(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Main geocoding method with automatic fallback

        Tries providers in order:
        1. Nominatim (free)
        2. Google Maps API (paid fallback)

        Args:
            address: Address string to geocode

        Returns:
            Tuple of (latitude, longitude) or None if all providers fail

        Examples:
            >>> geocode_location("Viale delle Egadi, Rome")
            (41.8902, 12.4922)

            >>> geocode_location("Eiffel Tower")
            (48.8584, 2.2945)

            >>> geocode_location("invalid address xyz123")
            None
        """
        if not address or not address.strip():
            logger.warning("Empty address provided to geocoding service")
            return None

        address = address.strip()
        logger.info(f"🌍 Geocoding request: '{address}'")

        # Try Nominatim first (free)
        coords = self._geocode_with_nominatim(address)
        if coords:
            return coords

        # Fallback to Google Maps
        coords = self._geocode_with_google_maps(address)
        if coords:
            return coords

        # All providers failed
        logger.error(f"❌ All geocoding providers failed for: '{address}'")
        return None

    def _geocode_with_nominatim(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Geocode using Nominatim (OpenStreetMap)

        Features:
        - Free service
        - Rate limited to 1 request/second
        - Good coverage worldwide
        - Sometimes fails on specific addresses

        Args:
            address: Address to geocode

        Returns:
            Coordinates or None if failed
        """
        if not self.nominatim:
            logger.debug("Nominatim not available, skipping")
            return None

        try:
            # Rate limiting: Ensure 1 second between calls
            time_since_last_call = time.time() - self._last_nominatim_call
            if time_since_last_call < self._nominatim_min_interval:
                sleep_time = self._nominatim_min_interval - time_since_last_call
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

            # Make geocoding request
            self._last_nominatim_call = time.time()
            location = self.nominatim.geocode(address)

            if location:
                lat = float(location.latitude)
                lng = float(location.longitude)

                # Validate coordinates
                if self.validate_coordinates(lat, lng):
                    logger.info(f"✅ Nominatim: '{address}' → ({lat:.6f}, {lng:.6f})")
                    return (lat, lng)
                else:
                    logger.warning(f"⚠️ Nominatim returned invalid coordinates: ({lat}, {lng})")
                    return None

            logger.warning(f"⚠️ Nominatim: No result for '{address}'")
            return None

        except GeocoderTimedOut:
            logger.warning(f"⚠️ Nominatim timeout for '{address}'")
            return None

        except GeocoderServiceError as e:
            logger.warning(f"⚠️ Nominatim service error for '{address}': {e}")
            return None

        except Exception as e:
            logger.error(f"❌ Nominatim unexpected error for '{address}': {e}")
            return None

    def _geocode_with_google_maps(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Geocode using Google Maps Geocoding API

        Features:
        - Paid service (requires API key)
        - Highly reliable
        - Excellent coverage worldwide
        - Better at handling ambiguous addresses

        Args:
            address: Address to geocode

        Returns:
            Coordinates or None if failed
        """
        google_api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)

        if not google_api_key:
            logger.debug("Google Maps API key not configured, skipping")
            return None

        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': address,
                'key': google_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"❌ Google Maps API HTTP {response.status_code} for '{address}'")
                return None

            data = response.json()

            if data.get('status') == 'OK' and data.get('results'):
                location = data['results'][0]['geometry']['location']
                lat = float(location['lat'])
                lng = float(location['lng'])

                # Validate coordinates
                if self.validate_coordinates(lat, lng):
                    logger.info(f"✅ Google Maps: '{address}' → ({lat:.6f}, {lng:.6f})")
                    return (lat, lng)
                else:
                    logger.warning(f"⚠️ Google Maps returned invalid coordinates: ({lat}, {lng})")
                    return None

            # Handle specific error statuses
            status = data.get('status')
            if status == 'ZERO_RESULTS':
                logger.warning(f"⚠️ Google Maps: No results for '{address}'")
            elif status == 'INVALID_REQUEST':
                logger.error(f"❌ Google Maps: Invalid request for '{address}'")
            elif status == 'REQUEST_DENIED':
                logger.error(f"❌ Google Maps: Request denied (check API key)")
            elif status == 'OVER_QUERY_LIMIT':
                logger.error(f"❌ Google Maps: Query limit exceeded")
            else:
                logger.warning(f"⚠️ Google Maps: Status '{status}' for '{address}'")

            return None

        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Google Maps timeout for '{address}'")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Google Maps request error for '{address}': {e}")
            return None

        except Exception as e:
            logger.error(f"❌ Google Maps unexpected error for '{address}': {e}")
            return None

    @staticmethod
    def validate_coordinates(lat: float, lng: float) -> bool:
        """
        Validate that coordinates are within valid ranges

        Args:
            lat: Latitude (-90 to 90)
            lng: Longitude (-180 to 180)

        Returns:
            True if coordinates are valid

        Examples:
            >>> validate_coordinates(41.8902, 12.4922)
            True

            >>> validate_coordinates(100, 200)  # Invalid ranges
            False

            >>> validate_coordinates(0, 0)  # Null Island check
            False
        """
        try:
            lat = float(lat)
            lng = float(lng)

            # Check ranges
            if not (-90 <= lat <= 90):
                return False
            if not (-180 <= lng <= 180):
                return False

            # Null Island check (0, 0) is unlikely to be a real query
            if lat == 0 and lng == 0:
                return False

            return True

        except (ValueError, TypeError):
            return False


# ============================================================================
# GLOBAL INSTANCE MANAGEMENT
# ============================================================================

_geocoding_service: Optional[GeocodingService] = None


def initialize_geocoding_service(config):
    """
    Initialize the global geocoding service instance

    This should be called once at application startup, after config is loaded.

    Args:
        config: Application configuration object

    Example:
        from location.geocoding import initialize_geocoding_service
        import config

        initialize_geocoding_service(config)
    """
    global _geocoding_service

    if _geocoding_service is not None:
        logger.warning("Geocoding service already initialized")
        return

    try:
        _geocoding_service = GeocodingService(config)
        logger.info("✅ Geocoding service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize geocoding service: {e}")
        raise


def get_geocoding_service() -> GeocodingService:
    """
    Get the global geocoding service instance

    Returns:
        GeocodingService instance

    Raises:
        RuntimeError: If service not initialized

    Example:
        from location.geocoding import get_geocoding_service

        service = get_geocoding_service()
        coords = service.geocode_location("Times Square, New York")
    """
    if _geocoding_service is None:
        raise RuntimeError(
            "Geocoding service not initialized. "
            "Call initialize_geocoding_service(config) first."
        )
    return _geocoding_service


# ============================================================================
# CONVENIENCE FUNCTION (PRIMARY API)
# ============================================================================

def geocode_location(address: str) -> Optional[Tuple[float, float]]:
    """
    Geocode an address to coordinates (convenience function)

    This is the primary API for geocoding. It uses the global service instance.

    Args:
        address: Address string to geocode

    Returns:
        Tuple of (latitude, longitude) or None if geocoding fails

    Examples:
        >>> from location.geocoding import geocode_location

        >>> geocode_location("Viale delle Egadi, Rome")
        (41.8902, 12.4922)

        >>> geocode_location("Eiffel Tower, Paris")
        (48.8584, 2.2945)

        >>> geocode_location("SoHo, New York")
        (40.7233, -74.0030)

        >>> geocode_location("invalid xyz123")
        None
    """
    try:
        service = get_geocoding_service()
        return service.geocode_location(address)
    except RuntimeError:
        logger.error("Geocoding service not initialized")
        return None
    except Exception as e:
        logger.error(f"Error in geocode_location: {e}")
        return None


# ============================================================================
# VALIDATION UTILITY (EXPOSED)
# ============================================================================

def validate_coordinates(lat: float, lng: float) -> bool:
    """
    Validate coordinate ranges (convenience function)

    Args:
        lat: Latitude
        lng: Longitude

    Returns:
        True if coordinates are valid

    Example:
        >>> from location.geocoding import validate_coordinates

        >>> validate_coordinates(41.8902, 12.4922)
        True

        >>> validate_coordinates(100, 200)
        False
    """
    return GeocodingService.validate_coordinates(lat, lng)