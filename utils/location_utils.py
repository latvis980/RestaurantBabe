# utils/location_utils.py
"""
Location Utilities

Provides utility functions for location-based operations:
- Distance calculations
- Coordinate validation
- Location data structures
"""

import logging
import math
from typing import Tuple, Optional, List. Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LocationPoint:
    """Structure for a geographical point"""
    latitude: float
    longitude: float
    name: Optional[str] = None

    def __post_init__(self):
        """Validate coordinates after initialization"""
        if not self.is_valid():
            raise ValueError(f"Invalid coordinates: ({self.latitude}, {self.longitude})")

    def is_valid(self) -> bool:
        """Check if coordinates are valid"""
        return (
            -90 <= self.latitude <= 90 and 
            -180 <= self.longitude <= 180
        )

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lat, lng) tuple"""
        return (self.latitude, self.longitude)

class LocationUtils:
    """
    Utility functions for location-based operations
    """

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def calculate_distance(
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate the distance between two GPS coordinates using Haversine formula

        Args:
            point1: (latitude, longitude) of first point
            point2: (latitude, longitude) of second point

        Returns:
            Distance in kilometers
        """
        try:
            lat1, lon1 = point1
            lat2, lon2 = point2

            # Validate coordinates
            if not LocationUtils.validate_coordinates(lat1, lon1):
                raise ValueError(f"Invalid coordinates for point1: ({lat1}, {lon1})")
            if not LocationUtils.validate_coordinates(lat2, lon2):
                raise ValueError(f"Invalid coordinates for point2: ({lat2}, {lon2})")

            # Convert to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * 
                 math.sin(dlon/2)**2)

            c = 2 * math.asin(math.sqrt(a))
            distance = LocationUtils.EARTH_RADIUS_KM * c

            return distance

        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')  # Return infinite distance on error

    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> bool:
        """
        Validate GPS coordinates

        Args:
            latitude: Latitude value
            longitude: Longitude value

        Returns:
            True if coordinates are valid
        """
        try:
            return (
                isinstance(latitude, (int, float)) and
                isinstance(longitude, (int, float)) and
                -90 <= latitude <= 90 and
                -180 <= longitude <= 180
            )
        except:
            return False

    @staticmethod
    def is_within_radius(
        center: Tuple[float, float],
        point: Tuple[float, float], 
        radius_km: float
    ) -> bool:
        """
        Check if a point is within a given radius of a center point

        Args:
            center: (lat, lng) of center point
            point: (lat, lng) of point to check
            radius_km: Radius in kilometers

        Returns:
            True if point is within radius
        """
        try:
            distance = LocationUtils.calculate_distance(center, point)
            return distance <= radius_km
        except:
            return False

    @staticmethod
    def find_nearby_points(
        center: Tuple[float, float],
        points: List[Tuple[float, float, Any]],
        radius_km: float
    ) -> List[Tuple[float, float, Any, float]]:
        """
        Find all points within a radius of a center point

        Args:
            center: (lat, lng) of center point
            points: List of (lat, lng, data) tuples
            radius_km: Search radius in kilometers

        Returns:
            List of (lat, lng, data, distance) tuples for points within radius
        """
        nearby = []

        for lat, lng, data in points:
            try:
                distance = LocationUtils.calculate_distance(center, (lat, lng))
                if distance <= radius_km:
                    nearby.append((lat, lng, data, distance))
            except Exception as e:
                logger.debug(f"Error processing point ({lat}, {lng}): {e}")
                continue

        # Sort by distance (closest first)
        nearby.sort(key=lambda x: x[3])
        return nearby

    @staticmethod
    def get_bounding_box(
        center: Tuple[float, float], 
        radius_km: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate a bounding box around a center point

        Args:
            center: (lat, lng) of center point
            radius_km: Radius in kilometers

        Returns:
            (min_lat, min_lng, max_lat, max_lng) tuple
        """
        try:
            lat, lng = center

            # Rough conversion: 1 degree ≈ 111 km
            lat_delta = radius_km / 111.0
            lng_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

            min_lat = max(-90, lat - lat_delta)
            max_lat = min(90, lat + lat_delta)
            min_lng = max(-180, lng - lng_delta)
            max_lng = min(180, lng + lng_delta)

            return (min_lat, min_lng, max_lat, max_lng)

        except Exception as e:
            logger.error(f"Error calculating bounding box: {e}")
            # Return a small box around the center as fallback
            return (lat - 0.01, lng - 0.01, lat + 0.01, lng + 0.01)

    @staticmethod
    def format_coordinates(
        latitude: float, 
        longitude: float, 
        precision: int = 6
    ) -> str:
        """
        Format coordinates as a readable string

        Args:
            latitude: Latitude value
            longitude: Longitude value
            precision: Number of decimal places

        Returns:
            Formatted coordinate string
        """
        try:
            if not LocationUtils.validate_coordinates(latitude, longitude):
                return "Invalid coordinates"

            lat_str = f"{latitude:.{precision}f}"
            lng_str = f"{longitude:.{precision}f}"

            # Add cardinal directions
            lat_dir = "N" if latitude >= 0 else "S"
            lng_dir = "E" if longitude >= 0 else "W"

            return f"{abs(float(lat_str))}°{lat_dir}, {abs(float(lng_str))}°{lng_dir}"

        except Exception as e:
            logger.error(f"Error formatting coordinates: {e}")
            return f"{latitude}, {longitude}"

    @staticmethod
    def create_google_maps_url(
        latitude: float, 
        longitude: float, 
        zoom: int = 15
    ) -> str:
        """
        Create a Google Maps URL for given coordinates

        Args:
            latitude: Latitude value
            longitude: Longitude value
            zoom: Map zoom level (1-20)

        Returns:
            Google Maps URL
        """
        try:
            if not LocationUtils.validate_coordinates(latitude, longitude):
                return ""

            return f"https://maps.google.com/maps?q={latitude},{longitude}&z={zoom}"

        except Exception as e:
            logger.error(f"Error creating Google Maps URL: {e}")
            return ""

    @staticmethod
    def format_distance(distance_km: float) -> str:
        """
        Format distance for display

        Args:
            distance_km: Distance in kilometers

        Returns:
            Formatted distance string
        """
        try:
            if distance_km < 0.1:
                return "< 100m"
            elif distance_km < 1.0:
                return f"{int(distance_km * 1000)}m"
            else:
                return f"{distance_km:.1f}km"
        except:
            return "Unknown distance"

    @staticmethod
    def generate_google_maps_url(
        latitude: float, 
        longitude: float, 
        name: str = ""
    ) -> str:
        """
        Generate Google Maps URL for a location

        Args:
            latitude: Latitude value
            longitude: Longitude value
            name: Optional place name

        Returns:
            Google Maps URL
        """
        try:
            if not LocationUtils.validate_coordinates(latitude, longitude):
                return ""

            if name:
                # URL encode the name
                import urllib.parse
                encoded_name = urllib.parse.quote(name)
                return f"https://maps.google.com/maps?q={encoded_name}@{latitude},{longitude}"
            else:
                return f"https://maps.google.com/maps?q={latitude},{longitude}"

        except Exception as e:
            logger.error(f"Error creating Google Maps URL: {e}")
            return ""

    @staticmethod
    def parse_coordinates_string(coord_string: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinates from various string formats

        Args:
            coord_string: String containing coordinates

        Returns:
            (latitude, longitude) tuple or None if parsing fails
        """
        try:
            # Remove common characters
            cleaned = coord_string.replace("(", "").replace(")", "").replace(" ", "")

            # Try comma-separated format
            if "," in cleaned:
                parts = cleaned.split(",")
                if len(parts) == 2:
                    lat = float(parts[0])
                    lng = float(parts[1])

                    if LocationUtils.validate_coordinates(lat, lng):
                        return (lat, lng)

            return None

        except Exception as e:
            logger.debug(f"Error parsing coordinates '{coord_string}': {e}")
            return None

# Export commonly used functions and classes
__all__ = [
    'LocationUtils',
    'LocationPoint'
]