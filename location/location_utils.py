# location/location_utils.py
"""
Location Utilities - MOVED to location folder

Provides utility functions for location-based operations:
- Distance calculations
- Coordinate validation
- Location data structures
"""

import logging
import math
from typing import Tuple, Optional, List, Any, Dict
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
    def extract_coordinates_from_state(state) -> Tuple[Optional[Tuple[float, float]], str]:
        """
        Extract and validate coordinates from unified search state.

        Priority order:
        1. GPS coordinates from state (user shared via button)
        2. Coordinates in SearchContext (geocoded by AI Chat Layer)
        3. Coordinates in location_data (fallback)

        Args:
            state: UnifiedSearchState dictionary

        Returns:
            Tuple of (coordinates, location_description)
            - coordinates: (lat, lng) tuple or None if not found
            - location_description: Human-readable description of location
        """
        coordinates = None
        location_description = "Unknown location"

        # Priority 1: GPS coordinates directly in state
        gps_coords = state.get("gps_coordinates")
        if gps_coords and len(gps_coords) == 2:
            try:
                lat, lng = float(gps_coords[0]), float(gps_coords[1])
                if LocationUtils.validate_coordinates(lat, lng):
                    coordinates = (lat, lng)
                    location_description = f"GPS: {lat:.4f}, {lng:.4f}"
                    logger.info(f"✅ Using GPS coordinates: {lat:.4f}, {lng:.4f}")
                    return coordinates, location_description
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid GPS coordinates in state: {e}")

        # Priority 2: SearchContext coordinates (geocoded by AI Chat Layer)
        search_context = state.get("search_context")
        if search_context:
            # Check for gps_coordinates attribute
            ctx_coords = getattr(search_context, 'gps_coordinates', None)
            if ctx_coords and len(ctx_coords) == 2:
                try:
                    lat, lng = float(ctx_coords[0]), float(ctx_coords[1])
                    if LocationUtils.validate_coordinates(lat, lng):
                        coordinates = (lat, lng)
                        # Use destination from context if available
                        destination = getattr(search_context, 'destination', None)
                        location_description = destination or f"GPS: {lat:.4f}, {lng:.4f}"
                        logger.info(f"✅ Using SearchContext coordinates: {lat:.4f}, {lng:.4f}")
                        return coordinates, location_description
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid SearchContext coordinates: {e}")

            # Also check 'coordinates' attribute (alternative naming)
            ctx_coords = getattr(search_context, 'coordinates', None)
            if ctx_coords and len(ctx_coords) == 2:
                try:
                    lat, lng = float(ctx_coords[0]), float(ctx_coords[1])
                    if LocationUtils.validate_coordinates(lat, lng):
                        coordinates = (lat, lng)
                        destination = getattr(search_context, 'destination', None)
                        location_description = destination or f"GPS: {lat:.4f}, {lng:.4f}"
                        logger.info(f"✅ Using SearchContext.coordinates: {lat:.4f}, {lng:.4f}")
                        return coordinates, location_description
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid SearchContext.coordinates: {e}")

        # Priority 3: location_data object
        location_data = state.get("location_data")
        if location_data is not None:
            # Try gps_coordinates attribute
            if hasattr(location_data, 'gps_coordinates') and location_data.gps_coordinates:
                gps = location_data.gps_coordinates
                if hasattr(gps, 'latitude') and hasattr(gps, 'longitude'):
                    try:
                        lat, lng = float(gps.latitude), float(gps.longitude)
                        if LocationUtils.validate_coordinates(lat, lng):
                            coordinates = (lat, lng)
                            location_description = getattr(location_data, 'description', f"GPS: {lat:.4f}, {lng:.4f}")
                            logger.info(f"✅ Using location_data.gps_coordinates: {lat:.4f}, {lng:.4f}")
                            return coordinates, location_description
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid location_data.gps_coordinates: {e}")

            # Try direct latitude/longitude attributes
            if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                if location_data.latitude is not None and location_data.longitude is not None:
                    try:
                        lat, lng = float(location_data.latitude), float(location_data.longitude)
                        if LocationUtils.validate_coordinates(lat, lng):
                            coordinates = (lat, lng)
                            location_description = getattr(location_data, 'description', f"GPS: {lat:.4f}, {lng:.4f}")
                            logger.info(f"✅ Using location_data coordinates: {lat:.4f}, {lng:.4f}")
                            return coordinates, location_description
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid location_data coordinates: {e}")

        # No valid coordinates found
        logger.warning("⚠️ No valid coordinates found in state")
        return None, location_description
    
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

            a = (math.sin(dlat / 2) ** 2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * 
                 math.sin(dlon / 2) ** 2)

            c = 2 * math.asin(math.sqrt(a))
            distance = LocationUtils.EARTH_RADIUS_KM * c

            return distance

        except Exception as e:
            logger.error(f"❌ Error calculating distance: {e}")
            return 0.0

    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> bool:
        """
        Validate GPS coordinates

        Args:
            latitude: GPS latitude (-90 to 90)
            longitude: GPS longitude (-180 to 180)

        Returns:
            bool: True if coordinates are valid
        """
        try:
            return (
                -90 <= float(latitude) <= 90 and 
                -180 <= float(longitude) <= 180 and
                not (latitude == 0 and longitude == 0)  # Null Island check
            )
        except (ValueError, TypeError):
            return False

    @staticmethod
    def format_distance(distance_km: float) -> str:
        """
        Format distance for human-readable display

        Args:
            distance_km: Distance in kilometers

        Returns:
            str: Formatted distance string
        """
        try:
            if distance_km < 0.1:
                return "< 100m"
            elif distance_km < 1.0:
                meters = int(distance_km * 1000)
                return f"{meters}m"
            else:
                return f"{distance_km:.1f}km"
        except (ValueError, TypeError):
            return "Distance unknown"

    @staticmethod
    def find_center_point(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Find the center point of a list of coordinates

        Args:
            points: List of (latitude, longitude) tuples

        Returns:
            Optional[Tuple[float, float]]: Center point coordinates or None if invalid
        """
        try:
            if not points:
                return None

            # Validate all points
            valid_points = []
            for lat, lng in points:
                if LocationUtils.validate_coordinates(lat, lng):
                    valid_points.append((lat, lng))

            if not valid_points:
                return None

            # Calculate center
            avg_lat = sum(point[0] for point in valid_points) / len(valid_points)
            avg_lng = sum(point[1] for point in valid_points) / len(valid_points)

            return (avg_lat, avg_lng)

        except Exception as e:
            logger.error(f"❌ Error finding center point: {e}")
            return None

    @staticmethod
    def geocode_location(address: str) -> Optional[Tuple[float, float]]:
        """Wrapper for centralized geocoding service"""
        try:
            from location.geocoding import geocode_location as geocode
            return geocode(address)
        except Exception as e:
            logger.error(f"Error geocoding '{address}': {e}")
            return None
    
    @staticmethod
    def is_within_radius(
        center: Tuple[float, float], 
        point: Tuple[float, float], 
        radius_km: float
    ) -> bool:
        """
        Check if a point is within a given radius of a center point

        Args:
            center: (latitude, longitude) of center point
            point: (latitude, longitude) of point to check
            radius_km: Radius in kilometers

        Returns:
            bool: True if point is within radius
        """
        try:
            distance = LocationUtils.calculate_distance(center, point)
            return distance <= radius_km
        except Exception:
            return False

    @staticmethod
    def sort_by_distance(
        points: List[Dict[str, Any]], 
        center: Tuple[float, float],
        lat_key: str = 'latitude',
        lng_key: str = 'longitude'
    ) -> List[Dict[str, Any]]:
        """
        Sort a list of points by distance from a center point

        Args:
            points: List of dictionaries containing coordinate data
            center: (latitude, longitude) of center point
            lat_key: Key name for latitude in the dictionaries
            lng_key: Key name for longitude in the dictionaries

        Returns:
            List[Dict[str, Any]]: Sorted list with distance info added
        """
        try:
            points_with_distance = []

            for point in points:
                point_copy = point.copy()

                try:
                    lat = float(point.get(lat_key, 0))
                    lng = float(point.get(lng_key, 0))

                    if LocationUtils.validate_coordinates(lat, lng):
                        distance = LocationUtils.calculate_distance(center, (lat, lng))
                        point_copy['distance_km'] = round(distance, 2)
                        point_copy['distance_text'] = LocationUtils.format_distance(distance)
                    else:
                        point_copy['distance_km'] = float('inf')
                        point_copy['distance_text'] = "Distance unknown"

                except (ValueError, TypeError):
                    point_copy['distance_km'] = float('inf')
                    point_copy['distance_text'] = "Distance unknown"

                points_with_distance.append(point_copy)

            # Sort by distance
            points_with_distance.sort(key=lambda x: x.get('distance_km', float('inf')))

            return points_with_distance

        except Exception as e:
            logger.error(f"❌ Error sorting by distance: {e}")
            return points