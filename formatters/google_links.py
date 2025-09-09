
 # formatters/google_links.py
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
from urllib.parse import quote

logger = logging.getLogger(__name__)

def build_google_maps_url(place_id: str, name: str = "") -> str:
    """Return canonical Google Maps URL for a place_id.

    Args:
        place_id: Google Maps place identifier.
        name: Optional place name to improve search queries.

    Returns:
        str: Canonical Google Maps search URL using the provided place_id.
    """
    if not place_id:
        return "#"

    query = quote(name.strip()) if name else "restaurant"
    return f"https://www.google.com/maps/search/?api=1&query={query}&query_place_id={place_id}"

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