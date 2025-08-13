# location/telegram_location_handler.py
"""
Telegram Location Handler - MOVED to location folder

Handles location-based messages from Telegram users:
1. GPS coordinates (location pins)
2. Natural location descriptions
3. Location request prompts
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LocationData:
    """Structure for location information"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None
    address: Optional[str] = None
    location_type: str = "unknown"  # "gps", "description", "address"
    confidence: float = 0.0

class TelegramLocationHandler:
    """
    Handles different types of location input from Telegram users
    """

    def __init__(self):
        # Keywords that suggest location-based requests
        self.location_keywords = [
            "near me", "nearby", "around here", "in this area",
            "close to", "walking distance", "in the neighborhood",
            "in this neighborhood", "around", "local", "where I am"
        ]

        # Common location indicators
        self.location_indicators = [
            "street", "avenue", "road", "plaza", "square", "district",
            "neighborhood", "area", "zone", "quarter", "block"
        ]

        logger.info("✅ Telegram Location Handler initialized")

    def detect_location_request(self, message_text: str) -> bool:
        """
        Detect if a message is requesting location-based search

        Args:
            message_text: The user's message text

        Returns:
            bool: True if this appears to be a location-based request
        """
        text_lower = message_text.lower()

        # Check for direct location keywords
        for keyword in self.location_keywords:
            if keyword in text_lower:
                return True

        # Check for location indicators + food/restaurant terms
        has_location_indicator = any(indicator in text_lower for indicator in self.location_indicators)
        has_food_terms = any(term in text_lower for term in [
            "restaurant", "bar", "cafe", "coffee", "wine", "food", 
            "eat", "drink", "dining", "brunch", "lunch", "dinner"
        ])

        if has_location_indicator and has_food_terms:
            return True

        # Check for patterns like "X in [location]"
        location_patterns = [
            r'\b(in|at|on|near)\s+[A-Z][a-z]+',  # "in Chinatown", "near Broadway"
            r'\b(restaurant|bar|cafe)\s+(in|at|on|near)',  # "restaurant in..."
        ]

        for pattern in location_patterns:
            if re.search(pattern, message_text):
                return True

        return False

    def extract_location_from_text(self, message_text: str) -> LocationData:
        """
        Extract location information from text description

        Args:
            message_text: The user's message text

        Returns:
            LocationData: Extracted location information
        """
        # Extract potential address/location from text
        location_patterns = [
            # "in [Location]" pattern
            r'\bin\s+([A-Z][a-zA-Z\s,]+?)(?:\s|$|[.!?])',
            # "near [Location]" pattern  
            r'\bnear\s+([A-Z][a-zA-Z\s,]+?)(?:\s|$|[.!?])',
            # "at [Location]" pattern
            r'\bat\s+([A-Z][a-zA-Z\s,]+?)(?:\s|$|[.!?])',
            # Street address pattern
            r'(\d+\s+[A-Z][a-zA-Z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Plaza|Square))',
        ]

        extracted_location = None
        confidence = 0.0

        for pattern in location_patterns:
            matches = re.findall(pattern, message_text, re.IGNORECASE)
            if matches:
                # Take the longest match (likely most specific)
                extracted_location = max(matches, key=len).strip()
                confidence = 0.8 if len(extracted_location) > 10 else 0.6
                break

        if not extracted_location:
            # Fallback: look for capitalized words that might be locations
            words = message_text.split()
            potential_locations = []

            for i, word in enumerate(words):
                if (word[0].isupper() and len(word) > 2 and 
                    word.lower() not in ['I', 'The', 'A', 'An', 'This', 'That']):
                    # Check if next word is also capitalized (compound location)
                    if i + 1 < len(words) and words[i + 1][0].isupper():
                        potential_locations.append(f"{word} {words[i + 1]}")
                    else:
                        potential_locations.append(word)

            if potential_locations:
                # Take the longest potential location
                extracted_location = max(potential_locations, key=len)
                confidence = 0.4

        return LocationData(
            description=extracted_location,
            location_type="description" if extracted_location else "unknown",
            confidence=confidence
        )

    def process_gps_location(self, latitude: float, longitude: float) -> LocationData:
        """
        Process GPS coordinates from Telegram location message

        Args:
            latitude: GPS latitude
            longitude: GPS longitude

        Returns:
            LocationData: Processed GPS location data
        """
        return LocationData(
            latitude=latitude,
            longitude=longitude,
            location_type="gps",
            confidence=1.0
        )

    def create_location_request_message(self, query_type: str = "general") -> str:
        """
        Create a message asking user for their location

        Args:
            query_type: Type of search (for customized messaging)

        Returns:
            str: Formatted message requesting location
        """
        if query_type == "wine":
            emoji = "🍷"
            context = "wine bars and natural wine spots"
        elif query_type == "coffee":
            emoji = "☕"
            context = "coffee shops and cafes"
        elif query_type == "fine_dining":
            emoji = "🍽️"
            context = "fine dining restaurants"
        else:
            emoji = "📍"
            context = "restaurants and bars"

        message = (
            f"{emoji} <b>Perfect! I'd love to help you find great {context} near you.</b>\n\n"
            "To give you the best recommendations, I need to know where you are:\n\n"
            "📌 <b>Option 1:</b> Send your location pin (tap the 📎 attachment button → Location)\n"
            "🗺️ <b>Option 2:</b> Tell me your neighborhood, street, or nearby landmark\n\n"
            "<i>Examples: \"I'm in Chinatown\", \"Near Times Square\", \"On Rua da Rosa in Lisbon\"</i>\n\n"
            "💡 <b>Don't worry:</b> I only use your location to find nearby places. I don't store it."
        )

        return message

    def validate_gps_coordinates(self, latitude: float, longitude: float) -> bool:
        """
        Validate GPS coordinates are reasonable

        Args:
            latitude: GPS latitude (-90 to 90)
            longitude: GPS longitude (-180 to 180)

        Returns:
            bool: True if coordinates are valid
        """
        return (
            -90 <= latitude <= 90 and 
            -180 <= longitude <= 180 and
            not (latitude == 0 and longitude == 0)  # Null Island check
        )

    def format_location_summary(self, location_data: LocationData) -> str:
        """
        Create a human-readable summary of location data

        Args:
            location_data: LocationData object

        Returns:
            str: Formatted location summary
        """
        if location_data.location_type == "gps":
            return f"GPS: {location_data.latitude:.4f}, {location_data.longitude:.4f}"
        elif location_data.location_type == "description":
            return f"Area: {location_data.description}"
        elif location_data.location_type == "address":
            return f"Address: {location_data.address}"
        else:
            return "Location: Unknown"