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
from typing import Dict, List, Any, Optional
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

    def __init__(self, config=None):
        """Initialize with config for AI-powered location extraction"""
        self.config = config

        # Initialize AI for location extraction
        if config and hasattr(config, 'OPENAI_API_KEY'):
            try:
                from langchain_openai import ChatOpenAI
                self.ai = ChatOpenAI(
                    model=getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini'),
                    temperature=0.1,
                    api_key=config.OPENAI_API_KEY
                )
                logger.info("✅ AI-powered location extraction enabled")
            except Exception as e:
                logger.warning(f"⚠️ AI location extraction disabled: {e}")
                self.ai = None
        else:
            logger.warning("⚠️ No config provided - AI location extraction disabled")
            self.ai = None

        logger.info("✅ Telegram Location Handler initialized")


    def extract_location_from_text(self, message_text: str) -> LocationData:
        """
        AI-powered location extraction from text
        Uses AI to extract the best location keywords for Google geocoding

        Args:
            message_text: The user's message text

        Returns:
            LocationData: Extracted location information
        """
        try:
            # Use AI to extract location for geocoding
            extracted_location = self._ai_extract_location(message_text)

            if extracted_location and len(extracted_location.strip()) > 2:
                return LocationData(
                    description=extracted_location,
                    location_type="description",
                    confidence=0.8
                )
            else:
                return LocationData(
                    location_type="unknown",
                    confidence=0.0
                )

        except Exception as e:
            logger.error(f"Error in AI location extraction: {e}")
            return LocationData(
                location_type="unknown",
                confidence=0.0
            )

    def _ai_extract_location(self, message_text: str) -> str:
        """
        Use AI to extract the best location keywords for Google geocoding
        """
        try:
            if not self.ai:
                logger.warning("AI not available for location extraction")
                return ""

            from langchain_core.prompts import ChatPromptTemplate

            # Create prompt for location extraction
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a location extraction specialist. Extract the most specific location from restaurant queries for Google Maps geocoding.

    RULES:
    1. Extract the most specific location mentioned (neighborhood > street > landmark > city)
    2. Return ONLY the location keywords that will work best with Google Maps
    3. Include city context if helpful for disambiguation
    4. Remove food/restaurant words - only return location
    5. If multiple locations, choose the most specific one

    EXAMPLES:
    "restaurants in Alcantara" → "Alcantara, Lisbon"
    "wine bars in SoHo" → "SoHo, New York"
    "sushi on Rua Augusta" → "Rua Augusta, Lisbon"
    "coffee near Times Square" → "Times Square, New York"
    "bars in Chinatown" → "Chinatown"
    "places in downtown" → "downtown"
    "food in the Mission district" → "Mission district, San Francisco"
    "restaurants in Belem" → "Belem"

    Return only the location string, nothing else."""),
                ("human", "Extract location from: {{user_message}}")
            ])

            # Create chain and get result
            chain = extraction_prompt | self.ai
            response = chain.invoke({{"user_message": message_text}})

            # Extract the location from AI response
            extracted = response.content.strip()

            # Clean up any extra formatting
            extracted = extracted.strip('"\'`')

            logger.debug(f"AI extracted location: '{extracted}' from '{message_text}'")

            return extracted

        except Exception as e:
            logger.error(f"Error in AI location extraction: {e}")
            return ""

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