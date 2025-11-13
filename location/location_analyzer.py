# location/location_analyzer.py
"""
SIMPLIFIED LocationAnalyzer - Pure Utility (No Conversation Logic)

This is now a PURE UTILITY that only:
1. Extracts location data from text
2. Geocodes addresses
3. Returns technical location information

ALL conversation logic (ambiguity detection, clarification, search mode detection)
has been moved to AI Chat Layer as the single source of truth.
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class LocationAnalyzer:
    """
    SIMPLIFIED: Pure location extraction utility
    
    Responsibilities (ONLY technical extraction):
    - Extract location mentions from text
    - Identify location types (neighborhood, street, landmark, city)
    - Provide confidence scores
    
    NO LONGER RESPONSIBLE FOR:
    - Conversation logic (moved to AI Chat Layer)
    - Ambiguity detection in conversation (moved to AI Chat Layer)
    - Search mode detection (moved to AI Chat Layer)
    - Clarification questions (moved to AI Chat Layer)
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for location extraction only
        self.ai = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )

        # Location extraction prompt (simplified)
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_extraction_system_prompt()),
            ("human", "Extract location from: {user_message}")
        ])

        self.extraction_chain = self.extraction_prompt | self.ai

        logger.info("✅ LocationAnalyzer initialized (simplified utility)")

    def _get_extraction_system_prompt(self) -> str:
        """System prompt for pure location extraction"""
        return """You are a location extraction utility. Extract location information from text.

Your ONLY job: Identify and extract location mentions with metadata.

WHAT TO EXTRACT:
- Location name (neighborhood, street, landmark, city, country)
- Location type (neighborhood, street, landmark, city, country)
- Confidence score (0.0-1.0)

WHAT NOT TO DO:
- NO conversation logic
- NO ambiguity detection
- NO clarification questions
- NO search mode detection

LOCATION TYPES:
- "neighborhood": SoHo, Chinatown, Lapa, Alfama, etc.
- "street": Broadway, Rua Augusta, Viale delle Egadi, etc.
- "landmark": Eiffel Tower, Times Square, Colosseum, etc.
- "city": Tokyo, Paris, Lisbon, New York, etc.
- "country": France, Italy, Portugal, etc.

RESPONSE FORMAT (JSON only):
{{
    "location_detected": "extracted location string",
    "location_type": "neighborhood" | "street" | "landmark" | "city" | "country" | "unknown",
    "confidence": 0.0-1.0,
    "components": {{
        "neighborhood": "name" | null,
        "city": "name" | null,
        "country": "name" | null
    }}
}}

EXAMPLES:

"restaurants in Lapa, Lisbon"
→ {{
    "location_detected": "Lapa, Lisbon",
    "location_type": "neighborhood",
    "confidence": 0.95,
    "components": {{"neighborhood": "Lapa", "city": "Lisbon", "country": null}}
}}

"Find good places around Viale delle Egadi in Rome"
→ {{
    "location_detected": "Viale delle Egadi, Rome",
    "location_type": "street",
    "confidence": 0.9,
    "components": {{"neighborhood": null, "city": "Rome", "country": null}}
}}

"best ramen in Tokyo"
→ {{
    "location_detected": "Tokyo",
    "location_type": "city",
    "confidence": 1.0,
    "components": {{"neighborhood": null, "city": "Tokyo", "country": null}}
}}

"coffee near Eiffel Tower"
→ {{
    "location_detected": "Eiffel Tower",
    "location_type": "landmark",
    "confidence": 0.95,
    "components": {{"neighborhood": null, "city": "Paris", "country": null}}
}}

"restaurants in Springfield"
→ {{
    "location_detected": "Springfield",
    "location_type": "city",
    "confidence": 0.7,
    "components": {{"neighborhood": null, "city": "Springfield", "country": null}}
}}

Note: This is PURE EXTRACTION. Don't worry about ambiguity or clarification - just extract what's mentioned.
"""

    async def extract_location(self, user_message: str) -> Dict[str, Any]:
        """
        Extract location information from user message
        
        Returns technical location data only - NO conversation logic
        """
        try:
            response = await self.extraction_chain.ainvoke({
                "user_message": user_message
            })

            if hasattr(response, 'content'):
                response_content = str(response.content)
            else:
                response_content = str(response)

            # Parse JSON response
            cleaned = response_content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            logger.info(f"📍 Extracted location: '{result.get('location_detected')}' "
                       f"(type: {result.get('location_type')}, confidence: {result.get('confidence')})")

            return result

        except Exception as e:
            logger.error(f"Error extracting location: {e}")
            return {
                "location_detected": None,
                "location_type": "unknown",
                "confidence": 0.0,
                "components": {}
            }

    def geocode_location(self, location_string: str) -> Optional[Tuple[float, float]]:
        """
        Geocode a location string to coordinates
        
        This is a placeholder - in production, use a real geocoding service
        """
        # TODO: Integrate with actual geocoding service (Google Maps, Nominatim, etc.)
        logger.warning(f"Geocoding not implemented for: {location_string}")
        return None

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate coordinate ranges"""
        return -90 <= lat <= 90 and -180 <= lon <= 180
