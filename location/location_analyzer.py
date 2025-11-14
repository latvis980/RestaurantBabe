# location/location_analyzer.py
"""
FIXED LocationAnalyzer - Pure Utility with Sync Support

Changes from original:
1. Added extract_location_sync() method for sync contexts (fixes async event loop error)
2. Removed geocode_location() placeholder method (was useless)
3. Kept async extract_location() for async contexts
4. No changes to prompts or other logic
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

CONFIDENCE LEVELS:
- 1.0: Very specific (street address, GPS coordinates)
- 0.9: Specific (neighborhood with city)
- 0.7: Clear (city or well-known landmark)
- 0.5: Vague (region or general area)
- 0.3: Ambiguous (could be multiple places)

EXAMPLES:
Input: "restaurants in SoHo, New York"
Output: {"location_detected": "SoHo, New York", "location_type": "neighborhood", "confidence": 0.9}

Input: "sushi in Tokyo"
Output: {"location_detected": "Tokyo", "location_type": "city", "confidence": 0.7}

Input: "pizza on Via Adamello, Rome"
Output: {"location_detected": "Via Adamello, Rome", "location_type": "street", "confidence": 1.0}

Input: "cafes near Eiffel Tower"
Output: {"location_detected": "Eiffel Tower", "location_type": "landmark", "confidence": 0.9}

RETURN FORMAT (JSON):
{
    "location_detected": "extracted location string or null",
    "location_type": "neighborhood|street|landmark|city|country|unknown",
    "confidence": 0.0-1.0,
    "components": {
        "neighborhood": "if present",
        "city": "if present",
        "country": "if present"
    }
}

Don't worry about ambiguity or clarification - just extract what's mentioned.
"""

    async def extract_location(self, user_message: str) -> Dict[str, Any]:
        """
        Extract location information from user message (ASYNC version)
        
        Returns technical location data only - NO conversation logic
        Use this when calling from async context (like async functions).
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

    def extract_location_sync(self, user_message: str) -> Dict[str, Any]:
        """
        Extract location information from user message (SYNC version)
        
        FIXED: This method handles sync contexts properly by creating its own event loop.
        Use this when calling from sync context (like langgraph_orchestrator).
        
        This fixes the "There is no current event loop in thread" error.
        """
        try:
            import asyncio
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async extraction
                result = loop.run_until_complete(self.extract_location(user_message))
                return result
            finally:
                # Clean up the event loop
                loop.close()
                asyncio.set_event_loop(None)
                
        except Exception as e:
            logger.error(f"Error in sync location extraction: {e}")
            return {
                "location_detected": None,
                "location_type": "unknown",
                "confidence": 0.0,
                "components": {}
            }

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate coordinate ranges"""
        return -90 <= lat <= 90 and -180 <= lon <= 180


# NOTE: The geocode_location() method has been REMOVED
# It was a useless placeholder that did nothing.
# Actual geocoding is done via:
# - utils/database.py -> geocode_address() (Nominatim -> Google Maps fallback)
# - location/location_utils.py -> LocationUtils.geocode_location() (wrapper)
