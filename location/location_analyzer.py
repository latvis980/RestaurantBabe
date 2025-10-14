# location/location_analyzer.py
"""
Location Analyzer Agent - FIXED VERSION

Works with the existing conversation AI to detect and analyze location-based requests.
Integrates with the current LangChain pipeline architecture.

FIXES:
- Examples now ALWAYS include city context
- Added stored_city_context parameter for context enrichment
- Better handling of Portugal/Lisbon locations
- Consistent city preservation
"""

import logging
import json
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationAnalyzer:
    """
    Analyzes messages to determine if they're location-based requests
    and extracts relevant information for the location search pipeline

    ENHANCED: Can use stored city context to enrich partial locations
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for location analysis
        self.ai = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,  # Lower temperature for more consistent analysis
            api_key=config.OPENAI_API_KEY
        )

        # Location analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_analysis_system_prompt()),
            ("human", """User message: {user_message}

CONTEXT ENRICHMENT:
Stored city context from recent search: {stored_city_context}

If the user mentions only a neighborhood/district and there's stored city context, enrich the location with the city.""")
        ])

        # Create the analysis chain
        self.analysis_chain = self.analysis_prompt | self.ai

        logger.info("✅ Location Analyzer initialized with context enrichment")

    def _get_analysis_system_prompt(self) -> str:
        """Get the enhanced system prompt for location analysis with ambiguity detection"""
        return """
    You are a location request analyzer for a restaurant recommendation system with ambiguity detection and context enrichment.

    Your job is to determine if a user message is requesting location-based (GPS/proximity/neighbourhood/street/landmark) search vs. general city-wide search, 
    AND detect if the location mentioned is ambiguous.

    CRITICAL DISTINCTION:
    - LOCATION_SEARCH = GPS-based proximity search (neighborhoods, "near me", specific addresses)
    - GENERAL_SEARCH = City-wide search using existing database/web pipeline

    LOCATION-BASED REQUEST INDICATORS (GPS/proximity search):
    - "near me", "nearby", "around here", "close to", "walking distance"
    - Neighborhoods/districts: "in SoHo", "in Chinatown", "in the Mission", "Palermo, Buenos Aires", "Le Marais", "Kreuzberg", etc.
    - Street names: "on Broadway", "near Times Square"
    - landmarks: "near the Eiffel Tower", "next to the Colosseum", "around St Paul's Cathedral"
    - "where I am", "local", "in this area", "in the neighborhood"
    - GPS coordinates or specific addresses

    GENERAL SEARCH INDICATORS (city-wide search):
    - City names: "in Paris", "in London", "in Newcastle", "in Tokyo"
    - Countries: "in France", "in Italy"
    - Vast areas, though not formally cities: "in Manhattan", "in Barcelona", "in Tuscany", "around Lake Como"

    CONTEXT ENRICHMENT RULES:
    - If user mentions ONLY a neighborhood/district (e.g., "Lapa", "Alfama", "Chiado")
    - AND there's stored city context from a recent search (e.g., "Lisbon")
    - ENRICH the location: "Lapa" → "Lapa, Lisbon"
    - Document this in the reasoning field

    DEFAULT CITY CONTEXT:
    - For Portugal neighborhoods without city context: Default to Lisbon
    - Examples: "Lapa" → "Lapa, Lisbon", "Belem" → "Belem, Lisbon", "Alfama" → "Alfama, Lisbon"

    AMBIGUITY DETECTION:
    Detect if the location mentioned could refer to multiple places:
    - Common neighborhood names that exist in multiple cities
    - Place names without clear city context that could be ambiguous
    - Well-known landmarks should include obvious city context

    RESPONSE FORMAT (JSON only):
    {{
        "request_type": "LOCATION_SEARCH" | "REQUEST_LOCATION" | "GENERAL_SEARCH" | "NOT_RESTAURANT",
        "location_detected": "specific location if found" | null,
        "city_context": "inferred city when obvious" | null,
        "is_ambiguous": true | false,
        "ambiguity_reason": "why ambiguous" | null,
        "cuisine_preference": "extracted cuisine/type preference" | null,
        "confidence": 0.1-1.0,
        "reasoning": "brief explanation including context enrichment if applied",
        "suggested_response": "what bot should ask user next",
        "context_enrichment_applied": true | false
    }}

    EXAMPLES WITH CITY CONTEXT (ALWAYS PRESERVE/ADD CITY):

    "natural wine bars in SoHo, New York" →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "SoHo, New York", 
        "city_context": "New York",
        "is_ambiguous": false,
        "ambiguity_reason": null,
        "cuisine_preference": "natural wine bars",
        "confidence": 0.9,
        "reasoning": "SoHo with city context - clear location",
        "suggested_response": "I'll search for natural wine bars in SoHo, NYC for you!",
        "context_enrichment_applied": false
    }}

    "restaurants in Lapa" (with stored_city_context: "Lisbon") →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "Lapa, Lisbon",
        "city_context": "Lisbon",
        "is_ambiguous": false,
        "ambiguity_reason": null,
        "cuisine_preference": "restaurants",
        "confidence": 0.9,
        "reasoning": "Lapa enriched with stored city context (Lisbon) from recent search",
        "suggested_response": "I'll find restaurants in Lapa, Lisbon for you!",
        "context_enrichment_applied": true
    }}

    "coffee places in Lapa" (NO stored context) →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "Lapa, Lisbon",
        "city_context": "Lisbon",
        "is_ambiguous": false,
        "ambiguity_reason": null,
        "cuisine_preference": "coffee places",
        "confidence": 0.85,
        "reasoning": "Lapa defaulted to Lisbon (Portugal neighborhood)",
        "suggested_response": "I'll search for coffee places in Lapa, Lisbon!",
        "context_enrichment_applied": true
    }}

    "restaurants in Springfield" →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "Springfield",
        "city_context": null,
        "is_ambiguous": true,
        "ambiguity_reason": "multiple cities named Springfield",
        "cuisine_preference": "restaurants",
        "confidence": 0.8,
        "reasoning": "Springfield exists in many US states - needs clarification",
        "suggested_response": "There are multiple places called Springfield. Which state or country did you mean?",
        "context_enrichment_applied": false
    }}

    "specialty coffee in Alfama" (with stored_city_context: "Lisbon") →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "Alfama, Lisbon",
        "city_context": "Lisbon",
        "is_ambiguous": false,
        "ambiguity_reason": null,
        "cuisine_preference": "specialty coffee",
        "confidence": 0.95,
        "reasoning": "Alfama enriched with stored Lisbon context",
        "suggested_response": "I'll find specialty coffee places in Alfama, Lisbon!",
        "context_enrichment_applied": true
    }}
    """

    def analyze_message(self, user_message: str, stored_city_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze user message to determine request type and extract location

        Args:
            user_message: The user's message text
            stored_city_context: Optional city from recent searches for context enrichment
                                Example: "Lisbon", "New York", "Tokyo"

        Returns:
            Dict with analysis results including enriched location
        """
        try:
            logger.info(f"🔍 Analyzing message: '{user_message}'")
            if stored_city_context:
                logger.info(f"   📍 Stored city context: '{stored_city_context}'")

            # Prepare context for enrichment
            context_str = stored_city_context or "None"

            # Get AI analysis
            response = self.analysis_chain.invoke({
                "user_message": user_message,
                "stored_city_context": context_str
            })

            # Parse JSON response
            analysis = self._parse_response(response.content)

            # Log if context enrichment was applied
            if analysis.get("context_enrichment_applied"):
                original_loc = user_message
                enriched_loc = analysis.get("location_detected")
                logger.info(f"✨ Context enrichment: '{original_loc}' → '{enriched_loc}'")

            # Add original message for reference
            analysis["original_message"] = user_message

            logger.info(f"✅ Analysis complete: {analysis.get('request_type')} - {analysis.get('location_detected')}")

            return analysis

        except Exception as e:
            logger.error(f"❌ Error analyzing message: {e}", exc_info=True)
            return {
                "request_type": "NOT_RESTAURANT",
                "location_detected": None,
                "city_context": None,
                "is_ambiguous": False,
                "ambiguity_reason": None,
                "cuisine_preference": None,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "suggested_response": "I had trouble understanding that. Could you rephrase?",
                "original_message": user_message,
                "context_enrichment_applied": false
            }

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response (handles markdown code blocks)"""
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nResponse: {response_text}")
            raise


    def validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate analysis result has required fields"""
        required_fields = ["request_type", "location_detected", "confidence", "reasoning"]
        return all(field in analysis for field in required_fields)