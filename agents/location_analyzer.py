# agents/location_analyzer.py
"""
Location Analyzer Agent

Works with the existing conversation AI to detect and analyze location-based requests.
Integrates with the current LangChain pipeline architecture.
"""

import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationAnalyzer:
    """
    Analyzes messages to determine if they're location-based requests
    and extracts relevant information for the location search pipeline
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
            ("human", "User message: {user_message}")
        ])

        # Create the analysis chain
        self.analysis_chain = self.analysis_prompt | self.ai

        logger.info("✅ Location Analyzer initialized")

    def _get_analysis_system_prompt(self) -> str:
        """Get the system prompt for location analysis"""
        return """
You are a location request analyzer for a restaurant recommendation system.

Your job is to determine if a user message is requesting location-based (GPS/proximity) search vs general city-wide search.

CRITICAL DISTINCTION:
- LOCATION_SEARCH = GPS-based proximity search (neighborhoods, "near me", specific addresses)
- GENERAL_SEARCH = City-wide search using existing database/web pipeline

LOCATION-BASED REQUEST INDICATORS (GPS/proximity search):
- "near me", "nearby", "around here", "close to", "walking distance"
- Neighborhoods/districts: "in SoHo", "in Chinatown", "in the Mission"
- Street names: "on Broadway", "near Times Square"
- "where I am", "local", "in this area", "in the neighborhood"
- GPS coordinates or specific addresses

GENERAL SEARCH INDICATORS (city-wide search):
- City names: "in Paris", "in London", "in Newcastle", "in Tokyo"
- Countries: "in France", "in Italy"
- Large areas: "in Manhattan", "in Barcelona"

ANALYSIS RULES:
1. GPS/Proximity requests → LOCATION_SEARCH
2. Requests needing GPS location → REQUEST_LOCATION  
3. City/country-wide requests → GENERAL_SEARCH
4. Off-topic → NOT_RESTAURANT

RESPONSE FORMAT (JSON only):
{{
    "request_type": "LOCATION_SEARCH" | "REQUEST_LOCATION" | "GENERAL_SEARCH" | "NOT_RESTAURANT",
    "location_detected": "specific location if found" | null,
    "cuisine_preference": "extracted cuisine/type preference" | null,
    "confidence": 0.1-1.0,
    "reasoning": "brief explanation",
    "suggested_response": "what bot should ask user next"
}}

EXAMPLES:

"natural wine bars in SoHo" →
{{
    "request_type": "LOCATION_SEARCH",
    "location_detected": "SoHo", 
    "cuisine_preference": "natural wine bars",
    "confidence": 0.9,
    "reasoning": "SoHo is a neighborhood - needs GPS proximity search",
    "suggested_response": "I'll search for natural wine bars in SoHo for you!"
}}

"best pubs in Newcastle" →
{{
    "request_type": "GENERAL_SEARCH",
    "location_detected": "Newcastle",
    "cuisine_preference": "pubs",
    "confidence": 0.9,
    "reasoning": "Newcastle is a city - use existing city-wide search pipeline",
    "suggested_response": "Perfect! Let me find the best pubs in Newcastle for you."
}}

"coffee shops near me" →
{{
    "request_type": "REQUEST_LOCATION",
    "location_detected": null,
    "cuisine_preference": "coffee shops", 
    "confidence": 0.8,
    "reasoning": "User wants nearby coffee shops but needs to specify GPS location",
    "suggested_response": "I'd love to find coffee shops near you! Could you share your location or tell me what neighborhood you're in?"
}}

"romantic restaurants in Paris" →
{{
    "request_type": "GENERAL_SEARCH", 
    "location_detected": "Paris",
    "cuisine_preference": "romantic restaurants",
    "confidence": 0.9,
    "reasoning": "Paris is a city - use existing city-wide search pipeline",
    "suggested_response": "Perfect! Let me find romantic restaurants in Paris for you."
}}

"sushi near Times Square" →
{{
    "request_type": "LOCATION_SEARCH",
    "location_detected": "Times Square",
    "cuisine_preference": "sushi",
    "confidence": 0.9,
    "reasoning": "Times Square is a specific landmark - needs GPS proximity search",
    "suggested_response": "I'll search for sushi restaurants near Times Square for you!"
}}
"""

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """
        Analyze a user message to determine if it's a location-based request

        Args:
            message: User's message text

        Returns:
            Dict with analysis results
        """
        try:
            response = self.analysis_chain.invoke({"user_message": message})

            # Parse AI response
            content = response.content.strip()

            # Clean up JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON response
            import json
            analysis_result = json.loads(content)

            # Add original message for reference
            analysis_result["original_message"] = message

            logger.debug(f"Location analysis result: {analysis_result}")
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse location analysis JSON: {e}")
            logger.error(f"Raw content: {content}")

            # Fallback analysis
            return self._fallback_analysis(message)

        except Exception as e:
            logger.error(f"Error in location analysis: {e}")
            return self._fallback_analysis(message)

    def _fallback_analysis(self, message: str) -> Dict[str, Any]:
        """
        Fallback analysis if AI parsing fails
        Uses simple keyword detection - CONSERVATIVE approach
        """
        message_lower = message.lower()

        # GPS/proximity keywords that indicate location-based search
        proximity_keywords = ["near me", "nearby", "around here", "close to", "walking distance", "in the neighborhood"]
        has_proximity = any(keyword in message_lower for keyword in proximity_keywords)

        # Neighborhood/district indicators (not cities)
        neighborhood_patterns = [
            r'\bin\s+(soho|chinatown|brooklyn|manhattan|times square|mission|castro)',
            r'\bnear\s+[A-Z][a-z]+\s+(square|street|avenue|road)',
        ]
        has_neighborhood = any(re.search(pattern, message, re.IGNORECASE) for pattern in neighborhood_patterns)

        # City keywords that indicate general search
        city_keywords = ["in paris", "in london", "in tokyo", "in rome", "in barcelona", "in newcastle", "in lisbon"]
        has_city = any(keyword in message_lower for keyword in city_keywords)

        # Food keyword detection  
        food_keywords = ["restaurant", "bar", "cafe", "coffee", "wine", "food", "eat", "drink", "pub"]
        has_food = any(keyword in message_lower for keyword in food_keywords)

        if has_proximity and has_food:
            return {
                "request_type": "REQUEST_LOCATION",
                "location_detected": None,
                "cuisine_preference": None,
                "confidence": 0.6,
                "reasoning": "Fallback analysis detected proximity + food keywords",
                "suggested_response": "I'd love to help you find restaurants nearby! Could you share your location?",
                "original_message": message
            }
        elif has_neighborhood and has_food:
            return {
                "request_type": "LOCATION_SEARCH",
                "location_detected": None,
                "cuisine_preference": None,
                "confidence": 0.5,
                "reasoning": "Fallback analysis detected neighborhood + food keywords",
                "suggested_response": "I'll search for restaurants in that area for you!",
                "original_message": message
            }
        elif has_city and has_food:
            return {
                "request_type": "GENERAL_SEARCH",
                "location_detected": None,
                "cuisine_preference": None,
                "confidence": 0.7,
                "reasoning": "Fallback analysis detected city + food keywords - use existing pipeline",
                "suggested_response": "I can help with restaurant recommendations! Let me search the city for you.",
                "original_message": message
            }
        elif has_food:
            return {
                "request_type": "GENERAL_SEARCH", 
                "location_detected": None,
                "cuisine_preference": None,
                "confidence": 0.6,
                "reasoning": "Fallback analysis detected food keywords only - default to general search",
                "suggested_response": "I can help with restaurant recommendations! Which city are you interested in?",
                "original_message": message
            }
        else:
            return {
                "request_type": "NOT_RESTAURANT",
                "location_detected": None, 
                "cuisine_preference": None,
                "confidence": 0.7,
                "reasoning": "Fallback analysis - no clear restaurant/location indicators",
                "suggested_response": "I specialize in restaurant recommendations! What type of dining are you looking for?",
                "original_message": message
            }

    def determine_search_type(self, analysis_result: Dict[str, Any]) -> str:
        """
        Determine which search pipeline to use based on analysis

        Args:
            analysis_result: Result from analyze_message()

        Returns:
            str: "location_search", "request_location", "general_search", or "clarify"
        """
        request_type = analysis_result.get("request_type", "NOT_RESTAURANT")
        confidence = analysis_result.get("confidence", 0.0)

        # High confidence location search
        if request_type == "LOCATION_SEARCH" and confidence >= 0.8:
            return "location_search"

        # User wants location-based search but needs to specify location
        elif request_type == "REQUEST_LOCATION":
            return "request_location"

        # General search (existing pipeline)
        elif request_type == "GENERAL_SEARCH":
            return "general_search"

        # Need clarification
        else:
            return "clarify"