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

Your job is to determine if a user message is requesting location-based restaurant/bar/cafe recommendations.

LOCATION-BASED REQUEST INDICATORS:
- "near me", "nearby", "around here", "close to"
- "in [neighborhood/area]", "at [location]" 
- Street names, landmarks, districts
- "where I am", "local", "in this area"
- GPS coordinates or specific addresses

ANALYSIS RULES:
1. If the message requests restaurants/bars/cafes near a specific location → LOCATION_SEARCH
2. If the message mentions location terms but is vague → REQUEST_LOCATION  
3. If it's a general restaurant query without location context → GENERAL_SEARCH
4. If completely off-topic → NOT_RESTAURANT

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

"natural wine bars in Chinatown" →
{{
    "request_type": "LOCATION_SEARCH",
    "location_detected": "Chinatown", 
    "cuisine_preference": "natural wine bars",
    "confidence": 0.9,
    "reasoning": "Clear location (Chinatown) and specific request (natural wine bars)",
    "suggested_response": "I'll search for natural wine bars in Chinatown for you!"
}}

"coffee shops near me" →
{{
    "request_type": "REQUEST_LOCATION",
    "location_detected": null,
    "cuisine_preference": "coffee shops", 
    "confidence": 0.8,
    "reasoning": "User wants nearby coffee shops but hasn't specified exact location",
    "suggested_response": "I'd love to find coffee shops near you! Could you share your location or tell me what neighborhood you're in?"
}}

"best pasta in the city" →
{{
    "request_type": "REQUEST_LOCATION",
    "location_detected": null,
    "cuisine_preference": "pasta restaurants",
    "confidence": 0.7, 
    "reasoning": "Food request but 'the city' is too vague - need specific location",
    "suggested_response": "I'd love to find great pasta places for you! Which city or neighborhood are you interested in?"
}}

"romantic restaurants in Paris" →
{{
    "request_type": "LOCATION_SEARCH", 
    "location_detected": "Paris",
    "cuisine_preference": "romantic restaurants",
    "confidence": 0.9,
    "reasoning": "Clear location (Paris) with specific dining preference (romantic)",
    "suggested_response": "Perfect! I'll find romantic restaurants in Paris for you."
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
        Uses simple keyword detection
        """
        message_lower = message.lower()

        # Simple location keyword detection
        location_keywords = ["near", "in ", "at ", "close to", "around", "nearby"]
        has_location = any(keyword in message_lower for keyword in location_keywords)

        # Simple food keyword detection  
        food_keywords = ["restaurant", "bar", "cafe", "coffee", "wine", "food", "eat", "drink"]
        has_food = any(keyword in message_lower for keyword in food_keywords)

        if has_location and has_food:
            return {
                "request_type": "REQUEST_LOCATION",
                "location_detected": None,
                "cuisine_preference": None,
                "confidence": 0.5,
                "reasoning": "Fallback analysis detected location + food keywords",
                "suggested_response": "I'd love to help you find restaurants! Could you tell me your specific location?",
                "original_message": message
            }
        elif has_food:
            return {
                "request_type": "GENERAL_SEARCH", 
                "location_detected": None,
                "cuisine_preference": None,
                "confidence": 0.6,
                "reasoning": "Fallback analysis detected food keywords only",
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