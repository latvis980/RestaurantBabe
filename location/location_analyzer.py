# agents/location_analyzer.py
"""
Location Analyzer Agent

Works with the existing conversation AI to detect and analyze location-based requests.
Integrates with the current LangChain pipeline architecture.
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
        """Get the enhanced system prompt for location analysis with ambiguity detection"""
        return """
    You are a location request analyzer for a restaurant recommendation system with ambiguity detection.

    Your job is to determine if a user message is requesting location-based (GPS/proximity) search vs general city-wide search, 
    AND detect if the location mentioned is ambiguous.

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
    - Large areas, though not formally cities: "in Manhattan", "in Barcelona", "in Tuscany", "around Lake Como

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
        "reasoning": "brief explanation",
        "suggested_response": "what bot should ask user next"
    }}

    EXAMPLES:

    "natural wine bars in SoHo" →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "SoHo", 
        "city_context": "New York",
        "is_ambiguous": false,
        "ambiguity_reason": null,
        "cuisine_preference": "natural wine bars",
        "confidence": 0.9,
        "reasoning": "SoHo typically refers to NYC neighborhood",
        "suggested_response": "I'll search for natural wine bars in SoHo, NYC for you!"
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
        "suggested_response": "I think there are multiple places called Springfield. Which state or country did you mean?"
    }}

    "coffee shops near Piccadilly Circus" →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "Piccadilly Circus",
        "city_context": "London",
        "is_ambiguous": false,
        "ambiguity_reason": null,
        "cuisine_preference": "coffee shops",
        "confidence": 0.9,
        "reasoning": "Piccadilly Circus is clearly London landmark",
        "suggested_response": "Sure, let's find coffee shops near Piccadilly Circus in London!"
    }}

    "bars in Cambridge" →
    {{
        "request_type": "LOCATION_SEARCH",
        "location_detected": "Cambridge",
        "city_context": null,
        "is_ambiguous": true,
        "ambiguity_reason": "could be Cambridge UK or Cambridge Massachusetts",
        "cuisine_preference": "bars",
        "confidence": 0.8,
        "reasoning": "Cambridge could refer to UK or Massachusetts",
        "suggested_response": "Which Cambridge did you mean - the one in England or Massachusetts?"
    }}
    """

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """
        Analyze a user message to determine if it's a location-based request
        CLEAN: Pure AI approach without hardcoded fallbacks

        Args:
            message: User's message text

        Returns:
            Dict with analysis results including ambiguity detection
        """
        try:
            response = self.analysis_chain.invoke({"user_message": message})
            content = response.content.strip()

            # Clean up JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            analysis_result = json.loads(content)
            analysis_result["original_message"] = message

            logger.debug(f"Location analysis result: {analysis_result}")
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse location analysis JSON: {e}")


            # CLEAN: Return safe default that won't break the flow
            return self._create_safe_default(message)

        except Exception as e:
            logger.error(f"Error in location analysis: {e}")

            # CLEAN: Return safe default that won't break the flow  
            return self._create_safe_default(message)

    def _create_safe_default(self, message: str) -> Dict[str, Any]:
        """
        Create a safe default response when AI analysis fails
        Routes to general search to maintain app flow
        """
        return {
            "request_type": "GENERAL_SEARCH",  # Safe default - use existing pipeline
            "location_detected": None,
            "city_context": None,
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "cuisine_preference": None,
            "confidence": 0.1,  # Low confidence indicates fallback
            "reasoning": "AI analysis failed - defaulting to general search",
            "suggested_response": "I can help you find restaurants! Could you be more specific about what you're looking for?",
            "original_message": message
        }
