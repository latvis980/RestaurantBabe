# utils/ai_destination_detector.py
"""
AI-Powered Destination Change Detection

Intelligently detects when a user is asking about a different location than their stored context,
using AI to understand semantic meaning rather than simple pattern matching.
"""

import json
import logging
import time
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class AIDestinationChangeDetector:
    """
    AI-powered detector for destination changes in user queries
    """

    def __init__(self, config):
        self.config = config
        self.ai_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for consistent decisions
            api_key=config.OPENAI_API_KEY
        )

        # Build the detection prompt
        self._build_prompt()

        logger.info("✅ AI Destination Change Detector initialized")

    def _build_prompt(self):
        """Build the AI prompt for destination change detection"""

        system_prompt = """
You are an AI that detects when a user is asking about restaurants in a DIFFERENT location than their previous search.

Your job: Compare the NEW user message with their STORED location context to determine if they're asking about a different destination.

DETECTION RULES:

🔄 DESTINATION CHANGED (return true):
- Different cities: "restaurants in Paris" vs "bars in Tokyo"
- Different countries: "cafes in France" vs "pizza in Italy"  
- Different major areas: "food in Manhattan" vs "restaurants in Brooklyn"
- City vs neighborhood: "restaurants in London" vs "bars in Soho" (London neighborhood)
- Completely different geographic contexts

✅ SAME DESTINATION (return false):
- Follow-up requests: "show me more", "other options", "what else"
- Same location with different food types: "pizza in Paris" vs "sushi in Paris"
- Refinements: "cheap restaurants in Tokyo" vs "expensive restaurants in Tokyo"
- Vague requests without location: "restaurants", "good food", "recommendations"
- Questions about the same area: "bars in SoHo" vs "restaurants in SoHo"

AMBIGUOUS CASES:
- If unsure, lean toward SAME DESTINATION to avoid clearing useful context
- Neighborhoods within same city = SAME DESTINATION
- Boroughs of same city = DIFFERENT (Manhattan vs Brooklyn)

RESPONSE FORMAT (JSON only):
{{
    "destination_changed": true | false,
    "confidence": 0.1-1.0,
    "reasoning": "brief explanation of decision",
    "old_location": "extracted from stored context",
    "new_location": "extracted from current message" | null,
    "geographic_relationship": "same_city" | "same_country" | "different_country" | "different_city" | "no_location"
}}

EXAMPLES:

Stored: "restaurants in Paris" 
New: "bars in London"
→ {{"destination_changed": true, "confidence": 0.95, "reasoning": "Different cities - Paris vs London", "old_location": "Paris", "new_location": "London", "geographic_relationship": "different_city"}}

Stored: "pizza in Tokyo"
New: "sushi in Tokyo" 
→ {{"destination_changed": false, "confidence": 0.9, "reasoning": "Same city Tokyo, different cuisine", "old_location": "Tokyo", "new_location": "Tokyo", "geographic_relationship": "same_city"}}

Stored: "restaurants in SoHo"
New: "show me more"
→ {{"destination_changed": false, "confidence": 0.95, "reasoning": "Follow-up request without new location", "old_location": "SoHo", "new_location": null, "geographic_relationship": "no_location"}}

Stored: "bars in Manhattan" 
New: "restaurants in Brooklyn"
→ {{"destination_changed": true, "confidence": 0.85, "reasoning": "Different NYC boroughs", "old_location": "Manhattan", "new_location": "Brooklyn", "geographic_relationship": "different_city"}}
"""

        self.detection_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
STORED LOCATION CONTEXT:
- Previous Query: "{stored_query}"
- Location Description: "{stored_location}"
- Time: {time_minutes} minutes ago

CURRENT USER MESSAGE: "{current_message}"

Does the current message indicate the user wants to search in a DIFFERENT destination than their stored context?
""")
        ])

        # Create the chain
        self.detection_chain = self.detection_prompt | self.ai_model

    def detect_destination_change(
        self, 
        current_message: str, 
        stored_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect if the current message indicates a destination change

        Args:
            current_message: The user's current message
            stored_context: Previous location search context

        Returns:
            Detection result with decision and reasoning
        """
        try:
            if not stored_context:
                return {
                    "destination_changed": False,
                    "confidence": 0.0,
                    "reasoning": "No stored context to compare against",
                    "old_location": None,
                    "new_location": None
                }

            # Calculate time since last search
            last_search_time = stored_context.get("last_search_time", 0)
            time_ago = time.time() - last_search_time
            time_minutes = int(time_ago / 60)

            # Prepare context information
            stored_query = stored_context.get("query", "")
            stored_location = stored_context.get("location_description", "")

            logger.info(f"🤖 AI analyzing destination change:")
            logger.info(f"   Current: '{current_message}'")
            logger.info(f"   Stored: '{stored_query}' in '{stored_location}' ({time_minutes}m ago)")

            # Call the AI
            response = self.detection_chain.invoke({
                "current_message": current_message,
                "stored_query": stored_query,
                "stored_location": stored_location,
                "time_minutes": time_minutes
            })

            # Parse the AI response
            result = self._parse_ai_response(response.content)

            # Log the decision
            decision = result.get("destination_changed", False)
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "No reasoning provided")

            if decision:
                logger.info(f"🔄 DESTINATION CHANGED (confidence: {confidence:.2f}): {reasoning}")
            else:
                logger.info(f"✅ SAME DESTINATION (confidence: {confidence:.2f}): {reasoning}")

            return result

        except Exception as e:
            logger.error(f"❌ Error in AI destination change detection: {e}")
            # Fallback: don't clear context if AI fails
            return {
                "destination_changed": False,
                "confidence": 0.0,
                "reasoning": f"AI detection failed: {str(e)}",
                "error": True
            }

    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response with fallback handling"""
        try:
            # Try to extract JSON from response
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate required fields
                required_fields = ["destination_changed", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = self._get_default_value(field)

                return parsed
            else:
                logger.warning(f"No JSON found in AI response: {response_content}")
                return self._get_fallback_result()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"Response content: {response_content}")
            return self._get_fallback_result()
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._get_fallback_result()

    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing fields"""
        defaults = {
            "destination_changed": False,
            "confidence": 0.0,
            "reasoning": "Default value used",
            "old_location": None,
            "new_location": None,
            "geographic_relationship": "unknown"
        }
        return defaults.get(field)

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Get fallback result when AI parsing fails"""
        return {
            "destination_changed": False,  # Conservative: don't clear context
            "confidence": 0.0,
            "reasoning": "AI parsing failed - defaulting to no change",
            "old_location": None,
            "new_location": None,
            "error": True
        }

    def validate_detection_logic(self) -> bool:
        """
        Validate the AI detection logic with test cases
        Returns True if validation passes
        """
        logger.info("🧪 Validating AI destination change detection...")

        test_cases = [
            # Should detect change
            {
                "stored": {"query": "restaurants in Paris", "location_description": "Paris"},
                "current": "bars in London",
                "expected": True,
                "description": "Different cities"
            },
            {
                "stored": {"query": "pizza in Tokyo", "location_description": "Tokyo"}, 
                "current": "restaurants in Dubai",
                "expected": True,
                "description": "Different cities"
            },
            {
                "stored": {"query": "bars in Manhattan", "location_description": "Manhattan"},
                "current": "restaurants in Brooklyn", 
                "expected": True,
                "description": "Different NYC boroughs"
            },

            # Should NOT detect change
            {
                "stored": {"query": "restaurants in Paris", "location_description": "Paris"},
                "current": "pizza in Paris",
                "expected": False, 
                "description": "Same city, different cuisine"
            },
            {
                "stored": {"query": "bars in Tokyo", "location_description": "Tokyo"},
                "current": "show me more",
                "expected": False,
                "description": "Follow-up request"
            },
            {
                "stored": {"query": "restaurants in SoHo", "location_description": "SoHo"},
                "current": "bars in SoHo",
                "expected": False,
                "description": "Same neighborhood"
            },
        ]

        passed = 0
        failed = 0

        for i, test_case in enumerate(test_cases, 1):
            try:
                # Add timestamp to stored context
                stored_context = test_case["stored"].copy()
                stored_context["last_search_time"] = time.time() - 300  # 5 minutes ago

                result = self.detect_destination_change(
                    test_case["current"],
                    stored_context
                )

                detected_change = result.get("destination_changed", False)
                expected_change = test_case["expected"]

                if detected_change == expected_change:
                    logger.info(f"✅ Test {i}: {test_case['description']} - PASS")
                    passed += 1
                else:
                    logger.error(f"❌ Test {i}: {test_case['description']} - FAIL")
                    logger.error(f"   Expected: {expected_change}, Got: {detected_change}")
                    logger.error(f"   Reasoning: {result.get('reasoning', 'No reasoning')}")
                    failed += 1

            except Exception as e:
                logger.error(f"❌ Test {i}: Exception - {e}")
                failed += 1

        success_rate = passed / len(test_cases)
        logger.info(f"📊 Validation Results: {passed}/{len(test_cases)} passed ({success_rate:.1%})")

        return success_rate >= 0.8  # 80% success rate threshold