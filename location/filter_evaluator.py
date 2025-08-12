# location/filter_evaluator.py
"""
Location-based database filtering and evaluation

Isolated copy of database_search and content_evaluation logic for location flow.
Key difference: sends ANY results immediately with "personal notes" message.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationFilterEvaluator:
    """
    Filter and evaluate database results for location-based searches

    Logic: If ANY relevant results found → send immediately 
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI for filtering
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
            max_tokens=2048
        )

        # Filtering prompt (based on database_search_agent)
        self.filter_prompt = ChatPromptTemplate.from_template("""
USER QUERY: "{{query}}"
LOCATION: {{location_description}}

You are analyzing restaurants from our database to find matches for this location-based query.

RESTAURANT LIST:
{{restaurants_text}}

TASK: Select restaurants that match the user's query intent.

MATCHING CRITERIA:
- Cuisine type relevance (direct matches, related cuisines)
- Dining style and atmosphere
- Special features mentioned (wine lists, vegan options, price range, etc.)
- General vibe from descriptions

OUTPUT: Return ONLY valid JSON with matching restaurant IDs:
{{
    "selected_restaurants": [
        {{
            "id": "ID",
            "relevance_score": 0.8,
            "reasoning": "why this matches the search intent"
        }}
    ]
}}

Include restaurants that are good matches. For location searches, be inclusive - if it's a reasonable match, include it.
""")

        # Evaluation prompt (simplified from content_evaluation_agent)
        self.eval_prompt = ChatPromptTemplate.from_template("""
USER QUERY: "{{query}}"
LOCATION: {{location_description}}
FOUND: {{count}} restaurants in database

EVALUATION TASK: 
For location-based searches, we send ANY relevant results immediately as "restaurants from my notes".

CRITERIA:
1. Are there ANY restaurants that match the query?
2. Quality of matches (relevance scores)

LOGIC:
- 1+ relevant matches → SEND IMMEDIATELY (database_sufficient: true)
- Zero matches → NO DATABASE RESULTS (database_sufficient: false)

Return ONLY JSON:
{{
    "database_sufficient": true/false,
    "reasoning": "brief explanation",
    "quality_score": 0.8,
    "send_immediately": true/false
}}

For location searches, be generous - even 1-2 good matches should be sent.
""")

        logger.info("✅ Location Filter Evaluator initialized")

    def filter_and_evaluate(
        self, 
        restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str = "GPS location"
    ) -> Dict[str, Any]:
        """
        Filter database restaurants and evaluate if sufficient for immediate sending

        Returns:
            Dict with filtered results and evaluation
        """
        try:
            logger.info(f"🔍 Filtering {len(restaurants)} restaurants for location query: '{query}'")

            if not restaurants:
                return self._create_empty_result("No restaurants found in database")

            # STEP 1: AI filtering to select relevant restaurants
            filtered_restaurants = self._filter_restaurants(restaurants, query, location_description)

            if not filtered_restaurants:
                return self._create_empty_result("No relevant matches found")

            logger.info(f"🎯 AI selected {len(filtered_restaurants)} relevant restaurants")

            # STEP 2: Evaluate if sufficient for immediate sending (location logic)
            evaluation = self._evaluate_for_location_search(filtered_restaurants, query, location_description)

            # STEP 3: Combine results
            return {
                "filtered_restaurants": filtered_restaurants,
                "evaluation": evaluation,
                "database_sufficient": evaluation.get("database_sufficient", False),
                "send_immediately": evaluation.get("send_immediately", False),
                "total_found": len(restaurants),
                "selected_count": len(filtered_restaurants),
                "reasoning": evaluation.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"❌ Error in filter and evaluate: {e}")
            return self._create_empty_result(f"Error during filtering: {str(e)}")

    def _filter_restaurants(
        self, 
        restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str
    ) -> List[Dict[str, Any]]:
        """Filter restaurants using AI (based on database_search_agent logic)"""
        try:
            # Create restaurant text for AI analysis
            restaurants_text = ""
            for i, restaurant in enumerate(restaurants):
                name = restaurant.get('name', 'Unknown')
                cuisine = restaurant.get('cuisine_tags', [])
                cuisine_str = ', '.join(cuisine) if cuisine else 'No cuisine info'
                description = restaurant.get('raw_description', 'No description')[:200]
                distance = restaurant.get('distance_km', 'Unknown')

                restaurants_text += f"{i+1}. ID: {restaurant.get('id')} | {name} ({distance}km)\n"
                restaurants_text += f"   Cuisine: {cuisine_str}\n"
                restaurants_text += f"   Description: {description}...\n\n"

            # Run AI filtering
            response = self.ai_model.invoke(
                self.filter_prompt.format(
                    query=query,
                    location_description=location_description,
                    restaurants_text=restaurants_text
                )
            )

            # Parse AI response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                ai_result = json.loads(content)
                selected_data = ai_result.get("selected_restaurants", [])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse AI filtering response: {content}")
                return []

            # Map selected IDs back to full restaurant objects
            restaurant_lookup = {str(r.get('id')): r for r in restaurants}
            selected_restaurants = []

            for selection in selected_data:
                restaurant_id = str(selection.get('id', ''))
                if restaurant_id in restaurant_lookup:
                    restaurant = restaurant_lookup[restaurant_id].copy()
                    restaurant['_relevance_score'] = selection.get('relevance_score', 0)
                    restaurant['_match_reasoning'] = selection.get('reasoning', '')
                    selected_restaurants.append(restaurant)

            return selected_restaurants

        except Exception as e:
            logger.error(f"❌ Error in AI filtering: {e}")
            return []

    def _evaluate_for_location_search(
        self, 
        filtered_restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """Evaluate if results are sufficient for location search (simplified logic)"""
        try:
            count = len(filtered_restaurants)

            # For location searches: ANY relevant results = send immediately
            if count > 0:
                quality_scores = [r.get('_relevance_score', 0.5) for r in filtered_restaurants]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

                return {
                    "database_sufficient": True,
                    "send_immediately": True,
                    "reasoning": f"Found {count} relevant restaurants from personal notes",
                    "quality_score": round(avg_quality, 2)
                }
            else:
                return {
                    "database_sufficient": False,
                    "send_immediately": False,
                    "reasoning": "No relevant matches found in database",
                    "quality_score": 0.0
                }

        except Exception as e:
            logger.error(f"❌ Error in evaluation: {e}")
            return {
                "database_sufficient": False,
                "send_immediately": False,
                "reasoning": f"Evaluation error: {str(e)}",
                "quality_score": 0.0
            }

    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "filtered_restaurants": [],
            "evaluation": {
                "database_sufficient": False,
                "send_immediately": False,
                "reasoning": reason,
                "quality_score": 0.0
            },
            "database_sufficient": False,
            "send_immediately": False,
            "total_found": 0,
            "selected_count": 0,
            "reasoning": reason
        }

    def format_personal_notes_message(
        self, 
        restaurants: List[Dict[str, Any]],
        query: str,
        location_description: str = "your location"
    ) -> str:
        """Format the 'personal notes' message for immediate sending"""
        try:
            if not restaurants:
                return "🤔 I don't have any restaurants from my notes for this location."

            count = len(restaurants)

            # Header message
            header = f"📝 <b>Here are {count} restaurants from my notes near {location_description}:</b>\n\n"

            # List restaurants
            restaurant_list = ""
            for i, restaurant in enumerate(restaurants[:8]):  # Limit to 8 for readability
                name = restaurant.get('name', 'Unknown')
                distance = restaurant.get('distance_km', '?')
                cuisine = restaurant.get('cuisine_tags', [])
                cuisine_str = ', '.join(cuisine[:2]) if cuisine else ''  # Max 2 cuisine tags

                restaurant_list += f"🍽 <b>{name}</b> ({distance}km)"
                if cuisine_str:
                    restaurant_list += f" - <i>{cuisine_str}</i>"
                restaurant_list += "\n"

            # Footer message
            footer = (
                f"\n💡 <b>These are from my personal notes.</b> "
                f"Let me know if you want me to make some calls and search for more good addresses!"
            )

            return header + restaurant_list + footer

        except Exception as e:
            logger.error(f"❌ Error formatting message: {e}")
            return "📝 Found some restaurants from my notes, but had trouble formatting the list."