# location/filter_evaluator.py
"""
Location-based database filtering and evaluation - FIXED WITH LANGCHAIN CHAINS

Converted direct AI invokes to proper LangChain chain composition for tracing.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda

logger = logging.getLogger(__name__)

class LocationFilterEvaluator:
    """
    Filter and evaluate database results for location-based searches with LangChain chains

    Logic: If ANY relevant results found → send immediately 
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI for filtering
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Filtering prompt (based on database_search_agent)
        self.filter_prompt = ChatPromptTemplate.from_template("""
USER QUERY: "{query}"
LOCATION: {location_description}

You are analyzing restaurants from our database to find matches for this location-based query.

RESTAURANT LIST:
{restaurants_text}

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
USER QUERY: "{query}"
LOCATION: {location_description}
FOUND: {count} restaurants in database

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

        # BUILD LANGCHAIN CHAINS - FIXED VERSION
        self._build_chains()

        logger.info("✅ Location Filter Evaluator initialized with LangChain chains")

    def _build_chains(self):
        """Build LangChain chains for filtering and evaluation"""

        # CHAIN 1: Restaurant filtering chain
        self.filter_chain = (
            self.filter_prompt 
            | self.ai_model.with_config(run_name="filter_restaurants")
            | StrOutputParser()
            | RunnableLambda(self._parse_filter_response, name="parse_filter_json")
        )

        # CHAIN 2: Results evaluation chain  
        self.evaluation_chain = (
            self.eval_prompt
            | self.ai_model.with_config(run_name="evaluate_results") 
            | StrOutputParser()
            | RunnableLambda(self._parse_evaluation_response, name="parse_evaluation_json")
        )

        logger.info("✅ LangChain chains built for filtering and evaluation")

    def _parse_filter_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI filter response to extract JSON"""
        try:
            content = response_content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI filtering response: {e}")
            return {"selected_restaurants": []}

    def _parse_evaluation_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI evaluation response to extract JSON"""
        try:
            content = response_content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI evaluation response: {e}")
            return {
                "database_sufficient": False,
                "reasoning": "Failed to parse evaluation",
                "quality_score": 0.0,
                "send_immediately": False
            }

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

            # STEP 1: AI filtering to select relevant restaurants (using LangChain chain)
            filtered_restaurants = self._filter_restaurants_with_chain(restaurants, query, location_description)

            if not filtered_restaurants:
                return self._create_empty_result("No relevant matches found")

            logger.info(f"🎯 AI selected {len(filtered_restaurants)} relevant restaurants")

            # STEP 2: Evaluate if sufficient for immediate sending (using LangChain chain)
            evaluation = self._evaluate_with_chain(filtered_restaurants, query, location_description)

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

    def _filter_restaurants_with_chain(
        self, 
        restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str
    ) -> List[Dict[str, Any]]:
        """Filter restaurants using LangChain chain (FIXED VERSION)"""
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

            # FIXED: Use LangChain chain instead of direct invoke
            ai_result = self.filter_chain.invoke({
                "query": query,
                "location_description": location_description,
                "restaurants_text": restaurants_text
            })

            selected_data = ai_result.get("selected_restaurants", [])

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
            logger.error(f"❌ Error in AI filtering chain: {e}")
            return []

    def _evaluate_with_chain(
        self, 
        filtered_restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """Evaluate if results are sufficient using LangChain chain"""
        try:
            count = len(filtered_restaurants)

            # For location searches: ANY relevant results = send immediately
            if count > 0:
                quality_scores = [r.get('_relevance_score', 0.5) for r in filtered_restaurants]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

                # FIXED: Use LangChain chain for evaluation
                evaluation_result = self.evaluation_chain.invoke({
                    "query": query,
                    "location_description": location_description,
                    "count": count
                })

                # Enhance with calculated quality score
                evaluation_result["quality_score"] = round(avg_quality, 2)
                return evaluation_result

            else:
                return {
                    "database_sufficient": False,
                    "send_immediately": False,
                    "reasoning": "No relevant restaurants found",
                    "quality_score": 0.0
                }

        except Exception as e:
            logger.error(f"❌ Error in evaluation chain: {e}")
            return {
                "database_sufficient": False,
                "send_immediately": False,
                "reasoning": f"Evaluation failed: {str(e)}",
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