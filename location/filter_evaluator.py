# location/filter_evaluator.py - UPDATED FILTERING LOGIC
"""
Location-based database filtering and evaluation - ENHANCED FOR SPECIFICITY

Made filtering more specific and relevant by requiring stronger matches.
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

    ENHANCED: More specific filtering logic for better relevance
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI for filtering
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # ENHANCED FILTERING PROMPT - More specific criteria
        self.filter_prompt = ChatPromptTemplate.from_template("""
USER QUERY: "{query}"
LOCATION: {location_description}

You are analyzing restaurants from our database to find matches for this location-based query.

RESTAURANT LIST:
{restaurants_text}

TASK: Select ONLY restaurants where cuisine tags and descriptions stronly suggest that match the user's query intent.

SELECTION RULES
1. **Cuisine Match**: Must be the requested cuisine type/dish or very closely related
2. **Query Keywords**: Look for specific features mentioned (price range, atmosphere, dietary needs)
3. **Context Analysis**: Consider what the user is actually looking for based on their exact words
4. **Quality Over Quantity**: Better to return 2 perfect matches than 5 "reasonable" ones
5. Ignore location keywords — the restaurants provided match location criteria already

SPECIFICITY RULES:
- If user asks for "cheap eats", focus on casual/affordable places, not upscale restaurants
- If user asks for "fine dining", focus on upscale/formal restaurants with detailed descriptions
- If user asks for specific features (rooftop, wine bar, vegan), select restaurants that offer that based on cuisine tags and descriptions
- If user asks for "famous chef restaurants", require chef mentions or high-end descriptions

DESCRIPTION ANALYSIS:
- Read the full description carefully - names and cuisine tags alone aren't enough
- Look for atmosphere indicators (casual, upscale, family-friendly, etc.)
- Check for specific features mentioned in descriptions
- Consider price indicators in descriptions

SOURCE ANALYSIS:
- The "Recommended by" field shows where this restaurant was featured (e.g., guide.michelin.com, eater.com, timeout.com)
- If user specifically asks for or excludes certain recommendation sources, respect that

OUTPUT: Return ONLY valid JSON with SPECIFICALLY matching restaurant IDs:
{{
    "selected_restaurants": [
        {{
            "id": "ID",
            "relevance_score": 0.8,
            "reasoning": "specific reason why this EXACTLY matches the query"
        }}
    ]
}}

IMPORTANT: Only include restaurants that are very good matches or none.
""")

        # ENHANCED EVALUATION PROMPT - More demanding criteria
        self.eval_prompt = ChatPromptTemplate.from_template("""
USER QUERY: "{{query}}"
LOCATION: {{location_description}}
FOUND: {{count}} specifically matching restaurants in database

EVALUATION TASK: 
Determine if these SPECIFIC matches are sufficient to answer the user's query immediately.

ENHANCED CRITERIA:
1. **Match Quality**: Are these restaurants EXACTLY what the user wants?
2. **Query Completeness**: Do we have enough variety/options for this specific request?
3. **Relevance Score**: Are all selected restaurants highly relevant (0.7+ scores)?

LOGIC:
- 3+ high-quality specific matches → SEND IMMEDIATELY (database_sufficient: true)
- 1-2 perfect matches for very specific queries → SEND IMMEDIATELY  
- Any low-relevance matches (< 0.7) → TRIGGER WEB SEARCH
- Zero truly relevant matches → NO DATABASE RESULTS (database_sufficient: false)

Return ONLY JSON:
{{
    "database_sufficient": true/false,
    "reasoning": "specific explanation based on match quality and query intent",
    "quality_score": 0.8,
    "send_immediately": true/false
}}

Focus on QUALITY and SPECIFICITY over quantity. Users prefer fewer, more relevant results.
""")

        # BUILD LANGCHAIN CHAINS
        self._build_chains()

        logger.info("✅ ENHANCED Location Filter Evaluator initialized with specific filtering")

    def _build_chains(self):
        """Build LangChain chains for filtering and evaluation"""

        # CHAIN 1: Restaurant filtering chain
        self.filter_chain = (
            self.filter_prompt 
            | self.ai_model.with_config(run_name="filter_restaurants_specific")
            | StrOutputParser()
            | RunnableLambda(self._parse_filter_response, name="parse_filter_json")
        )

        # CHAIN 2: Results evaluation chain  
        self.evaluation_chain = (
            self.eval_prompt
            | self.ai_model.with_config(run_name="evaluate_results_enhanced") 
            | StrOutputParser()
            | RunnableLambda(self._parse_evaluation_response, name="parse_evaluation_json")
        )

        logger.info("✅ LangChain chains built for enhanced filtering and evaluation")

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
        location_description: str = "GPS location",
        exclude_restaurants: Optional[List[str]] = None,
        supervisor_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Filter database restaurants and evaluate if sufficient for immediate sending

        ENHANCED: More specific filtering and evaluation
        """
        try:
            logger.info(f"🔍 ENHANCED filtering {len(restaurants)} restaurants for query: '{query}'")

            if not restaurants:
                return self._create_empty_result("No restaurants found in database")

            # STEP 1: AI filtering to select SPECIFICALLY relevant restaurants
            filtered_restaurants = self._filter_restaurants_with_chain(
                restaurants, query, location_description, 
                exclude_restaurants, supervisor_instructions
            )

            if not filtered_restaurants:
                return self._create_empty_result("No specifically relevant matches found")

            logger.info(f"🎯 AI selected {len(filtered_restaurants)} SPECIFIC restaurants")

            # STEP 2: Evaluate if sufficient for immediate sending
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
            logger.error(f"❌ Error in enhanced filter and evaluate: {e}")
            return self._create_empty_result(f"Error during filtering: {str(e)}")

    def _filter_restaurants_with_chain(
        self, 
        restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str,
        exclude_restaurants: Optional[List[str]] = None,
        supervisor_instructions: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter restaurants using LangChain chain with ENHANCED specificity"""
        try:
            # Pre-filter: Remove excluded restaurants
            if exclude_restaurants:
                original_count = len(restaurants)
                restaurants = [
                    r for r in restaurants 
                    if r.get('name', '').lower() not in [ex.lower() for ex in exclude_restaurants]
                ]
                excluded_count = original_count - len(restaurants)
                if excluded_count > 0:
                    logger.info(f"🚫 Pre-filtered {excluded_count} previously shown restaurants")

            # Create restaurant text for AI analysis - ENHANCED VERSION with SOURCES
            restaurants_text = ""
            for i, restaurant in enumerate(restaurants):
                name = restaurant.get('name', 'Unknown')
                cuisine = restaurant.get('cuisine_tags', [])
                cuisine_str = ', '.join(cuisine) if cuisine else 'No cuisine info'

                # ENHANCED: Include MORE description for better analysis
                description = restaurant.get('raw_description', 'No description')[:400]
                distance = restaurant.get('distance_km', 'Unknown')
                mention_count = restaurant.get('mention_count', 1)

                # NEW: Include sources/recommendation info
                sources_domains = restaurant.get('sources_domains', [])
                sources_str = ', '.join(sources_domains) if sources_domains else 'No source info'

                restaurants_text += f"{i+1}. ID: {restaurant.get('id')} | {name} ({distance}km)\n"
                restaurants_text += f"   Cuisine: {cuisine_str}\n"
                restaurants_text += f"   Recommended by: {sources_str}\n"
                restaurants_text += f"   Mentions: {mention_count}\n"
                restaurants_text += f"   Description: {description}...\n\n"

            # Build context-aware query that includes supervisor instructions
            effective_query = query
            if supervisor_instructions:
                effective_query = f"{query}\n\nADDITIONAL CONTEXT: {supervisor_instructions}"

            # Use enhanced LangChain chain
            ai_result = self.filter_chain.invoke({
                "query": effective_query,
                "location_description": location_description,
                "restaurants_text": restaurants_text
            })

            selected_data = ai_result.get("selected_restaurants", [])

            # Map selected IDs back to full restaurant objects with metadata
            restaurant_lookup = {str(r.get('id')): r for r in restaurants}
            selected_restaurants = []

            for selection in selected_data:
                restaurant_id = str(selection.get('id', ''))
                if restaurant_id in restaurant_lookup:
                    restaurant = restaurant_lookup[restaurant_id].copy()
                    restaurant['_relevance_score'] = selection.get('relevance_score', 0)
                    restaurant['_match_reasoning'] = selection.get('reasoning', '')
                    selected_restaurants.append(restaurant)

            logger.info(f"✅ Enhanced filtering selected {len(selected_restaurants)} specific matches")
            return selected_restaurants

        except Exception as e:
            logger.error(f"❌ Error in enhanced restaurant filtering: {e}")
            return []

    def _evaluate_with_chain(
        self, 
        filtered_restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """Evaluate filtered results with enhanced criteria"""
        try:
            # Enhanced evaluation with quality analysis
            evaluation = self.evaluation_chain.invoke({
                "query": query,
                "location_description": location_description,
                "count": len(filtered_restaurants)
            })

            logger.info(f"📊 Enhanced evaluation: {evaluation.get('reasoning', 'No reasoning')}")
            return evaluation

        except Exception as e:
            logger.error(f"❌ Error in enhanced evaluation: {e}")
            return {
                "database_sufficient": False,
                "reasoning": f"Enhanced evaluation failed: {str(e)}",
                "quality_score": 0.0,
                "send_immediately": False
            }

    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with enhanced logging"""
        logger.info(f"🚫 Enhanced filter returning empty result: {reason}")
        return {
            "filtered_restaurants": [],
            "evaluation": {
                "database_sufficient": False,
                "reasoning": reason,
                "quality_score": 0.0,
                "send_immediately": False
            },
            "database_sufficient": False,
            "send_immediately": False,
            "total_found": 0,
            "selected_count": 0,
            "reasoning": reason
        }