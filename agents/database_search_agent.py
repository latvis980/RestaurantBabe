# agents/database_search_agent.py
"""
FIXED Database Search Agent - Removed is_restaurant_list parameter

This fixes the LangChain prompt template error by removing the unnecessary
is_restaurant_list parameter that was causing the crash.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class DatabaseSearchAgent:
    """
    FIXED agent that analyzes all restaurants in a single AI call
    without the problematic is_restaurant_list parameter.
    """

    def __init__(self, config):
        self.config = config
        self.minimum_restaurant_threshold = getattr(config, 'MIN_DATABASE_RESTAURANTS', 3)

        # Initialize DeepSeek for single batch analysis
        self.llm = ChatDeepSeek(
            model_name=config.DEEPSEEK_MODEL,
            temperature=0.1
        )

        # Setup fixed prompts
        self._setup_prompts()
        logger.info("🚀 FIXED DatabaseSearchAgent initialized (removed is_restaurant_list parameter)")

    def _setup_prompts(self):
        """Setup fixed prompts for batch analysis - REMOVED is_restaurant_list parameter"""

        # FIXED: Single batch analysis prompt - analyzes ALL restaurants at once
        self.batch_analysis_prompt = ChatPromptTemplate.from_template("""
USER QUERY: {raw_query}
LOCATION: {destination}

You are analyzing restaurants from our database to see which ones match the user's query.

RESTAURANT LIST:
{restaurants_text}

TASK: Analyze this list and return the restaurant IDs that best match the user's query.

MATCHING CRITERIA:
- Cuisine type relevance (direct matches, related cuisines)
- Atmosphere and dining style 
- Special features mentioned (wine lists, vegan options, price range, etc.)
- Quality indicators from descriptions

SCORING:
- Perfect match (9-10): Direct cuisine match + special features match
- High relevance (7-8): Strong cuisine or feature match
- Moderate relevance (5-6): Some connection to query
- Low relevance (3-4): Weak connection
- Not relevant (0-2): No meaningful connection

Return ONLY valid JSON with the top matches:
{{
    "selected_restaurants": [
        {{
            "id": "restaurant_id",
            "name": "restaurant_name", 
            "relevance_score": 8,
            "reasoning": "why this restaurant matches"
        }}
    ],
    "total_analyzed": number_of_restaurants_analyzed,
    "query_analysis": "brief analysis of what user is looking for"
}}

IMPORTANT: Only include restaurants with score 5 or higher. Prioritize quality over quantity.
""")

        # FIXED: Quality evaluation prompt - decides if results are sufficient
        self.quality_evaluation_prompt = ChatPromptTemplate.from_template("""
USER QUERY: {raw_query}
LOCATION: {destination}

DATABASE RESULTS ({restaurant_count} restaurants found):
{restaurants_summary}

Evaluate if these database results sufficiently answer the user's query.

Consider:
- Do the restaurants match the query intent?
- Is there enough variety for the user?
- Are the descriptions detailed enough?
- Would web search likely find significantly better options?

Return JSON:
{{
    "sufficient": true/false,
    "confidence": 0-10,
    "reasoning": "explanation of decision",
    "missing_aspects": ["what might be missing if insufficient"]
}}
""")

    @log_function_call
    def search_and_evaluate(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Main method using single API call approach with corrected prompts
        """
        try:
            logger.info("🗃️ STARTING FIXED DATABASE SEARCH (single API call)")

            # Extract data from query analysis
            destination = query_analysis.get("destination", "Unknown")
            raw_query = query_analysis.get("raw_query", query_analysis.get("query", ""))

            if destination == "Unknown":
                logger.info("⚠️ No destination detected, will use web search")
                return self._create_web_search_response("no_destination")

            # Get all restaurants for the city
            all_restaurants = self._get_restaurants_for_city(destination)

            if not all_restaurants:
                logger.info("📭 No restaurants found in database for this city")
                return self._create_web_search_response("no_restaurants_in_city")

            logger.info(f"📊 Database query returned {len(all_restaurants)} restaurants for {destination}")

            # FIXED: Analyze all restaurants in a single API call
            selected_restaurants = self._batch_analyze_restaurants(
                all_restaurants, 
                raw_query, 
                destination
            )

            if not selected_restaurants:
                logger.info("📭 No relevant restaurants found after batch analysis")
                return self._create_web_search_response("no_relevant_results")

            # Evaluate quality of results
            quality_evaluation = self._evaluate_results_quality(
                selected_restaurants,
                raw_query,
                destination
            )

            # Make final decision
            if quality_evaluation["sufficient"]:
                logger.info(f"✅ DATABASE SUFFICIENT: {len(selected_restaurants)} relevant restaurants found")
                return self._create_database_response(selected_restaurants, quality_evaluation)
            else:
                logger.info(f"🌐 WEB SEARCH NEEDED: {quality_evaluation.get('reasoning', 'Insufficient results')}")
                return self._create_web_search_response(quality_evaluation.get("reasoning", "insufficient_quality"))

        except Exception as e:
            logger.error(f"❌ Error in fixed database search: {e}")
            return self._create_web_search_response(f"database_error: {str(e)}")

    def _get_restaurants_for_city(self, city: str) -> List[Dict[str, Any]]:
        """Get all restaurants for a city from database"""
        try:
            from utils.database import get_database
            db = get_database()

            # Get restaurants ordered by mention count (most mentioned first)
            restaurants = db.get_restaurants_by_city(city, limit=100)
            return restaurants

        except Exception as e:
            logger.error(f"Error getting restaurants for {city}: {e}")
            return []

    def _batch_analyze_restaurants(
        self, 
        restaurants: List[Dict[str, Any]], 
        raw_query: str, 
        destination: str
    ) -> List[Dict[str, Any]]:
        """
        FIXED: Analyze ALL restaurants in a single API call with corrected parameters
        """
        try:
            # Compile all restaurant data into a single text for analysis
            restaurants_text = self._compile_restaurants_for_analysis(restaurants)

            logger.info(f"🧠 Batch analyzing {len(restaurants)} restaurants in single API call...")

            # FIXED: Create the analysis chain with correct parameters only
            chain = self.batch_analysis_prompt | self.llm

            # FIXED: Single API call to analyze all restaurants - only pass expected parameters
            response = chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "restaurants_text": restaurants_text
            })

            # Parse the response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            analysis_result = json.loads(content)

            # Map the selected restaurant IDs back to full restaurant data
            selected_restaurants = self._map_selected_restaurants(
                analysis_result.get("selected_restaurants", []),
                restaurants
            )

            logger.info(f"✅ Batch analysis complete: {len(selected_restaurants)} restaurants selected")

            return selected_restaurants

        except Exception as e:
            logger.error(f"Error in batch restaurant analysis: {e}")
            # Fallback to simple keyword matching if AI analysis fails
            return self._fallback_keyword_matching(restaurants, raw_query)

    def _compile_restaurants_for_analysis(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Compile all restaurant data into optimized text format for single AI analysis
        """
        compiled_text = []

        for restaurant in restaurants:
            # Extract key information
            restaurant_id = restaurant.get('id', 'unknown')
            name = restaurant.get('name', 'Unknown')
            cuisine_tags = ', '.join(restaurant.get('cuisine_tags', []))
            description = restaurant.get('raw_description', '')[:300]  # Truncate to save tokens
            mention_count = restaurant.get('mention_count', 1)

            # Format for analysis (keep it concise)
            restaurant_entry = f"ID: {restaurant_id} | {name} | Tags: {cuisine_tags} | Mentions: {mention_count} | Desc: {description}"
            compiled_text.append(restaurant_entry)

        return "\n".join(compiled_text)

    def _map_selected_restaurants(
        self, 
        selected_data: List[Dict[str, Any]], 
        all_restaurants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map AI-selected restaurant IDs back to full restaurant data
        """
        selected_restaurants = []

        # Create a lookup dict for fast access
        restaurant_lookup = {str(r.get('id')): r for r in all_restaurants}

        for selection in selected_data:
            restaurant_id = str(selection.get('id', ''))

            if restaurant_id in restaurant_lookup:
                restaurant = restaurant_lookup[restaurant_id].copy()
                # Add AI analysis metadata
                restaurant['_ai_relevance_score'] = selection.get('relevance_score', 5)
                restaurant['_ai_reasoning'] = selection.get('reasoning', 'Selected by AI')
                selected_restaurants.append(restaurant)
            else:
                logger.warning(f"Restaurant ID {restaurant_id} not found in lookup")

        return selected_restaurants

    def _evaluate_results_quality(
        self, 
        restaurants: List[Dict[str, Any]], 
        raw_query: str, 
        destination: str
    ) -> Dict[str, Any]:
        """
        FIXED: Evaluate the quality of selected restaurants with correct parameters
        """
        try:
            # Create summary for evaluation
            restaurants_summary = self._create_restaurants_summary(restaurants)

            # FIXED: Create evaluation chain with correct parameters only
            chain = self.quality_evaluation_prompt | self.llm

            # FIXED: Single API call for quality evaluation - only pass expected parameters
            response = chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "restaurant_count": len(restaurants),
                "restaurants_summary": restaurants_summary
            })

            # Parse response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content)

            # Ensure we have the minimum threshold check
            if len(restaurants) < self.minimum_restaurant_threshold:
                evaluation["sufficient"] = False
                evaluation["reasoning"] = f"Only {len(restaurants)} restaurants found, minimum threshold is {self.minimum_restaurant_threshold}"

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating restaurant quality: {e}")
            # Fallback evaluation
            return {
                "sufficient": len(restaurants) >= self.minimum_restaurant_threshold,
                "confidence": 5,
                "reasoning": f"Fallback evaluation: {len(restaurants)} restaurants found",
                "missing_aspects": []
            }

    def _fallback_keyword_matching(self, restaurants: List[Dict[str, Any]], raw_query: str) -> List[Dict[str, Any]]:
        """
        Fallback method when AI analysis fails - simple keyword matching
        """
        logger.info("🔄 Using fallback keyword matching")

        query_keywords = raw_query.lower().split()
        matched_restaurants = []

        for restaurant in restaurants:
            # Check name, cuisine tags, and description for keyword matches
            text_to_search = (
                restaurant.get('name', '').lower() + ' ' +
                ' '.join(restaurant.get('cuisine_tags', [])).lower() + ' ' +
                restaurant.get('raw_description', '').lower()
            )

            # Count keyword matches
            matches = sum(1 for keyword in query_keywords if keyword in text_to_search)

            if matches > 0:
                restaurant_copy = restaurant.copy()
                restaurant_copy['_ai_relevance_score'] = min(10, matches * 2)  # Scale matches to score
                restaurant_copy['_ai_reasoning'] = f"Fallback matching: {matches} keyword matches"
                matched_restaurants.append(restaurant_copy)

        # Sort by score and return top results
        matched_restaurants.sort(key=lambda r: r.get('_ai_relevance_score', 0), reverse=True)
        return matched_restaurants[:10]  # Limit to top 10

    def _create_restaurants_summary(self, restaurants: List[Dict[str, Any]]) -> str:
        """Create a formatted summary of restaurants for evaluation"""
        if not restaurants:
            return "No restaurants found"

        formatted = []
        for restaurant in restaurants:
            name = restaurant.get('name', 'Unknown')
            tags = ', '.join(restaurant.get('cuisine_tags', []))
            score = restaurant.get('_ai_relevance_score', 'N/A')
            reasoning = restaurant.get('_ai_reasoning', 'No reasoning provided')

            formatted.append(f"• {name} (Score: {score}) - {tags} - {reasoning}")

        return "\n".join(formatted)

    def _create_database_response(
        self, 
        database_restaurants: List[Dict[str, Any]], 
        evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create response indicating database content should be used"""
        # Clean up AI metadata before returning
        cleaned_restaurants = []
        for restaurant in database_restaurants:
            cleaned = restaurant.copy()
            # Remove AI analysis fields
            cleaned.pop('_ai_relevance_score', None)
            cleaned.pop('_ai_reasoning', None)
            cleaned_restaurants.append(cleaned)

        return {
            "has_database_content": True,
            "database_results": cleaned_restaurants,
            "content_source": "database",
            "evaluation_details": evaluation_result,
            "skip_web_search": True
        }

    def _create_web_search_response(self, reason: str) -> Dict[str, Any]:
        """Create response indicating web search should be used"""
        return {
            "has_database_content": False,
            "database_results": [],
            "content_source": "web_search",
            "evaluation_details": {
                "sufficient": False,
                "reason": reason,
                "details": {}
            },
            "skip_web_search": False
        }

    # ============ UTILITY METHODS ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about database search performance"""
        try:
            from utils.database import get_database
            db = get_database()
            stats = db.get_database_stats()
            stats["optimization"] = "single_api_call_batch_analysis_fixed"
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    def set_minimum_threshold(self, new_threshold: int):
        """Update the minimum restaurant threshold"""
        old_threshold = self.minimum_restaurant_threshold
        self.minimum_restaurant_threshold = new_threshold
        logger.info(f"Updated minimum restaurant threshold: {old_threshold} → {new_threshold}")