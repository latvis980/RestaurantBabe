# agents/database_search_agent.py
"""
SIMPLIFIED Database Search Agent

Implements a clean two-step process:
1. Extract IDs + names + tags from database
2. Analyze in single API call to filter relevant restaurants

NO quality evaluation - that's handled by ContentEvaluationAgent
"""

import logging
from typing import Dict, List, Any, Optional
import json
import tempfile
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.debug_utils import log_function_call, dump_chain_state

logger = logging.getLogger(__name__)

class DatabaseSearchAgent:
    """
    SIMPLIFIED agent that only:
    1. Searches database for restaurants
    2. Filters them based on user query
    3. Returns results (no quality evaluation)
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for restaurant filtering
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('database_search', 2048)
        )

        # Single prompt for batch analysis
        self.batch_analysis_prompt = ChatPromptTemplate.from_template("""
USER QUERY: {{raw_query}}
LOCATION: {{destination}}

You are analyzing restaurants from our database to see which ones match the user's query.

RESTAURANT LIST:
{{restaurants_text}}

TASK: Analyze this list and return the restaurant IDs that best match the user's query.

MATCHING CRITERIA:
- Cuisine type match (Italian, Asian, French, etc.)
- Dining style match (casual, fine dining, wine bar, etc.)
- Special requirements (vegan, natural wine, coffee, etc.)
- General atmosphere/vibe alignment

OUTPUT: Return ONLY valid JSON with restaurant IDs that match:
{{
    "selected_restaurants": [
        {{
            "id": "restaurant_id",
            "relevance_score": 0.9,
            "reasoning": "why this matches"
        }}
    ]
}}

Include restaurants that are good matches, even if not perfect.
Prioritize quality over quantity.
""")

        logger.info("✅ Simplified DatabaseSearchAgent initialized")

    @log_function_call
    def search_and_extract(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        SIMPLIFIED: Main method that just searches and filters
        NO quality evaluation - that's ContentEvaluationAgent's job
        """
        try:
            logger.info("🗃️ STARTING DATABASE SEARCH")

            # Extract data from query analysis
            destination = query_analysis.get("destination", "Unknown")
            raw_query = query_analysis.get("raw_query", query_analysis.get("query", ""))

            if destination == "Unknown":
                logger.info("⚠️ No destination detected")
                return self._create_empty_response("no_destination")

            # STEP 1: Get all restaurants for the city
            all_restaurants = self._get_restaurants_for_city(destination)

            if not all_restaurants:
                logger.info("📭 No restaurants found in database for this city")
                return self._create_empty_response("no_restaurants_in_city")

            logger.info(f"📊 Found {len(all_restaurants)} restaurants for {destination}")

            # STEP 2: Filter restaurants in single API call
            filtered_restaurants = self._batch_filter_restaurants(
                all_restaurants, 
                raw_query, 
                destination
            )

            if not filtered_restaurants:
                logger.info("📭 No relevant restaurants after filtering")
                return self._create_empty_response("no_relevant_results")

            logger.info(f"✅ Found {len(filtered_restaurants)} relevant restaurants")

            # Return results WITHOUT quality evaluation
            return {
                "database_restaurants": filtered_restaurants,
                "has_content": len(filtered_restaurants) > 0,
                "restaurant_count": len(filtered_restaurants),
                "destination": destination
            }

        except Exception as e:
            logger.error(f"❌ Error in database search: {e}")
            return self._create_empty_response(f"error: {str(e)}")

    def _get_restaurants_for_city(self, city: str) -> List[Dict[str, Any]]:
        """Get all restaurants for a city from database"""
        try:
            from utils.database import get_database
            db = get_database()

            # Get restaurants ordered by mention count
            restaurants = db.get_restaurants_by_city(city, limit=100)
            return restaurants

        except Exception as e:
            logger.error(f"Error getting restaurants for {city}: {e}")
            return []

    def _batch_filter_restaurants(
        self, 
        restaurants: List[Dict[str, Any]], 
        raw_query: str, 
        destination: str
    ) -> List[Dict[str, Any]]:
        """
        Filter restaurants in a single API call
        """
        try:
            # Create temp file with restaurant data
            restaurants_text = self._compile_restaurants_for_analysis(restaurants)

            logger.info(f"🧠 Filtering {len(restaurants)} restaurants...")

            # Single API call to filter restaurants
            chain = self.batch_analysis_prompt | self.llm

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

            logger.info(f"✅ Filtered to {len(selected_restaurants)} restaurants")

            return selected_restaurants

        except Exception as e:
            logger.error(f"Error in batch filtering: {e}")
            # Fallback to simple keyword matching
            return self._fallback_keyword_matching(restaurants, raw_query)

    def _compile_restaurants_for_analysis(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Compile restaurant data into text format for AI analysis
        """
        compiled_text = []

        for restaurant in restaurants:
            restaurant_id = restaurant.get('id', 'unknown')
            name = restaurant.get('name', 'Unknown')
            cuisine_tags = ', '.join(restaurant.get('cuisine_tags', []))
            description = restaurant.get('raw_description', '')[:500] 
            sources = restaurant.get('sources', 1)

            # Format for analysis
            restaurant_entry = f"ID: {restaurant_id} | {name} | Tags: {cuisine_tags} | Sources: {sources} | Desc: {description}"
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

        # Create a lookup dict
        restaurant_lookup = {str(r.get('id')): r for r in all_restaurants}

        for selection in selected_data:
            restaurant_id = str(selection.get('id', ''))

            if restaurant_id in restaurant_lookup:
                restaurant = restaurant_lookup[restaurant_id].copy()

                # Add relevance metadata (for internal use)
                restaurant['_relevance_score'] = selection.get('relevance_score', 0)
                restaurant['_reasoning'] = selection.get('reasoning', '')

                selected_restaurants.append(restaurant)
            else:
                logger.warning(f"⚠️ Restaurant ID {restaurant_id} not found in lookup")

        return selected_restaurants

    def _fallback_keyword_matching(self, restaurants: List[Dict[str, Any]], raw_query: str) -> List[Dict[str, Any]]:
        """
        Simple keyword matching fallback when AI fails
        """
        logger.info("🔧 Using fallback keyword matching")

        query_lower = raw_query.lower()
        keywords = query_lower.split()

        matched_restaurants = []

        for restaurant in restaurants:
            # Check name
            name_lower = restaurant.get('name', '').lower()

            # Check cuisine tags
            cuisine_tags = [tag.lower() for tag in restaurant.get('cuisine_tags', [])]

            # Check description
            description_lower = restaurant.get('raw_description', '').lower()

            # Score based on matches
            score = 0
            for keyword in keywords:
                if keyword in name_lower:
                    score += 3
                if any(keyword in tag for tag in cuisine_tags):
                    score += 2
                if keyword in description_lower:
                    score += 1

            if score > 0:
                restaurant_copy = restaurant.copy()
                restaurant_copy['_relevance_score'] = score / (len(keywords) * 3)  # Normalize
                restaurant_copy['_reasoning'] = 'Keyword match fallback'
                matched_restaurants.append(restaurant_copy)

        # Sort by score and return top results
        matched_restaurants.sort(key=lambda x: x['_relevance_score'], reverse=True)
        return matched_restaurants[:20]  # Limit results

    def _create_empty_response(self, reason: str) -> Dict[str, Any]:
        """Create response when no restaurants found"""
        return {
            "database_restaurants": [],
            "has_content": False,
            "restaurant_count": 0,
            "reason": reason
        }

    def _create_temp_file(self, restaurants: List[Dict[str, Any]], query: str) -> str:
        """
        Create temporary file with restaurant data
        (Kept for compatibility but could be removed if not used)
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Query: {query}\n")
                f.write(f"Restaurant count: {len(restaurants)}\n\n")

                for i, restaurant in enumerate(restaurants, 1):
                    f.write(f"{i}. {restaurant.get('name', 'Unknown')}\n")
                    f.write(f"   ID: {restaurant.get('id')}\n")
                    f.write(f"   Tags: {', '.join(restaurant.get('cuisine_tags', []))}\n")
                    f.write(f"   Sources: {restaurant.get('sources', 1)}\n")
                    f.write(f"   Description: {restaurant.get('raw_description', '')[:200]}...\n")
                    f.write("\n")

                return f.name
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about database search performance"""
        try:
            from utils.database import get_database
            db = get_database()
            stats = db.get_database_stats()
            stats["agent_type"] = "simplified_database_search"
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}