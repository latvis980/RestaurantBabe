# agents/database_search_agent.py
"""
CORRECTED Database Search Agent - 4-Step Flow Implementation

Implements the exact flow you specified:
1. Get destination from query 
2. Extract basic data (IDs, names, cuisine_tags) from database for that city
3. Analyze as batch using AI with search queries (not raw query), return matching IDs
4. Fetch full descriptions and sources for selected restaurants using IDs

Key improvements:
- Uses search_queries (optimized for filtering) instead of verbose raw_query
- Only fetches basic data in step 2, full data in step 4
- Proper 4-step separation with clear logging
- Efficient database queries
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
    Implements proper 4-step database search flow

    Flow:
    1. Extract destination from query analysis
    2. Get basic restaurant data (id, name, cuisine_tags) for the city
    3. AI batch analysis using search_queries to filter relevant restaurants
    4. Fetch full details (descriptions, sources) for selected restaurants by ID
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

        # AI filtering prompt - uses search_queries (optimized) not raw_query (verbose)
        self.batch_analysis_prompt = ChatPromptTemplate.from_template("""
        SEARCH INTENT: {search_queries}
        LOCATION: {destination}

        You are analyzing restaurants from our database to find matches for this search intent.

        RESTAURANT LIST (Basic Data):
        {restaurants_text}

        TASK: Return restaurant IDs that match the search intent.

        IMPORTANT: Use the exact numbers shown after "ID:" in the list above. For example, if you see "ID: 5 | Silo Coffee", use 5 as the id.

        MATCHING CRITERIA:
        - Cuisine type alignment with search terms
        - Dining style match (casual, fine dining, wine bar, etc.)
        - Special features mentioned in search (vegan, natural wine, coffee, etc.)
        - General atmosphere/vibe from cuisine tags

        OUTPUT: Return ONLY valid JSON with matching restaurant IDs:
        {{
            "selected_restaurants": [
                {{
                    "id": "ID",
                    "relevance_score": score,
                    "reasoning": "why this matches the search intent"
                }}
            ]
        }}

        Include restaurants that are good matches. Focus on quality over quantity.
        """)

        logger.info("✅ Corrected DatabaseSearchAgent initialized (4-step flow)")

    @log_function_call
    def search_and_evaluate(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        CORRECTED: Main method implementing the exact 4-step flow

        Steps:
        1. Get destination
        2. Extract basic data (id, name, cuisine_tags) for city
        3. AI batch filter using search_queries
        4. Fetch full details for selected restaurants
        """
        try:
            logger.info("🗃️ STARTING 4-STEP DATABASE SEARCH")

            # STEP 1: Get destination from query analysis
            destination = query_analysis.get("destination", "Unknown")
            raw_query = query_analysis.get("raw_query", query_analysis.get("query", ""))
            search_queries = query_analysis.get("search_queries", [])

            if destination == "Unknown":
                logger.info("⚠️ STEP 1 FAILED: No destination detected")
                return self._create_empty_response("no_destination", raw_query)

            logger.info(f"✅ STEP 1 COMPLETE: Destination = {destination}")

            # Validate search queries for filtering
            if not search_queries:
                logger.warning("⚠️ No search_queries available, using raw_query as fallback")
                search_queries = [raw_query]

            # STEP 2: Extract basic data (id, name, cuisine_tags) for the city
            logger.info(f"📋 STEP 2: Extracting basic restaurant data for {destination}")
            basic_restaurants = self._get_basic_restaurant_data(destination)

            if not basic_restaurants:
                logger.info("📭 STEP 2 COMPLETE: No restaurants found in database for this city")
                return self._create_empty_response("no_restaurants_in_city", raw_query)

            logger.info(f"✅ STEP 2 COMPLETE: Found {len(basic_restaurants)} restaurants")
            logger.debug(f"📊 Basic data fields: {list(basic_restaurants[0].keys()) if basic_restaurants else 'none'}")

            # STEP 3: AI batch analysis using search_queries to filter relevant restaurants
            logger.info(f"🧠 STEP 3: AI filtering using search queries: {search_queries}")
            filtered_restaurants = self._batch_filter_restaurants(
                basic_restaurants, 
                search_queries, 
                destination
            )

            if not filtered_restaurants:
                logger.info("📭 STEP 3 COMPLETE: No relevant restaurants after AI filtering")
                return self._create_empty_response("no_relevant_results", raw_query)

            logger.info(f"✅ STEP 3 COMPLETE: AI selected {len(filtered_restaurants)} relevant restaurants")

            # STEP 4: Fetch full details (descriptions, sources) for selected restaurants
            logger.info(f"📄 STEP 4: Fetching full details for {len(filtered_restaurants)} selected restaurants")
            detailed_restaurants = self._get_full_restaurant_details(filtered_restaurants)

            logger.info(f"✅ STEP 4 COMPLETE: Retrieved full details for {len(detailed_restaurants)} restaurants")

            # Log description availability
            with_descriptions = sum(1 for r in detailed_restaurants if r.get('raw_description'))
            logger.info(f"📄 Description coverage: {with_descriptions}/{len(detailed_restaurants)} restaurants have descriptions")

            # Return final results
            return {
                "database_restaurants": detailed_restaurants,  # Full restaurant data
                "has_database_content": len(detailed_restaurants) > 0,
                "restaurant_count": len(detailed_restaurants),
                "destination": destination,
                "raw_query": raw_query,  # Preserve for content evaluation
                "search_flow": "4_step_corrected"  # Debug flag
            }

        except Exception as e:
            logger.error(f"❌ Error in 4-step database search: {e}")
            error_response = self._create_empty_response(f"error: {str(e)}", raw_query)
            return error_response

    def _get_basic_restaurant_data(self, city: str) -> List[Dict[str, Any]]:
        """
        STEP 2: Get basic restaurant data (id, name, cuisine_tags) for efficient filtering

        Only retrieves essential data needed for AI filtering - NOT full descriptions
        """
        try:
            from utils.database import get_database
            db = get_database()

            logger.debug(f"🔍 Querying database for basic restaurant data in {city}")

            # Query ONLY basic fields needed for filtering
            result = db.supabase.table('restaurants')\
                .select('id, name, cuisine_tags, mention_count')\
                .eq('city', city)\
                .order('mention_count', desc=True)\
                .limit(100)\
                .execute()

            restaurants = result.data or []

            logger.debug(f"📊 Retrieved {len(restaurants)} restaurants with basic data only")

            return restaurants

        except Exception as e:
            logger.error(f"❌ Error getting basic restaurant data for {city}: {e}")
            return []

    def _batch_filter_restaurants(
        self, 
        restaurants: List[Dict[str, Any]], 
        search_queries: List[str], 
        destination: str
    ) -> List[Dict[str, Any]]:
        """
        STEP 3: AI batch analysis using search_queries (optimized) not raw_query (verbose)

        Uses search_queries which are concise and optimized for filtering
        """
        try:
            # Format search queries for AI prompt
            search_intent = " | ".join(search_queries)

            # Create text representation of basic restaurant data
            restaurants_text = self._compile_basic_restaurants_for_analysis(restaurants)

            logger.info(f"🧠 AI filtering {len(restaurants)} restaurants using search intent: '{search_intent}'")
            logger.debug(f"📝 Compiled text length: {len(restaurants_text)} characters")

            # Single API call to filter restaurants
            chain = self.batch_analysis_prompt | self.llm

            response = chain.invoke({
                "search_queries": search_intent,
                "destination": destination,
                "restaurants_text": restaurants_text
            })

            # Parse AI response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                analysis_result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI response as JSON: {e}")
                logger.error(f"Raw response: {content[:500]}...")
                return self._fallback_keyword_matching(restaurants, search_queries)

            # Map selected restaurant IDs back to basic restaurant data
            selected_restaurants = self._map_selected_restaurants(
                analysis_result.get("selected_restaurants", []),
                restaurants
            )

            logger.info(f"✅ AI filtered to {len(selected_restaurants)} relevant restaurants")

            return selected_restaurants

        except Exception as e:
            logger.error(f"❌ Error in AI batch filtering: {e}")
            # Fallback to simple keyword matching
            return self._fallback_keyword_matching(restaurants, search_queries)

    def _compile_basic_restaurants_for_analysis(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Compile basic restaurant data for AI analysis - ONLY basic fields

        Format: ID: x | Name | Tags: cuisine1, cuisine2 | Mentions: x
        """
        compiled_text = []

        for restaurant in restaurants:
            restaurant_id = restaurant.get('id', 'unknown')
            name = restaurant.get('name', 'Unknown')
            cuisine_tags = ', '.join(restaurant.get('cuisine_tags', []))
            mention_count = restaurant.get('mention_count', 1)

            # Compact format for efficient AI processing - NO descriptions here
            restaurant_entry = f"ID: {restaurant_id} | {name} | Tags: {cuisine_tags} | Mentions: {mention_count}"
            compiled_text.append(restaurant_entry)

        return "\n".join(compiled_text)

    def _map_selected_restaurants(
        self, 
        selected_data: List[Dict[str, Any]], 
        all_restaurants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map AI-selected restaurant IDs back to basic restaurant data

        Preserves AI reasoning for later use
        """
        selected_restaurants = []

        # Create lookup dict for fast access
        restaurant_lookup = {str(r.get('id')): r for r in all_restaurants}

        for selection in selected_data:
            restaurant_id = str(selection.get('id', ''))

            if restaurant_id in restaurant_lookup:
                restaurant = restaurant_lookup[restaurant_id].copy()

                # Add AI filtering metadata for debugging
                restaurant['_relevance_score'] = selection.get('relevance_score', 0)
                restaurant['_reasoning'] = selection.get('reasoning', '')

                selected_restaurants.append(restaurant)
            else:
                logger.warning(f"⚠️ AI selected restaurant ID {restaurant_id} not found in basic data")

        return selected_restaurants

    def _parse_sources_field(self, restaurant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the sources field from TEXT to actual list

        The sources column in Supabase is stored as TEXT but contains JSON-like strings
        We need to convert these to actual Python lists
        """
        try:
            sources_raw = restaurant.get('sources', [])

            # If it's already a list, return as-is
            if isinstance(sources_raw, list):
                restaurant['sources'] = sources_raw
                return restaurant

            # If it's a string that looks like a JSON array, parse it
            if isinstance(sources_raw, str) and sources_raw.strip():
                try:
                    # Try JSON parsing first
                    sources_list = json.loads(sources_raw)
                    if isinstance(sources_list, list):
                        restaurant['sources'] = sources_list
                    else:
                        # If JSON parsing returns non-list, wrap in list
                        restaurant['sources'] = [str(sources_list)]
                except json.JSONDecodeError:
                    try:
                        # Try ast.literal_eval as fallback
                        sources_list = ast.literal_eval(sources_raw)
                        if isinstance(sources_list, list):
                            restaurant['sources'] = sources_list
                        else:
                            restaurant['sources'] = [str(sources_list)]
                    except (ValueError, SyntaxError):
                        # If all parsing fails, treat as single source
                        restaurant['sources'] = [sources_raw]
            else:
                # Empty or None sources
                restaurant['sources'] = []

            return restaurant

        except Exception as e:
            logger.error(f"Error parsing sources for restaurant {restaurant.get('name', 'Unknown')}: {e}")
            restaurant['sources'] = []
            return restaurant

    def _get_full_restaurant_details(self, filtered_restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        STEP 4: Fetch full details (descriptions, sources) for AI-selected restaurants
        FIXED: Parse sources field from TEXT to list
        """
        try:
            from utils.database import get_database
            db = get_database()

            detailed_restaurants = []
            restaurant_ids = [str(r.get('id')) for r in filtered_restaurants]

            logger.info(f"📋 Fetching full details for {len(restaurant_ids)} AI-selected restaurants")

            # Get complete restaurant data for selected IDs
            for restaurant_id in restaurant_ids:
                try:
                    # Query ALL fields for selected restaurants
                    result = db.supabase.table('restaurants')\
                        .select('*')\
                        .eq('id', restaurant_id)\
                        .execute()

                    if result.data:
                        full_restaurant = result.data[0]

                        # FIXED: Parse sources field from TEXT to list
                        full_restaurant = self._parse_sources_field(full_restaurant)

                        # Preserve AI filtering metadata from step 3
                        filtered_restaurant = next(
                            (r for r in filtered_restaurants if str(r.get('id')) == restaurant_id), 
                            {}
                        )

                        # Add AI analysis metadata
                        full_restaurant['_relevance_score'] = filtered_restaurant.get('_relevance_score', 0)
                        full_restaurant['_reasoning'] = filtered_restaurant.get('_reasoning', '')

                        # Log sources parsing success
                        sources_count = len(full_restaurant.get('sources', []))
                        sources_preview = full_restaurant.get('sources', [])[:2] if sources_count > 0 else []
                        logger.debug(f"✅ Restaurant {restaurant_id} sources parsed: {sources_count} sources - {sources_preview}")

                        # Log description availability for debugging
                        if not full_restaurant.get('raw_description'):
                            logger.warning(f"⚠️ Restaurant {restaurant_id} ({full_restaurant.get('name')}) missing description")
                        else:
                            desc_length = len(full_restaurant.get('raw_description', ''))
                            logger.debug(f"✅ Restaurant {restaurant_id} has description: {desc_length} chars")

                        detailed_restaurants.append(full_restaurant)
                    else:
                        logger.warning(f"⚠️ Restaurant {restaurant_id} not found when fetching full details")

                except Exception as e:
                    logger.error(f"❌ Error getting full details for restaurant {restaurant_id}: {e}")
                    continue

            logger.info(f"✅ Retrieved full details for {len(detailed_restaurants)} restaurants")

            # Log final sources summary
            total_sources = sum(len(r.get('sources', [])) for r in detailed_restaurants)
            restaurants_with_sources = sum(1 for r in detailed_restaurants if r.get('sources'))
            logger.info(f"📊 Sources summary: {restaurants_with_sources}/{len(detailed_restaurants)} restaurants have sources, {total_sources} total sources")

            return detailed_restaurants

        except Exception as e:
            logger.error(f"❌ Error getting full restaurant details: {e}")
            # Fallback: return basic data if full details fail
            logger.warning("⚠️ Falling back to basic restaurant data")
            return filtered_restaurants

    def _fallback_keyword_matching(self, restaurants: List[Dict[str, Any]], search_queries: List[str]) -> List[Dict[str, Any]]:
        """
        Simple keyword matching fallback when AI filtering fails
        """
        logger.info("🔧 Using fallback keyword matching")

        # Combine all search terms
        all_terms = []
        for query in search_queries:
            all_terms.extend(query.lower().split())

        matched_restaurants = []

        for restaurant in restaurants:
            name = restaurant.get('name', '').lower()
            cuisine_tags = [tag.lower() for tag in restaurant.get('cuisine_tags', [])]

            # Check if any search term matches name or cuisine tags
            name_match = any(term in name for term in all_terms)
            tag_match = any(term in tag for tag in all_terms for term in all_terms)

            if name_match or tag_match:
                restaurant['_relevance_score'] = 0.5  # Fallback score
                restaurant['_reasoning'] = 'keyword_fallback_match'
                matched_restaurants.append(restaurant)

        logger.info(f"🔧 Fallback matched {len(matched_restaurants)} restaurants")
        return matched_restaurants

    def _create_empty_response(self, reason: str, raw_query: str = "") -> Dict[str, Any]:
        """
        Create standardized empty response for various failure scenarios
        """
        return {
            "database_restaurants": [],
            "has_database_content": False,
            "restaurant_count": 0,
            "destination": "Unknown",
            "raw_query": raw_query,
            "search_flow": "4_step_corrected",
            "empty_reason": reason
        }

    def create_debug_file(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Create debug file for troubleshooting - shows the corrected 4-step flow
        """
        try:
            timestamp = f"{int(time.time())}"
            filename = f"database_search_debug_{timestamp}.txt"
            filepath = os.path.join(tempfile.gettempdir(), filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DATABASE SEARCH DEBUG - 4-STEP CORRECTED FLOW\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Query: {data.get('raw_query', 'Unknown')}\n")
                f.write(f"Destination: {data.get('destination', 'Unknown')}\n")
                f.write(f"Flow Type: {data.get('search_flow', 'unknown')}\n\n")

                restaurants = data.get('database_restaurants', [])
                f.write(f"RESULTS: {len(restaurants)} restaurants found\n")
                f.write("-" * 40 + "\n")

                for i, restaurant in enumerate(restaurants[:10], 1):  # Show first 10
                    f.write(f"{i}. {restaurant.get('name', 'Unknown')}\n")
                    f.write(f"   ID: {restaurant.get('id')}\n")
                    f.write(f"   Cuisine Tags: {', '.join(restaurant.get('cuisine_tags', []))}\n")
                    f.write(f"   AI Score: {restaurant.get('_relevance_score', 'N/A')}\n")
                    f.write(f"   AI Reasoning: {restaurant.get('_reasoning', 'N/A')}\n")
                    f.write(f"   Has Description: {'Yes' if restaurant.get('raw_description') else 'No'}\n")
                    f.write(f"   Description Length: {len(restaurant.get('raw_description', ''))} chars\n")
                    f.write("\n")

                return filepath
        except Exception as e:
            logger.error(f"❌ Error creating debug file: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about database search performance
        """
        try:
            from utils.database import get_database
            db = get_database()
            stats = db.get_database_stats()
            stats["agent_type"] = "4_step_corrected_database_search"
            stats["flow_steps"] = [
                "1. Get destination", 
                "2. Extract basic data (id, name, cuisine_tags)",
                "3. AI filter using search_queries", 
                "4. Fetch full details by ID"
            ]
            return stats
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {"error": str(e)}