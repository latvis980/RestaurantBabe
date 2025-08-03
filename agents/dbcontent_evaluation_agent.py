# agents/dbcontent_evaluation_agent.py
"""
ENHANCED Content Evaluation Agent with Restaurant Selection

Enhanced from the simplified version to add:
1. Restaurant selection from full database results
2. Splitting into database_restaurants_final vs database_restaurants_hybrid
3. Better evaluation that analyzes descriptions for matching
4. Maintains the clean simplified flow and orchestrator compatibility
"""

import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
import json

from utils.debug_utils import log_function_call, dump_chain_state

logger = logging.getLogger(__name__)

class ContentEvaluationAgent:
    """
    ENHANCED: Single prompt evaluator with restaurant selection and splitting

    Flow:
    1. Receives restaurant data from database search agent
    2. Evaluates AND selects best matching restaurants with single AI call
    3. Splits into final vs hybrid categories based on quantity and quality
    4. Routes: database sufficient OR web search (with hybrid mode)
    5. Maintains all orchestrator compatibility
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for evaluation
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_evaluation', 3072)  # Increased for selection
        )

        # Will be injected by orchestrator to avoid circular imports
        self.brave_search_agent = None

        logger.info("✅ ENHANCED ContentEvaluationAgent initialized with restaurant selection")

    def set_brave_search_agent(self, brave_search_agent):
        """Inject BraveSearchAgent to avoid circular imports"""
        self.brave_search_agent = brave_search_agent
        logger.info("🔗 BraveSearchAgent injected into ContentEvaluationAgent")

    @log_function_call
    def evaluate_and_route(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Complete evaluation workflow with restaurant selection and splitting

        Maintains orchestrator compatibility while adding restaurant selection
        """
        try:
            logger.info("🧠 STARTING ENHANCED EVALUATION WITH RESTAURANT SELECTION")

            # Extract required data (business logic moved here from orchestrator)
            database_restaurants = pipeline_data.get("database_restaurants", [])
            raw_query = pipeline_data.get("raw_query", pipeline_data.get("query", ""))
            destination = pipeline_data.get("destination", "Unknown")

            logger.info(f"🧠 Evaluating: '{raw_query}' in {destination}")
            logger.info(f"📊 Database restaurants: {len(database_restaurants)}")

            # Quick evaluation for empty results
            if not database_restaurants:
                logger.info("📝 No database content - triggering web search")
                return self._trigger_web_search_workflow(pipeline_data, "No restaurants found in database")

            # ENHANCED: AI evaluation + restaurant selection in single call
            evaluation = self._evaluate_and_select_with_ai(database_restaurants, raw_query, destination)

            database_sufficient = evaluation.get('database_sufficient', False)
            trigger_web_search = evaluation.get('trigger_web_search', True)
            selected_restaurants_data = evaluation.get('selected_restaurants', [])

            # Map selected restaurant IDs back to full restaurant data
            selected_restaurants = self._map_selected_restaurants(selected_restaurants_data, database_restaurants)

            logger.info(f"🎯 Database sufficient: {database_sufficient}")
            logger.info(f"🎯 Selected restaurants: {len(selected_restaurants)}")
            logger.info(f"🌐 Trigger web search: {trigger_web_search}")
            logger.info(f"💭 Reasoning: {evaluation.get('reasoning', 'No reasoning')}")

            # ENHANCED: Route with restaurant splitting
            if database_sufficient:
                return self._use_database_content_final(pipeline_data, selected_restaurants, evaluation)
            else:
                return self._trigger_web_search_with_hybrid(pipeline_data, selected_restaurants, evaluation)

        except Exception as e:
            logger.error(f"❌ Error in enhanced evaluation and routing: {e}")
            return self._handle_evaluation_error(pipeline_data, e)

    def _evaluate_and_select_with_ai(self, restaurants: List[Dict], raw_query: str, destination: str) -> Dict[str, Any]:
        """
        ENHANCED: Single AI call that evaluates AND selects best matching restaurants

        Based on the simplified version but now includes restaurant selection
        """
        try:
            # Format complete restaurant data for AI analysis (include descriptions for selection)
            restaurants_data = self._format_restaurants_for_selection(restaurants)

            # ENHANCED PROMPT: Evaluation + Selection in single call
            evaluation_prompt = f"""You are evaluating database restaurant results for a user query AND selecting the best matches.

USER QUERY: "{raw_query}"
DESTINATION: {destination}
DATABASE RESULTS: {len(restaurants)} restaurants found

{restaurants_data}

**STAGE 1: EVALUATE**: Are these database results sufficient, or do we need web search?

CRITERIA:
1. Query Match: Do restaurants match what user wants? (cuisine, style, price range, etc.)
2. Quantity: Enough variety? (4+ = usually sufficient)
3. Quality: Meaningful details and relevance?

**STAGE 2: SELECT**: Choose the BEST matching restaurants from the list above.
- Analyze descriptions carefully, not just names and cuisine tags
- Only select restaurants that truly match the user's query
- Include relevance scores and reasoning for each selection

**STAGE 3: ROUTE** 
FOLLOW THIS LOGIC:
• Perfect matches + sufficient quantity (4+) → USE DATABASE
• Poor matches → TRIGGER WEB SEARCH (discard results)  
• Good matches but need variety → TRIGGER WEB SEARCH (preserve matching results for hybrid)
• Partial matches → TRIGGER WEB SEARCH (preserve matching results for hybrid)
• No results → TRIGGER WEB SEARCH (discard)

HYBRID MODE (preserve + supplement):
In reasoning, use phrases like: "good matches but need more variety", "limited options supplement", "relevant results but too few"

DISCARD MODE (start fresh):
In reasoning, use phrases like: "poor matches for the query", "doesn't match requirements", "completely irrelevant"

Return ONLY JSON:
{{
    "database_sufficient": true/false,
    "trigger_web_search": true/false, 
    "reasoning": "brief explanation",
    "quality_score": 0.8,
    "selected_restaurants": [
        {{
            "id": "restaurant_id",
            "relevance_score": 0.9,
            "match_reasoning": "why this restaurant matches the query"
        }}
    ]
}}

IMPORTANT: Use exact restaurant IDs from the data above. Only select restaurants that match the user's query."""

            # Get AI evaluation + selection
            response = self.llm.invoke(evaluation_prompt)

            # Parse response (same logic as simplified version)
            content = response.content.strip()

            # Clean up JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            content = content.replace('```', '').strip()
            if content.startswith('json'):
                content = content[4:].strip()

            if not content:
                logger.warning("⚠️ AI returned empty evaluation")
                raise ValueError("Empty AI response")

            logger.debug(f"🔍 AI evaluation response: {content}")

            # Parse JSON
            try:
                evaluation = json.loads(content)
            except json.JSONDecodeError as json_error:
                logger.error(f"❌ JSON parsing failed: {json_error}")
                logger.error(f"📄 Raw content: '{content}'")
                raise ValueError(f"Invalid JSON response: {content[:100]}...")

            # Validate required fields
            required_fields = ['database_sufficient', 'trigger_web_search', 'reasoning']
            missing_fields = [field for field in required_fields if field not in evaluation]

            if missing_fields:
                logger.warning(f"⚠️ Missing fields: {missing_fields}")
                evaluation.setdefault('database_sufficient', False)
                evaluation.setdefault('trigger_web_search', True)
                evaluation.setdefault('reasoning', 'Incomplete evaluation')
                evaluation.setdefault('quality_score', 0.3)
                evaluation.setdefault('selected_restaurants', [])

            # Log selection results
            selected_count = len(evaluation.get('selected_restaurants', []))
            logger.info(f"✅ AI evaluation complete: {evaluation.get('reasoning', 'No reasoning')}")
            logger.info(f"🎯 Selected {selected_count}/{len(restaurants)} restaurants")

            return evaluation

        except Exception as e:
            logger.error(f"❌ Error in AI evaluation and selection: {e}")
            raise e

    def _format_restaurants_for_selection(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Format restaurant data for AI evaluation - include descriptions for proper selection
        """
        formatted_restaurants = []

        for restaurant in restaurants:
            restaurant_id = restaurant.get('id', 'unknown')
            name = restaurant.get('name', 'Unknown')
            cuisine_tags = ', '.join(restaurant.get('cuisine_tags', []))

            # Include description for better matching
            description = restaurant.get('raw_description', restaurant.get('description', ''))
            description_preview = description[:300] + "..." if len(description) > 300 else description

            mention_count = restaurant.get('mention_count', 1)

            # Format with key details for evaluation and selection
            restaurant_entry = f"""ID: {restaurant_id} | {name}
Cuisine: {cuisine_tags} | Mentions: {mention_count}
Description: {description_preview}
---"""

            formatted_restaurants.append(restaurant_entry)

        return "\n".join(formatted_restaurants)

    def _map_selected_restaurants(self, selected_data: List[Dict[str, Any]], all_restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map AI-selected restaurant IDs back to FULL restaurant data with selection metadata

        CRITICAL: This preserves ALL restaurant data (name, description, sources, etc.)
        """
        selected_restaurants = []

        # Create lookup dict for fast access to COMPLETE restaurant objects
        restaurant_lookup = {str(r.get('id')): r for r in all_restaurants}

        for selection in selected_data:
            restaurant_id = str(selection.get('id', ''))

            if restaurant_id in restaurant_lookup:
                # Get the COMPLETE restaurant object (with all fields)
                restaurant = restaurant_lookup[restaurant_id].copy()

                # Preserve ALL original data: name, raw_description, sources, cuisine_tags, etc.
                # The restaurant object already contains everything from database search agent step 4:
                # - id, name, cuisine_tags, mention_count  
                # - raw_description (full text)
                # - sources (where the data came from)
                # - All other database fields

                # Add AI selection metadata on top of existing data
                restaurant['_relevance_score'] = selection.get('relevance_score', 0)
                restaurant['_match_reasoning'] = selection.get('match_reasoning', '')
                restaurant['_ai_selected'] = True

                selected_restaurants.append(restaurant)

                # Log what data we're preserving
                desc_length = len(restaurant.get('raw_description', ''))
                sources_count = len(restaurant.get('sources', []))
                logger.debug(f"✅ Selected restaurant {restaurant_id}: {restaurant.get('name')} "
                           f"(desc: {desc_length} chars, sources: {sources_count})")
            else:
                logger.warning(f"⚠️ AI selected restaurant ID {restaurant_id} not found in database data")

        logger.info(f"🎯 Mapped {len(selected_restaurants)} complete restaurant objects with full data")
        return selected_restaurants

    def _use_database_content_final(self, pipeline_data: Dict[str, Any], selected_restaurants: List[Dict], evaluation: Dict) -> Dict[str, Any]:
        """
        ENHANCED: Handle database-only route with selected restaurants

        Returns database_restaurants_final for editor (4+ sufficient restaurants)
        """
        logger.info(f"✅ Using database content - {len(selected_restaurants)} selected restaurants")

        return {
            **pipeline_data,
            "evaluation_result": {
                "database_sufficient": True,
                "trigger_web_search": False,
                "content_source": "database",
                "reasoning": evaluation.get('reasoning', 'Database content sufficient'),
                "quality_score": evaluation.get('quality_score', 0.8),
                "evaluation_summary": {"reason": "database_sufficient"},
                "selected_count": len(selected_restaurants)
            },
            "content_source": "database",
            "skip_web_search": True,  # STANDARDIZED FLAG for orchestrator
            "database_restaurants_final": selected_restaurants,  # NEW: Final results for editor
            "database_restaurants_hybrid": [],  # Empty for database-only route
            "database_restaurants": selected_restaurants,  # LEGACY: Maintain compatibility
            "raw_query": pipeline_data.get("raw_query")  # PRESERVE
        }

    def _trigger_web_search_with_hybrid(self, pipeline_data: Dict[str, Any], selected_restaurants: List[Dict], evaluation: Dict) -> Dict[str, Any]:
        """
        ENHANCED: Handle web search route with hybrid mode support

        FIXED: Ensure search queries are properly passed to orchestrator search step
        """
        reasoning = evaluation.get('reasoning', 'Database insufficient')
        logger.info(f"🌐 Triggering web search workflow: {reasoning}")

        # Determine hybrid mode based on reasoning and selected restaurant quality
        use_hybrid = self._should_use_hybrid_mode(selected_restaurants, reasoning)

        # NEW: Split restaurants based on hybrid decision
        if use_hybrid:
            database_restaurants_final = []  # No final results - need web search
            database_restaurants_hybrid = selected_restaurants  # Preserve for hybrid
            content_source = "hybrid"
            logger.info(f"🔄 Hybrid mode: preserving {len(database_restaurants_hybrid)} selected restaurants")
        else:
            database_restaurants_final = []  # No final results - need web search
            database_restaurants_hybrid = []  # Discard all - start fresh
            content_source = "web_search"
            logger.info(f"🗑️ Discard mode: starting fresh web search")

        try:
            # Trigger web search when evaluation determines database is insufficient
            if self.brave_search_agent:
                logger.info("🚀 Executing web search through BraveSearchAgent")

                # FIXED: Prepare search data with multiple query formats for compatibility
                search_queries = pipeline_data.get('search_queries', [])
                english_queries = pipeline_data.get('english_queries', [])
                local_queries = pipeline_data.get('local_queries', [])

                # Ensure we have search queries in the expected format
                if not search_queries and (english_queries or local_queries):
                    search_queries = english_queries + local_queries
                    logger.info(f"🔧 Combined queries for search: {len(english_queries)} English + {len(local_queries)} local")

                # Fallback: generate from raw query if no queries available
                if not search_queries:
                    raw_query = pipeline_data.get('raw_query', '')
                    destination = pipeline_data.get('destination', 'Unknown')
                    if raw_query and destination != 'Unknown':
                        search_queries = [f"best restaurants {raw_query} {destination}"]
                        logger.info("🔧 Generated fallback search queries")

                destination = pipeline_data.get('destination', 'Unknown')
                query_metadata = {
                    'is_english_speaking': pipeline_data.get('is_english_speaking', True),
                    'local_language': pipeline_data.get('local_language')
                }

                # Execute search immediately
                search_results = self.brave_search_agent.search(search_queries, destination, query_metadata)

                logger.info(f"✅ Web search completed: {len(search_results)} results found")

                # ENHANCED: Return with proper restaurant splitting and FIXED search query passing
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": content_source,
                        "reasoning": reasoning,
                        "quality_score": evaluation.get('quality_score', 0.3),
                        "evaluation_summary": {"reason": "database_insufficient"},
                        "selected_count": len(selected_restaurants),
                        "hybrid_mode": use_hybrid
                    },
                    "content_source": content_source,
                    "skip_web_search": False,  # Let orchestrator handle search results we provide
                    "search_results": search_results,  # Results already found by internal search
                    "database_restaurants_final": database_restaurants_final,  # NEW: Final results (empty for web search)
                    "database_restaurants_hybrid": database_restaurants_hybrid,  # NEW: Hybrid results  
                    "database_restaurants": database_restaurants_hybrid,  # LEGACY: Maintain compatibility
                    "raw_query": pipeline_data.get("raw_query"),  # PRESERVE
                    # FIXED: Ensure search queries are available for orchestrator search step
                    "search_queries": search_queries,  # For orchestrator compatibility
                    "english_queries": english_queries,  # Original format
                    "local_queries": local_queries,  # Original format
                    "optimized_content": {
                        "database_restaurants": database_restaurants_hybrid,
                        "scraped_results": []  # Will be filled by scraper
                    }
                }
            else:
                logger.warning("⚠️ BraveSearchAgent not available - falling back to routing flags")

                # FIXED: Ensure search queries are passed even without BraveSearchAgent
                search_queries = pipeline_data.get('search_queries', [])
                english_queries = pipeline_data.get('english_queries', [])
                local_queries = pipeline_data.get('local_queries', [])

                if not search_queries and (english_queries or local_queries):
                    search_queries = english_queries + local_queries

                # Return routing decision for main web search pipeline
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": content_source,
                        "reasoning": reasoning,
                        "quality_score": evaluation.get('quality_score', 0.3),
                        "evaluation_summary": {"reason": "database_insufficient"},
                        "selected_count": len(selected_restaurants),
                        "hybrid_mode": use_hybrid
                    },
                    "content_source": content_source,
                    "skip_web_search": False,  # Allow main search step
                    "database_restaurants_final": database_restaurants_final,  # NEW: Empty for web search
                    "database_restaurants_hybrid": database_restaurants_hybrid,  # NEW: Hybrid or empty
                    "database_restaurants": database_restaurants_hybrid,  # LEGACY: Maintain compatibility
                    "raw_query": pipeline_data.get("raw_query"),  # PRESERVE
                    # FIXED: Pass search queries to orchestrator
                    "search_queries": search_queries,
                    "english_queries": english_queries,
                    "local_queries": local_queries,
                    "optimized_content": {
                        "database_restaurants": database_restaurants_hybrid,
                        "scraped_results": []
                    }
                }

        except Exception as e:
            logger.error(f"❌ Error in web search workflow: {e}")
            return self._handle_evaluation_error(pipeline_data, e)

    def _trigger_web_search_workflow(self, pipeline_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """
        Handle the web search route - for backwards compatibility (when no selected restaurants)
        """
        logger.info("🌐 Triggering web search workflow (no selection)")

        # This is the old method for when we have no selected restaurants
        database_restaurants = pipeline_data.get("database_restaurants", [])

        try:
            # Trigger web search when evaluation determines database is insufficient
            if self.brave_search_agent:
                logger.info("🚀 Executing web search through BraveSearchAgent")

                # Prepare search data
                search_queries = pipeline_data.get('search_queries', [])
                destination = pipeline_data.get('destination', 'Unknown')
                query_metadata = {
                    'is_english_speaking': pipeline_data.get('is_english_speaking', True),
                    'local_language': pipeline_data.get('local_language')
                }

                # Execute search immediately
                search_results = self.brave_search_agent.search(search_queries, destination, query_metadata)

                logger.info(f"✅ Web search completed: {len(search_results)} results found")

                # Determine if we should preserve database results for hybrid mode
                use_hybrid = self._should_use_hybrid_mode(database_restaurants, reasoning)

                if use_hybrid:
                    database_restaurants_final = []
                    database_restaurants_hybrid = database_restaurants  # Preserve all
                    content_source = "hybrid"
                    logger.info(f"🔄 Hybrid mode: preserving {len(database_restaurants_hybrid)} database restaurants")
                else:
                    database_restaurants_final = []
                    database_restaurants_hybrid = []  # Discard all
                    content_source = "web_search"
                    logger.info(f"🗑️ Discard mode: starting fresh web search")

                # Return with web search results
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": content_source,
                        "reasoning": reasoning,
                        "quality_score": 0.3,
                        "evaluation_summary": {"reason": "database_insufficient"},
                        "selected_count": 0,
                        "hybrid_mode": use_hybrid
                    },
                    "content_source": content_source,
                    "skip_web_search": False,
                    "search_results": search_results,
                    "database_restaurants_final": database_restaurants_final,  # NEW: Empty for web search
                    "database_restaurants_hybrid": database_restaurants_hybrid,  # NEW: Hybrid or empty
                    "database_restaurants": database_restaurants_hybrid,  # LEGACY: Maintain compatibility
                    "raw_query": pipeline_data.get("raw_query"),  # PRESERVE
                    "optimized_content": {
                        "database_restaurants": database_restaurants_hybrid,
                        "scraped_results": []  # Will be filled by scraper
                    }
                }
            else:
                logger.warning("⚠️ BraveSearchAgent not available - falling back to routing flags")
                # Return routing decision for main web search pipeline
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": "web_search",
                        "reasoning": reasoning,
                        "quality_score": 0.3,
                        "evaluation_summary": {"reason": "database_insufficient"},
                        "selected_count": 0,
                        "hybrid_mode": False
                    },
                    "content_source": "web_search",
                    "skip_web_search": False,  # Allow main search step
                    "database_restaurants_final": [],  # NEW: Empty for web search
                    "database_restaurants_hybrid": [],  # NEW: Empty
                    "database_restaurants": [],  # LEGACY: Empty
                    "raw_query": pipeline_data.get("raw_query"),  # PRESERVE
                    "optimized_content": {
                        "database_restaurants": [],
                        "scraped_results": []
                    }
                }

        except Exception as e:
            logger.error(f"❌ Error in web search workflow: {e}")
            return self._handle_evaluation_error(pipeline_data, e)

    def _handle_evaluation_error(self, pipeline_data: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """
        ENHANCED: Error handling that triggers web search and maintains new restaurant splitting
        """
        logger.error(f"❌ Evaluation error: {error}")

        dump_chain_state("content_evaluation_error", {
            "error": str(error),
            "raw_query": pipeline_data.get("raw_query", "unknown"),
            "database_count": len(pipeline_data.get("database_restaurants", []))
        })

        # When AI evaluation fails, trigger web search instead of using all database results
        logger.info("🔧 AI evaluation failed - triggering web search as fallback")

        try:
            # Try to execute web search immediately when evaluation fails
            if self.brave_search_agent:
                logger.info("🚀 Executing fallback web search through BraveSearchAgent")

                # Prepare search data
                search_queries = pipeline_data.get('search_queries', [])
                destination = pipeline_data.get('destination', 'Unknown')
                query_metadata = {
                    'is_english_speaking': pipeline_data.get('is_english_speaking', True),
                    'local_language': pipeline_data.get('local_language')
                }

                # Execute search as fallback
                search_results = self.brave_search_agent.search(search_queries, destination, query_metadata)

                logger.info(f"✅ Fallback web search completed: {len(search_results)} results found")

                # Return web search results with new restaurant splitting format
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": "web_search",
                        "reasoning": f"AI evaluation failed ({str(error)}) - using web search fallback",
                        "evaluation_summary": {"reason": "evaluation_error_web_search_fallback"},
                        "selected_count": 0,
                        "hybrid_mode": False
                    },
                    "evaluation_error": str(error),
                    "content_source": "web_search",
                    "trigger_web_search": True,
                    "skip_web_search": False,
                    "search_results": search_results,  # Results already found
                    "database_restaurants_final": [],  # NEW: Empty due to error
                    "database_restaurants_hybrid": [],  # NEW: Empty due to error
                    "database_restaurants": [],  # LEGACY: Empty due to error
                    "optimized_content": {
                        "database_restaurants": [],
                        "scraped_results": []  # Will be filled by scraper
                    }
                }
            else:
                logger.warning("⚠️ BraveSearchAgent not available for fallback - using routing flags")
                # Fallback: trigger web search when evaluation fails
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": "web_search",
                        "reasoning": f"AI evaluation failed ({str(error)}) - triggering web search",
                        "evaluation_summary": {"reason": "evaluation_error"},
                        "selected_count": 0,
                        "hybrid_mode": False
                    },
                    "evaluation_error": str(error),
                    "content_source": "web_search",
                    "trigger_web_search": True,
                    "skip_web_search": False,  # Allow main search step
                    "database_restaurants_final": [],  # NEW: Empty due to error
                    "database_restaurants_hybrid": [],  # NEW: Empty due to error
                    "database_restaurants": []  # LEGACY: Empty due to error
                }

        except Exception as search_error:
            logger.error(f"❌ Fallback web search also failed: {search_error}")
            # If web search also fails, return error response with new format
            return {
                **pipeline_data,
                "evaluation_result": {
                    "database_sufficient": False,
                    "trigger_web_search": True,
                    "content_source": "web_search",
                    "reasoning": f"Both AI evaluation and web search failed - using routing fallback",
                    "evaluation_summary": {"reason": "complete_evaluation_failure"},
                    "selected_count": 0,
                    "hybrid_mode": False
                },
                "evaluation_error": f"AI: {str(error)}, Search: {str(search_error)}",
                "content_source": "web_search",
                "trigger_web_search": True,
                "skip_web_search": False,
                "database_restaurants_final": [],  # NEW: Empty due to error
                "database_restaurants_hybrid": [],  # NEW: Empty due to error
                "database_restaurants": []  # LEGACY: Empty due to error
            }

    def _should_use_hybrid_mode(self, selected_restaurants: List[Dict], reasoning: str) -> bool:
        """
        ENHANCED: Determine hybrid mode based on selected restaurants and reasoning

        Updated to work with selected restaurants instead of all database restaurants
        """
        # Only consider hybrid if we have selected restaurants
        if len(selected_restaurants) == 0:
            logger.info("🗑️ No selected restaurants - discard mode")
            return False

        reasoning_lower = reasoning.lower()

        # HYBRID MODE: Preserve and supplement
        hybrid_indicators = [
            "good matches but need more variety",
            "limited options, supplement", 
            "relevant results but too few",
            "matches found but expand",
            "preserve these and find more",
            "need more variety",
            "too few choices",
            "expand selection",
            "supplement with additional",
            "more options needed",
            "limited variety",
            "few results",
            "additional restaurants"
        ]

        # DISCARD MODE: Start fresh (opposite of hybrid)
        discard_indicators = [
            "poor matches for the query",
            "doesn't match user requirements", 
            "wrong cuisine type entirely",
            "completely irrelevant",
            "start fresh with web search",
            "poor matches",
            "wrong cuisine",
            "irrelevant results",
            "doesn't match",
            "completely wrong",
            "poor quality matches"
        ]

        # Check for discard indicators first (they override hybrid)
        if any(indicator in reasoning_lower for indicator in discard_indicators):
            logger.info(f"🗑️ Discard mode detected: {reasoning}")
            return False

        # Check for hybrid indicators
        if any(indicator in reasoning_lower for indicator in hybrid_indicators):
            logger.info(f"🔄 Hybrid mode detected: {reasoning}")
            return True

        # ENHANCED: Default logic based on selected restaurant count and quality
        if 1 <= len(selected_restaurants) <= 3:
            # Check if selected restaurants have good relevance scores
            avg_score = sum(r.get('_relevance_score', 0) for r in selected_restaurants) / len(selected_restaurants)
            if avg_score >= 0.7:
                logger.info(f"🔄 High-quality hybrid mode: {len(selected_restaurants)} good matches (avg score: {avg_score:.2f})")
                return True
            else:
                logger.info(f"🗑️ Low-quality selected restaurants: avg score {avg_score:.2f} - discard mode")
                return False

        # If we have 4+ selected restaurants but AI still says web search, it's probably variety issues
        # Default to hybrid to preserve the good ones
        if len(selected_restaurants) >= 4:
            logger.info(f"🔄 Variety hybrid mode: preserving {len(selected_restaurants)} selected restaurants")
            return True

        # Fallback to hybrid if uncertain
        logger.info(f"🔄 Fallback hybrid mode: preserving {len(selected_restaurants)} selected restaurants")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        ENHANCED: Get evaluation statistics with selection metrics
        """
        return {
            "agent_type": "enhanced_content_evaluation_with_selection",
            "has_brave_search_agent": self.brave_search_agent is not None,
            "model": self.config.OPENAI_MODEL,
            "features": ["restaurant_selection", "final_hybrid_splitting", "single_ai_call"]
        }