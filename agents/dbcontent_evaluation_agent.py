# agents/dbcontent_evaluation_agent.py
"""
ENHANCED Content Evaluation Agent

Handles all business logic for:
1. Evaluating database results sufficiency 
2. Making routing decisions (database vs web search)
3. Triggering BraveSearchAgent when needed
4. Standardizing response structure for the pipeline
"""

import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

from utils.debug_utils import log_function_call, dump_chain_state

logger = logging.getLogger(__name__)

class ContentEvaluationAgent:
    """
    ENHANCED agent that:
    - Evaluates database content quality
    - Makes routing decisions (database vs web search)  
    - Triggers BraveSearchAgent when web search is needed
    - Handles all evaluation business logic
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for evaluation
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_evaluation', 1024)
        )

        # Evaluation prompt - database quality assessment
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_evaluation_system_prompt()),
            ("human", self._get_evaluation_human_prompt())
        ])

        # Create evaluation chain
        self.evaluation_chain = self.evaluation_prompt | self.llm

        # Will be injected by orchestrator to avoid circular imports
        self.brave_search_agent = None

        logger.info("✅ ENHANCED ContentEvaluationAgent initialized")

    def set_brave_search_agent(self, brave_search_agent):
        """Inject BraveSearchAgent to avoid circular imports"""
        self.brave_search_agent = brave_search_agent
        logger.info("🔗 BraveSearchAgent injected into ContentEvaluationAgent")

    def _get_evaluation_system_prompt(self) -> str:
        """System prompt for evaluating database content quality"""
        return """You are a restaurant recommendation evaluator.

    Your job is to decide if database results are sufficient for the user's query, or if web search is needed.

    EVALUATION CRITERIA:

    1. **Query Match**: Do the restaurants match what the user is asking for?
       - Cuisine type, dining style, price range, special requirements

    2. **Quantity**: Is there enough variety for the user to choose from?
       - 4+ restaurants = usually sufficient 
       - 1-3 restaurants = may need web search for more options
       - 0 restaurants = definitely need web search

    3. **Quality**: Are the restaurant details sufficient?
       - Name, location, cuisine type should be present
       - Descriptions should be meaningful

    DECISION LOGIC:
    - If database results are perfect matches with sufficient quantity (4+) → USE DATABASE
    - If results don't match query well at all → TRIGGER WEB SEARCH (discard database results)
    - If results match well but need more variety (1-3 good matches) → TRIGGER WEB SEARCH (preserve database results for hybrid)
    - If results partially match but descriptions are too brief → TRIGGER WEB SEARCH (preserve partial matches for hybrid)
    - If results match cuisine but wrong style/price → TRIGGER WEB SEARCH (preserve relevant ones for hybrid)
    - If no results → TRIGGER WEB SEARCH (nothing to preserve)

    HYBRID MODE INDICATORS:
    Use these phrases in your reasoning when database results should be PRESERVED and SUPPLEMENTED:
    - "good matches but need more variety"
    - "limited options, supplement with additional search"
    - "relevant results but too few choices"
    - "matches found but expand selection"
    - "preserve these and find more options"

    DISCARD MODE INDICATORS:
    Use these phrases when database results should be DISCARDED completely:
    - "poor matches for the query"
    - "doesn't match user requirements"
    - "wrong cuisine type entirely"
    - "completely irrelevant results"
    - "start fresh with web search"

    OUTPUT: JSON with decision and reasoning

    IMPORTANT: Always return valid JSON. Your reasoning phrase determines whether results are preserved (hybrid) or discarded (fresh search)."""

    def _get_evaluation_human_prompt(self) -> str:
        """Human prompt template for evaluation"""
        return """USER QUERY: "{{raw_query}}"
DESTINATION: {{destination}}

DATABASE RESTAURANTS ({{restaurant_count}} found):
{{restaurants_summary}}

Evaluate if these database results are sufficient for the user's query.

Return ONLY valid JSON in this exact format:
{{
    "database_sufficient": true/false,
    "trigger_web_search": true/false,
    "reasoning": "brief explanation of your decision",
    "quality_score": 0.8
}}"""

    @log_function_call
    def evaluate_and_route(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        MAIN METHOD: Complete evaluation workflow with routing and BraveSearchAgent integration.

        Takes the full pipeline data and returns standardized routing decision.
        Handles all business logic including web search triggering.

        Args:
            pipeline_data: Full pipeline data from orchestrator

        Returns:
            Dict with standardized routing decision and updated pipeline data
        """
        try:
            logger.info("🧠 STARTING COMPLETE EVALUATION AND ROUTING")

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

            # AI evaluation for non-empty results
            evaluation = self._evaluate_with_ai(database_restaurants, raw_query, destination)

            database_sufficient = evaluation.get('database_sufficient', False)
            trigger_web_search = evaluation.get('trigger_web_search', True)

            logger.info(f"🎯 Database sufficient: {database_sufficient}")
            logger.info(f"🌐 Trigger web search: {trigger_web_search}")
            logger.info(f"💭 Reasoning: {evaluation.get('reasoning', 'No reasoning')}")

            # Route based on evaluation
            if database_sufficient:
                return self._use_database_content(pipeline_data, database_restaurants, evaluation)
            else:
                return self._trigger_web_search_workflow(pipeline_data, evaluation.get('reasoning', 'Database insufficient'))

        except Exception as e:
            logger.error(f"❌ Error in evaluation and routing: {e}")
            return self._handle_evaluation_error(pipeline_data, e)


    def _use_database_content(self, pipeline_data: Dict[str, Any], database_restaurants: List[Dict], evaluation: Dict) -> Dict[str, Any]:
        """Handle the database content route - all business logic here"""
        logger.info("✅ Using database content")

        return {
            **pipeline_data,
            "evaluation_result": {
                "database_sufficient": True,
                "trigger_web_search": False,
                "content_source": "database",
                "reasoning": evaluation.get('reasoning', 'Database content sufficient'),
                "quality_score": evaluation.get('quality_score', 0.8),
                "evaluation_summary": {"reason": "database_sufficient"}
            },
            "content_source": "database",
            "skip_web_search": True,  # STANDARDIZED FLAG
            "database_restaurants": database_restaurants,  # STANDARDIZED for editor
            "raw_query": pipeline_data.get("raw_query")  # PRESERVE
        }

    def _trigger_web_search_workflow(self, pipeline_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """Handle the web search route - trigger BraveSearchAgent and return results"""
        logger.info("🌐 Triggering web search workflow")

        # Extract database restaurants for hybrid mode check
        database_restaurants = pipeline_data.get("database_restaurants", [])

        try:
            # FIXED: Actually trigger web search when evaluation fails or database is insufficient
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
                preserved_database_restaurants = []
                use_hybrid = self._should_use_hybrid_mode(database_restaurants, reasoning)
                if use_hybrid:
                    preserved_database_restaurants = database_restaurants
                    logger.info(f"💾 Preserving {len(preserved_database_restaurants)} database results for hybrid mode")
                    logger.info(f"🔄 Reason for hybrid: {reasoning}")

                # Return with web search results - make sure they flow to scraping
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": "hybrid" if preserved_database_restaurants else "web_search",
                        "reasoning": reasoning,
                        "quality_score": 0.3,
                        "evaluation_summary": {"reason": "database_insufficient"}
                    },
                    "content_source": "hybrid" if preserved_database_restaurants else "web_search",
                    "skip_web_search": False,
                    "search_results": search_results,
                    "database_restaurants": preserved_database_restaurants,  # PRESERVE for editor
                    "raw_query": pipeline_data.get("raw_query"),  # PRESERVE
                    "optimized_content": {
                        "database_restaurants": preserved_database_restaurants,
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
                        "evaluation_summary": {"reason": "database_insufficient"}
                    },
                    "content_source": "web_search",
                    "skip_web_search": False,  # Allow main search step
                    "database_restaurants": [],
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
        FIXED: Centralized error handling that triggers actual web search instead of just returning ALL database results
        """
        logger.error(f"❌ Evaluation error: {error}")

        dump_chain_state("content_evaluation_error", {
            "error": str(error),
            "raw_query": pipeline_data.get("raw_query", "unknown"),
            "database_count": len(pipeline_data.get("database_restaurants", []))
        })

        # FIXED: When AI evaluation fails, trigger web search instead of using all database results
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

                # Return web search results instead of database fallback
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": "web_search",
                        "reasoning": f"AI evaluation failed ({str(error)}) - using web search fallback",
                        "evaluation_summary": {"reason": "evaluation_error_web_search_fallback"}
                    },
                    "evaluation_error": str(error),
                    "content_source": "web_search",
                    "trigger_web_search": True,
                    "skip_web_search": False,  # FIXED: Don't skip - let orchestrator handle the results we provide
                    "search_results": search_results,  # Results already found
                    "database_restaurants": [],
                    "optimized_content": {
                        "database_restaurants": [],
                        "scraped_results": []  # Will be filled by scraper
                    }
                }
            else:
                logger.warning("⚠️ BraveSearchAgent not available for fallback - using routing flags")
                # Fallback: trigger web search when evaluation fails (old behavior but cleaner)
                return {
                    **pipeline_data,
                    "evaluation_result": {
                        "database_sufficient": False,
                        "trigger_web_search": True,
                        "content_source": "web_search",
                        "reasoning": f"AI evaluation failed ({str(error)}) - triggering web search",
                        "evaluation_summary": {"reason": "evaluation_error"}
                    },
                    "evaluation_error": str(error),
                    "content_source": "web_search",
                    "trigger_web_search": True,
                    "skip_web_search": False,  # Allow main search step
                    "database_restaurants": []
                }

        except Exception as search_error:
            logger.error(f"❌ Fallback web search also failed: {search_error}")
            # If web search also fails, we have no choice but to use a simple fallback
            return {
                **pipeline_data,
                "evaluation_result": {
                    "database_sufficient": False,
                    "trigger_web_search": True,
                    "content_source": "web_search",
                    "reasoning": f"Both AI evaluation and web search failed - using routing fallback",
                    "evaluation_summary": {"reason": "complete_evaluation_failure"}
                },
                "evaluation_error": f"AI: {str(error)}, Search: {str(search_error)}",
                "content_source": "web_search",
                "trigger_web_search": True,
                "skip_web_search": False,
                "database_restaurants": []
            }

    def _evaluate_with_ai(self, restaurants: List[Dict], raw_query: str, destination: str) -> Dict[str, Any]:
        """
        IMPROVED: Use AI to evaluate database content with better error handling
        """
        try:
            # Prepare restaurant summary for AI evaluation
            restaurants_summary = self._create_restaurants_summary(restaurants)

            # Format the evaluation prompt
            formatted_prompt = self.evaluation_prompt.format_messages(
                raw_query=raw_query,
                destination=destination,
                restaurant_count=len(restaurants),
                restaurants_summary=restaurants_summary
            )

            # Get AI evaluation
            response = self.evaluation_chain.invoke({"messages": formatted_prompt})

            # Parse response with improved error handling
            content = response.content.strip()

            # Clean up response - handle various formats
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Additional cleanup for common AI response patterns
            content = content.replace('```', '').strip()
            if content.startswith('json'):
                content = content[4:].strip()

            # Handle empty content
            if not content:
                logger.warning("⚠️ AI returned empty evaluation - using fallback")
                raise ValueError("Empty AI response")

            logger.debug(f"🔍 AI evaluation response: {content}")

            # Parse JSON with better error handling
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
                logger.warning(f"⚠️ AI evaluation missing fields: {missing_fields}")
                # Fill in missing fields with safe defaults
                evaluation.setdefault('database_sufficient', False)
                evaluation.setdefault('trigger_web_search', True)
                evaluation.setdefault('reasoning', 'Incomplete AI evaluation')
                evaluation.setdefault('quality_score', 0.3)

            logger.info(f"✅ AI evaluation successful: {evaluation.get('reasoning', 'No reasoning')}")
            return evaluation

        except Exception as e:
            logger.error(f"❌ Error in AI evaluation: {e}")
            # Re-raise to be handled by _handle_evaluation_error
            raise e

    def _create_restaurants_summary(self, restaurants: List[Dict]) -> str:
        """Create a concise summary of restaurants for AI evaluation"""
        if not restaurants:
            return "No restaurants available"

        summary_lines = []
        for i, restaurant in enumerate(restaurants[:10], 1):  # Limit to first 10
            name = restaurant.get('name', 'Unknown')
            cuisine_tags = restaurant.get('cuisine_tags', [])
            cuisine = ', '.join(cuisine_tags) if cuisine_tags else 'Unknown cuisine'
            mention_count = restaurant.get('mention_count', 0)
            sources = len(restaurant.get('sources', []))

            # Show brief description if available
            raw_description = restaurant.get('raw_description', '')

            # Basic restaurant info
            summary_lines.append(f"{i}. {name} - {cuisine}, Mentions: {mention_count}, Sources: {sources}")

            # Add description if it exists and has content
            if raw_description and raw_description.strip():  # ✅ FIXED: Check for actual content
                description_preview = raw_description[:300] + "..." if len(raw_description) > 300 else raw_description
                summary_lines.append(f"   Description: {description_preview}")
            else:
                summary_lines.append(f"   Description: [No description available]")

        if len(restaurants) > 10:
            summary_lines.append(f"... and {len(restaurants) - 10} more restaurants")

        return "\n".join(summary_lines)

    def _should_use_hybrid_mode(self, database_restaurants: List[Dict], reasoning: str) -> bool:
        """Determine if we should use hybrid mode (preserve DB + add web search)"""
        # Only consider hybrid if we have some database results
        if len(database_restaurants) == 0:
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

        # Default: if we have 1-3 restaurants and no explicit discard signal, use hybrid
        if 1 <= len(database_restaurants) <= 3:
            logger.info(f"🔄 Default hybrid mode: {len(database_restaurants)} restaurants found")
            return True

        # If we have 4+ restaurants but AI still says web search, it's probably quality issues
        # Default to hybrid to preserve any good ones
        logger.info(f"🔄 Fallback hybrid mode: preserving {len(database_restaurants)} restaurants")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "agent_type": "content_evaluation",
            "has_brave_search_agent": self.brave_search_agent is not None,
            "model": self.config.OPENAI_MODEL
        }