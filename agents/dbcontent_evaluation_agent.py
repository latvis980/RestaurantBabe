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
   - 3+ restaurants = usually sufficient 
   - 1-2 restaurants = may need web search for more options
   - 0 restaurants = definitely need web search

3. **Quality**: Are the restaurant details sufficient?
   - Name, location, cuisine type should be present
   - Descriptions should be meaningful

DECISION LOGIC:
- If database results are good matches with sufficient quantity → USE DATABASE
- If results don't match query well → TRIGGER WEB SEARCH  
- If too few results (less than 3) → TRIGGER WEB SEARCH
- If no results → TRIGGER WEB SEARCH

OUTPUT: JSON with decision and reasoning"""

    def _get_evaluation_human_prompt(self) -> str:
        """Human prompt template for evaluation"""
        return """USER QUERY: "{{raw_query}}"
DESTINATION: {{destination}}

DATABASE RESTAURANTS ({{restaurant_count}} found):
{{restaurants_summary}}

Evaluate if these database results are sufficient for the user's query.

Return JSON:
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
            database_restaurants = pipeline_data.get("database_results", [])
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
            "trigger_web_search": False,
            "skip_web_search": True,  # Control main search step
            "final_database_content": database_restaurants,
            "optimized_content": {
                "database_restaurants": database_restaurants,
                "scraped_results": []
            }
        }

    def _trigger_web_search_workflow(self, pipeline_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """Handle the web search route - trigger BraveSearchAgent and return results"""
        logger.info("🌐 Triggering web search workflow")

        try:
            # Option 1: Trigger BraveSearchAgent immediately (if you want immediate search)
            if self.brave_search_agent:
                logger.info("🚀 Triggering BraveSearchAgent immediately")
                # You could trigger the search here if you want immediate results
                # search_results = self.brave_search_agent.search(pipeline_data)
                # But based on your current architecture, you probably want to just set routing flags

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
                "trigger_web_search": True,
                "skip_web_search": False,  # Allow main search step
                "final_database_content": [],
                "optimized_content": {
                    "database_restaurants": [],
                    "scraped_results": []
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in web search workflow: {e}")
            return self._handle_evaluation_error(pipeline_data, e)

    def _handle_evaluation_error(self, pipeline_data: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """Centralized error handling with consistent fallback logic"""
        logger.error(f"❌ Evaluation error: {error}")

        dump_chain_state("content_evaluation_error", {
            "error": str(error),
            "raw_query": pipeline_data.get("raw_query", "unknown"),
            "database_count": len(pipeline_data.get("database_results", []))
        })

        # Fallback: trigger web search when evaluation fails
        return {
            **pipeline_data,
            "evaluation_result": {
                "database_sufficient": False,
                "trigger_web_search": True,
                "content_source": "web_search",
                "reasoning": f"Evaluation error: {str(error)}",
                "evaluation_summary": {"reason": "evaluation_error"}
            },
            "evaluation_error": str(error),
            "content_source": "web_search",
            "trigger_web_search": True,
            "skip_web_search": False,
            "final_database_content": []
        }

    def _evaluate_with_ai(self, restaurants: List[Dict], raw_query: str, destination: str) -> Dict[str, Any]:
        """Use AI to evaluate database content quality"""
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

            # Parse response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content)
            return evaluation

        except Exception as e:
            logger.error(f"❌ Error in AI evaluation: {e}")
            # Fallback decision
            return {
                "database_sufficient": len(restaurants) >= 3,  # Simple fallback logic
                "trigger_web_search": len(restaurants) < 3,
                "reasoning": f"AI evaluation failed, using fallback logic: {len(restaurants)} restaurants",
                "quality_score": 0.5
            }

    def _create_restaurants_summary(self, restaurants: List[Dict]) -> str:
        """Create a concise summary of restaurants for AI evaluation"""
        if not restaurants:
            return "No restaurants available"

        summary_lines = []
        for i, restaurant in enumerate(restaurants[:10], 1):  # Limit to first 10
            name = restaurant.get('name', 'Unknown')
            cuisine = restaurant.get('cuisine_type', 'Unknown cuisine')
            rating = restaurant.get('google_rating', 'No rating')
            price = restaurant.get('price_level', 'Unknown price')

            summary_lines.append(f"{i}. {name} - {cuisine}, Rating: {rating}, Price: {price}")

        if len(restaurants) > 10:
            summary_lines.append(f"... and {len(restaurants) - 10} more restaurants")

        return "\n".join(summary_lines)

    # Keep the original method for backward compatibility if needed
    def evaluate_database_content(self, 
                                database_restaurants: List[Dict[str, Any]], 
                                raw_query: str,
                                destination: str = "Unknown") -> Dict[str, Any]:
        """
        LEGACY METHOD: For backward compatibility.
        Use evaluate_and_route() for new implementations.
        """
        logger.warning("⚠️ Using legacy evaluate_database_content method. Consider using evaluate_and_route() instead.")

        # Convert to new format and call main method
        pipeline_data = {
            "database_results": database_restaurants,
            "raw_query": raw_query,
            "destination": destination
        }

        result = self.evaluate_and_route(pipeline_data)
        return result.get("evaluation_result", {})