# agents/content_evaluation_agent.py
"""
Content Evaluation Agent

Evaluates database results against raw query to determine if additional web search is needed.
Sits between DatabaseSearchAgent and EditorAgent in the pipeline.

Key responsibilities:
1. Evaluate database result quality vs raw query relevance
2. Assess quantity adequacy for query type
3. Trigger supplemental web search if needed
4. Coordinate content fusion from multiple sources
5. Pass optimized content to editor
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
    Intelligent agent that evaluates content quality and triggers supplemental searches
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for evaluation
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,  # Low temperature for consistent evaluation
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_evaluation', 1024)
        )

        # Quality evaluation prompt
        self.quality_evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_quality_evaluation_system_prompt()),
            ("human", self._get_quality_evaluation_human_prompt())
        ])

        # Quantity adequacy prompt
        self.quantity_evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_quantity_evaluation_system_prompt()),
            ("human", self._get_quantity_evaluation_human_prompt())
        ])

        # Create evaluation chains
        self.quality_chain = self.quality_evaluation_prompt | self.llm
        self.quantity_chain = self.quantity_evaluation_prompt | self.llm

        # Import agents for supplemental search (lazy loading to avoid circular imports)
        self._search_agent = None
        self._scraper = None

        logger.info("✅ ContentEvaluationAgent initialized")

    def _get_search_agent(self):
        """Lazy load search agent to avoid circular imports"""
        if self._search_agent is None:
            from agents.search_agent import BraveSearchAgent
            self._search_agent = BraveSearchAgent(self.config)
        return self._search_agent

    def _get_scraper(self):
        """Lazy load scraper to avoid circular imports"""
        if self._scraper is None:
            from agents.optimized_scraper import WebScraper
            self._scraper = WebScraper(self.config)
        return self._scraper

    def _get_quality_evaluation_system_prompt(self) -> str:
        """System prompt for evaluating database result quality"""
        return """You are a restaurant recommendation quality evaluator.

Your job is to assess whether database restaurants match a user's raw query well enough to provide a good recommendation, or if additional web search is needed.

EVALUATION CRITERIA:

1. **Relevance Match**:
   - Do the restaurants match the cuisine type requested?
   - Do they match the dining style (casual, fine dining, etc.)?
   - Do they match special requirements (vegetarian, gluten-free, affordable, etc.)?

2. **Query Intent Alignment**:
   - Does the result type match what the user is asking for?
   - "Best restaurants" vs "cheap eats" vs "romantic dinner" have different expectations

3. **Number of restaurants**:
    - Does the number of restaurants match the query type? Try to understand from the request if the user will be happy with the number of options or if they need more.

3. **Location Precision**:
   - Are restaurants in the right city/area?
   - For neighborhood queries, are they in the right district?

4. **Completeness**:
   - Are restaurant details sufficient (name, description, cuisine type)?
   - Is there enough information to make informed recommendations?

SCORING:
- 0.9-1.0: Excellent match, no web search needed
- 0.7-0.8: Good match, might benefit from web search
- 0.5-0.6: Partial match, web search recommended  
- 0.0-0.4: Poor match, web search required

OUTPUT: JSON with relevance_score (0.0-1.0), reasoning, and web_search_recommended (boolean)."""

    def _get_quality_evaluation_human_prompt(self) -> str:
        """Human prompt template for quality evaluation"""
        return """USER QUERY: "{{raw_query}}"
DESTINATION: {{destination}}

DATABASE RESTAURANTS:
{{restaurants_summary}}

Evaluate how well these database restaurants match the user's query and intent."""

    def _get_quantity_evaluation_system_prompt(self) -> str:
        """System prompt for evaluating result quantity"""
        return """You are a restaurant recommendation quantity evaluator.

Your job is to determine if the number of restaurants is adequate for the type of query, or if more results are needed from web search.

QUANTITY EXPECTATIONS BY QUERY TYPE:

**High Volume Expected (8-15+ restaurants):**
- "best restaurants in [city]"
- "where to eat in [city]" 
- "restaurants in [city]"
- General dining queries without specific constraints

**Medium Volume Expected (4-8 restaurants):**
- Specific cuisine queries: "Italian restaurants in [city]"
- Price point queries: "cheap eats in [city]"
- Meal-specific: "breakfast places in [city]"

**Low Volume Acceptable (2-4 restaurants):**
- Very specific queries: "Michelin star restaurants"
- Niche requirements: "vegan fine dining"
- Special occasions: "romantic restaurants"

**Single Result Acceptable (1-2 restaurants):**
- "best [specific dish] in [city]"
- Very niche combinations: "Georgian wine bar"

EVALUATION FACTORS:
- Query specificity vs result count
- City size (major cities should have more options)
- Cuisine popularity in the region

OUTPUT: JSON with quantity_adequate (boolean), expected_range, current_count, and reasoning."""

    def _get_quantity_evaluation_human_prompt(self) -> str:
        """Human prompt template for quantity evaluation"""
        return """USER QUERY: "{{raw_query}}"
DESTINATION: {{destination}}
CURRENT RESTAURANT COUNT: {{restaurant_count}}

RESTAURANTS:
{{restaurant_names}}

Is this quantity adequate for this type of query, or should we search for more options?"""

    @log_function_call
    def evaluate_and_enhance(self, 
                           database_restaurants: List[Dict[str, Any]], 
                           raw_query: str,
                           destination: str = "Unknown",
                           **context) -> Dict[str, Any]:
        """
        Main method: evaluate database content and enhance with web search if needed

        Args:
            database_restaurants: Results from database search
            raw_query: User's original query
            destination: Location being searched
            **context: Additional context from pipeline

        Returns:
            Dict with optimized content for editor and evaluation details
        """
        try:
            logger.info(f"🧠 Evaluating content quality for: '{raw_query}' in {destination}")
            logger.info(f"📊 Database restaurants available: {len(database_restaurants)}")

            # Step 1: Evaluate quality if we have database results
            quality_evaluation = None
            if database_restaurants:
                quality_evaluation = self._evaluate_quality(database_restaurants, raw_query, destination)
                logger.info(f"🎯 Quality score: {quality_evaluation.get('relevance_score', 0):.2f}")

            # Step 2: Evaluate quantity adequacy
            quantity_evaluation = self._evaluate_quantity(database_restaurants, raw_query, destination)
            logger.info(f"📊 Quantity adequate: {quantity_evaluation.get('quantity_adequate', False)}")

            # Step 3: Decide if supplemental web search is needed
            needs_web_search = self._should_trigger_web_search(quality_evaluation, quantity_evaluation)

            if needs_web_search:
                logger.info("🌐 Triggering supplemental web search")
                supplemental_results = self._perform_supplemental_search(
                    raw_query, destination, database_restaurants, context
                )

                # Step 4: Combine results
                final_content = self._combine_content_sources(
                    database_restaurants, supplemental_results, quality_evaluation
                )
                content_source = "hybrid"
            else:
                logger.info("✅ Database content sufficient, no web search needed")
                final_content = {
                    "database_restaurants": database_restaurants,
                    "scraped_results": []
                }
                content_source = "database"

            # Step 5: Prepare response for editor
            return {
                "optimized_content": final_content,
                "content_source": content_source,
                "quality_evaluation": quality_evaluation,
                "quantity_evaluation": quantity_evaluation,
                "web_search_triggered": needs_web_search,
                "evaluation_summary": self._create_evaluation_summary(
                    quality_evaluation, quantity_evaluation, needs_web_search
                )
            }

        except Exception as e:
            logger.error(f"❌ Error in content evaluation: {e}")
            dump_chain_state("content_evaluation_error", {
                "error": str(e),
                "raw_query": raw_query,
                "database_count": len(database_restaurants) if database_restaurants else 0
            })

            # Fallback: pass through database content
            return self._create_fallback_response(database_restaurants, str(e))

    def _evaluate_quality(self, restaurants: List[Dict], raw_query: str, destination: str) -> Dict[str, Any]:
        """Evaluate how well database restaurants match the raw query"""
        try:
            # Prepare restaurant summary for AI evaluation
            restaurants_summary = self._create_restaurants_summary(restaurants)

            # Format the evaluation prompt
            formatted_prompt = self.quality_evaluation_prompt.format_messages(
                raw_query=raw_query,
                destination=destination,
                restaurants_summary=restaurants_summary
            )

            # Get AI evaluation
            response = self.quality_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "restaurants_summary": restaurants_summary
            })

            # Parse response
            evaluation = self._parse_evaluation_response(response.content, "quality")

            logger.info(f"🎯 Quality evaluation: {evaluation.get('relevance_score', 0):.2f} - {evaluation.get('reasoning', 'No reasoning')}")

            return evaluation

        except Exception as e:
            logger.error(f"❌ Error in quality evaluation: {e}")
            return {
                "relevance_score": 0.5,  # Conservative fallback
                "reasoning": f"evaluation_error: {str(e)}",
                "web_search_recommended": True
            }

    def _evaluate_quantity(self, restaurants: List[Dict], raw_query: str, destination: str) -> Dict[str, Any]:
        """Evaluate if the quantity of restaurants is adequate for the query type"""
        try:
            restaurant_count = len(restaurants)
            restaurant_names = [r.get('name', 'Unknown') for r in restaurants[:10]]  # First 10 names

            # Get AI evaluation
            response = self.quantity_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "restaurant_count": restaurant_count,
                "restaurant_names": ", ".join(restaurant_names)
            })

            # Parse response
            evaluation = self._parse_evaluation_response(response.content, "quantity")

            logger.info(f"📊 Quantity evaluation: {evaluation.get('quantity_adequate', False)} - {evaluation.get('reasoning', 'No reasoning')}")

            return evaluation

        except Exception as e:
            logger.error(f"❌ Error in quantity evaluation: {e}")
            return {
                "quantity_adequate": restaurant_count >= 3,  # Simple fallback
                "expected_range": "3-8",
                "current_count": restaurant_count,
                "reasoning": f"evaluation_error: {str(e)}"
            }

    def _should_trigger_web_search(self, quality_eval: Optional[Dict], quantity_eval: Dict) -> bool:
        """Determine if supplemental web search should be triggered"""
        try:
            # Quality-based decision
            if quality_eval:
                relevance_score = quality_eval.get('relevance_score', 0)
                web_search_recommended = quality_eval.get('web_search_recommended', False)

                if relevance_score < 0.6 or web_search_recommended:
                    logger.info(f"🌐 Web search triggered by quality: score={relevance_score:.2f}")
                    return True

            # Quantity-based decision
            quantity_adequate = quantity_eval.get('quantity_adequate', False)
            if not quantity_adequate:
                logger.info("🌐 Web search triggered by insufficient quantity")
                return True

            logger.info("✅ Database content sufficient")
            return False

        except Exception as e:
            logger.error(f"❌ Error determining web search need: {e}")
            return True  # Conservative: search when in doubt

    def _perform_supplemental_search(self, 
                                   raw_query: str, 
                                   destination: str, 
                                   database_restaurants: List[Dict],
                                   context: Dict) -> List[Dict]:
        """Perform supplemental web search to enhance database results"""
        try:
            logger.info("🔍 Starting supplemental web search")

            # Generate search queries based on gaps in database content
            search_queries = self._generate_supplemental_search_queries(
                raw_query, destination, database_restaurants
            )

            logger.info(f"🔍 Generated {len(search_queries)} supplemental search queries")
            for i, query in enumerate(search_queries, 1):
                logger.info(f"  {i}. {query}")

            # Perform web search
            search_agent = self._get_search_agent()
            search_results = search_agent.search(search_queries, enable_ai_filtering=True)

            logger.info(f"🔍 Found {len(search_results)} supplemental search results")

            # Scrape the results
            if search_results:
                scraper = self._get_scraper()
                import asyncio
                import concurrent.futures

                def run_scraping():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(scraper.scrape_search_results(search_results))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    scraped_results = pool.submit(run_scraping).result()

                logger.info(f"🕷️ Scraped {len(scraped_results)} supplemental articles")
                return scraped_results
            else:
                logger.warning("⚠️ No supplemental search results found")
                return []

        except Exception as e:
            logger.error(f"❌ Error in supplemental search: {e}")
            return []

    def _generate_supplemental_search_queries(self, 
                                            raw_query: str, 
                                            destination: str, 
                                            database_restaurants: List[Dict]) -> List[str]:
        """Generate targeted search queries to fill gaps in database content"""
        queries = []

        # Base query variations
        base_query = f"{raw_query} {destination}".strip()
        queries.append(base_query)

        # Add source-specific queries to get professional reviews
        professional_sources = [
            f"{base_query} Michelin guide",
            f"{base_query} food critic review",
            f"{base_query} best restaurants guide",
            f"where to eat {destination} local guide"
        ]
        queries.extend(professional_sources)

        # Limit to reasonable number of queries
        return queries[:4]

    def _combine_content_sources(self, 
                               database_restaurants: List[Dict], 
                               scraped_results: List[Dict],
                               quality_evaluation: Optional[Dict]) -> Dict[str, Any]:
        """Intelligently combine database and web search results"""
        try:
            logger.info(f"🔗 Combining {len(database_restaurants)} database + {len(scraped_results)} web results")

            # Strategy: Use database as foundation, supplement with web content
            combined_content = {
                "database_restaurants": database_restaurants,
                "scraped_results": scraped_results,
                "combination_strategy": "database_foundation_web_supplement",
                "primary_source": "database" if database_restaurants else "web"
            }

            # Add quality context for editor
            if quality_evaluation:
                combined_content["quality_context"] = {
                    "database_quality_score": quality_evaluation.get('relevance_score', 0),
                    "recommendation": "prioritize_web" if quality_evaluation.get('relevance_score', 0) < 0.6 else "prioritize_database"
                }

            logger.info(f"🔗 Content combination complete: strategy={combined_content['combination_strategy']}")
            return combined_content

        except Exception as e:
            logger.error(f"❌ Error combining content sources: {e}")
            return {
                "database_restaurants": database_restaurants,
                "scraped_results": scraped_results,
                "combination_error": str(e)
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

    def _parse_evaluation_response(self, response_content: str, evaluation_type: str) -> Dict[str, Any]:
        """Parse AI evaluation response into structured data"""
        try:
            # Clean up response
            content = response_content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            evaluation = json.loads(content)
            return evaluation

        except Exception as e:
            logger.error(f"❌ Error parsing {evaluation_type} evaluation response: {e}")
            logger.error(f"Raw response: {response_content}")

            # Return fallback based on evaluation type
            if evaluation_type == "quality":
                return {
                    "relevance_score": 0.5,
                    "reasoning": "parsing_error",
                    "web_search_recommended": True
                }
            else:  # quantity
                return {
                    "quantity_adequate": False,
                    "reasoning": "parsing_error",
                    "expected_range": "unknown"
                }

    def _create_evaluation_summary(self, 
                                 quality_eval: Optional[Dict], 
                                 quantity_eval: Dict, 
                                 web_search_triggered: bool) -> Dict[str, Any]:
        """Create summary of evaluation results for logging and debugging"""
        return {
            "quality_score": quality_eval.get('relevance_score', 0) if quality_eval else None,
            "quantity_adequate": quantity_eval.get('quantity_adequate', False),
            "web_search_triggered": web_search_triggered,
            "decision_reasoning": {
                "quality": quality_eval.get('reasoning', 'No quality evaluation') if quality_eval else "No database content",
                "quantity": quantity_eval.get('reasoning', 'No reasoning'),
                "final_decision": "web_search" if web_search_triggered else "database_sufficient"
            }
        }

    def _create_fallback_response(self, database_restaurants: List[Dict], error: str) -> Dict[str, Any]:
        """Create fallback response when evaluation fails"""
        return {
            "optimized_content": {
                "database_restaurants": database_restaurants,
                "scraped_results": []
            },
            "content_source": "database",
            "evaluation_error": error,
            "web_search_triggered": False,
            "evaluation_summary": {
                "status": "error",
                "fallback": "using_database_content"
            }
        }