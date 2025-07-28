# agents/database_search_agent.py
"""
Unified Database Search Agent - Combines database queries with AI-powered semantic search.

This agent handles:
1. Semantic search using DeepSeek
2. AI evaluation of restaurant relevance
3. Intelligent decision making about web search necessity
4. All previous database search functionality
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class DatabaseSearchAgent:
    """
    Unified agent that handles database restaurant searches with semantic AI capabilities.
    Decides whether database content is sufficient or if web search is needed.
    """

    def __init__(self, config):
        self.config = config

        # Basic settings
        self.minimum_restaurant_threshold = getattr(config, 'MIN_DATABASE_RESTAURANTS', 3)
        self.ai_evaluation_enabled = getattr(config, 'DATABASE_AI_EVALUATION', True)  # Enable by default

        # Initialize Deepseek components for semantic search
        self.llm = ChatDeepSeek(
            model_name=config.DEEPSEEK_MODEL,
            temperature=0.1
        )

        # Setup AI prompts
        self._setup_prompts()

    
        logger.info(f"DatabaseSearchAgent initialized with semantic search capabilities")

    def _setup_prompts(self):
        """Setup AI prompts for different analysis tasks"""

        # Query intent analysis
        self.intent_analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert at understanding restaurant search queries. Analyze the user's intent and extract key information.

USER QUERY: "{query}"
DESTINATION: "{destination}"

Analyze this query and extract:

1. **search_intent**: What is the user primarily looking for? (cuisine_specific, atmosphere_specific, meal_specific, bar_drinks, general_dining, special_occasion)

2. **specificity_level**: How specific is the request? (very_specific, moderately_specific, general)

3. **key_concepts**: List the 3-5 most important concepts the user cares about

4. **search_context**: Brief description of what would make a restaurant relevant

IMPORTANT: Focus on the user's actual intent, not just keywords. Consider synonyms, related concepts, and implicit meanings.

Return ONLY valid JSON:
{{
    "search_intent": "...",
    "specificity_level": "...",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "search_context": "..."
}}
""")

        # Restaurant relevance evaluation
        self.relevance_analysis_prompt = ChatPromptTemplate.from_template("""
USER QUERY INTENT: {search_context}
KEY CONCEPTS: {key_concepts}

RESTAURANT TO EVALUATE:
Name: {restaurant_name}
Cuisine Tags: {cuisine_tags}
Description: {description}

Analyze how well this restaurant matches the user's search intent.

Consider:
- Direct cuisine/style matches
- Semantic similarity (e.g., "middle eastern" relates to "israeli")
- Implied characteristics from description
- Atmosphere and dining style
- Menu specialties and focus

Rate the relevance on a scale of 0-10:
- 0-2: Not relevant at all
- 3-4: Slightly relevant 
- 5-6: Moderately relevant
- 7-8: Highly relevant
- 9-10: Perfect match

Return ONLY valid JSON:
{{
    "relevance_score": 0-10,
    "reasoning": "Brief explanation of why this score was given",
    "matching_aspects": ["list", "of", "matching", "elements"]
}}
""")

        # Results quality evaluation
        self.quality_evaluation_prompt = ChatPromptTemplate.from_template("""
User Query: {raw_query}
Location: {destination}

Found {restaurant_count} restaurants:
{restaurants_summary}

Evaluate if these results sufficiently answer the user's query.
Consider:
- Do the restaurants match the query intent?
- Is the variety appropriate?
- Are descriptions detailed enough?
- Would web search add significant value?

Return JSON:
{{
    "sufficient": true/false,
    "confidence": 0-10,
    "reasoning": "explanation",
    "missing_aspects": ["what's missing if insufficient"]
}}
""")

    @log_function_call
    def search_and_evaluate(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method: Search database with semantic matching and decide if results are sufficient.

        Args:
            query_analysis: Output from QueryAnalyzer containing destination, raw_query, etc.

        Returns:
            Dict containing:
            - has_database_content: bool
            - database_results: List[Dict] (if has_database_content=True)
            - content_source: str ("database" or "web_search")
            - evaluation_details: Dict (for debugging/monitoring)
        """
        try:
            logger.info("🗃️ STARTING SEMANTIC DATABASE SEARCH AND EVALUATION")

            # Extract data from query analysis
            destination = query_analysis.get("destination", "Unknown")
            raw_query = query_analysis.get("raw_query", query_analysis.get("query", ""))

            if destination == "Unknown":
                logger.info("⚠️ No destination detected, will use web search")
                return self._create_web_search_response("no_destination")

            # Perform semantic search
            relevant_restaurants, should_scrape = self._search_database_intelligently(
                query=raw_query,
                destination=destination,
                min_results=self.minimum_restaurant_threshold,
                max_results=8
            )

            if not relevant_restaurants:
                logger.info("📭 No relevant restaurants found in database")
                return self._create_web_search_response("no_relevant_results")

            # Evaluate quality of results
            quality_evaluation = self._evaluate_results_quality(
                restaurants=relevant_restaurants,
                raw_query=raw_query,
                destination=destination
            )

            # Make final decision
            if quality_evaluation["sufficient"] and not should_scrape:
                logger.info(f"✅ DATABASE SUFFICIENT: {len(relevant_restaurants)} relevant restaurants found")
                return self._create_database_response(relevant_restaurants, quality_evaluation)
            else:
                logger.info(f"🌐 WEB SEARCH NEEDED: {quality_evaluation.get('reasoning', 'Insufficient results')}")
                return self._create_web_search_response(quality_evaluation.get("reasoning", "insufficient_quality"))

        except Exception as e:
            logger.error(f"❌ Error in semantic database search: {e}")
            dump_chain_state("database_search_error", {
                "error": str(e),
                "destination": query_analysis.get("destination", "Unknown")
            })
            return self._create_web_search_response("error")

    def _search_database_intelligently(
        self, 
        query: str, 
        destination: str, 
        min_results: int = 2,
        max_results: int = 8
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        AI-powered semantic search for relevant restaurants
        """
        try:
            logger.info(f"🤖 AI semantic search: '{query}' in {destination}")

            # Extract city from destination
            city = self._extract_city(destination)

            # Get all restaurants for the city
            all_restaurants = self._get_all_city_restaurants(city)

            if not all_restaurants:
                logger.info(f"📭 No restaurants in database for {city}")
                return [], True

            # Analyze the query intent
            query_analysis = self._analyze_query_intent(query, destination)
            logger.info(f"🧠 Query analysis: {query_analysis.get('search_intent')} - {query_analysis.get('specificity_level')}")

            # Find relevant restaurants using AI analysis
            relevant_restaurants = self._find_semantically_relevant_restaurants(
                all_restaurants, query_analysis, max_results * 2  # Get more for better filtering
            )

            # Apply final filtering and ranking
            final_restaurants = self._final_ranking_and_filtering(
                relevant_restaurants, query_analysis, max_results
            )

            # Determine if web scraping is needed
            should_scrape = len(final_restaurants) < min_results

            logger.info(
                f"📊 Semantic search results: {len(final_restaurants)}/{len(all_restaurants)} restaurants relevant, "
                f"scraping needed: {should_scrape}"
            )

            return final_restaurants, should_scrape

        except Exception as e:
            logger.error(f"❌ Error in semantic database search: {e}")
            return [], True

    def _extract_city(self, destination: str) -> str:
        """Extract city name from destination string"""
        if "," in destination:
            return destination.split(",")[0].strip()
        return destination.strip()

    def _get_all_city_restaurants(self, city: str) -> List[Dict[str, Any]]:
        """Get all restaurants for a city from database"""
        try:
            # Import database utility and search
            from utils.database import get_database
            db = get_database()

            # Query with generous limit to get full picture
            database_restaurants = db.get_restaurants_by_city(city, limit=100)

            logger.info(f"📊 Database query returned {len(database_restaurants) if database_restaurants else 0} restaurants for {city}")

            return database_restaurants or []

        except Exception as e:
            logger.error(f"❌ Error searching database for {city}: {e}")
            return []

    def _analyze_query_intent(self, query: str, destination: str) -> Dict[str, Any]:
        """Use AI to analyze query intent and extract key concepts"""
        try:
            # Create the chain by combining prompt and LLM
            chain = self.intent_analysis_prompt | self.llm

            response = chain.invoke({
                "query": query,
                "destination": destination
            })

            # Get the content from the response
            content = response.content.strip()

            # Parse the AI response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            if not content:
                raise ValueError("Empty response from AI")

            analysis = json.loads(content)

            # Validate required fields
            required_fields = ["search_intent", "specificity_level", "key_concepts", "search_context"]
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            return analysis

        except Exception as e:
            logger.error(f"❌ Error analyzing query intent: {e}")
            # Fallback analysis
            return self._create_fallback_analysis(query)

    def _create_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Create a fallback analysis when AI fails"""
        query_lower = query.lower()

        # Basic cuisine detection
        cuisine_keywords = {
            "persian": ["persian", "iranian"],
            "italian": ["italian", "pasta", "pizza"],
            "japanese": ["japanese", "sushi", "ramen"],
            "chinese": ["chinese"],
            "indian": ["indian", "curry"],
            "thai": ["thai"],
            "mexican": ["mexican", "tacos"],
            "french": ["french"],
            "israeli": ["israeli"],
            "turkish": ["turkish"],
            "korean": ["korean"],
            "lebanese": ["lebanese"],
            "vietnamese": ["vietnamese"]
        }

        detected_cuisines = []
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_cuisines.append(cuisine)

        # Determine intent based on detected elements
        if detected_cuisines:
            search_intent = "cuisine_specific"
            specificity_level = "very_specific"
            key_concepts = detected_cuisines + ["restaurants", "dining"]
            search_context = f"Restaurants serving {' or '.join(detected_cuisines)} cuisine"
        elif "bar" in query_lower or "cocktail" in query_lower:
            search_intent = "bar_drinks"
            specificity_level = "very_specific"
            key_concepts = ["cocktails", "bar", "drinks"]
            search_context = "Bars specializing in cocktails and mixed drinks"
        elif "brunch" in query_lower:
            search_intent = "meal_specific"
            specificity_level = "very_specific"
            key_concepts = ["brunch", "breakfast", "coffee"]
            search_context = "Restaurants serving brunch and breakfast"
        else:
            search_intent = "general_dining"
            specificity_level = "general"
            key_concepts = ["restaurants", "dining"]
            search_context = f"General restaurants related to: {query}"

        return {
            "search_intent": search_intent,
            "specificity_level": specificity_level,
            "key_concepts": key_concepts,
            "search_context": search_context
        }

    def _find_semantically_relevant_restaurants(
        self, 
        restaurants: List[Dict[str, Any]], 
        query_analysis: Dict[str, Any],
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Use AI to evaluate each restaurant's relevance to the query"""

        relevant_restaurants = []
        search_context = query_analysis.get("search_context", "")
        key_concepts = query_analysis.get("key_concepts", [])

        logger.info(f"🔍 Evaluating {len(restaurants)} restaurants for relevance...")

        # Process restaurants in batches to avoid hitting rate limits
        batch_size = 5
        for i in range(0, len(restaurants), batch_size):
            batch = restaurants[i:i + batch_size]

            for restaurant in batch:
                try:
                    # Evaluate restaurant relevance
                    relevance_data = self._evaluate_restaurant_relevance(
                        restaurant, search_context, key_concepts
                    )

                    relevance_score = relevance_data.get("relevance_score", 0)

                    # Only include restaurants with meaningful relevance
                    if relevance_score >= 5:  # Moderate relevance threshold
                        restaurant_copy = restaurant.copy()
                        restaurant_copy['_ai_relevance_score'] = relevance_score
                        restaurant_copy['_ai_reasoning'] = relevance_data.get("reasoning", "")
                        restaurant_copy['_matching_aspects'] = relevance_data.get("matching_aspects", [])
                        relevant_restaurants.append(restaurant_copy)

                except Exception as e:
                    logger.warning(f"Error evaluating restaurant {restaurant.get('name', 'unknown')}: {e}")
                    continue

            # Early exit if we have enough highly relevant results
            if len([r for r in relevant_restaurants if r['_ai_relevance_score'] >= 8]) >= max_candidates:
                break

        logger.info(f"✅ Found {len(relevant_restaurants)} semantically relevant restaurants")
        return relevant_restaurants

    def _evaluate_restaurant_relevance(
        self, 
        restaurant: Dict[str, Any], 
        search_context: str, 
        key_concepts: List[str]
    ) -> Dict[str, Any]:
        """Use AI to evaluate how relevant a restaurant is to the search query"""

        try:
            # Prepare restaurant data for analysis
            cuisine_tags = restaurant.get('cuisine_tags', [])
            description = restaurant.get('raw_description', '')[:500]  # Truncate for efficiency
            name = restaurant.get('name', 'Unknown')

            # Create the chain by combining prompt and LLM
            chain = self.relevance_analysis_prompt | self.llm

            response = chain.invoke({
                "search_context": search_context,
                "key_concepts": ", ".join(key_concepts),
                "restaurant_name": name,
                "cuisine_tags": ", ".join(cuisine_tags),
                "description": description
            })

            # Get content from response
            content = response.content.strip()

            # Parse response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            if not content:
                raise ValueError("Empty response from AI")

            relevance_data = json.loads(content)

            # Validate score
            score = relevance_data.get("relevance_score", 0)
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                score = 0

            relevance_data["relevance_score"] = score
            return relevance_data

        except Exception as e:
            logger.error(f"Error in AI relevance evaluation: {e}")
            # Fallback to simple keyword matching
            return self._fallback_relevance_evaluation(restaurant, key_concepts)

    def _fallback_relevance_evaluation(
        self, 
        restaurant: Dict[str, Any], 
        key_concepts: List[str]
    ) -> Dict[str, Any]:
        """Fallback relevance evaluation using keyword matching"""

        cuisine_tags = [tag.lower() for tag in restaurant.get('cuisine_tags', [])]
        description = restaurant.get('raw_description', '').lower()
        name = restaurant.get('name', '').lower()

        score = 0
        matching_aspects = []

        # Check each key concept
        for concept in key_concepts:
            concept_lower = concept.lower()

            # Direct tag match (highest score)
            if concept_lower in cuisine_tags:
                score += 3
                matching_aspects.append(f"cuisine tag: {concept}")
            # Name match
            elif concept_lower in name:
                score += 2
                matching_aspects.append(f"name contains: {concept}")
            # Description match
            elif concept_lower in description:
                score += 1
                matching_aspects.append(f"description mentions: {concept}")

        # Normalize score to 0-10 scale
        max_possible_score = len(key_concepts) * 3
        if max_possible_score > 0:
            normalized_score = min(10, (score / max_possible_score) * 10)
        else:
            normalized_score = 0

        return {
            "relevance_score": int(normalized_score),
            "reasoning": f"Keyword matching found {len(matching_aspects)} matches",
            "matching_aspects": matching_aspects
        }

    def _final_ranking_and_filtering(
        self, 
        restaurants: List[Dict[str, Any]], 
        query_analysis: Dict[str, Any],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Apply final ranking combining AI relevance with other factors"""

        if not restaurants:
            return []

        # Sort by AI relevance score first, then by mention count
        restaurants.sort(
            key=lambda r: (
                r.get('_ai_relevance_score', 0),
                r.get('mention_count', 1)
            ),
            reverse=True
        )

        # For very specific queries, be more strict
        specificity = query_analysis.get("specificity_level", "general")
        if specificity == "very_specific":
            # Only keep restaurants with high relevance scores
            restaurants = [r for r in restaurants if r.get('_ai_relevance_score', 0) >= 7]
        elif specificity == "moderately_specific":
            # Keep moderately relevant and above
            restaurants = [r for r in restaurants if r.get('_ai_relevance_score', 0) >= 6]

        # Remove AI scoring fields before returning
        final_restaurants = []
        for restaurant in restaurants[:max_results]:
            cleaned = restaurant.copy()

            # Log the AI reasoning for debugging
            ai_score = cleaned.pop('_ai_relevance_score', 0)
            ai_reasoning = cleaned.pop('_ai_reasoning', '')
            matching_aspects = cleaned.pop('_matching_aspects', [])

            logger.info(
                f"🎯 Selected: {cleaned.get('name')} (score: {ai_score}) - {ai_reasoning}"
            )

            final_restaurants.append(cleaned)

        return final_restaurants

    def _evaluate_results_quality(
        self,
        restaurants: List[Dict[str, Any]],
        raw_query: str,
        destination: str
    ) -> Dict[str, Any]:
        """Use AI to assess if results are good enough to skip web search"""

        try:
            # Format restaurants summary
            restaurants_summary = self._format_restaurants_for_evaluation(restaurants[:10])  # Limit to 10 for evaluation

            # Create the chain
            chain = self.quality_evaluation_prompt | self.llm

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

            # Validate and enhance evaluation
            evaluation["restaurant_count"] = len(restaurants)
            evaluation["evaluation_method"] = "ai_quality_assessment"

            return evaluation

        except Exception as e:
            logger.error(f"Error in quality evaluation: {e}")
            # Fallback to simple count-based evaluation
            return {
                "sufficient": len(restaurants) >= self.minimum_restaurant_threshold,
                "confidence": 5,
                "reasoning": f"Fallback evaluation: {len(restaurants)} restaurants found",
                "missing_aspects": [],
                "evaluation_method": "fallback_count"
            }

    def _format_restaurants_for_evaluation(self, restaurants: List[Dict[str, Any]]) -> str:
        """Format restaurant data for AI evaluation"""
        formatted = []
        for i, rest in enumerate(restaurants, 1):
            name = rest.get('name', 'Unknown')
            tags = ', '.join(rest.get('cuisine_tags', [])[:3])
            desc = rest.get('raw_description', '')[:100]
            formatted.append(f"{i}. {name} - {tags} - {desc}...")
        return "\n".join(formatted)


    def _create_database_response(
        self, 
        database_restaurants: List[Dict[str, Any]], 
        evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create response indicating database content should be used."""
        return {
            "has_database_content": True,
            "database_results": database_restaurants,
            "content_source": "database",
            "evaluation_details": evaluation_result,
            "skip_web_search": True  # Important flag for orchestrator
        }

    def _create_web_search_response(self, reason: str) -> Dict[str, Any]:
        """Create response indicating web search should be used."""
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

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about database search performance."""
        try:
            from utils.database import get_database
            db = get_database()
            stats = db.get_database_stats()

            # Add semantic search specific stats
            stats["semantic_search_enabled"] = self.ai_evaluation_enabled
        
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    def set_minimum_threshold(self, new_threshold: int):
        """Update the minimum restaurant threshold."""
        old_threshold = self.minimum_restaurant_threshold
        self.minimum_restaurant_threshold = new_threshold
        logger.info(f"Updated minimum restaurant threshold: {old_threshold} → {new_threshold}")

    def enable_ai_evaluation(self, enabled: bool = True):
        """Enable or disable AI evaluation."""
        self.ai_evaluation_enabled = enabled
        logger.info(f"AI evaluation {'enabled' if enabled else 'disabled'}")