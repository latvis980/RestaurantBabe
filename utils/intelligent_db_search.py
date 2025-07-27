# utils/intelligent_db_search.py
"""
AI-Powered Semantic Database Search Module

This module uses OpenAI embeddings and AI analysis for intelligent restaurant matching,
completely avoiding hardcoded keywords or rules.
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class SemanticDatabaseSearch:
    """
    Fully AI-powered database search using semantic similarity and intelligent analysis
    """

    def __init__(self, config, database):
        self.config = config
        self.database = database

        # Initialize OpenAI components
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1  # Low temperature for consistent analysis
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # Latest, efficient embedding model
        )

        # Setup prompts
        self._setup_prompts()

        # Cache for embeddings to avoid repeated API calls
        self._embedding_cache = {}

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
    "search_intent": "cuisine_specific|atmosphere_specific|meal_specific|bar_drinks|general_dining|special_occasion",
    "specificity_level": "very_specific|moderately_specific|general", 
    "key_concepts": ["concept1", "concept2", "concept3"],
    "search_context": "Brief description of what makes a restaurant relevant to this query"
}}

EXAMPLES:
"israeli food in berlin" → {{"search_intent": "cuisine_specific", "specificity_level": "very_specific", "key_concepts": ["israeli cuisine", "middle eastern food", "kosher options"], "search_context": "Restaurants serving Israeli or Middle Eastern cuisine"}}

"cocktail bars in berlin" → {{"search_intent": "bar_drinks", "specificity_level": "very_specific", "key_concepts": ["cocktails", "bar atmosphere", "drinks menu"], "search_context": "Bars specializing in cocktails and mixed drinks"}}

"romantic dinner spots" → {{"search_intent": "atmosphere_specific", "specificity_level": "moderately_specific", "key_concepts": ["romantic atmosphere", "intimate setting", "date night"], "search_context": "Restaurants with romantic, intimate atmosphere suitable for couples"}}
""")

        # Restaurant relevance analysis
        self.relevance_analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert at determining if restaurants match user search queries.

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

    def search_database_intelligently(
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
            all_restaurants = self.database.get_restaurants_by_city(city, limit=100)

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

    def _analyze_query_intent(self, query: str, destination: str) -> Dict[str, Any]:
        """Use AI to analyze query intent and extract key concepts"""
        try:
            response = self.llm.invoke({
                "query": query,
                "destination": destination
            })

            # Parse the AI response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

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
            return {
                "search_intent": "general_dining",
                "specificity_level": "general",
                "key_concepts": [query.lower()],
                "search_context": f"Restaurants related to: {query}"
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

            # Use the relevance analysis prompt
            chain = self.relevance_analysis_prompt | self.llm

            response = chain.invoke({
                "search_context": search_context,
                "key_concepts": ", ".join(key_concepts),
                "restaurant_name": name,
                "cuisine_tags": ", ".join(cuisine_tags),
                "description": description
            })

            # Parse response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            relevance_data = json.loads(content)

            # Validate score
            score = relevance_data.get("relevance_score", 0)
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                score = 0

            relevance_data["relevance_score"] = score
            return relevance_data

        except Exception as e:
            logger.error(f"Error in AI relevance evaluation: {e}")
            return {
                "relevance_score": 0,
                "reasoning": "Failed to analyze",
                "matching_aspects": []
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

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            embedding = self.embeddings.embed_query(text)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings"""
        try:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)

            if not emb1 or not emb2:
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0


# Convenience function for integration with existing code
def search_restaurants_intelligently(
    query: str, 
    destination: str, 
    config, 
    min_results: int = 2,
    max_results: int = 8
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Convenience function to perform AI-powered semantic database search

    Returns:
        Tuple of (restaurants, should_proceed_to_web_scraping)
    """
    from utils.database import get_database

    db = get_database()
    semantic_search = SemanticDatabaseSearch(config, db)

    return semantic_search.search_database_intelligently(
        query, destination, min_results, max_results
    )