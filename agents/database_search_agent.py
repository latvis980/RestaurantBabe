# agents/database_search_agent.py
"""
Database Search Agent - Handles database queries and decision logic for whether
to use database content or proceed with web search.

This agent centralizes all database-related logic that was previously scattered
in the langchain_orchestrator.py file.
"""

import logging
from typing import Dict, List, Any, Optional
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class DatabaseSearchAgent:
    """
    Handles database restaurant searches and decides whether database content
    is sufficient to answer user queries or if web search is needed.
    """

    def __init__(self, config):
        self.config = config

        # Current threshold (we'll make this smarter later)
        self.minimum_restaurant_threshold = getattr(config, 'MIN_DATABASE_RESTAURANTS', 3)

        # Future: AI evaluation settings
        self.ai_evaluation_enabled = getattr(config, 'DATABASE_AI_EVALUATION', False)

        logger.info(f"DatabaseSearchAgent initialized with threshold: {self.minimum_restaurant_threshold}")

    @log_function_call
    def search_and_evaluate(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method: Search database and decide if results are sufficient.

        This method handles ALL database logic including intelligent search.

        Args:
            query_analysis: Output from QueryAnalyzer containing destination, etc.

        Returns:
            Dict containing:
            - has_database_content: bool
            - database_results: List[Dict] (if has_database_content=True)
            - content_source: str ("database" or "web_search")
            - evaluation_details: Dict (for debugging/monitoring)
            - skip_web_search: bool (routing flag for orchestrator)
        """
        try:
            logger.info("🗃️ STARTING DATABASE SEARCH AND EVALUATION")

            # Extract destination and query from query analysis
            destination = query_analysis.get("destination", "Unknown")
            original_query = query_analysis.get("query", "")

            if destination == "Unknown":
                logger.info("⚠️ No destination detected, will use web search")
                return self._create_web_search_response("no_destination")

            if not original_query.strip():
                logger.warning("⚠️ No search query found")
                return self._create_web_search_response("no_query")

            # Try intelligent database search first (if available)
            try:
                intelligent_result = self._try_intelligent_search(original_query, destination)
                if intelligent_result:
                    return intelligent_result

            except ImportError:
                logger.info("🗃️ Intelligent search not available, using standard database search")
            except Exception as e:
                logger.warning(f"⚠️ Intelligent search failed: {e}, falling back to standard search")

            # Standard database search and evaluation
            return self._standard_database_search(destination, original_query, query_analysis)

        except Exception as e:
            logger.error(f"❌ Error in database search agent: {e}")
            dump_chain_state("database_search_error", {
                "error": str(e),
                "destination": query_analysis.get("destination", "Unknown")
            })
            return self._create_web_search_response("error")

    def _try_intelligent_search(self, search_query: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Try intelligent database search (if available).

        Returns:
            Dict with search results if successful, None if should fall back to standard search
        """
        try:
            from utils.intelligent_db_search import search_restaurants_intelligently

            logger.info(f"🧠 Trying intelligent search for: '{search_query}' in {destination}")

            relevant_restaurants, should_scrape = search_restaurants_intelligently(
                query=search_query,
                destination=destination,
                config=self.config,
                min_results=2,  # Need at least 2 relevant results
                max_results=8   # Maximum to return from database
            )

            if relevant_restaurants and not should_scrape:
                logger.info(f"✅ Intelligent search found {len(relevant_restaurants)} restaurants - skipping web search")
                return {
                    "has_database_content": True, 
                    "database_results": relevant_restaurants,
                    "skip_web_search": True,
                    "content_source": "database",
                    "evaluation_details": {
                        "sufficient": True,
                        "reason": f"intelligent_search_found_{len(relevant_restaurants)}_restaurants",
                        "details": {
                            "method": "intelligent_search",
                            "restaurant_count": len(relevant_restaurants),
                            "should_scrape": should_scrape
                        }
                    }
                }
            elif relevant_restaurants and should_scrape:
                logger.info(f"📊 Intelligent search found {len(relevant_restaurants)} restaurants but needs web search supplement")
                return {
                    "has_database_content": True, 
                    "database_results": relevant_restaurants,
                    "skip_web_search": False,
                    "content_source": "database_plus_web",
                    "evaluation_details": {
                        "sufficient": False,
                        "reason": f"intelligent_search_needs_supplement",
                        "details": {
                            "method": "intelligent_search",
                            "restaurant_count": len(relevant_restaurants),
                            "should_scrape": should_scrape
                        }
                    }
                }
            else:
                logger.info("📭 No relevant restaurants found in intelligent search - trying standard search")
                return None  # Fall back to standard search

        except ImportError:
            raise  # Re-raise ImportError to be caught by caller
        except Exception as e:
            logger.warning(f"⚠️ Intelligent search error: {e}")
            return None  # Fall back to standard search

    def _standard_database_search(self, destination: str, original_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard database search and evaluation logic.

        Args:
            destination: City/location to search
            original_query: Original user query
            query_analysis: Full query analysis from QueryAnalyzer

        Returns:
            Dict with search results and evaluation
        """
        try:
            logger.info(f"🔍 Standard database search for: {destination}")

            # Search database for restaurants
            database_restaurants = self._search_database(destination)

            # Evaluate if database results are sufficient
            evaluation_result = self._evaluate_database_results(
                database_restaurants, 
                query_analysis, 
                original_query
            )

            # Log final decision
            if evaluation_result["sufficient"]:
                logger.info(f"✅ DATABASE SUFFICIENT: Found {len(database_restaurants)} restaurants for {destination}")
                return self._create_database_response(database_restaurants, evaluation_result)
            else:
                logger.info(f"🌐 WEB SEARCH NEEDED: {evaluation_result['reason']}")
                return self._create_web_search_response(evaluation_result["reason"])

        except Exception as e:
            logger.error(f"❌ Error in standard database search: {e}")
            return self._create_web_search_response(f"standard_search_error: {str(e)}")


    def _search_database(self, destination: str) -> List[Dict[str, Any]]:
        """
        Search database for restaurants in the specified destination.

        Args:
            destination: City/location to search for

        Returns:
            List of restaurant dictionaries from database
        """
        try:
            # Extract city from destination (handle "Paris, France" format)
            city = destination
            if "," in destination:
                city = destination.split(",")[0].strip()

            logger.info(f"🔍 Searching database for restaurants in: {city}")

            # Import database utility and search
            from utils.database import get_database
            db = get_database()

            # Query with generous limit to get full picture
            database_restaurants = db.get_restaurants_by_city(city, limit=50)

            logger.info(f"📊 Database query returned {len(database_restaurants) if database_restaurants else 0} restaurants")

            return database_restaurants or []

        except Exception as e:
            logger.error(f"❌ Error searching database for {destination}: {e}")
            return []

    def _evaluate_database_results(
        self, 
        database_restaurants: List[Dict[str, Any]], 
        query_analysis: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """
        Evaluate if database results are sufficient to answer the user's query.

        Currently uses simple count-based logic, but designed to be extended 
        with AI evaluation in the future.

        Args:
            database_restaurants: List of restaurants from database
            query_analysis: Parsed query information
            original_query: Original user query string

        Returns:
            Dict with evaluation results:
            - sufficient: bool
            - reason: str (explanation)
            - details: Dict (additional info for debugging)
        """
        try:
            restaurant_count = len(database_restaurants)

            # Current simple evaluation logic
            evaluation = {
                "sufficient": False,
                "reason": "",
                "details": {
                    "restaurant_count": restaurant_count,
                    "threshold": self.minimum_restaurant_threshold,
                    "evaluation_method": "count_based",
                    "ai_evaluation_enabled": self.ai_evaluation_enabled
                }
            }

            # Simple count-based decision (current logic)
            if restaurant_count >= self.minimum_restaurant_threshold:
                evaluation["sufficient"] = True
                evaluation["reason"] = f"Database has {restaurant_count} restaurants (>= {self.minimum_restaurant_threshold})"
            else:
                evaluation["sufficient"] = False
                evaluation["reason"] = f"Only {restaurant_count} restaurants in database (< {self.minimum_restaurant_threshold})"

            # TODO: Future AI evaluation logic will go here
            if self.ai_evaluation_enabled:
                ai_evaluation = self._ai_evaluate_database_quality(
                    database_restaurants, 
                    query_analysis, 
                    original_query
                )
                evaluation["details"]["ai_evaluation"] = ai_evaluation
                # Could override the count-based decision here

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating database results: {e}")
            return {
                "sufficient": False,
                "reason": f"evaluation_error: {str(e)}",
                "details": {"error": str(e)}
            }

    def _ai_evaluate_database_quality(
        self, 
        database_restaurants: List[Dict[str, Any]], 
        query_analysis: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """
        Future: AI-powered evaluation of database restaurant quality and relevance.

        This method is a placeholder for upcoming AI evaluation features that will:
        1. Assess if database restaurants match the specific query
        2. Check for query-specific requirements (e.g., "new", "best", specific cuisine)
        3. Evaluate restaurant quality and description completeness
        4. Determine if additional web search would add value

        Args:
            database_restaurants: Restaurants from database
            query_analysis: Parsed query information  
            original_query: Original user query

        Returns:
            Dict with AI evaluation results
        """
        # Placeholder for future implementation
        logger.info("🤖 AI evaluation of database quality (placeholder - not yet implemented)")

        return {
            "implemented": False,
            "note": "AI evaluation will be implemented in future iteration",
            "would_evaluate": [
                "query_specificity_match",
                "restaurant_quality_assessment", 
                "description_completeness",
                "temporal_relevance",
                "cuisine_type_coverage"
            ]
        }

    def _create_database_response(
        self, 
        database_restaurants: List[Dict[str, Any]], 
        evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create response indicating database content should be used."""
        return {
            "has_database_content": True,
            "database_results": database_restaurants,
            "content_source": evaluation_result.get("details", {}).get("method") == "intelligent_search" 
                            and "database" or "database",
            "skip_web_search": True,
            "evaluation_details": evaluation_result
        }

    def _create_web_search_response(self, reason: str) -> Dict[str, Any]:
        """Create response indicating web search should be used."""
        return {
            "has_database_content": False,
            "database_results": [],
            "content_source": "web_search",
            "skip_web_search": False,
            "evaluation_details": {
                "sufficient": False,
                "reason": reason,
                "details": {}
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about database search performance."""
        try:
            from utils.database import get_database
            db = get_database()
            return db.get_database_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    def set_minimum_threshold(self, new_threshold: int):
        """
        Update the minimum restaurant threshold.

        Args:
            new_threshold: New minimum number of restaurants required
        """
        old_threshold = self.minimum_restaurant_threshold
        self.minimum_restaurant_threshold = new_threshold
        logger.info(f"Updated minimum restaurant threshold: {old_threshold} → {new_threshold}")

    def enable_ai_evaluation(self, enabled: bool = True):
        """
        Enable or disable AI evaluation (for future use).

        Args:
            enabled: Whether to enable AI evaluation
        """
        self.ai_evaluation_enabled = enabled
        logger.info(f"AI evaluation {'enabled' if enabled else 'disabled'}")