# agents/unified_restaurant_agent.py
"""
Unified LangGraph Restaurant Agent - ALL METHOD NAMES CORRECTED

This agent acts as a UNIFIED ORCHESTRATOR that:
1. Detects search flow type (city vs location)
2. Routes to appropriate specialized pipeline
3. PRESERVES all existing agent implementations
4. Provides single entry point for all restaurant searches

CORRECTED METHOD NAMES FROM PROJECT FILES:
✅ BraveSearchAgent.search() - with search_queries, destination, query_metadata
✅ BrowserlessRestaurantScraper.scrape_search_results() - with search_results parameter
✅ TextCleanerAgent.process_scraped_results_individually() - with scraped_results, query
✅ EditorAgent.edit() - with destination, database_restaurants, scraped_results, etc.
✅ TelegramFormatter.format_recommendations() - with recommendations_data parameter
✅ LocationFilterEvaluator.filter_and_evaluate() - with restaurants, query, location_description
✅ LocationMapSearchAgent.search_venues_with_ai_analysis() - with coordinates, query
✅ LocationTelegramFormatter.format_database_results() and format_google_maps_results()
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, TypedDict, Tuple, cast
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

from utils.async_utils import sync_to_async

# Import ALL existing agents (preserve their logic)
from agents.query_analyzer import QueryAnalyzer
from agents.database_search_agent import DatabaseSearchAgent
from agents.dbcontent_evaluation_agent import ContentEvaluationAgent
from agents.search_agent import BraveSearchAgent
from agents.browserless_scraper import BrowserlessRestaurantScraper
from agents.text_cleaner_agent import TextCleanerAgent
from agents.editor_agent import EditorAgent
from agents.follow_up_search_agent import FollowUpSearchAgent

# Location-specific agents (preserve their logic)
from location.location_utils import LocationUtils
from location.database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_database_ai_editor import LocationDatabaseAIEditor
from location.location_map_search_ai_editor import LocationMapSearchAIEditor
from location.location_map_search import LocationMapSearchAgent
from location.location_media_verification import LocationMediaVerificationAgent
from location.location_telegram_formatter import LocationTelegramFormatter

# Formatters
from formatters.telegram_formatter import TelegramFormatter

logger = logging.getLogger(__name__)

class UnifiedSearchState(TypedDict):
    """Unified state schema for both city and location searches"""
    # Input
    query: str
    raw_query: str
    user_id: Optional[int]
    gps_coordinates: Optional[Tuple[float, float]]
    location_data: Optional[Any]

    # Flow control
    search_flow: str
    current_step: str
    human_decision_pending: bool
    human_decision_result: Optional[str]

    # City search pipeline data
    query_analysis: Optional[Dict[str, Any]]
    destination: Optional[str]
    database_results: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]
    search_results: Optional[List[Dict[str, Any]]]
    scraped_results: Optional[List[Dict[str, Any]]]
    cleaned_file_path: Optional[str]
    edited_results: Optional[Dict[str, Any]]

    # Location search pipeline data
    location_coordinates: Optional[Tuple[float, float]]
    proximity_results: Optional[Dict[str, Any]]
    filtered_results: Optional[Dict[str, Any]]
    maps_results: Optional[Dict[str, Any]]
    media_verification_results: Optional[Dict[str, Any]]

    # Output
    final_restaurants: List[Dict[str, Any]]
    formatted_message: Optional[str]
    success: bool
    error_message: Optional[str]
    processing_time: Optional[float]


class UnifiedRestaurantAgent:
    """Unified LangGraph agent with ALL correct method names"""

    def __init__(self, config):
        self.config = config
        logger.info("🚀 Initializing Unified Restaurant Agent")

        # Initialize all agents
        self._init_city_search_agents()
        self._init_location_search_agents()
        self._init_formatters()

        # Build the unified graph
        self.checkpointer = MemorySaver()
        self.graph = self._build_unified_graph()

        logger.info("✅ Unified Restaurant Agent initialized")

    def _init_city_search_agents(self):
        """Initialize city search agents"""
        self.query_analyzer = QueryAnalyzer(self.config)
        self.database_search_agent = DatabaseSearchAgent(self.config)
        self.dbcontent_evaluation_agent = ContentEvaluationAgent(self.config)
        self.search_agent = BraveSearchAgent(self.config)
        self.scraper = BrowserlessRestaurantScraper(self.config)
        self.text_cleaner = TextCleanerAgent(self.config)
        self.editor_agent = EditorAgent(self.config)
        self.follow_up_search_agent = FollowUpSearchAgent(self.config)
        self.dbcontent_evaluation_agent.set_brave_search_agent(self.search_agent)

    def _init_location_search_agents(self):
        """Initialize location search agents"""
        self.location_utils = LocationUtils()
        self.location_database_service = LocationDatabaseService(self.config)
        self.location_filter_evaluator = LocationFilterEvaluator(self.config)
        self.location_database_ai_editor = LocationDatabaseAIEditor(self.config)
        self.location_map_search_ai_editor = LocationMapSearchAIEditor(self.config)
        self.location_map_search_agent = LocationMapSearchAgent(self.config)
        self.location_media_verification_agent = LocationMediaVerificationAgent(self.config)

    def _init_formatters(self):
        """Initialize formatters"""
        self.telegram_formatter = TelegramFormatter(self.config)
        self.location_formatter = LocationTelegramFormatter(self.config)

    def _build_unified_graph(self):
        """Build the unified LangGraph with flow routing"""
        graph = StateGraph(UnifiedSearchState)

        # Add nodes
        graph.add_node("detect_flow", self._detect_search_flow)
        graph.add_node("city_analyze_query", self._city_analyze_query)
        graph.add_node("city_search_database", self._city_search_database)
        graph.add_node("city_evaluate_content", self._city_evaluate_content)
        graph.add_node("city_web_search", self._city_web_search)
        graph.add_node("city_scrape_content", self._city_scrape_content)
        graph.add_node("city_clean_content", self._city_clean_content)
        graph.add_node("city_edit_content", self._city_edit_content)
        graph.add_node("city_format_results", self._city_format_results)
        graph.add_node("location_geocode", self._location_geocode)
        graph.add_node("location_search_database", self._location_search_database)
        graph.add_node("location_filter_results", self._location_filter_results)
        graph.add_node("location_human_decision", self._location_human_decision)
        graph.add_node("location_maps_search", self._location_maps_search)
        graph.add_node("location_media_verification", self._location_media_verification)
        graph.add_node("location_format_results", self._location_format_results)

        # Set entry point
        graph.set_entry_point("detect_flow")

        # Add routing edges
        graph.add_conditional_edges(
            "detect_flow",
            self._route_by_flow,
            {"city_search": "city_analyze_query", "location_search": "location_geocode"}
        )

        # City search flow
        graph.add_edge("city_analyze_query", "city_search_database")
        graph.add_edge("city_search_database", "city_evaluate_content")
        graph.add_conditional_edges(
            "city_evaluate_content",
            self._route_after_evaluation,
            {"sufficient": "city_edit_content", "needs_search": "city_web_search"}
        )
        graph.add_edge("city_web_search", "city_scrape_content")
        graph.add_edge("city_scrape_content", "city_clean_content")
        graph.add_edge("city_clean_content", "city_edit_content")
        graph.add_edge("city_edit_content", "city_format_results")
        graph.add_edge("city_format_results", END)

        # Location search flow
        graph.add_edge("location_geocode", "location_search_database")
        graph.add_edge("location_search_database", "location_filter_results")
        graph.add_conditional_edges(
            "location_filter_results",
            self._route_after_filtering,
            {"sufficient": "location_format_results", "needs_enhancement": "location_human_decision"}
        )
        graph.add_conditional_edges(
            "location_human_decision",
            self._route_after_human_decision,
            {"accept": "location_maps_search", "skip": "location_format_results"}
        )
        graph.add_edge("location_maps_search", "location_media_verification")
        graph.add_edge("location_media_verification", "location_format_results")
        graph.add_edge("location_format_results", END)

        return graph.compile(checkpointer=self.checkpointer)

    # Routing functions
    def _route_by_flow(self, state: UnifiedSearchState) -> str:
        return state["search_flow"]

    def _route_after_evaluation(self, state: UnifiedSearchState) -> str:
        evaluation_results = state.get("evaluation_results")
        if evaluation_results and evaluation_results.get("database_sufficient"):
            return "sufficient"
        return "needs_search"

    def _route_after_filtering(self, state: UnifiedSearchState) -> str:
        filtered_results = state.get("filtered_results")
        if filtered_results and filtered_results.get("enhancement_needed"):
            return "needs_enhancement"
        return "sufficient"

    def _route_after_human_decision(self, state: UnifiedSearchState) -> str:
        if state.get("human_decision_pending", False):
            return "accept"
        decision = state.get("human_decision_result", "skip")
        return "accept" if decision == "accept" else "skip"

    # ============================================================================
    # CITY SEARCH NODES - ALL METHOD NAMES CORRECTED
    # ============================================================================

    @traceable(name="detect_search_flow")
    async def _detect_search_flow(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Detect search flow type"""
        try:
            logger.info("🔍 Flow Detection")

            has_coordinates = bool(state.get("gps_coordinates"))
            has_location_data = bool(state.get("location_data"))
            location_keywords = ["near me", "nearby", "close", "around here", "in my area"]
            query_lower = state["query"].lower()
            has_location_keywords = any(keyword in query_lower for keyword in location_keywords)

            if has_coordinates or has_location_data or has_location_keywords:
                search_flow = "location_search"
                logger.info("🗺️ Detected: Location-based search")
            else:
                search_flow = "city_search"
                logger.info("🏙️ Detected: City-based search")

            return {**state, "search_flow": search_flow, "current_step": "flow_detected"}
        except Exception as e:
            logger.error(f"❌ Error in flow detection: {e}")
            return {**state, "search_flow": "city_search", "error_message": f"Flow detection failed: {str(e)}"}

    @traceable(name="city_analyze_query")
    async def _city_analyze_query(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use QueryAnalyzer.analyze()"""
        try:
            logger.info("🔍 City Query Analysis")
            analysis_result = await sync_to_async(self.query_analyzer.analyze)(state["query"])

            return {
                **state,
                "query_analysis": analysis_result,
                "destination": analysis_result.get("destination"),
                "current_step": "query_analyzed"
            }
        except Exception as e:
            logger.error(f"❌ Error in city query analysis: {e}")
            return {**state, "error_message": f"Query analysis failed: {str(e)}", "success": False}

    @traceable(name="city_search_database")
    async def _city_search_database(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use DatabaseSearchAgent.search_and_evaluate()"""
        try:
            logger.info("🗃️ City Database Search")

            query_analysis = state.get("query_analysis")
            if not query_analysis:
                raise ValueError("No query analysis available")

            db_results = await sync_to_async(self.database_search_agent.search_and_evaluate)(
                query_analysis=query_analysis,
                original_query=state["query"]
            )

            return {**state, "database_results": db_results, "current_step": "database_searched"}
        except Exception as e:
            logger.error(f"❌ Error in city database search: {e}")
            return {**state, "error_message": f"Database search failed: {str(e)}", "success": False}

    @traceable(name="city_evaluate_content")
    async def _city_evaluate_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use ContentEvaluationAgent.evaluate_and_route() with pipeline_data"""
        try:
            logger.info("⚖️ City Content Evaluation")

            database_results = state.get("database_results")
            query_analysis = state.get("query_analysis")
            destination = state.get("destination", "Unknown")

            if not database_results:
                raise ValueError("No database results available")

            pipeline_data = {
                "database_restaurants": database_results.get("restaurants", []),
                "query_analysis": query_analysis,
                "destination": destination,
                "raw_query": state["query"]
            }

            evaluation_results = await sync_to_async(self.dbcontent_evaluation_agent.evaluate_and_route)(
                pipeline_data=pipeline_data
            )

            return {**state, "evaluation_results": evaluation_results, "current_step": "content_evaluated"}
        except Exception as e:
            logger.error(f"❌ Error in city content evaluation: {e}")
            return {**state, "error_message": f"Content evaluation failed: {str(e)}", "success": False}

    @traceable(name="city_web_search")
    async def _city_web_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use BraveSearchAgent.search() with proper parameters"""
        try:
            logger.info("🌐 City Web Search")

            query_analysis = state.get("query_analysis")
            if not query_analysis:
                raise ValueError("No query analysis available")

            search_queries = query_analysis.get("search_queries", [])
            destination = state.get("destination", "Unknown")
            query_metadata = query_analysis.get("query_metadata", {})

            search_results = await sync_to_async(self.search_agent.search)(
                search_queries=search_queries,
                destination=destination,
                query_metadata=query_metadata
            )

            return {**state, "search_results": search_results, "current_step": "web_searched"}
        except Exception as e:
            logger.error(f"❌ Error in city web search: {e}")
            return {**state, "error_message": f"Web search failed: {str(e)}", "success": False}

    @traceable(name="city_scrape_content")
    async def _city_scrape_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use BrowserlessRestaurantScraper.scrape_search_results()"""
        try:
            logger.info("🕷️ City Content Scraping")

            search_results = state.get("search_results")
            if not search_results or not isinstance(search_results, list):
                raise ValueError("No valid search results available for scraping")

            scraped_results = await sync_to_async(self.scraper.scrape_search_results)(
                search_results=search_results
            )

            return {**state, "scraped_results": scraped_results, "current_step": "content_scraped"}
        except Exception as e:
            logger.error(f"❌ Error in city content scraping: {e}")
            return {**state, "error_message": f"Content scraping failed: {str(e)}", "success": False}

    @traceable(name="city_clean_content")
    async def _city_clean_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use TextCleanerAgent.process_scraped_results_individually()"""
        try:
            logger.info("🧹 City Content Cleaning")

            scraped_results = state.get("scraped_results")
            if not scraped_results or not isinstance(scraped_results, list):
                raise ValueError("No valid scraped results available for cleaning")

            query = state.get("query", "")
            if not query:
                raise ValueError("No query available for content cleaning")

            cleaned_file_path = await sync_to_async(self.text_cleaner.process_scraped_results_individually)(
                scraped_results=scraped_results,
                query=query
            )

            return {**state, "cleaned_file_path": cleaned_file_path, "current_step": "content_cleaned"}
        except Exception as e:
            logger.error(f"❌ Error in city content cleaning: {e}")
            return {**state, "error_message": f"Content cleaning failed: {str(e)}", "success": False}

    @traceable(name="city_edit_content")
    async def _city_edit_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use EditorAgent.edit() with database_restaurants parameter"""
        try:
            logger.info("✏️ City Content Editing")

            destination = state.get("destination")
            if not destination:
                raise ValueError("No destination available for editing")

            database_results = state.get("database_results")
            database_restaurants = None
            if database_results:
                if isinstance(database_results, dict):
                    database_restaurants = database_results.get("restaurants", [])
                elif isinstance(database_results, list):
                    database_restaurants = database_results

            scraped_results = state.get("scraped_results")
            cleaned_file_path = state.get("cleaned_file_path")

            edited_results = await sync_to_async(self.editor_agent.edit)(
                destination=destination,
                database_restaurants=database_restaurants,
                scraped_results=scraped_results,
                cleaned_file_path=cleaned_file_path,
                raw_query=state["query"]
            )

            final_restaurants = edited_results.get("edited_results", {}).get("main_list", [])
            return {
                **state,
                "edited_results": edited_results,
                "final_restaurants": final_restaurants,
                "current_step": "content_edited"
            }
        except Exception as e:
            logger.error(f"❌ Error in city content editing: {e}")
            return {**state, "error_message": f"Content editing failed: {str(e)}", "success": False}

    @traceable(name="city_format_results")
    async def _city_format_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use TelegramFormatter.format_recommendations()"""
        try:
            logger.info("📝 City Results Formatting")

            edited_results = state.get("edited_results")
            if not edited_results:
                raise ValueError("No edited results available")

            main_list = edited_results.get("edited_results", {}).get("main_list", [])
            recommendations_data = {"main_list": main_list}

            formatted_message = await sync_to_async(self.telegram_formatter.format_recommendations)(
                recommendations_data=recommendations_data
            )

            return {
                **state,
                "formatted_message": formatted_message,
                "final_restaurants": main_list,
                "success": True,
                "current_step": "results_formatted"
            }
        except Exception as e:
            logger.error(f"❌ Error in city results formatting: {e}")
            return {**state, "error_message": f"Results formatting failed: {str(e)}", "success": False}

    # ============================================================================
    # LOCATION SEARCH NODES - ALL METHOD NAMES CORRECTED
    # ============================================================================

    @traceable(name="location_geocode")
    def _location_geocode(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Location geocoding"""
        try:
            logger.info("🗺️ Location Geocoding")

            location_data = state.get("location_data")
            gps_coords = state.get("gps_coordinates")

            if gps_coords:
                coordinates = gps_coords
            elif location_data and hasattr(location_data, 'latitude'):
                coordinates = (location_data.latitude, location_data.longitude)
            else:
                coordinates = self.location_utils.geocode_location(state["query"])

            return {**state, "location_coordinates": coordinates, "current_step": "location_geocoded"}
        except Exception as e:
            logger.error(f"❌ Error in location geocoding: {e}")
            return {**state, "error_message": f"Location geocoding failed: {str(e)}", "success": False}

    @traceable(name="location_search_database")
    def _location_search_database(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use LocationDatabaseService.search_by_proximity()"""
        try:
            logger.info("🗃️ Location Database Search")

            coordinates = state.get("location_coordinates")
            if not coordinates:
                raise ValueError("No coordinates available for location search")

            # FIXED: Use correct parameter name 'coordinates' and get radius from config
            results = self.location_database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=getattr(self.config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
            )

            return {**state, "proximity_results": results, "current_step": "location_database_searched"}
        except Exception as e:
            logger.error(f"❌ Error in location database search: {e}")
            return {**state, "error_message": f"Location database search failed: {str(e)}", "success": False}

    @traceable(name="location_filter_results")
    def _location_filter_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use LocationFilterEvaluator.filter_and_evaluate() with all required parameters"""
        try:
            logger.info("🔍 Location Results Filtering")

            proximity_results = state.get("proximity_results")
            if not proximity_results:
                return {
                    **state,
                    "filtered_results": {"enhancement_needed": True, "restaurants": []},
                    "current_step": "location_filtered"
                }

            restaurants = proximity_results.get("restaurants", [])
            if not isinstance(restaurants, list):
                raise ValueError("Invalid proximity results format")

            query = state.get("query", "restaurant")
            location_description = f"Location search: {query}"

            filtered_results = self.location_filter_evaluator.filter_and_evaluate(
                restaurants=restaurants,
                query=query,
                location_description=location_description
            )

            return {**state, "filtered_results": filtered_results, "current_step": "location_filtered"}
        except Exception as e:
            logger.error(f"❌ Error in location filtering: {e}")
            return {**state, "error_message": f"Location filtering failed: {str(e)}", "success": False}

    @traceable(name="location_human_decision")
    def _location_human_decision(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Human decision handler"""
        try:
            logger.info("🤔 Location Human Decision")
            return {**state, "human_decision_pending": True, "current_step": "human_decision_pending"}
        except Exception as e:
            logger.error(f"❌ Error in location human decision: {e}")
            return {**state, "error_message": f"Human decision setup failed: {str(e)}", "success": False}

    @traceable(name="location_maps_search")
    def _location_maps_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use LocationMapSearchAgent.search_venues_with_ai_analysis()"""
        try:
            logger.info("🗺️ Location Maps Search")

            coordinates = state.get("location_coordinates")
            if not coordinates:
                raise ValueError("No coordinates available for maps search")

            query = state.get("query", "restaurant")

            maps_results = self.location_map_search_agent.search_venues_with_ai_analysis(
                coordinates=coordinates,
                query=query
            )

            return {**state, "maps_results": maps_results, "current_step": "location_maps_searched"}
        except Exception as e:
            logger.error(f"❌ Error in location maps search: {e}")
            return {**state, "error_message": f"Location maps search failed: {str(e)}", "success": False}

    @traceable(name="location_media_verification")
    def _location_media_verification(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use LocationMediaVerificationAgent.verify_venues_media_coverage()"""
        try:
            logger.info("📱 Location Media Verification")

            maps_results = state.get("maps_results")
            if not maps_results:
                raise ValueError("No maps results available for verification")

            venues = maps_results.get("venues", [])
            query = state.get("query", "")

            verification_results = self.location_media_verification_agent.verify_venues_media_coverage(
                venues=venues,
                query=query
            )

            return {**state, "media_verification_results": verification_results, "current_step": "location_media_verified"}
        except Exception as e:
            logger.error(f"❌ Error in location media verification: {e}")
            return {**state, "error_message": f"Location media verification failed: {str(e)}", "success": False}

    @traceable(name="location_format_results")
    def _location_format_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """CORRECTED: Use LocationTelegramFormatter.format_database_results() and format_google_maps_results()"""
        try:
            logger.info("📝 Location Results Formatting")

            query = state.get("query", "restaurant")
            location_description = f"Location search: {query}"

            if state.get("media_verification_results"):
                # Format Google Maps + verification results
                results = state["media_verification_results"]
                venues = results if isinstance(results, list) else (results.get("restaurants", []) if results else [])
                formatted_result = self.location_formatter.format_google_maps_results(
                    venues=venues,
                    query=query,
                    location_description=location_description
                )
                formatted_message = formatted_result.get("message", "")
                restaurants = venues
            else:
                # Format database-only results
                results = state.get("filtered_results", {})
                restaurants_list = results.get("restaurants", []) if results else []
                formatted_result = self.location_formatter.format_database_results(
                    restaurants=restaurants_list,
                    query=query,
                    location_description=location_description,
                    offer_more_search=True
                )
                formatted_message = formatted_result.get("message", "")
                restaurants = restaurants_list

            return {
                **state,
                "formatted_message": formatted_message,
                "final_restaurants": restaurants,
                "success": True,
                "current_step": "location_results_formatted"
            }
        except Exception as e:
            logger.error(f"❌ Error in location results formatting: {e}")
            return {**state, "error_message": f"Location results formatting failed: {str(e)}", "success": False}

    # ============================================================================
    # PUBLIC API
    # ============================================================================

    async def search_restaurants(
        self,
        query: str,
        user_id: Optional[int] = None,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        location_data: Optional[Any] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Unified public API for all restaurant searches"""
        start_time = time.time()

        try:
            logger.info(f"🚀 UNIFIED SEARCH: '{query}' (user: {user_id})")

            initial_state: UnifiedSearchState = {
                "query": query,
                "raw_query": query,
                "user_id": user_id,
                "gps_coordinates": gps_coordinates,
                "location_data": location_data,
                "search_flow": "",
                "current_step": "initialized",
                "human_decision_pending": False,
                "human_decision_result": None,
                "query_analysis": None,
                "destination": None,
                "database_results": None,
                "evaluation_results": None,
                "search_results": None,
                "scraped_results": None,
                "cleaned_file_path": None,
                "edited_results": None,
                "location_coordinates": None,
                "proximity_results": None,
                "filtered_results": None,
                "maps_results": None,
                "media_verification_results": None,
                "final_restaurants": [],
                "formatted_message": None,
                "success": False,
                "error_message": None,
                "processing_time": None
            }

            if not thread_id:
                thread_id = f"search_{user_id}_{int(time.time())}"

            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            result = await self.graph.ainvoke(initial_state, config)

            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time

            logger.info(f"✅ UNIFIED SEARCH COMPLETE: {processing_time}s")
            return result

        except Exception as e:
            logger.error(f"❌ Error in unified restaurant search: {e}")
            return {
                "success": False,
                "error_message": f"Search failed: {str(e)}",
                "final_restaurants": [],
                "processing_time": round(time.time() - start_time, 2)
            }

    def handle_human_decision(self, thread_id: str, decision: str) -> Dict[str, Any]:
        """Handle human-in-the-loop decision"""
        try:
            logger.info(f"🤔 Human decision: {decision}")

            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = self.checkpointer.get_tuple(config)

            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                current_state = dict(channel_values)
                current_state["human_decision_result"] = decision
                current_state["human_decision_pending"] = False

                result = self.graph.invoke(cast(UnifiedSearchState, current_state), config)
                return result
            else:
                logger.error("No conversation state found for thread_id")
                return {"success": False, "error_message": "No conversation found"}

        except Exception as e:
            logger.error(f"❌ Error handling human decision: {e}")
            return {"success": False, "error_message": f"Decision handling failed: {str(e)}"}


def create_unified_restaurant_agent(config):
    """Factory function to create the unified restaurant agent"""
    return UnifiedRestaurantAgent(config)