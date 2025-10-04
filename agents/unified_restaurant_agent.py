# agents/unified_restaurant_agent.py
"""
Unified LangGraph Restaurant Agent - ALL TYPE ERRORS FIXED

This agent acts as a UNIFIED ORCHESTRATOR that:
1. Detects search flow type (city vs location)
2. Routes to appropriate specialized pipeline
3. PRESERVES all existing agent implementations
4. Provides single entry point for all restaurant searches

Key Design Principles:
- All existing agents are used AS-IS (no logic changes)
- Orchestration only - no business logic here
- Drop-in replacement for both orchestrators
- Human-in-the-loop for location enhancement decisions
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Tuple
from typing import cast
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langsmith import traceable


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
    location_data: Optional[Any]  # LocationData object for location searches

    # Flow control
    search_flow: str  # "city_search" | "location_search"
    current_step: str
    human_decision_pending: bool
    human_decision_result: Optional[str]

    # City search pipeline data (preserve all existing state variables)
    query_analysis: Optional[Dict[str, Any]]
    destination: Optional[str]
    database_results: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]
    search_results: Optional[Dict[str, Any]]
    scraped_results: Optional[List[Dict[str, Any]]]
    cleaned_file_path: Optional[str]
    edited_results: Optional[Dict[str, Any]]

    # Location search pipeline data (preserve all existing state variables)
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

    # Metadata
    processing_time: Optional[float]
    pipeline_stats: Optional[Dict[str, Any]]


class UnifiedRestaurantAgent:
    """
    Unified LangGraph agent that preserves ALL existing agent logic
    while providing clean, single-entry-point orchestration
    """

    def __init__(self, config):
        self.config = config

        logger.info("🚀 Initializing Unified Restaurant Agent")

        # Initialize AI model for orchestration decisions
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Initialize ALL existing agents (preserve their logic)
        self._init_city_search_agents()
        self._init_location_search_agents()
        self._init_formatters()

        # Build the unified graph
        self.checkpointer = MemorySaver()
        self.graph = self._build_unified_graph()

        logger.info("✅ Unified Restaurant Agent initialized with preserved agent logic")

    def _init_city_search_agents(self):
        """Initialize city search agents (preserve existing logic)"""
        self.query_analyzer = QueryAnalyzer(self.config)
        self.database_search_agent = DatabaseSearchAgent(self.config)
        self.dbcontent_evaluation_agent = ContentEvaluationAgent(self.config)
        self.search_agent = BraveSearchAgent(self.config)
        self.scraper = BrowserlessRestaurantScraper(self.config)
        self.text_cleaner = TextCleanerAgent(self.config)
        self.editor_agent = EditorAgent(self.config)
        self.follow_up_search_agent = FollowUpSearchAgent(self.config)

        # Set up agent connections (preserve existing patterns)
        self.dbcontent_evaluation_agent.set_brave_search_agent(self.search_agent)

    def _init_location_search_agents(self):
        """Initialize location search agents (preserve existing logic)"""
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

        # === UNIVERSAL ENTRY POINT ===
        graph.add_node("detect_flow", self._detect_search_flow)

        # === CITY SEARCH NODES (preserve existing pipeline) ===
        graph.add_node("city_analyze_query", self._city_analyze_query)
        graph.add_node("city_search_database", self._city_search_database)
        graph.add_node("city_evaluate_content", self._city_evaluate_content)
        graph.add_node("city_web_search", self._city_web_search)
        graph.add_node("city_scrape_content", self._city_scrape_content)
        graph.add_node("city_clean_content", self._city_clean_content)
        graph.add_node("city_edit_content", self._city_edit_content)
        graph.add_node("city_format_results", self._city_format_results)

        # === LOCATION SEARCH NODES (preserve existing pipeline) ===
        graph.add_node("location_geocode", self._location_geocode)
        graph.add_node("location_search_database", self._location_search_database)
        graph.add_node("location_filter_results", self._location_filter_results)
        graph.add_node("location_human_decision", self._location_human_decision)
        graph.add_node("location_maps_search", self._location_maps_search)
        graph.add_node("location_media_verification", self._location_media_verification)
        graph.add_node("location_format_results", self._location_format_results)

        # === FLOW ROUTING ===
        graph.set_entry_point("detect_flow")

        # Route from flow detection
        graph.add_conditional_edges(
            "detect_flow",
            self._route_by_flow,
            {
                "city_search": "city_analyze_query",
                "location_search": "location_geocode"
            }
        )

        # === CITY SEARCH FLOW ===
        graph.add_edge("city_analyze_query", "city_search_database")
        graph.add_edge("city_search_database", "city_evaluate_content")

        # Route after evaluation
        graph.add_conditional_edges(
            "city_evaluate_content",
            self._route_after_evaluation,
            {
                "sufficient": "city_edit_content",
                "needs_search": "city_web_search"
            }
        )

        # Web search flow
        graph.add_edge("city_web_search", "city_scrape_content")
        graph.add_edge("city_scrape_content", "city_clean_content")
        graph.add_edge("city_clean_content", "city_edit_content")

        # Final city formatting
        graph.add_edge("city_edit_content", "city_format_results")
        graph.add_edge("city_format_results", END)

        # === LOCATION SEARCH FLOW ===
        graph.add_edge("location_geocode", "location_search_database")
        graph.add_edge("location_search_database", "location_filter_results")

        # Route after filtering
        graph.add_conditional_edges(
            "location_filter_results",
            self._route_after_filtering,
            {
                "sufficient": "location_format_results",
                "needs_enhancement": "location_human_decision"
            }
        )

        # Human decision flow
        graph.add_conditional_edges(
            "location_human_decision",
            self._route_after_human_decision,
            {
                "accept": "location_maps_search",
                "skip": "location_format_results"
            }
        )

        # Maps search flow
        graph.add_edge("location_maps_search", "location_media_verification")
        graph.add_edge("location_media_verification", "location_format_results")
        graph.add_edge("location_format_results", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ============================================================================
    # ROUTING FUNCTIONS
    # ============================================================================

    def _route_by_flow(self, state: UnifiedSearchState) -> str:
        """Route based on detected search flow"""
        return state["search_flow"]

    def _route_after_evaluation(self, state: UnifiedSearchState) -> str:
        """Route after database content evaluation"""
        evaluation_results = state.get("evaluation_results")
        if evaluation_results and evaluation_results.get("route") == "database_sufficient":
            return "sufficient"
        return "needs_search"

    def _route_after_filtering(self, state: UnifiedSearchState) -> str:
        """Route after location filtering"""
        filtered_results = state.get("filtered_results")
        if filtered_results and filtered_results.get("enhancement_needed"):
            return "needs_enhancement"
        return "sufficient"

    def _route_after_human_decision(self, state: UnifiedSearchState) -> str:
        """Route after human decision"""
        if state.get("human_decision_pending", False):
            # Wait for human input
            return "accept"  # Default, will be updated by decision handler

        decision = state.get("human_decision_result", "skip")
        return "accept" if decision == "accept" else "skip"

    # ============================================================================
    # UNIVERSAL NODES
    # ============================================================================

    @traceable(name="detect_search_flow")
    def _detect_search_flow(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Detect whether this is a city search or location search"""
        try:
            logger.info("🔍 Flow Detection")

            # Check for location-specific indicators
            has_coordinates = bool(state.get("gps_coordinates"))
            has_location_data = bool(state.get("location_data"))

            # Location keywords
            location_keywords = ["near me", "nearby", "close", "around here", "in my area"]
            query_lower = state["query"].lower()
            has_location_keywords = any(keyword in query_lower for keyword in location_keywords)

            # Determine flow
            if has_coordinates or has_location_data or has_location_keywords:
                search_flow = "location_search"
                logger.info("🗺️ Detected: Location-based search")
            else:
                search_flow = "city_search"
                logger.info("🏙️ Detected: City-based search")

            return {
                **state,
                "search_flow": search_flow,
                "current_step": "flow_detected"
            }
        except Exception as e:
            logger.error(f"❌ Error in flow detection: {e}")
            return {
                **state,
                "search_flow": "city_search",  # Default fallback
                "current_step": "flow_detection_failed",
                "error_message": f"Flow detection failed: {str(e)}"
            }

    # ============================================================================
    # CITY SEARCH NODES (PRESERVE EXISTING AGENT LOGIC)
    # ============================================================================

    @traceable(name="city_analyze_query")
    def _city_analyze_query(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing QueryAnalyzer (preserve logic)"""
        try:
            logger.info("🔍 City Query Analysis")

            # Use existing QueryAnalyzer agent AS-IS
            analysis_result = self.query_analyzer.analyze(state["query"])

            return {
                **state,
                "query_analysis": analysis_result,
                "destination": analysis_result.get("destination"),
                "current_step": "query_analyzed"
            }
        except Exception as e:
            logger.error(f"❌ Error in city query analysis: {e}")
            return {
                **state,
                "error_message": f"Query analysis failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_search_database")
    def _city_search_database(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing DatabaseSearchAgent (preserve logic)"""
        try:
            logger.info("🗃️ City Database Search")

            query_analysis = state.get("query_analysis")
            if not query_analysis:
                raise ValueError("No query analysis available")

            # Use existing DatabaseSearchAgent AS-IS
            db_results = self.database_search_agent.search_and_evaluate(
                query_analysis=query_analysis
            )

            return {
                **state,
                "database_results": db_results,
                "current_step": "database_searched"
            }
        except Exception as e:
            logger.error(f"❌ Error in city database search: {e}")
            return {
                **state,
                "error_message": f"Database search failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_evaluate_content")
    def _city_evaluate_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing ContentEvaluationAgent (preserve logic)"""
        try:
            logger.info("⚖️ City Content Evaluation")

            database_results = state.get("database_results")
            destination = state.get("destination", "Unknown")

            if not database_results:
                # No database results, needs web search
                return {
                    **state,
                    "evaluation_results": {"route": "web_search_needed"},
                    "current_step": "content_evaluated"
                }

            # Prepare pipeline_data in the format expected by ContentEvaluationAgent
            pipeline_data = {
                "database_search_result": database_results,
                "raw_query": state.get("raw_query", state["query"]),
                "query": state["query"],
                "destination": destination,
                "query_analysis": state.get("query_analysis", {}),
                # Add any other required fields that the agent expects
            }

            # Use existing ContentEvaluationAgent AS-IS
            evaluation = self.dbcontent_evaluation_agent.evaluate_and_route(
                pipeline_data=pipeline_data
            )

            return {
                **state,
                "evaluation_results": evaluation,
                "current_step": "content_evaluated"
            }
        except Exception as e:
            logger.error(f"❌ Error in city content evaluation: {e}")
            return {
                **state,
                "error_message": f"Content evaluation failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_web_search")
    def _city_web_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing BraveSearchAgent (preserve logic)"""
        try:
            logger.info("🌐 City Web Search")

            destination = state.get("destination")
            if not destination:
                raise ValueError("No destination available for search")

            # Use existing BraveSearchAgent AS-IS
            search_results = self.search_agent.search(
                query=state["query"],
                destination=destination
            )

            return {
                **state,
                "search_results": search_results,
                "current_step": "web_searched"
            }
        except Exception as e:
            logger.error(f"❌ Error in city web search: {e}")
            return {
                **state,
                "error_message": f"Web search failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_scrape_content")
    def _city_scrape_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing BrowserlessRestaurantScraper (preserve logic)"""
        try:
            logger.info("🕷️ City Content Scraping")

            search_results = state.get("search_results")
            if not search_results or not isinstance(search_results, list):
                raise ValueError("No valid search results available for scraping")

            # Use existing BrowserlessRestaurantScraper AS-IS
            scraped_results = self.scraper.scrape_search_results(
                search_results=search_results
            )

            return {
                **state,
                "scraped_results": scraped_results,
                "current_step": "content_scraped"
            }
        except Exception as e:
            logger.error(f"❌ Error in city content scraping: {e}")
            return {
                **state,
                "error_message": f"Content scraping failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_clean_content")
    def _city_clean_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing TextCleanerAgent (preserve logic)"""
        try:
            logger.info("🧹 City Content Cleaning")

            scraped_results = state.get("scraped_results")
            if not scraped_results or not isinstance(scraped_results, list):
                raise ValueError("No valid scraped results available for cleaning")

            query = state.get("query", "")
            if not query:
                raise ValueError("No query available for content cleaning")

            # Use existing TextCleanerAgent AS-IS
            cleaned_file_path = self.text_cleaner.process_scraped_results_individually(
                scraped_results=scraped_results,
                query=query
            )

            return {
                **state,
                "cleaned_file_path": cleaned_file_path,
                "current_step": "content_cleaned"
            }
        except Exception as e:
            logger.error(f"❌ Error in city content cleaning: {e}")
            return {
                **state,
                "error_message": f"Content cleaning failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_edit_content")
    def _city_edit_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing EditorAgent (preserve logic)"""
        try:
            logger.info("✏️ City Content Editing")

            destination = state.get("destination")
            if not destination:
                raise ValueError("No destination available for editing")

            # Extract database restaurants from database_results
            database_results = state.get("database_results")
            database_restaurants = None
            if database_results:
                # Handle different database result formats
                if isinstance(database_results, dict):
                    database_restaurants = database_results.get("restaurants", [])
                elif isinstance(database_results, list):
                    database_restaurants = database_results

            scraped_results = state.get("scraped_results")
            cleaned_file_path = state.get("cleaned_file_path")

            # Use existing EditorAgent AS-IS with correct parameter names
            edited_results = self.editor_agent.edit(
                destination=destination,
                database_restaurants=database_restaurants,  # Corrected parameter name
                scraped_results=scraped_results,
                cleaned_file_path=cleaned_file_path,
                raw_query=state["query"]
            )

            return {
                **state,
                "edited_results": edited_results,
                "current_step": "content_edited"
            }
        except Exception as e:
            logger.error(f"❌ Error in city content editing: {e}")
            return {
                **state,
                "error_message": f"Content editing failed: {str(e)}",
                "success": False
            }

    @traceable(name="city_format_results")
    def _city_format_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing TelegramFormatter (preserve logic)"""
        try:
            logger.info("📝 City Results Formatting")

            edited_results = state.get("edited_results")
            if not edited_results:
                raise ValueError("No edited results available for formatting")

            # Use existing TelegramFormatter AS-IS
            formatted_message = self.telegram_formatter.format_recommendations(edited_results)
            restaurants = edited_results.get("main_list", [])

            return {
                **state,
                "formatted_message": formatted_message,
                "final_restaurants": restaurants,
                "success": True,
                "current_step": "results_formatted"
            }
        except Exception as e:
            logger.error(f"❌ Error in city results formatting: {e}")
            return {
                **state,
                "error_message": f"Results formatting failed: {str(e)}",
                "success": False
            }

    # ============================================================================
    # LOCATION SEARCH NODES (PRESERVE EXISTING AGENT LOGIC)
    # ============================================================================

    @traceable(name="location_geocode")
    def _location_geocode(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing location utilities (preserve logic)"""
        try:
            logger.info("🗺️ Location Geocoding")

            location_data = state.get("location_data")
            gps_coords = state.get("gps_coordinates")

            if gps_coords:
                coordinates = gps_coords
            elif location_data and hasattr(location_data, 'latitude'):
                coordinates = (location_data.latitude, location_data.longitude)
            else:
                # Use existing geocoding logic
                coordinates = self.location_utils.geocode_location(state["query"])

            return {
                **state,
                "location_coordinates": coordinates,
                "current_step": "location_geocoded"
            }
        except Exception as e:
            logger.error(f"❌ Error in location geocoding: {e}")
            return {
                **state,
                "error_message": f"Location geocoding failed: {str(e)}",
                "success": False
            }

    @traceable(name="location_search_database")
    def _location_search_database(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing LocationDatabaseService (preserve logic)"""
        try:
            logger.info("🗃️ Location Database Search")

            coordinates = state.get("location_coordinates")
            if not coordinates:
                raise ValueError("No coordinates available for location search")

            # Use existing database service
            results = self.location_database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=self.config.DB_PROXIMITY_RADIUS_KM
            )

            return {
                **state,
                "proximity_results": results,
                "current_step": "location_database_searched"
            }
        except Exception as e:
            logger.error(f"❌ Error in location database search: {e}")
            return {
                **state,
                "error_message": f"Location database search failed: {str(e)}",
                "success": False
            }

    @traceable(name="location_filter_results")
    def _location_filter_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing LocationFilterEvaluator (preserve logic)"""
        try:
            logger.info("🔍 Location Results Filtering")

            proximity_results = state.get("proximity_results")
            if not proximity_results:
                # No results to filter, proceed to enhancement
                return {
                    **state,
                    "filtered_results": {"enhancement_needed": True, "restaurants": []},
                    "current_step": "location_filtered"
                }

            restaurants = proximity_results.get("restaurants", [])
            if not isinstance(restaurants, list):
                raise ValueError("Invalid proximity results format")

            # Use existing LocationFilterEvaluator AS-IS with required parameters
            query = state.get("query", "restaurant")
            location_description = f"Location search: {query}"

            filtered_results = self.location_filter_evaluator.filter_and_evaluate(
                restaurants=restaurants,
                query=query,
                location_description=location_description
            )

            return {
                **state,
                "filtered_results": filtered_results,
                "current_step": "location_filtered"
            }
        except Exception as e:
            logger.error(f"❌ Error in location filtering: {e}")
            return {
                **state,
                "error_message": f"Location filtering failed: {str(e)}",
                "success": False
            }

    @traceable(name="location_human_decision")
    def _location_human_decision(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Handle human-in-the-loop decision for location enhancement"""
        try:
            logger.info("🤔 Location Human Decision")

            # Mark that human decision is pending
            return {
                **state,
                "human_decision_pending": True,
                "current_step": "human_decision_pending"
            }
        except Exception as e:
            logger.error(f"❌ Error in location human decision: {e}")
            return {
                **state,
                "error_message": f"Human decision setup failed: {str(e)}",
                "success": False
            }

    @traceable(name="location_maps_search")
    def _location_maps_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing LocationMapSearchAgent (preserve logic)"""
        try:
            logger.info("🗺️ Location Maps Search")

            coordinates = state.get("location_coordinates")
            if not coordinates:
                raise ValueError("No coordinates available for maps search")

            query = state.get("query", "restaurant")
            location_description = f"near {coordinates[0]}, {coordinates[1]}"

            # Use existing LocationMapSearchAgent AS-IS with all required parameters
            maps_results = self.location_map_search_agent.search_venues_with_ai_analysis(
                query=query,
                location_description=location_description,
                coordinates=coordinates
            )

            return {
                **state,
                "maps_results": maps_results,
                "current_step": "location_maps_searched"
            }
        except Exception as e:
            logger.error(f"❌ Error in location maps search: {e}")
            return {
                **state,
                "error_message": f"Location maps search failed: {str(e)}",
                "success": False
            }

    @traceable(name="location_media_verification")
    def _location_media_verification(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing LocationMediaVerificationAgent (preserve logic)"""
        try:
            logger.info("📱 Location Media Verification")

            maps_results = state.get("maps_results")
            if not maps_results:
                raise ValueError("No maps results available for verification")

            venues = maps_results.get("venues", [])
            query = state.get("query", "")

            # Use existing LocationMediaVerificationAgent AS-IS
            verification_results = self.location_media_verification_agent.verify_venues_media_coverage(
                venues=venues,
                query=query
            )

            return {
                **state,
                "media_verification_results": verification_results,
                "current_step": "location_media_verified"
            }
        except Exception as e:
            logger.error(f"❌ Error in location media verification: {e}")
            return {
                **state,
                "error_message": f"Location media verification failed: {str(e)}",
                "success": False
            }

    @traceable(name="location_format_results")
    def _location_format_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing LocationTelegramFormatter (preserve logic)"""
        try:
            logger.info("📝 Location Results Formatting")

            # Get query and location description for formatting
            query = state.get("query", "restaurant")
            location_description = f"Location search: {query}"

            # Determine what results to format
            if state.get("media_verification_results"):
                # Format Google Maps + verification results
                results = state["media_verification_results"]
                venues = results if isinstance(results, list) else (results.get("verified_venues", []) if results else [])
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
                    location_description=location_description
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
            return {
                **state,
                "error_message": f"Location results formatting failed: {str(e)}",
                "success": False
            }

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
        """
        UNIFIED PUBLIC API for all restaurant searches

        Args:
            query: User's restaurant search query
            user_id: Optional user ID for tracking
            gps_coordinates: Optional GPS coordinates (lat, lng)
            location_data: Optional LocationData object
            thread_id: Optional thread ID for conversation continuity

        Returns:
            Dict containing search results and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"🚀 UNIFIED SEARCH: '{query}' (user: {user_id})")

            # Prepare initial state
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
                "processing_time": None,
                "pipeline_stats": None
            }

            # Generate thread_id if not provided
            if not thread_id:
                thread_id = f"search_{user_id}_{int(time.time())}"

            # Execute the unified graph
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            result = await self.graph.ainvoke(initial_state, config)

            # Add timing information
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

    def handle_human_decision(
        self,
        thread_id: str,
        decision: str  # "accept" or "skip"
    ) -> Dict[str, Any]:
        """
        Handle human-in-the-loop decision for location enhancement

        Args:
            thread_id: Thread ID for the conversation
            decision: User's decision ("accept" or "skip")

        Returns:
            Updated state or continuation result
        """
        try:
            logger.info(f"🤔 Human decision: {decision}")

            # Get current state from checkpointer using correct LangGraph pattern
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = self.checkpointer.get_tuple(config)

            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                # Access the values from the checkpoint
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})

                # Create updated state by casting the channel_values and updating specific fields
                current_state = dict(channel_values)  # Convert to mutable dict
                current_state["human_decision_result"] = decision
                current_state["human_decision_pending"] = False

                # Continue execution from the decision point
                result = self.graph.invoke(cast(UnifiedSearchState, current_state), config)
                return result
            else:
                logger.error("No conversation state found for thread_id")
                return {"success": False, "error_message": "No conversation found"}

        except Exception as e:
            logger.error(f"❌ Error handling human decision: {e}")
            return {"success": False, "error_message": f"Decision handling failed: {str(e)}"}


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_unified_restaurant_agent(config):
    """Factory function to create the unified restaurant agent"""
    return UnifiedRestaurantAgent(config)
