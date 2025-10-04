# agents/unified_restaurant_agent.py
"""
Unified LangGraph Restaurant Agent - PRESERVES ALL EXISTING AGENT LOGIC

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
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
    gps_coordinates: Optional[tuple]
    location_data: Optional[Any]  # LocationData object for location searches

    # Flow control
    search_flow: str  # "city_search" | "location_search"
    current_step: str
    human_decision_pending: bool
    human_decision_result: Optional[str]

    # City search pipeline data (preserve all existing state variables)
    query_analysis: Optional[Dict]
    destination: Optional[str]
    database_results: Optional[Dict]
    evaluation_results: Optional[Dict]
    search_results: Optional[Dict]
    scraped_results: Optional[Dict]
    cleaned_file_path: Optional[str]
    edited_results: Optional[Dict]

    # Location search pipeline data (preserve all existing state variables)
    location_coordinates: Optional[tuple]
    proximity_results: Optional[Dict]
    filtered_results: Optional[Dict]
    maps_results: Optional[Dict]
    media_verification_results: Optional[Dict]

    # Output
    final_restaurants: List[Dict]
    formatted_message: Optional[str]
    success: bool
    error_message: Optional[str]

    # Metadata
    processing_time: Optional[float]
    pipeline_stats: Optional[Dict]


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

    def _build_unified_graph(self) -> StateGraph:
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

        graph.add_conditional_edges(
            "city_evaluate_content",
            self._route_city_content_decision,
            {
                "web_search_needed": "city_web_search",
                "database_sufficient": "city_edit_content"
            }
        )

        graph.add_edge("city_web_search", "city_scrape_content")
        graph.add_edge("city_scrape_content", "city_clean_content")
        graph.add_edge("city_clean_content", "city_edit_content")
        graph.add_edge("city_edit_content", "city_format_results")
        graph.add_edge("city_format_results", END)

        # === LOCATION SEARCH FLOW ===
        graph.add_edge("location_geocode", "location_search_database")
        graph.add_edge("location_search_database", "location_filter_results")

        graph.add_conditional_edges(
            "location_filter_results",
            self._route_location_enhancement_decision,
            {
                "sufficient": "location_format_results",
                "offer_enhancement": "location_human_decision",
                "force_enhancement": "location_maps_search"
            }
        )

        # Human-in-the-loop routing
        graph.add_conditional_edges(
            "location_human_decision",
            lambda state: state.get("human_decision_result", "skip"),
            {
                "accept": "location_maps_search",
                "skip": "location_format_results"
            }
        )

        graph.add_edge("location_maps_search", "location_media_verification")
        graph.add_edge("location_media_verification", "location_format_results")
        graph.add_edge("location_format_results", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ============================================================================
    # FLOW DETECTION AND ROUTING
    # ============================================================================

    def _detect_search_flow(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Intelligent detection of search type (city vs location)"""
        query = state["query"].lower()
        has_gps = state.get("gps_coordinates") is not None
        has_location_data = state.get("location_data") is not None

        # Location indicators
        location_indicators = [
            "near", "around", "close to", "nearby", "within",
            "walking distance", "driving distance", "@"
        ]

        has_location_text = any(indicator in query for indicator in location_indicators)

        if has_gps or has_location_data or has_location_text:
            flow_type = "location_search"
            logger.info(f"🎯 Detected LOCATION search: GPS={has_gps}, LocationData={has_location_data}, TextIndicators={has_location_text}")
        else:
            flow_type = "city_search"
            logger.info(f"🎯 Detected CITY search for query: {query}")

        return {
            **state,
            "search_flow": flow_type,
            "current_step": f"{flow_type}_detected"
        }

    def _route_by_flow(self, state: UnifiedSearchState) -> str:
        """Route to appropriate search flow"""
        return state["search_flow"]

    # ============================================================================
    # CITY SEARCH NODES (PRESERVE EXISTING AGENT LOGIC)
    # ============================================================================

    @traceable(name="city_analyze_query")
    def _city_analyze_query(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing QueryAnalyzer (preserve logic)"""
        try:
            logger.info("🔍 City Query Analysis")
            result = self.query_analyzer.analyze(state["query"])

            return {
                **state,
                "query_analysis": result,
                "destination": result.get("destination", "Unknown"),
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
            result = self.database_search_agent.search_and_evaluate(state["query_analysis"])

            return {
                **state,
                "database_results": result,
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

            # Prepare data for evaluation agent (preserve existing interface)
            evaluation_data = {
                **state,
                "database_search_result": state["database_results"],
                "raw_query": state.get("raw_query", state["query"])
            }

            result = self.dbcontent_evaluation_agent.evaluate_and_route(evaluation_data)

            return {
                **state,
                "evaluation_results": result,
                "current_step": "content_evaluated"
            }
        except Exception as e:
            logger.error(f"❌ Error in city content evaluation: {e}")
            return {
                **state,
                "error_message": f"Content evaluation failed: {str(e)}",
                "success": False
            }

    def _route_city_content_decision(self, state: UnifiedSearchState) -> str:
        """Route based on content evaluation decision"""
        evaluation = state.get("evaluation_results", {})
        trigger_web = evaluation.get("trigger_web_search", False)

        if trigger_web:
            logger.info("🌐 Web search needed - routing to web pipeline")
            return "web_search_needed"
        else:
            logger.info("🗃️ Database content sufficient - skipping web search")
            return "database_sufficient"

    @traceable(name="city_web_search")
    def _city_web_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing BraveSearchAgent (preserve logic)"""
        try:
            logger.info("🌐 City Web Search")

            # Use evaluation results to trigger search (preserve existing pattern)
            evaluation = state["evaluation_results"]
            search_queries = evaluation.get("search_queries", [])

            if not search_queries:
                logger.warning("No search queries from evaluation - using fallback")
                search_queries = [f"restaurants {state.get('destination', state['query'])}"]

            # Execute web search using existing agent
            search_metadata = {
                'is_english_speaking': evaluation.get('is_english_speaking', True),
                'local_language': evaluation.get('local_language')
            }

            results = self.search_agent.search(
                search_queries, 
                state.get("destination", "Unknown"),
                query_metadata=search_metadata
            )

            return {
                **state,
                "search_results": results,
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

            search_results = state["search_results"]
            # Scraper is async, need to run in event loop
            loop = asyncio.get_event_loop()
            scraped_results = loop.run_until_complete(self.scraper.scrape_search_results(search_results))

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

            # Use existing text cleaner interface (async method)
            loop = asyncio.get_event_loop()
            cleaned_file_path = loop.run_until_complete(
                self.text_cleaner.process_scraped_results_individually(
                    state["scraped_results"],
                    state.get("destination", "Unknown")
                )
            )
            
            cleaned_data = {
                "cleaned_file_path": cleaned_file_path,
                "updated_scraped_results": state["scraped_results"]
            }

            return {
                **state,
                "cleaned_file_path": cleaned_data.get("cleaned_file_path"),
                "scraped_results": cleaned_data.get("updated_scraped_results", state["scraped_results"]),
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

            # Determine processing mode based on available data
            has_database = bool(state.get("database_results", {}).get("database_restaurants"))
            has_scraped = bool(state.get("scraped_results"))

            if has_database and has_scraped:
                mode = "hybrid"
            elif has_scraped:
                mode = "web_only"
            else:
                mode = "database_only"

            # Use existing editor agent interface
            edited_results = self.editor_agent.edit(
                processing_mode=mode,
                database_restaurants=state.get("database_results", {}).get("database_restaurants", []),
                scraped_results=state.get("scraped_results", []),
                raw_query=state.get("raw_query", state["query"]),
                destination=state.get("destination", "Unknown"),
                cleaned_file_path=state.get("cleaned_file_path")
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

            # Use existing formatter interface
            formatted_message = self.telegram_formatter.format_recommendations(
                state.get("edited_results", {})
            )

            restaurants = state.get("edited_results", {}).get("main_list", [])

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

            coordinates = state["location_coordinates"]

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
            logger.info("⚖️ Location Results Filtering")

            # Use existing filter evaluator
            filter_result = self.location_filter_evaluator.filter_and_evaluate(
                restaurants=state["proximity_results"],
                query=state["query"],
                location_description=f"GPS: {state['location_coordinates']}"
            )

            return {
                **state,
                "filtered_results": filter_result,
                "current_step": "location_results_filtered"
            }
        except Exception as e:
            logger.error(f"❌ Error in location results filtering: {e}")
            return {
                **state,
                "error_message": f"Location results filtering failed: {str(e)}",
                "success": False
            }

    def _route_location_enhancement_decision(self, state: UnifiedSearchState) -> str:
        """Decide if location enhancement is needed (human-in-the-loop logic)"""
        filtered_results = state.get("filtered_results", {})
        restaurant_count = len(filtered_results.get("restaurants", []))

        if restaurant_count >= 6:
            logger.info(f"🎯 Database sufficient: {restaurant_count} restaurants found")
            return "sufficient"
        elif restaurant_count >= 2:
            logger.info(f"🤔 Offering enhancement: {restaurant_count} restaurants found")
            return "offer_enhancement"
        else:
            logger.info(f"🔍 Force enhancement: only {restaurant_count} restaurants found")
            return "force_enhancement"

    @traceable(name="location_human_decision")
    def _location_human_decision(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Human-in-the-loop decision point"""
        restaurant_count = len(state.get("filtered_results", {}).get("restaurants", []))

        # This sets up the human decision point
        # The actual human decision would be handled by the calling code
        return {
            **state,
            "human_decision_pending": True,
            "human_decision_message": f"Found {restaurant_count} restaurants nearby. Search Google Maps for more options?",
            "current_step": "awaiting_human_decision"
        }

    @traceable(name="location_maps_search")
    def _location_maps_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Use existing LocationMapSearchAgent (preserve logic)"""
        try:
            logger.info("🗺️ Location Maps Search")

            # Use existing map search agent
            maps_results = self.location_map_search_agent.search_venues_with_ai_analysis(
                query=state["query"],
                coordinates=state["location_coordinates"],
                radius_km=self.config.LOCATION_SEARCH_RADIUS_KM
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
            logger.info("📺 Location Media Verification")

            venues = state.get("maps_results", {}).get("venues", [])

            # Use existing media verification agent
            verification_results = self.location_media_verification_agent.verify_venues_media_coverage(
                venues=venues,
                query=state["query"],
                coordinates=state["location_coordinates"]
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

            # Determine what results to format
            if state.get("media_verification_results"):
                # Format Google Maps + verification results
                results = state["media_verification_results"]
                formatted_message = self.location_formatter.format_google_maps_results(results)
                restaurants = results.get("verified_venues", [])
            else:
                # Format database-only results
                results = state.get("filtered_results", {})
                formatted_message = self.location_formatter.format_database_results(results)
                restaurants = results.get("restaurants", [])

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
        gps_coordinates: Optional[tuple] = None,
        location_data: Optional[Any] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified entry point for all restaurant searches

        Args:
            query: User's search query
            user_id: Optional user ID
            gps_coordinates: Optional GPS coordinates (lat, lng)
            location_data: Optional LocationData object
            thread_id: Optional thread ID for conversation persistence

        Returns:
            Dict with search results and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"🚀 Unified Restaurant Search: '{query}'")

            # Prepare initial state
            initial_state = {
                "query": query,
                "raw_query": query,
                "user_id": user_id,
                "gps_coordinates": gps_coordinates,
                "location_data": location_data,
                "search_flow": "",
                "current_step": "initialized",
                "human_decision_pending": False,
                "human_decision_result": None,
                "success": False,
                "error_message": None,
                "final_restaurants": [],
                "formatted_message": None,
                "processing_time": None,
                "pipeline_stats": None
            }

            # Execute the unified graph
            config = {"thread_id": thread_id} if thread_id else {}

            result = await self.graph.ainvoke(initial_state, config=config)

            # Add timing and metadata
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time

            if result.get("success"):
                logger.info(f"✅ Unified search completed in {processing_time}s: {len(result.get('final_restaurants', []))} restaurants")
            else:
                logger.error(f"❌ Unified search failed: {result.get('error_message', 'Unknown error')}")

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

            # Update the state with human decision
            current_state = self.checkpointer.get({"thread_id": thread_id})
            if current_state:
                current_state["human_decision_result"] = decision
                current_state["human_decision_pending"] = False

                # Continue execution from the decision point
                result = self.graph.invoke(current_state, config={"thread_id": thread_id})
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