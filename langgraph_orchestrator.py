# langgraph_orchestrator.py
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
from typing import TypedDict, Optional, Any, List, Dict, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from datetime import datetime, timezone

from utils.async_utils import sync_to_async
from utils.ai_memory_system import AIMemorySystem, RestaurantMemory, ConversationState

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
from location.location_analyzer import LocationAnalyzer
from location.location_utils import LocationUtils
from location.location_database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_database_ai_editor import LocationDatabaseAIEditor
from location.location_map_search_ai_editor import LocationMapSearchAIEditor
from location.location_map_search import LocationMapSearchAgent
from location.location_media_verification import LocationMediaVerificationAgent

from utils.handoff_protocol import HandoffMessage, SearchContext, SearchType, HandoffCommand
from utils.ai_chat_layer import AIChatLayer

# Formatters
from formatters.telegram_formatter import TelegramFormatter
from formatters.location_telegram_formatter import LocationTelegramFormatter

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class UnifiedSearchState(TypedDict, total=False):
    """Unified state schema for both city and location searches"""
    # Input
    query: str
    raw_query: str
    user_id: Optional[int]
    gps_coordinates: Optional[Tuple[float, float]]
    location_data: Optional[Any]
    search_context: Optional[SearchContext]

    # Flow control
    search_flow: str
    current_step: str

    skip_database: bool                      # Skip database, go direct to Maps
    exclude_restaurant_ids: List[str]        # Previously shown IDs to exclude
    is_follow_up_search: bool                # Flag for follow-up Maps request

    # City search pipeline data (unchanged)
    query_analysis: Optional[Dict[str, Any]]
    destination: Optional[str]
    database_results: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]
    search_results: Optional[List[Dict[str, Any]]]
    scraped_results: Optional[List[Dict[str, Any]]]
    cleaned_file_path: Optional[str]
    edited_results: Optional[Dict[str, Any]]
    database_restaurants_hybrid: Optional[List[Dict[str, Any]]]
    is_hybrid_mode: bool

    # Location search pipeline data (unchanged)
    location_coordinates: Optional[Tuple[float, float]]
    proximity_results: Optional[Dict[str, Any]]
    filtered_results: Optional[Dict[str, Any]]
    maps_results: Optional[Dict[str, Any]]
    media_verification_results: Optional[Dict[str, Any]]

    # Output (unchanged)
    final_restaurants: List[Dict[str, Any]]
    formatted_message: Optional[str]
    success: bool
    error_message: Optional[str]
    processing_time: Optional[float]

@dataclass
class LocationSearchSession:
    """Session state for location-based searches - enables follow-up requests"""
    user_id: int
    query: str                                    # "specialty coffee near me"
    coordinates: Tuple[float, float]              # (38.7119, -9.1596)

    # Results tracking
    database_results_shown: bool = False
    maps_results_shown: bool = False
    last_shown_restaurant_ids: List[str] = field(default_factory=list)
    last_shown_count: int = 0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    can_request_more: bool = True
    search_exhausted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'user_id': self.user_id,
            'query': self.query,
            'coordinates': self.coordinates,
            'database_results_shown': self.database_results_shown,
            'maps_results_shown': self.maps_results_shown,
            'last_shown_restaurant_ids': self.last_shown_restaurant_ids,
            'last_shown_count': self.last_shown_count,
            'timestamp': self.timestamp.isoformat(),
            'can_request_more': self.can_request_more,
            'search_exhausted': self.search_exhausted
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationSearchSession':
        """Create from dictionary"""
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


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

        self.memory_system = AIMemorySystem(config)

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
        self.location_search_sessions: Dict[int, LocationSearchSession] = {}
        self.location_analyzer = LocationAnalyzer(self.config)
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

        # ═══════════════════════════════════════════════════════════════════════
        # ADD ALL NODES
        # ═══════════════════════════════════════════════════════════════════════

        # Flow detection
        graph.add_node("detect_flow", self._detect_search_flow)

        # City search nodes
        graph.add_node("city_analyze_query", self._city_analyze_query)
        graph.add_node("city_search_database", self._city_search_database)
        graph.add_node("city_evaluate_content", self._city_evaluate_content)
        graph.add_node("city_web_search", self._city_web_search)
        graph.add_node("city_scrape_content", self._city_scrape_content)
        graph.add_node("city_clean_content", self._city_clean_content)
        graph.add_node("city_edit_content", self._city_edit_content)
        graph.add_node("city_follow_up_search", self._city_follow_up_search)
        graph.add_node("city_format_results", self._city_format_results)

        # Location search nodes - UPDATED
        graph.add_node("location_geocode", self._location_geocode)
        graph.add_node("location_search_database", self._location_search_database)
        graph.add_node("location_filter_results", self._location_filter_results)
        graph.add_node("location_maps_search", self._location_maps_search)
        graph.add_node("location_media_verification", self._location_media_verification)
        graph.add_node("location_format_results", self._location_format_results)

        # ═══════════════════════════════════════════════════════════════════════
        # SET ENTRY POINT
        # ═══════════════════════════════════════════════════════════════════════

        graph.set_entry_point("detect_flow")

        # ═══════════════════════════════════════════════════════════════════════
        # MAIN FLOW ROUTING (detect_flow splits to city vs location)
        # ═══════════════════════════════════════════════════════════════════════

        graph.add_conditional_edges(
            "detect_flow",
            self._route_by_flow,
            {
                "city_search": "city_analyze_query",
                "location_search": "location_geocode"
            }
        )

        # ═══════════════════════════════════════════════════════════════════════
        # CITY SEARCH FLOW (unchanged from original)
        # ═══════════════════════════════════════════════════════════════════════

        graph.add_edge("city_analyze_query", "city_search_database")
        graph.add_edge("city_search_database", "city_evaluate_content")

        graph.add_conditional_edges(
            "city_evaluate_content",
            self._route_after_evaluation,
            {
                "sufficient": "city_edit_content",
                "needs_search": "city_web_search"
            }
        )

        graph.add_edge("city_web_search", "city_scrape_content")
        graph.add_edge("city_scrape_content", "city_clean_content")
        graph.add_edge("city_clean_content", "city_edit_content")
        graph.add_edge("city_edit_content", "city_follow_up_search")
        graph.add_edge("city_follow_up_search", "city_format_results")
        graph.add_edge("city_format_results", END)

        # ═══════════════════════════════════════════════════════════════════════
        # LOCATION SEARCH FLOW - COMPLETELY REDESIGNED
        # ═══════════════════════════════════════════════════════════════════════

        # After geocoding, check if this is a follow-up request (skip database)
        graph.add_conditional_edges(
            "location_geocode",
            self._route_after_geocode,
            {
                "database_search": "location_search_database",  # Normal: search database first
                "maps_direct": "location_maps_search"           # Follow-up: skip database, go to Maps
            }
        )

        # Database search always goes to filtering
        graph.add_edge("location_search_database", "location_filter_results")

        # After filtering, decide based on result count
        graph.add_conditional_edges(
            "location_filter_results",
            self._route_after_filtering,
            {
                "format_database": "location_format_results",  # >0 results: show them
                "continue_to_maps": "location_maps_search"     # 0 results: auto-continue to Maps
            }
        )

        # ═══════════════════════════════════════════════════════════════════════
        # TWO SEPARATE ENDING PATHS
        # ═══════════════════════════════════════════════════════════════════════

        # Path 1: Database results shown → END (session stored for follow-up)
        graph.add_edge("location_format_database_results", END)

        # Path 2: Maps search → verification → format → END (session cleared)
        graph.add_edge("location_maps_search", "location_media_verification")
        graph.add_edge("location_media_verification", "location_format_results")
        graph.add_edge("location_format_results", END)

        # ═══════════════════════════════════════════════════════════════════════
        # COMPILE GRAPH
        # ═══════════════════════════════════════════════════════════════════════

        return graph.compile(checkpointer=self.checkpointer)

    def _detect_needs_location_button(self, query: str) -> bool:
        """
        Detect if query requires GPS coordinates but user hasn't provided them

        Keywords indicating user needs location button:
        - "near me", "nearby", "around here", "close to me"
        - "where I am", "my location", "around me"
        """
        query_lower = query.lower()

        location_keywords = [
            "near me",
            "nearby",
            "around here",
            "close to me",
            "around me",
            "where i am",
            "my location",
            "in my area",
            "in the area"
        ]

        return any(keyword in query_lower for keyword in location_keywords)

    async def process_user_message_with_ai_chat(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        telegram_bot=None,
        chat_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process user message with structured handoffs and video confirmation

        This is the main entry point that:
        1. Gets structured handoff from AI Chat Layer
        2. Routes by command (conversation vs search vs resume)
        3. Sends video confirmation for searches
        4. Executes search with structured context

        Args:
            query: User's message
            user_id: User ID
            gps_coordinates: Optional GPS coordinates
            thread_id: Thread ID for state management
            telegram_bot: Telegram bot instance for sending messages
            chat_id: Chat ID for sending messages

        Returns:
            Dict with success, ai_response, search results, etc.
        """

        start_time = time.time()

        try:
            # Generate thread ID if needed
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            logger.info(f"💬 Processing with AI Chat Layer: '{query[:50]}...'")

            # Initialize AI Chat Layer if not exists
            if not hasattr(self, 'ai_chat_layer'):
                self.ai_chat_layer = AIChatLayer(self.config)
                logger.info("✅ AI Chat Layer initialized")

            # 1. Get structured handoff from supervisor
            handoff: HandoffMessage = await self.ai_chat_layer.process_message(
                user_id=user_id,
                user_message=query,
                gps_coordinates=gps_coordinates,
                thread_id=thread_id
            )

            logger.info(f"🎯 Handoff Command: {handoff.command.value}")
            logger.info(f"📝 Reasoning: {handoff.reasoning}")

            # 2. Route by command

            # ================================================================
            # COMMAND 1: CONTINUE_CONVERSATION (no search needed)
            # ================================================================
            if handoff.command == HandoffCommand.CONTINUE_CONVERSATION:
                # Just conversation - no search
                processing_time = round(time.time() - start_time, 2)

                # Detect if AI requested GPS by checking reasoning
                needs_gps = ('gps_required' in handoff.reasoning.lower() or 
                             'gps coordinates' in handoff.reasoning.lower())

                return {
                    "success": True,
                    "ai_response": handoff.conversation_response,
                    "action_taken": "conversation",
                    "search_triggered": False,
                    "needs_location_button": needs_gps,
                    "processing_time": processing_time,
                    "reasoning": handoff.reasoning
                }

            # ================================================================
            # COMMAND 2: RESUME_WITH_DECISION (resume paused graph)
            # ================================================================
            elif handoff.command == HandoffCommand.RESUME_WITH_DECISION:
                # User wants to resume paused graph execution (e.g., "let's find more")
                logger.info("🔄 Resuming paused graph execution")

                decision = handoff.decision or "accept"
                resume_thread_id = handoff.thread_id or thread_id

                logger.info(f"   Thread ID: {resume_thread_id}")
                logger.info(f"   Decision: {decision}")

                try:
                    # Use existing handle_human_decision method (ASYNC)
                    result = await self.handle_human_decision(resume_thread_id, decision)

                    # Add processing time
                    processing_time = round(time.time() - start_time, 2)
                    result["processing_time"] = processing_time
                    result["action_taken"] = "resume_graph"
                    result["resumed_with_decision"] = decision

                    logger.info(f"✅ Graph resumed successfully in {processing_time}s")

                    return result

                except Exception as resume_error:
                    logger.error(f"❌ Error resuming graph: {resume_error}")

                    processing_time = round(time.time() - start_time, 2)
                    return {
                        "success": False,
                        "error_message": f"Failed to resume: {str(resume_error)}",
                        "ai_response": "I had trouble continuing the search. Could you try again?",
                        "processing_time": processing_time,
                        "reasoning": f"Resume error: {str(resume_error)}"
                    }

            # ================================================================
            # COMMAND 3: EXECUTE_SEARCH (start new search)
            # ================================================================
            elif handoff.command == HandoffCommand.EXECUTE_SEARCH:
                # Type guard: Ensure search_context exists
                search_ctx = handoff.search_context

                if search_ctx is None:
                    logger.error("❌ EXECUTE_SEARCH command but no search_context provided")
                    processing_time = round(time.time() - start_time, 2)
                    return {
                        "success": False,
                        "error_message": "No search context in handoff",
                        "ai_response": "I encountered an error processing your search.",
                        "search_triggered": False,
                        "processing_time": processing_time
                    }

                logger.info("🔍 EXECUTE_SEARCH command received")
                logger.info(f"   Destination: {search_ctx.destination}")
                logger.info(f"   Cuisine: {search_ctx.cuisine}")
                logger.info(f"   Type: {search_ctx.search_type.value}")

                # Build search query from context
                search_query = self._build_search_query_from_context(search_ctx)

                # Extract GPS coordinates from context if available
                search_gps = search_ctx.gps_coordinates or gps_coordinates

                # Send confirmation message with video
                confirmation_msg = None
                if telegram_bot and chat_id:
                    try:
                        confirmation_msg = self._send_search_confirmation_message(
                            telegram_bot=telegram_bot,
                            chat_id=chat_id,
                            search_query=search_query,
                            search_type=search_ctx.search_type.value
                        )
                    except Exception as conf_error:
                        logger.warning(f"⚠️ Could not send confirmation: {conf_error}")

                # Execute the search
                logger.info(f"🚀 Executing search: '{search_query}'")

                search_result = await self.search_restaurants(
                    query=search_query,
                    user_id=user_id,
                    gps_coordinates=search_gps,
                    thread_id=thread_id,
                    search_context=search_ctx
                )

                # Delete confirmation message after search completes
                if confirmation_msg and telegram_bot and chat_id:
                    try:
                        # Handle both Telegram Message object (has .message_id) and dict (has ['message_id'])
                        msg_id = None
                        if hasattr(confirmation_msg, 'message_id'):
                            msg_id = confirmation_msg.message_id
                        elif isinstance(confirmation_msg, dict) and 'message_id' in confirmation_msg:
                            msg_id = confirmation_msg['message_id']

                        if msg_id:
                            telegram_bot.delete_message(chat_id, msg_id)
                            logger.info("✅ Deleted confirmation message")
                        else:
                            logger.warning("⚠️ Confirmation message has no message_id")
                    except Exception as delete_error:
                        logger.warning(f"⚠️ Could not delete confirmation: {delete_error}")

                # Add processing time and return
                processing_time = round(time.time() - start_time, 2)
                search_result['processing_time'] = processing_time
                search_result['action_taken'] = 'search'
                search_result['search_triggered'] = True

                return search_result

            # ================================================================
            # UNKNOWN COMMAND (fallback)
            # ================================================================
            else:
                logger.warning(f"❌ Unknown handoff command: {handoff.command}")
                processing_time = round(time.time() - start_time, 2)
                return {
                    "success": False,
                    "error_message": f"Unknown command: {handoff.command}",
                    "ai_response": "I encountered an unexpected error. Please try again.",
                    "search_triggered": False,
                    "processing_time": processing_time
                }

        except Exception as e:
            logger.error(f"❌ Error in AI chat processing: {e}", exc_info=True)
            processing_time = round(time.time() - start_time, 2)
            return {
                "success": False,
                "error_message": str(e),
                "ai_response": "I encountered an error processing your request. Please try again.",
                "search_triggered": False,
                "processing_time": processing_time
            }

    async def _store_location_search_session(
        self,
        user_id: int,
        query: str,
        coordinates: Tuple[float, float],
        database_results_shown: bool = False,
        maps_results_shown: bool = False,
        last_shown_restaurant_ids: Optional[List[str]] = None,
        last_shown_count: int = 0,
        can_request_more: bool = True
    ) -> bool:
        """Store location search session for follow-up requests"""
        try:
            session = LocationSearchSession(
                user_id=user_id,
                query=query,
                coordinates=coordinates,
                database_results_shown=database_results_shown,
                maps_results_shown=maps_results_shown,
                last_shown_restaurant_ids=last_shown_restaurant_ids or [],
                last_shown_count=last_shown_count,
                can_request_more=can_request_more
            )

            self.location_search_sessions[user_id] = session

            logger.info(f"💾 Stored location search session for user {user_id}")
            logger.info(f"   Query: '{query}'")
            logger.info(f"   Coordinates: {coordinates}")
            logger.info(f"   Shown IDs: {len(last_shown_restaurant_ids or [])}")
            logger.info(f"   Can request more: {can_request_more}")

            return True

        except Exception as e:
            logger.error(f"❌ Error storing location search session: {e}")
            return False

    def _load_location_search_session(self, user_id: int) -> Optional[LocationSearchSession]:
        """Load location search session for follow-up requests"""
        try:
            session = self.location_search_sessions.get(user_id)

            if session:
                # Check if session is still valid (within last 30 minutes)
                age = (datetime.now(timezone.utc) - session.timestamp).total_seconds()
                if age > 1800:  # 30 minutes
                    logger.info(f"⏰ Location search session expired for user {user_id}")
                    del self.location_search_sessions[user_id]
                    return None

                logger.info(f"📂 Loaded location search session for user {user_id}")
                logger.info(f"   Query: '{session.query}'")
                logger.info(f"   Coordinates: {session.coordinates}")
                logger.info(f"   Can request more: {session.can_request_more}")

                return session

            logger.info(f"ℹ️ No location search session found for user {user_id}")
            return None

        except Exception as e:
            logger.error(f"❌ Error loading location search session: {e}")
            return None

    def _clear_location_search_session(self, user_id: int) -> bool:
        """Clear location search session (e.g., after Maps results shown)"""
        try:
            if user_id in self.location_search_sessions:
                del self.location_search_sessions[user_id]
                logger.info(f"🗑️ Cleared location search session for user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error clearing location search session: {e}")
            return False


    """
    HELPER METHOD - Build search query from structured context:
    """
    def _build_search_query_from_context(self, ctx: SearchContext) -> str:
        """
        Build search query string from structured context

        Args:
            ctx: SearchContext with search parameters

        Returns:
            Query string for search execution
        """
        parts = []

        # Add requirements
        if ctx.requirements:
            parts.extend(ctx.requirements)

        # Add cuisine
        if ctx.cuisine:
            parts.append(ctx.cuisine)

        # Add destination
        if ctx.destination:
            parts.append(f"in {ctx.destination}")

        # Build query
        query = " ".join(parts) if parts else ctx.user_query

        logger.info(f"📝 Built query from context: '{query}'")
        return query

    def _send_search_confirmation_message(
        self,
        telegram_bot,
        chat_id: int,
        search_query: str,
        search_type: str
    ) -> Optional[object]:
        """
        Send confirmation message with video before starting search
        Uses AI generation with static fallback for reliability

        Args:
            telegram_bot: Telegram bot instance
            chat_id: Chat ID to send message to
            search_query: User's search query
            search_type: "city_wide" or "location_based"

        Returns:
            Message object or None if failed
        """
        try:
            # HYBRID: Try AI generation first, fallback to static
            message = self._generate_confirmation_message_hybrid(search_query, search_type)

            # Choose video and emoji based on search type
            if search_type == "city_wide":
                video_path = 'media/searching.mp4'
                fallback_emoji = "🔍"
            else:  # location_based
                video_path = 'media/vicinity_search.mp4'
                fallback_emoji = "📍"

            # Try to send with video first
            try:
                import os
                if os.path.exists(video_path):
                    with open(video_path, 'rb') as video:
                        return telegram_bot.send_video(
                            chat_id,
                            video,
                            caption=f"{fallback_emoji} {message}",
                            parse_mode='HTML'
                        )
                else:
                    logger.warning(f"Video file not found: {video_path}")
                    raise FileNotFoundError("Video not available")

            except Exception as video_error:
                logger.warning(f"Could not send video: {video_error}")
                # Fallback to text message with emoji
                return telegram_bot.send_message(
                    chat_id,
                    f"{fallback_emoji} {message}",
                    parse_mode='HTML'
                )

        except Exception as e:
            logger.error(f"Error sending confirmation message: {e}")
            # Ultimate fallback
            try:
                return telegram_bot.send_message(
                    chat_id,
                    "🔍 <b>Searching for restaurants...</b>\n\nThis might take a moment.",
                    parse_mode='HTML'
                )
            except Exception:
                logger.error("Could not send any confirmation message")
                return None

    async def _send_search_confirmation(
        self, 
        telegram_bot, 
        chat_id: int, 
        ctx: SearchContext
    ) -> Optional[object]:
        """
        Send confirmation WITH VIDEO before search

        Args:
            telegram_bot: Telegram bot instance
            chat_id: Chat ID
            ctx: SearchContext with search details

        Returns:
            Message object or None if failed
        """
        try:
            cuisine_text = ctx.cuisine or "restaurants"
            destination_text = ctx.destination

            # Determine video and message based on context
            if ctx.is_new_destination:
                # Destination change
                message = (
                    f"🔄 <b>Switching to {destination_text}! "
                    f"Searching for {cuisine_text}...</b>\n\n"
                    f"⏱ This might take a minute while I check with my sources."
                )
                video_path = 'media/searching.mp4'
                emoji = "🔄"
            elif ctx.search_type == SearchType.LOCATION_SEARCH:
                # Location-based
                message = (
                    f"📍 <b>Searching for {cuisine_text} in {destination_text}...</b>\n\n"
                    f"⏱ Checking my curated collection and finding the best places nearby."
                )
                video_path = 'media/vicinity_search.mp4'
                emoji = "📍"
            else:
                # City-wide
                message = (
                    f"🔍 <b>Searching for {cuisine_text} in {destination_text}...</b>\n\n"
                    f"⏱ This might take a minute while I check with my sources."
                )
                video_path = 'media/searching.mp4'
                emoji = "🔍"

            # Try video first
            try:
                import os
                if os.path.exists(video_path):
                    with open(video_path, 'rb') as video:
                        return telegram_bot.send_video(
                            chat_id,
                            video,
                            caption=message,
                            parse_mode='HTML'
                        )
                else:
                    logger.warning(f"Video not found: {video_path}")
                    raise FileNotFoundError("Video not available")

            except Exception as video_error:
                logger.warning(f"Could not send video: {video_error}")
                # Fallback to text
                return telegram_bot.send_message(
                    chat_id,
                    f"{emoji} {message}",
                    parse_mode='HTML'
                )

        except Exception as e:
            logger.error(f"Confirmation failed: {e}")
            # Ultimate fallback
            try:
                return telegram_bot.send_message(
                    chat_id,
                    "🔍 <b>Searching...</b>",
                    parse_mode='HTML'
                )
            except Exception:
                return None


    def _generate_confirmation_message_hybrid(self, search_query: str, search_type: str) -> str:
        """
        HYBRID: Generate confirmation message with AI + static fallback

        Tries AI generation first with timeout, falls back to static messages for reliability

        Args:
            search_query: The accumulated user conversation context
            search_type: "city_wide" or "location_based"

        Returns:
            HTML-formatted confirmation message
        """
        # First try AI generation (with timeout)
        ai_message = self._try_ai_confirmation_message(search_query, search_type)

        if ai_message:
            logger.info(f"✅ Using AI-generated confirmation: {ai_message[:50]}...")
            return ai_message

        # Fallback to static messages
        logger.info("📝 Using static confirmation message (AI failed/timeout)")
        return self._get_static_confirmation_message(search_query, search_type)


    def _try_ai_confirmation_message(self, search_query: str, search_type: str) -> Optional[str]:
        """
        Try to generate AI confirmation message with timeout and error handling

        Returns:
            AI-generated message or None if failed
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            # Initialize AI if not exists
            if not hasattr(self, '_confirmation_ai'):
                self._confirmation_ai = ChatOpenAI(
                    model=getattr(self.config, 'AI_MESSAGE_MODEL', 'gpt-4o-mini'),
                    temperature=0.7,
                    max_completion_tokens=80,  # Keep it short
                    api_key=self.config.OPENAI_API_KEY,
                    timeout=3  # 3 second timeout
                )

            # Create prompt
            prompt = f"""Generate a brief confirmation message for a restaurant search bot.

    User request: {search_query}
    Search type: {search_type}

    Requirements:
    - 1-2 sentences max
    - Enthusiastic but professional  
    - Reference specific food/location if clear
    - Use <b> tags for emphasis
    - Mention it takes a moment

    Examples:
    - "sushi in Chiado" → "<b>Perfect! Searching for amazing sushi in Chiado.</b> Give me a moment to check my local contacts."
    - "best pizza" → "<b>Excellent! Finding the top pizza places for you.</b> This might take a minute while I check my curated list."

    Generate ONLY the message:"""

            # Execute with timeout
            def generate():
                response = self._confirmation_ai.invoke([HumanMessage(content=prompt)])
                return response.content.strip() if isinstance(response.content, str) else str(response.content).strip()

            with ThreadPoolExecutor() as executor:
                future = executor.submit(generate)
                try:
                    ai_message = future.result(timeout=2.5)  # 2.5 second timeout

                    # Validate the response
                    if ai_message and len(ai_message) > 10 and len(ai_message) < 200:
                        # Ensure proper HTML formatting
                        if not '<b>' in ai_message:
                            # Add bold to first sentence
                            sentences = ai_message.split('. ')
                            if sentences:
                                sentences[0] = f"<b>{sentences[0]}</b>"
                                ai_message = '. '.join(sentences)

                        return ai_message
                    else:
                        logger.warning(f"AI response invalid: {ai_message}")
                        return None

                except TimeoutError:
                    logger.warning("AI confirmation message generation timed out")
                    return None

        except Exception as e:
            logger.warning(f"AI confirmation message failed: {e}")
            return None


    def _get_static_confirmation_message(self, search_query: str, search_type: str) -> str:
        """
        Get static confirmation message with basic personalization

        Args:
            search_query: User's search query
            search_type: "city_wide" or "location_based"

        Returns:
            Static confirmation message with light personalization
        """
        # Extract basic info for light personalization
        query_lower = search_query.lower()

        # Detect cuisine type for basic personalization
        cuisine_detected = None
        cuisines = {
            'sushi': 'sushi', 'japanese': 'Japanese', 'ramen': 'ramen',
            'pizza': 'pizza', 'italian': 'Italian', 'pasta': 'Italian',
            'chinese': 'Chinese', 'thai': 'Thai', 'indian': 'Indian',
            'mexican': 'Mexican', 'burger': 'burger', 'steak': 'steak',
            'seafood': 'seafood', 'vegetarian': 'vegetarian', 'vegan': 'vegan'
        }

        for keyword, cuisine in cuisines.items():
            if keyword in query_lower:
                cuisine_detected = cuisine
                break

        # Detect location for basic personalization
        location_detected = None
        locations = {
            'chiado': 'Chiado', 'bairro alto': 'Bairro Alto', 
            'príncipe real': 'Príncipe Real', 'alfama': 'Alfama',
            'downtown': 'downtown', 'center': 'the center'
        }

        for keyword, location in locations.items():
            if keyword in query_lower:
                location_detected = location
                break

        # Generate personalized static message
        if search_type == "city_wide":
            if cuisine_detected:
                message = f"<b>Perfect! I'm searching for the best {cuisine_detected} places for you.</b>\n\nThis might take a minute while I check my curated collection."
            else:
                message = "<b>Perfect! I'm searching for the best restaurants for you.</b>\n\nThis might take a minute while I check my curated collection."
        else:  # location_based
            if cuisine_detected and location_detected:
                message = f"<b>Great! I'm searching for amazing {cuisine_detected} spots in {location_detected}.</b>\n\nGive me a moment to check my local guides."
            elif cuisine_detected:
                message = f"<b>Great! I'm searching for amazing {cuisine_detected} places in that area.</b>\n\nGive me a moment to check my local contacts."
            elif location_detected:
                message = f"<b>Great! I'm searching for amazing restaurants in {location_detected}.</b>\n\nGive me a moment to check my local guides."
            else:
                message = "<b>Great! I'm searching for amazing restaurants in that area.</b>\n\nGive me a moment to check my local guides and contacts."

        return message

    async def execute_restaurant_search_with_context(
        self,
        search_context: Dict[str, Any],
        search_type: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute restaurant search with full conversation context
        """
        try:
            # FIXED: Validate parameters before use
            if not search_context:
                logger.error("❌ No search context provided")
                return {
                    "success": False,
                    "error_message": "No search context provided",
                    "final_restaurants": [],
                    "processing_time": 0
                }

            if not search_type:
                logger.error("❌ No search type provided")
                return {
                    "success": False,
                    "error_message": "No search type provided", 
                    "final_restaurants": [],
                    "processing_time": 0
                }

            # Get the accumulated raw query and context
            raw_query = search_context.get('raw_query', '')
            collected_info = search_context.get('collected_info', {})

            logger.info(f"🔍 Executing {search_type} search with context: '{raw_query}'")

            # Call the existing restaurant_search_with_memory method with the accumulated query
            return await self.restaurant_search_with_memory(
                query=raw_query,  # Use accumulated query instead of single message
                user_id=user_id,
                gps_coordinates=gps_coordinates,
                thread_id=thread_id
            )

        except Exception as e:
            logger.error(f"❌ Error in context-aware search: {e}")
            return {
                "success": False,
                "error_message": f"Context-aware search failed: {str(e)}",
                "final_restaurants": [],
                "processing_time": 0
            }

    # Routing functions
    def _route_by_flow(self, state: UnifiedSearchState) -> str:
        return state.get("search_flow", "city_search")

    def _route_after_evaluation(self, state: UnifiedSearchState) -> str:
        """
        FIXED: Route after database evaluation

        Returns 'sufficient' for database-only, 'needs_search' for both hybrid and web-only.
        The distinction between hybrid and web-only is handled via state flags.
        """
        evaluation_results = state.get("evaluation_results")

        if not evaluation_results:
            logger.warning("⚠️ No evaluation results, defaulting to web search")
            return "needs_search"

        database_sufficient = evaluation_results.get("evaluation_result", {}).get("database_sufficient", False)
        is_hybrid = state.get("is_hybrid_mode", False)

        if database_sufficient:
            logger.info("✅ Database sufficient - routing to editor (database-only mode)")
            return "sufficient"

        if is_hybrid:
            logger.info("🔀 Hybrid mode - routing to web search (will merge with DB results)")
        else:
            logger.info("🌐 Database insufficient - routing to web search (web-only mode)")

        return "needs_search"

    def _route_after_filtering(self, state: UnifiedSearchState) -> str:
        """
        Route after filtering: ALWAYS go to human_decision for location searches

        CRITICAL CHANGE: We now ALWAYS pause at human_decision, even when database
        results are sufficient. This allows users to request more results via Maps.

        The human_decision node will:
        - If database results are good → show them, but graph stays paused for "more results"
        - If database results are poor → automatically trigger Maps search
        """
        filtered_results = state.get("filtered_results")

        if not filtered_results:
            logger.warning("⚠️ No filtered_results in state, routing to human decision")
            return "needs_enhancement"

        # Check database_sufficient flag from filter_evaluator
        database_sufficient = filtered_results.get("database_sufficient", False)
        filtered_restaurants = filtered_results.get("filtered_restaurants", [])

        if database_sufficient and len(filtered_restaurants) > 0:
            logger.info(f"✅ Database sufficient ({len(filtered_restaurants)} results) - routing to human decision (will show results and wait)")
        else:
            logger.info(f"🗺️ Database insufficient ({len(filtered_restaurants)} results) - routing to human decision (will trigger Maps)")

        # ALWAYS go to human_decision - this is the key fix!
        return "needs_enhancement"

    def _route_after_geocode(self, state: UnifiedSearchState) -> str:
        """
        Route after geocoding - check if this is a follow-up Maps request

        Logic:
        - skip_database=True → maps_direct (follow-up request, bypass database)
        - skip_database=False → database_search (normal flow)
        """
        skip_database = state.get("skip_database", False)

        if skip_database:
            logger.info("🗺️ Skipping database - direct to Google Maps (follow-up request)")
            return "maps_direct"
        else:
            logger.info("🗃️ Normal flow - searching database first")
            return "database_search"

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
        """
        FIXED: AI-powered flow detection using LocationAnalyzer
        UPDATED: Skip detection when resuming (search_flow already set)

        This method determines whether to route to:
        - city_search: City-wide search pipeline
        - location_search: GPS/location-based search pipeline

        On resume (when user says "let's find more"), it preserves the original flow
        instead of re-detecting, ensuring location searches stay in location pipeline.
        """
        try:
            # ====================================================================
            # NEW: Check if we're resuming - search_flow already set in state
            # ====================================================================
            existing_flow = state.get("search_flow")
            if existing_flow and existing_flow in ["city_search", "location_search"]:
                logger.info(f"🔄 Resuming with existing flow: {existing_flow} (skipping detection)")
                return {**state, "current_step": "flow_detected"}

            # ====================================================================
            # NEW SEARCH: Detect flow
            # ====================================================================
            logger.info("🔍 Flow Detection (new search)")

            has_coordinates = bool(state.get("gps_coordinates"))
            has_location_data = bool(state.get("location_data"))

            # If we have coordinates or location data, it's definitely location search
            if has_coordinates or has_location_data:
                search_flow = "location_search"
                logger.info("🗺️ Detected: Location-based search (GPS/location data provided)")
            else:
                # Use AI-powered LocationAnalyzer instead of hardcoded keywords
                query = state["query"]
                logger.info(f"🤖 Using AI to analyze query: '{query}'")

                try:
                    location_analysis = self.location_analyzer.analyze_message(query)
                    request_type = location_analysis.get("request_type", "GENERAL_SEARCH")
                    confidence = location_analysis.get("confidence", 0.0)
                    location_detected = location_analysis.get("location_detected")
                    reasoning = location_analysis.get("reasoning", "No reasoning provided")

                    logger.info("🤖 AI Analysis Result:")
                    logger.info(f"   Request Type: {request_type}")
                    logger.info(f"   Location Detected: {location_detected}")
                    logger.info(f"   Confidence: {confidence}")
                    logger.info(f"   Reasoning: {reasoning}")

                    if request_type == "LOCATION_SEARCH":
                        search_flow = "location_search"
                        logger.info("🗺️ Detected: Location-based search (AI analysis)")
                    else:
                        search_flow = "city_search"
                        logger.info("🏙️ Detected: City-based search (AI analysis)")

                except Exception as ai_error:
                    logger.error(f"❌ AI flow detection failed: {ai_error}")
                    # Fallback to city search if AI fails
                    search_flow = "city_search"
                    logger.info("🏙️ Detected: City-based search (AI fallback)")

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
                query_analysis=query_analysis
            )

            return {**state, "database_results": db_results, "current_step": "database_searched"}
        except Exception as e:
            logger.error(f"❌ Error in city database search: {e}")
            return {**state, "error_message": f"Database search failed: {str(e)}", "success": False}

    @traceable(name="city_evaluate_content")
    async def _city_evaluate_content(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """
        CRITICAL FIX: Evaluate database content and preserve hybrid restaurants in state

        The ContentEvaluationAgent already implements the 3-mode logic correctly.
        We just need to extract and preserve the data in state properly.
        """
        try:
            logger.info("⚖️ City Content Evaluation")

            database_results = state.get("database_results")
            query_analysis = state.get("query_analysis")
            destination = state.get("destination", "Unknown")

            # Type guard: Check database_results exists
            if not database_results:
                raise ValueError("No database results available")

            # Extract database_restaurants with proper type checking
            if isinstance(database_results, dict):
                database_restaurants = database_results.get("database_restaurants", [])
            elif isinstance(database_results, list):
                database_restaurants = database_results
            else:
                logger.error(f"Unexpected database_results type: {type(database_results)}")
                database_restaurants = []

            # Log what we received for debugging
            logger.info(f"📊 Database restaurants extracted: {len(database_restaurants)}")
            if not database_restaurants:
                logger.warning("⚠️ No database_restaurants found in database_results. "
                             f"Available keys: " 
                             f"{list(database_results.keys()) if isinstance(database_results, dict) else 'N/A'}")

            pipeline_data = {
                "database_restaurants": database_restaurants,
                "query_analysis": query_analysis,
                "destination": destination,
                "raw_query": state["query"]
            }

            evaluation_results = await sync_to_async(self.dbcontent_evaluation_agent.evaluate_and_route)(
                pipeline_data=pipeline_data
            )

            # CRITICAL FIX: Extract hybrid mode data from evaluation
            is_hybrid = evaluation_results.get("evaluation_result", {}).get("hybrid_mode", False)
            database_restaurants_hybrid = evaluation_results.get("database_restaurants_hybrid", [])

            logger.info("🔍 Evaluation complete:")
            logger.info(f"   - Database sufficient: {evaluation_results.get('evaluation_result', {}).get('database_sufficient', False)}")
            logger.info(f"   - Hybrid mode: {is_hybrid}")
            logger.info(f"   - Hybrid restaurants preserved: {len(database_restaurants_hybrid)}")

            return {
                **state,
                "evaluation_results": evaluation_results,
                "is_hybrid_mode": is_hybrid,  # NEW: Top-level flag
                "database_restaurants_hybrid": database_restaurants_hybrid,  # NEW: Top-level array
                "current_step": "content_evaluated"
            }

        except Exception as e:
            logger.error(f"❌ Error in city content evaluation: {e}")
            return {**state, "error_message": f"Content evaluation failed: {str(e)}", "success": False}

    @traceable(name="city_web_search")
    async def _city_web_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """Execute web search"""
        try:
            logger.info("🌐 City Web Search")

            # Log if we're in hybrid mode
            if state.get("is_hybrid_mode"):
                hybrid_count = len(state.get("database_restaurants_hybrid", []))  # type: ignore[arg-type]
                logger.info(f"🔀 HYBRID MODE: Preserving {hybrid_count} database restaurants")

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

            logger.info(f"🔍 Web search complete: {len(search_results)} results")

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
        """
        CRITICAL FIX: Editor now receives proper restaurant lists based on mode

        - Database-only: database_restaurants from database_results
        - Hybrid: database_restaurants_hybrid + scraped_results
        - Web-only: scraped_results only
        """
        try:
            logger.info("✏️ City Content Editing")

            destination = state.get("destination")
            if not destination:
                raise ValueError("No destination available for editing")

            # CRITICAL FIX: Determine which database restaurants to use
            is_hybrid = state.get("is_hybrid_mode", False)
            evaluation_results = state.get("evaluation_results") or {}
            database_sufficient = evaluation_results.get("evaluation_result", {}).get("database_sufficient", False)

            if database_sufficient:
                # Database-only mode: use full database results
                database_results = state.get("database_results")
                database_restaurants = None
                if database_results:
                    if isinstance(database_results, dict):
                        database_restaurants = database_results.get("database_restaurants", [])
                    elif isinstance(database_results, list):
                        database_restaurants = database_results
                logger.info(f"📊 DATABASE-ONLY MODE: Using {len(database_restaurants or [])} database restaurants")

            elif is_hybrid:
                # Hybrid mode: use preserved hybrid restaurants
                database_restaurants = state.get("database_restaurants_hybrid", [])
                logger.info(f"🔀 HYBRID MODE: Using {len(database_restaurants)} preserved database restaurants")  # type: ignore[arg-type]

            else:
                # Web-only mode: no database restaurants
                database_restaurants = None
                logger.info("🌐 WEB-ONLY MODE: No database restaurants")

            scraped_results = state.get("scraped_results")
            cleaned_file_path = state.get("cleaned_file_path")

            logger.info("📊 Sending to editor:")
            logger.info(f"   - Database restaurants: {len(database_restaurants) if database_restaurants else 0}")
            logger.info(f"   - Scraped results: {len(scraped_results) if scraped_results else 0}")
            logger.info(f"   - Cleaned file: {cleaned_file_path}")

            edited_results = await sync_to_async(self.editor_agent.edit)(
                destination=destination,
                database_restaurants=database_restaurants,
                scraped_results=scraped_results,
                cleaned_file_path=cleaned_file_path,
                raw_query=state["query"]
            )

            final_restaurants = edited_results.get("edited_results", {}).get("main_list", [])
            logger.info(f"✅ Editing complete: {len(final_restaurants)} restaurants")

            return {
                **state,
                "edited_results": edited_results,
                "final_restaurants": final_restaurants,
                "current_step": "content_edited"
            }

        except Exception as e:
            logger.error(f"❌ Error in city content editing: {e}")
            return {**state, "error_message": f"Content editing failed: {str(e)}", "success": False}

    @traceable(name="city_follow_up_search")
    async def _city_follow_up_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """
        Perform follow-up Google Maps verification and enhancement

        This integrates the existing FollowUpSearchAgent to:
        - Verify all restaurant addresses using Google Maps API
        - Add coordinates and proper Google Maps URLs with place_id format
        - Filter out low-rated or closed restaurants
        - Extract countries from formatted addresses
        - Update database with coordinates and corrected data
        """
        try:
            logger.info("🔍 City Follow-up Search (Google Maps verification)")

            edited_results = state.get("edited_results")
            destination = state.get("destination", "Unknown")

            if not edited_results:
                logger.warning("No edited results available for follow-up search")
                return {**state, "current_step": "follow_up_completed"}

            # Extract the actual restaurant data from the edited_results structure
            restaurants_data = edited_results
            if isinstance(edited_results, dict):
                if "edited_results" in edited_results:
                    restaurants_data = edited_results["edited_results"]
                elif "main_list" in edited_results:
                    restaurants_data = {"main_list": edited_results["main_list"]}

            logger.info(f"📊 Processing {len(restaurants_data.get('main_list', []))} restaurants for follow-up verification")

            # Perform the follow-up search using existing agent
            # This will verify addresses, add Google Maps URLs, filter low ratings, etc.
            enhanced_results = await sync_to_async(self.follow_up_search_agent.enhance)(
                edited_results=restaurants_data,
                follow_up_queries=None,  # Not needed for Google Maps verification
                destination=destination
            )

            logger.info("✅ Follow-up search completed - restaurants now have Google Maps URLs and verified data")

            # Update the state with enhanced results
            # Maintain the same structure as before but with enhanced restaurant data
            updated_state = {**state}
            if isinstance(edited_results, dict) and "edited_results" in edited_results:
                updated_state["edited_results"] = {
                    **edited_results,
                    "edited_results": enhanced_results
                }
            else:
                updated_state["edited_results"] = enhanced_results

            updated_state["current_step"] = "follow_up_completed"

            return updated_state

        except Exception as e:
            logger.error(f"❌ Error in city follow-up search: {e}")
            # Continue with unenhanced results rather than failing
            logger.warning("⚠️ Continuing with unenhanced results due to follow-up search error")
            return {
                **state, 
                "current_step": "follow_up_completed",
                "error_message": f"Follow-up search failed but continuing: {str(e)}"
            }
    
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
        """
        FIXED: Location geocoding with proper description handling

        Priority order:
        1. GPS coordinates from state
        2. Existing coordinates in location_data (if not None)
        3. Geocode location_data.description
        4. Use LocationAnalyzer to extract location from query, then geocode
        """
        try:
            logger.info("🗺️ Location Geocoding")

            location_data = state.get("location_data")
            gps_coords = state.get("gps_coordinates")
            coordinates = None

            # Priority 1: Use GPS coordinates if provided
            if gps_coords:
                coordinates = gps_coords
                logger.info(f"✅ Using GPS coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f}")

            # Priority 2: Check location_data for existing coordinates (verify not None)
            elif location_data:
                if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                    if location_data.latitude is not None and location_data.longitude is not None:
                        coordinates = (location_data.latitude, location_data.longitude)
                        logger.info(f"✅ Using location_data coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f}")

                    # Priority 3: Geocode from location_data.description if coordinates are None
                    elif hasattr(location_data, 'description') and location_data.description:
                        logger.info(f"🌍 Geocoding location from description: {location_data.description}")
                        geocoded_coords = self.location_utils.geocode_location(location_data.description)

                        if geocoded_coords:
                            coordinates = geocoded_coords
                            # Update location_data with geocoded coordinates
                            location_data.latitude = coordinates[0]
                            location_data.longitude = coordinates[1]
                            logger.info(f"✅ Geocoded to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                        else:
                            logger.warning(f"❌ Failed to geocode description: {location_data.description}")

            # Priority 4: Extract location from query using AI with context enrichment
            if not coordinates:
                logger.info(f"🤖 Extracting location from query using AI: {state['query']}")

                try:
                    # NEW: Get stored city context from AI Chat Layer if available
                    stored_city = None
                    if hasattr(self, 'ai_chat_layer') and self.ai_chat_layer:
                        user_id = state.get('user_id')
                        if user_id:
                            session_info = self.ai_chat_layer.get_session_info(user_id)
                            if session_info:
                                stored_city = session_info.get('last_searched_city')
                                if stored_city:
                                    logger.info(f"   📍 Using stored city context for enrichment: {stored_city}")

                    # Use LocationAnalyzer with context enrichment
                    analysis_result = self.location_analyzer.analyze_message(
                        state['query'],
                        stored_city_context=stored_city  # Pass stored city!
                    )
                    location_detected = analysis_result.get("location_detected", "")

                    # Log if context enrichment was applied
                    if analysis_result.get("context_enrichment_applied"):
                        logger.info(f"   ✨ Location enriched: {location_detected}")

                    if location_detected:
                        logger.info(f"🌍 AI extracted location: '{location_detected}', attempting to geocode")
                        coordinates = self.location_utils.geocode_location(location_detected)

                        if coordinates:
                            logger.info(f"✅ Geocoded extracted location to: {coordinates[0]:.4f}, {coordinates[1]:.4f}")

                            # Update location_data if it exists
                            if location_data:
                                location_data.latitude = coordinates[0]
                                location_data.longitude = coordinates[1]
                                if not hasattr(location_data, 'description') or not location_data.description:
                                    location_data.description = location_detected
                        else:
                            raise ValueError(f"Failed to geocode extracted location: {location_detected}")
                    else:
                        raise ValueError(f"Could not extract location from query: {state['query']}")

                except Exception as extract_error:
                    logger.error(f"❌ Location extraction/geocoding failed: {extract_error}")
                    raise ValueError(f"Location geocoding failed: {str(extract_error)}")

            # Final validation
            if not coordinates:
                raise ValueError("Could not obtain coordinates from any source")

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
        """CORRECTED: Handle proximity_results as a LIST, not a dict"""
        try:
            logger.info("🔍 Location Results Filtering")

            proximity_results = state.get("proximity_results")

            # ✅ FIX: proximity_results is a LIST of restaurants, not a dict!
            # The location database service returns a list directly
            if not proximity_results:
                logger.info("📊 No proximity results to filter")
                return {
                    **state,
                    "filtered_results": {"enhancement_needed": True, "restaurants": []},
                    "current_step": "location_filtered"
                }

            # ✅ FIX: Treat proximity_results as a list directly
            if not isinstance(proximity_results, list):
                logger.error(f"❌ Invalid proximity results format: expected list, got {type(proximity_results)}")
                return {
                    **state,
                    "filtered_results": {"enhancement_needed": True, "restaurants": []},
                    "current_step": "location_filtered"
                }

            restaurants = proximity_results  # It's already the list of restaurants!

            logger.info(f"📊 Location restaurants extracted: {len(restaurants)}")

            query = state.get("query", "restaurant")
            location_description = f"Location search: {query}"

            # Now pass the actual list to the filter evaluator
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
        """
        Human decision handler - ENHANCED to handle database-sufficient results

        NEW BEHAVIOR:
        - If database results are sufficient: Set flag to show them, but stay paused
        - If database results are insufficient: Automatically accept and trigger Maps
        """
        try:
            logger.info("🤔 Location Human Decision")

            # Check if we have sufficient database results
            filtered_results = state.get("filtered_results", {})
            database_sufficient = filtered_results.get("database_sufficient", False)
            filtered_restaurants = filtered_results.get("filtered_restaurants", [])

            if database_sufficient and len(filtered_restaurants) > 0:
                # We have good database results - show them but PAUSE for potential "more results" request
                logger.info(f"✅ Database results are sufficient - showing {len(filtered_restaurants)} results and pausing")
                logger.info("   User can say 'let's find more' to trigger Google Maps search")

                return {
                    **state,
                    "human_decision_pending": True,  # PAUSE here!
                    "database_results_shown": True,  # Flag that we showed database results
                    "current_step": "human_decision_pending"
                }
            else:
                # Database results insufficient - automatically trigger Maps without pausing
                logger.info("🗺️ Database insufficient - automatically triggering Maps search (no pause)")

                return {
                    **state,
                    "human_decision_pending": False,  # Don't pause - go straight to Maps
                    "human_decision_result": "accept",  # Auto-accept Maps search
                    "database_results_shown": False,
                    "current_step": "auto_triggering_maps"
                }

        except Exception as e:
            logger.error(f"❌ Error in location human decision: {e}")
            return {**state, "error_message": f"Human decision setup failed: {str(e)}", "success": False}


    @traceable(name="location_maps_search")
    def _location_maps_search(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """FIXED: Properly handle async search_venues_with_ai_analysis()"""
        try:
            logger.info("🗺️ Location Maps Search")

            coordinates = state.get("location_coordinates")
            if not coordinates:
                raise ValueError("No coordinates available for maps search")

            query = state.get("query", "restaurant")

            # FIXED: Run async method using asyncio.run()
            maps_results = asyncio.run(
                self.location_map_search_agent.search_venues_with_ai_analysis(
                    coordinates=coordinates,
                    query=query
                )
            )

            return {**state, "maps_results": maps_results, "current_step": "location_maps_searched"}
        except Exception as e:
            logger.error(f"❌ Error in location maps search: {e}")
            return {**state, "error_message": f"Location maps search failed: {str(e)}", "success": False}

    @traceable(name="location_media_verification")
    def _location_media_verification(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """FIXED: Properly handle async verify_venues_media_coverage()"""
        try:
            logger.info("📱 Location Media Verification")

            maps_results = state.get("maps_results")
            if not maps_results:
                raise ValueError("No maps results available for verification")

            # FIXED: maps_results is now a list of VenueSearchResult objects, not a dict
            # The async method returns a list directly
            venues = maps_results if isinstance(maps_results, list) else []
            query = state.get("query", "")

            if not venues:
                logger.warning("⚠️ No venues to verify")
                return {
                    **state, 
                    "media_verification_results": [], 
                    "current_step": "location_media_verified"
                }

            # FIXED: Run async method using asyncio.run()
            verification_results = asyncio.run(
                self.location_media_verification_agent.verify_venues_media_coverage(
                    venues=venues,
                    query=query
                )
            )

            return {**state, "media_verification_results": verification_results, "current_step": "location_media_verified"}
        except Exception as e:
            logger.error(f"❌ Error in location media verification: {e}")
            return {**state, "error_message": f"Location media verification failed: {str(e)}", "success": False}


    @traceable(name="location_format_results")
    async def _location_format_results(self, state: UnifiedSearchState) -> Dict[str, Any]:
        """
        Format final location results - ENHANCED to show database results first if available

        NEW: If we have database_results_shown flag, format database results and
        indicate to user they can request more via Maps
        """
        try:
            logger.info("📊 Location Format Results")

            # Check if we're showing database results (before any Maps search)
            database_results_shown = state.get("database_results_shown", False)
            filtered_results = state.get("filtered_results", {})

            if database_results_shown:
                # Format database results with "more results available" hint
                logger.info("📋 Formatting DATABASE results (user can request Maps search)")

                filtered_restaurants = filtered_results.get("filtered_restaurants", [])
                query = state.get("query", "restaurants")

                formatted = self.location_formatter.format_database_results(
                    restaurants=filtered_restaurants,  # Changed from venues to restaurants
                    query=query,
                    location_description=f"GPS search: {query}",
                    offer_more_search=True  # Changed from show_more_option to offer_more_search
                )

                return {
                    **state,
                    "formatted_message": formatted.get("message", ""),
                    "final_restaurants": filtered_restaurants,
                    "success": True,
                    "current_step": "database_results_formatted"  # Different from final END state
                }

            # Otherwise, format Maps results (after user requested more)
            logger.info("📋 Formatting MAPS results (user requested more options)")

            maps_results = state.get("maps_results", {})
            media_results = state.get("media_verification_results", {})

            # Use Maps results if available, otherwise fallback
            if maps_results and maps_results.get("restaurants"):
                restaurants = maps_results["restaurants"]
            elif media_results and media_results.get("restaurants"):
                restaurants = media_results["restaurants"]
            elif filtered_results and filtered_results.get("filtered_restaurants"):
                restaurants = filtered_results["filtered_restaurants"]
            else:
                return {
                    **state,
                    "formatted_message": "😔 No results found.",
                    "final_restaurants": [],
                    "success": False,
                    "current_step": "location_results_formatted"
                }

            query = state.get("query", "restaurants")

            formatted = self.location_formatter.format_google_maps_results(
                venues=restaurants,
                query=query,
                location_description=f"GPS search: {query}"
            )

            return {
                **state,
                "formatted_message": formatted.get("message", ""),
                "final_restaurants": restaurants,
                "success": True,
                "current_step": "location_results_formatted"  # Final END state
            }

        except Exception as e:
            logger.error(f"❌ Error in location formatting: {e}")
            return {**state, "error_message": f"Location formatting failed: {str(e)}", "success": False}


    # ============================================================================
    # PUBLIC API
    # ============================================================================

    """
    Complete search_restaurants method for langgraph_orchestrator.py
    With structured handoff support
    """

    async def search_restaurants(
        self,
        query: str,
        user_id: Optional[int] = None,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        location_data: Optional[Any] = None,
        thread_id: Optional[str] = None,
        search_context: Optional[SearchContext] = None  # NEW: Structured context
    ) -> Dict[str, Any]:
        """
        Unified search method with structured handoff support

        Args:
            query: Search query string
            user_id: User ID
            gps_coordinates: GPS coordinates if available
            location_data: LocationData object if available
            thread_id: Thread ID for state management
            search_context: NEW - Structured SearchContext from AI Chat Layer

        Returns:
            Dict with search results, restaurants, formatted message
        """

        start_time = time.time()

        self._current_search_context = search_context

        try:
            logger.info(f"🚀 UNIFIED SEARCH: '{query}' (user: {user_id})")

            # Log structured context if available
            if search_context:
                logger.info("📦 Using structured context:")
                logger.info(f"   Destination: {search_context.destination}")
                logger.info(f"   Cuisine: {search_context.cuisine}")
                logger.info(f"   Type: {search_context.search_type.value}")
                logger.info(f"   Clear previous: {search_context.clear_previous_context}")
                logger.info(f"   New destination: {search_context.is_new_destination}")

            # Check for location button requirement
            if self._detect_needs_location_button(query):
                if not gps_coordinates and not location_data:
                    logger.info(f"🔘 Query needs location but none provided: '{query}'")
                    return {
                        "success": False,
                        "needs_location_button": True,
                        "action": "REQUEST_LOCATION",
                        "query": query,
                        "message": "I'd love to help you find great restaurants nearby! Please share your location.",
                        "processing_time": round(time.time() - start_time, 2)
                    }

            # Initialize state
            initial_state: UnifiedSearchState = {
                "query": query,
                "raw_query": query,
                "user_id": user_id,
                "gps_coordinates": gps_coordinates,
                "location_data": location_data,
                "search_context": search_context,  # Pass structured context
                "search_flow": "",
                "current_step": "initialized",
                "query_analysis": None,
                "destination": None,
                "database_results": None,
                "evaluation_results": None,
                "search_results": None,
                "scraped_results": None,
                "cleaned_file_path": None,
                "edited_results": None,
                "database_restaurants_hybrid": [],  # CRITICAL FIX
                "is_hybrid_mode": False,  # CRITICAL FIX
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

            # Generate thread ID if needed
            if not thread_id:
                thread_id = f"search_{user_id}_{int(time.time())}"

            # Configure LangGraph
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            # Execute graph
            result = await self.graph.ainvoke(initial_state, config)

            # Add processing time
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time

            # CRITICAL FIX: Map formatted_message to langchain_formatted_results for BOTH flows
            # The telegram bot expects 'langchain_formatted_results', but the state has 'formatted_message'
            if result.get("formatted_message") and not result.get("langchain_formatted_results"):
                result["langchain_formatted_results"] = result["formatted_message"]
                logger.info(f"✅ Mapped formatted_message to langchain_formatted_results ({len(result['formatted_message'])} chars)")

            logger.info(f"✅ UNIFIED SEARCH COMPLETE: {processing_time}s")
            return result

        except Exception as e:
            logger.error(f"❌ Error in unified search: {e}", exc_info=True)
            return {
                "success": False,
                "error_message": str(e),
                "final_restaurants": [],
                "processing_time": round(time.time() - start_time, 2)
            }

        finally:
            self._current_search_context = None

    async def restaurant_search_with_memory(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        **kwargs  # NEW: To accept telegram_bot and chat_id
    ) -> Dict[str, Any]:
        """
        CRITICAL FIX: Main entry point now ALWAYS routes through AI Chat Layer
        """
        start_time = time.time()
        
        try:

            if not thread_id:
                thread_id = f"search_{user_id}_{int(time.time())}"

            # CRITICAL: Check if this is from AI Chat Layer
            is_from_ai_chat = query.startswith('[CONTEXT]')

            # FORCE AI Chat Layer routing for ALL new messages
            if not is_from_ai_chat:
                # This is a new user message - MUST route through AI Chat Layer first
                logger.info(f"🔄 FORCING route to AI Chat Layer for conversation management: user {user_id}")

                # Initialize AI Chat Layer if not exists (CRITICAL)
                if not hasattr(self, 'ai_chat_layer'):
                    from utils.ai_chat_layer import AIChatLayer
                    self.ai_chat_layer = AIChatLayer(self.config)
                    logger.info("✅ AI Chat Layer initialized on-demand")

                return await self.process_user_message_with_ai_chat(
                    query=query,
                    user_id=user_id,
                    gps_coordinates=gps_coordinates,
                    thread_id=thread_id,
                    telegram_bot=kwargs.get('telegram_bot'),  # NEW
                    chat_id=kwargs.get('chat_id')  # NEW
                )

            # If we reach here, AI Chat Layer determined we're ready to search
            query = query[9:]  # Remove '[CONTEXT]' prefix
            logger.info(f"🚀 Executing search with AI Chat context for user {user_id}: '{query[:50]}...'")

            # 1. Learn from user query
            await self.learn_from_user_query(user_id, query)

            # 2. Update conversation state
            await self.update_conversation_state(user_id, thread_id, "searching")

            # 3. Get personalized context
            memory_context = await self.get_personalized_context(user_id, thread_id)

            # 4. Perform the actual search using existing unified search
            search_result = await self.search_restaurants(
                query=query,
                user_id=user_id,
                gps_coordinates=gps_coordinates,
                thread_id=thread_id
            )

            # 5. If search was successful, process results with memory
            if search_result.get("success") and search_result.get("final_restaurants"):
                restaurants = search_result["final_restaurants"]

                # Extract city from query analysis or use a default
                city = search_result.get("destination", "Unknown")

                # Filter out already recommended restaurants
                filtered_restaurants = await self.filter_already_recommended(
                    user_id, restaurants, city
                )

                # Store new recommendations in memory
                if filtered_restaurants:
                    await self.store_search_results_in_memory(
                        user_id, filtered_restaurants, query, city
                    )

                    # Update the result with filtered restaurants
                    search_result["final_restaurants"] = filtered_restaurants
                    search_result["restaurants_filtered"] = len(restaurants) - len(filtered_restaurants)

                    # Generate memory-aware response
                    memory_response = self._generate_memory_aware_response(
                        filtered_restaurants, memory_context, query, city
                    )

                    # Combine with formatted message if it exists
                    formatted_message = search_result.get("formatted_message", "")
                    if formatted_message:
                        search_result["ai_response"] = memory_response + formatted_message
                    else:
                        search_result["ai_response"] = memory_response

                else:
                    # All restaurants were already recommended
                    search_result["ai_response"] = self._generate_already_recommended_response(
                        memory_context, query, city
                    )
                    search_result["final_restaurants"] = []
                    search_result["restaurants_filtered"] = len(restaurants)

            elif search_result.get("success") and not search_result.get("final_restaurants"):
                # Search succeeded but no restaurants found
                search_result["ai_response"] = f"I couldn't find any restaurants matching '{query}'. Would you like to try a different search or expand your criteria?"

            else:
                # Search failed
                error_msg = search_result.get("error_message", "Search encountered an error")
                search_result["ai_response"] = f"I'm having trouble finding restaurants right now. {error_msg}. Could you try again?"

            # 6. Update conversation state
            await self.update_conversation_state(user_id, thread_id, "presenting_results")

            # 7. Add processing time and context info
            processing_time = round(time.time() - start_time, 2)
            search_result["processing_time"] = processing_time
            search_result["memory_enhanced"] = True
            search_result["user_id"] = user_id

            # Add context flags for debugging
            search_result["routed_through_ai_chat"] = is_from_ai_chat
            search_result["has_memory_context"] = bool(memory_context)

            logger.info(f"✅ Memory-enhanced search completed in {processing_time}s")
            return search_result

        except Exception as e:
            logger.error(f"❌ Error in memory-enhanced restaurant search: {e}")
            processing_time = round(time.time() - start_time, 2) if 'start_time' in locals() else 0

            return {
                "success": False,
                "error_message": f"Search failed: {str(e)}",
                "final_restaurants": [],
                "ai_response": "I'm having trouble finding restaurants right now. Could you try again?",
                "processing_time": processing_time,
                "memory_enhanced": True,
                "user_id": user_id,
                "routed_through_ai_chat": False,
                "has_memory_context": False
            }

    def _generate_memory_aware_response(
        self, 
        restaurants: List[Dict[str, Any]], 
        memory_context: Dict[str, Any], 
        query: str, 
        city: str
    ) -> str:
        """Generate a personalized response based on user's memory"""
        try:
            preferences = memory_context.get('preferences', {})
            restaurant_count = len(restaurants)

            # Personalized greeting based on memory
            if preferences.get('preferred_cities') and city in preferences['preferred_cities']:
                intro = f"Great choice! I remember you love {city}. "
            else:
                intro = "Perfect! "

            intro += f"I found {restaurant_count} amazing {'spot' if restaurant_count == 1 else 'spots'} for you"

            # Add cuisine preference reference if relevant
            user_cuisines = preferences.get('preferred_cuisines', [])
            if user_cuisines:
                cuisine_match = any(cuisine.lower() in query.lower() for cuisine in user_cuisines)
                if cuisine_match:
                    intro += f" - I know you enjoy good {user_cuisines[0]} food"

            intro += "! 🍽️\n\n"

            # Use the existing formatted message but add personalized intro
            return intro + "Here are my top recommendations:\n\n"

        except Exception as e:
            logger.error(f"Error generating memory-aware response: {e}")
            return f"I found {len(restaurants)} great restaurants for you! 🍽️\n\n"

    def _generate_already_recommended_response(
        self, 
        memory_context: Dict[str, Any], 
        query: str, 
        city: str
    ) -> str:
        """Generate response when all restaurants were already recommended"""
        preferences = memory_context.get('preferences', {})

        response = f"I notice I've already recommended all the top spots for '{query}' in {city}! 🤔\n\n"

        # Suggest alternatives based on preferences
        user_cuisines = preferences.get('preferred_cuisines', [])
        if user_cuisines and len(user_cuisines) > 1:
            alt_cuisine = next((c for c in user_cuisines if c.lower() not in query.lower()), user_cuisines[0])
            response += f"Would you like to try some great {alt_cuisine} places instead? "
        else:
            response += "Would you like to explore a different cuisine or neighborhood? "

        response += "Just let me know what sounds good! ✨"

        return response

    async def handle_human_decision(self, thread_id: str, decision: str) -> Dict[str, Any]:
        """
        Handle human-in-the-loop decision - ASYNC version

        FIXED: Properly continues from paused checkpoint instead of restarting graph

        When resuming a location search with "let's find more", this now correctly
        continues from location_human_decision -> location_maps_search instead of
        looping back to location_geocode -> location_search_database.
        """
        try:
            logger.info(f"🤔 Human decision: {decision}")

            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            # Get the checkpoint to verify it exists
            checkpoint_tuple = self.checkpointer.get_tuple(config)

            if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
                logger.error("No conversation state found for thread_id")
                return {"success": False, "error_message": "No conversation found"}

            # Extract current state from checkpoint
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            current_state = dict(channel_values)

            # Update state with human decision
            current_state["human_decision_result"] = decision
            current_state["human_decision_pending"] = False

            logger.info(f"📍 Current step before resume: {current_state.get('current_step')}")
            logger.info(f"🔄 Resuming graph from checkpoint with decision: {decision}")

            # CRITICAL FIX: Use astream() with None input to continue from checkpoint
            # This tells LangGraph to continue from the last paused node instead of restarting
            final_state = None
            async for event in self.graph.astream(None, config, stream_mode="values"):
                final_state = event
                logger.debug(f"📊 Stream event - current_step: {event.get('current_step')}")

            if final_state:
                logger.info(f"✅ Graph completed at step: {final_state.get('current_step')}")
                return final_state
            else:
                logger.error("No final state from graph stream")
                return {"success": False, "error_message": "Graph execution failed"}

        except Exception as e:
            logger.error(f"❌ Error handling human decision: {e}", exc_info=True)
            return {"success": False, "error_message": f"Decision handling failed: {str(e)}"}

    async def filter_already_recommended(
        self, 
        user_id: int, 
        restaurants: List[Dict[str, Any]], 
        city: str
    ) -> List[Dict[str, Any]]:
        """
        Filter out restaurants that have already been recommended to the user

        FIXED: Delegates to AIMemorySystem.filter_already_recommended()
        """
        try:
            # Delegate to the memory system's built-in filtering
            return await self.memory_system.filter_already_recommended(
                user_id=user_id,
                restaurants=restaurants,
                city=city
            )

        except Exception as e:
            logger.error(f"Error filtering already recommended restaurants: {e}")
            return restaurants  # Return original list if filtering fails

    # FINAL CORRECTED CODE - store_search_results_in_memory method
    
    async def store_search_results_in_memory(
        self,
        user_id: int,
        restaurants: List[Dict[str, Any]],
        query: str,
        city: str
    ) -> bool:
        """
        Store search results in memory using AI-extracted preferences

        FIXED: All parameter errors, undefined variables, and type issues resolved
        """
        try:
            if not restaurants:
                logger.info(f"No restaurants to store for user {user_id}")
                return True

            logger.info(f"💾 Storing {len(restaurants)} restaurants for user {user_id}")

            # Step 1: Store each restaurant via AIMemorySystem
            stored_count = 0
            for restaurant in restaurants:
                try:
                    # FIXED: Convert cuisine list to string
                    cuisine = restaurant.get('cuisine', [])
                    if isinstance(cuisine, list):
                        cuisine_str = ', '.join(cuisine) if cuisine else 'Unknown'
                    else:
                        cuisine_str = str(cuisine) if cuisine else 'Unknown'

                    # FIXED: Create notes with all context info (BEFORE RestaurantMemory instantiation)
                    notes_parts = [f"Query: {query}"]

                    # Safe subscripting - get values first, check, then use
                    description = restaurant.get('description')
                    if description:
                        notes_parts.append(f"Description: {description[:200]}")

                    address = restaurant.get('address')
                    if address:
                        notes_parts.append(f"Address: {address}")

                    url = restaurant.get('url')
                    if url:
                        notes_parts.append(f"URL: {url}")

                    notes_str = ' | '.join(notes_parts)

                    # FIXED: All correct parameters for RestaurantMemory
                    restaurant_memory = RestaurantMemory(
                        restaurant_name=restaurant.get('name', 'Unknown'),
                        city=city,
                        cuisine=cuisine_str,                                      # ✅ String, not list
                        recommended_date=datetime.now(timezone.utc).isoformat(),  # ✅ ISO string format
                        user_feedback=None,                                       # ✅ Required parameter
                        rating_given=restaurant.get('rating'),                    # ✅ Correct parameter name
                        notes=notes_str,                                          # ✅ Required parameter with context
                        source=restaurant.get('source', 'web')                    # ✅ Correct parameter
                    )

                    # Delegate to AIMemorySystem
                    success = await self.memory_system.add_restaurant_memory(
                        user_id, restaurant_memory
                    )

                    if success:
                        stored_count += 1

                except Exception as e:
                    logger.error(f"Error storing restaurant {restaurant.get('name')}: {e}")
                    continue

            logger.info(f"✅ Stored {stored_count}/{len(restaurants)} restaurants")

            # Step 2: Learn preferences
            if not getattr(self.config, 'AUTO_UPDATE_USER_PREFERENCES', True):
                return stored_count > 0

            # Check if we have AI-extracted preferences from SearchContext
            extracted_cuisine = None
            extracted_requirements = []
            extracted_preferences = {}

            # Try to get SearchContext from current processing
            if hasattr(self, '_current_search_context') and self._current_search_context:
                ctx = self._current_search_context
                extracted_cuisine = ctx.cuisine
                extracted_requirements = ctx.requirements or []
                extracted_preferences = ctx.preferences or {}

                logger.info("📦 Using AI-extracted preferences from SearchContext:")
                logger.info(f"   Cuisine: {extracted_cuisine}")
                logger.info(f"   Requirements: {extracted_requirements}")
                logger.info(f"   Preferences: {extracted_preferences}")

            # Fallback: Extract cuisine from restaurant list if AI didn't extract it
            if not extracted_cuisine:
                cuisine_list = []
                for restaurant in restaurants[:3]:
                    if restaurant.get('cuisine'):
                        if isinstance(restaurant['cuisine'], list):
                            cuisine_list.extend(restaurant['cuisine'])
                        else:
                            cuisine_list.append(restaurant['cuisine'])

                if cuisine_list:
                    from collections import Counter
                    cuisine_counts = Counter(cuisine_list)
                    extracted_cuisine = cuisine_counts.most_common(1)[0][0] if cuisine_counts else None
                    logger.info(f"📊 Extracted cuisine from restaurants: {extracted_cuisine}")

            # Step 3: Delegate preference learning to AIMemorySystem
            preference_learned = await self.memory_system.learn_preferences_from_message(
                user_id=user_id,
                message=query,
                current_city=city,
                extracted_cuisine=extracted_cuisine,
                extracted_requirements=extracted_requirements,
                extracted_preferences=extracted_preferences
            )

            if preference_learned:
                logger.info(f"✅ Preferences learned for user {user_id}")

            # Step 4: Delegate conversation pattern learning to AIMemorySystem
            if getattr(self.config, 'LEARN_CONVERSATION_PATTERNS', True):
                await self.memory_system.learn_conversation_patterns(
                    user_id=user_id,
                    message=query
                )

            return True

        except Exception as e:
            logger.error(f"❌ Error storing results in memory: {e}")
            return False

    async def learn_from_user_query(self, user_id: int, query: str, city: Optional[str] = None) -> bool:
        """Learn user preferences from their search query"""
        try:
            return await self.memory_system.learn_preferences_from_message(
                user_id, query, city or ""
            )
        except Exception as e:
            logger.error(f"Error learning from user query: {e}")
            return False

    async def get_user_memory_summary(self, user_id: int) -> Dict[str, Any]:
        """Get a summary of user's memory for debugging/display"""
        try:
            memory_context = await self.memory_system.get_user_context(user_id, "summary")

            preferences = memory_context.get('preferences', {})
            restaurant_history = memory_context.get('restaurant_history', [])
            conversation_patterns = memory_context.get('conversation_patterns', {})

            return {
                "memory_summary": {
                    "total_restaurants": len(restaurant_history),
                    "preferred_cities": preferences.get('preferred_cities', []),
                    "preferred_cuisines": preferences.get('preferred_cuisines', []),
                    "current_city": memory_context.get('current_city'),
                    "conversation_style": conversation_patterns.get('user_communication_style', 'casual'),
                    "last_search_date": restaurant_history[-1].get('recommended_date') if restaurant_history else None
                }
            }

        except Exception as e:
            logger.error(f"Error getting user memory summary: {e}")
            return {"memory_summary": {}, "error": str(e)}

    async def update_conversation_state(
        self, 
        user_id: int, 
        thread_id: str, 
        state: str
    ) -> bool:
        """Update the current conversation state"""
        try:
            # Convert string to ConversationState enum if needed
            if isinstance(state, str):
                state_enum = ConversationState(state.lower())
            else:
                state_enum = state

            return await self.memory_system.set_session_state(user_id, thread_id, state_enum)

        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")
            return False

    async def get_personalized_context(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get personalized context for the user's current conversation"""
        try:
            return await self.memory_system.get_user_context(user_id, thread_id)
        except Exception as e:
            logger.error(f"Error getting personalized context: {e}")
            return {}

def create_unified_restaurant_agent(config):
    """Factory function to create the unified restaurant agent"""
    return UnifiedRestaurantAgent(config)