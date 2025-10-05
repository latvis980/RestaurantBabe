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

    async def process_user_message_with_ai_chat(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        telegram_bot=None,  # NEW: Accept telegram bot instance
        chat_id: Optional[int] = None  # NEW: Accept chat_id for sending confirmation
    ) -> Dict[str, Any]:
        """
        NEW: Main entry point with AI Chat Layer integration
        This method handles conversation flow and only triggers search when ready.

        ENHANCED: Now sends confirmation message with video when search is triggered
        """
        start_time = time.time()

        try:
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            logger.info(f"💬 Processing message with AI Chat Layer for user {user_id}: '{query[:50]}...'")

            # Import AI Chat Layer here to avoid circular imports
            from utils.ai_chat_layer import AIChatLayer, ActionType

            # Initialize AI Chat Layer if not exists
            if not hasattr(self, 'ai_chat_layer'):
                self.ai_chat_layer = AIChatLayer(self.config)

            # 1. Process message through AI Chat Layer
            chat_decision = await self.ai_chat_layer.process_message(
                user_id=user_id,
                user_message=query,
                gps_coordinates=gps_coordinates
            )

            logger.info(f"🎯 AI Chat Decision: {chat_decision.action.value} - {chat_decision.reasoning}")

            # 2. Handle different actions
            if chat_decision.action in [ActionType.CHAT_RESPONSE, ActionType.COLLECT_INFO, ActionType.CLARIFY_REQUEST]:
                # Continue conversation - no search needed
                processing_time = round(time.time() - start_time, 2)
                return {
                    "success": True,
                    "ai_response": chat_decision.response_text,
                    "action_taken": chat_decision.action.value,
                    "conversation_state": chat_decision.new_state.value if chat_decision.new_state else "unknown",
                    "search_triggered": False,
                    "processing_time": processing_time,
                    "reasoning": chat_decision.reasoning
                }

            elif chat_decision.action in [ActionType.TRIGGER_CITY_SEARCH, ActionType.TRIGGER_LOCATION_SEARCH]:
                # Ready to search! 
                logger.info(f"🚀 Triggering {chat_decision.search_type} search with accumulated context")

                # Get the full conversation context for search
                search_info = self.ai_chat_layer.get_search_ready_info(user_id)

                if search_info is None:
                    processing_time = round(time.time() - start_time, 2)
                    return {
                        "success": False,
                        "error_message": "No search context available",
                        "ai_response": "I need more information to help you find restaurants.",
                        "search_triggered": False,
                        "processing_time": processing_time
                    }

                # NEW: SEND CONFIRMATION MESSAGE WITH VIDEO BEFORE STARTING SEARCH
                confirmation_msg = None
                if telegram_bot and chat_id:
                    try:
                        # Determine search type
                        search_type = "location_based" if chat_decision.action == ActionType.TRIGGER_LOCATION_SEARCH else "city_wide"

                        # Get conversation context for message generation
                        conversation_context = search_info.get('raw_query', query)

                        # Send confirmation message with video
                        confirmation_msg = self._send_search_confirmation_message(
                            telegram_bot=telegram_bot,
                            chat_id=chat_id,
                            search_query=conversation_context,
                            search_type=search_type
                        )

                        logger.info(f"✅ Sent confirmation message for {search_type} search")

                    except Exception as e:
                        logger.warning(f"⚠️ Could not send confirmation message: {e}")

                search_type = chat_decision.search_type or "city_wide"

                # Execute restaurant search with accumulated context - use [CONTEXT] prefix
                search_result = await self.restaurant_search_with_memory(
                    query=f"[CONTEXT]{search_info.get('raw_query', query)}",
                    user_id=user_id,
                    gps_coordinates=gps_coordinates,
                    thread_id=thread_id
                )

                # Clean up confirmation message if it was sent
                if telegram_bot and chat_id and confirmation_msg:
                    try:
                        # Give search a moment to start
                        await asyncio.sleep(1)
                        telegram_bot.delete_message(chat_id, confirmation_msg.message_id)
                    except Exception:
                        pass  # Message might already be deleted

                # Add chat layer info to result
                search_result["ai_chat_response"] = chat_decision.response_text
                search_result["search_triggered"] = True
                search_result["conversation_context"] = search_info.get('raw_query', query)
                search_result["action_taken"] = chat_decision.action.value

                return search_result

            else:
                # Handle other actions
                processing_time = round(time.time() - start_time, 2)
                return {
                    "success": True,
                    "ai_response": chat_decision.response_text or "I'm here to help you find great restaurants!",
                    "action_taken": chat_decision.action.value,
                    "search_triggered": False,
                    "processing_time": processing_time,
                    "reasoning": chat_decision.reasoning
                }

        except Exception as e:
            logger.error(f"❌ Error in AI chat processing: {e}")
            processing_time = round(time.time() - start_time, 2)
            return {
                "success": False,
                "error_message": f"AI chat processing failed: {str(e)}",
                "ai_response": "I'm having a bit of trouble right now. Could you try asking again?",
                "search_triggered": False,
                "processing_time": processing_time
            }

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
            import asyncio
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            # Initialize AI if not exists
            if not hasattr(self, '_confirmation_ai'):
                self._confirmation_ai = ChatOpenAI(
                    model=getattr(self.config, 'AI_MESSAGE_MODEL', 'gpt-4o-mini'),
                    temperature=0.7,
                    max_tokens=80,  # Keep it short
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
                return response.content.strip()

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
                query_analysis=query_analysis
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

    async def store_search_results_in_memory(
        self, 
        user_id: int, 
        results: List[Dict[str, Any]], 
        query: str,
        city: str
    ) -> bool:
        """Store successful search results in user's memory"""
        try:
            for restaurant_data in results:
                restaurant_memory = RestaurantMemory(
                    restaurant_name=restaurant_data.get('name', 'Unknown'),
                    city=city,
                    cuisine=restaurant_data.get('cuisine', 'Unknown'),
                    recommended_date=datetime.now(timezone.utc).isoformat(),
                    user_feedback=None,
                    rating_given=None,
                    notes=f"Recommended for query: {query}",
                    source=restaurant_data.get('source', 'unified_search')
                )

                await self.memory_system.add_restaurant_memory(user_id, restaurant_memory)

            logger.info(f"Stored {len(results)} restaurants in memory for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing search results in memory: {e}")
            return False

    async def filter_already_recommended(
        self, 
        user_id: int, 
        restaurants: List[Dict[str, Any]], 
        city: str
    ) -> List[Dict[str, Any]]:
        """Filter out restaurants that have already been recommended to the user"""
        try:
            filtered_restaurants = []

            for restaurant in restaurants:
                restaurant_name = restaurant.get('name', '')
                if restaurant_name:
                    already_recommended = await self.memory_system.has_restaurant_been_recommended(
                        user_id, restaurant_name, city
                    )

                    if not already_recommended:
                        filtered_restaurants.append(restaurant)
                    else:
                        logger.info(f"Filtered out already recommended restaurant: {restaurant_name}")

            logger.info(f"Filtered {len(restaurants) - len(filtered_restaurants)} already recommended restaurants")
            return filtered_restaurants

        except Exception as e:
            logger.error(f"Error filtering already recommended restaurants: {e}")
            return restaurants  # Return original list if filtering fails

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