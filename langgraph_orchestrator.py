# langgraph_supervisor.py
"""
Simplified LangGraph Supervisor for Restaurant Recommendation Bot

This is a lightweight supervisor that:
1. Routes ALL user messages through AI Chat Layer for conversation management
2. Delegates search execution to specialized LCEL pipelines:
   - CitySearchOrchestrator for city-wide searches
   - LocationSearchOrchestrator for GPS/location-based searches
3. Handles structured handoffs (CONTINUE_CONVERSATION, EXECUTE_SEARCH, RESUME_WITH_DECISION)
4. Manages conversation state and user preferences

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Supervisor                                 │
│                                                                             │
│  ┌──────────────┐      ┌─────────────────┐      ┌───────────────────────┐  │
│  │ User Message │ ───▶ │  AI Chat Layer  │ ───▶ │   Handoff Command     │  │
│  └──────────────┘      │  (conversation) │      │                       │  │
│                        └─────────────────┘      └───────────┬───────────┘  │
│                                                             │              │
│         ┌───────────────────────────────────────────────────┼──────┐       │
│         │                       │                           │      │       │
│         ▼                       ▼                           ▼      ▼       │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐  │
│  │ CONTINUE    │    │ EXECUTE_SEARCH   │    │ RESUME_WITH_DECISION     │  │
│  │ CONVERSATION│    │                  │    │ (follow-up)              │  │
│  │ (no search) │    │  ┌────────────┐  │    └──────────────────────────┘  │
│  └─────────────┘    │  │ City LCEL  │  │                                  │
│                     │  │ Pipeline   │  │                                  │
│                     │  └────────────┘  │                                  │
│                     │        OR        │                                  │
│                     │  ┌────────────┐  │                                  │
│                     │  │ Location   │  │                                  │
│                     │  │ LCEL Pipe  │  │                                  │
│                     │  └────────────┘  │                                  │
│                     └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘

This replaces the heavy langgraph_orchestrator.py with a clean supervisor pattern.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langsmith import traceable

# Handoff protocol
from utils.handoff_protocol import (
    HandoffMessage, SearchContext, SearchType, HandoffCommand
)

# AI Chat Layer (conversation management)
from utils.ai_chat_layer import AIChatLayer

# LCEL Pipelines (search execution)
from city_search_orchestrator import CitySearchOrchestrator
from location_search_orchestrator import LocationSearchOrchestrator

# Location utilities
from location.telegram_location_handler import LocationData

logger = logging.getLogger(__name__)


class LangGraphSupervisor:
    """
    Simplified LangGraph Supervisor

    Responsibilities:
    - Route user messages through AI Chat Layer
    - Dispatch to appropriate LCEL pipeline based on handoff command
    - Handle conversation flow (continue, search, resume)
    - Maintain backward compatibility with existing telegram_bot.py
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI Chat Layer (conversation management)
        self.ai_chat_layer = AIChatLayer(config)
        logger.info("✅ AI Chat Layer initialized")

        # Initialize LCEL pipelines (search execution)
        self.city_pipeline = CitySearchOrchestrator(config)
        self.location_pipeline = LocationSearchOrchestrator(config)
        logger.info("✅ LCEL pipelines initialized (city + location)")

        # Statistics
        self.stats = {
            "total_messages": 0,
            "conversations": 0,
            "city_searches": 0,
            "location_searches": 0,
            "errors": 0
        }

        logger.info("✅ LangGraph Supervisor initialized")

    # ============================================================================
    # MAIN ENTRY POINT
    # ============================================================================

    @traceable(run_type="chain", name="supervisor_process_message")
    async def process_message(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        telegram_bot=None,
        chat_id: Optional[int] = None,
        cancel_check_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user messages.

        Routes through AI Chat Layer for conversation management,
        then dispatches to appropriate LCEL pipeline if search is needed.

        Args:
            query: User's message
            user_id: Telegram user ID
            gps_coordinates: Optional GPS coordinates (lat, lng)
            thread_id: Thread ID for state management
            telegram_bot: Telegram bot instance (for sending messages)
            chat_id: Telegram chat ID
            cancel_check_fn: Optional function to check for cancellation

        Returns:
            Dict with:
            - success: bool
            - ai_response: str (conversation response if no search)
            - search_triggered: bool
            - langchain_formatted_results: str (if search was executed)
            - needs_location_button: bool (if GPS is needed)
            - action_taken: str (conversation, city_search, location_search, etc.)
        """
        start_time = time.time()
        self.stats["total_messages"] += 1

        try:
            # Generate thread ID if needed
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            logger.info(f"💬 Supervisor processing: '{query[:50]}...' for user {user_id}")

            # ================================================================
            # STEP 1: Get structured handoff from AI Chat Layer
            # ================================================================
            handoff: HandoffMessage = await self.ai_chat_layer.process_message(
                user_id=user_id,
                user_message=query,
                gps_coordinates=gps_coordinates,
                thread_id=thread_id
            )

            logger.info(f"🎯 Handoff Command: {handoff.command.value}")
            logger.info(f"📝 Reasoning: {handoff.reasoning}")

            # ================================================================
            # STEP 2: Route by handoff command
            # ================================================================

            # ----- COMMAND 1: CONTINUE_CONVERSATION -----
            if handoff.command == HandoffCommand.CONTINUE_CONVERSATION:
                return self._handle_conversation(handoff, start_time)

            # ----- COMMAND 2: EXECUTE_SEARCH -----
            elif handoff.command == HandoffCommand.EXECUTE_SEARCH:
                return await self._handle_search(
                    handoff=handoff,
                    user_id=user_id,
                    gps_coordinates=gps_coordinates,
                    thread_id=thread_id,
                    telegram_bot=telegram_bot,
                    chat_id=chat_id,
                    cancel_check_fn=cancel_check_fn,
                    start_time=start_time
                )

            # ----- COMMAND 3: RESUME_WITH_DECISION -----
            elif handoff.command == HandoffCommand.RESUME_WITH_DECISION:
                return await self._handle_resume(
                    handoff=handoff,
                    user_id=user_id,
                    gps_coordinates=gps_coordinates,
                    cancel_check_fn=cancel_check_fn,
                    start_time=start_time
                )

            # ----- UNKNOWN COMMAND -----
            else:
                logger.warning(f"⚠️ Unknown handoff command: {handoff.command}")
                return {
                    "success": False,
                    "error_message": f"Unknown command: {handoff.command}",
                    "ai_response": "I'm not sure how to help with that. Could you tell me what restaurants you're looking for?",
                    "search_triggered": False,
                    "processing_time": round(time.time() - start_time, 2)
                }

        except Exception as e:
            logger.error(f"❌ Supervisor error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.stats["errors"] += 1

            return {
                "success": False,
                "error_message": str(e),
                "ai_response": "I encountered an error. Please try again.",
                "search_triggered": False,
                "processing_time": round(time.time() - start_time, 2)
            }

    # ============================================================================
    # COMMAND HANDLERS
    # ============================================================================

    def _handle_conversation(
        self,
        handoff: HandoffMessage,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle CONTINUE_CONVERSATION command.
        No search needed - just return conversation response.
        """
        self.stats["conversations"] += 1
        processing_time = round(time.time() - start_time, 2)

        logger.info(f"💬 Conversation response (no search), needs_gps={handoff.needs_gps}")

        return {
            "success": True,
            "ai_response": handoff.conversation_response,
            "action_taken": "conversation",
            "search_triggered": False,
            "needs_location_button": handoff.needs_gps,
            "processing_time": processing_time,
            "reasoning": handoff.reasoning
        }

    async def _handle_search(
        self,
        handoff: HandoffMessage,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]],
        thread_id: str,
        telegram_bot,
        chat_id: Optional[int],
        cancel_check_fn: Optional[callable],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle EXECUTE_SEARCH command.
        Routes to appropriate LCEL pipeline based on search type.
        """
        search_ctx = handoff.search_context

        if search_ctx is None:
            logger.error("❌ EXECUTE_SEARCH but no search_context")
            return {
                "success": False,
                "error_message": "No search context in handoff",
                "ai_response": "I encountered an error. Please try again.",
                "search_triggered": False,
                "processing_time": round(time.time() - start_time, 2)
            }

        logger.info(f"🔍 Search requested: type={search_ctx.search_type.value}, "
                   f"destination='{search_ctx.destination}', cuisine='{search_ctx.cuisine}'")

        # Build search query from context
        search_query = self._build_search_query(search_ctx)

        # Get GPS coordinates from context or parameter
        search_gps = search_ctx.gps_coordinates or gps_coordinates

        # ================================================================
        # ROUTE TO APPROPRIATE LCEL PIPELINE
        # ================================================================

        if search_ctx.search_type == SearchType.LOCATION_SEARCH:
            # ----- LOCATION SEARCH (GPS-based) -----
            return await self._execute_location_search(
                search_query=search_query,
                search_ctx=search_ctx,
                gps_coordinates=search_gps,
                cancel_check_fn=cancel_check_fn,
                start_time=start_time
            )
        else:
            # ----- CITY SEARCH (city-wide) -----
            return await self._execute_city_search(
                search_query=search_query,
                search_ctx=search_ctx,
                cancel_check_fn=cancel_check_fn,
                start_time=start_time
            )

    async def _handle_resume(
        self,
        handoff: HandoffMessage,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]],
        cancel_check_fn: Optional[callable],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle RESUME_WITH_DECISION command.
        Used for follow-up requests like "show more".
        """
        logger.info("🔄 Resume with decision (follow-up)")

        # Get the resume payload
        resume_payload = handoff.resume_payload or {}
        original_thread_id = resume_payload.get("thread_id")
        decision = resume_payload.get("answer", "yes")

        if not original_thread_id:
            return {
                "success": False,
                "ai_response": "I couldn't find your previous search. Could you tell me what you're looking for?",
                "search_triggered": False,
                "processing_time": round(time.time() - start_time, 2)
            }

        # For "show more" requests, use location pipeline with maps_only=True
        if gps_coordinates:
            # Create LocationData from coordinates
            location_data = LocationData(
                latitude=gps_coordinates[0],
                longitude=gps_coordinates[1],
                description=f"GPS: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}"
            )

            result = await self.location_pipeline.process_location_query_async(
                query="restaurants",  # Generic query for "show more"
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True  # Skip database, go directly to Maps
            )

            processing_time = round(time.time() - start_time, 2)

            return {
                "success": result.get("success", False),
                "langchain_formatted_results": result.get("location_formatted_results", ""),
                "location_formatted_results": result.get("location_formatted_results", ""),
                "action_taken": "location_search_resume",
                "search_triggered": True,
                "processing_time": processing_time,
                "restaurant_count": result.get("restaurant_count", 0)
            }
        else:
            # No GPS for resume - ask for location
            return {
                "success": True,
                "ai_response": "To show more restaurants, I need your location. Could you share it?",
                "needs_location_button": True,
                "search_triggered": False,
                "processing_time": round(time.time() - start_time, 2)
            }

    # ============================================================================
    # SEARCH EXECUTION
    # ============================================================================

    @traceable(run_type="chain", name="execute_city_search")
    async def _execute_city_search(
        self,
        search_query: str,
        search_ctx: SearchContext,
        cancel_check_fn: Optional[callable],
        start_time: float
    ) -> Dict[str, Any]:
        """Execute city-wide search using CitySearchOrchestrator LCEL pipeline"""
        self.stats["city_searches"] += 1

        logger.info(f"🏙️ Executing city search: '{search_query}'")

        try:
            # Execute the LCEL pipeline
            result = await self.city_pipeline.process_query_async(
                query=search_query,
                cancel_check_fn=cancel_check_fn
            )

            processing_time = round(time.time() - start_time, 2)

            # Map result to expected format
            return {
                "success": result.get("success", False),
                "langchain_formatted_results": result.get("langchain_formatted_results", ""),
                "action_taken": "city_search",
                "search_triggered": True,
                "processing_time": processing_time,
                "destination": search_ctx.destination,
                "cuisine": search_ctx.cuisine,
                "content_source": result.get("content_source", "unknown"),
                "restaurant_count": len(result.get("enhanced_results", {}).get("main_list", []))
            }

        except Exception as e:
            logger.error(f"❌ City search error: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "langchain_formatted_results": "Sorry, there was an error searching. Please try again.",
                "action_taken": "city_search_error",
                "search_triggered": True,
                "processing_time": round(time.time() - start_time, 2)
            }

    @traceable(run_type="chain", name="execute_location_search")
    async def _execute_location_search(
        self,
        search_query: str,
        search_ctx: SearchContext,
        gps_coordinates: Optional[Tuple[float, float]],
        cancel_check_fn: Optional[callable],
        start_time: float
    ) -> Dict[str, Any]:
        """Execute location-based search using LocationSearchOrchestrator LCEL pipeline"""
        self.stats["location_searches"] += 1

        logger.info(f"📍 Executing location search: '{search_query}'")

        if not gps_coordinates:
            # No GPS - return response asking for location
            return {
                "success": True,
                "ai_response": f"To search for {search_query} near you, I need your location. Could you share it?",
                "needs_location_button": True,
                "search_triggered": False,
                "processing_time": round(time.time() - start_time, 2)
            }

        try:
            # Create LocationData from coordinates
            location_data = LocationData(
                latitude=gps_coordinates[0],
                longitude=gps_coordinates[1],
                description=search_ctx.destination or f"GPS: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}"
            )

            # Execute the LCEL pipeline
            result = await self.location_pipeline.process_location_query_async(
                query=search_query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=False  # Normal flow with database first
            )

            processing_time = round(time.time() - start_time, 2)

            # Map result to expected format
            return {
                "success": result.get("success", False),
                "langchain_formatted_results": result.get("location_formatted_results", ""),
                "location_formatted_results": result.get("location_formatted_results", ""),
                "action_taken": "location_search",
                "search_triggered": True,
                "processing_time": processing_time,
                "coordinates": gps_coordinates,
                "source": result.get("source", "unknown"),
                "restaurant_count": result.get("restaurant_count", 0)
            }

        except Exception as e:
            logger.error(f"❌ Location search error: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "location_formatted_results": "Sorry, there was an error searching. Please try again.",
                "action_taken": "location_search_error",
                "search_triggered": True,
                "processing_time": round(time.time() - start_time, 2)
            }

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _build_search_query(self, search_ctx: SearchContext) -> str:
        """Build a search query string from SearchContext"""
        parts = []

        if search_ctx.cuisine:
            parts.append(search_ctx.cuisine)

        if search_ctx.requirements:
            parts.extend(search_ctx.requirements)

        if search_ctx.destination and search_ctx.search_type == SearchType.CITY_SEARCH:
            parts.append(f"in {search_ctx.destination}")

        if not parts:
            # Fallback to user query
            return search_ctx.user_query or "restaurants"

        return " ".join(parts)

    # ============================================================================
    # SYNCHRONOUS WRAPPERS (for backward compatibility)
    # ============================================================================

    def process_message_sync(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        telegram_bot=None,
        chat_id: Optional[int] = None,
        cancel_check_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_message.
        For use in telegram_bot.py which may not be fully async.
        """
        import concurrent.futures

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.process_message(
                        query=query,
                        user_id=user_id,
                        gps_coordinates=gps_coordinates,
                        thread_id=thread_id,
                        telegram_bot=telegram_bot,
                        chat_id=chat_id,
                        cancel_check_fn=cancel_check_fn
                    )
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_async).result()

    # ============================================================================
    # LEGACY COMPATIBILITY METHODS
    # ============================================================================

    def process_query(self, query: str, cancel_check_fn: Optional[callable] = None) -> Dict[str, Any]:
        """
        Legacy method for direct city search (bypasses AI Chat Layer).
        Used when telegram_bot.py calls perform_city_search directly.
        """
        logger.info(f"📜 Legacy process_query called: '{query[:50]}...'")
        return self.city_pipeline.process_query(query, cancel_check_fn)

    async def search_restaurants(
        self,
        query: str,
        user_id: Optional[int] = None,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        location_data: Optional[Any] = None,
        thread_id: Optional[str] = None,
        search_context: Optional[SearchContext] = None
    ) -> Dict[str, Any]:
        """
        Legacy method for unified search.
        Routes based on available context.
        """
        logger.info(f"📜 Legacy search_restaurants called: '{query[:50]}...'")

        # Determine search type
        if gps_coordinates or location_data:
            # Location search
            if location_data:
                loc_data = location_data
            else:
                loc_data = LocationData(
                    latitude=gps_coordinates[0],
                    longitude=gps_coordinates[1],
                    description=f"GPS: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}"
                )

            result = await self.location_pipeline.process_location_query_async(
                query=query,
                location_data=loc_data,
                maps_only=False
            )

            # Map to expected format
            return {
                "success": result.get("success", False),
                "formatted_message": result.get("location_formatted_results", ""),
                "langchain_formatted_results": result.get("location_formatted_results", ""),
                "final_restaurants": result.get("results", []),
                "search_flow": "location_search"
            }
        else:
            # City search
            result = await self.city_pipeline.process_query_async(query)

            return {
                "success": result.get("success", False),
                "formatted_message": result.get("langchain_formatted_results", ""),
                "langchain_formatted_results": result.get("langchain_formatted_results", ""),
                "final_restaurants": result.get("enhanced_results", {}).get("main_list", []),
                "search_flow": "city_search"
            }

    # ============================================================================
    # STATISTICS
    # ============================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get supervisor statistics"""
        return {
            "supervisor": self.stats,
            "city_pipeline": self.city_pipeline.get_stats(),
            "location_pipeline": self.location_pipeline.get_stats()
        }

    def get_info(self) -> Dict[str, Any]:
        """Get supervisor configuration info"""
        return {
            "type": "langgraph_supervisor",
            "version": "2.0",
            "architecture": "supervisor_with_lcel_pipelines",
            "components": {
                "ai_chat_layer": "conversation management",
                "city_pipeline": self.city_pipeline.get_pipeline_info(),
                "location_pipeline": self.location_pipeline.get_pipeline_info()
            },
            "handoff_commands": [
                "CONTINUE_CONVERSATION",
                "EXECUTE_SEARCH",
                "RESUME_WITH_DECISION"
            ]
        }

    # ============================================================================
    # ADDITIONAL LEGACY METHODS (for telegram_bot.py compatibility)
    # ============================================================================

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
        Legacy method name used by old langgraph_orchestrator.
        Redirects to process_message.
        """
        return await self.process_message(
            query=query,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=thread_id,
            telegram_bot=telegram_bot,
            chat_id=chat_id
        )

    async def process_user_message(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Legacy method name. Redirects to process_message.
        """
        return await self.process_message(
            query=query,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=thread_id,
            telegram_bot=kwargs.get('telegram_bot'),
            chat_id=kwargs.get('chat_id')
        )


# ============================================================================
# CLASS ALIAS (for backward compatibility)
# ============================================================================

# Alias so old imports still work
UnifiedRestaurantAgent = LangGraphSupervisor


# ============================================================================
# FACTORY FUNCTION (for backward compatibility)
# ============================================================================

def create_unified_restaurant_agent(config) -> LangGraphSupervisor:
    """
    Factory function to create the supervisor.
    Maintains backward compatibility with existing main.py
    """
    return LangGraphSupervisor(config)


# ============================================================================
# ORCHESTRATOR MANAGER COMPATIBILITY
# ============================================================================

_supervisor_instance = None

def get_supervisor(config=None) -> LangGraphSupervisor:
    """
    Get or create supervisor singleton.
    For use with orchestrator_manager.py pattern.
    """
    global _supervisor_instance

    if _supervisor_instance is None:
        if config is None:
            import config as app_config
            config = app_config
        _supervisor_instance = LangGraphSupervisor(config)

    return _supervisor_instance
