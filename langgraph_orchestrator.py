# langgraph_orchestrator.py
"""
LangGraph Supervisor for Restaurant Recommendation Bot - WITH MEMORY INTEGRATION

This supervisor:
1. Routes ALL user messages through AI Chat Layer for conversation management
2. Integrates with AIMemorySystem for persistent user preferences and history
3. Delegates search execution to specialized LCEL pipelines
4. Handles structured handoffs (CONTINUE_CONVERSATION, EXECUTE_SEARCH, RESUME_WITH_DECISION)

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Supervisor                                 │
│                                                                             │
│  ┌──────────────┐   ┌─────────────┐   ┌─────────────────┐                  │
│  │ User Message │──▶│   MEMORY    │──▶│  AI Chat Layer  │                  │
│  └──────────────┘   │   SYSTEM    │   │  (conversation) │                  │
│                     └─────────────┘   └────────┬────────┘                  │
│                                                │                            │
│         ┌──────────────────────────────────────┼──────────────┐            │
│         │                    │                 │              │            │
│         ▼                    ▼                 ▼              ▼            │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────┐  ┌──────────────┐   │
│  │ CONTINUE    │   │EXECUTE_SEARCH│   │  RESUME    │  │ SAVE TO      │   │
│  │ CONVERSATION│   │              │   │  (follow)  │  │ MEMORY       │   │
│  └─────────────┘   │ City | Loc   │   └────────────┘  └──────────────┘   │
│                    └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Tuple, Callable, List
from dataclasses import dataclass
import telebot.types

from langsmith import traceable

# Handoff protocol
from utils.handoff_protocol import (
    HandoffMessage, SearchContext, SearchType, HandoffCommand
)

# AI Chat Layer (conversation management)
from utils.ai_chat_layer import AIChatLayer

# Memory System
from utils.ai_memory_system import AIMemorySystem, RestaurantMemory

# LCEL Pipelines (search execution)
from city_search_orchestrator import CitySearchOrchestrator
from location_orchestrator import LocationOrchestrator

# Location utilities
from location.telegram_location_handler import LocationData

logger = logging.getLogger(__name__)


class LangGraphSupervisor:
    """
    LangGraph Supervisor with Memory Integration
    
    Responsibilities:
    - Load user context from memory before processing
    - Route user messages through AI Chat Layer
    - Dispatch to appropriate LCEL pipeline based on handoff command
    - Save recommendations and update preferences after searches
    - Handle conversation flow (continue, search, resume)
    """

    def __init__(self, config):
        self.config = config

        # Initialize Memory System FIRST
        self.memory_system = AIMemorySystem(config)
        logger.info("✅ Memory System initialized")

        # Initialize AI Chat Layer (conversation management)
        self.ai_chat_layer = AIChatLayer(config)
        logger.info("✅ AI Chat Layer initialized")

        # Initialize LCEL pipelines (search execution)
        self.city_pipeline = CitySearchOrchestrator(config)
        self.location_pipeline = LocationOrchestrator(config)
        logger.info("✅ LCEL pipelines initialized (city + location)")

        # Statistics
        self.stats = {
            "total_messages": 0,
            "conversations": 0,
            "city_searches": 0,
            "location_searches": 0,
            "memory_loads": 0,
            "memory_saves": 0,
            "errors": 0
        }

        logger.info("✅ LangGraph Supervisor initialized with Memory Integration")

    # ============================================================================
    # MEMORY OPERATIONS
    # ============================================================================

    async def _load_user_context(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """
        Load comprehensive user context from memory system.
        
        This provides the AI Chat Layer with:
        - User preferences (cuisines, dietary restrictions, etc.)
        - Recent restaurant recommendations (to avoid repeats)
        - Conversation patterns (communication style)
        """
        try:
            self.stats["memory_loads"] += 1
            
            # Get user context from memory system
            context = await self.memory_system.get_user_context(user_id, thread_id)
            
            # Log what we loaded
            prefs = context.get("preferences", {})
            history_count = len(context.get("restaurant_history", []))
            
            logger.info(f"🧠 Loaded memory for user {user_id}: "
                       f"{len(prefs.get('preferred_cuisines', []))} cuisines, "
                       f"{history_count} past restaurants")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error loading user context: {e}")
            # Return empty context on error - don't fail the whole request
            return {
                "user_id": user_id,
                "thread_id": thread_id,
                "preferences": None,
                "restaurant_history": [],
                "conversation_patterns": None,
                "error": str(e)
            }

    async def _save_recommendations_to_memory(
        self,
        user_id: int,
        restaurants: List[Dict[str, Any]],
        search_context: SearchContext
    ) -> bool:
        """
        Save recommended restaurants to user's memory.
        
        This allows:
        - Avoiding repeat recommendations
        - Referencing past suggestions
        - Learning user preferences over time
        """
        try:
            self.stats["memory_saves"] += 1
            
            saved_count = 0
            for restaurant in restaurants[:10]:  # Save up to 10 per search
                # Create RestaurantMemory object
                memory = RestaurantMemory(
                    restaurant_name=restaurant.get("name", "Unknown"),
                    city=search_context.destination or "Unknown",
                    cuisine=search_context.cuisine or restaurant.get("cuisine", ""),
                    recommended_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                    user_feedback=None,  # Will be updated if user gives feedback
                    rating_given=None,
                    notes=None,
                    source=restaurant.get("source", "search")
                )
                
                # Save to memory
                success = await self.memory_system.add_restaurant_memory(user_id, memory)
                if success:
                    saved_count += 1
            
            logger.info(f"💾 Saved {saved_count} restaurants to memory for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving recommendations to memory: {e}")
            return False

    async def _update_preferences_from_search(
        self,
        user_id: int,
        search_context: SearchContext
    ) -> bool:
        """
        Update user preferences based on their search.
        
        Learns:
        - Preferred cuisines
        - Frequently searched cities
        - Dining requirements (romantic, outdoor, etc.)
        """
        try:
            # Use the memory system's learning method
            success = await self.memory_system.learn_preferences_from_message(
                user_id=user_id,
                message=search_context.user_query,
                current_city=search_context.destination,
                extracted_cuisine=search_context.cuisine,
                extracted_requirements=search_context.requirements,
                extracted_preferences=search_context.preferences
            )
            
            if success:
                logger.info(f"📚 Updated preferences for user {user_id} from search")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error updating preferences: {e}")
            return False

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
        cancel_check_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user messages.

        Flow:
        1. Load user context from memory
        2. Route through AI Chat Layer (with context)
        3. Execute search if needed
        4. Save results to memory
        5. Return response
        """
        start_time = time.time()
        self.stats["total_messages"] += 1

        try:
            # Generate thread ID if needed
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            logger.info(f"💬 Supervisor processing: '{query[:50]}...' for user {user_id}")

            # ================================================================
            # STEP 1: Load user context from memory
            # ================================================================
            user_context = await self._load_user_context(user_id, thread_id)

            # ================================================================
            # STEP 2: Get structured handoff from AI Chat Layer (with context)
            # ================================================================
            handoff: HandoffMessage = await self.ai_chat_layer.process_message(
                user_id=user_id,
                user_message=query,
                gps_coordinates=gps_coordinates,
                thread_id=thread_id,
                user_context=user_context  # Pass memory context
            )

            logger.info(f"🎯 Handoff Command: {handoff.command.value}")
            logger.info(f"📝 Reasoning: {handoff.reasoning}")

            # ================================================================
            # STEP 3: Route by handoff command
            # ================================================================

            # ----- COMMAND 1: CONTINUE_CONVERSATION -----
            if handoff.command == HandoffCommand.CONTINUE_CONVERSATION:
                return self._handle_conversation(handoff, start_time)

            # ----- COMMAND 2: EXECUTE_SEARCH -----
            elif handoff.command == HandoffCommand.EXECUTE_SEARCH:
                result = await self._handle_search(
                    handoff=handoff,
                    user_id=user_id,
                    gps_coordinates=gps_coordinates,
                    thread_id=thread_id,
                    telegram_bot=telegram_bot,
                    chat_id=chat_id,
                    cancel_check_fn=cancel_check_fn,
                    start_time=start_time
                )
                
                # ============================================================
                # STEP 4: Save results to memory (after successful search)
                # ============================================================
                if result.get("success") and handoff.search_context:
                    # Get restaurants from result
                    restaurants = self._extract_restaurants_from_result(result)
                    
                    if restaurants:
                        # Save recommendations
                        await self._save_recommendations_to_memory(
                            user_id=user_id,
                            restaurants=restaurants,
                            search_context=handoff.search_context
                        )

                        # Add search results to conversation history for follow-up questions
                        formatted_results = result.get("langchain_formatted_results") or result.get("location_formatted_results") or result.get("ai_response", "")
                        if formatted_results:
                            self.ai_chat_layer.add_search_results(
                                user_id=user_id,
                                formatted_results=formatted_results,
                                search_context={
                                    'cuisine': handoff.search_context.cuisine if handoff.search_context else None,
                                    'destination': handoff.search_context.destination if handoff.search_context else None,
                                }
                            )
                        
                        # Update preferences
                        await self._update_preferences_from_search(
                            user_id=user_id,
                            search_context=handoff.search_context
                        )
                
                return result

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
                    "ai_response": "Could you tell me what restaurants you're looking for?",
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

    def _extract_restaurants_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract restaurant list from search result for memory storage"""
        # Try different keys where restaurants might be stored
        restaurants = result.get("final_restaurants", [])
        
        if not restaurants:
            restaurants = result.get("restaurants", [])
        
        if not restaurants:
            # Try to get from nested search_result
            search_result = result.get("search_result", {})
            restaurants = search_result.get("final_restaurants", [])
        
        if not restaurants:
            # Try enhanced_results for city search
            enhanced = result.get("enhanced_results", {})
            restaurants = enhanced.get("main_list", [])
        
        return restaurants

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
        cancel_check_fn: Optional[Callable],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle EXECUTE_SEARCH command.
        Sends confirmation video, then routes to appropriate LCEL pipeline.
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
        # SEND CONFIRMATION MESSAGE WITH VIDEO
        # ================================================================
        confirmation_msg = None
        if telegram_bot and chat_id:
            confirmation_msg = await self._send_search_confirmation(
                telegram_bot=telegram_bot,
                chat_id=chat_id,
                search_type=search_ctx.search_type,
                destination=search_ctx.destination,
                cuisine=search_ctx.cuisine
            )

        # ================================================================
        # ROUTE TO APPROPRIATE LCEL PIPELINE
        # ================================================================
        try:
            if search_ctx.search_type == SearchType.LOCATION_MAPS_SEARCH:
                # ----- MAPS-ONLY SEARCH (for "more results") -----
                result = await self._execute_location_maps_search(
                    search_query=search_query,
                    search_ctx=search_ctx,
                    gps_coordinates=search_gps,
                    cancel_check_fn=cancel_check_fn,
                    start_time=start_time
                )
                # No session update needed - this is a follow-up search

            elif search_ctx.search_type == SearchType.LOCATION_SEARCH:
                # ----- LOCATION SEARCH (database first, then maps) -----
                result = await self._execute_location_search(
                    search_query=search_query,
                    search_ctx=search_ctx,
                    gps_coordinates=search_gps,
                    cancel_check_fn=cancel_check_fn,
                    start_time=start_time
                )

                # ✅ NEW: Update AI Chat Layer session for follow-up "more" requests
                if result.get("success"):
                    restaurants = result.get("final_restaurants", [])
                    self.ai_chat_layer.update_last_search_context(
                        user_id=user_id,
                        search_type='location_search',
                        cuisine=search_ctx.cuisine,
                        destination=search_ctx.destination,
                        restaurants=restaurants,
                        coordinates=search_gps
                    )
            else:
                # ----- CITY SEARCH (city-wide) -----
                result = await self._execute_city_search(
                    search_query=search_query,
                    search_ctx=search_ctx,
                    cancel_check_fn=cancel_check_fn,
                    start_time=start_time
                )

                # ✅ NEW: Update AI Chat Layer session for follow-up "more" requests
                if result.get("success"):
                    restaurants = result.get("final_restaurants", [])
                    self.ai_chat_layer.update_last_search_context(
                        user_id=user_id,
                        search_type='city_search',
                        cuisine=search_ctx.cuisine,
                        destination=search_ctx.destination,
                        restaurants=restaurants
                    )
            
            
            # Clean up confirmation message after search completes
            if confirmation_msg and telegram_bot:
                try:
                    telegram_bot.delete_message(chat_id, confirmation_msg.message_id)
                except Exception as e:
                    logger.debug(f"Could not delete confirmation message: {e}")
            
            return result
            
        except Exception as e:
            # Clean up confirmation message on error
            if confirmation_msg and telegram_bot:
                try:
                    telegram_bot.delete_message(chat_id, confirmation_msg.message_id)
                except Exception:
                    pass
            raise

    async def _send_search_confirmation(
        self,
        telegram_bot,
        chat_id: int,
        search_type: SearchType,
        destination: Optional[str],
        cuisine: Optional[str]
    ):
        """
        Send confirmation message with video before search starts.

        Uses different videos for city vs location searches.
        Also removes any location keyboard that might be showing.
        """
        try:
            # Create keyboard removal markup
            remove_keyboard = telebot.types.ReplyKeyboardRemove()

            # Build caption based on search type
            if search_type == SearchType.LOCATION_SEARCH:
                video_path = 'media/vicinity_search.mp4'
                # Treat "unknown" as no destination (GPS-only)
                if destination and destination.lower() != "unknown":
                    if cuisine:
                        caption = f"📍 <b>Searching for {cuisine} near {destination}...</b>\n\n⏱ Checking my curated collection and finding the best places nearby."
                    else:
                        caption = f"📍 <b>Searching for restaurants near {destination}...</b>\n\n⏱ Checking my curated collection and finding the best places nearby."
                elif cuisine:
                    caption = f"📍 <b>Searching for {cuisine} near you...</b>\n\n⏱ Checking my curated collection and finding the best places nearby."
                else:
                    caption = "📍 <b>Searching for restaurants nearby...</b>\n\n⏱ Checking my curated collection and finding the best places nearby."
            else:
                video_path = 'media/searching.mp4'
                if cuisine and destination:
                    caption = f"🔍 <b>Searching for {cuisine} in {destination}...</b>\n\n⏱ This might take a minute while I check with my sources."
                elif destination:
                    caption = f"🔍 <b>Searching for restaurants in {destination}...</b>\n\n⏱ This might take a minute while I check with my sources."
                else:
                    caption = "🔍 <b>Searching for the best restaurants...</b>\n\n⏱ This might take a minute while I check with my sources."

            # Try to send video
            try:
                with open(video_path, 'rb') as video:
                    confirmation_msg = telegram_bot.send_video(
                        chat_id,
                        video,
                        caption=caption,
                        parse_mode='HTML',
                        reply_markup=remove_keyboard  # Remove location button!
                    )
                    logger.info(f"📹 Sent search confirmation with video: {video_path}")
                    return confirmation_msg
            except FileNotFoundError:
                logger.warning(f"⚠️ Video file not found: {video_path}")
                # Fallback to text message
                confirmation_msg = telegram_bot.send_message(
                    chat_id,
                    caption,
                    parse_mode='HTML',
                    reply_markup=remove_keyboard  # Remove location button!
                )
                return confirmation_msg
            except Exception as e:
                logger.warning(f"⚠️ Could not send video: {e}")
                # Fallback to text message
                confirmation_msg = telegram_bot.send_message(
                    chat_id,
                    caption,
                    parse_mode='HTML',
                    reply_markup=remove_keyboard  # Remove location button!
                )
                return confirmation_msg

        except Exception as e:
            logger.error(f"❌ Error sending search confirmation: {e}")
            return None

    async def _handle_resume(
        self,
        handoff: HandoffMessage,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]],
        cancel_check_fn: Optional[Callable],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle RESUME_WITH_DECISION command.
        Used for follow-up requests like "show more".

        Gets the last search context from AI Chat Layer and triggers
        a maps-only search to get more results.

        NEW: Now includes supervisor_instructions and exclude_restaurants
        for smarter follow-up filtering.
        """
        logger.info("🔄 Resume with decision (follow-up)")

        # Get last search context from AI Chat Layer
        last_context = self.ai_chat_layer.get_last_search_context(user_id)

        if not last_context or not last_context.get('search_type'):
            logger.warning(f"⚠️ No last search context found for user {user_id}")
            return {
                "success": True,
                "ai_response": "I couldn't find your previous search. What are you looking for?",
                "action_taken": "resume_no_context",
                "search_triggered": False,
                "processing_time": round(time.time() - start_time, 2)
            }

        search_type = last_context.get('search_type')
        cuisine = last_context.get('cuisine') or 'restaurants'
        destination = last_context.get('destination') or 'nearby'
        shown_restaurants = last_context.get('shown_restaurants', [])

        logger.info(f"🔄 Resuming search: type={search_type}, cuisine={cuisine}, dest={destination}")
        logger.info(f"🔄 Excluding {len(shown_restaurants)} previously shown restaurants")

        # Get stored GPS from AI Chat Layer session
        session = self.ai_chat_layer.user_sessions.get(user_id, {})
        stored_gps = gps_coordinates or session.get('gps_coordinates')

        # Build search query
        search_query = f"more {cuisine}" if cuisine else "more restaurants"

        # NEW: Generate supervisor instructions for downstream agents
        supervisor_instructions = None
        if shown_restaurants:
            exclude_list = ", ".join(shown_restaurants[:10])  # Limit to first 10 for prompt size
            supervisor_instructions = (
                f"User wants MORE results. EXCLUDE already shown restaurants: {exclude_list}. "
                f"Find DIFFERENT {cuisine} options in the same area."
            )

        # Create search context with follow-up fields
        search_ctx = SearchContext(
            cuisine=cuisine,
            destination=destination,
            search_type=SearchType.LOCATION_MAPS_SEARCH if search_type == 'location_search' else SearchType.CITY_SEARCH,
            user_query=search_query,
            requirements=[],
            preferences={},
            gps_coordinates=stored_gps,
            # NEW: Follow-up context
            supervisor_instructions=supervisor_instructions,
            exclude_restaurants=shown_restaurants,
            is_follow_up=True
        )

        # Now execute the search based on type
        if search_ctx.search_type == SearchType.LOCATION_MAPS_SEARCH:
            return await self._execute_location_maps_search(
                search_query=search_query,
                search_ctx=search_ctx,
                gps_coordinates=stored_gps,
                cancel_check_fn=cancel_check_fn,
                start_time=start_time
            )
        else:
            return await self._execute_city_search(
                search_query=search_query,
                search_ctx=search_ctx,
                cancel_check_fn=cancel_check_fn,
                start_time=start_time
            )

    # ============================================================================
    # SEARCH EXECUTION
    # ============================================================================

    @traceable(run_type="chain", name="execute_city_search")
    async def _execute_city_search(
        self,
        search_query: str,
        search_ctx: SearchContext,
        cancel_check_fn: Optional[Callable],
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

            # Extract restaurants for memory
            restaurants = result.get("enhanced_results", {}).get("main_list", [])

            # Map result to expected format
            return {
                "success": result.get("success", False),
                "ai_response": result.get("langchain_formatted_results", ""),
                "langchain_formatted_results": result.get("langchain_formatted_results", ""),
                "action_taken": "city_search",
                "search_triggered": True,
                "processing_time": processing_time,
                "destination": search_ctx.destination,
                "cuisine": search_ctx.cuisine,
                "content_source": result.get("content_source", "unknown"),
                "restaurant_count": len(restaurants),
                "final_restaurants": restaurants
            }

        except Exception as e:
            logger.error(f"❌ City search error: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "ai_response": "Sorry, there was an error searching. Please try again.",
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
        cancel_check_fn: Optional[Callable],
        start_time: float
    ) -> Dict[str, Any]:
        """Execute location-based search using LocationOrchestrator LCEL pipeline"""
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

            # Execute the LCEL pipeline with maps_only=True
            # NEW: Pass supervisor_instructions and exclude_restaurants for smarter filtering
            result = await self.location_pipeline.process_location_query(
                query=search_query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True,  # KEY: Skip database, go directly to Google Maps
                supervisor_instructions=search_ctx.supervisor_instructions,
                exclude_restaurants=search_ctx.exclude_restaurants
            )

            processing_time = round(time.time() - start_time, 2)

            # Extract restaurants for memory
            restaurants = result.get("results", []) or result.get("final_restaurants", [])

            # Map result to expected format
            return {
                "success": result.get("success", False),
                "ai_response": result.get("location_formatted_results", ""),
                "langchain_formatted_results": result.get("location_formatted_results", ""),
                "location_formatted_results": result.get("location_formatted_results", ""),
                "action_taken": "location_search",
                "search_triggered": True,
                "processing_time": processing_time,
                "coordinates": gps_coordinates,
                "source": result.get("source", "unknown"),
                "restaurant_count": result.get("restaurant_count", 0),
                "final_restaurants": restaurants
            }

        except Exception as e:
            logger.error(f"❌ Location search error: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "ai_response": "Sorry, there was an error searching. Please try again.",
                "location_formatted_results": "Sorry, there was an error searching. Please try again.",
                "action_taken": "location_search_error",
                "search_triggered": True,
                "processing_time": round(time.time() - start_time, 2)
            }

    @traceable(run_type="chain", name="execute_location_maps_search")
    async def _execute_location_maps_search(
        self,
        search_query: str,
        search_ctx: SearchContext,
        gps_coordinates: Optional[Tuple[float, float]],
        cancel_check_fn: Optional[Callable],
        start_time: float
    ) -> Dict[str, Any]:
        """Execute maps-only location search (skip database, go directly to Google Maps)

        Used when user requests "more results" after database results were shown.
        """
        self.stats["location_searches"] += 1

        logger.info(f"🗺️ Executing MAPS-ONLY location search: '{search_query}'")

        if not gps_coordinates:
            return {
                "success": True,
                "ai_response": "To search for more options, I need your location. Could you share it?",
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

            # Execute the LCEL pipeline with maps_only=True
            result = await self.location_pipeline.process_location_query(
                query=search_query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True  # KEY: Skip database, go directly to Google Maps
            )

            processing_time = round(time.time() - start_time, 2)

            # Extract restaurants for memory
            restaurants = result.get("results", []) or result.get("final_restaurants", [])

            return {
                "success": result.get("success", False),
                "ai_response": result.get("location_formatted_results", ""),
                "langchain_formatted_results": result.get("location_formatted_results", ""),
                "location_formatted_results": result.get("location_formatted_results", ""),
                "action_taken": "location_maps_search",
                "search_triggered": True,
                "processing_time": processing_time,
                "coordinates": gps_coordinates,
                "source": "google_maps",
                "restaurant_count": result.get("restaurant_count", 0),
                "final_restaurants": restaurants
            }

        except Exception as e:
            logger.error(f"❌ Location maps search error: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "ai_response": "Sorry, there was an error searching Google Maps. Please try again.",
                "location_formatted_results": "Sorry, there was an error searching. Please try again.",
                "action_taken": "location_maps_search_error",
                "search_triggered": True,
                "processing_time": round(time.time() - start_time, 2)
            }  

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _build_search_query(self, search_ctx: SearchContext) -> str:
        """Build a search query string from SearchContext

        NEW: Uses modified_query if AI Chat Layer provided one (e.g., user said
        "lunch not brunch" so modified_query would be "lunch restaurants")
        """
        # NEW: If AI provided a modified query, use it
        if search_ctx.modified_query:
            logger.info(f"🔄 Using AI-modified query: '{search_ctx.modified_query}'")
            return search_ctx.modified_query

        parts = []

        if search_ctx.cuisine:
            parts.append(search_ctx.cuisine)

        if search_ctx.requirements:
            parts.extend(search_ctx.requirements)

        if search_ctx.destination and search_ctx.search_type == SearchType.CITY_SEARCH:
            parts.append(f"in {search_ctx.destination}")

        if not parts:
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
        cancel_check_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for process_message."""
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

    def process_query(self, query: str, cancel_check_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Legacy method for direct city search (bypasses AI Chat Layer)."""
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
        """Legacy method for unified search."""
        logger.info(f"📜 Legacy search_restaurants called: '{query[:50]}...'")

        if gps_coordinates or location_data:
            if location_data:
                loc_data = location_data
            elif gps_coordinates:  # Explicit check narrows type for PyRight
                loc_data = LocationData(
                    latitude=gps_coordinates[0],
                    longitude=gps_coordinates[1],
                    description=f"GPS: {gps_coordinates[0]:.4f}, {gps_coordinates[1]:.4f}"
                )
            else:
                # Shouldn't reach here, but satisfies type checker
                raise ValueError("No GPS coordinates or location data provided")

            result = await self.location_pipeline.process_location_query(
                query=query,
                location_data=loc_data,
                cancel_check_fn=None,
                maps_only=False
            )

            return {
                "success": result.get("success", False),
                "formatted_message": result.get("location_formatted_results", ""),
                "langchain_formatted_results": result.get("location_formatted_results", ""),
                "final_restaurants": result.get("results", []),
                "search_flow": "location_search"
            }
        else:
            result = await self.city_pipeline.process_query_async(query)

            return {
                "success": result.get("success", False),
                "formatted_message": result.get("langchain_formatted_results", ""),
                "langchain_formatted_results": result.get("langchain_formatted_results", ""),
                "final_restaurants": result.get("enhanced_results", {}).get("main_list", []),
                "search_flow": "city_search"
            }

    async def process_user_message_with_ai_chat(
        self,
        query: str,
        user_id: int,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        telegram_bot=None,
        chat_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Legacy method name. Redirects to process_message."""
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
        """Legacy method name. Redirects to process_message."""
        return await self.process_message(
            query=query,
            user_id=user_id,
            gps_coordinates=gps_coordinates,
            thread_id=thread_id,
            telegram_bot=kwargs.get('telegram_bot'),
            chat_id=kwargs.get('chat_id')
        )

    # ============================================================================
    # STATISTICS
    # ============================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get supervisor statistics including memory operations"""
        return {
            "supervisor": self.stats,
            "city_pipeline": self.city_pipeline.get_stats(),
            "location_pipeline": self.location_pipeline.get_stats()
        }

    def get_info(self) -> Dict[str, Any]:
        """Get supervisor configuration info"""
        return {
            "type": "langgraph_supervisor",
            "version": "3.0",
            "architecture": "supervisor_with_memory_and_lcel_pipelines",
            "components": {
                "memory_system": "AIMemorySystem (Supabase)",
                "ai_chat_layer": "conversation management with context",
                "city_pipeline": self.city_pipeline.get_pipeline_info(),
                "location_pipeline": self.location_pipeline.get_pipeline_info()
            },
            "handoff_commands": [
                "CONTINUE_CONVERSATION",
                "EXECUTE_SEARCH",
                "RESUME_WITH_DECISION"
            ],
            "memory_features": [
                "user_preferences",
                "restaurant_history",
                "conversation_patterns",
                "preference_learning"
            ]
        }


# ============================================================================
# CLASS ALIAS (for backward compatibility)
# ============================================================================

UnifiedRestaurantAgent = LangGraphSupervisor


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_unified_restaurant_agent(config) -> LangGraphSupervisor:
    """Factory function to create the supervisor."""
    return LangGraphSupervisor(config)
