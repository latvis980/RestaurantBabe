# location_search_orchestrator.py
"""
LangChain LCEL Location Search Orchestrator

A clean, focused pipeline for location-based restaurant searches using LangChain LCEL.
Follows the same pattern as city_search_orchestrator.py for consistency.

PIPELINE FLOW:
1. Geocoding → Resolve location to coordinates (if needed)
2. Database Search → Proximity search in restaurant database
3. Filter Evaluation → AI filtering for relevance
4. [Branch Decision] → Based on sufficient_results:
   - IF sufficient: Description Editing → Formatting → Done
   - IF insufficient: Enhanced Verification (Maps + Media) → Description Generation → Formatting → Done
5. Formatting → Generate Telegram-ready output

FLOW MODES:
- database_flow: Sufficient database results → skip Google Maps
- maps_flow: Insufficient database → Google Maps + Media Verification
- maps_only: Skip database entirely (for "show more" requests)

CORRECTED METHOD NAMES (verified from project files):
✅ LocationUtils.geocode_location(description) → returns (lat, lng) tuple
✅ LocationDatabaseService.search_by_proximity(coordinates, radius_km, extract_descriptions) → returns list
✅ LocationFilterEvaluator.filter_and_evaluate(restaurants, query, location_description) → returns dict
✅ LocationAIEditor.create_descriptions_for_database_results(database_restaurants, user_query, cancel_check_fn) → async
✅ LocationAIEditor.create_descriptions_for_map_search_results(map_search_results, media_verification_results, user_query, cancel_check_fn) → async
✅ LocationMapSearchAgent.search_venues_with_ai_analysis(coordinates, query, cancel_check_fn) → async
✅ LocationMediaVerificationAgent.verify_venues_media_coverage(venues, query, cancel_check_fn) → async
✅ LocationTelegramFormatter.format_database_results(restaurants, query, location_description, offer_more_search)
✅ LocationTelegramFormatter.format_google_maps_results(venues, query, location_description)
"""

import logging
import asyncio
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableBranch
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain_core.tracers.langchain import wait_for_all_tracers

# Core location utilities
from location.location_utils import LocationUtils
from location.telegram_location_handler import LocationData

# Location-specific services
from location.database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_telegram_formatter import LocationTelegramFormatter

# AI Editors (facade pattern - delegates to specialized editors)
from location.location_ai_editor import LocationAIEditor

# Enhanced verification agents
from location.location_map_search import LocationMapSearchAgent
from location.location_media_verification import LocationMediaVerificationAgent

logger = logging.getLogger(__name__)


class LocationSearchOrchestrator:
    """
    LangChain LCEL-based orchestrator for location-based restaurant searches.
    
    This is a focused, deterministic pipeline that:
    - Uses LCEL RunnableSequence for clean step-by-step execution
    - Delegates ALL business logic to individual agents
    - Handles two flow modes: database_flow vs maps_flow
    - Provides full LangSmith tracing for debugging
    - Supports maps_only mode for "show more" requests
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for orchestration decisions
        self.ai = ChatOpenAI(
            model=getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Initialize location-specific services
        self.database_service = LocationDatabaseService(config)
        self.filter_evaluator = LocationFilterEvaluator(config)
        self.ai_editor = LocationAIEditor(config)  # Facade that delegates to specialized editors

        # Enhanced verification agents
        self.map_search_agent = LocationMapSearchAgent(config)
        self.media_verification_agent = LocationMediaVerificationAgent(config)

        # Formatter
        self.formatter = LocationTelegramFormatter(config)

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 2.0)
        self.min_db_matches = 2
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "flow_types": {"database": 0, "maps": 0, "maps_only": 0},
            "avg_processing_time": 0.0
        }

        # Build the LCEL pipeline
        self._build_pipeline()

        logger.info("✅ LocationSearchOrchestrator initialized with LangChain LCEL pipeline")

    def _build_pipeline(self):
        """Build the LangChain LCEL pipeline with conditional branching"""

        # Step 1: Geocoding
        self.geocoding_chain = RunnableLambda(
            self._geocode_step_traced,
            name="geocode_location"
        )

        # Step 2: Database Search
        self.database_search_chain = RunnableLambda(
            self._database_search_step_traced,
            name="database_proximity_search"
        )

        # Step 3: Filter Evaluation
        self.filter_chain = RunnableLambda(
            self._filter_evaluation_step_traced,
            name="ai_filter_evaluation"
        )

        # Step 4a: Description Editing (for database results)
        self.description_editing_chain = RunnableLambda(
            self._description_editing_step_traced,
            name="ai_description_editing"
        )

        # Step 4b: Enhanced Verification (Maps + Media)
        self.enhanced_verification_chain = RunnableLambda(
            self._enhanced_verification_step_traced,
            name="enhanced_verification"
        )

        # Step 5: Formatting
        self.formatting_chain = RunnableLambda(
            self._formatting_step_traced,
            name="telegram_formatting"
        )

        # Build the normal flow (database first, then branch)
        self.database_flow_sequence = (
            self.geocoding_chain |
            self.database_search_chain |
            self.filter_chain
        )

        # Branch after filtering: database sufficient OR maps needed
        self.post_filter_branch = RunnableBranch(
            # If sufficient database results → description editing → formatting
            (
                lambda x: x.get("sufficient_results", False) and len(x.get("filtered_restaurants", [])) >= self.min_db_matches,
                self.description_editing_chain | self.formatting_chain
            ),
            # Default: insufficient results → enhanced verification → formatting
            self.enhanced_verification_chain | self.formatting_chain
        )

        # Complete normal pipeline
        self.normal_pipeline = self.database_flow_sequence | self.post_filter_branch

        # Maps-only pipeline (skip database entirely, for "show more" requests)
        self.maps_only_pipeline = (
            self.geocoding_chain |
            self.enhanced_verification_chain |
            self.formatting_chain
        )

        # Top-level branch based on maps_only flag
        self.pipeline = RunnableBranch(
            # If maps_only=True → skip database, go directly to Google Maps
            (
                lambda x: x.get("maps_only", False),
                self.maps_only_pipeline
            ),
            # Default: normal flow with database search first
            self.normal_pipeline
        )

        logger.info("✅ LCEL pipeline built with database and maps-only branches")

    # ============================================================================
    # TRACED PIPELINE STEPS
    # ============================================================================

    @traceable(run_type="tool", name="geocode_location", metadata={"step": 1})
    async def _geocode_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Geocode location if needed (with tracing)"""
        return await self._geocode_step(pipeline_input)

    @traceable(run_type="retriever", name="database_proximity_search", metadata={"step": 2})
    async def _database_search_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Database proximity search (with tracing)"""
        return await self._database_search_step(pipeline_input)

    @traceable(run_type="llm", name="ai_filter_evaluation", metadata={"step": 3})
    async def _filter_evaluation_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: AI filter evaluation (with tracing)"""
        return await self._filter_evaluation_step(pipeline_input)

    @traceable(run_type="llm", name="ai_description_editing", metadata={"step": "4a"})
    async def _description_editing_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4a: AI description editing for database results (with tracing)"""
        return await self._description_editing_step(pipeline_input)

    @traceable(run_type="chain", name="enhanced_verification", metadata={"step": "4b"})
    async def _enhanced_verification_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4b: Enhanced verification - Maps + Media (with tracing)"""
        return await self._enhanced_verification_step(pipeline_input)

    @traceable(run_type="parser", name="telegram_formatting", metadata={"step": 5})
    async def _formatting_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Telegram formatting (with tracing)"""
        return await self._formatting_step(pipeline_input)

    # ============================================================================
    # PIPELINE STEP IMPLEMENTATIONS
    # ============================================================================

    async def _geocode_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Geocode location if needed
        
        Converts location description to coordinates if not already provided.
        Uses LocationUtils.geocode_location() which tries Nominatim then Google Maps.
        """
        try:
            location_data = pipeline_input.get("location_data")
            cancel_check_fn = pipeline_input.get("cancel_check_fn")

            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            logger.info("🌍 Step 1: Geocoding location")

            # Check if geocoding is needed
            if location_data:
                latitude = getattr(location_data, 'latitude', None)
                longitude = getattr(location_data, 'longitude', None)
                description = getattr(location_data, 'description', None)

                if (latitude is None or longitude is None) and description:
                    logger.info(f"📍 Geocoding location: '{description}'")

                    # Use LocationUtils.geocode_location() - returns (lat, lng) tuple
                    geocoded_coords = LocationUtils.geocode_location(description)

                    if geocoded_coords:
                        latitude = geocoded_coords[0]
                        longitude = geocoded_coords[1]
                        logger.info(f"✅ Geocoded to: {latitude:.4f}, {longitude:.4f}")
                    else:
                        raise ValueError(f"Failed to geocode: {description}")

                if latitude is None or longitude is None:
                    raise ValueError("Could not extract valid coordinates")

                coordinates = (float(latitude), float(longitude))
                location_desc = description or f"GPS: {latitude:.4f}, {longitude:.4f}"

            else:
                # Check for raw coordinates in pipeline_input
                coordinates = pipeline_input.get("coordinates")
                if not coordinates:
                    raise ValueError("No location data or coordinates provided")
                location_desc = f"GPS: {coordinates[0]:.4f}, {coordinates[1]:.4f}"

            logger.info(f"✅ Geocoding complete: {coordinates[0]:.4f}, {coordinates[1]:.4f}")

            return {
                **pipeline_input,
                "coordinates": coordinates,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "location_description": location_desc,
                "geocoding_success": True,
                "step_completed": "geocoding"
            }

        except Exception as e:
            logger.error(f"❌ Geocoding failed: {e}")
            raise ValueError(f"Geocoding failed: {str(e)}")

    async def _database_search_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Database Proximity Search
        
        Searches for restaurants within radius of coordinates.
        Delegates to LocationDatabaseService.search_by_proximity()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            coordinates = pipeline_input["coordinates"]

            logger.info(f"🗃️ Step 2: Database search within {self.db_search_radius}km")

            # Delegate to LocationDatabaseService (synchronous call)
            db_restaurants = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=self.db_search_radius,
                extract_descriptions=True
            )

            logger.info(f"✅ Database search complete: {len(db_restaurants)} restaurants found")

            return {
                **pipeline_input,
                "db_restaurants": db_restaurants,
                "db_restaurant_count": len(db_restaurants),
                "database_search_success": True,
                "step_completed": "database_search"
            }

        except Exception as e:
            logger.error(f"❌ Database search failed: {e}")
            # Don't fail entirely - continue to maps search
            return {
                **pipeline_input,
                "db_restaurants": [],
                "db_restaurant_count": 0,
                "database_search_success": False,
                "database_search_error": str(e),
                "step_completed": "database_search"
            }

    async def _filter_evaluation_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: AI Filter Evaluation
        
        Filters database results for relevance to query.
        Determines if database has sufficient results or if maps search is needed.
        Delegates to LocationFilterEvaluator.filter_and_evaluate()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            db_restaurants = pipeline_input.get("db_restaurants", [])
            query = pipeline_input.get("query", "restaurant")
            location_desc = pipeline_input.get("location_description", "")

            if not db_restaurants:
                logger.info("🤖 Step 3: No database restaurants to filter")
                return {
                    **pipeline_input,
                    "filtered_restaurants": [],
                    "filter_result": {"database_sufficient": False},
                    "sufficient_results": False,
                    "filtering_success": True,
                    "step_completed": "filter_evaluation"
                }

            logger.info(f"🤖 Step 3: AI filtering {len(db_restaurants)} database results")

            # Delegate to LocationFilterEvaluator (synchronous call)
            filter_result = self.filter_evaluator.filter_and_evaluate(
                restaurants=db_restaurants,
                query=query,
                location_description=location_desc
            )

            filtered_restaurants = filter_result.get("filtered_restaurants", [])
            sufficient_results = filter_result.get("database_sufficient", False)

            # Also check minimum count
            if len(filtered_restaurants) < self.min_db_matches:
                sufficient_results = False

            logger.info(f"✅ Filter complete: {len(filtered_restaurants)} passed, sufficient={sufficient_results}")

            return {
                **pipeline_input,
                "filtered_restaurants": filtered_restaurants,
                "filter_result": filter_result,
                "sufficient_results": sufficient_results,
                "filtering_success": True,
                "step_completed": "filter_evaluation"
            }

        except Exception as e:
            logger.error(f"❌ Filter evaluation failed: {e}")
            # Default to maps search on filter failure
            return {
                **pipeline_input,
                "filtered_restaurants": [],
                "filter_result": {"database_sufficient": False},
                "sufficient_results": False,
                "filtering_success": False,
                "filtering_error": str(e),
                "step_completed": "filter_evaluation"
            }

    async def _description_editing_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4a: AI Description Editing (for database results)
        
        Creates enhanced descriptions for database restaurants.
        Only called when database has sufficient results.
        Delegates to LocationAIEditor.create_descriptions_for_database_results()
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            filtered_restaurants = pipeline_input.get("filtered_restaurants", [])
            query = pipeline_input.get("query", "")

            if not filtered_restaurants:
                logger.info("✏️ Step 4a: No restaurants to edit descriptions for")
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "source": "database_empty",
                    "description_editing_success": True,
                    "step_completed": "description_editing"
                }

            logger.info(f"✏️ Step 4a: AI editing descriptions for {len(filtered_restaurants)} database restaurants")

            try:
                # Delegate to LocationAIEditor (async call)
                edited_restaurants = await self.ai_editor.create_descriptions_for_database_results(
                    database_restaurants=filtered_restaurants,
                    user_query=query,
                    cancel_check_fn=cancel_check_fn
                )

                # Convert dataclass objects to dicts if needed
                final_restaurants = []
                for restaurant in edited_restaurants:
                    if hasattr(restaurant, '__dict__'):
                        # It's a dataclass, convert to dict
                        restaurant_dict = {
                            'name': getattr(restaurant, 'name', 'Unknown'),
                            'address': getattr(restaurant, 'address', ''),
                            'google_maps_url': getattr(restaurant, 'maps_link', ''),
                            'distance_km': getattr(restaurant, 'distance_km', 0.0),
                            'description': getattr(restaurant, 'description', ''),
                            'sources': getattr(restaurant, 'sources', []),
                            'rating': getattr(restaurant, 'rating', None),
                            'user_ratings_total': getattr(restaurant, 'user_ratings_total', None),
                        }
                        final_restaurants.append(restaurant_dict)
                    else:
                        final_restaurants.append(restaurant)

                logger.info(f"✅ Description editing complete: {len(final_restaurants)} restaurants")

                return {
                    **pipeline_input,
                    "final_restaurants": final_restaurants,
                    "source": "database_with_editing",
                    "description_editing_success": True,
                    "step_completed": "description_editing"
                }

            except Exception as edit_error:
                logger.warning(f"⚠️ Description editing failed: {edit_error}")
                # Use filtered restaurants without enhanced descriptions
                return {
                    **pipeline_input,
                    "final_restaurants": filtered_restaurants,
                    "source": "database_without_editing",
                    "description_editing_success": False,
                    "description_editing_error": str(edit_error),
                    "step_completed": "description_editing"
                }

        except Exception as e:
            logger.error(f"❌ Description editing step failed: {e}")
            return {
                **pipeline_input,
                "final_restaurants": pipeline_input.get("filtered_restaurants", []),
                "source": "database_fallback",
                "description_editing_success": False,
                "step_completed": "description_editing"
            }

    async def _enhanced_verification_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4b: Enhanced Verification (Maps + Media)
        
        Called when database results are insufficient.
        Performs:
        1. Google Maps search via LocationMapSearchAgent
        2. Media verification via LocationMediaVerificationAgent
        3. AI description generation via LocationAIEditor
        """
        try:
            coordinates = pipeline_input["coordinates"]
            query = pipeline_input.get("query", "restaurant")
            location_desc = pipeline_input.get("location_description", "")
            cancel_check_fn = pipeline_input.get("cancel_check_fn")

            logger.info("🔍 Step 4b: Enhanced verification (Maps + Media)")

            # Sub-step 1: Google Maps search
            logger.info("🗺️ Sub-step 1: Google Maps search")

            map_venues = await self.map_search_agent.search_venues_with_ai_analysis(
                coordinates=coordinates,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "source": "cancelled",
                    "step_completed": "enhanced_verification"
                }

            logger.info(f"🗺️ Maps search found {len(map_venues)} venues")

            if not map_venues:
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "source": "maps_empty",
                    "enhanced_verification_metadata": {
                        "map_search_venues": 0,
                        "no_venues_found": True
                    },
                    "step_completed": "enhanced_verification"
                }

            # Sub-step 2: Media verification
            logger.info("📱 Sub-step 2: Media verification")

            try:
                media_verification_results = await self.media_verification_agent.verify_venues_media_coverage(
                    venues=map_venues[:self.max_venues_to_verify],
                    query=query,
                    cancel_check_fn=cancel_check_fn
                )
            except Exception as media_error:
                logger.warning(f"⚠️ Media verification failed: {media_error}")
                media_verification_results = []

            if cancel_check_fn and cancel_check_fn():
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "source": "cancelled",
                    "step_completed": "enhanced_verification"
                }

            logger.info(f"📱 Media verification complete: {len(media_verification_results)} results")

            # Sub-step 3: AI description generation
            logger.info("✏️ Sub-step 3: AI description generation")

            try:
                final_descriptions = await self.ai_editor.create_descriptions_for_map_search_results(
                    map_search_results=map_venues[:self.max_venues_to_verify],
                    media_verification_results=media_verification_results,
                    user_query=query,
                    cancel_check_fn=cancel_check_fn
                )

                # Convert dataclass objects to dicts if needed
                final_restaurants = []
                for restaurant in final_descriptions:
                    if hasattr(restaurant, '__dict__'):
                        restaurant_dict = {
                            'name': getattr(restaurant, 'name', 'Unknown'),
                            'address': getattr(restaurant, 'address', ''),
                            'google_maps_url': getattr(restaurant, 'maps_link', ''),
                            'place_id': getattr(restaurant, 'place_id', ''),
                            'distance_km': getattr(restaurant, 'distance_km', 0.0),
                            'description': getattr(restaurant, 'description', ''),
                            'sources': getattr(restaurant, 'sources', []),
                            'rating': getattr(restaurant, 'rating', None),
                            'user_ratings_total': getattr(restaurant, 'user_ratings_total', None),
                            'media_verified': len(getattr(restaurant, 'sources', [])) > 0,
                        }
                        final_restaurants.append(restaurant_dict)
                    else:
                        final_restaurants.append(restaurant)

                logger.info(f"✅ AI descriptions created for {len(final_restaurants)} venues")

                return {
                    **pipeline_input,
                    "final_restaurants": final_restaurants,
                    "source": "enhanced_verification",
                    "enhanced_verification_metadata": {
                        "map_search_venues": len(map_venues),
                        "media_verification_results": len(media_verification_results),
                        "final_descriptions": len(final_restaurants),
                        "agents_used": ["map_search", "media_verification", "ai_editor"]
                    },
                    "step_completed": "enhanced_verification"
                }

            except Exception as description_error:
                logger.warning(f"⚠️ AI description creation failed: {description_error}")
                # Return map venues without enhanced descriptions
                # Convert VenueSearchResult objects to dicts
                fallback_restaurants = []
                for venue in map_venues[:self.max_venues_to_verify]:
                    venue_dict = {
                        'name': getattr(venue, 'name', 'Unknown'),
                        'address': getattr(venue, 'address', ''),
                        'place_id': getattr(venue, 'place_id', ''),
                        'distance_km': getattr(venue, 'distance_km', 0.0),
                        'rating': getattr(venue, 'rating', None),
                        'user_ratings_total': getattr(venue, 'user_ratings_total', None),
                        'google_maps_url': f"https://www.google.com/maps/place/?q=place_id:{getattr(venue, 'place_id', '')}",
                    }
                    fallback_restaurants.append(venue_dict)

                return {
                    **pipeline_input,
                    "final_restaurants": fallback_restaurants,
                    "source": "maps_without_descriptions",
                    "enhanced_verification_metadata": {
                        "map_search_venues": len(map_venues),
                        "description_error": str(description_error)
                    },
                    "step_completed": "enhanced_verification"
                }

        except Exception as e:
            logger.error(f"❌ Enhanced verification failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                **pipeline_input,
                "final_restaurants": [],
                "source": "enhanced_verification_error",
                "enhanced_verification_metadata": {"error": str(e)},
                "step_completed": "enhanced_verification"
            }

    async def _formatting_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 5: Telegram Formatting
        
        Converts final restaurants to Telegram-ready formatted text.
        Uses different formatters based on source type.
        """
        try:
            final_restaurants = pipeline_input.get("final_restaurants", [])
            query = pipeline_input.get("query", "")
            coordinates = pipeline_input.get("coordinates")
            location_desc = pipeline_input.get("location_description", "")
            source = pipeline_input.get("source", "unknown")

            logger.info(f"📱 Step 5: Formatting {len(final_restaurants)} restaurants (source: {source})")

            if not final_restaurants:
                return {
                    **pipeline_input,
                    "success": False,
                    "location_formatted_results": "😔 No restaurants found matching your criteria.",
                    "restaurant_count": 0,
                    "results": [],
                    "step_completed": "formatting"
                }

            try:
                # Choose formatter based on source
                if source.startswith("database"):
                    # For database results, use format_database_results
                    formatted_results = self.formatter.format_database_results(
                        restaurants=final_restaurants,
                        query=query,
                        location_description=location_desc,
                        offer_more_search=True
                    )
                else:
                    # For maps results, use format_google_maps_results
                    formatted_results = self.formatter.format_google_maps_results(
                        venues=final_restaurants,
                        query=query,
                        location_description=location_desc
                    )

                formatted_message = formatted_results.get("message", "") if isinstance(formatted_results, dict) else str(formatted_results)

            except Exception as format_error:
                logger.warning(f"⚠️ Formatting method failed: {format_error}")
                # Basic fallback formatting
                formatted_message = f"Found {len(final_restaurants)} restaurants:\n\n"
                for i, restaurant in enumerate(final_restaurants[:5], 1):
                    name = restaurant.get('name', 'Unknown')
                    address = restaurant.get('address', 'No address')
                    rating = restaurant.get('rating')
                    rating_text = f" (★{rating})" if rating else ""
                    formatted_message += f"{i}. {name}{rating_text}\n   📍 {address}\n\n"

            logger.info(f"✅ Formatting complete: {len(formatted_message)} chars")

            return {
                **pipeline_input,
                "success": True,
                "location_formatted_results": formatted_message,
                "restaurant_count": len(final_restaurants),
                "results": final_restaurants,
                "source": source,
                "formatting_success": True,
                "step_completed": "formatting"
            }

        except Exception as e:
            logger.error(f"❌ Formatting failed: {e}")
            return {
                **pipeline_input,
                "success": False,
                "location_formatted_results": f"😔 Error formatting results: {str(e)}",
                "restaurant_count": 0,
                "results": [],
                "formatting_success": False,
                "formatting_error": str(e),
                "step_completed": "formatting"
            }

    # ============================================================================
    # PUBLIC API
    # ============================================================================

    @traceable(run_type="chain", name="location_search_pipeline", metadata={"pipeline_type": "langchain_lcel"})
    async def process_location_query_async(
        self,
        query: str,
        location_data: LocationData,
        cancel_check_fn: Optional[callable] = None,
        maps_only: bool = False
    ) -> Dict[str, Any]:
        """
        Process a location search query through the LCEL pipeline (async version)
        
        Args:
            query: User's search query
            location_data: LocationData object with coordinates/description
            cancel_check_fn: Optional function to check if search should be cancelled
            maps_only: If True, skip database and go directly to Google Maps
            
        Returns:
            Dict with location_formatted_results and metadata
        """
        start_time = time.time()

        try:
            flow_type = "MAPS-ONLY" if maps_only else "NORMAL"
            logger.info(f"🚀 Starting location search pipeline: '{query[:50]}...' | Flow: {flow_type}")

            # Prepare pipeline input
            pipeline_input = {
                "query": query,
                "raw_query": query,
                "location_data": location_data,
                "cancel_check_fn": cancel_check_fn,
                "maps_only": maps_only,
                "start_time": start_time
            }

            # Execute the pipeline with tracing
            result = await self.pipeline.ainvoke(
                pipeline_input,
                config={
                    "run_name": f"location_search_{{query='{query[:30]}...'}}",
                    "metadata": {
                        "user_query": query,
                        "location_type": getattr(location_data, 'location_type', 'unknown') if location_data else 'raw',
                        "maps_only": maps_only,
                        "pipeline_version": "lcel_v1.0"
                    },
                    "tags": ["location_search", "lcel_pipeline"]
                }
            )

            # Add timing metadata
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            result["pipeline_type"] = "langchain_lcel"

            # Flush traces
            try:
                wait_for_all_tracers()
            except Exception as flush_error:
                logger.warning(f"⚠️ Failed to flush traces: {flush_error}")

            # Update statistics
            self._update_stats(result, processing_time, maps_only)

            logger.info(f"✅ Location search pipeline complete in {processing_time}s")

            return result

        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            processing_time = round(time.time() - start_time, 2)

            return {
                "success": False,
                "error": str(e),
                "location_formatted_results": f"😔 Location search failed: {str(e)}",
                "restaurants": [],
                "restaurant_count": 0,
                "processing_time": processing_time,
                "pipeline_type": "langchain_lcel"
            }

    def process_location_query(
        self,
        query: str,
        location_data: LocationData,
        cancel_check_fn: Optional[callable] = None,
        maps_only: bool = False
    ) -> Dict[str, Any]:
        """
        Process a location search query (synchronous wrapper)
        
        This is the main entry point for the Telegram bot and LangGraph supervisor.
        """
        start_time = time.time()

        try:
            # Run async pipeline in new event loop
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.process_location_query_async(query, location_data, cancel_check_fn, maps_only)
                    )
                finally:
                    loop.close()

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(run_async).result()

            return result

        except Exception as e:
            logger.error(f"❌ Synchronous pipeline error: {e}")

            return {
                "success": False,
                "error": str(e),
                "location_formatted_results": f"😔 Location search failed: {str(e)}",
                "restaurants": [],
                "restaurant_count": 0,
                "processing_time": round(time.time() - start_time, 2),
                "pipeline_type": "langchain_lcel"
            }

    # ============================================================================
    # LEGACY COMPATIBILITY METHODS
    # ============================================================================

    async def search_and_verify_more_results(
        self,
        query: str,
        coordinates: Tuple[float, float],
        exclude_places: Optional[List[str]] = None,
        cancel_check_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Legacy method: Search for more results with exclusions
        Used by "show more" functionality
        """
        try:
            logger.info(f"🔍 Searching for more results: '{query}'")

            # Create a minimal LocationData object
            from location.telegram_location_handler import LocationData
            location_data = LocationData(
                latitude=coordinates[0],
                longitude=coordinates[1],
                description=f"GPS: {coordinates[0]:.4f}, {coordinates[1]:.4f}"
            )

            # Use maps_only flow
            result = await self.process_location_query_async(
                query=query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True
            )

            # Apply exclusions if provided
            if exclude_places and result.get("results"):
                filtered_restaurants = []
                for restaurant in result["results"]:
                    restaurant_name = restaurant.get("name", "").lower()
                    if not any(excluded.lower() in restaurant_name for excluded in exclude_places):
                        filtered_restaurants.append(restaurant)

                result["results"] = filtered_restaurants
                result["restaurant_count"] = len(filtered_restaurants)

                # Re-format if we filtered some out
                if len(filtered_restaurants) < len(result.get("results", [])):
                    try:
                        formatted = self.formatter.format_google_maps_results(
                            venues=filtered_restaurants,
                            query=query,
                            location_description=f"GPS: {coordinates[0]:.4f}, {coordinates[1]:.4f}"
                        )
                        result["location_formatted_results"] = formatted.get("message", "")
                    except Exception:
                        pass

            return result

        except Exception as e:
            logger.error(f"More results query error: {e}")
            return {
                "success": False,
                "error": str(e),
                "location_formatted_results": f"😔 More results search failed: {str(e)}",
                "restaurant_count": 0,
                "results": [],
                "coordinates": coordinates
            }

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _update_stats(self, result: Dict[str, Any], processing_time: float, maps_only: bool):
        """Update orchestrator statistics"""
        self.stats["total_queries"] += 1

        # Update average processing time
        current_avg = self.stats["avg_processing_time"]
        total = self.stats["total_queries"]
        self.stats["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total

        # Track flow type
        if maps_only:
            self.stats["flow_types"]["maps_only"] += 1
        elif result.get("source", "").startswith("database"):
            self.stats["flow_types"]["database"] += 1
        else:
            self.stats["flow_types"]["maps"] += 1

        # Track success
        if result.get("success", False):
            self.stats["successful_queries"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "orchestrator": self.stats,
            "pipeline_type": "langchain_lcel"
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "pipeline_type": "langchain_lcel",
            "steps": [
                "geocoding",
                "database_proximity_search",
                "ai_filter_evaluation",
                "description_editing (database path)",
                "enhanced_verification (maps path)",
                "telegram_formatting"
            ],
            "flow_modes": ["database_flow", "maps_flow", "maps_only"],
            "agents": [
                "LocationDatabaseService",
                "LocationFilterEvaluator",
                "LocationAIEditor (facade)",
                "LocationMapSearchAgent",
                "LocationMediaVerificationAgent",
                "LocationTelegramFormatter"
            ],
            "settings": {
                "db_search_radius_km": self.db_search_radius,
                "min_db_matches": self.min_db_matches,
                "max_venues_to_verify": self.max_venues_to_verify
            },
            "tracing_enabled": True,
            "version": "1.0"
        }
