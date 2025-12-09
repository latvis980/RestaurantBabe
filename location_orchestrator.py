# location/location_orchestrator.py
"""
LangChain LCEL Location Search Orchestrator - MERGED VERSION

Combines:
- Clean LCEL pipeline structure from LocationSearchOrchestrator
- Legacy compatibility methods for telegram_bot.py
- Direct AI editor imports (not the deprecated wrapper)

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

DIRECT AI EDITORS (not using deprecated wrapper):
- LocationDatabaseAIEditor for database results
- LocationMapSearchAIEditor for map search results
"""

import logging
import asyncio
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Callable

from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain_core.tracers.langchain import wait_for_all_tracers

# Core location utilities
from location.location_utils import LocationUtils
from location.telegram_location_handler import LocationData

# Location-specific services
from location.location_database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from formatters.location_telegram_formatter import LocationTelegramFormatter

# DIRECT AI Editors (not using deprecated LocationAIEditor wrapper)
from location.location_database_ai_editor import LocationDatabaseAIEditor
from location.location_map_search_ai_editor import LocationMapSearchAIEditor

# Enhanced verification agents
from location.location_map_search import LocationMapSearchAgent
from location.location_media_verification import LocationMediaVerificationAgent
from agents.follow_up_search_agent import FollowUpSearchAgent

logger = logging.getLogger(__name__)


class LocationOrchestrator:
    """
    LangChain LCEL-based orchestrator for location-based restaurant searches.
    
    MERGED VERSION with:
    - Clean LCEL pipeline structure
    - Direct AI editors (LocationDatabaseAIEditor, LocationMapSearchAIEditor)
    - Legacy compatibility methods for telegram_bot.py
    - Full LangSmith tracing
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

        # DIRECT AI Editors (not using deprecated wrapper)
        self.database_ai_editor = LocationDatabaseAIEditor(config)
        self.map_ai_editor = LocationMapSearchAIEditor(config)

        # Enhanced verification agents
        self.map_search_agent = LocationMapSearchAgent(config)
        self.media_verification_agent = LocationMediaVerificationAgent(config)

        # Formatter
        self.formatter = LocationTelegramFormatter(config)

        # Verification agent for database results
        self.verification_agent = FollowUpSearchAgent(config)

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 1.5)  # Default 1.5km
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

        logger.info("✅ LocationOrchestrator initialized with DIRECT AI EDITORS and LCEL pipeline")

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

        # Step 3b: Verification (NEW - adds place_id, filters closed/low-rated)
        self.verification_chain = RunnableLambda(
            self._verification_step_traced,
            name="database_verification"
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

        # ✅ DEFINE maps_only_pipeline FIRST (before it's used)
        self.maps_only_pipeline = (
            self.geocoding_chain |
            self.enhanced_verification_chain |
            self.formatting_chain
        )

        # Branch after filtering: database sufficient OR maps needed
        self.post_filter_branch = RunnableBranch(
            # If sufficient database results → verification → description editing → formatting
            (
                lambda x: isinstance(x, dict) and x.get("sufficient_results", False),
                self.verification_chain | self.description_editing_chain | self.formatting_chain
            ),
            # Default: insufficient results → enhanced verification → formatting
            self.enhanced_verification_chain | self.formatting_chain
        )

        # Complete normal pipeline
        self.normal_pipeline = self.database_flow_sequence | self.post_filter_branch

        # Top-level branch based on maps_only flag
        self.pipeline = RunnableBranch(
            # If maps_only=True → skip database, go directly to Google Maps
            (
                lambda x: isinstance(x, dict) and x.get("maps_only", False),
                self.maps_only_pipeline  # ✅ Now it's defined!
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

            # Use dynamic radius from pipeline_input (AI-determined or default)
            search_radius = pipeline_input.get("search_radius_km", self.db_search_radius)

            logger.info(f"🗃️ Step 2: Database search within {search_radius}km")

            # Delegate to LocationDatabaseService (synchronous call)
            db_restaurants = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=search_radius,
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
        Uses LocationDatabaseAIEditor.create_descriptions_for_database_results() DIRECTLY
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
                # DIRECT call to LocationDatabaseAIEditor (not wrapper)
                edited_restaurants = await self.database_ai_editor.create_descriptions_for_database_results(
                    database_restaurants=filtered_restaurants,
                    user_query=query,
                    cancel_check_fn=cancel_check_fn
                )

                # Convert dataclass objects to dicts if needed
                final_restaurants = []
                for restaurant in edited_restaurants:
                    if hasattr(restaurant, '__dict__') and hasattr(restaurant, 'name'):
                        # It's a dataclass, convert to dict
                        restaurant_dict = {
                            'name': getattr(restaurant, 'name', 'Unknown'),
                            'address': getattr(restaurant, 'address', ''),
                            'maps_link': getattr(restaurant, 'maps_link', ''),
                            'distance_km': getattr(restaurant, 'distance_km', 0.0),
                            'description': getattr(restaurant, 'description', ''),
                            'sources': getattr(restaurant, 'sources', []),
                            'rating': getattr(restaurant, 'rating', None),
                            'user_ratings_total': getattr(restaurant, 'user_ratings_total', None),
                        }
                        final_restaurants.append(restaurant_dict)
                    elif isinstance(restaurant, dict):
                        final_restaurants.append(restaurant)
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

    # ============ NEW: VERIFICATION STEP ============

    @traceable(
        run_type="tool",
        name="database_verification",
        metadata={"step": "verification", "component": "follow_up_search_agent"}
    )
    async def _verification_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Traced wrapper for verification step"""
        return await self._verification_step(pipeline_input)

    async def _verification_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3b: Verify Database Restaurants via Google Maps

        This step runs AFTER filtering but BEFORE description editing.
        Uses FollowUpSearchAgent to:
        1. Add place_id from Google Maps (fixes link formation)
        2. Add proper google_maps_url with place_id
        3. Filter out closed restaurants (CLOSED_TEMPORARILY/PERMANENTLY)
        4. Filter by rating threshold (default 4.1)
        5. Save updated geodata back to database

        This ensures all database results have proper Google Maps links
        and are verified as open/quality restaurants.
        """
        try:
            cancel_check_fn = pipeline_input.get("cancel_check_fn")
            if cancel_check_fn and cancel_check_fn():
                raise ValueError("Search cancelled by user")

            filtered_restaurants = pipeline_input.get("filtered_restaurants", [])
            location_desc = pipeline_input.get("location_description", "")

            if not filtered_restaurants:
                logger.info("✅ No restaurants to verify - skipping verification step")
                return {
                    **pipeline_input,
                    "verified_restaurants": [],
                    "verification_success": True,
                    "verification_stats": {"total": 0, "verified": 0, "rejected": 0},
                    "step_completed": "verification"
                }

            logger.info(f"🔍 Step 3b: Verifying {len(filtered_restaurants)} restaurants via Google Maps")

            # Convert to format expected by follow_up_search_agent
            edited_results = {"main_list": filtered_restaurants}

            # Run verification through follow_up_search_agent
            # This uses _verify_and_filter_restaurant for each restaurant
            verification_result = self.verification_agent.perform_follow_up_searches(
                edited_results=edited_results,
                destination=location_desc
            )

            verified_restaurants = verification_result.get("enhanced_results", {}).get("main_list", [])

            # Calculate stats
            original_count = len(filtered_restaurants)
            verified_count = len(verified_restaurants)
            rejected_count = original_count - verified_count

            logger.info(f"✅ Verification complete: {verified_count}/{original_count} passed")
            logger.info(f"   - Verified: {verified_count}")
            logger.info(f"   - Rejected (closed/low rating): {rejected_count}")

            # Log place_id additions for debugging
            place_id_count = sum(1 for r in verified_restaurants if r.get("place_id"))
            logger.info(f"   - With place_id: {place_id_count}/{verified_count}")

            return {
                **pipeline_input,
                "filtered_restaurants": verified_restaurants,  # Replace with verified list
                "verified_restaurants": verified_restaurants,
                "verification_success": True,
                "verification_stats": {
                    "total": original_count,
                    "verified": verified_count,
                    "rejected": rejected_count,
                    "with_place_id": place_id_count
                },
                "step_completed": "verification"
            }

        except Exception as e:
            logger.error(f"❌ Verification step failed: {e}")
            # Don't fail the pipeline - continue with unverified restaurants
            logger.warning("⚠️ Continuing with unverified restaurants")
            return {
                **pipeline_input,
                "verified_restaurants": pipeline_input.get("filtered_restaurants", []),
                "verification_success": False,
                "verification_error": str(e),
                "step_completed": "verification"
            }
    
    async def _enhanced_verification_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4b: Enhanced Verification (Maps + Media)
        
        Called when database results are insufficient.
        Performs:
        1. Google Maps search via LocationMapSearchAgent
        2. Media verification via LocationMediaVerificationAgent
        3. AI description generation via LocationMapSearchAIEditor DIRECTLY
        """
        try:
            coordinates = pipeline_input["coordinates"]
            query = pipeline_input.get("query", "restaurant")
            location_desc = pipeline_input.get("location_description", "")
            cancel_check_fn = pipeline_input.get("cancel_check_fn")

            logger.info("🔍 Step 4b: Enhanced verification (Maps + Media)")

            # Sub-step 1: Google Maps search
            logger.info("🗺️ Sub-step 1: Google Maps search")

            search_radius = pipeline_input.get("search_radius_km", self.db_search_radius)

            map_venues = await self.map_search_agent.search_venues_with_ai_analysis(
                coordinates=coordinates,
                query=query,
                cancel_check_fn=cancel_check_fn,
                search_radius_km=search_radius
            )

            if cancel_check_fn and cancel_check_fn():
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "source": "cancelled",
                    "step_completed": "enhanced_verification"
                }

            logger.info(f"🗺️ Maps search found {len(map_venues)} venues")

            # Sub-step 1.5: Enrich with reviews
            if map_venues:
                logger.info("📖 Sub-step 1.5: Enriching venues with Google reviews")
                enriched_venues = await self._enrich_venues_with_reviews(map_venues[:10])
                map_venues[:len(enriched_venues)] = enriched_venues

                venues_with_reviews = sum(1 for v in map_venues if hasattr(v, 'google_reviews') and v.google_reviews)
                total_reviews = sum(len(v.google_reviews) for v in map_venues if hasattr(v, 'google_reviews'))
                logger.info(f"✅ {venues_with_reviews}/{len(map_venues)} venues have reviews ({total_reviews} total)")
            
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

            # Sub-step 2: Media verification (conditional based on config)
            media_verification_results = []

            if getattr(self.config, 'ENABLE_MEDIA_VERIFICATION', False):
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
            else:
                logger.info("📱 Sub-step 2: Media verification SKIPPED (disabled in config)")

            if cancel_check_fn and cancel_check_fn():
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "source": "cancelled",
                    "step_completed": "enhanced_verification"
                }

            logger.info(f"📱 Media verification complete: {len(media_verification_results)} results")

            # Sub-step 3: AI description generation using LocationMapSearchAIEditor DIRECTLY
            # Sub-step 3: AI description generation using LocationMapSearchAIEditor DIRECTLY
            logger.info("✏️ Sub-step 3: AI description generation")

            # NEW: Get supervisor instructions for context-aware filtering
            supervisor_instructions = pipeline_input.get("supervisor_instructions")
            exclude_restaurants = pipeline_input.get("exclude_restaurants", [])

            # NEW: Apply exclusions BEFORE AI processing
            venues_to_process = map_venues[:self.max_venues_to_verify]
            if exclude_restaurants:
                original_count = len(venues_to_process)
                venues_to_process = [
                    v for v in venues_to_process 
                    if not any(
                        excluded.lower() in (getattr(v, 'name', '') or '').lower() 
                        for excluded in exclude_restaurants
                    )
                ]
                excluded_count = original_count - len(venues_to_process)
                if excluded_count > 0:
                    logger.info(f"🚫 Excluded {excluded_count} previously shown restaurants")

            try:
                # DIRECT call to LocationMapSearchAIEditor (not wrapper)
                # NEW: Pass supervisor_instructions for context-aware descriptions
                final_descriptions = await self.map_ai_editor.create_descriptions_for_map_search_results(
                    map_search_results=venues_to_process,
                    media_verification_results=media_verification_results,
                    user_query=query,
                    cancel_check_fn=cancel_check_fn,
                    supervisor_instructions=supervisor_instructions  # NEW
                )

                # Convert dataclass objects to dicts if needed
                final_restaurants = []
                for restaurant in final_descriptions:
                    if hasattr(restaurant, '__dict__') and hasattr(restaurant, 'name'):
                        restaurant_dict = {
                            'name': getattr(restaurant, 'name', 'Unknown'),
                            'address': getattr(restaurant, 'address', ''),
                            'maps_link': getattr(restaurant, 'maps_link', ''),
                            'place_id': getattr(restaurant, 'place_id', ''),
                            'distance_km': getattr(restaurant, 'distance_km', 0.0),
                            'description': getattr(restaurant, 'description', ''),
                            'media_sources': getattr(restaurant, 'media_sources', []),
                            'rating': getattr(restaurant, 'rating', None),
                            'user_ratings_total': getattr(restaurant, 'user_ratings_total', None),
                            'media_verified': len(getattr(restaurant, 'media_sources', [])) > 0,
                        }
                        final_restaurants.append(restaurant_dict)
                    elif isinstance(restaurant, dict):
                        final_restaurants.append(restaurant)
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
                        "agents_used": ["map_search", "media_verification", "map_ai_editor"]
                    },
                    "step_completed": "enhanced_verification"
                }

            except Exception as description_error:
                logger.warning(f"⚠️ AI description creation failed: {description_error}")
                # Return map venues without enhanced descriptions
                fallback_restaurants = []
                for venue in map_venues[:self.max_venues_to_verify]:
                    if hasattr(venue, 'name'):
                        venue_dict = {
                            'name': getattr(venue, 'name', 'Unknown'),
                            'address': getattr(venue, 'address', ''),
                            'place_id': getattr(venue, 'place_id', ''),
                            'distance_km': getattr(venue, 'distance_km', 0.0),
                            'rating': getattr(venue, 'rating', None),
                            'user_ratings_total': getattr(venue, 'user_ratings_total', None),
                            'maps_link': f"https://www.google.com/maps/place/?q=place_id:{getattr(venue, 'place_id', '')}",
                        }
                        fallback_restaurants.append(venue_dict)
                    elif isinstance(venue, dict):
                        fallback_restaurants.append(venue)

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

    async def _enrich_venues_with_reviews(self, venues: List) -> List:
        """Fetch Google reviews for venues"""
        if not venues:
            return venues

        logger.info(f"📖 Enriching {len(venues)} venues with reviews")
        enriched = []

        for venue in venues:
            try:
                if venue.place_id and self.map_search_agent.gmaps:
                    details = self.map_search_agent.gmaps.place(  # type: ignore[attr-defined]
                        place_id=venue.place_id,
                        fields=['reviews']
                    )
                    if details and 'result' in details:
                        venue.google_reviews = details['result'].get('reviews', [])
                        if venue.google_reviews:
                            logger.info(f"✅ {venue.name}: {len(venue.google_reviews)} reviews")
                enriched.append(venue)
            except Exception as e:
                logger.warning(f"Failed: {venue.name}: {e}")
                enriched.append(venue)

        return enriched

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
                    name = restaurant.get('name', 'Unknown') if isinstance(restaurant, dict) else getattr(restaurant, 'name', 'Unknown')
                    address = restaurant.get('address', 'No address') if isinstance(restaurant, dict) else getattr(restaurant, 'address', 'No address')
                    rating = restaurant.get('rating') if isinstance(restaurant, dict) else getattr(restaurant, 'rating', None)
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
    # PUBLIC API - MAIN ENTRY POINTS
    # ============================================================================

    async def process_location_query(
        self,
        query: str,
        location_data: LocationData,
        cancel_check_fn: Optional[Callable] = None,
        maps_only: bool = False,
        # Follow-up context from supervisor
        supervisor_instructions: Optional[str] = None,
        exclude_restaurants: Optional[List[str]] = None,
        # Dynamic search radius (AI-determined)
        search_radius_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a location search query through the LCEL pipeline (ASYNC)

        This is the main entry point for location searches.

        Args:
            query: User's search query
            location_data: LocationData object with coordinates/description
            cancel_check_fn: Optional function to check if search should be cancelled
            maps_only: If True, skip database and go directly to Google Maps
            supervisor_instructions: Natural language instructions from AI Chat Layer
                                     for filtering/editing (e.g., "User wants lunch not brunch")
            exclude_restaurants: List of restaurant names to exclude (already shown)

        Returns:
            Dict with location_formatted_results and metadata
        """
        start_time = time.time()

        try:
            flow_type = "MAPS-ONLY" if maps_only else "NORMAL"
            logger.info(f"🚀 Starting location search pipeline: '{query[:50]}...' | Flow: {flow_type}")

            # Determine effective search radius
            effective_radius = search_radius_km if search_radius_km is not None else self.db_search_radius
            logger.info(f"📏 Search radius: {effective_radius}km" + (" (AI-specified)" if search_radius_km else " (default)"))

            # Log supervisor context if present
            if supervisor_instructions:
                logger.info(f"📋 Supervisor instructions: {supervisor_instructions[:100]}...")
            if exclude_restaurants:
                logger.info(f"🚫 Excluding {len(exclude_restaurants)} restaurants")


            pipeline_input = {
                "query": query,
                "raw_query": query,
                "location_data": location_data,
                "cancel_check_fn": cancel_check_fn,
                "maps_only": maps_only,
                "start_time": start_time,
                # Follow-up context
                "supervisor_instructions": supervisor_instructions,
                "exclude_restaurants": exclude_restaurants or [],
                # Dynamic search radius
                "search_radius_km": effective_radius
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
                        "pipeline_version": "merged_v1.0"
                    },
                    "tags": ["location_search", "lcel_pipeline"]
                }
            )

            # Add timing metadata
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            result["pipeline_type"] = "langchain_lcel"
            
            result["search_radius_km"] = effective_radius

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

    # ============================================================================
    # LEGACY COMPATIBILITY METHODS (for telegram_bot.py)
    # ============================================================================

    async def process_more_results_query(
        self,
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn: Optional[Callable] = None,
        # Follow-up context from supervisor
        supervisor_instructions: Optional[str] = None,
        exclude_restaurants: Optional[List[str]] = None,
        # Dynamic search radius
        search_radius_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a "more results" query - goes directly to Google Maps

        This is called by telegram_bot.py when user wants more results 
        after seeing database results.

        Args:
            query: User's search query
            coordinates: GPS coordinates tuple (lat, lng)
            location_desc: Description of the location
            cancel_check_fn: Function to check if search should be cancelled
            supervisor_instructions: Natural language instructions from AI Chat Layer
            exclude_restaurants: List of restaurant names to exclude (already shown)

        Returns:
            Dict with search results and metadata
        """
        try:
            logger.info(f"🔍 Processing 'more results' query: '{query}' (maps-only)")

            # NEW: Log context
            if supervisor_instructions:
                logger.info(f"📋 With supervisor instructions: {supervisor_instructions[:80]}...")
            if exclude_restaurants:
                logger.info(f"🚫 Excluding {len(exclude_restaurants)} restaurants: {exclude_restaurants[:3]}...")

            # Create a LocationData object for the pipeline
            location_data = LocationData(
                latitude=coordinates[0],
                longitude=coordinates[1],
                description=location_desc,
                location_type="gps"
            )

            # Use the maps-only flow (skip database) with follow-up context
            result = await self.process_location_query(
                query=query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True,
                supervisor_instructions=supervisor_instructions,
                exclude_restaurants=exclude_restaurants,
                search_radius_km=search_radius_km
            )

            return result

        except Exception as e:
            logger.error(f"❌ More results query error: {e}")
            return {
                "success": False,
                "error": str(e),
                "location_formatted_results": f"😔 More results search failed: {str(e)}",
                "restaurant_count": 0,
                "results": [],
                "coordinates": coordinates
            }

    async def complete_media_verification(
        self,
        venues: List[Any],
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        LEGACY METHOD: Complete media verification for venues
        
        Maintained for backward compatibility with existing telegram_bot.py code.
        Redirects to enhanced verification flow.
        
        Args:
            venues: List of venues to verify (may be ignored, we search fresh)
            query: User's search query
            coordinates: GPS coordinates tuple
            location_desc: Description of the location
            cancel_check_fn: Function to check if search should be cancelled
            
        Returns:
            Dict with verified results
        """
        logger.info("Legacy complete_media_verification called - redirecting to enhanced flow")

        try:
            # Create a LocationData object
            location_data = LocationData(
                latitude=coordinates[0],
                longitude=coordinates[1],
                description=location_desc,
                location_type="gps"
            )

            # Use maps-only flow for verification
            result = await self.process_location_query(
                query=query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True
            )

            return result

        except Exception as e:
            logger.error(f"❌ Media verification error: {e}")
            return {
                "success": False,
                "error": str(e),
                "location_formatted_results": f"😔 Verification failed: {str(e)}",
                "restaurant_count": 0,
                "results": [],
                "coordinates": coordinates
            }

    async def search_and_verify_more_results(
        self,
        query: str,
        coordinates: Tuple[float, float],
        exclude_places: Optional[List[str]] = None,
        cancel_check_fn: Optional[Callable] = None,
        # NEW: Supervisor instructions for smarter filtering
        supervisor_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Legacy method: Search for more results with exclusions
        Used by "show more" functionality

        Args:
            query: User's search query
            coordinates: GPS coordinates tuple
            exclude_places: List of place names to exclude from results
            cancel_check_fn: Function to check if search should be cancelled
            supervisor_instructions: Natural language instructions for filtering

        Returns:
            Dict with search results
        """
        try:
            logger.info(f"🔍 Searching for more results: '{query}'")

            # Create a LocationData object
            location_data = LocationData(
                latitude=coordinates[0],
                longitude=coordinates[1],
                description=f"GPS: {coordinates[0]:.4f}, {coordinates[1]:.4f}",
                location_type="gps"
            )

            # Use maps_only flow with exclusions and supervisor instructions
            result = await self.process_location_query(
                query=query,
                location_data=location_data,
                cancel_check_fn=cancel_check_fn,
                maps_only=True,
                supervisor_instructions=supervisor_instructions,
                exclude_restaurants=exclude_places
            )

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

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics (legacy method name)"""
        return {
            'pipeline_type': 'langchain_lcel_merged',
            'database_service': True,
            'database_ai_editor': self.database_ai_editor is not None,
            'database_ai_editor_type': 'LocationDatabaseAIEditor',
            'map_ai_editor': self.map_ai_editor is not None,
            'map_ai_editor_type': 'LocationMapSearchAIEditor',
            'min_db_matches_trigger': self.min_db_matches,
            'langsmith_tracing_enabled': True,
            'stats': self.stats,
            'chain_components': [
                'geocoding_chain',
                'database_search_chain',
                'filter_chain',
                'description_editing_chain',
                'enhanced_verification_chain',
                'formatting_chain'
            ]
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "pipeline_type": "langchain_lcel_merged",
            "steps": [
                "geocoding",
                "database_proximity_search",
                "ai_filter_evaluation",
                "description_editing (database path)",
                "enhanced_verification (maps path)",
                "telegram_formatting"
            ],
            "flow_modes": ["database_flow", "maps_flow", "maps_only"],
            "ai_editors": {
                "database": "LocationDatabaseAIEditor (direct)",
                "maps": "LocationMapSearchAIEditor (direct)"
            },
            "agents": [
                "LocationDatabaseService",
                "LocationFilterEvaluator",
                "LocationDatabaseAIEditor",
                "LocationMapSearchAIEditor",
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
            "version": "merged_1.0"
        }
