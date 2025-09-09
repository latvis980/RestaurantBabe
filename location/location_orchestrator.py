# location/location_orchestrator.py
"""
Complete LangChain-Powered Location Search Orchestrator - ALL METHOD NAMES FIXED

Fixed Issues:
1. FIXED: wait_for_all_tracers() is not async - removed await
2. FIXED: All method names based on actual project files:
   - LocationMapSearchAgent.search_venues_with_ai_analysis()
   - LocationMediaVerificationAgent.verify_venues_media_coverage()
   - LocationAIEditor.create_descriptions_for_database_results()
   - LocationAIEditor.create_descriptions_for_map_search_results()
   - LocationTelegramFormatter.format_database_results()
   - LocationTelegramFormatter.format_google_maps_results()
3. FIXED: Type hints and error handling
4. FIXED: Proper fallbacks for all methods
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time

from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI

# FIXED: Import LangSmith tracing utilities
from langsmith import traceable
from langchain_core.tracers.langchain import wait_for_all_tracers

# Core location utilities
from location.location_utils import LocationUtils
from location.telegram_location_handler import LocationData

# Database and filtering (keep existing services but wrap with LangChain)
from location.database_search import LocationDatabaseService
from location.filter_evaluator import LocationFilterEvaluator
from location.location_telegram_formatter import LocationTelegramFormatter

# AI Editor and separate agents
from location.location_ai_editor import LocationAIEditor
from location.location_map_search import LocationMapSearchAgent
from location.location_media_verification import LocationMediaVerificationAgent

logger = logging.getLogger(__name__)

class LocationOrchestrator:
    """
    Complete LangChain-powered location search orchestrator with ALL METHOD NAMES FIXED

    Features:
    - Full LangChain implementation using | operator
    - FIXED: Proper @traceable decorators on all pipeline steps
    - FIXED: Correct method names for all agents based on project files
    - FIXED: Non-async wait_for_all_tracers() usage
    - Individual step tracing in LangSmith with correct run_types
    - Proper error handling with LangChain patterns
    - Conditional branching for enhanced verification
    - Complete observability for debugging
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for orchestration
        self.ai = ChatOpenAI(
            model=getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Initialize location-specific services (keep existing but wrap)
        self.database_service = LocationDatabaseService(config)
        self.filter_evaluator = LocationFilterEvaluator(config)
        self.description_editor = LocationAIEditor(config)

        # Enhanced verification agents
        self.map_search_agent = LocationMapSearchAgent(config)
        self.media_verification_agent = LocationMediaVerificationAgent(config)
        self.ai_editor = LocationAIEditor(config)

        self.formatter = LocationTelegramFormatter(config)

        # Pipeline settings
        self.db_search_radius = getattr(config, 'DB_PROXIMITY_RADIUS_KM', 3.0)
        self.min_db_matches = 2
        self.max_venues_to_verify = getattr(config, 'MAX_LOCATION_RESULTS', 8)

        # Build LangChain pipeline
        self._build_langchain_pipeline()

        logger.info("✅ LangChain Location Orchestrator initialized with ALL METHOD NAMES FIXED")

    def _build_langchain_pipeline(self):
        """Build the main LangChain pipeline with conditional branching"""

        # FIXED: Step 1: Geocoding chain with @traceable
        self.geocoding_chain = RunnableLambda(
            self._geocode_step_traced,
            name="geocode_location"
        )

        # FIXED: Step 2: Database search chain with @traceable
        self.database_chain = RunnableLambda(
            self._database_search_step_traced,
            name="database_proximity_search"
        )

        # FIXED: Step 3: Filter evaluation chain with @traceable
        self.filter_chain = RunnableLambda(
            self._filter_evaluation_step_traced,
            name="ai_filter_evaluation"
        )

        # FIXED: Step 4: Description editing chain with @traceable
        self.description_chain = RunnableLambda(
            self._description_editing_step_traced,
            name="ai_description_editing"
        )

        # FIXED: Step 5: Enhanced verification chain with @traceable
        self.enhanced_verification_chain = RunnableLambda(
            self._enhanced_verification_step_traced,
            name="enhanced_media_verification"
        )

        # FIXED: Step 6: Formatting chain with @traceable
        self.formatting_chain = RunnableLambda(
            self._formatting_step_traced,
            name="telegram_formatting"
        )

        # Use | operator for RunnableSequence construction
        basic_sequence = (
            self.geocoding_chain | 
            self.database_chain | 
            self.filter_chain
        )

        # Proper RunnableBranch syntax with condition tuples
        branch = RunnableBranch(
            # If sufficient database results -> direct to description editing
            (
                lambda x: x.get("sufficient_results", False) if isinstance(x, dict) else False and len(x.get("filtered_restaurants", []) if isinstance(x, dict) else []) >= self.min_db_matches,
                self.description_chain | self.formatting_chain
            ),
            # Default case: insufficient results -> enhanced verification flow
            self.enhanced_verification_chain | self.formatting_chain
        )

        # Complete pipeline
        self.location_pipeline = basic_sequence | branch

        logger.info("✅ LangChain pipeline built with FIXED tracing decorators")

    # ============ MAIN PUBLIC METHOD ============

    @traceable(
        run_type="chain",
        name="location_search_orchestrator",
        metadata={"pipeline_type": "langchain", "component": "orchestrator"}
    )
    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query using LangChain pipeline with FIXED LangSmith tracing

        Args:
            query: User's search query
            location_data: Location information
            cancel_check_fn: Function to check if search should be cancelled

        Returns:
            Dict with search results and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"🔍 Starting LangChain location search: '{query}'")

            # Prepare input for pipeline
            pipeline_input = {
                "query": query,
                "raw_query": query,
                "location_data": location_data,
                "cancel_check_fn": cancel_check_fn,
                "start_time": start_time,
                "orchestrator_config": {
                    "db_search_radius": self.db_search_radius,
                    "min_db_matches": self.min_db_matches,
                    "max_venues_to_verify": self.max_venues_to_verify
                }
            }

            # Execute the LangChain pipeline with tracing
            result = await self.location_pipeline.ainvoke(
                pipeline_input,
                config={
                    "run_name": f"location_search_{{query='{query[:30]}...'}}",
                    "metadata": {
                        "user_query": query,
                        "location_type": getattr(location_data, 'location_type', 'unknown'),
                        "has_coordinates": bool(
                            getattr(location_data, 'latitude', None) and 
                            getattr(location_data, 'longitude', None)
                        ),
                        "pipeline_version": "v2.0_all_methods_fixed"
                    },
                    "tags": ["location_search", "langchain_pipeline"]
                }
            )

            # Add timing and success metadata
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            result["pipeline_type"] = "langchain"
            result["tracing_enabled"] = True

            logger.info(f"✅ LangChain location search completed in {processing_time}s")

            # FIXED: wait_for_all_tracers() is NOT async - removed await
            try:
                wait_for_all_tracers()
                logger.debug("🔍 LangSmith traces flushed successfully")
            except Exception as flush_error:
                logger.warning(f"⚠️ Failed to flush traces: {flush_error}")

            return result

        except Exception as e:
            logger.error(f"❌ Error in LangChain location pipeline: {e}")
            # FIXED: wait_for_all_tracers() is NOT async - removed await
            try:
                wait_for_all_tracers()
            except:
                pass
            return self._create_error_response(f"Pipeline error: {str(e)}")

    # ============ FIXED: TRACED LANGCHAIN PIPELINE STEPS ============

    @traceable(
        run_type="tool", 
        name="geocode_location_step",
        metadata={"step": "geocoding", "component": "location_utils"}
    )
    async def _geocode_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: LangChain Step 1 - Geocode location if needed (with tracing)"""
        return await self._geocode_step(pipeline_input)

    @traceable(
        run_type="retriever",
        name="database_proximity_search",
        metadata={"step": "database_search", "component": "database_service"}
    )
    async def _database_search_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: LangChain Step 2 - Database proximity search (with tracing)"""
        return await self._database_search_step(pipeline_input)

    @traceable(
        run_type="llm",
        name="ai_filter_evaluation",
        metadata={"step": "filter_evaluation", "component": "ai_filter"}
    )
    async def _filter_evaluation_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: LangChain Step 3 - AI filter evaluation (with tracing)"""
        return await self._filter_evaluation_step(pipeline_input)

    @traceable(
        run_type="llm",
        name="ai_description_editing",
        metadata={"step": "description_editing", "component": "ai_editor"}
    )
    async def _description_editing_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: LangChain Step 4 - AI description editing (with tracing)"""
        return await self._description_editing_step(pipeline_input)

    @traceable(
        run_type="chain",
        name="enhanced_media_verification",
        metadata={"step": "enhanced_verification", "component": "verification_agents"}
    )
    async def _enhanced_verification_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: LangChain Step 5 - Enhanced verification (with tracing)"""
        return await self._enhanced_verification_step(pipeline_input)

    @traceable(
        run_type="prompt",
        name="telegram_formatting",
        metadata={"step": "formatting", "component": "telegram_formatter"}
    )
    async def _formatting_step_traced(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: LangChain Step 6 - Telegram formatting (with tracing)"""
        return await self._formatting_step(pipeline_input)

    # ============ ORIGINAL PIPELINE STEP IMPLEMENTATIONS ============

    async def _geocode_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 1: Geocode location if needed"""
        try:
            location_data = pipeline_input["location_data"]

            # Check if geocoding is needed
            if (location_data.latitude is None or location_data.longitude is None) and location_data.description:
                logger.info(f"🌍 Geocoding location: {location_data.description}")

                geocoded_coords = LocationUtils.geocode_location(location_data.description)
                if geocoded_coords:
                    location_data.latitude = geocoded_coords[0]
                    location_data.longitude = geocoded_coords[1]
                    logger.info(f"✅ Geocoded to: {geocoded_coords[0]:.4f}, {geocoded_coords[1]:.4f}")
                else:
                    raise ValueError(f"Failed to geocode: {location_data.description}")

            # Extract and validate coordinates
            coordinates = self._extract_coordinates(location_data)
            if not coordinates:
                raise ValueError("Could not extract valid coordinates")

            latitude, longitude = coordinates
            location_desc = getattr(location_data, 'description', f"GPS: {latitude:.4f}, {longitude:.4f}")

            return {
                **pipeline_input,
                "coordinates": coordinates,
                "latitude": latitude,
                "longitude": longitude,
                "location_description": location_desc,
                "geocoding_success": True
            }

        except Exception as e:
            logger.error(f"❌ Geocoding step failed: {e}")
            raise ValueError(f"Geocoding failed: {str(e)}")

    async def _database_search_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 2: Database proximity search"""
        try:
            # Check for cancellation
            if pipeline_input.get("cancel_check_fn") and pipeline_input["cancel_check_fn"]():
                raise ValueError("Search cancelled by user")

            coordinates = pipeline_input["coordinates"]
            config = pipeline_input["orchestrator_config"]

            logger.info(f"🗃️ Database search within {config['db_search_radius']}km")

            # Execute database search
            db_restaurants = self.database_service.search_by_proximity(
                coordinates=coordinates,
                radius_km=config["db_search_radius"],
                extract_descriptions=True
            )

            logger.info(f"🗃️ Found {len(db_restaurants)} restaurants in database")

            return {
                **pipeline_input,
                "db_restaurants": db_restaurants,
                "db_restaurant_count": len(db_restaurants),
                "database_search_success": True
            }

        except Exception as e:
            logger.error(f"❌ Database search step failed: {e}")
            raise ValueError(f"Database search failed: {str(e)}")

    async def _filter_evaluation_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 3: AI filter and evaluate relevance"""
        try:
            # Check for cancellation
            if pipeline_input.get("cancel_check_fn") and pipeline_input["cancel_check_fn"]():
                raise ValueError("Search cancelled by user")

            db_restaurants = pipeline_input.get("db_restaurants", [])
            query = pipeline_input["query"]
            location_desc = pipeline_input["location_description"]

            if not db_restaurants:
                logger.info("🤖 No database restaurants to filter")
                return {
                    **pipeline_input,
                    "filtered_restaurants": [],
                    "filter_result": {"database_sufficient": False},
                    "sufficient_results": False,
                    "filtering_success": True
                }

            logger.info(f"🤖 AI filtering {len(db_restaurants)} database results")

            # Execute AI filtering (use existing filter_evaluator but wrap call)
            filter_result = self.filter_evaluator.filter_and_evaluate(
                restaurants=db_restaurants,
                query=query,
                location_description=location_desc
            )

            filtered_restaurants = filter_result.get("filtered_restaurants", [])
            sufficient_results = filter_result.get("database_sufficient", False)

            logger.info(f"🤖 {len(filtered_restaurants)} restaurants passed AI filtering")

            return {
                **pipeline_input,
                "filtered_restaurants": filtered_restaurants,
                "filter_result": filter_result,
                "sufficient_results": sufficient_results,
                "filtering_success": True
            }

        except Exception as e:
            logger.error(f"❌ Filter evaluation step failed: {e}")
            raise ValueError(f"AI filtering failed: {str(e)}")

    async def _description_editing_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 4: AI edit descriptions for database results - FIXED METHOD NAME"""
        try:
            filtered_restaurants = pipeline_input.get("filtered_restaurants", [])
            query = pipeline_input["query"]

            if not filtered_restaurants:
                logger.info("✏️ No restaurants to edit descriptions for")
                return {
                    **pipeline_input,
                    "final_restaurants": [],
                    "description_editing_success": True
                }

            logger.info(f"✏️ AI editing descriptions for {len(filtered_restaurants)} database restaurants")

            # FIXED: Use the correct method name based on project files
            try:
                edited_restaurants = await self.description_editor.create_descriptions_for_database_results(
                    database_restaurants=filtered_restaurants,
                    user_query=query,
                    cancel_check_fn=pipeline_input.get("cancel_check_fn")
                )

                logger.info(f"✏️ Descriptions edited for {len(edited_restaurants)} database restaurants")

                return {
                    **pipeline_input,
                    "final_restaurants": edited_restaurants,
                    "source": "database_with_editing",
                    "description_editing_success": True
                }

            except Exception as edit_error:
                logger.warning(f"⚠️ Description editing failed: {edit_error}")
                # Don't fail the entire pipeline - use original restaurants
                return {
                    **pipeline_input,
                    "final_restaurants": filtered_restaurants,
                    "source": "database_without_editing",
                    "description_editing_success": False,
                    "description_editing_error": str(edit_error)
                }

        except Exception as e:
            logger.error(f"❌ Description editing step failed: {e}")
            raise ValueError(f"Description editing failed: {str(e)}")

    async def _enhanced_verification_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 5: Enhanced verification using separate agents"""
        try:
            coordinates = pipeline_input["coordinates"]
            query = pipeline_input["query"]
            location_desc = pipeline_input["location_description"]
            cancel_check_fn = pipeline_input.get("cancel_check_fn")

            logger.info("🔍 Starting enhanced verification with separate agents")

            # Execute enhanced verification using separate agents
            enhanced_results = await self._execute_enhanced_verification(
                query=query,
                coordinates=coordinates,
                location_desc=location_desc,
                cancel_check_fn=cancel_check_fn
            )

            return {
                **pipeline_input,
                "final_restaurants": enhanced_results.get("restaurants", []),
                "enhanced_verification_completed": True,
                "verification_metadata": enhanced_results.get("metadata", {}),
                "source": "enhanced_verification"
            }

        except Exception as e:
            logger.error(f"❌ Enhanced verification step failed: {e}")
            raise ValueError(f"Enhanced verification failed: {str(e)}")

    async def _formatting_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 6: Telegram formatting - FIXED METHOD NAMES"""
        try:
            final_restaurants = pipeline_input.get("final_restaurants", [])
            query = pipeline_input["query"]
            coordinates = pipeline_input["coordinates"]
            location_desc = pipeline_input["location_description"]
            source = pipeline_input.get("source", "unknown")

            logger.info(f"📱 Formatting {len(final_restaurants)} restaurants for Telegram")

            if not final_restaurants:
                return {
                    **pipeline_input,
                    "success": False,
                    "location_formatted_results": "😔 No restaurants found matching your criteria.",
                    "restaurant_count": 0,
                    "results": []
                }

            # FIXED: Use correct method names based on project files
            try:
                if source.startswith("database"):
                    # For database results, use format_database_results
                    formatted_results = self.formatter.format_database_results(
                        restaurants=final_restaurants,
                        query=query,
                        location_description=location_desc,
                        offer_more_search=True
                    )
                else:
                    # For map search results, use format_google_maps_results
                    formatted_results = self.formatter.format_google_maps_results(
                        venues=final_restaurants,
                        query=query,
                        location_description=location_desc
                    )

            except Exception as format_error:
                logger.warning(f"⚠️ Formatting method failed: {format_error}")
                # Create a basic format as fallback
                message = f"Found {len(final_restaurants)} restaurants:\n\n"
                for i, restaurant in enumerate(final_restaurants[:5], 1):
                    name = restaurant.get('name', 'Unknown')
                    address = restaurant.get('address', 'No address')
                    rating = restaurant.get('rating', '')
                    rating_text = f" (★{rating})" if rating else ""
                    message += f"{i}. {name}{rating_text}\n   📍 {address}\n\n"
                formatted_results = {"message": message}

            return {
                **pipeline_input,
                "success": True,
                "location_formatted_results": formatted_results.get("message", ""),
                "restaurant_count": len(final_restaurants),
                "results": final_restaurants,
                "source": source,
                "formatting_success": True
            }

        except Exception as e:
            logger.error(f"❌ Formatting step failed: {e}")
            return {
                **pipeline_input,
                "success": False,
                "location_formatted_results": f"😔 Error formatting results: {str(e)}",
                "restaurant_count": 0,
                "results": [],
                "formatting_success": False,
                "formatting_error": str(e)
            }

    # ============ ENHANCED VERIFICATION IMPLEMENTATION ============

    async def _execute_enhanced_verification(
        self,
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Execute enhanced verification using existing agents with FIXED method names"""
        try:
            logger.info("🔍 Enhanced verification: Starting map search")

            # FIXED: Step 1: Map search using correct method name
            map_venues = await self.map_search_agent.search_venues_with_ai_analysis(
                coordinates=coordinates,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            if cancel_check_fn and cancel_check_fn():
                return {"restaurants": [], "metadata": {"cancelled": True}}

            logger.info(f"🗺️ Map search found {len(map_venues)} venues")

            if not map_venues:
                return {
                    "restaurants": [],
                    "metadata": {
                        "enhanced_verification": True,
                        "map_search_venues": 0,
                        "no_venues_found": True
                    }
                }

            # Replace this section in location_orchestrator.py around lines 440-460

            # FIXED: Step 2: Media verification using correct method name
            logger.info("📱 Enhanced verification: Starting media verification")

            try:
                media_verification_results = await self.media_verification_agent.verify_venues_media_coverage(
                    venues=map_venues[:self.max_venues_to_verify],
                    query=query,
                    cancel_check_fn=cancel_check_fn
                )
            except Exception as media_error:
                logger.warning(f"⚠️ Media verification failed: {media_error}")
                # Set empty media results if verification fails
                media_verification_results = []

            if cancel_check_fn and cancel_check_fn():
                return {"restaurants": [], "metadata": {"cancelled": True}}

            logger.info(f"✅ Media verification completed: {len(media_verification_results)} results")

            # FIXED: Step 3: AI description editing using ORIGINAL map venues, not converted ones
            if map_venues:
                logger.info("✏️ Enhanced verification: Creating AI descriptions")

                try:
                    # FIXED: Pass the original map_venues (full VenueSearchResult objects)
                    # and the media_verification_results separately for media data
                    final_descriptions = await self.ai_editor.create_descriptions_for_map_search_results(
                        map_search_results=map_venues[:self.max_venues_to_verify],  # ✅ Use original full venue objects
                        media_verification_results=media_verification_results,       # ✅ Use media results for media data
                        user_query=query,
                        cancel_check_fn=cancel_check_fn
                    )

                    logger.info(f"📝 AI descriptions created for {len(final_descriptions)} venues")

                    return {
                        "restaurants": final_descriptions,
                        "metadata": {
                            "enhanced_verification": True,
                            "map_search_venues": len(map_venues),
                            "media_verification_results": len(media_verification_results),
                            "final_descriptions": len(final_descriptions),
                            "agents_used": ["map_search", "media_verification", "ai_editor"]
                        }
                    }

                except Exception as description_error:
                    logger.warning(f"⚠️ AI description creation failed: {description_error}")
                    # Return original map venues without enhanced descriptions
                    return {
                        "restaurants": map_venues[:self.max_venues_to_verify],
                        "metadata": {
                            "enhanced_verification": True,
                            "map_sea_format_sourcesrch_venues": len(map_venues),
                            "media_verification_results": len(media_verification_results),
                            "description_error": str(description_error),
                            "agents_used": ["map_search", "media_verification"]
                        }
                    }
            else:
                return {
                    "restaurants": [],
                    "metadata": {
                        "enhanced_verification": True,
                        "map_search_venues": len(map_venues),
                        "media_verification_results": len(media_verification_results),
                        "no_venues_found": True
                    }
                }

        except Exception as e:
            logger.error(f"❌ Enhanced verification failed: {e}")
            return {
                "restaurants": [],
                "metadata": {
                    "enhanced_verification": False,
                    "error": str(e)
                }
            }

    # ============ LEGACY COMPATIBILITY METHODS ============

    async def complete_media_verification(
        self,
        venues: List[Any],
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        LEGACY METHOD: Maintained for backward compatibility with existing telegram bot code
        Redirects to enhanced verification flow using separate agents
        """
        logger.info("Legacy complete_media_verification called - redirecting to enhanced flow")

        return await self._execute_enhanced_verification(
            query=query,
            coordinates=coordinates,
            location_desc=location_desc,
            cancel_check_fn=cancel_check_fn
        )

    async def search_and_verify_more_results(
        self,
        query: str,
        coordinates: Tuple[float, float],
        exclude_places: Optional[List[str]] = None,  # FIXED: Added Optional type hint
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        LEGACY METHOD: Search for more results with exclusions - FIXED TYPE HINTS
        """
        try:
            logger.info(f"🔍 Searching for more results: '{query}'")

            # Use enhanced verification but with exclusions
            enhanced_results = await self._execute_enhanced_verification(
                query=query,
                coordinates=coordinates,
                location_desc=f"GPS: {coordinates[0]:.4f}, {coordinates[1]:.4f}",
                cancel_check_fn=cancel_check_fn
            )

            # FIXED: Handle exclude_places properly
            if exclude_places and enhanced_results.get("restaurants"):
                filtered_restaurants = []
                for restaurant in enhanced_results["restaurants"]:
                    restaurant_name = restaurant.get("name", "").lower()
                    if not any(excluded.lower() in restaurant_name for excluded in exclude_places):
                        filtered_restaurants.append(restaurant)

                enhanced_results["restaurants"] = filtered_restaurants
                enhanced_results["excluded_count"] = len(enhanced_results.get("restaurants", [])) - len(filtered_restaurants)

            # Format results
            if enhanced_results.get("restaurants"):
                try:
                    formatted_results = self.formatter.format_google_maps_results(
                        venues=enhanced_results["restaurants"],
                        query=query,
                        location_description=f"GPS: {coordinates[0]:.4f}, {coordinates[1]:.4f}"
                    )
                except Exception as format_error:
                    logger.warning(f"⚠️ Formatting failed: {format_error}")
                    # Basic fallback formatting
                    message = f"Found {len(enhanced_results['restaurants'])} additional restaurants:\n\n"
                    for i, restaurant in enumerate(enhanced_results["restaurants"][:5], 1):
                        name = restaurant.get('name', 'Unknown')
                        address = restaurant.get('address', 'No address')
                        message += f"{i}. {name}\n   📍 {address}\n\n"
                    formatted_results = {"message": message}

                return {
                    "success": True,
                    "location_formatted_results": formatted_results.get("message", ""),
                    "restaurant_count": len(enhanced_results["restaurants"]),
                    "results": enhanced_results["restaurants"],
                    "coordinates": coordinates
                }
            else:
                return {
                    "success": False,
                    "location_formatted_results": "😔 No additional restaurants found.",
                    "restaurant_count": 0,
                    "results": [],
                    "coordinates": coordinates
                }

        except Exception as e:
            logger.error(f"More results query error: {e}")
            return self._create_error_response(f"More results search failed: {str(e)}")

    # ============ UTILITY METHODS ============

    def _extract_coordinates(self, location_data: LocationData) -> Optional[Tuple[float, float]]:
        """Extract coordinates from location data"""
        if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
            if location_data.latitude is not None and location_data.longitude is not None:
                return (float(location_data.latitude), float(location_data.longitude))
        return None

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "restaurants": [],
            "restaurant_count": 0,
            "message": f"Location search failed: {error_message}",
            "pipeline_type": "langchain"
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the LangChain pipeline"""
        return {
            'pipeline_type': 'langchain_all_methods_fixed',
            'database_service': True,
            'enhanced_verifier': self.media_verification_agent is not None,
            'description_editor': self.description_editor is not None,
            'ai_editor': self.ai_editor is not None,
            'min_db_matches_trigger': self.min_db_matches,
            'langsmith_tracing_enabled': True,
            'tracing_version': 'v2_all_methods_fixed',
            'chain_components': [
                'geocoding_chain_traced',
                'database_chain_traced', 
                'filter_chain_traced',
                'description_chain_traced',
                'enhanced_verification_chain_traced',
                'formatting_chain_traced'
            ]
        }