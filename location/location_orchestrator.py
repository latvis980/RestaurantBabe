# location/location_orchestrator.py
"""
LangChain-Powered Location Search Orchestrator - TYPE ERRORS FIXED

Fixed all type errors based on latest LangChain documentation:
- Correct RunnableSequence syntax using | operator 
- Proper RunnableBranch constructor with tuples
- Removed unused imports
- Fixed typing issues
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time

from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI

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
    LangChain-powered location search orchestrator with complete tracing

    Features:
    - Full LangChain implementation using | operator
    - Individual step tracing in LangSmith
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

        logger.info("✅ LangChain Location Orchestrator initialized with full tracing")

    def _build_langchain_pipeline(self):
        """Build the main LangChain pipeline with conditional branching"""

        # Step 1: Geocoding chain
        self.geocoding_chain = RunnableLambda(
            self._geocode_step,
            name="geocode_location"
        )

        # Step 2: Database search chain
        self.database_chain = RunnableLambda(
            self._database_search_step,
            name="database_proximity_search"
        )

        # Step 3: Filter evaluation chain
        self.filter_chain = RunnableLambda(
            self._filter_evaluation_step,
            name="ai_filter_evaluation"
        )

        # Step 4: Description editing chain  
        self.description_chain = RunnableLambda(
            self._description_editing_step,
            name="ai_description_editing"
        )

        # Step 5: Enhanced verification chain (conditional)
        self.enhanced_verification_chain = RunnableLambda(
            self._enhanced_verification_step,
            name="enhanced_media_verification"
        )

        # Step 6: Formatting chain
        self.formatting_chain = RunnableLambda(
            self._formatting_step,
            name="telegram_formatting"
        )

        # FIXED: Use | operator for RunnableSequence construction
        # First create the basic sequence
        basic_sequence = (
            self.geocoding_chain | 
            self.database_chain | 
            self.filter_chain
        )

        # FIXED: Proper RunnableBranch syntax with condition tuples
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

        logger.info("✅ LangChain pipeline built with conditional branching")

    # ============ MAIN PUBLIC METHOD ============

    async def process_location_query(
        self, 
        query: str, 
        location_data: LocationData,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Process location query using LangChain pipeline with full tracing

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
                        )
                    }
                }
            )

            # Add timing and success metadata
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            result["pipeline_type"] = "langchain"

            logger.info(f"✅ LangChain location search completed in {processing_time}s")
            return result

        except Exception as e:
            logger.error(f"❌ Error in LangChain location pipeline: {e}")
            return self._create_error_response(f"Pipeline error: {str(e)}")

    # ============ LANGCHAIN PIPELINE STEPS ============

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
        """LangChain Step 4: AI edit descriptions for database results - CORRECTED"""
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

            # FIXED: Use the new method specifically for database results
            # This skips the atmospheric filtering that's meant for map search results
            edited_restaurants = await self.description_editor.create_descriptions_for_database_results(
                database_restaurants=filtered_restaurants,  # These are already filtered by filter_evaluator
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

        except Exception as e:
            logger.error(f"❌ Description editing step failed: {e}")
            # Don't fail the entire pipeline - use original restaurants
            return {
                **pipeline_input,
                "final_restaurants": pipeline_input.get("filtered_restaurants", []),
                "source": "database_without_editing",
                "description_editing_success": False,
                "description_editing_error": str(e)
            }

    async def _enhanced_verification_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 5: Enhanced verification flow (when insufficient DB results)"""
        try:
            # Check for cancellation
            if pipeline_input.get("cancel_check_fn") and pipeline_input["cancel_check_fn"]():
                raise ValueError("Search cancelled by user")

            query = pipeline_input["query"]
            coordinates = pipeline_input["coordinates"]
            location_desc = pipeline_input["location_description"]

            logger.info("🔍 Starting enhanced verification flow")

            # Execute enhanced verification using existing agents
            enhanced_result = await self._execute_enhanced_verification(
                query=query,
                coordinates=coordinates,
                location_desc=location_desc,
                cancel_check_fn=pipeline_input.get("cancel_check_fn")
            )

            # Extract results from enhanced verification
            final_restaurants = enhanced_result.get("restaurants", [])

            return {
                **pipeline_input,
                "final_restaurants": final_restaurants,
                "source": "enhanced_verification",
                "enhanced_verification_success": True,
                "enhanced_verification_details": enhanced_result
            }

        except Exception as e:
            logger.error(f"❌ Enhanced verification step failed: {e}")
            raise ValueError(f"Enhanced verification failed: {str(e)}")

    async def _formatting_step(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Step 6: Format results for Telegram"""
        try:
            final_restaurants = pipeline_input.get("final_restaurants", [])
            query = pipeline_input["query"]
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

            # Use existing formatter (correct method name)
            formatted_results = self.formatter.format_database_results(
                restaurants=final_restaurants,
                query=query,
                location_description=location_desc,
                offer_more_search=True
            )

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

    # ============ HELPER METHODS ============

    async def _execute_enhanced_verification(
        self,
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """Execute enhanced verification using existing agents"""
        try:
            # Step 1: Map search (correct method name)
            map_result = await self.map_search_agent.search_venues_with_ai_analysis(
                coordinates=coordinates,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            venues = map_result  # The method returns the venues list directly
            if not venues:
                return {"restaurants": [], "error": "No venues found in map search"}

            # Step 2: Media verification (correct method name)
            media_result = await self.media_verification_agent.verify_venues_media_coverage(
                venues=venues,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

            verified_venues = media_result  # The method returns the results list directly
            if not verified_venues:
                return {"restaurants": [], "error": "No venues passed media verification"}

            # Step 3: AI editing - FIXED: Use the correct method name
            final_restaurants = await self.ai_editor.create_descriptions_for_map_search_results(
                map_search_results=venues,
                media_verification_results=verified_venues,
                user_query=query,
                cancel_check_fn=cancel_check_fn
            )

            return {
                "restaurants": final_restaurants,
                "map_search_count": len(venues),
                "verified_count": len(verified_venues), 
                "final_count": len(final_restaurants)
            }

        except Exception as e:
            logger.error(f"Enhanced verification error: {e}")
            return {"restaurants": [], "error": str(e)}

    def _extract_coordinates(self, location_data: LocationData) -> Optional[Tuple[float, float]]:
        """Extract coordinates from location data"""
        try:
            if hasattr(location_data, 'latitude') and hasattr(location_data, 'longitude'):
                if location_data.latitude is not None and location_data.longitude is not None:
                    lat, lng = location_data.latitude, location_data.longitude
                    if LocationUtils.validate_coordinates(lat, lng):
                        return (lat, lng)

            logger.warning(f"Invalid coordinates: lat={getattr(location_data, 'latitude', None)}, lng={getattr(location_data, 'longitude', None)}")
            return None

        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            return None

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "source": "error",
            "location_formatted_results": f"😔 {error_message}",
            "restaurant_count": 0,
            "pipeline_type": "langchain"
        }

    def _create_cancelled_response(self) -> Dict[str, Any]:
        """Create standardized cancellation response"""
        return {
            "success": False,
            "cancelled": True,
            "results": [],
            "source": "cancelled", 
            "location_formatted_results": "🔄 Search was cancelled.",
            "restaurant_count": 0,
            "pipeline_type": "langchain"
        }

    # ============ LEGACY COMPATIBILITY METHODS ============

    async def process_more_results_query(
        self,
        query: str,
        coordinates: Tuple[float, float],
        location_desc: str,
        cancel_check_fn=None
    ) -> Dict[str, Any]:
        """
        Legacy compatibility method for "more results" searches
        Routes to enhanced verification flow
        """
        logger.info("More results query - routing to enhanced verification")

        # Use enhanced verification directly
        try:
            enhanced_result = await self._execute_enhanced_verification(
                query=query,
                coordinates=coordinates,
                location_desc=location_desc,
                cancel_check_fn=cancel_check_fn
            )

            # Extract results from enhanced verification
            restaurants = enhanced_result.get("restaurants", [])
            if restaurants:
                # Format using the correct method name
                formatted_message = self.formatter.format_database_results(
                    restaurants=restaurants,
                    query=query,
                    location_description=location_desc,
                    offer_more_search=False  # Don't offer more for "more results"
                )

                return {
                    "success": True,
                    "location_formatted_results": formatted_message.get("message", ""),
                    "restaurant_count": len(restaurants),
                    "results": restaurants,
                    "source": "enhanced_verification",
                    "coordinates": coordinates
                }
            else:
                return {
                    "success": False,
                    "location_formatted_results": "😔 No additional restaurants found in this area.",
                    "restaurant_count": 0,
                    "results": [],
                    "coordinates": coordinates
                }

        except Exception as e:
            logger.error(f"More results query error: {e}")
            return self._create_error_response(f"More results search failed: {str(e)}")

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

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the LangChain pipeline"""
        return {
            'pipeline_type': 'langchain',
            'database_service': True,
            'enhanced_verifier': self.media_verification_agent is not None,
            'description_editor': self.description_editor is not None,
            'ai_editor': self.ai_editor is not None,
            'min_db_matches_trigger': self.min_db_matches,
            'langchain_tracing_enabled': True,
            'chain_components': [
                'geocoding_chain',
                'database_chain', 
                'filter_chain',
                'description_chain',
                'enhanced_verification_chain',
                'formatting_chain'
            ]
        }