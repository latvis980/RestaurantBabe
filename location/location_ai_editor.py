# location/location_ai_editor.py
"""
Location AI Editor - Unified facade for description generation

This facade handles the decision logic for which editor to use based on
the data source (database vs maps). The orchestrator delegates to this
class instead of containing the business logic itself.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Mapping
from dataclasses import dataclass

from location.location_database_ai_editor import LocationDatabaseAIEditor, DatabaseRestaurantDescription
from location.location_map_search_ai_editor import LocationMapSearchAIEditor, MapSearchRestaurantDescription

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRestaurantDescription:
    """Unified description format returned by the facade"""
    name: str
    address: str
    maps_link: str
    place_id: str
    distance_km: float
    description: str
    sources: List[str]
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    selection_score: Optional[float] = None
    selection_reason: Optional[str] = None
    source_type: str = "unknown"  # "database" or "maps"


class LocationAIEditor:
    """
    Unified facade for location-based restaurant description generation.

    Handles the decision logic for which specialized editor to use:
    - LocationDatabaseAIEditor for database results
    - LocationMapSearchAIEditor for Google Maps results

    The orchestrator should use this facade instead of containing
    the business logic for choosing between editors.
    """

    def __init__(self, config):
        self.config = config
        self.database_editor = LocationDatabaseAIEditor(config)
        self.map_search_editor = LocationMapSearchAIEditor(config)
        logger.info("LocationAIEditor facade initialized")

    async def generate_descriptions_from_state(
        self,
        state: Mapping[str, Any],
        cancel_check_fn=None
    ) -> List[UnifiedRestaurantDescription]:
        """
        Generate descriptions based on state - automatically chooses the right editor.

        Decision logic:
        1. If maps_results + media_verification_results exist → use map search editor
        2. If filtered_results (database) exist → use database editor
        3. Return empty list if no data

        Args:
            state: The unified search state dictionary
            cancel_check_fn: Optional cancellation check function

        Returns:
            List of UnifiedRestaurantDescription objects
        """
        query = state.get("query", "")

        # Check for maps results (these come after media verification)
        maps_results = state.get("maps_results", [])
        media_verification_results = state.get("media_verification_results", [])

        # Check for database results
        filtered_results = state.get("filtered_results", {})
        database_restaurants = filtered_results.get("filtered_restaurants", []) if filtered_results else []

        # Decision: Maps results take priority (they're more complete after verification)
        if maps_results:
            logger.info(f"🗺️ Using MAP SEARCH editor for {len(maps_results)} venues")
            return await self._generate_from_maps(
                maps_results=maps_results,
                media_verification_results=media_verification_results,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

        elif database_restaurants:
            logger.info(f"🗃️ Using DATABASE editor for {len(database_restaurants)} restaurants")
            return await self._generate_from_database(
                database_restaurants=database_restaurants,
                query=query,
                cancel_check_fn=cancel_check_fn
            )

        else:
            logger.warning("⚠️ No restaurants available for description generation")
            return []

    async def _generate_from_maps(
        self,
        maps_results: List[Any],
        media_verification_results: List[Any],
        query: str,
        cancel_check_fn=None
    ) -> List[UnifiedRestaurantDescription]:
        """Generate descriptions using the map search editor"""
        try:
            descriptions = await self.map_search_editor.create_descriptions_for_map_search_results(
                map_search_results=maps_results,
                media_verification_results=media_verification_results,
                user_query=query,
                cancel_check_fn=cancel_check_fn
            )

            # Convert to unified format
            return [self._convert_map_description(desc) for desc in descriptions]

        except Exception as e:
            logger.error(f"Error generating map search descriptions: {e}")
            return []

    async def _generate_from_database(
        self,
        database_restaurants: List[Dict[str, Any]],
        query: str,
        cancel_check_fn=None
    ) -> List[UnifiedRestaurantDescription]:
        """Generate descriptions using the database editor"""
        try:
            descriptions = await self.database_editor.create_descriptions_for_database_results(
                database_restaurants=database_restaurants,
                user_query=query,
                cancel_check_fn=cancel_check_fn
            )

            # Convert to unified format
            return [self._convert_database_description(desc) for desc in descriptions]

        except Exception as e:
            logger.error(f"Error generating database descriptions: {e}")
            return []

    def _convert_map_description(self, desc: MapSearchRestaurantDescription) -> UnifiedRestaurantDescription:
        """Convert MapSearchRestaurantDescription to unified format"""
        return UnifiedRestaurantDescription(
            name=desc.name,
            address=desc.address,
            maps_link=desc.google_maps_url,
            place_id=desc.place_id,
            distance_km=desc.distance_km,
            description=desc.description,
            sources=desc.media_sources or [],
            rating=desc.rating,
            user_ratings_total=desc.user_ratings_total,
            selection_score=desc.selection_score,
            selection_reason=desc.selection_reason,
            source_type="maps"
        )

    def _convert_database_description(self, desc: DatabaseRestaurantDescription) -> UnifiedRestaurantDescription:
        """Convert DatabaseRestaurantDescription to unified format"""
        return UnifiedRestaurantDescription(
            name=desc.name,
            address=desc.address,
            maps_link=desc.maps_link,
            place_id="",  # Database results may not have place_id
            distance_km=desc.distance_km,
            description=desc.description,
            sources=desc.sources or [],
            rating=desc.rating,
            user_ratings_total=desc.user_ratings_total,
            selection_score=desc.selection_score,
            selection_reason=None,
            source_type="database"
        )