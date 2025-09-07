# location/location_ai_editor.py
"""
DEPRECATED: AI Editor - Split into separate implementations

This file now redirects to the new specialized implementations:
- location_database_ai_editor.py: For database results
- location_map_search_ai_editor.py: For map search results

This separation provides cleaner code organization and distinct processing flows.
"""

import logging
from typing import List, Dict, Any, Optional

# Import the new specialized implementations
from location.location_database_ai_editor import LocationDatabaseAIEditor, DatabaseRestaurantDescription
from location.location_map_search_ai_editor import LocationMapSearchAIEditor, MapSearchRestaurantDescription

logger = logging.getLogger(__name__)

class LocationAIEditor:
    """
    DEPRECATED: Legacy wrapper that delegates to specialized implementations

    Use the specialized classes directly:
    - LocationDatabaseAIEditor for database results
    - LocationMapSearchAIEditor for map search results
    """

    def __init__(self, config):
        self.config = config

        # Initialize the new specialized editors
        self.database_editor = LocationDatabaseAIEditor(config)
        self.map_search_editor = LocationMapSearchAIEditor(config)

        logger.warning("LocationAIEditor is deprecated. Use LocationDatabaseAIEditor or LocationMapSearchAIEditor directly.")

    async def create_descriptions_for_database_results(
        self,
        database_restaurants: List[Dict[str, Any]],
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[DatabaseRestaurantDescription]:
        """
        DEPRECATED: Delegate to LocationDatabaseAIEditor
        """
        logger.info("Delegating to LocationDatabaseAIEditor.create_descriptions_for_database_results()")
        return await self.database_editor.create_descriptions_for_database_results(
            database_restaurants=database_restaurants,
            user_query=user_query,
            cancel_check_fn=cancel_check_fn
        )

    async def create_descriptions_for_map_search_results(
        self,
        map_search_results: List[Any],
        media_verification_results: Optional[List[Any]] = None,  
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[MapSearchRestaurantDescription]:
        """
        DEPRECATED: Delegate to LocationMapSearchAIEditor
        """
        logger.info("Delegating to LocationMapSearchAIEditor.create_descriptions_for_map_search_results()")
        return await self.map_search_editor.create_descriptions_for_map_search_results(
            map_search_results=map_search_results,
            media_verification_results=media_verification_results,
            user_query=user_query,
            cancel_check_fn=cancel_check_fn
        )

    # Legacy method redirects with deprecation warnings
    async def create_descriptions(self, *args, **kwargs):
        """
        DEPRECATED: Use create_descriptions_for_map_search_results instead.
        """
        logger.warning("create_descriptions() is deprecated. Use create_descriptions_for_map_search_results() instead.")
        return await self.create_descriptions_for_map_search_results(*args, **kwargs)


# Backwards compatibility exports
RestaurantDescription = MapSearchRestaurantDescription  # For backwards compatibility

# Re-export the new classes for direct import
__all__ = [
    'LocationAIEditor',  # Deprecated wrapper
    'LocationDatabaseAIEditor', 
    'LocationMapSearchAIEditor',
    'DatabaseRestaurantDescription',
    'MapSearchRestaurantDescription',
    'RestaurantDescription'  # Legacy alias
]