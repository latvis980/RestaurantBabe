# location/location_data_logger.py
"""
Data Logger for Location Editor Debugging

Creates JSON files with combined restaurant data for inspection.
Saves files with timestamps in /tmp/ directory for Railway SSH access.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

class LocationDataLogger:
    """Logger for debugging restaurant data passed to location editor"""

    def __init__(self, config=None, enabled: bool = True):
        # Use config settings if provided
        if config:
            self.enabled = getattr(config, 'ENABLE_LOCATION_DEBUG_LOGGING', enabled)
            self.log_directory = getattr(config, 'LOCATION_DEBUG_LOG_DIRECTORY', "/tmp/location_debug")
            self.cleanup_days = getattr(config, 'LOCATION_DEBUG_CLEANUP_DAYS', 3)
        else:
            self.enabled = enabled
            self.log_directory = "/tmp/location_debug"
            self.cleanup_days = 3

        # Create log directory if it doesn't exist
        if self.enabled:
            try:
                os.makedirs(self.log_directory, exist_ok=True)
                logger.info(f"Location data logger initialized. Files will be saved to: {self.log_directory}")
            except Exception as e:
                logger.error(f"Failed to create log directory {self.log_directory}: {e}")
                self.enabled = False

    def log_combined_data(
        self, 
        map_search_results: List[Any],
        media_verification_results: List[Any],
        combined_venues: List[Any],
        user_query: str = "",
        location_desc: str = ""
    ) -> Optional[str]:
        """
        Log all the data that goes into location editor
        Returns the file path where data was saved
        """
        if not self.enabled:
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"location_editor_data_{timestamp}.json"
            filepath = os.path.join(self.log_directory, filename)

            # Convert data to serializable format
            debug_data = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "location_description": location_desc,
                "summary": {
                    "map_search_results_count": len(map_search_results) if map_search_results else 0,
                    "media_verification_results_count": len(media_verification_results) if media_verification_results else 0,
                    "combined_venues_count": len(combined_venues) if combined_venues else 0
                },
                "map_search_results": self._serialize_map_results(map_search_results),
                "media_verification_results": self._serialize_media_results(media_verification_results),
                "combined_venues": self._serialize_combined_venues(combined_venues)
            }

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"📁 Location editor data saved to: {filepath}")
            logger.info(f"📊 Data summary: {len(map_search_results or [])} map results, "
                       f"{len(media_verification_results or [])} media results, "
                       f"{len(combined_venues or [])} combined venues")

            return filepath

        except Exception as e:
            logger.error(f"Failed to log location editor data: {e}")
            return None

    def log_ai_selection_data(
        self,
        venues_before_selection: List[Any],
        venues_after_selection: List[Any], 
        user_query: str = ""
    ) -> Optional[str]:
        """Log AI restaurant selection data"""
        if not self.enabled:
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_selection_data_{timestamp}.json"
            filepath = os.path.join(self.log_directory, filename)

            selection_data = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "selection_summary": {
                    "venues_before_count": len(venues_before_selection) if venues_before_selection else 0,
                    "venues_after_count": len(venues_after_selection) if venues_after_selection else 0,
                    "selection_ratio": len(venues_after_selection) / len(venues_before_selection) if venues_before_selection else 0
                },
                "venues_before_selection": self._serialize_combined_venues(venues_before_selection),
                "venues_after_selection": self._serialize_combined_venues(venues_after_selection)
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(selection_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"🎯 AI selection data saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to log AI selection data: {e}")
            return None

    def log_description_generation_data(
        self,
        selected_venues: List[Any],
        generated_descriptions: List[Any],
        user_query: str = ""
    ) -> Optional[str]:
        """Log description generation input and output"""
        if not self.enabled:
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"description_generation_{timestamp}.json"
            filepath = os.path.join(self.log_directory, filename)

            description_data = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "generation_summary": {
                    "input_venues_count": len(selected_venues) if selected_venues else 0,
                    "generated_descriptions_count": len(generated_descriptions) if generated_descriptions else 0
                },
                "input_venues": self._serialize_combined_venues(selected_venues),
                "generated_descriptions": self._serialize_descriptions(generated_descriptions)
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(description_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"📝 Description generation data saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to log description generation data: {e}")
            return None

    def _serialize_map_results(self, map_results: List[Any]) -> List[Dict]:
        """Convert map search results to serializable format"""
        if not map_results:
            return []

        serialized = []
        for result in map_results:
            try:
                if hasattr(result, '__dict__'):
                    # Convert object to dict
                    result_dict = {}
                    for key, value in result.__dict__.items():
                        if not key.startswith('_'):  # Skip private attributes
                            result_dict[key] = self._safe_serialize(value)
                    serialized.append(result_dict)
                else:
                    # Already a dict or other serializable type
                    serialized.append(self._safe_serialize(result))
            except Exception as e:
                logger.warning(f"Failed to serialize map result: {e}")
                serialized.append({"error": f"Serialization failed: {str(e)}"})

        return serialized

    def _serialize_media_results(self, media_results: List[Any]) -> List[Dict]:
        """Convert media verification results to serializable format"""
        if not media_results:
            return []

        serialized = []
        for result in media_results:
            try:
                if hasattr(result, '__dict__'):
                    # Convert MediaVerificationResult to dict
                    result_dict = {}
                    for key, value in result.__dict__.items():
                        if not key.startswith('_'):
                            result_dict[key] = self._safe_serialize(value)
                    serialized.append(result_dict)
                else:
                    serialized.append(self._safe_serialize(result))
            except Exception as e:
                logger.warning(f"Failed to serialize media result: {e}")
                serialized.append({"error": f"Serialization failed: {str(e)}"})

        return serialized

    def _serialize_combined_venues(self, combined_venues: List[Any]) -> List[Dict]:
        """Convert CombinedVenueData objects to serializable format"""
        if not combined_venues:
            return []

        serialized = []
        for venue in combined_venues:
            try:
                if hasattr(venue, '__dict__'):
                    # Convert CombinedVenueData to dict
                    venue_dict = {}
                    for key, value in venue.__dict__.items():
                        if not key.startswith('_'):
                            venue_dict[key] = self._safe_serialize(value)
                    serialized.append(venue_dict)
                else:
                    serialized.append(self._safe_serialize(venue))
            except Exception as e:
                logger.warning(f"Failed to serialize combined venue: {e}")
                serialized.append({"error": f"Serialization failed: {str(e)}"})

        return serialized

    def _serialize_descriptions(self, descriptions: List[Any]) -> List[Dict]:
        """Convert RestaurantDescription objects to serializable format"""
        if not descriptions:
            return []

        serialized = []
        for desc in descriptions:
            try:
                if hasattr(desc, '__dict__'):
                    # Convert RestaurantDescription to dict
                    desc_dict = {}
                    for key, value in desc.__dict__.items():
                        if not key.startswith('_'):
                            desc_dict[key] = self._safe_serialize(value)
                    serialized.append(desc_dict)
                else:
                    serialized.append(self._safe_serialize(desc))
            except Exception as e:
                logger.warning(f"Failed to serialize description: {e}")
                serialized.append({"error": f"Serialization failed: {str(e)}"})

        return serialized

    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize any object to JSON-compatible format"""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [self._safe_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): self._safe_serialize(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                # Convert object to dict
                return {k: self._safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            else:
                # Fallback to string representation
                return str(obj)
        except Exception:
            return f"<Serialization Error: {type(obj).__name__}>"

    def get_latest_files(self, limit: int = 5) -> List[str]:
        """Get list of latest debug files"""
        if not self.enabled or not os.path.exists(self.log_directory):
            return []

        try:
            files = []
            for filename in os.listdir(self.log_directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.log_directory, filename)
                    files.append((filepath, os.path.getmtime(filepath)))

            # Sort by modification time (newest first)
            files.sort(key=lambda x: x[1], reverse=True)

            return [filepath for filepath, _ in files[:limit]]

        except Exception as e:
            logger.error(f"Failed to get latest files: {e}")
            return []

    def cleanup_old_files(self, keep_days: int = 3):
        """Clean up debug files older than specified days"""
        if not self.enabled:
            return

        try:
            import time
            cutoff_time = time.time() - (keep_days * 24 * 60 * 60)

            for filename in os.listdir(self.log_directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.log_directory, filename)
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        logger.debug(f"Cleaned up old debug file: {filename}")

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")