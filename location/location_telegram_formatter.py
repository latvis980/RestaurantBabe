# location/location_telegram_formatter.py
"""
Unified Location Results Formatter

This creates a consistent format for ALL location-based results:
- Database results (Step 2a)
- Google Maps results (Steps 3-5)
- Consistent appearance regardless of source

Standard format for all location results:
Name
Address (formatted using canonical Place ID method)  
Distance from user
Description
Recommended by [sources]
"""

import logging
import re
import requests
from typing import Dict, List, Any, Optional
from html import escape

logger = logging.getLogger(__name__)

class LocationTelegramFormatter:
    """
    Unified formatter for all location-based search results
    Ensures consistent appearance regardless of source (database vs Google Maps)
    """

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config=None):
        """Initialize formatter with optional config"""
        self.config = config

    def format_database_results(
        self,
        restaurants: List[Dict[str, Any]],
        query: str,
        location_description: str,
        offer_more_search: bool = True
    ) -> Dict[str, Any]:
        """
        Format database results (Step 2a) with user choice option

        Args:
            restaurants: List of restaurant dictionaries from database
            query: Original user query
            location_description: Description of the searched location
            offer_more_search: Whether to offer additional Google Maps search

        Returns:
            Formatted results dictionary
        """
        try:
            logger.info(f"📋 Formatting {len(restaurants)} database results")

            if not restaurants:
                return self._create_empty_response(query, "database")

            # Build header message
            header = f"<b>🗂️ Here are some restaurants from my notes:</b>\n\n"

            # Format each restaurant using unified format
            restaurant_entries = []
            for i, restaurant in enumerate(restaurants, 1):
                entry = self._format_single_restaurant(restaurant, i, source_type="database")
                if entry:
                    restaurant_entries.append(entry)

            # Add user choice prompt if requested
            footer = ""
            if offer_more_search:
                footer = (
                    f"\n<b>💭 Do you want to try these restaurants first, or would you like me to search for more addresses?</b>\n\n"
                    "<i>Reply with 'search more' if you'd like additional options from Google Maps</i>"
                )

            # Combine all parts
            message_parts = [header] + restaurant_entries + [footer]
            message = ''.join(message_parts)

            # Apply length limit
            if len(message) > self.MAX_MESSAGE_LENGTH:
                message = message[:self.MAX_MESSAGE_LENGTH-3] + "…"

            return {
                "main_list": restaurants,
                "formatted_message": message,
                "source_type": "database",
                "search_info": {
                    "query": query,
                    "location": location_description,
                    "count": len(restaurants),
                    "needs_user_choice": offer_more_search
                }
            }

        except Exception as e:
            logger.error(f"❌ Error formatting database results: {e}")
            return self._create_empty_response(query, "database")

    def format_google_maps_results(
        self,
        restaurants: List[Dict[str, Any]],
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """
        Format Google Maps results (Steps 3-5) 

        Args:
            restaurants: List of verified venue dictionaries
            query: Original user query
            location_description: Description of the searched location

        Returns:
            Formatted results dictionary
        """
        try:
            logger.info(f"📋 Formatting {len(restaurants)} Google Maps results")

            if not restaurants:
                return self._create_empty_response(query, "google_maps")

            # Build header message
            header = f"<b>🗺️ Found {len(restaurants)} restaurants from my search:</b>\n\n"

            # Format each restaurant using unified format
            restaurant_entries = []
            for i, restaurant in enumerate(restaurants, 1):
                entry = self._format_single_restaurant(restaurant, i, source_type="google_maps")
                if entry:
                    restaurant_entries.append(entry)

            # Add footer note
            footer = "\n<i>Click the address to see venue photos and menu on Google Maps</i>"

            # Combine all parts
            message_parts = [header] + restaurant_entries + [footer]
            message = ''.join(message_parts)

            # Apply length limit
            if len(message) > self.MAX_MESSAGE_LENGTH:
                message = message[:self.MAX_MESSAGE_LENGTH-3] + "…"

            return {
                "main_list": restaurants,
                "formatted_message": message,
                "source_type": "google_maps",
                "search_info": {
                    "query": query,
                    "location": location_description,
                    "count": len(restaurants),
                    "needs_user_choice": False
                }
            }

        except Exception as e:
            logger.error(f"❌ Error formatting Google Maps results: {e}")
            return self._create_empty_response(query, "google_maps")

    def _format_single_restaurant(
        self, 
        restaurant: Dict[str, Any], 
        index: int,
        source_type: str = "database"
    ) -> str:
        """
        Format a single restaurant using the UNIFIED STANDARD FORMAT:

        Name
        Address (canonical Place ID method)
        Distance from user  
        Description
        Recommended by [sources]
        """
        try:
            name = restaurant.get('name', '').strip()
            if not name:
                return ""

            # STANDARD FORMAT COMPONENTS

            # 1. NAME (clean and bold)
            name_clean = self._clean_html(name)
            name_line = f"<b>{index}. {name_clean}</b>\n"

            # 2. ADDRESS (canonical Place ID method)
            address_line = self._format_address_link(restaurant)

            # 3. DISTANCE FROM USER
            distance_line = self._format_distance(restaurant)

            # 4. DESCRIPTION 
            description_line = self._format_description(restaurant)

            # 5. RECOMMENDED BY [SOURCES]
            sources_line = self._format_sources(restaurant, source_type)

            # Combine all components
            entry_parts = [
                name_line,
                address_line,
                distance_line,
                description_line,
                sources_line,
                "\n"  # Spacing between restaurants
            ]

            return ''.join(part for part in entry_parts if part)

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {restaurant.get('name', 'Unknown')}: {e}")
            return ""

    def _format_address_link(self, restaurant: Dict[str, Any]) -> str:
        """
        Format address with Google Maps link using canonical Place ID method
        """
        try:
            address = restaurant.get('address', '').strip()
            place_id = restaurant.get('place_id') or restaurant.get('google_place_id')

            if not address or address.lower() in ['unknown', 'not available', '']:
                address = "Address available on Google Maps"

            # Extract street address (remove postal codes and country)
            street_address = self._extract_street_address(address)
            clean_street = self._clean_html(street_address)

            # Get canonical Google Maps URL
            google_url = self._get_canonical_google_maps_url(restaurant, place_id)

            return f'📍 <a href="{escape(google_url, quote=True)}">{clean_street}</a>\n'

        except Exception as e:
            logger.error(f"❌ Error formatting address: {e}")
            return "📍 Address available\n"

    def _extract_street_address(self, full_address: str) -> str:
        """
        Extract street address part (remove postal codes and country)
        """
        try:
            if not full_address:
                return "Address available"

            # Split by commas and analyze parts
            parts = [part.strip() for part in full_address.split(',')]

            if len(parts) <= 2:
                return full_address  # Keep as-is if short

            # Remove the last part if it looks like a country
            if len(parts) >= 2:
                last_part = parts[-1].strip()
                # Common country patterns
                if (len(last_part) <= 4 or  # Short country codes (USA, UK, etc.)
                    last_part in ['United States', 'United Kingdom', 'Portugal', 'France', 'Germany', 'Spain', 'Italy']):
                    parts = parts[:-1]

            # Remove postal codes (last part if it contains numbers)
            if len(parts) >= 2:
                last_part = parts[-1].strip()
                if re.search(r'\d', last_part) and len(last_part) <= 10:
                    parts = parts[:-1]

            # Take first 2-3 parts for street address
            street_parts = parts[:3] if len(parts) > 3 else parts
            return ', '.join(street_parts)

        except Exception:
            return full_address

    def _get_canonical_google_maps_url(self, restaurant: Dict[str, Any], place_id: str) -> str:
        """
        Get canonical Google Maps URL using place ID or CID
        """
        try:
            # Check for existing canonical URL
            existing_url = (restaurant.get("google_maps_url") or 
                          restaurant.get("google_url") or 
                          restaurant.get("url") or "")

            if existing_url:
                # Check if already canonical
                if "cid=" in existing_url:
                    return existing_url

                # Try to extract CID from existing URL
                cid_match = re.search(r"[?&]cid=(\d+)", existing_url)
                if cid_match:
                    return f"https://maps.google.com/?cid={cid_match.group(1)}"

            # Use place_id if available
            if place_id:
                return f"https://www.google.com/maps/place/?q=place_id:{place_id}"

            # Fallback to existing URL or generic
            return existing_url or "#"

        except Exception:
            return "#"

    def _format_distance(self, restaurant: Dict[str, Any]) -> str:
        """
        Format distance from user
        """
        try:
            distance_km = restaurant.get('distance_km')
            distance_text = restaurant.get('distance_text')

            if distance_text:
                return f"📏 {distance_text} away\n"
            elif distance_km is not None:
                from location.location_utils import LocationUtils
                distance_text = LocationUtils.format_distance(float(distance_km))
                return f"📏 {distance_text} away\n"
            else:
                return ""

        except Exception:
            return ""

    def _format_description(self, restaurant: Dict[str, Any]) -> str:
        """
        Format restaurant description
        """
        try:
            description = (restaurant.get('description', '') or 
                         restaurant.get('raw_description', '')).strip()

            if not description or len(description) < 10:
                return ""

            # Clean and truncate description
            clean_desc = self._clean_html(description)
            if len(clean_desc) > 150:
                clean_desc = clean_desc[:150]
                last_period = clean_desc.rfind('.')
                if last_period > 100:
                    clean_desc = clean_desc[:last_period + 1]
                else:
                    clean_desc += "..."

            return f"💭 {clean_desc}\n"

        except Exception:
            return ""

    def _format_sources(self, restaurant: Dict[str, Any], source_type: str) -> str:
        """
        Format 'Recommended by [sources]' line
        """
        try:
            sources = restaurant.get('sources', [])

            if source_type == "database":
                if sources:
                    source_names = sources[:3]  # Limit to 3 sources
                    sources_text = ', '.join(source_names)
                    return f"📚 Recommended by {sources_text}\n"
                else:
                    return "📚 From my restaurant notes\n"

            elif source_type == "google_maps":
                if restaurant.get('media_verified', False) and sources:
                    source_names = sources[:2]  # Limit to 2 sources for space
                    sources_text = ', '.join(source_names)
                    return f"📰 Featured in {sources_text}\n"
                else:
                    return "🗺️ Found via Google Maps\n"

            return ""

        except Exception:
            return ""

    def _clean_html(self, text: str) -> str:
        """
        Clean text for HTML display (escape special characters)
        """
        try:
            if not text:
                return ""

            # Basic HTML escaping
            cleaned = escape(text)

            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())

            return cleaned

        except Exception:
            return str(text) if text else ""

    def _create_empty_response(self, query: str, source_type: str) -> Dict[str, Any]:
        """
        Create empty response when no results found
        """
        if source_type == "database":
            message = (
                f"🗂️ <b>No restaurants found in my notes for this location.</b>\n\n"
                "Would you like me to search Google Maps for restaurants nearby?"
            )
        else:
            message = (
                f"🗺️ <b>No restaurants found nearby for your search.</b>\n\n"
                "Try a different query or search a bit further from your location."
            )

        return {
            "main_list": [],
            "formatted_message": message,
            "source_type": source_type,
            "search_info": {
                "query": query,
                "count": 0,
                "empty_result": True
            }
        }