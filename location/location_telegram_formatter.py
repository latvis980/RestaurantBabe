# location/location_telegram_formatter.py
"""
Unified Location Results Formatter - FIXED for RestaurantDescription dataclass objects

FIXED: Handle both dictionary and RestaurantDescription dataclass objects
"""

import logging
import re
import requests
from typing import Dict, List, Any, Optional, Union
from html import escape
from urllib.parse import quote

logger = logging.getLogger(__name__)

class LocationTelegramFormatter:
    """
    Unified formatter for all location-based search results
    FIXED: Handles both dictionaries and RestaurantDescription dataclass objects
    """

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config=None):
        """Initialize formatter with optional config"""
        self.config = config

    def format_database_results(
        self,
        restaurants: List[Union[Dict[str, Any], Any]],  # FIXED: Accept both dicts and dataclass objects
        query: str,
        location_description: str,
        offer_more_search: bool = True
    ) -> Dict[str, Any]:
        """
        Format database results (Step 2a) with user choice option

        Args:
            restaurants: List of restaurant dictionaries OR RestaurantDescription objects
            query: Original user query
            location_description: Description of the searched location
            offer_more_search: Whether to offer additional Google Maps search

        Returns:
            Formatted results dictionary for telegram bot
        """
        try:
            if not restaurants:
                return {
                    "message": "No restaurants found in my notes for this area.",
                    "restaurant_count": 0,
                    "has_choice": False
                }

            # Build the results message
            message_parts = []

            # Header with personal notes context
            header = f"📝 <b>From my personal restaurant notes:</b>\n\n"

            # Format each restaurant
            for i, restaurant in enumerate(restaurants[:6], 1):  # Limit to 6 for choice flow
                formatted_restaurant = self._format_single_restaurant(restaurant, i)
                message_parts.append(formatted_restaurant)

            # Combine all parts
            full_message = header + "\n\n".join(message_parts)

            # Add choice explanation if offering more search
            if offer_more_search:
                choice_text = (
                    "\n\n💡 <b>These are from my curated collection.</b> "
                    "Would you like to see these options or search for more restaurants in the area?"
                )
                full_message += choice_text

            # Ensure message isn't too long
            if len(full_message) > self.MAX_MESSAGE_LENGTH:
                full_message = self._truncate_message(full_message, message_parts, header)

            return {
                "message": full_message,
                "restaurant_count": len(restaurants),
                "has_choice": offer_more_search,
                "formatted_restaurants": message_parts
            }

        except Exception as e:
            logger.error(f"❌ Error formatting database results: {e}")
            return {
                "message": f"Found {len(restaurants)} restaurants but had trouble formatting them.",
                "restaurant_count": len(restaurants),
                "has_choice": offer_more_search
            }

    def format_google_maps_results(
        self,
        venues: List[Any],  # VenueResult objects or verified venue dicts
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """
        Format Google Maps results with media verification (final results)

        UPDATED: Proper canonical place ID link formatting
        """
        try:
            if not venues:
                return {
                    "message": "No restaurants found after verification.",
                    "venues": [],
                    "restaurant_count": 0
                }

            # Build the results message
            message_parts = []

            # Header indicating results are from external search with verification
            header = "🔍 <b>Here's what I found in the area:</b>\n\n"

            # Format each venue
            for i, venue in enumerate(venues[:8], 1):  # Limit to 8 results
                formatted_venue = self._format_verified_venue(venue, i)
                message_parts.append(formatted_venue)

            # Combine all parts
            full_message = header + "\n\n".join(message_parts)

            # Add footer note about sources
            footer = (
                "\n\n<i>Verified through professional food guides and local media. "
                "Click addresses to view on Google Maps.</i>"
            )
            full_message += footer

            # Ensure message isn't too long
            if len(full_message) > self.MAX_MESSAGE_LENGTH:
                full_message = self._truncate_message(full_message, message_parts, header, footer)

            return {
                "message": full_message,
                "venues": venues,
                "restaurant_count": len(venues)
            }

        except Exception as e:
            logger.error(f"❌ Error formatting Google Maps results: {e}")
            return {
                "venues": venues if venues else [],
                "restaurant_count": len(venues) if venues else 0,
                "message": f"Found {len(venues)} restaurants but had trouble formatting them." if venues else "No restaurants found."
            }

    def _get_value(self, obj: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
        """
        FIXED: Safely get value from either dictionary or dataclass object

        Args:
            obj: Either a dictionary or dataclass object (like RestaurantDescription)
            key: The key/attribute name to get
            default: Default value if key/attribute doesn't exist

        Returns:
            The value from the object or default
        """
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            else:
                # Handle dataclass objects (like RestaurantDescription)
                return getattr(obj, key, default)
        except Exception:
            return default

    def _format_single_restaurant(self, restaurant: Union[Dict[str, Any], Any], index: int) -> str:
        """
        FIXED: Format a single restaurant from database (handles both dicts and dataclass objects)
        """
        try:
            # Restaurant name with index - FIXED to handle both dicts and dataclass objects
            name = self._get_value(restaurant, 'name', 'Unknown Restaurant')
            formatted_name = f"<b>{index}. {self._clean_html(name)}</b>\n"

            # Address with canonical Google Maps link
            address_line = self._format_address_link(restaurant)

            # Distance from user
            distance_line = self._format_distance(restaurant)

            # Description
            description_line = self._format_description(restaurant)

            # Sources/recommendations
            sources_line = self._format_sources(restaurant)

            # Combine all parts
            return f"{formatted_name}{address_line}{distance_line}{description_line}{sources_line}"

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {self._get_value(restaurant, 'name', 'Unknown')}: {e}")
            return f"<b>{index}. {self._get_value(restaurant, 'name', 'Unknown Restaurant')}</b>\nInformation unavailable\n"

    def _format_verified_venue(self, venue: Union[Dict[str, Any], Any], index: int) -> str:
        """
        Format a single verified venue from Google Maps with media verification
        UPDATED: Uses 2025 universal format for address links
        """
        try:
            # FIXED: Handle both VenueResult objects and dictionaries using universal getter
            name = self._get_value(venue, 'name', 'Unknown Restaurant')
            address = self._get_value(venue, 'address', '')
            distance_km = self._get_value(venue, 'distance_km')
            rating = self._get_value(venue, 'rating')
            place_id = self._get_value(venue, 'place_id')
            description = self._get_value(venue, 'description', '')
            sources = self._get_value(venue, 'sources', [])
            media_verified = self._get_value(venue, 'media_verified', False)
            google_maps_url = self._get_value(venue, 'google_maps_url', '')

            # Restaurant name with index
            formatted_name = f"<b>{index}. {self._clean_html(name)}</b>\n"

            # Address with 2025 universal Google Maps link
            universal_url = self._get_canonical_google_maps_url(place_id, name, google_maps_url)
            clean_address = self._extract_street_address(address)
            address_line = f'📍 <a href="{escape(universal_url, quote=True)}">{self._clean_html(clean_address)}</a>\n'

            # Distance from user
            distance_line = ""
            if distance_km is not None:
                distance_text = self._format_distance_km(distance_km)
                distance_line = f"📏 {distance_text}\n"

            # Rating (if available)
            rating_line = ""
            if rating and rating > 0:
                rating_stars = "⭐" * min(int(round(rating)), 5)
                rating_line = f"{rating_stars} {rating:.1f}\n"

            # Description from media sources
            description_line = ""
            if description and description.strip():
                clean_description = self._clean_html(description.strip())
                description_line = f"💭 {clean_description}\n"

            # Media verification status
            sources_line = ""
            if media_verified and sources:
                sources_text = ", ".join(sources[:4])  # Show up to 4 sources
                if len(sources) > 2:
                    sources_text += f" +{len(sources)-2} more"
                sources_line = f"✅ Recommended by {sources_text}\n"
            elif media_verified:
                sources_line = "✅ Verified in professional guides\n"

            # Combine all parts
            return f"{formatted_name}{address_line}{distance_line}{rating_line}{description_line}{sources_line}"

        except Exception as e:
            logger.error(f"❌ Error formatting venue {self._get_value(venue, 'name', 'Unknown')}: {e}")
            venue_name = self._get_value(venue, 'name', 'Unknown Restaurant')
            return f"<b>{index}. {venue_name}</b>\nInformation unavailable\n"

    def _get_canonical_google_maps_url(self, place_id: str, restaurant_name: str, fallback_url: str = "") -> str:
        """
        Create canonical Google Maps URL using place_id for consistent addressing
        Universal format for 2025 - works on all devices
        """
        try:
            if place_id:
                # Use place_id for canonical, device-agnostic Google Maps URL
                return f"https://maps.google.com/?cid={place_id}&q={quote(restaurant_name, safe='')}"
            elif fallback_url:
                return fallback_url
            else:
                # Fallback to search query if no place_id
                return f"https://maps.google.com/?q={quote(restaurant_name, safe='')}"
        except Exception:
            return fallback_url or "https://maps.google.com/"

    def _format_address_link(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        FIXED: Format address with Google Maps link (handles both dicts and dataclass objects)
        """
        try:
            address = self._get_value(restaurant, 'address', '')
            place_id = self._get_value(restaurant, 'place_id', '')
            restaurant_name = self._get_value(restaurant, 'name', '')
            google_maps_url = self._get_value(restaurant, 'google_maps_url', '')

            if not address:
                return "📍 Address available\n"

            clean_street = self._extract_street_address(address)

            google_url = self._get_canonical_google_maps_url(
                place_id, 
                restaurant_name, 
                google_maps_url
            )

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

    def _format_distance(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        FIXED: Format distance from user (handles both dicts and dataclass objects)
        """
        try:
            distance_km = self._get_value(restaurant, 'distance_km')
            distance_text = self._get_value(restaurant, 'distance_text')

            if distance_text:
                return f"📏 {distance_text}\n"
            elif distance_km is not None:
                return f"📏 {self._format_distance_km(distance_km)}\n"
            else:
                return ""

        except Exception:
            return ""

    def _format_distance_km(self, distance_km: float) -> str:
        """Format distance in km to readable text"""
        try:
            if distance_km < 1.0:
                return f"{int(distance_km * 1000)}m away"
            else:
                return f"{distance_km:.1f}km away"
        except Exception:
            return "Distance unknown"

    def _format_description(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        FIXED: Format restaurant description with robust field handling (handles both dicts and dataclass objects)
        """
        try:
            # Check both possible description field names
            description = (self._get_value(restaurant, 'description', '').strip() or 
                          self._get_value(restaurant, 'raw_description', '').strip())

            if not description:
                return ""

            # Clean and truncate description
            clean_description = self._clean_html(description)
            if len(clean_description) > 150:
                clean_description = clean_description[:150] + "..."

            return f"💭 {clean_description}\n"

        except Exception as e:
            logger.debug(f"Error formatting description: {e}")
            return ""

    def _format_sources(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        FIXED: Format restaurant sources/recommendations with robust handling (handles both dicts and dataclass objects)
        """
        try:
            sources = self._get_value(restaurant, 'sources', [])

            if not sources:
                # Also check for media_sources field (alternative name)
                sources = self._get_value(restaurant, 'media_sources', [])

            if not sources:
                return ""

            if isinstance(sources, list) and sources:
                # Filter out empty strings
                valid_sources = [s for s in sources if s and str(s).strip()]
                if not valid_sources:
                    return ""

                sources_text = ", ".join(valid_sources[:2])  # Show max 2 sources
                if len(valid_sources) > 2:
                    sources_text += f" +{len(valid_sources)-2} more"
                return f"📚 From {sources_text}\n"

            elif isinstance(sources, str) and sources.strip():
                return f"📚 From {sources.strip()}\n"
            else:
                return ""

        except Exception as e:
            logger.debug(f"Error formatting sources: {e}")
            return ""

    def _clean_html(self, text: str) -> str:
        """Clean and escape text for Telegram HTML parsing"""
        try:
            if not text:
                return ""

            # Remove any existing HTML tags
            text = re.sub(r'<[^>]+>', '', text)

            # Escape HTML entities for Telegram
            text = escape(text)

            return text.strip()
        except Exception:
            return ""

    def _truncate_message(self, full_message: str, message_parts: List[str], header: str, footer: str = "") -> str:
        """Truncate message to fit Telegram limits"""
        try:
            # Calculate available space
            available_space = self.MAX_MESSAGE_LENGTH - len(header) - len(footer) - 100  # Buffer

            # Include as many restaurants as possible
            truncated_parts = []
            current_length = 0

            for part in message_parts:
                if current_length + len(part) <= available_space:
                    truncated_parts.append(part)
                    current_length += len(part)
                else:
                    break

            # Add "more results" note if we truncated
            if len(truncated_parts) < len(message_parts):
                truncated_parts.append(f"\n<i>... and {len(message_parts) - len(truncated_parts)} more results</i>")

            return header + "\n\n".join(truncated_parts) + footer

        except Exception:
            return header + "Found restaurants but had trouble formatting them." + footer