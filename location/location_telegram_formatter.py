# location/location_telegram_formatter.py
"""
Unified Location Results Formatter - UPDATED for Google Maps with canonical addresses

UPDATED: Proper canonical place ID link formatting for Google Maps results
Ensures all addresses are formatted consistently using place_id links
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
    UPDATED: Canonical Place ID formatting for all results
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
            header = f"🔍 <b>Here's what I found in the area:</b>\n\n"

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

    def _format_single_restaurant(self, restaurant: Dict[str, Any], index: int) -> str:
        """Format a single restaurant from database"""
        try:
            # Restaurant name with index
            name = restaurant.get('name', 'Unknown Restaurant')
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
            logger.error(f"❌ Error formatting restaurant {restaurant.get('name', 'Unknown')}: {e}")
            return f"<b>{index}. {restaurant.get('name', 'Unknown Restaurant')}</b>\nInformation unavailable\n"

    def _format_verified_venue(self, venue: Dict[str, Any], index: int) -> str:
        """
        Format a single verified venue from Google Maps with media verification

        UPDATED: Handles both VenueResult objects and verified venue dicts
        """
        try:
            # Handle both VenueResult objects and dictionaries
            if hasattr(venue, 'name'):
                # VenueResult object
                name = venue.name
                address = venue.address
                distance_km = venue.distance_km
                rating = venue.rating
                place_id = venue.place_id
                description = getattr(venue, 'description', '')
                sources = getattr(venue, 'sources', [])
                media_verified = getattr(venue, 'media_verified', False)
                google_maps_url = venue.google_maps_url
            else:
                # Dictionary (verified venue)
                name = venue.get('name', 'Unknown Restaurant')
                address = venue.get('address', '')
                distance_km = venue.get('distance_km')
                rating = venue.get('rating')
                place_id = venue.get('place_id')
                description = venue.get('description', '')
                sources = venue.get('sources', [])
                media_verified = venue.get('media_verified', False)
                google_maps_url = venue.get('google_maps_url', '')

            # Restaurant name with index
            formatted_name = f"<b>{index}. {self._clean_html(name)}</b>\n"

            # Address with canonical Google Maps link (UPDATED)
            canonical_url = self._get_canonical_google_maps_url(place_id, google_maps_url)
            clean_address = self._extract_street_address(address)
            address_line = f'📍 <a href="{escape(canonical_url, quote=True)}">{self._clean_html(clean_address)}</a>\n'

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
                sources_text = ", ".join(sources[:2])  # Show max 2 sources
                if len(sources) > 2:
                    sources_text += f" +{len(sources)-2} more"
                sources_line = f"✅ Recommended by {sources_text}\n"
            elif media_verified:
                sources_line = "✅ Verified in professional guides\n"

            # Combine all parts
            return f"{formatted_name}{address_line}{distance_line}{rating_line}{description_line}{sources_line}"

        except Exception as e:
            logger.error(f"❌ Error formatting venue {venue.get('name', 'Unknown') if isinstance(venue, dict) else getattr(venue, 'name', 'Unknown')}: {e}")
            venue_name = venue.get('name', 'Unknown Restaurant') if isinstance(venue, dict) else getattr(venue, 'name', 'Unknown Restaurant')
            return f"<b>{index}. {venue_name}</b>\nInformation unavailable\n"

    def _get_canonical_google_maps_url(self, place_id: str, existing_url: str = "") -> str:
        """
        Get canonical Google Maps URL using place ID or CID (COPIED from location formatter)
        """
        try:
            # Check for existing canonical URL with CID
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
            google_url = self._get_canonical_google_maps_url(place_id, restaurant.get('google_maps_url', ''))

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

    def _format_distance(self, restaurant: Dict[str, Any]) -> str:
        """Format distance from user"""
        try:
            distance_km = restaurant.get('distance_km')
            distance_text = restaurant.get('distance_text')

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

    def _format_description(self, restaurant: Dict[str, Any]) -> str:
        """Format restaurant description"""
        try:
            description = restaurant.get('description', '').strip()

            if not description:
                return ""

            # Clean and truncate description
            clean_description = self._clean_html(description)
            if len(clean_description) > 150:
                clean_description = clean_description[:150] + "..."

            return f"💭 {clean_description}\n"

        except Exception:
            return ""

    def _format_sources(self, restaurant: Dict[str, Any]) -> str:
        """Format restaurant sources/recommendations"""
        try:
            sources = restaurant.get('sources', [])

            if not sources:
                return ""

            if isinstance(sources, list) and sources:
                sources_text = ", ".join(sources[:2])  # Show max 2 sources
                if len(sources) > 2:
                    sources_text += f" +{len(sources)-2} more"
                return f"📚 From {sources_text}\n"
            elif isinstance(sources, str) and sources.strip():
                return f"📚 From {sources.strip()}\n"
            else:
                return ""

        except Exception:
            return ""

    def _clean_html(self, text: str) -> str:
        """Clean text for HTML display"""
        try:
            if not text:
                return ""

            # Escape HTML characters
            cleaned = escape(str(text))

            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())

            return cleaned

        except Exception:
            return str(text) if text else ""

    def _truncate_message(self, full_message: str, message_parts: List[str], header: str, footer: str = "") -> str:
        """Truncate message to fit Telegram limits"""
        try:
            # Calculate available space for restaurants
            overhead = len(header) + len(footer) + 100  # Safety margin
            available_space = self.MAX_MESSAGE_LENGTH - overhead

            # Add restaurants until we hit the limit
            truncated_parts = []
            current_length = 0

            for part in message_parts:
                if current_length + len(part) + 10 < available_space:  # +10 for separators
                    truncated_parts.append(part)
                    current_length += len(part) + 10
                else:
                    break

            # Build truncated message
            if truncated_parts:
                truncated_message = header + "\n\n".join(truncated_parts)
                if len(truncated_parts) < len(message_parts):
                    truncated_message += f"\n\n<i>... and {len(message_parts) - len(truncated_parts)} more results</i>"
                truncated_message += footer
                return truncated_message
            else:
                return header + "Results too long to display." + footer

        except Exception as e:
            logger.error(f"❌ Error truncating message: {e}")
            return full_message[:self.MAX_MESSAGE_LENGTH-3] + "..."

    def format_location_summary(self, location_data) -> str:
        """Format location data for display"""
        try:
            if location_data.latitude and location_data.longitude:
                return f"GPS: {location_data.latitude:.4f}, {location_data.longitude:.4f}"
            elif location_data.description:
                return location_data.description
            else:
                return "Unknown location"
        except:
            return "Location information unavailable"