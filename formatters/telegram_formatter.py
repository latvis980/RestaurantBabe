# formatters/location_telegram_formatter.py
"""
FIXED Location-specific Telegram formatter - Proper HTML handling

CRITICAL FIX: 
- Fixed _clean_html() method that was double-escaping HTML
- Fixed Google Maps link formatting  
- Proper handling of Telegram HTML tags
- Better formatting for database and maps results

This was causing raw HTML to appear in the Telegram bot because the formatter
was escaping HTML tags then trying to remove them, which doesn't work.
"""

import re
import logging
from html import escape
from typing import Dict, List, Any, Union
from urllib.parse import urlparse, quote

from formatters.google_links import build_google_maps_url

logger = logging.getLogger(__name__)

class LocationTelegramFormatter:
    """
    FIXED: Location-specific Telegram formatter with proper HTML handling
    """

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config=None):
        self.config = config

    def format_database_results(
        self,
        restaurants: List[Union[Dict[str, Any], Any]], 
        query: str,
        location_description: str,
        offer_more_search: bool = True
    ) -> Dict[str, Any]:
        """
        Format database search results with proper HTML handling

        Args:
            restaurants: List of restaurant data from database
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
            header = "📝 <b>From my personal restaurant notes:</b>\n\n"

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

            # Truncate if necessary
            if len(full_message) > self.MAX_MESSAGE_LENGTH:
                full_message = self._truncate_message(full_message, message_parts, header)

            return {
                "message": full_message,
                "restaurant_count": len(restaurants),
                "has_choice": offer_more_search
            }

        except Exception as e:
            logger.error(f"❌ Error formatting database results: {e}")
            return {
                "message": "Sorry, I had trouble formatting the restaurant information.",
                "restaurant_count": 0,
                "has_choice": False
            }

    def format_google_maps_results(
        self,
        venues: List[Union[Dict[str, Any], Any]],
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """
        Format Google Maps search results with media verification
        """
        try:
            if not venues:
                return {
                    "message": f"😔 No restaurants found near {location_description}. Try expanding the search area or be more specific?",
                    "restaurant_count": 0,
                    "has_choice": False
                }

            # Build message parts
            message_parts = []
            header = f"🗺️ <b>Found these restaurants near {location_description}:</b>\n\n"

            for i, venue in enumerate(venues[:10], 1):  # Limit to 10 venues
                formatted_venue = self._format_verified_venue(venue, i)
                message_parts.append(formatted_venue)

            # Combine all parts
            full_message = header + "\n\n".join(message_parts)

            # Truncate if necessary
            if len(full_message) > self.MAX_MESSAGE_LENGTH:
                full_message = self._truncate_message(full_message, message_parts, header)

            return {
                "message": full_message,
                "restaurant_count": len(venues),
                "has_choice": False
            }

        except Exception as e:
            logger.error(f"❌ Error formatting Google Maps results: {e}")
            return {
                "message": f"Found restaurants near {location_description} but had trouble formatting them.",
                "restaurant_count": len(venues) if venues else 0,
                "has_choice": False
            }

    def _format_single_restaurant(self, restaurant: Union[Dict[str, Any], Any], index: int) -> str:
        """
        Format a single restaurant from database with proper HTML handling
        """
        try:
            # Restaurant name with index
            name = self._get_value(restaurant, 'name', 'Unknown Restaurant')
            formatted_name = f"<b>{index}. {self._clean_html_preserve_tags(name)}</b>\n"

            # Address with canonical Google Maps link
            address_line = self._format_address_link(restaurant)

            # Distance with proper "0.1 km" format
            distance_line = self._format_distance_enhanced(restaurant)

            # Enhanced description with more content from database
            description_line = self._format_description_enhanced(restaurant)

            # Sources showing only domain names
            sources_line = self._format_sources_domains_only(restaurant)

            # Combine all parts
            result = f"{formatted_name}{address_line}{distance_line}{description_line}{sources_line}"

            return result

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {self._get_value(restaurant, 'name', 'Unknown')}: {e}")
            return f"<b>{index}. {self._get_value(restaurant, 'name', 'Unknown Restaurant')}</b>\nInformation unavailable\n"

    def _format_verified_venue(self, venue: Union[Dict[str, Any], Any], index: int) -> str:
        """
        Format a single verified venue from Google Maps with media verification
        """
        try:
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
            formatted_name = f"<b>{index}. {self._clean_html_preserve_tags(name)}</b>\n"

            # Address with Google Maps link
            if place_id:
                google_url = build_google_maps_url(place_id, name)
            elif google_maps_url:
                google_url = google_maps_url
            else:
                encoded_query = quote(f"{name} {address}")
                google_url = f"https://www.google.com/maps/search/?api=1&query={encoded_query}"

            clean_address = self._extract_street_address(address)
            address_line = f'📍 <a href="{google_url}">{clean_address}</a>\n'

            # Distance and rating
            details_parts = []
            if distance_km is not None:
                details_parts.append(f"{distance_km:.1f} km")
            if rating:
                details_parts.append(f"⭐ {rating}")

            details_line = f"📊 {' • '.join(details_parts)}\n" if details_parts else ""

            # Description
            if description:
                clean_desc = self._clean_html_preserve_tags(description)
                description_line = f"{clean_desc}\n"
            else:
                description_line = ""

            # Sources (if media verified)
            sources_line = ""
            if media_verified and sources:
                domain_names = []
                for source in sources:
                    domain = self._extract_domain_from_url(source)
                    if domain and domain not in domain_names:
                        domain_names.append(domain)

                if domain_names:
                    sources_text = ", ".join(domain_names[:3])
                    sources_line = f"<i>✅ Featured in: {sources_text}</i>\n"

            return f"{formatted_name}{address_line}{details_line}{description_line}{sources_line}"

        except Exception as e:
            logger.error(f"❌ Error formatting venue {self._get_value(venue, 'name', 'Unknown')}: {e}")
            return f"<b>{index}. {self._get_value(venue, 'name', 'Unknown Restaurant')}</b>\nInformation unavailable\n"

    def _format_distance_enhanced(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """Format distance with proper spacing"""
        try:
            distance_km = self._get_value(restaurant, 'distance_km')
            if distance_km is not None:
                return f"📏 {distance_km:.1f} km\n"
            return ""
        except Exception:
            return ""

    def _format_description_enhanced(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """Format enhanced description from database"""
        try:
            description = self._get_value(restaurant, 'description', '')
            if description:
                clean_desc = self._clean_html_preserve_tags(description)
                return f"{clean_desc}\n"
            return ""
        except Exception:
            return ""

    def _format_sources_domains_only(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """Format sources showing only domain names"""
        try:
            sources = self._get_value(restaurant, 'sources', [])
            if not sources:
                return ""

            domain_names = []
            for source in sources:
                domain = self._extract_domain_from_url(source)
                if domain and domain not in domain_names:
                    domain_names.append(domain)

            if domain_names:
                sources_text = ", ".join(domain_names[:3])
                return f"<i>✅ Recommended by: {sources_text}</i>\n"

            return ""

        except Exception as e:
            logger.debug(f"Error formatting sources: {e}")
            return ""

    def _extract_domain_from_url(self, source: str) -> str:
        """Extract domain from URL"""
        try:
            if '://' in source or source.startswith('www.'):
                parsed_url = urlparse(source if '://' in source else f'http://{source}')
                domain = parsed_url.netloc.lower()

                # Remove 'www.' prefix if present
                if domain.startswith('www.'):
                    domain = domain[4:]

                return domain if domain else source
            else:
                # Not a URL, return as-is (might be a publication name)
                return source

        except Exception as e:
            logger.debug(f"Could not parse URL {source}: {e}")
            return source

    def _format_address_link(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        Create address line with Google Maps link
        """
        try:
            address = self._get_value(restaurant, 'address', '')
            place_id = self._get_value(restaurant, 'place_id', '')
            name = self._get_value(restaurant, 'name', '')

            if not address:
                return ""

            # Create Google Maps URL
            if place_id:
                google_url = build_google_maps_url(place_id, name)
            else:
                # Fallback to search-based URL
                encoded_name = quote(f"{name} {address}")
                google_url = f"https://www.google.com/maps/search/?api=1&query={encoded_name}"

            clean_address = self._extract_street_address(address)
            return f'📍 <a href="{google_url}">{self._clean_html_preserve_tags(clean_address)}</a>\n'

        except Exception as e:
            logger.debug(f"Error formatting address link: {e}")
            return ""

    def _extract_street_address(self, full_address: str) -> str:
        """
        Extract street address and city, removing postal codes and country
        """
        try:
            if not full_address:
                return "Address available"

            parts = str(full_address).split(',')

            # Remove country (common country names)
            if len(parts) >= 2:
                last_part = parts[-1].strip()
                if last_part in ['United States', 'United Kingdom', 'Portugal', 'France', 'Germany', 'Spain', 'Italy']:
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

    def _get_value(self, obj: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
        """
        Safely get value from either dictionary or dataclass object
        """
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            else:
                return getattr(obj, key, default)
        except Exception:
            return default

    def _clean_html_preserve_tags(self, text: str) -> str:
        """
        FIXED: Properly clean HTML content while preserving Telegram HTML tags

        This was the main issue - the old method was escaping HTML first,
        then trying to remove tags, which doesn't work.
        """
        if not text:
            return ""

        text = str(text).strip()

        # Remove unwanted HTML tags but preserve Telegram-allowed tags (<b>, <i>, <a>, <code>, <pre>)
        # This regex removes tags that are NOT in the Telegram-allowed list
        text = re.sub(r'<(?!/?(?:b|i|a|code|pre|strong|em)(?:\s[^>]*)?)[^>]*>', '', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Only escape ampersands that could break HTML parsing
        # Don't escape < and > since we want to preserve valid Telegram HTML tags
        text = re.sub(r'&(?!(?:amp|lt|gt|quot|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)

        return text.strip()

    def _clean_html(self, text: str) -> str:
        """Legacy method - redirects to the fixed version"""
        return self._clean_html_preserve_tags(text)

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