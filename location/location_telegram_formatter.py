"""
Unified Location Results Formatter - UPDATED for user choice flow

Creates consistent format for ALL location-based results:
- Database results (Step 2a) - with user choice option
- Google Maps results (Steps 3-5) - final results
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
    UPDATED: Properly handles user choice flow
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
        venues: List[Any],  # VenueResult objects
        query: str,
        location_description: str
    ) -> Dict[str, Any]:
        """
        Format Google Maps results (final results, no choice needed)

        Args:
            venues: List of VenueResult objects from Google Maps
            query: Original user query
            location_description: Description of the searched location

        Returns:
            Formatted results dictionary
        """
        try:
            if not venues:
                return {
                    "message": "No additional restaurants found through Google Maps search.",
                    "venues": [],
                    "restaurant_count": 0
                }

            # Convert VenueResult objects to restaurant format for existing telegram formatter
            formatted_venues = []

            for venue in venues:
                restaurant_dict = {
                    "name": venue.name,
                    "address": venue.address,
                    "distance_km": venue.distance_km,
                    "rating": venue.rating,
                    "user_ratings_total": venue.user_ratings_total,
                    "price_level": venue.price_level,
                    "google_maps_url": venue.google_maps_url,
                    "place_id": venue.place_id,
                    "types": venue.types,
                    # Add source information
                    "source": "google_maps",
                    "verification_status": "verified" if hasattr(venue, 'verified') else "unverified"
                }
                formatted_venues.append(restaurant_dict)

            return {
                "venues": formatted_venues,
                "restaurant_count": len(venues),
                "message": f"Found {len(venues)} restaurants through Google Maps search"
            }

        except Exception as e:
            logger.error(f"❌ Error formatting Google Maps results: {e}")
            return {
                "venues": [],
                "restaurant_count": 0,
                "message": "Found restaurants but had trouble formatting them."
            }

    def _format_single_restaurant(self, restaurant: Dict[str, Any], index: int) -> str:
        """Format a single restaurant for display"""
        try:
            # Restaurant name with index
            name = restaurant.get('name', 'Unknown Restaurant')
            formatted_name = f"<b>{index}. {escape(name)}</b>"

            # Distance
            distance = restaurant.get('distance_km')
            distance_text = f"📍 {distance:.1f}km away" if distance else "📍 Distance unknown"

            # Address (if available)
            address = restaurant.get('address', '').strip()
            address_text = f"📍 {escape(address[:50])}{'...' if len(address) > 50 else ''}" if address else ""

            # Cuisine information
            cuisine_tags = restaurant.get('cuisine_tags', [])
            if cuisine_tags:
                cuisine_text = f"🍽️ {', '.join(cuisine_tags[:3])}"  # Show up to 3 cuisine types
            else:
                cuisine_text = ""

            # Description (truncated)
            description = restaurant.get('raw_description', '') or restaurant.get('description', '')
            if description:
                # Clean and truncate description
                clean_desc = re.sub(r'<[^>]+>', '', description)  # Remove HTML
                clean_desc = ' '.join(clean_desc.split())  # Normalize whitespace
                desc_text = f"💭 {escape(clean_desc[:100])}{'...' if len(clean_desc) > 100 else ''}"
            else:
                desc_text = ""

            # Rating (if available)
            rating = restaurant.get('rating')
            rating_text = f"⭐ {rating:.1f}" if rating else ""

            # Combine all parts
            parts = [formatted_name]

            if distance_text:
                parts.append(distance_text)
            elif address_text:  # Use address if no distance
                parts.append(address_text)

            if cuisine_text:
                parts.append(cuisine_text)

            if rating_text:
                parts.append(rating_text)

            if desc_text:
                parts.append(desc_text)

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {restaurant.get('name', 'unknown')}: {e}")
            return f"<b>{index}. {escape(restaurant.get('name', 'Unknown Restaurant'))}</b>\n📍 Information unavailable"

    def _truncate_message(self, full_message: str, message_parts: List[str], header: str) -> str:
        """Truncate message if too long while preserving structure"""
        try:
            # Calculate space for header and footer
            footer = "\n\n💡 <b>These are from my curated collection.</b> Would you like to see these options or search for more restaurants in the area?"
            reserved_space = len(header) + len(footer) + 100  # Buffer

            available_space = self.MAX_MESSAGE_LENGTH - reserved_space

            # Include as many restaurants as possible
            included_parts = []
            current_length = 0

            for part in message_parts:
                part_length = len(part) + 2  # +2 for \n\n separator
                if current_length + part_length <= available_space:
                    included_parts.append(part)
                    current_length += part_length
                else:
                    break

            if not included_parts:
                # Fallback: include at least one restaurant
                included_parts = [message_parts[0][:available_space - 50] + "..."]

            # Add truncation notice if needed
            if len(included_parts) < len(message_parts):
                truncation_notice = f"\n\n<i>... and {len(message_parts) - len(included_parts)} more restaurants</i>"
                included_parts.append(truncation_notice)

            return header + "\n\n".join(included_parts) + footer

        except Exception as e:
            logger.error(f"❌ Error truncating message: {e}")
            return header + "Found restaurants but had trouble formatting them." + footer

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