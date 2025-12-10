# formatters/location_telegram_formatter.py
"""
Location-specific Telegram formatter - UPDATED with better formatting

FIXES:
- Longer, more detailed descriptions from database
- Distance in "0.1 km" format with space after number  
- Sources show only domain names, not full URLs
- Better use of database description content
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
    UPDATED: Enhanced Location-specific Telegram formatter with better descriptions and sources
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
        UPDATED: Format database search results with enhanced descriptions and proper sources

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

    def _format_single_restaurant(self, restaurant: Union[Dict[str, Any], Any], index: int) -> str:
        """
        ENHANCED DEBUG VERSION: Format a single restaurant from database with enhanced descriptions and sources
        """
        try:
            # Restaurant name with index
            name = self._get_value(restaurant, 'name', 'Unknown Restaurant')
            formatted_name = f"<b>{index}. {self._clean_html(name)}</b>\n"

            # DEBUG: Log restaurant processing
            logger.info(f"🔍 DEBUG - Processing restaurant {index}: {name}")
            logger.info(f"🔍 DEBUG - Restaurant type: {type(restaurant)}")

            # Address with canonical Google Maps link
            address_line = self._format_address_link(restaurant)

            # UPDATED: Distance with proper "0.1 km" format
            distance_line = self._format_distance_enhanced(restaurant)

            # UPDATED: Enhanced description with more content from database
            description_line = self._format_description_enhanced(restaurant)

            # UPDATED: Sources showing only domain names WITH DEBUG
            logger.info(f"🔍 DEBUG - About to format sources for {name}...")
            sources_line = self._format_sources_domains_only(restaurant)
            logger.info(f"🔍 DEBUG - Sources line result: '{sources_line}'")

            # Combine all parts
            result = f"{formatted_name}{address_line}{distance_line}{description_line}{sources_line}"

            # DEBUG: Log final result
            logger.info(f"🔍 DEBUG - Final formatted restaurant {index}:")
            logger.info(f"🔍 DEBUG - {result}")

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
            formatted_name = f"<b>{index}. {self._clean_html(name)}</b>\n"

            # Address with universal Google Maps link
            if place_id:
                universal_url = build_google_maps_url(place_id, name)
            else:
                universal_url = google_maps_url or "#"
            clean_address = self._extract_street_address(address)
            address_line = f'📍 <a href="{escape(universal_url, quote=True)}">{self._clean_html(clean_address)}</a>\n'

            # UPDATED: Distance with proper format
            distance_line = ""
            if distance_km is not None:
                distance_text = self._format_distance_km_enhanced(distance_km)
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

            # UPDATED: Media verification status with clickable domain links
            sources_line = ""
            if media_verified and sources:
                domains = self._extract_domains_from_sources(sources)
                if domains:
                    # Create clickable links for domains
                    domain_links = []
                    for domain in domains[:3]:
                        clean_domain = self._clean_html(domain)
                        domain_url = f"https://{domain}"
                        domain_links.append(f'<a href="{escape(domain_url, quote=True)}">{clean_domain}</a>')

                    sources_text = ", ".join(domain_links)
                    if len(domains) > 3:
                        sources_text += f" +{len(domains)-3} more"
                    sources_line = f"🔷 Recommended by {sources_text}\n"
            elif media_verified:
                sources_line = "🔷 Verified in professional guides\n"

            return f"{formatted_name}{address_line}{distance_line}{rating_line}{description_line}{sources_line}"

        except Exception as e:
            logger.error(f"❌ Error formatting venue {self._get_value(venue, 'name', 'Unknown')}: {e}")
            venue_name = self._get_value(venue, 'name', 'Unknown Restaurant')
            return f"<b>{index}. {self._clean_html(venue_name)}</b>\nInformation unavailable\n"

    def _format_distance_enhanced(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        UPDATED: Format distance with proper "0.1 km" format and space after number
        """
        try:
            distance_km = self._get_value(restaurant, 'distance_km')
            distance_text = self._get_value(restaurant, 'distance_text')

            if distance_text:
                # If we have pre-formatted text, use enhanced format
                return f"📏 {self._format_distance_km_enhanced(distance_km) if distance_km is not None else distance_text}\n"
            elif distance_km is not None:
                return f"📏 {self._format_distance_km_enhanced(distance_km)}\n"
            else:
                return ""

        except Exception:
            return ""

    def _format_distance_km_enhanced(self, distance_km: float) -> str:
        """
        UPDATED: Format distance in "0.1 km" format with space after number (not "0.1km away")
        """
        try:
            if distance_km < 0.1:
                return f"{int(distance_km * 1000)} m"  # "82 m" not "82m away"
            else:
                return f"{distance_km:.1f} km"  # "1.2 km" not "1.2km away"
        except Exception:
            return "Distance unknown"

    def _format_description_enhanced(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        UPDATED: Format restaurant description with MORE content from database (not truncated at 150 chars)
        """
        try:
            # Check multiple possible description fields
            description = (
                self._get_value(restaurant, 'description', '').strip() or 
                self._get_value(restaurant, 'raw_description', '').strip() or
                self._get_value(restaurant, 'full_description', '').strip()
            )

            if not description:
                return ""

            # Clean description - NO TRUNCATION, show full AI-generated description
            clean_description = self._clean_html(description)

            return f"💭 {clean_description}\n"

        except Exception as e:
            logger.debug(f"Error formatting description: {e}")
            return ""

    def _format_sources_domains_only(self, restaurant: Union[Dict[str, Any], Any]) -> str:
        """
        UPDATED: Format restaurant sources showing ONLY domain names (not full URLs)
        ENHANCED DEBUG VERSION
        """
        try:
            restaurant_name = self._get_value(restaurant, 'name', 'Unknown')

            # DEBUG: Log the restaurant object type and structure
            logger.info(f"🔍 DEBUG - Formatting sources for restaurant: {restaurant_name}")
            logger.info(f"🔍 DEBUG - Restaurant object type: {type(restaurant)}")

            # Try to get sources from multiple possible fields
            sources = self._get_value(restaurant, 'sources', [])
            logger.info(f"🔍 DEBUG - Sources field: {sources} (type: {type(sources)})")

            if not sources:
                # Also check for alternative field names
                media_sources = self._get_value(restaurant, 'media_sources', [])
                sources_domains = self._get_value(restaurant, 'sources_domains', [])
                logger.info(f"🔍 DEBUG - media_sources field: {media_sources}")
                logger.info(f"🔍 DEBUG - sources_domains field: {sources_domains}")

                sources = media_sources or sources_domains

            if not sources:
                logger.info(f"🔍 DEBUG - No sources found for {restaurant_name}")

                # DEBUG: Let's see what fields ARE available
                if hasattr(restaurant, '__dict__'):
                    logger.info(f"🔍 DEBUG - Available fields in restaurant object: {restaurant.__dict__.keys()}")
                elif isinstance(restaurant, dict):
                    logger.info(f"🔍 DEBUG - Available keys in restaurant dict: {restaurant.keys()}")

                return ""

            logger.info(f"🔍 DEBUG - Found sources for {restaurant_name}: {sources}")

            # Extract domains from sources
            domains = self._extract_domains_from_sources(sources)
            logger.info(f"🔍 DEBUG - Extracted domains: {domains}")

            if domains:
                # Show max 3 domains as clickable links to main page
                domain_links = []
                for domain in domains[:3]:
                    # Create clickable link to the main page of the guide
                    clean_domain = self._clean_html(domain)
                    domain_url = f"https://{domain}"
                    domain_links.append(f'<a href="{escape(domain_url, quote=True)}">{clean_domain}</a>')

                domains_text = ", ".join(domain_links)
                if len(domains) > 3:
                    domains_text += f" +{len(domains)-3} more"
                result = f"🔷 Recommended by {domains_text}\n"
                logger.info(f"🔍 DEBUG - Final sources line: {result}")
                return result

            logger.info(f"🔍 DEBUG - No valid domains extracted for {restaurant_name}")
            return ""

        except Exception as e:
            logger.error(f"❌ DEBUG - Error formatting sources for {restaurant_name}: {e}")
            return ""

    def _extract_domains_from_sources(self, sources: List[str]) -> List[str]:
        """
        UPDATED: Extract clean domain names from URLs or source names
        """
        try:
            domains = []
            seen_domains = set()

            for source in sources:
                if not source or not str(source).strip():
                    continue

                source_str = str(source).strip()
                domain = self._extract_domain_from_url(source_str)

                # Clean and deduplicate
                if domain and domain.lower() not in seen_domains:
                    domains.append(domain)
                    seen_domains.add(domain.lower())

            return domains

        except Exception as e:
            logger.debug(f"Error extracting domains: {e}")
            return []

    def _extract_domain_from_url(self, source: str) -> str:
        """
        Extract domain from URL, or return source as-is if not a URL
        """
        try:
            # Check if it looks like a URL
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
            return f'📍 <a href="{escape(google_url, quote=True)}">{self._clean_html(clean_address)}</a>\n'

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

    def _clean_html(self, text: str) -> str:
        """
        Clean HTML entities and tags
        """
        if not text:
            return ""

        text = str(text).strip()
        text = escape(text, quote=False)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

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