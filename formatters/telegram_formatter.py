# formatters/telegram_formatter.py
"""
FIXED Telegram HTML formatter - Proper HTML handling for Telegram

CRITICAL FIX: 
- Fixed _clean_html() method that was double-escaping HTML
- Fixed Google Maps link formatting
- Proper handling of Telegram HTML tags

This was causing raw HTML to appear in the Telegram bot because the formatter
was escaping HTML tags then trying to remove them, which doesn't work.
"""
import re
import logging
from html import escape
from urllib.parse import urlparse, quote
from formatters.google_links import build_google_maps_url

logger = logging.getLogger(__name__)

class TelegramFormatter:
    """Fixed Telegram HTML formatter with proper HTML handling"""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config=None):
        """Initialize formatter with config for AI features"""
        self.config = config

    def format_recommendations(self, recommendations_data):
        """Main entry point for formatting recommendations"""
        try:
            main_list = recommendations_data.get("main_list", [])
            logger.info(f"📋 Formatting {len(main_list)} restaurants for Telegram")

            if not main_list:
                return self._no_results_message()

            # Build message parts
            parts = ["<b>🍽️ Here's a selection for you:</b>\n\n"]

            for i, restaurant in enumerate(main_list, 1):
                restaurant_text = self._format_restaurant(restaurant, i)
                if restaurant_text:
                    parts.append(restaurant_text)

            parts.append("\n<i>Click the address to see the venue photos and menu on Google Maps</i>")

            # Join and apply length limit
            message = ''.join(parts)
            if len(message) > self.MAX_MESSAGE_LENGTH:
                message = message[:self.MAX_MESSAGE_LENGTH-3] + "…"

            logger.info("✅ Telegram formatting completed")
            return message

        except Exception as e:
            logger.error(f"❌ Error in Telegram formatting: {e}")
            return self._no_results_message()

    def _format_restaurant(self, restaurant, index):
        """Format single restaurant with all details"""
        try:
            name = restaurant.get("name", "").strip()
            address = restaurant.get("address", "").strip()
            description = restaurant.get("description", "").strip()
            sources = restaurant.get("sources", [])
            place_id = restaurant.get("place_id")

            if not name:
                logger.warning(f"Restaurant {index} missing name")
                return ""

            # Clean name (preserve Telegram HTML tags)
            name_clean = self._clean_html_preserve_tags(name)

            # Clean description (preserve Telegram HTML tags)
            desc_clean = self._clean_html_preserve_tags(description) if description else ""

            # Build restaurant text
            parts = [f"<b>{index}. {name_clean}</b>\n"]

            # Add address with Google Maps link
            if address:
                address_link = self._format_address_link(address, place_id, name)
                parts.append(address_link)

            # Add description
            if desc_clean:
                parts.append(f"{desc_clean}\n")

            # Add sources
            if sources:
                sources_text = self._format_sources(sources)
                parts.append(sources_text)

            parts.append("\n")  # Add spacing between restaurants

            return ''.join(parts)

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {index}: {e}")
            return f"<b>{index}. Restaurant information unavailable</b>\n\n"

    def _format_address_link(self, address, place_id, restaurant_name=""):
        """Create Google Maps link with proper formatting"""
        if not address:
            return ""

        clean_address = self._extract_street(address)

        # Use the build_google_maps_url function for consistent URL formatting
        if place_id:
            google_url = build_google_maps_url(place_id, restaurant_name)
        else:
            # Fallback to search-based URL
            encoded_query = quote(f"{restaurant_name} {address}")
            google_url = f"https://www.google.com/maps/search/?api=1&query={encoded_query}"

        return f'📍 <a href="{google_url}">{clean_address}</a>\n'

    def _extract_street(self, full_address):
        """Extract just the street address part"""
        if not full_address:
            return "Address available"

        # Split by comma and take first 1-2 parts
        parts = str(full_address).split(',')
        if len(parts) >= 2:
            return f"{parts[0].strip()}, {parts[1].strip()}"
        else:
            return parts[0].strip()

    def _format_sources(self, sources):
        """Format source list with domain names only"""
        if not sources:
            return ""

        try:
            # Extract domain names from URLs
            domain_names = []
            for source in sources:
                domain = self._extract_domain(source)
                if domain and domain not in domain_names:
                    domain_names.append(domain)

            if domain_names:
                # Limit to first 3 sources to keep message clean
                display_sources = domain_names[:3]
                sources_text = ", ".join(display_sources)
                return f"<i>✅ Recommended by: {sources_text}</i>\n"

        except Exception as e:
            logger.debug(f"Error formatting sources: {e}")

        return ""

    def _extract_domain(self, source):
        """Extract domain name from URL or return as-is if not a URL"""
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

    def _clean_html_preserve_tags(self, text):
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

    def _clean_html(self, text):
        """Legacy method - redirects to the fixed version"""
        return self._clean_html_preserve_tags(text)

    def _no_results_message(self):
        """Simple no results message"""
        return ("<b>Sorry, no restaurant recommendations found for your search.</b>\n\n"
                "Try rephrasing your query or searching for a different area.")