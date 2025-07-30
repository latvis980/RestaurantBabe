# formatters/telegram_formatter.py
"""
Complete Telegram HTML formatter - handles all Telegram-specific formatting
"""
import re
import logging
from html import escape, unescape

logger = logging.getLogger(__name__)

class TelegramFormatter:
    """Handles all Telegram HTML formatting for restaurant recommendations"""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self):
        self.allowed_tags = {'b', 'i', 'u', 's', 'a', 'code', 'pre'}

    def format_recommendations(self, recommendations_data):
        """Main entry point for formatting recommendations"""
        try:
            logger.info("🔧 Starting Telegram formatting")

            main_list = recommendations_data.get("main_list", [])
            logger.info(f"📋 Found {len(main_list)} restaurants to format")

            if not main_list:
                logger.warning("❌ No restaurants found to format")
                return self.format_no_results()

            formatted_html = self._format_restaurant_list(main_list)
            logger.info("✅ Telegram formatting completed successfully")

            return formatted_html

        except Exception as e:
            logger.error(f"❌ Error in Telegram formatting: {e}")
            return self.format_no_results()

    def _format_restaurant_list(self, restaurants):
        """Format the main restaurant list"""
        html_parts = ["<b>🍽️ Recommended Restaurants</b>\n\n"]

        formatted_count = 0
        for i, restaurant in enumerate(restaurants, 1):
            try:
                formatted_restaurant = self._format_single_restaurant(restaurant, i)
                if formatted_restaurant:  # Only add if formatting succeeded
                    html_parts.append(formatted_restaurant)
                    formatted_count += 1
            except Exception as e:
                logger.error(f"❌ Error formatting restaurant {i}: {e}")
                continue  # Skip this restaurant and continue with others

        if formatted_count == 0:
            return self.format_no_results()

        html_parts.append(self._format_footer())

        final_html = ''.join(html_parts)

        logger.info(f"✅ Successfully formatted {formatted_count} restaurants")
        return self._finalize_html(final_html)

    def _format_single_restaurant(self, restaurant, index):
        """Format a single restaurant entry"""
        # Safely extract data with defaults
        name = str(restaurant.get('name', 'Unknown Restaurant')).strip()
        description = str(restaurant.get('description', 'No description available')).strip()
        address = restaurant.get('address', 'Address unavailable')
        sources = restaurant.get('sources', [])

        # Skip if no valid name
        if not name or name == 'Unknown Restaurant':
            return ""

        # Clean and escape text
        name_escaped = self._clean_text(name)
        desc_escaped = self._clean_text(description)

        parts = [
            f"<b>{index}. {name_escaped}</b>\n",
            self._format_address(address),
            f"{desc_escaped}\n" if desc_escaped else "",
            self._format_sources(sources),
            "\n"  # Add spacing between restaurants
        ]

        return ''.join(filter(None, parts))

    def _format_address(self, address):
        """Format address with link support"""
        if not address or address == "Address unavailable":
            return "📍 Address unavailable\n"

        # Check for existing link format
        link_match = re.search(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', str(address))
        if link_match:
            url, address_text = link_match.groups()
            return f'📍 <a href="{url}">{self._clean_text(address_text)}</a>\n'

        return f"📍 {self._clean_text(str(address))}\n"

    def _format_sources(self, sources):
        """Format source attribution"""
        if not sources or not isinstance(sources, list):
            return ""

        valid_sources = []
        for source in sources:
            if source and str(source).strip():
                cleaned_source = self._clean_text(str(source).strip())
                if cleaned_source:
                    valid_sources.append(cleaned_source)

        if valid_sources:
            sources_text = ", ".join(valid_sources[:3])  # Limit to 3 sources
            return f"<i>✅ Sources: {sources_text}</i>\n"

        return ""

    def _format_footer(self):
        """Standard footer for recommendations"""
        return "<i>Recommendations compiled from reputable restaurant guides and critics.</i>"

    def format_no_results(self):
        """Message when no restaurants found"""
        return ("<b>Sorry, no restaurant recommendations found for your search.</b>\n\n"
                "Try rephrasing your query or searching for a different area.")

    def _clean_text(self, text):
        """Clean and prepare text for HTML formatting - ROBUST VERSION"""
        if not text:
            return ""

        text = str(text).strip()

        # Remove any existing HTML tags first to avoid conflicts
        text = re.sub(r'<[^>]*>', '', text)

        # Decode any existing HTML entities to get clean text
        text = unescape(text)

        # Now escape the essential characters for Telegram HTML
        # Do this in a specific order to avoid double-escaping

        # 1. Escape ampersands first (but not existing entities)
        text = re.sub(r'&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)', '&amp;', text)

        # 2. Escape < and >
        text = text.replace('<', '&lt;').replace('>', '&gt;')

        # 3. Remove or replace problematic characters that could break HTML
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Remove null bytes
        text = text.replace('\x00', '')

        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _finalize_html(self, html):
        """Apply final processing and length limits"""
        # Sanitize HTML for Telegram
        html = self._sanitize_for_telegram(html)

        # Apply length limit
        if len(html) > self.MAX_MESSAGE_LENGTH:
            html = html[:self.MAX_MESSAGE_LENGTH-3] + "…"
            logger.info(f"📏 Truncated message to {len(html)} characters")

        return html

    def _sanitize_for_telegram(self, text):
        """Ensure HTML is completely safe for Telegram API - ROBUST VERSION"""
        if not text:
            return ""

        # Remove any malformed or incomplete tags
        # This regex finds unclosed < or unmatched >
        text = re.sub(r'<(?![/]?(?:b|i|u|s|a|code|pre)(?:\s[^>]*)?>)', '&lt;', text)
        text = re.sub(r'(?<!<[/]?(?:b|i|u|s|a|code|pre)(?:\s[^>]*))>', '&gt;', text)

        # Remove any standalone closing tags without openers
        allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']

        # Track open tags
        open_tags = []
        result = []
        i = 0

        while i < len(text):
            if text[i] == '<':
                # Find the end of the tag
                end = text.find('>', i)
                if end == -1:
                    # No closing bracket, treat as escaped text
                    result.append('&lt;')
                    i += 1
                    continue

                tag_content = text[i+1:end].strip()

                # Check if it's a closing tag
                if tag_content.startswith('/'):
                    tag_name = tag_content[1:].strip().lower()
                    if tag_name in allowed_tags and open_tags and open_tags[-1] == tag_name:
                        # Valid closing tag
                        open_tags.pop()
                        result.append(text[i:end+1])
                    else:
                        # Invalid closing tag, escape it
                        result.append('&lt;' + tag_content + '&gt;')
                else:
                    # Opening tag
                    tag_parts = tag_content.split(None, 1)
                    tag_name = tag_parts[0].lower()

                    if tag_name in allowed_tags:
                        # Validate tag format
                        if tag_name == 'a' and len(tag_parts) > 1:
                            # Special validation for links
                            if 'href=' in tag_parts[1] and '"' in tag_parts[1]:
                                result.append(text[i:end+1])
                                open_tags.append(tag_name)
                            else:
                                # Invalid link tag
                                result.append('&lt;' + tag_content + '&gt;')
                        else:
                            # Valid formatting tag
                            result.append(text[i:end+1])
                            open_tags.append(tag_name)
                    else:
                        # Unknown tag, escape it
                        result.append('&lt;' + tag_content + '&gt;')

                i = end + 1
            else:
                result.append(text[i])
                i += 1

        # Close any remaining open tags
        while open_tags:
            tag = open_tags.pop()
            result.append(f'</{tag}>')

        final_text = ''.join(result)

        # Final safety check - remove any remaining problematic patterns
        final_text = re.sub(r'<\s*>', '', final_text)  # Remove empty tags
        final_text = re.sub(r'<[^>]{100,}>', '', final_text)  # Remove suspiciously long tags

        return final_text