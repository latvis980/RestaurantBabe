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
        """Clean and prepare text for HTML formatting - FIXED VERSION"""
        if not text:
            return ""

        text = str(text).strip()

        # First, decode any existing HTML entities to get clean text
        text = unescape(text)

        # Now escape only the characters that need escaping for Telegram HTML
        # Use simple replacements instead of complex lookbehind patterns

        # Replace & that aren't part of valid entities
        text = re.sub(r'&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)', '&amp;', text)

        # For < and >, use a different approach without variable-width lookbehind
        # Instead, we'll use a two-step process:

        # Step 1: Temporarily mark valid HTML tags
        valid_tag_pattern = r'</?(?:b|i|u|s|code|pre|a(?:\s[^>]*)?)\s*>'
        protected_tags = []

        def protect_tag(match):
            tag = match.group(0)
            placeholder = f"__PROTECTED_TAG_{len(protected_tags)}__"
            protected_tags.append(tag)
            return placeholder

        # Protect valid tags
        text = re.sub(valid_tag_pattern, protect_tag, text, flags=re.IGNORECASE)

        # Step 2: Now escape all remaining < and >
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')

        # Step 3: Restore the protected tags
        for i, tag in enumerate(protected_tags):
            placeholder = f"__PROTECTED_TAG_{i}__"
            text = text.replace(placeholder, tag)

        return text

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
        """Ensure HTML is safe for Telegram API"""
        if not text:
            return ""

        # First, escape any unescaped text content
        # Do this for any text between > and <
        def escape_text_content(match):
            content = match.group(1)
            # Only escape if it's not already escaped
            if '&' in content and ('&lt;' in content or '&gt;' in content or '&amp;' in content):
                return '>' + content + '<'
            return '>' + escape(content) + '<'

        # Fix unescaped content between tags
        text = re.sub(r'>([^<]+)<', escape_text_content, text)

        # Ensure all tags are properly closed
        # Stack to track open tags
        stack = []
        result = []
        i = 0

        while i < len(text):
            if text[i] == '<':
                # Find the end of the tag
                end = text.find('>', i)
                if end == -1:
                    # No closing bracket, treat as plain text
                    result.append('&lt;')
                    i += 1
                    continue

                tag_content = text[i+1:end]
                if tag_content.startswith('/'):
                    # Closing tag
                    tag_name = tag_content[1:].split()[0].lower()
                    if tag_name in self.allowed_tags:
                        if stack and stack[-1] == tag_name:
                            stack.pop()
                            result.append(text[i:end+1])
                        else:
                            # Mismatched closing tag, just add as text
                            result.append(escape(text[i:end+1]))
                    else:
                        # Not an allowed tag
                        result.append(escape(text[i:end+1]))
                else:
                    # Opening tag
                    tag_parts = tag_content.split(None, 1)
                    tag_name = tag_parts[0].lower()
                    if tag_name in self.allowed_tags:
                        if tag_name == 'a':
                            # Special handling for links
                            if len(tag_parts) > 1 and 'href=' in tag_parts[1]:
                                result.append(text[i:end+1])
                                stack.append(tag_name)
                            else:
                                result.append(escape(text[i:end+1]))
                        else:
                            result.append(text[i:end+1])
                            stack.append(tag_name)
                    else:
                        # Not an allowed tag
                        result.append(escape(text[i:end+1]))
                i = end + 1
            else:
                result.append(text[i])
                i += 1

        # Close any unclosed tags
        while stack:
            tag = stack.pop()
            result.append(f'</{tag}>')

        return ''.join(result)