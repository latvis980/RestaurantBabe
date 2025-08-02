# formatters/telegram_formatter.py
"""
Complete Telegram HTML formatter with CORRECT Google Maps URLs and street-only addresses
FIXED VERSION - uses official Google Maps URL format from documentation
"""
import re
import logging
from html import escape, unescape
import urllib.parse

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

        # Get Google Maps data for proper links
        place_id = restaurant.get('place_id')
        address_components = restaurant.get('address_components', [])

        # Skip if no valid name
        if not name or name == 'Unknown Restaurant':
            return ""

        # Clean and escape text
        name_escaped = self._clean_text(name)
        desc_escaped = self._clean_text(description)

        parts = [
            f"<b>{index}. {name_escaped}</b>\n",
            self._format_address_with_official_google_link(address, place_id, address_components),
            f"{desc_escaped}\n" if desc_escaped else "",
            self._format_sources(sources),
            "\n"  # Add spacing between restaurants
        ]

        return ''.join(filter(None, parts))

    def _format_address_with_official_google_link(self, full_address, place_id=None, address_components=None):
        """
        Format address using OFFICIAL Google Maps URL format with AI-cleaned street address

        URL Format: https://www.google.com/maps/search/?api=1&query=FALLBACK&query_place_id=PLACE_ID
        Display Text: AI-cleaned street address (no postal code, city, country)
        """
        if not full_address or full_address == "Address unavailable":
            return "📍 Address unavailable\n"

        # Use AI to extract clean street address from full address
        street_address = self._ai_extract_street_address(full_address)

        if not street_address:
            return "📍 Address unavailable\n"

        # Clean the street address for display
        clean_street = self._clean_text(street_address)

        # Create the OFFICIAL Google Maps URL format
        if place_id and place_id.strip():
            # Official format from Google documentation
            fallback_query = urllib.parse.quote(clean_street)
            maps_url = f"https://www.google.com/maps/search/?api=1&query={fallback_query}&query_place_id={place_id.strip()}"
            logger.debug(f"Created official Google Maps URL: {maps_url}")
        else:
            # Fallback: simple search URL
            encoded_street = urllib.parse.quote(clean_street)
            maps_url = f"https://www.google.com/maps/search/?api=1&query={encoded_street}"
            logger.debug(f"Created fallback search URL: {maps_url}")

        # Return properly formatted HTML link with street-only text
        return f'📍 <a href="{maps_url}">{clean_street}</a>\n'

    def _ai_extract_street_address(self, full_address):
        """
        Use simple AI logic to extract street address from full formatted address
        Removes postal code, city, country - keeps only street number and name

        Examples:
        - "123 Main St, New York, NY 10001, USA" -> "123 Main St"
        - "Via dei Tribunali, 94, 80139 Naples, Italy" -> "Via dei Tribunali, 94"
        - "Sheikh Mohammed bin Rashid Blvd - Downtown Dubai - Dubai - UAE" -> "Sheikh Mohammed bin Rashid Blvd"
        """
        if not full_address:
            return ""

        # Clean input
        address = str(full_address).strip()

        # Split by commas
        parts = [part.strip() for part in address.split(',')]

        if len(parts) == 1:
            # No commas - might be single street or foreign format
            # Check if it contains postal code at the end
            single_part = parts[0]
            # Remove postal codes (numbers at the end)
            cleaned = re.sub(r'\s+\d{4,6}\s*

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
        """
        Ensure HTML is completely safe for Telegram API - FIXED VERSION
        """
        if not text:
            return ""

        # First, protect valid HTML tags by temporarily replacing them
        valid_tag_pattern = r'<(/?)(?:b|i|u|s|a|code|pre)(?:\s[^>]*)?>'
        valid_tags = []
        placeholder_base = "___VALID_TAG_"

        def replace_valid_tag(match):
            tag_index = len(valid_tags)
            valid_tags.append(match.group(0))
            return f"{placeholder_base}{tag_index}___"

        # Temporarily replace valid tags with placeholders
        protected_text = re.sub(
            valid_tag_pattern,
            replace_valid_tag,
            text,
            flags=re.IGNORECASE
        )

        # Now escape any remaining < and > characters (these are invalid)
        protected_text = protected_text.replace('<', '&lt;').replace('>', '&gt;')

        # Restore the valid tags
        for i, original_tag in enumerate(valid_tags):
            placeholder = f"{placeholder_base}{i}___"
            protected_text = protected_text.replace(placeholder, original_tag)

        # Additional cleanup for safety
        result_text = self._additional_html_cleanup(protected_text)

        return result_text

    def _additional_html_cleanup(self, text):
        """Additional HTML cleanup without killing valid formatting"""

        # Remove truly empty tag pairs, e.g. <b></b> or <i   ></i>
        text = re.sub(
            r'<(?P<tag>b|i|u|s|code|pre)\b[^>]*>\s*</(?P=tag)>',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove suspiciously long single tags
        text = re.sub(r'<[^>]{100,}>', '', text)

        # Ensure proper tag nesting
        return self._fix_tag_nesting(text)

    def _fix_tag_nesting(self, text):
        """Fix HTML tag nesting using simple string parsing"""
        allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
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

        return ''.join(result), '', single_part)
            # Remove country names at the end
            cleaned = re.sub(r'\s+(?:USA|US|UK|UAE|Italy|France|Spain|Germany|Portugal|Netherlands|Belgium)

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
        """
        Ensure HTML is completely safe for Telegram API - FIXED VERSION
        """
        if not text:
            return ""

        # First, protect valid HTML tags by temporarily replacing them
        valid_tag_pattern = r'<(/?)(?:b|i|u|s|a|code|pre)(?:\s[^>]*)?>'
        valid_tags = []
        placeholder_base = "___VALID_TAG_"

        def replace_valid_tag(match):
            tag_index = len(valid_tags)
            valid_tags.append(match.group(0))
            return f"{placeholder_base}{tag_index}___"

        # Temporarily replace valid tags with placeholders
        protected_text = re.sub(
            valid_tag_pattern,
            replace_valid_tag,
            text,
            flags=re.IGNORECASE
        )

        # Now escape any remaining < and > characters (these are invalid)
        protected_text = protected_text.replace('<', '&lt;').replace('>', '&gt;')

        # Restore the valid tags
        for i, original_tag in enumerate(valid_tags):
            placeholder = f"{placeholder_base}{i}___"
            protected_text = protected_text.replace(placeholder, original_tag)

        # Additional cleanup for safety
        result_text = self._additional_html_cleanup(protected_text)

        return result_text

    def _additional_html_cleanup(self, text):
        """Additional HTML cleanup without killing valid formatting"""

        # Remove truly empty tag pairs, e.g. <b></b> or <i   ></i>
        text = re.sub(
            r'<(?P<tag>b|i|u|s|code|pre)\b[^>]*>\s*</(?P=tag)>',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove suspiciously long single tags
        text = re.sub(r'<[^>]{100,}>', '', text)

        # Ensure proper tag nesting
        return self._fix_tag_nesting(text)

    def _fix_tag_nesting(self, text):
        """Fix HTML tag nesting using simple string parsing"""
        allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
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

        return ''.join(result), '', cleaned, flags=re.IGNORECASE)
            return cleaned.strip()

        # Multiple parts - analyze to find street part
        street_candidates = []

        for i, part in enumerate(parts):
            part_lower = part.lower()

            # Skip parts that are clearly not street addresses
            if (re.search(r'^\d{4,6}

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
        """
        Ensure HTML is completely safe for Telegram API - FIXED VERSION
        """
        if not text:
            return ""

        # First, protect valid HTML tags by temporarily replacing them
        valid_tag_pattern = r'<(/?)(?:b|i|u|s|a|code|pre)(?:\s[^>]*)?>'
        valid_tags = []
        placeholder_base = "___VALID_TAG_"

        def replace_valid_tag(match):
            tag_index = len(valid_tags)
            valid_tags.append(match.group(0))
            return f"{placeholder_base}{tag_index}___"

        # Temporarily replace valid tags with placeholders
        protected_text = re.sub(
            valid_tag_pattern,
            replace_valid_tag,
            text,
            flags=re.IGNORECASE
        )

        # Now escape any remaining < and > characters (these are invalid)
        protected_text = protected_text.replace('<', '&lt;').replace('>', '&gt;')

        # Restore the valid tags
        for i, original_tag in enumerate(valid_tags):
            placeholder = f"{placeholder_base}{i}___"
            protected_text = protected_text.replace(placeholder, original_tag)

        # Additional cleanup for safety
        result_text = self._additional_html_cleanup(protected_text)

        return result_text

    def _additional_html_cleanup(self, text):
        """Additional HTML cleanup without killing valid formatting"""

        # Remove truly empty tag pairs, e.g. <b></b> or <i   ></i>
        text = re.sub(
            r'<(?P<tag>b|i|u|s|code|pre)\b[^>]*>\s*</(?P=tag)>',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove suspiciously long single tags
        text = re.sub(r'<[^>]{100,}>', '', text)

        # Ensure proper tag nesting
        return self._fix_tag_nesting(text)

    def _fix_tag_nesting(self, text):
        """Fix HTML tag nesting using simple string parsing"""
        allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
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

        return ''.join(result), part) or  # Pure postal codes
                part_lower in ['usa', 'us', 'uk', 'uae', 'italy', 'france', 'spain', 'germany', 'portugal', 'netherlands', 'belgium'] or
                re.search(r'^[A-Z]{2}

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
        """
        Ensure HTML is completely safe for Telegram API - FIXED VERSION
        """
        if not text:
            return ""

        # First, protect valid HTML tags by temporarily replacing them
        valid_tag_pattern = r'<(/?)(?:b|i|u|s|a|code|pre)(?:\s[^>]*)?>'
        valid_tags = []
        placeholder_base = "___VALID_TAG_"

        def replace_valid_tag(match):
            tag_index = len(valid_tags)
            valid_tags.append(match.group(0))
            return f"{placeholder_base}{tag_index}___"

        # Temporarily replace valid tags with placeholders
        protected_text = re.sub(
            valid_tag_pattern,
            replace_valid_tag,
            text,
            flags=re.IGNORECASE
        )

        # Now escape any remaining < and > characters (these are invalid)
        protected_text = protected_text.replace('<', '&lt;').replace('>', '&gt;')

        # Restore the valid tags
        for i, original_tag in enumerate(valid_tags):
            placeholder = f"{placeholder_base}{i}___"
            protected_text = protected_text.replace(placeholder, original_tag)

        # Additional cleanup for safety
        result_text = self._additional_html_cleanup(protected_text)

        return result_text

    def _additional_html_cleanup(self, text):
        """Additional HTML cleanup without killing valid formatting"""

        # Remove truly empty tag pairs, e.g. <b></b> or <i   ></i>
        text = re.sub(
            r'<(?P<tag>b|i|u|s|code|pre)\b[^>]*>\s*</(?P=tag)>',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove suspiciously long single tags
        text = re.sub(r'<[^>]{100,}>', '', text)

        # Ensure proper tag nesting
        return self._fix_tag_nesting(text)

    def _fix_tag_nesting(self, text):
        """Fix HTML tag nesting using simple string parsing"""
        allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
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

        return ''.join(result), part) or  # State codes like NY, CA
                part_lower in ['downtown', 'city center', 'centre', 'centro']):
                continue

            # Look for street indicators
            if (re.search(r'\d', part) and  # Contains numbers
                any(word in part_lower for word in ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'plaza', 'square', 'lane', 'ln', 'drive', 'dr', 'boulevard', 'blvd', 'way', 'via', 'rue', 'strada', 'calle'])):
                street_candidates.append(part)
            elif i == 0 and re.search(r'\d', part):
                # First part with numbers is likely street
                street_candidates.append(part)

        # Return the best street candidate
        if street_candidates:
            return street_candidates[0].strip()

        # Fallback: return first part if it looks like an address
        first_part = parts[0]
        if re.search(r'\d', first_part) or len(first_part) > 10:
            return first_part.strip()

        # Last resort: return first two parts joined
        if len(parts) >= 2:
            return f"{parts[0]}, {parts[1]}".strip()

        return parts[0].strip() if parts else ""

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
        """
        Ensure HTML is completely safe for Telegram API - FIXED VERSION
        """
        if not text:
            return ""

        # First, protect valid HTML tags by temporarily replacing them
        valid_tag_pattern = r'<(/?)(?:b|i|u|s|a|code|pre)(?:\s[^>]*)?>'
        valid_tags = []
        placeholder_base = "___VALID_TAG_"

        def replace_valid_tag(match):
            tag_index = len(valid_tags)
            valid_tags.append(match.group(0))
            return f"{placeholder_base}{tag_index}___"

        # Temporarily replace valid tags with placeholders
        protected_text = re.sub(
            valid_tag_pattern,
            replace_valid_tag,
            text,
            flags=re.IGNORECASE
        )

        # Now escape any remaining < and > characters (these are invalid)
        protected_text = protected_text.replace('<', '&lt;').replace('>', '&gt;')

        # Restore the valid tags
        for i, original_tag in enumerate(valid_tags):
            placeholder = f"{placeholder_base}{i}___"
            protected_text = protected_text.replace(placeholder, original_tag)

        # Additional cleanup for safety
        result_text = self._additional_html_cleanup(protected_text)

        return result_text

    def _additional_html_cleanup(self, text):
        """Additional HTML cleanup without killing valid formatting"""

        # Remove truly empty tag pairs, e.g. <b></b> or <i   ></i>
        text = re.sub(
            r'<(?P<tag>b|i|u|s|code|pre)\b[^>]*>\s*</(?P=tag)>',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove suspiciously long single tags
        text = re.sub(r'<[^>]{100,}>', '', text)

        # Ensure proper tag nesting
        return self._fix_tag_nesting(text)

    def _fix_tag_nesting(self, text):
        """Fix HTML tag nesting using simple string parsing"""
        allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
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

        return ''.join(result)