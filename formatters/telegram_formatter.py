# Suggested new file structure for better organization:

# 1. Create: formatters/telegram_formatter.py
"""
Dedicated Telegram HTML formatter - handles all Telegram-specific formatting
"""
import re
import html
from html import escape, unescape

class TelegramFormatter:
    """Handles all Telegram HTML formatting for restaurant recommendations"""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self):
        self.allowed_tags = {'b', 'i', 'u', 's', 'a', 'code', 'pre'}

    def format_recommendations(self, recommendations_data):
        """Main entry point for formatting recommendations"""
        main_list = recommendations_data.get("main_list", [])

        if not main_list:
            return self._format_no_results()

        return self._format_restaurant_list(main_list)

    def _format_restaurant_list(self, restaurants):
        """Format the main restaurant list"""
        html_parts = ["<b>🍽️ Recommended Restaurants</b>\n\n"]

        for i, restaurant in enumerate(restaurants, 1):
            html_parts.append(self._format_single_restaurant(restaurant, i))

        html_parts.append(self._format_footer())

        return self._finalize_html(''.join(html_parts))

    def _format_single_restaurant(self, restaurant, index):
        """Format a single restaurant entry"""
        name = self._clean_text(restaurant.get('name', 'Unknown'))
        description = self._clean_text(restaurant.get('description', ''))
        address = restaurant.get('address', 'Address unavailable')
        sources = restaurant.get('sources', [])

        parts = [
            f"<b>{index}. {escape(name)}</b>\n",
            self._format_address(address),
            f"{escape(description)}\n" if description else "",
            self._format_sources(sources),
            "\n"
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
            return f'📍 <a href="{url}">{escape(address_text)}</a>\n'

        return f"📍 {escape(str(address))}\n"

    def _format_sources(self, sources):
        """Format source attribution"""
        if not sources or not isinstance(sources, list):
            return ""

        valid_sources = []
        for source in sources:
            if source and str(source).strip():
                valid_sources.append(escape(str(source).strip()))

        if valid_sources:
            sources_text = ", ".join(valid_sources[:3])
            return f"<i>✅ Sources: {sources_text}</i>\n"

        return ""

    def _format_footer(self):
        """Standard footer for recommendations"""
        return "<i>Recommendations compiled from reputable restaurant guides and critics.</i>"

    def _format_no_results(self):
        """Message when no restaurants found"""
        return ("<b>Sorry, no restaurant recommendations found for your search.</b>\n\n"
                "Try rephrasing your query or searching for a different area.")

    def _clean_text(self, text):
        """Clean and prepare text for HTML formatting"""
        if not text:
            return ""

        text = str(text).strip()
        text = unescape(text)  # Decode existing entities

        # Selective escaping for Telegram
        text = re.sub(r'&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)', '&amp;', text)
        text = re.sub(r'<(?!/?(?:b|i|u|s|code|pre|a\s))', '&lt;', text)
        text = re.sub(r'(?<!(?:b|i|u|s|code|pre|a))>', '&gt;', text)

        return text

    def _finalize_html(self, html):
        """Apply final processing and length limits"""
        # Sanitize HTML for Telegram
        html = self._sanitize_for_telegram(html)

        # Apply length limit
        if len(html) > self.MAX_MESSAGE_LENGTH:
            html = html[:self.MAX_MESSAGE_LENGTH-3] + "…"

        return html

    def _sanitize_for_telegram(self, text):
        """Ensure HTML is safe for Telegram API"""
        # Implementation of your existing sanitize_html_for_telegram function
        # Move the existing function here
        pass


# 2. Create: formatters/base_formatter.py
"""
Base formatter interface for different output formats
"""
from abc import ABC, abstractmethod

class BaseFormatter(ABC):
    """Base class for all output formatters"""

    @abstractmethod
    def format_recommendations(self, recommendations_data):
        """Format recommendations for specific output format"""
        pass


# 3. Update: agents/langchain_orchestrator.py
"""
Simplified orchestrator that delegates formatting
"""
from formatters.telegram_formatter import TelegramFormatter

class LangChainOrchestrator:
    def __init__(self, config):
        # ... existing initialization ...

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter()

        # Simplified extract_html step
        def extract_html_step(x):
            """Extract HTML step - Now just delegates to formatter"""
            try:
                enhanced_recs = x.get("enhanced_recommendations", {})
                telegram_text = self.telegram_formatter.format_recommendations(enhanced_recs)

                return {
                    **x,
                    "telegram_formatted_text": telegram_text
                }
            except Exception as e:
                logger.error(f"Error in HTML formatting: {e}")
                return {
                    **x,
                    "telegram_formatted_text": self.telegram_formatter._format_no_results()
                }

        self.extract_html = RunnableLambda(extract_html_step, name="extract_html")


# 4. Future: Add other formatters
# formatters/json_formatter.py - for API responses
# formatters/markdown_formatter.py - for documentation
# formatters/email_formatter.py - for email recommendations