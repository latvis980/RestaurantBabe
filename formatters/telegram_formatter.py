# formatters/telegram_formatter.py
"""
SIMPLE Telegram HTML formatter - production approach
Based on Telegram bot best practices: keep it simple and reliable
"""
import re
import logging
from html import escape
import urllib.parse

logger = logging.getLogger(__name__)

class TelegramFormatter:
    """Simple, reliable Telegram HTML formatter following production best practices"""

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
            parts = ["<b>🍽️ Recommended Restaurants</b>\n\n"]

            for i, restaurant in enumerate(main_list, 1):
                restaurant_text = self._format_restaurant(restaurant, i)
                if restaurant_text:
                    parts.append(restaurant_text)

            parts.append("\n<i>Recommendations compiled from reputable restaurant guides and critics.</i>")

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
        """Format a single restaurant - simple and reliable"""
        name = restaurant.get('name', '').strip()
        if not name:
            return ""

        description = restaurant.get('description', '').strip()
        address = restaurant.get('address', '')
        sources = restaurant.get('sources', [])
        place_id = restaurant.get('place_id')

        # Clean text for HTML (simple escaping)
        name_clean = self._clean_html(name)
        desc_clean = self._clean_html(description)

        # Build restaurant entry
        parts = [f"<b>{index}. {name_clean}</b>\n"]

        # Add address with link
        address_line = self._format_address_link(address, place_id)
        if address_line:
            parts.append(address_line)

        # Add description
        if desc_clean:
            parts.append(f"{desc_clean}\n")

        # Add sources
        sources_line = self._format_sources(sources)
        if sources_line:
            parts.append(sources_line)

        parts.append("\n")  # Spacing between restaurants

        return ''.join(parts)

    def _format_address_link(self, address, place_id):
        """Create Google Maps link - use Google's official URL when available"""
        if not address or address == "Address unavailable":
            return "📍 Address unavailable\n"

        # Extract street address for display
        street_address = self._extract_street(address)
        clean_street = self._clean_html(street_address)

        # PRIORITY 1: Try to get the google_maps_url from restaurant data
        # This should contain the official Google URL from Places API
        google_url = None
        if hasattr(self, '_current_restaurant') and self._current_restaurant:
            google_url = (self._current_restaurant.get('google_maps_url') or 
                         self._current_restaurant.get('google_url') or
                         self._current_restaurant.get('url'))

            if google_url:
                logger.debug(f"✅ Using official Google URL: {google_url}")
            else:
                logger.debug(f"⚠️ No Google URL found in restaurant data. Available keys: {list(self._current_restaurant.keys())}")

        # PRIORITY 2: Create URL using place_id if we have it
        if not google_url and place_id:
            # Use the precision query method (most reliable for business listings)
            encoded_address = urllib.parse.quote(f"{clean_street}")
            google_url = f"https://www.google.com/maps/search/?api=1&query={encoded_address}&query_place_id={place_id}"
            logger.debug(f"🔧 Created place_id URL: {google_url}")

        # PRIORITY 3: Fallback to simple address search
        if not google_url:
            encoded_address = urllib.parse.quote(clean_street)
            google_url = f"https://www.google.com/maps/search/?api=1&query={encoded_address}"
            logger.debug(f"🔧 Created fallback URL: {google_url}")

        return f'📍 <a href="{google_url}">{clean_street}</a>\n'

    def _format_restaurant(self, restaurant, index):
        """Format a single restaurant - simple and reliable"""
        name = restaurant.get('name', '').strip()
        if not name:
            return ""

        description = restaurant.get('description', '').strip()
        address = restaurant.get('address', '')
        sources = restaurant.get('sources', [])
        place_id = restaurant.get('place_id')

        # Store current restaurant for URL access
        self._current_restaurant = restaurant

        # Clean text for HTML (simple escaping)
        name_clean = self._clean_html(name)
        desc_clean = self._clean_html(description)

        # Build restaurant entry
        parts = [f"<b>{index}. {name_clean}</b>\n"]

        # Add address with link
        address_line = self._format_address_link(address, place_id)
        if address_line:
            parts.append(address_line)

        # Add description
        if desc_clean:
            parts.append(f"{desc_clean}\n")

        # Add sources
        sources_line = self._format_sources(sources)
        if sources_line:
            parts.append(sources_line)

        parts.append("\n")  # Spacing between restaurants

        # Clear current restaurant
        self._current_restaurant = None

        return ''.join(parts)

    def _extract_street(self, full_address):
        """Extract street address using AI - handles international formats"""
        if not full_address:
            return "Address available"

        try:
            # Use DeepSeek for smart address cleaning
            cleaned_address = self._ai_clean_address(full_address)
            if cleaned_address:
                return cleaned_address
            else:
                logger.warning(f"AI cleaning failed, using full address: {full_address}")
                return self._clean_html(full_address)  # Use full address as fallback
        except Exception as e:
            logger.warning(f"AI address cleaning failed: {e}, using full address")
            return self._clean_html(full_address)  # Use full address as fallback

    def _ai_clean_address(self, full_address):
        """Use DeepSeek to intelligently remove postal codes and countries"""
        try:
            from openai import OpenAI

            # Check if we have config and API key
            if not self.config or not getattr(self.config, 'DEEPSEEK_API_KEY', None):
                logger.warning("No DeepSeek API key found, using full address")
                return self._clean_html(full_address)  # Return full address if no API key

            # Initialize DeepSeek client
            client = OpenAI(
                api_key=getattr(self.config, 'DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )

            prompt = f"""Clean this address by removing ONLY the postal code and country. Keep the street address and city.

Examples:
- Via dei Tribunali, 94, 80139 Naples, Italy → Via dei Tribunali, 94, Naples  
- 123 Main St, New York, NY 10001, USA → 123 Main St, New York, NY
- Sheikh Mohammed bin Rashid Blvd - Downtown Dubai - Dubai - UAE → Sheikh Mohammed bin Rashid Blvd - Downtown Dubai - Dubai
- Rua Augusta 123, 1100-048 Lisboa, Portugal → Rua Augusta 123, Lisboa
- 東京都渋谷区渋谷2-21-1, 150-8510 Japan → 東京都渋谷区渋谷2-21-1
- Av. Santa Fe 1860, C1123ABN CABA, Argentina → Av. Santa Fe 1860, CABA
- 上海市黄浦区南京东路300号, 200001 China → 上海市黄浦区南京东路300号
- ул. Арбат, 12, 119019 Москва, Россия → ул. Арбат, 12, Москва

Address to clean: "{full_address}"

Return only the cleaned address, nothing else."""

            response = client.chat.completions.create(
                model="deepseek-chat",  # Light model for speed
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1  # Low temperature for consistency
            )

            cleaned = response.choices[0].message.content.strip()

            # Basic validation - cleaned address should be shorter than original
            if cleaned and len(cleaned) < len(full_address) and len(cleaned) > 5:
                logger.debug(f"AI cleaned address: '{full_address}' → '{cleaned}'")
                return cleaned
            else:
                logger.warning(f"AI cleaning suspicious: '{full_address}' → '{cleaned}', using full address")
                return self._clean_html(full_address)  # Return full address if AI result is suspicious

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}, using full address")
            return self._clean_html(full_address)  # Return full address on any error

    def _simple_extract_street(self, full_address):
        """Fallback simple address extraction"""
        parts = str(full_address).split(',')
        street = parts[0].strip()

        # Keep street + city (first two parts)
        if len(parts) >= 2:
            return f"{parts[0].strip()}, {parts[1].strip()}"

        return street if street else "Address available"

    def _format_sources(self, sources):
        """Format sources - simple approach"""
        if not sources or not isinstance(sources, list):
            return ""

        # Clean and limit sources
        clean_sources = []
        for source in sources[:3]:  # Max 3 sources
            if source and str(source).strip():
                clean_source = self._clean_html(str(source).strip())
                if clean_source:
                    clean_sources.append(clean_source)

        if clean_sources:
            sources_text = ", ".join(clean_sources)
            return f"<i>✅ Sources: {sources_text}</i>\n"

        return ""

    def _clean_html(self, text):
        """Simple HTML cleaning - production approach"""
        if not text:
            return ""

        # Basic HTML escaping (only what's needed)
        text = str(text).strip()
        text = escape(text, quote=False)  # Escape HTML but keep quotes

        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _no_results_message(self):
        """Simple no results message"""
        return ("<b>Sorry, no restaurant recommendations found for your search.</b>\n\n"
                "Try rephrasing your query or searching for a different area.")