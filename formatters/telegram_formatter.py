# formatters/telegram_formatter.py
"""
SIMPLE Telegram HTML formatter - production approach
Based on Telegram bot best practices: keep it simple and reliable

FIXED: Domain extraction from full URLs in sources
"""
import re
import logging
from html import escape
from urllib.parse import urlparse
from formatters.google_links import build_google_maps_url

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

    def _format_address_link(self, address, place_id, restaurant_name=""):
        """Create Google Maps link with 2025 universal URL format"""
        if not address:
            return ""

        clean_address = self._extract_street(address)

        google_url = build_google_maps_url(place_id, restaurant_name)
        
        return f'📍 <a href="{google_url}">{clean_address}</a>\n'

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

            # Clean name
            name_clean = self._clean_html(name)

            # Clean description
            desc_clean = self._clean_html(description) if description else ""

            # Build restaurant text
            parts = [f"<b>{index}. {name_clean}</b>\n"]

            # Add address with link
            address_line = self._format_address_link(address, place_id)
            if address_line:
                parts.append(address_line)

            # Add description
            if desc_clean:
                parts.append(f"{desc_clean}\n")

            # Add sources - FIXED: Extract domains from URLs
            sources_line = self._format_sources(sources)
            if sources_line:
                parts.append(sources_line)
                # ADD THIS DEBUG LOGGING:
                logger.debug(f"🔍 TELEGRAM FORMATTER - Generated sources line: {sources_line}")
            else:
                logger.debug(f"⚠️ No sources line generated for {name}")

            parts.append("\n")  # Spacing between restaurants

            formatted_result = ''.join(parts)

            # ADD THIS DEBUG LOGGING:
            logger.debug(f"🔍 TELEGRAM FORMATTER - Final formatted restaurant:")
            logger.debug(f"🔍 TELEGRAM FORMATTER - Result: {formatted_result}")

            return formatted_result

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {index}: {e}")
            return ""

    def _extract_street(self, full_address):
        """Extract street address using AI - handles international formats"""
        if not full_address:
            return "Address available"

        try:
            # Use AI for smart address cleaning
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
        """Use AI to intelligently remove postal codes and countries"""
        try:
            from openai import OpenAI

            # Check if we have config and API key
            if not self.config or not getattr(self.config, 'DEEPSEEK_API_KEY', None):
                logger.warning("No API key found for address cleaning, using full address")
                return self._clean_html(full_address)  # Return full address if no API key

            # Initialize client
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
                model="deepseek-chat",  # CHANGED: Use GPT-4o Mini instead of deepseek-chat
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
            logger.error(f"AI API error: {e}, using full address")
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
        """Format sources - FIXED: Extract domains from URLs and deduplicate"""
        if not sources or not isinstance(sources, list):
            return ""

        # Extract domains from URLs and clean them
        clean_domains = []
        seen_domains = set()  # For deduplication

        for source in sources[:3]:  # Max 3 sources
            if source and str(source).strip():
                # Extract domain from URL if it's a URL, otherwise use as-is
                domain = self._extract_domain_from_url(str(source).strip())

                # Clean the domain/source
                clean_domain = self._clean_html(domain)

                # Add to list if not already seen (deduplicate)
                if clean_domain and clean_domain.lower() not in seen_domains:
                    clean_domains.append(clean_domain)
                    seen_domains.add(clean_domain.lower())

        if clean_domains:
            sources_text = ", ".join(clean_domains)
            return f"<i>✅ Recommended by: {sources_text}</i>\n"

        return ""

    def _extract_domain_from_url(self, source):
        """Extract domain from URL, or return source as-is if not a URL"""
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