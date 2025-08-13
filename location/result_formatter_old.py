# location/result_formatter.py
"""
Enhanced Location Result Formatter

Formats location-based search results with full restaurant details:
- Descriptions with proper formatting
- Source attribution 
- Properly formatted addresses with Google Maps links
- Uses the same address formatting logic as followup_search_agent
"""

import logging
import re
import requests
from typing import Dict, List, Any, Optional
from html import escape

logger = logging.getLogger(__name__)

class LocationResultFormatter:
    """
    Formats location-based search results with comprehensive restaurant information
    """

    def __init__(self, config=None):
        """Initialize formatter with optional config"""
        self.config = config

    def format_location_results(
        self, 
        restaurants: List[Dict[str, Any]], 
        query: str,
        location_description: str,
        source: str = "personal notes"
    ) -> Dict[str, Any]:
        """
        Format location search results for immediate sending

        Args:
            restaurants: List of restaurant dictionaries from database
            query: Original user query
            location_description: Description of the location searched
            source: Source type (e.g., "personal notes", "database")

        Returns:
            Formatted results dictionary for telegram formatter
        """
        try:
            logger.info(f"📋 Formatting {len(restaurants)} location results")

            if not restaurants:
                return self._create_empty_response(query, location_description)

            # Format each restaurant with full details
            formatted_restaurants = []
            for restaurant in restaurants:
                formatted_restaurant = self._format_single_restaurant(restaurant)
                if formatted_restaurant:
                    formatted_restaurants.append(formatted_restaurant)

            # Create response in format expected by telegram formatter
            return {
                "main_list": formatted_restaurants,
                "search_info": {
                    "query": query,
                    "location": location_description,
                    "source": source,
                    "count": len(formatted_restaurants)
                },
                "source_type": source,
                "immediate_send": True
            }

        except Exception as e:
            logger.error(f"❌ Error formatting location results: {e}")
            return self._create_empty_response(query, location_description)

    def _format_single_restaurant(self, restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format a single restaurant with all available information

        Args:
            restaurant: Raw restaurant data from database

        Returns:
            Formatted restaurant dictionary
        """
        try:
            # Extract basic information
            name = restaurant.get('name', '').strip()
            if not name:
                logger.warning("Restaurant missing name, skipping")
                return None

            # Format description with source attribution
            description = self._format_description(restaurant)

            # Format address with proper Google Maps link
            address_info = self._format_address(restaurant)

            # Format sources
            sources = self._format_sources(restaurant)

            # Get cuisine tags
            cuisine_tags = restaurant.get('cuisine_tags', [])

            # Build formatted restaurant
            formatted = {
                "name": name,
                "description": description,
                "address": address_info.get('address_text', ''),
                "google_maps_url": address_info.get('google_maps_url', ''),
                "place_id": restaurant.get('place_id'),
                "sources": sources,
                "cuisine_tags": cuisine_tags,
                "latitude": restaurant.get('latitude'),
                "longitude": restaurant.get('longitude'),
                "rating": restaurant.get('rating'),
                "distance": restaurant.get('distance_km'),  # Will be added by orchestrator
                "relevance_score": restaurant.get('_relevance_score', 0.5),
                "raw_data": restaurant  # Keep original data for debugging
            }

            return formatted

        except Exception as e:
            logger.error(f"❌ Error formatting restaurant {restaurant.get('name', 'Unknown')}: {e}")
            return None

    def _format_description(self, restaurant: Dict[str, Any]) -> str:
        """
        Format restaurant description with source attribution

        Args:
            restaurant: Restaurant data

        Returns:
            Formatted description string
        """
        try:
            # Get main description
            raw_description = restaurant.get('raw_description', '').strip()

            if not raw_description:
                # Fallback to basic info if no description
                cuisine_tags = restaurant.get('cuisine_tags', [])
                if cuisine_tags:
                    cuisine_list = [tag.replace('-', ' ').title() for tag in cuisine_tags[:3]]
                    return f"A {', '.join(cuisine_list)} establishment from my personal notes."
                else:
                    return "Restaurant from my personal notes."

            # Clean up the description
            description = self._clean_description_text(raw_description)

            # Add mention count if available
            mention_count = restaurant.get('mention_count', 1)
            if mention_count > 1:
                description += f" (Mentioned {mention_count} times in my sources)"

            return description

        except Exception as e:
            logger.error(f"❌ Error formatting description: {e}")
            return "Restaurant from my personal notes."

    def _clean_description_text(self, text: str) -> str:
        """
        Clean description text for better readability

        Args:
            text: Raw description text

        Returns:
            Cleaned description text
        """
        try:
            if not text:
                return ""

            # Remove multiple source sections if present (from merged descriptions)
            # Keep only the primary description for location results
            if "=== PRIMARY SOURCE:" in text:
                lines = text.split('\n')
                primary_lines = []
                in_primary = False

                for line in lines:
                    if "=== PRIMARY SOURCE:" in line:
                        in_primary = True
                        continue
                    elif "=== ADDITIONAL SOURCE" in line:
                        break
                    elif in_primary and line.strip():
                        primary_lines.append(line)

                text = '\n'.join(primary_lines).strip()

            # Basic text cleaning
            text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
            text = re.sub(r'\n+', ' ', text)  # Multiple newlines to space
            text = text.strip()

            # Limit length for location results (keep it concise)
            if len(text) > 300:
                # Find the last complete sentence within 300 chars
                truncated = text[:300]
                last_period = truncated.rfind('.')
                if last_period > 200:  # Only truncate at sentence if reasonable
                    text = truncated[:last_period + 1]
                else:
                    text = truncated + "..."

            return text

        except Exception as e:
            logger.error(f"❌ Error cleaning description text: {e}")
            return text  # Return original on error

    def _format_address(self, restaurant: Dict[str, Any]) -> Dict[str, str]:
        """
        Format address with Google Maps link using followup_search_agent logic

        Args:
            restaurant: Restaurant data

        Returns:
            Dictionary with address_text and google_maps_url
        """
        try:
            # Get address from database
            full_address = restaurant.get('address', '').strip()

            if not full_address or full_address.lower() in ['unknown', 'not available', '']:
                return {
                    'address_text': 'Address available on Google Maps',
                    'google_maps_url': self._create_fallback_maps_url(restaurant)
                }

            # Extract street address (removing postal codes and country)
            street_address = self._extract_street_address(full_address)

            # Get Google Maps URL using the same logic as followup_search_agent
            google_maps_url = self._get_canonical_google_maps_url(restaurant)

            return {
                'address_text': street_address,
                'google_maps_url': google_maps_url
            }

        except Exception as e:
            logger.error(f"❌ Error formatting address: {e}")
            return {
                'address_text': 'Address available',
                'google_maps_url': self._create_fallback_maps_url(restaurant)
            }

    def _extract_street_address(self, full_address: str) -> str:
        """
        Extract street address part (remove postal codes and country)
        Using similar logic to telegram_formatter._extract_street

        Args:
            full_address: Full address from database

        Returns:
            Cleaned street address
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
                    last_part.upper() in ['USA', 'UK', 'UAE', 'FRANCE', 'ITALY', 'SPAIN', 'PORTUGAL', 'JAPAN']):
                    parts = parts[:-1]

            # Remove postal codes from the last remaining part
            if parts:
                last_part = parts[-1]
                # Remove postal code patterns
                last_part = re.sub(r'\s+\d{4,6}[-\s]*\d*', '', last_part)  # Most postal codes
                last_part = re.sub(r'\s+[A-Z]\d[A-Z]\s*\d[A-Z]\d', '', last_part)  # Canadian postal codes
                parts[-1] = last_part.strip()

            # Rejoin the cleaned parts
            street_address = ', '.join(part for part in parts if part)

            return street_address if street_address else "Address available"

        except Exception as e:
            logger.error(f"❌ Error extracting street address: {e}")
            return full_address

    def _get_canonical_google_maps_url(self, restaurant: Dict[str, Any]) -> str:
        """
        Get canonical Google Maps URL using the same logic as followup_search_agent
        Preference: cid URL > place_id URL > coordinate URL

        Args:
            restaurant: Restaurant data

        Returns:
            Google Maps URL
        """
        try:
            # Check for existing Google Maps URL in restaurant data
            google_url = (restaurant.get('google_maps_url') or 
                         restaurant.get('google_url') or 
                         restaurant.get('url') or '')

            # Try to get canonical CID URL
            if google_url:
                canonical_url = self._canonical_cid_url(google_url)
                if canonical_url:
                    return canonical_url

            # Fallback to place_id URL
            place_id = restaurant.get('place_id')
            if place_id:
                return f"https://www.google.com/maps/place/?q=place_id:{place_id}"

            # Final fallback to coordinates
            return self._create_fallback_maps_url(restaurant)

        except Exception as e:
            logger.error(f"❌ Error getting Google Maps URL: {e}")
            return self._create_fallback_maps_url(restaurant)

    def _canonical_cid_url(self, url: str) -> str:
        """
        Convert to canonical Google Maps CID URL
        Same logic as followup_search_agent._canonical_cid_url

        Args:
            url: Original Google Maps URL

        Returns:
            Canonical CID URL or empty string
        """
        try:
            if not url:
                return ""

            # Check if already in canonical form
            cid_match = re.search(r"[?&]cid=(\d+)", url)
            if cid_match:
                return f"https://maps.google.com/?cid={cid_match.group(1)}"

            # Handle short URLs by following redirects
            if "goo.gl/maps" in url or "maps.app.goo.gl" in url:
                try:
                    response = requests.head(url, allow_redirects=True, timeout=3)
                    cid_match = re.search(r"[?&]cid=(\d+)", response.url)
                    if cid_match:
                        return f"https://maps.google.com/?cid={cid_match.group(1)}"
                except requests.exceptions.RequestException:
                    pass

            return ""  # Couldn't convert to CID format

        except Exception as e:
            logger.error(f"❌ Error converting to canonical CID URL: {e}")
            return ""

    def _create_fallback_maps_url(self, restaurant: Dict[str, Any]) -> str:
        """
        Create fallback Google Maps URL using coordinates or name

        Args:
            restaurant: Restaurant data

        Returns:
            Fallback Google Maps URL
        """
        try:
            # Try coordinates first
            latitude = restaurant.get('latitude')
            longitude = restaurant.get('longitude')

            if latitude and longitude:
                return f"https://maps.google.com/maps?q={latitude},{longitude}"

            # Fallback to name search
            name = restaurant.get('name', '')
            if name:
                import urllib.parse
                encoded_name = urllib.parse.quote(name)
                return f"https://maps.google.com/maps?q={encoded_name}"

            return "https://maps.google.com"  # Ultimate fallback

        except Exception as e:
            logger.error(f"❌ Error creating fallback Maps URL: {e}")
            return "https://maps.google.com"

    def _format_sources(self, restaurant: Dict[str, Any]) -> List[str]:
        """
        Format sources for display

        Args:
            restaurant: Restaurant data

        Returns:
            List of formatted source strings
        """
        try:
            sources = restaurant.get('sources', [])
            if not sources:
                return ["Personal notes"]

            # Clean up source URLs for display
            formatted_sources = []
            for source in sources[:3]:  # Limit to 3 sources for location results
                if isinstance(source, str):
                    # Extract domain from URL for cleaner display
                    if source.startswith('http'):
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(source)
                            domain = parsed.netloc.replace('www.', '')
                            formatted_sources.append(domain)
                        except:
                            formatted_sources.append(source)
                    else:
                        formatted_sources.append(source)

            return formatted_sources if formatted_sources else ["Personal notes"]

        except Exception as e:
            logger.error(f"❌ Error formatting sources: {e}")
            return ["Personal notes"]

    def _create_empty_response(self, query: str, location_description: str) -> Dict[str, Any]:
        """
        Create response for when no restaurants are found

        Args:
            query: Original search query
            location_description: Location description

        Returns:
            Empty response dictionary
        """
        return {
            "main_list": [],
            "search_info": {
                "query": query,
                "location": location_description,
                "source": "personal notes",
                "count": 0
            },
            "source_type": "personal notes",
            "immediate_send": True,
            "empty_reason": "no_matching_restaurants"
        }