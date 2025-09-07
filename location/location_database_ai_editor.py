# location/location_database_ai_editor.py
"""
AI Editor for DATABASE RESULTS specifically

Features:
- Processes restaurants from filter_evaluator (already filtered)
- Sources: Database sources field (domains from full article URLs in database)
- No media verification needed
- No atmospheric filtering (already done by filter_evaluator)
- Formats: name, address, link, distance, description, sources
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from location.location_data_logger import LocationDataLogger
from location.location_utils import CombinedVenueData

logger = logging.getLogger(__name__)

@dataclass
class DatabaseRestaurantDescription:
    """Database restaurant description with sources from database"""
    name: str
    address: str
    maps_link: str
    distance_km: float
    description: str
    sources: List[str]  # Domains from database sources field
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    selection_score: Optional[float] = None

class LocationDatabaseAIEditor:
    """
    AI-powered description editor specifically for DATABASE RESULTS

    Handles restaurants that come from the filter_evaluator - these are already
    filtered and just need description enhancement with proper source attribution.
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.description_temperature = getattr(config, 'DESCRIPTION_TEMPERATURE', 0.3)

        # Initialize AsyncOpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        self.data_logger = LocationDataLogger(config=config, enabled=True)

        logger.info("Location Database AI Editor initialized for database results")

    async def create_descriptions_for_database_results(
        self,
        database_restaurants: List[Dict[str, Any]],
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[DatabaseRestaurantDescription]:
        """
        Create descriptions for DATABASE RESULTS

        Features:
        - Sources: Database descriptions + database sources field (domains)
        - No media verification needed
        - No atmospheric filtering (already filtered by filter_evaluator)
        - Format: name, address, link, distance, description, sources
        """
        try:
            logger.info(f"Creating descriptions for {{len(database_restaurants)}} DATABASE restaurants")

            if not database_restaurants:
                return []

            # Step 1: Convert database results to venue format for processing
            combined_venues = self._convert_database_to_venue_format(database_restaurants)

            self.data_logger.log_combined_data(
                map_search_results=database_restaurants,
                media_verification_results=[],  # No media for database results
                combined_venues=combined_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: SKIP atmospheric filtering - database results are pre-filtered by filter_evaluator
            logger.info("Step 2: SKIPPING atmospheric filtering for database results (already filtered)")

            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=combined_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 3: Generate enhanced descriptions for DATABASE results
            logger.info("Step 3: Generating enhanced descriptions for DATABASE results")
            descriptions = await self._generate_database_descriptions(combined_venues, database_restaurants, user_query)

            self.data_logger.log_description_generation_data(
                selected_venues=combined_venues,
                generated_descriptions=descriptions,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Generated {{len(descriptions)}} enhanced descriptions for database results")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_descriptions_for_database_results: {{e}}")
            return []

    def _convert_database_to_venue_format(self, database_restaurants: List[Dict[str, Any]]) -> List[CombinedVenueData]:
        """Convert database restaurant format to CombinedVenueData for processing"""
        combined_venues = []

        for i, restaurant in enumerate(database_restaurants):
            try:
                # Extract basic info
                name = restaurant.get('name', 'Unknown Restaurant')
                address = restaurant.get('address', 'Address not available')

                # Create venue data structure
                venue_data = CombinedVenueData(
                    index=i + 1,
                    name=name,
                    address=address,
                    rating=restaurant.get('rating'),
                    user_ratings_total=restaurant.get('user_ratings_total', 0),
                    distance_km=restaurant.get('distance_km', 0.0),
                    maps_link=restaurant.get('maps_link', ''),
                    place_id=restaurant.get('place_id', ''),
                    cuisine_tags=restaurant.get('cuisine_tags', []),
                    description=restaurant.get('description', ''),
                    has_media_coverage=False,  # Database results don't have media verification
                    media_publications=[],
                    media_articles=[],
                    review_context=restaurant.get('review_context', ''),
                    source='database'
                )

                combined_venues.append(venue_data)

            except Exception as e:
                logger.error(f"Error converting database restaurant {{i}}: {{e}}")
                continue

        return combined_venues

    async def _generate_database_descriptions(
        self,
        venues: List[CombinedVenueData],
        original_database_restaurants: List[Dict[str, Any]],
        user_query: str
    ) -> List[DatabaseRestaurantDescription]:
        """
        Generate enhanced descriptions for DATABASE RESULTS with proper source attribution
        """
        try:
            if not venues:
                return []

            # Create combined prompt with all restaurants for DATABASE results
            all_restaurants_data = []
            for i, venue in enumerate(venues):
                # Get original database restaurant for sources
                original_restaurant = original_database_restaurants[i] if i < len(original_database_restaurants) else {{}}

                restaurant_data = {{
                    'index': venue.index,
                    'name': venue.name,
                    'rating': venue.rating or 'N/A',
                    'user_ratings_total': venue.user_ratings_total or 0,
                    'distance_km': venue.distance_km,
                    'database_description': venue.description,
                    'database_sources': self._extract_sources_from_database_restaurant(original_restaurant),
                    'cuisine_tags': venue.cuisine_tags,
                }}
                all_restaurants_data.append(restaurant_data)

            # Generate enhanced descriptions using AI
            descriptions_text = await self._call_database_description_ai(all_restaurants_data, user_query)

            # Parse AI response and create description objects
            descriptions = self._parse_database_descriptions_response(descriptions_text, venues, original_database_restaurants)

            return descriptions

        except Exception as e:
            logger.error(f"Error generating database descriptions: {{e}}")
            return self._create_fallback_database_descriptions(venues, original_database_restaurants)

    async def _call_database_description_ai(self, restaurants_data: List[Dict[str, Any]], user_query: str) -> str:
        """Call AI to generate enhanced descriptions for database results"""

        restaurants_text = self._format_database_restaurants_for_description(restaurants_data)

        prompt = f"""You are enhancing restaurant descriptions from my personal database for this query: "{{user_query}}"

{{restaurants_text}}

TASK: Create enhanced, engaging descriptions that highlight what makes each restaurant special and relevant to the user's query.

KEY REQUIREMENTS:
1. Keep the restaurant's core identity from the database description
2. Highlight aspects most relevant to the user's query
3. Be specific about cuisine, atmosphere, and standout features
4. Keep descriptions concise but engaging (2-3 sentences)
5. Don't invent facts not supported by the database description

OUTPUT FORMAT (JSON):
{{
    "descriptions": [
        {{
            "index": 1,
            "description": "Enhanced description highlighting query relevance..."
        }}
    ]
}}

Focus on making each description compelling while staying true to the database information."""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {{
                        "role": "system",
                        "content": "You are a restaurant recommendation expert enhancing descriptions from a personal database."
                    }},
                    {{
                        "role": "user", 
                        "content": prompt
                    }}
                ],
                temperature=self.description_temperature,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling database description AI: {{e}}")
            return "{{}}"

    def _format_database_restaurants_for_description(self, restaurants_data: List[Dict[str, Any]]) -> str:
        """Format DATABASE restaurant data for description prompt"""
        formatted = ""

        for restaurant in restaurants_data:
            formatted += f"\\n{{'='*60}}\\n"
            formatted += f"RESTAURANT {{restaurant['index']}}: {{restaurant['name']}}\\n"
            formatted += f"RATING: {{restaurant['rating']}}★ ({{restaurant['user_ratings_total']}} reviews)\\n"
            formatted += f"DISTANCE: {{restaurant['distance_km']:.1f}}km\\n"
            formatted += f"CUISINE: {{', '.join(restaurant['cuisine_tags']) if restaurant['cuisine_tags'] else 'Not specified'}}\\n"
            formatted += f"\\nDATABASE DESCRIPTION:\\n{{restaurant['database_description']}}\\n"

            # Show sources for context
            if restaurant['database_sources']:
                formatted += f"\\nSOURCES: {{', '.join(restaurant['database_sources'])}}\\n"

        return formatted

    def _parse_database_descriptions_response(
        self, 
        response_text: str, 
        venues: List[CombinedVenueData],
        original_database_restaurants: List[Dict[str, Any]]
    ) -> List[DatabaseRestaurantDescription]:
        """Parse AI response and create DatabaseRestaurantDescription objects"""
        try:
            # Clean up response text
            clean_response = response_text.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()

            parsed_response = json.loads(clean_response)
            descriptions = []

            descriptions_list = parsed_response.get("descriptions", [])

            for desc_data in descriptions_list:
                try:
                    index = desc_data.get("index", 1) - 1  # Convert to 0-based

                    if 0 <= index < len(venues):
                        venue = venues[index]
                        original_restaurant = original_database_restaurants[index] if index < len(original_database_restaurants) else {{}}

                        # Extract sources from original database restaurant
                        sources = self._extract_sources_from_database_restaurant(original_restaurant)

                        restaurant_desc = DatabaseRestaurantDescription(
                            name=venue.name,
                            address=venue.address,
                            maps_link=venue.maps_link,
                            distance_km=venue.distance_km,
                            description=desc_data.get("description", venue.description),
                            sources=sources,
                            rating=venue.rating,
                            user_ratings_total=venue.user_ratings_total,
                            selection_score=getattr(venue, 'selection_score', None)
                        )

                        descriptions.append(restaurant_desc)

                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing individual database description: {{e}}")
                    continue

            if not descriptions:
                logger.warning("No database descriptions parsed, creating fallback")
                return self._create_fallback_database_descriptions(venues, original_database_restaurants)

            logger.info(f"Parsed {{len(descriptions)}} database descriptions")
            return descriptions

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing database descriptions response: {{e}}")
            return self._create_fallback_database_descriptions(venues, original_database_restaurants)

    def _extract_sources_from_database_restaurant(self, restaurant: Dict[str, Any]) -> List[str]:
        """Extract and clean sources from database restaurant, returning domains only"""
        try:
            sources_field = restaurant.get('sources', [])

            if not sources_field:
                return []

            # Handle different source formats
            if isinstance(sources_field, list):
                # Already a list
                sources_list = sources_field
            elif isinstance(sources_field, str):
                # Parse string representation
                sources_str = sources_field.strip()

                if not sources_str or sources_str.lower() in ['none', 'null', '[]']:
                    return []

                # Try to parse as Python literal (list/tuple)
                try:
                    import ast
                    parsed_sources = ast.literal_eval(sources_str)
                    if isinstance(parsed_sources, (list, tuple)):
                        sources_list = [s for s in parsed_sources if s and str(s).strip()]
                    elif parsed_sources:
                        sources_list = [str(parsed_sources).strip()]
                    else:
                        sources_list = []
                except (ValueError, SyntaxError):
                    # Fall back to comma-separated parsing
                    if ',' in sources_str:
                        sources_list = [s.strip() for s in sources_str.split(',') if s.strip()]
                    else:
                        sources_list = [sources_str] if sources_str else []
            else:
                sources_list = []

            # Extract domains from full URLs and clean up
            cleaned_sources = []
            for source in sources_list:
                if source and str(source).strip():
                    source_str = str(source).strip()

                    # Extract domain from URL if it's a full URL
                    if source_str.startswith(('http://', 'https://')):
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(source_str).netloc
                            if domain:
                                cleaned_sources.append(domain)
                        except Exception:
                            cleaned_sources.append(source_str)
                    else:
                        # Already a domain or short reference
                        cleaned_sources.append(source_str)

            return cleaned_sources

        except Exception as e:
            logger.debug(f"Error extracting sources from database restaurant: {{e}}")
            return []

    def _create_fallback_database_descriptions(
        self, 
        venues: List[CombinedVenueData],
        original_database_restaurants: List[Dict[str, Any]]
    ) -> List[DatabaseRestaurantDescription]:
        """Create fallback descriptions when AI parsing fails"""
        descriptions = []

        for i, venue in enumerate(venues):
            try:
                original_restaurant = original_database_restaurants[i] if i < len(original_database_restaurants) else {{}}
                sources = self._extract_sources_from_database_restaurant(original_restaurant)

                # Get description from venue or original restaurant
                description = getattr(venue, 'description', '')
                if not description:
                    description = original_restaurant.get('description', '')
                if not description:
                    description = f"{{venue.name}} is a restaurant that may match your search criteria."

                fallback_desc = DatabaseRestaurantDescription(
                    name=venue.name,
                    address=venue.address,
                    maps_link=venue.google_maps_url,
                    distance_km=venue.distance_km,
                    description=description,
                    sources=sources,
                    rating=venue.rating,
                    user_ratings_total=venue.user_ratings_total,
                    selection_score=getattr(venue, 'selection_score', None)
                )

                descriptions.append(fallback_desc)

            except Exception as e:
                logger.error(f"Error creating fallback description for venue {{i}}: {{e}}")
                continue

        return descriptions