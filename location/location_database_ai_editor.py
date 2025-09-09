# location/location_database_ai_editor.py
"""
AI Editor for DATABASE RESULTS specifically - SIMPLIFIED VERSION

Features:
- Processes restaurants from filter_evaluator (already filtered)
- Sources: Database sources field (domains from full article URLs in database)
- No media verification needed (data comes from database)
- No atmospheric filtering (already done by filter_evaluator)
- No CombinedVenueData needed (works directly with database dict format)
- Formats: name, address, link, distance, description, sources

SIMPLIFIED APPROACH:
- Database restaurants are already complete data structures
- Just need to enhance descriptions and format sources properly
- No need for intermediate CombinedVenueData conversion
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from formatters.google_links import build_google_maps_url
from location.location_data_logger import LocationDataLogger

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

    SIMPLIFIED: Works directly with database restaurant dictionaries.
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

        SIMPLIFIED: Works directly with database restaurant dicts.
        """
        try:
            logger.info(f"Creating descriptions for {len(database_restaurants)} DATABASE restaurants")

            if not database_restaurants:
                return []

            # Log the input data
            self.data_logger.log_combined_data(
                map_search_results=database_restaurants,
                media_verification_results=[],  # No media for database results
                combined_venues=database_restaurants,  # Use same data for both
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Generate enhanced descriptions directly from database data
            logger.info("Generating enhanced descriptions for DATABASE results")
            descriptions = await self._generate_database_descriptions(database_restaurants, user_query)

            self.data_logger.log_description_generation_data(
                selected_venues=database_restaurants,  # Use database restaurants directly
                generated_descriptions=descriptions,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Generated {len(descriptions)} enhanced descriptions for database results")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_descriptions_for_database_results: {e}")
            return []

    async def _generate_database_descriptions(
        self,
        database_restaurants: List[Dict[str, Any]],
        user_query: str
    ) -> List[DatabaseRestaurantDescription]:
        """
        Generate enhanced descriptions for DATABASE RESULTS with proper source attribution

        SIMPLIFIED: Works directly with database restaurant dictionaries.
        """
        try:
            if not database_restaurants:
                return []

            # Create combined prompt with all restaurants for DATABASE results
            all_restaurants_data = []
            for i, restaurant in enumerate(database_restaurants):
                # Extract sources and clean them
                sources = self._extract_sources_from_database_restaurant(restaurant)

                restaurant_data = {
                    'index': i + 1,
                    'name': restaurant.get('name', 'Unknown Restaurant'),
                    'rating': restaurant.get('rating') or 'N/A',
                    'user_ratings_total': restaurant.get('user_ratings_total', 0),
                    'distance_km': restaurant.get('distance_km', 0.0),
                    'cuisine_tags': restaurant.get('cuisine_tags', []),
                    'raw_description': restaurant.get('raw_description', restaurant.get('description', '')),
                    'sources': sources,
                    'mention_count': restaurant.get('mention_count', 1)
                }
                all_restaurants_data.append(restaurant_data)

            # Generate description for all restaurants at once
            restaurants_text = json.dumps(all_restaurants_data, indent=2)

            # Sources instruction for database results
            sources_instruction = """
Include source attribution naturally in descriptions using domain names from the sources field.
Use phrases like "recommended by TimeOut" or "featured in Eater" when sources are available.
"""

            # Import typing utilities at the start
            from typing import cast, Any
            
            # Use proper typed message format for OpenAI
            try:
                from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam

                messages = [
                    ChatCompletionSystemMessageParam(
                        role="system", 
                        content=f"""You are an expert food writer enhancing restaurant descriptions for DATABASE RESULTS.

{sources_instruction}

Write enhanced, engaging descriptions that:
- Build upon the existing raw_description 
- Highlight unique character and specialties
- Include source attribution when available (use domain names naturally)
- Keep the authentic voice but make it more engaging
- Focus on what makes each place special

Keep descriptions concise but evocative (2-3 sentences max).
Return ONLY a JSON array with this structure:
[
  {{
    "index": 1,
    "description": "Enhanced description that builds on the original while adding atmosphere and source attribution.",
    "selection_score": 0.85
  }}
]"""
                    ),
                    ChatCompletionUserMessageParam(
                        role="user", 
                        content=f"""Enhance descriptions for these DATABASE restaurants based on user query: "{user_query}"

Restaurants data:
{restaurants_text}

Generate enhanced descriptions for each restaurant."""
                    )
                ]
            except ImportError:
                # Fallback for older OpenAI versions - use cast to bypass type checking
                messages = cast(Any, [
                    {"role": "system", "content": f"""You are an expert food writer enhancing restaurant descriptions for DATABASE RESULTS.

{sources_instruction}

Write enhanced, engaging descriptions that:
- Build upon the existing raw_description 
- Highlight unique character and specialties
- Include source attribution when available (use domain names naturally)
- Keep the authentic voice but make it more engaging
- Focus on what makes each place special

Keep descriptions concise but evocative (2-3 sentences max).
Return ONLY a JSON array with this structure:
[
  {{
    "index": 1,
    "description": "Enhanced description that builds on the original while adding atmosphere and source attribution.",
    "selection_score": 0.85
  }}
]"""},
                    {"role": "user", "content": f"""Enhance descriptions for these DATABASE restaurants based on user query: "{user_query}"

Restaurants data:
{restaurants_text}

Generate enhanced descriptions for each restaurant."""}
                ])

            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=self.description_temperature,
                max_tokens=2000
            )

            # Handle potential None return
            response_content = response.choices[0].message.content
            if not response_content:
                logger.warning("Empty response from AI description generation")
                return self._create_fallback_database_descriptions(database_restaurants)

            # Parse AI response
            descriptions_data = json.loads(response_content.strip())

            # Create DatabaseRestaurantDescription objects
            descriptions = []
            for desc_data in descriptions_data:
                try:
                    restaurant_index = desc_data['index'] - 1  # Convert to 0-based
                    if 0 <= restaurant_index < len(database_restaurants):
                        restaurant = database_restaurants[restaurant_index]

                        # Extract sources
                        sources = self._extract_sources_from_database_restaurant(restaurant)
                        
                        # Generate Google Maps link
                        place_id = restaurant.get('place_id') or restaurant.get('google_place_id', '')
                        restaurant_name = restaurant.get('name', 'Unknown Restaurant')
                        maps_link = build_google_maps_url(place_id, restaurant_name)

                        description = DatabaseRestaurantDescription(
                            name=restaurant_name,
                            address=restaurant.get('address', 'Address not available'),
                            distance_km=restaurant.get('distance_km', 0.0),
                            description=desc_data['description'],
                            sources=sources,
                            maps_link=maps_link,
                            rating=restaurant.get('rating'),
                            user_ratings_total=restaurant.get('user_ratings_total', 0),
                            selection_score=desc_data.get('selection_score', 0.8)
                        )
                        descriptions.append(description)

                except (KeyError, IndexError, ValueError) as e:
                    logger.error(f"Error processing description data: {e}")
                    continue

            return descriptions

        except Exception as e:
            logger.error(f"Error generating database descriptions: {e}")
            return self._create_fallback_database_descriptions(database_restaurants)

    def _extract_sources_from_database_restaurant(self, restaurant: Dict[str, Any]) -> List[str]:
        """Extract and clean sources from database restaurant, returning domains only"""
        try:
            sources_field = restaurant.get('sources', [])

            if not sources_field:
                return []

            # Handle different source formats
            if isinstance(sources_field, list):
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

            # Extract domains from URLs
            domains = []
            for source in sources_list:
                domain = self._extract_domain_from_url(str(source).strip())
                if domain and domain not in domains:
                    domains.append(domain)

            return domains[:5]  # Limit to 5 domains max

        except Exception as e:
            logger.error(f"Error extracting sources: {e}")
            return []

    def _extract_domain_from_url(self, url: str) -> Optional[str]:
        """Extract clean domain name from URL"""
        try:
            import re
            from urllib.parse import urlparse

            if not url:
                return None

            # Handle cases where it might already be just a domain
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if not domain:
                return None

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Convert common domains to readable names
            domain_mapping = {
                'timeout.com': 'Time Out',
                'eater.com': 'Eater',
                'nytimes.com': 'New York Times',
                'washingtonpost.com': 'Washington Post',
                'theguardian.com': 'The Guardian',
                'foodandwine.com': 'Food & Wine',
                'bonappetit.com': 'Bon Appétit',
                'zagat.com': 'Zagat',
                'yelp.com': 'Yelp',
                'tripadvisor.com': 'TripAdvisor'
            }

            return domain_mapping.get(domain, domain.replace('.com', '').title())

        except Exception:
            return None

    def _create_fallback_database_descriptions(
        self, 
        database_restaurants: List[Dict[str, Any]]
    ) -> List[DatabaseRestaurantDescription]:
        """Create fallback descriptions when AI generation fails"""
        descriptions = []

        for restaurant in database_restaurants:
            try:
                sources = self._extract_sources_from_database_restaurant(restaurant)

                # Use existing description or create basic one
                existing_desc = restaurant.get('raw_description', restaurant.get('description', ''))
                fallback_desc = existing_desc if existing_desc else "Restaurant serving quality cuisine."

                place_id = restaurant.get('place_id', '')
                maps_link = build_google_maps_url(place_id, restaurant.get('name', '')) if place_id else restaurant.get('maps_link', '')
                
                description = DatabaseRestaurantDescription(
                    name=restaurant.get('name', 'Unknown Restaurant'),
                    address=restaurant.get('address', 'Address not available'),
                    maps_link=maps_link,
                    distance_km=restaurant.get('distance_km', 0.0),
                    description=fallback_desc,
                    sources=sources,
                    rating=restaurant.get('rating'),
                    user_ratings_total=restaurant.get('user_ratings_total', 0),
                    selection_score=0.7
                )
                descriptions.append(description)

            except Exception as e:
                logger.error(f"Error creating fallback description: {e}")
                continue

        return descriptions