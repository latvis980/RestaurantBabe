# location/location_map_search_ai_editor.py
"""
AI Editor for MAP SEARCH RESULTS specifically

Features:
- Processes restaurants from Google Maps search + media verification
- Sources: Google reviews + media mentions (integrated into description text)
- Includes atmospheric filtering for map search results
- Formats: name, address, link, distance, description, media sources
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from location.location_data_logger import LocationDataLogger

# Import the original CombinedVenueData from location_ai_editor
from location.location_ai_editor import CombinedVenueData

logger = logging.getLogger(__name__)

@dataclass
class MapSearchRestaurantDescription:
    """Map search restaurant description with media integration"""
    name: str
    address: str
    maps_link: str
    distance_km: float
    description: str
    media_sources: List[str]  # Media publications for reference
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    selection_score: Optional[float] = None

class LocationMapSearchAIEditor:
    """
    AI-powered description editor specifically for MAP SEARCH RESULTS

    Handles restaurants from Google Maps search with media verification.
    Includes atmospheric filtering and media mention integration.
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.description_temperature = getattr(config, 'DESCRIPTION_TEMPERATURE', 0.3)
        self.enable_media_mention = getattr(config, 'ENABLE_MEDIA_MENTION', True)

        # Initialize AsyncOpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        self.data_logger = LocationDataLogger(config=config, enabled=True)

        logger.info("Location Map Search AI Editor initialized for map search results")

    async def create_descriptions_for_map_search_results(
        self,
        map_search_results: List[Any],
        media_verification_results: Optional[List[Any]] = None,  
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[MapSearchRestaurantDescription]:
        """
        Create descriptions for MAP SEARCH RESULTS

        Features:
        - Sources: Google reviews + media verification results
        - Media mentions: Integrated directly into description text (e.g., "featured in The Guardian")
        - Includes atmospheric filtering for quality control
        """
        try:
            logger.info(f"Creating descriptions for {{len(map_search_results)}} MAP SEARCH venues")

            if not map_search_results:
                return []

            # Step 1: Combine data from both agents
            combined_venues = self._combine_search_results(map_search_results, media_verification_results)

            self.data_logger.log_combined_data(
                map_search_results=map_search_results,
                media_verification_results=media_verification_results or [],
                combined_venues=combined_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: AI-powered restaurant selection filtering (atmospheric filtering)
            logger.info("Step 2: AI filtering for truly atmospheric restaurants")
            selected_venues = await self._filter_atmospheric_restaurants(combined_venues, user_query)

            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=selected_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Step 2: Selected {{len(selected_venues)}} truly atmospheric restaurants")

            # Step 3: Generate descriptions for MAP SEARCH results
            logger.info("Step 3: Generating descriptions for MAP SEARCH results")
            descriptions = await self._generate_map_search_descriptions(selected_venues, user_query)

            self.data_logger.log_description_generation_data(
                selected_venues=selected_venues,
                generated_descriptions=descriptions,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Generated {{len(descriptions)}} professional descriptions for map search")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_descriptions_for_map_search_results: {{e}}")
            return []

    def _combine_search_results(
        self, 
        map_search_results: List[Any], 
        media_verification_results: Optional[List[Any]] = None
    ) -> List[CombinedVenueData]:
        """Combine Google Maps results with media verification data"""
        combined_venues = []
        media_results = media_verification_results or []

        # Create a lookup for media results by restaurant name
        media_lookup = {{}}
        for media_result in media_results:
            if hasattr(media_result, 'restaurant_name'):
                media_lookup[media_result.restaurant_name.lower()] = media_result

        for i, map_result in enumerate(map_search_results):
            try:
                # Get basic info from map search
                name = getattr(map_result, 'name', 'Unknown Restaurant')

                # Find matching media verification
                media_match = media_lookup.get(name.lower())

                # Combine data
                venue_data = CombinedVenueData(
                    index=i + 1,
                    name=name,
                    address=getattr(map_result, 'address', 'Address not available'),
                    rating=getattr(map_result, 'rating', None),
                    user_ratings_total=getattr(map_result, 'user_ratings_total', 0),
                    distance_km=getattr(map_result, 'distance_km', 0.0),
                    maps_link=getattr(map_result, 'maps_link', ''),
                    place_id=getattr(map_result, 'place_id', ''),
                    cuisine_tags=getattr(map_result, 'cuisine_tags', []),
                    description=getattr(map_result, 'description', ''),
                    has_media_coverage=bool(media_match and getattr(media_match, 'articles', [])),
                    media_publications=getattr(media_match, 'publications', []) if media_match else [],
                    media_articles=getattr(media_match, 'articles', []) if media_match else [],
                    review_context=getattr(map_result, 'review_context', ''),
                    source='map_search'
                )

                combined_venues.append(venue_data)

            except Exception as e:
                logger.error(f"Error combining search results for venue {{i}}: {{e}}")
                continue

        return combined_venues

    async def _filter_atmospheric_restaurants(
        self, 
        venues: List[CombinedVenueData], 
        user_query: str
    ) -> List[CombinedVenueData]:
        """AI-powered filtering to select truly atmospheric restaurants from map search"""
        if not venues:
            return []

        try:
            # Prepare venue data for AI evaluation
            venues_text = self._format_venues_for_filtering(venues)

            # Call AI for atmospheric filtering
            selection_response = await self._call_atmospheric_filtering_ai(venues_text, user_query)

            # Parse response and filter venues
            selected_venues = self._parse_atmospheric_filtering_response(selection_response, venues)

            return selected_venues

        except Exception as e:
            logger.error(f"Error in atmospheric filtering: {{e}}")
            # Return original venues if filtering fails
            return venues

    def _format_venues_for_filtering(self, venues: List[CombinedVenueData]) -> str:
        """Format venue data for atmospheric filtering AI"""
        formatted = ""

        for venue in venues:
            formatted += f"\\n{{'='*60}}\\n"
            formatted += f"RESTAURANT {{venue.index}}: {{venue.name}}\\n"
            formatted += f"RATING: {{venue.rating or 'N/A'}}★ ({{venue.user_ratings_total}} reviews)\\n"
            formatted += f"DISTANCE: {{venue.distance_km:.1f}}km\\n"
            formatted += f"CUISINE: {{', '.join(venue.cuisine_tags) if venue.cuisine_tags else 'Not specified'}}\\n"

            if venue.has_media_coverage:
                formatted += f"MEDIA: Featured in {{', '.join(venue.media_publications)}}\\n"

            if venue.review_context:
                formatted += f"REVIEWS: {{venue.review_context[:300]}}...\\n"

            if venue.description:
                formatted += f"DESCRIPTION: {{venue.description[:200]}}...\\n"

        return formatted

    async def _call_atmospheric_filtering_ai(self, venues_text: str, user_query: str) -> str:
        """Call AI to filter for atmospheric restaurants"""

        prompt = f"""You are selecting restaurants that offer truly atmospheric dining experiences for this query: "{{user_query}}"

{{venues_text}}

SELECTION CRITERIA:
- Unique atmosphere, character, or ambiance
- Special dining experiences beyond basic food service
- Places with personality, charm, or distinctive character
- Settings that create memorable experiences
- Quality cuisine in atmospheric settings

AVOID:
- Generic chain restaurants
- Basic fast food or casual dining without character
- Places that are primarily functional rather than experiential

OUTPUT FORMAT (JSON):
{{
    "selected_restaurants": [
        {{
            "index": 1,
            "selection_score": 0.9,
            "reasoning": "Why this restaurant offers a special atmospheric experience..."
        }}
    ]
}}

Select restaurants that would create memorable dining experiences, not just satisfy hunger."""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {{
                        "role": "system",
                        "content": "You are an expert at identifying restaurants with exceptional atmosphere and dining experiences."
                    }},
                    {{
                        "role": "user", 
                        "content": prompt
                    }}
                ],
                temperature=0.2,  # Lower temperature for consistent filtering
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling atmospheric filtering AI: {{e}}")
            return "{{}}"

    def _parse_atmospheric_filtering_response(
        self, 
        response_text: str, 
        venues: List[CombinedVenueData]
    ) -> List[CombinedVenueData]:
        """Parse AI response and filter venues"""
        try:
            # Clean up response
            clean_response = response_text.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()

            parsed_response = json.loads(clean_response)
            selected_venues = []

            selected_list = parsed_response.get("selected_restaurants", [])

            for selection in selected_list:
                try:
                    index = selection.get("index", 1) - 1  # Convert to 0-based

                    if 0 <= index < len(venues):
                        venue = venues[index]
                        # Add selection score to venue
                        venue.selection_score = selection.get("selection_score", 0.8)
                        selected_venues.append(venue)

                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing selection {{selection}}: {{e}}")
                    continue

            if not selected_venues:
                logger.warning("No venues selected by atmospheric filtering, returning all")
                return venues

            logger.info(f"Atmospheric filtering selected {{len(selected_venues)}}/{{len(venues)}} venues")
            return selected_venues

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing atmospheric filtering response: {{e}}")
            return venues

    async def _generate_map_search_descriptions(
        self,
        venues: List[CombinedVenueData],
        user_query: str
    ) -> List[MapSearchRestaurantDescription]:
        """
        Generate descriptions for MAP SEARCH results with media integration
        """
        try:
            if not venues:
                return []

            # Create combined prompt with all restaurants for MAP SEARCH
            all_restaurants_data = []
            for venue in venues:
                restaurant_data = {{
                    'index': venue.index,
                    'name': venue.name,
                    'rating': venue.rating or 'N/A',
                    'user_ratings_total': venue.user_ratings_total or 0,
                    'distance_km': venue.distance_km,
                    'has_media_coverage': venue.has_media_coverage,
                    'media_publications': venue.media_publications,
                    'review_context': venue.review_context,
                    'cuisine_tags': venue.cuisine_tags,
                    'selection_score': getattr(venue, 'selection_score', 0.8)
                }}
                all_restaurants_data.append(restaurant_data)

            # Generate descriptions using AI
            descriptions_text = await self._call_map_search_description_ai(all_restaurants_data, user_query)

            # Parse AI response and create description objects
            descriptions = self._parse_map_search_descriptions_response(descriptions_text, venues)

            return descriptions

        except Exception as e:
            logger.error(f"Error generating map search descriptions: {{e}}")
            return self._create_fallback_map_search_descriptions(venues)

    async def _call_map_search_description_ai(self, restaurants_data: List[Dict[str, Any]], user_query: str) -> str:
        """Call AI to generate descriptions for map search results"""

        restaurants_text = self._format_map_search_restaurants_for_description(restaurants_data)

        media_instruction = ""
        if self.enable_media_mention:
            media_instruction = """
- If a restaurant has media coverage, naturally mention it (e.g., "featured in The Guardian" or "praised by Food & Wine")
- Media mentions should feel organic, not forced"""

        prompt = f"""You are creating engaging restaurant descriptions for this query: "{{user_query}}"

{{restaurants_text}}

TASK: Create compelling descriptions that highlight what makes each restaurant special.

KEY REQUIREMENTS:
1. Make each description unique and engaging (2-3 sentences)
2. Highlight aspects most relevant to the user's query
3. Use specific details from reviews and ratings{{media_instruction}}
4. Focus on atmosphere, cuisine quality, and distinctive features
5. Keep descriptions professional but enthusiastic

OUTPUT FORMAT (JSON):
{{
    "descriptions": [
        {{
            "index": 1,
            "description": "Engaging description with specific details and query relevance..."
        }}
    ]
}}

Make each restaurant sound appealing and distinctive based on the available information."""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {{
                        "role": "system",
                        "content": "You are a restaurant recommendation expert creating engaging descriptions for map search results."
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
            logger.error(f"Error calling map search description AI: {{e}}")
            return "{{}}"

    def _format_map_search_restaurants_for_description(self, restaurants_data: List[Dict[str, Any]]) -> str:
        """Format MAP SEARCH restaurant data for description prompt"""
        formatted = ""

        for restaurant in restaurants_data:
            formatted += f"\\n{{'='*60}}\\n"
            formatted += f"RESTAURANT {{restaurant['index']}}: {{restaurant['name']}}\\n"
            formatted += f"RATING: {{restaurant['rating']}}★ ({{restaurant['user_ratings_total']}} reviews)\\n"
            formatted += f"DISTANCE: {{restaurant['distance_km']:.1f}}km\\n"
            formatted += f"SELECTION SCORE: {{restaurant['selection_score']:.1f}}/1.0\\n"

            # Media coverage for inline mention
            if restaurant['has_media_coverage'] and restaurant['media_publications']:
                formatted += f"MEDIA COVERAGE: {{', '.join(restaurant['media_publications'])}}\\n"
            else:
                formatted += "MEDIA COVERAGE: None\\n"

            # Google reviews context
            if restaurant.get('review_context'):
                formatted += f"\\nREVIEW HIGHLIGHTS:\\n{{restaurant['review_context'][:400]}}...\\n"

            # Cuisine information
            if restaurant.get('cuisine_tags'):
                formatted += f"\\nCUISINE: {{', '.join(restaurant['cuisine_tags'])}}\\n"

        return formatted

    def _parse_map_search_descriptions_response(
        self, 
        response_text: str, 
        venues: List[CombinedVenueData]
    ) -> List[MapSearchRestaurantDescription]:
        """Parse AI response and create MapSearchRestaurantDescription objects"""
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

                        restaurant_desc = MapSearchRestaurantDescription(
                            name=venue.name,
                            address=venue.address,
                            maps_link=venue.maps_link,
                            distance_km=venue.distance_km,
                            description=desc_data.get("description", "Great restaurant option for your search."),
                            media_sources=venue.media_publications,
                            rating=venue.rating,
                            user_ratings_total=venue.user_ratings_total,
                            selection_score=getattr(venue, 'selection_score', None)
                        )

                        descriptions.append(restaurant_desc)

                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing individual map search description: {{e}}")
                    continue

            if not descriptions:
                logger.warning("No map search descriptions parsed, creating fallback")
                return self._create_fallback_map_search_descriptions(venues)

            logger.info(f"Parsed {{len(descriptions)}} map search descriptions")
            return descriptions

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing map search descriptions response: {{e}}")
            return self._create_fallback_map_search_descriptions(venues)

    def _create_fallback_map_search_descriptions(self, venues: List[CombinedVenueData]) -> List[MapSearchRestaurantDescription]:
        """Create fallback descriptions when AI parsing fails"""
        descriptions = []

        for venue in venues:
            try:
                # Create basic description
                description_parts = [f"{{venue.name}} is a well-rated restaurant"]

                if venue.cuisine_tags:
                    description_parts.append(f"serving {{', '.join(venue.cuisine_tags[:2])}} cuisine")

                if venue.rating and venue.rating >= 4.0:
                    description_parts.append(f"with excellent {{venue.rating}}★ ratings")

                if venue.has_media_coverage:
                    description_parts.append("and media recognition")

                fallback_description = " ".join(description_parts) + "."

                fallback_desc = MapSearchRestaurantDescription(
                    name=venue.name,
                    address=venue.address,
                    maps_link=venue.maps_link,
                    distance_km=venue.distance_km,
                    description=fallback_description,
                    media_sources=venue.media_publications,
                    rating=venue.rating,
                    user_ratings_total=venue.user_ratings_total,
                    selection_score=getattr(venue, 'selection_score', None)
                )

                descriptions.append(fallback_desc)

            except Exception as e:
                logger.error(f"Error creating fallback description for venue: {{e}}")
                continue

        return descriptions