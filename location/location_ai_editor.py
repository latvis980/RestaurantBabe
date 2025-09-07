# location/location_ai_editor.py
"""
Location AI Description Editor - Enhanced with Separate Methods - ALL TYPE ERRORS FIXED

Key improvements:
- Separate methods for map search vs database results
- Different formatting for media sources (inline vs sources field)
- Proper handling of media verification data vs database sources
- Enhanced prompts for each result type
- FIXED: All Pyright "possibly unbound" errors using proper initialization patterns
"""

import logging
import json
import ast
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from location.location_data_logger import LocationDataLogger

logger = logging.getLogger(__name__)

@dataclass
class CombinedVenueData:
    """Combined venue data from map search and media verification"""
    # Basic venue info (from map search)
    place_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float

    # Google data (from map search)
    business_status: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    google_reviews: Optional[List[Dict[str, Any]]] = None
    google_maps_url: str = ""

    # Media verification data
    has_professional_coverage: bool = False
    media_coverage_score: float = 0.0
    professional_sources: Optional[List[Dict[str, Any]]] = None
    scraped_content: Optional[List[Dict[str, Any]]] = None
    credibility_assessment: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.google_reviews is None:
            self.google_reviews = []
        if self.professional_sources is None:
            self.professional_sources = []
        if self.scraped_content is None:
            self.scraped_content = []
        if self.credibility_assessment is None:
            self.credibility_assessment = {}


@dataclass
class RestaurantDescription:
    """Final restaurant description result"""
    place_id: str
    name: str
    address: str
    distance_km: float
    description: str
    has_media_coverage: bool = False
    media_sources: Optional[List[str]] = None
    google_rating: Optional[float] = None
    selection_score: Optional[float] = None
    sources: Optional[List[str]] = None

    def __post_init__(self):
        if self.media_sources is None:
            self.media_sources = []
        if self.sources is None:
            self.sources = []

class LocationAIEditor:
    """
    AI-powered description editor with separate methods for different result types
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

        logger.info("Location AI Editor initialized with separate methods for map/database results")

    async def create_descriptions_for_map_search_results(
        self,
        map_search_results: List[Any],
        media_verification_results: Optional[List[Any]] = None,  
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[RestaurantDescription]:
        """
        Create descriptions for MAP SEARCH RESULTS

        Features:
        - Sources: Google reviews + media verification results
        - Media mentions: Integrated directly into description text (e.g., "featured in The Guardian")
        - No separate sources field needed
        """
        try:
            logger.info(f"Creating descriptions for {len(map_search_results)} MAP SEARCH venues")

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

            # Step 2: AI-powered restaurant selection filtering
            logger.info("Step 2: AI filtering for truly atmospheric restaurants")
            selected_venues = await self._filter_atmospheric_restaurants(combined_venues, user_query)

            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=selected_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Step 2: Selected {len(selected_venues)} truly atmospheric restaurants")

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

            logger.info(f"Generated {len(descriptions)} professional descriptions for map search")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_descriptions_for_map_search_results: {e}")
            return []

    async def create_descriptions_for_database_results(
        self,
        database_restaurants: List[Dict[str, Any]],
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[RestaurantDescription]:
        """
        Create descriptions for DATABASE RESULTS

        Features:
        - Sources: Database descriptions + database sources field
        - Media mentions: N/A (no media verification for database)
        - Separate sources field preserved from database
        """
        try:
            logger.info(f"Creating descriptions for {len(database_restaurants)} DATABASE restaurants")

            if not database_restaurants:
                return []

            # Step 1: Convert database results to venue format
            combined_venues = self._convert_database_to_venue_format(database_restaurants)

            self.data_logger.log_combined_data(
                map_search_results=database_restaurants,
                media_verification_results=[],  # No media for database results
                combined_venues=combined_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: SKIP atmospheric filtering - database results are pre-filtered
            logger.info("Step 2: SKIPPING atmospheric filtering for database results")

            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=combined_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 3: Generate descriptions for DATABASE results
            logger.info("Step 3: Generating descriptions for DATABASE results")
            descriptions = await self._generate_database_descriptions(combined_venues, database_restaurants, user_query)

            self.data_logger.log_description_generation_data(
                selected_venues=combined_venues,
                generated_descriptions=descriptions,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Generated {len(descriptions)} professional descriptions for database results")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_descriptions_for_database_results: {e}")
            return []

    async def _generate_map_search_descriptions(
        self,
        venues: List[CombinedVenueData],
        user_query: str
    ) -> List[RestaurantDescription]:
        """
        Generate descriptions for MAP SEARCH RESULTS

        Key differences:
        - Media sources mentioned directly in text (e.g., "featured in The Guardian")
        - No separate sources field
        - Focus on Google reviews + media verification data
        """
        try:
            if not venues:
                return []

            # Create combined prompt with all restaurants for MAP SEARCH
            all_restaurants_data = []
            for i, venue in enumerate(venues):
                # Prepare review context from Google reviews
                review_context = ""
                if venue.google_reviews:
                    review_context = "\nGOOGLE REVIEW CONTEXT:\n"
                    for review in venue.google_reviews[:5]:
                        rating = review.get('rating', 'N/A')
                        text = review.get('text', '')
                        review_context += f"- ({rating}★) {text}\n"

                # Prepare media context for inline mention
                media_publications = []
                if venue.has_professional_coverage and venue.professional_sources:
                    for source in venue.professional_sources[:3]:
                        source_name = source.get('source_name', '')
                        if source_name:
                            media_publications.append(source_name)

                restaurant_data = {
                    'index': i,
                    'name': venue.name,
                    'rating': venue.rating,
                    'user_ratings_total': venue.user_ratings_total,
                    'distance_km': venue.distance_km,
                    'review_context': review_context,
                    'media_publications': media_publications,
                    'has_media_coverage': venue.has_professional_coverage
                }
                all_restaurants_data.append(restaurant_data)

            # MAP SEARCH specific prompt
            combined_prompt = f"""You are a professional food journalist writing SHORT restaurant descriptions for MAP SEARCH RESULTS.

USER QUERY: "{user_query}"

Write a professional, engaging description for EACH restaurant below. For MAP SEARCH results:
- 1-2 complete sentences capturing the restaurant's unique character
- Use Google reviews to highlight specific dishes, atmosphere, or experiences
- If restaurant has media coverage, mention publications DIRECTLY IN TEXT (e.g., "featured in The Guardian" or "praised by Time Out")
- Professional yet warm tone
- Each description should feel DIFFERENT (vary sentence structure, focus areas)
- DO NOT use separate sources field - integrate media mentions into description text

RESTAURANT DATA:
{self._format_map_search_restaurants_for_description(all_restaurants_data)}

Return JSON format with descriptions for ALL venues:
{{{{
    "descriptions": [
        {{{{
            "index": 0,
            "restaurant_name": "Name",
            "description": "Professional description mentioning media coverage inline if available..."
        }}}},
        {{{{
            "index": 1,
            "restaurant_name": "Name", 
            "description": "Different style description..."
        }}}}
    ]
}}}}

Write ALL descriptions with variety and uniqueness. Integrate media mentions naturally into text."""

            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a professional restaurant critic writing engaging descriptions that integrate media coverage naturally into the text."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=self.description_temperature,
                max_tokens=2000
            )

            # Parse response and create RestaurantDescription objects
            descriptions = []

            # FIXED: Initialize all variables early to prevent "possibly unbound" errors
            response_text = ""
            result_data: Dict[str, Any] = {}

            try:
                # FIXED: Safe response handling with proper None checks
                response_content = response.choices[0].message.content
                if response_content is not None:
                    response_text = str(response_content).strip()
                else:
                    response_text = ""

                # Clean JSON markers
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()

                if response_text:
                    result_data = json.loads(response_text)
                else:
                    result_data = {"descriptions": []}

                for desc_data in result_data.get('descriptions', []):
                    venue_index = desc_data.get('index', 0)
                    description_text = desc_data.get('description', '')

                    if venue_index < len(venues):
                        venue = venues[venue_index]

                        restaurant_desc = RestaurantDescription(
                            place_id=venue.place_id,
                            name=venue.name,
                            address=venue.address,
                            distance_km=venue.distance_km,
                            description=description_text,
                            has_media_coverage=venue.has_professional_coverage,
                            media_sources=[],  # Empty for map search - media mentioned in text
                            google_rating=venue.rating,
                            selection_score=getattr(venue, 'selection_score', None),
                            sources=[]  # Empty for map search
                        )

                        descriptions.append(restaurant_desc)

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing map search descriptions response: {e}")
                return self._create_fallback_descriptions(venues, result_type="map_search")

            if not descriptions:
                logger.warning("No map search descriptions generated, creating fallback")
                return self._create_fallback_descriptions(venues, result_type="map_search")

            logger.info(f"Generated {len(descriptions)} map search descriptions")
            return descriptions

        except Exception as e:
            logger.error(f"Error generating map search descriptions: {e}")
            return self._create_fallback_descriptions(venues, result_type="map_search")

    async def _generate_database_descriptions(
        self,
        venues: List[CombinedVenueData],
        original_database_restaurants: List[Dict[str, Any]],
        user_query: str
    ) -> List[RestaurantDescription]:
        """
        FIXED: Generate descriptions for DATABASE RESULTS with proper type safety
        """
        try:
            if not venues:
                return []

            # Create combined prompt with all restaurants for DATABASE
            all_restaurants_data = []
            for i, venue in enumerate(venues):
                # Get original database data with bounds checking
                original_restaurant = original_database_restaurants[i] if i < len(original_database_restaurants) else {}

                # FIXED: Safe string extraction
                existing_description = (
                    original_restaurant.get('description', '') or 
                    original_restaurant.get('raw_description', '') or
                    ''
                )

                # FIXED: Safe source extraction
                existing_sources = self._extract_sources_from_database_restaurant(original_restaurant)

                restaurant_data = {
                    'index': i,
                    'name': str(venue.name),
                    'rating': venue.rating,
                    'distance_km': venue.distance_km,
                    'existing_description': str(existing_description),
                    'existing_sources': existing_sources,  # Guaranteed List[str]
                    'cuisine_tags': original_restaurant.get('cuisine_tags', []) or []
                }
                all_restaurants_data.append(restaurant_data)

            # DATABASE specific prompt
            combined_prompt = f"""You are a professional food journalist writing SHORT restaurant descriptions for DATABASE RESULTS.

USER QUERY: "{user_query}"

Write a professional, engaging description for EACH restaurant below. For DATABASE results:
- 1-2 complete sentences capturing the restaurant's unique character
- Use existing descriptions as foundation but make them more engaging and professional
- Focus on cuisine type, atmosphere, and unique features
- Professional yet warm tone
- Each description should feel DIFFERENT (vary sentence structure, focus areas)
- DO NOT mention specific media sources - sources are handled separately

RESTAURANT DATA:
{self._format_database_restaurants_for_description(all_restaurants_data)}

Return JSON format with descriptions for ALL venues:
{{{{
"descriptions": [
    {{{{
        "index": 0,
        "restaurant_name": "Name",
        "description": "Professional description based on existing database content..."
    }}}},
    {{{{
        "index": 1,
        "restaurant_name": "Name", 
        "description": "Different style description..."
    }}}}
]
}}}}

Write ALL descriptions with variety and uniqueness. Focus on cuisine and atmosphere."""

            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a professional restaurant critic enhancing existing database descriptions with engaging, professional language."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=self.description_temperature,
                max_tokens=2000
            )

            # FIXED: Initialize all variables early to prevent "possibly unbound" errors
            result_text = ""
            descriptions_result: Dict[str, Any] = {}

            try:
                # FIXED: Safe response handling
                response_content = response.choices[0].message.content
                if response_content is not None:
                    result_text = str(response_content).strip()
                else:
                    result_text = ""

                if not result_text:
                    logger.warning("AI returned empty response for database description generation")
                    return self._create_fallback_descriptions(venues, result_type="database")

                # FIXED: Safer JSON parsing with proper initialization
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}')

                if start_idx == -1 or end_idx == -1:
                    logger.warning("No JSON found in database description generation response")
                    return self._create_fallback_descriptions(venues, result_type="database")

                json_str = result_text[start_idx:end_idx + 1]
                descriptions_result = json.loads(json_str)

            except (json.JSONDecodeError, ValueError) as json_error:
                logger.warning(f"JSON parsing error in database description generation: {json_error}")
                logger.debug(f"Raw response: '{result_text}'")
                return self._create_fallback_descriptions(venues, result_type="database")

            # Build final results with FIXED sources transfer
            descriptions = []

            if isinstance(descriptions_result, dict) and 'descriptions' in descriptions_result:
                # Create a lookup by index
                description_lookup = {}
                for desc in descriptions_result['descriptions']:
                    if isinstance(desc, dict) and 'index' in desc:
                        description_lookup[desc['index']] = desc

                # Create RestaurantDescription objects with GUARANTEED types
                for i, venue in enumerate(venues):
                    desc_data = description_lookup.get(i, {})

                    # Get description from AI
                    description_text = desc_data.get('description', f"A quality restaurant in {venue.address.split(',')[0] if venue.address else 'a great location'}.")

                    # FIXED: Get original sources and place_id with type safety
                    original_restaurant = original_database_restaurants[i] if i < len(original_database_restaurants) else {}
                    original_sources = self._extract_sources_from_database_restaurant(original_restaurant)  # Guaranteed List[str]
                    original_place_id = (
                        original_restaurant.get('place_id') or 
                        original_restaurant.get('google_place_id') or
                        venue.place_id
                    )

                    # FIXED: Type-safe RestaurantDescription creation
                    restaurant_desc = RestaurantDescription(
                        place_id=str(original_place_id),
                        name=str(venue.name),
                        address=str(venue.address),
                        distance_km=float(venue.distance_km),
                        description=str(description_text),
                        has_media_coverage=False,
                        media_sources=[],  # FIXED: Always empty list for database
                        google_rating=venue.rating,
                        selection_score=getattr(venue, 'selection_score', None),
                        sources=original_sources  # FIXED: Guaranteed List[str], never None
                    )

                    descriptions.append(restaurant_desc)

            if not descriptions:
                logger.warning("No database descriptions generated, creating fallback")
                return self._create_fallback_descriptions(venues, result_type="database")

            logger.info(f"Generated {len(descriptions)} database descriptions WITH SOURCES")
            return descriptions

        except Exception as e:
            logger.error(f"Error generating database descriptions: {e}")
            return self._create_fallback_descriptions(venues, result_type="database")


    def _extract_sources_from_database_restaurant(self, restaurant: Dict[str, Any]) -> List[str]:
        """
        FIXED: Extract and parse sources with guaranteed List[str] return (never None)
        """
        try:
            sources_raw = restaurant.get('sources', [])

            # FIXED: Handle None case explicitly
            if sources_raw is None:
                return []

            # If already a list, return cleaned version
            if isinstance(sources_raw, list):
                return [str(s).strip() for s in sources_raw if s and str(s).strip()]

            # If string, try to parse
            if isinstance(sources_raw, str) and sources_raw.strip():
                sources_str = sources_raw.strip()

                # Try JSON parsing first
                try:
                    parsed_sources = json.loads(sources_str)
                    if isinstance(parsed_sources, list):
                        return [str(s).strip() for s in parsed_sources if s and str(s).strip()]
                    elif parsed_sources:  # Single item
                        return [str(parsed_sources).strip()]
                    else:
                        return []
                except json.JSONDecodeError:
                    pass

                # Try ast.literal_eval
                try:
                    parsed_sources = ast.literal_eval(sources_str)
                    if isinstance(parsed_sources, list):
                        return [str(s).strip() for s in parsed_sources if s and str(s).strip()]
                    elif parsed_sources:  # Single item
                        return [str(parsed_sources).strip()]
                    else:
                        return []
                except (ValueError, SyntaxError):
                    pass

                # Fall back to comma-separated or single string
                if ',' in sources_str:
                    return [s.strip() for s in sources_str.split(',') if s.strip()]
                else:
                    return [sources_str] if sources_str else []

            # FIXED: Always return List[str], never None
            return []

        except Exception as e:
            logger.debug(f"Error extracting sources from database restaurant: {e}")
            return []  # FIXED: Guaranteed List[str] return

    def _format_map_search_restaurants_for_description(self, restaurants_data: List[Dict[str, Any]]) -> str:
        """Format MAP SEARCH restaurant data for description prompt"""
        formatted = ""

        for restaurant in restaurants_data:
            formatted += f"\n{'='*60}\n"
            formatted += f"RESTAURANT {restaurant['index']}: {restaurant['name']}\n"
            formatted += f"RATING: {restaurant['rating']}★ ({restaurant['user_ratings_total']} reviews)\n"
            formatted += f"DISTANCE: {restaurant['distance_km']:.1f}km\n"

            # Media coverage for inline mention
            if restaurant['has_media_coverage'] and restaurant['media_publications']:
                formatted += f"MEDIA COVERAGE: {', '.join(restaurant['media_publications'])}\n"
            else:
                formatted += "MEDIA COVERAGE: None\n"

            # Google reviews context
            if restaurant.get('review_context'):
                formatted += restaurant['review_context']

            formatted += "\n"

        return formatted

    def _format_database_restaurants_for_description(self, restaurants_data: List[Dict[str, Any]]) -> str:
        """Format DATABASE restaurant data for description prompt"""
        formatted = ""

        for restaurant in restaurants_data:
            formatted += f"\n{'='*60}\n"
            formatted += f"RESTAURANT {restaurant['index']}: {restaurant['name']}\n"
            formatted += f"DISTANCE: {restaurant['distance_km']:.1f}km\n"

            # Cuisine tags
            if restaurant.get('cuisine_tags'):
                formatted += f"CUISINE: {', '.join(restaurant['cuisine_tags'])}\n"

            # Existing description from database
            if restaurant.get('existing_description'):
                formatted += f"EXISTING DESCRIPTION: {restaurant['existing_description']}\n"

            # Sources count (don't include actual sources in prompt)
            sources_count = len(restaurant.get('existing_sources', []))
            formatted += f"SOURCES AVAILABLE: {sources_count} professional sources\n"

            formatted += "\n"

        return formatted

    def _create_fallback_descriptions(self, venues: List[CombinedVenueData], result_type: str = "map_search") -> List[RestaurantDescription]:
        """
        FIXED: Create simple fallback descriptions with guaranteed types
        """
        fallback_descriptions = []

        for venue in venues:
            try:
                rating_text = f"{venue.rating:.1f}-star" if venue.rating else "highly rated"
                location_part = venue.address.split(',')[0] if venue.address else 'a great location'
                description = f"A {rating_text} restaurant offering quality dining in {location_part}."

                # FIXED: Type-safe RestaurantDescription creation
                restaurant_desc = RestaurantDescription(
                    place_id=str(venue.place_id),
                    name=str(venue.name),
                    address=str(venue.address),
                    distance_km=float(venue.distance_km),
                    description=str(description),
                    has_media_coverage=False,
                    media_sources=[],  # FIXED: Always empty list
                    google_rating=venue.rating,
                    selection_score=getattr(venue, 'selection_score', None),
                    sources=[]  # FIXED: Always empty list for fallback
                )

                fallback_descriptions.append(restaurant_desc)
            except Exception as e:
                logger.error(f"Error creating fallback description for {venue.name}: {e}")

        logger.info(f"Created {len(fallback_descriptions)} fallback descriptions for {result_type}")
        return fallback_descriptions

    # Helper methods - complete implementations
    def _combine_search_results(self, map_search_results: List[Any], media_verification_results: Optional[List[Any]] = None) -> List[CombinedVenueData]:
        """Combine map search and media verification results"""
        try:
            combined_venues = []

            # Create media lookup dictionary
            media_lookup = {}
            if media_verification_results:
                for media_result in media_verification_results:
                    venue_id = getattr(media_result, 'venue_id', '')
                    media_lookup[venue_id] = media_result

            for venue in map_search_results:
                # Extract basic venue info
                place_id = getattr(venue, 'place_id', str(venue.name))
                name = getattr(venue, 'name', 'Unknown')
                address = getattr(venue, 'address', '')
                latitude = getattr(venue, 'latitude', 0.0)
                longitude = getattr(venue, 'longitude', 0.0)
                distance_km = getattr(venue, 'distance_km', 0.0)
                business_status = getattr(venue, 'business_status', 'OPERATIONAL')
                rating = getattr(venue, 'rating', None)
                user_ratings_total = getattr(venue, 'user_ratings_total', None)
                google_reviews = getattr(venue, 'google_reviews', [])
                google_maps_url = getattr(venue, 'google_maps_url', '')

                # Find corresponding media verification data
                media_result = media_lookup.get(place_id)

                if media_result:
                    has_professional_coverage = getattr(media_result, 'has_professional_coverage', False)
                    media_coverage_score = getattr(media_result, 'media_coverage_score', 0.0)
                    professional_sources = getattr(media_result, 'professional_sources', [])
                    scraped_content = getattr(media_result, 'scraped_content', [])
                    credibility_assessment = getattr(media_result, 'credibility_assessment', {})
                else:
                    has_professional_coverage = False
                    media_coverage_score = 0.0
                    professional_sources = []
                    scraped_content = []
                    credibility_assessment = {}

                combined_venue = CombinedVenueData(
                    place_id=place_id,
                    name=name,
                    address=address,
                    latitude=latitude,
                    longitude=longitude,
                    distance_km=distance_km,
                    business_status=business_status,
                    rating=rating,
                    user_ratings_total=user_ratings_total,
                    google_reviews=google_reviews,
                    google_maps_url=google_maps_url,
                    has_professional_coverage=has_professional_coverage,
                    media_coverage_score=media_coverage_score,
                    professional_sources=professional_sources,
                    scraped_content=scraped_content,
                    credibility_assessment=credibility_assessment
                )

                combined_venues.append(combined_venue)

            logger.info(f"Combined {len(combined_venues)} venues with media verification data")
            return combined_venues

        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return []

    def _convert_database_to_venue_format(self, database_restaurants: List[Dict[str, Any]]) -> List[CombinedVenueData]:
        """
        FIXED: Convert database restaurant format to CombinedVenueData format with type safety
        """
        try:
            combined_venues = []

            for restaurant in database_restaurants:
                # FIXED: Type-safe extraction
                restaurant_id = str(restaurant.get('id', ''))
                name = str(restaurant.get('name', 'Unknown'))
                address = str(restaurant.get('address', ''))

                # FIXED: Safe float conversion
                try:
                    latitude = float(restaurant.get('latitude', 0.0))
                    longitude = float(restaurant.get('longitude', 0.0))
                    distance_km = float(restaurant.get('distance_km', 0.0))
                except (ValueError, TypeError):
                    latitude = 0.0
                    longitude = 0.0
                    distance_km = 0.0

                # FIXED: Extract place_id with type safety
                place_id = (
                    restaurant.get('place_id') or 
                    restaurant.get('google_place_id') or 
                    restaurant.get('google_maps_place_id') or
                    restaurant_id
                )
                place_id = str(place_id) if place_id else restaurant_id

                # FIXED: Type-safe venue creation with guaranteed non-None lists
                combined_venue = CombinedVenueData(
                    place_id=place_id,
                    name=name,
                    address=address,
                    latitude=latitude,
                    longitude=longitude,
                    distance_km=distance_km,
                    business_status="OPERATIONAL",
                    rating=None,
                    user_ratings_total=None,
                    google_reviews=[],  # FIXED: Always list, never None
                    google_maps_url="",
                    has_professional_coverage=False,
                    media_coverage_score=0.0,
                    professional_sources=[],  # FIXED: Always list, never None
                    scraped_content=[],  # FIXED: Always list, never None
                    credibility_assessment={}  # FIXED: Always dict, never None
                )

                combined_venues.append(combined_venue)

            logger.info(f"Converted {len(combined_venues)} database restaurants to venue format with place_id")
            return combined_venues

        except Exception as e:
            logger.error(f"Error converting database restaurants to venue format: {e}")
            return []

    async def _filter_atmospheric_restaurants(self, combined_venues: List[CombinedVenueData], user_query: str) -> List[CombinedVenueData]:
        """
        FIXED: Filter restaurants for atmospheric qualities using AI with complete type safety
        """
        try:
            if not combined_venues:
                return []

            # Prepare data for AI analysis with type safety
            restaurants_data = []
            for i, venue in enumerate(combined_venues):
                # FIXED: Safe attribute access with guaranteed types
                venue_name = getattr(venue, 'name', 'Unknown')
                venue_rating = getattr(venue, 'rating', None) or 0
                venue_review_count = getattr(venue, 'user_ratings_total', None) or 0
                venue_reviews = getattr(venue, 'google_reviews', None) or []
                venue_has_coverage = getattr(venue, 'has_professional_coverage', False)
                venue_professional_sources = getattr(venue, 'professional_sources', None) or []

                # FIXED: Safe media sources extraction
                media_sources = []
                if venue_professional_sources and isinstance(venue_professional_sources, list):
                    for source in venue_professional_sources[:3]:  # Top 3 sources
                        if source and isinstance(source, dict):
                            source_name = source.get('source_name', '')
                            if source_name:
                                media_sources.append(source_name)

                # FIXED: Safe reviews handling
                safe_reviews = []
                if venue_reviews and isinstance(venue_reviews, list):
                    for review in venue_reviews[:5]:  # Top 5 reviews
                        if review and isinstance(review, dict):
                            safe_reviews.append(review)

                restaurant_data = {
                    'index': i,
                    'name': venue_name,
                    'rating': venue_rating,
                    'review_count': venue_review_count,
                    'reviews': safe_reviews,
                    'has_media_coverage': venue_has_coverage,
                    'media_sources': media_sources
                }
                restaurants_data.append(restaurant_data)

            # AI selection prompt
            selection_prompt = f"""You are an expert restaurant curator selecting truly atmospheric, special places.

USER QUERY: "{user_query}"

Select restaurants that are TRULY ATMOSPHERIC and SPECIAL. Focus on:
- Emotional, detailed reviews mentioning specific experiences
- Unique character, ambiance, or exceptional quality
- Special occasions, memorable experiences, authentic atmosphere
- Avoid generic, chain, or purely functional restaurants

RESTAURANT DATA:
{self._format_restaurants_for_selection(restaurants_data)}

Return JSON with selected restaurant indices and scores:
{{{{
"selected_restaurants": [
    {{{{
        "index": 0,
        "name": "Restaurant Name",
        "selection_score": 8.5,
        "reason": "Brief reason for selection"
    }}}}
]
}}}}

Select 3-5 truly atmospheric restaurants with highest appeal."""

            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert restaurant curator focused on atmosphere and special experiences."},
                    {"role": "user", "content": selection_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            # FIXED: Initialize all variables early to prevent "possibly unbound" errors
            selected_venues = []
            response_text = ""
            selection_data: Dict[str, Any] = {}

            try:
                response_content = response.choices[0].message.content
                if response_content is not None:
                    response_text = str(response_content).strip()
                else:
                    response_text = ""

                if not response_text:
                    logger.warning("Empty response from atmospheric filtering AI")
                    return combined_venues[:3]  # Fallback to first 3

                # Clean JSON markers
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()

                # FIXED: Safe JSON parsing
                if response_text:
                    selection_data = json.loads(response_text)
                else:
                    selection_data = {"selected_restaurants": []}

                # FIXED: Safe data extraction with type checking
                selected_restaurants = selection_data.get('selected_restaurants', [])
                if not isinstance(selected_restaurants, list):
                    logger.warning("Invalid selection data format")
                    return combined_venues[:3]

                for selected in selected_restaurants:
                    if not isinstance(selected, dict):
                        continue

                    venue_index = selected.get('index')
                    selection_score = selected.get('selection_score', 0.0)

                    # FIXED: Safe index validation
                    if venue_index is not None and 0 <= venue_index < len(combined_venues):
                        venue = combined_venues[venue_index]
                        # Add selection score to venue (create new attribute)
                        venue.selection_score = float(selection_score)
                        selected_venues.append(venue)

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Error parsing atmospheric selection results: {e}")
                logger.debug(f"Raw response: '{response_text[:500]}'")
                # Fallback: return first 3 venues
                return combined_venues[:3]

            # FIXED: Safe sorting with getattr fallback
            if selected_venues:
                selected_venues.sort(key=lambda v: getattr(v, 'selection_score', 0.0), reverse=True)
            else:
                # If no venues selected, fallback to first 3
                selected_venues = combined_venues[:3]

            logger.info(f"AI selected {len(selected_venues)} atmospheric restaurants from {len(combined_venues)} candidates")
            return selected_venues

        except Exception as e:
            logger.error(f"Error in AI restaurant filtering: {e}")
            return combined_venues[:3]  # Return first 3 venues if filtering fails

    def _format_restaurants_for_selection(self, restaurants_data: List[Dict[str, Any]]) -> str:
        """Format restaurant data for AI selection prompt"""
        formatted = ""

        for restaurant in restaurants_data:
            formatted += f"\n{'='*50}\n"
            formatted += f"INDEX: {restaurant['index']}\n"
            formatted += f"NAME: {restaurant['name']}\n"
            formatted += f"RATING: {restaurant['rating']} ({restaurant['review_count']} reviews)\n"

            if restaurant['has_media_coverage']:
                formatted += f"MEDIA COVERAGE: Yes - {', '.join(restaurant['media_sources'])}\n"
            else:
                formatted += "MEDIA COVERAGE: No\n"

            formatted += "\nRECENT REVIEWS:\n"
            for i, review in enumerate(restaurant['reviews'][:3], 1):
                review_text = review.get('text', '')[:300]  # Limit review length
                review_rating = review.get('rating', 'N/A')
                formatted += f"Review {i} ({review_rating}★): {review_text}...\n\n"

            if not restaurant['reviews']:
                formatted += "No reviews available\n"

        return formatted

    # DEPRECATED - kept for backward compatibility
    async def create_professional_descriptions(self, *args, **kwargs):
        """
        DEPRECATED: Use create_descriptions_for_map_search_results instead
        """
        logger.warning("create_professional_descriptions is deprecated. Use create_descriptions_for_map_search_results instead.")
        return await self.create_descriptions_for_map_search_results(*args, **kwargs)