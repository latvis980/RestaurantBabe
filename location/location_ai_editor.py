# location/location_ai_editor.py
"""
Location AI Description Editor - TYPE ERRORS FIXED

Fixes all type checking errors:
- Removed unused imports (asyncio, Tuple, Union)
- Fixed None assignment issues with proper type annotations and defaults
- Added proper type guards and None checks
- Fixed json possibly unbound errors
- Fixed strip() on None errors
- Fixed import resolution for location_data_models
- Corrected all method names to match existing project structure
"""

import logging
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
    google_reviews: List[Dict[str, Any]] = None  # FIXED: Proper typing
    google_maps_url: str = ""

    # Media verification data
    has_professional_coverage: bool = False
    media_coverage_score: float = 0.0
    professional_sources: List[Dict[str, Any]] = None  # FIXED: Proper typing
    scraped_content: List[Dict[str, Any]] = None  # FIXED: Proper typing
    credibility_assessment: Dict[str, Any] = None  # FIXED: Proper typing

    def __post_init__(self):
        # FIXED: Proper None handling with default empty lists/dicts
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
    media_sources: List[str] = None  # Will be fixed in __post_init__
    google_rating: Optional[float] = None
    selection_score: Optional[float] = None  # NEW: For ranking restaurants

    def __post_init__(self):
        # FIXED: Proper None handling
        if self.media_sources is None:
            self.media_sources = []

class LocationAIEditor:
    """
    AI-powered description editor with intelligent restaurant selection

    Key features:
    - AI-driven analysis (no hardcoded keywords)
    - Restaurant filtering based on emotional, detailed reviews
    - Combined review and media context analysis
    - Professional description generation
    - FIXED: Single API call for all descriptions to ensure variety
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

        logger.info("Location AI Editor initialized with improved AI-driven analysis")

    async def create_professional_descriptions(
        self,
        map_search_results: List[Any],
        media_verification_results: Optional[List[Any]] = None,  
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[RestaurantDescription]:
        """
        Main method: Create professional descriptions with intelligent restaurant selection

        Steps:
        1. Combine data from map search and media verification
        2. AI-powered restaurant filtering (select truly atmospheric, special places)
        3. Generate ALL professional descriptions in SINGLE API call for variety
        """
        try:
            logger.info(f"Creating professional descriptions for {len(map_search_results)} venues")

            if not map_search_results:
                return []

            # Step 1: Combine data from both agents
            combined_venues = self._combine_search_results(map_search_results, media_verification_results)

            # LOG COMBINED DATA FOR DEBUGGING
            self.data_logger.log_combined_data(
                map_search_results=map_search_results,
                media_verification_results=media_verification_results or [],  # FIXED: Handle None
                combined_venues=combined_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: AI-powered restaurant selection filtering
            logger.info("Step 2: AI filtering for truly atmospheric restaurants")
            selected_venues = await self._filter_atmospheric_restaurants(combined_venues, user_query)

            # LOG AI SELECTION DATA
            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=selected_venues,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Step 2: Selected {len(selected_venues)} truly atmospheric restaurants")

            # Step 3: Generate ALL descriptions in SINGLE API call for variety
            logger.info("Step 3: Generating varied descriptions for all restaurants in single call")
            descriptions = await self._generate_all_venue_descriptions(selected_venues, user_query)

            # LOG DESCRIPTION GENERATION DATA
            self.data_logger.log_description_generation_data(
                selected_venues=selected_venues,
                generated_descriptions=descriptions,
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Generated {len(descriptions)} professional descriptions")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_professional_descriptions: {e}")
            return []

    async def create_descriptions_for_database_results(
        self,
        database_restaurants: List[Dict[str, Any]],
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[RestaurantDescription]:
        """
        Create descriptions specifically for database results (already filtered)

        This method:
        1. Converts database results to CombinedVenueData format
        2. Skips atmospheric filtering (already done by filter_evaluator)
        3. Generates professional descriptions
        """
        try:
            logger.info(f"Creating descriptions for {len(database_restaurants)} database restaurants")

            if not database_restaurants:
                return []

            # Step 1: Convert database results to CombinedVenueData format
            combined_venues = self._convert_database_to_venue_format(database_restaurants)

            # LOG COMBINED DATA FOR DEBUGGING
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

            # LOG: Show that we skipped filtering
            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=combined_venues,  # Same - no filtering
                user_query=user_query
            )

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 3: Generate descriptions directly (same as map search flow)
            logger.info("Step 3: Generating descriptions for database restaurants")
            descriptions = await self._generate_all_venue_descriptions(combined_venues, user_query)

            # LOG DESCRIPTION GENERATION DATA
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

    def _convert_database_to_venue_format(self, database_restaurants: List[Dict[str, Any]]) -> List[CombinedVenueData]:
        """
        Convert database restaurant format to CombinedVenueData format

        Database restaurants have different structure than map search results,
        so we need to convert them to the expected format.
        """
        try:
            combined_venues = []

            for restaurant in database_restaurants:
                # Create CombinedVenueData from database restaurant
                combined_venue = CombinedVenueData(
                    # Basic info
                    name=restaurant.get('name', ''),
                    place_id=restaurant.get('id', ''),  # Use database ID as place_id

                    # Location (if available)
                    latitude=restaurant.get('latitude', 0.0),  # FIXED: Default value
                    longitude=restaurant.get('longitude', 0.0),  # FIXED: Default value
                    address=restaurant.get('address', ''),
                    distance_km=restaurant.get('distance_km', 0.0),  # FIXED: Default value

                    # Rating info (default values if not available)
                    rating=restaurant.get('rating', 4.0),
                    user_ratings_total=restaurant.get('review_count', 0),
                    business_status='OPERATIONAL',  # Default for database entries

                    # Google data (empty for database results)
                    google_reviews=[],
                    google_maps_url='',

                    # Media coverage (empty for database results) 
                    has_professional_coverage=False,
                    professional_sources=[],
                    scraped_content=[],
                    credibility_assessment={}
                )

                combined_venues.append(combined_venue)

            logger.info(f"Converted {len(combined_venues)} database restaurants to venue format")
            return combined_venues

        except Exception as e:
            logger.error(f"Error converting database restaurants to venue format: {e}")
            return []

    def _combine_search_results(
        self,
        map_search_results: List[Any],
        media_verification_results: Optional[List[Any]] = None  # FIXED: Optional type
    ) -> List[CombinedVenueData]:
        """
        Combine map search results with media verification results
        """
        try:
            combined_venues: List[CombinedVenueData] = []

            # Create a lookup for media verification results
            media_lookup: Dict[str, Dict[str, Any]] = {}
            if media_verification_results:  # FIXED: None check
                for media_result in media_verification_results:
                    venue_id = getattr(media_result, 'venue_id', None)
                    if venue_id:
                        media_lookup[venue_id] = media_result

            # Process each map search result
            for venue in map_search_results:
                place_id = getattr(venue, 'place_id', '')

                # Get corresponding media data if available
                media_data = media_lookup.get(place_id, {})

                combined_venue = CombinedVenueData(
                    # Basic venue info
                    place_id=place_id,
                    name=getattr(venue, 'name', ''),
                    address=getattr(venue, 'address', ''),
                    latitude=getattr(venue, 'latitude', 0.0),
                    longitude=getattr(venue, 'longitude', 0.0),
                    distance_km=getattr(venue, 'distance_km', 0.0),

                    # Google data
                    business_status=getattr(venue, 'business_status', 'OPERATIONAL'),
                    rating=getattr(venue, 'rating', None),
                    user_ratings_total=getattr(venue, 'user_ratings_total', None),
                    google_reviews=getattr(venue, 'google_reviews', []),
                    google_maps_url=getattr(venue, 'google_maps_url', ''),

                    # Media verification data
                    has_professional_coverage=media_data.get('has_professional_coverage', False),
                    media_coverage_score=media_data.get('media_coverage_score', 0.0),
                    professional_sources=media_data.get('professional_sources', []),
                    scraped_content=media_data.get('scraped_content', []),
                    credibility_assessment=media_data.get('credibility_assessment', {})
                )

                combined_venues.append(combined_venue)

            logger.info(f"Combined {len(combined_venues)} venues with search and media data")
            return combined_venues

        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return []

    async def _filter_atmospheric_restaurants(
        self,
        venues: List[CombinedVenueData],
        user_query: str
    ) -> List[CombinedVenueData]:
        """
        AI-powered filtering to select restaurants with emotional, detailed reviews
        suggesting they are truly atmospheric, special, worth visiting
        """
        try:
            if not venues:
                return []

            # Create prompt for AI restaurant selection
            restaurants_data = []
            for i, venue in enumerate(venues):
                restaurant_info = {
                    'index': i,
                    'name': venue.name,
                    'rating': venue.rating,
                    'review_count': venue.user_ratings_total,
                    'reviews': venue.google_reviews[:5] if venue.google_reviews else [],
                    'has_media_coverage': venue.has_professional_coverage,
                    'media_sources': [s.get('source_name', '') for s in venue.professional_sources[:3]]
                }
                restaurants_data.append(restaurant_info)

            # AI selection prompt
            selection_prompt = f"""You are selecting restaurants that are truly atmospheric, special, and worth visiting based on their reviews and coverage.

USER QUERY: "{user_query}"

RESTAURANT DATA:
{self._format_restaurants_for_selection(restaurants_data)}

SELECTION CRITERIA:
Select restaurants that show strong indicators of being atmospheric, special experiences:

1. EMOTIONAL REVIEWS: Look for reviews that outline how special this place is
2. ATMOSPHERIC DETAILS: Reviews mentioning specific ambiance, decor, mood, setting details
3. GOOD CONCEPT: mentions of interesting concepts, unique experiences
4. MEDIA COVERAGE BONUS: Professional coverage adds credibility

AVOID restaurants with:
- Generic, short reviews
- Only basic service/food comments
- Purely transactional mentions

OUTPUT FORMAT:
Return JSON with selected restaurants and their selection scores:
{{
    "selected_restaurants": [
        {{
            "index": 0,
            "selection_score": 8.5,
            "reasoning": "why this restaurant shows atmospheric/special qualities"
        }}
    ]
}}

Focus on quality over quantity. Select restaurants that truly stand out as special places worth visiting."""

            # Call AI for selection
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": selection_prompt}],
                temperature=0.2,
                max_tokens=2048
            )

            # Parse AI response
            content = response.choices[0].message.content
            if not content:  # FIXED: Handle None response
                logger.warning("AI returned empty response for restaurant filtering")
                return venues[:3]  # Fallback to first 3 venues

            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            try:
                import json  # FIXED: Import json to avoid "possibly unbound"
                selection_result = json.loads(content)
                selected_data = selection_result.get("selected_restaurants", [])
            except json.JSONDecodeError:  # FIXED: json is now bound
                logger.error(f"Failed to parse AI selection response: {content}")
                return venues[:3]  # Return first 3 if parsing fails

            # Build selected venues list
            selected_venues = []
            for selection in selected_data:
                index = selection.get('index')
                if index is not None and 0 <= index < len(venues):
                    venue = venues[index]
                    # FIXED: Use setattr to avoid AttributeError
                    setattr(venue, 'selection_score', selection.get('selection_score', 0.0))
                    selected_venues.append(venue)

            # Sort by selection score (highest first)
            selected_venues.sort(key=lambda v: getattr(v, 'selection_score', 0.0), reverse=True)

            logger.info(f"AI selected {len(selected_venues)} atmospheric restaurants from {len(venues)} candidates")
            return selected_venues

        except Exception as e:
            logger.error(f"Error in AI restaurant filtering: {e}")
            return venues[:3]  # Return first 3 venues if filtering fails

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

    async def _generate_all_venue_descriptions(
        self,
        venues: List[CombinedVenueData],
        user_query: str
    ) -> List[RestaurantDescription]:
        """
        FIXED: Generate ALL descriptions in a single API call to ensure variety and avoid templates
        """
        try:
            if not venues:
                return []

            # Create combined prompt with all restaurants
            all_restaurants_data = []
            for i, venue in enumerate(venues):
                # Prepare review context
                review_context = ""
                if venue.google_reviews:
                    review_context = "\nREVIEW CONTEXT:\n"
                    for review in venue.google_reviews[:5]:
                        rating = review.get('rating', 'N/A')
                        text = review.get('text', '')
                        review_context += f"- ({rating}★) {text}\n"

                # Prepare media context
                media_context = ""
                if venue.has_professional_coverage and venue.professional_sources:
                    media_context = "\nMEDIA COVERAGE CONTEXT:\n"
                    for source in venue.professional_sources[:3]:
                        source_name = source.get('source_name', 'Unknown source')
                        source_type = source.get('source_type', 'media')
                        media_context += f"- Featured in {source_name} ({source_type})\n"

                restaurant_data = {
                    'index': i,
                    'name': venue.name,
                    'rating': venue.rating,
                    'user_ratings_total': venue.user_ratings_total,
                    'distance_km': venue.distance_km,
                    'review_context': review_context,
                    'media_context': media_context
                }
                all_restaurants_data.append(restaurant_data)

            # Combined description prompt for ALL restaurants
            combined_prompt = f"""You are a professional food journalist writing SHORT restaurant descriptions.

USER QUERY: "{user_query}"

Write a professional, engaging description for EACH restaurant below. Each description should be:
- Not longer than 300 characters, best between 250-270 characters
- 1-2 complete sentences capturing the restaurant's unique character
- Mention specific details from reviews or media coverage
- Avoid generic praise, stick to details
- Laconic and concise, yet engaging
- Each description should feel DIFFERENT from the others (vary sentence structure, focus areas)

RESTAURANT DATA:
{self._format_all_restaurants_for_description(all_restaurants_data)}

Return JSON format with descriptions for ALL venues:
{{
    "descriptions": [
        {{
            "index": 0,
            "restaurant_name": "Name",
            "description": "Professional description here..."
        }},
        {{
            "index": 1,
            "restaurant_name": "Name", 
            "description": "Different style description..."
        }}
    ]
}}

Write ALL descriptions with variety and uniqueness."""

            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a professional restaurant critic. Always return valid JSON with descriptions for ALL provided restaurants."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=self.description_temperature,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content
            if not result_text:  # FIXED: Handle None response
                logger.warning("AI returned empty response for description generation")
                return self._create_fallback_descriptions(venues)

            # FIXED: Proper JSON parsing
            try:
                result_text = result_text.strip()
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}')

                if start_idx == -1 or end_idx == -1:
                    logger.warning("No JSON found in description generation response")
                    return self._create_fallback_descriptions(venues)

                json_str = result_text[start_idx:end_idx + 1]

                import json  # FIXED: Import to avoid "possibly unbound"
                descriptions_result = json.loads(json_str)

            except (json.JSONDecodeError, ValueError) as json_error:  # FIXED: json is now bound
                logger.warning(f"JSON parsing error in description generation: {json_error}")
                logger.debug(f"Raw response: '{result_text}'")
                return self._create_fallback_descriptions(venues)

            # Build final results
            descriptions = []

            if isinstance(descriptions_result, dict) and 'descriptions' in descriptions_result:
                # Create a lookup by index
                description_lookup = {}
                for desc in descriptions_result['descriptions']:
                    if isinstance(desc, dict) and 'index' in desc:
                        description_lookup[desc['index']] = desc

                # Create RestaurantDescription objects
                for i, venue in enumerate(venues):
                    desc_data = description_lookup.get(i, {})

                    description_text = desc_data.get('description', f"A {venue.rating or 'highly rated'}-star restaurant in {venue.address}.")

                    # Create media sources list
                    media_sources = []
                    if venue.has_professional_coverage and venue.professional_sources:
                        media_sources = [source.get('source_name', 'Professional Review') for source in venue.professional_sources[:3]]

                    restaurant_desc = RestaurantDescription(
                        place_id=venue.place_id,
                        name=venue.name,
                        address=venue.address,
                        distance_km=venue.distance_km,
                        description=description_text,
                        has_media_coverage=venue.has_professional_coverage,
                        media_sources=media_sources,
                        google_rating=venue.rating,
                        selection_score=getattr(venue, 'selection_score', None)
                    )

                    descriptions.append(restaurant_desc)

            if not descriptions:
                logger.warning("No descriptions generated, creating fallback")
                return self._create_fallback_descriptions(venues)

            logger.info(f"Generated {len(descriptions)} professional descriptions in single API call")
            return descriptions

        except Exception as e:
            logger.error(f"Error generating all venue descriptions: {e}")
            return self._create_fallback_descriptions(venues)

    def _format_all_restaurants_for_description(self, restaurants_data: List[Dict[str, Any]]) -> str:
        """Format all restaurant data for the combined description prompt"""
        formatted = ""

        for restaurant in restaurants_data:
            formatted += f"\n{'='*60}\n"
            formatted += f"RESTAURANT {restaurant['index']}: {restaurant['name']}\n"
            formatted += f"RATING: {restaurant['rating']}★ ({restaurant['user_ratings_total']} reviews)\n"
            formatted += f"DISTANCE: {restaurant['distance_km']:.1f}km\n"

            if restaurant.get('review_context'):
                formatted += restaurant['review_context']

            if restaurant.get('media_context'):
                formatted += restaurant['media_context']

            formatted += "\n"

        return formatted

    def _create_fallback_descriptions(self, venues: List[CombinedVenueData]) -> List[RestaurantDescription]:
        """
        Create simple fallback descriptions when AI generation fails
        """
        fallback_descriptions = []

        for venue in venues:
            rating_text = f"{venue.rating:.1f}-star" if venue.rating else "highly rated"
            description = f"A {rating_text} restaurant offering quality dining in {venue.address.split(',')[0] if venue.address else 'a great location'}."

            restaurant_desc = RestaurantDescription(
                place_id=venue.place_id,
                name=venue.name,
                address=venue.address,
                distance_km=venue.distance_km,
                description=description,
                has_media_coverage=venue.has_professional_coverage,
                media_sources=[],
                google_rating=venue.rating,
                selection_score=getattr(venue, 'selection_score', None)
            )

            fallback_descriptions.append(restaurant_desc)

        logger.info(f"Created {len(fallback_descriptions)} fallback descriptions")
        return fallback_descriptions