# location/location_ai_editor.py
"""
Location AI Description Editor - Updated for Separate Agent Architecture

Creates professional restaurant descriptions by combining results from:
- LocationMapSearchAgent (Google reviews, ratings, basic venue data)
- LocationMediaVerificationAgent (professional media coverage, scraped content)

Works with the new separate agent architecture while maintaining compatibility
with the location orchestrator.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import openai

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
    google_reviews: List[Dict] = None
    google_maps_url: str = ""

    # Media verification data
    has_professional_coverage: bool = False
    media_coverage_score: float = 0.0
    professional_sources: List[Dict] = None
    scraped_content: List[Dict] = None
    credibility_assessment: Dict = None

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
    media_sources: List[str] = None
    google_rating: Optional[float] = None

    def __post_init__(self):
        if self.media_sources is None:
            self.media_sources = []

class LocationAIEditor:
    """
    AI-powered description editor that combines map search and media verification results

    This editor creates professional restaurant descriptions by intelligently combining:
    1. Google Maps data (reviews, ratings, basic info) 
    2. Professional media coverage (when available)
    3. AI analysis to create engaging, contextual descriptions
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.description_temperature = getattr(config, 'DESCRIPTION_TEMPERATURE', 0.3)
        self.max_description_length = getattr(config, 'DESCRIPTION_MAX_LENGTH', 150)
        self.enable_media_mention = getattr(config, 'ENABLE_MEDIA_MENTION', True)

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        logger.info("Location AI Editor initialized for separate agent architecture")

    async def create_professional_descriptions(
        self,
        map_search_results: List[Any],  # VenueSearchResult objects
        media_verification_results: List[Any] = None,  # MediaVerificationResult objects
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[RestaurantDescription]:
        """
        Main method: Create professional descriptions from separate agent results

        Args:
            map_search_results: Results from LocationMapSearchAgent
            media_verification_results: Results from LocationMediaVerificationAgent (optional)
            user_query: Original user query for context
            cancel_check_fn: Optional cancellation check function

        Returns:
            List of RestaurantDescription objects with professional descriptions
        """
        try:
            logger.info(f"Creating professional descriptions for {len(map_search_results)} venues")

            if not map_search_results:
                return []

            # Step 1: Combine data from both agents
            combined_venues = self._combine_search_results(map_search_results, media_verification_results)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: Generate descriptions using AI
            descriptions = []

            for venue in combined_venues:
                if cancel_check_fn and cancel_check_fn():
                    break

                try:
                    description_text = await self._generate_venue_description(venue, user_query)

                    # Create result object
                    restaurant_desc = RestaurantDescription(
                        place_id=venue.place_id,
                        name=venue.name,
                        address=venue.address,
                        distance_km=venue.distance_km,
                        description=description_text,
                        has_media_coverage=venue.has_professional_coverage,
                        media_sources=[
                            source.get('source_name', 'Unknown') 
                            for source in venue.professional_sources[:3]
                        ],
                        google_rating=venue.rating
                    )

                    descriptions.append(restaurant_desc)

                except Exception as e:
                    logger.error(f"Error generating description for {venue.name}: {e}")
                    # Create fallback description
                    fallback_desc = RestaurantDescription(
                        place_id=venue.place_id,
                        name=venue.name,
                        address=venue.address,
                        distance_km=venue.distance_km,
                        description="Quality restaurant featuring carefully prepared cuisine.",
                        has_media_coverage=False,
                        google_rating=venue.rating
                    )
                    descriptions.append(fallback_desc)

            logger.info(f"Successfully created {len(descriptions)} professional descriptions")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_professional_descriptions: {e}")
            return []

    def _combine_search_results(
        self,
        map_results: List[Any],
        media_results: List[Any] = None
    ) -> List[CombinedVenueData]:
        """Combine results from map search and media verification agents"""
        try:
            # Create lookup for media results by venue ID/name
            media_lookup = {}
            if media_results:
                for media_result in media_results:
                    # Try to match by venue_id first, then by name
                    venue_id = getattr(media_result, 'venue_id', None)
                    venue_name = getattr(media_result, 'venue_name', None)

                    if venue_id:
                        media_lookup[venue_id] = media_result
                    elif venue_name:
                        media_lookup[venue_name.lower()] = media_result

            combined_venues = []

            for map_result in map_results:
                # Extract map search data
                place_id = getattr(map_result, 'place_id', '')
                name = getattr(map_result, 'name', '')
                address = getattr(map_result, 'address', '')
                latitude = getattr(map_result, 'latitude', 0.0)
                longitude = getattr(map_result, 'longitude', 0.0)
                distance_km = getattr(map_result, 'distance_km', 0.0)
                business_status = getattr(map_result, 'business_status', 'OPERATIONAL')
                rating = getattr(map_result, 'rating', None)
                user_ratings_total = getattr(map_result, 'user_ratings_total', 0)
                google_reviews = getattr(map_result, 'google_reviews', [])
                google_maps_url = getattr(map_result, 'google_maps_url', '')

                # Find corresponding media data
                media_result = None
                if place_id and place_id in media_lookup:
                    media_result = media_lookup[place_id]
                elif name and name.lower() in media_lookup:
                    media_result = media_lookup[name.lower()]

                # Extract media data if available
                has_professional_coverage = False
                media_coverage_score = 0.0
                professional_sources = []
                scraped_content = []
                credibility_assessment = {}

                if media_result:
                    has_professional_coverage = getattr(media_result, 'has_professional_coverage', False)
                    media_coverage_score = getattr(media_result, 'media_coverage_score', 0.0)
                    professional_sources = getattr(media_result, 'professional_sources', [])
                    scraped_content = getattr(media_result, 'scraped_content', [])
                    credibility_assessment = getattr(media_result, 'credibility_assessment', {})

                # Create combined venue data
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

            logger.debug(f"Combined {len(map_results)} map results with {len(media_results or [])} media results")
            return combined_venues

        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return []

    async def _generate_venue_description(
        self,
        venue: CombinedVenueData,
        user_query: str
    ) -> str:
        """Generate professional description for a single venue"""
        try:
            # Synthesize all available information
            venue_analysis = self._analyze_venue_data(venue, user_query)

            # Create AI prompt
            prompt = self._create_description_prompt(venue, venue_analysis, user_query)

            # Generate description using OpenAI
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional food editor writing concise, engaging restaurant descriptions. Focus on specific details and unique features. Avoid generic phrases."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.description_temperature,
                    max_tokens=200
                ),
                timeout=15.0
            )

            if not response or not response.choices:
                raise Exception("Empty AI response")

            description = response.choices[0].message.content
            if not description:
                raise Exception("Empty description content")

            # Clean and validate description
            description = self._clean_description(description.strip())

            # Ensure description length is appropriate
            if len(description) > self.max_description_length:
                description = description[:self.max_description_length].rsplit(' ', 1)[0] + '.'

            return description

        except Exception as e:
            logger.error(f"Error generating description for {venue.name}: {e}")
            return "Quality restaurant featuring carefully prepared cuisine."

    def _analyze_venue_data(self, venue: CombinedVenueData, user_query: str) -> Dict[str, Any]:
        """Analyze all available venue data for description generation"""
        analysis = {
            'cuisine_type': self._infer_cuisine_type(venue.name),
            'key_features': [],
            'atmosphere_notes': [],
            'standout_dishes': [],
            'professional_recognition': [],
            'rating_context': '',
            'media_boost': False
        }

        # Analyze Google reviews
        if venue.google_reviews:
            review_insights = self._extract_review_insights(venue.google_reviews)
            analysis['key_features'].extend(review_insights.get('features', []))
            analysis['atmosphere_notes'].extend(review_insights.get('atmosphere', []))
            analysis['standout_dishes'].extend(review_insights.get('dishes', []))

        # Analyze professional media coverage
        if venue.has_professional_coverage and venue.professional_sources:
            media_insights = self._extract_media_insights(venue.professional_sources, venue.scraped_content)
            analysis['professional_recognition'].extend(media_insights.get('recognition', []))
            analysis['media_boost'] = venue.media_coverage_score > 7.0

        # Rating context
        if venue.rating:
            if venue.rating >= 4.5:
                analysis['rating_context'] = 'exceptional'
            elif venue.rating >= 4.0:
                analysis['rating_context'] = 'highly rated'
            else:
                analysis['rating_context'] = 'popular'

        return analysis

    def _create_description_prompt(
        self,
        venue: CombinedVenueData,
        analysis: Dict[str, Any],
        user_query: str
    ) -> str:
        """Create AI prompt for description generation"""
        prompt = f"""Create a professional restaurant description for "{venue.name}".

USER'S QUERY: "{user_query}"
CUISINE TYPE: {analysis['cuisine_type']}
RATING CONTEXT: {analysis['rating_context']}

AVAILABLE INFORMATION:
"""

        if analysis['key_features']:
            prompt += f"Key Features: {', '.join(analysis['key_features'][:3])}\n"

        if analysis['standout_dishes']:
            prompt += f"Notable Dishes: {', '.join(analysis['standout_dishes'][:3])}\n"

        if analysis['atmosphere_notes']:
            prompt += f"Atmosphere: {', '.join(analysis['atmosphere_notes'][:2])}\n"

        if analysis['professional_recognition']:
            prompt += f"Recognition: {', '.join(analysis['professional_recognition'][:2])}\n"

        prompt += f"""
REQUIREMENTS:
1. Write 1-2 concise sentences (max {self.max_description_length} characters)
2. Relate to user's query: "{user_query}"
3. Focus on specific details, not generic praise
4. Mention cuisine style and key features
5. Include standout dishes or atmosphere if notable"""

        if analysis['media_boost'] and self.enable_media_mention:
            prompt += "\n6. Subtly indicate professional recognition if appropriate"

        prompt += "\n\nWrite ONLY the description, no extra text or quotes."

        return prompt

    def _extract_review_insights(self, reviews: List[Dict]) -> Dict[str, List[str]]:
        """Extract insights from Google reviews"""
        insights = {
            'features': [],
            'atmosphere': [],
            'dishes': []
        }

        try:
            for review in reviews[:5]:  # Analyze top 5 reviews
                text = review.get('text', '').lower()
                rating = review.get('rating', 0)

                # Skip low ratings or very short reviews
                if rating < 4 or len(text) < 30:
                    continue

                # Extract dishes mentioned
                dish_indicators = ['pasta', 'pizza', 'burger', 'steak', 'salad', 'soup', 'sandwich', 'fish', 'chicken', 'dessert', 'cocktail', 'wine']
                for indicator in dish_indicators:
                    if indicator in text and indicator not in insights['dishes']:
                        insights['dishes'].append(indicator)

                # Extract atmosphere notes
                atmosphere_terms = ['cozy', 'intimate', 'lively', 'quiet', 'romantic', 'casual', 'upscale', 'family-friendly']
                for term in atmosphere_terms:
                    if term in text and term not in insights['atmosphere']:
                        insights['atmosphere'].append(term)

                # Extract service/feature notes
                feature_terms = ['great service', 'outdoor seating', 'live music', 'craft cocktails', 'wine selection', 'fresh ingredients']
                for term in feature_terms:
                    if term in text and term not in insights['features']:
                        insights['features'].append(term.replace('great ', ''))

        except Exception as e:
            logger.warning(f"Error extracting review insights: {e}")

        return insights

    def _extract_media_insights(self, professional_sources: List[Dict], scraped_content: List[Dict]) -> Dict[str, List[str]]:
        """Extract insights from professional media coverage"""
        insights = {
            'recognition': []
        }

        try:
            # Analyze professional sources
            for source in professional_sources[:3]:
                source_type = source.get('source_type', '')
                source_name = source.get('source_name', '')

                if source_type == 'food_magazine' and source_name:
                    insights['recognition'].append(f"featured in {source_name}")
                elif source_type == 'award_guide':
                    insights['recognition'].append("award recognition")
                elif source_type == 'local_newspaper':
                    insights['recognition'].append("local favorite")
                elif source_type == 'tourism_guide':
                    insights['recognition'].append("recommended destination")

            # Could analyze scraped content here when smart scraping is implemented

        except Exception as e:
            logger.warning(f"Error extracting media insights: {e}")

        return insights

    def _infer_cuisine_type(self, name: str) -> str:
        """Infer cuisine type from restaurant name"""
        name_lower = name.lower()

        # Italian
        if any(word in name_lower for word in ['pizza', 'pasta', 'trattoria', 'osteria', 'ristorante', 'italian']):
            return "Italian"
        # French
        elif any(word in name_lower for word in ['bistro', 'brasserie', 'café', 'chez', 'french']):
            return "French"
        # Asian
        elif any(word in name_lower for word in ['sushi', 'ramen', 'thai', 'chinese', 'vietnamese', 'asian']):
            return "Asian"
        # American
        elif any(word in name_lower for word in ['grill', 'steakhouse', 'tavern', 'gastropub', 'american']):
            return "American"
        # Mediterranean
        elif any(word in name_lower for word in ['mediterranean', 'greek', 'mezze']):
            return "Mediterranean"
        # Mexican
        elif any(word in name_lower for word in ['mexican', 'taco', 'cantina']):
            return "Mexican"
        # Seafood
        elif any(word in name_lower for word in ['seafood', 'fish', 'oyster', 'crab']):
            return "Seafood"
        else:
            return "Contemporary"

    def _clean_description(self, description: str) -> str:
        """Clean and format the generated description"""
        try:
            # Remove quotes
            description = description.strip('"\'')

            # Ensure proper punctuation
            if description and not description.endswith(('.', '!', '?')):
                description += '.'

            # Remove duplicate periods
            description = description.replace('..', '.')

            # Capitalize first letter
            if description:
                description = description[0].upper() + description[1:]

            return description

        except Exception as e:
            logger.warning(f"Error cleaning description: {e}")
            return description

    def format_final_results(
        self,
        descriptions: List[RestaurantDescription],
        user_coordinates: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Format descriptions for final user display"""
        try:
            if not descriptions:
                return {
                    "success": False,
                    "message": "No restaurant descriptions available.",
                    "count": 0
                }

            # Create formatted message
            message_parts = [f"Found {len(descriptions)} excellent restaurants:"]

            for i, desc in enumerate(descriptions, 1):
                # Format distance
                distance_str = f"{desc.distance_km:.1f}km" if desc.distance_km > 0 else ""

                # Format rating
                rating_str = f"({desc.google_rating:.1f}★)" if desc.google_rating else ""

                # Format media coverage indicator
                media_indicator = " 📰" if desc.has_media_coverage else ""

                restaurant_line = f"\n\n{i}. **{desc.name}**{media_indicator}\n"
                restaurant_line += f"{desc.address}"

                if distance_str or rating_str:
                    restaurant_line += f" • {distance_str}"
                    if rating_str:
                        restaurant_line += f" • {rating_str}"

                restaurant_line += f"\n{desc.description}"

                message_parts.append(restaurant_line)

            formatted_message = "".join(message_parts)

            return {
                "success": True,
                "message": formatted_message,
                "count": len(descriptions),
                "has_media_coverage": any(desc.has_media_coverage for desc in descriptions)
            }

        except Exception as e:
            logger.error(f"Error formatting final results: {e}")
            return {
                "success": False,
                "message": "Error formatting results.",
                "count": 0
            }

    def get_editor_stats(self) -> Dict[str, Any]:
        """Get editor configuration statistics"""
        return {
            'ai_model': self.openai_model,
            'description_temperature': self.description_temperature,
            'max_description_length': self.max_description_length,
            'enable_media_mention': self.enable_media_mention
        }