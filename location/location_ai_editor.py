# location/location_ai_editor.py
"""
Location AI Description Editor - IMPROVED VERSION

Removes all hardcoded methods and implements fully AI-driven analysis.
Creates professional restaurant descriptions by combining results from:
- LocationMapSearchAgent (Google reviews, ratings, basic venue data)
- LocationMediaVerificationAgent (professional media coverage, scraped content)

Key improvements:
- Removed all hardcoded dish_indicators, atmosphere_terms, feature_terms
- Added AI-driven restaurant selection filtering
- Both review context and media context are fully utilized by AI
- No character limits (messages won't be cut off)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from openai import AsyncOpenAI

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
    selection_score: Optional[float] = None  # NEW: For ranking restaurants

    def __post_init__(self):
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
    """

    def __init__(self, config):
        self.config = config

        # Configuration
        self.openai_model = getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.description_temperature = getattr(config, 'DESCRIPTION_TEMPERATURE', 0.3)
        self.enable_media_mention = getattr(config, 'ENABLE_MEDIA_MENTION', True)
        # REMOVED: max_description_length - no more character limits

        # Initialize AsyncOpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY')
        )

        logger.info("Location AI Editor initialized with improved AI-driven analysis")

    async def create_professional_descriptions(
        self,
        map_search_results: List[Any],
        media_verification_results: List[Any] = None,
        user_query: str = "",
        cancel_check_fn=None
    ) -> List[RestaurantDescription]:
        """
        Main method: Create professional descriptions with intelligent restaurant selection

        Steps:
        1. Combine data from map search and media verification
        2. AI-powered restaurant filtering (select truly atmospheric, special places)
        3. Generate professional descriptions for selected restaurants
        """
        try:
            logger.info(f"Creating professional descriptions for {len(map_search_results)} venues")

            if not map_search_results:
                return []

            # Step 1: Combine data from both agents
            combined_venues = self._combine_search_results(map_search_results, media_verification_results)

            if cancel_check_fn and cancel_check_fn():
                return []

            # Step 2: AI-powered restaurant selection filtering
            logger.info("Step 2: AI filtering for truly atmospheric restaurants")
            selected_venues = await self._filter_atmospheric_restaurants(combined_venues, user_query)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Step 2: Selected {len(selected_venues)} truly atmospheric restaurants")

            # Step 3: Generate descriptions using AI
            descriptions = []
            for venue in selected_venues:
                if cancel_check_fn and cancel_check_fn():
                    break

                try:
                    description_text = await self._generate_venue_description(venue, user_query)

                    restaurant_desc = RestaurantDescription(
                        place_id=venue.place_id,
                        name=venue.name,
                        address=venue.address,
                        distance_km=venue.distance_km,
                        description=description_text,
                        has_media_coverage=venue.has_professional_coverage,
                        media_sources=[s.get('source_name', '') for s in venue.professional_sources[:3]],
                        google_rating=venue.rating,
                        selection_score=getattr(venue, 'selection_score', None)
                    )

                    descriptions.append(restaurant_desc)

                except Exception as e:
                    logger.error(f"Error creating description for {venue.name}: {e}")
                    continue

            logger.info(f"Generated {len(descriptions)} professional descriptions")
            return descriptions

        except Exception as e:
            logger.error(f"Error in create_professional_descriptions: {e}")
            return []

    def _combine_search_results(
        self,
        map_search_results: List[Any],
        media_verification_results: List[Any] = None
    ) -> List[CombinedVenueData]:
        """Combine map search and media verification results into unified objects"""
        try:
            combined_venues = []
            media_lookup = {}

            # Create lookup for media results
            if media_verification_results:
                for media_result in media_verification_results:
                    venue_id = getattr(media_result, 'venue_id', None)
                    if venue_id:
                        media_lookup[venue_id] = media_result

            # Combine data
            for venue in map_search_results:
                venue_id = getattr(venue, 'place_id', None)
                media_data = media_lookup.get(venue_id)

                combined_venue = CombinedVenueData(
                    place_id=venue_id or "",
                    name=getattr(venue, 'name', ''),
                    address=getattr(venue, 'address', ''),
                    latitude=getattr(venue, 'latitude', 0.0),
                    longitude=getattr(venue, 'longitude', 0.0),
                    distance_km=getattr(venue, 'distance_km', 0.0),
                    business_status=getattr(venue, 'business_status', ''),
                    rating=getattr(venue, 'rating', None),
                    user_ratings_total=getattr(venue, 'user_ratings_total', None),
                    google_reviews=getattr(venue, 'google_reviews', []),
                    google_maps_url=getattr(venue, 'google_maps_url', ''),
                    has_professional_coverage=bool(media_data and getattr(media_data, 'has_professional_coverage', False)),
                    media_coverage_score=getattr(media_data, 'media_coverage_score', 0.0) if media_data else 0.0,
                    professional_sources=getattr(media_data, 'professional_sources', []) if media_data else [],
                    scraped_content=getattr(media_data, 'scraped_content', []) if media_data else [],
                    credibility_assessment=getattr(media_data, 'credibility_assessment', {}) if media_data else {}
                )

                combined_venues.append(combined_venue)

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
3. GOOD CONCEPT: mantions of interesting concepts, unique experiences
5. MEDIA COVERAGE BONUS: Professional coverage adds credibility

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
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            try:
                import json
                selection_result = json.loads(content)
                selected_data = selection_result.get("selected_restaurants", [])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse AI selection response: {content}")
                return venues  # Return all if parsing fails

            # Build selected venues list
            selected_venues = []
            for selection in selected_data:
                index = selection.get('index')
                if index is not None and 0 <= index < len(venues):
                    venue = venues[index]
                    venue.selection_score = selection.get('selection_score', 0.0)
                    selected_venues.append(venue)

            # Sort by selection score (highest first)
            selected_venues.sort(key=lambda v: getattr(v, 'selection_score', 0.0), reverse=True)

            logger.info(f"AI selected {len(selected_venues)} atmospheric restaurants from {len(venues)} candidates")
            return selected_venues

        except Exception as e:
            logger.error(f"Error in AI restaurant filtering: {e}")
            return venues  # Return all venues if filtering fails

    def _format_restaurants_for_selection(self, restaurants_data: List[Dict]) -> str:
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
                formatted += f"MEDIA COVERAGE: No\n"

            formatted += "\nRECENT REVIEWS:\n"
            for i, review in enumerate(restaurant['reviews'][:3], 1):
                review_text = review.get('text', '')[:300]  # Limit review length
                review_rating = review.get('rating', 'N/A')
                formatted += f"Review {i} ({review_rating}★): {review_text}...\n\n"

            if not restaurant['reviews']:
                formatted += "No reviews available\n"

        return formatted

    async def _generate_venue_description(
        self,
        venue: CombinedVenueData,
        user_query: str
    ) -> str:
        """Generate AI description using both review context and media context"""
        try:
            # Prepare context for AI
            review_context = ""
            if venue.google_reviews:
                review_context = "\nREVIEW CONTEXT:\n"
                for review in venue.google_reviews[:5]:
                    rating = review.get('rating', 'N/A')
                    text = review.get('text', '')
                    review_context += f"- ({rating}★) {text}\n"

            media_context = ""
            if venue.has_professional_coverage and venue.professional_sources:
                media_context = "\nMEDIA COVERAGE CONTEXT:\n"
                for source in venue.professional_sources[:3]:
                    source_name = source.get('source_name', 'Unknown source')
                    source_type = source.get('source_type', 'media')
                    media_context += f"- Featured in {source_name} ({source_type})\n"

            # AI description prompt - fully context-driven
            description_prompt = f"""Create a restaurant description for "{venue.name}".

USER'S QUERY: "{user_query}"
RESTAURANT RATING: {venue.rating}★ ({venue.user_ratings_total} reviews)
DISTANCE: {venue.distance_km:.1f}km

{review_context}

{media_context}

INSTRUCTIONS:
1. Write a very brief description that captures what makes this restaurant special
2. Use the review context to identify authentic details about food, atmosphere, and experience
3. If media coverage exists, subtly incorporate the professional recognition
4. Focus on specific details rather than generic praise
5. Make it relevant to the user's query: "{user_query}"

**Examples of descriptions:**

Locals' favourite with a great selection of natural wines from Europe and small plates to go with them. Just next to Estrela park.

The owner, Joao, changes the menu every day. GQ magazine wrote some good things about this place. 

Cozy, whimsical, a true hidden jem in Bairro Alto. Crafted cocktails like "Bairro negroni" and "Mango smash". 

Possibly best sourdough on this side of town and Sunday brunches with lush pastries and egg dishes. Featured in The Guardian. 


Write ONLY the description, no extra formatting or quotes."""

            # Generate description
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": description_prompt}],
                temperature=self.description_temperature,
                max_tokens=1024  # Increased for fuller descriptions
            )

            description = response.choices[0].message.content.strip()
            return description

        except Exception as e:
            logger.error(f"Error generating description for {venue.name}: {e}")
            return "Quality restaurant featuring carefully prepared cuisine."

    def format_final_results(self, descriptions: List[RestaurantDescription]) -> Dict[str, Any]:
        """Format the final results for Telegram display using HTML formatting"""
        try:
            if not descriptions:
                return {
                    "success": False,
                    "message": "No exceptional restaurants found in this area.",
                    "count": 0
                }

            # Create formatted message using HTML like the rest of the app
            message_parts = [f"<b>Found {len(descriptions)} exceptional restaurants:</b>\n\n"]

            for i, desc in enumerate(descriptions, 1):
                # Format distance and rating
                distance_str = f"{desc.distance_km:.1f}km" if desc.distance_km > 0 else ""
                rating_str = f"({desc.google_rating:.1f}★)" if desc.google_rating else ""

                # Media coverage indicator
                media_indicator = " 📰" if desc.has_media_coverage else ""

                # Selection score indicator for highly rated selections
                score_indicator = ""
                if desc.selection_score and desc.selection_score >= 8.0:
                    score_indicator = " ⭐"

                # Restaurant name with HTML bold formatting
                restaurant_line = f"<b>{i}. {self._clean_html(desc.name)}{media_indicator}{score_indicator}</b>\n"

                # Address with Google Maps link (following app pattern)
                if desc.place_id:
                    google_url = f"https://www.google.com/maps/place/?q=place_id:{desc.place_id}"
                    restaurant_line += f'📍 <a href="{google_url}">{self._clean_html(desc.address)}</a>'
                else:
                    restaurant_line += f"📍 {self._clean_html(desc.address)}"

                # Add distance and rating info
                if distance_str or rating_str:
                    info_parts = []
                    if distance_str:
                        info_parts.append(distance_str)
                    if rating_str:
                        info_parts.append(rating_str)
                    restaurant_line += f" • {' • '.join(info_parts)}"

                restaurant_line += f"\n{self._clean_html(desc.description)}\n\n"

                message_parts.append(restaurant_line)

            # Add footer note
            message_parts.append("<i>Click the address to see the venue photos and menu on Google Maps</i>")

            formatted_message = "".join(message_parts)

            return {
                "success": True,
                "message": formatted_message,
                "count": len(descriptions),
                "has_media_coverage": any(desc.has_media_coverage for desc in descriptions),
                "avg_selection_score": sum(d.selection_score or 0 for d in descriptions) / len(descriptions) if descriptions else 0
            }

        except Exception as e:
            logger.error(f"Error formatting final results: {e}")
            return {
                "success": False,
                "message": "Error formatting results.",
                "count": 0
            }

    def _clean_html(self, text: str) -> str:
        """Clean text for HTML display (matching app pattern)"""
        try:
            if not text:
                return ""

            from html import escape
            import re

            # Escape HTML characters
            cleaned = escape(str(text))

            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())

            return cleaned

        except Exception:
            return str(text) if text else ""

    def get_editor_stats(self) -> Dict[str, Any]:
        """Get editor configuration statistics"""
        return {
            'ai_model': self.openai_model,
            'description_temperature': self.description_temperature,
            'enable_media_mention': self.enable_media_mention,
            'hardcoded_methods': False,  # Now removed
            'ai_driven_selection': True,  # New feature
            'character_limits': False    # Removed limits
        }