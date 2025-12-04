# location/location_map_search_ai_editor.py
"""
AI Editor for MAP SEARCH RESULTS specifically - ALL TYPE ERRORS FIXED

Features:
- Processes restaurants from Google Maps search + media verification
- Sources: Google reviews + media mentions (integrated into description text)
- Includes atmospheric filtering for map search results
- Formats: name, address, link, distance, description, media sources

FIXES APPLIED:
- Fixed CombinedVenueData import (now defined locally)
- Fixed OpenAI message format (using proper typed messages)
- Fixed all f-string without placeholders
- Fixed unused exception variables
- Fixed set operations with unhashable dict types
- Fixed return type annotations
- Fixed mutable default arguments using field(default_factory=list)
- Removed all set operations with dictionaries (not hashable)
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from langsmith import traceable

from formatters.google_links import build_google_maps_url
from location.location_data_logger import LocationDataLogger

from location.location_review_logger import (
    log_google_reviews_to_langsmith,
    build_review_context_with_logging,
    log_reviews_before_ai_processing
)

logger = logging.getLogger(__name__)

# FIXED: Define CombinedVenueData locally with proper field defaults
@dataclass
class CombinedVenueData:
    """Combined venue data structure for processing"""
    index: int
    name: str
    address: str
    rating: Optional[float] = None
    user_ratings_total: int = 0
    distance_km: float = 0.0
    maps_link: str = ""
    place_id: str = ""
    cuisine_tags: List[str] = field(default_factory=list)  # FIXED: Use field(default_factory=list)
    description: str = ""
    has_media_coverage: bool = False
    media_publications: List[str] = field(default_factory=list)  # FIXED: Use field(default_factory=list)
    media_articles: List[Dict] = field(default_factory=list)  # FIXED: Use field(default_factory=list)
    review_context: str = ""
    source: str = ""
    selection_score: Optional[float] = None
    selection_reason: str = ""

@dataclass
class MapSearchRestaurantDescription:
    """Map search restaurant description with media integration"""
    name: str
    address: str
    google_maps_url: str
    place_id: str
    distance_km: float
    description: str
    media_sources: List[str]  # Media publications for reference
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    selection_score: Optional[float] = None
    selection_reason: Optional[str] = None

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

    @traceable(
        run_type="chain",
        name="map_search_description_generation",
        metadata={"component": "ai_editor", "editor_type": "map_search"}
    )
    async def create_descriptions_for_map_search_results(
        self,
        map_search_results: List[Any],
        media_verification_results: Optional[List[Any]] = None,  
        user_query: str = "",
        cancel_check_fn=None,
        # NEW: Supervisor instructions for context-aware filtering/descriptions
        supervisor_instructions: Optional[str] = None
    ) -> List[MapSearchRestaurantDescription]:
        """
        Create descriptions for MAP SEARCH RESULTS

        Features:
        - Sources: Google reviews + media verification results
        - Media mentions: Integrated directly into description text (e.g., "featured in The Guardian")
        - Includes atmospheric filtering for quality control
        - NEW: Supervisor instructions for follow-up context (e.g., "user wants lunch not brunch")
        """
        try:
            logger.info(f"Creating descriptions for {len(map_search_results)} MAP SEARCH venues")

            # NEW: Log supervisor instructions if present
            if supervisor_instructions:
                logger.info(f"📋 Supervisor instructions: {supervisor_instructions[:100]}...")

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
            selected_venues = await self._filter_atmospheric_restaurants(
                combined_venues, 
                user_query,
                supervisor_instructions  # NEW: Pass supervisor instructions
            )

            self.data_logger.log_ai_selection_data(
                venues_before_selection=combined_venues,
                venues_after_selection=selected_venues,
                user_query=user_query
            )
            log_reviews_before_ai_processing(selected_venues)

            if cancel_check_fn and cancel_check_fn():
                return []

            logger.info(f"Step 2: Selected {len(selected_venues)} truly atmospheric restaurants")

            # Step 3: Generate descriptions for MAP SEARCH results
            logger.info("Step 3: Generating descriptions for MAP SEARCH results")
            descriptions = await self._generate_map_search_descriptions(
                selected_venues, 
                user_query,
                supervisor_instructions  # NEW: Pass supervisor instructions
            )

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

    @traceable(
        run_type="tool",
        name="combine_search_results", 
        metadata={"component": "ai_editor", "step": "data_combination"}
    )
    def _combine_search_results(
        self, 
        map_search_results: List[Any], 
        media_verification_results: Optional[List[Any]] = None
    ) -> List[CombinedVenueData]:
        """
        FIXED: Combine Google Maps results with media verification data

        Main fix: Proper field mapping from VenueSearchResult to CombinedVenueData
        The issue was that VenueSearchResult objects have specific field names that 
        weren't being mapped correctly.
        """
        combined_venues = []
        media_results = media_verification_results or []

        # FIXED: Create a lookup for media results by venue name (using dict, not set)
        media_lookup: Dict[str, Any] = {}
        for media_result in media_results:
            # FIXED: Handle different possible name fields in media results
            venue_name = None
            if hasattr(media_result, 'venue_name'):
                venue_name = media_result.venue_name
            elif hasattr(media_result, 'restaurant_name'):
                venue_name = media_result.restaurant_name
            elif hasattr(media_result, 'name'):
                venue_name = media_result.name

            if venue_name:
                media_lookup[venue_name.lower()] = media_result

        for i, map_result in enumerate(map_search_results):
            try:
                # FIXED: Proper field mapping from VenueSearchResult to CombinedVenueData
                # Debug: Log what we're getting from map search
                logger.debug(f"🔍 Processing map result {i+1}: {type(map_result)}")

                # The VenueSearchResult class has these exact fields (from your location_map_search.py):
                # place_id, name, address, latitude, longitude, distance_km, business_status,
                # rating, user_ratings_total, google_reviews, search_source, google_maps_url

                name = getattr(map_result, 'name', 'Unknown Restaurant')
                address = getattr(map_result, 'address', 'Address not available')
                rating = getattr(map_result, 'rating', None)
                user_ratings_total = getattr(map_result, 'user_ratings_total', 0) or 0
                distance_km = getattr(map_result, 'distance_km', 0.0) or 0.0
                place_id = getattr(map_result, 'place_id', '')

                # Get canonical Google Maps URL
                maps_link = build_google_maps_url(place_id, name) if place_id else getattr(map_result, 'google_maps_url', '')

                # Extract review context - USE ALL 5 REVIEWS with MORE text
                google_reviews = getattr(map_result, 'google_reviews', []) or []
                if google_reviews:
                    log_google_reviews_to_langsmith(name, google_reviews, max_reviews=5)
                    # Use all 5 reviews with 400 chars each for better context
                    review_context = build_review_context_with_logging(name, google_reviews, max_reviews=5)
                else:
                    review_context = ""

                # Find matching media verification
                media_match = media_lookup.get(name.lower())

                # FIXED: Create combined venue data with CORRECT field mapping
                venue_data = CombinedVenueData(
                    index=i + 1,
                    name=name,
                    address=address,
                    rating=rating,
                    user_ratings_total=user_ratings_total,
                    distance_km=distance_km,
                    maps_link=maps_link,
                    place_id=place_id,
                    cuisine_tags=getattr(map_result, 'cuisine_tags', []) or [],
                    description=getattr(map_result, 'description', '') or '',
                    has_media_coverage=bool(media_match and hasattr(media_match, 'has_professional_coverage') and media_match.has_professional_coverage),
                    media_publications=getattr(media_match, 'media_publications', []) if media_match else [],
                    media_articles=getattr(media_match, 'professional_sources', []) if media_match else [],
                    review_context=review_context,
                    source='map_search'
                )

                combined_venues.append(venue_data)

                # DEBUG: Log successful mapping
                logger.info(f"✅ DEBUG - Mapped venue {i+1}: '{name}' at '{address}' ({distance_km}km)")

            except Exception as e:
                logger.error(f"❌ Error combining search results for venue {i+1}: {e}")

                # DEBUG: Log the actual structure we received
                if hasattr(map_result, '__dict__'):
                    logger.error(f"   Map result fields: {list(map_result.__dict__.keys())}")
                    logger.error(f"   Map result values: {map_result.__dict__}")
                else:
                    logger.error(f"   Map result type: {type(map_result)}")
                    logger.error(f"   Map result dir: {dir(map_result)}")
                continue

        logger.info(f"✅ Successfully combined {len(combined_venues)} venues from {len(map_search_results)} map results")
        return combined_venues

    @traceable(
        run_type="llm",
        name="atmospheric_filtering",
        metadata={"component": "ai_editor", "step": "venue_filtering"}
    )
    async def _filter_atmospheric_restaurants(
        self, 
        venues: List[CombinedVenueData], 
        user_query: str,
        supervisor_instructions: Optional[str] = None  # NEW
    ) -> List[CombinedVenueData]:
        """AI-powered filtering to select truly atmospheric restaurants from map search

        NEW: Uses supervisor_instructions for context-aware filtering
        (e.g., "user wants lunch not brunch", "prefer closer options")
        """
        if not venues:
            return []

        try:
            # Prepare venue data for AI evaluation
            venues_text = self._format_venues_for_filtering(venues)

            # Call AI for atmospheric filtering with supervisor context
            selection_response = await self._call_atmospheric_filtering_ai(
                venues_text, 
                user_query,
                supervisor_instructions  # NEW
            )

            # Parse response and filter venues
            selected_venues = self._parse_atmospheric_filtering_response(selection_response, venues)

            return selected_venues

        except Exception as e:
            logger.error(f"Error in atmospheric filtering: {e}")
            # Return original venues if filtering fails
            return venues

    def _format_venues_for_filtering(self, venues: List[CombinedVenueData]) -> str:
        """Format venue data for atmospheric filtering AI"""
        formatted = ""

        for venue in venues:
            formatted += f"\n{'='*60}\n"
            formatted += f"RESTAURANT {venue.index}: {venue.name}\n"
            formatted += f"RATING: {venue.rating or 'N/A'}★ ({venue.user_ratings_total} reviews)\n"
            formatted += f"DISTANCE: {venue.distance_km:.1f}km\n"
            formatted += f"CUISINE: {', '.join(venue.cuisine_tags) if venue.cuisine_tags else 'Not specified'}\n"

            if venue.has_media_coverage:
                formatted += f"MEDIA: Featured in {', '.join(venue.media_publications)}\n"

            if venue.review_context:
                formatted += f"REVIEWS: {venue.review_context[:300]}...\n"

            if venue.description:
                formatted += f"DESCRIPTION: {venue.description[:200]}...\n"

        return formatted

    @traceable(
        run_type="llm",
        name="atmospheric_filtering_llm_call",
        metadata={"component": "ai_editor", "step": "venue_filtering_llm"},
    )
    async def _call_atmospheric_filtering_ai(
        self, 
        venues_text: str, 
        user_query: str,
        supervisor_instructions: Optional[str] = None  # NEW
    ) -> str:
        """Call AI to filter for atmospheric restaurants

        NEW: Uses supervisor_instructions for context-aware filtering
        """

        # NEW: Build supervisor context section if instructions provided
        supervisor_context = ""
        if supervisor_instructions:
            supervisor_context = f"""
    IMPORTANT - SUPERVISOR INSTRUCTIONS (from conversation analysis):
    {supervisor_instructions}

    Apply these instructions when selecting restaurants. They represent what the user 
    actually wants based on conversation context.
    """

        prompt = f"""You are selecting restaurants that offer exceptional experiences that match this query: "{user_query}"
    {supervisor_context}
    {venues_text}

    SELECTION CRITERIA:
    - Unique atmosphere, character, or ambiance
    - Special dining experiences beyond basic food service
    - If the query mentions a specific type or cuisine or dish, make sure the restaurant offers those
    - IMPORTANT: Follow any supervisor instructions above (they reflect user's actual intent)

    AVOID:
    - Generic chain restaurants
    - Basic fast food or casual dining without character
    - Restaurants that don't match supervisor instructions (if provided)

    OUTPUT FORMAT (JSON):
    {{
    "selected_restaurants": [
        {{
            "index": 1,
            "selection_score": 0.9,
            "reasoning": "Why this restaurant is special and matches the query..."
        }}
    ]
    }}

    Select restaurants that would create memorable experiences, not just satisfy hunger."""

        # Import typing utilities at the start
        from typing import cast, Any
        
        try:
            # FIXED: Import and use proper OpenAI types for type safety
            try:
                from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam

                # FIXED: Use proper typed message format
                messages = [
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="You are an expert at identifying truly special and high-quality restaurants"
                    ),
                    ChatCompletionUserMessageParam(
                        role="user", 
                        content=prompt
                    )
                ]
            except ImportError:
                # Fallback for older OpenAI versions
                messages = cast(Any, [
                    {"role": "system", "content": "You are an expert at identifying restaurants with exceptional atmosphere and dining experiences."},
                    {"role": "user", "content": prompt}
                ])

            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.2,  # Lower temperature for consistent filtering
                max_tokens=1500
            )

            # FIXED: Handle potential None return
            return response.choices[0].message.content or "{}"

        except Exception as e:
            logger.error(f"Error calling atmospheric filtering AI: {e}")
            return "{}"

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

            # FIXED: Use list to collect selected venues, not set
            selected_venues = []

            selected_list = parsed_response.get("selected_restaurants", [])

            for selection in selected_list:
                try:
                    index = selection.get("index", 1) - 1  # Convert to 0-based

                    if 0 <= index < len(venues):
                        venue = venues[index]
                        # Add selection score to venue
                        venue.selection_score = selection.get("selection_score", 0.8)
                        venue.selection_reason = selection.get("reasoning", "")
                        selected_venues.append(venue)

                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing selection {selection}: {e}")
                    continue

            if not selected_venues:
                logger.warning("No venues selected by atmospheric filtering, returning all")
                return venues

            logger.info(f"Atmospheric filtering selected {len(selected_venues)}/{len(venues)} venues")
            return selected_venues

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing atmospheric filtering response: {e}")
            return venues

    @traceable(
        run_type="llm", 
        name="generate_descriptions",
        metadata={"component": "ai_editor", "step": "description_generation"}
    )
    async def _generate_map_search_descriptions(
        self,
        venues: List[CombinedVenueData],
        user_query: str,
        supervisor_instructions: Optional[str] = None  # NEW
    ) -> List[MapSearchRestaurantDescription]:
        """
        Generate descriptions for MAP SEARCH results with media integration
        STRICT: Only use facts from review_context, never invent details

        NEW: Uses supervisor_instructions for context-aware descriptions
        (e.g., emphasize lunch options if user said "lunch not brunch")
        """
        if not venues:
            return []

        # Create mapping of sequential indices to venues
        index_to_venue = {i: v for i, v in enumerate(venues, start=1)}

        # Create combined prompt with all restaurants for MAP SEARCH
        all_restaurants_data = []
        for idx, venue in index_to_venue.items():
            restaurant_data = {
                "index": idx,
                "name": venue.name,
                "rating": venue.rating or "N/A",
                "user_ratings_total": venue.user_ratings_total or 0,
                "distance_km": venue.distance_km,
                "has_media_coverage": venue.has_media_coverage,
                "media_publications": venue.media_publications,
                "review_context": venue.review_context,  # THIS IS THE ONLY SOURCE OF TRUTH
            }
            all_restaurants_data.append(restaurant_data)

        # Generate description for all restaurants at once
        restaurants_text = json.dumps(all_restaurants_data, indent=2)

        # NEW: Build supervisor context for descriptions
        supervisor_context = ""
        if supervisor_instructions:
            supervisor_context = f"""
        SUPERVISOR INSTRUCTIONS (apply these when writing descriptions):
        {supervisor_instructions}

        Use these instructions to emphasize relevant aspects. For example, if user wants 
        "lunch not brunch", focus on lunch-related details from reviews.
        """

        # STRICT instruction - no invention allowed
        strict_instruction = f"""CRITICAL RULES - READ CAREFULLY:

            1. ONLY use information from the "review_context" field for each restaurant
            2. If review_context is empty or has no details, write: "No detailed reviews available yet."
            3. Mention specific dishes, ingredients, or menu items explicitly stated in review_context
            4. Mention atmosphere details (cozy, intimate, spacious, etc.) mentioned in review_context
            5. Mention price range, service quality, or ambiance from review_context
            {supervisor_context}
            WHAT TO DO:
            - Paraphrase what real customers said in their reviews
            - Give preferrence to concrete details rather than generic statements
            - Keep it short and factual
            - If supervisor instructions mention specific preferences, highlight those aspects"""

        # Import typing utilities
        from typing import cast, Any

        try:
            from openai.types.chat import (
                ChatCompletionUserMessageParam,
                ChatCompletionSystemMessageParam,
            )

            messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"""You are a restaurant description writer who ONLY uses verified information from customer reviews.

    {strict_instruction}

    Your descriptions MUST be:
    - Based ONLY on review_context content
    - Short (1-2 sentences maximum)
    - Factual, not creative or imaginative
    - Honest about lack of detail if reviews are sparse

    Return ONLY a JSON array:
    [
    {{
    "index": 1,
    "description": "Brief factual description based only on review_context",
    "selection_score": 0.9
    }}
    ]"""
                    ),
                    ChatCompletionUserMessageParam(
                        role="user", 
                        content=f"""Write descriptions for these restaurants based ONLY on their review_context.

    User query: "{user_query}"

    Restaurants data:
    {restaurants_text}

    Remember: ONLY use facts from review_context. If a restaurant has little/no review detail, say so."""
                    )
                ]
        except ImportError:
            messages = cast(Any, [
                {"role": "system", "content": f"""You are a restaurant description writer who ONLY uses verified information from customer reviews.

    {strict_instruction}

    Your descriptions MUST be:
    - Based ONLY on review_context content  
    - Short (1-2 sentences maximum)
    - Factual, not creative or imaginative
    - Honest about lack of detail if reviews are sparse

    Return ONLY a JSON array:
    [
    {{
    "index": 1,
    "description": "Brief factual description based only on review_context",
    "selection_score": 0.9
    }}
    ]"""},
                {"role": "user", "content": f"""Write descriptions for these restaurants based ONLY on their review_context.

    User query: "{user_query}"

    Restaurants data:
    {restaurants_text}

    Remember: ONLY use facts from review_context. If a restaurant has little/no review detail, say so."""}
            ])

        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0.1,  # LOWERED from 0.3 - less creative = less hallucination
            max_tokens=2000
        )

        # Rest of the method stays the same...
        response_content = response.choices[0].message.content
        if not response_content:
            logger.warning("Empty response from AI description generation")
            return []

        # Parse AI response
        response_content = response_content.strip()
        if response_content.startswith("```"):
            parts = response_content.split("```")
            if len(parts) >= 3:
                response_content = parts[1]
            response_content = response_content.lstrip()
            if response_content.lower().startswith("json"):
                response_content = response_content.split("\n", 1)[1] if "\n" in response_content else ""
            response_content = response_content.strip()

        try:
            descriptions_data = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse AI description JSON: %s. Raw content: %s",
                e,
                response_content[:500],
            )
            return []

        # Create MapSearchRestaurantDescription objects
        descriptions = []
        for desc_data in descriptions_data:
            try:
                index = desc_data.get('index')
                venue = index_to_venue.get(index)
                if not venue:
                    logger.warning(f"Invalid or missing venue index: {index}")
                    continue

                description = MapSearchRestaurantDescription(
                    name=venue.name,
                    address=venue.address,
                    google_maps_url=venue.maps_link,
                    place_id=venue.place_id,
                    distance_km=venue.distance_km,
                    description=desc_data['description'],
                    media_sources=venue.media_publications,
                    rating=venue.rating,
                    user_ratings_total=venue.user_ratings_total,
                    selection_score=desc_data.get('selection_score', 0.8),
                    selection_reason=venue.selection_reason
                )
                descriptions.append(description)

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Error generating map search descriptions: {e}")

        return descriptions

    # DEPRECATED methods - redirect to maintain compatibility
    async def create_descriptions(self, *args, **kwargs):
        """DEPRECATED: Use create_descriptions_for_map_search_results instead"""
        logger.warning("create_descriptions is deprecated. Use create_descriptions_for_map_search_results")
        return await self.create_descriptions_for_map_search_results(*args, **kwargs)