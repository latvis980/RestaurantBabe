# location/location_text_editor.py
"""
Location Text Editor Agent

Creates professional restaurant descriptions by combining:
- Google Reviews analysis
- Professional media content (when available)
- AI-powered description generation in food editor style

Final output format: name, address, distance, professional description
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from location.enhanced_media_verification import EnhancedVenueData

logger = logging.getLogger(__name__)

@dataclass
class RestaurantResult:
    """Final formatted restaurant result"""
    name: str
    address: str
    distance_km: float
    description: str
    has_media_coverage: bool = False
    media_sources: Optional[List[str]] = None

class LocationTextEditor:
    """
    Creates professional restaurant descriptions combining Google reviews 
    and professional media coverage in food editor style
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for description generation
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,  # Slightly higher for more creative descriptions
            api_key=config.OPENAI_API_KEY
        )

        # Setup prompts
        self._setup_prompts()

        logger.info("✅ Location Text Editor initialized")

    def _setup_prompts(self):
        """Setup AI prompts for professional description generation"""

        # Professional description prompt - using simple string format
        self.description_prompt = """You are a professional food editor writing restaurant descriptions for a quality dining guide.

Create one concise, professional description that synthesizes all available information without mentioning sources.

STYLE GUIDELINES:
- Write 2-3 concise, descriptive sentences maximum
- Focus on specific details: cuisine style, standout dishes, atmosphere, unique features
- Avoid generic phrases like "great place!" or "amazing food!"
- Include specific menu items or specialties when available
- Mention atmosphere, service style, or what makes this restaurant special
- DO NOT mention "reviews," "Google," "sources," or "coverage"
- Write as if you personally visited and are recommending it

TONE: 
- Professional but warm and inviting
- Specific and informative
- Authoritative food editor/critic style
- Confident recommendations

MEDIA INTEGRATION:
- If professional media sources are available, subtly integrate insights without attribution
- Use phrases like "known for," "celebrated for," "standout," "notable"
- Keep it natural - as if this is your professional assessment

Restaurant: {name}
Location: {address}

Available Information:
{combined_information}

Write one concise professional description (2-3 sentences maximum):"""

    async def create_professional_descriptions(
        self, 
        venues: List[EnhancedVenueData]
    ) -> List[RestaurantResult]:
        """
        Create professional descriptions for all venues
        """
        try:
            logger.info(f"📝 Creating professional descriptions for {len(venues)} restaurants")

            results = []

            for venue in venues:
                try:
                    description = await self._generate_description(venue)

                    # Create final result
                    result = RestaurantResult(
                        name=venue.name,
                        address=venue.address,
                        distance_km=venue.distance_km,
                        description=description,
                        has_media_coverage=venue.has_professional_coverage,
                        media_sources=[s.get('title', 'Unknown') for s in venue.professional_sources] if venue.professional_sources else None
                    )

                    results.append(result)

                except Exception as e:
                    logger.warning(f"Error generating description for {venue.name}: {e}")
                    # Fallback simple description
                    result = RestaurantResult(
                        name=venue.name,
                        address=venue.address,
                        distance_km=venue.distance_km,
                        description="Restaurant serving quality cuisine in the area.",
                        has_media_coverage=False,
                        media_sources=None
                    )
                    results.append(result)

            logger.info(f"✅ Generated {len(results)} professional descriptions")
            return results

        except Exception as e:
            logger.error(f"❌ Error in description generation: {e}")
            return []

    async def _generate_description(self, venue: EnhancedVenueData) -> str:
        """
        Generate a professional description for a single venue
        """
        try:
            # Combine all available information into one coherent summary
            combined_information = self._synthesize_all_information(venue)

            # Create prompt variables
            prompt_vars = {
                'name': venue.name,
                'address': venue.address,
                'combined_information': combined_information
            }

            # Generate description using the AI model with formatted prompt
            formatted_prompt = self.description_prompt.format(**prompt_vars)

            response = await self.ai_model.ainvoke([HumanMessage(content=formatted_prompt)])

            # Handle different response types safely
            if hasattr(response, 'content') and response.content:
                description = str(response.content).strip()
            else:
                description = str(response).strip()

            # Ensure we have a valid description
            if not description or description.lower().startswith('error'):
                description = "Quality restaurant featuring carefully prepared cuisine."

            # Clean up any formatting issues
            description = self._clean_description(description)

            logger.debug(f"Generated description for {venue.name}: {description[:50]}...")
            return description

        except Exception as e:
            logger.error(f"Error generating description for {venue.name}: {e}")
            return "Quality restaurant featuring carefully prepared cuisine."

    def _synthesize_all_information(self, venue: EnhancedVenueData) -> str:
        """
        Synthesize all available information into one coherent summary for the AI
        """
        try:
            info_parts = []

            # Basic restaurant info
            info_parts.append(f"Restaurant Type: {self._infer_cuisine_type(venue)}")

            # Extract key details from Google reviews
            review_insights = self._extract_review_insights(venue.google_reviews)
            if review_insights:
                info_parts.append(f"Notable Features: {review_insights}")

            # Add professional media insights if available
            if venue.has_professional_coverage and venue.scraped_content:
                media_insights = self._extract_media_insights(venue.scraped_content)
                if media_insights:
                    info_parts.append(f"Professional Recognition: {media_insights}")

            # Rating context (without mentioning source)
            if venue.rating and venue.rating >= 4.5:
                info_parts.append("Highly rated establishment")

            return "\n".join(info_parts)

        except Exception as e:
            logger.warning(f"Error synthesizing information for {venue.name}: {e}")
            return "Quality dining establishment"

    def _extract_review_insights(self, reviews: List[Dict]) -> str:
        """
        Extract key insights from all review sources without mentioning the sources
        """
        try:
            if not reviews:
                return ""

            # Extract key details from reviews
            mentioned_dishes = set()
            atmosphere_notes = set()
            service_notes = set()

            for review in reviews[:5]:  # Analyze up to 5 reviews
                text = review.get('text', '').lower()
                rating = review.get('rating', 0)

                if rating < 4 or len(text) < 30:  # Skip low ratings or very short reviews
                    continue

                # Extract mentioned dishes/drinks (expanded list)
                food_keywords = [
                    # Italian
                    'pasta', 'pizza', 'risotto', 'gnocchi', 'tiramisu', 'osso buco',
                    'carbonara', 'bolognese', 'parmigiana', 'bruschetta',
                    # French
                    'coq au vin', 'bouillabaisse', 'ratatouille', 'cassoulet', 'escargot',
                    # Drinks
                    'wine', 'cocktail', 'negroni', 'aperol', 'prosecco', 'chianti',
                    'martini', 'manhattan', 'old fashioned',
                    # General
                    'steak', 'salmon', 'chicken', 'burger', 'sandwich',
                    'dessert', 'appetizer', 'salad', 'soup', 'seafood',
                    # Cooking styles
                    'grilled', 'roasted', 'braised', 'sautéed', 'wood-fired'
                ]

                for keyword in food_keywords:
                    if keyword in text:
                        mentioned_dishes.add(keyword)

                # Extract atmosphere/ambiance notes
                if any(word in text for word in ['cozy', 'intimate', 'romantic', 'charming']):
                    atmosphere_notes.add('intimate atmosphere')
                if any(word in text for word in ['lively', 'vibrant', 'energetic', 'bustling']):
                    atmosphere_notes.add('vibrant atmosphere')
                if any(word in text for word in ['elegant', 'upscale', 'sophisticated', 'refined']):
                    atmosphere_notes.add('refined setting')
                if any(word in text for word in ['casual', 'relaxed', 'laid-back', 'comfortable']):
                    atmosphere_notes.add('relaxed setting')

                # Extract service notes
                if any(word in text for word in ['excellent service', 'attentive', 'friendly staff', 'professional']):
                    service_notes.add('attentive service')
                if any(word in text for word in ['knowledgeable', 'helpful', 'accommodating']):
                    service_notes.add('knowledgeable staff')

            # Build insights summary
            insights = []

            if mentioned_dishes:
                unique_dishes = sorted(list(mentioned_dishes))[:4]  # Top 4 most mentioned
                if len(unique_dishes) > 2:
                    insights.append(f"specializes in {', '.join(unique_dishes[:2])} and {unique_dishes[2]}")
                elif len(unique_dishes) == 2:
                    insights.append(f"known for {' and '.join(unique_dishes)}")
                elif len(unique_dishes) == 1:
                    insights.append(f"noted for {unique_dishes[0]}")

            if atmosphere_notes:
                insights.append(list(atmosphere_notes)[0])  # Take first atmosphere note

            if service_notes and len(insights) < 2:
                insights.append(list(service_notes)[0])  # Add service note if we have space

            return "; ".join(insights) if insights else ""

        except Exception as e:
            logger.warning(f"Error extracting review insights: {e}")
            return ""

    def _extract_media_insights(self, scraped_content: List[Dict]) -> str:
        """
        Extract insights from professional media coverage without attribution
        """
        try:
            if not scraped_content:
                return ""

            insights = []

            for content in scraped_content[:2]:  # Use top 2 sources
                source_type = content.get('source_type', 'media')

                # Create insights based on source type without mentioning the source
                if source_type == 'food_magazine':
                    insights.append("recognized for culinary excellence")
                elif source_type == 'local_newspaper':
                    insights.append("celebrated by local food scene")
                elif source_type == 'tourism_guide':
                    insights.append("featured destination for food enthusiasts")
                elif source_type == 'food_critic':
                    insights.append("acclaimed by culinary experts")
                else:
                    insights.append("notable dining destination")

            return "; ".join(insights) if insights else ""

        except Exception as e:
            logger.warning(f"Error extracting media insights: {e}")
            return ""

    def _infer_cuisine_type(self, venue: EnhancedVenueData) -> str:
        """
        Infer cuisine type from venue name, address, and available information
        """
        try:
            name_lower = venue.name.lower()
            address_lower = venue.address.lower()

            # Italian indicators
            if any(word in name_lower for word in ['pizza', 'pasta', 'trattoria', 'osteria', 'ristorante']):
                return "Italian cuisine"

            # French indicators  
            if any(word in name_lower for word in ['bistro', 'brasserie', 'café', 'chez']):
                return "French cuisine"

            # Asian indicators
            if any(word in name_lower for word in ['sushi', 'ramen', 'thai', 'chinese', 'vietnamese']):
                return "Asian cuisine"

            # American/Contemporary
            if any(word in name_lower for word in ['grill', 'steakhouse', 'tavern', 'gastropub']):
                return "Contemporary American cuisine"

            # Mediterranean
            if any(word in name_lower for word in ['mediterranean', 'greek', 'mezze']):
                return "Mediterranean cuisine"

            # Default
            return "Contemporary cuisine"

        except Exception:
            return "Restaurant cuisine"

    def _clean_description(self, description: str) -> str:
        """
        Clean up the generated description
        """
        try:
            # Remove any quotes that AI might add
            description = description.strip('"\'')

            # Ensure it ends with proper punctuation
            if description and not description.endswith(('.', '!', '?')):
                description += '.'

            # Remove any duplicate periods
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
        results: List[RestaurantResult],
        user_coordinates: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Format final results for user display
        """
        try:
            if not results:
                return {
                    "success": False,
                    "message": "No restaurants found matching your criteria."
                }

            # Sort by distance
            results.sort(key=lambda x: x.distance_km)

            # Format for display
            formatted_restaurants = []

            for result in results:
                formatted_restaurants.append({
                    'name': result.name,
                    'address': result.address,
                    'distance': f"{result.distance_km:.1f}km away",
                    'description': result.description,
                    'has_media_coverage': result.has_media_coverage
                })

            # Create user message
            if len(results) == 1:
                message = "Found an excellent restaurant recommendation:"
            else:
                message = f"Found {len(results)} excellent restaurant recommendations:"

            return {
                "success": True,
                "message": message,
                "restaurants": formatted_restaurants,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Error formatting final results: {e}")
            return {
                "success": False,
                "message": "Error formatting restaurant recommendations."
            }