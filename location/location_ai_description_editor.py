# location/location_ai_description_editor.py
"""
AI Description Editor for Location DATABASE Results

This service takes filtered restaurants from the database and creates AI-edited
descriptions that are contextual to the user's initial query and optimized for Telegram.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class LocationAIDescriptionEditor:
    """
    AI-powered description editor for location search results

    Creates concise, query-relevant descriptions for Telegram display
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI for description editing
        self.ai_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,  # Slightly creative but consistent
            api_key=config.OPENAI_API_KEY,
            max_tokens=1024
        )

        # Description editing prompt
        self.description_prompt = ChatPromptTemplate.from_template("""
You are creating concise, engaging descriptions for restaurants near a user's location.

USER'S ORIGINAL REQUEST: "{user_query}"
LOCATION: {location_description}

RESTAURANT DATA:
{restaurants_data}

TASK: Create brief, telegram-friendly descriptions (15-25 words each) that:

1. RELATE TO USER'S QUERY: Highlight aspects that match what they asked for
2. BE SPECIFIC: Use concrete details from raw_descriptions, not generic phrases  
3. STAY CONCISE: Perfect for mobile/telegram reading
4. INCLUDE APPEAL: Why this restaurant suits their request

FORMATTING GUIDELINES:
- Focus on cuisine, atmosphere, or standout features
- Mention specific dishes/specialties when available
- Include price range hints if mentioned in raw descriptions
- Don't repeat the restaurant name in the description
- Use active, engaging language

OUTPUT: Return ONLY valid JSON:
{{
    "edited_restaurants": [
        {{
            "id": "restaurant_id",
            "name": "Restaurant Name",
            "address": "Full Address",
            "distance_km": 1.2,
            "description": "Concise 15-25 word description relating to user's query",
            "sources_domains": ["domain1.com", "domain2.com"],
            "original_sources": ["full_url1", "full_url2"]
        }}
    ]
}}

Make each description unique and query-relevant. If raw description is minimal, focus on cuisine type and location relevance.
""")

        logger.info("✅ Location AI Description Editor initialized")

    def edit_descriptions(
        self, 
        filtered_restaurants: List[Dict[str, Any]], 
        user_query: str,
        location_description: str = "your location"
    ) -> List[Dict[str, Any]]:
        """
        Create AI-edited descriptions for filtered restaurants

        Args:
            filtered_restaurants: List of restaurants from filter_evaluator
            user_query: User's original search query 
            location_description: Description of the search location

        Returns:
            List of restaurants with AI-edited descriptions
        """
        try:
            if not filtered_restaurants:
                logger.warning("No restaurants to edit descriptions for")
                return []

            logger.info(f"🎨 Creating AI-edited descriptions for {len(filtered_restaurants)} restaurants")
            logger.info(f"📝 User query: '{user_query}'")

            # Prepare restaurant data for AI
            restaurants_text = self._format_restaurants_for_ai(filtered_restaurants)

            # Create the AI chain
            chain = self.description_prompt | self.ai_model

            # Get AI response
            response = chain.invoke({
                "user_query": user_query,
                "location_description": location_description,
                "restaurants_data": restaurants_text
            })

            # Parse AI response
            edited_restaurants = self._parse_ai_response(response.content, filtered_restaurants)

            logger.info(f"✅ Successfully created {len(edited_restaurants)} AI-edited descriptions")

            return edited_restaurants

        except Exception as e:
            logger.error(f"❌ Error in AI description editing: {e}")
            # Fallback: return restaurants with basic descriptions
            return self._create_fallback_descriptions(filtered_restaurants, user_query)

    def _format_restaurants_for_ai(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Format restaurant data for AI analysis

        Include all relevant information: name, cuisine, description, sources
        """
        try:
            formatted_entries = []

            for restaurant in restaurants:
                # Basic info
                restaurant_id = restaurant.get('id', 'unknown')
                name = restaurant.get('name', 'Unknown Restaurant')
                address = restaurant.get('address', 'Address not available')
                distance = restaurant.get('distance_km', 0)

                # Description data
                raw_description = restaurant.get('raw_description', '') or restaurant.get('description', '')
                cuisine_tags = restaurant.get('cuisine_tags', [])

                # Sources data
                sources_domains = restaurant.get('sources_domains', [])
                sources = restaurant.get('sources', [])

                # Format entry
                entry = f"""
ID: {restaurant_id}
NAME: {name}
ADDRESS: {address}
DISTANCE: {distance}km
CUISINE_TAGS: {', '.join(cuisine_tags) if cuisine_tags else 'Not specified'}
RAW_DESCRIPTION: {raw_description[:500] if raw_description else 'No description available'}
SOURCES: {', '.join(sources_domains) if sources_domains else 'No sources'}
""".strip()

                formatted_entries.append(entry)

            return "\n\n" + "="*50 + "\n\n".join(formatted_entries)

        except Exception as e:
            logger.error(f"Error formatting restaurants for AI: {e}")
            return "No restaurant data available"

    def _parse_ai_response(self, ai_response: str, original_restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse AI response and merge with original restaurant data
        """
        try:
            # Extract JSON from AI response
            ai_response = ai_response.strip()

            # Try to find JSON in the response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No valid JSON found in AI response")
                return self._create_fallback_descriptions(original_restaurants, "")

            json_str = ai_response[start_idx:end_idx]
            parsed_response = json.loads(json_str)

            edited_restaurants_data = parsed_response.get('edited_restaurants', [])

            if not edited_restaurants_data:
                logger.error("No edited_restaurants found in AI response")
                return self._create_fallback_descriptions(original_restaurants, "")

            # Create lookup dictionary for original restaurants
            original_lookup = {str(r.get('id')): r for r in original_restaurants}

            # Merge AI-edited descriptions with original data
            final_restaurants = []

            for edited_restaurant in edited_restaurants_data:
                restaurant_id = str(edited_restaurant.get('id', ''))

                if restaurant_id in original_lookup:
                    # Start with original restaurant data
                    final_restaurant = original_lookup[restaurant_id].copy()

                    # Override with AI-edited description
                    final_restaurant['ai_description'] = edited_restaurant.get('description', '')
                    final_restaurant['description'] = edited_restaurant.get('description', '')

                    # Ensure we have sources data
                    if 'sources_domains' not in final_restaurant:
                        final_restaurant['sources_domains'] = final_restaurant.get('sources_domains', [])
                    if 'sources' not in final_restaurant:
                        final_restaurant['sources'] = final_restaurant.get('sources', [])

                    # Add to final list
                    final_restaurants.append(final_restaurant)
                else:
                    logger.warning(f"AI returned restaurant ID {restaurant_id} not found in original data")

            logger.info(f"✅ Successfully parsed {len(final_restaurants)} restaurants with AI descriptions")
            return final_restaurants

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"AI response was: {ai_response[:200]}...")
            return self._create_fallback_descriptions(original_restaurants, "")
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._create_fallback_descriptions(original_restaurants, "")

    def _create_fallback_descriptions(self, restaurants: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """
        Create basic fallback descriptions when AI fails
        """
        try:
            fallback_restaurants = []

            for restaurant in restaurants:
                fallback_restaurant = restaurant.copy()

                # Create basic description
                name = restaurant.get('name', 'Restaurant')
                cuisine_tags = restaurant.get('cuisine_tags', [])
                distance = restaurant.get('distance_km', 0)

                if cuisine_tags:
                    cuisine_text = f"{', '.join(cuisine_tags[:2])}"
                    description = f"Popular {cuisine_text.lower()} restaurant {distance}km away"
                else:
                    description = f"Local restaurant {distance}km from your location"

                fallback_restaurant['description'] = description
                fallback_restaurant['ai_description'] = description

                # Ensure sources data is available
                if 'sources_domains' not in fallback_restaurant:
                    fallback_restaurant['sources_domains'] = []
                if 'sources' not in fallback_restaurant:
                    fallback_restaurant['sources'] = []

                fallback_restaurants.append(fallback_restaurant)

            logger.info(f"✅ Created {len(fallback_restaurants)} fallback descriptions")
            return fallback_restaurants

        except Exception as e:
            logger.error(f"Error creating fallback descriptions: {e}")
            return []

    def create_telegram_formatted_results(
        self, 
        edited_restaurants: List[Dict[str, Any]], 
        user_query: str,
        location_description: str = "your location"
    ) -> Dict[str, Any]:
        """
        Format the final results for Telegram display

        Args:
            edited_restaurants: Restaurants with AI-edited descriptions
            user_query: Original user query
            location_description: Location context

        Returns:
            Dict with formatted message and metadata
        """
        try:
            if not edited_restaurants:
                return {
                    "success": False,
                    "message": "No restaurants found matching your criteria.",
                    "restaurant_count": 0
                }

            # Create header
            header = f"🏠 From my personal restaurant notes:\n\n"

            # Format each restaurant
            restaurant_entries = []

            for i, restaurant in enumerate(edited_restaurants, 1):
                name = restaurant.get('name', 'Unknown Restaurant')
                address = restaurant.get('address', 'Address not available')
                distance = restaurant.get('distance_km', 0)
                description = restaurant.get('description', 'Quality local restaurant')
                sources_domains = restaurant.get('sources_domains', [])

                # Create entry
                entry = f"{i}. **{name}**\n"
                entry += f"📍 {address}\n"
                entry += f"🚶 {distance}km away\n"
                entry += f"💭 {description}\n"

                # Add sources if available
                if sources_domains:
                    sources_text = ', '.join(sources_domains[:3])  # Limit to 3 sources
                    if len(sources_domains) > 3:
                        sources_text += f" (+{len(sources_domains) - 3} more)"
                    entry += f"📰 Sources: {sources_text}\n"

                restaurant_entries.append(entry)

            # Combine all entries
            message = header + "\n\n".join(restaurant_entries)

            # Add footer if needed
            if len(edited_restaurants) > 1:
                message += f"\n\n💡 These {len(edited_restaurants)} restaurants match your request for {location_description}."

            return {
                "success": True,
                "message": message,
                "restaurant_count": len(edited_restaurants),
                "location_description": location_description,
                "user_query": user_query
            }

        except Exception as e:
            logger.error(f"Error formatting Telegram results: {e}")
            return {
                "success": False,
                "message": "Error formatting restaurant recommendations.",
                "restaurant_count": 0
            }