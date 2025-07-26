# agents/supabase_update_agent.py - COMPLETE FIXED VERSION
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from utils.database import get_database
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseUpdateAgent:
    """
    Agent responsible for processing ALL scraped content and updating the Supabase restaurants database.

    This agent:
    1. Extracts ALL restaurant information from scraped web content
    2. Combines duplicate restaurants mentioned across multiple sources  
    3. Extracts comprehensive cuisine tags using AI
    4. Saves ALL processed data to Supabase restaurants table (not just final recommendations)
    5. Handles geographic data when available
    """

    def __init__(self, config):
        self.config = config

        # Get the shared database instance
        self.db = get_database()
        logger.info("✅ Database connection obtained for restaurant updates")

        # Initialize OpenAI
        openai_key = getattr(config, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
        if not openai_key:
            raise ValueError("OPENAI_API_KEY must be configured")

        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=openai_key
        )
        logger.info("✅ OpenAI GPT-4o initialized for restaurant processing")

        # Restaurant extraction and processing prompt
        self.processing_prompt = ChatPromptTemplate.from_template("""
You are a restaurant data processor. Your task is to extract and organize ALL restaurant information from scraped web content.

CRITICAL: Extract EVERY restaurant mentioned, even if mentioned only once. We want comprehensive coverage.

INSTRUCTIONS:
1. Identify ALL restaurants mentioned in the content from ALL sources
2. For restaurants mentioned multiple times across sources, combine their descriptions under one entry
3. For restaurants mentioned only once, include them as individual entries
4. For each restaurant, extract:
   - Name (clean, standardized format)
   - Raw description (combine all mentions, preserve original text exactly as written)
   - Address (if mentioned anywhere, otherwise null)
   - Cuisine tags (be extremely thorough - extract multiple relevant tags)

5. Return data as JSON array with this structure:
[
  {{
    "name": "Restaurant Name",
    "raw_description": "Combined descriptions from all mentions across sources...",
    "address": "Full address if found or null",
    "cuisine_tags": ["tag1", "tag2", "tag3", "tag4", "tag5", ...],
    "mention_count": 2,
    "source_urls": ["url1", "url2"]
  }}
]

CUISINE TAG GUIDELINES - BE EXTREMELY THOROUGH:
- Extract 5-15 tags per restaurant covering all aspects
- Include cuisine type, dining style, specialties, atmosphere, features
- Examples for an Italian restaurant: ["italian", "modern italian", "pasta", "pizza", "wine bar", "romantic", "fine dining", "chef's table", "natural wines", "cocktails", "neighborhood gem", "family-owned", "outdoor seating", "authentic", "traditional"]
- Extract from context clues and descriptions
- Include both specific (e.g., "neapolitan pizza", "sicilian cuisine") and general (e.g., "italian", "pizza") tags
- Always use lowercase for tags
- Common tag categories to extract:
  * Cuisine: italian, french, japanese, mexican, indian, chinese, thai, mediterranean, etc.
  * Style: fine dining, casual, bistro, trattoria, steakhouse, brasserie, taverna, etc.
  * Specialties: pasta, pizza, seafood, steaks, burgers, sushi, ramen, cocktails, wine, coffee, etc.
  * Atmosphere: romantic, family-friendly, trendy, cozy, elegant, rustic, modern, traditional, etc.
  * Features: outdoor seating, live music, chef's table, wine cellar, rooftop, waterfront, etc.
  * Price: affordable, mid-range, upscale, luxury, budget-friendly, etc.
  * Service: michelin starred, chef-owned, locally sourced, organic, farm-to-table, etc.

IMPORTANT: We want to save ALL restaurants found in the scraped content to build a comprehensive database.

CITY: {{city}}
COUNTRY: {{country}}
SOURCES: {{sources}}

CONTENT TO PROCESS:
{{scraped_content}}
""")

    def process_scraped_content(self, scraped_content: str, sources: List[str], city: str, country: str) -> List[Dict[str, Any]]:
        """
        Process scraped content and extract ALL restaurant information using AI
        """
        try:
            logger.info(f"🤖 Processing scraped content for {city}, {country}")
            logger.info(f"📄 Content length: {len(scraped_content)} chars from {len(sources)} sources")

            # Prepare the prompt
            formatted_prompt = self.processing_prompt.format(
                city=city,
                country=country,
                sources="\n".join(sources),
                scraped_content=scraped_content
            )

            # Call OpenAI
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])

            # Parse JSON response
            try:
                restaurants_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                content = response.content
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    restaurants_data = json.loads(content[start_idx:end_idx])
                else:
                    logger.error("❌ Failed to parse JSON from OpenAI response")
                    logger.error(f"Response content: {content[:500]}...")
                    return []

            if not isinstance(restaurants_data, list):
                logger.error("❌ OpenAI response is not a list")
                return []

            # Add city/country to each restaurant and clean data
            for restaurant in restaurants_data:
                restaurant['city'] = city
                restaurant['country'] = country

                # Ensure sources is always a list
                if 'source_urls' in restaurant:
                    restaurant['sources'] = restaurant['source_urls']
                    del restaurant['source_urls']
                else:
                    restaurant['sources'] = sources

                # Ensure mention_count is a valid integer
                if not isinstance(restaurant.get('mention_count'), int) or restaurant.get('mention_count') < 1:
                    restaurant['mention_count'] = 1

                # Clean address field
                if restaurant.get('address') in ['null', '', None]:
                    restaurant['address'] = None

                # Ensure cuisine_tags is a list
                if not isinstance(restaurant.get('cuisine_tags'), list):
                    restaurant['cuisine_tags'] = []

            logger.info(f"✅ Extracted {len(restaurants_data)} restaurants from content")

            # Log some sample extractions for debugging
            for i, restaurant in enumerate(restaurants_data[:3]):  # Show first 3
                logger.info(f"Sample restaurant {i+1}: {restaurant['name']} - {len(restaurant.get('cuisine_tags', []))} tags")

            return restaurants_data

        except Exception as e:
            logger.error(f"❌ Error processing scraped content: {e}")
            return []

    def save_all_restaurants_to_supabase(self, restaurants_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save ALL processed restaurant data to Supabase with intelligent deduplication
        Returns statistics about the save operation
        """
        try:
            logger.info(f"💾 Saving ALL {len(restaurants_data)} restaurants to database")

            stats = {
                'total_processed': len(restaurants_data),
                'new_restaurants': 0,
                'updated_restaurants': 0,
                'failed_saves': 0,
                'total_tags_added': 0
            }

            for restaurant in restaurants_data:
                try:
                    # Use the database's save method (handles deduplication automatically)
                    restaurant_id = self.db.save_restaurant({
                        'name': restaurant['name'],
                        'raw_description': restaurant['raw_description'],
                        'address': restaurant.get('address'),
                        'city': restaurant['city'],
                        'country': restaurant['country'],
                        'cuisine_tags': restaurant.get('cuisine_tags', []),
                        'sources': restaurant.get('sources', [])
                    })

                    if restaurant_id:
                        # Check if this was a new restaurant or update
                        existing = self.db._find_existing_restaurant(restaurant['name'], restaurant['city'])
                        if existing and existing.get('first_added'):
                            # This was an update
                            stats['updated_restaurants'] += 1
                        else:
                            # This was a new restaurant
                            stats['new_restaurants'] += 1

                        stats['total_tags_added'] += len(restaurant.get('cuisine_tags', []))
                        logger.info(f"✅ Processed restaurant: {restaurant['name']} ({len(restaurant.get('cuisine_tags', []))} tags)")

                    else:
                        stats['failed_saves'] += 1

                except Exception as e:
                    logger.error(f"❌ Error saving restaurant {restaurant.get('name', 'unknown')}: {e}")
                    stats['failed_saves'] += 1
                    continue

            logger.info(f"✅ Database update complete:")
            logger.info(f"   - Total processed: {stats['total_processed']}")
            logger.info(f"   - New restaurants: {stats['new_restaurants']}")
            logger.info(f"   - Updated restaurants: {stats['updated_restaurants']}")
            logger.info(f"   - Failed saves: {stats['failed_saves']}")
            logger.info(f"   - Total cuisine tags added: {stats['total_tags_added']}")

            return stats

        except Exception as e:
            logger.error(f"❌ Error in save_all_restaurants_to_supabase: {e}")
            return {
                'total_processed': len(restaurants_data),
                'new_restaurants': 0,
                'updated_restaurants': 0,
                'failed_saves': len(restaurants_data),
                'total_tags_added': 0
            }

    def update_all_restaurants_with_geodata(self, city: str, country: str) -> Dict[str, Any]:
        """
        Update ALL restaurants in the database for a city with address and coordinate data
        This runs during follow-up search to enhance ALL stored restaurants, not just recommendations
        """
        try:
            logger.info(f"🗺️ Starting geodata update for ALL restaurants in {city}, {country}")

            # Get all restaurants for this city that don't have coordinates
            restaurants_without_coords = self.db.supabase.table('restaurants')\
                .select('id, name, address')\
                .eq('city', city)\
                .eq('country', country)\
                .is_('coordinates', 'null')\
                .execute()

            if not restaurants_without_coords.data:
                logger.info(f"ℹ️ No restaurants without coordinates found in {city}")
                return {'updated_count': 0, 'failed_count': 0}

            logger.info(f"📍 Found {len(restaurants_without_coords.data)} restaurants without coordinates")

            # Use Google Maps to find coordinates for each restaurant
            updated_count = 0
            failed_count = 0

            for restaurant_record in restaurants_without_coords.data:
                try:
                    restaurant_id = restaurant_record['id']
                    restaurant_name = restaurant_record['name']

                    # Try to get coordinates using Google Maps
                    coordinates = self._get_coordinates_from_google_maps(restaurant_name, city)

                    if coordinates:
                        # Update the restaurant with coordinates
                        lat, lng = coordinates
                        self.db.update_restaurant_geodata(restaurant_id, restaurant_record.get('address', ''), (lat, lng))
                        updated_count += 1
                        logger.info(f"📍 Updated coordinates for: {restaurant_name}")
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️ Could not find coordinates for: {restaurant_name}")

                except Exception as e:
                    logger.error(f"❌ Error updating geodata for restaurant {restaurant_record.get('name')}: {e}")
                    failed_count += 1

            logger.info(f"✅ Geodata update complete for {city}:")
            logger.info(f"   - Updated: {updated_count}")
            logger.info(f"   - Failed: {failed_count}")

            return {
                'updated_count': updated_count,
                'failed_count': failed_count,
                'total_processed': len(restaurants_without_coords.data)
            }

        except Exception as e:
            logger.error(f"❌ Error in update_all_restaurants_with_geodata: {e}")
            return {'updated_count': 0, 'failed_count': 0}

    def _get_coordinates_from_google_maps(self, restaurant_name: str, city: str) -> Optional[tuple]:
        """
        Get coordinates for a restaurant using Google Maps API
        Returns (lat, lng) tuple or None
        """
        try:
            import googlemaps

            # Get Google Maps API key from config
            api_key = getattr(self.config, 'GOOGLE_MAPS_API_KEY', None)
            if not api_key:
                logger.warning("Google Maps API key not configured - skipping coordinate lookup")
                return None

            gmaps = googlemaps.Client(key=api_key)

            # Search for the restaurant
            search_query = f"{restaurant_name} restaurant {city}"
            search_response = gmaps.places(query=search_query)

            results = search_response.get("results", [])
            if not results:
                return None

            # Get coordinates from first result
            first_result = results[0]
            geometry = first_result.get("geometry", {})
            location = geometry.get("location", {})

            if location.get("lat") and location.get("lng"):
                return (float(location["lat"]), float(location["lng"]))

            return None

        except Exception as e:
            logger.error(f"Error getting coordinates from Google Maps: {e}")
            return None

    def check_existing_restaurants(self, city: str, cuisine_type: str = None) -> List[Dict[str, Any]]:
        """Check existing restaurants using database methods"""
        try:
            if cuisine_type:
                return self.db.search_restaurants_by_cuisine(city, [cuisine_type.lower()])
            else:
                return self.db.get_restaurants_by_city(city)
        except Exception as e:
            logger.error(f"❌ Error checking existing restaurants: {e}")
            return []

    def get_restaurants_for_city(self, city: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get restaurants using database method"""
        return self.db.get_restaurants_by_city(city, limit)


# ========================================
# STANDALONE FUNCTIONS (called by langchain_orchestrator.py)
# ========================================

def process_all_scraped_restaurants(scraped_content: str, sources: List[str], city: str, country: str, config) -> Dict[str, Any]:
    """
    MAIN FUNCTION: Process ALL scraped content and save ALL restaurants to database

    Args:
        scraped_content: Combined scraped text content from all sources
        sources: List of source URLs
        city: City where restaurants are located  
        country: Country where restaurants are located
        config: Application configuration object

    Returns:
        Dictionary with processing statistics and restaurant data
    """
    try:
        agent = SupabaseUpdateAgent(config)

        # Process scraped content to extract ALL restaurants
        restaurants_data = agent.process_scraped_content(scraped_content, sources, city, country)

        if restaurants_data:
            # Save ALL restaurants to database
            save_stats = agent.save_all_restaurants_to_supabase(restaurants_data)

            logger.info(f"✅ Successfully processed ALL {len(restaurants_data)} restaurants for {city}")

            return {
                'success': True,
                'restaurants_processed': restaurants_data,
                'save_statistics': save_stats,
                'total_restaurants': len(restaurants_data)
            }
        else:
            logger.warning("⚠️ No restaurants extracted from content")
            return {
                'success': False,
                'restaurants_processed': [],
                'save_statistics': {'total_processed': 0, 'new_restaurants': 0, 'updated_restaurants': 0},
                'total_restaurants': 0
            }

    except Exception as e:
        logger.error(f"❌ Error in process_all_scraped_restaurants: {e}")
        return {
            'success': False,
            'restaurants_processed': [],
            'save_statistics': {'total_processed': 0, 'new_restaurants': 0, 'updated_restaurants': 0},
            'total_restaurants': 0
        }


def update_city_geodata(city: str, country: str, config) -> Dict[str, Any]:
    """
    Update ALL restaurants in a city with geographic data (coordinates and addresses)

    Args:
        city: City to update
        country: Country the city is in
        config: Application configuration

    Returns:
        Dictionary with update statistics
    """
    try:
        agent = SupabaseUpdateAgent(config)
        return agent.update_all_restaurants_with_geodata(city, country)

    except Exception as e:
        logger.error(f"❌ Error in update_city_geodata: {e}")
        return {'updated_count': 0, 'failed_count': 0}


def check_city_coverage(city: str, cuisine_type: str, config) -> Dict[str, Any]:
    """
    Check if we have good coverage for a city/cuisine combination

    Args:
        city: City to check
        cuisine_type: Cuisine type to check
        config: Application configuration

    Returns:
        Dictionary with coverage information
    """
    try:
        agent = SupabaseUpdateAgent(config)
        existing_restaurants = agent.check_existing_restaurants(city, cuisine_type)

        total_count = len(existing_restaurants)
        cuisine_matches = len([r for r in existing_restaurants if cuisine_type.lower() in [tag.lower() for tag in r.get('cuisine_tags', [])]])

        return {
            'has_data': total_count > 0,
            'total_restaurants': total_count,
            'cuisine_matches': cuisine_matches,
            'sufficient_coverage': cuisine_matches >= 5,  # Threshold for good coverage
            'restaurants': existing_restaurants
        }

    except Exception as e:
        logger.error(f"❌ Error checking city coverage: {e}")
        return {
            'has_data': False,
            'total_restaurants': 0,
            'cuisine_matches': 0,
            'sufficient_coverage': False,
            'restaurants': []
        }