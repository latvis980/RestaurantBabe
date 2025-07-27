# agents/supabase_update_agent.py - COMPLETE WORKING VERSION
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

        # Initialize Claude instead of OpenAI
        anthropic_key = getattr(config, 'ANTHROPIC_API_KEY', os.getenv("ANTHROPIC_API_KEY"))
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY must be configured")

        # Use Claude Sonnet 4 with large context window
        from langchain_anthropic import ChatAnthropic

        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",  # Latest Claude with 200K context
            temperature=0.1,
            anthropic_api_key=anthropic_key,
            max_tokens=4000  # Enough for detailed JSON response
        )
        logger.info("✅ Claude Sonnet initialized for restaurant processing (200K context window)")

        # Enhanced prompt for Claude (can handle much longer content)
        self.processing_prompt = ChatPromptTemplate.from_template("""
    You are a restaurant data extraction specialist. Your task is to process web content about restaurants and food recommendations to build a comprehensive database.

    TASK: Extract ALL restaurants mentioned in the content from multiple web sources about {city}, {country}.

    INSTRUCTIONS:
    1. Read through ALL the provided content carefully
    2. Identify EVERY restaurant mentioned, even if only mentioned once
    3. For restaurants mentioned multiple times across sources, combine their descriptions
    4. Extract comprehensive information for each restaurant
    5. Be thorough with cuisine tags - include cuisine type, style, atmosphere, specialties

    EXTRACTION REQUIREMENTS:
    - Name: Clean, standardized restaurant name
    - Raw description: Combine ALL mentions, preserve original context and details
    - Address: Full address if mentioned anywhere, otherwise null
    - Cuisine tags: 5-15 comprehensive tags covering cuisine, style, atmosphere, features
    - Mention count: How many sources mentioned this restaurant

    CUISINE TAG EXAMPLES:
    - Basic: ["italian", "pizza", "pasta"]  
    - Comprehensive: ["italian", "neapolitan pizza", "traditional", "family-owned", "cozy", "authentic", "wood-fired", "casual dining", "neighborhood gem", "wine selection"]

    OUTPUT FORMAT - Return ONLY valid JSON:
    [
      {{
        "name": "Restaurant Name",
        "raw_description": "Complete combined descriptions from all mentions with full context and details",
        "address": "Full address if found, otherwise null",
        "cuisine_tags": ["cuisine1", "style1", "atmosphere1", "feature1", "specialty1", ...],
        "mention_count": 2,
        "source_urls": ["url1", "url2"]
      }}
    ]

    CONTENT SOURCES:
    {sources}

    WEB CONTENT TO ANALYZE:
    {scraped_content}

    Please extract ALL restaurants and return comprehensive JSON data. Focus on building a complete database entry for each restaurant.
    """)

    def process_scraped_content(self, scraped_content: str, sources: List[str], city: str, country: str) -> List[Dict[str, Any]]:
        """
        Process scraped content using Claude with large context window - can handle much longer content
        """
        try:
            logger.info(f"🤖 Processing scraped content with Claude Sonnet for {city}, {country}")
            logger.info(f"📄 Content length: {len(scraped_content)} chars from {len(sources)} sources")

            # Claude can handle much larger content, but let's still be reasonable
            max_content_length = 150000  # 150K chars - well within Claude's limits
            if len(scraped_content) > max_content_length:
                logger.warning(f"⚠️ Content very long ({len(scraped_content)} chars), truncating to {max_content_length}")
                scraped_content = scraped_content[:max_content_length] + "\n\n[Content truncated for processing]"

            # Clean content
            scraped_content = scraped_content.replace('\x00', '').strip()

            # Prepare the prompt
            try:
                formatted_prompt = self.processing_prompt.format(
                    city=city,
                    country=country,
                    sources="\n".join(sources),  # Include all sources for Claude
                    scraped_content=scraped_content
                )
            except Exception as e:
                logger.error(f"❌ Error formatting prompt: {e}")
                return []

            logger.info(f"📤 Sending request to Claude (prompt length: {len(formatted_prompt)} chars)")

            # Call Claude
            try:
                response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
                logger.info(f"📥 Received response from Claude")
            except Exception as e:
                logger.error(f"❌ Claude API error: {e}")
                return self._fallback_restaurant_extraction(scraped_content, city, country)

            # Parse Claude's response
            restaurants_data = self._parse_claude_response(response.content)

            if not restaurants_data:
                logger.warning("⚠️ No restaurants parsed from Claude response, trying fallback")
                return self._fallback_restaurant_extraction(scraped_content, city, country)

            # Process and clean the data
            for restaurant in restaurants_data:
                restaurant['city'] = city
                restaurant['country'] = country

                # Ensure sources is properly set
                if 'source_urls' in restaurant:
                    restaurant['sources'] = restaurant['source_urls']
                    del restaurant['source_urls']
                else:
                    restaurant['sources'] = sources

                # Validate and clean fields
                if not isinstance(restaurant.get('mention_count'), int) or restaurant.get('mention_count') < 1:
                    restaurant['mention_count'] = 1

                if restaurant.get('address') in ['null', '', None, 'null']:
                    restaurant['address'] = None

                if not isinstance(restaurant.get('cuisine_tags'), list):
                    restaurant['cuisine_tags'] = []

                # Ensure raw_description is not empty
                if not restaurant.get('raw_description'):
                    restaurant['raw_description'] = f"Restaurant in {city} mentioned in food recommendations"

            logger.info(f"✅ Claude extracted {len(restaurants_data)} restaurants from content")

            # Log sample for debugging
            for i, restaurant in enumerate(restaurants_data[:3]):
                logger.info(f"Sample {i+1}: {restaurant['name']} - {len(restaurant.get('cuisine_tags', []))} tags")

            return restaurants_data

        except Exception as e:
            logger.error(f"❌ Error processing scraped content with Claude: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_restaurant_extraction(scraped_content, city, country)

    def _parse_claude_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse Claude's response - Claude is usually better at following JSON format"""
        try:
            # Claude typically returns clean JSON, try direct parsing first
            restaurants_data = json.loads(content.strip())
            if isinstance(restaurants_data, list):
                logger.info(f"✅ Successfully parsed Claude JSON response")
                return restaurants_data
        except json.JSONDecodeError:
            pass

        try:
            # Look for JSON array in the content
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                restaurants_data = json.loads(json_str)
                if isinstance(restaurants_data, list):
                    logger.info(f"✅ Extracted JSON from Claude response")
                    return restaurants_data
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            # Check for code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
                restaurants_data = json.loads(json_str)
                if isinstance(restaurants_data, list):
                    logger.info(f"✅ Extracted JSON from Claude code block")
                    return restaurants_data
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.error(f"❌ Could not parse Claude response as JSON. Content: {content[:300]}...")
        return []

    def _fallback_restaurant_extraction(self, scraped_content: str, city: str, country: str) -> List[Dict[str, Any]]:
        """Enhanced fallback extraction using multiple patterns"""
        try:
            logger.info("🔄 Using enhanced fallback restaurant extraction method")

            import re

            restaurants = []
            found_names = set()

            # Enhanced patterns for restaurant detection
            patterns = [
                # Restaurant + name patterns
                r'(?:restaurant|ristorante|bistro|brasserie|tavern|trattoria|osteria)\s+([A-Z][a-zA-Z\s\'&\-\.]+)',
                r'([A-Z][a-zA-Z\s\'&\-\.]+)\s+(?:restaurant|ristorante|bistro|brasserie)',

                # Action + restaurant patterns  
                r'(?:visit|try|dine\s+at|eat\s+at|go\s+to)\s+([A-Z][a-zA-Z\s\'&\-\.]+)',

                # Quoted restaurant names
                r'"([A-Z][a-zA-Z\s\'&\-\.]+)"',
                r"'([A-Z][a-zA-Z\s\'&\-\.]+)'",

                # Chef's restaurant patterns
                r'(?:chef|owned\s+by)\s+[a-zA-Z]+\s+at\s+([A-Z][a-zA-Z\s\'&\-\.]+)',

                # Location patterns
                r'([A-Z][a-zA-Z\s\'&\-\.]+)\s+(?:in|on|at)\s+(?:[A-Z][a-zA-Z\s]+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, scraped_content, re.IGNORECASE)
                for match in matches:
                    name = match.strip().title()
                    # Filter out common false positives
                    if (len(name) >= 3 and len(name) <= 50 and 
                        not any(word in name.lower() for word in ['the', 'and', 'with', 'from', 'this', 'that', 'some', 'many', 'best', 'top'])):
                        found_names.add(name)

            # Convert to restaurant objects
            for name in list(found_names)[:15]:  # Limit to 15 restaurants
                restaurants.append({
                    'name': name,
                    'raw_description': f"Restaurant mentioned in {city} food and dining content",
                    'address': None,
                    'cuisine_tags': ["restaurant", "dining"],
                    'mention_count': 1,
                    'city': city,
                    'country': country,
                    'sources': []
                })

            logger.info(f"🔄 Fallback extraction found {len(restaurants)} restaurants")
            return restaurants

        except Exception as e:
            logger.error(f"❌ Fallback extraction failed: {e}")
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
            # Check for both PostGIS coordinates AND latitude/longitude columns
            restaurants_without_coords = self.db.supabase.table('restaurants')\
                .select('id, name, address')\
                .eq('city', city)\
                .eq('country', country)\
                .or_('coordinates.is.null,latitude.is.null')\
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
        logger.info(f"🚀 STARTING AI-POWERED SUPABASE UPDATE AGENT")
        logger.info(f"📄 Processing content for {city}, {country}")

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
        import traceback
        traceback.print_exc()
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