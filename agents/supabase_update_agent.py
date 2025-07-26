# agents/supabase_update_agent.py
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
    Agent responsible for processing scraped content and updating the Supabase restaurants database.

    This agent:
    1. Extracts restaurant information from scraped web content
    2. Combines duplicate restaurants mentioned across multiple sources
    3. Extracts comprehensive cuisine tags using AI
    4. Saves processed data to Supabase restaurants table
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
You are a restaurant data processor. Your task is to extract and organize restaurant information from scraped web content.

INSTRUCTIONS:
1. Identify all restaurants mentioned in the content
2. For restaurants mentioned multiple times, combine their descriptions under one entry
3. For each restaurant, extract:
   - Name (clean, standardized)
   - Raw description (combine all mentions, preserve original text)
   - Address (if mentioned, otherwise null)
   - Cuisine tags (be thorough - extract multiple relevant tags)

4. Return data as JSON array with this structure:
[
  {{
    "name": "Restaurant Name",
    "raw_description": "Combined descriptions from all mentions...",
    "address": "Full address or null",
    "cuisine_tags": ["tag1", "tag2", "tag3", ...],
    "mention_count": 2
  }}
]

CUISINE TAG GUIDELINES:
- Be comprehensive: include cuisine type, dining style, specialties, atmosphere
- Examples: ["italian", "modern italian", "pasta", "pizza", "wine bar", "romantic", "fine dining", "chef's table", "natural wines", "cocktails", "neighborhood gem", "family-owned"]
- Extract from context clues in descriptions
- Include both specific (e.g., "neapolitan pizza") and general (e.g., "italian") tags
- Always use lowercase for tags
- Common tag categories:
  * Cuisine: italian, french, japanese, mexican, indian, etc.
  * Style: fine dining, casual, bistro, trattoria, steakhouse, etc.
  * Specialties: pasta, pizza, sushi, tacos, burgers, seafood, etc.
  * Atmosphere: romantic, family-friendly, cozy, modern, traditional, etc.
  * Features: wine bar, cocktails, rooftop, outdoor seating, etc.
  * Chef-driven: chef's table, chef's cuisine, celebrity chef, etc.

IMPORTANT RULES:
- Always extract restaurants even if information is incomplete
- Combine multiple mentions of the same restaurant
- Preserve original descriptions without editing
- Be thorough with cuisine tags - one restaurant should have 5-10 relevant tags
- If address is partial or unclear, still include it
- If no address mentioned, use null

SCRAPED CONTENT:
{{scraped_content}}

SOURCES (URLs):
{{sources}}

Return only the JSON array, no other text.""")

    def process_scraped_content(self, scraped_content: str, sources: List[str], city: str, country: str) -> List[Dict[str, Any]]:
        """
        Process scraped content and extract restaurant information using AI

        Args:
            scraped_content: Combined text content from all scraped sources
            sources: List of source URLs
            city: City where restaurants are located
            country: Country where restaurants are located

        Returns:
            List of processed restaurant data dictionaries
        """
        try:
            logger.info(f"🔄 Processing scraped content for {city}, {country}")
            logger.info(f"📄 Content length: {len(scraped_content)} chars from {len(sources)} sources")

            # Process content with LLM
            response = self.llm.invoke([
                HumanMessage(content=self.processing_prompt.format(
                    scraped_content=scraped_content,
                    sources=sources
                ))
            ])

            # Parse JSON response
            try:
                restaurants_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"LLM Response: {response.content[:500]}...")
                return []

            # Validate that we got a list
            if not isinstance(restaurants_data, list):
                logger.error(f"Expected list from LLM, got {type(restaurants_data)}")
                return []

            # Add city, country, and sources to each restaurant
            for restaurant in restaurants_data:
                restaurant['city'] = city
                restaurant['country'] = country
                restaurant['sources'] = sources

                # Ensure cuisine_tags is a list
                if not isinstance(restaurant.get('cuisine_tags'), list):
                    restaurant['cuisine_tags'] = []

                # Ensure mention_count is valid
                if not isinstance(restaurant.get('mention_count'), int) or restaurant.get('mention_count') < 1:
                    restaurant['mention_count'] = 1

                # Clean address field
                if restaurant.get('address') == 'null' or restaurant.get('address') == '':
                    restaurant['address'] = None

            logger.info(f"✅ Extracted {len(restaurants_data)} restaurants from content")
            return restaurants_data

        except Exception as e:
            logger.error(f"❌ Error processing scraped content: {e}")
            return []

    def save_restaurants_to_supabase(self, restaurants_data: List[Dict[str, Any]]) -> bool:
        """
        Save processed restaurant data to Supabase with intelligent deduplication
        """
        try:
            logger.info(f"💾 Saving {len(restaurants_data)} restaurants to database")

            saved_count = 0

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
                        saved_count += 1
                        logger.info(f"✅ Processed restaurant: {restaurant['name']}")

                except Exception as e:
                    logger.error(f"❌ Error saving restaurant {restaurant.get('name', 'unknown')}: {e}")
                    continue

            logger.info(f"✅ Database update complete: {saved_count} restaurants processed")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving to database: {e}")
            return False

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

    def update_restaurant_geodata(self, restaurant_id: int, address: str, coordinates: tuple):
        """Update restaurant with address and coordinates"""
        try:
            # Use the database method instead of direct Supabase call
            self.db.update_restaurant_geodata(restaurant_id, address, coordinates)
            logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id}")
        except Exception as e:
            logger.error(f"❌ Error updating geodata: {e}")

    def process_search_results(scraped_content: str, sources: List[str], city: str, country: str, config) -> List[Dict[str, Any]]:
        """
        Main function to process search results and update database

        Args:
            scraped_content: Combined scraped text content
            sources: List of source URLs
            city: City where restaurants are located  
            country: Country where restaurants are located
            config: Application configuration object

        Returns:
            List of processed restaurant data
        """
        try:
            agent = SupabaseUpdateAgent(config)

            # Process scraped content
            restaurants_data = agent.process_scraped_content(scraped_content, sources, city, country)

            if restaurants_data:
                # Save to database
                success = agent.save_restaurants_to_supabase(restaurants_data)
                if success:
                    logger.info(f"✅ Successfully processed {len(restaurants_data)} restaurants for {city}")
                    return restaurants_data
                else:
                    logger.error("❌ Failed to save restaurants to database")
                    return []
            else:
                logger.warning("⚠️ No restaurants extracted from content")
                return []

        except Exception as e:
            logger.error(f"❌ Error in process_search_results: {e}")
            return []

    # Additional helper function for checking existing data
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
            cuisine_matches = len([r for r in existing_restaurants if cuisine_type.lower() in r.get('cuisine_tags', [])])

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