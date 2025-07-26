# utils/supabase_manager.py - UPDATED FOR NEW SIMPLIFIED SCHEMA
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from supabase import create_client, Client
import numpy as np

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages all Supabase operations for the AI-powered restaurant bot with simplified schema"""

    def __init__(self, config):
        """Initialize SupabaseManager with new simplified schema support"""
        self.config = config

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("✅ Supabase client initialized for new schema")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise

        # Initialize geocoder (optional for address geocoding)
        try:
            from geopy.geocoders import Nominatim
            self.geocoder = Nominatim(user_agent="ai-restaurant-bot")
            logger.info("✅ Geocoder initialized")
        except Exception as e:
            logger.warning(f"⚠️ Geocoder initialization failed: {e}")
            self.geocoder = None

    # ============ DOMAIN INTELLIGENCE METHODS (KEEP EXISTING) ============

    def save_domain_intelligence(self, domain: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save domain intelligence data (existing functionality)"""
        try:
            # Check if domain already exists
            existing = self.get_domain_intelligence(domain)

            if existing:
                # Update existing domain
                self.supabase.table('domain_intelligence')\
                    .update(intelligence_data)\
                    .eq('domain', domain)\
                    .execute()
            else:
                # Insert new domain
                data = {'domain': domain, **intelligence_data}
                self.supabase.table('domain_intelligence').insert(data).execute()

            logger.debug(f"💾 Saved domain intelligence for {domain}")
            return True

        except Exception as e:
            logger.error(f"Error saving domain intelligence for {domain}: {e}")
            return False

    def get_domain_intelligence(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain intelligence data"""
        try:
            result = self.supabase.table('domain_intelligence').select('*').eq('domain', domain).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting domain intelligence for {domain}: {e}")
            return None

    def update_domain_success(self, domain: str, success: bool, restaurants_found: int = 0):
        """Update domain success/failure counts"""
        try:
            current = self.get_domain_intelligence(domain)
            if not current:
                logger.warning(f"No domain intelligence found for {domain}")
                return

            # Update counts
            if success:
                new_success = current['success_count'] + 1
                new_failure = current['failure_count']
                update_data = {
                    'success_count': new_success,
                    'total_restaurants_found': current['total_restaurants_found'] + restaurants_found,
                    'last_successful_scrape': datetime.now(timezone.utc).isoformat()
                }
            else:
                new_success = current['success_count'] 
                new_failure = current['failure_count'] + 1
                update_data = {
                    'failure_count': new_failure
                }

                # Block domain if too many failures
                if new_failure >= getattr(self.config, 'DOMAIN_FAILURE_LIMIT', 10):
                    update_data.update({
                        'is_blocked': True,
                        'blocked_at': datetime.now(timezone.utc).isoformat()
                    })

            # Calculate new confidence
            total_attempts = new_success + new_failure
            if total_attempts > 0:
                update_data['confidence'] = new_success / total_attempts

            update_data['last_updated_at'] = datetime.now(timezone.utc).isoformat()

            # Update database
            self.supabase.table('domain_intelligence').update(update_data).eq('domain', domain).execute()
            logger.info(f"Updated domain intelligence for {domain}: success={success}")

        except Exception as e:
            logger.error(f"Error updating domain success for {domain}: {e}")

    def get_trusted_domains(self, min_confidence: float = None) -> List[str]:
        """Get list of trusted domains based on success rate"""
        try:
            min_conf = min_confidence or getattr(self.config, 'DOMAIN_SUCCESS_THRESHOLD', 0.7)

            result = self.supabase.table('domain_intelligence')\
                .select('domain')\
                .gte('confidence', min_conf)\
                .eq('is_blocked', False)\
                .execute()

            return [row['domain'] for row in result.data]

        except Exception as e:
            logger.error(f"Error getting trusted domains: {e}")
            return []

    def load_all_domain_intelligence(self) -> List[Dict[str, Any]]:
        """Load all domain intelligence records"""
        try:
            result = self.supabase.table('domain_intelligence').select('*').execute()
            return result.data
        except Exception as e:
            logger.error(f"Error loading domain intelligence: {e}")
            return []

    # ============ NEW RESTAURANT METHODS FOR SIMPLIFIED SCHEMA ============

    def save_restaurant(self, restaurant_data: Dict[str, Any]) -> Optional[str]:
        """Save restaurant with new simplified schema"""
        try:
            name = restaurant_data.get('name', '').strip()
            city = restaurant_data.get('city', '').strip()

            if not name or not city:
                logger.warning(f"Missing required fields: name='{name}', city='{city}'")
                return None

            # Check if restaurant already exists (by name and city)
            existing = self._find_existing_restaurant(name, city)

            if existing:
                # Update existing restaurant
                restaurant_id = existing['id']

                # Combine descriptions
                new_description = restaurant_data.get('raw_description', '')
                if new_description:
                    combined_description = existing['raw_description'] + "\n\n--- NEW MENTION ---\n\n" + new_description
                else:
                    combined_description = existing['raw_description']

                # Combine and deduplicate cuisine tags
                existing_tags = existing.get('cuisine_tags', [])
                new_tags = restaurant_data.get('cuisine_tags', [])
                combined_tags = list(set(existing_tags + new_tags))

                # Combine and deduplicate sources
                existing_sources = existing.get('sources', [])
                new_sources = restaurant_data.get('sources', [])
                combined_sources = list(set(existing_sources + new_sources))

                update_data = {
                    'raw_description': combined_description,
                    'cuisine_tags': combined_tags,
                    'sources': combined_sources,
                    'mention_count': existing['mention_count'] + 1,
                    'last_updated': datetime.now().isoformat(),
                    # Update address if we have a new one and existing is null
                    'address': restaurant_data.get('address') if existing['address'] is None else existing['address']
                }

                self.supabase.table('restaurants')\
                    .update(update_data)\
                    .eq('id', restaurant_id)\
                    .execute()

                logger.info(f"🔄 Updated existing restaurant: {name}")
                return str(restaurant_id)

            else:
                # Insert new restaurant
                insert_data = {
                    'name': name,
                    'raw_description': restaurant_data.get('raw_description', ''),
                    'address': restaurant_data.get('address'),
                    'city': city,
                    'country': restaurant_data.get('country', ''),
                    'cuisine_tags': restaurant_data.get('cuisine_tags', []),
                    'sources': restaurant_data.get('sources', []),
                    'mention_count': 1,
                    'first_added': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }

                # Add coordinates if available
                if restaurant_data.get('coordinates'):
                    coords = restaurant_data['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        insert_data['coordinates'] = f"POINT({coords[1]} {coords[0]})"  # lon, lat for PostGIS

                result = self.supabase.table('restaurants').insert(insert_data).execute()

                if result.data:
                    restaurant_id = result.data[0]['id']
                    logger.info(f"➕ Inserted new restaurant: {name}")
                    return str(restaurant_id)
                else:
                    logger.error(f"Failed to insert restaurant: {name}")
                    return None

        except Exception as e:
            logger.error(f"Error saving restaurant: {e}")
            logger.error(f"Restaurant data: {restaurant_data}")
            return None

    def _find_existing_restaurant(self, name: str, city: str) -> Optional[Dict]:
        """Find existing restaurant by name and city"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('name', name)\
                .eq('city', city)\
                .execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error finding existing restaurant: {e}")
            return None

    def get_restaurants_by_city(self, city: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all restaurants for a city, ordered by mention count"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('city', city)\
                .order('mention_count', desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error getting restaurants for {city}: {e}")
            return []

    def search_restaurants_by_cuisine(self, city: str, cuisine_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Search restaurants by city and cuisine tags using array overlap"""
        try:
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('city', city)\
                .overlaps('cuisine_tags', cuisine_tags)\
                .order('mention_count', desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error searching restaurants by cuisine: {e}")
            return []

    def update_restaurant_geodata(self, restaurant_id: int, address: str, coordinates: Tuple[float, float]):
        """Update restaurant with address and coordinates"""
        try:
            update_data = {
                'address': address,
                'coordinates': f"POINT({coordinates[1]} {coordinates[0]})",  # lon, lat for PostGIS
                'last_updated': datetime.now().isoformat()
            }

            self.supabase.table('restaurants')\
                .update(update_data)\
                .eq('id', restaurant_id)\
                .execute()

            logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id}")

        except Exception as e:
            logger.error(f"Error updating geodata: {e}")

    def get_restaurants_by_preference_tags(self, city: str, preference_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Get restaurants that match any of the preference tags"""
        try:
            # Use array overlap to find restaurants with matching cuisine tags
            result = self.supabase.table('restaurants')\
                .select('*')\
                .eq('city', city)\
                .overlaps('cuisine_tags', preference_tags)\
                .order('mention_count', desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error getting restaurants by preference tags: {e}")
            return []

    # ============ SIMPLIFIED CACHE METHODS ============

    def cache_search_results(self, query: str, results: Dict[str, Any]) -> bool:
        """Cache search results (simplified version)"""
        try:
            # Create query hash
            query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()

            # For now, we'll store in a simple JSON format
            # Note: We removed the search_cache table, so this is a simplified implementation
            # You might want to implement caching differently or skip it
            logger.debug(f"Would cache search results for: {query}")
            return True

        except Exception as e:
            logger.error(f"Error caching search results: {e}")
            return False

    def get_cached_results(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search results (simplified version)"""
        try:
            # Since we removed search_cache table, return None
            # This forces fresh searches each time
            return None

        except Exception as e:
            logger.error(f"Error getting cached results: {e}")
            return None

    # ============ USER PREFERENCES (OPTIONAL - DEPENDS ON YOUR NEEDS) ============

    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Save user preferences (if you still need this feature)"""
        try:
            # Note: We removed user_preferences table
            # If you need this feature, you'll need to create a user_preferences table
            logger.debug(f"Would save user preferences for: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
            return False

    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences (if you still need this feature)"""
        try:
            # Note: We removed user_preferences table
            return None

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None

    def add_to_search_history(self, user_id: str, query: str, results_count: int = 0):
        """Add search to user's history (if you still need this feature)"""
        try:
            # Note: We removed user_preferences table
            logger.debug(f"Would add to search history for user {user_id}: {query}")

        except Exception as e:
            logger.error(f"Error adding to search history: {e}")

    # ============ DATABASE STATS AND MAINTENANCE ============

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            stats = {}

            # Count restaurants
            restaurants_count = self.supabase.table('restaurants').select('id', count='exact').execute()
            stats['total_restaurants'] = restaurants_count.count

            # Count domains
            domains_count = self.supabase.table('domain_intelligence').select('domain', count='exact').execute()
            stats['total_domains'] = domains_count.count

            # Get top cities
            cities_result = self.supabase.table('restaurants')\
                .select('city', count='exact')\
                .group_by('city')\
                .order('count', desc=True)\
                .limit(10)\
                .execute()

            stats['top_cities'] = cities_result.data

            # Get top cuisine tags
            # Note: This is more complex with arrays, so we'll skip for now
            stats['top_cuisines'] = []

            logger.info(f"📊 Database stats: {stats['total_restaurants']} restaurants, {stats['total_domains']} domains")
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_restaurants': 0,
                'total_domains': 0,
                'top_cities': [],
                'top_cuisines': []
            }

    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data (simplified version)"""
        try:
            # Since we have a simplified schema, cleanup is minimal
            # We mainly keep restaurant data

            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()

            # Could clean up old domain intelligence if needed
            logger.info(f"Would clean up data older than {days_old} days")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    # ============ GEOCODING HELPER ============

    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to get coordinates"""
        try:
            if not self.geocoder:
                logger.warning("Geocoder not available")
                return None

            location = self.geocoder.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None

        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
            return None

    # ============ BACKWARD COMPATIBILITY METHODS ============

    def save_restaurant_data(self, restaurant_data: Dict[str, Any]) -> Optional[str]:
        """Backward compatibility wrapper"""
        return self.save_restaurant(restaurant_data)

    def search_similar_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simplified content search (since we removed RAG tables)"""
        try:
            # For now, return empty list since we simplified the schema
            # If you need RAG functionality, you'll need to add content_chunks table back
            return []

        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []