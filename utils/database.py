# utils/database.py - SIMPLIFIED: Direct Supabase interface (no wrapper)
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from supabase import create_client, Client
import numpy as np

logger = logging.getLogger(__name__)

class Database:
    """
    Direct Supabase database interface for the AI-powered restaurant bot.
    Simplified, single-purpose database class with clear naming.
    """

    def __init__(self, config):
        """Initialize direct Supabase connection"""
        self.config = config

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("✅ Database (Supabase) initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise

        # Initialize geocoder (optional)
        try:
            from geopy.geocoders import Nominatim
            self.geocoder = Nominatim(user_agent="ai-restaurant-bot")
            logger.info("✅ Geocoder initialized")
        except Exception as e:
            logger.warning(f"⚠️ Geocoder initialization failed: {e}")
            self.geocoder = None

    # ======== SOURCE QUALITY METHODS ========

    def store_source_quality(self, destination: str, url: str, score: float) -> bool:
        """
        Store source quality information in the database

        Args:
            destination: The destination/city from the query
            url: The cleaned URL (main domain)
            score: Quality score from AI evaluation (0.0-1.0)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if this destination+url combination already exists
            existing = self.supabase.table('source_quality')\
                .select('*')\
                .eq('destination', destination)\
                .eq('url', url)\
                .execute()

            if existing.data:
                # Update existing record with new score (keep the highest score)
                current_score = existing.data[0].get('score', 0.0)
                new_score = max(current_score, score)

                self.supabase.table('source_quality')\
                    .update({
                        'score': new_score,
                        'last_updated': datetime.now(timezone.utc).isoformat(),
                        'mention_count': existing.data[0].get('mention_count', 0) + 1
                    })\
                    .eq('id', existing.data[0]['id'])\
                    .execute()

                logger.debug(f"📊 Updated source quality: {url} for {destination} (score: {new_score})")
            else:
                # Insert new record
                self.supabase.table('source_quality').insert({
                    'destination': destination,
                    'url': url,
                    'score': score,
                    'mention_count': 1,
                    'first_added': datetime.now(timezone.utc).isoformat(),
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }).execute()

                logger.debug(f"➕ Added new source quality: {url} for {destination} (score: {score})")

            return True

        except Exception as e:
            logger.error(f"❌ Error storing source quality: {e}")
            return False

    # Also add this convenience function at the bottom of utils/database.py

    def store_source_quality_data(destination: str, url: str, score: float) -> bool:
        """Store source quality data - convenience function"""
        return get_database().store_source_quality(destination, url, score)
    
    # ============ RESTAURANT METHODS ============

    def save_restaurant(self, restaurant_data: Dict[str, Any]) -> Optional[str]:
        """Save restaurant with new simplified schema - SAFER coordinate approach"""
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
                    combined_description = existing.get('raw_description', '') + "\n\n--- NEW MENTION ---\n\n" + new_description
                else:
                    combined_description = existing.get('raw_description', '')

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
                    'mention_count': existing.get('mention_count', 1) + 1,
                    'last_updated': datetime.now().isoformat(),
                    # Update address if we have a new one and existing is null
                    'address': restaurant_data.get('address') if existing.get('address') is None else existing.get('address'),
                    # Update country if we have a new one and existing is null
                    'country': restaurant_data.get('country') if existing.get('country') is None else existing.get('country')
                }

                # Handle coordinates safely
                if restaurant_data.get('coordinates'):
                    coords = restaurant_data['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        lat, lng = coords
                        update_data['latitude'] = lat
                        update_data['longitude'] = lng

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

                # Handle coordinates safely during insert
                if restaurant_data.get('coordinates'):
                    coords = restaurant_data['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        lat, lng = coords
                        insert_data['latitude'] = lat
                        insert_data['longitude'] = lng

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
            return None

    def _geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to get coordinates using the same approach as the older implementation"""
        try:
            if not self.geocoder:
                return None

            location = self.geocoder.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
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
        """Update restaurant with address and coordinates - SAFER approach"""
        try:
            lat, lng = coordinates

            # First update the address
            update_data = {
                'address': address,
                'last_updated': datetime.now().isoformat()
            }

            self.supabase.table('restaurants')\
                .update(update_data)\
                .eq('id', restaurant_id)\
                .execute()

            # Use RPC function for coordinates (safer than raw SQL)
            try:
                # Try using RPC first (if the function exists in your Supabase)
                self.supabase.rpc('update_restaurant_coordinates', {
                    'restaurant_id': restaurant_id,
                    'lat': lat,
                    'lng': lng
                }).execute()
                logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id} with coords ({lat}, {lng}) via RPC")
            except Exception as rpc_error:
                # Fallback: use the latitude/longitude columns approach like your old implementation
                logger.warning(f"RPC failed, using fallback coordinate storage: {rpc_error}")

                # Update with separate lat/lng columns (like your old working implementation)
                coord_update = {
                    'latitude': lat,
                    'longitude': lng
                }

                self.supabase.table('restaurants')\
                    .update(coord_update)\
                    .eq('id', restaurant_id)\
                    .execute()

                logger.info(f"📍 Updated geodata for restaurant ID: {restaurant_id} with coords ({lat}, {lng}) via lat/lng columns")

        except Exception as e:
            logger.error(f"Error updating geodata: {e}")

    def get_restaurants_by_preference_tags(self, city: str, preference_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Get restaurants that match any of the preference tags"""
        try:
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

    # ============ DOMAIN INTELLIGENCE METHODS ============

    # Add these updated methods to your Database class in utils/database.py

    # ============ SIMPLIFIED DOMAIN INTELLIGENCE METHODS ============

    def save_domain_intelligence(self, domain: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save domain intelligence data - SIMPLIFIED VERSION"""
        try:
            # For the simplified table, we use the update_domain_stats function instead
            # This method is kept for compatibility but delegates to the SQL function

            strategy = intelligence_data.get('strategy', 'enhanced_http')
            success_count = intelligence_data.get('success_count', 0)
            total_attempts = max(intelligence_data.get('total_attempts', 1), 1)

            # Calculate if this represents a success based on the data
            success_rate = success_count / total_attempts
            is_success = success_rate > 0.5  # Treat as success if > 50% success rate

            # Use the SQL function to update
            self.supabase.rpc('update_domain_stats', {
                'p_domain': domain,
                'p_strategy': strategy,
                'p_success': is_success
            }).execute()

            logger.debug(f"💾 Saved domain intelligence for {domain}")
            return True

        except Exception as e:
            logger.error(f"Error saving domain intelligence for {domain}: {e}")
            return False

    def get_domain_intelligence(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain intelligence data - SIMPLIFIED VERSION"""
        try:
            result = self.supabase.table('domain_intelligence')\
                .select('*')\
                .eq('domain', domain)\
                .execute()

            if result.data:
                # Convert to the format expected by the smart scraper
                data = result.data[0]
                return {
                    'domain': data['domain'],
                    'strategy': data['strategy'],
                    'confidence': data['confidence'],
                    'success_count': data['success_count'],
                    'total_attempts': data['total_attempts'],
                    'cost_per_scrape': data['cost_per_scrape'],
                    'created_at': data['created_at'],
                    'updated_at': data['updated_at']
                }

            return None

        except Exception as e:
            logger.error(f"Error getting domain intelligence for {domain}: {e}")
            return None

    def update_domain_success(self, domain: str, success: bool, restaurants_found: int = 0):
        """Update domain success/failure counts - SIMPLIFIED VERSION"""
        try:
            # Get current data to determine strategy
            current = self.get_domain_intelligence(domain)

            if current:
                strategy = current['strategy']
            else:
                # Default strategy for new domains
                strategy = 'enhanced_http'

            # Use the SQL function to update
            self.supabase.rpc('update_domain_stats', {
                'p_domain': domain,
                'p_strategy': strategy,
                'p_success': success
            }).execute()

            logger.debug(f"Updated domain intelligence for {domain}: success={success}")

        except Exception as e:
            logger.error(f"Error updating domain success for {domain}: {e}")

    def get_trusted_domains(self, min_confidence: float = None) -> List[str]:
        """Get list of trusted domains based on success rate - SIMPLIFIED VERSION"""
        try:
            min_conf = min_confidence or 0.7

            result = self.supabase.table('domain_intelligence')\
                .select('domain')\
                .gte('confidence', min_conf)\
                .gte('total_attempts', 2)\
                .execute()

            return [row['domain'] for row in result.data]

        except Exception as e:
            logger.error(f"Error getting trusted domains: {e}")
            return []

    def load_all_domain_intelligence(self) -> List[Dict[str, Any]]:
        """Load all domain intelligence records - SIMPLIFIED VERSION"""
        try:
            result = self.supabase.table('domain_intelligence')\
                .select('*')\
                .order('confidence', desc=True)\
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Error loading domain intelligence: {e}")
            return []

    def get_domain_intelligence_stats(self) -> Dict[str, Any]:
        """Get domain intelligence statistics"""
        try:
            # Get all domain data
            all_domains = self.load_all_domain_intelligence()

            if not all_domains:
                return {
                    'total_domains': 0,
                    'strategy_breakdown': {},
                    'average_confidence': 0,
                    'high_confidence_domains': 0
                }

            # Calculate statistics
            strategy_counts = {}
            total_confidence = 0
            high_confidence_count = 0

            for domain in all_domains:
                strategy = domain.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                confidence = domain.get('confidence', 0)
                total_confidence += confidence

                if confidence >= 0.8:
                    high_confidence_count += 1

            return {
                'total_domains': len(all_domains),
                'strategy_breakdown': strategy_counts,
                'average_confidence': round(total_confidence / len(all_domains), 2),
                'high_confidence_domains': high_confidence_count,
                'confidence_rate': round((high_confidence_count / len(all_domains)) * 100, 1)
            }

        except Exception as e:
            logger.error(f"Error getting domain intelligence stats: {e}")
            return {
                'total_domains': 0,
                'strategy_breakdown': {},
                'average_confidence': 0,
                'high_confidence_domains': 0
            }
            
    # ============ STATISTICS AND MONITORING ============

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring - FIXED VERSION"""
        try:
            stats = {}

            # Count restaurants
            restaurants_count = self.supabase.table('restaurants').select('id', count='exact').execute()
            stats['total_restaurants'] = restaurants_count.count

            # Count domains
            domains_count = self.supabase.table('domain_intelligence').select('domain', count='exact').execute()
            stats['total_domains'] = domains_count.count

            # FIXED: Remove group_by which doesn't exist in current Supabase client
            # Get cities using a simple approach
            try:
                cities_result = self.supabase.table('restaurants')\
                    .select('city')\
                    .execute()

                # Count cities manually
                city_counts = {}
                for row in cities_result.data:
                    city = row.get('city', 'Unknown')
                    city_counts[city] = city_counts.get(city, 0) + 1

                # Sort by count and take top 10
                top_cities = sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                stats['top_cities'] = [{'city': city, 'count': count} for city, count in top_cities]

            except Exception as e:
                logger.warning(f"Could not get city stats: {e}")
                stats['top_cities'] = []

            # Coordinate coverage
            with_coords = self.supabase.table('restaurants')\
                .select('id', count='exact')\
                .not_.is_('coordinates', 'null')\
                .execute()

            stats['coordinate_coverage'] = {
                'with_coordinates': with_coords.count,
                'without_coordinates': restaurants_count.count - with_coords.count,
                'coverage_percentage': round((with_coords.count / restaurants_count.count * 100), 1) if restaurants_count.count > 0 else 0
            }

            logger.info(f"📊 Database stats: {stats['total_restaurants']} restaurants, {stats['coordinate_coverage']['coverage_percentage']}% have coordinates")
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_restaurants': 0,
                'total_domains': 0,
                'top_cities': [],
                'coordinate_coverage': {'coverage_percentage': 0}
            }

    # ============ SIMPLE CACHE (OPTIONAL) ============

    def cache_search_results(self, query: str, results: Dict[str, Any]) -> bool:
        """Simple in-memory cache (placeholder for future implementation)"""
        # For now, just log that we would cache
        logger.debug(f"Would cache search results for: {query}")
        return True

    def get_cached_results(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search results (placeholder)"""
        # For now, always return None to force fresh searches
        return None

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

# ============ GLOBAL DATABASE INSTANCE ============

_database = None

def initialize_database(config):
    """Initialize global database instance"""
    global _database

    if _database is not None:
        return  # Already initialized

    try:
        _database = Database(config)

        # Log connection success with stats
        stats = _database.get_database_stats()
        logger.info(f"✅ Database initialized: {stats['total_restaurants']} restaurants, {stats['coordinate_coverage']['coverage_percentage']}% have coordinates")

    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        raise

def get_database() -> Database:
    """Get the global database instance"""
    if _database is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _database

# ============ CONVENIENCE FUNCTIONS ============
# These maintain compatibility with existing code

def save_restaurant_data(restaurant_data: Dict[str, Any]) -> Optional[str]:
    """Save restaurant data"""
    return get_database().save_restaurant(restaurant_data)

def get_restaurants_by_city(city: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get restaurants for a city"""
    return get_database().get_restaurants_by_city(city, limit)

def search_restaurants_by_cuisine(city: str, cuisine_tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Search restaurants by cuisine tags"""
    return get_database().search_restaurants_by_cuisine(city, cuisine_tags, limit)

def save_domain_intelligence(domain: str, intelligence_data: Dict[str, Any]) -> bool:
    """Save domain intelligence data"""
    return get_database().save_domain_intelligence(domain, intelligence_data)

def get_domain_intelligence(domain: str) -> Optional[Dict[str, Any]]:
    """Get domain intelligence data"""
    return get_database().get_domain_intelligence(domain)

def update_domain_success(domain: str, success: bool, restaurants_found: int = 0):
    """Update domain success metrics"""
    get_database().update_domain_success(domain, success, restaurants_found)

def get_trusted_domains(min_confidence: float = None) -> List[str]:
    """Get list of trusted domains"""
    return get_database().get_trusted_domains(min_confidence)

def load_all_domain_intelligence() -> List[Dict[str, Any]]:
    """Load all domain intelligence records"""
    return get_database().load_all_domain_intelligence()

def cache_search_results(query: str, results: Dict[str, Any]) -> bool:
    """Cache search results"""
    return get_database().cache_search_results(query, results)

def get_cached_results(query: str) -> Optional[Dict[str, Any]]:
    """Get cached search results"""
    return get_database().get_cached_results(query)

# ============ LEGACY COMPATIBILITY STUBS ============
# These prevent errors but don't do anything with the simplified schema

def add_to_search_history(user_id: str, query: str, results_count: int = 0):
    """Legacy user history tracking (disabled in simplified schema)"""
    logger.debug(f"Search history tracking disabled: {user_id} searched for {query}")

def save_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """Legacy user preferences (disabled in simplified schema)"""
    logger.debug(f"User preferences disabled: {user_id}")
    return True

def get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    """Legacy user preferences (disabled in simplified schema)"""
    return None

# Deprecated RAG functions (return safe defaults)
def save_scraped_content(source_url: str, content: str, restaurant_mentions: Optional[List[str]] = None, source_domain: str = None) -> bool:
    """DEPRECATED: RAG disabled in simplified schema"""
    return True

def search_similar_content(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """DEPRECATED: Vector search disabled in simplified schema"""
    return []

# ============ BACKWARDS COMPATIBILITY FOR MAIN.PY ============

# For main.py to work without changes
def initialize_db(config):
    """Alias for initialize_database for main.py compatibility"""
    initialize_database(config)

def get_supabase_manager():
    """Alias for get_database for backwards compatibility"""
    return get_database()