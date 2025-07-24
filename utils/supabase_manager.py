# utils/supabase_manager.py
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from geopy.geocoders import Nominatim
import numpy as np

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages all Supabase operations for the restaurant bot"""

    def __init__(self, config):
        """Initialize SupabaseManager with all necessary components"""
        self.config = config

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("✅ Supabase client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise

        # Initialize geocoder
        try:
            self.geocoder = Nominatim(user_agent="restaurant-bot")
            logger.info("✅ Geocoder initialized")
        except Exception as e:
            logger.warning(f"⚠️ Geocoder initialization failed: {e}")
            self.geocoder = None

        # Initialize embedding model for RAG
        try:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info(f"✅ Embedding model loaded: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            logger.info("⚠️ RAG will use text search fallback")
            self.embedding_model = None

    def _fallback_text_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback text-based search when vector search fails"""
        try:
            result = self.supabase.table('content_chunks')\
                .select('*')\
                .ilike('content_text', f'%{query}%')\
                .limit(limit)\
                .execute()

            # Add fake similarity score for compatibility
            for item in result.data:
                item['similarity'] = 0.7

            logger.info(f"📝 Text search found {len(result.data)} results")
            return result.data

        except Exception as e:
            logger.error(f"Text search also failed: {e}")
            return []

    def search_similar_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity - ENHANCED"""
        try:
            if not hasattr(self, 'embedding_model') or not self.embedding_model:
                logger.warning("Embedding model not available - falling back to text search")
                return self._fallback_text_search(query, limit)

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search for similar content chunks using RPC
            result = self.supabase.rpc('match_content_chunks', {
                'query_embedding': query_embedding,
                'match_threshold': self.config.SIMILARITY_THRESHOLD,
                'match_count': limit
            }).execute()

            if result.data:
                logger.info(f"🔍 Vector search found {len(result.data)} results for: {query}")
                return result.data
            else:
                logger.info(f"🔍 No vector search results for: {query}")
                return []

        except Exception as e:
            logger.warning(f"Vector search failed: {e} - falling back to text search")
            return self._fallback_text_search(query, limit)

    # ============ DOMAIN INTELLIGENCE METHODS (keeping your existing system) ============

    def save_domain_intelligence(self, domain: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save or update domain intelligence data"""
        try:
            # Prepare data for upsert
            data = {
                'domain': domain,
                'complexity': intelligence_data.get('complexity', 'moderate'),
                'scraper_type': intelligence_data.get('scraper_type', 'basic'),
                'cost_per_scrape': intelligence_data.get('cost', 0.0),
                'confidence': intelligence_data.get('confidence', 0.5),
                'reasoning': intelligence_data.get('reasoning', ''),
                'last_updated_at': datetime.now(timezone.utc).isoformat(),
                'metadata': intelligence_data.get('metadata', {})
            }

            # Upsert (insert or update)
            result = self.supabase.table('domain_intelligence').upsert(data).execute()

            logger.info(f"Saved domain intelligence for {domain}")
            return True

        except Exception as e:
            logger.error(f"Error saving domain intelligence for {domain}: {e}")
            return False

    def get_domain_intelligence(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain intelligence data"""
        try:
            result = self.supabase.table('domain_intelligence').select('*').eq('domain', domain).execute()

            if result.data:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error getting domain intelligence for {domain}: {e}")
            return None

    def update_domain_success(self, domain: str, success: bool, restaurants_found: int = 0):
        """Update domain success/failure counts"""
        try:
            # Get current data
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
                    'last_successful_scrape': datetime.utcnow().isoformat()
                }
            else:
                new_success = current['success_count'] 
                new_failure = current['failure_count'] + 1
                update_data = {
                    'failure_count': new_failure
                }

                # Block domain if too many failures
                if new_failure >= self.config.DOMAIN_FAILURE_LIMIT:
                    update_data.update({
                        'is_blocked': True,
                        'blocked_at': datetime.now(timezone.utc).isoformat()  # FIXED
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
            min_conf = min_confidence or self.config.DOMAIN_SUCCESS_THRESHOLD

            result = self.supabase.table('domain_intelligence')\
                .select('domain')\
                .gte('confidence', min_conf)\
                .eq('is_blocked', False)\
                .execute()

            return [row['domain'] for row in result.data]

        except Exception as e:
            logger.error(f"Error getting trusted domains: {e}")
            return []

    # ============ RESTAURANT DATA METHODS ============

    def save_restaurant(self, restaurant_data: Dict[str, Any]) -> Optional[str]:
        """Save restaurant data with automatic geocoding"""
        try:
            # Geocode address if coordinates not provided
            if not restaurant_data.get('latitude') and restaurant_data.get('address'):
                coords = self._geocode_address(restaurant_data['address'])
                if coords:
                    restaurant_data['latitude'], restaurant_data['longitude'] = coords

            # Check if restaurant already exists
            existing = self._find_existing_restaurant(restaurant_data)

            if existing:
                # Update existing restaurant
                restaurant_id = existing['id']
                update_data = {
                    'total_mentions': existing['total_mentions'] + 1,
                    'last_updated_at': datetime.now(timezone.utc).isoformat()  # FIXED
                }

                # Update credibility if this is from a professional source
                if restaurant_data.get('is_professional', False):
                    update_data['professional_mentions'] = existing['professional_mentions'] + 1
                    # Recalculate credibility score
                    update_data['credibility_score'] = min(1.0, 
                        existing['credibility_score'] + 0.1)

                self.supabase.table('restaurants').update(update_data).eq('id', restaurant_id).execute()
                logger.info(f"Updated existing restaurant: {restaurant_data.get('name')}")

            else:
                # Insert new restaurant
                insert_data = {
                    'name': restaurant_data['name'],
                    'address': restaurant_data.get('address'),
                    'latitude': restaurant_data.get('latitude'),
                    'longitude': restaurant_data.get('longitude'),
                    'neighborhood': restaurant_data.get('neighborhood'),
                    'city': restaurant_data.get('city'),
                    'country': restaurant_data.get('country'),
                    'cuisine_type': restaurant_data.get('cuisine_type'),
                    'phone': restaurant_data.get('phone'),
                    'website': restaurant_data.get('website'),
                    'credibility_score': restaurant_data.get('credibility_score', self.config.DEFAULT_CREDIBILITY_SCORE),
                    'professional_mentions': 1 if restaurant_data.get('is_professional') else 0,
                    'metadata': restaurant_data.get('metadata', {})
                }

                result = self.supabase.table('restaurants').insert(insert_data).execute()
                restaurant_id = result.data[0]['id']
                logger.info(f"Saved new restaurant: {restaurant_data.get('name')}")

            return restaurant_id

        except Exception as e:
            logger.error(f"Error saving restaurant: {e}")
            return None

    def _find_existing_restaurant(self, restaurant_data: Dict[str, Any]) -> Optional[Dict]:
        """Find existing restaurant by name and location"""
        try:
            # Search by name and city first
            result = self.supabase.table('restaurants')\
                .select('*')\
                .ilike('name', f"%{restaurant_data['name']}%")\
                .eq('city', restaurant_data.get('city', ''))\
                .execute()

            # TODO: Add more sophisticated matching (fuzzy matching, distance-based)
            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error finding existing restaurant: {e}")
            return None

    def _geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to get coordinates"""
        try:
            location = self.geocoder.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
            return None

    # ============ CONTENT STORAGE FOR RAG ============

    async def save_scraped_content(self, source_url: str, content: str, restaurant_mentions: List[str] = None, source_domain: str = None) -> bool:
        """Save scraped content with AI-powered metadata extraction"""
        try:
            logger.info(f"🔄 Saving content for RAG: {source_url} ({len(content)} chars)")

            # Check if embedding model is available
            if not hasattr(self, 'embedding_model') or not self.embedding_model:
                logger.error("❌ Embedding model not available - cannot save content for RAG")
                return False

            # First save/get the source
            source_id = self._save_source(source_url, source_domain)
            if not source_id:
                logger.error(f"❌ Failed to save source: {source_url}")
                return False

            # Split content into chunks
            chunks = self._split_content_into_chunks(content)
            logger.info(f"📄 Split into {len(chunks)} chunks")

            saved_chunks = 0
            for i, chunk_text in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk_text).tolist()

                    # AI-powered metadata extraction
                    if restaurant_mentions:
                        restaurants_mentioned = restaurant_mentions
                        neighborhood_tags = []
                        cuisine_tags = []
                    else:
                        # Use AI to extract metadata
                        restaurants_mentioned = await self._extract_restaurant_names(chunk_text)
                        neighborhood_tags = await self._extract_neighborhoods(chunk_text)
                        cuisine_tags = await self._extract_cuisines(chunk_text)

                    # Save chunk with embedding and AI-extracted metadata
                    chunk_data = {
                        'source_id': source_id,
                        'content_text': chunk_text,
                        'embedding': embedding,
                        'restaurants_mentioned': restaurants_mentioned,
                        'neighborhood_tags': neighborhood_tags,
                        'cuisine_tags': cuisine_tags,
                        'chunk_position': i,
                        'word_count': len(chunk_text.split())
                    }

                    result = self.supabase.table('content_chunks').insert(chunk_data).execute()
                    if result.data:
                        saved_chunks += 1
                        logger.debug(f"✅ Saved chunk {i+1}/{len(chunks)} with {len(restaurants_mentioned)} restaurants")
                    else:
                        logger.warning(f"⚠️ Failed to save chunk {i+1}")

                except Exception as e:
                    logger.error(f"❌ Error saving chunk {i}: {e}")
                    continue

            logger.info(f"✅ Saved {saved_chunks}/{len(chunks)} content chunks for {source_url}")
            return saved_chunks > 0

        except Exception as e:
            logger.error(f"❌ Error saving scraped content: {e}")
            return False

    async def _extract_restaurant_names(self, text: str) -> List[str]:
        """Extract restaurant names using AI"""
        try:
            from utils.unified_model_manager import get_unified_model_manager

            # Use your existing model manager
            model_manager = get_unified_model_manager(self.config)

            prompt = f"""Extract restaurant names from this text. Return only a JSON list of restaurant names, nothing else.

    Text: {text[:1000]}

    Return format: ["Restaurant Name 1", "Restaurant Name 2"]
    Maximum 5 restaurants. Return empty list [] if no restaurants found."""

            response = await model_manager.rate_limited_call(
                'metadata_extraction',  # Use fast model for this task
                prompt
            )

            # Parse the JSON response
            import json
            try:
                result = json.loads(response.content.strip())
                if isinstance(result, list):
                    return [name for name in result if isinstance(name, str) and len(name) > 2][:5]
            except json.JSONDecodeError:
                # Fallback: extract from response text
                import re
                matches = re.findall(r'"([^"]+)"', response.content)
                return [name for name in matches if len(name) > 2][:5]

            return []

        except Exception as e:
            logger.warning(f"AI restaurant extraction failed: {e}")
            return []

    async def _extract_neighborhoods(self, text: str) -> List[str]:
        """Extract neighborhoods using AI"""
        try:
            from utils.unified_model_manager import get_unified_model_manager

            model_manager = get_unified_model_manager(self.config)

            prompt = f"""Extract neighborhood or district names from this restaurant text. Return only a JSON list.

    Text: {text[:800]}

    Return format: ["Neighborhood 1", "District 2"]
    Maximum 3 locations. Return empty list [] if no neighborhoods found."""

            response = await model_manager.rate_limited_call(
                'metadata_extraction',
                prompt
            )

            import json
            try:
                result = json.loads(response.content.strip())
                if isinstance(result, list):
                    return [name for name in result if isinstance(name, str) and len(name) > 2][:3]
            except json.JSONDecodeError:
                import re
                matches = re.findall(r'"([^"]+)"', response.content)
                return [name for name in matches if len(name) > 2][:3]

            return []

        except Exception as e:
            logger.warning(f"AI neighborhood extraction failed: {e}")
            return []

    async def _extract_cuisines(self, text: str) -> List[str]:
        """Extract cuisine types using AI"""
        try:
            from utils.unified_model_manager import get_unified_model_manager

            model_manager = get_unified_model_manager(self.config)

            prompt = f"""Extract cuisine types from this restaurant text. Return only a JSON list.

    Text: {text[:800]}

    Return format: ["French", "Italian", "Seafood"]
    Maximum 3 cuisines. Return empty list [] if no cuisine types found."""

            response = await model_manager.rate_limited_call(
                'metadata_extraction',
                prompt
            )

            import json
            try:
                result = json.loads(response.content.strip())
                if isinstance(result, list):
                    return [name for name in result if isinstance(name, str) and len(name) > 1][:3]
            except json.JSONDecodeError:
                import re
                matches = re.findall(r'"([^"]+)"', response.content)
                return [name for name in matches if len(name) > 1][:3]

            return []

        except Exception as e:
            logger.warning(f"AI cuisine extraction failed: {e}")
            return []
    
    def _save_source(self, source_url: str, source_domain: str = None) -> Optional[str]:
        """Save source information - ENHANCED VERSION"""
        try:
            # Extract domain for domain intelligence lookup
            from urllib.parse import urlparse
            domain = urlparse(source_url).netloc

            # Use provided domain or extract from URL
            domain_name = source_domain or domain

            # Get domain intelligence
            domain_info = self.get_domain_intelligence(domain)

            source_data = {
                'domain': domain,
                'source_name': domain_name,
                'source_url': source_url,
                'source_type': 'guide',
                'credibility_rating': domain_info.get('confidence', 0.5) if domain_info else 0.5
            }

            # Check if source already exists
            existing = self.supabase.table('sources').select('id').eq('source_url', source_url).execute()

            if existing.data:
                return existing.data[0]['id']

            # Insert new source
            result = self.supabase.table('sources').insert(source_data).execute()
            return result.data[0]['id']

        except Exception as e:
            logger.error(f"Error saving source: {e}")
            return None

    def _split_content_into_chunks(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        max_length = self.config.CHUNK_MAX_LENGTH
        overlap = self.config.CHUNK_OVERLAP

        words = content.split()
        chunks = []

        for i in range(0, len(words), max_length - overlap):
            chunk_words = words[i:i + max_length]
            chunk_text = ' '.join(chunk_words)
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk_text)

        return chunks

    # ============ SEARCH AND RETRIEVAL ============


    def cache_search_results(self, query: str, results: Dict[str, Any]) -> bool:
        """
        Cache search results with proper duplicate handling

        FIXED: Use explicit check + insert/update instead of upsert to avoid constraint violations
        """
        try:
            # Create query hash
            query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()

            # Check if cache entry already exists
            existing = self.supabase.table('search_cache')\
                .select('id, usage_count')\
                .eq('query_hash', query_hash)\
                .execute()

            cache_data = {
                'query_hash': query_hash,
                'original_query': query,
                'normalized_query': query.lower().strip(),
                'results_json': results,
                'expires_at': (datetime.now(timezone.utc) + timedelta(days=self.config.CACHE_EXPIRY_DAYS)).isoformat(),  # FIXED
                'created_at': datetime.now(timezone.utc).isoformat()  # FIXED
            }

            if existing.data:
                # Update existing entry
                cache_id = existing.data[0]['id']
                current_usage = existing.data[0].get('usage_count', 0)

                cache_data['usage_count'] = current_usage + 1
                cache_data['updated_at'] = datetime.now(timezone.utc).isoformat()

                result = self.supabase.table('search_cache')\
                    .update(cache_data)\
                    .eq('id', cache_id)\
                    .execute()

                logger.info(f"Updated cached search results for query: {query}")
            else:
                # Insert new entry
                cache_data['usage_count'] = 1

                result = self.supabase.table('search_cache')\
                    .insert(cache_data)\
                    .execute()

                logger.info(f"Cached new search results for query: {query}")

            return True

        except Exception as e:
            logger.error(f"Error caching search results: {e}")
            return False

    def get_cached_results(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search results if they exist and haven't expired"""
        try:
            query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()

            result = self.supabase.table('search_cache')\
                .select('*')\
                .eq('query_hash', query_hash)\
                .gt('expires_at', datetime.now(timezone.utc).isoformat())\
                .execute()

            if result.data:
                # Update usage count
                cache_entry = result.data[0]
                self.supabase.table('search_cache')\
                    .update({'usage_count': cache_entry['usage_count'] + 1})\
                    .eq('id', cache_entry['id'])\
                    .execute()

                logger.info(f"Found cached results for query: {query}")
                return cache_entry['results_json']

            return None

        except Exception as e:
            logger.error(f"Error getting cached results: {e}")
            return None

    # ============ USER PREFERENCES ============

    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Save user preferences"""
        try:
            data = {
                'user_id': str(user_id),
                'preferences': preferences,
                'last_active_at': datetime.now(timezone.utc).isoformat()  # FIXED
            }

            self.supabase.table('user_preferences').upsert(data).execute()

            logger.info(f"Saved preferences for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
            return False

    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences"""
        try:
            result = self.supabase.table('user_preferences')\
                .select('preferences')\
                .eq('user_id', str(user_id))\
                .execute()

            if result.data:
                return result.data[0]['preferences']
            return None

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None

    def add_to_search_history(self, user_id: str, query: str, results_count: int = 0):
        """Add search to user's history"""
        try:
            # Get current user data
            current = self.supabase.table('user_preferences')\
                .select('search_history')\
                .eq('user_id', str(user_id))\
                .execute()

            # Prepare new search entry
            search_entry = {
                'query': query,
                'timestamp': datetime.now(timezone.utc).isoformat(),  # FIXED
                'results_count': results_count
            }

            if current.data:
                # Update existing history
                history = current.data[0].get('search_history', [])
                history.append(search_entry)

                # Keep only last 50 searches
                history = history[-50:]

                self.supabase.table('user_preferences')\
                .update({
                    'search_history': history,
                    'last_active_at': datetime.now(timezone.utc).isoformat()  # FIXED
                })\
                .eq('user_id', str(user_id))\
                .execute()
            else:
                # Create new user record
                self.supabase.table('user_preferences').insert({
                    'user_id': str(user_id),
                    'search_history': [search_entry],
                    'preferences': {},
                    'last_active_at': datetime.now(timezone.utc).isoformat()
                }).execute()

            logger.info(f"Added search to history for user {user_id}")

        except Exception as e:
            logger.error(f"Error adding to search history: {e}")

    # ============ ANALYTICS AND MAINTENANCE ============

    def get_top_restaurants(self, city: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top-rated restaurants by credibility score"""
        try:
            query = self.supabase.table('restaurants')\
                .select('*')\
                .gte('credibility_score', self.config.MIN_CREDIBILITY_FOR_RECOMMENDATION)\
                .order('credibility_score', desc=True)\
                .limit(limit)

            if city:
                query = query.eq('city', city)

            result = query.execute()
            return result.data

        except Exception as e:
            logger.error(f"Error getting top restaurants: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            stats = {}

            # Count restaurants
            restaurants_count = self.supabase.table('restaurants').select('id', count='exact').execute()
            stats['total_restaurants'] = restaurants_count.count

            # Count sources
            sources_count = self.supabase.table('sources').select('id', count='exact').execute()
            stats['total_sources'] = sources_count.count

            # Count content chunks
            chunks_count = self.supabase.table('content_chunks').select('id', count='exact').execute()
            stats['total_content_chunks'] = chunks_count.count

            # Count domains
            domains_count = self.supabase.table('domain_intelligence').select('domain', count='exact').execute()
            stats['total_domains'] = domains_count.count

            # Get top cities
            top_cities = self.supabase.table('restaurants')\
                .select('city', count='exact')\
                .group_by('city')\
                .order('count', desc=True)\
                .limit(10)\
                .execute()
            stats['top_cities'] = top_cities.data

            # Cache hit rate
            cache_total = self.supabase.table('search_cache').select('usage_count', count='exact').execute()
            if cache_total.data:
                total_usage = sum(row['usage_count'] for row in cache_total.data)
                stats['cache_entries'] = cache_total.count
                stats['total_cache_usage'] = total_usage

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        try:
            result = self.supabase.table('search_cache')\
                .delete()\
                .lt('expires_at', datetime.now(timezone.utc).isoformat())\
                .execute()

            logger.info(f"Cleaned up {len(result.data)} expired cache entries")

        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

    # ============ VECTOR SEARCH HELPER FUNCTION ============

    def create_vector_search_function(self):
        """Check if vector search function exists (function should be created manually in SQL Editor)"""
        try:
            # Test if the function exists by trying to call it with dummy data
            test_embedding = [0.0] * 384  # 384-dimensional zero vector
            result = self.supabase.rpc('match_content_chunks', {
                'query_embedding': test_embedding,
                'match_threshold': 0.9,  # High threshold so no results expected
                'match_count': 1
            }).execute()

            logger.info("Vector search function exists and is working")
            return True

        except Exception as e:
            logger.info(f"Vector search function not available yet: {e}")
            logger.info("Vector search function should be created manually in Supabase SQL Editor")
            # Don't fail initialization
            return False