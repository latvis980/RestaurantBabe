# agents/rag_search_agent.py
# IMPROVED VERSION - Better cache logic to save API calls
import logging
import time
from typing import Dict, List, Any, Optional
from utils.database import search_similar_content, get_cached_results

logger = logging.getLogger(__name__)

class RAGSearchAgent:
    """
    RAG (Retrieval-Augmented Generation) Search Agent

    Uses stored content to enhance searches and provide context from previous scraping.
    This makes your bot smarter over time as it accumulates knowledge.

    IMPROVED: Better cache logic to avoid unnecessary web searches and API calls.
    """

    def __init__(self, config):
        self.config = config
        self.stats = {
            "rag_queries": 0,
            "rag_hits": 0,
            "rag_misses": 0,
            "cache_hits": 0,
            "content_reused": 0,
            "queries_enhanced": 0,
            "web_searches_skipped": 0  # NEW: Track web searches we saved
        }

        # Cache the last check to avoid redundant calls
        self._last_cache_check = None
        self._last_query = None

    def should_skip_web_search(self, query: str, destination: str = None) -> bool:
        """
        Determine if we have enough RAG content to skip web search entirely

        IMPROVED: Check exact cache hits first (highest priority)

        Args:
            query: User's query
            destination: Destination if known

        Returns:
            True if web search can be skipped
        """
        try:
            # ============ PRIORITY 1: EXACT CACHE HIT ============
            # This is the most reliable source - 100% confidence
            cached_results = self._check_exact_cache(query)
            if cached_results:
                self.stats["cache_hits"] += 1
                self.stats["web_searches_skipped"] += 1
                self._last_cache_check = {
                    "type": "exact_cache",
                    "data": cached_results,
                    "confidence": 1.0
                }
                self._last_query = query

                logger.info(f"⚡ SKIPPING WEB SEARCH - Exact cache hit for: {query}")
                return True

            # ============ PRIORITY 2: SEMANTIC RAG SEARCH ============
            # Only check similarity search if no exact cache hit
            rag_results = self.search_knowledge_base(query, destination, limit=5)

            # Skip web search if we have high-confidence RAG results
            confidence = rag_results.get("confidence", 0.0)
            content_count = len(rag_results.get("content", []))

            # Threshold for skipping web search
            skip_threshold = 0.85
            min_content_pieces = 3

            should_skip = (confidence >= skip_threshold and content_count >= min_content_pieces)

            if should_skip:
                self.stats["web_searches_skipped"] += 1
                self._last_cache_check = {
                    "type": "semantic_search",
                    "data": rag_results,
                    "confidence": confidence
                }
                self._last_query = query

                logger.info(f"⚡ SKIPPING WEB SEARCH - High-confidence RAG content (confidence: {confidence:.2f}, pieces: {content_count})")
                return True

            # Reset cache if we're proceeding with web search
            self._last_cache_check = None
            self._last_query = None

            logger.info(f"🔍 PROCEEDING WITH WEB SEARCH - Insufficient RAG content (confidence: {confidence:.2f}, pieces: {content_count})")
            return False

        except Exception as e:
            logger.error(f"Error determining if web search should be skipped: {e}")
            return False

    def search_knowledge_base(self, query: str, destination: str = None, limit: int = 5) -> Dict[str, Any]:
        """
        Search the RAG knowledge base for relevant content

        IMPROVED: Use cached results if available from recent should_skip_web_search call

        Args:
            query: User's restaurant query
            destination: City/location if known
            limit: Maximum number of content chunks to return

        Returns:
            Dict with relevant content and metadata
        """
        try:
            self.stats["rag_queries"] += 1

            # ============ OPTIMIZATION: USE RECENT CACHE CHECK ============
            # If we just checked this query in should_skip_web_search, reuse the results
            if (self._last_query == query and 
                self._last_cache_check and 
                self._last_cache_check["type"] == "exact_cache"):

                cached_data = self._last_cache_check["data"]
                logger.info(f"🔄 Reusing exact cache check for: {query}")

                return {
                    "type": "exact_cache",
                    "confidence": 0.95,
                    "content": self._convert_cache_to_content_format(cached_data),
                    "source": "previous_search",
                    "reusable": True
                }

            # ============ FRESH EXACT CACHE CHECK ============
            # Try exact cache hit first (fastest)
            cached_results = self._check_exact_cache(query)
            if cached_results:
                self.stats["cache_hits"] += 1
                logger.info(f"🎯 Exact cache hit for: {query}")
                return {
                    "type": "exact_cache",
                    "confidence": 0.95,
                    "content": self._convert_cache_to_content_format(cached_results),
                    "source": "previous_search",
                    "reusable": True
                }

            # ============ SEMANTIC SEARCH ============
            # Build semantic search queries
            search_queries = self._build_rag_queries(query, destination)

            all_relevant_content = []
            for search_query in search_queries:
                try:
                    similar_content = search_similar_content(search_query, limit=limit)
                    if similar_content:
                        all_relevant_content.extend(similar_content)
                        logger.info(f"🔍 Found {len(similar_content)} chunks for: {search_query}")
                except Exception as e:
                    logger.warning(f"Error searching for '{search_query}': {e}")

            if all_relevant_content:
                self.stats["rag_hits"] += 1

                # Process and rank the content
                processed_content = self._process_rag_results(all_relevant_content, query)

                logger.info(f"✅ RAG search found {len(processed_content)} relevant pieces")
                return {
                    "type": "semantic_search",
                    "confidence": self._calculate_confidence(processed_content),
                    "content": processed_content,
                    "source": "knowledge_base",
                    "reusable": True,
                    "total_chunks": len(all_relevant_content)
                }
            else:
                self.stats["rag_misses"] += 1
                logger.info(f"❌ No RAG content found for: {query}")
                return {
                    "type": "no_results",
                    "confidence": 0.0,
                    "content": [],
                    "source": "none",
                    "reusable": False
                }

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            self.stats["rag_misses"] += 1
            return {
                "type": "error",
                "confidence": 0.0,
                "content": [],
                "source": "error",
                "reusable": False
            }

    def _convert_cache_to_content_format(self, cached_results: Dict) -> List[Dict]:
        """
        Convert cached search results to RAG content format

        This bridges the gap between cached search results and RAG content format
        """
        try:
            if not cached_results or not isinstance(cached_results, dict):
                return []

            # Extract restaurant data from cached results
            restaurants = cached_results.get("main_list", [])
            if not restaurants:
                return []

            # Convert restaurant data to content chunks
            content_chunks = []
            for i, restaurant in enumerate(restaurants[:10]):  # Limit to 10 restaurants

                # Build content text from restaurant data
                content_parts = []

                name = restaurant.get("name", "")
                if name:
                    content_parts.append(f"Restaurant: {name}")

                address = restaurant.get("address", "")
                if address:
                    content_parts.append(f"Address: {address}")

                cuisine = restaurant.get("cuisine", "")
                if cuisine:
                    content_parts.append(f"Cuisine: {cuisine}")

                description = restaurant.get("description", "")
                if description:
                    content_parts.append(f"Description: {description}")

                highlights = restaurant.get("highlights", [])
                if highlights and isinstance(highlights, list):
                    content_parts.append(f"Highlights: {', '.join(highlights[:3])}")  # First 3 highlights

                content_text = " | ".join(content_parts)

                if content_text:
                    content_chunks.append({
                        "id": f"cache_{i}",
                        "content_text": content_text,
                        "source_summary": f"Cached: {name}",
                        "similarity": 0.95,  # High similarity since it's an exact cache hit
                        "restaurant_data": restaurant  # Include original restaurant data
                    })

            logger.info(f"💾 Converted {len(content_chunks)} cached restaurants to content format")
            return content_chunks

        except Exception as e:
            logger.error(f"Error converting cache to content format: {e}")
            return []

    def enhance_web_search_with_rag(self, web_results: List[Dict], query: str, destination: str = None) -> List[Dict]:
        """
        Enhance web search results with RAG content

        Args:
            web_results: Results from web search
            query: Original query
            destination: Destination if known

        Returns:
            Enhanced results with RAG content mixed in
        """
        try:
            # Get relevant RAG content
            rag_results = self.search_knowledge_base(query, destination, limit=3)

            if not rag_results.get("reusable", False):
                return web_results

            rag_content = rag_results.get("content", [])
            if not rag_content:
                return web_results

            self.stats["queries_enhanced"] += 1

            # Convert RAG content to web result format
            rag_as_web_results = []
            for content in rag_content[:2]:  # Limit to top 2 RAG results
                rag_result = {
                    "title": f"💾 From Knowledge Base: {content.get('source_summary', 'Previous Search')}",
                    "url": f"rag://content/{content.get('id', 'unknown')}",
                    "description": content.get("content_text", "")[:200] + "...",
                    "scraped_content": content.get("content_text", ""),
                    "scraping_success": True,
                    "scraping_method": "rag_retrieval",
                    "source_info": {
                        "name": "Knowledge Base",
                        "url": "internal://rag",
                        "extraction_method": "vector_search"
                    },
                    "rag_source": True,
                    "rag_confidence": content.get("similarity", 0.8)
                }
                rag_as_web_results.append(rag_result)

            # Insert RAG results at the beginning (highest priority)
            enhanced_results = rag_as_web_results + web_results

            logger.info(f"🚀 Enhanced {len(web_results)} web results with {len(rag_as_web_results)} RAG results")
            self.stats["content_reused"] += len(rag_as_web_results)

            return enhanced_results

        except Exception as e:
            logger.error(f"Error enhancing web search with RAG: {e}")
            return web_results

    def get_restaurant_context(self, restaurant_name: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Get additional context about a specific restaurant from RAG

        Args:
            restaurant_name: Name of the restaurant
            destination: City/location

        Returns:
            Additional context if found
        """
        try:
            # Search for specific restaurant mentions
            restaurant_query = f"{restaurant_name} {destination} restaurant"
            similar_content = search_similar_content(restaurant_query, limit=3)

            if similar_content:
                # Process the content for restaurant-specific information
                context = {
                    "additional_info": [],
                    "mentions_count": len(similar_content),
                    "confidence": similar_content[0].get("similarity", 0.0) if similar_content else 0.0
                }

                for content in similar_content:
                    content_text = content.get("content_text", "")
                    # Extract relevant snippets that mention the restaurant
                    if restaurant_name.lower() in content_text.lower():
                        snippet = self._extract_restaurant_snippet(content_text, restaurant_name)
                        if snippet:
                            context["additional_info"].append({
                                "snippet": snippet,
                                "source": content.get("source_id", "unknown"),
                                "confidence": content.get("similarity", 0.0)
                            })

                return context if context["additional_info"] else None

            return None

        except Exception as e:
            logger.error(f"Error getting restaurant context for {restaurant_name}: {e}")
            return None

    def _check_exact_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check for exact cache hit of previous search results"""
        try:
            cached = get_cached_results(query)
            if cached and isinstance(cached, dict):
                # Check if cache is recent (within 24 hours)
                cache_timestamp = cached.get("timestamp", 0)
                age_hours = (time.time() - cache_timestamp) / 3600

                if age_hours < 24:  # Fresh cache
                    results = cached.get("results", {})
                    if results and results.get("main_list"):
                        logger.info(f"📋 Found cached results: {len(results.get('main_list', []))} restaurants")
                        return results

            return None
        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            return None

    def _build_rag_queries(self, query: str, destination: str = None) -> List[str]:
        """Build multiple search queries for RAG"""
        queries = [query]  # Original query

        if destination:
            # Add destination-specific variations
            queries.extend([
                f"restaurants in {destination}",
                f"{destination} dining guide",
                f"where to eat {destination}"
            ])

        # Add cuisine-specific queries if detected
        cuisine_keywords = ["italian", "french", "japanese", "chinese", "indian", "mexican", "thai"]
        query_lower = query.lower()

        for cuisine in cuisine_keywords:
            if cuisine in query_lower and destination:
                queries.append(f"{cuisine} restaurants {destination}")

        return list(set(queries))  # Remove duplicates

    def _process_rag_results(self, raw_results: List[Dict], query: str) -> List[Dict]:
        """Process and rank RAG results by relevance"""
        processed = []

        for result in raw_results:
            # Add query relevance score
            content_text = result.get("content_text", "").lower()
            query_words = query.lower().split()

            # Simple relevance scoring
            relevance_score = 0
            for word in query_words:
                if word in content_text:
                    relevance_score += 1

            result["relevance_score"] = relevance_score / len(query_words) if query_words else 0
            processed.append(result)

        # Sort by relevance and similarity
        processed.sort(key=lambda x: (x.get("similarity", 0) + x.get("relevance_score", 0)), reverse=True)

        return processed[:10]  # Return top 10 results

    def _calculate_confidence(self, content_list: List[Dict]) -> float:
        """Calculate overall confidence score for RAG content"""
        if not content_list:
            return 0.0

        # Average similarity score
        similarities = [content.get("similarity", 0) for content in content_list]
        avg_similarity = sum(similarities) / len(similarities)

        # Boost for quantity (more content = higher confidence)
        quantity_boost = min(0.1 * len(content_list), 0.3)  # Max 0.3 boost

        return min(1.0, avg_similarity + quantity_boost)

    def _extract_restaurant_snippet(self, content_text: str, restaurant_name: str) -> Optional[str]:
        """Extract a relevant snippet mentioning the restaurant"""
        try:
            sentences = content_text.split('.')
            relevant_sentences = []

            for sentence in sentences:
                if restaurant_name.lower() in sentence.lower():
                    relevant_sentences.append(sentence.strip())

            if relevant_sentences:
                # Return first relevant sentence, limited to reasonable length
                snippet = relevant_sentences[0]
                return snippet[:200] + "..." if len(snippet) > 200 else snippet

            return None

        except Exception as e:
            logger.warning(f"Error extracting snippet: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG search statistics"""
        total_queries = max(1, self.stats["rag_queries"])

        return {
            **self.stats,
            "hit_rate": (self.stats["rag_hits"] / total_queries) * 100,
            "cache_hit_rate": (self.stats["cache_hits"] / total_queries) * 100,
            "enhancement_rate": (self.stats["queries_enhanced"] / total_queries) * 100,
            "web_search_savings": (self.stats["web_searches_skipped"] / total_queries) * 100  # NEW
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "rag_queries": 0,
            "rag_hits": 0,
            "rag_misses": 0,
            "cache_hits": 0,
            "content_reused": 0,
            "queries_enhanced": 0,
            "web_searches_skipped": 0
        }
        self._last_cache_check = None
        self._last_query = None