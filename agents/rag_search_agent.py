# agents/rag_search_agent.py
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
    """

    def __init__(self, config):
        self.config = config
        self.stats = {
            "rag_queries": 0,
            "rag_hits": 0,
            "rag_misses": 0,
            "cache_hits": 0,
            "content_reused": 0,
            "queries_enhanced": 0
        }

    def search_knowledge_base(self, query: str, destination: str = None, limit: int = 5) -> Dict[str, Any]:
        """
        Search the RAG knowledge base for relevant content

        Args:
            query: User's restaurant query
            destination: City/location if known
            limit: Maximum number of content chunks to return

        Returns:
            Dict with relevant content and metadata
        """
        try:
            self.stats["rag_queries"] += 1

            # Try exact cache hit first (fastest)
            cached_results = self._check_exact_cache(query)
            if cached_results:
                self.stats["cache_hits"] += 1
                logger.info(f"🎯 Exact cache hit for: {query}")
                return {
                    "type": "exact_cache",
                    "confidence": 0.95,
                    "content": cached_results,
                    "source": "previous_search",
                    "reusable": True
                }

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

    def should_skip_web_search(self, query: str, destination: str = None) -> bool:
        """
        Determine if we have enough RAG content to skip web search entirely

        Args:
            query: User's query
            destination: Destination if known

        Returns:
            True if web search can be skipped
        """
        try:
            rag_results = self.search_knowledge_base(query, destination, limit=5)

            # Skip web search if we have high-confidence RAG results
            confidence = rag_results.get("confidence", 0.0)
            content_count = len(rag_results.get("content", []))

            # Threshold for skipping web search
            skip_threshold = 0.85
            min_content_pieces = 3

            should_skip = (confidence >= skip_threshold and content_count >= min_content_pieces)

            if should_skip:
                logger.info(f"⚡ Skipping web search - sufficient RAG content (confidence: {confidence:.2f}, pieces: {content_count})")

            return should_skip

        except Exception as e:
            logger.error(f"Error determining if web search should be skipped: {e}")
            return False

    def _check_exact_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check for exact cache hit of previous search results"""
        try:
            cached = get_cached_results(query)
            if cached and isinstance(cached, dict):
                # Check if cache is recent (within 24 hours)
                cache_timestamp = cached.get("timestamp", 0)
                age_hours = (time.time() - cache_timestamp) / 3600

                if age_hours < 24:  # Fresh cache
                    return cached.get("results", {})

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
            word_matches = sum(1 for word in query_words if word in content_text)
            relevance_score = word_matches / len(query_words) if query_words else 0

            processed_result = {
                **result,
                "relevance_score": relevance_score,
                "source_summary": self._generate_source_summary(result)
            }
            processed.append(processed_result)

        # Sort by similarity * relevance
        processed.sort(key=lambda x: (x.get("similarity", 0) * x.get("relevance_score", 0)), reverse=True)

        return processed[:5]  # Return top 5

    def _calculate_confidence(self, processed_content: List[Dict]) -> float:
        """Calculate confidence score for RAG results"""
        if not processed_content:
            return 0.0

        # Average similarity score weighted by relevance
        total_score = 0
        total_weight = 0

        for content in processed_content:
            similarity = content.get("similarity", 0)
            relevance = content.get("relevance_score", 0)
            weight = max(0.1, relevance)  # Minimum weight

            total_score += similarity * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_source_summary(self, result: Dict) -> str:
        """Generate a summary of the content source"""
        source_id = result.get("source_id", "")
        content_preview = result.get("content_text", "")[:100]

        # Try to extract publication or domain name from content
        if "timeout" in content_preview.lower():
            return "Time Out Guide"
        elif "eater" in content_preview.lower():
            return "Eater Restaurant Guide"
        elif "michelin" in content_preview.lower():
            return "Michelin Guide"
        else:
            return "Restaurant Guide"

    def _extract_restaurant_snippet(self, content: str, restaurant_name: str) -> Optional[str]:
        """Extract relevant snippet about a specific restaurant"""
        try:
            sentences = content.split('.')
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
            "enhancement_rate": (self.stats["queries_enhanced"] / total_queries) * 100
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "rag_queries": 0,
            "rag_hits": 0,
            "rag_misses": 0,
            "cache_hits": 0,
            "content_reused": 0,
            "queries_enhanced": 0
        }