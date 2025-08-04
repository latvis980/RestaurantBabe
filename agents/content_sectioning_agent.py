# agents/content_sectioning_agent.py - ENHANCED FOR NEW SMART SCRAPER
"""
Enhanced Content Sectioning Agent optimized for the new smart scraper flow

IMPROVEMENTS:
1. ✅ Optimized for simple and enhanced HTTP scraping only
2. ✅ Better restaurant content detection
3. ✅ Improved sectioning results format
4. ✅ Enhanced performance with DeepSeek
5. ✅ Better integration with domain intelligence
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SectioningResult:
    """Enhanced result structure for content sectioning"""
    optimized_content: str
    original_length: int
    optimized_length: int
    sections_identified: List[str]
    restaurants_density: float
    restaurants_found: List[str]
    sectioning_method: str
    confidence: float
    processing_time: float = 0.0

class ContentSectioningAgent:
    """
    Enhanced Content Sectioning Agent for Smart Scraper Integration

    Optimized specifically for simple and enhanced HTTP scraping.
    Uses DeepSeek for ultra-fast restaurant content extraction.
    """

    def __init__(self, config):
        self.config = config

        # Use DeepSeek for fast content sectioning
        self.sectioner = ChatOpenAI(
            model=config.DEEPSEEK_CHAT_MODEL if hasattr(config, 'DEEPSEEK_CHAT_MODEL') else config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.DEEPSEEK_API_KEY if hasattr(config, 'DEEPSEEK_API_KEY') else config.OPENAI_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL if hasattr(config, 'DEEPSEEK_BASE_URL') else None
        )

        # Content cache for repeated URLs
        self._content_cache = {}
        self._cache_size_limit = 500

        # Performance statistics
        self.stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "avg_processing_time": 0.0,
            "avg_content_reduction": 0.0,
            "restaurants_detected": 0,
            "high_confidence_results": 0
        }

        # Enhanced sectioning prompt optimized for restaurant content
        self.sectioning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content analyzer specializing in restaurant and food content extraction.

Your task: Extract and optimize restaurant-related content from web pages.

FOCUS ON:
- Restaurant names, descriptions, and details
- Food and cuisine information  
- Dining experiences and reviews
- Chef information and specialties
- Menu items and recommendations
- Location and atmosphere details
- Price ranges and value information

IGNORE:
- Navigation menus and headers
- Advertisements and promotional content
- Comment sections and user-generated content
- Unrelated articles or sidebar content
- Technical website elements

OUTPUT FORMAT:
Return JSON with extracted restaurant sections:
{{
    "restaurant_sections": [
        {{
            "content": "extracted restaurant text...",
            "restaurant_names": ["Restaurant A", "Restaurant B"],
            "confidence": 0.9,
            "section_type": "restaurant_list|review|guide"
        }}
    ],
    "restaurants_found": ["Restaurant A", "Restaurant B", "Restaurant C"],
    "content_summary": "Brief description of content type",
    "restaurant_density": 0.8,
    "overall_confidence": 0.85
}}

Be selective - only include content with clear restaurant relevance."""),
            ("human", """Extract restaurant content from this web page ({content_length} chars):

URL: {url}
Content:
{content}

Focus on restaurant information and optimize for restaurant discovery.""")
        ])

    async def process_content(self, content: str, url: str = "", source_method: str = "unknown") -> SectioningResult:
        """
        Main content processing method optimized for restaurant extraction
        """
        start_time = time.time()

        # Quick pre-filtering
        if not self._has_restaurant_indicators(content):
            logger.debug(f"No restaurant indicators found in content from {url}")
            return self._create_empty_result(content, start_time)

        # Check cache
        cache_key = self._get_cache_key(content, url)
        if cache_key in self._content_cache:
            cached_result = self._content_cache[cache_key]
            self.stats["cache_hits"] += 1
            logger.debug(f"🚀 Cache hit for content sectioning: {url}")
            return cached_result

        # Limit content size for processing
        limited_content = self._smart_content_limiting(content)

        try:
            # Process with AI sectioning
            sectioned_result = await self._process_with_ai(limited_content, url)

            # Cache successful results
            if sectioned_result.confidence > 0.5:
                self._update_cache(cache_key, sectioned_result)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(sectioned_result, processing_time)

            logger.info(f"⚡ Content sectioning: {len(content)} → {len(sectioned_result.optimized_content)} chars in {processing_time:.1f}s (confidence: {sectioned_result.confidence:.2f})")

            return sectioned_result

        except Exception as e:
            logger.error(f"Content sectioning failed for {url}: {e}")
            return self._create_fallback_result(content, start_time, str(e))

    async def _process_with_ai(self, content: str, url: str) -> SectioningResult:
        """Process content using AI sectioning"""
        start_time = time.time()

        # Format prompt
        formatted_prompt = self.sectioning_prompt.format(
            content=content,
            content_length=len(content),
            url=url
        )

        # Make AI call
        response = await self.sectioner.ainvoke(formatted_prompt)
        response_content = response.content.strip()

        # Parse JSON response
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()

        try:
            result_data = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response for {url}: {e}")
            return self._create_fallback_result(content, start_time, "JSON parse error")

        # Extract and combine restaurant sections
        restaurant_sections = result_data.get("restaurant_sections", [])
        optimized_content = "\n\n".join([
            section["content"] for section in restaurant_sections 
            if section.get("confidence", 0) > 0.6
        ])

        # If no good sections found, use original content (truncated)
        if not optimized_content.strip():
            optimized_content = content[:2000] + "..." if len(content) > 2000 else content

        # Extract restaurant information
        restaurants_found = result_data.get("restaurants_found", [])
        restaurant_density = result_data.get("restaurant_density", 0.0)
        overall_confidence = result_data.get("overall_confidence", 0.5)

        # Identify section types
        sections_identified = []
        for section in restaurant_sections:
            section_type = section.get("section_type", "unknown")
            if section_type not in sections_identified:
                sections_identified.append(section_type)

        processing_time = time.time() - start_time

        return SectioningResult(
            optimized_content=optimized_content,
            original_length=len(content),
            optimized_length=len(optimized_content),
            sections_identified=sections_identified,
            restaurants_density=restaurant_density,
            restaurants_found=restaurants_found,
            sectioning_method="ai_sectioning",
            confidence=overall_confidence,
            processing_time=processing_time
        )

    def _has_restaurant_indicators(self, content: str) -> bool:
        """Enhanced restaurant content detection"""
        if not content or len(content) < 300:
            return False

        content_lower = content.lower()

        # Primary restaurant keywords (high weight)
        primary_keywords = [
            'restaurant', 'menu', 'chef', 'dining', 'cuisine', 'bistro', 
            'cafe', 'bar', 'eatery', 'food', 'dish', 'meal'
        ]

        # Secondary keywords (medium weight)
        secondary_keywords = [
            'taste', 'flavor', 'kitchen', 'cook', 'recipe', 'ingredient',
            'wine', 'drink', 'service', 'atmosphere', 'reservation'
        ]

        # Location/review keywords (low weight)
        location_keywords = [
            'address', 'location', 'hours', 'phone', 'review', 'rating',
            'michelin', 'zagat', 'yelp', 'guide', 'recommend'
        ]

        # Calculate weighted score
        primary_count = sum(1 for keyword in primary_keywords if keyword in content_lower)
        secondary_count = sum(1 for keyword in secondary_keywords if keyword in content_lower)
        location_count = sum(1 for keyword in location_keywords if keyword in content_lower)

        weighted_score = (primary_count * 3) + (secondary_count * 2) + (location_count * 1)

        # Dynamic threshold based on content length
        content_length_factor = min(len(content) / 1000, 3.0)  # Max factor of 3
        threshold = max(6, int(8 * content_length_factor))

        return weighted_score >= threshold

    def _smart_content_limiting(self, content: str) -> str:
        """Intelligently limit content while preserving restaurant information"""
        max_chars = getattr(self.config, 'CONTENT_SECTIONER_LIMIT', 8000)

        if len(content) <= max_chars:
            return content

        # Try to preserve restaurant-rich sections
        content_lower = content.lower()
        restaurant_keywords = ['restaurant', 'menu', 'chef', 'cuisine', 'dining', 'food']

        # Split content into chunks and score them
        chunk_size = max_chars // 4
        chunks = []

        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunk_lower = chunk.lower()

            # Score chunk based on restaurant keyword density
            keyword_count = sum(1 for keyword in restaurant_keywords if keyword in chunk_lower)
            score = keyword_count / len(chunk) * 1000  # Keywords per 1000 chars

            chunks.append((chunk, score, i))

        # Sort by score and take top chunks
        chunks.sort(key=lambda x: x[1], reverse=True)

        selected_content = ""
        for chunk, score, position in chunks:
            if len(selected_content) + len(chunk) <= max_chars:
                selected_content += chunk
            else:
                remaining_space = max_chars - len(selected_content)
                if remaining_space > 200:  # Only add if we have meaningful space left
                    selected_content += chunk[:remaining_space]
                break

        # If we couldn't build good content, fall back to simple truncation
        if len(selected_content) < max_chars * 0.5:
            selected_content = content[:max_chars]
            # Try to end at a sentence boundary
            last_period = selected_content.rfind('.')
            if last_period > max_chars * 0.8:
                selected_content = selected_content[:last_period + 1]

        return selected_content.strip()

    def _get_cache_key(self, content: str, url: str) -> str:
        """Generate cache key for content"""
        # Use first 1000 chars + URL for cache key
        content_sample = content[:1000] if len(content) > 1000 else content
        cache_string = f"{url}:{content_sample}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _update_cache(self, cache_key: str, result: SectioningResult):
        """Update content cache with size management"""
        self._content_cache[cache_key] = result

        # Simple cache size management
        if len(self._content_cache) > self._cache_size_limit:
            # Remove oldest 25% of entries
            keys_to_remove = list(self._content_cache.keys())[:self._cache_size_limit // 4]
            for key in keys_to_remove:
                del self._content_cache[key]

    def _create_empty_result(self, content: str, start_time: float) -> SectioningResult:
        """Create empty result for non-restaurant content"""
        processing_time = time.time() - start_time

        return SectioningResult(
            optimized_content="",
            original_length=len(content),
            optimized_length=0,
            sections_identified=["no_restaurant_content"],
            restaurants_density=0.0,
            restaurants_found=[],
            sectioning_method="pre_filtered",
            confidence=0.9,  # High confidence in rejection
            processing_time=processing_time
        )

    def _create_fallback_result(self, content: str, start_time: float, error_msg: str) -> SectioningResult:
        """Create fallback result when processing fails"""
        processing_time = time.time() - start_time

        # Use truncated original content as fallback
        fallback_content = content[:2000] + "..." if len(content) > 2000 else content

        return SectioningResult(
            optimized_content=fallback_content,
            original_length=len(content),
            optimized_length=len(fallback_content),
            sections_identified=["fallback"],
            restaurants_density=0.3,  # Assume some restaurant content
            restaurants_found=[],
            sectioning_method="fallback",
            confidence=0.3,
            processing_time=processing_time
        )

    def _update_stats(self, result: SectioningResult, processing_time: float):
        """Update performance statistics"""
        self.stats["total_processed"] += 1

        # Update average processing time
        current_avg = self.stats["avg_processing_time"]
        total_processed = self.stats["total_processed"]
        self.stats["avg_processing_time"] = (current_avg * (total_processed - 1) + processing_time) / total_processed

        # Update content reduction stats
        if result.original_length > 0:
            reduction = (result.original_length - result.optimized_length) / result.original_length
            current_reduction_avg = self.stats["avg_content_reduction"]
            self.stats["avg_content_reduction"] = (current_reduction_avg * (total_processed - 1) + reduction) / total_processed

        # Count restaurants and high confidence results
        if result.restaurants_found:
            self.stats["restaurants_detected"] += len(result.restaurants_found)

        if result.confidence > 0.8:
            self.stats["high_confidence_results"] += 1

    async def section_content(self, content: str, url: str = "") -> Dict[str, Any]:
        """
        Legacy compatibility method for existing code
        """
        result = await self.process_content(content, url)

        # Convert to legacy format
        return {
            "content": result.optimized_content,
            "restaurants_found": result.restaurants_found,
            "confidence": result.confidence,
            "sections_identified": result.sections_identified,
            "restaurant_density": result.restaurants_density
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self.stats.copy()

        # Calculate additional metrics
        if stats["total_processed"] > 0:
            stats["cache_hit_rate"] = (stats["cache_hits"] / stats["total_processed"]) * 100
            stats["high_confidence_rate"] = (stats["high_confidence_results"] / stats["total_processed"]) * 100
            stats["avg_restaurants_per_process"] = stats["restaurants_detected"] / stats["total_processed"]
        else:
            stats["cache_hit_rate"] = 0
            stats["high_confidence_rate"] = 0
            stats["avg_restaurants_per_process"] = 0

        return stats

    def log_stats(self):
        """Log comprehensive statistics"""
        stats = self.get_stats()

        logger.info("=" * 50)
        logger.info("🔍 CONTENT SECTIONING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"📊 Total Processed: {stats['total_processed']}")
        logger.info(f"🚀 Cache Hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1f}%)")
        logger.info(f"⏱️ Avg Processing Time: {stats['avg_processing_time']:.2f}s")
        logger.info(f"📉 Avg Content Reduction: {stats['avg_content_reduction']*100:.1f}%")
        logger.info(f"🍽️ Restaurants Detected: {stats['restaurants_detected']}")
        logger.info(f"🎯 High Confidence Results: {stats['high_confidence_results']} ({stats['high_confidence_rate']:.1f}%)")
        logger.info(f"📈 Avg Restaurants/Process: {stats['avg_restaurants_per_process']:.1f}")
        logger.info("=" * 50)

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.log_stats()
            self._content_cache.clear()
            logger.info("🔍 Content sectioning agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during content sectioning cleanup: {e}")