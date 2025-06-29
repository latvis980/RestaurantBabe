# agents/content_sectioning_agent.py
import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class SectioningResult:
    """Result from content sectioning analysis"""
    optimized_content: str
    original_length: int
    optimized_length: int
    sections_identified: List[str]
    restaurants_density: float
    sectioning_method: str
    confidence: float

class ContentSection(BaseModel):
    """Individual content section classification"""
    section_type: str = Field(description="Type: intro, restaurant_list, restaurant_detail, author_bio, navigation, footer, advertisement")
    content: str = Field(description="The actual text content of this section")
    priority: int = Field(description="Priority 1-5 (1=highest, 5=lowest) for restaurant extraction")
    restaurant_count_estimate: int = Field(description="Estimated number of restaurants mentioned in this section")
    confidence: float = Field(description="Confidence 0.0-1.0 in this classification")

class ContentAnalysis(BaseModel):
    """Complete content sectioning analysis"""
    sections: List[ContentSection] = Field(description="All identified content sections")
    content_type: str = Field(description="Overall content type: restaurant_guide, single_review, listicle, blog_post, news_article")
    total_restaurant_estimate: int = Field(description="Total estimated restaurants in entire content")
    sectioning_confidence: float = Field(description="Overall confidence in sectioning accuracy")

class ContentSectioningAgent:
    """
    AI-powered content sectioning agent that identifies and prioritizes
    restaurant-rich content sections while filtering out intro/footer fluff.
    """

    def __init__(self, config):
        self.config = config
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,  # Uses GPT-4o as requested
            temperature=0.1,  # Low temperature for consistent analysis
            api_key=config.OPENAI_API_KEY
        )

        # Output parser for structured sectioning
        self.parser = PydanticOutputParser(pydantic_object=ContentAnalysis)

        # Main sectioning prompt
        self.sectioning_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert content analyst specializing in restaurant guide articles. Your task is to analyze article content and identify distinct sections, prioritizing restaurant-rich content over introductory fluff.

SECTION TYPES TO IDENTIFY:
1. **intro** - Opening paragraphs, author introduction, general city/area description
2. **restaurant_list** - Sections with multiple restaurants (numbered lists, "best of" sections)
3. **restaurant_detail** - Individual restaurant descriptions with specific details
4. **author_bio** - Author information, credentials, about the writer
5. **navigation** - Site navigation, related articles, "you might also like"
6. **footer** - Closing thoughts, disclaimers, publishing info
7. **advertisement** - Promotional content, sponsored sections

PRIORITIZATION RULES:
- Priority 1 (HIGHEST): restaurant_list, restaurant_detail sections
- Priority 2: Content that mentions specific restaurant names with descriptions
- Priority 3: Location-specific food content without specific restaurant names
- Priority 4: General food/travel advice
- Priority 5 (LOWEST): intro, author_bio, navigation, footer, advertisement

ANALYSIS GOALS:
- Identify where restaurant information is concentrated
- Estimate restaurant density per section
- Distinguish between restaurant content and filler content
- Provide confidence scores for each classification

Be especially good at identifying:
- Numbered restaurant lists ("1. Restaurant Name", "2. Another Place")
- Restaurant sections with addresses, descriptions, specialties
- Mixed content where restaurants are embedded in longer paragraphs

{format_instructions}
            """),
            ("human", """
Analyze this restaurant article content and break it into prioritized sections:

CONTENT TO ANALYZE:
{content}

Please identify all sections, estimate restaurant density, and provide priority scores for optimal content extraction.
            """)
        ])

        # Content optimization prompt (for final selection)
        self.optimization_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a content optimizer. Given sectioned content with priorities, select and combine the most valuable sections for restaurant extraction within a target character limit.

OPTIMIZATION STRATEGY:
1. Always include ALL Priority 1 sections (restaurant lists/details)
2. Include Priority 2 sections if space allows
3. Include brief context from Priority 3 if needed for coherence
4. Drop Priority 4-5 sections (intro, bio, footer) unless critical for context
5. Maintain readability - don't create choppy fragments

OUTPUT REQUIREMENTS:
- Optimized content should flow naturally
- Include section transitions if needed ("The recommended restaurants include:")
- Maximize restaurant density within character limit
- Preserve complete restaurant descriptions (don't cut mid-sentence)
            """),
            ("human", """
SECTIONED CONTENT WITH PRIORITIES:
{sectioned_content}

TARGET CHARACTER LIMIT: {target_limit}
CURRENT TOTAL LENGTH: {current_length}

Please combine the highest-priority sections into optimized content that maximizes restaurant information density while staying within the character limit.
            """)
        ])

        # Create processing chains
        self.sectioning_chain = (
            {
                "content": lambda x: x["content"],
                "format_instructions": lambda _: self.parser.get_format_instructions()
            }
            | self.sectioning_prompt
            | self.model
            | self.parser
        )

        self.optimization_chain = self.optimization_prompt | self.model

        # Statistics tracking
        self.stats = {
            "sections_processed": 0,
            "content_optimized": 0,
            "average_compression_ratio": 0.0,
            "restaurant_density_improvement": 0.0
        }

    async def section_content(
        self, 
        content: str, 
        target_limit: int = 6000,
        min_restaurant_density: float = 0.1
    ) -> SectioningResult:
        """
        Main entry point: analyze content and return optimized restaurant-focused sections.

        Args:
            content: Raw cleaned text content (from any scraping method)
            target_limit: Target character limit for optimized content
            min_restaurant_density: Minimum restaurant mentions per 1000 chars to include section

        Returns:
            SectioningResult with optimized content and analysis metadata
        """
        try:
            original_length = len(content)

            # Skip sectioning for very short content
            if original_length < 1000:
                logger.info(f"Content too short ({original_length} chars), skipping sectioning")
                return SectioningResult(
                    optimized_content=content,
                    original_length=original_length,
                    optimized_length=original_length,
                    sections_identified=["short_content"],
                    restaurants_density=0.0,
                    sectioning_method="passthrough",
                    confidence=1.0
                )

            logger.info(f"🔍 Analyzing content structure ({original_length:,} characters)")

            # Step 1: AI-powered content sectioning
            analysis = await self.sectioning_chain.ainvoke({"content": content})

            # Step 2: Filter and prioritize sections
            high_priority_sections = self._filter_priority_sections(
                analysis, min_restaurant_density
            )

            # Step 3: Optimize content within target limit
            optimized_content = await self._optimize_content_selection(
                high_priority_sections, target_limit, original_length
            )

            # Step 4: Calculate results
            optimized_length = len(optimized_content)
            sections_identified = [s.section_type for s in analysis.sections]

            # Calculate restaurant density improvement
            original_density = self._estimate_restaurant_density(content)
            optimized_density = self._estimate_restaurant_density(optimized_content)

            # Update stats
            self.stats["sections_processed"] += 1
            self.stats["content_optimized"] += 1
            compression_ratio = optimized_length / original_length
            self.stats["average_compression_ratio"] = (
                (self.stats["average_compression_ratio"] * (self.stats["content_optimized"] - 1) + compression_ratio)
                / self.stats["content_optimized"]
            )

            density_improvement = optimized_density / max(original_density, 0.01)
            self.stats["restaurant_density_improvement"] = (
                (self.stats["restaurant_density_improvement"] * (self.stats["content_optimized"] - 1) + density_improvement)
                / self.stats["content_optimized"]
            )

            logger.info(f"✨ Content optimized: {original_length:,} → {optimized_length:,} chars "
                       f"({compression_ratio:.1%} compression, {density_improvement:.1f}x density)")

            return SectioningResult(
                optimized_content=optimized_content,
                original_length=original_length,
                optimized_length=optimized_length,
                sections_identified=sections_identified,
                restaurants_density=optimized_density,
                sectioning_method="ai_sectioning",
                confidence=analysis.sectioning_confidence
            )

        except Exception as e:
            logger.error(f"Content sectioning failed: {e}")
            # Fallback: intelligent truncation
            return self._fallback_intelligent_truncation(content, target_limit)

    def _filter_priority_sections(
        self, 
        analysis: ContentAnalysis, 
        min_restaurant_density: float
    ) -> List[ContentSection]:
        """Filter sections based on priority and restaurant density"""

        filtered_sections = []

        for section in analysis.sections:
            # Always include high-priority restaurant sections
            if section.priority <= 2:
                filtered_sections.append(section)
                continue

            # Check restaurant density for medium priority sections
            if section.priority == 3:
                section_length = len(section.content)
                if section_length > 0:
                    density = section.restaurant_count_estimate / (section_length / 1000)
                    if density >= min_restaurant_density:
                        filtered_sections.append(section)
                        continue

            # Skip low priority sections (intro, footer, etc.) unless very high restaurant count
            if section.priority >= 4:
                if section.restaurant_count_estimate >= 3:  # Exception for intro sections with many restaurants
                    filtered_sections.append(section)

        logger.info(f"📝 Filtered {len(filtered_sections)}/{len(analysis.sections)} sections as restaurant-relevant")
        return filtered_sections

    async def _optimize_content_selection(
        self, 
        sections: List[ContentSection], 
        target_limit: int,
        original_length: int
    ) -> str:
        """Use AI to intelligently combine sections within character limit"""

        if not sections:
            return ""

        # Calculate total length of filtered sections
        total_filtered_length = sum(len(section.content) for section in sections)

        # If filtered content is already within limit, combine directly
        if total_filtered_length <= target_limit:
            combined_content = self._combine_sections_naturally(sections)
            logger.info(f"🎯 All filtered sections fit within limit ({total_filtered_length}/{target_limit} chars)")
            return combined_content

        # Need AI optimization to fit within target limit
        logger.info(f"📐 Need optimization: {total_filtered_length} chars → {target_limit} chars target")

        # Prepare sectioned content for AI optimization
        sectioned_content = self._format_sections_for_optimization(sections)

        try:
            response = await self.optimization_chain.ainvoke({
                "sectioned_content": sectioned_content,
                "target_limit": target_limit,
                "current_length": total_filtered_length
            })

            optimized_content = response.content.strip()

            # Ensure we didn't exceed the limit (with small buffer)
            if len(optimized_content) > target_limit * 1.1:  # 10% buffer
                logger.warning(f"AI optimization exceeded limit, applying hard truncation")
                optimized_content = self._smart_truncate(optimized_content, target_limit)

            return optimized_content

        except Exception as e:
            logger.error(f"AI optimization failed: {e}, falling back to priority truncation")
            return self._fallback_priority_truncation(sections, target_limit)

    def _combine_sections_naturally(self, sections: List[ContentSection]) -> str:
        """Combine sections with natural transitions"""

        # Sort by priority (highest first)
        sorted_sections = sorted(sections, key=lambda s: s.priority)

        content_parts = []

        for section in sorted_sections:
            content = section.content.strip()
            if content:
                # Add section transition if needed
                if section.section_type == "restaurant_list" and content_parts:
                    if not any("restaurant" in part.lower() for part in content_parts[-1:]):
                        content = f"The recommended restaurants include:\n\n{content}"

                content_parts.append(content)

        return "\n\n".join(content_parts)

    def _format_sections_for_optimization(self, sections: List[ContentSection]) -> str:
        """Format sections for AI optimization prompt"""

        formatted_parts = []

        for i, section in enumerate(sections, 1):
            formatted_parts.append(
                f"SECTION {i} (Priority {section.priority} - {section.section_type}):\n"
                f"Restaurant Count: {section.restaurant_count_estimate}\n"
                f"Length: {len(section.content)} chars\n"
                f"Content:\n{section.content}\n"
                f"---"
            )

        return "\n\n".join(formatted_parts)

    def _fallback_priority_truncation(self, sections: List[ContentSection], target_limit: int) -> str:
        """Fallback: combine sections by priority until limit reached"""

        sorted_sections = sorted(sections, key=lambda s: s.priority)
        combined_content = ""

        for section in sorted_sections:
            potential_content = combined_content + "\n\n" + section.content.strip() if combined_content else section.content.strip()

            if len(potential_content) <= target_limit:
                combined_content = potential_content
            else:
                # Try to fit partial content from this section
                remaining_space = target_limit - len(combined_content) - 2  # Account for \n\n
                if remaining_space > 200:  # Only if meaningful space left
                    partial_content = self._smart_truncate(section.content, remaining_space)
                    if partial_content:
                        combined_content += f"\n\n{partial_content}"
                break

        return combined_content

    def _smart_truncate(self, content: str, limit: int) -> str:
        """Smart truncation that preserves complete sentences and restaurant mentions"""

        if len(content) <= limit:
            return content

        # Find the last complete sentence within limit
        truncated = content[:limit]

        # Look for sentence endings working backwards
        for end_char in ['. ', '.\n', '!', '?']:
            last_sentence_end = truncated.rfind(end_char)
            if last_sentence_end > limit * 0.7:  # Don't truncate too aggressively
                return content[:last_sentence_end + 1].strip()

        # Fallback: find last complete word
        last_space = truncated.rfind(' ')
        if last_space > limit * 0.8:
            return content[:last_space].strip()

        # Hard truncation as last resort
        return content[:limit].strip()

    def _fallback_intelligent_truncation(self, content: str, target_limit: int) -> SectioningResult:
        """Fallback when AI sectioning fails: intelligent rule-based truncation"""

        logger.warning("Using fallback intelligent truncation")

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        # Simple heuristic: prioritize paragraphs with restaurant indicators
        restaurant_indicators = ['restaurant', 'cafe', 'bar', 'bistro', 'eatery', 'food', 'menu', 'chef', 'cuisine']

        scored_paragraphs = []
        for para in paragraphs:
            score = sum(1 for indicator in restaurant_indicators if indicator.lower() in para.lower())
            scored_paragraphs.append((score, para))

        # Sort by score (highest first)
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        # Combine until target limit
        optimized_content = ""
        for score, para in scored_paragraphs:
            potential_content = optimized_content + "\n\n" + para if optimized_content else para
            if len(potential_content) <= target_limit:
                optimized_content = potential_content
            else:
                break

        return SectioningResult(
            optimized_content=optimized_content,
            original_length=len(content),
            optimized_length=len(optimized_content),
            sections_identified=["fallback_truncation"],
            restaurants_density=self._estimate_restaurant_density(optimized_content),
            sectioning_method="fallback_heuristic",
            confidence=0.5
        )

    def _estimate_restaurant_density(self, content: str) -> float:
        """Estimate restaurants per 1000 characters"""
        if not content:
            return 0.0

        # Simple heuristic: count restaurant name patterns
        import re

        # Look for restaurant name patterns (capitalized words, common suffixes)
        restaurant_patterns = [
            r'\b[A-Z][a-zA-Z\s&\'-]{2,30}(?:Restaurant|Cafe|Bar|Bistro|Kitchen|House|Room|Place)\b',
            r'\b(?:Restaurant|Cafe|Bar|Bistro)\s+[A-Z][a-zA-Z\s&\'-]{2,30}\b',
            r'\b[A-Z][a-zA-Z\s&\'-]{2,30}(?:\s+serves|\s+offers|\s+specializes)\b',
            r'^\s*\d+\.\s+[A-Z][a-zA-Z\s&\'-]{2,50}',  # Numbered lists
        ]

        restaurant_count = 0
        for pattern in restaurant_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            restaurant_count += len(matches)

        # Avoid double counting - take max from patterns
        restaurant_count = min(restaurant_count, len(content.split('\n\n')))

        return (restaurant_count / len(content)) * 1000

    def get_stats(self) -> Dict:
        """Get sectioning performance statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "sections_processed": 0,
            "content_optimized": 0,
            "average_compression_ratio": 0.0,
            "restaurant_density_improvement": 0.0
        }