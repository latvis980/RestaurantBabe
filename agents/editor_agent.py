# enhanced_editor_agent.py
"""
Enhanced Editor Agent with Maximum Decision Transparency

This version adds comprehensive logging and analysis to understand:
- Why specific restaurants are selected/rejected
- How content quality affects extraction
- Geographic relevance scoring  
- Content-to-restaurant mapping
- Decision reasoning for each step

Use this for debugging instead of the regular editor_agent.py
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.debug_utils import log_function_call, dump_chain_state

logger = logging.getLogger(__name__)

class EditorAgent:
    """
    Editor Agent with Maximum Decision Transparency

    Adds detailed logging and analysis to understand decision-making process
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('editor', 4096)
        )

        # Enhanced prompts with reasoning requirements
        self._setup_enhanced_prompts()

        logger.info("✅ Enhanced Editor Agent initialized with transparency features")

    def _setup_enhanced_prompts(self):
        """Setup enhanced prompts that require reasoning output"""

        # Enhanced system prompt that requires detailed reasoning
        self.enhanced_system_prompt = """You are a sophisticated restaurant curator and concierge with TRANSPARENCY REQUIREMENTS.

Your task: Extract restaurants from web content while providing detailed reasoning for your decisions.

EXTRACTION APPROACH:
- Act as a knowledgeable local who knows the restaurant scene
- Include restaurants even if they don't perfectly match all requirements, but explain your reasoning diplomatically
- Focus on quality and authenticity over perfect specification matching
- Be generous but discerning - include places a food-savvy friend would recommend

GEOGRAPHIC RELEVANCE:
- Prioritize restaurants in the specified destination city
- Include nearby areas only if clearly relevant (same metro area)
- Flag geographic mismatches but still include if they're exceptional

TRANSPARENCY REQUIREMENTS:
You must provide detailed reasoning for your decisions including:
1. Content quality assessment for each source
2. Geographic relevance scoring  
3. Why each restaurant was selected or notable ones were excluded
4. How you determined the final count
5. Content-to-restaurant mapping

FORMAT YOUR RESPONSE AS:
```json
{
  "restaurants": [
    {
      "name": "Restaurant Name",
      "description": "Rich description from content...",
      "address": "Full address or null",
      "cuisine_tags": ["tag1", "tag2"],
      "sources": ["domain1", "domain2"],
      "geographic_relevance": 0.9,
      "content_quality": 0.8,
      "selection_reasoning": "Why this restaurant was selected..."
    }
  ],
  "processing_analysis": {
    "content_sources_analyzed": 5,
    "total_restaurants_mentioned": 12,
    "restaurants_extracted": 5,
    "geographic_filtering": "Details about location filtering...",
    "content_quality_assessment": "Analysis of source content quality...",
    "extraction_reasoning": "Why this number of restaurants was extracted...",
    "notable_exclusions": "Restaurants mentioned but not included and why..."
  },
  "follow_up_queries": ["query1", "query2"]
}
```

CRITICAL: Always provide the processing_analysis section with detailed reasoning."""

        # Create the enhanced prompt template
        self.enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", self.enhanced_system_prompt),
            ("human", """
Original user request: {{raw_query}}
Destination: {{destination}}

Scraped content from multiple sources:
{{scraped_content}}

Extract restaurants using the diplomatic concierge approach with FULL TRANSPARENCY.
Provide detailed reasoning for all decisions in the processing_analysis section.
""")
        ])

        # Create the chain
        self.enhanced_chain = self.enhanced_prompt | self.model

    @log_function_call
    def edit(
        self, scraped_results=None, database_restaurants=None, 
        raw_query="", destination="Unknown", **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced edit method with maximum transparency and decision logging
        """
        try:
            logger.info("✏️ ENHANCED EDITOR: Starting with transparency features")
            logger.info(f"📊 Input: {len(scraped_results) if scraped_results else 0} scraped results")
            logger.info(f"📊 Input: {len(database_restaurants) if database_restaurants else 0} database restaurants")

            # For this test, we're focusing on web-only pipeline
            if not scraped_results:
                logger.warning("⚠️ No scraped results to process")
                return self._create_empty_response("No scraped content available")

            # =================================================================
            # CONTENT PREPARATION WITH ANALYSIS
            # =================================================================

            content_analysis = self._analyze_input_content(scraped_results, destination)
            formatted_content = self._prepare_enhanced_scraped_content(scraped_results)

            logger.info(f"📄 Content analysis: {content_analysis['total_chars']:,} chars from {content_analysis['source_count']} sources")
            logger.info(f"🎯 Geographic relevance: {content_analysis['geographic_relevance']:.1f}/10")
            logger.info(f"📊 Content quality: {content_analysis['avg_quality']:.1f}/10")

            # =================================================================
            # AI PROCESSING WITH ENHANCED PROMPT
            # =================================================================

            logger.info("🤖 Calling AI with enhanced transparency prompt")

            response = self.enhanced_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "scraped_content": formatted_content
            })

            # =================================================================
            # RESPONSE PARSING WITH VALIDATION
            # =================================================================

            parsed_response = self._parse_enhanced_response(response.content)

            if not parsed_response:
                logger.error("❌ Failed to parse AI response")
                return self._create_empty_response("AI response parsing failed")

            # =================================================================
            # TRANSPARENCY LOGGING
            # =================================================================

            self._log_transparency_analysis(parsed_response, content_analysis, raw_query, destination)

            # =================================================================
            # FORMAT FOR ORCHESTRATOR COMPATIBILITY
            # =================================================================

            formatted_output = self._format_enhanced_output(parsed_response)

            logger.info(f"✅ Enhanced editor complete: {len(formatted_output['edited_results']['main_list'])} restaurants")

            return formatted_output

        except Exception as e:
            logger.error(f"❌ Error in enhanced editor: {e}")
            return self._create_empty_response(f"Editor error: {str(e)}")

    def _analyze_input_content(self, scraped_results: List[Dict], destination: str) -> Dict[str, Any]:
        """
        Analyze input content quality and geographic relevance
        """
        total_chars = 0
        source_count = len(scraped_results)
        quality_scores = []
        geographic_scores = []

        for result in scraped_results:
            content = result.get('scraped_content', '')
            url = result.get('url', '')
            domain = url.split('/')[2] if '/' in url else 'unknown'

            # Content quality scoring
            content_length = len(content)
            total_chars += content_length

            # Quality score based on length and content richness
            quality_score = min(10, content_length / 1000)  # 1000 chars = score of 1
            if 'restaurant' in content.lower() or 'menu' in content.lower():
                quality_score += 2
            if any(word in content.lower() for word in ['chef', 'cuisine', 'dish', 'food']):
                quality_score += 1

            quality_scores.append(min(quality_score, 10))

            # Geographic relevance scoring
            destination_words = destination.lower().split()
            geo_score = 0
            for word in destination_words:
                if word in content.lower():
                    geo_score += 3  # Found destination mention

            # Bonus for local terms
            if any(term in content.lower() for term in ['local', 'neighborhood', 'district', 'area']):
                geo_score += 1

            geographic_scores.append(min(geo_score, 10))

        return {
            'total_chars': total_chars,
            'source_count': source_count,
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'geographic_relevance': sum(geographic_scores) / len(geographic_scores) if geographic_scores else 0,
            'quality_distribution': quality_scores,
            'geographic_distribution': geographic_scores
        }

    def _prepare_enhanced_scraped_content(self, scraped_results: List[Dict]) -> str:
        """
        Prepare scraped content with enhanced formatting and source tracking
        """
        formatted_content = []

        for i, result in enumerate(scraped_results, 1):
            url = result.get('url', 'Unknown')
            domain = url.split('/')[2] if '/' in url else 'unknown'
            title = result.get('title', 'Untitled')
            content = result.get('scraped_content', '')

            if content and len(content.strip()) > 50:
                # Enhanced formatting with metadata
                formatted_content.append(f"""
CONTENT SOURCE {i}:
URL: {url}
Domain: {domain}
Title: {title}
Content Length: {len(content):,} characters
Content Quality: {'🟢 Rich' if len(content) > 2000 else '🟡 Medium' if len(content) > 500 else '🔴 Limited'}

CONTENT:
{content[:10000]}{'...' if len(content) > 10000 else ''}

---
""")

        return "\n".join(formatted_content)

    def _parse_enhanced_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse enhanced AI response with better error handling
        """
        try:
            # Clean response content
            content = response_content.strip()

            # Extract JSON from markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            parsed = json.loads(content)

            # Validate required structure
            if not isinstance(parsed, dict):
                logger.error("❌ Response is not a dictionary")
                return None

            if 'restaurants' not in parsed:
                logger.error("❌ Response missing 'restaurants' field")
                return None

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.error(f"📄 Raw content: {response_content[:500]}...")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing response: {e}")
            return None

    def _log_transparency_analysis(
        self, parsed_response: Dict[str, Any], content_analysis: Dict[str, Any], 
        query: str, destination: str
    ):
        """
        Log detailed transparency analysis of editor decisions
        """
        restaurants = parsed_response.get('restaurants', [])
        analysis = parsed_response.get('processing_analysis', {})

        logger.info("🔍 TRANSPARENCY ANALYSIS:")
        logger.info(f"   Query: {query}")
        logger.info(f"   Destination: {destination}")
        logger.info(f"   Content Sources: {content_analysis['source_count']}")
        logger.info(f"   Total Content: {content_analysis['total_chars']:,} chars")
        logger.info(f"   Avg Content Quality: {content_analysis['avg_quality']:.1f}/10")
        logger.info(f"   Geographic Relevance: {content_analysis['geographic_relevance']:.1f}/10")
        logger.info(f"   Restaurants Extracted: {len(restaurants)}")

        # Log AI's reasoning
        logger.info("🧠 AI REASONING:")
        logger.info(f"   Content Sources Analyzed: {analysis.get('content_sources_analyzed', 'Not provided')}")
        logger.info(f"   Total Mentioned: {analysis.get('total_restaurants_mentioned', 'Not provided')}")
        logger.info(f"   Extraction Reasoning: {analysis.get('extraction_reasoning', 'Not provided')}")
        logger.info(f"   Notable Exclusions: {analysis.get('notable_exclusions', 'Not provided')}")

        # Log individual restaurant decisions
        for i, restaurant in enumerate(restaurants, 1):
            name = restaurant.get('name', 'Unknown')
            geo_score = restaurant.get('geographic_relevance', 0)
            quality_score = restaurant.get('content_quality', 0)
            reasoning = restaurant.get('selection_reasoning', 'No reasoning provided')

            logger.info(f"   Restaurant {i}: {name}")
            logger.info(f"     Geographic: {geo_score:.1f}/10")
            logger.info(f"     Quality: {quality_score:.1f}/10") 
            logger.info(f"     Reasoning: {reasoning}")

    def _format_enhanced_output(self, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format enhanced response for orchestrator compatibility while preserving transparency data
        """
        restaurants = parsed_response.get('restaurants', [])
        analysis = parsed_response.get('processing_analysis', {})
        follow_up_queries = parsed_response.get('follow_up_queries', [])

        # Convert to standard format expected by orchestrator
        formatted_restaurants = []

        for restaurant in restaurants:
            formatted_restaurant = {
                'name': restaurant.get('name', ''),
                'description': restaurant.get('description', ''),
                'address': restaurant.get('address'),
                'cuisine_tags': restaurant.get('cuisine_tags', []),
                'sources': restaurant.get('sources', []),

                # Enhanced transparency fields
                '_geographic_relevance': restaurant.get('geographic_relevance', 0),
                '_content_quality': restaurant.get('content_quality', 0),
                '_selection_reasoning': restaurant.get('selection_reasoning', ''),
                '_transparency_enabled': True
            }
            formatted_restaurants.append(formatted_restaurant)

        return {
            'edited_results': {
                'main_list': formatted_restaurants,
                '_processing_analysis': analysis,  # Preserve AI's analysis
                '_transparency_mode': True
            },
            'follow_up_queries': follow_up_queries,
            '_editor_reasoning': analysis.get('extraction_reasoning', ''),
            '_content_analysis': analysis
        }

    def _create_empty_response(self, reason: str) -> Dict[str, Any]:
        """Create empty response with transparency info"""
        return {
            'edited_results': {
                'main_list': [],
                '_processing_analysis': {'reason': reason},
                '_transparency_mode': True
            },
            'follow_up_queries': [],
            '_editor_reasoning': reason,
            '_content_analysis': {}
        }

    # =================================================================
    # CONTENT QUALITY ANALYSIS METHODS
    # =================================================================

    def analyze_content_for_restaurants(self, scraped_results: List[Dict], destination: str) -> Dict[str, Any]:
        """
        Pre-analyze content to predict editor behavior

        This helps understand BEFORE AI processing:
        - How many restaurants might be extractable
        - Content quality issues
        - Geographic relevance problems
        """
        analysis = {
            'total_sources': len(scraped_results),
            'total_content_chars': 0,
            'estimated_restaurants': 0,
            'geographic_relevance': 0,
            'content_quality_issues': [],
            'source_analysis': []
        }

        destination_keywords = destination.lower().split()
        restaurant_keywords = ['restaurant', 'café', 'bistro', 'bar', 'taverna', 'trattoria', 'brasserie']

        for i, result in enumerate(scraped_results, 1):
            url = result.get('url', '')
            content = result.get('scraped_content', '')
            domain = url.split('/')[2] if '/' in url else 'unknown'

            content_length = len(content)
            analysis['total_content_chars'] += content_length

            # Count potential restaurant mentions
            restaurant_mentions = 0
            for keyword in restaurant_keywords:
                restaurant_mentions += content.lower().count(keyword)

            # Geographic relevance check
            geo_mentions = 0
            for dest_word in destination_keywords:
                geo_mentions += content.lower().count(dest_word)

            # Content quality assessment
            quality_score = 0
            if content_length > 2000:
                quality_score += 3
            elif content_length > 500:
                quality_score += 1

            if restaurant_mentions > 3:
                quality_score += 2
            elif restaurant_mentions > 0:
                quality_score += 1

            if geo_mentions > 0:
                quality_score += 2

            # Identify quality issues
            issues = []
            if content_length < 500:
                issues.append("Short content")
            if restaurant_mentions == 0:
                issues.append("No restaurant keywords found")
            if geo_mentions == 0:
                issues.append(f"No mention of {destination}")
            if 'error' in content.lower() or 'not found' in content.lower():
                issues.append("Error content detected")

            source_info = {
                'domain': domain,
                'content_length': content_length,
                'restaurant_mentions': restaurant_mentions,
                'geographic_mentions': geo_mentions,
                'quality_score': quality_score,
                'issues': issues,
                'estimated_restaurants': min(restaurant_mentions // 3, 5)  # Rough estimate
            }

            analysis['source_analysis'].append(source_info)
            analysis['estimated_restaurants'] += source_info['estimated_restaurants']

        # Overall assessments
        if analysis['total_sources'] > 0:
            avg_geo_score = sum(s['geographic_mentions'] for s in analysis['source_analysis']) / analysis['total_sources']
            analysis['geographic_relevance'] = min(avg_geo_score * 2, 10)  # Scale to 0-10

        # Predict potential issues
        if analysis['estimated_restaurants'] == 0:
            analysis['content_quality_issues'].append("No restaurant content detected")
        if analysis['geographic_relevance'] < 2:
            analysis['content_quality_issues'].append(f"Low geographic relevance to {destination}")
        if analysis['total_content_chars'] < 2000:
            analysis['content_quality_issues'].append("Limited total content")

        logger.info(f"📊 Pre-analysis: ~{analysis['estimated_restaurants']} restaurants expected")
        logger.info(f"🌍 Geographic relevance: {analysis['geographic_relevance']:.1f}/10")

        if analysis['content_quality_issues']:
            logger.warning(f"⚠️ Potential issues: {', '.join(analysis['content_quality_issues'])}")

        return analysis

    def save_transparency_report(
        self, filepath: str, query: str, destination: str, 
        scraped_results: List[Dict], editor_output: Dict[str, Any],
        pre_analysis: Dict[str, Any], timing_data: Dict[str, Any]
    ):
        """
        Save comprehensive transparency report
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("ENHANCED EDITOR TRANSPARENCY REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Editor Mode: Enhanced with transparency\n")
                f.write("=" * 80 + "\n\n")

                # =================================================================
                # PRE-PROCESSING ANALYSIS
                # =================================================================
                f.write("PRE-PROCESSING CONTENT ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Sources: {pre_analysis['total_sources']}\n")
                f.write(f"Total Content: {pre_analysis['total_content_chars']:,} characters\n")
                f.write(f"Estimated Restaurants: {pre_analysis['estimated_restaurants']}\n")
                f.write(f"Geographic Relevance: {pre_analysis['geographic_relevance']:.1f}/10\n")

                if pre_analysis['content_quality_issues']:
                    f.write(f"Potential Issues: {', '.join(pre_analysis['content_quality_issues'])}\n")

                f.write("\nSOURCE-BY-SOURCE ANALYSIS:\n")
                for i, source in enumerate(pre_analysis['source_analysis'], 1):
                    f.write(f"  {i}. {source['domain']}\n")
                    f.write(f"     Content: {source['content_length']:,} chars\n")
                    f.write(f"     Restaurant mentions: {source['restaurant_mentions']}\n")
                    f.write(f"     Geographic mentions: {source['geographic_mentions']}\n")
                    f.write(f"     Quality score: {source['quality_score']:.1f}/10\n")
                    f.write(f"     Estimated restaurants: {source['estimated_restaurants']}\n")
                    if source['issues']:
                        f.write(f"     Issues: {', '.join(source['issues'])}\n")
                    f.write("\n")

                # =================================================================
                # AI PROCESSING RESULTS
                # =================================================================
                f.write("AI PROCESSING RESULTS\n")
                f.write("-" * 40 + "\n")

                restaurants = editor_output.get('edited_results', {}).get('main_list', [])
                ai_analysis = editor_output.get('_content_analysis', {})

                f.write(f"Restaurants Extracted: {len(restaurants)}\n")
                f.write(f"Follow-up Queries: {len(editor_output.get('follow_up_queries', []))}\n\n")

                # AI's reasoning (if available)
                if ai_analysis:
                    f.write("AI DECISION REASONING:\n")
                    f.write(f"  Sources Analyzed: {ai_analysis.get('content_sources_analyzed', 'Not provided')}\n")
                    f.write(f"  Total Mentioned: {ai_analysis.get('total_restaurants_mentioned', 'Not provided')}\n")
                    f.write(f"  Extraction Logic: {ai_analysis.get('extraction_reasoning', 'Not provided')}\n")
                    f.write(f"  Notable Exclusions: {ai_analysis.get('notable_exclusions', 'Not provided')}\n\n")

                # =================================================================
                # RESTAURANT-BY-RESTAURANT ANALYSIS
                # =================================================================
                f.write("DETAILED RESTAURANT ANALYSIS\n")
                f.write("-" * 40 + "\n")

                if restaurants:
                    for i, restaurant in enumerate(restaurants, 1):
                        name = restaurant.get('name', 'Unknown')
                        description = restaurant.get('description', '')
                        cuisine_tags = restaurant.get('cuisine_tags', [])
                        address = restaurant.get('address', '')
                        sources = restaurant.get('sources', [])

                        # Transparency fields
                        geo_relevance = restaurant.get('_geographic_relevance', 0)
                        content_quality = restaurant.get('_content_quality', 0)
                        selection_reasoning = restaurant.get('_selection_reasoning', 'No reasoning provided')

                        f.write(f"RESTAURANT {i}: {name}\n")
                        f.write(f"  Cuisine: {', '.join(cuisine_tags[:5]) if cuisine_tags else 'Not specified'}\n")
                        f.write(f"  Address: {address if address else 'Not available'}\n")
                        f.write(f"  Description: {len(description)} characters\n")
                        f.write(f"  Sources: {len(sources)} domains\n")
                        f.write(f"  Geographic Score: {geo_relevance:.1f}/10\n")
                        f.write(f"  Content Quality: {content_quality:.1f}/10\n")
                        f.write(f"  Selection Reasoning: {selection_reasoning}\n")

                        # Description analysis
                        if description:
                            word_count = len(description.split())
                            f.write(f"  Description Quality: {'✅ Rich' if word_count > 50 else '⚠️ Basic' if word_count > 20 else '❌ Minimal'} ({word_count} words)\n")

                            # Show first 200 chars
                            preview = description[:200] + "..." if len(description) > 200 else description
                            f.write(f"  Preview: {preview}\n")
                        else:
                            f.write(f"  Description Quality: ❌ Missing\n")

                        f.write("\n")
                else:
                    f.write("❌ NO RESTAURANTS EXTRACTED\n\n")

                # =================================================================
                # DECISION TREE ANALYSIS
                # =================================================================
                f.write("EDITOR DECISION TREE ANALYSIS\n")
                f.write("-" * 40 + "\n")

                f.write("WHY THIS NUMBER OF RESTAURANTS?\n\n")

                expected_count = pre_analysis['estimated_restaurants']
                actual_count = len(restaurants)

                if actual_count == 0:
                    f.write("❌ ZERO EXTRACTION ANALYSIS:\n")
                    f.write(f"  Expected: {expected_count} restaurants\n")
                    f.write(f"  Actual: 0 restaurants\n")
                    f.write("  Possible causes:\n")
                    f.write("    • Content doesn't contain clear restaurant information\n")
                    f.write("    • Geographic mismatch (content about wrong city)\n")
                    f.write("    • Content quality too low (errors, paywalls, etc.)\n")
                    f.write("    • AI prompt too restrictive\n")
                    f.write("    • Content language issues\n\n")

                elif actual_count < expected_count:
                    f.write(f"⚠️ LOWER THAN EXPECTED ({actual_count} vs {expected_count}):\n")
                    f.write("  Possible causes:\n")
                    f.write("    • AI being selective about quality\n")
                    f.write("    • Some mentions were not full restaurants\n")
                    f.write("    • Geographic filtering removed some options\n")
                    f.write("    • Content had duplicate mentions\n\n")

                elif actual_count == expected_count:
                    f.write(f"✅ AS EXPECTED ({actual_count} restaurants):\n")
                    f.write("  • Pre-analysis prediction was accurate\n")
                    f.write("  • Content quality supported extraction\n")
                    f.write("  • AI found extractable restaurant information\n\n")

                else:
                    f.write(f"📈 MORE THAN EXPECTED ({actual_count} vs {expected_count}):\n")
                    f.write("  • Content richer than initial analysis suggested\n")
                    f.write("  • AI found additional restaurant mentions\n")
                    f.write("  • High-quality sources with detailed information\n\n")

                # =================================================================
                # PERFORMANCE ANALYSIS
                # =================================================================
                f.write("PERFORMANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Processing Time: {timing_data.get('total_time', 0):.2f}s\n")
                f.write(f"Editor Processing Time: {timing_data.get('step6_time', 0):.2f}s\n")

                if analysis['total_content_chars'] > 0:
                    chars_per_second = analysis['total_content_chars'] / timing_data.get('step6_time', 1)
                    f.write(f"Processing Speed: {chars_per_second:,.0f} chars/second\n")

                if actual_count > 0:
                    restaurants_per_source = actual_count / len(scraped_results)
                    f.write(f"Extraction Efficiency: {restaurants_per_source:.1f} restaurants/source\n")

                f.write("\n")

                # =================================================================
                # IMPROVEMENT RECOMMENDATIONS
                # =================================================================
                f.write("IMPROVEMENT RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")

                if actual_count < 3:
                    f.write("TO GET MORE RESTAURANTS:\n")
                    f.write("  1. Search for more restaurant-specific sources\n")
                    f.write("  2. Include local food blogs and guides\n")
                    f.write("  3. Add cuisine-specific search terms\n")
                    f.write("  4. Search in local language if applicable\n")
                    f.write("  5. Check if sources are being blocked/paywall\n\n")

                if pre_analysis['geographic_relevance'] < 5:
                    f.write("TO IMPROVE GEOGRAPHIC RELEVANCE:\n")
                    f.write("  1. Add city name to search queries\n")
                    f.write("  2. Include local area/district names\n")
                    f.write("  3. Search for location-specific food guides\n")
                    f.write("  4. Filter search results by domain location\n\n")

                if any(len(r.get('description', '')) < 100 for r in restaurants):
                    f.write("TO IMPROVE DESCRIPTION QUALITY:\n")
                    f.write("  1. Target food review sites vs listing sites\n")
                    f.write("  2. Include professional food critic sources\n")
                    f.write("  3. Search for detailed restaurant guides\n")
                    f.write("  4. Avoid aggregator sites with minimal content\n\n")

                # =================================================================
                # RAW AI RESPONSE (for debugging)
                # =================================================================
                f.write("RAW AI RESPONSE (DEBUG)\n")
                f.write("-" * 40 + "\n")
                f.write(json.dumps(parsed_response, indent=2, ensure_ascii=False))
                f.write("\n\n")

                f.write("✅ ENHANCED EDITOR ANALYSIS COMPLETE\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❌ Error saving transparency report: {e}")


# =================================================================
# USAGE INTEGRATION
# =================================================================

def create_enhanced_editor_test(config, orchestrator):
    """
    Factory function to create enhanced editor test
    """
    return EditorTest(config, orchestrator)


# =================================================================
# MONKEY PATCH FOR TEMPORARY ENHANCEMENT
# =================================================================

def patch_editor_agent_for_transparency(editor_agent, config):
    """
    Temporarily enhance existing editor agent with transparency features

    Use this to add transparency to your existing editor without replacing it
    """
    enhanced_agent = EnhancedEditorAgent(config)

    # Store original methods
    editor_agent._original_edit = editor_agent.edit

    # Replace with enhanced version
    editor_agent.edit_with_transparency = enhanced_agent.edit_with_transparency
    editor_agent.analyze_content_for_restaurants = enhanced_agent.analyze_content_for_restaurants
    editor_agent.save_transparency_report = enhanced_agent.save_transparency_report

    logger.info("✅ Editor agent patched with transparency features")
    return editor_agent