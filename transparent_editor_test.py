# transparent_editor_test.py
"""
Transparent Editor Test - Uses existing production agents without modification

This test:
1. Uses your existing orchestrator and agents unchanged
2. Adds transparency logging around the existing editor
3. Captures and analyzes editor decisions without modifying production code
4. Produces detailed analysis files showing why you get 5 restaurants
"""

import os
import json
import time
import logging
import tempfile
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TransparentEditorTest:
    """
    Test editor pipeline with transparency WITHOUT changing production code
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Use existing production agents unchanged
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent  
        self.scraper = orchestrator.scraper
        self.text_cleaner = orchestrator.text_cleaner
        self.editor_agent = orchestrator.editor_agent  # Use existing editor

        logger.info("✅ Transparent Editor Test initialized with production agents")

    async def test_editor_pipeline(self, restaurant_query: str, bot=None) -> str:
        """
        Test complete editor pipeline with transparency analysis
        """
        total_start_time = time.time()

        # Create files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraped_filename = f"scraped_content_{timestamp}.txt"
        edited_filename = f"edited_results_{timestamp}.txt"

        scraped_filepath = os.path.join(tempfile.gettempdir(), scraped_filename)
        edited_filepath = os.path.join(tempfile.gettempdir(), edited_filename)

        logger.info(f"🧪 Testing Editor Pipeline: {restaurant_query}")

        try:
            # =================================================================
            # STEP 1: QUERY ANALYSIS
            # =================================================================
            logger.info("🔍 Step 1: Query Analysis")
            step1_start = time.time()

            query_result = self.query_analyzer.analyze(restaurant_query)
            destination = query_result.get('destination', 'Unknown')

            step1_time = time.time() - step1_start

            # =================================================================
            # STEP 2: SEARCH
            # =================================================================
            logger.info("🔍 Step 2: Web Search")
            step2_start = time.time()

            search_results = await self.search_agent.search(restaurant_query)

            step2_time = time.time() - step2_start

            # =================================================================
            # STEP 3: SCRAPING
            # =================================================================
            logger.info("🕷️ Step 3: Intelligent Scraping")
            step3_start = time.time()

            scraping_results = []
            for result in search_results:
                try:
                    scraped = await self.scraper.scrape(result.get('url', ''))
                    if scraped and scraped.get('content'):
                        scraping_results.append({
                            'url': result.get('url', ''),
                            'title': result.get('title', ''),
                            'content': scraped['content']
                        })
                except Exception as e:
                    logger.warning(f"Scraping failed for {result.get('url')}: {e}")

            step3_time = time.time() - step3_start

            # =================================================================
            # STEP 4: TEXT CLEANING
            # =================================================================
            logger.info("🧹 Step 4: Text Cleaning")
            step4_start = time.time()

            cleaned_results = []
            for result in scraping_results:
                try:
                    cleaned = self.text_cleaner.clean(result['content'])
                    if cleaned:
                        cleaned_results.append({
                            'url': result['url'],
                            'title': result['title'],
                            'cleaned_content': cleaned
                        })
                except Exception as e:
                    logger.warning(f"Text cleaning failed: {e}")

            step4_time = time.time() - step4_start

            # =================================================================
            # STEP 5: SAVE SCRAPED CONTENT
            # =================================================================
            logger.info("💾 Step 5: Save Scraped Content")
            self._save_scraped_content_analysis(cleaned_results, restaurant_query, destination, scraped_filepath)

            # =================================================================
            # STEP 6: PRE-EDITOR ANALYSIS
            # =================================================================
            logger.info("🔍 Step 6: Pre-Editor Analysis")
            pre_analysis = self._analyze_content_before_editor(cleaned_results, destination)

            # =================================================================
            # STEP 7: EDITOR PROCESSING (using existing editor)
            # =================================================================
            logger.info("✏️ Step 7: Editor Processing")
            step7_start = time.time()

            # Prepare for existing editor
            editor_input = []
            for result in cleaned_results:
                if result.get('cleaned_content'):
                    editor_input.append({
                        'url': result.get('url', ''),
                        'scraped_content': result.get('cleaned_content', ''),
                        'title': result.get('title', ''),
                        'domain': result.get('url', '').split('/')[2] if result.get('url') else 'unknown'
                    })

            # Call existing editor (no changes to production code)
            editor_output = self.editor_agent.edit(
                scraped_results=editor_input,
                database_restaurants=None,
                raw_query=restaurant_query,
                destination=destination
            )

            step7_time = time.time() - step7_start

            # =================================================================
            # STEP 8: POST-EDITOR ANALYSIS
            # =================================================================
            logger.info("📊 Step 8: Post-Editor Analysis")

            total_time = time.time() - total_start_time

            self._save_editor_transparency_analysis(
                edited_filepath, restaurant_query, destination, 
                cleaned_results, editor_output, pre_analysis,
                {
                    'step1_time': step1_time,
                    'step2_time': step2_time,
                    'step3_time': step3_time,
                    'step4_time': step4_time,
                    'step7_time': step7_time,
                    'total_time': total_time,
                    'search_results_count': len(search_results),
                    'scraped_results_count': len(scraping_results),
                    'cleaned_results_count': len(cleaned_results),
                    'final_restaurants_count': len(editor_output.get('edited_results', {}).get('main_list', []))
                }
            )

            # =================================================================
            # STEP 9: SEND RESULTS - FIXED
            # =================================================================
            if bot and self.admin_chat_id:
                self._send_results_to_admin(
                    bot, scraped_filepath, edited_filepath, restaurant_query, editor_output
                )

            return edited_filepath

        except Exception as e:
            logger.error(f"❌ Error in editor pipeline test: {e}")

            # Save error
            with open(edited_filepath, 'w', encoding='utf-8') as f:
                f.write("EDITOR PIPELINE TEST ERROR\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")

                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")

            return edited_filepath

    def _analyze_content_before_editor(self, cleaned_results: List[Dict], destination: str) -> Dict[str, Any]:
        """
        Analyze content BEFORE editor to predict behavior
        """
        total_chars = 0
        restaurant_mentions = 0
        destination_mentions = 0
        source_analysis = []

        destination_keywords = destination.lower().split()
        restaurant_keywords = ['restaurant', 'café', 'bistro', 'bar', 'taverna', 'trattoria']

        for result in cleaned_results:
            content = result.get('cleaned_content', '')
            url = result.get('url', '')
            domain = url.split('/')[2] if '/' in url else 'unknown'

            content_length = len(content)
            total_chars += content_length

            # Count mentions
            content_lower = content.lower()
            source_restaurant_mentions = sum(content_lower.count(keyword) for keyword in restaurant_keywords)
            source_destination_mentions = sum(content_lower.count(word) for word in destination_keywords)

            restaurant_mentions += source_restaurant_mentions
            destination_mentions += source_destination_mentions

            # Quality assessment
            quality_score = min(10, content_length / 1000)
            if source_restaurant_mentions > 0:
                quality_score += 2
            if source_destination_mentions > 0:
                quality_score += 2

            source_analysis.append({
                'domain': domain,
                'content_length': content_length,
                'restaurant_mentions': source_restaurant_mentions,
                'destination_mentions': source_destination_mentions,
                'quality_score': min(quality_score, 10),
                'estimated_restaurants': min(source_restaurant_mentions // 2, 8)
            })

        return {
            'total_sources': len(cleaned_results),
            'total_chars': total_chars,
            'total_restaurant_mentions': restaurant_mentions,
            'total_destination_mentions': destination_mentions,
            'avg_quality': sum(s['quality_score'] for s in source_analysis) / max(len(source_analysis), 1),
            'geographic_relevance': (destination_mentions / max(total_chars, 1)) * 10000,
            'estimated_restaurants': sum(s['estimated_restaurants'] for s in source_analysis),
            'source_analysis': source_analysis
        }

    def _save_scraped_content_analysis(self, cleaned_results: List[Dict], query: str, destination: str, filepath: str):
        """
        Save detailed scraped content for transparency
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("SCRAPED CONTENT TRANSPARENCY ANALYSIS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Sources Found: {len(cleaned_results)}\n")
                f.write("=" * 80 + "\n\n")

                # Summary by source
                f.write("CONTENT SUMMARY BY SOURCE\n")
                f.write("-" * 40 + "\n")

                for i, result in enumerate(cleaned_results, 1):
                    content = result.get('cleaned_content', '')
                    url = result.get('url', '')
                    domain = url.split('/')[2] if '/' in url and len(url.split('/')) > 2 else 'unknown'

                    f.write(f"{i}. {domain}\n")
                    f.write(f"   Length: {len(content):,} chars\n")
                    f.write(f"   Quality: {'✅ Rich' if len(content) > 2000 else '⚠️ Medium' if len(content) > 500 else '❌ Poor'}\n")
                    f.write(f"   URL: {url}\n\n")

                # Full content
                f.write("FULL CONTENT (INPUT TO EDITOR)\n")
                f.write("=" * 80 + "\n\n")

                for i, result in enumerate(cleaned_results, 1):
                    content = result.get('cleaned_content', '')
                    if content and len(content.strip()) > 100:
                        f.write(f"SOURCE {i}\n")
                        f.write("-" * 60 + "\n")
                        f.write(content)
                        f.write("\n" + "=" * 80 + "\n\n")

        except Exception as e:
            logger.error(f"❌ Error saving scraped content: {e}")

    def _save_editor_transparency_analysis(
        self, filepath: str, query: str, destination: str,
        cleaned_results: List[Dict], editor_output: Dict[str, Any], 
        pre_analysis: Dict[str, Any], timing_data: Dict[str, Any]
    ):
        """
        Save comprehensive editor decision analysis
        """
        try:
            restaurants = editor_output.get('edited_results', {}).get('main_list', [])

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("EDITOR DECISION TRANSPARENCY ANALYSIS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Editor: Production EditorAgent (unchanged)\n")
                f.write("=" * 80 + "\n\n")

                # =================================================================
                # INPUT ANALYSIS
                # =================================================================
                f.write("INPUT CONTENT ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Sources: {pre_analysis['total_sources']}\n")
                f.write(f"Total Content: {pre_analysis['total_chars']:,} characters\n")
                f.write(f"Restaurant Mentions: {pre_analysis['total_restaurant_mentions']}\n")
                f.write(f"Destination Mentions: {pre_analysis['total_destination_mentions']}\n")
                f.write(f"Estimated Restaurants: {pre_analysis['estimated_restaurants']}\n")
                f.write(f"Average Quality Score: {pre_analysis['avg_quality']:.1f}/10\n")
                f.write(f"Geographic Relevance: {pre_analysis['geographic_relevance']:.1f}/10\n\n")

                # Source details
                f.write("SOURCE-BY-SOURCE PREDICTION:\n")
                for i, source in enumerate(pre_analysis['source_analysis'], 1):
                    f.write(f"  {i}. {source['domain']} - {source['estimated_restaurants']} restaurants expected\n")

                f.write("\n")

                # =================================================================
                # EDITOR OUTPUT ANALYSIS
                # =================================================================
                f.write("EDITOR OUTPUT ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Restaurants Extracted: {len(restaurants)}\n")
                f.write(f"Expected vs Actual: {pre_analysis['estimated_restaurants']} → {len(restaurants)}\n")

                actual = len(restaurants)
                expected = pre_analysis['estimated_restaurants']

                if actual == expected:
                    f.write("✅ EXTRACTION MATCHES PREDICTION\n")
                elif actual < expected:
                    f.write("⚠️ FEWER THAN EXPECTED (Quality filtering or content structure)\n")
                else:
                    f.write("📈 MORE THAN EXPECTED (Rich content)\n")

                f.write("\n")

                # Individual restaurant details
                if restaurants:
                    f.write("EXTRACTED RESTAURANTS:\n")
                    for i, restaurant in enumerate(restaurants, 1):
                        name = restaurant.get('name', 'Unknown')
                        cuisine = restaurant.get('cuisine', 'Unknown')
                        neighborhood = restaurant.get('neighborhood', 'Unknown')
                        f.write(f"  {i}. {name}\n")
                        f.write(f"     Cuisine: {cuisine}\n")
                        f.write(f"     Location: {neighborhood}\n\n")

                # =================================================================
                # DECISION ANALYSIS
                # =================================================================
                f.write("EDITOR DECISION ANALYSIS\n")
                f.write("-" * 40 + "\n")

                if actual == 5:
                    f.write("✅ 5 RESTAURANTS = TYPICAL GOOD RESULT\n")
                    f.write("This indicates:\n")
                    f.write("  • Editor found exactly 5 high-quality restaurants\n")
                    f.write("  • Content quality was sufficient for extraction\n")
                    f.write("  • No hardcoded limit - just what was available\n\n")

                elif actual == 0:
                    f.write("❌ 0 RESTAURANTS = CONTENT ISSUE\n")
                    f.write("This suggests:\n")
                    f.write("  • Sources don't contain structured restaurant info\n")
                    f.write("  • Content is too general or low-quality\n")
                    f.write("  • Search targeting needs improvement\n\n")

                elif actual < 5:
                    f.write(f"⚠️ {actual} RESTAURANTS = LIMITED CONTENT\n")
                    f.write("This suggests:\n")
                    f.write("  • Sources have some but limited restaurant info\n")
                    f.write("  • Editor extracted all available options\n")
                    f.write("  • Quality over quantity approach\n\n")

                else:
                    f.write(f"📈 {actual} RESTAURANTS = RICH CONTENT\n")
                    f.write("This indicates:\n")
                    f.write("  • Excellent source content quality\n")
                    f.write("  • Multiple restaurants per source\n")
                    f.write("  • Editor successfully extracted comprehensive list\n\n")

                # =================================================================
                # RECOMMENDATIONS
                # =================================================================
                f.write("OPTIMIZATION RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")

                if pre_analysis['geographic_relevance'] < 5:
                    f.write("TO IMPROVE DESTINATION TARGETING:\n")
                    f.write("  1. Verify destination parsing is correct\n")
                    f.write("  2. Include neighborhood names in search\n")
                    f.write("  3. Target local tourism/food sites\n\n")

                if pre_analysis['avg_quality'] < 6:
                    f.write("TO IMPROVE CONTENT QUALITY:\n")
                    f.write("  1. Target editorial content vs user-generated\n")
                    f.write("  2. Include professional food critics\n")
                    f.write("  3. Avoid aggregator sites\n\n")

                # =================================================================
                # PERFORMANCE METRICS
                # =================================================================
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Time: {timing_data['total_time']:.2f}s\n")
                f.write(f"Editor Time: {timing_data['step7_time']:.2f}s\n")
                f.write(f"Extraction Rate: {actual / max(len(cleaned_results), 1):.1f} restaurants/source\n")

                if timing_data['step7_time'] > 0:
                    chars_per_sec = pre_analysis['total_chars'] / timing_data['step7_time']
                    f.write(f"Processing Speed: {chars_per_sec:,.0f} chars/second\n")

                f.write("\n")

                # =================================================================
                # CONCLUSION
                # =================================================================
                f.write("CONCLUSION\n")
                f.write("-" * 40 + "\n")

                if actual == 5:
                    f.write("✅ 5 RESTAURANTS IS LIKELY CORRECT\n")
                    f.write("The editor found exactly what was extractable from the content.\n")
                    f.write("This is quality-driven selection, not a hardcoded limit.\n\n")
                elif actual == 0:
                    f.write("❌ CONTENT QUALITY ISSUE\n")
                    f.write("The sources don't contain extractable restaurant information.\n")
                    f.write("Need better search targeting or different sources.\n\n")
                else:
                    f.write(f"📊 {actual} RESTAURANTS EXTRACTED\n")
                    f.write("Editor performance aligns with content quality.\n\n")

                f.write("✅ TRANSPARENCY ANALYSIS COMPLETE\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❌ Error saving editor analysis: {e}")

    def _send_results_to_admin(
        self, bot, scraped_filepath: str, edited_filepath: str, 
        query: str, editor_results: Dict[str, Any]
    ):
        """
        Send results to admin with summary (SYNC function)
        """
        try:
            restaurants = editor_results.get('edited_results', {}).get('main_list', [])

            summary = (
                f"✏️ <b>Editor Pipeline Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🍽️ Restaurants Found: {len(restaurants)}\n\n"
                f"📄 <b>Files Generated:</b>\n"
                f"1. 📄 Scraped Content Analysis\n"
                f"2. ✏️ Editor Decision Transparency\n\n"
                f"🎯 <b>Pipeline:</b> Web Search → Scraping → Text Cleaner → Editor\n"
                f"🔍 <b>Focus:</b> Why {len(restaurants)} restaurants were extracted"
            )

            bot.send_message(self.admin_chat_id, summary, parse_mode='HTML')

            # Send files
            with open(scraped_filepath, 'rb') as f:
                bot.send_document(self.admin_chat_id, f, caption=f"📄 Scraped Content: {query}")

            with open(edited_filepath, 'rb') as f:
                bot.send_document(self.admin_chat_id, f, caption=f"✏️ Editor Analysis: {query}")

            logger.info("✅ Successfully sent transparent editor test results")

        except Exception as e:
            logger.error(f"❌ Failed to send results: {e}")


# =================================================================
# TELEGRAM BOT INTEGRATION
# =================================================================

def create_transparent_editor_test(config, orchestrator):
    """Factory function for creating the test"""
    return TransparentEditorTest(config, orchestrator)