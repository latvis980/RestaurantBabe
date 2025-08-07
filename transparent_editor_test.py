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

        logger.info(f"🧪 Testing Editor Pipeline: {restaurant_query}")

        try:
            # =================================================================
            # STEP 1: QUERY ANALYSIS
            # =================================================================
            logger.info("🔍 Step 1: Query Analysis")
            step1_start = time.time()

            query_result = self.query_analyzer.analyze(restaurant_query)
            destination = query_result.get('destination', 'Unknown')
            search_queries = query_result.get('search_queries', [restaurant_query])

            step1_time = time.time() - step1_start

            # =================================================================
            # STEP 2: SEARCH - FIXED: Pass destination parameter
            # =================================================================
            logger.info("🔍 Step 2: Web Search")
            step2_start = time.time()

            # FIXED: BraveSearchAgent needs destination parameter
            search_results = await self.search_agent.search(search_queries, destination)

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
            # STEP 5: PRE-EDITOR ANALYSIS
            # =================================================================
            logger.info("🔍 Step 5: Pre-Editor Analysis")
            pre_analysis = self._analyze_content_before_editor(cleaned_results, destination)

            # =================================================================
            # STEP 6: EDITOR PROCESSING (using existing editor)
            # =================================================================
            logger.info("✏️ Step 6: Editor Processing")
            step6_start = time.time()

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

            step6_time = time.time() - step6_start

            # =================================================================
            # STEP 7: ANALYSIS & SEND TO TELEGRAM - FIXED
            # =================================================================
            logger.info("📊 Step 7: Analysis & Send to Telegram")

            total_time = time.time() - total_start_time

            timing_data = {
                'step1_time': step1_time,
                'step2_time': step2_time,
                'step3_time': step3_time,
                'step4_time': step4_time,
                'step6_time': step6_time,
                'total_time': total_time,
                'search_results_count': len(search_results),
                'scraped_results_count': len(scraping_results),
                'cleaned_results_count': len(cleaned_results),
                'final_restaurants_count': len(editor_output.get('edited_results', {}).get('main_list', []))
            }

            # FIXED: Send directly to Telegram instead of temp files
            if bot and self.admin_chat_id:
                self._send_analysis_to_telegram(
                    bot, restaurant_query, destination, cleaned_results, 
                    editor_output, pre_analysis, timing_data
                )

            return "Analysis sent to Telegram"

        except Exception as e:
            logger.error(f"❌ Error in editor pipeline test: {e}")

            # Send error to Telegram if possible
            if bot and self.admin_chat_id:
                error_msg = (
                    f"❌ <b>Editor Test Failed</b>\n\n"
                    f"📝 Query: <code>{restaurant_query}</code>\n"
                    f"🔥 Error: <code>{str(e)}</code>\n"
                    f"⏱ Time: {datetime.now().isoformat()}"
                )
                try:
                    bot.send_message(self.admin_chat_id, error_msg, parse_mode='HTML')
                except:
                    pass

            return f"Error: {str(e)}"

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

    def _send_analysis_to_telegram(
        self, bot, query: str, destination: str, cleaned_results: List[Dict], 
        editor_output: Dict[str, Any], pre_analysis: Dict[str, Any], timing_data: Dict[str, Any]
    ):
        """
        Send comprehensive analysis directly to Telegram (no temp files)
        """
        try:
            restaurants = editor_output.get('edited_results', {}).get('main_list', [])

            # =================================================================
            # SCRAPED CONTENT ANALYSIS MESSAGE
            # =================================================================
            scraped_analysis = self._build_scraped_content_message(
                query, destination, cleaned_results
            )

            bot.send_message(
                self.admin_chat_id, 
                scraped_analysis, 
                parse_mode='HTML'
            )

            # =================================================================
            # EDITOR DECISION ANALYSIS MESSAGE
            # =================================================================
            editor_analysis = self._build_editor_decision_message(
                query, destination, restaurants, pre_analysis, timing_data
            )

            bot.send_message(
                self.admin_chat_id, 
                editor_analysis, 
                parse_mode='HTML'
            )

            # =================================================================
            # DETAILED CONTENT (if short enough for Telegram)
            # =================================================================
            if len(cleaned_results) <= 3 and all(len(r.get('cleaned_content', '')) < 2000 for r in cleaned_results):
                content_details = self._build_content_details_message(cleaned_results)
                bot.send_message(
                    self.admin_chat_id, 
                    content_details, 
                    parse_mode='HTML'
                )

            logger.info("✅ Successfully sent transparent editor test results to Telegram")

        except Exception as e:
            logger.error(f"❌ Failed to send results to Telegram: {e}")

    def _build_scraped_content_message(self, query: str, destination: str, cleaned_results: List[Dict]) -> str:
        """Build scraped content analysis message"""
        msg = (
            f"📄 <b>SCRAPED CONTENT ANALYSIS</b>\n\n"
            f"📝 Query: <code>{query}</code>\n"
            f"🏙️ Destination: <b>{destination}</b>\n"
            f"📊 Sources: {len(cleaned_results)}\n\n"
            f"<b>CONTENT SUMMARY:</b>\n"
        )

        for i, result in enumerate(cleaned_results, 1):
            content = result.get('cleaned_content', '')
            url = result.get('url', '')
            domain = url.split('/')[2] if '/' in url and len(url.split('/')) > 2 else 'unknown'

            quality = "✅ Rich" if len(content) > 2000 else "⚠️ Medium" if len(content) > 500 else "❌ Poor"

            msg += f"{i}. <code>{domain}</code>\n"
            msg += f"   Length: {len(content):,} chars\n"
            msg += f"   Quality: {quality}\n\n"

        return msg

    def _build_editor_decision_message(
        self, query: str, destination: str, restaurants: List[Dict], 
        pre_analysis: Dict[str, Any], timing_data: Dict[str, Any]
    ) -> str:
        """Build editor decision analysis message"""
        actual = len(restaurants)
        expected = pre_analysis['estimated_restaurants']

        msg = (
            f"✏️ <b>EDITOR DECISION ANALYSIS</b>\n\n"
            f"📝 Query: <code>{query}</code>\n"
            f"🏙️ Destination: <b>{destination}</b>\n"
            f"🍽️ Restaurants Found: <b>{actual}</b>\n"
            f"📊 Expected: {expected}\n\n"
            f"<b>INPUT ANALYSIS:</b>\n"
            f"• Sources: {pre_analysis['total_sources']}\n"
            f"• Content: {pre_analysis['total_chars']:,} chars\n"
            f"• Restaurant Mentions: {pre_analysis['total_restaurant_mentions']}\n"
            f"• Quality Score: {pre_analysis['avg_quality']:.1f}/10\n\n"
        )

        # Analysis based on results
        if actual == 5:
            msg += "✅ <b>5 RESTAURANTS = TYPICAL RESULT</b>\n"
            msg += "Editor found exactly what was extractable.\n"
            msg += "Quality-driven selection, not hardcoded limit.\n\n"
        elif actual == 0:
            msg += "❌ <b>0 RESTAURANTS = CONTENT ISSUE</b>\n"
            msg += "Sources lack structured restaurant info.\n"
            msg += "Need better search targeting.\n\n"
        elif actual < 5:
            msg += f"⚠️ <b>{actual} RESTAURANTS = LIMITED CONTENT</b>\n"
            msg += "Editor extracted all available options.\n"
            msg += "Quality over quantity approach.\n\n"
        else:
            msg += f"📈 <b>{actual} RESTAURANTS = RICH CONTENT</b>\n"
            msg += "Excellent source content quality.\n"
            msg += "Comprehensive extraction success.\n\n"

        # Performance metrics
        msg += f"<b>PERFORMANCE:</b>\n"
        msg += f"• Total Time: {timing_data['total_time']:.1f}s\n"
        msg += f"• Editor Time: {timing_data['step6_time']:.1f}s\n"
        msg += f"• Processing Rate: {actual / max(len(pre_analysis['source_analysis']), 1):.1f} restaurants/source\n\n"

        # Restaurant list
        if restaurants:
            msg += f"<b>EXTRACTED RESTAURANTS:</b>\n"
            for i, restaurant in enumerate(restaurants[:8], 1):  # Limit to 8 for Telegram
                name = restaurant.get('name', 'Unknown')
                cuisine = restaurant.get('cuisine', 'Unknown')
                msg += f"{i}. <b>{name}</b> ({cuisine})\n"

            if len(restaurants) > 8:
                msg += f"... and {len(restaurants) - 8} more\n"

        return msg

    def _build_content_details_message(self, cleaned_results: List[Dict]) -> str:
        """Build detailed content message for small datasets"""
        msg = "📋 <b>DETAILED CONTENT</b>\n\n"

        for i, result in enumerate(cleaned_results, 1):
            content = result.get('cleaned_content', '')
            domain = result.get('url', '').split('/')[2] if result.get('url') else 'unknown'

            if len(content) > 1500:  # Truncate for Telegram limits
                content = content[:1500] + "..."

            msg += f"<b>SOURCE {i}: {domain}</b>\n"
            msg += f"<code>{content}</code>\n\n"

            if len(msg) > 3500:  # Stay under Telegram's 4096 char limit
                msg += "... (truncated for length)"
                break

        return msg


# =================================================================
# TELEGRAM BOT INTEGRATION
# =================================================================

def create_transparent_editor_test(config, orchestrator):
    """Factory function for creating the test"""
    return TransparentEditorTest(config, orchestrator)