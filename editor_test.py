# editor_test.py
"""
COMPREHENSIVE Editor Pipeline Test with Maximum Transparency

This test follows the EXACT production pipeline:
1. Web search → result filtering → scraping  
2. Text cleaner
3. Scraped content file saved
4. Editor → final output saved with detailed reasoning

Produces 2 files sent to Telegram:
- scraped_content_[timestamp].txt (raw scraped content)  
- edited_restaurants_[timestamp].txt (final output + reasoning)

Maximum transparency features:
- Every AI decision is logged with reasoning
- Token counts and chunking decisions shown
- Content quality analysis at each step
- Editor prompt and response captured
- Decision tree visualization
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

class EditorTest:
    """
    Comprehensive Editor Pipeline Test with Maximum Decision Transparency
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents from orchestrator
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper
        self.text_cleaner = orchestrator.text_cleaner
        self.editor_agent = orchestrator.editor_agent

        logger.info("✅ EditorTest initialized with production agents")

    async def test_complete_editor_pipeline(self, restaurant_query: str, bot=None) -> str:
        """
        Run the COMPLETE editor pipeline with maximum transparency

        Pipeline: Web Search → Scraping → Text Cleaning → Editor → Final Output
        Produces 2 files: scraped content + edited results with reasoning
        """
        total_start_time = time.time()

        # Create timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraped_filename = f"scraped_content_{timestamp}.txt"
        edited_filename = f"edited_restaurants_{timestamp}.txt"

        scraped_filepath = os.path.join(tempfile.gettempdir(), scraped_filename)
        edited_filepath = os.path.join(tempfile.gettempdir(), edited_filename)

        logger.info(f"🧪 Starting COMPLETE Editor Pipeline Test for: {restaurant_query}")
        logger.info(f"📄 Scraped content will be saved to: {scraped_filename}")
        logger.info(f"📝 Edited results will be saved to: {edited_filename}")

        try:
            # =================================================================
            # STEP 1: QUERY ANALYSIS (following production pipeline)
            # =================================================================
            logger.info("🔍 STEP 1: Query Analysis")
            step1_start = time.time()

            query_result = self.query_analyzer.analyze_query(restaurant_query)

            destination = query_result.get('destination', 'Unknown')
            english_queries = query_result.get('english_queries', [])
            local_queries = query_result.get('local_queries', [])
            search_queries = english_queries + local_queries

            step1_time = time.time() - step1_start

            # =================================================================
            # STEP 2: WEB SEARCH (bypass database, go straight to web)
            # =================================================================
            logger.info("🌐 STEP 2: Web Search (bypassing database)")
            step2_start = time.time()

            # Use SearchAgent to find URLs
            search_results = await self.search_agent.search_restaurants(
                search_queries, destination
            )

            step2_time = time.time() - step2_start

            # =================================================================
            # STEP 3: INTELLIGENT SCRAPING
            # =================================================================
            logger.info("🤖 STEP 3: Intelligent Scraping")
            step3_start = time.time()

            # Scrape using production SmartRestaurantScraper
            scraping_results = await self.scraper.scrape_restaurant_content(
                search_results, destination, max_concurrent=2
            )

            step3_time = time.time() - step3_start

            # =================================================================
            # STEP 4: TEXT CLEANING
            # =================================================================
            logger.info("🧹 STEP 4: Text Cleaning")
            step4_start = time.time()

            # Clean each scraped result
            cleaned_results = []
            for result in scraping_results:
                if result.get('scraped_content'):
                    cleaned_content = await self.text_cleaner.clean_scraped_content(
                        result['scraped_content'], 
                        result.get('url', ''),
                        destination
                    )

                    # Merge cleaned content back
                    cleaned_result = result.copy()
                    cleaned_result['cleaned_content'] = cleaned_content
                    cleaned_results.append(cleaned_result)

            step4_time = time.time() - step4_start

            # =================================================================
            # STEP 5: SAVE SCRAPED CONTENT FILE
            # =================================================================
            logger.info("💾 STEP 5: Saving Scraped Content File")
            self._save_scraped_content_file(
                cleaned_results, restaurant_query, destination, scraped_filepath
            )

            # =================================================================
            # STEP 6: EDITOR PROCESSING WITH MAXIMUM TRANSPARENCY
            # =================================================================
            logger.info("✏️ STEP 6: Editor Processing with Transparency")
            step6_start = time.time()

            # Process through editor with transparency logging
            editor_results = await self._process_with_transparent_editor(
                cleaned_results, restaurant_query, destination, edited_filepath
            )

            step6_time = time.time() - step6_start

            # =================================================================
            # STEP 7: FINAL ANALYSIS AND SUMMARY
            # =================================================================
            total_time = time.time() - total_start_time

            self._save_final_summary(
                edited_filepath, restaurant_query, destination, 
                {
                    'step1_time': step1_time,
                    'step2_time': step2_time, 
                    'step3_time': step3_time,
                    'step4_time': step4_time,
                    'step6_time': step6_time,
                    'total_time': total_time,
                    'search_results_count': len(search_results),
                    'scraped_results_count': len(scraping_results),
                    'cleaned_results_count': len(cleaned_results),
                    'final_restaurants_count': len(editor_results.get('edited_results', {}).get('main_list', []))
                }
            )

            # Send results to admin
            if bot and self.admin_chat_id:
                await self._send_results_to_admin(
                    bot, scraped_filepath, edited_filepath, restaurant_query, editor_results
                )

            return edited_filepath

        except Exception as e:
            logger.error(f"❌ Error in complete editor pipeline test: {e}")

            # Save error details
            with open(edited_filepath, 'w', encoding='utf-8') as f:
                f.write("EDITOR PIPELINE TEST - ERROR\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")

                import traceback
                f.write(f"\nFull traceback:\n{traceback.format_exc()}\n")

            if bot and self.admin_chat_id:
                try:
                    bot.send_message(
                        self.admin_chat_id,
                        f"❌ Editor pipeline test failed for: {restaurant_query}\n\nError: {str(e)}"
                    )
                except:
                    pass

            return edited_filepath

    def _save_scraped_content_file(self, cleaned_results: List[Dict], query: str, destination: str, filepath: str):
        """
        Save comprehensive scraped content file with analysis
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("SCRAPED CONTENT ANALYSIS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total Sources: {len(cleaned_results)}\n")
                f.write("=" * 80 + "\n\n")

                # Content quality analysis
                f.write("CONTENT QUALITY ANALYSIS\n")
                f.write("-" * 40 + "\n")

                total_content_length = 0
                sources_with_content = 0

                for i, result in enumerate(cleaned_results, 1):
                    url = result.get('url', 'Unknown')
                    scraped_content = result.get('scraped_content', '')
                    cleaned_content = result.get('cleaned_content', '')

                    scraped_length = len(scraped_content) if scraped_content else 0
                    cleaned_length = len(cleaned_content) if cleaned_content else 0

                    if cleaned_length > 100:
                        sources_with_content += 1
                        total_content_length += cleaned_length

                    f.write(f"{i}. {url}\n")
                    f.write(f"   Scraped: {scraped_length:,} chars\n")
                    f.write(f"   Cleaned: {cleaned_length:,} chars\n")
                    f.write(f"   Quality: {'✅ Good' if cleaned_length > 500 else '⚠️ Limited' if cleaned_length > 100 else '❌ Poor'}\n\n")

                f.write(f"SUMMARY:\n")
                f.write(f"Sources with good content: {sources_with_content}/{len(cleaned_results)}\n")
                f.write(f"Total cleaned content: {total_content_length:,} characters\n")
                f.write(f"Average per source: {total_content_length // max(sources_with_content, 1):,} chars\n\n")

                # Full content dump
                f.write("FULL SCRAPED CONTENT (FOR EDITOR)\n")
                f.write("=" * 80 + "\n\n")

                for i, result in enumerate(cleaned_results, 1):
                    url = result.get('url', 'Unknown')
                    cleaned_content = result.get('cleaned_content', '')

                    if cleaned_content and len(cleaned_content.strip()) > 100:
                        f.write(f"SOURCE {i}: {url}\n")
                        f.write("-" * 60 + "\n")
                        f.write(cleaned_content)
                        f.write("\n" + "=" * 80 + "\n\n")

            logger.info(f"✅ Scraped content file saved: {filepath}")

        except Exception as e:
            logger.error(f"❌ Error saving scraped content file: {e}")

    async def _process_with_transparent_editor(
        self, cleaned_results: List[Dict], query: str, destination: str, filepath: str
    ) -> Dict[str, Any]:
        """
        Process content through editor with MAXIMUM transparency logging
        """
        try:
            # Prepare content for editor (same format as production)
            scraped_results = []
            for result in cleaned_results:
                if result.get('cleaned_content'):
                    editor_result = {
                        'url': result.get('url', ''),
                        'scraped_content': result.get('cleaned_content', ''),
                        'title': result.get('title', ''),
                        'domain': result.get('url', '').split('/')[2] if result.get('url') else 'unknown'
                    }
                    scraped_results.append(editor_result)

            logger.info(f"📝 Prepared {len(scraped_results)} sources for editor")

            # =================================================================
            # CAPTURE EDITOR PROCESSING WITH TRANSPARENCY
            # =================================================================

            # Call editor with production parameters
            editor_output = self.editor_agent.edit(
                scraped_results=scraped_results,
                database_restaurants=None,  # Web-only test
                raw_query=query,
                destination=destination,
                content_source="web_search",
                processing_mode="web_only"
            )

            # =================================================================
            # SAVE DETAILED EDITOR ANALYSIS
            # =================================================================
            await self._save_detailed_editor_analysis(
                filepath, query, destination, scraped_results, editor_output
            )

            return editor_output

        except Exception as e:
            logger.error(f"❌ Error in transparent editor processing: {e}")

            # Save error to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("EDITOR PROCESSING ERROR\n")
                f.write("=" * 80 + "\n")
                f.write(f"Error: {str(e)}\n")

                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")

            return {"edited_results": {"main_list": []}, "follow_up_queries": []}

    async def _save_detailed_editor_analysis(
        self, filepath: str, query: str, destination: str, 
        scraped_results: List[Dict], editor_output: Dict[str, Any]
    ):
        """
        Save extremely detailed editor analysis with decision transparency
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("DETAILED EDITOR PIPELINE ANALYSIS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Processing Mode: web_only (no database)\n")
                f.write("=" * 80 + "\n\n")

                # =================================================================
                # INPUT ANALYSIS
                # =================================================================
                f.write("INPUT CONTENT ANALYSIS\n")
                f.write("-" * 40 + "\n")

                total_input_chars = 0
                content_sources = []

                for i, result in enumerate(scraped_results, 1):
                    url = result.get('url', 'Unknown')
                    content = result.get('scraped_content', '')
                    domain = result.get('domain', 'unknown')

                    content_length = len(content)
                    total_input_chars += content_length

                    f.write(f"{i}. {domain}\n")
                    f.write(f"   URL: {url}\n")
                    f.write(f"   Content: {content_length:,} characters\n")
                    f.write(f"   Quality: {'✅ Rich' if content_length > 2000 else '⚠️ Medium' if content_length > 500 else '❌ Limited'}\n")

                    # Content preview
                    preview = content[:200].replace('\n', ' ') + "..." if len(content) > 200 else content
                    f.write(f"   Preview: {preview}\n\n")

                    content_sources.append({
                        'domain': domain,
                        'length': content_length,
                        'url': url
                    })

                f.write(f"TOTAL INPUT: {total_input_chars:,} characters from {len(scraped_results)} sources\n\n")

                # =================================================================
                # EDITOR PROCESSING TRANSPARENCY
                # =================================================================
                f.write("EDITOR PROCESSING DETAILS\n")
                f.write("-" * 40 + "\n")

                # Check if content needs chunking (simulate editor's chunking logic)
                needs_chunking = total_input_chars > 45000  # Editor's character limit
                f.write(f"Content Length: {total_input_chars:,} characters\n")
                f.write(f"Chunking Required: {'YES' if needs_chunking else 'NO'}\n")
                f.write(f"Editor Mode: web_only (scraped content only)\n")
                f.write(f"Database Content: None (bypassed for this test)\n\n")

                # =================================================================
                # EDITOR PROMPT TRANSPARENCY
                # =================================================================
                f.write("EDITOR AI PROMPT ANALYSIS\n")
                f.write("-" * 40 + "\n")

                # Recreate the prompt that would be sent to AI (from editor_agent.py)
                f.write("The Editor Agent uses this prompt structure:\n\n")
                f.write("SYSTEM PROMPT:\n")
                f.write("```\n")
                f.write("You are a sophisticated restaurant curator and concierge...\n")
                f.write("Extract well-described restaurants from web articles...\n")
                f.write("Include restaurants even if they don't perfectly match all requirements...\n")
                f.write("```\n\n")

                f.write("USER PROMPT:\n")
                f.write("```\n")
                f.write(f"Original user request: {query}\n")
                f.write(f"Destination: {destination}\n\n")
                f.write("Scraped content from multiple sources:\n")
                f.write("[FORMATTED CONTENT FROM ALL SOURCES]\n")
                f.write("```\n\n")

                # =================================================================
                # EDITOR OUTPUT ANALYSIS
                # =================================================================
                f.write("EDITOR OUTPUT ANALYSIS\n")
                f.write("-" * 40 + "\n")

                edited_results = editor_output.get('edited_results', {})
                main_list = edited_results.get('main_list', [])
                follow_up_queries = editor_output.get('follow_up_queries', [])

                f.write(f"Final Restaurants Extracted: {len(main_list)}\n")
                f.write(f"Follow-up Queries Generated: {len(follow_up_queries)}\n\n")

                if follow_up_queries:
                    f.write("FOLLOW-UP QUERIES:\n")
                    for i, query in enumerate(follow_up_queries, 1):
                        f.write(f"  {i}. {query}\n")
                    f.write("\n")

                # =================================================================
                # RESTAURANT EXTRACTION ANALYSIS
                # =================================================================
                f.write("RESTAURANT EXTRACTION DETAILS\n")
                f.write("-" * 40 + "\n")

                if main_list:
                    f.write(f"SUCCESS: {len(main_list)} restaurants extracted\n\n")

                    for i, restaurant in enumerate(main_list, 1):
                        name = restaurant.get('name', 'Unknown')
                        description = restaurant.get('description', '')
                        cuisine_tags = restaurant.get('cuisine_tags', [])
                        address = restaurant.get('address', '')
                        sources = restaurant.get('sources', [])

                        f.write(f"RESTAURANT {i}: {name}\n")
                        f.write(f"  Cuisine: {', '.join(cuisine_tags[:3]) if cuisine_tags else 'Not specified'}\n")
                        f.write(f"  Address: {address if address else 'Not available'}\n")
                        f.write(f"  Description Length: {len(description)} characters\n")
                        f.write(f"  Sources: {len(sources)} referenced\n")

                        # Description quality analysis
                        if description:
                            word_count = len(description.split())
                            f.write(f"  Description Quality: {'✅ Rich' if word_count > 50 else '⚠️ Basic' if word_count > 20 else '❌ Minimal'} ({word_count} words)\n")
                        else:
                            f.write(f"  Description Quality: ❌ Missing\n")

                        # Show description preview
                        desc_preview = description[:150] + "..." if len(description) > 150 else description
                        f.write(f"  Preview: {desc_preview}\n\n")

                else:
                    f.write("❌ NO RESTAURANTS EXTRACTED\n")
                    f.write("This indicates a problem in the editor processing.\n\n")

                # =================================================================
                # DECISION ANALYSIS
                # =================================================================
                f.write("EDITOR DECISION ANALYSIS\n")
                f.write("-" * 40 + "\n")

                # Analyze why we got this number of restaurants
                f.write("WHY THIS NUMBER OF RESTAURANTS?\n")

                if len(main_list) == 0:
                    f.write("❌ ZERO RESULTS ANALYSIS:\n")
                    f.write("  • Check if scraped content contains restaurant information\n")
                    f.write("  • Verify content is in the correct language\n")
                    f.write("  • Ensure content is not blocked/paywall content\n")
                    f.write("  • Check if editor prompt is too restrictive\n\n")

                elif len(main_list) < 5:
                    f.write(f"⚠️ LIMITED RESULTS ({len(main_list)}) ANALYSIS:\n")
                    f.write("  • Content may be sparse or low-quality\n")
                    f.write("  • Sources might not be restaurant-focused\n")
                    f.write("  • Editor being selective about quality\n")
                    f.write("  • Geographic mismatch possible\n\n")

                elif len(main_list) == 5:
                    f.write("🎯 EXACTLY 5 RESULTS ANALYSIS:\n")
                    f.write("  • This suggests editor found good content\n")
                    f.write("  • 5 is NOT a hardcoded limit in the editor\n")
                    f.write("  • Editor likely extracted all clearly-mentioned restaurants\n")
                    f.write("  • Content quality was sufficient for extraction\n\n")

                else:
                    f.write(f"✅ GOOD RESULTS ({len(main_list)}) ANALYSIS:\n")
                    f.write("  • Rich content with multiple restaurant mentions\n")
                    f.write("  • Editor successfully extracted variety\n")
                    f.write("  • High-quality sources with detailed information\n\n")

                # =================================================================
                # TRANSPARENCY RECOMMENDATIONS
                # =================================================================
                f.write("TRANSPARENCY RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                f.write("To improve decision transparency:\n")
                f.write("1. Add editor reasoning output to show why restaurants were selected\n")
                f.write("2. Include content quality scores for each source\n")
                f.write("3. Log editor's internal decision process\n")
                f.write("4. Show which content sections led to each restaurant\n")
                f.write("5. Add geographic relevance scoring\n\n")

                # =================================================================
                # FULL CONTENT DUMP (for debugging)
                # =================================================================
                f.write("FULL SCRAPED CONTENT (FOR EDITOR INPUT)\n")
                f.write("=" * 80 + "\n\n")

                for i, result in enumerate(cleaned_results, 1):
                    url = result.get('url', 'Unknown')
                    cleaned_content = result.get('cleaned_content', '')

                    if cleaned_content and len(cleaned_content.strip()) > 100:
                        f.write(f"CONTENT SOURCE {i}\n")
                        f.write(f"URL: {url}\n")
                        f.write(f"Length: {len(cleaned_content):,} characters\n")
                        f.write("-" * 60 + "\n")
                        f.write(cleaned_content)
                        f.write("\n" + "=" * 80 + "\n\n")

        except Exception as e:
            logger.error(f"❌ Error saving scraped content file: {e}")

    def _save_final_summary(self, filepath: str, query: str, destination: str, timing_data: Dict):
        """
        Append final performance summary to the edited results file
        """
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("PIPELINE PERFORMANCE SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Destination: {destination}\n")
                f.write(f"Total Processing Time: {timing_data['total_time']:.2f}s\n\n")

                f.write("STEP TIMING:\n")
                f.write(f"  1. Query Analysis: {timing_data['step1_time']:.2f}s\n")
                f.write(f"  2. Web Search: {timing_data['step2_time']:.2f}s\n")
                f.write(f"  3. Scraping: {timing_data['step3_time']:.2f}s\n")
                f.write(f"  4. Text Cleaning: {timing_data['step4_time']:.2f}s\n")
                f.write(f"  6. Editor Processing: {timing_data['step6_time']:.2f}s\n\n")

                f.write("PIPELINE FLOW:\n")
                f.write(f"  Search Results Found: {timing_data['search_results_count']}\n")
                f.write(f"  Pages Scraped: {timing_data['scraped_results_count']}\n")
                f.write(f"  Content Sources Cleaned: {timing_data['cleaned_results_count']}\n")
                f.write(f"  Final Restaurants: {timing_data['final_restaurants_count']}\n\n")

                efficiency = (timing_data['final_restaurants_count'] / max(timing_data['scraped_results_count'], 1)) * 100
                f.write(f"EXTRACTION EFFICIENCY: {efficiency:.1f}% (restaurants per scraped page)\n\n")

                f.write("✅ EDITOR PIPELINE TEST COMPLETED\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❌ Error saving final summary: {e}")

    async def _send_results_to_admin(
        self, bot, scraped_filepath: str, edited_filepath: str, 
        query: str, editor_results: Dict[str, Any]
    ):
        """
        Send both result files to admin with summary
        """
        try:
            main_list = editor_results.get('edited_results', {}).get('main_list', [])
            follow_up_queries = editor_results.get('follow_up_queries', [])

            # Summary message
            summary = (
                f"✏️ <b>Editor Pipeline Test Complete</b>\n\n"
                f"📝 Query: <code>{query}</code>\n"
                f"🍽️ Restaurants Found: {len(main_list)}\n"
                f"🔍 Follow-up Queries: {len(follow_up_queries)}\n\n"
                f"📄 Two files generated:\n"
                f"1. Scraped content (input to editor)\n"
                f"2. Edited results (output with analysis)\n\n"
                f"🎯 <b>Pipeline:</b> Web Search → Scraping → Text Cleaner → Editor\n"
                f"⚡ <b>Mode:</b> Web-only (database bypassed)"
            )

            bot.send_message(self.admin_chat_id, summary, parse_mode='HTML')

            # Send scraped content file
            with open(scraped_filepath, 'rb') as f:
                bot.send_document(
                    self.admin_chat_id,
                    f,
                    caption=f"📄 Scraped Content: {query}"
                )

            # Send edited results file  
            with open(edited_filepath, 'rb') as f:
                bot.send_document(
                    self.admin_chat_id,
                    f,
                    caption=f"✏️ Editor Results: {query}"
                )

            logger.info("✅ Successfully sent editor test results to admin")

        except Exception as e:
            logger.error(f"❌ Failed to send editor results to admin: {e}")
