# scrape_test.py - UPDATED FOR SMART SCRAPER WITH SECTIONING & DOMAIN INTEL
# Full pipeline test with enhanced smart scraper features

import asyncio
import time
import tempfile
import os
import threading
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ScrapeTest:
    """
    UPDATED: Full pipeline test that uses SmartRestaurantScraper with new features.

    This test shows:
    - Complete database search and evaluation
    - NEW: Domain intelligence caching and learning
    - NEW: Content sectioning for restaurant extraction
    - NEW: Strategy cost tracking and optimization
    - Complete web search and smart scraping process
    - Editor agent processing with enhanced content

    Follows the EXACT production pipeline path with all enhancements.
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents directly from orchestrator (SmartRestaurantScraper, not wrapper)
        self.query_analyzer = orchestrator.query_analyzer
        self.database_search_agent = orchestrator.database_search_agent
        self.dbcontent_evaluation_agent = orchestrator.dbcontent_evaluation_agent  
        self.search_agent = orchestrator.search_agent
        self.scraper = orchestrator.scraper  # This is now SmartRestaurantScraper directly
        self.editor_agent = orchestrator.editor_agent

    async def test_scraping_process(self, restaurant_query: str, bot=None) -> str:
        """
        Run the COMPLETE production pipeline with enhanced smart scraping features.
        Shows domain intelligence, content sectioning, and cost optimization.
        """
        total_start_time = time.time()

        # Create timestamped filename for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        logger.info(f"Starting full pipeline test with enhanced SmartRestaurantScraper for: {restaurant_query}")
        logger.info(f"Results will be saved to: {filepath}")

        # Pipeline data collection
        pipeline_data = {}

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("ENHANCED FULL PIPELINE TEST RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test Type: Complete Enhanced Pipeline (Database → SmartScraper → Editor)\n")
                f.write("=" * 80 + "\n\n")

                f.write("ENHANCED PRODUCTION PIPELINE OVERVIEW:\n")
                f.write("1. Query Analysis → Search Terms Generation\n")
                f.write("2. Database Search → Existing Restaurant Check\n")
                f.write("3. Database Content Evaluation → Quality Assessment\n")
                f.write("4. Web Search → URL Discovery (if needed)\n")
                f.write("5. Enhanced Smart Scraper → Domain Intelligence + Sectioning\n")
                f.write("   ├── 🧠 Domain intelligence caching\n")
                f.write("   ├── 🤖 AI strategy classification\n")
                f.write("   ├── 🆓 Specialized (RSS/Sitemap)\n")
                f.write("   ├── 🟢 Simple HTTP + sectioning\n")
                f.write("   ├── 🟡 Enhanced HTTP + sectioning\n")
                f.write("   └── 🔴 Firecrawl (last resort)\n")
                f.write("6. Content Sectioning → Restaurant extraction\n")
                f.write("7. Editor Agent → Final recommendation compilation\n\n")

                # STEP 1: Query Analysis
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("=" * 60 + "\n")
                query_start_time = time.time()

                analyzed_query = self.query_analyzer.analyze(restaurant_query)
                query_time = time.time() - query_start_time
                pipeline_data['analyzed_query'] = analyzed_query

                f.write(f"Original Query: {restaurant_query}\n")
                f.write(f"Processing Time: {query_time:.2f}s\n")
                f.write(f"Search Terms Generated: {analyzed_query.get('search_terms', [])}\n")
                f.write(f"Location Extracted: {analyzed_query.get('location', 'Not specified')}\n")
                f.write(f"Cuisine Type: {analyzed_query.get('cuisine_type', 'Not specified')}\n")
                f.write(f"Price Range: {analyzed_query.get('price_range', 'Not specified')}\n\n")

                # STEP 2: Database Search
                f.write("STEP 2: DATABASE SEARCH\n")
                f.write("=" * 60 + "\n")
                db_search_start_time = time.time()

                database_results = await self.database_search_agent.search_restaurants(analyzed_query)
                db_search_time = time.time() - db_search_start_time
                pipeline_data['database_results'] = database_results

                f.write(f"Database Search Time: {db_search_time:.2f}s\n")
                f.write(f"Restaurants Found in Database: {len(database_results)}\n\n")

                if database_results:
                    f.write("DATABASE RESTAURANTS FOUND:\n")
                    for i, restaurant in enumerate(database_results[:10], 1):  # Show first 10
                        f.write(f"{i}. {restaurant.get('name', 'No Name')}\n")
                        f.write(f"   ID: {restaurant.get('id', 'No ID')}\n")
                        f.write(f"   Location: {restaurant.get('location', 'No location')}\n")
                        f.write(f"   Cuisine Tags: {restaurant.get('cuisine_tags', [])}\n")
                        f.write(f"   Rating: {restaurant.get('rating', 'No rating')}\n")
                        f.write(f"   Raw Description Length: {len(restaurant.get('raw_description', ''))}\n\n")
                else:
                    f.write("No restaurants found in database\n\n")

                # STEP 3: Database Content Evaluation
                f.write("STEP 3: DATABASE CONTENT EVALUATION\n")
                f.write("=" * 60 + "\n")

                if database_results:
                    eval_start_time = time.time()
                    evaluation_result = await self.dbcontent_evaluation_agent.evaluate_database_content(
                        analyzed_query, database_results
                    )
                    eval_time = time.time() - eval_start_time
                    pipeline_data['evaluation_result'] = evaluation_result

                    f.write(f"Evaluation Time: {eval_time:.2f}s\n")
                    f.write(f"Recommendation: {evaluation_result.get('recommendation', 'No recommendation')}\n")
                    f.write(f"Quality Score: {evaluation_result.get('quality_score', 'No score')}\n")
                    f.write(f"Content Gaps: {evaluation_result.get('content_gaps', [])}\n")
                    f.write(f"Reasoning: {evaluation_result.get('reasoning', 'No reasoning')}\n\n")

                    # Check if web search is needed
                    needs_web_search = evaluation_result.get('recommendation') == 'search_web'
                else:
                    f.write("No database content to evaluate - will proceed to web search\n\n")
                    needs_web_search = True

                # STEP 4: Web Search (if needed)
                if needs_web_search:
                    f.write("STEP 4: WEB SEARCH\n")
                    f.write("=" * 60 + "\n")
                    web_search_start_time = time.time()

                    search_results = await self.search_agent.search_restaurants(analyzed_query)
                    web_search_time = time.time() - web_search_start_time
                    pipeline_data['search_results'] = search_results

                    f.write(f"Web Search Time: {web_search_time:.2f}s\n")
                    f.write(f"URLs Found: {len(search_results)}\n\n")

                    if search_results:
                        f.write("WEB SEARCH RESULTS:\n")
                        for i, result in enumerate(search_results[:10], 1):
                            f.write(f"{i}. {result.get('title', 'No Title')}\n")
                            f.write(f"   URL: {result.get('url', 'No URL')}\n")
                            f.write(f"   Quality Score: {result.get('quality_score', 'N/A')}\n")
                            f.write(f"   Description: {result.get('description', 'No description')[:150]}...\n\n")
                    else:
                        f.write("❌ No web search results found\n\n")
                        return filepath
                else:
                    f.write("STEP 4: WEB SEARCH SKIPPED (Database content sufficient)\n")
                    f.write("=" * 60 + "\n\n")
                    search_results = []

                # STEP 5: ENHANCED SMART SCRAPING (if web search was performed)
                if search_results:
                    f.write("STEP 5: ENHANCED SMART SCRAPING WITH DOMAIN INTELLIGENCE\n")
                    f.write("=" * 80 + "\n")
                    scraping_start_time = time.time()

                    # Clear scraper stats for clean test
                    if hasattr(self.scraper, 'stats'):
                        self.scraper.stats = {
                            "total_processed": 0,
                            "strategy_breakdown": {"specialized": 0, "simple_http": 0, "enhanced_http": 0, "firecrawl": 0},
                            "ai_analysis_calls": 0,
                            "domain_cache_hits": 0,
                            "new_domains_learned": 0,
                            "total_cost_estimate": 0.0,
                            "cost_saved_vs_all_firecrawl": 0.0,
                            "sectioning_calls": 0,
                            "firecrawl_attempts": 0,
                            "firecrawl_success_rate": 0.0
                        }

                    enriched_results = await self.scraper.scrape_search_results(search_results)
                    scraping_time = time.time() - scraping_start_time
                    pipeline_data['enriched_results'] = enriched_results

                    f.write(f"Enhanced Scraping Time: {scraping_time:.2f}s\n")
                    f.write(f"URLs Processed: {len(enriched_results)}\n\n")

                    # Show enhanced scraper statistics
                    if hasattr(self.scraper, 'stats'):
                        stats = self.scraper.stats
                        f.write("ENHANCED SCRAPING STATISTICS:\n")
                        f.write("-" * 50 + "\n")
                        f.write(f"Total URLs Processed: {stats.get('total_processed', 0)}\n")
                        f.write(f"AI Analysis Calls: {stats.get('ai_analysis_calls', 0)}\n")
                        f.write(f"Domain Cache Hits: {stats.get('domain_cache_hits', 0)}\n")
                        f.write(f"New Domains Learned: {stats.get('new_domains_learned', 0)}\n")
                        f.write(f"Content Sectioning Calls: {stats.get('sectioning_calls', 0)}\n")
                        f.write(f"Firecrawl Attempts: {stats.get('firecrawl_attempts', 0)}\n\n")

                        # Strategy breakdown with costs
                        f.write("STRATEGY BREAKDOWN & COST ANALYSIS:\n")
                        f.write("-" * 50 + "\n")
                        strategy_breakdown = stats.get('strategy_breakdown', {})

                        total_urls = sum(strategy_breakdown.values())
                        estimated_cost = stats.get('total_cost_estimate', 0.0)
                        cost_saved = stats.get('cost_saved_vs_all_firecrawl', 0.0)

                        for strategy, count in strategy_breakdown.items():
                            if count > 0:
                                percentage = (count / total_urls * 100) if total_urls > 0 else 0
                                f.write(f"• {strategy.upper()}: {count} URLs ({percentage:.1f}%)\n")

                        f.write(f"\nCOST OPTIMIZATION:\n")
                        f.write(f"• Estimated Cost: {estimated_cost:.2f} credits\n")
                        f.write(f"• Cost if All Firecrawl: {total_urls * 10:.1f} credits\n")
                        f.write(f"• Cost Savings: {cost_saved:.2f} credits ({(cost_saved/(total_urls * 10) * 100) if total_urls > 0 else 0:.1f}%)\n\n")

                    # Show successful scrapes
                    successful_scrapes = len([r for r in enriched_results if r.get('scraping_success')])
                    f.write(f"Successful Scrapes: {successful_scrapes}/{len(enriched_results)}\n")
                    f.write(f"Success Rate: {round((successful_scrapes/max(len(enriched_results),1))*100, 1)}%\n\n")

                    # DETAILED SCRAPING RESULTS WITH DOMAIN INTELLIGENCE
                    f.write("DETAILED SCRAPING RESULTS WITH DOMAIN INTELLIGENCE:\n")
                    f.write("=" * 80 + "\n")

                    total_content_length = 0
                    sectioned_content_count = 0

                    for i, result in enumerate(enriched_results, 1):
                        f.write(f"SCRAPE #{i}:\n")
                        f.write("-" * 60 + "\n")
                        f.write(f"URL: {result.get('url', 'No URL')}\n")
                        f.write(f"Title: {result.get('title', 'No Title')}\n")
                        f.write(f"Original Quality Score: {result.get('quality_score', 'N/A')}\n")

                        # Show domain intelligence info
                        url = result.get('url', '')
                        if url:
                            from urllib.parse import urlparse
                            domain = urlparse(url).netloc.lower().replace('www.', '')
                            f.write(f"Domain: {domain}\n")

                        scraping_method = result.get('scraping_method', 'Unknown')
                        f.write(f"Scraping Strategy: {scraping_method}\n")

                        # Show if domain intelligence was used
                        if 'domain_cache_used' in result:
                            f.write(f"Domain Cache: {'✅ Used' if result['domain_cache_used'] else '❌ New Domain'}\n")

                        scraped_content = result.get('scraped_content')
                        if scraped_content:
                            content_length = len(scraped_content)
                            total_content_length += content_length

                            f.write(f"Status: ✅ SUCCESS\n")
                            f.write(f"Content Length: {content_length} characters\n")
                            f.write(f"Processing Time: {result.get('scraping_time', 'N/A')}s\n")

                            # Show content sectioning results
                            if result.get('content_sectioned'):
                                sectioned_content_count += 1
                                f.write(f"Content Sectioning: ✅ Applied\n")

                                sectioning_result = result.get('sectioning_result', {})
                                if sectioning_result:
                                    f.write(f"   - Original Length: {sectioning_result.get('original_length', 'N/A')}\n")
                                    f.write(f"   - Optimized Length: {sectioning_result.get('optimized_length', 'N/A')}\n")
                                    f.write(f"   - Sections Found: {sectioning_result.get('sections_identified', [])}\n")
                                    f.write(f"   - Restaurant Density: {sectioning_result.get('restaurants_density', 0):.2f}\n")
                                    f.write(f"   - Restaurants Found: {len(sectioning_result.get('restaurants_found', []))}\n")
                            else:
                                f.write(f"Content Sectioning: ❌ Not applied\n")

                            # Show restaurants found
                            restaurants_found = result.get('restaurants_found', [])
                            if restaurants_found:
                                f.write(f"Restaurants Extracted: {len(restaurants_found)} restaurants\n")

                            f.write(f"\nSCRAPED CONTENT PREVIEW:\n")
                            f.write("~" * 60 + "\n")
                            f.write(scraped_content[:1500])  # Show first 1500 chars
                            if len(scraped_content) > 1500:
                                f.write(f"\n... [Content truncated - Total: {len(scraped_content)} characters]")
                            f.write("\n" + "~" * 60 + "\n\n")

                        else:
                            f.write(f"Status: ❌ FAILED\n")
                            error_msg = result.get('error_message', 'Unknown error')
                            f.write(f"Error: {error_msg}\n\n")

                else:
                    f.write("STEP 5: SMART SCRAPING SKIPPED (No web search performed)\n")
                    f.write("=" * 60 + "\n\n")
                    enriched_results = []

                # STEP 6: EDITOR AGENT (Final compilation)
                f.write("STEP 6: EDITOR AGENT - FINAL RECOMMENDATION COMPILATION\n")
                f.write("=" * 80 + "\n")
                editor_start_time = time.time()

                # Prepare content for editor
                all_content = []

                # Add database results if available
                if database_results:
                    for restaurant in database_results:
                        content_item = {
                            'source': 'database',
                            'name': restaurant.get('name', ''),
                            'location': restaurant.get('location', ''),
                            'cuisine_tags': restaurant.get('cuisine_tags', []),
                            'rating': restaurant.get('rating', ''),
                            'raw_description': restaurant.get('raw_description', ''),
                            'url': restaurant.get('url', '')
                        }
                        all_content.append(content_item)

                # Add scraped content if available
                if enriched_results:
                    for result in enriched_results:
                        if result.get('scraping_success') and result.get('scraped_content'):
                            content_item = {
                                'source': 'web_scraping',
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'content': result.get('scraped_content', ''),
                                'scraping_method': result.get('scraping_method', ''),
                                'restaurants_found': result.get('restaurants_found', [])
                            }
                            all_content.append(content_item)

                if all_content:
                    # Process with editor agent
                    editor_result = await self.editor_agent.process_search_results(
                        analyzed_query, all_content
                    )
                    editor_time = time.time() - editor_start_time
                    pipeline_data['editor_result'] = editor_result

                    f.write(f"Editor Processing Time: {editor_time:.2f}s\n")
                    f.write(f"Content Sources Processed: {len(all_content)}\n\n")

                    f.write("FINAL EDITOR RESULTS:\n")
                    f.write("-" * 50 + "\n")

                    if isinstance(editor_result, dict):
                        recommendations = editor_result.get('recommendations', [])
                        f.write(f"Restaurants Recommended: {len(recommendations)}\n")

                        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
                            f.write(f"\n{i}. {rec.get('name', 'Unknown Restaurant')}\n")
                            f.write(f"   Location: {rec.get('location', 'Unknown')}\n")
                            f.write(f"   Cuisine: {rec.get('cuisine', 'Unknown')}\n")
                            f.write(f"   Rating: {rec.get('rating', 'N/A')}\n")
                            f.write(f"   Why Recommended: {rec.get('why_recommended', 'No reason given')}\n")
                            f.write(f"   Source: {rec.get('source', 'Unknown')}\n")

                        f.write(f"\nFULL EDITOR RESPONSE:\n")
                        f.write("=" * 60 + "\n")
                        if 'final_response' in editor_result:
                            f.write(editor_result['final_response'])
                        else:
                            f.write(str(editor_result))
                        f.write("\n" + "=" * 60 + "\n\n")
                    else:
                        f.write(f"Editor Response:\n{str(editor_result)}\n\n")
                else:
                    f.write("No content available for editor processing\n")
                    editor_time = time.time() - editor_start_time

                # ENHANCED FINAL SUMMARY
                f.write("ENHANCED PIPELINE TEST SUMMARY:\n")
                f.write("=" * 80 + "\n")
                total_time = time.time() - total_start_time

                f.write(f"Total Pipeline Time: {total_time:.2f}s\n")
                f.write(f"1. Query Analysis: {query_time:.2f}s\n")
                f.write(f"2. Database Search: {db_search_time:.2f}s\n")
                if 'eval_time' in locals():
                    f.write(f"3. DB Content Evaluation: {eval_time:.2f}s\n")
                if 'web_search_time' in locals():
                    f.write(f"4. Web Search: {web_search_time:.2f}s\n")
                if 'scraping_time' in locals():
                    f.write(f"5. Enhanced Smart Scraping: {scraping_time:.2f}s\n")
                f.write(f"6. Editor Processing: {editor_time:.2f}s\n\n")

                f.write("PIPELINE RESULTS:\n")
                f.write(f"• Database Restaurants: {len(database_results) if database_results else 0}\n")
                f.write(f"• Web URLs Found: {len(search_results) if search_results else 0}\n")
                f.write(f"• URLs Successfully Scraped: {len([r for r in enriched_results if r.get('scraping_success')]) if enriched_results else 0}\n")
                f.write(f"• Content Sources for Editor: {len(all_content) if 'all_content' in locals() else 0}\n")
                f.write(f"• Final Recommendations: {len(editor_result.get('recommendations', [])) if 'editor_result' in locals() and isinstance(editor_result, dict) else 'N/A'}\n\n")

                # Enhanced scraper summary
                if hasattr(self.scraper, 'stats') and search_results:
                    stats = self.scraper.stats
                    f.write("ENHANCED SCRAPER SUMMARY:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"• Domain Cache Hits: {stats.get('domain_cache_hits', 0)}\n")
                    f.write(f"• New Domains Learned: {stats.get('new_domains_learned', 0)}\n")
                    f.write(f"• AI Classifications: {stats.get('ai_analysis_calls', 0)}\n")
                    f.write(f"• Content Sectioning Applied: {sectioned_content_count if 'sectioned_content_count' in locals() else 0}\n")
                    f.write(f"• Cost Optimization: {stats.get('cost_saved_vs_all_firecrawl', 0):.2f} credits saved\n")
                    f.write(f"• Total Content Extracted: {total_content_length if 'total_content_length' in locals() else 0} characters\n")

                return filepath

        except Exception as e:
            logger.error(f"Error in enhanced full pipeline test: {e}")
            return f"Error: {e}"

    def send_results_to_admin(self, file_path: str, query: str, bot=None):
        """Send enhanced test results to admin via Telegram"""
        if not bot or not self.admin_chat_id:
            logger.warning("Bot or admin chat ID not available for sending results")
            return

        try:
            # Create enhanced summary
            summary = (
                f"🧪 <b>Enhanced Full Pipeline Test Complete</b>\n\n"
                f"📝 <b>Query:</b> <code>{query}</code>\n"
                f"🔄 <b>Pipeline:</b> Database → Enhanced Scraper → Editor\n"
                f"🆕 <b>New Features Tested:</b>\n"
                f"   • Domain intelligence caching\n"
                f"   • Content sectioning\n"
                f"   • Cost optimization\n"
                f"   • Strategy fallback chains\n"
                f"   • Enhanced content extraction\n\n"
                f"📄 <b>Complete results in attached file</b>"
            )

            bot.send_message(
                self.admin_chat_id,
                summary,
                parse_mode='HTML'
            )

            # Send the results file
            with open(file_path, 'rb') as f:
                bot.send_document(
                    self.admin_chat_id,
                    f,
                    caption=f"🧪 Enhanced Pipeline test: {query}"
                )

            logger.info("Successfully sent enhanced pipeline test results to admin")

        except Exception as e:
            logger.error(f"Failed to send enhanced pipeline results to admin: {e}")


# Convenience function for telegram bot integration
def add_scrape_test_command(bot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot
    This function is kept for backward compatibility
    """
    logger.info("Note: enhanced scrape test commands are now handled directly in telegram_bot.py")