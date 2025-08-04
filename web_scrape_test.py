# web_scrape_test.py - UPDATED FOR SMART SCRAPER WITH SECTIONING & DOMAIN INTEL
# Uses SmartRestaurantScraper with domain intelligence and content sectioning

import asyncio
import time
import tempfile
import os
import threading
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class WebScrapeTest:
    """
    UPDATED: Web-only test that uses SmartRestaurantScraper with new features.

    This test shows:
    - NEW: Domain intelligence caching and learning
    - NEW: Content sectioning for restaurant extraction 
    - NEW: Strategy cost tracking and savings calculations
    - Detailed search results and filtering
    - Complete SMART SCRAPING process with strategy breakdown
    - AI strategy classification for new domains

    Pipeline tested: query_analyzer → search_agent → SmartRestaurantScraper 
    (with domain intelligence + sectioning)
    Command: /test_wscrape
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        # Get agents directly from orchestrator (now using SmartRestaurantScraper)
        self.query_analyzer = orchestrator.query_analyzer
        self.search_agent = orchestrator.search_agent  
        self.scraper = orchestrator.scraper  # This is now SmartRestaurantScraper directly

    async def test_web_scraping_only(self, restaurant_query: str, bot=None) -> str:
        """
        Test ONLY the web search and smart scraping pipeline with new features.
        Shows domain intelligence, sectioning, and cost optimization.
        """
        total_start_time = time.time()

        # Create timestamped filename for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webscrape_test_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        logger.info(f"Starting enhanced smart web scraping test for: {restaurant_query}")
        logger.info(f"Results will be saved to: {filepath}")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("ENHANCED SMART WEB SCRAPING TEST RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Query: {restaurant_query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test Type: Web Search + SmartRestaurantScraper ONLY (Database Bypassed)\n")
                f.write("=" * 80 + "\n\n")

                f.write("ENHANCED SMART SCRAPING PIPELINE:\n")
                f.write("1. Query Analysis → Search Terms Generation\n")
                f.write("2. Web Search → URL Discovery & Filtering\n")
                f.write("3. Smart Scraper → Domain Intelligence Check\n")
                f.write("   ├── 🧠 Check domain cache (bypass AI if known)\n")
                f.write("   ├── 🤖 AI strategy classification (new domains only)\n")
                f.write("   ├── 🆓 Specialized (RSS/Sitemap) - 0 credits\n")
                f.write("   ├── 🟢 Simple HTTP (0.1 credits) + sectioning\n")
                f.write("   ├── 🟡 Enhanced HTTP (0.5 credits) + sectioning\n")
                f.write("   └── 🔴 Firecrawl (10.0 credits) - last resort\n")
                f.write("4. Content Sectioning → Restaurant extraction\n")
                f.write("5. Domain Intelligence → Learning & caching\n\n")

                # STEP 1: Query Analysis
                f.write("STEP 1: QUERY ANALYSIS\n")
                f.write("=" * 60 + "\n")
                query_start_time = time.time()

                analyzed_query = await self.query_analyzer.analyze_query(restaurant_query)
                query_time = time.time() - query_start_time

                f.write(f"Original Query: {restaurant_query}\n")
                f.write(f"Processing Time: {query_time:.2f}s\n")
                f.write(f"Search Terms Generated: {analyzed_query.get('search_terms', [])}\n")
                f.write(f"Location Extracted: {analyzed_query.get('location', 'Not specified')}\n")
                f.write(f"Cuisine Type: {analyzed_query.get('cuisine_type', 'Not specified')}\n")
                f.write(f"Price Range: {analyzed_query.get('price_range', 'Not specified')}\n\n")

                # STEP 2: Web Search
                f.write("STEP 2: WEB SEARCH\n")
                f.write("=" * 60 + "\n")
                search_start_time = time.time()

                search_results = await self.search_agent.search_restaurants(
                    analyzed_query, 
                    force_search=True  # Force web search, bypass database
                )
                search_time = time.time() - search_start_time

                f.write(f"Search Time: {search_time:.2f}s\n")
                f.write(f"URLs Found: {len(search_results)}\n\n")

                if search_results:
                    f.write("SEARCH RESULTS:\n")
                    for i, result in enumerate(search_results[:10], 1):  # Show first 10
                        f.write(f"{i}. {result.get('title', 'No Title')}\n")
                        f.write(f"   URL: {result.get('url', 'No URL')}\n")
                        f.write(f"   Quality Score: {result.get('quality_score', 'N/A')}\n")
                        f.write(f"   Description: {result.get('description', 'No description')[:150]}...\n\n")
                else:
                    f.write("❌ No search results found\n\n")
                    return filepath

                # STEP 3: ENHANCED SMART SCRAPING
                f.write("STEP 3: ENHANCED SMART SCRAPING WITH DOMAIN INTELLIGENCE\n")
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

                f.write(f"Enhanced Processing Time: {scraping_time:.2f}s\n")
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

                # DETAILED SCRAPING RESULTS WITH DOMAIN INTELLIGENCE
                f.write("DETAILED SCRAPING RESULTS WITH DOMAIN INTELLIGENCE:\n")
                f.write("=" * 80 + "\n")

                total_content_length = 0
                successful_scrapes = 0
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
                        successful_scrapes += 1

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

                        f.write(f"\nFULL SCRAPED CONTENT:\n")
                        f.write("~" * 70 + "\n")
                        f.write(scraped_content[:2000])  # Show first 2000 chars
                        if len(scraped_content) > 2000:
                            f.write(f"\n... [Content truncated - Total: {len(scraped_content)} characters]")
                        f.write("\n" + "~" * 70 + "\n\n")

                    else:
                        f.write(f"Status: ❌ FAILED\n")
                        error_msg = result.get('error_message', 'Unknown error')
                        f.write(f"Error: {error_msg}\n\n")

                # ENHANCED FINAL SUMMARY
                f.write("ENHANCED TEST SUMMARY:\n")
                f.write("=" * 60 + "\n")
                total_time = time.time() - total_start_time
                f.write(f"Total Test Time: {total_time:.2f}s\n")
                f.write(f"Query Analysis: {query_time:.2f}s\n")
                f.write(f"Web Search: {search_time:.2f}s\n")
                f.write(f"Smart Scraping: {scraping_time:.2f}s\n\n")

                f.write(f"URLs Found: {len(search_results)}\n")
                f.write(f"URLs Processed: {len(enriched_results)}\n")
                f.write(f"Successful Scrapes: {successful_scrapes}\n")
                f.write(f"Content Sectioning Applied: {sectioned_content_count}\n")
                f.write(f"Success Rate: {round((successful_scrapes/max(len(enriched_results),1))*100, 1)}%\n")
                f.write(f"Total Content Extracted: {total_content_length} characters\n\n")

                # Domain intelligence summary
                if hasattr(self.scraper, 'stats'):
                    stats = self.scraper.stats
                    f.write("DOMAIN INTELLIGENCE SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"• Cache Hits: {stats.get('domain_cache_hits', 0)}\n")
                    f.write(f"• New Domains Learned: {stats.get('new_domains_learned', 0)}\n")
                    f.write(f"• AI Classifications: {stats.get('ai_analysis_calls', 0)}\n")
                    f.write(f"• Cost Optimization: {stats.get('cost_saved_vs_all_firecrawl', 0):.2f} credits saved\n")

                return filepath

        except Exception as e:
            logger.error(f"Error in enhanced smart web scraping test: {e}")
            return f"Error: {e}"

    def send_results_to_admin(self, file_path: str, query: str, bot=None):
        """Send test results to admin via Telegram with enhanced summary"""
        if not bot or not self.admin_chat_id:
            logger.warning("Bot or admin chat ID not available for sending results")
            return

        try:
            # Create enhanced summary
            summary = (
                f"🧠 <b>Enhanced Smart Scraping Test Complete</b>\n\n"
                f"📝 <b>Query:</b> <code>{query}</code>\n"
                f"🆕 <b>Features Tested:</b>\n"
                f"   • Domain intelligence caching\n"
                f"   • Content sectioning\n"
                f"   • Cost optimization\n"
                f"   • Strategy fallback chains\n\n"
                f"📄 <b>Detailed results in attached file</b>"
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
                    caption=f"🧠 Enhanced SmartRestaurantScraper test: {query}"
                )

            logger.info("Successfully sent enhanced smart scraping test results to admin")

        except Exception as e:
            logger.error(f"Failed to send enhanced smart scraping results to admin: {e}")