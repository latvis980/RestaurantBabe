# agents/langchain_orchestrator.py
# CORRECTED VERSION - With Supabase integration points

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures
from urllib.parse import urlparse

# Updated imports for Supabase integration
from utils.database import (
    cache_search_results, 
    save_domain_intelligence, 
    update_domain_success,
    save_scraped_content,
    save_restaurant_data,
    add_to_search_history
)
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.telegram_formatter import TelegramFormatter

# Create logger
logger = logging.getLogger("restaurant-recommender.orchestrator")

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents with correct file names
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.list_analyzer import ListAnalyzer
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.list_analyzer = ListAnalyzer(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter()

        self.config = config

        # Build the pipeline steps
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the LangChain pipeline with clean step separation"""

        # Step 1: Analyze Query
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"]
            },
            name="analyze_query"
        )

        # Step 2: Search
        self.search = RunnableLambda(
            lambda x: {
                **x,
                "search_results": self.search_agent.search(x["search_queries"])
            },
            name="search"
        )

        # Step 3: Scrape with Supabase Integration
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )

        # Step 4: Analyze Results
        self.analyze_results = RunnableLambda(
            self._analyze_results_step,
            name="analyze_results"
        )

        # Step 5: Edit
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )

        # Step 6: Follow-up Search
        self.follow_up_search = RunnableLambda(
            self._follow_up_step,
            name="follow_up_search"
        )

        # Step 7: Format for Telegram with Database Storage
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )

        # Create the complete chain
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.search,
                self.scrape,
                self.analyze_results,
                self.edit,
                self.follow_up_search,
                self.format_output,
            ],
            last=RunnableLambda(lambda x: x),
            name="restaurant_recommendation_chain"
        )

    def _scrape_step(self, x):
        """Handle async scraping with Supabase integration"""
        search_results = x.get("search_results", [])
        logger.info(f"🔍 Scraping {len(search_results)} search results with Supabase integration")

        def run_scraping():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.scraper.scrape_search_results(search_results)
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            enriched_results = pool.submit(run_scraping).result()

        # ============ SUPABASE INTEGRATION POINT 1: DOMAIN INTELLIGENCE ============
        self._save_domain_intelligence_from_scraping_results(enriched_results)

        # ============ SUPABASE INTEGRATION POINT 2: SAVE SCRAPED CONTENT FOR RAG ============
        self._save_scraped_content_for_rag(enriched_results)

        # Log usage after scraping
        self._log_firecrawl_usage()

        logger.info(f"✅ Scraping completed with {len(enriched_results)} enriched results, data saved to Supabase")
        return {**x, "enriched_results": enriched_results}

    def _save_domain_intelligence_from_scraping_results(self, enriched_results):
        """Extract and save domain intelligence from scraping results"""
        try:
            domain_stats = {}

            for result in enriched_results:
                url = result.get("url", "")
                if not url:
                    continue

                try:
                    domain = urlparse(url).netloc.lower()
                    if domain.startswith('www.'):
                        domain = domain[4:]

                    # Initialize domain stats if not seen before
                    if domain not in domain_stats:
                        domain_stats[domain] = {
                            'success_count': 0,
                            'failure_count': 0,
                            'total_restaurants_found': 0,
                            'total_content_length': 0,
                            'scraping_methods': set(),
                            'urls_processed': 0
                        }

                    stats = domain_stats[domain]
                    stats['urls_processed'] += 1

                    # Determine if scraping was successful
                    scraping_success = result.get("scraping_success", False)
                    scraped_content = result.get("scraped_content", "")
                    scraping_method = result.get("scraping_method", "unknown")

                    if scraping_success and scraped_content:
                        stats['success_count'] += 1
                        stats['total_content_length'] += len(scraped_content)
                        stats['scraping_methods'].add(scraping_method)

                        # Estimate restaurant count from content
                        content_lower = scraped_content.lower()
                        restaurant_indicators = content_lower.count('restaurant') + content_lower.count('cafe') + content_lower.count('bar')
                        stats['total_restaurants_found'] += min(restaurant_indicators, 20)  # Cap at reasonable number
                    else:
                        stats['failure_count'] += 1

                except Exception as e:
                    logger.warning(f"Error processing domain intelligence for {url}: {e}")
                    continue

            # Save domain intelligence to Supabase
            for domain, stats in domain_stats.items():
                try:
                    # Calculate metrics
                    total_attempts = stats['success_count'] + stats['failure_count']
                    confidence = stats['success_count'] / total_attempts if total_attempts > 0 else 0.5
                    avg_content_length = stats['total_content_length'] / max(stats['success_count'], 1)

                    # Determine complexity based on success rate and scraping methods
                    if confidence > 0.8:
                        complexity = 'simple_html'
                    elif confidence > 0.5:
                        complexity = 'moderate_js'
                    else:
                        complexity = 'heavy_js'

                    # Determine best scraper type
                    scraper_type = 'enhanced_http'  # Default
                    if 'specialized' in stats['scraping_methods']:
                        scraper_type = 'specialized'
                    elif 'firecrawl' in stats['scraping_methods']:
                        scraper_type = 'firecrawl'
                    elif 'simple_http' in stats['scraping_methods']:
                        scraper_type = 'simple_http'

                    intelligence_data = {
                        'complexity': complexity,
                        'scraper_type': scraper_type,
                        'cost': 0.1 if scraper_type == 'specialized' else (0.5 if scraper_type == 'firecrawl' else 0.05),
                        'confidence': confidence,
                        'reasoning': f'Based on {total_attempts} attempts: {stats["success_count"]} successful, avg content length: {avg_content_length:.0f}',
                        'success_count': stats['success_count'],
                        'failure_count': stats['failure_count'],
                        'total_restaurants_found': stats['total_restaurants_found'],
                        'avg_content_length': int(avg_content_length),
                        'was_successful': stats['success_count'] > 0,
                        'metadata': {
                            'scraping_methods_used': list(stats['scraping_methods']),
                            'urls_processed': stats['urls_processed'],
                            'last_analysis': time.time()
                        }
                    }

                    # Save to Supabase
                    success = save_domain_intelligence(domain, intelligence_data)
                    if success:
                        logger.info(f"💾 Saved domain intelligence for {domain}: {confidence:.2f} confidence, {stats['success_count']}/{total_attempts} success rate")
                    else:
                        logger.warning(f"❌ Failed to save domain intelligence for {domain}")

                except Exception as e:
                    logger.error(f"Error saving domain intelligence for {domain}: {e}")

            logger.info(f"📊 Domain intelligence saved for {len(domain_stats)} domains")

        except Exception as e:
            logger.error(f"Error in domain intelligence processing: {e}")

    def _save_scraped_content_for_rag(self, enriched_results):
        """Save scraped content to Supabase for RAG"""
        try:
            saved_count = 0

            for result in enriched_results:
                url = result.get("url", "")
                scraped_content = result.get("scraped_content", "")
                scraping_success = result.get("scraping_success", False)

                if scraping_success and scraped_content and len(scraped_content) > 100:
                    try:
                        # Extract restaurant mentions from content (basic approach)
                        content_lower = scraped_content.lower()
                        restaurant_mentions = []

                        # Simple restaurant name extraction (you can enhance this later)
                        import re
                        restaurant_patterns = [
                            r'\b([A-Z][a-z]+ (?:Restaurant|Café|Bar|Bistro|Trattoria|Osteria))\b',
                            r'\b(Restaurant [A-Z][a-z]+)\b',
                            r'\b([A-Z][a-z]+ & [A-Z][a-z]+)\b'
                        ]

                        for pattern in restaurant_patterns:
                            matches = re.findall(pattern, scraped_content)
                            restaurant_mentions.extend(matches[:5])  # Limit to 5 per pattern

                        # Save to Supabase RAG system
                        success = save_scraped_content(url, scraped_content, restaurant_mentions)
                        if success:
                            saved_count += 1
                            logger.debug(f"💾 Saved content for RAG: {url[:60]}... ({len(scraped_content)} chars)")

                    except Exception as e:
                        logger.warning(f"Error saving scraped content for {url}: {e}")

            logger.info(f"📚 Saved {saved_count} articles to RAG system")

        except Exception as e:
            logger.error(f"Error in RAG content saving: {e}")

    def _analyze_results_step(self, x):
        """Handle async result analysis"""
        def run_analysis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._analyze_results_async(x))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_analysis).result()

    async def _analyze_results_async(self, x):
        """Async analysis of search results"""
        try:
            dump_chain_state("pre_analyze_results", {
                "enriched_results_count": len(x.get("enriched_results", [])),
                "keywords": x.get("keywords_for_analysis", []),
                "destination": x.get("destination", "Unknown")
            })

            recommendations = await self.list_analyzer.analyze(
                search_results=x["enriched_results"],
                keywords_for_analysis=x.get("keywords_for_analysis", []),
                primary_search_parameters=x.get("primary_search_parameters", []),
                secondary_filter_parameters=x.get("secondary_filter_parameters", []),
                destination=x.get("destination")
            )

            # Standardize the recommendations structure
            standardized = self._standardize_recommendations(recommendations)

            return {**x, "recommendations": standardized}

        except Exception as e:
            logger.error(f"Error in analyze_results: {e}")
            dump_chain_state("analyze_results_error", x, error=e)
            return {**x, "recommendations": {"main_list": []}}

    def _standardize_recommendations(self, recommendations):
        """Convert recommendations to standard format"""
        if not isinstance(recommendations, dict):
            return {"main_list": []}

        all_restaurants = []

        # Get restaurants from main_list
        main_list = recommendations.get("main_list", [])
        if isinstance(main_list, list):
            all_restaurants.extend(main_list)

        # Get restaurants from hidden_gems and add to main list
        hidden_gems = recommendations.get("hidden_gems", [])
        if isinstance(hidden_gems, list):
            all_restaurants.extend(hidden_gems)

        # Handle legacy format
        if "recommended" in recommendations and not all_restaurants:
            recommended = recommendations.get("recommended", [])
            if isinstance(recommended, list):
                all_restaurants.extend(recommended)

        return {"main_list": all_restaurants}

    def _edit_step(self, x):
        """Edit step - processes scraped_results and returns edited_results"""
        try:
            dump_chain_state("pre_edit", {
                "available_keys": list(x.keys()),
                "enriched_results_count": len(x.get("enriched_results", [])),
                "query": x.get("query", "")
            })

            # Get the scraped results from previous step
            scraped_results = x.get("enriched_results", [])  # enriched_results = scraped_results
            original_query = x.get("query", "")
            destination = x.get("destination", "Unknown")

            if not scraped_results:
                logger.warning("No scraped results available for editing")
                return {
                    **x,
                    "edited_results": {"main_list": []},
                    "follow_up_queries": []
                }

            # Call the editor with scraped results
            edit_output = self.editor_agent.edit(
                scraped_results=scraped_results,
                original_query=original_query,
                destination=destination
            )

            dump_chain_state("post_edit", {
                "edit_output_keys": list(edit_output.keys() if edit_output else {}),
                "main_list_count": len(edit_output.get("edited_results", {}).get("main_list", []))
            })

            return {
                **x, 
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"Error in edit step: {e}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)

            # Return fallback response
            return {
                **x,
                "edited_results": {"main_list": []},
                "follow_up_queries": []
            }

    def _follow_up_step(self, x):
        """Follow-up search step - processes edited_results and returns enhanced_results"""
        try:
            dump_chain_state("pre_follow_up", {
                "edited_results_keys": list(x.get("edited_results", {}).keys()),
                "destination": x.get("destination", "Unknown")
            })

            edited_results = x.get("edited_results", {})
            follow_up_queries = x.get("follow_up_queries", [])

            if not edited_results.get("main_list"):
                logger.warning("No restaurants available for follow-up search")
                return {**x, "enhanced_results": {"main_list": []}}

            # Call follow-up search with edited results
            followup_output = self.follow_up_search_agent.perform_follow_up_searches(
                edited_results=edited_results,
                follow_up_queries=follow_up_queries,
                destination=x.get("destination", "Unknown"),
                secondary_filter_parameters=x.get("secondary_filter_parameters")
            )

            enhanced_results = followup_output.get("enhanced_results", {"main_list": []})

            dump_chain_state("post_follow_up", {
                "enhanced_count": len(enhanced_results.get("main_list", [])),
                "destination": x.get("destination", "Unknown")
            })

            return {**x, "enhanced_results": enhanced_results}

        except Exception as e:
            logger.error(f"Error in follow-up step: {e}")
            dump_chain_state("follow_up_error", x, error=e)
            return {**x, "enhanced_results": {"main_list": []}}

    def _format_step(self, x):
        """Format step - converts enhanced_results to telegram_formatted_text with restaurant data saving"""
        try:
            dump_chain_state("pre_format", {
                "enhanced_results_keys": list(x.get("enhanced_results", {}).keys()),
                "destination": x.get("destination", "Unknown")
            })

            enhanced_results = x.get("enhanced_results", {})
            main_list = enhanced_results.get("main_list", [])

            if not main_list:
                logger.warning("No restaurants to format for Telegram")
                return {
                    **x,
                    "telegram_formatted_text": "Sorry, no restaurant recommendations found for your query."
                }

            # ============ SUPABASE INTEGRATION POINT 3: SAVE RESTAURANT DATA ============
            self._save_restaurant_data_from_results(main_list, x.get("destination", "Unknown"))

            # Format for Telegram using the formatter
            telegram_text = self.telegram_formatter.format_recommendations(
                enhanced_results  # Pass the entire enhanced_results dict
            )

            dump_chain_state("post_format", {
                "telegram_text_length": len(telegram_text),
                "restaurant_count": len(main_list)
            })

            return {
                **x,
                "telegram_formatted_text": telegram_text,
                "final_results": enhanced_results  # Keep the enhanced results for any further processing
            }

        except Exception as e:
            logger.error(f"Error in format step: {e}")
            dump_chain_state("format_error", x, error=e)
            return {
                **x,
                "telegram_formatted_text": "Sorry, there was an error formatting the restaurant recommendations."
            }

    def _save_restaurant_data_from_results(self, main_list, destination):
        """Extract and save structured restaurant data to Supabase"""
        try:
            saved_count = 0

            for restaurant in main_list:
                try:
                    # Extract restaurant information
                    name = restaurant.get("name", "").strip()
                    if not name:
                        continue

                    # Build restaurant data object
                    restaurant_data = {
                        'name': name,
                        'address': restaurant.get("address", ""),
                        'neighborhood': restaurant.get("neighborhood", ""),
                        'city': destination if destination != "Unknown" else "",
                        'country': "",  # Could be enhanced later
                        'cuisine_type': restaurant.get("cuisine", ""),
                        'phone': restaurant.get("phone", ""),
                        'website': restaurant.get("website", ""),
                        'credibility_score': self._calculate_credibility_score(restaurant),
                        'is_professional': True,  # Coming from professional sources
                        'metadata': {
                            'price_range': restaurant.get("price_range", ""),
                            'description': restaurant.get("description", ""),
                            'highlights': restaurant.get("highlights", []),
                            'source_urls': restaurant.get("source_urls", []),
                            'extraction_timestamp': time.time()
                        }
                    }

                    # Save to Supabase
                    restaurant_id = save_restaurant_data(restaurant_data)
                    if restaurant_id:
                        saved_count += 1
                        logger.debug(f"🏪 Saved restaurant: {name}")

                except Exception as e:
                    logger.warning(f"Error saving restaurant data: {e}")

            logger.info(f"🏪 Saved {saved_count} restaurants to database")

        except Exception as e:
            logger.error(f"Error in restaurant data saving: {e}")

    def _calculate_credibility_score(self, restaurant):
        """Calculate credibility score for a restaurant based on available data"""
        try:
            score = 0.5  # Base score

            # Boost for multiple sources
            source_urls = restaurant.get("source_urls", [])
            if len(source_urls) > 1:
                score += 0.2

            # Boost for complete information
            if restaurant.get("address"):
                score += 0.1
            if restaurant.get("phone"):
                score += 0.1
            if restaurant.get("website"):
                score += 0.1

            # Boost for detailed description
            description = restaurant.get("description", "")
            if len(description) > 100:
                score += 0.1

            return min(1.0, score)  # Cap at 1.0

        except Exception:
            return 0.5  # Default score

    def _log_firecrawl_usage(self):
        """Log Firecrawl usage statistics"""
        try:
            stats = self.scraper.get_stats()
            logger.info("=" * 50)
            logger.info("FIRECRAWL USAGE REPORT")
            logger.info("=" * 50)
            logger.info(f"URLs scraped: {stats.get('total_scraped', 0)}")
            logger.info(f"Successful extractions: {stats.get('successful_extractions', 0)}")
            logger.info(f"Credits used: {stats.get('credits_used', 0)}")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error logging Firecrawl usage: {e}")

    @log_function_call
    def process_query(self, user_query: str, user_preferences: dict = None, user_id: str = None) -> dict:
        """
        Process a restaurant query through the complete pipeline with Supabase integration.

        Args:
            user_query: The user's restaurant request
            user_preferences: Optional user preferences dict
            user_id: Optional user ID for tracking search history

        Returns:
            Dict with telegram_formatted_text and other results
        """

        # Generate trace ID for debugging
        trace_id = f"query_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                dump_chain_state("process_query_start", {"query": user_query, "trace_id": trace_id})

                # Prepare input data
                input_data = {
                    "query": user_query,
                    "user_preferences": user_preferences or {}
                }

                # Execute the chain
                result = self.chain.invoke(input_data)

                # Log completion
                dump_chain_state("process_query_complete", {
                    "result_keys": list(result.keys()),
                    "has_enhanced_results": "enhanced_results" in result,
                    "has_telegram_text": "telegram_formatted_text" in result,
                    "destination": result.get("destination", "Unknown")
                })

                # Final usage summary
                self._log_firecrawl_usage()

                # ============ SUPABASE INTEGRATION POINT 4: CACHE FINAL RESULTS ============
                enhanced_results = result.get("enhanced_results", {})
                main_list = enhanced_results.get("main_list", [])

                # Cache the complete search results
                cache_data = {
                    "query": user_query,
                    "destination": result.get("destination", "Unknown"),
                    "results": enhanced_results,
                    "restaurant_count": len(main_list),
                    "trace_id": trace_id,
                    "timestamp": time.time(),
                    "firecrawl_stats": self.scraper.get_stats()
                }

                cache_search_results(user_query, cache_data)

                # ============ SUPABASE INTEGRATION POINT 5: USER SEARCH HISTORY ============
                if user_id:
                    add_to_search_history(user_id, user_query, len(main_list))

                # Extract results with correct key names
                telegram_text = result.get("telegram_formatted_text", 
                                         "Sorry, no recommendations found.")

                logger.info(f"✅ Final result - {len(main_list)} restaurants for {result.get('destination', 'Unknown')}, all data saved to Supabase")

                # Return with correct key names that telegram_bot.py expects
                return {
                    "telegram_formatted_text": telegram_text,
                    "enhanced_results": enhanced_results,
                    "main_list": main_list,
                    "destination": result.get("destination"),
                    "firecrawl_stats": self.scraper.get_stats()
                }

            except Exception as e:
                logger.error(f"Error in chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query}, error=e)
                self._log_firecrawl_usage()

                return {
                    "main_list": [],
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",
                    "firecrawl_stats": self.scraper.get_stats()
                }