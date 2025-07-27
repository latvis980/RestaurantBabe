# agents/langchain_orchestrator.py - CLEANED VERSION WITHOUT RAG
# Removed all Supabase update functionality AND RAG search

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures
import os
import requests
from urllib.parse import urlparse
from typing import Dict, Any, List
from datetime import datetime

# Updated imports - removed Supabase update agent and RAG
from utils.database import (
    cache_search_results, 
    save_domain_intelligence, 
    update_domain_success,
    add_to_search_history,
    get_restaurants_by_city  # Direct database access instead of RAG
)
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.telegram_formatter import TelegramFormatter

# Create logger
logger = logging.getLogger("restaurant-recommender.orchestrator")

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents with correct file names (no RAG agent)
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.list_analyzer import ListAnalyzer
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents (no RAG agent)
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
        """Build the simplified LangChain pipeline without Supabase updates or RAG"""
        logger.info("🚀 BUILDING SIMPLIFIED PIPELINE - No Supabase Updates, No RAG")

        # Step 1: Analyze Query (with country detection)
        self.analyze_query = RunnableLambda(
            self._analyze_query_with_country_detection,
            name="analyze_query_with_country"
        )
        logger.info(f"✅ Step 1 defined: {self.analyze_query}")

        # Step 2: Check Existing Database Coverage (Direct database query)  
        self.check_database = RunnableLambda(
            self._check_database_coverage_direct,
            name="check_database_coverage_direct"
        )
        logger.info(f"✅ Step 2 defined: {self.check_database}")

        # Step 3: Search (only if no database content)
        self.search = RunnableLambda(
            self._search_step,
            name="search"
        )
        logger.info(f"✅ Step 3 defined: {self.search}")

        # Step 4: Scrape Content and Save to File
        self.scrape = RunnableLambda(
            self._scrape_and_save_step,
            name="scrape_and_save"
        )
        logger.info(f"✅ Step 4 defined: {self.scrape}")

        # Step 5: Analyze Results (from database OR scraped content)
        self.analyze_results = RunnableLambda(
            self._analyze_results_step,
            name="analyze_results"
        )
        logger.info(f"✅ Step 5 defined: {self.analyze_results}")

        # Step 6: Edit Recommendations
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )
        logger.info(f"✅ Step 6 defined: {self.edit}")

        # Step 7: Follow-up Search (lightweight version)
        self.follow_up_search = RunnableLambda(
            self._follow_up_step,
            name="follow_up_search"
        )
        logger.info(f"✅ Step 7 defined: {self.follow_up_search}")

        # Step 8: Format for Telegram
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )
        logger.info(f"✅ Step 8 defined: {self.format_output}")

        # Build the complete pipeline
        self.pipeline = (
            self.analyze_query |
            self.check_database |
            self.search |
            self.scrape |
            self.analyze_results |
            self.edit |
            self.follow_up_search |
            self.format_output
        )

        # Also create the chain property for backward compatibility
        self.chain = self.pipeline

        logger.info("✅ PIPELINE BUILT SUCCESSFULLY")

    def _scrape_and_save_step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape content and save to file, then send to Supabase Manager service
        """
        try:
            logger.info("🔍 ENTERING SCRAPE AND SAVE STEP")

            search_results = x.get("search_results", [])
            city = x.get("city", "Unknown")
            country = x.get("country", "Unknown")

            if not search_results:
                logger.info("📄 No search results to scrape (database-only mode)")
                return {**x, "enriched_results": [], "scraped_content_saved": False}

            logger.info(f"🌐 Scraping {len(search_results)} URLs")

            # Scrape all URLs
            enriched_results = []
            all_scraped_content = ""
            sources = []

            for result in search_results:
                try:
                    url = result.get('url', '')
                    if not url:
                        continue

                    logger.info(f"🌐 Scraping: {url}")
                    scraped_content = self.scraper.scrape_url(url)

                    if scraped_content and len(scraped_content.strip()) > 100:
                        enriched_result = {
                            **result,
                            "scraped_content": scraped_content,
                            "content_length": len(scraped_content)
                        }
                        enriched_results.append(enriched_result)

                        # Combine content for saving
                        all_scraped_content += f"\n\n--- FROM {url} ---\n\n{scraped_content}"
                        sources.append(url)

                        logger.info(f"✅ Scraped {len(scraped_content)} chars from {url}")
                    else:
                        logger.warning(f"⚠️ No substantial content from {url}")

                except Exception as e:
                    logger.error(f"❌ Error scraping {result.get('url', 'unknown')}: {e}")
                    continue

            # Save scraped content to file and send to Supabase Manager
            content_saved = False
            if all_scraped_content.strip():
                try:
                    # Save to local file first (for backup/debugging)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"scraped_{city.replace(' ', '_')}_{timestamp}.txt"

                    os.makedirs("scraped_content", exist_ok=True)
                    file_path = os.path.join("scraped_content", filename)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(all_scraped_content)

                    logger.info(f"💾 Saved scraped content to: {file_path}")

                    # Send to Supabase Manager service (async, don't wait for response)
                    self._send_to_supabase_manager_async(all_scraped_content, {
                        'city': city,
                        'country': country,
                        'sources': sources,
                        'query': x.get('original_query', ''),
                        'scraped_at': datetime.now().isoformat()
                    })

                    content_saved = True

                except Exception as e:
                    logger.error(f"❌ Error saving scraped content: {e}")

            return {
                **x,
                "enriched_results": enriched_results,
                "scraped_content_saved": content_saved,
                "scraped_sources": sources
            }

        except Exception as e:
            logger.error(f"❌ Error in scrape_and_save_step: {e}")
            return {**x, "enriched_results": [], "scraped_content_saved": False}

    def _send_to_supabase_manager_async(self, content: str, metadata: Dict[str, Any]):
        """
        Send scraped content to Supabase Manager service asynchronously
        This doesn't block the main response to the user
        """
        def send_content():
            try:
                supabase_manager_url = getattr(self.config, 'SUPABASE_MANAGER_URL', None)
                if not supabase_manager_url:
                    logger.warning("⚠️ SUPABASE_MANAGER_URL not configured - skipping background update")
                    return

                payload = {
                    'content': content,
                    'metadata': metadata
                }

                response = requests.post(
                    f"{supabase_manager_url}/process_scraped_content",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    logger.info("✅ Successfully sent content to Supabase Manager")
                else:
                    logger.warning(f"⚠️ Supabase Manager returned status {response.status_code}")

            except Exception as e:
                logger.error(f"❌ Error sending to Supabase Manager: {e}")

        # Run in background thread so it doesn't block user response
        import threading
        thread = threading.Thread(target=send_content, daemon=True)
        thread.start()

    def _analyze_query_with_country_detection(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query and detect country"""
        try:
            query = x.get("query", "")
            if not query:
                raise ValueError("No query provided")

            logger.info(f"🔍 Analyzing query: {query}")

            # Use the correct method name - analyze() with query as parameter
            analysis_result = self.query_analyzer.analyze(query)

            if not analysis_result:
                raise ValueError("Query analysis failed")

            logger.info(f"✅ Query analysis complete:")
            logger.info(f"   - Destination: {analysis_result.get('destination')}")
            logger.info(f"   - Primary params: {analysis_result.get('primary_search_parameters')}")

            # Map the analysis result to expected format
            city = analysis_result.get("destination", "Unknown")
            # Extract city from destination if it contains comma
            if "," in city:
                city_parts = city.split(",")
                city = city_parts[0].strip()
                country = city_parts[1].strip() if len(city_parts) > 1 else "Unknown"
            else:
                country = "Unknown"

            return {
                **x,
                "original_query": query,
                "city": city,
                "country": country,
                "cuisine_type": analysis_result.get("primary_search_parameters", []),
                "analysis_result": analysis_result
            }

        except Exception as e:
            logger.error(f"❌ Error in query analysis: {e}")
            return {
                **x,
                "city": "Unknown",
                "country": "Unknown",
                "error": str(e)
            }

    def _check_database_coverage_direct(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Check database for existing restaurant content using direct database query"""
        try:
            city = x.get("city")
            country = x.get("country")

            if not city:
                logger.info("🔍 No city detected - proceeding with web search")
                return {**x, "database_results": [], "has_database_content": False}

            logger.info(f"🗃️ Checking database for {city}, {country}")

            # Direct database query (no RAG needed)
            database_results = get_restaurants_by_city(city)

            # Filter by country if specified
            if country and database_results:
                database_results = [r for r in database_results if r.get('country', '').lower() == country.lower()]

            has_content = len(database_results) > 0

            logger.info(f"🗃️ Database check result: {len(database_results)} restaurants found")

            return {
                **x,
                "database_results": database_results,
                "has_database_content": has_content
            }

        except Exception as e:
            logger.error(f"❌ Error checking database: {e}")
            return {**x, "database_results": [], "has_database_content": False}

    def _search_step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Simple search step (only runs if no database content)"""
        try:
            has_database_content = x.get("has_database_content", False)

            if has_database_content:
                logger.info("📚 Using database content - skipping web search")
                return {**x, "search_results": []}

            logger.info("🌐 No database content found - proceeding with web search")

            # Get search queries from the analysis result
            analysis_result = x.get("analysis_result", {})
            search_queries = analysis_result.get("search_queries", [])

            # If no search queries from analysis, build a simple one
            if not search_queries:
                city = x.get("city", "Unknown")
                country = x.get("country", "Unknown")
                cuisine_type = x.get("cuisine_type")

                # Handle cuisine_type whether it's a string or list
                if isinstance(cuisine_type, list):
                    cuisine_str = " ".join(cuisine_type) if cuisine_type else ""
                else:
                    cuisine_str = cuisine_type or ""

                # Build search query
                search_query = f"best restaurants {city}"
                if country and country != "Unknown":
                    search_query += f" {country}"
                if cuisine_str:
                    search_query += f" {cuisine_str}"

                search_queries = [search_query]

            logger.info(f"🔍 Search queries: {search_queries}")

            # IMPORTANT: Pass the list of queries, not individual strings
            search_results = self.search_agent.search(search_queries)

            logger.info(f"🌐 Search returned {len(search_results)} results")

            return {
                **x,
                "search_queries": search_queries,
                "search_results": search_results
            }

        except Exception as e:
            logger.error(f"❌ Error in search step: {e}")
            return {**x, "search_results": []}

    def _analyze_results_step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from either database or scraped content"""
        try:
            has_database_content = x.get("has_database_content", False)

            if has_database_content:
                # Use database results
                logger.info("📊 Analyzing database results")
                database_results = x.get("database_results", [])

                analysis_result = {
                    "restaurants": database_results,
                    "source": "database",
                    "total_found": len(database_results)
                }

                logger.info(f"📊 Database analysis: {len(database_results)} restaurants")

            else:
                # Use scraped content
                logger.info("📊 Analyzing scraped content")
                enriched_results = x.get("enriched_results", [])

                if not enriched_results:
                    logger.warning("⚠️ No content to analyze")
                    return {**x, "analysis_result": {"restaurants": [], "source": "none", "total_found": 0}}

                combined_content = ""
                for result in enriched_results:
                    content = result.get("scraped_content", "")
                    if content:
                        combined_content += f"\n\n{content}"

                analysis_result = self.list_analyzer.analyze_scraped_content(
                    content=combined_content,
                    city=x.get("city", "Unknown"),
                    country=x.get("country", "Unknown")
                )

                logger.info(f"📊 Content analysis: {analysis_result.get('total_found', 0)} restaurants")

            return {
                **x,
                "analysis_result": analysis_result
            }

        except Exception as e:
            logger.error(f"❌ Error in analysis step: {e}")
            return {**x, "analysis_result": {"restaurants": [], "source": "error", "total_found": 0}}

    def _edit_step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Edit and format recommendations"""
        try:
            logger.info("✏️ ENTERING EDIT STEP")

            analysis_result = x.get("analysis_result", {})
            restaurants = analysis_result.get("restaurants", [])

            if not restaurants:
                logger.warning("⚠️ No restaurants to edit")
                return {**x, "final_recommendations": "No restaurants found for your query."}

            logger.info(f"✏️ Editing recommendations for {len(restaurants)} restaurants")

            # Prepare context for editor
            context = {
                "city": x.get("city", "Unknown"),
                "country": x.get("country", "Unknown"),
                "cuisine_type": x.get("cuisine_type", ""),
                "original_query": x.get("original_query", ""),
                "source": analysis_result.get("source", "unknown")
            }

            edited_response = self.editor_agent.edit_restaurant_recommendations(
                restaurants=restaurants,
                context=context
            )

            logger.info("✅ Editing complete")

            return {
                **x,
                "final_recommendations": edited_response,
                "restaurant_count": len(restaurants)
            }

        except Exception as e:
            logger.error(f"❌ Error in edit step: {e}")
            return {**x, "final_recommendations": f"Error formatting recommendations: {str(e)}"}

    def _follow_up_step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight follow-up search (no geodata updates)"""
        try:
            logger.info("🔍 ENTERING FOLLOW-UP STEP")

            city = x.get("city")
            country = x.get("country")

            if not city:
                logger.info("⚠️ No city for follow-up search")
                return {**x, "follow_up_complete": False}

            # Just do a quick follow-up search for additional context
            follow_up_query = f"best restaurants {city} {country} 2024 guide"

            logger.info(f"🔍 Follow-up search: {follow_up_query}")

            follow_up_results = self.follow_up_search_agent.search_additional_info(
                city=city,
                country=country,
                restaurants=x.get("analysis_result", {}).get("restaurants", [])
            )

            logger.info(f"✅ Follow-up search complete")

            return {
                **x,
                "follow_up_info": follow_up_results,
                "follow_up_complete": True
            }

        except Exception as e:
            logger.error(f"❌ Error in follow-up step: {e}")
            return {**x, "follow_up_complete": False}

    def _format_step(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Format final output for Telegram"""
        try:
            logger.info("📱 ENTERING FORMAT STEP")

            final_recommendations = x.get("final_recommendations", "")
            analysis_result = x.get("analysis_result", {})
            restaurants = analysis_result.get("restaurants", [])

            if not final_recommendations and not restaurants:
                return {**x, "formatted_response": "No restaurant recommendations could be generated."}

            # If we have a formatted text response, use it directly
            if final_recommendations and isinstance(final_recommendations, str):
                logger.info("✅ Using pre-formatted response")
                return {**x, "formatted_response": final_recommendations}

            # Otherwise, format using TelegramFormatter
            if restaurants:
                logger.info(f"📋 Formatting {len(restaurants)} restaurants using TelegramFormatter")

                # Prepare data in the format TelegramFormatter expects
                recommendations_data = {
                    "main_list": restaurants
                }

                # Use the correct method name - format_recommendations, not format_restaurant_recommendations
                formatted_response = self.telegram_formatter.format_recommendations(recommendations_data)

                logger.info("✅ Formatting complete")
                return {**x, "formatted_response": formatted_response}
            else:
                return {**x, "formatted_response": "No restaurant recommendations could be generated."}

        except Exception as e:
            logger.error(f"❌ Error in format step: {e}")
            return {**x, "formatted_response": "Error formatting response."}

    def get_recommendations(self, query: str) -> str:
        """
        Main entry point for getting restaurant recommendations
        Compatible with your existing telegram bot and other calling code
        """
        try:
            logger.info(f"🚀 STARTING RECOMMENDATION PIPELINE")
            logger.info(f"Query: {query}")

            # Run the pipeline
            result = self.pipeline.invoke({"query": query})

            formatted_response = result.get("formatted_response", "")

            if not formatted_response:
                return "I couldn't find restaurant recommendations for your query. Please try rephrasing or specifying a city."

            logger.info("✅ PIPELINE COMPLETE")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

    def process_query(self, user_query: str, user_preferences: Dict[str, Any] = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process restaurant query and return formatted results for telegram bot

        This method maintains compatibility with your existing telegram bot code.
        Returns the expected dictionary structure with telegram_formatted_text.
        """
        try:
            logger.info(f"🚀 PROCESSING QUERY: {user_query}")

            # Run the pipeline
            result = self.pipeline.invoke({"query": user_query})

            # Extract key information
            formatted_response = result.get("formatted_response", "")
            analysis_result = result.get("analysis_result", {})
            restaurants = analysis_result.get("restaurants", [])
            city = result.get("city", "Unknown")
            country = result.get("country", "Unknown")
            source = analysis_result.get("source", "unknown")

            # Build response in expected format for telegram bot
            response = {
                "telegram_formatted_text": formatted_response or "Sorry, no recommendations found.",
                "main_list": restaurants,
                "destination": f"{city}, {country}" if country != "Unknown" else city,
                "enhanced_results": result.get("enriched_results", []),
                "ai_features": {
                    "used_ai_database": source == "database",
                    "restaurants_processed": result.get("scraped_content_saved", False),
                    "search_preferences": user_preferences or {},
                    "country_detected": country
                },
                "search_method": f"simplified_{source}",
                "firecrawl_stats": {
                    "total_scraped": len(result.get("scraped_sources", [])),
                    "successful_extractions": len(result.get("enriched_results", [])),
                    "credits_used": 0  # No longer tracking since we moved to background processing
                }
            }

            logger.info(f"✅ Query processed successfully")
            logger.info(f"   - Destination: {response['destination']}")
            logger.info(f"   - Restaurants found: {len(restaurants)}")
            logger.info(f"   - Source: {source}")

            return response

        except Exception as e:
            logger.error(f"❌ Error in process_query: {e}")

            # Return error response in expected format
            return {
                "telegram_formatted_text": "Sorry, there was an error processing your request.",
                "main_list": [],
                "destination": "Unknown",
                "enhanced_results": [],
                "ai_features": {
                    "used_ai_database": False,
                    "restaurants_processed": False,
                    "search_preferences": user_preferences or {},
                    "country_detected": "Unknown"
                },
                "search_method": "error",
                "firecrawl_stats": {
                    "total_scraped": 0,
                    "successful_extractions": 0,
                    "credits_used": 0
                }
            }

    # Legacy methods for backward compatibility with existing code
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy invoke method for compatibility"""
        return self.pipeline.invoke(input_data)

    def run(self, query: str) -> str:
        """Legacy run method - alias for get_recommendations"""
        return self.get_recommendations(query)