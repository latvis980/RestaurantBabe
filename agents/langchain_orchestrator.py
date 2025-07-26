# agents/langchain_orchestrator.py - COMPLETE UPDATED FILE WITH AI-POWERED FEATURES
# CORRECTED VERSION - With complete RAG integration and Supabase integration points

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures
from urllib.parse import urlparse
from typing import Dict, Any, List

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

# NEW IMPORT: Supabase Update Agent
from agents.supabase_update_agent import process_all_scraped_restaurants, check_city_coverage, update_city_geodata

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
        from agents.rag_search_agent import RAGSearchAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.list_analyzer = ListAnalyzer(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)
        self.rag_search_agent = RAGSearchAgent(config)

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter()

        self.config = config

        # Build the pipeline steps
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the LangChain pipeline with Supabase Update Agent integration"""
        logger.info("🚀 BUILDING PIPELINE - VERSION 2025-07-26-PIPE-OPERATOR")

        # Step 1: Analyze Query (with country detection)
        self.analyze_query = RunnableLambda(
            self._analyze_query_with_country_detection,
            name="analyze_query_with_country"
        )
        logger.info(f"✅ Step 1 defined: {self.analyze_query}")

        # Step 2: Check Existing Database Coverage (NEW - AI-powered)  
        self.check_database = RunnableLambda(
            self._check_database_coverage_ai,
            name="check_database_coverage_ai"
        )
        logger.info(f"✅ Step 2 defined: {self.check_database}")

        # Step 3: RAG-Enhanced Search
        self.search = RunnableLambda(
            self._rag_enhanced_search_step,
            name="rag_enhanced_search"
        )
        logger.info(f"✅ Step 3 defined: {self.search}")

        # Step 4: Scrape with Supabase Integration
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )
        logger.info(f"✅ Step 4 defined: {self.scrape}")

        # Step 5: Process with Supabase Update Agent (NEW - CRITICAL STEP)
        self.supabase_update = RunnableLambda(
            self._supabase_update_step,
            name="supabase_update"
        )
        logger.info(f"✅ Step 5 defined: {self.supabase_update}")

        # Step 6: Analyze Results
        self.analyze_results = RunnableLambda(
            self._analyze_results_step,
            name="analyze_results"
        )
        logger.info(f"✅ Step 6 defined: {self.analyze_results}")

        # Step 7: Edit (now works with database-stored restaurants)
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )
        logger.info(f"✅ Step 7 defined: {self.edit}")

        # Step 8: Follow-up Search (with geodata updates)
        self.follow_up_search = RunnableLambda(
            self._follow_up_step_with_geodata,
            name="follow_up_search_geodata"
        )
        logger.info(f"✅ Step 8 defined: {self.follow_up_search}")

        # Step 9: Format for Telegram
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )
        logger.info(f"✅ Step 9 defined: {self.format_output}")

        # Verify all methods exist
        method_checks = [
            ("_analyze_query_with_country_detection", hasattr(self, "_analyze_query_with_country_detection")),
            ("_check_database_coverage_ai", hasattr(self, "_check_database_coverage_ai")),
            ("_rag_enhanced_search_step", hasattr(self, "_rag_enhanced_search_step")),
            ("_scrape_step", hasattr(self, "_scrape_step")),
            ("_supabase_update_step", hasattr(self, "_supabase_update_step")),
            ("_analyze_results_step", hasattr(self, "_analyze_results_step")),
            ("_edit_step", hasattr(self, "_edit_step")),
            ("_follow_up_step_with_geodata", hasattr(self, "_follow_up_step_with_geodata")),
            ("_format_step", hasattr(self, "_format_step"))
        ]

        missing_methods = [name for name, exists in method_checks if not exists]
        if missing_methods:
            logger.error(f"❌ Missing methods: {missing_methods}")
            raise ValueError(f"Missing required methods: {missing_methods}")
        else:
            logger.info("✅ All required methods exist")

        # Create the complete chain using MODERN PIPE OPERATOR SYNTAX
        try:
            logger.info("🔧 Creating RunnableSequence using pipe operator (recommended LangChain approach)")

            # Use the modern pipe operator approach (most reliable)
            self.chain = (
                self.analyze_query |
                self.check_database |
                self.search |
                self.scrape |
                self.supabase_update |
                self.analyze_results |
                self.edit |
                self.follow_up_search |
                self.format_output
            )

            logger.info(f"✅ RunnableSequence created successfully using pipe operator!")
            logger.info(f"✅ Chain type: {type(self.chain)}")

        except Exception as e:
            logger.error(f"❌ Failed to create RunnableSequence with pipe operator: {e}")

            # Fallback: try the list constructor
            try:
                logger.info("🔄 Trying fallback: RunnableSequence with list constructor")
                all_steps = [
                    self.analyze_query,
                    self.check_database,
                    self.search,
                    self.scrape,
                    self.supabase_update,
                    self.analyze_results,
                    self.edit,
                    self.follow_up_search,
                    self.format_output,
                ]

                from langchain_core.runnables import RunnableSequence
                self.chain = RunnableSequence(*all_steps)

                logger.info(f"✅ Fallback successful! Chain type: {type(self.chain)}")

            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                raise

    # NEW STEP 1: Enhanced Query Analysis with AI Country Detection
    def _analyze_query_with_country_detection(self, x):
        """Enhanced query analysis that includes AI-powered country detection"""
        try:
            query = x["query"]
            logger.info(f"🔍 Analyzing query with AI country detection: {query}")

            # Get basic analysis
            analysis = self.query_analyzer.analyze(query)

            # Add AI-powered country detection
            city = analysis.get("destination", "Unknown")
            country = self._detect_country_from_city(city)

            logger.info(f"📍 Detected location: {city}, {country}")

            return {
                **analysis,
                "query": query,
                "city": city,
                "country": country
            }

        except Exception as e:
            logger.error(f"❌ Error in query analysis: {e}")
            return {
                "query": x["query"],
                "destination": "Unknown",
                "city": "Unknown", 
                "country": "Unknown",
                "search_queries": [x["query"]],
                "primary_search_parameters": [],
                "secondary_filter_parameters": [],
                "keywords_for_analysis": []
            }

    # NEW STEP 2: AI-Powered Database Coverage Check
    def _check_database_coverage_ai(self, x):
        """Check database coverage using AI-powered semantic matching"""
        try:
            city = x.get("city", "Unknown")
            primary_params = x.get("primary_search_parameters", [])
            secondary_params = x.get("secondary_filter_parameters", [])

            logger.info(f"🔍 AI-powered database coverage check for {city}")
            logger.info(f"📋 Search parameters: {primary_params + secondary_params}")

            # Use AI to extract cuisine types and atmosphere preferences
            search_preferences = self._extract_search_preferences_ai(primary_params + secondary_params)

            logger.info(f"🤖 AI extracted preferences: {search_preferences}")

            # Check existing coverage with AI-enhanced matching
            coverage = self._check_database_with_ai_matching(city, search_preferences)

            # Log coverage results
            logger.info(f"📊 Database coverage: {coverage['total_restaurants']} total, {coverage['preference_matches']} preference matches")

            return {
                **x,
                "database_coverage": coverage,
                "search_preferences": search_preferences,
                "has_sufficient_data": coverage["sufficient_coverage"],
                "should_supplement_search": not coverage["sufficient_coverage"] or coverage["total_restaurants"] < 8
            }

        except Exception as e:
            logger.error(f"❌ Error in AI database coverage check: {e}")
            return {
                **x,
                "database_coverage": {"has_data": False, "total_restaurants": 0},
                "search_preferences": {},
                "has_sufficient_data": False,
                "should_supplement_search": True
            }

    # MODIFIED STEP 3: Conditional Search Based on AI Database Coverage
    def _rag_enhanced_search_step(self, x):
        """Enhanced search that considers AI-analyzed database coverage"""
        try:
            should_search = x.get("should_supplement_search", True)

            if not should_search:
                logger.info("🎯 Sufficient AI-matched database coverage found, skipping web search")
                return {**x, "search_results": [], "used_database_only": True}

            logger.info("🌐 Performing web search to supplement AI-analyzed database")

            # Your existing search logic
            search_queries = x.get("search_queries", [])
            destination = x.get("destination", "Unknown")

            if not search_queries:
                logger.warning("No search queries available")
                return {**x, "search_results": []}

            # Perform search
            search_results = self.search_agent.search(search_queries)

            # Add RAG enhancement if available
            try:
                rag_info = self.rag_search_agent.enhance_search(
                    search_queries, search_results, destination
                )
                logger.info(f"🧠 RAG enhancement added {len(rag_info.get('additional_sources', []))} sources")
            except Exception as e:
                logger.warning(f"RAG enhancement failed: {e}")
                rag_info = {}

            logger.info(f"✅ Search completed: {len(search_results)} results")

            return {
                **x, 
                "search_results": search_results,
                "rag_enhancement": rag_info,
                "used_database_only": False
            }

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return {**x, "search_results": [], "used_database_only": False}

    # EXISTING STEP 4: Scraping (keep your existing implementation)
    def _scrape_step(self, x):
        """Handle async scraping with Supabase integration"""
        search_results = x.get("search_results", [])

        if not search_results:
            logger.info("🔍 No search results to scrape (using AI-matched database only)")
            return {**x, "enriched_results": []}

        logger.info(f"🔍 Scraping {len(search_results)} search results")

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

        # Domain intelligence and content saving
        self._save_domain_intelligence_from_scraping_results(enriched_results)
        self._save_scraped_content_for_rag(enriched_results)

        logger.info(f"✅ Scraping completed with {len(enriched_results)} enriched results")
        return {**x, "enriched_results": enriched_results}


    # Replace your existing _supabase_update_step method with this fixed version

    def _supabase_update_step(self, x):
        """
        Process ALL scraped content and update Supabase restaurants database with AI
        This step happens BEFORE the editor agent, saving ALL restaurants found
        """
        try:
            enriched_results = x.get("enriched_results", [])
            city = x.get("city", "Unknown")
            country = x.get("country", "Unknown")

            if not enriched_results:
                logger.info("🔍 No scraped content to process (database-only mode)")
                return {**x, "restaurants_processed": 0, "all_restaurants_saved": []}

            logger.info(f"🤖 Processing ALL scraped content with AI-powered Supabase Update Agent")
            logger.info(f"📄 Processing {len(enriched_results)} scraped articles for {city}, {country}")

            # Combine all scraped content
            combined_content = ""
            sources = []

            for result in enriched_results:
                content = result.get("scraped_content", result.get("content", ""))
                url = result.get("url", "")

                if content and len(content.strip()) > 100:
                    combined_content += f"\n\n--- FROM {url} ---\n\n{content}"
                    sources.append(url)

            if not combined_content.strip():
                logger.warning("⚠️ No substantial content found in scraped results")
                return {**x, "restaurants_processed": 0, "all_restaurants_saved": []}

            logger.info(f"📝 Combined content: {len(combined_content)} chars from {len(sources)} sources")

            # Process ALL scraped content and save to database
            processing_result = process_all_scraped_restaurants(
                scraped_content=combined_content,
                sources=sources,
                city=city,
                country=country,
                config=self.config
            )

            restaurants_processed = processing_result.get('restaurants_processed', [])
            save_stats = processing_result.get('save_statistics', {})
            total_count = processing_result.get('total_restaurants', 0)

            logger.info(f"✅ AI-powered Supabase Update Agent processed {total_count} restaurants")
            logger.info(f"   - New restaurants added: {save_stats.get('new_restaurants', 0)}")
            logger.info(f"   - Updated restaurants: {save_stats.get('updated_restaurants', 0)}")
            logger.info(f"   - Failed saves: {save_stats.get('failed_saves', 0)}")
            logger.info(f"   - Total cuisine tags added: {save_stats.get('total_tags_added', 0)}")

            return {
                **x,
                "restaurants_processed": total_count,
                "processed_restaurants": restaurants_processed,
                "save_statistics": save_stats,
                "all_restaurants_saved": restaurants_processed  # Keep ALL restaurants for editor
            }

        except Exception as e:
            logger.error(f"❌ Error in AI Supabase update step: {e}")
            import traceback
            traceback.print_exc()
            return {**x, "restaurants_processed": 0, "processed_restaurants": [], "all_restaurants_saved": []}

    def _analyze_results_step(self, x):
        """Analyze results considering AI-matched database and newly scraped data"""
        try:
            # Get AI database coverage info
            database_coverage = x.get("database_coverage", {})
            existing_restaurants = database_coverage.get("restaurants", [])
            enriched_results = x.get("enriched_results", [])

            if existing_restaurants and not enriched_results:
                # Pure database mode - return existing restaurants
                logger.info(f"📊 Using AI-matched database restaurants: {len(existing_restaurants)}")
                return {
                    **x,
                    "using_ai_database": True,
                    "recommendations": {"main_list": existing_restaurants[:15]}
                }

            elif enriched_results:
                # Traditional analysis with scraped content
                logger.info(f"📊 Analyzing {len(enriched_results)} scraped results")

                # Run async analysis in thread pool
                def run_analysis():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self._analyze_results_async(x))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(run_analysis).result()

                return {
                    **result,
                    "using_ai_database": False
                }

            else:
                logger.warning("No data available for analysis")
                return {
                    **x,
                    "using_ai_database": False,
                    "recommendations": {"main_list": []}
                }

        except Exception as e:
            logger.error(f"❌ Error in analysis step: {e}")
            return {
                **x,
                "using_ai_database": False,
                "recommendations": {"main_list": []}
            }

    # MODIFIED STEP 7: Editor works with AI-matched database restaurants
    def _edit_step(self, x):
        """Edit step that handles AI-matched database and scraped content"""
        try:
            using_ai_database = x.get("using_ai_database", False)

            if using_ai_database:
                # Work with AI-matched database restaurants
                database_coverage = x.get("database_coverage", {})
                restaurants = database_coverage.get("restaurants", [])

                logger.info(f"📝 Editor processing {len(restaurants)} AI-matched restaurants from database")

                # Convert database format to editor format with AI enhancement
                main_list = []
                for restaurant in restaurants[:15]:  # Limit to top 15
                    main_list.append({
                        "name": restaurant.get("name", ""),
                        "address": restaurant.get("address", ""),
                        "description": self._enhance_description_with_ai(restaurant),
                        "sources": [self._clean_source_url(url) for url in restaurant.get("sources", [])],
                        "mention_count": restaurant.get("mention_count", 1),
                        "cuisine_tags": restaurant.get("cuisine_tags", []),
                        "from_database": True
                    })

                return {
                    **x,
                    "edited_results": {"main_list": main_list},
                    "follow_up_queries": []
                }

            else:
                # Traditional editor processing with scraped content
                enriched_results = x.get("enriched_results", [])
                original_query = x.get("query", "")
                destination = x.get("destination", "Unknown")

                if not enriched_results:
                    logger.warning("No scraped results available for editing")
                    return {
                        **x,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call the editor with scraped results
                edit_output = self.editor_agent.edit(
                    scraped_results=enriched_results,
                    original_query=original_query,
                    destination=destination
                )

                return {
                    **x, 
                    "edited_results": edit_output.get("edited_results", {"main_list": []}),
                    "follow_up_queries": edit_output.get("follow_up_queries", [])
                }

        except Exception as e:
            logger.error(f"❌ Error in AI edit step: {e}")
            return {
                **x,
                "edited_results": {"main_list": []},
                "follow_up_queries": []
            }

    # MODIFIED STEP 8: Follow-up with AI geodata updates
    def _follow_up_step_with_geodata(self, x):
        """Follow-up search with AI-enhanced geodata updates to Supabase for ALL restaurants"""
        try:
            edited_results = x.get("edited_results", {})
            follow_up_queries = x.get("follow_up_queries", [])
            destination = x.get("destination", "Unknown")
            city = x.get("city", "Unknown")
            country = x.get("country", "Unknown")

            if not edited_results.get("main_list"):
                logger.warning("No restaurants available for follow-up")
                return {**x, "enhanced_results": {"main_list": []}}

            # Run follow-up search for final recommendations
            enhanced_output = self.follow_up_search_agent.enhance(
                edited_results=edited_results,
                follow_up_queries=follow_up_queries,
                destination=destination
            )

            # NEW: Update geodata for ALL restaurants in the city (not just recommendations)
            logger.info(f"🗺️ Starting city-wide geodata update for {city}, {country}")

            try:
                geodata_stats = update_city_geodata(
                    city=city,
                    country=country,
                    config=self.config
                )

                logger.info(f"✅ City-wide geodata update complete:")
                logger.info(f"   - Restaurants updated: {geodata_stats.get('updated_count', 0)}")
                logger.info(f"   - Update failures: {geodata_stats.get('failed_count', 0)}")

            except Exception as geo_error:
                logger.warning(f"⚠️ Geodata update failed: {geo_error}")
                geodata_stats = {'updated_count': 0, 'failed_count': 0}

            return {
                **x, 
                **enhanced_output,
                "geodata_update_stats": geodata_stats
            }

        except Exception as e:
            logger.error(f"❌ Error in AI follow-up step: {e}")
            import traceback
            traceback.print_exc()
            return {**x, "enhanced_results": {"main_list": []}}

    # NEW AI-POWERED HELPER METHODS

    def _detect_country_from_city(self, city: str) -> str:
        """AI-powered country detection for any city worldwide"""
        if not city or city == "Unknown":
            return "Unknown"

        try:
            from langchain_openai import ChatOpenAI

            # Use lightweight model for simple country detection
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Faster and cheaper
                temperature=0,
                openai_api_key=getattr(self.config, 'OPENAI_API_KEY', None)
            )

            prompt = f"""What country is the city "{city}" located in? 

Respond with ONLY the country name in English. Examples:
- Paris -> France
- Tokyo -> Japan  
- New York -> United States
- São Paulo -> Brazil
- Mumbai -> India

City: {city}
Country:"""

            response = llm.invoke(prompt)
            country = response.content.strip()

            # Basic validation
            if len(country) > 50 or len(country) < 2:
                logger.warning(f"AI returned invalid country name: {country}")
                return "Unknown"

            logger.info(f"🌍 AI detected country: {city} -> {country}")
            return country

        except Exception as e:
            logger.warning(f"AI country detection failed for {city}: {e}")
            return "Unknown"

    def _extract_search_preferences_ai(self, parameters: List[str]) -> Dict[str, List[str]]:
        """AI-powered extraction of cuisine types, atmosphere, and dining preferences"""
        if not parameters:
            return {"cuisines": [], "atmosphere": [], "dining_style": [], "features": []}

        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=getattr(self.config, 'OPENAI_API_KEY', None)
            )

            parameters_text = ", ".join(parameters)

            prompt = f"""Analyze these restaurant search parameters and extract structured preferences.

Parameters: {parameters_text}

Extract and categorize into:
1. CUISINES: Types of food (italian, french, japanese, etc.)
2. ATMOSPHERE: Mood/setting (romantic, casual, cozy, upscale, etc.)  
3. DINING_STYLE: Style of dining (fine dining, bistro, family-friendly, etc.)
4. FEATURES: Special features (outdoor seating, wine bar, rooftop, etc.)

Return ONLY a JSON object:
{{
  "cuisines": ["cuisine1", "cuisine2"],
  "atmosphere": ["atmosphere1", "atmosphere2"], 
  "dining_style": ["style1", "style2"],
  "features": ["feature1", "feature2"]
}}

Use lowercase, be comprehensive but precise. Include related/similar terms."""

            response = llm.invoke(prompt)

            try:
                preferences = json.loads(response.content.strip())

                # Validate structure
                expected_keys = ["cuisines", "atmosphere", "dining_style", "features"]
                for key in expected_keys:
                    if key not in preferences or not isinstance(preferences[key], list):
                        preferences[key] = []

                logger.info(f"🎯 AI extracted preferences: {preferences}")
                return preferences

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse AI preferences response: {response.content}")
                return {"cuisines": [], "atmosphere": [], "dining_style": [], "features": []}

        except Exception as e:
            logger.warning(f"AI preference extraction failed: {e}")
            return {"cuisines": [], "atmosphere": [], "dining_style": [], "features": []}

    def _check_database_with_ai_matching(self, city: str, search_preferences: Dict[str, List[str]]) -> Dict[str, Any]:
        """Check database using AI-powered semantic matching instead of exact matches"""
        try:
            # Get all restaurants for the city
            coverage = check_city_coverage(city, "", self.config)  # Get all restaurants first
            all_restaurants = coverage.get("restaurants", [])

            if not all_restaurants:
                return {
                    "has_data": False,
                    "total_restaurants": 0,
                    "preference_matches": 0,
                    "sufficient_coverage": False,
                    "restaurants": []
                }

            # AI-powered semantic matching
            matched_restaurants = self._ai_match_restaurants(all_restaurants, search_preferences)

            return {
                "has_data": True,
                "total_restaurants": len(all_restaurants),
                "preference_matches": len(matched_restaurants),
                "sufficient_coverage": len(matched_restaurants) >= 5,
                "restaurants": matched_restaurants,
                "search_preferences": search_preferences
            }

        except Exception as e:
            logger.error(f"Error in AI database matching: {e}")
            return {
                "has_data": False,
                "total_restaurants": 0,
                "preference_matches": 0,
                "sufficient_coverage": False,
                "restaurants": []
            }

    def _ai_match_restaurants(self, restaurants: List[Dict], search_preferences: Dict[str, List[str]]) -> List[Dict]:
        """Use AI to semantically match restaurants to search preferences"""
        try:
            if not search_preferences or not restaurants:
                return restaurants[:10]  # Return top 10 if no preferences

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=getattr(self.config, 'OPENAI_API_KEY', None)
            )

            # Create a concise restaurant summary for AI analysis
            restaurant_summaries = []
            for i, restaurant in enumerate(restaurants[:20]):  # Limit to 20 for cost
                summary = {
                    "index": i,
                    "name": restaurant.get("name", ""),
                    "cuisine_tags": restaurant.get("cuisine_tags", []),
                    "description": restaurant.get("raw_description", "")[:200]  # Truncate for cost
                }
                restaurant_summaries.append(summary)

            prompt = f"""Match restaurants to search preferences using semantic similarity.

SEARCH PREFERENCES:
Cuisines: {search_preferences.get('cuisines', [])}
Atmosphere: {search_preferences.get('atmosphere', [])}
Dining Style: {search_preferences.get('dining_style', [])}
Features: {search_preferences.get('features', [])}

RESTAURANTS:
{json.dumps(restaurant_summaries, indent=2)}

Return restaurant indices that match the preferences (semantic matching, not exact).
Examples:
- "romantic" matches restaurants tagged "cozy", "intimate", "date night"
- "italian" matches "neapolitan", "tuscan", "pasta", "pizza"
- "casual" matches "bistro", "neighborhood", "family-friendly"

Return ONLY a JSON array of matching indices: [0, 3, 7, 12]
Maximum 10 matches, ordered by relevance."""

            response = llm.invoke(prompt)

            try:
                matching_indices = json.loads(response.content.strip())

                if not isinstance(matching_indices, list):
                    logger.warning(f"AI returned non-list: {matching_indices}")
                    return restaurants[:8]  # Fallback

                # Return matched restaurants
                matched = []
                for idx in matching_indices:
                    if isinstance(idx, int) and 0 <= idx < len(restaurants):
                        matched.append(restaurants[idx])

                logger.info(f"🎯 AI matched {len(matched)} restaurants from {len(restaurants)} total")
                return matched[:10]  # Limit to 10

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse AI matching response: {response.content}")
                return restaurants[:8]  # Fallback

        except Exception as e:
            logger.warning(f"AI restaurant matching failed: {e}")
            return restaurants[:8]  # Fallback

    def _enhance_description_with_ai(self, restaurant: Dict) -> str:
        """AI-enhance restaurant description for better user experience"""
        try:
            raw_description = restaurant.get("raw_description", "")
            name = restaurant.get("name", "")
            cuisine_tags = restaurant.get("cuisine_tags", [])

            if not raw_description or len(raw_description) < 50:
                return f"{name} - {', '.join(cuisine_tags[:3])}"

            # Return first 150 chars of raw description with cuisine context
            description = raw_description[:150].rsplit(' ', 1)[0]  # Cut at word boundary
            if cuisine_tags:
                cuisine_context = f" ({', '.join(cuisine_tags[:2])})"
                return description + cuisine_context

            return description

        except Exception as e:
            logger.warning(f"Description enhancement failed: {e}")
            return restaurant.get("name", "Restaurant")

    def _clean_source_url(self, url: str) -> str:
        """Clean source URL to domain name"""
        try:
            if not url:
                return "unknown"

            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            if domain.startswith('www.'):
                domain = domain[4:]

            return domain

        except Exception:
            return "unknown"

    def _update_restaurant_geodata_ai(self, enhanced_results: Dict[str, Any]):
        """AI-enhanced update of restaurant geodata in Supabase"""
        try:
            from agents.supabase_update_agent import SupabaseUpdateAgent

            agent = SupabaseUpdateAgent(self.config)
            restaurants = enhanced_results.get("main_list", [])

            for restaurant in restaurants:
                name = restaurant.get("name", "")
                address = restaurant.get("address", "")

                # Check if we have coordinates from follow-up search
                if hasattr(restaurant, 'latitude') and hasattr(restaurant, 'longitude'):
                    coordinates = (restaurant.latitude, restaurant.longitude)

                    # Find restaurant in database and update with AI validation
                    try:
                        existing = agent.supabase.table('restaurants')\
                            .select('id, name')\
                            .eq('name', name)\
                            .execute()

                        if existing.data:
                            restaurant_id = existing.data[0]['id']
                            agent.update_restaurant_geodata(restaurant_id, address, coordinates)
                            logger.info(f"📍 AI-updated geodata for: {name}")

                    except Exception as e:
                        logger.warning(f"Failed to update geodata for {name}: {e}")

        except Exception as e:
            logger.error(f"Error in AI geodata update: {e}")

    # KEEP ALL YOUR EXISTING METHODS (just add the new ones above)

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

    async def _extract_restaurant_names_with_ai(self, content: str) -> List[str]:
        """Extract restaurant names using AI instead of regex patterns"""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=getattr(self.config, 'OPENAI_API_KEY', None)
            )

            prompt = f"""Extract restaurant names from this content. Return only a JSON list of restaurant names.

Content: {content[:1500]}

Return format: ["Restaurant Name 1", "Restaurant Name 2"]
Maximum 10 restaurants. Return empty list [] if no restaurants found.
Focus on actual restaurant names, not generic terms."""

            response = llm.invoke(prompt)

            try:
                # Try to parse as JSON
                result = json.loads(response.content.strip())
                if isinstance(result, list):
                    return [name for name in result if isinstance(name, str) and len(name) > 2][:10]
            except json.JSONDecodeError:
                # Fallback: extract quoted strings
                import re
                matches = re.findall(r'"([^"]+)"', response.content)
                return [name for name in matches if len(name) > 2][:10]

            return []

        except Exception as e:
            logger.warning(f"AI restaurant extraction failed: {e}")
            return []

    def _save_scraped_content_for_rag(self, enriched_results):
        """Save scraped content to Supabase for RAG - SYNC WRAPPER VERSION"""
        try:
            # Run the async function in a new event loop
            async def async_save():
                saved_count = 0

                for result in enriched_results:
                    url = result.get("url", "")
                    scraped_content = result.get("scraped_content", "")
                    scraping_success = result.get("scraping_success", False)

                    if scraping_success and scraped_content and len(scraped_content) > 100:
                        try:
                            # Extract domain name for attribution
                            domain_name = self._extract_domain_from_url(url)

                            # AI-powered restaurant name extraction
                            restaurant_mentions = await self._extract_restaurant_names_with_ai(scraped_content)

                            # Save to RAG system
                            success = save_scraped_content(
                                url, 
                                scraped_content, 
                                restaurant_mentions,
                                source_domain=domain_name
                            )
                            if success:
                                saved_count += 1
                                logger.info(f"✅ Saved RAG content from {domain_name}: {len(scraped_content)} chars, {len(restaurant_mentions)} restaurants")

                        except Exception as e:
                            logger.warning(f"Error saving scraped content for {url}: {e}")

                logger.info(f"📚 Saved {saved_count} articles to RAG system with AI-extracted restaurant names")

            # Run the async function
            try:
                # Check if there's already an event loop running
                loop = asyncio.get_running_loop()
                # If we're in an event loop, use create_task and don't wait
                asyncio.create_task(async_save())
                logger.info("🚀 Started async RAG saving task")
            except RuntimeError:
                # No loop running, create new one
                asyncio.run(async_save())

        except Exception as e:
            logger.error(f"Error in RAG content saving: {e}")

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract clean domain name from URL"""
        try:
            if not url or not isinstance(url, str):
                return "unknown-source"

            domain = urlparse(url.lower()).netloc

            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain or "unknown-source"

        except Exception as e:
            logger.debug(f"Error extracting domain: {e}")
            return "unknown-source"

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

            # Save restaurant data from results
            self._save_restaurant_data_from_results(main_list, x.get("destination", "Unknown"))

            # Format for Telegram using the formatter
            telegram_text = self.telegram_formatter.format_recommendations(
                enhanced_results
            )

            dump_chain_state("post_format", {
                "telegram_text_length": len(telegram_text),
                "restaurant_count": len(main_list)
            })

            return {
                **x,
                "telegram_formatted_text": telegram_text,
                "final_results": enhanced_results
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
    def process_query(self, user_query: str, user_preferences: Dict[str, Any] = None, user_id: str = None) -> Dict[str, Any]:
        """Process restaurant query through the AI-enhanced LangChain pipeline"""
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

                # Execute the AI-enhanced chain (synchronously)
                result = self.chain.invoke(input_data)

                # Log completion
                dump_chain_state("process_query_complete", {
                    "result_keys": list(result.keys()),
                    "has_enhanced_results": "enhanced_results" in result,
                    "has_telegram_text": "telegram_formatted_text" in result,
                    "destination": result.get("destination", "Unknown"),
                    "used_ai_database": result.get("using_ai_database", False),
                    "restaurants_processed": result.get("restaurants_processed", 0)
                })

                # Final usage summary
                self._log_firecrawl_usage()

                # Cache final results with AI enhancement info
                enhanced_results = result.get("enhanced_results", {})
                main_list = enhanced_results.get("main_list", [])

                cache_data = {
                    "query": user_query,
                    "destination": result.get("destination", "Unknown"),
                    "results": enhanced_results,
                    "restaurant_count": len(main_list),
                    "trace_id": trace_id,
                    "timestamp": time.time(),
                    "firecrawl_stats": self.scraper.get_stats(),
                    "ai_features": {
                        "used_ai_database": result.get("using_ai_database", False),
                        "restaurants_processed": result.get("restaurants_processed", 0),
                        "search_preferences": result.get("search_preferences", {}),
                        "country_detected": result.get("country", "Unknown")
                    },
                    "search_method": "ai_enhanced_" + ("database" if result.get("using_ai_database") else "web")
                }

                cache_search_results(user_query, cache_data)

                # User search history
                if user_id:
                    add_to_search_history(user_id, user_query, len(main_list))

                # Extract results with correct key names
                telegram_text = result.get("telegram_formatted_text", 
                                         "Sorry, no recommendations found.")

                logger.info(f"✅ AI-Enhanced Result - {len(main_list)} restaurants for {result.get('destination', 'Unknown')}")
                logger.info(f"🤖 AI Features: DB={result.get('using_ai_database', False)}, Processed={result.get('restaurants_processed', 0)}")

                # Return with correct key names that telegram_bot.py expects
                return {
                    "telegram_formatted_text": telegram_text,
                    "enhanced_results": enhanced_results,
                    "main_list": main_list,
                    "destination": result.get("destination"),
                    "firecrawl_stats": self.scraper.get_stats(),
                    "ai_features": {
                        "used_ai_database": result.get("using_ai_database", False),
                        "restaurants_processed": result.get("restaurants_processed", 0),
                        "search_preferences": result.get("search_preferences", {}),
                        "country_detected": result.get("country", "Unknown")
                    },
                    "search_method": "ai_enhanced_" + ("database" if result.get("using_ai_database") else "web")
                }

            except Exception as e:
                logger.error(f"Error in AI-enhanced chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query}, error=e)
                self._log_firecrawl_usage()

                return {
                    "main_list": [],
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",
                    "firecrawl_stats": self.scraper.get_stats(),
                    "ai_features": {
                        "used_ai_database": False,
                        "restaurants_processed": 0,
                        "search_preferences": {},
                        "country_detected": "Unknown"
                    },
                    "search_method": "error"
                }