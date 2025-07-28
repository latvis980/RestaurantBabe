# agents/langchain_orchestrator.py
# UPDATED VERSION - Now uses DatabaseSearchAgent while preserving all existing features

import os
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import asyncio
import logging
import concurrent.futures

from utils.database import get_restaurants_by_city
from utils.debug_utils import dump_chain_state, log_function_call
from formatters.telegram_formatter import TelegramFormatter

# Create logger
logger = logging.getLogger("restaurant-recommender.orchestrator")

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents with correct file names
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent  # NEW AGENT
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)  # NEW AGENT
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)

        # Initialize formatter
        self.telegram_formatter = TelegramFormatter()

        self.config = config

        # Build the pipeline steps
        self._build_pipeline()

    def _build_pipeline(self):
        """Build pipeline with database agent integration"""

        # Step 1: Analyze Query
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"]
            },
            name="analyze_query"
        )

        # Step 2: Database Search and Decision (UPDATED - uses DatabaseSearchAgent)
        self.check_database = RunnableLambda(
            self._check_database_coverage,
            name="check_database"
        )

        # Step 3: Search (conditional - only if no database content)
        self.search = RunnableLambda(
            self._search_step,
            name="search"
        )

        # Step 4: Scrape (conditional - only if search happened)
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )

        # Step 5: Edit (handles both database restaurants and scraped content)
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )

        # Step 6: Follow-up Search
        self.follow_up_search = RunnableLambda(
            self._follow_up_step,
            name="follow_up_search"
        )

        # Step 7: Format for Telegram
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )

        # Create the complete chain
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.check_database,
                self.search,
                self.scrape,
                self.edit,
                self.follow_up_search,
                self.format_output,
            ],
            last=RunnableLambda(lambda x: x),
            name="restaurant_recommendation_chain"
        )

    def _check_database_coverage(self, x):
        """
        UPDATED: Enhanced database check that uses DatabaseSearchAgent and intelligent search.

        This method now delegates database logic to DatabaseSearchAgent while maintaining
        compatibility with intelligent search features and all existing functionality.
        """
        try:
            logger.info("🧠 CHECKING DATABASE WITH INTELLIGENT SEARCH")

            destination = x.get("destination", "Unknown")
            search_query = x.get("query", "")

            if destination == "Unknown":
                logger.info("⚠️ No destination detected, will search web")
                return {**x, "has_database_content": False, "database_results": []}

            if not search_query.strip():
                logger.warning("⚠️ No search query found")
                return {**x, "has_database_content": False, "database_results": []}

            logger.info(f"🔍 Intelligent search for: '{search_query}' in {destination}")

            # Try intelligent database search first (if available)
            try:
                from utils.intelligent_db_search import search_restaurants_intelligently

                relevant_restaurants, should_scrape = search_restaurants_intelligently(
                    query=search_query,
                    destination=destination,
                    config=self.config,
                    min_results=2,  # Need at least 2 relevant results
                    max_results=8   # Maximum to return from database
                )

                if relevant_restaurants and not should_scrape:
                    logger.info(
                        f"✅ Found {len(relevant_restaurants)} relevant restaurants in database - "
                        "skipping web scraping"
                    )
                    return {
                        **x, 
                        "has_database_content": True, 
                        "database_results": relevant_restaurants,
                        "skip_web_search": True,  # Flag to skip web search
                        "content_source": "database"
                    }
                elif relevant_restaurants and should_scrape:
                    logger.info(
                        f"📊 Found {len(relevant_restaurants)} relevant restaurants but need more - "
                        "will supplement with web scraping"
                    )
                    return {
                        **x, 
                        "has_database_content": True, 
                        "database_results": relevant_restaurants,
                        "skip_web_search": False,  # Continue to web search for more results
                        "content_source": "database_plus_web"
                    }
                else:
                    logger.info("📭 No relevant restaurants found in intelligent search - trying DatabaseSearchAgent")
                    # Fall through to DatabaseSearchAgent

            except ImportError:
                logger.warning("⚠️ Intelligent search not available, using DatabaseSearchAgent")
                # Fall through to DatabaseSearchAgent

            # Use DatabaseSearchAgent as fallback or primary method
            logger.info("🗃️ Using DatabaseSearchAgent for evaluation")
            database_result = self.database_search_agent.search_and_evaluate(x)

            # Merge result with pipeline state and add any missing fields
            result = {**x, **database_result}

            # Ensure skip_web_search flag is set correctly
            if result.get("has_database_content", False):
                result["skip_web_search"] = True
            else:
                result["skip_web_search"] = False

            return result

        except Exception as e:
            logger.error(f"❌ Error in database coverage check: {e}")
            # Fallback to basic database check
            return self._fallback_database_check(x)

    def _fallback_database_check(self, x):
        """Fallback database check using the original method"""
        try:
            destination = x.get("destination", "Unknown")

            if destination == "Unknown":
                return {**x, "has_database_content": False, "database_results": []}

            # Extract city from destination (simple parsing)
            city = destination
            if "," in destination:
                city = destination.split(",")[0].strip()

            logger.info(f"🔍 Basic database check for: {city}")

            # Query database for existing restaurants
            from utils.database import get_database
            db = get_database()
            database_restaurants = db.get_restaurants_by_city(city, limit=50)

            # Use lower threshold for basic search since it's less targeted
            if database_restaurants and len(database_restaurants) >= 5:
                logger.info(f"✅ Found {len(database_restaurants)} restaurants in database")
                return {
                    **x, 
                    "has_database_content": True, 
                    "database_results": database_restaurants[:8],  # Limit to 8 for processing
                    "content_source": "database",
                    "skip_web_search": True
                }
            else:
                logger.info(f"📭 Only {len(database_restaurants) if database_restaurants else 0} restaurants in database - not enough")
                return {
                    **x, 
                    "has_database_content": False, 
                    "database_results": [],
                    "content_source": "web_search",
                    "skip_web_search": False
                }
        except Exception as e:
            logger.error(f"❌ Error in fallback database check: {e}")
            return {
                **x, 
                "has_database_content": False, 
                "database_results": [],
                "content_source": "web_search",
                "skip_web_search": False
            }

    def _search_step(self, x):
        """Enhanced search step that can skip web search if database provided enough results"""
        try:
            logger.info("🔍 SEARCH STEP")

            # Check if we should skip web search
            if x.get("skip_web_search", False):
                logger.info("⏭️ Skipping web search - database provided sufficient results")
                return {**x, "search_results": []}

            destination = x.get("destination", "Unknown")
            search_terms = x.get("search_terms", [])
            language = x.get("language", "en")

            # If no search terms from query analyzer, create them from the original query
            if not search_terms:
                query = x.get("query", "")
                if query and destination != "Unknown":
                    # Simple search term extraction
                    search_terms = [query.replace(f" in {destination.lower()}", "").strip()]
                    logger.info(f"🔧 Created search terms from query: {search_terms}")

            if destination == "Unknown" or not search_terms:
                logger.warning("Missing destination or search terms for web search")
                return {**x, "search_results": []}

            # Build search query
            primary_term = search_terms[0] if search_terms else "restaurants"
            query = f"{primary_term} in {destination}"

            logger.info(f"🌐 Searching web for: {query}")

            # Perform search using existing search agent
            search_results = self.search_agent.search([query])  # Pass as list

            logger.info(f"✅ Web search completed: {len(search_results)} results")

            return {**x, "search_results": search_results}

        except Exception as e:
            logger.error(f"❌ Error in search step: {e}")
            return {**x, "search_results": []}

    def _scrape_step(self, x):
        """Scrape step - only runs if search happened"""
        try:
            # Check if we should skip scraping (database branch)
            if x.get("has_database_content", False):
                logger.info("⏭️ SKIPPING SCRAPING - using database content")
                logger.info("⏭️ → NO FILES SENT TO SUPABASE MANAGER (using existing data)")
                return {**x, "enriched_results": []}

            logger.info("🕷️ RUNNING WEB SCRAPING")
            logger.info("🕷️ → WILL SEND FILES TO SUPABASE MANAGER AFTER SCRAPING")

            search_results = x.get("search_results", [])

            if not search_results:
                logger.warning("⚠️ No search results to scrape")
                return {**x, "enriched_results": []}

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

            # Log usage after scraping
            self._log_firecrawl_usage()

            logger.info(f"✅ Scraped {len(enriched_results)} articles")

            # Save scraped content for Supabase manager
            if enriched_results:
                logger.info("💾 Proceeding to save scraped content...")
                self._save_scraped_content_for_processing_simple(x, enriched_results)
            else:
                logger.warning("⚠️ No enriched results to save")

            return {**x, "enriched_results": enriched_results}

        except Exception as e:
            logger.error(f"❌ Error in scraping step: {e}")
            return {**x, "enriched_results": []}

    def _edit_step(self, x):
        """Edit step - handles both database restaurants and scraped content"""
        try:
            logger.info("✏️ ENTERING EDIT STEP")

            has_database_content = x.get("has_database_content", False)
            original_query = x.get("query", "")
            destination = x.get("destination", "Unknown")

            if has_database_content:
                # DATABASE BRANCH: Format existing restaurants
                logger.info("🗃️ Processing DATABASE restaurants")

                database_results = x.get("database_results", [])

                if not database_results:
                    logger.warning("⚠️ No database results to process")
                    return {
                        **x,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call editor with database restaurants
                edit_output = self.editor_agent.edit(
                    scraped_results=None,
                    database_restaurants=database_results,
                    original_query=original_query,
                    destination=destination
                )

                logger.info(f"✅ Formatted {len(edit_output.get('edited_results', {}).get('main_list', []))} database restaurants")

            else:
                # WEB SEARCH BRANCH: Process scraped content
                logger.info("🌐 Processing SCRAPED content")

                scraped_results = x.get("enriched_results", [])

                if not scraped_results:
                    logger.warning("⚠️ No scraped results to process")
                    return {
                        **x,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # Call editor with scraped content
                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=None,
                    original_query=original_query,
                    destination=destination
                )

                logger.info(f"✅ Processed {len(edit_output.get('edited_results', {}).get('main_list', []))} restaurants from scraped content")

            return {
                **x, 
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"❌ Error in edit step: {e}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)
            return {
                **x,
                "edited_results": {"main_list": []},
                "follow_up_queries": []
            }

    def _follow_up_step(self, x):
        """Follow-up search step - processes edited_results and returns enhanced_results"""
        try:
            logger.info("🔍 ENTERING FOLLOW-UP STEP")

            edited_results = x.get("edited_results", {})
            follow_up_queries = x.get("follow_up_queries", [])

            if not edited_results.get("main_list"):
                logger.warning("⚠️ No restaurants available for follow-up search")
                return {**x, "enhanced_results": {"main_list": []}}

            logger.info(f"🔍 Processing {len(edited_results['main_list'])} restaurants for follow-up")

            # Call follow-up search with edited results
            followup_output = self.follow_up_search_agent.perform_follow_up_searches(
                edited_results=edited_results,
                follow_up_queries=follow_up_queries,
                destination=x.get("destination", "Unknown"),
                secondary_filter_parameters=x.get("secondary_filter_parameters")
            )

            enhanced_results = followup_output.get("enhanced_results", {"main_list": []})

            logger.info(f"✅ Follow-up complete: {len(enhanced_results.get('main_list', []))} restaurants remain after filtering")

            return {**x, "enhanced_results": enhanced_results}

        except Exception as e:
            logger.error(f"❌ Error in follow-up step: {e}")
            dump_chain_state("follow_up_error", x, error=e)
            return {**x, "enhanced_results": {"main_list": []}}

    def _format_step(self, x):
        """Format step - converts enhanced_results to telegram_formatted_text"""
        try:
            logger.info("📱 ENTERING FORMAT STEP")

            enhanced_results = x.get("enhanced_results", {})
            main_list = enhanced_results.get("main_list", [])

            if not main_list:
                logger.warning("⚠️ No restaurants to format for Telegram")
                return {
                    **x,
                    "telegram_formatted_text": "Sorry, no restaurant recommendations found for your query."
                }

            logger.info(f"📱 Formatting {len(main_list)} restaurants for Telegram")

            # Format for Telegram using the formatter
            telegram_text = self.telegram_formatter.format_recommendations(
                enhanced_results  # Pass the entire enhanced_results dict
            )

            logger.info("✅ Telegram formatting complete")

            return {
                **x,
                "telegram_formatted_text": telegram_text,
                "final_results": enhanced_results
            }

        except Exception as e:
            logger.error(f"❌ Error in format step: {e}")
            dump_chain_state("format_error", x, error=e)
            return {
                **x,
                "telegram_formatted_text": "Sorry, there was an error formatting the restaurant recommendations."
            }

    def _save_scraped_content_for_processing_simple(self, x, enriched_results):
        """Save scraped content and send to Supabase manager (working version)"""
        try:
            logger.info("💾 ENTERING SIMPLE SAVE SCRAPED CONTENT")
            logger.info(f"💾 Enriched results count: {len(enriched_results)}")

            from datetime import datetime
            import threading
            import requests

            # Extract metadata from pipeline context
            query = x.get("query", "")
            destination = x.get("destination", "Unknown")

            # Parse destination into city/country
            city = destination
            country = "Unknown"
            if "," in destination:
                parts = [p.strip() for p in destination.split(",")]
                city = parts[0]
                if len(parts) > 1:
                    country = parts[1]

            # Combine all scraped content for saving
            all_scraped_content = ""
            sources = []

            for result in enriched_results:
                try:
                    content = result.get("scraped_content", result.get("content", ""))
                    url = result.get("url", "")

                    if content and len(content.strip()) > 100:
                        all_scraped_content += f"\n\n--- FROM {url} ---\n\n{content}"
                        sources.append(url)
                        logger.info(f"✅ Got {len(content)} chars from {url}")
                    else:
                        logger.warning(f"⚠️ No substantial content from {url}")

                except Exception as e:
                    logger.error(f"❌ Error processing result: {e}")
                    continue

            if not all_scraped_content.strip():
                logger.warning("⚠️ No content to save")
                return

            # Save to local file first (for backup/debugging)
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scraped_{city.replace(' ', '_')}_{timestamp}.txt"

                os.makedirs("scraped_content", exist_ok=True)
                file_path = os.path.join("scraped_content", filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(all_scraped_content)

                logger.info(f"💾 Saved scraped content to: {file_path}")

            except Exception as e:
                logger.error(f"❌ Error saving local file: {e}")

            # Send to Supabase Manager service (async, don't wait for response)
            def send_to_supabase_manager():
                try:
                    supabase_manager_url = getattr(self.config, 'SUPABASE_MANAGER_URL', '')

                    if not supabase_manager_url:
                        logger.warning("⚠️ SUPABASE_MANAGER_URL not configured - skipping background update")
                        return

                    logger.info(f"📤 Sending content to Supabase Manager: {supabase_manager_url}")

                    # Prepare payload (same format as working version)
                    payload = {
                        'content': all_scraped_content,
                        'metadata': {
                            'city': city,
                            'country': country,
                            'sources': sources,
                            'query': query,
                            'scraped_at': datetime.now().isoformat()
                        }
                    }

                    # Send to the correct endpoint
                    response = requests.post(
                        f"{supabase_manager_url}/process_scraped_content",
                        json=payload,
                        timeout=180
                    )

                    if response.status_code == 200:
                        logger.info("✅ Successfully sent content to Supabase Manager")
                    else:
                        logger.warning(f"⚠️ Supabase Manager returned status {response.status_code}")
                        logger.warning(f"Response: {response.text}")

                except Exception as e:
                    logger.error(f"❌ Error sending to Supabase Manager: {e}")

            # Run in background thread so it doesn't block user response
            thread = threading.Thread(target=send_to_supabase_manager, daemon=True)
            thread.start()
            logger.info("📤 Started background thread to send content to Supabase Manager")

        except Exception as e:
            logger.error(f"❌ Error in simple save scraped content: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def _save_scraped_content_for_processing(self, x, enriched_results):
        """Save scraped content to TXT file for Supabase manager processing"""
        try:
            logger.info("💾 ENTERING SAVE SCRAPED CONTENT")
            logger.info(f"💾 Enriched results count: {len(enriched_results)}")

            import tempfile
            import json
            from datetime import datetime

            # Extract metadata from pipeline context
            query = x.get("query", "")
            destination = x.get("destination", "Unknown")

            logger.info(f"💾 Query: {query}")
            logger.info(f"💾 Destination: {destination}")

            # Parse destination into city/country
            city = destination
            country = "Unknown"
            if "," in destination:
                parts = [p.strip() for p in destination.split(",")]
                city = parts[0]
                if len(parts) > 1:
                    country = parts[1]

            logger.info(f"💾 Parsed city: {city}, country: {country}")

            # Prepare metadata for Supabase manager
            metadata = {
                "query": query,
                "city": city,
                "country": country,
                "destination": destination,
                "timestamp": datetime.now().isoformat(),
                "source": "web_scraping",
                "total_articles": len(enriched_results),
                "search_queries": x.get("search_queries", [])
            }

            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Use configured path or temp directory
            if hasattr(self.config, 'SCRAPED_CONTENT_PATH'):
                base_path = self.config.SCRAPED_CONTENT_PATH
                logger.info(f"💾 Using configured path: {base_path}")
            else:
                base_path = tempfile.gettempdir()
                logger.info(f"💾 Using temp directory: {base_path}")

            # Ensure directory exists
            os.makedirs(base_path, exist_ok=True)
            logger.info(f"💾 Directory created/verified: {base_path}")

            # Save RAW CONTENT as TXT file
            content_filename = f"scraped_content_{city}_{timestamp}.txt"
            content_filepath = os.path.join(base_path, content_filename)

            logger.info(f"💾 Content file path: {content_filepath}")

            # Combine all scraped content into one text file
            with open(content_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SCRAPED RESTAURANT CONTENT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Query: {query}\n")
                f.write(f"City: {city}\n")
                f.write(f"Country: {country}\n")
                f.write(f"Timestamp: {metadata['timestamp']}\n")
                f.write(f"Total Articles: {len(enriched_results)}\n")
                f.write(f"Search Queries: {', '.join(metadata['search_queries'])}\n")
                f.write("\n" + "=" * 80 + "\n\n")

                # Add each scraped article
                for i, article in enumerate(enriched_results, 1):
                    url = article.get('url', 'Unknown URL')
                    title = article.get('title', 'No title')
                    content = article.get('content', '')

                    f.write(f"ARTICLE {i}:\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Title: {title}\n")
                    f.write("-" * 40 + "\n")

                    if content:
                        f.write(content)
                    else:
                        f.write("[No content available]")

                    f.write("\n\n" + "=" * 40 + "\n\n")

            logger.info(f"💾 Content file written successfully")

            # Save METADATA as separate JSON file
            metadata_filename = f"metadata_{city}_{timestamp}.json"
            metadata_filepath = os.path.join(base_path, metadata_filename)

            logger.info(f"💾 Metadata file path: {metadata_filepath}")

            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Metadata file written successfully")
            logger.info(f"💾 Saved scraped content: {content_filepath}")
            logger.info(f"📊 Content: {len(enriched_results)} articles for {city}")

            # Send to Supabase manager if URL configured
            supabase_manager_url = getattr(self.config, 'SUPABASE_MANAGER_URL', '')
            logger.info(f"💾 Checking Supabase Manager URL: '{supabase_manager_url}'")

            if supabase_manager_url:
                logger.info(f"📤 Found Supabase Manager URL: {supabase_manager_url}")
                self._send_to_supabase_manager(content_filepath, metadata_filepath, metadata)
            else:
                logger.warning("⚠️ No SUPABASE_MANAGER_URL configured - content saved locally only")
                logger.info(f"📁 Files saved locally:")
                logger.info(f"📁 - Content: {content_filepath}")
                logger.info(f"📁 - Metadata: {metadata_filepath}")

        except Exception as e:
            logger.error(f"❌ Error saving scraped content: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def _send_to_supabase_manager(self, content_filepath, metadata_filepath, metadata):
        """Send scraped content and metadata files to Supabase manager service"""
        try:
            import requests

            supabase_manager_url = self.config.SUPABASE_MANAGER_URL

            logger.info(f"📤 Sending content to Supabase manager: {supabase_manager_url}")

            # Send both files to Supabase manager
            with open(content_filepath, 'rb') as content_file, \
                 open(metadata_filepath, 'rb') as metadata_file:

                files = {
                    'content_file': (os.path.basename(content_filepath), content_file, 'text/plain'),
                    'metadata_file': (os.path.basename(metadata_filepath), metadata_file, 'application/json')
                }

                response = requests.post(
                    f"{supabase_manager_url}/process_content",
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:
                logger.info("✅ Successfully sent content to Supabase manager")

                # Delete local files after successful send
                try:
                    os.remove(content_filepath)
                    os.remove(metadata_filepath)
                    logger.info(f"🗑️ Deleted local files")
                except:
                    pass
            else:
                logger.error(f"❌ Failed to send to Supabase manager: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error sending to Supabase manager: {e}")
            logger.info(f"📁 Content saved locally: {content_filepath}")

    def _log_firecrawl_usage(self):
        """Log Firecrawl usage statistics"""
        try:
            stats = self.scraper.get_stats()
            logger.info("=" * 50)
            logger.info("FIRECRAWL USAGE REPORT")
            logger.info("=" * 50)
            logger.info(f"URLs scraped: {stats.get('total_scraped', 0)}")
            logger.info(f"Successful extractions: {stats.get('successful_extractions', 0)}")
            logger.info(f