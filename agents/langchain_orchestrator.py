# agents/langchain_orchestrator.py
# UPDATED VERSION - Now preserves raw query throughout pipeline while maintaining all existing features

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
        from agents.dbcontent_evaluation_agent import ContentEvaluationAgent  # NEW AGENT
        from agents.search_agent import BraveSearchAgent
        from agents.optimized_scraper import WebScraper
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.database_search_agent = DatabaseSearchAgent(config)  # NEW AGENT
        self.dbcontent_evaluation_agent = ContentEvaluationAgent(config)  # NEW AGENT
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
        """Build pipeline with ContentEvaluationAgent integration"""

        # Step 1: Analyze Query
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"],
                "raw_query": x["query"]
            },
            name="analyze_query"
        )

        # Step 2: Database Search
        self.check_database = RunnableLambda(
            self._check_database_coverage,
            name="check_database"
        )

        # Step 3: Content Evaluation (NEW STEP)
        self.evaluate_content = RunnableLambda(
            self._evaluate_and_enhance_content,
            name="evaluate_content"  
        )

        # Step 4: Search (conditional - now only if content evaluation says so)
        self.search = RunnableLambda(
            self._search_step,
            name="search"
        )

        # Step 5: Scrape (conditional - only if search happened)
        self.scrape = RunnableLambda(
            self._scrape_step,
            name="scrape"
        )

        # Step 6: Edit (receives optimized content from evaluation agent)
        self.edit = RunnableLambda(
            self._edit_step,
            name="edit"
        )

        # Step 7: Follow-up Search
        self.follow_up_search = RunnableLambda(
            self._follow_up_step,
            name="follow_up_search"
        )

        # Step 8: Format for Telegram
        self.format_output = RunnableLambda(
            self._format_step,
            name="format_output"
        )

        # Create the complete chain
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.check_database,
                self.evaluate_content,  # NEW STEP
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
        UPDATED: Enhanced to pass raw query to DatabaseSearchAgent.

        SIMPLIFIED: Pure routing method that delegates ALL database logic to DatabaseSearchAgent.
        The orchestrator only handles routing - no business logic here.
        """
        try:
            logger.info("🗃️ ROUTING TO DATABASE SEARCH AGENT")

            # Pass the raw query along with other data
            query_data_with_raw = {
                **x,
                "raw_query": x.get("raw_query", x.get("query", ""))  # Ensure raw query is passed
            }

            # Delegate everything to the DatabaseSearchAgent
            database_result = self.database_search_agent.search_and_evaluate(query_data_with_raw)

            # Simple merge and return - no logic here, but preserve raw query
            return {
                **x, 
                **database_result,
                "raw_query": x.get("raw_query", x.get("query", ""))  # Preserve raw query
            }

        except Exception as e:
            logger.error(f"❌ Error routing to database search agent: {e}")
            # Simple fallback - no complex logic in orchestrator
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query even on error
                "has_database_content": False,
                "database_results": [],
                "content_source": "web_search",
                "skip_web_search": False,
                "evaluation_details": {
                    "sufficient": False,
                    "reason": f"routing_error: {str(e)}",
                    "details": {"error": str(e)}
                }
            }

    def _evaluate_and_enhance_content(self, x):
        """
        NEW STEP: Route to ContentEvaluationAgent for intelligent content assessment
        This is where the magic happens - decides if database content is sufficient
        or if supplemental web search is needed
        """
        try:
            logger.info("🧠 ROUTING TO CONTENT EVALUATION AGENT")

            # Check if we have database content to evaluate
            database_restaurants = x.get("database_results", [])
            has_database_content = x.get("has_database_content", False)

            if not has_database_content or not database_restaurants:
                logger.info("📝 No database content to evaluate - proceeding to web search")
                return {
                    **x,
                    "content_evaluation_result": {
                        "content_source": "web_search",
                        "web_search_triggered": True,
                        "skip_database": True,
                        "evaluation_summary": {"reason": "no_database_content"}
                    },
                    "skip_web_search": False  # Ensure web search happens
                }

            # Delegate to ContentEvaluationAgent
            evaluation_result = self.dcontent_evaluation_agent.evaluate_and_enhance(
                database_restaurants=database_restaurants,
                raw_query=x.get("raw_query", x.get("query", "")),
                destination=x.get("destination", "Unknown"),
                search_queries=x.get("search_queries", []),
                primary_search_parameters=x.get("primary_search_parameters", []),
                secondary_filter_parameters=x.get("secondary_filter_parameters", [])
            )

            # Update pipeline state based on evaluation
            content_source = evaluation_result.get("content_source", "database")
            web_search_triggered = evaluation_result.get("web_search_triggered", False)
            optimized_content = evaluation_result.get("optimized_content", {})

            logger.info(f"🧠 Content evaluation complete:")
            logger.info(f"   📊 Content source: {content_source}")
            logger.info(f"   🌐 Web search triggered: {web_search_triggered}")

            # Prepare response with optimized content
            response = {
                **x,
                "content_evaluation_result": evaluation_result,
                "content_source": content_source,
                "web_search_triggered": web_search_triggered,
                "skip_web_search": not web_search_triggered,  # Control web search step
            }

            # Add optimized content to response
            if content_source == "hybrid":
                # Hybrid: has both database and supplemental web content
                response.update({
                    "database_results": optimized_content.get("database_restaurants", []),
                    "supplemental_scraped_results": optimized_content.get("scraped_results", []),
                    "has_database_content": True,
                    "has_supplemental_content": True
                })
            elif content_source == "database":
                # Database only: evaluation determined it's sufficient
                response.update({
                    "database_results": optimized_content.get("database_restaurants", []),
                    "has_database_content": True,
                    "has_supplemental_content": False
                })
            else:
                # Web search: database was insufficient or missing
                response.update({
                    "skip_web_search": False,  # Ensure web search happens
                    "has_database_content": False,
                    "content_source": "web_search"
                })

            return response

        except Exception as e:
            logger.error(f"❌ Error in content evaluation step: {e}")
            # Fallback: proceed with original database content
            return {
                **x,
                "content_evaluation_error": str(e),
                "skip_web_search": False  # Allow web search as fallback
            }

    def _search_step(self, x):
        """
        FIXED: Simple orchestrator method that just bridges query_analyzer to search_agent
        No business logic, no query generation - just pass through the AI-generated queries
        """
        try:
            logger.info("🔍 SEARCH STEP")

            # Check if we should skip web search
            if x.get("skip_web_search", False):
                logger.info("⏭️ Skipping web search - database provided sufficient results")
                return {**x, "search_results": []}

            # FIXED: Use the search_queries that query_analyzer already generated!
            search_queries = x.get("search_queries", [])
            destination = x.get("destination", "Unknown")

            if not search_queries or destination == "Unknown":
                logger.warning("Missing search queries or destination for web search")
                logger.warning(f"Available keys in x: {list(x.keys())}")
                return {**x, "search_results": []}

            logger.info(f"🌐 Using {len(search_queries)} AI-generated search queries:")
            for i, query in enumerate(search_queries, 1):
                logger.info(f"  {i}. {query}")

            # FIXED: Pass the AI-generated queries directly to search agent
            search_results = self.search_agent.search(search_queries, destination)

            logger.info(f"✅ Web search completed: {len(search_results)} results")

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "search_results": search_results
            }

        except Exception as e:
            logger.error(f"❌ Error in search step: {e}")
            logger.error(f"Available data: {x.keys()}")
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

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "enriched_results": enriched_results
            }

        except Exception as e:
            logger.error(f"❌ Error in scraping step: {e}")
            return {**x, "enriched_results": []}

    def _edit_step(self, x):
        """
        FIXED: Updated to use correct parameter names for EditorAgent
        """
        try:
            logger.info("✏️ ROUTING TO EDITOR AGENT")

            has_database_content = x.get("has_database_content", False)
            raw_query = x.get("raw_query", x.get("query", ""))
            destination = x.get("destination", "Unknown")

            if has_database_content:
                # DATABASE BRANCH: Process database restaurants
                logger.info("🗃️ Processing DATABASE restaurants")

                database_results = x.get("database_results", [])

                if not database_results:
                    logger.warning("⚠️ No database results to process")
                    return {
                        **x,
                        "raw_query": raw_query,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # FIXED: Use correct parameter names
                edit_output = self.editor_agent.edit(
                    scraped_results=None,
                    database_restaurants=database_results,
                    original_query=raw_query,  # Use original_query instead of raw_query
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
                        "raw_query": raw_query,
                        "edited_results": {"main_list": []},
                        "follow_up_queries": []
                    }

                # FIXED: Use correct parameter names
                edit_output = self.editor_agent.edit(
                    scraped_results=scraped_results,
                    database_restaurants=None,
                    original_query=raw_query,  # Use original_query instead of raw_query
                    destination=destination
                )

                logger.info(f"✅ Processed {len(edit_output.get('edited_results', {}).get('main_list', []))} restaurants from scraped content")

            return {
                **x,
                "raw_query": raw_query,
                "edited_results": edit_output.get("edited_results", {"main_list": []}),
                "follow_up_queries": edit_output.get("follow_up_queries", [])
            }

        except Exception as e:
            logger.error(f"❌ Error in edit step: {e}")
            dump_chain_state("edit_error", {"error": str(e), "available_keys": list(x.keys())}, error=e)
            return {
                **x,
                "raw_query": x.get("raw_query", x.get("query", "")),
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

            return {
                **x, 
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
                "enhanced_results": enhanced_results
            }

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
                "raw_query": x.get("raw_query", x.get("query", "")),  # Preserve raw query
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
            logger.info(f"Credits used: {stats.get('credits_used', 0)}")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error logging Firecrawl usage: {e}")

    @log_function_call
    def process_query(self, user_query: str, user_preferences: dict = None) -> dict:
        """
        Process a restaurant query through the complete pipeline.

        Args:
            user_query: The user's restaurant request
            user_preferences: Optional user preferences dict

        Returns:
            Dict with telegram_formatted_text and other results
        """

        # Generate trace ID for debugging
        trace_id = f"query_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                logger.info(f"🚀 STARTING RECOMMENDATION PIPELINE")
                logger.info(f"Query: {user_query}")

                # Prepare input data (UPDATED to include raw query from the start)
                input_data = {
                    "query": user_query,
                    "raw_query": user_query,  # Add raw query from the beginning
                    "user_preferences": user_preferences or {}
                }

                # Execute the chain
                result = self.chain.invoke(input_data)

                # Log completion
                content_source = result.get("content_source", "unknown")
                logger.info("✅ PIPELINE COMPLETE")
                logger.info(f"📊 Content source: {content_source}")

                # Final usage summary (only if we used scraping)
                if content_source == "web_search":
                    self._log_firecrawl_usage()

                # Save process record (simplified - just log it)
                logger.info(f"📊 Process completed: {user_query} → {result.get('destination', 'Unknown')} → {content_source}")

                # Could save to database here if needed:
                # process_record = {
                #     "query": user_query,
                #     "destination": result.get("destination", "Unknown"),
                #     "content_source": content_source,
                #     "trace_id": trace_id,
                #     "timestamp": time.time()
                # }

                # Extract results with correct key names
                telegram_text = result.get("telegram_formatted_text", 
                                         "Sorry, no recommendations found.")

                enhanced_results = result.get("enhanced_results", {})
                main_list = enhanced_results.get("main_list", [])

                logger.info(f"📊 Final result: {len(main_list)} restaurants for {result.get('destination', 'Unknown')}")
                logger.info(f"📊 Source: {content_source}")

                # Return with correct key names that telegram_bot.py expects
                return {
                    "telegram_formatted_text": telegram_text,
                    "enhanced_results": enhanced_results,
                    "main_list": main_list,
                    "destination": result.get("destination"),
                    "content_source": content_source,
                    "raw_query": result.get("raw_query", user_query),  # Include raw query in response
                    "firecrawl_stats": self.scraper.get_stats() if content_source == "web_search" else {}
                }

            except Exception as e:
                logger.error(f"❌ Error in chain execution: {e}")
                dump_chain_state("process_query_error", {"query": user_query}, error=e)

                # Log usage even on error
                try:
                    self._log_firecrawl_usage()
                except:
                    pass

                return {
                    "main_list": [],
                    "telegram_formatted_text": "Sorry, there was an error processing your request.",
                    "raw_query": user_query,  # Include raw query even on error
                    "firecrawl_stats": self.scraper.get_stats()
                }