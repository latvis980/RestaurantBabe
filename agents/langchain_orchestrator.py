# agents/langchain_orchestrator.py
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
import re
from html import escape
from utils.database import save_data
from utils.debug_utils import dump_chain_state, log_function_call
from utils.async_utils import sync_to_async
import asyncio
import logging

# Create logger. 
logger = logging.getLogger("restaurant-recommender.orchestrator")

# Move the sanitize_html_for_telegram function here instead of importing from telegram_bot
def sanitize_html_for_telegram(text):
    """Clean HTML text to ensure it's safe for Telegram API"""
    if not text:
        return ""

    # First, escape any unescaped text content
    # Do this for any text between > and <
    def escape_text_content(match):
        content = match.group(1)
        # Only escape if it's not already escaped
        if '&' in content and ('&lt;' in content or '&gt;' in content or '&amp;' in content):
            return '>' + content + '<'
        return '>' + escape(content) + '<'

    # Fix unescaped content between tags
    text = re.sub(r'>([^<]+)<', escape_text_content, text)

    # Ensure all tags are properly closed
    # Stack to track open tags
    stack = []
    result = []
    i = 0

    allowed_tags = {'b', 'i', 'u', 's', 'a', 'code', 'pre'}

    while i < len(text):
        if text[i] == '<':
            # Find the end of the tag
            end = text.find('>', i)
            if end == -1:
                # No closing bracket, treat as plain text
                result.append('&lt;')
                i += 1
                continue

            tag_content = text[i+1:end]
            if tag_content.startswith('/'):
                # Closing tag
                tag_name = tag_content[1:].split()[0].lower()
                if tag_name in allowed_tags:
                    if stack and stack[-1] == tag_name:
                        stack.pop()
                        result.append(text[i:end+1])
                    else:
                        # Mismatched closing tag, just add as text
                        result.append(escape(text[i:end+1]))
                else:
                    # Not an allowed tag
                    result.append(escape(text[i:end+1]))
            else:
                # Opening tag
                tag_parts = tag_content.split(None, 1)
                tag_name = tag_parts[0].lower()
                if tag_name in allowed_tags:
                    if tag_name == 'a':
                        # Special handling for links
                        if len(tag_parts) > 1 and 'href=' in tag_parts[1]:
                            result.append(text[i:end+1])
                            stack.append(tag_name)
                        else:
                            result.append(escape(text[i:end+1]))
                    else:
                        result.append(text[i:end+1])
                        stack.append(tag_name)
                else:
                    # Not an allowed tag
                    result.append(escape(text[i:end+1]))
            i = end + 1
        else:
            result.append(text[i])
            i += 1

    # Close any unclosed tags
    while stack:
        tag = stack.pop()
        result.append(f'</{tag}>')

    return ''.join(result)

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.scraper import WebScraper
        from agents.list_analyzer import ListAnalyzer
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent
        from agents.translator import TranslatorAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.list_analyzer = ListAnalyzer(config)
        self.editor_agent = EditorAgent(config)
        self.follow_up_search_agent = FollowUpSearchAgent(config)
        self.translator = TranslatorAgent(config)

        self.config = config

        # Create runnable lambdas for each step
        self.analyze_query = RunnableLambda(
            lambda x: {
                **self.query_analyzer.analyze(x["query"]),
                "query": x["query"]  # Keep the original query in the chain
            },
            name="analyze_query"
        )

        self.search = RunnableLambda(
            lambda x: {
                **x,
                "search_results": self.search_agent.search(x["search_queries"])
            },
            name="search"
        )

        # Define a wrapper that properly converts async to sync
        def scrape_helper(x):
            """Wrapper to handle async scraping in the LangChain"""
            import asyncio
            import concurrent.futures

            # Get the search results from the chain data
            search_results = x.get("search_results", [])
            logger.info(f"Scrape helper running for {len(search_results)} results")

            # Define a function that runs in its own thread with a fresh event loop
            def run_in_new_event_loop():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async function in this new loop
                try:
                    return loop.run_until_complete(self.scraper.filter_and_scrape_results(search_results))
                finally:
                    loop.close()

            # Execute the function in a thread
            with concurrent.futures.ThreadPoolExecutor() as pool:
                enriched_results = pool.submit(run_in_new_event_loop).result()

            # Return the results with the original data
            logger.info(f"Scrape completed with {len(enriched_results)} enriched results")
            return {**x, "enriched_results": enriched_results}

        # Assign the helper to the scrape step
        self.scrape = RunnableLambda(scrape_helper, name="scrape")

        async def analyze_results_with_debug_async(x):
            try:
                # Debug log before analysis
                dump_chain_state("pre_analyze_results", {
                    "enriched_results_count": len(x.get("enriched_results", [])),
                    "keywords": x.get("keywords_for_analysis", []),
                    "primary_params": x.get("primary_search_parameters", []),
                    "secondary_params": x.get("secondary_filter_parameters", []),
                    "destination": x.get("destination", "Unknown")  # Log the destination
                })

                # Execute list analyzer with the destination - AWAIT the async call
                recommendations = await self.list_analyzer.analyze(
                    search_results=x["enriched_results"],  # Changed parameter name
                    keywords_for_analysis=x.get("keywords_for_analysis", []),  # Changed parameter name
                    primary_search_parameters=x.get("primary_search_parameters", []),  # Changed parameter name
                    secondary_filter_parameters=x.get("secondary_filter_parameters", []),  # Changed parameter name
                    destination=x.get("destination")  # Pass the destination!
                )

                # Debug log after analysis
                dump_chain_state("post_analyze_results", {
                    "recommendations_keys": list(recommendations.keys() if recommendations else {}),
                    "recommendations": recommendations
                })

                # Convert to single list structure - combine all restaurants into main_list
                if isinstance(recommendations, dict):
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

                    # Return standardized structure with only main_list
                    standardized = {
                        "main_list": all_restaurants
                    }
                else:
                    # Initialize empty structure
                    standardized = {
                        "main_list": []
                    }

                # Return the result
                return {**x, "recommendations": standardized}
            except Exception as e:
                print(f"Error in analyze_results: {e}")
                # Log the error and return a fallback
                dump_chain_state("analyze_results_error", x, error=e)
                return {
                    **x,
                    "recommendations": {
                        "main_list": []
                    }
                }

        # And modify the RunnableLambda to handle async:
        def async_analyze_results(x):
            """Wrapper to run async analyze_results in the chain"""
            import asyncio
            import concurrent.futures

            # Define a function that runs in its own thread with a fresh event loop
            def run_in_new_event_loop():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async function in this new loop
                try:
                    return loop.run_until_complete(analyze_results_with_debug_async(x))
                finally:
                    loop.close()

            # Execute the function in a thread
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(run_in_new_event_loop).result()

            return result

        # Replace this line in the orchestrator:
        self.analyze_results = RunnableLambda(
            async_analyze_results,  # Use the new async wrapper
            name="analyze_results"
        )

        # Improved editor step with debug logging
        def editor_step(x):
            try:
                # Debug before edit
                dump_chain_state("pre_edit", {
                    "recommendations_keys": list(x.get("recommendations", {}).keys()),
                    "query": x.get("query", "")
                })

                # Get recommendations
                recommendations = x.get("recommendations", {})

                # Execute editor
                formatted_results = self.editor_agent.edit(recommendations, x["query"])

                # Debug after edit
                dump_chain_state("post_edit", {
                    "formatted_results_keys": list(formatted_results.keys() if formatted_results else {}),
                    "formatted_results": formatted_results
                })

                # Ensure proper structure is returned
                return {**x, "formatted_recommendations": formatted_results}
            except Exception as e:
                print(f"Error in editor step: {e}")
                # Log the error and return a fallback
                dump_chain_state("editor_error", x, error=e)
                return {
                    **x,
                    "formatted_recommendations": {
                        "formatted_recommendations": x.get("recommendations", {})
                    }
                }

        self.edit = RunnableLambda(
            editor_step,
            name="edit"
        )

        # Improved follow-up search step
        def follow_up_step(x):
            try:
                # Debug before follow_up
                dump_chain_state("pre_follow_up", {
                    "formatted_recommendations_keys": list(x.get("formatted_recommendations", {}).keys())
                })

                # Get formatted recommendations
                formatted_recs = x.get("formatted_recommendations", {})

                # Extract the actual recommendations
                if "formatted_recommendations" in formatted_recs:
                    actual_recs = formatted_recs.get("formatted_recommendations", {})
                else:
                    actual_recs = formatted_recs

                # Get follow up queries focusing on mandatory fields
                follow_up_queries = formatted_recs.get("follow_up_queries", [])

                # Get secondary filter parameters from original query analysis
                secondary_params = x.get("secondary_filter_parameters", [])

                # Execute follow up search
                enhanced_recommendations = self.follow_up_search_agent.perform_follow_up_searches(
                    actual_recs,
                    follow_up_queries,
                    secondary_params
                )

                # Debug after follow_up
                dump_chain_state("post_follow_up", {
                    "enhanced_recommendations_keys": list(enhanced_recommendations.keys() if enhanced_recommendations else {})
                })

                # Return result
                return {**x, "enhanced_recommendations": enhanced_recommendations}
            except Exception as e:
                print(f"Error in follow-up step: {e}")
                # Log the error and return a fallback
                dump_chain_state("follow_up_error", x, error=e)
                return {
                    **x,
                    "enhanced_recommendations": x.get("formatted_recommendations", {}).get("formatted_recommendations", {})
                }

        self.follow_up_search = RunnableLambda(
            follow_up_step,
            name="follow_up_search"
        )

        # HTML extraction - updated to handle single list
        def extract_html_step(x):
            try:
                # Debug before html extraction
                dump_chain_state("pre_extract_html", {
                    "enhanced_recommendations_keys": list(x.get("enhanced_recommendations", {}).keys())
                })

                # Get the recommendations - THIS IS THE KEY FIX
                enhanced_recommendations = x.get("enhanced_recommendations", {})

                # Log what we're actually getting
                logger.info(f"Enhanced recommendations structure: {list(enhanced_recommendations.keys())}")
                logger.info(f"Main list count: {len(enhanced_recommendations.get('main_list', []))}")

                # Create HTML output with the correct structure
                telegram_text = self._create_detailed_html(enhanced_recommendations)

                # Debug after html extraction
                dump_chain_state("post_extract_html", {
                    "telegram_text_length": len(telegram_text) if telegram_text else 0,
                    "telegram_text_preview": telegram_text[:200] if telegram_text else None
                })

                # Return result with proper structure
                return {
                    **x, 
                    "telegram_formatted_text": telegram_text,
                    "final_recommendations": enhanced_recommendations  # Keep the original data for debugging
                }
            except Exception as e:
                logger.error(f"Error in extract_html: {e}")
                # Log the error and return a fallback
                dump_chain_state("extract_html_error", x, error=e)
                return {
                    **x, 
                    "telegram_formatted_text": "<b>Извините, не удалось найти рестораны по вашему запросу.</b>"
                }

        # Extract HTML with debugging
        self.extract_html = RunnableLambda(
            extract_html_step,
            name="extract_html"
        )

        # Create the sequence WITHOUT translation
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.search,
                self.scrape,
                self.analyze_results,
                self.edit,
                self.follow_up_search,
                self.extract_html,  # Extract HTML without translating
            ],
            last=RunnableLambda(lambda x: x),  # Pass through everything
            name="restaurant_recommendation_chain"
        )

    def _extract_user_preferences(self, query):
        """
        Extract user preferences from query if they're provided

        Args:
            query (str): User query which might contain preferences

        Returns:
            tuple: (cleaned_query, preference_list)
        """
        # Check if preferences are included in the query
        preference_marker = "User preferences:"

        if preference_marker in query:
            # Split the query to extract preferences
            parts = query.split(preference_marker)
            clean_query = parts[0].strip()

            # Get preferences as a list
            if len(parts) > 1:
                preferences_text = parts[1].strip()
                preferences = [p.strip() for p in preferences_text.split(',') if p.strip()]
                return clean_query, preferences

        # No preferences found
        return query, []

    @log_function_call
    def _create_detailed_html(self, recommendations):
        """Create elegant, emoji-light HTML output for Telegram - single list only."""
        try:
            # ――― Debug input ―――
            logger.info(f"Creating HTML for recommendations: {list(recommendations.keys()) if isinstance(recommendations, dict) else type(recommendations)}")

            # ――― Get restaurant list ―――
            main_list = recommendations.get("main_list", [])

            # Check for legacy format
            if not main_list and "recommended" in recommendations:
                main_list = recommendations.get("recommended", [])

            # Debug log count
            logger.info(f"Main list count: {len(main_list)}")

            # ――― If no restaurants found, return early ―――
            if not main_list:
                logger.warning("No restaurants found in recommendations")
                return "<b>No restaurants found for your query.</b>"

            # ――― Build HTML ―――
            html_parts = []
            html_parts.append("<b>Recommended Restaurants</b>\n\n")

            def format_restaurant_block(restaurants):
                """Format a block of restaurants for HTML output"""
                block_parts = []

                for i, restaurant in enumerate(restaurants, 1):
                    # Safely get restaurant details
                    name = str(restaurant.get("name", "Restaurant")).strip()
                    addr = str(restaurant.get("address", "Address unavailable")).strip()
                    desc = str(restaurant.get("description", "")).strip()
                    price = str(restaurant.get("price_range", "")).strip()
                    dishes = restaurant.get("recommended_dishes", [])
                    sources = restaurant.get("sources", [])

                    # Debug individual restaurant
                    logger.info(f"Formatting restaurant {i}: {name}")

                    # Format restaurant entry with proper escaping
                    block_parts.append(f"<b>{i}. {escape(name)}</b>\n")

                    # Handle address
                    if addr and addr != "Address unavailable":
                        # Check if it's already a formatted link
                        if "<a href=" in addr:
                            # Validate the link format
                            link_match = re.search(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', addr)
                            if link_match:
                                url, address_text = link_match.groups()
                                # Properly format the link for Telegram
                                block_parts.append(f'📍 <a href="{url}">{escape(address_text)}</a>\n')
                            else:
                                # Invalid link format, treat as plain text
                                block_parts.append(f"📍 {escape(addr)}\n")
                        else:
                            # Plain text address
                            block_parts.append(f"📍 {escape(addr)}\n")
                    else:
                        block_parts.append("📍 Address unavailable\n")

                    # Add description
                    if desc:
                        block_parts.append(f"{escape(desc)}\n")
                    else:
                        block_parts.append("Description unavailable\n")

                    # Add signature dishes
                    if dishes and isinstance(dishes, list):
                        # Filter and escape dishes
                        valid_dishes = []
                        for d in dishes:
                            if d and str(d).strip():
                                valid_dishes.append(escape(str(d).strip()))
                        if valid_dishes:
                            dishes_str = ", ".join(valid_dishes[:3])
                            block_parts.append(f"<i>Signature dishes:</i> {dishes_str}\n")

                    # Add sources
                    if sources and isinstance(sources, list):
                        # Filter and escape sources
                        valid_sources = []
                        seen = set()
                        for s in sources:
                            if s and str(s).strip():
                                source_str = escape(str(s).strip())
                                if source_str.lower() not in seen:
                                    valid_sources.append(source_str)
                                    seen.add(source_str.lower())
                        if valid_sources:
                            sources_str = ", ".join(valid_sources[:3])
                            block_parts.append(f"<i>Recommended by:</i> {sources_str}\n")

                    # Add price range
                    if price:
                        block_parts.append(f"<i>Price range:</i> {escape(price)}\n")

                    # Add spacing after restaurant
                    block_parts.append("\n")

                return ''.join(block_parts)

            # Add all restaurants in main list
            if main_list:
                logger.info(f"Formatting {len(main_list)} restaurants")
                html_parts.append(format_restaurant_block(main_list))

            # Add footer
            html_parts.append("<i>Recommendations compiled from reputable critic and guide sources.</i>")

            # Join all parts
            html = ''.join(html_parts)

            # Apply final sanitization
            html = sanitize_html_for_telegram(html)

            # Respect Telegram's message length limit (4096 characters)
            if len(html) > 4096:
                html = html[:4093] + "…"
                logger.info(f"Truncated HTML to {len(html)} characters")

            logger.info(f"Final HTML length: {len(html)}")
            logger.info(f"Successfully created HTML for {len(main_list)} restaurants")

            return html

        except Exception as e:
            logger.error(f"Error in _create_detailed_html: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Absolute fallback
            return "<b>Sorry, we couldn't format the restaurant list due to an error.</b>"

    @log_function_call
    def process_query(self, user_query, standing_prefs=None):
        """
        Process a user query using the LangChain sequence

        Args:
            user_query (str): The user's query about restaurant recommendations
            standing_prefs (list, optional): List of user's standing preferences

        Returns:
            dict: The formatted recommendations for Telegram
        """
        # Extract user preferences if included in the query
        clean_query, explicit_prefs = self._extract_user_preferences(user_query)

        # Combine explicit preferences from query with standing preferences
        user_preferences = list(set(explicit_prefs + (standing_prefs or [])))

        # Create a unique trace ID for this request
        trace_id = f"restaurant_rec_{int(time.time())}"

        # Use LangSmith tracing
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # Create initial input with preferences
                input_data = {"query": clean_query, "user_preferences": user_preferences}

                # Execute the chain with our input data
                result = self.chain.invoke(input_data)

                # Log completion and dump final state
                dump_chain_state("process_query_complete", {
                    "result_keys": list(result.keys()),
                    "has_recommendations": "enhanced_recommendations" in result,
                    "has_telegram_text": "telegram_formatted_text" in result
                })

                # Save process results to database
                process_record = {
                    "query": user_query,
                    "trace_id": trace_id,
                    "timestamp": time.time()
                }

                try:
                    save_data(
                        self.config.DB_TABLE_PROCESSES,
                        process_record,
                        self.config
                    )
                except Exception as db_error:
                    print(f"Error saving to database: {db_error}")

                # Get the telegram text
                telegram_text = result.get("telegram_formatted_text", 
                                         "<b>Извините, не удалось найти рестораны по вашему запросу.</b>")

                # Get the enhanced recommendations
                enhanced_recommendations = result.get("enhanced_recommendations", {})

                # Extract main_list from enhanced_recommendations
                main_list = enhanced_recommendations.get("main_list", [])

                # ADD THIS DEBUG LOG
                logger.info(f"Final result - Main list: {len(main_list)} restaurants")

                # Return the complete data structure (no hidden_gems)
                return {
                    "telegram_text": telegram_text,
                    "enhanced_recommendations": enhanced_recommendations,  # Include the full structure
                    "main_list": main_list,
                    "destination": result.get("destination")
                }

            except Exception as e:
                logger.error(f"Error in chain execution: {e}")
                # Log the error
                dump_chain_state("process_query_error", {"query": user_query}, error=e)

                # Return a basic error message
                return {
                    "main_list": [
                        {
                            "name": "Ошибка обработки запроса",
                            "description": "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз позже."
                        }
                    ],
                    "telegram_text": "<b>Извините, произошла ошибка при обработке вашего запроса.</b>"
                }