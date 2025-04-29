# agents/langchain_orchestrator.py
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
from utils.database import save_data, ensure_city_table
from utils.debug_utils import dump_chain_state, log_function_call

class LangChainOrchestrator:
    def __init__(self, config):
        # Import agents
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.local_search_agent import LocalSourceSearchAgent
        from agents.scraper import WebScraper
        from agents.list_analyzer import ListAnalyzer
        from agents.editor_agent import EditorAgent
        from agents.follow_up_search_agent import FollowUpSearchAgent
        from agents.translator import TranslatorAgent

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.local_search_agent = LocalSourceSearchAgent(config)
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

        # Add a new step for local source search
        self.local_search = RunnableLambda(
            lambda x: {
                **x,
                "local_search_results": self._perform_local_search(
                    x["destination"],
                    x["search_queries"],
                    x.get("local_language")
                ) if not x.get("is_english_speaking", True) else []
            },
            name="local_search"
        )

        # Combine regular and local search results
        self.combine_results = RunnableLambda(
            lambda x: {
                **x,
                "combined_results": self._combine_search_results(
                    x["search_results"],
                    x.get("local_search_results", [])
                )
            },
            name="combine_results"
        )

        self.scrape = RunnableLambda(
            lambda x: {
                **x,
                "enriched_results": self.scraper.scrape_search_results(x["combined_results"])
            },
            name="scrape"
        )

        # Modify analyze_results to dump state
        def analyze_results_with_debug(x):
            try:
                # Debug log before analysis
                dump_chain_state("pre_analyze_results", {
                    "enriched_results_count": len(x.get("enriched_results", [])),
                    "keywords": x.get("keywords_for_analysis", []),
                    "primary_params": x.get("primary_search_parameters", []),
                    "secondary_params": x.get("secondary_filter_parameters", [])
                })

                # Execute list analyzer
                recommendations = self.list_analyzer.analyze(
                    x["enriched_results"],
                    x.get("keywords_for_analysis", []),
                    x.get("primary_search_parameters", []),
                    x.get("secondary_filter_parameters", [])
                )

                # Debug log after analysis
                dump_chain_state("post_analyze_results", {
                    "recommendations_keys": list(recommendations.keys() if recommendations else {}),
                    "recommendations": recommendations
                })

                # Explicitly standardize the structure
                if isinstance(recommendations, dict):
                    # Check if we have the old format (recommended/hidden_gems)
                    if "recommended" in recommendations:
                        # Convert to new format
                        standardized = {
                            "main_list": recommendations.get("recommended", []),
                            "hidden_gems": recommendations.get("hidden_gems", [])
                        }
                    else:
                        # Already in the right format or needs to be initialized
                        standardized = recommendations
                else:
                    # Initialize empty structure
                    standardized = {
                        "main_list": [],
                        "hidden_gems": []
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
                        "main_list": [],
                        "hidden_gems": []
                    }
                }

        self.analyze_results = RunnableLambda(
            analyze_results_with_debug,
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

        # Improved HTML extraction
        def extract_html_step(x):
            try:
                # Debug before html extraction
                dump_chain_state("pre_extract_html", {
                    "enhanced_recommendations_keys": list(x.get("enhanced_recommendations", {}).keys())
                })

                # Get the recommendations
                enhanced_recommendations = x.get("enhanced_recommendations", {})

                # Create HTML output
                telegram_text = self._create_detailed_html(enhanced_recommendations)

                # Debug after html extraction
                dump_chain_state("post_extract_html", {
                    "telegram_text_length": len(telegram_text) if telegram_text else 0,
                    "telegram_text_preview": telegram_text[:200] if telegram_text else None
                })

                # Return result
                return {**x, "telegram_formatted_text": telegram_text}
            except Exception as e:
                print(f"Error in extract_html: {e}")
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
                self.local_search,
                self.combine_results,
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

    def _perform_local_search(self, location, search_queries, local_language=None):
        """Perform local source search if we're in a non-English speaking location"""
        try:
            # Only perform local search if we have valid inputs
            if not location or not search_queries:
                return []

            # Perform the local source search
            local_results = self.local_search_agent.search_local_sources(
                location,
                search_queries,
                local_language
            )

            return local_results
        except Exception as e:
            print(f"Error in local search: {e}")
            return []

    def _combine_search_results(self, standard_results, local_results):
        """Combine standard search results with local source results"""
        # Start with local results as they're more valuable
        combined = local_results.copy() if local_results else []

        # Add regular results that aren't duplicates
        seen_urls = {result.get("url") for result in combined}

        for result in standard_results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)

        return combined

    @log_function_call
    def _create_detailed_html(self, recommendations):
        """Create elegant, emoji-light HTML output for Telegram."""
        try:
            # ――― Headings ―――
            html = "<b>Recommended Restaurants</b>\n\n"

            main_list   = recommendations.get("main_list", []) or recommendations.get("recommended", [])
            hidden_gems = recommendations.get("hidden_gems", [])

            def block(restaurants, title=None):
                nonlocal html
                if title:
                    html += f"<b>{title}</b>\n\n"
                for i, r in enumerate(restaurants, 1):
                    name = r.get("name", "Restaurant")
                    addr = r.get("address", "Address unavailable")
                    desc = r.get("description", "")
                    price = r.get("price_range", "")
                    dishes = ", ".join(r.get("recommended_dishes", [])[:3])
                    sources = ", ".join(sorted(set(r.get("sources", [])))[:3])

                    html += (
                        f"<b>{i}. {name}</b>\n"
                        f"📍 {addr}\n"              # keep the single map pin for scannability
                        f"{desc}\n"
                    )

                    if dishes:
                        html += f"<i>Signature dishes:</i> {dishes}\n"
                    if sources:
                        html += f"<i>Recommended by:</i> {sources}\n"
                    if price:
                        html += f"<i>Price range:</i> {price}\n"
                    html += "\n"

            block(main_list)
            if hidden_gems:
                block(hidden_gems, title="Hidden Gems")

            html += "<i>Recommendations compiled from reputable critic and guide sources.</i>"
            return html[:3997] + "…" if len(html) > 4000 else html

        except Exception as e:
            print("HTML format error:", e)
            return "<b>Sorry, we couldn't format the restaurant list.</b>"

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

                # Modify the analyze_query lambda to handle preferences
                # This is a new definition that should replace the existing one
                self.analyze_query = RunnableLambda(
                    lambda x: {
                        **self.query_analyzer.analyze(x["query"], x.get("user_preferences", [])),
                        "query": x["query"],  # Keep the original query in the chain
                        "user_preferences": x.get("user_preferences", [])  # Keep preferences
                    },
                    name="analyze_query"
                )

                # Re-create the chain with the updated analyze_query
                self.chain = RunnableSequence(
                    first=self.analyze_query,
                    middle=[
                        self.search,
                        self.local_search,
                        self.combine_results,
                        self.scrape,
                        self.analyze_results,
                        self.edit,
                        self.follow_up_search,
                        self.extract_html,
                    ],
                    last=RunnableLambda(lambda x: x),
                    name="restaurant_recommendation_chain"
                )

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

                # Check for different formats and standardize
                main_list = []
                hidden_gems = []

                if isinstance(enhanced_recommendations, dict):
                    # Direct access to main_list
                    if "main_list" in enhanced_recommendations:
                        main_list = enhanced_recommendations["main_list"]
                    # For backward compatibility with old format
                    elif "recommended" in enhanced_recommendations:
                        main_list = enhanced_recommendations["recommended"]

                    if "hidden_gems" in enhanced_recommendations:
                        hidden_gems = enhanced_recommendations["hidden_gems"]

                    # Check nested structure
                    if "formatted_recommendations" in enhanced_recommendations:
                        formatted_rec = enhanced_recommendations["formatted_recommendations"]
                        if isinstance(formatted_rec, dict):
                            if "main_list" in formatted_rec:
                                main_list = formatted_rec["main_list"]
                            elif "recommended" in formatted_rec:
                                main_list = formatted_rec["recommended"]

                            if "hidden_gems" in formatted_rec:
                                hidden_gems = formatted_rec["hidden_gems"]

                # Build final result dictionary with consistent naming using main_list
                final_result = {
                    "main_list": main_list,
                    "hidden_gems": hidden_gems,
                    "telegram_text": telegram_text
                }

                # Debug log the final result
                dump_chain_state("final_result", {
                    "main_list_count": len(main_list),
                    "hidden_gems_count": len(hidden_gems),
                    "telegram_text_length": len(telegram_text)
                })

                return final_result

            except Exception as e:
                print(f"Error in chain execution: {e}")
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
                    "hidden_gems": [],
                    "telegram_text": "<b>Извините, произошла ошибка при обработке вашего запроса.</b>"
                }