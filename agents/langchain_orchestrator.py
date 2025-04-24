# agents/langchain_orchestrator.py
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
import json
from utils.database import save_data, ensure_city_table

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

        self.analyze_results = RunnableLambda(
            lambda x: {
                **x,
                "recommendations": self.list_analyzer.analyze(
                    x["enriched_results"],
                    x.get("keywords_for_analysis", []),
                    x.get("primary_search_parameters", []),
                    x.get("secondary_filter_parameters", [])
                )
            },
            name="analyze_results"
        )

        # Improved editor step with debug logging
        self.edit = RunnableLambda(
            lambda x: {
                print(f"Editor step received recommendations structure: {list(x.get('recommendations', {}).keys())}")
                return {
                    **x,
                    "formatted_recommendations": self._safe_edit(
                        x.get("recommendations", {}),
                        x["query"]
                    )
                }
            },
            name="edit"
        )

        # Improved follow-up search step
        self.follow_up_search = RunnableLambda(
            lambda x: {
                print(f"Follow-up search received formatted_recommendations structure: {list(x.get('formatted_recommendations', {}).keys())}")
                recs = x.get("formatted_recommendations", {})
                formatted_recs = recs.get("formatted_recommendations", recs)
                return {
                    **x,
                    "enhanced_recommendations": self._safe_follow_up_search(
                        formatted_recs,
                        recs.get("follow_up_queries", [])
                    )
                }
            },
            name="follow_up_search"
        )

        # Extract HTML without translation (for testing)
        self.extract_html = RunnableLambda(
            lambda x: {
                **x,
                "telegram_formatted_text": self._extract_html_output(x["enhanced_recommendations"])
            },
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
            last=self.extract_html,  # Extract HTML is the last step
            name="restaurant_recommendation_chain"
        )

    def _safe_edit(self, recommendations, query):
        """Safely apply the editor agent with proper error handling"""
        try:
            print(f"Starting editor agent with: {list(recommendations.keys() if isinstance(recommendations, dict) else [])}")

            # Handle structure conversion if needed
            if isinstance(recommendations, dict) and "recommended" in recommendations:
                # Convert old "recommended" to new "main_list"
                recommendations["main_list"] = recommendations.pop("recommended")
                print("Converted 'recommended' to 'main_list'")

            return self.editor_agent.edit(recommendations, query)
        except Exception as e:
            print(f"Error in edit step: {e}")
            # Return a basic structure as fallback
            if isinstance(recommendations, dict):
                if "recommended" in recommendations:
                    # Convert old format to new format
                    return {
                        "formatted_recommendations": {
                            "main_list": recommendations.get("recommended", []),
                            "hidden_gems": recommendations.get("hidden_gems", [])
                        },
                        "follow_up_queries": []
                    }
                elif "main_list" in recommendations:
                    return {
                        "formatted_recommendations": recommendations,
                        "follow_up_queries": []
                    }

            # Ultimate fallback
            return {
                "formatted_recommendations": {
                    "main_list": [],
                    "hidden_gems": []
                },
                "follow_up_queries": []
            }

    def _safe_follow_up_search(self, formatted_recommendations, follow_up_queries):
        """Safely apply follow-up search with proper error handling"""
        try:
            print(f"Starting follow-up search with: {list(formatted_recommendations.keys() if isinstance(formatted_recommendations, dict) else [])}")

            # Handle structure conversion if needed
            if isinstance(formatted_recommendations, dict) and "recommended" in formatted_recommendations:
                # Convert old "recommended" to new "main_list"
                formatted_recommendations["main_list"] = formatted_recommendations.pop("recommended")
                print("Converted 'recommended' to 'main_list' in follow-up search")

            return self.follow_up_search_agent.perform_follow_up_searches(
                formatted_recommendations, 
                follow_up_queries, 
                []  # No secondary parameters for simplicity
            )
        except Exception as e:
            print(f"Error in follow-up search: {e}")

            # Return the input as fallback
            if isinstance(formatted_recommendations, dict):
                # Make sure we're using the new structure
                if "recommended" in formatted_recommendations:
                    return {
                        "main_list": formatted_recommendations.get("recommended", []),
                        "hidden_gems": formatted_recommendations.get("hidden_gems", [])
                    }
                else:
                    return formatted_recommendations

            # Ultimate fallback
            return {
                "main_list": [],
                "hidden_gems": []
            }

    def _extract_html_output(self, recommendations):
        """Extract HTML output from recommendations without translation"""
        try:
            print(f"Extracting HTML from: {list(recommendations.keys() if isinstance(recommendations, dict) else [])}")

            # If html_formatted is directly in recommendations, use it
            if isinstance(recommendations, dict) and "html_formatted" in recommendations:
                return recommendations["html_formatted"]

            # Check if formatted_recommendations has html_formatted
            if isinstance(recommendations, dict) and "formatted_recommendations" in recommendations:
                if "html_formatted" in recommendations["formatted_recommendations"]:
                    return recommendations["formatted_recommendations"]["html_formatted"]

            # Otherwise create basic HTML output
            return self._create_basic_html(recommendations)
        except Exception as e:
            print(f"Error extracting HTML: {e}")
            return self._create_basic_html(recommendations)

    def _create_basic_html(self, recommendations):
        """Create basic HTML output if none is available"""
        try:
            html_output = "<b>🍽️ RECOMMENDED RESTAURANTS:</b>\n\n"

            # Get restaurant lists from appropriate keys
            main_list = []
            hidden_gems = []

            if isinstance(recommendations, dict):
                if "main_list" in recommendations:
                    main_list = recommendations["main_list"]
                elif "recommended" in recommendations:
                    main_list = recommendations["recommended"]

                if "hidden_gems" in recommendations:
                    hidden_gems = recommendations["hidden_gems"]

            # Format main list
            if main_list:
                for i, restaurant in enumerate(main_list, 1):
                    name = restaurant.get("name", "Restaurant")
                    html_output += f"<b>{i}. {name}</b>\n"

                    if "address" in restaurant:
                        html_output += f"📍 {restaurant['address']}\n"

                    if "description" in restaurant:
                        html_output += f"{restaurant['description']}\n"

                    sources = None
                    if "recommended_by" in restaurant:
                        sources = restaurant["recommended_by"]
                    elif "sources" in restaurant:
                        sources = restaurant["sources"]

                    if sources:
                        if isinstance(sources, list):
                            sources_text = ", ".join(sources[:3])
                            html_output += f"<i>✅ Recommended by: {sources_text}</i>\n"
                        else:
                            html_output += f"<i>✅ Recommended by: {sources}</i>\n"

                    html_output += "\n"
            else:
                html_output += "Sorry, no recommended restaurants found.\n\n"

            # Format hidden gems
            if hidden_gems:
                html_output += "<b>💎 HIDDEN GEMS:</b>\n\n"

                for i, restaurant in enumerate(hidden_gems, 1):
                    name = restaurant.get("name", "Restaurant")
                    html_output += f"<b>{i}. {name}</b>\n"

                    if "address" in restaurant:
                        html_output += f"📍 {restaurant['address']}\n"

                    if "description" in restaurant:
                        html_output += f"{restaurant['description']}\n"

                    sources = None
                    if "recommended_by" in restaurant:
                        sources = restaurant["recommended_by"]
                    elif "sources" in restaurant:
                        sources = restaurant["sources"]

                    if sources:
                        if isinstance(sources, list):
                            sources_text = ", ".join(sources[:3])
                            html_output += f"<i>✅ Recommended by: {sources_text}</i>\n"
                        else:
                            html_output += f"<i>✅ Recommended by: {sources}</i>\n"

                    html_output += "\n"

            # Add footer
            html_output += "<i>Recommendations based on analysis of expert sources.</i>"

            return html_output
        except Exception as e:
            print(f"Error creating basic HTML: {e}")
            return "<b>Sorry, couldn't format restaurant recommendations properly.</b>"

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

    def process_query(self, user_query):
        """
        Process a user query using the LangChain sequence

        Args:
            user_query (str): The user's query about restaurant recommendations

        Returns:
            str: The final formatted text for Telegram in English
        """
        # Create a unique trace ID for this request
        trace_id = f"restaurant_rec_{int(time.time())}"

        # Use LangSmith tracing
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # Execute the chain
                result = self.chain.invoke({"query": user_query})

                # Log performance metrics
                print(f"Regular search returned {len(result.get('search_results', []))} results")
                print(f"Local search returned {len(result.get('local_search_results', []))} results")
                print(f"Combined search returned {len(result.get('combined_results', []))} results")

                # Save process results to database
                process_record = {
                    "query": user_query,
                    "trace_id": trace_id,
                    "timestamp": time.time(),
                    "result": result.get("enhanced_recommendations", {})
                }

                try:
                    save_data(
                        self.config.DB_TABLE_PROCESSES,
                        process_record,
                        self.config
                    )
                except Exception as db_error:
                    print(f"Error saving to database: {db_error}")

                # Return the formatted text for Telegram (in English for now)
                telegram_text = result.get("telegram_formatted_text", "Sorry, couldn't find restaurant recommendations.")

                # For backwards compatibility with telegram_bot.py
                # We need to also return the structured data
                enhanced_recommendations = result.get("enhanced_recommendations", {})

                # Telegram bot expects a dict with recommended and hidden_gems
                if isinstance(enhanced_recommendations, dict):
                    if "main_list" in enhanced_recommendations:
                        final_result = {
                            "recommended": enhanced_recommendations["main_list"],
                            "hidden_gems": enhanced_recommendations.get("hidden_gems", []),
                            "telegram_text": telegram_text  # Add the formatted text
                        }
                        return final_result

                # Fallback if the structure is unexpected
                return {
                    "recommended": [],
                    "hidden_gems": [],
                    "telegram_text": telegram_text
                }

            except Exception as e:
                print(f"Error in chain execution: {e}")
                # Return a basic error message
                return {
                    "recommended": [
                        {
                            "name": "Error Processing Request",
                            "description": "We encountered an error while processing your request. Please try again later."
                        }
                    ],
                    "hidden_gems": [],
                    "telegram_text": "<b>Sorry, an error occurred while processing your request.</b>"
                }