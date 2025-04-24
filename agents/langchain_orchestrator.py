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
                "query": x["query"]  # ✅ Keep the original query in the chain
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
                    x.get("primary_search_parameters", []),  # Pass primary parameters
                    x.get("secondary_filter_parameters", [])  # Pass secondary parameters
                )
            },
            name="analyze_results"
        )

        self.edit = RunnableLambda(
            lambda x: {
                **x,
                "formatted_recommendations": self.editor_agent.edit(
                    x["recommendations"],
                    x["query"]  # Pass the original query
                )
            },
            name="edit"
        )

        self.follow_up_search = RunnableLambda(
            lambda x: {
                **x,
                "enhanced_recommendations": self.follow_up_search_agent.perform_follow_up_searches(
                    x["formatted_recommendations"].get("formatted_recommendations", {}),
                    x["formatted_recommendations"].get("follow_up_queries", []),
                    x.get("secondary_filter_parameters", [])  # Pass secondary parameters
                )
            },
            name="follow_up_search"
        )

        self.translate = RunnableLambda(
            lambda x: {
                **x,
                "translated_recommendations": self._safe_translate(x["enhanced_recommendations"])
            },
            name="translate"
        )

        # Create the complete sequence including local search
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
                # Comment out the translate step or remove it temporarily
                # self.translate,
            ],
            # If you removed translate from middle, adjust 'last' to be the last element in middle
            last=self.follow_up_search,  # Changed from self.translate to self.follow_up_search
            name="restaurant_recommendation_chain"
        )

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

    def _safe_translate(self, recommendations):
        """Safely translate recommendations with error handling"""
        try:
            # Ensure the recommendations are properly formatted before translation
            if not recommendations:
                return {"recommended": [], "hidden_gems": []}

            # Make sure we have the expected structure
            if "recommended" not in recommendations or "hidden_gems" not in recommendations:
                # Try to convert to expected format if possible
                if isinstance(recommendations, dict):
                    return self.translator.translate(recommendations)
                else:
                    return {"recommended": [], "hidden_gems": []}

            return self.translator.translate(recommendations)
        except Exception as e:
            print(f"Translation error: {e}")
            # Return the untranslated recommendations if translation fails
            return recommendations

    def process_query(self, user_query):
        """
        Process a user query using the LangChain sequence

        Args:
            user_query (str): The user's query about restaurant recommendations

        Returns:
            dict: The final translated recommendations
        """
        # Create a unique trace ID for this request
        trace_id = f"restaurant_rec_{int(time.time())}"

        # Use LangSmith tracing
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # Execute the chain
                result = self.chain.invoke({"query": user_query})

                # Log some performance metrics
                print(f"Regular search returned {len(result.get('search_results', []))} results")
                print(f"Local search returned {len(result.get('local_search_results', []))} results")
                print(f"Combined search returned {len(result.get('combined_results', []))} results")

                # Save the complete process and results to database
                process_record = {
                    "query": user_query,
                    "trace_id": trace_id,
                    "timestamp": time.time(),
                    "result": result.get("translated_recommendations", {})
                }

                try:
                    save_data(
                        self.config.DB_TABLE_PROCESSES,
                        process_record,
                        self.config
                    )
                except Exception as db_error:
                    print(f"Error saving to database: {db_error}")

                # Return just the translated recommendations
                return result.get("enhanced_recommendations", {})

            except Exception as e:
                print(f"Error in chain execution: {e}")
                # Return a basic structure as fallback
                return {
                    "recommended": [
                        {
                            "name": "Error Processing Request",
                            "description": "We encountered an error while processing your request. Please try again later.",
                        }
                    ],
                    "hidden_gems": []
                }