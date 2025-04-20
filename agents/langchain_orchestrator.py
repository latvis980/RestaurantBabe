# agents/langchain_orchestrator.py
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled
import time
from utils.database import save_to_mongodb

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
            lambda x: self.query_analyzer.analyze(x["query"]),
            name="analyze_query"
        )

        self.search = RunnableLambda(
            lambda x: {
                **x,
                "search_results": self.search_agent.search(x["search_queries"])
            },
            name="search"
        )

        self.scrape = RunnableLambda(
            lambda x: {
                **x,
                "enriched_results": self.scraper.scrape_search_results(x["search_results"])
            },
            name="scrape"
        )

        self.analyze_results = RunnableLambda(
            lambda x: {
                **x,
                "recommendations": self.list_analyzer.analyze(
                    x["enriched_results"],
                    x.get("user_preferences", ""),
                    x.get("keywords_for_analysis", [])
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
                    x["formatted_recommendations"].get("follow_up_queries", [])
                )
            },
            name="follow_up_search"
        )

        self.translate = RunnableLambda(
            lambda x: {
                **x,
                "translated_recommendations": self.translator.translate(x["enhanced_recommendations"])
            },
            name="translate"
        )

        # Create the complete sequence
        self.chain = RunnableSequence(
            first=self.analyze_query,
            middle=[
                self.search,
                self.scrape,
                self.analyze_results,
                self.edit,
                self.follow_up_search,
            ],
            last=self.translate,
            name="restaurant_recommendation_chain"
        )

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

        # Use LangSmith tracing - following current API (no run_id parameter)
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Execute the chain
            result = self.chain.invoke({"query": user_query})

            # Save the complete process and results to database
            process_record = {
                "query": user_query,
                "trace_id": trace_id,
                "timestamp": time.time(),
                "result": result.get("translated_recommendations", {})
            }
            save_to_mongodb(
                self.config.MONGODB_COLLECTION_PROCESSES,
                process_record,
                self.config
            )

            # Return just the translated recommendations
            return result.get("translated_recommendations", {})