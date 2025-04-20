# agents/orchestrator.py
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.runnables import RunnableSequence, RunnableLambda
import time
from utils.database import save_data

class RestaurantRecommendationOrchestrator:
    def __init__(self, config):
        # Import agents
        from agents.query_analyzer import QueryAnalyzer
        from agents.search_agent import BraveSearchAgent
        from agents.scraper import WebScraper
        from agents.list_analyzer import ListAnalyzer
        # We'll add more agents as we implement them

        # Initialize agents
        self.query_analyzer = QueryAnalyzer(config)
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)
        self.list_analyzer = ListAnalyzer(config)
        # We'll add more agents as we implement them

        self.config = config

    def process_query(self, user_query):
        """
        Process a user query and return restaurant recommendations

        Args:
            user_query (str): The user's query about restaurant recommendations

        Returns:
            dict: The final recommendations and analysis
        """
        # Create a unique trace ID for this request
        trace_id = f"restaurant_rec_{int(time.time())}"

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Step 1: Analyze the query
            print(f"Step 1: Analyzing query: {user_query}")
            query_analysis = self.query_analyzer.analyze(user_query)

            # Step 2: Search for restaurants
            print(f"Step 2: Searching with queries: {query_analysis['search_queries']}")
            search_results = self.search_agent.search(query_analysis['search_queries'])

            # Step 3: Scrape detailed content from search results
            print(f"Step 3: Scraping content from {len(search_results)} search results")
            enriched_results = self.scraper.scrape_search_results(search_results)

            # Step 4: Analyze and rank the restaurants
            print(f"Step 4: Analyzing search results to find best restaurants")
            recommendations = self.list_analyzer.analyze(
                enriched_results,
                query_analysis.get('user_preferences', ''),
                query_analysis.get('keywords_for_analysis', [])
            )

            # Later we'll add more steps (editor, follow-up searches, etc.)

            # Save the complete process and results to database
            process_record = {
                "query": user_query,
                "query_analysis": query_analysis,
                "recommendations": recommendations,
                "timestamp": time.time(),
                "trace_id": trace_id
            }
            save_data(
                self.config.DB_TABLE_PROCESSES,
                process_record,
                self.config
            )

            return recommendations

    # We'll add more functions as we implement more agents