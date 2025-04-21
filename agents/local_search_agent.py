# agents/local_search_agent.py
import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from urllib.parse import urlparse
from utils.database import find_data, find_all_data
from langchain_core.tracers.context import tracing_v2_enabled

class LocalSourceSearchAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Import here to avoid circular imports
        from agents.scraper import WebScraper
        self.scraper = WebScraper(config)

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        ]

    def search_local_sources(self, location, search_queries, local_language=None):
        """
        Search local reputable sources for restaurant recommendations

        Args:
            location (str): City or location name
            search_queries (list): List of search queries
            local_language (str, optional): Local language code

        Returns:
            list: Search results from local sources
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Get local sources for this location
            local_sources = self._get_local_sources(location)

            if not local_sources:
                self.logger.warning(f"No local sources found for {location}")
                return []

            self.logger.info(f"Found {len(local_sources)} local sources for {location}")

            # Prepare results list
            local_results = []

            # Search each local source with each query
            for source in local_sources:
                source_name = source.get("name", "Unknown Source")
                source_url = source.get("url", "")

                if not source_url:
                    continue

                for query in search_queries:
                    try:
                        # Create site-specific search
                        domain = urlparse(source_url).netloc
                        site_results = self._search_site(domain, query)

                        # Add source information to results
                        for result in site_results:
                            result["source_name"] = source_name
                            result["source_domain"] = domain
                            result["is_local_source"] = True
                            result["is_reputable"] = True  # Pre-validate as reputable
                            local_results.append(result)

                        # Respect rate limits
                        time.sleep(random.uniform(1.0, 2.0))
                    except Exception as e:
                        self.logger.error(f"Error searching {source_name}: {e}")

            # Scrape the content for each result
            enriched_results = self.scraper.scrape_search_results(local_results)

            # Mark these as high priority local sources
            for result in enriched_results:
                result["is_local_source"] = True
                result["priority"] = "high"

            return enriched_results

    def _get_local_sources(self, location):
        """Get local sources for a specific location from database"""
        # Create sanitized table name for city-specific sources
        city_table_name = f"sources_{location.lower().replace(' ', '_').replace('-', '_')}"

        # Try to find sources in the city-specific table first
        sources = find_all_data(
            city_table_name, 
            {"city": location}, 
            self.config
        )

        if sources and len(sources) > 0:
            # If there's a sources array in the first result, use that
            if "sources" in sources[0]:
                return sources[0].get("sources", [])
            return sources

        # Try the general sources table as fallback
        general_result = find_data(
            self.config.DB_TABLE_SOURCES,
            {"location": location},
            self.config
        )

        if general_result and "sources" in general_result:
            return general_result.get("sources", [])

        return []

    def _search_site(self, domain, query):
        """Search a specific site using Brave Search site: operator"""
        from agents.search_agent import BraveSearchAgent

        # Initialize the search agent
        search_agent = BraveSearchAgent(self.config)

        # Create site-specific query
        site_query = f"site:{domain} {query}"

        # Execute search
        results = search_agent._execute_search(site_query)

        # Filter results
        filtered_results = search_agent._filter_results(results)

        return filtered_results