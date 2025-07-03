from __future__ import annotations

"""Follow‑up search agent

This version adds a thin Google Maps integration that is executed **before** we run the
traditional Web follow‑up search. The Maps look‑up supplies:

* `formatted_address` – turned into a clickable `<a>` element that opens Google Maps
  directly on the place.
* `opening_hours` – concatenated into a single string.
* `rating` – used as a **hard filter**. Places that score below ``MIN_ACCEPTABLE_RATING``
  (defaults to 4.5) are discarded and never reach the rest of the pipeline.

Everything else – Brave search look‑ups, scraping, global‑guide checks – remains exactly
as before.

Updated to return only main_list, no hidden_gems.
"""

import asyncio
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import googlemaps
from langchain_core.tracers.context import tracing_v2_enabled

from agents.search_agent import BraveSearchAgent
from agents.optimized_scraper import WebScraper
from utils.debug_utils import dump_chain_state

# ---------------------------------------------------------------------------
# Configuration constants – tweak here if you need different limits
# ---------------------------------------------------------------------------

MIN_ACCEPTABLE_RATING = 4.1           # rating threshold – <  ► rejected
MAX_RESULTS_PER_QUERY = 3             # courtesy cap for scraping
MAPS_FIELDS = [                       # fields we request from Place Details
    "url",
    "formatted_address",
    "rating",
    "opening_hours",
]


class FollowUpSearchAgent:
    """Adds missing data to restaurant candidates.

    The constructor requires a ``config`` object that MUST contain a
    ``GOOGLE_MAPS_API_KEY`` attribute.
    """

    def __init__(self, config: Any):
        self.config = config
        self.search_agent = BraveSearchAgent(config)
        self.scraper = WebScraper(config)

        # Modified: Access as attribute, not with get()
        api_key = config.GOOGLE_MAPS_API_KEY if hasattr(config, 'GOOGLE_MAPS_API_KEY') else None
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY missing in config – please supply it.")
        self.gmaps = googlemaps.Client(key=api_key)

    def _run_async_scraping(self, filtered_results):
        """Helper method to run async scraping in a new event loop"""
        def run_scraping():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.scraper.scrape_search_results(filtered_results)
                )
            finally:
                loop.close()

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_scraping).result()

    def perform_follow_up_searches(
        self,
        formatted_recommendations: Dict[str, List[Dict[str, Any]]],
        follow_up_queries: List[Dict[str, Any]],
        secondary_filter_parameters: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Enrich every restaurant in *formatted_recommendations*.

        Returns only main_list, no hidden_gems.
        Restaurants that do not meet the minimum Google rating are silently
        excluded from the output.
        """

        # Only process main_list
        main_list = formatted_recommendations.get("main_list", [])
        if not main_list:
            return {"main_list": []}

        enhanced_main_list = []

        for restaurant in main_list:
            try:
                enhanced_restaurant = self._enhance_single_restaurant(
                    restaurant, 
                    follow_up_queries, 
                    secondary_filter_parameters
                )

                # Only add restaurants that pass the rating filter
                if enhanced_restaurant:
                    enhanced_main_list.append(enhanced_restaurant)

            except Exception as exc:
                dump_chain_state(
                    "restaurant_enhancement_error",
                    {
                        "restaurant": restaurant.get("name", "Unknown"),
                        "error": str(exc),
                    },
                    error=exc,
                )
                # Continue with the original restaurant if enhancement fails
                enhanced_main_list.append(restaurant)

        return {"main_list": enhanced_main_list}

    def _enhance_single_restaurant(
        self,
        restaurant: Dict[str, Any],
        follow_up_queries: List[Dict[str, Any]],
        secondary_filter_parameters: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Enhance a single restaurant with additional data from Google Maps and web searches.

        Returns None if restaurant doesn't meet rating criteria, otherwise returns enhanced restaurant.
        """
        restaurant_name = restaurant.get("name", "Unknown Restaurant")

        # ------------------------------------------------------------------
        # 1️⃣  Google Maps integration (for address, rating, hours)
        # ------------------------------------------------------------------
        maps_data = self._fetch_google_maps_data(restaurant)

        # Apply rating filter - if restaurant doesn't meet minimum rating, exclude it
        if maps_data and maps_data.get("rating"):
            if maps_data["rating"] < MIN_ACCEPTABLE_RATING:
                dump_chain_state(
                    "restaurant_filtered_low_rating",
                    {
                        "restaurant": restaurant_name,
                        "rating": maps_data["rating"],
                        "minimum_required": MIN_ACCEPTABLE_RATING,
                    }
                )
                return None  # Exclude this restaurant

        # ------------------------------------------------------------------
        # 2️⃣  Build specific queries for this restaurant
        # ------------------------------------------------------------------
        specific_queries = self._build_restaurant_queries(restaurant, follow_up_queries)

        # ------------------------------------------------------------------
        # 3️⃣  Check global guide presence
        # ------------------------------------------------------------------
        global_guide_info = self._check_global_guides(restaurant_name)

        # ------------------------------------------------------------------
        # 4️⃣  Web searches + scraping
        # ------------------------------------------------------------------
        all_search_results: List[Dict[str, Any]] = []
        for query in specific_queries:
            try:
                # Add restaurant genre to the query if available
                if restaurant.get("genre") and "restaurant" in query:
                    genre = restaurant.get("genre")
                    # Replace generic "restaurant" with specific genre
                    enhanced_query = query.replace("restaurant", genre)
                else:
                    enhanced_query = query

                results = self.search_agent._execute_search(enhanced_query)
                filtered = self.search_agent._filter_results(results)[:MAX_RESULTS_PER_QUERY]

                # FIX: Use the helper method to properly await the async scraping
                scraped = self._run_async_scraping(filtered)
                all_search_results.extend(scraped)
                time.sleep(1)  # be nice
            except Exception as exc:
                dump_chain_state(
                    "follow_up_search_error",
                    {
                        "restaurant": restaurant_name,
                        "query": query,
                        "error": str(exc),
                    },
                    error=exc,
                )

        # ------------------------------------------------------------------
        # 5️⃣  Compile sources & extra details
        # ------------------------------------------------------------------
        combined_sources: List[str] = []
        existing = restaurant.get("sources", [])
        if isinstance(existing, str):
            combined_sources = [existing]
        elif isinstance(existing, list):
            combined_sources = existing[:]

        # Add sources from scraped results
        for result in all_search_results:
            source_name = self._extract_source_name(result.get("url", ""))
            if source_name and source_name not in combined_sources:
                combined_sources.append(source_name)

        # ------------------------------------------------------------------
        # 6️⃣  Compile extra details from all sources
        # ------------------------------------------------------------------
        extra_details = []
        for result in all_search_results:
            content = result.get("content", "")
            if content and len(content.strip()) > 50:
                extra_details.append(content[:500])  # Truncate for brevity

        # Add global guide info to extra details
        if global_guide_info:
            extra_details.append(f"Global Guide Recognition: {global_guide_info}")

        # ------------------------------------------------------------------
        # 7️⃣  Merge everything back into the restaurant
        # ------------------------------------------------------------------
        enhanced_restaurant = restaurant.copy()
        enhanced_restaurant["sources"] = combined_sources

        if extra_details:
            enhanced_restaurant["extra_details"] = " | ".join(extra_details)

        # Add Google Maps data
        if maps_data:
            enhanced_restaurant.update(maps_data)

        return enhanced_restaurant

    def _fetch_google_maps_data(self, restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch Google Maps data for a restaurant."""
        try:
            restaurant_name = restaurant.get("name", "")
            location = restaurant.get("location", "")

            if not restaurant_name:
                return None

            # Build search query
            search_query = restaurant_name
            if location:
                search_query += f" {location}"

            # Search for the place
            places_result = self.gmaps.places(
                query=search_query,
                type="restaurant"
            )

            if not places_result.get("results"):
                return None

            # Get the first result (most relevant)
            place = places_result["results"][0]
            place_id = place.get("place_id")

            if not place_id:
                return None

            # Get detailed information
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=MAPS_FIELDS
            )

            if not place_details.get("result"):
                return None

            result = place_details["result"]
            maps_data = {}

            # Extract formatted address
            if result.get("formatted_address"):
                maps_data["formatted_address"] = result["formatted_address"]
                # Create clickable Google Maps link
                maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                maps_data["google_maps_url"] = maps_url

            # Extract rating
            if result.get("rating"):
                maps_data["rating"] = result["rating"]

            # Extract opening hours
            if result.get("opening_hours", {}).get("weekday_text"):
                hours_text = "\n".join(result["opening_hours"]["weekday_text"])
                maps_data["opening_hours"] = hours_text

            return maps_data

        except Exception as e:
            dump_chain_state(
                "google_maps_error",
                {
                    "restaurant": restaurant.get("name", "Unknown"),
                    "error": str(e),
                }
            )
            return None

    def _build_restaurant_queries(
        self, 
        restaurant: Dict[str, Any], 
        follow_up_queries: List[Dict[str, Any]]
    ) -> List[str]:
        """Build specific search queries for a restaurant."""
        restaurant_name = restaurant.get("name", "")
        location = restaurant.get("location", "")

        queries = []

        for query_info in follow_up_queries:
            template = query_info.get("query", "")

            # Replace placeholders
            specific_query = template.replace("{restaurant_name}", restaurant_name)
            specific_query = specific_query.replace("{location}", location)

            queries.append(specific_query)

        return queries

    def _check_global_guides(self, restaurant_name: str) -> Optional[str]:
        """Check if restaurant appears in major global guides."""
        global_guides = [
            "Michelin Guide",
            "World's 50 Best Restaurants",
            "Zagat",
            "James Beard Awards"
        ]

        # This is a simplified implementation
        # In practice, you might want to do actual searches for these
        for guide in global_guides:
            # Placeholder for actual guide checking logic
            pass

        return None

    def _extract_source_name(self, url: str) -> str:
        """Extract readable source name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # Clean up common domain extensions
            if domain.endswith('.com'):
                domain = domain[:-4]
            elif domain.endswith('.org'):
                domain = domain[:-4]
            elif domain.endswith('.net'):
                domain = domain[:-4]

            return domain.title()
        except Exception:
            return "Web Source"