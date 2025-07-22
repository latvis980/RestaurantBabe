# agents/follow_up_search_agent.py
# FIXED VERSION - Properly preserves location from initial query

from __future__ import annotations

"""Follow‑up search agent

MAJOR FIX: Now properly stores and uses the original destination from the query
throughout the entire pipeline, ensuring Google Maps searches have proper location context.

The location flow is now:
1. Query Analyzer extracts destination → stored in analysis result
2. Orchestrator passes destination to follow-up search
3. Follow-up search uses destination consistently for all restaurants
4. Google Maps gets: "{restaurant_name} {genre} {destination}" 

Updated to return only main_list, no hidden_gems.
"""

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

    FIXED: Now properly handles location throughout the pipeline.
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

    def perform_follow_up_searches(
        self,
        formatted_recommendations: Dict[str, List[Dict[str, Any]]],
        follow_up_queries: List[Dict[str, Any]],
        destination: str,  # FIXED: Added destination parameter
        secondary_filter_parameters: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Enrich every restaurant in *formatted_recommendations*.

        FIXED: Now requires destination parameter for consistent location handling.

        Args:
            formatted_recommendations: Restaurant data from list analyzer
            follow_up_queries: Specific queries for each restaurant
            destination: Original destination from query (e.g., "Paris", "Tokyo")
            secondary_filter_parameters: Additional filter criteria

        Returns:
            Dict with only main_list, no hidden_gems.
            Restaurants that do not meet the minimum Google rating are silently
            excluded from the output.
        """
        dump_chain_state(
            "follow_up_search_start",
            {
                "destination": destination,
                "restaurant_count": len(formatted_recommendations.get("main_list", [])),
                "min_rating": MIN_ACCEPTABLE_RATING,
            },
        )

        enhanced_restaurants = []
        main_list = formatted_recommendations.get("main_list", [])

        for restaurant in main_list:
            # FIXED: Pass destination to enhancement function
            enhanced = self._enhance_restaurant(
                restaurant, 
                follow_up_queries, 
                destination,  # FIXED: Always use original destination
                secondary_filter_parameters
            )
            if enhanced is not None:
                enhanced_restaurants.append(enhanced)

        dump_chain_state(
            "follow_up_search_complete",
            {
                "original_count": len(main_list),
                "enhanced_count": len(enhanced_restaurants),
                "rejected_count": len(main_list) - len(enhanced_restaurants),
            },
        )

        return {"main_list": enhanced_restaurants}

    def _enhance_restaurant(
        self,
        restaurant: Dict[str, Any],
        follow_up_queries: List[Dict[str, Any]],
        destination: str,  # FIXED: Added destination parameter
        secondary_filter_parameters: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Enrich *restaurant* or return ``None`` if it should be discarded.

        FIXED: Now uses the original destination consistently instead of trying
        to extract location from incomplete restaurant data.
        """
        restaurant_name = restaurant.get("name", "")

        # FIXED: Always use the original destination from the query
        # Don't try to extract from incomplete address data
        restaurant_city = destination

        dump_chain_state(
            "restaurant_enhancement_start",
            {
                "restaurant": restaurant_name,
                "destination": destination,  # FIXED: Log the actual destination being used
            },
        )

        # ------------------------------------------------------------------
        # 1️⃣  Google Maps pass – decides early rejection / base metadata
        # ------------------------------------------------------------------
        # FIXED: Pass destination instead of trying to extract from restaurant data
        maps_info = self._get_google_maps_info(restaurant_name, destination, restaurant)

        if maps_info and maps_info.get("rating") is not None:
            rating = float(maps_info["rating"])
            if rating < MIN_ACCEPTABLE_RATING:
                dump_chain_state(
                    "restaurant_rejected_low_rating",
                    {
                        "restaurant": restaurant_name,
                        "destination": destination,
                        "rating": rating,
                    },
                )
                return None  # discard completely

            # Update basic fields from Maps
            restaurant["rating"] = rating
            if maps_info.get("address"):
                # clickable address – useful both for UI and SEO
                safe_address = maps_info["address"]
                link = maps_info["url"]
                restaurant["address"] = (
                    f'<a href="{link}" target="_blank" rel="noopener noreferrer">'
                    f"{safe_address}</a>"
                )
            if maps_info.get("hours") and not restaurant.get("hours"):
                restaurant["hours"] = maps_info["hours"]
            if maps_info.get("genre"):
                restaurant["genre"] = maps_info["genre"]

        else:
            # No Maps hit – we keep the candidate but do NOT add rating
            dump_chain_state(
                "google_maps_no_hit",
                {
                    "restaurant": restaurant_name,
                    "destination": destination,  # FIXED: Log destination instead of restaurant_location
                },
            )

        # ------------------------------------------------------------------
        # 2️⃣  Determine follow‑up search queries
        # ------------------------------------------------------------------
        specific_queries: List[str] = []
        for qs in follow_up_queries:
            if qs.get("restaurant_name") == restaurant_name:
                specific_queries = qs.get("queries", [])
                break

        if not specific_queries:
            # FIXED: Use destination instead of city/restaurant_location
            specific_queries = self._default_queries_for(
                restaurant, restaurant_name, destination
            )

        missing_info = restaurant.get("missing_info", [])
        specific_queries.extend(
            f"{restaurant_name} restaurant {destination} {info}" for info in missing_info  # FIXED: Use destination
        )

        if secondary_filter_parameters:
            specific_queries.extend(
                f"{restaurant_name} restaurant {destination} {param}"  # FIXED: Use destination
                for param in secondary_filter_parameters
            )

        # ------------------------------------------------------------------
        # 3️⃣  Global guide check – independent of Maps rating
        # ------------------------------------------------------------------
        # FIXED: Use destination instead of restaurant_location or city
        global_guide_info = self._check_global_guides(restaurant_name, destination)
        global_guide_sources = self._extract_sources(global_guide_info)

        # ------------------------------------------------------------------
        # 4️⃣  Run remaining follow‑up searches
        # ------------------------------------------------------------------
        search_results = []
        for query in specific_queries[:MAX_RESULTS_PER_QUERY]:
            try:
                results = self.search_agent.search(query)
                filtered = [r for r in results if not self._should_exclude_domain(r.get("url", ""))]
                search_results.extend(filtered[:2])
                time.sleep(0.1)
            except Exception as exc:
                dump_chain_state("follow_up_search_error", {"query": query}, error=exc)

        # ------------------------------------------------------------------
        # 5️⃣  Scrape additional details
        # ------------------------------------------------------------------
        scraped_results = []
        for result in search_results[:MAX_RESULTS_PER_QUERY]:
            url = result.get("url")
            if url:
                try:
                    scraped = self.scraper.scrape_url(url)
                    if scraped:
                        scraped_results.append({**result, **scraped})
                        time.sleep(0.2)
                except Exception as exc:
                    dump_chain_state("scraping_error", {"url": url}, error=exc)

        # ------------------------------------------------------------------
        # 6️⃣  Extract additional details from scraped content
        # ------------------------------------------------------------------
        additional_details = self._extract_additional_details(scraped_results)

        # ------------------------------------------------------------------
        # 7️⃣  Merge everything
        # ------------------------------------------------------------------
        for key, value in additional_details.items():
            if key not in restaurant or not restaurant[key]:
                restaurant[key] = value

        # Add global guide sources
        existing_sources = restaurant.get("sources", [])
        if isinstance(existing_sources, str):
            existing_sources = [existing_sources]
        all_sources = list(set(existing_sources + global_guide_sources))
        restaurant["sources"] = all_sources

        # FIXED: Ensure city is set to destination
        restaurant["city"] = destination
        restaurant["location"] = destination

        dump_chain_state(
            "restaurant_enhancement_complete",
            {
                "restaurant": restaurant_name,
                "destination": destination,
                "has_address": bool(restaurant.get("address")),
                "has_rating": bool(restaurant.get("rating")),
                "source_count": len(all_sources),
            },
        )

        return restaurant

    def _get_google_maps_info(self, name: str, location: str, restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Look up the place on Google Maps and return basic metadata.

        FIXED: Now uses the original destination location consistently.
        We run a *Text Search* first because it does a decent job at matching
        ambiguous names. The first candidate is fed into *Place Details*
        to get a stable *place_id* URL, rating, etc.

        Includes restaurant genre to improve search accuracy.
        """
        try:
            # Extract restaurant genre to help differentiate from other businesses
            genre = self._extract_restaurant_genre(restaurant)

            # FIXED: Use consistent location parameter (the destination)
            text_query = f"{name} {genre} {location}"

            dump_chain_state(
                "google_maps_search",
                {
                    "restaurant": name,
                    "genre": genre,
                    "location": location,  # FIXED: This is now always the destination
                    "full_query": text_query
                },
            )

            search_resp = self.gmaps.places(query=text_query)
            candidates = search_resp.get("results", [])
            if not candidates:
                return None

            first = candidates[0]
            place_id = first["place_id"]

            # Pull details – this counts as a separate request
            details = self.gmaps.place(place_id=place_id, fields=MAPS_FIELDS)
            result = details.get("result", {})

            rating = result.get("rating")
            formatted_address = result.get("formatted_address")
            url = result.get("url") or f"https://www.google.com/maps/place/?q=place_id:{place_id}"

            opening_hours = None
            oh_data = result.get("opening_hours", {})
            if oh_data and "weekday_text" in oh_data:
                opening_hours = "; ".join(oh_data["weekday_text"])

            return {
                "place_id": place_id,
                "rating": rating,
                "address": formatted_address,
                "url": url,
                "hours": opening_hours,
                "genre": genre  # Include the genre we used in the search
            }
        except Exception as exc:  # noqa: BLE001 – we want to log every failure
            dump_chain_state(
                "google_maps_error",
                {
                    "restaurant": name,
                    "location": location,
                    "error": str(exc),
                },
                error=exc,
            )
            return None

    def _default_queries_for(self, restaurant: Dict[str, Any], name: str, location: str) -> List[str]:
        """Generate default follow‑up queries for a restaurant.

        FIXED: Now uses the location parameter (destination) consistently.
        """
        queries: List[str] = []

        # Get restaurant genre if available
        genre = restaurant.get("genre", "restaurant")

        # FIXED: All queries now use the location parameter (destination)
        if not restaurant.get("address") or restaurant.get("address") == "Address unavailable":
            queries.append(f"{name} {genre} {location} address location")
        if not restaurant.get("price_range"):
            queries.append(f"{name} {genre} {location} price range cost")
        if not restaurant.get("recommended_dishes") or len(restaurant.get("recommended_dishes", [])) < 2:
            queries.append(f"{name} {genre} {location} signature dishes menu specialties")
        if not restaurant.get("sources") or len(restaurant.get("sources", [])) < 2:
            queries.append(f"{name} {genre} {location} reviews recommended by critic guide")
        # Always check major guides even if everything else is complete
        queries.append(f"{name} {genre} {location} michelin guide awards")
        return queries[:4]  # keep it short – we do a lot of calls already

    def _extract_additional_details(self, search_results: List[Dict[str, Any]]) -> Dict[str, str]:
        additional: Dict[str, str] = {}
        combined_content = "\n\n".join(r.get("scraped_content", "") for r in search_results if r.get("scraped_content"))

        # Simple heuristic – first 400 chars make a decent description
        if combined_content and len(combined_content) > 400:
            additional["description"] = combined_content[:400].rstrip() + "…"

        low = combined_content.lower()
        if any(k in low for k in ["opening hours", "open from", "open:", "hours:", "we are open"]):
            for indicator in ["opening hours", "open from", "open:", "hours:", "we are open"]:
                if indicator in low:
                    pos = low.find(indicator) + len(indicator)
                    hours_text = combined_content[pos : pos + 100]
                    end = hours_text.find(".")
                    if end > 0:
                        additional["hours"] = hours_text[:end].strip()
                    else:
                        additional["hours"] = hours_text[:50].strip()
                    break

        return additional

    def _should_exclude_domain(self, url: str) -> bool:
        """Check if the URL's domain should be excluded from follow‑up searches."""
        excluded_domains = [
            "tripadvisor.com", "yelp.com", "booking.com", "expedia.com",
            "hotels.com", "agoda.com", "trivago.com", "kayak.com"
        ]

        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return any(excluded in domain for excluded in excluded_domains)
        except Exception:
            return False

    def _check_global_guides(self, restaurant_name: str, location: str) -> List[Dict[str, Any]]:
        """Check if the restaurant is mentioned in major global guides.

        FIXED: Now uses location parameter (destination) consistently.
        """
        global_guides = [
            "guide.michelin.com",
            "theworlds50best.com", 
            "worldofmouth.app",
            "culinarybackstreets.com",
            "oadguides.com",
            "laliste.com"
        ]

        results = []
        for guide in global_guides:
            try:
                # FIXED: Use location parameter (destination)
                query = f"site:{guide} {restaurant_name} {location}"
                guide_results = self.search_agent.search(query)
                if guide_results:
                    for result in guide_results[:2]:
                        result["guide"] = guide
                        results.append(result)
                time.sleep(0.1)
            except Exception as exc:
                dump_chain_state("global_guide_search_error", {"guide": guide, "query": query}, error=exc)

        return results

    def _extract_sources(self, guide_results: List[Dict[str, Any]]) -> List[str]:
        """Extract source names from guide results."""
        sources = []
        for result in guide_results:
            guide = result.get("guide", "")
            if "michelin" in guide:
                sources.append("Michelin Guide")
            elif "50best" in guide:
                sources.append("World's 50 Best")
            elif "worldofmouth" in guide:
                sources.append("World of Mouth")
            elif "culinarybackstreets" in guide:
                sources.append("Culinary Backstreets")
            elif "oadguides" in guide:
                sources.append("OAD")
            elif "laliste" in guide:
                sources.append("La Liste")

        return list(set(sources))  # remove duplicates

    def _extract_restaurant_genre(self, restaurant: Dict[str, Any]) -> str:
        """Extract restaurant genre/type from the restaurant data.

        This helps distinguish restaurants from other businesses with similar names.
        """
        # Check description for genre hints
        description = restaurant.get("description", "").lower()

        # Common restaurant types
        genres = {
            "italian restaurant": ["italian", "pasta", "pizza", "trattoria"],
            "japanese restaurant": ["japanese", "sushi", "ramen", "izakaya"],
            "chinese restaurant": ["chinese", "dim sum", "dumpling"],
            "indian restaurant": ["indian", "curry"],
            "french restaurant": ["french", "bistro", "brasserie"],
            "steakhouse": ["steak", "steakhouse", "grill"],
            "seafood restaurant": ["seafood", "fish"],
            "cafe": ["cafe", "café", "coffee", "brunch"],
            "cocktail bar": ["cocktail", "bar", "pub", "tapas"],
            "fine dining restaurant": ["michelin", "fine dining", "gourmet"]
        }

        # Look for genre hints in description
        for genre, keywords in genres.items():
            if any(kw in description for kw in keywords):
                return genre

        # Default to "restaurant" if no specific genre found
        return "restaurant"