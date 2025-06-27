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

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Collect all restaurants from both main_list and hidden_gems
            all_restaurants = []

            # Add main_list restaurants
            main_list = formatted_recommendations.get("main_list", [])
            if isinstance(main_list, list):
                all_restaurants.extend(main_list)

            # Add hidden_gems restaurants to the main list
            hidden_gems = formatted_recommendations.get("hidden_gems", [])
            if isinstance(hidden_gems, list):
                all_restaurants.extend(hidden_gems)

            dump_chain_state(
                "follow_up_search_start",
                {
                    "total_restaurant_count": len(all_restaurants),
                    "follow_up_queries_count": len(follow_up_queries),
                    "secondary_parameters": secondary_filter_parameters,
                },
            )

            enhanced_recommendations: Dict[str, List[Dict[str, Any]]] = {
                "main_list": [],
            }

            # Process all restaurants and put them in main_list
            for restaurant in all_restaurants:
                restaurant = self._enhance_restaurant(
                    restaurant, follow_up_queries, secondary_filter_parameters
                )
                if restaurant:  # None == rejected (rating < MIN_ACCEPTABLE_RATING)
                    enhanced_recommendations["main_list"].append(restaurant)

            dump_chain_state(
                "follow_up_search_complete",
                {
                    "enhanced_main_list_count": len(enhanced_recommendations["main_list"]),
                },
            )

            return enhanced_recommendations

    def _extract_restaurant_genre(self, restaurant: Dict[str, Any]) -> str:
        """Extract restaurant genre/type from available data to improve Maps search.

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

    def _get_google_maps_info(self, name: str, location: str, restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Look up the place on Google Maps and return basic metadata.

        We run a *Text Search* first because it does a decent job at matching
        ambiguous names. The first candidate is fed into *Place Details*
        to get a stable *place_id* URL, rating, etc.

        Includes restaurant genre to improve search accuracy.
        """
        try:
            # Extract restaurant genre to help differentiate from other businesses
            genre = self._extract_restaurant_genre(restaurant)

            # Add genre to the search query to improve matching
            text_query = f"{name} {genre} {location}"

            dump_chain_state(
                "google_maps_search",
                {
                    "restaurant": name,
                    "genre": genre,
                    "location": location,
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

    def _enhance_restaurant(
        self,
        restaurant: Dict[str, Any],
        follow_up_queries: List[Dict[str, Any]],
        secondary_filter_parameters: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Enrich *restaurant* or return ``None`` if it should be discarded."""

        restaurant_name = restaurant.get("name", "")
        restaurant_location = (
            restaurant.get("address", "").split(",")[0] if restaurant.get("address") else ""
        )
        city = restaurant.get("city", restaurant_location)

        # ------------------------------------------------------------------
        # 1️⃣  Google Maps pass – decides early rejection / base metadata
        # ------------------------------------------------------------------
        maps_info = self._get_google_maps_info(restaurant_name, city or restaurant_location, restaurant)
        if maps_info and maps_info.get("rating") is not None:
            rating = float(maps_info["rating"])
            if rating < MIN_ACCEPTABLE_RATING:
                dump_chain_state(
                    "restaurant_rejected_low_rating",
                    {
                        "restaurant": restaurant_name,
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
                    "location": restaurant_location,
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
            specific_queries = self._default_queries_for(
                restaurant, restaurant_name, restaurant_location or city
            )

        missing_info = restaurant.get("missing_info", [])
        specific_queries.extend(
            f"{restaurant_name} restaurant {restaurant_location or city} {info}" for info in missing_info
        )

        if secondary_filter_parameters:
            specific_queries.extend(
                f"{restaurant_name} restaurant {restaurant_location or city} {param}"
                for param in secondary_filter_parameters
            )

        # ------------------------------------------------------------------
        # 3️⃣  Global guide check – independent of Maps rating
        # ------------------------------------------------------------------
        global_guide_info = self._check_global_guides(restaurant_name, restaurant_location or city)
        global_guide_sources = self._extract_sources(global_guide_info)

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
                scraped = self.scraper.scrape_search_results(filtered)
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
            existing = [existing]
        combined_sources.extend(existing)
        combined_sources.extend(global_guide_sources)
        combined_sources.extend(self._extract_sources(all_search_results))
        combined_sources = list({s for s in combined_sources if s})  # dedupe / remove blanks

        banned_sources = {"Tripadvisor", "Yelp", "Google"}
        cleaned_sources = [s for s in combined_sources if s not in banned_sources]

        extra = self._extract_additional_details(all_search_results)

        # ------------------------------------------------------------------
        # 6️⃣  Final assembly
        # ------------------------------------------------------------------
        enhanced = restaurant.copy()
        enhanced["sources"] = cleaned_sources

        # Prefer longer description if we found something better
        if extra.get("description") and len(extra["description"]) > len(enhanced.get("description", "")):
            enhanced["description"] = extra["description"]

        if extra.get("hours") and not enhanced.get("hours"):
            enhanced["hours"] = extra["hours"]
        if extra.get("price_info") and not enhanced.get("price_range"):
            enhanced["price_range"] = extra["price_info"]

        enhanced["additional_info"] = {
            "follow_up_results": all_search_results + global_guide_info,
            "global_guide_results": global_guide_info,
            "secondary_parameters_checked": secondary_filter_parameters or [],
        }

        return enhanced

    def _default_queries_for(self, restaurant: Dict[str, Any], name: str, location: str) -> List[str]:
        """Return a minimal set of follow‑up queries based on missing fields."""
        queries: List[str] = []

        # Get restaurant genre if available
        genre = restaurant.get("genre", "restaurant")

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
                        hours_text = hours_text[: end + 1]
                    additional["hours"] = hours_text.strip()
                    break

        for indicator in ["price", "cost", "menu is", "dishes from", "dishes cost", "€", "$", "£"]:
            if indicator in low:
                pos = low.find(indicator)
                additional["price_info"] = combined_content[max(0, pos - 20) : pos + 80].strip()
                break

        return additional

    def _extract_sources(self, source_results: List[Dict[str, Any]]) -> List[str]:
        sources: List[str] = []
        banned = {"tripadvisor", "yelp", "zagat"}

        for result in source_results:
            # Explicit source_name from scraper
            if src := result.get("source_name"):
                if src not in sources and src.lower() not in banned:
                    sources.append(src)
                continue

            domain = result.get("source_domain", "")
            if not domain:
                continue

            base = domain.replace("www.", "").split(".")[0]
            base = " ".join(part.capitalize() for part in base.replace("-", " ").replace("_", " ").split())

            special = {
                "michelin": "Michelin Guide",
                "foodandwine": "Food & Wine",
                "eater": "Eater",
                "infatuation": "The Infatuation",
                "50best": "World's 50 Best",
                "worlds50best": "World's 50 Best",
                "worldofmouth": "World of Mouth",
                "nytimes": "New York Times",
                "timeout": "Time Out",
                "forbes": "Forbes",
                "telegraph": "The Telegraph",
                "guardian": "The Guardian",
                "cntraveler": "Condé Nast Traveler",
            }
            for key, val in special.items():
                if key in domain:
                    base = val
                    break

            if base not in sources and base.lower() not in banned:
                sources.append(base)

            # Guide field – treat similarly
            if guide := result.get("guide"):
                gdomain = guide.replace("www.", "").split(".")[0]
                gdomain = gdomain.lower()
                mapping = {
                    "theworlds50best": "World's 50 Best",
                    "50best": "World's 50 Best",
                    "michelin": "Michelin Guide",
                    "wordofmouth": "World of Mouth",
                    "oadguides": "OAD Guides",
                    "culinarybackstreets": "Culinary Backstreets",
                }
                guide_name = mapping.get(gdomain, gdomain.capitalize())
                if guide_name not in sources and guide_name.lower() not in banned:
                    sources.append(guide_name)

        return sources

    def _check_global_guides(self, restaurant_name: str, location: str) -> List[Dict[str, Any]]:
        guides = [
            "theworlds50best.com",
            "worldofmouth.app",
            "guide.michelin.com",
            "culinarybackstreets.com",
            "oadguides.com",
            "laliste.com",
        ]
        results: List[Dict[str, Any]] = []

        for guide in guides:
            try:
                dump_chain_state(
                    "checking_global_guide",
                    {
                        "restaurant_name": restaurant_name,
                        "guide": guide,
                    },
                )

                query = f"site:{guide} {restaurant_name} {location}"
                guide_results = self.search_agent._execute_search(query)
                filtered = self.search_agent._filter_results(guide_results)

                dump_chain_state(
                    "global_guide_results",
                    {
                        "restaurant_name": restaurant_name,
                        "guide": guide,
                        "results_count": len(filtered),
                    },
                )

                for r in filtered:
                    r["guide"] = guide
                    results.append(r)
                time.sleep(1)
            except Exception as exc:
                dump_chain_state(
                    "global_guide_error",
                    {
                        "restaurant_name": restaurant_name,
                        "guide": guide,
                        "error": str(exc),
                    },
                    error=exc,
                )
        return results