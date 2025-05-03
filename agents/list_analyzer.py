from __future__ import annotations
"""
Re‑worked **ListAnalyzer** – now covers *all* incoming data instead of just the
first N articles.  The class can operate in two modes that are auto‑detected at
runtime:

1. **Article‑mode** (legacy)
   • Input = list of crawled search‑results (each dict has ``title``,
     ``scraped_content`` …).
   • We still construct a token‑aware prompt but shuffle the list first and then
     take *evenly spaced* samples so every part of the list has a chance to be
     represented.

2. **Restaurant‑list mode** (NEW)
   • Input = several **lists of restaurants** already extracted from reputable
     sources; every item must contain at least ``source_name`` (or
     ``source_domain``) *and* ``restaurants`` (array of either strings or
     dicts with ``name``/``description``/``address``).
   • We aggregate all entries programmatically **before** calling the LLM so it
     sees the *complete* picture with mention‑counts and source names.
   • Hidden‑gem / main‑list separation is done *after* the final merge, so no
     single chunk can bias the outcome.

The public signature stays the same – just pass your data as the first
argument and the analyzer will figure out the correct path.
"""

import json
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_mistralai import ChatMistralAI
from tiktoken import get_encoding

from utils.database import save_data
from utils.debug_utils import dump_chain_state, log_function_call

# ---------------------------------------------------------------------------
#  helper dataclasses – kept minimal to avoid extra deps
# ---------------------------------------------------------------------------
class _RestaurantBucket:
    """Collects all evidence we have for a single restaurant name."""

    def __init__(self, canonical_name: str):
        self.name = canonical_name
        self.sources: set[str] = set()
        self.descriptions: list[str] = []
        self.addresses: set[str] = set()

    def add(self, source: str, desc: str | None = None, address: str | None = None):
        if source:
            self.sources.add(source)
        if desc:
            self.descriptions.append(desc.strip())
        if address:
            self.addresses.add(address.strip())

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def to_prompt_block(self, max_desc: int = 2) -> str:
        show_desc = self.descriptions[:max_desc]
        lines = [
            f"• {self.name}",
            f"  Sources ({len(self.sources)}): {', '.join(sorted(self.sources))}",
        ]
        for d in show_desc:
            lines.append(f"  – {d}")
        if len(self.descriptions) > max_desc:
            lines.append("  – …")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
class ListAnalyzer:
    def __init__(
        self,
        config,
        *,
        max_prompt_tokens: int = 12_000,
        encoding_name: str = "cl100k_base",
        per_article_head_tokens: int = 200,
    ) -> None:
        # ––– LLM ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.8,
            mistral_api_key=config.MISTRAL_API_KEY,
        )

        # ––– PROMPT –––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.system_prompt = (
            "You are a restaurant recommendation expert analysing web search results"
            "for data to identify the *best* restaurants and *promising hidden gems*."
            "\n\nTASK:\n"
            "1. From the supplied material extract restaurant names and compile a list"\n
            "2. Analyse the descriptions and sources to identify the best restaurants"
            "3. Produce two lists of restaurants: *main_list* (widely endorsed) and "
            "*hidden_gems* (≤ 2 sources but glowing praise).\n"
            "4. Merge all descriptions for the same restaurant into a single 40‑60 word paragraph per place."\n
            "5. Assume price‑range (€, €€, €€€)"\n 
            "6. Extract up to 3 recommended dishes if available"\n
            "Guidelines: Ignore TripAdvisor/Yelp; prefer guides, critics, local publications; always return at least 7 total restaurants in the main_list, 2 restaurants in hidden_gems." )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "CITY: {city}\nPRIMARY SEARCH PARAMETERS: {primary}\n\n"\
                    "SECONDARY FILTER PARAMETERS: {secondary}\n\n"\
                    "DATA:\n{payload}\n\nPlease return JSON with arrays *main_list* and *hidden_gems*.",
                ),
            ]
        )
        self.chain = self.prompt | self.model

        # ––– token accounting –––––––––––––––––––––––––––––––––––––––––––
        self.max_prompt_tokens = max_prompt_tokens
        self._enc = get_encoding(encoding_name)
        self._per_article_head_tokens = per_article_head_tokens

        self.config = config

    # ------------------------------------------------------------------
    #  Public entry
    # ------------------------------------------------------------------
    @log_function_call
    def analyze(
        self,
        data: List[Dict[str, Any]],
        keywords_for_analysis: List[str] | str,
        primary_parameters: Optional[List[str] | str] = None,
        secondary_parameters: Optional[List[str] | str] = None,
        destination: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route to the correct sub‑pipeline based on the shape of *data*."""
        if not data:
            raise ValueError("Empty data passed to ListAnalyzer")

        # Heuristic: if **every** item has a "restaurants" key → list‑mode
        is_list_mode = all("restaurants" in item for item in data)

        if is_list_mode:
            return self._analyze_restaurant_lists(
                data, keywords_for_analysis, primary_parameters, secondary_parameters, destination
            )
        else:
            return self._analyze_articles(
                data, keywords_for_analysis, primary_parameters, secondary_parameters, destination
            )

    # ------------------------------------------------------------------
    #  ––– Article mode (legacy) –––
    # ------------------------------------------------------------------
    def _analyze_articles(
        self,
        search_results: List[Dict[str, Any]],
        keywords_for_analysis: List[str] | str,
        primary_parameters: Optional[List[str] | str],
        secondary_parameters: Optional[List[str] | str],
        destination: Optional[str],
    ) -> Dict[str, Any]:
        # Shuffle so top‑of‑list bias disappears, then sample evenly
        random.shuffle(search_results)

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            city = self._extract_city(primary_parameters, destination)
            formatted = self._format_search_results_balanced(search_results)

            kw_str = self._list_to_str(keywords_for_analysis)
            primary_str = self._list_to_str(primary_parameters)
            secondary_str = self._list_to_str(secondary_parameters)

            resp = self.chain.invoke(
                {
                    "payload": formatted,
                    "city": city,
                    "primary": primary_str,
                    "secondary": secondary_str,
                }
            )
            return self._postprocess_response(resp, city)

    # ------------------------------------------------------------------
    #  ––– Restaurant‑list mode (NEW) –––
    # ------------------------------------------------------------------
    def _analyze_restaurant_lists(
        self,
        restaurant_lists: List[Dict[str, Any]],
        keywords_for_analysis: List[str] | str,
        primary_parameters: Optional[List[str] | str],
        secondary_parameters: Optional[List[str] | str],
        destination: Optional[str],
    ) -> Dict[str, Any]:
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            city = self._extract_city(primary_parameters, destination)

            # 1. Aggregate all evidence first so we cover *every* restaurant
            buckets = self._aggregate_restaurant_lists(restaurant_lists)
            payload = "\n\n".join(b.to_prompt_block() for b in buckets.values())

            kw_str = self._list_to_str(keywords_for_analysis)
            primary_str = self._list_to_str(primary_parameters)
            secondary_str = self._list_to_str(secondary_parameters)

            resp = self.chain.invoke(
                {
                    "payload": payload,
                    "city": city,
                    "primary": primary_str,
                    "secondary": secondary_str,
                }
            )
            return self._postprocess_response(resp, city)

    # ------------------------------------------------------------------
    #  helpers – fmting & aggregation
    # ------------------------------------------------------------------
    def _format_search_results_balanced(self, search_results: List[Dict[str, Any]]) -> str:
        """Evenly sample articles across the full list until the token budget is
        exhausted (as opposed to walking from the top)."""
        lines: list[str] = []
        used_tokens = 0
        N = len(search_results)
        # Take strides of size sqrt(N) to spread coverage
        stride = max(1, int(N ** 0.5))
        idx = 0
        visited = set()
        while len(visited) < N:
            res = search_results[idx]
            visited.add(idx)
            base: list[str] = [
                f"Title: {res.get('title', 'Unknown')}",
                f"URL: {res.get('url', 'Unknown')}",
                f"Source: {res.get('source_domain', 'Unknown')}",
            ]
            if res.get("source_name"):
                base.append(f"Source Name: {res['source_name']}")
            if res.get("description"):
                base.append(f"Description: {res['description']}")

            content = res.get("scraped_content", "")
            allowed = self.max_prompt_tokens - used_tokens - self._tokens("\n".join(base))
            if allowed <= 0:
                break

            if content:
                content_tokens = self._tokens(content)
                take = min(content_tokens, min(self._per_article_head_tokens, allowed))
                excerpt = self._enc.decode(self._enc.encode(content)[:take])
                base.append(f"Content: {excerpt}{' …' if content_tokens > take else ''}")
                used_tokens += take + self._tokens("\n".join(base[:-1]))
            else:
                used_tokens += self._tokens("\n".join(base))

            lines.append("\n".join(base))
            if used_tokens >= self.max_prompt_tokens:
                break
            idx = (idx + stride) % N
        return "\n\n".join(lines)

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _aggregate_restaurant_lists(self, restaurant_lists: List[Dict[str, Any]]) -> Dict[str, _RestaurantBucket]:
        buckets: Dict[str, _RestaurantBucket] = {}
        for block in restaurant_lists:
            source = block.get("source_name") or block.get("source_domain") or "Unknown"
            for itm in block.get("restaurants", []):
                # Support both raw strings and dict objects
                if isinstance(itm, str):
                    name = itm.strip()
                    desc = None
                    addr = None
                else:
                    name = itm.get("name", "").strip()
                    desc = itm.get("description")
                    addr = itm.get("address")
                if not name:
                    continue
                key = name.lower()
                bucket = buckets.setdefault(key, _RestaurantBucket(name))
                bucket.add(source, desc, addr)
        return buckets

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _list_to_str(self, maybe_list: List[str] | str | None) -> str:
        if maybe_list is None:
            return ""
        return ", ".join(maybe_list) if isinstance(maybe_list, list) else maybe_list

    # ------------------------------------------------------------------
    def _tokens(self, txt: str) -> int:
        return len(self._enc.encode(txt))

    # ------------------------------------------------------------------
    def _postprocess_response(self, response, city: str) -> Dict[str, Any]:
        """Same logic as before – untouched."""
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            dump_chain_state("analyze_raw_response", {"raw_response": content[:1000]})
            results: Dict[str, Any] = json.loads(content)

            if "restaurants" in results and "main_list" not in results:
                results["main_list"] = results.pop("restaurants")

            results.setdefault("main_list", [])
            results.setdefault("hidden_gems", [])

            for rec in results["main_list"] + results["hidden_gems"]:
                rec.setdefault("price_range", "€€")
                rec.setdefault("recommended_dishes", [])
                rec.setdefault("missing_info", [])
                rec["city"] = city

            self._save_restaurants_to_db(results["main_list"] + results["hidden_gems"], city)

            if not (results["main_list"] or results["hidden_gems"]):
                results["main_list"] = [
                    {
                        "name": "Поиск не дал результатов",
                        "address": "Адрес недоступен",
                        "description": "К сожалению, мы не смогли найти рестораны, соответствующие вашему запросу.",
                        "sources": ["Системное сообщение"],
                        "price_range": "€€",
                        "recommended_dishes": [],
                        "missing_info": [],
                        "city": city,
                    }
                ]
            dump_chain_state(
                "analyze_final_results",
                {
                    "main_list_count": len(results["main_list"]),
                    "hidden_gems_count": len(results["hidden_gems"]),
                },
            )
            return results
        except (json.JSONDecodeError, AttributeError) as exc:
            dump_chain_state(
                "analyze_json_error",
                {"error": str(exc), "response_preview": getattr(response, "content", "")[:500]},
            )
            return {
                "main_list": [
                    {
                        "name": "Ошибка обработки результатов",
                        "address": "Адрес недоступен",
                        "description": "Произошла ошибка при обработке результатов поиска.",
                        "sources": ["Системное сообщение"],
                        "price_range": "€€",
                        "recommended_dishes": [],
                        "missing_info": [],
                        "city": city,
                    }
                ],
                "hidden_gems": [],
            }

    # ------------------------------------------------------------------
    def _extract_city(self, primary_parameters, destination=None):
        if destination and destination != "Unknown":
            return destination
        if isinstance(primary_parameters, list):
            for p in primary_parameters:
                low = p.lower()
                for marker in ("in ", "at ", "near "):
                    if marker in low:
                        return low.split(marker)[1].strip()
        return "unknown_location"

    # ------------------------------------------------------------------
    def _save_restaurants_to_db(self, restaurants: List[Dict[str, Any]], city: str):
        try:
            table = f"restaurants_{city.lower().replace(' ', '_')}"
            for r in restaurants:
                r["timestamp"] = time.time()
                r["id"] = f"{r['name']}_{r['address']}".lower().replace(" ", "_")
                save_data(table, r, self.config)
            print(f"Saved {len(restaurants)} restaurants to table {table}")
        except Exception as exc:
            print(f"Error saving restaurants to database: {exc}")
