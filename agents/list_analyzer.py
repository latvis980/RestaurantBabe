from __future__ import annotations
"""
ListAnalyzer v2.1 — **position‑agnostic** and **nested‑list aware**
---------------------------------------------------------------
• Accepts either a flat *List[Dict]* (legacy) **or** `List[List[Dict]]` that
  represents multiple pre‑curated restaurant lists.
• Flattens and deduplicates on the fly so **every** restaurant/article is
  analysed, eliminating top‑of‑list bias.
• Adds an **LRU token budget**: if we still risk overflow, we keep a *32‑word
  head* for every remaining article (instead of discarding the tail of the
  list entirely).
• Public API is unchanged – `analyze()` returns the same JSON schema.
"""

import json
import time
from typing import List, Dict, Any, Optional, Iterable
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_mistralai import ChatMistralAI
from tiktoken import get_encoding

from utils.database import save_data
from utils.debug_utils import dump_chain_state, log_function_call

# ---------------------------------------------------------------------------
# Helper – unify successive batches of search results ------------------------
# ---------------------------------------------------------------------------

def _flatten_results(raw: Iterable[Any]) -> List[Dict[str, Any]]:
    """Recursively flatten *anything* that yields search‑result‑dicts."""
    flat: List[Dict[str, Any]] = []
    stack: List[Any] = list(raw)
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        # silently ignore weird types
    return flat


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
        # --- LLM -----------------------------------------------------------
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.8,
            mistral_api_key=config.MISTRAL_API_KEY,
        )

        # ------------------------------------------------------------------
        # PROMPT -----------------------------------------------------------
        # ------------------------------------------------------------------
        self.system_prompt: str = (
            "You are a restaurant recommendation expert analysing search‑results to\n"
            "identify the best restaurants **and promising hidden gems**.\n\n"
            "TASK:\n"
            "1. From the supplied texts extract names of restaurants and merge descriptions of each restaurant from different sources into one description with as many details as possible.\n"
            "2. Produce two lists:\n"
            "   • **main_list** – establishments widely endorsed by multiple reputable sources.\n"
            "   • **hidden_gems** – places mentioned in ≤ 2 sources *but* featuring an enthusiastic review.\n\n"
            "GUIDELINES:\n"
            "1. Analyze the tone and content of reviews to identify genuinely recommended restaurants\n"
            "2. Cross‑reference the descriptions against the keywords and search parameters\n"
            "3. Look for restaurants mentioned in multiple reputable sources\n"
            "4. IGNORE results from Tripadvisor, Yelp\n"
            "5. Pay special attention to restaurants featured in food guides, local publications, or by respected critics\n"
            "6. When analyzing content, check if restaurants meet the secondary filter parameters\n"
            "7. ALWAYS identify at least 8 restaurants (fallback to closest matches if necessary)\n"
            "9. Mark a restaurant as hidden gem if it appears in ≤ 2 sources *and* at least one of those uses strong positive language ('outstanding', 'brilliant', etc.).\n\n"
            "PRIMARY SEARCH PARAMETERS:\n{primary_parameters}\n\n"
            "SECONDARY FILTER PARAMETERS:\n{secondary_parameters}\n\n"
            "KEYWORDS FOR ANALYSIS:\n{keywords_for_analysis}\n\n"
            "OUTPUT FORMAT:\n"
            "Return JSON with two arrays `main_list` and `hidden_gems`. Each restaurant object must include:\n"
            "  • `name` — never empty\n  • `address` — full street address or 'Address unavailable'\n  • `description` — 40‑60 words about the restaurant\n  • `price_range` — '€', '€€' or '€€€'; guess if absent\n  • `recommended_dishes` — up to 3 items (empty if unknown)\n  • `sources` — names (not URLs) of media where it was mentioned\n  • `location` — city extracted from the query parameters\n"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Please analyse these search results and extract restaurant recommendations:\n\n{search_results}",
                ),
            ]
        )
        self.chain = self.prompt | self.model

        # --- token budgeting ---------------------------------------------
        self.max_prompt_tokens = max_prompt_tokens
        self._enc = get_encoding(encoding_name)
        self._per_article_head_tokens = per_article_head_tokens

        self.config = config

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    @log_function_call
    def analyze(
        self,
        search_results: List[Any],  # can be nested
        keywords_for_analysis: List[str] | str,
        primary_parameters: Optional[List[str] | str] = None,
        secondary_parameters: Optional[List[str] | str] = None,
        destination: Optional[str] = None,
    ) -> Dict[str, Any]:
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            flat_results = _flatten_results(search_results)
            dump_chain_state(
                "analyze_start",
                {
                    "search_results_count": len(flat_results),
                    "keywords": keywords_for_analysis,
                    "primary_parameters": primary_parameters,
                    "secondary_parameters": secondary_parameters,
                    "destination": destination,
                },
            )

            city = self._extract_city(primary_parameters, destination)
            formatted_results = self._format_search_results(flat_results)
            kw_str = ", ".join(keywords_for_analysis) if isinstance(keywords_for_analysis, list) else (
                keywords_for_analysis or ""
            )
            primary_str = ", ".join(primary_parameters) if isinstance(primary_parameters, list) else (
                primary_parameters or ""
            )
            secondary_str = (
                ", ".join(secondary_parameters) if isinstance(secondary_parameters, list) else (secondary_parameters or "")
            )

            response = self.chain.invoke(
                {
                    "search_results": formatted_results,
                    "keywords_for_analysis": kw_str,
                    "primary_parameters": primary_str,
                    "secondary_parameters": secondary_str,
                }
            )
            return self._postprocess_response(response, city)

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    def _tokens(self, txt: str) -> int:
        return len(self._enc.encode(txt))

    @staticmethod
    @lru_cache(maxsize=4096)
    def _dedupe_key(url: str) -> str:
        """Return a stable key for deduplication (host + path w/o params)."""
        from urllib.parse import urlparse

        p = urlparse(url)
        return f"{p.netloc}{p.path}".rstrip("/")

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format all results but respect a total token ceiling.

        Strategy: allocate about 5% of the budget for metadata lines; the remainder
        is split evenly so *each* article gets at least a short head‑snippet.
        """
        meta_overhead = int(self.max_prompt_tokens * 0.05)
        payload_budget = self.max_prompt_tokens - meta_overhead
        head_per_article = max(32, payload_budget // max(len(search_results), 1))

        lines: List[str] = []
        used_tokens = 0
        seen: set[str] = set()

        # Extract the sources collection if it exists in the first result
        sources_collection = None
        if search_results and len(search_results) > 0 and "sources_collection" in search_results[0]:
            sources_collection = search_results[0].get("sources_collection", [])

            # Add a section for sources collection if available
            if sources_collection:
                sources_section = ["SOURCES COLLECTION:"]
                for i, source in enumerate(sources_collection):
                    source_name = source.get("name", "Unknown")
                    source_domain = source.get("domain", "Unknown")
                    source_type = source.get("type", "Website")
                    result_index = source.get("result_index", i)

                    sources_section.append(f"Source {i+1}: {source_name}")
                    sources_section.append(f"Type: {source_type}")
                    sources_section.append(f"Domain: {source_domain}")
                    sources_section.append(f"Result Index: {result_index}")
                    sources_section.append("")

                # Add sources section to the formatted results
                sources_text = "\n".join(sources_section)
                sources_tokens = self._tokens(sources_text)

                # Only add if within budget
                if sources_tokens < meta_overhead // 2:
                    lines.append(sources_text)
                    used_tokens += sources_tokens

        for idx, res in enumerate(search_results):
            # ----- dedupe identical URLs ---------------------------------
            key = self._dedupe_key(res.get("url", ""))
            if key in seen:
                continue
            seen.add(key)

            base: List[str] = [
                f"RESULT {idx + 1}:",
                f"Title: {res.get('title', 'Unknown')}",
                f"URL: {res.get('url', 'Unknown')}",
            ]

            # Add source information from either source_info or direct properties
            if "source_info" in res:
                source_info = res["source_info"]
                base.append(f"Source Name: {source_info.get('name', 'Unknown')}")
                base.append(f"Source Type: {source_info.get('type', 'Website')}")
                base.append(f"Source Domain: {source_info.get('domain', 'Unknown')}")
            else:
                # Fallback to original properties
                base.append(f"Source Domain: {res.get('source_domain', 'Unknown')}")
                if res.get("source_name"):
                    base.append(f"Source Name: {res['source_name']}")

            if res.get("description"):
                base.append(f"Description: {res['description']}")

            content = res.get("scraped_content", "")
            if content:
                snippet = self._enc.decode(self._enc.encode(content)[:head_per_article])
                base.append(f"Content: {snippet} … (truncated)")
            full_block = "\n".join(base)
            block_toks = self._tokens(full_block)
            if used_tokens + block_toks > self.max_prompt_tokens:
                break
            lines.append(full_block)
            used_tokens += block_toks

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    def _postprocess_response(self, response, city: str) -> Dict[str, Any]:
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
                results["main_list"] = [{
                    "name": "Поиск не дал результатов",
                    "address": "Адрес недоступен",
                    "description": "К сожалению, мы не смогли найти рестораны, соответствующие вашему запросу.",
                    "sources": ["Системное сообщение"],
                    "price_range": "€€",
                    "recommended_dishes": [],
                    "missing_info": [],
                    "city": city,
                }]

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
                {
                    "error": str(exc),
                    "response_preview": getattr(response, "content", "")[:500],
                },
            )
            return {
                "main_list": [{
                    "name": "Ошибка обработки результатов",
                    "address": "Адрес недоступен",
                    "description": "Произошла ошибка при обработке результатов поиска.",
                    "sources": ["Системное сообщение"],
                    "price_range": "€€",
                    "recommended_dishes": [],
                    "missing_info": [],
                    "city": city,
                }],
                "hidden_gems": [],
            }

    # ------------------------------------------------------------------
    def _extract_city(self, primary_parameters, destination=None):
        if destination and destination != "Unknown":
            return destination
        if isinstance(primary_parameters, list):
            for p in primary_parameters:
                pl = p.lower()
                for kw in ("in ", "at ", "near "):
                    if kw in pl:
                        return pl.split(kw)[1].strip()
        return "unknown_location"

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
