from __future__ import annotations

"""Token‑aware ListAnalyzer – now adds *price_range*, *recommended_dishes* and
splits out *hidden_gems* (places with few mentions but rave reviews from a
credible source).

* Hidden‑gem heuristic (simple): restaurant appears in **≤ 2** total sources but
  at least **one** of those sources is tagged as reputable (guide / major
  publication / critic) **and** the article tone contains superlatives such as
  “best”, “brilliant”, “exceptional”, etc.  (We rely on the LLM to detect this
  in the prompt.)

* Main prompt section changed accordingly; output format now contains
  `main_list` **and** `hidden_gems` arrays, and each restaurant additionally
  carries `price_range` ("€", "€€", or "€€€") and `recommended_dishes` (≤ 3
  dishes).

Rest of the code – token budgeting, saving to DB – remains mostly intact.
"""

import json
import time
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_mistralai import ChatMistralAI
from tiktoken import get_encoding

from utils.database import save_data
from utils.debug_utils import dump_chain_state, log_function_call


class ListAnalyzer:
    def __init__(
        self,
        config,
        *,
        max_prompt_tokens: int = 12_000,
        encoding_name: str = "cl100k_base",
        per_article_head_tokens: int = 200,
    ) -> None:
        # --- LLM -------------------------------------------------------
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.8,
            mistral_api_key=config.MISTRAL_API_KEY,
        )

        # ------------------------------------------------------------------
        # PROMPT
        # ------------------------------------------------------------------
        self.system_prompt: str = """
        You are a restaurant recommendation expert analysing search‑results to
        identify the best restaurants **and promising hidden gems**.

        TASK:
        From the supplied search snippets produce two lists:
        1. **main_list** – establishments widely endorsed by multiple reputable
           sources.
        2. **hidden_gems** – places mentioned in ≤ 2 sources *but* featuring an
           enthusiastic review from a knowledgeable critic or respected medium.

        PRIMARY SEARCH PARAMETERS:
        {primary_parameters}

        SECONDARY FILTER PARAMETERS:
        {secondary_parameters}

        KEYWORDS FOR ANALYSIS:
        {keywords_for_analysis}

        GUIDELINES:
        1. Determine sentiment & credibility of each article.
        2. Count how many distinct sources mention the place.
        3. Treat Michelin Guide, World’s 50 Best, The Guardian, Eater, New York
           Times, Condé Nast Traveler, World of Mouth, OAD, La Liste, national
           broadsheets and well‑known local food magazines as *credible*.
        4. **Ignore Tripadvisor, Yelp and Google user reviews.**
        5. When extracting data, also look for:
           • Typical price indicator (cheap € to expensive €€€).
           • 2‑3 signature dishes (look for phrases like “don’t miss…”, “must‑try…”, etc.).
        6. Mark a restaurant as hidden gem if it appears in ≤ 2 sources *and* at
           least one of those uses strong positive language ("outstanding",
           "brilliant", "game‑changing", etc.).
        7. If overall matches are scarce, still return at least 5 across both
           lists.

        OUTPUT FORMAT:
        Return JSON with two arrays `main_list` and `hidden_gems`.
        Each restaurant object must include:
          • `name` – never empty
          • `address` – full street or "Address unavailable"
          • `description` – 40‑60 words
          • `price_range` – "€", "€€" or "€€€"; guess if absent
          • `recommended_dishes` – array up to 3 items (empty if unknown)
          • `sources` – array of *names* (not URLs) where it was mentioned
          • `location` – city extracted from the query parameters
        """

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

        # --- token budgeting -----------------------------------------
        self.max_prompt_tokens = max_prompt_tokens
        self._enc = get_encoding(encoding_name)
        self._per_article_head_tokens = per_article_head_tokens

        self.config = config

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    @log_function_call
    def analyze(
        self,
        search_results: List[Dict[str, Any]],
        keywords_for_analysis: List[str] | str,
        primary_parameters: Optional[List[str] | str] = None,
        secondary_parameters: Optional[List[str] | str] = None,
    ) -> Dict[str, Any]:
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            dump_chain_state(
                "analyze_start",
                {
                    "search_results_count": len(search_results),
                    "keywords": keywords_for_analysis,
                    "primary_parameters": primary_parameters,
                    "secondary_parameters": secondary_parameters,
                },
            )

            city = self._extract_city(primary_parameters)
            formatted_results = self._format_search_results(search_results)
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

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        used_tokens = 0
        for idx, res in enumerate(search_results):
            base: List[str] = [
                f"RESULT {idx + 1}:",
                f"Title: {res.get('title', 'Unknown')}",
                f"URL: {res.get('url', 'Unknown')}",
                f"Source: {res.get('source_domain', 'Unknown')}",
            ]
            if res.get("source_name"):
                base.append(f"Source Name: {res['source_name']}")
            if res.get("description"):
                base.append(f"Description: {res['description']}")

            content = res.get("scraped_content", "")
            if content:
                allowed = self.max_prompt_tokens - used_tokens - self._tokens("\n".join(base))
                if allowed <= 0:
                    break
                content_tokens = self._tokens(content)
                if content_tokens > allowed:
                    head_txt = self._enc.decode(
                        self._enc.encode(content)[: self._per_article_head_tokens]
                    )
                    base.append(f"Content: {head_txt} … (truncated)")
                    used_tokens += self._tokens("\n".join(base))
                else:
                    base.append(f"Content: {content}")
                    used_tokens += content_tokens + self._tokens("\n".join(base[:-1]))
            else:
                used_tokens += self._tokens("\n".join(base))
            lines.append("\n".join(base))
            if used_tokens >= self.max_prompt_tokens:
                break
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    def _postprocess_response(self, response, city: str) -> Dict[str, Any]:
        """Parse the LLM output, normalise keys, add safe defaults and persist."""
        try:
            # ── 1. strip markdown fences ─────────────────────────────────────────
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            dump_chain_state("analyze_raw_response", {"raw_response": content[:1000]})
            results: Dict[str, Any] = json.loads(content)

            # ── 2. normalise keys ────────────────────────────────────────────────
            if "restaurants" in results and "main_list" not in results:
                results["main_list"] = results.pop("restaurants")

            results.setdefault("main_list", [])
            results.setdefault("hidden_gems", [])

            # ── 3. add mandatory defaults to every record ────────────────────────
            for rec in results["main_list"] + results["hidden_gems"]:
                rec.setdefault("price_range", "€€")
                rec.setdefault("recommended_dishes", [])
                rec.setdefault("missing_info", [])
                rec["city"] = city

            # ── 4. write to DB ───────────────────────────────────────────────────
            self._save_restaurants_to_db(
                results["main_list"] + results["hidden_gems"], city
            )

            # ── 5. graceful fallback if everything is empty ──────────────────────
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

        # ── 6. JSON / attribute errors ──────────────────────────────────────────
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
    def _extract_city(self, primary_parameters):
        if isinstance(primary_parameters, list):
            for p in primary_parameters:
                if "in " in p.lower():
                    return p.lower().split("in ")[1].strip()
        return "unknown_location"

    def _save_restaurants_to_db(self, restaurants: List[Dict[str, Any]], city: str):
        try:
            table = f"restaurants_{city.lower().replace(' ', '_') }"
            for r in restaurants:
                r["timestamp"] = time.time()
                r["id"] = f"{r['name']}_{r['address']}".lower().replace(" ", "_")
                save_data(table, r, self.config)
            print(f"Saved {len(restaurants)} restaurants to table {table}")
        except Exception as exc:
            print(f"Error saving restaurants to database: {exc}")
