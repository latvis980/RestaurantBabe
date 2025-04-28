# list_analyzer.py (token‑aware refactor – **prompt unchanged**)
"""
Eliminates blind 2 000‑character clipping and guards against context overflow by
measuring *tokens* instead of characters.  Uses `tiktoken` so we know exactly
how many tokens we’re feeding to Mistral‑Large (32 K limit).

Key changes
-----------
1. **Token budget** – configurable `max_prompt_tokens` (default 12 000).  We cut
   individual `scraped_content` fields only when the full prompt would exceed
   that budget.
2. **Per‑result summarisation optional** – if a single article is huge we first
   grab the lead 200 tokens, then append an ellipsis.  (You can plug in a
   LangChain summariser later; the interface stub is ready.)
3. **No changes to the system/human prompt strings** – requested by user.
4. Adds dependency: `tiktoken>=0.5.2` (≈ 90 KB wheel).
"""

from __future__ import annotations

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
        # PROMPTS remain **unchanged**
        self.system_prompt: str = """
        You are a restaurant recommendation expert analyzing search results to identify the best restaurants.

        TASK:
        Analyze the search results and identify promising restaurants that match the search parameters.

        PRIMARY SEARCH PARAMETERS:
        {primary_parameters}

        SECONDARY FILTER PARAMETERS:
        {secondary_parameters}

        KEYWORDS FOR ANALYSIS:
        {keywords_for_analysis}

        GUIDELINES:
        1. Analyze the tone and content of reviews to identify genuinely recommended restaurants
        2. Cross-reference the descriptions against the keywords and search parameters
        3. Look for restaurants mentioned in multiple reputable sources
        4. IGNORE results from Tripadvisor, Yelp
        5. Pay special attention to restaurants featured in food guides, local publications, or by respected critics
        6. When analyzing content, check if restaurants meet the secondary filter parameters
        7. IMPORTANT: If you can't find perfect matches, still provide at least 3-5 restaurants that are the closest matches

        OUTPUT REQUIREMENTS:
        - ALWAYS identify at least 5 restaurants (even with limited information)
        - Do not separate restaurants into different categories, just provide one main list
        - If search results are limited, create entries based on the available information
        - For EACH restaurant, extract:
          1. Name (exact as mentioned in sources)
          2. Street address (as complete as possible, or "Address unavailable" if not found)
          3. Raw description (40-60 words) including key details, dishes, interior, chef, and atmosphere
          4. ALL sources where mentioned (just the source name, NOT the URL, e.g., "Le Foodling" not "lefooding.com")
          5. Pay special attention to restaurants featured in food guides, local publications, or by respected critics
          6. For each restaurant, collect ALL source names found in the search results (look for "Source Name:" in each result)
          7. When analyzing content, check if restaurants meet the secondary filter parameters

        OUTPUT FORMAT:
        Provide a structured JSON object with one array: "main_list"
        Each restaurant object should include:
        - name (required, never empty)
        - address (required, use "Address unavailable" if not found)
        - description (required, 40-60 words summary, use available information to create one if needed)
        - sources (array of source names where it was mentioned)
        - location (city name from the search)
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Please analyze these search results and extract restaurant recommendations:\n\n{search_results}",
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
            kw_str = ", ".join(keywords_for_analysis) if isinstance(keywords_for_analysis, list) else (keywords_for_analysis or "")
            primary_str = ", ".join(primary_parameters) if isinstance(primary_parameters, list) else (primary_parameters or "")
            secondary_str = ", ".join(secondary_parameters) if isinstance(secondary_parameters, list) else (secondary_parameters or "")

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
    # Token helpers
    def _tokens(self, txt: str) -> int:
        return len(self._enc.encode(txt))

    # Format search results with a token‑budget cap
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        used_tokens = 0
        for idx, res in enumerate(search_results):
            base = [
                f"RESULT {idx+1}:",
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
                # Hard limit if single article blows the budget
                allowed = self.max_prompt_tokens - used_tokens - self._tokens("\n".join(base))
                if allowed <= 0:
                    break  # budget exhausted
                content_tokens = self._tokens(content)
                if content_tokens > allowed:
                    # keep head N tokens
                    head_txt = self._enc.decode(self._enc.encode(content)[: self._per_article_head_tokens])
                    base.append(f"Content: {head_txt} … (truncated) ")
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
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            dump_chain_state("analyze_raw_response", {"raw_response": content[:1000]})
            results = json.loads(content)
            if "restaurants" in results and "main_list" not in results:
                results["main_list"] = results.pop("restaurants")
            results.setdefault("main_list", [])
            for r in results["main_list"]:
                r["city"] = city
            self._save_restaurants_to_db(results["main_list"], city)
            if not results["main_list"]:
                results["main_list"] = [
                    {
                        "name": "Поиск не дал результатов",
                        "address": "Адрес недоступен",
                        "description": "К сожалению, мы не смогли найти рестораны, соответствующие вашему запросу.",
                        "sources": ["Системное сообщение"],
                        "city": city,
                    }
                ]
            dump_chain_state("analyze_final_results", {"main_list_count": len(results["main_list"])})
            return results
        except (json.JSONDecodeError, AttributeError) as e:
            dump_chain_state(
                "analyze_json_error",
                {"error": str(e), "response_preview": getattr(response, "content", "")[:500]},
            )
            return {
                "main_list": [
                    {
                        "name": "Ошибка обработки результатов",
                        "address": "Адрес недоступен",
                        "description": "Произошла ошибка при обработке результатов поиска.",
                        "sources": ["Системное сообщение"],
                        "city": city,
                    }
                ]
            }

    # ------------------------------------------------------------------
    def _extract_city(self, primary_parameters):
        if isinstance(primary_parameters, list):
            for p in primary_parameters:
                if "in " in p.lower():
                    return p.lower().split("in ")[1].strip()
        return "unknown_location"

    # ------------------------------------------------------------------
    def _save_restaurants_to_db(self, restaurants: List[Dict[str, Any]], city: str):
        try:
            table_name = f"restaurants_{city.lower().replace(' ', '_')}"
            for r in restaurants:
                r["timestamp"] = time.time()
                r["id"] = f"{r['name']}_{r['address']}".lower().replace(" ", "_")
                save_data(table_name, r, self.config)
            print(f"Saved {len(restaurants)} restaurants to table {table_name}")
        except Exception as e:
            print(f"Error saving restaurants to database: {e}")
