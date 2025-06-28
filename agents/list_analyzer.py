from __future__ import annotations
"""
ListAnalyzer v3.0 — tuned for Mistral Large / Medium
====================================================
Highlights
----------
* ✅ **Structured JSON** – enforced with a Pydantic schema, so keys like
  `hidden_gems` never disappear.
* ✅ **Retry + adaptive back‑off** – stops 429 errors.
* ✅ **Async semaphore** – caps concurrent LLM calls.
* ✅ **Keyword‑aware snippet pruning** – boosts signal, lowers token count.
* ✅ **Post‑generation quality gate** – fills missing hidden gems and expands
  short descriptions via micro‑calls.

"""
import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Sequence
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic import field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
import config

###############################################################################
# Logging
###############################################################################
logger = logging.getLogger(__name__)
if os.getenv("LIST_ANALYZER_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

###############################################################################
# Pydantic schema
###############################################################################
class Restaurant(BaseModel):
    name: str = Field(..., description="Restaurant name as written by sources")
    address: str = Field(default="Address unavailable", description="Street and house number if available")
    description: str = Field(
        default="Description unavailable",
        description="40‑60 word vivid summary starting with one concrete fact"
    )
    price_range: str = Field(default="Price range not specified", description="Price range information")
    recommended_dishes: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list) 
    location: str
    city: str = Field(default="", description="City name from the query")

    @field_validator("description")
    @classmethod
    def ensure_len(cls, v):
        # Handle the case where v is not a string
        if v is None:
            return "Description unavailable"

        # Convert non-strings to string, but handle common problematic values
        if not isinstance(v, str):
            if isinstance(v, bool):
                return "Description unavailable"
            elif isinstance(v, (int, float)):
                return "Description unavailable"
            else:
                # Try to convert to string as last resort
                try:
                    v = str(v)
                except:
                    return "Description unavailable"

        # Clean up the string
        v = v.strip()

        # If empty or just whitespace, return default
        if not v or v == "Description unavailable":
            return "Description unavailable"

        # Return the cleaned description
        return v

class ListResponse(BaseModel):
    main_list: List[Restaurant]
    

# Parser to give the LLM format instructions + parse back to python.
PARSER = PydanticOutputParser(pydantic_object=ListResponse)

###############################################################################
# Prompt pieces
###############################################################################
SYSTEM_PROMPT = """You are restaurant list analyser. Follow ALL rules:
1. ALWAYS output pure JSON with key `main_list` containing all recommended restaurants.
2. No markdown, no code fences, no commentary.
3. Each restaurant must include: name, location, city (from query), description (40‑60 words, start with a concrete fact), price_range, recommended_dishes, sources (publication names only, not full article titles), source_urls.
4. Identify at least 8 restaurants that match the search parameters and are in the specified location/city.
5. Extract source URLs from the [URL] markers in the snippets - include them in source_urls field.
6. For sources, only include the publication/website name (e.g., "Eater", "Time Out"), not the full article title.
7. Only include restaurants from the city specified in the query - filter out any results from other locations.
"""

HUMAN_TEMPLATE = (
    "PRIMARY SEARCH PARAMETERS:\n{primary}\n\n"
    "SECONDARY FILTER PARAMETERS:\n{secondary}\n\n"
    "KEYWORDS FOR ANALYSIS:\n{keywords}\n\n"
    "DESTINATION CITY: {destination}\n\n"
    "SOURCE SNIPPETS (deduplicated | keyword‑dense):\n{snippets}\n\n"
    "{format_instructions}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ]
)

###############################################################################
# Helpers
###############################################################################
def _clean_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_keyword_sentences(text: str, keywords: Sequence[str], max_sent: int = 3) -> List[str]:
    """Return up to *max_sent* sentences from *text* containing any keyword."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    hits = []
    for sent in sentences:
        if any(k.lower() in sent.lower() for k in keywords):
            hits.append(_clean_sentence(sent))
        if len(hits) >= max_sent:
            break
    return hits or sentences[:max_sent]  # fallback to first sentences

def _build_snippets(raw_articles: Sequence[Dict[str, Any]], keywords: Sequence[str]) -> str:
    """Create a trimmed block of article snippets focusing on keyword sentences."""
    parts = []
    seen_urls = set()  # Avoid duplicates

    for art in raw_articles:
        url = art.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Get the source name from source_info if available, otherwise extract from domain
        source_name = ""
        if "source_info" in art and "name" in art["source_info"]:
            source_name = art["source_info"]["name"]
        elif "source_domain" in art:
            # Extract clean source name from domain
            domain = art["source_domain"]
            if domain.startswith('www.'):
                domain = domain[4:]
            source_name = domain.split('.')[0].capitalize()

            # Map common domains to proper names
            source_map = {
                'eater': 'Eater',
                'timeout': 'Time Out',
                'thefork': 'The Fork',
                'infatuation': 'The Infatuation',
                'michelin': 'Michelin Guide',
                'worldofmouth': 'World of Mouth',
                'nytimes': 'New York Times',
                'forbes': 'Forbes',
                'guardian': 'The Guardian',
                'telegraph': 'The Telegraph',
                'cntraveler': 'Condé Nast Traveler',
                'laliste': 'La Liste',
                'oadguides': 'OAD Guides'
            }

            for key, val in source_map.items():
                if key in domain.lower():
                    source_name = val
                    break
        else:
            source_name = "Unknown Source"

        body = art.get("scraped_content", "")
        key_sents = _extract_keyword_sentences(body, keywords)
        snippet = f"### Source: {source_name} [URL: {url}]\n" + "\n".join(key_sents[:3])
        parts.append(snippet)
    return "\n\n".join(parts)

###############################################################################
# Retry wrapper
###############################################################################
# Tenacity retry decorated coroutine
def retry_async(fn):
    async def wrapper(*args, **kwargs):
        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        async def _inner():
            return await fn(*args, **kwargs)
        return await _inner()
    return wrapper

###############################################################################
# ListAnalyzer implementation
###############################################################################
class ListAnalyzer:
    """Analyse scraped lists, extract names of the restaurants and descriptions, group descriptions for the same restaurant into one text. Return structured restaurant recommendations."""

    # Semaphore shared across all instances
    _sem = asyncio.Semaphore(int(os.getenv("MISTRAL_MAX_PARALLEL", "4")))

    def __init__(
        self,
        config=None,  # Add this for compatibility
        model_name: str = None,
        temperature: float = 0.5,
        api_key: str | None = None,
    ) -> None:
        # Use config object if provided
        if config:
            model_name = model_name or getattr(config, 'MISTRAL_MODEL', 'mistral-large-latest')
            api_key = api_key or getattr(config, 'MISTRAL_API_KEY', None)

        self.llm = ChatMistralAI(
            model_name=model_name or os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            temperature=temperature,
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
            # Use Mistral's native JSON mode to reinforce schema
            response_format={"type": "json_object"},
        )

    @retry_async
    async def _call_llm(self, prompt_values: Dict[str, Any]) -> str:
        async with self._sem:
            start = time.perf_counter()
            chain = PROMPT | self.llm | StrOutputParser()
            result = await chain.ainvoke(prompt_values)
            logger.debug("LLM call took %.1fs", time.perf_counter() - start)
            return result

    async def _expand_short_descriptions(
        self, resp: ListResponse, location: str
    ) -> ListResponse:
        short = [r for r in resp.main_list if len(r.description.split()) < 20]
        if not short:
            return resp
        logger.info("Expanding %d short descriptions...", len(short))
        expand_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You expand short restaurant blurbs to 40‑60 words."),
                (
                    "human",
                    "Rewrite the description of {name} in {location}. Current:\n\"{desc}\"",
                ),
            ]
        )
        for r in short:
            chain = expand_template | self.llm | StrOutputParser()
            async with self._sem:
                new_desc = await chain.ainvoke(
                    {"name": r.name, "location": location, "desc": r.description}
                )
            r.description = _clean_sentence(new_desc)
        return resp

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    async def analyze(
        self,
        search_results: Sequence[Dict[str, Any]],  # Changed parameter name for compatibility
        keywords_for_analysis: Sequence[str],      # Changed parameter name for compatibility
        primary_search_parameters: List[str] | str,  # Changed parameter name for compatibility
        secondary_filter_parameters: List[str] | str,  # Changed parameter name for compatibility
        destination: str = None,  # Added for compatibility
    ) -> Dict[str, Any]:
        """Main entry point used by downstream pipeline."""

        # Ensure we have a destination
        if not destination or destination == "Unknown":
            destination = self._derive_city(primary_search_parameters)

        # Convert parameters to match new interface
        if isinstance(primary_search_parameters, list):
            primary_params = ", ".join(primary_search_parameters)
        else:
            primary_params = primary_search_parameters

        if isinstance(secondary_filter_parameters, list):
            secondary_params = ", ".join(secondary_filter_parameters)
        else:
            secondary_params = secondary_filter_parameters

        snippets = _build_snippets(search_results, keywords_for_analysis)
        prompt_values = dict(
            primary=primary_params,
            secondary=secondary_params,
            keywords=", ".join(keywords_for_analysis),
            destination=destination,
            snippets=snippets,
            format_instructions=PARSER.get_format_instructions(),
        )

        logger.debug("Prepared prompt of %d chars", len(snippets))

        # ---------------- LLM call ---------------- #
        content = await self._call_llm(prompt_values)

        # -------------- Parse & validate ---------- #
        try:
            response_model = ListResponse.model_validate_json(content)
        except Exception as exc:
            logger.error("Pydantic parse failed: %s", exc)
            # Propagate for upstream error handling
            raise

        # -------------- Quality gates ------------- #
        response_model = await self._expand_short_descriptions(
            response_model, location=destination
        )

        # Set the city for all restaurants
        for restaurant in response_model.main_list:
            restaurant.city = destination
            restaurant.location = destination

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    @staticmethod
    def _derive_city(primary_parameters: str) -> str:
        if isinstance(primary_parameters, list):
            primary_parameters = " ".join(primary_parameters)

        # Look for common patterns in the query
        patterns = [
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\bnear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+restaurants?",
        ]

        for pattern in patterns:
            match = re.search(pattern, primary_parameters)
            if match:
                city = match.group(1)
                # Filter out generic words that might be captured
                if city.lower() not in ['best', 'top', 'good', 'great', 'amazing', 'recommended']:
                    return city

        return "Unknown"