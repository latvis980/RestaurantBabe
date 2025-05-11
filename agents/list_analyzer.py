
from __future__ import annotations

"""
ListAnalyzer v3.0 — tuned for Mistral Large / Medium
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
* Compatible with previous public API (`await ListAnalyzer().analyze(...)`).

This is drop‑in: replace your old *list_analyzer.py* with this file and
adjust the import path if needed.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Sequence

from langchain.chat_models import ChatMistralAI
from langchain.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential


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
    address: str = Field(..., description="Street and house number if available")
    description: str = Field(
        ...,
        description="40‑60 word vivid summary starting with one concrete fact"
    )
    price_range: str
    recommended_dishes: List[str] = Field(default_factory=list)
    sources: List[str]
    source_urls: List[str] = Field(default_factory=list) 
    location: str

    @validator("description")
    def ensure_len(cls, v):
        if len(v.split()) < 20:
            raise ValueError("Description too short")
        return v


class ListResponse(BaseModel):
    main_list: List[Restaurant]
    hidden_gems: List[Restaurant]


# Parser to give the LLM format instructions + parse back to python.
PARSER = PydanticOutputParser(pydantic_object=ListResponse)

###############################################################################
# Prompt pieces
###############################################################################

SYSTEM_PROMPT = """You are restaurant list analyser. Follow ALL rules:
1. ALWAYS output pure JSON with keys `main_list` with restaurants praised by multiple experts and `hidden_gems` highly recommended by one or two sources.
2. No markdown, no code fences, no commentary.
3. Each restaurant must include: name, location, description (40‑60 words, start with a concrete fact), price_range, recommended_dishes, sources.
4. Identify at least 8 restaurants that match the search parameters.
5. Extract source URLs from the [URL] markers in the snippets.
"""

HUMAN_TEMPLATE = (
    "PRIMARY SEARCH PARAMETERS:\n{primary}\n\n"
    "SECONDARY FILTER PARAMETERS:\n{secondary}\n\n"
    "KEYWORDS FOR ANALYSIS:\n{keywords}\n\n"
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
    for art in raw_articles:
        # Extract source info from your current format
        src_name = art.get("title", "") or art.get("source_domain", "unknown")
        src_url = art.get("url", "")
        body = art.get("scraped_content", "")

        key_sents = _extract_keyword_sentences(body, keywords)
        snippet = f"### {src_name} [{src_url}]\n" + "\n".join(key_sents[:3])
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
        model_name: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        temperature: float = 0.5,
        api_key: str | None = None,
    ) -> None:
        self.llm = ChatMistralAI(
            model_name=model_name,
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

    async def _ensure_hidden_gems(
        self, response: ListResponse
    ) -> ListResponse:
        """If hidden_gems empty, ask LLM to pick them from main_list."""
        if response.hidden_gems:
            return response
        logger.info("hidden_gems empty – running follow‑up selection")
        gems_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Select up to 3 hidden gems (appear in <=2 sources) from the list."),
                (
                    "human",
                    (
                        "Here is the list as JSON:\n{raw}\n\n"
                        "Return JSON with key `hidden_gems` only."
                    ),
                ),
            ]
        )
        follow_chain = gems_prompt | self.llm | StrOutputParser()
        raw = response.json()
        async with self._sem:
            gems_json = await follow_chain.ainvoke({"raw": raw})
        try:
            gems = ListResponse.model_validate_json(
                '{"main_list": [], "hidden_gems": ' + gems_json + "}"
            ).hidden_gems
            response.hidden_gems = gems
        except Exception as e:
            logger.warning("Failed to parse hidden gems follow‑up: %s", e)
        return response

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
        primary_parameters: str,
        secondary_parameters: str,
        keywords: Sequence[str],
        raw_articles: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Main entry point used by downstream pipeline."""

        snippets = _build_snippets(raw_articles, keywords)
        prompt_values = dict(
            primary=primary_parameters,
            secondary=secondary_parameters,
            keywords=", ".join(keywords),
            snippets=snippets,
            format_instructions=PARSER.get_format_instructions(),
        )

        logger.debug("Prepared prompt of %d chars", len(snippets))

        # ---------------- LLM call ---------------- #
        content = await self._call_llm(prompt_values)

        # -------------- Parse & validate ---------- #
        try:
            response_model = PARSER.parse(content)
        except Exception as exc:
            logger.error("Pydantic parse failed: %s", exc)
            # Propagate for upstream error handling
            raise

        # -------------- Quality gates ------------- #
        response_model = await self._ensure_hidden_gems(response_model)
        response_model = await self._expand_short_descriptions(
            response_model, location=self._derive_city(primary_parameters)
        )

        return response_model.model_dump()

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #

    @staticmethod
    def _derive_city(primary_parameters: str) -> str:
        match = re.search(r"\b(in|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", primary_parameters)
        return match.group(2) if match else ""
