from __future__ import annotations

import asyncio
import random
import time
import logging
import brotli  # Added missing import
from typing import List, Dict, Any, Optional

import httpx
from bs4 import BeautifulSoup
from diskcache import Cache
from langchain_core.tracers.context import tracing_v2_enabled
from readability import Document
from urllib.parse import urlparse
from utils.source_validator import check_source_reputation, evaluate_source_quality

# Global set for tracking pending tasks
_PENDING_TASKS: set[asyncio.Task] = set()

# Define logger at module level
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        concurrency: int = 5,
        html_ttl: int = 60 * 60 * 24 * 30,  # 30 days
        cache_dir: str = ".web_cache",
    ) -> None:
        """
        Initialize the scraper. The AsyncClient is *not* created here so that
        it is always bound to the event-loop that actually runs the scraping.

        Args:
            config: Configuration dictionary
            concurrency: Maximum number of concurrent requests
            html_ttl: Time-to-live for cached HTML (in seconds)
            cache_dir: Directory to store cached HTML
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]

        # Defer AsyncClient creation until we're inside an event loop
        self._client: httpx.AsyncClient | None = None

        # Limit parallel requests
        self._sem = asyncio.Semaphore(concurrency)

        # HTML disk cache
        self._html_cache = Cache(cache_dir)
        self._html_ttl = html_ttl

    @property
    def DEFAULT_HEADERS(self):
        """Default HTTP headers for requests"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

    # ------------------------------------------------------------------
    # Public orchestrator ------------------------------------------------
    async def filter_and_scrape_results(
        self,
        search_results: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        1. Discard search hits from untrusted sources.
        2. Fetch HTML for each remaining page concurrently.
        3. Attach quality / reputation metadata and return the enriched list.

        Args:
            search_results: List of search result dictionaries
            max_retries: Maximum number of retry attempts for failed requests

        Returns:
            List of enriched search results with HTML content and quality scores
        """
        # Create a client bound to THIS event-loop and close it before we exit.
        async with httpx.AsyncClient(http2=True, timeout=httpx.Timeout(10.0)) as client:
            self._client = client  # used by _fetch_html()

            # ── Step 1: filter by reputation ────────────────────────────────
            vetted: List[Dict[str, Any]] = []
            for result in search_results:
                url = result.get("url") or result.get("href")
                if not url or not check_source_reputation(url, self.config):  # Fixed: Added self.config
                    continue
                vetted.append(result)

            # ── Step 2 + 3: scrape & enrich in parallel ────────────────────
            async def _enrich(item: Dict[str, Any]) -> Dict[str, Any]:
                html = await self._fetch_html(
                    item["url"],
                    max_retries=max_retries,
                )
                item["html"] = html or ""
                item["scraped_content"] = self._extract_clean_text(html or "", item["url"])[0] if html else ""

                # Try to get quality score using the function signature
                try:
                    item["quality_score"] = evaluate_source_quality(item["url"], html or "")
                except TypeError:
                    # If first attempt fails, try with config
                    try:
                        item["quality_score"] = evaluate_source_quality(item["url"], html or "", self.config)
                    except Exception as e:
                        self.logger.warning(f"Failed to evaluate quality for {item['url']}: {e}")
                        item["quality_score"] = 0.5  # Default score if evaluation fails

                return item

            # _fetch_html already respects self._sem, so plain gather is fine
            tasks = [_enrich(r) for r in vetted]
            enriched_results = await asyncio.gather(*tasks)

        # Client is closed here; nothing will try to touch a dead event-loop.
        self._client = None
        return enriched_results

    # Keep external‑facing API compatible with old sync call ----------------
    def scrape_search_results(
        self, search_results: List[Dict[str, Any]], max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Scrape and enrich search results, handling async execution.
        This method provides a synchronous interface to the async scraping process.

        Args:
            search_results: List of search result dictionaries
            max_retries: Maximum number of retry attempts for failed requests

        Returns:
            List of enriched search results
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet – safe to run
            return asyncio.run(self.filter_and_scrape_results(search_results, max_retries))

        if loop.is_running():
            # Fire‑and‑forget; return shallow copy so callers are not blocked
            copy = [r.copy() for r in search_results]
            t = loop.create_task(self.filter_and_scrape_results(search_results, max_retries))
            _PENDING_TASKS.add(t)
            t.add_done_callback(_PENDING_TASKS.discard)
            self.logger.warning("Returning before async scraping completes.")
            return copy
        # Loop exists but not running – rare
        return loop.run_until_complete(
            self.filter_and_scrape_results(search_results, max_retries)
        )

    # ------------------------------------------------------------------
    # Fetch helpers -----------------------------------------------------
    async def _fetch_html(self, url: str, max_retries: int = 3) -> str | None:
        """
        Download, *fully* decompress, and pre-clean HTML.
        Returns Unicode HTML or None on hard failure.

        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            HTML content as string, or None if fetching fails
        """
        if cached := self._html_cache.get(url):
            return cached

        for attempt in range(1, max_retries + 1):
            try:
                async with self._sem:  # Respect concurrency limits
                    # FIX: Use proper await instead of async context manager
                    r = await self._client.get(url, headers=self.DEFAULT_HEADERS, timeout=20)
                    if r.status_code != 200:
                        raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)

                    # ---- 1. make sure the bytes are Unicode ----
                    encoding = r.headers.get("content-encoding")
                    if encoding == "br":
                        # httpx hands us raw bytes when brotli isn't auto-handled
                        html_bytes = brotli.decompress(await r.aread())
                        html = html_bytes.decode("utf-8", "replace")
                    elif encoding in ("gzip", "deflate", None, ""):
                        html = await r.atext()  # FIX: Use atext() instead of text()
                    else:
                        # rare encodings – grab raw and hope chardet gets it right
                        html = (await r.aread()).decode("utf-8", "replace")

                    # ---- 2. run Readability to strip boilerplate ----
                    main_doc = Document(html)
                    cleaned_html = main_doc.summary() or html   # fallback if Readability fails

                    # ---- 3. (optional) quick tag sanitise via BS ----
                    soup = BeautifulSoup(cleaned_html, "lxml")
                    # kill script / style tags that sometimes survive Readability
                    for bad in soup(["script", "style", "noscript"]):
                        bad.decompose()

                    final_html = str(soup)

                    # ---- 4. cache & go ----
                    self._html_cache.set(url, final_html, expire=self._html_ttl)
                    return final_html

            except Exception as e:
                self.logger.warning("Fetch %s failed (try %d/%d): %s", url, attempt, max_retries, e)
                await asyncio.sleep(1.5 * attempt)  # simple back-off

        self.logger.error("Giving up on %s after %d attempts", url, max_retries)
        return None

    # ------------------------------------------------------------------
    # Content cleaning ---------------------------------------------------
    def _extract_clean_text(self, html: str, url: str) -> tuple[str, str]:
        """Return cleaned article text and a prettified source name.

        Strategy
        --------
        1. Try python-readability (`Document`) to isolate the main HTML.
        2. If the result is < 500 chars or any error occurs, fall back to simply
           concatenating all <p> tags (or the whole page text as a last resort).
        3. Convert the domain to a human-friendly source name with
           `_format_source_name`.
        4. If a title was captured, prepend it to the text separated by a blank
           line.

        Args:
            html: HTML content to extract text from
            url: URL of the page (for source name extraction)

        Returns:
            tuple: (cleaned text, source name)
        """
        try:
            # ── readability first ───────────────────────────────────────────
            doc = Document(html)
            summary_html = doc.summary()            # main article HTML
            title = doc.short_title()               # may be None
            text = BeautifulSoup(
                summary_html, "html.parser"
            ).get_text(" ", strip=True)

            # fallback if content still too short
            if len(text) < 500:
                raise ValueError("Readability result too short")

        except Exception:
            # ── naive extraction fallback ──────────────────────────────────
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = " ".join(
                p.get_text(" ", strip=True) for p in soup.find_all("p")
            )
            text = paragraphs if paragraphs else soup.get_text(" ", strip=True)
            title = None

        # ── prettify the domain to a source name ────────────────────────────
        source_name = self._format_source_name(urlparse(url).netloc)

        # ── prepend title if we have one ────────────────────────────────────
        if title:
            text = f"{title}\n\n{text}"

        return text, source_name

    # ------------------------------------------------------------------
    # Utility methods ----------------------------------------------------
    @staticmethod
    def _should_skip_url(url: str) -> bool:
        """
        Check if a URL should be skipped based on file extension

        Args:
            url: URL to check

        Returns:
            bool: True if URL should be skipped, False otherwise
        """
        non_html_ext = (
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".zip",
            ".mp3",
            ".mp4",
        )
        return url.lower().endswith(non_html_ext)

    @staticmethod
    def _format_source_name(domain: str) -> str:
        """
        Format a domain into a prettified source name

        Args:
            domain: Domain to format

        Returns:
            str: Prettified source name
        """
        clean = domain.replace("www.", "").split(".")[0]
        pretty = " ".join(word.capitalize() for word in clean.replace("-", " ").split())
        domain_map = {
            "michelin": "Michelin Guide",
            "foodandwine": "Food & Wine",
            "eater": "Eater",
            "zagat": "Zagat",
            "infatuation": "The Infatuation",
            "50best": "World's 50 Best",
            "worlds50best": "World's 50 Best",
            "worldofmouth": "World of Mouth",
            "nytimes": "New York Times",
            "washingtonpost": "Washington Post",
            "timeout": "Time Out",
            "bonappetit": "Bon Appétit",
            "saveur": "Saveur",
            "foodrepublic": "Food Republic",
            "epicurious": "Epicurious",
            "seriouseats": "Serious Eats",
            "forbes": "Forbes",
            "thrillist": "Thrillist",
            "gq": "GQ",
            "vogue": "Vogue",
            "esquire": "Esquire",
            "telegraph": "The Telegraph",
            "guardian": "The Guardian",
            "independent": "The Independent",
            "finedininglovers": "Fine Dining Lovers",
            "oadguides": "OAD Guides",
            "laliste": "La Liste",
            "culinarybackstreets": "Culinary Backstreets",
            "cntraveler": "Condé Nast Traveler",
        }
        for key, val in domain_map.items():
            if key in domain:
                return val
        return pretty

    # ------------------------------------------------------------------
    async def aclose(self):
        """Close the scraper's resources"""
        if self._client:
            await self._client.aclose()
        self._html_cache.close()

    # Context‑manager sugar so you can use `async with WebScraper(...)`
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()