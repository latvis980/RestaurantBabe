# async_web_scraper.py
"""
Async + Readability + Disk cache refactor of your original WebScraper.

Key design choices
------------------
1. **httpx.AsyncClient** with a semaphore (default 5) so we hit many URLs in parallel
   while staying polite.
2. **DiskCache** (https://pypi.org/project/diskcache/) – an on‑disk, TTL‑aware
   key‑value store.  *HTML* is cached for 30 days by default; you can adjust via
   `html_ttl` in the constructor.
3. **Mozilla Readability** (readability‑lxml) to extract the main article body and
   title.  We fall back to the old BeautifulSoup logic if Readability doesn’t
   find enough text (> 500 chars).
4. The external source‑reputation utilities you already have are kept intact;
   the scraper only worries about downloading & cleaning.
5. Public API is still **scrape_search_results()** so you don’t need to touch the
   downstream LangChain code.

Install requirements
--------------------
```bash
pip install httpx[http2] readability-lxml diskcache beautifulsoup4
```
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import List, Dict, Any, Optional

import httpx
from bs4 import BeautifulSoup
from diskcache import Cache
from langchain_core.tracers.context import tracing_v2_enabled
from readability import Document
from urllib.parse import urlparse

# --- utils --------------------------------------------------------------

_PENDING_TASKS: set[asyncio.Task] = set()

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}


# -----------------------------------------------------------------------
class WebScraper:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        concurrency: int = 5,
        html_ttl: int = 60 * 60 * 24 * 30,  # 30 days
        cache_dir: str = ".web_cache",
    ) -> None:
        self.config = config
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        self._client = httpx.AsyncClient(http2=True, timeout=httpx.Timeout(10.0))
        self._sem = asyncio.Semaphore(concurrency)
        self._html_cache = Cache(cache_dir)
        self._html_ttl = html_ttl

    # ------------------------------------------------------------------
    # Public orchestrator ------------------------------------------------
    async def filter_and_scrape_results(
        self, search_results: List[Dict[str, Any]], max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """Filter by reputation (caller already vets some URLs) and scrape."""

        from utils.source_validator import check_source_reputation, evaluate_source_quality

        enriched_results: List[Dict[str, Any]] = []

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # — Step 1 : group by reputation status so we can run tasks concurrently
            reputation_buckets: list[tuple[int, str, Optional[bool]]] = []
            for i, result in enumerate(search_results):
                url = result.get("url")
                if not url or self._should_skip_url(url):
                    continue
                rep = check_source_reputation(url, self.config)
                reputation_buckets.append((i, url, rep))

            async def scrape_if_ok(idx: int, url: str) -> None:
                nonlocal enriched_results
                result = search_results[idx]
                html = await self._fetch_html(url, max_retries)
                if not html:
                    return
                clean_text, source_name = self._extract_clean_text(html, url)
                domain = urlparse(url).netloc
                title = result.get("title", "Unknown Title")
                source_prefix = (
                    f"SOURCE: {domain}\nTITLE: {title}\nURL: {url}\n\nCONTENT:\n"
                )
                result["scraped_content"] = source_prefix + clean_text
                result["source_domain"] = domain
                result["source_name"] = source_name
                enriched_results.append(result)

            tasks: list[asyncio.Task] = []
            # 1 ) process known reputable first (fast path)
            for idx, url, rep in reputation_buckets:
                if rep is True:
                    tasks.append(asyncio.create_task(scrape_if_ok(idx, url)))

            await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()

            # 2 ) evaluate unknowns, then scrape if approved
            for idx, url, rep in reputation_buckets:
                if rep is not None:
                    continue  # already handled
                try:
                    approved = await evaluate_source_quality(url, self.config)
                except Exception as e:
                    print(f"Reputation check failed for {url}: {e}")
                    continue
                if not approved:
                    continue
                tasks.append(asyncio.create_task(scrape_if_ok(idx, url)))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return enriched_results

    # Keep external‑facing API compatible with old sync call ----------------
    def scrape_search_results(
        self, search_results: List[Dict[str, Any]], max_retries: int = 2
    ) -> List[Dict[str, Any]]:
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
            print("Warning: Returning before async scraping completes.")
            return copy
        # Loop exists but not running – rare
        return loop.run_until_complete(
            self.filter_and_scrape_results(search_results, max_retries)
        )

    # ------------------------------------------------------------------
    # Fetch helpers -----------------------------------------------------
    async def _fetch_html(self, url: str, max_retries: int) -> Optional[str]:
        """Return HTML from cache or network."""
        cached = self._html_cache.get(url)
        if cached:
            return cached

        headers = DEFAULT_HEADERS | {"User-Agent": random.choice(self.user_agents)}
        attempt = 0
        while attempt <= max_retries:
            try:
                async with self._sem:  # honour concurrency limit
                    r = await self._client.get(url, headers=headers)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith(
                    "text/html"
                ):
                    html = r.text
                    self._html_cache.set(url, html, expire=self._html_ttl)
                    return html
            except httpx.HTTPError as e:
                print(f"Attempt {attempt+1}/{max_retries+1} failed for {url}: {e}")
            attempt += 1
            if attempt <= max_retries:
                await asyncio.sleep(random.uniform(1.0, 3.0))
        return None

    # ------------------------------------------------------------------
    # Content cleaning ---------------------------------------------------
    def _extract_clean_text(self, html: str, url: str) -> tuple[str, str]:
        """Readability first; fallback to simple paragraph join."""
        try:
            doc = Document(html)
            summary_html = doc.summary()  # main content HTML
            title = doc.short_title()
            text = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
            if len(text) < 500:  # too short – fallback
                raise ValueError("Readability too short")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            text = paragraphs if paragraphs else soup.get_text(" ", strip=True)
            title = None
        source_name = self._format_source_name(urlparse(url).netloc)
        if title:
            text = f"{title}\n\n{text}"
        return text, source_name

    # ------------------------------------------------------------------
    # Utility methods ----------------------------------------------------
    @staticmethod
    def _should_skip_url(url: str) -> bool:
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
        await self._client.aclose()
        self._html_cache.close()

    # Context‑manager sugar so you can use `async with WebScraper(...)`
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()
