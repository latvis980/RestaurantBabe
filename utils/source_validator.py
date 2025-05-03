from __future__ import annotations
"""
utils/source_validator.py  v3.0
------------------------------

* Single‑source of truth for **domain reputation**, backed by:
  • `functools.lru_cache` (process‑lifetime)
  • a TTL‑controlled in‑memory cache (fast refresh, 24 h / 7 d)
  • the existing Mongo/Postgre table (long‑term)
* Eliminates the duplicate `evaluate_source_quality` definition that used to
  shadow the heuristic version.
* Public API preserved: `check_source_reputation()`, `evaluate_source_quality()`,
  `preload_source_reputations()`, `store_source_evaluation()`.
"""
from urllib.parse import urlparse
import aiohttp
import random
import time
from functools import lru_cache
from typing import Optional

from utils.database import (
    find_data,
    save_data,
    update_data,
    find_all_data,
)
from utils.async_utils import sync_to_async
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# Constants & in‑memory caches
# ---------------------------------------------------------------------------

# Fast‑refresh cache – survives for the lifetime of the Python process &
# supports TTL so domains can be re‑checked periodically.
_DOMAIN_CACHE: dict[str, tuple[bool, float]] = {}

# Long‑lived list of obviously reputable guides – skip the AI step entirely.
REPUTABLE_GUIDES = [
    "theworlds50best.com",
    "worldofmouth.app",
    "guide.michelin.com",
    "culinarybackstreets.com",
    "oadguides.com",
    "laliste.com",
    "eater.com",
    "bonappetit.com",
    "foodandwine.com",
    "infatuation.com",
    "nytimes.com"
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_domain(url_or_domain: str) -> str:
    """Return a lower‑case domain without leading *www.* and trailing slash."""
    domain = urlparse(url_or_domain).netloc or url_or_domain
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.rstrip("/")


async def _fetch_quick_preview(url: str, max_bytes: int = 5000) -> Optional[str]:
    """Grab the first *max_bytes* of the page so the LLM has some context."""
    headers = {
        "User-Agent": random.choice(
            [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            ]
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=6) as resp:
                if resp.status != 200:
                    return None
                ctype = resp.headers.get("Content-Type", "")
                encoding = None
                if "charset=" in ctype:
                    encoding = ctype.split("charset=")[1].split(";")[0].strip()
                try:
                    text = await resp.text(encoding=encoding) if encoding else await resp.text()
                except UnicodeDecodeError:
                    raw = await resp.read()
                    try:
                        text = raw.decode("latin-1")
                    except Exception:
                        text = raw.decode("utf-8", errors="ignore")
                return text[:max_bytes]
    except Exception as exc:
        print(f"[_fetch_quick_preview] {exc}")
    return None


# ---------------------------------------------------------------------------
# Core reputation logic – cached aggressively with LRU
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _domain_reputation(domain: str, config) -> Optional[bool]:
    """Return *True/False* if known, otherwise *None* (meaning unknown)."""
    # 1) Known guide → True
    if any(guide in domain for guide in REPUTABLE_GUIDES):
        return True

    # 2) DB look‑up (long‑term store)
    rec = find_data(config.DB_TABLE_SOURCES, {"domain": domain}, config)
    if rec:
        return rec.get("is_reputable", False)

    # 3) Unknown
    return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def check_source_reputation(url: str, config) -> Optional[bool]:
    """Lightweight sync helper used by crawlers before deciding to scrape."""
    domain = _normalise_domain(url)

    # 1) short‑term cache (24 h)
    if domain in _DOMAIN_CACHE:
        is_rep, ts = _DOMAIN_CACHE[domain]
        if time.time() - ts < 86_400:
            return is_rep

    # 2) lru‑cached DB / whitelist check
    rep = _domain_reputation(domain, config)
    if rep is not None:
        _DOMAIN_CACHE[domain] = (rep, time.time())
    return rep


def preload_source_reputations(config) -> int:
    """Load the first 1000 rows into the fast in‑memory cache so startup is quiet."""
    try:
        rows = find_all_data(config.DB_TABLE_SOURCES, {}, config, limit=1000)
        for row in rows:
            if "domain" in row and "is_reputable" in row:
                _DOMAIN_CACHE[_normalise_domain(row["domain"])] = (
                    row["is_reputable"],
                    time.time(),
                )
        print(f"[source_validator] Preloaded {len(rows)} reputation rows")
        return len(rows)
    except Exception as exc:
        print(f"[source_validator] preload error: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Slow path — called only when we truly don’t know a domain
# ---------------------------------------------------------------------------

async def _ai_evaluate_source(url: str, preview: str, config) -> bool:
    """Ask an LLM if this looks like a professional, trustworthy restaurant source."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    system_msg = SystemMessage(
        content="""
You are an expert in media credibility for restaurant information.
Respond with a single word – "yes" if the site is reputable or "no" otherwise.
        """,
    )
    human_msg = HumanMessage(
        content=f"URL: {url}\n\nPreview:\n{preview[:1500]}\n\nIs this source reputable?",
    )
    try:
        resp = await llm.ainvoke([system_msg, human_msg])
        txt = getattr(resp, "content", str(resp)).lower()
        return "yes" in txt and "no" not in txt
    except Exception as exc:
        print(f"[_ai_evaluate_source] {exc}")
        return False


def store_source_evaluation(url: str, is_reputable: bool, config):
    """Persist the outcome and update both caches."""
    domain = _normalise_domain(url)
    now = time.time()

    row = {
        "domain": domain,
        "full_url": url,
        "is_reputable": is_reputable,
        "evaluated_at": now,
        "evaluation_count": 1,
    }

    _DOMAIN_CACHE[domain] = (is_reputable, now)

    existing = find_data(config.DB_TABLE_SOURCES, {"domain": domain}, config)
    if existing:
        row["evaluation_count"] = existing.get("evaluation_count", 0) + 1
        update_data(config.DB_TABLE_SOURCES, {"domain": domain}, row, config)
    else:
        save_data(config.DB_TABLE_SOURCES, row, config)


# ---------------------------------------------------------------------------
# Public async evaluation (slow path wrapper)
# ---------------------------------------------------------------------------

@sync_to_async  # keep signature compatible with legacy callers
async def evaluate_source_quality(url: str, config) -> bool:
    """Full pipeline returning a boolean; will cache & persist the result."""
    domain = _normalise_domain(url)

    # 1) Quick exit if we already know
    cached = check_source_reputation(url, config)
    if cached is not None:
        return cached

    # 2) Quick preview + heuristic (cheap)
    preview = await _fetch_quick_preview(url)
    if not preview:
        return False

    # Minimal heuristic: long‑form content earns +0.1, .com/.org earns +0.05
    score = 0.5
    if len(preview) > 5000:
        score += 0.1
    if domain.endswith((".com", ".org")):
        score += 0.05
    if any(g in domain for g in REPUTABLE_GUIDES):
        score = 1.0

    if score >= 0.95:
        is_rep = True
    elif score <= 0.55:
        is_rep = False
    else:
        # 3) Ask the LLM (expensive, only for borderline cases)
        is_rep = await _ai_evaluate_source(url, preview, config)

    # 4) Persist & cache
    store_source_evaluation(url, is_rep, config)
    return is_rep
