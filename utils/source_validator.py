# utils/source_validator.py
from urllib.parse import urlparse
import aiohttp
import random
import time
from typing import Optional, Dict, Any

from utils.database import find_data, save_data, update_data
from utils.async_utils import sync_to_async
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Simple in-memory cache with TTL
_DOMAIN_CACHE: Dict[str, tuple[bool, float]] = {}

def _normalize_domain(url: str) -> str:
    """Extract and normalize domain from URL"""
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.rstrip("/")

def check_source_reputation(url: str, config) -> bool:
    """
    Check if the source is reputable.
    First checks cache, then database, then defaults to accepting unknown domains.
    """
    # For testing purposes or emergencies, bypass all checks
    if getattr(config, 'BYPASS_REPUTATION_CHECK', False):
        return True

    domain = _normalize_domain(url)

    # Check in-memory cache first (24h TTL)
    if domain in _DOMAIN_CACHE:
        is_reputable, timestamp = _DOMAIN_CACHE[domain]
        if time.time() - timestamp < 86400:  # 24 hours
            return is_reputable

    # Check database
    try:
        record = find_data(config.DB_TABLE_SOURCES, {"domain": domain}, config)
        if record and "is_reputable" in record:
            # Update cache and return result
            _DOMAIN_CACHE[domain] = (record["is_reputable"], time.time())
            return record["is_reputable"]
    except Exception as e:
        print(f"Database error in check_source_reputation: {e}")

    # If not in cache or database, default to accepting
    # We'll evaluate it properly in the background
    return True

async def _fetch_quick_preview(url: str) -> Optional[str]:
    """Fetch a small preview of the webpage content"""
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        ]),
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    return None

                # Try to detect encoding
                content_type = response.headers.get('Content-Type', '')
                encoding = None
                if 'charset=' in content_type:
                    encoding = content_type.split('charset=')[1].split(';')[0].strip()

                # Read content with proper encoding if available
                try:
                    if encoding:
                        content = await response.text(encoding=encoding)
                    else:
                        content = await response.text()
                except UnicodeDecodeError:
                    # Fallback to binary read and manual decode
                    raw = await response.read()
                    try:
                        content = raw.decode('utf-8', errors='replace')
                    except Exception:
                        content = raw.decode('latin-1', errors='replace')

                # Return trimmed preview
                return content[:5000]
    except Exception as e:
        print(f"Error fetching preview for {url}: {e}")
        return None

async def _ai_evaluate_source(url: str, content: str, config) -> bool:
    """Use AI to evaluate if a source is reputable for restaurant information"""
    try:
        # Initialize the LLM
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Construct messages
        system_message = SystemMessage(content="""
        You are an expert at evaluating source credibility for restaurant information.
        Your task is to determine if a website is a reputable source for restaurant recommendations.

        INDICATIONS OF REPUTABLE SOURCES:
        - Professional food critics or publications
        - Local newspapers or city magazines with proper editorial standards
        - Food and travel blogs by established food writers
        - Industry publications with proper editorial standards
        - Culinary awards or recognition organizations
        - International or local food guides
        - City-specific food and dining websites with original content


        INDICATIONS OF NON-REPUTABLE SOURCES:
        - Generic travel sites with crowdsourced reviews (like TripAdvisor, Yelp)
        - Sites that primarily aggregate other reviews with minimal original content
        - SEO-optimized listicles with thin content and no original research
        - Content farm sites with generic recommendations
        - Sites with excessive advertisements and little editorial oversight
        
        Respond ONLY with "yes" for reputable sources or "no" for non-reputable sources.
        """)

        human_message = HumanMessage(content=f"""
        URL: {url}
        Domain: {_normalize_domain(url)}

        Content Preview:
        {content[:1500] if content else "No content available"}

        Is this a reputable source for restaurant recommendations? Answer only yes or no.
        """)

        # Invoke the model
        response = await model.ainvoke([system_message, human_message])

        # Process response
        response_text = response.content.lower().strip()
        return "yes" in response_text and "no" not in response_text
    except Exception as e:
        print(f"Error in AI evaluation for {url}: {e}")
        # Default to True in case of evaluation error
        return True

def store_source_evaluation(url: str, is_reputable: bool, config) -> None:
    """Store source evaluation in cache and database"""
    domain = _normalize_domain(url)

    # Update memory cache
    _DOMAIN_CACHE[domain] = (is_reputable, time.time())

    # Prepare data for storage
    source_data = {
        "domain": domain,
        "full_url": url,
        "is_reputable": is_reputable,
        "evaluated_at": time.time(),
        "evaluation_count": 1
    }

    # Check if already exists
    try:
        existing = find_data(config.DB_TABLE_SOURCES, {"domain": domain}, config)

        if existing:
            # Update existing record
            source_data["evaluation_count"] = existing.get("evaluation_count", 0) + 1
            update_data(config.DB_TABLE_SOURCES, {"domain": domain}, source_data, config)
        else:
            # Create new record
            save_data(config.DB_TABLE_SOURCES, source_data, config)

        print(f"Source evaluation stored for {domain}: {'reputable' if is_reputable else 'not reputable'}")
    except Exception as e:
        print(f"Error storing source evaluation: {e}")

def preload_source_reputations(config) -> int:
    """
    Simplified preload function that just initializes the system.
    No need to load everything upfront - we'll use the database as needed.
    """
    try:
        # Just log that we're starting up
        print("[source_validator] Initialized reputation system")
        return 0
    except Exception as e:
        print(f"[source_validator] preload error: {e}")
        return 0

@sync_to_async
async def evaluate_source_quality(url: str, config) -> bool:
    """
    Evaluate source quality using AI.
    This is the main function that should be called from other components.
    """
    domain = _normalize_domain(url)

    # Check cache and database first (fast path)
    cached_result = check_source_reputation(url, config)

    # For known sources, return immediately
    in_cache = domain in _DOMAIN_CACHE
    if in_cache:
        return cached_result

    # For unknown sources, fetch content and evaluate
    content = await _fetch_quick_preview(url)

    # If we can't fetch content, default to accepting
    if not content:
        store_source_evaluation(url, True, config)
        return True

    # Use AI to evaluate the source
    is_reputable = await _ai_evaluate_source(url, content, config)

    # Store the result
    store_source_evaluation(url, is_reputable, config)

    return is_reputable