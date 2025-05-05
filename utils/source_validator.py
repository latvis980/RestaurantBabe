# utils/source_validator.py
import logging
import json
import os
import time
import random
from urllib.parse import urlparse
from typing import Dict, Any, Optional

import aiohttp
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Remove database import since we don't use the sources table anymore
# from utils.database import find_data, save_data
from utils.async_utils import sync_to_async

logger = logging.getLogger("restaurant-recommender.source_validator")

# Simple in-memory cache with TTL
_DOMAIN_CACHE: Dict[str, tuple[bool, float]] = {}
_SOURCE_REPUTATION_CACHE = {}

def _normalize_domain(url: str) -> str:
    """Extract and normalize domain from URL"""
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.rstrip("/")

def preload_source_reputations(config):
    """Load source reputation data from file"""
    global _SOURCE_REPUTATION_CACHE

    try:
        # Check for source reputation file
        source_file = os.path.join(os.path.dirname(__file__), "../data/source_reputation.json")
        if os.path.exists(source_file):
            with open(source_file, 'r') as f:
                _SOURCE_REPUTATION_CACHE = json.load(f)
                logger.info(f"Preloaded reputation data for {len(_SOURCE_REPUTATION_CACHE)} sources")
        else:
            # Initialize with a few known high-quality sources
            _SOURCE_REPUTATION_CACHE = {
                "eater.com": {"score": 0.9, "reason": "Professional food publication"},
                "bonappetit.com": {"score": 0.9, "reason": "Professional food publication"},
                "guide.michelin.com": {"score": 1.0, "reason": "Premier restaurant guide"},
                "theworlds50best.com": {"score": 1.0, "reason": "Restaurant ranking authority"},
                "nytimes.com": {"score": 0.9, "reason": "Major newspaper with food section"},
                "theguardian.com": {"score": 0.9, "reason": "Major newspaper with food section"},
                "culinarybackstreets.com": {"score": 0.9, "reason": "Specialized food guide"}
            }
            logger.info(f"Initialized default reputation data for {len(_SOURCE_REPUTATION_CACHE)} sources")
    except Exception as e:
        logger.error(f"Error preloading source reputations: {e}")

def validate_source(domain: str, config) -> Dict[str, Any]:
    """
    Validate if a source is reputable - synchronous version

    Args:
        domain: Domain to validate
        config: Application config

    Returns:
        Dict with validation results:
            - is_valid: Boolean indicating if source is valid
            - reputation_score: Float score from 0.0 to 1.0
            - reason: String describing reason for validation result
    """
    # Normalize the domain
    if domain.startswith("http"):
        domain = _normalize_domain(domain)

    # Check excluded domains from config
    if any(excluded in domain for excluded in config.EXCLUDED_RESTAURANT_SOURCES):
        return {
            "is_valid": False,
            "reputation_score": 0.0,
            "reason": "Domain is in excluded list"
        }

    # For testing purposes or emergencies, bypass all checks
    if getattr(config, 'BYPASS_REPUTATION_CHECK', False):
        return {
            "is_valid": True,
            "reputation_score": 1.0,
            "reason": "Reputation check bypassed by config"
        }

    # Check in-memory cache first (24h TTL)
    if domain in _DOMAIN_CACHE:
        is_reputable, timestamp = _DOMAIN_CACHE[domain]
        if time.time() - timestamp < 86400:  # 24 hours
            return {
                "is_valid": is_reputable,
                "reputation_score": 0.8 if is_reputable else 0.2,
                "reason": "Based on cached evaluation"
            }

    # Check preloaded reputation cache
    if domain in _SOURCE_REPUTATION_CACHE:
        reputation = _SOURCE_REPUTATION_CACHE[domain]
        is_valid = reputation.get("score", 0) >= 0.5
        return {
            "is_valid": is_valid,
            "reputation_score": reputation.get("score", 0.5),
            "reason": reputation.get("reason", "Based on preloaded reputation data")
        }

    # REMOVED: Database check since the sources table no longer exists

    # Use a simple heuristic for initial validation without calling LLM
    # This is a fast pre-filter before using AI evaluation
    reputable_keywords = [
        "guide", "critic", "food", "restaurant", "dining", "culinary", 
        "chef", "cuisine", "magazine", "news", "review", "taste", "travel"
    ]

    spam_keywords = [
        "hotel", "booking", "clickbait", "coupon", "discount", "promo",
        "deal", "cheap", "offer", "casino", "realestate", "price"
    ]

    domain_parts = domain.split('.')
    name_parts = []
    if len(domain_parts) >= 2:
        name_parts = domain_parts[0].split('-')

    reputation_score = 0.5  # Neutral starting point

    # Check for reputable keywords in domain
    for keyword in reputable_keywords:
        if keyword in domain_parts[0]:
            reputation_score += 0.05

    # Check for spam keywords in domain
    for keyword in spam_keywords:
        if keyword in domain_parts[0]:
            reputation_score -= 0.1

    # Adjust for common TLDs
    if domain.endswith('.com'):
        pass  # Neutral
    elif domain.endswith('.org') or domain.endswith('.edu'):
        reputation_score += 0.1
    elif domain.endswith('.net'):
        reputation_score += 0.05
    elif any(domain.endswith(tld) for tld in ['.gov', '.edu']):
        reputation_score += 0.2
    elif domain.endswith('.info'):
        reputation_score -= 0.05

    # Cap the score between 0.1 and 0.9
    reputation_score = max(0.1, min(0.9, reputation_score))

    # For new domains, store in cache with a neutral/positive bias for exploration
    is_valid = reputation_score >= 0.4  # Lower threshold to allow more results for AI to evaluate
    _DOMAIN_CACHE[domain] = (is_valid, time.time())

    return {
        "is_valid": is_valid,
        "reputation_score": reputation_score,
        "reason": "Quick heuristic validation - pending AI evaluation"
    }

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
        logger.error(f"Error fetching preview for {url}: {e}")
        return None

async def _ai_evaluate_source(url: str, content: str, config) -> bool:
    """Use AI to evaluate if a source is reputable for restaurant information"""
    try:
        # Initialize the LLM
        model = ChatOpenAI(
            model="gpt-4o",  # Using GPT-4o as requested
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
        - social media platforms with user-generated content (like Facebook, Instagram)

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
        logger.error(f"Error in AI evaluation for {url}: {e}")
        # Default to True in case of evaluation error
        return True

def store_source_evaluation(url: str, is_reputable: bool) -> None:
    """Store source evaluation in cache"""
    domain = _normalize_domain(url)

    # Update memory cache
    _DOMAIN_CACHE[domain] = (is_reputable, time.time())

    # Update reputation cache
    reputation_score = 0.8 if is_reputable else 0.2
    _SOURCE_REPUTATION_CACHE[domain] = {
        "score": reputation_score,
        "reason": f"AI-based evaluation: {'reputable' if is_reputable else 'not reputable'}"
    }

    logger.info(f"Source evaluation stored for {domain}: {'reputable' if is_reputable else 'not reputable'}")

    # Optionally save to file
    try:
        source_file = os.path.join(os.path.dirname(__file__), "../data/source_reputation.json")
        os.makedirs(os.path.dirname(source_file), exist_ok=True)
        with open(source_file, 'w') as f:
            json.dump(_SOURCE_REPUTATION_CACHE, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving source reputation data: {e}")

def update_source_reputation(domain: str, score: float, reason: str = None):
    """
    Update the reputation score for a domain - used for admin interfaces

    Args:
        domain: Domain to update
        score: New reputation score
        reason: Reason for the score
    """
    global _SOURCE_REPUTATION_CACHE

    _SOURCE_REPUTATION_CACHE[domain] = {
        "score": score,
        "reason": reason or "Manual update"
    }

    # Optionally save to file
    try:
        source_file = os.path.join(os.path.dirname(__file__), "../data/source_reputation.json")
        os.makedirs(os.path.dirname(source_file), exist_ok=True)
        with open(source_file, 'w') as f:
            json.dump(_SOURCE_REPUTATION_CACHE, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving source reputation data: {e}")

@sync_to_async
async def evaluate_source_quality(url: str, config) -> Dict[str, Any]:
    """
    Evaluate source quality using AI - asynchronous version

    Args:
        url: URL to evaluate
        config: Application config

    Returns:
        Dict with validation results
    """
    domain = _normalize_domain(url)

    # Quick check on excluded domains
    if any(excluded in domain for excluded in config.EXCLUDED_RESTAURANT_SOURCES):
        return {
            "is_valid": False,
            "reputation_score": 0.0,
            "reason": "Domain is in excluded list"
        }

    # For testing purposes or emergencies, bypass all checks
    if getattr(config, 'BYPASS_REPUTATION_CHECK', False):
        return {
            "is_valid": True,
            "reputation_score": 1.0,
            "reason": "Reputation check bypassed by config"
        }

    # Check in-memory cache first (24h TTL)
    if domain in _DOMAIN_CACHE:
        is_reputable, timestamp = _DOMAIN_CACHE[domain]
        if time.time() - timestamp < 86400:  # 24 hours
            return {
                "is_valid": is_reputable,
                "reputation_score": 0.8 if is_reputable else 0.2,
                "reason": "Based on cached evaluation"
            }

    # REMOVED: Database check since the sources table no longer exists

    # For unknown sources, fetch content and evaluate
    content = await _fetch_quick_preview(url)

    # If we can't fetch content, default to accepting with moderate confidence
    if not content:
        store_source_evaluation(url, True)
        return {
            "is_valid": True,
            "reputation_score": 0.6,
            "reason": "Could not fetch content, defaulting to accept"
        }

    # Use AI to evaluate the source
    is_reputable = await _ai_evaluate_source(url, content, config)

    # Store the result
    store_source_evaluation(url, is_reputable)

    return {
        "is_valid": is_reputable,
        "reputation_score": 0.8 if is_reputable else 0.2,
        "reason": "Based on AI evaluation"
    }