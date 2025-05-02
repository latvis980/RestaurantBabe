# utils/source_validator.py with improved caching
from urllib.parse import urlparse
import aiohttp
import random
import time
from utils.database import find_data, save_data, update_data, find_all_data
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.async_utils import sync_to_async, track_async_task

# In-memory cache to avoid repeated database lookups in the same session
_DOMAIN_CACHE = {}  # Domain -> (is_reputable, timestamp)

async def fetch_quick_preview(url):
    """Fetch a small preview of the page content for analysis"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    # Try to get the encoding from the response
                    content_type = response.headers.get('Content-Type', '')
                    encoding = None

                    # Extract charset from Content-Type if available
                    if 'charset=' in content_type:
                        encoding = content_type.split('charset=')[-1].strip()

                    try:
                        # Try with the specified encoding first, if available
                        if encoding:
                            content_sample = await response.text(encoding=encoding)
                        else:
                            content_sample = await response.text(encoding='utf-8')
                    except UnicodeDecodeError:
                        # Fallback to reading raw bytes and using a more liberal decoder
                        raw_content = await response.read()
                        try:
                            # Try with 'latin-1' which accepts any byte value
                            content_sample = raw_content.decode('latin-1')
                        except:
                            # Last resort: ignore problematic characters
                            content_sample = raw_content.decode('utf-8', errors='ignore')

                    return content_sample[:5000]  # Return first 5000 chars
        return None
    except Exception as e:
        print(f"Error in quick preview fetch: {e}")
        return None

def evaluate_source_quality(url, html_content):
    """
    Evaluate the quality of a source based on its URL and content

    Args:
        url (str): The URL of the source
        html_content (str): The HTML content of the source

    Returns:
        float: A quality score between 0 and 1
    """
    domain = urlparse(url).netloc

    # Check for known high-quality domains
    reputable_guides = [
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

    for guide in reputable_guides:
        if guide in domain:
            return 1.0  # Maximum score for known reputable guides

    # Basic heuristics for other sources
    score = 0.5  # Default score

    # Length-based heuristic (longer content often means more detailed reviews)
    if len(html_content) > 5000:
        score += 0.1

    # Domain-based heuristics (well-established domains often have editorial standards)
    if domain.endswith(".com") or domain.endswith(".org"):
        score += 0.05

    # Cap at 0.95 for sources that aren't in our known reputable list
    return min(score, 0.95)

def check_source_reputation(url, config):
    """Check if a source is already in our reputation database or in-memory cache"""
    # Extract and normalize domain from URL
    domain = urlparse(url).netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]  # Remove www. prefix

    # Normalize common domains to avoid duplicates
    domain_map = {
        'guide.michelin.com': 'michelin.com',
        'lexpress.fr': 'lexpress.fr',
        'timeout.fr': 'timeout.fr',
        # Add more mappings as needed
    }

    # Apply mapping if domain is in the map
    if domain in domain_map:
        domain = domain_map[domain]

    # Add logging to see what domains we're checking
    print(f"Checking reputation for normalized domain: {domain}")

    # First check in-memory cache with longer TTL (1 week)
    if domain in _DOMAIN_CACHE:
        is_reputable, timestamp = _DOMAIN_CACHE[domain]
        if time.time() - timestamp < 604800:  # 604800 seconds = 1 week
            print(f"Using cached reputation for {domain}: {is_reputable}")
            return is_reputable

    # Check if domain exists in known sources database
    result = find_data(
        config.DB_TABLE_SOURCES,
        {"domain": domain},
        config
    )

    if result:
        # Store result in cache
        _DOMAIN_CACHE[domain] = (result.get("is_reputable", False), time.time())

        # Return the stored reputation
        return result.get("is_reputable", False)

    # Not in database or cache, need AI verification
    return None

def preload_source_reputations(config):
    """Preload known source reputations into memory cache"""
    try:
        print("Preloading source reputations...")
        # Get all source records (up to 1000)
        sources = find_all_data(
            config.DB_TABLE_SOURCES,
            {},  # Empty query to get all records
            config,
            limit=1000
        )

        # Load into memory cache (with domain normalization)
        count = 0
        for source in sources:
            if "domain" in source and "is_reputable" in source:
                # Normalize domain to avoid duplicates (remove www. and trailing slash)
                domain = source["domain"].replace("www.", "").rstrip("/")
                _DOMAIN_CACHE[domain] = (source["is_reputable"], time.time())
                count += 1

        print(f"Preloaded {count} source reputation records")
        return count
    except Exception as e:
        print(f"Error preloading source reputations: {e}")
        return 0

async def ai_evaluate_source(content_sample, url, config):
    """Use AI to evaluate if a source is reputable"""
    # Extract domain for caching
    domain = urlparse(url).netloc

    # Check cache again (might have been updated by another process)
    if domain in _DOMAIN_CACHE:
        is_reputable, timestamp = _DOMAIN_CACHE[domain]

        # Only use cache entries that are less than 1 day old
        if time.time() - timestamp < 86400:  # 86400 seconds = 24 hours
            return is_reputable

    # Create a lightweight model instance
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )

    # Create system and human messages directly
    from langchain_core.messages import SystemMessage, HumanMessage

    system_message = SystemMessage(content="""
    You are an expert at evaluating source credibility for restaurant information.
    Your task is to determine if a website is a reputable source for restaurant recommendations.

    INDICATIONS OF REPUTABLE SOURCES:
    - Professional food critics or publications
    - Local newspapers or city magazines
    - Chef interviews or industry publications
    - Culinary awards or recognition organizations
    - Respected food bloggers with established expertise
    - Trendy blogs or websites with a strong food focus
    - International or local food guides (Michelin, Gault&Millau, etc.)

    INDICATIONS OF NON-REPUTABLE SOURCES:
    - Generic travel sites with crowdsourced reviews
    - Sites that primarily aggregate other reviews
    - SEO-optimized listicles with thin content
    - Content farm sites with generic recommendations
    - Sites with excessive advertisements

    Respond ONLY with "yes" for reputable sources or "no" for non-reputable sources.
    """)

    human_message = HumanMessage(content=f"""
    URL: {url}

    Content Preview:
    {content_sample[:1500]}

    Is this a reputable source for restaurant recommendations? Answer only yes or no.
    """)

    # Create messages list
    messages = [system_message, human_message]

    try:
        # Invoke the model with messages
        response = await model.ainvoke(messages)

        # Check response type and extract the text
        response_text = ""
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict) and 'content' in response:
            response_text = response['content']
        else:
            # If we can't identify the structure, convert to string
            response_text = str(response)

        # Determine if the source is reputable based on the response text
        response_text = response_text.lower().strip()
        is_reputable = "yes" in response_text and not "no" in response_text

        # Update cache
        _DOMAIN_CACHE[domain] = (is_reputable, time.time())

        return is_reputable

    except Exception as e:
        print(f"Error in AI evaluation: {e}")
        return False

@sync_to_async
async def evaluate_source_quality(url, config):
    """Full process to evaluate a source's quality with caching"""
    # Extract domain from URL
    domain = urlparse(url).netloc

    # Special cases - we know these are always reputable guides
    reputable_guides = [
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
        "nytimes.com",
        "zagat.com"
    ]

    # Check if this is a known reputable guide
    for guide in reputable_guides:
        if guide in domain:
            print(f"Domain {domain} matches known reputable guide {guide}")
            _DOMAIN_CACHE[domain] = (True, time.time())
            store_source_evaluation(url, True, config)
            return True

    # Check in-memory cache first
    if domain in _DOMAIN_CACHE:
        is_reputable, timestamp = _DOMAIN_CACHE[domain]
        if time.time() - timestamp < 86400:  # 86400 seconds = 24 hours
            return is_reputable

    # Not in cache, check database
    result = find_data(
        config.DB_TABLE_SOURCES,
        {"domain": domain},
        config
    )

    if result:
        _DOMAIN_CACHE[domain] = (result.get("is_reputable", False), time.time())
        return result.get("is_reputable", False)

    # Not in database or cache, need to evaluate
    content_sample = await fetch_quick_preview(url)
    if not content_sample:
        return False

    is_reputable = await ai_evaluate_source(content_sample, url, config)

    # Store result in database
    store_source_evaluation(url, is_reputable, config)

    return is_reputable

def store_source_evaluation(url, is_reputable, config):
    """Store the source evaluation in the database"""
    domain = urlparse(url).netloc

    source_data = {
        "domain": domain,
        "full_url": url,
        "is_reputable": is_reputable,
        "evaluated_at": time.time(),
        "evaluation_count": 1
    }

    # Update in-memory cache
    _DOMAIN_CACHE[domain] = (is_reputable, time.time())

    # Check if already exists and update instead of insert
    existing = find_data(config.DB_TABLE_SOURCES, {"domain": domain}, config)

    if existing:
        # Update existing record
        source_data["evaluation_count"] = existing.get("evaluation_count", 0) + 1
        update_data(
            config.DB_TABLE_SOURCES,
            {"domain": domain},
            source_data,
            config
        )
    else:
        # Insert new record
        save_data(
            config.DB_TABLE_SOURCES,
            source_data,
            config
        )