# utils/source_validator.py
from urllib.parse import urlparse
import aiohttp
import random
import time
from utils.database import find_data, save_data, update_data
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
                    # Read just enough to evaluate (about 5KB)
                    content_sample = await response.text(encoding='utf-8')
                    return content_sample[:5000]  # Return first 5000 chars
        return None
    except Exception as e:
        print(f"Error in quick preview fetch: {e}")
        return None

def check_source_reputation(url, config):
    """Check if a source is already in our reputation database"""
    # Extract domain from URL
    domain = urlparse(url).netloc

    # Check if domain exists in known sources database
    result = find_data(
        config.DB_TABLE_SOURCES,
        {"domain": domain},
        config
    )

    if result:
        # Return the stored reputation
        return result.get("is_reputable", False)

    # Not in database, need AI verification
    return None

async def ai_evaluate_source(content_sample, url, config):
    """Use AI to evaluate if a source is reputable"""
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
    - International or local food guides (Michelin, Gault&Millau, etc.)

    INDICATIONS OF NON-REPUTABLE SOURCES:
    - Generic travel sites with crowdsourced reviews
    - Sites that primarily aggregate other reviews
    - SEO-optimized listicles with thin content
    - Content farm sites with generic recommendations
    - Sites with excessive advertisements
    - Sites that lack author attribution or expertise

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

        return is_reputable

    except Exception as e:
        print(f"Error in AI evaluation: {e}")
        return False

async def evaluate_source_quality(url, config):
    """Full process to evaluate a source's quality"""
    # Quick fetch of just headers and metadata
    content_sample = await fetch_quick_preview(url)

    if not content_sample:
        return False

    # Use OpenAI to evaluate the source
    is_reputable = await ai_evaluate_source(content_sample, url, config)

    # Store result in database for future reference
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