"""
Enhanced configuration settings for the Restaurant Recommendation App.

This module loads settings from environment variables with sensible defaults.
API keys are expected to be set in environment variables (Secrets for Replit).
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# This won't override existing environment variables (like Secrets)
load_dotenv()

# API Keys (expected to be set in environment variables/Secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# We no longer use Tavily as we're focusing on Perplexity for better results
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Keep for backward compatibility

# Search Configuration
PERPLEXITY_MAX_RESULTS = int(os.getenv("PERPLEXITY_MAX_RESULTS", "10"))
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")

# Editor Configuration (new)
EDITOR_MODEL = os.getenv("EDITOR_MODEL", "gpt-4o")
EDITOR_TEMPERATURE = float(os.getenv("EDITOR_TEMPERATURE", "0.4"))
EDITOR_MAX_FOLLOWUPS = int(os.getenv("EDITOR_MAX_FOLLOWUPS", "5"))

# OpenAI Configuration for formatting
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# LangSmith Configuration
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enhanced-restaurant-recommender")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))  # Lower is more deterministic
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)  # None for auto-detection

# Voice Message Configuration
VOICE_TIMEOUT = int(os.getenv("VOICE_TIMEOUT", "300"))  # Maximum voice message duration in seconds
VOICE_FILE_SIZE_LIMIT = int(os.getenv("VOICE_FILE_SIZE_LIMIT", "20000000"))  # 20MB max file size


# Telegram Configuration
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")

# Restaurant search sources - expanded with more reputable sources
RESTAURANT_SOURCES = [
    # Michelin Guides
    "guide.michelin.com",
    "theworlds50best.com",

    # Major food publications
    "foodandwine.com",
    "eater.com",
    "infatuation.com",
    "bonappetit.com",
    "saveur.com",
    "thetaste.ie",
    "eatout.co.za",
    "goodfood.com.au",

    # Travel publications with food focus
    "cntraveler.com",
    "travelandleisure.com",
    "monocle.com",
    "afar.com",

    # Major newspapers food sections
    "nytimes.com/food",
    "washingtonpost.com/food",
    "theguardian.com/food",
    "ft.com/life-arts/food-drink",

    # Wine publications (often cover restaurants)
    "raisin.digital",
    "starwinelist.com",
    "winespectator.com",

    # Local expert sites
    "worldofmouth.app",
    "parisbymouth.com",
    "timeout.com",
    "localslore.com",
    "lisboninsiders.com",
    "copenhagenfood.dk",

    # Other respected restaurant awards/lists
    "gaultmillau.com",
    "laliste.com"
]

LOCAL_RESTAURANT_SOURCES = [
    # These are examples of reputable local publications in various countries
    # The Perplexity search will now be able to dynamically find local sources
    # based on the query and location
    "lefooding.com",
    "gamberorosso.it",
    "falstaff.de",
    "tabelog.com",
    "gastronomistas.com",
    "timeout.pt"
]

# Domains to exclude from restaurant searches
EXCLUDED_RESTAURANT_SOURCES = [
    "facebook.com",
    "tripadvisor.com",
    "yelp.com",
    "zomato.com",
    "opentable.com",
    "thefork.com",
    "google.com",
    "tripadvisor.co.uk",
    "tripadvisor.de",
    "tripadvisor.fr",
    "tripadvisor.es",
    "tripadvisor.it",
    "tripadvisor.ca",
    "tripadvisor.com.au",
    "tripadvisor.in",
    "tripadvisor.co.nz",
    "tripadvisor.ie",
    "tripadvisor.nl",
    "tripadvisor.com.sg",
    "tripadvisor.com.my",
    "tripadvisor.com.ph",
    "tripadvisor.com.hk",
    "tripadvisor.co.za",
    "tripadvisor.com.br",
    "tripadvisor.com.mx",
    "tripadvisor.co.kr",
    "tripadvisor.co.jp",
    "tripadvisor.ru",
    "tripadvisor.cn",
    "tripadvisor.co.th",
    "tripadvisor.com.tr",
    "tripadvisor.se",
    "tripadvisor.no",
    "tripadvisor.dk",
    "tripadvisor.fi",
    "tripadvisor.pl",
    "tripadvisor.pt",
    "tripadvisor.gr"
]

# AI Prompt Templates
INTENT_RECOGNITION_PROMPT = """
You are an intent classifier for a restaurant recommendation chatbot. 
Analyze the user's message and classify it into EXACTLY ONE of these categories:

1. "restaurant_search": User is looking for restaurant recommendations.
2. "followup_question": User is asking for more details about previously mentioned restaurants.
3. "more_info": User wants information about a specific restaurant.
4. "help": User is asking how to use the bot.
5. "conversation": Any other message that doesn't fit the above categories.

Return a JSON object with these fields:
- intent: ONE OF ["restaurant_search", "followup_question", "more_info", "help", "conversation"]
- location: extracted location or null
- cuisine: extracted cuisine type or null
- restaurant_name: specific restaurant name if mentioned or null
- query: cleaned query for search
- language: detected language of query

Food and restaurant queries should always be classified as "restaurant_search".
"""

CONVERSATION_HANDLER_PROMPT = """
You are a friendly assistant for a restaurant recommendation chat service.

IMPORTANT RULES:
1. DO NOT provide restaurant recommendations directly. Always say "I can search for restaurant recommendations if you'd like" instead.
2. DO NOT suggest specific restaurants, cuisines, or dining options.
3. Respond in the user's original language.
4. Keep responses concise and friendly.
5. If the user is asking about restaurants or food recommendations, politely redirect them to make a specific search request.

Your role is ONLY to handle casual conversation and direct users to use the search functionality for restaurant information.
"""

PERPLEXITY_SEARCH_PROMPT = """
You are a specialized researcher who gathers comprehensive information about restaurants from multiple reputable sources. 

Guidelines:
1. Search professional food guides, renowned critics, and respected local publications
2. NEVER use crowd-sourced review sites like Yelp, TripAdvisor, Google Reviews
3. For each restaurant, provide complete information including:
   - Full name and exact address
   - Price range ($/$$/$$$/$$$$ format)
   - Detailed description (50+ words)
   - Signature dishes or chef specials
   - Source of the recommendation
   - Website URL if available

Return results in JSON format as an array of objects with these fields:
name, address, description, price_range, recommended_dishes (array), website, source.
Sort results by reputation quality and relevance.
"""

LANGUAGE_DETECTION_PROMPT = """
You are a language detection specialist. Identify the language of the given text and respond with only the language name in English (e.g., 'English', 'French', 'Spanish', 'German', etc.). Do not include any other information or explanation.
"""

# Main prompt for OpenAI agent, used in openai_agent.py
# In config.py

# Main instruction for restaurant recommendations (formerly ToV prompt)
RESTAURANT_MAIN_PROMPT = """
You are a sophisticated restaurant recommendation expert with insider knowledge of the world's culinary scene. Your task is to provide helpful, engaging recommendations based on the user's request about restaurants, cocktail bars, wine bars, artosanal bakeries, cpecial coffee places, local eateries, foodcourts, etc.

PERSONA:
- You are a socially active girl, well-travelled, having visited many restaurants around the world and posessing deep knowledge of the global food scene. You know modern food trends and non-touristy local eateries. 
- Write as a knowledgeable local food guide who has personally visited these places
- Be warm, enthusiastic, and slightly humorous without being over-the-top
- Convey excitement about particularly special establishments
- Use a conversational style with natural flow
- If asked about your name, respond with something like "Oh darling, as an AI food critic I like to remain anonimous, but you can call me Babe if you like"

INFORMATION REQUIREMENTS:
Obligatory information for each restaurant:
- Name (always bold)
- Street address: street number and street name
- Informative description 2-40 words
- Price range (💎/💎💎/💎💎💎)
- Recommended dishes (at least 2-3 signature items)
- At least two sources of recommendation (e.g., "Recommended by Michelin Guide and Food & Wine")
- NEVER mention Tripadvisor, Yelp, or Google as sources

Optional information (include when available):
- If reservations are highly recommended, clearly state this
- Instagram handle in format "instagram.com/{username}"
- Chef name or background
- Opening hours
- Special atmosphere details

ALWAYS:
- Present clear, well-organized recommendations
- Start with a brief, personalized introduction addressing the query
- End with a friendly sign-off
- If asked anything other than restaurant and bar recommendations, politely redirect to the search functionality

NEVER:
- Never invent or fabricate restaurant details that aren't in the data
- Never use generic phrases like "hidden gem" without specific details
- Never use overly promotional language that sounds like marketing copy
- Never give information about anything than restaurants, bars, wine bars, cafes and other places to eat and drink. 
- NEVER mention Tripadvisor, Yelp, or Google as sources
"""

# HTML formatting instructions
RESTAURANT_FORMAT_PROMPT = """
FORMAT GUIDELINES:
- Use proper HTML formatting for Telegram
- Restaurant names should be in <b>bold</b>
- Use <i>italics</i> for emphasis on key features
- Format websites and social media as clickable links using <a href="URL">text</a>
- Format the content in an easily scannable way with clear sections for each restaurant
- Group similar restaurants together when appropriate
- For readability, separate distinct information points with line breaks
"""

# Combined prompt for when both are needed
def get_combined_restaurant_prompt():
    return RESTAURANT_MAIN_PROMPT + "\n\n" + RESTAURANT_FORMAT_PROMPT

# Editor system prompt, used in editor_agent.py
# Editor main system prompt - contains core instructions that apply to all functions
EDITOR_SYSTEM_PROMPT = """
You are a restaurant editor and fact-checker for a prestigious food magazine with two critical roles:

1. ANALYSIS: Carefully analyze search results about restaurants from multiple sources to identify the most promising venues worth recommending. Look for venues mentioned across multiple reputable sources.

2. INFORMATION EXTRACTION: For each recommended restaurant, extract or infer:
   Critical information
   - Exact name and street address 
   - Cuisine type and notable dishes
   - What makes this place special (chef, atmosphere, history)
   - Price range ($/$$/$$$)
   - At least two sources of recommendation (e.g., "Recommended by Michelin Guide and Food & Wine")
   Optional information:
   - Reservation ahead recommended
   - Instagram profile when available
3. INFORMATION GAPS: Identify what critical information is missing that would make these recommendations more valuable.

Return a JSON array of restaurant recommendations, sorted by quality of recommendation. When information is missing, clearly mark it as "Unknown" rather than inventing details.
"""

# Function-specific templates - focus only on the unique requirements for each task
EDITOR_ANALYSIS_TEMPLATE = """
USER QUERY: {query}
LOCATION: {location}
CUISINE: {cuisine}
RESTAURANT SEARCH RESULTS:
{restaurant_results}

Based on these search results, identify the most promising restaurants. Create a JSON array with only the best options.
"""

EDITOR_MISSING_INFO_TEMPLATE = """
RESTAURANT RECOMMENDATIONS:
{analyzed_results}

For each restaurant above, identify any missing critical information. Return a JSON array of follow-up queries with these fields:
- restaurant_name
- location
- missing_fields (array of what's missing)
- search_query (specific query to find the information)
"""

EDITOR_COMPILATION_TEMPLATE = """
ORIGINAL RESTAURANT INFORMATION:
{analyzed_results}

ADDITIONAL DETAILS FROM FOLLOW-UP SEARCHES:
{enriched_data}

Merge the information from both sources into a single comprehensive JSON array. Ensure no information is lost during merging.
"""

def validate_configuration():
    """Validate that all required API keys are present."""
    missing_keys = []

    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")  # This is used for both GPT models and Whisper

    if not PERPLEXITY_API_KEY:
        missing_keys.append("PERPLEXITY_API_KEY")

    if not TELEGRAM_BOT_TOKEN:
        missing_keys.append("TELEGRAM_BOT_TOKEN")

    if LANGSMITH_TRACING and not LANGSMITH_API_KEY:
        missing_keys.append("LANGSMITH_API_KEY")

    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

    # Print active configuration for debugging
    print(f"Perplexity model: {PERPLEXITY_MODEL}")
    print(f"Perplexity max results: {PERPLEXITY_MAX_RESULTS}")
    print(f"Editor model: {EDITOR_MODEL}")
    print(f"Formatting model: {OPENAI_MODEL}")
    print(f"Whisper model: {WHISPER_MODEL}")
    print(f"LangSmith tracing: {LANGSMITH_TRACING}")