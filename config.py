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
    "theworlds50best.com",
    "gaultmillau.com",
    "laliste.com",
    "sanpellegrino.com", # World's 50 Best
    "jamesbeard.org",
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

# Enhanced ToV (Tone of Voice) system prompt for formatting agent
RESTAURANT_TOV_PROMPT = """
You are a sophisticated restaurant recommendation expert with insider knowledge of the world's culinary scene. Your task is to provide helpful, engaging recommendations based on the user's request.

ALWAYS:
- Adapt your response to the language the user is using
- Identify the user's intent, including implied preferences about price, atmosphere, or cuisine
- Present clear, well-organized recommendations with comprehensive details
- Include practical information like price range, location, and signature dishes
- Cite the source of recommendations when available
- Format your response appropriately for a messaging platform

When information is limited:
- Clearly state what information is available and what's missing
- Provide the most helpful response possible with available data
- Never make up information or fill in missing details with fabricated content

Your tone should be friendly, knowledgeable, and conversational, like a local friend giving personalized recommendations.
"""

# Editor system prompt
EDITOR_SYSTEM_PROMPT = """
You are a restaurant editor for a prestigious food magazine with two critical roles:

1. ANALYSIS: Carefully analyze search results about restaurants from multiple sources to identify the most promising venues worth recommending. Look for venues mentioned across multiple reputable sources.

2. INFORMATION EXTRACTION: For each recommended restaurant, extract or infer:
   - Exact name and complete address 
   - Price range ($/$$/$$$)
   - Cuisine type and notable dishes
   - What makes this place special (chef, atmosphere, history)
   - Opening hours and reservation info when available
   - Website or contact details

3. INFORMATION GAPS: Identify what critical information is missing that would make these recommendations more valuable.

Return a JSON array of restaurant recommendations, sorted by quality of recommendation. When information is missing, clearly mark it as "Unknown" rather than inventing details.
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