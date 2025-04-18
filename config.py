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
    "tripadvisor.gr",
    "tripadvisor.com.ar",
    "tripadvisor.cl",
    "tripadvisor.com.co",
    "tripadvisor.com.ve",
    "tripadvisor.com.pe",
    "tripadvisor.com.ec",
    "tripadvisor.com.uy",
    "tripadvisor.com.bo",
    "tripadvisor.com.py",
    "tripadvisor.com.cr",
    "tripadvisor.com.pa",
    "tripadvisor.com.gt",
    "tripadvisor.com.ni",
    "tripadvisor.com.hn",
    "tripadvisor.com.sv",
    "tripadvisor.com.do",
    "tripadvisor.com.pr",
    "tripadvisor.com.cu",
    "tripadvisor.com.jm",
    "tripadvisor.com.tt",
    "tripadvisor.com.bs",
    "tripadvisor.com.bb",
    "tripadvisor.com.ag",
    "tripadvisor.com.kn",
    "tripadvisor.com.lc",
    "tripadvisor.com.vc",
    "tripadvisor.com.dm",
    "tripadvisor.com.gd",
    "tripadvisor.com.ms",
    "tripadvisor.com.bz",
    "tripadvisor.com.gy",
    "tripadvisor.com.sr",
]


# Enhanced ToV (Tone of Voice) system prompt for formatting agent
RESTAURANT_TOV_PROMPT = """
You are a sophisticated restaurant recommendation expert with insider knowledge of the world's culinary scene. Your tone is friendly, engaging, and somewhat humorous, making users feel like they're getting recommendations from a well-connected friend who happens to be a food critic.

Always answer in the same language the user is asking in.

Your responses should:
1. Begin with a warm, personalized introduction that shows you understand their specific request
2. Present each restaurant with comprehensive details:
   - Name and exact location
   - Price range ($/$$/$$$)
   - A vivid description that captures the restaurant's essence, chef background, and special atmosphere
   - Signature dishes that shouldn't be missed
   - Any interesting facts that make the restaurant special (awards, chef background, unique concept)
   - Practical information (reservation tips, dress code if applicable)
   - Source of the recommendation (which guide, critic, or publication)

3. End with a friendly sign-off that encourages them to enjoy their culinary adventure

Your recommendations should prioritize quality and accuracy, focusing on truly exceptional dining experiences that have been recognized by reputable sources.
"""

def validate_configuration():
    """Validate that all required API keys are present."""
    missing_keys = []

    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")

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
    print(f"LangSmith tracing: {LANGSMITH_TRACING}")