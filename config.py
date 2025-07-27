# Configuration settings for the restaurant recommendation app
import os

# API keys from environment variables (Replit / Railway Secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")  # Keep for other agents if needed
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # For Claude
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# NEW: DeepSeek API configuration for ultra-fast processing
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# SUPABASE SETTINGS (NEW - replacing PostgreSQL)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Use service_role key for server operations
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # For admin operations

# SUPABASE MANAGER SERVICE SETTINGS (FIXED!)
# This is the URL of your separate Supabase Manager service on Railway
SUPABASE_MANAGER_URL = os.environ.get("SUPABASE_MANAGER_URL", "https://restobabesupabasemanager-production.up.railway.app")
SUPABASE_MANAGER_API_KEY = os.environ.get("SUPABASE_MANAGER_API_KEY", "")  # Optional API key for authentication

# OpenAI API settings (for components that need highest quality)
OPENAI_MODEL = "gpt-4o"  # Always using GPT-4o as requested
SEARCH_EVALUATION_MODEL = "gpt-4o-mini"  # Using GPT-4o-mini for search evaluation
SEARCH_EVALUATION_TEMPERATURE = 0.2
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_RETRIES = 1          # Aggressive - prevent delays
OPENAI_TIMEOUT = 45.0           # Shorter timeout
OPENAI_REQUEST_TIMEOUT = 20.0   # Connection timeout

# NEW: DeepSeek API settings (for speed-critical components)
DEEPSEEK_CHAT_MODEL = "deepseek-chat"  # DeepSeek-V3 - ultra fast
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"  # DeepSeek Reasoner for complex thinking

# Brave Search settings  
BRAVE_SEARCH_API_KEY = BRAVE_API_KEY  # Alias for consistency
BRAVE_SEARCH_COUNT = 20          # Number of results to fetch per search
BRAVE_SEARCH_TIMEOUT = 30.0      # Timeout for search requests

# Tavily Search settings (backup search engine)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_SEARCH_COUNT = 15
TAVILY_SEARCH_TIMEOUT = 25.0

# Firecrawl settings (premium scraping service)
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
FIRECRAWL_ENABLED = os.environ.get("FIRECRAWL_ENABLED", "false").lower() == "true"
FIRECRAWL_MAX_REQUESTS = int(os.environ.get("FIRECRAWL_MAX_REQUESTS", "5"))  # Limit for cost control

# Scraping settings
SCRAPING_TIMEOUT = 20.0
SCRAPING_MAX_RETRIES = 2
SCRAPING_CONCURRENT_LIMIT = 5

# LangSmith tracing settings
LANGSMITH_PROJECT = "restaurant-recommender"
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

# Geographic data settings  
GOOGLE_PLACES_API_KEY = GOOGLE_MAPS_API_KEY  # Use same key for consistency
GEOCODING_ENABLED = True
FOLLOWUP_GEODATA_ENABLED = True

# Telegram Bot settings
TELEGRAM_WEBHOOK_URL = os.environ.get("TELEGRAM_WEBHOOK_URL")
TELEGRAM_WEBHOOK_PATH = "/telegram-webhook"
TELEGRAM_WEBHOOK_PORT = int(os.environ.get("PORT", 8080))

# Application settings
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
MAX_MESSAGE_LENGTH = 4096       # Telegram limit
SEARCH_TIMEOUT = 60.0           # Overall search timeout
PROCESSING_TIMEOUT = 90.0       # Overall processing timeout

# Filtering settings (avoid blocked sources)
BLOCKED_DOMAINS = [
    "tripadvisor.com", "tripadvisor.co.uk", "tripadvisor.fr", "tripadvisor.de",
    "yelp.com", "yelp.co.uk", "yelp.fr", "yelp.de",
    "zomato.com", "zomato.co.uk", "zomato.fr",
    "opentable.com", "opentable.co.uk", "opentable.fr",
    "foursquare.com", "swarm.foursquare.com"
]

# Quality control settings
MIN_CONTENT_LENGTH = 200        # Minimum scraped content length to consider valid
MIN_RESTAURANTS_PER_QUERY = 3   # Minimum restaurants to extract before considering response complete
MAX_RESTAURANTS_PER_QUERY = 25  # Maximum restaurants to include in final response

# Performance optimization
ENABLE_CONCURRENT_PROCESSING = True
ENABLE_RESULT_CACHING = True
CACHE_TTL_HOURS = 6

# Response formatting
DEFAULT_LANGUAGE = "en"
ENABLE_RICH_FORMATTING = True
INCLUDE_SOURCE_ATTRIBUTION = True

# Error handling
MAX_RETRIES_PER_STEP = 2
FALLBACK_TO_BASIC_SEARCH = True
ENABLE_GRACEFUL_DEGRADATION = True