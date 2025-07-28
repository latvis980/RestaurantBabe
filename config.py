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

# EXTERNAL SUPABASE MANAGER SERVICE SETTINGS (NEW!)
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
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"  # DeepSeek-R1 - for complex reasoning
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_MAX_RETRIES = 2
DEEPSEEK_TIMEOUT = 120.0  

# FIXED: Model selection strategy - search_evaluation now uses OpenAI
MODEL_STRATEGY = {
    # Speed-critical components using DeepSeek
    'content_sectioning': 'deepseek',     # MAJOR bottleneck fix
    'strategy_analysis': 'deepseek',      # Domain analysis speedup

    # FIXED: Search evaluation now uses OpenAI (gpt-4o-mini)
    'search_evaluation': 'openai',        # Changed from 'deepseek' to 'openai'

    # Quality-critical components (keep existing)
    'restaurant_extraction': 'openai',    # Keep quality for core function
    'list_analysis': 'claude',            # Keep Claude for final analysis
    'conversation': 'openai',             # Keep for user chat
    'editor': 'openai'                    # Keep for restaurant formatting
}

# Component-specific token limits optimized for each model
OPENAI_MAX_TOKENS_BY_COMPONENT = {
    'search_agent': 512,           # URL evaluation - now using OpenAI
    'search_evaluation': 512,      # Added explicit search evaluation limit
    'conversation': 1024,          # Telegram chat  
    'editor_agent': 4096,          # Restaurant formatting
    'firecrawl_scraper': 6144,     # Restaurant extraction
    'default': 2048                # Fallback
}

# NEW: DeepSeek token limits (optimized for speed)
DEEPSEEK_MAX_TOKENS_BY_COMPONENT = {
    'content_sectioning': 2048,    # Fast content analysis
    'strategy_analysis': 512,      # Fast domain analysis
    'default': 1024                # General purpose
}

# Content processing limits (character-based)
CONTENT_PROCESSING_LIMITS = {
    'firecrawl_content_limit': 12000,      # Increased from 4000
    'content_sectioner_limit': 8000,      # Optimized for DeepSeek
    'simple_scraper_limit': 6000,         # For basic HTTP scraping
    'search_snippet_limit': 500           # Keep search evaluation lightweight
}

# Rate limiting settings (prevents the retry delays you're seeing)
MAX_CONCURRENT_OPENAI_CALLS = 2  # Reduced since using DeepSeek for many tasks
MAX_CONCURRENT_DEEPSEEK_CALLS = 5  # DeepSeek can handle more concurrent calls
OPENAI_RATE_LIMIT_DELAY = 0.5    
DEEPSEEK_RATE_LIMIT_DELAY = 0.2  # Faster rate limiting for DeepSeek

# Claude API settings (high token limit for final analysis)
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Latest Claude Sonnet 4
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 8192

# Add these lines to config.py in the appropriate section

# Database Search Agent Settings (NEW)
MIN_DATABASE_RESTAURANTS = 3  # Minimum restaurants needed to use database instead of web search
DATABASE_AI_EVALUATION = False  # Enable AI evaluation of database results (future feature)
MIN_ACCEPTABLE_RATING = 4.1  # Minimum Google rating for restaurant filtering (used by follow_up_search_agent)

# Keep these - they're used in domain intelligence
DOMAIN_FAILURE_LIMIT = 5
DOMAIN_SUCCESS_THRESHOLD = 0.7  # Used in domain intelligence
CACHE_EXPIRY_DAYS = 7  # Used for general caching
# Restaurant Data Settings (NEW)
MAX_RESTAURANTS_PER_QUERY = 10
CACHE_EXPIRY_DAYS = 7

# Domain Intelligence Settings (NEW - enhancing your existing system)
DOMAIN_INTELLIGENCE_ENABLED = True
AUTO_UPDATE_DOMAIN_INTELLIGENCE = True
DOMAIN_SUCCESS_THRESHOLD = 0.7  # Minimum success rate to trust a domain
DOMAIN_FAILURE_LIMIT = 5  # Block domain after this many failures

# Google My Maps Integration (NEW)
GOOGLE_MAPS_ENABLED = True
MY_MAPS_AUTO_UPDATE = True
MY_MAPS_MAX_RESTAURANTS = 500  # Limit for performance

# Search settings
EXCLUDED_RESTAURANT_SOURCES = ["tripadvisor.com", "opentable.com",  "yelp.com", "google.com/maps"]

# Brave search settings
BRAVE_SEARCH_COUNT = 15
BRAVE_SEARCH_API_KEY = BRAVE_API_KEY  # Alias for consistency
BRAVE_SEARCH_TIMEOUT = 30.0      # Timeout for search requests

# Firecrawl AI scraping
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
FIRECRAWL_ENABLED = os.environ.get("FIRECRAWL_ENABLED", "false").lower() == "true"
FIRECRAWL_MAX_REQUESTS = int(os.environ.get("FIRECRAWL_MAX_REQUESTS", "5"))  # Limit for cost control

# Tavily Search settings (backup search engine)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_SEARCH_COUNT = 15
TAVILY_SEARCH_TIMEOUT = 25.0

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
BLOCKED_DOMAINS = EXCLUDED_RESTAURANT_SOURCES  # Alias for consistency

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

# Admin alerts
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# Helper functions to get appropriate settings for each component
def get_model_for_component(component_name: str) -> str:
    """Get the appropriate model (openai/deepseek/claude) for a component"""
    return MODEL_STRATEGY.get(component_name, 'openai')

def get_token_limit_for_component(component_name: str, model_type: str = None) -> int:
    """Get the appropriate token limit for a component based on its assigned model"""
    if model_type == 'deepseek' or get_model_for_component(component_name) == 'deepseek':
        return DEEPSEEK_MAX_TOKENS_BY_COMPONENT.get(component_name, DEEPSEEK_MAX_TOKENS_BY_COMPONENT['default'])
    else:
        return OPENAI_MAX_TOKENS_BY_COMPONENT.get(component_name, OPENAI_MAX_TOKENS_BY_COMPONENT['default'])

def get_content_limit_for_component(component_name: str) -> int:
    """Get the appropriate content character limit for a component"""
    return CONTENT_PROCESSING_LIMITS.get(f"{component_name}_limit", 6000)