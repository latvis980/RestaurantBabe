# config.py - Complete Configuration for Smart Restaurant Scraper System
"""
Complete configuration file combining your existing settings with the new Smart Scraper System.
This integrates all components while maintaining backward compatibility.
"""

import os

# ============================================================================
# API KEYS - Environment Variables (Replit / Railway Secrets)
# ============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")  # Keep for other agents if needed
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # For Claude
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# NEW: DeepSeek API configuration for ultra-fast processing
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_EMBEDDING_MODEL = "deepseek-embedding-model-name"

# ============================================================================
# SUPABASE CONFIGURATION
# ============================================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Use service_role key for server operations
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # For admin operations

# EXTERNAL SUPABASE MANAGER SERVICE SETTINGS
SUPABASE_MANAGER_URL = os.environ.get("SUPABASE_MANAGER_URL", "https://restobabesupabasemanager-production.up.railway.app")
SUPABASE_MANAGER_API_KEY = os.environ.get("SUPABASE_MANAGER_API_KEY", "")

# ============================================================================
# SMART SCRAPER SYSTEM CONFIGURATION
# ============================================================================

# Enable the smart scraper system
SMART_SCRAPER_ENABLED = True

# Content limits for different scraping strategies
CONTENT_LIMITS = {
    "content_sectioner": 12000,    # DeepSeek sectioning limit
    "simple_http": 6000,           # Simple HTTP target
    "enhanced_http": 8000,         # Enhanced HTTP target  
    "firecrawl": 10000,           # Firecrawl target (maximize expensive calls)
    "specialized": 15000           # Specialized handlers (RSS/sitemap)
}

# Strategy override patterns (domain -> strategy)
STRATEGY_OVERRIDES = {
    # Known simple sites
    "cntraveller.com": "simple_http",
    "lisbonlux.com": "simple_http",
    "queroviajarmais.com": "simple_http",
    "nomadicfoodist.com": "simple_http",
    "samiraholma.com": "simple_http",

    # Known complex sites that need specialized handling
    "timeout.com": "specialized",  # Use RSS when possible, fallback to firecrawl
    "eater.com": "specialized",    # Use RSS when possible
    "bonappetit.com": "specialized",
    "foodandwine.com": "specialized",

    # Sites that definitely need Firecrawl
    "thrillist.com": "firecrawl",
    "resy.com": "firecrawl",

    # Moderate complexity sites
    "theinfatuation.com": "enhanced_http",
    "cntraveler.com": "enhanced_http",
    "bestguide.pt": "enhanced_http"
}

# Concurrent request limits by strategy
SCRAPER_CONCURRENCY = {
    "specialized": 10,      # RSS/sitemap can handle more
    "simple_http": 8,       # Basic HTTP scraping
    "enhanced_http": 5,     # Readability processing
    "firecrawl": 3          # Expensive Firecrawl calls
}

# Restaurant keyword weights for analysis
RESTAURANT_KEYWORDS = {
    "high_value": ["restaurant", "menu", "cuisine", "chef", "dining"],
    "medium_value": ["food", "dish", "meal", "bistro", "cafe", "bar"],
    "low_value": ["eat", "taste", "price", "address", "rating"]
}

# Smart scraper cache settings
DOMAIN_CACHE_TTL = 3600 * 24  # 24 hours cache for domain analysis
SCRAPER_CACHE_SIZE = 1000     # Max cached domain analyses

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================

# OpenAI API settings (for components that need highest quality)
OPENAI_MODEL = "gpt-4o-mini"  
SEARCH_EVALUATION_MODEL = "gpt-4o-mini"  # Using GPT-4o-mini for search evaluation
SEARCH_EVALUATION_TEMPERATURE = 0.2
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_RETRIES = 1          # Aggressive - prevent delays
OPENAI_TIMEOUT = 45.0           # Shorter timeout
OPENAI_REQUEST_TIMEOUT = 20.0   # Connection timeout

# DeepSeek API settings (for speed-critical components)
DEEPSEEK_CHAT_MODEL = "deepseek-chat"  # DeepSeek-V3 - ultra fast
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"  # DeepSeek-R1 - for complex reasoning
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_MAX_RETRIES = 2
DEEPSEEK_TIMEOUT = 120.0  

# Claude API settings (high token limit for final analysis)
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Latest Claude Sonnet 4
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 8192

# Model selection strategy - which AI model to use for each component
MODEL_STRATEGY = {
    # Speed-critical components using DeepSeek
    'content_sectioning': 'deepseek',     # MAJOR bottleneck fix
    'strategy_analysis': 'deepseek',      # Domain analysis speedup

    # Search evaluation now uses OpenAI (gpt-4o-mini)
    'search_evaluation': 'openai',        # High quality search filtering

    # Quality-critical components
    'restaurant_extraction': 'openai',    # Keep quality for core function
    'list_analysis': 'claude',            # Keep Claude for final analysis
    'conversation': 'openai',             # Keep for user chat
    'editor': 'openai'                    # Keep for restaurant formatting
}

# Component-specific token limits optimized for each model
OPENAI_MAX_TOKENS_BY_COMPONENT = {
    'search_agent': 512,           # URL evaluation
    'search_evaluation': 512,      # Search result filtering
    'conversation': 1024,          # Telegram chat  
    'editor_agent': 4096,          # Restaurant formatting
    'firecrawl_scraper': 6144,     # Restaurant extraction
    'default': 2048                # Fallback
}

# DeepSeek token limits (optimized for speed)
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

# Rate limiting settings
MAX_CONCURRENT_OPENAI_CALLS = 2  # Reduced since using DeepSeek for many tasks
MAX_CONCURRENT_DEEPSEEK_CALLS = 5  # DeepSeek can handle more concurrent calls
OPENAI_RATE_LIMIT_DELAY = 0.5    
DEEPSEEK_RATE_LIMIT_DELAY = 0.2  # Faster rate limiting for DeepSeek

# ============================================================================
# LOCATION AND SEARCH SETTINGS
# ============================================================================

# Location-based search settings
GOOGLE_MAPS_KEY2 = os.environ.get("GOOGLE_MAPS_KEY2")
LOCATION_SEARCH_RADIUS_KM = 2.0
MAX_LOCATION_RESULTS = 8

# AI source mapping settings  
AI_SOURCE_MAPPING_ENABLED = True
DEFAULT_FOOD_SOURCES = ["michelin", "timeout"]
DEFAULT_WINE_SOURCES = ["raisin", "wine-list"] 
DEFAULT_COFFEE_SOURCES = ["sprudge", "coffee-review"]

# Search settings
EXCLUDED_RESTAURANT_SOURCES = ["tripadvisor.com", "opentable.com", "yelp.com", "google.com/maps"]
BLOCKED_DOMAINS = EXCLUDED_RESTAURANT_SOURCES  # Alias for consistency

# Brave search settings
BRAVE_SEARCH_COUNT = 15
BRAVE_SEARCH_API_KEY = BRAVE_API_KEY  # Alias for consistency
BRAVE_SEARCH_TIMEOUT = 30.0

# Tavily Search settings (backup search engine)
TAVILY_SEARCH_COUNT = 20
TAVILY_SEARCH_TIMEOUT = 25.0

# ============================================================================
# SCRAPING CONFIGURATION
# ============================================================================

# Firecrawl AI scraping
FIRECRAWL_ENABLED = os.environ.get("FIRECRAWL_ENABLED", "false").lower() == "true"
FIRECRAWL_MAX_REQUESTS = int(os.environ.get("FIRECRAWL_MAX_REQUESTS", "5"))  # Limit for cost control

# General scraping settings
SCRAPING_TIMEOUT = 20.0
SCRAPING_MAX_RETRIES = 2
SCRAPING_CONCURRENT_LIMIT = 5

# Quality control settings
MIN_CONTENT_LENGTH = 200        # Minimum scraped content length to consider valid
MIN_RESTAURANTS_PER_QUERY = 3   # Minimum restaurants to extract before considering response complete
MAX_RESTAURANTS_PER_QUERY = 25  # Maximum restaurants to include in final response

# ============================================================================
# DATABASE AND CACHING SETTINGS
# ============================================================================

# Database Search Agent Settings
MIN_DATABASE_RESTAURANTS = 3  # Minimum restaurants needed to use database instead of web search
DATABASE_AI_EVALUATION = False  # Enable AI evaluation of database results (future feature)
MIN_ACCEPTABLE_RATING = 4.1  # Minimum Google rating for restaurant filtering

# Restaurant Data Settings
MAX_RESTAURANTS_PER_QUERY = 10
CACHE_EXPIRY_DAYS = 7

# Domain Intelligence Settings
DOMAIN_INTELLIGENCE_ENABLED = True
AUTO_UPDATE_DOMAIN_INTELLIGENCE = True
DOMAIN_SUCCESS_THRESHOLD = 0.7  # Minimum success rate to trust a domain
DOMAIN_FAILURE_LIMIT = 5  # Block domain after this many failures

# Performance optimization
ENABLE_CONCURRENT_PROCESSING = True
ENABLE_RESULT_CACHING = True
CACHE_TTL_HOURS = 6

# ============================================================================
# GOOGLE MAPS INTEGRATION
# ============================================================================

# Google My Maps Integration
GOOGLE_MAPS_ENABLED = True
MY_MAPS_AUTO_UPDATE = True
MY_MAPS_MAX_RESTAURANTS = 500  # Limit for performance

# Geographic data settings  
GOOGLE_PLACES_API_KEY = GOOGLE_MAPS_API_KEY  # Use same key for consistency
GEOCODING_ENABLED = True
FOLLOWUP_GEODATA_ENABLED = True

# ============================================================================
# TELEGRAM BOT SETTINGS
# ============================================================================

# Voice processing settings
WHISPER_MODEL = "whisper-1"  # OpenAI Whisper model
MAX_VOICE_FILE_SIZE = 25 * 1024 * 1024  # 25MB (OpenAI limit)

# Telegram Bot settings
TELEGRAM_WEBHOOK_URL = os.environ.get("TELEGRAM_WEBHOOK_URL")
TELEGRAM_WEBHOOK_PATH = "/telegram-webhook"
TELEGRAM_WEBHOOK_PORT = int(os.environ.get("PORT", 8080))

# Response formatting
DEFAULT_LANGUAGE = "en"
ENABLE_RICH_FORMATTING = True
INCLUDE_SOURCE_ATTRIBUTION = True
MAX_MESSAGE_LENGTH = 4096       # Telegram limit

# ============================================================================
# MONITORING AND DEBUGGING
# ============================================================================

# LangSmith tracing settings
LANGSMITH_PROJECT = "restaurant-recommender"
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

# Application settings
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# Timeout settings
SEARCH_TIMEOUT = 60.0           # Overall search timeout
PROCESSING_TIMEOUT = 90.0       # Overall processing timeout

# Error handling
MAX_RETRIES_PER_STEP = 2
FALLBACK_TO_BASIC_SEARCH = True
ENABLE_GRACEFUL_DEGRADATION = True

# ============================================================================
# SOURCE MAPPING GUIDELINES
# ============================================================================

SOURCE_MAPPING_GUIDELINES = {
    "fine_dining": ["michelin", "worlds 50 best", "james beard", "local food critics"],
    "natural_wine": ["raisin", "wine list", "punch magazine", "natural wine company"],
    "coffee": ["sprudge", "perfect daily grind", "coffee review", "specialty coffee"],
    "cocktails": ["worlds 50 best bars", "punch magazine", "difford's guide", "imbibe"],
    "general": ["timeout", "eater", "conde nast traveler", "local food media"]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    # Check smart scraper limits first
    if component_name in CONTENT_LIMITS:
        return CONTENT_LIMITS[component_name]

    # Fallback to legacy limits
    legacy_key = f"{component_name}_limit"
    return CONTENT_PROCESSING_LIMITS.get(legacy_key, 6000)

def get_strategy_override(self, domain: str) -> str:
    """Get strategy override for a domain if it exists"""
    clean_domain = domain.lower().replace("www.", "")
    return STRATEGY_OVERRIDES.get(clean_domain)

def get_concurrency_limit(self, strategy: str) -> int:
    """Get concurrency limit for a scraping strategy"""
    return SCRAPER_CONCURRENCY.get(strategy, 5)

def is_smart_scraper_enabled() -> bool:
    """Check if smart scraper system is enabled"""
    return SMART_SCRAPER_ENABLED

def get_restaurant_keywords() -> dict:
    """Get restaurant keywords for content analysis"""
    return RESTAURANT_KEYWORDS

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration():
    """Validate that essential configuration is present"""
    required_keys = [
        'OPENAI_API_KEY',
        'TELEGRAM_BOT_TOKEN',
        'BRAVE_API_KEY'
    ]

    missing_keys = []
    for key in required_keys:
        if not globals().get(key):
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")

    # Warn about optional but recommended keys
    recommended_keys = {
        'DEEPSEEK_API_KEY': 'DeepSeek API for fast content sectioning',
        'FIRECRAWL_API_KEY': 'Firecrawl API for JavaScript-heavy sites',
        'LANGSMITH_API_KEY': 'LangSmith for debugging and monitoring'
    }

    for key, description in recommended_keys.items():
        if not globals().get(key):
            print(f"⚠️ Recommended: {key} not configured ({description})")

# Validate configuration on import
try:
    validate_configuration()
    print("✅ Configuration validation passed")
except ValueError as e:
    print(f"❌ Configuration error: {e}")

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# Legacy aliases for backward compatibility
CONTENT_SECTIONER_LIMIT = CONTENT_LIMITS.get("content_sectioner", 8000)
FIRECRAWL_CONTENT_LIMIT = CONTENT_LIMITS.get("firecrawl", 10000)
SIMPLE_SCRAPER_LIMIT = CONTENT_LIMITS.get("simple_http", 6000)

# Legacy functions (maintain compatibility)
def get_firecrawl_limit():
    """Legacy function for getting Firecrawl content limit"""
    return get_content_limit_for_component("firecrawl")

def get_sectioner_limit():
    """Legacy function for getting content sectioner limit"""
    return get_content_limit_for_component("content_sectioner")

# ============================================================================
# EXPORT CONFIGURATION CLASS (Optional)
# ============================================================================

class Config:
    """
    Optional configuration class for object-oriented access.
    Can be used alongside or instead of module-level variables.
    """

    def __init__(self):
        # Copy all module-level variables to instance
        import sys
        current_module = sys.modules[__name__]

        for attr_name in dir(current_module):
            if not attr_name.startswith('_') and attr_name.isupper():
                setattr(self, attr_name, getattr(current_module, attr_name))

    def get_content_limit_for_component(self, component: str) -> int:
        """Get content limit for a specific component"""
        return get_content_limit_for_component(component)

    def get_strategy_override(self, domain: str) -> str:
        """Get strategy override for a domain if it exists"""
        clean_domain = domain.lower().replace("www.", "")
        return STRATEGY_OVERRIDES.get(clean_domain)

    def get_concurrency_limit(self, strategy: str) -> int:
        """Get concurrency limit for a scraping strategy"""
        return SCRAPER_CONCURRENCY.get(strategy, 5)

# Create a default config instance
default_config = Config()

print("🚀 Smart Restaurant Scraper Configuration Loaded")
print(f"   📊 Smart Scraper: {'✅ Enabled' if SMART_SCRAPER_ENABLED else '❌ Disabled'}")
print(f"   🤖 DeepSeek API: {'✅ Configured' if DEEPSEEK_API_KEY else '⚠️ Not configured'}")
print(f"   🔥 Firecrawl API: {'✅ Configured' if FIRECRAWL_API_KEY else '⚠️ Not configured'}")
print(f"   📈 Strategy Overrides: {len(STRATEGY_OVERRIDES)} domains configured")