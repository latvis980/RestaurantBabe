# config.py
"""
Clean Configuration for Restaurant Recommendation System
Updated for current architecture with Human Mimic Scraper + Text Cleaner Agent
"""

import os
from enum import Enum

# ============================================================================
# API KEYS - Environment Variables
# ============================================================================

# Required APIs
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# AI Model APIs
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Optional APIs
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================

# OpenAI settings
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_RETRIES = 1
OPENAI_TIMEOUT = 45.0

# DeepSeek settings
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_MAX_RETRIES = 2
DEEPSEEK_TIMEOUT = 120.0

# Claude settings (for complex reasoning when needed)
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 8192

# Model routing strategy
MODEL_STRATEGY = {
    # Fast components - use DeepSeek
    'content_sectioning': 'deepseek',
    'content_cleaning': 'deepseek',
    'strategy_analysis': 'deepseek',

    # Quality components - use OpenAI
    'search_evaluation': 'openai',
    'restaurant_extraction': 'openai',
    'editor': 'openai',
    'conversation': 'openai',

    # Complex reasoning - use Claude when needed
    'complex_analysis': 'claude'
}

# Token limits by model
OPENAI_MAX_TOKENS_BY_COMPONENT = {
    'search_agent': 512,
    'search_evaluation': 512,
    'conversation': 1024,
    'editor_agent': 4096,
    'restaurant_extraction': 6144,
    'content_cleaning': 2048,
    'default': 2048
}

DEEPSEEK_MAX_TOKENS_BY_COMPONENT = {
    'content_sectioning': 2048,
    'strategy_analysis': 512,
    'content_cleaning': 2048,
    'default': 1024
}

# ============================================================================
# SCRAPING STRATEGIES AND CONFIGURATION
# ============================================================================

class ScrapingStrategy(Enum):
    SIMPLE_HTTP = "simple_http"
    ENHANCED_HTTP = "enhanced_http"
    HUMAN_MIMIC = "human_mimic"
    SPECIALIZED = "specialized"

# Strategy costs (for optimization)
STRATEGY_COSTS = {
    "specialized": 0.0,      # FREE - RSS/Sitemap
    "simple_http": 0.1,      # Basic HTTP
    "enhanced_http": 0.5,    # HTTP + readability
    "human_mimic": 2.0,      # Browser automation
}

# Content limits for different strategies
CONTENT_LIMITS = {
    "simple_http": 6000,
    "enhanced_http": 8000,
    "human_mimic": 15000,
    "content_sectioning": 8000,
    "content_cleaning": 15000
}

# Concurrent request limits
SCRAPER_CONCURRENCY = {
    "specialized": 10,
    "simple_http": 8,
    "enhanced_http": 5,
    "human_mimic": 2
}

# Strategy overrides for known domains
STRATEGY_OVERRIDES = {
    # Human Mimic works well for these
    'timeout.com': 'human_mimic',
    'eater.com': 'human_mimic',
    'guide.michelin.com': 'human_mimic',
    'zagat.com': 'human_mimic',
    'thrillist.com': 'human_mimic',

    # Simple HTTP for news sites
    'cntraveller.com': 'simple_http',
    'lisbonlux.com': 'simple_http',
    'queroviajarmais.com': 'simple_http',

    # Enhanced HTTP for CMS sites
    'theinfatuation.com': 'enhanced_http',
    'bestguide.pt': 'enhanced_http'
}

# ============================================================================
# HUMAN MIMIC SCRAPER CONFIGURATION
# ============================================================================

# Human Mimic scraper settings
HUMAN_MIMIC_ENABLED = os.environ.get("HUMAN_MIMIC_ENABLED", "true").lower() == "true"
HUMAN_MIMIC_MAX_CONCURRENT = int(os.environ.get("HUMAN_MIMIC_MAX_CONCURRENT", "2"))
HUMAN_MIMIC_DEFAULT_TIMEOUT = int(os.environ.get("HUMAN_MIMIC_DEFAULT_TIMEOUT", "30000"))
HUMAN_MIMIC_SLOW_TIMEOUT = int(os.environ.get("HUMAN_MIMIC_SLOW_TIMEOUT", "60000"))

# Human-like timing
HUMAN_MIMIC_LOAD_WAIT = float(os.environ.get("HUMAN_MIMIC_LOAD_WAIT", "3.0"))
HUMAN_MIMIC_INTERACTION_DELAY = float(os.environ.get("HUMAN_MIMIC_INTERACTION_DELAY", "0.5"))

# Known slow domains
HUMAN_MIMIC_SLOW_DOMAINS = [
    'guide.michelin.com',
    'timeout.com',
    'zagat.com',
    'opentable.com',
    'resy.com'
]

# Railway deployment optimization
RAILWAY_MEMORY_LIMIT_MB = int(os.environ.get("RAILWAY_MEMORY_LIMIT_MB", "512"))
HUMAN_MIMIC_MEMORY_PER_CONTEXT_MB = 80

def get_optimal_concurrent_contexts():
    """Calculate optimal browser contexts based on available memory"""
    available_memory = RAILWAY_MEMORY_LIMIT_MB - 200
    max_contexts = available_memory // HUMAN_MIMIC_MEMORY_PER_CONTEXT_MB
    return min(max(max_contexts, 1), HUMAN_MIMIC_MAX_CONCURRENT)

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Search settings
BRAVE_SEARCH_COUNT = 15
BRAVE_SEARCH_TIMEOUT = 30.0

# Excluded domains
EXCLUDED_RESTAURANT_SOURCES = [
    "tripadvisor.com", 
    "opentable.com", 
    "yelp.com", 
    "google.com/maps"
]

# Restaurant keywords for analysis
RESTAURANT_KEYWORDS = {
    "high_value": ["restaurant", "menu", "cuisine", "chef", "dining"],
    "medium_value": ["food", "dish", "meal", "bistro", "cafe", "bar"],
    "low_value": ["eat", "taste", "price", "address", "rating"]
}

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Database search settings
MIN_DATABASE_RESTAURANTS = 3
MIN_ACCEPTABLE_RATING = 4.1
MAX_RESTAURANTS_PER_QUERY = 25
CACHE_EXPIRY_DAYS = 7

# Geographic settings
LOCATION_SEARCH_RADIUS_KM = 2.0
MAX_LOCATION_RESULTS = 8
GEOCODING_ENABLED = True

# ============================================================================
# CURRENT AGENTS CONFIGURATION
# ============================================================================

# Active agents in your system
ACTIVE_AGENTS = [
    'query_analyzer',
    'database_search_agent', 
    'dbcontent_evaluation_agent',
    'search_agent',
    'smart_scraper',
    'human_mimic_scraper',
    'text_cleaner_agent',  # NEW
    'editor_agent',
    'follow_up_search_agent'
]

# Components that use each model
DEEPSEEK_COMPONENTS = ['content_sectioning', 'content_cleaning', 'strategy_analysis']
OPENAI_COMPONENTS = ['search_evaluation', 'restaurant_extraction', 'editor', 'conversation']
CLAUDE_COMPONENTS = ['complex_analysis']  # When needed

# ============================================================================
# TELEGRAM BOT SETTINGS
# ============================================================================

# Voice processing
WHISPER_MODEL = "whisper-1"
MAX_VOICE_FILE_SIZE = 25 * 1024 * 1024  # 25MB

# Bot response settings
DEFAULT_LANGUAGE = "en"
MAX_MESSAGE_LENGTH = 4096
ENABLE_RICH_FORMATTING = True

# Webhook settings
TELEGRAM_WEBHOOK_URL = os.environ.get("TELEGRAM_WEBHOOK_URL")
TELEGRAM_WEBHOOK_PATH = "/telegram-webhook"
TELEGRAM_WEBHOOK_PORT = int(os.environ.get("PORT", 8080))

# ============================================================================
# MONITORING AND PERFORMANCE
# ============================================================================

# LangSmith tracing
LANGSMITH_PROJECT = "restaurant-recommender"
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

# Performance settings
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
SEARCH_TIMEOUT = 60.0
PROCESSING_TIMEOUT = 90.0
MAX_RETRIES_PER_STEP = 2

# Rate limiting
MAX_CONCURRENT_OPENAI_CALLS = 2
MAX_CONCURRENT_DEEPSEEK_CALLS = 5
OPENAI_RATE_LIMIT_DELAY = 0.5
DEEPSEEK_RATE_LIMIT_DELAY = 0.2

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_for_component(component_name: str) -> str:
    """Get the appropriate model for a component"""
    return MODEL_STRATEGY.get(component_name, 'openai')

def get_token_limit_for_component(component_name: str, model_type: str = None) -> int:
    """Get token limit for a component based on its model"""
    if model_type == 'deepseek' or get_model_for_component(component_name) == 'deepseek':
        return DEEPSEEK_MAX_TOKENS_BY_COMPONENT.get(component_name, 1024)
    else:
        return OPENAI_MAX_TOKENS_BY_COMPONENT.get(component_name, 2048)

def get_content_limit_for_component(component_name: str) -> int:
    """Get content character limit for a component"""
    return CONTENT_LIMITS.get(component_name, 6000)

def get_strategy_override(domain: str) -> str:
    """Get strategy override for a domain"""
    clean_domain = domain.lower().replace("www.", "")
    return STRATEGY_OVERRIDES.get(clean_domain)

def get_concurrency_limit(strategy: str) -> int:
    """Get concurrency limit for a scraping strategy"""
    return SCRAPER_CONCURRENCY.get(strategy, 5)

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration():
    """Validate essential configuration"""
    required_keys = [
        'OPENAI_API_KEY',
        'TELEGRAM_BOT_TOKEN', 
        'BRAVE_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_KEY'
    ]

    missing_keys = [key for key in required_keys if not globals().get(key)]

    if missing_keys:
        raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")

    # Warn about optional but useful keys
    optional_keys = {
        'DEEPSEEK_API_KEY': 'Fast content processing',
        'ANTHROPIC_API_KEY': 'Claude for complex reasoning', 
        'GOOGLE_MAPS_API_KEY': 'Location services',
        'LANGSMITH_API_KEY': 'Debugging and monitoring'
    }

    for key, description in optional_keys.items():
        if not globals().get(key):
            print(f"⚠️ Optional: {key} not configured ({description})")

def get_active_models():
    """Get list of configured AI models"""
    models = []
    if OPENAI_API_KEY:
        models.append('OpenAI GPT-4o-mini')
    if DEEPSEEK_API_KEY:
        models.append('DeepSeek Chat')
    if ANTHROPIC_API_KEY:
        models.append('Claude Sonnet')
    return models

# ============================================================================
# COMPONENT CONFIGURATION CLASS
# ============================================================================

class Config:
    """Configuration class for object-oriented access"""

    def __init__(self):
        # Copy all module variables
        import sys
        current_module = sys.modules[__name__]

        for attr_name in dir(current_module):
            if not attr_name.startswith('_') and attr_name.isupper():
                setattr(self, attr_name, getattr(current_module, attr_name))

    def get_model_for_component(self, component: str) -> str:
        """Get model for component"""
        return get_model_for_component(component)

    def get_token_limit(self, component: str) -> int:
        """Get token limit for component"""
        return get_token_limit_for_component(component)

    def get_content_limit(self, component: str) -> int:
        """Get content limit for component"""
        return get_content_limit_for_component(component)

# ============================================================================
# INITIALIZATION
# ============================================================================

# Validate configuration on import
try:
    validate_configuration()
    active_models = get_active_models()
    print("✅ Configuration validated")
    print(f"🤖 Active AI models: {', '.join(active_models)}")
    print(f"🎭 Human Mimic Scraper: {'✅ Enabled' if HUMAN_MIMIC_ENABLED else '❌ Disabled'}")
    print(f"🧹 Text Cleaner: Uses {MODEL_STRATEGY.get('content_cleaning', 'openai')} model")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    print("Check your environment variables (.env file or Railway secrets)")

# Export default config instance
default_config = Config()