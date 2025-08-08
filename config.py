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
BRAVE_MEDIA_API_KEY = os.environ.get("BRAVE_MEDIA_API_KEY")  # Separate key for media searches
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

# MISSING CONFIG: Add search evaluation model
SEARCH_EVALUATION_MODEL = "gpt-4o-mini"  # Cost-optimized model for search evaluation

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
    'content_cleaning': 'openai',

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
    'editor': 2048,
    'content_evaluation': 3072,
    'restaurant_extraction': 4096,
    'query_analysis': 1024,
    'follow_up_search': 1024,
    'source_mapping': 1024,  # Add this for source mapping agent
    'location_analysis': 512   # Add this for location analysis
}

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

# Location-specific database settings
DB_PROXIMITY_RADIUS_KM = 2.0  # Radius for database proximity search
MIN_DB_MATCHES_REQUIRED = 3   # Minimum matches before triggering web search

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
    'text_cleaner_agent',
    'editor_agent',
    'follow_up_search_agent',
    'location_analyzer',       # Add location agents
    'location_search_agent',
    'media_serach_agent'
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

# Admin settings
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

# Flask settings for webhook/polling services
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("PORT", 8000))
FLASK_DEBUG = False

# Scheduler settings
ENABLE_SCHEDULER = True
BUCKET_POLL_INTERVAL_MINUTES = 10
BUCKET_MAX_FILES_PER_POLL = 10

# Webhook security
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

# ============================================================================
# GOOGLE MAPS CONFIGURATION 
# ============================================================================

# Google Maps keys (prioritize GOOGLE_MAPS_KEY2 if available)
GOOGLE_MAPS_KEY2 = os.environ.get("GOOGLE_MAPS_KEY2")

# Google Places search configuration
GOOGLE_PLACES_SEARCH_TYPES = [
    "restaurant", "bar", "cafe", "meal_takeaway", 
    "meal_delivery", "food", "bakery"
]

# Location search timeout
LOCATION_SEARCH_TIMEOUT = 30.0

# ============================================================================
# SUPABASE BUCKET CONFIGURATION
# ============================================================================

# Bucket settings
BUCKET_NAME = "scraped-content"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that required configuration is present"""
    required_vars = [
        "OPENAI_API_KEY",
        "BRAVE_API_KEY", 
        "TELEGRAM_BOT_TOKEN",
        "SUPABASE_URL",
        "SUPABASE_KEY"
    ]

    missing = []
    for var in required_vars:
        if not globals().get(var):
            missing.append(var)

    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    return True

# Auto-validate on import
if __name__ != "__main__":
    validate_config()