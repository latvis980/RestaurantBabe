# config.py
"""
Complete Configuration for Restaurant Recommendation System
Updated for current architecture with all required settings
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

# Search APIs - FIXED: Added missing TAVILY_API_KEY
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

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

# Search evaluation model
SEARCH_EVALUATION_MODEL = "gpt-4o-mini"  # Cost-optimized model for search evaluation

# DeepSeek settings
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_MAX_RETRIES = 2
DEEPSEEK_TIMEOUT = 120.0

# Claude settings (for complex reasoning when needed)
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_RETRIES = 2
CLAUDE_TIMEOUT = 120.0
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

# ============================================================================
# ENHANCED TOKEN LIMITS FOR INDIVIDUAL FILE PROCESSING
# ============================================================================

# DEEPSEEK Token limits - SIGNIFICANTLY INCREASED for individual file processing
DEEPSEEK_MAX_TOKENS_BY_COMPONENT = {
    'content_sectioning': 8192,      # Increased from 4096
    'content_cleaning': 12288,       # MAJOR INCREASE from 2048 for individual file processing
    'strategy_analysis': 6144,       # Increased from 3072
    'url_analysis': 4096,           # Increased from 2048
    'restaurant_deduplication': 8192, # NEW: For restaurant merging logic
    'text_combination': 10240        # NEW: For combining individual files
}

# OPENAI Token limits - SIGNIFICANTLY INCREASED for individual file processing  
OPENAI_MAX_TOKENS_BY_COMPONENT = {
    'search_agent': 1024,           # Unchanged - sufficient
    'search_evaluation': 1024,      # Increased from 512
    'conversation': 2048,           # Increased from 1024
    'editor': 4096,                 # Increased from 2048
    'content_evaluation': 6144,     # Increased from 3072
    'restaurant_extraction': 8192,  # Increased from 4096
    'query_analysis': 2048,         # Increased from 1024
    'follow_up_search': 2048,       # Increased from 1024
    'source_mapping': 2048,         # Increased from 1024
    'location_analysis': 1024,      # Increased from 512
    'database_search': 4096,        # Increased from 2048
    'dbcontent_evaluation': 6144,   # Increased from 3072
    'content_cleaning': 12288,      # MAJOR INCREASE from 2048 for individual file processing
    'text_cleaner': 12288,          # MAJOR INCREASE from 2048 for individual file processing
    'smart_scraper': 8192,          # Increased from 4096
    'restaurant_deduplication': 8192, # NEW: For restaurant merging logic
    'text_combination': 10240,      # NEW: For combining individual files
    'individual_cleaning': 12288    # NEW: Specific for individual file cleaning
}

# ============================================================================
# NEW: INDIVIDUAL FILE PROCESSING CONFIGURATION
# ============================================================================

# Individual file processing settings
INDIVIDUAL_FILE_PROCESSING = {
    'enabled': True,
    'max_files_per_batch': 10,           # Process up to 10 files individually
    'individual_timeout': 120,           # 2 minutes per individual file
    'combination_timeout': 180,          # 3 minutes for combining files
    'deduplication_enabled': True,       # Enable restaurant deduplication
    'save_individual_files': True,       # Keep individual cleaned files
    'individual_files_directory': 'scraped_content/individual',
    'combined_files_directory': 'scraped_content/combined'
}

# Restaurant deduplication settings
RESTAURANT_DEDUPLICATION = {
    'name_similarity_threshold': 0.85,   # 85% similarity to consider same restaurant
    'address_similarity_threshold': 0.70, # 70% similarity for address matching
    'combine_descriptions': True,        # Combine descriptions from multiple sources
    'preserve_best_source': True,        # Keep the most authoritative source info
    'max_sources_per_restaurant': 5     # Limit sources per restaurant entry
}

# File management for individual processing
INDIVIDUAL_FILE_CLEANUP = {
    'cleanup_individual_files_after_hours': 72,  # Keep individual files for 3 days
    'cleanup_combined_files_after_hours': 168,   # Keep combined files for 1 week
    'max_individual_files_per_query': 50        # Prevent file system overflow
}
}

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Brave Search settings
BRAVE_SEARCH_COUNT = 15
BRAVE_SEARCH_TIMEOUT = 30.0

# Excluded sources
EXCLUDED_RESTAURANT_SOURCES = [
    "tripadvisor.com", 
    "opentable.com", 
    "yelp.com", 
    "google.com/maps",
    'doordash.com', 'ubereats.com',
    'grubhub.com', 'foursquare.com', 'zomato.com',
    'quora.com', 'reddit.com', 'facebook.com', 'instagram.com',
    'booking.com', 'expedia.com', 'airbnb.com'
]

# Quality thresholds
SOURCE_QUALITY_THRESHOLD = 0.7    # Minimum AI quality score for sources

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
# ORCHESTRATOR CONFIGURATION
# ============================================================================

# Retry settings
MAX_SEARCH_RETRIES = 2
MAX_FORMATTING_RETRIES = 2

# Timeout settings
RESPONSE_TIMEOUT = 300.0
SEARCH_TIMEOUT = 120.0
FORMATTING_TIMEOUT = 60.0

# Preview settings
ENABLE_PREVIEWS = True
PREVIEW_TIMEOUT = 30.0
PREVIEW_BATCH_SIZE = 10

# ============================================================================
# TELEGRAM BOT CONFIGURATION
# ============================================================================

# Bot settings
BOT_USERNAME = "RestaurantRecommenderBot"
MAX_MESSAGE_LENGTH = 4096
VOICE_TIMEOUT = 60.0

# Voice recognition
WHISPER_MODEL = "whisper-1"
MAX_VOICE_FILE_SIZE = 25 * 1024 * 1024  # 25MB

# Admin settings
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# Webhook settings
WEBHOOK_PORT = int(os.environ.get("PORT", 8000))
WEBHOOK_PATH = "/webhook"

# Scheduler settings
ENABLE_SCHEDULER = True
BUCKET_POLL_INTERVAL_MINUTES = 10
BUCKET_MAX_FILES_PER_POLL = 10

# Webhook security
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

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
    'location_analyzer',
    'location_search_agent',
    'media_search_agent'  # FIXED: Corrected typo from 'media_serach_agent'
]

# Components that use each model
DEEPSEEK_COMPONENTS = ['content_sectioning', 'content_cleaning', 'strategy_analysis']
OPENAI_COMPONENTS = ['search_evaluation', 'restaurant_extraction', 'editor', 'conversation']
CLAUDE_COMPONENTS = ['complex_analysis']  # When needed

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

# Flask settings for webhook/polling services
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("PORT", 8000))
FLASK_DEBUG = False

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