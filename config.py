# config.py
"""
Complete Configuration for Restaurant Recommendation System
Updated for current architecture with all required settings
"""

import os
from enum import Enum
import logging

logger = logging.getLogger(__name__)

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
    'preserve_all_sources': True,        # FIXED: Keep ALL sources, not just best
    'max_sources_per_restaurant': 5     # Limit sources per restaurant entry
}

# File management for individual processing
INDIVIDUAL_FILE_CLEANUP = {
    'cleanup_individual_files_after_hours': 72,  # Keep individual files for 3 days
    'cleanup_combined_files_after_hours': 168,   # Keep combined files for 1 week
    'max_individual_files_per_query': 50        # Prevent file system overflow
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

ENHANCED_RATING_THRESHOLD = 4.3  # Only verify venues with rating >= this threshold
MIN_DATABASE_RESULTS_TRIGGER = 2  # Trigger enhanced search when DB results < this number
MAX_VENUES_TO_VERIFY = 5


# Google Places search configuration
GOOGLE_PLACES_SEARCH_TYPES = [
    "restaurant", "bar", "cafe", "meal_takeaway", 
    "meal_delivery", "food", "bakery"
]

# Keep only the service account credentials for Places API (New)
GOOGLE_APPLICATION_CREDENTIALS_JSON_PRIMARY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON_PRIMARY")
GOOGLE_APPLICATION_CREDENTIALS_JSON_SECONDARY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON_SECONDARY")  # Optional secondary

# Location search timeout
LOCATION_SEARCH_TIMEOUT = 30.0

# Enhanced Google Maps settings
ENHANCED_GOOGLE_MAPS_FIELDS = [
    'name', 'formatted_address', 'geometry', 
    'business_status', 'rating', 'user_ratings_total',
    'reviews', 'place_id'
]

# Review analysis settings
MAX_REVIEWS_FOR_ANALYSIS = 5  # Number of Google reviews to analyze per restaurant
REVIEW_QUALITY_THRESHOLD = 6.0  # Minimum quality score (0-10) for venue selection

# Media verification settings (existing but may need updates)
TAVILY_SEARCH_MAX_RESULTS = 10  # Results per Tavily search
MEDIA_SEARCH_TIMEOUT = 30.0  # Timeout for media searches
MAX_PROFESSIONAL_SOURCES = 3  # Max professional sources to scrape per venue

# Smart scraping configuration (for future integration)
SMART_SCRAPER_API_KEY = os.environ.get("SMART_SCRAPER_API_KEY")  # Thunderbit, Browse AI, etc.
ENABLE_SMART_SCRAPING = False  # Enable when smart scraper integration is ready
MAX_SCRAPING_COST_PER_VENUE = 0.10  # Cost control for scraping services

# AI model assignments for enhanced verification
ENHANCED_VERIFICATION_MODELS = {
    'review_analysis': 'gpt-4o-mini',  # For analyzing Google reviews
    'media_analysis': 'gpt-4o-mini',   # For analyzing media sources
    'description_generation': 'gpt-4o-mini'  # For creating professional descriptions
}

# Enhanced verification quality thresholds
PROFESSIONAL_SOURCE_MIN_SCORE = 7.0  # Minimum credibility score for professional sources
MEDIA_COVERAGE_BOOST = 1.5  # Rating boost for restaurants with professional coverage

# Text editor settings
DESCRIPTION_TEMPERATURE = 0.3  # Slightly higher for creative descriptions
DESCRIPTION_MAX_LENGTH = 150  # Max characters for descriptions
ENABLE_MEDIA_MENTION = True  # Whether to mention media sources in descriptions

# Performance and cost optimization
ENABLE_PARALLEL_PROCESSING = True  # Process multiple venues in parallel
MAX_CONCURRENT_VERIFICATIONS = 3  # Max number of venues to verify simultaneously
CACHE_MEDIA_RESULTS_HOURS = 24  # Hours to cache Tavily/media results

# Error handling and fallbacks
ENABLE_GRACEFUL_DEGRADATION = True  # Continue with fewer venues if some fail
FALLBACK_TO_SIMPLE_DESCRIPTIONS = True  # Use simple descriptions if AI fails
MAX_RETRY_ATTEMPTS = 2  # Max retries for failed API calls

# Logging and monitoring
LOG_VERIFICATION_STATS = True  # Log detailed verification statistics
LOG_AI_ANALYSIS_RESULTS = False  # Log AI analysis details (for debugging)
TRACK_API_COSTS = True  # Track API usage costs

# Additional Places API (New) field mappings
PLACES_API_NEW_FIELDS = {
    'basic': [
        'id', 'displayName', 'formattedAddress', 'location', 
        'businessStatus', 'rating', 'userRatingCount'
    ],
    'atmosphere': [
        'reviews', 'editorialSummary', 'generativeSummary'
    ]
}

# Rate limiting for enhanced verification
ENHANCED_VERIFICATION_RATE_LIMITS = {
    'google_maps_calls_per_minute': 30,
    'tavily_calls_per_minute': 10,
    'openai_calls_per_minute': 50
}


# ============================================================================
# SUPABASE BUCKET CONFIGURATION
# ============================================================================

# Bucket settings
BUCKET_NAME = "scraped-content"

# ============================================================================
# VALIDATION
# ============================================================================


def validate_enhanced_config():
    """Validate enhanced verification configuration"""
    required_enhanced_vars = [
        "TAVILY_API_KEY",  # Required for media searches
    ]

    missing = []
    for var in required_enhanced_vars:
        if not globals().get(var):
            missing.append(var)

    if missing:
        logger.warning(f"⚠️ Enhanced verification features disabled - missing: {missing}")
        return False

    return True

# Update the main validation function
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

    # Validate enhanced features
    enhanced_available = validate_enhanced_config()

    return {
        'basic_config_valid': True,
        'enhanced_features_available': enhanced_available
    }


# Auto-validate on import
if __name__ != "__main__":
    validate_config()