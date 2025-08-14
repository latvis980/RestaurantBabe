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

# Search APIs - ADD THIS LINE
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
CLAUDE_MAX_RETRIES = 2
CLAUDE_TIMEOUT = 120.0

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Brave Search settings
BRAVE_SEARCH_COUNT = 8
BRAVE_SEARCH_TIMEOUT = 30.0

# Excluded sources
EXCLUDED_RESTAURANT_SOURCES = [
    'tripadvisor.com', 'yelp.com', 'doordash.com', 'ubereats.com',
    'grubhub.com', 'foursquare.com', 'zomato.com', 'opentable.com',
    'quora.com', 'reddit.com', 'facebook.com', 'instagram.com',
    'booking.com', 'expedia.com', 'airbnb.com'
]

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