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

# SUPABASE SETTINGS (replacing PostgreSQL)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Use service_role key for server operations
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # For admin operations

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
    'strategy_analysis': 'deepseek',      # Fast strategy decisions
    'list_analyzer': 'openai',            # Keep quality high for final results
    'search_evaluation': 'openai',        # FIXED: Use OpenAI for quality filtering
    'telegram_formatter': 'openai',       # Keep formatting quality high
}

# RAG and Vector Search Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
SIMILARITY_THRESHOLD = 0.7  # For semantic search
CHUNK_MAX_LENGTH = 1000  # Maximum characters per content chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

# Restaurant Data Settings
DEFAULT_CREDIBILITY_SCORE = 0.5
MIN_CREDIBILITY_FOR_RECOMMENDATION = 0.6
MAX_RESTAURANTS_PER_QUERY = 10
CACHE_EXPIRY_DAYS = 7

# Domain Intelligence Settings (keeping your existing system)
DOMAIN_INTELLIGENCE_ENABLED = True
AUTO_UPDATE_DOMAIN_INTELLIGENCE = True
DOMAIN_SUCCESS_THRESHOLD = 0.7  # Minimum success rate to trust a domain
DOMAIN_FAILURE_LIMIT = 5  # Block domain after this many failures

# Google My Maps Integration
GOOGLE_MAPS_ENABLED = True
MY_MAPS_AUTO_UPDATE = True
MY_MAPS_MAX_RESTAURANTS = 500  # Limit for performance

# Firecrawl settings (keeping your existing scraping setup)
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = "https://api.firecrawl.dev"

# Search and scraping configuration
MAX_PAGES_PER_SEARCH = 3
MAX_SEARCH_RESULTS = 20
SCRAPER_TIMEOUT = 120
MAX_CONCURRENT_SCRAPES = 5

# Telegram bot settings
TELEGRAM_ADMIN_CHAT_ID = os.environ.get("TELEGRAM_ADMIN_CHAT_ID")
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_PARSE_MODE = 'HTML'

# LangSmith tracing
LANGSMITH_PROJECT = "restaurant-recommender-supabase"
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "true").lower() == "true"

# Development/Debug settings
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
SAVE_DEBUG_FILES = DEBUG_MODE
DEBUG_LOGS_DIR = "debug_logs"

# Legacy table names (remove these after migration)
# These were for your old PostgreSQL setup
# DB_TABLE_USER_PREFS = "user_preferences" 
# DB_TABLE_SEARCHES = "searches"
# DB_TABLE_PROCESSES = "processes"