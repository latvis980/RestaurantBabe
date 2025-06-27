# Configuration settings for the restaurant recommendation app
import os

# API keys from environment variables (Replit / Railway Secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# PostgreSQL settings (using variable in Railway)
DATABASE_URL = os.environ.get("DATABASE_URL")

# OpenAI API settings
OPENAI_MODEL = "gpt-4o"  # Always using GPT-4o as requested
OPENAI_TEMPERATURE = 0.2

# Database table names - keeping only what we need
DB_TABLE_USER_PREFS = "user_preferences"
DB_TABLE_SEARCHES = "searches"
DB_TABLE_PROCESSES = "processes"

# Search settings
EXCLUDED_RESTAURANT_SOURCES = ["tripadvisor.com", "yelp.com", "google.com/maps"]

# Brave search settings
BRAVE_SEARCH_COUNT = 15

SCRAPER_TYPE = os.environ.get("SCRAPER_TYPE", "default")  # "default" or "enhanced"
SCRAPER_MAX_RETRIES = int(os.environ.get("SCRAPER_MAX_RETRIES", "3"))
SCRAPER_BASE_DELAY = int(os.environ.get("SCRAPER_BASE_DELAY", "1"))
SCRAPER_MAX_DELAY = int(os.environ.get("SCRAPER_MAX_DELAY", "10"))
SCRAPER_CONCURRENCY = int(os.environ.get("SCRAPER_CONCURRENCY", "3"))