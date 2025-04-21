# Configuration settings for the restaurant recommendation app
import os

# API keys from environment variables (Replit Secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
# Add Mistral API key
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# PostgreSQL settings (using variable in Railway)
DATABASE_URL = os.environ.get("DATABASE_URL")

# OpenAI API settings
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.2

# Database table names
DB_TABLE_SOURCES = "local_sources"
DB_TABLE_SOURCES = "sources"  # Table for source reputation
DB_TABLE_RESTAURANTS = "restaurants"
DB_TABLE_PROCESSES = "processes"
DB_TABLE_SEARCHES = "searches"

# Search settings
EXCLUDED_RESTAURANT_SOURCES = ["tripadvisor.com", "yelp.com", "google.com/maps"]

# Brave search settings
BRAVE_SEARCH_COUNT = 10