# Configuration settings for the restaurant recommendation app
import os

# API keys from environment variables (Replit Secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

# OpenAI API settings
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.2

# Search settings
EXCLUDED_RESTAURANT_SOURCES = ["tripadvisor.com", "yelp.com", "google.com/maps"]

# Brave search settings
BRAVE_SEARCH_COUNT = 10