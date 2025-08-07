# main.py - Updated to include Supabase Storage initialization
import os
import logging
import time
import traceback

os.makedirs("debug_logs", exist_ok=True)

import config
from pydantic import config as pydantic_config
from utils.orchestrator_manager import initialize_orchestrator
from langchain_core.tracers.langchain import wait_for_all_tracers
from utils.database import initialize_db 
from utils.supabase_storage import initialize_storage_manager  # NEW IMPORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure LangSmith tracing if API key is available
if hasattr(config, 'LANGSMITH_API_KEY') and config.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = "restaurant-recommender"
    logger.info("LangSmith tracing enabled")
else:
    logger.warning("LangSmith API key not found - tracing disabled")

# Initialize database
initialize_db(config)

# NEW: Initialize Supabase Storage Manager
try:
    if hasattr(config, 'SUPABASE_URL') and hasattr(config, 'SUPABASE_KEY'):
        storage_manager = initialize_storage_manager(
            supabase_url=config.SUPABASE_URL,
            supabase_key=config.SUPABASE_KEY
        )
        logger.info("✅ Supabase Storage Manager initialized successfully")
    else:
        logger.warning("⚠️ Missing Supabase credentials for Storage Manager")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase Storage Manager: {e}")
    # Don't fail the entire app if storage manager fails
    pass

def setup_orchestrator():
    """
    Initialize and return the orchestrator using singleton pattern.
    This replaces the old setup_orchestrator function.
    """
    logger.info("Setting up restaurant recommendation orchestrator")
    return initialize_orchestrator(config)

def main():
    """Main entry point: initialize orchestrator and start the Telegram bot"""
    # Initialize the orchestrator once at startup
    setup_orchestrator()
    logger.info("✅ Application initialization complete")

    # Start the Telegram bot
    from telegram_bot import main as telegram_main
    telegram_main()

if __name__ == "__main__":
    main()