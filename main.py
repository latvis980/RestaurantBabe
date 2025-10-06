# main.py
"""
Restaurant Recommendation System - Main Entry Point

CLEAN VERSION: Uses only AI Chat Layer with LangGraph orchestration
NO old conversation handler dependencies
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config
import config

# Initialize database FIRST
try:
    from utils.database import initialize_database
    initialize_database(config)
    logger.info("✅ Database initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize database: {e}")
    raise

# Initialize Supabase Storage Manager (optional) - Uses Railway environment variables
try:
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')

    if supabase_url and supabase_key:
        from utils.supabase_storage import initialize_storage_manager
        storage_manager = initialize_storage_manager(
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
        logger.info("✅ Supabase Storage Manager initialized successfully")
        logger.info("📦 Storage used for: scraped restaurant content files")
    else:
        logger.warning("⚠️ Missing SUPABASE_URL or SUPABASE_KEY environment variables")
        logger.info("💡 Storage Manager is optional - app will work without it")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase Storage Manager: {e}")
    # Don't fail the entire app if storage manager fails
    pass

# Initialize automatic file cleanup (optional)
try:
    from utils.file_cleanup import start_automatic_cleanup
    cleanup_manager = start_automatic_cleanup(config)
    logger.info("✅ Automatic file cleanup initialized")
    logger.info("🧹 Temp files will be cleaned every 6 hours, full cleanup every 24 hours")
except Exception as e:
    logger.error(f"❌ Failed to initialize file cleanup: {e}")
    # Don't fail the entire app if cleanup fails
    pass

def setup_ai_chat_system():
    """
    Initialize the AI Chat Layer with LangGraph memory system.
    This replaces the old conversation handler.
    """
    logger.info("🚀 Setting up AI Chat Layer with LangGraph orchestration")

    try:
        # Import and create the unified restaurant agent
        from langgraph_orchestrator import create_unified_restaurant_agent
        unified_agent = create_unified_restaurant_agent(config)
        logger.info("✅ Memory-enhanced unified agent initialized")
        return unified_agent

    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Chat Layer: {e}")
        raise

def main():
    """Main entry point: initialize AI Chat Layer and start the Telegram bot"""
    logger.info("🤖 Starting Restaurant Recommendation System")
    logger.info("🎯 Using AI Chat Layer with LangGraph memory (NO old conversation handler)")

    # Initialize the AI Chat Layer once at startup
    setup_ai_chat_system()
    logger.info("✅ AI Chat Layer initialization complete")

    # Log tracing status for debugging
    if os.environ.get("LANGSMITH_TRACING_V2") == "true":
        logger.info("🔍 LangSmith tracing is ENABLED")
        logger.info(f"📊 Project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
    else:
        logger.warning("⚠️ LangSmith tracing is DISABLED")

    # Start the Telegram bot with AI Chat Layer
    logger.info("🚀 Starting Telegram bot with AI Chat Layer...")

    try:
        from telegram_bot import main as telegram_main
        telegram_main()
    except Exception as e:
        logger.error(f"❌ Failed to start Telegram bot: {e}")
        raise

if __name__ == "__main__":
    main()