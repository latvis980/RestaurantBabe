# main.py - Updated with correct LangSmith environment variables
import os
import logging
import time
import traceback

os.makedirs("debug_logs", exist_ok=True)

import config
from pydantic import config as pydantic_config
from langchain_core.tracers.langchain import wait_for_all_tracers
from utils.database import initialize_db 
from utils.supabase_storage import initialize_storage_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FIXED: Configure LangSmith tracing with correct environment variables
if hasattr(config, 'LANGSMITH_API_KEY') and config.LANGSMITH_API_KEY:
    # CRITICAL: Use LANGSMITH_TRACING_V2 instead of LANGSMITH_TRACING
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY

    # Set project name - use LANGSMITH_PROJECT instead of LANGCHAIN_PROJECT
    project_name = getattr(config, 'LANGSMITH_PROJECT', 'restaurant-recommender')
    os.environ["LANGSMITH_PROJECT"] = project_name

    # Optional: Set endpoint if using self-hosted or EU region
    if hasattr(config, 'LANGSMITH_ENDPOINT'):
        os.environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT

    # Optional: Set workspace ID if you have multiple workspaces
    if hasattr(config, 'LANGSMITH_WORKSPACE_ID'):
        os.environ["LANGSMITH_WORKSPACE_ID"] = config.LANGSMITH_WORKSPACE_ID

    # ALSO set the legacy LANGCHAIN variables for backward compatibility
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = project_name

    logger.info(f"✅ LangSmith tracing enabled for project: {project_name}")
    logger.info("🔍 Both LANGSMITH_TRACING_V2 and LANGCHAIN_TRACING_V2 set to 'true'")
else:
    logger.warning("⚠️ LangSmith API key not found - tracing disabled")
    logger.warning("💡 Set LANGSMITH_API_KEY in config.py to enable tracing")

# Initialize database
initialize_db(config)

# Initialize Supabase Storage Manager
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

# NEW: Initialize automatic file cleanup
try:
    from utils.file_cleanup import start_automatic_cleanup
    cleanup_manager = start_automatic_cleanup(config)
    logger.info("✅ Automatic file cleanup initialized")
    logger.info("🧹 Temp files will be cleaned every 6 hours, full cleanup every 24 hours")
except Exception as e:
    logger.error(f"❌ Failed to initialize file cleanup: {e}")
    # Don't fail the entire app if cleanup fails
    pass

def setup_orchestrator():
    """
    Initialize and return the LangGraph restaurant agent using singleton pattern.
    """
    logger.info("🚀 Setting up LangGraph restaurant agent")
    from utils.langgraph_orchestrator_manager import initialize_langgraph_agent
    return initialize_langgraph_agent(config)

def main():
    """Main entry point: initialize orchestrator and start the Telegram bot"""
    # Initialize the orchestrator once at startup
    setup_orchestrator()
    logger.info("✅ Application initialization complete")

    # Log tracing status for debugging
    if os.environ.get("LANGSMITH_TRACING_V2") == "true":
        logger.info("🔍 LangSmith tracing is ENABLED")
        logger.info(f"📊 Project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
    else:
        logger.warning("⚠️ LangSmith tracing is DISABLED")

    # Start the Telegram bot
    from telegram_bot import main as telegram_main
    telegram_main()

if __name__ == "__main__":
    main()