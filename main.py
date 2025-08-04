# main.py - Updated to use orchestrator singleton with domain intelligence
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

# Initialize databases
initialize_db(config)

# ADD THIS: Initialize domain intelligence database
try:
    from utils.database_domain_intelligence import initialize_domain_intelligence
    initialize_domain_intelligence(config)
    logger.info("✅ Domain intelligence database initialized")
except Exception as e:
    logger.warning(f"⚠️ Domain intelligence initialization failed: {e}")
    logger.warning("Domain intelligence features will be disabled")

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