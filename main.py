# main.py
import os
import logging
import time
import traceback
import os
from agents.langchain_orchestrator import LangChainOrchestrator
from langchain_core.tracers.langchain import wait_for_all_tracers

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

def setup_orchestrator():
    """Initialize and return the orchestrator"""
    logger.info("Initializing restaurant recommendation orchestrator")
    return LangChainOrchestrator(config)

def main():
    """Main entry point: start the Telegram bot"""
    from telegram_bot import main as telegram_main
    telegram_main()

if __name__ == "__main__":
    main()