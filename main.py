# main.py
import os
import logging
import time
from agents.langchain_orchestrator import LangChainOrchestrator
import config
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
    """Main entry point for the application"""
    logger.info("Starting Restaurant Recommendation Service")

    # Initialize orchestrator
    orchestrator = setup_orchestrator()

    # The main application logic would typically integrate with the Telegram bot
    # For testing purposes, we can process a sample query
    test_query = "I want to find some amazing brunch places in Lisbon with unusual brunch dishes, something I haven't tried before."
    logger.info(f"Processing test query: {test_query}")

    try:
        start_time = time.time()
        result = orchestrator.process_query(test_query)
        end_time = time.time()

        logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
        logger.info(f"Result: {result}")
    finally:
        # Make sure all traces are submitted before exiting (best practice from docs)
        wait_for_all_tracers()

if __name__ == "__main__":
    main()