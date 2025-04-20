# main.py
import os
import logging
import time
import traceback
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

        # Log a summary of the results instead of the full result
        if result and isinstance(result, dict):
            num_recommended = len(result.get("recommended", []))
            num_hidden_gems = len(result.get("hidden_gems", []))
            logger.info(f"Results: {num_recommended} recommendations and {num_hidden_gems} hidden gems")
        else:
            logger.warning(f"Unexpected result format: {type(result)}")
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Make sure all traces are submitted before exiting (best practice from docs)
        logger.info("Waiting for all tracers to complete...")
        wait_for_all_tracers()
        logger.info("All tracers completed")

if __name__ == "__main__":
    main()