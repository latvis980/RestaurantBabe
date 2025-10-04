"""
LangGraph Orchestrator Manager

Manages the LangGraph-based restaurant recommendation agent as a singleton,
providing easy integration with the existing conversation handler.
"""

import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LangGraphOrchestratorManager:
    """
    Singleton manager for the LangGraph restaurant agent.
    Ensures only one agent instance exists and provides thread-safe access.
    """

    _instance: Optional['LangGraphOrchestratorManager'] = None
    _agent = None
    _config = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure only one instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config):
        """
        Initialize the LangGraph agent with configuration.
        This should be called once during app startup.

        Args:
            config: Configuration object with all required settings
        """
        instance = cls()

        if instance._agent is None:
            with cls._lock:
                if instance._agent is None:
                    logger.info("🚀 Initializing LangGraph Restaurant Agent (singleton)")

                    from agents.langgraph_restaurant_agent import create_langgraph_agent

                    instance._config = config
                    instance._agent = create_langgraph_agent(config)

                    logger.info("✅ LangGraph agent initialized successfully")
                else:
                    logger.info("♻️ LangGraph agent already initialized, reusing existing instance")
        else:
            logger.info("♻️ LangGraph agent already initialized, reusing existing instance")

        return instance._agent

    @classmethod
    def get_agent(cls):
        """
        Get the current LangGraph agent instance.
        Raises error if not initialized.

        Returns:
            LangGraphRestaurantAgent: The singleton agent instance

        Raises:
            RuntimeError: If agent hasn't been initialized
        """
        instance = cls()

        if instance._agent is None:
            raise RuntimeError(
                "LangGraph agent not initialized! Call LangGraphOrchestratorManager.initialize(config) first."
            )

        return instance._agent

    @classmethod
    def get_config(cls):
        """Get the configuration used to initialize the agent"""
        instance = cls()
        return instance._config

    @classmethod
    def reset(cls):
        """Reset the singleton (mainly for testing)"""
        instance = cls()
        with cls._lock:
            instance._agent = None
            instance._config = None
            logger.info("🔄 LangGraph agent reset")


def get_langgraph_agent():
    """
    Convenience function to get the LangGraph agent instance.

    Returns:
        LangGraphRestaurantAgent: The singleton agent instance

    Raises:
        RuntimeError: If agent hasn't been initialized
    """
    return LangGraphOrchestratorManager.get_agent()


def initialize_langgraph_agent(config):
    """
    Convenience function to initialize the LangGraph agent.

    Args:
        config: Configuration object

    Returns:
        LangGraphRestaurantAgent: The initialized agent instance
    """
    return LangGraphOrchestratorManager.initialize(config)
