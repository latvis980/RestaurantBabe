# utils/orchestrator_manager.py
"""
Singleton Orchestrator Manager
Consolidates all orchestrator initialization patterns while preserving test functionality
"""
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

class OrchestratorManager:
    """
    Singleton manager for the LangChain orchestrator.
    Ensures only one orchestrator instance exists and provides thread-safe access.
    """

    _instance: Optional['OrchestratorManager'] = None
    _orchestrator = None
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
        Initialize the orchestrator with configuration.
        This should be called once during app startup.

        Args:
            config: Configuration object with all required settings
        """
        instance = cls()

        if instance._orchestrator is None:
            with cls._lock:
                if instance._orchestrator is None:
                    logger.info("🚀 Initializing LangChain orchestrator (singleton)")

                    # Import here to avoid circular imports
                    from agents.langchain_orchestrator import LangChainOrchestrator

                    instance._config = config
                    instance._orchestrator = LangChainOrchestrator(config)

                    logger.info("✅ Orchestrator initialized successfully")
                else:
                    logger.info("♻️ Orchestrator already initialized, reusing existing instance")
        else:
            logger.info("♻️ Orchestrator already initialized, reusing existing instance")

        return instance._orchestrator

    @classmethod
    def get_orchestrator(cls):
        """
        Get the current orchestrator instance.
        Raises error if not initialized.

        Returns:
            LangChainOrchestrator: The singleton orchestrator instance

        Raises:
            RuntimeError: If orchestrator hasn't been initialized
        """
        instance = cls()

        if instance._orchestrator is None:
            raise RuntimeError(
                "Orchestrator not initialized! Call OrchestratorManager.initialize(config) first."
            )

        return instance._orchestrator

    @classmethod
    def get_config(cls):
        """Get the configuration used to initialize the orchestrator"""
        instance = cls()
        return instance._config

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if orchestrator has been initialized"""
        instance = cls()
        return instance._orchestrator is not None

    @classmethod
    def reset(cls):
        """
        Reset the singleton (useful for testing).
        This will force recreation on next initialize() call.
        """
        instance = cls()
        with cls._lock:
            instance._orchestrator = None
            instance._config = None
            logger.info("🔄 Orchestrator manager reset")


# Convenience functions for backward compatibility
def initialize_orchestrator(config):
    """
    Initialize the global orchestrator instance.

    Args:
        config: Configuration object

    Returns:
        LangChainOrchestrator: The initialized orchestrator
    """
    return OrchestratorManager.initialize(config)


def get_orchestrator():
    """
    Get the global orchestrator instance.

    Returns:
        LangChainOrchestrator: The orchestrator instance

    Raises:
        RuntimeError: If not initialized
    """
    return OrchestratorManager.get_orchestrator()


def get_orchestrator_config():
    """
    Get the configuration used for the orchestrator.

    Returns:
        Config object or None if not initialized
    """
    return OrchestratorManager.get_config()