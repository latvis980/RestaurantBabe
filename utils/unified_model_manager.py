# utils/unified_model_manager.py
"""
Unified model manager that automatically routes to the best model (OpenAI/DeepSeek/Claude)
based on the component and task requirements. This eliminates the retry delays and 
optimizes for both speed and quality.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

class UnifiedModelManager:
    """
    Unified manager that automatically routes requests to the optimal model
    based on component requirements. Fixes retry delays and optimizes performance.
    """

    _instance: Optional['UnifiedModelManager'] = None
    _openai_semaphore: Optional[asyncio.Semaphore] = None
    _deepseek_semaphore: Optional[asyncio.Semaphore] = None
    _last_openai_call: float = 0
    _last_deepseek_call: float = 0

    def __new__(cls, config=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return

        if config is None:
            raise ValueError("Config required for first initialization")

        self.config = config
        self._initialized = True

        # Create semaphores for limiting concurrent calls
        self._openai_semaphore = asyncio.Semaphore(
            getattr(config, 'MAX_CONCURRENT_OPENAI_CALLS', 2)
        )
        self._deepseek_semaphore = asyncio.Semaphore(
            getattr(config, 'MAX_CONCURRENT_DEEPSEEK_CALLS', 5)
        )

        # Rate limiting delays
        self.openai_rate_limit = getattr(config, 'OPENAI_RATE_LIMIT_DELAY', 0.5)
        self.deepseek_rate_limit = getattr(config, 'DEEPSEEK_RATE_LIMIT_DELAY', 0.2)

        # Cache for model clients
        self._client_cache: Dict[str, Any] = {}

        logger.info(f"Unified Model Manager initialized")
        logger.info(f"OpenAI concurrent limit: {config.MAX_CONCURRENT_OPENAI_CALLS}")
        logger.info(f"DeepSeek concurrent limit: {config.MAX_CONCURRENT_DEEPSEEK_CALLS}")

    def get_optimal_client(self, component_name: str, temperature: float = None, max_tokens: int = None):
        """
        Get the optimal model client for a component based on strategy.

        Args:
            component_name: Name of component (e.g., 'content_sectioning', 'search_evaluation')
            temperature: Override default temperature
            max_tokens: Override component-specific max tokens

        Returns:
            Configured model client (OpenAI, DeepSeek, or Claude)
        """
        model_type = self.config.get_model_for_component(component_name)

        # Create cache key
        cache_key = f"{model_type}_{component_name}_{temperature}_{max_tokens}"

        if cache_key in self._client_cache:
            return self._client_cache[cache_key], model_type

        # Get component-specific token limit
        if max_tokens is None:
            max_tokens = self.config.get_token_limit_for_component(component_name, model_type)

        # Get appropriate temperature
        if temperature is None:
            if model_type == 'deepseek':
                temperature = self.config.DEEPSEEK_TEMPERATURE
            elif model_type == 'claude':
                temperature = self.config.CLAUDE_TEMPERATURE
            else:
                temperature = self.config.OPENAI_TEMPERATURE

        # Create the appropriate client
        if model_type == 'deepseek':
            client = self._create_deepseek_client(component_name, temperature, max_tokens)
        elif model_type == 'claude':
            client = self._create_claude_client(temperature, max_tokens)
        else:  # openai
            client = self._create_openai_client(temperature, max_tokens)

        # Cache the client
        self._client_cache[cache_key] = client

        logger.debug(f"Created {model_type} client for {component_name} with {max_tokens} tokens")
        return client, model_type

    def _create_deepseek_client(self, component_name: str, temperature: float, max_tokens: int):
        """Create a DeepSeek client using OpenAI-compatible API"""

        # Choose the right DeepSeek model based on component needs
        if component_name in ['restaurant_extraction', 'complex_reasoning']:
            model = self.config.DEEPSEEK_REASONER_MODEL  # DeepSeek-R1 for complex tasks
        else:
            model = self.config.DEEPSEEK_CHAT_MODEL  # DeepSeek-V3 for fast tasks

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.config.DEEPSEEK_API_KEY,
            base_url=self.config.DEEPSEEK_BASE_URL,
            max_retries=self.config.DEEPSEEK_MAX_RETRIES,
            timeout=self.config.DEEPSEEK_TIMEOUT,
        )

    def _create_openai_client(self, temperature: float, max_tokens: int):
        """Create an OpenAI client with optimized settings"""
        return ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.config.OPENAI_API_KEY,
            max_retries=self.config.OPENAI_MAX_RETRIES,
            timeout=self.config.OPENAI_TIMEOUT,
        )

    def _create_claude_client(self, temperature: float, max_tokens: int):
        """Create a Claude client for high-quality final analysis"""
        return ChatAnthropic(
            model=self.config.CLAUDE_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.config.ANTHROPIC_API_KEY,
        )

    async def rate_limited_call(self, component_name: str, *args, **kwargs):
        """
        Make a rate-limited API call using the optimal model for the component.

        Args:
            component_name: Name of component for model selection
            *args, **kwargs: Arguments to pass to the client

        Returns:
            API response
        """
        client, model_type = self.get_optimal_client(component_name)

        # Select appropriate semaphore and rate limiting
        if model_type == 'deepseek':
            semaphore = self._deepseek_semaphore
            rate_limit = self.deepseek_rate_limit
            last_call_attr = '_last_deepseek_call'
        else:  # openai or claude
            semaphore = self._openai_semaphore
            rate_limit = self.openai_rate_limit
            last_call_attr = '_last_openai_call'

        async with semaphore:
            # Implement rate limiting
            current_time = time.time()
            time_since_last_call = current_time - getattr(self, last_call_attr)

            if time_since_last_call < rate_limit:
                sleep_time = rate_limit - time_since_last_call
                logger.debug(f"Rate limiting {model_type}: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

            setattr(self, last_call_attr, time.time())

            try:
                # Make the API call
                if hasattr(client, 'ainvoke'):
                    return await client.ainvoke(*args, **kwargs)
                else:
                    # Fallback to sync call in thread
                    import concurrent.futures
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        return await loop.run_in_executor(
                            executor, lambda: client.invoke(*args, **kwargs)
                        )
            except Exception as e:
                logger.error(f"{model_type} API call failed: {e}")
                raise

# Singleton access functions
def get_unified_model_manager(config=None) -> UnifiedModelManager:
    """Get the singleton unified model manager"""
    return UnifiedModelManager(config)

# Convenience functions for backward compatibility
def create_optimized_client(config, component_name: str, temperature: float = None, max_tokens: int = None):
    """
    Create an optimized client for a specific component.
    Automatically selects the best model (OpenAI/DeepSeek/Claude) based on component needs.
    """
    manager = get_unified_model_manager(config)
    client, model_type = manager.get_optimal_client(component_name, temperature, max_tokens)
    return client

def get_model_type_for_component(config, component_name: str) -> str:
    """Get which model type is assigned to a component"""
    return config.get_model_for_component(component_name)