# utils/unified_model_manager.py
"""
Unified Model Manager for Smart Scraper System

Handles routing between different AI models:
- OpenAI GPT-4o for main analysis
- DeepSeek-V3 for fast content sectioning
- Rate limiting and fallback handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("DeepSeek not available, falling back to OpenAI for content sectioning")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    CLAUDE = "claude"

@dataclass
class RateLimitInfo:
    requests_per_minute: int
    tokens_per_minute: int
    current_requests: int = 0
    current_tokens: int = 0
    window_start: float = 0

class UnifiedModelManager:
    """
    Manages multiple AI models with intelligent routing and rate limiting.

    Routing logic:
    - Content sectioning → DeepSeek (fast + cheap)
    - URL analysis → OpenAI GPT-4o (accurate)
    - Restaurant extraction → OpenAI GPT-4o (accurate)
    """

    def __init__(self, config):
        self.config = config

        # Initialize models
        self.models = {}
        self._init_openai()
        self._init_deepseek()

        # Rate limiting tracking
        self.rate_limits = {
            ModelType.OPENAI: RateLimitInfo(requests_per_minute=500, tokens_per_minute=150000),
            ModelType.DEEPSEEK: RateLimitInfo(requests_per_minute=1000, tokens_per_minute=500000)
        }

        # Model routing configuration
        self.routing_config = {
            "content_sectioning": ModelType.DEEPSEEK,
            "url_analysis": ModelType.OPENAI,
            "restaurant_extraction": ModelType.OPENAI,
            "default": ModelType.OPENAI
        }

        # Statistics
        self.stats = {
            "total_calls": 0,
            "calls_by_model": {model.value: 0 for model in ModelType},
            "calls_by_task": {},
            "rate_limit_hits": 0,
            "fallback_usage": 0
        }

    def _init_openai(self):
        """Initialize OpenAI model"""
        try:
            self.models[ModelType.OPENAI] = ChatOpenAI(
                model=self.config.OPENAI_MODEL,
                temperature=0.1,
                api_key=self.config.OPENAI_API_KEY,
                max_retries=3
            )
            logger.info("✅ OpenAI model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")

    def _init_deepseek(self):
        """Initialize DeepSeek model"""
        if not DEEPSEEK_AVAILABLE:
            logger.warning("⚠️ DeepSeek not available, content sectioning will use OpenAI")
            return

        if not hasattr(self.config, 'DEEPSEEK_API_KEY') or not self.config.DEEPSEEK_API_KEY:
            logger.warning("⚠️ DEEPSEEK_API_KEY not configured, using OpenAI fallback")
            return

        try:
            self.models[ModelType.DEEPSEEK] = ChatDeepSeek(
                model="deepseek-chat",
                api_key=self.config.DEEPSEEK_API_KEY,
                temperature=0.1,
                max_retries=2
            )
            logger.info("✅ DeepSeek model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DeepSeek: {e}")

    async def rate_limited_call(self, task_type: str, prompt: Any, **kwargs) -> Any:
        """
        Make a rate-limited AI call with intelligent model routing.

        Args:
            task_type: Type of task (content_sectioning, url_analysis, etc.)
            prompt: Prompt to send to the model
            **kwargs: Additional arguments for the model call

        Returns:
            Model response
        """
        # Determine which model to use
        target_model = self.routing_config.get(task_type, ModelType.OPENAI)

        # Check if target model is available, fallback if needed
        if target_model not in self.models:
            target_model = ModelType.OPENAI
            self.stats["fallback_usage"] += 1
            logger.debug(f"🔄 Fallback to OpenAI for {task_type}")

        # Check rate limits
        if not await self._check_rate_limit(target_model):
            # Try fallback model
            fallback_model = ModelType.OPENAI if target_model != ModelType.OPENAI else ModelType.DEEPSEEK
            if fallback_model in self.models and await self._check_rate_limit(fallback_model):
                target_model = fallback_model
                self.stats["fallback_usage"] += 1
                logger.debug(f"🚦 Rate limit hit, using fallback model for {task_type}")
            else:
                # Wait for rate limit to reset
                await self._wait_for_rate_limit_reset(target_model)

        # Make the call
        try:
            model = self.models[target_model]

            # Convert prompt format if needed
            if isinstance(prompt, ChatPromptTemplate):
                # LangChain prompt template
                response = await prompt.ainvoke(kwargs) if hasattr(prompt, 'ainvoke') else prompt.invoke(kwargs)
                result = await model.ainvoke(response.messages if hasattr(response, 'messages') else [response])
            elif isinstance(prompt, str):
                # Simple string prompt
                result = await model.ainvoke([HumanMessage(content=prompt)])
            else:
                # Direct message format
                result = await model.ainvoke(prompt)

            # Update statistics
            self._update_call_stats(target_model, task_type)

            return result

        except Exception as e:
            logger.error(f"❌ Model call failed for {task_type} with {target_model.value}: {e}")

            # Try fallback on error
            if target_model != ModelType.OPENAI and ModelType.OPENAI in self.models:
                try:
                    logger.info(f"🔄 Retrying {task_type} with OpenAI fallback")
                    model = self.models[ModelType.OPENAI]

                    if isinstance(prompt, str):
                        result = await model.ainvoke([HumanMessage(content=prompt)])
                    else:
                        result = await model.ainvoke(prompt)

                    self._update_call_stats(ModelType.OPENAI, task_type)
                    self.stats["fallback_usage"] += 1

                    return result

                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
                    raise
            else:
                raise

    async def _check_rate_limit(self, model_type: ModelType) -> bool:
        """Check if we're within rate limits for a model"""
        limit_info = self.rate_limits[model_type]
        current_time = time.time()

        # Reset window if needed
        if current_time - limit_info.window_start >= 60:
            limit_info.current_requests = 0
            limit_info.current_tokens = 0
            limit_info.window_start = current_time

        # Check limits
        if limit_info.current_requests >= limit_info.requests_per_minute:
            self.stats["rate_limit_hits"] += 1
            return False

        return True

    async def _wait_for_rate_limit_reset(self, model_type: ModelType):
        """Wait for rate limit window to reset"""
        limit_info = self.rate_limits[model_type]
        current_time = time.time()
        wait_time = 60 - (current_time - limit_info.window_start)

        if wait_time > 0:
            logger.info(f"⏳ Waiting {wait_time:.1f}s for {model_type.value} rate limit reset")
            await asyncio.sleep(wait_time)

            # Reset counters
            limit_info.current_requests = 0
            limit_info.current_tokens = 0
            limit_info.window_start = time.time()

    def _update_call_stats(self, model_type: ModelType, task_type: str):
        """Update call statistics"""
        self.stats["total_calls"] += 1
        self.stats["calls_by_model"][model_type.value] += 1

        if task_type not in self.stats["calls_by_task"]:
            self.stats["calls_by_task"][task_type] = 0
        self.stats["calls_by_task"][task_type] += 1

        # Update rate limit tracking
        limit_info = self.rate_limits[model_type]
        limit_info.current_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        return {
            **self.stats,
            "available_models": list(self.models.keys()),
            "routing_config": {k: v.value for k, v in self.routing_config.items()},
            "rate_limit_status": {
                model.value: {
                    "current_requests": info.current_requests,
                    "requests_per_minute": info.requests_per_minute,
                    "utilization": info.current_requests / info.requests_per_minute
                }
                for model, info in self.rate_limits.items()
                if model in self.models
            }
        }

    def set_routing(self, task_type: str, model_type: ModelType):
        """Update routing configuration"""
        if model_type in self.models:
            self.routing_config[task_type] = model_type
            logger.info(f"🔄 Updated routing: {task_type} → {model_type.value}")
        else:
            logger.warning(f"⚠️ Cannot route {task_type} to {model_type.value} - model not available")


# Global instance management
_model_manager_instance = None

def get_unified_model_manager(config) -> UnifiedModelManager:
    """Get or create global model manager instance"""
    global _model_manager_instance

    if _model_manager_instance is None:
        _model_manager_instance = UnifiedModelManager(config)
        logger.info("🚀 Unified Model Manager initialized")

    return _model_manager_instance

def reset_model_manager():
    """Reset global model manager (for testing)"""
    global _model_manager_instance
    _model_manager_instance = None