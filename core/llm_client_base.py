"""
Base LLM Client

Provides shared LLM client functionality for all modules.
Each module creates its own instance with its own configuration.

Features:
- Supports both Ollama and VLLM backends
- Module-specific configuration (backend, URL, model, etc.)
- Connection pooling per instance
- Comprehensive logging
- Error handling

Usage:
    # In module's llm_client.py
    from core.llm_client_base import BaseLLMClient, LLMConfig

    config = LLMConfig(
        backend="ollama",  # or "vllm"
        ollama_url="http://localhost:11434",
        model="gemma3:4b",
        task_name="translate"
    )

    client = BaseLLMClient(config)
    response = await client.generate_text_with_logging(prompt)
"""

import time
import asyncio
import aiohttp
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from config import get_model_context_length
from logs.logging_config import (
    get_llm_logger,
    log_llm_request,
    log_llm_response,
    log_metrics,
    log_context_usage,
)

logger = get_llm_logger()


@dataclass
class LLMConfig:
    """
    Configuration for an LLM client instance.

    Each module creates its own LLMConfig with module-specific settings.
    This allows different modules to use different backends, models, URLs, etc.

    Example:
        # Translation module - uses Ollama for POC
        translation_config = LLMConfig(
            backend="ollama",
            ollama_url="http://localhost:11434",
            model="gemma3:4b",
            task_name="translate"
        )

        # Summarization module - uses VLLM for production
        summarization_config = LLMConfig(
            backend="vllm",
            vllm_url="http://prod-gpu:8000",
            model="llama3:70b",
            task_name="summarize"
        )
    """
    # Backend selection: "ollama" or "vllm"
    backend: str = "ollama"

    # Ollama settings
    ollama_url: str = "http://localhost:11434"

    # VLLM settings
    vllm_url: str = "http://localhost:8000"

    # Model settings
    model: str = "gemma3:4b"
    temperature: float = 0.3
    max_tokens: int = 2048

    # Connection settings
    timeout: int = 300
    pool_limit: int = 50

    # Logging identifier
    task_name: str = "unknown"

    def get_backend_url(self) -> str:
        """Get the URL for the configured backend."""
        if self.backend == "vllm":
            return self.vllm_url
        return self.ollama_url

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "backend": self.backend,
            "ollama_url": self.ollama_url,
            "vllm_url": self.vllm_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "pool_limit": self.pool_limit,
            "task_name": self.task_name,
        }


class BaseLLMClient:
    """
    Base LLM client with shared logic for Ollama and VLLM backends.

    Each module creates its OWN INSTANCE with its OWN CONFIGURATION.
    The base class provides the shared implementation.

    Features:
    - Automatic backend selection based on config
    - Connection pooling (per instance)
    - Request/response logging
    - Metrics collection
    - Error handling with proper cleanup

    Example:
        config = LLMConfig(backend="ollama", model="gemma3:4b")
        client = BaseLLMClient(config)

        response = await client.generate_text_with_logging(
            prompt="Translate this to Spanish",
            temperature=0.3
        )
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client with module-specific configuration.

        Args:
            config: LLMConfig with backend, URL, model, and other settings
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

        logger.debug(
            f"[{config.task_name.upper()}_LLM] Initialized | "
            f"backend={config.backend} | model={config.model} | "
            f"url={config.get_backend_url()}"
        )

    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session for this instance.

        Each BaseLLMClient instance maintains its own session,
        allowing different modules to have independent connection pools.
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.pool_limit,
                limit_per_host=self.config.pool_limit
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            logger.debug(
                f"[{self.config.task_name.upper()}_LLM] Session created | "
                f"backend={self.config.backend}"
            )
        return self._session

    async def close(self):
        """Close this instance's session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug(f"[{self.config.task_name.upper()}_LLM] Session closed")

    async def generate_text_with_logging(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        task: str = None,
    ) -> str:
        """
        Generate text using the configured backend with full logging.

        Automatically routes to Ollama or VLLM based on config.backend.

        Args:
            prompt: The prompt to send to the LLM
            model: Override model (uses config.model if not specified)
            temperature: Override temperature (uses config.temperature if not specified)
            max_tokens: Override max_tokens (uses config.max_tokens if not specified)
            task: Override task name for logging (uses config.task_name if not specified)

        Returns:
            Generated text response
        """
        # Use config defaults, allow overrides
        model_name = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        task_name = task or self.config.task_name

        # Get context limit for model
        context_limit = get_model_context_length(model_name)

        # Log request
        request_id = log_llm_request(
            model=model_name,
            backend=self.config.backend,
            task=task_name,
            prompt=prompt,
            temperature=temp,
            max_tokens=max_tok
        )

        # Log context usage
        context_stats = log_context_usage(
            request_id=request_id,
            model=model_name,
            prompt=prompt,
            context_limit=context_limit
        )

        start_time = time.time()

        try:
            # Route to correct backend
            if self.config.backend == "vllm":
                response = await self._call_vllm(prompt, model_name, temp, max_tok)
            else:
                response = await self._call_ollama(prompt, model_name, temp)

            latency_ms = (time.time() - start_time) * 1000

            # Log successful response
            log_llm_response(
                request_id=request_id,
                model=model_name,
                backend=self.config.backend,
                response=response,
                latency_ms=latency_ms,
                status="success"
            )

            # Log metrics
            log_metrics(
                request_id=request_id,
                model=model_name,
                backend=self.config.backend,
                task=task_name,
                latency_ms=latency_ms,
                prompt_chars=len(prompt),
                response_chars=len(response),
                status="success",
                context_limit=context_stats["context_limit"],
                estimated_tokens=context_stats["estimated_tokens"],
                context_usage_percent=context_stats["usage_percent"]
            )

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log error response
            log_llm_response(
                request_id=request_id,
                model=model_name,
                backend=self.config.backend,
                response="",
                latency_ms=latency_ms,
                status="error",
                error_message=str(e)
            )

            # Log error metrics
            log_metrics(
                request_id=request_id,
                model=model_name,
                backend=self.config.backend,
                task=task_name,
                latency_ms=latency_ms,
                prompt_chars=len(prompt),
                response_chars=0,
                status="error",
                context_limit=context_stats["context_limit"],
                estimated_tokens=context_stats["estimated_tokens"],
                context_usage_percent=context_stats["usage_percent"]
            )

            raise

    async def _call_ollama(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> str:
        """
        Call Ollama API using this instance's configured URL.

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Generation temperature

        Returns:
            Generated text
        """
        url = f"{self.config.ollama_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        logger.debug(
            f"[{self.config.task_name.upper()}_LLM] Calling Ollama | "
            f"url={url} | model={model}"
        )

        try:
            session = await self.get_session()
            async with session.post(url, json=payload) as r:
                r.raise_for_status()
                response_data = await r.json()
                return response_data.get("response", "").strip()

        except asyncio.TimeoutError:
            logger.error(
                f"[{self.config.task_name.upper()}_LLM] Ollama timeout | model={model}"
            )
            raise RuntimeError(
                f"{self.config.task_name.title()} LLM request timed out. Please try again."
            )

        except aiohttp.ClientError as e:
            logger.error(
                f"[{self.config.task_name.upper()}_LLM] Ollama request failed | "
                f"model={model} | error={e}"
            )
            raise RuntimeError(
                f"{self.config.task_name.title()} LLM service unavailable. Please try again later."
            )

    async def _call_vllm(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Call VLLM API using this instance's configured URL.

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        url = f"{self.config.vllm_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        logger.debug(
            f"[{self.config.task_name.upper()}_LLM] Calling VLLM | "
            f"url={url} | model={model}"
        )

        try:
            session = await self.get_session()
            async with session.post(url, json=payload) as r:
                r.raise_for_status()
                response_data = await r.json()
                return response_data["choices"][0]["message"]["content"].strip()

        except asyncio.TimeoutError:
            logger.error(
                f"[{self.config.task_name.upper()}_LLM] VLLM timeout | model={model}"
            )
            raise RuntimeError(
                f"{self.config.task_name.title()} LLM request timed out. Please try again."
            )

        except aiohttp.ClientError as e:
            logger.error(
                f"[{self.config.task_name.upper()}_LLM] VLLM request failed | "
                f"model={model} | error={e}"
            )
            raise RuntimeError(
                f"{self.config.task_name.title()} LLM service unavailable. Please try again later."
            )

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about this client's backend configuration.

        Returns:
            Dictionary with backend configuration details
        """
        return {
            "backend": self.config.backend,
            "active_url": self.config.get_backend_url(),
            "ollama_url": self.config.ollama_url,
            "vllm_url": self.config.vllm_url,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "pool_limit": self.config.pool_limit,
            "task_name": self.config.task_name,
        }
