"""
Translation LLM Client

Module-specific LLM client for translation service.
Uses BaseLLMClient with translation-specific configuration.
"""

from core import BaseLLMClient, LLMConfig
from .config import (
    TRANSLATION_LLM_BACKEND,
    TRANSLATION_OLLAMA_URL,
    TRANSLATION_VLLM_URL,
    TRANSLATION_DEFAULT_MODEL,
    TRANSLATION_TEMPERATURE,
    TRANSLATION_MAX_TOKENS,
    TRANSLATION_CONNECTION_TIMEOUT,
    TRANSLATION_CONNECTION_POOL_LIMIT,
)

# Create module-specific configuration
_config = LLMConfig(
    backend=TRANSLATION_LLM_BACKEND,
    ollama_url=TRANSLATION_OLLAMA_URL,
    vllm_url=TRANSLATION_VLLM_URL,
    model=TRANSLATION_DEFAULT_MODEL,
    temperature=TRANSLATION_TEMPERATURE,
    max_tokens=TRANSLATION_MAX_TOKENS,
    timeout=TRANSLATION_CONNECTION_TIMEOUT,
    pool_limit=TRANSLATION_CONNECTION_POOL_LIMIT,
    task_name="translate"
)

# Create module-specific client instance
_client = BaseLLMClient(_config)


async def generate_text_with_logging(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    task: str = "translate"
) -> str:
    """
    Generate text using the translation-specific LLM backend.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (uses TRANSLATION_DEFAULT_MODEL if not specified)
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        task: Task identifier for logging

    Returns:
        Generated text response
    """
    return await _client.generate_text_with_logging(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        task=task
    )


async def close_session():
    """Close the translation session. Call this on application shutdown."""
    await _client.close()


def get_backend_info() -> dict:
    """Get information about the translation LLM backend configuration."""
    return _client.get_backend_info()
