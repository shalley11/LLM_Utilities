"""
Summarization LLM Client

Module-specific LLM client for summarization service.
Uses BaseLLMClient with summarization-specific configuration.
"""

from core import BaseLLMClient, LLMConfig
from .config import (
    SUMMARIZATION_LLM_BACKEND,
    SUMMARIZATION_OLLAMA_URL,
    SUMMARIZATION_VLLM_URL,
    SUMMARIZATION_DEFAULT_MODEL,
    SUMMARIZATION_TEMPERATURE,
    SUMMARIZATION_MAX_TOKENS,
    SUMMARIZATION_CONNECTION_TIMEOUT,
    SUMMARIZATION_CONNECTION_POOL_LIMIT,
)

# Create module-specific configuration
_config = LLMConfig(
    backend=SUMMARIZATION_LLM_BACKEND,
    ollama_url=SUMMARIZATION_OLLAMA_URL,
    vllm_url=SUMMARIZATION_VLLM_URL,
    model=SUMMARIZATION_DEFAULT_MODEL,
    temperature=SUMMARIZATION_TEMPERATURE,
    max_tokens=SUMMARIZATION_MAX_TOKENS,
    timeout=SUMMARIZATION_CONNECTION_TIMEOUT,
    pool_limit=SUMMARIZATION_CONNECTION_POOL_LIMIT,
    task_name="summarize"
)

# Create module-specific client instance
_client = BaseLLMClient(_config)


async def generate_text_with_logging(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    task: str = "summarize"
) -> str:
    """
    Generate text using the summarization-specific LLM backend.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (uses SUMMARIZATION_DEFAULT_MODEL if not specified)
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
    """Close the summarization session. Call this on application shutdown."""
    await _client.close()


def get_backend_info() -> dict:
    """Get information about the summarization LLM backend configuration."""
    return _client.get_backend_info()
