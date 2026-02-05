"""
Editor Toolkit LLM Client

Module-specific LLM client for editor toolkit service.
Uses BaseLLMClient with editor-specific configuration.
"""

from core import BaseLLMClient, LLMConfig
from .config import (
    EDITOR_LLM_BACKEND,
    EDITOR_OLLAMA_URL,
    EDITOR_VLLM_URL,
    EDITOR_DEFAULT_MODEL,
    EDITOR_TEMPERATURE,
    EDITOR_MAX_TOKENS,
    EDITOR_CONNECTION_TIMEOUT,
    EDITOR_CONNECTION_POOL_LIMIT,
)

# Create module-specific configuration
_config = LLMConfig(
    backend=EDITOR_LLM_BACKEND,
    ollama_url=EDITOR_OLLAMA_URL,
    vllm_url=EDITOR_VLLM_URL,
    model=EDITOR_DEFAULT_MODEL,
    temperature=EDITOR_TEMPERATURE,
    max_tokens=EDITOR_MAX_TOKENS,
    timeout=EDITOR_CONNECTION_TIMEOUT,
    pool_limit=EDITOR_CONNECTION_POOL_LIMIT,
    task_name="edit"
)

# Create module-specific client instance
_client = BaseLLMClient(_config)


async def generate_text_with_logging(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    task: str = "edit"
) -> str:
    """
    Generate text using the editor-specific LLM backend.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (uses EDITOR_DEFAULT_MODEL if not specified)
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
    """Close the editor session. Call this on application shutdown."""
    await _client.close()


def get_backend_info() -> dict:
    """Get information about the editor LLM backend configuration."""
    return _client.get_backend_info()
