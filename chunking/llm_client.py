"""
Chunking LLM Client

Module-specific LLM client for chunking service.
Uses BaseLLMClient with chunking-specific configuration.
Primarily used for vision processing of images in documents.
"""

from core import BaseLLMClient, LLMConfig
from .config import (
    CHUNKING_LLM_BACKEND,
    CHUNKING_OLLAMA_URL,
    CHUNKING_VLLM_URL,
    CHUNKING_DEFAULT_MODEL,
    CHUNKING_CONNECTION_TIMEOUT,
    CHUNKING_CONNECTION_POOL_LIMIT,
    VISION_MODEL,
    VISION_TEMPERATURE,
    VISION_MAX_TOKENS,
)

# Create module-specific configuration
_config = LLMConfig(
    backend=CHUNKING_LLM_BACKEND,
    ollama_url=CHUNKING_OLLAMA_URL,
    vllm_url=CHUNKING_VLLM_URL,
    model=CHUNKING_DEFAULT_MODEL,
    temperature=VISION_TEMPERATURE,
    max_tokens=VISION_MAX_TOKENS,
    timeout=CHUNKING_CONNECTION_TIMEOUT,
    pool_limit=CHUNKING_CONNECTION_POOL_LIMIT,
    task_name="chunk"
)

# Create module-specific client instance
_client = BaseLLMClient(_config)


async def generate_text_with_logging(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    task: str = "chunk"
) -> str:
    """
    Generate text using the chunking-specific LLM backend.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (uses CHUNKING_DEFAULT_MODEL if not specified)
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
    """Close the chunking session. Call this on application shutdown."""
    await _client.close()


def get_backend_info() -> dict:
    """Get information about the chunking LLM backend configuration."""
    info = _client.get_backend_info()
    info["vision_model"] = VISION_MODEL
    return info
