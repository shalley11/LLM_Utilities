"""
LLM Module

Provides:
- LLM client with async support for Ollama/VLLM backends
- Prompt templates for various text processing tasks
"""

from .llm_client import (
    generate_text,
    generate_text_with_logging,
    stream_text,
    stream_ollama,
    stream_vllm,
    get_session,
    close_session
)

from .prompts import (
    get_summary_prompt,
    get_prompt,
    get_refinement_prompt,
    get_regenerate_prompt,
    get_proofreading_prompt,
    get_iterative_refinement_prompt
)

__all__ = [
    # LLM Client
    "generate_text",
    "generate_text_with_logging",
    "stream_text",
    "stream_ollama",
    "stream_vllm",
    "get_session",
    "close_session",
    # Prompts
    "get_summary_prompt",
    "get_prompt",
    "get_refinement_prompt",
    "get_regenerate_prompt",
    "get_proofreading_prompt",
    "get_iterative_refinement_prompt"
]
