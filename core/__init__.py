"""
Core Module

Shared infrastructure components for all modules:
- LLM client base class
- Configuration framework
"""

from .llm_client_base import BaseLLMClient, LLMConfig

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
]
