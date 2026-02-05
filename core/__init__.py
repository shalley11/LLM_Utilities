"""
Core Module

Shared infrastructure components for all modules:
- LLM client base class
- Configuration framework
- Validators
"""

from .llm_client_base import BaseLLMClient, LLMConfig
from .validators import (
    validate_token_count,
    validate_page_count,
    validate_text_length,
    validate_required_field,
)

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "validate_token_count",
    "validate_page_count",
    "validate_text_length",
    "validate_required_field",
]
