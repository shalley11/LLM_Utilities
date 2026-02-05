"""
Core Validators

Shared validation functions for all modules.
"""

from typing import Optional


def validate_token_count(
    estimated_tokens: int,
    context_limit: int,
    module_name: str = "Module"
) -> None:
    """
    Validate that estimated tokens don't exceed context limit.

    Args:
        estimated_tokens: Estimated token count for the request
        context_limit: Maximum context length for the model
        module_name: Name of the module for error messages

    Raises:
        ValueError: If tokens exceed the context limit
    """
    if estimated_tokens > context_limit:
        raise ValueError(
            f"{module_name}: Estimated tokens ({estimated_tokens}) exceed "
            f"context limit ({context_limit}). Please reduce input size."
        )


def validate_page_count(
    page_count: int,
    max_pages: int,
    module_name: str = "Module"
) -> None:
    """
    Validate that page count doesn't exceed maximum.

    Args:
        page_count: Number of pages in document
        max_pages: Maximum allowed pages
        module_name: Name of the module for error messages

    Raises:
        ValueError: If pages exceed the maximum
    """
    if page_count > max_pages:
        raise ValueError(
            f"{module_name}: Document has {page_count} pages, "
            f"exceeds maximum of {max_pages} pages."
        )


def validate_text_length(
    text: str,
    max_chars: int,
    module_name: str = "Module"
) -> None:
    """
    Validate that text length doesn't exceed maximum.

    Args:
        text: Text to validate
        max_chars: Maximum allowed characters
        module_name: Name of the module for error messages

    Raises:
        ValueError: If text length exceeds maximum
    """
    if len(text) > max_chars:
        raise ValueError(
            f"{module_name}: Text length ({len(text)}) exceeds "
            f"maximum of {max_chars} characters."
        )


def validate_required_field(
    value: Optional[str],
    field_name: str,
    module_name: str = "Module"
) -> None:
    """
    Validate that a required field is not empty.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        module_name: Name of the module for error messages

    Raises:
        ValueError: If value is None or empty
    """
    if not value or not value.strip():
        raise ValueError(
            f"{module_name}: {field_name} is required and cannot be empty."
        )
