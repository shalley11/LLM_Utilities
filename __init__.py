"""
LLM Utilities Package (Async)

Provides:
- Async LLM client with Ollama/VLLM backend support
- Connection pooling via aiohttp for optimal performance
- Comprehensive logging for all LLM calls
- Async streaming support (AsyncGenerator)
- Request tracing
- Refinement cycle with Redis storage
"""

from .config import (
    LLM_BACKEND,
    DEFAULT_MODEL,
    OLLAMA_URL,
    VLLM_URL,
    MODEL_CONTEXT_LENGTHS,
    DEFAULT_CONTEXT_LENGTH,
    CONTEXT_WARNING_THRESHOLD,
    CONTEXT_ERROR_THRESHOLD,
    CONTEXT_REJECT_THRESHOLD,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REFINEMENT_TTL,
    get_model_context_length,
    estimate_tokens
)
from .refinement_store import (
    get_refinement_store,
    init_refinement_store,
    RefinementData,
    RefinementStore
)
from .llm_client import (
    generate_text,
    generate_text_with_logging,
    stream_text,
    stream_ollama,
    stream_vllm,
    get_session,
    close_session
)
from .logging_config import (
    setup_llm_logging,
    get_llm_logger,
    get_metrics_logger,
    log_llm_request,
    log_llm_response,
    log_metrics,
    log_context_usage,
    log_llm_call,
    RequestContext,
    SessionContext,
    UserContext,
    set_user_id,
    get_user_id,
    clear_user_id,
    LOG_DIR,
    ContextUsageLog
)

__all__ = [
    # Config
    "LLM_BACKEND",
    "DEFAULT_MODEL",
    "OLLAMA_URL",
    "VLLM_URL",
    "MODEL_CONTEXT_LENGTHS",
    "DEFAULT_CONTEXT_LENGTH",
    "CONTEXT_WARNING_THRESHOLD",
    "CONTEXT_ERROR_THRESHOLD",
    "CONTEXT_REJECT_THRESHOLD",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REFINEMENT_TTL",
    "get_model_context_length",
    "estimate_tokens",
    # LLM Client (all async)
    "generate_text",
    "generate_text_with_logging",
    "stream_text",
    "stream_ollama",
    "stream_vllm",
    "get_session",
    "close_session",
    # Logging
    "setup_llm_logging",
    "get_llm_logger",
    "get_metrics_logger",
    "log_llm_request",
    "log_llm_response",
    "log_metrics",
    "log_context_usage",
    "log_llm_call",
    "RequestContext",
    "SessionContext",
    "UserContext",
    "set_user_id",
    "get_user_id",
    "clear_user_id",
    "LOG_DIR",
    "ContextUsageLog",
    # Refinement Store
    "get_refinement_store",
    "init_refinement_store",
    "RefinementData",
    "RefinementStore"
]
