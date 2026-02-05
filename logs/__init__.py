"""
Logs Module

Provides:
- Comprehensive logging configuration for LLM calls
- Request/Response logging with metrics
- Context tracking (request_id, session_id, user_id)
"""

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
    set_request_id,
    get_request_id,
    clear_request_id,
    set_session_id,
    clear_session_id,
    generate_request_id,
    LOG_DIR,
    ContextUsageLog,
    LLMRequestLog,
    LLMResponseLog,
    LLMMetrics
)

__all__ = [
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
    "set_request_id",
    "get_request_id",
    "clear_request_id",
    "set_session_id",
    "clear_session_id",
    "generate_request_id",
    "LOG_DIR",
    "ContextUsageLog",
    "LLMRequestLog",
    "LLMResponseLog",
    "LLMMetrics"
]
