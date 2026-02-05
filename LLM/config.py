"""
LLM Client Configuration

Module-specific settings for LLM client (Ollama/VLLM backends).
"""
import os

# =========================
# Connection Settings
# =========================

# HTTP client timeout in seconds
LLM_CONNECTION_TIMEOUT = int(os.getenv("LLM_CONNECTION_TIMEOUT", "300"))

# Connection pool limits
LLM_CONNECTION_POOL_LIMIT = int(os.getenv("LLM_CONNECTION_POOL_LIMIT", "100"))
LLM_CONNECTION_POOL_LIMIT_PER_HOST = int(os.getenv("LLM_CONNECTION_POOL_LIMIT_PER_HOST", "20"))

# =========================
# Generation Defaults
# =========================

# Default temperature for text generation
LLM_DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.3"))

# Default max tokens for generation
LLM_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "1024"))
