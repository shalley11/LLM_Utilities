"""
Translation Configuration

Module-specific settings for text translation.
"""
import os

# =========================
# LLM Backend Configuration
# =========================

# Backend type: ollama | vllm (falls back to global config)
TRANSLATION_LLM_BACKEND = os.getenv("TRANSLATION_LLM_BACKEND", os.getenv("LLM_BACKEND", "ollama"))

# Ollama URL for translation service
TRANSLATION_OLLAMA_URL = os.getenv("TRANSLATION_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))

# VLLM URL for translation service
TRANSLATION_VLLM_URL = os.getenv("TRANSLATION_VLLM_URL", os.getenv("VLLM_URL", "http://localhost:8000"))

# =========================
# Model Settings
# =========================

TRANSLATION_DEFAULT_MODEL = os.getenv("TRANSLATION_DEFAULT_MODEL", os.getenv("DEFAULT_MODEL", "gemma3:4b"))

# =========================
# Translation Settings
# =========================

TRANSLATION_AUTO_DETECT_SOURCE = True
TRANSLATION_DEFAULT_TARGET_LANGUAGE = os.getenv("TRANSLATION_DEFAULT_TARGET_LANGUAGE", "english")

# =========================
# Batch Processing Settings
# =========================

TRANSLATION_MAX_BATCH_SIZE = int(os.getenv("TRANSLATION_MAX_BATCH_SIZE", "50"))

# =========================
# LLM Settings for Translation
# =========================

TRANSLATION_TEMPERATURE = float(os.getenv("TRANSLATION_TEMPERATURE", "0.3"))
TRANSLATION_MAX_TOKENS = int(os.getenv("TRANSLATION_MAX_TOKENS", "2048"))

# =========================
# Connection Settings
# =========================

TRANSLATION_CONNECTION_TIMEOUT = int(os.getenv("TRANSLATION_CONNECTION_TIMEOUT", "300"))
TRANSLATION_CONNECTION_POOL_LIMIT = int(os.getenv("TRANSLATION_CONNECTION_POOL_LIMIT", "50"))

# =========================
# Token Limits
# =========================

# Maximum tokens allowed for input text (percentage of model context)
TRANSLATION_MAX_TOKEN_PERCENT = int(os.getenv("TRANSLATION_MAX_TOKEN_PERCENT", "80"))
