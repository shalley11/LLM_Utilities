"""
Editor Toolkit Configuration

Module-specific settings for text editing tasks.
"""
import os

# =========================
# LLM Backend Configuration
# =========================

# Backend type: ollama | vllm (falls back to global config)
EDITOR_LLM_BACKEND = os.getenv("EDITOR_LLM_BACKEND", os.getenv("LLM_BACKEND", "ollama"))

# Ollama URL for editor service
EDITOR_OLLAMA_URL = os.getenv("EDITOR_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))

# VLLM URL for editor service
EDITOR_VLLM_URL = os.getenv("EDITOR_VLLM_URL", os.getenv("VLLM_URL", "http://localhost:8000"))

# =========================
# Model Settings
# =========================

EDITOR_DEFAULT_MODEL = os.getenv("EDITOR_DEFAULT_MODEL", os.getenv("DEFAULT_MODEL", "gemma3:4b"))

# =========================
# Supported Tasks
# =========================

EDITOR_SUPPORTED_TASKS = ["rephrase", "professional", "proofread", "concise"]

# Task descriptions for documentation
EDITOR_TASK_DESCRIPTIONS = {
    "rephrase": "Rephrase text to improve clarity, readability, and remove repetitions while preserving meaning",
    "professional": "Rewrite text in a formal, professional tone suitable for business communication",
    "proofread": "Fix grammar, spelling, punctuation, and clarity issues",
    "concise": "Shorten text by removing unnecessary words while preserving essential meaning",
}

# =========================
# Batch Processing Settings
# =========================

EDITOR_MAX_BATCH_SIZE = int(os.getenv("EDITOR_MAX_BATCH_SIZE", "50"))

# =========================
# LLM Settings for Editor
# =========================

EDITOR_TEMPERATURE = float(os.getenv("EDITOR_TEMPERATURE", "0.3"))
EDITOR_MAX_TOKENS = int(os.getenv("EDITOR_MAX_TOKENS", "2048"))

# =========================
# Connection Settings
# =========================

EDITOR_CONNECTION_TIMEOUT = int(os.getenv("EDITOR_CONNECTION_TIMEOUT", "300"))
EDITOR_CONNECTION_POOL_LIMIT = int(os.getenv("EDITOR_CONNECTION_POOL_LIMIT", "50"))

# =========================
# Proofread Focus Options
# =========================

EDITOR_PROOFREAD_FOCUS_OPTIONS = ["general", "grammar", "punctuation", "clarity"]
EDITOR_DEFAULT_PROOFREAD_FOCUS = "general"

# =========================
# Token Limits
# =========================

# Maximum tokens allowed for input text (percentage of model context)
EDITOR_MAX_TOKEN_PERCENT = int(os.getenv("EDITOR_MAX_TOKEN_PERCENT", "80"))
