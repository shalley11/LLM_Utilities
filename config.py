"""
Global Configuration

Shared settings used across all modules.
Module-specific settings are in each module's config.py file.

All settings can be overridden via environment variables or a .env file.
See .env.example for a complete list of configurable variables.
"""
import os
from dotenv import load_dotenv

# Load .env file before any os.getenv() calls
load_dotenv()
from functools import lru_cache

# Try to import tiktoken for accurate token estimation
# For airgapped systems, set TIKTOKEN_CACHE_DIR to a directory containing pre-cached encoding files.
try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4/Claude compatible
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False
    _encoder = None

# =========================
# LLM Backend Configuration
# =========================

LLM_BACKEND = os.getenv("LLM_BACKEND", "vllm")  # ollama | vllm
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3:4b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# =========================
# Model Context Lengths
# =========================

MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,   # POC
    "gemma3:12b": 8192,  # PROD
}

DEFAULT_CONTEXT_LENGTH = 8192  # Fallback for unknown models

# Context usage warning thresholds (percentage)
CONTEXT_WARNING_THRESHOLD = 80
CONTEXT_ERROR_THRESHOLD = 95
CONTEXT_REJECT_THRESHOLD = 100  # Reject requests exceeding this percentage

# =========================
# Redis Configuration
# =========================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REFINEMENT_TTL = int(os.getenv("REFINEMENT_TTL", "7200"))  # 2 hours TTL for session expiry


# =========================
# Utility Functions
# =========================

@lru_cache(maxsize=32)
def get_model_context_length(model: str) -> int:
    """Get context length for a model (cached)."""
    return MODEL_CONTEXT_LENGTHS.get(model, DEFAULT_CONTEXT_LENGTH)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken if available, otherwise fallback to char-based estimation.

    tiktoken provides accurate token counts compatible with modern LLMs.
    Fallback uses ~4 chars per token approximation.
    """
    if TIKTOKEN_AVAILABLE and _encoder is not None:
        return len(_encoder.encode(text))
    # Fallback: ~4 chars per token (less accurate)
    return len(text) // 4


def estimate_tokens_with_info(text: str) -> dict:
    """
    Estimate tokens and return additional info about the estimation method.

    Returns:
        dict with 'tokens', 'method', and 'accurate' keys
    """
    if TIKTOKEN_AVAILABLE and _encoder is not None:
        return {
            "tokens": len(_encoder.encode(text)),
            "method": "tiktoken",
            "accurate": True
        }
    return {
        "tokens": len(text) // 4,
        "method": "char_estimate",
        "accurate": False
    }
