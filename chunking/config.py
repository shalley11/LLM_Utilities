"""
Chunking Configuration

Module-specific settings for text chunking.
"""
import os

# =========================
# Chunking Settings
# =========================

CHUNKING_DEFAULT_OVERLAP = int(os.getenv("CHUNKING_DEFAULT_OVERLAP", "200"))
CHUNKING_DEFAULT_RESERVE_FOR_PROMPT = int(os.getenv("CHUNKING_DEFAULT_RESERVE_FOR_PROMPT", "1000"))
CHUNKING_DEFAULT_PROCESS_IMAGES = os.getenv("CHUNKING_DEFAULT_PROCESS_IMAGES", "true").lower() == "true"

# Minimum text length for chunking
CHUNKING_MIN_TEXT_LENGTH = int(os.getenv("CHUNKING_MIN_TEXT_LENGTH", "10"))

# Default context length for unknown models
CHUNKING_DEFAULT_CONTEXT_LENGTH = int(os.getenv("CHUNKING_DEFAULT_CONTEXT_LENGTH", "8192"))

# Characters per token estimate (used for chunk size calculation)
CHUNKING_CHARS_PER_TOKEN = float(os.getenv("CHUNKING_CHARS_PER_TOKEN", "4"))

# =========================
# Batch Processing Settings
# =========================

CHUNKING_MAX_BATCH_SIZE = int(os.getenv("CHUNKING_MAX_BATCH_SIZE", "100"))

# =========================
# Vision Processor Settings
# =========================

# Vision model for image processing (uses Ollama)
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:4b")

# Ollama URL for vision processing (falls back to global OLLAMA_URL)
VISION_OLLAMA_URL = os.getenv("VISION_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))

# Request timeout for vision API calls (seconds)
VISION_REQUEST_TIMEOUT = int(os.getenv("VISION_REQUEST_TIMEOUT", "120"))

# Maximum concurrent vision requests
VISION_MAX_CONCURRENT = int(os.getenv("VISION_MAX_CONCURRENT", "3"))

# Temperature for vision model
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.3"))

# Max tokens for vision model output
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "500"))
