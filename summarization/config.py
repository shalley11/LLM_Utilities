"""
Summarization Configuration

Module-specific settings for text summarization.
"""
import os

# =========================
# Model Settings
# =========================

SUMMARIZATION_DEFAULT_MODEL = os.getenv("SUMMARIZATION_DEFAULT_MODEL", os.getenv("DEFAULT_MODEL", "gemma3:4b"))

# =========================
# Summary Type Settings
# =========================

SUMMARIZATION_DEFAULT_TYPE = os.getenv("SUMMARIZATION_DEFAULT_TYPE", "detailed")
SUMMARIZATION_SUPPORTED_TYPES = ["brief", "detailed", "bulletwise"]

# =========================
# Batch Processing Settings
# =========================

SUMMARIZATION_MAX_WORDS_PER_BATCH = int(os.getenv("SUMMARIZATION_MAX_WORDS_PER_BATCH", "3000"))
SUMMARIZATION_MAX_CHUNKS_PER_BATCH = int(os.getenv("SUMMARIZATION_MAX_CHUNKS_PER_BATCH", "10"))

# =========================
# Summary Length Settings
# =========================

SUMMARIZATION_INTERMEDIATE_WORDS = int(os.getenv("SUMMARIZATION_INTERMEDIATE_WORDS", "300"))
SUMMARIZATION_FINAL_WORDS = int(os.getenv("SUMMARIZATION_FINAL_WORDS", "500"))

# =========================
# LLM Settings for Summarization
# =========================

SUMMARIZATION_TEMPERATURE = float(os.getenv("SUMMARIZATION_TEMPERATURE", "0.3"))

# =========================
# Token Limits
# =========================

# Maximum tokens allowed for input text (percentage of model context)
SUMMARIZATION_MAX_TOKEN_PERCENT = int(os.getenv("SUMMARIZATION_MAX_TOKEN_PERCENT", "80"))
