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
CHUNKING_DEFAULT_PROCESS_IMAGES = True

# Minimum text length for chunking
CHUNKING_MIN_TEXT_LENGTH = int(os.getenv("CHUNKING_MIN_TEXT_LENGTH", "10"))

# =========================
# Batch Processing Settings
# =========================

CHUNKING_MAX_BATCH_SIZE = int(os.getenv("CHUNKING_MAX_BATCH_SIZE", "100"))
