"""
Translation Configuration

Module-specific settings for text translation.
"""
import os

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
