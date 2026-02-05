"""
Text Extractor Configuration

Module-specific settings for document text extraction.
"""
import os

# =========================
# File Processing Settings
# =========================

# Supported file types for document processing
EXTRACTOR_SUPPORTED_FILE_TYPES = {".pdf", ".docx", ".doc", ".txt"}

# Directory settings
EXTRACTOR_UPLOAD_DIR = os.getenv("EXTRACTOR_UPLOAD_DIR", "/tmp/text_extractor/uploads")
EXTRACTOR_IMAGE_DIR = os.getenv("EXTRACTOR_IMAGE_DIR", "/tmp/text_extractor/images")

# =========================
# Default Extraction Options
# =========================

EXTRACTOR_DEFAULT_INCLUDE_TABLES = True
EXTRACTOR_DEFAULT_INCLUDE_IMAGES = True
EXTRACTOR_DEFAULT_INCLUDE_BLOCKS = False

# =========================
# Processing Limits
# =========================

EXTRACTOR_MAX_FILE_SIZE_MB = int(os.getenv("EXTRACTOR_MAX_FILE_SIZE_MB", "50"))
EXTRACTOR_MAX_PAGES = int(os.getenv("EXTRACTOR_MAX_PAGES", "50"))  # Maximum pages allowed for document processing
