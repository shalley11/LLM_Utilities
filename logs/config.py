"""
Logging Configuration

Module-specific settings for logging.
"""
import os
from pathlib import Path

# =========================
# Log Directory
# =========================

# Default log directory (project_root/logs/output/)
LOG_OUTPUT_DIR = os.getenv("LOG_OUTPUT_DIR", str(Path(__file__).parent / "output"))

# =========================
# File Handler Settings
# =========================

# Maximum log file size in bytes (default: 10MB)
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))

# Number of backup files to keep
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# =========================
# Log Content Settings
# =========================

# Preview length for prompts/responses in logs
LOG_PREVIEW_LENGTH = int(os.getenv("LOG_PREVIEW_LENGTH", "200"))

# =========================
# Log Formats
# =========================

LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(request_id)-36s | %(user_id)-20s | "
    "%(name)-25s | %(funcName)-20s | %(message)s"
)

LOG_SIMPLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(request_id)-36s | %(user_id)-20s | %(message)s"

LOG_JSON_FORMAT = "%(message)s"

# =========================
# Log File Names
# =========================

LOG_FILE_REQUESTS = os.getenv("LOG_FILE_REQUESTS", "llm_requests.log")
LOG_FILE_ERRORS = os.getenv("LOG_FILE_ERRORS", "llm_errors.log")
LOG_FILE_METRICS = os.getenv("LOG_FILE_METRICS", "llm_metrics.log")
LOG_FILE_DEBUG = os.getenv("LOG_FILE_DEBUG", "llm_debug.log")
