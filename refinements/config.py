"""
Refinements Configuration

Module-specific settings for refinement session storage.
"""
import os

# =========================
# Redis Key Settings
# =========================

# Prefix for refinement session keys in Redis
REFINEMENT_KEY_PREFIX = os.getenv("REFINEMENT_KEY_PREFIX", "refine")

# =========================
# Session Settings
# =========================

# Default TTL for refinement sessions in seconds (default: 2 hours)
# Note: Can also be set via REFINEMENT_TTL in root config.py
REFINEMENT_DEFAULT_TTL = int(os.getenv("REFINEMENT_DEFAULT_TTL", "7200"))

# Maximum refinements allowed per session
REFINEMENT_MAX_ITERATIONS = int(os.getenv("REFINEMENT_MAX_ITERATIONS", "10"))

# Maximum regenerations allowed per session
REFINEMENT_MAX_REGENERATIONS = int(os.getenv("REFINEMENT_MAX_REGENERATIONS", "5"))
