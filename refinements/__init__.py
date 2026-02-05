"""
Refinements Module

Provides:
- Redis-based session storage for refinement cycles
- Session management with TTL support
- FastAPI service endpoints
"""

from .refinement_store import (
    get_refinement_store,
    init_refinement_store,
    close_refinement_store,
    RefinementData,
    RefinementStore
)
from .service import router
from .config import (
    REFINEMENT_KEY_PREFIX,
    REFINEMENT_DEFAULT_TTL,
    REFINEMENT_MAX_ITERATIONS,
    REFINEMENT_MAX_REGENERATIONS,
)

__all__ = [
    # Store
    "get_refinement_store",
    "init_refinement_store",
    "close_refinement_store",
    "RefinementData",
    "RefinementStore",
    # Service
    "router",
    # Config
    "REFINEMENT_KEY_PREFIX",
    "REFINEMENT_DEFAULT_TTL",
    "REFINEMENT_MAX_ITERATIONS",
    "REFINEMENT_MAX_REGENERATIONS",
]
