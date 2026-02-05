"""
Refinements Service

FastAPI endpoints for managing refinement sessions.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import REFINEMENT_TTL
from .config import (
    REFINEMENT_KEY_PREFIX,
    REFINEMENT_DEFAULT_TTL,
    REFINEMENT_MAX_ITERATIONS,
    REFINEMENT_MAX_REGENERATIONS,
)
from .refinement_store import get_refinement_store, RefinementData

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/docAI/v1/refinements", tags=["Refinements"])


# =====================
# Request/Response Models
# =====================

class CreateSessionRequest(BaseModel):
    """Request to create a new refinement session."""
    task: str = Field(..., description="Task type (summary, rephrase, translate, etc.)")
    result: str = Field(..., description="Initial result to store")
    original_text: str = Field(..., description="Original input text")
    model: str = Field(..., description="Model used for generation")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field(None, description="User name for logging")
    summary_type: Optional[str] = Field(None, description="Summary type (brief, detailed, bulletwise)")
    target_language: Optional[str] = Field(None, description="Target language for translation")


class SessionResponse(BaseModel):
    """Response containing session data."""
    request_id: str
    task: str
    current_result: str
    original_text: str
    model: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    summary_type: Optional[str] = None
    target_language: Optional[str] = None
    refinement_count: int
    regeneration_count: int
    created_at: str
    updated_at: str
    ttl_seconds: Optional[int] = None


class UpdateSessionRequest(BaseModel):
    """Request to update a refinement session."""
    request_id: str = Field(..., description="Session request ID")
    new_result: str = Field(..., description="New result to store")
    user_id: Optional[str] = Field(None, description="User identifier")


class RegenerateSessionRequest(BaseModel):
    """Request to update session with regenerated result."""
    request_id: str = Field(..., description="Session request ID")
    new_result: str = Field(..., description="New regenerated result")
    user_id: Optional[str] = Field(None, description="User identifier")


class ExtendTTLRequest(BaseModel):
    """Request to extend session TTL."""
    request_id: str = Field(..., description="Session request ID")
    ttl_seconds: int = Field(REFINEMENT_DEFAULT_TTL, description="New TTL in seconds")


class SessionStatusResponse(BaseModel):
    """Response containing session status."""
    request_id: str
    exists: bool
    ttl_seconds: int
    refinement_count: Optional[int] = None
    regeneration_count: Optional[int] = None
    task: Optional[str] = None
    user_id: Optional[str] = None


# =====================
# Helper Functions
# =====================

def _session_to_response(data: RefinementData, ttl: int = None) -> SessionResponse:
    """Convert RefinementData to SessionResponse."""
    return SessionResponse(
        request_id=data.request_id,
        task=data.task,
        current_result=data.current_result,
        original_text=data.original_text,
        model=data.model,
        user_id=data.user_id,
        user_name=data.user_name,
        summary_type=data.summary_type,
        target_language=data.target_language,
        refinement_count=data.refinement_count,
        regeneration_count=data.regeneration_count,
        created_at=data.created_at,
        updated_at=data.updated_at,
        ttl_seconds=ttl
    )


# =====================
# API Endpoints
# =====================

@router.post("/create", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Create a new refinement session.

    **Request Body:**
    - `task`: Task type (summary, rephrase, translate, etc.)
    - `result`: Initial result to store
    - `original_text`: Original input text
    - `model`: Model used for generation
    - `user_id`: User identifier (optional)
    - `user_name`: User name for logging (optional)
    - `summary_type`: Summary type for summary tasks (optional)
    - `target_language`: Target language for translation tasks (optional)

    **Returns:**
    - Session data with generated request_id
    """
    logger.info(
        f"[REFINEMENT_CREATE] task={request.task} | "
        f"user_id={request.user_id} | user_name={request.user_name}"
    )

    store = get_refinement_store()

    try:
        data = await store.create(
            task=request.task,
            result=request.result,
            original_text=request.original_text,
            model=request.model,
            user_id=request.user_id,
            user_name=request.user_name,
            summary_type=request.summary_type,
            target_language=request.target_language
        )

        ttl = await store.get_ttl(data.request_id)

        logger.info(f"[REFINEMENT_CREATE] SUCCESS | request_id={data.request_id}")

        return _session_to_response(data, ttl)

    except Exception as e:
        logger.error(f"[REFINEMENT_CREATE] ERROR | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/get/{request_id}", response_model=SessionResponse)
async def get_session(request_id: str):
    """
    Get a refinement session by request_id.

    **Path Parameters:**
    - `request_id`: The session request ID

    **Returns:**
    - Session data or 404 if not found/expired
    """
    logger.info(f"[REFINEMENT_GET] request_id={request_id}")

    store = get_refinement_store()

    data = await store.get(request_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"Session with request_id '{request_id}' not found or expired",
                "request_id": request_id
            }
        )

    ttl = await store.get_ttl(request_id)

    return _session_to_response(data, ttl)


@router.post("/update", response_model=SessionResponse)
async def update_session(request: UpdateSessionRequest):
    """
    Update a refinement session with new result.

    Increments the refinement_count.

    **Request Body:**
    - `request_id`: Session request ID
    - `new_result`: New result to store
    - `user_id`: User identifier (optional)

    **Returns:**
    - Updated session data or 404 if not found
    """
    logger.info(f"[REFINEMENT_UPDATE] request_id={request.request_id} | user_id={request.user_id}")

    store = get_refinement_store()

    # Check iteration limit
    existing = await store.get(request.request_id)
    if existing and existing.refinement_count >= REFINEMENT_MAX_ITERATIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "max_iterations_exceeded",
                "message": f"Maximum refinement iterations ({REFINEMENT_MAX_ITERATIONS}) reached",
                "refinement_count": existing.refinement_count,
                "max_iterations": REFINEMENT_MAX_ITERATIONS
            }
        )

    data = await store.update(
        request_id=request.request_id,
        new_result=request.new_result,
        user_id=request.user_id
    )

    if not data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"Session with request_id '{request.request_id}' not found or expired",
                "request_id": request.request_id
            }
        )

    ttl = await store.get_ttl(request.request_id)

    logger.info(f"[REFINEMENT_UPDATE] SUCCESS | request_id={request.request_id} | count={data.refinement_count}")

    return _session_to_response(data, ttl)


@router.post("/regenerate", response_model=SessionResponse)
async def regenerate_session(request: RegenerateSessionRequest):
    """
    Update a session with regenerated result (from original text).

    Increments the regeneration_count.

    **Request Body:**
    - `request_id`: Session request ID
    - `new_result`: New regenerated result
    - `user_id`: User identifier (optional)

    **Returns:**
    - Updated session data or 404 if not found
    """
    logger.info(f"[REFINEMENT_REGENERATE] request_id={request.request_id} | user_id={request.user_id}")

    store = get_refinement_store()

    # Check regeneration limit
    existing = await store.get(request.request_id)
    if existing and existing.regeneration_count >= REFINEMENT_MAX_REGENERATIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "max_regenerations_exceeded",
                "message": f"Maximum regenerations ({REFINEMENT_MAX_REGENERATIONS}) reached",
                "regeneration_count": existing.regeneration_count,
                "max_regenerations": REFINEMENT_MAX_REGENERATIONS
            }
        )

    data = await store.update_regeneration(
        request_id=request.request_id,
        new_result=request.new_result,
        user_id=request.user_id
    )

    if not data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"Session with request_id '{request.request_id}' not found or expired",
                "request_id": request.request_id
            }
        )

    ttl = await store.get_ttl(request.request_id)

    logger.info(f"[REFINEMENT_REGENERATE] SUCCESS | request_id={request.request_id} | count={data.regeneration_count}")

    return _session_to_response(data, ttl)


@router.delete("/delete/{request_id}")
async def delete_session(request_id: str):
    """
    Delete a refinement session.

    **Path Parameters:**
    - `request_id`: The session request ID

    **Returns:**
    - Success status
    """
    logger.info(f"[REFINEMENT_DELETE] request_id={request_id}")

    store = get_refinement_store()

    deleted = await store.delete(request_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"Session with request_id '{request_id}' not found or already deleted",
                "request_id": request_id
            }
        )

    logger.info(f"[REFINEMENT_DELETE] SUCCESS | request_id={request_id}")

    return {
        "status": "deleted",
        "request_id": request_id
    }


@router.post("/extend", response_model=SessionStatusResponse)
async def extend_session_ttl(request: ExtendTTLRequest):
    """
    Extend the TTL for a refinement session.

    **Request Body:**
    - `request_id`: Session request ID
    - `ttl_seconds`: New TTL in seconds (default: 7200)

    **Returns:**
    - Session status with new TTL
    """
    logger.info(f"[REFINEMENT_EXTEND] request_id={request.request_id} | ttl={request.ttl_seconds}")

    store = get_refinement_store()

    extended = await store.extend_ttl(request.request_id, request.ttl_seconds)

    if not extended:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"Session with request_id '{request.request_id}' not found or expired",
                "request_id": request.request_id
            }
        )

    data = await store.get(request.request_id)
    ttl = await store.get_ttl(request.request_id)

    logger.info(f"[REFINEMENT_EXTEND] SUCCESS | request_id={request.request_id} | new_ttl={ttl}")

    return SessionStatusResponse(
        request_id=request.request_id,
        exists=True,
        ttl_seconds=ttl,
        refinement_count=data.refinement_count if data else None,
        regeneration_count=data.regeneration_count if data else None,
        task=data.task if data else None,
        user_id=data.user_id if data else None
    )


@router.get("/status/{request_id}", response_model=SessionStatusResponse)
async def get_session_status(request_id: str):
    """
    Get the status of a refinement session.

    **Path Parameters:**
    - `request_id`: The session request ID

    **Returns:**
    - Session status including existence and TTL
    """
    logger.info(f"[REFINEMENT_STATUS] request_id={request_id}")

    store = get_refinement_store()

    exists = await store.exists(request_id)
    ttl = await store.get_ttl(request_id) if exists else -1

    data = await store.get(request_id) if exists else None

    return SessionStatusResponse(
        request_id=request_id,
        exists=exists,
        ttl_seconds=ttl,
        refinement_count=data.refinement_count if data else None,
        regeneration_count=data.regeneration_count if data else None,
        task=data.task if data else None,
        user_id=data.user_id if data else None
    )


@router.get("/config")
async def get_refinement_config():
    """
    Get the refinement service configuration.

    **Returns:**
    - Configuration settings
    """
    return {
        "key_prefix": REFINEMENT_KEY_PREFIX,
        "default_ttl_seconds": REFINEMENT_DEFAULT_TTL,
        "max_iterations": REFINEMENT_MAX_ITERATIONS,
        "max_regenerations": REFINEMENT_MAX_REGENERATIONS
    }
