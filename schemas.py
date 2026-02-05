from pydantic import BaseModel, Field
from typing import Literal, Optional

# Task types supported by the API
TaskType = Literal["summary", "translate", "rephrase", "remove_repetitions"]
SummaryType = Literal["brief", "detailed", "bullets", "executive"]


class TextTaskRequest(BaseModel):
    """Request for text processing tasks."""
    text: str
    task: TaskType
    summary_type: Optional[SummaryType] = None
    target_language: Optional[str] = Field(
        None,
        description="Target language for translation tasks (e.g., 'Spanish', 'French')"
    )
    model: Optional[str] = None
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking and logging purposes"
    )
    user_name: Optional[str] = Field(
        None,
        description="Optional user name for logging and display purposes"
    )


class TextTaskResponse(BaseModel):
    """Response for text processing tasks with request_id for refinement."""
    request_id: str = Field(..., description="Unique ID for refinement cycle")
    task: str
    model: str
    output: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None


# =========================
# Refinement Models
# =========================

class RefinementRequest(BaseModel):
    """Request for refining a previous result."""
    request_id: str = Field(
        ...,
        description="Request ID from process response"
    )
    user_feedback: str = Field(
        ...,
        description="User's feedback or instructions for refinement",
        min_length=1,
        max_length=2000
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking"
    )
    user_name: Optional[str] = Field(
        None,
        description="Optional user name for logging"
    )


class RefinementResponse(BaseModel):
    """Response after refinement."""
    request_id: str = Field(..., description="The request ID")
    refined_output: str = Field(..., description="The refined result")
    refinement_count: int = Field(..., description="Total refinements made")
    task: str = Field(..., description="Original task type")
    model: str = Field(..., description="Model used")
    user_id: Optional[str] = None


class RefinementStatusResponse(BaseModel):
    """Status of a refinement session."""
    request_id: str
    task: str
    model: str
    current_output: str
    refinement_count: int = Field(..., description="Count of refinements (without original text)")
    regeneration_count: int = Field(0, description="Count of regenerations (with original text)")
    user_id: Optional[str] = None
    created_at: str
    updated_at: str
    ttl_seconds: int = Field(..., description="Remaining time-to-live in seconds")


# =========================
# Regenerate Models
# =========================

class RegenerateRequest(BaseModel):
    """Request for regenerating from original text with new instructions."""
    request_id: str = Field(
        ...,
        description="Request ID from process response"
    )
    user_feedback: str = Field(
        ...,
        description="User's instructions for regeneration (applied to original text)",
        min_length=1,
        max_length=2000
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking"
    )
    user_name: Optional[str] = Field(
        None,
        description="Optional user name for logging"
    )


class RegenerateResponse(BaseModel):
    """Response after regeneration."""
    request_id: str = Field(..., description="The request ID")
    regenerated_output: str = Field(..., description="The regenerated result from original text")
    regeneration_count: int = Field(..., description="Total regenerations made")
    task: str = Field(..., description="Original task type")
    model: str = Field(..., description="Model used")
    user_id: Optional[str] = None


# =========================
# Session Management Models
# =========================

class SessionStatusRequest(BaseModel):
    """Request for getting session status."""
    request_id: str = Field(
        ...,
        description="Request ID to check status"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking"
    )
    user_name: Optional[str] = Field(
        None,
        description="Optional user name for logging"
    )


class SessionDeleteRequest(BaseModel):
    """Request for deleting a session."""
    request_id: str = Field(
        ...,
        description="Request ID to delete"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking"
    )
    user_name: Optional[str] = Field(
        None,
        description="Optional user name for logging"
    )


class SessionExtendRequest(BaseModel):
    """Request for extending session TTL."""
    request_id: str = Field(
        ...,
        description="Request ID to extend"
    )
    ttl_seconds: int = Field(
        3600,
        description="TTL extension in seconds (default: 1 hour)"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking"
    )
    user_name: Optional[str] = Field(
        None,
        description="Optional user name for logging"
    )
