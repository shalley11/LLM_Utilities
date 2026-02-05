"""
Schemas for Editor Toolkit service.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from .config import (
    EDITOR_SUPPORTED_TASKS,
    EDITOR_TASK_DESCRIPTIONS,
)

# Re-export for backwards compatibility
SUPPORTED_TASKS = EDITOR_SUPPORTED_TASKS
TASK_DESCRIPTIONS = EDITOR_TASK_DESCRIPTIONS


class EditorRequest(BaseModel):
    """Request model for text editing."""
    request_id: Optional[str] = Field(None, description="Request ID for tracking (generated if not provided)")
    text: str = Field(..., description="Text to edit", min_length=1)
    task: Literal["rephrase", "professional", "proofread", "concise"] = Field(
        ..., description="Editing task to perform"
    )
    model: Optional[str] = Field(None, description="Model to use (defaults to config)")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class EditorResponse(BaseModel):
    """Response model for text editing."""
    request_id: str = Field(..., description="Request ID for tracking")
    original_text: str = Field(..., description="Original input text")
    edited_text: str = Field(..., description="Edited output text")
    task: str = Field(..., description="Task performed")
    model: str = Field(..., description="Model used")
    char_count: int = Field(..., description="Character count of input")
    word_count: int = Field(..., description="Word count of input")
    status: str = Field(default="completed", description="Processing status")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field(None, description="User name")


class RefinementRequest(BaseModel):
    """Request model for refining a previous result."""
    request_id: Optional[str] = Field(None, description="Request ID for tracking (generated if not provided)")
    current_result: str = Field(..., description="Current result to refine", min_length=1)
    user_feedback: str = Field(..., description="User's feedback/instructions for refinement", min_length=1)
    task: Literal["rephrase", "professional", "proofread", "concise"] = Field(
        ..., description="Original task type"
    )
    original_text: Optional[str] = Field(None, description="Original input text for context")
    model: Optional[str] = Field(None, description="Model to use (defaults to config)")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class RefinementResponse(BaseModel):
    """Response model for refinement."""
    request_id: str = Field(..., description="Request ID for tracking")
    original_result: str = Field(..., description="Original result before refinement")
    refined_result: str = Field(..., description="Refined result after applying feedback")
    user_feedback: str = Field(..., description="User's feedback applied")
    task: str = Field(..., description="Original task type")
    model: str = Field(..., description="Model used")
    status: str = Field(default="completed", description="Processing status")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field(None, description="User name")


class BatchEditorRequest(BaseModel):
    """Request model for batch text editing."""
    request_id: Optional[str] = Field(None, description="Request ID for tracking (generated if not provided)")
    texts: List[str] = Field(..., description="List of texts to edit", min_items=1, max_items=50)
    task: Literal["rephrase", "professional", "proofread", "concise"] = Field(
        ..., description="Editing task to perform"
    )
    model: Optional[str] = Field(None, description="Model to use (defaults to config)")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class EditedItem(BaseModel):
    """Individual edited item in batch response."""
    index: int = Field(..., description="Item index in original list")
    original_text: str = Field(..., description="Original text")
    edited_text: str = Field(..., description="Edited text")


class BatchEditorResponse(BaseModel):
    """Response model for batch text editing."""
    request_id: str = Field(..., description="Request ID for tracking")
    edits: List[EditedItem] = Field(..., description="List of edited items")
    task: str = Field(..., description="Task performed")
    model: str = Field(..., description="Model used")
    total_items: int = Field(..., description="Total number of items processed")
    status: str = Field(default="completed", description="Processing status")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field(None, description="User name")
