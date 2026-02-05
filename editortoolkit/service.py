"""
Editor Toolkit Service

FastAPI endpoints for text editing using LLM.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from .config import (
    EDITOR_DEFAULT_MODEL,
    EDITOR_SUPPORTED_TASKS,
    EDITOR_TASK_DESCRIPTIONS,
    EDITOR_MAX_BATCH_SIZE,
    EDITOR_MAX_TOKEN_PERCENT,
)
from config import estimate_tokens, get_model_context_length
from .schemas import (
    EditorRequest,
    EditorResponse,
    RefinementRequest,
    RefinementResponse,
    BatchEditorRequest,
    BatchEditorResponse,
    EditedItem,
    SUPPORTED_TASKS,
    TASK_DESCRIPTIONS,
)
from .editor import Editor

logger = logging.getLogger(__name__)


def validate_token_count(text: str, model: str) -> None:
    """
    Validate that text does not exceed token limit for the model.

    Raises:
        HTTPException: If token count exceeds limit
    """
    estimated_tokens = estimate_tokens(text)
    model_context = get_model_context_length(model)
    max_tokens = int(model_context * (EDITOR_MAX_TOKEN_PERCENT / 100))

    if estimated_tokens > max_tokens:
        usage_percent = (estimated_tokens / max_tokens) * 100
        logger.warning(
            f"[GUARDRAIL] Token limit exceeded | tokens={estimated_tokens} | "
            f"max={max_tokens} | model={model} | usage={usage_percent:.1f}%"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "token_limit_exceeded",
                "message": f"Text contains approximately {estimated_tokens:,} tokens which exceeds the maximum allowed limit of {max_tokens:,} tokens.",
                "estimated_tokens": estimated_tokens,
                "max_tokens": max_tokens,
                "model": model,
                "model_context_length": model_context,
                "usage_percent": round(usage_percent, 2),
                "suggestion": "Please reduce the text length or split into smaller chunks for processing."
            }
        )


# Create router
router = APIRouter(prefix="/api/docAI/v1/editor", tags=["Editor Toolkit"])


# =====================
# API Endpoints
# =====================

@router.post("/edit", response_model=EditorResponse)
async def edit_text_endpoint(request: EditorRequest):
    """
    Edit text based on the specified task.

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `text`: Text to edit (required)
    - `task`: Editing task - rephrase, professional, proofread, concise (required)
    - `model`: Model to use (optional)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `original_text`: Original input text
    - `edited_text`: Edited output text
    - `task`: Task performed
    - `model`: Model used
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    model = request.model or EDITOR_DEFAULT_MODEL

    logger.info(
        f"[EDITOR] START | request_id={request_id} | "
        f"chars={len(request.text)} | task={request.task} | "
        f"user_id={request.user_id} | user_name={request.user_name}"
    )

    # Validate token count
    validate_token_count(request.text, model)

    editor = Editor(model=model)

    try:
        edited_text = await editor.edit(
            text=request.text,
            task=request.task
        )

        logger.info(
            f"[EDITOR] END | request_id={request_id} | "
            f"output_chars={len(edited_text)} | user_id={request.user_id}"
        )

        return EditorResponse(
            request_id=request_id,
            original_text=request.text,
            edited_text=edited_text,
            task=request.task,
            model=editor.model,
            char_count=len(request.text),
            word_count=len(request.text.split()),
            status="completed",
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(
            f"[EDITOR] ERROR | request_id={request_id} | "
            f"user_id={request.user_id} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Editing failed: {str(e)}")

    finally:
        await editor.close()


@router.post("/refine", response_model=RefinementResponse)
async def refine_text_endpoint(request: RefinementRequest):
    """
    Refine a previous editing result based on user feedback.

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `current_result`: Current result to refine (required)
    - `user_feedback`: User's feedback/instructions for refinement (required)
    - `task`: Original task type (required)
    - `original_text`: Original input text for context (optional)
    - `model`: Model to use (optional)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `original_result`: Result before refinement
    - `refined_result`: Result after refinement
    - `user_feedback`: Feedback applied
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    model = request.model or EDITOR_DEFAULT_MODEL

    logger.info(
        f"[EDITOR_REFINE] START | request_id={request_id} | "
        f"task={request.task} | feedback_len={len(request.user_feedback)} | "
        f"user_id={request.user_id} | user_name={request.user_name}"
    )

    # Validate token count for combined input
    combined_text = request.current_result + request.user_feedback
    if request.original_text:
        combined_text += request.original_text
    validate_token_count(combined_text, model)

    editor = Editor(model=model)

    try:
        refined_result = await editor.refine(
            current_result=request.current_result,
            user_feedback=request.user_feedback,
            task=request.task,
            original_text=request.original_text
        )

        logger.info(
            f"[EDITOR_REFINE] END | request_id={request_id} | "
            f"output_chars={len(refined_result)} | user_id={request.user_id}"
        )

        return RefinementResponse(
            request_id=request_id,
            original_result=request.current_result,
            refined_result=refined_result,
            user_feedback=request.user_feedback,
            task=request.task,
            model=editor.model,
            status="completed",
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(
            f"[EDITOR_REFINE] ERROR | request_id={request_id} | "
            f"user_id={request.user_id} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")

    finally:
        await editor.close()


@router.post("/batch", response_model=BatchEditorResponse)
async def edit_batch_endpoint(request: BatchEditorRequest):
    """
    Edit multiple texts based on the specified task.

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `texts`: List of texts to edit (required)
    - `task`: Editing task - rephrase, professional, proofread, concise (required)
    - `model`: Model to use (optional)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `edits`: List of edited items
    - `total_items`: Total number of items edited
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    model = request.model or EDITOR_DEFAULT_MODEL

    logger.info(
        f"[EDITOR_BATCH] START | request_id={request_id} | "
        f"items={len(request.texts)} | task={request.task} | "
        f"user_id={request.user_id} | user_name={request.user_name}"
    )

    # Validate token count for each text in batch
    for i, text in enumerate(request.texts):
        try:
            validate_token_count(text, model)
        except HTTPException as e:
            # Add index info to the error
            e.detail["batch_index"] = i
            raise e

    editor = Editor(model=model)

    try:
        results = await editor.edit_batch(
            texts=request.texts,
            task=request.task
        )

        edits = [
            EditedItem(
                index=i,
                original_text=original,
                edited_text=edited
            )
            for i, (original, edited) in enumerate(results)
        ]

        logger.info(
            f"[EDITOR_BATCH] END | request_id={request_id} | "
            f"items={len(edits)} | user_id={request.user_id}"
        )

        return BatchEditorResponse(
            request_id=request_id,
            edits=edits,
            task=request.task,
            model=editor.model,
            total_items=len(edits),
            status="completed",
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(
            f"[EDITOR_BATCH] ERROR | request_id={request_id} | "
            f"user_id={request.user_id} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Batch editing failed: {str(e)}")

    finally:
        await editor.close()


@router.get("/tasks")
async def get_supported_tasks():
    """
    Get list of supported editing tasks.

    **Returns:**
    - `tasks`: List of supported task names
    - `descriptions`: Task descriptions
    - `total`: Total number of supported tasks
    """
    return {
        "tasks": SUPPORTED_TASKS,
        "descriptions": TASK_DESCRIPTIONS,
        "total": len(SUPPORTED_TASKS)
    }


@router.get("/config")
async def get_editor_config():
    """
    Get the default editor configuration.

    **Returns:**
    - Default model and settings
    """
    return {
        "default_model": EDITOR_DEFAULT_MODEL,
        "supported_tasks": EDITOR_SUPPORTED_TASKS,
        "task_descriptions": EDITOR_TASK_DESCRIPTIONS,
        "max_batch_size": EDITOR_MAX_BATCH_SIZE
    }
