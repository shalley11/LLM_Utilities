"""
FastAPI router for hierarchical summarization endpoints.

Pipeline Architecture:
1. Text input → Chunking Service → Summarization
2. File input → Extraction Service → Chunking Service → Summarization

Supports two input types:
1. String data (JSON body) - calls chunking service directly
2. File upload (form data) - calls extraction service, then chunking service
"""
import uuid
import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Literal, List, Dict

from .config import (
    SUMMARIZATION_DEFAULT_MODEL,
    SUMMARIZATION_DEFAULT_TYPE,
    SUMMARIZATION_MAX_WORDS_PER_BATCH,
    SUMMARIZATION_MAX_CHUNKS_PER_BATCH,
    SUMMARIZATION_INTERMEDIATE_WORDS,
    SUMMARIZATION_MAX_TOKEN_PERCENT,
)
from config import estimate_tokens, get_model_context_length
from text_extractor.config import EXTRACTOR_SUPPORTED_FILE_TYPES
from chunking.config import CHUNKING_DEFAULT_OVERLAP, CHUNKING_DEFAULT_PROCESS_IMAGES
from logs.logging_config import get_llm_logger, RequestContext
from .schemas import (
    SummarizationRequest,
    SummarizationResponse,
    TextSummarizationRequest,
)
from .summarizer import (
    summarize_chunks_async,
    SummarizerConfig,
)

logger = get_llm_logger()


def validate_token_count(text: str, model: str) -> None:
    """
    Validate that text does not exceed token limit for the model.

    Raises:
        HTTPException: If token count exceeds limit
    """
    estimated_tokens = estimate_tokens(text)
    model_context = get_model_context_length(model)
    max_tokens = int(model_context * (SUMMARIZATION_MAX_TOKEN_PERCENT / 100))

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


router = APIRouter(prefix="/api/docAI/v1/summarize", tags=["Summarization"])


# =====================
# Service Layer Functions
# =====================

async def call_extraction_service(file_path: str) -> Dict:
    """
    Call the extraction service to extract text from a file.

    Args:
        file_path: Path to the file to extract

    Returns:
        Dictionary with extracted markdown text and metadata
    """
    from text_extractor.extractor import TextExtractor

    logger.info(f"[PIPELINE] Calling extraction service | file={file_path}")

    extractor = TextExtractor()
    result = await extractor.extract(file_path)

    markdown = result.get("markdown", "") or result.get("text", "")
    image_paths = result.get("image_paths", [])

    logger.info(f"[PIPELINE] Extraction complete | chars={len(markdown)} | images={len(image_paths)}")

    return {
        "markdown": markdown,
        "image_paths": image_paths,
        "metadata": result.get("metadata", {})
    }


async def call_chunking_service(
    text: str,
    document_id: str,
    model: str = "gemma3:4b",
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 200,
    image_paths: Optional[List[str]] = None,
    process_images: bool = False
) -> List[Dict]:
    """
    Call the chunking service to split text into chunks.

    Args:
        text: Text to chunk
        document_id: Document identifier
        model: Model for chunk size calculation
        chunk_size: Override chunk size (optional)
        chunk_overlap: Overlap between chunks
        image_paths: Image paths to process (optional)
        process_images: Whether to process images with vision model

    Returns:
        List of chunk dictionaries with 'text' field
    """
    from chunking.schemas import ChunkConfig, ChunkingRequest
    from chunking.chunker import Chunker

    logger.info(f"[PIPELINE] Calling chunking service | chars={len(text)} | model={model}")

    # Create config
    config = ChunkConfig(
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Create chunker
    chunker = Chunker(config=config)

    try:
        chunks, image_descriptions = await chunker.chunk(
            markdown_text=text,
            document_id=document_id,
            image_paths=image_paths,
            process_images=process_images
        )

        # Convert to simple dict format for summarizer
        chunk_list = [
            {
                "text": c.content,
                "page_no": c.page_start,
                "chunk_index": c.chunk_index
            }
            for c in chunks
        ]

        logger.info(f"[PIPELINE] Chunking complete | chunks={len(chunk_list)} | images_processed={len(image_descriptions)}")

        return chunk_list

    finally:
        await chunker.close()


# =====================
# API Endpoints
# =====================

@router.post("/text", response_model=SummarizationResponse)
async def summarize_text_endpoint(request: TextSummarizationRequest):
    """
    Summarize text using the pipeline: Text → Chunking → Summarization

    **Pipeline Flow:**
    1. Receives text input
    2. Calls Chunking Service to split into chunks
    3. Calls Summarization to generate summary

    **Summary Types:**
    - `brief`: Short, focused summary
    - `bullets`: Bullet-point format
    - `detailed`: Comprehensive narrative
    - `executive`: High-level for leadership

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `text`: Text to summarize (min 10 characters)
    - `summary_type`: Type of summary (default: "detailed")
    - `chunk_size`: Characters per chunk (auto-calculated if not provided)
    - `model`: Model to use (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `summary`: Generated summary
    - `method`: "direct" or "hierarchical"
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())
    document_id = f"text_{request_id[:8]}"

    with RequestContext(request_id):
        text_length = len(request.text)
        model = request.model or SUMMARIZATION_DEFAULT_MODEL

        logger.info(f"[SUMMARIZE_TEXT] START | request_id={request_id} | chars={text_length} | type={request.summary_type} | user_id={request.user_id} | user_name={request.user_name}")
        logger.info(f"[SUMMARIZE_TEXT] Pipeline: Text → Chunking → Summarization")

        try:

            # Step 1: Call Chunking Service
            chunks = await call_chunking_service(
                text=request.text,
                document_id=document_id,
                model=model,
                chunk_size=request.chunk_size,
                chunk_overlap=CHUNKING_DEFAULT_OVERLAP,
                process_images=False
            )

            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks generated from text")

            # Step 2: Call Summarization
            config = SummarizerConfig(model=model)

            result = await summarize_chunks_async(
                chunks=chunks,
                summary_type=request.summary_type,
                config=config
            )

            logger.info(f"[SUMMARIZE_TEXT] END | method={result['method']} | batches={result['batches']} | user_id={request.user_id}")

            return SummarizationResponse(
                request_id=request_id,
                summary=result["summary"],
                summary_type=request.summary_type,
                method=result["method"],
                total_chunks=result["total_chunks"],
                total_words=result["total_words"],
                batches=result["batches"],
                levels=result["levels"],
                model=result["model"],
                user_id=request.user_id,
                user_name=request.user_name
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[SUMMARIZE_TEXT] ERROR | request_id={request_id} | user_id={request.user_id} | error={str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/file", response_model=SummarizationResponse)
async def summarize_file_endpoint(
    file: UploadFile = File(..., description="File to summarize (PDF, DOCX, DOC, TXT)"),
    request_id: Optional[str] = Form(None, description="Request ID (generated if not provided)"),
    summary_type: Literal["brief", "bullets", "detailed", "executive"] = Form(SUMMARIZATION_DEFAULT_TYPE, description="Type of summary"),
    model: Optional[str] = Form(None, description="Model to use (optional)"),
    process_images: bool = Form(False, description="Process images with Vision model"),
    user_id: Optional[str] = Form(None, description="User identifier for logging"),
    user_name: Optional[str] = Form(None, description="User name for logging")
):
    """
    Summarize a file using the full pipeline: File → Extraction → Chunking → Summarization

    **Pipeline Flow:**
    1. Receives file upload
    2. Calls Extraction Service to extract text from file
    3. Calls Chunking Service to split into chunks
    4. Calls Summarization to generate summary

    **Supported File Types:**
    - PDF (.pdf)
    - Word Document (.docx, .doc)
    - Text File (.txt)

    **Summary Types:**
    - `brief`: Short, focused summary with key points only
    - `bullets`: Bullet-point format covering all main topics
    - `detailed`: Comprehensive narrative summary
    - `executive`: High-level summary for leadership

    **Form Parameters:**
    - `file`: The file to summarize (required)
    - `request_id`: Request ID for tracking (generated if not provided)
    - `summary_type`: Type of summary (default: "detailed")
    - `model`: Model to use (optional)
    - `process_images`: Process images with Vision model (default: false)
    - `user_id`: User identifier for tracking (optional)
    - `user_name`: User name for tracking (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `summary`: Generated summary
    - `method`: "direct" or "hierarchical"
    - `total_chunks`: Number of chunks processed
    """
    # Generate request_id if not provided (fresh request)
    request_id = request_id or str(uuid.uuid4())

    with RequestContext(request_id):
        # Validate file type
        filename = file.filename or "unknown"
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in EXTRACTOR_SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(EXTRACTOR_SUPPORTED_FILE_TYPES)}"
            )

        logger.info(f"[SUMMARIZE_FILE] START | request_id={request_id} | filename={filename} | type={summary_type} | user_id={user_id} | user_name={user_name}")
        logger.info(f"[SUMMARIZE_FILE] Pipeline: File → Extraction → Chunking → Summarization")

        temp_path = None
        document_id = f"file_{request_id[:8]}"

        try:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name

            logger.info(f"[SUMMARIZE_FILE] Saved to temp: {temp_path}")

            model_to_use = model or SUMMARIZATION_DEFAULT_MODEL

            # Step 1: Call Extraction Service
            extraction_result = await call_extraction_service(temp_path)

            markdown = extraction_result["markdown"]
            image_paths = extraction_result.get("image_paths", [])

            if not markdown or len(markdown.strip()) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract sufficient text from the file"
                )

            # Step 2: Call Chunking Service
            chunks = await call_chunking_service(
                text=markdown,
                document_id=document_id,
                model=model_to_use,
                chunk_overlap=CHUNKING_DEFAULT_OVERLAP,
                image_paths=image_paths if process_images else None,
                process_images=process_images
            )

            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks generated from file")

            # Step 3: Call Summarization
            config = SummarizerConfig(model=model_to_use)

            result = await summarize_chunks_async(
                chunks=chunks,
                summary_type=summary_type,
                config=config
            )

            logger.info(f"[SUMMARIZE_FILE] END | method={result['method']} | batches={result['batches']} | user_id={user_id}")

            return SummarizationResponse(
                request_id=request_id,
                summary=result["summary"],
                summary_type=summary_type,
                method=result["method"],
                total_chunks=result["total_chunks"],
                total_words=result["total_words"],
                batches=result["batches"],
                levels=result["levels"],
                model=result["model"],
                user_id=user_id,
                user_name=user_name
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[SUMMARIZE_FILE] ERROR | request_id={request_id} | user_id={user_id} | error={str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


@router.post("/chunks", response_model=SummarizationResponse)
async def summarize_chunks_endpoint(request: SummarizationRequest):
    """
    Summarize pre-chunked content (direct summarization, no pipeline).

    Use this endpoint when you already have chunked content and want to
    skip the extraction and chunking steps.

    **Summary Types:**
    - `brief`: Short, focused summary with key points only
    - `bullets`: Bullet-point format covering all main topics
    - `detailed`: Comprehensive narrative summary
    - `executive`: High-level summary for leadership

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `chunks`: List of text chunks to summarize
    - `summary_type`: Type of summary (default: "detailed")
    - `model`: Model to use (optional)

    **Returns:**
    - `request_id`: Unique identifier for this request
    - `summary`: Generated summary text
    - `method`: "direct" or "hierarchical"
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    with RequestContext(request_id):
        logger.info(f"[SUMMARIZE_CHUNKS] START | request_id={request_id} | chunks={len(request.chunks)} | type={request.summary_type} | user_id={request.user_id} | user_name={request.user_name}")
        logger.info(f"[SUMMARIZE_CHUNKS] Direct summarization (no pipeline)")

        model = request.model or SUMMARIZATION_DEFAULT_MODEL

        # Validate token count for each chunk
        for i, chunk in enumerate(request.chunks):
            try:
                validate_token_count(chunk.text, model)
            except HTTPException as e:
                # Add chunk index info to the error
                e.detail["chunk_index"] = i
                e.detail["page_no"] = chunk.page_no
                raise e

        try:
            # Build config
            config = SummarizerConfig(
                max_words_per_batch=request.max_words_per_batch or SUMMARIZATION_MAX_WORDS_PER_BATCH,
                max_chunks_per_batch=request.max_chunks_per_batch or SUMMARIZATION_MAX_CHUNKS_PER_BATCH,
                intermediate_summary_words=request.intermediate_summary_words or SUMMARIZATION_INTERMEDIATE_WORDS,
                model=request.model or SUMMARIZATION_DEFAULT_MODEL
            )

            # Convert chunks to dict format
            chunks = [{"text": c.text, "page_no": c.page_no} for c in request.chunks]

            # Run summarization
            result = await summarize_chunks_async(
                chunks=chunks,
                summary_type=request.summary_type,
                config=config
            )

            logger.info(f"[SUMMARIZE_CHUNKS] END | method={result['method']} | batches={result['batches']} | user_id={request.user_id}")

            return SummarizationResponse(
                request_id=request_id,
                summary=result["summary"],
                summary_type=request.summary_type,
                method=result["method"],
                total_chunks=result["total_chunks"],
                total_words=result["total_words"],
                batches=result["batches"],
                levels=result["levels"],
                model=result["model"],
                user_id=request.user_id,
                user_name=request.user_name
            )

        except Exception as e:
            logger.error(f"[SUMMARIZE_CHUNKS] ERROR | request_id={request_id} | user_id={request.user_id} | error={str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_default_config():
    """
    Get the default summarization configuration.

    Returns the default values used for hierarchical summarization.
    """
    config = SummarizerConfig()
    return {
        "max_words_per_batch": config.max_words_per_batch,
        "max_chunks_per_batch": config.max_chunks_per_batch,
        "intermediate_summary_words": config.intermediate_summary_words,
        "final_summary_words": config.final_summary_words,
        "temperature": config.temperature,
        "model": config.model,
        "summary_types": ["brief", "bullets", "detailed", "executive"],
        "supported_file_types": list(EXTRACTOR_SUPPORTED_FILE_TYPES),
        "pipeline": {
            "text_input": "Text → Chunking → Summarization",
            "file_input": "File → Extraction → Chunking → Summarization"
        }
    }


@router.get("/types")
async def get_summary_types():
    """
    Get available summary types and their descriptions.
    """
    return {
        "types": {
            "brief": "Short, focused summary with key points only. Best for quick overviews.",
            "bullets": "Bullet-point format covering all main topics. Best for structured information.",
            "detailed": "Comprehensive narrative summary. Best for thorough understanding.",
            "executive": "High-level summary for leadership. Focuses on outcomes and implications."
        },
        "supported_file_types": list(EXTRACTOR_SUPPORTED_FILE_TYPES),
        "pipeline": {
            "text_input": ["Chunking Service", "Summarization"],
            "file_input": ["Extraction Service", "Chunking Service", "Summarization"]
        }
    }
