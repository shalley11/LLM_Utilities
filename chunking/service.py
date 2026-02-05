"""
Chunking Service

FastAPI endpoints for chunking text extraction output.

Supports two input types:
1. String data (JSON body) - markdown text or plain text
2. File upload (form data) - PDF, DOCX, DOC, TXT
"""

import os
import uuid
import logging
import tempfile
from typing import Optional, List, Literal

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from .schemas import ChunkConfig, ChunkingRequest, ChunkingResponse
from .chunker import Chunker
from .config import (
    CHUNKING_DEFAULT_OVERLAP,
    CHUNKING_DEFAULT_RESERVE_FOR_PROMPT,
    CHUNKING_DEFAULT_PROCESS_IMAGES,
    CHUNKING_MIN_TEXT_LENGTH,
)
from config import DEFAULT_MODEL
from text_extractor.config import EXTRACTOR_SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/docAI/v1/chunk", tags=["Chunking"])


# =====================
# Request/Response Models
# =====================

class ChunkRequest(BaseModel):
    """Request for chunking markdown text."""
    request_id: Optional[str] = Field(None, description="Request ID (generated if not provided)")
    markdown_text: str = Field(..., description="Markdown text from text extraction API")
    document_id: Optional[str] = Field(None, description="Document ID")
    image_paths: Optional[List[str]] = Field(None, description="Image paths to process")
    model: str = Field(DEFAULT_MODEL, description="Model for chunk size calculation")
    chunk_size: Optional[int] = Field(None, description="Override chunk size (characters)")
    chunk_overlap: int = Field(CHUNKING_DEFAULT_OVERLAP, description="Overlap between chunks (characters)")
    process_images: bool = Field(CHUNKING_DEFAULT_PROCESS_IMAGES, description="Process images with Vision model")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class TextChunkRequest(BaseModel):
    """Request for chunking plain text."""
    request_id: Optional[str] = Field(None, description="Request ID (generated if not provided)")
    text: str = Field(..., description="Text to chunk", min_length=CHUNKING_MIN_TEXT_LENGTH)
    document_id: Optional[str] = Field(None, description="Document ID")
    model: str = Field(DEFAULT_MODEL, description="Model for chunk size calculation")
    chunk_size: Optional[int] = Field(None, description="Override chunk size (characters)")
    chunk_overlap: int = Field(CHUNKING_DEFAULT_OVERLAP, description="Overlap between chunks (characters)")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class ChunkInfo(BaseModel):
    """Single chunk information."""
    chunk_id: str
    chunk_index: int
    page_start: Optional[int]
    page_end: Optional[int]
    content: str
    char_count: int
    word_count: int
    has_images: bool
    has_tables: bool
    overlap_with_previous: int


class ChunkResponse(BaseModel):
    """Response from chunking."""
    request_id: str
    document_id: str
    total_chunks: int
    chunks: List[ChunkInfo]
    config: dict
    images_processed: int
    status: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None


class ConfigRequest(BaseModel):
    """Request for chunk config calculation."""
    model: str = Field(DEFAULT_MODEL, description="Model name")
    context_length: Optional[int] = Field(None, description="Override context length")
    chunk_overlap: int = Field(CHUNKING_DEFAULT_OVERLAP, description="Overlap between chunks")
    reserve_for_prompt: int = Field(CHUNKING_DEFAULT_RESERVE_FOR_PROMPT, description="Tokens reserved for prompt")


class ConfigResponse(BaseModel):
    """Calculated chunk configuration."""
    model: str
    context_length: int
    chunk_size: int
    chunk_overlap: int
    estimated_tokens_per_chunk: int


# =====================
# Helper Functions
# =====================

async def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file using text_extractor module."""
    from text_extractor.extractor import TextExtractor
    from text_extractor.schemas import ExtractionRequest

    extractor = TextExtractor()
    request = ExtractionRequest(
        file_path=file_path,
        include_images=True,
        include_tables=True
    )
    result = extractor.extract(request)
    return result.markdown_text or ""


# =====================
# Service
# =====================

class ChunkingService:
    """Service for chunking operations."""

    def __init__(self):
        self._chunker: Optional[Chunker] = None

    async def chunk(self, request: ChunkingRequest) -> ChunkingResponse:
        """Process chunking request."""
        # Create config
        config = ChunkConfig(
            model=request.model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        # Create chunker
        chunker = Chunker(config=config)

        try:
            chunks, image_descriptions = await chunker.chunk(
                markdown_text=request.markdown_text,
                document_id=request.document_id,
                image_paths=request.image_paths,
                process_images=request.process_images
            )

            return ChunkingResponse(
                document_id=chunks[0].document_id if chunks else request.document_id or "unknown",
                total_chunks=len(chunks),
                chunks=chunks,
                config=config,
                images_processed=len(image_descriptions),
                status="completed"
            )

        except Exception as e:
            logger.error(f"Chunking failed: {e}", exc_info=True)
            raise

        finally:
            await chunker.close()


# Global service
_service: Optional[ChunkingService] = None


def get_service() -> ChunkingService:
    """Get or create service instance."""
    global _service
    if _service is None:
        _service = ChunkingService()
    return _service


# =====================
# API Endpoints
# =====================

@router.post("/process", response_model=ChunkResponse)
async def create_chunks(request: ChunkRequest):
    """
    Create chunks from markdown text.

    - Takes markdown output from text extraction API
    - Creates page-wise chunks with overlap
    - Processes images with Vision Gemma3 (if enabled)
    - Replaces image paths with descriptions

    Returns chunks ready for summarization.

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `markdown_text`: Markdown text to chunk (required)
    - `document_id`: Document identifier (optional)
    - `image_paths`: Image paths to process (optional)
    - `model`: Model for chunk size calculation (default: gemma3:4b)
    - `chunk_size`: Override chunk size (optional)
    - `chunk_overlap`: Overlap between chunks (default: 200)
    - `process_images`: Process images with Vision model (default: true)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    service = get_service()
    document_id = request.document_id or str(uuid.uuid4())

    logger.info(f"[CHUNK_PROCESS] START | request_id={request_id} | document_id={document_id} | chars={len(request.markdown_text)} | user_id={request.user_id} | user_name={request.user_name}")

    try:
        chunking_request = ChunkingRequest(
            markdown_text=request.markdown_text,
            document_id=document_id,
            image_paths=request.image_paths,
            model=request.model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            process_images=request.process_images
        )

        result = await service.chunk(chunking_request)

        logger.info(f"[CHUNK_PROCESS] END | request_id={request_id} | document_id={document_id} | chunks={result.total_chunks} | user_id={request.user_id}")

        return ChunkResponse(
            request_id=request_id,
            document_id=result.document_id,
            total_chunks=result.total_chunks,
            chunks=[
                ChunkInfo(
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    content=c.content,
                    char_count=c.char_count,
                    word_count=c.word_count,
                    has_images=c.has_images,
                    has_tables=c.has_tables,
                    overlap_with_previous=c.overlap_with_previous
                )
                for c in result.chunks
            ],
            config=result.config.to_dict(),
            images_processed=result.images_processed,
            status=result.status,
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(f"[CHUNK_PROCESS] ERROR | request_id={request_id} | document_id={document_id} | user_id={request.user_id} | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@router.post("/text", response_model=ChunkResponse)
async def chunk_text(request: TextChunkRequest):
    """
    Create chunks from plain text.

    - Takes plain text input
    - Creates chunks with overlap
    - Does not process images

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `text`: Text to chunk (min 10 characters)
    - `document_id`: Optional document identifier
    - `model`: Model for chunk size calculation (default: gemma3:4b)
    - `chunk_size`: Override chunk size in characters (optional)
    - `chunk_overlap`: Overlap between chunks (default: 200)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Request identifier
    - `document_id`: Document identifier
    - `total_chunks`: Number of chunks created
    - `chunks`: List of chunk objects with content and metadata
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    service = get_service()
    document_id = request.document_id or str(uuid.uuid4())

    logger.info(f"[CHUNK_TEXT] START | request_id={request_id} | document_id={document_id} | chars={len(request.text)} | user_id={request.user_id} | user_name={request.user_name}")

    try:
        chunking_request = ChunkingRequest(
            markdown_text=request.text,
            document_id=document_id,
            image_paths=None,
            model=request.model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            process_images=False  # No images in plain text
        )

        result = await service.chunk(chunking_request)

        logger.info(f"[CHUNK_TEXT] END | request_id={request_id} | document_id={document_id} | chunks={result.total_chunks} | user_id={request.user_id}")

        return ChunkResponse(
            request_id=request_id,
            document_id=result.document_id,
            total_chunks=result.total_chunks,
            chunks=[
                ChunkInfo(
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    content=c.content,
                    char_count=c.char_count,
                    word_count=c.word_count,
                    has_images=c.has_images,
                    has_tables=c.has_tables,
                    overlap_with_previous=c.overlap_with_previous
                )
                for c in result.chunks
            ],
            config=result.config.to_dict(),
            images_processed=0,
            status=result.status,
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(f"[CHUNK_TEXT] ERROR | request_id={request_id} | document_id={document_id} | user_id={request.user_id} | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@router.post("/file", response_model=ChunkResponse)
async def chunk_file(
    file: UploadFile = File(..., description="File to chunk (PDF, DOCX, DOC, TXT)"),
    request_id: Optional[str] = Form(None, description="Request ID (generated if not provided)"),
    model: str = Form(DEFAULT_MODEL, description="Model for chunk size calculation"),
    chunk_size: Optional[int] = Form(None, description="Override chunk size (characters)"),
    chunk_overlap: int = Form(CHUNKING_DEFAULT_OVERLAP, description="Overlap between chunks (characters)"),
    process_images: bool = Form(CHUNKING_DEFAULT_PROCESS_IMAGES, description="Process images with Vision model"),
    user_id: Optional[str] = Form(None, description="User identifier for logging"),
    user_name: Optional[str] = Form(None, description="User name for logging")
):
    """
    Create chunks from an uploaded file.

    Upload a file (PDF, DOCX, DOC, or TXT), extract text, and create chunks.

    **Supported File Types:**
    - PDF (.pdf)
    - Word Document (.docx, .doc)
    - Text File (.txt)

    **Form Parameters:**
    - `file`: The file to chunk (required)
    - `request_id`: Request ID for tracking (generated if not provided)
    - `model`: Model for chunk size calculation (default: gemma3:4b)
    - `chunk_size`: Override chunk size in characters (optional)
    - `chunk_overlap`: Overlap between chunks (default: 200)
    - `process_images`: Process images with Vision model (default: true)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Request identifier
    - `document_id`: Generated document identifier
    - `total_chunks`: Number of chunks created
    - `chunks`: List of chunk objects with content and metadata
    - `images_processed`: Number of images processed
    """
    # Generate request_id if not provided (fresh request)
    request_id = request_id or str(uuid.uuid4())

    # Validate file type
    filename = file.filename or "unknown"
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in EXTRACTOR_SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(EXTRACTOR_SUPPORTED_FILE_TYPES)}"
        )

    document_id = str(uuid.uuid4())
    logger.info(f"[CHUNK_FILE] START | request_id={request_id} | document_id={document_id} | filename={filename} | user_id={user_id} | user_name={user_name}")

    temp_path = None
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        logger.info(f"[CHUNK_FILE] Saved to temp: {temp_path}")

        # Extract text from file
        text = await extract_text_from_file(temp_path)

        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from the file"
            )

        logger.info(f"[CHUNK_FILE] Extracted {len(text)} characters")

        # Create chunking request
        service = get_service()
        chunking_request = ChunkingRequest(
            markdown_text=text,
            document_id=document_id,
            image_paths=None,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            process_images=process_images
        )

        result = await service.chunk(chunking_request)

        logger.info(f"[CHUNK_FILE] END | request_id={request_id} | document_id={document_id} | chunks={result.total_chunks} | user_id={user_id}")

        return ChunkResponse(
            request_id=request_id,
            document_id=result.document_id,
            total_chunks=result.total_chunks,
            chunks=[
                ChunkInfo(
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    content=c.content,
                    char_count=c.char_count,
                    word_count=c.word_count,
                    has_images=c.has_images,
                    has_tables=c.has_tables,
                    overlap_with_previous=c.overlap_with_previous
                )
                for c in result.chunks
            ],
            config=result.config.to_dict(),
            images_processed=result.images_processed,
            status=result.status,
            user_id=user_id,
            user_name=user_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHUNK_FILE] ERROR | request_id={request_id} | document_id={document_id} | user_id={user_id} | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.post("/config", response_model=ConfigResponse)
async def calculate_config(request: ConfigRequest):
    """
    Calculate optimal chunk configuration for a model.

    Use this to preview chunk settings before processing.
    """
    config = ChunkConfig(
        model=request.model,
        context_length=request.context_length,
        chunk_overlap=request.chunk_overlap,
        reserve_for_prompt=request.reserve_for_prompt
    )

    return ConfigResponse(
        model=config.model,
        context_length=config.context_length,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        estimated_tokens_per_chunk=int(config.chunk_size / config.chars_per_token)
    )


@router.get("/models")
async def list_supported_models():
    """List models with known context lengths."""
    from .schemas import MODEL_CONTEXT_LENGTHS

    return {
        "models": [
            {"name": name, "context_length": length}
            for name, length in MODEL_CONTEXT_LENGTHS.items()
        ],
        "default": DEFAULT_MODEL,
        "supported_file_types": list(EXTRACTOR_SUPPORTED_FILE_TYPES)
    }
