"""
Chunking Service

FastAPI endpoints for chunking text extraction output.
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .schemas import ChunkConfig, ChunkingRequest, ChunkingResponse
from .chunker import Chunker

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/docAI/v1/chunk", tags=["Chunking"])


# =====================
# Request/Response Models
# =====================

class ChunkRequest(BaseModel):
    """Request for chunking."""
    markdown_text: str = Field(..., description="Markdown text from text extraction API")
    document_id: Optional[str] = Field(None, description="Document ID")
    image_paths: Optional[List[str]] = Field(None, description="Image paths to process")
    model: str = Field("gemma3:4b", description="Model for chunk size calculation")
    chunk_size: Optional[int] = Field(None, description="Override chunk size (characters)")
    chunk_overlap: int = Field(200, description="Overlap between chunks (characters)")
    process_images: bool = Field(True, description="Process images with Vision model")


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
    document_id: str
    total_chunks: int
    chunks: List[ChunkInfo]
    config: dict
    images_processed: int
    status: str


class ConfigRequest(BaseModel):
    """Request for chunk config calculation."""
    model: str = Field("gemma3:4b", description="Model name")
    context_length: Optional[int] = Field(None, description="Override context length")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    reserve_for_prompt: int = Field(1000, description="Tokens reserved for prompt")


class ConfigResponse(BaseModel):
    """Calculated chunk configuration."""
    model: str
    context_length: int
    chunk_size: int
    chunk_overlap: int
    estimated_tokens_per_chunk: int


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

    Returns chunks ready for summarization_backup.
    """
    service = get_service()

    try:
        chunking_request = ChunkingRequest(
            markdown_text=request.markdown_text,
            document_id=request.document_id,
            image_paths=request.image_paths,
            model=request.model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            process_images=request.process_images
        )

        result = await service.chunk(chunking_request)

        return ChunkResponse(
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
            status=result.status
        )

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


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
        "default": "gemma3:4b"
    }
