"""
Chunking Module

Page-wise chunking with overlap for text extraction output.
- Processes images with Vision Gemma3
- Replaces image paths with descriptions
- Creates chunks ready for summarization_backup
"""

from .schemas import (
    ChunkType,
    ChunkConfig,
    ContentItem,
    Chunk,
    ChunkingRequest,
    ChunkingResponse,
    ProcessingStatus
)
from .chunker import Chunker, chunk_markdown
from .vision_processor import VisionProcessor, get_image_description
from .service import router, ChunkingService, get_service

__all__ = [
    # Schemas
    "ChunkType",
    "ChunkConfig",
    "ContentItem",
    "Chunk",
    "ChunkingRequest",
    "ChunkingResponse",
    "ProcessingStatus",
    # Chunker
    "Chunker",
    "chunk_markdown",
    # Vision
    "VisionProcessor",
    "get_image_description",
    # Service
    "router",
    "ChunkingService",
    "get_service"
]
