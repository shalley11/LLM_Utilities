"""
Chunking Module

Page-wise chunking with overlap for text extraction output.
- Processes images with Vision Gemma3
- Replaces image paths with descriptions
- Creates chunks ready for summarization
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
from .config import (
    CHUNKING_DEFAULT_OVERLAP,
    CHUNKING_DEFAULT_RESERVE_FOR_PROMPT,
    CHUNKING_DEFAULT_PROCESS_IMAGES,
    CHUNKING_MIN_TEXT_LENGTH,
    CHUNKING_DEFAULT_CONTEXT_LENGTH,
    CHUNKING_CHARS_PER_TOKEN,
    CHUNKING_MAX_BATCH_SIZE,
    VISION_MODEL,
    VISION_OLLAMA_URL,
    VISION_REQUEST_TIMEOUT,
    VISION_MAX_CONCURRENT,
    VISION_TEMPERATURE,
    VISION_MAX_TOKENS,
)

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
    "get_service",
    # Config
    "CHUNKING_DEFAULT_OVERLAP",
    "CHUNKING_DEFAULT_RESERVE_FOR_PROMPT",
    "CHUNKING_DEFAULT_PROCESS_IMAGES",
    "CHUNKING_MIN_TEXT_LENGTH",
    "CHUNKING_DEFAULT_CONTEXT_LENGTH",
    "CHUNKING_CHARS_PER_TOKEN",
    "CHUNKING_MAX_BATCH_SIZE",
    "VISION_MODEL",
    "VISION_OLLAMA_URL",
    "VISION_REQUEST_TIMEOUT",
    "VISION_MAX_CONCURRENT",
    "VISION_TEMPERATURE",
    "VISION_MAX_TOKENS",
]
