"""
Schemas for Chunking Module

Page-wise chunking with overlap and vision processing for images/tables.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from config import MODEL_CONTEXT_LENGTHS
from .config import (
    CHUNKING_DEFAULT_OVERLAP,
    CHUNKING_DEFAULT_RESERVE_FOR_PROMPT,
    CHUNKING_DEFAULT_CONTEXT_LENGTH,
    CHUNKING_CHARS_PER_TOKEN,
)


class ChunkType(str, Enum):
    """Type of content in a chunk."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    MIXED = "mixed"  # Contains multiple types


class ProcessingStatus(str, Enum):
    """Status of chunk processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChunkConfig:
    """
    Configuration for chunking based on model context length.
    """
    model: str = "gemma3:4b"
    context_length: Optional[int] = None
    chunk_size: Optional[int] = None  # Max characters per chunk
    chunk_overlap: int = CHUNKING_DEFAULT_OVERLAP  # Overlap between chunks in characters
    reserve_for_prompt: int = CHUNKING_DEFAULT_RESERVE_FOR_PROMPT  # Tokens reserved for system prompt
    chars_per_token: float = CHUNKING_CHARS_PER_TOKEN

    def __post_init__(self):
        """Calculate chunk_size if not provided."""
        if self.context_length is None:
            self.context_length = MODEL_CONTEXT_LENGTHS.get(
                self.model, CHUNKING_DEFAULT_CONTEXT_LENGTH
            )

        if self.chunk_size is None:
            # Calculate safe chunk size
            available_tokens = self.context_length - self.reserve_for_prompt
            # Use 50% of available for input (leave room for output)
            safe_tokens = int(available_tokens * 0.5)
            self.chunk_size = int(safe_tokens * self.chars_per_token)

    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_overlap(self) -> int:
        return self.chunk_overlap

    @classmethod
    def for_model(cls, model: str, **kwargs) -> "ChunkConfig":
        return cls(model=model, **kwargs)

    @classmethod
    def custom(cls, chunk_size: int, overlap: int = 200) -> "ChunkConfig":
        return cls(chunk_size=chunk_size, chunk_overlap=overlap)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "context_length": self.context_length,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "reserve_for_prompt": self.reserve_for_prompt
        }


@dataclass
class ContentItem:
    """
    Single content item within a chunk (text, image description, or table).
    """
    item_id: str
    content_type: ChunkType
    content: str  # Text content or vision-generated description
    original_reference: Optional[str] = None  # Original image path or table markdown
    page: Optional[int] = None
    sequence: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content_type": self.content_type.value,
            "content": self.content,
            "original_reference": self.original_reference,
            "page": self.page,
            "sequence": self.sequence,
            "metadata": self.metadata
        }


@dataclass
class Chunk:
    """
    A chunk of content for summarization_backup.
    Page-wise with overlap, contains processed content.
    """
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str  # Combined content ready for LLM
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    items: List[ContentItem] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    has_images: bool = False
    has_tables: bool = False
    overlap_with_previous: int = 0  # Characters overlapping with previous chunk
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content": self.content,
            "items": [item.to_dict() for item in self.items],
            "char_count": self.char_count,
            "word_count": self.word_count,
            "has_images": self.has_images,
            "has_tables": self.has_tables,
            "overlap_with_previous": self.overlap_with_previous,
            "status": self.status.value,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        data["status"] = ProcessingStatus(data.get("status", "completed"))
        items_data = data.pop("items", [])
        chunk = cls(**data)
        chunk.items = [
            ContentItem(
                item_id=i["item_id"],
                content_type=ChunkType(i["content_type"]),
                content=i["content"],
                original_reference=i.get("original_reference"),
                page=i.get("page"),
                sequence=i.get("sequence", 0),
                metadata=i.get("metadata", {})
            )
            for i in items_data
        ]
        return chunk


@dataclass
class ChunkingRequest:
    """Request for chunking extracted text."""
    markdown_text: str  # Output from text extraction API
    document_id: Optional[str] = None
    image_paths: Optional[List[str]] = None  # Image paths to process with vision
    model: str = "gemma3:4b"
    chunk_size: Optional[int] = None
    chunk_overlap: int = 200
    process_images: bool = True  # Use vision model for images
    process_tables: bool = True  # Use vision model for table images


@dataclass
class ChunkingResponse:
    """Response from chunking service."""
    document_id: str
    total_chunks: int
    chunks: List[Chunk]
    config: ChunkConfig
    images_processed: int = 0
    tables_processed: int = 0
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "chunks": [c.to_dict() for c in self.chunks],
            "config": self.config.to_dict(),
            "images_processed": self.images_processed,
            "tables_processed": self.tables_processed,
            "status": self.status,
            "error": self.error
        }
