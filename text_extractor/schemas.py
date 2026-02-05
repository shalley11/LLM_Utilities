"""
Schemas for Text Extraction Module

Supports: PDF, DOCX, DOC, TXT files
Extracts text while maintaining document structure.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


class FileType(str, Enum):
    """Supported file types for extraction."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"


class ContentType(str, Enum):
    """Types of content extracted from documents."""
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class ExtractedBlock:
    """
    Single block of extracted content with structure preserved.
    """
    block_id: str
    document_id: str
    sequence: int
    content_type: ContentType
    text: str
    page: Optional[int] = None
    level: Optional[int] = None  # For headings (1-6)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "document_id": self.document_id,
            "sequence": self.sequence,
            "content_type": self.content_type.value,
            "text": self.text,
            "page": self.page,
            "level": self.level,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedBlock":
        data["content_type"] = ContentType(data["content_type"])
        return cls(**data)


@dataclass
class DocumentMetadata:
    """Document information and extraction summary."""
    document_id: str
    filename: str
    file_type: FileType
    file_size_bytes: int
    total_pages: Optional[int] = None
    total_blocks: int = 0
    word_count: int = 0
    char_count: int = 0
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type.value,
            "file_size_bytes": self.file_size_bytes,
            "total_pages": self.total_pages,
            "total_blocks": self.total_blocks,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "extracted_at": self.extracted_at,
            "status": self.status,
            "error": self.error
        }


@dataclass
class ExtractionRequest:
    """Request for text extraction."""
    file_path: str
    document_id: Optional[str] = None
    user_id: Optional[str] = None
    include_tables: bool = True
    include_images: bool = True
    image_output_dir: Optional[str] = None  # Directory to save extracted images


@dataclass
class ExtractionResult:
    """Result of text extraction with structured content."""
    metadata: DocumentMetadata
    blocks: List[ExtractedBlock]
    markdown_text: str  # Full text in markdown format
    image_paths: List[str] = field(default_factory=list)  # Paths to extracted images

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "blocks": [b.to_dict() for b in self.blocks],
            "markdown_text": self.markdown_text,
            "image_paths": self.image_paths
        }
