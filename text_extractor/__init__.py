"""
Text Extractor Module

Extracts text from PDF, DOCX, DOC, TXT files.
- Output is in markdown format
- Embedded images are saved to folder with paths in markdown
- Document structure is preserved (headings, paragraphs, tables, lists)
"""

from .schemas import (
    FileType,
    ContentType,
    ExtractedBlock,
    DocumentMetadata,
    ExtractionRequest,
    ExtractionResult
)
from .extractor import TextExtractor
from .service import router, TextExtractionService, get_service

__all__ = [
    "TextExtractor",
    "FileType",
    "ContentType",
    "ExtractedBlock",
    "DocumentMetadata",
    "ExtractionRequest",
    "ExtractionResult",
    "router",
    "TextExtractionService",
    "get_service"
]
