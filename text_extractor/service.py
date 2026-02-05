"""
Text Extraction Service

FastAPI endpoints for extracting text from PDF, DOCX, DOC, TXT files.
Returns markdown formatted output with embedded image paths.
"""

import os
import uuid
import shutil
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from .extractor import TextExtractor
from .schemas import ExtractionRequest
from .config import (
    EXTRACTOR_UPLOAD_DIR,
    EXTRACTOR_IMAGE_DIR,
    EXTRACTOR_SUPPORTED_FILE_TYPES,
    EXTRACTOR_DEFAULT_INCLUDE_TABLES,
    EXTRACTOR_DEFAULT_INCLUDE_IMAGES,
    EXTRACTOR_DEFAULT_INCLUDE_BLOCKS,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/docAI/v1/extract", tags=["Text Extraction"])

# Ensure directories exist
Path(EXTRACTOR_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(EXTRACTOR_IMAGE_DIR).mkdir(parents=True, exist_ok=True)


# =====================
# Response Models
# =====================

class DocumentMetadataResponse(BaseModel):
    """Document metadata in response."""
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    total_pages: Optional[int] = None
    total_blocks: int
    word_count: int
    char_count: int
    extracted_at: str
    status: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None


class ExtractedBlockResponse(BaseModel):
    """Single extracted block."""
    block_id: str
    sequence: int
    content_type: str
    text: str
    page: Optional[int] = None
    level: Optional[int] = None


class ExtractionResponse(BaseModel):
    """Full extraction response."""
    request_id: str
    metadata: DocumentMetadataResponse
    markdown_text: str
    image_paths: List[str]
    blocks: Optional[List[ExtractedBlockResponse]] = None


class SimpleExtractionResponse(BaseModel):
    """Simplified response with just markdown."""
    request_id: str
    document_id: str
    filename: str
    markdown_text: str
    image_paths: List[str]
    word_count: int
    status: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None


# =====================
# Service Class
# =====================

class TextExtractionService:
    """Service wrapper for text extraction."""

    def __init__(self, upload_dir: str = EXTRACTOR_UPLOAD_DIR, image_dir: str = EXTRACTOR_IMAGE_DIR):
        self.upload_dir = Path(upload_dir)
        self.image_dir = Path(image_dir)
        self.extractor = TextExtractor()

    async def save_upload(self, file: UploadFile, document_id: str) -> Path:
        """Save uploaded file to disk."""
        # Create document directory
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = doc_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return file_path

    def extract_from_path(
        self,
        file_path: Path,
        document_id: str,
        include_tables: bool = True,
        include_images: bool = True
    ):
        """Extract text from file path."""
        # Setup image output directory
        image_output_dir = self.image_dir / document_id
        image_output_dir.mkdir(parents=True, exist_ok=True)

        request = ExtractionRequest(
            file_path=str(file_path),
            document_id=document_id,
            include_tables=include_tables,
            include_images=include_images,
            image_output_dir=str(image_output_dir)
        )

        return self.extractor.extract(request)

    def cleanup(self, document_id: str):
        """Remove uploaded file and images for a document."""
        doc_upload_dir = self.upload_dir / document_id
        doc_image_dir = self.image_dir / document_id

        if doc_upload_dir.exists():
            shutil.rmtree(doc_upload_dir)
        if doc_image_dir.exists():
            shutil.rmtree(doc_image_dir)


# Global service instance
_service: Optional[TextExtractionService] = None


def get_service() -> TextExtractionService:
    """Get or create service instance."""
    global _service
    if _service is None:
        _service = TextExtractionService()
    return _service


# =====================
# API Endpoints
# =====================

@router.post("/upload", response_model=ExtractionResponse)
async def extract_from_upload(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, DOC, TXT)"),
    request_id: Optional[str] = Form(None, description="Request ID (generated if not provided)"),
    include_tables: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_TABLES, description="Extract tables"),
    include_images: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_IMAGES, description="Extract and save images"),
    include_blocks: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_BLOCKS, description="Include individual blocks in response"),
    user_id: Optional[str] = Form(None, description="User identifier for logging"),
    user_name: Optional[str] = Form(None, description="User name for logging")
):
    """
    Extract text from uploaded document.

    - Accepts PDF, DOCX, DOC, TXT files
    - Returns markdown formatted text
    - Saves embedded images to folder
    - Image paths included in markdown

    **Form Parameters:**
    - `file`: Document file (required)
    - `request_id`: Request ID for tracking (generated if not provided)
    - `include_tables`: Extract tables (default: true)
    - `include_images`: Extract and save images (default: true)
    - `include_blocks`: Include individual blocks in response (default: false)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)
    """
    # Generate request_id if not provided (fresh request)
    request_id = request_id or str(uuid.uuid4())

    # Validate file type
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in EXTRACTOR_SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {EXTRACTOR_SUPPORTED_FILE_TYPES}"
        )

    service = get_service()
    document_id = str(uuid.uuid4())

    # Log with user info
    logger.info(f"[EXTRACT] START | request_id={request_id} | document_id={document_id} | filename={file.filename} | user_id={user_id} | user_name={user_name}")

    try:
        # Save uploaded file
        file_path = await service.save_upload(file, document_id)

        # Extract text
        result = service.extract_from_path(
            file_path=file_path,
            document_id=document_id,
            include_tables=include_tables,
            include_images=include_images
        )

        logger.info(f"[EXTRACT] END | request_id={request_id} | document_id={document_id} | blocks={result.metadata.total_blocks} | user_id={user_id}")

        # Build response
        metadata = DocumentMetadataResponse(
            document_id=result.metadata.document_id,
            filename=result.metadata.filename,
            file_type=result.metadata.file_type.value,
            file_size_bytes=result.metadata.file_size_bytes,
            total_pages=result.metadata.total_pages,
            total_blocks=result.metadata.total_blocks,
            word_count=result.metadata.word_count,
            char_count=result.metadata.char_count,
            extracted_at=result.metadata.extracted_at,
            status=result.metadata.status,
            user_id=user_id,
            user_name=user_name
        )

        blocks = None
        if include_blocks:
            blocks = [
                ExtractedBlockResponse(
                    block_id=b.block_id,
                    sequence=b.sequence,
                    content_type=b.content_type.value,
                    text=b.text,
                    page=b.page,
                    level=b.level
                )
                for b in result.blocks
            ]

        return ExtractionResponse(
            request_id=request_id,
            metadata=metadata,
            markdown_text=result.markdown_text,
            image_paths=result.image_paths,
            blocks=blocks
        )

    except FileNotFoundError as e:
        logger.error(f"[EXTRACT] ERROR | request_id={request_id} | document_id={document_id} | user_id={user_id} | error=file_not_found")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"[EXTRACT] ERROR | request_id={request_id} | document_id={document_id} | user_id={user_id} | error=value_error")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[EXTRACT] ERROR | request_id={request_id} | document_id={document_id} | user_id={user_id} | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/upload/simple", response_model=SimpleExtractionResponse)
async def extract_simple(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, DOC, TXT)"),
    request_id: Optional[str] = Form(None, description="Request ID (generated if not provided)"),
    include_images: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_IMAGES, description="Extract and save images"),
    user_id: Optional[str] = Form(None, description="User identifier for logging"),
    user_name: Optional[str] = Form(None, description="User name for logging")
):
    """
    Simple extraction endpoint - returns just markdown text.

    Lighter response without individual blocks.

    **Form Parameters:**
    - `file`: Document file (required)
    - `request_id`: Request ID for tracking (generated if not provided)
    - `include_images`: Extract and save images (default: true)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)
    """
    # Generate request_id if not provided (fresh request)
    request_id = request_id or str(uuid.uuid4())

    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )

    service = get_service()
    document_id = str(uuid.uuid4())

    logger.info(f"[EXTRACT_SIMPLE] START | request_id={request_id} | document_id={document_id} | filename={file.filename} | user_id={user_id} | user_name={user_name}")

    try:
        file_path = await service.save_upload(file, document_id)

        result = service.extract_from_path(
            file_path=file_path,
            document_id=document_id,
            include_tables=True,
            include_images=include_images
        )

        logger.info(f"[EXTRACT_SIMPLE] END | request_id={request_id} | document_id={document_id} | words={result.metadata.word_count} | user_id={user_id}")

        return SimpleExtractionResponse(
            request_id=request_id,
            document_id=result.metadata.document_id,
            filename=result.metadata.filename,
            markdown_text=result.markdown_text,
            image_paths=result.image_paths,
            word_count=result.metadata.word_count,
            status="completed",
            user_id=user_id,
            user_name=user_name
        )

    except Exception as e:
        logger.error(f"[EXTRACT_SIMPLE] ERROR | request_id={request_id} | document_id={document_id} | user_id={user_id} | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/path", response_model=ExtractionResponse)
async def extract_from_path(
    file_path: str = Form(..., description="Path to document file"),
    request_id: Optional[str] = Form(None, description="Request ID (generated if not provided)"),
    document_id: Optional[str] = Form(None, description="Custom document ID"),
    include_tables: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_TABLES),
    include_images: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_IMAGES),
    include_blocks: bool = Form(EXTRACTOR_DEFAULT_INCLUDE_BLOCKS),
    user_id: Optional[str] = Form(None, description="User identifier for logging"),
    user_name: Optional[str] = Form(None, description="User name for logging")
):
    """
    Extract text from file path on server.

    Use this when file is already on the server.

    **Form Parameters:**
    - `file_path`: Path to document file (required)
    - `request_id`: Request ID for tracking (generated if not provided)
    - `document_id`: Custom document ID (optional)
    - `include_tables`: Extract tables (default: true)
    - `include_images`: Extract and save images (default: true)
    - `include_blocks`: Include individual blocks (default: false)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)
    """
    # Generate request_id if not provided (fresh request)
    request_id = request_id or str(uuid.uuid4())

    path = Path(file_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if path.suffix.lower() not in EXTRACTOR_SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {path.suffix}. Allowed: {EXTRACTOR_SUPPORTED_FILE_TYPES}"
        )

    service = get_service()
    doc_id = document_id or str(uuid.uuid4())

    logger.info(f"[EXTRACT_PATH] START | request_id={request_id} | document_id={doc_id} | path={file_path} | user_id={user_id} | user_name={user_name}")

    try:
        result = service.extract_from_path(
            file_path=path,
            document_id=doc_id,
            include_tables=include_tables,
            include_images=include_images
        )

        logger.info(f"[EXTRACT_PATH] END | request_id={request_id} | document_id={doc_id} | blocks={result.metadata.total_blocks} | user_id={user_id}")

        metadata = DocumentMetadataResponse(
            document_id=result.metadata.document_id,
            filename=result.metadata.filename,
            file_type=result.metadata.file_type.value,
            file_size_bytes=result.metadata.file_size_bytes,
            total_pages=result.metadata.total_pages,
            total_blocks=result.metadata.total_blocks,
            word_count=result.metadata.word_count,
            char_count=result.metadata.char_count,
            extracted_at=result.metadata.extracted_at,
            status=result.metadata.status,
            user_id=user_id,
            user_name=user_name
        )

        blocks = None
        if include_blocks:
            blocks = [
                ExtractedBlockResponse(
                    block_id=b.block_id,
                    sequence=b.sequence,
                    content_type=b.content_type.value,
                    text=b.text,
                    page=b.page,
                    level=b.level
                )
                for b in result.blocks
            ]

        return ExtractionResponse(
            request_id=request_id,
            metadata=metadata,
            markdown_text=result.markdown_text,
            image_paths=result.image_paths,
            blocks=blocks
        )

    except Exception as e:
        logger.error(f"[EXTRACT_PATH] ERROR | request_id={request_id} | document_id={doc_id} | user_id={user_id} | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.delete("/cleanup/{document_id}")
async def cleanup_document(document_id: str):
    """
    Remove uploaded files and extracted images for a document.
    """
    service = get_service()
    service.cleanup(document_id)
    logger.info(f"[CLEANUP] document_id={document_id}")
    return {"status": "cleaned", "document_id": document_id}


@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported file types."""
    return {
        "supported_types": [
            {"extension": ".pdf", "description": "PDF documents"},
            {"extension": ".docx", "description": "Microsoft Word (2007+)"},
            {"extension": ".doc", "description": "Microsoft Word (legacy)"},
            {"extension": ".txt", "description": "Plain text files"}
        ]
    }
