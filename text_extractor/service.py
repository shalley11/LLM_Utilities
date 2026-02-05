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

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/docAI/v1/extract", tags=["Text Extraction"])

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/text_extractor/uploads")
IMAGE_DIR = os.getenv("IMAGE_DIR", "/tmp/text_extractor/images")

# Ensure directories exist
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)


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
    metadata: DocumentMetadataResponse
    markdown_text: str
    image_paths: List[str]
    blocks: Optional[List[ExtractedBlockResponse]] = None


class SimpleExtractionResponse(BaseModel):
    """Simplified response with just markdown."""
    document_id: str
    filename: str
    markdown_text: str
    image_paths: List[str]
    word_count: int
    status: str


# =====================
# Service Class
# =====================

class TextExtractionService:
    """Service wrapper for text extraction."""

    def __init__(self, upload_dir: str = UPLOAD_DIR, image_dir: str = IMAGE_DIR):
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
    include_tables: bool = Form(True, description="Extract tables"),
    include_images: bool = Form(True, description="Extract and save images"),
    include_blocks: bool = Form(False, description="Include individual blocks in response")
):
    """
    Extract text from uploaded document.

    - Accepts PDF, DOCX, DOC, TXT files
    - Returns markdown formatted text
    - Saves embedded images to folder
    - Image paths included in markdown
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    service = get_service()
    document_id = str(uuid.uuid4())

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
            status=result.metadata.status
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
            metadata=metadata,
            markdown_text=result.markdown_text,
            image_paths=result.image_paths,
            blocks=blocks
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/upload/simple", response_model=SimpleExtractionResponse)
async def extract_simple(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, DOC, TXT)"),
    include_images: bool = Form(True, description="Extract and save images")
):
    """
    Simple extraction endpoint - returns just markdown text.

    Lighter response without individual blocks.
    """
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )

    service = get_service()
    document_id = str(uuid.uuid4())

    try:
        file_path = await service.save_upload(file, document_id)

        result = service.extract_from_path(
            file_path=file_path,
            document_id=document_id,
            include_tables=True,
            include_images=include_images
        )

        return SimpleExtractionResponse(
            document_id=result.metadata.document_id,
            filename=result.metadata.filename,
            markdown_text=result.markdown_text,
            image_paths=result.image_paths,
            word_count=result.metadata.word_count,
            status="completed"
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/path", response_model=ExtractionResponse)
async def extract_from_path(
    file_path: str = Form(..., description="Path to document file"),
    document_id: Optional[str] = Form(None, description="Custom document ID"),
    include_tables: bool = Form(True),
    include_images: bool = Form(True),
    include_blocks: bool = Form(False)
):
    """
    Extract text from file path on server.

    Use this when file is already on the server.
    """
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    if path.suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {path.suffix}"
        )

    service = get_service()
    doc_id = document_id or str(uuid.uuid4())

    try:
        result = service.extract_from_path(
            file_path=path,
            document_id=doc_id,
            include_tables=include_tables,
            include_images=include_images
        )

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
            status=result.metadata.status
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
            metadata=metadata,
            markdown_text=result.markdown_text,
            image_paths=result.image_paths,
            blocks=blocks
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.delete("/cleanup/{document_id}")
async def cleanup_document(document_id: str):
    """
    Remove uploaded files and extracted images for a document.
    """
    service = get_service()
    service.cleanup(document_id)
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
