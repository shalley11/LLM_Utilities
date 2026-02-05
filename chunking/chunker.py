"""
Chunker

Page-wise chunking with overlap.
Takes markdown text from text extraction API and creates chunks.
Processes images with Vision Gemma3 to replace paths with descriptions.
"""

import re
import uuid
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .schemas import (
    ChunkType,
    ChunkConfig,
    ContentItem,
    Chunk,
    ProcessingStatus
)
from .vision_processor import VisionProcessor

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Parsed content from a single page."""
    page_number: int
    content: str
    images: List[str]  # Image paths found on this page
    tables: List[str]  # Table content on this page
    char_count: int


class Chunker:
    """
    Creates page-wise chunks with overlap from markdown text.
    Processes images using Vision model.
    """

    # Regex patterns
    PAGE_PATTERN = re.compile(r'^## Page (\d+)', re.MULTILINE)
    IMAGE_PATTERN = re.compile(r'!\[image\]\(([^)]+)\)')
    TABLE_PATTERN = re.compile(r'(\|[^\n]+\|\n)+', re.MULTILINE)

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        vision_processor: Optional[VisionProcessor] = None
    ):
        """
        Initialize chunker.

        Args:
            config: Chunk configuration (uses defaults if None)
            vision_processor: Vision processor for images (created if None)
        """
        self.config = config or ChunkConfig()
        self._vision_processor = vision_processor

    async def _get_vision_processor(self) -> VisionProcessor:
        """Get or create vision processor."""
        if self._vision_processor is None:
            self._vision_processor = VisionProcessor(model=self.config.model)
        return self._vision_processor

    async def close(self):
        """Cleanup resources."""
        if self._vision_processor:
            await self._vision_processor.close()

    def parse_pages(self, markdown_text: str) -> List[ParsedPage]:
        """
        Parse markdown text into pages.

        Args:
            markdown_text: Markdown output from text extraction

        Returns:
            List of ParsedPage objects
        """
        pages = []

        # Find all page markers
        page_matches = list(self.PAGE_PATTERN.finditer(markdown_text))

        if not page_matches:
            # No page markers - treat entire text as single page
            images = self.IMAGE_PATTERN.findall(markdown_text)
            tables = self.TABLE_PATTERN.findall(markdown_text)
            pages.append(ParsedPage(
                page_number=1,
                content=markdown_text,
                images=images,
                tables=tables,
                char_count=len(markdown_text)
            ))
            return pages

        # Extract content for each page
        for i, match in enumerate(page_matches):
            page_num = int(match.group(1))
            start = match.end()

            # Find end of this page (start of next page or end of text)
            if i + 1 < len(page_matches):
                end = page_matches[i + 1].start()
            else:
                end = len(markdown_text)

            page_content = markdown_text[start:end].strip()

            # Find images and tables on this page
            images = self.IMAGE_PATTERN.findall(page_content)
            tables = self.TABLE_PATTERN.findall(page_content)

            pages.append(ParsedPage(
                page_number=page_num,
                content=page_content,
                images=images,
                tables=tables,
                char_count=len(page_content)
            ))

        return pages

    async def process_images(
        self,
        image_paths: List[str],
        process_images: bool = True
    ) -> Dict[str, str]:
        """
        Process images with vision model.

        Args:
            image_paths: List of image file paths
            process_images: Whether to actually call vision model

        Returns:
            Dict mapping image_path to description
        """
        if not process_images or not image_paths:
            # Return placeholder descriptions
            return {path: f"[Image: {path}]" for path in image_paths}

        vision = await self._get_vision_processor()
        return await vision.process_images_batch(image_paths)

    def replace_images_with_descriptions(
        self,
        content: str,
        image_descriptions: Dict[str, str]
    ) -> str:
        """
        Replace image markdown with descriptions.

        Args:
            content: Markdown content with image references
            image_descriptions: Dict mapping paths to descriptions

        Returns:
            Content with image descriptions instead of paths
        """
        def replace_image(match):
            image_path = match.group(1)
            description = image_descriptions.get(
                image_path,
                f"[Image: {image_path}]"
            )
            return f"\n[IMAGE DESCRIPTION]: {description}\n"

        return self.IMAGE_PATTERN.sub(replace_image, content)

    def create_chunks_from_pages(
        self,
        pages: List[ParsedPage],
        document_id: str,
        image_descriptions: Dict[str, str]
    ) -> List[Chunk]:
        """
        Create chunks from parsed pages with overlap.

        Args:
            pages: List of ParsedPage objects
            document_id: Document identifier
            image_descriptions: Dict of image path to description

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        current_content = ""
        current_items = []
        current_page_start = None
        current_page_end = None
        item_sequence = 0
        chunk_index = 0

        for page in pages:
            # Replace images in page content
            page_content = self.replace_images_with_descriptions(
                page.content,
                image_descriptions
            )

            # Add page marker
            page_text = f"\n--- Page {page.page_number} ---\n{page_content}"

            if current_page_start is None:
                current_page_start = page.page_number

            current_page_end = page.page_number

            # Check if adding this page exceeds chunk size
            if len(current_content) + len(page_text) > chunk_size and current_content:
                # Create chunk from current content
                chunk = self._create_chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=current_content,
                    items=current_items,
                    page_start=current_page_start,
                    page_end=current_page_end - 1,  # Previous page
                    overlap_size=0 if chunk_index == 0 else overlap
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                overlap_content = current_content[-overlap:] if overlap > 0 else ""
                current_content = overlap_content + page_text
                current_items = []
                current_page_start = page.page_number
            else:
                current_content += page_text

            # Track content items
            # Add text item
            current_items.append(ContentItem(
                item_id=f"{document_id}_item_{item_sequence:05d}",
                content_type=ChunkType.TEXT,
                content=page_content,
                page=page.page_number,
                sequence=item_sequence
            ))
            item_sequence += 1

            # Add image items
            for img_path in page.images:
                desc = image_descriptions.get(img_path, f"[Image: {img_path}]")
                current_items.append(ContentItem(
                    item_id=f"{document_id}_item_{item_sequence:05d}",
                    content_type=ChunkType.IMAGE,
                    content=desc,
                    original_reference=img_path,
                    page=page.page_number,
                    sequence=item_sequence
                ))
                item_sequence += 1

        # Don't forget the last chunk
        if current_content:
            chunk = self._create_chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=current_content,
                items=current_items,
                page_start=current_page_start,
                page_end=current_page_end,
                overlap_size=0 if chunk_index == 0 else overlap
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        items: List[ContentItem],
        page_start: int,
        page_end: int,
        overlap_size: int
    ) -> Chunk:
        """Create a Chunk object."""
        has_images = any(i.content_type == ChunkType.IMAGE for i in items)
        has_tables = any(i.content_type == ChunkType.TABLE for i in items)

        return Chunk(
            chunk_id=f"{document_id}_chunk_{chunk_index:03d}",
            document_id=document_id,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
            content=content.strip(),
            items=items,
            char_count=len(content),
            word_count=len(content.split()),
            has_images=has_images,
            has_tables=has_tables,
            overlap_with_previous=overlap_size,
            status=ProcessingStatus.COMPLETED
        )

    async def chunk(
        self,
        markdown_text: str,
        document_id: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        process_images: bool = True
    ) -> Tuple[List[Chunk], Dict[str, str]]:
        """
        Main chunking method.

        Args:
            markdown_text: Markdown output from text extraction
            document_id: Optional document ID (generated if not provided)
            image_paths: Optional list of image paths to process
            process_images: Whether to call vision model for images

        Returns:
            Tuple of (chunks, image_descriptions)
        """
        document_id = document_id or str(uuid.uuid4())

        logger.info(f"Starting chunking for document: {document_id}")
        logger.info(f"Config: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

        # Parse pages
        pages = self.parse_pages(markdown_text)
        logger.info(f"Parsed {len(pages)} pages")

        # Collect all image paths
        all_images = set(image_paths or [])
        for page in pages:
            all_images.update(page.images)

        # Process images with vision model
        image_descriptions = await self.process_images(
            list(all_images),
            process_images=process_images
        )
        logger.info(f"Processed {len(image_descriptions)} images")

        # Create chunks
        chunks = self.create_chunks_from_pages(pages, document_id, image_descriptions)
        logger.info(f"Created {len(chunks)} chunks")

        return chunks, image_descriptions


# Convenience function
async def chunk_markdown(
    markdown_text: str,
    model: str = "gemma3:4b",
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 200,
    process_images: bool = True
) -> List[Chunk]:
    """
    Quick function to chunk markdown text.

    Args:
        markdown_text: Markdown from text extraction
        model: Model for context length calculation
        chunk_size: Override chunk size
        chunk_overlap: Overlap between chunks
        process_images: Whether to process images with vision

    Returns:
        List of Chunk objects
    """
    config = ChunkConfig(
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunker = Chunker(config=config)
    try:
        chunks, _ = await chunker.chunk(
            markdown_text,
            process_images=process_images
        )
        return chunks
    finally:
        await chunker.close()
