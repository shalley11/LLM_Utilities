"""
Vision Processor

Calls Vision Gemma3 model to get descriptions for images and tables.
Supports Ollama backend.
"""

import base64
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict

import aiohttp

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434"
DEFAULT_VISION_MODEL = "gemma3:4b"
REQUEST_TIMEOUT = 120


class VisionProcessor:
    """
    Processes images using Vision Gemma3 model via Ollama.
    Returns text descriptions for images and tables.
    """

    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        ollama_url: str = OLLAMA_URL,
        max_concurrent: int = 3,
        timeout: int = REQUEST_TIMEOUT
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _read_image_as_base64(self, image_path: str) -> Optional[str]:
        """Read image file and convert to base64."""
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return None

    def _get_image_prompt(self) -> str:
        """Prompt for image description."""
        return """Describe this image in detail. Include:
- Main subject or content
- Key visual elements
- Any text visible in the image
- Context or purpose if apparent

Provide a clear, concise description."""

    def _get_table_prompt(self) -> str:
        """Prompt for table description."""
        return """Analyze this table image and describe:
- What data the table contains
- Column headers and their meaning
- Key data points or trends
- The purpose of this table

Provide a structured summary."""

    async def process_image(
        self,
        image_path: str,
        is_table: bool = False,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Process a single image and return description.

        Args:
            image_path: Path to image file
            is_table: Whether this is a table image
            custom_prompt: Optional custom prompt

        Returns:
            Text description of the image
        """
        # Read image
        image_b64 = self._read_image_as_base64(image_path)
        if not image_b64:
            return f"[Image not found: {image_path}]"

        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
        elif is_table:
            prompt = self._get_table_prompt()
        else:
            prompt = self._get_image_prompt()

        # Call Ollama vision API
        async with self._semaphore:
            return await self._call_ollama_vision(image_b64, prompt)

    async def _call_ollama_vision(self, image_b64: str, prompt: str) -> str:
        """Call Ollama API with image."""
        session = await self._get_session()

        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500
            }
        }

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama error: {response.status} - {error_text}")
                    return f"[Vision processing failed: {response.status}]"

                result = await response.json()
                return result.get("response", "").strip()

        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            return "[Vision processing timed out]"
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            return f"[Vision service unavailable: {str(e)}]"
        except Exception as e:
            logger.error(f"Vision processing error: {e}")
            return f"[Vision processing error: {str(e)}]"

    async def process_images_batch(
        self,
        image_paths: List[str],
        table_paths: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Process multiple images concurrently.

        Args:
            image_paths: List of image file paths
            table_paths: List of table image paths (for table-specific prompts)

        Returns:
            Dict mapping image_path to description
        """
        table_paths = set(table_paths or [])
        results = {}

        if not image_paths:
            return results

        logger.info(f"Processing {len(image_paths)} images with vision model")

        # Create tasks
        tasks = []
        for path in image_paths:
            is_table = path in table_paths
            tasks.append(self.process_image(path, is_table=is_table))

        # Execute concurrently
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results
        for path, desc in zip(image_paths, descriptions):
            if isinstance(desc, Exception):
                results[path] = f"[Processing failed: {str(desc)}]"
            else:
                results[path] = desc

        logger.info(f"Vision processing complete. {len(results)} images processed.")
        return results


# Convenience function
async def get_image_description(
    image_path: str,
    model: str = DEFAULT_VISION_MODEL,
    is_table: bool = False
) -> str:
    """
    Quick function to get description for a single image.
    """
    processor = VisionProcessor(model=model)
    try:
        return await processor.process_image(image_path, is_table=is_table)
    finally:
        await processor.close()
