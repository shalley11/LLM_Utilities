"""
Vision Processor

Calls Vision Gemma3 model to get descriptions for images and tables.
Supports Ollama and VLLM backends.
"""

import re
import base64
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict

import aiohttp

from .config import (
    VISION_LLM_BACKEND,
    VISION_MODEL,
    VISION_OLLAMA_URL,
    VISION_VLLM_URL,
    VISION_REQUEST_TIMEOUT,
    VISION_MAX_CONCURRENT,
    VISION_TEMPERATURE,
    VISION_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class VisionProcessor:
    """
    Processes images using Vision model via Ollama or VLLM.
    Returns text descriptions for images and tables.
    """

    def __init__(
        self,
        model: str = VISION_MODEL,
        backend: str = VISION_LLM_BACKEND,
        ollama_url: str = VISION_OLLAMA_URL,
        vllm_url: str = VISION_VLLM_URL,
        max_concurrent: int = VISION_MAX_CONCURRENT,
        timeout: int = VISION_REQUEST_TIMEOUT
    ):
        self.model = model
        self.backend = backend
        self.ollama_url = ollama_url
        self.vllm_url = vllm_url
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.temperature = VISION_TEMPERATURE
        self.max_tokens = VISION_MAX_TOKENS
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session (no global timeout — each request has its own)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
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

    def _clean_vision_output(self, text: str, max_chars: int = 1000) -> str:
        """
        Clean vision model output: detect repetition loops and truncate.

        Small models sometimes repeat the same tokens endlessly.
        This detects repeating patterns and cuts them off.
        """
        if not text or len(text) <= max_chars:
            return text

        # Detect repeating patterns (4-50 char sequences repeated 3+ times)
        match = re.search(r'(.{4,50}?)\1{2,}', text)
        if match:
            # Cut at where repetition starts, keep one instance
            repeat_start = match.start()
            one_instance = match.group(1).strip()
            cleaned = text[:repeat_start].strip()
            if one_instance and len(one_instance) > 4:
                cleaned += f" {one_instance}"
            logger.warning(f"Vision output had repetition loop, truncated at char {repeat_start}")
            return cleaned[:max_chars]

        # No repetition but still too long — hard truncate at sentence boundary
        truncated = text[:max_chars]
        last_sentence = max(
            truncated.rfind('. '),
            truncated.rfind('.\n'),
            truncated.rfind('? '),
            truncated.rfind('! ')
        )
        if last_sentence > max_chars // 2:
            return truncated[:last_sentence + 1]
        return truncated

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

        # Call vision API based on backend
        async with self._semaphore:
            if self.backend == "vllm":
                raw = await self._call_vllm_vision(image_b64, prompt)
            else:
                raw = await self._call_ollama_vision(image_b64, prompt)
            return self._clean_vision_output(raw)

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
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            req_timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, json=payload, timeout=req_timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama error: {response.status} - {error_text}")
                    return f"[Vision processing failed: {response.status}]"

                result = await response.json()
                return result.get("response", "").strip()

        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return "[Vision processing timed out]"
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            return f"[Vision service unavailable: {str(e)}]"
        except Exception as e:
            logger.error(f"Vision processing error: {e}")
            return f"[Vision processing error: {str(e)}]"

    async def _call_vllm_vision(self, image_b64: str, prompt: str) -> str:
        """Call VLLM API with image using OpenAI-compatible vision format."""
        session = await self._get_session()

        url = f"{self.vllm_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            req_timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, json=payload, timeout=req_timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"VLLM error: {response.status} - {error_text}")
                    return f"[Vision processing failed: {response.status}]"

                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()

        except asyncio.TimeoutError:
            logger.error(f"VLLM request timed out after {self.timeout}s")
            return "[Vision processing timed out]"
        except aiohttp.ClientError as e:
            logger.error(f"VLLM connection error: {e}")
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
    model: str = VISION_MODEL,
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
