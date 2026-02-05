"""
Core translation logic using LLM.
"""
import re
import logging
from typing import Optional, List, Tuple

from .llm_client import generate_text_with_logging
from .config import TRANSLATION_DEFAULT_MODEL, TRANSLATION_TEMPERATURE
from .prompts import (
    TRANSLATION_SYSTEM_PROMPT,
    get_translation_prompt,
    get_batch_translation_prompt,
)

logger = logging.getLogger(__name__)


class Translator:
    """Translator class using LLM for text translation."""

    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize the translator.

        Args:
            model: Model to use for translation
            temperature: LLM temperature (lower = more deterministic)
        """
        self.model = model or TRANSLATION_DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else TRANSLATION_TEMPERATURE

    async def close(self):
        """Close method for compatibility (no-op since we use module-level session)."""
        pass

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language (auto-detected if not provided)

        Returns:
            Tuple of (translated_text, detected_source_language)
        """
        # Combine system prompt with translation prompt
        full_prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\n{get_translation_prompt(text, target_language, source_language)}"

        logger.info(f"[TRANSLATOR] Translating {len(text)} chars to {target_language}")

        response = await generate_text_with_logging(
            prompt=full_prompt,
            model=self.model,
            task="translation",
            temperature=self.temperature
        )

        translated_text = response.strip()

        # Return detected language as source_language or "auto-detected"
        detected_language = source_language or "auto-detected"

        logger.info(f"[TRANSLATOR] Translation complete | output_chars={len(translated_text)}")

        return translated_text, detected_language

    async def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Translate multiple texts to target language.

        Args:
            texts: List of texts to translate
            target_language: Target language
            source_language: Source language (auto-detected if not provided)

        Returns:
            List of tuples (original_text, translated_text)
        """
        # Combine system prompt with batch translation prompt
        full_prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\n{get_batch_translation_prompt(texts, target_language, source_language)}"

        logger.info(f"[TRANSLATOR] Batch translating {len(texts)} items to {target_language}")

        response = await generate_text_with_logging(
            prompt=full_prompt,
            model=self.model,
            task="batch_translation",
            temperature=self.temperature
        )

        # Parse numbered response
        translations = self._parse_batch_response(response, len(texts))

        logger.info(f"[TRANSLATOR] Batch translation complete | items={len(translations)}")

        return list(zip(texts, translations))

    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse numbered batch translation response.

        Args:
            response: LLM response with numbered translations
            expected_count: Expected number of translations

        Returns:
            List of translated texts
        """
        lines = response.strip().split('\n')
        translations = []

        # Try to extract numbered items
        pattern = r'^\d+\.\s*(.+)$'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                translations.append(match.group(1).strip())
            elif translations:
                # Continuation of previous line
                translations[-1] += ' ' + line

        # If parsing failed, try to split by double newlines or return as single items
        if len(translations) != expected_count:
            # Fallback: split by empty lines
            parts = re.split(r'\n\s*\n', response.strip())
            if len(parts) == expected_count:
                translations = [p.strip() for p in parts]
            elif len(translations) < expected_count:
                # Pad with empty strings if needed
                translations.extend([''] * (expected_count - len(translations)))
            else:
                # Truncate if too many
                translations = translations[:expected_count]

        return translations


async def translate_text(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    model: Optional[str] = None
) -> dict:
    """
    Convenience function to translate text.

    Args:
        text: Text to translate
        target_language: Target language
        source_language: Source language (optional)
        model: Model to use (optional)

    Returns:
        Dictionary with translation results
    """
    translator = Translator(model=model)

    try:
        translated_text, detected_language = await translator.translate(
            text=text,
            target_language=target_language,
            source_language=source_language
        )

        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": detected_language,
            "target_language": target_language,
            "model": translator.model,
            "char_count": len(text),
            "word_count": len(text.split()),
            "status": "completed"
        }

    finally:
        await translator.close()
