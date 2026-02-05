"""
Translation Service

FastAPI endpoints for text translation using LLM.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from .config import (
    TRANSLATION_DEFAULT_MODEL,
    TRANSLATION_AUTO_DETECT_SOURCE,
    TRANSLATION_MAX_BATCH_SIZE,
)
from .schemas import (
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse,
    TranslatedItem,
)
from .translator import Translator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/docAI/v1/translate", tags=["Translation"])


# =====================
# API Endpoints
# =====================

@router.post("/text", response_model=TranslationResponse)
async def translate_text_endpoint(request: TranslationRequest):
    """
    Translate text to a target language.

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `text`: Text to translate (required)
    - `target_language`: Target language for translation (required)
    - `source_language`: Source language (auto-detected if not provided)
    - `model`: Model to use (optional)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `original_text`: Original input text
    - `translated_text`: Translated text
    - `source_language`: Detected or provided source language
    - `target_language`: Target language
    - `model`: Model used
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    logger.info(
        f"[TRANSLATE_TEXT] START | request_id={request_id} | "
        f"chars={len(request.text)} | target={request.target_language} | "
        f"user_id={request.user_id} | user_name={request.user_name}"
    )

    translator = Translator(model=request.model or TRANSLATION_DEFAULT_MODEL)

    try:
        translated_text, detected_language = await translator.translate(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language
        )

        logger.info(
            f"[TRANSLATE_TEXT] END | request_id={request_id} | "
            f"output_chars={len(translated_text)} | user_id={request.user_id}"
        )

        return TranslationResponse(
            request_id=request_id,
            original_text=request.text,
            translated_text=translated_text,
            source_language=detected_language,
            target_language=request.target_language,
            model=translator.model,
            char_count=len(request.text),
            word_count=len(request.text.split()),
            status="completed",
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(
            f"[TRANSLATE_TEXT] ERROR | request_id={request_id} | "
            f"user_id={request.user_id} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    finally:
        await translator.close()


@router.post("/batch", response_model=BatchTranslationResponse)
async def translate_batch_endpoint(request: BatchTranslationRequest):
    """
    Translate multiple texts to a target language.

    **Request Body:**
    - `request_id`: Request ID for tracking (generated if not provided)
    - `texts`: List of texts to translate (required)
    - `target_language`: Target language for translation (required)
    - `source_language`: Source language (auto-detected if not provided)
    - `model`: Model to use (optional)
    - `user_id`: User identifier for logging (optional)
    - `user_name`: User name for logging (optional)

    **Returns:**
    - `request_id`: Unique identifier
    - `translations`: List of translated items
    - `total_items`: Total number of items translated
    """
    # Generate request_id if not provided (fresh request)
    request_id = request.request_id or str(uuid.uuid4())

    logger.info(
        f"[TRANSLATE_BATCH] START | request_id={request_id} | "
        f"items={len(request.texts)} | target={request.target_language} | "
        f"user_id={request.user_id} | user_name={request.user_name}"
    )

    translator = Translator(model=request.model or TRANSLATION_DEFAULT_MODEL)

    try:
        results = await translator.translate_batch(
            texts=request.texts,
            target_language=request.target_language,
            source_language=request.source_language
        )

        translations = [
            TranslatedItem(
                index=i,
                original_text=original,
                translated_text=translated
            )
            for i, (original, translated) in enumerate(results)
        ]

        logger.info(
            f"[TRANSLATE_BATCH] END | request_id={request_id} | "
            f"items={len(translations)} | user_id={request.user_id}"
        )

        return BatchTranslationResponse(
            request_id=request_id,
            translations=translations,
            source_language=request.source_language or "auto-detected",
            target_language=request.target_language,
            model=translator.model,
            total_items=len(translations),
            status="completed",
            user_id=request.user_id,
            user_name=request.user_name
        )

    except Exception as e:
        logger.error(
            f"[TRANSLATE_BATCH] ERROR | request_id={request_id} | "
            f"user_id={request.user_id} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")

    finally:
        await translator.close()


@router.get("/config")
async def get_translation_config():
    """
    Get the default translation configuration.

    **Returns:**
    - Default model and settings
    """
    return {
        "default_model": TRANSLATION_DEFAULT_MODEL,
        "auto_detect_source": TRANSLATION_AUTO_DETECT_SOURCE,
        "max_batch_size": TRANSLATION_MAX_BATCH_SIZE
    }
