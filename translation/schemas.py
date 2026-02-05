"""
Pydantic schemas for translation API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class TranslationRequest(BaseModel):
    """Request for text translation."""
    request_id: Optional[str] = Field(None, description="Request ID (generated if not provided)")
    text: str = Field(..., description="Text to translate", min_length=1)
    target_language: str = Field(..., description="Target language for translation")
    source_language: Optional[str] = Field(None, description="Source language (auto-detected if not provided)")
    model: Optional[str] = Field(None, description="Model to use (defaults to config)")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class TranslationResponse(BaseModel):
    """Response from translation."""
    request_id: str = Field(..., description="Unique request identifier")
    original_text: str = Field(..., description="Original input text")
    translated_text: str = Field(..., description="Translated text")
    source_language: Optional[str] = Field(None, description="Detected or provided source language")
    target_language: str = Field(..., description="Target language")
    model: str = Field(..., description="Model used for translation")
    char_count: int = Field(..., description="Character count of original text")
    word_count: int = Field(..., description="Word count of original text")
    status: str = Field(..., description="Translation status")
    user_id: Optional[str] = Field(None, description="User identifier if provided")
    user_name: Optional[str] = Field(None, description="User name if provided")


class BatchTranslationRequest(BaseModel):
    """Request for batch text translation."""
    request_id: Optional[str] = Field(None, description="Request ID (generated if not provided)")
    texts: List[str] = Field(..., description="List of texts to translate", min_length=1)
    target_language: str = Field(..., description="Target language for translation")
    source_language: Optional[str] = Field(None, description="Source language (auto-detected if not provided)")
    model: Optional[str] = Field(None, description="Model to use (defaults to config)")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class TranslatedItem(BaseModel):
    """Single translated item in batch response."""
    index: int = Field(..., description="Index of the item in the batch")
    original_text: str = Field(..., description="Original input text")
    translated_text: str = Field(..., description="Translated text")


class BatchTranslationResponse(BaseModel):
    """Response from batch translation."""
    request_id: str = Field(..., description="Unique request identifier")
    translations: List[TranslatedItem] = Field(..., description="List of translated items")
    source_language: Optional[str] = Field(None, description="Detected or provided source language")
    target_language: str = Field(..., description="Target language")
    model: str = Field(..., description="Model used for translation")
    total_items: int = Field(..., description="Total number of items translated")
    status: str = Field(..., description="Translation status")
    user_id: Optional[str] = Field(None, description="User identifier if provided")
    user_name: Optional[str] = Field(None, description="User name if provided")
