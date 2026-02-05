"""
Pydantic schemas for hierarchical summarization API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

from .config import (
    SUMMARIZATION_DEFAULT_TYPE,
    SUMMARIZATION_MAX_WORDS_PER_BATCH,
    SUMMARIZATION_MAX_CHUNKS_PER_BATCH,
    SUMMARIZATION_INTERMEDIATE_WORDS,
    SUMMARIZATION_FINAL_WORDS,
    SUMMARIZATION_TEMPERATURE,
)

# Summary types
SummaryType = Literal["brief", "bullets", "detailed", "executive"]


class ChunkInput(BaseModel):
    """Input chunk for summarization."""
    text: str = Field(..., description="Chunk text content")
    page_no: Optional[int] = Field(None, description="Page number")
    chunk_index: Optional[int] = Field(None, description="Chunk index")


class SummarizationRequest(BaseModel):
    """Request for hierarchical summarization."""
    request_id: Optional[str] = Field(None, description="Request ID (generated if not provided)")
    chunks: List[ChunkInput] = Field(..., description="List of text chunks to summarize")
    summary_type: SummaryType = Field(SUMMARIZATION_DEFAULT_TYPE, description="Type of summary to generate")
    model: Optional[str] = Field(None, description="Model to use (defaults to config)")
    max_words_per_batch: Optional[int] = Field(None, description=f"Max words per batch (defaults to {SUMMARIZATION_MAX_WORDS_PER_BATCH})")
    max_chunks_per_batch: Optional[int] = Field(None, description=f"Max chunks per batch (defaults to {SUMMARIZATION_MAX_CHUNKS_PER_BATCH})")
    intermediate_summary_words: Optional[int] = Field(None, description=f"Target words for batch summaries (defaults to {SUMMARIZATION_INTERMEDIATE_WORDS})")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class SummarizationResponse(BaseModel):
    """Response from hierarchical summarization."""
    request_id: str = Field(..., description="Unique request identifier")
    summary: str = Field(..., description="Generated summary")
    summary_type: str = Field(..., description="Type of summary generated")
    method: str = Field(..., description="Method used: direct or hierarchical")
    total_chunks: int = Field(..., description="Total chunks processed")
    total_words: int = Field(..., description="Total words in input")
    batches: int = Field(..., description="Number of batches used")
    levels: int = Field(..., description="Reduction levels used")
    model: str = Field(..., description="Model used")
    user_id: Optional[str] = Field(None, description="User identifier if provided")
    user_name: Optional[str] = Field(None, description="User name if provided")


class TextSummarizationRequest(BaseModel):
    """Simple text summarization request."""
    request_id: Optional[str] = Field(None, description="Request ID (generated if not provided)")
    text: str = Field(..., description="Text to summarize", min_length=10)
    summary_type: SummaryType = Field(SUMMARIZATION_DEFAULT_TYPE, description="Type of summary")
    chunk_size: Optional[int] = Field(None, description="Chunk size in characters (auto-calculated if not provided)")
    model: Optional[str] = Field(None, description="Model to use")
    user_id: Optional[str] = Field(None, description="User identifier for logging")
    user_name: Optional[str] = Field(None, description="User name for logging")


class SummarizerConfig(BaseModel):
    """Configuration for the summarizer."""
    max_words_per_batch: int = Field(SUMMARIZATION_MAX_WORDS_PER_BATCH, description="Maximum words per batch")
    max_chunks_per_batch: int = Field(SUMMARIZATION_MAX_CHUNKS_PER_BATCH, description="Maximum chunks per batch")
    intermediate_summary_words: int = Field(SUMMARIZATION_INTERMEDIATE_WORDS, description="Target words for intermediate summaries")
    final_summary_words: int = Field(SUMMARIZATION_FINAL_WORDS, description="Target words for final summary")
    temperature: float = Field(SUMMARIZATION_TEMPERATURE, description="LLM temperature")
    model: Optional[str] = Field(None, description="Model to use")


class SummarizationStatus(BaseModel):
    """Status of a summarization request."""
    request_id: str
    status: str
    summary_type: str
    method: Optional[str] = None
    total_chunks: Optional[int] = None
    batches_completed: Optional[int] = None
    total_batches: Optional[int] = None
    summary: Optional[str] = None
    error: Optional[str] = None
