"""
Summarization Module

Provides hierarchical summarization for large documents using map-reduce approach:
- Splits content into manageable batches
- Summarizes each batch in parallel
- Combines batch summaries into final summary

Supports multiple summary types:
- brief: Short focused summary
- bullets: Bullet-point format
- detailed: Comprehensive narrative
- executive: High-level for leadership
"""

from .service import router
from .summarizer import (
    summarize_chunks_async,
    summarize_chunks_sync,
    split_text_into_chunks,
    SummarizerConfig
)
from .schemas import (
    SummarizationRequest,
    SummarizationResponse,
    TextSummarizationRequest,
    ChunkInput,
    SummaryType
)

__all__ = [
    # Router
    "router",
    # Summarizer functions
    "summarize_chunks_async",
    "summarize_chunks_sync",
    "split_text_into_chunks",
    "SummarizerConfig",
    # Schemas
    "SummarizationRequest",
    "SummarizationResponse",
    "TextSummarizationRequest",
    "ChunkInput",
    "SummaryType"
]
