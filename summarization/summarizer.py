"""
Hierarchical summarization for large documents.

Handles documents with many chunks using a map-reduce approach:
1. Group chunks into batches that fit within LLM context
2. Summarize each batch (MAP phase)
3. Combine batch summaries into final summary (REDUCE phase)

Supports async processing for parallel batch summarization.
"""
import time
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import get_model_context_length
from logs.logging_config import get_llm_logger
from .config import (
    SUMMARIZATION_LLM_BACKEND,
    SUMMARIZATION_OLLAMA_URL,
    SUMMARIZATION_VLLM_URL,
    SUMMARIZATION_DEFAULT_MODEL,
    SUMMARIZATION_MAX_WORDS_PER_BATCH,
    SUMMARIZATION_MAX_CHUNKS_PER_BATCH,
    SUMMARIZATION_INTERMEDIATE_WORDS,
    SUMMARIZATION_FINAL_WORDS,
    SUMMARIZATION_TEMPERATURE,
    SUMMARIZATION_MAX_TOKENS,
    SUMMARIZATION_CONNECTION_TIMEOUT,
    SUMMARIZATION_MAP_CONCURRENT,
    SUMMARIZATION_REDUCE_CONCURRENT,
)
from .prompts import (
    get_batch_summary_prompt,
    get_final_combine_prompt,
    get_direct_summary_prompt
)

logger = get_llm_logger()

# Default configuration (from module config)
DEFAULT_MAX_WORDS_PER_BATCH = SUMMARIZATION_MAX_WORDS_PER_BATCH
DEFAULT_MAX_CHUNKS_PER_BATCH = SUMMARIZATION_MAX_CHUNKS_PER_BATCH
DEFAULT_INTERMEDIATE_WORDS = SUMMARIZATION_INTERMEDIATE_WORDS
DEFAULT_FINAL_WORDS = SUMMARIZATION_FINAL_WORDS
DEFAULT_TEMPERATURE = SUMMARIZATION_TEMPERATURE
DEFAULT_TIMEOUT = SUMMARIZATION_CONNECTION_TIMEOUT
MAX_REDUCE_LEVELS = 5
REDUCE_GROUP_SIZE = 5


@dataclass
class SummarizerConfig:
    """Configuration for hierarchical summarization."""
    max_words_per_batch: int = DEFAULT_MAX_WORDS_PER_BATCH
    max_chunks_per_batch: int = DEFAULT_MAX_CHUNKS_PER_BATCH
    intermediate_summary_words: int = DEFAULT_INTERMEDIATE_WORDS
    final_summary_words: int = DEFAULT_FINAL_WORDS
    temperature: float = DEFAULT_TEMPERATURE
    model: str = field(default_factory=lambda: SUMMARIZATION_DEFAULT_MODEL)


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _chunk_into_batches(
    chunks: List[Dict],
    config: SummarizerConfig
) -> List[List[Dict]]:
    """
    Split chunks into batches that fit within context limits.
    Groups chunks by page proximity when possible.
    """
    logger.debug(f"[BATCHING] START | total_chunks={len(chunks)} | max_words={config.max_words_per_batch}")

    batches = []
    current_batch = []
    current_word_count = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_words = _count_words(chunk_text)

        would_exceed_words = (current_word_count + chunk_words) > config.max_words_per_batch
        would_exceed_chunks = len(current_batch) >= config.max_chunks_per_batch

        if current_batch and (would_exceed_words or would_exceed_chunks):
            batches.append(current_batch)
            current_batch = []
            current_word_count = 0

        current_batch.append(chunk)
        current_word_count += chunk_words

    if current_batch:
        batches.append(current_batch)

    logger.debug(f"[BATCHING] END | total_batches={len(batches)}")
    return batches


def _prepare_batch_content(chunks: List[Dict]) -> str:
    """Prepare content from a batch of chunks."""
    content_parts = []
    current_page = None

    for chunk in chunks:
        page_no = chunk.get("page_no", 0)

        if page_no and page_no != current_page:
            content_parts.append(f"\n[Page {page_no}]\n")
            current_page = page_no

        content_parts.append(chunk.get("text", "") + "\n")

    return "".join(content_parts)


async def _call_llm_async(
    prompt: str,
    config: SummarizerConfig,
    session: aiohttp.ClientSession,
    context: str = "unknown"
) -> str:
    """Call LLM asynchronously using module-specific backend."""
    prompt_words = _count_words(prompt)
    backend = SUMMARIZATION_LLM_BACKEND
    logger.debug(f"[SUMMARIZATION_LLM] {context} | model={config.model} | backend={backend} | prompt_words={prompt_words}")

    start_time = time.time()

    try:
        if backend == "vllm":
            # VLLM API call
            url = f"{SUMMARIZATION_VLLM_URL}/v1/chat/completions"
            payload = {
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": SUMMARIZATION_MAX_TOKENS
            }
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)) as response:
                response.raise_for_status()
                data = await response.json()
                result = data["choices"][0]["message"]["content"].strip()
        else:
            # Ollama API call
            url = f"{SUMMARIZATION_OLLAMA_URL}/api/generate"
            payload = {
                "model": config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": SUMMARIZATION_MAX_TOKENS
                }
            }
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)) as response:
                response.raise_for_status()
                data = await response.json()
                result = data.get("response", "").strip()

        elapsed = time.time() - start_time
        result_words = _count_words(result)

        logger.info(f"[SUMMARIZATION_LLM] {context} | SUCCESS | elapsed={elapsed:.2f}s | response_words={result_words}")
        return result

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[SUMMARIZATION_LLM] {context} | TIMEOUT after {elapsed:.2f}s")
        raise RuntimeError(f"Summarization LLM timeout after {elapsed:.2f}s")

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.error(f"[SUMMARIZATION_LLM] {context} | REQUEST_ERROR after {elapsed:.2f}s | error={str(e)}")
        raise RuntimeError(f"Summarization LLM request failed: {e}")


async def _summarize_batch_async(
    content: str,
    batch_index: int,
    total_batches: int,
    config: SummarizerConfig,
    session: aiohttp.ClientSession
) -> str:
    """Generate summary for a single batch asynchronously."""
    prompt = get_batch_summary_prompt(
        content=content,
        batch_index=batch_index,
        total_batches=total_batches,
        word_count=config.intermediate_summary_words
    )
    return await _call_llm_async(
        prompt, config, session,
        context=f"batch_{batch_index + 1}_of_{total_batches}"
    )


async def _process_batches_parallel(
    batches: List[List[Dict]],
    config: SummarizerConfig,
    session: aiohttp.ClientSession,
    max_concurrent: int = SUMMARIZATION_MAP_CONCURRENT
) -> List[str]:
    """
    Process multiple batches in parallel using asyncio.gather().
    """
    num_batches = len(batches)
    logger.info(f"[ASYNC_MAP] START | batches={num_batches} | max_concurrent={max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)
    batch_summaries = [None] * num_batches

    async def process_single_batch(idx: int, batch: List[Dict]):
        """Process a single batch with semaphore control."""
        async with semaphore:
            content = _prepare_batch_content(batch)
            content_words = _count_words(content)
            logger.info(f"[ASYNC_MAP] Batch {idx+1}/{num_batches} | chunks={len(batch)} | words={content_words}")

            summary = await _summarize_batch_async(
                content, idx, num_batches, config, session
            )
            return idx, summary

    tasks = [
        process_single_batch(idx, batch)
        for idx, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(result)
            logger.error(f"[ASYNC_MAP] Batch failed: {result}")
        else:
            idx, summary = result
            batch_summaries[idx] = summary

    if errors:
        raise RuntimeError(f"{len(errors)} batches failed during parallel processing")

    logger.info(f"[ASYNC_MAP] END | all {num_batches} batches completed")
    return batch_summaries


async def _reduce_summaries_async(
    summaries: List[str],
    config: SummarizerConfig,
    summary_type: str,
    session: aiohttp.ClientSession,
    max_concurrent: int = SUMMARIZATION_REDUCE_CONCURRENT
) -> tuple:
    """Reduce summaries hierarchically."""
    levels = 1
    current_summaries = summaries

    combined_text = " ".join(current_summaries)
    combined_words = _count_words(combined_text)
    logger.debug(f"[ASYNC_REDUCE] Initial combined words: {combined_words}")

    while combined_words > config.max_words_per_batch and levels < MAX_REDUCE_LEVELS:
        logger.info(f"[ASYNC_REDUCE] Level {levels} | summaries={len(current_summaries)} | combined_words={combined_words}")

        # Group summaries into batches
        groups = [current_summaries[i:i+REDUCE_GROUP_SIZE] for i in range(0, len(current_summaries), REDUCE_GROUP_SIZE)]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def reduce_group(idx: int, group: List[str]):
            async with semaphore:
                prompt = get_final_combine_prompt(group, "detailed")
                return await _call_llm_async(
                    prompt, config, session,
                    context=f"reduce_level_{levels}_group_{idx+1}"
                )

        tasks = [reduce_group(idx, group) for idx, group in enumerate(groups)]
        new_summaries = await asyncio.gather(*tasks)

        current_summaries = list(new_summaries)
        combined_text = " ".join(current_summaries)
        combined_words = _count_words(combined_text)
        levels += 1

    # Final combination
    logger.info(f"[ASYNC_REDUCE] Final combine | summaries={len(current_summaries)} | target_type={summary_type}")

    prompt = get_final_combine_prompt(current_summaries, summary_type)
    final_summary = await _call_llm_async(
        prompt, config, session,
        context=f"final_{summary_type}"
    )

    return final_summary, levels


async def summarize_chunks_async(
    chunks: List[Dict],
    summary_type: str = "detailed",
    config: Optional[SummarizerConfig] = None,
) -> Dict:
    """
    Summarize chunks using parallel async processing.

    Uses a single shared aiohttp session across MAP and REDUCE phases
    for connection reuse. Concurrency is controlled via env vars:
    SUMMARIZATION_MAP_CONCURRENT and SUMMARIZATION_REDUCE_CONCURRENT.

    Args:
        chunks: List of document chunks with 'text' field
        summary_type: brief, bullets, detailed, or executive
        config: Summarization configuration

    Returns:
        Dictionary with summary and metadata
    """
    start_time = time.time()
    logger.info(f"[SUMMARIZE] START | chunks={len(chunks)} | type={summary_type}")

    if config is None:
        config = SummarizerConfig()

    total_chunks = len(chunks)
    total_words = sum(_count_words(c.get("text", "")) for c in chunks)

    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Check if content fits in single batch (direct summarization)
        is_small_doc = total_words <= config.max_words_per_batch and total_chunks <= config.max_chunks_per_batch

        if is_small_doc:
            logger.info(f"[SUMMARIZE] Using DIRECT method (single LLM call)")

            content = _prepare_batch_content(chunks)
            prompt = get_direct_summary_prompt(content, summary_type)
            summary = await _call_llm_async(prompt, config, session, context=f"direct_{summary_type}")

            elapsed = time.time() - start_time
            logger.info(f"[SUMMARIZE] END | method=direct | elapsed={elapsed:.2f}s")

            return {
                "summary": summary,
                "method": "direct",
                "total_chunks": total_chunks,
                "total_words": total_words,
                "batches": 1,
                "levels": 1,
                "model": config.model
            }

        # Hierarchical processing with parallel batches
        logger.info(f"[SUMMARIZE] Using HIERARCHICAL method (map-reduce)")
        batches = _chunk_into_batches(chunks, config)
        num_batches = len(batches)

        # MAP PHASE: Summarize each batch in parallel
        map_start = time.time()
        batch_summaries = await _process_batches_parallel(batches, config, session)
        map_elapsed = time.time() - map_start
        logger.info(f"[SUMMARIZE] MAP_PHASE | elapsed={map_elapsed:.2f}s")

        # REDUCE PHASE: Combine summaries (reuses same session)
        reduce_start = time.time()
        final_summary, levels = await _reduce_summaries_async(
            batch_summaries, config, summary_type, session
        )
        reduce_elapsed = time.time() - reduce_start
        logger.info(f"[SUMMARIZE] REDUCE_PHASE | elapsed={reduce_elapsed:.2f}s | levels={levels}")

    total_elapsed = time.time() - start_time
    logger.info(f"[SUMMARIZE] END | batches={num_batches} | elapsed={total_elapsed:.2f}s")

    return {
        "summary": final_summary,
        "method": "hierarchical",
        "total_chunks": total_chunks,
        "total_words": total_words,
        "batches": num_batches,
        "levels": levels + 1,
        "model": config.model
    }


def summarize_chunks_sync(
    chunks: List[Dict],
    summary_type: str = "detailed",
    config: Optional[SummarizerConfig] = None,
) -> Dict:
    """
    Synchronous wrapper for parallel summarization.

    Use this when calling from synchronous code.
    """
    return asyncio.run(
        summarize_chunks_async(
            chunks=chunks,
            summary_type=summary_type,
            config=config,
        )
    )


def split_text_into_chunks(text: str, chunk_size: int = 2000) -> List[Dict]:
    """
    Split a large text into chunks for summarization.

    Args:
        text: The text to split
        chunk_size: Approximate characters per chunk

    Returns:
        List of chunk dictionaries with 'text' field
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chunk_size and current_chunk:
            chunks.append({"text": " ".join(current_chunk)})
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append({"text": " ".join(current_chunk)})

    return chunks
