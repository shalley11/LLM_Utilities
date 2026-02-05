"""
LLM Client with comprehensive logging support (Async Version).

Supports:
- Ollama backend
- VLLM backend
- Streaming responses (async generators)
- Automatic request/response logging
- Metrics collection
- Connection pooling via shared aiohttp session
"""
import time
import json
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from config import (
    LLM_BACKEND,
    VLLM_URL,
    OLLAMA_URL,
    DEFAULT_MODEL,
    get_model_context_length
)
from .config import (
    LLM_CONNECTION_TIMEOUT,
    LLM_CONNECTION_POOL_LIMIT,
    LLM_CONNECTION_POOL_LIMIT_PER_HOST,
    LLM_DEFAULT_TEMPERATURE,
    LLM_DEFAULT_MAX_TOKENS,
)
from logs.logging_config import (
    get_llm_logger,
    log_llm_request,
    log_llm_response,
    log_metrics,
    log_context_usage,
    RequestContext
)

logger = get_llm_logger()

# Global session for connection pooling
_session: Optional[aiohttp.ClientSession] = None


async def get_session() -> aiohttp.ClientSession:
    """Get or create the global aiohttp session for connection pooling."""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=LLM_CONNECTION_TIMEOUT)
        connector = aiohttp.TCPConnector(
            limit=LLM_CONNECTION_POOL_LIMIT,
            limit_per_host=LLM_CONNECTION_POOL_LIMIT_PER_HOST
        )
        _session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _session


async def close_session():
    """Close the global session. Call this on application shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def generate_text(prompt: str, model: str | None = None) -> str:
    """
    Generate text from either Ollama or VLLM based on LLM_BACKEND.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (uses DEFAULT_MODEL if not specified)

    Returns:
        Generated text response
    """
    model_name = model or DEFAULT_MODEL
    logger.debug(f"generate_text called | model={model_name} | backend={LLM_BACKEND}")

    if LLM_BACKEND == "vllm":
        return await _call_vllm_internal(prompt, model_name)

    return await _call_ollama_internal(prompt, model_name)


async def generate_text_with_logging(
    prompt: str,
    model: str | None = None,
    task: str = "generate",
    temperature: float = None,
    max_tokens: int = None
) -> str:
    """
    Generate text with explicit logging control.

    Args:
        prompt: The prompt to send
        model: Model name
        task: Task identifier for logging
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response
    """
    model_name = model or DEFAULT_MODEL
    temp = temperature if temperature is not None else LLM_DEFAULT_TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else LLM_DEFAULT_MAX_TOKENS

    # Get context limit for model
    context_limit = get_model_context_length(model_name)

    # Log request
    request_id = log_llm_request(
        model=model_name,
        backend=LLM_BACKEND,
        task=task,
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tok
    )

    # Log context usage and get stats
    context_stats = log_context_usage(
        request_id=request_id,
        model=model_name,
        prompt=prompt,
        context_limit=context_limit
    )

    start_time = time.time()

    try:
        if LLM_BACKEND == "vllm":
            response = await _call_vllm_internal(prompt, model_name, temp, max_tok)
        else:
            response = await _call_ollama_internal(prompt, model_name, temp)

        latency_ms = (time.time() - start_time) * 1000

        # Log successful response
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            response=response,
            latency_ms=latency_ms,
            status="success"
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=len(response),
            status="success",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        return response

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        # Log error
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            response="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=0,
            status="error",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        raise


async def _call_ollama_internal(
    prompt: str,
    model: str,
    temperature: float = None
) -> str:
    """
    Call Ollama API with full parameters.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature

    Returns:
        Generated text
    """
    temp = temperature if temperature is not None else LLM_DEFAULT_TEMPERATURE

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp
        }
    }

    logger.debug(f"Calling Ollama | url={OLLAMA_URL}/api/generate | model={model}")

    try:
        session = await get_session()
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload
        ) as r:
            r.raise_for_status()
            response_data = await r.json()
            response_text = response_data.get("response", "").strip()

            # Log token info if available
            if "eval_count" in response_data:
                logger.debug(
                    f"Ollama tokens | eval_count={response_data.get('eval_count')} | "
                    f"eval_duration={response_data.get('eval_duration')}"
                )

            return response_text

    except asyncio.TimeoutError:
        logger.error(f"Ollama timeout | model={model}")
        raise RuntimeError("LLM request timed out. Please try again.")

    except aiohttp.ClientError as e:
        logger.error(f"Ollama request failed | model={model} | error={e}")
        raise RuntimeError("LLM service unavailable. Please try again later.")


async def _call_vllm_internal(
    prompt: str,
    model: str,
    temperature: float = None,
    max_tokens: int = None
) -> str:
    """
    Call VLLM API with full parameters.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    temp = temperature if temperature is not None else LLM_DEFAULT_TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else LLM_DEFAULT_MAX_TOKENS

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tok
    }

    logger.debug(f"Calling VLLM | url={VLLM_URL}/v1/chat/completions | model={model}")

    try:
        session = await get_session()
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload
        ) as r:
            r.raise_for_status()
            response_data = await r.json()
            response_text = response_data["choices"][0]["message"]["content"].strip()

            # Log usage info if available
            usage = response_data.get("usage", {})
            if usage:
                logger.debug(
                    f"VLLM tokens | prompt={usage.get('prompt_tokens')} | "
                    f"completion={usage.get('completion_tokens')} | "
                    f"total={usage.get('total_tokens')}"
                )

            return response_text

    except asyncio.TimeoutError:
        logger.error(f"VLLM timeout | model={model}")
        raise RuntimeError("LLM request timed out. Please try again.")

    except aiohttp.ClientError as e:
        logger.error(f"VLLM request failed | model={model} | error={e}")
        raise RuntimeError("LLM service unavailable. Please try again later.")


# =========================
# Streaming Support (Async)
# =========================

async def stream_ollama(
    prompt: str,
    model: str | None = None,
    temperature: float = None,
    task: str = "stream"
) -> AsyncGenerator[str, None]:
    """
    Stream response from Ollama token by token.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        task: Task identifier for logging

    Yields:
        Individual tokens as they're generated
    """
    model_name = model or DEFAULT_MODEL
    temp = temperature if temperature is not None else LLM_DEFAULT_TEMPERATURE

    # Get context limit for model
    context_limit = get_model_context_length(model_name)

    # Log request
    request_id = log_llm_request(
        model=model_name,
        backend="ollama",
        task=task,
        prompt=prompt,
        temperature=temp
    )

    # Log context usage
    context_stats = log_context_usage(
        request_id=request_id,
        model=model_name,
        prompt=prompt,
        context_limit=context_limit
    )

    start_time = time.time()
    full_response = []

    try:
        session = await get_session()
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temp}
            }
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        token = data.get("response", "")
                        if token:
                            full_response.append(token)
                            yield token

                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        latency_ms = (time.time() - start_time) * 1000
        complete_response = "".join(full_response)

        # Log response
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            response=complete_response,
            latency_ms=latency_ms,
            status="success"
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=len(complete_response),
            status="success",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            response="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=0,
            status="error",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        logger.error(f"Stream error | {e}")


async def stream_vllm(
    prompt: str,
    model: str | None = None,
    temperature: float = None,
    max_tokens: int = None,
    task: str = "stream"
) -> AsyncGenerator[str, None]:
    """
    Stream response from VLLM token by token.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens
        task: Task identifier for logging

    Yields:
        Individual tokens as they're generated
    """
    model_name = model or DEFAULT_MODEL
    temp = temperature if temperature is not None else LLM_DEFAULT_TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else LLM_DEFAULT_MAX_TOKENS

    # Get context limit for model
    context_limit = get_model_context_length(model_name)

    # Log request
    request_id = log_llm_request(
        model=model_name,
        backend="vllm",
        task=task,
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tok
    )

    # Log context usage
    context_stats = log_context_usage(
        request_id=request_id,
        model=model_name,
        prompt=prompt,
        context_limit=context_limit
    )

    start_time = time.time()
    full_response = []

    try:
        session = await get_session()
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": max_tok,
                "stream": True
            }
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                full_response.append(token)
                                yield token
                        except json.JSONDecodeError:
                            continue

        latency_ms = (time.time() - start_time) * 1000
        complete_response = "".join(full_response)

        # Log response
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            response=complete_response,
            latency_ms=latency_ms,
            status="success"
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=len(complete_response),
            status="success",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            response="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=0,
            status="error",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        logger.error(f"Stream error | {e}")


async def stream_text(
    prompt: str,
    model: str | None = None,
    task: str = "stream"
) -> AsyncGenerator[str, None]:
    """
    Stream text from the configured backend.

    Args:
        prompt: The prompt text
        model: Model name
        task: Task identifier for logging

    Yields:
        Individual tokens
    """
    if LLM_BACKEND == "vllm":
        async for token in stream_vllm(prompt, model, task=task):
            yield token
    else:
        async for token in stream_ollama(prompt, model, task=task):
            yield token
