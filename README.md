# LLM_Utilities

A production-ready async Python module for LLM interactions with comprehensive logging, context tracking, and iterative refinement support.

## Features

- **Async I/O**: Non-blocking LLM calls using `aiohttp` for high concurrency
- **Connection Pooling**: Shared HTTP session for optimal performance
- **Multi-Backend Support**: Ollama and VLLM backends
- **Context Length Tracking**: Model-wise context limits with warnings and rejection
- **Refinement Cycle**: Iteratively improve outputs with user feedback
- **Regeneration**: Start fresh from original text with task-specific prompts
- **Comprehensive Logging**: Structured logs with request tracing and metrics
- **Redis Session Storage**: Persistent refinement sessions with TTL

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `aiohttp` - Async HTTP client for LLM API calls
- `pydantic` - Data validation
- `fastapi` - Async web framework
- `uvicorn` - ASGI server
- `redis` - Session storage

### Prerequisites
- Redis server running on `localhost:6379`
- Ollama running on `localhost:11434` (or VLLM on `localhost:8000`)

---

## Quick Start

### Run the API Server

```bash
cd LLM_Utilities
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Configuration

### `config.py`

```python
# Backend selection
LLM_BACKEND = "ollama"  # or "vllm"

# Service URLs
OLLAMA_URL = "http://localhost:11434"
VLLM_URL = "http://localhost:8000"

# Default model
DEFAULT_MODEL = "gemma3:4b"

# Model context lengths
MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,   # POC
    "gemma3:12b": 8192,  # PROD
}

# Context thresholds (percentage)
CONTEXT_WARNING_THRESHOLD = 80   # Log WARNING
CONTEXT_ERROR_THRESHOLD = 95     # Log ERROR
CONTEXT_REJECT_THRESHOLD = 100   # Reject request
```

---

## API Endpoints

### POST `/process-text`
Process text with various tasks. Returns a `request_id` for refinement/regeneration.

**Request:**
```json
{
  "text": "Your long document here...",
  "task": "summary",
  "summary_type": "brief",
  "target_language": "Spanish",
  "model": "gemma3:4b",
  "user_id": "user123"
}
```

**Tasks:** `summary`, `translate`, `rephrase`, `remove_repetitions`

**Summary Types:** `brief`, `detailed`, `bulletwise`

**Target Language:** Any language string (for `translate` task)

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "task": "summary",
  "model": "gemma3:4b",
  "output": "Generated summary...",
  "user_id": "user123"
}
```

---

### POST `/refine/{request_id}`
Refine the **current result** based on user feedback. Uses smaller context.

**Request:**
```json
{
  "user_feedback": "Make it shorter and add bullet points",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "refined_output": "Refined result...",
  "refinement_count": 1,
  "task": "summary",
  "model": "gemma3:4b",
  "user_id": "user123"
}
```

---

### POST `/regenerate/{request_id}`
Regenerate from **original text** using task-specific prompts with user feedback. Uses the full task prompt (summary, rephrase, etc.) combined with user instructions.

**Request:**
```json
{
  "user_feedback": "Focus only on financial aspects",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "regenerated_output": "New result from original...",
  "regeneration_count": 1,
  "task": "summary",
  "model": "gemma3:4b",
  "user_id": "user123"
}
```

**Note:** Regenerate uses the original task-specific prompt (e.g., full summary prompt with rules) plus the user's feedback as "ADDITIONAL USER INSTRUCTIONS".

---

### GET `/refine/{request_id}`
Get current session status.

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "task": "summary",
  "model": "gemma3:4b",
  "current_output": "Latest output...",
  "refinement_count": 2,
  "regeneration_count": 1,
  "user_id": "user123",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:35:00",
  "ttl_seconds": 3200
}
```

---

### DELETE `/refine/{request_id}`
End session and cleanup Redis data.

---

### POST `/refine/{request_id}/extend`
Extend session TTL.

**Query Parameter:** `ttl_seconds` (default: 3600)

---

### GET `/health`
Health check endpoint.

---

### GET `/logs/stats`
Get log file statistics.

---

## Refine vs Regenerate

| Aspect | `/refine` | `/regenerate` |
|--------|-----------|---------------|
| **Input** | `current_result` | `original_text` |
| **Prompt** | Generic refinement prompt | Task-specific prompt (summary/rephrase/etc.) |
| **Context Size** | Smaller | Larger |
| **Use Case** | Polish, tweak, minor fixes | Start fresh, change direction |
| **Counter** | `refinement_count` | `regeneration_count` |

### When to Use Refine
- Output is mostly good, needs small improvements
- "Make it shorter", "Fix grammar", "Add conclusion"

### When to Use Regenerate
- Output went in wrong direction
- Need completely different focus
- "Focus on X instead of Y", "Ignore section Z"
- Want to apply different summarization style

---

## Regenerate Prompt Structure

Regenerate uses the **full task-specific prompt** with user feedback incorporated:

### Example: Summary Task
```
You are a summarization system.

RULES:
- Use ONLY the information present in the text
- Do NOT assume, infer, or add anything not stated
- Preserve factual accuracy

TASK:
Generate a brief summary that captures the core idea and key outcome.
Exclude minor or repetitive details.

ADDITIONAL USER INSTRUCTIONS:
{user_feedback}

TEXT:
{original_text}

OUTPUT:
Summary:
```

### Example: Translate Task
```
You are a translation system.

RULES:
- Use ONLY the information present in the text
- Do NOT add, omit, or explain content
- Preserve meaning, tone, and factual accuracy

TASK:
Translate the text into {target_language}.

ADDITIONAL USER INSTRUCTIONS:
{user_feedback}

TEXT:
{original_text}

OUTPUT:
Translated Text:
```

---

## Context Length Protection

The API validates context length before processing:

| Usage | Action |
|-------|--------|
| < 80% | Normal processing |
| 80-95% | WARNING logged |
| 95-100% | ERROR logged |
| >= 100% | Request REJECTED (HTTP 400) |

**Error Response:**
```json
{
  "detail": {
    "error": "context_length_exceeded",
    "message": "Input text is too long and would be truncated.",
    "model": "gemma3:4b",
    "context_limit": 8192,
    "estimated_tokens": 9500,
    "usage_percent": 115.97
  }
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (Async)                        │
│                         (main.py)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ /process-   │    │  /refine    │    │    /regenerate      │  │
│  │    text     │    │             │    │                     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘  │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Context Validation                         │ │
│  │              (config.py - estimate_tokens)                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Prompt Builder                            │ │
│  │    (prompts.py - task-specific + user feedback)             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Async LLM Client (aiohttp)                      │ │
│  │    (llm_client.py - connection pooling, non-blocking)       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  Refinement Store                            │ │
│  │   (refinement_store.py - Redis with summary_type tracking)  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flow Diagrams

### Initial Processing Flow
```
User Request
     │
     ▼
POST /process-text
     │
     ├─► Build Prompt (prompts.py)
     │
     ├─► Check Context Length
     │        │
     │        ├─► >= 100%? ──► HTTP 400 Error
     │        │
     │        └─► < 100%? ──► Continue
     │
     ├─► Call LLM (async, non-blocking)
     │        │
     │        └─► Log context usage, metrics
     │
     ├─► Store in Redis (with summary_type, target_language)
     │        │
     │        └─► TTL: 2 hours
     │
     └─► Return response with request_id
```

### Refinement Flow
```
POST /refine/{request_id}
     │
     ├─► Fetch from Redis (current_result)
     │
     ├─► Build Refinement Prompt
     │        │
     │        └─► current_result + user_feedback
     │
     ├─► Check Context Length
     │
     ├─► Call LLM (async)
     │
     ├─► Update Redis (overwrite current_result)
     │        │
     │        └─► Increment refinement_count
     │
     └─► Return refined output
```

### Regeneration Flow
```
POST /regenerate/{request_id}
     │
     ├─► Fetch from Redis (original_text, summary_type, target_language)
     │
     ├─► Build Task-Specific Regeneration Prompt
     │        │
     │        └─► Full task prompt + user_feedback + original_text
     │
     ├─► Check Context Length
     │
     ├─► Call LLM (async)
     │
     ├─► Update Redis (overwrite current_result)
     │        │
     │        └─► Increment regeneration_count
     │
     └─► Return regenerated output
```

---

## Logging

### Log Files (in `logs/` directory)

| File | Content |
|------|---------|
| `llm_requests.log` | All LLM API calls |
| `llm_errors.log` | Error-level logs |
| `llm_metrics.log` | Performance metrics (JSON) |
| `llm_debug.log` | Detailed debug info |

### Log Format
```
2024-01-15 10:30:00 | INFO     | abc-123-uuid | user123 | LLM_REQUEST | model=gemma3:4b | task=summary
2024-01-15 10:30:05 | WARNING  | abc-123-uuid | user123 | CONTEXT_USAGE_HIGH | model=gemma3:4b | usage=85.2%
2024-01-15 10:30:10 | INFO     | abc-123-uuid | user123 | LLM_RESPONSE | status=success | latency=5000ms
```

### Context Usage Logs
```
# Normal
DEBUG | CONTEXT_USAGE | model=gemma3:4b | usage=25.0% | tokens=500/8192

# Warning (>80%)
WARNING | CONTEXT_USAGE_HIGH | model=gemma3:4b | usage=85.2% | tokens=6980/8192

# Error (>95%)
ERROR | CONTEXT_NEAR_LIMIT | model=gemma3:4b | usage=96.1% | tokens=7872/8192

# Critical (>=100%)
CRITICAL | CONTEXT_EXCEEDED | model=gemma3:4b | usage=105.3% | tokens=8626/8192
```

---

## Python SDK Usage (Async)

```python
import asyncio
from LLM_Utilities import (
    generate_text,
    generate_text_with_logging,
    stream_text,
    close_session,
    setup_llm_logging,
    get_model_context_length,
    estimate_tokens
)

# Initialize logging
setup_llm_logging()

async def main():
    # Simple generation
    result = await generate_text("Summarize this text...")

    # Generation with logging
    result = await generate_text_with_logging(
        prompt="Summarize this text...",
        model="gemma3:4b",
        task="summary"
    )

    # Streaming (async generator)
    async for token in stream_text("Summarize this text..."):
        print(token, end='', flush=True)

    # Check context before calling
    prompt = "Your long prompt..."
    model = "gemma3:4b"
    context_limit = get_model_context_length(model)
    tokens = estimate_tokens(prompt)
    usage = (tokens / context_limit) * 100

    if usage < 100:
        result = await generate_text(prompt)
    else:
        print(f"Prompt too long: {usage:.1f}% of context")

    # Cleanup on shutdown
    await close_session()

# Run
asyncio.run(main())
```

---

## File Structure

```
LLM_Utilities/
├── __init__.py           # Package exports (async functions)
├── config.py             # Configuration & context limits
├── llm_client.py         # Async LLM backend integration (aiohttp)
├── logging_config.py     # Structured logging system
├── prompts.py            # Task-specific prompt templates
├── refinement_store.py   # Redis session storage (with summary_type)
├── schemas.py            # Pydantic models
├── main.py               # FastAPI application (async endpoints)
├── requirements.txt      # Dependencies
├── README.md             # This documentation
└── logs/                 # Log files (auto-created)
    ├── llm_requests.log
    ├── llm_errors.log
    ├── llm_metrics.log
    └── llm_debug.log
```

---

## Adding New Models

Update `config.py`:

```python
MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,
    "gemma3:12b": 8192,
    "llama3:8b": 8192,
    "mistral:7b": 32768,  # Add new model
}
```

---

## Error Handling

| HTTP Code | Reason |
|-----------|--------|
| 400 | Context length exceeded |
| 404 | Request ID not found or expired |
| 500 | Internal server error |

---

## Session Management

- **TTL**: 2 hours default
- **Extend**: `POST /refine/{id}/extend?ttl_seconds=3600`
- **Cleanup**: `DELETE /refine/{id}` or auto-expire

### Stored Session Data
```python
{
    "request_id": "uuid",
    "task": "summary",
    "current_result": "Latest output...",
    "original_text": "Original input...",
    "model": "gemma3:4b",
    "user_id": "user123",
    "summary_type": "brief",        # For summary task
    "target_language": "Spanish",   # For translate task
    "refinement_count": 2,
    "regeneration_count": 1,
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T10:35:00"
}
```

---

## Performance Benefits (Async)

| Aspect | Sync (Before) | Async (Now) |
|--------|---------------|-------------|
| HTTP Client | `requests` (blocking) | `aiohttp` (non-blocking) |
| Concurrent Requests | Thread-limited | Event loop (thousands) |
| Connection Reuse | None | Connection pooling |
| Memory per Request | Higher (threads) | Lower (coroutines) |

### Connection Pool Settings
```python
# In llm_client.py
connector = aiohttp.TCPConnector(
    limit=100,           # Total connections
    limit_per_host=20    # Per-host limit
)
timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
```

---

## Version History

| Version | Changes |
|---------|---------|
| 2.2.0 | Async I/O with aiohttp, connection pooling, task-specific regenerate prompts |
| 2.1.0 | Added `/regenerate` endpoint, context length protection |
| 2.0.0 | Added refinement cycle with Redis storage |
| 1.0.0 | Initial release with basic LLM integration |
