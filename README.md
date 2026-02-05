# LLM_Utilities

A production-ready async Python API for document processing with LLM-powered text extraction, chunking, summarization, translation, and text editing.

## Features

- **Text Extraction**: Extract text from PDF, DOCX, DOC, TXT files with image handling
- **Chunking**: Smart text chunking with page-wise splitting and overlap
- **Summarization**: Hierarchical summarization for large documents
- **Translation**: Multi-language translation with batch support
- **Editor Toolkit**: Text editing (rephrase, professional, proofread, concise)
- **Async I/O**: Non-blocking LLM calls using `aiohttp` for high concurrency
- **Multi-Backend Support**: Ollama and VLLM backends
- **Context Length Protection**: Model-wise context limits with warnings and rejection
- **Refinement Cycle**: Iteratively improve outputs with user feedback
- **Comprehensive Logging**: Structured logs with request tracing and metrics
- **Modular Architecture**: Each service has its own config and can use different models

---

## Installation

```bash
pip install -r requirements.txt
```

### Prerequisites
- Redis server running on `localhost:6379`
- Ollama running on `localhost:11434` (or VLLM on `localhost:8000`)

---

## Quick Start

### Run the API Server

**Using the startup script (recommended):**
```bash
./start.sh                    # Start with default settings
./start.sh --reload           # Start with auto-reload for development
./start.sh --port 9000        # Custom port
./start.sh --workers 4        # Multiple workers for production
./start.sh --help             # Show all options
```

**Using uvicorn directly:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The startup script automatically checks prerequisites (Redis, Ollama) and provides helpful status messages.

### Access Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Project Structure

```
LLM_Utilities/
├── config.py                 # Global shared configuration
├── main.py                   # FastAPI application
├── schemas.py                # Core Pydantic models
├── requirements.txt          # Dependencies
├── start.sh                  # Service startup script
│
├── LLM/                      # LLM client module
│   ├── config.py             # LLM client settings
│   ├── llm_client.py         # Async LLM backend (Ollama/VLLM)
│   └── prompts.py            # Core task prompts
│
├── text_extractor/           # Text extraction module
│   ├── config.py             # Extractor-specific settings
│   ├── service.py            # FastAPI endpoints
│   ├── extractor.py          # Core extraction logic
│   └── schemas.py            # Pydantic models
│
├── chunking/                 # Chunking module
│   ├── config.py             # Chunking and vision settings
│   ├── llm_client.py         # Module-specific LLM client
│   ├── service.py            # FastAPI endpoints
│   ├── chunker.py            # Core chunking logic
│   ├── vision_processor.py   # Vision model for images
│   └── schemas.py            # Pydantic models
│
├── summarization/            # Summarization module
│   ├── config.py             # Summarization-specific settings
│   ├── llm_client.py         # Module-specific LLM client
│   ├── service.py            # FastAPI endpoints
│   ├── summarizer.py         # Hierarchical summarization logic
│   ├── prompts.py            # Summary prompts
│   └── schemas.py            # Pydantic models
│
├── translation/              # Translation module
│   ├── config.py             # Translation-specific settings
│   ├── llm_client.py         # Module-specific LLM client
│   ├── service.py            # FastAPI endpoints
│   ├── translator.py         # Core translation logic
│   ├── prompts.py            # Translation prompts
│   └── schemas.py            # Pydantic models
│
├── editortoolkit/            # Editor toolkit module
│   ├── config.py             # Editor-specific settings
│   ├── llm_client.py         # Module-specific LLM client
│   ├── service.py            # FastAPI endpoints
│   ├── editor.py             # Core editing logic
│   ├── prompts.py            # Editing prompts
│   └── schemas.py            # Pydantic models
│
├── refinements/              # Refinement session storage
│   └── refinement_store.py   # Redis session management
│
└── logs/                     # Log files (auto-created)
    └── logging_config.py     # Logging configuration
```

---

## API Endpoints

### Text Extraction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/extract/upload` | POST | Extract text from uploaded file |
| `/api/docAI/v1/extract/upload/simple` | POST | Simple extraction (markdown only) |
| `/api/docAI/v1/extract/path` | POST | Extract from server file path |
| `/api/docAI/v1/extract/cleanup/{document_id}` | DELETE | Cleanup extracted files |
| `/api/docAI/v1/extract/supported-types` | GET | List supported file types |

### Chunking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/chunk/process` | POST | Create chunks from markdown text |
| `/api/docAI/v1/chunk/text` | POST | Create chunks from plain text |
| `/api/docAI/v1/chunk/file` | POST | Create chunks from uploaded file |
| `/api/docAI/v1/chunk/config` | POST | Calculate optimal chunk configuration |
| `/api/docAI/v1/chunk/models` | GET | List supported models |

### Summarization

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/summarize/chunks` | POST | Hierarchical summarization of chunks |
| `/api/docAI/v1/summarize/text` | POST | Summarize text with auto-chunking |
| `/api/docAI/v1/summarize/file` | POST | Summarize uploaded file |
| `/api/docAI/v1/summarize/config` | GET | Get summarization config |
| `/api/docAI/v1/summarize/types` | GET | Get available summary types |

### Translation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/translate/text` | POST | Translate single text |
| `/api/docAI/v1/translate/batch` | POST | Translate multiple texts |
| `/api/docAI/v1/translate/config` | GET | Get translation config |

### Editor Toolkit

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/editor/edit` | POST | Edit text (rephrase, professional, proofread, concise) |
| `/api/docAI/v1/editor/refine` | POST | Refine previous edit with feedback |
| `/api/docAI/v1/editor/batch` | POST | Batch edit multiple texts |
| `/api/docAI/v1/editor/tasks` | GET | List supported editing tasks |
| `/api/docAI/v1/editor/config` | GET | Get editor config |

### Refinements (Session Management)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/refinements/create` | POST | Create new refinement session |
| `/api/docAI/v1/refinements/get/{request_id}` | GET | Get session data |
| `/api/docAI/v1/refinements/update` | POST | Update session with refined result |
| `/api/docAI/v1/refinements/regenerate` | POST | Update session with regenerated result |
| `/api/docAI/v1/refinements/delete/{request_id}` | DELETE | Delete session |
| `/api/docAI/v1/refinements/extend` | POST | Extend session TTL |
| `/api/docAI/v1/refinements/status/{request_id}` | GET | Get session status |
| `/api/docAI/v1/refinements/config` | GET | Get refinement config |

### Core Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/docAI/v1/process` | POST | Process text with task |
| `/api/docAI/v1/refine` | POST | Refine previous result |
| `/api/docAI/v1/regenerate` | POST | Regenerate from original |
| `/api/docAI/v1/status` | POST | Get session status |
| `/api/docAI/v1/delete` | POST | End session |
| `/api/docAI/v1/extend` | POST | Extend session TTL |
| `/health` | GET | Health check |

---

## Configuration

### Global Configuration (`config.py`)

```python
# LLM Backend
LLM_BACKEND = "ollama"  # or "vllm"
OLLAMA_URL = "http://localhost:11434"
VLLM_URL = "http://localhost:8000"
DEFAULT_MODEL = "gemma3:4b"

# Model Context Lengths
MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,
    "gemma3:12b": 8192,
}

# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REFINEMENT_TTL = 7200  # 2 hours
```

### Module-Specific Configuration

Each module has its own `config.py` with module-specific settings:

**LLM Client** (`LLM/config.py`):
```python
LLM_CONNECTION_TIMEOUT = 300          # HTTP timeout in seconds
LLM_CONNECTION_POOL_LIMIT = 100       # Max connections
LLM_CONNECTION_POOL_LIMIT_PER_HOST = 20  # Max per host
LLM_DEFAULT_TEMPERATURE = 0.3         # Generation temperature
LLM_DEFAULT_MAX_TOKENS = 1024         # Max tokens to generate
```

**Summarization** (`summarization/config.py`):
```python
# LLM Backend (can use different endpoint than global)
SUMMARIZATION_LLM_BACKEND = "ollama"  # ollama | vllm
SUMMARIZATION_OLLAMA_URL = "http://localhost:11434"
SUMMARIZATION_VLLM_URL = "http://localhost:8000"
SUMMARIZATION_DEFAULT_MODEL = "gemma3:4b"
SUMMARIZATION_DEFAULT_TYPE = "detailed"
SUMMARIZATION_MAX_WORDS_PER_BATCH = 3000
SUMMARIZATION_TEMPERATURE = 0.3
SUMMARIZATION_MAX_TOKENS = 2048
SUMMARIZATION_CONNECTION_TIMEOUT = 300
SUMMARIZATION_CONNECTION_POOL_LIMIT = 50
SUMMARIZATION_MAX_TOKEN_PERCENT = 80  # Token limit guardrail
```

**Translation** (`translation/config.py`):
```python
# LLM Backend (can use different endpoint than global)
TRANSLATION_LLM_BACKEND = "ollama"    # ollama | vllm
TRANSLATION_OLLAMA_URL = "http://localhost:11434"
TRANSLATION_VLLM_URL = "http://localhost:8000"
TRANSLATION_DEFAULT_MODEL = "gemma3:4b"
TRANSLATION_MAX_BATCH_SIZE = 50
TRANSLATION_TEMPERATURE = 0.3
TRANSLATION_MAX_TOKENS = 2048
TRANSLATION_CONNECTION_TIMEOUT = 300
TRANSLATION_CONNECTION_POOL_LIMIT = 50
TRANSLATION_MAX_TOKEN_PERCENT = 80    # Token limit guardrail
```

**Editor Toolkit** (`editortoolkit/config.py`):
```python
# LLM Backend (can use different endpoint than global)
EDITOR_LLM_BACKEND = "ollama"         # ollama | vllm
EDITOR_OLLAMA_URL = "http://localhost:11434"
EDITOR_VLLM_URL = "http://localhost:8000"
EDITOR_DEFAULT_MODEL = "gemma3:4b"
EDITOR_SUPPORTED_TASKS = ["rephrase", "professional", "proofread", "concise"]
EDITOR_TEMPERATURE = 0.3
EDITOR_MAX_TOKENS = 2048
EDITOR_CONNECTION_TIMEOUT = 300
EDITOR_CONNECTION_POOL_LIMIT = 50
EDITOR_MAX_TOKEN_PERCENT = 80         # Token limit guardrail
```

**Text Extractor** (`text_extractor/config.py`):
```python
EXTRACTOR_MAX_PAGES = 50              # Page limit guardrail
EXTRACTOR_MAX_FILE_SIZE_MB = 50       # File size limit
```

**Logging** (`logs/config.py`):
```python
LOG_OUTPUT_DIR = "logs/output/"       # Log files directory
LOG_MAX_BYTES = 10 * 1024 * 1024      # Max log file size (10MB)
LOG_BACKUP_COUNT = 5                  # Number of backup files
LOG_PREVIEW_LENGTH = 200              # Preview length in logs
```

**Refinements** (`refinements/config.py`):
```python
REFINEMENT_KEY_PREFIX = "refine"      # Redis key prefix
REFINEMENT_DEFAULT_TTL = 7200         # Session TTL (2 hours)
REFINEMENT_MAX_ITERATIONS = 10        # Max refinements per session
REFINEMENT_MAX_REGENERATIONS = 5      # Max regenerations per session
```

**Chunking** (`chunking/config.py`):
```python
# LLM Backend (can use different endpoint than global)
CHUNKING_LLM_BACKEND = "ollama"       # ollama | vllm
CHUNKING_OLLAMA_URL = "http://localhost:11434"
CHUNKING_VLLM_URL = "http://localhost:8000"
CHUNKING_DEFAULT_MODEL = "gemma3:4b"
CHUNKING_CONNECTION_TIMEOUT = 300
CHUNKING_CONNECTION_POOL_LIMIT = 50

# Chunking settings
CHUNKING_DEFAULT_OVERLAP = 200        # Overlap between chunks (characters)
CHUNKING_DEFAULT_RESERVE_FOR_PROMPT = 1000  # Tokens reserved for prompt
CHUNKING_DEFAULT_PROCESS_IMAGES = True  # Process images with Vision model
CHUNKING_MIN_TEXT_LENGTH = 10         # Minimum text length for chunking
CHUNKING_DEFAULT_CONTEXT_LENGTH = 8192  # Default context for unknown models
CHUNKING_CHARS_PER_TOKEN = 4          # Characters per token estimate
CHUNKING_MAX_BATCH_SIZE = 100         # Max batch size for processing

# Vision processor settings
VISION_MODEL = "gemma3:4b"            # Vision model for image processing
VISION_OLLAMA_URL = "http://localhost:11434"  # Ollama URL for vision
VISION_REQUEST_TIMEOUT = 120          # Request timeout (seconds)
VISION_MAX_CONCURRENT = 3             # Max concurrent vision requests
VISION_TEMPERATURE = 0.3              # Vision model temperature
VISION_MAX_TOKENS = 500               # Max tokens for vision output
```

### Environment Variables

Configure via environment variables:

```bash
# Global
export DEFAULT_MODEL="gemma3:12b"
export LLM_BACKEND="vllm"
export OLLAMA_URL="http://localhost:11434"
export VLLM_URL="http://localhost:8000"

# LLM Client
export LLM_CONNECTION_TIMEOUT="300"
export LLM_CONNECTION_POOL_LIMIT="100"
export LLM_CONNECTION_POOL_LIMIT_PER_HOST="20"
export LLM_DEFAULT_TEMPERATURE="0.3"
export LLM_DEFAULT_MAX_TOKENS="1024"

# Module-specific LLM backends (each module can use different endpoints)
# Translation
export TRANSLATION_LLM_BACKEND="ollama"
export TRANSLATION_OLLAMA_URL="http://localhost:11434"
export TRANSLATION_VLLM_URL="http://localhost:8000"
export TRANSLATION_DEFAULT_MODEL="gemma3:12b"
export TRANSLATION_MAX_TOKENS="2048"
export TRANSLATION_CONNECTION_TIMEOUT="300"

# Summarization
export SUMMARIZATION_LLM_BACKEND="ollama"
export SUMMARIZATION_OLLAMA_URL="http://localhost:11434"
export SUMMARIZATION_VLLM_URL="http://localhost:8000"
export SUMMARIZATION_DEFAULT_MODEL="llama3:8b"
export SUMMARIZATION_MAX_TOKENS="2048"
export SUMMARIZATION_CONNECTION_TIMEOUT="300"

# Editor Toolkit
export EDITOR_LLM_BACKEND="ollama"
export EDITOR_OLLAMA_URL="http://localhost:11434"
export EDITOR_VLLM_URL="http://localhost:8000"
export EDITOR_DEFAULT_MODEL="gemma3:4b"
export EDITOR_MAX_TOKENS="2048"
export EDITOR_CONNECTION_TIMEOUT="300"

# Chunking
export CHUNKING_LLM_BACKEND="ollama"
export CHUNKING_OLLAMA_URL="http://localhost:11434"
export CHUNKING_VLLM_URL="http://localhost:8000"
export CHUNKING_DEFAULT_MODEL="gemma3:4b"
export CHUNKING_CONNECTION_TIMEOUT="300"

# Guardrails
export EXTRACTOR_MAX_PAGES="50"
export TRANSLATION_MAX_TOKEN_PERCENT="80"
export EDITOR_MAX_TOKEN_PERCENT="80"
export SUMMARIZATION_MAX_TOKEN_PERCENT="80"

# Logging
export LOG_OUTPUT_DIR="logs/output/"
export LOG_MAX_BYTES="10485760"
export LOG_BACKUP_COUNT="5"
export LOG_PREVIEW_LENGTH="200"

# Refinements
export REFINEMENT_KEY_PREFIX="refine"
export REFINEMENT_DEFAULT_TTL="7200"
export REFINEMENT_MAX_ITERATIONS="10"
export REFINEMENT_MAX_REGENERATIONS="5"

# Chunking
export CHUNKING_DEFAULT_OVERLAP="200"
export CHUNKING_DEFAULT_RESERVE_FOR_PROMPT="1000"
export CHUNKING_DEFAULT_PROCESS_IMAGES="true"
export CHUNKING_MIN_TEXT_LENGTH="10"
export CHUNKING_DEFAULT_CONTEXT_LENGTH="8192"
export CHUNKING_CHARS_PER_TOKEN="4"
export CHUNKING_MAX_BATCH_SIZE="100"

# Vision Processor
export VISION_MODEL="gemma3:4b"
export VISION_OLLAMA_URL="http://localhost:11434"
export VISION_REQUEST_TIMEOUT="120"
export VISION_MAX_CONCURRENT="3"
export VISION_TEMPERATURE="0.3"
export VISION_MAX_TOKENS="500"

# Redis
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

---

## API Examples

### Translation

```bash
curl -X POST http://localhost:8000/api/docAI/v1/translate/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "target_language": "spanish",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "request_id": "uuid-here",
  "original_text": "Hello, how are you?",
  "translated_text": "Hola, ¿cómo estás?",
  "source_language": "auto-detected",
  "target_language": "spanish",
  "model": "gemma3:4b",
  "status": "completed"
}
```

### Editor Toolkit

```bash
curl -X POST http://localhost:8000/api/docAI/v1/editor/edit \
  -H "Content-Type: application/json" \
  -d '{
    "text": "i think we should probably maybe consider this",
    "task": "professional",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "request_id": "uuid-here",
  "original_text": "i think we should probably maybe consider this",
  "edited_text": "We should consider this matter.",
  "task": "professional",
  "model": "gemma3:4b",
  "status": "completed"
}
```

### Summarization

```bash
curl -X POST http://localhost:8000/api/docAI/v1/summarize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long document text here...",
    "summary_type": "brief",
    "user_id": "user123
  }'
```

---

## Request Tracking

All endpoints support `request_id` for tracking:

- **Fresh request**: If `request_id` is not provided, a new UUID is generated
- **Subsequent call**: If `request_id` is provided, it's used for tracking the chain of requests

```json
{
  "request_id": "existing-uuid",  // Use existing ID for tracking
  "text": "...",
  "task": "..."
}
```

---

## Editor Tasks

| Task | Description |
|------|-------------|
| `rephrase` | Improve clarity, readability, and remove repetitions |
| `professional` | Rewrite in formal business tone |
| `proofread` | Fix grammar, spelling, punctuation |
| `concise` | Shorten text, remove unnecessary words |

---

## Summary Types

| Type | Description |
|------|-------------|
| `brief` | Short summary capturing core idea |
| `detailed` | Comprehensive summary covering all points |
| `bulletwise` | Bullet-point summary |

---

## Guardrails

The API includes built-in guardrails to prevent processing failures and ensure reliable operation.

### Document Page Limit

Documents exceeding **50 pages** are rejected during extraction.

**Configuration** (`text_extractor/config.py`):
```python
EXTRACTOR_MAX_PAGES = 50  # Configurable via EXTRACTOR_MAX_PAGES env var
```

**Error Response:**
```json
{
  "detail": {
    "error": "document_too_long",
    "message": "Document has 75 pages which exceeds the maximum allowed limit of 50 pages.",
    "total_pages": 75,
    "max_pages": 50,
    "suggestion": "Please split the document into smaller parts (max 50 pages each) and process them separately."
  }
}
```

### Token Limit

Text exceeding **80% of model context length** is rejected before processing.

**Configuration** (in each module's `config.py`):
```python
# translation/config.py
TRANSLATION_MAX_TOKEN_PERCENT = 80

# editortoolkit/config.py
EDITOR_MAX_TOKEN_PERCENT = 80

# summarization/config.py
SUMMARIZATION_MAX_TOKEN_PERCENT = 80
```

**Error Response:**
```json
{
  "detail": {
    "error": "token_limit_exceeded",
    "message": "Text contains approximately 8,500 tokens which exceeds the maximum allowed limit of 6,553 tokens.",
    "estimated_tokens": 8500,
    "max_tokens": 6553,
    "model": "gemma3:4b",
    "model_context_length": 8192,
    "usage_percent": 129.72,
    "suggestion": "Please reduce the text length or split into smaller chunks for processing."
  }
}
```

### Guardrail Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACTOR_MAX_PAGES` | 50 | Maximum pages for document extraction |
| `TRANSLATION_MAX_TOKEN_PERCENT` | 80 | Max input tokens as % of context |
| `EDITOR_MAX_TOKEN_PERCENT` | 80 | Max input tokens as % of context |
| `SUMMARIZATION_MAX_TOKEN_PERCENT` | 80 | Max input tokens as % of context |

---

## Error Handling

| HTTP Code | Reason |
|-----------|--------|
| 400 | Bad request / Document too long / Token limit exceeded |
| 404 | Request ID not found or expired |
| 500 | Internal server error |

---

## Logging

Log files in `logs/` directory:

| File | Content |
|------|---------|
| `llm_requests.log` | All LLM API calls |
| `llm_errors.log` | Error-level logs |
| `llm_metrics.log` | Performance metrics (JSON) |
| `llm_debug.log` | Detailed debug info |

---

## Version History

| Version | Changes |
|---------|---------|
| 3.1.0 | Guardrails: page limit (50), token limit validation across all services |
| 3.0.0 | Modular architecture, translation, editor toolkit, module-specific configs |
| 2.5.0 | Text extraction, chunking, summarization modules |
| 2.2.0 | Async I/O with aiohttp, connection pooling |
| 2.0.0 | Refinement cycle with Redis storage |
| 1.0.0 | Initial release |
