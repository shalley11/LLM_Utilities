#!/bin/bash

# =============================================================================
# LLM Utilities - Service Startup Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================="
    echo "  LLM Utilities - Document AI API"
    echo "============================================================================="
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    echo ""
    print_info "Checking prerequisites..."

    # Check Python
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python3 is not installed"
        exit 1
    fi

    # Check pip packages
    if python3 -c "import fastapi" 2>/dev/null; then
        print_success "FastAPI: installed"
    else
        print_error "FastAPI is not installed. Run: pip install -r requirements.txt"
        exit 1
    fi

    if python3 -c "import uvicorn" 2>/dev/null; then
        print_success "Uvicorn: installed"
    else
        print_error "Uvicorn is not installed. Run: pip install -r requirements.txt"
        exit 1
    fi

    # Check Redis
    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"

    if check_command redis-cli; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &>/dev/null; then
            print_success "Redis: running at $REDIS_HOST:$REDIS_PORT"
        else
            print_warning "Redis: not responding at $REDIS_HOST:$REDIS_PORT"
            print_warning "  Refinement sessions will not work without Redis"
            print_warning "  Start Redis with: redis-server"
        fi
    else
        print_warning "Redis CLI not found - cannot verify Redis status"
    fi

    # Check Ollama
    OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

    if curl -s "${OLLAMA_URL}/api/tags" &>/dev/null; then
        print_success "Ollama: running at $OLLAMA_URL"

        # List available models
        MODELS=$(curl -s "${OLLAMA_URL}/api/tags" | python3 -c "import sys, json; data=json.load(sys.stdin); print(', '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null || echo "")
        if [ -n "$MODELS" ]; then
            print_info "  Available models: $MODELS"
        fi
    else
        print_warning "Ollama: not responding at $OLLAMA_URL"
        print_warning "  LLM features will not work without Ollama"
        print_warning "  Start Ollama with: ollama serve"
    fi

    # Check offline docs assets
    STATIC_DIR="$(dirname "$0")/static"
    if [ -d "$STATIC_DIR/swagger-ui" ] && [ -f "$STATIC_DIR/swagger-ui/swagger-ui-bundle.js" ]; then
        print_success "Offline docs: assets found"
    else
        print_warning "Offline docs: static assets not found"
        print_warning "  Run ./setup_docs.sh to download Swagger UI/ReDoc for offline use"
        print_warning "  /docs and /redoc won't load without internet until assets are downloaded"
    fi

    echo ""
}

# =============================================================================
# Main
# =============================================================================

print_header

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD="true"
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --skip-checks)
            SKIP_CHECKS="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --workers N        Number of worker processes (default: 1)"
            echo "  --reload           Enable auto-reload for development"
            echo "  --log-level LEVEL  Log level: debug, info, warning, error (default: info)"
            echo "  --skip-checks      Skip prerequisite checks"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  HOST, PORT, WORKERS, RELOAD, LOG_LEVEL"
            echo "  REDIS_HOST, REDIS_PORT"
            echo "  OLLAMA_URL, VLLM_URL"
            echo "  DEFAULT_MODEL, LLM_BACKEND"
            echo ""
            echo "Examples:"
            echo "  $0                           # Start with defaults"
            echo "  $0 --reload                  # Start with auto-reload"
            echo "  $0 --port 9000 --workers 4   # Custom port and workers"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run prerequisite checks
if [ "$SKIP_CHECKS" != "true" ]; then
    check_prerequisites
fi

# Build uvicorn command
UVICORN_CMD="python3 -m uvicorn main:app --host $HOST --port $PORT --log-level $LOG_LEVEL"

if [ "$RELOAD" == "true" ]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
    print_info "Auto-reload: enabled"
else
    UVICORN_CMD="$UVICORN_CMD --workers $WORKERS"
    print_info "Workers: $WORKERS"
fi

# Print startup info
print_info "Starting LLM Utilities API..."
print_info "Host: $HOST"
print_info "Port: $PORT"
print_info "Log level: $LOG_LEVEL"
echo ""
print_info "API Documentation:"
print_info "  Swagger UI: http://${HOST}:${PORT}/docs"
print_info "  ReDoc:      http://${HOST}:${PORT}/redoc"
echo ""
print_info "Press Ctrl+C to stop the server"
echo ""
echo "============================================================================="
echo ""

# Start the server
exec $UVICORN_CMD
