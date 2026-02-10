#!/bin/bash

# =============================================================================
# Download Swagger UI and ReDoc assets for offline API documentation
#
# Run this script ONCE while you have internet access.
# After running, /docs and /redoc will work without internet.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_DIR="$SCRIPT_DIR/static"

echo "[INFO] Setting up offline API documentation assets..."

# Create directories
mkdir -p "$STATIC_DIR/swagger-ui"
mkdir -p "$STATIC_DIR/redoc"

# Swagger UI assets (matching FastAPI's default CDN version)
echo "[INFO] Downloading Swagger UI assets..."
curl -sL "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js" \
    -o "$STATIC_DIR/swagger-ui/swagger-ui-bundle.js"
curl -sL "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" \
    -o "$STATIC_DIR/swagger-ui/swagger-ui.css"

# ReDoc assets
echo "[INFO] Downloading ReDoc assets..."
curl -sL "https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js" \
    -o "$STATIC_DIR/redoc/redoc.standalone.js"

# Favicon
echo "[INFO] Downloading favicon..."
curl -sL "https://fastapi.tiangolo.com/img/favicon.png" \
    -o "$STATIC_DIR/favicon.png"

# Verify downloads
echo "[INFO] Verifying downloads..."
FAIL=0
for f in \
    "$STATIC_DIR/swagger-ui/swagger-ui-bundle.js" \
    "$STATIC_DIR/swagger-ui/swagger-ui.css" \
    "$STATIC_DIR/redoc/redoc.standalone.js" \
    "$STATIC_DIR/favicon.png"; do
    if [ -s "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OK] $(basename "$f") ($SIZE)"
    else
        echo "  [FAIL] $(basename "$f") - file is empty or missing"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 1 ]; then
    echo ""
    echo "[ERROR] Some downloads failed. Check your internet connection and try again."
    exit 1
fi

echo ""
echo "[SUCCESS] Offline docs assets are ready in: $STATIC_DIR"
echo "  /docs and /redoc will now work without internet access."
