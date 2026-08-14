#!/bin/sh
# Tiny boot probe so RunPod container logs show Python started (venv/symlink
# crashes otherwise print nothing and look like a start-container loop).
set -eu
echo "ocr-worker boot $(date -u +%Y-%m-%dT%H:%M:%SZ) python=$(command -v python3.11 2>/dev/null || echo missing)"
python3.11 -c "import sys; print('python_ok', sys.executable, sys.version.replace(chr(10), ' '))"
exec python3.11 -u /app/src/handler.py
