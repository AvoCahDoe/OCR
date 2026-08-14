# syntax=docker/dockerfile:1.7
# linux/amd64 only — Paddle GPU wheels are not published for Windows/macOS.
# Slim runtime: GPU wheels + handler only. Models download on first worker start.
# No venv: Kaniko snapshotMode often leaves /opt/venv/bin/python as a dead symlink
# so the worker crash-loops with zero container logs.
# Build: docker build --platform linux/amd64 -t ghcr.io/avocahdoe/ocr-worker:1.0 .

ARG CUDA_IMAGE=nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.11
ARG PADDLE_VERSION=3.2.1
ARG PADDLEOCR_VERSION=3.6.0

FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION
ARG PADDLE_VERSION
ARG PADDLEOCR_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/bin:/usr/bin:$PATH \
    PYTHONPATH=/app/src \
    PADDLE_MODEL_DIR=/models/paddleocr \
    PADDLE_PDX_CACHE_HOME=/models/paddleocr \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_fraction_of_gpu_memory_to_use=0.90 \
    FLAGS_allocator_strategy=auto_growth \
    GPU_TYPE=A5000 \
    GPU_PRICE_PER_SEC=0.0001917 \
    LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        libexpat1 \
        libsqlite3-0 \
        zlib1g \
        libffi8 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ca-certificates \
    && python${PYTHON_VERSION} -c "import sys; print(sys.version)" \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && python${PYTHON_VERSION} -m ensurepip --upgrade \
    && python${PYTHON_VERSION} -m pip install --no-cache-dir --upgrade pip \
    && ldd /usr/bin/python${PYTHON_VERSION} \
    && ldd /usr/bin/python${PYTHON_VERSION} | grep -q libexpat \
    && if ldd /usr/bin/python${PYTHON_VERSION} | grep -q 'not found'; then echo missing_libs >&2; exit 1; fi \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
        paddlepaddle-gpu==${PADDLE_VERSION} \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && python${PYTHON_VERSION} -m pip install --no-cache-dir \
        "paddleocr[doc-parser]==${PADDLEOCR_VERSION}" \
    && python${PYTHON_VERSION} -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && python${PYTHON_VERSION} -c "import sys, runpod, PIL, numpy; print('imports_ok', sys.executable, sys.version)" \
    && find /usr/local/lib -type d -name "__pycache__" -prune -exec rm -rf {} + || true

WORKDIR /app
COPY scripts/entrypoint.sh /app/entrypoint.sh
COPY src /app/src
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
