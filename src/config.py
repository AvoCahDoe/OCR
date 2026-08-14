"""Pinned versions, limits, and runtime configuration."""

from __future__ import annotations

import os
from typing import Any

SCHEMA_VERSION = "1.0"

OCR_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
OCR_PIPELINE_VERSION = "v1.6"

# Cache dir on the worker disk. Missing weights download from Paddle Hub at runtime.
PADDLE_MODEL_DIR = os.environ.get("PADDLE_MODEL_DIR", "/models/paddleocr")
PADDLE_GPU_MEMORY_FRACTION = float(os.environ.get("PADDLE_GPU_MEMORY_FRACTION", "0.90"))

ACTIONS = ("ocr", "health")
DEFAULT_ACTION = "ocr"
OUTPUT_FORMATS = ("plain", "markdown", "layout_json")
DEFAULT_OUTPUT_FORMAT = "plain"

MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "2560"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "5"))
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "TIFF", "BMP"})
URL_FETCH_TIMEOUT_S = float(os.environ.get("URL_FETCH_TIMEOUT_S", "30"))

IDEMPOTENCY_TTL_S = float(os.environ.get("IDEMPOTENCY_TTL_S", "600"))

DEFAULT_GPU_TYPE = os.environ.get("GPU_TYPE", "A5000")
# Serverless 24GB class (A5000 / L4 / 3090 / MIG 24GB) = $0.69/hr
DEFAULT_PRICE_PER_SEC = float(os.environ.get("GPU_PRICE_PER_SEC", "0.0001917"))

GPU_PRICE_TABLE: dict[str, float] = {
    "A5000": 0.0001917,
    "RTX A5000": 0.0001917,
    "L4": 0.0001917,
    "3090": 0.0001917,
    "RTX 3090": 0.0001917,
    "MIG 24GB": 0.0001917,
}


def skip_model_load() -> bool:
    return os.environ.get("SKIP_MODEL_LOAD", "").lower() in ("1", "true", "yes")


def worker_id() -> str:
    return os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_WORKER_ID") or "local"


def gpu_type() -> str:
    return os.environ.get("GPU_TYPE", DEFAULT_GPU_TYPE)


def price_per_sec(gpu: str | None = None) -> float:
    override = os.environ.get("GPU_PRICE_PER_SEC")
    if override:
        return float(override)
    name = gpu or gpu_type()
    return GPU_PRICE_TABLE.get(name, DEFAULT_PRICE_PER_SEC)


def model_versions() -> dict[str, Any]:
    return {"ocr_model": OCR_MODEL_ID}
