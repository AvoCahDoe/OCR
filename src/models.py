"""Module-level PaddleOCR-VL singleton. Load once per worker, not per request."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from config import OCR_PIPELINE_VERSION, PADDLE_GPU_MEMORY_FRACTION, skip_model_load
from paddle_compat import apply_paddleocr_compat, wrap_vl_rec_model
from weights import resolve_paddle_dir

logger = logging.getLogger(__name__)

_paddle_lock = threading.Lock()
_ocr_pipeline: Any = None
_ocr_load_error: str | None = None


def _configure_runtime_env() -> None:
    os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", str(PADDLE_GPU_MEMORY_FRACTION))
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    cache = str(resolve_paddle_dir())
    os.environ["PADDLE_PDX_CACHE_HOME"] = cache
    os.environ["PADDLE_MODEL_DIR"] = cache


def warmup() -> None:
    """Download (if needed) then disk→GPU load once per worker process."""
    if skip_model_load():
        logger.info("SKIP_MODEL_LOAD=1; skipping model warmup")
        return
    _configure_runtime_env()
    logger.info("Warmup paddle_cache=%s", os.environ.get("PADDLE_PDX_CACHE_HOME"))
    get_ocr_pipeline()


def ocr_loaded() -> bool:
    return _ocr_pipeline is not None


def ocr_load_error() -> str | None:
    return _ocr_load_error


def get_ocr_pipeline() -> Any:
    global _ocr_pipeline, _ocr_load_error
    if _ocr_pipeline is not None:
        return _ocr_pipeline
    with _paddle_lock:
        if _ocr_pipeline is not None:
            return _ocr_pipeline
        try:
            _configure_runtime_env()
            # PaddleX asserts if `paddle` is imported first. Keep this the first
            # paddle* import in the process (do not call gpu_available() before).
            from paddleocr import PaddleOCRVL

            apply_paddleocr_compat()
            # Native Paddle backend. Do not pass engine="transformers": that package
            # is not in the image, and it also forces layout detection onto HF.
            _ocr_pipeline = PaddleOCRVL(
                pipeline_version=OCR_PIPELINE_VERSION,
                precision="fp32",
                device="gpu:0",
                vl_rec_backend="native",
                use_queues=False,
            )
            apply_paddleocr_compat()
            wrap_vl_rec_model(_ocr_pipeline)
            marker = Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / ".baked"
            if not marker.is_file():
                marker.write_text(f"PaddleOCR-VL {OCR_PIPELINE_VERSION}\n", encoding="utf-8")
            _ocr_load_error = None
            logger.info(
                "Loaded PaddleOCR-VL %s cache=%s",
                OCR_PIPELINE_VERSION,
                os.environ.get("PADDLE_PDX_CACHE_HOME"),
            )
            return _ocr_pipeline
        except Exception as exc:
            _ocr_load_error = str(exc)
            logger.exception("Failed to load PaddleOCR-VL")
            raise


def gpu_available() -> bool:
    """True if a CUDA device is visible. Never imports paddle before paddleocr."""
    smi = _nvidia_smi_has_gpu()
    if smi is not None:
        return smi
    if "paddleocr" not in sys.modules:
        return False
    try:
        import paddle

        return bool(paddle.device.cuda.device_count())
    except Exception:
        return False


def _nvidia_smi_has_gpu() -> bool | None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "-L"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return False
    return bool((proc.stdout or b"").strip())
