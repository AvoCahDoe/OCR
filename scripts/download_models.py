#!/usr/bin/env python3
"""Optional: download PaddleOCR-VL-1.6 weights to a local cache."""

from __future__ import annotations

import os
from pathlib import Path

PADDLE_DIR = Path(os.environ.get("PADDLE_MODEL_DIR", "/models/paddleocr"))
OCR_PIPELINE_VERSION = os.environ.get("OCR_PIPELINE_VERSION", "v1.6")


def download_paddle() -> None:
    PADDLE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLE_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    print(f"Downloading PaddleOCR-VL {OCR_PIPELINE_VERSION} -> {PADDLE_DIR}", flush=True)
    from paddleocr import PaddleOCRVL

    # Instantiating on CPU pulls layout + VL weights into PADDLE_PDX_CACHE_HOME.
    PaddleOCRVL(pipeline_version=OCR_PIPELINE_VERSION, device="cpu", precision="fp32")
    (PADDLE_DIR / ".baked").write_text(f"PaddleOCR-VL {OCR_PIPELINE_VERSION}\n", encoding="utf-8")
    print("PaddleOCR-VL weights ready", flush=True)


if __name__ == "__main__":
    download_paddle()
