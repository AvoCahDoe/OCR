"""Resolve local PaddleOCR-VL cache dirs. Runtime download is allowed."""

from __future__ import annotations

import os
from pathlib import Path

BAKED_PADDLE_DIR = Path("/models/paddleocr")
VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
DEFAULT_PADDLE_CACHE = Path("/models/paddleocr")


def _volume_paddle() -> Path:
    return VOLUME_ROOT / "paddleocr"


def _is_usable_dir(path: Path, *, markers: tuple[str, ...]) -> bool:
    if not path.is_dir():
        return False
    if any((path / name).is_file() for name in markers):
        return True
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _candidate_paddle_dirs() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("PADDLE_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend((_volume_paddle(), BAKED_PADDLE_DIR, DEFAULT_PADDLE_CACHE))
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_paddle_dir() -> Path:
    """Writable PaddleOCR-VL cache. Existing weights preferred; else first candidate (created)."""
    candidates = _candidate_paddle_dirs()
    for path in candidates:
        if _is_usable_dir(path, markers=(".baked",)):
            return path.resolve()
    # Dockerfile default PADDLE_MODEL_DIR=/models/paddleocr would otherwise win over an
    # empty network volume. Prefer the volume for new downloads when it is mounted.
    if VOLUME_ROOT.is_dir():
        cache = _volume_paddle()
    else:
        cache = candidates[0]
    cache.mkdir(parents=True, exist_ok=True)
    return cache.resolve()
