"""Stage timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timed_ms() -> Generator[list[float], None, None]:
    """Yield a one-element list that receives elapsed milliseconds on exit."""
    box: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield box
    finally:
        box[0] = (time.perf_counter() - start) * 1000.0


def now_ms() -> float:
    return time.perf_counter() * 1000.0
