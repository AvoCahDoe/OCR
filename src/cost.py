"""Estimated RunPod serverless cost (not an invoice)."""

from __future__ import annotations

import math

from config import gpu_type, price_per_sec


def estimate_cost(total_ms: float, gpu: str | None = None) -> dict[str, float | str]:
    name = gpu or gpu_type()
    pps = price_per_sec(name)
    billed_seconds = int(math.ceil(total_ms / 1000.0)) if total_ms > 0 else 0
    return {
        "gpu_type": name,
        "price_per_sec": pps,
        "billed_seconds": billed_seconds,
        "estimated_cost_usd": round(billed_seconds * pps, 10),
    }
