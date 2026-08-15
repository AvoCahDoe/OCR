"""Versioned input validation and response builder."""

from __future__ import annotations

from typing import Any

from config import (
    ACTIONS,
    DEFAULT_ACTION,
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    SCHEMA_VERSION,
    model_versions,
    worker_id,
)
from cost import estimate_cost


class InputError(ValueError):
    def __init__(self, message: str, stage: str = "validation"):
        super().__init__(message)
        self.stage = stage
        self.message = message


def parse_input(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise InputError("job must be an object")
    raw = job.get("input", job)
    if not isinstance(raw, dict):
        raise InputError("input must be an object")

    action = str(raw.get("action") or DEFAULT_ACTION).strip().lower()
    if action not in ACTIONS:
        raise InputError(f"Unknown action {action!r}; expected one of {list(ACTIONS)}")

    output_format = str(raw.get("output_format") or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if output_format not in OUTPUT_FORMATS:
        raise InputError(
            f"Unknown output_format {output_format!r}; expected one of {list(OUTPUT_FORMATS)}"
        )

    request_id = raw.get("request_id")
    if request_id is not None:
        request_id = str(request_id).strip() or None

    parsed = {
        "action": action,
        "image": raw.get("image"),
        "output_format": output_format,
        "lang": raw.get("lang"),
        "request_id": request_id,
    }

    if action == "ocr":
        if not parsed["image"] or not str(parsed["image"]).strip():
            raise InputError("image (base64 or URL) is required for this action")
        parsed["image"] = str(parsed["image"]).strip()
    if parsed["lang"] is not None:
        parsed["lang"] = str(parsed["lang"])

    return parsed


def build_response(
    *,
    success: bool,
    request_id: str | None,
    output: dict[str, Any] | None = None,
    timing: dict[str, float | None] | None = None,
    warning: dict[str, str] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    timing = timing or {"ocr_ms": None, "total_ms": 0.0}
    total_ms = float(timing.get("total_ms") or 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "success": success,
        "request_id": request_id,
        "worker_id": worker_id(),
        "output": output,
        "timing": {
            "ocr_ms": _round_ms(timing.get("ocr_ms")),
            "total_ms": _round_ms(total_ms) or 0.0,
        },
        "cost": estimate_cost(total_ms),
        "model_versions": model_versions(),
        "warning": warning,
        "error": error,
    }


def error_response(
    *,
    stage: str,
    message: str,
    request_id: str | None = None,
    timing: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    return build_response(
        success=False,
        request_id=request_id,
        output=None,
        timing=timing,
        warning=None,
        error={"stage": stage, "message": _clip_error(message)},
    )


def _clip_error(message: str, limit: int = 400) -> str:
    text = str(message).split("There are two common workarounds")[0].strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)
