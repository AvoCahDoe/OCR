"""Versioned input validation and response builder."""

from __future__ import annotations

from typing import Any

from config import (
    ACTIONS,
    ALLOWED_BBOX_FORMATS,
    DEFAULT_ACTION,
    DEFAULT_BBOX_FORMAT,
    DEFAULT_OUTPUT_FORMAT,
    MAX_BBOXES,
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
        "bboxes": None,
        "bbox_format": DEFAULT_BBOX_FORMAT,
    }

    if action == "ocr":
        if not parsed["image"] or not str(parsed["image"]).strip():
            raise InputError("image (base64 or URL) is required for this action")
        parsed["image"] = str(parsed["image"]).strip()
        parsed["bboxes"] = _parse_bboxes(raw)
        if parsed["bboxes"] is not None:
            parsed["bbox_format"] = _parse_bbox_format(raw)
            if _looks_like_pdf_spec(parsed["image"]):
                raise InputError("bboxes are not supported for PDF input")
    if parsed["lang"] is not None:
        parsed["lang"] = str(parsed["lang"])

    return parsed


def _parse_bbox_format(raw: dict[str, Any]) -> str:
    fmt = str(raw.get("bbox_format") or DEFAULT_BBOX_FORMAT).strip().lower()
    if fmt not in ALLOWED_BBOX_FORMATS:
        raise InputError(
            f"Unknown bbox_format {fmt!r}; expected one of {sorted(ALLOWED_BBOX_FORMATS)}"
        )
    return fmt


def _parse_bboxes(raw: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "bboxes" not in raw or raw.get("bboxes") is None:
        return None
    boxes = raw["bboxes"]
    if not isinstance(boxes, list):
        raise InputError("bboxes must be a list")
    if not boxes:
        raise InputError("bboxes must be a non-empty list")
    if len(boxes) > MAX_BBOXES:
        raise InputError(f"bboxes exceeds MAX_BBOXES ({MAX_BBOXES}); got {len(boxes)}")
    return [_parse_one_bbox(item, index) for index, item in enumerate(boxes)]


def _parse_one_bbox(item: Any, index: int) -> dict[str, Any]:
    label = None
    ident = str(index)
    coords: Any
    if isinstance(item, (list, tuple)):
        coords = item
    elif isinstance(item, dict):
        coords = item.get("bbox") if item.get("bbox") is not None else item.get("coordinate")
        if item.get("id") is not None:
            ident = str(item["id"]).strip() or ident
        if item.get("label") is not None:
            label = str(item["label"])
    else:
        raise InputError(f"bboxes[{index}] must be [x0,y0,x1,y1] or an object with bbox")
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise InputError(f"bboxes[{index}] bbox must have 4 numbers")
    try:
        x0, y0, x1, y1 = (float(value) for value in coords)
    except (TypeError, ValueError) as exc:
        raise InputError(f"bboxes[{index}] bbox must be numeric") from exc
    return {"id": ident, "bbox": [x0, y0, x1, y1], "label": label}


def _looks_like_pdf_spec(image_spec: str) -> bool:
    lowered = image_spec.strip().lower()
    if lowered.startswith("data:application/pdf"):
        return True
    path = lowered.split("?", 1)[0]
    return path.endswith(".pdf")


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


def _clip_error(message: str, limit: int = 240) -> str:
    text = str(message).split("There are two common workarounds")[0]
    text = text.split("\n")[0].strip().replace("`", "'")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)
