"""PaddleOCR-VL-1.6 vision OCR stage."""

from __future__ import annotations

import logging
import re
from typing import Any

from models import get_ocr_pipeline
from paddle_compat import ensure_dynamic_graph

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"[ \t]+\n")


def run_ocr(visual: dict[str, Any]) -> dict[str, Any]:
    """Run full PaddleOCR-VL pipeline. Returns plain, markdown, and layout."""
    pipeline = get_ocr_pipeline()
    paddle_input = visual["path"] if visual["kind"] == "pdf" else visual["array"]
    ensure_dynamic_graph()
    results = pipeline.predict(paddle_input, use_queues=False)
    if results is None:
        raise RuntimeError("PaddleOCR-VL returned no results")
    if not isinstance(results, list):
        results = list(results)

    plains: list[str] = []
    markdowns: list[str] = []
    layouts: list[Any] = []
    for page in results:
        plains.append(_extract_plain(page))
        markdowns.append(_extract_markdown(page))
        layouts.append(_extract_layout(page))

    plain = "\n\n".join(p for p in plains if p).strip()
    markdown = "\n\n".join(m for m in markdowns if m).strip()
    return {
        "plain": plain,
        "markdown": markdown or plain,
        "layout": layouts if len(layouts) > 1 else (layouts[0] if layouts else None),
    }


def _extract_plain(page: Any) -> str:
    for attr in ("text", "rec_text", "ocr_text"):
        value = getattr(page, attr, None)
        if isinstance(value, str) and value.strip():
            return _normalize_plain(value)
    data = _as_dict(page)
    for key in ("text", "rec_text", "ocr_text", "plain"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_plain(value)
    markdown = _extract_markdown(page)
    if markdown:
        return _markdown_to_plain(markdown)
    layout = data.get("layout_det_res") or data.get("layout") or {}
    boxes = []
    if isinstance(layout, dict):
        boxes = layout.get("boxes") or layout.get("res") or []
    elif isinstance(layout, list):
        boxes = layout
    texts = []
    for box in boxes or []:
        if not isinstance(box, dict):
            continue
        for key in ("text", "transcription", "content", "block_content", "label"):
            if isinstance(box.get(key), str) and box[key].strip() and key != "label":
                texts.append(box[key].strip())
                break
    if texts:
        return _normalize_plain("\n".join(texts))
    printed = str(page)
    return _normalize_plain(printed) if printed and printed != str(type(page)) else ""


def _extract_markdown(page: Any) -> str:
    md = getattr(page, "markdown", None)
    if isinstance(md, str) and md.strip():
        return md.strip()
    if isinstance(md, dict):
        for key in ("text", "markdown", "md"):
            if isinstance(md.get(key), str) and md[key].strip():
                return md[key].strip()
    data = _as_dict(page)
    for key in ("markdown", "md"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            inner = value.get("text") or value.get("markdown")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def _extract_layout(page: Any) -> Any:
    for attr in ("json", "res", "layout"):
        value = getattr(page, attr, None)
        if value is not None and not callable(value):
            if isinstance(value, dict) or isinstance(value, list):
                return _jsonify(value)
    data = _as_dict(page)
    if data:
        return _jsonify(data)
    return None


def _as_dict(page: Any) -> dict[str, Any]:
    if isinstance(page, dict):
        if "res" in page and isinstance(page["res"], dict):
            return page["res"]
        return page
    for attr in ("json", "res"):
        value = getattr(page, attr, None)
        if isinstance(value, dict):
            if "res" in value and isinstance(value["res"], dict):
                return value["res"]
            return value
    if hasattr(page, "__dict__"):
        return {k: v for k, v in vars(page).items() if not k.startswith("_")}
    return {}


def _jsonify(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _markdown_to_plain(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~]{1,3}", "", text)
    return _normalize_plain(text)


def _normalize_plain(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
