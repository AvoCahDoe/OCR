"""PaddleOCR-VL-1.6 vision OCR stage."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from models import get_ocr_pipeline
from paddle_compat import ensure_dynamic_graph

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"[ \t]+\n")
_PRINTED_CONTENT_RE = re.compile(r"content:\s*(.*?)(?:\n#{3,}|\Z)", re.DOTALL)
_TITLE_LABELS = frozenset({"paragraph_title", "doc_title", "title"})
_SKIP_LABELS = frozenset(
    {
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
        "image",
        "chart",
        "figure",
    }
)


def run_ocr(visual: dict[str, Any]) -> dict[str, Any]:
    """Run full PaddleOCR-VL pipeline. Returns plain, markdown, and layout."""
    pipeline = get_ocr_pipeline()
    paddle_input = visual["path"] if visual["kind"] == "pdf" else visual["array"]
    ensure_dynamic_graph()
    results = pipeline.predict(
        paddle_input,
        use_queues=False,
        max_new_tokens=768,
    )
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
    from_blocks = _plain_from_parsing_blocks(page)
    if from_blocks:
        return from_blocks
    for attr in ("text", "rec_text", "ocr_text"):
        value = getattr(page, attr, None)
        if isinstance(value, str) and value.strip() and not _looks_like_raw_dump(value):
            return _normalize_plain(value)
    data = _as_dict(page)
    for key in ("text", "rec_text", "ocr_text", "plain"):
        value = data.get(key)
        if isinstance(value, str) and value.strip() and not _looks_like_raw_dump(value):
            return _normalize_plain(value)
    markdown = _extract_markdown(page)
    if markdown:
        return _markdown_to_plain(markdown)
    printed = str(page)
    parsed = _plain_from_printed_dump(printed)
    if parsed:
        return parsed
    return ""


def _extract_markdown(page: Any) -> str:
    md = getattr(page, "markdown", None)
    if callable(md) and not isinstance(md, (str, dict)):
        try:
            md = md()
        except TypeError:
            md = None
    extracted = _markdown_from_value(md)
    if extracted:
        return extracted
    data = _as_dict(page)
    extracted = _markdown_from_value(data.get("markdown") or data.get("md"))
    if extracted:
        return extracted
    return _markdown_from_parsing_blocks(page)


def _extract_layout(page: Any) -> Any:
    data = _as_dict(page)
    blocks = []
    for block in _parsing_blocks(page):
        label = _block_field(block, "label")
        content = _block_content(block)
        if not content and label in _SKIP_LABELS:
            continue
        entry: dict[str, Any] = {
            "label": label,
            "content": content or None,
        }
        bbox = _block_field(block, "bbox") or _block_field(block, "coordinate")
        if bbox is not None:
            entry["bbox"] = _jsonify(bbox)
        score = _block_field(block, "score")
        if score is not None:
            entry["score"] = _jsonify(score)
        blocks.append(entry)
    layout: dict[str, Any] = {
        "width": data.get("width"),
        "height": data.get("height"),
        "blocks": blocks,
    }
    det = data.get("layout_det_res")
    raw_boxes = det.get("boxes") if isinstance(det, dict) else data.get("boxes")
    if isinstance(raw_boxes, list):
        layout["boxes"] = [
            {
                "label": box.get("label"),
                "score": _jsonify(box.get("score")),
                "coordinate": _jsonify(box.get("coordinate")),
            }
            for box in raw_boxes
            if isinstance(box, dict)
        ]
    return layout


def _plain_from_parsing_blocks(page: Any) -> str:
    texts = []
    for block in _parsing_blocks(page):
        label = _block_field(block, "label")
        if label in _SKIP_LABELS:
            continue
        content = _block_content(block)
        if content:
            texts.append(content)
    return _normalize_plain("\n\n".join(texts)) if texts else ""


def _markdown_from_parsing_blocks(page: Any) -> str:
    parts = []
    for block in _parsing_blocks(page):
        label = str(_block_field(block, "label") or "")
        if label in _SKIP_LABELS:
            continue
        content = _block_content(block)
        if not content:
            continue
        if label in _TITLE_LABELS:
            parts.append(f"## {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts).strip()


def _parsing_blocks(page: Any) -> list[Any]:
    data = _as_dict(page)
    blocks = data.get("parsing_res_list")
    if blocks is None:
        blocks = getattr(page, "parsing_res_list", None)
    if blocks is None:
        try:
            blocks = page["parsing_res_list"]  # type: ignore[index]
        except Exception:
            blocks = None
    if not isinstance(blocks, (list, tuple)):
        return []
    return list(blocks)


def _block_content(block: Any) -> str:
    if isinstance(block, str):
        return _content_from_printed_block(block)
    for key in ("content", "block_content", "text", "transcription"):
        value = _block_field(block, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    printed = str(block)
    return _content_from_printed_block(printed)


def _block_field(block: Any, key: str) -> Any:
    if isinstance(block, Mapping):
        return block.get(key)
    return getattr(block, key, None)


def _content_from_printed_block(printed: str) -> str:
    match = _PRINTED_CONTENT_RE.search(printed)
    if not match:
        return ""
    return match.group(1).strip()


def _plain_from_printed_dump(printed: str) -> str:
    if not _looks_like_raw_dump(printed):
        return _normalize_plain(printed) if printed else ""
    parts = [_content_from_printed_block(chunk) for chunk in printed.split("#################")]
    texts = [p for p in parts if p]
    return _normalize_plain("\n\n".join(texts)) if texts else ""


def _looks_like_raw_dump(text: str) -> bool:
    return "parsing_res_list" in text or "layout_det_res" in text or "dtype=uint8" in text


def _markdown_from_value(value: Any) -> str:
    if isinstance(value, str) and value.strip() and not _looks_like_raw_dump(value):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "markdown", "md"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def _as_dict(page: Any) -> dict[str, Any]:
    if isinstance(page, dict):
        if "res" in page and isinstance(page["res"], dict):
            return page["res"]
        return page
    if isinstance(page, Mapping):
        try:
            return {str(k): page[k] for k in page.keys()}
        except Exception:
            pass
    for attr in ("json", "res"):
        value = getattr(page, attr, None)
        if callable(value):
            continue
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
