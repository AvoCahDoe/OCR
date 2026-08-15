"""Decode, validate, and optionally downscale input images / PDFs."""

from __future__ import annotations

import base64
import io
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader, PdfWriter

from config import (
    ALLOWED_IMAGE_FORMATS,
    BBOX_PAD_PX,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    MAX_PDF_PAGES,
    URL_FETCH_TIMEOUT_S,
)
from schema import InputError

_DATA_URI_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def load_visual(image_spec: str, *, keep_full_res: bool = False) -> dict[str, Any]:
    """Return a Paddle-ready input: numpy image or temp PDF path.

    Result keys:
      kind: "image" | "pdf"
      array: np.ndarray | None  (RGB uint8 for images)
      path: str | None          (temp PDF path)
      cleanup: list[str]        (paths to delete after inference)
    """
    raw = _fetch_bytes(image_spec)
    if len(raw) > MAX_IMAGE_BYTES:
        raise InputError(
            f"Image exceeds MAX_IMAGE_BYTES ({MAX_IMAGE_BYTES} bytes); got {len(raw)}"
        )
    if _looks_like_pdf(raw):
        return _prepare_pdf(raw)
    return _prepare_image(raw, downscale=not keep_full_res)


def _fetch_bytes(image_spec: str) -> bytes:
    spec = image_spec.strip()
    data_uri = _DATA_URI_RE.match(spec)
    if data_uri:
        payload = data_uri.group(3)
        try:
            return base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise InputError("Invalid base64 data URI") from exc
    if _URL_RE.match(spec):
        parsed = urlparse(spec)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise InputError("Invalid image URL")
        try:
            return _download_url_bytes(spec)
        except requests.RequestException as exc:
            raise InputError(f"Failed to fetch image URL: {exc}") from exc
    try:
        return base64.b64decode(spec, validate=False)
    except Exception as exc:
        raise InputError("image must be a URL, data URI, or base64 string") from exc


def _download_url_bytes(url: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    with requests.get(url, timeout=URL_FETCH_TIMEOUT_S, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise InputError(
                    f"Image URL body exceeds MAX_IMAGE_BYTES ({MAX_IMAGE_BYTES} bytes)"
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _looks_like_pdf(raw: bytes) -> bool:
    return raw[:5] == b"%PDF-"


def _prepare_image(raw: bytes, *, downscale: bool = True) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            fmt = (img.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise InputError(
                    f"Unsupported image format {fmt or 'unknown'}; "
                    f"allowed: {sorted(ALLOWED_IMAGE_FORMATS)}"
                )
            rgb = img.convert("RGB")
            if downscale:
                rgb = _downscale(rgb)
            import numpy as np

            array = np.asarray(rgb)
            height, width = array.shape[:2]
    except InputError:
        raise
    except UnidentifiedImageError as exc:
        raise InputError("Image is not decodable") from exc
    except Exception as exc:
        raise InputError(f"Failed to decode image: {exc}") from exc
    return {
        "kind": "image",
        "array": array,
        "path": None,
        "cleanup": [],
        "page_count": 1,
        "width": int(width),
        "height": int(height),
    }


def _downscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_SIDE:
        return img
    scale = MAX_IMAGE_SIDE / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def crop_regions(
    visual: dict[str, Any],
    bboxes: list[dict[str, Any]],
    *,
    pad_px: int | None = None,
) -> list[dict[str, Any]]:
    """Crop original-pixel xyxy boxes from a decoded image. Per-item errors, no throw."""
    if visual.get("kind") != "image" or visual.get("array") is None:
        raise InputError("bboxes are not supported for PDF input")
    array = visual["array"]
    height, width = int(array.shape[0]), int(array.shape[1])
    pad = BBOX_PAD_PX if pad_px is None else int(pad_px)
    crops: list[dict[str, Any]] = []
    for item in bboxes:
        ident = str(item.get("id", ""))
        raw_bbox = [float(v) for v in item["bbox"]]
        label = item.get("label")
        try:
            x0, y0, x1, y1 = _clamp_padded_xyxy(raw_bbox, width, height, pad)
        except InputError as exc:
            crops.append(
                {
                    "id": ident,
                    "bbox": raw_bbox,
                    "label": label,
                    "array": None,
                    "error": exc.message,
                }
            )
            continue
        crop = array[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
            crops.append(
                {
                    "id": ident,
                    "bbox": raw_bbox,
                    "label": label,
                    "array": None,
                    "error": "empty crop",
                }
            )
            continue
        crops.append(
            {
                "id": ident,
                "bbox": raw_bbox,
                "label": label,
                "array": _maybe_downscale_array(crop),
                "error": None,
            }
        )
    return crops


def _clamp_padded_xyxy(
    bbox: list[float], width: int, height: int, pad: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0i = max(0, int(x0) - pad)
    y0i = max(0, int(y0) - pad)
    x1i = min(width, int(x1) + pad)
    y1i = min(height, int(y1) + pad)
    if x1i <= x0i or y1i <= y0i:
        raise InputError("empty crop")
    return x0i, y0i, x1i, y1i


def _maybe_downscale_array(array: Any) -> Any:
    height, width = array.shape[:2]
    if max(height, width) <= MAX_IMAGE_SIDE:
        return array
    import numpy as np

    img = Image.fromarray(array)
    return np.asarray(_downscale(img))


def _prepare_pdf(raw: bytes) -> dict[str, Any]:
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise InputError(f"Invalid PDF: {exc}") from exc
    page_count = len(reader.pages)
    if page_count < 1:
        raise InputError("PDF has no pages")
    if page_count > MAX_PDF_PAGES:
        writer = PdfWriter()
        for page in reader.pages[:MAX_PDF_PAGES]:
            writer.add_page(page)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            writer.write(tmp)
            tmp.flush()
        finally:
            tmp.close()
        return {
            "kind": "pdf",
            "array": None,
            "path": tmp.name,
            "cleanup": [tmp.name],
            "page_count": MAX_PDF_PAGES,
        }
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(raw)
        tmp.flush()
    finally:
        tmp.close()
    return {
        "kind": "pdf",
        "array": None,
        "path": tmp.name,
        "cleanup": [tmp.name],
        "page_count": page_count,
    }


def cleanup_visual(visual: dict[str, Any]) -> None:
    for path in visual.get("cleanup") or []:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
