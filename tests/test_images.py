from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter

from images import load_visual
from schema import InputError


def _png_b64(width: int = 32, height: int = 32, color=(255, 0, 0)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_load_png():
    visual = load_visual(_png_b64())
    assert visual["kind"] == "image"
    assert visual["array"].ndim == 3
    assert visual["array"].shape[2] == 3


def test_reject_empty_base64_garbage():
    with pytest.raises(InputError):
        load_visual("@@@not-valid-base64-or-url@@@")


def test_reject_oversized_bytes(monkeypatch):
    monkeypatch.setattr("images.MAX_IMAGE_BYTES", 16)
    with pytest.raises(InputError, match="MAX_IMAGE_BYTES"):
        load_visual(_png_b64())


def test_downscale_large_image(monkeypatch):
    monkeypatch.setattr("images.MAX_IMAGE_SIDE", 64)
    visual = load_visual(_png_b64(width=200, height=100, color=(0, 128, 255)))
    h, w = visual["array"].shape[:2]
    assert max(h, w) <= 64


def test_url_download_stops_at_size_cap(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=65536):
            yield b"x" * 32
            yield b"y" * 32

    monkeypatch.setattr("images.MAX_IMAGE_BYTES", 40)
    monkeypatch.setattr("images.requests.get", lambda *_a, **_k: FakeResponse())
    with pytest.raises(InputError, match="MAX_IMAGE_BYTES"):
        load_visual("https://example.com/huge.png")


def test_pdf_page_limit(monkeypatch):
    monkeypatch.setattr("images.MAX_PDF_PAGES", 2)
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    visual = load_visual(base64.b64encode(buf.getvalue()).decode("ascii"))
    assert visual["kind"] == "pdf"
    assert visual["path"]
    from pypdf import PdfReader

    assert len(PdfReader(visual["path"]).pages) == 2
    Path(visual["path"]).unlink(missing_ok=True)
