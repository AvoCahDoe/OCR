from __future__ import annotations

from unittest.mock import patch

import handler as handler_mod
from handler import handler


def _reset_cache():
    handler_mod._response_cache.clear()


def test_ocr_plain():
    _reset_cache()
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "image", "array": "img", "path": None, "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch(
            "handler.run_ocr",
            return_value={"plain": "hello", "markdown": "# hello", "layout": [{"t": 1}]},
        ) as ocr,
    ):
        resp = handler({"input": {"image": "aGVsbG8=", "request_id": "ocr-1"}})
    assert resp["success"] is True
    assert resp["output"]["text"] == "hello"
    assert resp["output"]["markdown"] is None
    assert resp["output"]["layout"] is None
    assert "corrected_text" not in resp["output"]
    ocr.assert_called_once()


def test_ocr_markdown_format():
    _reset_cache()
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "image", "array": "img", "path": None, "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch(
            "handler.run_ocr",
            return_value={"plain": "hi", "markdown": "# hi", "layout": None},
        ),
    ):
        resp = handler({"input": {"image": "x", "output_format": "markdown"}})
    assert resp["output"]["text"] == "hi"
    assert resp["output"]["markdown"] == "# hi"
    assert resp["output"]["layout"] is None


def test_layout_format():
    _reset_cache()
    layout = [{"label": "text", "text": "hi"}]
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "image", "array": "img", "path": None, "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch(
            "handler.run_ocr",
            return_value={"plain": "hi", "markdown": "# hi", "layout": layout},
        ),
    ):
        resp = handler({"input": {"image": "x", "output_format": "layout_json"}})
    assert resp["output"]["layout"] == layout
    assert resp["output"]["markdown"] is None
    assert resp["output"]["text"] == "hi"


def test_ocr_failure():
    _reset_cache()
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "image", "array": "img", "path": None, "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch("handler.run_ocr", side_effect=RuntimeError("paddle down")),
    ):
        resp = handler({"input": {"image": "x"}})
    assert resp["success"] is False
    assert resp["error"]["stage"] == "ocr"
    assert "paddle down" in resp["error"]["message"]


def test_health_action():
    _reset_cache()
    with (
        patch("handler.gpu_available", return_value=True),
        patch("handler.ocr_loaded", return_value=True),
    ):
        resp = handler({"input": {"action": "health"}})
    assert resp["success"] is True
    assert resp["output"]["gpu_available"] is True
    assert resp["output"]["ocr_loaded"] is True


def test_validation_error():
    _reset_cache()
    resp = handler({"input": {"action": "ocr"}})
    assert resp["success"] is False
    assert resp["error"]["stage"] == "validation"
    assert "image" in resp["error"]["message"]


def test_idempotency_cache():
    _reset_cache()
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "image", "array": "img", "path": None, "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch(
            "handler.run_ocr",
            return_value={"plain": "once", "markdown": None, "layout": None},
        ) as ocr,
    ):
        a = handler({"input": {"image": "x", "request_id": "same"}})
        b = handler({"input": {"image": "x", "request_id": "same"}})
    assert a["output"]["text"] == "once"
    assert b["output"]["text"] == "once"
    assert b["timing"]["ocr_ms"] == a["timing"]["ocr_ms"]
    assert ocr.call_count == 1
