from __future__ import annotations

import logging
from unittest.mock import patch

import handler as handler_mod
from handler import handler
import worker_status as ws


def _reset_cache():
    handler_mod._response_cache.clear()
    ws.reset_for_tests()


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
    assert resp["output"]["regions"] is None
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


def test_ocr_logs_state_input_output(caplog):
    _reset_cache()
    with (
        patch(
            "handler.load_visual",
            return_value={
                "kind": "image",
                "array": "img",
                "path": None,
                "cleanup": [],
                "page_count": 1,
                "width": 482,
                "height": 284,
            },
        ),
        patch("handler.cleanup_visual"),
        patch(
            "handler.run_ocr",
            return_value={"plain": "Article 2. – L'indemnisation", "markdown": None, "layout": None},
        ),
        caplog.at_level(logging.INFO),
    ):
        handler(
            {
                "id": "job-9",
                "input": {
                    "image": "https://cdn.example.com/scan.png?token=secret",
                    "request_id": "art-2",
                },
            }
        )
    text = caplog.text
    assert "JOB start #1" in text
    assert "url:https://cdn.example.com/scan.png" in text
    assert "secret" not in text
    assert "INPUT decoded kind=image 482x284 pages=1" in text
    assert "JOB done #1 ok" in text
    assert "Article 2." in text
    assert "images=1" in text
    assert "idle_left=" in text


def test_ocr_regions_path():
    _reset_cache()
    regions = [
        {"id": "r1", "bbox": [1, 2, 3, 4], "text": "Hello", "error": None},
        {"id": "r2", "bbox": [5, 6, 7, 8], "text": "World", "error": None},
    ]
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "image", "array": "img", "path": None, "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch("handler.run_ocr") as page_ocr,
        patch(
            "handler.run_ocr_regions",
            return_value={
                "plain": "Hello\n\nWorld",
                "markdown": None,
                "layout": None,
                "regions": regions,
            },
        ) as region_ocr,
    ):
        resp = handler(
            {
                "input": {
                    "image": "x",
                    "bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
                    "request_id": "reg-1",
                }
            }
        )
    assert resp["success"] is True
    assert resp["output"]["text"] == "Hello\n\nWorld"
    assert resp["output"]["regions"] == regions
    assert resp["output"]["layout"] is None
    region_ocr.assert_called_once()
    page_ocr.assert_not_called()


def test_pdf_with_bboxes_fails():
    _reset_cache()
    with (
        patch(
            "handler.load_visual",
            return_value={"kind": "pdf", "array": None, "path": "/tmp/x.pdf", "cleanup": []},
        ),
        patch("handler.cleanup_visual"),
        patch("handler.run_ocr") as page_ocr,
        patch("handler.run_ocr_regions") as region_ocr,
    ):
        resp = handler({"input": {"image": "x", "bboxes": [[0, 0, 1, 1]]}})
    assert resp["success"] is False
    assert "PDF" in resp["error"]["message"]
    page_ocr.assert_not_called()
    region_ocr.assert_not_called()
