from __future__ import annotations

import pytest

from schema import InputError, build_response, error_response, parse_input


def test_parse_defaults():
    parsed = parse_input({"input": {"image": "http://example.com/a.png"}})
    assert parsed["action"] == "ocr"
    assert parsed["output_format"] == "plain"
    assert parsed["request_id"] is None


def test_parse_unknown_action():
    with pytest.raises(InputError, match="Unknown action"):
        parse_input({"input": {"action": "nope", "image": "x"}})


def test_ocr_requires_image():
    with pytest.raises(InputError, match="image"):
        parse_input({"input": {"action": "ocr"}})


def test_health_needs_no_image():
    parsed = parse_input({"input": {"action": "health"}})
    assert parsed["action"] == "health"


def test_unknown_output_format():
    with pytest.raises(InputError, match="output_format"):
        parse_input({"input": {"image": "x", "output_format": "pdf"}})


def test_build_response_shape():
    resp = build_response(
        success=True,
        request_id="abc",
        output={"text": "hello", "markdown": None, "layout": None},
        timing={"ocr_ms": 12.34, "total_ms": 20.0},
    )
    assert resp["schema_version"] == "1.0"
    assert resp["success"] is True
    assert resp["request_id"] == "abc"
    assert resp["worker_id"] == "test-worker"
    assert resp["error"] is None
    assert resp["warning"] is None
    assert resp["model_versions"]["ocr_model"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert "correction_model" not in resp["model_versions"]
    assert resp["cost"]["gpu_type"] == "A5000"
    assert resp["cost"]["billed_seconds"] == 1
    assert resp["timing"]["ocr_ms"] == 12.34
    assert "correction_ms" not in resp["timing"]


def test_error_response():
    resp = error_response(stage="ocr", message="boom", request_id="r1")
    assert resp["success"] is False
    assert resp["error"] == {"stage": "ocr", "message": "boom"}
    assert resp["output"] is None
