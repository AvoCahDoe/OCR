from __future__ import annotations

import logging
import time

import worker_status as ws
from worker_status import (
    WorkerStatus,
    preview_text,
    summarize_image_spec,
)


def test_summarize_url_redacts_query():
    line = summarize_image_spec("https://cdn.example.com/a.png?token=secret")
    assert line.startswith("url:https://cdn.example.com/a.png")
    assert "secret" not in line
    assert "…" in line


def test_summarize_base64_does_not_dump_payload():
    payload = "aGVsbG8=" * 40
    line = summarize_image_spec(payload)
    assert line.startswith("base64 chars=")
    assert payload not in line


def test_summarize_data_uri():
    line = summarize_image_spec("data:image/png;base64,iVBORw0KGgo=")
    assert "data_uri" in line
    assert "image/png" in line
    assert "iVBORw0KGgo=" not in line


def test_preview_collapses_and_truncates():
    text = "Article 2.\n\nL'indemnisation  " + ("x" * 400)
    preview = preview_text(text, limit=40)
    assert "\n" not in preview
    assert preview.endswith("…")
    assert len(preview) == 40


def test_idle_countdown_pauses_while_busy(monkeypatch):
    monkeypatch.setenv("IDLE_TIMEOUT_S", "30")
    status = WorkerStatus()
    status.last_idle_at = time.monotonic() - 10
    status.state = "idle"
    left = status.idle_left_s()
    assert left is not None
    assert 15 <= left <= 30

    parsed = {"action": "ocr", "image": "https://example.com/x.png", "request_id": "r1"}
    status.job_begin(parsed, job_id="job-1")
    assert status.state == "busy"
    assert status.idle_left_s() is None


def test_job_end_counts_images_and_logs_preview(caplog):
    status = WorkerStatus()
    parsed = {
        "action": "ocr",
        "image": "https://example.com/doc.png",
        "request_id": "art-2",
        "output_format": "plain",
    }
    seq = status.job_begin(parsed)
    visual = {"kind": "image", "page_count": 1, "width": 482, "height": 284}
    response = {
        "success": True,
        "timing": {"ocr_ms": 640.2, "total_ms": 720.8},
        "output": {"text": "Article 2. – L'indemnisation"},
        "error": None,
    }
    with caplog.at_level(logging.INFO, logger="ocr"):
        status.job_end(parsed, response, seq=seq, visual=visual)
    assert status.images_processed == 1
    assert status.pages_processed == 1
    assert status.state == "idle"
    assert status.idle_left_s() is not None
    joined = "\n".join(caplog.messages)
    assert "JOB done" in joined
    assert "Article 2." in joined
    assert "idle_left=" in joined
    assert "images=1" in joined


def test_reset_for_tests_replaces_singleton():
    ws.status.jobs_total = 9
    ws.reset_for_tests()
    assert ws.status.jobs_total == 0
    assert ws.status.state == "booting"
