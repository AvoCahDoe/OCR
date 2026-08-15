"""Developer-facing worker logs: state, I/O preview, idle countdown, counters."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

from config import OCR_MODEL_ID, worker_id

logger = logging.getLogger("ocr")

_URL_PREFIXES = ("http://", "https://")
_PREVIEW_LIMIT = 160
_HEARTBEAT_DEFAULT_S = 15.0
_SCALE_DOWN_WARN_S = 20.0


def idle_timeout_s() -> float:
    raw = os.environ.get("IDLE_TIMEOUT_S") or os.environ.get("RUNPOD_IDLE_TIMEOUT") or "120"
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 120.0


def heartbeat_interval_s() -> float:
    raw = os.environ.get("HEARTBEAT_S") or str(_HEARTBEAT_DEFAULT_S)
    try:
        return max(5.0, float(raw))
    except ValueError:
        return _HEARTBEAT_DEFAULT_S


def summarize_image_spec(spec: Any) -> str:
    """One-line input descriptor. Never logs raw base64."""
    if spec is None:
        return "none"
    text = str(spec).strip()
    if not text:
        return "empty"
    lowered = text.lower()
    if lowered.startswith(_URL_PREFIXES):
        return f"url:{_redact_url(text)}"
    if lowered.startswith("data:"):
        header, _, _payload = text.partition(",")
        mime = header[5:].split(";")[0] or "unknown"
        return f"data_uri mime={mime} chars={len(text)}"
    return f"base64 chars={len(text)} bytes≈{len(text) * 3 // 4}"


def preview_text(text: Any, limit: int = _PREVIEW_LIMIT) -> str:
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _redact_url(url: str, limit: int = 180) -> str:
    parsed = urlparse(url)
    host_path = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        host_path += "?…"
    if len(host_path) > limit:
        return host_path[: limit - 1] + "…"
    return host_path


def _visual_line(visual: dict[str, Any] | None) -> str:
    if not visual:
        return "none"
    kind = visual.get("kind") or "unknown"
    pages = visual.get("page_count")
    width = visual.get("width")
    height = visual.get("height")
    parts = [f"kind={kind}"]
    if width and height:
        parts.append(f"{width}x{height}")
    if pages is not None:
        parts.append(f"pages={pages}")
    return " ".join(parts)


def _page_count(visual: dict[str, Any] | None, layout: Any) -> int:
    if visual and visual.get("page_count"):
        try:
            return max(1, int(visual["page_count"]))
        except (TypeError, ValueError):
            pass
    if isinstance(layout, list) and layout:
        return len(layout)
    return 1


class WorkerStatus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "booting"
        self.started_at = time.monotonic()
        self.last_idle_at = time.monotonic()
        self.jobs_total = 0
        self.jobs_ok = 0
        self.jobs_fail = 0
        self.jobs_health = 0
        self.jobs_cached = 0
        self.images_processed = 0
        self.pages_processed = 0
        self.last_ocr_ms: float | None = None
        self.last_total_ms: float | None = None
        self.warmup_s: float | None = None
        self._warmup_started_at: float | None = None
        self._busy = False
        self._scale_warned = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def idle_left_s(self) -> float | None:
        if self._busy or self.state in {"booting", "warming", "busy"}:
            return None
        elapsed = time.monotonic() - self.last_idle_at
        return max(0.0, idle_timeout_s() - elapsed)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            left = self.idle_left_s()
            return {
                "state": self.state,
                "worker_id": worker_id(),
                "uptime_s": round(time.monotonic() - self.started_at, 1),
                "jobs_total": self.jobs_total,
                "jobs_ok": self.jobs_ok,
                "jobs_fail": self.jobs_fail,
                "jobs_health": self.jobs_health,
                "jobs_cached": self.jobs_cached,
                "images_processed": self.images_processed,
                "pages_processed": self.pages_processed,
                "last_ocr_ms": self.last_ocr_ms,
                "last_total_ms": self.last_total_ms,
                "idle_timeout_s": idle_timeout_s(),
                "idle_left_s": None if left is None else round(left, 1),
                "warmup_s": self.warmup_s,
            }

    def log_boot(self) -> None:
        logger.info(
            "[ocr] BOOT worker=%s idle_timeout_s=%s heartbeat_s=%s gpu_type=%s",
            worker_id(),
            int(idle_timeout_s()),
            int(heartbeat_interval_s()),
            os.environ.get("GPU_TYPE", "unknown"),
        )

    def mark_warming(self) -> None:
        with self._lock:
            self.state = "warming"
            self._warmup_started_at = time.monotonic()
        logger.info(
            "[ocr] state=warming cache=%s",
            os.environ.get("PADDLE_PDX_CACHE_HOME") or os.environ.get("PADDLE_MODEL_DIR"),
        )

    def mark_ready(self) -> None:
        with self._lock:
            if self._warmup_started_at is not None and self.warmup_s is None:
                self.warmup_s = round(time.monotonic() - self._warmup_started_at, 2)
            if not self._busy:
                self.state = "idle"
                self.last_idle_at = time.monotonic()
                self._scale_warned = False
        logger.info(
            "[ocr] state=ready model=%s warmup_s=%s",
            OCR_MODEL_ID,
            self.warmup_s if self.warmup_s is not None else "n/a",
        )
        self._log_idle_line()

    def mark_warmup_failed(self, message: str) -> None:
        with self._lock:
            if not self._busy:
                self.state = "idle"
                self.last_idle_at = time.monotonic()
        logger.info("[ocr] state=idle warmup_failed error=%r", preview_text(message, 200))
        self._log_idle_line()

    def job_begin(self, parsed: dict[str, Any], *, job_id: str | None = None) -> int:
        with self._lock:
            self.jobs_total += 1
            seq = self.jobs_total
            self._busy = True
            self.state = "busy"
            self._scale_warned = False
            action = parsed.get("action") or "ocr"
            if action == "health":
                self.jobs_health += 1
        logger.info(
            "[ocr] JOB start #%s action=%s request_id=%s job_id=%s format=%s lang=%s input=%s",
            seq,
            parsed.get("action"),
            parsed.get("request_id") or "-",
            job_id or "-",
            parsed.get("output_format") or "-",
            parsed.get("lang") or "-",
            summarize_image_spec(parsed.get("image")),
        )
        return seq

    def note_visual(self, visual: dict[str, Any] | None) -> None:
        logger.info("[ocr] INPUT decoded %s", _visual_line(visual))

    def job_cached(self, parsed: dict[str, Any]) -> None:
        with self._lock:
            self.jobs_cached += 1
            self._busy = False
            self.state = "idle"
            self.last_idle_at = time.monotonic()
        logger.info(
            "[ocr] JOB cache-hit request_id=%s images=%s idle_left=%s",
            parsed.get("request_id") or "-",
            self.images_processed,
            _fmt_left(self.idle_left_s()),
        )

    def job_end(
        self,
        parsed: dict[str, Any],
        response: dict[str, Any],
        *,
        seq: int | None = None,
        visual: dict[str, Any] | None = None,
    ) -> None:
        success = bool(response.get("success"))
        action = parsed.get("action") or "ocr"
        timing = response.get("timing") or {}
        ocr_ms = timing.get("ocr_ms")
        total_ms = timing.get("total_ms")
        output = response.get("output") or {}
        error = response.get("error") or {}
        pages = _page_count(visual, output.get("layout")) if action == "ocr" and success else 0

        with self._lock:
            if success:
                self.jobs_ok += 1
                if action == "ocr":
                    self.images_processed += 1
                    self.pages_processed += pages
            else:
                self.jobs_fail += 1
            self.last_ocr_ms = ocr_ms
            self.last_total_ms = total_ms
            self._busy = False
            self.state = "idle"
            self.last_idle_at = time.monotonic()
            self._scale_warned = False
            images = self.images_processed
            pages_total = self.pages_processed
            fail = self.jobs_fail

        if success and action == "ocr":
            text = output.get("text") or ""
            logger.info(
                "[ocr] JOB done #%s ok action=ocr request_id=%s ocr_ms=%s total_ms=%s "
                "pages=%s text_chars=%s preview=%r images=%s pages_total=%s idle_left=%s",
                seq or "-",
                parsed.get("request_id") or "-",
                _fmt_ms(ocr_ms),
                _fmt_ms(total_ms),
                pages,
                len(text),
                preview_text(text),
                images,
                pages_total,
                _fmt_left(self.idle_left_s()),
            )
        elif success:
            logger.info(
                "[ocr] JOB done #%s ok action=%s request_id=%s gpu=%s ocr_loaded=%s "
                "total_ms=%s images=%s idle_left=%s",
                seq or "-",
                action,
                parsed.get("request_id") or "-",
                output.get("gpu_available"),
                output.get("ocr_loaded"),
                _fmt_ms(total_ms),
                images,
                _fmt_left(self.idle_left_s()),
            )
        else:
            logger.info(
                "[ocr] JOB fail #%s action=%s request_id=%s stage=%s error=%r "
                "total_ms=%s fail=%s images=%s idle_left=%s",
                seq or "-",
                action,
                parsed.get("request_id") or "-",
                error.get("stage") or "unknown",
                preview_text(error.get("message"), 200),
                _fmt_ms(total_ms),
                fail,
                images,
                _fmt_left(self.idle_left_s()),
            )

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return
        thread = threading.Thread(
            target=self._heartbeat_loop,
            name="ocr-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread = thread
        thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()

    def _heartbeat_loop(self) -> None:
        interval = heartbeat_interval_s()
        while not self._heartbeat_stop.wait(interval):
            self._maybe_warn_scale_down()
            if self.state == "busy":
                logger.info(
                    "[ocr] HEARTBEAT state=busy jobs=%s images=%s pages=%s scale_down=paused",
                    self.jobs_total,
                    self.images_processed,
                    self.pages_processed,
                )
                continue
            self._log_idle_line()

    def _log_idle_line(self) -> None:
        snap = self.snapshot()
        logger.info(
            "[ocr] HEARTBEAT state=%s jobs=%s ok=%s fail=%s health=%s "
            "images=%s pages=%s last_ocr_ms=%s uptime=%ss idle_left=%s",
            snap["state"],
            snap["jobs_total"],
            snap["jobs_ok"],
            snap["jobs_fail"],
            snap["jobs_health"],
            snap["images_processed"],
            snap["pages_processed"],
            _fmt_ms(snap["last_ocr_ms"]),
            int(snap["uptime_s"]),
            _fmt_left(snap["idle_left_s"]),
        )

    def _maybe_warn_scale_down(self) -> None:
        left = self.idle_left_s()
        if left is None:
            self._scale_warned = False
            return
        if left <= _SCALE_DOWN_WARN_S and not self._scale_warned:
            self._scale_warned = True
            logger.info(
                "[ocr] SCALE-DOWN soon idle_left=%s images=%s jobs=%s",
                _fmt_left(left),
                self.images_processed,
                self.jobs_total,
            )
        elif left > _SCALE_DOWN_WARN_S + 5:
            self._scale_warned = False


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "n/a"


def _fmt_left(value: float | None) -> str:
    if value is None:
        return "paused"
    return f"{int(round(value))}s"


status = WorkerStatus()


def reset_for_tests() -> None:
    status.stop_heartbeat()
    status.__init__()
