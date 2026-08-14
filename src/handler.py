"""RunPod serverless handler: PaddleOCR-VL only."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Visible even if later imports fail; Kaniko/venv crashes print nothing.
print("ocr-worker python start", sys.version.replace("\n", " "), flush=True)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import IDEMPOTENCY_TTL_S, SCHEMA_VERSION, skip_model_load  # noqa: E402
from images import cleanup_visual, load_visual  # noqa: E402
from models import gpu_available, ocr_load_error, ocr_loaded, warmup  # noqa: E402
from ocr import run_ocr  # noqa: E402
from cost import estimate_cost  # noqa: E402
from schema import InputError, build_response, error_response, parse_input  # noqa: E402
from timing import timed_ms  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ocr_worker")

_cache_lock = threading.Lock()
_response_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_get(request_id: str | None) -> dict[str, Any] | None:
    if not request_id:
        return None
    now = time.time()
    with _cache_lock:
        hit = _response_cache.get(request_id)
        if not hit:
            return None
        ts, payload = hit
        if now - ts > IDEMPOTENCY_TTL_S:
            _response_cache.pop(request_id, None)
            return None
        return payload


def _cache_set(request_id: str | None, payload: dict[str, Any]) -> None:
    if not request_id or not payload.get("success"):
        return
    with _cache_lock:
        now = time.time()
        stale = [k for k, (ts, _) in _response_cache.items() if now - ts > IDEMPOTENCY_TTL_S]
        for key in stale:
            _response_cache.pop(key, None)
        _response_cache[request_id] = (now, payload)


def _log_metrics(response: dict[str, Any]) -> None:
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "request_id": response.get("request_id"),
        "worker_id": response.get("worker_id"),
        "success": response.get("success"),
        "timing": response.get("timing"),
        "cost": response.get("cost"),
        "error": response.get("error"),
        "warning": response.get("warning"),
    }
    logger.info("metrics %s", json.dumps(metrics, default=str))


def handler(job: dict[str, Any]) -> dict[str, Any]:
    request_id: str | None = None
    try:
        parsed = parse_input(job if isinstance(job, dict) else {})
        request_id = parsed["request_id"]
    except InputError as exc:
        response = error_response(
            stage=exc.stage,
            message=exc.message,
            request_id=_peek_request_id(job),
            timing={"ocr_ms": None, "total_ms": 0.0},
        )
        _log_metrics(response)
        return response

    cached = _cache_get(request_id)
    if cached is not None:
        logger.info("idempotent hit request_id=%s", request_id)
        return cached

    with timed_ms() as total_box:
        try:
            action = parsed["action"]
            if action == "health":
                response = _handle_health(request_id, total_box)
            else:
                response = _handle_ocr(parsed, total_box)
        except InputError as exc:
            response = error_response(
                stage=exc.stage,
                message=exc.message,
                request_id=request_id,
                timing={"ocr_ms": None, "total_ms": total_box[0]},
            )
        except Exception as exc:
            logger.exception("unhandled handler error")
            response = error_response(
                stage="handler",
                message=str(exc),
                request_id=request_id,
                timing={"ocr_ms": None, "total_ms": total_box[0]},
            )

    response.setdefault("timing", {})["total_ms"] = round(total_box[0], 3)
    response["cost"] = estimate_cost(total_box[0])
    _cache_set(request_id, response)
    _log_metrics(response)
    return response


def _peek_request_id(job: Any) -> str | None:
    if not isinstance(job, dict):
        return None
    raw = job.get("input", job)
    if not isinstance(raw, dict) or raw.get("request_id") is None:
        return None
    value = str(raw["request_id"]).strip()
    return value or None


def _handle_health(request_id: str | None, total_box: list[float]) -> dict[str, Any]:
    gpu_ok = gpu_available()
    ocr_ok = ocr_loaded()
    if not skip_model_load() and not ocr_ok:
        try:
            from models import get_ocr_pipeline

            get_ocr_pipeline()
            ocr_ok = ocr_loaded()
        except Exception:
            ocr_ok = False
    healthy = gpu_ok and ocr_ok
    warning = None
    if not ocr_ok:
        warning = {"stage": "health", "message": ocr_load_error() or "PaddleOCR-VL not loaded"}
    elif not gpu_ok:
        warning = {"stage": "health", "message": "GPU not available"}
    return build_response(
        success=healthy,
        request_id=request_id,
        output={
            "text": None,
            "markdown": None,
            "layout": None,
            "gpu_available": gpu_ok,
            "ocr_loaded": ocr_ok,
        },
        timing={"ocr_ms": None, "total_ms": total_box[0]},
        warning=warning,
    )


def _handle_ocr(parsed: dict[str, Any], total_box: list[float]) -> dict[str, Any]:
    visual = None
    ocr_ms: float | None = None
    try:
        visual = load_visual(parsed["image"])
        with timed_ms() as ocr_box:
            ocr_result = run_ocr(visual)
        ocr_ms = ocr_box[0]
    except InputError:
        raise
    except Exception as exc:
        logger.exception("OCR stage failed")
        return error_response(
            stage="ocr",
            message=str(exc),
            request_id=parsed["request_id"],
            timing={"ocr_ms": ocr_ms, "total_ms": total_box[0]},
        )
    finally:
        if visual is not None:
            cleanup_visual(visual)

    plain = ocr_result.get("plain") or ""
    markdown = ocr_result.get("markdown") or None
    layout = ocr_result.get("layout")
    fmt = parsed["output_format"]

    return build_response(
        success=True,
        request_id=parsed["request_id"],
        output={
            "text": plain,
            "markdown": markdown if fmt == "markdown" else None,
            "layout": layout if fmt == "layout_json" else None,
        },
        timing={"ocr_ms": ocr_ms, "total_ms": total_box[0]},
    )


# Do NOT warmup at import. RunPod kills the container if serverless.start()
# is delayed (image pull + Paddle download is minutes). Load in a daemon
# thread after the handler registers, or lazily on the first job.


def _local_test_payload() -> dict[str, Any] | None:
    if "--test_input" in sys.argv:
        idx = sys.argv.index("--test_input")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("--test_input requires a JSON argument")
        return json.loads(sys.argv[idx + 1])
    for candidate in (
        Path.cwd() / "test_input.json",
        SRC_DIR.parent / "test_input.json",
    ):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


if __name__ == "__main__":
    # runpod.serverless imports fcntl (Linux-only). On Windows, invoke the handler directly.
    if os.name == "nt" or os.environ.get("LOCAL_HANDLER", "").lower() in ("1", "true", "yes"):
        payload = _local_test_payload()
        if payload is None:
            raise SystemExit("Provide --test_input JSON or test_input.json for local runs")
        print(json.dumps(handler(payload), indent=2, default=str))
    else:
        import runpod

        if not skip_model_load():
            threading.Thread(target=warmup, daemon=True, name="model-warmup").start()
        runpod.serverless.start({"handler": handler})
