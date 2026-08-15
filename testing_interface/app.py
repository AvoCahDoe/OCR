"""Local Gradio lab for the RunPod OCR worker. Not baked into the image."""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ENDPOINT = os.environ.get("RUNPOD_ENDPOINT_ID", "7ltawf1fgpzchm")
API_BASE = "https://api.runpod.ai/v2"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
POLL_S = 2.0
POLL_MAX_S = 15 * 60
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})


def _new_request_id() -> str:
    return f"lab-{uuid.uuid4().hex[:12]}"


def _sample_png_path() -> str:
    img = Image.new("RGB", (640, 200), color=(248, 248, 246))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((32, 70), "OCR lab sample", fill=(20, 20, 20), font=font)
    dest = Path(os.environ.get("TMP", "/tmp")) / "ocr-lab-sample.png"
    img.save(dest, format="PNG")
    return str(dest)


def _file_to_data_uri(path: str | None) -> str | None:
    if not path:
        return None
    raw = Path(path).read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"File is {len(raw) / 1e6:.1f} MB; worker cap is 20 MB")
    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _unwrap_worker(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    inner = payload.get("output")
    if isinstance(inner, dict) and (
        "schema_version" in inner or "success" in inner or "output" in inner
    ):
        if "schema_version" in inner:
            return inner
    if "schema_version" in payload or "success" in payload:
        return payload
    return None


def _call_runpod(
    *,
    endpoint_id: str,
    api_key: str,
    mode: str,
    payload: dict[str, Any],
    progress: gr.Progress,
) -> dict[str, Any]:
    endpoint_id = (endpoint_id or "").strip()
    api_key = (api_key or "").strip()
    if not endpoint_id:
        raise ValueError("Endpoint id is required")
    if not api_key:
        raise ValueError("RunPod API key is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps({"input": payload})
    started = time.perf_counter()

    if mode == "/runsync":
        progress(0.2, desc="runsync")
        res = requests.post(
            f"{API_BASE}/{endpoint_id}/runsync",
            headers=headers,
            data=body,
            timeout=120,
        )
        raw = res.json() if res.content else {}
        if not res.ok:
            raise RuntimeError(raw.get("error") or f"RunPod HTTP {res.status_code}")
        if raw.get("status") in ("IN_QUEUE", "IN_PROGRESS") and raw.get("id"):
            return _poll_job(endpoint_id, api_key, raw["id"], started, progress, raw)
        return _pack(raw, started)

    progress(0.1, desc="queue job")
    res = requests.post(
        f"{API_BASE}/{endpoint_id}/run",
        headers=headers,
        data=body,
        timeout=60,
    )
    raw = res.json() if res.content else {}
    if not res.ok:
        raise RuntimeError(raw.get("error") or f"RunPod HTTP {res.status_code}")
    job_id = raw.get("id")
    if not job_id:
        raise RuntimeError("RunPod /run did not return a job id")
    return _poll_job(endpoint_id, api_key, job_id, started, progress, raw)


def _poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    started: float,
    progress: gr.Progress,
    last: dict[str, Any],
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + POLL_MAX_S
    while time.time() < deadline:
        status = str(last.get("status") or "").upper()
        elapsed = time.time() - (deadline - POLL_MAX_S)
        progress(min(0.95, 0.15 + elapsed / POLL_MAX_S), desc=status or "polling")
        if status in TERMINAL:
            return _pack(last, started)
        time.sleep(POLL_S)
        res = requests.get(
            f"{API_BASE}/{endpoint_id}/status/{job_id}",
            headers=headers,
            timeout=60,
        )
        last = res.json() if res.content else {}
        if not res.ok:
            raise RuntimeError(last.get("error") or f"Status HTTP {res.status_code}")
    raise TimeoutError(f"Timed out polling job {job_id} (last status {last.get('status')})")


def _pack(raw: dict[str, Any], started: float) -> dict[str, Any]:
    worker = _unwrap_worker(raw.get("output") if isinstance(raw, dict) else None)
    if worker is None:
        worker = _unwrap_worker(raw)
    return {
        "raw": raw,
        "worker": worker,
        "job_id": raw.get("id") if isinstance(raw, dict) else None,
        "status": (raw.get("status") if isinstance(raw, dict) else None) or "UNKNOWN",
        "client_ms": round((time.perf_counter() - started) * 1000),
        "delay_ms": raw.get("delayTime") if isinstance(raw, dict) else None,
        "execution_ms": raw.get("executionTime") if isinstance(raw, dict) else None,
    }


def _format_result(result: dict[str, Any]) -> tuple[str, str, Any, str]:
    worker = result.get("worker") or {}
    output = worker.get("output") if isinstance(worker, dict) else None
    output = output if isinstance(output, dict) else {}
    text = output.get("text") or ""
    markdown = output.get("markdown") or text or ""
    layout = output.get("layout")
    error = worker.get("error") if isinstance(worker, dict) else None
    warning = worker.get("warning") if isinstance(worker, dict) else None
    timing = worker.get("timing") if isinstance(worker, dict) else None
    cost = worker.get("cost") if isinstance(worker, dict) else None
    summary = {
        "status": result.get("status"),
        "success": worker.get("success") if isinstance(worker, dict) else None,
        "job_id": result.get("job_id"),
        "worker_id": worker.get("worker_id") if isinstance(worker, dict) else None,
        "client_ms": result.get("client_ms"),
        "queue_ms": result.get("delay_ms"),
        "execution_ms": result.get("execution_ms"),
        "timing": timing,
        "cost": cost,
        "gpu_available": output.get("gpu_available"),
        "ocr_loaded": output.get("ocr_loaded"),
        "warning": warning,
        "error": error,
    }
    return text, markdown, layout, json.dumps(summary, indent=2, default=str)


def run_health(endpoint_id: str, api_key: str, mode: str, progress=gr.Progress()):
    result = _call_runpod(
        endpoint_id=endpoint_id,
        api_key=api_key,
        mode=mode,
        payload={"action": "health", "request_id": _new_request_id()},
        progress=progress,
    )
    _text, _md, _layout, summary = _format_result(result)
    return summary, result.get("raw")


def run_ocr(
    endpoint_id: str,
    api_key: str,
    mode: str,
    image_path: str | None,
    image_url: str,
    output_format: str,
    progress=gr.Progress(),
):
    image = _file_to_data_uri(image_path) or (image_url or "").strip()
    if not image:
        raise ValueError("Upload an image or paste a URL. Upload is faster (worker skips the fetch).")
    result = _call_runpod(
        endpoint_id=endpoint_id,
        api_key=api_key,
        mode=mode,
        payload={
            "action": "ocr",
            "image": image,
            "output_format": output_format,
            "request_id": _new_request_id(),
        },
        progress=progress,
    )
    text, markdown, layout, summary = _format_result(result)
    return text, markdown, layout, summary, result.get("raw")


def _theme() -> gr.themes.Base:
    return gr.themes.Soft(primary_hue="slate", neutral_hue="stone")


with gr.Blocks(title="OCR worker lab", theme=_theme()) as demo:
    gr.Markdown(
        """
# OCR worker lab
Local Gradio UI for the RunPod PaddleOCR-VL worker. Not part of the Docker image.

Prefer **upload** over a remote URL — the worker otherwise downloads the file itself
(the Paddle demo host can take ~25s). Use **`/run` + poll** for the first call after a
cold start; `/runsync` is fine once a worker is warm.
"""
    )
    with gr.Accordion("Connection", open=True):
        with gr.Row():
            endpoint = gr.Textbox(
                label="Endpoint id",
                value=DEFAULT_ENDPOINT,
                info="Default is the live ocr-vl endpoint",
            )
            api_key = gr.Textbox(
                label="RunPod API key",
                value=os.environ.get("RUNPOD_API_KEY", ""),
                type="password",
                info="Or set RUNPOD_API_KEY. Never committed.",
            )
            mode = gr.Radio(
                choices=["/run", "/runsync"],
                value="/run",
                label="Call mode",
            )
        health_btn = gr.Button("Health ping", variant="secondary")

    with gr.Row():
        with gr.Column():
            image = gr.Image(
                label="Image or PDF page scan",
                type="filepath",
                sources=["upload", "clipboard"],
            )
            image_url = gr.Textbox(
                label="Image URL (optional)",
                placeholder="https://…  — slower; worker fetches this",
            )
            output_format = gr.Radio(
                choices=["plain", "markdown", "layout_json"],
                value="plain",
                label="output_format",
            )
            with gr.Row():
                sample_btn = gr.Button("Load sample PNG")
                ocr_btn = gr.Button("Run OCR", variant="primary")
        with gr.Column():
            text_out = gr.Textbox(label="text", lines=12)
            md_out = gr.Markdown(label="markdown")
            layout_out = gr.JSON(label="layout")
            summary_out = gr.Code(label="timing / health", language="json")
            raw_out = gr.JSON(label="raw RunPod envelope")

    sample_btn.click(fn=_sample_png_path, outputs=image)
    health_btn.click(
        fn=run_health,
        inputs=[endpoint, api_key, mode],
        outputs=[summary_out, raw_out],
    )
    ocr_btn.click(
        fn=run_ocr,
        inputs=[endpoint, api_key, mode, image, image_url, output_format],
        outputs=[text_out, md_out, layout_out, summary_out, raw_out],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
