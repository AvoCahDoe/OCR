# RunPod OCR worker

Serverless worker for **PaddleOCR-VL-1.6** (vision OCR). One action: OCR. Optional health ping.

`schema_version`: **1.0**

**Image:** `ghcr.io/avocahdoe/ocr-worker:1.0`

## Architecture

```
Request → handler (action) → PaddleOCR-VL-1.6 → text
                                              → markdown / layout if requested
                                              → versioned JSON + timing + estimated cost
```

| Action | Input | Behavior |
| --- | --- | --- |
| `ocr` (default) | `image` | PaddleOCR-VL only |
| `health` | — | GPU + model-load check (warmup ping) |

## Input

```json
{
  "input": {
    "action": "ocr",
    "image": "<https URL or base64 / data URI>",
    "output_format": "plain",
    "lang": "en",
    "request_id": "optional-client-id"
  }
}
```

- `image`: JPEG / PNG / WEBP / TIFF / BMP, or PDF (first `MAX_PDF_PAGES`, default 5). Max **20 MB**. Longest side downscaled to **2560**.
- `output_format`: `plain` | `markdown` | `layout_json`.
- `lang`: accepted for forward compatibility; VL-1.6 is already multilingual (109 languages) and does not take a lang switch.
- `request_id`: worker-local idempotency TTL (default 600s). Not shared across workers.

## Output

```json
{
  "schema_version": "1.0",
  "success": true,
  "request_id": "optional-client-id",
  "worker_id": "<RUNPOD_POD_ID>",
  "output": {
    "text": "...",
    "markdown": null,
    "layout": null
  },
  "timing": { "ocr_ms": 123.4, "total_ms": 130.0 },
  "cost": {
    "gpu_type": "A5000",
    "price_per_sec": 0.0001917,
    "billed_seconds": 1,
    "estimated_cost_usd": 0.0001917
  },
  "model_versions": {
    "ocr_model": "PaddlePaddle/PaddleOCR-VL-1.6"
  },
  "warning": null,
  "error": null
}
```

`text` is always the plain OCR string. `markdown` / `layout` are filled only when `output_format` is `markdown` or `layout_json`.

`cost.estimated_cost_usd` is **not a bill**. RunPod charges wall-clock worker seconds (including idle timeout after the job), which this figure does not include. `billed_seconds` is `ceil(total_ms / 1000)`.

## Local tests (no GPU)

Windows / CPU is enough for unit tests and handler routing.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# bash:    source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

RunPod SDK local mode (skips weight load):

```bash
export SKIP_MODEL_LOAD=1   # Windows cmd: set SKIP_MODEL_LOAD=1
python src/handler.py --test_input "{\"input\": {\"action\": \"health\"}}"
```

Or drop `test_input.json` next to the handler invocation:

```bash
SKIP_MODEL_LOAD=1 PYTHONPATH=src python src/handler.py
```

On **Windows**, `runpod` imports `fcntl` (Linux-only), so the handler runs directly and prints JSON. On Linux/WSL/Docker, `runpod.serverless.start` is used (including `--rp_serve_api`).

Local API (Linux/WSL; still no real models unless CUDA + weights are present):

```bash
SKIP_MODEL_LOAD=1 PYTHONPATH=src python src/handler.py --rp_serve_api --rp_api_port 8000
curl -s -X POST http://localhost:8000/runsync -H "Content-Type: application/json" -d @examples/runsync_health.json
```

`/run` is async only on RunPod; locally use `/runsync`.

## Cold starts

Two different costs get conflated:

1. **Image pull** — first time a RunPod host runs this tag. Subsequent workers on that host reuse the cached layers.
2. **Model load** — Python reading weight files from disk into VRAM (Paddle init). This happens on every **new worker process**, not every request.

The image is **slim** (GPU wheels + handler). On a new worker, PaddleOCR-VL **downloads at runtime** into `/models/paddleocr` (container disk), then loads to GPU. Later requests on that process reuse VRAM. A new host repeats the download.

- `warmup()` runs in a **daemon thread** after the handler registers, not at import and not inside `handler()`.
- Optional: point `PADDLE_MODEL_DIR` at a network volume to persist caches across workers. Not required.

| Lever | What to set | When |
| --- | --- | --- |
| Min workers ≥ 1 | endpoint `workersMin` | Steady traffic; pays idle GPU, zero cold starts on that worker |
| Idle timeout | 5–10s+ | Bursty-but-frequent; keep the process warm between bursts |
| FlashBoot | enable on the endpoint if your tier supports it | Snapshots a warmed worker so later cold starts skip much of init |

## Docker build

linux/amd64 + NVIDIA Container Toolkit. Do **not** expect Windows-native GPU inference. Paddle GPU wheels are Linux-only.

```bash
docker build --platform linux/amd64 -t ghcr.io/avocahdoe/ocr-worker:1.0 .
```

Pinned stack:

| Component | Version |
| --- | --- |
| CUDA base | `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04` |
| Python | 3.11 |
| PaddlePaddle | 3.2.1 GPU (cu126) |
| PaddleOCR | 3.6.0 `[doc-parser]` |

Push to **public GHCR**:

```bash
echo $GITHUB_PAT | docker login ghcr.io -u avocahdoe --password-stdin
docker push ghcr.io/avocahdoe/ocr-worker:1.0
```

Optional offline populate: [`scripts/download_models.py`](scripts/download_models.py) into `/models` or a volume.

## RunPod endpoint

Queue-based serverless worker (this image’s `CMD` already calls `runpod.serverless.start`).

**GPU:** 24GB class. Allow multiple types — A5000 stock is often tight. Use **A5000 / L4 / RTX 3090 / MIG 24GB** (`$0.69/hr` = `$0.0001917/s`). Paddle uses ~90% of GPU memory (`FLAGS_fraction_of_gpu_memory_to_use=0.90`).

**Workers:** `concurrency = 1`. Dev: `min workers = 0`. App-facing: `min workers = 1` to avoid cold starts. Idle timeout 5–10s. Enable **FlashBoot**. Container disk **~30GB** (runtime weight cache).

**Env vars** (set on the endpoint, not baked secrets):

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_TYPE` | `A5000` | Label in `cost.gpu_type` |
| `GPU_PRICE_PER_SEC` | `0.0001917` | Estimate multiplier |
| `PADDLE_MODEL_DIR` | `/models/paddleocr` | Paddle VL download cache |
| `MAX_IMAGE_BYTES` | `20971520` | 20 MB |
| `MAX_IMAGE_SIDE` | `2560` | Downscale longest side |
| `MAX_PDF_PAGES` | `5` | PDF page cap |
| `IDEMPOTENCY_TTL_S` | `600` | `request_id` cache TTL |
| `LOG_LEVEL` | `INFO` | stdout logs (one JSON `metrics` line per job) |

### Sample calls

`/runsync` (small / health):

```bash
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @examples/runsync_ocr.json
```

`/run` (async; poll or pass RunPod’s webhook):

```bash
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @examples/runsync_layout.json
```

Example payloads live in [`examples/`](examples/).

## Project layout

```
src/handler.py       RunPod entry + action router
src/ocr.py           run_ocr(image) → plain / markdown / layout
src/images.py        URL / base64 / PDF ingest
src/models.py        PaddleOCR-VL singleton
src/schema.py        v1.0 contract
scripts/download_models.py   optional offline/local weight fetch
```
