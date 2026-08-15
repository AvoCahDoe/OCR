# OCR worker lab (dev only)

Local Gradio UI for the serverless worker: **health** and **ocr**.

Not baked into the Docker image. Do not deploy this.

## Run

```bash
cd testing_interface
python -m venv .venv
# Windows: .venv\Scripts\activate
# bash:    source .venv/bin/activate
pip install -r requirements.txt
set RUNPOD_API_KEY=your_key   # optional; you can also paste it in the UI
python app.py
```

Open http://127.0.0.1:7860

Endpoint id defaults to `7ltawf1fgpzchm`. The API key stays in this process only.

Use **`/run`** for the first health/OCR call — a cold start (image pull + weight load) can exceed `/runsync`’s wait. Upload a file instead of a remote URL so the worker does not fetch it.

## What to try first

1. **Health ping** — GPU + model-load check.
2. **Load sample PNG** then **Run OCR**, or drop your own scan.
3. Switch `output_format` to `markdown` or `layout_json` for those fields.
