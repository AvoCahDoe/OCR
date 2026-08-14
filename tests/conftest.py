import os

# Must be set before handler/models import so unit tests never touch GPU weights.
os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("GPU_TYPE", "A5000")
os.environ.setdefault("RUNPOD_POD_ID", "test-worker")
