from __future__ import annotations

import sys
from types import SimpleNamespace

import models


def test_gpu_available_does_not_import_paddle(monkeypatch):
    monkeypatch.setattr(models.shutil, "which", lambda _name: None)
    sys.modules.pop("paddle", None)
    assert models.gpu_available() is False
    assert "paddle" not in sys.modules


def test_gpu_available_nvidia_smi(monkeypatch):
    monkeypatch.setattr(models.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        models.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=b"GPU 0: NVIDIA L4\n"),
    )
    sys.modules.pop("paddle", None)
    assert models.gpu_available() is True
    assert "paddle" not in sys.modules
