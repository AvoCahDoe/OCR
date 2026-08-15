from __future__ import annotations

import weights as weights_mod
from weights import resolve_paddle_dir


def test_paddle_creates_cache_when_empty(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    monkeypatch.setenv("PADDLE_MODEL_DIR", str(empty))
    monkeypatch.setattr(weights_mod, "VOLUME_ROOT", tmp_path / "no-vol")
    monkeypatch.setattr(weights_mod, "BAKED_PADDLE_DIR", tmp_path / "no-bake")
    resolved = resolve_paddle_dir()
    assert resolved == empty.resolve()
    assert empty.is_dir()


def test_paddle_accepts_baked_sentinel(tmp_path, monkeypatch):
    baked = tmp_path / "paddle"
    baked.mkdir()
    (baked / ".baked").write_text("ok", encoding="utf-8")
    monkeypatch.setenv("PADDLE_MODEL_DIR", str(baked))
    assert resolve_paddle_dir() == baked.resolve()


def test_paddle_prefers_network_volume(tmp_path, monkeypatch):
    monkeypatch.delenv("PADDLE_MODEL_DIR", raising=False)
    vol_root = tmp_path / "runpod-volume"
    paddle = vol_root / "paddleocr"
    paddle.mkdir(parents=True)
    (paddle / ".baked").write_text("ok", encoding="utf-8")
    monkeypatch.setattr(weights_mod, "VOLUME_ROOT", vol_root)
    monkeypatch.setattr(weights_mod, "BAKED_PADDLE_DIR", tmp_path / "missing-baked")
    monkeypatch.setattr(weights_mod, "DEFAULT_PADDLE_CACHE", tmp_path / "default-cache")
    assert resolve_paddle_dir() == paddle.resolve()


def test_paddle_empty_cache_uses_mounted_volume(tmp_path, monkeypatch):
    """Dockerfile default PADDLE_MODEL_DIR must not steal an empty mounted volume."""
    env_cache = tmp_path / "models-paddleocr"
    vol_root = tmp_path / "runpod-volume"
    vol_root.mkdir()
    monkeypatch.setenv("PADDLE_MODEL_DIR", str(env_cache))
    monkeypatch.setattr(weights_mod, "VOLUME_ROOT", vol_root)
    monkeypatch.setattr(weights_mod, "BAKED_PADDLE_DIR", tmp_path / "no-bake")
    monkeypatch.setattr(weights_mod, "DEFAULT_PADDLE_CACHE", tmp_path / "default-cache")
    resolved = resolve_paddle_dir()
    assert resolved == (vol_root / "paddleocr").resolve()
    assert resolved.is_dir()
    assert not env_cache.exists()
