"""Keep PaddleOCR-VL in eager mode and make int(Tensor) safe.

Layout detection may flip the process into static graph. The native VL
preprocessor then does `int(tensor)` and Paddle raises. Call these helpers
only after `paddleocr` has been imported.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def ensure_dynamic_graph() -> None:
    try:
        import paddle
    except ImportError:
        return
    if hasattr(paddle, "in_dynamic_mode") and paddle.in_dynamic_mode():
        return
    if hasattr(paddle, "disable_static"):
        paddle.disable_static()


def apply_paddleocr_compat() -> None:
    """Idempotent. Safe to call from the loader after `import paddleocr`."""
    global _PATCHED
    ensure_dynamic_graph()
    if _PATCHED:
        return
    _patch_tensor_int()
    _patch_vl_python_int_sites()
    _PATCHED = True
    logger.info("Applied PaddleOCR-VL dynamic-graph compatibility patches")


def wrap_vl_rec_model(pipeline: Any) -> None:
    """Force eager mode each time the VL recognizer runs."""
    inner = pipeline
    for attr in ("paddlex_pipeline", "_pipeline"):
        inner = getattr(inner, attr, inner)
    vl = getattr(inner, "vl_rec_model", None)
    if vl is None or getattr(vl, "_ocr_dynamic_wrapped", False):
        return
    for name in ("predict", "generate"):
        orig = getattr(vl, name, None)
        if orig is None or getattr(orig, "_ocr_wrapped", False):
            continue
        setattr(vl, name, _wrap_before(orig, ensure_dynamic_graph))
    vl._ocr_dynamic_wrapped = True


def as_python_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return int(item())
        except Exception:
            pass
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        arr = numpy()
        return int(arr.reshape(-1)[0])
    return int(value)


def _wrap_before(orig, before):
    def wrapped(*args, **kwargs):
        before()
        return orig(*args, **kwargs)

    wrapped._ocr_wrapped = True  # type: ignore[attr-defined]
    return wrapped


def _patch_tensor_int() -> None:
    import paddle

    tensor_cls = getattr(paddle, "Tensor", None)
    if tensor_cls is None or getattr(tensor_cls, "_ocr_int_patched", False):
        return

    def _int(self: Any) -> int:
        ensure_dynamic_graph()
        return as_python_int(self)

    try:
        tensor_cls.__int__ = _int  # type: ignore[method-assign]
        tensor_cls._ocr_int_patched = True  # type: ignore[attr-defined]
    except TypeError:
        logger.warning("Could not patch paddle.Tensor.__int__")


def _patch_vl_python_int_sites() -> None:
    try:
        from paddlex.inference.models.doc_vlm.processors.paddleocr_vl._paddleocr_vl import (  # noqa: E501
            PaddleOCRVLProcessor,
        )
        from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._projector import (
            Projector,
        )
    except Exception:
        logger.warning("Could not import PaddleX VL modules for int() patches")
        return

    if not getattr(PaddleOCRVLProcessor, "_ocr_patched", False):
        orig_pre = PaddleOCRVLProcessor.preprocess
        PaddleOCRVLProcessor.preprocess = _wrap_before(orig_pre, ensure_dynamic_graph)
        PaddleOCRVLProcessor._ocr_patched = True

    if not getattr(Projector, "_ocr_patched", False):
        orig_fwd = Projector.forward
        Projector.forward = _wrap_before(orig_fwd, ensure_dynamic_graph)
        Projector._ocr_patched = True
