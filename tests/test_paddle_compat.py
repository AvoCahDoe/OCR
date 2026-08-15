from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from paddle_compat import as_python_int, wrap_vl_rec_model


def test_as_python_int_plain():
    assert as_python_int(7) == 7


def test_as_python_int_numpy_scalar():
    assert as_python_int(np.array(4)) == 4


def test_as_python_int_vector_uses_first():
    assert as_python_int(SimpleNamespace(numpy=lambda: np.array([3, 9]))) == 3


def test_wrap_vl_rec_model_calls_predict():
    called = []

    def predict(x):
        called.append(x)
        return x

    pipeline = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(
            _pipeline=SimpleNamespace(vl_rec_model=SimpleNamespace(predict=predict))
        )
    )
    wrap_vl_rec_model(pipeline)
    wrap_vl_rec_model(pipeline)
    out = pipeline.paddlex_pipeline._pipeline.vl_rec_model.predict("img")
    assert out == "img"
    assert called == ["img"]
