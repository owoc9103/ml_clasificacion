"""Tests del registro multi-modelo."""

import numpy as np
import pandas as pd

from quantfund.models.registry import MODEL_NAMES, build_model, predict_positive_proba


def test_all_models_build():
    for name in MODEL_NAMES:
        model = build_model(name)
        assert hasattr(model, "fit")


def test_predict_proba_on_toy_data():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + X["b"] > 0).astype(int)
    model = build_model("logistic_regression")
    model.fit(X, y)
    p = predict_positive_proba(model, X.iloc[:5])
    assert len(p) == 5
    assert np.all((p >= 0) & (p <= 1))
