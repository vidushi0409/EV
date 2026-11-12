"""Model save/load and metric tests. Run with:  python -m pytest -q"""
import numpy as np
import pytest

from src.evaluate_models import regression_metrics
from src.train_models import MODEL_NAMES, is_trained


def test_metrics_on_known_values():
    m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["MAE"] == pytest.approx(2 / 3)
    assert m["MSE"] == pytest.approx(4 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(4 / 3))
    assert m["R2"] == pytest.approx(1 - 4 / 2)


@pytest.mark.skipif(not all(is_trained(n) for n in MODEL_NAMES), reason="models not trained")
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_saved_model_loads_and_predicts(name):
    from src.feature_engineering import FEATURES, load_features
    from src.evaluate_models import predict
    from src.train_models import load_model
    model = load_model(name)
    X = load_features().query("split == 'test'")[FEATURES].head(100).astype("float32")
    pred = predict(model, X)
    assert pred.shape == (100,) and np.isfinite(pred).all() and (pred >= 0).all()
