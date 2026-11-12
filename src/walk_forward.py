"""Walk-forward (expanding-window) forecasting for the dataset period.

For each month M in config.WALK_FORWARD_MONTHS, all three next-hour models are
trained on every row BEFORE the first hour of M and then predict M. Every
prediction is therefore made for data the model has never seen, which lets
users pick any date from October 2022 to February 2023 and compare the
prediction with the actual value.

LightGBM's early stopping uses the last 7 days of each fold's own training
window, so no later data influences a fold.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.evaluate_models import predict, regression_metrics
from src.feature_engineering import FEATURES, load_features
from src.train_models import MODEL_NAMES, build_model

WF_DIR = config.ARTIFACTS_DIR / "walk_forward"
WF_PREDICTIONS = WF_DIR / "predictions.parquet"
WF_MONTHLY = WF_DIR / "monthly_metrics.csv"


def _fit(name, X, y, times):
    model = build_model(name)
    if name == "LightGBM":
        stop_start = times.max() - pd.Timedelta(days=7)
        inner = (times <= stop_start).values
        model.fit(X[inner], y[inner], eval_set=[(X[~inner], y[~inner])], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X, y)
    return model


def run_walk_forward(feats=None, log=print):
    feats = load_features() if feats is None else feats
    X_all = feats[FEATURES].astype("float32")
    y_all = feats[config.TARGET].astype("float32").values
    parts, rows = [], []
    for month in config.WALK_FORWARD_MONTHS:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        train = (feats.time < start).values
        test = ((feats.time >= start) & (feats.time < end)).values
        out = feats.loc[test, ["time", "zone", config.TARGET]].rename(
            columns={config.TARGET: "actual"}).reset_index(drop=True)
        for name in MODEL_NAMES:
            t0 = time.time()
            model = _fit(name, X_all[train], y_all[train], feats.time[train])
            out[name] = predict(model, X_all[test]).astype("float32")
            m = regression_metrics(out.actual, out[name])
            rows.append({"month": month, "model": name, "train_rows": int(train.sum()),
                         "test_rows": int(test.sum()), **m,
                         "seconds": round(time.time() - t0, 1)})
            log(f"[walk-forward] {month} {name}: RMSE {m['RMSE']:.1f}  ({rows[-1]['seconds']} s)")
        parts.append(out)

    preds = pd.concat(parts, ignore_index=True)
    preds["zone"] = preds["zone"].astype("int32")
    monthly = pd.DataFrame(rows)
    overall = [{"month": "Oct-Feb (all)", "model": n, "train_rows": np.nan,
                "test_rows": len(preds), **regression_metrics(preds.actual, preds[n]), "seconds": np.nan}
               for n in MODEL_NAMES]
    monthly = pd.concat([monthly, pd.DataFrame(overall)], ignore_index=True)
    WF_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(WF_PREDICTIONS, index=False)
    monthly.to_csv(WF_MONTHLY, index=False)
    return preds, monthly


def load_walk_forward():
    if not WF_PREDICTIONS.exists():
        raise FileNotFoundError("Walk-forward predictions not found. Run: python train.py")
    return pd.read_parquet(WF_PREDICTIONS), pd.read_csv(WF_MONTHLY)
