"""Training of the three regression models.

All models use the same features, the same training period (Sep-Dec 2022) and
the same random seed, so the comparison is fair. January 2023 (validation) is
used only for LightGBM early stopping and for model selection.
"""
import json
import time
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_engineering import FEATURES

MODEL_NAMES = ["Linear Regression", "Random Forest", "LightGBM"]
MODEL_FILES = {
    "Linear Regression": config.MODELS_DIR / "linear_regression.joblib",
    "Random Forest": config.MODELS_DIR / "random_forest.joblib",
    "LightGBM": config.MODELS_DIR / "lightgbm.joblib",
}
TRAINING_LOG = config.MODELS_DIR / "training_log.json"


def get_xy(feats, split):
    part = feats[feats["split"] == split]
    return part[FEATURES].astype("float32"), part[config.TARGET].astype("float32").values


def build_model(name):
    """Create an untrained model with its hyperparameters."""
    if name == "Linear Regression":
        # Scaling is fitted on training data only (inside the pipeline)
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "Random Forest":
        # Size limits keep memory and the saved file manageable on a laptop
        return RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=20, max_features=0.5,
            max_samples=0.3, n_jobs=-1, random_state=config.RANDOM_SEED)
    if name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=config.RANDOM_SEED, n_jobs=-1, verbose=-1)
    raise ValueError(f"Unknown model: {name}")


def train_model(name, feats):
    """Train one model on the training period. Returns (model, info dict)."""
    X_train, y_train = get_xy(feats, "train")
    model = build_model(name)
    start = time.time()
    if name == "LightGBM":
        X_val, y_val = get_xy(feats, "val")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        extra = {"best_iteration": int(model.best_iteration_)}
    else:
        model.fit(X_train, y_train)
        extra = {}
    info = {"model": name, "train_rows": int(len(y_train)), "n_features": len(FEATURES),
            "training_seconds": round(time.time() - start, 1),
            "trained_at": datetime.now().isoformat(timespec="seconds"), **extra}
    return model, info


def save_model(name, model):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILES[name], compress=3)


def load_model(name):
    """Load a saved model and check its features match the current pipeline."""
    path = MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{name} has not been trained yet. Run: python train.py")
    bundle = joblib.load(path)
    if bundle["features"] != FEATURES:
        raise ValueError(f"{name} was trained with different features. Re-run: python train.py --force")
    return bundle["model"]


def is_trained(name):
    return MODEL_FILES[name].exists()


def read_training_log():
    return json.loads(TRAINING_LOG.read_text()) if TRAINING_LOG.exists() else {}


def train_all(feats, names=None, force=False, log=print):
    """Train the requested models, skipping ones already saved unless force=True."""
    names = names or MODEL_NAMES
    history = read_training_log()
    for name in names:
        if is_trained(name) and not force:
            log(f"[skip] {name} already trained (use --force to retrain)")
            continue
        log(f"[train] {name} ...")
        np.random.seed(config.RANDOM_SEED)
        model, info = train_model(name, feats)
        save_model(name, model)
        info["file_size_mb"] = round(MODEL_FILES[name].stat().st_size / 1e6, 1)
        history[name] = info
        TRAINING_LOG.write_text(json.dumps(history, indent=2))
        log(f"[done] {name}: {info['training_seconds']} s, {info['file_size_mb']} MB")
    return history
