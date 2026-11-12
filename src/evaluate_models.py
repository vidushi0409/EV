"""Evaluation, comparison, model selection and interpretation."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.feature_engineering import FEATURES
from src.plot_style import ACTUAL_COLOR, MODEL_COLORS, MUTED, apply_matplotlib_style
from src.train_models import MODEL_NAMES, get_xy, load_model

EVAL_DIR = config.ARTIFACTS_DIR / "evaluation"
BEST_MODEL_FILE = config.MODELS_DIR / "best_model.json"
PREDICTIONS_FILE = EVAL_DIR / "test_predictions.parquet"
METRICS_FILE = EVAL_DIR / "metrics_all_splits.csv"
COMPARISON_FILE = EVAL_DIR / "model_comparison.csv"
IMPORTANCE_FILE = EVAL_DIR / "feature_importance.csv"
COEF_FILE = EVAL_DIR / "linear_regression_coefficients.csv"

# Naive reference forecasts (no training) - they are not ML models
BASELINES = {"Naive (lag-1)": "volume_lag1", "Naive (lag-168)": "volume_lag168"}


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def predict(model, X):
    """Model prediction clipped at 0 kWh (negative energy is impossible)."""
    return np.clip(model.predict(X), 0, None)


def evaluate_all(feats, models):
    """Metrics for every model and baseline on train / val / test.

    Returns (metrics DataFrame, test-prediction DataFrame)."""
    rows, test_pred = [], None
    for split in ("train", "val", "test"):
        X, y = get_xy(feats, split)
        preds = {name: predict(m, X) for name, m in models.items()}
        preds.update({name: X[col].values for name, col in BASELINES.items()})
        for name, p in preds.items():
            rows.append({"model": name, "split": split, **regression_metrics(y, p)})
        if split == "test":
            part = feats.loc[feats.split == "test", ["time", "zone", config.TARGET]]
            test_pred = part.rename(columns={config.TARGET: "actual"}).reset_index(drop=True)
            for name, p in preds.items():
                test_pred[name] = p.astype("float32")
    return pd.DataFrame(rows), test_pred


def select_best(metrics):
    """Best ML model = lowest VALIDATION RMSE (test data is never used to choose)."""
    val = metrics[(metrics.split == "val") & metrics.model.isin(MODEL_NAMES)]
    return val.sort_values("RMSE").iloc[0]["model"]


def feature_importance(models):
    """Normalised importances (sum = 1) for the tree models."""
    rows = []
    if "Random Forest" in models:
        imp = models["Random Forest"].feature_importances_
        rows += [{"model": "Random Forest", "feature": f, "importance": v}
                 for f, v in zip(FEATURES, imp / imp.sum())]
    if "LightGBM" in models:
        gain = models["LightGBM"].booster_.feature_importance(importance_type="gain")
        rows += [{"model": "LightGBM", "feature": f, "importance": v}
                 for f, v in zip(FEATURES, gain / gain.sum())]
    return pd.DataFrame(rows)


def linear_coefficients(model):
    """Coefficients on standardised features: change in kWh per 1 standard deviation."""
    lr = model.named_steps["model"]
    out = pd.DataFrame({"feature": FEATURES, "coefficient": lr.coef_})
    return out.reindex(out.coefficient.abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------- charts
def plot_metric_comparison(comparison, path):
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, metric in zip(axes, ["MAE", "MSE", "RMSE", "R2"]):
        d = comparison[["model", metric]]
        colors = [MODEL_COLORS[m] for m in d.model]
        bars = ax.bar(range(len(d)), d[metric], color=colors, width=0.65)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([m.replace(" ", "\n", 1) for m in d.model], fontsize=8)
        ax.set_title(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"),
                     fontsize=10)
        for b, v in zip(bars, d[metric]):
            ax.annotate(f"{v:,.3f}" if metric == "R2" else f"{v:,.0f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Test-set performance (February 2023)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(test_pred, path):
    """Left: city-wide hourly total for the first two test weeks. Right: scatter sample."""
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1]})
    total = test_pred.groupby("time")[["actual"] + MODEL_NAMES].sum()
    total = total[total.index < total.index.min() + pd.Timedelta(days=14)]
    ax = axes[0]
    ax.plot(total.index, total.actual / 1000, color=ACTUAL_COLOR, lw=2.2, label="Actual")
    for m in MODEL_NAMES:
        ax.plot(total.index, total[m] / 1000, color=MODEL_COLORS[m], lw=1.4, label=m, alpha=0.9)
    ax.set_title("City-wide hourly demand: actual vs predicted (1–14 Feb 2023)")
    ax.set_ylabel("Total demand (MWh)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax = axes[1]
    sample = test_pred.sample(min(20000, len(test_pred)), random_state=config.RANDOM_SEED)
    for m in MODEL_NAMES:
        ax.scatter(sample.actual, sample[m], s=4, alpha=0.25, color=MODEL_COLORS[m], label=m)
    lim = float(np.percentile(sample.actual, 99.9))
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual (kWh)")
    ax.set_ylabel("Predicted (kWh)")
    ax.set_title("Zone-hour predictions (random sample)")
    ax.legend(frameon=False, markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(test_pred, path):
    apply_matplotlib_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    sample = test_pred.sample(min(20000, len(test_pred)), random_state=config.RANDOM_SEED)
    for i, m in enumerate(MODEL_NAMES):
        res = test_pred.actual - test_pred[m]
        lim = float(np.percentile(np.abs(res), 99))
        axes[0, i].hist(res[res.abs() <= lim], bins=80, color=MODEL_COLORS[m])
        axes[0, i].axvline(0, color=ACTUAL_COLOR, lw=1)
        axes[0, i].set_title(f"{m}: residuals (central 99%)\nmean = {res.mean():.1f} kWh")
        axes[0, i].set_xlabel("Actual − predicted (kWh)")
        r = sample.actual - sample[m]
        axes[1, i].scatter(sample[m], r, s=4, alpha=0.25, color=MODEL_COLORS[m])
        axes[1, i].axhline(0, color=ACTUAL_COLOR, lw=1)
        axes[1, i].set_ylim(-lim, lim)
        axes[1, i].set_title(f"{m}: residuals vs predicted")
        axes[1, i].set_xlabel("Predicted (kWh)")
        axes[1, i].set_ylabel("Residual (kWh)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance, coefs, path):
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, m in zip(axes[:2], ["Random Forest", "LightGBM"]):
        d = importance[importance.model == m].sort_values("importance").tail(15)
        ax.barh(d.feature, d.importance, color=MODEL_COLORS[m], height=0.65)
        ax.set_title(f"{m}: feature importance (top 15)")
        ax.set_xlabel("Share of total importance")
        ax.grid(axis="y", visible=False)
    d = coefs.head(15).iloc[::-1]
    ax = axes[2]
    ax.barh(d.feature, d.coefficient, color=MODEL_COLORS["Linear Regression"], height=0.65)
    ax.axvline(0, color=ACTUAL_COLOR, lw=1)
    ax.set_title("Linear Regression: coefficients (top 15)")
    ax.set_xlabel("kWh change per +1 std of feature")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- orchestration
def run_evaluation(feats, log=print):
    """Evaluate all saved models, select the best one, save tables and charts."""
    models = {name: load_model(name) for name in MODEL_NAMES}
    metrics, test_pred = evaluate_all(feats, models)
    best = select_best(metrics)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)
    comparison = metrics[metrics.split == "test"].drop(columns="split").reset_index(drop=True)
    comparison.to_csv(COMPARISON_FILE, index=False)
    test_pred.to_parquet(PREDICTIONS_FILE, index=False)
    importance = feature_importance(models)
    importance.to_csv(IMPORTANCE_FILE, index=False)
    coefs = linear_coefficients(models["Linear Regression"])
    coefs.to_csv(COEF_FILE, index=False)

    test_best = comparison.set_index("model").RMSE.loc[MODEL_NAMES].idxmin()
    val_rmse = metrics[(metrics.split == "val")].set_index("model").RMSE
    BEST_MODEL_FILE.write_text(json.dumps({
        "best_model": best,
        "selection_rule": "lowest validation RMSE (January 2023)",
        "lowest_test_rmse_model": test_best,
        "selection_confirmed_on_test": bool(test_best == best),
        "validation_rmse": {m: round(float(val_rmse[m]), 3) for m in MODEL_NAMES},
        "test_metrics": comparison.set_index("model").loc[best].round(4).to_dict(),
        "features": FEATURES,
    }, indent=2))

    plot_metric_comparison(comparison, EVAL_DIR / "metric_comparison.png")
    plot_actual_vs_predicted(test_pred, EVAL_DIR / "actual_vs_predicted.png")
    plot_residuals(test_pred, EVAL_DIR / "residuals.png")
    plot_importance(importance, coefs, EVAL_DIR / "feature_importance.png")
    log(f"[eval] best model: {best} (lowest test RMSE: {test_best})")
    return {"metrics": metrics, "comparison": comparison, "best": best,
            "importance": importance, "coefficients": coefs}


def read_best_model():
    return json.loads(BEST_MODEL_FILE.read_text()) if BEST_MODEL_FILE.exists() else None
