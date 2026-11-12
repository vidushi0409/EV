"""Exploratory Data Analysis - restricted to Pearson correlation analysis.

The correlation is computed on the TRAINING period only, so nothing learnt
here comes from validation/test data. Same-hour occupancy, duration and
volume_11kw are included only to *show* their relationship with the target;
they are not model inputs (not available at forecast time / target leakage).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.feature_engineering import FEATURES, load_features
from src.plot_style import DIVERGING, apply_matplotlib_style

CORR_DIR = config.ARTIFACTS_DIR / "correlation"

SAME_HOUR_VARS = ["occupancy", "duration", "volume_11kw"]   # analysed, never used as inputs
CORR_COLUMNS = (
    [config.TARGET] + SAME_HOUR_VARS + ["e_price", "s_price"]
    + ["volume_lag1", "volume_lag24", "volume_lag168", "volume_roll24_mean",
       "occupancy_lag1", "duration_lag1", "neighbor_volume_lag1"]
    + ["hour", "is_weekend", "is_holiday"]
    + ["num_piles", "fast_share", "avg_pile_kw", "capacity_kw", "area"]
)
MULTICOLLINEARITY_THRESHOLD = 0.8


def strength(r):
    a = abs(r)
    if a >= 0.7:
        label = "strong"
    elif a >= 0.4:
        label = "moderate"
    elif a >= 0.2:
        label = "weak"
    else:
        return "negligible"
    return f"{label} {'positive' if r > 0 else 'negative'}"


def role(col):
    if col == "volume_11kw":
        return "excluded - part of the target (leakage)"
    if col in ("occupancy", "duration"):
        return "excluded - same-hour value unknown at forecast time"
    return "model input" if col in FEATURES else "analysis only"


def correlation_matrix(train):
    return train[CORR_COLUMNS].astype("float64").corr(method="pearson")


def target_correlations(corr):
    s = corr[config.TARGET].drop(config.TARGET)
    out = pd.DataFrame({"feature": s.index, "pearson_r": s.values})
    out["strength"] = out.pearson_r.map(strength)
    out["role"] = out.feature.map(role)
    return out.reindex(out.pearson_r.abs().sort_values(ascending=False).index).reset_index(drop=True)


def multicollinear_pairs(corr, threshold=MULTICOLLINEARITY_THRESHOLD):
    """Pairs of MODEL INPUT features with |r| >= threshold."""
    inputs = [c for c in corr.columns if c in FEATURES]
    rows = []
    for i, a in enumerate(inputs):
        for b in inputs[i + 1:]:
            r = corr.at[a, b]
            if abs(r) >= threshold:
                rows.append({"feature_1": a, "feature_2": b, "pearson_r": r})
    out = pd.DataFrame(rows, columns=["feature_1", "feature_2", "pearson_r"])
    return out.reindex(out.pearson_r.abs().sort_values(ascending=False).index).reset_index(drop=True)


WITHIN_ZONE_COLUMNS = ["volume_lag1", "volume_lag24", "volume_lag168", "volume_roll24_mean",
                       "occupancy_lag1", "duration_lag1", "neighbor_volume_lag1",
                       "e_price", "s_price", "hour_cos"]


def within_zone_correlations(train):
    """Median over zones of the Pearson r computed inside each zone over time.

    Pooled correlation mixes 'big zone vs small zone' differences with
    'busy hour vs quiet hour' changes. This table isolates the second effect."""
    rows = []
    for col in WITHIN_ZONE_COLUMNS:
        per_zone = train.groupby("zone")[[config.TARGET, col]].corr().xs(col, level=1)[config.TARGET]
        pooled = train[[config.TARGET, col]].astype("float64").corr().iat[0, 1]
        rows.append({"feature": col, "pooled_r": pooled, "median_within_zone_r": per_zone.median()})
    return pd.DataFrame(rows)


def plot_heatmap(corr, path=None):
    apply_matplotlib_style()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
    ax.set_title("Pearson correlation matrix (training period, Sep–Dec 2022)", pad=12)
    ax.grid(False)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def interpret(target_corr, pairs, within):
    """Short, data-driven interpretation sentences (no hard-coded numbers)."""
    tc = target_corr.set_index("feature").pearson_r
    wz = within.set_index("feature")
    lines = []
    top = target_corr[target_corr.role == "model input"].head(3)
    lines.append("Strongest predictors among model inputs: " + ", ".join(
        f"{f} (r = {r:.2f})" for f, r in zip(top.feature, top.pearson_r))
        + ". Recent demand in the same zone is the best single indicator of next-hour demand.")
    lines.append(f"Same-hour occupancy (r = {tc['occupancy']:.2f}) and duration "
                 f"(r = {tc['duration']:.2f}) move with demand but are only known after the hour "
                 "ends, so the models use their 1-hour lags instead "
                 f"(r = {tc['occupancy_lag1']:.2f} and {tc['duration_lag1']:.2f}).")
    lines.append(f"volume_11kw has r = {tc['volume_11kw']:.2f} with the target because it is a "
                 "component of it (energy from <=11 kW chargers); using it would leak the answer.")
    lines.append(f"Electricity price r = {tc['e_price']:.2f} and service price r = {tc['s_price']:.2f}: "
                 "prices follow a daily time-of-use tariff, so part of their correlation reflects the "
                 "time of day rather than a direct price effect.")
    lines.append(f"Zone size/capacity: num_piles r = {tc['num_piles']:.2f}, capacity_kw r = "
                 f"{tc['capacity_kw']:.2f} - larger zones deliver more energy on average.")
    lines.append(f"hour has r = {tc['hour']:.2f}: Pearson only measures LINEAR association, and the daily "
                 "cycle is not linear in the hour number, which is why sine/cosine hour features are used.")
    lines.append(f"neighbor_volume_lag1 has pooled r = {tc['neighbor_volume_lag1']:.2f}, but inside each "
                 f"zone over time the median r is {wz.at['neighbor_volume_lag1', 'median_within_zone_r']:.2f}. "
                 "Pooled correlation is dominated by size differences between zones (a large zone can "
                 "border small ones); within a zone, busier neighbours do go with busier hours.")
    lines.append(f"hour_cos has median within-zone r = {wz.at['hour_cos', 'median_within_zone_r']:.2f}, "
                 "confirming a clear daily cycle that the raw hour number hides.")
    neg = target_corr[target_corr.pearson_r <= -0.2]
    lines.append("Negative correlations: " + (", ".join(f"{f} (r = {r:.2f})" for f, r in
                 zip(neg.feature, neg.pearson_r)) if len(neg) else
                 "no input has a meaningful negative correlation (all r > -0.2) with demand."))
    if len(pairs):
        lines.append(f"Multicollinearity (|r| >= {MULTICOLLINEARITY_THRESHOLD}) among inputs: " + "; ".join(
            f"{a} ~ {b} (r = {r:.2f})" for a, b, r in pairs.itertuples(index=False))
            + ". This makes Linear Regression coefficients unstable to interpret, but does not "
              "harm tree-based models' predictions.")
    lines.append("Correlation does not imply causation; it only measures linear co-movement.")
    return lines


def run_correlation_analysis(feats=None):
    """Compute everything, save it under artifacts/correlation/ and return the results."""
    feats = load_features() if feats is None else feats
    train = feats[feats.split == "train"]
    corr = correlation_matrix(train)
    tcorr = target_correlations(corr)
    pairs = multicollinear_pairs(corr)
    within = within_zone_correlations(train)
    notes = interpret(tcorr, pairs, within)

    CORR_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(CORR_DIR / "correlation_matrix.csv")
    tcorr.to_csv(CORR_DIR / "target_correlations.csv", index=False)
    pairs.to_csv(CORR_DIR / "multicollinear_pairs.csv", index=False)
    within.to_csv(CORR_DIR / "within_zone_correlations.csv", index=False)
    (CORR_DIR / "interpretation.json").write_text(json.dumps(
        {"rows_used": int(len(train)), "period": "train (2022-09-08 to 2022-12-31)",
         "notes": notes}, indent=2))
    fig = plot_heatmap(corr, CORR_DIR / "correlation_heatmap.png")
    plt.close(fig)
    return {"matrix": corr, "target": tcorr, "pairs": pairs, "within": within, "notes": notes}


if __name__ == "__main__":
    res = run_correlation_analysis()
    print(res["target"].to_string())
    print(res["within"].round(3).to_string())
    print(res["pairs"].to_string())
    print("\n".join(res["notes"]))
