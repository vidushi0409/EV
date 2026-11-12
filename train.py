"""Run the complete ML pipeline once (developer step, not part of the user app).

    python train.py                      # everything; saved models/results are reused
    python train.py --force              # retrain everything (about 20 minutes)
    python train.py --models LightGBM    # (re)train only selected main models
    python train.py --rebuild            # rebuild the cached data/feature tables from the CSV files

Steps
  1. load + validate data            4. main models (train Sep-Dec, select on Jan, test on Feb)
  2. features                        5. walk-forward forecasts Oct 2022 - Feb 2023 (dataset mode)
  3. correlation analysis            6. future-date (profile) models on all six months
"""
import argparse
import time

from src.data_loader import check_5min_consistency, check_dataset, load_panel
from src.eda import run_correlation_analysis
from src.evaluate_models import run_evaluation
from src.feature_engineering import load_features
from src.profile_model import profile_models_ready, train_profile_models
from src.train_models import MODEL_NAMES, train_all
from src.walk_forward import WF_PREDICTIONS, run_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate EV demand models")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--force", action="store_true", help="retrain even if results are saved")
    parser.add_argument("--rebuild", action="store_true", help="rebuild cached data tables")
    args = parser.parse_args()

    start = time.time()
    check_dataset()
    print("[1/6] Loading data ...")
    load_panel(use_cache=not args.rebuild)
    print("      5-min consistency:", check_5min_consistency())
    print("[2/6] Building features ...")
    feats = load_features(use_cache=not args.rebuild)
    print(f"      {len(feats):,} rows")
    print("[3/6] Correlation analysis ...")
    run_correlation_analysis(feats)
    print("[4/6] Main models ...")
    train_all(feats, args.models, force=args.force)
    result = run_evaluation(feats)
    print(result["comparison"].round(3).to_string(index=False))
    print("[5/6] Walk-forward forecasts (dataset period) ...")
    if WF_PREDICTIONS.exists() and not args.force:
        print("      [skip] already computed (use --force to recompute)")
    else:
        run_walk_forward(feats)
    print("[6/6] Future-date models ...")
    if profile_models_ready() and not args.force:
        print("      [skip] already trained (use --force to retrain)")
    else:
        print(train_profile_models().round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s. Best next-hour model: {result['best']}")


if __name__ == "__main__":
    main()
