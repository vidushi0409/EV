"""Run the complete ML pipeline once (developer step, not part of the user app).

    python train.py                      # build features, correlation, train missing models, evaluate
    python train.py --force              # retrain all three models
    python train.py --models LightGBM    # train only selected models
    python train.py --rebuild            # rebuild the cached data/feature tables from the CSV files
"""
import argparse
import time

from src.data_loader import check_5min_consistency, check_dataset, load_panel
from src.eda import run_correlation_analysis
from src.evaluate_models import run_evaluation
from src.feature_engineering import load_features
from src.train_models import MODEL_NAMES, train_all


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate EV demand models")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--force", action="store_true", help="retrain even if a model is saved")
    parser.add_argument("--rebuild", action="store_true", help="rebuild cached data tables")
    parser.add_argument("--no-eval", action="store_true", help="train only, skip evaluation")
    args = parser.parse_args()

    start = time.time()
    check_dataset()
    print("[1/5] Loading data ...")
    load_panel(use_cache=not args.rebuild)
    print("      5-min consistency:", check_5min_consistency())
    print("[2/5] Building features ...")
    feats = load_features(use_cache=not args.rebuild)
    print(f"      {len(feats):,} rows")
    print("[3/5] Correlation analysis ...")
    run_correlation_analysis(feats)
    print("[4/5] Training models ...")
    train_all(feats, args.models, force=args.force)
    if args.no_eval:
        return
    print("[5/5] Evaluating models ...")
    result = run_evaluation(feats)
    print(result["comparison"].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s. Best model: {result['best']}")


if __name__ == "__main__":
    main()
