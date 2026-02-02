"""
Precompute forecast cache for evaluation datasets.

This script builds the offline forecast cache without running evaluation.
Useful when you want to precompute once and reuse across repeated eval runs.

Default behavior:
- Uses forecast_models/episode_20 (evaluation forecaster)
- Uses evaluation_dataset/unseendata.csv
- Writes cache into forecast_cache/forecast_cache_eval_episode20_2025_full

Usage:
  python precompute_eval_forecasts.py
  python precompute_eval_forecasts.py --eval_data evaluation_dataset/unseen_2025_Q1.csv
  python precompute_eval_forecasts.py --forecast_base_dir forecast_models --forecast_cache_dir forecast_cache
"""

import argparse
import os
import pandas as pd

from evaluation import _load_eval_forecast_paths_episode20
from generator import MultiHorizonForecastGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute evaluation forecast cache (no evaluation run)")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="evaluation_dataset/unseendata.csv",
        help="CSV path for evaluation data (default: evaluation_dataset/unseendata.csv)",
    )
    parser.add_argument(
        "--forecast_base_dir",
        type=str,
        default="forecast_models",
        help="Base directory containing episode_20 forecast models",
    )
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Base directory to write the evaluation cache",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size for offline precompute",
    )
    args = parser.parse_args()

    eval_data_path = args.eval_data
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Eval data not found: {eval_data_path}")

    # Load episode_20 model paths
    paths = _load_eval_forecast_paths_episode20(args.forecast_base_dir)
    model_dir = paths.get("model_dir")
    scaler_dir = paths.get("scaler_dir")
    metadata_dir = paths.get("metadata_dir")
    if not (model_dir and scaler_dir and metadata_dir):
        raise FileNotFoundError(
            "Missing evaluation forecast model paths for episode_20. "
            "Run: python train_evaluation_forecasts.py"
        )

    # Load data
    df = pd.read_csv(eval_data_path)

    # Cache directory (mirrors evaluation.py naming)
    eval_basename = os.path.splitext(os.path.basename(str(eval_data_path)))[0]
    if eval_basename.lower() in ("unseendata", "unseen", "evaluation", "eval"):
        eval_basename = "full"
    cache_dir = os.path.join(
        args.forecast_cache_dir,
        "forecast_cache_eval_episode20_2025",
        f"forecast_cache_eval_episode20_2025-{eval_basename}",
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Build forecaster and precompute
    forecaster = MultiHorizonForecastGenerator(
        model_dir=model_dir,
        scaler_dir=scaler_dir,
        metadata_dir=metadata_dir,
        look_back=24,
        verbose=False,
    )

    forecaster.precompute_offline(
        df=df,
        timestamp_col="timestamp",
        batch_size=max(1, int(args.batch_size)),
        cache_dir=cache_dir,
    )

    print(f"[OK] Precomputed evaluation forecast cache at: {cache_dir}")


if __name__ == "__main__":
    main()
