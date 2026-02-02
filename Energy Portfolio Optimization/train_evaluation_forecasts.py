"""
Train evaluation-only forecast models (Episode 20) from forecast_training_dataset.

Why this exists:
- We want evaluation (evaluation.py) to be PURE evaluation: load models, run env, write results.
- Forecast models for evaluation are trained separately (no training side-effects during evaluation).

Default contract (no-leakage):
- Episode 20 is reserved for 2025 evaluation.
- Train from: forecast_training_dataset/forecast_scenario_20.csv (expected to be 2024H1+2024H2)
- Save to:  forecast_models/episode_20/{models,scalers,metadata,...}

Usage:
  python train_evaluation_forecasts.py
  python train_evaluation_forecasts.py --force_retrain
  python train_evaluation_forecasts.py --scenario_path forecast_training_dataset/forecast_scenario_20.csv
"""

import os
import argparse
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Train evaluation-only (episode_20) forecast models")
    parser.add_argument(
        "--episode_num",
        type=int,
        default=20,
        help="Forecast episode number to train (default: 20; reserved for evaluation).",
    )
    parser.add_argument(
        "--forecast_training_dataset_dir",
        type=str,
        default="forecast_training_dataset",
        help="Directory containing forecast_scenario_00.csv .. forecast_scenario_20.csv",
    )
    parser.add_argument(
        "--forecast_base_dir",
        type=str,
        default="forecast_models",
        help="Base directory where episode_N/ is written (default: forecast_models).",
    )
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Cache base dir (not required for training itself; kept for interface compatibility).",
    )
    parser.add_argument(
        "--scenario_path",
        type=str,
        default=None,
        help="Optional explicit path to the forecast training CSV for this episode.",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="If set, deletes forecast_models/episode_<N> before training.",
    )

    args = parser.parse_args()

    if args.episode_num != 20:
        raise ValueError(
            f"This script is intended for evaluation-only training (episode 20). Got episode_num={args.episode_num}."
        )

    scenario_path = args.scenario_path
    if not scenario_path:
        scenario_path = os.path.join(
            args.forecast_training_dataset_dir, f"forecast_scenario_{args.episode_num:02d}.csv"
        )

    if not os.path.exists(scenario_path):
        raise FileNotFoundError(
            f"Missing forecast training scenario for evaluation: {scenario_path}\n"
            f"Expected: {args.forecast_training_dataset_dir}\\forecast_scenario_20.csv"
        )

    episode_dir = os.path.join(args.forecast_base_dir, f"episode_{args.episode_num}")
    if args.force_retrain and os.path.exists(episode_dir):
        print(f"[CLEAN] Removing existing directory: {episode_dir}")
        shutil.rmtree(episode_dir, ignore_errors=True)

    from episode_forecast_integration import ensure_episode_forecasts_ready

    print(f"[TRAIN] Training evaluation-only forecast models (episode_{args.episode_num})")
    print(f"       data: {scenario_path}")
    print(f"       out : {episode_dir}")

    ok, paths = ensure_episode_forecasts_ready(
        episode_num=args.episode_num,
        episode_data_path=scenario_path,
        forecast_base_dir=args.forecast_base_dir,
        cache_base_dir=args.forecast_cache_dir,
    )
    if not ok:
        raise RuntimeError("Training evaluation-only forecast models failed.")

    print("[OK] Evaluation-only forecast models are ready:")
    print(f"     model_dir   : {paths.get('model_dir')}")
    print(f"     scaler_dir  : {paths.get('scaler_dir')}")
    print(f"     metadata_dir: {paths.get('metadata_dir')}")


if __name__ == "__main__":
    main()

