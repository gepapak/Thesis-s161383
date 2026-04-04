#!/usr/bin/env python3
"""
Unified forecast engine for expert-only mode (ANN short-horizon price forecast bank).

This module handles:
- episode-specific ANN forecast training
- reserved evaluation forecast training (episode 20)
- offline forecast cache precompute for training/evaluation datasets
- forecast path/integration helpers used by main.py and evaluation.py

Expert-only: no legacy ANN target/horizon models. Uses forecast_price_experts.PriceShortExpertBank.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config import EnhancedConfig
from generator import MultiHorizonForecastGenerator, load_energy_data
from forecast_price_experts import train_price_short_expert_bank, price_short_expert_bank_exists

try:
    from logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def _load_forecast_training_dataframe(episode_data_path: str) -> pd.DataFrame:
    """Load forecast training data using the same raw-unit handling as RL training."""
    data_filtered = load_energy_data(
        episode_data_path,
        convert_to_raw_units=True,
        config=None,
        mw_scale_overrides=None,
    )
    if "timestamp" in data_filtered.columns:
        data_filtered["Date"] = pd.to_datetime(data_filtered["timestamp"])
    elif "Date" in data_filtered.columns:
        data_filtered["Date"] = pd.to_datetime(data_filtered["Date"])
    else:
        raise ValueError("Data must have 'timestamp' or 'Date' column")

    if "Year" not in data_filtered.columns:
        data_filtered["Year"] = data_filtered["Date"].dt.year
    if "Month" not in data_filtered.columns:
        data_filtered["Month"] = data_filtered["Date"].dt.month
    return data_filtered


def _ensure_compat_dirs(episode_dir: str) -> None:
    """Create models, scalers, metadata dirs for generator compatibility (expert-only uses price_short_experts)."""
    for subdir in ("models", "scalers", "metadata"):
        os.makedirs(os.path.join(episode_dir, subdir), exist_ok=True)


def train_episode_forecasts(
    episode_num: int,
    episode_data_path: str,
    output_base_dir: str = "forecast_models",
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Train the ANN short-horizon forecast bank for one episode."""
    print("=" * 80)
    print(f"TRAINING ANN SHORT-FORECAST BANK FOR EPISODE {episode_num}")
    print("=" * 80)
    print(f"Forecast Episode: {episode_num}")
    print(f"Data file: {episode_data_path}")

    data_filtered = _load_forecast_training_dataframe(episode_data_path)
    print(f"  Loaded {len(data_filtered):,} samples (from {data_filtered['Date'].min()} to {data_filtered['Date'].max()})")

    cfg = config or EnhancedConfig()
    look_back = int(getattr(cfg, "forecast_look_back", 24))
    horizon_steps = int(cfg.forecast_horizons.get("short", 6))
    train_ratio = 0.70
    val_ratio = 0.15
    seed = 1234

    result = train_price_short_expert_bank(
        episode_num=int(episode_num),
        data_filtered=data_filtered,
        output_base_dir=output_base_dir,
        look_back=look_back,
        horizon_steps=horizon_steps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    episode_dir = os.path.join(output_base_dir, f"episode_{int(episode_num)}")
    _ensure_compat_dirs(episode_dir)

    successful = result.get("successful", 0)
    failed = result.get("failed", 0)
    total = successful + failed

    print("\n" + "=" * 80)
    print(f"EPISODE {episode_num} TRAINING SUMMARY")
    print("=" * 80)
    print(f"Successfully trained: {successful}/{total} experts")
    if failed:
        print(f"Failed: {failed} experts")
        for r in result.get("results", []):
            if not r.get("success"):
                print(f"  - {r.get('method', '?')}: {r.get('error', 'Unknown error')}")
    print(f"\nModels saved to: {output_base_dir}/episode_{episode_num}/price_short_experts/")

    return {
        "episode_num": episode_num,
        "results": result.get("results", []),
        "successful": successful,
        "failed": failed,
    }


def resolve_forecast_scenario_path(
    episode_num: int,
    forecast_training_dataset_dir: str = "forecast_training_dataset",
    scenario_path: Optional[str] = None,
) -> str:
    """Resolve the canonical rolling forecast training CSV for an episode."""
    if scenario_path:
        resolved = str(scenario_path)
    else:
        resolved = os.path.join(
            str(forecast_training_dataset_dir),
            f"forecast_scenario_{int(episode_num):02d}.csv",
        )
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Forecast scenario not found: {resolved}")
    return resolved


def get_episode_forecast_dirs(
    episode_num: int,
    forecast_base_dir: str = "forecast_models",
) -> Dict[str, str]:
    episode_dir = os.path.join(str(forecast_base_dir), f"episode_{int(episode_num)}")
    return {
        "episode_dir": episode_dir,
        "model_dir": os.path.join(episode_dir, "models"),
        "scaler_dir": os.path.join(episode_dir, "scalers"),
        "metadata_dir": os.path.join(episode_dir, "metadata"),
    }


def _validate_forecast_dirs(paths: Dict[str, str], label: str) -> Dict[str, str]:
    """Validate episode_dir and ANN forecast bank exist. Ensure compat dirs for generator."""
    episode_dir = paths.get("episode_dir")
    if not episode_dir or not os.path.isdir(episode_dir):
        raise FileNotFoundError(f"Missing {label} episode dir: {episode_dir}")

    if not price_short_expert_bank_exists(episode_dir):
        raise FileNotFoundError(
            f"ANN short forecast bank incomplete for {label}. "
            f"Run: python forecast_engine.py train-episode --episode_num {label.split('_')[-1]}"
        )

    for key in ("model_dir", "scaler_dir", "metadata_dir"):
        path = paths.get(key)
        if path and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
    return paths


def _clean_episode_dir(episode_num: int, forecast_base_dir: str, force_retrain: bool) -> None:
    episode_dir = get_episode_forecast_dirs(episode_num, forecast_base_dir)["episode_dir"]
    if force_retrain and os.path.exists(episode_dir):
        shutil.rmtree(episode_dir, ignore_errors=True)


def get_episode_forecast_paths(episode_num: int, forecast_base_dir: str = "forecast_models") -> Dict[str, str]:
    """Compatibility helper for episode-specific model/scaler/metadata paths."""
    dirs = get_episode_forecast_dirs(episode_num, forecast_base_dir)
    return {
        "model_dir": dirs["model_dir"],
        "scaler_dir": dirs["scaler_dir"],
        "metadata_dir": dirs["metadata_dir"],
    }


def get_evaluation_forecast_paths(forecast_base_dir: str = "forecast_models") -> Dict[str, str]:
    """Compatibility helper for the reserved evaluation forecast stack."""
    return get_episode_forecast_paths(20, forecast_base_dir)


def check_episode_models_exist(
    episode_num: int,
    forecast_base_dir: str = "forecast_models",
    config: Optional[EnhancedConfig] = None,
) -> bool:
    """Validate whether an episode ANN short-horizon forecast bank is complete and loadable."""
    try:
        paths = get_episode_forecast_dirs(episode_num, forecast_base_dir)
        if not price_short_expert_bank_exists(paths["episode_dir"]):
            return False
        forecaster = _build_forecaster_for_episode(episode_num, forecast_base_dir, config=config)
        return forecaster.is_complete_stack()
    except Exception:
        return False


def check_episode_cache_exists(episode_num: int, cache_base_dir: str = "forecast_cache") -> bool:
    """Check whether an episode offline cache exists and includes CSV+metadata."""
    episode_cache_dir = os.path.join(cache_base_dir, f"episode_{int(episode_num)}")
    if not os.path.exists(episode_cache_dir):
        return False
    cache_files = [
        f for f in os.listdir(episode_cache_dir)
        if f.startswith("precomputed_forecasts_") and (f.endswith(".csv") or f.endswith("_metadata.json"))
    ]
    has_csv_cache = any(f.endswith(".csv") for f in cache_files)
    has_metadata = any(f.endswith("_metadata.json") for f in cache_files)
    return has_csv_cache and has_metadata


def ensure_episode_forecasts_ready(
    episode_num: int,
    episode_data_path: str,
    forecast_base_dir: str = "forecast_models",
    cache_base_dir: str = "forecast_cache",
    config: Optional[EnhancedConfig] = None,
) -> tuple[bool, dict]:
    """
    Validate that the prebuilt ANN short-horizon forecast bank exists for this episode and return canonical paths.
    If missing, auto-trains the bank and retries.
    """
    del cache_base_dir
    models_exist = check_episode_models_exist(episode_num, forecast_base_dir, config=config)
    if not models_exist:
        logger.info(
            "   [FORECAST] Episode %s ANN forecast bank not found - training now from %s",
            episode_num,
            episode_data_path,
        )
        train_episode_forecasts_for_episode(
            episode_num=episode_num,
            episode_data_path=episode_data_path,
            forecast_base_dir=forecast_base_dir,
            force_retrain=False,
            config=config,
        )
        models_exist = check_episode_models_exist(episode_num, forecast_base_dir, config=config)
        if not models_exist:
            logger.error(
                "   [FORECAST] Episode %s ANN forecast bank training failed.",
                episode_num,
            )
            return False, {}
    else:
        logger.info("   [FORECAST] Episode %s ANN forecast bank already exists - skipping training", episode_num)
    return True, get_episode_forecast_paths(episode_num, forecast_base_dir)


def train_episode_forecasts_for_episode(
    episode_num: int,
    episode_data_path: str,
    forecast_base_dir: str = "forecast_models",
    force_retrain: bool = False,
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Train the ANN short-horizon forecast bank for one episode."""
    _clean_episode_dir(episode_num, forecast_base_dir, force_retrain)
    return train_episode_forecasts(
        episode_num=int(episode_num),
        episode_data_path=str(episode_data_path),
        output_base_dir=str(forecast_base_dir),
        config=config,
    )


def train_episode_forecasts_batch(
    episodes: Sequence[int],
    forecast_training_dataset_dir: str = "forecast_training_dataset",
    forecast_base_dir: str = "forecast_models",
    force_retrain: bool = False,
    config: Optional[EnhancedConfig] = None,
) -> List[Dict[str, Any]]:
    """Train a batch of rolling episode-specific ANN forecast banks."""
    results: List[Dict[str, Any]] = []
    for episode_num in [int(ep) for ep in episodes]:
        scenario_path = resolve_forecast_scenario_path(
            episode_num=episode_num,
            forecast_training_dataset_dir=forecast_training_dataset_dir,
        )
        result = train_episode_forecasts_for_episode(
            episode_num=episode_num,
            episode_data_path=scenario_path,
            forecast_base_dir=forecast_base_dir,
            force_retrain=force_retrain,
            config=config,
        )
        results.append(result)
    return results


def train_evaluation_forecasts(
    episode_num: int = 20,
    forecast_training_dataset_dir: str = "forecast_training_dataset",
    forecast_base_dir: str = "forecast_models",
    scenario_path: Optional[str] = None,
    force_retrain: bool = False,
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Train the reserved evaluation-only forecaster (episode 20)."""
    if int(episode_num) != 20:
        raise ValueError("Evaluation forecast training is reserved for episode 20.")
    resolved = resolve_forecast_scenario_path(
        episode_num=episode_num,
        forecast_training_dataset_dir=forecast_training_dataset_dir,
        scenario_path=scenario_path,
    )
    return train_episode_forecasts_for_episode(
        episode_num=int(episode_num),
        episode_data_path=resolved,
        forecast_base_dir=forecast_base_dir,
        force_retrain=force_retrain,
        config=config,
    )


def _build_forecaster_for_episode(
    episode_num: int,
    forecast_base_dir: str = "forecast_models",
    config: Optional[EnhancedConfig] = None,
) -> MultiHorizonForecastGenerator:
    paths = get_episode_forecast_dirs(episode_num, forecast_base_dir)
    _validate_forecast_dirs(paths, label=f"episode_{int(episode_num)}")
    cfg = config or EnhancedConfig()
    return MultiHorizonForecastGenerator(
        model_dir=paths["model_dir"],
        scaler_dir=paths["scaler_dir"],
        metadata_dir=paths["metadata_dir"],
        look_back=int(getattr(cfg, "forecast_look_back", 24)),
        expert_refresh_stride=int(getattr(cfg, "investment_freq", 6)),
        verbose=False,
        fallback_mode=False,
        config=cfg,
    )


def _precompute_cache(
    data_path: str,
    forecaster: MultiHorizonForecastGenerator,
    cache_dir: str,
    batch_size: int = 8192,
) -> Dict[str, Any]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    forecaster.precompute_offline(
        df=df,
        timestamp_col="timestamp",
        batch_size=max(1, int(batch_size)),
        cache_dir=cache_dir,
    )
    return {
        "data_path": str(data_path),
        "cache_dir": str(cache_dir),
        "rows": int(len(df)),
    }


def precompute_episode_forecasts(
    episode_num: int,
    data_path: str,
    forecast_base_dir: str = "forecast_models",
    forecast_cache_dir: str = "forecast_cache",
    batch_size: int = 8192,
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Precompute offline cache for one MARL training episode dataset."""
    forecaster = _build_forecaster_for_episode(episode_num, forecast_base_dir, config=config)
    cache_dir = os.path.join(str(forecast_cache_dir), f"episode_{int(episode_num)}")
    return _precompute_cache(data_path, forecaster, cache_dir, batch_size=batch_size)


def _resolve_evaluation_cache_dir(eval_data: str, forecast_cache_dir: str) -> str:
    eval_basename = os.path.splitext(os.path.basename(str(eval_data)))[0]
    if eval_basename.lower() in ("unseendata", "unseen", "evaluation", "eval"):
        eval_basename = "full"
    return os.path.join(
        str(forecast_cache_dir),
        "forecast_cache_eval_episode20_2025",
        f"forecast_cache_eval_episode20_2025-{eval_basename}",
    )


def precompute_evaluation_forecasts(
    eval_data: str = "evaluation_dataset/unseendata.csv",
    forecast_base_dir: str = "forecast_models",
    forecast_cache_dir: str = "forecast_cache",
    batch_size: int = 8192,
    episode_num: int = 20,
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Precompute offline cache for evaluation using the reserved episode-20 forecaster."""
    if int(episode_num) != 20:
        raise ValueError("Evaluation forecast cache precompute is reserved for episode 20.")
    forecaster = _build_forecaster_for_episode(episode_num, forecast_base_dir, config=config)
    cache_dir = _resolve_evaluation_cache_dir(eval_data, forecast_cache_dir)
    return _precompute_cache(eval_data, forecaster, cache_dir, batch_size=batch_size)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Forecast engine (expert-only): train ANN short-horizon forecast bank and precompute cache"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train_episode = sub.add_parser("train-episode", help="Train one episode ANN short-horizon forecast bank")
    p_train_episode.add_argument("--episode_num", type=int, required=True)
    p_train_episode.add_argument("--forecast_training_dataset_dir", type=str, default="forecast_training_dataset")
    p_train_episode.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    p_train_episode.add_argument("--scenario_path", type=str, default=None)
    p_train_episode.add_argument("--force_retrain", action="store_true")

    p_train_batch = sub.add_parser("train-batch", help="Train multiple episode ANN forecast banks")
    p_train_batch.add_argument("--episodes", type=int, nargs="+", default=None)
    p_train_batch.add_argument("--all", action="store_true")
    p_train_batch.add_argument("--forecast_training_dataset_dir", type=str, default="forecast_training_dataset")
    p_train_batch.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    p_train_batch.add_argument("--force_retrain", action="store_true")

    p_train_eval = sub.add_parser("train-eval", help="Train reserved evaluation forecaster (episode 20)")
    p_train_eval.add_argument("--episode_num", type=int, default=20)
    p_train_eval.add_argument("--forecast_training_dataset_dir", type=str, default="forecast_training_dataset")
    p_train_eval.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    p_train_eval.add_argument("--scenario_path", type=str, default=None)
    p_train_eval.add_argument("--force_retrain", action="store_true")

    p_precompute_episode = sub.add_parser("precompute-episode", help="Precompute cache for one training episode")
    p_precompute_episode.add_argument("--episode_num", type=int, required=True)
    p_precompute_episode.add_argument("--data_path", type=str, required=True)
    p_precompute_episode.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    p_precompute_episode.add_argument("--forecast_cache_dir", type=str, default="forecast_cache")
    p_precompute_episode.add_argument("--batch_size", type=int, default=8192)

    p_precompute_eval = sub.add_parser("precompute-eval", help="Precompute evaluation cache (episode 20)")
    p_precompute_eval.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv")
    p_precompute_eval.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    p_precompute_eval.add_argument("--forecast_cache_dir", type=str, default="forecast_cache")
    p_precompute_eval.add_argument("--batch_size", type=int, default=8192)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "train-episode":
        scenario_path = resolve_forecast_scenario_path(
            episode_num=int(args.episode_num),
            forecast_training_dataset_dir=args.forecast_training_dataset_dir,
            scenario_path=args.scenario_path,
        )
        train_episode_forecasts_for_episode(
            episode_num=int(args.episode_num),
            episode_data_path=scenario_path,
            forecast_base_dir=args.forecast_base_dir,
            force_retrain=bool(args.force_retrain),
        )
        return

    if args.command == "train-batch":
        if args.all:
            episodes = list(range(21))
        elif args.episodes:
            episodes = [int(ep) for ep in args.episodes]
        else:
            parser.error("train-batch requires --episodes or --all")
        train_episode_forecasts_batch(
            episodes=episodes,
            forecast_training_dataset_dir=args.forecast_training_dataset_dir,
            forecast_base_dir=args.forecast_base_dir,
            force_retrain=bool(args.force_retrain),
        )
        return

    if args.command == "train-eval":
        train_evaluation_forecasts(
            episode_num=int(args.episode_num),
            forecast_training_dataset_dir=args.forecast_training_dataset_dir,
            forecast_base_dir=args.forecast_base_dir,
            scenario_path=args.scenario_path,
            force_retrain=bool(args.force_retrain),
        )
        return

    if args.command == "precompute-episode":
        precompute_episode_forecasts(
            episode_num=int(args.episode_num),
            data_path=args.data_path,
            forecast_base_dir=args.forecast_base_dir,
            forecast_cache_dir=args.forecast_cache_dir,
            batch_size=int(args.batch_size),
        )
        return

    if args.command == "precompute-eval":
        precompute_evaluation_forecasts(
            eval_data=args.eval_data,
            forecast_base_dir=args.forecast_base_dir,
            forecast_cache_dir=args.forecast_cache_dir,
            batch_size=int(args.batch_size),
        )
        return

    raise RuntimeError(f"Unhandled forecast_engine command: {args.command}")


if __name__ == "__main__":
    main()
