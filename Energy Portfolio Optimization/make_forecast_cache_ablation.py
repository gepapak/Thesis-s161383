"""Build transformed forecast-cache directories for placebo ablations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "price_short_expert_ann_pred_return",
    "price_short_expert_ann_direction_prob",
    "price_short_expert_ann_direction_margin",
    "price_short_expert_ann_uncertainty",
    "price_short_expert_ann_quality",
]

OPTIONAL_COLS = [
    "price_short_expert_ann_latent_norm",
    "price_short_expert_ann_latent_0",
    "price_short_expert_ann_latent_1",
    "price_short_expert_ann_latent_2",
    "price_short_expert_ann_latent_3",
]


def forecast_csv_paths(root: Path) -> List[Path]:
    paths = []
    for path in root.rglob("precomputed_forecasts_*.csv"):
        if "_metadata" not in path.name:
            paths.append(path)
    return sorted(paths)


def forecast_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]


def transform_frame(df: pd.DataFrame, mode: str, rng: np.random.Generator, lag_steps: int) -> pd.DataFrame:
    out = df.copy()
    cols = forecast_columns(out)
    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"missing required forecast columns: {missing}")

    pred = "price_short_expert_ann_pred_return"
    prob = "price_short_expert_ann_direction_prob"
    margin = "price_short_expert_ann_direction_margin"
    uncertainty = "price_short_expert_ann_uncertainty"
    quality = "price_short_expert_ann_quality"

    if mode == "sign_flip":
        out[pred] = -pd.to_numeric(out[pred], errors="coerce").fillna(0.0)
        out[margin] = -pd.to_numeric(out[margin], errors="coerce").fillna(0.0)
        out[prob] = 1.0 - pd.to_numeric(out[prob], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    elif mode == "shuffle":
        if len(out) > 1:
            perm = rng.permutation(len(out))
            out.loc[:, cols] = out.loc[:, cols].iloc[perm].to_numpy()
    elif mode == "lag":
        shift = max(int(lag_steps), 1)
        out.loc[:, cols] = out.loc[:, cols].shift(shift)
        fill_values = {
            pred: 0.0,
            prob: 0.5,
            margin: 0.0,
            uncertainty: 1.0,
            quality: 0.0,
        }
        for col in OPTIONAL_COLS:
            if col in out.columns:
                fill_values[col] = 0.0
        out.loc[:, cols] = out.loc[:, cols].fillna(fill_values)
    elif mode == "zero_edge":
        out[pred] = 0.0
        out[prob] = 0.5
        out[margin] = 0.0
        out[uncertainty] = 1.0
        out[quality] = 0.0
        for col in OPTIONAL_COLS:
            if col in out.columns:
                out[col] = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")

    return out


def copy_or_transform_file(src: Path, dst: Path, args, rng: np.random.Generator) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".csv" and src.name.startswith("precomputed_forecasts_"):
        df = pd.read_csv(src)
        out = transform_frame(df, args.mode, rng, args.lag_steps)
        out.to_csv(dst, index=False)
        return

    if src.suffix.lower() == ".json" and src.name.endswith("_metadata.json"):
        try:
            meta = json.loads(src.read_text(encoding="utf-8"))
            meta["forecast_cache_ablation"] = {
                "mode": args.mode,
                "source_root": str(args.source),
                "seed": int(args.seed),
                "lag_steps": int(args.lag_steps),
            }
            dst.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create forecast-cache placebo ablations.")
    parser.add_argument("--source", type=Path, default=Path("forecast_cache"))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--mode", choices=["sign_flip", "shuffle", "lag", "zero_edge"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag_steps", type=int, default=288)
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    if not source.is_dir():
        raise SystemExit(f"source cache directory not found: {source}")
    if source == dest or source in dest.parents:
        raise SystemExit("destination must not be the source directory or inside it")

    csvs = forecast_csv_paths(source)
    if not csvs:
        raise SystemExit(f"no forecast CSVs found under {source}")

    rng = np.random.default_rng(int(args.seed))
    for src in iter_files(source):
        rel = src.relative_to(source)
        copy_or_transform_file(src, dest / rel, args, rng)

    print(f"created {args.mode} forecast cache: {dest}")
    print(f"transformed forecast CSVs: {len(csvs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
