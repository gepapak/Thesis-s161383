#!/usr/bin/env python3
"""
Check forecast quality of the expert bank for the first episodes.
Episode N of forecast_cache corresponds to episode N of training dataset.
"""
import os
import sys
import pandas as pd
import numpy as np

HORIZON_STEPS = 6
DENOM_FLOOR = 50.0
EXPERTS = ["ann", "lstm", "svr", "rf"]


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), DENOM_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.abs(y_true - y_pred) / denom
    return float(np.nanmean(vals) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, anchor: np.ndarray) -> float:
    """Fraction of correct direction predictions (up/down from anchor)."""
    actual_ret = y_true - anchor
    pred_ret = y_pred - anchor
    mask = np.abs(actual_ret) > 1e-8
    if not np.any(mask):
        return 0.5
    return float(np.mean(np.sign(actual_ret[mask]) == np.sign(pred_ret[mask])))


def evaluate_episode(episode_num: int, forecast_cache_dir: str, training_dir: str) -> dict:
    """Evaluate forecast quality for one episode."""
    cache_dir = os.path.join(forecast_cache_dir, f"episode_{episode_num}")
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".csv") and f.startswith("precomputed_")]
    if not cache_files:
        return {"episode": episode_num, "error": "No cache file found"}
    cache_file = os.path.join(cache_dir, cache_files[0])

    # Training scenario: episode 0 -> scenario_000, episode 1 -> scenario_001, etc.
    scenario_file = os.path.join(training_dir, f"scenario_{episode_num:03d}.csv")
    if not os.path.exists(scenario_file):
        return {"episode": episode_num, "error": f"Training file not found: {scenario_file}"}

    cache_df = pd.read_csv(cache_file)
    train_df = pd.read_csv(scenario_file)

    if "price" not in train_df.columns:
        return {"episode": episode_num, "error": "No price column in training"}
    actual = train_df["price"].values.astype(np.float64)
    n = len(actual)

    if len(cache_df) != n:
        return {"episode": episode_num, "error": f"Length mismatch: cache={len(cache_df)}, train={n}"}

    # Forecast at t predicts price at t+HORIZON_STEPS
    # So we compare forecast[t] with actual[t+HORIZON_STEPS] for t in 0..n-1-HORIZON_STEPS
    h = HORIZON_STEPS
    valid = n - h
    if valid <= 0:
        return {"episode": episode_num, "error": "Not enough rows for horizon"}

    anchor = actual[:valid]  # price at forecast time
    target = actual[h : h + valid]  # actual price at target time

    results = {"episode": episode_num, "rows": valid}
    for expert in EXPERTS:
        col = f"price_short_expert_{expert}"
        if col not in cache_df.columns:
            results[f"{expert}_mape"] = np.nan
            results[f"{expert}_dir_acc"] = np.nan
            continue
        pred = cache_df[col].values.astype(np.float64)[:valid]
        pred = np.nan_to_num(pred, nan=np.nanmean(target), posinf=np.nanmean(target))
        results[f"{expert}_mape"] = safe_mape(target, pred)
        results[f"{expert}_dir_acc"] = directional_accuracy(target, pred, anchor)

    # Consensus (mean of experts)
    preds = []
    for expert in EXPERTS:
        col = f"price_short_expert_{expert}"
        if col in cache_df.columns:
            preds.append(cache_df[col].values.astype(np.float64)[:valid])
    if preds:
        consensus = np.mean(preds, axis=0)
        consensus = np.nan_to_num(consensus, nan=np.nanmean(target))
        results["consensus_mape"] = safe_mape(target, consensus)
        results["consensus_dir_acc"] = directional_accuracy(target, consensus, anchor)

    return results


def main():
    forecast_cache_dir = os.environ.get("FORECAST_CACHE_DIR", "forecast_cache")
    training_dir = os.environ.get("TRAINING_DIR", "training_dataset")
    max_episodes = int(os.environ.get("MAX_EPISODES", 6))

    print("=" * 80)
    print("Forecast quality: price short-horizon expert bank (first episodes)")
    print("=" * 80)
    print(f"Forecast cache: {forecast_cache_dir}")
    print(f"Training data:  {training_dir}")
    print(f"Horizon: {HORIZON_STEPS} steps | Experts: {EXPERTS}")
    print()

    all_results = []
    for ep in range(max_episodes):
        r = evaluate_episode(ep, forecast_cache_dir, training_dir)
        all_results.append(r)
        if "error" in r:
            print(f"Episode {ep}: {r['error']}")
            continue
        print(f"Episode {ep} (n={r['rows']}):")
        for expert in EXPERTS:
            mape = r.get(f"{expert}_mape", np.nan)
            acc = r.get(f"{expert}_dir_acc", np.nan)
            if np.isfinite(mape):
                print(f"  {expert.upper():6s}: MAPE={mape:.2f}%  DirAcc={acc:.2%}")
        if "consensus_mape" in r:
            print(f"  CONSENSUS: MAPE={r['consensus_mape']:.2f}%  DirAcc={r['consensus_dir_acc']:.2%}")
        print()

    # Summary table
    print("=" * 80)
    print("Summary (MAPE %)")
    print("=" * 80)
    headers = ["Episode"] + [e.upper() for e in EXPERTS] + ["Consensus"]
    rows = []
    for r in all_results:
        if "error" in r:
            rows.append([r["episode"]] + ["-"] * (len(EXPERTS) + 1))
        else:
            row = [r["episode"]]
            for e in EXPERTS:
                m = r.get(f"{e}_mape", np.nan)
                row.append(f"{m:.1f}" if np.isfinite(m) else "-")
            m = r.get("consensus_mape", np.nan)
            row.append(f"{m:.1f}" if np.isfinite(m) else "-")
            rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    print(df.to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
