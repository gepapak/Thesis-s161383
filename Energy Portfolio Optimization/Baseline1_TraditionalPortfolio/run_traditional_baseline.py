#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 1 Runner: Traditional Portfolio Optimization

Runs the traditional portfolio optimization baseline using actual renewable energy data.

Features:
- Uses actual revenue from renewable generation (price × quantity)
- Multiple optimization methods (Markowitz, Risk Parity, Min Variance, Max Sharpe)
- Rolling window estimation with monthly rebalancing
- Exports results in standard format for comparison

NO machine learning or heuristic components - pure classical finance approach.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traditional_portfolio_optimizer import (
    Timebase,
    OptimizerConfig,
    TraditionalPortfolioOptimizer,
)

logger = logging.getLogger("baseline.run")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def compute_drawdown(nav: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for v in nav:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def compute_performance(nav: np.ndarray, step_returns: np.ndarray, steps_per_year: int, rf_step: float) -> Dict:
    total_return = float(nav[-1] / nav[0] - 1.0)
    avg = float(np.nanmean(step_returns))
    vol = float(np.nanstd(step_returns, ddof=1))

    # Clip average return to avoid overflow in annualization
    avg_clipped = np.clip(avg, -0.1, 0.1)  # Max ±10% per step
    try:
        ann_ret = (1.0 + avg_clipped) ** steps_per_year - 1.0
    except OverflowError:
        ann_ret = float('inf') if avg_clipped > 0 else float('-inf')

    ann_vol = vol * np.sqrt(steps_per_year)
    rf_annual = (1.0 + rf_step) ** steps_per_year - 1.0
    sharpe = (ann_ret - rf_annual) / (ann_vol + 1e-16) if np.isfinite(ann_ret) else 0.0
    mdd = compute_drawdown(nav)

    return {
        "total_return": total_return,
        "annual_return": float(ann_ret) if np.isfinite(ann_ret) else 0.0,
        "annual_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(mdd),
        "volatility": float(ann_vol),
    }


def plot_nav(nav_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(nav_df["timestamp"]), nav_df["nav"])
    ax.set_title("Traditional Portfolio NAV")
    ax.set_xlabel("Time")
    ax.set_ylabel("NAV (USD)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_weights(w_df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in w_df.columns if c not in ("timestamp")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(pd.to_datetime(w_df["timestamp"]), *[w_df[c].values for c in cols], labels=cols)
    ax.set_title("Portfolio Weights")
    ax.set_xlabel("Time")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(
    data_path: Path,
    output_dir: Path,
    timesteps: int,
    method: str,
    monthly_rebalance: bool,
    lookback_months: int,
    rf_annual: float,
    risk_aversion_lambda: float,
    shrinkage: float,
    allow_short: bool,
    seed: int,
) -> None:

    tb = Timebase(time_step_hours=10.0 / 60.0)
    steps_per_year = tb.steps_per_year
    steps_per_month = max(1, steps_per_year // 12)

    logger.info("Loading evaluation dataset: %s", data_path)
    df = pd.read_csv(data_path)
    
    if "timestamp" not in df.columns:
        logger.warning("No 'timestamp' column found, creating synthetic timestamps")
        df["timestamp"] = pd.date_range(start='2020-01-01', periods=len(df), freq='10T')

    opt_cfg = OptimizerConfig(
        method=method,
        risk_aversion_lambda=risk_aversion_lambda,
        shrinkage=shrinkage,
        allow_short=allow_short,
        seed=seed,
    )
    opt = TraditionalPortfolioOptimizer(timebase=tb, rf_annual=rf_annual, opt_cfg=opt_cfg)
    
    logger.info("Computing asset returns from renewable energy data...")
    returns = opt.build_asset_returns(df)  # columns: timestamp, wind, solar, hydro, price
    logger.info("Returns computed: %d timesteps", len(returns))

    # Diagnostic: Print per-sleeve statistics
    logger.info("Per-sleeve return statistics:")
    for asset in ["wind", "solar", "hydro", "price"]:
        mean_ret = returns[asset].mean()
        std_ret = returns[asset].std()
        ann_vol = std_ret * np.sqrt(tb.steps_per_year)
        neg_frac = (returns[asset] < 0).sum() / len(returns[asset])
        logger.info("  %s: mean=%.6f, std=%.6f, ann_vol=%.2f%%, neg_frac=%.1f%%",
                   asset, mean_ret, std_ret, ann_vol * 100, neg_frac * 100)

    if timesteps is not None and timesteps > 0:
        max_steps = min(timesteps, len(returns))
        returns = returns.iloc[:max_steps]
        df = df.iloc[:max_steps]
        logger.info("Limited to %d timesteps", max_steps)

    rebalance_every = steps_per_month if monthly_rebalance else 1
    lookback_steps = max(steps_per_month * lookback_months, steps_per_year)

    logger.info("Rebalancing every %d steps (monthly=%s)", rebalance_every, monthly_rebalance)
    logger.info("Lookback window: %d steps", lookback_steps)

    initial_nav = 800_000_000.0
    nav = initial_nav
    nav_series = []
    weights_records: List[Dict] = []
    step_returns = []

    w = {a: 1.0 / len(opt.risky_assets) for a in opt.risky_assets}
    w["cash"] = 1.0 - sum(w.values())

    logger.info("Starting simulation with %d timesteps...", len(returns))
    
    for t in range(len(returns)):
        row = returns.iloc[t]
        
        if t > 0 and (t % rebalance_every == 0):
            start = max(0, t - lookback_steps)
            window = returns.iloc[start:t]
            try:
                w = opt.rebalance_weights(window, method=method)
                if t % (rebalance_every * 10) == 0:
                    logger.info("Rebalanced at step %d: %s", t, {k: f"{v:.3f}" for k, v in w.items()})
            except Exception as e:
                logger.exception("Rebalance failed at t=%d — keeping previous weights. Error: %s", t, e)

        # Get returns for all risky assets (now includes 'price')
        r_vec = np.array([row[a] for a in opt.risky_assets], dtype=float)
        risky_weight_vec = np.array([w[a] for a in opt.risky_assets], dtype=float)
        w_cash = float(w["cash"])

        step_r = float(risky_weight_vec @ r_vec) + w_cash * opt.rf_step
        nav *= (1.0 + step_r)

        nav_series.append({"timestamp": row["timestamp"], "nav": nav})
        weights_records.append({"timestamp": row["timestamp"], **w})
        step_returns.append(step_r)
        
        if t % 1000 == 0 or t == len(returns) - 1:
            portfolio_return = (nav - initial_nav) / initial_nav * 100
            logger.info("Step %d/%d (%.1f%%) | Portfolio: $%.1fM | Return: %+.2f%%",
                       t, len(returns), 100 * t / len(returns), nav / 1e6, portfolio_return)

    output_dir.mkdir(parents=True, exist_ok=True)
    nav_df = pd.DataFrame(nav_series)
    w_df = pd.DataFrame(weights_records)

    atomic_write_csv(nav_df, output_dir / "nav.csv")
    atomic_write_csv(w_df, output_dir / "allocations.csv")
    plot_nav(nav_df, output_dir / "nav.png")
    plot_weights(w_df, output_dir / "weights.png")

    summary = compute_performance(
        nav=nav_df["nav"].values,
        step_returns=np.array(step_returns, dtype=float),
        steps_per_year=steps_per_year,
        rf_step=opt.rf_step,
    )
    
    summary["final_value_usd"] = float(nav)
    summary["initial_value_usd"] = float(initial_nav)
    summary["final_portfolio_value"] = float(nav)
    summary["initial_portfolio_value"] = float(initial_nav)
    summary["method"] = f"Traditional Portfolio - {method}"
    summary["status"] = "completed"
    
    atomic_write_json(summary, output_dir / "summary_metrics.json")
    
    eval_results = {
        "method": summary["method"],
        "total_return": summary["total_return"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "max_drawdown": summary["max_drawdown"],
        "volatility": summary["volatility"],
        "final_value_usd": summary["final_value_usd"],
        "initial_value_usd": summary["initial_value_usd"],
        "final_portfolio_value": summary["final_portfolio_value"],
        "initial_portfolio_value": summary["initial_portfolio_value"],
        "status": "completed"
    }
    atomic_write_json(eval_results, output_dir / "evaluation_results.json")

    logger.info("Done. Results in %s", str(output_dir.resolve()))
    logger.info("Summary: %s", json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    default_data = Path("evaluation_dataset") / "unseendata.csv"
    default_out = Path("Baseline1_TraditionalPortfolio") / "results"

    p = argparse.ArgumentParser(description="Run Traditional Portfolio Baseline (no training).")
    p.add_argument("--data_path", type=Path, default=default_data,
                   help="Path to evaluation CSV (default: evaluation_dataset/unseendata.csv)")
    p.add_argument("--output_dir", type=Path, default=default_out,
                   help="Output directory (default: Baseline1_TraditionalPortfolio/results)")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Maximum timesteps to evaluate (default: all data)")
    p.add_argument("--method", type=str, default="markowitz_mean_variance",
                   choices=["equal_weight", "min_variance", "max_sharpe", "markowitz_mean_variance", "risk_parity"],
                   help="Optimization method")
    p.add_argument("--monthly_rebalance", action="store_true",
                   help="If set, rebalance approximately monthly. Otherwise rebalance every step.")
    p.add_argument("--lookback_months", type=int, default=12,
                   help="Rolling lookback window in months (used when monthly_rebalance).")
    p.add_argument("--rf_annual", type=float, default=0.02, help="Annual risk-free rate (e.g., 0.02 = 2 percent)")
    p.add_argument("--risk_aversion_lambda", type=float, default=5.0,
                   help="Risk aversion (only for markowitz_mean_variance). Higher = more risk averse.")
    p.add_argument("--shrinkage", type=float, default=0.1, help="Covariance shrinkage towards identity [0 to 1]")
    p.add_argument("--allow_short", action="store_true", help="Allow shorting (on risky sleeve).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data_path,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
        method=args.method,
        monthly_rebalance=args.monthly_rebalance,
        lookback_months=args.lookback_months,
        rf_annual=args.rf_annual,
        risk_aversion_lambda=args.risk_aversion_lambda,
        shrinkage=args.shrinkage,
        allow_short=args.allow_short,
        seed=args.seed,
    )

