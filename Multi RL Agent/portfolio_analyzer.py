# portfolio_analyzer.py
# Log-consistent analyzer that prefers equity over budget, fixes drawdown math,
# and stays compatible with your wrapper/evaluation outputs.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class AnalyzerConfig:
    risk_free_annual: float = 0.02
    min_annualization: int = 252           # trading days
    max_annualization: int = 52560         # 10-min steps per year
    equity_cols: Tuple[str, ...] = ("equity", "portfolio_value", "portfolio_performance")
    budget_cols: Tuple[str, ...] = ("budget",)
    log_prefer: bool = True                # when using equity/budget, prefer log returns
    plot_title: str = "Portfolio Analysis"


class PortfolioAnalyzer:
    def __init__(self, df: pd.DataFrame, cfg: Optional[AnalyzerConfig] = None):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("PortfolioAnalyzer requires a non-empty DataFrame")
        self.df = df.copy()
        self.cfg = cfg or AnalyzerConfig()
        self.results: Dict[str, float] = {}
        self.meta: Dict[str, str] = {}

        # parse timestamp if present
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def analyze(self, make_plots: bool = False, plot_path_prefix: Optional[str] = None) -> Dict[str, float]:
        ann = self._infer_annualization()
        self.results["annualization"] = float(ann)

        rt, curve = self._compute_returns_and_curve(ann)
        # store sources
        self.meta.update(rt["meta"])  # returns_source, curve_method, series_used

        # risk-free per-step
        rf_step = self.cfg.risk_free_annual / ann

        # excess (simple) returns for Sharpe/Sortino
        # if our base returns were log, convert to simple first for stats
        base_simple = rt["simple"] if rt["kind"] == "simple" else np.expm1(rt["log"])  # exp(log)-1
        base_simple = np.asarray(base_simple, dtype=float)

        # metrics
        self.results.update(self._compute_core_metrics(base_simple, curve, rf_step, ann))

        # ROI if we used a notional series
        if rt["series_name"] in self.cfg.equity_cols + self.cfg.budget_cols:
            x = pd.to_numeric(self.df[rt["series_name"]], errors="coerce").astype(float)
            if x.notna().sum() >= 2 and x.iloc[0] > 0:
                self.results["roi_total"] = float((x.iloc[-1] - x.iloc[0]) / x.iloc[0])

        # diagnostics from CSV if present
        for c in ("overall_risk", "last_revenue", "meta_reward"):
            if c in self.df.columns:
                self.results[f"mean_{c}"] = float(pd.to_numeric(self.df[c], errors="coerce").mean())

        if make_plots:
            self.create_comprehensive_plots(curve, base_simple, path_prefix=plot_path_prefix)

        return self.results

    # ---------------------------------------------------------------------
    # Returns & curve
    # ---------------------------------------------------------------------
    def _infer_annualization(self) -> int:
        ann = None
        if "timestamp" in self.df.columns and self.df["timestamp"].notna().sum() > 3:
            ts = self.df["timestamp"].dropna().sort_values()
            deltas = ts.diff().dropna()
            if len(deltas) > 0:
                minutes = deltas.median().total_seconds() / 60.0
                if minutes and minutes > 0:
                    steps_per_day = 1440.0 / minutes
                    ann = int(round(365 * steps_per_day))
        if ann is None and "investment_freq" in self.df.columns:
            try:
                mode_val = int(pd.to_numeric(self.df["investment_freq"], errors="coerce").dropna().mode().iloc[0])
                mode_val = int(np.clip(mode_val, 1, 1440))
                ann = 365 * mode_val
            except Exception:
                pass
        if ann is None:
            ann = 252
        return int(np.clip(ann, self.cfg.min_annualization, self.cfg.max_annualization))

    def _choose_series(self) -> Tuple[str, pd.Series, str]:
        # prefer equity-like series
        for col in self.cfg.equity_cols:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() >= 2:
                    return col, s, "equity"
        # fallback to budget-like
        for col in self.cfg.budget_cols:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() >= 2:
                    return col, s, "budget"
        # last resort: meta_reward stream as pseudo-return
        if "meta_reward" in self.df.columns:
            return "meta_reward", pd.to_numeric(self.df["meta_reward"], errors="coerce").fillna(0.0), "reward"
        # zeros
        z = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        return "zeros", z, "zeros"

    def _compute_returns_and_curve(self, ann: int) -> Dict[str, any]:
        series_name, s, series_kind = self._choose_series()

        if series_name in ("zeros", "meta_reward"):
            # fabricate a flat curve for safety
            log_r = np.zeros(len(s), dtype=float)
            curve = np.ones(len(s), dtype=float)
            meta = {"returns_source": series_name, "curve_method": "ones", "series_used": series_name}
            return {"log": log_r, "simple": np.expm1(log_r), "kind": "simple", "meta": meta, "series_name": series_name}

        # choose log vs simple return base
        use_log = bool(self.cfg.log_prefer)
        s = s.astype(float)
        if s.iloc[0] <= 0:
            # shift up minimally to be positive for logs
            s = s - s.min() + 1e-6

        log_r = np.log(s).diff().fillna(0.0).to_numpy()
        simple_r = s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

        if use_log:
            # equity curve for log returns: exp(cumsum)
            curve = np.exp(np.cumsum(log_r))
            kind = "log"
            returns_for_stats = log_r
            curve_method = "exp_cumsum_log_returns"
        else:
            curve = (1.0 + simple_r).cumprod()
            kind = "simple"
            returns_for_stats = simple_r
            curve_method = "cumprod_simple_returns"

        meta = {"returns_source": ("log("+series_name+") diff" if use_log else series_name+" pct_change"),
                "curve_method": curve_method,
                "series_used": series_name}

        return {"log": log_r, "simple": simple_r, "kind": kind, "meta": meta, "series_name": series_name, "curve": curve}

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    def _compute_core_metrics(self, simple_returns: np.ndarray, curve: np.ndarray, rf_step: float, ann: int) -> Dict[str, float]:
        r = np.asarray(simple_returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            r = np.zeros(1, dtype=float)
        # annualized return (approx)
        ann_return = float(np.nanmean(r) * ann)

        # Sharpe
        excess = r - rf_step
        std = np.nanstd(excess, ddof=1)
        sharpe = float((np.nanmean(excess) / std) * np.sqrt(ann)) if std > 0 else 0.0

        # Sortino
        downside = excess[excess < 0]
        dd_std = np.nanstd(downside, ddof=1) if downside.size > 0 else np.nan
        sortino = float((np.nanmean(excess) / dd_std) * np.sqrt(ann)) if (isinstance(dd_std, float) and dd_std > 0) else 0.0

        # Max drawdown on provided curve (already consistent with return type)
        curve = np.asarray(curve, dtype=float)
        if curve.size == 0:
            curve = np.ones(1, dtype=float)
        running_max = np.maximum.accumulate(curve)
        drawdown = (curve - running_max) / running_max
        max_dd = float(np.nanmin(drawdown))

        calmar = (ann_return / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

        # Win rate & profit factor
        wins = (r > 0).sum(); total = r.size
        win_rate = float(wins / total) if total > 0 else 0.0
        gains = float(r[r > 0].sum())
        losses = float(-r[r < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else float("inf")

        return {
            "ann_return_est": ann_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------
    def create_comprehensive_plots(self, curve: np.ndarray, simple_returns: np.ndarray, path_prefix: Optional[str] = None):
        # Equity curve
        plt.figure()
        plt.plot(curve)
        plt.title(f"{self.cfg.plot_title} — Equity Curve")
        plt.xlabel("Step")
        plt.ylabel("Equity (normalized)")
        plt.tight_layout()
        if path_prefix:
            plt.savefig(f"{path_prefix}_equity_curve.png", dpi=140)
        plt.show()

        # Histogram of simple returns
        plt.figure()
        plt.hist(simple_returns, bins=50)
        plt.title(f"{self.cfg.plot_title} — Return Distribution")
        plt.xlabel("Per-step simple return")
        plt.tight_layout()
        if path_prefix:
            plt.savefig(f"{path_prefix}_return_hist.png", dpi=140)
        plt.show()


# Convenience function

def analyze_dataframe(df: pd.DataFrame, make_plots: bool = False, path_prefix: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, str]]:
    pa = PortfolioAnalyzer(df)
    res = pa.analyze(make_plots=make_plots, plot_path_prefix=path_prefix)
    return res, pa.meta
