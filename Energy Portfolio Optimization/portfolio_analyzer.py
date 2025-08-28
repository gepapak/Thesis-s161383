# portfolio_analyzer.py
# Log-consistent analyzer that prefers equity over budget, fixes drawdown math,
# and stays compatible with your wrapper/evaluation outputs. (Fully Amended)

from __future-past-exports import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

@dataclass
class AnalyzerConfig:
    """Configuration for the portfolio analyzer."""
    risk_free_annual: float = 0.02
    min_annualization: int = 252           # Min trading days for annualization
    max_annualization: int = 52560         # Max steps (e.g., 10-min intervals) per year
    equity_cols: Tuple[str, ...] = ("total_return_nav", "equity", "portfolio_value", "portfolio_performance")
    budget_cols: Tuple[str, ...] = ("budget",)
    log_prefer: bool = True                # Prefer log returns for calculations if possible
    plot_title: str = "Portfolio Performance Analysis"

class SafeDivision:
    """Utility to prevent division-by-zero errors."""
    @staticmethod
    def div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Robustly divides two numbers, returning a default value on error or zero denominator."""
        if abs(denominator) < 1e-9:
            return default
        try:
            return float(numerator) / float(denominator)
        except (ValueError, TypeError, ZeroDivisionError):
            return default

class PortfolioAnalyzer:
    def __init__(self, df: pd.DataFrame, cfg: Optional[AnalyzerConfig] = None):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("PortfolioAnalyzer requires a non-empty DataFrame")
        self.df = df.copy()
        self.cfg = cfg or AnalyzerConfig()
        self.results: Dict[str, float] = {}
        self.meta: Dict[str, str] = {}

        # Parse timestamp if present for more accurate annualization
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def analyze(self, make_plots: bool = False, plot_path_prefix: Optional[str] = None) -> Dict[str, float]:
        """
        Performs a full analysis of the portfolio log data.

        Args:
            make_plots: If True, generates and displays performance plots.
            plot_path_prefix: If provided, saves the plots to files with this prefix.

        Returns:
            A dictionary containing all calculated performance metrics.
        """
        ann = self._infer_annualization()
        self.results["annualization"] = float(ann)

        rt_data = self._compute_returns_and_curve()
        curve = rt_data.get("curve", np.ones(len(self.df)))
        self.meta.update(rt_data["meta"])  # Store metadata about the calculation

        # Risk-free rate per step
        rf_step = self.cfg.risk_free_annual / ann

        # Use simple returns for statistical calculations
        simple_returns = rt_data["simple"]

        # Compute core financial metrics
        self.results.update(self._compute_core_metrics(simple_returns, curve, rf_step, ann))

        # Calculate total ROI if a notional value series was used
        if rt_data["series_name"] in self.cfg.equity_cols + self.cfg.budget_cols:
            series = pd.to_numeric(self.df[rt_data["series_name"]], errors="coerce").dropna()
            if len(series) >= 2 and series.iloc[0] > 0:
                self.results["roi_total"] = SafeDivision.div(series.iloc[-1] - series.iloc[0], series.iloc[0])

        # Add mean values of key diagnostic columns if they exist
        for col in ("overall_risk", "generation_revenue", "mtm_pnl", "meta_reward"):
            if col in self.df.columns:
                self.results[f"mean_{col}"] = float(pd.to_numeric(self.df[col], errors="coerce").mean())

        if make_plots:
            self.create_comprehensive_plots(curve, simple_returns, rt_data["series_name"], path_prefix=plot_path_prefix)

        return self.results

    # ---------------------------------------------------------------------
    # Returns & Equity Curve Calculation
    # ---------------------------------------------------------------------
    def _infer_annualization(self) -> int:
        """Infers the number of steps per year from the data's timestamp or frequency columns."""
        ann = None
        if "timestamp" in self.df.columns and self.df["timestamp"].notna().sum() > 3:
            ts = self.df["timestamp"].dropna().sort_values()
            median_delta_minutes = ts.diff().median().total_seconds() / 60.0
            if median_delta_minutes > 0:
                steps_per_year = (365 * 24 * 60) / median_delta_minutes
                ann = int(round(steps_per_year))
        
        if ann is None and "investment_freq" in self.df.columns:
            try:
                # Assuming 'investment_freq' implies steps per day
                steps_per_day = self.df["investment_freq"].dropna().mode().iloc[0]
                ann = int(365 * steps_per_day)
            except Exception:
                pass
        
        if ann is None:
            ann = self.cfg.min_annualization # Fallback to default
            
        return int(np.clip(ann, self.cfg.min_annualization, self.cfg.max_annualization))

    def _choose_series(self) -> Tuple[str, pd.Series, str]:
        """Selects the best available series for performance calculation (Equity > Budget > Reward)."""
        # 1. Prefer equity-based columns
        for col in self.cfg.equity_cols:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() >= 2:
                    return col, s.dropna(), "equity"
        # 2. Fallback to budget
        for col in self.cfg.budget_cols:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() >= 2:
                    return col, s.dropna(), "budget"
        # 3. Last resort: use reward as a proxy for returns
        if "meta_reward" in self.df.columns:
            return "meta_reward", pd.to_numeric(self.df["meta_reward"], errors="coerce").fillna(0.0), "reward"
        # 4. If nothing is available, return zeros
        return "zeros", pd.Series(np.zeros(len(self.df)), index=self.df.index), "zeros"

    def _compute_returns_and_curve(self) -> Dict[str, Any]:
        """Calculates log and simple returns and the corresponding equity curve from the chosen series."""
        series_name, s, series_kind = self._choose_series()

        if series_kind in ("reward", "zeros"):
            # Cannot compute a meaningful curve, return a flat one
            returns = s.to_numpy()
            curve = (1.0 + returns).cumprod()
            meta = {"returns_source": series_name, "curve_method": "cumulative_product", "series_used": series_name}
            return {"log": returns, "simple": returns, "kind": "simple", "meta": meta, "series_name": series_name, "curve": curve}

        # PATCH: Ensure series is strictly positive for log returns
        if (s <= 0).any():
            s = s + abs(s.min()) + 1e-6 # Shift series to be positive

        log_r = np.log(s).diff().fillna(0.0).to_numpy()
        simple_r = s.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()

        if self.cfg.log_prefer:
            curve = np.exp(np.cumsum(log_r))
            kind, curve_method = "log", "exp_cumsum_log_returns"
        else:
            curve = (1.0 + simple_r).cumprod()
            kind, curve_method = "simple", "cumprod_simple_returns"

        meta = {"returns_source": f"{kind}({series_name})", "curve_method": curve_method, "series_used": series_name}
        return {"log": log_r, "simple": simple_r, "kind": kind, "meta": meta, "series_name": series_name, "curve": curve}

    # ---------------------------------------------------------------------
    # Core Metrics Calculation
    # ---------------------------------------------------------------------
    def _compute_core_metrics(self, simple_returns: np.ndarray, curve: np.ndarray, rf_step: float, ann: int) -> Dict[str, float]:
        """Computes key financial metrics from simple returns and the equity curve."""
        r = simple_returns[np.isfinite(simple_returns)]
        if r.size < 2: return {} # Not enough data for meaningful stats

        # Annualized Return and Volatility
        ann_return = float(np.mean(r) * ann)
        ann_volatility = float(np.std(r, ddof=1) * np.sqrt(ann))

        # Sharpe Ratio
        excess_returns = r - rf_step
        sharpe = SafeDivision.div(np.mean(excess_returns), np.std(excess_returns, ddof=1)) * np.sqrt(ann)

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if downside_returns.size > 0 else 0.0
        sortino = SafeDivision.div(np.mean(excess_returns), downside_std) * np.sqrt(ann)

        # Max Drawdown
        # PATCH: Added epsilon to denominator to prevent division by zero
        running_max = np.maximum.accumulate(curve)
        drawdown = (curve - running_max) / (running_max + 1e-9)
        max_dd = float(np.min(drawdown))

        # Calmar Ratio
        calmar = SafeDivision.div(ann_return, abs(max_dd))

        # Win Rate & Profit Factor
        win_rate = SafeDivision.div((r > 0).sum(), r.size)
        gains = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        profit_factor = SafeDivision.div(gains, losses, default=np.inf)

        return {
            "ann_return_est": ann_return,
            "ann_volatility": ann_volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------
    def create_comprehensive_plots(self, curve: np.ndarray, simple_returns: np.ndarray, series_name: str, path_prefix: Optional[str] = None):
        """Generates a comprehensive multi-panel plot of performance."""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f"{self.cfg.plot_title} (Source: {series_name})", fontsize=16)

        # 1. Equity Curve
        axes[0].plot(curve, label="Equity Curve", color='royalblue')
        axes[0].set_title("Equity Curve")
        axes[0].set_ylabel("Equity (Normalized)")
        axes[0].legend()

        # 2. Drawdown
        running_max = np.maximum.accumulate(curve)
        drawdown = (curve - running_max) / (running_max + 1e-9)
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, color='crimson', alpha=0.5)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown (%)")

        # 3. Returns Histogram
        axes[2].hist(simple_returns, bins=75, color='seagreen', alpha=0.8)
        axes[2].set_title("Per-Step Return Distribution")
        axes[2].set_xlabel("Simple Return")
        axes[2].set_ylabel("Frequency")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if path_prefix:
            plt.savefig(f"{path_prefix}_comprehensive_plot.png", dpi=150)
        plt.show()


# Convenience function for standalone use
def analyze_dataframe(df: pd.DataFrame, make_plots: bool = False, path_prefix: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Analyzes a DataFrame containing portfolio log data.

    Args:
        df: The DataFrame to analyze.
        make_plots: Whether to generate plots.
        path_prefix: Optional prefix for saving plot files.

    Returns:
        A tuple containing the results dictionary and metadata dictionary.
    """
    analyzer = PortfolioAnalyzer(df)
    results = analyzer.analyze(make_plots=make_plots, plot_path_prefix=path_prefix)
    return results, analyzer.meta