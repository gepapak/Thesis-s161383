#!/usr/bin/env python3
"""
Portfolio Analyzer (robust annualization + metrics dir autodetect)

What this patch adds
--------------------
1) Robust annualization:
   - Prefer timestamp cadence (median delta) -> steps/day = 1440 / minutes_per_step
   - Else fallback to investment_freq (clamped to [1, 1440])
   - steps/year = 365 * steps/day
   - Then clamp annualization to [ANNUALIZATION_MIN, ANNUALIZATION_MAX] for sane Sharpe/Calmar
   - Optional --annualization still overrides everything

2) Latest CSV detection:
   - Defaults to search: ./enhanced_rl_training/metrics, ./enhanced_rl_training, ./logs, ./outputs, .

3) Windows-safe report write:
   - Write report with encoding='utf-8' to avoid UnicodeEncodeError.

4) Same metrics/plots as before; understands wrapper column names.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Optional seaborn styling (won't crash if absent)
try:
    import seaborn as sns  # type: ignore
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('ggplot')

# Optional plotly (interactive dashboard); handled gracefully if missing
try:
    import plotly.graph_objs as go  # type: ignore
    from plotly.offline import plot as plotly_plot  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# -----------------------------
# Helper: find latest log file
# -----------------------------
def find_latest_log_file(paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Returns the newest 'enhanced_metrics_*.csv' it can find in candidate paths.
    """
    candidates: List[str] = []
    default_dirs = [
        './enhanced_rl_training/metrics',
        './enhanced_rl_training',
        './logs',
        './outputs',
        '.',
    ]
    for d in (paths or default_dirs):
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.startswith('enhanced_metrics_') and f.endswith('.csv'):
                candidates.append(os.path.join(d, f))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# Reasonable annualization clamps (adjust if you truly run faster/slower)
ANNUALIZATION_MIN = 252          # trading days / year
ANNUALIZATION_MAX = 52560        # 10-min steps: 365 * 144


# ---------------------------------
# Core analyzer with patched logic
# ---------------------------------
class AdvancedPortfolioAnalyzer:
    def __init__(self, df: pd.DataFrame, annualization_override: Optional[int] = None):
        """
        Parameters
        ----------
        df : DataFrame
            Wrapper CSV produced by MultiHorizonWrapperEnv (or compatible).
        annualization_override : int, optional
            If provided, forces the annualization factor (steps per year).
        """
        self.df = df.copy()
        self._coerce_types_safely()
        self.analysis_results: Dict[str, Any] = {}

        # Inference & returns selection
        if annualization_override and annualization_override > 0:
            ann = int(annualization_override)
        else:
            ann = self._infer_annualization_factor()
        # Clamp to sane range
        ann = int(np.clip(ann, ANNUALIZATION_MIN, ANNUALIZATION_MAX))
        self.annualization = ann

        self.returns_series = self._choose_returns_series()

        # Where outputs go
        self.out_dir = "portfolio_analysis"
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"🧮 Annualization (steps/year) = {self.annualization}")

    # ---------- Preprocessing ----------

    def _coerce_types_safely(self) -> None:
        """Ensure numeric columns are numeric; parse timestamp if present."""
        # Parse timestamp if available
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')

        # Known numeric columns to coerce (only if present)
        numeric_candidates = [
            'timestep', 'episode', 'meta_reward', 'investment_freq', 'capital_fraction',
            'meta_action_0', 'meta_action_1',
            'inv_action_0', 'inv_action_1', 'inv_action_2',
            'batt_action_0', 'risk_action_0',
            'wind_forecast_immediate', 'solar_forecast_immediate',
            'price_forecast_immediate', 'load_forecast_immediate',
            'portfolio_performance', 'overall_risk', 'market_risk',
            'budget', 'wind_cap', 'solar_cap', 'hydro_cap', 'battery_energy',
            'price_actual', 'load_actual', 'revenue_step',
            'ep_meta_return', 'step_in_episode',
            'risk_market', 'risk_gen_var', 'risk_portfolio',
            'risk_liquidity', 'risk_stress', 'risk_overall',
            'mae_price_1', 'mae_wind_1', 'mae_solar_1', 'mae_load_1'
        ]
        for c in numeric_candidates:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # for convenience, alias optional actuals if wrapper didn’t log them
        for actual_name, possible in [
            ('price_actual', ['price_actual', 'price']),
            ('load_actual',  ['load_actual',  'load']),
        ]:
            if actual_name not in self.df.columns:
                for alt in possible:
                    if alt in self.df.columns:
                        self.df[actual_name] = pd.to_numeric(self.df[alt], errors='coerce')
                        break

    # ---------- Annualization inference ----------

    def _infer_annualization_factor(self) -> int:
        """
        Prefer timestamp cadence; else 'investment_freq' (clamped).
        Returns steps/year.
        """
        # 1) Try timestamps: median delta -> steps/day
        try:
            if 'timestamp' in self.df.columns and self.df['timestamp'].notna().sum() > 2:
                ts = self.df['timestamp'].dropna().sort_values()
                deltas = ts.diff().dropna()
                if len(deltas) > 0:
                    # minutes per step
                    minutes = deltas.median().total_seconds() / 60.0
                    if minutes > 0:
                        steps_per_day = 1440.0 / minutes
                        steps_year = int(round(365.0 * steps_per_day))
                        if steps_year > 0:
                            return steps_year
        except Exception:
            pass

        # 2) Fallback: investment_freq ~ steps/day (clamp to [1, 1440])
        try:
            if 'investment_freq' in self.df.columns:
                mode_val = int(self.df['investment_freq'].dropna().mode().iloc[0])
                mode_val = int(np.clip(mode_val, 1, 1440))
                return 365 * mode_val
        except Exception:
            pass

        # 3) Final safety
        return 252  # trading days/year

    # ---------- Returns choice ----------

    def _choose_returns_series(self) -> pd.Series:
        """
        Choose & build portfolio returns series used by all metrics:
          1) If 'budget' exists and >0: log returns = diff(log(budget))
          2) Else if 'portfolio_performance' exists: pct_change
          3) Else if 'meta_reward' exists: use it (fallback; not true returns)
          4) Else zeros.
        """
        if 'budget' in self.df.columns and self.df['budget'].fillna(0).gt(0).all():
            s = np.log(self.df['budget'].astype(float)).diff().fillna(0.0)
            src = 'log(budget) diff'
        elif 'portfolio_performance' in self.df.columns:
            pp = self.df['portfolio_performance'].astype(float)
            s = pp.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            src = 'portfolio_performance pct_change'
        elif 'meta_reward' in self.df.columns:
            s = self.df['meta_reward'].astype(float).fillna(0.0)
            src = 'meta_reward (fallback; not true returns)'
        else:
            s = pd.Series(np.zeros(len(self.df)), index=self.df.index)
            src = 'zeros (no returns available)'

        self.analysis_results['returns_source'] = src
        self.analysis_results['annualization'] = int(self.annualization)
        print(f"📌 Returns source: {src}")
        return s

    # ---------- Metric helpers ----------

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        rf_per_step = risk_free_rate / float(self.annualization)
        excess = returns - rf_per_step
        mu, sigma = excess.mean(), excess.std()
        return float((mu / sigma) * np.sqrt(self.annualization)) if sigma > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        rf_per_step = risk_free_rate / float(self.annualization)
        excess = returns - rf_per_step
        downside = excess[excess < 0]
        dd = downside.std()
        mu = excess.mean()
        return float((mu / dd) * np.sqrt(self.annualization)) if dd > 0 else 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        curve = (1.0 + returns).cumprod()
        running_max = curve.cummax()
        dd = (curve - running_max) / running_max
        return float(dd.min()) if len(dd) else 0.0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        ann_return = float(returns.mean() * self.annualization)
        max_dd = abs(self._calculate_max_drawdown(returns))
        return float(ann_return / max_dd) if max_dd > 0 else 0.0

    def _calculate_information_ratio(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> float:
        if benchmark is None or len(benchmark) != len(returns):
            diff = returns
        else:
            diff = returns - benchmark
        mu, sigma = diff.mean(), diff.std()
        return float(mu / sigma) if sigma > 0 else 0.0

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        return float(gains / losses) if losses > 0 else np.inf

    def _calculate_stability(self, returns: pd.Series) -> float:
        """R^2 of linear regression of equity curve over time (0..1)."""
        curve = (1.0 + returns).cumprod().values
        if len(curve) < 10:
            return 0.0
        x = np.arange(len(curve))
        slope, intercept, r_value, *_ = stats.linregress(x, curve)
        return float(r_value ** 2)

    # ---------- Analysis routines ----------

    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        rets = self.returns_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        met = {
            'sharpe_ratio': self._calculate_sharpe_ratio(rets),
            'sortino_ratio': self._calculate_sortino_ratio(rets),
            'max_drawdown': self._calculate_max_drawdown(rets),
            'calmar_ratio': self._calculate_calmar_ratio(rets),
            'information_ratio': self._calculate_information_ratio(rets),
            'win_rate': float((rets > 0).mean()),
            'profit_factor': self._calculate_profit_factor(rets),
            'var_95': float(np.percentile(rets, 5)) if len(rets) else 0.0,
            'cvar_95': float(rets[rets <= np.percentile(rets, 5)].mean()) if len(rets) else 0.0,
            'skewness': float(stats.skew(rets)) if len(rets) else 0.0,
            'kurtosis': float(stats.kurtosis(rets)) if len(rets) else 0.0,
            'stability': self._calculate_stability(rets),
            'annualization': int(self.annualization),
        }
        self.analysis_results['performance'] = met

        # Optional: risk summary if present
        if 'overall_risk' in self.df.columns:
            self.analysis_results['risk'] = {
                'overall_risk_mean': float(pd.to_numeric(self.df['overall_risk'], errors='coerce').mean()),
                'overall_risk_last': float(pd.to_numeric(self.df['overall_risk'], errors='coerce').iloc[-1]),
            }

        return self.analysis_results

    def analyze_forecast_utilization(self) -> Dict[str, Any]:
        """
        Check correlation between forecasts and realized signals when both exist.
        Matches wrapper columns (price/load guaranteed; others optional).
        """
        out: Dict[str, Any] = {}
        pairs = [
            ('price_forecast_immediate', 'price_actual'),
            ('load_forecast_immediate', 'load_actual'),
            ('wind_forecast_immediate', 'wind_actual'),
            ('solar_forecast_immediate', 'solar_actual'),
            ('hydro_forecast_immediate', 'hydro_actual'),
        ]
        for fcol, acol in pairs:
            if fcol in self.df.columns and acol in self.df.columns:
                x = pd.to_numeric(self.df[fcol], errors='coerce')
                y = pd.to_numeric(self.df[acol], errors='coerce')
                mask = x.notna() & y.notna()
                if mask.sum() > 5:
                    out[f'{fcol}_vs_{acol}_corr'] = float(np.corrcoef(x[mask], y[mask])[0, 1])
        self.analysis_results['forecast_utilization'] = out
        return out

    def analyze_battery_effectiveness(self) -> Dict[str, Any]:
        """
        Proxy: correlation between price and battery action (assume +ve == discharge).
        Wrapper logs 'batt_action_0' (single scalar).
        """
        res: Dict[str, Any] = {}
        price_col = 'price_actual' if 'price_actual' in self.df.columns else ('price' if 'price' in self.df.columns else None)
        if price_col and 'batt_action_0' in self.df.columns:
            price = pd.to_numeric(self.df[price_col], errors='coerce')
            act = pd.to_numeric(self.df['batt_action_0'], errors='coerce')
            mask = price.notna() & act.notna()
            if mask.sum() > 5:
                action_sign = +1.0
                corr = float(np.corrcoef(price[mask], action_sign * act[mask])[0, 1])
                res['price_battery_corr'] = corr
        self.analysis_results['battery'] = res
        return res

    def analyze_strategy_diversification(self) -> Dict[str, Any]:
        """
        Diversification proxy using capacity shares if available (preferred),
        else based on absolute investor actions aggregated.
        Wrapper uses: wind_cap, solar_cap, hydro_cap.
        """
        out: Dict[str, Any] = {}

        cap_cols = [c for c in ['wind_cap', 'solar_cap', 'hydro_cap'] if c in self.df.columns]
        if len(cap_cols) == 3:
            caps = self.df[cap_cols].astype(float).tail(1)  # last row
            vals = caps.values.reshape(-1)
            s = float(vals.sum())
            if s > 0:
                w = vals / s
                hhi = float((w ** 2).sum())
                out['capacity_shares'] = dict(zip(cap_cols, w.tolist()))
                out['hhi_capacity'] = hhi
                self.analysis_results['diversification'] = out
                return out

        # Fallback: use absolute investor actions over the run if columns exist
        inv_cols = [c for c in self.df.columns if c.startswith('inv_action_')]
        if inv_cols:
            agg = self.df[inv_cols].abs().sum()
            s = float(agg.sum())
            if s > 0:
                w = (agg / s).values
                hhi = float((w ** 2).sum())
                out['action_share'] = {k: float((v / s)) for k, v in agg.items()}
                out['hhi_action'] = hhi

        self.analysis_results['diversification'] = out
        return out

    # ---------- Plotting ----------

    def _x_axis(self, n: int) -> List:
        """Prefer timestamp for x-axis; else use range(n)."""
        if 'timestamp' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            return self.df['timestamp'].iloc[-n:].tolist() if n < len(self.df) else self.df['timestamp'].tolist()
        return list(range(n))

    def create_comprehensive_plots(self) -> None:
        """
        Saves static PNG plots into self.out_dir and an interactive dashboard (if Plotly available).
        """
        os.makedirs(self.out_dir, exist_ok=True)
        rets = self.returns_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        curve = (1.0 + rets).cumprod()

        # Time series: returns (per step)
        x = self._x_axis(len(rets))
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(x, rets.values)
        plt.title("Portfolio Returns (per step)")
        plt.xlabel("time" if len(x) > 0 and isinstance(x[0], (np.datetime64, pd.Timestamp)) else "timestep")
        plt.ylabel("return")
        plt.tight_layout()
        fig1_path = os.path.join(self.out_dir, "returns.png")
        fig1.savefig(fig1_path); plt.close(fig1)

        # Equity curve
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(x, curve.values)
        plt.title("Equity Curve (cumulative product of 1+returns)")
        plt.xlabel("time" if len(x) > 0 and isinstance(x[0], (np.datetime64, pd.Timestamp)) else "timestep")
        plt.ylabel("equity")
        plt.tight_layout()
        fig2_path = os.path.join(self.out_dir, "equity_curve.png")
        fig2.savefig(fig2_path); plt.close(fig2)

        # Price vs battery action (if available)
        price_col = 'price_actual' if 'price_actual' in self.df.columns else ('price' if 'price' in self.df.columns else None)
        if price_col and 'batt_action_0' in self.df.columns:
            price = pd.to_numeric(self.df[price_col], errors='coerce')
            batt = pd.to_numeric(self.df['batt_action_0'], errors='coerce')
            t = self._x_axis(len(self.df))
            fig3 = plt.figure(figsize=(10, 5))
            ax1 = plt.gca()
            ax1.plot(t, price.values, label='price')
            ax1.set_ylabel('price')
            ax2 = ax1.twinx()
            ax2.plot(t, batt.values, alpha=0.5, label='batt_action_0')
            ax2.set_ylabel('battery action')
            ax1.set_title("Price vs Battery Action")
            fig3.tight_layout()
            fig3_path = os.path.join(self.out_dir, "price_vs_battery.png")
            fig3.savefig(fig3_path); plt.close(fig3)

        # Interactive dashboard
        self._create_interactive_dashboard(rets, curve)

    def _create_interactive_dashboard(self, returns: pd.Series, equity_curve: pd.Series) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        if not _HAS_PLOTLY:
            print("ℹ️ Plotly not installed; skipping interactive dashboard.")
            return

        # x-axis
        if 'timestamp' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            x = self.df['timestamp'].astype(str).tolist()
        else:
            x = list(range(len(returns)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=returns.tolist(), name="Returns"))
        fig.add_trace(go.Scatter(x=x, y=equity_curve.tolist(), name="Equity", yaxis="y2"))
        fig.update_layout(
            title="Portfolio Dashboard",
            xaxis=dict(title="time" if isinstance(x[0], str) else "timestep"),
            yaxis=dict(title="return"),
            yaxis2=dict(title="equity", overlaying='y', side='right'),
            legend=dict(orientation='h')
        )
        out_path = os.path.join(self.out_dir, "interactive_dashboard.html")
        plotly_plot(fig, filename=out_path, auto_open=False)
        print(f"✅ Interactive dashboard saved to: {out_path}")

    # ---------- Text report ----------

    def generate_comprehensive_report(self) -> str:
        """
        Writes a human-readable text report and returns its path.
        """
        lines: List[str] = []
        lines.append("PORTFOLIO ANALYSIS REPORT")
        lines.append("=" * 32)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Rows: {len(self.df):,}")
        lines.append("")

        src = self.analysis_results.get('returns_source', 'unknown')
        ann = self.analysis_results.get('annualization', self.annualization)
        lines.append(f"Returns source: {src}")
        lines.append(f"Annualization (steps/year): {ann}")
        lines.append("")

        perf = self.analysis_results.get('performance', {})
        if perf:
            lines.append("Performance metrics:")
            for k, v in perf.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")

        risk = self.analysis_results.get('risk', {})
        if risk:
            lines.append("Risk summary:")
            for k, v in risk.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")

        fc = self.analysis_results.get('forecast_utilization', {})
        if fc:
            lines.append("Forecast utilization (corr):")
            for k, v in fc.items():
                lines.append(f"  - {k}: {v:.3f}")
            lines.append("")

        bat = self.analysis_results.get('battery', {})
        if bat:
            lines.append("Battery effectiveness:")
            for k, v in bat.items():
                lines.append(f"  - {k}: {v:.3f}")
            lines.append("")

        div = self.analysis_results.get('diversification', {})
        if div:
            lines.append("Diversification:")
            for k, v in div.items():
                if isinstance(v, dict):
                    lines.append(f"  - {k}:")
                    for kk, vv in v.items():
                        lines.append(f"      • {kk}: {vv:.4f}")
                else:
                    lines.append(f"  - {k}: {v}")
            lines.append("")

        out_path = os.path.join(self.out_dir, "report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ Report saved to: {out_path}")
        return out_path


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze portfolio logs from the RL+DL system.")
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to enhanced_metrics_*.csv (if omitted, the latest one is used).")
    ap.add_argument("--annualization", type=int, default=None,
                    help="Override steps/year (e.g., 365*144 for 10-minute steps = 52560).")
    args = ap.parse_args()

    csv_path = args.csv or find_latest_log_file()
    if not csv_path or not os.path.exists(csv_path):
        print("❌ Could not find a valid log CSV. Provide --csv path or place enhanced_metrics_*.csv in a known folder.")
        return

    print(f"📦 Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    analyzer = AdvancedPortfolioAnalyzer(df, annualization_override=args.annualization)
    analyzer.calculate_advanced_metrics()
    analyzer.analyze_forecast_utilization()
    analyzer.analyze_battery_effectiveness()
    analyzer.analyze_strategy_diversification()
    analyzer.create_comprehensive_plots()
    analyzer.generate_comprehensive_report()

    print("🎉 Analysis complete.")


if __name__ == "__main__":
    main()
