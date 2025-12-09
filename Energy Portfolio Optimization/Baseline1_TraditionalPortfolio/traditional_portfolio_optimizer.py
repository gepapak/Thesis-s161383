#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional Portfolio Optimizer - Improved Version

Implements classical portfolio optimization using actual renewable energy data:
- Equal Weight
- Minimum Variance
- Maximum Sharpe Ratio
- Markowitz Mean-Variance (with risk aversion)
- Risk Parity

Uses actual revenue from renewable generation (price × quantity) for realistic returns.

Based on Modern Portfolio Theory and classical financial optimization.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from math import sqrt
from scipy.optimize import minimize

# ----------------------------- Logging ------------------------------------- #

logger = logging.getLogger("baseline.traditional")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------- Config ------------------------------------- #

@dataclass
class Timebase:
    """Timebase configuration (hours per step)."""
    time_step_hours: float = 10.0 / 60.0  # default: 10-minute steps

    @property
    def steps_per_day(self) -> int:
        return int(round(24.0 / self.time_step_hours))

    @property
    def steps_per_year(self) -> int:
        return self.steps_per_day * 365

    def rf_per_step(self, rf_annual: float) -> float:
        """Convert annual risk-free rate to per-step rate via geometric compounding."""
        return (1.0 + rf_annual) ** (self.time_step_hours / (24.0 * 365.0)) - 1.0


# ----------------------------- Utilities ----------------------------------- #

def _shrink_cov_identity(cov: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Simple Ledoit-Wolf style shrinkage towards identity."""
    n = cov.shape[0]
    diag = np.trace(cov) / n
    identity = np.eye(n) * diag
    return (1 - shrinkage) * cov + shrinkage * identity


def _risk_parity_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 10_000) -> np.ndarray:
    """
    Solve risk parity via cyclic coordinate descent on log-weights.
    Ensures w>=0 and sum w = 1.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    target = (w @ cov @ w) / n  # equal risk contributions target

    for _ in range(max_iter):
        rc = w * (cov @ w)  # risk contributions
        err = rc - target
        if np.linalg.norm(err, ord=1) < tol:
            break
        # update rule
        w *= target / np.clip(rc, 1e-16, None)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s == 0:
            w = np.ones(n) / n
        else:
            w /= s
        # refresh target
        port_var = max(w @ cov @ w, 1e-16)
        target = port_var / n
    return w


def _slsqp_box_simplex(
    objective: Callable[[np.ndarray], float],
    grad: Optional[Callable[[np.ndarray], np.ndarray]],
    n: int,
    lb: float = 0.0,
    ub: float = 1.0,
    w0: Optional[np.ndarray] = None,
    bounds_override: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Minimize objective(w) with SLSQP subject to sum w = 1 and lb<=w<=ub,
    with optional per-asset bounds.
    """
    if bounds_override is not None:
        bounds = bounds_override
    else:
        bounds = [(lb, ub) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if w0 is None:
        w0 = np.ones(n) / n
    res = minimize(
        objective, w0, jac=grad, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 200, "ftol": 1e-9, "disp": False}
    )
    if not res.success or not np.isfinite(res.fun):
        logger.warning("SLSQP did not converge (%s) — falling back to equal weight.", res.message)
        return np.ones(n) / n
    w = res.x
    w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
    w /= w.sum() if w.sum() != 0 else 1.0
    return w


# --------------------------- Optimizer Config ------------------------------- #

@dataclass
class OptimizerConfig:
    method: str = "markowitz_mean_variance"
    risk_aversion_lambda: float = 5.0   # only for markowitz_mean_variance
    shrinkage: float = 0.1
    allow_short: bool = False
    seed: int = 42


# --------------------------- Main Optimizer Class --------------------------- #

class TraditionalPortfolioOptimizer:
    """
    Rolling-window optimizer for hybrid infrastructure fund.

    Tradable assets:
    - Physical: wind, solar, hydro (operational yields)
    - Financial: price (MTM returns, capped at 12% weight)
    - Cash: risk-free asset

    Implements classical portfolio optimization with proper hybrid fund structure.
    """

    risky_assets: Tuple[str, ...] = ("wind", "solar", "hydro", "price")

    def __init__(
        self,
        timebase: Timebase,
        rf_annual: float = 0.02,
        opt_cfg: Optional[OptimizerConfig] = None,
        initial_budget: float = 800_000_000.0,  # $800M USD
        dkk_to_usd_rate: float = 0.145,
        lookback_window: Optional[int] = None,
        rebalance_freq: Optional[int] = None,
    ):
        self.timebase = timebase
        self.rf_annual = rf_annual
        self.rf_step = timebase.rf_per_step(rf_annual)
        self.opt_cfg = opt_cfg or OptimizerConfig()
        np.random.seed(self.opt_cfg.seed)

        # Portfolio parameters
        self.initial_budget_usd = initial_budget
        self.current_budget_usd = initial_budget
        self.dkk_to_usd_rate = dkk_to_usd_rate

        # Time-aware defaults from Timebase
        self.rebalance_freq = rebalance_freq or max(1, self.timebase.steps_per_year // 12)  # ~monthly
        self.lookback_window = lookback_window or self.timebase.steps_per_year  # ~1 year

        # Weights: 3 risky assets + cash
        self.n_assets = len(self.risky_assets) + 1
        self.weights = np.ones(self.n_assets) / self.n_assets  # Equal weight start
        
        # Performance tracking
        self.portfolio_values = []
        self.returns_history = []
        
        # Selection map to functions
        self._methods: Dict[str, Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = {
            "equal_weight": self._equal_weight,
            "min_variance": self._min_variance,
            "max_sharpe": self._max_sharpe,
            "markowitz_mean_variance": self._markowitz_mv,
            "risk_parity": self._risk_parity,
        }

    # ------------------ Returns construction (core accuracy) ---------------- #

    def build_asset_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build asset returns for a hybrid fund *correctly separated* into:
        - Operational yields for wind/solar/hydro (no price drift baked in)
        - A separate 'price' asset that captures mark-to-market price return

        This lets the optimizer allocate at most 12% to price (via bounds),
        while physical sleeves remain driven by operations.
        """
        req = {"timestamp", "wind", "solar", "hydro", "price"}
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- Physical revenue (ownership & opex) ---
        # FIXED: Match MARL economic model - use actual revenue without artificial scaling
        wind_own, solar_own, hydro_own = 0.18, 0.10, 0.04
        wind_opex, solar_opex, hydro_opex = 0.05, 0.03, 0.08

        price = df["price"].astype(float)
        wind_rev  = df["wind"].astype(float)  * wind_own  * price * (1 - wind_opex)
        solar_rev = df["solar"].astype(float) * solar_own * price * (1 - solar_opex)
        hydro_rev = df["hydro"].astype(float) * hydro_own * price * (1 - hydro_opex)

        # Physical capital base (DKK), derived from fund config (88% of $800M)
        physical_capex_dkk = (800_000_000 * 0.88) / 0.145

        # Revenue scaling factor to match MARL's comprehensive cost accounting
        # MARL deducts: variable costs (2.5%), maintenance (~24 DKK/MWh), grid fees (~3.5 DKK/MWh),
        # transmission (~8.3 DKK/MWh), insurance (0.4% annual), property tax (0.5% annual),
        # debt service (1.5% annual), management fees (1% annual)
        # This scaling factor calibrates baseline's simple (1-opex) model to match MARL's net revenue
        revenue_scaling = 0.0216

        def op_yield(rev: pd.Series) -> pd.Series:
            # Scale revenue to match MARL's net revenue after comprehensive costs
            scaled = rev * revenue_scaling
            y = (scaled / physical_capex_dkk).fillna(0.0).astype(float)
            return y

        wind_r  = op_yield(wind_rev)
        solar_r = op_yield(solar_rev)
        hydro_r = op_yield(hydro_rev)

        # --- Financial sleeve as its own asset ---
        # Use pct_change with cap ±0.1% per 10-min
        price_ret = price.pct_change().fillna(0.0).clip(-0.001, 0.001)

        return pd.DataFrame({
            "timestamp": df["timestamp"],
            "wind":  wind_r.values,
            "solar": solar_r.values,
            "hydro": hydro_r.values,
            "price": price_ret.values,   # separate sleeve
        })

    # ---------------------------- Optimization Methods ---------------------------------- #

    def _equal_weight(self, mu: np.ndarray, cov: np.ndarray, rf_step: float) -> np.ndarray:
        n = len(mu)
        if n == 4:  # wind, solar, hydro, price
            # Cap price at 12%, distribute rest equally among physical assets
            w_price = 0.12
            w_physical_each = (1.0 - w_price) / 3  # ~29.33% each for wind/solar/hydro
            w_risky = np.array([w_physical_each, w_physical_each, w_physical_each, w_price])
        else:
            w_risky = np.ones(n) / n
        return w_risky

    def _min_variance(self, mu: np.ndarray, cov: np.ndarray, rf_step: float) -> np.ndarray:
        n = len(mu)
        cov = _shrink_cov_identity(cov, self.opt_cfg.shrinkage)
        # Per-asset bounds: (0,1) for physicals, (0,0.12) for price
        bounds = [(0.0, 1.0) for _ in range(n)]
        if n == 4:  # wind, solar, hydro, price
            bounds[-1] = (0.0, 0.12)  # cap price at 12%
        def obj(w):
            return float(w @ cov @ w)
        return _slsqp_box_simplex(obj, None, n, bounds_override=bounds)

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, rf_step: float) -> np.ndarray:
        n = len(mu)
        cov = _shrink_cov_identity(cov, self.opt_cfg.shrinkage)
        # Per-asset bounds: (0,1) for physicals, (0,0.12) for price
        bounds = [(0.0, 1.0) for _ in range(n)]
        if n == 4:  # wind, solar, hydro, price
            bounds[-1] = (0.0, 0.12)  # cap price at 12%
        def obj(w):
            port_mu = float(mu @ w)
            port_var = max(float(w @ cov @ w), 1e-16)
            port_vol = sqrt(port_var)
            # Maximize (port_mu - rf) / vol  == minimize negative
            sharpe = (port_mu - rf_step) / port_vol
            return -sharpe if np.isfinite(sharpe) else 1e6
        return _slsqp_box_simplex(obj, None, n, bounds_override=bounds)

    def _markowitz_mv(self, mu: np.ndarray, cov: np.ndarray, rf_step: float) -> np.ndarray:
        """
        Mean-variance with risk_aversion_lambda: minimize 0.5 * lambda * w^T C w - (mu - rf)·w
        """
        n = len(mu)
        lam = float(self.opt_cfg.risk_aversion_lambda)
        cov = _shrink_cov_identity(cov, self.opt_cfg.shrinkage)
        adj_mu = mu - rf_step  # excess returns
        # Per-asset bounds: (0,1) for physicals, (0,0.12) for price
        bounds = [(0.0, 1.0) for _ in range(n)]
        if n == 4:  # wind, solar, hydro, price
            bounds[-1] = (0.0, 0.12)  # cap price at 12%
        def obj(w):
            return 0.5 * lam * float(w @ cov @ w) - float(adj_mu @ w)
        return _slsqp_box_simplex(obj, None, n, bounds_override=bounds)

    def _risk_parity(self, mu: np.ndarray, cov: np.ndarray, rf_step: float) -> np.ndarray:
        cov = _shrink_cov_identity(cov, self.opt_cfg.shrinkage)
        w = _risk_parity_weights(cov)
        # Cap price at 12% and renormalize
        if len(w) == 4:  # wind, solar, hydro, price
            w[-1] = min(w[-1], 0.12)
            w /= w.sum()
        return w

    # ------------------------ Rolling optimization ------------------------- #

    def rebalance_weights(
        self,
        returns_window: pd.DataFrame,
        method: str,
    ) -> Dict[str, float]:
        method = method.lower()
        if method not in self._methods:
            raise ValueError(f"Unknown method '{method}'. Valid: {list(self._methods)}")

        # assemble mu/cov for risky assets
        R = returns_window[list(self.risky_assets)].dropna()
        if len(R) < max(30, len(self.risky_assets) * 3):
            # not enough data => equal-weight risky + rest cash
            w_risky = np.ones(len(self.risky_assets)) / len(self.risky_assets)
        else:
            mu = R.mean(axis=0).values  # arithmetic mean per step
            cov = np.cov(R.values, rowvar=False)
            w_risky = self._methods[method](mu, cov, self.rf_step)

        # combine with cash as residual
        w_risky = np.clip(w_risky, 0.0, 1.0)
        w_cash = 1.0 - float(np.sum(w_risky))
        if w_cash < 0 and not self.opt_cfg.allow_short:
            # normalize to simplex if numerical drift
            s = np.sum(np.clip(w_risky, 0.0, 1.0))
            w_risky = np.clip(w_risky, 0.0, 1.0) / (s if s > 0 else 1.0)
            w_cash = 1.0 - float(np.sum(w_risky))
            w_cash = max(w_cash, 0.0)

        weights: Dict[str, float] = {a: float(w) for a, w in zip(self.risky_assets, w_risky)}
        weights["cash"] = float(w_cash)
        return weights

    # ------------------------ Portfolio Execution ------------------------- #

    def step(self, data: pd.DataFrame, t: int, returns_df: pd.DataFrame) -> Dict:
        """
        Execute one portfolio step.

        Args:
            data: Full market data DataFrame
            t: Current timestep
            returns_df: Pre-computed returns DataFrame

        Returns:
            dict: Portfolio metrics and state
        """
        # Get current returns (if available)
        if t >= len(returns_df):
            current_returns = np.zeros(len(self.risky_assets))
        else:
            current_returns = returns_df.iloc[t][list(self.risky_assets)].values

        # Rebalance if needed
        if t > 0 and t % self.rebalance_freq == 0 and len(returns_df) >= self.lookback_window:
            # Get lookback window
            start_idx = max(0, t - self.lookback_window)
            returns_window = returns_df.iloc[start_idx:t]

            # Rebalance weights
            try:
                weights_dict = self.rebalance_weights(returns_window, self.opt_cfg.method)
                # Convert to array in risky_assets order + cash: [wind, solar, hydro, price, cash]
                self.weights = np.array(
                    [weights_dict[a] for a in self.risky_assets] + [weights_dict["cash"]]
                )
            except Exception as e:
                logger.warning(f"Rebalancing failed at step {t}: {e}")
                # Keep current weights

        # Calculate portfolio return
        # Risky assets return
        risky_weights = self.weights[:len(self.risky_assets)]
        risky_return = np.sum(risky_weights * current_returns)

        # Cash return (risk-free rate)
        cash_weight = self.weights[-1]
        cash_return = cash_weight * self.rf_step

        # Total portfolio return
        portfolio_return = risky_return + cash_return

        # Update portfolio value
        self.current_budget_usd *= (1 + portfolio_return)
        self.portfolio_values.append(self.current_budget_usd)
        self.returns_history.append(portfolio_return)

        # Calculate metrics
        metrics = self._calculate_metrics()

        return {
            'portfolio_value': self.current_budget_usd,
            'weights': self.weights.copy(),
            'returns': portfolio_return,
            'metrics': metrics
        }

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            }

        values = np.array(self.portfolio_values)
        returns = np.array(self.returns_history)

        # Total return
        total_return = (self.current_budget_usd - self.initial_budget_usd) / self.initial_budget_usd

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            excess_returns = returns - self.rf_step
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.timebase.steps_per_year)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(self.timebase.steps_per_year)
        else:
            volatility = 0.0

        # Calmar ratio
        if abs(max_drawdown) > 1e-6:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio
        }

    def get_summary(self) -> Dict[str, float]:
        """Get final portfolio summary."""
        metrics = self._calculate_metrics()

        return {
            'final_value': self.current_budget_usd,
            'final_value_usd': self.current_budget_usd,
            'initial_value_usd': self.initial_budget_usd,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'volatility': metrics['volatility'],
            'calmar_ratio': metrics['calmar_ratio'],
            'final_weights': self.weights.copy(),
            'optimization_method': self.opt_cfg.method
        }

