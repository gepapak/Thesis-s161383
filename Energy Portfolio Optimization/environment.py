# environment.py
# FULLY PATCHED HYBRID MODEL VERSION - Complete Original Functionality

"""
Multi-agent renewable energy investment environment with hybrid economic model.

HYBRID FUND STRUCTURE ($800M Total Capital):
==============================================================================
ECONOMIC MODEL: Clear separation between physical ownership and financial trading

1) PHYSICAL OWNERSHIP ($704M deployed - 88% allocation):
   - Wind farms: 270 MW ($540M) - Fractional ownership: 18% of 1,500MW wind farm
   - Solar farms: 100 MW ($100M) - Fractional ownership: 10% of 1,000MW solar farm
   - Hydro plants: 40 MW ($60M) - Fractional ownership: 4% of 1,000MW hydro plant
   - Battery storage: 10 MWh ($4M) - Direct ownership
   - Total: 420 MW physical capacity generating real electricity

2) FINANCIAL TRADING ($96M allocated - 12% allocation):
   - Renewable energy index derivatives
   - Wind/solar/hydro futures contracts
   - Energy storage arbitrage instruments
   - Mark-to-market positions (not physical assets)

KEY FEATURES:
- Physical assets generate actual electricity revenue
- Financial instruments provide additional exposure and hedging
- AI optimizes both operational decisions and trading strategies
- Optional ANN forecast-cache utilization adjusts investor execution only
- Multi-agent environment supports different investment strategies

This environment simulates realistic renewable energy fund operations with:
- Comprehensive risk management across both asset classes
- Portfolio optimization using deep learning
- Performance tracking and enhanced metrics
- Fixed observation contract with no forecast feature expansion
"""

from __future__ import annotations

# ---- Robust PettingZoo import for parallel API across versions ----
try:
    from pettingzoo.utils import ParallelEnv  # modern
except Exception:  # pragma: no cover
    from pettingzoo import ParallelEnv        # older fallback

from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from collections import deque
import os, gc, traceback, math, json
try:
    import psutil as _psutil
except Exception:
    _psutil = None

from risk import EnhancedRiskController
from utils import SafeDivision, UnifiedMemoryManager, ErrorHandler, safe_operation  # UNIFIED: Import from single source of truth
from config import (
    normalize_price,
    BASE_FEATURE_DIM,
    ENV_MARKET_STRESS_DEFAULT,
    ENV_OVERALL_RISK_DEFAULT,
    ENV_MARKET_RISK_DEFAULT,
    ENV_POSITION_EXPOSURE_THRESHOLD,
    ENV_EXPLORATION_BONUS_MULTIPLIER,
)  # UNIFIED: Import from single source of truth
from logger import RewardLogger, get_logger  # Step-by-step logging for Tier comparison

# Centralized logging - ALL logging goes through logger.py
logger = get_logger(__name__)

from observation_builder import ObservationBuilder as ProductionObservationBuilder


def _p95_robust(values: np.ndarray, min_scale: float = 0.1) -> float:
    """Robust positive scale estimate used by rolling_past and global normalization."""
    try:
        arr = np.asarray(values, dtype=float)
        p95_val = float(np.nanpercentile(arr, 95))
        if math.isfinite(p95_val) and p95_val > 0.0:
            return max(p95_val, float(min_scale))
    except Exception:
        pass
    return max(float(min_scale), 1.0)


def _init_running_moments(
    existing_state: Optional[Dict[str, Any]],
    prior_mean: float,
    prior_std: float,
    prior_count: float,
) -> Tuple[float, float, float]:
    """Initialize persistent online mean/variance state."""
    if isinstance(existing_state, dict):
        try:
            count = float(existing_state.get("count", 0.0))
            mean = float(existing_state.get("mean", prior_mean))
            m2 = float(existing_state.get("m2", 0.0))
            if count > 0.0 and math.isfinite(mean) and math.isfinite(m2) and m2 >= 0.0:
                return count, mean, m2
        except Exception:
            pass

    count = max(float(prior_count), 1.0)
    mean = float(prior_mean)
    std = max(abs(float(prior_std)), 1e-6)
    m2 = (std * std) * count
    return count, mean, m2


def _online_normalize_with_state(
    values: np.ndarray,
    existing_state: Optional[Dict[str, Any]],
    *,
    prior_mean: float,
    prior_std: float,
    prior_count: float,
    std_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Normalize a series using only past information plus persistent prior state."""
    arr = np.asarray(values, dtype=float)
    count, mean, m2 = _init_running_moments(existing_state, prior_mean, prior_std, prior_count)
    floor = max(float(std_floor), 1e-6)

    normalized = np.zeros(len(arr), dtype=float)
    mean_arr = np.zeros(len(arr), dtype=float)
    std_arr = np.zeros(len(arr), dtype=float)

    for idx, raw_val in enumerate(arr):
        denom = max(math.sqrt(max(m2, 0.0) / max(count, 1.0)), floor)
        mean_arr[idx] = mean
        std_arr[idx] = denom

        if math.isfinite(raw_val):
            z_val = (float(raw_val) - mean) / denom
        else:
            z_val = 0.0
        normalized[idx] = float(np.clip(z_val, -3.0, 3.0) / 3.0)

        if math.isfinite(raw_val):
            count_new = count + 1.0
            delta = float(raw_val) - mean
            mean = mean + (delta / count_new)
            delta2 = float(raw_val) - mean
            m2 = max(m2 + (delta * delta2), 0.0)
            count = count_new

    state = {"count": float(count), "mean": float(mean), "m2": float(max(m2, 0.0))}
    return normalized, mean_arr, std_arr, state


def _resolve_persistent_scale(
    config: Any,
    attr_name: str,
    episode_values: np.ndarray,
    *,
    global_value: Optional[float],
    alpha: float,
    min_scale: float = 0.1,
) -> float:
    """
    Return the carried-forward scale for the current episode and update config
    with a smoothed next-episode scale using current episode data.
    """
    prior_scale = None
    if config is not None:
        try:
            prior_scale = float(getattr(config, attr_name))
        except Exception:
            prior_scale = None

    if prior_scale is None or (not math.isfinite(prior_scale)) or prior_scale <= 0.0:
        if global_value is not None and math.isfinite(float(global_value)) and float(global_value) > 0.0:
            prior_scale = float(global_value)
        else:
            prior_scale = _p95_robust(episode_values, min_scale=min_scale)

    episode_scale = _p95_robust(episode_values, min_scale=min_scale)
    blend = float(np.clip(alpha, 0.0, 1.0))
    next_scale = ((1.0 - blend) * float(prior_scale)) + (blend * float(episode_scale))

    if config is not None:
        setattr(config, attr_name, float(max(next_scale, min_scale)))

    return float(max(prior_scale, min_scale))


# =============================================================================
# Observation specs (Tier-1 only; no forecast features in observations)
# =============================================================================
class StabilizedObservationManager:
    def __init__(self, env: 'RenewableMultiAgentEnv'):
        self.env = env
        self.config = env.config if hasattr(env, 'config') else None
        self.observation_specs = self._build_specs()
        self.base_spaces = self._build_spaces()

    def _build_specs(self) -> Dict[str, Dict[str, Any]]:
        # Investor now 12D (was 9D): added step_pnl_return, medium_momentum, intraday_sin, intraday_cos.
        return {
            "investor_0": {"base": 12},
            "battery_operator_0": {"base": 12},
            "risk_controller_0": {"base": 12},
            "meta_controller_0": {"base": 12},
        }

    def _build_spaces(self) -> Dict[str, spaces.Box]:
        sp: Dict[str, spaces.Box] = {}

        # Investor (12D): price_momentum, realized_volatility, budget,
        # aggregate_exposure, mtm_pnl_norm, is_decision_step,
        # capital_allocation_fraction, risk_exposure_cap, local_drawdown,
        # step_pnl_return ([-1,1]), medium_momentum ([-1,1]),
        # intraday_sin ([-1,1]), intraday_cos ([-1,1])
        inv_low  = np.array([-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        inv_high = np.array([ 1.0, 1.0, 1.0,  1.0,  1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0], dtype=np.float32)

        # Battery (12D): price, soc, charge/discharge headroom, short-horizon
        # price-edge signals, load, and cyclical intraday/intraweek features.
        bat_low = np.array(
            [-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
        bat_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )

        # Risk controller (12D): price_n, vol, stress, positions (3), cap_frac, equity_n,
        # risk_mult_n, overall_risk_snapshot, drawdown, is_decision_step â€” all in [-1,1] or [0,1]
        risk_low = np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        risk_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Meta controller (12D): budget_n, positions (3), price_n, overall_risk, perf_n,
        # market_risk, vol, stress, cap_frac, is_decision_step â€” all in [-1,1] or [0,1]
        meta_low = np.array([0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        meta_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        sp["investor_0"] = spaces.Box(low=inv_low, high=inv_high, shape=(12,), dtype=np.float32)
        sp["battery_operator_0"] = spaces.Box(low=bat_low, high=bat_high, shape=(12,), dtype=np.float32)
        sp["risk_controller_0"] = spaces.Box(low=risk_low, high=risk_high, shape=(12,), dtype=np.float32)
        sp["meta_controller_0"] = spaces.Box(low=meta_low, high=meta_high, shape=(12,), dtype=np.float32)

        return sp

    def obs_space(self, agent: str) -> spaces.Box:
        return self.base_spaces[agent]

    def base_dim(self, agent: str) -> int:
        return self.observation_specs[agent]["base"]


# =============================================================================
# FIXED: Profit-focused reward calculation with proper separation
# =============================================================================
class ProfitFocusedRewardCalculator:
    def __init__(self, initial_budget: float, target_annual_return: float = 0.05, config=None):
        self.initial_budget = float(max(1.0, initial_budget))
        self.target_annual_return = target_annual_return  # REALISTIC: 5% target for institutional fund
        self.config = config  # Store config for access to parameters

        # FIXED: Separate tracking for different value sources
        self.portfolio_history = deque(maxlen=252)  # Fund NAV history
        self.cash_flow_history = deque(maxlen=252)  # Actual cash flow history
        self.return_history = deque(maxlen=252)
        # Recent per-step trading MTM samples for reward shaping / diagnostics.
        self.trading_mtm_history = deque(maxlen=20)
        self.profit_history = deque(maxlen=100)  # Track recent profits

        # REALISTIC: Conservative profitability thresholds for institutional fund
        self.min_acceptable_return = 0.02  # 2% minimum annual return (above risk-free)
        self.target_return_threshold = 0.05  # 5% target return threshold
        self.excellent_return_threshold = 0.08  # 8% excellent return threshold (realistic for renewables)

        # Portfolio tracking (no emergency liquidation for infrastructure fund)
        self.portfolio_peak = float(self.initial_budget)
        self.emergency_liquidation_enabled = False  # Disabled: can't liquidate wind farms instantly
        self.emergency_liquidation_threshold = 0.10  # Only for extreme cases (90% loss)

        # Technology operational volatilities (daily-ish proxies)
        self.operational_vols = {'wind': 0.03, 'solar': 0.025, 'hydro': 0.015}
        self.operational_correlations = {'wind_solar': 0.4, 'wind_hydro': 0.2, 'solar_hydro': 0.3}

        # FIXED: Use reward weights from config to maintain single source of truth
        # Paper setup: rewards are identical across baseline/FGB variants.
        # Forecasts (if any) are backend-only for variance reduction and never change env reward.

        if config and hasattr(config, 'profit_reward_weight'):
            # Use config-driven reward weights (fair by default)
            # NAV-first objective (fair across tiers): emphasize NAV stability/return,
            # keep risk + hedging as small penalties.
            # More trading/NAV growth: boost revenue, trim stability
            base_weights = {
                'operational_revenue': 0.42,
                'risk_management': 0.15,
                'hedging_effectiveness': 0.15,
                'nav_stability': 0.28,
                'cash_flow': 0.0,
            }
            self.reward_weights = {
                **base_weights,
            }
        else:
            # Non-config path mirrors logic above (fair by default)
            # NAV-first objective (fair across tiers): emphasize NAV stability/return,
            # keep risk + hedging as small penalties.
            base_weights = {
                'operational_revenue': 0.42,
                'risk_management': 0.15,
                'hedging_effectiveness': 0.15,
                'nav_stability': 0.28,
                'cash_flow': 0.0,
            }
            self.reward_weights = {
                **base_weights,
            }

        # Progressive drawdown de-risking for all tiers:
        # - above soft threshold: reduce size
        # - above medium threshold: reduce size more
        # - above hard threshold: disable trading
        base_dd = float(getattr(config, 'max_drawdown_threshold', 0.10)) if config is not None else 0.10
        self.max_drawdown_threshold = float(max(base_dd, 1e-6))
        self.soft_drawdown_threshold = float(self.max_drawdown_threshold)
        self.medium_drawdown_threshold = float(max(self.soft_drawdown_threshold * 2.0, self.soft_drawdown_threshold + 1e-6))
        self.hard_drawdown_disable_threshold = float(max(self.medium_drawdown_threshold * 1.5, self.medium_drawdown_threshold + 1e-6))
        self.current_drawdown = 0.0
        self.peak_nav = float(self.initial_budget)
        self.trading_enabled = True

        logger.info(f"[REWARD] Weights: {self.reward_weights}")

    def update_trading_mtm(self, mtm_pnl: float):
        self.trading_mtm_history.append(mtm_pnl)

    @property
    def recent_trading_mtm(self) -> float:
        if len(self.trading_mtm_history) == 0:
            return 0.0
        return float(np.mean(list(self.trading_mtm_history)))

    def calculate_reward(self, fund_nav: float, cash_flow: float,
                         risk_level: float, efficiency: float) -> float:
        fund_nav = float(fund_nav) if not isinstance(fund_nav, np.ndarray) else float(fund_nav.item() if fund_nav.size == 1 else fund_nav.flatten()[0])
        cash_flow = float(cash_flow) if not isinstance(cash_flow, np.ndarray) else float(cash_flow.item() if cash_flow.size == 1 else cash_flow.flatten()[0])
        risk_level = float(risk_level) if not isinstance(risk_level, np.ndarray) else float(risk_level.item() if risk_level.size == 1 else risk_level.flatten()[0])
        efficiency = float(efficiency) if not isinstance(efficiency, np.ndarray) else float(efficiency.item() if efficiency.size == 1 else efficiency.flatten()[0])
        self.portfolio_history.append(fund_nav)
        self.cash_flow_history.append(cash_flow)

        peak_nav_float = float(self.peak_nav) if not isinstance(self.peak_nav, np.ndarray) else float(self.peak_nav.item() if self.peak_nav.size == 1 else self.peak_nav.flatten()[0])
        if fund_nav > peak_nav_float:
            self.peak_nav = fund_nav
        peak_nav_float = float(self.peak_nav) if not isinstance(self.peak_nav, np.ndarray) else float(self.peak_nav.item() if self.peak_nav.size == 1 else self.peak_nav.flatten()[0])
        self.current_drawdown = float((peak_nav_float - fund_nav) / peak_nav_float if peak_nav_float > 0 else 0.0)

        drawdown_float = float(self.current_drawdown)
        hard_dd = float(getattr(self, 'hard_drawdown_disable_threshold', getattr(self, 'max_drawdown_threshold', 0.1)))
        medium_dd = float(getattr(self, 'medium_drawdown_threshold', max(getattr(self, 'max_drawdown_threshold', 0.1) * 2.0, 1e-6)))
        soft_dd = float(getattr(self, 'soft_drawdown_threshold', getattr(self, 'max_drawdown_threshold', 0.1)))

        if drawdown_float > hard_dd:
            self.trading_enabled = False
        elif drawdown_float > medium_dd:
            self.trading_enabled = True
        elif drawdown_float > soft_dd:
            self.trading_enabled = True
        else:
            self.trading_enabled = True

        if len(self.portfolio_history) < 2:
            return 0.0

        recent_cash_flows = list(self.cash_flow_history)[-10:]
        avg_operational_revenue = np.mean(recent_cash_flows) if recent_cash_flows else 0.0
        operational_target = getattr(self.config, 'operational_revenue_target', 1200.0) if self.config else 1200.0
        operational_score = float(np.clip(avg_operational_revenue / operational_target, -2.0, 3.0))

        if len(self.portfolio_history) >= 20:
            recent_navs = np.array(list(self.portfolio_history)[-20:], dtype=np.float64)
            nav_diff = np.diff(recent_navs)
            nav_base = recent_navs[:-1]
            nav_returns = nav_diff / np.maximum(nav_base, 1.0)
            portfolio_volatility = np.std(nav_returns) if len(nav_returns) > 1 else 0.0
        else:
            portfolio_volatility = 0.0

        volatility_penalty = float(np.clip(portfolio_volatility * 100.0, 0.0, 2.0))
        drawdown_penalty = float(np.clip(self.current_drawdown * 10.0, 0.0, 3.0))
        risk_management_score = -(volatility_penalty + drawdown_penalty)

        hedging_score = self._calculate_hedging_effectiveness()

        if len(self.portfolio_history) >= 2:
            prev_nav = float(self.portfolio_history[-2])
            nav_return = float((fund_nav - prev_nav) / max(prev_nav, 1.0))
        else:
            nav_return = 0.0

        # Align shared reward to the deployed objective without a sticky bonus for
        # already-profitable books. NAV growth carries trading upside into the
        # reward directly, so recent trading-gain level should not be added again
        # by default.
        profit_weight = float(getattr(self.config, 'profit_reward_weight', 1.0)) if self.config else 1.0
        trading_score_weight = float(getattr(self.config, 'shared_trading_score_weight', 0.0)) if self.config else 0.0
        nav_growth_score = float(np.clip(nav_return * 100.0, -2.0, 2.0))
        trading_scale = float(max(self.initial_budget * 0.001, 1.0))
        trading_score = float(np.clip(self.recent_trading_mtm / trading_scale, -2.0, 2.0))
        nav_quality_raw = float(
            profit_weight * (nav_growth_score + trading_score_weight * trading_score)
            - 0.25 * volatility_penalty
            - 0.25 * drawdown_penalty
        )
        nav_stability_score = float(np.clip(nav_quality_raw, -2.0, 2.0))

        self._combined_confidence = 0.0
        self.last_operational_score = operational_score
        self.last_risk_score = risk_management_score
        self.last_hedging_score = hedging_score
        self.last_nav_stability_score = nav_stability_score

        rw = self.reward_weights
        reward = float(
            rw['operational_revenue'] * operational_score +
            rw['risk_management'] * risk_management_score +
            rw['hedging_effectiveness'] * hedging_score +
            rw['nav_stability'] * nav_stability_score
        )

        # NOTE (fair comparison):
        # All tiers train/evaluate on the same economic reward signal.

        if hasattr(self, 'config') and self.config:
            clip_min = getattr(self.config, 'reward_clip_min', -10.0)
            clip_max = getattr(self.config, 'reward_clip_max', 10.0)
        else:
            clip_min, clip_max = -10.0, 10.0

        reward = float(np.clip(reward, clip_min, clip_max))
        self.last_reward_components = {
            'operational_revenue': operational_score,
            'risk_management': risk_management_score,
            'hedging_effectiveness': hedging_score,
            'nav_stability': nav_stability_score,
            'nav_growth': nav_growth_score,
            'trading_performance': trading_score,
            'warmup_factor': 1.0,
        }

        return reward

    def _calculate_hedging_effectiveness(self) -> float:
        try:
            if not hasattr(self, 'hedge_effectiveness_tracker'):
                self.hedge_effectiveness_tracker = {
                    'ops_returns': deque(maxlen=200),
                    'trading_returns': deque(maxlen=200),
                    'effectiveness_multiplier': 1.0,
                    'last_score': 0.0,
                }

            tracker = self.hedge_effectiveness_tracker
            ops_returns = tracker['ops_returns']
            trading_returns = tracker['trading_returns']

            if hasattr(self, 'last_generation_revenue'):
                ops_returns.append(float(self.last_generation_revenue))
            if hasattr(self, 'last_mtm_pnl'):
                trading_returns.append(float(self.last_mtm_pnl))

            if len(ops_returns) < 5 or len(trading_returns) < 5:
                return 0.0

            ops_returns_array = np.array(ops_returns, dtype=np.float64)
            trading_returns_array = np.array(trading_returns, dtype=np.float64)

            ops_volatility = np.std(ops_returns_array)
            trading_volatility = np.std(trading_returns_array)

            if ops_volatility < 1e-6 and trading_volatility < 1e-6:
                return 0.0

            ops_returns_norm = (ops_returns_array - np.mean(ops_returns_array)) / np.maximum(ops_volatility, 1e-6)
            trading_returns_norm = (trading_returns_array - np.mean(trading_returns_array)) / np.maximum(trading_volatility, 1e-6)

            corr = float(np.corrcoef(ops_returns_norm, trading_returns_norm)[0, 1]) if len(ops_returns_norm) > 1 else 0.0
            corr = np.clip(corr, -1.0, 1.0)

            hedging_score = -corr

            tracker['effectiveness_multiplier'] = float(np.clip(tracker['effectiveness_multiplier'] * 0.99 + hedging_score * 0.01, 0.5, 1.2))
            score = float(np.clip(hedging_score * tracker['effectiveness_multiplier'], -3.0, 3.0))
            tracker['last_score'] = score
            return score
        except Exception as e:
            logger.warning(f"Hedging effectiveness calculation failed: {e}")
            tracker = getattr(self, 'hedge_effectiveness_tracker', {})
            try:
                fallback = float(np.clip(float(tracker.get('last_score', 0.0)), -3.0, 3.0))
            except Exception:
                fallback = 0.0
            return fallback

# =============================================================================
# HYBRID MODEL Environment with Complete Original Functionality
# =============================================================================
class RenewableMultiAgentEnv(ParallelEnv):
    """
    HYBRID MODEL: Physical Assets + Financial Instruments with Clear Separation

    Architecture:
    1. PHYSICAL LAYER: Actual renewable assets (wind farms, solar plants, etc.)
       - Fixed capacity in MW after purchase
       - Generate actual electricity sold at market prices
       - Have operational costs and maintenance

    2. FINANCIAL LAYER: Tradeable instruments for price exposure
       - wind_instrument_value: Financial position value (can be negative)
       - Mark-to-market with price movements
       - Used for additional price exposure beyond physical generation

    3. CLEAR SEPARATION: Physical assets â‰  Financial instruments
    """
    metadata = {"name": "renewable_hybrid_fund:v1"}

    # ----- meta/risk knob ranges used by meta controller -----
    META_FREQ_MIN = 6       # every hour if 10-min data
    META_FREQ_MAX = 24      # FIXED: Max 4 hours (was daily=288) to ensure active trading
    META_CAP_MIN  = 0.10    # USER SPECIFIED: 10% minimum capital allocation
    META_CAP_MAX  = 0.75    # USER SPECIFIED: 75% maximum capital allocation
    SAT_EPS       = 1e-3

    def __init__(
        self,
        data: pd.DataFrame,
        investment_freq: int = 6,
        enhanced_risk_controller: bool = True,
        config: Optional[Any] = None,  # Enhanced config object
        init_budget: Optional[float] = None,  # Override config if provided
        max_memory_mb: Optional[float] = None,  # Override config if provided
        initial_asset_plan: Optional[dict] = None,
        asset_capex: Optional[dict] = None,
        log_dir: Optional[str] = None,  # NEW: Optional log directory (defaults to "debug_logs")
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.max_steps = int(len(self.data))
        self.investment_freq = max(1, int(investment_freq))

        # Import config if not provided
        if config is None:
            from config import EnhancedConfig
            config = EnhancedConfig()
        self.config = config

        # Use config values with optional overrides
        self.init_budget = float(init_budget) if init_budget is not None else self.config.init_budget
        self.max_memory_mb = float(max_memory_mb) if max_memory_mb is not None else self.config.max_memory_mb
        self.enhanced_risk_controller = enhanced_risk_controller

        # Randomness (seeded from config for reproducibility)
        # CRITICAL: config.seed MUST be set. An unseeded RNG produced
        # non-reproducible episode rollouts across seeds that claimed to be
        # identical (same --seed flag, same data, same weights) and silently
        # broke multi-seed Tier-1/Tier1 comparisons because the entropy
        # source was the OS clock. We fail fast instead of papering over it
        # with default_rng(None).
        env_seed = getattr(config, 'seed', None) if config else None
        if env_seed is None:
            raise ValueError(
                "RenewableMultiAgentEnv requires config.seed to be set for "
                "reproducible multi-seed comparisons. Pass --seed <int> to main.py "
                "/ evaluation.py (or set config.seed explicitly before env construction)."
            )
        self._rng = np.random.default_rng(int(env_seed))
        self._last_seed = int(env_seed)

        # =====================================================================
        # HYBRID MODEL: CLEAR SEPARATION (FIXED)
        # =====================================================================

        # 1) PHYSICAL ASSETS (Fixed after purchase, generate actual electricity)
        self.physical_assets = {
            'wind_capacity_mw': 0.0,     # MW of actual wind farms owned
            'solar_capacity_mw': 0.0,    # MW of actual solar farms owned
            'hydro_capacity_mw': 0.0,    # MW of actual hydro plants owned
            'battery_capacity_mwh': 0.0,  # MWh of actual battery storage owned
        }
        self.assets_deployed = False  # FIXED: One-time deployment flag

        # 2) FINANCIAL INSTRUMENTS (one-factor trading sleeve, split into bookkeeping sleeves)
        self.financial_positions = {
            'wind_instrument_value': 0.0,   # Bookkeeping sleeve of the shared energy-price factor
            'solar_instrument_value': 0.0,  # Bookkeeping sleeve of the shared energy-price factor
            'hydro_instrument_value': 0.0,  # Bookkeeping sleeve of the shared energy-price factor
        }
        # CORRECTNESS: Track true MTM equity separately from exposure.
        # - The investor controls one aggregate signed factor exposure.
        # - `financial_positions` stores the per-sleeve bookkeeping notional used for
        #   risk accounting and attribution (changes only on trades).
        # - `financial_mtm_positions` stores cumulative MTM value / PnL (changes only with price returns).
        self.financial_mtm_positions = {
            'wind_instrument_value': 0.0,
            'solar_instrument_value': 0.0,
            'hydro_instrument_value': 0.0,
        }

        # 3) OPERATIONAL STATE
        self.operational_state = {
            'battery_energy': 0.0,         # Current battery charge (MWh)
            'battery_discharge_power': 0.0, # Current discharge rate (MW)
        }

        # =====================================================================

        # ---- economics knobs (single source of truth) ----
        # REALISTIC RENEWABLE ENERGY FUND ECONOMICS - NOW FROM CONFIG
        self.fund_owns_assets = True     # Fund owns assets 100%, not profit-sharing
        self.electricity_markup = self.config.electricity_markup
        self.currency_conversion = self.config.currency_conversion

        # REALISTIC COST STRUCTURE: From config
        self.operating_cost_rate = self.config.operating_cost_rate
        self.maintenance_cost_mwh = self.config.maintenance_cost_mwh
        self.insurance_rate = self.config.insurance_rate
        self.management_fee_rate = self.config.management_fee_rate
        self.property_tax_rate = self.config.property_tax_rate
        self.debt_service_rate = self.config.debt_service_rate
        self.distribution_rate = self.config.distribution_rate

        # self.regulatory_compliance_rate = 0.0002  # Removed to match PrototypeTestTuned
        # self.audit_legal_rate = 0.0001           # Removed to match PrototypeTestTuned
        # self.custody_fee_rate = 0.0001           # Removed to match PrototypeTestTuned
        self.administration_fee_rate = self.config.administration_fee_rate  # From config

        # OPERATIONAL costs (apply only after asset deployment) - FIXED: Use DKK values from config
        self.performance_fee_rate = self.config.performance_fee_rate        # From config
        self.trading_cost_rate = self.config.trading_cost_rate              # From config
        self.grid_connection_fee_mwh = self.config.grid_connection_fee_mwh  # From config
        self.transmission_fee_mwh = self.config.transmission_fee_mwh        # From config

        # Battery physics/economics (PyPSA specifications: 10 MWh / 5 MW) - FIXED: Use DKK from config
        self.batt_eta_charge = self.config.batt_eta_charge        # From config
        self.batt_eta_discharge = self.config.batt_eta_discharge  # From config
        self.batt_degradation_cost = self.config.battery_degradation_cost_mwh  # ~6.9 DKK/MWh (from config)
        self.batt_power_c_rate = self.config.batt_power_c_rate    # From config
        self.batt_soc_min = self.config.batt_soc_min
        self.batt_soc_max = self.config.batt_soc_max

        # CAPEX tables from config (get DKK values for internal calculations)
        self.asset_capex = self.config.get_asset_capex(currency='DKK')
        # Note: If overrides are needed later, they should be passed as explicit parameters

        # Initialize fund with proper allocation structure
        self.total_fund_value = float(self.init_budget)  # Total fund: 5.52B DKK

        # Calculate physical and trading allocations
        self.physical_allocation_budget = self.total_fund_value * self.config.physical_allocation  # 88% for assets
        self.trading_allocation_budget = self.total_fund_value * self.config.financial_allocation   # 12% for trading

        # Budget represents available trading capital (starts as trading allocation)
        self.budget = float(self.trading_allocation_budget)  # ~662M DKK for trading
        self.equity = float(self.init_budget)

        # CRITICAL FIX: Ensure self.t is 0 before asset deployment and NAV calculation
        # This prevents depreciation from being applied during initialization
        # Save current t if it exists, then set to 0
        original_t_init = getattr(self, 't', None)
        self.t = 0

        # FIXED: Deploy initial assets (ONE-TIME ONLY)
        if not self.assets_deployed:
            self._deploy_initial_assets_once(initial_asset_plan)
        self.operational_state['battery_energy'] = self._get_initial_battery_energy()

        # CRITICAL FIX: Initialize reward calculator AFTER asset deployment with correct baseline NAV
        if getattr(self, 'reward_calculator', None) is None:
            # CRITICAL: Ensure t=0 for initial NAV calculation (no depreciation)
            self.t = 0
            # Use post-CAPEX NAV as baseline instead of pre-CAPEX init_budget
            # CRITICAL FIX: Pass current_timestep=0 explicitly to ensure no depreciation
            post_capex_nav = self._calculate_fund_nav(current_timestep=0)
            # Restore original t if it was set (should be None at init, but be safe)
            if original_t_init is not None:
                self.t = original_t_init
            self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
            # Store config reference for reward calculator
            if hasattr(self.reward_calculator, 'config'):
                self.reward_calculator.config = self.config
            logger.info(f"Reward calculator initialized with post-CAPEX baseline NAV: {post_capex_nav:,.0f} DKK")
        else:
            logger.info(f"Reward calculator already exists: {type(self.reward_calculator)}")
        # Ensure environment always exposes a usable reward_weights snapshot
        self.reward_weights = dict(getattr(self.reward_calculator, 'reward_weights', {})) if self.reward_calculator else {}
        
        # LOGGING: Initialize reward logger
        tier_name = "tier1"
        # NEW: Use provided log_dir or default to "debug_logs"
        debug_log_dir = log_dir if log_dir is not None else "debug_logs"
        self.debug_tracker = RewardLogger(log_dir=debug_log_dir, tier_name=tier_name)
        self.current_episode = 0
        self._episode_counter = -1  # Will be incremented to 0 on first reset

        # FIXED: Currency conversion and data loading
        # Convert DKK prices to USD (Danish data) - SINGLE CONVERSION POINT
        DKK_TO_USD = self.config.dkk_to_usd_rate  # From config: 0.145 (1 USD = ~6.9 DKK)

        # vectorized series with currency conversion
        self._wind  = self.data.get('wind',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._solar = self.data.get('solar', pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._hydro = self.data.get('hydro', pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()

        # CLEAN APPROACH: Keep everything in DKK throughout the system
        price_dkk = self.data.get('price', pd.Series(250.0, index=self.data.index)).astype(float)

        # Step 1: Filter extreme outliers (keep realistic DKK range) - from config
        price_dkk_filtered = np.clip(price_dkk, self.config.minimum_price_filter, self.config.maximum_price_cap)

        # Step 2: Keep DKK throughout system - NO EARLY CONVERSION
        # Store conversion rate for final reporting only
        self._dkk_to_usd_rate = DKK_TO_USD

        # Initialize conversion logging flag first
        if not hasattr(self, '_conversion_logged'):
            self._conversion_logged = False

        # Log DKK price range for verification
        if not self._conversion_logged:
            logger.info(f"Price system: Using DKK throughout, USD conversion rate = {DKK_TO_USD:.3f}")
            logger.info(f"DKK price range: {price_dkk_filtered.min():.1f}-{price_dkk_filtered.max():.1f} DKK/MWh")
            self._conversion_logged = True

        # Step 3: PRICE NORMALIZATION - All in DKK
        self._price_raw = price_dkk_filtered.to_numpy()  # Raw DKK prices for revenue calculation
        # Initialize 1-step price return array for Tier 1 PnL reward
        self._price_return_1step = np.zeros(len(self._price_raw), dtype=np.float32)
        
        # FAIR SPARSE REWARD FIX: Multi-step rolling return buffer
        # This makes rewards less sparse by using rolling average of price returns

        # Price normalization for agent observations
        # - Default: rolling mean/std inside the episode (non-stationary across episode boundaries)
        # - Episode training (optional): global mean/std computed once from full training_dataset
        use_global_norm = bool(getattr(self.config, "use_global_normalization", False))
        g_mean = getattr(self.config, "global_price_mean", None)
        g_std = getattr(self.config, "global_price_std", None)

        window_size = min(4320, len(price_dkk_filtered))
        if use_global_norm and (g_mean is not None) and (g_std is not None):
            mean_val = float(g_mean)
            std_val = float(max(float(g_std), 1e-6))
            price_normalized = (price_dkk_filtered - mean_val) / std_val
            price_normalized_clipped = np.clip(price_normalized, -3.0, 3.0)
            self._price = (price_normalized_clipped / 3.0).to_numpy()
            # store constant arrays for any code that expects per-timestep mean/std
            self._price_mean = np.full(len(self._price_raw), mean_val, dtype=float)
            self._price_std = np.full(len(self._price_raw), std_val, dtype=float)
        else:
            prior_mean = float(self._price_raw[0])
            prior_std = float(max(getattr(self.config, "rolling_past_price_std_floor", 75.0), 1e-6))
            prior_count = 1.0

            std_floor = float(max(getattr(self.config, "rolling_past_price_std_floor", 75.0), 1e-6))
            price_normalized, price_mean_arr, price_std_arr, price_state = _online_normalize_with_state(
                self._price_raw,
                getattr(self.config, "rolling_past_price_state", None),
                prior_mean=prior_mean,
                prior_std=prior_std,
                prior_count=prior_count,
                std_floor=std_floor,
            )
            price_normalized_clipped = np.clip(price_normalized, -3.0, 3.0)  # Â±3 sigma bounds

            # Ensure downstream components (wrapper/CV) see identical [-1, 1] scaling
            self._price = (price_normalized_clipped / 3.0)

            # Store normalization parameters for revenue calculations
            self._price_mean = price_mean_arr
            self._price_std = price_std_arr
            self.config.rolling_past_price_state = price_state

        # FIXED: Prices remain in DKK throughout system for consistency
        # Raw prices in _price_raw are DKK, normalized prices in _price are z-scores

        self._load  = self.data.get('load',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._riskS = self.data.get('risk',  pd.Series(0.3, index=self.data.index)).astype(float).to_numpy()
        
        # CRITICAL FIX: Compute rolling mean/std for generation assets (wind, solar, hydro).
        # Needed for consistent z-score normalization used by calibration and CV components.
        # Using same 30-day window (4320 timesteps) as price for consistency.
        wind_series = pd.Series(self._wind, index=self.data.index[:len(self._wind)])
        solar_series = pd.Series(self._solar, index=self.data.index[:len(self._solar)])
        hydro_series = pd.Series(self._hydro, index=self.data.index[:len(self._hydro)])
        
        wind_rolling_mean = wind_series.rolling(window=window_size, min_periods=1).mean()
        wind_rolling_std = wind_series.rolling(window=window_size, min_periods=1).std()
        wind_rolling_mean = wind_rolling_mean.fillna(wind_series.mean())
        wind_rolling_std = wind_rolling_std.fillna(wind_series.std())
        wind_rolling_std = wind_rolling_std.replace(0.0, wind_series.std())
        
        solar_rolling_mean = solar_series.rolling(window=window_size, min_periods=1).mean()
        solar_rolling_std = solar_series.rolling(window=window_size, min_periods=1).std()
        solar_rolling_mean = solar_rolling_mean.fillna(solar_series.mean())
        solar_rolling_std = solar_rolling_std.fillna(solar_series.std())
        solar_rolling_std = solar_rolling_std.replace(0.0, solar_series.std())
        
        hydro_rolling_mean = hydro_series.rolling(window=window_size, min_periods=1).mean()
        hydro_rolling_std = hydro_series.rolling(window=window_size, min_periods=1).std()
        hydro_rolling_mean = hydro_rolling_mean.fillna(hydro_series.mean())
        hydro_rolling_std = hydro_rolling_std.fillna(hydro_series.std())
        hydro_rolling_std = hydro_rolling_std.replace(0.0, hydro_series.std())
        
        # Store rolling statistics for z-score computation
        self._wind_mean = wind_rolling_mean.to_numpy()
        self._wind_std = wind_rolling_std.to_numpy()
        self._solar_mean = solar_rolling_mean.to_numpy()
        self._solar_std = solar_rolling_std.to_numpy()
        self._hydro_mean = hydro_rolling_mean.to_numpy()
        self._hydro_std = hydro_rolling_std.to_numpy()

        if use_global_norm:
            # Prefer global p95 scales when available, fall back to per-episode robust p95.
            self.wind_scale  = float(getattr(self.config, "global_wind_scale", None) or _p95_robust(self._wind,  min_scale=0.1))
            self.solar_scale = float(getattr(self.config, "global_solar_scale", None) or _p95_robust(self._solar, min_scale=0.1))
            self.hydro_scale = float(getattr(self.config, "global_hydro_scale", None) or _p95_robust(self._hydro, min_scale=0.1))
            self.load_scale  = float(getattr(self.config, "global_load_scale", None) or _p95_robust(self._load,  min_scale=0.1))
        else:
            scale_alpha = float(getattr(self.config, "rolling_past_scale_ema_alpha", 0.10))
            self.wind_scale = _resolve_persistent_scale(
                self.config,
                "rolling_past_wind_scale",
                self._wind,
                global_value=None,
                alpha=scale_alpha,
                min_scale=0.1,
            )
            self.solar_scale = _resolve_persistent_scale(
                self.config,
                "rolling_past_solar_scale",
                self._solar,
                global_value=None,
                alpha=scale_alpha,
                min_scale=0.1,
            )
            self.hydro_scale = _resolve_persistent_scale(
                self.config,
                "rolling_past_hydro_scale",
                self._hydro,
                global_value=None,
                alpha=scale_alpha,
                min_scale=0.1,
            )
            self.load_scale = _resolve_persistent_scale(
                self.config,
                "rolling_past_load_scale",
                self._load,
                global_value=None,
                alpha=scale_alpha,
                min_scale=0.1,
            )

        # agents
        self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]
        self.agents = self.possible_agents[:]

        # observation manager & spaces (BASE only here)
        self.obs_manager = StabilizedObservationManager(self)
        self.observation_spaces = {a: self.obs_manager.obs_space(a) for a in self.possible_agents}

        # Action spaces:
        # - investor: bounded continuous exposure control
        # - battery: discrete power ladder for charge / idle / discharge
        investor_shape = (1,)
        self.action_spaces = {
            # Investor action: exposure-only [exposure_raw] in [-1, 1]
            "investor_0":         spaces.Box(low=-1.0, high=1.0, shape=investor_shape, dtype=np.float32),
            "battery_operator_0": spaces.Discrete(len(getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0]))),
            "risk_controller_0":  spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            # Meta controller adjusts capital allocation only. Trading cadence is fixed in config.
            "meta_controller_0":  spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        }

        # CRITICAL FIX: Reward calculator is now initialized AFTER asset deployment above
        # No need to set to None here as it's already properly initialized

        if enhanced_risk_controller:
            try:
                self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144, config=self.config)
            except TypeError as e:
                logger.warning(
                    "EnhancedRiskController rejected lookback_window; retrying alternate constructor signature: %s",
                    e,
                )
                self.enhanced_risk_controller = EnhancedRiskController(config=self.config)
            except Exception:
                logger.exception("EnhancedRiskController initialization failed")
                raise
        else:
            self.enhanced_risk_controller = None

        # UNIFIED MEMORY MANAGER (NEW)
        self.memory_manager = UnifiedMemoryManager(max_memory_mb=self.max_memory_mb)

        # buffers
        # Investor uses a plain Box(12,) observation.
        def _alloc_obs_slot(agent_name: str):
            sp = self.observation_spaces[agent_name]
            if isinstance(sp, spaces.Dict):
                return {k: np.zeros(sub.shape, np.float32) for k, sub in sp.spaces.items()}
            return np.zeros(sp.shape, np.float32)

        self._obs_buf: Dict[str, Any] = {a: _alloc_obs_slot(a) for a in self.possible_agents}
        self._rew_buf: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self._done_buf: Dict[str, bool] = {a: False for a in self.possible_agents}
        self._trunc_buf: Dict[str, bool] = {a: False for a in self.possible_agents}
        self._info_buf: Dict[str, Dict[str, Any]] = {a: {} for a in self.possible_agents}

        # histories
        self.performance_history = {
            'revenue_history': deque(maxlen=512),
            'payout_efficiency': deque(maxlen=256),
            'battery_revenue_history': deque(maxlen=512),
            'generation_revenue_history': deque(maxlen=512),
            'nav_history': deque(maxlen=512),
        }

        # Fund performance tracking
        self.cumulative_battery_revenue = 0.0
        self.cumulative_generation_revenue = 0.0
        self.accumulated_operational_revenue = 0.0  # CRITICAL: Initialize operational revenue tracking

        # runtime vars
        self.t = 0
        self.step_in_episode = 0
        self._last_saturation_count = 0
        self._clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        # regime snapshots (wrapper reads these)
        self.market_volatility = 0.0
        self.market_stress = ENV_MARKET_STRESS_DEFAULT
        self.overall_risk_snapshot = ENV_OVERALL_RISK_DEFAULT
        self.market_risk_snapshot = ENV_MARKET_RISK_DEFAULT

        # meta knobs - FROM CONFIG
        self.capital_allocation_fraction = self.config.capital_allocation_fraction
        self._meta_rule_last_update_step = -1
        self._meta_rule_signal_ema = 0.0
        self._meta_rule_target_capital_fraction = float(self.capital_allocation_fraction)

        # Update meta controller ranges from config
        self.META_FREQ_MIN = self.config.meta_freq_min
        self.META_FREQ_MAX = self.config.meta_freq_max
        self.META_CAP_MIN = self.config.meta_cap_min
        self.META_CAP_MAX = self.config.meta_cap_max
        self.SAT_EPS = self.config.sat_eps

        # tracked finance state - FROM CONFIG
        self.investment_capital = float(self.init_budget)
        self.distributed_profits = 0.0
        self.total_distributions = 0.0
        self.cumulative_returns = 0.0
        self._previous_nav = 0.0  # Track previous NAV for cumulative return accounting
        self.max_leverage = self.config.max_leverage
        self.risk_multiplier = self.config.risk_multiplier

        # performance tracking
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.last_realized_dnav = 0.0
        self.last_realized_dnav_return = 0.0
        # Investor-sleeve realized dNAV (used for forecast-backend supervision/logging).
        self.last_realized_investor_dnav = 0.0
        self.last_realized_investor_dnav_return = 0.0
        self.last_realized_investor_return_denom = 1.0
        self._investor_local_return_history = deque(
            maxlen=max(1, int(getattr(self.config, 'investor_trading_history_lookback', 48)))
        )
        self._investor_local_path_value = 1.0
        self._investor_local_peak_value = 1.0
        self._investor_local_drawdown = 0.0
        self._investor_local_recent_mean = 0.0
        self._investor_local_recent_vol = 0.0
        self._investor_local_quality = 0.0
        self._investor_local_quality_delta = 0.0
        self._investor_mean_history = deque(
            maxlen=max(1, int(getattr(self.config, 'investor_mean_collapse_window', 256)))
        )
        self._investor_mean_clip_hit_history = deque(
            maxlen=max(1, int(getattr(self.config, 'investor_mean_clip_hit_window', 256)))
        )
        self._investor_mean_abs_rolling = 0.0
        self._investor_mean_sign_consistency = 0.0
        self._investor_mean_clip_hit_rate = 0.0
        self._fund_nav_prev = None
        self._last_investor_transaction_cost = 0.0
        self._last_investor_exposure_pretrade = 0.0
        # Diagnostic cumulative MTM tracker; not equal to live sleeve value.
        self.cumulative_mtm_pnl = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

        # Cache-only forecast utilization state. No live model object, no forecast
        # observation expansion, and no forecast-shaped reward channel.
        self._forecast_prior_features = None
        self._forecast_prior_processor = None
        self._last_forecast_prior = {}

        logger.info("=" * 70)
        logger.info("TIER-1 HYBRID RL BASELINE")
        logger.info("=" * 70)

        # NOTE (Model B accounting):
        # Financial instruments are treated as a margin-style exposure sleeve:
        # - `financial_positions` are exposures (not equity / not "position value")
        # - NAV impact comes from MTM PnL, tracked separately in `financial_mtm_positions`
        # Therefore we do NOT maintain "open position cost basis" bookkeeping.

        self._last_nav = None  # Track NAV changes for per-step NAV logging


        usd_value = self.init_budget * getattr(self.config, 'dkk_to_usd_rate', 0.145)
        logger.info(f"Hybrid renewable fund initialized with ${usd_value/1e6:,.0f}M USD")
        self._log_fund_structure()

        # GUARDRAIL: Startup assert to prevent regression
        assert hasattr(self, "_price_raw") and hasattr(self, "_price"), "Price arrays not initialized"

        self._action_convert_warned = set()

    @property
    def battery_energy(self) -> float:
        return self.operational_state['battery_energy']

    @battery_energy.setter
    def battery_energy(self, value: float) -> None:
        self.operational_state['battery_energy'] = float(value)

    @property
    def battery_discharge_power(self) -> float:
        return self.operational_state['battery_discharge_power']

    @battery_discharge_power.setter
    def battery_discharge_power(self, value: float) -> None:
        self.operational_state['battery_discharge_power'] = float(value)

    # =====================================================================
    # FIXED: ONE-TIME ASSET DEPLOYMENT
    # =====================================================================

    def _deploy_initial_assets_once(self, plan: Optional[dict]):
        """FIXED: ONE-TIME ONLY asset deployment with proper accounting"""
        if self.assets_deployed:
            logger.info("Assets already deployed, skipping")
            return

        if plan is None:
            # Get default asset plan from config
            plan = self.config.get_initial_asset_plan()

        try:
            total_capex = 0.0

            # Calculate total CAPEX required (using consistent DKK values)
            for asset_type, specs in plan.items():
                if asset_type == 'wind':
                    # Use DKK CAPEX for all wind calculations (standardized)
                    capex = specs['capacity_mw'] * self.asset_capex['wind_mw']
                elif asset_type == 'solar':
                    capex = specs['capacity_mw'] * self.asset_capex['solar_mw']
                elif asset_type == 'hydro':
                    capex = specs['capacity_mw'] * self.asset_capex['hydro_mw']
                elif asset_type == 'battery':
                    capex = specs['capacity_mwh'] * self.asset_capex['battery_mwh']
                else:
                    continue
                total_capex += capex

            # FIXED: Check physical allocation budget sufficiency (not trading budget)
            if total_capex > self.physical_allocation_budget:
                logger.warning(f"Insufficient physical allocation: {total_capex:,.0f} DKK required, {self.physical_allocation_budget:,.0f} DKK available")
                # Scale down proportionally
                scale_factor = (self.physical_allocation_budget * 0.95) / total_capex  # Use 95% of physical allocation
                logger.info(f"Scaling asset plan by {scale_factor:.2f}")
            else:
                scale_factor = 1.0
                logger.info(f"Physical allocation sufficient: {total_capex:,.0f} DKK required, {self.physical_allocation_budget:,.0f} DKK available")

            # Deploy physical assets (using physical allocation, not trading budget)
            total_physical_capex_used = 0.0

            # CRITICAL FIX: Reset physical_assets to zero first to ensure clean state
            self.physical_assets = {
                'wind_capacity_mw': 0.0,
                'solar_capacity_mw': 0.0,
                'hydro_capacity_mw': 0.0,
                'battery_capacity_mwh': 0.0,
            }

            for asset_type, specs in plan.items():
                if asset_type == 'wind':
                    capacity = float(specs['capacity_mw'] * scale_factor)  # CRITICAL: Explicit float conversion
                    self.physical_assets['wind_capacity_mw'] = capacity
                    capex_used = capacity * self.asset_capex['wind_mw']

                elif asset_type == 'solar':
                    capacity = float(specs['capacity_mw'] * scale_factor)  # CRITICAL: Explicit float conversion
                    self.physical_assets['solar_capacity_mw'] = capacity
                    capex_used = capacity * self.asset_capex['solar_mw']

                elif asset_type == 'hydro':
                    capacity = float(specs['capacity_mw'] * scale_factor)  # CRITICAL: Explicit float conversion
                    self.physical_assets['hydro_capacity_mw'] = capacity
                    capex_used = capacity * self.asset_capex['hydro_mw']

                elif asset_type == 'battery':
                    capacity = float(specs['capacity_mwh'] * scale_factor)  # CRITICAL: Explicit float conversion
                    self.physical_assets['battery_capacity_mwh'] = capacity
                    capex_used = capacity * self.asset_capex['battery_mwh']
                else:
                    continue

                # Track physical CAPEX spending (don't deduct from trading budget)
                total_physical_capex_used += capex_used

            # Store physical CAPEX for NAV calculation
            self.physical_capex_deployed = total_physical_capex_used

            # Mark as deployed (PERMANENT)
            self.assets_deployed = True

            # Trading budget remains unchanged (it's separate from physical allocation)
            # self.budget stays as trading_allocation_budget (~662M DKK)

            # CRITICAL FIX: Pass current_timestep=0 explicitly to ensure no depreciation
            # No need to modify self.t - just pass the correct timestep as parameter
            nav_after_deploy = self._calculate_fund_nav(current_timestep=0)

            # CRITICAL DEBUG: Log deployment details for consistency verification
            logger.info("[ASSET_DEPLOY] Asset deployment complete:")
            logger.info(f"  Physical allocation budget: {self.physical_allocation_budget:,.0f} DKK")
            logger.info(f"  Total CAPEX required: {total_capex:,.0f} DKK")
            logger.info(f"  Scale factor: {scale_factor:.6f}")
            logger.info(f"  Physical CAPEX deployed: {self.physical_capex_deployed:,.0f} DKK")
            logger.info(f"  Wind: {self.physical_assets['wind_capacity_mw']:.6f} MW (CAPEX: {self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw']:,.0f} DKK)")
            logger.info(f"  Solar: {self.physical_assets['solar_capacity_mw']:.6f} MW (CAPEX: {self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw']:,.0f} DKK)")
            logger.info(f"  Hydro: {self.physical_assets['hydro_capacity_mw']:.6f} MW (CAPEX: {self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw']:,.0f} DKK)")
            logger.info(f"  Battery: {self.physical_assets['battery_capacity_mwh']:.6f} MWh (CAPEX: {self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']:,.0f} DKK)")
            
            # CRITICAL: Calculate and log actual physical book value from NAV calculation
            # This will show if depreciation is being applied
            self.t = 0  # Ensure t=0 for NAV calculation
            calculated_nav = self._calculate_fund_nav()
            calculated_fin_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
            calculated_physical = calculated_nav - self.budget - getattr(self, 'accumulated_operational_revenue', 0.0) - calculated_fin_mtm
            logger.info(f"  Calculated NAV: {calculated_nav:,.0f} DKK")
            logger.info(f"  Calculated Physical Assets (from NAV): {calculated_physical:,.0f} DKK")
            logger.info(f"  Expected Physical Assets (no dep): {self.physical_capex_deployed:,.0f} DKK")
            logger.info(f"  Difference: {calculated_physical - self.physical_capex_deployed:,.0f} DKK ({(calculated_physical / self.physical_capex_deployed - 1.0) * 100:.4f}%)")
            
            logger.info(f"  Total CAPEX: {total_capex * scale_factor:,.0f} DKK (${total_capex * scale_factor * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
            logger.info(f"  Remaining cash: {self.budget:,.0f} DKK (${self.budget * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")

            # SANITY ASSERTS: Verify deployment using config values
            expected_values = self.config.get_expected_physical_values()
            expected_wind = expected_values['wind']
            expected_solar = expected_values['solar']
            expected_hydro = expected_values['hydro']
            expected_battery = expected_values['battery']
            expected_physical_book = expected_values['physical_book_value']
            expected_cash_min = expected_values['cash_min']

            actual_wind = self.physical_assets['wind_capacity_mw']
            actual_solar = self.physical_assets['solar_capacity_mw']
            actual_hydro = self.physical_assets['hydro_capacity_mw']
            actual_battery = self.physical_assets['battery_capacity_mwh']

            # Allow 5% tolerance for scaling
            tolerance = 0.05
            assert abs(actual_wind - expected_wind) / expected_wind < tolerance, \
                f"Wind capacity mismatch: {actual_wind:.1f} MW vs expected {expected_wind:.1f} MW"
            assert abs(actual_solar - expected_solar) / expected_solar < tolerance, \
                f"Solar capacity mismatch: {actual_solar:.1f} MW vs expected {expected_solar:.1f} MW"
            assert abs(actual_hydro - expected_hydro) / expected_hydro < tolerance, \
                f"Hydro capacity mismatch: {actual_hydro:.1f} MW vs expected {expected_hydro:.1f} MW"
            assert abs(actual_battery - expected_battery) / expected_battery < tolerance, \
                f"Battery capacity mismatch: {actual_battery:.1f} MWh vs expected {expected_battery:.1f} MWh"
            assert self.budget >= expected_cash_min, \
                f"Insufficient cash for trading: ${self.budget:,.0f} vs minimum ${expected_cash_min:,.0f}"

            logger.info("[OK] Option 1 deployment verified successfully")

        except Exception as e:
            logger.error(f"Asset deployment failed: {e}")
            self.assets_deployed = False

    def _log_fund_structure(self):
        """Log the fund's hybrid structure with clear separation"""
        usd_rate = float(getattr(self.config, 'dkk_to_usd_rate', 0.145) or 0.145)

        def _usd_m(amount_dkk: float) -> str:
            return f"${(float(amount_dkk) * usd_rate) / 1e6:,.0f}M USD"

        logger.info("=" * 60)
        logger.info("HYBRID RENEWABLE ENERGY FUND STRUCTURE")
        logger.info("=" * 60)
        logger.info("ECONOMIC MODEL: Hybrid approach with clear separation")
        logger.info("1) PHYSICAL OWNERSHIP: Direct ownership of renewable assets")
        logger.info("2) FINANCIAL TRADING: Derivatives on renewable energy indices")
        logger.info("")

        # Physical assets (owned infrastructure)
        total_physical_mw = (self.physical_assets['wind_capacity_mw'] +
                           self.physical_assets['solar_capacity_mw'] +
                           self.physical_assets['hydro_capacity_mw'])
        physical_value = (self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] +
                         self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] +
                         self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] +
                         self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh'])

        logger.info("1. PHYSICAL ASSETS (Owned Infrastructure - Generate Real Electricity):")
        logger.info(f"   Wind farms: {self.physical_assets['wind_capacity_mw']:.1f} MW ({_usd_m(self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'])})")
        logger.info(f"   Solar farms: {self.physical_assets['solar_capacity_mw']:.1f} MW ({_usd_m(self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'])})")
        logger.info(f"   Hydro plants: {self.physical_assets['hydro_capacity_mw']:.1f} MW ({_usd_m(self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'])})")
        logger.info(f"   Battery storage: {self.physical_assets['battery_capacity_mwh']:.1f} MWh ({_usd_m(self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh'])})")
        logger.info(f"   Total Physical: {total_physical_mw:.1f} MW ({_usd_m(physical_value)} book value)")
        logger.info("")

        # Financial instruments (derivatives trading)
        total_financial = sum(abs(v) for v in self.financial_positions.values())
        logger.info("2. FINANCIAL INSTRUMENTS (Derivatives Trading - Mark-to-Market):")
        logger.info(f"   Wind index exposure: {_usd_m(self.financial_positions['wind_instrument_value'])}")
        logger.info(f"   Solar index exposure: {_usd_m(self.financial_positions['solar_instrument_value'])}")
        logger.info(f"   Hydro index exposure: {_usd_m(self.financial_positions['hydro_instrument_value'])}")
        logger.info(f"   Total Financial Exposure: {_usd_m(total_financial)}")
        logger.info("")

        # Fund summary
        fund_nav = self._calculate_fund_nav()
        logger.info("3. FUND SUMMARY:")
        logger.info(f"   Cash position: {_usd_m(self.budget)}")
        logger.info(f"   Physical assets (book): {_usd_m(physical_value)}")
        logger.info(f"   Financial positions (MTM): {_usd_m(total_financial)}")
        logger.info(f"   Total Fund NAV: {_usd_m(fund_nav)}")
        logger.info(f"   Initial capital: {_usd_m(self.init_budget)}")
        logger.info(f"   Total return: {((fund_nav - self.init_budget) / self.init_budget * 100):+.2f}%")
        logger.info("=" * 60)

    # =====================================================================
    # FIXED: PROPER NAV CALCULATION
    # =====================================================================

    def _calculate_fund_nav(self, current_timestep: Optional[int] = None) -> float:
        """
        REFACTORED: Calculate true fund NAV using FinancialEngine.
        NAV = Trading Cash + Physical Asset Book Value + Accumulated Operational Revenue + Financial Instrument MTM
        
        Args:
            current_timestep: Optional timestep to use for depreciation. If None, uses self.t
        """
        try:
            # CRITICAL FIX: Use provided timestep parameter, or fall back to self.t
            # This ensures consistent NAV calculation when called from _update_finance with i parameter
            if current_timestep is None:
                current_timestep = getattr(self, 't', 0)
            accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
            
            # CRITICAL DEBUG: Log when NAV is calculated at timestep 0
            if current_timestep == 0:
                usd_rate = float(getattr(self.config, 'dkk_to_usd_rate', 0.145) or 0.145)
                logger.info("[NAV_CALC] Calculating NAV at t=0:")
                logger.info(f"  current_timestep param = {current_timestep}")
                logger.info(f"  self.t = {getattr(self, 't', 0)}")
                logger.info(f"  budget = ${(self.budget * usd_rate) / 1_000_000:.2f}M")
                logger.info(f"  physical_assets = {self.physical_assets}")
                logger.info(f"  physical_capex_deployed = ${(getattr(self, 'physical_capex_deployed', 0.0) * usd_rate) / 1_000_000:.2f}M")
            
            # Use FinancialEngine for calculation
            from financial_engine import FinancialEngine
            nav = FinancialEngine.calculate_fund_nav(
                budget=self.budget,
                physical_assets=self.physical_assets,
                asset_capex=self.asset_capex,
                # CRITICAL: NAV must include MTM VALUE (PnL), not NOTIONAL exposure.
                financial_mtm_values=getattr(self, 'financial_mtm_positions', self.financial_positions),
                accumulated_operational_revenue=accumulated_operational_revenue,
                current_timestep=current_timestep,
                config=self.config
            )
            
            # CRITICAL DEBUG: Log calculated NAV at timestep 0
            if current_timestep == 0:
                usd_rate = float(getattr(self.config, 'dkk_to_usd_rate', 0.145) or 0.145)
                logger.info(f"[NAV_CALC] NAV calculated: ${(nav * usd_rate) / 1_000_000:.2f}M")
            
            self.equity = nav
            return nav

        except Exception as e:
            msg = (
                f"[NAV_CALC_FATAL] NAV calculation failed at t={getattr(self, 't', 'unknown')}: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    def get_fund_nav_breakdown(
        self,
        fund_nav: Optional[float] = None,
        current_timestep: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Return the additive fund NAV components plus cumulative gain diagnostics.

        Identity:
            fund_nav_dkk = trading_cash_dkk + physical_book_value_dkk
                         + accumulated_operational_revenue_dkk + financial_mtm_dkk

        The cumulative gain diagnostics are useful for training progress, but do
        not themselves sum to fund NAV.
        """
        try:
            if current_timestep is None:
                current_timestep = int(getattr(self, 't', 0))
            if fund_nav is None or (not np.isfinite(float(fund_nav))):
                fund_nav = float(self._calculate_fund_nav(current_timestep=current_timestep))

            years_elapsed = float(current_timestep / (365.25 * 24 * 6))
            annual_dep = float(getattr(self.config, 'annual_depreciation_rate', 0.02))
            max_dep = float(getattr(self.config, 'max_depreciation_ratio', 0.75))
            depreciation_ratio = float(min(years_elapsed * annual_dep, max_dep))

            pa = getattr(self, 'physical_assets', {}) or {}
            capex = getattr(self, 'asset_capex', {}) or {}
            physical_book_value_dkk = float(
                float(pa.get('wind_capacity_mw', 0.0)) * float(capex.get('wind_mw', 0.0)) * (1.0 - depreciation_ratio) +
                float(pa.get('solar_capacity_mw', 0.0)) * float(capex.get('solar_mw', 0.0)) * (1.0 - depreciation_ratio) +
                float(pa.get('hydro_capacity_mw', 0.0)) * float(capex.get('hydro_mw', 0.0)) * (1.0 - depreciation_ratio) +
                float(pa.get('battery_capacity_mwh', 0.0)) * float(capex.get('battery_mwh', 0.0)) * (1.0 - depreciation_ratio)
            )

            accumulated_operational_revenue_dkk = float(getattr(self, 'accumulated_operational_revenue', 0.0))
            financial_mtm_dkk = float(
                sum((getattr(self, 'financial_mtm_positions', getattr(self, 'financial_positions', {})) or {}).values())
            )
            financial_exposure_dkk = float(
                sum(abs(float(v)) for v in (getattr(self, 'financial_positions', {}) or {}).values())
            )
            trading_cash_dkk = float(max(0.0, float(getattr(self, 'budget', 0.0))))
            trading_sleeve_value_dkk = float(trading_cash_dkk + financial_mtm_dkk)
            trading_mtm_tracker_dkk = float(getattr(self, 'cumulative_mtm_pnl', 0.0))
            battery_cash_contribution_dkk = float(getattr(self, 'cumulative_battery_revenue', 0.0))
            trading_cash_core_dkk = float(trading_cash_dkk - battery_cash_contribution_dkk)
            total_distributions_dkk = float(max(0.0, getattr(self, 'total_distributions', 0.0)))
            distribution_adjusted_nav_dkk = float(fund_nav + total_distributions_dkk)
            distribution_adjusted_trading_sleeve_dkk = float(trading_sleeve_value_dkk + total_distributions_dkk)
            usd_rate = float(getattr(self, 'usd_conversion_rate', getattr(self.config, 'usd_conversion_rate', 0.145)))

            return {
                'fund_nav_dkk': float(fund_nav),
                'fund_nav_usd': float(fund_nav * usd_rate),
                'distribution_adjusted_nav_dkk': distribution_adjusted_nav_dkk,
                'distribution_adjusted_nav_usd': float(distribution_adjusted_nav_dkk * usd_rate),
                'total_distributions_dkk': total_distributions_dkk,
                'trading_cash_dkk': trading_cash_dkk,
                'trading_cash_core_dkk': trading_cash_core_dkk,
                'battery_cash_contribution_dkk': battery_cash_contribution_dkk,
                'trading_sleeve_value_dkk': trading_sleeve_value_dkk,
                'distribution_adjusted_trading_sleeve_dkk': distribution_adjusted_trading_sleeve_dkk,
                'physical_book_value_dkk': physical_book_value_dkk,
                'accumulated_operational_revenue_dkk': accumulated_operational_revenue_dkk,
                'financial_mtm_dkk': financial_mtm_dkk,
                'financial_exposure_dkk': financial_exposure_dkk,
                'depreciation_ratio': depreciation_ratio,
                'years_elapsed': years_elapsed,
                'trading_mtm_tracker_dkk': trading_mtm_tracker_dkk,
                'battery_revenue_tracker_dkk': float(getattr(self, 'cumulative_battery_revenue', 0.0)),
                'operating_revenue_dkk': accumulated_operational_revenue_dkk,
            }
        except Exception as e:
            raise RuntimeError(f"[NAV_BREAKDOWN_FATAL] Failed to build NAV breakdown: {e}") from e

    # ------------------------------------------------------------------
    # SB3/wrapper expect callable per-agent space getters
    # ------------------------------------------------------------------
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    # Wrapper helper for base dims
    def _get_base_observation_dim(self, agent: str) -> int:
        return self.obs_manager.base_dim(agent)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Seed reproducibly
        if seed is not None:
            try:
                self._rng = np.random.default_rng(seed)
                self._last_seed = int(seed)
            except Exception:
                self._rng = np.random.default_rng()
                self._last_seed = None

        logger.info(
            f"[RESET] Environment reset: base_feature_dim={BASE_FEATURE_DIM}, "
            f"forecast_utilization={bool(getattr(self.config, 'enable_forecast_utilization', False))}"
        )

        # Check if this is a true episode reset (end of data) or just PPO buffer reset
        # FIXED: Safety check for self.t attribute
        current_timestep = getattr(self, 't', 0)
        is_true_episode_end = (current_timestep >= self.max_steps)

        # CRITICAL FIX: For Episode 0 first reset, ensure t=0 before any NAV calculations
        # This prevents depreciation from being applied during initial reset
        if not hasattr(self, '_episode_counter') or self._episode_counter == -1:
            # First reset (Episode 0) - ensure t=0
            self.t = 0
            # Ensure config-global step counter exists (persists across episodes, since config is reused).
            if self.config is not None and not hasattr(self.config, "training_global_step"):
                try:
                    self.config.training_global_step = 0
                except Exception:
                    pass

        # Only reset time if true episode end
        if is_true_episode_end:
            self.t = 0
            self._episode_counter += 1
            if hasattr(self, 'debug_tracker'):
                self.debug_tracker.start_episode(self._episode_counter)
            logger.info(f"[RESET] TRUE EPISODE RESET: End of data reached, resetting to start (Episode {self._episode_counter})")
        else:
            # First reset (episode 0) - initialize debug tracker
            if hasattr(self, 'debug_tracker') and self._episode_counter == -1:
                self._episode_counter = 0
            if hasattr(self, 'debug_tracker'):
                self.debug_tracker.start_episode(self._episode_counter)
            logger.info(f"[PPO] PPO BUFFER RESET: Preserving financial state at step {current_timestep}")

        self.step_in_episode = 0
        self.agents = self.possible_agents[:]

        # CRITICAL FIX: EVERY episode should fully reset (except agent learning)
        # This ensures each episode starts from identical initial state
        # Only agent's learned policy (neural network weights) persists
        if is_true_episode_end:
            # Full reset at end of data
            self._reset_financial_state_full()
        else:
            # RESTORED: For PPO buffer resets, preserve financial gains
            # This allows gains to accumulate across episodes, showing learning progress
            # Only reset operational state, not financial positions/budget
            logger.info(f"[RESET] PPO buffer reset: Preserving financial state (gains accumulate)")
            
            # Use partial reset that preserves financial gains
            self._reset_financial_state()

        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()

        # OPTION 2: Clear z-score history buffer on reset
        # Forecast-cache prior: load per-episode ANN forecast cache for the
        # investor execution prior. This does not change observation shape.
        if bool(getattr(self.config, "enable_forecast_utilization", False)):
            try:
                from forecast_prior import load_forecast_prior_features, ConformalForecastPrior
                ep_idx = int(getattr(self, "_episode_counter", 0) or 0)
                cache_dir = str(getattr(self.config, "forecast_cache_dir", "forecast_cache"))
                feats = load_forecast_prior_features(cache_dir, ep_idx)
                self._forecast_prior_features = feats
                _decision_freq = int(getattr(self.config, "investment_freq", 6) or 6)
                _configured_prior_horizon = getattr(self.config, "forecast_prior_horizon_steps", None)
                if _configured_prior_horizon is not None:
                    _horizon_steps = int(_configured_prior_horizon or 6)
                else:
                    _fh = getattr(self.config, "forecast_horizons", None)
                    if isinstance(_fh, dict) and "short" in _fh:
                        _horizon_steps = int(_fh.get("short", 6) or 6)
                    else:
                        _horizon_steps = int(getattr(self.config, "short_horizon_steps", 6) or 6)
                _denom_floor = float(
                    getattr(
                        self.config,
                        "forecast_prior_denom_floor",
                        getattr(self.config, "minimum_price_filter", 10.0),
                    )
                )
                self._forecast_prior_processor = ConformalForecastPrior(
                    raw_features=feats,
                    decision_freq=_decision_freq,
                    horizon_steps=_horizon_steps,
                    window_size=int(getattr(self.config, "forecast_prior_window", 500) or 500),
                    min_samples=int(getattr(self.config, "forecast_prior_min_samples", 50) or 50),
                    hit_lcb_z=float(getattr(self.config, "forecast_prior_hit_lcb_z", 1.64)),
                    residual_quantile=float(getattr(self.config, "forecast_prior_residual_quantile", 0.70)),
                    default_residual=float(getattr(self.config, "forecast_prior_default_residual", 0.10)),
                    edge_gain=float(getattr(self.config, "forecast_prior_edge_gain", 3.0)),
                    error_hurdle=float(getattr(self.config, "forecast_prior_error_hurdle", 0.50)),
                    skill_power=float(getattr(self.config, "forecast_prior_skill_power", 1.0)),
                    max_abs_exposure=float(getattr(self.config, "forecast_prior_max_abs_exposure", 0.60)),
                    denom_floor=_denom_floor,
                    vol_half_life_steps=int(getattr(self.config, "forecast_prior_vol_half_life_steps", 288) or 288),
                    vol_target=float(getattr(self.config, "forecast_prior_vol_target", 0.15)),
                )
                self._last_forecast_prior = {}
                if feats is None:
                    msg = (
                        f"[FORECAST_PRIOR] Cache miss for episode {ep_idx} under "
                        f"'{cache_dir}'; cannot run enabled forecast-utilization variant."
                    )
                    if bool(getattr(self.config, "forecast_prior_fail_fast", True)):
                        raise RuntimeError(msg)
                    logger.warning(f"{msg} Investor prior will be zero.")
                else:
                    required_rows = int(max(1, getattr(self, "max_steps", feats.shape[0])))
                    if int(feats.shape[0]) < required_rows:
                        msg = (
                            f"[FORECAST_PRIOR] Cache for episode {ep_idx} has {feats.shape[0]} rows "
                            f"but environment requires {required_rows}; refusing partial enabled prior."
                        )
                        if bool(getattr(self.config, "forecast_prior_fail_fast", True)):
                            raise RuntimeError(msg)
                        logger.warning(f"{msg} Prior will be zero beyond cache length.")
                    logger.info(
                        f"[FORECAST_PRIOR] Loaded {feats.shape[0]} rows x "
                        f"{feats.shape[1]} cols from forecast cache "
                        f"(episode {ep_idx}); processor: "
                        f"dec_freq={_decision_freq}, horizon={_horizon_steps}."
                    )
            except Exception as e:
                logger.error(f"[FORECAST_PRIOR] Failed to load forecast cache: {e}")
                self._forecast_prior_features = None
                self._forecast_prior_processor = None
                self._last_forecast_prior = {}
                if bool(getattr(self.config, "forecast_prior_fail_fast", True)):
                    raise RuntimeError(
                        f"[FORECAST_PRIOR] Forecast utilization enabled but cache initialization failed: {e}"
                    ) from e
        else:
            self._forecast_prior_features = None
            self._forecast_prior_processor = None
            self._last_forecast_prior = {}

        self._fill_obs()
        return self._obs_buf, {}

    def _reset_financial_state(self):
        """CRITICAL FIX: Only reset episode tracking, PRESERVE financial gains"""
        # PRESERVE physical assets (they are permanent)
        # PRESERVE financial positions (gains should accumulate)
        # PRESERVE operational revenue (gains should accumulate)
        # PRESERVE cumulative performance (gains should accumulate)

        # Only reset operational state for new episode
        self.operational_state = {
            'battery_energy': self._get_initial_battery_energy(),
            'battery_discharge_power': 0.0,
        }

        # CRITICAL: DO NOT reset budget - let gains/losses accumulate
        # CRITICAL: DO NOT reset financial_positions - let trading gains accumulate
        # CRITICAL: DO NOT reset accumulated_operational_revenue - let operational gains accumulate
        # CRITICAL: DO NOT reset cumulative_mtm_pnl - keep the diagnostic MTM tracker accumulating
        # CRITICAL: DO NOT reset cumulative_returns - let return performance accumulate
        # CRITICAL: DO NOT reset distributed_profits - let profit distributions accumulate
        # CRITICAL: DO NOT reset investment_capital - let capital changes accumulate

        # Only reset per-step tracking (not cumulative)
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.last_realized_dnav = 0.0
        self.last_realized_dnav_return = 0.0
        self.last_realized_investor_dnav = 0.0
        self.last_realized_investor_dnav_return = 0.0
        self.last_realized_investor_return_denom = 1.0
        # Preserve investor-local sleeve state across PPO buffer resets.
        # Financial capital and positions intentionally accumulate here, so the
        # local return / drawdown tracker must remain continuous as well.
        if not hasattr(self, '_investor_local_return_history'):
            self._investor_local_return_history = deque(
                maxlen=max(1, int(getattr(self.config, 'investor_trading_history_lookback', 48)))
            )
            self._investor_local_path_value = 1.0
            self._investor_local_peak_value = 1.0
            self._investor_local_drawdown = 0.0
            self._investor_local_recent_mean = 0.0
            self._investor_local_recent_vol = 0.0
            self._investor_local_quality = 0.0
            self._investor_local_quality_delta = 0.0
        if not hasattr(self, '_investor_mean_history'):
            self._investor_mean_history = deque(
                maxlen=max(1, int(getattr(self.config, 'investor_mean_collapse_window', 256)))
            )
            self._investor_mean_abs_rolling = 0.0
            self._investor_mean_sign_consistency = 0.0
        if not hasattr(self, '_investor_mean_clip_hit_history'):
            self._investor_mean_clip_hit_history = deque(
                maxlen=max(1, int(getattr(self.config, 'investor_mean_clip_hit_window', 256)))
            )
            self._investor_mean_clip_hit_rate = 0.0
        self._last_investor_transaction_cost = 0.0
        self._last_investor_exposure_pretrade = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}
        self._meta_rule_last_update_step = -1
        self._meta_rule_signal_ema = 0.0
        self._meta_rule_target_capital_fraction = float(
            np.clip(
                getattr(self, 'capital_allocation_fraction', getattr(self.config, 'capital_allocation_fraction', 0.0)),
                getattr(self, 'META_CAP_MIN', getattr(self.config, 'meta_cap_min', 0.10)),
                getattr(self, 'META_CAP_MAX', getattr(self.config, 'meta_cap_max', 0.80)),
            )
        )
        

        # CRITICAL FIX: Recalculate NAV with current state (preserving financial gains)
        # Use current timestep for depreciation calculation
        current_t = getattr(self, 't', 0)
        current_nav = self._calculate_fund_nav(current_timestep=current_t)
        
        # Reset the cumulative-return baseline to the current NAV after reset.
        self._previous_nav = current_nav if current_nav > 1e-6 else float(self.init_budget)
        self._fund_nav_prev = current_nav if current_nav > 1e-6 else float(self.init_budget)

        # SAFETY: Ensure reward_calculator exists after reset (avoids fallback logging path)
        if getattr(self, 'reward_calculator', None) is None:
            try:
                post_capex_nav = float(self._calculate_fund_nav())
            except Exception:
                post_capex_nav = float(self.init_budget)
            self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
            self.reward_weights = dict(getattr(self.reward_calculator, 'reward_weights', {}))

    def _reset_financial_state_full(self):
        """FULL RESET: Only used at true episode end (end of data)"""
        logger.info(f"ðŸ’° FULL FINANCIAL RESET: Resetting all gains to initial state")

        # CRITICAL FIX: Reset assets_deployed flag to allow proper recalculation
        # This ensures physical_capex_deployed is recalculated consistently
        self.assets_deployed = False

        # Reset financial instruments to zero
        self.financial_positions = {
            'wind_instrument_value': 0.0,
            'solar_instrument_value': 0.0,
            'hydro_instrument_value': 0.0,
        }
        # Reset true MTM values to zero (exposure-based MTM is accumulated here)
        self.financial_mtm_positions = {
            'wind_instrument_value': 0.0,
            'solar_instrument_value': 0.0,
            'hydro_instrument_value': 0.0,
        }

        # Reset operational state
        self.operational_state = {
            'battery_energy': self._get_initial_battery_energy(),
            'battery_discharge_power': 0.0,
        }

        # CRITICAL FIX: Ensure total_fund_value is consistent before resetting budget
        if not hasattr(self, 'total_fund_value') or self.total_fund_value != self.init_budget:
            self.total_fund_value = float(self.init_budget)

        # CRITICAL: Recalculate trading_allocation_budget to ensure consistency
        self.trading_allocation_budget = self.total_fund_value * self.config.financial_allocation
        self.budget = float(self.trading_allocation_budget)

        # Reset performance tracking
        self.investment_capital = float(self.init_budget)
        self.distributed_profits = 0.0
        self.total_distributions = 0.0
        self.cumulative_returns = 0.0
        self._previous_nav = 0.0
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.last_realized_dnav = 0.0
        self.last_realized_dnav_return = 0.0
        self.last_realized_investor_dnav = 0.0
        self.last_realized_investor_dnav_return = 0.0
        self.last_realized_investor_return_denom = 1.0
        self._last_investor_transaction_cost = 0.0
        self._last_investor_exposure_pretrade = 0.0
        self.cumulative_mtm_pnl = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}
        self._meta_rule_last_update_step = -1
        self._meta_rule_signal_ema = 0.0
        self.capital_allocation_fraction = float(
            np.clip(
                getattr(self.config, 'capital_allocation_fraction', 0.60),
                getattr(self.config, 'meta_cap_min', 0.10),
                getattr(self.config, 'meta_cap_max', 0.80),
            )
        )
        self.investment_freq = int(max(1, getattr(self.config, 'investment_freq', 6)))
        self._meta_rule_target_capital_fraction = float(self.capital_allocation_fraction)

        # Reset accumulated operational revenue
        self.accumulated_operational_revenue = 0.0
        
        # CRITICAL FIX: Ensure self.t is 0 before calculating NAV to prevent depreciation differences
        # For reset, we always want t=0 (no depreciation)
        self.t = 0
        
        # CRITICAL FIX: Recalculate physical_capex_deployed from config to ensure consistency
        # This ensures consistent physical assets across runs
        plan = self.config.get_initial_asset_plan()
        total_capex = 0.0
        for asset_type, specs in plan.items():
            if asset_type == 'wind': 
                capex = specs['capacity_mw'] * self.asset_capex['wind_mw']
            elif asset_type == 'solar': 
                capex = specs['capacity_mw'] * self.asset_capex['solar_mw']
            elif asset_type == 'hydro': 
                capex = specs['capacity_mw'] * self.asset_capex['hydro_mw']
            elif asset_type == 'battery': 
                capex = specs['capacity_mwh'] * self.asset_capex['battery_mwh']
            else: 
                continue
            total_capex += capex
        
        # Check if scaling is needed (same logic as _deploy_initial_assets_once)
        if total_capex > self.physical_allocation_budget:
            scale_factor = (self.physical_allocation_budget * 0.95) / total_capex
        else:
            scale_factor = 1.0
        
        # Recalculate physical_capex_deployed with scaling
        # CRITICAL FIX: Use explicit float() conversions to ensure identical values between tiers
        physical_book_value = 0.0
        for asset_type, specs in plan.items():
            if asset_type == 'wind': 
                capacity = float(specs['capacity_mw'] * scale_factor)
                capex = float(capacity * self.asset_capex['wind_mw'])
            elif asset_type == 'solar': 
                capacity = float(specs['capacity_mw'] * scale_factor)
                capex = float(capacity * self.asset_capex['solar_mw'])
            elif asset_type == 'hydro': 
                capacity = float(specs['capacity_mw'] * scale_factor)
                capex = float(capacity * self.asset_capex['hydro_mw'])
            elif asset_type == 'battery': 
                capacity = float(specs['capacity_mwh'] * scale_factor)
                capex = float(capacity * self.asset_capex['battery_mwh'])
            else: 
                continue
            physical_book_value = float(physical_book_value + capex)
        
        # CRITICAL FIX: Store recalculated physical_capex_deployed to ensure consistency
        self.physical_capex_deployed = physical_book_value
        
        # CRITICAL FIX: Reset physical assets to match recalculated values
        self.physical_assets = {
            'wind_capacity_mw': 0.0,
            'solar_capacity_mw': 0.0,
            'hydro_capacity_mw': 0.0,
            'battery_capacity_mwh': 0.0,
        }
        # CRITICAL FIX: Use explicit float() conversions to ensure identical values between tiers
        for asset_type, specs in plan.items():
            if asset_type == 'wind':
                self.physical_assets['wind_capacity_mw'] = float(specs['capacity_mw'] * scale_factor)
            elif asset_type == 'solar':
                self.physical_assets['solar_capacity_mw'] = float(specs['capacity_mw'] * scale_factor)
            elif asset_type == 'hydro':
                self.physical_assets['hydro_capacity_mw'] = float(specs['capacity_mw'] * scale_factor)
            elif asset_type == 'battery':
                self.physical_assets['battery_capacity_mwh'] = float(specs['capacity_mwh'] * scale_factor)
        
        # Mark as deployed again (after recalculation)
        self.assets_deployed = True
        
        # CRITICAL DEBUG: Log reset details for consistency verification
        logger.info("[RESET] Physical assets recalculated:")
        logger.info(f"  Physical allocation budget: {self.physical_allocation_budget:,.0f} DKK")
        logger.info(f"  Total CAPEX required: {total_capex:,.0f} DKK")
        logger.info(f"  Scale factor: {scale_factor:.10f}")  # More precision
        logger.info(f"  Physical CAPEX deployed: {self.physical_capex_deployed:,.0f} DKK")
        logger.info(f"  Physical book value: {physical_book_value:,.0f} DKK")
        logger.info(f"  Wind: {self.physical_assets['wind_capacity_mw']:.10f} MW")  # More precision
        logger.info(f"  Solar: {self.physical_assets['solar_capacity_mw']:.10f} MW")
        logger.info(f"  Hydro: {self.physical_assets['hydro_capacity_mw']:.10f} MW")
        logger.info(f"  Battery: {self.physical_assets['battery_capacity_mwh']:.10f} MWh")

        # At reset, operational revenue and financial MTM are zero.
        operational_revenue_value = 0.0
        financial_mtm_value = 0.0

        # Use trading_allocation_budget directly to ensure consistency
        trading_cash = float(self.trading_allocation_budget)

        # Calculate NAV from the single source of truth (FinancialEngine) to avoid duplicated NAV math.
        # Keep reset semantics identical: OpRev=0, MTM=0 at t=0.
        try:
            from financial_engine import FinancialEngine
            initial_nav = FinancialEngine.calculate_fund_nav(
                budget=trading_cash,
                physical_assets=self.physical_assets,
                asset_capex=self.asset_capex,
                financial_mtm_values={
                    'wind_instrument_value': 0.0,
                    'solar_instrument_value': 0.0,
                    'hydro_instrument_value': 0.0,
                },
                accumulated_operational_revenue=operational_revenue_value,
                current_timestep=0,
                config=self.config,
            )
        except Exception as e:
            logger.error(f"[RESET_NAV_FATAL] Failed to initialize NAV from FinancialEngine: {e}")
            raise RuntimeError("[RESET_NAV_FATAL] Failed to initialize NAV at reset") from e
        
        # CRITICAL FIX: Force NAV to be consistent by setting equity explicitly
        self.equity = float(initial_nav)
        self._fund_nav_prev = float(initial_nav)
        
        # CRITICAL FIX: Keep t=0 after reset (don't restore original_t)
        # This ensures NAV calculations after reset use t=0 (no depreciation)
        # t will be incremented in step() function
        self.t = 0
        
        # CRITICAL FIX: Store the initial NAV from reset to use for logging at timestep 0
        # This ensures identical NAV values at timestep 0, before any trades are executed
        self._initial_nav_from_reset = float(initial_nav)
        
        # Log for debugging
        if hasattr(self, 'config'):
            # Add a more detailed log to check components if needed
            if abs(initial_nav - self.init_budget) > 1000: # Log if NAV is significantly off from init_budget
                 logger.warning(f"[RESET] Initial NAV components: Cash={trading_cash:,.0f}, "
                               f"Assets={physical_book_value:,.0f}, OpRev={operational_revenue_value:,.0f}, MTM={financial_mtm_value:,.0f}")
            logger.info(f"[RESET] Initial NAV after full reset: {initial_nav:,.0f} DKK (${initial_nav * self.config.dkk_to_usd_rate / 1e6:.4f}M USD), init_budget: {self.init_budget:,.0f} DKK")

    def step(self, actions: Dict[str, Any]):
        # FIXED: Safety check for self.t attribute
        current_timestep = getattr(self, 't', 0)
        if current_timestep >= self.max_steps:
            return self._terminal_step()

        try:
            i = self.t
            # Persisting step counter for warmups/regularizers:
            # Use config.training_global_step because episode training creates a NEW env each episode.
            try:
                if self.config is not None:
                    self.config.training_global_step = int(getattr(self.config, "training_global_step", 0)) + 1
            except Exception:
                pass


            acts = self._validate_actions(actions)

            # CRITICAL FIX: Store actions as plain dict to prevent tensor memory leaks
            # Convert any tensors to numpy arrays to avoid keeping computation graphs alive
            self._last_actions = {}
            for key, val in acts.items():
                # Normalize to numpy early
                if hasattr(val, 'detach'):  # PyTorch tensor
                    v = val.detach().cpu().numpy()
                elif hasattr(val, 'copy'):  # Numpy array
                    v = val.copy()
                else:  # Scalar or other
                    v = val

                # For logging: store structured action dicts (so CSVs don't become NaN/empty)
                try:
                    if key == 'investor_0':
                        arr = np.array(v, dtype=np.float32).reshape(-1)
                        # Exposure-only policy output (1D). Keep keys for logging compatibility.
                        exposure_raw = float(arr[0]) if arr.size > 0 else 1.0
                        self._last_actions[key] = {
                            'exposure_raw': exposure_raw,
                            'wind': 0.0,
                            'solar': 0.0,
                            'hydro': 0.0,
                        }
                        # Preserve raw for debugging if needed
                        self._last_actions[key + '_raw'] = arr.astype(float).tolist()

                        # Build execution diagnostics immediately. On decision
                        # rows this is the target that will be traded below;
                        # between decision rows it must represent the actually
                        # held exposure, not a hypothetical fresh target.
                        try:
                            exposure_raw_exec = float(np.clip(arr[0] if arr.size > 0 else 1.0, -1.0, 1.0))
                            available_capital_exec = float(max(self.budget * self.capital_allocation_fraction, 0.0))
                            max_pos_size_exec = float(max(available_capital_exec * self.config.max_position_size, 1.0))
                            current_alloc = np.array(
                                [
                                    float(self.financial_positions.get("wind_instrument_value", 0.0)) / max_pos_size_exec,
                                    float(self.financial_positions.get("solar_instrument_value", 0.0)) / max_pos_size_exec,
                                    float(self.financial_positions.get("hydro_instrument_value", 0.0)) / max_pos_size_exec,
                                ],
                                dtype=np.float32,
                            )
                            current_alloc = np.clip(current_alloc, -1.0, 1.0).astype(np.float32)
                            current_exposure_exec = float(self._estimate_current_investor_exposure(available_capital_exec))

                            if i != 0 and self._is_investor_decision_step(i):
                                current_exposure_exec, control_signal_exec, delta_exposure_exec, exposure, prior_diag_exec = (
                                    self._map_investor_action_to_exposure(
                                        exposure_raw_exec,
                                        tradeable_capital=available_capital_exec,
                                        timestep=i,
                                    )
                                )

                                if bool(getattr(self.config, "investor_use_risk_budget_weights", True)):
                                    w = np.asarray(self._get_risk_budget_allocation(), dtype=np.float32)
                                else:
                                    w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                                w = (w / float(np.sum(np.abs(w)))).astype(np.float32)
                                alloc = (exposure * w).astype(np.float32)
                            else:
                                prior_diag_exec = (
                                    self._get_forecast_prior_snapshot(i)
                                    if bool(getattr(self.config, "enable_forecast_utilization", False))
                                    else {"prior_exposure": 0.0, "active": False, "reason": "disabled"}
                                )
                                control_signal_exec = 0.0
                                delta_exposure_exec = 0.0
                                exposure = current_exposure_exec
                                alloc = current_alloc
                                w = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                            self._last_actions["investor_0_exec"] = {
                                "wind": float(alloc[0]),
                                "solar": float(alloc[1]),
                                "hydro": float(alloc[2]),
                                "exposure": float(exposure),
                                "current_exposure": float(current_exposure_exec),
                                "delta_exposure": float(delta_exposure_exec),
                                "control_signal": float(control_signal_exec),
                                "forecast_prior_exposure": float(prior_diag_exec.get("prior_exposure", 0.0)),
                                "forecast_prior_active": float(1.0 if prior_diag_exec.get("active", False) else 0.0),
                                "forecast_prior_skill": float(prior_diag_exec.get("skill", 0.0)),
                                "forecast_prior_hit_lcb": float(prior_diag_exec.get("hit_lcb", 0.5)),
                                "forecast_prior_residual_q": float(prior_diag_exec.get("residual_q", 0.0)),
                                "forecast_prior_magnitude": float(prior_diag_exec.get("magnitude", 0.0)),
                                "forecast_prior_edge_excess": float(prior_diag_exec.get("edge_excess", 0.0)),
                                "forecast_prior_blend": float(prior_diag_exec.get("prior_blend", 0.0)),
                                "prior_target": float(prior_diag_exec.get("prior_target", exposure)),
                                "forecast_prior_reason": str(prior_diag_exec.get("reason", "")),
                                "w_wind": float(w[0]),
                                "w_solar": float(w[1]),
                                "w_hydro": float(w[2]),
                            }
                            self._last_investor_action_exec = alloc.copy()
                        except Exception:
                            pass
                    elif key == 'battery_operator_0':
                        if isinstance(self.action_spaces.get(key), spaces.Discrete):
                            if isinstance(v, (list, tuple, np.ndarray)):
                                action_idx = int(np.round(np.asarray(v).reshape(-1)[0])) if np.size(v) > 0 else 1
                            else:
                                action_idx = int(np.round(float(v)))
                            action_idx = int(np.clip(action_idx, 0, self.action_spaces[key].n - 1))
                            self._last_actions[key] = {
                                'raw': float(action_idx),
                                'mode': action_idx,
                                'charge': 1.0 if action_idx == 0 else 0.0,
                                'idle': 1.0 if action_idx == 1 else 0.0,
                                'discharge': 1.0 if action_idx == 2 else 0.0,
                            }
                            self._last_actions[key + '_raw'] = [float(action_idx)]
                        else:
                            arr = np.array(v, dtype=np.float32).reshape(-1)
                            u_raw = float(arr[0]) if arr.size > 0 else 0.0
                            self._last_actions[key] = {
                                'raw': u_raw,
                                'charge': float(max(0.0, -u_raw)),
                                'discharge': float(max(0.0, u_raw)),
                            }
                            self._last_actions[key + '_raw'] = [u_raw]
                    else:
                        self._last_actions[key] = v
                except Exception:
                    # Fall back to raw value if anything unexpected occurs
                    self._last_actions[key] = v

            # meta & risk knobs â€” applied only on their respective strategic cadences.
            # Between decision steps the last-set value persists naturally via self.risk_exposure_cap
            # and self.capital_allocation_fraction, giving a stable context to the investor.
            if self._is_risk_decision_step(i):
                self._apply_risk_control(acts['risk_controller_0'])
            if self._is_meta_decision_step(i):
                self._apply_meta_control(acts['meta_controller_0'])

            # ------------------------------------------------------------------
            # CRITICAL: Add MTM BEFORE trades - price move (i-1â†’i) applies to PRE-trade positions.
            # Correct accounting: we held these positions during the price move.
            self._add_mtm_for_step(i)

            # investor + battery ops (battery returns realized cash delta)
            # CRITICAL FIX: Pass i (timestep) to _execute_investor_trades for consistency
            trade_amount = self._execute_investor_trades(acts['investor_0'], timestep=i)
            battery_cash_delta = self._execute_battery_ops(acts['battery_operator_0'], i)

            # FIXED: finance update (costs, realized rev incl. battery cash; MTM already added above)
            # CRITICAL FIX: Ensure i is the correct timestep (should be self.t, but verify)
            # At timestep 0, i should be 0 to ensure no depreciation
            if i != self.t:
                logger.warning(f"[NAV_FIX] Timestep mismatch: i={i}, self.t={self.t}")
            financial = self._update_finance(i, trade_amount, battery_cash_delta)

            # regime updates & rewards
            self._update_market_conditions(i)
            self._update_risk_snapshots(i)

            # FIXED: assign rewards
            self._assign_rewards(financial)

            # step forward
            self.t += 1
            self.step_in_episode = self.t

            index = max(0, self.t - 1)
            self._fill_obs()

            self._populate_info(index, financial, acts)

            return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

        except Exception as e:
            msg = f"step error at t={self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _warn_once(self, bucket_attr: str, key: Any, msg: str, *args) -> None:
        """Log a warning once per (bucket, key) pair to prevent noisy logs."""
        seen = getattr(self, bucket_attr, None)
        if seen is None:
            seen = set()
            setattr(self, bucket_attr, seen)
        try:
            marker = (str(key),)
            if marker in seen:
                return
            logger.warning(msg, *args)
            seen.add(marker)
        except Exception:
            # Avoid secondary failures from logging path.
            pass

    @staticmethod
    def _clip01(value: Any) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _clip11(value: Any) -> float:
        return float(np.clip(float(value), -1.0, 1.0))

    def _to_numpy_safe(self, a_in):
        """Bring tensors (torch/jax/etc.) to CPU numpy by duck-typing."""
        obj_name = type(a_in).__name__
        try:
            for method_name in ("detach", "cpu", "numpy"):
                fn = getattr(a_in, method_name, None)
                if not callable(fn):
                    continue
                try:
                    a_in = fn()
                except Exception as e:
                    self._warn_once(
                        "_action_convert_warned",
                        f"{method_name}:{obj_name}",
                        "[ACTION_CONVERT] %s() failed for %s: %s",
                        method_name,
                        obj_name,
                        e,
                    )
        except Exception as e:
            self._warn_once(
                "_action_convert_warned",
                f"outer:{obj_name}",
                "[ACTION_CONVERT] unexpected conversion error for %s: %s",
                obj_name,
                e,
            )
        return a_in

    def _validate_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        self._clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        for agent in self.possible_agents:
            space = self.action_spaces[agent]
            a = actions.get(agent, None)

            key = ("investor" if agent.startswith("investor")
                   else "battery" if agent.startswith("battery")
                   else "risk" if agent.startswith("risk")
                   else "meta")

            if isinstance(space, spaces.Discrete):
                if a is None:
                    out[agent] = int(space.n // 2)
                    continue
                a = self._to_numpy_safe(a)
                if isinstance(a, (list, tuple, np.ndarray)) and np.size(a) > 0:
                    a_int = int(np.round(np.asarray(a).reshape(-1)[0]))
                else:
                    a_int = int(np.round(float(a)))
                clipped = int(np.clip(a_int, 0, space.n - 1))
                out[agent] = clipped
                self._clip_counts[key] += int(clipped != a_int)
                continue

            # fallback to midpoint if missing
            if a is None:
                mid = (space.low + space.high) / 2.0
                out[agent] = np.array(mid, dtype=np.float32).reshape(space.shape)
                continue

            a = self._to_numpy_safe(a)
            arr = np.array(a, dtype=np.float32).flatten()

            need = int(np.prod(space.shape))
            if arr.size != need:
                mid = (space.low + space.high) / 2.0
                pad_val = float(mid.flatten()[0]) if hasattr(mid, "flatten") else float(mid)
                if arr.size < need:
                    arr = np.concatenate([arr, np.full(need - arr.size, pad_val, np.float32)])
                else:
                    arr = arr[:need]

            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            before = arr.copy()
            arr = np.minimum(np.maximum(arr, space.low), space.high).astype(np.float32)
            out[agent] = arr.reshape(space.shape)

            self._clip_counts[key] += int(np.any(np.abs(before - arr) > 1e-12))

        return out

    def _apply_risk_control(self, risk_action: np.ndarray):
        """
        REFACTORED: Apply risk control using TradingEngine.
        """
        try:
            from trading_engine import TradingEngine
            risk_cap = float(
                TradingEngine.apply_risk_control(
                    risk_action,
                    risk_exposure_cap_min=float(getattr(self.config, "risk_exposure_cap_min", 0.25)),
                    risk_exposure_cap_max=float(getattr(self.config, "risk_exposure_cap_max", 1.0)),
                )
            )
            # Keep risk_multiplier for backward-compatible logs/observations, but
            # its live meaning is now "max sleeve exposure" instead of leverage.
            self.risk_multiplier = risk_cap
            self.risk_exposure_cap = risk_cap
        except Exception as e:
            msg = f"Risk control failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _is_investor_decision_step(self, t: Optional[int] = None) -> bool:
        """
        Whether the investor is allowed to act on timestep ``t``.

        This is the single source of truth for the investor trading cadence.
        Tier-1 observations, info/debug logs, Tier1 sample collection, and
        cadence-gated controllers should all derive from this helper.
        """
        current_t = int(getattr(self, "t", 0) if t is None else t)
        investment_freq = int(max(getattr(self, "investment_freq", 1) or 1, 1))
        return bool(current_t > 0 and current_t % investment_freq == 0)

    def _is_risk_decision_step(self, t: Optional[int] = None) -> bool:
        """Whether the risk controller should apply a new cap on timestep ``t``.

        Defaults to every 12 steps (2 hours @ 10-min bars).  Must be a multiple
        of investment_freq so the cap change always aligns with an investor
        decision point.
        """
        current_t = int(getattr(self, "t", 0) if t is None else t)
        freq = int(max(getattr(self.config, "risk_action_freq", 12), 1))
        return bool(current_t > 0 and current_t % freq == 0)

    def _is_meta_decision_step(self, t: Optional[int] = None) -> bool:
        """Whether the meta controller should update capital allocation on timestep ``t``.

        Defaults to every 24 steps (4 hours @ 10-min bars).
        """
        current_t = int(getattr(self, "t", 0) if t is None else t)
        freq = int(max(getattr(self.config, "meta_action_freq", 24), 1))
        return bool(current_t > 0 and current_t % freq == 0)

    def _decision_step_flag(self, t: Optional[int] = None) -> float:
        """Float version of ``_is_investor_decision_step`` for obs/info/logging."""
        return 1.0 if self._is_investor_decision_step(t) else 0.0

    def _apply_meta_control(self, meta_action: np.ndarray):
        """
        Apply meta control: RL policy sets capital allocation fraction and investment freq.
        """
        try:
            from trading_engine import TradingEngine
            proposed_capital_fraction, proposed_investment_freq = TradingEngine.apply_meta_control(
                meta_action=meta_action,
                meta_cap_min=self.META_CAP_MIN,
                meta_cap_max=self.META_CAP_MAX,
                meta_freq_min=self.META_FREQ_MIN,
                meta_freq_max=self.META_FREQ_MAX,
                confidence_signal=0.5,
                disable_confidence_scaling=True,
            )
            self.capital_allocation_fraction = float(proposed_capital_fraction)
            self.investment_freq = int(proposed_investment_freq)
        except Exception as e:
            msg = f"Meta control failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _compute_rule_based_risk_action(self) -> np.ndarray:
        risk_min = float(getattr(self.config, "risk_exposure_cap_min", 0.25))
        risk_max = float(getattr(self.config, "risk_exposure_cap_max", 1.0))
        drawdown = float(
            np.clip(
                getattr(self.reward_calculator, "current_drawdown", 0.0) if self.reward_calculator else 0.0,
                0.0,
                1.0,
            )
        )
        overall_risk = float(np.clip(getattr(self, "overall_risk_snapshot", 0.0), 0.0, 1.0))
        market_vol = float(np.clip(getattr(self, "market_volatility", 0.0), 0.0, 1.0))
        risk_pressure = float(np.clip(0.50 * overall_risk + 0.30 * market_vol + 0.20 * min(drawdown / 0.10, 1.0), 0.0, 1.0))
        target_cap = float(np.clip(risk_max - risk_pressure * (risk_max - risk_min), risk_min, risk_max))
        span = max(risk_max - risk_min, 1e-6)
        action = float(np.clip(2.0 * (target_cap - risk_min) / span - 1.0, -1.0, 1.0))
        return np.array([action], dtype=np.float32)

    def _compute_rule_based_meta_action(self) -> np.ndarray:
        meta_min = float(getattr(self, "META_CAP_MIN", getattr(self.config, "meta_cap_min", 0.10)))
        meta_max = float(getattr(self, "META_CAP_MAX", getattr(self.config, "meta_cap_max", 0.80)))
        neutral_cap = float(
            np.clip(
                getattr(self.config, "capital_allocation_fraction", meta_min),
                meta_min,
                meta_max,
            )
        )
        span = max(meta_max - meta_min, 1e-6)
        ema_alpha = float(np.clip(getattr(self.config, "meta_rule_signal_ema_alpha", 0.10), 0.0, 1.0))
        current_capital_fraction = float(
            np.clip(
                getattr(self, "_meta_rule_target_capital_fraction", getattr(self, "capital_allocation_fraction", neutral_cap)),
                meta_min,
                meta_max,
            )
        )
        update_on_decision_step = bool(
            getattr(self.config, "meta_rule_update_only_on_investor_step", True)
        )
        is_decision_step = self._is_investor_decision_step()
        if update_on_decision_step and not is_decision_step:
            hold_action = float(
                np.clip(2.0 * (current_capital_fraction - meta_min) / span - 1.0, -1.0, 1.0)
            )
            return np.array([hold_action], dtype=np.float32)
        drawdown = float(
            np.clip(
                getattr(self.reward_calculator, "current_drawdown", 0.0) if self.reward_calculator else 0.0,
                0.0,
                1.0,
            )
        )
        overall_risk = float(np.clip(getattr(self, "overall_risk_snapshot", 0.0), 0.0, 1.0))
        realized_return = float(getattr(self, "last_realized_investor_dnav_return", 0.0))
        return_scale = float(max(getattr(self.config, "investor_trading_return_scale", 0.003), 1e-6))
        edge_signal = float(np.clip(realized_return / return_scale, -1.0, 1.0))
        mtm_scale = float(max(0.01 * float(getattr(self, "init_budget", 1.0)), 1.0))
        perf_signal = float(np.clip(float(getattr(self, "cumulative_mtm_pnl", 0.0)) / mtm_scale, -1.0, 1.0))
        raw_meta_signal = float(np.clip(0.70 * edge_signal + 0.30 * perf_signal, -1.0, 1.0))
        prev_signal_ema = float(np.clip(getattr(self, "_meta_rule_signal_ema", 0.0), -1.0, 1.0))
        signal_ema = float(np.clip(prev_signal_ema + ema_alpha * (raw_meta_signal - prev_signal_ema), -1.0, 1.0))
        self._meta_rule_signal_ema = signal_ema
        risk_pressure = float(np.clip(0.60 * min(drawdown / 0.10, 1.0) + 0.40 * overall_risk, 0.0, 1.0))
        target_cap = float(
            np.clip(
                neutral_cap + 0.20 * span * signal_ema - 0.35 * span * risk_pressure,
                meta_min,
                meta_max,
            )
        )
        deadband = float(max(getattr(self.config, "meta_rule_cap_deadband", 0.03), 0.0))
        if abs(target_cap - current_capital_fraction) < deadband:
            target_cap = current_capital_fraction
        action = float(np.clip(2.0 * (target_cap - meta_min) / span - 1.0, -1.0, 1.0))
        return np.array([action], dtype=np.float32)

    def get_rule_based_agent_action(self, agent_name: str, obs=None) -> np.ndarray:
        if agent_name == "risk_controller_0":
            return self._compute_rule_based_risk_action()
        if agent_name == "meta_controller_0":
            return self._compute_rule_based_meta_action()
        action_space = getattr(self, "action_spaces", {}).get(agent_name, None)
        if isinstance(action_space, spaces.Discrete):
            return int(action_space.n // 2)
        shape = getattr(action_space, "shape", (1,))
        return np.zeros(shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Now using portfolio-level _calculate_portfolio_hedge_intensity() instead
    # ------------------------------------------------------------------
    def _get_risk_budget_allocation(self) -> np.ndarray:
        """
        Get risk budget allocation for hedge distribution across sleeves
        Based on config risk budgets, not separate price bets
        """
        try:
            if hasattr(self.config, 'risk_budget_allocation') and isinstance(self.config.risk_budget_allocation, dict):
                # Use config-defined risk budgets (new dict format)
                allocation = self.config.risk_budget_allocation
                return np.array([
                    allocation.get('wind', 0.40),
                    allocation.get('solar', 0.35),
                    allocation.get('hydro', 0.25)
                ])
            else:
                # Default risk budgets based on capacity and volatility
                return np.array([0.40, 0.35, 0.25])  # 40% wind, 35% solar, 25% hydro

        except Exception as e:
            # Fallback to equal allocation
            return np.array([1/3, 1/3, 1/3])

    def _calculate_portfolio_hedge_intensity(self) -> float:
        """
        Calculate portfolio-level hedge intensity based on market conditions
        Single-price logic: one intensity for the entire portfolio
        """
        try:
            # Get current market data
            current_data = self.data.iloc[min(self.t, len(self.data) - 1)]
            current_price = float(current_data.get('price', 250.0))

            # Calculate price volatility (last 20 steps)
            lookback = min(20, self.t)
            if lookback < 5:
                return 1.0  # Default intensity

            start_idx = max(0, self.t - lookback)
            end_idx = min(self.t + 1, len(self.data))
            recent_prices = self.data.iloc[start_idx:end_idx]['price'].values

            price_volatility = float(np.std(recent_prices)) / max(float(np.mean(recent_prices)), 1.0)

            # Total portfolio generation vs expected
            total_current_gen = (float(current_data.get('wind', 0)) +
                               float(current_data.get('solar', 0)) +
                               float(current_data.get('hydro', 0))) / 1000.0

            wind_cap = self.physical_assets.get('wind_capacity_mw', 225)
            solar_cap = self.physical_assets.get('solar_capacity_mw', 100)
            hydro_cap = self.physical_assets.get('hydro_capacity_mw', 40)
            expected_total_gen = (wind_cap * 0.35 + solar_cap * 0.20 + hydro_cap * 0.45) / 1000.0

            generation_shortfall = max(0, (expected_total_gen - total_current_gen) / expected_total_gen)

            # Base intensity: higher when generation is low or prices are volatile
            base_intensity = 1.0 + generation_shortfall * 0.5  # Up to 1.5x for major shortfall
            volatility_multiplier = 1.0 + min(price_volatility, 0.5)  # Up to 1.5x for high volatility

            portfolio_intensity = base_intensity * volatility_multiplier

            # Cap intensity to reasonable bounds - WIDENED FROM [0.5, 2.0] TO [0.3, 2.5] FOR baseline LEARNING
            return float(np.clip(portfolio_intensity, 0.3, 2.5))

        except Exception as e:
            raise RuntimeError(
                f"[PORTFOLIO_HEDGE_INTENSITY_FATAL] Failed at step {getattr(self, 't', 'N/A')}: {e}"
            ) from e

    # def set_wrapper_reference(self, wrapper_env):
    #     """CRITICAL: Set reference to wrapper for profit-seeking expert"""
    #     self._wrapper_ref = wrapper_env

    # Optional policy encoders may learn richer feature relationships.
    # Forecast models (ANN/LSTM/SVR/RF) are backend signals only; observations stay fixed.
    # NOT rule-based expert suggestions. Expert suggestions interfered with PPO learning and are deprecated

    # ------------------------------------------------------------------
    # FIXED: Financial Instrument Trading (Separate from Physical Assets)
    # ------------------------------------------------------------------

    def _execute_investor_trades(self, inv_action: np.ndarray, timestep: Optional[int] = None) -> float:
        """
        CORRECTED & FINAL: Executes trades based DIRECTLY on the RL agent's action.

        The agent's action vector [-1, 1] is mapped to a target allocation
        of the available trading capital. The RL agent is always in control.

        Returns total traded notional for transaction costs
        """
        # CRITICAL FIX: Use timestep parameter if provided, otherwise use self.t
        t = timestep if timestep is not None else getattr(self, 't', 0)
        
        # Trading is only allowed at the specified frequency
        # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
        # CRITICAL FIX: Prevent trading at timestep 0 to ensure identical initial NAV
        # At timestep 0, we want to log the reset NAV before any trades are executed
        if t == 0:
            self._last_value_diag = {}
            return 0.0  # No trading at timestep 0 - ensures identical initial NAV
        if not self._is_investor_decision_step(t):
            self._last_value_diag = {}
            return 0.0

        # Clean sizing contract: hard drawdown disables the sleeve by forcing it
        # flat, not by silently skipping trades while keeping stale exposure.
        force_flat_exposure = bool(not self.reward_calculator.trading_enabled)

        try:
            prev_decision_anchor = int(
                getattr(
                    self,
                    "_last_investor_decision_step",
                    max(0, t - int(max(getattr(self, "investment_freq", 1) or 1, 1))),
                )
            )
            self._prev_investor_decision_step = int(max(0, min(prev_decision_anchor, t)))
            self._last_investor_decision_step = int(t)

            # === STEP 1: Determine available capital for this trade ===
            available_capital = self.budget * self.capital_allocation_fraction
            risk_exposure_cap = float(
                np.clip(
                    getattr(self, 'risk_exposure_cap', getattr(self, 'risk_multiplier', 1.0)),
                    getattr(self.config, 'risk_exposure_cap_min', 0.25),
                    getattr(self.config, 'risk_exposure_cap_max', 1.0),
                )
            )

            # Structural cleanup: Tier-1 investor execution should not depend on
            # an extra dormant strategy multiplier layer. Meta allocation,
            # risk control, and the investor policy already define the live size.
            tradeable_capital = float(max(available_capital, 0.0))

            # DEBUG: Log risk multiplier application (always applied now)
            # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
            if t % 500 == 0 and t > 0:
                logger.debug(
                    f"[SIZING_MULT] Step {t}: "
                    f"cap_alloc={self.capital_allocation_fraction:.4f}, "
                    f"risk_cap={risk_exposure_cap:.4f}"
                )

            # === STEP 2: Map the agent's normalized control signal to target exposure ===
            # In healthy Tier-1 mode, the investor controls an exposure adjustment by
            # default. This makes "hold" a near-zero action instead of requiring
            # large persistent actions just to maintain a position.
            a = np.asarray(inv_action, dtype=np.float32).reshape(-1)
            exposure_raw = float(a[0]) if a.size >= 1 else 1.0
            current_exposure, control_signal, delta_exposure, exposure, prior_diag = self._map_investor_action_to_exposure(
                exposure_raw,
                tradeable_capital=tradeable_capital,
                timestep=t,
            )
            if force_flat_exposure:
                exposure = 0.0
            else:
                exposure = float(np.clip(exposure, -risk_exposure_cap, risk_exposure_cap))
            delta_exposure = float(np.clip(exposure - current_exposure, -1.0, 1.0))
            # Log exposure at decision cadence for action-scale debugging
            try:
                inv_freq = int(getattr(self, "investment_freq", 6))
            except Exception:
                inv_freq = 6
            if t % inv_freq == 0:
                logger.debug(
                    f"[INVESTOR_EXPOSURE_RAW] t={t} exposure_raw={exposure_raw:.4f} "
                    f"current_exposure={current_exposure:.4f} "
                    f"control_signal={control_signal:.4f} "
                    f"delta_exposure={delta_exposure:.4f} "
                    f"exposure_final={exposure:.4f} "
                    f"risk_cap={risk_exposure_cap:.4f} forced_flat={int(force_flat_exposure)} "
                    f"prior={float(prior_diag.get('prior_exposure', 0.0)):.4f} "
                    f"prior_active={int(bool(prior_diag.get('active', False)))} "
                )
            self._last_risk_multiplier = float(risk_exposure_cap)
            self._last_tradeable_capital = float(tradeable_capital)

            # In single-price mode, distribute the aggregate hedge across bookkeeping
            # sleeves using the configured risk budget instead of fake equal weights.
            if bool(getattr(self.config, "investor_use_risk_budget_weights", True)):
                w = np.asarray(self._get_risk_budget_allocation(), dtype=np.float32)
            else:
                w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            w = (w / float(np.sum(np.abs(w)))).astype(np.float32)

            alloc = (exposure * w).astype(np.float32)
            action_wind, action_solar, action_hydro = float(alloc[0]), float(alloc[1]), float(alloc[2])

            # CRITICAL: Store executed (post-simplex) action for logging + penalties.
            # Without this, logs/regularizers can target the raw policy output while the environment trades on
            # a different normalized action, which breaks our saturation metrics and allows "penalty gaming".
            try:
                if not hasattr(self, "_last_actions") or not isinstance(getattr(self, "_last_actions", None), dict):
                    self._last_actions = {}
                self._last_actions["investor_0_exec"] = {
                    # executed allocation (what the environment actually trades on)
                    "wind": float(action_wind),
                    "solar": float(action_solar),
                    "hydro": float(action_hydro),
                    # extra interpretability fields
                    "exposure": float(exposure),
                    "current_exposure": float(current_exposure),
                    "delta_exposure": float(delta_exposure),
                    "control_signal": float(control_signal),
                    "forecast_prior_exposure": float(prior_diag.get("prior_exposure", 0.0)),
                    "forecast_prior_active": float(1.0 if prior_diag.get("active", False) else 0.0),
                    "forecast_prior_skill": float(prior_diag.get("skill", 0.0)),
                    "forecast_prior_hit_lcb": float(prior_diag.get("hit_lcb", 0.5)),
                    "forecast_prior_residual_q": float(prior_diag.get("residual_q", 0.0)),
                    "forecast_prior_magnitude": float(prior_diag.get("magnitude", 0.0)),
                    "forecast_prior_edge_excess": float(prior_diag.get("edge_excess", 0.0)),
                    "forecast_prior_blend": float(prior_diag.get("prior_blend", 0.0)),
                    "prior_target": float(prior_diag.get("prior_target", exposure)),
                    "forecast_prior_reason": str(prior_diag.get("reason", "")),
                    "w_wind": float(w[0]),
                    "w_solar": float(w[1]),
                    "w_hydro": float(w[2]),
                }
                self._last_investor_action_exec = np.array([action_wind, action_solar, action_hydro], dtype=np.float32)
            except Exception:
                pass

            # Single aggregate sizing: use one cap and split equally across assets.
            max_pos_size = tradeable_capital * self.config.max_position_size

            # Allocate proportional to weights; avoid extra 1/3 shrink so exposure maps 1:1 to max_pos_size.
            target_wind = action_wind * max_pos_size
            target_solar = action_solar * max_pos_size
            target_hydro = action_hydro * max_pos_size

            # Log combined multipliers periodically
            # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
            if t % 500 == 0 and t > 0:
                logger.info(f"[SIZING] Step {t}: exposure={exposure:.2f}, "
                           f"risk_cap={risk_exposure_cap:.2f}, "
                           f"max_pos={max_pos_size:,.0f}")

            # === STEP 3: STRATEGY 3 - MTM Loss Management ===
            # Check for large unrealized losses and force exit if threshold exceeded
            # This prevents positions from dragging NAV down
            current_wind = self.financial_positions.get('wind_instrument_value', 0.0)
            current_solar = self.financial_positions.get('solar_instrument_value', 0.0)
            current_hydro = self.financial_positions.get('hydro_instrument_value', 0.0)

            # MTM loss threshold (configurable, default 3% of initial position value)
            mtm_loss_threshold_pct = getattr(self.config, 'mtm_loss_exit_threshold_pct', 0.03)
            
            # Check each position for large unrealized losses
            positions_to_check = [
                ('wind', current_wind),
                ('solar', current_solar),
                ('hydro', current_hydro)
            ]
            
            mtm_exits_forced = []
            for asset, current_pos in positions_to_check:
                if abs(current_pos) > 100:  # Only check meaningful positions
                    # Model B: exposures are NOT "position values"; unrealized PnL lives in financial_mtm_positions.
                    # Use cumulative MTM relative to current exposure as a loss proxy.
                    mtm_bucket = getattr(self, 'financial_mtm_positions', None) or {}
                    mtm_value = float(mtm_bucket.get(f'{asset}_instrument_value', 0.0))
                    denom = max(abs(float(current_pos)), 1.0)
                    mtm_loss_pct = (-mtm_value / denom) if mtm_value < 0 else 0.0

                    if mtm_loss_pct > mtm_loss_threshold_pct:
                        # Force exit: set target to zero
                        mtm_exits_forced.append(asset)
                        if asset == 'wind':
                            target_wind = 0.0
                        elif asset == 'solar':
                            target_solar = 0.0
                        elif asset == 'hydro':
                            target_hydro = 0.0

                        # Log the forced exit
                        if t % 100 == 0 or len(mtm_exits_forced) > 0:
                            logger.info(
                                f"[MTM_LOSS_EXIT] Forced exit {asset}: "
                                f"loss_pct={mtm_loss_pct:.2%} > thr={mtm_loss_threshold_pct:.2%}, "
                                f"exposure={current_pos:,.0f} DKK, mtm_value={mtm_value:,.0f} DKK"
                            )

            self._last_mtm_exit_count = int(len(mtm_exits_forced))
            
            # Update current positions after MTM exits
            current_wind = self.financial_positions.get('wind_instrument_value', 0.0)
            current_solar = self.financial_positions.get('solar_instrument_value', 0.0)
            current_hydro = self.financial_positions.get('hydro_instrument_value', 0.0)

            # === STEP 3.5: Calculate trades needed to reach target positions ===
            trade_wind = target_wind - current_wind
            trade_solar = target_solar - current_solar
            trade_hydro = target_hydro - current_hydro

            # === STEP 3.6: SETTLEMENT (Model B) ===
            # When exposure is reduced/closed or flipped, we settle the corresponding share of MTM into cash
            # and reduce the MTM bucket. This prevents MTM "sticking around" after exposure is closed.
            mtm_bucket = getattr(self, 'financial_mtm_positions', None)
            if mtm_bucket is None or not isinstance(mtm_bucket, dict):
                self.financial_mtm_positions = {
                    'wind_instrument_value': 0.0,
                    'solar_instrument_value': 0.0,
                    'hydro_instrument_value': 0.0,
                }
                mtm_bucket = self.financial_mtm_positions

            def _settle(asset: str, old_exp: float, new_exp: float) -> None:
                key = f'{asset}_instrument_value'
                old_exp = float(old_exp)
                new_exp = float(new_exp)
                old_mtm = float(mtm_bucket.get(key, 0.0))
                if abs(old_exp) <= 100:
                    return

                closed_fraction = 0.0
                # Full close on sign flip
                if old_exp * new_exp < 0:
                    closed_fraction = 1.0
                # Partial close on reduction in absolute exposure (same sign)
                elif abs(new_exp) < abs(old_exp) and old_exp * new_exp >= 0:
                    closed_fraction = (abs(old_exp) - abs(new_exp)) / max(abs(old_exp), 1e-9)

                if closed_fraction <= 0.0:
                    return

                realized = old_mtm * float(np.clip(closed_fraction, 0.0, 1.0))
                # Realize into cash; remaining MTM stays attached to remaining exposure.
                self.budget += realized
                mtm_bucket[key] = float(old_mtm - realized)

            _settle('wind', current_wind, target_wind)
            _settle('solar', current_solar, target_solar)
            _settle('hydro', current_hydro, target_hydro)

            # === STEP 4: Calculate and deduct transaction costs ===
            total_traded_notional = abs(trade_wind) + abs(trade_solar) + abs(trade_hydro)

            # Minimum notional to bother executing a trade (fraction of max position, in DKK)
            _no_trade_frac = float(getattr(self.config, "no_trade_threshold", 0.0))
            threshold_dkk = float(_no_trade_frac * max(float(max_pos_size), 1.0))
            if total_traded_notional > threshold_dkk:
                transaction_cost_bps = self.config.transaction_cost_bps
                fixed_cost = self.config.transaction_fixed_cost
                transaction_costs = (total_traded_notional * transaction_cost_bps / 10000.0) + fixed_cost

                self.budget -= transaction_costs
                self._last_investor_transaction_cost = float(transaction_costs)
                
                # Log trade execution proof (periodically to show agents are making decisions)
                if t % 5000 == 0 and t > 0:
                    logger.info(f"[EXECUTION_PROOF] Step {t}: Agent executed trades - "
                               f"Wind: {trade_wind:,.0f} DKK, Solar: {trade_solar:,.0f} DKK, "
                               f"Hydro: {trade_hydro:,.0f} DKK | Total notional: {total_traded_notional:,.0f} DKK | "
                               f"Transaction cost: {transaction_costs:,.0f} DKK | "
                               f"New positions - Wind: {target_wind:,.0f}, Solar: {target_solar:,.0f}, "
                               f"Hydro: {target_hydro:,.0f} DKK")

                # Initialize cumulative transaction costs if not present
                if not hasattr(self, 'cumulative_transaction_costs'):
                    self.cumulative_transaction_costs = 0.0
                self.cumulative_transaction_costs += transaction_costs

                # === STEP 5: Execute trades by updating financial positions ===
                self.financial_positions['wind_instrument_value'] = target_wind
                self.financial_positions['solar_instrument_value'] = target_solar
                self.financial_positions['hydro_instrument_value'] = target_hydro

                self.budget = max(0.0, self.budget)

                return total_traded_notional
            else:
                self._last_investor_transaction_cost = 0.0
                return 0.0  # Ignore very small trades and incur no costs

        except Exception as e:
            msg = f"[TRADE_EXEC_FATAL] _execute_investor_trades failed at step {self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _norm_price(self, p: float, t: int) -> float:
        """
        Normalize price for active Tier1 features using z-score with clipped bounds.
        Aligns with wrapper normalization so the live Tier1 path sees consistent inputs.
        """
        try:
            # Prefer rolling mean/std if available
            if hasattr(self, '_price_mean') and hasattr(self, '_price_std'):
                if t < len(self._price_mean) and t < len(self._price_std):
                    mean = float(self._price_mean[t])
                    std = float(self._price_std[t])
                else:
                    mean = getattr(self.config, 'price_fallback_mean', 250.0)
                    std = getattr(self.config, 'price_fallback_std', 50.0)
            else:
                mean = getattr(self.config, 'price_fallback_mean', 250.0)
                std = getattr(self.config, 'price_fallback_std', 50.0)

            # Z-score normalization
            z = (p - mean) / (std if std > 1e-6 else 1.0)
            # Clip to Â±3Ïƒ and scale to [-1, 1]
            z_clipped = np.clip(z, -3.0, 3.0) / 3.0
            return float(z_clipped)
        except Exception as e:
            raise RuntimeError(
                f"[PRICE_NORM_FATAL] _norm_price failed at step={t} with price={p}: {e}"
            ) from e

    def _execute_battery_ops(self, bat_action: np.ndarray, i: int) -> float:
        """
        REFACTORED: Execute battery operations using TradingEngine.
        Returns battery cash delta for proper reward accounting.
        """
        try:
            from trading_engine import TradingEngine
            
            # Battery dispatch is policy-driven; no forecast input is available
            # in the cache-only forecast-utilization runtime.
            forecast_price = None
            use_heuristic_dispatch = bool(getattr(self.config, 'battery_use_heuristic_dispatch', False))
            battery_action_mode = str(getattr(self.config, 'battery_action_mode', 'target_soc') or 'target_soc')
            battery_action_threshold = float(np.clip(getattr(self.config, 'battery_action_threshold', 0.35), 0.0, 1.0))

            # --- Battery debug snapshot (so logs reflect real behavior) ---
            try:
                price_now = float(np.clip(self._price_raw[i], self.config.minimum_price_filter, self.config.maximum_price_cap))
                vol = float(getattr(self, '_last_price_volatility_forecast', 0.0))

                heuristic_decision = 'idle'
                heuristic_inten = 0.0
                if use_heuristic_dispatch:
                    heuristic_decision, heuristic_inten = TradingEngine.calculate_battery_dispatch_policy(
                        current_price=price_now,
                        forecast_price=forecast_price,
                        battery_hurdle_min_dkk=getattr(self.config, 'battery_hurdle_min_dkk', 5.0),
                        battery_price_sensitivity=getattr(self.config, 'battery_price_sensitivity', 0.02),
                        battery_rt_loss_weight=getattr(self.config, 'battery_rt_loss_weight', 0.3),
                        batt_eta_charge=self.batt_eta_charge,
                        batt_eta_discharge=self.batt_eta_discharge,
                        minimum_price_filter=self.config.minimum_price_filter,
                        maximum_price_cap=self.config.maximum_price_cap,
                        price_volatility_forecast=vol,
                    )

                decoded = TradingEngine.decode_battery_action(
                    bat_action=bat_action,
                    battery_capacity_mwh=self.physical_assets['battery_capacity_mwh'],
                    battery_energy=self.operational_state['battery_energy'],
                    batt_soc_min=self.batt_soc_min,
                    batt_soc_max=self.batt_soc_max,
                    max_energy_this_step=(self.physical_assets['battery_capacity_mwh'] * self.batt_power_c_rate * (10.0 / 60.0)),
                    action_mode=battery_action_mode,
                    action_threshold=battery_action_threshold,
                    discrete_action_levels=getattr(self.config, 'battery_discrete_action_levels', None),
                )
                agent_decision = str(decoded['decision'])
                agent_intensity = float(decoded['intensity'])
                u_raw = float(decoded.get('u_raw', 0.0))
                # Store action index for reward shaping (used in _assign_rewards)
                self._batt_action_idx = int(decoded.get('action_idx', 2))
                self._batt_action_u = float(decoded.get('u_raw', 0.0))  # signed continuous level

                if not use_heuristic_dispatch:
                    decision = agent_decision
                    inten = agent_intensity
                else:
                    if agent_decision == heuristic_decision:
                        decision = agent_decision
                        inten = 0.9 * agent_intensity + 0.1 * float(heuristic_inten)
                    elif agent_decision == 'idle':
                        decision = heuristic_decision
                        inten = 0.1 * float(heuristic_inten)
                    elif heuristic_decision == 'idle':
                        decision = agent_decision
                        inten = 0.9 * agent_intensity
                    else:
                        decision = agent_decision
                        inten = 0.8 * agent_intensity
                inten = float(np.clip(inten, 0.0, 1.0))

                # Compute spread/hurdles for logging (if forecast exists)
                spread = 0.0
                adjusted_hurdle = 0.0
                volatility_adjustment = 1.0
                if forecast_price is not None and np.isfinite(forecast_price):
                    p_fut = float(np.clip(forecast_price, self.config.minimum_price_filter, self.config.maximum_price_cap))
                    spread = float(p_fut - price_now)
                    needed = max(
                        float(getattr(self.config, 'battery_hurdle_min_dkk', 5.0)),
                        float(getattr(self.config, 'battery_price_sensitivity', 0.02)) * abs(price_now)
                    )
                    rt_loss = (1.0 / (max(self.batt_eta_charge * self.batt_eta_discharge, 1e-6)) - 1.0) * 10.0
                    hurdle = needed + float(getattr(self.config, 'battery_rt_loss_weight', 0.3)) * rt_loss
                    volatility_adjustment = 1.0 - (0.3 * float(np.clip(vol, 0.0, 1.0)))
                    adjusted_hurdle = float(hurdle * volatility_adjustment)

                self._last_battery_decision = str(decision)
                self._last_battery_intensity = float(inten)
                self._last_battery_spread = float(spread)
                self._last_battery_adjusted_hurdle = float(adjusted_hurdle)
                self._last_battery_volatility_adj = float(volatility_adjustment)
            except Exception:
                # Keep safe defaults (logged as 0/idle)
                self._last_battery_decision = 'idle'
                self._last_battery_intensity = 0.0
                self._last_battery_spread = 0.0
                self._last_battery_adjusted_hurdle = 0.0
                self._last_battery_volatility_adj = 0.0
            
            # Create dispatch policy function only when heuristic dispatch is enabled.
            dispatch_policy_fn = None
            if use_heuristic_dispatch:
                def dispatch_policy_fn(timestep: int):
                    return TradingEngine.calculate_battery_dispatch_policy(
                        current_price=float(np.clip(self._price_raw[timestep], self.config.minimum_price_filter, self.config.maximum_price_cap)),
                        forecast_price=forecast_price,
                        battery_hurdle_min_dkk=getattr(self.config, 'battery_hurdle_min_dkk', 5.0),
                        battery_price_sensitivity=getattr(self.config, 'battery_price_sensitivity', 0.02),
                        battery_rt_loss_weight=getattr(self.config, 'battery_rt_loss_weight', 0.3),
                        batt_eta_charge=self.batt_eta_charge,
                        batt_eta_discharge=self.batt_eta_discharge,
                        minimum_price_filter=self.config.minimum_price_filter,
                        maximum_price_cap=self.config.maximum_price_cap,
                        price_volatility_forecast=float(getattr(self, '_last_price_volatility_forecast', 0.0))
                    )
            
            # Execute battery operations
            price = float(np.clip(self._price_raw[i], self.config.minimum_price_filter, self.config.maximum_price_cap))
            cash_delta, updated_state = TradingEngine.execute_battery_operations(
                bat_action=bat_action,
                timestep=i,
                battery_capacity_mwh=self.physical_assets['battery_capacity_mwh'],
                battery_energy=self.operational_state['battery_energy'],
                price=price,
                batt_power_c_rate=self.batt_power_c_rate,
                batt_eta_charge=self.batt_eta_charge,
                batt_eta_discharge=self.batt_eta_discharge,
                batt_soc_min=self.batt_soc_min,
                batt_soc_max=self.batt_soc_max,
                batt_degradation_cost=self.batt_degradation_cost,
                battery_dispatch_policy_fn=dispatch_policy_fn,
                config=self.config,
                use_heuristic_dispatch=use_heuristic_dispatch,
                action_threshold=battery_action_threshold,
            )
            
            # Capture old battery energy before updating (for throughput calculation)
            old_battery_energy = float(self.operational_state.get('battery_energy', 0.0))
            
            # Update operational state
            self.operational_state['battery_energy'] = updated_state['battery_energy']
            self.operational_state['battery_discharge_power'] = updated_state['battery_discharge_power']

            # Persist realized battery cash delta for logging/NAV attribution
            self._last_battery_cash_delta = float(cash_delta)
            
            # NEW: Store additional battery metrics for logging
            self._last_battery_energy = float(updated_state['battery_energy'])
            self._last_battery_capacity = float(self.physical_assets['battery_capacity_mwh'])
            battery_soc = float(updated_state['battery_energy'] / max(self.physical_assets['battery_capacity_mwh'], 1.0))
            self._last_battery_soc = float(np.clip(battery_soc, 0.0, 1.0))
            self._last_battery_target_soc = float(np.clip(updated_state.get('battery_target_soc', battery_soc), 0.0, 1.0))
            
            # Calculate throughput and degradation cost from state changes
            battery_energy_change = abs(updated_state['battery_energy'] - old_battery_energy)
            # Throughput is the energy moved (charge or discharge)
            if getattr(self, '_last_battery_decision', 'idle') in ['charge', 'discharge']:
                self._last_battery_throughput = float(battery_energy_change)
                self._last_battery_degradation_cost = float(self.batt_degradation_cost * battery_energy_change)
            else:
                self._last_battery_throughput = 0.0
                self._last_battery_degradation_cost = 0.0
            
            # Store efficiency values
            self._last_battery_eta_charge = float(self.batt_eta_charge)
            self._last_battery_eta_discharge = float(self.batt_eta_discharge)
            
            return float(cash_delta)
        except Exception as e:
            msg = f"[BATTERY_EXEC_FATAL] _execute_battery_ops failed at step {self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    # ------------------------------------------------------------------
    # FIXED: Finance & rewards with proper separation
    # ------------------------------------------------------------------
    def _compute_mtm_for_step(self, i: int, current_price: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute MTM for price move (i-1â†’i) using current financial_positions (PRE-trade).
        Accumulates to financial_mtm_positions. Call BEFORE _execute_investor_trades.
        """
        from financial_engine import FinancialEngine
        price_returns = FinancialEngine.calculate_price_returns(
            timestep=i, current_price=current_price, price_history=self._price_raw
        )
        price_return = price_returns['price_return']
        self._correlation_debug = price_returns['correlation_debug']
        self._latest_price_returns = {
            'one_step': price_return,
        }
        if i < len(self._price_return_1step):
            self._price_return_1step[i] = price_return
        if not hasattr(self, 'financial_mtm_positions') or not isinstance(getattr(self, 'financial_mtm_positions', None), dict):
            self.financial_mtm_positions = {
                'wind_instrument_value': 0.0,
                'solar_instrument_value': 0.0,
                'hydro_instrument_value': 0.0,
            }
        # Pre-trade gross notional is still tracked for diagnostics, but the
        # investor reward now normalizes by allocated sleeve capacity rather than
        # by current exposure. That makes the local reward reflect actual trading
        # profit on capital instead of collapsing to raw market direction.
        try:
            pretrade_exposure = 0.0
            for v in (getattr(self, "financial_positions", {}) or {}).values():
                fv = float(v)
                if np.isfinite(fv):
                    pretrade_exposure += abs(fv)
            self._last_investor_exposure_pretrade = float(pretrade_exposure)
        except Exception:
            self._last_investor_exposure_pretrade = 0.0
        mtm_pnl, per_asset_mtm = FinancialEngine.calculate_mtm_pnl_from_exposure(
            financial_exposures=self.financial_positions,
            price_return=price_return,
            config=self.config
        )
        for k in ('wind_instrument_value', 'solar_instrument_value', 'hydro_instrument_value'):
            self.financial_mtm_positions[k] = float(self.financial_mtm_positions.get(k, 0.0)) + float(per_asset_mtm.get(k, 0.0))
        return float(mtm_pnl), per_asset_mtm

    def _add_mtm_for_step(self, i: int) -> None:
        """
        Add MTM for the price move (i-1â†’i) using PRE-trade positions.
        CRITICAL: Must run BEFORE _execute_investor_trades so MTM is attributed to
        the positions we actually held during the price move.
        """
        current_price = float(np.clip(self._price_raw[i], -1000.0, 1e9))
        mtm_pnl, per_asset_mtm = self._compute_mtm_for_step(i, current_price)
        self._last_mtm_pnl_added = mtm_pnl
        self._last_per_asset_mtm_added = per_asset_mtm

    def _update_finance(self, i: int, trade_amount: float, battery_cash_delta: float) -> Dict[str, float]:
        """
        REFACTORED: Update all financial components with proper separation.
        MTM is already added in _add_mtm_for_step (before trades) to ensure correct order.
        """
        try:
            # CRITICAL FIX: Use raw prices for generation revenue
            current_price = float(np.clip(self._price_raw[i], -1000.0, 1e9))

            # 1) Generation revenue from physical assets (CASH FLOW)
            generation_revenue = self._calculate_generation_revenue(i, current_price)

            # 2) MTM already added in _add_mtm_for_step (before trades); use stored value
            mtm_pnl = float(getattr(self, '_last_mtm_pnl_added', 0.0))
            per_asset_mtm = getattr(
                self, '_last_per_asset_mtm_added',
                {'wind_instrument_value': 0.0, 'solar_instrument_value': 0.0, 'hydro_instrument_value': 0.0}
            )

            # 3) Transaction costs (CASH FLOW) - FIXED: Already deducted in _execute_investor_trades
            txn_costs = 0.0  # Costs already handled in trading execution

            # 4) Battery operational costs (CASH FLOW) - from config
            battery_opex = self.config.battery_opex_rate * self.physical_assets['battery_capacity_mwh']

            net_cash_flow = generation_revenue + battery_cash_delta - txn_costs - battery_opex

            # FIXED: Separate operational revenue from trading cash
            # Only battery arbitrage affects trading cash (since it uses trading capital)
            # Generation revenue increases fund value but not trading cash allocation
            operational_revenue = generation_revenue - battery_opex  # Pure operational revenue
            trading_cash_flow = battery_cash_delta - txn_costs      # Only trading-related cash flows

            # 7) Update cash position (only trading-related cash flows)
            self.budget = max(0.0, self.budget + trading_cash_flow)

            # 7.5) INFRASTRUCTURE FUND: Distribute excess cash to maintain realistic cash levels
            self._distribute_excess_cash()

            # 8) Track operational revenue separately (for NAV calculation)
            if not hasattr(self, 'accumulated_operational_revenue'):
                self.accumulated_operational_revenue = 0.0
            self.accumulated_operational_revenue += operational_revenue

            # 9) ENHANCED: Check for emergency reallocation from operational gains to trading capital
            if self.config.allow_operational_reallocation:
                self._check_emergency_reallocation(operational_revenue)

            # 8) Calculate fund NAV
            # CRITICAL FIX: Pass i (the timestep parameter) directly to _calculate_fund_nav
            # This ensures consistent NAV calculation even if self.t changes
            # No need to modify self.t - just pass the correct timestep as parameter
            # CRITICAL DEBUG: Log when calculating NAV at timestep 0 and 1
            if i <= 1 and logger.isEnabledFor(10):
                accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
                financial_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
                logger.debug(f"[UPDATE_FINANCE] Calculating NAV in _update_finance at i={i}:")
                logger.debug(f"  i parameter = {i}")
                logger.debug(f"  self.t = {getattr(self, 't', 0)}")
                logger.debug(f"  budget = {self.budget:,.0f} DKK")
                logger.debug(f"  physical_assets = {self.physical_assets}")
                logger.debug(f"  accumulated_operational_revenue = {accumulated_operational_revenue:,.0f} DKK")
                logger.debug(f"  financial_positions MTM = {financial_mtm:,.0f} DKK")
            fund_nav = self._calculate_fund_nav(current_timestep=i)
            if i <= 1 and logger.isEnabledFor(10):
                # Calculate components to see what's different
                accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
                financial_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
                # Calculate physical assets from NAV
                physical_from_nav = fund_nav - self.budget - accumulated_operational_revenue - financial_mtm
                logger.debug(f"[UPDATE_FINANCE] NAV calculated at i={i}: {fund_nav:,.0f} DKK (${fund_nav * 0.145 / 1_000_000:.2f}M)")
                logger.debug(f"  NAV breakdown: budget={self.budget:,.0f}, physical={physical_from_nav:,.0f}, operational={accumulated_operational_revenue:,.0f}, mtm={financial_mtm:,.0f}")
                logger.debug(f"  current_timestep passed to _calculate_fund_nav = {i}")

            # 9) Track performance
            self.performance_history['revenue_history'].append(net_cash_flow)
            self.performance_history['generation_revenue_history'].append(generation_revenue)
            self.performance_history['nav_history'].append(fund_nav)

            # Realized investor-sleeve dNAV/return for forecast-backend diagnostics.
            # Use pure MTM on pre-trade exposure for this step's price move.
            # Normalize by allocated sleeve capacity, not by current exposure.
            # This keeps the positive signal aligned with actual trading profits
            # rather than collapsing to the underlying market return.
            realized_investor_dnav = float(mtm_pnl)
            investor_return_denom = float(
                max(
                    float(getattr(self, "_last_tradeable_capital", 0.0))
                    * float(max(getattr(self.config, "max_position_size", 0.0), 0.0)),
                    0.0,
                )
            )
            if (not np.isfinite(investor_return_denom)) or investor_return_denom <= 0.0:
                investor_return_denom = float(max(getattr(self, "_last_investor_exposure_pretrade", 0.0), 1.0))
            realized_investor_return = float(realized_investor_dnav / max(investor_return_denom, 1.0))
            if not np.isfinite(realized_investor_return):
                realized_investor_return = 0.0
            self.last_realized_investor_dnav = float(realized_investor_dnav)
            self.last_realized_investor_dnav_return = float(realized_investor_return)
            self.last_realized_investor_return_denom = float(max(investor_return_denom, 1.0))

            # Investor-local trading sleeve tracker used for the investor reward.
            # This deliberately excludes operational revenue so the investor policy
            # is trained on trading performance rather than the fixed operating book.
            try:
                tracker_clip = float(max(getattr(self.config, 'investor_tracker_return_clip', 0.50), 1e-6))
                tracker_return = float(np.clip(realized_investor_return, -tracker_clip, tracker_clip))
                if not np.isfinite(tracker_return):
                    tracker_return = 0.0

                if not hasattr(self, '_investor_local_return_history'):
                    self._investor_local_return_history = deque(
                        maxlen=max(1, int(getattr(self.config, 'investor_trading_history_lookback', 48)))
                    )
                    self._investor_local_path_value = 1.0
                    self._investor_local_peak_value = 1.0
                    self._investor_local_drawdown = 0.0
                    self._investor_local_recent_mean = 0.0
                    self._investor_local_recent_vol = 0.0
                    self._investor_local_quality = 0.0
                    self._investor_local_quality_delta = 0.0

                self._investor_local_return_history.append(tracker_return)
                self._investor_local_path_value = float(
                    max(1e-6, float(getattr(self, '_investor_local_path_value', 1.0)) * (1.0 + tracker_return))
                )
                self._investor_local_peak_value = float(
                    max(float(getattr(self, '_investor_local_peak_value', 1.0)), self._investor_local_path_value)
                )
                peak_local = float(getattr(self, '_investor_local_peak_value', 1.0))
                self._investor_local_drawdown = float(
                    (peak_local - self._investor_local_path_value) / peak_local if peak_local > 0.0 else 0.0
                )

                hist = np.asarray(list(self._investor_local_return_history), dtype=np.float64)
                self._investor_local_recent_mean = float(np.mean(hist)) if hist.size > 0 else 0.0
                self._investor_local_recent_vol = float(np.std(hist)) if hist.size > 1 else 0.0
                quality_clip = float(max(getattr(self.config, 'investor_trading_quality_clip', 2.0), 1e-6))
                vol_floor = float(max(getattr(self.config, 'investor_trading_vol_floor', 5e-4), 1e-8))
                prev_quality = float(getattr(self, '_investor_local_quality', 0.0))
                self._investor_local_quality = float(
                    np.clip(self._investor_local_recent_mean / max(self._investor_local_recent_vol, vol_floor),
                            -quality_clip, quality_clip)
                )
                self._investor_local_quality_delta = float(
                    np.clip(self._investor_local_quality - prev_quality, -quality_clip, quality_clip)
                )
            except Exception:
                self._investor_local_drawdown = 0.0
                self._investor_local_recent_mean = 0.0
                self._investor_local_recent_vol = 0.0
                self._investor_local_quality = 0.0
                self._investor_local_quality_delta = 0.0

            # Realized step dNAV relative to the prior fund NAV snapshot.
            prev_nav_for_step = getattr(self, '_fund_nav_prev', None)
            if prev_nav_for_step is None:
                realized_dnav = 0.0
                realized_dnav_return = 0.0
            else:
                try:
                    prev_nav_for_step = float(prev_nav_for_step)
                except Exception:
                    prev_nav_for_step = 0.0
                if (not np.isfinite(prev_nav_for_step)) or prev_nav_for_step <= 0.0:
                    realized_dnav = 0.0
                    realized_dnav_return = 0.0
                else:
                    realized_dnav = float(fund_nav - prev_nav_for_step)
                    realized_dnav_return = float(realized_dnav / max(prev_nav_for_step, 1.0))

            self.last_realized_dnav = float(realized_dnav)
            self.last_realized_dnav_return = float(realized_dnav_return)
            self._fund_nav_prev = float(fund_nav)

            # 10) Store values for logging/rewards
            self.last_revenue = net_cash_flow
            self.last_generation_revenue = generation_revenue
            self.last_mtm_pnl = mtm_pnl


            # CRITICAL: Update reward calculator with per-step trading MTM for DL training
            if hasattr(self, 'reward_calculator') and self.reward_calculator is not None:
                self.reward_calculator.update_trading_mtm(mtm_pnl)

            # 11) Update cumulative tracking
            self.cumulative_generation_revenue += generation_revenue
            self.cumulative_battery_revenue += battery_cash_delta
            self.cumulative_mtm_pnl += mtm_pnl  # Diagnostic cumulative MTM tracker

            # CRITICAL FIX: Update cumulative returns (total fund return)
            current_nav = fund_nav
            if hasattr(self, '_previous_nav') and self._previous_nav > 0:
                nav_return = (current_nav - self._previous_nav) / self._previous_nav
                self.cumulative_returns += nav_return
            self._previous_nav = current_nav

            # ENHANCED: Update hedge effectiveness tracking
            if hasattr(self, 'hedge_effectiveness_tracker'):
                tracker = self.hedge_effectiveness_tracker
                tracker['ops_returns'].append(generation_revenue)
                tracker['trading_returns'].append(mtm_pnl)

                # Calculate rolling correlation if we have enough data
                if len(tracker['ops_returns']) >= 20:  # Need at least 20 points
                    ops_array = np.array(tracker['ops_returns'])
                    trading_array = np.array(tracker['trading_returns'])

                    # Calculate correlation (should be negative for effective hedging)
                    if np.std(ops_array) > 1e-6 and np.std(trading_array) > 1e-6:
                        correlation = np.corrcoef(ops_array, trading_array)[0, 1]
                        tracker['correlation'] = correlation if np.isfinite(correlation) else 0.0

                        # Update effectiveness multiplier based on correlation
                        if tracker['correlation'] > -0.1:  # Poor hedging effectiveness
                            tracker['effectiveness_multiplier'] = max(0.5, tracker['effectiveness_multiplier'] * 0.95)
                        elif tracker['correlation'] < -0.3:  # Good hedging effectiveness
                            tracker['effectiveness_multiplier'] = min(1.0, tracker['effectiveness_multiplier'] * 1.02)
                        # Else: maintain current multiplier

            return {
                'revenue': net_cash_flow,
                'generation_revenue': generation_revenue,
                'battery_cash_delta': battery_cash_delta,
                'mtm_pnl': mtm_pnl,
                'transaction_costs': txn_costs,
                'fund_nav': fund_nav,
                'total_generation_mwh': self._get_total_generation_mwh(i),
                'portfolio_value': fund_nav,
                'equity': fund_nav,
                'efficiency': self._calculate_generation_efficiency(i),
                'trading_mtm_tracker': getattr(self, 'cumulative_mtm_pnl', 0.0),
                'operating_revenue_tracker': getattr(self, 'cumulative_generation_revenue', 0.0),
            }

        except Exception as e:
            msg = f"[FINANCE_UPDATE_FATAL] Finance update error at t={self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _calculate_generation_revenue(self, i: int, price: float) -> float:
        """
        REFACTORED: Calculate revenue from PHYSICAL ASSETS using FinancialEngine.
        Revenue = (Physical Generation * Market Price) - Operating Costs

        IMPORTANT: Only apply operational costs AFTER assets are deployed!
        During asset acquisition phase, no operational costs should apply.
        """
        try:
            from financial_engine import FinancialEngine
            return FinancialEngine.calculate_generation_revenue(
                timestep=i,
                price=price,
                wind_data=self._wind,
                solar_data=self._solar,
                hydro_data=self._hydro,
                wind_scale=self.wind_scale,
                solar_scale=self.solar_scale,
                hydro_scale=self.hydro_scale,
                physical_assets=self.physical_assets,
                asset_capex=self.asset_capex,
                config=self.config,
                electricity_markup=self.electricity_markup,
                currency_conversion=self.currency_conversion
            )
        except Exception as e:
            logger.error(f"[GEN_REVENUE_FATAL] Generation revenue calculation failed: {e}")
            raise RuntimeError("[GEN_REVENUE_FATAL] Generation revenue calculation failed") from e

    def _distribute_excess_cash(self):
        """
        REFACTORED: Distribute excess cash using FinancialEngine.
        Infrastructure funds typically maintain 5-15% cash and distribute excess to investors.
        """
        try:
            from financial_engine import FinancialEngine
            
            current_fund_value = self._calculate_fund_nav()
            new_budget, distribution_amount = FinancialEngine.distribute_excess_cash(
                budget=self.budget,
                current_fund_nav=current_fund_value,
                init_budget=self.init_budget,
                config=self.config
            )
            
            # Update budget
            self.budget = new_budget
            
            # Track distributions for reporting
            if distribution_amount > 0:
                if not hasattr(self, 'total_distributions'):
                    self.total_distributions = 0.0
                self.total_distributions += distribution_amount

        except Exception as e:
            logger.error(f"[CASH_DISTRIBUTION_FATAL] Cash distribution failed: {e}")
            raise RuntimeError("[CASH_DISTRIBUTION_FATAL] Cash distribution failed") from e

    def _check_emergency_reallocation(self, operational_revenue: float):
        """
        REFACTORED: Emergency reallocation from operational gains to trading capital.
        Uses FinancialEngine for calculation.
        """
        try:
            from financial_engine import FinancialEngine
            
            # Initialize tracking if needed
            if not hasattr(self, 'total_reallocated'):
                self.total_reallocated = 0.0
            if not hasattr(self, 'accumulated_operational_revenue'):
                self.accumulated_operational_revenue = 0.0
            
            new_budget, new_accumulated_operational_revenue, new_total_reallocated = FinancialEngine.check_emergency_reallocation(
                budget=self.budget,
                accumulated_operational_revenue=self.accumulated_operational_revenue,
                operational_revenue=operational_revenue,
                trading_allocation_budget=self.trading_allocation_budget,
                total_fund_value=self.total_fund_value,
                total_reallocated=self.total_reallocated,
                config=self.config
            )
            
            # Update state
            self.budget = new_budget
            self.accumulated_operational_revenue = new_accumulated_operational_revenue
            self.total_reallocated = new_total_reallocated

        except Exception as e:
            logger.error(f"[EMERGENCY_REALLOCATION_FATAL] Emergency reallocation check failed: {e}")
            raise RuntimeError("[EMERGENCY_REALLOCATION_FATAL] Emergency reallocation check failed") from e

    def _get_wind_capacity_factor(self, i: int) -> float:
        try:
            raw_wind = float(self._wind[i]) if i < len(self._wind) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic wind CF range (15-45%) - industry standard
            normalized = raw_wind / max(self.wind_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.15 + (normalized * 0.30)  # Map to 15-45% range
            return float(np.clip(realistic_cf, 0.0, 0.45))
        except Exception as e:
            raise RuntimeError(
                f"[WIND_CF_FATAL] _get_wind_capacity_factor failed at step={i}: {e}"
            ) from e

    def _get_solar_capacity_factor(self, i: int) -> float:
        try:
            raw_solar = float(self._solar[i]) if i < len(self._solar) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic solar CF range (5-30%) - industry standard
            normalized = raw_solar / max(self.solar_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.05 + (normalized * 0.25)  # Map to 5-30% range
            return float(np.clip(realistic_cf, 0.0, 0.30))
        except Exception as e:
            raise RuntimeError(
                f"[SOLAR_CF_FATAL] _get_solar_capacity_factor failed at step={i}: {e}"
            ) from e

    def _get_hydro_capacity_factor(self, i: int) -> float:
        try:
            raw_hydro = float(self._hydro[i]) if i < len(self._hydro) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic hydro CF range (35-65%) - industry standard
            normalized = raw_hydro / max(self.hydro_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.35 + (normalized * 0.30)  # Map to 35-65% range
            return float(np.clip(realistic_cf, 0.35, 0.65))
        except Exception as e:
            raise RuntimeError(
                f"[HYDRO_CF_FATAL] _get_hydro_capacity_factor failed at step={i}: {e}"
            ) from e

    def _get_total_generation_mwh(self, i: int) -> float:
        """Get total electricity generation this timestep"""
        try:
            time_step_hours = 10.0 / 60.0
            wind_mwh = self.physical_assets['wind_capacity_mw'] * self._get_wind_capacity_factor(i) * time_step_hours
            solar_mwh = self.physical_assets['solar_capacity_mw'] * self._get_solar_capacity_factor(i) * time_step_hours
            hydro_mwh = self.physical_assets['hydro_capacity_mw'] * self._get_hydro_capacity_factor(i) * time_step_hours
            return float(wind_mwh + solar_mwh + hydro_mwh)
        except Exception as e:
            raise RuntimeError(
                f"[GEN_MWH_FATAL] _get_total_generation_mwh failed at step={i}: {e}"
            ) from e

    def _calculate_generation_efficiency(self, i: int) -> float:
        try:
            wind_cf = self._get_wind_capacity_factor(i)
            solar_cf = self._get_solar_capacity_factor(i)
            hydro_cf = self._get_hydro_capacity_factor(i)

            total_financial = sum(abs(v) for v in self.financial_positions.values())
            if total_financial <= 0:
                return 0.0

            wind_weight = abs(self.financial_positions['wind_instrument_value']) / total_financial
            solar_weight = abs(self.financial_positions['solar_instrument_value']) / total_financial
            hydro_weight = abs(self.financial_positions['hydro_instrument_value']) / total_financial

            portfolio_cf = wind_weight * wind_cf + solar_weight * solar_cf + hydro_weight * hydro_cf
            diversification_bonus = 1.0 - (wind_weight**2 + solar_weight**2 + hydro_weight**2)
            efficiency = portfolio_cf * (1.0 + 0.2 * diversification_bonus)
            return float(np.clip(efficiency, 0.0, 1.0))
        except Exception as e:
            raise RuntimeError(
                f"[GEN_EFF_FATAL] _calculate_generation_efficiency failed at step={i}: {e}"
            ) from e

    def _assign_rewards(self, financial: Dict[str, float]):
        """FIXED: Reward assignment with proper separation of value sources"""
        try:
            # CRITICAL FIX: Use fund_nav from financial dict (already calculated with correct timestep in _update_finance)
            # This ensures consistent NAV values across runs
            # _update_finance calculates NAV with self.t = i, so it's already correct
            fund_nav = float(financial.get('fund_nav', 0.0))
            if fund_nav == 0.0:
                # Fallback: recalculate if not present (should not happen, but be safe)
                # CRITICAL: Use current self.t (which should be the timestep from step())
                fund_nav = float(self._calculate_fund_nav())

            # Get cash flow (actual money earned this step) - ensure float
            cash_flow = float(financial.get('revenue', 0.0))

            # Get risk level - ensure float (handle array case)
            risk_snapshot = self.overall_risk_snapshot
            if isinstance(risk_snapshot, np.ndarray):
                risk_snapshot = float(risk_snapshot.item() if risk_snapshot.size == 1 else risk_snapshot.flatten()[0])
            risk_level = float(np.clip(risk_snapshot, 0.0, 1.0))

            # Get efficiency - ensure float
            efficiency = float(financial.get('efficiency', 0.0))

            # Calculate reward using FIXED calculator
            if self.reward_calculator is None:
                logger.error(f"Reward calculator is None at step {self.t}. Initializing now...")
                # Emergency initialization
                post_capex_nav = self._calculate_fund_nav()
                self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
                logger.info(f"Emergency reward calculator initialized with NAV: {post_capex_nav:,.0f} DKK")
                self.reward_weights = dict(getattr(self.reward_calculator, 'reward_weights', {}))

            reward = self.reward_calculator.calculate_reward(
                fund_nav=fund_nav,
                cash_flow=cash_flow,
                risk_level=risk_level,
                efficiency=efficiency
            )
            # Keep environment-level snapshot in sync for downstream logging
            self.reward_weights = dict(getattr(self.reward_calculator, 'reward_weights', {}))

            # FIXED: Robust reward processing with safety checks
            # CRITICAL: Convert to float first to avoid array boolean ambiguity
            if isinstance(reward, np.ndarray):
                reward = float(reward.item() if reward.size == 1 else reward.flatten()[0])
            else:
                reward = float(reward)
            
            # CRITICAL: Use scalar check, not array check
            try:
                reward_float = float(reward)
                if not np.isfinite(reward_float):
                    reward = 0.0
                else:
                    reward = reward_float
            except (ValueError, TypeError):
                reward = 0.0

            # Use reward calculator's clipping (which respects config)
            # The reward calculator already clips to config.reward_clip_min/max
            base_reward = float(reward)

            agent_rewards = {}
            
            # Keep all agents anchored to the shared fund objective. The investor
            # additionally gets a meaningful local trading-sleeve objective so it
            # learns profitable trading behavior rather than a pure shared-reward hedge.
            investor_base = float(base_reward) * float(getattr(self.config, 'investor_base_reward_weight', 1.0))
            battery_base = float(base_reward) * float(getattr(self.config, 'battery_base_reward_weight', 1.0))
            risk_base = float(base_reward) * float(getattr(self.config, 'risk_base_reward_weight', 1.0))
            meta_base = float(base_reward) * float(getattr(self.config, 'meta_base_reward_weight', 1.0))
            
            # ===== INVESTOR AGENT =====
            # Direct delta for trading performance.
            # This is intentionally configurable so we can strengthen the
            # investor's incentive to improve the trading sleeve without
            # distorting the shared base reward across tiers.
            mtm_pnl = float(financial.get('mtm_pnl', 0.0))
            mtm_reward_weight = float(getattr(self.config, 'investor_mtm_reward_weight', 0.0))
            mtm_reward_scale = float(max(getattr(self.config, 'investor_mtm_reward_scale', 5000.0), 1.0))
            mtm_reward_clip = float(max(getattr(self.config, 'investor_mtm_reward_clip', 1.0), 1e-6))
            investor_mtm_delta = float(
                mtm_reward_weight * np.clip(mtm_pnl / mtm_reward_scale, -mtm_reward_clip, mtm_reward_clip)
            )

            # Optional direct profit reward. Keep this disabled by default to
            # avoid paying the investor twice for the same sleeve PnL (profit + return).
            profit_weight = float(getattr(self.config, 'investor_trading_profit_weight', 0.0))
            profit_scale = float(max(getattr(self.config, 'investor_trading_profit_scale', 3000.0), 1.0))
            profit_clip = float(max(getattr(self.config, 'investor_trading_profit_clip', 1.5), 1e-6))
            investor_trading_profit_delta = float(
                profit_weight * np.clip(mtm_pnl / profit_scale, -profit_clip, profit_clip)
            )

            # Hedging alignment (optional): reward negative correlation with operational revenue.
            # Disabled by default; investor focuses on profitable trading.
            hedging_weight = float(getattr(self.config, 'investor_hedging_reward_weight', 0.0))
            hedging_score = float(getattr(self.reward_calculator, 'last_hedging_score', 0.0)) if self.reward_calculator else 0.0
            investor_hedging_delta = float(
                hedging_weight * np.clip(hedging_score / 3.0, -1.0, 1.0)
            )

            # Investor policy health diagnostics:
            # keep the metrics for analysis/export, but do not compute reward-side
            # PPO health penalties in the environment.
            investor_mean_clip_hit = 0.0
            try:
                # Deterministic deployed-policy anti-collapse:
                # use the policy mean (not the sampled action) because evaluation
                # runs with deterministic=True.
                mu_hist = getattr(self, '_investor_mean_history', None)
                if mu_hist is None:
                    mu_hist = deque(
                        maxlen=max(1, int(getattr(self.config, 'investor_mean_collapse_window', 256)))
                    )
                    self._investor_mean_history = mu_hist
                current_mu = float(np.clip(getattr(self, '_inv_tanh_mu', 0.0), -1.0, 1.0))
                mu_hist.append(current_mu)
                mu_vals = np.asarray(mu_hist, dtype=np.float32)
                self._investor_mean_abs_rolling = float(np.abs(mu_vals.mean())) if mu_vals.size else 0.0
                active = mu_vals[np.abs(mu_vals) > 0.05]
                if active.size > 0:
                    self._investor_mean_sign_consistency = float(np.abs(np.sign(active).mean()))
                else:
                    self._investor_mean_sign_consistency = 0.0

                # Investor policy-mean saturation occupancy:
                # with vanilla PPO there is no internal mean clamp, so track
                # how often the deterministic policy mean itself sits too close
                # to the action boundary.
                clip_hist = getattr(self, '_investor_mean_clip_hit_history', None)
                if clip_hist is None:
                    clip_hist = deque(
                        maxlen=max(1, int(getattr(self.config, 'investor_mean_clip_hit_window', 256)))
                    )
                    self._investor_mean_clip_hit_history = clip_hist
                current_mu_mag = float(abs(getattr(self, '_inv_tanh_mu', 0.0)))
                clip_hit_margin = float(np.clip(getattr(self.config, 'investor_mean_clip_hit_margin', 0.98), 0.0, 1.0))
                investor_mean_clip_hit = float(
                    1.0 if current_mu_mag >= clip_hit_margin else 0.0
                )
                clip_hist.append(investor_mean_clip_hit)
                clip_vals = np.asarray(clip_hist, dtype=np.float32)
                self._investor_mean_clip_hit_rate = float(clip_vals.mean()) if clip_vals.size else 0.0
            except Exception:
                investor_mean_clip_hit = 0.0
                self._investor_mean_clip_hit_rate = 0.0
                self._investor_mean_abs_rolling = 0.0
                self._investor_mean_sign_consistency = 0.0

            # Store investor health diagnostics for episode analysis / health export.
            try:
                if self.config is not None:
                    self._last_penalty_diag = {
                        'training_global_step': int(getattr(self.config, 'training_global_step', 0)),
                        'inv_mean_clip_hit': float(investor_mean_clip_hit),
                        'inv_mean_clip_hit_rate': float(getattr(self, '_investor_mean_clip_hit_rate', 0.0)),
                        'inv_mu_abs_roll': float(getattr(self, '_investor_mean_abs_rolling', 0.0)),
                        'inv_mu_sign_consistency': float(getattr(self, '_investor_mean_sign_consistency', 0.0)),
                    }
            except Exception:
                self._last_penalty_diag = {
                    'training_global_step': 0,
                    'inv_mean_clip_hit': 0.0,
                    'inv_mean_clip_hit_rate': 0.0,
                    'inv_mu_abs_roll': 0.0,
                    'inv_mu_sign_consistency': 0.0,
                }
            
            position_alignment_status = 'disabled'
            realized_investor_return = float(getattr(self, 'last_realized_investor_dnav_return', 0.0))
            history_lookback = max(1, int(getattr(self.config, 'investor_trading_history_lookback', 48)))
            history_len = len(getattr(self, '_investor_local_return_history', []))
            history_frac = float(np.clip(history_len / float(history_lookback), 0.0, 1.0))

            return_weight = float(getattr(self.config, 'investor_trading_return_weight', 0.26))
            return_scale = float(max(getattr(self.config, 'investor_trading_return_scale', 0.003), 1e-8))
            return_clip = float(max(getattr(self.config, 'investor_trading_return_clip', 1.5), 1e-6))
            return_signal = float(np.clip(realized_investor_return / return_scale, -return_clip, return_clip))
            investor_trading_return_delta = float(
                history_frac * return_weight * return_signal
            )

            quality_weight = float(getattr(self.config, 'investor_trading_quality_weight', 0.0))
            quality_clip = float(max(getattr(self.config, 'investor_trading_quality_clip', 2.0), 1e-6))
            investor_local_quality = float(getattr(self, '_investor_local_quality', 0.0))
            investor_local_quality_delta = float(getattr(self, '_investor_local_quality_delta', 0.0))
            quality_signal = float(np.clip(investor_local_quality_delta / quality_clip, -1.0, 1.0))
            # Quality is retained for diagnostics/meta context, but the clean
            # investor reward contract does not pay a second non-PnL bonus path.
            investor_trading_quality_delta = float(
                history_frac
                * quality_weight
                * quality_signal
            )

            drawdown_weight = float(getattr(self.config, 'investor_trading_drawdown_weight', 0.06))
            drawdown_scale = float(max(getattr(self.config, 'investor_trading_drawdown_scale', 0.05), 1e-8))
            investor_local_drawdown = float(np.clip(getattr(self, '_investor_local_drawdown', 0.0), 0.0, 1.0))
            drawdown_excess = float(max(investor_local_drawdown - drawdown_scale, 0.0))
            investor_trading_drawdown_penalty = float(
                history_frac * drawdown_weight * np.clip(drawdown_excess / drawdown_scale, 0.0, quality_clip)
            )

            cost_weight = float(getattr(self.config, 'investor_trading_cost_weight', 0.05))
            cost_scale = float(max(getattr(self.config, 'investor_trading_cost_scale', 0.002), 1e-8))
            txn_cost = float(max(getattr(self, '_last_investor_transaction_cost', 0.0), 0.0))
            txn_cost_denom = float(max(getattr(self, 'last_realized_investor_return_denom', 0.0), 1.0))
            txn_cost_return = float(txn_cost / txn_cost_denom) if txn_cost_denom > 0.0 else 0.0
            investor_trading_cost_penalty = float(
                cost_weight * np.clip(txn_cost_return / cost_scale, 0.0, quality_clip)
            )
            current_exposure_norm = float(
                np.clip(
                    self._estimate_current_investor_exposure(getattr(self, '_last_tradeable_capital', None)),
                    -1.0,
                    1.0,
                )
            )
            active_risk_weight = float(getattr(self.config, 'investor_active_risk_weight', 0.0))
            active_risk_free_band = float(np.clip(getattr(self.config, 'investor_active_risk_free_band', 0.40), 0.0, 1.0))
            active_risk_vol_mult = float(max(getattr(self.config, 'investor_active_risk_vol_mult', 1.0), 0.0))
            realized_volatility_regime = float(np.clip(self._compute_investor_realized_volatility(min(self.t, self.max_steps - 1)), 0.0, 1.0))
            exposure_excess = float(max(abs(current_exposure_norm) - active_risk_free_band, 0.0))
            investor_active_risk_penalty = float(
                active_risk_weight
                * (exposure_excess ** 2)
                * (1.0 + active_risk_vol_mult * realized_volatility_regime)
            )

            clean_reward_contract = bool(getattr(self.config, 'investor_clean_reward_contract', True))
            if clean_reward_contract:
                # Clean economic contract:
                # realized sleeve return
                # minus explicit cost / drawdown / inventory risk
                # PPO policy health is handled in the optimizer, not the reward.
                investor_local_edge_delta = float(investor_trading_return_delta)
                investor_optional_edge_delta = 0.0
            else:
                investor_local_edge_delta = float(investor_trading_return_delta)
                investor_optional_edge_delta = float(
                    investor_trading_profit_delta + investor_trading_quality_delta
                )
            investor_risk_penalty = float(
                investor_trading_cost_penalty
                + investor_trading_drawdown_penalty
                + investor_active_risk_penalty
            )

            agent_rewards['investor_0'] = (
                investor_base
                + investor_mtm_delta
                + investor_hedging_delta
                + investor_local_edge_delta
                + investor_optional_edge_delta
                - investor_risk_penalty
            )
            
            # ===== BATTERY OPERATOR AGENT =====
            # Reward design principle: the DQN must learn TIMING, not immediate cash.
            # battery_arbitrage_delta (cash-based) is deliberately excluded because charging
            # always has a negative immediate cash delta, which causes the DQN to permanently
            # avoid charging regardless of price conditions (greedy depletion).
            #
            # Instead we reward action QUALITY = alignment between chosen direction and price signal.
            # This provides an immediate, unbiased per-step signal about decision quality,
            # solving the temporal-credit-assignment problem for intertemporal arbitrage.

            # --- Action intent (from stored action index, NOT from realized cash) ---
            # _batt_action_idx is stored in _execute_battery_ops from decode_battery_action.
            # discrete levels: [-1.0, -0.5, 0.0, 0.5, 1.0]  (idx 0=full charge, 4=full discharge)
            _batt_action_idx = int(getattr(self, '_batt_action_idx', 2))  # default: idle
            _levels = list(getattr(self.config, 'battery_discrete_action_levels', [-1.0, -0.5, 0.0, 0.5, 1.0]))
            _action_u = float(_levels[min(_batt_action_idx, len(_levels) - 1)]) if _levels else 0.0
            # _action_u: -1.0=full charge, 0.0=idle, +1.0=full discharge

            # --- Price-timing signal: +1=price expensive vs history, -1=cheap ---
            _batt_price_signal = self._compute_battery_price_reversion_signal(self.t)

            # --- SOC state ---
            _batt_soc = float(getattr(self, '_last_battery_soc', self.batt_soc_min))
            _batt_soc_min = float(self.batt_soc_min)
            _batt_soc_max = float(self.batt_soc_max)
            _batt_soc_range = max(_batt_soc_max - _batt_soc_min, 1e-6)
            _floor_slack = (_batt_soc - _batt_soc_min) / _batt_soc_range   # 0=at floor, 1=full
            _soc_at_floor = _floor_slack < 0.02   # essentially empty
            _soc_at_ceiling = _floor_slack > 0.98  # essentially full

            # --- Timing quality reward ---
            # Positive when: discharge AND price high, OR charge AND price low.
            # Negative when: discharge AND price low, OR charge AND price high (penalises bad timing).
            _timing_alpha = float(getattr(self.config, 'battery_timing_reward_weight', 0.20))
            battery_timing_bonus = _timing_alpha * float(np.clip(_batt_price_signal * _action_u, -1.0, 1.0))

            # --- SOC floor penalty: continuous incentive to maintain inventory ---
            _floor_penalty_weight = float(getattr(self.config, 'battery_floor_penalty_weight', 0.10))
            soc_penalty = -_floor_penalty_weight * float(np.clip(1.0 - _floor_slack, 0.0, 1.0))

            # --- Infeasibility penalty: penalise physically-impossible actions ---
            # Prevents the DQN from being stuck at floor choosing "discharge" forever
            # because that action earned reward when it WAS possible.
            _infeasible_weight = float(getattr(self.config, 'battery_infeasible_penalty_weight', 0.10))
            soc_infeasible_penalty = 0.0
            if _action_u > 0.0 and _soc_at_floor:     # discharge intent but already empty
                soc_infeasible_penalty = -_infeasible_weight * float(_action_u)
            elif _action_u < 0.0 and _soc_at_ceiling:  # charge intent but already full
                soc_infeasible_penalty = -_infeasible_weight * float(abs(_action_u))

            battery_pnl_delta = 0.0

            agent_rewards['battery_operator_0'] = (
                battery_base + battery_timing_bonus + soc_penalty + soc_infeasible_penalty
            )

            # ===== RISK CONTROLLER AGENT =====
            # Reward structure: allow risk-taking when returns are positive (Sharpe incentive),
            # penalize for high vol/drawdown.  Both terms are bounded to [-1, +1].
            drawdown = float(getattr(self.reward_calculator, 'current_drawdown', 0.0)) if self.reward_calculator else 0.0
            risk_vol_n   = float(np.clip(risk_level, 0.0, 1.0))
            risk_draw_n  = float(np.clip(drawdown, 0.0, 1.0))
            risk_pen_n   = float(np.clip(0.5 * risk_vol_n + 0.5 * risk_draw_n, 0.0, 1.0))
            # Positive term: reward the agent when the investor is making money under a tight cap
            # (Sharpe-like: returns with controlled risk).
            investor_return_signal = float(np.clip(investor_local_edge_delta, -1.0, 1.0))
            risk_enable_bonus = float(investor_return_signal * (1.0 - risk_pen_n))  # earn when returns good, risk low
            risk_management_delta = float(
                0.30 * risk_enable_bonus        # positive: allow profitable risk-taking
                - 0.20 * risk_pen_n             # negative: penalize high vol/drawdown
            )
            agent_rewards['risk_controller_0'] = risk_base + risk_management_delta
            
            # Meta controls the live capital allocated to the single-factor trading sleeve.
            # Reward it for two things:
            # 1) improving sleeve quality/risk outcomes through its local signal mix
            # 2) matching capital allocation to the sleeve's current quality and risk regime
            coordination_delta = 0.0
            investor_meta_signal = float(
                investor_local_edge_delta
                + 0.25 * investor_optional_edge_delta
                - investor_risk_penalty
            )
            meta_investor_weight = float(getattr(self.config, 'meta_local_investor_weight', 0.20))
            meta_battery_weight = float(getattr(self.config, 'meta_local_battery_weight', 0.00))
            meta_risk_weight = float(getattr(self.config, 'meta_local_risk_weight', 0.10))
            meta_signal_clip = float(max(getattr(self.config, 'meta_local_signal_clip', 2.0), 1e-6))
            meta_reward_weight = float(getattr(self.config, 'meta_controller_reward_weight', 1.0))
            risk_pressure = float(np.clip(0.55 * investor_local_drawdown + 0.45 * np.clip(risk_level, 0.0, 1.0), 0.0, 1.0))
            meta_local_signal = float(
                meta_investor_weight * investor_meta_signal
                + meta_battery_weight * 0.0  # battery_arbitrage_delta removed in battery 3-bug fix; weight defaults to 0.0
                - meta_risk_weight * risk_pressure
            )
            meta_local_delta = float(
                meta_reward_weight * np.clip(meta_local_signal, -meta_signal_clip, meta_signal_clip)
            )
            meta_cap_min = float(getattr(self, 'META_CAP_MIN', getattr(self.config, 'meta_cap_min', 0.10)))
            meta_cap_max = float(getattr(self, 'META_CAP_MAX', getattr(self.config, 'meta_cap_max', 0.75)))
            meta_cap_span = float(max(meta_cap_max - meta_cap_min, 1e-6))
            current_capital_fraction = float(
                np.clip(getattr(self, 'capital_allocation_fraction', meta_cap_min), meta_cap_min, meta_cap_max)
            )
            neutral_capital_fraction = float(
                np.clip(getattr(self.config, 'capital_allocation_fraction', current_capital_fraction), meta_cap_min, meta_cap_max)
            )
            investor_signal_norm = float(np.clip(investor_meta_signal / meta_signal_clip, -1.0, 1.0))
            desired_capital_fraction = float(
                np.clip(
                    neutral_capital_fraction
                    + 0.50 * meta_cap_span * investor_signal_norm
                    - 0.35 * meta_cap_span * risk_pressure,
                    meta_cap_min,
                    meta_cap_max,
                )
            )
            cap_error_norm = float(np.clip(abs(current_capital_fraction - desired_capital_fraction) / meta_cap_span, 0.0, 1.0))
            meta_capital_alignment_signal = float(np.clip(1.0 - 2.0 * cap_error_norm, -1.0, 1.0))
            meta_capital_alignment_weight = float(getattr(self.config, 'meta_capital_alignment_weight', 0.10))
            meta_capital_alignment_delta = float(
                meta_capital_alignment_weight * meta_capital_alignment_signal
            )
            
            agent_rewards['meta_controller_0'] = (
                meta_base
                + meta_local_delta
                + meta_capital_alignment_delta
            )

            # FIXED: Safe reward assignment with agent-specific rewards
            if isinstance(self.possible_agents, (list, tuple)):
                for agent in self.possible_agents:
                    if isinstance(agent, str) and agent in self._rew_buf:
                        self._rew_buf[agent] = agent_rewards.get(agent, base_reward)
            else:
                # Fallback for unexpected agent structure
                for agent in ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]:
                    if agent in self._rew_buf:
                        self._rew_buf[agent] = agent_rewards.get(agent, base_reward)

            # Store reward breakdown for logging (including agent-specific rewards)
            self.last_reward_breakdown = {
                'base_reward': base_reward,
                'investor_reward': agent_rewards.get('investor_0', base_reward),
                'battery_reward': agent_rewards.get('battery_operator_0', base_reward),
                'risk_reward': agent_rewards.get('risk_controller_0', base_reward),
                'meta_reward': agent_rewards.get('meta_controller_0', base_reward),
                'investor_mtm_delta': investor_mtm_delta,
                'investor_trading_profit_delta': investor_trading_profit_delta,
                'investor_hedging_delta': investor_hedging_delta,
                'investor_trading_return_delta': investor_trading_return_delta,
                'investor_trading_quality_delta': investor_trading_quality_delta,
                'investor_clean_reward_contract': bool(clean_reward_contract),
                'investor_local_edge_delta': investor_local_edge_delta,
                'investor_optional_edge_delta': investor_optional_edge_delta,
                'investor_risk_penalty': investor_risk_penalty,
                'investor_trading_cost_penalty': investor_trading_cost_penalty,
                'investor_trading_drawdown_penalty': investor_trading_drawdown_penalty,
                'investor_mean_clip_hit': float(investor_mean_clip_hit),
                'investor_mean_clip_hit_rate': float(getattr(self, '_investor_mean_clip_hit_rate', 0.0)),
                'investor_mean_abs_rolling': float(getattr(self, '_investor_mean_abs_rolling', 0.0)),
                'investor_mean_sign_consistency': float(getattr(self, '_investor_mean_sign_consistency', 0.0)),
                'investor_local_quality': investor_local_quality,
                'investor_local_quality_delta': investor_local_quality_delta,
                'investor_local_drawdown': investor_local_drawdown,
                'investor_active_risk_penalty': investor_active_risk_penalty,
                'investor_current_exposure_norm': current_exposure_norm,
                'investor_realized_volatility_regime': realized_volatility_regime,
                'forecast_prior_enabled': bool(getattr(self.config, "enable_forecast_utilization", False)),
                'forecast_prior_exposure': float((getattr(self, "_last_forecast_prior", {}) or {}).get("prior_exposure", 0.0)),
                'forecast_prior_skill': float((getattr(self, "_last_forecast_prior", {}) or {}).get("skill", 0.0)),
                'forecast_prior_hit_lcb': float((getattr(self, "_last_forecast_prior", {}) or {}).get("hit_lcb", 0.5)),
                'forecast_prior_residual_q': float((getattr(self, "_last_forecast_prior", {}) or {}).get("residual_q", 0.0)),
                'forecast_prior_active': bool((getattr(self, "_last_forecast_prior", {}) or {}).get("active", False)),
                'battery_arbitrage_delta': 0.0,
                'risk_management_delta': risk_management_delta,
                'meta_local_delta': meta_local_delta,
                'meta_capital_alignment_delta': meta_capital_alignment_delta,
                'meta_capital_alignment_signal': meta_capital_alignment_signal,
                'meta_desired_capital_fraction': desired_capital_fraction,
                'meta_current_capital_fraction': current_capital_fraction,
                'meta_local_signal': meta_local_signal,
                'coordination_delta': coordination_delta,
                'fund_nav': fund_nav,
                'cash_flow': cash_flow,
                'risk_level': risk_level,
                'efficiency': efficiency,
                'reward_components': self.reward_calculator.reward_weights.copy() if self.reward_calculator else {}
            }
            self.last_reward_weights = self.reward_calculator.reward_weights.copy() if self.reward_calculator else {}
            
            # DEEP DEBUGGING: Log reward breakdown
            # CRITICAL FIX: Always log, even if reward_calculator is None (for debugging)
            if hasattr(self, 'debug_tracker'):
                # Ensure debug tracker is initialized (safety check)
                if self.debug_tracker.csv_writer is None:
                    if self._episode_counter < 0:
                        self._episode_counter = 0
                    self.debug_tracker.start_episode(self._episode_counter)
                    logger.info(f"[DEBUG_TRACKER] Initialized CSV writer for episode {self._episode_counter}")

                # DIAGNOSTIC: Log every 100 steps to verify logging is working
            if self.t % 100 == 0:
                logger.debug(f"[DEBUG_TRACKER] t={self.t} csv_writer={'EXISTS' if self.debug_tracker.csv_writer else 'NONE'} "
                           f"reward_calc={'EXISTS' if self.reward_calculator else 'NONE'}")

            # Ensure USD conversion rate is present
            if not hasattr(self, 'usd_conversion_rate'):
                self.usd_conversion_rate = float(getattr(self.config, 'usd_conversion_rate', 0.145))

            # Ensure reward_calculator exists before logging
            if self.reward_calculator is None:
                try:
                    post_capex_nav = float(self._calculate_fund_nav())
                except Exception:
                    post_capex_nav = float(getattr(self, 'init_budget', 0.0))
                self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
                self.reward_weights = dict(getattr(self.reward_calculator, 'reward_weights', {}))

            if hasattr(self, 'debug_tracker') and self.reward_calculator is not None:
                # PRINT to console to verify this block is reached
                if self.t % 100 == 0:
                    logger.debug(f"[DEBUG_TRACKER] t={self.t} ENTERING MAIN LOGGING BLOCK (reward_calculator EXISTS)")

                reward_weights = self.reward_calculator.reward_weights
                
                # Get reward component scores from calculator (ensure they're floats, not arrays)
                operational_score = float(getattr(self.reward_calculator, 'last_operational_score', 0.0))
                risk_score = float(getattr(self.reward_calculator, 'last_risk_score', 0.0))
                hedging_score = float(getattr(self.reward_calculator, 'last_hedging_score', 0.0))
                nav_stability_score = float(getattr(self.reward_calculator, 'last_nav_stability_score', 0.0))
                
            # Retrieve latest info dict (may be set in step); fall back to empty
            info = getattr(self, '_latest_info', {})
            if not isinstance(info, dict):
                info = {}

            # Get actions from last step (stored in _last_actions)
            last_actions = getattr(self, '_last_actions', {})
            # Prefer executed (post-simplex) investor action if available
            investor_action = last_actions.get('investor_0_exec', last_actions.get('investor_0', {}))
            battery_action = last_actions.get('battery_operator_0', {})
            
            # Ensure price_return_1step (and horizon-aligned) are available in info for aligned aux targets (metacontroller)
            info_investor = info.get('investor_0', {})
            try:
                info_investor['price_return_1step'] = float(
                    getattr(self, '_latest_price_returns', {}).get('one_step', 0.0)
                )
            except Exception:
                info_investor['price_return_1step'] = 0.0
            # Horizon-aligned return over investment_freq steps (decision cadence)
            try:
                anchor_step = int(
                    max(
                        0,
                        min(
                            int(getattr(self, '_prev_investor_decision_step', i - int(max(getattr(self, 'investment_freq', 1) or 1, 1)))),
                            max(i - 1, 0),
                        ),
                    )
                )
                if anchor_step >= 0 and hasattr(self, '_price_raw') and i < len(self._price_raw):
                    prev_p = float(self._price_raw[anchor_step])
                    curr_p = float(self._price_raw[i])
                    ret_h = (curr_p - prev_p) / max(abs(prev_p), 1e-6)
                    info_investor['price_return_invfreq'] = float(np.clip(ret_h, -0.5, 0.5))
                else:
                    info_investor['price_return_invfreq'] = 0.0
            except Exception:
                info_investor['price_return_invfreq'] = 0.0
            info_investor['training_return'] = float(
                info_investor.get('price_return_invfreq', info_investor.get('price_return_1step', 0.0))
            )
            info_investor['training_horizon_steps'] = float(max(getattr(self, 'investment_freq', 1) or 1, 1))
            info['investor_0'] = info_investor

            # Fund NAV component breakdown (DKK) for drop attribution and progress reporting.
            try:
                current_timestep = int(self.t)
                nav_breakdown = self.get_fund_nav_breakdown(fund_nav=fund_nav, current_timestep=current_timestep)
                usd_rate = float(getattr(self, 'usd_conversion_rate', getattr(self.config, 'usd_conversion_rate', 0.145)))
                fund_nav_dkk = float(nav_breakdown['fund_nav_dkk'])
                current_fund_nav_usd = float(nav_breakdown['fund_nav_usd'])
                trading_cash_dkk = float(nav_breakdown['trading_cash_dkk'])
                physical_book_value_dkk = float(nav_breakdown['physical_book_value_dkk'])
                accumulated_operational_revenue_dkk = float(nav_breakdown['accumulated_operational_revenue_dkk'])
                financial_mtm_dkk = float(nav_breakdown['financial_mtm_dkk'])
                financial_exposure_dkk = float(nav_breakdown['financial_exposure_dkk'])
                total_distributions_dkk = float(nav_breakdown.get('total_distributions_dkk', 0.0))
                distribution_adjusted_nav_dkk = float(nav_breakdown.get('distribution_adjusted_nav_dkk', fund_nav_dkk + total_distributions_dkk))
                depreciation_ratio = float(nav_breakdown['depreciation_ratio'])
                years_elapsed = float(nav_breakdown['years_elapsed'])
                current_price_raw = float(self._price_raw[current_timestep] if current_timestep < len(self._price_raw) else 0.0)
                current_cash_dkk = float(getattr(self, 'budget', 0.0))
                trading_sleeve_value_dkk = float(nav_breakdown.get('trading_sleeve_value_dkk', trading_cash_dkk + financial_mtm_dkk))
                distribution_adjusted_trading_sleeve_dkk = float(
                    nav_breakdown.get(
                        'distribution_adjusted_trading_sleeve_dkk',
                        trading_sleeve_value_dkk + total_distributions_dkk,
                    )
                )
                cumulative_mtm = float(nav_breakdown.get('trading_mtm_tracker_dkk', 0.0))
                current_trading_sleeve_value_usd = float(trading_sleeve_value_dkk * usd_rate / 1000.0)
                current_trading_mtm_tracker_usd = float(cumulative_mtm * usd_rate / 1000.0)
                current_operating_revenue_usd = float(accumulated_operational_revenue_dkk * usd_rate / 1000.0)
            except Exception as e:
                logger.exception(f"[DEBUG_TRACKER] logging failed at t={self.t}: {e}")
                raise

            # PRINT to verify we're about to call log_step
            if self.t % 100 == 0:
                logger.debug(f"[DEBUG_TRACKER] t={self.t} CALLING log_step() NOW!")

            # Investor action/exposure diagnostics (logging only)
            action_dir = 0.0
            exposure_val = 0.0
            delta_exposure = 0.0
            try:
                if isinstance(investor_action, dict):
                    weights = []
                    for k in ('wind', 'solar', 'hydro'):
                        if k in investor_action:
                            v = investor_action.get(k, 0.0)
                            if isinstance(v, np.ndarray):
                                v = float(v.item() if v.size == 1 else v.flatten()[0])
                            weights.append(float(v))
                    if weights:
                        action_dir = float(np.sign(np.sum(weights)))
                    if 'exposure' in investor_action:
                        exposure_val = float(investor_action.get('exposure', 0.0))
                else:
                    arr = np.asarray(investor_action, dtype=np.float32).flatten()
                    if arr.size >= 3:
                        action_dir = float(np.sign(float(np.sum(arr[:3]))))
                    if arr.size >= 1:
                        exposure_val = float(arr[0])
            except Exception:
                action_dir = 0.0
                exposure_val = 0.0
                delta_exposure = 0.0

            decision_step = self._decision_step_flag(self.t)
            exposure_exec = 0.0
            try:
                if isinstance(getattr(self, "_last_actions", None), dict):
                    exec_action = self._last_actions.get("investor_0_exec", {})
                    exposure_exec = float(exec_action.get("exposure", 0.0))
                    delta_exposure = float(exec_action.get("delta_exposure", 0.0))
            except Exception:
                exposure_exec = 0.0
                delta_exposure = 0.0
            try:
                value_diag = dict(getattr(self, "_last_value_diag", {}) or {})
            except Exception:
                value_diag = {}

            action_sign = float(np.sign(exposure_exec)) if abs(exposure_exec) > 1e-9 else float(np.sign(exposure_val))

            # Derive logging-only diagnostics from executed action + live finance state.
            # This avoids stale zeros from debug dict fields.
            exec_action = {}
            try:
                if isinstance(getattr(self, "_last_actions", None), dict):
                    exec_action = self._last_actions.get("investor_0_exec", {}) or {}
                exec_wind = float(exec_action.get("wind", 0.0))
                exec_solar = float(exec_action.get("solar", 0.0))
                exec_hydro = float(exec_action.get("hydro", 0.0))
            except Exception:
                exec_wind = 0.0
                exec_solar = 0.0
                exec_hydro = 0.0

            prior_signal_log = 0.0
            try:
                prior_signal_log = float(exec_action.get("forecast_prior_exposure", exec_action.get("prior_target", 0.0)))
            except Exception:
                prior_signal_log = 0.0
            trade_signal_active = 1.0 if (
                float(exec_action.get("forecast_prior_active", 0.0)) > 0.5
                and abs(prior_signal_log) > 1e-9
            ) else 0.0
            trade_signal_sign = float(np.sign(prior_signal_log)) if trade_signal_active > 0.0 else 0.0

            position_signed_log = float(exposure_exec)
            if abs(position_signed_log) <= 1e-9:
                position_signed_log = float(np.sign(exposure_val)) if abs(exposure_val) > 1e-9 else 0.0
            position_direction_log = int(np.sign(position_signed_log)) if abs(position_signed_log) > 1e-9 else 0

            current_capital_fraction_log = float(
                np.clip(
                    getattr(self, 'capital_allocation_fraction', getattr(self.config, 'capital_allocation_fraction', 0.0)),
                    0.0,
                    1.0,
                )
            )
            max_position_notional_dkk = float(
                max(
                    float(getattr(self.config, 'max_position_size', 0.0))
                    * float(self.init_budget)
                    * current_capital_fraction_log,
                    1.0,
                )
            )
            investor_total_position_dkk = float(
                sum(abs(float(v)) for v in getattr(self, 'financial_positions', {}).values())
            )
            investor_position_ratio_log = float(
                np.clip(investor_total_position_dkk / max_position_notional_dkk, 0.0, 1.0)
            )

            latest_price_returns = getattr(self, '_latest_price_returns', {}) or {}
            price_return_1step_log = float(latest_price_returns.get('one_step', 0.0))
            if not np.isfinite(price_return_1step_log):
                price_return_1step_log = 0.0
            fallback_horizon_steps_log = int(max(1, getattr(self, 'investment_freq', 1) or 1))
            t_idx_log = int(max(0, int(getattr(self, 't', 0))))
            decision_anchor_step_log = int(
                max(
                    0,
                    min(
                        int(getattr(self, '_prev_investor_decision_step', t_idx_log - fallback_horizon_steps_log)),
                        max(t_idx_log - 1, 0),
                    ),
                )
            )
            training_horizon_steps_log = int(max(1, t_idx_log - decision_anchor_step_log))
            price_return_invfreq_log = 0.0
            price_series_log = getattr(self, '_price_raw', None)
            if (
                price_series_log is not None
                and decision_anchor_step_log < len(price_series_log)
                and t_idx_log < len(price_series_log)
            ):
                prev_price_log = float(price_series_log[decision_anchor_step_log])
                curr_price_log = float(price_series_log[t_idx_log])
                if np.isfinite(prev_price_log) and np.isfinite(curr_price_log):
                    price_return_invfreq_log = float(
                        np.clip((curr_price_log - prev_price_log) / max(abs(prev_price_log), 1e-6), -0.5, 0.5)
                    )
            if not np.isfinite(price_return_invfreq_log):
                price_return_invfreq_log = 0.0
            training_return_log = float(
                price_return_invfreq_log if np.isfinite(price_return_invfreq_log) else price_return_1step_log
            )

            pnl_generation_log = float(financial.get('generation_revenue', 0.0))
            pnl_hedge_log = float(financial.get('mtm_pnl', 0.0))
            pnl_battery_log = float(getattr(self, '_last_battery_cash_delta', financial.get('battery_cash_delta', 0.0)))
            pnl_total_log = float(pnl_generation_log + pnl_hedge_log + pnl_battery_log)
            cash_delta_ops_log = float(financial.get('revenue', 0.0))
            # Keep last-* debug attributes consistent with what we write to CSV.
            self._last_generation_pnl = pnl_generation_log
            self._last_hedge_pnl = pnl_hedge_log
            self._last_total_pnl = pnl_total_log
            self._last_cash_ops = cash_delta_ops_log

            # Log per-step fund NAV movement, not the trading-budget fallback.
            nav_start_raw = getattr(self, '_last_nav', None)
            if nav_start_raw is None or (not np.isfinite(float(nav_start_raw))):
                nav_start_raw = getattr(self, 'init_budget', fund_nav)
            nav_start_log = float(nav_start_raw)
            nav_end_log = float(fund_nav)

            # Ensure all values are floats (not arrays) to avoid "ambiguous truth value" error
            self.debug_tracker.log_step(
                timestep=int(self.t),
                # Portfolio metrics (logged at every timestep)
                portfolio_value_usd_millions=float(current_fund_nav_usd / 1_000_000.0),
                fund_nav_usd=float(current_fund_nav_usd),
                cash_dkk=float(current_cash_dkk),
                trading_sleeve_value_usd_thousands=float(current_trading_sleeve_value_usd),
                trading_mtm_tracker_usd_thousands=float(current_trading_mtm_tracker_usd),
                operating_revenue_usd_thousands=float(current_operating_revenue_usd),
                # Fund NAV component breakdown (DKK)
                fund_nav_dkk=float(fund_nav_dkk),
                distribution_adjusted_nav_dkk=float(distribution_adjusted_nav_dkk),
                total_distributions_dkk=float(total_distributions_dkk),
                trading_cash_dkk=float(trading_cash_dkk),
                trading_sleeve_value_dkk=float(trading_sleeve_value_dkk),
                distribution_adjusted_trading_sleeve_dkk=float(distribution_adjusted_trading_sleeve_dkk),
                physical_book_value_dkk=float(physical_book_value_dkk),
                accumulated_operational_revenue_dkk=float(accumulated_operational_revenue_dkk),
                financial_mtm_dkk=float(financial_mtm_dkk),
                financial_exposure_dkk=float(financial_exposure_dkk),
                depreciation_ratio=float(depreciation_ratio),
                years_elapsed=float(years_elapsed),
                # Price data (for forward-looking accuracy analysis)
                    price_current=float(current_price_raw),
                    # Position info (ensure floats)
                    position_signed=float(position_signed_log),
                    # Recompute from current financial notional so the log reflects held exposure.
                    position_exposure=float(
                        np.clip(
                            (
                                sum(abs(float(v)) for v in getattr(self, 'financial_positions', {}).values())
                                / max(
                                    float(getattr(self.config, 'max_position_size', 0.0))
                                    * float(self.init_budget)
                                    * current_capital_fraction_log,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    ),
                    decision_step=float(decision_step),
                    exposure_exec=float(exposure_exec),
                    action_sign=float(action_sign),
                    trade_signal_active=float(trade_signal_active),
                    trade_signal_sign=float(trade_signal_sign),
                    risk_multiplier=float(getattr(self, "_last_risk_multiplier", 1.0)),
                    tradeable_capital=float(getattr(self, "_last_tradeable_capital", 0.0)),
                    mtm_exit_count=float(getattr(self, "_last_mtm_exit_count", 0.0)),
                    forecast_prior_exposure=float(exec_action.get("forecast_prior_exposure", 0.0)),
                    forecast_prior_target=float(exec_action.get("prior_target", exec_action.get("exposure", 0.0))),
                    forecast_prior_blend=float(exec_action.get("forecast_prior_blend", exec_action.get("prior_blend", 0.0))),
                    forecast_prior_active=float(exec_action.get("forecast_prior_active", 0.0)),
                    forecast_prior_skill=float(exec_action.get("forecast_prior_skill", 0.0)),
                    forecast_prior_hit_lcb=float(exec_action.get("forecast_prior_hit_lcb", 0.5)),
                    forecast_prior_residual_q=float(exec_action.get("forecast_prior_residual_q", 0.0)),
                    forecast_prior_magnitude=float(exec_action.get("forecast_prior_magnitude", 0.0)),
                    forecast_prior_edge_excess=float(exec_action.get("forecast_prior_edge_excess", 0.0)),
                    forecast_prior_reason=str(exec_action.get("forecast_prior_reason", "")),
                    # Price returns (ensure floats)
                    price_return_1step=float(price_return_1step_log),
                    price_return_invfreq=float(price_return_invfreq_log),
                    training_return=float(training_return_log),
                    training_horizon_steps=float(training_horizon_steps_log),
                    # Main reward components (already floats from above)
                    operational_score=operational_score,
                    risk_score=risk_score,
                    hedging_score=hedging_score,
                    nav_stability_score=nav_stability_score,
                    # Reward weights (ensure floats)
                    weight_operational=float(reward_weights.get('operational_revenue', 0.0)),
                    weight_risk=float(reward_weights.get('risk_management', 0.0)),
                    weight_hedging=float(reward_weights.get('hedging_effectiveness', 0.0)),
                    weight_nav=float(reward_weights.get('nav_stability', 0.0)),
                    unrealized_pnl_norm=0.0,
                    combined_confidence=0.0,
                    adaptive_multiplier=0.0,
                    wind_pos_norm=float(exec_wind),
                    solar_pos_norm=float(exec_solar),
                    hydro_pos_norm=float(exec_hydro),
                    # NEW: Position alignment diagnostics (Bug #1 fix tracking)
                    position_alignment_status=str(position_alignment_status) if 'position_alignment_status' in locals() else 'no_strategy',
                    investor_position_ratio=float(investor_position_ratio_log),
                    investor_position_direction=int(position_direction_log),
                    investor_total_position=float(investor_total_position_dkk),
                    investor_exploration_bonus=float(exploration_bonus) if 'exploration_bonus' in locals() else 0.0,
                    # NEW: Price floor diagnostics (Bug #4 fix tracking)
                    price_raw=float(getattr(self, '_price_raw', [0.0])[self.t] if hasattr(self, '_price_raw') and self.t < len(self._price_raw) else 0.0),
                    price_floor_used=float(getattr(self, '_current_price_floor', 0.0)),
                    # NEW: NAV attribution drivers (per-step fund NAV breakdown)
                    nav_start=float(nav_start_log),
                    nav_end=float(nav_end_log),
                    pnl_total=float(pnl_total_log),
                    pnl_battery=float(pnl_battery_log),
                    pnl_generation=float(pnl_generation_log),
                    pnl_hedge=float(pnl_hedge_log),
                    cash_delta_ops=float(cash_delta_ops_log),
                    # NEW: Battery dispatch metrics (FIX #5)
                    battery_decision=str(getattr(self, '_last_battery_decision', 'idle')),
                    battery_intensity=float(getattr(self, '_last_battery_intensity', 0.0)),
                    battery_spread=float(getattr(self, '_last_battery_spread', 0.0)),
                    battery_adjusted_hurdle=float(getattr(self, '_last_battery_adjusted_hurdle', 0.0)),
                    battery_volatility_adj=float(getattr(self, '_last_battery_volatility_adj', 0.0)),
                    # NEW: Battery state metrics for diagnostics
                    battery_energy=float(getattr(self, '_last_battery_energy', self.operational_state.get('battery_energy', 0.0))),
                    battery_capacity=float(getattr(self, '_last_battery_capacity', self.physical_assets.get('battery_capacity_mwh', 0.0))),
                    battery_soc=float(getattr(self, '_last_battery_soc', 0.0)),
                    battery_cash_delta=float(getattr(self, '_last_battery_cash_delta', 0.0)),
                    battery_throughput=float(getattr(self, '_last_battery_throughput', 0.0)),
                    battery_degradation_cost=float(getattr(self, '_last_battery_degradation_cost', 0.0)),
                    battery_eta_charge=float(getattr(self, '_last_battery_eta_charge', self.batt_eta_charge)),
                    battery_eta_discharge=float(getattr(self, '_last_battery_eta_discharge', self.batt_eta_discharge)),
                    # Final rewards (ensure floats)
                    base_reward=float(base_reward),
                    investor_reward=float(agent_rewards.get('investor_0', base_reward)),
                    battery_reward=float(agent_rewards.get('battery_operator_0', base_reward)),
                    # Financial metrics (ensure floats)
                    mtm_pnl=float(mtm_pnl),  # Per-step MTM PnL, not cumulative MTM tracker
                    # Investor health diagnostics
                    training_global_step=int(getattr(self.config, 'training_global_step', 0)) if self.config is not None else 0,
                    inv_mean_clip_hit=float(getattr(self, '_last_penalty_diag', {}).get('inv_mean_clip_hit', 0.0)),
                    inv_mean_clip_hit_rate=float(getattr(self, '_last_penalty_diag', {}).get('inv_mean_clip_hit_rate', 0.0)),
                    inv_mu_abs_roll=float(getattr(self, '_last_penalty_diag', {}).get('inv_mu_abs_roll', 0.0)),
                    inv_mu_sign_consistency=float(getattr(self, '_last_penalty_diag', {}).get('inv_mu_sign_consistency', 0.0)),
                    # Forecast diagnostics
                    # Investor policy diagnostics: native distribution stats plus
                    # deployed action-space mean/sample.
                    inv_mu_raw=float(getattr(self, '_inv_mu_raw', 0.0)),
                    inv_sigma_raw=float(getattr(self, '_inv_sigma_raw', 0.0)),
                    inv_a_raw=float(getattr(self, '_inv_a_raw', 0.0)),
                    inv_action_mean=float(getattr(self, '_inv_action_mean', getattr(self, '_inv_tanh_mu', 0.0))),
                    inv_action_sigma=float(getattr(self, '_inv_action_sigma', getattr(self, '_inv_sigma_raw', 0.0))),
                    inv_action_sample=float(getattr(self, '_inv_action_sample', getattr(self, '_inv_tanh_a', 0.0))),
                    inv_tanh_mu=float(getattr(self, '_inv_tanh_mu', 0.0)),
                    inv_tanh_a=float(getattr(self, '_inv_tanh_a', 0.0)),
                    inv_sat_mean=float(getattr(self, '_inv_sat_mean', 0.0)),
                    inv_sat_sample=float(getattr(self, '_inv_sat_sample', 0.0)),
                    inv_sat_noise_only=float(getattr(self, '_inv_sat_noise_only', 0.0)),
                    # Investor TD-error proxy (credit assignment diagnostics)
                    inv_reward_step=float(getattr(self, '_inv_reward_step', 0.0)),
                    inv_value=float(getattr(self, '_inv_value', 0.0)),
                    inv_value_next=float(getattr(self, '_inv_value_next', 0.0)),
                    inv_td_error=float(getattr(self, '_inv_td_error', 0.0)),
                    # Battery operator health diagnostics
                    bat_decision_step=float(self.t % max(int(getattr(self.config, 'battery_action_freq', 1)), 1) == 0),
                    bat_action_idx=int(getattr(self, '_batt_action_idx', 0)),
                    bat_q_max=float(getattr(self, '_batt_q_max', 0.0)),
                    bat_q_chosen=float(getattr(self, '_batt_q_chosen', 0.0)),
                    bat_reward_step=float(getattr(self, '_batt_reward_step', 0.0)),
                    bat_soc=float(getattr(self, '_last_battery_soc', 0.0)),
                    bat_spread=float(getattr(self, '_last_battery_spread', 0.0)),
                    bat_energy=float(getattr(self, '_last_battery_energy', self.operational_state.get('battery_energy', 0.0))),
                    bat_capacity=float(getattr(self, '_last_battery_capacity', self.physical_assets.get('battery_capacity_mwh', 0.0))),
                    # Risk controller health diagnostics
                    risk_decision_step=float(self._is_risk_decision_step(self.t)),
                    risk_action_raw=float(getattr(self, '_risk_action_raw', 0.0)),
                    risk_reward_step=float(getattr(self, '_risk_reward_step', 0.0)),
                    risk_value=float(getattr(self, '_risk_value', 0.0)),
                    risk_value_next=float(getattr(self, '_risk_value_next', 0.0)),
                    risk_td_error=float(getattr(self, '_risk_td_error', 0.0)),
                    risk_drawdown=float(np.clip(getattr(self, 'current_drawdown', 0.0), 0.0, 1.0)),
                    overall_risk_snapshot=float(np.clip(getattr(self, 'overall_risk_snapshot', 0.0), 0.0, 1.0)),
                    # Meta controller health diagnostics
                    meta_decision_step=float(self._is_meta_decision_step(self.t)),
                    meta_action_raw=float(getattr(self, '_meta_action_raw', 0.0)),
                    meta_reward_step=float(getattr(self, '_meta_reward_step', 0.0)),
                    meta_value=float(getattr(self, '_meta_value', 0.0)),
                    meta_value_next=float(getattr(self, '_meta_value_next', 0.0)),
                    meta_td_error=float(getattr(self, '_meta_td_error', 0.0)),
                    meta_cap_fraction=float(np.clip(getattr(self, 'capital_allocation_fraction', 0.0), 0.0, 1.0)),
                    meta_budget_n=float(np.clip(
                        getattr(self, 'budget', 0.0) / max(float(getattr(self, 'trading_allocation_budget', getattr(self, 'init_budget', 1.0) * 0.12)), 1.0),
                        0.0, 2.0,
                    )),
                    # Actions
                    investor_action=investor_action,
                    battery_action=battery_action,
                )
            # Initialize _last_nav on first step
            if self._last_nav is None:
                self._last_nav = fund_nav
            else:
                self._last_nav = fund_nav

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Reward assignment error: {e}")
            logger.error(f"Traceback: {tb_str}")
            # FIXED: Safe error handling with fallback
            try:
                if isinstance(self.possible_agents, (list, tuple)):
                    for agent in self.possible_agents:
                        if isinstance(agent, str) and agent in self._rew_buf:
                            self._rew_buf[agent] = 0.0
                else:
                    # Fallback for unexpected agent structure
                    for agent in ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]:
                        if agent in self._rew_buf:
                            self._rew_buf[agent] = 0.0
            except Exception as e2:
                logger.error(f"Error in reward error handling: {e2}")
                # Last resort: direct assignment
                self._rew_buf = {a: 0.0 for a in ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]}

    def _update_market_conditions(self, i: int):
        try:
            hist = list(self.performance_history['revenue_history'])
            if len(hist) > 4:
                r = np.diff(np.asarray(hist, dtype=float))
                self.market_volatility = float(np.clip(np.std(r) / (np.mean(np.abs(hist)) + 1e-8), 0.0, 1.0))
            self.market_stress = float(np.clip(self._riskS[i], 0.0, 1.0))
        except Exception:
            self.market_volatility = 0.2
            self.market_stress = 0.5

    def _update_risk_snapshots(self, i: int):
        try:
            if self.enhanced_risk_controller is None:
                return

            if hasattr(self.enhanced_risk_controller, "update_risk_history") and \
               hasattr(self.enhanced_risk_controller, "calculate_comprehensive_risk"):
                fp = getattr(self, 'financial_positions', {}) or {}
                env_state = {
                    'price': float(self._price_raw[i]) if hasattr(self, '_price_raw') and i < len(self._price_raw)
                             else float(self._price[i] * self._price_std[i] + self._price_mean[i]) if hasattr(self, '_price_std') and hasattr(self, '_price_mean') and i < len(self._price_std) and i < len(self._price_mean)
                             else float(self._price[i]),
                    'budget': self.budget,
                    'initial_budget': self.init_budget,
                    'timestep': self.t,
                    # Canonical keys expected by EnhancedRiskController:
                    'wind_capacity_mw': self.physical_assets['wind_capacity_mw'],
                    'solar_capacity_mw': self.physical_assets['solar_capacity_mw'],
                    'hydro_capacity_mw': self.physical_assets['hydro_capacity_mw'],
                    'battery_capacity_mwh': self.physical_assets['battery_capacity_mwh'],
                    # Short keys expected by EnhancedRiskController / risk adapters:
                    'wind_capacity': self.physical_assets['wind_capacity_mw'],
                    'solar_capacity': self.physical_assets['solar_capacity_mw'],
                    'hydro_capacity': self.physical_assets['hydro_capacity_mw'],
                    'battery_capacity': self.physical_assets['battery_capacity_mwh'],
                    'wind': float(self._wind[i]) if i < len(self._wind) else 0.0,
                    'solar': float(self._solar[i]) if i < len(self._solar) else 0.0,
                    'hydro': float(self._hydro[i]) if i < len(self._hydro) else 0.0,
                    'revenue': self.last_revenue,
                    'market_stress': self.market_stress,
                    # Trading exposure for portfolio risk (investor positions)
                    'financial_positions': dict(fp),
                }
                self.enhanced_risk_controller.update_risk_history(env_state)
                comp = self.enhanced_risk_controller.calculate_comprehensive_risk(env_state)

                self.overall_risk_snapshot = float(np.clip(comp.get('overall_risk', 0.5), 0.0, 1.0))
                self.market_risk_snapshot = float(np.clip(comp.get('market_risk', 0.3), 0.0, 1.0))
                self.portfolio_risk_snapshot = float(np.clip(comp.get('portfolio_risk', 0.25), 0.0, 1.0))
                self.liquidity_risk_snapshot = float(np.clip(comp.get('liquidity_risk', 0.15), 0.0, 1.0))
                return

        except Exception as e:
            msg = f"[RISK_SNAPSHOT_FATAL] Risk snapshot update failed at step {self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    # ------------------------------------------------------------------
    # Observations (no forecast-augmented obs)
    # ------------------------------------------------------------------
    def _compute_investor_price_momentum(self, t: int) -> float:
        """
        Compute normalized price momentum for investor observations.
        Uses return over investment horizon (6 steps = 1h) for directional signal.
        Positive = price rising, negative = falling. Replaces price level for trading decisions.
        """
        try:
            horizon = int(getattr(self, "investment_freq", 6))
            price_floor = float(max(getattr(self.config, "minimum_price_filter", 10.0), 1.0))
            momentum_scale = float(getattr(self.config, "investor_price_momentum_scale", 0.08))
            price_raw = getattr(self, "_price_raw", None)
            if price_raw is None or len(price_raw) < 2:
                return 0.0
            t_prev = max(0, t - horizon)
            p_now = float(np.clip(price_raw[t] if t < len(price_raw) else price_raw[-1], -1000.0, 1e9))
            p_prev = float(np.clip(price_raw[t_prev] if t_prev < len(price_raw) else price_raw[0], -1000.0, 1e9))
            denom = max(abs(p_prev), price_floor)
            raw_return = (p_now - p_prev) / denom
            momentum_n = float(np.clip(raw_return / momentum_scale, -1.0, 1.0))
            return momentum_n
        except Exception:
            return 0.0

    def _compute_investor_realized_volatility(self, t: int) -> float:
        """Compute normalized short-horizon realized volatility for the investor."""
        try:
            price_raw = getattr(self, "_price_raw", None)
            if price_raw is None or len(price_raw) < 3:
                return 0.0

            lookback = max(2, int(getattr(self.config, "investor_recent_return_lookback", 12)))
            vol_scale = float(max(getattr(self.config, "investor_realized_vol_scale", 0.05), 1e-6))
            price_floor = float(max(getattr(self.config, "minimum_price_filter", 10.0), 1.0))

            start = max(1, t - lookback + 1)
            window = np.asarray(price_raw[start - 1:t + 1], dtype=np.float64)
            if window.size < 3:
                return 0.0

            prev = np.maximum(np.abs(window[:-1]), price_floor)
            returns = np.diff(window) / prev
            returns = returns[np.isfinite(returns)]
            if returns.size == 0:
                return 0.0

            realized_vol = float(np.std(returns)) if returns.size > 1 else 0.0
            realized_vol_n = float(np.clip(realized_vol / vol_scale, 0.0, 1.0))
            return realized_vol_n
        except Exception:
            return 0.0

    def _compute_battery_price_reversion_signal(self, t: int) -> float:
        """
        Mean-reversion style battery edge signal from historical normalized price.
        Positive means current price is rich versus recent history (favor discharge),
        negative means cheap versus recent history (favor charge).
        """
        try:
            price_series = getattr(self, "_price", None)
            if price_series is None or len(price_series) < 4:
                return 0.0
            window = 36
            start = max(0, t - window + 1)
            hist = np.asarray(price_series[start:t + 1], dtype=np.float64)
            if hist.size < 4:
                return 0.0
            baseline = hist[:-1]
            mean = float(np.mean(baseline))
            std = max(float(np.std(baseline)), 0.10)
            signal = (float(hist[-1]) - mean) / (2.0 * std)
            return float(np.clip(signal, -1.0, 1.0))
        except Exception:
            return 0.0

    def _compute_battery_price_momentum(self, t: int) -> float:
        """Short-horizon battery price momentum from historical normalized price."""
        try:
            price_series = getattr(self, "_price", None)
            if price_series is None or len(price_series) < 2:
                return 0.0
            lookback = 6
            t_prev = max(0, t - lookback)
            p_now = float(price_series[t] if t < len(price_series) else price_series[-1])
            p_prev = float(price_series[t_prev] if t_prev < len(price_series) else price_series[0])
            return float(np.clip(p_now - p_prev, -1.0, 1.0))
        except Exception:
            return 0.0

    def _compute_battery_realized_price_volatility(self, t: int) -> float:
        """Realized short-horizon volatility from normalized price dynamics."""
        try:
            price_series = getattr(self, "_price", None)
            if price_series is None or len(price_series) < 4:
                return 0.0
            window = 36
            start = max(0, t - window + 1)
            hist = np.asarray(price_series[start:t + 1], dtype=np.float64)
            if hist.size < 4:
                return 0.0
            diffs = np.diff(hist)
            if diffs.size == 0:
                return 0.0
            vol = float(np.std(diffs))
            return float(np.clip(vol / 0.25, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_battery_time_features(self, t: int) -> Tuple[float, float, float, float]:
        """Cyclical time features for daily/weekly storage arbitrage structure."""
        try:
            if hasattr(self, "data") and self.data is not None and "timestamp" in self.data.columns:
                idx = min(max(int(t), 0), max(len(self.data) - 1, 0))
                ts = pd.to_datetime(self.data.iloc[idx]["timestamp"], errors="coerce")
                if pd.notna(ts):
                    tod_frac = ((int(ts.hour) * 60) + int(ts.minute)) / (24.0 * 60.0)
                    dow_frac = (((int(ts.dayofweek) * 24 + int(ts.hour)) * 60) + int(ts.minute)) / (7.0 * 24.0 * 60.0)
                    tod_angle = 2.0 * np.pi * tod_frac
                    dow_angle = 2.0 * np.pi * dow_frac
                    return (
                        float(np.sin(tod_angle)),
                        float(np.cos(tod_angle)),
                        float(np.sin(dow_angle)),
                        float(np.cos(dow_angle)),
                    )
        except Exception:
            pass

        step_h = float(getattr(self.config, "time_step_hours", 10.0 / 60.0))
        steps_per_day = max(int(round(24.0 / max(step_h, 1e-6))), 1)
        steps_per_week = max(steps_per_day * 7, 1)
        tod_angle = 2.0 * np.pi * ((int(t) % steps_per_day) / float(steps_per_day))
        dow_angle = 2.0 * np.pi * ((int(t) % steps_per_week) / float(steps_per_week))
        return (
            float(np.sin(tod_angle)),
            float(np.cos(tod_angle)),
            float(np.sin(dow_angle)),
            float(np.cos(dow_angle)),
        )

    def _get_initial_battery_energy(self) -> float:
        """Neutral in-band ESS initialization instead of an empty battery."""
        try:
            capacity = float(self.physical_assets.get("battery_capacity_mwh", 0.0))
            if capacity <= 0.0:
                return 0.0
            init_soc = float(
                np.clip(
                    getattr(self.config, "battery_initial_soc", 0.5),
                    self.batt_soc_min,
                    self.batt_soc_max,
                )
            )
            return float(init_soc * capacity)
        except Exception:
            return 0.0

    def _fill_obs(self):
        """Fill all agent observations.

        Investor (12D) + battery (12D) are built by ObservationBuilder.
        Risk (11D) + meta (11D) are populated here (kept simple and stable).
        """
        try:
            i = min(self.t, self.max_steps - 1)

            market_features = ProductionObservationBuilder.normalize_market_features(
                price=self._price[i],
                load=self._load[i],
                wind=self._wind[i],
                solar=self._solar[i],
                hydro=self._hydro[i],
                load_scale=self.load_scale,
                wind_scale=self.wind_scale,
                solar_scale=self.solar_scale,
                hydro_scale=self.hydro_scale,
            )
            price_n = market_features["price_n"]
            load_n = market_features["load_n"]

            # Investor (12D): direct trading investor with explicit constraint context.
            price_momentum = self._compute_investor_price_momentum(i)
            realized_volatility = self._compute_investor_realized_volatility(i)
            is_decision_step = self._decision_step_flag(self.t)
            risk_exposure_cap = self._clip01(
                getattr(self, "risk_exposure_cap", getattr(self, "risk_multiplier", 1.0))
            )
            investor_local_drawdown = self._clip01(getattr(self, "_investor_local_drawdown", 0.0))
            inv = self._obs_buf["investor_0"]
            _exposure_for_obs = float(
                self._estimate_current_investor_exposure(getattr(self, "_last_tradeable_capital", None))
            )
            # Step-level PnL return (new obs[9]): normalized to [-1, 1] via return_scale
            _step_pnl_return_raw = float(getattr(self, "last_realized_investor_dnav_return", 0.0))
            _step_pnl_return_scale = float(max(getattr(self.config, "investor_trading_return_scale", 0.003), 1e-6))
            _step_pnl_return = float(np.clip(_step_pnl_return_raw / _step_pnl_return_scale, -1.0, 1.0))
            # Medium-horizon momentum (new obs[10]): 4-hr / 24-step return
            _med_horizon = max(1, int(getattr(self.config, "investment_freq", 6)) * 4)
            _t_prev_med = max(0, i - _med_horizon)
            _price_raw = getattr(self, "_price_raw", None)
            if _price_raw is not None and len(_price_raw) > 1 and i < len(_price_raw):
                _p_now_med = float(_price_raw[i])
                _p_prev_med = float(_price_raw[_t_prev_med])
                _denom_med = max(abs(_p_prev_med), float(max(getattr(self.config, "minimum_price_filter", 10.0), 1.0)))
                _med_momentum = float(np.clip((_p_now_med - _p_prev_med) / _denom_med / _step_pnl_return_scale, -1.0, 1.0))
            else:
                _med_momentum = 0.0
            # Intraday sin/cos (new obs[11]): reuse battery time feature
            _id_sin, _id_cos, _iw_sin, _iw_cos = self._compute_battery_time_features(i)
            ProductionObservationBuilder.build_investor_observations(
                obs_array=inv,
                price_momentum=price_momentum,
                realized_volatility=realized_volatility,
                budget=self.budget,
                init_budget=self.init_budget,
                financial_positions=self.financial_positions,
                max_position_size=self.config.max_position_size,
                capital_allocation_fraction=float(
                    getattr(self, "capital_allocation_fraction", self.config.capital_allocation_fraction)
                ),
                cumulative_mtm_pnl=getattr(self, "cumulative_mtm_pnl", 0.0),
                is_decision_step=is_decision_step,
                current_exposure_norm=_exposure_for_obs,
                risk_exposure_cap=risk_exposure_cap,
                local_drawdown=investor_local_drawdown,
                step_pnl_return=_step_pnl_return,
                medium_momentum=_med_momentum,
                intraday_sin=_id_sin,
            )

            # Forecast-cache prior is an execution prior, not an observation.
            # Update its online calibration from raw DKK prices so the next
            # investor decision can read a current prior without changing the
            # 12D investor observation contract.
            if bool(getattr(self.config, "enable_forecast_utilization", False)):
                self._update_forecast_prior_snapshot(int(self.t), i)

            if self.t % 1000 == 0 or self.t == 0:
                logger.info(
                    f"[INVESTOR_OBS] t={self.t} | mom={inv[0]:.3f}, vol={inv[1]:.3f}, "
                    f"budget={inv[2]:.3f}, exp={inv[3]:.3f}, pnl={inv[4]:.3f}, "
                    f"dec={inv[5]:.1f}, cap={inv[6]:.3f}, rcap={inv[7]:.3f}, "
                    f"dd={inv[8]:.3f}, step_ret={inv[9]:.3f}, med_mom={inv[10]:.3f}, "
                    f"tod_sin={inv[11]:.3f}"
                )

            # Battery (12D): direct arbitrage state with explicit price-edge and time context.
            batt = self._obs_buf["battery_operator_0"]
            soc_norm = self._clip01(
                self.operational_state["battery_energy"] / max(self.physical_assets["battery_capacity_mwh"], 1.0)
            )
            soc_band = max(self.batt_soc_max - self.batt_soc_min, 1e-6)
            charge_headroom = self._clip01((self.batt_soc_max - soc_norm) / soc_band)
            discharge_headroom = self._clip01((soc_norm - self.batt_soc_min) / soc_band)
            intraday_sin, intraday_cos, intraweek_sin, intraweek_cos = self._compute_battery_time_features(i)
            ProductionObservationBuilder.build_battery_observations(
                obs_array=batt,
                price_n=price_n,
                battery_energy=self.operational_state["battery_energy"],
                battery_capacity_mwh=self.physical_assets["battery_capacity_mwh"],
                load_n=load_n,
                charge_headroom=charge_headroom,
                discharge_headroom=discharge_headroom,
                price_reversion_signal=self._compute_battery_price_reversion_signal(i),
                price_momentum=self._compute_battery_price_momentum(i),
                realized_price_volatility=self._compute_battery_realized_price_volatility(i),
                intraday_sin=intraday_sin,
                intraday_cos=intraday_cos,
                intraweek_sin=intraweek_sin,
                intraweek_cos=intraweek_cos,
            )

            if self.t % 1000 == 0 or self.t == 0:
                logger.info(
                    f"[BATTERY_OBS] t={self.t} | price={batt[0]:.3f}, soc={batt[1]:.3f}, "
                    f"chg_room={batt[2]:.3f}, dis_room={batt[3]:.3f}, "
                    f"reversion={batt[4]:.3f}, momentum={batt[5]:.3f}, "
                    f"vol={batt[6]:.3f}, load={batt[7]:.3f}, "
                    f"tod=({batt[8]:.3f},{batt[9]:.3f}), week=({batt[10]:.3f},{batt[11]:.3f})"
                )

            # Risk controller (12D): all values normalized to [-1,1] or [0,1].
            # Feature[11] = is_decision_step flag so the policy knows when its action
            # will be applied (1.0) vs when the last cap simply persists (0.0).
            drawdown = float(getattr(self.reward_calculator, 'current_drawdown', 0.0)) if self.reward_calculator else 0.0
            rsk = self._obs_buf["risk_controller_0"]
            rsk[:12] = (
                price_n,
                float(np.clip(self.market_volatility, 0.0, 1.0)),
                float(np.clip(self.market_stress, 0.0, 1.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["wind_instrument_value"], self.init_budget), -1.0, 1.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["solar_instrument_value"], self.init_budget), -1.0, 1.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["hydro_instrument_value"], self.init_budget), -1.0, 1.0)),
                float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)),
                float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 2.0)) * 0.5,
                float(np.clip(self.risk_multiplier, 0.0, 2.0)) * 0.5,
                float(np.clip(self.overall_risk_snapshot, 0.0, 1.0)),
                float(np.clip(drawdown, 0.0, 1.0)),
                float(self._is_risk_decision_step(self.t)),
            )

            # Meta controller (12D): all values normalized to [0,1] or [-1,1].
            # Feature[11] = is_decision_step flag so the policy knows when its action
            # will be applied (1.0) vs when the last allocation simply persists (0.0).
            perf_ratio = float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 2.0)) * 0.5
            # Normalise trading-sleeve cash against trading_allocation_budget (not total fund)
            # so the obs starts at ~1.0 and declines as capital is deployed.
            # NUMERICAL STABILITY: halve to keep budget_n in [0, 1] (was [0, 2]).
            # An out-of-range feature drives unbounded MLP logits and is the dominant
            # cause of meta-controller value-head drift -> NaN actions in PPO.
            _trading_alloc = max(float(getattr(self, 'trading_allocation_budget', self.init_budget * 0.12)), 1.0)
            meta = self._obs_buf["meta_controller_0"]
            meta[:12] = (
                float(np.clip(SafeDivision.div(self.budget, _trading_alloc, 0.0), 0.0, 2.0)) * 0.5,
                float(np.clip(SafeDivision.div(self.financial_positions["wind_instrument_value"], self.init_budget, 0.0), -1.0, 1.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["solar_instrument_value"], self.init_budget, 0.0), -1.0, 1.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["hydro_instrument_value"], self.init_budget, 0.0), -1.0, 1.0)),
                price_n,
                float(np.clip(self.overall_risk_snapshot, 0.0, 1.0)),
                perf_ratio,
                float(np.clip(self.market_risk_snapshot, 0.0, 1.0)),
                float(np.clip(self.market_volatility, 0.0, 1.0)),
                float(np.clip(self.market_stress, 0.0, 1.0)),
                float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)),
                float(self._is_meta_decision_step(self.t)),
            )

        except Exception as e:
            msg = f"[OBS_BUILD_FATAL] Error building observations at step {self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    # ------------------------------------------------------------------
    def get_risk_level(self) -> float:
        """
        Evaluation helper: expose the current overall risk snapshot.
        """
        try:
            return float(np.clip(getattr(self, "overall_risk_snapshot", ENV_OVERALL_RISK_DEFAULT), 0.0, 1.0))
        except Exception:
            return float(ENV_OVERALL_RISK_DEFAULT)

    def get_progress_snapshot(self) -> Dict[str, float]:
        """
        Training-progress helper for terminal display.

        Exact NAV identity:
            cash + physical_book + operating + mtm = total_nav

        `cash` is shown both as a total and as a display-only split:
            trading_cash_core + battery_cash_contribution = cash

        The battery component is the cumulative realized battery contribution
        that has flowed into the live trading cash bucket.
        """
        try:
            breakdown = self.get_fund_nav_breakdown()
            usd_rate = float(getattr(self, 'usd_conversion_rate', getattr(self.config, 'usd_conversion_rate', 0.145)))
            return {
                'total_nav_usd_m': float(breakdown['fund_nav_dkk'] * usd_rate / 1_000_000.0),
                'cash_usd_m': float(breakdown['trading_cash_dkk'] * usd_rate / 1_000_000.0),
                'trading_cash_core_usd_m': float(breakdown.get('trading_cash_core_dkk', 0.0) * usd_rate / 1_000_000.0),
                'battery_cash_contribution_usd_m': float(breakdown.get('battery_cash_contribution_dkk', 0.0) * usd_rate / 1_000_000.0),
                'trading_sleeve_usd_m': float(breakdown['trading_sleeve_value_dkk'] * usd_rate / 1_000_000.0),
                'physical_book_usd_m': float(breakdown['physical_book_value_dkk'] * usd_rate / 1_000_000.0),
                'operating_revenue_usd_m': float(breakdown['accumulated_operational_revenue_dkk'] * usd_rate / 1_000_000.0),
                'mtm_usd_m': float(breakdown['financial_mtm_dkk'] * usd_rate / 1_000_000.0),
                'trading_mtm_tracker_usd_m': float(breakdown['trading_mtm_tracker_dkk'] * usd_rate / 1_000_000.0),
                'battery_revenue_tracker_usd_m': float(breakdown['battery_revenue_tracker_dkk'] * usd_rate / 1_000_000.0),
            }
        except Exception:
            return {}

    def get_investor_health_summary(self) -> Dict[str, float]:
        """
        Best-effort investor PPO health snapshot for checkpoint/final metadata.
        """
        try:
            history = np.asarray(list(getattr(self, "_investor_local_return_history", []) or []), dtype=np.float64)
        except Exception:
            history = np.asarray([], dtype=np.float64)

        return {
            "mean_clip_hit_rate": float(getattr(self, "_investor_mean_clip_hit_rate", 0.0)),
            "mu_abs_roll": float(getattr(self, "_investor_mean_abs_rolling", 0.0)),
            "mu_sign_consistency": float(getattr(self, "_investor_mean_sign_consistency", 0.0)),
            "policy_action_mean": float(
                np.clip(getattr(self, "_inv_action_mean", getattr(self, "_inv_tanh_mu", 0.0)), -1.0, 1.0)
            ),
            "policy_action_sigma": float(
                max(getattr(self, "_inv_action_sigma", getattr(self, "_inv_sigma_raw", 0.0)), 0.0)
            ),
            "policy_mu_tanh": float(
                np.clip(getattr(self, "_inv_action_mean", getattr(self, "_inv_tanh_mu", 0.0)), -1.0, 1.0)
            ),
            "policy_sigma_raw": float(max(getattr(self, "_inv_sigma_raw", 0.0), 0.0)),
            "last_exposure_exec": float(np.clip(
                (getattr(self, "_last_actions", {}) or {}).get("investor_0_exec", {}).get("exposure", 0.0)
                if isinstance((getattr(self, "_last_actions", {}) or {}).get("investor_0_exec", {}), dict)
                else 0.0,
                -1.0,
                1.0,
            )),
            "local_quality": float(getattr(self, "_investor_local_quality", 0.0)),
            "local_drawdown": float(np.clip(getattr(self, "_investor_local_drawdown", 0.0), 0.0, 1.0)),
            "local_return_mean": float(np.mean(history)) if history.size else 0.0,
            "local_return_std": float(np.std(history)) if history.size else 0.0,
            "risk_snapshot": float(np.clip(getattr(self, "overall_risk_snapshot", ENV_OVERALL_RISK_DEFAULT), 0.0, 1.0)),
        }

    def get_battery_health_summary(self) -> Dict[str, float]:
        """Best-effort battery operator DQN health snapshot for checkpoint/final metadata."""
        return {
            "last_action_idx": int(getattr(self, "_batt_action_idx", 0)),
            "last_q_max": float(getattr(self, "_batt_q_max", 0.0)),
            "last_q_chosen": float(getattr(self, "_batt_q_chosen", 0.0)),
            "last_reward_step": float(getattr(self, "_batt_reward_step", 0.0)),
            "last_soc": float(getattr(self, "_last_battery_soc", 0.0)),
            "last_spread": float(getattr(self, "_last_battery_spread", 0.0)),
            "last_energy": float(getattr(self, "_last_battery_energy",
                                         self.operational_state.get("battery_energy", 0.0))),
            "capacity": float(getattr(self, "_last_battery_capacity",
                                      self.physical_assets.get("battery_capacity_mwh", 0.0))),
        }

    def get_risk_health_summary(self) -> Dict[str, float]:
        """Best-effort risk controller PPO health snapshot for checkpoint/final metadata."""
        return {
            "last_action_raw": float(getattr(self, "_risk_action_raw", 0.0)),
            "last_reward_step": float(getattr(self, "_risk_reward_step", 0.0)),
            "last_value": float(getattr(self, "_risk_value", 0.0)),
            "last_value_next": float(getattr(self, "_risk_value_next", 0.0)),
            "last_td_error": float(getattr(self, "_risk_td_error", 0.0)),
            "last_drawdown": float(np.clip(getattr(self, "current_drawdown", 0.0), 0.0, 1.0)),
            "last_risk_cap": float(np.clip(getattr(self, "risk_exposure_cap", 1.0), 0.0, 1.0)),
            "overall_risk_snapshot": float(np.clip(
                getattr(self, "overall_risk_snapshot", ENV_OVERALL_RISK_DEFAULT), 0.0, 1.0)),
        }

    def get_meta_health_summary(self) -> Dict[str, float]:
        """Best-effort meta controller PPO health snapshot for checkpoint/final metadata."""
        return {
            "last_action_raw": float(getattr(self, "_meta_action_raw", 0.0)),
            "last_reward_step": float(getattr(self, "_meta_reward_step", 0.0)),
            "last_value": float(getattr(self, "_meta_value", 0.0)),
            "last_value_next": float(getattr(self, "_meta_value_next", 0.0)),
            "last_td_error": float(getattr(self, "_meta_td_error", 0.0)),
            "last_cap_fraction": float(np.clip(
                getattr(self, "capital_allocation_fraction", 0.0), 0.0, 1.0)),
            "last_budget_n": float(np.clip(
                getattr(self, "budget", 0.0) / max(float(getattr(self, "trading_allocation_budget", getattr(self, "init_budget", 1.0) * 0.12)), 1.0),
                0.0, 2.0)),
            "overall_risk_snapshot": float(np.clip(
                getattr(self, "overall_risk_snapshot", ENV_OVERALL_RISK_DEFAULT), 0.0, 1.0)),
        }

    def _update_forecast_prior_snapshot(self, step: int, price_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Update/read the conformal forecast-cache exposure prior for ``step``.

        The processor expects raw DKK prices because it calibrates realized
        price returns and volatility. The returned dict is diagnostic state;
        it is not appended to investor observations.
        """
        if not bool(getattr(self.config, "enable_forecast_utilization", False)):
            self._last_forecast_prior = {}
            return {}
        processor = getattr(self, "_forecast_prior_processor", None)
        if processor is None:
            self._last_forecast_prior = {
                "step": int(step),
                "prior_exposure": 0.0,
                "active": False,
                "reason": "processor_missing",
            }
            return dict(self._last_forecast_prior)
        try:
            idx = int(step if price_index is None else price_index)
            price_raw = 0.0
            if hasattr(self, "_price_raw") and 0 <= idx < len(self._price_raw):
                price_raw = float(self._price_raw[idx])
            signal = processor.update_and_compute(int(step), price_raw)
            self._last_forecast_prior = dict(signal)
            return dict(signal)
        except Exception as e:
            logger.warning(f"[FORECAST_PRIOR] Failed to update prior at step={step}: {e}")
            self._last_forecast_prior = {
                "step": int(step),
                "prior_exposure": 0.0,
                "active": False,
                "reason": "update_failed",
            }
            return dict(self._last_forecast_prior)

    def _get_forecast_prior_snapshot(self, step: int) -> Dict[str, Any]:
        prior = getattr(self, "_last_forecast_prior", None)
        if isinstance(prior, dict) and int(prior.get("step", -1)) == int(step):
            return dict(prior)
        return self._update_forecast_prior_snapshot(int(step), int(step))

    def _estimate_current_investor_exposure(self, tradeable_capital: Optional[float] = None) -> float:
        """
        Estimate the current signed aggregate investor exposure in normalized
        Tier-1 units [-1, 1].
        """
        try:
            if tradeable_capital is None:
                available_capital = float(max(self.budget * self.capital_allocation_fraction, 0.0))
                tradeable_capital = float(max(available_capital, 0.0))
            max_pos_size = float(max(tradeable_capital * self.config.max_position_size, 1.0))
            aggregate_position = 0.0
            for v in (getattr(self, "financial_positions", {}) or {}).values():
                fv = float(v)
                if np.isfinite(fv):
                    aggregate_position += fv
            return float(np.clip(aggregate_position / max_pos_size, -1.0, 1.0))
        except Exception:
            return 0.0

    def _map_investor_control_to_exposure(
        self,
        exposure_raw: float,
        tradeable_capital: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Map the policy output to a target normalized exposure.

        The default Tier-1 contract is delta-based: policy output adjusts current
        exposure instead of re-specifying the full target every decision step.
        That is materially healthier for PPO because "hold current risk" maps to
        an interior action near zero.
        """
        exposure_clipped = float(np.clip(exposure_raw, -1.0, 1.0))
        exposure_power = float(max(getattr(self.config, 'investor_exposure_power', 1.0), 1e-6))
        control_signal = float(np.sign(exposure_clipped) * (abs(exposure_clipped) ** exposure_power))
        current_exposure = float(self._estimate_current_investor_exposure(tradeable_capital))
        action_mode = str(getattr(self.config, 'investor_exposure_action_mode', 'delta') or 'delta').strip().lower()
        if action_mode == 'absolute':
            delta_exposure = float(np.clip(control_signal - current_exposure, -1.0, 1.0))
            target_exposure = float(np.clip(control_signal, -1.0, 1.0))
        else:
            delta_scale = float(np.clip(getattr(self.config, 'investor_delta_exposure_scale', 0.35), 0.0, 1.0))
            delta_exposure = float(np.clip(control_signal * delta_scale, -delta_scale, delta_scale))
            target_exposure = float(np.clip(current_exposure + delta_exposure, -1.0, 1.0))
        return current_exposure, control_signal, delta_exposure, target_exposure

    def _map_investor_action_to_exposure(
        self,
        exposure_raw: float,
        tradeable_capital: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> Tuple[float, float, float, float, Dict[str, Any]]:
        """
        Map investor policy output to target exposure.

        Baseline mode uses the standard delta controller. Forecast-prior mode
        treats the policy output as a residual around the conformal ANN-cache
        prior, so a zero policy action means "follow the calibrated prior".
        """
        t = int(getattr(self, "t", 0) if timestep is None else timestep)
        if not bool(getattr(self.config, "enable_forecast_utilization", False)):
            current, control, delta, target = self._map_investor_control_to_exposure(
                exposure_raw,
                tradeable_capital=tradeable_capital,
            )
            return current, control, delta, target, {
                "prior_exposure": 0.0,
                "residual_scale": 0.0,
                "active": False,
                "reason": "disabled",
            }

        exposure_clipped = float(np.clip(exposure_raw, -1.0, 1.0))
        exposure_power = float(max(getattr(self.config, 'investor_exposure_power', 1.0), 1e-6))
        residual_control = float(
            np.sign(exposure_clipped) * (abs(exposure_clipped) ** exposure_power)
        )
        current_exposure = float(self._estimate_current_investor_exposure(tradeable_capital))
        prior = self._get_forecast_prior_snapshot(t)
        prior_exposure = float(np.clip(prior.get("prior_exposure", 0.0), -1.0, 1.0))
        residual_scale = float(np.clip(getattr(self.config, "forecast_prior_residual_scale", 0.10), 0.0, 1.0))
        prior_blend = float(np.clip(getattr(self.config, "forecast_prior_blend", 0.85), 0.0, 1.0))
        prior_target = float(current_exposure + prior_blend * (prior_exposure - current_exposure))
        target_exposure = float(np.clip(prior_target + residual_scale * residual_control, -1.0, 1.0))
        delta_exposure = float(np.clip(target_exposure - current_exposure, -1.0, 1.0))
        diag = dict(prior)
        diag.update(
            {
                "prior_exposure": prior_exposure,
                "prior_blend": prior_blend,
                "prior_target": prior_target,
                "residual_control": residual_control,
                "residual_scale": residual_scale,
                "target_exposure_pre_cap": target_exposure,
            }
        )
        return current_exposure, residual_control, delta_exposure, target_exposure, diag

    # ------------------------------------------------------------------
    # Info & helpers
    # ------------------------------------------------------------------
    def _populate_info(self, i: int, financial: Dict[str, float], acts: Dict[str, np.ndarray]):
        try:
            price_return_1step = float(self._price_return_1step[i]) if hasattr(self, '_price_return_1step') and i < len(self._price_return_1step) else 0.0
            decision_step = self._decision_step_flag(i)
            fallback_horizon_steps = int(max(1, getattr(self, 'investment_freq', 1) or 1))
            decision_anchor_step = int(
                max(
                    0,
                    min(
                        int(getattr(self, '_prev_investor_decision_step', i - fallback_horizon_steps)),
                        max(i - 1, 0),
                    ),
                )
            )
            decision_horizon_steps = int(max(1, i - decision_anchor_step))
            if (
                decision_horizon_steps > 0
                and decision_anchor_step >= 0
                and hasattr(self, '_price_raw')
                and i < len(self._price_raw)
            ):
                prev_price = float(self._price_raw[decision_anchor_step])
                curr_price = float(self._price_raw[i])
                price_return_invfreq = float(
                    np.clip((curr_price - prev_price) / max(abs(prev_price), 1e-6), -0.5, 0.5)
                )
            else:
                price_return_invfreq = 0.0
            action_investor = np.asarray(acts['investor_0']).reshape(-1).tolist()
            action_battery = np.atleast_1d(acts['battery_operator_0']).astype(float).tolist()
            action_risk = np.asarray(acts['risk_controller_0']).reshape(-1).tolist()
            action_meta = np.asarray(acts['meta_controller_0']).reshape(-1).tolist()
            executed_investor_exposure = float(
                np.clip(
                    float(
                        (
                            ((getattr(self, '_last_actions', {}) or {}).get('investor_0_exec', {}) or {})
                            .get('exposure', 0.0)
                        )
                    ),
                    -1.0,
                    1.0,
                )
            )
            for a in self.possible_agents:
                self._info_buf[a] = {
                    # Market data
                    'wind': float(self._wind[i]),
                    'solar': float(self._solar[i]),
                    'hydro': float(self._hydro[i]),
                    'price_dkk': float(self._price_raw[i]) if hasattr(self, '_price_raw') and i < len(self._price_raw) else float(self._price[i] * self._price_std[i] + self._price_mean[i]) if hasattr(self, '_price_std') and hasattr(self, '_price_mean') and i < len(self._price_std) and i < len(self._price_mean) else float(self._price[i]),
                    'price_z': float(self._price[i]),  # Z-score normalized price for clarity
                    'load':  float(self._load[i]),

                    # PHYSICAL ASSETS (Fixed capacities)
                    'wind_capacity_mw': self.physical_assets['wind_capacity_mw'],
                    'solar_capacity_mw': self.physical_assets['solar_capacity_mw'],
                    'hydro_capacity_mw': self.physical_assets['hydro_capacity_mw'],
                    'battery_capacity_mwh': self.physical_assets['battery_capacity_mwh'],

                    # Short keys mirror physical_assets_* (PettingZoo info contract / risk helpers).
                    'wind_capacity': self.physical_assets['wind_capacity_mw'],
                    'solar_capacity': self.physical_assets['solar_capacity_mw'],
                    'hydro_capacity': self.physical_assets['hydro_capacity_mw'],

                    # FINANCIAL INSTRUMENTS (Mark-to-market values)
                    'wind_instrument_value': self.financial_positions['wind_instrument_value'],
                    'solar_instrument_value': self.financial_positions['solar_instrument_value'],
                    'hydro_instrument_value': self.financial_positions['hydro_instrument_value'],

                    # OPERATIONAL STATE
                    'battery_energy': self.operational_state['battery_energy'],
                    'battery_discharge_power': self.operational_state['battery_discharge_power'],

                    # Fund state
                    'budget': self.budget,
                    'initial_budget': self.init_budget,
                    'equity': self.equity,
                    'timestep': i,
                    'step_in_episode': i,

                    # Performance
                    'last_revenue': self.last_revenue,
                    'last_generation_revenue': self.last_generation_revenue,
                    'last_mtm_pnl': self.last_mtm_pnl,

                    # Actions
                    'action_investor': action_investor,
                    'action_battery':  action_battery,
                    'action_risk':     action_risk,
                    'action_meta':     action_meta,

                    # Reward breakdown
                    'reward_breakdown': dict(self.last_reward_breakdown),
                    'reward_weights': dict(getattr(self, 'last_reward_weights', {})),

                    # Financial breakdown
                    'fund_nav': financial.get('fund_nav', self.equity),
                    'total_generation_mwh': financial.get('total_generation_mwh', 0.0),
                    'assets_deployed': self.assets_deployed,
                    'price_return_1step': float(price_return_1step),
                    'price_return_invfreq': float(price_return_invfreq),
                    'decision_step': float(decision_step),

                    # Compact investor state for diagnostics.
                    # Keep both:
                    # - realized_investor_return: executed sleeve PnL return (for diagnostics/reward analysis)
                    # - training_return: exposure-independent market return aligned to the
                    #   investor decision cadence for label generation
                    'realized_investor_return': float(getattr(self, 'last_realized_investor_dnav_return', 0.0)),
                    'training_return': float(
                        price_return_invfreq if np.isfinite(price_return_invfreq) else price_return_1step
                    ),
                    'training_horizon_steps': float(decision_horizon_steps),
                    'executed_investor_exposure': executed_investor_exposure,
                    'forecast_prior_exposure': float((getattr(self, "_last_forecast_prior", {}) or {}).get("prior_exposure", 0.0)),
                    'forecast_prior_skill': float((getattr(self, "_last_forecast_prior", {}) or {}).get("skill", 0.0)),
                    'forecast_prior_hit_lcb': float((getattr(self, "_last_forecast_prior", {}) or {}).get("hit_lcb", 0.5)),
                    'forecast_prior_active': float(1.0 if (getattr(self, "_last_forecast_prior", {}) or {}).get("active", False) else 0.0),
                }
        except Exception as e:
            logger.error(f"info populate: {e}")

    # ------------------------------------------------------------------
    # Terminal / safety
    # ------------------------------------------------------------------
    def _terminal_step(self):
        for a in self.possible_agents:
            self._done_buf[a] = True
            self._trunc_buf[a] = True
            self._rew_buf[a] = 0.0
            self._info_buf[a].clear()
        return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

    def __del__(self):
        try:
            self.memory_manager.cleanup_if_needed(force=True)
        except Exception:
            pass

    # =====================================================================
    # COMPATIBILITY LAYER PROPERTIES (Class level)
    # =====================================================================

    @property
    def wind_capacity_mw(self):
        return self.physical_assets['wind_capacity_mw']

    @wind_capacity_mw.setter
    def wind_capacity_mw(self, value):
        self.physical_assets['wind_capacity_mw'] = float(value)

    @property
    def solar_capacity_mw(self):
        return self.physical_assets['solar_capacity_mw']

    @solar_capacity_mw.setter
    def solar_capacity_mw(self, value):
        self.physical_assets['solar_capacity_mw'] = float(value)

    @property
    def hydro_capacity_mw(self):
        return self.physical_assets['hydro_capacity_mw']

    @hydro_capacity_mw.setter
    def hydro_capacity_mw(self, value):
        self.physical_assets['hydro_capacity_mw'] = float(value)

    @property
    def battery_capacity_mwh(self):
        return self.physical_assets['battery_capacity_mwh']

    @battery_capacity_mwh.setter
    def battery_capacity_mwh(self, value):
        self.physical_assets['battery_capacity_mwh'] = float(value)

    @property
    def wind_instrument_value(self):
        return self.financial_positions['wind_instrument_value']

    @wind_instrument_value.setter
    def wind_instrument_value(self, value):
        self.financial_positions['wind_instrument_value'] = float(value)

    @property
    def solar_instrument_value(self):
        return self.financial_positions['solar_instrument_value']

    @solar_instrument_value.setter
    def solar_instrument_value(self, value):
        self.financial_positions['solar_instrument_value'] = float(value)

    @property
    def hydro_instrument_value(self):
        return self.financial_positions['hydro_instrument_value']

    @hydro_instrument_value.setter
    def hydro_instrument_value(self, value):
        self.financial_positions['hydro_instrument_value'] = float(value)
