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
- Forecasting drives storage/trading timing decisions
- Multi-agent environment supports different investment strategies

This environment simulates realistic renewable energy fund operations with:
- Comprehensive risk management across both asset classes
- Portfolio optimization using deep learning
- Performance tracking and enhanced metrics
- Multi-horizon forecasting integration
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
    TIER2_VALUE_CORE_DIM,
    TIER2_VALUE_MEMORY_CHANNELS,
    TIER2_VALUE_MEMORY_STEPS,
    TIER2_VALUE_FEATURE_DIM,
    ENV_MARKET_STRESS_DEFAULT,
    ENV_OVERALL_RISK_DEFAULT,
    ENV_MARKET_RISK_DEFAULT,
    ENV_POSITION_EXPOSURE_THRESHOLD,
    ENV_EXPLORATION_BONUS_MULTIPLIER,
)  # UNIFIED: Import from single source of truth
from tier2 import (
    TIER2_ACTIVE_EXPERT_NAMES,
    TIER2_PRIMARY_EXPERT_NAME,
    extract_tier2_core_state_from_features,
    extract_tier2_memory_state_from_features,
    resolve_no_trade_threshold_dkk,
    tier2_expert_quality_blend,
    tier2_quality_gain,
    tier2_risk_gain,
    tier2_short_expert_realized_utility_score,
)
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
        # Investor always 9D.  Tier-2 DL→RL feedback flows through the
        # exposure delta only; the DL model's own 11D core features still
        # contain ACRP signals for internal conditioning.
        return {
            "investor_0": {"base": 9},
            "battery_operator_0": {"base": 12},
            "risk_controller_0": {"base": 11},
            "meta_controller_0": {"base": 11},
        }

    def _build_spaces(self) -> Dict[str, spaces.Box]:
        sp: Dict[str, spaces.Box] = {}

        # Investor base (9D): price_momentum, realized_volatility, budget,
        # aggregate_exposure, mtm_pnl_norm, is_decision_step,
        # capital_allocation_fraction, risk_exposure_cap, local_drawdown
        # Investor always 9D (no Tier-2 obs expansion).
        inv_low = np.array([-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        inv_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

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

        # Risk controller (11D): price_n, vol, stress, positions (3), cap_frac, equity, risk_mult,
        # overall_risk_snapshot, drawdown (so policy sees what it's penalized for)
        risk_low = np.array([-1.0, 0.0, 0.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        risk_high = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        # Meta controller (11D): budget, positions (3), price_n, risks (4), perf, cap_frac
        meta_low = np.array([0.0, -10.0, -10.0, -10.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        meta_high = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        sp["investor_0"] = spaces.Box(low=inv_low, high=inv_high, shape=(9,), dtype=np.float32)
        sp["battery_operator_0"] = spaces.Box(low=bat_low, high=bat_high, shape=(12,), dtype=np.float32)
        sp["risk_controller_0"] = spaces.Box(low=risk_low, high=risk_high, shape=(11,), dtype=np.float32)
        sp["meta_controller_0"] = spaces.Box(low=meta_low, high=meta_high, shape=(11,), dtype=np.float32)
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
            # Fair comparison: keep reward identical across tiers.
            # Forecast utilisation should help via better observations, not extra reward shaping.
            forecast_weight = 0.0
            adjusted_nav = max(0.0, base_weights['nav_stability'] - forecast_weight)
            self.reward_weights = {
                **{k: v for k, v in base_weights.items() if k != 'nav_stability'},
                'nav_stability': adjusted_nav,
                'forecast': forecast_weight,
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
            # Fair comparison: keep reward identical across tiers.
            # Forecast utilisation should help via better observations, not extra reward shaping.
            forecast_weight = 0.0
            adjusted_nav = max(0.0, base_weights['nav_stability'] - forecast_weight)
            self.reward_weights = {
                **{k: v for k, v in base_weights.items() if k != 'nav_stability'},
                'nav_stability': adjusted_nav,
                'forecast': forecast_weight,
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

        logger.info("[REWARD] Forecast component: DISABLED (paper setup; no forecast reward shaping)")
        logger.info(f"[REWARD] Forecast weight: {self.reward_weights['forecast']:.3f}")
        logger.info(f"[REWARD] Weights: {self.reward_weights}")

    def update_trading_mtm(self, mtm_pnl: float):
        self.trading_mtm_history.append(mtm_pnl)

    @property
    def recent_trading_mtm(self) -> float:
        if len(self.trading_mtm_history) == 0:
            return 0.0
        return float(np.mean(list(self.trading_mtm_history)))

    def calculate_reward(self, fund_nav: float, cash_flow: float,
                         risk_level: float, efficiency: float,
                         forecast_signal_score: float = 0.0) -> float:
        fund_nav = float(fund_nav) if not isinstance(fund_nav, np.ndarray) else float(fund_nav.item() if fund_nav.size == 1 else fund_nav.flatten()[0])
        cash_flow = float(cash_flow) if not isinstance(cash_flow, np.ndarray) else float(cash_flow.item() if cash_flow.size == 1 else cash_flow.flatten()[0])
        risk_level = float(risk_level) if not isinstance(risk_level, np.ndarray) else float(risk_level.item() if risk_level.size == 1 else risk_level.flatten()[0])
        efficiency = float(efficiency) if not isinstance(efficiency, np.ndarray) else float(efficiency.item() if efficiency.size == 1 else efficiency.flatten()[0])
        forecast_signal_score = float(forecast_signal_score) if not isinstance(forecast_signal_score, np.ndarray) else float(forecast_signal_score.item() if forecast_signal_score.size == 1 else forecast_signal_score.flatten()[0])

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

        forecast_score = float(np.clip(forecast_signal_score * 5.0, -5.0, 5.0))

        reward_weights = getattr(self, 'reward_weights', {})
        base_forecast_weight = float(reward_weights.get('forecast', 0.0))

        # The live paper path no longer adapts the reward-side forecast weight from
        # generic confidence heuristics. Keep the exported fields deterministic so
        # telemetry remains stable without carrying legacy methodology noise.
        combined_confidence = 0.0
        adaptive_forecast_weight = base_forecast_weight
        self._combined_confidence = combined_confidence
        self._adaptive_forecast_weight = adaptive_forecast_weight

        self.last_forecast_score = forecast_score
        self.last_operational_score = operational_score
        self.last_risk_score = risk_management_score
        self.last_hedging_score = hedging_score
        self.last_nav_stability_score = nav_stability_score

        warmup_steps = max(1, int(getattr(self.config, 'forecast_reward_warmup_steps', 1000)))
        forecast_horizons = getattr(self.config, 'forecast_horizons', {}) if self.config else {}
        horizon_short = int(forecast_horizons.get('short', 6))
        current_step = float(getattr(self, 't', 0))
        history_ready = float(np.clip(current_step / max(horizon_short, 1), 0.0, 1.0))
        config_ready = float(np.clip(current_step / warmup_steps, 0.0, 1.0))
        warmup_factor = float(np.clip(0.5 * history_ready + 0.5 * config_ready, 0.0, 1.0))
        self._forecast_warmup_factor = warmup_factor

        rw = self.reward_weights
        reward = float(
            rw['operational_revenue'] * operational_score +
            rw['risk_management'] * risk_management_score +
            rw['hedging_effectiveness'] * hedging_score +
            rw['nav_stability'] * nav_stability_score +
            (adaptive_forecast_weight if hasattr(self, '_adaptive_forecast_weight') else rw.get('forecast', 0.0)) * forecast_score
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
            'forecast': forecast_score,
            'adaptive_forecast_weight': adaptive_forecast_weight,
            'warmup_factor': getattr(self, '_forecast_warmup_factor', 1.0),
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

    3. CLEAR SEPARATION: Physical assets ≠ Financial instruments
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
        forecast_generator: Optional[Any] = None,
        dl_adapter: Optional[Any] = None,
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
        self.forecast_generator = forecast_generator
        self.dl_adapter = dl_adapter
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
        # CRITICAL: Initialize with seed from config to ensure reproducibility
        # If no seed in config, will be seeded in reset()
        env_seed = getattr(config, 'seed', None) if config else None
        if env_seed is not None:
            self._rng = np.random.default_rng(int(env_seed))
            self._last_seed = int(env_seed)
        else:
            self._rng = np.random.default_rng()
            self._last_seed = None

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

        # Legacy compatibility
        self.battery_capacity = 0.0

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
        # Keep baseline naming as "tier1"; Tier-2 is the routed residual-aware overlay path.
        tier_name = "tier1"
        if bool(getattr(self.config, "forecast_baseline_enable", False)):
            ablate_forecast_features = bool(
                getattr(
                    self.config,
                    "tier2_value_ablate_forecast_features",
                    False,
                )
            )
            tier_name = "tier2_ablated" if ablate_forecast_features else "tier2"
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
            prior_std = float(max(getattr(self.config, "rolling_past_price_std_floor", 50.0), 1e-6))
            prior_count = 1.0

            std_floor = float(max(getattr(self.config, "rolling_past_price_std_floor", 50.0), 1e-6))
            price_normalized, price_mean_arr, price_std_arr, price_state = _online_normalize_with_state(
                self._price_raw,
                getattr(self.config, "rolling_past_price_state", None),
                prior_mean=prior_mean,
                prior_std=prior_std,
                prior_count=prior_count,
                std_floor=std_floor,
            )
            price_normalized_clipped = np.clip(price_normalized, -3.0, 3.0)  # ±3 sigma bounds

            # Ensure downstream components (wrapper/CV) see identical [-1, 1] scaling
            self._price = price_normalized

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

        # Pre-allocate forecast array for Tier-2 (short horizon only; immediate/medium/long removed)
        self._price_forecast_short = np.full(self.max_steps, np.nan, dtype=float)

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
                    "EnhancedRiskController rejected lookback_window; retrying legacy constructor signature: %s",
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

        # Forecast accuracy tracking
        from collections import defaultdict, deque as _deque
        # Note: deque is already imported at module level (line 50), so we can use it directly
        self._forecast_errors = defaultdict(lambda: _deque(maxlen=100))
        # CRITICAL FIX: Separate forecast history for different purposes
        # 1. Per-target forecast values for MAPE tracking (dict of deques)
        self._forecast_history_per_target = defaultdict(lambda: _deque(maxlen=200))
        # 2. Price forecast snapshots for horizon-matched calibration (single deque of dicts)
        self._forecast_history = deque(maxlen=200)  # Deque of dicts (for price calibration)
        self._forecast_accuracy_window = 50
        
        # Per-horizon MAPE tracking (Tier-2 short-horizon only; medium/long removed)
        self._horizon_mape = {'short': deque(maxlen=100)}
        self._horizon_sign_hit = {'short': deque(maxlen=200)}
        # Store (timestep, forecast_price, current_price) for delayed MAPE at t+horizon
        self._horizon_forecast_pairs = {'short': deque(maxlen=200)}
        self._horizon_return_pairs = {'short': deque(maxlen=200)}
        self._horizon_correlations = {'short': 0.0}
        self._tier2_short_expert_forecast_pairs = {
            name: deque(maxlen=200) for name in TIER2_ACTIVE_EXPERT_NAMES
        }
        self._tier2_short_expert_mape = {
            name: deque(maxlen=200) for name in TIER2_ACTIVE_EXPERT_NAMES
        }
        self._tier2_short_expert_sign_hit = {
            name: deque(maxlen=200) for name in TIER2_ACTIVE_EXPERT_NAMES
        }
        self._tier2_short_expert_return_pairs = {
            name: deque(maxlen=200) for name in TIER2_ACTIVE_EXPERT_NAMES
        }
        self._tier2_short_expert_economic_skill = {
            name: deque(maxlen=200) for name in TIER2_ACTIVE_EXPERT_NAMES
        }
        
        # OPTION 2: Forward-looking reward alignment - z-score history buffer
        # Stores z-scores with timesteps for delayed reward calculation
        # Format: {timestep: {'z_short': float, 'z_medium': float, 'z_long': float, ...}}
        self._z_score_history = {}  # Dict[int, dict] - stores z-scores by timestep
        self._z_score_history_max_age = 200  # Keep last 200 timesteps (enough for long horizon=144)

        # buffers
        self._obs_buf: Dict[str, np.ndarray] = {a: np.zeros(self.observation_spaces[a].shape, np.float32) for a in self.possible_agents}
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

        # =====================================================================
        # TIER-2 POLICY-IMPROVEMENT STATE
        # =====================================================================
        # IMPORTANT (fairness): forecast backend signals are NOT injected into policy observations.
        # Tier-2: forecast-conditioned short-horizon controller for the investor only.
        # - learned policy-improvement model trained on realized investor outcomes
        # - short-horizon runtime overlay on investor exposure
        self.tier2_value_adapter = None  # Set by main.py when forecast_baseline_enable
        self.tier2_value_trainer = None
        self.tier2_value_buffer = None

        # =====================================================================
        # FGB: FORECAST-GUIDED BASELINE STATE
        # =====================================================================
        self.calibration_tracker = None  # Will be set by main.py for forecast trust tracking
        self._tier2_forecast_horizon = "short"
        self._forecast_base_output = {}  # Cache latest backend forecast metadata for diagnostics
        self._forecast_trust = 0.0  # τₜ: forecast trust score
        self._expected_dnav = 0.0  # Tier-2 expected next-step investor sleeve dNAV under current residual target
        # Tier-2 value diagnostics (for logs/ablations)
        self._last_tier2_value_diag = self._default_tier2_value_diag()
        self._tier2_forecast_snapshot_step = -1
        self._tier2_forecast_snapshot = {}

        # =====================================================================
        # PAPER MODES: Baseline vs Tier-2 short-horizon forecast layer (no forecast-augmented obs)
        # =====================================================================
        # Forecasts (if enabled) are backend-only (CV/trust diagnostics), not policy observations.

        logger.info("=" * 70)
        if bool(getattr(self.config, "forecast_baseline_enable", False)):
            logger.info("TIER-2: short-horizon forecast-conditioned policy-improvement controller (runtime investor overlay)")
        else:
            logger.info("TIER-1 HYBRID RL BASELINE: Tier-1 observations (no forecasts, no Tier-2 layer)")
        logger.info("=" * 70)

        # NOTE (Model B accounting):
        # Financial instruments are treated as a margin-style exposure sleeve:
        # - `financial_positions` are exposures (not equity / not "position value")
        # - NAV impact comes from MTM PnL, tracked separately in `financial_mtm_positions`
        # Therefore we do NOT maintain "open position cost basis" bookkeeping.

        self._last_nav = None  # Track NAV changes for per-step NAV logging

        # =====================================================================
        # TIER-2 VALUE FEATURE STATE (training itself lives in metacontroller)
        # =====================================================================
        self._active_tier2_features = None
        self._active_tier2_features_step = -1
        self._tier2_feature_cache_step = -1
        self._tier2_feature_cache = {}

        # =====================================================================
        # DELTA NORMALIZATION STATE (EMA std tracking for stabilization)
        # =====================================================================
        # CRITICAL FIX: Track EMA of FORECAST DELTA magnitudes, not realized volatility
        # Forecast deltas (forecast - current) are typically 50-200 DKK (forecast error)
        # Realized volatility (current - previous) is only 1-10 DKK (actual movement)
        # We need to normalize forecast deltas by their own distribution, not realized volatility
        # FIX Issue #1: Initialize with placeholder values (will be updated adaptively)
        self.ema_std_short = 100.0   # EMA of |forecast_delta_short| in DKK (initial placeholder)
        self.ema_std_medium = 0.0    # Unused (short-only); kept for compatibility
        self.ema_std_long = 0.0     # Unused (short-only); kept for compatibility
        # CRITICAL FIX: Increased EMA alpha from 0.05 to 0.1 for faster adaptation (Issue #9)
        # Was 0.05 (takes ~20 steps to adapt), now 0.1 (takes ~10 steps to adapt)
        self.ema_alpha = getattr(self.config, 'ema_alpha', 0.1) if self.config else 0.1  # Configurable, default 0.1
        
        # FIX Issue #2: Add EMA std for generation forecasts (wind, solar, hydro)
        self.ema_std_wind = 10.0     # EMA of |forecast_delta_wind| (normalized)
        self.ema_std_solar = 10.0   # EMA of |forecast_delta_solar| (normalized)
        self.ema_std_hydro = 10.0   # EMA of |forecast_delta_hydro| (normalized)
        
        # FIX Issue #1: Track initialization state
        self._ema_std_init_samples = []  # Store first N samples for adaptive initialization
        self._ema_std_init_count = 0    # Count of initialization samples collected
        self._ema_std_initialized = False  # Flag to track if EMA std has been initialized
        
        # FIX Issue #5: Store last good z-scores for forecast failure handling
        self._last_good_z_short_price = 0.0
        self._last_good_z_medium_price = 0.0
        self._last_good_z_long_price = 0.0
        self._last_good_z_short_wind = 0.0
        self._last_good_z_medium_wind = 0.0
        self._last_good_z_long_wind = 0.0
        self._last_good_z_short_solar = 0.0
        self._last_good_z_medium_solar = 0.0
        self._last_good_z_long_solar = 0.0
        self._last_good_z_short_hydro = 0.0
        self._last_good_z_medium_hydro = 0.0
        self._last_good_z_long_hydro = 0.0
        
        # Initialize z-prev values to avoid first-step issues
        self.z_short_price_prev = 0.0
        self.z_medium_price_prev = 0.0
        self.z_long_price_prev = 0.0
        self.z_short_wind_prev = 0.0
        self.z_short_solar_prev = 0.0
        self.z_short_hydro_prev = 0.0

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
                    self.battery_capacity = capacity  # Legacy sync
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
            logger.info("[TIER1_ASSET_DEPLOY] Asset deployment complete:")
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
                logger.info("[TIER1_NAV_CALC] Calculating NAV at t=0:")
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
                logger.info(f"[TIER1_NAV_CALC] NAV calculated: ${(nav * usd_rate) / 1_000_000:.2f}M")
            
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
            usd_rate = float(getattr(self, 'usd_conversion_rate', getattr(self.config, 'usd_conversion_rate', 0.145)))

            return {
                'fund_nav_dkk': float(fund_nav),
                'fund_nav_usd': float(fund_nav * usd_rate),
                'trading_cash_dkk': trading_cash_dkk,
                'trading_cash_core_dkk': trading_cash_core_dkk,
                'battery_cash_contribution_dkk': battery_cash_contribution_dkk,
                'trading_sleeve_value_dkk': trading_sleeve_value_dkk,
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
    # Forecast Accuracy Tracking (normalized, consistent with obs scales)
    # ------------------------------------------------------------------
    def _track_forecast_accuracy(self, forecasts: Dict[str, float]):
        """Track forecast accuracy against realized values (price + short only)."""
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return
        if self.t == 0:
            return
        forecast_key = "price_forecast_short"
        if forecast_key not in forecasts:
            return
        try:
            target = "price"
            actual_normalized = float(getattr(self, f"_{target}")[self.t]) if hasattr(self, f"_{target}") else 0.0
            fv = float(forecasts[forecast_key])
            i = min(self.t, len(self._price_mean) - 1)
            mean = float(self._price_mean[i])
            std = max(float(self._price_std[i]), 1e-6)
            forecast_normalized = (fv - mean) / std
            forecast_normalized = float(np.clip(forecast_normalized, -3.0, 3.0))

            error = abs(actual_normalized - forecast_normalized) / (abs(actual_normalized) + 1e-6)
            self._forecast_errors[target].append(float(np.clip(error, 0.0, 10.0)))
            self._forecast_history_per_target[target].append(float(forecast_normalized))

            if self.t % 500 == 0 and len(self._forecast_errors[target]) >= 10:
                recent_mape = np.mean(list(self._forecast_errors[target])[-10:])
                logger.info(
                    f"[PRICE_ZSPACE_TRACKING] t={self.t} recent_err={recent_mape:.4f} "
                    f"samples={len(self._forecast_errors[target])} "
                    f"actual_z={actual_normalized:.4f} forecast_z={forecast_normalized:.4f}"
                )
        except Exception as e:
            # Log errors for debugging (MAPE tracking failures are critical)
            if self.t % 1000 == 0:
                logger.warning(f"[MAPE_TRACKING] Failed to track forecast accuracy at t={self.t}: {e}")
            pass

    def _get_aligned_price_forecast(self, t: int, default: float = None) -> Optional[float]:
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return default
        try:
            d = self.forecast_generator.predict_all_horizons(timestep=t)
            if isinstance(d, dict):
                val = d.get("price_forecast_short", default)
                if val is not None and np.isfinite(val):
                    return float(val)
        except Exception as e:
            raise RuntimeError(
                f"[ALIGNED_PRICE_FORECAST_FATAL] Failed to get aligned price forecast at step={t}: {e}"
            ) from e
        return default

    def populate_forecast_arrays(self, t: int, forecasts: Dict[str, float]):
        """
        Populate forecast arrays for downstream diagnostics and legacy consumers.

        In the live Tier-2 expert-only setup we persist only the short price forecast and
        keep the other price horizons neutral at the current price. Generation/load forecasts
        are intentionally left untouched because the live method no longer uses them.
        """
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return
        try:
            if 0 <= t < self.max_steps:
                populated_count = 0
                current_price = (
                    float(self._price_raw[t])
                    if hasattr(self, "_price_raw") and t < len(self._price_raw)
                    else 0.0
                )
                short_price = float(forecasts.get("price_forecast_short", current_price))
                if hasattr(self, "_price_forecast_short"):
                    self._price_forecast_short[t] = short_price
                    populated_count += 1

                if t == 0 and populated_count > 0:
                    logger.info(f"[FORECAST_ARRAYS] Step 0: Populated short-price forecast")
                    delta = short_price - current_price
                    logger.info(f"[FORECAST_VALUES] t=0 current_price={current_price:.2f}, short_forecast={short_price:.2f}, delta={delta:.4f}")
        except Exception as e:
            raise RuntimeError(
                f"[FORECAST_ARRAYS_FATAL] Population error at t={t}: {e}"
            ) from e

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

        # DEBUG: Log Tier-2 overlay configuration at reset
        base_feature_dim = BASE_FEATURE_DIM
        tier2_feature_dim = getattr(self, 'tier2_feature_dim', None)
        has_forecaster = getattr(self, 'forecast_generator', None) is not None
        logger.info(
            f"[RESET] Environment reset: base_feature_dim={base_feature_dim}, "
            f"tier2_feature_dim={tier2_feature_dim}, "
            f"has_forecaster={has_forecaster}"
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
            
            # FIX Issue #2: Reset EMA std state for new episode (Option A: reset per episode)
            # Each episode should adapt to its own forecast error distribution
            self.ema_std_short = 100.0   # Reset to initial placeholder
            self.ema_std_medium = 0.0
            self.ema_std_long = 0.0
            self.ema_std_wind = 10.0     # Reset to initial placeholder
            self.ema_std_solar = 10.0    # Reset to initial placeholder
            self.ema_std_hydro = 10.0    # Reset to initial placeholder
            
            # Reset initialization state
            self._ema_std_init_samples = []
            self._ema_std_init_count = 0
            self._ema_std_initialized = False
            if hasattr(self, '_ema_std_init_steps'):
                self._ema_std_init_steps = 0
            
            logger.info(f"[EMA_STD_RESET] Reset EMA std state for new episode")
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
        self._z_score_history.clear()
        
        # Reset momentum tracking for temporal alignment
        if hasattr(self, '_z_medium_prev_obs'):
            delattr(self, '_z_medium_prev_obs')
        
        # Clear per-horizon MAPE and correlation tracking on reset (short only)
        if hasattr(self, '_horizon_mape') and 'short' in self._horizon_mape:
            self._horizon_mape['short'].clear()
            self._horizon_forecast_pairs['short'].clear()
        if hasattr(self, '_horizon_return_pairs') and 'short' in self._horizon_return_pairs:
            self._horizon_return_pairs['short'].clear()
            self._horizon_correlations['short'] = 0.0
        if hasattr(self, '_tier2_short_expert_forecast_pairs'):
            for name in TIER2_ACTIVE_EXPERT_NAMES:
                self._tier2_short_expert_forecast_pairs[name].clear()
                self._tier2_short_expert_mape[name].clear()
                self._tier2_short_expert_sign_hit[name].clear()
                self._tier2_short_expert_return_pairs[name].clear()
                self._tier2_short_expert_economic_skill[name].clear()

        # CRITICAL FIX: Initialize forecast generator history at reset
        # The forecast generator needs initial observations to make predictions
        if self.forecast_generator is not None and hasattr(self.forecast_generator, 'update'):
            try:
                # Feed initial observations to forecast generator
                # Use first few timesteps to build history buffer
                for t_init in range(min(self.forecast_generator.look_back, len(self._price_raw))):
                    init_obs = {
                        'price': float(self._price_raw[t_init]),
                        'wind': float(self._wind[t_init]),
                        'solar': float(self._solar[t_init]),
                        'hydro': float(self._hydro[t_init]),
                        'load': float(self._load[t_init]),
                    }
                    self.forecast_generator.update(init_obs)
                if is_true_episode_end:
                    logger.info(f"[FORECAST] Initialized forecast generator with {min(self.forecast_generator.look_back, len(self._price_raw))} observations")
            except Exception as e:
                logger.warning(f"[FORECAST] Failed to initialize forecast generator: {e}")

        # CRITICAL: Compute forecast deltas before filling observations so the Tier-2 forecast backend is ready from t=0.
        # FIX Issue #3: Initialize z-prev values on first reset
        forecast_backend_enabled = bool(getattr(self.config, 'forecast_baseline_enable', False))
        if forecast_backend_enabled and self.forecast_generator is not None:
            self._compute_forecast_deltas(self.t)
            # After first computation, ensure z-prev values are set
            if not hasattr(self, 'z_short_price_prev'):
                self.z_short_price_prev = getattr(self, 'z_short_price', 0.0)
                self.z_medium_price_prev = getattr(self, 'z_medium_price', 0.0)
                self.z_long_price_prev = getattr(self, 'z_long_price', 0.0)
                self.z_short_wind_prev = getattr(self, 'z_short_wind', 0.0)
                self.z_short_solar_prev = getattr(self, 'z_short_solar', 0.0)
                self.z_short_hydro_prev = getattr(self, 'z_short_hydro', 0.0)

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
        self._last_tier2_value_diag = self._default_tier2_value_diag()
        self._tier2_forecast_horizon = "short"
        self._tier2_forecast_snapshot_step = -1
        self._tier2_forecast_snapshot = {}
        self._tier2_forecast_memory = deque(
            maxlen=max(1, int(getattr(self.config, "tier2_value_memory_steps", TIER2_VALUE_MEMORY_STEPS)))
        )
        self._tier2_forecast_memory_step = -1
        self._tier2_forecast_memory_entry = None
        self._active_tier2_features = None
        self._active_tier2_features_step = -1
        self._tier2_feature_cache_step = -1
        self._tier2_feature_cache = {}
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
        logger.info(f"💰 FULL FINANCIAL RESET: Resetting all gains to initial state")

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
        self._last_tier2_value_diag = self._default_tier2_value_diag()
        self._tier2_forecast_horizon = "short"
        self._tier2_forecast_snapshot_step = -1
        self._tier2_forecast_snapshot = {}
        self._active_tier2_features = None
        self._active_tier2_features_step = -1
        self._tier2_feature_cache_step = -1
        self._tier2_feature_cache = {}
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
        # This ensures Tier 1 and Tier 2 have identical physical assets
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
        logger.info("[TIER1_RESET] Physical assets recalculated:")
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
                 logger.warning(f"[TIER1_RESET] Initial NAV components: Cash={trading_cash:,.0f}, "
                               f"Assets={physical_book_value:,.0f}, OpRev={operational_revenue_value:,.0f}, MTM={financial_mtm_value:,.0f}")
            logger.info(f"[TIER1_RESET] Initial NAV after full reset: {initial_nav:,.0f} DKK (${initial_nav * self.config.dkk_to_usd_rate / 1e6:.4f}M USD), init_budget: {self.init_budget:,.0f} DKK")

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

                        # ALSO compute executed allocation immediately (exposure-only mode).
                        # Ensures logs reflect actual trades and penalties use executed exposure.
                        try:
                            exposure_raw_exec = float(np.clip(arr[0] if arr.size > 0 else 1.0, -1.0, 1.0))
                            current_exposure_exec, control_signal_exec, delta_exposure_exec, exposure = (
                                self._map_investor_control_to_exposure(exposure_raw_exec)
                            )

                            if bool(getattr(self.config, "investor_use_risk_budget_weights", True)):
                                w = np.asarray(self._get_risk_budget_allocation(), dtype=np.float32)
                            else:
                                w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                            w = (w / float(np.sum(np.abs(w)))).astype(np.float32)
                            alloc = (exposure * w).astype(np.float32)

                            self._last_actions["investor_0_exec"] = {
                                "wind": float(alloc[0]),
                                "solar": float(alloc[1]),
                                "hydro": float(alloc[2]),
                                "exposure": float(exposure),
                                "current_exposure": float(current_exposure_exec),
                                "delta_exposure": float(delta_exposure_exec),
                                "control_signal": float(control_signal_exec),
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

            # ------------------------------------------------------------------
            # Forecast backend signals are computed from PRE-ACTION state.
            # ------------------------------------------------------------------
            forecast_backend_enabled = bool(getattr(self.config, 'forecast_baseline_enable', False))

            # Avoid stale forecast-base predictions if inference fails on this step.
            if forecast_backend_enabled:
                self._forecast_base_output = {}
                self._expected_dnav = 0.0

            # Compute z-scores/deltas for CURRENT timestep i (used for calibration/trust, logging, etc.).
            if forecast_backend_enabled:
                self._compute_forecast_deltas(i, update_calibration=True)

            # Forecast-base output is produced implicitly by the Tier-2 forecast-fusion path.

            # meta & risk knobs
            self._apply_risk_control(acts['risk_controller_0'])
            self._apply_meta_control(acts['meta_controller_0'])

            # ------------------------------------------------------------------
            # CRITICAL: Add MTM BEFORE trades - price move (i-1→i) applies to PRE-trade positions.
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

            # Track forecast accuracy if forecaster is available
            # CRITICAL FIX: Use i (captured timestep) instead of self.t for consistency
            if self.forecast_generator and hasattr(self.forecast_generator, 'predict_all_horizons'):
                try:
                    current_forecasts = self.forecast_generator.predict_all_horizons(timestep=i)
                    if isinstance(current_forecasts, dict):
                        self._track_forecast_accuracy(current_forecasts)
                except Exception as e:
                    raise RuntimeError(
                        f"[FORECAST_ACCURACY_FATAL] Forecast accuracy tracking failed at step {self.t}: {e}"
                    ) from e

            # === ENHANCEMENT 6: DIAGNOSTIC LOGGING ===
            # Log Tier-2 feature state for monitoring
            if i % 500 == 0 and i > 0 and bool(getattr(self.config, 'forecast_baseline_enable', False)):
                mwdir = getattr(self, 'mwdir', 0.0)
                tier2_horizon = str(getattr(self, "_tier2_forecast_horizon", self._resolve_tier2_forecast_horizon()) or "short")
                ema_short = getattr(self, 'ema_std_short', 0.01)
                ema_medium = getattr(self, 'ema_std_medium', 0.01)
                ema_long = getattr(self, 'ema_std_long', 0.01)
                conf = self._get_tier2_expert_confidence()
                logger.info(
                    f"[CV_BASE] step={i} h={tier2_horizon} mwdir={mwdir:.3f} "
                    f"conf={conf:.2f} ema_stds=[{ema_short:.4f}, {ema_medium:.4f}, {ema_long:.4f}]"
                )

            # Tier-2: policy-improvement layer only (no legacy forecast-augmented experience collection).

            # FIXED: assign rewards
            self._assign_rewards(financial)

            # step forward
            self.t += 1
            self.step_in_episode = self.t

            index = max(0, self.t - 1)
            # Update forecast generator with the just-realized observation first.
            # This keeps next-step forecast deltas synchronized with latest state.
            if self.forecast_generator is not None and hasattr(self.forecast_generator, 'update'):
                try:
                    current_obs = {
                        'price': float(self._price_raw[i]) if i < len(self._price_raw) else 250.0,
                        'wind': float(self._wind[i]) if i < len(self._wind) else 0.0,
                        'solar': float(self._solar[i]) if i < len(self._solar) else 0.0,
                        'hydro': float(self._hydro[i]) if i < len(self._hydro) else 0.0,
                        'load': float(self._load[i]) if i < len(self._load) else 0.0,
                    }
                    self.forecast_generator.update(current_obs)
                except Exception as e:
                    raise RuntimeError(
                        f"[FORECAST_UPDATE_FATAL] Failed to update forecast generator at t={i}: {e}"
                    ) from e

            # ------------------------------------------------------------------
            # Keep forecast backend state aligned to the next timestep (no double-counting).
            # ------------------------------------------------------------------
            # After incrementing self.t, the next observation is for timestep `self.t`.
            # Recompute forecast deltas for that timestep WITHOUT updating calibration/trust.
            if forecast_backend_enabled and self.t < self.max_steps:
                try:
                    self._compute_forecast_deltas(self.t, update_calibration=False)
                except Exception as e:
                    raise RuntimeError(
                        f"[FORECAST_DELTAS_FATAL] Next-step update failed at step {self.t}: {e}"
                    ) from e
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
        Tier-1 observations, info/debug logs, Tier-2 sample collection, and
        cadence-gated controllers should all derive from this helper.
        """
        current_t = int(getattr(self, "t", 0) if t is None else t)
        investment_freq = int(max(getattr(self, "investment_freq", 1) or 1, 1))
        return bool(current_t > 0 and current_t % investment_freq == 0)

    def _decision_step_flag(self, t: Optional[int] = None) -> float:
        """Float version of ``_is_investor_decision_step`` for obs/info/logging."""
        return 1.0 if self._is_investor_decision_step(t) else 0.0

    def _apply_meta_control(self, meta_action: np.ndarray):
        """
        REFACTORED: Apply meta control using TradingEngine.
        Meta is a capital-allocation controller; trade cadence is fixed by config.
        """
        try:
            if bool(getattr(self.config, "meta_controller_rule_based", False)):
                update_on_decision_step = bool(
                    getattr(self.config, "meta_rule_update_only_on_investor_step", True)
                )
                is_decision_step = self._is_investor_decision_step()
                if update_on_decision_step and not is_decision_step:
                    return

            from trading_engine import TradingEngine
            proposed_capital_fraction, proposed_investment_freq = TradingEngine.apply_meta_control(
                meta_action=meta_action,
                meta_cap_min=self.META_CAP_MIN,
                meta_cap_max=self.META_CAP_MAX,
                meta_freq_min=self.META_FREQ_MIN,
                meta_freq_max=self.META_FREQ_MAX,
                # Keep meta-control dynamics identical across variants (no confidence-based scaling).
                forecast_confidence=0.5,
                disable_confidence_scaling=True,
            )
            if bool(getattr(self.config, "meta_controller_rule_based", False)):
                prev_target = float(
                    np.clip(
                        getattr(self, "_meta_rule_target_capital_fraction", self.capital_allocation_fraction),
                        self.META_CAP_MIN,
                        self.META_CAP_MAX,
                    )
                )
                smoothing_alpha = float(
                    np.clip(getattr(self.config, "meta_rule_target_smoothing_alpha", 0.20), 0.0, 1.0)
                )
                deadband = float(max(getattr(self.config, "meta_rule_cap_deadband", 0.03), 0.0))
                new_target = float(
                    np.clip(
                        prev_target + smoothing_alpha * (float(proposed_capital_fraction) - prev_target),
                        self.META_CAP_MIN,
                        self.META_CAP_MAX,
                    )
                )
                if abs(new_target - prev_target) < deadband:
                    new_target = prev_target
                self._meta_rule_target_capital_fraction = new_target
                self.capital_allocation_fraction = new_target
                self.investment_freq = int(proposed_investment_freq)
                self._meta_rule_last_update_step = int(getattr(self, "t", 0))
            else:
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

            # Cap intensity to reasonable bounds - WIDENED FROM [0.5, 2.0] TO [0.3, 2.5] FOR OVERLAY LEARNING
            return float(np.clip(portfolio_intensity, 0.3, 2.5))

        except Exception as e:
            raise RuntimeError(
                f"[PORTFOLIO_HEDGE_INTENSITY_FATAL] Failed at step {getattr(self, 't', 'N/A')}: {e}"
            ) from e

    # def set_wrapper_reference(self, wrapper_env):
    #     """CRITICAL: Set reference to wrapper for profit-seeking expert"""
    #     self._wrapper_ref = wrapper_env

    # Optional policy encoders may learn richer feature relationships (Tier-1 observation space).
    # Forecast models (ANN/LSTM/SVR/RF) are used only for the Tier-2 forecast backend signals (no extra observations).
    # NOT rule-based expert suggestions. Expert suggestions interfered with PPO learning and are deprecated

    # ------------------------------------------------------------------
    # FIXED: Financial Instrument Trading (Separate from Physical Assets)
    # ------------------------------------------------------------------

    def _execute_investor_trades(self, inv_action: np.ndarray, timestep: Optional[int] = None) -> float:
        """
        CORRECTED & FINAL: Executes trades based DIRECTLY on the RL agent's action.

        The agent's action vector [-1, 1] is mapped to a target allocation
        of the available trading capital. The RL agent is always in control.

        In Tier-2, the investor still owns the exposure decision; the residual
        layer only nudges that exposure when short-horizon forecast quality is strong.

        Returns total traded notional for transaction costs
        """
        # CRITICAL FIX: Use timestep parameter if provided, otherwise use self.t
        t = timestep if timestep is not None else getattr(self, 't', 0)
        
        # Trading is only allowed at the specified frequency
        # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
        # CRITICAL FIX: Prevent trading at timestep 0 to ensure identical initial NAV
        # At timestep 0, we want to log the reset NAV before any trades are executed
        if t == 0:
            self._last_tier2_value_diag = self._default_tier2_value_diag()
            return 0.0  # No trading at timestep 0 - ensures identical initial NAV
        if not self._is_investor_decision_step(t):
            self._last_tier2_value_diag = self._default_tier2_value_diag()
            return 0.0

        # Clean sizing contract: hard drawdown disables the sleeve by forcing it
        # flat, not by silently skipping trades while keeping stale exposure.
        force_flat_exposure = bool(not self.reward_calculator.trading_enabled)

        try:
            prev_decision_anchor = int(
                getattr(
                    self,
                    "_tier2_last_investor_decision_step",
                    max(0, t - int(max(getattr(self, "investment_freq", 1) or 1, 1))),
                )
            )
            self._tier2_prev_investor_decision_step = int(max(0, min(prev_decision_anchor, t)))
            self._tier2_last_investor_decision_step = int(t)

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
            current_exposure, control_signal, delta_exposure, exposure = self._map_investor_control_to_exposure(
                exposure_raw,
                tradeable_capital=tradeable_capital,
            )
            # Tier-2: the forecast-backed DL layer applies a learned residual delta
            # on top of the Tier-1 investor exposure.
            exposure_pre_tier2 = float(exposure)
            if bool(getattr(self.config, "forecast_baseline_enable", False)):
                exposure = float(self._apply_tier2_exposure_adjustment(exposure, t, tradeable_capital=tradeable_capital))
            else:
                # Keep Tier-1 architecturally pure: do not route baseline trades
                # through the Tier-2 runtime overlay path when forecasting is off.
                self._last_tier2_value_diag = self._default_tier2_value_diag()
                self._expected_dnav = 0.0
            if force_flat_exposure:
                exposure = 0.0
            else:
                exposure = float(np.clip(exposure, -risk_exposure_cap, risk_exposure_cap))
            tier2_value_diag = dict(getattr(self, "_last_tier2_value_diag", {}) or {})
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
                    f"exposure_pre_tier2={exposure_pre_tier2:.4f} exposure_final={exposure:.4f} "
                    f"risk_cap={risk_exposure_cap:.4f} forced_flat={int(force_flat_exposure)} "
                    f"tier2_delta={float(tier2_value_diag.get('delta', 0.0)):.3f} "
                    f"tier2_rel={float(tier2_value_diag.get('reliability', 1.0)):.3f} "
                    f"context={float(tier2_value_diag.get('context_strength', 0.0)):.3f}"
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

            threshold_dkk = float(
                resolve_no_trade_threshold_dkk(
                    getattr(self.config, "no_trade_threshold", 0.0),
                    max_position_notional=max_pos_size,
                )
            )
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
        Normalize price for active Tier-2 features using z-score with clipped bounds.
        Aligns with wrapper normalization so the live Tier-2 path sees consistent inputs.
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
            # Clip to ±3σ and scale to [-1, 1]
            z_clipped = np.clip(z, -3.0, 3.0) / 3.0
            return float(z_clipped)
        except Exception as e:
            raise RuntimeError(
                f"[PRICE_NORM_FATAL] _norm_price failed at step={t} with price={p}: {e}"
            ) from e

    def _compute_price_short_only_forecast_deltas(
        self,
        i: int,
        forecasts: Dict[str, float],
        update_calibration: bool = True,
    ) -> None:
        """Live Tier-2 forecast backend: short-horizon price only."""
        price_raw = float(self._price_raw[i] if i < len(self._price_raw) else 250.0)
        price_short_raw = float(forecasts.get("price_forecast_short", price_raw))
        horizon_short = int(self.config.forecast_horizons.get("short", 6))

        self._horizon_forecast_pairs["short"].append({
            "timestep": i,
            "forecast_price": price_short_raw,
            "current_price": price_raw,
        })
        for expert_name in TIER2_ACTIVE_EXPERT_NAMES:
            expert_price = float(forecasts.get(f"price_short_expert_{expert_name}", price_short_raw))
            self._tier2_short_expert_forecast_pairs[expert_name].append({
                "timestep": i,
                "forecast_price": expert_price,
                "current_price": price_raw,
            })

        delta_price_short_raw = float(price_short_raw - price_raw)
        init_samples = int(getattr(self.config, "ema_std_init_samples", 20) or 20)
        if not self._ema_std_initialized and self._ema_std_init_count < init_samples:
            self._ema_std_init_samples.append({
                "short": abs(delta_price_short_raw),
                "medium": 0.0,
                "long": 0.0,
                "wind": 0.0,
                "solar": 0.0,
                "hydro": 0.0,
            })
            self._ema_std_init_count += 1
            if self._ema_std_init_count >= init_samples:
                short_values = [s["short"] for s in self._ema_std_init_samples]
                self.ema_std_short = float(np.median(short_values)) if short_values else 0.05
                self.ema_std_medium = 0.0
                self.ema_std_long = 0.0
                self.ema_std_wind = 0.0
                self.ema_std_solar = 0.0
                self.ema_std_hydro = 0.0
                self._ema_std_init_alpha = float(getattr(self.config, "ema_std_init_alpha", 0.2) or 0.2)
                self._ema_std_init_steps = 100
                self._ema_std_initialized = True

        if self._ema_std_initialized and hasattr(self, "_ema_std_init_steps") and self._ema_std_init_steps > 0:
            current_alpha = self._ema_std_init_alpha
            self._ema_std_init_steps -= 1
        else:
            current_alpha = self.ema_alpha

        denom_floor = float(getattr(self.config, "forecast_return_denom_floor", 0.25) or 0.25)
        denom = max(abs(price_raw), denom_floor, 1.0)
        forecast_return_short = float(delta_price_short_raw / denom)
        ret_clip = float(getattr(self.config, "forecast_return_clip", 0.15) or 0.15)
        if ret_clip > 0:
            forecast_return_short = float(np.clip(forecast_return_short, -ret_clip, ret_clip))
        tanh_scale = float(getattr(self.config, "forecast_return_tanh_scale", 1.5) or 1.5)
        z_short = float(np.tanh(forecast_return_short * tanh_scale))

        self.ema_std_short = float(
            np.clip(
                (1.0 - current_alpha) * float(getattr(self, "ema_std_short", 0.05)) + current_alpha * abs(forecast_return_short),
                float(getattr(self.config, "ema_std_min_return", 0.01) or 0.01),
                float(getattr(self.config, "ema_std_max_return", 1.0) or 1.0),
            )
        )

        if not hasattr(self, "z_short_price"):
            self.z_short_price_prev = z_short
        else:
            self.z_short_price_prev = float(getattr(self, "z_short_price", 0.0))
        self.z_medium_price_prev = 0.0
        self.z_long_price_prev = 0.0
        self.z_short_wind_prev = 0.0
        self.z_short_solar_prev = 0.0
        self.z_short_hydro_prev = 0.0

        self.z_short_price = float(np.clip(z_short, -1.0, 1.0))
        self.z_medium_price = 0.0
        self.z_long_price = 0.0
        self._last_good_z_short_price = self.z_short_price
        self._last_good_z_medium_price = 0.0
        self._last_good_z_long_price = 0.0

        self.z_short_wind = 0.0
        self.z_medium_wind = 0.0
        self.z_long_wind = 0.0
        self.z_short_solar = 0.0
        self.z_medium_solar = 0.0
        self.z_long_solar = 0.0
        self.z_short_hydro = 0.0
        self.z_medium_hydro = 0.0
        self.z_long_hydro = 0.0
        self._last_good_z_short_wind = 0.0
        self._last_good_z_medium_wind = 0.0
        self._last_good_z_long_wind = 0.0
        self._last_good_z_short_solar = 0.0
        self._last_good_z_medium_solar = 0.0
        self._last_good_z_long_solar = 0.0
        self._last_good_z_short_hydro = 0.0
        self._last_good_z_medium_hydro = 0.0
        self._last_good_z_long_hydro = 0.0

        self._z_score_history[i] = {
            "z_short": float(z_short),
            "z_medium": 0.0,
            "z_long": 0.0,
            "z_short_wind": 0.0,
            "z_short_solar": 0.0,
            "z_short_hydro": 0.0,
            "forecast_trust": float(getattr(self, "_forecast_trust", 0.5)),
        }
        if len(self._z_score_history) > self._z_score_history_max_age:
            sorted_keys = sorted(self._z_score_history.keys())
            for key in sorted_keys[:-self._z_score_history_max_age]:
                del self._z_score_history[key]

        self.z_short = self.z_short_price
        self.z_medium = 0.0
        self.z_long = 0.0
        self._tier2_forecast_horizon = "short"
        self.direction_consistency = float(np.sign(z_short) if abs(z_short) > 1e-12 else 0.0)
        self.mwdir = float(np.clip(z_short, -1.0, 1.0))
        self.abs_mwdir = float(np.clip(abs(z_short), 0.0, 1.0))

        if update_calibration:
            self._ema_std_short_for_calibration = self.ema_std_short
            self._ema_std_medium_for_calibration = 0.0
            self._ema_std_long_for_calibration = 0.0
            self._update_calibration_tracker(i)

            metric = str(getattr(self.config, "forecast_trust_metric", "hitrate") or "hitrate").lower()

            def _mean_last_short_mape(n: int = 10) -> float:
                xs = list(getattr(self, "_horizon_mape", {}).get("short", []))[-n:]
                xs = [float(x) for x in xs if np.isfinite(x)]
                if xs:
                    return float(np.mean(xs))
                return float(getattr(self, "_mape_thresholds", {}).get("short", 0.12))

            def _mean_last_short_hit(n: int = 50) -> float:
                xs = list(getattr(self, "_horizon_sign_hit", {}).get("short", []))[-n:]
                xs = [float(x) for x in xs if np.isfinite(x)]
                return float(np.mean(xs)) if xs else 0.5

            if metric in ("hitrate", "absdir", "sign", "direction"):
                trust_target = float(np.clip(_mean_last_short_hit(50), 0.0, 1.0))
            else:
                weighted_mape = float(_mean_last_short_mape(10))
                if weighted_mape < 0.05:
                    trust_target = 1.0
                elif weighted_mape < 0.10:
                    trust_target = 0.8
                elif weighted_mape < 0.15:
                    trust_target = 0.6
                elif weighted_mape < 0.20:
                    trust_target = 0.4
                else:
                    trust_target = 0.2

            prev = float(getattr(self, "_forecast_trust", 0.5))
            self._forecast_trust = float(np.clip(0.7 * trust_target + 0.3 * prev, 0.0, 1.0))
            try:
                self._forecast_trust = float(self.get_fgb_trust_for_agent("investor_0"))
            except Exception:
                pass

            trust_floor = float(getattr(self.config, "forecast_trust_floor", 0.0) or 0.0)
            if trust_floor > 0.0 and self._should_apply_forecast_trust_floor("investor_0"):
                self._forecast_trust = float(np.maximum(self._forecast_trust, trust_floor))

        self._forecast_deltas_raw = {
            "short": float(forecast_return_short),
        }
        self._current_price_floor = float(max(abs(price_raw), denom_floor, 1.0))
        self._mape_thresholds = {
            "short": float(getattr(self.config, "forecast_mape_threshold_short", 0.08) or 0.08),
        }

        if update_calibration and i % 200 == 0:
            logger.info(
                "[FORECAST_RETURNS] t=%s price_raw=%.2f return_short=%.6f (%.2f%%) abs_short=%.2f",
                i,
                price_raw,
                forecast_return_short,
                forecast_return_short * 100.0,
                delta_price_short_raw,
            )
        if i == 0:
            logger.info("[FORECAST_DELTAS] t=%s price_raw=%.2f", i, price_raw)
            logger.info("  price_short_raw=%.2f (delta=%.4f, z=%.4f)", price_short_raw, delta_price_short_raw, z_short)
            logger.info("  EMA std short=%.4f", self.ema_std_short)
            logger.info("  Forecast trust=%.4f", self._forecast_trust)

    def _compute_forecast_deltas(self, i: int, update_calibration: bool = True) -> None:
        """
        Compute forecast deltas/z-scores for the Tier-2 forecast backend.

        IMPORTANT (fairness): these are NOT appended to agent observations in the paper setup.
        They feed only the live short-price expert Tier-2 feature builder and calibration/trust tracking.

        Computes and stores:
        - Price z-scores: z_short_price, z_medium_price, z_long_price
        - Neutral generation z-scores in expert-only mode
        - Forecast trust: _forecast_trust
        """
        try:
            if i == 0:
                logger.info(f"[FORECAST_DELTAS] _compute_forecast_deltas called at t={i}")
                logger.info(f"  forecast_generator: {self.forecast_generator}")

            fail_fast = bool(getattr(self.config, "fgb_fail_fast", False)) and bool(
                getattr(self.config, "forecast_baseline_enable", False)
            )

            if self.forecast_generator is None:
                msg = f"[FORECAST_DELTAS] forecast_generator is None at step {i}"
                if fail_fast:
                    raise RuntimeError(msg)
                if i == 0:
                    logger.warning(msg)
                return

            # CRITICAL FIX: Predict for CURRENT timestep (i), not next (i+1)
            # Agent acts at timestep i, so it needs forecasts FROM timestep i
            # The forecast model internally adds horizon offsets (i+6, i+24, i+144)
            forecasts = self.forecast_generator.predict_all_horizons(timestep=i)
            if i == 0:
                logger.info(f"[FORECAST_DELTAS] Got forecasts: {list(forecasts.keys()) if forecasts else 'None'}")
            if not forecasts:
                msg = f"[FORECAST_DELTAS] No forecasts returned at step {i}"
                if fail_fast:
                    raise RuntimeError(msg)
                if i == 0:
                    logger.warning(msg)
                return

            expert_only_mode = bool(getattr(self.config, "forecast_price_short_expert_only", False))

            # Populate per-target forecast arrays for the active forecast-feature pipeline.
            # This is the single source of truth for forecast arrays (no wrapper required).
            self.populate_forecast_arrays(i, forecasts)
            if not expert_only_mode:
                raise RuntimeError(
                    "Legacy multi-target forecast delta backend has been retired. "
                    "The live Tier-2 design requires forecast_price_short_expert_only=True."
                )
            self._compute_price_short_only_forecast_deltas(i, forecasts, update_calibration=update_calibration)
            return

        except Exception as e:
            if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(
                getattr(self.config, "forecast_baseline_enable", False)
            ):
                raise RuntimeError(f"[FORECAST_DELTAS] Failed to compute forecast deltas at step {i}: {e}") from e
            # FIX Issue #5: On forecast failure, use previous z-scores with decay instead of zero
            logger.warning(f"[FORECAST_DELTAS] Failed to compute forecast deltas at step {i}: {e}")
            logger.warning(f"[FORECAST_DELTAS] Using previous z-scores with decay (fallback mode)")
            
            # Get decay factor from config
            decay_factor = getattr(self.config, 'forecast_failure_decay', 0.95)
            
            # Use last good z-scores with exponential decay
            if hasattr(self, '_last_good_z_short_price'):
                self.z_short_price = float(np.clip(self._last_good_z_short_price * decay_factor, -1.0, 1.0))
                self.z_medium_price = float(np.clip(self._last_good_z_medium_price * decay_factor, -1.0, 1.0))
                self.z_long_price = float(np.clip(self._last_good_z_long_price * decay_factor, -1.0, 1.0))
            else:
                # Fallback to zero if no previous values
                self.z_short_price = 0.0
                self.z_medium_price = 0.0
                self.z_long_price = 0.0
            
            if hasattr(self, '_last_good_z_short_wind'):
                self.z_short_wind = float(np.clip(self._last_good_z_short_wind * decay_factor, -1.0, 1.0))
                self.z_medium_wind = float(np.clip(self._last_good_z_medium_wind * decay_factor, -1.0, 1.0))
                self.z_long_wind = float(np.clip(self._last_good_z_long_wind * decay_factor, -1.0, 1.0))
                self.z_short_solar = float(np.clip(self._last_good_z_short_solar * decay_factor, -1.0, 1.0))
                self.z_medium_solar = float(np.clip(self._last_good_z_medium_solar * decay_factor, -1.0, 1.0))
                self.z_long_solar = float(np.clip(self._last_good_z_long_solar * decay_factor, -1.0, 1.0))
                self.z_short_hydro = float(np.clip(self._last_good_z_short_hydro * decay_factor, -1.0, 1.0))
                self.z_medium_hydro = float(np.clip(self._last_good_z_medium_hydro * decay_factor, -1.0, 1.0))
                self.z_long_hydro = float(np.clip(self._last_good_z_long_hydro * decay_factor, -1.0, 1.0))
            else:
                # Fallback to zero if no previous values
                self.z_short_wind = 0.0
                self.z_medium_wind = 0.0
                self.z_long_wind = 0.0
                self.z_short_solar = 0.0
                self.z_medium_solar = 0.0
                self.z_long_solar = 0.0
                self.z_short_hydro = 0.0
                self.z_medium_hydro = 0.0
                self.z_long_hydro = 0.0
            
            self._forecast_trust = 0.5
    def _update_calibration_tracker(self, i: int):
        """
        Update CalibrationTracker with forecast/realized pairs to compute trust.

        CRITICAL: This must be called independently of any feature-builder helper so
        Tier-2 forecast calibration stays aligned with the active forecast-backend path.

        Args:
            i: Current timestep index
        """
        # CRITICAL FIX: Horizon-matched calibration
        #
        # IMPORTANT: Our Tier2 forecast z-scores are RETURN-based (bias-immune):
        #   forecast_return = (price_forecast - price_now) / max(|price_now|, denom_floor)
        #   z_short = tanh(clip(forecast_return, ±ret_clip) * tanh_scale)
        #
        # Therefore the realized signal must be computed in the *same space*:
        #   realized_return = (price_realized - price_now) / max(|price_now|, denom_floor)
        #   realized_signal = tanh(clip(realized_return, ±ret_clip) * tanh_scale)
        #
        # Using delta/ema_std here would be a unit mismatch (ema_std is tracking return magnitudes),
        # which saturates realized_signal to ±1 and makes trust meaningless.
        # IMPORTANT:
        # - Horizon-MAPE tracking must run for Tier2 as well (even when calibration_tracker is None).
        # - calibration_tracker (FGB) is Tier3-only and should not gate basic forecast quality stats.
        if hasattr(self, '_price_raw') and i >= 0:
            try:
                # Return-based z-score parameters (must match _compute_forecast_deltas)
                denom_floor = float(getattr(self.config, 'forecast_return_denom_floor', 0.25) or 0.25)
                ret_clip = float(getattr(self.config, 'forecast_return_clip', 0.15) or 0.15)
                tanh_scale = float(getattr(self.config, 'forecast_return_tanh_scale', 1.5) or 1.5)

                price_now_raw = float(self._price_raw[i])
                return_denom_now = float(max(abs(price_now_raw), denom_floor, 1.0))

                # CRITICAL FIX: Store ema_std along with forecast for consistent normalization
                self._forecast_history.append({
                    't': i,
                    'price_raw': price_now_raw,
                    'z_short': self.z_short,  # NOW using current step's value!
                    'z_medium': self.z_medium,
                    'z_long': self.z_long,
                    'mwdir': self.mwdir,
                    # Return-space calibration metadata
                    'return_denom': return_denom_now,
                    'ret_clip': ret_clip,
                    'tanh_scale': tanh_scale,
                })

                # Deque automatically keeps only last 200 steps (no manual trimming needed)

                # OPTIMIZATION: Calculate per-horizon MAPE for adaptive weighting (PRICE-RELATIVE)
                # Compare stored forecasts with actual prices at matching horizons.
                #
                # We use price-relative MAPE with a denominator floor to avoid explosion at low prices:
                #   mape = |actual - forecast| / max(|price_at_forecast_time|, floor)
                horizon_short = int(self.config.forecast_horizons.get('short', 6))
                # Online expert-skill updates must stay on the same horizon as
                # the routed short-price expert forecasts. The routed decision
                # horizon is configured to match this short horizon; if it is
                # overridden inconsistently, keep online expert scoring pinned
                # to the expert forecast horizon rather than mixing horizons.
                tier2_decision_horizon = int(
                    getattr(self.config, "tier2_value_decision_horizon", horizon_short) or horizon_short
                )
                if tier2_decision_horizon != horizon_short:
                    tier2_decision_horizon = horizon_short
                current_price_raw = float(self._price_raw[i])
                
                # Calculate MAPE for short horizon
                # CRITICAL FIX: Forecast at step i-horizon_short predicts price at timestep i
                # So we compare forecast from i-horizon_short with actual at i (matching horizon)
                mape_denom_floor = float(getattr(self.config, 'minimum_price_floor', 50.0) or 50.0)
                signal_denom_floor = float(getattr(self.config, "forecast_return_denom_floor", 0.25) or 0.25)
                return_clip = float(getattr(self.config, "forecast_return_clip", 0.15) or 0.15)
                if i >= horizon_short and len(self._horizon_forecast_pairs['short']) > 0:
                    # Find forecast made at i-horizon_short (matches corrected horizon logic)
                    for pair in reversed(self._horizon_forecast_pairs['short']):
                        if pair['timestep'] == i - horizon_short:  # = i - 6 for horizon=6
                            forecast_price = pair['forecast_price']
                            try:
                                fp = float(forecast_price)
                                pt = float(pair.get('current_price', 0.0))
                                if np.isfinite(fp) and np.isfinite(pt) and abs(fp) > 1e-6:
                                    denom = max(abs(pt), mape_denom_floor, 1.0)
                                    mape_short = abs(current_price_raw - fp) / denom
                                    self._horizon_mape['short'].append(float(np.clip(mape_short, 0.0, 1.0)))
                            except Exception:
                                pass
                            break

                # Track forecast returns and actual returns for correlation-based weighting
                horizon_specs = [('short', horizon_short)]
                for horizon_name, horizon_steps in horizon_specs:
                    if i >= horizon_steps:
                        # Find forecast made at i - horizon_steps
                        for pair in reversed(self._horizon_forecast_pairs[horizon_name]):
                            if pair['timestep'] == i - horizon_steps:
                                forecast_price = pair['forecast_price']
                                price_at_forecast_time = pair['current_price']
                                
                                # Calculate forecast return: (forecast_price - price_at_forecast_time) / price_at_forecast_time
                                if abs(price_at_forecast_time) > 1e-6:
                                    forecast_return = (forecast_price - price_at_forecast_time) / price_at_forecast_time
                                    
                                    # Calculate actual return: (current_price - price_at_forecast_time) / price_at_forecast_time
                                    actual_return = (current_price_raw - price_at_forecast_time) / price_at_forecast_time

                                    # Directional hit-rate (sign accuracy): did we predict up/down correctly?
                                    try:
                                        f_sign = float(np.sign(forecast_return))
                                        a_sign = float(np.sign(actual_return))
                                        if abs(a_sign) > 1e-12:
                                            self._horizon_sign_hit[horizon_name].append(1.0 if f_sign == a_sign else 0.0)
                                    except Exception:
                                        pass
                                    
                                    # Store return pair for correlation computation
                                    self._horizon_return_pairs[horizon_name].append({
                                        'forecast_return': float(forecast_return),
                                        'actual_return': float(actual_return)
                                    })
                                    
                                    # Compute correlation when we have enough samples
                                    if len(self._horizon_return_pairs[horizon_name]) >= 20:
                                        forecast_returns = np.array([p['forecast_return'] for p in self._horizon_return_pairs[horizon_name]])
                                        actual_returns = np.array([p['actual_return'] for p in self._horizon_return_pairs[horizon_name]])
                                        
                                        if len(forecast_returns) > 1 and np.std(forecast_returns) > 1e-6 and np.std(actual_returns) > 1e-6:
                                            corr = np.corrcoef(forecast_returns, actual_returns)[0, 1]
                                            # CRITICAL FIX: Check for NaN/infinity before storing
                                            if not np.isnan(corr) and not np.isinf(corr):
                                                self._horizon_correlations[horizon_name] = float(np.clip(corr, -1.0, 1.0))
                                                # Diagnostic logging for correlation updates
                                                if i % 1000 == 0:
                                                    logger.info(f"[CORRELATION_UPDATE] t={i} | {horizon_name} correlation={corr:.4f} | "
                                                                f"samples={len(self._horizon_return_pairs[horizon_name])} | "
                                                                f"forecast_std={np.std(forecast_returns):.6f} | actual_std={np.std(actual_returns):.6f}")
                                            else:
                                                # Keep previous correlation if calculation fails
                                                if horizon_name not in self._horizon_correlations:
                                                    self._horizon_correlations[horizon_name] = 0.0
                                                if i % 1000 == 0:
                                                    logger.warning(f"[CORRELATION_UPDATE] t={i} | {horizon_name} correlation is NaN/inf, keeping previous value")
                                
                                break

                if i >= tier2_decision_horizon:
                    for expert_name in TIER2_ACTIVE_EXPERT_NAMES:
                        try:
                            expert_pairs = self._tier2_short_expert_forecast_pairs.get(expert_name, deque())
                            for pair in reversed(expert_pairs):
                                if pair['timestep'] != i - tier2_decision_horizon:
                                    continue
                                forecast_price = float(pair['forecast_price'])
                                price_at_forecast_time = float(pair.get('current_price', 0.0))
                                if not (np.isfinite(forecast_price) and np.isfinite(price_at_forecast_time)):
                                    break
                                denom = max(abs(price_at_forecast_time), mape_denom_floor, 1.0)
                                mape_short = abs(current_price_raw - forecast_price) / denom
                                self._tier2_short_expert_mape[expert_name].append(float(np.clip(mape_short, 0.0, 1.0)))
                                signal_denom = max(abs(price_at_forecast_time), signal_denom_floor, 1.0)
                                if signal_denom > 1e-6:
                                    forecast_return = (forecast_price - price_at_forecast_time) / signal_denom
                                    actual_return = (current_price_raw - price_at_forecast_time) / signal_denom
                                    if return_clip > 0:
                                        forecast_return = float(np.clip(forecast_return, -return_clip, return_clip))
                                        actual_return = float(np.clip(actual_return, -return_clip, return_clip))
                                    a_sign = float(np.sign(actual_return))
                                    if abs(a_sign) > 1e-12:
                                        f_sign = float(np.sign(forecast_return))
                                        self._tier2_short_expert_sign_hit[expert_name].append(1.0 if f_sign == a_sign else 0.0)
                                    self._tier2_short_expert_return_pairs[expert_name].append({
                                        'forecast_return': float(forecast_return),
                                        'actual_return': float(actual_return),
                                    })
                                    utility_ref = float(max(getattr(self.config, "tier2_value_target_scale", 0.01) or 0.01, 1e-4))
                                    downside_weight = float(
                                        getattr(self.config, "tier2_value_decision_downside_weight", 0.75) or 0.75
                                    )
                                    delta_limit = float(
                                        getattr(self.config, "tier2_value_delta_max", 0.20) or 0.20
                                    )
                                    runtime_gain = float(
                                        getattr(self.config, "tier2_runtime_gain", 1.0) or 1.0
                                    )
                                    economic_skill = float(
                                        tier2_short_expert_realized_utility_score(
                                            forecast_return=float(forecast_return),
                                            actual_return=float(actual_return),
                                            utility_scale=utility_ref,
                                            delta_limit=delta_limit,
                                            runtime_gain=runtime_gain,
                                            downside_weight=downside_weight,
                                        )
                                    )
                                    self._tier2_short_expert_economic_skill[expert_name].append(economic_skill)
                                break
                        except Exception:
                            continue

                # Update calibration tracker with SHORT HORIZON (6 steps ahead) (Tier3/FGB only)
                # At time t, we have:
                # - forecast_signal = z_short[t] = tanh((price_forecast_short[t] - price[t]) / ema_std[t])
                # - This predicts the price change from t to t+6
                # At time t+6, we can compute:
                # - realized_signal = tanh((price[t+6] - price[t]) / ema_std[t])  <-- SAME ema_std!
                # - This is the actual price change from t to t+6
                if self.calibration_tracker is not None and len(self._forecast_history) >= 7:  # Need at least 7 entries (t-6 to t)
                    forecast_entry = self._forecast_history[-7]  # Entry at time t-6
                    forecast_signal = forecast_entry['z_short']  # Forecast made at t-6 for price change from t-6 to t

                    # CRITICAL FIX: Compute realized price change from t-6 to t (now)
                    # This matches the forecast horizon (6 steps)
                    current_price_raw = float(self._price_raw[i])
                    price_at_forecast_time = forecast_entry['price_raw']  # Price at t-6
                    realized_delta_raw = current_price_raw - price_at_forecast_time

                    # CRITICAL FIX: Compute realized signal in the SAME return-space as forecast z-scores
                    denom_at_forecast_time = float(
                        forecast_entry.get(
                            'return_denom',
                            max(abs(price_at_forecast_time), signal_denom_floor, 1.0),
                        )
                    )
                    realized_return = realized_delta_raw / max(denom_at_forecast_time, 1e-6)
                    if ret_clip > 0:
                        realized_return = float(np.clip(realized_return, -ret_clip, ret_clip))
                    realized_signal = float(np.tanh(realized_return * tanh_scale))

                    # Update calibration tracker
                    self.calibration_tracker.update(
                        forecast=float(forecast_signal),
                        realized=float(realized_signal),
                        horizon="short"
                    )

                    # Log trust updates periodically with DETAILED price info
                    if i % 1000 == 0:
                        current_trust = self.calibration_tracker.get_trust(horizon="short")
                        buffer_size = len(self.calibration_tracker.forecast_history)
                        trust_window = int(getattr(self.config, "forecast_trust_window", 2016)) if self.config is not None else 2016

                        logger.info(
                            f"[CALIBRATION] t={i} | trust={current_trust:.3f} | buffer_size={buffer_size}/{trust_window} | "
                            f"forecast={forecast_signal:.3f} | realized={realized_signal:.3f} | "
                            f"delta_raw={realized_delta_raw:.2f} DKK | denom={denom_at_forecast_time:.2f}"
                        )
                        logger.info(f"[CALIBRATION_DEBUG] t={i} | price_now={current_price_raw:.2f} DKK | price_at_t-6={price_at_forecast_time:.2f} DKK | realized_delta={realized_delta_raw:.2f} DKK | forecast_t={forecast_entry['t']}")

                        # CRITICAL DEBUG: Check what the forecast models actually predict
                        # Compare z_short (stored in forecast history) with what the model predicts NOW for the same timestep
                        try:
                            # Get the forecast that was made 6 steps ago (at t-6)
                            # This forecast predicted the price change from t-6 to t (now)
                            # Let's see what the model predicts NOW for that same period
                            if hasattr(self, 'forecast_generator') and self.forecast_generator is not None:
                                # Get forecast for t-6 (the timestep where the forecast was made)
                                forecast_t_minus_6 = forecast_entry['t']

                                # Check if we have precomputed forecasts
                                if hasattr(self, '_price_forecast_short') and len(self._price_forecast_short) > forecast_t_minus_6:
                                    # Get the forecasted price at t (6 steps ahead from t-6)
                                    price_forecast_at_t = float(self._price_forecast_short[forecast_t_minus_6])
                                    price_at_t_minus_6 = forecast_entry['price_raw']
                                    delta_forecast_raw = price_forecast_at_t - price_at_t_minus_6
                                    # Recompute z_short using the SAME return-based mapping as _compute_forecast_deltas
                                    denom_for_check = float(
                                        forecast_entry.get(
                                            'return_denom',
                                            max(abs(price_at_t_minus_6), signal_denom_floor, 1.0),
                                        )
                                    )
                                    forecast_return_check = float(delta_forecast_raw / max(denom_for_check, 1e-6))
                                    if ret_clip > 0:
                                        forecast_return_check = float(np.clip(forecast_return_check, -ret_clip, ret_clip))
                                    z_short_recomputed = float(np.tanh(forecast_return_check * tanh_scale))

                                    logger.info(f"[CALIBRATION_FORECAST_CHECK] t={i} | forecast_made_at_t={forecast_t_minus_6}")
                                    logger.info(f"[CALIBRATION_FORECAST_CHECK] price_at_t-6={price_at_t_minus_6:.2f} DKK | price_forecast_at_t={price_forecast_at_t:.2f} DKK | delta_forecast={delta_forecast_raw:.2f} DKK")
                                    logger.info(f"[CALIBRATION_FORECAST_CHECK] z_short_stored={forecast_signal:.3f} | z_short_recomputed={z_short_recomputed:.3f} | MATCH={abs(forecast_signal - z_short_recomputed) < 0.01}")
                                else:
                                    logger.warning(f"[CALIBRATION_FORECAST_CHECK] No precomputed forecasts available (has_attr={hasattr(self, '_price_forecast_short')}, len={len(getattr(self, '_price_forecast_short', []))})")
                            else:
                                logger.warning(f"[CALIBRATION_FORECAST_CHECK] No forecast_generator available")
                        except Exception as e:
                            logger.warning(f"[CALIBRATION_FORECAST_CHECK] Failed: {e}")
                            import traceback
                            logger.warning(f"[CALIBRATION_FORECAST_CHECK] Traceback: {traceback.format_exc()}")

            except Exception as e:
                if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(
                    getattr(self.config, "forecast_baseline_enable", False)
                ):
                    raise RuntimeError(f"[CALIBRATION] Update failed at t={i}: {e}") from e
                if i % 1000 == 0:
                    logger.warning(f"[CALIBRATION] Update failed at t={i}: {e}")
                import traceback
                if i % 1000 == 0:
                    logger.warning(f"[CALIBRATION] Traceback: {traceback.format_exc()}")

    def _get_tier2_expert_confidence(self) -> float:
        """
        Method-consistent short-price expert confidence in [0, 1].

        This is now a diagnostic/logging helper for the live Tier-2 design,
        not a generic multi-asset forecast-confidence aggregator.
        """
        try:
            expert_only_mode = bool(getattr(self.config, "forecast_price_short_expert_only", False))
            if not expert_only_mode:
                return 0.5

            margin = float(max(getattr(self.config, "tier2_runtime_conformal_margin", 0.05) or 0.05, 0.0))
            conformal_scale = float(max(getattr(self.config, "tier2_value_conformal_scale", 1.0) or 1.0, 0.0))
            primary_name = str(getattr(self.config, "tier2_primary_expert", TIER2_PRIMARY_EXPERT_NAME) or TIER2_PRIMARY_EXPERT_NAME).strip().lower()
            if primary_name not in TIER2_ACTIVE_EXPERT_NAMES:
                primary_name = TIER2_PRIMARY_EXPERT_NAME
            primary_quality = float(self._get_tier2_short_expert_quality(primary_name))
            primary_risk = float(self._get_tier2_short_expert_conformal_risk(primary_name))
            other_qualities = [
                float(self._get_tier2_short_expert_quality(expert_name))
                for expert_name in TIER2_ACTIVE_EXPERT_NAMES
                if expert_name != primary_name
            ]
            other_risks = [
                float(self._get_tier2_short_expert_conformal_risk(expert_name))
                for expert_name in TIER2_ACTIVE_EXPERT_NAMES
                if expert_name != primary_name
            ]
            next_best_quality = float(max(other_qualities)) if other_qualities else primary_quality
            other_risk_mean = float(np.mean(other_risks)) if other_risks else primary_risk
            quality_gap = float(np.clip(primary_quality - next_best_quality, -1.0, 1.0))
            risk_gap = float(np.clip(other_risk_mean - primary_risk, -1.0, 1.0))
            leadership = float(
                np.clip(
                    0.45 * primary_quality
                    + 0.20 * (1.0 - primary_risk)
                    + 0.20 * max(quality_gap, 0.0)
                    + 0.15 * max(risk_gap, 0.0),
                    0.0,
                    1.0,
                )
            )
            score = float(
                np.clip(
                    tier2_quality_gain(primary_quality)
                    * tier2_risk_gain(
                        primary_risk,
                        conformal_risk_scale=conformal_scale,
                        runtime_conformal_margin=margin,
                    )
                    * (0.50 + 0.50 * leadership),
                    0.0,
                    1.0,
                )
            )
            if np.isfinite(score):
                return score
            return 0.5
        except Exception:
            return 0.5

    def _execute_battery_ops(self, bat_action: np.ndarray, i: int) -> float:
        """
        REFACTORED: Execute battery operations using TradingEngine.
        Returns battery cash delta for proper reward accounting.
        """
        try:
            from trading_engine import TradingEngine
            
            # Get forecast price for dispatch policy
            forecast_price = self._get_aligned_price_forecast(i, default=None)
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
        Compute MTM for price move (i-1→i) using current financial_positions (PRE-trade).
        Accumulates to financial_mtm_positions. Call BEFORE _execute_investor_trades.
        """
        from financial_engine import FinancialEngine
        horizon_correlations = getattr(self, '_horizon_correlations', None)
        price_returns = FinancialEngine.calculate_price_returns(
            timestep=i, current_price=current_price, price_history=self._price_raw,
            enable_forecast_util=False, horizon_correlations=horizon_correlations
        )
        price_return = price_returns['price_return']
        self._correlation_debug = price_returns['correlation_debug']
        self._latest_price_returns = {
            'short': price_returns['price_return_short'],
            'medium': price_returns['price_return_medium'],
            'long': price_returns['price_return_long'],
            'forecast': price_returns['price_return_forecast'],
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
        Add MTM for the price move (i-1→i) using PRE-trade positions.
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
                logger.debug(f"[TIER1_UPDATE_FINANCE] Calculating NAV in _update_finance at i={i}:")
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
                logger.debug(f"[TIER1_UPDATE_FINANCE] NAV calculated at i={i}: {fund_nav:,.0f} DKK (${fund_nav * 0.145 / 1_000_000:.2f}M)")
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

            # 11) Calculate forecast signal if available
            # CRITICAL FIX: Use z-scores from PREVIOUS timestep (when agent made decision)
            #
            # RL Timing:
            # - At timestep t-1: Agent observes z-scores and takes action
            # - At timestep t: Environment transitions, we compute reward
            # - Reward should use z-scores from t-1 (what agent saw) to predict return from t-1 to t
            #
            # Logic:
            # - z_short_price_prev > 0 means agent saw forecast predicting price RISE
            # - price_return > 0 means price actually ROSE from t-1 to t
            # - If both positive or both negative → correct prediction → positive reward
            # - If signs mismatch → incorrect prediction → negative reward
            #
            # This aligns reward with observations (agents rewarded for z-scores they actually saw)
            forecast_signal_score = 0.0
            forecast_backend_enabled = bool(getattr(self.config, "forecast_baseline_enable", False))
            tau_now = float(getattr(self, "_forecast_trust", 0.0))
            exp_dnav_now = float(getattr(self, "_expected_dnav", 0.0))
            trust_ok = bool(np.isfinite(tau_now) and tau_now > 0.0)

            # Forecast diagnostics:
            # "forecast_used" should reflect active short-horizon signal usage in Tier-2.
            short_signal = float((getattr(self, "_forecast_deltas_raw", {}) or {}).get("short", 0.0))
            signal_ok = bool(np.isfinite(short_signal) and abs(short_signal) > 1e-12)
            tier2_value_diag = dict(getattr(self, "_last_tier2_value_diag", {}) or {})
            tier2_gate_active = bool(float(tier2_value_diag.get("enabled", 0.0)) > 0.5)
            tier2_value_gate_ok = bool(float(tier2_value_diag.get("value_gate", 0.0)) > 0.5)
            tier2_nav_gate_ok = bool(float(tier2_value_diag.get("nav_gate", 0.0)) > 0.5)
            tier2_return_floor_gate_ok = bool(float(tier2_value_diag.get("return_floor_gate", 0.0)) > 0.5)
            tier2_delta_gate_ok = bool(float(tier2_value_diag.get("delta_gate", 0.0)) > 0.5)
            tier2_learned_gate_ok = bool(float(tier2_value_diag.get("learned_gate_pass", 0.0)) > 0.5)
            tier2_trust_gate_ok = bool(float(tier2_value_diag.get("trust_gate", 1.0)) > 0.5)

            # Keep forecast availability diagnostics separate from whether Tier-2
            # ultimately chose to act. Otherwise abstention makes the backend look
            # artificially "inactive" in post-run analysis.
            try:
                decision_step_now = bool(self._decision_step_flag(getattr(self, "t", 0)))
            except Exception:
                decision_step_now = False
            forecast_gate_passed = bool(forecast_backend_enabled and trust_ok)
            forecast_used_flag = bool(forecast_gate_passed and signal_ok and tier2_gate_active)
            forecast_direction = float(
                tier2_value_diag.get("forecast_sign", np.sign(short_signal) if signal_ok else 0.0)
            )
            position_direction = float(
                tier2_value_diag.get("action_sign", 0.0)
            )
            agent_followed_forecast = bool(
                forecast_used_flag
                and abs(forecast_direction) > 1e-9
                and abs(position_direction) > 1e-9
                and np.sign(forecast_direction) == np.sign(position_direction)
            )
            self._last_obs_trade_signal = float(short_signal if signal_ok else 0.0)

            if not forecast_backend_enabled:
                forecast_usage_reason = "fgb_disabled"
            elif not trust_ok:
                forecast_usage_reason = "trust_zero_or_invalid"
            elif forecast_used_flag:
                forecast_usage_reason = "used"
            elif not decision_step_now:
                forecast_usage_reason = "tier2_not_decision_step"
            elif not tier2_gate_active:
                if not tier2_learned_gate_ok:
                    forecast_usage_reason = "tier2_learned_gate_fail"
                elif not tier2_delta_gate_ok:
                    forecast_usage_reason = "tier2_delta_gate_fail"
                else:
                    forecast_usage_reason = "tier2_gate_inactive"
            elif not signal_ok:
                forecast_usage_reason = "signal_zero"
            else:
                forecast_usage_reason = "forecast_available_no_overlay"

            def _mean_hit(horizon_name: str, n: int = 100) -> float:
                try:
                    xs = list((getattr(self, "_horizon_sign_hit", {}) or {}).get(horizon_name, []))[-n:]
                    xs = [float(x) for x in xs if np.isfinite(x)]
                    return float(np.mean(xs)) if len(xs) > 0 else 0.0
                except Exception:
                    return 0.0

            self._debug_fgb_backend = {
                "forecast_used": bool(forecast_used_flag),
                "forecast_not_used_reason": forecast_usage_reason,
                "forecast_gate_passed": bool(forecast_gate_passed),
                "forecast_decision_step": bool(decision_step_now),
                "forecast_trust": float(tau_now if np.isfinite(tau_now) else 0.0),
                "expected_dnav": float(exp_dnav_now if np.isfinite(exp_dnav_now) else 0.0),
                "forecast_signal_short": float(short_signal if np.isfinite(short_signal) else 0.0),
                "forecast_signal_active": bool(signal_ok),
                "forecast_direction": float(forecast_direction),
                "position_direction": float(position_direction),
                "agent_followed_forecast": bool(agent_followed_forecast),
                "tier2_value_gate": float(tier2_value_gate_ok),
                "tier2_nav_gate": float(tier2_nav_gate_ok),
                "tier2_return_floor_gate": float(tier2_return_floor_gate_ok),
                "tier2_delta_gate": float(tier2_delta_gate_ok),
                "direction_accuracy_short": float(_mean_hit("short", 100)),
                # Tier-2 now uses a short-horizon-only contract. Keep the legacy
                # columns for logger compatibility, but hard-zero them to avoid
                # implying that medium/long horizons are active in the method.
                "direction_accuracy_medium": 0.0,
                "direction_accuracy_long": 0.0,
                "forecast_usage_bonus": 0.0,
            }

            # Forecast reward shaping remains disabled in the paper setup.
            # These debug flags represent backend signal availability only.

            # =====================================================================
            # AGENT-SPECIFIC REWARDS
            # =====================================================================
            # Calculate agent-specific rewards based on forecast_signal_score
            # (computed above in lines 5112-5335)

            # Continue to agent reward calculation below...

            # =====================================================================
            # ACTUAL AGENT REWARD CALCULATION STARTS HERE
            # =====================================================================
            # The forecast_signal_score computed above (lines 5112-5335) is used
            # in the agent reward calculation below


            # =====================================================================
            # ACTUAL REWARD CALCULATION (STARTS HERE)
            # =====================================================================
            # The forecast_signal_score is already computed above (lines 5112-5335)
            # Now we calculate the final reward using the reward calculator

            # Use the forecast_signal_score already computed above (lines 5112-5335)
            # No additional processing needed - the score is ready to use
            combined_forecast_score = forecast_signal_score

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
                'forecast_signal_score': combined_forecast_score,  # CRITICAL FIX: Use combined score (price + generation)
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

    # ------------------------------------------------------------------
    # FIXED: Reward System with Proper Separation
    # ------------------------------------------------------------------

    def _assign_rewards(self, financial: Dict[str, float]):
        """FIXED: Reward assignment with proper separation of value sources"""
        try:
            # CRITICAL FIX: Use fund_nav from financial dict (already calculated with correct timestep in _update_finance)
            # This ensures consistent NAV values between Tier 1 and Tier 2
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

            # Get forecast signal score - ensure float
            forecast_signal_score = float(financial.get('forecast_signal_score', 0.0))

            # Calculate reward using FIXED calculator
            if self.reward_calculator is None:
                logger.error(f"Reward calculator is None at step {self.current_step}. Initializing now...")
                # Emergency initialization
                post_capex_nav = self._calculate_fund_nav()
                self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
                logger.info(f"Emergency reward calculator initialized with NAV: {post_capex_nav:,.0f} DKK")
                self.reward_weights = dict(getattr(self.reward_calculator, 'reward_weights', {}))

            reward = self.reward_calculator.calculate_reward(
                fund_nav=fund_nav,
                cash_flow=cash_flow,
                risk_level=risk_level,
                efficiency=efficiency,
                forecast_signal_score=forecast_signal_score
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
            debug_forecast = getattr(self, '_debug_fgb_backend', {})
            
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
            # Battery is a direct RL arbitrage agent: reward realized battery cashflow.
            battery_cash = float(financial.get('battery_cash_delta', 0.0))
            battery_reward_weight = float(getattr(self.config, 'battery_arbitrage_reward_weight', 1.00))
            battery_reward_scale = float(max(getattr(self.config, 'battery_arbitrage_reward_scale_dkk', 250.0), 1e-6))
            battery_reward_clip = float(max(getattr(self.config, 'battery_arbitrage_reward_clip', 5.0), 1e-6))
            battery_arbitrage_delta = float(
                battery_reward_weight
                * np.clip(battery_cash / battery_reward_scale, -battery_reward_clip, battery_reward_clip)
            )
            battery_pnl_delta = 0.0
            soc_penalty = 0.0
            battery_forecast_delta = 0.0

            agent_rewards['battery_operator_0'] = battery_base + battery_arbitrage_delta
            
            # ===== RISK CONTROLLER AGENT =====
            volatility_penalty = float(np.clip(risk_level * 10.0, 0.0, 2.0))
            drawdown = float(getattr(self.reward_calculator, 'current_drawdown', 0.0)) if self.reward_calculator else 0.0
            drawdown_penalty = float(np.clip(drawdown * 10.0, 0.0, 2.0))
            risk_management_delta = float(-0.05 * (volatility_penalty + drawdown_penalty))  # REDUCED: was -0.20
            risk_rule_based = bool(getattr(self.config, 'risk_controller_rule_based', False))
            agent_rewards['risk_controller_0'] = 0.0 if risk_rule_based else (risk_base + risk_management_delta)
            
            # ===== META CONTROLLER AGENT =====
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
                + meta_battery_weight * float(battery_arbitrage_delta)
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
            
            meta_rule_based = bool(getattr(self.config, 'meta_controller_rule_based', False))
            agent_rewards['meta_controller_0'] = 0.0 if meta_rule_based else (
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
                'battery_arbitrage_delta': battery_arbitrage_delta,
                'battery_forecast_delta': battery_forecast_delta,
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
                'forecast_signal_score': forecast_signal_score,
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
                forecast_gate_flag = bool(debug_forecast.get('forecast_gate_passed', False))
                forecast_used_flag = bool(debug_forecast.get('forecast_used', False))
                
                # Get reward component scores from calculator (ensure they're floats, not arrays)
                operational_score = float(getattr(self.reward_calculator, 'last_operational_score', 0.0))
                risk_score = float(getattr(self.reward_calculator, 'last_risk_score', 0.0))
                hedging_score = float(getattr(self.reward_calculator, 'last_hedging_score', 0.0))
                nav_stability_score = float(getattr(self.reward_calculator, 'last_nav_stability_score', 0.0))
                forecast_score = float(getattr(self.reward_calculator, 'last_forecast_score', 0.0))
                
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
                info_investor['price_return_1step'] = float(price_return)
            except Exception:
                info_investor['price_return_1step'] = 0.0
            # Horizon-aligned return over investment_freq steps (decision cadence)
            try:
                anchor_step = int(
                    max(
                        0,
                        min(
                            int(getattr(self, '_tier2_prev_investor_decision_step', i - int(max(getattr(self, 'investment_freq', 1) or 1, 1)))),
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
            info_investor['tier2_training_return'] = float(
                info_investor.get('price_return_invfreq', info_investor.get('price_return_1step', 0.0))
            )
            info_investor['tier2_training_horizon_steps'] = float(max(getattr(self, 'investment_freq', 1) or 1, 1))
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
                depreciation_ratio = float(nav_breakdown['depreciation_ratio'])
                years_elapsed = float(nav_breakdown['years_elapsed'])
                current_price_raw = float(self._price_raw[current_timestep] if current_timestep < len(self._price_raw) else 0.0)
                current_cash_dkk = float(getattr(self, 'budget', 0.0))
                trading_sleeve_value_dkk = float(nav_breakdown.get('trading_sleeve_value_dkk', trading_cash_dkk + financial_mtm_dkk))
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

            # Always define a fallback reason if forecasts were gated
            forecast_usage_reason = 'unknown'
            if debug_forecast:
                forecast_usage_reason = str(debug_forecast.get('forecast_not_used_reason', forecast_usage_reason))

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

            probe_r2_base = 0.0
            probe_r2_base_plus_signal = 0.0
            probe_delta_r2 = 0.0
            # Tier-2 probe removed (no forecast observations in paper setup).

            decision_step = self._decision_step_flag(self.t)
            exposure_exec = 0.0
            tier2_value_diag = {}
            try:
                if isinstance(getattr(self, "_last_actions", None), dict):
                    exec_action = self._last_actions.get("investor_0_exec", {})
                    exposure_exec = float(exec_action.get("exposure", 0.0))
                    delta_exposure = float(exec_action.get("delta_exposure", 0.0))
            except Exception:
                exposure_exec = 0.0
                delta_exposure = 0.0
            try:
                tier2_value_diag = dict(getattr(self, "_last_tier2_value_diag", {}) or {})
            except Exception:
                tier2_value_diag = {}

            action_sign = float(np.sign(exposure_exec)) if abs(exposure_exec) > 1e-9 else float(np.sign(exposure_val))
            try:
                obs_trade_signal_val = float(getattr(self, "_last_obs_trade_signal", 0.0))
            except Exception:
                obs_trade_signal_val = 0.0
            if abs(obs_trade_signal_val) <= 1e-12:
                try:
                    obs_trade_signal_val = float(debug_forecast.get("forecast_signal_short", 0.0))
                except Exception:
                    obs_trade_signal_val = 0.0
            trade_signal_active = 1.0 if abs(obs_trade_signal_val) > 1e-9 else 0.0
            trade_signal_sign = float(np.sign(obs_trade_signal_val)) if trade_signal_active > 0.0 else 0.0
            forecast_direction_log = float(debug_forecast.get("forecast_direction", trade_signal_sign))
            position_direction_log = float(debug_forecast.get("position_direction", action_sign))
            agent_followed_forecast_log = bool(
                debug_forecast.get(
                    "agent_followed_forecast",
                    forecast_used_flag
                    and abs(forecast_direction_log) > 1e-9
                    and abs(position_direction_log) > 1e-9
                    and np.sign(forecast_direction_log) == np.sign(position_direction_log),
                )
            )

            # Derive logging-only diagnostics from executed action + live finance state.
            # This avoids stale zeros from legacy debug dict fields.
            try:
                exec_action = {}
                if isinstance(getattr(self, "_last_actions", None), dict):
                    exec_action = self._last_actions.get("investor_0_exec", {}) or {}
                exec_wind = float(exec_action.get("wind", 0.0))
                exec_solar = float(exec_action.get("solar", 0.0))
                exec_hydro = float(exec_action.get("hydro", 0.0))
            except Exception:
                exec_wind = 0.0
                exec_solar = 0.0
                exec_hydro = 0.0

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

            forecast_returns = getattr(self, '_forecast_deltas_raw', {}) or {}
            forecast_return_short_log = float(forecast_returns.get('short', 0.0))
            forecast_return_medium_log = float(forecast_returns.get('medium', 0.0))
            forecast_return_long_log = float(forecast_returns.get('long', 0.0))
            if not np.isfinite(forecast_return_short_log):
                forecast_return_short_log = 0.0
            if not np.isfinite(forecast_return_medium_log):
                forecast_return_medium_log = 0.0
            if not np.isfinite(forecast_return_long_log):
                forecast_return_long_log = 0.0

            latest_price_returns = getattr(self, '_latest_price_returns', {}) or {}
            price_return_1step_log = float(latest_price_returns.get('one_step', 0.0))
            price_return_forecast_log = float(latest_price_returns.get('forecast', price_return_1step_log))
            if not np.isfinite(price_return_1step_log):
                price_return_1step_log = 0.0
            if not np.isfinite(price_return_forecast_log):
                price_return_forecast_log = price_return_1step_log

            current_price_dkk_log = float(current_price_raw)
            forecast_price_short_dkk_log = float(current_price_dkk_log * (1.0 + forecast_return_short_log))
            forecast_price_medium_dkk_log = float(current_price_dkk_log * (1.0 + forecast_return_medium_log))
            forecast_price_long_dkk_log = float(current_price_dkk_log * (1.0 + forecast_return_long_log))

            mape_short_recent_log = float(np.mean(list(self._horizon_mape.get('short', [0.0]))[-10:]) if len(self._horizon_mape.get('short', [])) > 0 else 0.0)
            mape_medium_recent_log = float(np.mean(list(self._horizon_mape.get('medium', [0.0]))[-10:]) if len(self._horizon_mape.get('medium', [])) > 0 else 0.0)
            mape_long_recent_log = float(np.mean(list(self._horizon_mape.get('long', [0.0]))[-10:]) if len(self._horizon_mape.get('long', [])) > 0 else 0.0)
            forecast_error_short_pct_log = float(np.clip(mape_short_recent_log, 0.0, 10.0) * 100.0)
            forecast_error_medium_pct_log = float(np.clip(mape_medium_recent_log, 0.0, 10.0) * 100.0)
            forecast_error_long_pct_log = float(np.clip(mape_long_recent_log, 0.0, 10.0) * 100.0)

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

            # Log per-step fund NAV movement, not the legacy trading-budget fallback.
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
                trading_cash_dkk=float(trading_cash_dkk),
                trading_sleeve_value_dkk=float(trading_sleeve_value_dkk),
                physical_book_value_dkk=float(physical_book_value_dkk),
                accumulated_operational_revenue_dkk=float(accumulated_operational_revenue_dkk),
                financial_mtm_dkk=float(financial_mtm_dkk),
                financial_exposure_dkk=float(financial_exposure_dkk),
                depreciation_ratio=float(depreciation_ratio),
                years_elapsed=float(years_elapsed),
                # Price data (for forward-looking accuracy analysis)
                    price_current=float(current_price_raw),  # Raw price in DKK/MWh
                    # Forecast signals (ensure floats)
                    # IMPORTANT: log the *real backend* z-scores/trust (not the retired Tier-2 debug dict).
                    z_short=float(getattr(self, 'z_short', 0.0)),
                    z_medium=float(getattr(self, 'z_medium', 0.0)),
                    z_long=float(getattr(self, 'z_long', 0.0)),
                    z_combined=float(getattr(self, 'mwdir', getattr(self, 'z_combined', 0.0))),
                    tier2_expert_confidence=float(self._get_tier2_expert_confidence()) if getattr(self, 'forecast_generator', None) is not None else 0.0,
                    forecast_confidence=float(self._get_tier2_expert_confidence()) if getattr(self, 'forecast_generator', None) is not None else 0.0,
                    forecast_trust=float(getattr(self, '_forecast_trust', 0.0)),
                    # OBSERVATION-LEVEL forecast signals (what the POLICY actually sees after warmup/ablation)
                    obs_z_short=float(getattr(self, '_last_obs_z_short', 0.0)),
                    obs_z_long=float(getattr(self, '_last_obs_z_long', 0.0)),
                    obs_forecast_trust=float(getattr(self, '_last_obs_forecast_trust', 0.0)),
                    obs_normalized_error=float(getattr(self, '_last_obs_normalized_error', 0.0)),
                    obs_trade_signal=float(obs_trade_signal_val),
                    # Position info (ensure floats)
                    position_signed=float(position_signed_log),
                    # FIX: debug_forecast.position_exposure can be stale/zero. Recompute from current financial notional.
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
                    # Price returns (ensure floats)
                    price_return_1step=float(price_return_1step_log),
                    price_return_forecast=float(price_return_forecast_log),
                    # Main reward components (already floats from above)
                    operational_score=operational_score,
                    risk_score=risk_score,
                    hedging_score=hedging_score,
                    nav_stability_score=nav_stability_score,
                    forecast_score=forecast_score,
                    # Reward weights (ensure floats)
                    weight_operational=float(reward_weights.get('operational_revenue', 0.0)),
                    weight_risk=float(reward_weights.get('risk_management', 0.0)),
                    weight_hedging=float(reward_weights.get('hedging_effectiveness', 0.0)),
                    weight_nav=float(reward_weights.get('nav_stability', 0.0)),
                    weight_forecast=float(reward_weights.get('forecast', 0.0)),
                    # NEW: Enhanced debugging fields
                    base_alignment=float(debug_forecast.get('base_alignment', 0.0)),
                    profitability_factor=float(debug_forecast.get('profitability_factor', 0.0)),
                    alignment_multiplier=float(debug_forecast.get('alignment_multiplier', 0.0)),
                    misalignment_penalty_mult=float(debug_forecast.get('misalignment_penalty_mult', 0.0)),
                    # CRITICAL FIX: Pass forecast_direction and position_direction to logger
                    forecast_direction=float(forecast_direction_log),
                    position_direction=float(position_direction_log),
                    is_aligned=float(agent_followed_forecast_log),
                    alignment_status=str('ALIGNED' if agent_followed_forecast_log else 'MISALIGNED'),
                    corr_short=float(debug_forecast.get('corr_short', 0.0)),
                    corr_medium=float(debug_forecast.get('corr_medium', 0.0)),
                    corr_long=float(debug_forecast.get('corr_long', 0.0)),
                    weight_short=float(debug_forecast.get('weight_short', 0.0)),
                    weight_medium=float(debug_forecast.get('weight_medium', 0.0)),
                    weight_long=float(debug_forecast.get('weight_long', 0.0)),
                    use_short=bool(debug_forecast.get('use_short', False)),
                    use_medium=bool(debug_forecast.get('use_medium', False)),
                    use_long=bool(debug_forecast.get('use_long', False)),
                    unrealized_pnl_norm=float(debug_forecast.get('unrealized_pnl_norm', 0.0)),
                    combined_confidence=float(debug_forecast.get('combined_confidence', 0.0)),
                    adaptive_multiplier=float(debug_forecast.get('adaptive_multiplier', 0.0)),
                    wind_pos_norm=float(exec_wind),
                    solar_pos_norm=float(exec_solar),
                    hydro_pos_norm=float(exec_hydro),
                    forecast_error=float(debug_forecast.get('forecast_error', 0.0)),
                    forecast_mape=float(debug_forecast.get('forecast_mape', 0.0)),
                    realized_vs_forecast=float(debug_forecast.get('realized_vs_forecast', 0.0)),
                    # NEW: Battery and generation forecast debugging
                    z_short_wind=float(getattr(self, 'z_short_wind', 0.0)),
                    z_short_solar=float(getattr(self, 'z_short_solar', 0.0)),
                    z_short_hydro=float(getattr(self, 'z_short_hydro', 0.0)),
                    z_short_total_gen=float(getattr(self, 'z_short_wind', 0.0) + getattr(self, 'z_short_solar', 0.0) + getattr(self, 'z_short_hydro', 0.0)),
                    # NEW: Adaptive forecast weight
                    adaptive_forecast_weight=float(getattr(self.reward_calculator, '_adaptive_forecast_weight', reward_weights.get('forecast', 0.0))),
                    # NEW: Forecast utilization flag
                    enable_forecast_util=False,
                    forecast_gate_passed=forecast_gate_flag,
                    forecast_used=forecast_used_flag,
                    forecast_not_used_reason=str(debug_forecast.get('forecast_not_used_reason', forecast_usage_reason)),
                    # NEW: Position alignment diagnostics (Bug #1 fix tracking)
                    position_alignment_status=str(position_alignment_status) if 'position_alignment_status' in locals() else 'no_strategy',
                    investor_position_ratio=float(investor_position_ratio_log),
                    investor_position_direction=int(position_direction_log),
                    investor_total_position=float(investor_total_position_dkk),
                    investor_exploration_bonus=float(exploration_bonus) if 'exploration_bonus' in locals() else 0.0,
                    # NEW: Per-horizon MAPE tracking
                    mape_short=float(np.mean(list(self._horizon_mape.get('short', [0.0]))[-100:]) if len(self._horizon_mape.get('short', [])) > 0 else 0.0),
                    mape_medium=float(np.mean(list(self._horizon_mape.get('medium', [0.0]))[-100:]) if len(self._horizon_mape.get('medium', [])) > 0 else 0.0),
                    mape_long=float(np.mean(list(self._horizon_mape.get('long', [0.0]))[-100:]) if len(self._horizon_mape.get('long', [])) > 0 else 0.0),
                    mape_short_recent=float(mape_short_recent_log),
                    mape_medium_recent=float(mape_medium_recent_log),
                    mape_long_recent=float(mape_long_recent_log),
                    # NEW: Per-horizon correlation tracking
                    horizon_corr_short=float(self._horizon_correlations.get('short', 0.0) if hasattr(self, '_horizon_correlations') else 0.0),
                    horizon_corr_medium=float(self._horizon_correlations.get('medium', 0.0) if hasattr(self, '_horizon_correlations') else 0.0),
                    horizon_corr_long=float(self._horizon_correlations.get('long', 0.0) if hasattr(self, '_horizon_correlations') else 0.0),
                    # NEW: Per-asset forecast accuracy
                    mape_wind=float(np.mean(list(self._forecast_errors.get('wind', [0.0]))[-10:]) if len(self._forecast_errors.get('wind', [])) > 0 else 0.0),
                    mape_solar=float(np.mean(list(self._forecast_errors.get('solar', [0.0]))[-10:]) if len(self._forecast_errors.get('solar', [])) > 0 else 0.0),
                    mape_hydro=float(np.mean(list(self._forecast_errors.get('hydro', [0.0]))[-10:]) if len(self._forecast_errors.get('hydro', [])) > 0 else 0.0),
                    mape_price=float(np.mean(list(self._forecast_errors.get('price', [0.0]))[-10:]) if len(self._forecast_errors.get('price', [])) > 0 else 0.0),
                    mape_load=float(np.mean(list(self._forecast_errors.get('load', [0.0]))[-10:]) if len(self._forecast_errors.get('load', [])) > 0 else 0.0),
                    # NEW: Forecast deltas (for debugging strategy)
                    forecast_delta_short=float(getattr(self, '_forecast_deltas_raw', {}).get('short', 0.0)),
                    forecast_delta_medium=float(getattr(self, '_forecast_deltas_raw', {}).get('medium', 0.0)),
                    forecast_delta_long=float(getattr(self, '_forecast_deltas_raw', {}).get('long', 0.0)),
                    # NEW: MAPE thresholds (for debugging strategy)
                    mape_threshold_short=float(getattr(self, '_mape_thresholds', {}).get('short', 0.0)),
                    mape_threshold_medium=float(getattr(self, '_mape_thresholds', {}).get('medium', 0.0)),
                    mape_threshold_long=float(getattr(self, '_mape_thresholds', {}).get('long', 0.0)),
                    # NEW: Price floor diagnostics (Bug #4 fix tracking)
                    price_raw=float(getattr(self, '_price_raw', [0.0])[self.t] if hasattr(self, '_price_raw') and self.t < len(self._price_raw) else 0.0),
                    price_floor_used=float(getattr(self, '_current_price_floor', 0.0)),
                    # Directional accuracy (short horizon active; medium/long kept as compatibility zeros)
                    direction_accuracy_short=float(debug_forecast.get('direction_accuracy_short', 0.0)),
                    direction_accuracy_medium=float(debug_forecast.get('direction_accuracy_medium', 0.0)),
                    direction_accuracy_long=float(debug_forecast.get('direction_accuracy_long', 0.0)),
                    # NEW: Battery forecast bonus
                    battery_forecast_bonus=float(battery_forecast_delta),
                    # NEW: Forecast vs actual comparison (deep debugging)
                    current_price_dkk=float(current_price_dkk_log),
                    forecast_price_short_dkk=float(forecast_price_short_dkk_log),
                    forecast_price_medium_dkk=float(forecast_price_medium_dkk_log),
                    forecast_price_long_dkk=float(forecast_price_long_dkk_log),
                    forecast_error_short_pct=float(forecast_error_short_pct_log),
                    forecast_error_medium_pct=float(forecast_error_medium_pct_log),
                    forecast_error_long_pct=float(forecast_error_long_pct_log),
                    agent_followed_forecast=bool(agent_followed_forecast_log),
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
                    # Tier-2 overlay diagnostics
                    tier2_overlay_enabled=float(tier2_value_diag.get('enabled', 0.0)),
                    tier2_overlay_delta=float(tier2_value_diag.get('delta', 0.0)),
                    tier2_overlay_pred_sigma=float(tier2_value_diag.get('pred_sigma', 0.0)),
                    tier2_overlay_reliability=float(tier2_value_diag.get('reliability', 1.0)),
                    tier2_overlay_gain=float(tier2_value_diag.get('overlay_gain', 0.0)),
                    tier2_overlay_forecast_trust=float(
                        tier2_value_diag.get('forecast_trust', tier2_value_diag.get('trust_component', 0.0))
                    ),
                    tier2_overlay_calibrated_forecast_trust=float(
                        tier2_value_diag.get('calibrated_forecast_trust', 0.0)
                    ),
                    tier2_overlay_mape_quality=float(tier2_value_diag.get('mape_component', 1.0)),
                    tier2_overlay_alignment=float(tier2_value_diag.get('alignment', 0.0)),
                    tier2_overlay_forecast_signal=float(tier2_value_diag.get('forecast_signal', 0.0)),
                    tier2_overlay_edge_signal=float(tier2_value_diag.get('forecast_edge_signal', 0.0)),
                    tier2_overlay_consensus_signal=float(tier2_value_diag.get('forecast_consensus_signal', 0.0)),
                    tier2_overlay_curvature_signal=float(tier2_value_diag.get('forecast_curvature_signal', 0.0)),
                    tier2_overlay_uncertainty_quality=float(tier2_value_diag.get('uncertainty_quality', 1.0)),
                    tier2_overlay_route_uncertainty_quality=float(tier2_value_diag.get('uncertainty_quality', 1.0)),
                    tier2_overlay_metadata_skill=float(tier2_value_diag.get('metadata_skill', 0.5)),
                    tier2_overlay_imbalance_signal=float(
                        tier2_value_diag.get('imbalance_signal', tier2_value_diag.get('physical_pressure_signal', 0.0))
                    ),
                    tier2_overlay_context_strength=float(tier2_value_diag.get('context_strength', 0.0)),
                    tier2_overlay_realized_gain_signal=float(tier2_value_diag.get('realized_gain_signal', 0.0)),
                    tier2_overlay_target_delta=float(tier2_value_diag.get('target_delta', 0.0)),
                    tier2_overlay_delta_prediction=float(tier2_value_diag.get('delta_prediction', 0.0)),
                    tier2_overlay_sharpe_signal=float(tier2_value_diag.get('sharpe_signal', 0.0)),
                    tier2_overlay_sharpe_before=float(tier2_value_diag.get('sharpe_before', 0.0)),
                    tier2_overlay_sharpe_after=float(tier2_value_diag.get('sharpe_after', 0.0)),
                    tier2_overlay_sharpe_delta=float(tier2_value_diag.get('sharpe_delta', 0.0)),
                    tier2_overlay_value_prediction=float(tier2_value_diag.get('value_prediction', 0.0)),
                    tier2_overlay_value_lcb=float(tier2_value_diag.get('value_lcb', 0.0)),
                    tier2_overlay_nav_prediction=float(tier2_value_diag.get('nav_prediction', 0.0)),
                    tier2_overlay_nav_lcb=float(tier2_value_diag.get('nav_lcb', 0.0)),
                    tier2_overlay_return_floor_prediction=float(tier2_value_diag.get('return_floor_prediction', 0.0)),
                    tier2_overlay_return_floor_lcb=float(tier2_value_diag.get('return_floor_lcb', 0.0)),
                    tier2_overlay_tail_risk_prediction=float(tier2_value_diag.get('tail_risk_prediction', 0.0)),
                    tier2_overlay_value_gate=float(tier2_value_diag.get('value_gate', 0.0)),
                    tier2_overlay_nav_gate=float(tier2_value_diag.get('nav_gate', 0.0)),
                    tier2_overlay_return_floor_gate=float(tier2_value_diag.get('return_floor_gate', 0.0)),
                    tier2_overlay_delta_gate=float(tier2_value_diag.get('delta_gate', 0.0)),
                    tier2_overlay_learned_gate=float(tier2_value_diag.get('learned_gate', tier2_value_diag.get('projection_strength', tier2_value_diag.get('reliability', 0.0)))),
                    tier2_overlay_certified_override_gain=float(tier2_value_diag.get('certified_override_gain', 0.0)),
                    tier2_overlay_dominant_expert_weight=float(tier2_value_diag.get('dominant_expert_weight', 0.0)),
                    tier2_overlay_expert_weight_entropy=float(tier2_value_diag.get('expert_weight_entropy', 0.0)),
                    tier2_overlay_internal_base_weight=float(tier2_value_diag.get('internal_base_weight', 0.0)),
                    tier2_overlay_internal_trend_weight=float(tier2_value_diag.get('internal_trend_weight', 0.0)),
                    tier2_overlay_internal_reversion_weight=float(tier2_value_diag.get('internal_reversion_weight', 0.0)),
                    tier2_overlay_internal_defensive_weight=float(tier2_value_diag.get('internal_defensive_weight', 0.0)),
                    # ACRP-v2 diagnostics
                    tier2_overlay_acrp_trust_radius=float(tier2_value_diag.get('acrp_learned_trust_radius', 0.0)),
                    tier2_overlay_acrp_advantage_strength=float(tier2_value_diag.get('acrp_advantage_strength', 0.0)),
                    tier2_overlay_acrp_da_learned_scalar=float(tier2_value_diag.get('acrp_da_learned_scalar', 0.0)),
                    tier2_overlay_acrp_attn_gate_weight=float(tier2_value_diag.get('acrp_attn_gate_weight', 0.0)),
                    tier2_overlay_acrp_attn_conf_weight=float(tier2_value_diag.get('acrp_attn_conf_weight', 0.0)),
                    tier2_overlay_acrp_attn_outlook_weight=float(tier2_value_diag.get('acrp_attn_outlook_weight', 0.0)),
                    tier2_reference_quality_gap=float(tier2_value_diag.get('reference_quality_gap', 0.0)),
                    tier2_reference_risk_gap=float(tier2_value_diag.get('reference_risk_gap', 0.0)),
                    tier2_reference_vs_consensus_gap=float(tier2_value_diag.get('reference_vs_consensus_gap', 0.0)),
                    tier2_reference_leadership=float(tier2_value_diag.get('reference_leadership', 0.0)),
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
                    probe_r2_base=float(probe_r2_base),
                    probe_r2_base_plus_signal=float(probe_r2_base_plus_signal),
                    probe_delta_r2=float(probe_delta_r2),
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
            import traceback
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
                    # Backward-compatible aliases:
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
    # Observations (Tier1-only; no forecast-augmented obs)
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

        Investor (9D) + battery (12D) are built by ObservationBuilder.
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

            # Investor (9D): direct trading investor with explicit constraint context.
            price_momentum = self._compute_investor_price_momentum(i)
            realized_volatility = self._compute_investor_realized_volatility(i)
            is_decision_step = self._decision_step_flag(self.t)
            risk_exposure_cap = self._clip01(
                getattr(self, "risk_exposure_cap", getattr(self, "risk_multiplier", 1.0))
            )
            investor_local_drawdown = self._clip01(getattr(self, "_investor_local_drawdown", 0.0))
            inv = self._obs_buf["investor_0"]
            # Tier-2 ACRP signals flow through the DL model's internal 11D
            # core features only; RL investor always sees 9D.
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
                current_exposure_norm=float(
                    self._estimate_current_investor_exposure(getattr(self, "_last_tradeable_capital", None))
                ),
                risk_exposure_cap=risk_exposure_cap,
                local_drawdown=investor_local_drawdown,
            )

            if self.t % 1000 == 0 or self.t == 0:
                logger.info(
                    f"[INVESTOR_OBS] t={self.t} | price_momentum={inv[0]:.3f}, "
                    f"realized_vol={inv[1]:.3f}, budget={inv[2]:.3f}, "
                    f"exposure={inv[3]:.3f}, mtm_pnl={inv[4]:.3f}, decision_step={inv[5]:.1f}, "
                    f"cap_frac={inv[6]:.3f}, risk_cap={inv[7]:.3f}, drawdown={inv[8]:.3f}"
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

            # Risk controller (11D): add overall_risk_snapshot and drawdown so policy sees what it's penalized for
            drawdown = float(getattr(self.reward_calculator, 'current_drawdown', 0.0)) if self.reward_calculator else 0.0
            rsk = self._obs_buf["risk_controller_0"]
            rsk[:11] = (
                price_n,
                float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
                float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.financial_positions["wind_instrument_value"], self.init_budget), -1.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.financial_positions["solar_instrument_value"], self.init_budget), -1.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.financial_positions["hydro_instrument_value"], self.init_budget), -1.0, 1.0)) * 10.0,
                float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0)),
                float(np.clip(self.risk_multiplier, 0.0, 2.0)) * 5.0,
                float(np.clip(self.overall_risk_snapshot, 0.0, 1.0)) * 10.0,
                float(np.clip(drawdown, 0.0, 1.0)) * 10.0,
            )

            # Meta controller (11D)
            perf_ratio = float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0))
            meta = self._obs_buf["meta_controller_0"]
            meta[:11] = (
                float(np.clip(SafeDivision.div(self.budget, self.init_budget / 10.0, 0.0), 0.0, 10.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["wind_instrument_value"], self.init_budget, 0.0), -10.0, 10.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["solar_instrument_value"], self.init_budget, 0.0), -10.0, 10.0)),
                float(np.clip(SafeDivision.div(self.financial_positions["hydro_instrument_value"], self.init_budget, 0.0), -10.0, 10.0)),
                price_n,
                float(np.clip(self.overall_risk_snapshot, 0.0, 1.0)) * 10.0,
                perf_ratio,
                float(np.clip(self.market_risk_snapshot, 0.0, 1.0)) * 10.0,
                float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
                float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
                float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
            )

        except Exception as e:
            msg = f"[OBS_BUILD_FATAL] Error building observations at step {self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    # ------------------------------------------------------------------
    # FGB: Forecast-Guided Baseline Signals
    # ------------------------------------------------------------------
    def _resolve_fgb_trust_horizon(
        self,
        agent_name: str,
        default_pref: str = "auto",
        prefer_cached: bool = True,
    ) -> str:
        """
        Resolve the effective horizon used by forecast-trust logic for one agent.

        This keeps the main trust path and the trust-floor helper on the same
        selection policy, especially when `auto` mode is used.
        """
        valid_horizons = ("short", "medium", "long")
        try:
            min_samples = int(getattr(self.config, "fgb_trust_min_samples", 50) or 50)
        except Exception:
            min_samples = 50

        agent_key = str(agent_name or "").strip() or "investor_0"
        if prefer_cached:
            try:
                cached_horizon = str((getattr(self, "_fgb_trust_horizon_cache", {}) or {}).get(agent_key, "")).strip().lower()
            except Exception:
                cached_horizon = ""
            if cached_horizon in valid_horizons:
                return cached_horizon

        try:
            pref_map = getattr(self.config, "fgb_trust_horizon_by_agent", {})
            if isinstance(pref_map, dict):
                pref = pref_map.get(agent_key, default_pref)
            else:
                pref = default_pref
        except Exception:
            pref = default_pref
        pref = str(pref or default_pref).strip().lower()
        horizon = pref if pref in valid_horizons else "auto"
        if horizon != "auto":
            return horizon

        best_h = None
        best_score = -1.0
        try:
            hits = getattr(self, "_horizon_sign_hit", {}) or {}
        except Exception:
            hits = {}
        for h in valid_horizons:
            xs = list(hits.get(h, []))
            if len(xs) < min_samples:
                continue
            xs = [float(x) for x in xs[-min_samples:] if np.isfinite(x)]
            if len(xs) == 0:
                continue
            score = float(np.mean(xs))
            if score > best_score:
                best_score = score
                best_h = h

        if best_h is not None:
            return best_h

        counts = []
        for idx, h in enumerate(valid_horizons):
            try:
                counts.append((len(list(hits.get(h, []))), -idx, h))
            except Exception:
                counts.append((0, -idx, h))
        counts.sort(reverse=True)
        return counts[0][2] if counts else "short"

    def _resolve_tier2_forecast_horizon(self) -> str:
        """
        Resolve the Tier-2 forecast horizon.

        Tier-2 stays on the canonical short horizon by design. The live trade
        cadence is fixed separately by the environment/meta-controller config.
        """
        return "short"

    def _get_tier2_forecast_snapshot(self) -> Dict[str, float]:
        """
        Return the current full-horizon forecast snapshot.

        When a forecast cache has been precomputed, predict_all_horizons()
        resolves against cached per-timestep arrays rather than recomputing
        forecasts from scratch, so Tier-2 can safely consume richer forecast
        context without changing Tier-1 observations.
        """
        step = int(getattr(self, "t", 0))
        if int(getattr(self, "_tier2_forecast_snapshot_step", -1)) == step:
            cached = getattr(self, "_tier2_forecast_snapshot", {})
            if isinstance(cached, dict):
                return cached

        snapshot: Dict[str, float] = {}
        fg = getattr(self, "forecast_generator", None)
        if fg is not None and hasattr(fg, "predict_all_horizons"):
            try:
                raw = fg.predict_all_horizons(timestep=step)
                if isinstance(raw, dict):
                    snapshot = raw
            except Exception:
                snapshot = {}
        self._tier2_forecast_snapshot_step = step
        self._tier2_forecast_snapshot = dict(snapshot)
        return snapshot

    def _get_tier2_short_expert_metadata_skill(self, expert_name: str) -> float:
        fg = getattr(self, "forecast_generator", None)
        if fg is None or not hasattr(fg, "get_price_short_expert_metadata_quality"):
            return 0.5
        try:
            return float(np.clip(fg.get_price_short_expert_metadata_quality(expert_name), 0.0, 1.0))
        except Exception:
            return 0.5

    def _get_tier2_short_expert_direction_accuracy(self, expert_name: str, window: int = 50) -> float:
        try:
            xs = list((getattr(self, "_tier2_short_expert_sign_hit", {}) or {}).get(expert_name, []))[-max(1, int(window)):]
            vals = [float(x) for x in xs if np.isfinite(x)]
        except Exception:
            vals = []
        if len(vals) < 5:
            return self._get_tier2_short_expert_metadata_skill(expert_name)
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _get_tier2_short_expert_economic_skill(self, expert_name: str, window: int = 100) -> float:
        try:
            xs = list((getattr(self, "_tier2_short_expert_economic_skill", {}) or {}).get(expert_name, []))[-max(1, int(window)):]
            vals = [float(x) for x in xs if np.isfinite(x)]
        except Exception:
            vals = []
        direction_prior = self._get_tier2_short_expert_direction_accuracy(expert_name)
        metadata_prior = self._get_tier2_short_expert_metadata_skill(expert_name)
        prior = float(
            np.clip(
                0.60 * direction_prior + 0.40 * metadata_prior,
                0.0,
                1.0,
            )
        )
        if len(vals) < 5:
            return prior
        realized_mean = float(np.mean(vals))
        shrink = float(np.clip(len(vals) / max(25.0, 0.5 * float(max(1, int(window)))), 0.0, 1.0))
        return float(np.clip((1.0 - shrink) * prior + shrink * realized_mean, 0.0, 1.0))

    def _get_tier2_short_expert_mape_quality(self, expert_name: str, window: int = 10) -> float:
        try:
            xs = list((getattr(self, "_tier2_short_expert_mape", {}) or {}).get(expert_name, []))[-max(1, int(window)):]
            vals = [float(x) for x in xs if np.isfinite(x)]
        except Exception:
            vals = []
        if len(vals) < 3:
            return 0.5
        try:
            ref = float(getattr(self.config, "tier2_mape_reference", 0.25) or 0.25)
        except Exception:
            ref = 0.25
        ref = max(ref, 1e-6)
        robust_mape = float(0.5 * np.median(vals) + 0.5 * np.quantile(vals, 0.80))
        return float(np.clip(1.0 - (robust_mape / ref), 0.0, 1.0))

    def _get_tier2_short_expert_conformal_risk(self, expert_name: str, window: int = 50) -> float:
        try:
            pairs = list((getattr(self, "_tier2_short_expert_return_pairs", {}) or {}).get(expert_name, []))[-max(5, int(window)):]
        except Exception:
            pairs = []
        residuals = []
        for pair in pairs:
            try:
                f_ret = float(pair.get("forecast_return", 0.0))
                a_ret = float(pair.get("actual_return", 0.0))
                if np.isfinite(f_ret) and np.isfinite(a_ret):
                    residuals.append(a_ret - f_ret)
            except Exception:
                continue
        if len(residuals) < 5:
            return 0.5
        residual_np = np.asarray(residuals, dtype=np.float32)
        try:
            ref = float(getattr(self.config, "tier2_value_target_scale", 0.01) or 0.01)
        except Exception:
            ref = 0.01
        ref = max(ref, 1e-6)
        try:
            conformal_alpha = float(getattr(self.config, "tier2_value_conformal_alpha", 0.10) or 0.10)
        except Exception:
            conformal_alpha = 0.10
        try:
            conformal_scale = float(getattr(self.config, "tier2_value_conformal_scale", 1.0) or 1.0)
        except Exception:
            conformal_scale = 1.0
        tail = float(np.quantile(np.abs(residual_np), 1.0 - float(np.clip(conformal_alpha, 0.01, 0.49))))
        return float(np.clip(np.tanh((tail / ref) * max(conformal_scale, 1e-6)), 0.0, 1.0))

    def _get_tier2_short_expert_quality(self, expert_name: str) -> float:
        return float(
            tier2_expert_quality_blend(
                direction_accuracy=self._get_tier2_short_expert_direction_accuracy(expert_name),
                mape_quality=self._get_tier2_short_expert_mape_quality(expert_name),
                metadata_skill=self._get_tier2_short_expert_metadata_skill(expert_name),
                economic_skill=self._get_tier2_short_expert_economic_skill(expert_name),
            )
        )

    def _extract_tier2_short_memory_state(self, tier2_features: np.ndarray) -> Dict[str, float]:
        """
        Decode the latest Tier-2 short-memory row from the active feature block.

        This keeps runtime overlay scaling and diagnostics aligned with the
        exact tensor that the learned controller consumes. That matters for the
        full forecast-memory ablation: when the forecast-memory block is
        neutralized, the runtime path must also decode that neutralized
        short-horizon context rather than the raw live forecast state.
        """
        try:
            return extract_tier2_memory_state_from_features(
                tier2_features,
                core_dim=TIER2_VALUE_CORE_DIM,
            )
        except Exception:
            defaults = {
                "quality_context": 0.5,
                "expert_consensus_signal": 0.0,
                "expert_disagreement": 0.0,
                "short_imbalance_signal": 0.0,
                "context_strength": 0.0,
                "primary_signal": 0.0,
                "primary_quality": 0.5,
                "primary_risk": 0.5,
                "primary_quality_gap": 0.0,
                "primary_risk_gap": 0.0,
                "primary_vs_consensus_gap": 0.0,
                "primary_leadership": 0.0,
            }
            for expert_name in TIER2_ACTIVE_EXPERT_NAMES:
                defaults[f"{expert_name}_signal"] = 0.0
                defaults[f"{expert_name}_quality"] = 0.5
                defaults[f"{expert_name}_risk"] = 0.5
            return defaults

    def _default_tier2_value_diag(self) -> Dict[str, float]:
        """Default diagnostics for the ANN-primary Tier-2 overlay."""
        return {
            "enabled": 0.0,
            "delta": 0.0,
            "pred_sigma": 0.0,
            "realized_gain_signal": 0.0,
            "target_delta": 0.0,
            "sharpe_signal": 0.0,
            "sharpe_before": 0.0,
            "sharpe_after": 0.0,
            "sharpe_delta": 0.0,
            "reliability": 0.0,
            "overlay_gain": 0.0,
            "forecast_trust": 0.0,
            "trust_component": 0.0,
            "mape_component": 0.0,
            "uncertainty_quality": 0.0,
            "metadata_skill": 0.0,
            "residual_bias_component": 0.0,
            "residual_dispersion_quality_component": 0.0,
            "residual_sign_bias_component": 0.0,
            "conformal_residual_risk": 0.0,
            "dl_component": 0.0,
            "alignment": 0.0,
            "forecast_signal": 0.0,
            "forecast_edge_signal": 0.0,
            "forecast_consensus_signal": 0.0,
            "forecast_curvature_signal": 0.0,
            "imbalance_signal": 0.0,
            "physical_pressure_signal": 0.0,
            "context_strength": 0.0,
            "action_sign": 0.0,
            "forecast_sign": 0.0,
            "temporal_direction_signal": 0.0,
            "temporal_structural_signal": 0.0,
            "certified_override_gain": 0.0,
            "decision_improvement": 0.0,
            "value_prediction": 0.0,
            "value_pred_sigma": 0.0,
            "value_lcb": 0.0,
            "value_ucb": 0.0,
            "nav_lcb": 0.0,
            "return_floor_lcb": 0.0,
            "overlay_confidence": 0.0,
            "projection_strength": 0.0,
            "nav_prediction": 0.0,
            "return_floor_prediction": 0.0,
            "tail_risk_prediction": 0.0,
            "value_gate": 0.0,
            "nav_gate": 0.0,
            "return_floor_gate": 0.0,
            "delta_gate": 0.0,
            "learned_gate": 0.0,
            "learned_gate_pass": 0.0,
            "trust_gate": 1.0,
            "factor_reward_prediction": 0.0,
            "reference_quality_gap": 0.0,
            "reference_risk_gap": 0.0,
            "reference_vs_consensus_gap": 0.0,
            "reference_leadership": 0.0,
            "delta_prediction": 0.0,
            "controller_confidence": 0.0,
            "residual_prediction": 0.0,
            "dominant_expert_weight": 0.0,
            "expert_weight_entropy": 0.0,
            "internal_base_weight": 0.0,
            "internal_trend_weight": 0.0,
            "internal_reversion_weight": 0.0,
            "internal_defensive_weight": 0.0,
            "acrp_learned_trust_radius": 0.0,
            "acrp_advantage_strength": 0.0,
            "acrp_da_learned_scalar": 0.0,
            "acrp_attn_gate_weight": 0.0,
            "acrp_attn_conf_weight": 0.0,
            "acrp_attn_outlook_weight": 0.0,
            "policy_scale": 0.0,
            "deployment_strength_mean": 0.0,
        }

    def _log_tier2_runtime_issue(self, tag: str, step: int, err: Exception) -> None:
        """Throttle Tier-2 runtime warnings so failures stay visible without flooding logs."""
        try:
            signature = f"{tag}:{type(err).__name__}:{err}"
            count = int(getattr(self, "_tier2_runtime_issue_count", 0)) + 1
            last_signature = getattr(self, "_last_tier2_runtime_issue_signature", None)
            self._tier2_runtime_issue_count = count
            self._last_tier2_runtime_issue_signature = signature
            if signature != last_signature or count in (1, 5, 20, 100):
                logger.warning("[TIER2_%s] step=%s count=%s error=%s", tag, int(step), count, signature)
        except Exception:
            pass

    def _compute_tier2_short_expert_signal(
        self,
        step: int,
        expert_name: str = TIER2_PRIMARY_EXPERT_NAME,
        snapshot: Optional[Dict[str, float]] = None,
    ) -> float:
        """Decode the short-horizon directional signal for one price expert."""
        pred_return = self._compute_tier2_short_expert_return(
            step,
            expert_name=expert_name,
            snapshot=snapshot,
        )
        tanh_scale = float(getattr(self.config, "forecast_return_tanh_scale", 1.5) or 1.5)
        return float(np.clip(np.tanh(pred_return * tanh_scale), -1.0, 1.0))

    def _compute_tier2_short_expert_return(
        self,
        step: int,
        expert_name: str = TIER2_PRIMARY_EXPERT_NAME,
        snapshot: Optional[Dict[str, float]] = None,
    ) -> float:
        """Decode the clipped short-horizon forecast return for one price expert."""
        step = int(max(step, 0))
        snapshot = self._get_tier2_forecast_snapshot() if snapshot is None else (snapshot or {})
        current_price_raw = float(self._price_raw[step]) if step < len(self._price_raw) else float(snapshot.get("price_forecast_short", 0.0))
        denom_floor = float(getattr(self.config, "forecast_return_denom_floor", 0.25) or 0.25)
        return_clip = float(getattr(self.config, "forecast_return_clip", 0.15) or 0.15)
        denom = max(abs(current_price_raw), denom_floor, 1.0)
        pred_price = float(snapshot.get(f"price_short_expert_{expert_name}", snapshot.get("price_forecast_short", current_price_raw)))
        pred_return = (pred_price - current_price_raw) / denom if denom > 1e-8 else 0.0
        return float(np.clip(pred_return, -return_clip, return_clip))

    def _build_tier2_forecast_memory_entry(self, step: int, proposed_exposure: float) -> Dict[str, float]:
        """
        Build one short-horizon Tier-2 memory row from the ANN forecaster outputs.

        The active Tier-2 contract keeps the frozen Tier-1 core separate and
        feeds the overlay a richer temporal sequence:
        - direct ANN outputs: signal, quality/risk, return, direction margin,
          latent coordinates
        - raw market context: normalized price/return and physical/load levels

        This keeps the runtime path forecast-backed but moves the actual
        representation burden into the learned temporal encoder instead of
        hand-built summary gates.
        """
        step = int(max(step, 0))
        proposed_exposure = float(np.clip(proposed_exposure, -1.0, 1.0))
        self._tier2_forecast_horizon = "short"
        snapshot = self._get_tier2_forecast_snapshot()

        primary_name = str(
            getattr(self.config, "tier2_primary_expert", TIER2_PRIMARY_EXPERT_NAME) or TIER2_PRIMARY_EXPERT_NAME
        ).strip().lower()
        if primary_name not in TIER2_ACTIVE_EXPERT_NAMES:
            primary_name = TIER2_PRIMARY_EXPERT_NAME

        ann_signal = float(self._compute_tier2_short_expert_signal(step, expert_name=primary_name, snapshot=snapshot))
        ann_return_raw = float(
            snapshot.get(
                f"price_short_expert_{primary_name}_pred_return",
                self._compute_tier2_short_expert_return(step, expert_name=primary_name, snapshot=snapshot),
            )
        )
        return_clip = float(getattr(self.config, "forecast_return_clip", 0.15) or 0.15)
        ann_return_signal = float(np.clip(ann_return_raw / max(return_clip, 1e-6), -1.0, 1.0))
        ann_abs_return = float(np.clip(abs(ann_return_raw) / max(return_clip, 1e-6), 0.0, 1.0))
        ann_direction_margin = float(
            np.clip(
                snapshot.get(f"price_short_expert_{primary_name}_direction_margin", ann_signal),
                -1.0,
                1.0,
            )
        )
        recent_short_hits = [
            float(x)
            for x in list((getattr(self, "_horizon_sign_hit", {}) or {}).get("short", []))[-100:]
            if np.isfinite(x)
        ]
        direction_accuracy_short = float(
            np.clip(
                np.mean(recent_short_hits) if len(recent_short_hits) > 0 else ann_quality,
                0.0,
                1.0,
            )
        )
        ann_uncertainty = float(
            np.clip(
                snapshot.get(
                    f"price_short_expert_{primary_name}_uncertainty",
                    0.5,
                ),
                0.0,
                1.0,
            )
        )
        ann_quality = float(
            np.clip(
                snapshot.get(
                    f"price_short_expert_{primary_name}_quality",
                    0.5,
                ),
                0.0,
                1.0,
            )
        )
        ann_risk = ann_uncertainty

        try:
            load_short = float(self._load[step])
        except Exception:
            load_short = 0.0
        try:
            total_gen_short = float(self._wind[step]) + float(self._solar[step]) + float(self._hydro[step])
        except Exception:
            total_gen_short = 0.0
        imbalance_denom = max(abs(load_short) + abs(total_gen_short), 1.0)
        short_imbalance_signal = float(np.clip((load_short - total_gen_short) / imbalance_denom, -1.0, 1.0))

        current_price_raw = float(self._price_raw[step]) if step < len(self._price_raw) else 0.0
        ann_pred_price = float(
            snapshot.get(
                f"price_short_expert_{primary_name}",
                snapshot.get("price_forecast_short", current_price_raw),
            )
        )
        ann_price_level_signal = float(np.clip(normalize_price(ann_pred_price), -1.0, 1.0))
        prev_price_raw = float(self._price_raw[max(step - 1, 0)]) if len(self._price_raw) > 0 else current_price_raw
        price_denom = max(abs(prev_price_raw), 25.0, 1.0)
        price_return_signal = float(
            np.clip(
                ((current_price_raw - prev_price_raw) / price_denom) / max(return_clip, 1e-6),
                -1.0,
                1.0,
            )
        )
        wind_level_signal = float(np.clip(float(self._wind[step]) / 300.0, 0.0, 1.0)) if step < len(self._wind) else 0.0
        solar_level_signal = float(np.clip(float(self._solar[step]) / 120.0, 0.0, 1.0)) if step < len(self._solar) else 0.0
        hydro_level_signal = float(np.clip(float(self._hydro[step]) / 50.0, 0.0, 1.0)) if step < len(self._hydro) else 0.0
        load_level_signal = float(np.clip(float(self._load[step]) / 450.0, 0.0, 1.0)) if step < len(self._load) else 0.0
        ann_latents = [
            float(np.clip(snapshot.get(f"price_short_expert_{primary_name}_latent_{idx}", 0.0), -1.0, 1.0))
            for idx in range(4)
        ]
        ann_latent_norm = float(
            np.clip(
                snapshot.get(
                    f"price_short_expert_{primary_name}_latent_norm",
                    np.linalg.norm(np.asarray(ann_latents, dtype=np.float32)) / max(np.sqrt(4.0), 1e-6),
                ),
                0.0,
                1.0,
            )
        )
        primary_signal = ann_signal
        primary_quality = ann_quality
        primary_risk = ann_risk
        consensus_signal = float(np.clip(primary_signal, -1.0, 1.0))
        disagreement = float(np.clip(ann_uncertainty, 0.0, 1.0))
        primary_quality_gap = 0.0
        primary_risk_gap = 0.0
        primary_vs_consensus_gap = 0.0
        quality_context = float(np.clip(primary_quality, 0.0, 1.0))
        context_strength = float(np.clip(max(abs(primary_signal), ann_latent_norm), 0.0, 1.0))
        entry = {
            "step": float(step),
            "expert_consensus_signal": consensus_signal,
            "expert_disagreement": disagreement,
            "short_imbalance_signal": short_imbalance_signal,
            "ann_return_signal": ann_return_signal,
            "ann_abs_return": ann_abs_return,
            "ann_price_level_signal": ann_price_level_signal,
            "ann_direction_margin": float(np.clip(ann_direction_margin, -1.0, 1.0)),
            "ann_direction_accuracy": direction_accuracy_short,
            "ann_latent_0": ann_latents[0],
            "ann_latent_1": ann_latents[1],
            "ann_latent_2": ann_latents[2],
            "ann_latent_3": ann_latents[3],
            "ann_latent_norm": ann_latent_norm,
            "price_level_signal": ann_price_level_signal,
            "price_return_signal": price_return_signal,
            "wind_level_signal": wind_level_signal,
            "solar_level_signal": solar_level_signal,
            "hydro_level_signal": hydro_level_signal,
            "load_level_signal": load_level_signal,
            "quality": float(np.clip(quality_context, 0.0, 1.0)),
            "context_strength": context_strength,
            "alignment": float(np.clip(proposed_exposure * primary_signal, -1.0, 1.0)),
            "signal_target": primary_signal,
        }
        for expert_name in TIER2_ACTIVE_EXPERT_NAMES:
            entry[f"{expert_name}_signal"] = float(primary_signal)
            entry[f"{expert_name}_quality"] = float(primary_quality)
            entry[f"{expert_name}_risk"] = float(primary_risk)
        return entry

    def _ensure_tier2_forecast_memory(self, step: int, proposed_exposure: float) -> Dict[str, float]:
        """Update the cached short-horizon memory row once per env step."""
        step = int(max(step, 0))
        cached_step = int(getattr(self, "_tier2_forecast_memory_step", -1))
        cached_entry = getattr(self, "_tier2_forecast_memory_entry", None)
        if cached_step == step and isinstance(cached_entry, dict):
            return dict(cached_entry)

        if not isinstance(getattr(self, "_tier2_forecast_memory", None), deque):
            self._tier2_forecast_memory = deque(
            maxlen=max(1, int(getattr(self.config, "tier2_value_memory_steps", TIER2_VALUE_MEMORY_STEPS)))
            )

        entry = self._build_tier2_forecast_memory_entry(step, proposed_exposure=proposed_exposure)
        if len(self._tier2_forecast_memory) > 0 and int(self._tier2_forecast_memory[-1].get("step", -1)) == step:
            self._tier2_forecast_memory[-1] = dict(entry)
        else:
            self._tier2_forecast_memory.append(dict(entry))
        self._tier2_forecast_memory_step = step
        self._tier2_forecast_memory_entry = dict(entry)
        return dict(entry)

    def _get_tier2_forecast_memory_matrix(self, step: int, proposed_exposure: float) -> np.ndarray:
        """Return the padded short-memory matrix for the ANN-primary Tier-2 contract."""
        current_entry = self._ensure_tier2_forecast_memory(step, proposed_exposure=proposed_exposure)
        memory_steps = max(1, int(getattr(self.config, "tier2_value_memory_steps", TIER2_VALUE_MEMORY_STEPS)))
        channel_count = int(TIER2_VALUE_MEMORY_CHANNELS)
        channel_keys = (
            [f"{e}_signal" for e in TIER2_ACTIVE_EXPERT_NAMES]
            + [f"{e}_quality" for e in TIER2_ACTIVE_EXPERT_NAMES]
            + [f"{e}_risk" for e in TIER2_ACTIVE_EXPERT_NAMES]
            + ["expert_consensus_signal", "expert_disagreement", "short_imbalance_signal"]
            + [
                "ann_return_signal",
                "ann_abs_return",
                "ann_price_level_signal",
                "ann_direction_margin",
                "ann_direction_accuracy",
                "ann_latent_0",
                "ann_latent_1",
                "ann_latent_2",
                "ann_latent_3",
                "ann_latent_norm",
                "price_level_signal",
                "price_return_signal",
                "wind_level_signal",
                "solar_level_signal",
                "hydro_level_signal",
                "load_level_signal",
            ]
        )

        rows = []
        history = list(getattr(self, "_tier2_forecast_memory", []) or [])
        if not history or int(history[-1].get("step", -1)) != int(current_entry.get("step", -1)):
            history.append(dict(current_entry))
        history = history[-memory_steps:]
        for entry in history:
            row = []
            for k in channel_keys:
                v = float(entry.get(k, 0.0))
                is_signed = (
                    ("signal" in k and "disagreement" not in k)
                    or ("imbalance" in k)
                    or k.endswith("_return_signal")
                    or k.endswith("_direction_margin")
                    or ("latent_" in k and not k.endswith("_norm"))
                )
                row.append(float(np.clip(v, -1.0 if is_signed else 0.0, 1.0)))
            rows.append(row)
        while len(rows) < memory_steps:
            rows.insert(0, [0.0] * channel_count)
        return np.asarray(rows, dtype=np.float32)

    def _get_tier2_reference_exposure(self) -> float:
        """Best-effort investor exposure anchor for Tier-2 feature construction."""
        try:
            if isinstance(getattr(self, "_last_actions", None), dict):
                exec_action = self._last_actions.get("investor_0_exec", {}) or {}
                if "exposure" in exec_action:
                    return float(np.clip(float(exec_action.get("exposure", 0.0)), -1.0, 1.0))
        except Exception:
            pass

        try:
            investor_obs = np.asarray((getattr(self, "_obs_buf", {}) or {}).get("investor_0", []), dtype=np.float32).flatten()
            if investor_obs.size >= 4 and np.isfinite(float(investor_obs[3])):
                return float(np.clip(float(investor_obs[3]), -1.0, 1.0))
        except Exception:
            pass

        return 0.0

    def _build_tier2_core_features(
        self,
        step: int,
        proposed_exposure: Optional[float] = None,
        tradeable_capital: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build the nonforecast Tier-2 context core.

        This keeps the ablation scientifically clean: Tier-2 can retain state
        about the investor sleeve while every forecast signal stays confined to
        the separate ANN memory block.
        """
        step = int(max(step, 0))
        exposure_ref = self._get_tier2_reference_exposure() if proposed_exposure is None else float(proposed_exposure)
        exposure_ref = float(np.clip(exposure_ref, -1.0, 1.0))

        try:
            gross_exposure = float(sum(abs(float(v)) for v in (getattr(self, "financial_positions", {}) or {}).values()))
        except Exception:
            gross_exposure = 0.0
        capital_den = float(max(getattr(self, "init_budget", 0.0), 1.0))
        gross_exposure_ratio = float(np.clip(gross_exposure / capital_den, 0.0, 1.0))

        tradeable = float(tradeable_capital if tradeable_capital is not None else getattr(self, "_last_tradeable_capital", 0.0))
        tradeable_capital_ratio = float(np.clip(tradeable / capital_den, 0.0, 1.0))

        try:
            realized_volatility_regime = float(np.clip(self._compute_investor_realized_volatility(step), 0.0, 1.0))
        except Exception:
            realized_volatility_regime = 0.0

        local_quality = float(np.clip(getattr(self, "_investor_local_quality", 0.0), -3.0, 3.0) / 3.0)
        local_quality_delta = float(np.clip(getattr(self, "_investor_local_quality_delta", 0.0), -1.0, 1.0))
        local_drawdown = float(np.clip(getattr(self, "_investor_local_drawdown", 0.0), 0.0, 1.0))

        # DL internal self-conditioning: DISABLED.
        # Architectural fix: self-conditioning created a positive feedback
        # loop where high deployment_scale at T → high core features at T+1
        # → even higher deployment_scale.  The only successful run ("broken"
        # run with zeros) worked precisely because this loop was broken.
        # Zeroing these channels forces the model to make each decision
        # independently from market features, not from its own prior output.
        tier2_trust_radius = 0.0
        tier2_advantage_strength = 0.0
        tier2_deployment_scale = 0.0

        return np.asarray(
            [
                exposure_ref,
                abs(exposure_ref),
                gross_exposure_ratio,
                tradeable_capital_ratio,
                realized_volatility_regime,
                local_quality,
                local_quality_delta,
                local_drawdown,
                tier2_trust_radius,
                tier2_advantage_strength,
                tier2_deployment_scale,
            ],
            dtype=np.float32,
        )

    def _apply_tier2_exposure_adjustment(self, exposure: float, t: int, tradeable_capital: Optional[float] = None) -> float:
        """
        Tier-2 runtime overlay.

        Runtime is intentionally minimal:
        - build the current forecast-backed Tier-2 state
        - generate one continuous overlay action from the learned actor
        - gate and scale actions directly from learned Tier-2 heads
        - keep only hard rails: exposure clipping and route validity
        """
        exposure = float(np.clip(exposure, -1.0, 1.0))
        self._last_tier2_value_diag = self._default_tier2_value_diag()

        if not bool(getattr(self.config, "forecast_baseline_enable", False)):
            self._expected_dnav = 0.0
            return exposure
        if not bool(getattr(self.config, "tier2_runtime_overlay_enable", True)):
            self._expected_dnav = 0.0
            return exposure

        try:
            step = int(max(t, 0))
            tier2_features = self._build_tier2_features(step, proposed_exposure=exposure, tradeable_capital=tradeable_capital)
            self._active_tier2_features = np.asarray(tier2_features, dtype=np.float32).copy()
            self._active_tier2_features_step = step
            core_state = extract_tier2_core_state_from_features(
                tier2_features,
                core_dim=TIER2_VALUE_CORE_DIM,
            )
            memory_state = self._extract_tier2_short_memory_state(tier2_features)
            context_strength = float(np.clip(memory_state.get("context_strength", 0.0), 0.0, 1.0))
            investor_obs = np.asarray((getattr(self, "_obs_buf", {}) or {}).get("investor_0", []), dtype=np.float32).flatten()
            delta_mean = 0.0
            confidence_mean = 0.5
            value_mean = 0.0
            value_sigma = 0.0
            tier2_components = {}
            runtime_bundle = {}
            adapter = getattr(self, "tier2_value_adapter", None)
            if adapter is not None and investor_obs.size > 0:
                try:
                    mc_samples = max(1, int(getattr(self.config, "tier2_runtime_mc_samples", 7) or 7))
                    runtime_bundle = adapter.predict_runtime_bundle(
                        investor_obs,
                        tier2_features,
                        samples=mc_samples,
                    )
                    delta_mean = float(runtime_bundle.get("selected_delta", 0.0))
                    confidence_mean = float(
                        runtime_bundle.get(
                            "selected_intervene_probability",
                            runtime_bundle.get("selected_probability", 0.0),
                        )
                    )
                    value_mean = float(runtime_bundle.get("selected_value_mean", 0.0))
                    value_sigma = float(runtime_bundle.get("selected_value_std", 0.0))
                    tier2_components = dict(runtime_bundle.get("components", {}) or {})
                except Exception as runtime_bundle_err:
                    self._log_tier2_runtime_issue("RUNTIME_BUNDLE_ERROR", step, runtime_bundle_err)
                    raise RuntimeError(
                        f"Tier-2 runtime bundle failed at step {step}"
                    ) from runtime_bundle_err

            sigma_scale = float(max(getattr(self.config, "tier2_runtime_sigma_scale", 0.015), 1e-6))
            route_uncertainty_quality = float(
                np.clip(
                    tier2_components.get(
                        "route_uncertainty_quality",
                        tier2_components.get("controller_confidence", 1.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            sigma_quality = float(np.clip(1.0 / (1.0 + (max(value_sigma, 0.0) / sigma_scale)), 0.0, 1.0))
            learned_delta = float(np.clip(delta_mean, -1.0, 1.0))
            learned_confidence = float(np.clip(confidence_mean, 0.0, 1.0))
            predicted_value = float(
                tier2_components.get(
                    "selected_value_prediction",
                    tier2_components.get("value_prediction", value_mean if np.isfinite(value_mean) else 0.0),
                )
            )
            full_value_prediction = float(tier2_components.get("full_value_prediction", predicted_value))
            ablated_value_prediction = float(tier2_components.get("ablated_value_prediction", 0.0))
            uplift_value_prediction = float(tier2_components.get("uplift_value_prediction", predicted_value))
            predicted_value_sigma = float(max(value_sigma, 0.0))
            uncertainty_quality = float(np.clip(sigma_quality * route_uncertainty_quality, 0.0, 1.0))
            runtime_forecast_trust = float(np.clip(self.get_fgb_trust_for_agent("investor_0"), 0.0, 1.0))
            calibrated_forecast_trust = float(np.clip(self._get_tier2_expert_confidence(), 0.0, 1.0))
            trust_component = float(np.clip(0.5 * runtime_forecast_trust + 0.5 * calibrated_forecast_trust, 0.0, 1.0))
            consensus_signal = float(np.clip(memory_state.get("expert_consensus_signal", 0.0), -1.0, 1.0))
            disagreement_signal = float(np.clip(memory_state.get("expert_disagreement", 0.0), 0.0, 1.0))
            primary_signal = float(
                np.clip(
                    memory_state.get(
                        f"{TIER2_PRIMARY_EXPERT_NAME}_signal",
                        memory_state.get("primary_signal", tier2_components.get("direction_signal", consensus_signal)),
                    ),
                    -1.0,
                    1.0,
                )
            )
            primary_quality_gap = float(
                np.clip(
                    memory_state.get(
                        f"{TIER2_PRIMARY_EXPERT_NAME}_quality_gap",
                        memory_state.get("primary_quality_gap", tier2_components.get("primary_quality_gap", 0.0)),
                    ),
                    -1.0,
                    1.0,
                )
            )
            primary_risk_gap = float(
                np.clip(
                    memory_state.get(
                        f"{TIER2_PRIMARY_EXPERT_NAME}_risk_gap",
                        memory_state.get("primary_risk_gap", tier2_components.get("primary_risk_gap", 0.0)),
                    ),
                    -1.0,
                    1.0,
                )
            )
            primary_vs_consensus_gap = float(
                np.clip(
                    memory_state.get(
                        f"{TIER2_PRIMARY_EXPERT_NAME}_vs_consensus_gap",
                        memory_state.get("primary_vs_consensus_gap", tier2_components.get("primary_vs_consensus_gap", 0.0)),
                    ),
                    -1.0,
                    1.0,
                )
            )
            primary_leadership = float(
                np.clip(
                    memory_state.get(
                        f"{TIER2_PRIMARY_EXPERT_NAME}_leadership",
                        memory_state.get("primary_leadership", tier2_components.get("primary_leadership", 0.0)),
                    ),
                    0.0,
                    1.0,
                )
            )
            edge_signal = float(np.clip(primary_signal * (1.0 - 0.5 * disagreement_signal), -1.0, 1.0))
            curvature_signal = float(np.clip(disagreement_signal, 0.0, 1.0))
            forecast_signal = primary_signal
            imbalance_signal = float(np.clip(memory_state.get("short_imbalance_signal", 0.0), -1.0, 1.0))
            mape_component = float(np.clip(self._get_tier2_short_expert_mape_quality(TIER2_PRIMARY_EXPERT_NAME), 0.0, 1.0))
            metadata_skill = float(np.clip(self._get_tier2_short_expert_metadata_skill(TIER2_PRIMARY_EXPERT_NAME), 0.0, 1.0))
            magnitude_prediction = float(
                np.clip(tier2_components.get("magnitude_prediction", abs(learned_delta)), 0.0, 1.0)
            )
            dl_direction_signal = float(np.clip(tier2_components.get("direction_signal", primary_signal), -1.0, 1.0))
            predicted_value_lcb = float(
                runtime_bundle.get(
                    "selected_value_lcb",
                    tier2_components.get("selected_value_lcb", predicted_value),
                )
            )
            sharpe_before = float(np.clip(getattr(self, "_investor_local_quality", 0.0), -3.0, 3.0))
            unit_return_scale = float(max(getattr(self.config, "tier2_value_target_scale", 0.01), 1e-6))
            delta_max = float(max(getattr(self.config, "tier2_value_delta_max", 0.20), 1e-6))
            runtime_delta_scale = float(
                np.clip(
                    getattr(self.config, "tier2_runtime_delta_scale", 1.0) or 1.0,
                    0.0,
                    1.0,
                )
            )
            candidate_target_exposure = float(
                np.clip(runtime_bundle.get("target_exposure", exposure + learned_delta), -1.0, 1.0)
            )
            candidate_delta = float(np.clip(candidate_target_exposure - exposure, -delta_max, delta_max))
            if runtime_delta_scale < 0.999999:
                candidate_delta = float(np.clip(candidate_delta * runtime_delta_scale, -delta_max, delta_max))
                candidate_target_exposure = float(np.clip(exposure + candidate_delta, -1.0, 1.0))
            learned_gate_score = float(
                np.clip(
                    tier2_components.get(
                        "gate_head_probability",
                        tier2_components.get("learned_gate_head_probability",
                            tier2_components.get("policy_scale",
                                tier2_components.get("learned_gate_score",
                                    tier2_components.get("projection_strength", learned_confidence)))),
                    ),
                    0.0,
                    1.0,
                )
            )
            projected_confidence = float(np.clip(learned_gate_score, 0.0, 1.0))
            adjusted_exposure_candidate = float(np.clip(candidate_target_exposure, -1.0, 1.0))
            executable_candidate_delta = float(adjusted_exposure_candidate - exposure)
            nav_prediction = float(
                tier2_components.get("nav_preservation_prediction", tier2_components.get("nav_prediction", 0.0))
            )
            nav_certified_lcb = float(
                tier2_components.get(
                    "nav_preservation_certified_lcb",
                    tier2_components.get("nav_preservation_lcb", tier2_components.get("nav_prediction", 0.0)),
                )
            )
            return_floor_certified_lcb = float(
                tier2_components.get(
                    "return_floor_certified_lcb",
                    tier2_components.get("return_floor_lcb", tier2_components.get("return_floor_prediction", 0.0)),
                )
            )
            return_floor_prediction = float(
                tier2_components.get("return_floor_prediction", return_floor_certified_lcb)
            )
            value_gate = float(np.clip(tier2_components.get("value_support", 0.0), 0.0, 1.0))
            nav_gate = float(
                np.clip(
                    1.0 / (1.0 + np.exp(-nav_certified_lcb / max(unit_return_scale, 1e-6))),
                    0.0,
                    1.0,
                )
            )
            return_floor_gate = float(np.clip(tier2_components.get("floor_support", 0.0), 0.0, 1.0))
            delta_gate = bool(
                np.isfinite(executable_candidate_delta)
                and abs(executable_candidate_delta) >= float(max(getattr(self.config, "tier2_runtime_min_abs_delta", 0.0) or 0.0, 0.0))
            )
            trust_gate = True
            structural_score = float(np.clip(projected_confidence, 0.0, 1.0))
            structural_gate = True
            value_lcb_gate = True
            nav_lcb_gate = True
            return_floor_lcb_gate = True
            no_harm_gate = True
            learned_gate = bool(projected_confidence >= 0.50)
            overlay_allowed = bool(delta_gate and learned_gate)
            overlay_gain = float(projected_confidence if overlay_allowed else 0.0)
            predicted_decision_improvement = float(uplift_value_prediction)
            certified_override_gain = float(predicted_value * overlay_gain)
            predicted_sharpe_delta = float(
                np.clip(predicted_value / unit_return_scale, -1.0, 1.0)
                * min(abs(executable_candidate_delta) / max(delta_max, 1e-6), 1.0)
            )
            risk_scale = float(np.clip(tier2_components.get("gate_head_probability", tier2_components.get("policy_scale", projected_confidence)), 0.0, 1.0))
            if overlay_allowed:
                delta = float(np.clip(executable_candidate_delta, -delta_max, delta_max))
                adjusted_exposure = float(np.clip(exposure + delta, -1.0, 1.0))
            else:
                delta = 0.0
                adjusted_exposure = float(exposure)
            overlay_enabled = 1.0 if (overlay_allowed and abs(delta) > 1e-9) else 0.0

            tradeable = float(tradeable_capital if tradeable_capital is not None else getattr(self, "_last_tradeable_capital", 0.0))
            tradeable = float(max(tradeable, 1.0))
            max_position_size = float(max(getattr(self.config, "max_position_size", 0.35) or 0.35, 0.0))
            self._expected_dnav = float(tradeable * max_position_size * max(predicted_value, 0.0))

            sharpe_after = float(np.clip(sharpe_before + predicted_sharpe_delta, -3.0, 3.0))
            alignment_metric = float(np.clip(np.sign(delta * primary_signal), -1.0, 1.0)) if overlay_enabled > 0.0 else 0.0
            realized_gain_signal = float(
                getattr(self, "last_realized_investor_dnav_return", 0.0) if overlay_enabled > 0.0 else 0.0
            )

            self._last_tier2_value_diag = {
                **self._default_tier2_value_diag(),
                "enabled": float(overlay_enabled),
                "delta": float(delta),
                "pred_sigma": float(max(predicted_value_sigma, 0.0)),
                "realized_gain_signal": realized_gain_signal,
                "target_delta": float(candidate_delta),
                "sharpe_signal": float(np.clip(predicted_value_lcb, -1.0, 1.0)),
                "sharpe_before": float(sharpe_before),
                "sharpe_after": float(sharpe_after),
                "sharpe_delta": float(sharpe_after - sharpe_before),
                "reliability": float(projected_confidence),
                "overlay_gain": float(overlay_gain),
                "forecast_trust": runtime_forecast_trust,
                "calibrated_forecast_trust": calibrated_forecast_trust,
                "trust_component": trust_component,
                "mape_component": mape_component,
                "uncertainty_quality": float(uncertainty_quality),
                "metadata_skill": metadata_skill,
                "residual_bias_component": 0.0,
                "residual_dispersion_quality_component": 0.0,
                "residual_sign_bias_component": 0.0,
                "conformal_residual_risk": float(np.clip(tier2_components.get("conformal_residual_risk", 0.0), 0.0, 1.0)),
                "dl_component": float(predicted_value),
                "controller_confidence": float(projected_confidence),
                "alignment": float(np.clip(alignment_metric, -1.0, 1.0)),
                "forecast_signal": float(forecast_signal),
                "forecast_edge_signal": float(edge_signal),
                "forecast_consensus_signal": float(consensus_signal),
                "forecast_curvature_signal": float(curvature_signal),
                "imbalance_signal": float(imbalance_signal),
                "physical_pressure_signal": float(imbalance_signal),  # Backward-compatible alias
                "context_strength": float(context_strength),
                "action_sign": float(np.sign(adjusted_exposure)) if abs(adjusted_exposure) > 1e-9 else 0.0,
                "forecast_sign": float(np.sign(forecast_signal)) if abs(forecast_signal) > 1e-9 else 0.0,
                "temporal_direction_signal": float(dl_direction_signal),
                "temporal_structural_signal": float(learned_delta),
                "certified_override_gain": float(certified_override_gain),
                "decision_improvement": float(predicted_decision_improvement),
                "value_prediction": float(predicted_value),
                "value_pred_sigma": float(predicted_value_sigma),
                "value_lcb": float(predicted_value_lcb),
                "value_ucb": float(runtime_bundle.get("selected_value_ucb", predicted_value)),
                "nav_lcb": float(nav_certified_lcb),
                "return_floor_lcb": float(return_floor_certified_lcb),
                "overlay_confidence": float(projected_confidence),
                "projection_strength": float(tier2_components.get("projection_strength", projected_confidence)),
                "nav_prediction": float(tier2_components.get("nav_preservation_prediction", 0.0)),
                "return_floor_prediction": float(tier2_components.get("return_floor_prediction", 0.0)),
                "tail_risk_prediction": float(tier2_components.get("tail_risk_prediction", 0.0)),
                "value_gate": float(value_gate),
                "nav_gate": float(nav_gate),
                "return_floor_gate": float(return_floor_gate),
                "delta_gate": float(delta_gate),
                "no_harm_gate": float(no_harm_gate),
                "value_lcb_gate": float(value_lcb_gate),
                "nav_lcb_gate": float(nav_lcb_gate),
                "return_floor_lcb_gate": float(return_floor_lcb_gate),
                "structural_score": float(structural_score),
                "structural_gate": float(structural_gate),
                "learned_gate": float(learned_gate_score),
                "learned_gate_pass": float(learned_gate),
                "trust_gate": float(trust_gate),
                "risk_scale": float(risk_scale),
                "runtime_forecast_trust": runtime_forecast_trust,
                "factor_reward_prediction": float(tier2_components.get("factor_opportunity", 0.0)),
                "reference_quality_gap": float(primary_quality_gap),
                "reference_risk_gap": float(primary_risk_gap),
                "reference_vs_consensus_gap": float(primary_vs_consensus_gap),
                "reference_leadership": float(primary_leadership),
                "delta_prediction": float(candidate_delta),
                "residual_prediction": float(uplift_value_prediction),
                "dominant_expert_weight": float(tier2_components.get("dominant_expert_weight", 0.0)),
                "expert_weight_entropy": float(tier2_components.get("expert_weight_entropy", 0.0)),
                "internal_base_weight": float(tier2_components.get("internal_base_weight", 0.0)),
                "internal_trend_weight": float(tier2_components.get("internal_trend_weight", 0.0)),
                "internal_reversion_weight": float(tier2_components.get("internal_reversion_weight", 0.0)),
                "internal_defensive_weight": float(tier2_components.get("internal_defensive_weight", 0.0)),
                # ACRP-v3 diagnostics from DL model
                "acrp_learned_trust_radius": float(tier2_components.get("acrp_learned_trust_radius", 0.0)),
                "acrp_advantage_strength": float(tier2_components.get("acrp_advantage_strength", 0.0)),
                "acrp_da_learned_scalar": float(tier2_components.get("acrp_da_learned_scalar", 0.0)),
                "acrp_attn_gate_weight": float(tier2_components.get("acrp_attn_gate_weight", 0.0)),
                "acrp_attn_conf_weight": float(tier2_components.get("acrp_attn_conf_weight", 0.0)),
                "acrp_attn_outlook_weight": float(tier2_components.get("acrp_attn_outlook_weight", 0.0)),
                "policy_scale": float(tier2_components.get("policy_scale", 0.0)),
                "deployment_strength_mean": float(tier2_components.get("policy_scale", 0.0)),
            }
            return adjusted_exposure
        except Exception as tier2_err:
            self._log_tier2_runtime_issue("RUNTIME_ERROR", t, tier2_err)
            raise RuntimeError(
                f"Tier-2 policy-improvement layer failed at step {int(max(t, 0))}"
            ) from tier2_err

    def _build_tier2_features(
        self,
        i: int,
        proposed_exposure: Optional[float] = None,
        tradeable_capital: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build the Tier-2 temporal state used by the learned policy-improvement layer.

        Layout:
        - 8D investor-state core:
          proposed_exposure, abs_proposed_exposure, gross_exposure_ratio,
          tradeable_capital_ratio, realized_volatility, investor_local_quality,
          investor_local_quality_delta, investor_local_drawdown
        - flattened short-memory block:
          8 steps x 21 channels =
          raw ANN forecast state plus raw market context
        """
        TIER2_FEATURE_DIM = int(TIER2_VALUE_FEATURE_DIM)
        if not bool(getattr(self.config, "forecast_baseline_enable", False)):
            return np.zeros(TIER2_FEATURE_DIM, dtype=np.float32)
        try:
            step = int(max(i, 0))
            if int(getattr(self, "_tier2_feature_cache_step", -1)) != step:
                self._tier2_feature_cache_step = step
                self._tier2_feature_cache = {}
            exposure_ref = self._get_tier2_reference_exposure() if proposed_exposure is None else float(proposed_exposure)
            tradeable_key = None if tradeable_capital is None else round(float(tradeable_capital), 4)
            ablate_forecast_features = bool(
                getattr(self.config, "tier2_value_ablate_forecast_features", False)
            )
            cache_key = (
                round(float(exposure_ref), 6),
                tradeable_key,
                ablate_forecast_features,
            )
            cache = getattr(self, "_tier2_feature_cache", {})
            cached_features = cache.get(cache_key)
            if cached_features is not None:
                return np.asarray(cached_features, dtype=np.float32).copy()
            core = self._build_tier2_core_features(
                step,
                proposed_exposure=exposure_ref,
                tradeable_capital=tradeable_capital,
            )
            memory_width = int(TIER2_VALUE_MEMORY_STEPS * TIER2_VALUE_MEMORY_CHANNELS)
            memory_matrix = np.asarray(
                self._get_tier2_forecast_memory_matrix(step, proposed_exposure=exposure_ref),
                dtype=np.float32,
            ).copy()
            if ablate_forecast_features:
                # Full forecast-memory ablation: keep the Tier-2 controller and
                # core investor state identical, but neutralize the entire
                # short-horizon forecast memory so the ablated control path is
                # a true no-forecast Tier-2 baseline.
                memory_matrix[...] = 0.0
            memory_flat = np.asarray(memory_matrix, dtype=np.float32).reshape(-1)
            if memory_flat.size < memory_width:
                pad = np.zeros(memory_width - memory_flat.size, dtype=np.float32)
                memory_flat = np.concatenate([memory_flat, pad], axis=0)
            elif memory_flat.size > memory_width:
                memory_flat = memory_flat[:memory_width]
            features = np.concatenate([core, memory_flat.astype(np.float32)], axis=0).astype(np.float32)
            cache[cache_key] = features.copy()
            self._tier2_feature_cache = cache
            return features.copy()
        except Exception:
            return np.zeros(TIER2_FEATURE_DIM, dtype=np.float32)

    def get_fgb_trust_for_agent(self, agent_name: str) -> float:
        """
        Return per-agent forecast trust (tau_t) for diagnostics/calibration logging.

        IMPORTANT:
        - This does NOT change environment observations or rewards (Tier1 remains intact).
        - The Tier-2 forecast-fusion path uses tau_t only as backend calibration context.
        """
        name = str(agent_name or "").strip()

        # Cache per env step so multiple agents see a consistent value (no double-smoothing).
        try:
            step_key = int(getattr(self.config, "training_global_step", getattr(self, "t", 0))) if self.config is not None else int(getattr(self, "t", 0))
        except Exception:
            step_key = int(getattr(self, "t", 0))

        if not hasattr(self, "_fgb_trust_cache_step"):
            self._fgb_trust_cache_step = None
            self._fgb_trust_cache: Dict[str, float] = {}
            self._fgb_trust_prev: Dict[str, float] = {}
            self._fgb_trust_horizon_cache: Dict[str, str] = {}

        if self._fgb_trust_cache_step != step_key:
            self._fgb_trust_cache_step = step_key
            self._fgb_trust_cache = {}
            self._fgb_trust_horizon_cache = {}

        if name in self._fgb_trust_cache:
            return float(self._fgb_trust_cache[name])

        if bool(getattr(self.config, "forecast_price_short_expert_only", False)):
            horizon = "short"
        else:
            horizon = self._resolve_fgb_trust_horizon(name, default_pref="auto", prefer_cached=False)

        # Compute trust.
        trust_val: float
        tracker = getattr(self, "calibration_tracker", None)
        if tracker is not None:
            # Use CalibrationTracker trust for the selected horizon when available.
            recent_mape = None
            try:
                mape_hist = list(getattr(self, "_horizon_mape", {}).get(horizon, []))[-10:]
                mape_hist = [float(x) for x in mape_hist if np.isfinite(x)]
                if len(mape_hist) > 0:
                    recent_mape = float(np.mean(mape_hist))
            except Exception:
                recent_mape = None
            try:
                trust_val = float(tracker.get_trust(horizon=horizon, recent_mape=recent_mape))
            except Exception:
                trust_val = float(getattr(self, "_forecast_trust", 0.5))
        else:
            # No tracker available: fall back to env directional hit-rate / MAPE stats.
            metric = str(getattr(self.config, "forecast_trust_metric", "hitrate") or "hitrate").strip().lower() if self.config is not None else "hitrate"

            def _mean_last_hit(h: str, n: int = 50) -> float:
                try:
                    xs = list(getattr(self, "_horizon_sign_hit", {}).get(h, []))[-n:]
                    xs = [float(x) for x in xs if np.isfinite(x)]
                    return float(np.mean(xs)) if len(xs) > 0 else 0.5
                except Exception:
                    return 0.5

            def _mean_last_mape(h: str, n: int = 10) -> float:
                try:
                    xs = list(getattr(self, "_horizon_mape", {}).get(h, []))[-n:]
                    xs = [float(x) for x in xs if np.isfinite(x)]
                    if len(xs) > 0:
                        return float(np.mean(xs))
                except Exception:
                    pass
                try:
                    return float(getattr(self, "_mape_thresholds", {}).get(h, 0.12))
                except Exception:
                    return 0.12

            if metric in ("hitrate", "absdir", "sign", "direction"):
                trust_target = float(np.clip(_mean_last_hit(horizon, 50), 0.0, 1.0))
            elif metric == "combo":
                # Minimal combo: mostly direction, lightly penalize high MAPE.
                hit = float(np.clip(_mean_last_hit(horizon, 50), 0.0, 1.0))
                mape = float(np.clip(_mean_last_mape(horizon, 10), 0.0, 2.0))
                mape_factor = max(0.0, 1.0 - min(mape / 2.0, 1.0))
                trust_target = float(np.clip(0.8 * hit + 0.2 * mape_factor, 0.0, 1.0))
            else:
                # MAPE-only fallback: map low error -> high trust.
                mape = float(_mean_last_mape(horizon, 10))
                thr = float(getattr(self, "_mape_thresholds", {}).get(horizon, 0.12))
                trust_target = float(np.clip(1.0 - (mape / max(thr, 1e-6)), 0.0, 1.0))

            prev = float(self._fgb_trust_prev.get(name, 0.5))
            trust_val = float(np.clip(0.7 * trust_target + 0.3 * prev, 0.0, 1.0))
            self._fgb_trust_prev[name] = trust_val

        self._fgb_trust_cache[name] = float(np.clip(trust_val, 0.0, 1.0))
        self._fgb_trust_horizon_cache[name] = str(horizon)
        return float(self._fgb_trust_cache[name])

    def _should_apply_forecast_trust_floor(self, agent_name: str = "investor_0") -> bool:
        """
        Apply the trust floor only when forecast quality has explicitly cleared the
        configured horizon checks. This prevents the floor from forcing the baseline
        on during weak / low-information regimes.
        """
        try:
            trust_floor = float(getattr(self.config, "forecast_trust_floor", 0.0) or 0.0)
        except Exception:
            trust_floor = 0.0
        if trust_floor <= 0.0:
            return False

        try:
            min_samples = int(getattr(self.config, "fgb_trust_min_samples", 50) or 50)
        except Exception:
            min_samples = 50
        min_samples = max(10, min_samples)
        if bool(getattr(self.config, "forecast_price_short_expert_only", False)):
            horizon = "short"
        else:
            horizon = self._resolve_fgb_trust_horizon(agent_name, default_pref="short", prefer_cached=True)

        try:
            hit_hist = list((getattr(self, "_horizon_sign_hit", {}) or {}).get(horizon, []))
        except Exception:
            hit_hist = []
        try:
            mape_hist = list((getattr(self, "_horizon_mape", {}) or {}).get(horizon, []))
        except Exception:
            mape_hist = []

        hit_vals = [float(x) for x in hit_hist[-min_samples:] if np.isfinite(x)]
        mape_vals = [float(x) for x in mape_hist[-min_samples:] if np.isfinite(x)]
        if len(hit_vals) < min_samples or len(mape_vals) < min_samples:
            return False

        recent_hit = float(np.mean(hit_vals))
        recent_mape = float(np.mean(mape_vals[-10:])) if len(mape_vals) > 0 else float("inf")
        try:
            threshold = float((getattr(self, "_mape_thresholds", {}) or {}).get(horizon, 0.12))
        except Exception:
            threshold = 0.12
        if not np.isfinite(threshold) or threshold <= 0.0:
            threshold = 0.12

        # Explicit quality checks:
        # - directional hit-rate must clear a modest positive edge, not just random
        # - recent MAPE must remain within the configured acceptable band
        return bool(np.isfinite(recent_hit) and np.isfinite(recent_mape) and recent_hit >= 0.52 and recent_mape <= threshold)

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
                        int(getattr(self, '_tier2_prev_investor_decision_step', i - fallback_horizon_steps)),
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
            shared_tier2_features = (
                np.asarray(self._active_tier2_features, dtype=np.float32).copy()
                if int(getattr(self, '_active_tier2_features_step', -1)) == int(i)
                and getattr(self, '_active_tier2_features', None) is not None
                else self._build_tier2_features(i)
            )
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

                    # Legacy compatibility
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
                    'forecast_signal_score': financial.get('forecast_signal_score', 0.0),
                    'price_return_1step': float(price_return_1step),
                    'price_return_invfreq': float(price_return_invfreq),
                    'decision_step': float(decision_step),

                    # Forecast backend signals
                    'forecast_trust': float(self._forecast_trust),  # τₜ ∈ [0,1]
                    'expected_dnav': float(self._expected_dnav),    # E[ΔNAV] for next step
                    # Tier-2 compact forecast state for the learned forecast-fusion overlay.
                    # Keep both:
                    # - realized_investor_return: executed sleeve PnL return (for diagnostics/reward analysis)
                    # - tier2_training_return: exposure-independent market return aligned to the
                    #   investor decision cadence for Tier-2 label generation
                    'realized_investor_return': float(getattr(self, 'last_realized_investor_dnav_return', 0.0)),
                    'tier2_training_return': float(
                        price_return_invfreq if np.isfinite(price_return_invfreq) else price_return_1step
                    ),
                    'tier2_training_horizon_steps': float(decision_horizon_steps),
                    'executed_investor_exposure': executed_investor_exposure,
                    'tier2_features': shared_tier2_features,
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
        self.battery_capacity = float(value)

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
