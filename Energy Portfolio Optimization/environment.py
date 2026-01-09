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
from risk import ForecastRiskManager  # NEW: Risk management using forecasts
from forecast_engine import ForecastEngine
from config import (
    normalize_price,
    OVERLAY_FEATURE_DIM,
    ENV_MARKET_STRESS_DEFAULT,
    ENV_OVERALL_RISK_DEFAULT,
    ENV_MARKET_RISK_DEFAULT,
    ENV_POSITION_EXPOSURE_THRESHOLD,
    ENV_EXPLORATION_BONUS_MULTIPLIER,
)  # UNIFIED: Import from single source of truth
from logger import RewardLogger, get_logger  # Step-by-step logging for Tier comparison

# Centralized logging - ALL logging goes through logger.py
logger = get_logger(__name__)


# =============================================================================
# Observation specs (BASE-dim only; wrapper appends forecasts)
# =============================================================================
class StabilizedObservationManager:
    def __init__(self, env: 'RenewableMultiAgentEnv'):
        self.env = env
        self.config = env.config if hasattr(env, 'config') else None
        self.observation_specs = self._build_specs()
        self.base_spaces = self._build_spaces()

    def _build_specs(self) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}

        # PHASE 1 IMPLEMENTATION: Conditional dimensions based on forecast utilisation
        # Tier 1 (Baseline MARL): No forecasts
        # Tier 2 (With Forecasts): Investor gets price forecasts, Battery gets generation+price forecasts
        enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False) if self.config else False

        if enable_forecast_util:
            # TIER 2: Forecasts enabled
            # TEMPORAL ALIGNMENT + ENGINEERED FEATURES + TRADE SIGNAL: Investor: 14D = 6 base + 8 forecast
            # Forecast features: z_short, z_medium_lagged, direction, momentum, strength, trust, normalized_error, trade_signal
            specs["investor_0"] = {"base": 14}

            # Battery: 10D = 4 base + 3 separate generation (wind, solar, hydro) + 3 price (all horizons)
            # CHANGE: Use separate generation signals to preserve asset-specific patterns and correlations
            specs["battery_operator_0"] = {"base": 10}

            # Risk Controller: 12D = 9 base + 3 forecast signals
            # NEW: Added price trend, volatility forecast, and trust
            specs["risk_controller_0"] = {"base": 12}

            # Meta Controller: 13D = 11 base + 2 forecast signals
            # NEW: Added trust and expected return
            specs["meta_controller_0"] = {"base": 13}
        else:
            # TIER 1: Baseline MARL (no forecasts)
            # Investor: 6D = price, budget, wind_pos, solar_pos, hydro_pos, mtm_pnl
            specs["investor_0"] = {"base": 6}

            # Battery: 4D = price, energy, capacity, load
            specs["battery_operator_0"] = {"base": 4}

            # Risk Controller: 9D = regime cues + positions + knobs
            specs["risk_controller_0"] = {"base": 9}

            # Meta Controller: 11D = budget, positions, price_n, risks, perf, knobs
            specs["meta_controller_0"] = {"base": 11}

        return specs

    def _build_spaces(self) -> Dict[str, spaces.Box]:
        sp: Dict[str, spaces.Box] = {}

        enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False) if self.config else False

        # CONDITIONAL: investor_0 space depends on forecast utilisation
        if enable_forecast_util:
            # TIER 22: Full forecast features (14D = 6 base + 8 forecast)
            # Base (6D): price, budget, wind_pos, solar_pos, hydro_pos, mtm_pnl
            # Forecast (8D): z_short, z_medium_lagged, direction, momentum, strength, forecast_trust, normalized_error, trade_signal
            inv_low  = np.array([
                -1.0,  # price_current (normalized, can be negative)
                0.0,   # budget_normalized (always positive, [0, 1])
                -1.0,  # wind_position_norm (can be short)
                -1.0,  # solar_position_norm (can be short)
                -1.0,  # hydro_position_norm (can be short)
                -1.0,  # mtm_pnl_norm (can be negative)
                -1.0,  # z_short_price (short-term price forecast z-score)
                -1.0,  # z_medium_lagged (medium-term price forecast, temporally aligned)
                -1.0,  # direction (sign of forecast signal: -1, 0, +1)
                -1.0,  # momentum (change in forecast signal)
                0.0,   # strength (absolute magnitude of forecast signal)
                0.0,   # forecast_trust (forecast trust score)
                0.0,   # normalized_error (recent forecast error normalized by threshold)
                -1.0   # trade_signal (direction * strength * trust: composite actionable signal, [-1, 1])
            ], dtype=np.float32)
            inv_high = np.array([
                1.0,   # price_current
                1.0,   # budget_normalized
                1.0,   # wind_position_norm
                1.0,   # solar_position_norm
                1.0,   # hydro_position_norm
                1.0,   # mtm_pnl_norm
                1.0,   # z_short_price
                1.0,   # z_medium_lagged
                1.0,   # direction
                1.0,   # momentum
                1.0,   # strength
                1.0,   # forecast_trust
                1.0,   # normalized_error (0.0 = perfect, 1.0 = at threshold or worse) - CRITICAL FIX: Changed from 2.0 to 1.0 to match observation scale
                1.0    # trade_signal
            ], dtype=np.float32)
            inv_shape = (14,)
        else:
            # Tier 1: 6D baseline observations (same structure as Tier 2, just no forecasts)
            # Base (6D): price, budget, wind_pos, solar_pos, hydro_pos, mtm_pnl
            inv_low  = np.array([
                -1.0,  # price_current
                0.0,   # budget_normalized
                -1.0,  # wind_position_norm
                -1.0,  # solar_position_norm
                -1.0,  # hydro_position_norm
                -1.0   # mtm_pnl_norm
            ], dtype=np.float32)
            inv_high = np.array([
                1.0,   # price_current
                1.0,   # budget_normalized (normalized to [0, 1])
                1.0,   # wind_position_norm
                1.0,   # solar_position_norm
                1.0,   # hydro_position_norm
                1.0    # mtm_pnl_norm
            ], dtype=np.float32)
            inv_shape = (6,)

        # CONDITIONAL: battery_operator_0 space depends on forecast utilisation
        if enable_forecast_util:
            # TIER 2: Battery with SEPARATE generation forecasts (10D)
            # Base (4D): price, energy, capacity, load
            # Generation forecasts (3D): z_short_wind, z_short_solar, z_short_hydro (SEPARATE signals)
            # Price forecasts (3D): z_short, z_medium, z_long (ALL HORIZONS)
            # CHANGE: Use separate generation signals instead of merged total to preserve asset-specific patterns
            bat_low  = np.array([
                -1.0,  # price_current
                0.0,   # battery_soc_normalized (NEW: changed from absolute energy to normalized SOC)
                0.0,   # battery_capacity
                0.0,   # load_current
                -1.0,  # z_short_wind (separate wind generation forecast)
                -1.0,  # z_short_solar (separate solar generation forecast)
                -1.0,  # z_short_hydro (separate hydro generation forecast)
                -1.0,  # z_short_price (immediate arbitrage)
                -1.0,  # z_medium_price (4-hour planning)
                -1.0   # z_long_price (day-ahead planning)
            ], dtype=np.float32)
            bat_high = np.array([
                1.0,   # price_current
                1.0,   # battery_soc_normalized (NEW: normalized SOC [0, 1])
                10.0,  # battery_capacity
                1.0,   # load_current
                1.0,   # z_short_wind
                1.0,   # z_short_solar
                1.0,   # z_short_hydro
                1.0,   # z_short_price
                1.0,   # z_medium_price
                1.0    # z_long_price
            ], dtype=np.float32)
            bat_shape = (10,)
        else:
            # TIER 1: Battery baseline (4D)
            bat_low  = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # price, soc_normalized, capacity, load
            bat_high = np.array([ 1.0, 1.0, 10.0, 1.0], dtype=np.float32)  # soc_normalized is [0, 1] not [0, 10]
            bat_shape = (4,)

        # CONDITIONAL: risk_controller_0 space depends on forecast utilisation
        if enable_forecast_util:
            # TIER 2: Risk controller with forecast signals (12D)
            # Base (9D): price_n, vol, stress, positions (3), cap_frac, equity, risk_mult
            # Forecast (3D): z_short_price, price_volatility_forecast, forecast_trust
            risk_low  = np.array([
                -1.0,  # price_n
                0.0, 0.0,  # vol, stress
                0.0, 0.0, 0.0,  # wind_pos, solar_pos, hydro_pos
                0.0, 0.0, 0.0,  # cap_frac, equity, risk_mult
                -1.0,  # z_short_price (price trend)
                0.0,   # price_volatility_forecast (expected volatility)
                0.0    # forecast_trust (forecast quality)
            ], dtype=np.float32)
            risk_high = np.array([
                1.0,  # price_n
                10.0, 10.0,  # vol, stress
                10.0, 10.0, 10.0,  # wind_pos, solar_pos, hydro_pos
                10.0, 10.0, 10.0,  # cap_frac, equity, risk_mult
                1.0,   # z_short_price
                10.0,  # price_volatility_forecast
                1.0    # forecast_trust
            ], dtype=np.float32)
            risk_shape = (12,)
        else:
            # TIER 1: Risk controller baseline (9D)
            risk_low  = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            risk_high = np.array([ 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
            risk_shape = (9,)

        # CONDITIONAL: meta_controller_0 space depends on forecast utilisation
        if enable_forecast_util:
            # TIER 2: Meta controller with forecast signals (13D)
            # Base (11D): budget, positions (3), price_n, risks (4), cap_frac
            # Forecast (2D): forecast_trust, expected_return
            meta_low  = np.array([
                0.0, 0.0, 0.0, 0.0,  # budget, positions (3)
                -1.0,  # price_n
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # risks (4), perf, cap_frac
                0.0,   # forecast_trust
                -1.0   # expected_return (can be negative)
            ], dtype=np.float32)
            meta_high = np.array([
                1.0, 1.0, 1.0, 1.0,  # budget, positions (3)
                1.0,  # price_n
                1.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # risks (4), perf, cap_frac
                1.0,   # forecast_trust
                1.0    # expected_return
            ], dtype=np.float32)
            meta_shape = (13,)
        else:
            # TIER 1: Meta controller baseline (11D)
            meta_low  = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            meta_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
            meta_shape = (11,)

        sp["investor_0"]         = spaces.Box(low=inv_low,  high=inv_high,  shape=inv_shape,  dtype=np.float32)
        sp["battery_operator_0"] = spaces.Box(low=bat_low,  high=bat_high,  shape=bat_shape,  dtype=np.float32)
        sp["risk_controller_0"]  = spaces.Box(low=risk_low, high=risk_high, shape=risk_shape,  dtype=np.float32)
        sp["meta_controller_0"]  = spaces.Box(low=meta_low, high=meta_high, shape=meta_shape, dtype=np.float32)
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
        self.trading_gains_history = deque(maxlen=20)  # CRITICAL: Track recent trading gains for DL training
        self.profit_history = deque(maxlen=100)  # Track recent profits

        # REALISTIC: Conservative profitability thresholds for institutional fund
        self.min_acceptable_return = 0.02  # 2% minimum annual return (above risk-free)
        self.target_return_threshold = 0.05  # 5% target return threshold
        self.excellent_return_threshold = 0.08  # 8% excellent return threshold (realistic for renewables)

        # Portfolio tracking (no emergency liquidation for infrastructure fund)
        self.portfolio_peak = float(self.initial_budget)
        self.emergency_liquidation_enabled = False  # Disabled: can't liquidate wind farms instantly
        self.emergency_liquidation_threshold = 0.10  # Only for extreme cases (90% loss)
        # NOTE: max_drawdown_threshold will be set from config below

        # Technology operational volatilities (daily-ish proxies)
        self.operational_vols = {'wind': 0.03, 'solar': 0.025, 'hydro': 0.015}
        self.operational_correlations = {'wind_solar': 0.4, 'wind_hydro': 0.2, 'solar_hydro': 0.3}

        # FIXED: Use reward weights from config to maintain single source of truth
        # Default fairness: identical non-forecast weights across tiers
        forecast_enabled = getattr(config, 'enable_forecast_utilisation', False) if config else False

        if config and hasattr(config, 'profit_reward_weight'):
            # Use config-driven reward weights (fair by default)
            base_weights = {
                'operational_revenue': 0.35,
                'risk_management': 0.25,
                'hedging_effectiveness': 0.20,
                'nav_stability': 0.20,
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
            base_weights = {
                'operational_revenue': 0.35,
                'risk_management': 0.25,
                'hedging_effectiveness': 0.20,
                'nav_stability': 0.20,
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

            self.max_drawdown_threshold = 0.10
        self.current_drawdown = 0.0
        self.peak_nav = float(self.initial_budget)
        self.trading_enabled = True
        self.position_size_multiplier = 1.0

        logger.info(f"[REWARD] Forecast utilization: {'ENABLED' if forecast_enabled else 'DISABLED'}")
        logger.info(f"[REWARD] Forecast weight: {self.reward_weights['forecast']:.3f}")
        logger.info(f"[REWARD] Weights: {self.reward_weights}")

    def update_trading_gains(self, mtm_pnl: float):
        self.trading_gains_history.append(mtm_pnl)

    @property
    def recent_trading_gains(self) -> float:
        if len(self.trading_gains_history) == 0:
            return 0.0
        return float(np.mean(list(self.trading_gains_history)))

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
        if drawdown_float > getattr(self, 'max_drawdown_threshold', 0.1):
            self.trading_enabled = False
        elif drawdown_float > 0.45:
            self.trading_enabled = True
            self.position_size_multiplier = 0.3
            self._base_position_multiplier = 0.3
        elif drawdown_float > 0.30:
            self.trading_enabled = True
            self.position_size_multiplier = 0.6
            self._base_position_multiplier = 0.6
        else:
            self.trading_enabled = True
            self.position_size_multiplier = 1.0
            self._base_position_multiplier = 1.0

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
            nav_return_abs = float(abs(nav_return))
            if nav_return_abs < 0.01:
                nav_stability_score = 1.0
            else:
                nav_stability_score = float(np.clip(1.0 - nav_return_abs * 50.0, -2.0, 1.0))
        else:
            nav_stability_score = 1.0

        forecast_score = float(np.clip(forecast_signal_score * 5.0, -5.0, 5.0))

        reward_weights = getattr(self, 'reward_weights', {})
        base_forecast_weight = float(reward_weights.get('forecast', 0.0))

        forecast_trust = float(getattr(self, '_forecast_trust', 0.5))
        z_combined_val = getattr(self, 'z_combined', None)
        if z_combined_val is not None:
            forecast_confidence = float(np.clip(abs(z_combined_val), 0.0, 1.0))
        else:
            forecast_confidence = forecast_trust

        dl_confidence = None
        if hasattr(self, 'dl_adapter_overlay') and self.dl_adapter_overlay is not None:
            overlay_out = getattr(self, '_last_overlay_output', {})
            if overlay_out and isinstance(overlay_out, dict) and 'pred_reward' in overlay_out:
                try:
                    pred_reward = overlay_out['pred_reward']
                    if isinstance(pred_reward, np.ndarray):
                        pred_reward = float(pred_reward.flatten()[0])
                    else:
                        pred_reward = float(pred_reward)
                    dl_confidence = float(np.clip(abs(pred_reward), 0.0, 1.0))
                except Exception:
                    dl_confidence = None

        combined_confidence = 0.5 * forecast_trust + 0.5 * forecast_confidence
        if dl_confidence is not None:
            combined_confidence = 0.4 * combined_confidence + 0.6 * dl_confidence
        combined_confidence = float(np.clip(combined_confidence, 0.0, 1.0))
        self._combined_confidence = combined_confidence

        adaptive_forecast_weight = base_forecast_weight
        if base_forecast_weight > 0.0:
            if combined_confidence > 0.7:
                adaptive_forecast_weight = min(0.48, base_forecast_weight * 1.2)
            elif combined_confidence < 0.3:
                adaptive_forecast_weight = max(0.32, base_forecast_weight * 0.8)
        self._adaptive_forecast_weight = adaptive_forecast_weight

        self.last_forecast_score = forecast_score
        self.last_operational_score = operational_score
        self.last_risk_score = risk_management_score
        self.last_hedging_score = hedging_score
        self.last_nav_stability_score = nav_stability_score

        warmup_steps = max(1, int(getattr(self.config, 'forecast_reward_warmup_steps', 1000)))
        forecast_horizons = getattr(self.config, 'forecast_horizons', {}) if self.config else {}
        horizon_long = int(forecast_horizons.get('long', 144))
        current_step = float(getattr(self, 't', 0))
        history_ready = float(np.clip(current_step / max(horizon_long, 1), 0.0, 1.0))
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

        lambda_w = getattr(self.config, "overlay_pred_reward_lambda", 0.0) if self.config else 0.0
        pred_reward_enabled = getattr(self.config, "overlay_pred_reward_enable", True) if self.config else True
        feature_dim = getattr(self, 'feature_dim', OVERLAY_FEATURE_DIM)
        if lambda_w > 1e-9 and pred_reward_enabled and feature_dim == OVERLAY_FEATURE_DIM:
            try:
                if hasattr(self, '_overlay_pred_r_hist') and len(self._overlay_pred_r_hist) > 0:
                    pred_smoothed = float(np.mean(list(self._overlay_pred_r_hist)))
                    pred_reward_contrib = float(lambda_w * pred_smoothed)
                    reward = float(reward) + pred_reward_contrib

                    try:
                        mwdir = getattr(self, 'mwdir', 0.0)
                        if self.t > 0:
                            price_current_raw = self._price_raw[self.t] if self.t < len(self._price_raw) else 0.0
                            price_prev_raw = self._price_raw[self.t-1] if self.t > 0 and self.t-1 < len(self._price_raw) else price_current_raw
                            realized_return = (price_current_raw - price_prev_raw) / (abs(price_prev_raw) + 1e-6)
                            mw_signal = float(np.clip(mwdir, -1.0, 1.0))
                            alignment = np.sign(realized_return) * np.sign(mw_signal)
                            lambda_mw = float(lambda_w * 0.5)
                            mw_reward_contrib = float(lambda_mw * alignment)
                            reward = float(reward) + mw_reward_contrib
                    except Exception:
                        pass
            except Exception:
                pass

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

            if hasattr(self, 'risk_multiplier'):
                risk_mult = float(self.risk_multiplier)
                hedging_score *= float(np.clip(1.0 + 0.1 * (risk_mult - 1.0), 0.5, 1.5))

            tracker['effectiveness_multiplier'] = float(np.clip(tracker['effectiveness_multiplier'] * 0.99 + hedging_score * 0.01, 0.5, 1.2))
            return float(np.clip(hedging_score * tracker['effectiveness_multiplier'], -3.0, 3.0))
        except Exception:
            return 0.0

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
    # NOTE: These will be moved to config in the constructor
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
        investment_freq: int = 12,
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

        # 2) FINANCIAL INSTRUMENTS (Mark-to-market, tradeable positions)
        self.financial_positions = {
            'wind_instrument_value': 0.0,   # Financial exposure to wind prices
            'solar_instrument_value': 0.0,  # Financial exposure to solar prices
            'hydro_instrument_value': 0.0,  # Financial exposure to hydro prices
        }
        # CORRECTNESS: Track true MTM equity separately from exposure.
        # - `financial_positions` is NOTIONAL EXPOSURE (changes only on trades)
        # - `financial_mtm_positions` is CUMULATIVE MTM VALUE / PnL (changes only with price returns)
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

        # REMOVED: Extra costs that were causing the 25M loss dip (not in PrototypeTestTuned)
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
        if getattr(self.config, 'enable_forecast_utilisation', False):
            if getattr(self.config, 'fgb_mode', None) == 'meta':
                tier_name = "tier3"
            else:
                tier_name = "tier2"
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
        # REMOVED: Multi-step rolling returns and MTM delta buffers (reverted to original single-step rewards)

        # Calculate rolling statistics for normalization (30-day window = 4320 timesteps)
        window_size = min(4320, len(price_dkk_filtered))
        price_rolling_mean = price_dkk_filtered.rolling(window=window_size, min_periods=1).mean()
        price_rolling_std = price_dkk_filtered.rolling(window=window_size, min_periods=1).std()

        # FIXED: Handle NaN values in rolling statistics
        price_rolling_mean = price_rolling_mean.fillna(price_dkk_filtered.mean())
        price_rolling_std = price_rolling_std.fillna(price_dkk_filtered.std())
        price_rolling_std = price_rolling_std.replace(0.0, price_dkk_filtered.std())  # Avoid zero std

        # Normalize prices for agent observations (z-score with bounds)
        price_normalized = (price_dkk_filtered - price_rolling_mean) / (price_rolling_std + 1e-6)
        price_normalized_clipped = np.clip(price_normalized, -3.0, 3.0)  # ±3 sigma bounds

        # Ensure downstream components (wrapper/overlay) see identical [-1, 1] scaling
        self._price = (price_normalized_clipped / 3.0).to_numpy()

        # Store normalization parameters for revenue calculations
        self._price_mean = price_rolling_mean.to_numpy()
        self._price_std = price_rolling_std.to_numpy()

        # FIXED: Prices remain in DKK throughout system for consistency
        # Raw prices in _price_raw are DKK, normalized prices in _price are z-scores

        self._load  = self.data.get('load',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._riskS = self.data.get('risk',  pd.Series(0.3, index=self.data.index)).astype(float).to_numpy()
        
        # CRITICAL FIX: Compute rolling mean/std for generation assets (wind, solar, hydro)
        # This is needed for proper z-score computation in forecast_engine.py
        # Using same 30-day window (4320 timesteps) as price for consistency
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

        # FIXED: Pre-allocate forecast arrays for DL overlay labeler
        # Multi-horizon forecasts (immediate, short, medium, long)
        for target in ["price", "wind", "solar", "hydro", "load"]:
            for horizon in ["immediate", "short", "medium", "long"]:
                array_name = f"_{target}_forecast_{horizon}"
                setattr(self, array_name, np.full(self.max_steps, np.nan, dtype=float))

        # normalization scales (95th percentile with minimum thresholds)
        def p95_robust(x, min_scale=0.1):
            try:
                p95_val = float(np.nanpercentile(np.asarray(x, dtype=float), 95))
                return max(p95_val, min_scale) if p95_val > 0 else 1.0
            except Exception:
                return 1.0

        self.wind_scale  = p95_robust(self._wind, min_scale=0.1)
        self.solar_scale = p95_robust(self._solar, min_scale=0.1)
        self.hydro_scale = p95_robust(self._hydro, min_scale=0.1)
        self.load_scale  = p95_robust(self._load, min_scale=0.1)

        # agents
        self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]
        self.agents = self.possible_agents[:]

        # observation manager & spaces (BASE only here)
        self.obs_manager = StabilizedObservationManager(self)
        self.observation_spaces = {a: self.obs_manager.obs_space(a) for a in self.possible_agents}

        # FIXED: Normalized action spaces - all agents use [-1, 1] range for consistent learning
        self.action_spaces = {
            "investor_0":         spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "battery_operator_0": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "risk_controller_0":  spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # FIXED: [-1,1] instead of [0,2]
            "meta_controller_0":  spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # FIXED: [-1,1] instead of [0,1]
        }

        # CRITICAL FIX: Reward calculator is now initialized AFTER asset deployment above
        # No need to set to None here as it's already properly initialized

        try:
            self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144, config=self.config) if enhanced_risk_controller else None
        except Exception:
            self.enhanced_risk_controller = EnhancedRiskController(config=self.config) if enhanced_risk_controller else None

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
        
        # OPTIMIZATION: Per-horizon MAPE tracking for adaptive weighting
        # Track MAPE separately for each horizon to enable adaptive z-score weighting
        # Lower MAPE = higher weight (more accurate forecasts get more influence)
        self._horizon_mape = {
            'short': deque(maxlen=100),   # Track last 100 MAPE values for short horizon
            'medium': deque(maxlen=100),  # Track last 100 MAPE values for medium horizon
            'long': deque(maxlen=100),    # Track last 100 MAPE values for long horizon
        }
        # FIX #21: CRITICAL BUG - maxlen must be >= horizon length!
        # Long horizon = 144 steps, so we need at least 144 pairs stored
        # Using 200 for safety margin (allows for 200 - 144 = 56 extra pairs)
        self._horizon_forecast_pairs = {
            'short': deque(maxlen=200),   # Store (forecast, actual) pairs for delayed MAPE calculation (horizon=6)
            'medium': deque(maxlen=200),  # Store (forecast, actual) pairs for delayed MAPE calculation (horizon=24)
            'long': deque(maxlen=200),    # Store (forecast, actual) pairs for delayed MAPE calculation (horizon=144)
        }
        
        # CRITICAL FIX: Correlation-based horizon weighting
        # Track forecast returns and actual returns for each horizon to compute correlation
        # Higher correlation = higher weight (more predictive forecasts get more influence)
        # Remove horizons with negative correlation (not predictive)
        self._horizon_return_pairs = {
            'short': deque(maxlen=200),   # Store (forecast_return, actual_return) pairs for correlation
            'medium': deque(maxlen=200),
            'long': deque(maxlen=200),
        }
        self._horizon_correlations = {
            'short': 0.0,   # Current correlation for short horizon
            'medium': 0.0,  # Current correlation for medium horizon
            'long': 0.0,    # Current correlation for long horizon
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
        self._previous_nav = 0.0  # STRATEGY 4: Track previous NAV for NAV-based reward calculation
        self.max_leverage = self.config.max_leverage
        self.risk_multiplier = self.config.risk_multiplier

        # performance tracking
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.cumulative_mtm_pnl = 0.0  # ENHANCED: Track cumulative trading performance
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

        # =====================================================================
        # DL OVERLAY STATE (34D FORECAST-AWARE MODE: 28D base + 6D deltas)
        # =====================================================================
        self._overlay_bridge_cache = {"investor_0": np.zeros((self.config.overlay_bridge_dim,), np.float32)}
        self._overlay_bridge_cache_batt = np.zeros((self.config.overlay_bridge_dim,), np.float32)
        self._overlay_risk_multiplier = 1.0
        self._overlay_pred_r_hist = deque(maxlen=getattr(self.config, "overlay_pred_reward_window", 20))
        self.dl_adapter_overlay = None  # Will be set by main.py after instantiation (DLAdapter for overlay inference)
        self.overlay_trainer = None  # Will be set by main.py after instantiation

        # =====================================================================
        # FGB: FORECAST-GUIDED BASELINE STATE
        # =====================================================================
        self.calibration_tracker = None  # Will be set by main.py (CalibrationTracker for trust & ΔNAV)
        self._last_overlay_output = {}  # Cache latest overlay output for forecast signals
        self._last_overlay_features = None  # FAMC: Cache 34D overlay features for meta head training
        self._forecast_trust = 0.0  # τₜ: forecast trust score
        self._expected_dnav = 0.0  # E[ΔNAV]: expected NAV change

        # =====================================================================
        # FORECAST RISK MANAGEMENT (AUTO-ENABLED WITH FORECASTS)
        # =====================================================================
        # TIER 2: When forecasts are enabled, automatically enable risk management
        # This ensures forecasts are ALWAYS used correctly (risk management, not directional trading)
        # REFACTORED: Use ForecastEngine for truly optional forecast integration (merged with adapter)
        from forecast_engine import ForecastEngine
        self.forecast_engine = ForecastEngine(
            env=self,
            config=config,
            forecast_generator=forecast_generator
        )
        
        enable_forecast_util = self.forecast_engine.is_enabled()
        if enable_forecast_util:
            # TIER 2: Forecasts enabled - can optionally enable forecast risk management as add-on
            forecast_risk_mgmt_mode = getattr(config, 'forecast_risk_management_mode', False)
            if forecast_risk_mgmt_mode:
                from risk import ForecastRiskManager
                self.forecast_risk_manager = ForecastRiskManager(config)
                logger.info("="*70)
                logger.info("TIER 2: FORECAST OBSERVATIONS + RISK MANAGEMENT ENABLED")
                logger.info("="*70)
            else:
                # Forecast observations only (fair comparison mode)
                self.forecast_risk_manager = None
                logger.info("="*70)
                logger.info("TIER 2: FORECAST OBSERVATIONS ENABLED (FAIR COMPARISON MODE)")
                logger.info("="*70)
        else:
            # TIER 1: No forecasts, no risk management
            self.forecast_risk_manager = None
            logger.info("="*70)
            logger.info("TIER 1: BASELINE MARL (NO FORECASTS)")
            logger.info("="*70)

        # =====================================================================
        # KELLY POSITION SIZING & REGIME DETECTION (Portfolio Optimization)
        # =====================================================================
        from utils import MultiAssetKellySizer, MarketRegimeDetector

        self.kelly_sizer = MultiAssetKellySizer(
            assets=['wind', 'solar', 'hydro'],
            correlation_lookback=100,
            lookback=100,
            max_kelly_fraction=0.25,
            min_trades_required=20
        )

        self.regime_detector = MarketRegimeDetector(
            lookback_short=20,
            lookback_long=50,
            vol_lookback=100,
            vol_regime_threshold=1.5,
            hurst_mr_threshold=0.4,
            trend_strength_threshold=0.6,
            ma_signal_threshold=0.02
        )

        # NOTE (Model B accounting):
        # Financial instruments are treated as a margin-style overlay:
        # - `financial_positions` are exposures (not equity / not "position value")
        # - NAV impact comes from MTM PnL, tracked separately in `financial_mtm_positions`
        # Therefore we do NOT maintain "open position cost basis" bookkeeping.
        # Kelly is updated from per-step MTM PnL in `_update_finance()`.
        self._current_regime = {'regime': 'neutral', 'position_multiplier': 1.0, 'metrics': {}}

        # OPTIMIZED: Track forecast accuracy for adaptive confidence
        self._forecast_accuracy_tracker = {
            'immediate': deque(maxlen=100),  # Track last 100 immediate forecasts
            'short': deque(maxlen=100),
            'medium': deque(maxlen=100),
            'long': deque(maxlen=100)
        }
        self._forecast_predictions = {
            'immediate': deque(maxlen=100),  # Store predictions to compare with actuals
            'short': deque(maxlen=100),
            'medium': deque(maxlen=100),
            'long': deque(maxlen=100)
        }

        # P&L-BASED TRAINING: Track actions and outcomes for direct profit optimization
        # This enables the DL overlay to learn from ACTUAL TRADING OUTCOMES, not just price predictions
        # Method: Look back at action taken N steps ago, measure P&L, use as training label
        # Result: Model learns "what actions led to profit" instead of "what prices will be"
        self._action_history = deque(maxlen=200)  # Store (timestep, action, positions, price, nav)
        self._pnl_history = deque(maxlen=200)     # Store (timestep, trading_pnl, nav_change, reward)
        self._last_nav = None  # Track NAV changes for P&L attribution

        # =====================================================================
        # OVERLAY TRAINING STATE (Experience collection for trainer)
        # =====================================================================
        self._overlay_feature_history = deque(maxlen=20)  # Recent features for target computation
        self._overlay_reward_history = deque(maxlen=20)   # Recent rewards for pred_reward target
        self._overlay_action_history = deque(maxlen=20)   # Recent actions for bridge_vec target
        self._overlay_pnl_history = deque(maxlen=20)      # Recent P&L for risk_budget target

        # =====================================================================
        # DELTA NORMALIZATION STATE (EMA std tracking for stabilization)
        # =====================================================================
        # CRITICAL FIX: Track EMA of FORECAST DELTA magnitudes, not realized volatility
        # Forecast deltas (forecast - current) are typically 50-200 DKK (forecast error)
        # Realized volatility (current - previous) is only 1-10 DKK (actual movement)
        # We need to normalize forecast deltas by their own distribution, not realized volatility
        # FIX Issue #1: Initialize with placeholder values (will be updated adaptively)
        self.ema_std_short = 100.0   # EMA of |forecast_delta_short| in DKK (initial placeholder)
        self.ema_std_medium = 120.0  # EMA of |forecast_delta_medium| in DKK (initial placeholder)
        self.ema_std_long = 150.0    # EMA of |forecast_delta_long| in DKK (initial placeholder)
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

        # Display both DKK and USD for clarity
        usd_value = self.init_budget * getattr(self.config, 'dkk_to_usd_rate', 0.145)
        logger.info(f"Hybrid renewable fund initialized with {self.init_budget:,.0f} DKK (~${usd_value/1e6:,.0f}M USD)")
        self._log_fund_structure()

        # GUARDRAIL: Startup assert to prevent regression
        assert hasattr(self, "_price_raw") and hasattr(self, "_price"), "Price arrays not initialized"

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
            tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
            logger.info(f"[{tier_name}_ASSET_DEPLOY] Asset deployment complete:")
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
        logger.info(f"   Wind farms: {self.physical_assets['wind_capacity_mw']:.1f} MW ({self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw']:,.0f} DKK)")
        logger.info(f"   Solar farms: {self.physical_assets['solar_capacity_mw']:.1f} MW ({self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw']:,.0f} DKK)")
        logger.info(f"   Hydro plants: {self.physical_assets['hydro_capacity_mw']:.1f} MW ({self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw']:,.0f} DKK)")
        logger.info(f"   Battery storage: {self.physical_assets['battery_capacity_mwh']:.1f} MWh ({self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']:,.0f} DKK)")
        logger.info(f"   Total Physical: {total_physical_mw:.1f} MW ({physical_value:,.0f} DKK book value, ${physical_value * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logger.info("")

        # Financial instruments (derivatives trading)
        total_financial = sum(abs(v) for v in self.financial_positions.values())
        logger.info("2. FINANCIAL INSTRUMENTS (Derivatives Trading - Mark-to-Market):")
        logger.info(f"   Wind index exposure: {self.financial_positions['wind_instrument_value']:,.0f} DKK")
        logger.info(f"   Solar index exposure: {self.financial_positions['solar_instrument_value']:,.0f} DKK")
        logger.info(f"   Hydro index exposure: {self.financial_positions['hydro_instrument_value']:,.0f} DKK")
        logger.info(f"   Total Financial Exposure: {total_financial:,.0f} DKK")
        logger.info("")

        # Fund summary
        fund_nav = self._calculate_fund_nav()
        logger.info("3. FUND SUMMARY:")
        logger.info(f"   Cash position: {self.budget:,.0f} DKK (${self.budget * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logger.info(f"   Physical assets (book): {physical_value:,.0f} DKK (${physical_value * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logger.info(f"   Financial positions (MTM): {total_financial:,.0f} DKK")
        logger.info(f"   Total Fund NAV: {fund_nav:,.0f} DKK (${fund_nav * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logger.info(f"   Initial capital: {self.init_budget:,.0f} DKK (${self.init_budget * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logger.info(f"   Total return: {((fund_nav - self.init_budget) / self.init_budget * 100):+.2f}%")
        logger.info("=" * 60)

    def set_overlay_params(self, **kwargs):
        """
        Setter to receive meta-knobs for overlay tuning.
        Allows metacontroller to adjust overlay parameters dynamically.
        """
        cfg = getattr(self, 'config', None)
        if cfg is None:
            return

        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                if getattr(self, '_verbose_overlay', False):
                    logger.info(f"[Overlay] Updated {k} = {v}")

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
                tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
                logger.info(f"[{tier_name}_NAV_CALC] Calculating NAV at t=0:")
                logger.info(f"  current_timestep param = {current_timestep}")
                logger.info(f"  self.t = {getattr(self, 't', 0)}")
                logger.info(f"  budget = {self.budget:,.0f} DKK")
                logger.info(f"  physical_assets = {self.physical_assets}")
                logger.info(f"  physical_capex_deployed = {getattr(self, 'physical_capex_deployed', 0.0):,.0f} DKK")
            
            # Use FinancialEngine for calculation
            from financial_engine import FinancialEngine
            nav = FinancialEngine.calculate_fund_nav(
                budget=self.budget,
                physical_assets=self.physical_assets,
                asset_capex=self.asset_capex,
                # CRITICAL: NAV must include MTM VALUE (PnL), not NOTIONAL exposure.
                financial_positions=getattr(self, 'financial_mtm_positions', self.financial_positions),
                accumulated_operational_revenue=accumulated_operational_revenue,
                current_timestep=current_timestep,
                config=self.config
            )
            
            # CRITICAL DEBUG: Log calculated NAV at timestep 0
            if current_timestep == 0:
                tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
                logger.info(f"[{tier_name}_NAV_CALC] NAV calculated: {nav:,.0f} DKK (${nav * 0.145 / 1_000_000:.2f}M)")
            
            self.equity = nav
            return nav

        except Exception as e:
            logger.error(f"NAV calculation error: {e}")
            self.equity = max(self.budget, self.init_budget * 0.01)
            return self.equity

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
        """Track forecast accuracy against realized values in consistent normalized units."""
        # Skip forecast tracking if no forecast generator (disabled by default)
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return
        if self.t == 0:
            return
        try:
            for target in ['wind', 'solar', 'hydro', 'price', 'load']:
                forecast_key = f"{target}_forecast_immediate"
                if forecast_key not in forecasts:
                    continue

                # FIXED: Use consistent normalization for both actual and forecast
                if target == "price":
                    # For price: use z-score normalization for both actual and forecast
                    # self._price[t] is already z-score normalized
                    actual_normalized = float(getattr(self, f"_{target}")[self.t]) if hasattr(self, f"_{target}") else 0.0

                    # Convert forecast from raw DKK to z-score using same rolling stats
                    fv = float(forecasts[forecast_key])  # raw DKK forecast
                    i = min(self.t, len(self._price_mean) - 1)
                    mean = float(self._price_mean[i])
                    std = max(float(self._price_std[i]), 1e-6)
                    forecast_normalized = (fv - mean) / std
                    # Apply same clipping as in price normalization
                    forecast_normalized = float(np.clip(forecast_normalized, -3.0, 3.0))

                elif target in ["wind", "solar", "hydro"]:
                    # For renewables: use scale normalization for both
                    actual_raw = getattr(self, f"_{target}")[self.t] if hasattr(self, f"_{target}") else 0.0
                    scale = getattr(self, f"{target}_scale", 1.0)
                    actual_normalized = actual_raw / max(scale, 1e-9)

                    fv = float(forecasts[forecast_key])
                    forecast_normalized = fv / max(scale, 1e-9)

                elif target == "load":
                    # For load: use load_scale normalization for both
                    actual_raw = getattr(self, f"_{target}")[self.t] if hasattr(self, f"_{target}") else 0.0
                    actual_normalized = actual_raw / max(getattr(self, "load_scale", 1.0), 1e-9)

                    fv = float(forecasts[forecast_key])
                    forecast_normalized = fv / max(getattr(self, "load_scale", 1.0), 1e-9)

                else:
                    # For other targets: use raw values
                    actual_normalized = getattr(self, f"_{target}")[self.t] if hasattr(self, f"_{target}") else 0.0
                    forecast_normalized = float(forecasts[forecast_key])

                # MAPE-like error vs current forecast (online)
                error = abs(actual_normalized - forecast_normalized) / (abs(actual_normalized) + 1e-6)
                self._forecast_errors[target].append(float(np.clip(error, 0.0, 10.0)))
                self._forecast_history_per_target[target].append(float(forecast_normalized))

                # DIAGNOSTIC LOGGING: Track MAPE for all targets (every 500 steps)
                if self.t % 500 == 0 and len(self._forecast_errors[target]) >= 10:
                    recent_mape = np.mean(list(self._forecast_errors[target])[-10:])
                    logger.info(f"[MAPE_TRACKING] t={self.t} target={target} recent_mape={recent_mape:.4f} "
                               f"samples={len(self._forecast_errors[target])} "
                               f"actual={actual_normalized:.4f} forecast={forecast_normalized:.4f}")
        except Exception as e:
            # Log errors for debugging (MAPE tracking failures are critical)
            if self.t % 1000 == 0:
                logger.warning(f"[MAPE_TRACKING] Failed to track forecast accuracy at t={self.t}: {e}")
            pass

    def get_forecast_accuracy_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for target, errors in self._forecast_errors.items():
            if len(errors) > 0:
                errors_list = list(errors)
                stats[target] = {
                    'mean_mape': float(np.mean(errors_list)),
                    'std_mape': float(np.std(errors_list)),
                    'recent_mape': float(np.mean(errors_list[-10:])) if len(errors_list) >= 10 else float(np.mean(errors_list)),
                    'samples': len(errors_list)
                }
        return stats

    # ------------------------------------------------------------------
    # PATCH: aligned-horizon forecast helpers
    # ------------------------------------------------------------------
    def _aligned_horizon_steps(self) -> int:
        """DEPRECATED: Use config-driven horizon selection instead."""
        # This method is kept for backward compatibility but should use config
        try:
            inv_freq = int(max(1, getattr(self, "investment_freq", 6)))

            # FAIL-FAST: Strict horizon lookup without silent defaults
            horizons = self.config.forecast_horizons
            if not horizons:
                raise ValueError("config.forecast_horizons is empty or missing")

            # FAIL-FAST: No hardcoded horizon lists allowed for production safety
            # All required horizons must be defined in config.forecast_horizons
            if not horizons:
                raise ValueError("config.forecast_horizons is empty. Define all required horizons in config to maintain single source of truth.")

            # Verify minimum required horizons exist (get from config if available)
            if hasattr(self.config, 'required_forecast_horizons'):
                required_horizons = self.config.required_forecast_horizons
            else:
                # If not defined in config, require at least these basic horizons
                required_horizons = ['immediate', 'short', 'medium', 'long', 'strategic']

            missing = [h for h in required_horizons if h not in horizons]
            if missing:
                raise ValueError(f"Missing required horizons in config: {missing}. "
                               f"Available: {list(horizons.keys())}. "
                               f"Add missing horizons to config.forecast_horizons to maintain single source of truth.")

            # Use strict horizon steps for alignment (no .get() defaults)
            if inv_freq <= horizons['immediate']:
                return horizons['immediate']
            elif inv_freq <= horizons['short']:
                return horizons['short']
            elif inv_freq <= horizons['medium']:
                return horizons['medium']
            elif inv_freq <= horizons['long']:
                return horizons['long']
            else:
                return horizons['strategic']
        except Exception as e:
            # FAIL-FAST: No silent fallback to hardcoded values
            raise ValueError(f"Cannot align horizon steps: config.forecast_horizons invalid or missing. "
                           f"Investment frequency: {getattr(self, 'investment_freq', 'unknown')}. "
                           f"Config state: {getattr(self.config, 'forecast_horizons', 'missing')}. "
                           f"Original error: {e}")

    def _get_aligned_price_forecast(self, t: int, default: float = None) -> Optional[float]:
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return default
        try:
            # FIXED: Use named horizon alignment instead of numeric
            h_steps = self._aligned_horizon_steps()

            # CANONICAL: Map steps to horizon names using config source of truth
            horizon_name = "immediate"  # default
            for name, steps in self.config.forecast_horizons.items():
                if h_steps <= steps:
                    horizon_name = name
                    break
            else:
                # If h_steps exceeds all defined horizons, use the largest one
                horizon_name = max(self.config.forecast_horizons.keys(),
                                 key=lambda k: self.config.forecast_horizons[k])

            if hasattr(self.forecast_generator, "predict_all_horizons"):
                d = self.forecast_generator.predict_all_horizons(timestep=t)
                if isinstance(d, dict):
                    # Try aligned horizon name first
                    aligned_key = f"price_forecast_{horizon_name}"
                    if aligned_key in d and np.isfinite(d[aligned_key]):
                        return float(d[aligned_key])

                    # Fallback to immediate
                    for k in ("price_forecast_immediate",):
                        if k in d and np.isfinite(d[k]): return float(d[k])

            if hasattr(self.forecast_generator, "predict_for_agent"):
                d = self.forecast_generator.predict_for_agent(agent="investor_0", timestep=t)
                if isinstance(d, dict):
                    # Try aligned horizon name first
                    aligned_key = f"price_forecast_{horizon_name}"
                    if aligned_key in d and np.isfinite(d[aligned_key]):
                        return float(d[aligned_key])

                    # Fallback to immediate
                    for k in ("price_forecast_immediate",):
                        if k in d and np.isfinite(d[k]): return float(d[k])
        except Exception:
            pass
        return default

    def populate_forecast_arrays(self, t: int, forecasts: Dict[str, float]):
        """
        FIXED: Populate forecast arrays for DL overlay labeler access.
        Called from wrapper after computing forecasts.

        Now handles ALL horizons (immediate, short, medium, long) for each target.
        This ensures the 34D overlay features (28D base + 6D deltas) get the full multi-horizon signal.

        FIX: Verify arrays are pre-allocated and populated correctly to prevent
        silent fallback to current values when forecasts are missing.
        """
        # Skip forecast array population if no forecast generator
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return
        try:
            if 0 <= t < self.max_steps:
                # Populate all targets × horizons
                populated_count = 0
                for target in ["price", "wind", "solar", "hydro", "load"]:
                    for horizon in ["immediate", "short", "medium", "long"]:
                        key = f"{target}_forecast_{horizon}"
                        if key in forecasts:
                            array_name = f"_{target}_forecast_{horizon}"
                            if hasattr(self, array_name):
                                getattr(self, array_name)[t] = float(forecasts[key])
                                populated_count += 1

                # FIX: Log at t==0 to verify arrays are being populated
                if t == 0 and populated_count > 0:
                    logger.info(f"[FORECAST_ARRAYS] Step 0: Populated {populated_count} forecast values")
                    logger.info(f"[FORECAST_ARRAYS] Expected: 5 targets × 4 horizons = 20 values per step")
                    # Verify all arrays exist
                    missing_arrays = []
                    for target in ["price", "wind", "solar", "hydro", "load"]:
                        for horizon in ["immediate", "short", "medium", "long"]:
                            array_name = f"_{target}_forecast_{horizon}"
                            if not hasattr(self, array_name):
                                missing_arrays.append(array_name)
                    if missing_arrays:
                        logger.warning(f"[FORECAST_ARRAYS] Missing arrays: {missing_arrays}")
                    else:
                        logger.info(f"[FORECAST_ARRAYS] All 20 forecast arrays pre-allocated ✓")

                    # DIAGNOSTIC: Show actual forecast values vs current
                    current_price = float(self.price[0]) if len(self.price) > 0 else 0.0
                    logger.info(f"[FORECAST_VALUES] t=0 current_price={current_price:.2f}")
                    for horizon in ["short", "medium", "long"]:
                        key = f"price_forecast_{horizon}"
                        if key in forecasts:
                            fval = forecasts[key]
                            delta = fval - current_price
                            logger.info(f"  {horizon}: forecast={fval:.2f}, delta={delta:.4f}")

                    # NEW: dump full multi-target snapshot for parity checks
                    snapshot = {k: float(v) for k, v in forecasts.items() if k.startswith(("price_", "wind_", "solar_", "hydro_", "load_"))}
                    logger.info("[FORECAST_SNAPSHOT] t=0 → %s", json.dumps(snapshot, sort_keys=True))
        except Exception as e:
            logger.debug(f"[FORECAST_ARRAYS] Population error at t={t}: {e}")
            pass  # Silently ignore errors to avoid breaking main flow

    def _get_forecast_safe(self, target: str, horizon: str, default: float = 0.0) -> float:
        """
        Get forecast value with fallback to current value.

        CRITICAL FIX: Centralized method to avoid 4x duplication.

        Args:
            target: Target variable (price, wind, solar, hydro, load)
            horizon: Forecast horizon (immediate, short, medium, long)
            default: Default value if both forecast and current unavailable

        Returns:
            Forecast value, or current value, or default
        """
        # Try to get forecast value
        array_name = f"_{target}_forecast_{horizon}"
        if hasattr(self, array_name) and self.t < len(getattr(self, array_name)):
            val = getattr(self, array_name)[self.t]
            if not np.isnan(val) and np.isfinite(val):
                return float(val)

        # Fallback to current value
        current_array = f"_{target}"
        if hasattr(self, current_array) and self.t < len(getattr(self, current_array)):
            return float(getattr(self, current_array)[self.t])

        # Final fallback
        return default

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

        # DEBUG: Log overlay configuration at reset
        overlay_enabled = getattr(self.config, 'overlay_enabled', False)
        feature_dim = getattr(self, 'feature_dim', OVERLAY_FEATURE_DIM)  # Use canonical constant
        has_forecaster = getattr(self, 'forecast_generator', None) is not None
        logger.info(f"[RESET] Environment reset: overlay_enabled={overlay_enabled}, feature_dim={feature_dim} (34D: 28D base + 6D deltas), has_forecaster={has_forecaster}")

        # Check if this is a true episode reset (end of data) or just PPO buffer reset
        # FIXED: Safety check for self.t attribute
        current_timestep = getattr(self, 't', 0)
        is_true_episode_end = (current_timestep >= self.max_steps)

        # CRITICAL FIX: For Episode 0 first reset, ensure t=0 before any NAV calculations
        # This prevents depreciation from being applied during initial reset
        if not hasattr(self, '_episode_counter') or self._episode_counter == -1:
            # First reset (Episode 0) - ensure t=0
            self.t = 0

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
                self.debug_tracker.start_episode(0)
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
            self.ema_std_medium = 120.0  # Reset to initial placeholder
            self.ema_std_long = 150.0    # Reset to initial placeholder
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
        
        # OPTIMIZATION: Clear per-horizon MAPE tracking on reset
        if hasattr(self, '_horizon_mape'):
            for horizon in ['short', 'medium', 'long']:
                self._horizon_mape[horizon].clear()
                self._horizon_forecast_pairs[horizon].clear()
        
        # CRITICAL FIX: Clear correlation tracking on reset
        if hasattr(self, '_horizon_return_pairs'):
            for horizon in ['short', 'medium', 'long']:
                self._horizon_return_pairs[horizon].clear()
                self._horizon_correlations[horizon] = 0.0

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

        # CRITICAL: Compute forecast deltas before filling observations
        # This ensures Tier 2 observations have z-scores from the start
        # FIX Issue #3: Initialize z-prev values on first reset
        enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False)
        if enable_forecast_util and self.forecast_generator is not None:
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
            'battery_energy': 0.0,
            'battery_discharge_power': 0.0,
        }

        # CRITICAL: DO NOT reset budget - let gains/losses accumulate
        # CRITICAL: DO NOT reset financial_positions - let trading gains accumulate
        # CRITICAL: DO NOT reset accumulated_operational_revenue - let operational gains accumulate
        # CRITICAL: DO NOT reset cumulative_mtm_pnl - let trading performance accumulate
        # CRITICAL: DO NOT reset cumulative_returns - let return performance accumulate
        # CRITICAL: DO NOT reset distributed_profits - let profit distributions accumulate
        # CRITICAL: DO NOT reset investment_capital - let capital changes accumulate

        # Only reset per-step tracking (not cumulative)
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}
        
        # REMOVED: Buffer clearing (reverted to original - no buffers needed)

        # CRITICAL FIX: Recalculate NAV with current state (preserving financial gains)
        # Use current timestep for depreciation calculation
        current_t = getattr(self, 't', 0)
        current_nav = self._calculate_fund_nav(current_timestep=current_t)
        
        # STRATEGY 4: Reset previous NAV for NAV-based reward calculation
        # Initialize with current NAV so first step's reward is based on change from reset NAV
        self._previous_nav = current_nav if current_nav > 1e-6 else float(self.init_budget)

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
            'battery_energy': 0.0,
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
        self.cumulative_mtm_pnl = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

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
        tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
        logger.info(f"[{tier_name}_RESET] Physical assets recalculated:")
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

        # Calculate NAV directly from consistent components
        initial_nav = trading_cash + physical_book_value + operational_revenue_value + financial_mtm_value
        
        # CRITICAL FIX: Force NAV to be consistent by setting equity explicitly
        self.equity = float(initial_nav)
        
        # CRITICAL FIX: Keep t=0 after reset (don't restore original_t)
        # This ensures NAV calculations after reset use t=0 (no depreciation)
        # t will be incremented in step() function
        self.t = 0
        
        # CRITICAL FIX: Store the initial NAV from reset to use for logging at timestep 0
        # This ensures identical NAV values at timestep 0, before any trades are executed
        self._initial_nav_from_reset = float(initial_nav)
        
        # Log for debugging
        if hasattr(self, 'config'):
            tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
            # Add a more detailed log to check components if needed
            if abs(initial_nav - self.init_budget) > 1000: # Log if NAV is significantly off from init_budget
                 logger.warning(f"[{tier_name}_RESET] Initial NAV components: Cash={trading_cash:,.0f}, "
                               f"Assets={physical_book_value:,.0f}, OpRev={operational_revenue_value:,.0f}, MTM={financial_mtm_value:,.0f}")
            logger.info(f"[{tier_name}_RESET] Initial NAV after full reset: {initial_nav:,.0f} DKK (${initial_nav * self.config.dkk_to_usd_rate / 1e6:.4f}M USD), init_budget: {self.init_budget:,.0f} DKK")

    def step(self, actions: Dict[str, Any]):
        # FIXED: Safety check for self.t attribute
        current_timestep = getattr(self, 't', 0)
        if current_timestep >= self.max_steps:
            return self._terminal_step()

        try:
            i = self.t

            # REMOVED: Expert suggestion cache clearing (legacy code removed)

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
                        # Map first 3 dims to wind/solar/hydro (ignore extra dims if any)
                        self._last_actions[key] = {
                            'wind': float(arr[0]) if arr.size > 0 else 0.0,
                            'solar': float(arr[1]) if arr.size > 1 else 0.0,
                            'hydro': float(arr[2]) if arr.size > 2 else 0.0,
                        }
                        # Preserve raw for debugging if needed
                        self._last_actions[key + '_raw'] = arr.astype(float).tolist()
                    elif key == 'battery_operator_0':
                        arr = np.array(v, dtype=np.float32).reshape(-1)
                        u_raw = float(arr[0]) if arr.size > 0 else 0.0
                        # Keep logger-compatible keys (charge/discharge)
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

            # meta & risk knobs
            self._apply_risk_control(acts['risk_controller_0'])
            self._apply_meta_control(acts['meta_controller_0'])

            # investor + battery ops (battery returns realized cash delta)
            # CRITICAL FIX: Pass i (timestep) to _execute_investor_trades for consistency
            trade_amount = self._execute_investor_trades(acts['investor_0'], timestep=i)
            battery_cash_delta = self._execute_battery_ops(acts['battery_operator_0'], i)

            # FIXED: finance update (MTM, costs, realized rev incl. battery cash)
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
                except Exception:
                    pass

            # ===== DL OVERLAY INFERENCE (34D FORECAST-AWARE MODE WITH DELTAS) =====
            # CRITICAL FIX: Compute forecast deltas BEFORE overlay check
            # Tier 2 needs z-scores for observations even when overlay is disabled
            # This ensures forecast deltas are always available when forecasts are loaded
            enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False)
            if enable_forecast_util and self.forecast_generator is not None:
                # Compute z-scores for Tier 2 observations (even if overlay is disabled)
                self._compute_forecast_deltas(i)

            # Run overlay inference once per step with 34D features (28D base + 6D deltas)
            # Use dl_adapter_overlay (DLAdapter) for overlay inference
            overlay_adapter = getattr(self, 'dl_adapter_overlay', None)
            if overlay_adapter is not None and getattr(self.config, 'overlay_enabled', False):
                try:
                    # Build 34D features for overlay (28D base + 6D deltas: Δprice_short/med/long, dir_consistency, mwdir, |mwdir|)
                    base_features = self._build_overlay_features(i)
                    if base_features is not None:
                        # DEBUG: Log feature building success
                        if i % 1000 == 0 and i > 0:
                            logger.info(f"[OVERLAY] Step {i}: Built {base_features.shape} features (34D: 28D base + 6D deltas)")

                        # Run shared inference
                        overlay_outs = overlay_adapter.shared_inference(base_features, training=False)

                        # DEBUG: Log inference success
                        if i % 1000 == 0 and i > 0:
                            # Show which outputs are actually USED (not just computed)
                            bridge_enabled = getattr(self.config, 'overlay_bridge_enable', True)
                            pred_reward_enabled = getattr(self.config, 'overlay_pred_reward_enable', True)
                            used_outputs = []
                            if bridge_enabled:
                                used_outputs.append('bridge_vec')
                            used_outputs.append('risk_budget')  # Always used
                            if pred_reward_enabled:
                                used_outputs.append('pred_reward')
                            used_outputs.extend(['strat_immediate', 'strat_short', 'strat_medium', 'strat_long'])  # Always used for deltas

                            all_outputs = list(overlay_outs.keys()) if overlay_outs else []
                            logger.info(f"[OVERLAY] Step {i}: Inference successful, computed={all_outputs}, USED={used_outputs}")

                        # Cache results for wrapper and reward calculator
                        if overlay_outs:
                            # Bridge vectors (append to agent obs in wrapper)
                            bridge_vec = overlay_outs.get("bridge_vec", np.zeros((self.config.overlay_bridge_dim,)))
                            if isinstance(bridge_vec, np.ndarray) and bridge_vec.size > 0:
                                self._overlay_bridge_cache["investor_0"] = bridge_vec[0] if len(bridge_vec.shape) > 1 else bridge_vec
                            if getattr(self.config, 'overlay_bridge_enable_battery', False):
                                self._overlay_bridge_cache_batt = bridge_vec[0] if len(bridge_vec.shape) > 1 else bridge_vec

                            # Risk budget multiplier (scale position sizes)
                            risk_budget = overlay_outs.get("risk_budget", np.array([[1.0]]))
                            if isinstance(risk_budget, np.ndarray):
                                self._overlay_risk_multiplier = float(risk_budget.flatten()[0])
                                # DEBUG: Log risk multiplier values to verify they're changing
                                if i % 100 == 0 and i > 0:
                                    logger.debug(f"[OVERLAY] Step {i}: risk_mult={self._overlay_risk_multiplier:.4f}, ema={self._overlay_risk_ema:.4f}")

                            # Predictive reward (for 28D reward shaping)
                            pred_reward = overlay_outs.get("pred_reward", np.array([[0.0]]))
                            if isinstance(pred_reward, np.ndarray):
                                self._overlay_pred_r_hist.append(float(pred_reward.flatten()[0]))

                            # FGB: Cache overlay output for forecast signals
                            self._last_overlay_output = overlay_outs
                            # FAMC: Cache overlay features (34D) for meta head training
                            self._last_overlay_features = base_features.copy()
                except Exception as e:
                    # FIX: Upgrade to WARNING level - overlay failures are CRITICAL and must be visible
                    logger.warning(f"[OVERLAY_INFERENCE] CRITICAL FAILURE at step {self.t}: {e}")
                    logger.debug(f"[OVERLAY_INFERENCE] Full traceback: {traceback.format_exc()}")

            # ===== REGIME DETECTION UPDATE =====
            # Update regime detector with current price
            # CRITICAL FIX: Use i (captured timestep) instead of self.t for consistency
            if hasattr(self, 'regime_detector') and hasattr(self, '_price') and i < len(self._price):
                current_price = float(self._price[i])
                self.regime_detector.update(current_price)

                # Detect regime every 6 steps (hourly)
                # CRITICAL FIX: Use i (captured timestep) instead of self.t for consistency
                if i % 6 == 0:
                    regime_info = self.regime_detector.detect_regime()
                    self._current_regime = regime_info

                    # Log regime changes daily
                    if i % 144 == 0:
                        logger.info(f"[REGIME] Step {i}: {regime_info['regime']} "
                                   f"(confidence={regime_info['confidence']:.2f}, "
                                   f"mult={regime_info['position_multiplier']:.2f}, "
                                   f"hurst={regime_info['metrics'].get('hurst_exponent', 0.5):.3f})")

            # NOTE: CALIBRATION TRACKER UPDATE MOVED TO AFTER z_short IS SET (line ~2930)

            # === ENHANCEMENT 6: DIAGNOSTIC LOGGING ===
            # Log overlay state for monitoring
            if i % 500 == 0 and i > 0 and getattr(self.config, 'overlay_enabled', False):
                mwdir = getattr(self, 'mwdir', 0.0)
                ema_short = getattr(self, 'ema_std_short', 0.01)
                ema_medium = getattr(self, 'ema_std_medium', 0.01)
                ema_long = getattr(self, 'ema_std_long', 0.01)
                conf = self._get_forecast_confidence()
                logger.info(f"[overlay] step={i} mwdir={mwdir:.3f} conf={conf:.2f} ema_stds=[{ema_short:.4f}, {ema_medium:.4f}, {ema_long:.4f}]")

            # ===== COLLECT OVERLAY TRAINING EXPERIENCE =====
            # Collect data for training the overlay model
            if self.overlay_trainer is not None and getattr(self.config, 'overlay_enabled', False):
                try:
                    self._collect_overlay_experience(i)
                except Exception as e:
                    # FIX: Upgrade to WARNING level - training data collection failures are important
                    logger.warning(f"[OVERLAY_TRAINING] Experience collection failed at step {self.t}: {e}")
                    logger.debug(f"[OVERLAY_TRAINING] Full traceback: {traceback.format_exc()}")

            # FIXED: assign rewards
            self._assign_rewards(financial)

            # step forward
            self.t += 1
            self.step_in_episode = self.t

            index = max(0, self.t - 1)
            self._fill_obs()

            # CRITICAL FIX: Update forecast generator with current observations
            # The forecast generator needs to see actual values to make predictions for next timestep
            if self.forecast_generator is not None and hasattr(self.forecast_generator, 'update'):
                try:
                    # Build observation dict with current values
                    current_obs = {
                        'price': float(self._price_raw[i]) if i < len(self._price_raw) else 250.0,
                        'wind': float(self._wind[i]) if i < len(self._wind) else 0.0,
                        'solar': float(self._solar[i]) if i < len(self._solar) else 0.0,
                        'hydro': float(self._hydro[i]) if i < len(self._hydro) else 0.0,
                        'load': float(self._load[i]) if i < len(self._load) else 0.0,
                    }
                    self.forecast_generator.update(current_obs)
                except Exception as e:
                    if i % 1000 == 0:
                        logger.warning(f"[FORECAST] Failed to update forecast generator at t={i}: {e}")

            # FGB: Compute forecast signals for baseline adjustment
            # REFACTORED: Use forecast engine for forecast signals
            if self.forecast_engine.is_enabled():
                self.forecast_engine.update_forecast_trust(
                    calibration_tracker=getattr(self, 'calibration_tracker', None)
                )
                self._compute_forecast_signals()

            self._populate_info(index, financial, acts)

            return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

        except Exception as e:
            logger.error(f"step error at t={self.t}: {e}")
            return self._safe_step()

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _to_numpy_safe(self, a_in):
        """Bring tensors (torch/jax/etc.) to CPU numpy by duck-typing."""
        try:
            if hasattr(a_in, "detach") and callable(a_in.detach):
                try: a_in = a_in.detach()
                except Exception: pass
            if hasattr(a_in, "cpu") and callable(a_in.cpu):
                try: a_in = a_in.cpu()
                except Exception: pass
            if hasattr(a_in, "numpy") and callable(a_in.numpy):
                try: a_in = a_in.numpy()
                except Exception: pass
        except Exception:
            pass
        return a_in

    def _validate_actions(self, actions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        self._clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        for agent in self.possible_agents:
            space = self.action_spaces[agent]
            a = actions.get(agent, None)

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

            # clipping count for diagnostics
            key = ("investor" if agent.startswith("investor")
                   else "battery" if agent.startswith("battery")
                   else "risk" if agent.startswith("risk")
                   else "meta")
            self._clip_counts[key] += int(np.any(np.abs(before - arr) > 1e-12))

        return out

    def _apply_risk_control(self, risk_action: np.ndarray):
        """
        REFACTORED: Apply risk control using TradingEngine.
        """
        try:
            from trading_engine import TradingEngine
            self.risk_multiplier = TradingEngine.apply_risk_control(risk_action)
        except Exception as e:
            logger.warning(f"Risk control failed: {e}")
            self.risk_multiplier = 1.0

    def _apply_meta_control(self, meta_action: np.ndarray):
        """
        REFACTORED: Apply meta control using TradingEngine.
        """
        try:
            from trading_engine import TradingEngine
            # Provide forecast confidence from forecast_engine if available
            forecast_confidence = 0.5
            try:
                if hasattr(self, 'forecast_engine') and self.forecast_engine is not None:
                    forecast_confidence = float(getattr(self.forecast_engine, 'forecast_trust', 0.5))
            except Exception:
                pass
            self.capital_allocation_fraction, self.investment_freq = TradingEngine.apply_meta_control(
                meta_action=meta_action,
                meta_cap_min=self.META_CAP_MIN,
                meta_cap_max=self.META_CAP_MAX,
                meta_freq_min=self.META_FREQ_MIN,
                meta_freq_max=self.META_FREQ_MAX,
                forecast_confidence=forecast_confidence,
                # Fair comparison: keep meta-control dynamics identical across tiers.
                # Forecast utilisation should help via better observations, not extra sizing mechanics.
                disable_confidence_scaling=True
            )
        except Exception as e:
            logger.warning(f"Meta control failed: {e}")
            # Keep current values on error

    # ------------------------------------------------------------------
    # REMOVED: Old asset-specific hedging logic (conflicted with single-price approach)
    # Now using portfolio-level _calculate_portfolio_hedge_intensity() instead
    # ------------------------------------------------------------------



    def _get_generation_exposure_weights(self) -> np.ndarray:
        """
        Calculate exposure weights based on expected generation (MWh)
        Used for portfolio-level hedge intensity calculation
        """
        try:
            # Get expected generation for each asset
            wind_capacity = self.physical_assets.get('wind_capacity_mw', 0)
            solar_capacity = self.physical_assets.get('solar_capacity_mw', 0)
            hydro_capacity = self.physical_assets.get('hydro_capacity_mw', 0)

            # Estimate annual generation (capacity factor assumptions)
            wind_cf = 0.35  # 35% capacity factor for wind
            solar_cf = 0.20  # 20% capacity factor for solar
            hydro_cf = 0.45  # 45% capacity factor for hydro

            wind_generation = wind_capacity * wind_cf * 8760  # MWh/year
            solar_generation = solar_capacity * solar_cf * 8760
            hydro_generation = hydro_capacity * hydro_cf * 8760

            total_generation = wind_generation + solar_generation + hydro_generation

            if total_generation > 0:
                return np.array([
                    wind_generation / total_generation,
                    solar_generation / total_generation,
                    hydro_generation / total_generation
                ])
            else:
                # Fallback to equal weights
                return np.array([1/3, 1/3, 1/3])

        except Exception as e:
            # Fallback to equal weights
            return np.array([1/3, 1/3, 1/3])

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
            return 1.0  # Default intensity

    # def set_wrapper_reference(self, wrapper_env):
    #     """CRITICAL: Set reference to wrapper for profit-seeking expert"""
    #     self._wrapper_ref = wrapper_env

    # REMOVED: get_expert_suggestion() - Legacy expert suggestion system removed
    # GNN encoder optionally learns feature relationships (works for both Tier 1 and Tier 2)
    # (when --enable_gnn_encoder flag enabled, optionally combined with --enable_forecast_utilisation for Tier 2).
    # Pre-trained ANN/LSTM forecasts provide additional observations.
    # NOT rule-based expert suggestions. Expert suggestions interfered with PPO learning and are deprecated

    def _calculate_portfolio_hedge_intensity_with_params(self, intensity: float, bias: float, risk_alloc=None) -> dict:
        """
        IMPROVED: Balanced signal mapping for optimal DL overlay impact.
        - intensity [0.5,2.0] → amplitude [0,1] with normal scaling
        - bias [0,1] → direction [-1,1]
        - Uses model's risk_allocation split
        - Output always in [-1,1]
        - ADAPTIVE: Applies overlay risk_budget multiplier and defense gates

        FIX 1: REBALANCED amplitude divisor from 0.5 back to 1.0 for normal strength signals
        - Reason: 2x stronger signals were too aggressive during cold start
        - Impact: Smaller positions, less risk, better performance
        """
        try:
            if risk_alloc is not None:
                w, s, h = np.clip(np.asarray(risk_alloc, dtype=np.float32), 0.0, 1.0)
                ssum = float(w + s + h) or 1.0
                weights = np.array([w/ssum, s/ssum, h/ssum], dtype=np.float32)
            else:
                weights = np.asarray(self._get_risk_budget_allocation(), dtype=np.float32)

            direction = (bias - 0.5) * 2.0  # [-1,1]
            # REVERTED: Use 1.5 divisor (old working version) instead of 1.0 or 0.5
            # Old version with 1.5 was working better than current 1.0
            amplitude = np.clip((intensity - 0.5) / 1.5, 0.0, 1.0)  # [0,1] from ~[0.5,2.0], weaker signals

            # ===== ADAPTIVE RISK BUDGETING (34D: 28D base + 6D deltas) =====
            # CRITICAL FIX: Only apply risk_budget to legacy hedge calculations
            # This function is used for fallback hedging when overlay is not active
            # When overlay_alpha=1.0, this function should NOT be called (expert suggestions used instead)
            # When overlay_alpha<1.0, this provides baseline hedge that gets blended

            # DECISION: Do NOT apply overlay risk_budget here
            # REASON: This creates triple-scaling:
            #   1. risk_budget scales amplitude here
            #   2. risk_budget scales positions in _execute_investor_trades (when alpha<1.0)
            #   3. Blending with expert suggestions (which already account for risk)
            # SOLUTION: Remove risk_budget application from this legacy function

            # Defense gates are still valid (reduce hedging during losses/drawdowns)
            defense_multiplier = 1.0

            # Apply defense gates: reduce amplitude if recent P&L is bad or drawdown is high
            defense_gate_moderate_dkk = getattr(self.config, 'overlay_defense_gate_dkk', -50_000) if self.config else -50_000
            defense_gate_severe_dkk = getattr(self.config, 'overlay_defense_gate_severe_dkk', -100_000) if self.config else -100_000
            defense_gate_dd = getattr(self.config, 'overlay_dd_gate', 0.01) if self.config else 0.01

            # Check recent P&L
            if hasattr(self, 'reward_calculator') and self.reward_calculator is not None:
                recent_gains = float(getattr(self.reward_calculator, 'recent_trading_gains', 0.0))
                if recent_gains < defense_gate_severe_dkk:
                    # Severe losses: reduce amplitude significantly
                    defense_multiplier *= 0.3
                elif recent_gains < defense_gate_moderate_dkk:
                    # Moderate losses: reduce amplitude moderately
                    defense_multiplier *= 0.6

            # Check drawdown
            if hasattr(self, 'reward_calculator') and self.reward_calculator is not None:
                current_dd = float(getattr(self.reward_calculator, 'current_drawdown', 0.0))
                if current_dd > defense_gate_dd:
                    # High drawdown: reduce amplitude
                    defense_multiplier *= max(0.3, 1.0 - current_dd * 5.0)

            # Apply defense multiplier to amplitude (NOT risk_budget)
            amplitude = amplitude * defense_multiplier
            amplitude = np.clip(amplitude, 0.0, 1.0)

            vec = np.clip(weights * amplitude * direction, -1.0, 1.0).astype(np.float32)
            return {'wind': float(vec[0]), 'solar': float(vec[1]), 'hydro': float(vec[2])}
        except Exception:
            return {'wind': 0.0, 'solar': 0.0, 'hydro': 0.0}

    # ------------------------------------------------------------------
    # FIXED: Financial Instrument Trading (Separate from Physical Assets)
    # ------------------------------------------------------------------

    def _execute_investor_trades(self, inv_action: np.ndarray, timestep: Optional[int] = None) -> float:
        """
        CORRECTED & FINAL: Executes trades based DIRECTLY on the RL agent's action.

        The agent's action vector [-1, 1] is mapped to a target allocation
        of the available trading capital. The RL agent is always in control.

        The DL overlay (if present) provides expert guidance through the observation
        space, not through action override.

        Returns total traded notional for transaction costs
        """
        # CRITICAL FIX: Use timestep parameter if provided, otherwise use self.t
        t = timestep if timestep is not None else getattr(self, 't', 0)
        
        # === FGB: ACTION BLENDING REMOVED ===
        # DEPRECATED: Action blending has been replaced by forecast-guided baseline.
        # The DL overlay now informs the PPO baseline and risk sizing, not action execution.
        # PPO remains the action decision-maker; we reduce advantage variance via baseline adjustment.
        #
        # Old code path (enable_action_blending) is disabled by default.
        # If accidentally enabled, emit deprecation warning and skip.
        if getattr(self.config, 'enable_action_blending', False):
            if getattr(self.config, 'deprecation_warnings', True):
                logger.warning(
                    "[FGB] DEPRECATED: enable_action_blending is no longer supported. "
                    "Action blending has been replaced by forecast-guided baseline. "
                    "Set enable_action_blending=False and use forecast_baseline_enable=True instead."
                )
            # Do NOT blend actions - use RL policy directly
            # Fall through to use original policy action

        # Trading is only allowed at the specified frequency
        # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
        # CRITICAL FIX: Prevent trading at timestep 0 to ensure identical initial NAV
        # At timestep 0, we want to log the reset NAV before any trades are executed
        if t == 0:
            return 0.0  # No trading at timestep 0 - ensures identical initial NAV
        if t > 0 and t % self.investment_freq != 0:
            return 0.0

        # Do not trade if disabled by high drawdown
        if not self.reward_calculator.trading_enabled:
            return 0.0

        try:
            # === STEP 1: Determine available capital for this trade ===
            available_capital = self.budget * self.capital_allocation_fraction
            position_size_multiplier = getattr(self.reward_calculator, 'position_size_multiplier', 1.0)

            # ENHANCEMENT: Apply overlay risk budget multiplier (smoothed with EMA)
            # This scales position sizes based on market conditions learned by the overlay
            overlay_risk_multiplier = getattr(self, '_overlay_risk_multiplier', 1.0)
            # Smooth with EMA to avoid sudden jumps (alpha=0.2 for 5-step smoothing)
            if not hasattr(self, '_overlay_risk_ema'):
                self._overlay_risk_ema = overlay_risk_multiplier
            else:
                alpha = 0.2  # EMA smoothing factor
                self._overlay_risk_ema = alpha * overlay_risk_multiplier + (1.0 - alpha) * self._overlay_risk_ema

            # CRITICAL FIX: Clamp smoothed multiplier to [0.5, 1.5] to match model output range
            # Previous [0.7, 1.2] was too tight and limited overlay's effectiveness
            smoothed_overlay_mult = float(np.clip(self._overlay_risk_ema, 0.5, 1.5))

            # VOLATILITY BRAKE: Reduce positions if realized volatility is elevated
            # If vol > 1.8x median, multiply by 0.8 to reduce risk
            # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
            volatility_brake_mult = 1.0
            if t > 50:
                try:
                    # Compute recent realized volatility
                    recent_price_vol = float(np.std(self._price[max(0, t-50):t]))
                    median_price_vol = float(np.median(np.std(self._price[max(0, i-50):i]) for i in range(max(50, t-500), t, 50)))

                    vol_threshold = getattr(self.config, 'volatility_brake_threshold', 1.8)
                    if median_price_vol > 1e-6 and recent_price_vol > vol_threshold * median_price_vol:
                        volatility_brake_mult = 0.8
                        # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
                        if t % 500 == 0:
                            logger.info(f"[VOLATILITY_BRAKE] Step {t}: recent_vol={recent_price_vol:.4f} > {vol_threshold:.1f}x median={median_price_vol:.4f}, reducing by 20%")
                except Exception:
                    pass  # Silently ignore volatility brake errors

            # FGB: Apply overlay risk budget when enabled
            # Action blending is deprecated (overlay_alpha is always 0.0).
            # The overlay's risk_budget multiplier is a key value lever.
            # This ensures the overlay's learned risk regime is always respected.
            #
            # RATIONALE:
            # - Blending is removed; overlay_alpha is always 0.0 (no action override)
            # - Risk budget scaling is orthogonal to action execution
            # - Overlay's learned risk multiplier should always be applied (unless explicitly disabled)
            #
            # This maximizes overlay's contribution to NAV improvement.

            # CRITICAL FIX: Apply risk controller's multiplier
            # Risk controller sets self.risk_multiplier via _apply_risk_control()
            # This must be included in position sizing to ensure risk controller's actions affect trades
            risk_controller_mult = getattr(self, 'risk_multiplier', 1.0)
            
            # Check if overlay risk budget should be applied
            always_apply_overlay = getattr(self.config, 'overlay_apply_risk_budget', True)

            if always_apply_overlay:
                # Apply all multipliers: position_size, risk_controller, overlay, volatility_brake
                combined_multiplier = position_size_multiplier * risk_controller_mult * smoothed_overlay_mult * volatility_brake_mult
            else:
                # Skip overlay risk budget (RL-only sizing), but still apply risk controller
                combined_multiplier = position_size_multiplier * risk_controller_mult * volatility_brake_mult

            strategy_multiplier = 1.0
            strategy_meta = getattr(self, '_current_investor_strategy', None)
            if strategy_meta:
                strategy_label = strategy_meta.get('strategy', 'neutral')
                strategy_scale = float(np.clip(strategy_meta.get('position_scale', 0.0), 0.0, 1.5))
                strat_min = float(getattr(self.config, 'investor_position_scale_min', 0.3))
                strat_max = float(getattr(self.config, 'investor_position_scale_max', 2.0))
                if strategy_label == 'aggressive_trade':
                    boost = float(getattr(self.config, 'investor_aggressive_scale_boost', 1.4))
                    strategy_multiplier = max(strategy_scale, 0.6) * boost
                elif strategy_label == 'trade':
                    boost = float(getattr(self.config, 'investor_trade_scale_boost', 1.1))
                    strategy_multiplier = max(strategy_scale, 0.4) * boost
                elif strategy_label == 'hedge':
                    boost = float(getattr(self.config, 'investor_hedge_scale_boost', 0.7))
                    strategy_multiplier = max(strategy_scale, 0.2) * boost
                else:
                    strategy_multiplier = max(strategy_scale, 0.3)
                strategy_multiplier = float(np.clip(strategy_multiplier, strat_min, strat_max))

            combined_multiplier *= strategy_multiplier
            self._last_investor_strategy_multiplier = strategy_multiplier

            tradeable_capital = available_capital * combined_multiplier

            # DEBUG: Log risk multiplier application (always applied now)
            # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
            if t % 500 == 0 and t > 0:
                logger.debug(f"[OVERLAY RISK] Step {t}: "
                            f"risk_controller={risk_controller_mult:.4f}, rl_mult={position_size_multiplier:.4f}, "
                            f"risk_budget={smoothed_overlay_mult:.4f}, vol_brake={volatility_brake_mult:.4f}, "
                            f"combined={combined_multiplier:.4f}")

            # === STEP 2: Map the agent's normalized actions to target DKK positions ===
            # ENHANCED: Apply Kelly position sizing + regime detection
            action_wind, action_solar, action_hydro = inv_action

            # Get Kelly multipliers (asset-specific)
            # FIX: Extract per-asset forecast confidence from overlay
            forecast_confidences = {
                'wind': 1.0,
                'solar': 1.0,
                'hydro': 1.0
            }

            # Extract forecast confidence from overlay output if available
            if hasattr(self, '_last_overlay_output') and self._last_overlay_output:
                overlay_out = self._last_overlay_output

                # FIX: Use _forecast_errors (per-asset tracking) instead of _forecast_accuracy_tracker
                # _forecast_errors is populated by _track_forecast_accuracy() and tracks by target (wind/solar/hydro)
                # UPDATED: Use configurable confidence floor (default 0.6)
                confidence_floor = getattr(self.config, 'confidence_floor', 0.6)

                if hasattr(self, '_forecast_errors'):
                    for asset in ['wind', 'solar', 'hydro']:
                        # Get recent forecast errors for this asset
                        asset_errors = self._forecast_errors.get(asset, [])
                        if len(asset_errors) >= 10:
                            # Lower MAPE = higher confidence
                            recent_mape = np.mean(list(asset_errors)[-10:])
                            # Map MAPE to confidence: 0% error = 1.0, 50% error = 0.5
                            # Apply configurable floor (default 0.6)
                            confidence = 1.0 - np.clip(recent_mape, 0.0, 0.5)
                            confidence = np.clip(confidence, confidence_floor, 1.0)
                            forecast_confidences[asset] = confidence

                # Fallback: Use pred_reward if forecast errors not available
                elif 'pred_reward' in overlay_out:
                    pred_reward = float(overlay_out['pred_reward'].flatten()[0]) if isinstance(overlay_out['pred_reward'], np.ndarray) else float(overlay_out['pred_reward'])
                    # Map predicted reward to confidence with floor
                    confidence = 0.7 + 0.3 * np.clip(pred_reward / 100.0, 0.0, 1.0)
                    confidence = max(confidence, confidence_floor)
                    forecast_confidences = {'wind': confidence, 'solar': confidence, 'hydro': confidence}

            # Get current volatility regime
            vol_regime = self._current_regime.get('metrics', {}).get('volatility_regime', 1.0) if hasattr(self, '_current_regime') else 1.0

            # Get Kelly multipliers (already includes vol_regime adjustment internally)
            kelly_multipliers = self.kelly_sizer.get_multipliers(forecast_confidences, vol_regime)

            # Get regime multiplier (market-wide)
            regime_mult = self._current_regime.get('position_multiplier', 1.0) if hasattr(self, '_current_regime') else 1.0

            # FIX: Combine multipliers more carefully to avoid extreme values
            # Strategy: Use geometric mean for independent factors, arithmetic for dependent
            #
            # combined_multiplier already includes: position_size_multiplier × overlay_risk × vol_brake
            # kelly_multipliers already includes: kelly_fraction × confidence^2 × vol_adj × corr_adj
            # regime_mult is: regime-based adjustment (0.5-1.5)
            #
            # Problem: Multiplicative stacking can give 0.125x - 3.375x range (too extreme!)
            # Solution: Use weighted geometric mean to dampen extremes

            final_multipliers = {}
            for asset in ['wind', 'solar', 'hydro']:
                # Geometric mean of kelly and regime (dampens extremes)
                kelly_regime = np.sqrt(kelly_multipliers[asset] * regime_mult)

                # Multiply by combined_multiplier (drawdown brake + overlay + vol brake)
                # This is appropriate because combined_multiplier is a safety constraint
                final_mult = combined_multiplier * kelly_regime

                # Clip to reasonable range [0.3, 2.0]
                final_multipliers[asset] = float(np.clip(final_mult, 0.3, 2.0))

            # Apply asset-specific multipliers
            max_pos_size_wind = tradeable_capital * self.config.max_position_size * final_multipliers['wind']
            max_pos_size_solar = tradeable_capital * self.config.max_position_size * final_multipliers['solar']
            max_pos_size_hydro = tradeable_capital * self.config.max_position_size * final_multipliers['hydro']

            target_wind = action_wind * max_pos_size_wind
            target_solar = action_solar * max_pos_size_solar
            target_hydro = action_hydro * max_pos_size_hydro

            # Log combined multipliers periodically
            # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
            if t % 500 == 0 and t > 0:
                kelly_regime_geom = np.sqrt(kelly_multipliers['wind'] * regime_mult)
                logger.info(f"[SIZING] Step {t}: "
                           f"conf={forecast_confidences['wind']:.2f}, "
                           f"kelly={kelly_multipliers['wind']:.2f}, "
                           f"regime={regime_mult:.2f}, "
                           f"k×r_geom={kelly_regime_geom:.2f}, "
                           f"combined={combined_multiplier:.2f}, "
                           f"final={final_multipliers['wind']:.2f}")

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

            if total_traded_notional > self.config.no_trade_threshold:
                transaction_cost_bps = self.config.transaction_cost_bps
                fixed_cost = self.config.transaction_fixed_cost
                transaction_costs = (total_traded_notional * transaction_cost_bps / 10000.0) + fixed_cost

                self.budget -= transaction_costs
                
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

                # P&L-BASED TRAINING: Track action taken for later P&L attribution
                # CRITICAL FIX: Use t (timestep parameter) instead of self.t for consistency
                current_price = float(self._price[t]) if t < len(self._price) else 250.0
                self._action_history.append({
                    'timestep': t,
                    'action': inv_action.copy(),  # The action that was taken
                    'positions': {
                        'wind': target_wind,
                        'solar': target_solar,
                        'hydro': target_hydro
                    },
                    'price': current_price,
                    'nav': self._calculate_fund_nav()
                })

                return total_traded_notional
            else:
                return 0.0  # Ignore very small trades and incur no costs

        except Exception as e:
            # Proper logging should be used here in a production system
            logger.error(f"ERROR in _execute_investor_trades at step {self.t}: {e}")
            return 0.0

    def _norm_price(self, p: float, t: int) -> float:
        """
        Normalize price for overlay features using z-score with ±3σ clipping.
        Aligns with wrapper's price normalization to ensure consistent inputs to overlay model.
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
        except Exception:
            return 0.0

    def _compute_forecast_deltas(self, i: int) -> None:
        """
        Compute forecast z-scores for Tier 2 observations.

        CRITICAL: This must run independently of overlay status!
        Tier 2 needs z-scores for observations even when overlay is disabled.

        Computes and stores:
        - Price z-scores: z_short_price, z_medium_price, z_long_price
        - Generation z-scores: z_short/medium/long_wind/solar/hydro
        - Forecast trust: _forecast_trust
        """
        try:
            if i == 0:
                logger.info(f"[FORECAST_DELTAS] _compute_forecast_deltas called at t={i}")
                logger.info(f"  forecast_generator: {self.forecast_generator}")

            if self.forecast_generator is None:
                if i == 0:
                    logger.warning(f"[FORECAST_DELTAS] forecast_generator is None!")
                return

            # CRITICAL FIX: Predict for CURRENT timestep (i), not next (i+1)
            # Agent acts at timestep i, so it needs forecasts FROM timestep i
            # The forecast model internally adds horizon offsets (i+6, i+24, i+144)
            forecasts = self.forecast_generator.predict_all_horizons(timestep=i)
            if i == 0:
                logger.info(f"[FORECAST_DELTAS] Got forecasts: {list(forecasts.keys()) if forecasts else 'None'}")
            if not forecasts:
                if i == 0:
                    logger.warning(f"[FORECAST_DELTAS] No forecasts returned!")
                return

            # Get current values (raw DKK for price, normalized for generation)
            price_raw = float(self._price_raw[i] if i < len(self._price_raw) else 250.0)
            wind_current = float(self._wind[i] / max(self.wind_scale, 1e-6) if i < len(self._wind) else 0.0)
            solar_current = float(self._solar[i] / max(self.solar_scale, 1e-6) if i < len(self._solar) else 0.0)
            hydro_current = float(self._hydro[i] / max(self.hydro_scale, 1e-6) if i < len(self._hydro) else 0.0)

            # Get forecast values (raw DKK for price, normalized for generation)
            price_short_raw = forecasts.get('price_forecast_short', price_raw)
            price_medium_raw = forecasts.get('price_forecast_medium', price_raw)
            price_long_raw = forecasts.get('price_forecast_long', price_raw)
            
            # OPTIMIZATION: Store forecast/actual pairs for delayed MAPE calculation
            # These will be compared with actual prices at i+6, i+24, i+144 respectively
            horizon_short = self.config.forecast_horizons.get('short', 6)
            horizon_medium = self.config.forecast_horizons.get('medium', 24)
            horizon_long = self.config.forecast_horizons.get('long', 144)
            
            # Store forecast prices for future MAPE calculation
            self._horizon_forecast_pairs['short'].append({
                'timestep': i,
                'forecast_price': price_short_raw,
                'current_price': price_raw,
            })
            self._horizon_forecast_pairs['medium'].append({
                'timestep': i,
                'forecast_price': price_medium_raw,
                'current_price': price_raw,
            })
            self._horizon_forecast_pairs['long'].append({
                'timestep': i,
                'forecast_price': price_long_raw,
                'current_price': price_raw,
            })

            wind_short = forecasts.get('wind_forecast_short', wind_current)
            wind_medium = forecasts.get('wind_forecast_medium', wind_current)
            wind_long = forecasts.get('wind_forecast_long', wind_current)

            solar_short = forecasts.get('solar_forecast_short', solar_current)
            solar_medium = forecasts.get('solar_forecast_medium', solar_current)
            solar_long = forecasts.get('solar_forecast_long', solar_current)

            hydro_short = forecasts.get('hydro_forecast_short', hydro_current)
            hydro_medium = forecasts.get('hydro_forecast_medium', hydro_current)
            hydro_long = forecasts.get('hydro_forecast_long', hydro_current)

            # Compute RAW price deltas (in DKK)
            delta_price_short_raw = price_short_raw - price_raw
            delta_price_medium_raw = price_medium_raw - price_raw
            delta_price_long_raw = price_long_raw - price_raw

            # Compute generation deltas (normalized)
            delta_wind_short = wind_short - wind_current
            delta_wind_medium = wind_medium - wind_current
            delta_wind_long = wind_long - wind_current

            delta_solar_short = solar_short - solar_current
            delta_solar_medium = solar_medium - solar_current
            delta_solar_long = solar_long - solar_current

            delta_hydro_short = hydro_short - hydro_current
            delta_hydro_medium = hydro_medium - hydro_current
            delta_hydro_long = hydro_long - hydro_current

            # FIX Issue #1: Adaptive EMA std initialization (price + generation)
            # Collect first N samples for initialization
            init_samples = getattr(self.config, 'ema_std_init_samples', 20)
            if not self._ema_std_initialized and self._ema_std_init_count < init_samples:
                self._ema_std_init_samples.append({
                    'short': abs(delta_price_short_raw),
                    'medium': abs(delta_price_medium_raw),
                    'long': abs(delta_price_long_raw),
                    'wind': abs(delta_wind_short),      # FIX: Add generation samples
                    'solar': abs(delta_solar_short),    # FIX: Add generation samples
                    'hydro': abs(delta_hydro_short)     # FIX: Add generation samples
                })
                self._ema_std_init_count += 1
                
                # Initialize EMA std from collected samples
                if self._ema_std_init_count >= init_samples:
                    short_values = [s['short'] for s in self._ema_std_init_samples]
                    medium_values = [s['medium'] for s in self._ema_std_init_samples]
                    long_values = [s['long'] for s in self._ema_std_init_samples]
                    wind_values = [s['wind'] for s in self._ema_std_init_samples]      # FIX: Add generation
                    solar_values = [s['solar'] for s in self._ema_std_init_samples]    # FIX: Add generation
                    hydro_values = [s['hydro'] for s in self._ema_std_init_samples]    # FIX: Add generation
                    
                    # Use median for robustness (less sensitive to outliers)
                    self.ema_std_short = float(np.median(short_values)) if short_values else 100.0
                    self.ema_std_medium = float(np.median(medium_values)) if medium_values else 120.0
                    self.ema_std_long = float(np.median(long_values)) if long_values else 150.0
                    
                    # FIX: Initialize generation EMA std from samples
                    self.ema_std_wind = float(np.median(wind_values)) if wind_values else 10.0
                    self.ema_std_solar = float(np.median(solar_values)) if solar_values else 10.0
                    self.ema_std_hydro = float(np.median(hydro_values)) if hydro_values else 10.0
                    
                    # Use higher alpha for first 100 steps (faster convergence)
                    init_alpha = getattr(self.config, 'ema_std_init_alpha', 0.2)
                    self._ema_std_init_alpha = init_alpha
                    self._ema_std_init_steps = 100
                    self._ema_std_initialized = True
                    
                    logger.info(f"[EMA_STD_INIT] Initialized from {init_samples} samples: "
                               f"price: short={self.ema_std_short:.2f}, medium={self.ema_std_medium:.2f}, "
                               f"long={self.ema_std_long:.2f} DKK | "
                               f"generation: wind={self.ema_std_wind:.4f}, solar={self.ema_std_solar:.4f}, "
                               f"hydro={self.ema_std_hydro:.4f}")
            
            # Use adaptive alpha during initialization period
            if self._ema_std_initialized and hasattr(self, '_ema_std_init_steps') and self._ema_std_init_steps > 0:
                current_alpha = self._ema_std_init_alpha
                self._ema_std_init_steps -= 1
            else:
                current_alpha = self.ema_alpha

            # =====================================================================
            # NOVEL: USE FORECAST RETURNS (Bias-Immune) for z-scores
            # =====================================================================
            # Instead of normalizing absolute deltas by EMA std, use RETURNS directly
            # Returns are already normalized by current price, so they're bias-immune
            # We still apply tanh for bounded z-scores, but with a different scale

            # Compute forecast returns (already computed in _forecast_deltas_raw)
            forecast_return_short = delta_price_short_raw / max(abs(price_raw), 1.0)
            forecast_return_medium = delta_price_medium_raw / max(abs(price_raw), 1.0)
            forecast_return_long = delta_price_long_raw / max(abs(price_raw), 1.0)

            # Apply tanh with scaling factor (typical returns are ±5%, so scale by 10x)
            # This maps ±5% returns to ±0.46 z-scores, ±10% to ±0.76, ±20% to ±0.96
            z_short = np.tanh(forecast_return_short * 10.0)
            z_medium = np.tanh(forecast_return_medium * 10.0)
            z_long = np.tanh(forecast_return_long * 10.0)

            # Update EMA std using FORECAST RETURN magnitudes (for tracking typical return volatility)
            # This tracks the typical magnitude of forecast returns, not absolute deltas
            self.ema_std_short = (1.0 - current_alpha) * self.ema_std_short + current_alpha * abs(forecast_return_short)
            self.ema_std_medium = (1.0 - current_alpha) * self.ema_std_medium + current_alpha * abs(forecast_return_medium)
            self.ema_std_long = (1.0 - current_alpha) * self.ema_std_long + current_alpha * abs(forecast_return_long)
            
            # FIX Issue #6: Validate and clamp EMA std values (RETURN-BASED)
            # Returns are typically 0.01-0.50 (1%-50%), so use return-based ranges
            ema_std_min = getattr(self.config, 'ema_std_min_return', 0.01)  # 1% min
            ema_std_max = getattr(self.config, 'ema_std_max_return', 1.0)   # 100% max
            self.ema_std_short = float(np.clip(self.ema_std_short, ema_std_min, ema_std_max))
            self.ema_std_medium = float(np.clip(self.ema_std_medium, ema_std_min, ema_std_max))
            self.ema_std_long = float(np.clip(self.ema_std_long, ema_std_min, ema_std_max))

            # CRITICAL FIX: Save PREVIOUS z-scores before updating (for reward calculation)
            # Rewards should use z-scores from when the agent made the decision (previous timestep)
            # This ensures reward aligns with what the agent observed
            # FIX Issue #3: Only check attribute existence, not value (0.0 is valid)
            if not hasattr(self, 'z_short_price'):
                # First step: use current z-scores (will be computed below)
                self.z_short_price_prev = z_short
                self.z_medium_price_prev = z_medium
                self.z_long_price_prev = z_long
            else:
                # Normal case: use previous values (even if 0.0 is a valid value)
                self.z_short_price_prev = float(getattr(self, 'z_short_price', 0.0))
                self.z_medium_price_prev = float(getattr(self, 'z_medium_price', 0.0))
                self.z_long_price_prev = float(getattr(self, 'z_long_price', 0.0))

            if not hasattr(self, 'z_short_wind'):
                # First step: will be set below
                self.z_short_wind_prev = 0.0
                self.z_short_solar_prev = 0.0
                self.z_short_hydro_prev = 0.0
            else:
                self.z_short_wind_prev = float(getattr(self, 'z_short_wind', 0.0))
                self.z_short_solar_prev = float(getattr(self, 'z_short_solar', 0.0))
                self.z_short_hydro_prev = float(getattr(self, 'z_short_hydro', 0.0))

            # Store PRICE z-scores (for investor observations)
            self.z_short_price = float(np.clip(z_short, -1.0, 1.0))
            self.z_medium_price = float(np.clip(z_medium, -1.0, 1.0))
            self.z_long_price = float(np.clip(z_long, -1.0, 1.0))

            # DEBUG: Log at step 500
            if i == 500:
                logger.debug(f"[DEBUG_COMPUTE_DELTAS] t={i} SET z_short_price={self.z_short_price:.6f} z_medium_price={self.z_medium_price:.6f}")

            # DIAGNOSTIC: Log z-score computation (every 1000 steps)
            if i % 1000 == 0:
                logger.info(f"[Z_SCORE_COMPUTATION_RETURNS] t={i}")
                logger.info(f"  forecast_return_short={forecast_return_short*100:.2f}%, ema_std_short={self.ema_std_short*100:.2f}%")
                logger.info(f"  z_short_raw={z_short:.4f} → z_short_price={self.z_short_price:.4f} (clipped)")
                logger.info(f"  z_medium_raw={z_medium:.4f} → z_medium_price={self.z_medium_price:.4f} (clipped)")
                logger.info(f"  z_long_raw={z_long:.4f} → z_long_price={self.z_long_price:.4f} (clipped)")

            # FIX Issue #5: Store last good z-scores for forecast failure handling
            self._last_good_z_short_price = self.z_short_price
            self._last_good_z_medium_price = self.z_medium_price
            self._last_good_z_long_price = self.z_long_price

            # CRITICAL FIX: Store GENERATION z-scores using proper z-score formula with TRAINING statistics
            # Use scaler statistics from training models (mean/std from training data), not rolling episode stats
            # This ensures z-scores match the training distribution (consistent with forecast models)
            # Fallback to rolling stats only if scalers not available
            
            # Try to get training mean/std from forecast generator scalers first
            wind_mean_val = None
            wind_std_val = None
            solar_mean_val = None
            solar_std_val = None
            hydro_mean_val = None
            hydro_std_val = None
            
            if (self.forecast_generator and hasattr(self.forecast_generator, 'scalers')):
                try:
                    wind_model_key = f"wind_{self.config.forecast_horizons.get('short', 'short')}"
                    if wind_model_key in self.forecast_generator.scalers and 'scaler_y' in self.forecast_generator.scalers[wind_model_key]:
                        scaler_y_wind = self.forecast_generator.scalers[wind_model_key]['scaler_y']
                        if hasattr(scaler_y_wind, 'mean_') and len(scaler_y_wind.mean_) > 0:
                            wind_mean_val = float(scaler_y_wind.mean_[0])
                        if hasattr(scaler_y_wind, 'scale_') and len(scaler_y_wind.scale_) > 0:
                            wind_std_val = max(float(scaler_y_wind.scale_[0]), 1e-6)
                    
                    solar_model_key = f"solar_{self.config.forecast_horizons.get('short', 'short')}"
                    if solar_model_key in self.forecast_generator.scalers and 'scaler_y' in self.forecast_generator.scalers[solar_model_key]:
                        scaler_y_solar = self.forecast_generator.scalers[solar_model_key]['scaler_y']
                        if hasattr(scaler_y_solar, 'mean_') and len(scaler_y_solar.mean_) > 0:
                            solar_mean_val = float(scaler_y_solar.mean_[0])
                        if hasattr(scaler_y_solar, 'scale_') and len(scaler_y_solar.scale_) > 0:
                            solar_std_val = max(float(scaler_y_solar.scale_[0]), 1e-6)
                    
                    hydro_model_key = f"hydro_{self.config.forecast_horizons.get('short', 'short')}"
                    if hydro_model_key in self.forecast_generator.scalers and 'scaler_y' in self.forecast_generator.scalers[hydro_model_key]:
                        scaler_y_hydro = self.forecast_generator.scalers[hydro_model_key]['scaler_y']
                        if hasattr(scaler_y_hydro, 'mean_') and len(scaler_y_hydro.mean_) > 0:
                            hydro_mean_val = float(scaler_y_hydro.mean_[0])
                        if hasattr(scaler_y_hydro, 'scale_') and len(scaler_y_hydro.scale_) > 0:
                            hydro_std_val = max(float(scaler_y_hydro.scale_[0]), 1e-6)
                except Exception as e:
                    logger.debug(f"[Z_SCORE] Failed to get scaler statistics: {e}")
            
            # Fallback to rolling stats if scalers not available
            if wind_mean_val is None:
                wind_mean_val = float(self._wind_mean[i]) if i < len(self._wind_mean) else (float(self._wind[i]) if i < len(self._wind) else 0.0)
            if wind_std_val is None:
                wind_std_val = max(float(self._wind_std[i]), 1e-6) if i < len(self._wind_std) else 1.0
            
            if solar_mean_val is None:
                solar_mean_val = float(self._solar_mean[i]) if i < len(self._solar_mean) else (float(self._solar[i]) if i < len(self._solar) else 0.0)
            if solar_std_val is None:
                solar_std_val = max(float(self._solar_std[i]), 1e-6) if i < len(self._solar_std) else 1.0
            
            if hydro_mean_val is None:
                hydro_mean_val = float(self._hydro_mean[i]) if i < len(self._hydro_mean) else (float(self._hydro[i]) if i < len(self._hydro) else 0.0)
            if hydro_std_val is None:
                hydro_std_val = max(float(self._hydro_std[i]), 1e-6) if i < len(self._hydro_std) else 1.0
            
            # Get raw forecast values (forecast generator returns raw MW values)
            # wind_current, solar_current, hydro_current are normalized, so convert back to raw
            wind_current_raw = float(self._wind[i]) if i < len(self._wind) else 0.0
            solar_current_raw = float(self._solar[i]) if i < len(self._solar) else 0.0
            hydro_current_raw = float(self._hydro[i]) if i < len(self._hydro) else 0.0
            
            # Forecasts from predict_all_horizons are RAW values (not normalized)
            # If defaults were used (wind_current), they're normalized, so convert to raw
            wind_short_raw = float(wind_short * max(self.wind_scale, 1e-6)) if wind_short == wind_current else float(wind_short)
            wind_medium_raw = float(wind_medium * max(self.wind_scale, 1e-6)) if wind_medium == wind_current else float(wind_medium)
            wind_long_raw = float(wind_long * max(self.wind_scale, 1e-6)) if wind_long == wind_current else float(wind_long)
            
            solar_short_raw = float(solar_short * max(self.solar_scale, 1e-6)) if solar_short == solar_current else float(solar_short)
            solar_medium_raw = float(solar_medium * max(self.solar_scale, 1e-6)) if solar_medium == solar_current else float(solar_medium)
            solar_long_raw = float(solar_long * max(self.solar_scale, 1e-6)) if solar_long == solar_current else float(solar_long)
            
            hydro_short_raw = float(hydro_short * max(self.hydro_scale, 1e-6)) if hydro_short == hydro_current else float(hydro_short)
            hydro_medium_raw = float(hydro_medium * max(self.hydro_scale, 1e-6)) if hydro_medium == hydro_current else float(hydro_medium)
            hydro_long_raw = float(hydro_long * max(self.hydro_scale, 1e-6)) if hydro_long == hydro_current else float(hydro_long)
            
            # Compute proper z-scores: (forecast - mean) / std
            self.z_short_wind = float(np.clip((wind_short_raw - wind_mean_val) / wind_std_val, -3.0, 3.0))
            self.z_medium_wind = float(np.clip((wind_medium_raw - wind_mean_val) / wind_std_val, -3.0, 3.0))
            self.z_long_wind = float(np.clip((wind_long_raw - wind_mean_val) / wind_std_val, -3.0, 3.0))

            self.z_short_solar = float(np.clip((solar_short_raw - solar_mean_val) / solar_std_val, -3.0, 3.0))
            self.z_medium_solar = float(np.clip((solar_medium_raw - solar_mean_val) / solar_std_val, -3.0, 3.0))
            self.z_long_solar = float(np.clip((solar_long_raw - solar_mean_val) / solar_std_val, -3.0, 3.0))

            self.z_short_hydro = float(np.clip((hydro_short_raw - hydro_mean_val) / hydro_std_val, -3.0, 3.0))
            self.z_medium_hydro = float(np.clip((hydro_medium_raw - hydro_mean_val) / hydro_std_val, -3.0, 3.0))
            self.z_long_hydro = float(np.clip((hydro_long_raw - hydro_mean_val) / hydro_std_val, -3.0, 3.0))
            
            # OPTION 2: Store z-scores in history buffer for forward-looking reward alignment
            # Store current z-scores with timestep i for future reward calculation
            # When computing reward at step i+6, we'll look up z-scores from step i
            # CRITICAL: Must be AFTER all z-scores are computed (including generation z-scores)
            self._z_score_history[i] = {
                'z_short': float(z_short),
                'z_medium': float(z_medium),
                'z_long': float(z_long),
                'z_short_wind': float(self.z_short_wind),
                'z_short_solar': float(self.z_short_solar),
                'z_short_hydro': float(self.z_short_hydro),
                'forecast_trust': float(getattr(self, '_forecast_trust', 0.5)),
            }
            
            # CRITICAL FIX: Cleanup old history (keep only last _z_score_history_max_age timesteps)
            # Remove multiple old entries at once to prevent OOM
            if len(self._z_score_history) > self._z_score_history_max_age:
                # Remove all entries older than max_age
                sorted_keys = sorted(self._z_score_history.keys())
                keys_to_remove = sorted_keys[:-self._z_score_history_max_age]
                for key in keys_to_remove:
                    del self._z_score_history[key]
            
            # FIX Issue #4: Update EMA std for generation forecasts using adaptive alpha
            self.ema_std_wind = (1.0 - current_alpha) * self.ema_std_wind + current_alpha * abs(delta_wind_short)
            self.ema_std_solar = (1.0 - current_alpha) * self.ema_std_solar + current_alpha * abs(delta_solar_short)
            self.ema_std_hydro = (1.0 - current_alpha) * self.ema_std_hydro + current_alpha * abs(delta_hydro_short)
            
            # FIX Issue #5: Validate and clamp EMA std values (use normalized range for generation)
            ema_std_min = getattr(self.config, 'ema_std_min', 10.0)
            ema_std_max = getattr(self.config, 'ema_std_max', 500.0)
            # Generation is normalized, use different range from config
            gen_ema_std_min = getattr(self.config, 'gen_ema_std_min', 0.01)  # Normalized (not DKK)
            gen_ema_std_max = getattr(self.config, 'gen_ema_std_max', 1.0)   # Normalized (not DKK)
            self.ema_std_wind = float(np.clip(self.ema_std_wind, gen_ema_std_min, gen_ema_std_max))
            self.ema_std_solar = float(np.clip(self.ema_std_solar, gen_ema_std_min, gen_ema_std_max))
            self.ema_std_hydro = float(np.clip(self.ema_std_hydro, gen_ema_std_min, gen_ema_std_max))
            
            # FIX Issue #5: Store last good generation z-scores
            self._last_good_z_short_wind = self.z_short_wind
            self._last_good_z_short_solar = self.z_short_solar
            self._last_good_z_short_hydro = self.z_short_hydro
            self._last_good_z_medium_wind = self.z_medium_wind
            self._last_good_z_medium_solar = self.z_medium_solar
            self._last_good_z_medium_hydro = self.z_medium_hydro
            self._last_good_z_long_wind = self.z_long_wind
            self._last_good_z_long_solar = self.z_long_solar
            self._last_good_z_long_hydro = self.z_long_hydro

            # Compute direction consistency and mwdir (needed for overlay features)
            direction_consistency = float(np.mean([np.sign(z) if z != 0 else 0 for z in [z_short, z_medium, z_long]]))
            horizon_weights = {'short': 0.26, 'medium': 0.12, 'long': 0.04}
            mwdir = (horizon_weights['short'] * z_short +
                    horizon_weights['medium'] * z_medium +
                    horizon_weights['long'] * z_long)

            # Legacy variables (for backward compatibility)
            self.z_short = self.z_short_price
            self.z_medium = self.z_medium_price
            self.z_long = self.z_long_price
            self.direction_consistency = float(np.clip(direction_consistency, -1.0, 1.0))
            self.mwdir = float(np.clip(mwdir, -1.0, 1.0))
            self.abs_mwdir = float(np.clip(abs(mwdir), 0.0, 1.0))

            # CRITICAL FIX: Update calibration tracker HERE (not in _build_overlay_features)
            # This ensures trust is computed even when overlay_enabled=False (Tier 2)
            # The calibration tracker needs the EMA std values from BEFORE the update
            # Store them for the calibration tracker
            self._ema_std_short_for_calibration = self.ema_std_short
            self._ema_std_medium_for_calibration = self.ema_std_medium
            self._ema_std_long_for_calibration = self.ema_std_long

            # Update calibration tracker with forecast/realized pairs
            self._update_calibration_tracker(i)

            # Store forecast trust (from CalibrationTracker if available)
            # This must come AFTER _update_calibration_tracker() to get the latest trust value
            # FIX: Pass recent MAPE to make trust responsive to forecast quality
            if hasattr(self, 'calibration_tracker') and self.calibration_tracker is not None:
                # Get recent MAPE from short horizon (most relevant for trust)
                recent_mape = None
                if hasattr(self, '_horizon_mape') and len(self._horizon_mape.get('short', [])) > 0:
                    # Use average of last 10 MAPE values for stability
                    recent_mape_values = list(self._horizon_mape['short'])[-10:]
                    if len(recent_mape_values) > 0:
                        recent_mape = float(np.mean(recent_mape_values))

                self._forecast_trust = self.calibration_tracker.get_trust(horizon="short", recent_mape=recent_mape)
            else:
                self._forecast_trust = 0.5  # Default neutral trust

            # =====================================================================
            # NOVEL: FORECAST RETURNS INTEGRATION (Bias-Immune)
            # =====================================================================
            # Instead of capacity-based deltas, use RETURNS (percentage changes)
            # Returns are stationary and bias cancels out in the division
            # This focuses on what matters: price movements, not absolute levels

            # Compute FORECAST RETURNS (percentage change from current price)
            forecast_return_short = delta_price_short_raw / max(abs(price_raw), 1.0)
            forecast_return_medium = delta_price_medium_raw / max(abs(price_raw), 1.0)
            forecast_return_long = delta_price_long_raw / max(abs(price_raw), 1.0)

            # Store as forecast deltas (for backward compatibility with existing code)
            self._forecast_deltas_raw = {
                'short': float(forecast_return_short),   # Return-based (bias-immune)
                'medium': float(forecast_return_medium), # Return-based (bias-immune)
                'long': float(forecast_return_long),     # Return-based (bias-immune)
            }

            # Store price capacity for CSV logging (backward compatibility)
            price_capacity = 6982.0  # Forecast model capacity (for MAPE calculation)
            self._current_price_floor = float(price_capacity)

            # DIAGNOSTIC LOGGING: Track forecast returns (every 200 steps)
            if i % 200 == 0:
                logger.debug(f"[FORECAST_RETURNS] t={i} price={price_raw:.2f} short={forecast_return_short*100:.2f}% med={forecast_return_medium*100:.2f}% long={forecast_return_long*100:.2f}%")
                logger.info(f"[FORECAST_RETURNS] t={i} price_raw={price_raw:.2f} "
                           f"return_short={forecast_return_short:.6f} ({forecast_return_short*100:.2f}%) "
                           f"return_medium={forecast_return_medium:.6f} ({forecast_return_medium*100:.2f}%) "
                           f"return_long={forecast_return_long:.6f} ({forecast_return_long*100:.2f}%) "
                           f"abs_short={delta_price_short_raw:.2f} abs_medium={delta_price_medium_raw:.2f} abs_long={delta_price_long_raw:.2f}")

            # CRITICAL FIX: Store MAPE thresholds in CAPACITY-BASED form (Fix #18)
            # FIX #19: MAPE and deltas now use same denominator (price_capacity)
            # FIX #22: price_capacity = 6982 DKK (forecast model capacity)
            # This ensures consistent comparison between forecast deltas and MAPE thresholds
            self._mape_thresholds = {}
            for horizon in ['short', 'medium', 'long']:
                if hasattr(self, '_horizon_mape') and len(self._horizon_mape.get(horizon, [])) > 0:
                    # Use average of last 10 MAPE values (capacity-based after Fix #18)
                    recent_mape_values = list(self._horizon_mape[horizon])[-10:]
                    if len(recent_mape_values) > 0:
                        # MAPE is capacity-based (e.g., 0.005 = 10 DKK error / 2000 DKK capacity)
                        mape_capacity = float(np.mean(recent_mape_values))
                        self._mape_thresholds[horizon] = mape_capacity  # Store as capacity-based
                    else:
                        # Default: 0.01 (1% of capacity = 20 DKK error)
                        self._mape_thresholds[horizon] = 0.01
                else:
                    # Default: 2% MAPE
                    self._mape_thresholds[horizon] = 0.02

            # DIAGNOSTIC LOGGING: Track MAPE thresholds (every 200 steps)
            if i % 200 == 0:
                logger.debug(f"[MAPE] t={i} short={self._mape_thresholds.get('short', 0.0)*100:.2f}% med={self._mape_thresholds.get('medium', 0.0)*100:.2f}% long={self._mape_thresholds.get('long', 0.0)*100:.2f}%")
                logger.info(f"[MAPE_THRESHOLDS] t={i} "
                           f"short={self._mape_thresholds.get('short', 0.0):.4f} ({len(self._horizon_mape.get('short', []))} samples) "
                           f"medium={self._mape_thresholds.get('medium', 0.0):.4f} ({len(self._horizon_mape.get('medium', []))} samples) "
                           f"long={self._mape_thresholds.get('long', 0.0):.4f} ({len(self._horizon_mape.get('long', []))} samples)")

            # Diagnostic logging (first step only)
            if i == 0:
                logger.info(f"[FORECAST_DELTAS] t={i} price_raw={price_raw:.2f}")
                logger.info(f"  price_short_raw={price_short_raw:.2f} (delta={delta_price_short_raw:.4f}, z={z_short:.4f})")
                logger.info(f"  price_medium_raw={price_medium_raw:.2f} (delta={delta_price_medium_raw:.4f}, z={z_medium:.4f})")
                logger.info(f"  price_long_raw={price_long_raw:.2f} (delta={delta_price_long_raw:.4f}, z={z_long:.4f})")
                logger.info(f"  EMA stds: short={self.ema_std_short:.4f}, med={self.ema_std_medium:.4f}, long={self.ema_std_long:.4f}")
                logger.info(f"  direction_consistency={direction_consistency:.4f}, mwdir={mwdir:.4f}")
                logger.info(f"  Forecast trust: {self._forecast_trust:.4f}")

        except Exception as e:
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

        CRITICAL: This must be called independently of overlay_enabled setting!
        Previously this was inside _build_overlay_features(), causing Tier 2 to never
        update the calibration tracker (since overlay_enabled=False in Tier 2).

        Args:
            i: Current timestep index
        """
        # CRITICAL FIX: Horizon-matched calibration
        # z_short at time t = tanh((price_forecast_short[t] - price[t]) / ema_std)
        # This is a forecast of the price change from t to t+6
        # We need to compare this with the REALIZED price change from t to t+6
        if self.calibration_tracker is not None and hasattr(self, '_price_raw') and i >= 6:
            try:
                # Get ema_std values that were used for forecast computation
                # These are stored in _compute_forecast_deltas BEFORE the EMA update
                ema_std_short_for_forecast = getattr(self, '_ema_std_short_for_calibration', getattr(self, 'ema_std_short', 10.0))
                ema_std_medium_for_forecast = getattr(self, '_ema_std_medium_for_calibration', getattr(self, 'ema_std_medium', 10.0))
                ema_std_long_for_forecast = getattr(self, '_ema_std_long_for_calibration', getattr(self, 'ema_std_long', 10.0))

                # CRITICAL FIX: Store ema_std along with forecast for consistent normalization
                # The forecast z_short was computed using ema_std at time t (BEFORE update)
                # We must use the SAME ema_std when computing realized signal at t+6
                # Otherwise, changing volatility creates systematic bias!
                self._forecast_history.append({
                    't': i,
                    'price_raw': float(self._price_raw[i]),
                    'z_short': self.z_short,  # NOW using current step's value!
                    'z_medium': self.z_medium,
                    'z_long': self.z_long,
                    'mwdir': self.mwdir,
                    'ema_std_short': ema_std_short_for_forecast,  # CRITICAL: Store ema_std BEFORE update
                    'ema_std_medium': ema_std_medium_for_forecast,
                    'ema_std_long': ema_std_long_for_forecast,
                })

                # Deque automatically keeps only last 200 steps (no manual trimming needed)

                # OPTIMIZATION: Calculate per-horizon MAPE for adaptive weighting
                # Compare stored forecasts with actual prices at matching horizons
                horizon_short = self.config.forecast_horizons.get('short', 6)
                horizon_medium = self.config.forecast_horizons.get('medium', 24)
                horizon_long = self.config.forecast_horizons.get('long', 144)
                
                current_price_raw = float(self._price_raw[i])
                
                # Calculate MAPE for short horizon
                # CRITICAL FIX: Forecast at step i-horizon_short predicts price at timestep i
                # So we compare forecast from i-horizon_short with actual at i (matching horizon)
                # FIX #18: Use CAPACITY-BASED MAPE (like forecast models) instead of STANDARD MAPE
                # This prevents MAPE explosion at low prices (10-50 DKK)
                # FIX #22: Use forecast model capacity (6982 DKK) to match training data
                price_capacity = 6982.0  # Forecast model capacity (max price in training data)
                if i >= horizon_short and len(self._horizon_forecast_pairs['short']) > 0:
                    # Find forecast made at i-horizon_short (matches corrected horizon logic)
                    for pair in reversed(self._horizon_forecast_pairs['short']):
                        if pair['timestep'] == i - horizon_short:  # = i - 6 for horizon=6
                            forecast_price = pair['forecast_price']
                            if abs(forecast_price) > 1e-6:
                                # FIX #18: Capacity-based MAPE (matches forecast model training)
                                # OLD: mape_short = abs(current_price_raw - forecast_price) / abs(forecast_price)
                                # NEW: mape_short = abs(current_price_raw - forecast_price) / price_capacity
                                mape_short = abs(current_price_raw - forecast_price) / price_capacity
                                self._horizon_mape['short'].append(float(np.clip(mape_short, 0.0, 1.0)))
                            break

                # Calculate MAPE for medium horizon (compare forecast from i-25 with actual at i)
                if i >= horizon_medium and len(self._horizon_forecast_pairs['medium']) > 0:
                    for pair in reversed(self._horizon_forecast_pairs['medium']):
                        if pair['timestep'] == i - horizon_medium:  # = i - 24 for horizon=24
                            forecast_price = pair['forecast_price']
                            if abs(forecast_price) > 1e-6:
                                # FIX #18: Capacity-based MAPE
                                mape_medium = abs(current_price_raw - forecast_price) / price_capacity
                                self._horizon_mape['medium'].append(float(np.clip(mape_medium, 0.0, 1.0)))
                            break

                # Calculate MAPE for long horizon (compare forecast from i-145 with actual at i)
                # FIX #21: Deque maxlen increased to 200 to ensure pairs from 144 timesteps ago are available
                if i >= horizon_long and len(self._horizon_forecast_pairs['long']) > 0:
                    for pair in reversed(self._horizon_forecast_pairs['long']):
                        if pair['timestep'] == i - horizon_long:  # = i - 144 for horizon=144
                            forecast_price = pair['forecast_price']
                            if abs(forecast_price) > 1e-6:
                                # FIX #18: Capacity-based MAPE
                                mape_long = abs(current_price_raw - forecast_price) / price_capacity
                                self._horizon_mape['long'].append(float(np.clip(mape_long, 0.0, 1.0)))
                            break
                
                # CRITICAL FIX: Track forecast returns and actual returns for correlation-based weighting
                # Calculate returns for each horizon and store pairs for correlation computation
                for horizon_name, horizon_steps in [('short', horizon_short), ('medium', horizon_medium), ('long', horizon_long)]:
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

                # Update calibration tracker with SHORT HORIZON (6 steps ahead)
                # At time t, we have:
                # - forecast_signal = z_short[t] = tanh((price_forecast_short[t] - price[t]) / ema_std[t])
                # - This predicts the price change from t to t+6
                # At time t+6, we can compute:
                # - realized_signal = tanh((price[t+6] - price[t]) / ema_std[t])  <-- SAME ema_std!
                # - This is the actual price change from t to t+6
                if len(self._forecast_history) >= 7:  # Need at least 7 entries (t-6 to t)
                    forecast_entry = self._forecast_history[-7]  # Entry at time t-6
                    forecast_signal = forecast_entry['z_short']  # Forecast made at t-6 for price change from t-6 to t

                    # CRITICAL FIX: Compute realized price change from t-6 to t (now)
                    # This matches the forecast horizon (6 steps)
                    current_price_raw = float(self._price_raw[i])
                    price_at_forecast_time = forecast_entry['price_raw']  # Price at t-6
                    realized_delta_raw = current_price_raw - price_at_forecast_time

                    # CRITICAL FIX: Use ema_std from t-6 (when forecast was made) for consistent normalization
                    # This ensures forecast and realized use the SAME volatility scaling
                    ema_std_at_forecast_time = forecast_entry.get('ema_std_short', getattr(self, 'ema_std_short', 10.0))
                    realized_signal = np.tanh(realized_delta_raw / (ema_std_at_forecast_time + 1e-6))

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

                        logger.info(f"[CALIBRATION] t={i} | trust={current_trust:.3f} | buffer_size={buffer_size}/2016 | forecast={forecast_signal:.3f} | realized={realized_signal:.3f} | delta_raw={realized_delta_raw:.2f} DKK | ema_std={ema_std_at_forecast_time:.2f}")
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
                                    z_short_recomputed = np.tanh(delta_forecast_raw / (ema_std_at_forecast_time + 1e-6))

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
                if i % 1000 == 0:
                    logger.warning(f"[CALIBRATION] Update failed at t={i}: {e}")
                import traceback
                if i % 1000 == 0:
                    logger.warning(f"[CALIBRATION] Traceback: {traceback.format_exc()}")

    def _build_overlay_features(self, i: int) -> Optional[np.ndarray]:
        """
        Build features for DL overlay inference (34D: 28D base + 6D deltas).
        Returns shape (1, 34) for batch processing.

        Features (34D):
        - Market (0-5): [price_n, wind_n, solar_n, hydro_n, load_n, risk]
        - Positions (6-8): [wind_pos, solar_pos, hydro_pos] (normalized by fund size)
        - Forecasts (9-24): [immediate, short, medium, long] × [wind, solar, hydro, price]
        - Portfolio (25-27): [capital_allocation, investment_freq, forecast_confidence]
        - Deltas (28-33): [Δprice_short, Δprice_med, Δprice_long, direction_consistency, mwdir, |mwdir|]

        STRICT: Only 34D mode supported (28D base + 6D deltas). Fails fast if feature_dim != 34.
        """
        try:
            feature_dim = getattr(self, 'feature_dim', OVERLAY_FEATURE_DIM)

            # STRICT: Only 34D mode supported (28D base + 6D deltas)
            if feature_dim != OVERLAY_FEATURE_DIM:
                raise ValueError(f"_build_overlay_features only supports 34D mode (28D base + 6D deltas), got {feature_dim}D")

            # DEBUG: Log feature dimension at start (use self.t instead of i)
            if self.t == 0:
                logger.info(f"[OVERLAY_FEATURES] Building 34D features (28D base + 6D deltas: Δprice_short/med/long, dir_consistency, mwdir, |mwdir|)")

            if feature_dim == OVERLAY_FEATURE_DIM:
                # 34D: Market(6) + Positions(3) + Forecasts(16) + Portfolio(3) + Deltas(6)
                # Comprehensive feature set for 34D Forecast-Aware mode with directional signals
                # Market: [price, wind, solar, hydro, load, risk]
                # Forecasts: [immediate, short, medium, long] × [wind, solar, hydro, price] = 16D
                # Portfolio: [capital_allocation, investment_freq, forecast_confidence]
                # Deltas: [Δprice_short, Δprice_med, Δprice_long, direction_consistency, mwdir, |mwdir|]
                features = np.zeros((1, 34), dtype=np.float32)

                # Market state (0-5) - CRITICAL: Keep rich market state for 34D mode (28D base + 6D deltas)
                # CRITICAL FIX: Use raw DKK price for delta computation, not normalized price!
                price_raw = float(self._price_raw[i] if i < len(self._price_raw) else 0.0)
                features[0, 0] = self._norm_price(price_raw, i)
                features[0, 1] = float(self._wind[i] / max(self.wind_scale, 1e-6) if i < len(self._wind) else 0.0)
                features[0, 2] = float(self._solar[i] / max(self.solar_scale, 1e-6) if i < len(self._solar) else 0.0)
                features[0, 3] = float(self._hydro[i] / max(self.hydro_scale, 1e-6) if i < len(self._hydro) else 0.0)
                features[0, 4] = float(self._load[i] / max(self.load_scale, 1e-6) if i < len(self._load) else 0.0)
                features[0, 5] = float(self._riskS[i] if i < len(self._riskS) else 0.3)

                # Positions (6-8)
                fund_size = max(self.init_budget, 1e6)
                features[0, 6] = float(self.financial_positions.get('wind_instrument_value', 0.0) / fund_size)
                features[0, 7] = float(self.financial_positions.get('solar_instrument_value', 0.0) / fund_size)
                features[0, 8] = float(self.financial_positions.get('hydro_instrument_value', 0.0) / fund_size)

                # Multi-horizon forecasts (9-24) - 16 features
                # CRITICAL FIX: Use actual multi-horizon forecasts (immediate, short, medium, long)
                # Indices 9-12: Immediate horizon [wind, solar, hydro, price]
                # Indices 13-16: Short horizon [wind, solar, hydro, price]
                # Indices 17-20: Medium horizon [wind, solar, hydro, price]
                # Indices 21-24: Long horizon [wind, solar, hydro, price]
                wind_scale = max(self.wind_scale, 1e-6)
                solar_scale = max(self.solar_scale, 1e-6)
                hydro_scale = max(self.hydro_scale, 1e-6)

                # Helper to safely get forecast value with fallback
                def get_forecast(target: str, horizon: str, fallback: float) -> float:
                    """Get forecast value, falling back to nearest available horizon if needed."""
                    array_name = f"_{target}_forecast_{horizon}"
                    if hasattr(self, array_name) and i < len(getattr(self, array_name)):
                        val = getattr(self, array_name)[i]
                        if not np.isnan(val):
                            return float(val)
                    # Fallback to nearest available horizon
                    for alt_horizon in ["immediate", "short", "medium", "long"]:
                        if alt_horizon == horizon:
                            continue
                        alt_array_name = f"_{target}_forecast_{alt_horizon}"
                        if hasattr(self, alt_array_name) and i < len(getattr(self, alt_array_name)):
                            alt_val = getattr(self, alt_array_name)[i]
                            if not np.isnan(alt_val):
                                return float(alt_val)
                    return fallback

                # Current values as baseline
                wind_current = float(self._wind[i] / wind_scale if i < len(self._wind) else 0.0)
                solar_current = float(self._solar[i] / solar_scale if i < len(self._solar) else 0.0)
                hydro_current = float(self._hydro[i] / hydro_scale if i < len(self._hydro) else 0.0)
                price_current = self._norm_price(price_raw, i)

                # Immediate horizon (9-12) - use current values
                features[0, 9] = wind_current
                features[0, 10] = solar_current
                features[0, 11] = hydro_current
                features[0, 12] = price_current

                # Short horizon (13-16) - use actual short-term forecasts
                wind_short_raw = get_forecast("wind", "short", wind_current * wind_scale)
                solar_short_raw = get_forecast("solar", "short", solar_current * solar_scale)
                hydro_short_raw = get_forecast("hydro", "short", hydro_current * hydro_scale)
                price_short_raw = get_forecast("price", "short", price_raw)

                wind_short = wind_short_raw / wind_scale
                solar_short = solar_short_raw / solar_scale
                hydro_short = hydro_short_raw / hydro_scale
                price_short = self._norm_price(price_short_raw, i)

                features[0, 13] = wind_short
                features[0, 14] = solar_short
                features[0, 15] = hydro_short
                features[0, 16] = price_short

                # Medium horizon (17-20) - use actual medium-term forecasts
                wind_medium_raw = get_forecast("wind", "medium", wind_current * wind_scale)
                solar_medium_raw = get_forecast("solar", "medium", solar_current * solar_scale)
                hydro_medium_raw = get_forecast("hydro", "medium", hydro_current * hydro_scale)
                price_medium_raw = get_forecast("price", "medium", price_raw)

                wind_medium = wind_medium_raw / wind_scale
                solar_medium = solar_medium_raw / solar_scale
                hydro_medium = hydro_medium_raw / hydro_scale
                price_medium = self._norm_price(price_medium_raw, i)

                features[0, 17] = wind_medium
                features[0, 18] = solar_medium
                features[0, 19] = hydro_medium
                features[0, 20] = price_medium

                # Long horizon (21-24) - use actual long-term forecasts
                wind_long_raw = get_forecast("wind", "long", wind_current * wind_scale)
                solar_long_raw = get_forecast("solar", "long", solar_current * solar_scale)
                hydro_long_raw = get_forecast("hydro", "long", hydro_current * hydro_scale)
                price_long_raw = get_forecast("price", "long", price_raw)

                wind_long = wind_long_raw / wind_scale
                solar_long = solar_long_raw / solar_scale
                hydro_long = hydro_long_raw / hydro_scale
                price_long = self._norm_price(price_long_raw, i)

                features[0, 21] = wind_long
                features[0, 22] = solar_long
                features[0, 23] = hydro_long
                features[0, 24] = price_long

                # Compute price deltas for delta features (28-33)
                delta_price_short_raw = price_short_raw - price_raw
                delta_price_medium_raw = price_medium_raw - price_raw
                delta_price_long_raw = price_long_raw - price_raw

                # Normalize deltas by EMA std and apply tanh for bounded z-scores
                z_short = np.tanh(delta_price_short_raw / (getattr(self, 'ema_std_short', 10.0) + 1e-6))
                z_medium = np.tanh(delta_price_medium_raw / (getattr(self, 'ema_std_medium', 10.0) + 1e-6))
                z_long = np.tanh(delta_price_long_raw / (getattr(self, 'ema_std_long', 10.0) + 1e-6))

                # Compute direction consistency and mwdir
                direction_consistency = float(np.mean([np.sign(z) if z != 0 else 0 for z in [z_short, z_medium, z_long]]))
                horizon_weights = {'short': 0.26, 'medium': 0.12, 'long': 0.04}
                mwdir = (horizon_weights['short'] * z_short +
                        horizon_weights['medium'] * z_medium +
                        horizon_weights['long'] * z_long)

                # Portfolio metrics (25-27) - 34D base (28D base + 6D deltas)
                features[0, 25] = float(self.capital_allocation_fraction)
                features[0, 26] = float(self.investment_freq / 100.0)
                features[0, 27] = float(self._get_forecast_confidence())

                # === APPEND DELTA FEATURES (28-33) ===
                # These 6 features expose forecast directionality to the overlay model
                # Using normalized deltas (z_short, z_medium, z_long) for stability
                features[0, 28] = float(np.clip(z_short, -1.0, 1.0))  # Normalized Δprice_short
                features[0, 29] = float(np.clip(z_medium, -1.0, 1.0))  # Normalized Δprice_medium
                features[0, 30] = float(np.clip(z_long, -1.0, 1.0))    # Normalized Δprice_long
                features[0, 31] = float(np.clip(direction_consistency, -1.0, 1.0))  # direction_consistency
                features[0, 32] = float(np.clip(mwdir, -1.0, 1.0))               # mwdir (normalized)
                features[0, 33] = float(np.clip(abs(mwdir), 0.0, 1.0))           # |mwdir| (magnitude)

                # DEBUG: Log forecast deltas and directional signals (with normalization info)
                if i % 500 == 0 and i > 0:
                    has_short = (wind_short != wind_current or solar_short != solar_current or
                                hydro_short != hydro_current or price_short != price_current)
                    has_medium = (wind_medium != wind_current or solar_medium != solar_current or
                                 hydro_medium != hydro_current or price_medium != price_current)
                    has_long = (wind_long != wind_current or solar_long != solar_current or
                               hydro_long != hydro_current or price_long != price_current)
                    logger.debug(f"[overlay/deltas] Step {i}: has_short={has_short}, has_medium={has_medium}, has_long={has_long} | "
                                f"RAW_deltas_DKK: Δshort={delta_price_short_raw:.2f}, Δmed={delta_price_medium_raw:.2f}, Δlong={delta_price_long_raw:.2f} | "
                                f"normalized_z: z_short={z_short:.3f}, z_medium={z_medium:.3f}, z_long={z_long:.3f} | "
                                f"ema_stds_DKK: short={self.ema_std_short:.2f}, med={self.ema_std_medium:.2f}, long={self.ema_std_long:.2f} | "
                                f"direction_consistency={direction_consistency:.3f}, mwdir={mwdir:.3f} | "
                                f"features[28-33]=[{features[0,28]:.3f}, {features[0,29]:.3f}, {features[0,30]:.3f}, {features[0,31]:.3f}, {features[0,32]:.3f}, {features[0,33]:.3f}]")

                return features

            else:
                logger.warning(f"[OVERLAY_FEATURES] Unknown feature_dim={feature_dim}, expected 34D (28D base + 6D deltas), returning None")
                return None

        except Exception as e:
            logger.debug(f"[OVERLAY_FEATURES] Error building overlay features at step {i}: {e}")
            return None

    def _collect_overlay_experience(self, i: int):
        """Collect experience for overlay model training"""
        try:
            # Get current features
            features = self._build_overlay_features(i)
            if features is None:
                return

            # Store in history for target computation
            self._overlay_feature_history.append(features[0])  # Remove batch dimension

            # Compute targets based on recent history
            if len(self._overlay_feature_history) < 5:
                return  # Need some history to compute meaningful targets

            # 1. bridge_vec_target: Smooth recent agent actions
            bridge_target = self._compute_bridge_target()

            # 2. risk_budget_target: Based on recent P&L and drawdown
            risk_target = self._compute_risk_budget_target()

            # 3. pred_reward_target: Look-ahead reward signal
            pred_reward_target = self._compute_pred_reward_target()

            # 4. Strategy targets: Based on market regime
            strat_targets = self._compute_strategy_targets()

            # Create experience tuple (OPTIMAL: All 4 horizons)
            experience = {
                'features': features[0].astype(np.float32),
                'bridge_vec_target': bridge_target.astype(np.float32),
                'risk_budget_target': float(risk_target),
                'pred_reward_target': float(pred_reward_target),
                'strat_immediate_target': strat_targets['immediate'].astype(np.float32),
                'strat_short_target': strat_targets['short'].astype(np.float32),
                'strat_medium_target': strat_targets['medium'].astype(np.float32),
                'strat_long_target': strat_targets['long'].astype(np.float32),
                'outcome': float(getattr(self.reward_calculator, 'recent_trading_gains', 0.0)) if self.reward_calculator else 0.0,
                'timestamp': self.t
            }

            # Add to trainer buffer
            self.overlay_trainer.add_experience(experience)

            # DEBUG: Log experience collection every 500 steps
            if self.t % 500 == 0:
                buffer_size = self.overlay_trainer.buffer.size()
                logger.info(f"[OVERLAY_EXPERIENCE] t={self.t} | buffer_size={buffer_size} | strat_immediate={strat_targets['immediate']}")

        except Exception as e:
            logger.debug(f"Experience collection failed: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _compute_bridge_target(self) -> np.ndarray:
        """
        Compute bridge vector target (coordination signal).

        TRULY FORWARD-LOOKING: Use forecast volatility to predict optimal coordination.
        Computes volatility across forecast horizons (short/medium/long) to anticipate
        market regime changes, not react to past volatility.
        """
        try:
            # Get forecasts for next 3 horizons (short/medium/long)
            # FIXED: Use centralized _get_forecast_safe() method
            wind_forecasts = []
            solar_forecasts = []
            hydro_forecasts = []

            for horizon in ["short", "medium", "long"]:
                wind_forecasts.append(self._get_forecast_safe("wind", horizon))
                solar_forecasts.append(self._get_forecast_safe("solar", horizon))
                hydro_forecasts.append(self._get_forecast_safe("hydro", horizon))

            # Compute forecast volatility (spread across horizons = anticipated regime uncertainty)
            wind_vol = float(np.std(wind_forecasts)) if len(wind_forecasts) > 1 else 0.1
            solar_vol = float(np.std(solar_forecasts)) if len(solar_forecasts) > 1 else 0.1
            hydro_vol = float(np.std(hydro_forecasts)) if len(hydro_forecasts) > 1 else 0.1

            # Normalize volatilities to [-1, 1]
            max_vol = max(wind_vol, solar_vol, hydro_vol, 0.01)
            bridge_target = np.array([
                np.clip(wind_vol / max_vol - 0.5, -1.0, 1.0),
                np.clip(solar_vol / max_vol - 0.5, -1.0, 1.0),
                np.clip(hydro_vol / max_vol - 0.5, -1.0, 1.0),
                0.0  # 4th dimension: neutral
            ], dtype=np.float32)

            # DEBUG: Log forecast-based volatility every 500 steps
            if self.t % 500 == 0 and self.t > 0:
                logger.debug(f"[BRIDGE_TARGET] t={self.t} forecast_vol: wind={wind_vol:.4f}, solar={solar_vol:.4f}, hydro={hydro_vol:.4f}")

            return bridge_target[:self.config.overlay_bridge_dim]

        except Exception as e:
            logger.debug(f"Bridge target computation failed: {e}")
            return np.zeros(self.config.overlay_bridge_dim, dtype=np.float32)

    def _compute_risk_budget_target(self) -> float:
        """
        Compute risk budget target (optimal position sizing).

        TRULY FORWARD-LOOKING: Use forecast volatility and forecast confidence
        to predict optimal risk level. High forecast volatility or low confidence
        → reduce risk. Low forecast volatility and high confidence → increase risk.
        """
        try:
            base = 1.0

            # Get price forecasts across horizons
            # FIXED: Use centralized _get_forecast_safe() method
            price_forecasts = []
            for horizon in ["short", "medium", "long"]:
                price_forecasts.append(self._get_forecast_safe("price", horizon))

            # Compute forecast volatility (forward-looking market uncertainty)
            if len(price_forecasts) > 1:
                price_vol_forecast = float(np.std(price_forecasts))

                # Scale risk inversely with forecast volatility
                # High forecast volatility = uncertain future → reduce risk
                # Low forecast volatility = stable future → increase risk
                if price_vol_forecast > 15.0:  # High forecast volatility
                    base = 0.7
                elif price_vol_forecast > 10.0:
                    base = 0.85
                elif price_vol_forecast < 5.0:  # Low forecast volatility
                    base = 1.2
                elif price_vol_forecast < 8.0:
                    base = 1.1
                else:
                    base = 1.0

                # DEBUG: Log forecast volatility every 500 steps
                if self.t % 500 == 0 and self.t > 0:
                    logger.debug(f"[RISK_TARGET] t={self.t} price_vol_forecast={price_vol_forecast:.2f}, base_risk={base:.2f}")

            # ALSO consider forecast confidence (forward-looking risk indicator)
            # Low confidence = uncertain forecasts → reduce risk
            # High confidence = reliable forecasts → can take more risk
            forecast_conf = self._get_forecast_confidence()
            if forecast_conf < 0.5:
                base *= 0.8  # Low confidence: reduce risk
                if self.t % 500 == 0 and self.t > 0:
                    logger.debug(f"[RISK_TARGET] Low confidence ({forecast_conf:.2f}), reducing risk to {base:.2f}")
            elif forecast_conf > 0.8:
                base *= 1.1  # High confidence: can take more risk
                if self.t % 500 == 0 and self.t > 0:
                    logger.debug(f"[RISK_TARGET] High confidence ({forecast_conf:.2f}), increasing risk to {base:.2f}")

            # SAFETY: Also consider current drawdown (backward-looking safety constraint)
            # This is intentionally backward-looking as a safety brake
            if self.reward_calculator:
                current_dd = float(getattr(self.reward_calculator, 'current_drawdown', 0.0))
                if current_dd > 0.05:
                    base *= 0.6  # Severe drawdown: cut risk
                elif current_dd > 0.02:
                    base *= 0.8

            return float(np.clip(base, 0.5, 1.5))

        except Exception as e:
            logger.debug(f"Risk budget target computation failed: {e}")
            return 1.0

    def _compute_pred_reward_target(self) -> float:
        """
        Compute predictive reward target (future reward signal).

        TRULY FORWARD-LOOKING: Use forecast momentum to predict future reward.
        Positive forecast momentum (price rising) → positive reward signal.
        Negative forecast momentum (price falling) → negative reward signal.
        """
        try:
            # CRITICAL FIX: Use RAW prices (DKK) for momentum calculation
            # Forecasts are in raw DKK, so current price must also be in raw DKK
            current_price_raw = float(self._price_raw[self.t]) if self.t < len(self._price_raw) else 250.0
            current_price_raw = max(current_price_raw, 1.0)  # Avoid division by zero

            # Get forecast prices (these are in raw DKK)
            # FIXED: Use centralized _get_forecast_safe() method
            price_short = self._get_forecast_safe("price", "short", default=250.0)
            price_medium = self._get_forecast_safe("price", "medium", default=250.0)

            # Compute forecast momentum (where price is GOING, not where it's BEEN)
            short_momentum = (price_short - current_price_raw) / current_price_raw
            medium_momentum = (price_medium - current_price_raw) / current_price_raw

            # Blend horizons (weight short more than medium for near-term reward prediction)
            forecast_momentum = 0.7 * short_momentum + 0.3 * medium_momentum

            # Normalize to [-1, 1]
            # Scale by 10x to make typical 1-2% price movements map to reasonable reward signals
            pred_target = np.clip(forecast_momentum * 10.0, -1.0, 1.0)

            # DEBUG: Log forecast momentum every 500 steps
            if self.t % 500 == 0 and self.t > 0:
                logger.debug(f"[PRED_REWARD_TARGET] t={self.t} current_price_raw={current_price_raw:.2f}, "
                            f"price_short={price_short:.2f}, price_medium={price_medium:.2f}, "
                            f"forecast_momentum={forecast_momentum:.4f}, pred_target={pred_target:.3f}")

            return float(pred_target)

        except Exception as e:
            logger.debug(f"Pred reward target computation failed: {e}")
            return 0.0

    def _compute_strategy_targets(self) -> Dict[str, np.ndarray]:
        """
        ROBUST: Compute strategy targets with multiple fallback strategies.

        Priority order:
        1. P&L-based targets (best - actual outcomes)
        2. Realized price movement targets (good - actual market movements)
        3. Forecast-based targets (fallback - predicted movements)
        4. Neutral targets (last resort)
        """
        try:
            # PRIORITY 1: P&L-based targets (if enough history)
            if len(self._action_history) >= 10 and len(self._pnl_history) >= 10:
                return self._compute_pnl_based_targets()

            # PRIORITY 2: Realized price movement targets (use actual future prices for training)
            # This is NOT data leakage because we're training the overlay, not the RL policy
            # The overlay learns to predict profitable actions based on current state
            if self.t >= 144:  # Need enough history for all horizons
                return self._compute_realized_movement_targets()

            # PRIORITY 3: Forecast-based targets (if forecasts are available)
            if self._are_forecasts_available():
                return self._compute_forecast_based_targets()

            # PRIORITY 4: Neutral targets (cold start)
            return self._compute_neutral_targets()

        except Exception as e:
            logger.warning(f"Strategy target computation failed: {e}")
            return self._compute_neutral_targets()

    def _compute_pnl_based_targets(self) -> Dict[str, np.ndarray]:
        """
        P&L-BASED: Compute targets from actual trading P&L.

        For each horizon:
        1. Find action taken N steps ago
        2. Measure P&L from that action to now
        3. Normalize P&L to [-1, 1] range
        4. Use as target (positive P&L → positive target, negative P&L → negative target)
        """
        try:
            horizons = {
                'immediate': 1,   # 10 min ago
                'short': 6,       # 1 hour ago
                'medium': 24,     # 4 hours ago
                'long': 144       # 24 hours ago
            }

            targets = {}

            for horizon_name, steps_back in horizons.items():
                # Find action taken N steps ago
                target_timestep = self.t - steps_back

                # Search for action at that timestep
                action_data = None
                for action_record in reversed(self._action_history):
                    if action_record['timestep'] == target_timestep:
                        action_data = action_record
                        break

                if action_data is None:
                    # No action found, use neutral target
                    targets[horizon_name] = np.zeros(4, dtype=np.float32)
                    continue

                # Measure P&L from that action to now
                action_nav = action_data['nav']
                current_nav = self._calculate_fund_nav()
                nav_change = current_nav - action_nav

                # Also get trading P&L over this period
                trading_pnl = 0.0
                for pnl_record in reversed(self._pnl_history):
                    if pnl_record['timestep'] > target_timestep and pnl_record['timestep'] <= self.t:
                        trading_pnl += pnl_record['trading_pnl']

                # Normalize P&L to [-1, 1] range
                # Use adaptive scaling based on typical P&L magnitude
                typical_pnl = 50000.0  # 50k DKK typical for 10-min trading
                if horizon_name == 'short':
                    typical_pnl = 200000.0  # 200k DKK for 1 hour
                elif horizon_name == 'medium':
                    typical_pnl = 500000.0  # 500k DKK for 4 hours
                elif horizon_name == 'long':
                    typical_pnl = 2000000.0  # 2M DKK for 24 hours

                # Combine NAV change and trading P&L (weight trading P&L more)
                combined_pnl = 0.3 * nav_change + 0.7 * trading_pnl
                normalized_pnl = np.clip(combined_pnl / typical_pnl, -1.0, 1.0)

                # Create target: if action led to profit, reinforce it; if loss, discourage it
                # The action that was taken becomes the target (scaled by P&L outcome)
                action_taken = action_data['action']  # [wind, solar, hydro]

                # Scale action by P&L outcome
                # Positive P&L → keep action direction, Negative P&L → reverse action direction
                pnl_scaled_action = action_taken * normalized_pnl

                # Create 4D target [wind, solar, hydro, price]
                # For price, use the average of asset actions
                price_action = float(np.mean(action_taken))

                target_4d = np.array([
                    float(pnl_scaled_action[0]),  # wind
                    float(pnl_scaled_action[1]),  # solar
                    float(pnl_scaled_action[2]),  # hydro
                    price_action * normalized_pnl  # price
                ], dtype=np.float32)

                targets[horizon_name] = np.clip(target_4d, -1.0, 1.0)

            # Log P&L-based learning every 500 steps
            if self.t % 500 == 0 and self.t > 0:
                logger.info(f"[PNL_TARGETS] Step {self.t}: immediate={targets.get('immediate', [0,0,0,0])}, trading_pnl={trading_pnl:.0f}")

            return targets

        except Exception as e:
            logger.debug(f"P&L-based target computation failed: {e}")
            return self._compute_price_based_targets()

    def _compute_realized_movement_targets(self) -> Dict[str, np.ndarray]:
        """
        FORECAST-DRIVEN: Use ACTUAL REALIZED price movements for training targets.

        This creates strong supervision signals based on what actually happened.
        The overlay learns: "given forecasts predicting X, take action Y to profit from actual outcome Z"

        This is valid for overlay training because:
        - Overlay is a supervised learning model (not RL)
        - It learns: "given forecast state X, what action leads to profit?"
        - Training uses hindsight; inference uses only current forecasts
        """
        try:
            horizons = {
                'immediate': 1,
                'short': 6,
                'medium': 24,
                'long': 144
            }

            targets = {}
            # CRITICAL FIX: Use RAW prices (DKK) for return calculation
            current_price_raw = float(self._price_raw[self.t]) if self.t < len(self._price_raw) else 250.0
            current_price_raw = max(current_price_raw, 1.0)

            for horizon_name, steps_ahead in horizons.items():
                future_idx = min(self.t + steps_ahead, len(self._price_raw) - 1)
                future_price_raw = float(self._price_raw[future_idx]) if future_idx < len(self._price_raw) else current_price_raw

                # Realized price movement (what actually happened)
                price_return = (future_price_raw - current_price_raw) / current_price_raw
                price_return = np.clip(price_return, -0.5, 0.5)

                # AGGRESSIVE scaling for strong training signals
                # The model will learn to moderate these during inference
                if horizon_name == 'immediate':
                    scale = 4.0  # Very aggressive for 10-min movements
                elif horizon_name == 'short':
                    scale = 3.0  # Aggressive for 1-hour movements
                elif horizon_name == 'medium':
                    scale = 2.0  # Moderate for 4-hour movements
                else:  # long
                    scale = 1.5  # Conservative for 24-hour movements

                # Create directional target based on realized outcome
                # Positive return → long positions, Negative return → short positions
                target_4d = np.array([
                    np.clip(price_return * scale, -1.0, 1.0),  # wind
                    np.clip(price_return * scale, -1.0, 1.0),  # solar
                    np.clip(price_return * scale, -1.0, 1.0),  # hydro
                    np.clip(price_return * scale, -1.0, 1.0)   # price
                ], dtype=np.float32)

                targets[horizon_name] = target_4d

            # Log realized targets periodically
            if self.t % 1000 == 0 and self.t > 0:
                logger.info(f"[REALIZED_TARGETS] t={self.t} price_return={price_return:.4f} immediate={targets['immediate']}")

            return targets

        except Exception as e:
            logger.debug(f"Realized movement targets failed: {e}")
            return self._compute_neutral_targets()

    def _are_forecasts_available(self) -> bool:
        """Check if real forecast models are loaded (not fallback)"""
        if not hasattr(self, 'forecast_generator') or self.forecast_generator is None:
            return False

        # Check if any price forecast model is actually loaded
        for horizon in ['immediate', 'short', 'medium', 'long']:
            key = f"price_{horizon}"
            if self.forecast_generator._model_available.get(key, False):
                return True

        return False

    def _compute_forecast_based_targets(self) -> Dict[str, np.ndarray]:
        """
        FORECAST-DRIVEN: Compute targets based on FORECAST predictions.
        Only called when real forecast models are loaded.
        """
        return self._compute_price_based_targets()

    def _compute_neutral_targets(self) -> Dict[str, np.ndarray]:
        """Neutral targets for cold start"""
        return {
            'immediate': np.zeros(4, dtype=np.float32),
            'short': np.zeros(4, dtype=np.float32),
            'medium': np.zeros(4, dtype=np.float32),
            'long': np.zeros(4, dtype=np.float32)
        }

    def _compute_price_based_targets(self) -> Dict[str, np.ndarray]:
        """
        FORECAST-DRIVEN: Compute strategy targets based on FORECAST predictions.

        This is the core of forecast-aware trading:
        - Uses multi-horizon price forecasts
        - Creates directional targets based on predicted price movements
        - NO dampening - full forecast signals
        - Aggressive scaling for strong learning
        """
        try:
            # CRITICAL FIX: Use RAW prices (DKK) for forecast comparison
            # Forecasts are in raw DKK, so current price must also be in raw DKK
            current_price_raw = float(self._price_raw[self.t]) if self.t < len(self._price_raw) else 250.0
            current_price_raw = max(current_price_raw, 1.0)

            # Get FORECASTS for all horizons (these are in raw DKK)
            # FIXED: Use centralized _get_forecast_safe() method
            price_immediate = self._get_forecast_safe("price", "immediate", default=current_price_raw)
            price_short = self._get_forecast_safe("price", "short", default=current_price_raw)
            price_medium = self._get_forecast_safe("price", "medium", default=current_price_raw)
            price_long = self._get_forecast_safe("price", "long", default=current_price_raw)

            # Compute FORECAST-BASED price movements
            immediate_pred = (price_immediate - current_price_raw) / current_price_raw
            short_pred = (price_short - current_price_raw) / current_price_raw
            medium_pred = (price_medium - current_price_raw) / current_price_raw
            long_pred = (price_long - current_price_raw) / current_price_raw

            # Clip to reasonable bounds
            immediate_pred = np.clip(immediate_pred, -0.5, 0.5)
            short_pred = np.clip(short_pred, -0.5, 0.5)
            medium_pred = np.clip(medium_pred, -0.5, 0.5)
            long_pred = np.clip(long_pred, -0.5, 0.5)

            # Log forecast deltas periodically
            if self.t % 1000 == 0 and self.t > 0:
                logger.info(f"[FORECAST_TARGETS] t={self.t} current={current_price:.2f} | "
                            f"deltas: imm={immediate_pred:.4f}, short={short_pred:.4f}, "
                            f"med={medium_pred:.4f}, long={long_pred:.4f}")

            # AGGRESSIVE SCALING: Strong signals for effective learning
            # NO DAMPENING - use full forecast signals

            # IMMEDIATE: Very aggressive (5x) - capitalize on immediate forecast movements
            immediate_strat = np.array([
                np.clip(immediate_pred * 5.0, -1.0, 1.0),
                np.clip(immediate_pred * 5.0, -1.0, 1.0),
                np.clip(immediate_pred * 5.0, -1.0, 1.0),
                np.clip(immediate_pred * 5.0, -1.0, 1.0)
            ], dtype=np.float32)

            # SHORT: Aggressive (4x) - strong tactical positions
            short_strat = np.array([
                np.clip(short_pred * 4.0, -1.0, 1.0),
                np.clip(short_pred * 4.0, -1.0, 1.0),
                np.clip(short_pred * 4.0, -1.0, 1.0),
                np.clip(short_pred * 4.0, -1.0, 1.0)
            ], dtype=np.float32)

            # MEDIUM: Moderate (3x) - strategic positioning
            medium_strat = np.array([
                np.clip(medium_pred * 3.0, -1.0, 1.0),
                np.clip(medium_pred * 3.0, -1.0, 1.0),
                np.clip(medium_pred * 3.0, -1.0, 1.0),
                np.clip(medium_pred * 3.0, -1.0, 1.0)
            ], dtype=np.float32)

            # LONG: Conservative (2x) - long-term positioning
            long_strat = np.array([
                np.clip(long_pred * 2.0, -1.0, 1.0),
                np.clip(long_pred * 2.0, -1.0, 1.0),
                np.clip(long_pred * 2.0, -1.0, 1.0),
                np.clip(long_pred * 2.0, -1.0, 1.0)
            ], dtype=np.float32)

            return {
                'immediate': immediate_strat,
                'short': short_strat,
                'medium': medium_strat,
                'long': long_strat
            }

        except Exception as e:
            logger.debug(f"Forecast-based strategy targets failed: {e}")
            return self._compute_neutral_targets()

    def _get_forecast_confidence(self) -> float:
        """
        Get forecast confidence score [0,1] based on recent forecast accuracy (MAPE).

        UPDATED: Uses MAPE-based confidence with configurable floor (default 0.6).
        """
        try:
            # Get confidence floor from config (default 0.6)
            confidence_floor = getattr(self.config, 'confidence_floor', 0.6)

            # If no forecast generator, return floor confidence
            if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
                return confidence_floor

            # PRIMARY METHOD: Use MAPE-based confidence from forecast error tracking
            if hasattr(self, '_forecast_errors'):
                # Calculate average confidence across all assets
                confidences = []
                for asset in ['wind', 'solar', 'hydro']:
                    asset_errors = self._forecast_errors.get(asset, [])
                    if len(asset_errors) >= 10:
                        # Lower MAPE = higher confidence
                        recent_mape = np.mean(list(asset_errors)[-10:])
                        # Map MAPE to confidence: 0% error = 1.0, 50% error = 0.5
                        confidence = 1.0 - np.clip(recent_mape, 0.0, 0.5)
                        confidences.append(confidence)

                if confidences:
                    # Use average confidence across assets
                    avg_confidence = np.mean(confidences)
                    # Apply floor
                    return float(np.clip(avg_confidence, confidence_floor, 1.0))

            # FALLBACK 1: Use generator's confidence calculation
            if hasattr(self.forecast_generator, "calculate_forecast_confidence"):
                agent_names = getattr(self.forecast_generator, 'agent_targets', {}).keys()
                if agent_names:
                    primary_agent = next(iter(agent_names))
                    confidence = self.forecast_generator.calculate_forecast_confidence(primary_agent, self.t)
                    return float(np.clip(confidence, confidence_floor, 1.0))

            # FALLBACK 2: Check for confidence in forecast output
            if hasattr(self.forecast_generator, "predict_all_horizons"):
                forecasts = self.forecast_generator.predict_all_horizons(timestep=self.t)
                if isinstance(forecasts, dict):
                    for key in ['confidence', 'forecast_confidence', 'price_confidence']:
                        if key in forecasts and np.isfinite(forecasts[key]):
                            return float(np.clip(forecasts[key], confidence_floor, 1.0))

            # Default: return floor confidence
            return confidence_floor

        except Exception:
            return getattr(self.config, 'confidence_floor', 0.6)

    def _get_volatility_factor(self) -> float:
        """Get current market volatility factor [0,1]"""
        try:
            # Use market volatility from market conditions
            return float(np.clip(getattr(self, 'market_volatility', 0.5), 0.0, 1.0))
        except Exception:
            return 0.5

    def _calculate_position_size_multiplier(self, volatility_factor: float) -> float:
        """Calculate position size multiplier based on volatility"""
        try:
            # Reduce positions during high volatility, increase during stable periods
            # volatility_factor: 0 = stable, 1 = very volatile

            if volatility_factor < 0.3:  # Low volatility (stable)
                return 1.2  # Increase positions by 20%
            elif volatility_factor < 0.6:  # Medium volatility
                return 1.0  # Normal positions
            elif volatility_factor < 0.8:  # High volatility
                return 0.7  # Reduce positions by 30%
            else:  # Very high volatility
                return 0.5  # Reduce positions by 50%

        except Exception:
            return 1.0

    # ----------------------
    # FIXED: Battery dispatch
    # ----------------------
    def _battery_dispatch_policy(self, i: int) -> Tuple[str, float]:
        """
        FIXED: Decide ('charge'/'discharge'/'idle', intensity 0..1) from price now vs. aligned forecast.
        Uses consistent DKK pricing for both current price and forecast comparisons.
        """
        try:
            # Current price (use raw DKK price for consistent units)
            p_now = float(np.clip(self._price_raw[i], self.config.minimum_price_filter, self.config.maximum_price_cap))

            # Get raw forecast and normalize it to match current price scale
            p_fut_raw = self._get_aligned_price_forecast(i, default=None)
            if p_fut_raw is None or not np.isfinite(p_fut_raw):
                return ("idle", 0.0)

            # Use raw DKK prices for both current and forecast (no normalization needed)
            p_fut = float(np.clip(p_fut_raw, self.config.minimum_price_filter, self.config.maximum_price_cap))

            spread = p_fut - p_now

            # PHASE 5.10 FIX: Use config parameters for battery dispatch thresholds
            hurdle_min = getattr(self.config, 'battery_hurdle_min_dkk', 5.0)
            price_sens = getattr(self.config, 'battery_price_sensitivity', 0.02)
            rt_loss_weight = getattr(self.config, 'battery_rt_loss_weight', 0.3)

            needed = max(hurdle_min, price_sens * abs(p_now))  # Minimum threshold in DKK/MWh
            rt_loss = (1.0/(max(self.batt_eta_charge*self.batt_eta_discharge, 1e-6)) - 1.0) * 10.0  # Round-trip loss in DKK/MWh
            hurdle = needed + rt_loss_weight * rt_loss

            if spread > hurdle:
                inten = float(np.clip(spread / (abs(p_now) + 0.1), 0.0, 1.0))
                return ("charge", inten)
            elif spread < -hurdle:
                inten = float(np.clip((-spread) / (abs(p_now) + 0.1), 0.0, 1.0))
                return ("discharge", inten)
            else:
                return ("idle", 0.0)
        except Exception:
            return ("idle", 0.0)

    def _execute_battery_ops(self, bat_action: np.ndarray, i: int) -> float:
        """
        REFACTORED: Execute battery operations using TradingEngine.
        Returns battery cash delta for proper reward accounting.
        """
        try:
            from trading_engine import TradingEngine
            
            # Get forecast price for dispatch policy
            forecast_price = self._get_aligned_price_forecast(i, default=None)

            # --- Battery debug snapshot (so logs reflect real behavior) ---
            try:
                price_now = float(np.clip(self._price_raw[i], self.config.minimum_price_filter, self.config.maximum_price_cap))
                u_raw = float(np.clip(np.array(bat_action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
                vol = float(getattr(self, '_last_price_volatility_forecast', 0.0))

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

                # Mirror TradingEngine.execute_battery_operations decision blending
                agent_intensity = abs(u_raw)
                if u_raw < -0.2:
                    agent_decision = 'charge'
                elif u_raw > 0.2:
                    agent_decision = 'discharge'
                else:
                    agent_decision = 'idle'

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
            
            # Create dispatch policy function
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
                config=self.config
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
            logger.error(f"battery ops: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # FIXED: Finance & rewards with proper separation
    # ------------------------------------------------------------------
    def _update_finance(self, i: int, trade_amount: float, battery_cash_delta: float) -> Dict[str, float]:
        """
        REFACTORED: Update all financial components with proper separation.
        Uses FinancialEngine for core calculations while keeping environment-specific logic here.
        
        Components:
        1. Physical asset generation revenue (cash flow)
        2. Financial instrument mark-to-market (unrealized)
        3. Battery operations (cash flow)
        4. Transaction costs (cash flow)
        """
        try:
            from financial_engine import FinancialEngine
            
            # CRITICAL FIX: Use raw prices for MTM calculations, not normalized prices
            current_price = float(np.clip(self._price_raw[i], -1000.0, 1e9))

            # REFACTORED: Use FinancialEngine for price return calculations
            enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False)
            horizon_correlations = getattr(self, '_horizon_correlations', None)
            
            price_returns = FinancialEngine.calculate_price_returns(
                timestep=i,
                current_price=current_price,
                price_history=self._price_raw,
                enable_forecast_util=enable_forecast_util,
                horizon_correlations=horizon_correlations
            )
            
            price_return = price_returns['price_return']
            price_return_short = price_returns['price_return_short']
            price_return_medium = price_returns['price_return_medium']
            price_return_long = price_returns['price_return_long']
            price_return_forecast = price_returns['price_return_forecast']
            
            # Store correlation debug info
            self._correlation_debug = price_returns['correlation_debug']

            # Persist latest price return snapshot for forecast engine usage
            self._latest_price_returns = {
                'short': price_return_short,
                'medium': price_return_medium,
                'long': price_return_long,
                'forecast': price_return_forecast,
                'one_step': price_return,
            }
            
            # FIX: Store 1-step price return for Tier 1 PnL reward calculation
            if i < len(self._price_return_1step):
                self._price_return_1step[i] = price_return

            # 1) Generation revenue from physical assets (CASH FLOW)
            generation_revenue = self._calculate_generation_revenue(i, current_price)

            # 2) Mark-to-market on financial instruments (UNREALIZED)
            # CORRECTNESS: MTM must be computed from EXPOSURE and accumulated into a separate MTM bucket.
            if not hasattr(self, 'financial_mtm_positions') or not isinstance(getattr(self, 'financial_mtm_positions', None), dict):
                self.financial_mtm_positions = {
                    'wind_instrument_value': 0.0,
                    'solar_instrument_value': 0.0,
                    'hydro_instrument_value': 0.0,
                }
            mtm_pnl, per_asset_mtm = FinancialEngine.calculate_mtm_pnl_from_exposure(
                financial_exposures=self.financial_positions,
                price_return=price_return,
                config=self.config
            )
            # Accumulate true MTM value (PnL) over time; do NOT mutate exposure here.
            for k in ('wind_instrument_value', 'solar_instrument_value', 'hydro_instrument_value'):
                self.financial_mtm_positions[k] = float(self.financial_mtm_positions.get(k, 0.0)) + float(per_asset_mtm.get(k, 0.0))

            # 3) Transaction costs (CASH FLOW) - FIXED: Already deducted in _execute_investor_trades
            txn_costs = 0.0  # Costs already handled in trading execution

            # 4) Battery operational costs (CASH FLOW) - from config
            battery_opex = self.config.battery_opex_rate * self.physical_assets['battery_capacity_mwh']

            # 5) Net cash flow this step (REMOVED admin costs for better agent learning)
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
            if i <= 1:
                tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
                accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
                financial_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
                logger.info(f"[{tier_name}_UPDATE_FINANCE] Calculating NAV in _update_finance at i={i}:")
                logger.info(f"  i parameter = {i}")
                logger.info(f"  self.t = {getattr(self, 't', 0)}")
                logger.info(f"  budget = {self.budget:,.0f} DKK")
                logger.info(f"  physical_assets = {self.physical_assets}")
                logger.info(f"  accumulated_operational_revenue = {accumulated_operational_revenue:,.0f} DKK")
                logger.info(f"  financial_positions MTM = {financial_mtm:,.0f} DKK")
            fund_nav = self._calculate_fund_nav(current_timestep=i)
            if i <= 1:
                tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
                # Calculate components to see what's different
                accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
                financial_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
                # Calculate physical assets from NAV
                physical_from_nav = fund_nav - self.budget - accumulated_operational_revenue - financial_mtm
                logger.info(f"[{tier_name}_UPDATE_FINANCE] NAV calculated at i={i}: {fund_nav:,.0f} DKK (${fund_nav * 0.145 / 1_000_000:.2f}M)")
                logger.info(f"  NAV breakdown: budget={self.budget:,.0f}, physical={physical_from_nav:,.0f}, operational={accumulated_operational_revenue:,.0f}, mtm={financial_mtm:,.0f}")
                logger.info(f"  current_timestep passed to _calculate_fund_nav = {i}")

            # 9) Track performance
            self.performance_history['revenue_history'].append(net_cash_flow)
            self.performance_history['generation_revenue_history'].append(generation_revenue)
            self.performance_history['nav_history'].append(fund_nav)

            # 10) Store values for logging/rewards
            self.last_revenue = net_cash_flow
            self.last_generation_revenue = generation_revenue
            self.last_mtm_pnl = mtm_pnl

            # REMOVED: Multi-step rolling returns and MTM delta buffer updates (reverted to original)

            # CRITICAL: Update reward calculator with trading gains for DL training
            if hasattr(self, 'reward_calculator') and self.reward_calculator is not None:
                self.reward_calculator.update_trading_gains(mtm_pnl)

            # 11) Update cumulative tracking
            self.cumulative_generation_revenue += generation_revenue
            self.cumulative_battery_revenue += battery_cash_delta
            self.cumulative_mtm_pnl += mtm_pnl  # ENHANCED: Track cumulative trading performance

            # ===== KELLY SIZER UPDATE (Model B): per-step MTM PnL samples =====
            # Under Model B, exposures are not "position values". We compute per-step MTM PnL from exposure
            # (`per_asset_mtm`) and feed those PnL samples directly into Kelly sizing.
            if hasattr(self, 'kelly_sizer'):
                try:
                    # Avoid feeding long runs of exact zeros (Kelly treats zeros as losses and will spam warnings).
                    eps = 1e-9
                    w = float(per_asset_mtm.get('wind_instrument_value', 0.0))
                    s = float(per_asset_mtm.get('solar_instrument_value', 0.0))
                    h = float(per_asset_mtm.get('hydro_instrument_value', 0.0))
                    if abs(w) > eps:
                        self.kelly_sizer.update('wind', w)
                    if abs(s) > eps:
                        self.kelly_sizer.update('solar', s)
                    if abs(h) > eps:
                        self.kelly_sizer.update('hydro', h)
                except Exception as e:
                    logger.warning(f"[KELLY] Update failed at t={i}: {e}")

                # Log Kelly metrics periodically
                if i % 1000 == 0 and i > 0:
                    try:
                        kelly_metrics = self.kelly_sizer.get_all_metrics()
                        wind_metrics = kelly_metrics.get('wind', {})
                        logger.info(f"[KELLY] Step {i}: "
                                   f"wind_kelly={wind_metrics.get('kelly_fraction', 0.5):.3f}, "
                                   f"win_rate={wind_metrics.get('win_rate', 0.5):.2f}, "
                                   f"num_trades={wind_metrics.get('num_trades', 0)}")
                    except Exception:
                        pass

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
            enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False)
            forecast_gate_passed = False
            forecast_used_flag = False
            forecast_usage_reason = 'no_forecast'

            # DEBUG: Log which tier we're in
            if i == 500:
                logger.debug(f"[DEBUG_TIER] t={i} enable_forecast_util={enable_forecast_util}")

            if enable_forecast_util and self.forecast_engine is not None:
                price_returns_payload = getattr(self, '_latest_price_returns', None)
                if not price_returns_payload:
                    price_returns_payload = {
                        'short': price_return_short,
                        'medium': price_return_medium,
                        'long': price_return_long,
                        'forecast': price_return_forecast,
                        'one_step': price_return,
                    }

                try:
                    forecast_result = self.forecast_engine.compute(
                        timestep=i,
                        mtm_pnl=mtm_pnl,
                        price_returns=price_returns_payload,
                    )
                except Exception as exc:
                    logger.warning(f"[FORECAST_ENGINE] Failed at step {i}: {exc}")
                    forecast_result = None

                if forecast_result is not None:
                    forecast_signal_score = forecast_result.forecast_signal_score
                    forecast_gate_passed = forecast_result.forecast_gate_passed
                    forecast_used_flag = forecast_result.forecast_used_flag
                    forecast_usage_reason = forecast_result.forecast_usage_reason
                    self._debug_forecast_reward = forecast_result.debug_payload or {}
                else:
                    forecast_signal_score = 0.0
                    forecast_gate_passed = False
                    forecast_used_flag = False
                    forecast_usage_reason = 'engine_failure'

            elif enable_forecast_util:
                # OPTION 2: Forward-looking reward alignment
                # Use z-scores from historical timesteps that match forecast horizons
                # - Short horizon (6 steps): Use z-scores from step i-6
                # - Medium horizon (24 steps): Use z-scores from step i-24
                # - Long horizon (144 steps): Use z-scores from step i-144
                # This aligns rewards with what the forecast actually predicted
                
                # Get horizon steps from config
                horizon_short = self.config.forecast_horizons.get('short', 6)
                horizon_medium = self.config.forecast_horizons.get('medium', 24)
                horizon_long = self.config.forecast_horizons.get('long', 144)
                
                # Look up historical z-scores matching forecast horizons
                # Forecast generated at step i-horizon predicts price at step i
                # so we align reward with z-scores stored at timestep i-horizon
                t_short = i - horizon_short if i >= horizon_short else None  # = i - 6 for horizon=6
                t_medium = i - horizon_medium if i >= horizon_medium else None  # = i - 24 for horizon=24
                t_long = i - horizon_long if i >= horizon_long else None  # = i - 144 for horizon=144
                
                # Retrieve z-scores from history (fallback to current if not available)
                if t_short is not None and t_short in self._z_score_history:
                    z_short = float(self._z_score_history[t_short]['z_short'])
                else:
                    z_short = float(getattr(self, 'z_short_price', 0.0))  # Fallback to current
                
                if t_medium is not None and t_medium in self._z_score_history:
                    z_medium = float(self._z_score_history[t_medium]['z_medium'])
                else:
                    z_medium = float(getattr(self, 'z_medium_price', 0.0))  # Fallback to current
                
                if t_long is not None and t_long in self._z_score_history:
                    z_long = float(self._z_score_history[t_long]['z_long'])
                else:
                    z_long = float(getattr(self, 'z_long_price', 0.0))  # Fallback to current
                
                # Log horizon matching for debugging
                if i % 1000 == 0 or i == 0:
                    logger.info(f"[HORIZON_MATCHING] t={i} | Using z-scores from: "
                               f"short={t_short if t_short is not None else 'current'}, "
                               f"medium={t_medium if t_medium is not None else 'current'}, "
                               f"long={t_long if t_long is not None else 'current'}")

                # CRITICAL FIX: Correlation-based horizon weighting
                # Higher correlation = higher weight (more predictive forecasts get more influence)
                # Remove horizons with negative correlation (not predictive)
                
                # Get correlations for each horizon
                corr_short = self._horizon_correlations.get('short', 0.0)
                corr_medium = self._horizon_correlations.get('medium', 0.0)
                corr_long = self._horizon_correlations.get('long', 0.0)
                
                # CRITICAL: Remove short horizon if correlation is negative
                # Negative correlation means forecasts are counter-predictive (worse than random)
                use_short = corr_short > 0.0
                use_medium = corr_medium > 0.0
                use_long = corr_long > 0.0
                
                # If no horizons have positive correlation, fall back to fixed weights
                if not (use_short or use_medium or use_long):
                    # Fallback: use fixed weights if no correlation data available
                    weight_short = 0.7 if use_short else 0.0
                    weight_medium = 0.2 if use_medium else 0.0
                    weight_long = 0.1 if use_long else 0.0
                else:
                    # Convert correlation to weights (higher correlation = higher weight)
                    # Use max(0, correlation) to ensure only positive correlations contribute
                    epsilon = 1e-6
                    weight_short_raw = max(0.0, corr_short) if use_short else 0.0
                    weight_medium_raw = max(0.0, corr_medium) if use_medium else 0.0
                    weight_long_raw = max(0.0, corr_long) if use_long else 0.0
                    
                    # Normalize weights to sum to 1.0
                    total_weight = weight_short_raw + weight_medium_raw + weight_long_raw
                    if total_weight > epsilon:
                        weight_short = weight_short_raw / total_weight
                        weight_medium = weight_medium_raw / total_weight
                        weight_long = weight_long_raw / total_weight
                    else:
                        # Fallback: equal weights for available horizons
                        num_horizons = sum([use_short, use_medium, use_long])
                        if num_horizons > 0:
                            weight_short = (1.0 / num_horizons) if use_short else 0.0
                            weight_medium = (1.0 / num_horizons) if use_medium else 0.0
                            weight_long = (1.0 / num_horizons) if use_long else 0.0
                        else:
                            weight_short = 0.0
                            weight_medium = 0.0
                            weight_long = 0.0
                
                # Use correlation-based weights for z-score combination
                # If short horizon is excluded, it gets 0 weight
                z_combined = float(weight_short * z_short + weight_medium * z_medium + weight_long * z_long)
                
                # Store z_combined as attribute for adaptive forecast weight calculation
                self.z_combined = z_combined
                
                # Log correlation-based weights periodically for monitoring
                if i % 1000 == 0 or i == 0:
                    logger.info(f"[CORRELATION_BASED_WEIGHTS] t={i} | "
                               f"Correlations: short={corr_short:.4f} (use={use_short}), medium={corr_medium:.4f} (use={use_medium}), long={corr_long:.4f} (use={use_long}) | "
                               f"Weights: short={weight_short:.3f}, medium={weight_medium:.3f}, long={weight_long:.3f} | "
                               f"z-values: short={z_short:.4f}, medium={z_medium:.4f}, long={z_long:.4f} | "
                               f"z_combined={z_combined:.4f}")

                # CRITICAL FIX V3: HYBRID REWARD (Alignment + PnL)
                # Problem with V2: PnL-only reward is backward-looking and sparse
                # - Only rewards AFTER price moves (delayed feedback)
                # - Only non-zero when positions exist AND price moves (sparse signal)
                # - Weak learning signal
                #
                # NEW HYBRID APPROACH: Combine forward-looking + backward-looking rewards
                #
                # Component 1: ALIGNMENT REWARD (Forward-looking)
                # - Rewards taking positions aligned with forecast direction
                # - Dense signal (immediate feedback every step)
                # - Encourages position-taking when forecast is confident
                # - Formula: z_combined * position_exposure * 5.0
                #
                # Component 2: PnL REWARD (Backward-looking)
                # - Rewards actual profitability from positions
                # - Strong signal when trades work
                # - Aligns with actual objective
                # - Formula: actual_pnl * |z_combined| * 50.0
                #
                # TOTAL = Alignment + PnL
                # This gives 6x stronger learning signal than PnL-only!
                #
                # Example scenarios:
                # 1. Strong forecast + NO position → 0.0 (encourages taking position)
                # 2. Strong forecast + SMALL position → 1.44 (6x stronger than V2!)
                # 3. Strong forecast + LARGE position + profit → 2.88 (6x stronger!)

                # Get current position exposure (normalized by initial budget)
                max_pos = self.config.max_position_size * self.init_budget * self.config.capital_allocation_fraction if self.init_budget > 0 else 1.0
                wind_pos_value = self.financial_positions.get('wind_instrument_value', 0.0)
                solar_pos_value = self.financial_positions.get('solar_instrument_value', 0.0)
                hydro_pos_value = self.financial_positions.get('hydro_instrument_value', 0.0)

                # Normalize by max position size (KEEP SIGN for alignment check!)
                wind_pos_norm = float(wind_pos_value / max(max_pos, 1.0))
                solar_pos_norm = float(solar_pos_value / max(max_pos, 1.0))
                hydro_pos_norm = float(hydro_pos_value / max(max_pos, 1.0))

                # SIGNED total position (for alignment reward - checks direction match)
                position_signed = float(wind_pos_norm + solar_pos_norm + hydro_pos_norm)

                # UNSIGNED total position exposure (for PnL reward - checks size)
                # CRITICAL FIX: Cap position exposure at 0.3 to prevent catastrophic losses from wrong forecasts
                # Issue: Mean exposure was 0.4705, causing -$43.5M in large losses
                # Solution: Cap at 0.3 to limit risk while still allowing meaningful positions
                raw_position_exposure = float(abs(wind_pos_norm) + abs(solar_pos_norm) + abs(hydro_pos_norm))
                position_exposure = float(np.clip(raw_position_exposure, 0.0, 0.3))  # Cap at 0.3 (30% of max position)

                # Forecast confidence
                forecast_confidence = float(abs(z_combined))

                # COMPONENT 1: ALIGNMENT REWARD (Forward-looking)
                # PROFITABILITY FIX: Scale alignment reward by unrealized P&L
                # Problem: Agents were getting positive rewards for holding losing positions
                # Solution: Reduce reward when position is losing money

                # Calculate unrealized P&L for current positions
                current_price = float(np.clip(self._price_raw[i], -1000.0, 1e9))
                # Model B: Use the current MTM bucket as the unrealized P&L proxy.
                unrealized_pnl_total = float(sum(getattr(self, 'financial_mtm_positions', {}).values()))

                # Normalize unrealized P&L by max position size
                max_pos = self.config.max_position_size * self.init_budget * self.config.capital_allocation_fraction if self.init_budget > 0 else 1.0
                unrealized_pnl_norm = float(np.clip(unrealized_pnl_total / max(max_pos, 1.0), -3.0, 3.0))

                # =====================================================================
                # NOVEL: FORECAST-BASED TRADING FOR NAV GROWTH
                # =====================================================================
                # Goal: Use forecasts to increase NAV through profitable trading
                # Approach: Multi-horizon consensus + MAPE filtering + PnL-based rewards

                # DEBUG: Check why risk management is not active
                has_risk_mgr = self.forecast_risk_manager is not None
                has_mode_flag = getattr(self.config, 'forecast_risk_management_mode', False)
                enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False)
                if i == 0:  # Log only on first step
                    logger.info(f"[DEBUG_RISK_MGR] Step {i}:")
                    logger.info(f"  has_risk_mgr={has_risk_mgr}")
                    logger.info(f"  has_mode_flag={has_mode_flag}")
                    logger.info(f"  enable_forecast_util={enable_forecast_util}")
                    logger.info(f"  config object ID: {id(self.config)}")
                    logger.info(f"  forecast_risk_manager object ID: {id(self.forecast_risk_manager) if has_risk_mgr else 'None'}")
                    if not has_risk_mgr:
                        logger.warning(f"[DEBUG_RISK_MGR] forecast_risk_manager is None!")
                    if not has_mode_flag:
                        logger.warning(f"[DEBUG_RISK_MGR] forecast_risk_management_mode is False!")
                    if not enable_forecast_util:
                        logger.warning(f"[DEBUG_RISK_MGR] enable_forecast_utilisation is False!")

                use_risk_management = (self.forecast_risk_manager is not None and
                                      getattr(self.config, 'forecast_risk_management_mode', False))

                # DEBUG: Log risk management status
                if i == 500:
                    logger.debug(f"[DEBUG_RISK_MGR] use_risk_management={use_risk_management}")
                    logger.debug(f"[DEBUG_RISK_MGR] forecast_risk_manager={self.forecast_risk_manager is not None}")
                    logger.debug(f"[DEBUG_RISK_MGR] forecast_risk_management_mode={getattr(self.config, 'forecast_risk_management_mode', False)}")

                # =====================================================================
                # NOVEL: FORECAST RETURNS INTEGRATION (Bias-Immune)
                # =====================================================================
                # Instead of using absolute forecast levels (biased), use RETURNS
                # Returns are stationary and bias cancels out in the division
                # This focuses on what matters: price movements, not absolute levels

                current_price = float(self._price_raw[i]) if i < len(self._price_raw) else 250.0

                # Get forecast prices (raw DKK values)
                forecast_price_short = 0.0
                forecast_price_medium = 0.0
                forecast_price_long = 0.0
                if self.forecast_generator:
                    try:
                        fcast = self.forecast_generator.predict_all_horizons(timestep=i)
                        if fcast:
                            forecast_price_short = fcast.get('price_forecast_short', current_price)
                            forecast_price_medium = fcast.get('price_forecast_medium', current_price)
                            forecast_price_long = fcast.get('price_forecast_long', current_price)

                            # DEBUG: Print forecast values every 500 steps
                            if i % 500 == 0:
                                logger.debug(f"[FORECAST_PRICES] t={i} current={current_price:.2f} short={forecast_price_short:.2f} med={forecast_price_medium:.2f} long={forecast_price_long:.2f}")
                    except Exception as e:
                        if i % 500 == 0:
                            logger.warning(f"[FORECAST_PRICES] ERROR at t={i}: {e}")

                # NOVEL: Compute FORECAST RETURNS instead of absolute deltas
                # This is immune to bias because bias cancels out in the division
                forecast_return_short = (forecast_price_short - current_price) / max(current_price, 1.0) if forecast_price_short > 0 else 0.0
                forecast_return_medium = (forecast_price_medium - current_price) / max(current_price, 1.0) if forecast_price_medium > 0 else 0.0
                forecast_return_long = (forecast_price_long - current_price) / max(current_price, 1.0) if forecast_price_long > 0 else 0.0

                # Store for CSV logging (backward compatibility)
                forecast_error_short = forecast_return_short
                forecast_error_medium = forecast_return_medium
                forecast_error_long = forecast_return_long

                # ================================================================
                # TIER 2: FORECAST INTEGRATION (ALWAYS ENABLED when enable_forecast_util=True)
                # ================================================================
                # CRITICAL FIX: Remove `if use_risk_management:` check
                # This was causing Tier 2 to fall through to Tier 1 code!

                # Get forecast data
                forecast_deltas = getattr(self, '_forecast_deltas_raw', None)
                mape_thresholds_capacity = getattr(self, '_mape_thresholds', None)

                def _to_price_relative_mapes(mape_dict: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
                    """Convert capacity-based MAPE values into price-relative percentages."""
                    if not mape_dict:
                        return None
                    price_capacity = getattr(self.config, 'forecast_price_capacity', 6982.0)
                    price_floor = getattr(self.config, 'minimum_price_floor', 50.0)
                    denom = max(abs(current_price), price_floor, 1e-6)
                    scale = price_capacity / denom
                    converted: Dict[str, float] = {}
                    for horizon, value in mape_dict.items():
                        try:
                            val = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(val):
                            continue
                        converted[horizon] = float(val * scale)
                    return converted if converted else None

                mape_thresholds = _to_price_relative_mapes(mape_thresholds_capacity)
                forecast_trust = float(getattr(self, '_forecast_trust', 0.5))

                # Current positions for risk manager (Model B): use normalized exposures
                max_pos_dkk = self.config.max_position_size * self.init_budget * self.config.capital_allocation_fraction if self.init_budget > 0 else 1.0
                max_pos_dkk = max(float(max_pos_dkk), 1.0)
                current_positions = {
                    'wind': float(self.financial_positions.get('wind_instrument_value', 0.0) / max_pos_dkk),
                    'solar': float(self.financial_positions.get('solar_instrument_value', 0.0) / max_pos_dkk),
                    'hydro': float(self.financial_positions.get('hydro_instrument_value', 0.0) / max_pos_dkk),
                }

                # Compute investor strategy (NOVEL: multi-horizon consensus + MAPE thresholds)
                quality_mult = 1.0

                if forecast_deltas is not None and mape_thresholds is not None:
                    # Build measured MAPE dictionary (prefer recent observed errors)
                    def _recent_mape(h, n=10):
                        try:
                            seq = list(getattr(self, '_horizon_mape', {}).get(h, []))
                            if len(seq) > 0:
                                return float(np.mean(seq[-n:]))
                        except Exception:
                            pass
                        return float('nan')

                    mape_measured_capacity = {
                        'short': _recent_mape('short'),
                        'medium': _recent_mape('medium'),
                        'long': _recent_mape('long')
                    }

                    # Where measured values are missing, fall back to configured thresholds
                    for h in ['short', 'medium', 'long']:
                        if np.isnan(mape_measured_capacity[h]):
                            fallback = mape_thresholds_capacity.get(h, 0.02) if mape_thresholds_capacity else 0.02
                            mape_measured_capacity[h] = float(fallback)

                    mape_measured = _to_price_relative_mapes(mape_measured_capacity)
                    if mape_measured is None:
                        mape_measured = mape_thresholds.copy() if mape_thresholds is not None else None

                    # CRITICAL FIX: Only call ForecastRiskManager if it exists
                    # Tier 2/3 without risk management (observations only mode) have forecast_risk_manager = None
                    if self.forecast_risk_manager is not None:
                        # Pass measured MAPE into risk manager (it expects a dict of per-horizon MAPE)
                        investor_strategy = self.forecast_risk_manager.compute_investor_strategy(
                        forecast_deltas=forecast_deltas,
                        mape_thresholds=mape_measured,
                        current_positions=current_positions,
                        forecast_trust=forecast_trust
                        )

                        # Store for later use in agent rewards
                        self._current_investor_strategy = investor_strategy

                        # Track when the MAPE/consensus gates permit using forecasts
                        if investor_strategy is not None:
                            forecast_gate_passed = forecast_gate_passed or bool(
                                investor_strategy.get('trade_signal', False)
                            )
                    else:
                        # Tier 2/3 observations-only mode: No risk manager, no strategy scaling
                        investor_strategy = None
                        self._current_investor_strategy = None
                else:
                    investor_strategy = None
                    self._current_investor_strategy = None

                # Compute risk adjustments (position scaling, exit signals, etc.)
                # CRITICAL FIX: Only call ForecastRiskManager if it exists
                if self.forecast_risk_manager is not None:
                    risk_adj = self.forecast_risk_manager.compute_risk_adjustments(
                        z_short=z_short,
                        z_medium=z_medium,
                        z_long=z_long,
                        forecast_trust=forecast_trust,
                        position_pnl=unrealized_pnl_total,
                        timestep=i,
                        forecast_deltas=forecast_deltas,
                        mape_thresholds=mape_thresholds
                    )

                    if risk_adj is not None:
                        forecast_gate_passed = forecast_gate_passed or bool(risk_adj.get('trade_signal', False))
                else:
                    # Tier 2/3 observations-only mode: No risk manager, no risk adjustments
                    risk_adj = None

                # ================================================================
                # CRITICAL FIX: FORECAST-FIRST REWARD LOGIC
                # Make forecasts the PRIMARY signal, PnL is SECONDARY validation
                # This ensures Tier 2 learns to USE forecasts, not ignore them!
                # ================================================================

                # Get forecast data
                max_pos = self.config.max_position_size * self.init_budget * self.config.capital_allocation_fraction if self.init_budget > 0 else 1.0
                total_position = sum(current_positions.values())
                position_normalized = total_position / max(max_pos, 1.0)
                position_exposure = abs(position_normalized)

                # Component 1: FORECAST ALIGNMENT REWARD (PRIMARY - 60%)
                # Reward agent for aligning positions with forecast signals
                forecast_alignment_reward = 0.0
                alignment_score = 0.0
                misalignment_penalty_total = 0.0

                # ALWAYS compute forecast_direction from z-scores (even if investor_strategy is None)
                delta_short = float(getattr(self, 'z_short_price', 0.0))
                delta_medium = float(getattr(self, 'z_medium_price', 0.0))
                delta_long = float(getattr(self, 'z_long_price', 0.0))

                # DEBUG: Log z-scores at step 500
                if i == 500:
                    logger.debug(f"[DEBUG_Z_SCORES] t={i} delta_short={delta_short:.6f} delta_medium={delta_medium:.6f} delta_long={delta_long:.6f}")
                    logger.debug(f"[DEBUG_Z_SCORES] t={i} z_short_price={getattr(self, 'z_short_price', 'NOT_SET')}")

                # Weighted forecast direction (reduce short-horizon dominance)
                forecast_direction = (
                    0.35 * np.sign(delta_short) +
                    0.40 * np.sign(delta_medium) +
                    0.25 * np.sign(delta_long)
                )
                forecast_direction_normalized = np.clip(forecast_direction, -1.0, 1.0)

                # DEBUG: Log forecast_direction at step 500
                if i == 500:
                    logger.debug(f"[DEBUG_FORECAST_DIR] t={i} forecast_direction={forecast_direction:.6f} (sign_short={np.sign(delta_short)}, sign_med={np.sign(delta_medium)}, sign_long={np.sign(delta_long)})")

                # Helper for recent MAPE extraction (safe defaults)
                def _mean_recent_mape(horizon, n=10):
                    try:
                        if hasattr(self, '_horizon_mape') and horizon in self._horizon_mape:
                            seq = list(self._horizon_mape.get(horizon, []))
                            if len(seq) > 0:
                                return float(np.mean(seq[-n:]))
                    except Exception:
                        pass
                    return float('nan')

                mape_threshold_short = mape_thresholds.get('short', 0.02) if mape_thresholds else 0.02
                mape_threshold_medium = mape_thresholds.get('medium', 0.02) if mape_thresholds else 0.02
                mape_threshold_long = mape_thresholds.get('long', 0.02) if mape_thresholds else 0.02

                measured_mape_short = _mean_recent_mape('short', n=10)
                measured_mape_medium = _mean_recent_mape('medium', n=10)
                measured_mape_long = _mean_recent_mape('long', n=10)

                def _resolve_mape(measured_val, threshold_val):
                    return measured_val if not np.isnan(measured_val) else threshold_val

                weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
                ms = _resolve_mape(measured_mape_short, mape_threshold_short)
                mm = _resolve_mape(measured_mape_medium, mape_threshold_medium)
                ml = _resolve_mape(measured_mape_long, mape_threshold_long)
                avg_mape = (weights['short'] * ms +
                            weights['medium'] * mm +
                            weights['long'] * ml)

                # Default quality multiplier derived from avg_mape (applies even when investor_strategy missing)
                if avg_mape < 0.03:
                    quality_mult = 2.0
                elif avg_mape < 0.08:
                    quality_mult = 1.0
                else:
                    quality_mult = 0.3

                if investor_strategy is not None and forecast_deltas is not None:

                    # Position direction
                    position_direction = np.sign(position_normalized)
                    agent_followed_forecast = (
                        forecast_direction_normalized * position_direction > 0
                    ) if abs(forecast_direction_normalized) > 0.01 else False

                    if forecast_gate_passed:
                        alignment_value = position_direction * forecast_direction_normalized * position_exposure
                        alignment_score = float(np.clip(alignment_value, -1.0, 1.0))
                        forecast_alignment_reward = np.clip(5.0 * alignment_score * quality_mult, -5.0, 5.0)

                        signal_strength = float(investor_strategy.get('signal_strength', 0.5)) if investor_strategy else 0.0
                        trade_signal_active = bool(investor_strategy and investor_strategy.get('trade_signal', False))
                        hedge_signal_active = bool(investor_strategy and investor_strategy.get('hedge_signal', False))
                        direction_conflict = trade_signal_active and position_exposure >= 0.12 and not agent_followed_forecast

                        if trade_signal_active and agent_followed_forecast:
                            # FIXED: Increased reward for following forecasts with positions
                            follow_scale = float(np.clip(position_exposure / 0.15, 0.0, 1.0))
                            follow_bonus = 0.50 * signal_strength * quality_mult * max(follow_scale, 0.15)  # Increased from 0.35
                            forecast_alignment_reward += follow_bonus
                            
                            # NEW: Additional bonus for taking meaningful positions (>0.05 exposure)
                            if position_exposure > 0.05:
                                position_bonus = 0.20 * signal_strength * quality_mult * min(position_exposure / 0.3, 1.0)
                                forecast_alignment_reward += position_bonus

                        if direction_conflict:
                            penalty_scale = float(np.clip((position_exposure - 0.1) / 0.3, 0.0, 1.0))
                            penalty = 0.35 * signal_strength * quality_mult * penalty_scale
                            forecast_alignment_reward -= penalty
                            misalignment_penalty_total += penalty
                        elif trade_signal_active and position_exposure < 0.05:
                            # FIXED: Stronger penalty when agent ignores strong entry signals
                            entry_gap = float(np.clip(1.0 - (position_exposure / 0.05), 0.0, 1.0))
                            penalty = 0.25 * signal_strength * quality_mult * entry_gap  # Increased from 0.12
                            forecast_alignment_reward -= penalty
                            misalignment_penalty_total += penalty

                        if not trade_signal_active and not hedge_signal_active:
                            # FIXED: Reduced penalty for positions when no signal (allow some positions)
                            # Only penalize if position is very large (>0.2) when no signal
                            neutral_penalty = np.clip(position_exposure - 0.2, 0.0, 1.0)  # Increased threshold from 0.1 to 0.2
                            neutral_penalty_value = 0.6 * neutral_penalty * quality_mult  # Reduced from 1.2 to 0.6
                            forecast_alignment_reward -= neutral_penalty_value
                            misalignment_penalty_total += neutral_penalty_value
                    else:
                        forecast_alignment_reward = 0.0

                # CRITICAL FIX: Use SAME base reward structure as Tier 1 for fair comparison
                # Tier 2 should get the SAME base rewards as Tier 1, PLUS forecast bonuses when gate passes
                # This ensures the only difference is forecast usage, not base reward structure
                
                # Component 1: BASE PnL REWARD (REVERTED: Original single-step price return)
                # REVERTED: Back to original single-step price return calculation (no multi-step, no MTM delta)
                current_price = float(np.clip(self._price_raw[i], 50.0, 1e9))
                if i > 0:
                    prev_price = float(np.clip(self._price_raw[i-1], 50.0, 1e9))
                    price_return_raw = (current_price - prev_price) / max(abs(prev_price), 1e-6)
                    price_return_calc = np.clip(price_return_raw, -0.1, 0.1)
                else:
                    price_return_calc = 0.0
                
                # Base PnL reward from realized returns (original single-step calculation)
                base_pnl_reward = float(position_normalized * price_return_calc * 100.0)
                
                # Component 2: BASE EXPLORATION BONUS (SAME as Tier 1 for fair comparison)
                # FAIR COMPARISON: Tier 2/3 get EXACTLY same exploration bonus as Tier 1
                # Any performance difference must come from forecast observations + GNN architecture, NOT reward differences
                episode_count = getattr(self, '_episode_counter', 0)
                base_exploration_bonus = 0.0
                if episode_count < 15:  # Same as Tier 1
                    if total_position > 1e-6:  # Only if has positions
                        position_ratio = float(np.clip(total_position / max(max_pos, 1.0), 0.0, 1.0))
                        base_exploration_bonus = 0.05 * position_ratio  # SAME as Tier 1 (0.05)
                
                # BASE REWARD (identical to Tier 1)
                base_forecast_score = base_pnl_reward + base_exploration_bonus
                
                # Component 3: FORECAST BONUS (only when gate passes)
                # FAIR COMPARISON: Tier 2/3 get NO explicit forecast rewards in observations-only mode
                # They must learn to use forecasts through better PnL outcomes only
                forecast_bonus = 0.0
                if forecast_gate_passed:
                    # Use forecast_alignment_reward as bonus (not replacement for base)
                    forecast_bonus = forecast_alignment_reward
                
                # Component 4: RISK MANAGEMENT REWARD (only when gate passes)
                # For fair comparison: Only add risk/penalty components when forecasts are actually used
                risk_reward = 0.0
                extreme_delta_penalty = 0.0
                
                if forecast_gate_passed:
                    # Only apply risk rewards and penalties when forecast gate passes
                    # This ensures Tier 2 gets EXACTLY same base rewards as Tier 1 when gate is blocked
                    risk_reward = risk_adj.get('risk_reward', 0.0) if risk_adj else 0.0
                    
                    # EXTREME DELTA PENALTY: Prevent trading on noise (deltas > 50%)
                if forecast_deltas is not None and position_exposure > 0:
                    delta_short_raw = abs(forecast_deltas.get('short', 0.0))
                    delta_medium_raw = abs(forecast_deltas.get('medium', 0.0))
                    delta_long_raw = abs(forecast_deltas.get('long', 0.0))

                    if delta_short_raw > 0.5 or delta_medium_raw > 0.5 or delta_long_raw > 0.5:
                        extreme_delta_penalty = -200.0 * position_exposure

                # Combined forecast score: BASE (same as Tier 1) + FORECAST BONUS (only when gate passes)
                # When gate blocked: EXACTLY same as Tier 1 (base_forecast_score only)
                # When gate passes: base + forecast_bonus + risk + penalties
                forecast_signal_score = (
                    base_forecast_score +      # Same base rewards as Tier 1 (always)
                    forecast_bonus +          # Additional forecast bonus (only when gate passes)
                    0.1 * risk_reward +        # Risk component (only when gate passes)
                    extreme_delta_penalty      # Penalties (only when gate passes)
                )
                
                # For backward compatibility, also compute pnl_reward (used in logging)
                pnl_reward = base_pnl_reward

                # Get price_return_forecast from latest price returns (same as used in forecast engine)
                latest_price_returns = getattr(self, '_latest_price_returns', None)
                if latest_price_returns and 'price_return_forecast' in latest_price_returns:
                    price_return_forecast = float(latest_price_returns['price_return_forecast'])
                else:
                    price_return_forecast = price_return_calc  # Fallback to actual return
                realized_vs_forecast = float(price_return_calc - price_return_forecast)

                # Store for debugging
                self._last_forecast_risk_adj = risk_adj
                self._last_pnl_reward = pnl_reward
                self._last_forecast_alignment_reward = forecast_alignment_reward
                self._last_risk_reward = risk_reward

                # DIAGNOSTIC LOGGING: Track new reward components (every 500 steps)
                if i % 500 == 0:
                    logger.info(f"[FORECAST_REWARD] t={i} total={forecast_signal_score:.2f} base={base_forecast_score:.2f} forecast_bonus={forecast_bonus:.2f} pnl={pnl_reward:.2f}")
                    logger.info(f"[FORECAST_REWARD_NEW] t={i} total={forecast_signal_score:.2f} "
                               f"base={base_forecast_score:.2f} (same as Tier1) forecast_bonus={forecast_bonus:.2f} "
                               f"risk={risk_reward:.2f} extreme_penalty={extreme_delta_penalty:.2f} gate_passed={forecast_gate_passed}")
                    if investor_strategy is not None:
                        logger.info(f"  strategy={investor_strategy['strategy']} signal={investor_strategy.get('signal_strength', 0.0):.3f} pos_exp={position_exposure:.3f}")
                        logger.info(f"  strategy={investor_strategy['strategy']} signal_strength={investor_strategy.get('signal_strength', 0.0):.3f} "
                                   f"consensus={investor_strategy.get('consensus', False)} position_exp={position_exposure:.3f}")
                    if forecast_deltas:
                        logger.debug(f"  deltas: short={forecast_deltas.get('short', 0.0):.4f} med={forecast_deltas.get('medium', 0.0):.4f} long={forecast_deltas.get('long', 0.0):.4f}")
                    if mape_thresholds_capacity:
                        logger.debug(f"  MAPE(capacity): short={mape_thresholds_capacity.get('short', 0.0):.4f} med={mape_thresholds_capacity.get('medium', 0.0):.4f} long={mape_thresholds_capacity.get('long', 0.0):.4f}")

                    # Get MAPE values (for CSV logging) - log measured when available, else thresholds
                mape_short = float(measured_mape_short) if not np.isnan(measured_mape_short) else float(mape_threshold_short)
                mape_medium = float(measured_mape_medium) if not np.isnan(measured_mape_medium) else float(mape_threshold_medium)
                mape_long = float(measured_mape_long) if not np.isnan(measured_mape_long) else float(mape_threshold_long)

                # Check alignment: Did agent follow forecast?
                # forecast_direction is already computed from z-scores on line 5329
                position_direction = np.sign(total_position)
                agent_followed_forecast = (forecast_direction * position_direction > 0) if abs(forecast_direction) > 0.01 else False
                # Lower exposure requirement so hedged-but-correct positions still earn forecast credit
                forecast_used_flag = bool(
                    forecast_gate_passed and agent_followed_forecast and position_exposure > 0.02
                )

                if forecast_gate_passed:
                    if forecast_used_flag:
                        forecast_usage_reason = 'used'
                    elif position_exposure <= 0.02:
                        forecast_usage_reason = 'low_exposure'
                    elif not agent_followed_forecast:
                        forecast_usage_reason = 'direction_mismatch'
                    else:
                        forecast_usage_reason = 'risk_blocked'
                else:
                    forecast_usage_reason = 'gate_blocked'

                def _direction_accuracy(pred_signal: float, realized_ret: float, min_abs: float = 1e-4) -> float:
                    if pred_signal is None or realized_ret is None:
                        return 0.0
                    if abs(realized_ret) < min_abs or abs(pred_signal) < min_abs:
                        return 0.0
                    return 1.0 if np.sign(pred_signal) == np.sign(realized_ret) else 0.0

                direction_accuracy_short = _direction_accuracy(z_short, price_return_short)
                direction_accuracy_medium = _direction_accuracy(z_medium, price_return_medium)
                direction_accuracy_long = _direction_accuracy(z_long, price_return_long)
                combined_conf_debug = float(getattr(self, '_combined_confidence', 0.5))
                reward_weights = getattr(self, 'reward_weights', {})
                adaptive_multiplier_debug = float(
                    getattr(self, '_adaptive_forecast_weight', reward_weights.get('forecast', 0.0))
                )
                warmup_factor_debug = float(getattr(self, '_forecast_warmup_factor', 1.0))

                # FIX: Store Tier 2 debug info in _debug_forecast_reward for CSV logging
                if i == 500:
                    logger.debug(f"[DEBUG_DICT_TIER2] t={i} CREATING Tier 2 dictionary with forecast_direction={forecast_direction:.6f}")
                self._debug_forecast_reward = {
                    'position_signed': total_position,
                    'position_exposure': position_exposure,
                    'price_return_1step': price_return_calc,
                    'price_return_forecast': price_return_forecast,
                    'forecast_signal_score': forecast_signal_score,
                    'wind_pos_norm': current_positions.get('wind', 0.0) / max(max_pos, 1.0),
                    'solar_pos_norm': current_positions.get('solar', 0.0) / max(max_pos, 1.0),
                    'hydro_pos_norm': current_positions.get('hydro', 0.0) / max(max_pos, 1.0),
                    'z_short': delta_short,  # Use z-scores computed on line 5324
                    'z_medium': delta_medium,  # Use z-scores computed on line 5325
                    'z_long': delta_long,  # Use z-scores computed on line 5326
                    'z_combined': 0.0,  # Not used in new approach
                    # Store raw MAPE and a conventional confidence (higher is better)
                    'forecast_mape': float(avg_mape) if 'avg_mape' in locals() else 0.0,
                    'forecast_confidence': float(np.clip(1.0 / (1.0 + avg_mape), 0.0, 1.0)) if 'avg_mape' in locals() else 0.0,
                    'forecast_trust': forecast_trust,
                    'trust_scale': forecast_trust,
                    'forecast_gate_passed': forecast_gate_passed,
                    'forecast_used': forecast_used_flag,
                    'forecast_not_used_reason': forecast_usage_reason,
                    'forecast_usage_bonus': 0.0,
                    'investor_strategy_multiplier': float(getattr(self, '_last_investor_strategy_multiplier', 1.0)),
                    'alignment_reward': forecast_alignment_reward,
                    'pnl_reward': pnl_reward,
                    'base_alignment': alignment_score,
                    'profitability_factor': pnl_reward,
                    'alignment_multiplier': quality_mult,
                    'misalignment_penalty_mult': misalignment_penalty_total,
                    'strategy_reward': 0.0,  # Merged into alignment_reward
                    'risk_reward': risk_reward,
                    'exploration_bonus': 0.0,
                    'position_size_bonus': 0.0,
                    'signal_gate_multiplier': float(
                        risk_adj.get('signal_gate_multiplier', getattr(self.forecast_risk_manager, '_last_gate_multiplier', 0.0) if self.forecast_risk_manager is not None else 0.0)
                    ) if risk_adj else float(getattr(self.forecast_risk_manager, '_last_gate_multiplier', 0.0) if self.forecast_risk_manager is not None else 0.0),
                    'adaptive_multiplier': adaptive_multiplier_debug,
                    # NEW: Forecast vs actual comparison
                    'current_price_dkk': current_price,
                    'forecast_price_short_dkk': forecast_price_short,
                    'forecast_price_medium_dkk': forecast_price_medium,
                    'forecast_price_long_dkk': forecast_price_long,
                    'forecast_error_short_pct': forecast_error_short,
                    'forecast_error_medium_pct': forecast_error_medium,
                    'forecast_error_long_pct': forecast_error_long,
                    'forecast_error': realized_vs_forecast,
                    'realized_vs_forecast': realized_vs_forecast,
                    'mape_short': mape_short,
                    'mape_medium': mape_medium,
                    'mape_long': mape_long,
                    'forecast_direction': forecast_direction,
                    'position_direction': position_direction,
                    'agent_followed_forecast': agent_followed_forecast,
                    'combined_confidence': combined_conf_debug,
                    'combined_forecast_score': forecast_signal_score,
                    'warmup_factor': warmup_factor_debug,
                    'direction_accuracy_short': direction_accuracy_short,
                    'direction_accuracy_medium': direction_accuracy_medium,
                    'direction_accuracy_long': direction_accuracy_long,
                }

                # LOG: Append Tier 2 portfolio attribution row to tier2/logs/tier2_portfolio_attrib_ep{episode}.csv
                # FIXED: Use the log_dir passed to environment instead of hardcoded path
                try:
                    episode_idx = int(getattr(self, '_episode_counter', 0))
                    # Use the same log_dir as RewardLogger (passed to environment constructor)
                    # CRITICAL: Always use debug_tracker.log_dir - no hardcoded fallback
                    if not (hasattr(self, 'debug_tracker') and hasattr(self.debug_tracker, 'log_dir')):
                        # Should never happen, but if it does, skip this logging to avoid creating wrong folder
                        logger.warning(f"[TIER2_LOG] debug_tracker.log_dir not available, skipping portfolio attribution log")
                    else:
                        log_dir = self.debug_tracker.log_dir
                    os.makedirs(log_dir, exist_ok=True)
                    csv_path = os.path.join(log_dir, f"tier2_portfolio_attrib_ep{episode_idx}.csv")
                    # Prepare standardized schema for portfolio attribution
                    episode_idx_val = episode_idx
                    price_current_val = float(np.clip(self._price_raw[i], self.config.minimum_price_filter, self.config.maximum_price_cap))
                    price_return_1step_val = float(self._price_return_1step[i]) if i < len(self._price_return_1step) else 0.0
                    price_return_forecast_val = float(self._debug_forecast_reward.get('price_return_forecast', 0.0))

                    fieldnames = [
                        'episode','timestep',
                        'nav_start','nav_end','pnl_total','pnl_battery','pnl_generation','pnl_hedge','cash_delta_ops',
                        'position_exposure_norm','capital_alloc_frac','confidence_multiplier',
                        'forecast_trust','forecast_gate_passed','combined_forecast_score','mape_short','mape_medium','mape_long',
                        'battery_decision','battery_intensity','battery_spread','battery_adjusted_hurdle','battery_volatility_adj',
                            'battery_energy','battery_capacity','battery_soc','battery_cash_delta','battery_throughput','battery_degradation_cost','battery_eta_charge','battery_eta_discharge',
                        'price_current','price_return_1step','price_return_forecast'
                    ]
                    row = {
                        'episode': episode_idx_val,
                        'timestep': i,
                        'nav_start': float(getattr(self, 'nav_start', getattr(self, 'init_budget', 0.0))),
                        'nav_end': float(getattr(self, 'nav_end', getattr(self, 'budget', 0.0))),
                        'pnl_total': float(getattr(self, '_last_total_pnl', 0.0)),
                        'pnl_battery': float(getattr(self, '_last_battery_cash_delta', 0.0)),
                        'pnl_generation': float(getattr(self, '_last_generation_pnl', 0.0)),
                        'pnl_hedge': float(getattr(self, '_last_hedge_pnl', 0.0)),
                        'cash_delta_ops': float(getattr(self, '_last_cash_ops', 0.0)),
                        'position_exposure_norm': float(self._debug_forecast_reward.get('position_exposure', 0.0)),
                        'capital_alloc_frac': float(getattr(self, 'capital_allocation_fraction', 0.0)),
                        'confidence_multiplier': float(getattr(self, '_last_investor_strategy_multiplier', 1.0)),
                        'forecast_trust': float(self._debug_forecast_reward.get('forecast_trust', 0.0)),
                        'forecast_gate_passed': bool(self._debug_forecast_reward.get('forecast_gate_passed', False)),
                        'combined_forecast_score': float(self._debug_forecast_reward.get('combined_forecast_score', 0.0)),
                        'mape_short': float(self._debug_forecast_reward.get('mape_short', 0.0)),
                        'mape_medium': float(self._debug_forecast_reward.get('mape_medium', 0.0)),
                        'mape_long': float(self._debug_forecast_reward.get('mape_long', 0.0)),
                        'battery_decision': str(getattr(self, '_last_battery_decision', 'idle')),
                        'battery_intensity': float(getattr(self, '_last_battery_intensity', 0.0)),
                        'battery_spread': float(getattr(self, '_last_battery_spread', 0.0)),
                        'battery_adjusted_hurdle': float(getattr(self, '_last_battery_adjusted_hurdle', 0.0)),
                        'battery_volatility_adj': float(getattr(self, '_last_battery_volatility_adj', 0.0)),
                        'battery_energy': float(getattr(self, '_last_battery_energy', self.operational_state.get('battery_energy', 0.0))),
                        'battery_capacity': float(getattr(self, '_last_battery_capacity', self.physical_assets.get('battery_capacity_mwh', 0.0))),
                        'battery_soc': float(getattr(self, '_last_battery_soc', 0.0)),
                        'battery_cash_delta': float(getattr(self, '_last_battery_cash_delta', 0.0)),
                        'battery_throughput': float(getattr(self, '_last_battery_throughput', 0.0)),
                        'battery_degradation_cost': float(getattr(self, '_last_battery_degradation_cost', 0.0)),
                        'battery_eta_charge': float(getattr(self, '_last_battery_eta_charge', self.batt_eta_charge)),
                        'battery_eta_discharge': float(getattr(self, '_last_battery_eta_discharge', self.batt_eta_discharge)),
                        'price_current': price_current_val,
                        'price_return_1step': price_return_1step_val,
                        'price_return_forecast': price_return_forecast_val,
                    }
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='') as f:
                        import csv
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                except Exception:
                    pass

                # LOG: Mirror key NAV attribution into Tier 2 debug CSV as well
                # FIXED: Use the log_dir passed to environment instead of hardcoded path
                try:
                    episode_idx = int(getattr(self, '_episode_counter', 0))
                    # Use the same log_dir as RewardLogger (passed to environment constructor)
                    # CRITICAL: Always use debug_tracker.log_dir - no hardcoded fallback
                    if not (hasattr(self, 'debug_tracker') and hasattr(self.debug_tracker, 'log_dir')):
                        # Should never happen, but if it does, skip this logging to avoid creating wrong folder
                        logger.warning(f"[TIER2_LOG] debug_tracker.log_dir not available, skipping debug CSV log")
                    else:
                        log_dir = self.debug_tracker.log_dir
                    os.makedirs(log_dir, exist_ok=True)
                    csv_path_dbg = os.path.join(log_dir, f"tier2_debug_ep{episode_idx}.csv")
                    dbg_row = {
                        'timestep': i,
                        'nav_start': float(getattr(self, 'nav_start', getattr(self, 'init_budget', 0.0))),
                        'nav_end': float(getattr(self, 'nav_end', getattr(self, 'budget', 0.0))),
                        'pnl_total': float(getattr(self, '_last_total_pnl', 0.0)),
                        'pnl_battery': float(getattr(self, '_last_battery_cash_delta', 0.0)),
                        'pnl_generation': float(getattr(self, '_last_generation_pnl', 0.0)),
                        'pnl_hedge': float(getattr(self, '_last_hedge_pnl', 0.0)),
                        'cash_delta_ops': float(getattr(self, '_last_cash_ops', 0.0)),
                        'position_exposure_norm': float(self._debug_forecast_reward.get('position_exposure', 0.0)),
                        'capital_alloc_frac': float(getattr(self, 'capital_allocation_fraction', 0.0)),
                        'confidence_multiplier': float(getattr(self, '_last_investor_strategy_multiplier', 1.0)),
                        'forecast_trust': float(self._debug_forecast_reward.get('forecast_trust', 0.0)),
                        'forecast_gate_passed': bool(self._debug_forecast_reward.get('forecast_gate_passed', False)),
                        'combined_forecast_score': float(self._debug_forecast_reward.get('combined_forecast_score', 0.0)),
                        'mape_short': float(self._debug_forecast_reward.get('mape_short', 0.0)),
                        'mape_medium': float(self._debug_forecast_reward.get('mape_medium', 0.0)),
                        'mape_long': float(self._debug_forecast_reward.get('mape_long', 0.0)),
                        'battery_decision': str(getattr(self, '_last_battery_decision', 'idle')),
                        'battery_intensity': float(getattr(self, '_last_battery_intensity', 0.0)),
                        'battery_spread': float(getattr(self, '_last_battery_spread', 0.0)),
                        'battery_adjusted_hurdle': float(getattr(self, '_last_battery_adjusted_hurdle', 0.0)),
                        'battery_volatility_adj': float(getattr(self, '_last_battery_volatility_adj', 0.0)),
                    }
                    write_header_dbg = not os.path.exists(csv_path_dbg)
                    with open(csv_path_dbg, 'a', newline='') as f:
                        import csv
                        w = csv.DictWriter(f, fieldnames=list(dbg_row.keys()))
                        if write_header_dbg:
                            w.writeheader()
                        w.writerow(dbg_row)
                except Exception:
                    pass

            else:
                # TIER 1: No forecasts - Pure PnL-based reward (FAIR COMPARISON)
                # CRITICAL FIX: Remove exploration/position bonuses that give Tier 1 unfair advantage
                # Both tiers should use IDENTICAL reward structure, just Tier 2 has forecast signals

                # Get current position exposure
                max_pos = self.config.max_position_size * self.init_budget * self.config.capital_allocation_fraction if self.init_budget > 0 else 1.0
                wind_pos_value = self.financial_positions.get('wind_instrument_value', 0.0)
                solar_pos_value = self.financial_positions.get('solar_instrument_value', 0.0)
                hydro_pos_value = self.financial_positions.get('hydro_instrument_value', 0.0)

                # Normalize positions
                wind_pos_norm = float(wind_pos_value / max(max_pos, 1.0))
                solar_pos_norm = float(solar_pos_value / max(max_pos, 1.0))
                hydro_pos_norm = float(hydro_pos_value / max(max_pos, 1.0))

                # SIGNED total position
                position_signed = float(wind_pos_norm + solar_pos_norm + hydro_pos_norm)
                position_exposure = float(abs(wind_pos_norm) + abs(solar_pos_norm) + abs(hydro_pos_norm))

                # TIER 1: REVERTED to original single-step price return calculation
                # REVERTED: Back to original single-step price return (no multi-step, no MTM delta)
                current_price = float(np.clip(self._price_raw[i], 50.0, 1e9))
                if i > 0:
                    prev_price = float(np.clip(self._price_raw[i-1], 50.0, 1e9))
                    price_return_raw = (current_price - prev_price) / max(abs(prev_price), 1e-6)
                    price_return_calc = np.clip(price_return_raw, -0.1, 0.1)
                else:
                    price_return_calc = 0.0

                # Current positions (Model B): use normalized exposures (DKK / max_pos_dkk)
                max_pos_dkk = max(float(max_pos), 1.0)
                current_positions = {
                    'wind': float(wind_pos_value / max_pos_dkk),
                    'solar': float(solar_pos_value / max_pos_dkk),
                    'hydro': float(hydro_pos_value / max_pos_dkk),
                }
                total_position = float(sum(current_positions.values()))
                position_normalized = float(total_position)  # already normalized by max_pos_dkk
                
                # Base PnL reward from realized returns (original single-step calculation)
                pnl_reward = float(position_normalized * price_return_calc * 100.0)

                # FAIR COMPARISON: Same exploration bonus for both tiers
                # This ensures any performance difference is due to forecast integration, not exploration parameters
                exploration_bonus_tier1 = 0.0
                episode_count = getattr(self, '_episode_counter', 0)
                if episode_count < 15:  # Same as Tier 2 for fair comparison
                    if total_position > 1e-6:  # Only if has positions
                        position_ratio_tier1 = float(np.clip(total_position / max(max_pos, 1.0), 0.0, 1.0))
                        exploration_bonus_tier1 = 0.05 * position_ratio_tier1  # Same as Tier 2 for fair comparison
                
                # Tier 1 gets PnL reward + exploration bonus (no strategy/risk bonuses)
                forecast_signal_score = pnl_reward + exploration_bonus_tier1

                # FIX: Store Tier 1 debug info in _debug_forecast_reward for logging
                if i == 500:
                    logger.debug(f"[DEBUG_DICT_TIER1] t={i} CREATING Tier 1 dictionary (OVERWRITING Tier 2!)")
                self._debug_forecast_reward = {
                    'position_signed': position_signed,
                    'position_exposure': position_exposure,
                    'price_return_1step': price_return,
                    'price_return_forecast': price_return,
                    'forecast_signal_score': forecast_signal_score,
                    'wind_pos_norm': wind_pos_norm,
                    'solar_pos_norm': solar_pos_norm,
                    'hydro_pos_norm': hydro_pos_norm,
                    'z_short': 0.0,
                    'z_medium': 0.0,
                    'z_long': 0.0,
                    'z_combined': 0.0,
                    'forecast_confidence': 0.0,
                    'forecast_trust': 0.0,
                    'alignment_reward': 0.0,
                    'pnl_reward': pnl_reward,
                    'exploration_bonus': exploration_bonus_tier1,
                    'position_size_bonus': 0.0,
                }

                # LOG: For Tier 1, append similar attribution for fair comparison
                # FIXED: Use the log_dir passed to environment instead of hardcoded path
                try:
                    episode_idx = int(getattr(self, '_episode_counter', 0))
                    # Use the same log_dir as RewardLogger (passed to environment constructor)
                    # CRITICAL: Always use debug_tracker.log_dir - no hardcoded fallback
                    if not (hasattr(self, 'debug_tracker') and hasattr(self.debug_tracker, 'log_dir')):
                        # Should never happen, but if it does, skip this logging to avoid creating wrong folder
                        logger.warning(f"[TIER1_LOG] debug_tracker.log_dir not available, skipping portfolio attribution log")
                    else:
                        log_dir = self.debug_tracker.log_dir
                    os.makedirs(log_dir, exist_ok=True)
                    csv_path = os.path.join(log_dir, f"tier1_portfolio_ep{episode_idx}.csv")
                    row = {
                        'timestep': i,
                        'nav_start': float(getattr(self, 'nav_start', getattr(self, 'init_budget', 0.0))),
                        'nav_end': float(getattr(self, 'nav_end', getattr(self, 'budget', 0.0))),
                        'pnl_total': float(getattr(self, '_last_total_pnl', 0.0)),
                        'pnl_battery': float(getattr(self, '_last_battery_cash_delta', 0.0)),
                        'pnl_generation': float(getattr(self, '_last_generation_pnl', 0.0)),
                        'pnl_hedge': float(getattr(self, '_last_hedge_pnl', 0.0)),
                        'cash_delta_ops': float(getattr(self, '_last_cash_ops', 0.0)),
                        'position_exposure_norm': float(self._debug_forecast_reward.get('position_exposure', 0.0)),
                        'capital_alloc_frac': float(getattr(self, 'capital_allocation_fraction', 0.0)),
                        'confidence_multiplier': float(getattr(self, '_last_investor_strategy_multiplier', 1.0)),
                    }
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='') as f:
                        import csv
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                except Exception:
                    pass
            # END: TIER 1 (no forecasts)
            # END: if enable_forecast_util

            # =====================================================================
            # AGENT-SPECIFIC REWARDS
            # =====================================================================
            # Calculate agent-specific rewards based on forecast_signal_score
            # (computed above in lines 5112-5335)

            # Continue to agent reward calculation below...
            # All orphaned alignment code has been removed

            # =====================================================================
            # ACTUAL AGENT REWARD CALCULATION STARTS HERE
            # =====================================================================
            # The forecast_signal_score computed above (lines 5112-5335) is used
            # in the agent reward calculation below

            # Orphaned logging code removed - forecast_signal_score is already computed above

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
                # FIX: Add trading_gains and operating_gains for debug logging
                'trading_gains': getattr(self, 'cumulative_mtm_pnl', 0.0),
                'operating_gains': getattr(self, 'cumulative_generation_revenue', 0.0),
            }

        except Exception as e:
            logger.error(f"Finance update error: {e}")
            return {
                'revenue': 0.0,
                'generation_revenue': 0.0,
                'battery_cash_delta': 0.0,
                'mtm_pnl': 0.0,
                'fund_nav': self.equity,
                'portfolio_value': self.equity,
                # FIX: Add trading_gains and operating_gains for debug logging (even in error case)
                'trading_gains': getattr(self, 'cumulative_mtm_pnl', 0.0),
                'operating_gains': getattr(self, 'cumulative_generation_revenue', 0.0),
                'forecast_signal_score': 0.0,
                'efficiency': 0.0,
            }

    def _calculate_timestep_costs(self, fund_physical_investment: float, gross_revenue: float,
                                   fund_total_generation_mwh: float) -> float:
        """
        REFACTORED: Calculate total operational and administrative costs using FinancialEngine.

        Args:
            fund_physical_investment: Total capital invested in physical assets (DKK)
            gross_revenue: Gross revenue from electricity sales (DKK)
            fund_total_generation_mwh: Total generation for this timestep (MWh)

        Returns:
            Total operating costs for this timestep (DKK)
        """
        from financial_engine import FinancialEngine
        return FinancialEngine.calculate_timestep_costs(
            fund_physical_investment,
            gross_revenue,
            fund_total_generation_mwh,
            self.config
        )

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
            logger.warning(f"Generation revenue calculation failed: {e}")
            return 0.0

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
            logger.warning(f"Cash distribution failed: {e}")

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
            logger.error(f"Emergency reallocation check failed: {e}")

    def _calculate_fund_administration_costs(self) -> float:
        """
        Calculate minimal fund administration costs that apply during asset acquisition phase.
        These are basic costs of running the fund (~0.05% annually total):
        - Regulatory compliance, audit, custody, basic administration
        - NO management fees, insurance, or operational costs during acquisition
        """
        try:
            time_step_hours = 10.0 / 60.0  # 10-minute timesteps
            annual_to_timestep = time_step_hours / 8760

            # REMOVED: Minimal administration costs to match PrototypeTestTuned (no extra cost dip)
            # Only keep basic administration fees like PrototypeTestTuned
            administration_fees = self.init_budget * self.administration_fee_rate * annual_to_timestep

            total_admin_costs = administration_fees
            return float(total_admin_costs)

        except Exception as e:
            logger.warning(f"Fund administration cost calculation failed: {e}")
            return 0.0

    def _get_wind_capacity_factor(self, i: int) -> float:
        try:
            raw_wind = float(self._wind[i]) if i < len(self._wind) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic wind CF range (15-45%) - industry standard
            normalized = raw_wind / max(self.wind_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.15 + (normalized * 0.30)  # Map to 15-45% range
            return float(np.clip(realistic_cf, 0.0, 0.45))
        except Exception:
            return 0.25  # Typical wind CF

    def _get_solar_capacity_factor(self, i: int) -> float:
        try:
            raw_solar = float(self._solar[i]) if i < len(self._solar) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic solar CF range (5-30%) - industry standard
            normalized = raw_solar / max(self.solar_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.05 + (normalized * 0.25)  # Map to 5-30% range
            return float(np.clip(realistic_cf, 0.0, 0.30))
        except Exception:
            return 0.15  # Typical solar CF

    def _get_hydro_capacity_factor(self, i: int) -> float:
        try:
            raw_hydro = float(self._hydro[i]) if i < len(self._hydro) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic hydro CF range (35-65%) - industry standard
            normalized = raw_hydro / max(self.hydro_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.35 + (normalized * 0.30)  # Map to 35-65% range
            return float(np.clip(realistic_cf, 0.35, 0.65))
        except Exception:
            return 0.50  # Typical hydro CF

    def _get_total_generation_mwh(self, i: int) -> float:
        """Get total electricity generation this timestep"""
        try:
            time_step_hours = 10.0 / 60.0
            wind_mwh = self.physical_assets['wind_capacity_mw'] * self._get_wind_capacity_factor(i) * time_step_hours
            solar_mwh = self.physical_assets['solar_capacity_mw'] * self._get_solar_capacity_factor(i) * time_step_hours
            hydro_mwh = self.physical_assets['hydro_capacity_mw'] * self._get_hydro_capacity_factor(i) * time_step_hours
            return float(wind_mwh + solar_mwh + hydro_mwh)
        except Exception:
            return 0.0

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
        except Exception:
            return 0.5

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

            # Pass overlay history to reward calculator for predictive reward shaping
            if hasattr(self, '_overlay_pred_r_hist'):
                self.reward_calculator._overlay_pred_r_hist = self._overlay_pred_r_hist

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

            # CRITICAL FIX: Decompose base_reward into agent-specific components
            # BEFORE: Each agent computed INDEPENDENT reward = base_reward + large bonus
            # Result: PPO saw 2x magnitude, 30% clipping, 58% variance inflation
            # 
            # AFTER: Each agent gets FRACTIONAL share of base_reward + SMALL delta
            # Result: PPO sees normalized signal, <5% clipping, clean gradient flow
            # This is ESSENTIAL for forecast integration to work (signal/noise improves)
            
            agent_rewards = {}
            debug_forecast = getattr(self, '_debug_forecast_reward', {})
            
            # Allocate base_reward among agents: investor(40%), battery(30%), risk(20%), meta(10%)
            # These ratios reflect each agent's contribution to overall portfolio management
            investor_base = base_reward * 0.40
            battery_base = base_reward * 0.30
            risk_base = base_reward * 0.20
            meta_base = base_reward * 0.10
            
            # ===== INVESTOR AGENT =====
            # Small delta for MTM gains (trading performance)
            mtm_pnl = float(financial.get('mtm_pnl', 0.0))
            investor_mtm_delta = float(0.05 * np.clip(mtm_pnl / 10000.0, -1.0, 1.0))  # REDUCED: was 0.20
            
            # Small delta for strategy alignment (forecast-driven positioning)
            investor_strategy_delta = 0.0
            position_alignment_status = 'no_strategy'  # For logging
            if self.forecast_risk_manager is not None and getattr(self.config, 'forecast_risk_management_mode', False):
                investor_strategy = getattr(self, '_current_investor_strategy', None)
                if investor_strategy is not None:
                    # Get current positions
                    wind_pos = self.financial_positions.get('wind_instrument_value', 0.0)
                    solar_pos = self.financial_positions.get('solar_instrument_value', 0.0)
                    hydro_pos = self.financial_positions.get('hydro_instrument_value', 0.0)
                    total_pos = abs(wind_pos) + abs(solar_pos) + abs(hydro_pos)
                    max_pos = self.config.max_position_size * self.init_budget * self.config.capital_allocation_fraction
                    position_ratio = float(np.clip(total_pos / max(max_pos, 1.0), 0.0, 1.0))
                    
                    strategy_direction = investor_strategy.get('direction', 0)
                    position_direction = np.sign(wind_pos + solar_pos + hydro_pos) if total_pos > 1e-6 else 0
                    
                    max_delta = max(abs(self._forecast_deltas_raw.get('short', 0.0)),
                                    abs(self._forecast_deltas_raw.get('medium', 0.0)),
                                    abs(self._forecast_deltas_raw.get('long', 0.0)))
                    
                    if investor_strategy['trade_signal'] and investor_strategy['consensus']:
                        if position_direction == strategy_direction and position_ratio > 0.1:
                            # FIXED: Increased reward for aligned positions to encourage position-taking
                            investor_strategy_delta = 0.10 * investor_strategy['signal_strength'] * position_ratio  # Increased from 0.05
                            position_alignment_status = 'aligned_strong'
                            if max_delta > 0.5:
                                investor_strategy_delta -= 0.025  # REDUCED: was 0.10
                                position_alignment_status = 'aligned_strong_extreme_delta'
                        elif position_ratio > 0.05:
                            # FIXED: Increased reward for smaller aligned positions
                            investor_strategy_delta = 0.05 * position_ratio  # Increased from 0.02
                            position_alignment_status = 'aligned_weak'
                            if max_delta > 0.5:
                                investor_strategy_delta -= 0.01  # REDUCED: was 0.05
                                position_alignment_status = 'aligned_weak_extreme_delta'
                        else:
                            # FIXED: Stronger penalty for not taking positions when signals are strong
                            signal_strength = investor_strategy.get('signal_strength', 0.5)
                            investor_strategy_delta = -0.05 * signal_strength  # Increased from -0.01
                            position_alignment_status = 'penalty_no_position'
                    elif investor_strategy['hedge_signal']:
                        if position_ratio > 0.05:
                            # FIXED: Increased reward for hedge positions
                            investor_strategy_delta = 0.05 * investor_strategy['signal_strength'] * position_ratio  # Increased from 0.025
                            position_alignment_status = 'hedge_position'
                        else:
                            # FIXED: Small penalty for not taking hedge positions
                            investor_strategy_delta = -0.02  # Changed from 0.0 to encourage positions
                            position_alignment_status = 'hedge_no_position'
                    else:
                        if position_ratio < 0.1:
                            investor_strategy_delta = 0.01  # REDUCED: was 0.05
                            position_alignment_status = 'neutral_correct'
                        else:
                            investor_strategy_delta = 0.0
                            position_alignment_status = 'neutral_wrong'
                    
                    # FIXED: Stronger exploration bonus to encourage position-taking
                    exploration_bonus = 0.0
                    episode_count = getattr(self, 'episode_count', 0)
                    # Extend exploration bonus to episode 15 (was 5) and increase magnitude
                    if episode_count < 15:
                        if total_pos > 1e-6:
                            # Increased from 0.01 to 0.05 to strongly encourage positions
                            exploration_bonus = 0.05 * position_ratio
                            investor_strategy_delta += exploration_bonus
                        elif investor_strategy and investor_strategy.get('trade_signal', False):
                            # NEW: Bonus for taking positions when signals are active (even if small)
                            signal_strength = investor_strategy.get('signal_strength', 0.5)
                            exploration_bonus = 0.03 * signal_strength  # Encourage taking positions
                            investor_strategy_delta += exploration_bonus
                    
                    self._last_investor_strategy = investor_strategy
                    if self.t % 100 == 0:
                        logger.info(f"[POSITION_ALIGNMENT] t={self.t} strategy={investor_strategy['strategy']} "
                                   f"pos_ratio={position_ratio:.3f} pos_dir={position_direction} strat_dir={strategy_direction} "
                                   f"delta={investor_strategy_delta:.4f} status={position_alignment_status}")
            
            forecast_usage_bonus = 0.0
            latent_follow_bonus = 0.0

            # FAIR COMPARISON: Do not add forecast-specific reward shaping when forecasts are
            # enabled purely as extra observations.
            if (
                debug_forecast.get('forecast_used')
                and self.forecast_risk_manager is not None
                and getattr(self.config, 'forecast_risk_management_mode', False)
            ):
                scaled_signal = float(np.clip(debug_forecast.get('forecast_signal_score', 0.0) / 50.0, -0.1, 0.1))
                investor_strategy_delta += scaled_signal
                if mtm_pnl > 0.0:
                    bonus_scale = getattr(self.config, 'forecast_usage_bonus_scale', 0.02)
                    bonus_norm = max(1.0, getattr(self.config, 'forecast_usage_bonus_mtm_scale', 15000.0))
                    forecast_usage_bonus = bonus_scale * float(np.clip(mtm_pnl / bonus_norm, 0.0, 1.0))
                    investor_strategy_delta += forecast_usage_bonus

            if debug_forecast and self.forecast_risk_manager is not None and getattr(self.config, 'forecast_risk_management_mode', False):
                if debug_forecast.get('agent_followed_forecast'):
                    exposure = float(np.clip(debug_forecast.get('position_exposure', 0.0), 0.0, 1.0))
                    follow_bonus_scale = getattr(self.config, 'forecast_follow_bonus_scale', 0.015)
                    if exposure < 0.12:
                        low_exposure_gap = float(np.clip(0.12 - exposure, 0.0, 0.12)) / 0.12
                        latent_follow_bonus = follow_bonus_scale * low_exposure_gap
                        investor_strategy_delta += latent_follow_bonus

                debug_forecast['forecast_usage_bonus'] = forecast_usage_bonus + latent_follow_bonus

            # === STRATEGY 4: NAV-based Reward Component ===
            # Reward NAV growth to encourage holding winning positions and overall portfolio health
            investor_nav_delta = 0.0
            try:
                # Get current NAV
                current_nav = float(financial.get('fund_nav', self.equity))
                
                # Track previous NAV (initialize on first call)
                if not hasattr(self, '_previous_nav'):
                    self._previous_nav = current_nav
                
                # Calculate NAV change reward
                if self._previous_nav > 1e-6:  # Avoid division by zero
                    nav_return = (current_nav - self._previous_nav) / self._previous_nav
                    
                    # Scale reward: 100x for percentage points (so 1% NAV increase = 1.0 reward)
                    # Clip to reasonable range to prevent extreme rewards
                    nav_reward = nav_return * 100.0
                    nav_reward = np.clip(nav_reward, -2.0, 2.0)  # Cap at ±2.0 reward
                    
                    # Weight: NAV reward is important but shouldn't dominate (20% weight)
                    investor_nav_delta = nav_reward * 0.20
                    
                    # Update previous NAV for next step
                    self._previous_nav = current_nav
                    
                    # Log periodically
                    if self.t % 500 == 0:
                        logger.debug(f"[NAV_REWARD] t={self.t}: nav_return={nav_return:.4%}, "
                                   f"nav_reward={nav_reward:.4f}, investor_nav_delta={investor_nav_delta:.4f}")
                else:
                    # First call or invalid NAV - initialize and skip reward
                    self._previous_nav = current_nav
            except Exception as e:
                logger.warning(f"[NAV_REWARD] Error calculating NAV reward: {e}")
                # Initialize previous_nav if not set
                if not hasattr(self, '_previous_nav'):
                    self._previous_nav = float(financial.get('fund_nav', self.equity))
            
            agent_rewards['investor_0'] = investor_base + investor_mtm_delta + investor_strategy_delta + investor_nav_delta
            
            # ===== BATTERY OPERATOR AGENT =====
            # FIXED: Improved battery reward calculation with stronger PnL signal
            battery_cash = float(financial.get('battery_cash_delta', 0.0))
            # Increased multiplier from 0.05 to 0.20 (4x stronger signal)
            # Reduced divisor from 2000 to 1000 (less aggressive clipping)
            # Increased clip range from [-1, 1] to [-2, 2] (allow stronger signals)
            battery_arbitrage_delta = float(0.20 * np.clip(battery_cash / 1000.0, -2.0, 2.0))
            
            # NEW: Add direct PnL component for better alignment
            pnl_battery = float(getattr(self, '_last_battery_cash_delta', 0.0))
            battery_pnl_delta = float(0.10 * np.clip(pnl_battery / 500.0, -1.0, 1.0))
            
            # NEW: SOC-aware penalty for invalid actions (discharge at min SOC, charge at max SOC)
            soc_penalty = 0.0
            battery_soc = self.operational_state.get('battery_energy', 0.0) / max(self.physical_assets.get('battery_capacity_mwh', 1.0), 1.0)
            battery_soc = float(np.clip(battery_soc, 0.0, 1.0))
            
            # Get last action to check if invalid command was issued
            last_actions = getattr(self, '_last_actions', {})
            battery_action = last_actions.get('battery_operator_0', np.array([0.0]))
            try:
                if hasattr(battery_action, 'reshape'):
                    u_raw = float(battery_action.reshape(-1)[0])
                elif isinstance(battery_action, (list, np.ndarray)):
                    u_raw = float(battery_action[0] if len(battery_action) > 0 else 0.0)
                else:
                    u_raw = float(battery_action)
            except Exception:
                u_raw = 0.0
            
            # CRITICAL FIX: Penalty for trying to discharge when at minimum SOC
            if u_raw > 0.2 and battery_soc <= self.batt_soc_min + 1e-6:
                soc_penalty = -0.5  # Strong penalty for discharge command at min SOC
            elif u_raw < -0.2 and battery_soc >= self.batt_soc_max - 1e-6:
                soc_penalty = -0.2  # Moderate penalty for charge command at max SOC
            
            battery_forecast_delta = 0.0
            if self.forecast_risk_manager is not None and getattr(self.config, 'forecast_risk_management_mode', False):
                z_short_gen = float(getattr(self, '_z_short_total_gen', 0.0))
                z_short_price = float(getattr(self, '_z_short_price', 0.0))
                soc = self.operational_state.get('battery_energy', 0.0) / max(self.physical_assets.get('battery_capacity_mwh', 1.0), 1.0)
                
                if z_short_gen > 0.5 and z_short_price < -0.5 and soc < 0.7:
                    battery_forecast_delta = 0.025  # REDUCED: was 0.10
                elif z_short_gen < -0.5 and z_short_price > 0.5 and soc > 0.3:
                    battery_forecast_delta = 0.025  # REDUCED: was 0.10
                elif abs(z_short_gen) < 0.3 and abs(z_short_price) < 0.3:
                    battery_forecast_delta = 0.01  # REDUCED: was 0.05
            
            agent_rewards['battery_operator_0'] = battery_base + battery_arbitrage_delta + battery_pnl_delta + battery_forecast_delta + soc_penalty
            
            # ===== RISK CONTROLLER AGENT =====
            volatility_penalty = float(np.clip(risk_level * 10.0, 0.0, 2.0))
            drawdown = float(getattr(self.reward_calculator, 'current_drawdown', 0.0)) if self.reward_calculator else 0.0
            drawdown_penalty = float(np.clip(drawdown * 10.0, 0.0, 2.0))
            risk_management_delta = float(-0.05 * (volatility_penalty + drawdown_penalty))  # REDUCED: was -0.20
            
            agent_rewards['risk_controller_0'] = risk_base + risk_management_delta
            
            # ===== META CONTROLLER AGENT =====
            # Coordination bonus when all agents performing above baseline
            coordination_delta = 0.0
            if float(investor_mtm_delta) > 0 and float(battery_arbitrage_delta) > 0 and float(risk_management_delta) > -0.025:
                coordination_delta = 0.05  # REDUCED: was 0.20
            elif float(investor_mtm_delta) < -0.025 or float(battery_arbitrage_delta) < -0.025 or float(risk_management_delta) < -0.075:
                coordination_delta = -0.05  # REDUCED: was -0.20
            
            agent_rewards['meta_controller_0'] = meta_base + coordination_delta

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
                'investor_strategy_delta': investor_strategy_delta,
                'battery_arbitrage_delta': battery_arbitrage_delta,
                'battery_forecast_delta': battery_forecast_delta,
                'risk_management_delta': risk_management_delta,
                'coordination_delta': coordination_delta,
                'forecast_usage_bonus': float(debug_forecast.get('forecast_usage_bonus', 0.0)) if debug_forecast else 0.0,
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
                
                # Get actions from last step (stored in _last_actions)
                last_actions = getattr(self, '_last_actions', {})
                investor_action = last_actions.get('investor_0', {})
                battery_action = last_actions.get('battery_operator_0', {})
                
                # Calculate current portfolio metrics (logged at every timestep)
                dkk_to_usd = 0.145  # Conversion rate
                # CRITICAL FIX: At timestep 0, use the NAV from reset (before any trades)
                # This ensures identical starting NAV values between Tier 1 and Tier 2
                # After timestep 0, use the NAV from _update_finance (which includes trades)
                # NOTE: self.t is still 0 here because it's incremented AFTER _assign_rewards is called
                if int(self.t) == 0 and hasattr(self, '_initial_nav_from_reset'):
                    current_fund_nav_dkk = self._initial_nav_from_reset
                    tier_name = "TIER2" if getattr(self.config, 'enable_forecast_utilisation', False) else "TIER1"
                    logger.info(f"[{tier_name}_NAV_LOG_FIX] Using initial NAV from reset at t=0: {current_fund_nav_dkk:,.0f} DKK (${current_fund_nav_dkk * 0.145 / 1_000_000:.2f}M)")
                else:
                    # CRITICAL FIX: Use fund_nav from _update_finance (already calculated with correct timestep i)
                    # _update_finance sets self.t = i before calculating NAV, ensuring correct depreciation
                    current_fund_nav_dkk = fund_nav
                current_fund_nav_usd = current_fund_nav_dkk * dkk_to_usd / 1_000_000  # Convert to millions USD
                current_cash_dkk = getattr(self, 'budget', 0.0)
                current_trading_gains = getattr(self, 'cumulative_mtm_pnl', 0.0)
                current_trading_gains_usd = current_trading_gains * dkk_to_usd / 1_000  # Convert to thousands USD
                current_operating_gains = getattr(self, 'cumulative_generation_revenue', 0.0)
                current_operating_gains_usd = current_operating_gains * dkk_to_usd / 1_000  # Convert to thousands USD

                # Fund NAV component breakdown (DKK) for drop attribution
                # Mirror FinancialEngine.calculate_fund_nav component math so we can explain jumps.
                try:
                    current_timestep = int(self.t)
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
                    financial_mtm_dkk = float(sum(getattr(self, 'financial_mtm_positions', getattr(self, 'financial_positions', {})).values()))
                    # Exposure (not equity): logged separately for interpretability
                    financial_exposure_dkk = float(sum(abs(float(v)) for v in getattr(self, 'financial_positions', {}).values()))
                    trading_cash_dkk = float(max(0.0, float(getattr(self, 'budget', 0.0))))
                    fund_nav_dkk = float(current_fund_nav_dkk)
                except Exception:
                    years_elapsed = 0.0
                    depreciation_ratio = 0.0
                    physical_book_value_dkk = 0.0
                    accumulated_operational_revenue_dkk = 0.0
                    financial_mtm_dkk = 0.0
                    financial_exposure_dkk = 0.0
                    trading_cash_dkk = float(max(0.0, float(getattr(self, 'budget', 0.0))))
                    fund_nav_dkk = float(current_fund_nav_dkk)
                
                # Get current price for forward-looking accuracy analysis
                current_timestep = int(self.t)
                current_price_raw = float(self._price_raw[current_timestep] if current_timestep < len(self._price_raw) else 0.0)

                # PRINT to verify we're about to call log_step
                if self.t % 100 == 0:
                    logger.debug(f"[DEBUG_TRACKER] t={self.t} CALLING log_step() NOW!")

                # Always define a fallback reason if forecasts were gated
                forecast_usage_reason = 'unknown'
                if debug_forecast:
                    forecast_usage_reason = str(debug_forecast.get('forecast_not_used_reason', forecast_usage_reason))

                # Ensure all values are floats (not arrays) to avoid "ambiguous truth value" error
                self.debug_tracker.log_step(
                    timestep=int(self.t),
                    # Portfolio metrics (logged at every timestep)
                    portfolio_value_usd_millions=float(current_fund_nav_usd),
                    cash_dkk=float(current_cash_dkk),
                    trading_gains_usd_thousands=float(current_trading_gains_usd),
                    operating_gains_usd_thousands=float(current_operating_gains_usd),
                    # Fund NAV component breakdown (DKK)
                    fund_nav_dkk=float(fund_nav_dkk),
                    trading_cash_dkk=float(trading_cash_dkk),
                    physical_book_value_dkk=float(physical_book_value_dkk),
                    accumulated_operational_revenue_dkk=float(accumulated_operational_revenue_dkk),
                    financial_mtm_dkk=float(financial_mtm_dkk),
                    financial_exposure_dkk=float(financial_exposure_dkk),
                    depreciation_ratio=float(depreciation_ratio),
                    years_elapsed=float(years_elapsed),
                    # Price data (for forward-looking accuracy analysis)
                    price_current=float(current_price_raw),  # Raw price in DKK/MWh
                    # Forecast signals (ensure floats)
                    z_short=float(debug_forecast.get('z_short', 0.0)),
                    z_medium=float(debug_forecast.get('z_medium', 0.0)),
                    z_long=float(debug_forecast.get('z_long', 0.0)),
                    z_combined=float(debug_forecast.get('z_combined', 0.0)),
                    forecast_confidence=float(debug_forecast.get('forecast_confidence', 0.0)),
                    forecast_trust=float(debug_forecast.get('forecast_trust', 0.0)),
                    signal_gate_multiplier=float(debug_forecast.get('signal_gate_multiplier', 0.0)),
                    # Position info (ensure floats)
                    position_signed=float(debug_forecast.get('position_signed', 0.0)),
                    position_exposure=float(debug_forecast.get('position_exposure', 0.0)),
                    # Price returns (ensure floats)
                    price_return_1step=float(debug_forecast.get('price_return_1step', 0.0)),
                    price_return_forecast=float(debug_forecast.get('price_return_forecast', 0.0)),
                    # Forecast reward components (ensure floats)
                    alignment_reward=float(debug_forecast.get('alignment_reward', 0.0)),
                    pnl_reward=float(debug_forecast.get('pnl_reward', 0.0)),
                    forecast_signal_score=float(debug_forecast.get('forecast_signal_score', 0.0)),
                    generation_forecast_score=float(debug_forecast.get('generation_forecast_score', 0.0)),
                    combined_forecast_score=float(debug_forecast.get('combined_forecast_score', 0.0)),
                    trust_scale=float(debug_forecast.get('trust_scale', 1.0)),
                    warmup_factor=float(debug_forecast.get('warmup_factor', 1.0)),
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
                    forecast_direction=float(debug_forecast.get('forecast_direction', 0.0)),
                    position_direction=float(debug_forecast.get('position_direction', 0.0)),
                    is_aligned=float(debug_forecast.get('agent_followed_forecast', 0.0)),
                    alignment_status=str('ALIGNED' if debug_forecast.get('agent_followed_forecast', False) else 'MISALIGNED'),
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
                    wind_pos_norm=float(debug_forecast.get('wind_pos_norm', 0.0)),
                    solar_pos_norm=float(debug_forecast.get('solar_pos_norm', 0.0)),
                    hydro_pos_norm=float(debug_forecast.get('hydro_pos_norm', 0.0)),
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
                    enable_forecast_util=bool(getattr(self.config, 'enable_forecast_utilisation', False)),
                    forecast_gate_passed=forecast_gate_flag,
                    forecast_used=forecast_used_flag,
                    forecast_not_used_reason=str(debug_forecast.get('forecast_not_used_reason', forecast_usage_reason)),
                    # NOVEL: Investor strategy fields
                    investor_strategy=str(getattr(self, '_last_investor_strategy', {}).get('strategy', 'none')),
                    investor_position_scale=float(getattr(self, '_last_investor_strategy', {}).get('position_scale', 0.0)),
                    investor_signal_strength=float(getattr(self, '_last_investor_strategy', {}).get('signal_strength', 0.0)),
                    investor_consensus=bool(getattr(self, '_last_investor_strategy', {}).get('consensus', False)),
                    investor_direction=int(getattr(self, '_last_investor_strategy', {}).get('direction', 0)),
                    investor_strategy_bonus=float(investor_strategy_delta),
                    investor_strategy_multiplier=float(debug_forecast.get('investor_strategy_multiplier', getattr(self, '_last_investor_strategy_multiplier', 1.0))),
                    forecast_usage_bonus=float(debug_forecast.get('forecast_usage_bonus', 0.0)),
                    # NEW: Position alignment diagnostics (Bug #1 fix tracking)
                    position_alignment_status=str(position_alignment_status) if 'position_alignment_status' in locals() else 'no_strategy',
                    investor_position_ratio=float(position_ratio) if 'position_ratio' in locals() else 0.0,
                    investor_position_direction=int(position_direction) if 'position_direction' in locals() else 0,
                    investor_total_position=float(total_pos) if 'total_pos' in locals() else 0.0,
                    investor_exploration_bonus=float(exploration_bonus) if 'exploration_bonus' in locals() else 0.0,
                    # NEW: Per-horizon MAPE tracking
                    mape_short=float(np.mean(list(self._horizon_mape.get('short', [0.0]))[-100:]) if len(self._horizon_mape.get('short', [])) > 0 else 0.0),
                    mape_medium=float(np.mean(list(self._horizon_mape.get('medium', [0.0]))[-100:]) if len(self._horizon_mape.get('medium', [])) > 0 else 0.0),
                    mape_long=float(np.mean(list(self._horizon_mape.get('long', [0.0]))[-100:]) if len(self._horizon_mape.get('long', [])) > 0 else 0.0),
                    mape_short_recent=float(np.mean(list(self._horizon_mape.get('short', [0.0]))[-10:]) if len(self._horizon_mape.get('short', [])) > 0 else 0.0),
                    mape_medium_recent=float(np.mean(list(self._horizon_mape.get('medium', [0.0]))[-10:]) if len(self._horizon_mape.get('medium', [])) > 0 else 0.0),
                    mape_long_recent=float(np.mean(list(self._horizon_mape.get('long', [0.0]))[-10:]) if len(self._horizon_mape.get('long', [])) > 0 else 0.0),
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
                    # NEW: Directional accuracy
                    direction_accuracy_short=float(debug_forecast.get('direction_accuracy_short', 0.0)),
                    direction_accuracy_medium=float(debug_forecast.get('direction_accuracy_medium', 0.0)),
                    direction_accuracy_long=float(debug_forecast.get('direction_accuracy_long', 0.0)),
                    # NEW: Battery forecast bonus
                    battery_forecast_bonus=float(battery_forecast_delta),
                    # NEW: Forecast vs actual comparison (deep debugging)
                    current_price_dkk=float(debug_forecast.get('current_price_dkk', 0.0)),
                    forecast_price_short_dkk=float(debug_forecast.get('forecast_price_short_dkk', 0.0)),
                    forecast_price_medium_dkk=float(debug_forecast.get('forecast_price_medium_dkk', 0.0)),
                    forecast_price_long_dkk=float(debug_forecast.get('forecast_price_long_dkk', 0.0)),
                    forecast_error_short_pct=float(debug_forecast.get('forecast_error_short_pct', 0.0)),
                    forecast_error_medium_pct=float(debug_forecast.get('forecast_error_medium_pct', 0.0)),
                    forecast_error_long_pct=float(debug_forecast.get('forecast_error_long_pct', 0.0)),
                    agent_followed_forecast=bool(debug_forecast.get('agent_followed_forecast', False)),
                    # NEW: NAV attribution drivers (per-step financial breakdown)
                    nav_start=float(getattr(self, 'nav_start', getattr(self, 'init_budget', 0.0))),
                    nav_end=float(getattr(self, 'nav_end', getattr(self, 'budget', 0.0))),
                    pnl_total=float(getattr(self, '_last_total_pnl', 0.0)),
                    pnl_battery=float(getattr(self, '_last_battery_cash_delta', 0.0)),
                    pnl_generation=float(getattr(self, '_last_generation_pnl', 0.0)),
                    pnl_hedge=float(getattr(self, '_last_hedge_pnl', 0.0)),
                    cash_delta_ops=float(getattr(self, '_last_cash_ops', 0.0)),
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
                    # Note: fund_nav, trading_gains, operating_gains removed - duplicates of checkpoint summary fields
                    # fund_nav = portfolio_value_usd_millions (just different units)
                    # trading_gains = trading_gains_usd_thousands (just different units)
                    # operating_gains = operating_gains_usd_thousands (just different units)
                    mtm_pnl=float(mtm_pnl),  # Per-step MTM PnL (not cumulative, different from trading_gains)
                    # Actions
                    investor_action=investor_action,
                    battery_action=battery_action,
                )
            else:
                # FALLBACK: Log minimal data if reward_calculator is None
                if hasattr(self, 'debug_tracker'):
                    logger.warning(f"[DEBUG_TRACKER] t={self.t} reward_calculator is NONE! Logging minimal data...")
                    debug_forecast = getattr(self, '_debug_forecast_reward', {})
                    forecast_gate_flag = bool(debug_forecast.get('forecast_gate_passed', False))
                    forecast_used_flag = bool(debug_forecast.get('forecast_used', False))

                    # Calculate current portfolio metrics
                    dkk_to_usd = 0.145
                    current_fund_nav_dkk = fund_nav
                    current_fund_nav_usd = current_fund_nav_dkk * dkk_to_usd / 1_000_000
                    current_cash_dkk = getattr(self, 'budget', 0.0)
                    current_price_raw = float(self._price_raw[self.t] if self.t < len(self._price_raw) else 0.0)

                    self.debug_tracker.log_step(
                        timestep=int(self.t),
                        portfolio_value_usd_millions=float(current_fund_nav_usd),
                        cash_dkk=float(current_cash_dkk),
                        price_current=float(current_price_raw),
                        z_short=float(debug_forecast.get('z_short', 0.0)),
                        z_medium=float(debug_forecast.get('z_medium', 0.0)),
                        z_long=float(debug_forecast.get('z_long', 0.0)),
                        forecast_trust=float(debug_forecast.get('forecast_trust', 0.0)),
                        position_signed=float(debug_forecast.get('position_signed', 0.0)),
                        position_exposure=float(debug_forecast.get('position_exposure', 0.0)),
                        alignment_reward=float(debug_forecast.get('alignment_reward', 0.0)),
                        pnl_reward=float(debug_forecast.get('pnl_reward', 0.0)),
                        forecast_signal_score=float(debug_forecast.get('forecast_signal_score', 0.0)),
                        base_reward=float(base_reward),
                        enable_forecast_util=bool(getattr(self.config, 'enable_forecast_utilisation', False)),
                        forecast_gate_passed=forecast_gate_flag,
                        forecast_used=forecast_used_flag,
                    )

            # P&L-BASED TRAINING: Track P&L for action attribution
            if hasattr(self.reward_calculator, 'recent_trading_gains'):
                trading_pnl = float(self.reward_calculator.recent_trading_gains)
            else:
                trading_pnl = 0.0

            # Initialize _last_nav on first step
            if self._last_nav is None:
                self._last_nav = fund_nav
                nav_change = 0.0
            else:
                nav_change = fund_nav - self._last_nav
                self._last_nav = fund_nav

            self._pnl_history.append({
                'timestep': self.t,
                'trading_pnl': trading_pnl,
                'nav_change': nav_change,
                'nav': fund_nav,
                'reward': reward
            })

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
                env_state = {
                    'price': float(self._price_raw[i]) if hasattr(self, '_price_raw') and i < len(self._price_raw)
                             else float(self._price[i] * self._price_std[i] + self._price_mean[i]) if hasattr(self, '_price_std') and hasattr(self, '_price_mean') and i < len(self._price_std) and i < len(self._price_mean)
                             else float(self._price[i]),
                    'budget': self.budget,
                    'initial_budget': self.init_budget,
                    'timestep': self.t,
                    'wind_capacity': self.physical_assets['wind_capacity_mw'],
                    'solar_capacity': self.physical_assets['solar_capacity_mw'],
                    'hydro_capacity': self.physical_assets['hydro_capacity_mw'],
                    'battery_capacity': self.physical_assets['battery_capacity_mwh'],
                    'wind': float(self._wind[i]) if i < len(self._wind) else 0.0,
                    'solar': float(self._solar[i]) if i < len(self._solar) else 0.0,
                    'hydro': float(self._hydro[i]) if i < len(self._hydro) else 0.0,
                    'revenue': self.last_revenue,
                    'market_stress': self.market_stress,
                }
                self.enhanced_risk_controller.update_risk_history(env_state)
                comp = self.enhanced_risk_controller.calculate_comprehensive_risk(env_state)

                self.overall_risk_snapshot = float(np.clip(comp.get('overall_risk', 0.5), 0.0, 1.0))
                self.market_risk_snapshot = float(np.clip(comp.get('market_risk', 0.3), 0.0, 1.0))
                self.portfolio_risk_snapshot = float(np.clip(comp.get('portfolio_risk', 0.25), 0.0, 1.0))
                self.liquidity_risk_snapshot = float(np.clip(comp.get('liquidity_risk', 0.15), 0.0, 1.0))
                return

        except Exception as e:
            logger.warning(f"Risk snapshot update failed: {e}")
            self.overall_risk_snapshot = 0.5
            self.market_risk_snapshot = 0.3
            self.portfolio_risk_snapshot = 0.25
            self.liquidity_risk_snapshot = 0.15

    # ------------------------------------------------------------------
    # Observations (BASE only)
    # ------------------------------------------------------------------
    def _fill_obs(self):
        """
        REFACTORED: Fill observations using ObservationBuilder for investor and battery agents.
        Risk and meta controller observations remain in this method due to complexity.
        """
        try:
            from observation_builder import ObservationBuilder
            
            i = min(self.t, self.max_steps - 1)
            
            # REFACTORED: Use ObservationBuilder for market feature normalization
            market_features = ObservationBuilder.normalize_market_features(
                price=self._price[i],
                load=self._load[i],
                wind=self._wind[i],
                solar=self._solar[i],
                hydro=self._hydro[i],
                load_scale=self.load_scale,
                wind_scale=self.wind_scale,
                solar_scale=self.solar_scale,
                hydro_scale=self.hydro_scale
            )
            price_n = market_features['price_n']
            load_n = market_features['load_n']
            
            enable_forecast_util = getattr(self.config, 'enable_forecast_utilisation', False)

            # IMPORTANT: In this environment, the canonical forecast z-scores/trust are maintained on the env
            # (e.g., self.z_short_price, self._forecast_trust) by the forecast-delta pipeline.
            # ForecastEngine's internal z_* fields are not guaranteed to be updated in lockstep.
            # To ensure Tier2 agents actually SEE the same forecast signals that rewards/debugging use,
            # we source observation forecast values from the env state.
            z_short_price_obs = float(getattr(self, 'z_short_price', 0.0))
            z_medium_price_obs = float(getattr(self, 'z_medium_price', 0.0))
            z_long_price_obs = float(getattr(self, 'z_long_price', 0.0))
            z_short_wind_obs = float(getattr(self, 'z_short_wind', 0.0))
            z_short_solar_obs = float(getattr(self, 'z_short_solar', 0.0))
            z_short_hydro_obs = float(getattr(self, 'z_short_hydro', 0.0))
            forecast_trust_obs = float(getattr(self, '_forecast_trust', getattr(self.forecast_engine, 'forecast_trust', 0.5)))

            # Compute normalized forecast error consistently with ForecastEngine.build_observation_features
            # CRITICAL FIX: Clip to [0.0, 1.0] to match observation space bounds and forecast_engine.py
            normalized_error_obs = 0.0
            try:
                mape_history = getattr(self, '_horizon_mape', {}).get('short', [])
                if len(mape_history) > 0:
                    recent_mape = float(np.mean(list(mape_history)[-10:]))
                    mape_threshold_short = float(getattr(self, '_mape_thresholds', {}).get('short', 0.02))
                    normalized_error_obs = float(np.clip(recent_mape / max(mape_threshold_short, 1e-6), 0.0, 1.0))
            except Exception:
                normalized_error_obs = 0.0
            
            # TEMPORAL ALIGNMENT: Get lagged z_medium from t-24 to align with return at t+24
            # The forecast made 24 steps ago predicted what's happening now
            z_medium_lagged = z_medium_price_obs  # Default to current if no lag available
            horizon_medium = self.config.forecast_horizons.get('medium', 24)
            if i >= horizon_medium:
                lag_timestep = i - horizon_medium
                if hasattr(self, '_z_score_history') and lag_timestep in self._z_score_history:
                    z_medium_lagged = float(self._z_score_history[lag_timestep].get('z_medium', z_medium_lagged))
            
            # ENGINEERED FEATURES: Direction, Momentum, Strength
            # Direction: sign of z_medium_lagged (-1 for down, +1 for up, 0 for neutral)
            direction = float(np.sign(z_medium_lagged))
            
            # Momentum: change in z_medium (current - previous)
            if not hasattr(self, '_z_medium_prev_obs'):
                self._z_medium_prev_obs = z_medium_lagged
            momentum = float(np.clip(z_medium_lagged - self._z_medium_prev_obs, -1.0, 1.0))
            self._z_medium_prev_obs = z_medium_lagged  # Store for next iteration
            
            # Strength: absolute value of z_medium_lagged (magnitude of signal)
            strength = float(np.clip(abs(z_medium_lagged), 0.0, 1.0))
            
            # TRADE SIGNAL: Composite feature combining direction, strength, and trust
            # Range: [-1, 1] where positive = buy signal, negative = sell signal, magnitude = confidence
            trade_signal = float(np.clip(direction * strength * forecast_trust_obs, -1.0, 1.0)) if enable_forecast_util else None
            
            # REFACTORED: Use ObservationBuilder for investor observations
            inv = self._obs_buf['investor_0']
            
            ObservationBuilder.build_investor_observations(
                obs_array=inv,
                price_n=price_n,
                budget=self.budget,
                init_budget=self.init_budget,
                financial_positions=self.financial_positions,
                cumulative_mtm_pnl=getattr(self, 'cumulative_mtm_pnl', 0.0),
                max_position_size=self.config.max_position_size,
                capital_allocation_fraction=self.config.capital_allocation_fraction,
                enable_forecast_util=enable_forecast_util,
                z_short_price=z_short_price_obs if enable_forecast_util else None,
                z_medium_lagged=z_medium_lagged if enable_forecast_util else None,
                direction=direction if enable_forecast_util else None,
                momentum=momentum if enable_forecast_util else None,
                strength=strength if enable_forecast_util else None,
                forecast_trust=forecast_trust_obs if enable_forecast_util else None,
                normalized_error=normalized_error_obs if enable_forecast_util else None,
                trade_signal=trade_signal if enable_forecast_util else None,
            )
            
            # DIAGNOSTIC: Log investor observations periodically
            if self.t % 1000 == 0 or self.t == 0:
                if enable_forecast_util:
                    logger.info(f"[INVESTOR_OBS_TIER2] t={self.t} | TOTAL_DIM={len(inv)}")
                    logger.info(f"  BASE (6D): price={inv[0]:.3f}, budget={inv[1]:.3f}, wind_pos={inv[2]:.3f}, solar_pos={inv[3]:.3f}, hydro_pos={inv[4]:.3f}, mtm_pnl={inv[5]:.3f}")
                    if len(inv) >= 14:
                        logger.info(f"  FORECAST (8D): z_short={inv[6]:.3f}, z_medium_lagged={inv[7]:.3f}, dir={inv[8]:.3f}, momentum={inv[9]:.3f}, strength={inv[10]:.3f}, trust={inv[11]:.3f}, err={inv[12]:.3f}, trade_signal={inv[13]:.3f}")
                    elif len(inv) >= 8:
                        logger.info(f"  FORECAST (2D): trade_signal={inv[6]:.3f}, normalized_error={inv[7]:.3f}")
                    else:
                        logger.info(f"  FORECAST: unexpected dimension {len(inv)}")
                else:
                    logger.info(f"[INVESTOR_OBS] t={self.t} | price={inv[0]:.3f}, budget={inv[1]:.3f}")
                    logger.info(f"  Positions: wind={inv[2]:.3f}, solar={inv[3]:.3f}, hydro={inv[4]:.3f}, mtm_pnl={inv[5]:.3f}")

            # REFACTORED: Use ObservationBuilder for battery observations
            batt = self._obs_buf['battery_operator_0']
            
            ObservationBuilder.build_battery_observations(
                obs_array=batt,
                price_n=price_n,
                battery_energy=self.operational_state['battery_energy'],
                battery_capacity_mwh=self.physical_assets['battery_capacity_mwh'],
                load_n=load_n,
                enable_forecast_util=enable_forecast_util,
                z_short_wind=z_short_wind_obs if enable_forecast_util else None,
                z_short_solar=z_short_solar_obs if enable_forecast_util else None,
                z_short_hydro=z_short_hydro_obs if enable_forecast_util else None,
                z_short_price=z_short_price_obs if enable_forecast_util else None,
                z_medium_price=z_medium_price_obs if enable_forecast_util else None,
                z_long_price=z_long_price_obs if enable_forecast_util else None,
            )
            
            # DIAGNOSTIC: Log battery observations periodically
            if self.t % 1000 == 0 or self.t == 0:
                if enable_forecast_util:
                    if len(batt) >= 10:
                        logger.info(f"[BATTERY_OBS] t={self.t} | price={batt[0]:.3f}, energy={batt[1]:.3f}, capacity={batt[2]:.3f}, load={batt[3]:.3f}")
                        logger.info(f"  Generation forecasts: wind={batt[4]:.3f}, solar={batt[5]:.3f}, hydro={batt[6]:.3f}")
                        logger.info(f"  Price forecasts: short={batt[7]:.3f}, med={batt[8]:.3f}, long={batt[9]:.3f}")
                    else:
                        # Backward compatibility: 8D format
                        logger.info(f"[BATTERY_OBS] t={self.t} | price={batt[0]:.3f}, energy={batt[1]:.3f}, capacity={batt[2]:.3f}, load={batt[3]:.3f}")
                        logger.info(f"  Total generation (short): {batt[4]:.3f}")
                        logger.info(f"  Price forecasts: short={batt[5]:.3f}, med={batt[6]:.3f}, long={batt[7]:.3f}")
                else:
                    logger.info(f"[BATTERY_OBS] t={self.t} | price={batt[0]:.3f}, energy={batt[1]:.3f}, capacity={batt[2]:.3f}, load={batt[3]:.3f}")

            # PHASE 3: RISK CONTROLLER OBSERVATIONS
            rsk = self._obs_buf['risk_controller_0']

            # Base observations (9D)
            rsk[:9] = (
            price_n,
            float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
            float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.financial_positions['wind_instrument_value'],  self.init_budget), 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.financial_positions['solar_instrument_value'], self.init_budget), 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.financial_positions['hydro_instrument_value'], self.init_budget), 0.0, 1.0)) * 10.0,
            float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0)),
            float(np.clip(self.risk_multiplier, 0.0, 2.0)) * 5.0,
            )

            # CONDITIONAL: Add forecast signals if forecast utilisation is enabled
            if enable_forecast_util:
                # TIER 2: Add forecast signals (3D) → Total 12D
                # Price trend for regime detection
                rsk[9] = z_short_price_obs

                # Expected volatility from forecast spread (higher spread = higher uncertainty)
                z_short = z_short_price_obs
                z_medium = z_medium_price_obs
                z_long = z_long_price_obs
                forecast_spread = abs(z_short - z_medium) + abs(z_medium - z_long) + abs(z_short - z_long)
                price_volatility_forecast = float(np.clip(forecast_spread / 3.0, 0.0, 1.0)) * 10.0
                rsk[10] = price_volatility_forecast

                # Forecast trust (quality indicator)
                rsk[11] = forecast_trust_obs

                # DIAGNOSTIC: Log risk controller observations periodically
                if self.t % 1000 == 0 or self.t == 0:
                    logger.info(f"[RISK_OBS] t={self.t} | price={rsk[0]:.3f}, vol={rsk[1]:.3f}, stress={rsk[2]:.3f}")
                    logger.info(f"  Forecasts: z_short={rsk[9]:.3f}, vol_forecast={rsk[10]:.3f}, trust={rsk[11]:.3f}")

            # PHASE 4: META CONTROLLER OBSERVATIONS
            perf_ratio = float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0))
            meta = self._obs_buf['meta_controller_0']

            # Base observations (11D)
            meta[:11] = (
            float(np.clip(SafeDivision.div(self.budget, self.init_budget/10.0, 0.0), 0.0, 10.0)),
            float(np.clip(SafeDivision.div(self.financial_positions['wind_instrument_value'],  self.init_budget, 0.0), 0.0, 10.0)),
            float(np.clip(SafeDivision.div(self.financial_positions['solar_instrument_value'], self.init_budget, 0.0), 0.0, 10.0)),
            float(np.clip(SafeDivision.div(self.financial_positions['hydro_instrument_value'], self.init_budget, 0.0), 0.0, 10.0)),
            price_n,
            float(np.clip(self.overall_risk_snapshot, 0.0, 1.0)) * 10.0,
            perf_ratio,
            float(np.clip(self.market_risk_snapshot, 0.0, 1.0)) * 10.0,
            float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
            float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
            float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
            )

            # CONDITIONAL: Add forecast signals if forecast utilisation is enabled
            if enable_forecast_util:
                # TIER 2: Add forecast signals (2D) → Total 13D
                # Forecast trust (overall quality)
                meta[11] = forecast_trust_obs

                # Expected return from forecasts (weighted combination of price forecasts)
                z_short = z_short_price_obs
                z_medium = z_medium_price_obs
                z_long = z_long_price_obs
                trust = forecast_trust_obs

                # Weighted expected return (trust-scaled)
                expected_return = (0.7 * z_short + 0.2 * z_medium + 0.1 * z_long) * trust
                meta[12] = float(np.clip(expected_return, -1.0, 1.0))

                # DIAGNOSTIC: Log meta controller observations periodically
                if self.t % 1000 == 0 or self.t == 0:
                    logger.info(f"[META_OBS] t={self.t} | budget={meta[0]:.3f}, perf={meta[6]:.3f}")
                    logger.info(f"  Forecasts: trust={meta[11]:.3f}, expected_return={meta[12]:.3f}")

        except Exception as e:
            logger.error(f"Error building observations: {e}")
            # Fallback: set default observations to prevent crashes
            for agent in self.agents:
                if agent in self._obs_buf:
                    self._obs_buf[agent].fill(0.0)

    # ------------------------------------------------------------------
    # FGB: Forecast-Guided Baseline Signals
    # ------------------------------------------------------------------
    def _compute_forecast_signals(self):
        """
        FGB: Compute forecast trust (τₜ) and expected ΔNAV for baseline adjustment.

        These signals are used by metacontroller to adjust PPO rewards:
            r'_t = r_t - λ · τₜ · E[ΔNAV | s_t]

        This reduces advantage variance and improves convergence without blending actions.
        """
        try:
            # Initialize to zero (no-op if overlay disabled)
            self._forecast_trust = 0.0
            self._expected_dnav = 0.0

            # Only compute if calibration tracker is available
            if self.calibration_tracker is None:
                return

            # Get trust from calibration tracker
            self._forecast_trust = float(self.calibration_tracker.get_trust(horizon="short"))

            # Compute expected ΔNAV if we have overlay output
            if self._last_overlay_output:
                self._expected_dnav = float(self.calibration_tracker.expected_dnav(
                    overlay_output=self._last_overlay_output,
                    positions=self.financial_positions,
                    costs={
                        'bps': self.config.transaction_cost_bps,
                        'fixed': self.config.transaction_fixed_cost
                    }
                ))

            # Optional: Apply risk uplift based on trust (modulate position sizing)
            # INDEPENDENT: Works with or without DL overlay - uses forecast deltas directly
            if getattr(self.config, 'risk_uplift_enable', False) and self._forecast_trust >= getattr(self.config, 'forecast_trust_min', 0.6):
                try:
                    # Check drawdown gate
                    dd_gate = getattr(self.config, 'risk_uplift_drawdown_gate', 0.07)
                    if self.current_drawdown >= dd_gate:
                        # Drawdown too high, disable uplift
                        if self.t % 1000 == 0:
                            logger.info(f"[RISK_UPLIFT] DISABLED by drawdown gate: DD={self.current_drawdown:.4f} >= {dd_gate:.4f}")
                        return

                    # Check volatility gate (optional)
                    vol_gate = getattr(self.config, 'risk_uplift_vol_gate', 0.02)
                    if vol_gate > 0 and self.market_volatility >= vol_gate:
                        # Volatility too high, disable uplift
                        if self.t % 1000 == 0:
                            logger.info(f"[RISK_UPLIFT] DISABLED by volatility gate: vol={self.market_volatility:.4f} >= {vol_gate:.4f}")
                        return

                    # Compute uplift multiplier
                    # κ_uplift: max 15% sizing uplift
                    # Scale by mwdir confidence (forecast directional signal)
                    kappa = getattr(self.config, 'risk_uplift_kappa', 0.15)
                    cap = getattr(self.config, 'risk_uplift_cap', 1.15)

                    # INDEPENDENT: Use forecast deltas directly (not overlay output)
                    # mwdir is computed from normalized z-scores (line 2843-2848)
                    # Already bounded to [-1, 1] due to tanh normalization
                    # This provides volatility-adaptive uplift: same DKK delta → smaller uplift in high volatility
                    mwdir = float(getattr(self, 'mwdir', 0.0))

                    # Uplift = 1.0 + κ * |mwdir|
                    # Higher directional confidence → higher position sizing
                    # Example: mwdir=0.24 → uplift=1.036 (3.6% boost)
                    # Maximum: mwdir=1.0 → uplift=1.15 (15% boost)
                    uplift = 1.0 + kappa * min(abs(mwdir), 1.0)  # min() is redundant but safe
                    uplift = float(np.clip(uplift, 1.0, cap))

                    # CRITICAL FIX: Initialize base multiplier storage (once)
                    # This prevents exponential compounding over timesteps
                    if hasattr(self.reward_calculator, 'position_size_multiplier'):
                        if not hasattr(self.reward_calculator, '_base_position_multiplier'):
                            # Store the base multiplier (from drawdown control)
                            self.reward_calculator._base_position_multiplier = self.reward_calculator.position_size_multiplier

                        # CRITICAL FIX: SET (not multiply) to base * uplift
                        # This prevents exponential compounding: multiplier = base * uplift (not *= uplift)
                        self.reward_calculator.position_size_multiplier = self.reward_calculator._base_position_multiplier * uplift

                        # Log uplift application
                        if self.t % 1000 == 0:
                            logger.info(f"[RISK_UPLIFT] Applied: base={self.reward_calculator._base_position_multiplier:.3f}, "
                                       f"mwdir={mwdir:.3f}, uplift={uplift:.3f}, "
                                       f"final={self.reward_calculator.position_size_multiplier:.3f}, "
                                       f"trust={self._forecast_trust:.3f}")

                except Exception as e:
                    logger.debug(f"[RISK_UPLIFT] Failed: {e}")

        except Exception as e:
            logger.debug(f"[FGB] Forecast signal computation failed: {e}")

    # ------------------------------------------------------------------
    # Info & helpers
    # ------------------------------------------------------------------
    def _populate_info(self, i: int, financial: Dict[str, float], acts: Dict[str, np.ndarray]):
        try:
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
                    'action_investor': acts['investor_0'].tolist(),
                    'action_battery':  acts['battery_operator_0'].tolist(),
                    'action_risk':     acts['risk_controller_0'].tolist(),
                    'action_meta':     acts['meta_controller_0'].tolist(),

                    # Reward breakdown
                    'reward_breakdown': dict(self.last_reward_breakdown),
                    'reward_weights': dict(getattr(self, 'last_reward_weights', {})),

                    # Financial breakdown
                    'fund_nav': financial.get('fund_nav', self.equity),
                    'total_generation_mwh': financial.get('total_generation_mwh', 0.0),
                    'assets_deployed': self.assets_deployed,
                    'forecast_signal_score': financial.get('forecast_signal_score', 0.0),

                    # FGB: Forecast-guided baseline signals
                    'forecast_trust': float(self._forecast_trust),  # τₜ ∈ [0,1]
                    'expected_dnav': float(self._expected_dnav),    # E[ΔNAV] for next step
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

    def _safe_step(self):
        self._fill_obs()
        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

    # ------------------------------------------------------------------
    # Fund Performance and Diagnostics
    # ------------------------------------------------------------------

    def get_fund_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive fund performance summary for thesis validation."""
        try:
            fund_nav = self._calculate_fund_nav()
            total_return = (fund_nav - self.init_budget) / self.init_budget

            # Asset allocation percentages
            total_financial = sum(abs(v) for v in self.financial_positions.values())
            total_physical_value = (
                self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] +
                self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] +
                self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] +
                self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']
            )

            asset_allocation = {
                'cash_pct': (self.budget / fund_nav) * 100 if fund_nav > 0 else 0,
                'physical_pct': (total_physical_value / fund_nav) * 100 if fund_nav > 0 else 0,
                'financial_pct': (total_financial / fund_nav) * 100 if fund_nav > 0 else 0,
            }

            # Revenue breakdown
            revenue_breakdown = {
                'generation_revenue': self.cumulative_generation_revenue,
                'battery_arbitrage_revenue': self.cumulative_battery_revenue,
                'total_operational_revenue': self.cumulative_generation_revenue + self.cumulative_battery_revenue,
                'mtm_gains': self.cumulative_returns - (self.cumulative_generation_revenue + self.cumulative_battery_revenue)
            }

            return {
                'fund_nav': fund_nav,
                'initial_capital': self.init_budget,
                'total_return_pct': total_return * 100,
                'distributed_profits': self.distributed_profits,
                'current_equity': self.equity,
                'asset_allocation': asset_allocation,
                'revenue_breakdown': revenue_breakdown,
                'physical_assets': dict(self.physical_assets),
                'financial_positions': dict(self.financial_positions),
                'operational_state': dict(self.operational_state),
                'assets_deployed': self.assets_deployed,
            }
        except Exception as e:
            logger.error(f"Fund performance summary failed: {e}")
            return {'error': str(e)}

    def validate_hybrid_model_integrity(self) -> bool:
        """Validate the hybrid model maintains proper separation"""
        issues = []

        # Check physical assets are non-negative and fixed after deployment
        for asset, capacity in self.physical_assets.items():
            if capacity < 0:
                issues.append(f"Negative physical capacity: {asset} = {capacity}")

        # Check financial positions are within allocated trading capital limits
        total_financial = sum(abs(v) for v in self.financial_positions.values())

        # Get proper trading capital limits from config
        trading_limits = self.config.get_trading_capital_limits()
        max_financial_exposure = trading_limits['max_financial_exposure_dkk']

        if total_financial > max_financial_exposure:
            trading_capital = trading_limits['trading_capital_dkk']
            leverage = trading_limits['max_leverage']
            issues.append(f"Excessive financial exposure: {total_financial:.0f} DKK exceeds limit of {max_financial_exposure:.0f} DKK (trading capital {trading_capital:.0f} DKK × {leverage}x leverage)")

        # Check fund NAV is reasonable - more conservative bounds
        nav = self._calculate_fund_nav()
        # Allow reasonable growth but prevent excessive portfolio inflation
        min_nav = self.init_budget * 0.1   # Minimum 10% of initial
        max_nav = self.init_budget * 2.0   # Maximum 200% of initial (reduced from 10x)
        if nav < min_nav or nav > max_nav:
            issues.append(f"Unrealistic NAV: {nav:.0f}")

        # Check battery state consistency
        if self.operational_state['battery_energy'] > self.physical_assets['battery_capacity_mwh']:
            issues.append("Battery energy exceeds capacity")

        if issues:
            logger.warning("Hybrid model integrity issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False

        return True

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
        self.battery_capacity = float(value)  # Keep legacy sync

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