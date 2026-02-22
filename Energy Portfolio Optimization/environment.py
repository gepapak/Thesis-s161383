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
# Observation specs (Tier-1 only; no forecast features in observations)
# =============================================================================
class StabilizedObservationManager:
    def __init__(self, env: 'RenewableMultiAgentEnv'):
        self.env = env
        self.config = env.config if hasattr(env, 'config') else None
        self.observation_specs = self._build_specs()
        self.base_spaces = self._build_spaces()

    def _build_specs(self) -> Dict[str, Dict[str, Any]]:
        # Paper setup: keep observation spaces fixed across all modes (baseline vs FGB online vs FGB meta).
        # Forecasts are used ONLY in the DL overlay / FGB control-variate path, not injected into observations.
        return {
            "investor_0": {"base": 6},
            "battery_operator_0": {"base": 4},
            "risk_controller_0": {"base": 9},
            "meta_controller_0": {"base": 11},
        }

    def _build_spaces(self) -> Dict[str, spaces.Box]:
        sp: Dict[str, spaces.Box] = {}

        # Investor (6D): price, budget, wind_pos, solar_pos, hydro_pos, mtm_pnl
        inv_low = np.array([-1.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        inv_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Battery (4D): price, soc, capacity, load
        bat_low = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bat_high = np.array([1.0, 1.0, 10.0, 1.0], dtype=np.float32)

        # Risk controller (9D): price_n, vol, stress, positions (3), cap_frac, equity, risk_mult
        risk_low = np.array([-1.0, 0.0, 0.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        risk_high = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        # Meta controller (11D): budget, positions (3), price_n, risks (4), perf, cap_frac
        meta_low = np.array([0.0, -10.0, -10.0, -10.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        meta_high = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        sp["investor_0"] = spaces.Box(low=inv_low, high=inv_high, shape=(6,), dtype=np.float32)
        sp["battery_operator_0"] = spaces.Box(low=bat_low, high=bat_high, shape=(4,), dtype=np.float32)
        sp["risk_controller_0"] = spaces.Box(low=risk_low, high=risk_high, shape=(9,), dtype=np.float32)
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

            self.max_drawdown_threshold = 0.10
        self.current_drawdown = 0.0
        self.peak_nav = float(self.initial_budget)
        self.trading_enabled = True
        self.position_size_multiplier = 1.0

        logger.info("[REWARD] Forecast component: DISABLED (paper setup; no forecast reward shaping)")
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
        except Exception as e:
            logger.warning(f"Hedging effectiveness calculation failed: {e}")
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
        # Keep baseline naming as "tier1" for backwards compatibility; use explicit labels for paper variants.
        tier_name = "tier1"
        if bool(getattr(self.config, "forecast_baseline_enable", False)):
            if str(getattr(self.config, "fgb_mode", "online")).lower() == "meta" and bool(getattr(self.config, "meta_baseline_enable", False)):
                tier_name = "fgb_meta"
            else:
                tier_name = "fgb_online"
        elif bool(getattr(self.config, "overlay_enabled", False)):
            tier_name = "dl_overlay"
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

        if use_global_norm and (g_mean is not None) and (g_std is not None):
            mean_val = float(g_mean)
            std_val = float(max(float(g_std), 1e-6))
            price_normalized = (price_dkk_filtered - mean_val) / std_val
            price_normalized_clipped = np.clip(price_normalized, -3.0, 3.0)
            self._price = (price_normalized_clipped / 3.0).to_numpy()
            # store constant arrays for any code that expects per-timestep mean/std
            self._price_mean = np.full(len(self._price_raw), mean_val, dtype=float)
            self._price_std = np.full(len(self._price_raw), std_val, dtype=float)
            window_size = min(4320, len(price_dkk_filtered))
        else:
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
        
        # CRITICAL FIX: Compute rolling mean/std for generation assets (wind, solar, hydro).
        # Needed for consistent z-score normalization used by calibration/overlay components.
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

        if use_global_norm:
            # Prefer global p95 scales when available, fall back to per-episode robust p95.
            self.wind_scale  = float(getattr(self.config, "global_wind_scale", None) or p95_robust(self._wind,  min_scale=0.1))
            self.solar_scale = float(getattr(self.config, "global_solar_scale", None) or p95_robust(self._solar, min_scale=0.1))
            self.hydro_scale = float(getattr(self.config, "global_hydro_scale", None) or p95_robust(self._hydro, min_scale=0.1))
            self.load_scale  = float(getattr(self.config, "global_load_scale", None) or p95_robust(self._load,  min_scale=0.1))
        else:
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
        # MINIMAL MODE: Investor action is exposure-only (single scalar).
        investor_shape = (1,)
        self.action_spaces = {
            # Investor action (structural anti-collapse):
            # - exposure-only: [exposure_raw] in [-1,1]
            "investor_0":         spaces.Box(low=-1.0, high=1.0, shape=investor_shape, dtype=np.float32),
            "battery_operator_0": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "risk_controller_0":  spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # FIXED: [-1,1] instead of [0,2]
            "meta_controller_0":  spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # FIXED: [-1,1] instead of [0,1]
        }

        # Meta-controlled exposure (used when investor_exposure_controller="meta")
        self.meta_exposure = 1.0

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
        # Directional (sign) hit-rate tracking per horizon for trust calibration.
        # 1.0 means forecast and realized direction match for that horizon, 0.0 otherwise.
        self._horizon_sign_hit = {
            'short': deque(maxlen=200),
            'medium': deque(maxlen=200),
            'long': deque(maxlen=200),
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
        self.last_realized_dnav = 0.0
        self.last_realized_dnav_return = 0.0
        self._overlay_prev_nav = None
        self.cumulative_mtm_pnl = 0.0  # ENHANCED: Track cumulative trading performance
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

        # =====================================================================
        # DL OVERLAY STATE (Tier 3: FGB/FAMC only)
        # =====================================================================
        # IMPORTANT (fairness): overlay outputs are NOT injected into env rewards or trade sizing.
        # The overlay is used only for FGB/FAMC variance reduction (baseline / meta-critic signals).
        self.dl_adapter_overlay = None  # Set by main.py (DLAdapter for overlay inference)
        self.overlay_trainer = None  # Set by main.py (online overlay trainer; pred_reward head only)

        # =====================================================================
        # FGB: FORECAST-GUIDED BASELINE STATE
        # =====================================================================
        self.calibration_tracker = None  # Will be set by main.py (CalibrationTracker for trust & ΔNAV)
        self._last_overlay_output = {}  # Cache latest overlay output for forecast signals
        self._last_overlay_features = None  # FAMC: Cache overlay features for meta head training
        self._forecast_trust = 0.0  # τₜ: forecast trust score
        self._expected_dnav = 0.0  # E[ΔNAV]: expected NAV change

        # =====================================================================
        # PAPER MODES: Baseline vs FGB-online vs FGB-meta (no forecast-augmented obs)
        # =====================================================================
        # Forecasts (if enabled) are backend-only for FGB/FAMC variance reduction.

        logger.info("=" * 70)
        if bool(getattr(self.config, "forecast_baseline_enable", False)):
            if str(getattr(self.config, "fgb_mode", "online")).lower() == "meta" and bool(getattr(self.config, "meta_baseline_enable", False)):
                logger.info("FGB META (FAMC): Tier-1 observations + learned control variate (meta-critic)")
            else:
                logger.info("FGB ONLINE: Tier-1 observations + online control variate (expected_dnav)")
        elif bool(getattr(self.config, "overlay_enabled", False)):
            logger.info("DL OVERLAY: Tier-1 observations (overlay enabled; no baseline correction)")
        else:
            logger.info("BASELINE MARL: Tier-1 observations (no forecasts, no overlay)")
        logger.info("=" * 70)

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
        self._overlay_pnl_history = deque(maxlen=20)      # Recent P&L for pred_reward target

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
                logger.info("[TIER1_NAV_CALC] Calculating NAV at t=0:")
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
                logger.info(f"[TIER1_NAV_CALC] NAV calculated: {nav:,.0f} DKK (${nav * 0.145 / 1_000_000:.2f}M)")
            
            self.equity = nav
            return nav

        except Exception as e:
            msg = (
                f"[NAV_CALC_FATAL] NAV calculation failed at t={getattr(self, 't', 'unknown')}: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

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

                # DIAGNOSTIC LOGGING (every 500 steps)
                # IMPORTANT: this tracker is in *normalized space* (z-score for price; scaled for wind/solar/hydro/load).
                # It is NOT the same as the horizon MAPE used for Tier2 trust/error (which is computed in raw price space).
                if self.t % 500 == 0 and len(self._forecast_errors[target]) >= 10:
                    recent_mape = np.mean(list(self._forecast_errors[target])[-10:])
                    if target == "price":
                        # Avoid confusion: this is z-space tracking, not raw-price MAPE.
                        logger.info(
                            f"[PRICE_ZSPACE_TRACKING] t={self.t} recent_err={recent_mape:.4f} "
                            f"samples={len(self._forecast_errors[target])} "
                            f"actual_z={actual_normalized:.4f} forecast_z={forecast_normalized:.4f}"
                        )
                    else:
                        logger.info(
                            f"[MAPE_TRACKING] t={self.t} target={target} recent_mape={recent_mape:.4f} "
                            f"samples={len(self._forecast_errors[target])} "
                            f"actual={actual_normalized:.4f} forecast={forecast_normalized:.4f}"
                        )
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
        except Exception as e:
            raise RuntimeError(
                f"[ALIGNED_PRICE_FORECAST_FATAL] Failed to get aligned price forecast at step={t}: {e}"
            ) from e
        return default

    def populate_forecast_arrays(self, t: int, forecasts: Dict[str, float]):
        """
        FIXED: Populate forecast arrays for DL overlay labeler access.
        Called from wrapper after computing forecasts.

        Now handles ALL horizons (immediate, short, medium, long) for each target.
        Overlay features use config.overlay_horizon (default: short).

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
            raise RuntimeError(
                f"[FORECAST_ARRAYS_FATAL] Population error at t={t}: {e}"
            ) from e

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
        overlay_h = str(getattr(self.config, 'overlay_horizon', 'short') or 'short')
        logger.info(f"[RESET] Environment reset: overlay_enabled={overlay_enabled}, feature_dim={feature_dim} ({overlay_h}-horizon), has_forecaster={has_forecaster}")

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

        # CRITICAL: Compute forecast deltas before filling observations so the DL overlay/FGB backend is ready from t=0.
        # FIX Issue #3: Initialize z-prev values on first reset
        forecast_backend_enabled = bool(
            getattr(self.config, 'overlay_enabled', False)
            or getattr(self.config, 'forecast_baseline_enable', False)
        )
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
        self.last_realized_dnav = 0.0
        self.last_realized_dnav_return = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}
        

        # CRITICAL FIX: Recalculate NAV with current state (preserving financial gains)
        # Use current timestep for depreciation calculation
        current_t = getattr(self, 't', 0)
        current_nav = self._calculate_fund_nav(current_timestep=current_t)
        
        # STRATEGY 4: Reset previous NAV for NAV-based reward calculation
        # Initialize with current NAV so first step's reward is based on change from reset NAV
        self._previous_nav = current_nav if current_nav > 1e-6 else float(self.init_budget)
        self._overlay_prev_nav = current_nav if current_nav > 1e-6 else float(self.init_budget)

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
        self.last_realized_dnav = 0.0
        self.last_realized_dnav_return = 0.0
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
                financial_positions={
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
        self._overlay_prev_nav = float(initial_nav)
        
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
                            exposure_controller = str(getattr(self.config, "investor_exposure_controller", "investor") or "investor").lower()
                            if exposure_controller == "meta":
                                exposure = float(np.clip(getattr(self, "meta_exposure", 1.0), -1.0, 1.0))
                            else:
                                exposure = float(np.clip(arr[0] if arr.size > 0 else 1.0, -1.0, 1.0))

                            # Fixed equal weights across instruments (aggregate instrument behavior).
                            w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                            w = (w / float(np.sum(np.abs(w)))).astype(np.float32)
                            alloc = (exposure * w).astype(np.float32)

                            self._last_actions["investor_0_exec"] = {
                                "wind": float(alloc[0]),
                                "solar": float(alloc[1]),
                                "hydro": float(alloc[2]),
                                "exposure": float(exposure),
                                "w_wind": float(w[0]),
                                "w_solar": float(w[1]),
                                "w_hydro": float(w[2]),
                            }
                            self._last_investor_action_exec = alloc.copy()
                        except Exception:
                            pass
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

            # ------------------------------------------------------------------
            # FGB/FAMC: control-variate signals MUST be computed from PRE-ACTION state.
            # This keeps the baseline action-independent on decision steps (paper-clean).
            # ------------------------------------------------------------------
            forecast_backend_enabled = bool(
                getattr(self.config, 'overlay_enabled', False)
                or getattr(self.config, 'forecast_baseline_enable', False)
            )

            overlay_features_pre = None
            # Avoid stale baseline/meta predictions if overlay inference fails on this step.
            # (We want control-variate signals to be state-only for *this* timestep.)
            if forecast_backend_enabled:
                self._last_overlay_output = {}
                self._last_overlay_features = None

            # Compute z-scores/deltas for CURRENT timestep i (used for calibration/trust, logging, etc.).
            if forecast_backend_enabled and self.forecast_generator is not None:
                self._compute_forecast_deltas(i, update_calibration=True)

            # Run overlay inference once per step, PRE-ACTION.
            overlay_adapter = getattr(self, 'dl_adapter_overlay', None)
            if overlay_adapter is not None and getattr(self.config, 'overlay_enabled', False):
                try:
                    base_features = self._build_overlay_features(i)
                    if base_features is not None:
                        overlay_features_pre = base_features.copy()
                        overlay_outs = overlay_adapter.shared_inference(base_features, training=False)

                        # Cache results for FGB/FAMC (never injected into env reward/obs)
                        if overlay_outs:
                            self._last_overlay_output = overlay_outs
                            self._last_overlay_features = base_features.copy()
                        elif bool(getattr(self.config, "fgb_fail_fast", False)):
                            raise RuntimeError("[OVERLAY_INFERENCE] shared_inference returned empty output")
                    elif bool(getattr(self.config, "fgb_fail_fast", False)):
                        raise RuntimeError("[OVERLAY_INFERENCE] _build_overlay_features returned None")
                except Exception as e:
                    msg = f"[OVERLAY_INFERENCE] CRITICAL FAILURE (pre-action) at step {self.t}: {e}"
                    if bool(getattr(self.config, "fgb_fail_fast", False)):
                        raise RuntimeError(msg) from e
                    logger.warning(msg)
                    logger.debug(f"[OVERLAY_INFERENCE] Full traceback: {traceback.format_exc()}")

            # FGB: Compute expected_dnav for baseline adjustment (control variate only), PRE-ACTION.
            if bool(getattr(self.config, 'forecast_baseline_enable', False)) and bool(getattr(self.config, 'overlay_enabled', False)):
                self._compute_forecast_signals()

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
                except Exception:
                    pass

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
                    # Use pre-action features for state-only training targets (outcome realized post-action).
                    if overlay_features_pre is not None:
                        self._collect_overlay_experience(i, features=overlay_features_pre)
                except Exception as e:
                    raise RuntimeError(
                        f"[OVERLAY_TRAINING_FATAL] Experience collection failed at step {self.t}: {e}"
                    ) from e

            # FIXED: assign rewards
            self._assign_rewards(financial)

            # step forward
            self.t += 1
            self.step_in_episode = self.t

            index = max(0, self.t - 1)
            # ------------------------------------------------------------------
            # Keep forecast backend state aligned to the next timestep (no double-counting).
            # ------------------------------------------------------------------
            # After incrementing self.t, the next observation is for timestep `self.t`.
            # Recompute forecast deltas for that timestep WITHOUT updating calibration/trust.
            if forecast_backend_enabled and self.forecast_generator is not None and self.t < self.max_steps:
                try:
                    self._compute_forecast_deltas(self.t, update_calibration=False)
                except Exception as e:
                    raise RuntimeError(
                        f"[FORECAST_DELTAS_FATAL] Next-step update failed at step {self.t}: {e}"
                    ) from e
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
                    raise RuntimeError(
                        f"[FORECAST_UPDATE_FATAL] Failed to update forecast generator at t={i}: {e}"
                    ) from e

            self._populate_info(index, financial, acts)

            return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

        except Exception as e:
            msg = f"step error at t={self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

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
            msg = f"Risk control failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _apply_meta_control(self, meta_action: np.ndarray):
        """
        REFACTORED: Apply meta control using TradingEngine.
        """
        try:
            from trading_engine import TradingEngine
            self.capital_allocation_fraction, self.investment_freq = TradingEngine.apply_meta_control(
                meta_action=meta_action,
                meta_cap_min=self.META_CAP_MIN,
                meta_cap_max=self.META_CAP_MAX,
                meta_freq_min=self.META_FREQ_MIN,
                meta_freq_max=self.META_FREQ_MAX,
                # Keep meta-control dynamics identical across variants (no confidence-based scaling).
                forecast_confidence=0.5,
                disable_confidence_scaling=True,
            )

            # Meta-controlled exposure (structural anti-collapse):
            # Interpret meta_action[0] as an exposure dial in [-1,1] when enabled.
            exposure_controller = str(
                getattr(self.config, "investor_exposure_controller", "investor") or "investor"
            ).lower()
            if exposure_controller == "meta":
                a0 = float(np.array(meta_action, dtype=np.float32).reshape(-1)[0])
                # Signed exposure so meta can take short positions
                self.meta_exposure = float(np.clip(a0, -1.0, 1.0))
        except Exception as e:
            msg = f"Meta control failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

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
            return 1.0  # Default intensity

    # def set_wrapper_reference(self, wrapper_env):
    #     """CRITICAL: Set reference to wrapper for profit-seeking expert"""
    #     self._wrapper_ref = wrapper_env

    # Optional policy encoders may learn richer feature relationships (Tier-1 observation space).
    # Forecast models (ANN/LSTM) are used only for the DL overlay / FGB backend signals (no extra observations).
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

            # ===== ADAPTIVE RISK BUDGETING =====
            # This function is used for fallback hedging when overlay is not active

            # REASON: This creates triple-scaling:
            #   3. Blending with expert suggestions (which already account for risk)

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

            # CRITICAL FIX: Apply risk controller's multiplier
            # Risk controller sets self.risk_multiplier via _apply_risk_control()
            # This must be included in position sizing to ensure risk controller's actions affect trades
            risk_controller_mult = getattr(self, 'risk_multiplier', 1.0)

            # FAIR TIER 3: Remove overlay risk-budget sizing.
            # Tier 3 remains Tier 2 observations + FGB/FAMC (baseline correction), without changing trade sizing logic.
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
                logger.debug(
                    f"[SIZING_MULT] Step {t}: "
                    f"risk_controller={risk_controller_mult:.4f}, rl_mult={position_size_multiplier:.4f}, "
                    f"vol_brake={volatility_brake_mult:.4f}, combined={combined_multiplier:.4f}"
                )

            # === STEP 2: Map the agent's normalized action to target DKK positions ===
            # MINIMAL MODE: Investor action is exposure-only; allocation across wind/solar/hydro is fixed.
            a = np.asarray(inv_action, dtype=np.float32).reshape(-1)
            exposure_controller = str(getattr(self.config, "investor_exposure_controller", "investor") or "investor").lower()
            if exposure_controller == "meta":
                # Meta controls exposure in minimal mode as well.
                exposure_raw = float(getattr(self, "meta_exposure", 1.0))
            else:
                exposure_raw = float(a[0]) if a.size >= 1 else 1.0
            # Signed exposure: allow shorting to align with negative forecast signals.
            exposure = float(np.clip(exposure_raw, -1.0, 1.0))
            # Log exposure at decision cadence for action-scale debugging
            try:
                inv_freq = int(getattr(self, "investment_freq", 6))
            except Exception:
                inv_freq = 6
            if t % inv_freq == 0:
                logger.debug(
                    f"[INVESTOR_EXPOSURE_RAW] t={t} exposure_raw={exposure_raw:.4f} exposure_clipped={exposure:.4f} "
                    f"controller={exposure_controller}"
                )
            self._last_risk_multiplier = float(risk_controller_mult)
            self._last_vol_brake_mult = float(volatility_brake_mult)
            self._last_strategy_multiplier = float(strategy_multiplier)
            self._last_combined_multiplier = float(combined_multiplier)
            self._last_tradeable_capital = float(tradeable_capital)

            # Fixed equal weights across instruments (aggregate instrument behavior).
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
                           f"combined={combined_multiplier:.2f}, "
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
            msg = f"[TRADE_EXEC_FATAL] _execute_investor_trades failed at step {self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

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
        except Exception as e:
            raise RuntimeError(
                f"[PRICE_NORM_FATAL] _norm_price failed at step={t} with price={p}: {e}"
            ) from e

    def _compute_forecast_deltas(self, i: int, update_calibration: bool = True) -> None:
        """
        Compute forecast deltas/z-scores for the forecast backend (DL overlay / FGB).

        IMPORTANT (fairness): these are NOT appended to agent observations in the paper setup.
        They feed only the DL overlay feature builder and calibration/trust tracking.

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

            # Populate per-target forecast arrays so the overlay feature builder can access them via _get_forecast_safe().
            # This is the single source of truth for forecast arrays (no wrapper required).
            self.populate_forecast_arrays(i, forecasts)

            # Get current values (raw DKK for price; generation normalized to [0,1] using env scales).
            # NOTE: ForecastGenerator returns RAW MW values for generation targets. We normalize them here
            # to keep a single contract across env overlay features and wrapper post-processing.
            price_raw = float(self._price_raw[i] if i < len(self._price_raw) else 250.0)
            wind_scale = max(self.wind_scale, 1e-6)
            solar_scale = max(self.solar_scale, 1e-6)
            hydro_scale = max(self.hydro_scale, 1e-6)

            wind_current_raw_mw = float(self._wind[i]) if i < len(self._wind) else 0.0
            solar_current_raw_mw = float(self._solar[i]) if i < len(self._solar) else 0.0
            hydro_current_raw_mw = float(self._hydro[i]) if i < len(self._hydro) else 0.0

            wind_current = float(np.clip(wind_current_raw_mw / wind_scale, 0.0, 1.0))
            solar_current = float(np.clip(solar_current_raw_mw / solar_scale, 0.0, 1.0))
            hydro_current = float(np.clip(hydro_current_raw_mw / hydro_scale, 0.0, 1.0))

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

            # Forecast generation values are RAW MW. Normalize+clip to [0,1] for delta/EMA logic.
            wind_short_raw_mw = float(forecasts.get('wind_forecast_short', wind_current_raw_mw))
            wind_medium_raw_mw = float(forecasts.get('wind_forecast_medium', wind_current_raw_mw))
            wind_long_raw_mw = float(forecasts.get('wind_forecast_long', wind_current_raw_mw))

            solar_short_raw_mw = float(forecasts.get('solar_forecast_short', solar_current_raw_mw))
            solar_medium_raw_mw = float(forecasts.get('solar_forecast_medium', solar_current_raw_mw))
            solar_long_raw_mw = float(forecasts.get('solar_forecast_long', solar_current_raw_mw))

            hydro_short_raw_mw = float(forecasts.get('hydro_forecast_short', hydro_current_raw_mw))
            hydro_medium_raw_mw = float(forecasts.get('hydro_forecast_medium', hydro_current_raw_mw))
            hydro_long_raw_mw = float(forecasts.get('hydro_forecast_long', hydro_current_raw_mw))

            wind_short = float(np.clip(wind_short_raw_mw / wind_scale, 0.0, 1.0))
            wind_medium = float(np.clip(wind_medium_raw_mw / wind_scale, 0.0, 1.0))
            wind_long = float(np.clip(wind_long_raw_mw / wind_scale, 0.0, 1.0))

            solar_short = float(np.clip(solar_short_raw_mw / solar_scale, 0.0, 1.0))
            solar_medium = float(np.clip(solar_medium_raw_mw / solar_scale, 0.0, 1.0))
            solar_long = float(np.clip(solar_long_raw_mw / solar_scale, 0.0, 1.0))

            hydro_short = float(np.clip(hydro_short_raw_mw / hydro_scale, 0.0, 1.0))
            hydro_medium = float(np.clip(hydro_medium_raw_mw / hydro_scale, 0.0, 1.0))
            hydro_long = float(np.clip(hydro_long_raw_mw / hydro_scale, 0.0, 1.0))

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
            # ANTI-SATURATION: use denominator floor + clip returns + configurable tanh scale.
            denom_floor = float(getattr(self.config, 'forecast_return_denom_floor', 1.0) or 1.0)
            denom = max(abs(price_raw), denom_floor, 1.0)
            forecast_return_short = delta_price_short_raw / denom
            forecast_return_medium = delta_price_medium_raw / denom
            forecast_return_long = delta_price_long_raw / denom

            ret_clip = float(getattr(self.config, 'forecast_return_clip', 1e9) or 1e9)
            if ret_clip > 0:
                forecast_return_short = float(np.clip(forecast_return_short, -ret_clip, ret_clip))
                forecast_return_medium = float(np.clip(forecast_return_medium, -ret_clip, ret_clip))
                forecast_return_long = float(np.clip(forecast_return_long, -ret_clip, ret_clip))

            tanh_scale = float(getattr(self.config, 'forecast_return_tanh_scale', 10.0) or 10.0)
            z_short = np.tanh(forecast_return_short * tanh_scale)
            z_medium = np.tanh(forecast_return_medium * tanh_scale)
            z_long = np.tanh(forecast_return_long * tanh_scale)

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

            # DIAGNOSTIC LOGGING
            # We only emit heavy diagnostics on the "main" pass (update_calibration=True) to prevent duplicate logs.
            if update_calibration and i % 1000 == 0:
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
            
            # Raw forecast values (MW) from ForecastGenerator are already available.
            wind_current_raw = float(wind_current_raw_mw)
            solar_current_raw = float(solar_current_raw_mw)
            hydro_current_raw = float(hydro_current_raw_mw)

            wind_short_raw = float(wind_short_raw_mw)
            wind_medium_raw = float(wind_medium_raw_mw)
            wind_long_raw = float(wind_long_raw_mw)

            solar_short_raw = float(solar_short_raw_mw)
            solar_medium_raw = float(solar_medium_raw_mw)
            solar_long_raw = float(solar_long_raw_mw)

            hydro_short_raw = float(hydro_short_raw_mw)
            hydro_medium_raw = float(hydro_medium_raw_mw)
            hydro_long_raw = float(hydro_long_raw_mw)
            
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

            # Single horizon: use short only (mwdir and direction_consistency)
            direction_consistency = float(np.sign(z_short) if z_short != 0 else 0.0)
            mwdir = float(z_short)

            # Legacy variables (for backward compatibility)
            self.z_short = self.z_short_price
            self.z_medium = self.z_medium_price
            self.z_long = self.z_long_price
            self.direction_consistency = float(np.clip(direction_consistency, -1.0, 1.0))
            self.mwdir = float(np.clip(mwdir, -1.0, 1.0))
            self.abs_mwdir = float(np.clip(abs(mwdir), 0.0, 1.0))

            # ------------------------------------------------------------------
            # Calibration / trust update
            # ------------------------------------------------------------------
            # IMPORTANT: We must NOT update calibration/trust twice per environment step.
            # - During `step()`, we compute deltas once for the current timestep i (update_calibration=True),
            #   then (after incrementing self.t) we may recompute deltas for the next observation (update_calibration=False)
            #   to avoid a 1-step "staleness" in the policy-visible forecast features.
            if update_calibration:
                # CRITICAL FIX: Update calibration tracker HERE (not in _build_overlay_features)
                # This ensures trust is computed even when overlay_enabled=False (Tier 2)
                self._ema_std_short_for_calibration = self.ema_std_short
                self._ema_std_medium_for_calibration = self.ema_std_medium
                self._ema_std_long_for_calibration = self.ema_std_long

                # Update calibration tracker with forecast/realized pairs
                self._update_calibration_tracker(i)

                # Store forecast trust (τ) after calibration update.
                #
                # IMPORTANT:
                # - Tier 3 (DL overlay enabled): keep using CalibrationTracker (FGB-style trust).
                # - Tier 2 (overlay disabled): prefer env-tracked directional accuracy, and FIX the
                #   trust horizon to "medium" so meta-controlled investment_freq can't cause trust jitter.
                overlay_enabled = bool(getattr(self.config, "overlay_enabled", False))
                if overlay_enabled and hasattr(self, 'calibration_tracker') and self.calibration_tracker is not None:
                    # Tier 3 / FGB trust path.
                    recent_mape = None
                    if hasattr(self, '_horizon_mape') and len(self._horizon_mape.get('short', [])) > 0:
                        recent_mape_values = list(self._horizon_mape['short'])[-10:]
                        if len(recent_mape_values) > 0:
                            recent_mape = float(np.mean(recent_mape_values))

                    self._forecast_trust = self.calibration_tracker.get_trust(horizon="short", recent_mape=recent_mape)
                else:
                    # Tier 2: trust = rolling directional accuracy (sign hit-rate) on MEDIUM horizon.
                    # This matches the investor's medium-horizon trade_signal.
                    metric = str(getattr(self.config, 'forecast_trust_metric', 'hitrate') or 'hitrate').lower()

                    def _mean_last_mape(h: str, n: int = 10) -> float:
                        try:
                            xs = list(getattr(self, '_horizon_mape', {}).get(h, []))[-n:]
                            xs = [float(x) for x in xs if np.isfinite(x)]
                            return float(np.mean(xs)) if len(xs) > 0 else float(getattr(self, '_mape_thresholds', {}).get(h, 0.12))
                        except Exception:
                            return float(getattr(self, '_mape_thresholds', {}).get(h, 0.12))

                    def _mean_last_hit(h: str, n: int = 50) -> float:
                        try:
                            xs = list(getattr(self, '_horizon_sign_hit', {}).get(h, []))[-n:]
                            xs = [float(x) for x in xs if np.isfinite(x)]
                            return float(np.mean(xs)) if len(xs) > 0 else 0.5
                        except Exception:
                            return 0.5

                    # Use configured trust horizon (short for investment_freq=6 alignment).
                    h_by_agent = getattr(self.config, 'fgb_trust_horizon_by_agent', None) or {}
                    trust_h = str(h_by_agent.get("investor_0", "short"))

                    if metric in ('hitrate', 'absdir', 'sign', 'direction'):
                        # Use hit-rate directly as trust target (0..1). 0.5 ~= random.
                        trust_target = float(np.clip(_mean_last_hit(trust_h, 50), 0.0, 1.0))
                    else:
                        # Fallback: legacy MAPE-banded trust.
                        mape_short = _mean_last_mape('short', 10)
                        mape_medium = _mean_last_mape('medium', 10)
                        mape_long = _mean_last_mape('long', 10)
                        weighted_mape = float(0.70 * mape_short + 0.20 * mape_medium + 0.10 * mape_long)

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

                    prev = float(getattr(self, '_forecast_trust', 0.5))
                    self._forecast_trust = float(np.clip(0.7 * trust_target + 0.3 * prev, 0.0, 1.0))

                # Per-agent trust horizon selection for FGB/FAMC baseline scaling.
                # This does NOT change Tier1 observations/rewards; it only affects the control variate.
                #
                # - investor_0: uses configured horizon (short for investment_freq=6)
                # - others: "auto" (best recent directional accuracy)
                try:
                    self._forecast_trust = float(self.get_fgb_trust_for_agent("investor_0"))
                except Exception:
                    pass

                # Apply trust floor to avoid underusing control variate when forecasts pass quality
                trust_floor = float(getattr(self.config, 'forecast_trust_floor', 0.0) or 0.0)
                if trust_floor > 0.0:
                    self._forecast_trust = float(np.maximum(self._forecast_trust, trust_floor))

            # =====================================================================
            # NOVEL: FORECAST RETURNS INTEGRATION (Bias-Immune)
            # =====================================================================
            # Instead of capacity-based deltas, use RETURNS (percentage changes)
            # Returns are stationary and bias cancels out in the division
            # This focuses on what matters: price movements, not absolute levels

            # Compute FORECAST RETURNS (percentage change from current price)
            denom_price = max(abs(price_raw), 1.0)
            # Include immediate + strategic for aux-task horizon selection (one horizon per feature, minimal).
            try:
                delta_price_immediate_raw = float(forecasts.get('price_forecast_immediate', price_raw)) - float(price_raw)
            except Exception:
                delta_price_immediate_raw = 0.0
            try:
                delta_price_strategic_raw = float(forecasts.get('price_forecast_strategic', price_raw)) - float(price_raw)
            except Exception:
                delta_price_strategic_raw = 0.0

            forecast_return_immediate = delta_price_immediate_raw / denom_price
            forecast_return_short = delta_price_short_raw / denom_price
            forecast_return_medium = delta_price_medium_raw / denom_price
            forecast_return_long = delta_price_long_raw / denom_price
            forecast_return_strategic = delta_price_strategic_raw / denom_price

            # Store as forecast deltas (for backward compatibility with existing code)
            self._forecast_deltas_raw = {
                'immediate': float(forecast_return_immediate),
                'short': float(forecast_return_short),   # Return-based (bias-immune)
                'medium': float(forecast_return_medium), # Return-based (bias-immune)
                'long': float(forecast_return_long),     # Return-based (bias-immune)
                'strategic': float(forecast_return_strategic),
            }

            # Store price capacity for CSV logging (backward compatibility)
            price_capacity = 6982.0  # Forecast model capacity (for MAPE calculation)
            self._current_price_floor = float(price_capacity)

            # DIAGNOSTIC LOGGING: Track forecast returns (every 200 steps) — main pass only
            if update_calibration and i % 200 == 0:
                logger.debug(
                    f"[FORECAST_RETURNS] t={i} price={price_raw:.2f} short={forecast_return_short*100:.2f}% "
                    f"med={forecast_return_medium*100:.2f}% long={forecast_return_long*100:.2f}%"
                )
                logger.info(
                    f"[FORECAST_RETURNS] t={i} price_raw={price_raw:.2f} "
                    f"return_short={forecast_return_short:.6f} ({forecast_return_short*100:.2f}%) "
                    f"return_medium={forecast_return_medium:.6f} ({forecast_return_medium*100:.2f}%) "
                    f"return_long={forecast_return_long:.6f} ({forecast_return_long*100:.2f}%) "
                    f"abs_short={delta_price_short_raw:.2f} abs_medium={delta_price_medium_raw:.2f} abs_long={delta_price_long_raw:.2f}"
                )

            # Forecast quality thresholds (price-relative MAPE).
            # These are fixed "acceptable error" bands used to normalize error features
            # and to keep trust/error semantics stable across training + evaluation.
            self._mape_thresholds = {
                'short': float(getattr(self.config, 'forecast_mape_threshold_short', 0.08) or 0.08),
                'medium': float(getattr(self.config, 'forecast_mape_threshold_medium', 0.12) or 0.12),
                'long': float(getattr(self.config, 'forecast_mape_threshold_long', 0.20) or 0.20),
            }

            # DIAGNOSTIC LOGGING: Track MAPE thresholds (every 200 steps) — main pass only
            if update_calibration and i % 200 == 0:
                logger.debug(
                    f"[MAPE] t={i} short={self._mape_thresholds.get('short', 0.0)*100:.2f}% "
                    f"med={self._mape_thresholds.get('medium', 0.0)*100:.2f}% "
                    f"long={self._mape_thresholds.get('long', 0.0)*100:.2f}%"
                )
                logger.info(
                    f"[MAPE_THRESHOLDS] t={i} "
                    f"short={self._mape_thresholds.get('short', 0.0):.4f} ({len(self._horizon_mape.get('short', []))} samples) "
                    f"medium={self._mape_thresholds.get('medium', 0.0):.4f} ({len(self._horizon_mape.get('medium', []))} samples) "
                    f"long={self._mape_thresholds.get('long', 0.0):.4f} ({len(self._horizon_mape.get('long', []))} samples)"
                )

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
            if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
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

        CRITICAL: This must be called independently of overlay_enabled setting!
        Previously this was inside _build_overlay_features(), causing Tier 2 to never
        update the calibration tracker (since overlay_enabled=False in Tier 2).

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
                denom_floor = float(getattr(self.config, 'forecast_return_denom_floor', 1.0) or 1.0)
                ret_clip = float(getattr(self.config, 'forecast_return_clip', 1e9) or 1e9)
                tanh_scale = float(getattr(self.config, 'forecast_return_tanh_scale', 10.0) or 10.0)

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
                horizon_short = self.config.forecast_horizons.get('short', 6)
                horizon_medium = self.config.forecast_horizons.get('medium', 24)
                horizon_long = self.config.forecast_horizons.get('long', 144)
                
                current_price_raw = float(self._price_raw[i])
                
                # Calculate MAPE for short horizon
                # CRITICAL FIX: Forecast at step i-horizon_short predicts price at timestep i
                # So we compare forecast from i-horizon_short with actual at i (matching horizon)
                denom_floor = float(getattr(self.config, 'minimum_price_floor', 50.0) or 50.0)
                if i >= horizon_short and len(self._horizon_forecast_pairs['short']) > 0:
                    # Find forecast made at i-horizon_short (matches corrected horizon logic)
                    for pair in reversed(self._horizon_forecast_pairs['short']):
                        if pair['timestep'] == i - horizon_short:  # = i - 6 for horizon=6
                            forecast_price = pair['forecast_price']
                            try:
                                fp = float(forecast_price)
                                pt = float(pair.get('current_price', 0.0))
                                if np.isfinite(fp) and np.isfinite(pt) and abs(fp) > 1e-6:
                                    denom = max(abs(pt), denom_floor, 1.0)
                                    mape_short = abs(current_price_raw - fp) / denom
                                    self._horizon_mape['short'].append(float(np.clip(mape_short, 0.0, 1.0)))
                            except Exception:
                                pass
                            break

                # Calculate MAPE for medium horizon (compare forecast from i-25 with actual at i)
                if i >= horizon_medium and len(self._horizon_forecast_pairs['medium']) > 0:
                    for pair in reversed(self._horizon_forecast_pairs['medium']):
                        if pair['timestep'] == i - horizon_medium:  # = i - 24 for horizon=24
                            forecast_price = pair['forecast_price']
                            try:
                                fp = float(forecast_price)
                                pt = float(pair.get('current_price', 0.0))
                                if np.isfinite(fp) and np.isfinite(pt) and abs(fp) > 1e-6:
                                    denom = max(abs(pt), denom_floor, 1.0)
                                    mape_medium = abs(current_price_raw - fp) / denom
                                    self._horizon_mape['medium'].append(float(np.clip(mape_medium, 0.0, 1.0)))
                            except Exception:
                                pass
                            break

                # Calculate MAPE for long horizon (compare forecast from i-145 with actual at i)
                # FIX #21: Deque maxlen increased to 200 to ensure pairs from 144 timesteps ago are available
                if i >= horizon_long and len(self._horizon_forecast_pairs['long']) > 0:
                    for pair in reversed(self._horizon_forecast_pairs['long']):
                        if pair['timestep'] == i - horizon_long:  # = i - 144 for horizon=144
                            forecast_price = pair['forecast_price']
                            try:
                                fp = float(forecast_price)
                                pt = float(pair.get('current_price', 0.0))
                                if np.isfinite(fp) and np.isfinite(pt) and abs(fp) > 1e-6:
                                    denom = max(abs(pt), denom_floor, 1.0)
                                    mape_long = abs(current_price_raw - fp) / denom
                                    self._horizon_mape['long'].append(float(np.clip(mape_long, 0.0, 1.0)))
                            except Exception:
                                pass
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
                    denom_at_forecast_time = float(forecast_entry.get('return_denom', max(abs(price_at_forecast_time), denom_floor, 1.0)))
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
                                    denom_for_check = float(forecast_entry.get('return_denom', max(abs(price_at_t_minus_6), denom_floor, 1.0)))
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

                # Update calibration tracker for MEDIUM and LONG horizons as well (Tier3/FGB only).
                # Keep the detailed debug check above for short; medium/long updates are lightweight.
                if self.calibration_tracker is not None:
                    try:
                        current_price_raw = float(self._price_raw[i])
                        for horizon_name, horizon_steps, z_key in [
                            ("medium", horizon_medium, "z_medium"),
                            ("long", horizon_long, "z_long"),
                        ]:
                            needed = int(horizon_steps) + 1
                            if len(self._forecast_history) < needed:
                                continue

                            forecast_entry = self._forecast_history[-needed]
                            forecast_signal = float(forecast_entry.get(z_key, 0.0))

                            price_at_forecast_time = float(forecast_entry.get('price_raw', 0.0))
                            realized_delta_raw = current_price_raw - price_at_forecast_time

                            denom_at_forecast_time = float(
                                forecast_entry.get('return_denom', max(abs(price_at_forecast_time), denom_floor, 1.0))
                            )
                            realized_return = realized_delta_raw / max(denom_at_forecast_time, 1e-6)
                            if ret_clip > 0:
                                realized_return = float(np.clip(realized_return, -ret_clip, ret_clip))
                            realized_signal = float(np.tanh(realized_return * tanh_scale))

                            self.calibration_tracker.update(
                                forecast=float(forecast_signal),
                                realized=float(realized_signal),
                                horizon=str(horizon_name),
                            )
                    except Exception as e:
                        if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
                            raise RuntimeError(f"[CALIBRATION] Medium/long horizon update failed at t={i}: {e}") from e
                        # Calibration is best-effort unless fail-fast is enabled.
                        pass

            except Exception as e:
                if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
                    raise RuntimeError(f"[CALIBRATION] Update failed at t={i}: {e}") from e
                if i % 1000 == 0:
                    logger.warning(f"[CALIBRATION] Update failed at t={i}: {e}")
                import traceback
                if i % 1000 == 0:
                    logger.warning(f"[CALIBRATION] Traceback: {traceback.format_exc()}")

    def _build_overlay_features(self, i: int) -> Optional[np.ndarray]:
        """
        Build features for DL overlay inference (18D short-horizon only).
        Returns shape (1, 18) for batch processing.

        Features (18D):
        - Market (0-5): [price_n, wind_n, solar_n, hydro_n, load_n, risk]
        - Positions (6-8): [wind_pos, solar_pos, hydro_pos] (normalized by fund size)
        - Forecasts short (9-12): [wind, solar, hydro, price] for 1h horizon only
        - Portfolio (13-15): [capital_allocation, investment_freq, forecast_confidence]
        - Deltas (16-17): [z_short, |z_short|] short-horizon price delta
        """
        try:
            feature_dim = getattr(self, 'feature_dim', OVERLAY_FEATURE_DIM)
            if feature_dim != OVERLAY_FEATURE_DIM:
                raise ValueError(f"_build_overlay_features expects {OVERLAY_FEATURE_DIM}D (short-only), got {feature_dim}D")

            overlay_h = str(getattr(self.config, 'overlay_horizon', 'short') or 'short').lower()

            if self.t == 0:
                logger.info(f"[OVERLAY_FEATURES] Building {OVERLAY_FEATURE_DIM}D features ({overlay_h}-horizon only)")

            features = np.zeros((1, OVERLAY_FEATURE_DIM), dtype=np.float32)
            price_raw = float(self._price_raw[i] if i < len(self._price_raw) else 0.0)

            # Market (0-5)
            features[0, 0] = self._norm_price(price_raw, i)
            features[0, 1] = float(np.clip(self._wind[i] / max(self.wind_scale, 1e-6) if i < len(self._wind) else 0.0, 0.0, 1.0))
            features[0, 2] = float(np.clip(self._solar[i] / max(self.solar_scale, 1e-6) if i < len(self._solar) else 0.0, 0.0, 1.0))
            features[0, 3] = float(np.clip(self._hydro[i] / max(self.hydro_scale, 1e-6) if i < len(self._hydro) else 0.0, 0.0, 1.0))
            features[0, 4] = float(np.clip(self._load[i] / max(self.load_scale, 1e-6) if i < len(self._load) else 0.0, 0.0, 1.0))
            features[0, 5] = float(self._riskS[i] if i < len(self._riskS) else 0.3)

            # Positions (6-8)
            fund_size = max(self.init_budget, 1e6)
            features[0, 6] = float(self.financial_positions.get('wind_instrument_value', 0.0) / fund_size)
            features[0, 7] = float(self.financial_positions.get('solar_instrument_value', 0.0) / fund_size)
            features[0, 8] = float(self.financial_positions.get('hydro_instrument_value', 0.0) / fund_size)

            # Forecasts short only (9-12) - single horizon aligned with investment_freq=6
            wind_scale = max(self.wind_scale, 1e-6)
            solar_scale = max(self.solar_scale, 1e-6)
            hydro_scale = max(self.hydro_scale, 1e-6)
            wind_current = float(np.clip(self._wind[i] / wind_scale if i < len(self._wind) else 0.0, 0.0, 1.0))
            solar_current = float(np.clip(self._solar[i] / solar_scale if i < len(self._solar) else 0.0, 0.0, 1.0))
            hydro_current = float(np.clip(self._hydro[i] / hydro_scale if i < len(self._hydro) else 0.0, 0.0, 1.0))

            def get_forecast(target: str, fallback: float) -> float:
                array_name = f"_{target}_forecast_{overlay_h}"
                if hasattr(self, array_name) and i < len(getattr(self, array_name)):
                    val = getattr(self, array_name)[i]
                    if not np.isnan(val):
                        return float(val)
                return fallback

            wind_fc = float(np.clip(get_forecast("wind", wind_current * wind_scale) / wind_scale, 0.0, 1.0))
            solar_fc = float(np.clip(get_forecast("solar", solar_current * solar_scale) / solar_scale, 0.0, 1.0))
            hydro_fc = float(np.clip(get_forecast("hydro", hydro_current * hydro_scale) / hydro_scale, 0.0, 1.0))
            price_fc_raw = get_forecast("price", price_raw)
            price_fc = self._norm_price(price_fc_raw, i)

            features[0, 9] = wind_fc
            features[0, 10] = solar_fc
            features[0, 11] = hydro_fc
            features[0, 12] = price_fc

            # Portfolio (13-15)
            features[0, 13] = float(self.capital_allocation_fraction)
            features[0, 14] = float(self.investment_freq / 100.0)
            features[0, 15] = float(self._get_forecast_confidence())

            # Deltas short only (16-17): z_short, |z_short|
            delta_price_raw = price_fc_raw - price_raw
            ema_std = getattr(self, 'ema_std_short', getattr(self, 'ema_std_medium', 10.0))
            z_short = np.tanh(delta_price_raw / (float(ema_std) + 1e-6))
            features[0, 16] = float(np.clip(z_short, -1.0, 1.0))
            features[0, 17] = float(np.clip(abs(z_short), 0.0, 1.0))

            if i % 500 == 0 and i > 0:
                logger.debug(f"[overlay/deltas] Step {i}: Δprice_{overlay_h}={delta_price_raw:.2f} z_{overlay_h}={z_short:.3f}")

            return features

        except Exception as e:
            if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
                raise RuntimeError(f"[OVERLAY_FEATURES] Error building overlay features at step {i}: {e}") from e
            logger.debug(f"[OVERLAY_FEATURES] Error building overlay features at step {i}: {e}")
            return None

    def _collect_overlay_experience(self, i: int, features: Optional[np.ndarray] = None):
        """
        Collect experience for overlay model training.

        IMPORTANT (paper-clean): `features` should be the PRE-ACTION state features for step i.
        The outcome/targets are realized post-action, but the model itself is state-only.
        """
        try:
            # Get features (prefer pre-action features from step()).
            if features is None:
                features = self._build_overlay_features(i)
            if features is None:
                return
            if isinstance(features, np.ndarray) and features.ndim == 1:
                features = features.reshape(1, -1)

            # Store in history for target computation
            self._overlay_feature_history.append(features[0])  # Remove batch dimension

            # Compute targets based on recent history
            if len(self._overlay_feature_history) < 5:
                return  # Need some history to compute meaningful targets

            # FGB/FAMC-only overlay training:
            # Only train what is needed for baseline correction, i.e., a predicted reward proxy.
            pred_reward_target = self._compute_pred_reward_target()

            experience = {
                'features': features[0].astype(np.float32),
                'pred_reward_target': float(pred_reward_target),
                'outcome': float(getattr(self, 'last_realized_dnav', 0.0)),
                'timestamp': self.t
            }

            # Add to trainer buffer
            self.overlay_trainer.add_experience(experience)

            # DEBUG: Log experience collection every 500 steps
            if self.t % 500 == 0:
                buffer_size = self.overlay_trainer.buffer.size()
                logger.info(f"[OVERLAY_EXPERIENCE] t={self.t} | buffer_size={buffer_size} | pred_reward_target={pred_reward_target:.3f}")

        except Exception as e:
            if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
                raise RuntimeError(f"[OVERLAY_EXPERIENCE] Failed to collect experience at t={self.t}: {e}") from e
            logger.debug(f"Experience collection failed: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _compute_pred_reward_target(self) -> float:
        """
        Overlay supervision target for pred_reward (FGB/FAMC overlay only).

        Paper-clean: outcome-aligned.
        We supervise pred_reward using realized step dNAV for the current step
        (computed after execution), normalized by an exposure proxy consistent with
        CalibrationTracker.expected_dnav().

        Returns:
            float in [-1, 1]
        """
        try:
            # Realized incremental NAV change (DKK) from this step (set in _update_finance).
            realized_dnav = float(getattr(self, 'last_realized_dnav', 0.0))
            if not np.isfinite(realized_dnav):
                realized_dnav = 0.0

            # Exposure proxy: sum absolute notional across financial instruments (DKK).
            positions = getattr(self, 'financial_positions', {}) or {}
            exposure = 0.0
            for v in positions.values():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    exposure += abs(fv)

            init_budget = float(getattr(self, 'init_budget', 0.0) or 0.0)
            if exposure < 1e-6:
                exposure = 0.01 * max(init_budget, 1.0)

            # dNAV return proxy (unitless) for stable training.
            realized_return = float(realized_dnav / max(exposure, 1.0))

            # Squash to [-1, 1] for the tanh head.
            clip_r = 0.20
            scale = 20.0
            if self.config is not None:
                clip_r = float(
                    getattr(
                        self.config,
                        'overlay_dnav_return_clip',
                        getattr(self.config, 'overlay_mtm_return_clip', clip_r),
                    )
                    or clip_r
                )
                scale = float(
                    getattr(
                        self.config,
                        'overlay_dnav_return_tanh_scale',
                        getattr(self.config, 'overlay_mtm_return_tanh_scale', scale),
                    )
                    or scale
                )
            if clip_r > 0.0:
                realized_return = float(np.clip(realized_return, -clip_r, clip_r))
            pred_target = float(np.tanh(realized_return * scale))

            # Lightweight diagnostics.
            if self.t % 500 == 0 and self.t > 0:
                try:
                    logger.debug(
                        f"[PRED_REWARD_TARGET] t={self.t} dnav={realized_dnav:,.0f} "
                        f"exposure={exposure:,.0f} ret={realized_return:+.5f} target={pred_target:+.3f}"
                    )
                except Exception:
                    pass

            return float(np.clip(pred_target, -1.0, 1.0))

        except Exception as e:
            if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
                raise RuntimeError(f"[PRED_REWARD_TARGET] computation failed at t={self.t}: {e}") from e
            logger.debug(f'Pred reward target computation failed: {e}')
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
        enable_forecast_util = False
        horizon_correlations = getattr(self, '_horizon_correlations', None)
        price_returns = FinancialEngine.calculate_price_returns(
            timestep=i, current_price=current_price, price_history=self._price_raw,
            enable_forecast_util=enable_forecast_util, horizon_correlations=horizon_correlations
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
            if i <= 1:
                accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
                financial_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
                logger.info(f"[TIER1_UPDATE_FINANCE] Calculating NAV in _update_finance at i={i}:")
                logger.info(f"  i parameter = {i}")
                logger.info(f"  self.t = {getattr(self, 't', 0)}")
                logger.info(f"  budget = {self.budget:,.0f} DKK")
                logger.info(f"  physical_assets = {self.physical_assets}")
                logger.info(f"  accumulated_operational_revenue = {accumulated_operational_revenue:,.0f} DKK")
                logger.info(f"  financial_positions MTM = {financial_mtm:,.0f} DKK")
            fund_nav = self._calculate_fund_nav(current_timestep=i)
            if i <= 1:
                # Calculate components to see what's different
                accumulated_operational_revenue = getattr(self, 'accumulated_operational_revenue', 0.0)
                financial_mtm = sum(getattr(self, 'financial_mtm_positions', self.financial_positions).values())
                # Calculate physical assets from NAV
                physical_from_nav = fund_nav - self.budget - accumulated_operational_revenue - financial_mtm
                logger.info(f"[TIER1_UPDATE_FINANCE] NAV calculated at i={i}: {fund_nav:,.0f} DKK (${fund_nav * 0.145 / 1_000_000:.2f}M)")
                logger.info(f"  NAV breakdown: budget={self.budget:,.0f}, physical={physical_from_nav:,.0f}, operational={accumulated_operational_revenue:,.0f}, mtm={financial_mtm:,.0f}")
                logger.info(f"  current_timestep passed to _calculate_fund_nav = {i}")

            # 9) Track performance
            self.performance_history['revenue_history'].append(net_cash_flow)
            self.performance_history['generation_revenue_history'].append(generation_revenue)
            self.performance_history['nav_history'].append(fund_nav)

            # Realized step dNAV for overlay supervision (pred_reward target).
            prev_nav_for_overlay = getattr(self, '_overlay_prev_nav', None)
            if prev_nav_for_overlay is None:
                realized_dnav = 0.0
                realized_dnav_return = 0.0
            else:
                try:
                    prev_nav_for_overlay = float(prev_nav_for_overlay)
                except Exception:
                    prev_nav_for_overlay = 0.0
                if (not np.isfinite(prev_nav_for_overlay)) or prev_nav_for_overlay <= 0.0:
                    realized_dnav = 0.0
                    realized_dnav_return = 0.0
                else:
                    realized_dnav = float(fund_nav - prev_nav_for_overlay)
                    realized_dnav_return = float(realized_dnav / max(prev_nav_for_overlay, 1.0))

            self.last_realized_dnav = float(realized_dnav)
            self.last_realized_dnav_return = float(realized_dnav_return)
            self._overlay_prev_nav = float(fund_nav)

            # 10) Store values for logging/rewards
            self.last_revenue = net_cash_flow
            self.last_generation_revenue = generation_revenue
            self.last_mtm_pnl = mtm_pnl


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
            forecast_gate_passed = False
            forecast_used_flag = False
            forecast_usage_reason = "disabled"
            self._debug_forecast_reward = {
                "forecast_used": False,
                "forecast_not_used_reason": forecast_usage_reason,
                "forecast_gate_passed": False,
            }

            # Tier-2 forecast engine deleted (paper refactor).
            # Keep default "disabled" payload so downstream debug logging is stable.

            # NOTE: Do NOT force forecast_used_flag to True here.
            # We keep the engine's decision (or the default False) so usage stats
            # reflect actual gate/usage, not observation availability.
            # If you want the old behavior, wrap the block below in a config flag.
            # (No action needed; just rely on the values already set above.)

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
                # FIX: Add trading_gains and operating_gains for debug logging
                'trading_gains': getattr(self, 'cumulative_mtm_pnl', 0.0),
                'operating_gains': getattr(self, 'cumulative_generation_revenue', 0.0),
            }

        except Exception as e:
            msg = f"[FINANCE_UPDATE_FATAL] Finance update error at t={self.t}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

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

            # Only keep basic administration fees like PrototypeTestTuned
            administration_fees = self.init_budget * self.administration_fee_rate * annual_to_timestep

            total_admin_costs = administration_fees
            return float(total_admin_costs)

        except Exception as e:
            msg = f"[ADMIN_COST_FATAL] Fund administration cost calculation failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

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

            # Minimal anti-saturation regularizer (warmup):
            # Penalize large-magnitude action vectors to discourage always-at-bounds allocations.
            # This targets the observed failure mode: PPO collapsing to constant corner actions in ep1+.
            investor_action_penalty = 0.0
            investor_action_boundary_penalty = 0.0
            investor_exposure_penalty = 0.0
            investor_exposure_stuck_penalty = 0.0
            warm = 0.0
            try:
                w = float(getattr(self.config, 'investor_action_l2_penalty', 0.0))
                warm_steps = max(1, int(getattr(self.config, 'investor_action_penalty_warmup_steps', 2000)))
                # IMPORTANT: use config.training_global_step so warmup does not reset at episode boundaries
                t_now = int(getattr(self.config, 'training_global_step', getattr(self, 't', 0))) if self.config is not None else int(getattr(self, 't', 0))
                warm = float(np.clip(t_now / warm_steps, 0.0, 1.0))
                # IMPORTANT: penalize the executed (post-simplex) action, not the raw policy output.
                a = getattr(self, '_last_actions', {}).get('investor_0_exec', getattr(self, '_last_actions', {}).get('investor_0', {}))
                vec = None
                if isinstance(a, dict):
                    vec = np.array([a.get('wind', 0.0), a.get('solar', 0.0), a.get('hydro', 0.0)], dtype=np.float32)
                else:
                    arr = np.array(a, dtype=np.float32).reshape(-1)
                    if arr.size >= 3:
                        vec = arr[:3].astype(np.float32)
                    else:
                        vec = np.pad(arr, (0, 3 - arr.size), constant_values=0.0).astype(np.float32)

                l2 = float(np.mean(vec * vec))
                investor_action_penalty = float(warm * w * l2)

                # Boundary-only penalty (novel surgical): punish only near-saturation, not interior exploration.
                bw = float(getattr(self.config, 'investor_action_boundary_penalty', 0.0))
                thr = float(getattr(self.config, 'investor_action_boundary_threshold', 0.85))
                if bw > 0.0:
                    excess = np.maximum(np.abs(vec) - thr, 0.0)
                    investor_action_boundary_penalty = float(warm * bw * float(np.mean(excess * excess)))

                # Exposure-only hinge penalty (targets new collapse mode: exposure pegged at 1.0 with fixed weights).
                # Uses executed exposure if available; falls back to sum(abs(vec)) (which equals exposure under signed-simplex).
                ew = float(getattr(self.config, 'investor_action_exposure_penalty', 0.0))
                eth = float(getattr(self.config, 'investor_action_exposure_threshold', 0.0))
                if ew > 0.0:
                    exposure_val = None
                    if isinstance(a, dict):
                        exposure_val = a.get('exposure', None)
                    if exposure_val is None:
                        exposure_val = float(np.sum(np.abs(vec)))
                    exposure_val = float(np.clip(exposure_val, 0.0, 1.0))
                    excess_exp = float(max(exposure_val - eth, 0.0))
                    investor_exposure_penalty = float(warm * ew * (excess_exp ** 2))

                    # Collapse-breaker: if exposure stays near 1.0 for many consecutive steps,
                    # ramp a penalty up to a fixed cap to force exploration away from the saturated fixed point.
                    stuck_thr = float(getattr(self.config, 'investor_exposure_stuck_threshold', 0.95))
                    stuck_steps = max(1, int(getattr(self.config, 'investor_exposure_stuck_steps', 2000)))
                    stuck_w = float(getattr(self.config, 'investor_exposure_stuck_penalty', 0.0))
                    if stuck_w > 0.0:
                        if not hasattr(self, '_exposure_stuck_count'):
                            self._exposure_stuck_count = 0
                        if exposure_val >= stuck_thr:
                            self._exposure_stuck_count = int(self._exposure_stuck_count) + 1
                        else:
                            self._exposure_stuck_count = 0
                        ramp = float(np.clip(float(self._exposure_stuck_count) / float(stuck_steps), 0.0, 1.0))
                        # Penalize squared exposure (0..1) scaled by ramp; at exposure=1, penalty ramps to stuck_w.
                        investor_exposure_stuck_penalty = float(warm * stuck_w * ramp * (exposure_val ** 2))
            except Exception:
                investor_action_penalty = 0.0
                investor_action_boundary_penalty = 0.0
                investor_exposure_penalty = 0.0
                investor_exposure_stuck_penalty = 0.0
                warm = 0.0

            # Store minimal penalty diagnostics for CSV logging / deep debugging
            try:
                if self.config is not None:
                    self._last_penalty_diag = {
                        'training_global_step': int(getattr(self.config, 'training_global_step', 0)),
                        'inv_penalty_warm': float(warm),
                        'inv_pen_boundary': float(investor_action_boundary_penalty),
                        'inv_pen_exposure': float(investor_exposure_penalty),
                        'inv_pen_exposure_stuck': float(investor_exposure_stuck_penalty),
                    }
            except Exception:
                self._last_penalty_diag = {'training_global_step': 0, 'inv_penalty_warm': 0.0, 'inv_pen_boundary': 0.0, 'inv_pen_exposure': 0.0, 'inv_pen_exposure_stuck': 0.0}
            
            # Forecast risk management removed.
            investor_strategy_delta = 0.0
            position_alignment_status = 'disabled'

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
            
            agent_rewards['investor_0'] = (
                investor_base
                + investor_mtm_delta
                + investor_strategy_delta
                + investor_nav_delta
                - investor_action_penalty
                - investor_action_boundary_penalty
                - investor_exposure_penalty
                - investor_exposure_stuck_penalty
            )
            
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
            
            # Forecast risk management removed.
            battery_forecast_delta = 0.0
            
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
                inv_freq = int(getattr(self, 'investment_freq', 1) or 1)
                if inv_freq > 0 and i - inv_freq >= 0 and hasattr(self, '_price_raw') and i < len(self._price_raw):
                    prev_p = float(self._price_raw[i - inv_freq])
                    curr_p = float(self._price_raw[i])
                    ret_h = (curr_p - prev_p) / max(abs(prev_p), 1e-6)
                    info_investor['price_return_invfreq'] = float(np.clip(ret_h, -0.5, 0.5))
                else:
                    info_investor['price_return_invfreq'] = 0.0
            except Exception:
                info_investor['price_return_invfreq'] = 0.0
            info['investor_0'] = info_investor

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
                usd_rate = float(getattr(self, 'usd_conversion_rate', getattr(self.config, 'usd_conversion_rate', 0.145)))
                fund_nav_dkk = float(fund_nav)
                current_fund_nav_usd = float(fund_nav_dkk * usd_rate)
                current_price_raw = float(self._price_raw[current_timestep] if current_timestep < len(self._price_raw) else 0.0)
                current_cash_dkk = float(getattr(self, 'budget', 0.0))
                current_trading_gains_usd = float(self.reward_calculator.recent_trading_gains if hasattr(self.reward_calculator, 'recent_trading_gains') else 0.0)
                current_operating_gains_usd = float(accumulated_operational_revenue_dkk * usd_rate / 1000.0)
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

            obs_trade_action_corr = 0.0
            obs_trade_exposure_corr = 0.0
            obs_trade_delta_exposure_corr = 0.0

            decision_step = 1.0 if (self.t > 0 and self.t % self.investment_freq == 0) else 0.0
            exposure_exec = 0.0
            try:
                if isinstance(getattr(self, "_last_actions", None), dict):
                    exec_action = self._last_actions.get("investor_0_exec", {})
                    exposure_exec = float(exec_action.get("exposure", 0.0))
            except Exception:
                exposure_exec = 0.0

            action_sign = float(np.sign(exposure_exec)) if abs(exposure_exec) > 1e-9 else float(np.sign(exposure_val))
            trade_signal_active = 0.0
            trade_signal_sign = 0.0

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
                    # IMPORTANT: log the *real backend* z-scores/trust (not the retired Tier-2 debug dict).
                    z_short=float(getattr(self, 'z_short', 0.0)),
                    z_medium=float(getattr(self, 'z_medium', 0.0)),
                    z_long=float(getattr(self, 'z_long', 0.0)),
                    z_combined=float(getattr(self, 'mwdir', getattr(self, 'z_combined', 0.0))),
                    forecast_confidence=float(self._get_forecast_confidence()) if getattr(self, 'forecast_generator', None) is not None else 0.0,
                    forecast_trust=float(getattr(self, '_forecast_trust', 0.0)),
                    # OBSERVATION-LEVEL forecast signals (what the POLICY actually sees after warmup/ablation)
                    obs_z_short=float(getattr(self, '_last_obs_z_short', 0.0)),
                    obs_z_long=float(getattr(self, '_last_obs_z_long', 0.0)),
                    obs_forecast_trust=float(getattr(self, '_last_obs_forecast_trust', 0.0)),
                    obs_normalized_error=float(getattr(self, '_last_obs_normalized_error', 0.0)),
                    obs_trade_signal=float(getattr(self, '_last_obs_trade_signal', 0.0)),
                    obs_trade_action_corr=float(obs_trade_action_corr),
                    obs_trade_exposure_corr=float(obs_trade_exposure_corr),
                    obs_trade_delta_exposure_corr=float(obs_trade_delta_exposure_corr),
                    signal_gate_multiplier=float(debug_forecast.get('signal_gate_multiplier', 0.0)),
                    # Position info (ensure floats)
                    position_signed=float(debug_forecast.get('position_signed', 0.0)),
                    # FIX: debug_forecast.position_exposure can be stale/zero. Recompute from current financial notional.
                    position_exposure=float(
                        np.clip(
                            (
                                sum(abs(float(v)) for v in getattr(self, 'financial_positions', {}).values())
                                / max(
                                    float(getattr(self.config, 'max_position_size', 0.0))
                                    * float(self.init_budget)
                                    * float(getattr(self.config, 'capital_allocation_fraction', 0.0)),
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
                    vol_brake_mult=float(getattr(self, "_last_vol_brake_mult", 1.0)),
                    strategy_multiplier=float(getattr(self, "_last_strategy_multiplier", 1.0)),
                    combined_multiplier=float(getattr(self, "_last_combined_multiplier", 1.0)),
                    tradeable_capital=float(getattr(self, "_last_tradeable_capital", 0.0)),
                    mtm_exit_count=float(getattr(self, "_last_mtm_exit_count", 0.0)),
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
                    enable_forecast_util=False,
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
                    # fund_nav = portfolio_value_usd_millions (just different units)
                    # trading_gains = trading_gains_usd_thousands (just different units)
                    # operating_gains = operating_gains_usd_thousands (just different units)
                    mtm_pnl=float(mtm_pnl),  # Per-step MTM PnL (not cumulative, different from trading_gains)
                    # Penalty diagnostics (anti-collapse verification)
                    training_global_step=int(getattr(self.config, 'training_global_step', 0)) if self.config is not None else 0,
                    inv_penalty_warm=float(getattr(self, '_last_penalty_diag', {}).get('inv_penalty_warm', 0.0)),
                    inv_pen_boundary=float(getattr(self, '_last_penalty_diag', {}).get('inv_pen_boundary', 0.0)),
                    inv_pen_exposure=float(getattr(self, '_last_penalty_diag', {}).get('inv_pen_exposure', 0.0)),
                    inv_pen_exposure_stuck=float(getattr(self, '_last_penalty_diag', {}).get('inv_pen_exposure_stuck', 0.0)),
                    # Investor policy distribution diagnostics (pre-tanh / pre-postprocess)
                    inv_mu_raw=float(getattr(self, '_inv_mu_raw', 0.0)),
                    inv_sigma_raw=float(getattr(self, '_inv_sigma_raw', 0.0)),
                    inv_a_raw=float(getattr(self, '_inv_a_raw', 0.0)),
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
    # Observations (Tier1-only; no forecast-augmented obs)
    # ------------------------------------------------------------------
    def _fill_obs(self):
        """Fill all agent observations.

        Investor (6D) + battery (4D) are built by ObservationBuilder.
        Risk (9D) + meta (11D) are populated here (kept simple and stable).
        """
        try:
            from observation_builder import ObservationBuilder

            i = min(self.t, self.max_steps - 1)

            market_features = ObservationBuilder.normalize_market_features(
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

            # Investor (6D)
            inv = self._obs_buf["investor_0"]
            ObservationBuilder.build_investor_observations(
                obs_array=inv,
                price_n=price_n,
                budget=self.budget,
                init_budget=self.init_budget,
                financial_positions=self.financial_positions,
                cumulative_mtm_pnl=getattr(self, "cumulative_mtm_pnl", 0.0),
                max_position_size=self.config.max_position_size,
                # Use meta-controlled capital allocation (what actually sizes trades) for consistent normalization.
                capital_allocation_fraction=float(
                    getattr(self, "capital_allocation_fraction", self.config.capital_allocation_fraction)
                ),
            )

            if self.t % 1000 == 0 or self.t == 0:
                logger.info(f"[INVESTOR_OBS] t={self.t} | price={inv[0]:.3f}, budget={inv[1]:.3f}")
                logger.info(
                    f"  Positions: wind={inv[2]:.3f}, solar={inv[3]:.3f}, hydro={inv[4]:.3f}, mtm_pnl={inv[5]:.3f}"
                )

            # Battery (4D)
            batt = self._obs_buf["battery_operator_0"]
            ObservationBuilder.build_battery_observations(
                obs_array=batt,
                price_n=price_n,
                battery_energy=self.operational_state["battery_energy"],
                battery_capacity_mwh=self.physical_assets["battery_capacity_mwh"],
                load_n=load_n,
            )

            if self.t % 1000 == 0 or self.t == 0:
                logger.info(
                    f"[BATTERY_OBS] t={self.t} | price={batt[0]:.3f}, energy={batt[1]:.3f}, capacity={batt[2]:.3f}, load={batt[3]:.3f}"
                )

            # Risk controller (9D)
            rsk = self._obs_buf["risk_controller_0"]
            rsk[:9] = (
                price_n,
                float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
                float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.financial_positions["wind_instrument_value"], self.init_budget), -1.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.financial_positions["solar_instrument_value"], self.init_budget), -1.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.financial_positions["hydro_instrument_value"], self.init_budget), -1.0, 1.0)) * 10.0,
                float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
                float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0)),
                float(np.clip(self.risk_multiplier, 0.0, 2.0)) * 5.0,
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
    def get_fgb_trust_for_agent(self, agent_name: str) -> float:
        """
        FGB/FAMC: Return per-agent forecast trust (tau_t) used to scale the control variate.

        IMPORTANT:
        - This does NOT change environment observations or rewards (Tier1 remains intact).
        - It is only consumed inside the learner (metacontroller) for advantage correction.
        """
        name = str(agent_name or "").strip()
        # Forecast ablation: force tau=0 so the control variate is effectively disabled,
        # while keeping the rest of the plumbing (overlay, cache, logging) identical.
        try:
            if bool(getattr(self.config, "fgb_ablate_forecasts", False)):
                return 0.0
        except Exception:
            pass

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

        # Horizon preference per agent (default: auto; investor fixed to medium via config).
        pref = "auto"
        try:
            pref_map = getattr(self.config, "fgb_trust_horizon_by_agent", {})
            if isinstance(pref_map, dict):
                pref = pref_map.get(name, "auto")
        except Exception:
            pref = "auto"
        pref = str(pref or "auto").strip().lower()

        valid_horizons = ("short", "medium", "long")
        try:
            min_samples = int(getattr(self.config, "fgb_trust_min_samples", 50) or 50)
        except Exception:
            min_samples = 50

        # Auto-select the best horizon by recent directional hit-rate (if enough samples).
        horizon = pref if pref in valid_horizons else "auto"
        if horizon == "auto":
            best_h = None
            best_score = -1.0
            hits = getattr(self, "_horizon_sign_hit", {}) or {}
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

            if best_h is None:
                # Warmup fallback: pick the horizon with most samples (short matures first).
                counts = []
                for idx, h in enumerate(valid_horizons):
                    try:
                        counts.append((len(list(hits.get(h, []))), -idx, h))
                    except Exception:
                        counts.append((0, -idx, h))
                counts.sort(reverse=True)
                horizon = counts[0][2] if counts else "short"
            else:
                horizon = best_h

        # Compute trust.
        trust_val: float
        overlay_enabled = bool(getattr(self.config, "overlay_enabled", False)) if self.config is not None else False
        tracker = getattr(self, "calibration_tracker", None)
        if overlay_enabled and tracker is not None:
            # Tier2/Tier3 (overlay enabled): use CalibrationTracker trust for the selected horizon.
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

    def _compute_forecast_signals(self):
        """
        FGB: Compute expected ΔNAV for baseline adjustment (control variate only).

        These signals are used by metacontroller to adjust PPO rewards:
            r'_t = r_t - λ · τₜ · E[ΔNAV | s_t]

        This reduces advantage variance and improves convergence without blending actions.
        """
        try:
            # Initialize to zero (no-op if overlay disabled / inference missing)
            self._expected_dnav = 0.0

            # Forecast ablation: keep the baseline machinery on, but force the control variate to 0.
            # This makes FGB/FAMC effectively a no-op while preserving all other code paths.
            try:
                if bool(getattr(self.config, "fgb_ablate_forecasts", False)):
                    self._forecast_trust = 0.0
                    self._expected_dnav = 0.0
                    return
            except Exception:
                pass

            # Only compute if calibration tracker is available
            if self.calibration_tracker is None:
                return

            # Trust (tau_t) is updated in _compute_forecast_deltas(update_calibration=True).
            # Fall back to calibration tracker if missing/invalid.
            try:
                tau = float(getattr(self, "_forecast_trust", 0.0))
            except Exception:
                tau = 0.0
            if (not np.isfinite(tau)) or tau < 0.0:
                tau = 0.0
            if tau == 0.0:
                try:
                    # Use the configured per-agent trust horizon (investor_0 uses "short" for investment_freq=6).
                    self._forecast_trust = float(self.get_fgb_trust_for_agent("investor_0"))
                except Exception:
                    try:
                        self._forecast_trust = float(self.calibration_tracker.get_trust(horizon="short"))
                    except Exception:
                        self._forecast_trust = 0.0

                # Apply trust floor (same as in step() path)
                trust_floor = float(getattr(self.config, 'forecast_trust_floor', 0.0) or 0.0)
                if trust_floor > 0.0:
                    self._forecast_trust = float(np.maximum(self._forecast_trust, trust_floor))

            # Compute expected ΔNAV if we have overlay output.
            # IMPORTANT: expected_dnav needs mwdir; attach it explicitly (overlay outputs do not include mwdir by default).
            if self._last_overlay_output:
                overlay_out = dict(self._last_overlay_output)
                overlay_out["mwdir"] = float(getattr(self, "mwdir", 0.0))
                # Snapshot positions so this remains a pre-action, state-only signal for this step.
                positions_snapshot = dict(getattr(self, "financial_positions", {}) or {})
                self._expected_dnav = float(self.calibration_tracker.expected_dnav(
                    overlay_output=overlay_out,
                    positions=positions_snapshot,
                    costs={
                        'bps': self.config.transaction_cost_bps,
                        'fixed': self.config.transaction_fixed_cost
                    }
                ))
            # NOTE: Risk uplift removed. Forecasts are used for FGB/FAMC variance reduction only,
            # and do not directly alter position sizing or reward.

        except Exception as e:
            msg = f"[FGB] Forecast signal computation failed: {e}"
            if bool(getattr(self.config, "fgb_fail_fast", False)) and bool(getattr(self.config, "forecast_baseline_enable", False)):
                raise RuntimeError(msg) from e
            logger.warning(msg)

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
                    'price_return_1step': float(self._price_return_1step[i]) if hasattr(self, '_price_return_1step') and i < len(self._price_return_1step) else 0.0,

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


