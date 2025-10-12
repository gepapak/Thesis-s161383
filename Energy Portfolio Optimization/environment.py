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
import logging, os, gc
try:
    import psutil as _psutil
except Exception:
    _psutil = None

from risk import EnhancedRiskController


# =============================================================================
# Utilities
# =============================================================================
class SafeDivision:
    @staticmethod
    def div(n: float, d: float, default: float = 0.0) -> float:
        try:
            if d is None or abs(d) < 1e-9:
                return default
            return float(n) / float(d)
        except Exception:
            return default


class LightweightMemoryManager:
    """Very light cache/memory watchdog used by the env."""
    def __init__(self, max_memory_mb: float = 1500.0):
        self.max_memory_mb = max_memory_mb
        self._history = deque(maxlen=256)

    def current_mb(self) -> float:
        try:
            if _psutil is None:
                return 0.0
            return _psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def should_cleanup(self, force: bool = False) -> bool:
        cur = self.current_mb()
        self._history.append(cur)
        return force or (cur > self.max_memory_mb * 0.9)

    def cleanup_if_needed(self, force: bool = False) -> bool:
        if self.should_cleanup(force):
            gc.collect()
            return True
        return False


# =============================================================================
# Observation specs (BASE-dim only; wrapper appends forecasts)
# =============================================================================
class StabilizedObservationManager:
    def __init__(self, env: 'RenewableMultiAgentEnv'):
        self.env = env
        self.observation_specs = self._build_specs()
        self.base_spaces = self._build_spaces()

    def _build_specs(self) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        # BASE dims are fixed; wrapper will add forecast features to reach model totals
        specs["investor_0"]         = {"base": 6}   # wind, solar, hydro, price_n, load_n, budget_n
        specs["battery_operator_0"] = {"base": 4}   # price_n, batt_energy, batt_capacity, load_n
        specs["risk_controller_0"]  = {"base": 9}   # regime cues + positions + knobs
        specs["meta_controller_0"]  = {"base": 11}  # budget, positions, price_n, risks, perf, knobs
        return specs

    def _build_spaces(self) -> Dict[str, spaces.Box]:
        sp: Dict[str, spaces.Box] = {}

        # Use per-dimension bounds with price_n in [-1,1] (z-score clipped to [-3,3] then divided by 3)
        inv_low  = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        inv_high = np.array([1.0, 1.0, 1.0,  1.0, 1.0, 10.0], dtype=np.float32)  # budget_n still uses /10 scaling

        bat_low  = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bat_high = np.array([ 1.0, 10.0, 10.0, 1.0], dtype=np.float32)  # price_n [-1,1], others keep existing scales

        # risk: [price_n, vol*10, stress*10, wind_pos_rel*10, solar_pos_rel*10, hydro_pos_rel*10,
        #        cap_frac*10, equity_rel*10, risk_multiplier*5] - price_n now [-1,1]
        risk_low  = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        risk_high = np.array([ 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        # meta: idx4 is price_n which can be negative (now [-1,1])
        meta_low  = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        meta_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        sp["investor_0"]         = spaces.Box(low=inv_low,  high=inv_high,  shape=(6,),  dtype=np.float32)
        sp["battery_operator_0"] = spaces.Box(low=bat_low,  high=bat_high,  shape=(4,),  dtype=np.float32)
        sp["risk_controller_0"]  = spaces.Box(low=risk_low, high=risk_high, shape=(9,),  dtype=np.float32)
        sp["meta_controller_0"]  = spaces.Box(low=meta_low, high=meta_high, shape=(11,), dtype=np.float32)
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
        self.profit_history = deque(maxlen=100)  # Track recent profits

        # REALISTIC: Conservative profitability thresholds for institutional fund
        self.min_acceptable_return = 0.02  # 2% minimum annual return (above risk-free)
        self.target_return_threshold = 0.05  # 5% target return threshold
        self.excellent_return_threshold = 0.08  # 8% excellent return threshold (realistic for renewables)

        # Portfolio tracking (no emergency liquidation for infrastructure fund)
        self.portfolio_peak = float(self.initial_budget)
        self.emergency_liquidation_enabled = False  # Disabled: can't liquidate wind farms instantly
        self.emergency_liquidation_threshold = 0.10  # Only for extreme cases (90% loss)
        self.max_drawdown_threshold = 0.80          # Allow large drawdowns for infrastructure

        # Technology operational volatilities (daily-ish proxies)
        self.operational_vols = {'wind': 0.03, 'solar': 0.025, 'hydro': 0.015}
        self.operational_correlations = {'wind_solar': 0.4, 'wind_hydro': 0.2, 'solar_hydro': 0.3}

        # FIXED: Use reward weights from config to maintain single source of truth
        if config and hasattr(config, 'profit_reward_weight'):
            # Use config-driven reward weights
            self.reward_weights = {
                'operational_revenue': getattr(config, 'profit_reward_weight', 1.0) * 0.4,  # Scale profit weight
                'risk_management': getattr(config, 'risk_penalty_weight', 5.0) * 0.1,      # Scale risk penalty
                'hedging_effectiveness': 0.15,  # Default hedging weight
                'nav_stability': 0.35,          # Remaining weight for stability
                'cash_flow': 0.0,               # Merged into operational_revenue
                'forecast': getattr(config, 'forecast_accuracy_reward_weight', 0.3) * 0.1,  # Scale forecast weight
            }
            # Trading controls from config
            self.forecast_confidence_threshold = getattr(config, 'forecast_confidence_threshold', 0.70)
            self.max_drawdown_threshold = 0.60  # Conservative for learning
        else:
            # Fallback weights if no config available
            self.reward_weights = {
                'operational_revenue': 0.50,    # PRIMARY: Maximize renewable energy revenue
                'risk_management': 0.25,        # SECONDARY: Minimize volatility and drawdowns
                'hedging_effectiveness': 0.15,  # TERTIARY: Reduce price exposure (not profit)
                'nav_stability': 0.10,          # QUATERNARY: Stable fund value
                'cash_flow': 0.0,               # Merged into operational_revenue
                'forecast': 0.0,                # Disabled - focus on fundamentals
            }
            # REALISTIC: Conservative trading controls for institutional fund
            self.forecast_confidence_threshold = 0.70  # 70% confidence threshold for full trading
            self.max_drawdown_threshold = 0.60       # USER SPECIFIED: 60% drawdown threshold (learning mode)
        self.current_drawdown = 0.0
        self.peak_nav = float(self.initial_budget)  # FIXED: Initialize to initial budget, not 0
        self.trading_enabled = True
        self.position_size_multiplier = 1.0  # Full position size initially

    def calculate_reward(self, fund_nav: float, cash_flow: float,
                        risk_level: float, efficiency: float,
                        forecast_signal_score: float = 0.0) -> float:
        """
        IMPROVED: Portfolio-focused reward calculation with drawdown control

        Args:
            fund_nav: Total fund NAV (cash + physical + financial)
            cash_flow: Actual cash received this step (operations)
            risk_level: Current risk level [0,1]
            efficiency: Operational efficiency [0,1]
            forecast_signal_score: Forecast accuracy bonus [-1,1]
        """
        self.portfolio_history.append(fund_nav)
        self.cash_flow_history.append(cash_flow)

        # Update peak NAV and drawdown tracking
        if fund_nav > self.peak_nav:
            self.peak_nav = fund_nav

        self.current_drawdown = (self.peak_nav - fund_nav) / self.peak_nav if self.peak_nav > 0 else 0.0

        # Gradual trading restrictions based on drawdown (LEARNING MODE: More permissive)
        if self.current_drawdown > self.max_drawdown_threshold:  # >60%: Disable trading
            self.trading_enabled = False
        elif self.current_drawdown > 0.45:  # 45-60%: Allow limited trading for recovery
            self.trading_enabled = True
            self.position_size_multiplier = 0.3  # 30% of normal position size
        elif self.current_drawdown > 0.30:  # 30-45%: Reduced trading
            self.trading_enabled = True
            self.position_size_multiplier = 0.6  # 60% of normal position size
        else:  # <30%: Full trading
            self.trading_enabled = True
            self.position_size_multiplier = 1.0  # 100% of normal position size

        if len(self.portfolio_history) < 2:
            return 0.0

        # 1) OPERATIONAL REVENUE COMPONENT (50% weight) - Maximize renewable energy revenue
        recent_cash_flows = list(self.cash_flow_history)[-10:]  # Last 10 steps
        avg_operational_revenue = np.mean(recent_cash_flows) if recent_cash_flows else 0.0
        # Scale to realistic target from config
        # Target from config: operational_revenue_target DKK per 10-min step
        operational_target = getattr(self.config, 'operational_revenue_target', 1200.0) if self.config else 1200.0
        operational_score = float(np.clip(avg_operational_revenue / operational_target, -2.0, 3.0))

        # 2) RISK MANAGEMENT COMPONENT (25% weight) - Minimize volatility and drawdowns
        # Portfolio volatility (last 20 steps)
        if len(self.portfolio_history) >= 20:
            recent_navs_list = list(self.portfolio_history)[-20:]
            recent_navs = np.array(recent_navs_list, dtype=np.float64)

            # FIXED: Ensure proper array slicing
            nav_diff = np.diff(recent_navs)
            nav_base = recent_navs[:-1]
            nav_returns = nav_diff / np.maximum(nav_base, 1.0)
            portfolio_volatility = np.std(nav_returns) if len(nav_returns) > 1 else 0.0
        else:
            portfolio_volatility = 0.0

        # Risk management score: reward low volatility and low drawdowns
        volatility_penalty = float(np.clip(portfolio_volatility * 100.0, 0.0, 2.0))  # Scale volatility
        drawdown_penalty = float(np.clip(self.current_drawdown * 10.0, 0.0, 3.0))   # Penalize drawdowns
        risk_management_score = -(volatility_penalty + drawdown_penalty)  # Negative because we want to minimize

        # 3) HEDGING EFFECTIVENESS COMPONENT (15% weight) - Reduce price exposure, not profit
        hedging_score = self._calculate_hedging_effectiveness()

        # 4) NAV STABILITY COMPONENT (10% weight) - Stable fund value
        if len(self.portfolio_history) >= 2:
            prev_nav = float(self.portfolio_history[-2])
            nav_return = (fund_nav - prev_nav) / max(prev_nav, 1.0)
            # Reward small, positive returns; penalize large swings
            if abs(nav_return) < 0.01:  # Less than 1% change is good
                nav_stability_score = 1.0
            else:
                nav_stability_score = float(np.clip(1.0 - abs(nav_return) * 50.0, -2.0, 1.0))
        else:
            nav_stability_score = 1.0  # No previous NAV to compare

        # OPERATIONAL EXCELLENCE FOCUSED: New reward calculation
        rw = self.reward_weights
        reward = (
            rw['operational_revenue'] * operational_score +
            rw['risk_management'] * risk_management_score +
            rw['hedging_effectiveness'] * hedging_score +
            rw['nav_stability'] * nav_stability_score
        )

        # CONSERVATIVE: Stricter penalties for institutional fund
        if self.current_drawdown > 0.05:  # 5% drawdown penalty
            reward -= 0.5
        if self.current_drawdown > 0.10:  # 10% drawdown penalty
            reward -= 1.5
        if self.current_drawdown > 0.15:  # 15% drawdown penalty
            reward -= 3.0

        # Use config values for reward clipping if available
        if hasattr(self, 'config') and self.config:
            clip_min = getattr(self.config, 'reward_clip_min', -10.0)
            clip_max = getattr(self.config, 'reward_clip_max', 10.0)
        else:
            clip_min, clip_max = -10.0, 10.0  # Fallback values

        return float(np.clip(reward, clip_min, clip_max))  # Configurable clipping

    def _calculate_hedging_effectiveness(self) -> float:
        """
        Calculate hedging effectiveness based on correlation between operational and trading cash flows
        Good hedging: Trading losses offset operational revenue volatility
        """
        if not hasattr(self, 'financial_positions') or len(self.cash_flow_history) < 10:
            return 0.0

        try:
            # Get recent operational cash flows and trading PnL
            recent_ops_list = list(self.cash_flow_history)[-10:]
            if len(recent_ops_list) < 2:
                return 0.0

            recent_ops = np.array(recent_ops_list, dtype=np.float64)

            # Calculate operational revenue volatility
            if len(recent_ops) < 2:
                return 0.0

            # FIXED: Ensure proper array slicing
            ops_diff = np.diff(recent_ops)
            ops_base = recent_ops[:-1]
            ops_returns = ops_diff / np.maximum(ops_base, 1.0)
            ops_volatility = np.std(ops_returns)

            # Get trading positions value changes
            current_trading_value = (
                self.financial_positions.get('wind_instrument_value', 0) +
                self.financial_positions.get('solar_instrument_value', 0) +
                self.financial_positions.get('hydro_instrument_value', 0)
            )

            # Store trading history for correlation calculation
            if not hasattr(self, 'trading_value_history'):
                self.trading_value_history = []
            self.trading_value_history.append(current_trading_value)

            # Keep only recent history
            if len(self.trading_value_history) > 20:
                self.trading_value_history = self.trading_value_history[-20:]

            if len(self.trading_value_history) < 10:
                return 0.0

            # Calculate trading returns
            trading_values_list = list(self.trading_value_history)[-10:]
            if len(trading_values_list) < 2:
                return 0.0

            trading_values = np.array(trading_values_list, dtype=np.float64)

            # FIXED: Use signed base to preserve correlation meaning
            trading_diff = np.diff(trading_values)
            trading_base = np.where(np.abs(trading_values[:-1]) < 1.0, 1.0, trading_values[:-1])
            trading_returns = trading_diff / trading_base

            # Calculate correlation between operational and trading returns
            if len(ops_returns) == len(trading_returns) and len(ops_returns) > 2:
                correlation = np.corrcoef(ops_returns, trading_returns)[0, 1]

                # ENHANCED: Good hedging = negative correlation + actual risk reduction
                if np.isfinite(correlation):
                    # Reward negative correlation, penalize positive correlation
                    correlation_score = -correlation  # Negative correlation is good

                    # Calculate actual volatility reduction achieved
                    combined_returns = ops_returns + trading_returns
                    combined_volatility = np.std(combined_returns)
                    volatility_reduction = max(0, ops_volatility - combined_volatility) / max(ops_volatility, 1e-6)

                    # Calculate downside protection (hedge performance during operational losses)
                    downside_protection = 0.0
                    if len(ops_returns) > 0:
                        negative_ops_mask = ops_returns < 0
                        if np.any(negative_ops_mask):
                            # When operational returns are negative, trading should be positive (hedging)
                            hedge_performance = np.mean(trading_returns[negative_ops_mask])
                            downside_protection = max(0, hedge_performance) * 2.0  # Reward positive hedge during losses

                    # Combined hedging effectiveness score
                    hedging_effectiveness = (
                        correlation_score * 0.4 +           # 40% correlation
                        volatility_reduction * 0.4 +        # 40% volatility reduction
                        downside_protection * 0.2           # 20% downside protection
                    )

                    # Bonus for consistent hedging performance
                    if volatility_reduction > 0.1 and correlation_score > 0.3:
                        consistency_bonus = 0.3
                    else:
                        consistency_bonus = 0.0

                    # Final hedging score
                    hedging_score = hedging_effectiveness + consistency_bonus
                    return float(np.clip(hedging_score, -2.0, 3.0))

            return 0.0

        except Exception as e:
            # Fallback for any calculation errors
            return 0.0







    # Keep remaining methods for compatibility
    def _calculate_diversification_benefit(self, positions: Dict[str, float]) -> float:
        if not positions or sum(abs(pos) for pos in positions.values()) == 0:
            return 0.0
        total_position = sum(abs(pos) for pos in positions.values())
        weights = {asset: abs(pos) / total_position for asset, pos in positions.items()}
        weighted_individual_vol = sum(
            weights.get(asset, 0.0) * self.operational_vols.get(asset, 0.03)
            for asset in ['wind', 'solar', 'hydro']
        )
        portfolio_vol = self._calculate_portfolio_operational_vol(weights)
        if weighted_individual_vol > 0:
            diversification_benefit = (weighted_individual_vol - portfolio_vol) / weighted_individual_vol
            return float(np.clip(diversification_benefit, 0.0, 0.5))
        return 0.0

    def _calculate_portfolio_operational_vol(self, weights: Dict[str, float]) -> float:
        assets = ['wind', 'solar', 'hydro']
        portfolio_variance = 0.0
        for i, a1 in enumerate(assets):
            w1 = weights.get(a1, 0.0); v1 = self.operational_vols.get(a1, 0.03)
            for j, a2 in enumerate(assets):
                w2 = weights.get(a2, 0.0); v2 = self.operational_vols.get(a2, 0.03)
                if i == j:
                    corr = 1.0
                else:
                    corr_key = f"{a1}_{a2}" if i < j else f"{a2}_{a1}"
                    corr = self.operational_correlations.get(corr_key, 0.0)
                portfolio_variance += w1 * w2 * v1 * v2 * corr
        return float(np.sqrt(max(0.0, portfolio_variance)))

    def calculate_optimal_position_size(self, expected_return: float, volatility: float,
                                        max_position: float = 0.25) -> float:
        if volatility <= 0:
            return 0.0
        risk_free_rate = 0.02 / 252
        kelly_fraction = (expected_return - risk_free_rate) / (volatility ** 2)
        safe_kelly = kelly_fraction * 0.25
        return float(np.clip(safe_kelly, 0.0, max_position))

    def update_portfolio_peak_and_check_stops(self, current_portfolio_value: float) -> bool:
        if current_portfolio_value > self.portfolio_peak:
            self.portfolio_peak = current_portfolio_value

        # Emergency liquidation disabled for infrastructure funds
        if not getattr(self, 'emergency_liquidation_enabled', False):
            return False

        # Only trigger in extreme cases (90%+ loss)
        if current_portfolio_value < self.initial_budget * self.emergency_liquidation_threshold:
            return True
        if self.portfolio_peak > 0:
            dd = (self.portfolio_peak - current_portfolio_value) / self.portfolio_peak
            if dd > self.max_drawdown_threshold:
                return True
        return False


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

        # Randomness (seeded in reset)
        self._rng = np.random.default_rng()
        self._last_seed: Optional[int] = None

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
        
        # 3) OPERATIONAL STATE
        self.operational_state = {
            'battery_energy': 0.0,         # Current battery charge (MWh)
            'battery_discharge_power': 0.0, # Current discharge rate (MW)
        }

        # Note: Properties moved to class level (after __init__ method)
        
        @property
        def battery_energy(self): return self.operational_state['battery_energy']
        @battery_energy.setter
        def battery_energy(self, value): self.operational_state['battery_energy'] = float(value)
        
        @property  
        def battery_discharge_power(self): return self.operational_state['battery_discharge_power']
        @battery_discharge_power.setter
        def battery_discharge_power(self, value): self.operational_state['battery_discharge_power'] = float(value)

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

        # FIXED: Deploy initial assets (ONE-TIME ONLY)
        if not self.assets_deployed:
            self._deploy_initial_assets_once(initial_asset_plan)

        # CRITICAL FIX: Initialize reward calculator AFTER asset deployment with correct baseline NAV
        if getattr(self, 'reward_calculator', None) is None:
            # Use post-CAPEX NAV as baseline instead of pre-CAPEX init_budget
            post_capex_nav = self._calculate_fund_nav()
            self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
            # Store config reference for reward calculator
            if hasattr(self.reward_calculator, 'config'):
                self.reward_calculator.config = self.config
            logging.info(f"Reward calculator initialized with post-CAPEX baseline NAV: {post_capex_nav:,.0f} DKK")
        else:
            logging.info(f"Reward calculator already exists: {type(self.reward_calculator)}")

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
            logging.info(f"Price system: Using DKK throughout, USD conversion rate = {DKK_TO_USD:.3f}")
            logging.info(f"DKK price range: {price_dkk_filtered.min():.1f}-{price_dkk_filtered.max():.1f} DKK/MWh")
            self._conversion_logged = True

        # Step 3: PRICE NORMALIZATION - All in DKK
        self._price_raw = price_dkk_filtered.to_numpy()  # Raw DKK prices for revenue calculation

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

        self._price = price_normalized_clipped.to_numpy()  # Normalized for agents

        # Store normalization parameters for revenue calculations
        self._price_mean = price_rolling_mean.to_numpy()
        self._price_std = price_rolling_std.to_numpy()

        # FIXED: Prices remain in DKK throughout system for consistency
        # Raw prices in _price_raw are DKK, normalized prices in _price are z-scores

        self._load  = self.data.get('load',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._riskS = self.data.get('risk',  pd.Series(0.3, index=self.data.index)).astype(float).to_numpy()

        # FIXED: Pre-allocate forecast arrays for DL overlay labeler
        self._price_forecast_immediate = np.full(self.max_steps, np.nan, dtype=float)
        self._wind_forecast_immediate = np.full(self.max_steps, np.nan, dtype=float)
        self._solar_forecast_immediate = np.full(self.max_steps, np.nan, dtype=float)
        self._hydro_forecast_immediate = np.full(self.max_steps, np.nan, dtype=float)
        self._load_forecast_immediate = np.full(self.max_steps, np.nan, dtype=float)

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
            self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144) if enhanced_risk_controller else None
        except Exception:
            self.enhanced_risk_controller = EnhancedRiskController() if enhanced_risk_controller else None

        # memory manager
        self.memory_manager = LightweightMemoryManager(max_memory_mb=max_memory_mb)

        # Forecast accuracy tracking
        from collections import defaultdict, deque as _deque
        self._forecast_errors = defaultdict(lambda: _deque(maxlen=100))
        self._forecast_history = defaultdict(lambda: _deque(maxlen=10))
        self._forecast_accuracy_window = 50

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
        self.market_stress = 0.5
        self.overall_risk_snapshot = 0.5
        self.market_risk_snapshot = 0.5

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
        self._previous_nav = 0.0  # CRITICAL: Track previous NAV for return calculation
        self.max_leverage = self.config.max_leverage
        self.risk_multiplier = self.config.risk_multiplier

        # performance tracking
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.cumulative_mtm_pnl = 0.0  # ENHANCED: Track cumulative trading performance
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

        # Display both DKK and USD for clarity
        usd_value = self.init_budget * getattr(self.config, 'dkk_to_usd_rate', 0.145)
        logging.info(f"Hybrid renewable fund initialized with {self.init_budget:,.0f} DKK (~${usd_value/1e6:,.0f}M USD)")
        self._log_fund_structure()

        # GUARDRAIL: Startup assert to prevent regression
        assert hasattr(self, "_price_raw") and hasattr(self, "_price"), "Price arrays not initialized"

    # =====================================================================
    # FIXED: ONE-TIME ASSET DEPLOYMENT
    # =====================================================================
    
    def _deploy_initial_assets_once(self, plan: Optional[dict]):
        """FIXED: ONE-TIME ONLY asset deployment with proper accounting"""
        if self.assets_deployed:
            logging.info("Assets already deployed, skipping")
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
                logging.warning(f"Insufficient physical allocation: {total_capex:,.0f} DKK required, {self.physical_allocation_budget:,.0f} DKK available")
                # Scale down proportionally
                scale_factor = (self.physical_allocation_budget * 0.95) / total_capex  # Use 95% of physical allocation
                logging.info(f"Scaling asset plan by {scale_factor:.2f}")
            else:
                scale_factor = 1.0
                logging.info(f"Physical allocation sufficient: {total_capex:,.0f} DKK required, {self.physical_allocation_budget:,.0f} DKK available")
            
            # Deploy physical assets (using physical allocation, not trading budget)
            total_physical_capex_used = 0.0

            for asset_type, specs in plan.items():
                if asset_type == 'wind':
                    capacity = specs['capacity_mw'] * scale_factor
                    self.physical_assets['wind_capacity_mw'] = capacity
                    capex_used = capacity * self.asset_capex['wind_mw']

                elif asset_type == 'solar':
                    capacity = specs['capacity_mw'] * scale_factor
                    self.physical_assets['solar_capacity_mw'] = capacity
                    capex_used = capacity * self.asset_capex['solar_mw']

                elif asset_type == 'hydro':
                    capacity = specs['capacity_mw'] * scale_factor
                    self.physical_assets['hydro_capacity_mw'] = capacity
                    capex_used = capacity * self.asset_capex['hydro_mw']

                elif asset_type == 'battery':
                    capacity = specs['capacity_mwh'] * scale_factor
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
            
            # Update equity
            self._calculate_fund_nav()
            
            logging.info(f"Asset deployment complete:")
            logging.info(f"  Wind: {self.physical_assets['wind_capacity_mw']:.1f} MW")
            logging.info(f"  Solar: {self.physical_assets['solar_capacity_mw']:.1f} MW")
            logging.info(f"  Hydro: {self.physical_assets['hydro_capacity_mw']:.1f} MW")
            logging.info(f"  Battery: {self.physical_assets['battery_capacity_mwh']:.1f} MWh")
            logging.info(f"  Total CAPEX: {total_capex * scale_factor:,.0f} DKK (${total_capex * scale_factor * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
            logging.info(f"  Remaining cash: {self.budget:,.0f} DKK (${self.budget * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")

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

            logging.info("[OK] Option 1 deployment verified successfully")
            
        except Exception as e:
            logging.error(f"Asset deployment failed: {e}")
            self.assets_deployed = False

    def _log_fund_structure(self):
        """Log the fund's hybrid structure with clear separation"""
        logging.info("=" * 60)
        logging.info("HYBRID RENEWABLE ENERGY FUND STRUCTURE")
        logging.info("=" * 60)
        logging.info("ECONOMIC MODEL: Hybrid approach with clear separation")
        logging.info("1) PHYSICAL OWNERSHIP: Direct ownership of renewable assets")
        logging.info("2) FINANCIAL TRADING: Derivatives on renewable energy indices")
        logging.info("")

        # Physical assets (owned infrastructure)
        total_physical_mw = (self.physical_assets['wind_capacity_mw'] +
                           self.physical_assets['solar_capacity_mw'] +
                           self.physical_assets['hydro_capacity_mw'])
        physical_value = (self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] +
                         self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] +
                         self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] +
                         self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh'])

        logging.info("1. PHYSICAL ASSETS (Owned Infrastructure - Generate Real Electricity):")
        logging.info(f"   Wind farms: {self.physical_assets['wind_capacity_mw']:.1f} MW ({self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw']:,.0f} DKK)")
        logging.info(f"   Solar farms: {self.physical_assets['solar_capacity_mw']:.1f} MW ({self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw']:,.0f} DKK)")
        logging.info(f"   Hydro plants: {self.physical_assets['hydro_capacity_mw']:.1f} MW ({self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw']:,.0f} DKK)")
        logging.info(f"   Battery storage: {self.physical_assets['battery_capacity_mwh']:.1f} MWh ({self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']:,.0f} DKK)")
        logging.info(f"   Total Physical: {total_physical_mw:.1f} MW ({physical_value:,.0f} DKK book value, ${physical_value * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logging.info("")

        # Financial instruments (derivatives trading)
        total_financial = sum(abs(v) for v in self.financial_positions.values())
        logging.info("2. FINANCIAL INSTRUMENTS (Derivatives Trading - Mark-to-Market):")
        logging.info(f"   Wind index exposure: {self.financial_positions['wind_instrument_value']:,.0f} DKK")
        logging.info(f"   Solar index exposure: {self.financial_positions['solar_instrument_value']:,.0f} DKK")
        logging.info(f"   Hydro index exposure: {self.financial_positions['hydro_instrument_value']:,.0f} DKK")
        logging.info(f"   Total Financial Exposure: {total_financial:,.0f} DKK")
        logging.info("")

        # Fund summary
        fund_nav = self._calculate_fund_nav()
        logging.info("3. FUND SUMMARY:")
        logging.info(f"   Cash position: {self.budget:,.0f} DKK (${self.budget * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logging.info(f"   Physical assets (book): {physical_value:,.0f} DKK (${physical_value * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logging.info(f"   Financial positions (MTM): {total_financial:,.0f} DKK")
        logging.info(f"   Total Fund NAV: {fund_nav:,.0f} DKK (${fund_nav * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logging.info(f"   Initial capital: {self.init_budget:,.0f} DKK (${self.init_budget * self.config.dkk_to_usd_rate / 1e6:.0f}M USD)")
        logging.info(f"   Total return: {((fund_nav - self.init_budget) / self.init_budget * 100):+.2f}%")
        logging.info("=" * 60)

    # =====================================================================
    # FIXED: PROPER NAV CALCULATION
    # =====================================================================
    
    def _calculate_fund_nav(self) -> float:
        """
        FIXED: Calculate true fund NAV with proper separation:
        NAV = Trading Cash + Physical Asset Book Value + Accumulated Operational Revenue + Financial Instrument MTM
        """
        try:
            # 1) Trading cash position (separate from operational revenue)
            trading_cash_value = max(0.0, self.budget)

            # 2) Physical assets with realistic depreciation
            # Infrastructure assets depreciate over time (typical 20-30 year life)
            # FIXED: Safety check for self.t attribute
            current_timestep = getattr(self, 't', 0)
            years_elapsed = current_timestep / (365.25 * 24 * 6)  # Convert timesteps to years (10-min intervals)

            # Apply 2% annual straight-line depreciation (realistic for renewable infrastructure)
            annual_depreciation_rate = 0.02  # 2% per year
            total_depreciation = min(years_elapsed * annual_depreciation_rate, 0.75)  # Max 75% over asset life

            # Apply uniform 2% annual depreciation to all physical assets
            wind_depreciation = total_depreciation
            solar_depreciation = total_depreciation
            hydro_depreciation = total_depreciation
            battery_depreciation = total_depreciation

            physical_book_value = (
                self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] * (1.0 - wind_depreciation) +
                self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] * (1.0 - solar_depreciation) +
                self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] * (1.0 - hydro_depreciation) +
                self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh'] * (1.0 - battery_depreciation)
            )

            # 3) Accumulated operational revenue (separate from trading cash)
            operational_revenue_value = getattr(self, 'accumulated_operational_revenue', 0.0)

            # 3.5) Add back distributed cash (represents investor value)
            distributed_value = getattr(self, 'total_distributions', 0.0)

            # 4) Financial instruments at mark-to-market
            financial_mtm_value = (
                self.financial_positions['wind_instrument_value'] +
                self.financial_positions['solar_instrument_value'] +
                self.financial_positions['hydro_instrument_value']
            )

            # 5) Total NAV (including distributed value for total return calculation)
            total_nav = trading_cash_value + physical_book_value + operational_revenue_value + financial_mtm_value + distributed_value
            
            # THESIS MODE: No artificial NAV bounds - let the system show natural behavior
            # Keep financial instrument bounds for realism (based on allocated trading capital)
            trading_limits = self.config.get_trading_capital_limits()
            max_financial_exposure = trading_limits['max_financial_exposure_dkk']

            if abs(financial_mtm_value) > max_financial_exposure:
                # Clip financial instruments to allocated trading capital × leverage
                financial_mtm_value = float(np.clip(financial_mtm_value,
                                                  -max_financial_exposure,
                                                   max_financial_exposure))
                total_nav = trading_cash_value + physical_book_value + operational_revenue_value + financial_mtm_value + distributed_value

            # THESIS: Use unconstrained NAV to show natural fund behavior
            self.equity = float(max(total_nav, 0.0))  # Only prevent negative NAV
            return self.equity
            
        except Exception as e:
            logging.error(f"NAV calculation error: {e}")
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
                self._forecast_history[target].append(float(forecast_normalized))
        except Exception:
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
        """
        # Skip forecast array population if no forecast generator
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return
        try:
            if 0 <= t < self.max_steps:
                if 'price_forecast_immediate' in forecasts:
                    self._price_forecast_immediate[t] = float(forecasts['price_forecast_immediate'])
                if 'wind_forecast_immediate' in forecasts:
                    self._wind_forecast_immediate[t] = float(forecasts['wind_forecast_immediate'])
                if 'solar_forecast_immediate' in forecasts:
                    self._solar_forecast_immediate[t] = float(forecasts['solar_forecast_immediate'])
                if 'hydro_forecast_immediate' in forecasts:
                    self._hydro_forecast_immediate[t] = float(forecasts['hydro_forecast_immediate'])
                if 'load_forecast_immediate' in forecasts:
                    self._load_forecast_immediate[t] = float(forecasts['load_forecast_immediate'])
        except Exception:
            pass  # Silently ignore errors to avoid breaking main flow

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

        # Check if this is a true episode reset (end of data) or just PPO buffer reset
        # FIXED: Safety check for self.t attribute
        current_timestep = getattr(self, 't', 0)
        is_true_episode_end = (current_timestep >= self.max_steps)

        # Only reset time if true episode end
        if is_true_episode_end:
            self.t = 0
            print(f"[RESET] TRUE EPISODE RESET: End of data reached, resetting to start")
        else:
            print(f"[PPO] PPO BUFFER RESET: Preserving financial state at step {current_timestep}")

        self.step_in_episode = 0
        self.agents = self.possible_agents[:]

        # CRITICAL FIX: Only reset financial state on true episode end
        if is_true_episode_end:
            # Full reset at end of data
            self._reset_financial_state_full()
        else:
            # Preserve gains for PPO buffer resets
            self._reset_financial_state()

        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
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

        # Recalculate NAV with current state (don't reset to initial)
        self._calculate_fund_nav()

    def _reset_financial_state_full(self):
        """FULL RESET: Only used at true episode end (end of data)"""
        print(f"💰 FULL FINANCIAL RESET: Resetting all gains to initial state")

        # Reset physical assets (they are permanent)
        # Reset financial instruments to zero
        self.financial_positions = {
            'wind_instrument_value': 0.0,
            'solar_instrument_value': 0.0,
            'hydro_instrument_value': 0.0,
        }

        # Reset operational state
        self.operational_state = {
            'battery_energy': 0.0,
            'battery_discharge_power': 0.0,
        }

        # Reset cash to trading allocation (not remaining after CAPEX)
        if hasattr(self, 'trading_allocation_budget'):
            self.budget = float(self.trading_allocation_budget)  # Reset to full trading allocation
        else:
            # Fallback calculation if trading allocation not set
            self.budget = float(self.total_fund_value * self.config.financial_allocation)

        # Reset performance tracking
        self.investment_capital = float(self.init_budget)
        self.distributed_profits = 0.0
        self.cumulative_returns = 0.0
        self._previous_nav = 0.0  # Reset NAV tracking for return calculation
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.cumulative_mtm_pnl = 0.0  # Reset cumulative trading performance
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

        # Reset accumulated operational revenue
        self.accumulated_operational_revenue = 0.0

        # Calculate initial NAV
        self._calculate_fund_nav()

    def step(self, actions: Dict[str, Any]):
        # FIXED: Safety check for self.t attribute
        current_timestep = getattr(self, 't', 0)
        if current_timestep >= self.max_steps:
            return self._terminal_step()

        try:
            i = self.t
            acts = self._validate_actions(actions)

            # meta & risk knobs
            self._apply_risk_control(acts['risk_controller_0'])
            self._apply_meta_control(acts['meta_controller_0'])

            # investor + battery ops (battery returns realized cash delta)
            trade_amount = self._execute_investor_trades(acts['investor_0'])
            battery_cash_delta = self._execute_battery_ops(acts['battery_operator_0'], i)

            # FIXED: finance update (MTM, costs, realized rev incl. battery cash)
            financial = self._update_finance(i, trade_amount, battery_cash_delta)

            # regime updates & rewards
            self._update_market_conditions(i)
            self._update_risk_snapshots(i)

            # Track forecast accuracy if forecaster is available
            if self.forecast_generator and hasattr(self.forecast_generator, 'predict_all_horizons'):
                try:
                    current_forecasts = self.forecast_generator.predict_all_horizons(timestep=self.t)
                    if isinstance(current_forecasts, dict):
                        self._track_forecast_accuracy(current_forecasts)
                except Exception:
                    pass

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
            logging.error(f"step error at t={self.t}: {e}")
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
        # FIXED: map [-1,1] -> 0.5..2.0 (normalized action space)
        val = float(np.clip(risk_action.reshape(-1)[0], -1.0, 1.0))
        # Convert [-1,1] to [0,2] then to [0.5,2.0]
        normalized_val = (val + 1.0)  # [-1,1] -> [0,2]
        self.risk_multiplier = 0.5 + 0.75 * normalized_val  # [0.5,2.0]

    def _apply_meta_control(self, meta_action: np.ndarray):
        # FIXED: Handle normalized [-1,1] action space with symmetric mapping
        a0, a1 = np.array(meta_action, dtype=np.float32).reshape(-1)[:2]

        # Generic symmetric mapping function for [-1,1] -> [min,max]
        def map_from_minus1_1(x, lo, hi):
            x = float(np.clip(x, -1.0, 1.0))
            return lo + (x + 1.0) * 0.5 * (hi - lo)

        # Apply symmetric mapping to both components
        cap = map_from_minus1_1(a0, self.META_CAP_MIN, self.META_CAP_MAX)
        self.capital_allocation_fraction = float(np.clip(cap, self.META_CAP_MIN, self.META_CAP_MAX))

        freq = int(round(map_from_minus1_1(a1, self.META_FREQ_MIN, self.META_FREQ_MAX)))
        self.investment_freq = int(np.clip(freq, self.META_FREQ_MIN, self.META_FREQ_MAX))

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

            # Cap intensity to reasonable bounds
            return float(np.clip(portfolio_intensity, 0.5, 2.0))

        except Exception as e:
            return 1.0  # Default intensity

    # ------------------------------------------------------------------
    # FIXED: Financial Instrument Trading (Separate from Physical Assets)
    # ------------------------------------------------------------------

    def _execute_investor_trades(self, inv_action: np.ndarray) -> float:
        """
        SINGLE-PRICE HEDGING: Portfolio protection against electricity price risk

        Key Insight: All assets (wind/solar/hydro) sell at the SAME wholesale price
        Therefore: Need ONE net hedge, not three separate price bets

        Strategy:
        1. Calculate total portfolio exposure to price risk
        2. Determine single hedge direction and intensity
        3. Allocate hedge across sleeves using risk budgets
        4. Prevent internal cancellation between sleeves

        Returns total traded notional for transaction costs
        """
        # FIXED: Enhanced trading frequency logic with debug output
        # CRITICAL FIX: Prevent trading on step 0 to avoid overwriting existing positions
        trading_allowed = (self.t > 0 and self.t % self.investment_freq == 0)

        # Debug output for trading decisions (only in ultra fast mode to avoid spam)
        if hasattr(self, 'ultra_fast_mode') and self.ultra_fast_mode and self.t % 100 == 0:
            steps_to_next_trade = self.investment_freq - (self.t % self.investment_freq)
            print(f"[TRADING DEBUG] Step {self.t}: freq={self.investment_freq}, allowed={trading_allowed}, next_in={steps_to_next_trade}")

        if not trading_allowed:
            return 0.0  # No trading outside frequency

        # Check if trading is disabled due to drawdown
        if not self.reward_calculator.trading_enabled:
            if hasattr(self, 'ultra_fast_mode') and self.ultra_fast_mode:
                print(f"[TRADING DEBUG] Step {self.t}: Trading disabled due to drawdown ({self.reward_calculator.current_drawdown:.1%})")
            return 0.0  # No trading during high drawdown periods

        try:
            # Check if forecasting is enabled
            if hasattr(self, "forecast_generator") and self.forecast_generator is not None:
                # Forecasting enabled: use confidence-based trading logic
                forecast_confidence = self._get_forecast_confidence()

                # ENHANCED: Confidence-based nonlinearity with curved ramp
                if forecast_confidence < 0.40:
                    # Very low confidence: no trading (capital preservation)
                    forecast_confidence = 0.0
                elif forecast_confidence < self.reward_calculator.forecast_confidence_threshold:
                    # Medium confidence: quadratic ramp for smoother scaling
                    # Map [0.4, threshold] to [0, 1] with quadratic curve
                    normalized = (forecast_confidence - 0.40) / (self.reward_calculator.forecast_confidence_threshold - 0.40)
                    forecast_confidence = normalized ** 2  # Quadratic ramp: slower start, faster finish
                else:
                    # High confidence: full trading
                    forecast_confidence = 1.0
            else:
                # No forecasting: use full trading confidence (baseline behavior)
                forecast_confidence = 1.0

            # STEP 1: Calculate portfolio-level hedge intensity and direction
            # SINGLE-PRICE LOGIC: Use portfolio-level conditions only (no asset-specific signals)

            # DL HEDGE OVERLAY: Use trained model predictions if available
            dl_hedge_params = None
            if self.dl_adapter is not None:
                try:
                    # Train the model periodically
                    self.dl_adapter.maybe_learn(self.t)

                    # Create consistent feature set for DL overlay (independent of forecasting add-on)
                    # Core features that are always available
                    state_feats = np.array([
                        float(self.budget / self.init_budget),
                        float(self.equity / self.init_budget),
                        float(self.market_volatility),
                        float(self.market_stress)
                    ], dtype=np.float32)

                    position_feats = np.array([
                        float(self.financial_positions['wind_instrument_value'] / self.init_budget),
                        float(self.financial_positions['solar_instrument_value'] / self.init_budget),
                        float(self.financial_positions['hydro_instrument_value'] / self.init_budget)
                    ], dtype=np.float32)

                    # Optional forecast features (add-on)
                    if hasattr(self, 'forecast_generator') and self.forecast_generator is not None:
                        try:
                            forecasts = self.forecast_generator.predict_all_horizons(timestep=self.t)
                            if isinstance(forecasts, dict):
                                # ENHANCED: Use investment_freq-aligned horizon instead of always immediate
                                # This improves signal relevance for the DL overlay
                                h = 'immediate'  # default
                                if hasattr(self, 'investment_freq'):
                                    # FAIL-FAST: Strict horizon lookup without silent defaults
                                    inv_freq = self.investment_freq
                                    horizons = self.config.forecast_horizons
                                    if not horizons:
                                        raise ValueError("config.forecast_horizons is empty or missing")

                                    # Use strict horizon steps for alignment (no .get() defaults)
                                    if inv_freq <= horizons['immediate']:
                                        h = 'immediate'
                                    elif inv_freq <= horizons['short']:
                                        h = 'short'
                                    elif inv_freq <= horizons['medium']:
                                        h = 'medium'
                                    elif inv_freq <= horizons['long']:
                                        h = 'long'
                                    else:
                                        h = 'strategic'

                                # CRITICAL FIX: Use correct forecast key format that generator produces
                                # Generator produces keys like 'wind_forecast_short', not 'wind_short'
                                gen_forecast = np.array([
                                    float(forecasts.get(f'wind_forecast_{h}', 0.0)),
                                    float(forecasts.get(f'solar_forecast_{h}', 0.0)),
                                    float(forecasts.get(f'hydro_forecast_{h}', 0.0))
                                ], dtype=np.float32)
                            else:
                                gen_forecast = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        except Exception:
                            gen_forecast = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        # No forecasting add-on - use zeros
                        gen_forecast = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                    # Portfolio metrics
                    portfolio_metrics = np.array([
                        float(self.capital_allocation_fraction),
                        float(self.investment_freq / 100.0),
                        float(forecast_confidence)
                    ], dtype=np.float32)

                    # Validate array shapes before concatenation
                    assert state_feats.shape == (4,), f"state_feats shape: {state_feats.shape}, expected (4,)"
                    assert position_feats.shape == (3,), f"position_feats shape: {position_feats.shape}, expected (3,)"
                    assert gen_forecast.shape == (3,), f"gen_forecast shape: {gen_forecast.shape}, expected (3,)"
                    assert portfolio_metrics.shape == (3,), f"portfolio_metrics shape: {portfolio_metrics.shape}, expected (3,)"

                    # Always create 13-feature vector (4+3+3+3) regardless of add-ons
                    features = np.concatenate([state_feats, position_feats, gen_forecast, portfolio_metrics], axis=0)
                    features = features.reshape(1, -1)  # Ensure 2D shape (1, 13)

                    # Get DL predictions
                    pred = self.dl_adapter.model(features, training=False)
                    dl_hedge_params = {
                        'hedge_intensity': float(pred['hedge_intensity'].numpy().squeeze()),
                        'hedge_direction': float(pred['hedge_direction'].numpy().squeeze()),
                        'risk_allocation': pred['risk_allocation'].numpy().squeeze()
                    }
                except Exception as e:
                    logging.warning(f"DL hedge overlay failed: {e}")
                    dl_hedge_params = None

            # Calculate portfolio-level hedge intensity based on market conditions (fallback/baseline)
            generation_weights = self._get_generation_exposure_weights()
            portfolio_hedge_intensity = self._calculate_portfolio_hedge_intensity()

            # Calculate net hedge direction based on PORTFOLIO-LEVEL conditions
            # Single-price logic: direction should be based on total portfolio exposure, not individual assets
            current_data = self.data.iloc[min(self.t, len(self.data) - 1)]

            # Total current generation (normalized)
            total_current_gen = (float(current_data.get('wind', 0)) +
                               float(current_data.get('solar', 0)) +
                               float(current_data.get('hydro', 0))) / 1000.0

            # Expected total generation (based on capacity factors)
            wind_cap = self.physical_assets.get('wind_capacity_mw', 225)
            solar_cap = self.physical_assets.get('solar_capacity_mw', 100)
            hydro_cap = self.physical_assets.get('hydro_capacity_mw', 40)
            expected_total_gen = (wind_cap * 0.35 + solar_cap * 0.20 + hydro_cap * 0.45) / 1000.0

            # Portfolio-level hedge direction (use DL prediction if available, otherwise heuristic)
            if dl_hedge_params is not None:
                # Use DL model predictions with blending
                # NOTE: DL overlay outputs hedge_intensity in [0.5, 2.0] by design
                dl_intensity = np.clip(dl_hedge_params['hedge_intensity'], 0.5, 2.0)
                dl_direction = np.clip(dl_hedge_params['hedge_direction'], 0.0, 1.0)

                # Convert direction from [0,1] to [-1,1] range
                net_hedge_direction = (dl_direction - 0.5) * 2.0  # [0,1] -> [-1,1]

                # Blend DL intensity with heuristic (80% DL, 20% heuristic for stability)
                portfolio_hedge_intensity = 0.8 * dl_intensity + 0.2 * portfolio_hedge_intensity
            else:
                # Fallback to heuristic hedge direction
                if total_current_gen < expected_total_gen * 0.8:
                    net_hedge_direction = 1.0   # LONG: hedge against high prices during low generation
                elif total_current_gen > expected_total_gen * 1.2:
                    net_hedge_direction = -1.0  # SHORT: hedge against low prices during high generation
                else:
                    net_hedge_direction = 0.5   # Mild LONG bias: default portfolio protection

            # STEP 2: Calculate total hedge size based on allocated trading capital
            # Get proper trading capital limits from config
            trading_limits = self.config.get_trading_capital_limits()
            allocated_trading_capital = trading_limits['trading_capital_dkk']
            max_financial_exposure = trading_limits['max_financial_exposure_dkk']

            # Available capital for financial instruments (from allocated trading capital)
            available_capital = min(self.budget * self.capital_allocation_fraction, allocated_trading_capital)

            # Combined position sizing: volatility + drawdown + portfolio intensity
            volatility_factor = self._get_volatility_factor()
            volatility_multiplier = self._calculate_position_size_multiplier(volatility_factor)
            drawdown_multiplier = getattr(self.reward_calculator, 'position_size_multiplier', 1.0)
            position_size_multiplier = volatility_multiplier * drawdown_multiplier

            # Base hedge size (conservative for risk management)
            # ENHANCED: Cap by allocated trading capital with leverage, not total NAV
            capital_based_limit = available_capital * 0.8  # 80% of allocated trading capital
            leverage_based_limit = max_financial_exposure * 0.8  # 80% of max allowed exposure
            base_target_gross = min(capital_based_limit, leverage_based_limit)  # Take the more conservative limit

            # ENHANCED: Apply hedge effectiveness multiplier
            effectiveness_multiplier = getattr(self, 'hedge_effectiveness_tracker', {}).get('effectiveness_multiplier', 1.0)

            # Scale by portfolio hedge intensity, forecast confidence, and effectiveness
            target_gross = (base_target_gross *
                          portfolio_hedge_intensity *
                          position_size_multiplier *
                          forecast_confidence *
                          effectiveness_multiplier)

            # STEP 3: Allocate hedge across sleeves using risk budgets
            # Use DL risk allocation if available, otherwise use risk budget allocation
            if dl_hedge_params is not None and 'risk_allocation' in dl_hedge_params:
                dl_allocation = dl_hedge_params['risk_allocation']
                # Normalize to ensure sum = 1
                dl_allocation = dl_allocation / (dl_allocation.sum() + 1e-9)
                # Blend DL allocation with heuristic (70% DL, 30% heuristic for stability)
                heuristic_allocation = self._get_risk_budget_allocation()
                risk_budget_allocation = 0.7 * dl_allocation + 0.3 * np.array(heuristic_allocation)
                # Ensure it's a proper allocation
                risk_budget_allocation = risk_budget_allocation / risk_budget_allocation.sum()
            else:
                # Use risk budget allocation (not separate price bets)
                risk_budget_allocation = self._get_risk_budget_allocation()

            # Current financial positions
            current_wind = self.financial_positions['wind_instrument_value']
            current_solar = self.financial_positions['solar_instrument_value']
            current_hydro = self.financial_positions['hydro_instrument_value']

            # Target positions: ALL sleeves have SAME direction (single hedge)
            # Positive = Long position (benefits from price increases)
            # Negative = Short position (benefits from price decreases)
            target_wind = target_gross * risk_budget_allocation[0] * net_hedge_direction
            target_solar = target_gross * risk_budget_allocation[1] * net_hedge_direction
            target_hydro = target_gross * risk_budget_allocation[2] * net_hedge_direction

            # STEP 4: Apply position limits before trading (based on trading capital, not total NAV)
            max_position_size = allocated_trading_capital * getattr(self.config, 'max_position_size', 0.05)
            target_wind = np.clip(target_wind, -max_position_size, max_position_size)
            target_solar = np.clip(target_solar, -max_position_size, max_position_size)
            target_hydro = np.clip(target_hydro, -max_position_size, max_position_size)

            # STEP 5: Calculate trades needed
            trade_wind = target_wind - current_wind
            trade_solar = target_solar - current_solar
            trade_hydro = target_hydro - current_hydro

            # STEP 6: Apply gross and net position limits based on trading capital allocation
            # Gross limit: total absolute exposure (consistent with base_target_gross scaling)
            gross_exposure = abs(target_wind) + abs(target_solar) + abs(target_hydro)
            # ENHANCED: Use trading capital allocation limits, not total NAV
            gross_limit = min(capital_based_limit, leverage_based_limit)  # Consistent with base calculation

            if gross_exposure > gross_limit:
                scale_factor = gross_limit / max(gross_exposure, 1e-6)
                target_wind *= scale_factor
                target_solar *= scale_factor
                target_hydro *= scale_factor
                # Recalculate trades after scaling
                trade_wind = target_wind - current_wind
                trade_solar = target_solar - current_solar
                trade_hydro = target_hydro - current_hydro

            # ENHANCED: Apply no-trade band to reduce churn
            no_trade_threshold = 0.02  # 2% of sleeve target
            if abs(target_wind) > 0 and abs(trade_wind) < abs(target_wind) * no_trade_threshold:
                trade_wind = 0.0
            if abs(target_solar) > 0 and abs(trade_solar) < abs(target_solar) * no_trade_threshold:
                trade_solar = 0.0
            if abs(target_hydro) > 0 and abs(trade_hydro) < abs(target_hydro) * no_trade_threshold:
                trade_hydro = 0.0

            # FIXED: Remove cash instrument logic - these are derivatives (futures/swaps)
            # No upfront cash needed for derivatives, only margin/collateral requirements
            # Position sizing is controlled by risk limits, not cash constraints

            # STEP 7: Calculate transaction costs (OPTIMIZED for cash efficiency) - FIXED: Use DKK from config
            total_traded_notional = abs(trade_wind) + abs(trade_solar) + abs(trade_hydro)
            transaction_cost_bps = self.config.transaction_cost_bps  # From config: institutional rates
            fixed_cost = self.config.transaction_fixed_cost  # From config
            total_transaction_costs = (total_traded_notional * transaction_cost_bps / 10000.0 +
                                     (fixed_cost if total_traded_notional > 0 else 0))

            # STEP 8: Execute trades (update financial positions)
            self.financial_positions['wind_instrument_value'] += trade_wind
            self.financial_positions['solar_instrument_value'] += trade_solar
            self.financial_positions['hydro_instrument_value'] += trade_hydro

            # FIXED: Trading should only affect cash through transaction costs (positions are marked-to-market)
            # Financial instruments are derivatives - no upfront cash payment, only margin/collateral
            cash_flow = -total_transaction_costs  # Only transaction costs affect cash
            self.budget += cash_flow
            self.budget = max(0.0, self.budget)

            # Track transaction costs for PnL attribution
            if not hasattr(self, 'cumulative_transaction_costs'):
                self.cumulative_transaction_costs = 0.0
            self.cumulative_transaction_costs += total_transaction_costs

            # ENHANCED: Track cash efficiency metrics
            if not hasattr(self, 'cash_efficiency_tracker'):
                self.cash_efficiency_tracker = {
                    'total_cash_used': 0.0,
                    'total_trading_gains': 0.0,
                    'efficiency_ratio': 0.0
                }

            # Update cash efficiency tracking
            cash_used_this_trade = abs(cash_flow) if cash_flow < 0 else 0.0
            self.cash_efficiency_tracker['total_cash_used'] += cash_used_this_trade

            # ENHANCED: Track hedge effectiveness with rolling correlation
            if not hasattr(self, 'hedge_hit_rate_tracker'):
                self.hedge_hit_rate_tracker = {'hits': 0, 'total': 0}

            if not hasattr(self, 'hedge_effectiveness_tracker'):
                from collections import deque
                self.hedge_effectiveness_tracker = {
                    'ops_returns': deque(maxlen=100),  # Last 100 steps
                    'trading_returns': deque(maxlen=100),
                    'correlation': 0.0,
                    'effectiveness_multiplier': 1.0
                }

            # Calculate hedge effectiveness for this step
            ops_return = getattr(self, 'last_generation_revenue', 0.0)
            trading_return = getattr(self, 'last_mtm_pnl', 0.0)

            # Count hedge hits: ops_return < 0 & trading_return > 0 (or symmetric case)
            if (ops_return < 0 and trading_return > 0) or (ops_return > 0 and trading_return < 0):
                self.hedge_hit_rate_tracker['hits'] += 1
            self.hedge_hit_rate_tracker['total'] += 1

            return total_traded_notional

        except Exception as e:
            logging.error(f"Trading execution error: {e}")
            return 0.0

    def _get_forecast_confidence(self) -> float:
        """Get forecast confidence score [0,1]"""
        try:
            if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
                return 1.0  # FIXED: Full confidence for baseline hedging when no forecasting

            # Use the generator's confidence calculation for the primary trading agent
            if hasattr(self.forecast_generator, "calculate_forecast_confidence"):
                # Get confidence for the main trading agent (assuming first agent is primary)
                agent_names = getattr(self.forecast_generator, 'agent_targets', {}).keys()
                if agent_names:
                    primary_agent = next(iter(agent_names))  # Get first agent
                    confidence = self.forecast_generator.calculate_forecast_confidence(primary_agent, self.t)
                    return float(np.clip(confidence, 0.0, 1.0))

            # Fallback: check for confidence in forecast output
            if hasattr(self.forecast_generator, "predict_all_horizons"):
                forecasts = self.forecast_generator.predict_all_horizons(timestep=self.t)
                if isinstance(forecasts, dict):
                    # Look for confidence scores in forecast output
                    for key in ['confidence', 'forecast_confidence', 'price_confidence']:
                        if key in forecasts and np.isfinite(forecasts[key]):
                            return float(np.clip(forecasts[key], 0.0, 1.0))

            return 0.5  # Default confidence

        except Exception:
            return 0.5

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

            # Adjust thresholds for DKK scale (typical range 50-500 DKK/MWh)
            needed = max(5.0, 0.02 * abs(p_now))  # Minimum threshold in DKK/MWh
            rt_loss = (1.0/(max(self.batt_eta_charge*self.batt_eta_discharge, 1e-6)) - 1.0) * 10.0  # Round-trip loss in DKK/MWh
            hurdle = needed + 0.3 * rt_loss

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
        """FIXED: Return battery cash delta for proper reward accounting."""
        try:
            u_raw = float(np.clip(bat_action.reshape(-1)[0], -1.0, 1.0))

            if self.physical_assets['battery_capacity_mwh'] <= 0.0:
                self.operational_state['battery_energy'] = 0.0
                self.operational_state['battery_discharge_power'] = 0.0
                return 0.0

            step_h = 10.0 / 60.0
            max_power_mw = self.physical_assets['battery_capacity_mwh'] * self.batt_power_c_rate
            max_energy_this_step = max_power_mw * step_h

            decision, inten = self._battery_dispatch_policy(i)
            bias = (u_raw - 0.0)
            inten = float(np.clip(inten + 0.25 * abs(bias), 0.0, 1.0))
            if bias < -0.6 and decision == "charge":
                decision = "idle"
            if bias > 0.6 and decision == "discharge":
                decision = "discharge"

            soc = 0.0 if self.physical_assets['battery_capacity_mwh'] <= 0 else self.operational_state['battery_energy'] / self.physical_assets['battery_capacity_mwh']
            soc = float(np.clip(soc, 0.0, 1.0))
            soc_min_e = self.batt_soc_min * self.physical_assets['battery_capacity_mwh']
            soc_max_e = self.batt_soc_max * self.physical_assets['battery_capacity_mwh']

            price = float(np.clip(self._price_raw[i], self.config.minimum_price_filter, self.config.maximum_price_cap))
            cash_delta = 0.0
            throughput_mwh = 0.0

            if decision == "discharge" and soc > self.batt_soc_min + 1e-6:
                energy_possible = min(self.operational_state['battery_energy'] - soc_min_e, max_energy_this_step * inten)
                energy_possible = max(0.0, energy_possible)
                delivered_mwh = energy_possible * self.batt_eta_discharge
                self.operational_state['battery_energy'] -= energy_possible
                throughput_mwh += energy_possible
                self.operational_state['battery_discharge_power'] = delivered_mwh / step_h
                cash_delta += delivered_mwh * price

            elif decision == "charge" and soc < self.batt_soc_max - 1e-6:
                room = soc_max_e - self.operational_state['battery_energy']
                energy_possible = min(room, max_energy_this_step * inten)
                energy_possible = max(0.0, energy_possible)
                grid_mwh = energy_possible / max(self.batt_eta_charge, 1e-6)
                self.operational_state['battery_energy'] += energy_possible
                throughput_mwh += energy_possible
                self.operational_state['battery_discharge_power'] = 0.0
                cash_delta -= grid_mwh * price

            else:
                self.operational_state['battery_discharge_power'] = 0.0

            deg_cost = self.batt_degradation_cost * throughput_mwh
            cash_delta -= deg_cost

            # Bounds check
            self.operational_state['battery_energy'] = float(np.clip(
                self.operational_state['battery_energy'], 
                0.0, 
                self.physical_assets['battery_capacity_mwh']
            ))

            return float(cash_delta)
        except Exception as e:
            logging.error(f"battery ops: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # FIXED: Finance & rewards with proper separation
    # ------------------------------------------------------------------
    def _update_finance(self, i: int, trade_amount: float, battery_cash_delta: float) -> Dict[str, float]:
        """
        FIXED: Update all financial components with proper separation:
        1. Physical asset generation revenue (cash flow)
        2. Financial instrument mark-to-market (unrealized)
        3. Battery operations (cash flow)
        4. Transaction costs (cash flow)
        """
        try:
            # CRITICAL FIX: Use raw prices for MTM calculations, not normalized prices
            current_price = float(np.clip(self._price_raw[i], -1000.0, 1e9))
            prev_price = float(np.clip(self._price_raw[i-1] if i > 0 else current_price, -1000.0, 1e9))
            price_return = (current_price - prev_price) / max(abs(prev_price), 1e-6)
            
            # 1) Generation revenue from physical assets (CASH FLOW)
            generation_revenue = self._calculate_generation_revenue(i, current_price)
            
            # 2) Mark-to-market on financial instruments (UNREALIZED)
            # FIXED: Limit MTM P&L to reasonable bounds to prevent currency-induced explosions
            total_financial_exposure = (
                self.financial_positions['wind_instrument_value'] +
                self.financial_positions['solar_instrument_value'] +
                self.financial_positions['hydro_instrument_value']
            )

            # Cap price returns to realistic energy market volatility (from config)
            cap_min = getattr(self.config, 'mtm_price_return_cap_min', -0.015)  # FIXED: Use config values
            cap_max = getattr(self.config, 'mtm_price_return_cap_max', 0.015)   # FIXED: Use config values
            capped_price_return = float(np.clip(price_return, cap_min, cap_max))
            mtm_pnl = total_financial_exposure * capped_price_return

            # FIXED: Apply MTM to position values for consistent market-value approach
            # Update position values to reflect current market value
            if abs(mtm_pnl) > getattr(self.config, 'mtm_update_threshold', 1e-9):
                # Apply MTM proportionally to each position based on exposure
                if abs(total_financial_exposure) > 1e-9:
                    wind_mtm = self.financial_positions['wind_instrument_value'] * capped_price_return
                    solar_mtm = self.financial_positions['solar_instrument_value'] * capped_price_return
                    hydro_mtm = self.financial_positions['hydro_instrument_value'] * capped_price_return

                    # Update position values with MTM (market-value approach)
                    self.financial_positions['wind_instrument_value'] += wind_mtm
                    self.financial_positions['solar_instrument_value'] += solar_mtm
                    self.financial_positions['hydro_instrument_value'] += hydro_mtm
            
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
            fund_nav = self._calculate_fund_nav()

            # 9) Track performance
            self.performance_history['revenue_history'].append(net_cash_flow)
            self.performance_history['generation_revenue_history'].append(generation_revenue)
            self.performance_history['nav_history'].append(fund_nav)

            # 10) Store values for logging/rewards
            self.last_revenue = net_cash_flow
            self.last_generation_revenue = generation_revenue
            self.last_mtm_pnl = mtm_pnl

            # 11) Update cumulative tracking
            self.cumulative_generation_revenue += generation_revenue
            self.cumulative_battery_revenue += battery_cash_delta
            self.cumulative_mtm_pnl += mtm_pnl  # ENHANCED: Track cumulative trading performance

            # CRITICAL FIX: Update cumulative returns (total fund return)
            current_nav = fund_nav
            if hasattr(self, '_previous_nav') and self._previous_nav > 0:
                nav_return = (current_nav - self._previous_nav) / self._previous_nav
                self.cumulative_returns += nav_return
            self._previous_nav = current_nav
            
            # 11) Calculate forecast signal if available
            pf_aligned = self._get_aligned_price_forecast(i, default=None)
            if pf_aligned is not None and np.isfinite(pf_aligned):
                forecast_signal_score = float(np.sign(pf_aligned - current_price) * price_return)
            else:
                forecast_signal_score = 0.0

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
                'forecast_signal_score': forecast_signal_score,
                'efficiency': self._calculate_generation_efficiency(i),
            }
            
        except Exception as e:
            logging.error(f"Finance update error: {e}")
            return {
                'revenue': 0.0,
                'generation_revenue': 0.0,
                'battery_cash_delta': 0.0,
                'mtm_pnl': 0.0,
                'fund_nav': self.equity,
                'portfolio_value': self.equity,
                'forecast_signal_score': 0.0,
                'efficiency': 0.0,
            }

    def _calculate_generation_revenue(self, i: int, price: float) -> float:
        """
        FIXED: Calculate revenue from PHYSICAL ASSETS only
        Revenue = (Physical Generation * Market Price) - Operating Costs

        IMPORTANT: Only apply operational costs AFTER assets are deployed!
        During asset acquisition phase, no operational costs should apply.
        """
        try:
            # FIXED: Restore immediate revenue generation for profitable agent learning
            # Removed asset deployment check that was preventing early revenue
            time_step_hours = self.config.time_step_hours  # From config: 10-minute timesteps

            # Get capacity factors from physical assets
            # No capacity factor calculation needed - use training data directly

            # CORRECTED: Use training data directly as actual generation, apply fractional ownership to revenue only
            # Training data contains actual generation values (MW) for full farms
            # AI agents see and optimize the full farm generation
            full_wind_generation_mw = float(self._wind[i]) if i < len(self._wind) else 0.0
            full_solar_generation_mw = float(self._solar[i]) if i < len(self._solar) else 0.0
            full_hydro_generation_mw = float(self._hydro[i]) if i < len(self._hydro) else 0.0

            # Convert to MWh for this timestep
            full_wind_generation_mwh = full_wind_generation_mw * time_step_hours
            full_solar_generation_mwh = full_solar_generation_mw * time_step_hours
            full_hydro_generation_mwh = full_hydro_generation_mw * time_step_hours
            full_total_generation_mwh = full_wind_generation_mwh + full_solar_generation_mwh + full_hydro_generation_mwh

            # Fractional ownership percentages (applied only to revenue) - FULLY CONFIG-DRIVEN
            if not self.config:
                raise ValueError("Config object required for ownership fraction calculations")

            wind_ownership_pct = self.config.wind_ownership_fraction
            solar_ownership_pct = self.config.solar_ownership_fraction
            hydro_ownership_pct = self.config.hydro_ownership_fraction

            # Fund's revenue share (fractional ownership applied here only)
            fund_wind_generation_mwh = full_wind_generation_mwh * wind_ownership_pct
            fund_solar_generation_mwh = full_solar_generation_mwh * solar_ownership_pct
            fund_hydro_generation_mwh = full_hydro_generation_mwh * hydro_ownership_pct
            fund_total_generation_mwh = fund_wind_generation_mwh + fund_solar_generation_mwh + fund_hydro_generation_mwh

            # Safety check
            if fund_total_generation_mwh <= 0.001:
                return 0.0

            # Revenue from electricity sales (based on fund's fractional ownership)
            # NOTE: price is in DKK/MWh (all calculations in DKK, converted to USD only for reporting)
            # FIXED: Apply minimum price floor to prevent negative revenue during training (from config)
            effective_price = max(price, self.config.minimum_price_floor)  # From config: minimum price floor
            gross_revenue = fund_total_generation_mwh * effective_price * self.electricity_markup * self.currency_conversion

            # Operating costs (based on fund's actual generation)
            variable_costs = gross_revenue * self.operating_cost_rate
            maintenance_costs = fund_total_generation_mwh * self.maintenance_cost_mwh
            grid_connection_costs = fund_total_generation_mwh * self.grid_connection_fee_mwh  # RESTORED: Essential grid costs
            transmission_costs = fund_total_generation_mwh * self.transmission_fee_mwh        # RESTORED: Essential transmission costs

            # Annual costs prorated to timestep
            # CORRECTED: Only pay costs on fund's actual investment in physical assets
            fund_physical_investment = (
                self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] +
                self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] +
                self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] +
                self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']
            )

            annual_to_timestep = time_step_hours / 8760
            # CORRECTED: Asset-based costs only on fund's actual physical investment
            insurance_costs = fund_physical_investment * self.insurance_rate * annual_to_timestep
            property_taxes = fund_physical_investment * self.property_tax_rate * annual_to_timestep  # RESTORED: Property taxes
            debt_service = fund_physical_investment * self.debt_service_rate * annual_to_timestep    # RESTORED: Debt service

            # REALISTIC: Management fees at contractual rate (no artificial reduction)
            management_fees = self.init_budget * self.management_fee_rate * annual_to_timestep

            # REMOVED: All admin costs for better agent learning environment
            # regulatory_costs = self.init_budget * self.regulatory_compliance_rate * annual_to_timestep
            # audit_legal_costs = self.init_budget * self.audit_legal_rate * annual_to_timestep
            # custody_fees = self.init_budget * self.custody_fee_rate * annual_to_timestep
            # administration_fees = self.init_budget * self.administration_fee_rate * annual_to_timestep

            # REMOVED: Trading costs for better agent learning
            # trading_costs = abs(gross_revenue) * self.trading_cost_rate * 0.1  # 10% of revenue involves trading

            total_operating_costs = (variable_costs + maintenance_costs + grid_connection_costs +
                                    transmission_costs + insurance_costs + property_taxes +
                                    debt_service + management_fees)

            net_revenue = max(0.0, gross_revenue - total_operating_costs)
            return float(net_revenue)

        except Exception as e:
            logging.warning(f"Generation revenue calculation failed: {e}")
            return 0.0

    def _distribute_excess_cash(self):
        """
        INFRASTRUCTURE FUND: Distribute excess cash to maintain realistic cash levels.
        Infrastructure funds typically maintain 5-15% cash and distribute excess to investors.
        """
        try:
            # Calculate target cash level (10% of fund value for infrastructure)
            target_cash_ratio = 0.10  # 10% target cash level
            current_fund_value = self._calculate_fund_nav()
            target_cash_level = current_fund_value * target_cash_ratio

            # Calculate excess cash above target
            excess_cash = self.budget - target_cash_level

            # Distribute excess if above threshold (quarterly distributions)
            distribution_threshold = self.init_budget * 0.02  # 2% of initial fund

            if excess_cash > distribution_threshold:
                # Distribute 80% of excess (keep some buffer)
                distribution_amount = excess_cash * 0.80

                # Execute distribution
                self.budget -= distribution_amount

                # Track distributions for reporting
                if not hasattr(self, 'total_distributions'):
                    self.total_distributions = 0.0
                self.total_distributions += distribution_amount

                # Log significant distributions
                if distribution_amount > self.init_budget * 0.01:  # >1% of fund
                    logging.info(f"Cash distribution: {distribution_amount:,.0f} DKK (${distribution_amount * self.config.dkk_to_usd_rate / 1e6:.1f}M USD)")

        except Exception as e:
            logging.warning(f"Cash distribution failed: {e}")

    def _check_emergency_reallocation(self, operational_revenue: float):
        """
        ENHANCED: Emergency reallocation from operational gains to trading capital
        Allows operational revenue to replenish trading capital under strict controls
        """
        try:
            # Initialize tracking if needed
            if not hasattr(self, 'total_reallocated'):
                self.total_reallocated = 0.0

            # Check if trading capital is below emergency threshold
            original_trading_allocation = self.trading_allocation_budget
            current_trading_ratio = self.budget / original_trading_allocation if original_trading_allocation > 0 else 0

            # Emergency conditions
            below_threshold = current_trading_ratio < self.config.trading_capital_emergency_threshold
            has_operational_gains = operational_revenue > 0
            under_total_limit = self.total_reallocated < (self.total_fund_value * self.config.max_total_reallocation)

            if below_threshold and has_operational_gains and under_total_limit:
                # Calculate reallocation amount
                max_from_operational = operational_revenue * self.config.max_reallocation_rate
                max_remaining_total = (self.total_fund_value * self.config.max_total_reallocation) - self.total_reallocated

                reallocation_amount = min(max_from_operational, max_remaining_total)

                if reallocation_amount > 1000:  # Minimum 1000 DKK threshold
                    # Execute reallocation
                    self.budget += reallocation_amount
                    self.accumulated_operational_revenue -= reallocation_amount
                    self.total_reallocated += reallocation_amount

                    logging.info(f"Emergency reallocation: {reallocation_amount:,.0f} DKK from operational to trading")
                    logging.info(f"Trading capital ratio: {current_trading_ratio:.1%} → {(self.budget / original_trading_allocation):.1%}")
                    logging.info(f"Total reallocated: {self.total_reallocated:,.0f} DKK ({self.total_reallocated / self.total_fund_value:.1%} of fund)")

        except Exception as e:
            logging.error(f"Emergency reallocation check failed: {e}")

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
            logging.warning(f"Fund administration cost calculation failed: {e}")
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
            # Calculate fund NAV
            fund_nav = self._calculate_fund_nav()
            
            # Get cash flow (actual money earned this step)
            cash_flow = financial.get('revenue', 0.0)
            
            # Get risk level
            risk_level = float(np.clip(self.overall_risk_snapshot, 0.0, 1.0))
            
            # Get efficiency
            efficiency = financial.get('efficiency', 0.0)
            
            # Get forecast signal score
            forecast_signal_score = financial.get('forecast_signal_score', 0.0)
            
            # Calculate reward using FIXED calculator
            if self.reward_calculator is None:
                logging.error(f"Reward calculator is None at step {self.current_step}. Initializing now...")
                # Emergency initialization
                post_capex_nav = self._calculate_fund_nav()
                self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=post_capex_nav, config=self.config)
                logging.info(f"Emergency reward calculator initialized with NAV: {post_capex_nav:,.0f} DKK")

            reward = self.reward_calculator.calculate_reward(
                fund_nav=fund_nav,
                cash_flow=cash_flow,
                risk_level=risk_level,
                efficiency=efficiency,
                forecast_signal_score=forecast_signal_score
            )

            # FIXED: Robust reward processing with safety checks
            if not np.isfinite(reward):
                reward = 0.0

            clipped_reward = float(np.clip(reward, -20.0, 20.0))

            # FIXED: Safe reward assignment with type checking
            if isinstance(self.possible_agents, (list, tuple)):
                for agent in self.possible_agents:
                    if isinstance(agent, str) and agent in self._rew_buf:
                        self._rew_buf[agent] = clipped_reward
            else:
                # Fallback for unexpected agent structure
                for agent in ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]:
                    if agent in self._rew_buf:
                        self._rew_buf[agent] = clipped_reward
            
            # Store reward breakdown for logging
            self.last_reward_breakdown = {
                'total_reward': reward,
                'fund_nav': fund_nav,
                'cash_flow': cash_flow,
                'risk_level': risk_level,
                'efficiency': efficiency,
                'forecast_signal_score': forecast_signal_score,
                'reward_components': self.reward_calculator.reward_weights.copy() if self.reward_calculator else {}
            }
            self.last_reward_weights = self.reward_calculator.reward_weights.copy() if self.reward_calculator else {}
            
        except Exception as e:
            logging.error(f"Reward assignment error: {e}")
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
                logging.error(f"Error in reward error handling: {e2}")
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
            logging.warning(f"Risk snapshot update failed: {e}")
            self.overall_risk_snapshot = 0.5
            self.market_risk_snapshot = 0.3
            self.portfolio_risk_snapshot = 0.25
            self.liquidity_risk_snapshot = 0.15

    # ------------------------------------------------------------------
    # Observations (BASE only)
    # ------------------------------------------------------------------
    def _fill_obs(self):
        i = min(self.t, self.max_steps - 1)
        # FIXED: Price normalization - z-score already clipped to [-3,3], divide by 3 to get [-1,1]
        price_n = float(np.clip(self._price[i] / 3.0, -1.0, 1.0))
        load_n  = float(np.clip(SafeDivision.div(self._load[i],  self.load_scale, 0.0), 0.0, 1.0))
        windf   = float(np.clip(SafeDivision.div(self._wind[i],  self.wind_scale, 0.0), 0.0, 1.0))
        solarf  = float(np.clip(SafeDivision.div(self._solar[i], self.solar_scale, 0.0), 0.0, 1.0))
        hydrof  = float(np.clip(SafeDivision.div(self._hydro[i], self.hydro_scale, 0.0), 0.0, 1.0))

        init_div = self.init_budget / 10.0 if self.init_budget > 0 else 1.0
        budget_n = float(np.clip(SafeDivision.div(self.budget, init_div, 0.0), 0.0, 10.0))

        inv = self._obs_buf['investor_0']
        inv[:6] = (windf, solarf, hydrof, price_n, load_n, budget_n)

        batt = self._obs_buf['battery_operator_0']
        batt[:4] = (
            price_n,
            float(np.clip(self.operational_state['battery_energy'], 0.0, 10.0)),
            float(np.clip(self.physical_assets['battery_capacity_mwh'], 0.0, 10.0)),
            load_n
        )

        rsk = self._obs_buf['risk_controller_0']
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

        perf_ratio = float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0))
        meta = self._obs_buf['meta_controller_0']
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
                }
        except Exception as e:
            logging.error(f"info populate: {e}")

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
            logging.error(f"Fund performance summary failed: {e}")
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
            logging.warning("Hybrid model integrity issues:")
            for issue in issues:
                logging.warning(f"  - {issue}")
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