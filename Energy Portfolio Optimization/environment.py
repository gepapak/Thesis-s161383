# environment.py
# FULLY PATCHED HYBRID MODEL VERSION - Complete Original Functionality

"""
Multi-agent renewable energy investment environment with hybrid economic model.

HYBRID FUND STRUCTURE ($500M Total Capital):
==============================================================================
ECONOMIC MODEL: Clear separation between physical ownership and financial trading

1) PHYSICAL OWNERSHIP ($250M deployed - 50% allocation):
   - Wind farms: 75 MW ($125M) - Fractional ownership: 5% of 1,500MW wind farm (5 WTGs of 15MW each)
   - Solar farms: 50 MW ($60M) - Fractional ownership: 5% of 1,000MW solar farm
   - Hydro plants: 20 MW ($60M) - Fractional ownership: 2% of 1,000MW hydro plant
   - Battery storage: 10 MWh / 5MW ($6M) - Direct ownership matching PyPSA specifications
   - Total: 145 MW physical capacity generating real electricity

2) FINANCIAL TRADING ($250M allocated - 50% allocation):
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

        # Use per-dimension bounds to allow negative normalized price where applicable.
        inv_low  = np.array([0.0, 0.0, 0.0, -10.0, 0.0, 0.0], dtype=np.float32)
        inv_high = np.array([10.0, 10.0, 10.0,  10.0, 10.0, 10.0], dtype=np.float32)

        bat_low  = np.array([-10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bat_high = np.array([ 10.0,10.0,10.0,10.0], dtype=np.float32)

        # risk: [price_n, vol*10, stress*10, wind_pos_rel*10, solar_pos_rel*10, hydro_pos_rel*10,
        #        cap_frac*10, equity_rel*10, risk_multiplier*5]
        risk_low  = np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        risk_high = np.array([ 10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0], dtype=np.float32)

        # meta: idx4 is price_n which can be negative
        meta_low  = np.array([0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        meta_high = np.array([10.0]*11, dtype=np.float32)

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

        # ENHANCED: Reward weights from config (centralized configuration)
        if config and hasattr(config, 'forecast_accuracy_reward_weight'):
            self.reward_weights = {
                'nav_growth': 0.65,      # Primary: Fund NAV appreciation (65% - slightly reduced for forecast)
                'cash_flow': 0.20,       # Secondary: Actual cash generation (20% - stable)
                'risk_adjusted': getattr(config, 'risk_penalty_weight', 0.10),   # Risk penalty from config
                'efficiency': 0.0,       # Disabled to focus on performance
                'forecast': getattr(config, 'forecast_accuracy_reward_weight', 0.05),  # From config
            }
            # Trading controls from config
            self.forecast_confidence_threshold = getattr(config, 'forecast_confidence_threshold', 0.7)
            self.max_drawdown_threshold = 0.10       # More conservative drawdown limit (10%)
        else:
            # Fallback to hardcoded values if no config
            self.reward_weights = {
                'nav_growth': 0.65,      # Primary: Fund NAV appreciation (65% - slightly reduced for forecast)
                'cash_flow': 0.20,       # Secondary: Actual cash generation (20% - stable)
                'risk_adjusted': 0.10,   # Risk penalty (10% - stable)
                'efficiency': 0.0,       # Disabled to focus on performance
                'forecast': 0.05,        # ENABLED: Forecast accuracy bonus (5% - demonstrates AI value)
            }
            # REALISTIC: Conservative trading controls for institutional fund
            self.forecast_confidence_threshold = 0.7  # Higher confidence threshold (70%)
            self.max_drawdown_threshold = 0.10       # More conservative drawdown limit (10%)
        self.current_drawdown = 0.0
        self.peak_nav = 0.0
        self.trading_enabled = True

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

        # Disable trading if drawdown exceeds threshold
        if self.current_drawdown > self.max_drawdown_threshold:
            self.trading_enabled = False
        elif self.current_drawdown < 0.05:  # Re-enable when drawdown < 5%
            self.trading_enabled = True

        if len(self.portfolio_history) < 2:
            return 0.0

        # 1) NAV growth component (ENHANCED for better learning signals)
        nav_return = (fund_nav - self.portfolio_history[-2]) / max(self.portfolio_history[-2], 1.0)
        # ENHANCED: Stronger scaling for better learning signals in RL training
        nav_score = float(np.clip(nav_return * 200.0, -5.0, 5.0))  # Increased scaling for stronger learning signals

        # 2) Cash flow component (STABILITY - 30% weight)
        recent_cash_flows = list(self.cash_flow_history)[-20:]
        avg_cash_flow = np.mean(recent_cash_flows) if recent_cash_flows else 0.0
        # REALISTIC: Scale cash flow rewards to institutional fund size
        cash_flow_score = float(np.clip(avg_cash_flow / (self.initial_budget * 0.0005), -1.5, 1.5))

        # 3) Risk penalty (REDUCED for better learning signals)
        risk_penalty = float(np.clip(risk_level * 1.0, 0.0, 1.0))  # Further reduced penalty for learning
        drawdown_penalty = float(np.clip(self.current_drawdown * 5.0, 0.0, 1.5))  # Further reduced penalty for learning
        total_risk_penalty = risk_penalty + drawdown_penalty

        # ENHANCED: Reward calculation with forecast signal for AI value demonstration
        rw = self.reward_weights
        reward = (
            rw['nav_growth'] * nav_score +
            rw['cash_flow'] * cash_flow_score -
            rw['risk_adjusted'] * total_risk_penalty +
            rw['forecast'] * forecast_signal_score  # ENABLED: Reward accurate forecasting
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
    META_FREQ_MAX = 288     # daily
    META_CAP_MIN  = 0.02
    META_CAP_MAX  = 0.50
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
        self.administration_fee_rate = 0.0001    # 0.01% of fund value annually (basic admin)

        # OPERATIONAL costs (apply only after asset deployment)
        self.performance_fee_rate = 0.20         # 20% of profits above benchmark (institutional standard)
        self.trading_cost_rate = 0.001           # 0.1% of transaction value (trading costs)
        self.grid_connection_fee_mwh = 0.5       # $0.5/MWh grid connection fees
        self.transmission_fee_mwh = 1.2          # $1.2/MWh transmission fees

        # Battery physics/economics (PyPSA specifications: 10 MWh / 5 MW) - FROM CONFIG
        self.batt_eta_charge = 0.90     # charge efficiency (PyPSA: 90%)
        self.batt_eta_discharge = 0.90  # discharge efficiency (PyPSA: 90%)
        self.batt_degradation_cost = 1.0  # $ per MWh energy THROUGHPUT (reduced cost)
        self.batt_power_c_rate = 0.5    # max power: 5MW for 10MWh = 0.5 C-rate (PyPSA specifications)
        self.batt_soc_min = self.config.batt_soc_min
        self.batt_soc_max = self.config.batt_soc_max

        # CAPEX tables from config (override via asset_capex parameter)
        self.asset_capex = self.config.get_asset_capex()
        if isinstance(asset_capex, dict):
            try: self.asset_capex.update(asset_capex)
            except Exception: pass

        # Initialize fund
        self.budget = float(self.init_budget)
        self.equity = float(self.init_budget)
        
        # Legacy compatibility
        self.battery_capacity = 0.0

        # FIXED: Deploy initial assets (ONE-TIME ONLY)
        if not self.assets_deployed:
            self._deploy_initial_assets_once(initial_asset_plan)

        # FIXED: Currency conversion and data loading
        # Convert DKK prices to USD (Danish data) - SINGLE CONVERSION POINT
        DKK_TO_USD = self.config.dkk_to_usd_rate  # From config: 0.145 (1 USD = ~6.9 DKK)

        # vectorized series with currency conversion
        self._wind  = self.data.get('wind',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._solar = self.data.get('solar', pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._hydro = self.data.get('hydro', pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()

        # CLEAN APPROACH: Keep everything in DKK throughout the system
        price_dkk = self.data.get('price', pd.Series(250.0, index=self.data.index)).astype(float)

        # Step 1: Filter extreme outliers (keep realistic DKK range)
        price_dkk_filtered = np.clip(price_dkk, 10.0, 2000.0)  # Realistic DKK range

        # Step 2: Keep DKK throughout system - NO EARLY CONVERSION
        # Store conversion rate for final reporting only
        self._dkk_to_usd_rate = DKK_TO_USD

        # Log DKK price range for verification
        if hasattr(self, '_conversion_logged') and not self._conversion_logged:
            logging.info(f"Price system: Using DKK throughout, USD conversion rate = {DKK_TO_USD:.3f}")
            logging.info(f"DKK price range: {price_dkk_filtered.min():.1f}-{price_dkk_filtered.max():.1f} DKK/MWh")
            self._conversion_logged = True

        # Step 3: PRICE NORMALIZATION - All in DKK
        self._price_raw = price_dkk_filtered.to_numpy()  # Raw DKK prices for revenue calculation

        # Calculate rolling statistics for normalization (30-day window = 4320 timesteps)
        window_size = min(4320, len(price_dkk_filtered))
        price_rolling_mean = price_dkk_filtered.rolling(window=window_size, min_periods=1).mean()
        price_rolling_std = price_dkk_filtered.rolling(window=window_size, min_periods=1).std()

        # Normalize prices for agent observations (z-score with bounds)
        price_normalized = (price_dkk_filtered - price_rolling_mean) / (price_rolling_std + 1e-6)
        price_normalized_clipped = np.clip(price_normalized, -3.0, 3.0)  # ±3 sigma bounds

        self._price = price_normalized_clipped.to_numpy()  # Normalized for agents

        # Store normalization parameters for revenue calculations
        self._price_mean = price_rolling_mean.to_numpy()
        self._price_std = price_rolling_std.to_numpy()

        # IMPORTANT: Prices are now in USD, so currency_conversion = 1.0 in revenue calc
        # This prevents double conversion in _calculate_generation_revenue()

        # Initialize conversion logging flag
        self._conversion_logged = False

        self._load  = self.data.get('load',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._riskS = self.data.get('risk',  pd.Series(0.3, index=self.data.index)).astype(float).to_numpy()

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

        # FIXED: reward calc & risk controller with config
        self.reward_calculator = ProfitFocusedRewardCalculator(initial_budget=self.init_budget, config=self.config)
        # Store config reference for reward calculator
        if hasattr(self.reward_calculator, 'config'):
            self.reward_calculator.config = self.config
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
        self.max_leverage = self.config.max_leverage
        self.risk_multiplier = self.config.risk_multiplier

        # performance tracking
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}

        logging.info(f"Hybrid renewable fund initialized with ${self.init_budget:,.0f}")
        self._log_fund_structure()

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
            
            # Calculate total CAPEX required
            for asset_type, specs in plan.items():
                if asset_type == 'wind':
                    # Special handling for wind farm fractional ownership
                    # 75 MW = 5 WTGs × 15MW each × $25M per WTG = $125M
                    if specs['capacity_mw'] == 75.0:
                        capex = 125_000_000.0  # Fixed cost for 5 × 15MW WTGs
                    else:
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
            
            # Check budget sufficiency
            if total_capex > self.budget:
                logging.warning(f"Insufficient budget: ${total_capex:,.0f} required, ${self.budget:,.0f} available")
                # Scale down proportionally
                scale_factor = (self.budget * 0.9) / total_capex  # Use 90% of budget
                logging.info(f"Scaling asset plan by {scale_factor:.2f}")
            else:
                scale_factor = 1.0
            
            # Deploy physical assets
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
                    
                # Deduct from budget
                self.budget -= capex_used
                
            # Mark as deployed (PERMANENT)
            self.assets_deployed = True
            self.budget = max(0.0, self.budget)  # Ensure non-negative
            
            # Update equity
            self._calculate_fund_nav()
            
            logging.info(f"Asset deployment complete:")
            logging.info(f"  Wind: {self.physical_assets['wind_capacity_mw']:.1f} MW")
            logging.info(f"  Solar: {self.physical_assets['solar_capacity_mw']:.1f} MW")
            logging.info(f"  Hydro: {self.physical_assets['hydro_capacity_mw']:.1f} MW")
            logging.info(f"  Battery: {self.physical_assets['battery_capacity_mwh']:.1f} MWh")
            logging.info(f"  Total CAPEX: ${total_capex * scale_factor:,.0f}")
            logging.info(f"  Remaining cash: ${self.budget:,.0f}")

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

            logging.info("✅ Option 1 deployment verified successfully")
            
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
        logging.info(f"   Wind farms: {self.physical_assets['wind_capacity_mw']:.1f} MW (${self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw']:,.0f})")
        logging.info(f"   Solar farms: {self.physical_assets['solar_capacity_mw']:.1f} MW (${self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw']:,.0f})")
        logging.info(f"   Hydro plants: {self.physical_assets['hydro_capacity_mw']:.1f} MW (${self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw']:,.0f})")
        logging.info(f"   Battery storage: {self.physical_assets['battery_capacity_mwh']:.1f} MWh (${self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']:,.0f})")
        logging.info(f"   Total Physical: {total_physical_mw:.1f} MW (${physical_value:,.0f} book value)")
        logging.info("")

        # Financial instruments (derivatives trading)
        total_financial = sum(abs(v) for v in self.financial_positions.values())
        logging.info("2. FINANCIAL INSTRUMENTS (Derivatives Trading - Mark-to-Market):")
        logging.info(f"   Wind index exposure: ${self.financial_positions['wind_instrument_value']:,.0f}")
        logging.info(f"   Solar index exposure: ${self.financial_positions['solar_instrument_value']:,.0f}")
        logging.info(f"   Hydro index exposure: ${self.financial_positions['hydro_instrument_value']:,.0f}")
        logging.info(f"   Total Financial Exposure: ${total_financial:,.0f}")
        logging.info("")

        # Fund summary
        fund_nav = self._calculate_fund_nav()
        logging.info("3. FUND SUMMARY:")
        logging.info(f"   Cash position: ${self.budget:,.0f}")
        logging.info(f"   Physical assets (book): ${physical_value:,.0f}")
        logging.info(f"   Financial positions (MTM): ${total_financial:,.0f}")
        logging.info(f"   Total Fund NAV: ${fund_nav:,.0f}")
        logging.info(f"   Initial capital: ${self.init_budget:,.0f}")
        logging.info(f"   Total return: {((fund_nav - self.init_budget) / self.init_budget * 100):+.2f}%")
        logging.info("=" * 60)

    # =====================================================================
    # FIXED: PROPER NAV CALCULATION
    # =====================================================================
    
    def _calculate_fund_nav(self) -> float:
        """
        FIXED: Calculate true fund NAV with proper separation:
        NAV = Cash + Physical Asset Book Value + Financial Instrument MTM
        """
        try:
            # 1) Cash position
            cash_value = max(0.0, self.budget)
            
            # 2) Physical assets at book value (cost basis)
            physical_book_value = (
                self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] +
                self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] +
                self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] +
                self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']
            )
            
            # 3) Financial instruments at mark-to-market
            financial_mtm_value = (
                self.financial_positions['wind_instrument_value'] +
                self.financial_positions['solar_instrument_value'] +
                self.financial_positions['hydro_instrument_value']
            )
            
            # 4) Total NAV
            total_nav = cash_value + physical_book_value + financial_mtm_value
            
            # FIXED: Apply realistic bounds to prevent currency-induced explosions
            min_nav = self.init_budget * 0.1   # Minimum 10% of initial (more realistic)
            max_nav = self.init_budget * 3.0   # Maximum 300% of initial (more conservative)

            # Additional safety check for financial instrument values
            if abs(financial_mtm_value) > self.init_budget * 2.0:
                financial_mtm_value = float(np.clip(financial_mtm_value,
                                                  -self.init_budget * 0.5,
                                                   self.init_budget * 0.5))
                total_nav = cash_value + physical_book_value + financial_mtm_value

            self.equity = float(np.clip(total_nav, min_nav, max_nav))
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
        if self.t == 0:
            return
        try:
            for target in ['wind', 'solar', 'hydro', 'price', 'load']:
                forecast_key = f"{target}_forecast_immediate"
                if forecast_key not in forecasts:
                    continue

                # Normalize actual
                actual_raw = getattr(self, f"_{target}")[self.t] if hasattr(self, f"_{target}") else 0.0
                if target == "price":
                    actual_normalized = actual_raw / 100.0  # DKK scale: ~345 DKK/MWh → ~3.45
                elif target in ["wind", "solar", "hydro"]:
                    scale = getattr(self, f"{target}_scale", 1.0)
                    actual_normalized = actual_raw / max(scale, 1e-9)
                elif target == "load":
                    actual_normalized = actual_raw / max(getattr(self, "load_scale", 1.0), 1e-9)
                else:
                    actual_normalized = actual_raw

                # Normalize forecast to same scale (DKK system)
                fv = float(forecasts[forecast_key])
                if target == "price":
                    forecast_normalized = fv / 100.0  # DKK scale: ~345 DKK/MWh → ~3.45
                elif target in ["wind", "solar", "hydro"]:
                    scale = getattr(self, f"{target}_scale", 1.0)
                    forecast_normalized = fv / max(scale, 1e-9)
                elif target == "load":
                    forecast_normalized = fv / max(getattr(self, "load_scale", 1.0), 1e-9)
                else:
                    forecast_normalized = fv

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
        try:
            k = int(max(1, getattr(self, "investment_freq", 6)))
            if   k <= 6:  return 6
            elif k <= 12: return 12
            else:         return min(24, k)
        except Exception:
            return 6

    def _get_aligned_price_forecast(self, t: int, default: float = None) -> Optional[float]:
        if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
            return default
        try:
            h = self._aligned_horizon_steps()
            if hasattr(self.forecast_generator, "predict_all_horizons"):
                d = self.forecast_generator.predict_all_horizons(timestep=t)
                if isinstance(d, dict):
                    for k in (f"price_forecast_{h}", f"price_forecast_h{h}", f"price_h{h}", "price_forecast_aligned"):
                        if k in d and np.isfinite(d[k]): return float(d[k])
                    for k in ("price_forecast_immediate", "price_forecast_1", "price_h1"):
                        if k in d and np.isfinite(d[k]): return float(d[k])
            if hasattr(self.forecast_generator, "predict_for_agent"):
                d = self.forecast_generator.predict_for_agent(agent="investor_0", timestep=t)
                if isinstance(d, dict):
                    for k in (f"price_forecast_{h}", f"price_forecast_h{h}", f"price_h{h}", "price_forecast_aligned"):
                        if k in d and np.isfinite(d[k]): return float(d[k])
                    for k in ("price_forecast_immediate", "price_forecast_1", "price_h1"):
                        if k in d and np.isfinite(d[k]): return float(d[k])
        except Exception:
            pass
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

        self.t = 0
        self.step_in_episode = 0
        self.agents = self.possible_agents[:]
        
        # FIXED: Reset financial state but keep physical assets
        self._reset_financial_state()
        
        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        self._fill_obs()
        return self._obs_buf, {}

    def _reset_financial_state(self):
        """FIXED: Reset financial state while preserving physical assets"""
        # Keep physical assets (they are permanent)
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
        
        # Reset cash to remaining amount after CAPEX
        if self.assets_deployed:
            total_capex = (
                self.physical_assets['wind_capacity_mw'] * self.asset_capex['wind_mw'] +
                self.physical_assets['solar_capacity_mw'] * self.asset_capex['solar_mw'] +
                self.physical_assets['hydro_capacity_mw'] * self.asset_capex['hydro_mw'] +
                self.physical_assets['battery_capacity_mwh'] * self.asset_capex['battery_mwh']
            )
            self.budget = max(0.0, self.init_budget - total_capex)
        else:
            self.budget = self.init_budget
        
        # Reset performance tracking
        self.investment_capital = float(self.init_budget)
        self.distributed_profits = 0.0
        self.cumulative_returns = 0.0
        self.last_revenue = 0.0
        self.last_generation_revenue = 0.0
        self.last_mtm_pnl = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = {}
        
        # Calculate initial NAV
        self._calculate_fund_nav()

    def step(self, actions: Dict[str, Any]):
        if self.t >= self.max_steps:
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
        # FIXED: Handle normalized [-1,1] action space
        a0, a1 = np.array(meta_action, dtype=np.float32).reshape(-1)[:2]
        # Convert [-1,1] to [0,1] for capital allocation
        normalized_a0 = (float(np.clip(a0, -1.0, 1.0)) + 1.0) / 2.0  # [-1,1] -> [0,1]
        cap = self.META_CAP_MIN + normalized_a0 * (self.META_CAP_MAX - self.META_CAP_MIN)
        self.capital_allocation_fraction = float(np.clip(cap, self.META_CAP_MIN, self.META_CAP_MAX))
        freq = int(round(self.META_FREQ_MIN + float(np.clip(a1, 0.0, 1.0)) * (self.META_FREQ_MAX - self.META_FREQ_MIN)))
        self.investment_freq = int(np.clip(freq, self.META_FREQ_MIN, self.META_FREQ_MAX))

    # ------------------------------------------------------------------
    # FIXED: Financial Instrument Trading (Separate from Physical Assets)
    # ------------------------------------------------------------------
    
    def _execute_investor_trades(self, inv_action: np.ndarray) -> float:
        """
        IMPROVED: Execute trades with forecast confidence and volatility controls
        Returns total traded notional for transaction costs
        """
        if self.t % self.investment_freq != 0:
            return 0.0  # No trading outside frequency

        # Check if trading is disabled due to drawdown
        if not self.reward_calculator.trading_enabled:
            return 0.0  # No trading during high drawdown periods

        try:
            # Check if we're in ultra fast mode
            ultra_fast_mode = getattr(self, 'ultra_fast_mode', False)

            if ultra_fast_mode:
                # Ultra fast mode: enable trading regardless of forecast confidence
                forecast_confidence = 1.0  # Override confidence for ultra fast mode
            else:
                # Normal mode: get actual forecast confidence
                forecast_confidence = self._get_forecast_confidence()

                # Only trade if forecast confidence exceeds threshold
                if forecast_confidence < self.reward_calculator.forecast_confidence_threshold:
                    return 0.0  # Skip trading on low confidence forecasts

            # Convert action to target weights
            action_array = np.array(inv_action, dtype=np.float32).reshape(-1)[:3]
            weights = (action_array + 1.0) / 2.0  # [-1,1] -> [0,1]
            weights = weights / max(np.sum(weights), 1e-6)  # Normalize

            # Volatility-based position sizing
            volatility_factor = self._get_volatility_factor()
            position_size_multiplier = self._calculate_position_size_multiplier(volatility_factor)

            # Available capital for financial instruments (separate from physical assets)
            available_capital = self.budget * self.capital_allocation_fraction
            base_target_gross = available_capital * 0.8  # Use 80% max

            # Adjust target gross based on volatility and confidence
            target_gross = base_target_gross * position_size_multiplier * forecast_confidence

            # Current financial positions
            current_wind = self.financial_positions['wind_instrument_value']
            current_solar = self.financial_positions['solar_instrument_value']
            current_hydro = self.financial_positions['hydro_instrument_value']

            # Target positions
            target_wind = target_gross * weights[0]
            target_solar = target_gross * weights[1]
            target_hydro = target_gross * weights[2]

            # Calculate trades needed
            trade_wind = target_wind - current_wind
            trade_solar = target_solar - current_solar
            trade_hydro = target_hydro - current_hydro

            # Check budget constraints for net purchases
            total_purchases = max(0, trade_wind) + max(0, trade_solar) + max(0, trade_hydro)
            total_sales = max(0, -trade_wind) + max(0, -trade_solar) + max(0, -trade_hydro)
            net_cash_needed = total_purchases - total_sales

            if net_cash_needed > self.budget:
                # Scale down trades
                scale = self.budget / max(net_cash_needed, 1e-6)
                trade_wind *= scale
                trade_solar *= scale
                trade_hydro *= scale

            # Execute trades (update financial positions)
            self.financial_positions['wind_instrument_value'] += trade_wind
            self.financial_positions['solar_instrument_value'] += trade_solar
            self.financial_positions['hydro_instrument_value'] += trade_hydro

            # RESTORED: Unlimited financial exposure like PrototypeTestTuned for maximum profitability
            # Removed 50% cap to allow aggressive trading strategies

            # Update cash (cash decreases for purchases, increases for sales)
            cash_flow = -(max(0, trade_wind) + max(0, trade_solar) + max(0, trade_hydro)) + \
                        (max(0, -trade_wind) + max(0, -trade_solar) + max(0, -trade_hydro))
            self.budget += cash_flow
            self.budget = max(0.0, self.budget)

            return abs(trade_wind) + abs(trade_solar) + abs(trade_hydro)

        except Exception as e:
            logging.error(f"Trading execution error: {e}")
            return 0.0

    def _get_forecast_confidence(self) -> float:
        """Get forecast confidence score [0,1]"""
        try:
            if not hasattr(self, "forecast_generator") or self.forecast_generator is None:
                return 0.5  # Default confidence when no forecaster

            if hasattr(self.forecast_generator, "predict_all_horizons"):
                forecasts = self.forecast_generator.predict_all_horizons(timestep=self.t)
                if isinstance(forecasts, dict):
                    # Look for confidence scores in forecast output
                    for key in ['confidence', 'forecast_confidence', 'price_confidence']:
                        if key in forecasts and np.isfinite(forecasts[key]):
                            return float(np.clip(forecasts[key], 0.0, 1.0))

                    # Calculate confidence based on forecast consistency
                    price_forecasts = []
                    for key in forecasts:
                        if 'price' in key and np.isfinite(forecasts[key]):
                            price_forecasts.append(forecasts[key])

                    if len(price_forecasts) > 1:
                        # Higher consistency = higher confidence
                        std_dev = np.std(price_forecasts)
                        mean_val = np.mean(np.abs(price_forecasts))
                        if mean_val > 0:
                            cv = std_dev / mean_val  # Coefficient of variation
                            confidence = max(0.0, 1.0 - cv)  # Lower CV = higher confidence
                            return float(np.clip(confidence, 0.0, 1.0))

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
        Decide ('charge'/'discharge'/'idle', intensity 0..1) from price now vs. aligned forecast.
        Uses aligned horizon to match investment_freq.
        """
        try:
            p_now = float(np.clip(self._price[i], -1000.0, 1e9))
            p_fut = self._get_aligned_price_forecast(i, default=None)
            if p_fut is None or not np.isfinite(p_fut):
                return ("idle", 0.0)

            spread = (p_fut - p_now)
            needed = max(0.005 * abs(p_now), 0.5)  # ≥ $0.5/MWh or ~0.5% (more aggressive)
            rt_loss = (1.0/(max(self.batt_eta_charge*self.batt_eta_discharge, 1e-6)) - 1.0) * abs(p_now)
            hurdle = needed + 0.3 * rt_loss  # reduced hurdle for more trading

            if spread > hurdle:
                inten = float(np.clip(spread / (abs(p_now) + 1e-6), 0.0, 1.0))
                return ("charge", inten)
            elif spread < -hurdle:
                inten = float(np.clip((-spread) / (abs(p_now) + 1e-6), 0.0, 1.0))
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

            price = float(np.clip(self._price[i], -1000.0, 1e9))
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
            cap_min = getattr(self.config, 'mtm_price_return_cap_min', -0.02)
            cap_max = getattr(self.config, 'mtm_price_return_cap_max', 0.02)
            capped_price_return = float(np.clip(price_return, cap_min, cap_max))
            mtm_pnl = total_financial_exposure * capped_price_return

            # Apply MTM to financial positions with capped returns
            self.financial_positions['wind_instrument_value'] *= (1.0 + capped_price_return)
            self.financial_positions['solar_instrument_value'] *= (1.0 + capped_price_return)
            self.financial_positions['hydro_instrument_value'] *= (1.0 + capped_price_return)
            
            # 3) Transaction costs (CASH FLOW)
            txn_costs = 0.0005 * trade_amount
            
            # 4) Battery operational costs (CASH FLOW)
            battery_opex = 0.0002 * self.physical_assets['battery_capacity_mwh']
            
            # 5) Net cash flow this step (REMOVED admin costs for better agent learning)
            net_cash_flow = generation_revenue + battery_cash_delta - txn_costs - battery_opex

            # 7) Update cash position
            self.budget = max(0.0, self.budget + net_cash_flow)
            
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
            
            # 11) Calculate forecast signal if available
            pf_aligned = self._get_aligned_price_forecast(i, default=None)
            if pf_aligned is not None and np.isfinite(pf_aligned):
                forecast_signal_score = float(np.sign(pf_aligned - current_price) * price_return)
            else:
                forecast_signal_score = 0.0
            
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
            time_step_hours = 10.0 / 60.0  # 10-minute timesteps

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

            # Fractional ownership percentages (applied only to revenue)
            wind_ownership_pct = 0.05   # 5% ownership of wind farm
            solar_ownership_pct = 0.05  # 5% ownership of solar farm
            hydro_ownership_pct = 0.02  # 2% ownership of hydro plant

            # Fund's revenue share (fractional ownership applied here only)
            fund_wind_generation_mwh = full_wind_generation_mwh * wind_ownership_pct
            fund_solar_generation_mwh = full_solar_generation_mwh * solar_ownership_pct
            fund_hydro_generation_mwh = full_hydro_generation_mwh * hydro_ownership_pct
            fund_total_generation_mwh = fund_wind_generation_mwh + fund_solar_generation_mwh + fund_hydro_generation_mwh

            # Safety check
            if fund_total_generation_mwh <= 0.001:
                return 0.0

            # Revenue from electricity sales (based on fund's fractional ownership)
            # NOTE: price is already in USD (converted once during data loading)
            # currency_conversion = 1.0 to prevent double conversion
            gross_revenue = fund_total_generation_mwh * price * self.electricity_markup * self.currency_conversion

            # Operating costs (based on fund's actual generation)
            variable_costs = gross_revenue * self.operating_cost_rate
            maintenance_costs = fund_total_generation_mwh * self.maintenance_cost_mwh
            # REMOVED: grid_connection_costs = fund_total_generation_mwh * self.grid_connection_fee_mwh
            # REMOVED: transmission_costs = fund_total_generation_mwh * self.transmission_fee_mwh

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
            # REMOVED: property_taxes = fund_physical_investment * self.property_tax_rate * annual_to_timestep
            # REMOVED: debt_service = fund_physical_investment * self.debt_service_rate * annual_to_timestep

            # RESTORED: Management fees on total fund like PrototypeTestBaseline/Tuned for consistent behavior
            management_fees = self.init_budget * self.management_fee_rate * annual_to_timestep

            # REMOVED: All admin costs for better agent learning environment
            # regulatory_costs = self.init_budget * self.regulatory_compliance_rate * annual_to_timestep
            # audit_legal_costs = self.init_budget * self.audit_legal_rate * annual_to_timestep
            # custody_fees = self.init_budget * self.custody_fee_rate * annual_to_timestep
            # administration_fees = self.init_budget * self.administration_fee_rate * annual_to_timestep

            # REMOVED: Trading costs for better agent learning
            # trading_costs = abs(gross_revenue) * self.trading_cost_rate * 0.1  # 10% of revenue involves trading

            total_operating_costs = (variable_costs + maintenance_costs + insurance_costs + management_fees)

            net_revenue = max(0.0, gross_revenue - total_operating_costs)
            return float(net_revenue)

        except Exception as e:
            logging.warning(f"Generation revenue calculation failed: {e}")
            return 0.0

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
            # Scale to realistic wind CF range (10-35%) - more conservative
            normalized = raw_wind / max(self.wind_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.10 + (normalized * 0.25)  # Map to 10-35% range
            return float(np.clip(realistic_cf, 0.0, 0.35))
        except Exception:
            return 0.25  # Typical wind CF

    def _get_solar_capacity_factor(self, i: int) -> float:
        try:
            raw_solar = float(self._solar[i]) if i < len(self._solar) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic solar CF range (0-25%) - more conservative
            normalized = raw_solar / max(self.solar_scale, 1e-6)  # 0-1 from data range
            realistic_cf = normalized * 0.25  # Map to 0-25% range
            return float(np.clip(realistic_cf, 0.0, 0.25))
        except Exception:
            return 0.15  # Typical solar CF

    def _get_hydro_capacity_factor(self, i: int) -> float:
        try:
            raw_hydro = float(self._hydro[i]) if i < len(self._hydro) else 0.0
            # FIXED: Realistic capacity factor calculation
            # Scale to realistic hydro CF range (25-55%) - more conservative
            normalized = raw_hydro / max(self.hydro_scale, 1e-6)  # 0-1 from data range
            realistic_cf = 0.25 + (normalized * 0.30)  # Map to 25-55% range
            return float(np.clip(realistic_cf, 0.25, 0.55))
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
            reward = self.reward_calculator.calculate_reward(
                fund_nav=fund_nav,
                cash_flow=cash_flow, 
                risk_level=risk_level,
                efficiency=efficiency,
                forecast_signal_score=forecast_signal_score
            )
            
            # FIXED: Less aggressive reward clipping for better learning signals
            clipped_reward = float(np.clip(reward, -20.0, 20.0))
            
            for agent in self.possible_agents:
                self._rew_buf[agent] = clipped_reward
            
            # Store reward breakdown for logging
            self.last_reward_breakdown = {
                'total_reward': reward,
                'fund_nav': fund_nav,
                'cash_flow': cash_flow,
                'risk_level': risk_level,
                'efficiency': efficiency,
                'forecast_signal_score': forecast_signal_score,
                'reward_components': self.reward_calculator.reward_weights.copy()
            }
            self.last_reward_weights = self.reward_calculator.reward_weights.copy()
            
        except Exception as e:
            logging.error(f"Reward assignment error: {e}")
            for agent in self.possible_agents:
                self._rew_buf[agent] = 0.0

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
                    'price': float(self._price[i]),
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
        price_n = float(np.clip(SafeDivision.div(self._price[i], 10.0, 0.0), -10.0, 10.0))
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
                    'price': float(self._price[i]),
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
        
        # Check financial positions are within reasonable bounds
        total_financial = sum(abs(v) for v in self.financial_positions.values())
        if total_financial > self.init_budget * 2:  # Allow 2x leverage
            issues.append(f"Excessive financial exposure: {total_financial:.0f}")
        
        # Check fund NAV is reasonable
        nav = self._calculate_fund_nav()
        if nav < self.init_budget * 0.01 or nav > self.init_budget * 10:
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