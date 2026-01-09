from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED CONSTANTS (SINGLE SOURCE OF TRUTH)
# ============================================================================
# These constants ensure consistent configuration across all modes and code paths

# Overlay feature dimension (CANONICAL)
OVERLAY_FEATURE_DIM = 34    # 28D base features + 6D deltas (price, direction, mwdir)

# Price normalization constants
PRICE_MEAN = 250.0          # Historical mean price (DKK/MWh)
PRICE_STD = 50.0            # Historical std dev (DKK/MWh)
PRICE_CLIP_SIGMA = 3.0      # Clip at ±3 sigma for outlier handling

# Risk controller constants
RISK_LOOKBACK_WINDOW_DEFAULT = 144  # Default lookback window for risk calculations
RISK_LOOKBACK_WINDOW_MAX = 200      # Maximum lookback window (memory cap)

# Adaptive scaling constants
ADAPTIVE_SCALE_SATURATION_THRESHOLD = 0.95  # Threshold for adaptive scaling saturation prevention
ADAPTIVE_SCALE_MIN_OFFSET = 0.01            # Small offset from min/max boundaries
ADAPTIVE_SCALE_COMPRESSION_FACTOR = 5.0     # Compression factor for values near boundaries

# Regulatory risk constants
REGULATORY_RISK_BASE_STRESS_WEIGHT = 0.4    # Weight for market stress in regulatory risk
REGULATORY_RISK_FALLBACK = 0.20             # Fallback regulatory risk when no history
REGULATORY_RISK_SEASON_BASE = 0.05          # Base seasonal component
REGULATORY_RISK_SEASON_AMPLITUDE = 0.05     # Amplitude of seasonal variation
REGULATORY_RISK_FALLBACK_ERROR = 0.35       # Fallback value on calculation error

# Risk fallback values (on calculation errors)
RISK_FALLBACK_MARKET = 0.30
RISK_FALLBACK_OPERATIONAL = 0.20
RISK_FALLBACK_PORTFOLIO = 0.25
RISK_FALLBACK_LIQUIDITY = 0.15
RISK_FALLBACK_REGULATORY = 0.35
RISK_FALLBACK_OVERALL = 0.25

# Risk action defaults
RISK_ACTION_MULTIPLIER_DEFAULT = 1.0
RISK_ACTION_MAX_INVESTMENT_DEFAULT = 0.30
RISK_ACTION_CASH_RESERVE_DEFAULT = 0.10
RISK_ACTION_HEDGE_DEFAULT = 0.50
RISK_ACTION_REBALANCE_DEFAULT = 0.30
RISK_ACTION_TOLERANCE_DEFAULT = 0.70

# DL Overlay constants
DL_OVERLAY_WINDOW_SIZE_DEFAULT = 2016       # ~2 weeks @ 10-min steps
DL_OVERLAY_INIT_BUDGET_DEFAULT = 1e8        # 100M DKK default budget
DL_OVERLAY_DIRECTION_WEIGHT_DEFAULT = 0.8   # 80% direction, 20% magnitude

# DL Overlay loss weights (28D mode) - normalized to sum to 1.0
DL_OVERLAY_LOSS_WEIGHTS = {
    'bridge_vec': 0.083,       # ~8.3%: Bridge vector guidance
    'risk_budget': 0.167,      # ~16.7%: Risk budget allocation
    'pred_reward': 0.167,      # ~16.7%: Predicted reward
    'strat_immediate': 0.208,  # ~20.8%: HIGHEST - Full forecast, most actionable
    'strat_short': 0.167,      # ~16.7%: HIGH - 95% confidence tactical adjustments
    'strat_medium': 0.125,     # ~12.5%: MEDIUM - 90% confidence strategic shifts
    'strat_long': 0.083,       # ~8.3%: LOW - Risk-only long-term positioning
}  # Total: 1.000

# Wrapper cache defaults (fallback when no config)
WRAPPER_FORECAST_CACHE_SIZE_DEFAULT = 1000
WRAPPER_AGENT_CACHE_SIZE_DEFAULT = 2000
WRAPPER_MEMORY_LIMIT_MB_DEFAULT = 1024.0

# Environment constants
ENV_MARKET_STRESS_DEFAULT = 0.5
ENV_OVERALL_RISK_DEFAULT = 0.5
ENV_MARKET_RISK_DEFAULT = 0.5
ENV_POSITION_EXPOSURE_THRESHOLD = 0.001     # Threshold for exploration bonus
ENV_EXPLORATION_BONUS_MULTIPLIER = 0.5      # Multiplier for exploration bonus

def normalize_price(price_raw: float) -> float:
    """
    UNIFIED PRICE NORMALIZATION (SINGLE SOURCE OF TRUTH)

    Converts raw price to [-1, 1] range using z-score normalization with clipping.
    This function is used everywhere in the codebase to ensure consistency.

    Uses z-score normalization with clipping:
    - Converts raw price to standard deviations from mean
    - Clips extreme values to prevent outliers
    - Divides by PRICE_CLIP_SIGMA to get [-1,1] range

    Args:
        price_raw: Raw price in DKK/MWh

    Returns:
        Normalized price in [-1.0, 1.0] range
    """
    import numpy as np
    z_score = (price_raw - PRICE_MEAN) / PRICE_STD
    z_clipped = np.clip(z_score, -PRICE_CLIP_SIGMA, PRICE_CLIP_SIGMA)
    return float(z_clipped / PRICE_CLIP_SIGMA)


class EnhancedConfig:
    """Enhanced configuration class with optimization support and centralized hardcoded values"""

    # Configuration version for validation
    CONFIG_VERSION = "2.2.0"  # Updated for unified normalization

    def __init__(self, optimized_params: Optional[Dict[str, Any]] = None):
        # (Removed) Fair-comparison flag: fairness is now the default behavior.
        # =============================================================================
        # FUND STRUCTURE AND ECONOMICS
        # =============================================================================

        # Currency conversion (DKK to USD) - SINGLE SOURCE OF TRUTH
        self.dkk_to_usd_rate = 0.145  # 1 USD = ~6.9 DKK (2024 rate)
        self.currency_conversion = 1.0  # Keep at 1.0 - all calculations in DKK, convert only for final reporting

        # Fund size and allocation
        # UPDATED: New fund size $800M with updated allocations
        self.init_budget_usd = 8e8  # $800M fund size in USD
        self.init_budget = self.init_budget_usd / self.dkk_to_usd_rate  # Convert to DKK (~4.83B DKK)

        # Calculate actual allocations based on new fund structure
        # Physical CAPEX: $704M (88%), Trading capital: $96M (12%)
        self.physical_allocation = 0.88  # 88% physical assets ($704M of $800M)
        self.financial_allocation = 0.12  # 12% financial instruments ($96M of $800M)

        # Asset ownership fractions (of total installed capacity) - REALISTIC with market-rate CAPEX
        self.wind_ownership_fraction = 0.18   # 18% of 1,500MW wind farm (270MW owned)
        self.solar_ownership_fraction = 0.10  # 10% of 1,000MW solar farm (100MW owned)
        self.hydro_ownership_fraction = 0.04  # 4% of 1,000MW hydro plant (40MW owned)

        # Total installed capacities (MW)
        self.total_wind_capacity_mw = 1500.0  # Total wind farm capacity
        self.total_solar_capacity_mw = 1000.0  # Total solar farm capacity
        self.total_hydro_capacity_mw = 1000.0  # Total hydro plant capacity

        # Derived owned capacities (calculated from fractions)
        self.owned_wind_capacity_mw = self.wind_ownership_fraction * self.total_wind_capacity_mw  # 270 MW
        self.owned_solar_capacity_mw = self.solar_ownership_fraction * self.total_solar_capacity_mw  # 100 MW
        self.owned_hydro_capacity_mw = self.hydro_ownership_fraction * self.total_hydro_capacity_mw  # 40 MW
        self.owned_battery_capacity_mwh = 10.0  # 10 MWh direct ownership

        # CAPEX values ($/MW or $/MWh) - REALISTIC MARKET RATES
        self.wind_capex_per_mw = 2000000.0   # $2.0M/MW (market rate)
        self.solar_capex_per_mw = 1000000.0  # $1.0M/MW (market rate)
        self.hydro_capex_per_mw = 1500000.0  # $1.5M/MW (market rate)
        self.battery_capex_per_mwh = 400000.0  # $400k/MWh (market rate)

        # Operating costs and fees - FIXED: Convert USD to DKK
        self.operating_cost_rate = 0.025  # 3% of revenue
        self.maintenance_cost_mwh = 3.5 / self.dkk_to_usd_rate  # $3.5/MWh → ~24.1 DKK/MWh
        self.insurance_rate = 0.004  # 0.4% of asset value annually
        self.management_fee_rate = 0.01  # 1.5% of fund value annually (realistic for infrastructure)
        self.property_tax_rate = 0.005  # 0.5% of asset value annually
        self.debt_service_rate = 0.015  # 1.5% of asset value annually (realistic debt service)
        self.distribution_rate = 0.10  # 10% of excess cash distributed
        self.target_cash_ratio = 0.15  # FIXED: Increased to 15% to give agents more working capital
        self.min_distribution_threshold_ratio = 0.01  # FIXED: Increased to 1% to reduce distribution frequency
        self.administration_fee_rate = 0.0001  # 0.01% of fund value annually (basic admin)

        # Grid and transmission fees - FIXED: Convert USD to DKK
        self.grid_connection_fee_mwh = 0.5 / self.dkk_to_usd_rate  # $0.5/MWh → ~3.5 DKK/MWh
        self.transmission_fee_mwh = 1.2 / self.dkk_to_usd_rate    # $1.2/MWh → ~8.3 DKK/MWh

        # Battery costs and parameters - FIXED: Convert USD to DKK
        self.battery_degradation_cost_mwh = 1.0 / self.dkk_to_usd_rate  # $1/MWh → ~6.9 DKK/MWh
        self.batt_soc_min = 0.1  # 10% minimum state of charge
        self.batt_soc_max = 0.9  # 90% maximum state of charge
        self.batt_eta_charge = 0.90     # charge efficiency (PyPSA: 90%)
        self.batt_eta_discharge = 0.90  # discharge efficiency (PyPSA: 90%)
        self.batt_power_c_rate = 0.5    # max power: 5MW for 10MWh = 0.5 C-rate (PyPSA specifications)

        # Trading costs - FIXED: Convert USD to DKK
        self.transaction_fixed_cost = 25.0 / self.dkk_to_usd_rate  # $25/trade → ~172 DKK/trade
        self.transaction_cost_bps = 0.5  # 0.5 basis points (institutional rates)

        # Battery dispatch economic thresholds
        self.battery_hurdle_min_dkk = 5.0  # Minimum hurdle rate in DKK/MWh
        self.battery_price_sensitivity = 0.02  # Price-based hurdle factor
        self.battery_rt_loss_weight = 0.3  # Weight of round-trip loss in hurdle calculation

        # Battery operational parameters - FIXED: Convert USD to DKK
        self.battery_opex_rate = 0.0002  # Battery operational cost rate per MWh capacity
        self.performance_fee_rate = 0.20  # 20% of profits above benchmark (institutional standard)
        self.trading_cost_rate = 0.001   # 0.1% of transaction value (trading costs)

        # Price and revenue parameters
        self.minimum_price_floor = 50.0  # Minimum price floor in DKK/MWh
        self.maximum_price_cap = 2000.0  # Maximum price cap in DKK/MWh for filtering
        self.minimum_price_filter = 10.0  # Minimum price for data filtering in DKK/MWh
        self.time_step_hours = 10.0 / 60.0  # 10-minute timesteps in hours

        # Financial parameters - INFRASTRUCTURE FUND FOCUS
        self.max_leverage = 1.05  # INFRASTRUCTURE: Minimal leverage (5% max) typical for conservative infrastructure funds
        self.electricity_markup = 1.0  # Fund sells at market price

        # ENHANCED: Operational revenue reallocation policy - DISABLED FOR CAPITAL PRESERVATION
        self.allow_operational_reallocation = False  # DISABLED: Prevent unrealistic cash accumulation
        self.max_reallocation_rate = 0.0  # DISABLED: No reallocation allowed
        self.trading_capital_emergency_threshold = 0.50  # Not used when disabled
        self.max_total_reallocation = 0.0  # DISABLED: No lifetime reallocation

        # Mark-to-market volatility controls - REALISTIC (10-minute intervals)
        self.mtm_price_return_cap_min = -0.001  # REALISTIC: -0.1% maximum loss per 10-min step (≈-14% daily)
        self.mtm_price_return_cap_max = 0.001   # REALISTIC: +0.1% maximum gain per 10-min step (≈+14% daily)

        # FAIR SPARSE REWARD FIXES: Multi-step returns and MTM inclusion
        # These apply to ALL tiers equally, making rewards less sparse without unfair advantages
        # REDUCED: Smaller window (2 instead of 3) to preserve reward signal strength
        # REDUCED: Lower MTM weight (0.2 instead of 0.3) to reduce noise
        self.pnl_reward_multi_step_window = 2  # Use rolling average of 2 timesteps (reduced from 3 to preserve signal)
        self.pnl_reward_include_mtm = True  # Include unrealized PnL delta in rewards (fair for all tiers)
        self.pnl_reward_mtm_weight = 0.2  # Weight for MTM component (20% MTM, 80% realized return) - reduced from 0.3 to reduce noise

        # MTM threshold for position updates
        self.mtm_update_threshold = 1e-9  # Threshold for applying MTM updates to positions

        # NAV bounds for renewable energy fund - INFRASTRUCTURE REALISTIC
        # NOTE: These are currently UNUSED by design (THESIS MODE)
        # The environment intentionally uses unconstrained NAV to allow realistic fund dynamics
        # Only prevents negative NAV; no artificial min/max bounds applied
        self.nav_min_ratio = 0.90  # INFRASTRUCTURE: Minimum 90% of initial (conservative downside protection) [UNUSED]
        self.nav_max_ratio = 1.10  # INFRASTRUCTURE: Maximum 110% of initial (5% return cap - very realistic for infrastructure) [UNUSED]



        # =============================================================================
        # ENVIRONMENT AND SIMULATION PARAMETERS
        # =============================================================================

        # Meta controller ranges (28D Forecast-Aware Mode)
        self.meta_freq_min = 6  # Every hour if 10-min data
        self.meta_freq_max = 144  # 24 hours max (more responsive)
        self.meta_cap_min = 0.05  # 5% minimum capital allocation
        self.meta_cap_max = 0.75  # 75% maximum capital allocation
        self.sat_eps = 1e-3

        # Investment and operational parameters - FORECAST OPTIMIZATION
        self.investment_freq = 6   # CAPITAL PRESERVATION: Every 4 hours for careful positioning
        self.min_investment_freq = 4   # CAPITAL PRESERVATION: 2 hour minimum
        self.max_investment_freq = 12  # CAPITAL PRESERVATION: 48 hours maximum (very patient)
        self.capital_allocation_fraction = 0.40  # INCREASED: 40% (from 35%) to allow larger positions for Tier 2/3 forecast learning
        self.risk_multiplier = 0.5  # CAPITAL PRESERVATION: Minimal risk multiplier
        self.no_trade_threshold = 0.02  # 2% of target position - trades below this are ignored

        # Battery operational parameters
        self.batt_soc_min = 0.1  # 10% minimum state of charge
        self.batt_soc_max = 0.9  # 90% maximum state of charge
        self.batt_efficiency = 0.85  # 85% round-trip efficiency

        # =============================================================================
        # MEMORY AND PERFORMANCE PARAMETERS
        # =============================================================================

        # Memory limits (MB) - INCREASED FOR PERFORMANCE
        self.max_memory_mb = 6000.0  # Environment memory limit (INCREASED)
        self.metacontroller_memory_mb = 12000  # Meta controller memory limit (INCREASED)
        self.wrapper_memory_mb = 2000  # Wrapper memory limit (INCREASED)

        # Cache sizes - INCREASED FOR PERFORMANCE
        self.forecast_cache_size = 2000  # Increased for better performance
        self.agent_forecast_cache_size = 4000  # Increased for better performance
        self.lru_cache_size = 3000  # Increased for better performance
        self.lru_memory_limit_mb = 100.0  # Increased for better performance

        # =============================================================================
        # RISK MANAGEMENT PARAMETERS
        # =============================================================================

        # Risk calculation lookback
        self.risk_lookback_window = 96  # OPTIMIZED: Reduced from 144 to 96 for more responsive risk management

        # OPERATIONAL EXCELLENCE: Capital preservation risk management
        self.max_drawdown_threshold = 0.10  # CAPITAL PRESERVATION: 10% maximum drawdown
        self.volatility_lookback = 60  # CAPITAL PRESERVATION: 60 days for very stable calculations
        self.max_position_size = 0.08  # INCREASED: 8% maximum single position size (from 5%) to encourage larger positions for Tier 2/3
        self.volatility_scaling = 0.8   # CAPITAL PRESERVATION: High scaling for minimal risk
        self.target_sharpe_ratio = 2.0  # CAPITAL PRESERVATION: High target for excellent risk-adjusted returns
        self.risk_free_rate = 0.02      # 2% risk-free rate (Danish government bonds)
        self.max_asset_correlation = 0.3 # CAPITAL PRESERVATION: Low correlation for maximum diversification

        # SINGLE-PRICE HEDGING: Risk budget allocation for hedge distribution (CONSOLIDATED)
        self.risk_budget_allocation = {
            'wind': 0.40,      # 40% of risk budget to wind (highest volatility)
            'solar': 0.35,     # 35% of risk budget to solar (medium volatility)
            'hydro': 0.25,     # 25% of risk budget to hydro (lowest volatility)
        }

        # OPERATIONAL EXCELLENCE: Renewable energy fund focus parameters
        self.operational_revenue_target = 1200.0   # RENEWABLE ENERGY: Realistic 1200 DKK per step target (~6% annual return)
        self.hedging_effectiveness_target = 0.95  # CAPITAL PRESERVATION: 95% hedge effectiveness target
        self.max_portfolio_volatility = 0.06      # CAPITAL PRESERVATION: Maximum 6% annual volatility
        self.operational_revenue_weight = 0.95    # CAPITAL PRESERVATION: 95% focus on operations vs 5% trading

        # Risk weights (should sum to ~1.0)
        self.risk_weights = {
            'market': 0.25,
            'operational': 0.20,
            'portfolio': 0.25,
            'liquidity': 0.15,
            'regulatory': 0.15
        }

        # Risk thresholds - OPTIMIZED for better risk-return balance
        self.risk_thresholds = {
            'market_stress_high': 0.80,      # OPTIMIZED: Reduced from 0.85 for earlier risk response
            'market_stress_medium': 0.60,    # OPTIMIZED: Reduced from 0.65 for better sensitivity
            'volatility_high': 0.75,         # OPTIMIZED: Reduced from 0.80 for better volatility control
            'volatility_medium': 0.45,       # OPTIMIZED: Reduced from 0.50 for earlier intervention
            'portfolio_concentration_high': 0.70,  # OPTIMIZED: Reduced from 0.75 for better diversification
            'liquidity_stress_high': 0.75,   # OPTIMIZED: Reduced from 0.80 for better liquidity management
        }

        # Risk calculation component weights - PHASE 5.10 FIX: Parameterized risk model
        self.market_risk_volatility_weight = 0.6
        self.market_risk_momentum_weight = 0.4
        self.market_risk_momentum_factor = 0.5
        self.operational_risk_volatility_weight = 0.7
        self.operational_risk_intermittency_weight = 0.3
        self.portfolio_risk_concentration_weight = 0.6
        self.portfolio_risk_capital_weight = 0.4
        self.liquidity_risk_buffer_weight = 0.6
        self.liquidity_risk_cashflow_weight = 0.4

        # REMOVED: Duplicate risk_budget_allocation (consolidated above)

        # NEW: Infrastructure fund performance targets
        self.annual_return_target = 0.06   # INFRASTRUCTURE: 6% annual return target (conservative infrastructure)
        self.max_annual_volatility = 0.08  # INFRASTRUCTURE: 8% maximum annual volatility (very stable)
        self.min_sharpe_ratio = 1.5        # INFRASTRUCTURE: High risk-adjusted returns for conservative funds

        # Physical asset depreciation parameters
        self.annual_depreciation_rate = 0.02  # 2% annual straight-line depreciation
        self.max_depreciation_ratio = 0.75  # Maximum 75% depreciation over asset life

        # =============================================================================
        # FORECASTING AND PRICING PARAMETERS
        # =============================================================================

        # Price scaling and limits
        self.price_scale = 10.0  # Price normalization scale
        self.price_clip_min = -1000.0  # Minimum price for clipping
        self.price_clip_max = 1e9  # Maximum price for clipping

        # PRICE NORMALIZATION: Unified rule across all code paths
        # Actual implementation: price_n = clip(z_score, -3, 3) / 3 → [-1, 1]
        # This ensures consistent feature scaling across environment, wrapper, and DL overlay
        # NOTE: price_normalization_divisor and price_z_score_clip below are LEGACY and UNUSED
        # The actual pipeline uses: z_score clipped to ±3σ, then divided by 3 to get [-1,1]
        self.price_normalization_divisor = 10.0  # LEGACY: Kept for compatibility only [UNUSED]
        self.price_z_score_clip = 3.0  # ACTUAL: Clip z-scores to ±3σ before dividing by 3

        # FALLBACK STATS: For consistent two-step normalization when rolling stats unavailable
        self.price_fallback_mean = 250.0  # Typical DKK/MWh price for fallback z-score calculation
        self.price_fallback_std = 50.0    # Typical price volatility for fallback z-score calculation

        # CANONICAL HORIZONS: Single source of truth matching trained models
        # These exact values must match the saved model files
        self.forecast_horizons = {
            "immediate": 1,      # 10 min - direct next step
            "short": 6,          # 1 hour - 6 steps ahead
            "medium": 24,        # 4 hours - 24 steps ahead
            "long": 144,         # 24 hours - 144 steps ahead
            "strategic": 1008    # 7 days - 1008 steps ahead
        }

        # Forecast integration constants (Tier D add-ons)
        self.forecast_price_capacity = 6982.0
        self.forecast_direction_conflict_exposure = 0.12
        self.forecast_low_exposure_threshold = 0.05
        self.forecast_follow_target_exposure = 0.12
        self.forecast_follow_bonus_scale = 0.015
        # FIX: Increased penalty scales to strongly discourage misalignment
        # This helps agent learn to align position direction with forecast direction
        self.forecast_direction_penalty_scale = 1.0  # Increased from 0.35 to strongly penalize direction conflicts
        self.forecast_entry_penalty_scale = 0.12  # Keep moderate for entry gaps
        self.forecast_neutral_penalty_scale = 1.2  # Keep for neutral positions
        self.forecast_usage_bonus_scale = 0.02
        self.forecast_usage_bonus_mtm_scale = 15000.0
        self.forecast_usage_exposure_threshold = 0.02

        # PHASE 1 FIX: Forecast confidence gating thresholds (balanced)
        # Adjusted based on actual forecast quality: trust 0.48-0.51, confidence 0.99, z_combined 0.0-0.5
        # Relaxed from overly strict values to enable forecast usage while maintaining quality
        self.forecast_trust_threshold = 0.45  # Relaxed from 0.6 - allow moderate trust (0.48-0.51)
        self.forecast_confidence_threshold = 0.7  # Keep (already passing with 0.99)
        self.forecast_signal_threshold = 0.2  # Relaxed from 0.3 - allow weaker signals (z_combined often 0.0-0.5)

        # PHASE 1 FIX: Forecast error penalty parameters (addresses reward mismatch)
        self.forecast_error_penalty_scale = 0.5  # Scale for forecast error penalty in alignment reward
        self.forecast_error_penalty_lambda = 0.3  # Lambda for forecast error penalty in signal score
        self.loss_penalty_multiplier = 2.0  # More aggressive penalty for losses (was 1.0)
        self.profitability_bonus_multiplier = 1.5  # Bonus multiplier for profitable positions

        # CANONICAL FORECAST TARGETS: Single source of truth for all forecast models
        self.forecast_targets = ["wind", "solar", "hydro", "price", "load"]

        # CANONICAL REQUIRED HORIZONS: Minimum horizons that must exist
        self.required_forecast_horizons = ["immediate", "short", "medium", "long", "strategic"]

        # CAPACITY-FACTOR TO MW CONVERSION SCALES: Configurable for different datasets
        # These values are used when converting capacity factors [0,1] to raw MW values
        # Default values derived from training scaler analysis - can be overridden
        self.mw_conversion_scales = {
            'wind': 1103,    # From training scaler mean: 1103.4 MW
            'solar': 100,    # From training scaler mean: 61.5 MW (min 100 for stability)
            'hydro': 534,    # From training scaler mean: 534.1 MW
            'load': 2999,    # From training scaler mean: 2999.8 MW
        }

        # Default forecast values (MW or DKK/MWh) - FIXED: Standardized on DKK
        self.default_forecasts = {
            "wind": 330.0,    # ~30% of capacity
            "solar": 20.0,    # ~20% of capacity
            "hydro": 267.0,   # ~50% of capacity
            "price": 345.0,   # DKK/MWh (consistent with environment)
            "load": 1800.0,   # ~60% of capacity
        }

        # =============================================================================
        # DEEP LEARNING PARAMETERS
        # =============================================================================

        # Portfolio optimization neural network
        self.portfolio_learning_rate = 0.002  # USER SPECIFIED: Higher learning rate for early profitability
        self.portfolio_num_assets = 5
        self.portfolio_dropout_rate_1 = 0.2
        self.portfolio_dropout_rate_2 = 0.3

        # Loss function weights
        self.portfolio_risk_loss_weight = 0.5
        self.portfolio_weight_sum_loss_weight = 10.0
        self.portfolio_negative_weight_penalty_weight = 5.0
        self.portfolio_concentration_penalty_weight = 0.1
        self.portfolio_transaction_cost_penalty = 0.001  # 0.1% penalty

        # =============================================================================
        # REWARD AND FORECAST PARAMETERS
        # =============================================================================

        # Forecast confidence thresholds - REMOVED legacy unused parameters
        # Actual confidence control via confidence_floor (0.6) in MAPE-based calculation below

        # Reward calculation parameters - CAPITAL PRESERVATION FOCUS
        self.base_reward_scale = 1.0  # CAPITAL PRESERVATION: Minimal reward scale for stability
        self.profit_reward_weight = 1.0  # CAPITAL PRESERVATION: Equal weight to profit (not dominant)
        self.risk_penalty_weight = 5.0  # CAPITAL PRESERVATION: Very strong risk penalty for safety
        self.forecast_accuracy_reward_weight = 0.30  # CAPITAL PRESERVATION: High weight on forecast accuracy

        # Reward normalization and clipping
        self.reward_clip_min = -10.0  # Minimum reward value
        self.reward_clip_max = 10.0   # Maximum reward value
        self.reward_normalization_factor = 1e6  # Factor to normalize large monetary values

        # Agent-specific reward weights
        self.investor_reward_weight = 1.0      # Investor agent reward scaling
        self.battery_reward_weight = 1.0       # Battery operator reward scaling
        self.risk_controller_reward_weight = 1.0  # Risk controller reward scaling
        self.meta_controller_reward_weight = 1.0   # Meta controller reward scaling

        # NEW: Expert guidance parameters for profit-seeking logic
        self.expert_price_rise_threshold = 1.05  # Suggest 'long' if price forecast is >5% higher
        self.expert_price_fall_threshold = 0.95  # Suggest 'short' if price forecast is <5% lower
        self.expert_suggestion_strength = 1.2   # Strength of the suggestion (e.g., +1.2 or -1.2) - INCREASED FROM 0.8 FOR FULL PUNCH

        # =============================================================================
        # DL OVERLAY PARAMETERS (28D FORECAST-AWARE MODE ONLY)
        # =============================================================================

        # === Overlay controls ===
        self.overlay_enabled = False  # Master switch for DL overlay (only enabled with --dl_overlay flag)
        self.overlay_mode = "auto"  # "auto" | "defense" | "off"
        self.overlay_intensity = 1.0  # [0.5, 1.5] scales amplitude

        # === Information Bridge (28D) ===
        # CONDITIONAL: Disabled when using direct deltas (Tier 2/3) to prevent interference
        # Bridge vectors force PPO to use DL overlay's representation instead of learning its own
        self.overlay_bridge_dim = 4  # Dims appended per agent from shared embedding
        self.overlay_bridge_enable_battery = True  # Append bridge to battery_operator_0
        self.overlay_bridge_enable = True  # Master flag: disable when using direct deltas (auto-set)

        # === Predictive reward shaping (28D auxiliary) ===
        # CONDITIONAL: Disabled when using direct deltas (Tier 2/3) to prevent reward contamination
        # Pred_reward forces PPO to chase DL overlay's predictions instead of learning from actual outcomes
        self.overlay_pred_reward_lambda = 0.4  # Weight for predictive shaping in reward calc - INCREASED FROM 0.1 TO 0.4 FOR STRONGER AGENT LEARNING
        self.overlay_pred_reward_window = 20  # Steps to smooth overlay predicted reward
        self.overlay_pred_reward_enable = True  # Master flag: disable when using direct deltas (auto-set)

        # === Multi-horizon strategy blending (28D) ===
        # Blend weights for 4 horizons: immediate, short, medium, long
        # Tuned for: immediate=99%, short=95%, medium=90%, long=80% (risk-only)
        # Weights reflect relative accuracy: immediate is most accurate, long is least
        self.overlay_blend_weights = {
            'immediate': 0.58,  # Immediate: Full forecast, most actionable (99% accuracy)
            'short': 0.26,      # Short (1h): 95% confidence tactical adjustments
            'medium': 0.12,     # Medium (4h): 90% confidence strategic shifts
            'long': 0.04,       # Long (24h): Risk-only, minimal directional signal (80% accuracy)
        }

        # === Confidence scaling ===
        # FORECAST OPTIMIZATION: Lowered confidence floor from 0.6 to 0.5 for more trading opportunities
        self.force_full_confidence = False  # DISABLED: Use actual forecast confidence from MAPE tracking
        self.confidence_floor = 0.5  # Minimum confidence (50%) - lowered from 60% to allow more forecast-driven trades
        self.overlay_forecast_conf_thresh = 0.5  # Threshold for overlay strategy activation (50%)

        # === Risk budget ===
        self.risk_budget_minmax = (0.5, 1.5)  # Risk budget scaling range
        self.volatility_brake_threshold = 1.8  # Multiply size by 0.8 if vol > 1.8x median

        # === Per-horizon confidence thresholds (for future gating experiments) ===
        self.horizon_confidence_thresholds = {
            'immediate': 0.70,  # Immediate: High confidence required
            'short': 0.65,      # Short: Medium-high confidence
            'medium': 0.60,     # Medium: Medium confidence
            'long': 0.50,       # Long: Low confidence (risk-only anyway)
        }

        # === DEPRECATED: Action blending (replaced by forecast-guided baseline) ===
        # FGB: These parameters are deprecated and will be removed in a future version.
        # Action blending has been replaced by forecast-guided value baseline.
        # The DL overlay now informs the PPO baseline and risk sizing, not action execution.
        # CRITICAL: These are ALWAYS disabled. Do NOT change these values.
        self.enable_action_blending = False  # DEPRECATED: ALWAYS False (use forecast baseline instead)
        self.overlay_alpha = 0.0             # DEPRECATED: ALWAYS 0.0 (no action blending)
        self.force_positive_alpha = False    # DEPRECATED: ALWAYS False (no longer used)
        self.log_overlay_blend = False       # DEPRECATED: ALWAYS False (no longer used)
        self.deprecation_warnings = True     # Emit one-time warnings for deprecated parameters

        # === FGB: Forecast-Guided Baseline (replaces action blending) ===
        # The DL overlay remains a forecaster, not a controller.
        # PPO remains the action decision-maker; we reduce advantage variance via baseline adjustment.
        # FIXED: Default to False (must be explicitly enabled via command-line flag)
        self.forecast_baseline_enable = False  # Enable forecast-guided value baseline (default: OFF)
        self.forecast_baseline_lambda = 0.5    # λ ∈ [0,1]: weight for baseline adjustment (used in "fixed" mode)

        # === FGB: Forecast Trust Calibration ===
        # Rolling window for online calibration of forecast trust (τₜ)
        # CRITICAL FIX: Reduced from 2016 to 500 for faster adaptation (Issue #4)
        self.forecast_trust_window = 500      # ~3.5 days @ 10-min steps (was 2016 = ~2 weeks)
        self.forecast_trust_min = 0.45        # Minimum trust threshold for risk uplift (aligns with medium band)
        self.forecast_trust_metric = "hitrate"  # {"combo", "hitrate", "absdir"}: trust computation method
        self.forecast_trust_direction_weight = 0.8  # Weight for directional accuracy in combo metric (only used if metric="combo")
        # CRITICAL FIX: Increased trust scale min from 0.5 to 0.8 to reduce penalty for low trust (Issue #4)
        self.forecast_trust_scale_min = 0.8    # Minimum trust scale (was 0.5) - less penalty for low trust
        self.forecast_trust_scale_max = 1.5    # Maximum trust scale
        # RATIONALE: Forecast models have ~72% directional accuracy (trust ≈ 0.45 with hitrate metric).
        # Using pure hitrate metric (ignores magnitude error) and lowering threshold to 0.4 allows risk uplift to activate.
        # This is appropriate because forecasts get DIRECTION right 72% of time, which is valuable for trading.

        # === Forecast Reward Warmup ===
        # Gradually turn on forecast-based rewards so early training behaves like Tier 1
        # and forecasts can only start to influence behavior once statistics (EMA, MAPE, trust)
        # have had time to stabilize.
        #
        # Typical episode has ~26,000 timesteps, so 5,000 steps is a gentle warmup.
        self.forecast_reward_warmup_steps = 5000

        # === FGB: Risk Uplift (trust-weighted position sizing) ===
        # Modulate risk budget conservatively when forecasts are credible (no action blending)
        # Controlled by --enable_forecast_risk_management flag (unified with forecast_risk_management_mode)
        self.risk_uplift_enable = False       # Enable trust-weighted risk sizing (controlled by --enable_forecast_risk_management)
        self.risk_uplift_kappa = 0.15         # κ_uplift: 15% sizing uplift (FIXED: match command-line default)
        self.risk_uplift_cap = 1.15           # Maximum risk multiplier (1.0 + kappa)

        # === FAMC: Forecast-Aware Meta-Critic (Learned Control-Variate Baseline) ===
        # Advanced variance reduction using learned state-only critic with online λ* optimization
        self.fgb_mode = "fixed"               # {"fixed", "online", "meta"}: baseline mode
        self.fgb_lambda_max = 0.8             # Maximum λ* for online/meta modes (clip ceiling)
        self.fgb_clip_bps = 0.10              # FIXED: Per-step correction cap (10 bp = 0.10, was 0.01)
        self.fgb_warmup_steps = 2000          # No correction before this many steps (warm-up period)
        self.fgb_moment_beta = 0.01           # EMA rate for Cov/Var moments (0.01 = ~100-step window)

        # === FAMC: Meta-Critic Head Training ===
        # FIXED: Default to False (must be explicitly enabled via command-line flag)
        self.meta_baseline_enable = False     # Enable meta-critic head g_φ(x_t) (default: OFF)
        self.meta_baseline_loss = "mse"       # {"mse", "corr"}: loss function for meta head
        self.meta_baseline_head_dim = 32      # Hidden dimension for meta-critic head
        self.meta_train_every = 512           # Train meta head every N steps
        self.risk_uplift_drawdown_gate = 0.07 # Disable uplift if drawdown > 7%
        self.risk_uplift_vol_gate = 0.0       # Disable vol gate for A/B testing (was 0.02)

        # === Forecast Reward Parameters (Tier 2/3) ===
        # FIX: Moderate increase from 5.0 to 10.0 (2x instead of 4x) for balanced learning
        # Analysis showed agent alignment is random (49%) - need stronger incentive but not too aggressive
        # The profitability gate in forecast_engine.py prevents over-rewarding bad trades
        # Moderate multiplier encourages learning without destabilizing reward signal
        self.forecast_alignment_multiplier = 10.0  # Moderate increase from 5.0 (2x) - balanced approach
        self.forecast_alignment_multiplier_battery = 15.0  # Keep enabled for battery (arbitrage benefits from forecasts)

        # FIX: Make generation forecast weight configurable (Issue #8)
        self.generation_forecast_weight = 0.3  # Weight for generation forecasts in combined score (default: 0.3 = 30%)
        # FIX: EMA std initialization parameters (Issue #1)
        self.ema_std_init_samples = 20  # Number of samples to use for adaptive EMA std initialization
        self.ema_std_init_alpha = 0.2   # Higher alpha for first 100 steps (faster convergence)
        
        # CRITICAL FIX: Reduce warmup to make forecast integration more aggressive
        # Original: 5000 steps = ~20% of episode, too conservative
        # New: 1000 steps = ~4% of episode, forecasts kick in quickly
        self.forecast_reward_warmup_steps = 1000  # Reduced from 5000 to make forecasts impactful sooner
        self.ema_std_min = 10.0         # Minimum EMA std value (DKK) - prevents division by tiny values
        # CRITICAL FIX: Increased from 500.0 to 2000.0 to prevent saturation (Issue #1)
        self.ema_std_max = 2000.0       # Maximum EMA std value (DKK) - prevents extreme values (was 500.0)
        # CRITICAL FIX: Increased EMA alpha from 0.05 to 0.1 for faster adaptation (Issue #9)
        self.ema_alpha = 0.1            # EMA smoothing factor (was 0.05) - faster adaptation to forecast error distribution
        # FIX: Forecast failure handling (Issue #5)
        self.forecast_failure_decay = 0.95  # Decay factor for previous z-scores on forecast failure
        # FIX: Generation EMA std clamping range (Issue #5)
        self.gen_ema_std_min = 0.01  # Minimum EMA std for generation (normalized, not DKK)
        self.gen_ema_std_max = 1.0   # Maximum EMA std for generation (normalized, not DKK)
        self.forecast_pnl_reward_scale = 500.0  # Scale realized MTM contribution in reward shaping

        # === Expert Blending Mode (CRITICAL FIX: Added missing config attributes) ===
        self.expert_blend_mode = "none"       # {"none", "fixed", "adaptive", "residual"}: expert blending mode
        self.expert_blend_weight = 0.0        # Blend weight for fixed mode (0.0-1.0)

        # === Forecast Utilisation (2-TIER SYSTEM) ===
        # Tier 1 (Baseline MARL): enable_forecast_utilisation=False
        # Tier 2 (Forecast-Enhanced Observations): enable_forecast_utilisation=True
        #   - Forecast features added to observations (14D = 6 base + 8 forecast)
        #   - Optional add-ons: forecast_risk_management_mode, risk_uplift_enable
        self.enable_forecast_utilisation = False  # Master flag: enables forecast loading + observation features
        # NOTE: When enabled, uses full forecast features (Tier 22): z_short, z_medium_lagged, direction, momentum, strength, forecast_trust, normalized_error, trade_signal (14D total = 6 base + 8 forecast)

        # === Forecast Risk Management Mode (OPTIONAL ADD-ON - SEPARATE FLAG) ===
        # Controlled by --enable_forecast_risk_management flag (separate from enable_forecast_utilisation)
        # When enabled, also enables risk_uplift_enable (both controlled by same unified flag)
        # Uses forecasts for risk management instead of directional trading
        self.forecast_risk_management_mode = False  # Optional add-on: controlled by --enable_forecast_risk_management flag

        # Risk management parameters
        self.forecast_confidence_high_threshold = 0.48  # Above this: normal positions (~top 20% trust)
        self.forecast_confidence_medium_threshold = 0.45  # Above this: 70% positions (~top 40% trust)
        self.forecast_confidence_low_scale = 0.5  # Below medium: 50% positions

        self.forecast_volatility_high_threshold = 1.5  # Above this: 50% positions
        self.forecast_volatility_medium_threshold = 1.0  # Above this: 75% positions

        self.forecast_direction_change_exit_fraction = 0.5  # Exit 50% on direction flip
        self.forecast_direction_change_min_strength = 0.5  # Min z-score to trigger

        self.forecast_extreme_threshold = 2.0  # z-score threshold for extremes
        self.forecast_extreme_profit_taking_fraction = 0.3  # Take 30% profits

        self.forecast_risk_management_weight = 0.20  # 20% of total reward for risk management

        # === NOVEL: Multi-Horizon Signal Filtering ===
        # Only trade when forecast delta > MAPE threshold (statistically significant)
        # This filters out noise and focuses on high-confidence opportunities
        self.enable_signal_filtering = True  # Enable MAPE-based signal filtering
        self.signal_filter_mape_multiplier = 0.8  # Trade when |delta| > multiplier * MAPE
        self.signal_filter_horizon_weights = {  # Weights for multi-horizon aggregation
            'short': 0.5,   # Short-term (6 steps) gets 50% weight
            'medium': 0.3,  # Medium-term (24 steps) gets 30% weight
            'long': 0.2     # Long-term (144 steps) gets 20% weight
        }
        self.signal_filter_min_horizons = 1  # Minimum number of horizons with significant signals
        self.signal_filter_neutral_position_scale = 0.6  # Scale positions to 60% when no signal (increased from 0.1)
        # Gradually relax the gate so filters remain usable late in an episode
        self.signal_gate_initial_multiplier = 3.0  # Initial multiple of MAPE required to trade
        self.signal_gate_min_multiplier = 0.8      # Floor multiple once decay completes
        self.signal_gate_decay_start = 720         # Timesteps before decay begins
        self.signal_gate_decay_duration = 2880     # Steps to move from initial -> min multiplier
        self.enable_signal_gate_decay = True

        # === NOVEL: Investor-Specific Hedging vs Trading Thresholds ===
        # Different MAPE multipliers for different strategies
        # HEDGING: Protect existing positions when small forecast deviations detected
        # TRADING: Take new positions only when large forecast deviations detected
        self.investor_hedge_mape_multiplier = 0.4   # Hedge when |delta| > 0.4x MAPE (more responsive)
        self.investor_trade_mape_multiplier = 1.1   # Trade when |delta| > 1.1x MAPE (moderate threshold)
        self.investor_aggressive_trade_mape_multiplier = 1.6  # Aggressive trade when |delta| > 1.6x MAPE

        # Position sizing for different signal strengths
        self.investor_hedge_position_scale = 0.7    # 70% positions for hedging (increased from 0.5)
        self.investor_trade_position_scale = 0.9    # 90% positions for normal trading (increased from 0.7)
        self.investor_aggressive_position_scale = 1.2  # 120% positions for strong signals (increased from 1.0)

        # Multi-horizon consensus requirements
        self.investor_require_consensus = False     # Allow majority vote without hard consensus veto
        self.investor_consensus_min_horizons = 1    # At least 1 horizon must agree (already lenient)
        self.investor_consensus_direction_threshold = 0.35  # 35% weighted alignment now enough to trade

        # Reward tuning for forecast utilisation bonuses (investor agent only)
        self.forecast_usage_bonus_scale = 0.02  # Max extra reward added when MTM > 0 and forecast used
        self.forecast_usage_bonus_mtm_scale = 15000.0  # Normalization for MTM (DKK) when computing bonus

        # === Overlay Risk Budget Application ===
        # Control whether overlay's risk multiplier is always applied
        self.overlay_apply_risk_budget = True # Always apply overlay risk budget (unless explicitly disabled)

        # =============================================================================
        # REINFORCEMENT LEARNING PARAMETERS
        # =============================================================================

        # Training defaults - STRENGTHENED FOR BETTER LEARNING
        self.update_every = 128  # OPTIMIZED: Reduced from 32 for more responsive learning
        self.lr = 1.5e-3  # STRENGTHENED: Increased from 8e-4 to 1.5e-3 for faster learning (87% increase)
        self.ent_coef = 0.010  # FAIR COMPARISON: Same for both tiers (increased to encourage position-taking)
        self.verbose = 1
        self.seed = 42
        self.multithreading = True

        # PPO-specific parameters - STRENGTHENED FOR BETTER LEARNING
        self.batch_size = 256  # STRENGTHENED: Increased from 128 for more stable gradients
        self.gamma = 0.995  # STRENGTHENED: Increased from 0.99 for longer horizon credit assignment
        self.gae_lambda = 0.98  # STRENGTHENED: Increased from 0.95 for more stable value estimates
        self.clip_range = 0.15  # STRENGTHENED: Reduced from 0.2 for more conservative updates
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.n_epochs = 10  # STRENGTHENED: Increased from default 5 for more gradient updates
        self.n_steps = 1024  # STRENGTHENED: Increased from 512 for more experience per update

        # Network architecture
        self.net_arch = [256, 128, 64]  # Deeper network for better pattern recognition
        self.activation_fn = "relu"  # Changed from tanh for better gradient flow

        # GNN Encoder (works for both Tier 1 and Tier 2)
        self.enable_gnn_encoder = False  # Enable GNN observation encoder for relationship learning (works on 6D base or 14D forecast-enhanced observations)
        
        # Tier 1 defaults (6D base observations)
        self.gnn_features_dim = 18  # Output dimension of GNN encoder (divisible by num_heads=3)
        self.gnn_hidden_dim = 30  # Hidden dimension of GAT layers (divisible by num_heads=3: 30/3=10)
        self.gnn_num_layers = 2  # Number of GAT layers
        self.gnn_num_heads = 3  # IMPROVED: Multi-head attention (3 heads for different relationship types)
        self.gnn_dropout = 0.1  # Dropout rate for GAT
        self.gnn_graph_type = 'full'  # 'full' (fully-connected) or 'learned' (learnable adjacency)
        self.gnn_use_attention_pooling = True  # IMPROVED: Use attention pooling instead of mean pooling
        self.gnn_net_arch = [128, 64]  # Smaller MLP after GNN (GNN does feature extraction)
        
        # Tier 2 settings (14D: 6 base + 8 forecast features)
        # PUBLICATION-FAIR: Match Tier 1 output capacity while keeping hierarchical structure
        # The hierarchical architecture is the KEY DIFFERENCE needed to utilize forecast features
        # Matched capacities (output_dim, MLP) isolate the architecture effect
        self.gnn_features_dim_tier2 = 18  # MATCHED: Same final output as Tier 1 (each sub-encoder → 9D, fused → 18D)
        self.gnn_hidden_dim_tier2 = 30  # MATCHED: Same as Tier 1 (each sub-encoder uses 15)
        self.gnn_num_layers_tier2 = 2  # MATCHED: Same depth (sub-encoders use full 2 layers each, fusion adds structure)
        self.gnn_num_heads_tier2 = 3  # MATCHED: Same as Tier 1
        self.gnn_dropout_tier2 = 0.1  # MATCHED: Same as Tier 1
        self.gnn_graph_type_tier2 = 'hierarchical'  # KEY DIFFERENCE: Hierarchical structure (only architectural difference)
        self.gnn_use_attention_pooling_tier2 = True  # MATCHED: Same as Tier 1
        self.gnn_net_arch_tier2 = [128, 64]  # MATCHED: Same MLP as Tier 1
        
        # NOTE FOR PUBLICATION:
        # - Output dimensions matched: Both produce 18D final representation
        # - MLP matched: Both use [128, 64]
        # - Depth matched: Both use 2 layers (hierarchical splits into 2 parallel paths)
        # - Only difference: Hierarchical structure (separate base/forecast encoders + cross-attention)
        # This isolates whether the hierarchical architecture (necessary for forecast features) provides benefit

        # Agent policies
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "PPO"},  # battery_operator_0
            {"mode": "PPO"},  # risk_controller_0
            {"mode": "SAC"},  # meta_controller_0
        ]

        if optimized_params:
            self._apply_optimized_params(optimized_params)

    # =============================================================================
    # HELPER METHODS FOR DERIVED VALUES
    # =============================================================================

    def get_initial_asset_plan(self) -> Dict[str, Dict[str, float]]:
        """Get the initial asset deployment plan based on configured ownership fractions."""
        return {
            'wind': {'capacity_mw': self.owned_wind_capacity_mw},
            'solar': {'capacity_mw': self.owned_solar_capacity_mw},
            'hydro': {'capacity_mw': self.owned_hydro_capacity_mw},
            'battery': {'capacity_mwh': self.owned_battery_capacity_mwh}
        }

    def get_asset_capex(self, currency: str = 'DKK') -> Dict[str, float]:
        """Get the CAPEX values for different asset types.

        Args:
            currency: 'USD' for original market rates, 'DKK' for internal calculations
        """
        capex_usd = {
            'wind_mw': self.wind_capex_per_mw,
            'solar_mw': self.solar_capex_per_mw,
            'hydro_mw': self.hydro_capex_per_mw,
            'battery_mwh': self.battery_capex_per_mwh
        }

        if currency == 'USD':
            return capex_usd
        elif currency == 'DKK':
            # Convert USD to DKK for internal calculations
            return {key: value / self.dkk_to_usd_rate for key, value in capex_usd.items()}
        else:
            raise ValueError(f"Unsupported currency: {currency}")

    def get_expected_physical_values(self) -> Dict[str, float]:
        """Get expected values for validation checks (in DKK for environment validation)."""
        # Calculate CAPEX in DKK for environment validation
        actual_physical_capex_dkk = self.calculate_total_physical_capex(currency='DKK')
        remaining_cash_dkk = self.init_budget - actual_physical_capex_dkk

        return {
            'wind': self.owned_wind_capacity_mw,
            'solar': self.owned_solar_capacity_mw,
            'hydro': self.owned_hydro_capacity_mw,
            'battery': self.owned_battery_capacity_mwh,
            'physical_book_value': actual_physical_capex_dkk,  # CAPEX in DKK
            'cash_min': remaining_cash_dkk * 0.8  # At least 80% of remaining cash for trading
        }

    def get_trading_capital_limits(self) -> Dict[str, float]:
        """Get trading capital allocation limits for validation."""
        # Calculate allocated trading capital in DKK
        trading_capital_dkk = self.init_budget * self.financial_allocation
        trading_capital_usd = trading_capital_dkk * self.dkk_to_usd_rate

        # Maximum financial exposure with leverage
        max_financial_exposure_dkk = trading_capital_dkk * self.max_leverage
        max_financial_exposure_usd = max_financial_exposure_dkk * self.dkk_to_usd_rate

        return {
            'trading_capital_dkk': trading_capital_dkk,
            'trading_capital_usd': trading_capital_usd,
            'max_financial_exposure_dkk': max_financial_exposure_dkk,
            'max_financial_exposure_usd': max_financial_exposure_usd,
            'max_leverage': self.max_leverage
        }

    def calculate_total_physical_capex(self, currency: str = 'USD') -> float:
        """Calculate total expected CAPEX for physical assets.

        Args:
            currency: 'USD' for original market rates, 'DKK' for internal calculations
        """
        capex_values = self.get_asset_capex(currency=currency)
        return (
            self.owned_wind_capacity_mw * capex_values['wind_mw'] +
            self.owned_solar_capacity_mw * capex_values['solar_mw'] +
            self.owned_hydro_capacity_mw * capex_values['hydro_mw'] +
            self.owned_battery_capacity_mwh * capex_values['battery_mwh']
        )

    def validate_configuration(self) -> bool:
        """
        CONFIGURATION VALIDATION (NEW)

        Validates that all configuration parameters are consistent and within valid ranges.
        Raises ValueError if validation fails.

        Returns:
            True if all validations pass

        Raises:
            ValueError: If any validation fails
        """
        errors = []

        # 1. Allocation fractions should sum to 1.0
        total_allocation = self.physical_allocation + self.financial_allocation
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Allocation fractions sum to {total_allocation}, expected 1.0")

        # 2. Ownership fractions should be in [0, 1]
        for name, value in [
            ("wind_ownership_fraction", self.wind_ownership_fraction),
            ("solar_ownership_fraction", self.solar_ownership_fraction),
            ("hydro_ownership_fraction", self.hydro_ownership_fraction),
        ]:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name}={value} not in [0, 1]")

        # 3. Meta controller ranges should be valid
        if self.meta_freq_min >= self.meta_freq_max:
            errors.append(f"meta_freq_min ({self.meta_freq_min}) >= meta_freq_max ({self.meta_freq_max})")
        if self.meta_cap_min >= self.meta_cap_max:
            errors.append(f"meta_cap_min ({self.meta_cap_min}) >= meta_cap_max ({self.meta_cap_max})")

        # 4. Investment frequency ranges should be valid
        if self.min_investment_freq >= self.max_investment_freq:
            errors.append(f"min_investment_freq ({self.min_investment_freq}) >= max_investment_freq ({self.max_investment_freq})")
        if self.investment_freq < self.min_investment_freq or self.investment_freq > self.max_investment_freq:
            errors.append(f"investment_freq ({self.investment_freq}) not in [{self.min_investment_freq}, {self.max_investment_freq}]")

        # 5. Battery SOC bounds should be valid
        if self.batt_soc_min >= self.batt_soc_max:
            errors.append(f"batt_soc_min ({self.batt_soc_min}) >= batt_soc_max ({self.batt_soc_max})")
        if not (0.0 <= self.batt_soc_min <= 1.0) or not (0.0 <= self.batt_soc_max <= 1.0):
            errors.append(f"Battery SOC bounds not in [0, 1]")

        # 6. Battery efficiency should be in (0, 1]
        if not (0.0 < self.batt_efficiency <= 1.0):
            errors.append(f"batt_efficiency ({self.batt_efficiency}) not in (0, 1]")

        # 7. Price bounds should be valid
        if self.minimum_price_floor >= self.maximum_price_cap:
            errors.append(f"minimum_price_floor ({self.minimum_price_floor}) >= maximum_price_cap ({self.maximum_price_cap})")
        if self.minimum_price_filter >= self.minimum_price_floor:
            errors.append(f"minimum_price_filter ({self.minimum_price_filter}) >= minimum_price_floor ({self.minimum_price_floor})")

        # 8. Leverage should be >= 1.0
        if self.max_leverage < 1.0:
            errors.append(f"max_leverage ({self.max_leverage}) < 1.0")

        # 9. Rate parameters should be in [0, 1]
        rate_params = [
            ("operating_cost_rate", self.operating_cost_rate),
            ("insurance_rate", self.insurance_rate),
            ("management_fee_rate", self.management_fee_rate),
            ("property_tax_rate", self.property_tax_rate),
            ("debt_service_rate", self.debt_service_rate),
            ("distribution_rate", self.distribution_rate),
            ("target_cash_ratio", self.target_cash_ratio),
            ("battery_opex_rate", self.battery_opex_rate),
            ("performance_fee_rate", self.performance_fee_rate),
            ("trading_cost_rate", self.trading_cost_rate),
        ]
        for name, value in rate_params:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name}={value} not in [0, 1]")

        # 10. Forecast horizons should be positive integers
        for horizon_name, horizon_steps in self.forecast_horizons.items():
            if not isinstance(horizon_steps, int) or horizon_steps <= 0:
                errors.append(f"forecast_horizons[{horizon_name}]={horizon_steps} not a positive integer")

        # 11. Overlay parameters should be valid
        if self.overlay_bridge_dim <= 0:
            errors.append(f"overlay_bridge_dim ({self.overlay_bridge_dim}) <= 0")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("[OK] Configuration validation passed")
        return True

    def _apply_optimized_params(self, params: Dict[str, Any]):
        print("Applying optimized hyperparameters...")

        # Accept either 'update_every' or 'n_steps'
        self.update_every = int(params.get('update_every', params.get('n_steps', self.update_every)))

        # Learning parameters
        self.lr = float(params.get('lr', self.lr))
        self.ent_coef = params.get('ent_coef', self.ent_coef)
        self.batch_size = int(params.get('batch_size', self.batch_size))
        self.gamma = float(params.get('gamma', self.gamma))
        self.gae_lambda = float(params.get('gae_lambda', self.gae_lambda))
        self.clip_range = float(params.get('clip_range', self.clip_range))
        self.vf_coef = float(params.get('vf_coef', self.vf_coef))
        self.max_grad_norm = float(params.get('max_grad_norm', self.max_grad_norm))

        # Net arch: take explicit list if provided; otherwise map from size label
        if isinstance(params.get('net_arch'), (list, tuple)):
            self.net_arch = list(params['net_arch'])
        else:
            net_arch_mapping = {
                'small': [64, 32],
                'medium': [128, 64],
                'large': [256, 128, 64]
            }
            self.net_arch = net_arch_mapping.get(params.get('net_arch_size', 'medium'), [128, 64])

        # Activation: accept 'activation' or 'activation_fn'
        self.activation_fn = params.get('activation', params.get('activation_fn', self.activation_fn))

        # Agent modes: if a full list is provided, use it; otherwise accept *_mode aliases
        if isinstance(params.get('agent_policies'), list) and params['agent_policies']:
            self.agent_policies = params['agent_policies']
        else:
            self.agent_policies = [
                {"mode": params.get('investor_mode', 'PPO')},
                {"mode": params.get('battery_mode', 'PPO')},
                {"mode": params.get('risk_mode', 'PPO')},
                {"mode": params.get('meta_mode', 'SAC')},
            ]

        print(f"   Learning rate: {self.lr:.2e}")
        print(f"   Entropy coefficient: {self.ent_coef}")
        print(f"   Network architecture: {self.net_arch}")
        print(f"   Agent modes: {[p['mode'] for p in self.agent_policies]}")
        print(f"   Update every: {self.update_every}")
        print(f"   Batch size: {self.batch_size}")


# =====================================================================
# 2-TIER SYSTEM: SIMPLIFIED
# =====================================================================
# Tier 1: EnhancedConfig() with enable_forecast_utilisation=False
# Tier 2: EnhancedConfig() with enable_forecast_utilisation=True (auto-enables risk management)
#
# The ImprovedTier2Config class below is DEPRECATED and kept only for backward compatibility.
# Just use: config = EnhancedConfig(); config.enable_forecast_utilisation = True

class ImprovedTier2Config(EnhancedConfig):
    """
    Improved Tier 2 configuration that uses forecasts for risk management.

    Key changes from original Tier 2:
    1. Disable forecast alignment rewards (was causing -$643K losses)
    2. Enable forecast risk management mode
    3. Use forecasts to adjust position sizes based on confidence
    4. Use forecasts to detect volatility and reduce exposure
    5. NOVEL: Multi-horizon signal filtering (only trade when delta > MAPE)
    6. Reward risk-adjusted performance, not forecast alignment

    Expected outcome: Tier 2 should outperform Tier 1 by $60K-$110K per episode
    """

    def __init__(self):
        super().__init__()

        # ============================================================
        # PHASE 1: DISABLE BROKEN FORECAST REWARDS
        # ============================================================

        # CRITICAL FIX: Disable forecast alignment rewards for INVESTORS
        # Original: 20.0 (caused agent to get +20.91 reward for losing $13,625)
        # New: 0.0 (no rewards for forecast alignment)
        self.forecast_alignment_multiplier = 0.0

        # Battery keeps using forecasts (arbitrage works differently)
        self.forecast_alignment_multiplier_battery = 15.0

        # Enable risk management mode
        self.forecast_risk_management_mode = True

        # Keep forecasts enabled for observations
        self.enable_forecast_utilisation = True

        # ============================================================
        # PHASE 2: RISK MANAGEMENT PARAMETERS
        # ============================================================

        # Position sizing based on forecast confidence (MADE MORE AGGRESSIVE)
        self.forecast_confidence_high_threshold = 0.40  # Above this: normal positions (lowered from 0.48)
        self.forecast_confidence_medium_threshold = 0.35  # Above this: 70% positions (lowered from 0.45)
        self.forecast_confidence_low_scale = 0.7  # Below medium: 70% positions (increased from 0.5)

        # Volatility-based position scaling
        self.forecast_volatility_high_threshold = 1.5  # Above this: 50% positions
        self.forecast_volatility_medium_threshold = 1.0  # Above this: 75% positions

        # Exit signals on forecast direction changes
        self.forecast_direction_change_exit_fraction = 0.5  # Exit 50% on direction flip
        self.forecast_direction_change_min_strength = 0.5  # Min z-score to trigger

        # Profit-taking on extreme forecasts
        self.forecast_extreme_threshold = 2.0  # z-score threshold for extremes
        self.forecast_extreme_profit_taking_fraction = 0.3  # Take 30% profits

        # Risk management reward weight
        self.forecast_risk_management_weight = 0.20
        
        # NOTE: ent_coef is inherited from EnhancedConfig (0.010) - same for both tiers for fair comparison  # 20% of total reward

        # ============================================================
        # NOVEL: MULTI-HORIZON SIGNAL FILTERING
        # ============================================================

        # Only trade when forecast delta > MAPE threshold (statistically significant)
        self.enable_signal_filtering = True
        self.signal_filter_mape_multiplier = 1.0  # Trade when |delta| > 1.0 * MAPE
        self.signal_filter_horizon_weights = {
            'short': 0.5,   # Short-term (6 steps) gets 50% weight
            'medium': 0.3,  # Medium-term (24 steps) gets 30% weight
            'long': 0.2     # Long-term (144 steps) gets 20% weight
        }
        self.signal_filter_min_horizons = 1  # Minimum number of horizons with significant signals
        self.signal_filter_neutral_position_scale = 0.6  # Scale positions to 60% when no signal (increased from 0.1)

        # ============================================================
        # REWARD WEIGHTS (ADJUSTED FOR RISK MANAGEMENT)
        # ============================================================

        # Rebalance weights to emphasize risk management
        # Total should sum to 1.0
        self.profit_reward_weight = 0.35  # Operational revenue (same as Tier 1)
        self.risk_reward_weight = 0.30    # Risk management (increased from 0.25)
        self.hedging_reward_weight = 0.15  # Hedging effectiveness
        self.nav_stability_weight = 0.20   # NAV stability (reduced from 0.25)

        # NO forecast directional reward
        self.forecast_directional_weight = 0.0  # Disabled

        # ============================================================
        # POSITION SIZING (MORE CONSERVATIVE)
        # ============================================================

        # Reduce max position size to be more conservative
        # Tier 1 takes up to 66% positions, Tier 2 will be more adaptive
        self.max_position_size = 0.008  # 0.8% (reduced from 1.0%)

        # This will be scaled by forecast confidence and volatility
        # Effective range: 0.4% - 0.8% depending on conditions

        # ============================================================
        # FORECAST TRUST CALIBRATION (FASTER ADAPTATION)
        # ============================================================

        # Faster trust adaptation for quicker response to forecast quality
        self.forecast_trust_window = 300  # 300 steps (~2 days)
        self.forecast_trust_min = 0.45  # Minimum trust threshold (aligns with medium band)

        # ============================================================
        # LOGGING AND DEBUGGING
        # ============================================================

        # Enable detailed logging for risk management decisions
        self.log_forecast_risk_management = True
        self.log_position_scaling_decisions = True

        print("\n" + "="*70)
        print("IMPROVED TIER 2 CONFIGURATION LOADED")
        print("="*70)
        print("Forecast Mode: RISK MANAGEMENT (not directional trading)")
        print(f"Alignment Multiplier: {self.forecast_alignment_multiplier} (disabled)")
        print(f"Risk Management Weight: {self.forecast_risk_management_weight}")
        print(f"Max Position Size: {self.max_position_size*100:.1f}%")
        print(f"Confidence Thresholds: High={self.forecast_confidence_high_threshold}, Med={self.forecast_confidence_medium_threshold}")
        print(f"Volatility Thresholds: High={self.forecast_volatility_high_threshold}, Med={self.forecast_volatility_medium_threshold}")
        print(f"Signal Filtering: Enabled (trade when |delta| > {self.signal_filter_mape_multiplier}x MAPE)")
        print("="*70 + "\n")


def get_improved_tier2_config():
    """
    Get the improved Tier 2 configuration.

    Returns:
        ImprovedTier2Config instance
    """
    return ImprovedTier2Config()