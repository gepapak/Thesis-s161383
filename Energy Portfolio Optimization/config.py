from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED CONSTANTS (SINGLE SOURCE OF TRUTH)
# ============================================================================
# These constants ensure consistent configuration across all modes and code paths

# Overlay feature dimension (CANONICAL) - short-horizon only (aligned with investment_freq=6)
OVERLAY_FEATURE_DIM = 18    # Market(6) + Positions(3) + Forecasts_short(4) + Portfolio(3) + Deltas(2)

# Price normalization constants
PRICE_MEAN = 250.0          # Historical mean price (DKK/MWh)
PRICE_STD = 50.0            # Historical std dev (DKK/MWh)
PRICE_CLIP_SIGMA = 3.0      # Clip at ±3 sigma for outlier handling

# Risk controller constants
RISK_LOOKBACK_WINDOW_DEFAULT = 144  # Default lookback window for risk calculations
RISK_LOOKBACK_WINDOW_MAX = 200      # Maximum lookback window (memory cap)

# Adaptive scaling constants
ADAPTIVE_SCALE_SATURATION_THRESHOLD = 0.95
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

# DL Overlay loss weights
DL_OVERLAY_LOSS_WEIGHTS = {
    'pred_reward': 1.0,
}

# Wrapper cache defaults (fallback when no config)
WRAPPER_FORECAST_CACHE_SIZE_DEFAULT = 1000
WRAPPER_AGENT_CACHE_SIZE_DEFAULT = 2000
WRAPPER_MEMORY_LIMIT_MB_DEFAULT = 1024.0

# Environment constants
ENV_MARKET_STRESS_DEFAULT = 0.5
ENV_OVERALL_RISK_DEFAULT = 0.5
ENV_MARKET_RISK_DEFAULT = 0.5
ENV_POSITION_EXPOSURE_THRESHOLD = 0.001
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
        # The environment intentionally uses unconstrained NAV to allow realistic fund dynamics
        # Only prevents negative NAV; no artificial min/max bounds applied
        self.nav_min_ratio = 0.90  # INFRASTRUCTURE: Minimum 90% of initial (conservative downside protection) [UNUSED]
        self.nav_max_ratio = 1.10  # INFRASTRUCTURE: Maximum 110% of initial (5% return cap - very realistic for infrastructure) [UNUSED]



        # =============================================================================
        # ENVIRONMENT AND SIMULATION PARAMETERS
        # =============================================================================

        # Meta controller ranges
        #
        # Goal: high notional exposure WITHOUT forcing the investor policy to saturate at +/-1.
        # We achieve this by keeping capital allocation reliably high, and fixing the trade cadence
        # to the same horizon as the investor's forecast features (MEDIUM = 24 steps).
        self.meta_freq_min = 24   # 4h @ 10-min steps (aligned with medium horizon)
        self.meta_freq_max = 24   # fixed cadence (removes freq-jitter that hurts learning)
        self.meta_cap_min = 0.25  # keep at least 25% of cash allocated to trading
        self.meta_cap_max = 0.95  # allow up to 95% of cash allocated to trading
        self.sat_eps = 1e-3

        # Investment and operational parameters - FORECAST OPTIMIZATION
        self.investment_freq = 6   # CAPITAL PRESERVATION: Every 4 hours for careful positioning
        self.min_investment_freq = 4   # CAPITAL PRESERVATION: 2 hour minimum
        self.max_investment_freq = 12  # CAPITAL PRESERVATION: 48 hours maximum (very patient)
        self.capital_allocation_fraction = 0.90  # starting point; meta-controller clamps within [meta_cap_min, meta_cap_max]
        self.risk_multiplier = 1.0  # Raised from 0.5 for full-size baseline trades
        self.no_trade_threshold = 0.01  # 1% of target position - trades below this are ignored

        # Battery operational parameters
        self.batt_soc_min = 0.1  # 10% minimum state of charge
        self.batt_soc_max = 0.9  # 90% maximum state of charge
        self.batt_efficiency = 0.85  # 85% round-trip efficiency

        # =============================================================================
        # MEMORY AND PERFORMANCE PARAMETERS
        # =============================================================================

        # Paper-grade correctness: never silently pad/truncate observation vectors.
        # If an observation dimension mismatch occurs, abort with a clear error so the run is reproducible.
        self.strict_obs_validation = True

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
        self.max_position_size = 0.35  # larger per-asset sizing (primary lever for notional exposure)
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
        self.forecast_direction_penalty_scale = 2.0  # Increased to penalize direction conflicts more strongly
        self.forecast_entry_penalty_scale = 0.12  # Keep moderate for entry gaps
        self.forecast_neutral_penalty_scale = 1.2  # Keep for neutral positions
        # Disable forecast-only bonuses to keep reward parity across tiers
        self.forecast_usage_bonus_scale = 0.0
        self.forecast_usage_bonus_mtm_scale = 15000.0
        # Lower threshold so forecasts count as “used” even for small exploratory positions (Tier 2 obs-only fairness)
        self.forecast_usage_exposure_threshold = 0.001
        # Raise MTM loss exit threshold to allow positions to develop before forced exits (from 3% -> 6%)
        self.mtm_loss_exit_threshold_pct = 0.06

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
        # DL OVERLAY PARAMETERS (18D MEDIUM-HORIZON ONLY)
        # =============================================================================

        # === Overlay controls ===
        self.overlay_enabled = False  # Master switch for DL overlay (only enabled with --dl_overlay flag)
        self.overlay_mode = "auto"  # "auto" | "defense" | "off"
        self.overlay_intensity = 1.0  # [0.5, 1.5] scales amplitude


        # === FGB/FAMC only ===
        # The overlay produces a pred_reward proxy used for FGB/FAMC baseline signals.
        # It is never injected into the environment reward.

        # === DL Overlay: short-horizon only (aligned with investment_freq=6) ===
        # Single-horizon overlay uses short (1h) forecast exclusively
        # Short horizon aligns with 6-step trading cadence; typically more accurate than medium
        self.overlay_horizon = "short"

        # === Confidence scaling ===
        # FORECAST OPTIMIZATION: Lowered confidence floor from 0.6 to 0.5 for more trading opportunities
        self.force_full_confidence = False  # DISABLED: Use actual forecast confidence from MAPE tracking
        self.confidence_floor = 0.5  # Minimum confidence (50%) - lowered from 60% to allow more forecast-driven trades
        self.overlay_forecast_conf_thresh = 0.5  # Threshold for overlay strategy activation (50%)

        # === Volatility brake (general risk control; independent of overlay) ===
        self.volatility_brake_threshold = 2.5  # Multiply size by 0.8 if vol > 2.5x median (less frequent brake)

        # === Per-horizon confidence thresholds (for future gating experiments) ===
        self.horizon_confidence_thresholds = {
            'immediate': 0.70,  # Immediate: High confidence required
            'short': 0.65,      # Short: Medium-high confidence
            'medium': 0.60,     # Medium: Medium confidence
            'long': 0.50,       # Long: Low confidence (risk-only anyway)
        }

        # === FGB: Forecast-Guided Baseline (replaces action blending) ===
        # The DL overlay remains a forecaster, not a controller.
        # PPO remains the action decision-maker; we reduce advantage variance via baseline adjustment.
        # FIXED: Default to False (must be explicitly enabled via command-line flag)
        self.forecast_baseline_enable = False  # Enable forecast-guided value baseline (default: OFF)
        # Ablation: keep the full FGB/FAMC plumbing on, but force the control variate to 0
        # (tau=0, expected_dnav=0). This isolates the *forecast signal* contribution.
        self.fgb_ablate_forecasts = False  # Default: OFF

        # === FGB: Forecast Trust Calibration ===
        # Rolling window for online calibration of forecast trust (τₜ)
        # CRITICAL FIX: Reduced from 2016 to 500 for faster adaptation (Issue #4)
        self.forecast_trust_window = 500      # ~3.5 days @ 10-min steps (was 2016 = ~2 weeks)
        self.forecast_trust_metric = "combo"  # {"combo", "hitrate", "absdir"}: MAPE + directional for robustness
        self.forecast_trust_direction_weight = 0.8  # Weight for directional accuracy in combo metric (only used if metric="combo")
        # CRITICAL FIX: Increased trust scale min from 0.5 to 0.8 to reduce penalty for low trust (Issue #4)
        self.forecast_trust_scale_min = 0.8    # Minimum trust scale (was 0.5) - less penalty for low trust
        self.forecast_trust_scale_max = 1.5    # Maximum trust scale
        # Optional multiplicative trust optimism boost. Keep 0.0 by default for unbiased calibration.
        self.forecast_trust_boost = 0.0
        # RATIONALE: directional accuracy is often more actionable than magnitude error in trading-style decisions,
        # so "hitrate" is a sensible default for trust calibration.

        # === FGB/FAMC: Per-agent trust horizon selection (paper-clean) ===
        # Tier1 observations remain unchanged; this only affects the *baseline/control-variate* scaling (tau_t)
        # used inside PPO advantage correction when FGB/FAMC is enabled.
        #
        # - "auto": pick the horizon with the best recent directional accuracy (requires enough samples).
        # - explicit: {"short", "medium", "long"}
        #
        # Default choice: investor uses "short" (aligned with investment_freq=6; 1h cadence).
        self.fgb_trust_horizon_by_agent = {
            "investor_0": "short",
            "battery_operator_0": "auto",
            "risk_controller_0": "auto",
        }
        self.fgb_trust_min_samples = 50  # Minimum samples per horizon before "auto" trusts the estimate.
        self.forecast_trust_floor = 0.3  # Minimum τ when forecasts pass quality checks (avoids underuse of control variate)

        # ---------------------------------------------------------------------
        # DL overlay supervision (used only when --dl_overlay is enabled)
        # ---------------------------------------------------------------------
        # The overlay pred_reward head is supervised with a realized return proxy:
        #   realized_return ~= delta_nav / exposure
        # and then squashed with tanh for a stable [-1, 1] regression target.
        # These settings do NOT affect Tier-1 (overlay is disabled there).
        self.overlay_mtm_return_clip = 0.20
        self.overlay_mtm_return_tanh_scale = 20.0
        # Preferred names for the same controls (kept alongside legacy keys for compatibility).
        self.overlay_dnav_return_clip = 0.20
        self.overlay_dnav_return_tanh_scale = 20.0

        # === Forecast Reward Warmup ===
        # Gradually turn on forecast-based rewards so early training behaves like Tier 1
        # and forecasts can only start to influence behavior once statistics (EMA, MAPE, trust)
        # have had time to stabilize.
        #
        # Typical episode has ~26,000 timesteps, so 5,000 steps is a gentle warmup.
        self.forecast_reward_warmup_steps = 5000

        # --------------------------------------------------------------
        # Forecast quality thresholds (price-relative MAPE)
        # --------------------------------------------------------------
        # These thresholds must be in the SAME units as the MAPE we compute online.
        # We use price-relative MAPE with a denominator floor (see environment._update_calibration_tracker),
        # which makes the signal robust even when spot prices approach 0.
        #
        # These values define fixed "acceptable error" bands used by online calibration
        # (see environment._update_calibration_tracker) so trust/error semantics are stable.
        self.forecast_mape_threshold_short = 0.10   # 10% (loosened from 8% for more forecast usage)
        self.forecast_mape_threshold_medium = 0.15  # 15% (loosened from 12%)
        self.forecast_mape_threshold_long = 0.25    # 25% (loosened from 20%)

        # === Anti-saturation: Forecast return -> z-score mapping ===
        # Problem: z_short/z_medium were often pinned at +/-1 due to extreme return spikes
        # (e.g., low price denominators / floor behavior) + aggressive tanh scaling.
        #
        # Fix: (a) clip forecast returns to a reasonable bound, (b) use configurable tanh scale,
        # (c) use a denominator floor so returns can't explode when prices are low.
        # This makes forecast-derived obs informative (not binary), improving policy sensitivity.
        # Increased signal strength (Tier2 obs-only usage): higher scale + slightly looser clip + lower denom floor.
        # This makes forecast z-scores less "washed out" while staying bounded.
        self.forecast_return_clip = 0.15              # tighter clip to curb saturation at decision cadence
        self.forecast_return_tanh_scale = 1.5         # lower tanh scale for smoother signal
        self.forecast_return_denom_floor = 0.25       # lower floor to keep calibrated trade_signal visible

        # === FAMC: Forecast-Aware Meta-Critic (Learned Control-Variate Baseline) ===
        # Advanced variance reduction using learned state-only critic with online λ* optimization
        self.fgb_mode = "online"              # {"online", "meta"}: baseline mode
        self.fgb_lambda_max = 0.8             # Maximum λ* for online/meta modes (clip ceiling)
        # NOTE: This clips the *per-step advantage correction* (delta A_t) in advantage units.
        # It is NOT a return/bps unit. (Kept small to avoid distorting PPO advantage ranking.)
        self.fgb_clip_adv = 0.10
        # Backward-compatible alias (deprecated; kept for old scripts/CLI flags).
        self.fgb_clip_bps = self.fgb_clip_adv
        self.fgb_warmup_steps = 2000          # No correction before this many steps (warm-up period)
        self.fgb_moment_beta = 0.01           # EMA rate for Cov/Var moments (0.01 = ~100-step window)
        # If True, allow negative lambda* when Cov(A,C) < 0 (variance-optimal); otherwise clamp to [0, lambda_max].
        self.fgb_allow_negative_lambda = True
        # Fail-fast guard for Tier-2/Tier-3 baseline paths (recommended for research runs).
        # When enabled, forecast/meta failures raise immediately instead of silently degrading.
        self.fgb_fail_fast = True
        # expected_dnav fallback exposure floor ratio (0.0 disables synthetic exposure when flat).
        self.fgb_expected_dnav_min_exposure_ratio = 0.0
        # expected_dnav blend weights (pred_reward + mwdir). Kept explicit for reproducible tuning.
        self.fgb_expected_dnav_pred_weight = 0.85
        self.fgb_expected_dnav_mwdir_weight = 0.15

        # === FAMC: Meta-Critic Head Training ===
        # FIXED: Default to False (must be explicitly enabled via command-line flag)
        self.meta_baseline_enable = False     # Enable meta-critic head g_φ(x_t) (default: OFF)
        self.meta_baseline_loss = "corr"      # {"mse", "corr"}: corr maximizes correlation with advantage
        self.meta_baseline_head_dim = 128     # Hidden dimension for meta-critic head (increased for capacity)
        self.meta_train_every = 128           # Train meta head every N steps (more frequent updates)
        self.meta_train_min_samples = 128     # Minimum meta buffer size before any head update
        self.meta_train_min_valid_samples = 64  # Minimum valid feature rows per meta update

        # === Forecast Reward Parameters (Tier 2/3) ===
        # FIX: Moderate increase from 5.0 to 10.0 (2x instead of 4x) for balanced learning
        # Analysis showed agent alignment is random (49%) - need stronger incentive but not too aggressive
        # Profitability gating prevents over-rewarding bad trades
        # Moderate multiplier encourages learning without destabilizing reward signal
        self.forecast_alignment_multiplier = 14.0  # Stronger reward for aligning with forecasts
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

        # =============================================================================
        # REINFORCEMENT LEARNING PARAMETERS
        # =============================================================================

        # Training defaults - STRENGTHENED FOR BETTER LEARNING
        self.update_every = 128  # OPTIMIZED: Reduced from 32 for more responsive learning
        self.lr = 1.5e-3  # STRENGTHENED: Increased from 8e-4 to 1.5e-3 for faster learning (87% increase)
        self.ent_coef = 0.030  # Increased to encourage exploration/position-taking (investor especially)
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

        # PPO exploration knobs (optional; recommended for Tier2 to avoid action collapse)
        # - gSDE tends to produce smoother, state-conditioned exploration in continuous control.
        # - log_std_init controls initial action noise scale (std = exp(log_std_init)).
        self.ppo_use_sde = True
        # NOTE: Our training loop is custom (multi-agent) and does not use SB3's rollout collector,
        # so we must explicitly reset gSDE noise ourselves. A small sample freq (1) reduces
        # persistent sign-bias across long episodes.
        self.ppo_sde_sample_freq = 1
        self.ppo_log_std_init = -0.5  # std init ~= 0.61 (kept inside configured clamp range)
        # gSDE only: clamp the actor latent passed into the StateDependentNoiseDistribution.
        # Without this, latent magnitudes can explode and effective sigma becomes huge even if log_std is clamped,
        # which then causes tanh-squashed actions to stick near +/-1 (high saturation, poor learning).
        self.ppo_sde_latent_clip = 0.5
        # Debug: force-disable gSDE regardless of CLI flags
        self.force_disable_sde = False

        # Anti-collapse: prevent PPO exploration std from shrinking to ~0 (which makes actions effectively deterministic).
        # This is especially important across episode boundaries where state distribution shifts.
        # Anti-saturation (critical for avoiding "static +/-1 actions" in evaluation and across episodes)
        # These tighter defaults keep actions in a learnable interior region while preserving exploration.
        # Allow larger exploration floor for investor to take more size
        self.ppo_log_std_min = -2.0    # std >= exp(-2.0) ~= 0.14
        self.ppo_log_std_max = -0.5    # std <= exp(-0.5) ~= 0.61 (matches ppo_log_std_init; avoids forced init clipping)
        self.ppo_mean_clip = 0.85      # clamp mean action logits; tanh(0.85) ~= 0.691

        # Minimal, thesis-defensible regularizer to discourage always-at-bounds allocations.
        # Interpretable as a soft leverage/turnover proxy; applied with warmup in the env reward.
        self.investor_action_l2_penalty = 0.0
        self.investor_action_penalty_warmup_steps = 200

        # Global training step counter (persists across episode environments in episode training).
        # Needed because episode training creates a NEW env each episode, so env-local counters reset.
        self.training_global_step = 0

        # Investor action is exposure-only (single scalar). Allocation is fixed internally.
        # Exposure controller:
        # - "investor": investor policy outputs exposure scalar (1D action)
        # - "meta": meta-controller outputs exposure; investor outputs weights only (3D action)
        self.investor_exposure_controller = "investor"

        # Action squashing (anti-clip artifact). If True, continuous actions that exceed [-1,1]
        # are smoothly squashed with tanh before clipping. If False, we rely on clipping alone.
        # Default OFF: with meta-controlled exposure, we generally prefer not to soften actions unless needed.
        # Turn ON for an ablation if you see hard-clip induced saturation again.
        self.enable_action_tanh_squash = True

        # Novel surgical fix: "boundary barrier" penalty to prevent PPO collapsing to constant +/-1 actions.
        # Penalizes only the portion of |a| above a threshold (e.g., 0.85), so interior actions are unaffected.
        # Investor anti-saturation: keep the policy away from the hard boundary, but don't punish
        # "healthy size" (we want high notional exposure mainly via sizing, not via saturated actions).
        self.investor_action_boundary_penalty = 0.08
        self.investor_action_boundary_threshold = 0.95

        # Structural collapse fix (post 4D action redesign): penalize exposure pegging at 1.0.
        # Hinge penalty: no penalty below threshold; only punishes "always max exposure" failure mode.
        # Stronger default now that warmup is global (won't reset in ep1).
        self.investor_action_exposure_penalty = 0.06
        self.investor_action_exposure_threshold = 0.90  # allow high exposure; only penalize near-max

        # Episode-boundary collapse breaker (targets: exposure stuck at ~1.0 for long stretches).
        # This ramps up ONLY when exposure stays above a high threshold for consecutive steps.
        # It is designed to "unstick" PPO in ep1+ without affecting normal, dynamic regimes.
        self.investor_exposure_stuck_threshold = 0.98
        self.investor_exposure_stuck_steps = 500
        self.investor_exposure_stuck_penalty = 0.50

        # Meta exposure alignment (auxiliary shaping): encourage meta exposure to follow forecast trade_signal.
        # This does NOT add new environment rewards; it only biases training for the meta controller.
        self.meta_exposure_align_weight = 0.02
        self.meta_exposure_align_only_action_steps = True

        # =============================================================================
        # GLOBAL NORMALIZATION (ANTI EPISODE-BOUNDARY DISTRIBUTION SHIFT)
        # =============================================================================
        # Episode training currently normalizes price via rolling mean/std computed inside each 6-month episode.
        # That makes the observation distribution non-stationary across episodes (ep0 vs ep1), which can trigger
        # policy saturation/collapse even when weights are carried over correctly.
        #
        # When enabled, the environment uses global normalization statistics (computed once from the full
        # training_dataset) for price z-scoring, and fixed p95 scales for load/wind/solar/hydro. This makes
        # the observation distribution consistent across episodes and improves continual learning stability.
        self.use_global_normalization = True
        self.global_price_mean = 485.22962686711367
        self.global_price_std = 589.9661249957974
        # Fixed p95 scales derived from full training_dataset (scenario_*.csv, seed 789 set)
        self.global_wind_scale = 1500.0
        self.global_solar_scale = 654.4126
        self.global_hydro_scale = 867.15735
        self.global_load_scale = 4088.5974563779437

        # Network architecture
        self.net_arch = [256, 128, 64]  # Deeper network for better pattern recognition
        self.activation_fn = "relu"  # Changed from tanh for better gradient flow

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
        # Allow fixed settings via min == max (useful for ablations / stabilizing training).
        if self.meta_freq_min > self.meta_freq_max:
            errors.append(f"meta_freq_min ({self.meta_freq_min}) > meta_freq_max ({self.meta_freq_max})")
        if self.meta_cap_min > self.meta_cap_max:
            errors.append(f"meta_cap_min ({self.meta_cap_min}) > meta_cap_max ({self.meta_cap_max})")

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
# PAPER MODES (baseline vs FGB-online vs FGB-meta)
# =====================================================================
# Baseline: forecast_baseline_enable=False, dl_overlay=False
# FGB-online: forecast_baseline_enable=True, fgb_mode='online' (no extra observations)
# FGB-meta: forecast_baseline_enable=True, fgb_mode='meta', meta_baseline_enable=True (no extra observations)
# FGB-ablate-forecasts: same as FGB-online/meta but fgb_ablate_forecasts=True (tau/expected_dnav forced to 0)
