from typing import Optional, Dict, Any

class EnhancedConfig:
    """Enhanced configuration class with optimization support and centralized hardcoded values"""

    # Configuration version for validation
    CONFIG_VERSION = "2.1.0"

    def __init__(self, optimized_params: Optional[Dict[str, Any]] = None):
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
        self.operating_cost_rate = 0.03  # 3% of revenue
        self.maintenance_cost_mwh = 3.5 / self.dkk_to_usd_rate  # $3.5/MWh → ~24.1 DKK/MWh
        self.insurance_rate = 0.004  # 0.4% of asset value annually
        self.management_fee_rate = 0.005  # 0.5% of fund value annually (realistic for infrastructure)
        self.property_tax_rate = 0.005  # 0.5% of asset value annually
        self.debt_service_rate = 0.008  # 0.8% of asset value annually (realistic debt service)
        self.distribution_rate = 0.30  # 30% of positive cash distributed
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

        # MTM threshold for position updates
        self.mtm_update_threshold = 1e-9  # Threshold for applying MTM updates to positions

        # NAV bounds for renewable energy fund - INFRASTRUCTURE REALISTIC
        self.nav_min_ratio = 0.90  # INFRASTRUCTURE: Minimum 90% of initial (conservative downside protection)
        self.nav_max_ratio = 1.05  # INFRASTRUCTURE: Maximum 105% of initial (5% return cap - very realistic for infrastructure)



        # =============================================================================
        # ENVIRONMENT AND SIMULATION PARAMETERS
        # =============================================================================

        # Meta controller ranges (BASELINE4 PROVEN CONFIG - KEY FIX!)
        self.meta_freq_min = 6  # Every hour if 10-min data
        self.meta_freq_max = 144  # 24 hours max (was 288 - more responsive)
        self.meta_cap_min = 0.05  # 5% minimum (Baseline4 - was 0.10)
        self.meta_cap_max = 0.75  # 75% maximum capital allocation
        self.sat_eps = 1e-3

        # Investment and operational parameters - CAPITAL PRESERVATION
        self.investment_freq = 24   # CAPITAL PRESERVATION: Every 4 hours for careful positioning
        self.min_investment_freq = 12   # CAPITAL PRESERVATION: 2 hour minimum
        self.max_investment_freq = 288  # CAPITAL PRESERVATION: 48 hours maximum (very patient)
        self.capital_allocation_fraction = 0.15  # CAPITAL PRESERVATION: Only 15% capital allocation
        self.risk_multiplier = 0.5  # CAPITAL PRESERVATION: Minimal risk multiplier

        # Battery operational parameters
        self.batt_soc_min = 0.1  # 10% minimum state of charge
        self.batt_soc_max = 0.9  # 90% maximum state of charge
        self.batt_efficiency = 0.85  # 85% round-trip efficiency

        # =============================================================================
        # MEMORY AND PERFORMANCE PARAMETERS
        # =============================================================================

        # Memory limits (MB)
        self.max_memory_mb = 1500.0  # Environment memory limit
        self.metacontroller_memory_mb = 6000  # Meta controller memory limit
        self.wrapper_memory_mb = 500  # Wrapper memory limit

        # Cache sizes
        self.forecast_cache_size = 1000
        self.agent_forecast_cache_size = 2000
        self.lru_cache_size = 2000
        self.lru_memory_limit_mb = 50.0

        # =============================================================================
        # RISK MANAGEMENT PARAMETERS
        # =============================================================================

        # Risk calculation lookback
        self.risk_lookback_window = 96  # OPTIMIZED: Reduced from 144 to 96 for more responsive risk management

        # OPERATIONAL EXCELLENCE: Capital preservation risk management
        self.max_drawdown_threshold = 0.10  # CAPITAL PRESERVATION: 10% maximum drawdown
        self.volatility_lookback = 60  # CAPITAL PRESERVATION: 60 days for very stable calculations
        self.max_position_size = 0.01  # INFRASTRUCTURE: 1% maximum single position size (minimal trading)
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

        # REMOVED: Duplicate risk_budget_allocation (consolidated above)

        # NEW: Infrastructure fund performance targets
        self.annual_return_target = 0.06   # INFRASTRUCTURE: 6% annual return target (conservative infrastructure)
        self.max_annual_volatility = 0.08  # INFRASTRUCTURE: 8% maximum annual volatility (very stable)
        self.min_sharpe_ratio = 1.5        # INFRASTRUCTURE: High risk-adjusted returns for conservative funds

        # =============================================================================
        # FORECASTING AND PRICING PARAMETERS
        # =============================================================================

        # Price scaling and limits
        self.price_scale = 10.0  # Price normalization scale
        self.price_clip_min = -1000.0  # Minimum price for clipping
        self.price_clip_max = 1e9  # Maximum price for clipping

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

        # Forecast confidence thresholds
        self.forecast_confidence_threshold = 0.30  # AGGRESSIVE: 30% threshold for maximum trading
        self.forecast_confidence_threshold_zero = 0.0  # Zero forecast confidence (no forecast reward)

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

        # =============================================================================
        # REINFORCEMENT LEARNING PARAMETERS
        # =============================================================================

        # Training defaults - OPTIMIZED for better convergence
        self.update_every = 24  # OPTIMIZED: Reduced from 32 for more responsive learning
        self.lr = 8e-4  # OPTIMIZED: Reduced from 1e-3 for more stable convergence
        self.ent_coef = 0.015  # OPTIMIZED: Increased from 0.01 for better exploration
        self.verbose = 1
        self.seed = 42
        self.multithreading = True

        # PPO-specific parameters
        self.batch_size = 128  # USER SPECIFIED: Larger batch for more stable early learning
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

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