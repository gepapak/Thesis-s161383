from typing import Optional, Dict, Any

class EnhancedConfig:
    """Enhanced configuration class with optimization support and centralized hardcoded values"""
    def __init__(self, optimized_params: Optional[Dict[str, Any]] = None):
        # =============================================================================
        # FUND STRUCTURE AND ECONOMICS
        # =============================================================================

        # Currency conversion (DKK to USD) - SINGLE SOURCE OF TRUTH
        self.dkk_to_usd_rate = 0.145  # 1 USD = ~6.9 DKK (2024 rate)
        self.currency_conversion = 1.0  # Keep at 1.0 - all calculations in DKK, convert only for final reporting

        # Fund size and allocation
        # FIXED: Convert USD budget to DKK since system operates in DKK
        self.init_budget_usd = 5e8  # $500M fund size in USD
        self.init_budget = self.init_budget_usd / self.dkk_to_usd_rate  # Convert to DKK (~3.45B DKK)

        # Calculate actual allocations based on CAPEX deployment
        # Physical CAPEX: $259M, Remaining cash: $241M
        self.physical_allocation = 0.518  # 51.8% physical assets ($259M)
        self.financial_allocation = 0.482  # 48.2% financial instruments ($241M)

        # Asset ownership fractions (of total installed capacity)
        self.wind_ownership_fraction = 0.05  # 5% of 1,500MW wind farm
        self.solar_ownership_fraction = 0.05  # 5% of 1,000MW solar farm
        self.hydro_ownership_fraction = 0.02  # 2% of 1,000MW hydro plant

        # Total installed capacities (MW)
        self.total_wind_capacity_mw = 1500.0  # Total wind farm capacity
        self.total_solar_capacity_mw = 1000.0  # Total solar farm capacity
        self.total_hydro_capacity_mw = 1000.0  # Total hydro plant capacity

        # Derived owned capacities (calculated from fractions)
        self.owned_wind_capacity_mw = self.wind_ownership_fraction * self.total_wind_capacity_mw  # 75 MW
        self.owned_solar_capacity_mw = self.solar_ownership_fraction * self.total_solar_capacity_mw  # 50 MW
        self.owned_hydro_capacity_mw = self.hydro_ownership_fraction * self.total_hydro_capacity_mw  # 20 MW
        self.owned_battery_capacity_mwh = 10.0  # 10 MWh direct ownership

        # CAPEX values ($/MW or $/MWh)
        self.wind_capex_per_mw = 1800000.0  # $1.8M/MW
        self.solar_capex_per_mw = 1200000.0  # $1.2M/MW
        self.hydro_capex_per_mw = 3000000.0  # $3.0M/MW
        self.battery_capex_per_mwh = 400000.0  # $400k/MWh

        # Operating costs and fees
        self.operating_cost_rate = 0.035  # 3.5% of revenue
        self.maintenance_cost_mwh = 3.5  # $3.5/MWh
        self.insurance_rate = 0.004  # 0.4% of asset value annually
        self.management_fee_rate = 0.015  # 1.5% of fund value annually
        self.property_tax_rate = 0.005  # 0.5% of asset value annually
        self.debt_service_rate = 0.025  # 2.5% of asset value annually
        self.distribution_rate = 0.30  # 30% of positive cash distributed

        # Financial parameters
        self.max_leverage = 2.0  # Increased from 1.5 for higher profit potential
        self.electricity_markup = 1.0  # Fund sells at market price

        # Mark-to-market volatility controls
        self.mtm_price_return_cap_min = -0.02  # Max -2% price return per timestep
        self.mtm_price_return_cap_max = 0.02   # Max +2% price return per timestep



        # =============================================================================
        # ENVIRONMENT AND SIMULATION PARAMETERS
        # =============================================================================

        # Meta controller ranges
        self.meta_freq_min = 6  # Every hour if 10-min data
        self.meta_freq_max = 288  # Daily
        self.meta_cap_min = 0.05  # Increased from 0.02 for more aggressive trading
        self.meta_cap_max = 0.75  # Increased from 0.50 for higher profit potential
        self.sat_eps = 1e-3

        # Investment and operational parameters
        self.investment_freq = 12  # Default investment frequency
        self.capital_allocation_fraction = 0.20  # Increased from 0.10 for more aggressive starting position
        self.risk_multiplier = 1.0

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
        self.risk_lookback_window = 144  # Steps for risk calculation

        # Risk weights (should sum to ~1.0)
        self.risk_weights = {
            'market': 0.25,
            'operational': 0.20,
            'portfolio': 0.25,
            'liquidity': 0.15,
            'regulatory': 0.15
        }

        # Risk thresholds
        self.risk_thresholds = {
            'market_stress_high': 0.85,
            'market_stress_medium': 0.65,
            'volatility_high': 0.80,
            'volatility_medium': 0.50,
            'portfolio_concentration_high': 0.75,
            'liquidity_stress_high': 0.80,
        }

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
        self.portfolio_learning_rate = 0.001
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
        self.forecast_confidence_threshold = 0.05  # Default forecast confidence threshold
        self.forecast_confidence_threshold_zero = 0.0  # Zero forecast confidence (no forecast reward)

        # Ultra fast mode trading override
        self.ultra_fast_mode_trading_enabled = True  # Enable trading in ultra fast mode regardless of forecasts

        # Reward calculation parameters
        self.base_reward_scale = 1.5  # Increased from 1.0 for stronger learning signals
        self.profit_reward_weight = 2.0  # Increased from 1.0 to prioritize profitability
        self.risk_penalty_weight = 0.05  # Reduced from 0.1 to allow more aggressive strategies
        self.forecast_accuracy_reward_weight = 0.05  # Weight for forecast accuracy rewards

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

        # Training defaults
        self.update_every = 64  # Reduced from 128 for more frequent updates
        self.lr = 5e-4  # Increased from 3e-4 for faster learning
        self.ent_coef = 0.02  # Increased from 0.01 for more exploration
        self.verbose = 1
        self.seed = 42
        self.multithreading = True

        # PPO-specific parameters
        self.batch_size = 64
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

    def get_asset_capex(self) -> Dict[str, float]:
        """Get the CAPEX values for different asset types."""
        return {
            'wind_mw': self.wind_capex_per_mw,
            'solar_mw': self.solar_capex_per_mw,
            'hydro_mw': self.hydro_capex_per_mw,
            'battery_mwh': self.battery_capex_per_mwh
        }

    def get_expected_physical_values(self) -> Dict[str, float]:
        """Get expected values for validation checks."""
        actual_physical_capex = self.calculate_total_physical_capex()
        remaining_cash = self.init_budget - actual_physical_capex

        return {
            'wind': self.owned_wind_capacity_mw,
            'solar': self.owned_solar_capacity_mw,
            'hydro': self.owned_hydro_capacity_mw,
            'battery': self.owned_battery_capacity_mwh,
            'physical_book_value': actual_physical_capex,  # Actual CAPEX: $259M
            'cash_min': remaining_cash * 0.8  # At least 80% of remaining cash for trading
        }

    def calculate_total_physical_capex(self) -> float:
        """Calculate total expected CAPEX for physical assets."""
        return (
            self.owned_wind_capacity_mw * self.wind_capex_per_mw +
            self.owned_solar_capacity_mw * self.solar_capex_per_mw +
            self.owned_hydro_capacity_mw * self.hydro_capex_per_mw +
            self.owned_battery_capacity_mwh * self.battery_capex_per_mwh
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