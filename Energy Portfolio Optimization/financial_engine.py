#!/usr/bin/env python3
"""
Financial Engine Module

Handles financial calculations for renewable energy portfolio:
- Mark-to-Market (MTM) calculations
- Net Asset Value (NAV) tracking
- Profit & Loss (PnL) calculations
- Position management

Extracted from environment.py to improve code organization and maintainability.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from utils import SafeDivision, safe_clip, ErrorHandler
from config import EnhancedConfig
from logger import get_logger

logger = get_logger(__name__)


class FinancialEngine:
    """
    Manages financial calculations and position tracking.
    
    Responsibilities:
    - Calculate Mark-to-Market (MTM) values
    - Track Net Asset Value (NAV)
    - Calculate Profit & Loss (PnL)
    - Manage cash and financial positions
    """
    
    def __init__(self, config: EnhancedConfig):
        """
        Initialize financial engine.
        
        Args:
            config: Enhanced configuration object
        """
        self.config = config
        
        # Initial values
        self.initial_nav = config.init_budget
        self.cash = config.init_budget
        
        # Current values
        self.current_nav = config.init_budget
        self.total_value = config.init_budget
        
        # Financial positions (normalized)
        self.financial_wind_position = 0.0
        self.financial_solar_position = 0.0
        self.financial_hydro_position = 0.0
        
        # PnL tracking
        self.cumulative_pnl = 0.0
        self.step_pnl = 0.0
        
        # MTM tracking
        self.last_mtm_price = None
        
        logger.info(f"FinancialEngine initialized with NAV: {self.initial_nav:,.0f} DKK")
    
    def reset(self) -> None:
        """Reset financial engine to initial state."""
        self.cash = self.initial_nav
        self.current_nav = self.initial_nav
        self.total_value = self.initial_nav
        self.financial_wind_position = 0.0
        self.financial_solar_position = 0.0
        self.financial_hydro_position = 0.0
        self.cumulative_pnl = 0.0
        self.step_pnl = 0.0
        self.last_mtm_price = None
    
    def update_financial_positions(self, wind_action: float, solar_action: float, 
                                   hydro_action: float) -> None:
        """
        Update financial derivative positions.
        
        Args:
            wind_action: Wind position change [-1, 1]
            solar_action: Solar position change [-1, 1]
            hydro_action: Hydro position change [-1, 1]
        """
        # Clip actions to valid range
        wind_action = safe_clip(wind_action, -1.0, 1.0)
        solar_action = safe_clip(solar_action, -1.0, 1.0)
        hydro_action = safe_clip(hydro_action, -1.0, 1.0)
        
        # Update positions
        self.financial_wind_position = safe_clip(
            self.financial_wind_position + wind_action * 0.1, -1.0, 1.0
        )
        self.financial_solar_position = safe_clip(
            self.financial_solar_position + solar_action * 0.1, -1.0, 1.0
        )
        self.financial_hydro_position = safe_clip(
            self.financial_hydro_position + hydro_action * 0.1, -1.0, 1.0
        )
    
    def calculate_mtm(self, current_price: float, physical_positions: Dict[str, float],
                     generation: Dict[str, float]) -> float:
        """
        Calculate Mark-to-Market value of all positions.
        
        Args:
            current_price: Current electricity price (DKK/MWh)
            physical_positions: Dict with wind_position, solar_position, hydro_position
            generation: Dict with wind_gen, solar_gen, hydro_gen (MW)
        
        Returns:
            MTM value (DKK)
        """
        # Physical asset MTM (based on generation capacity and price)
        physical_mtm = 0.0
        for asset in ['wind', 'solar', 'hydro']:
            position = physical_positions.get(f'{asset}_position', 0.0)
            gen = generation.get(f'{asset}_gen', 0.0)
            # MTM = position * generation * price * time_horizon
            # Simplified: use daily generation estimate
            physical_mtm += position * gen * current_price * 24  # 24 hours
        
        # Financial derivative MTM (simplified)
        financial_mtm = 0.0
        financial_mtm += self.financial_wind_position * current_price * 1000
        financial_mtm += self.financial_solar_position * current_price * 1000
        financial_mtm += self.financial_hydro_position * current_price * 1000
        
        total_mtm = physical_mtm + financial_mtm
        
        return total_mtm
    
    def calculate_nav(self, mtm_value: float, battery_value: float = 0.0) -> float:
        """
        Calculate Net Asset Value.
        
        Args:
            mtm_value: Mark-to-Market value of positions
            battery_value: Value of battery storage
        
        Returns:
            NAV (DKK)
        """
        nav = self.cash + mtm_value + battery_value
        self.current_nav = nav
        self.total_value = nav
        return nav
    
    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate Profit & Loss for current step.
        
        Args:
            current_price: Current electricity price (DKK/MWh)
        
        Returns:
            Step PnL (DKK)
        """
        if self.last_mtm_price is None:
            self.last_mtm_price = current_price
            self.step_pnl = 0.0
            return 0.0
        
        # Calculate price change
        price_change = current_price - self.last_mtm_price
        
        # Calculate PnL from financial positions
        pnl = 0.0
        pnl += self.financial_wind_position * price_change * 1000
        pnl += self.financial_solar_position * price_change * 1000
        pnl += self.financial_hydro_position * price_change * 1000
        
        # Update tracking
        self.step_pnl = pnl
        self.cumulative_pnl += pnl
        self.last_mtm_price = current_price
        
        return pnl
    
    def get_state(self) -> Dict[str, float]:
        """
        Get current financial engine state.
        
        Returns:
            Dictionary with current state
        """
        return {
            'cash': self.cash,
            'current_nav': self.current_nav,
            'total_value': self.total_value,
            'financial_wind_position': self.financial_wind_position,
            'financial_solar_position': self.financial_solar_position,
            'financial_hydro_position': self.financial_hydro_position,
            'cumulative_pnl': self.cumulative_pnl,
            'step_pnl': self.step_pnl,
        }
    
    # ============================================================================
    # REFACTORED: Finance calculation methods extracted from environment.py
    # These methods are stateless and take all needed state as parameters
    # ============================================================================
    
    @staticmethod
    def calculate_fund_nav(
        budget: float,
        physical_assets: Dict[str, float],
        asset_capex: Dict[str, float],
        financial_positions: Dict[str, float],
        accumulated_operational_revenue: float,
        current_timestep: int,
        config: EnhancedConfig
    ) -> float:
        """
        REFACTORED: Calculate true fund NAV with proper separation.
        NAV = Trading Cash + Physical Asset Book Value + Accumulated Operational Revenue + Financial Instrument MTM
        
        Args:
            budget: Trading cash position
            physical_assets: Dict with wind_capacity_mw, solar_capacity_mw, hydro_capacity_mw, battery_capacity_mwh
            asset_capex: Dict with wind_mw, solar_mw, hydro_mw, battery_mwh (CAPEX per unit)
            financial_positions: Dict with wind_instrument_value, solar_instrument_value, hydro_instrument_value
            accumulated_operational_revenue: Total accumulated operational revenue
            current_timestep: Current timestep for depreciation calculation
            config: Configuration object
            
        Returns:
            Fund NAV (DKK)
        """
        try:
            # 1) Trading cash position (separate from operational revenue)
            trading_cash_value = max(0.0, budget)
            
            # 2) Physical assets with realistic depreciation
            # Infrastructure assets depreciate over time (typical 20-30 year life)
            years_elapsed = current_timestep / (365.25 * 24 * 6)  # Convert timesteps to years (10-min intervals)
            
            # Use config parameters for depreciation
            annual_depreciation_rate = getattr(config, 'annual_depreciation_rate', 0.02)
            max_depreciation = getattr(config, 'max_depreciation_ratio', 0.75)
            total_depreciation = min(years_elapsed * annual_depreciation_rate, max_depreciation)
            
            # Apply uniform depreciation to all physical assets
            wind_depreciation = total_depreciation
            solar_depreciation = total_depreciation
            hydro_depreciation = total_depreciation
            battery_depreciation = total_depreciation
            
            physical_book_value = (
                physical_assets['wind_capacity_mw'] * asset_capex['wind_mw'] * (1.0 - wind_depreciation) +
                physical_assets['solar_capacity_mw'] * asset_capex['solar_mw'] * (1.0 - solar_depreciation) +
                physical_assets['hydro_capacity_mw'] * asset_capex['hydro_mw'] * (1.0 - hydro_depreciation) +
                physical_assets['battery_capacity_mwh'] * asset_capex['battery_mwh'] * (1.0 - battery_depreciation)
            )
            
            # 3) Accumulated operational revenue (separate from trading cash)
            operational_revenue_value = accumulated_operational_revenue
            
            # 4) Financial instruments at mark-to-market
            financial_mtm_value = (
                financial_positions['wind_instrument_value'] +
                financial_positions['solar_instrument_value'] +
                financial_positions['hydro_instrument_value']
            )
            
            # NAV calculation
            fund_equity = trading_cash_value + physical_book_value + operational_revenue_value + financial_mtm_value
            
            # Keep financial instrument bounds for realism (based on allocated trading capital)
            trading_limits = config.get_trading_capital_limits()
            max_financial_exposure = trading_limits['max_financial_exposure_dkk']
            
            if abs(financial_mtm_value) > max_financial_exposure:
                # Clip financial instruments to allocated trading capital × leverage
                financial_mtm_value = float(np.clip(financial_mtm_value,
                                                  -max_financial_exposure,
                                                   max_financial_exposure))
                fund_equity = trading_cash_value + physical_book_value + operational_revenue_value + financial_mtm_value
            
            # Use unconstrained NAV to show natural fund behavior
            # Fund equity is the remaining equity (excluding distributed profits)
            return float(max(fund_equity, 0.0))  # Only prevent negative NAV
            
        except Exception as e:
            logger.error(f"NAV calculation error: {e}")
            return max(budget, config.init_budget * 0.01)
    
    @staticmethod
    def calculate_timestep_costs(
        fund_physical_investment: float,
        gross_revenue: float,
        fund_total_generation_mwh: float,
        config: EnhancedConfig
    ) -> float:
        """
        REFACTORED: Calculate total operational and administrative costs for one timestep.
        
        Args:
            fund_physical_investment: Total capital invested in physical assets (DKK)
            gross_revenue: Gross revenue from electricity sales (DKK)
            fund_total_generation_mwh: Total generation for this timestep (MWh)
            config: Configuration object
            
        Returns:
            Total operating costs for this timestep (DKK)
        """
        time_step_hours = config.time_step_hours
        annual_to_timestep = time_step_hours / 8760
        
        # Variable costs (based on gross revenue/generation)
        variable_costs = gross_revenue * config.operating_cost_rate
        maintenance_costs = fund_total_generation_mwh * config.maintenance_cost_mwh
        grid_connection_costs = fund_total_generation_mwh * config.grid_connection_fee_mwh
        transmission_costs = fund_total_generation_mwh * config.transmission_fee_mwh
        
        # Fixed annual costs (prorated by fund's physical investment)
        insurance_costs = fund_physical_investment * config.insurance_rate * annual_to_timestep
        property_taxes = fund_physical_investment * config.property_tax_rate * annual_to_timestep
        debt_service = fund_physical_investment * config.debt_service_rate * annual_to_timestep
        management_fees = config.init_budget * config.management_fee_rate * annual_to_timestep
        
        total_costs = (variable_costs + maintenance_costs + grid_connection_costs +
                      transmission_costs + insurance_costs + property_taxes +
                      debt_service + management_fees)
        
        return float(total_costs)
    
    @staticmethod
    def calculate_generation_revenue(
        timestep: int,
        price: float,
        wind_data: np.ndarray,
        solar_data: np.ndarray,
        hydro_data: np.ndarray,
        wind_scale: float,
        solar_scale: float,
        hydro_scale: float,
        physical_assets: Dict[str, float],
        asset_capex: Dict[str, float],
        config: EnhancedConfig,
        electricity_markup: float = 1.0,
        currency_conversion: float = 1.0
    ) -> float:
        """
        REFACTORED: Calculate revenue from PHYSICAL ASSETS only.
        Revenue = (Physical Generation * Market Price) - Operating Costs
        
        Args:
            timestep: Current timestep index
            price: Current electricity price (DKK/MWh)
            wind_data: Wind generation data array
            solar_data: Solar generation data array
            hydro_data: Hydro generation data array
            wind_scale: Wind data normalization scale
            solar_scale: Solar data normalization scale
            hydro_scale: Hydro data normalization scale
            physical_assets: Dict with asset capacities
            asset_capex: Dict with CAPEX per unit
            config: Configuration object
            electricity_markup: Markup multiplier for electricity sales
            currency_conversion: Currency conversion factor
            
        Returns:
            Net generation revenue (DKK)
        """
        try:
            time_step_hours = config.time_step_hours  # From config: 10-minute timesteps
            
            # Get actual generation from training data
            full_wind_generation_mw = float(wind_data[timestep]) if timestep < len(wind_data) else 0.0
            full_solar_generation_mw = float(solar_data[timestep]) if timestep < len(solar_data) else 0.0
            full_hydro_generation_mw = float(hydro_data[timestep]) if timestep < len(hydro_data) else 0.0
            
            # Convert to MWh for this timestep
            full_wind_generation_mwh = full_wind_generation_mw * time_step_hours
            full_solar_generation_mwh = full_solar_generation_mw * time_step_hours
            full_hydro_generation_mwh = full_hydro_generation_mw * time_step_hours
            full_total_generation_mwh = full_wind_generation_mwh + full_solar_generation_mwh + full_hydro_generation_mwh
            
            # Fractional ownership percentages (applied only to revenue)
            wind_ownership_pct = config.wind_ownership_fraction
            solar_ownership_pct = config.solar_ownership_fraction
            hydro_ownership_pct = config.hydro_ownership_fraction
            
            # Fund's revenue share (fractional ownership applied here only)
            fund_wind_generation_mwh = full_wind_generation_mwh * wind_ownership_pct
            fund_solar_generation_mwh = full_solar_generation_mwh * solar_ownership_pct
            fund_hydro_generation_mwh = full_hydro_generation_mwh * hydro_ownership_pct
            fund_total_generation_mwh = fund_wind_generation_mwh + fund_solar_generation_mwh + fund_hydro_generation_mwh
            
            # Safety check
            if fund_total_generation_mwh <= 0.001:
                return 0.0
            
            # Revenue from electricity sales
            effective_price = max(price, config.minimum_price_floor)
            gross_revenue = fund_total_generation_mwh * effective_price * electricity_markup * currency_conversion
            
            # Calculate fund's physical investment
            fund_physical_investment = (
                physical_assets['wind_capacity_mw'] * asset_capex['wind_mw'] +
                physical_assets['solar_capacity_mw'] * asset_capex['solar_mw'] +
                physical_assets['hydro_capacity_mw'] * asset_capex['hydro_mw'] +
                physical_assets['battery_capacity_mwh'] * asset_capex['battery_mwh']
            )
            
            # Calculate operating costs
            total_operating_costs = FinancialEngine.calculate_timestep_costs(
                fund_physical_investment,
                gross_revenue,
                fund_total_generation_mwh,
                config
            )
            
            net_revenue = max(0.0, gross_revenue - total_operating_costs)
            return float(net_revenue)
            
        except Exception as e:
            logger.warning(f"Generation revenue calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def distribute_excess_cash(
        budget: float,
        current_fund_nav: float,
        init_budget: float,
        config: EnhancedConfig
    ) -> Tuple[float, float]:
        """
        REFACTORED: Distribute excess cash to maintain realistic cash levels.
        Infrastructure funds typically maintain 5-15% cash and distribute excess to investors.
        
        Args:
            budget: Current trading cash budget
            current_fund_nav: Current fund NAV
            init_budget: Initial budget for logging thresholds
            config: Configuration object
            
        Returns:
            Tuple of (new_budget, distribution_amount)
        """
        try:
            target_cash_ratio = getattr(config, 'target_cash_ratio', 0.10)
            min_distribution_ratio = getattr(config, 'min_distribution_threshold_ratio', 0.005)
            
            target_cash_level = current_fund_nav * target_cash_ratio
            excess_cash = budget - target_cash_level
            distribution_threshold = current_fund_nav * min_distribution_ratio
            
            if excess_cash > distribution_threshold:
                distribution_rate = getattr(config, 'distribution_rate', 0.30)
                distribution_amount = excess_cash * distribution_rate
                new_budget = budget - distribution_amount
                
                # Log significant distributions
                if distribution_amount > init_budget * 0.01:  # >1% of fund
                    logger.info(f"Cash distribution: {distribution_amount:,.0f} DKK (${distribution_amount * config.dkk_to_usd_rate / 1e6:.1f}M USD)")
                
                return new_budget, distribution_amount
            else:
                return budget, 0.0
                
        except Exception as e:
            logger.warning(f"Cash distribution failed: {e}")
            return budget, 0.0
    
    @staticmethod
    def check_emergency_reallocation(
        budget: float,
        accumulated_operational_revenue: float,
        operational_revenue: float,
        trading_allocation_budget: float,
        total_fund_value: float,
        total_reallocated: float,
        config: EnhancedConfig
    ) -> Tuple[float, float, float]:
        """
        REFACTORED: Emergency reallocation from operational gains to trading capital.
        Allows operational revenue to replenish trading capital under strict controls.
        
        Args:
            budget: Current trading cash budget
            accumulated_operational_revenue: Total accumulated operational revenue
            operational_revenue: Operational revenue for this timestep
            trading_allocation_budget: Original trading allocation budget
            total_fund_value: Total fund value
            total_reallocated: Total amount already reallocated
            config: Configuration object
            
        Returns:
            Tuple of (new_budget, new_accumulated_operational_revenue, new_total_reallocated)
        """
        try:
            # Check if trading capital is below emergency threshold
            current_trading_ratio = budget / trading_allocation_budget if trading_allocation_budget > 0 else 0
            
            # Emergency conditions
            below_threshold = current_trading_ratio < config.trading_capital_emergency_threshold
            has_operational_gains = operational_revenue > 0
            under_total_limit = total_reallocated < (total_fund_value * config.max_total_reallocation)
            
            if below_threshold and has_operational_gains and under_total_limit:
                # Calculate reallocation amount
                max_from_operational = operational_revenue * config.max_reallocation_rate
                max_remaining_total = (total_fund_value * config.max_total_reallocation) - total_reallocated
                
                reallocation_amount = min(max_from_operational, max_remaining_total)
                
                if reallocation_amount > 1000:  # Minimum 1000 DKK threshold
                    # Execute reallocation
                    new_budget = budget + reallocation_amount
                    new_accumulated_operational_revenue = accumulated_operational_revenue - reallocation_amount
                    new_total_reallocated = total_reallocated + reallocation_amount
                    
                    logger.info(f"Emergency reallocation: {reallocation_amount:,.0f} DKK from operational to trading")
                    logger.info(f"Trading capital ratio: {current_trading_ratio:.1%} -> {(new_budget / trading_allocation_budget):.1%}")
                    logger.info(f"Total reallocated: {new_total_reallocated:,.0f} DKK ({new_total_reallocated / total_fund_value:.1%} of fund)")
                    
                    return new_budget, new_accumulated_operational_revenue, new_total_reallocated
            
            return budget, accumulated_operational_revenue, total_reallocated
            
        except Exception as e:
            logger.error(f"Emergency reallocation check failed: {e}")
            return budget, accumulated_operational_revenue, total_reallocated
    
    @staticmethod
    def calculate_price_returns(
        timestep: int,
        current_price: float,
        price_history: np.ndarray,
        enable_forecast_util: bool,
        horizon_correlations: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        REFACTORED: Calculate multi-horizon price returns for forecast reward alignment.
        
        Args:
            timestep: Current timestep index
            current_price: Current electricity price (DKK/MWh)
            price_history: Historical price array
            enable_forecast_util: Whether forecast utilization is enabled
            horizon_correlations: Optional dict with 'short', 'medium', 'long' correlations
            
        Returns:
            Dict with price returns and correlation-based weights
        """
        try:
            prev_price = float(np.clip(price_history[timestep-1] if timestep > 0 else current_price, -1000.0, 1e9))
            current_price = float(np.clip(current_price, -1000.0, 1e9))
            
            price_return_short = None
            price_return_medium = None
            price_return_long = None
            
            if enable_forecast_util and timestep >= 6:
                # Short horizon: 6 steps back
                price_6_steps_ago = float(np.clip(price_history[timestep-6], -1000.0, 1e9))
                price_return_short = (current_price - price_6_steps_ago) / max(abs(price_6_steps_ago), 1e-6)
                
                # Medium horizon: 24 steps back
                if timestep >= 24:
                    price_24_steps_ago = float(np.clip(price_history[timestep-24], -1000.0, 1e9))
                    price_return_medium = (current_price - price_24_steps_ago) / max(abs(price_24_steps_ago), 1e-6)
                else:
                    price_return_medium = price_return_short
                
                # Long horizon: 144 steps back
                if timestep >= 144:
                    price_144_steps_ago = float(np.clip(price_history[timestep-144], -1000.0, 1e9))
                    price_return_long = (current_price - price_144_steps_ago) / max(abs(price_144_steps_ago), 1e-6)
                else:
                    price_return_long = price_return_medium
                
                # Correlation-based weighting
                if horizon_correlations:
                    corr_short = horizon_correlations.get('short', 0.0)
                    corr_medium = horizon_correlations.get('medium', 0.0)
                    corr_long = horizon_correlations.get('long', 0.0)
                    
                    use_short = corr_short > 0.0
                    use_medium = corr_medium > 0.0
                    use_long = corr_long > 0.0
                    
                    if not (use_short or use_medium or use_long):
                        weight_short = 0.7 if use_short else 0.0
                        weight_medium = 0.2 if use_medium else 0.0
                        weight_long = 0.1 if use_long else 0.0
                    else:
                        epsilon = 1e-6
                        weight_short_raw = max(0.0, corr_short) if use_short else 0.0
                        weight_medium_raw = max(0.0, corr_medium) if use_medium else 0.0
                        weight_long_raw = max(0.0, corr_long) if use_long else 0.0
                        
                        total_weight = weight_short_raw + weight_medium_raw + weight_long_raw
                        if total_weight > epsilon:
                            weight_short = weight_short_raw / total_weight
                            weight_medium = weight_medium_raw / total_weight
                            weight_long = weight_long_raw / total_weight
                        else:
                            num_horizons = sum([use_short, use_medium, use_long])
                            if num_horizons > 0:
                                weight_short = (1.0 / num_horizons) if use_short else 0.0
                                weight_medium = (1.0 / num_horizons) if use_medium else 0.0
                                weight_long = (1.0 / num_horizons) if use_long else 0.0
                            else:
                                weight_short, weight_medium, weight_long = 0.0, 0.0, 0.0
                else:
                    weight_short, weight_medium, weight_long = 0.7, 0.2, 0.1
                    corr_short = corr_medium = corr_long = 0.0
                    use_short = use_medium = use_long = False
                
                # Combined forecast return
                price_return_forecast = float(
                    weight_short * price_return_short + 
                    weight_medium * price_return_medium + 
                    weight_long * price_return_long
                )
            else:
                # Fallback: 1-step return
                price_return_forecast = float((current_price - prev_price) / max(abs(prev_price), 1e-6))
                weight_short = weight_medium = weight_long = 0.0
                corr_short = corr_medium = corr_long = 0.0
                use_short = use_medium = use_long = False
            
            # 1-step return for MTM
            price_return = float((current_price - prev_price) / max(abs(prev_price), 1e-6))
            
            # Fill in missing values
            if price_return_short is None:
                price_return_short = price_return
            if price_return_medium is None:
                price_return_medium = price_return
            if price_return_long is None:
                price_return_long = price_return
            
            return {
                'price_return': price_return,
                'price_return_short': price_return_short,
                'price_return_medium': price_return_medium,
                'price_return_long': price_return_long,
                'price_return_forecast': price_return_forecast,
                'correlation_debug': {
                    'corr_short': corr_short if enable_forecast_util else 0.0,
                    'corr_medium': corr_medium if enable_forecast_util else 0.0,
                    'corr_long': corr_long if enable_forecast_util else 0.0,
                    'weight_short': weight_short if enable_forecast_util else 0.0,
                    'weight_medium': weight_medium if enable_forecast_util else 0.0,
                    'weight_long': weight_long if enable_forecast_util else 0.0,
                    'use_short': use_short if enable_forecast_util else False,
                    'use_medium': use_medium if enable_forecast_util else False,
                    'use_long': use_long if enable_forecast_util else False,
                }
            }
            
        except Exception as e:
            logger.error(f"Price return calculation failed: {e}")
            price_return = float((current_price - prev_price) / max(abs(prev_price), 1e-6))
            return {
                'price_return': price_return,
                'price_return_short': price_return,
                'price_return_medium': price_return,
                'price_return_long': price_return,
                'price_return_forecast': price_return,
                'correlation_debug': {}
            }
    
    @staticmethod
    def calculate_mtm_pnl(
        financial_positions: Dict[str, float],
        price_return: float,
        config: EnhancedConfig
    ) -> Tuple[float, Dict[str, float]]:
        """
        REFACTORED: Calculate Mark-to-Market P&L for financial instruments.
        
        Args:
            financial_positions: Dict with wind_instrument_value, solar_instrument_value, hydro_instrument_value
            price_return: 1-step price return
            config: Configuration object
            
        Returns:
            Tuple of (mtm_pnl, updated_positions)
        """
        try:
            total_financial_exposure = (
                financial_positions['wind_instrument_value'] +
                financial_positions['solar_instrument_value'] +
                financial_positions['hydro_instrument_value']
            )
            
            # Cap price returns to realistic energy market volatility
            cap_min = getattr(config, 'mtm_price_return_cap_min', -0.015)
            cap_max = getattr(config, 'mtm_price_return_cap_max', 0.015)
            capped_price_return = float(np.clip(price_return, cap_min, cap_max))
            
            mtm_pnl = total_financial_exposure * capped_price_return
            
            # Update position values with MTM
            updated_positions = financial_positions.copy()
            if abs(mtm_pnl) > getattr(config, 'mtm_update_threshold', 1e-9):
                if abs(total_financial_exposure) > 1e-9:
                    wind_mtm = financial_positions['wind_instrument_value'] * capped_price_return
                    solar_mtm = financial_positions['solar_instrument_value'] * capped_price_return
                    hydro_mtm = financial_positions['hydro_instrument_value'] * capped_price_return
                    
                    updated_positions['wind_instrument_value'] += wind_mtm
                    updated_positions['solar_instrument_value'] += solar_mtm
                    updated_positions['hydro_instrument_value'] += hydro_mtm
            
            return mtm_pnl, updated_positions
            
        except Exception as e:
            logger.error(f"MTM P&L calculation failed: {e}")
            return 0.0, financial_positions.copy()

