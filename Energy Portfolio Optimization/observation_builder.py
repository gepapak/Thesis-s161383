#!/usr/bin/env python3
"""
Observation Builder Module

Handles observation building for renewable energy portfolio agents:
- Normalize features
- Build agent-specific observations
- Handle forecast integration

Extracted from environment.py to improve code organization and maintainability.
"""

import numpy as np
from typing import Dict, Optional, Any
from utils import SafeDivision, safe_clip
from config import EnhancedConfig
from logger import get_logger

logger = get_logger(__name__)


class ObservationBuilder:
    """
    Manages observation building for all agents.
    
    Responsibilities:
    - Normalize market features
    - Build investor observations
    - Build battery operator observations
    - Build risk controller observations
    - Build meta controller observations
    """
    
    @staticmethod
    def normalize_market_features(
        price: float,
        load: float,
        wind: float,
        solar: float,
        hydro: float,
        load_scale: float,
        wind_scale: float,
        solar_scale: float,
        hydro_scale: float
    ) -> Dict[str, float]:
        """
        REFACTORED: Normalize market features to [0, 1] or [-1, 1].
        
        Args:
            price: Price z-score (already normalized)
            load: Load value
            wind: Wind generation value
            solar: Solar generation value
            hydro: Hydro generation value
            load_scale: Load normalization scale
            wind_scale: Wind normalization scale
            solar_scale: Solar normalization scale
            hydro_scale: Hydro normalization scale
            
        Returns:
            Dict with normalized features
        """
        try:
            # Price: z-score already clipped to [-3,3], divide by 3 to get [-1,1]
            price_n = float(np.clip(price / 3.0, -1.0, 1.0))
            
            # Load and generation: normalize to [0, 1]
            load_n = float(np.clip(SafeDivision.div(load, load_scale, 0.0), 0.0, 1.0))
            wind_n = float(np.clip(SafeDivision.div(wind, wind_scale, 0.0), 0.0, 1.0))
            solar_n = float(np.clip(SafeDivision.div(solar, solar_scale, 0.0), 0.0, 1.0))
            hydro_n = float(np.clip(SafeDivision.div(hydro, hydro_scale, 0.0), 0.0, 1.0))
            
            return {
                'price_n': price_n,
                'load_n': load_n,
                'wind_n': wind_n,
                'solar_n': solar_n,
                'hydro_n': hydro_n
            }
        except Exception as e:
            logger.warning(f"Market feature normalization failed: {e}")
            return {
                'price_n': 0.0,
                'load_n': 0.0,
                'wind_n': 0.0,
                'solar_n': 0.0,
                'hydro_n': 0.0
            }
    
    @staticmethod
    def build_investor_observations(
        obs_array: np.ndarray,
        price_n: float,
        budget: float,
        init_budget: float,
        financial_positions: Dict[str, float],
        cumulative_mtm_pnl: float,
        max_position_size: float,
        capital_allocation_fraction: float,
        enable_forecast_util: bool,
        z_short_price: Optional[float] = None,
        z_medium_price: Optional[float] = None,
        forecast_trust: Optional[float] = None
    ) -> None:
        """
        REFACTORED: Build investor agent observations.
        
        Args:
            obs_array: Observation array to fill (modified in-place)
            price_n: Normalized price
            budget: Current budget
            init_budget: Initial budget
            financial_positions: Dict with wind/solar/hydro instrument values
            cumulative_mtm_pnl: Cumulative MTM P&L
            max_position_size: Maximum position size multiplier
            capital_allocation_fraction: Capital allocation fraction
            enable_forecast_util: Whether forecast utilization is enabled
            z_short_price: Short-term price z-score (optional)
            z_medium_price: Medium-term price z-score (optional)
            forecast_trust: Forecast trust score (optional)
        """
        try:
            # Normalize budget to [0, 1]
            budget_n = float(np.clip(SafeDivision.div(budget, init_budget, 0.0), 0.0, 1.0))
            
            # Calculate position normalizations
            max_pos = max_position_size * init_budget * capital_allocation_fraction if init_budget > 0 else 1.0
            wind_pos_norm = float(np.clip(financial_positions['wind_instrument_value'] / max(max_pos, 1.0), -1.0, 1.0))
            solar_pos_norm = float(np.clip(financial_positions['solar_instrument_value'] / max(max_pos, 1.0), -1.0, 1.0))
            hydro_pos_norm = float(np.clip(financial_positions['hydro_instrument_value'] / max(max_pos, 1.0), -1.0, 1.0))
            
            # Calculate MTM P&L normalization
            mtm_pnl_norm = float(np.clip(cumulative_mtm_pnl / max(init_budget, 1.0), -1.0, 1.0))
            
            # Base observations (6D)
            obs_array[0] = price_n
            obs_array[1] = budget_n
            obs_array[2] = wind_pos_norm
            obs_array[3] = solar_pos_norm
            obs_array[4] = hydro_pos_norm
            obs_array[5] = mtm_pnl_norm
            
            # Add forecast features if enabled
            if enable_forecast_util and len(obs_array) >= 9:
                obs_array[6] = float(z_short_price if z_short_price is not None else 0.0)
                obs_array[7] = float(z_medium_price if z_medium_price is not None else 0.0)
                obs_array[8] = float(forecast_trust if forecast_trust is not None else 0.5)
                
        except Exception as e:
            logger.warning(f"Investor observation building failed: {e}")
    
    @staticmethod
    def build_battery_observations(
        obs_array: np.ndarray,
        price_n: float,
        battery_energy: float,
        battery_capacity_mwh: float,
        load_n: float,
        enable_forecast_util: bool,
        z_short_wind: Optional[float] = None,
        z_short_solar: Optional[float] = None,
        z_short_hydro: Optional[float] = None,
        z_short_price: Optional[float] = None,
        z_medium_price: Optional[float] = None,
        z_long_price: Optional[float] = None
    ) -> None:
        """
        REFACTORED: Build battery operator observations.
        
        Args:
            obs_array: Observation array to fill (modified in-place)
            price_n: Normalized price
            battery_energy: Current battery energy (MWh)
            battery_capacity_mwh: Battery capacity (MWh)
            load_n: Normalized load
            enable_forecast_util: Whether forecast utilization is enabled
            z_short_wind: Short-term wind z-score (optional)
            z_short_solar: Short-term solar z-score (optional)
            z_short_hydro: Short-term hydro z-score (optional)
            z_short_price: Short-term price z-score (optional)
            z_medium_price: Medium-term price z-score (optional)
            z_long_price: Long-term price z-score (optional)
        """
        try:
            # Base observations (4D)
            obs_array[0] = price_n
            obs_array[1] = float(np.clip(battery_energy, 0.0, 10.0))
            obs_array[2] = float(np.clip(battery_capacity_mwh, 0.0, 10.0))
            obs_array[3] = load_n
            
            # Add forecast features if enabled
            if enable_forecast_util and len(obs_array) >= 8:
                # Total generation forecast (sum of wind+solar+hydro)
                z_short_total_gen = float(
                    (z_short_wind if z_short_wind is not None else 0.0) +
                    (z_short_solar if z_short_solar is not None else 0.0) +
                    (z_short_hydro if z_short_hydro is not None else 0.0)
                )
                obs_array[4] = float(np.clip(z_short_total_gen, -3.0, 3.0))
                
                # Price forecasts (all horizons)
                obs_array[5] = float(z_short_price if z_short_price is not None else 0.0)
                obs_array[6] = float(z_medium_price if z_medium_price is not None else 0.0)
                obs_array[7] = float(z_long_price if z_long_price is not None else 0.0)
                
        except Exception as e:
            logger.warning(f"Battery observation building failed: {e}")

