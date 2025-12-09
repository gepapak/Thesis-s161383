#!/usr/bin/env python3
"""
Asset Manager Module

Handles physical asset management for renewable energy portfolio:
- Wind farm operations
- Solar farm operations
- Hydro plant operations
- Battery storage management

Extracted from environment.py to improve code organization and maintainability.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from utils import SafeDivision, safe_clip, ErrorHandler
from config import EnhancedConfig

logger = logging.getLogger(__name__)


class AssetManager:
    """
    Manages physical renewable energy assets and battery storage.
    
    Responsibilities:
    - Track asset positions (wind, solar, hydro)
    - Manage battery state (charge, discharge, SoC)
    - Calculate generation revenue
    - Update asset states based on actions
    """
    
    def __init__(self, config: EnhancedConfig):
        """
        Initialize asset manager.
        
        Args:
            config: Enhanced configuration object
        """
        self.config = config
        
        # Asset positions (normalized)
        self.wind_position = 0.0
        self.solar_position = 0.0
        self.hydro_position = 0.0
        
        # Battery state
        self.battery_soc = 0.5  # State of charge [0, 1]
        self.battery_capacity_mwh = config.battery_capacity_mwh
        
        # Revenue tracking
        self.cumulative_generation_revenue = 0.0
        self.cumulative_battery_revenue = 0.0
        
        logger.info(f"AssetManager initialized with battery capacity: {self.battery_capacity_mwh} MWh")
    
    def reset(self) -> None:
        """Reset asset manager to initial state."""
        self.wind_position = 0.0
        self.solar_position = 0.0
        self.hydro_position = 0.0
        self.battery_soc = 0.5
        self.cumulative_generation_revenue = 0.0
        self.cumulative_battery_revenue = 0.0
    
    def update_positions(self, wind_action: float, solar_action: float, hydro_action: float) -> None:
        """
        Update asset positions based on investor actions.
        
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
        self.wind_position = safe_clip(self.wind_position + wind_action * 0.1, -1.0, 1.0)
        self.solar_position = safe_clip(self.solar_position + solar_action * 0.1, -1.0, 1.0)
        self.hydro_position = safe_clip(self.hydro_position + hydro_action * 0.1, -1.0, 1.0)
    
    def update_battery(self, charge_action: float, price: float, dt_hours: float = 1/6) -> float:
        """
        Update battery state based on charge/discharge action.
        
        Args:
            charge_action: Charge/discharge action [-1, 1]
                          -1 = full discharge, +1 = full charge
            price: Current electricity price (DKK/MWh)
            dt_hours: Time step duration in hours (default: 10 min = 1/6 hour)
        
        Returns:
            Battery revenue for this step (DKK)
        """
        # Clip action to valid range
        charge_action = safe_clip(charge_action, -1.0, 1.0)
        
        # Calculate energy change (MWh)
        # Positive = charging (buying), Negative = discharging (selling)
        max_power_mw = self.battery_capacity_mwh  # Assume 1C rate
        energy_change_mwh = charge_action * max_power_mw * dt_hours
        
        # Apply SoC constraints
        new_soc = self.battery_soc + (energy_change_mwh / self.battery_capacity_mwh)
        new_soc = safe_clip(new_soc, 0.0, 1.0)
        
        # Actual energy change (accounting for constraints)
        actual_energy_change = (new_soc - self.battery_soc) * self.battery_capacity_mwh
        
        # Update SoC
        self.battery_soc = new_soc
        
        # Calculate revenue (negative for charging, positive for discharging)
        # Discharging (selling) = positive revenue
        # Charging (buying) = negative revenue (cost)
        battery_revenue = -actual_energy_change * price
        
        # Update cumulative revenue
        self.cumulative_battery_revenue += battery_revenue
        
        return battery_revenue
    
    def calculate_generation_revenue(self, wind_gen: float, solar_gen: float, 
                                    hydro_gen: float, price: float) -> float:
        """
        Calculate revenue from renewable generation.
        
        Args:
            wind_gen: Wind generation (MW)
            solar_gen: Solar generation (MW)
            hydro_gen: Hydro generation (MW)
            price: Electricity price (DKK/MWh)
        
        Returns:
            Generation revenue (DKK)
        """
        # Calculate total generation (MW)
        total_gen_mw = wind_gen + solar_gen + hydro_gen
        
        # Convert to revenue (DKK)
        # Assume 10-minute time step = 1/6 hour
        dt_hours = 1/6
        revenue = total_gen_mw * price * dt_hours
        
        # Update cumulative revenue
        self.cumulative_generation_revenue += revenue
        
        return revenue
    
    def get_state(self) -> Dict[str, float]:
        """
        Get current asset manager state.
        
        Returns:
            Dictionary with current state
        """
        return {
            'wind_position': self.wind_position,
            'solar_position': self.solar_position,
            'hydro_position': self.hydro_position,
            'battery_soc': self.battery_soc,
            'cumulative_generation_revenue': self.cumulative_generation_revenue,
            'cumulative_battery_revenue': self.cumulative_battery_revenue,
        }

