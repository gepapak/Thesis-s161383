#!/usr/bin/env python3
"""
Tier System Utilities - Centralized Tier Detection and Configuration

This module provides a single source of truth for tier determination,
observation dimensions, and tier-specific logic.

TIER DEFINITIONS:
- Tier 1 (MARL Baseline): No forecasts, no DL overlay
  - Investor: 6D, Battery: 4D, Risk: 9D, Meta: 11D
  
- Tier 2 (MARL + Forecast Integration): Forecasts enabled, no DL overlay
  - Investor: 14D (6 base + 8 forecast)
  - Battery: 10D (4 base + 6 forecast: 3 gen z-scores + 3 price horizons)
  - Risk: 12D (9 base + 3 forecast)
  - Meta: 13D (11 base + 2 forecast)
  
- Tier 3 (MARL + Forecast + FAMC): Forecasts + DL overlay (FGB/FAMC only)
  - Observation spaces are intentionally identical to Tier 2 for fair comparison.
"""

from typing import Dict, Tuple, Optional
import numpy as np

# Tier names
TIER_1 = "TIER_1"
TIER_2 = "TIER_2"
TIER_3 = "TIER_3"


def determine_tier(enable_forecast_utilisation: bool, 
                   forecast_baseline_enable: bool,
                   dl_overlay_enabled: bool) -> str:
    """
    Determine which tier is active based on configuration flags.
    
    Args:
        enable_forecast_utilisation: Whether forecast integration is enabled
        forecast_baseline_enable: Whether FGB (Forecast-Guided Baseline) is enabled
        dl_overlay_enabled: Whether DL overlay is enabled
        
    Returns:
        Tier name: TIER_1, TIER_2, or TIER_3
    """
    if forecast_baseline_enable or dl_overlay_enabled:
        return TIER_3
    elif enable_forecast_utilisation:
        return TIER_2
    else:
        return TIER_1


def get_expected_observation_dims(tier: str, agent_name: str) -> int:
    """
    Get expected observation dimension for an agent in a given tier.
    
    Args:
        tier: Tier name (TIER_1, TIER_2, or TIER_3)
        agent_name: Agent name ('investor_0', 'battery_operator_0', etc.)
        
    Returns:
        Expected observation dimension
    """
    base_dims = {
        'investor_0': 6,
        'battery_operator_0': 4,
        'risk_controller_0': 9,
        'meta_controller_0': 11,
    }
    
    if tier == TIER_1:
        return base_dims.get(agent_name, 6)
    
    elif tier == TIER_2:
        # Add forecast dimensions
        forecast_dims = {
            'investor_0': 8,      # 6 base + 8 forecast = 14D
            'battery_operator_0': 6,  # 4 base + 6 forecast = 10D
            'risk_controller_0': 3,   # 9 base + 3 forecast = 12D
            'meta_controller_0': 2,   # 11 base + 2 forecast = 13D
        }
        return base_dims.get(agent_name, 6) + forecast_dims.get(agent_name, 0)
    
    elif tier == TIER_3:
        # Tier 3 (FGB/FAMC) is intentionally observation-identical to Tier 2 for fair comparison:
        # no extra bridge-vector observation dimensions.
        #
        # RATIONALE:
        # - We want to isolate the effect of FGB/FAMC variance reduction from representation changes.
        # - Extra bridge dims can add noise / non-stationarity and break fair tier comparisons.
        forecast_dims = {
            'investor_0': 8,      # 6 base + 8 forecast = 14D
            'battery_operator_0': 6,  # 4 base + 6 forecast = 10D
            'risk_controller_0': 3,   # 9 base + 3 forecast = 12D
            'meta_controller_0': 2,   # 11 base + 2 forecast = 13D
        }
        base = base_dims.get(agent_name, 6)
        forecast = forecast_dims.get(agent_name, 0)
        return base + forecast
    
    else:
        raise ValueError(f"Unknown tier: {tier}")


def get_tier_from_config(config) -> str:
    """
    Determine tier from a config object.
    
    Args:
        config: Configuration object with tier flags
        
    Returns:
        Tier name
    """
    enable_forecast_util = getattr(config, 'enable_forecast_utilisation', False)
    forecast_baseline_enable = getattr(config, 'forecast_baseline_enable', False)
    overlay_enabled = getattr(config, 'overlay_enabled', False)
    
    return determine_tier(enable_forecast_util, forecast_baseline_enable, overlay_enabled)


def get_tier_from_env(env) -> str:
    """
    Determine tier from an environment object.
    
    Args:
        env: Environment object with config attribute
        
    Returns:
        Tier name
    """
    if not hasattr(env, 'config'):
        return TIER_1
    
    config = env.config
    enable_forecast_util = getattr(config, 'enable_forecast_utilisation', False)
    forecast_baseline_enable = getattr(config, 'forecast_baseline_enable', False)
    
    # Check for DL overlay (either flag or adapter exists)
    overlay_enabled = getattr(config, 'overlay_enabled', False)
    has_dl_adapter = hasattr(env, 'dl_adapter_overlay') and env.dl_adapter_overlay is not None
    
    return determine_tier(enable_forecast_util, forecast_baseline_enable, overlay_enabled or has_dl_adapter)


def should_include_bridge_dimensions(config) -> bool:
    """
    Determine if bridge dimensions should be included in observation spaces.
    
    Args:
        config: Configuration object
        
    Returns:
        True if bridge dimensions should be included
    """
    # FAIR TIER 3: no bridge-vector dimensions are included in agent observations.
    # Tier 3 is defined as Tier 2 observations + FGB/FAMC (baseline/advantage correction),
    # not as an observation-augmentation method.
    return False


def get_tier_description(tier: str) -> str:
    """
    Get human-readable description of a tier.
    
    Args:
        tier: Tier name
        
    Returns:
        Description string
    """
    descriptions = {
        TIER_1: "MARL Baseline - No forecasts, no DL overlay",
        TIER_2: "MARL + Forecast Integration - Forecasts enabled, no DL overlay",
        TIER_3: "MARL + Forecast + FAMC - Tier 2 observations + FGB/FAMC (DL-assisted baseline only)",
    }
    return descriptions.get(tier, f"Unknown tier: {tier}")

