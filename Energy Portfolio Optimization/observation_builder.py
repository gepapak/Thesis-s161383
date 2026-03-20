#!/usr/bin/env python3
"""
Observation Builder Module

Handles observation building for renewable energy portfolio agents:
- Normalize features
- Build agent-specific observations

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
            price: Normalized price signal.
                - Preferred: already scaled to [-1, 1] (as produced by `environment.py` via z-score clipping / 3.0)
                - Backward-compatible: if caller still passes clipped z-score in [-3, 3], this function will scale it to [-1, 1]
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
            # Price: avoid double-normalization.
            # - If input looks like a clipped z-score (magnitude > ~1.5), scale to [-1, 1] via /3.
            # - Otherwise assume it is already normalized to [-1, 1].
            price_f = float(price)
            if abs(price_f) > 1.5:
                price_f = price_f / 3.0
            price_n = float(np.clip(price_f, -1.0, 1.0))
            
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
            raise RuntimeError(f"[OBS_NORMALIZE_FATAL] Market feature normalization failed: {e}") from e
    
    @staticmethod
    def build_investor_observations(
        obs_array: np.ndarray,
        price_momentum: float,
        realized_volatility: float,
        budget: float,
        init_budget: float,
        financial_positions: Dict[str, float],
        max_position_size: float,
        capital_allocation_fraction: float,
        cumulative_mtm_pnl: float,
        is_decision_step: float,
        current_exposure_norm: Optional[float] = None,
        risk_exposure_cap: float = 1.0,
        local_drawdown: float = 0.0,
    ) -> None:
        """
        REFACTORED: Build investor agent observations.
        
        Direct trading investor over a single effective financial factor.
        Uses price momentum plus realized-volatility regime context instead of
        the older per-sleeve bookkeeping positions.
        
        Args:
            obs_array: Observation array to fill (modified in-place)
            price_momentum: Normalized price return/momentum in [-1, 1] (replaces price level)
            realized_volatility: Normalized recent realized volatility in [0, 1]
            budget: Current budget
            init_budget: Initial budget
            financial_positions: Dict with wind/solar/hydro instrument values
            max_position_size: Maximum position size multiplier
            capital_allocation_fraction: Capital allocation fraction
            cumulative_mtm_pnl: Cumulative mark-to-market PnL for sleeve profitability signal
            is_decision_step: 1.0 on investor decision steps else 0.0
            current_exposure_norm: Optional current normalized exposure from the
                live execution contract. If provided, this is preferred over the
                older static normalization formula.
            risk_exposure_cap: Live risk-controller exposure cap seen by the investor.
            local_drawdown: Investor-sleeve local drawdown used by the reward.
        """
        try:
            # Normalize budget to [0, 1]
            budget_n = float(np.clip(SafeDivision.div(budget, init_budget, 0.0), 0.0, 1.0))
            
            # Aggregate exposure normalization. The bookkeeping sleeves share the
            # same traded price factor, so the investor should observe one signed
            # aggregate exposure rather than three redundant per-sleeve values.
            if current_exposure_norm is not None and np.isfinite(float(current_exposure_norm)):
                exposure_norm = float(np.clip(float(current_exposure_norm), -1.0, 1.0))
            else:
                max_pos = max_position_size * init_budget * capital_allocation_fraction if init_budget > 0 else 1.0
                aggregate_position = float(
                    financial_positions['wind_instrument_value']
                    + financial_positions['solar_instrument_value']
                    + financial_positions['hydro_instrument_value']
                )
                exposure_norm = float(np.clip(aggregate_position / max(max_pos, 1.0), -1.0, 1.0))
            
            # mtm_pnl_norm: cumulative PnL / init_budget, clipped to [-1, 1]
            mtm_pnl_norm = float(np.clip(
                SafeDivision.div(cumulative_mtm_pnl, init_budget, 0.0), -1.0, 1.0
            ))
            
            # Base observations (9D): direct trading investor with explicit
            # live execution constraints and local drawdown context.
            obs_array[0] = float(np.clip(price_momentum, -1.0, 1.0))
            obs_array[1] = float(np.clip(realized_volatility, 0.0, 1.0))
            obs_array[2] = budget_n
            obs_array[3] = exposure_norm
            obs_array[4] = mtm_pnl_norm
            obs_array[5] = float(np.clip(is_decision_step, 0.0, 1.0))
            obs_array[6] = float(np.clip(capital_allocation_fraction, 0.0, 1.0))
            obs_array[7] = float(np.clip(risk_exposure_cap, 0.0, 1.0))
            obs_array[8] = float(np.clip(local_drawdown, 0.0, 1.0))
                
        except Exception as e:
            raise RuntimeError(f"[OBS_INVESTOR_FATAL] Investor observation building failed: {e}") from e
    
    @staticmethod
    def build_battery_observations(
        obs_array: np.ndarray,
        price_n: float,
        battery_energy: float,
        battery_capacity_mwh: float,
        load_n: float,
        charge_headroom: float,
        discharge_headroom: float,
        price_reversion_signal: float,
        price_momentum: float,
        realized_price_volatility: float,
        intraday_sin: float,
        intraday_cos: float,
        intraweek_sin: float,
        intraweek_cos: float,
    ) -> None:
        """
        REFACTORED: Build battery operator observations.
        
        Args:
            obs_array: Observation array to fill (modified in-place)
            price_n: Normalized price
            battery_energy: Current battery energy (MWh)
            battery_capacity_mwh: Battery capacity (MWh)
            load_n: Normalized load
            charge_headroom: Available normalized room to charge
            discharge_headroom: Available normalized room to discharge
            price_reversion_signal: Current normalized price-vs-history edge signal
            price_momentum: Short-horizon normalized price momentum
            realized_price_volatility: Short-horizon realized price volatility
        """
        try:
            # Base observations (12D): expose actual arbitrage state instead of
            # only raw price/SOC/load. Capacity is constant in Tier-1 and does
            # not help the policy learn timing. Add cyclical time context for
            # day/week seasonality, which is standard in storage arbitrage RL.
            obs_array[0] = price_n

            soc_normalized = float(np.clip(battery_energy / max(battery_capacity_mwh, 1.0), 0.0, 1.0))
            obs_array[1] = soc_normalized
            obs_array[2] = float(np.clip(charge_headroom, 0.0, 1.0))
            obs_array[3] = float(np.clip(discharge_headroom, 0.0, 1.0))
            obs_array[4] = float(np.clip(price_reversion_signal, -1.0, 1.0))
            obs_array[5] = float(np.clip(price_momentum, -1.0, 1.0))
            obs_array[6] = float(np.clip(realized_price_volatility, 0.0, 1.0))
            obs_array[7] = float(np.clip(load_n, 0.0, 1.0))
            obs_array[8] = float(np.clip(intraday_sin, -1.0, 1.0))
            obs_array[9] = float(np.clip(intraday_cos, -1.0, 1.0))
            obs_array[10] = float(np.clip(intraweek_sin, -1.0, 1.0))
            obs_array[11] = float(np.clip(intraweek_cos, -1.0, 1.0))
            
        except Exception as e:
            raise RuntimeError(f"[OBS_BATTERY_FATAL] Battery observation building failed: {e}") from e
