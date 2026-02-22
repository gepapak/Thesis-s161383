#!/usr/bin/env python3
"""
Trading Engine Module

Handles trading operations for renewable energy portfolio:
- Investor trade execution
- Battery operations
- Risk control application
- Meta control application

Extracted from environment.py to improve code organization and maintainability.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from utils import SafeDivision, safe_clip, ErrorHandler
from config import EnhancedConfig

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Manages trading operations and control applications.
    
    Responsibilities:
    - Execute investor trades
    - Execute battery operations
    - Apply risk control
    - Apply meta control
    """
    
    @staticmethod
    def apply_risk_control(risk_action: np.ndarray) -> float:
        """
        REFACTORED: Apply risk control multiplier from risk controller action.
        
        Args:
            risk_action: Risk controller action array [-1, 1]
            
        Returns:
            Risk multiplier [0.3, 2.5]
        """
        try:
            val = float(np.clip(risk_action.reshape(-1)[0], -1.0, 1.0))
            # Convert [-1,1] to [0,2] then to [0.3,2.5]
            normalized_val = (val + 1.0)  # [-1,1] -> [0,2]
            risk_multiplier = 0.3 + 1.1 * normalized_val  # [0.3,2.5]
            return float(np.clip(risk_multiplier, 0.3, 2.5))
        except Exception as e:
            msg = f"[RISK_CONTROL_FATAL] Risk control application failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
    
    @staticmethod
    def apply_meta_control(
        meta_action: np.ndarray,
        meta_cap_min: float,
        meta_cap_max: float,
        meta_freq_min: int,
        meta_freq_max: int,
        forecast_confidence: float = 0.5,
        disable_confidence_scaling: bool = False
    ) -> Tuple[float, int]:
        """
        FIX #4: Apply meta control with forecast confidence scaling.
        
        Apply meta control for capital allocation and trading frequency,
        scaled by forecast confidence to increase/decrease position sizing
        based on forecast quality.
        
        Args:
            meta_action: Meta controller action array [-1, 1]
            meta_cap_min: Minimum capital allocation fraction
            meta_cap_max: Maximum capital allocation fraction
            meta_freq_min: Minimum trading frequency
            meta_freq_max: Maximum trading frequency
            forecast_confidence: [0.2, 1.0] from forecast_trust
            
        Returns:
            Tuple of (capital_allocation_fraction, investment_freq)
        """
        try:
            a0, a1 = np.array(meta_action, dtype=np.float32).reshape(-1)[:2]
            
            # Generic symmetric mapping function for [-1,1] -> [min,max]
            def map_from_minus1_1(x, lo, hi):
                x = float(np.clip(x, -1.0, 1.0))
                return lo + (x + 1.0) * 0.5 * (hi - lo)
            
            # Apply symmetric mapping to both components
            cap = map_from_minus1_1(a0, meta_cap_min, meta_cap_max)
            
            # FIX #4: Scale capital allocation by forecast confidence (unless disabled for fair A/B)
            if disable_confidence_scaling:
                confidence_multiplier = 1.0
            else:
                # Map confidence [0.2, 1.0] to multiplier [0.7, 1.5]
                forecast_confidence = float(np.clip(forecast_confidence, 0.2, 1.0))
                confidence_multiplier = 0.7 + (forecast_confidence - 0.2) / 0.8 * 0.8  # [0.7, 1.5]
            
            # Apply confidence scaling to capital allocation
            # SAFETY: Never allow allocating more than 100% of capital.
            clip_lo = max(0.0, float(meta_cap_min) * 0.7)  # Allow up to 30% reduction
            clip_hi = min(1.0, float(meta_cap_max) * 1.5)  # Allow up to 50% increase but cap at 1.0
            if clip_hi < clip_lo:
                clip_hi = clip_lo
            capital_allocation_fraction = float(np.clip(cap * confidence_multiplier, clip_lo, clip_hi))
            
            freq = int(round(map_from_minus1_1(a1, meta_freq_min, meta_freq_max)))
            investment_freq = int(np.clip(freq, meta_freq_min, meta_freq_max))
            
            return capital_allocation_fraction, investment_freq
            
        except Exception as e:
            msg = f"[META_CONTROL_FATAL] Meta control application failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
    
    @staticmethod
    def calculate_battery_dispatch_policy(
        current_price: float,
        forecast_price: Optional[float],
        battery_hurdle_min_dkk: float,
        battery_price_sensitivity: float,
        battery_rt_loss_weight: float,
        batt_eta_charge: float,
        batt_eta_discharge: float,
        minimum_price_filter: float,
        maximum_price_cap: float,
        price_volatility_forecast: float = 0.0
    ) -> Tuple[str, float]:
        """
        FIX #5: Decide battery dispatch policy with volatility weighting.
        
        Decide battery dispatch policy ('charge'/'discharge'/'idle', intensity 0..1),
        with volatility weighting to increase arbitrage in high-volatility periods.
        
        Args:
            current_price: Current electricity price (DKK/MWh)
            forecast_price: Forecasted future price (DKK/MWh) or None
            battery_hurdle_min_dkk: Minimum price spread hurdle (DKK/MWh)
            battery_price_sensitivity: Price sensitivity factor
            battery_rt_loss_weight: Round-trip loss weight
            batt_eta_charge: Battery charge efficiency
            batt_eta_discharge: Battery discharge efficiency
            minimum_price_filter: Minimum price filter
            maximum_price_cap: Maximum price cap
            price_volatility_forecast: Expected price volatility [0, 1]
            
        Returns:
            Tuple of (decision, intensity) where decision is 'charge', 'discharge', or 'idle'
        """
        try:
            p_now = float(np.clip(current_price, minimum_price_filter, maximum_price_cap))
            
            if forecast_price is None or not np.isfinite(forecast_price):
                return ("idle", 0.0)
            
            p_fut = float(np.clip(forecast_price, minimum_price_filter, maximum_price_cap))
            spread = p_fut - p_now
            
            needed = max(battery_hurdle_min_dkk, battery_price_sensitivity * abs(p_now))
            rt_loss = (1.0/(max(batt_eta_charge*batt_eta_discharge, 1e-6)) - 1.0) * 10.0
            hurdle = needed + battery_rt_loss_weight * rt_loss
            
            # FIX #5: Reduce hurdle by up to 30% when volatility is high
            # High volatility = more profitable arbitrage opportunities
            price_volatility_forecast = float(np.clip(price_volatility_forecast, 0.0, 1.0))
            volatility_adjustment = 1.0 - (0.3 * price_volatility_forecast)  # [1.0, 0.7]
            adjusted_hurdle = hurdle * volatility_adjustment
            
            if spread > adjusted_hurdle:
                inten = float(np.clip(spread / (abs(p_now) + 0.1), 0.0, 1.0))
                # Boost intensity slightly for high volatility
                inten = float(np.clip(inten * (1.0 + 0.3 * price_volatility_forecast), 0.0, 1.0))
                return ("charge", inten)
            elif spread < -adjusted_hurdle:
                inten = float(np.clip((-spread) / (abs(p_now) + 0.1), 0.0, 1.0))
                # Boost intensity slightly for high volatility
                inten = float(np.clip(inten * (1.0 + 0.3 * price_volatility_forecast), 0.0, 1.0))
                return ("discharge", inten)
            else:
                return ("idle", 0.0)
                
        except Exception as e:
            msg = f"[BATTERY_DISPATCH_FATAL] Battery dispatch policy calculation failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
    
    @staticmethod
    def execute_battery_operations(
        bat_action: np.ndarray,
        timestep: int,
        battery_capacity_mwh: float,
        battery_energy: float,
        price: float,
        batt_power_c_rate: float,
        batt_eta_charge: float,
        batt_eta_discharge: float,
        batt_soc_min: float,
        batt_soc_max: float,
        batt_degradation_cost: float,
        battery_dispatch_policy_fn: Optional[callable] = None,
        config: Optional[EnhancedConfig] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        REFACTORED: Execute battery operations and return cash delta.
        
        Args:
            bat_action: Battery operator action array [-1, 1]
            timestep: Current timestep
            battery_capacity_mwh: Battery capacity (MWh)
            battery_energy: Current battery energy (MWh)
            price: Current electricity price (DKK/MWh)
            batt_power_c_rate: Battery power C-rate
            batt_eta_charge: Battery charge efficiency
            batt_eta_discharge: Battery discharge efficiency
            batt_soc_min: Minimum state of charge
            batt_soc_max: Maximum state of charge
            batt_degradation_cost: Degradation cost per MWh
            battery_dispatch_policy_fn: Optional function to get dispatch policy
            config: Configuration object
            
        Returns:
            Tuple of (cash_delta, updated_operational_state)
        """
        try:
            u_raw = float(np.clip(bat_action.reshape(-1)[0], -1.0, 1.0))
            
            if battery_capacity_mwh <= 0.0:
                return 0.0, {'battery_energy': 0.0, 'battery_discharge_power': 0.0}
            
            step_h = 10.0 / 60.0
            max_power_mw = battery_capacity_mwh * batt_power_c_rate
            max_energy_this_step = max_power_mw * step_h
            
            # Get heuristic policy if available
            heuristic_decision = "idle"
            heuristic_inten = 0.0
            if battery_dispatch_policy_fn:
                try:
                    heuristic_decision, heuristic_inten = battery_dispatch_policy_fn(timestep)
                except Exception as e:
                    raise RuntimeError(
                        f"[BATTERY_POLICY_FATAL] battery_dispatch_policy_fn failed at timestep {timestep}: {e}"
                    ) from e
            
            # Agent action determines decision and intensity
            agent_intensity = abs(u_raw)
            if u_raw < -0.2:
                agent_decision = "charge"
            elif u_raw > 0.2:
                agent_decision = "discharge"
            else:
                agent_decision = "idle"
            
            # Blend decisions: 90% agent, 10% heuristic
            if agent_decision == heuristic_decision:
                decision = agent_decision
                inten = 0.9 * agent_intensity + 0.1 * heuristic_inten
            elif agent_decision == "idle":
                decision = heuristic_decision
                inten = 0.1 * heuristic_inten
            elif heuristic_decision == "idle":
                decision = agent_decision
                inten = 0.9 * agent_intensity
            else:
                decision = agent_decision
                inten = 0.8 * agent_intensity
            
            inten = float(np.clip(inten, 0.0, 1.0))
            
            soc = battery_energy / battery_capacity_mwh if battery_capacity_mwh > 0 else 0.0
            soc = float(np.clip(soc, 0.0, 1.0))
            soc_min_e = batt_soc_min * battery_capacity_mwh
            soc_max_e = batt_soc_max * battery_capacity_mwh
            
            cash_delta = 0.0
            throughput_mwh = 0.0
            new_battery_energy = battery_energy
            invalid_action_penalty = 0.0  # NEW: Track invalid action penalty
            
            if decision == "discharge" and soc > batt_soc_min + 1e-6:
                energy_possible = min(battery_energy - soc_min_e, max_energy_this_step * inten)
                energy_possible = max(0.0, energy_possible)
                delivered_mwh = energy_possible * batt_eta_discharge
                new_battery_energy -= energy_possible
                throughput_mwh += energy_possible
                battery_discharge_power = delivered_mwh / step_h
                cash_delta += delivered_mwh * price
            elif decision == "charge" and soc < batt_soc_max - 1e-6:
                room = soc_max_e - battery_energy
                energy_possible = min(room, max_energy_this_step * inten)
                energy_possible = max(0.0, energy_possible)
                grid_mwh = energy_possible / max(batt_eta_charge, 1e-6)
                new_battery_energy += energy_possible
                throughput_mwh += energy_possible
                battery_discharge_power = 0.0
                cash_delta -= grid_mwh * price
            else:
                battery_discharge_power = 0.0
                # NEW: Add penalty for invalid actions (discharge at min SOC or charge at max SOC)
                if decision == "discharge" and soc <= batt_soc_min + 1e-6:
                    # CRITICAL FIX: Penalty for trying to discharge empty battery
                    # Use degradation cost as reference for penalty magnitude (ensures meaningful signal)
                    invalid_action_penalty = -batt_degradation_cost * max_energy_this_step * 2.0  # 2x degradation cost
                elif decision == "charge" and soc >= batt_soc_max - 1e-6:
                    # Moderate penalty for trying to charge full battery
                    invalid_action_penalty = -batt_degradation_cost * max_energy_this_step * 1.0  # 1x degradation cost
            
            deg_cost = batt_degradation_cost * throughput_mwh
            cash_delta -= deg_cost
            
            # NEW: Apply invalid action penalty to cash_delta
            cash_delta += invalid_action_penalty
            
            # Bounds check
            new_battery_energy = float(np.clip(new_battery_energy, 0.0, battery_capacity_mwh))
            
            return float(cash_delta), {
                'battery_energy': new_battery_energy,
                'battery_discharge_power': battery_discharge_power,
                'invalid_action_penalty': invalid_action_penalty  # NEW: Return penalty for logging
            }
            
        except Exception as e:
            msg = f"[BATTERY_OPS_FATAL] Battery operations failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

