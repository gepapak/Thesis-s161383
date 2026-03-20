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
    def apply_risk_control(
        risk_action: np.ndarray,
        risk_exposure_cap_min: float = 0.25,
        risk_exposure_cap_max: float = 1.0,
    ) -> float:
        """
        Apply risk controller action as an absolute exposure cap.
        
        Args:
            risk_action: Risk controller action array [-1, 1]
            
        Returns:
            Risk exposure cap in [risk_exposure_cap_min, risk_exposure_cap_max]
        """
        try:
            val = float(np.clip(risk_action.reshape(-1)[0], -1.0, 1.0))
            lo = float(min(risk_exposure_cap_min, risk_exposure_cap_max))
            hi = float(max(risk_exposure_cap_min, risk_exposure_cap_max))
            cap = lo + 0.5 * (val + 1.0) * (hi - lo)
            return float(np.clip(cap, lo, hi))
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
        Apply meta control for capital allocation and trading frequency.

        The live Tier-1 rule-based path uses this as a deterministic mapping
        from a normalized action to a sleeve capital target. Trading cadence is
        typically fixed by config, so the action is usually 1D and only
        controls capital allocation.
        
        Args:
            meta_action: Meta controller action array [-1, 1]
            meta_cap_min: Minimum capital allocation fraction
            meta_cap_max: Maximum capital allocation fraction
            meta_freq_min: Minimum trading frequency
            meta_freq_max: Maximum trading frequency
            forecast_confidence: Optional confidence scaling input when enabled
            
        Returns:
            Tuple of (capital_allocation_fraction, investment_freq)
        """
        try:
            meta_action_arr = np.array(meta_action, dtype=np.float32).reshape(-1)
            a0 = float(meta_action_arr[0]) if meta_action_arr.size >= 1 else 0.0
            a1 = float(meta_action_arr[1]) if meta_action_arr.size >= 2 else 0.0
            
            # Generic symmetric mapping function for [-1,1] -> [min,max]
            def map_from_minus1_1(x, lo, hi):
                x = float(np.clip(x, -1.0, 1.0))
                return lo + (x + 1.0) * 0.5 * (hi - lo)
            
            # Apply symmetric mapping to both components
            cap = map_from_minus1_1(a0, meta_cap_min, meta_cap_max)
            
            # Optional confidence scaling for legacy learned-meta variants.
            if disable_confidence_scaling:
                confidence_multiplier = 1.0
            else:
                forecast_confidence = float(np.clip(forecast_confidence, 0.2, 1.0))
                confidence_multiplier = 0.7 + (forecast_confidence - 0.2) / 0.8 * 0.8

            clip_lo = max(0.0, float(meta_cap_min) * 0.7)  # Allow up to 30% reduction
            clip_hi = min(1.0, float(meta_cap_max) * 1.5)  # Allow up to 50% increase but cap at 1.0
            if clip_hi < clip_lo:
                clip_hi = clip_lo
            capital_allocation_fraction = float(np.clip(cap * confidence_multiplier, clip_lo, clip_hi))
            
            if int(meta_freq_min) == int(meta_freq_max) or meta_action_arr.size < 2:
                investment_freq = int(meta_freq_min)
            else:
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
    def decode_battery_action(
        bat_action: np.ndarray,
        battery_capacity_mwh: float,
        battery_energy: float,
        batt_soc_min: float,
        batt_soc_max: float,
        max_energy_this_step: float,
        *,
        action_mode: str = "target_soc",
        action_threshold: float = 0.35,
        discrete_action_levels: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Decode the battery policy action into a physically-meaningful dispatch intent.

        `target_soc` is the clean default contract: the policy chooses desired
        inventory within the valid SOC band, and the environment applies the
        feasible energy move for this timestep.
        """
        raw_arr = np.asarray(bat_action).reshape(-1)
        raw_val = raw_arr[0] if raw_arr.size > 0 else 1
        mode = str(action_mode or "target_soc").strip().lower()

        if mode == "discrete":
            levels = discrete_action_levels or [-1.0, -0.5, 0.0, 0.5, 1.0]
            action_idx = int(np.clip(int(np.round(float(raw_val))), 0, len(levels) - 1))
            u_discrete = float(np.clip(float(levels[action_idx]), -1.0, 1.0))
            soc = float(np.clip(battery_energy / max(battery_capacity_mwh, 1e-6), 0.0, 1.0))
            if battery_capacity_mwh <= 0.0 or max_energy_this_step <= 0.0:
                return {
                    "u_raw": u_discrete,
                    "decision": "idle",
                    "intensity": 0.0,
                    "energy_request_mwh": 0.0,
                    "target_soc": soc,
                    "action_mode": mode,
                    "action_idx": action_idx,
                }
            if abs(u_discrete) <= 1e-9:
                decision = "idle"
                energy_request = 0.0
                intensity = 0.0
            elif u_discrete < 0.0:
                decision = "charge"
                intensity = abs(u_discrete)
                energy_request = float(max_energy_this_step * intensity)
            else:
                decision = "discharge"
                intensity = abs(u_discrete)
                energy_request = float(-max_energy_this_step * intensity)

            # Feasibility projection: invalid discrete actions become no-ops,
            # which is cleaner than teaching physics through penalties.
            if decision == "charge" and soc >= batt_soc_max - 1e-6:
                decision = "idle"
                intensity = 0.0
                energy_request = 0.0
            elif decision == "discharge" and soc <= batt_soc_min + 1e-6:
                decision = "idle"
                intensity = 0.0
                energy_request = 0.0
            return {
                "u_raw": u_discrete,
                "decision": decision,
                "intensity": intensity,
                "energy_request_mwh": energy_request,
                "target_soc": soc,
                "action_mode": mode,
                "action_idx": action_idx,
            }

        u_raw = float(np.clip(np.asarray(bat_action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        if battery_capacity_mwh <= 0.0 or max_energy_this_step <= 0.0:
            return {
                "u_raw": u_raw,
                "decision": "idle",
                "intensity": 0.0,
                "energy_request_mwh": 0.0,
                "target_soc": float(np.clip(batt_soc_min, 0.0, 1.0)),
                "action_mode": mode,
            }

        soc = float(np.clip(battery_energy / battery_capacity_mwh, 0.0, 1.0))
        if mode == "target_soc":
            soc_span = max(batt_soc_max - batt_soc_min, 1e-6)
            target_soc = float(np.clip(
                batt_soc_min + 0.5 * (u_raw + 1.0) * soc_span,
                batt_soc_min,
                batt_soc_max,
            ))
            target_energy = target_soc * battery_capacity_mwh
            energy_request = float(np.clip(target_energy - battery_energy, -max_energy_this_step, max_energy_this_step))
            intensity = float(np.clip(abs(energy_request) / max(max_energy_this_step, 1e-6), 0.0, 1.0))
            if intensity <= 1e-9:
                decision = "idle"
            elif energy_request > 0.0:
                decision = "charge"
            else:
                decision = "discharge"
            return {
                "u_raw": u_raw,
                "decision": decision,
                "intensity": intensity,
                "energy_request_mwh": energy_request,
                "target_soc": target_soc,
                "action_mode": mode,
            }

        active_mag = max(abs(u_raw) - action_threshold, 0.0)
        active_denom = max(1.0 - action_threshold, 1e-6)
        intensity = float(np.clip(active_mag / active_denom, 0.0, 1.0))
        if intensity <= 0.0:
            decision = "idle"
        elif u_raw < 0.0:
            decision = "charge"
        else:
            decision = "discharge"
        energy_request = 0.0
        if decision == "charge":
            energy_request = float(max_energy_this_step * intensity)
        elif decision == "discharge":
            energy_request = float(-max_energy_this_step * intensity)
        return {
            "u_raw": u_raw,
            "decision": decision,
            "intensity": intensity,
            "energy_request_mwh": energy_request,
            "target_soc": soc,
            "action_mode": mode,
        }

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
        config: Optional[EnhancedConfig] = None,
        use_heuristic_dispatch: bool = True,
        action_threshold: float = 0.35,
    ) -> Tuple[float, Dict[str, float]]:
        """
        REFACTORED: Execute battery operations and return cash delta.
        
        Args:
            bat_action: Battery operator action (discrete ladder or continuous value)
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
            bat_arr = np.asarray(bat_action).reshape(-1)
            raw_val = float(bat_arr[0]) if bat_arr.size > 0 else 1.0
            u_raw = float(np.clip(raw_val, -1.0, 1.0))
            
            if battery_capacity_mwh <= 0.0:
                return 0.0, {'battery_energy': 0.0, 'battery_discharge_power': 0.0}
            
            step_h = 10.0 / 60.0
            max_power_mw = battery_capacity_mwh * batt_power_c_rate
            max_energy_this_step = max_power_mw * step_h
            
            if config is not None:
                use_heuristic_dispatch = bool(
                    getattr(config, 'battery_use_heuristic_dispatch', use_heuristic_dispatch)
                )
                action_mode = str(getattr(config, 'battery_action_mode', 'target_soc') or 'target_soc')
                action_threshold = float(
                    np.clip(getattr(config, 'battery_action_threshold', action_threshold), 0.0, 1.0)
                )
            else:
                action_mode = "target_soc"

            # Get heuristic policy if available
            heuristic_decision = "idle"
            heuristic_inten = 0.0
            if use_heuristic_dispatch and battery_dispatch_policy_fn:
                try:
                    heuristic_decision, heuristic_inten = battery_dispatch_policy_fn(timestep)
                except Exception as e:
                    raise RuntimeError(
                        f"[BATTERY_POLICY_FATAL] battery_dispatch_policy_fn failed at timestep {timestep}: {e}"
                    ) from e
            
            decoded = TradingEngine.decode_battery_action(
                bat_action=bat_action,
                battery_capacity_mwh=battery_capacity_mwh,
                battery_energy=battery_energy,
                batt_soc_min=batt_soc_min,
                batt_soc_max=batt_soc_max,
                max_energy_this_step=max_energy_this_step,
                action_mode=action_mode,
                action_threshold=action_threshold,
                discrete_action_levels=getattr(config, 'battery_discrete_action_levels', None) if config is not None else None,
            )
            agent_decision = str(decoded["decision"])
            agent_intensity = float(decoded["intensity"])
            agent_energy_request = float(decoded["energy_request_mwh"])
            agent_target_soc = float(decoded["target_soc"])

            if not use_heuristic_dispatch:
                decision = agent_decision
                inten = agent_intensity
            else:
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
            invalid_action_penalty = 0.0
            
            if not use_heuristic_dispatch and action_mode == "target_soc":
                requested = float(agent_energy_request)
                if requested < -1e-9:
                    energy_possible = min(max(-requested, 0.0), battery_energy - soc_min_e)
                    energy_possible = max(0.0, energy_possible)
                    delivered_mwh = energy_possible * batt_eta_discharge
                    new_battery_energy -= energy_possible
                    throughput_mwh += energy_possible
                    battery_discharge_power = delivered_mwh / step_h
                    cash_delta += delivered_mwh * price
                    decision = "discharge" if energy_possible > 0.0 else "idle"
                    inten = float(np.clip(energy_possible / max(max_energy_this_step, 1e-6), 0.0, 1.0))
                elif requested > 1e-9:
                    energy_possible = min(max(requested, 0.0), soc_max_e - battery_energy)
                    energy_possible = max(0.0, energy_possible)
                    grid_mwh = energy_possible / max(batt_eta_charge, 1e-6)
                    new_battery_energy += energy_possible
                    throughput_mwh += energy_possible
                    battery_discharge_power = 0.0
                    cash_delta -= grid_mwh * price
                    decision = "charge" if energy_possible > 0.0 else "idle"
                    inten = float(np.clip(energy_possible / max(max_energy_this_step, 1e-6), 0.0, 1.0))
                else:
                    battery_discharge_power = 0.0
                    decision = "idle"
                    inten = 0.0
            elif decision == "discharge" and soc > batt_soc_min + 1e-6:
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
            
            deg_cost = batt_degradation_cost * throughput_mwh
            cash_delta -= deg_cost
            
            # Bounds check
            new_battery_energy = float(np.clip(new_battery_energy, 0.0, battery_capacity_mwh))
            
            return float(cash_delta), {
                'battery_energy': new_battery_energy,
                'battery_discharge_power': battery_discharge_power,
                'invalid_action_penalty': invalid_action_penalty,
                'battery_target_soc': agent_target_soc,
            }
            
        except Exception as e:
            msg = f"[BATTERY_OPS_FATAL] Battery operations failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

