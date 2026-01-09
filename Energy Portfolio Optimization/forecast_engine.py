from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils import (
    convert_mape_to_price_relative,
    compute_forecast_direction,
    exposure_with_cap,
    normalize_position,
    SafeDivision,
    safe_clip,
)
from config import EnhancedConfig
from logger import get_logger

# Centralized logging - ALL logging goes through logger.py
logger = get_logger(__name__)


@dataclass
class ForecastComputationResult:
    forecast_signal_score: float
    forecast_gate_passed: bool
    forecast_used_flag: bool
    forecast_usage_reason: str
    debug_payload: Dict[str, Any]


@dataclass
class ForecastObservationFeatures:
    """Forecast features to add to observations."""
    investor_features: Optional[np.ndarray] = None  # 8D (Tier 22): z_short, z_medium_lagged, direction, momentum, strength, forecast_trust, normalized_error, trade_signal
    battery_features: Optional[np.ndarray] = None   # 6D: z_short_wind, z_short_solar, z_short_hydro, z_short_price, z_medium_price, z_long_price
    risk_features: Optional[np.ndarray] = None     # 3D: z_short, vol_forecast, trust
    meta_features: Optional[np.ndarray] = None      # 2D: trust, expected_return


class ForecastEngine:
    """
    REFACTORED: Complete forecast integration engine (merged with adapter).
    
    Encapsulates ALL forecast-related logic, making it a truly optional add-on.
    When enabled: computes z-scores, builds observations, computes rewards
    When disabled: returns zero/empty values (Tier 1 mode)
    """

    def __init__(
        self,
        env: Optional[Any] = None,
        config: Optional[EnhancedConfig] = None,
        forecast_generator: Optional[Any] = None
    ):
        """
        Initialize forecast engine.
        
        Args:
            env: Environment reference (for backward compatibility)
            config: Configuration object
            forecast_generator: Optional forecast generator
        """
        self.env = env
        self.config = config if config is not None else (env.config if env else None)
        self.forecast_generator = forecast_generator if forecast_generator is not None else (env.forecast_generator if env else None)
        
        # Check if forecasts are enabled
        self.enabled = (
            getattr(self.config, 'enable_forecast_utilisation', False) 
            and self.forecast_generator is not None
        ) if self.config else False
        
        # Forecast state tracking
        self.z_short_price = 0.0
        self.z_medium_price = 0.0
        self.z_long_price = 0.0
        self.z_short_wind = 0.0
        self.z_short_solar = 0.0
        self.z_short_hydro = 0.0
        self.forecast_trust = 0.5
        self._horizon_correlations = {'short': 0.0, 'medium': 0.0, 'long': 0.0}
        self._z_score_history = {}
        self._forecast_deltas_raw = {}
        
        if self.enabled:
            logger.info("[FORECAST_ENGINE] Forecast integration ENABLED")
        else:
            logger.info("[FORECAST_ENGINE] Forecast integration DISABLED (Tier 1 mode)")
    
    def is_enabled(self) -> bool:
        """Check if forecast integration is enabled."""
        return self.enabled

    def compute(
        self,
        timestep: int,
        mtm_pnl: float,
        price_returns: Dict[str, Optional[float]],
    ) -> ForecastComputationResult:
        """
        Compute forecast-based reward contribution.
        
        Returns ForecastComputationResult with reward signal.
        Returns zero result if forecasts disabled.
        """
        if not self.enabled:
            return ForecastComputationResult(0.0, False, False, 'no_forecast', {})
        
        env = self.env
        config = self.config
        
        if env is None or config is None:
            return ForecastComputationResult(0.0, False, False, 'no_env', {})

        forecast_gate_passed = False
        forecast_used_flag = False
        forecast_usage_reason = 'no_forecast'

        horizon_short = config.forecast_horizons.get('short', 6)
        horizon_medium = config.forecast_horizons.get('medium', 24)
        horizon_long = config.forecast_horizons.get('long', 144)

        t_short = timestep - horizon_short if timestep >= horizon_short else None
        t_medium = timestep - horizon_medium if timestep >= horizon_medium else None
        t_long = timestep - horizon_long if timestep >= horizon_long else None

        # Use internal z-score state if available, otherwise fall back to env
        if t_short is not None and hasattr(env, '_z_score_history') and t_short in env._z_score_history:
            z_short = float(env._z_score_history[t_short]['z_short'])
        else:
            z_short = float(self.z_short_price if self.z_short_price != 0.0 else getattr(env, 'z_short_price', 0.0))

        if t_medium is not None and hasattr(env, '_z_score_history') and t_medium in env._z_score_history:
            z_medium = float(env._z_score_history[t_medium]['z_medium'])
        else:
            z_medium = float(self.z_medium_price if self.z_medium_price != 0.0 else getattr(env, 'z_medium_price', 0.0))

        if t_long is not None and hasattr(env, '_z_score_history') and t_long in env._z_score_history:
            z_long = float(env._z_score_history[t_long]['z_long'])
        else:
            z_long = float(self.z_long_price if self.z_long_price != 0.0 else getattr(env, 'z_long_price', 0.0))

        if timestep % 1000 == 0 or timestep == 0:
            logger.info(
                "[HORIZON_MATCHING] t=%s | Using z-scores from: short=%s, medium=%s, long=%s",
                timestep,
                t_short if t_short is not None else 'current',
                t_medium if t_medium is not None else 'current',
                t_long if t_long is not None else 'current',
            )

        # Use internal correlations if available, otherwise fall back to env
        if hasattr(env, '_horizon_correlations'):
            corr_short = env._horizon_correlations.get('short', self._horizon_correlations.get('short', 0.0))
            corr_medium = env._horizon_correlations.get('medium', self._horizon_correlations.get('medium', 0.0))
            corr_long = env._horizon_correlations.get('long', self._horizon_correlations.get('long', 0.0))
        else:
            corr_short = self._horizon_correlations.get('short', 0.0)
            corr_medium = self._horizon_correlations.get('medium', 0.0)
            corr_long = self._horizon_correlations.get('long', 0.0)

        # FIX: Check if correlations are actually computed (non-zero)
        # If correlations are 0.0, they're either not yet computed or truly zero
        # In both cases, use default weights to ensure z_combined is computed
        correlations_computed = (corr_short != 0.0 or corr_medium != 0.0 or corr_long != 0.0)
        
        use_short = corr_short > 0.0
        use_medium = corr_medium > 0.0
        use_long = corr_long > 0.0

        if not correlations_computed:
            # No correlations computed yet - use default weights (warmup period)
            weight_short = 0.7
            weight_medium = 0.2
            weight_long = 0.1
        elif not (use_short or use_medium or use_long):
            # Correlations computed but all negative/zero - use default weights as fallback
            # This ensures z_combined is still computed even with poor correlations
            weight_short = 0.7
            weight_medium = 0.2
            weight_long = 0.1
        else:
            # Use correlation-based weighting (only positive correlations)
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
                # Fallback to equal weights if correlations too small
                num_horizons = sum([use_short, use_medium, use_long])
                if num_horizons > 0:
                    weight_short = (1.0 / num_horizons) if use_short else 0.0
                    weight_medium = (1.0 / num_horizons) if use_medium else 0.0
                    weight_long = (1.0 / num_horizons) if use_long else 0.0
                else:
                    # Last resort: use default weights
                    weight_short = 0.7
                    weight_medium = 0.2
                    weight_long = 0.1

        z_combined = float(weight_short * z_short + weight_medium * z_medium + weight_long * z_long)
        
        # CRITICAL FIX: Remove z_combined normalization by correlation strength
        # This normalization was distorting forecast signals, making them harder to learn from
        # Forecast features should be passed directly to observations without distortion
        # The GNN encoder can learn to use forecasts based on their correlation with PnL rewards
        # Simple weighted combination preserves forecast signal quality
        z_combined = float(np.clip(z_combined, -2.0, 2.0))  # Clip to reasonable range only
        
        if env is not None:
            env.z_combined = z_combined

        if timestep % 1000 == 0 or timestep == 0:
            logger.info(
                "[CORRELATION_BASED_WEIGHTS] t=%s | Correlations: short=%.4f (use=%s), medium=%.4f (use=%s), long=%.4f (use=%s) | "
                "Weights: short=%.3f, medium=%.3f, long=%.3f | z-values: short=%.4f, medium=%.4f, long=%.4f | z_combined=%.4f",
                timestep,
                corr_short,
                use_short,
                corr_medium,
                use_medium,
                corr_long,
                use_long,
                weight_short,
                weight_medium,
                weight_long,
                z_short,
                z_medium,
                z_long,
                z_combined,
            )

        max_pos = config.max_position_size * env.init_budget * config.capital_allocation_fraction if env.init_budget > 0 else 1.0
        wind_pos_value = env.financial_positions.get('wind_instrument_value', 0.0)
        solar_pos_value = env.financial_positions.get('solar_instrument_value', 0.0)
        hydro_pos_value = env.financial_positions.get('hydro_instrument_value', 0.0)

        wind_pos_norm = float(wind_pos_value / max(max_pos, 1.0))
        solar_pos_norm = float(solar_pos_value / max(max_pos, 1.0))
        hydro_pos_norm = float(hydro_pos_value / max(max_pos, 1.0))

        position_signed = float(wind_pos_norm + solar_pos_norm + hydro_pos_norm)
        raw_position_exposure = float(abs(wind_pos_norm) + abs(solar_pos_norm) + abs(hydro_pos_norm))
        position_exposure = float(np.clip(raw_position_exposure, 0.0, 0.3))

        current_price = float(env._price_raw[timestep]) if env and timestep < len(env._price_raw) else 250.0

        forecast_price_short = current_price
        forecast_price_medium = current_price
        forecast_price_long = current_price
        forecast_error_short = 0.0
        forecast_error_medium = 0.0
        forecast_error_long = 0.0
        if self.forecast_generator:
            try:
                fcast = self.forecast_generator.predict_all_horizons(timestep=timestep)
                if fcast:
                    forecast_price_short = fcast.get('price_forecast_short', current_price)
                    forecast_price_medium = fcast.get('price_forecast_medium', current_price)
                    forecast_price_long = fcast.get('price_forecast_long', current_price)
                    if timestep % 500 == 0:
                        logger.info(
                            "[FORECAST_PRICES] t=%s current=%.2f short=%.2f med=%.2f long=%.2f",
                            timestep,
                            current_price,
                            forecast_price_short,
                            forecast_price_medium,
                            forecast_price_long,
                        )
            except Exception as err:
                if timestep % 500 == 0:
                    logger.warning("[FORECAST_PRICES] ERROR at t=%s: %s", timestep, err)

        denom_price = max(abs(current_price), 1.0)
        forecast_error_short = (forecast_price_short - current_price) / denom_price
        forecast_error_medium = (forecast_price_medium - current_price) / denom_price
        forecast_error_long = (forecast_price_long - current_price) / denom_price
        unrealized_pnl_total = 0.0
        if env and hasattr(env, '_open_positions'):
            for asset in ['wind', 'solar', 'hydro']:
                pos = env._open_positions[asset]
                if abs(pos['notional']) > 1e-6 and abs(pos['entry_price']) > 1e-6:
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['notional']
                    unrealized_pnl_total += unrealized_pnl

        unrealized_pnl_norm = float(np.clip(unrealized_pnl_total / max(max_pos, 1.0), -3.0, 3.0))

        current_positions = {
            'wind': float(getattr(env, '_open_positions', {}).get('wind', {}).get('notional', 0.0)) if env else 0.0,
            'solar': float(getattr(env, '_open_positions', {}).get('solar', {}).get('notional', 0.0)) if env else 0.0,
            'hydro': float(getattr(env, '_open_positions', {}).get('hydro', {}).get('notional', 0.0)) if env else 0.0,
        }
        total_position = sum(current_positions.values())
        position_normalized = normalize_position(total_position, max_pos)
        position_exposure_norm = exposure_with_cap(total_position, max_pos, cap=0.3)

        forecast_direction = compute_forecast_direction(z_short, z_medium, z_long)
        forecast_direction_normalized = np.clip(forecast_direction, -1.0, 1.0)
        agent_followed_forecast = (
            forecast_direction_normalized * np.sign(position_normalized) > 0
        ) if abs(forecast_direction_normalized) > 0.01 else False

        forecast_deltas = getattr(env, '_forecast_deltas_raw', None) if env else None
        mape_thresholds_capacity = getattr(env, '_mape_thresholds', None) if env else None
        forecast_trust = float(self.forecast_trust if self.forecast_trust != 0.5 else (getattr(env, '_forecast_trust', 0.5) if env else 0.5))

        def _recent_mape(h, n=10):
            try:
                seq = list(getattr(env, '_horizon_mape', {}).get(h, []))
                if len(seq) > 0:
                    return float(np.mean(seq[-n:]))
            except Exception:
                pass
            return float('nan')

        investor_strategy = None
        quality_mult = 1.0
        if forecast_deltas is not None and mape_thresholds_capacity is not None:
            mape_measured_capacity = {
                'short': _recent_mape('short'),
                'medium': _recent_mape('medium'),
                'long': _recent_mape('long'),
            }
            for h in ['short', 'medium', 'long']:
                if math.isnan(mape_measured_capacity[h]):
                    fallback = mape_thresholds_capacity.get(h, 0.02)
                    mape_measured_capacity[h] = float(fallback)

            mape_measured = convert_mape_to_price_relative(
                mape_measured_capacity,
                current_price,
                getattr(config, 'forecast_price_capacity', 6982.0),
                getattr(config, 'minimum_price_floor', 50.0),
            )
            if mape_measured is None:
                mape_measured = convert_mape_to_price_relative(
                    mape_thresholds_capacity,
                    current_price,
                    getattr(config, 'forecast_price_capacity', 6982.0),
                    getattr(config, 'minimum_price_floor', 50.0),
                )

            investor_strategy = env.forecast_risk_manager.compute_investor_strategy(
                forecast_deltas=forecast_deltas,
                mape_thresholds=mape_measured,
                current_positions=current_positions,
                forecast_trust=forecast_trust,
            ) if env and env.forecast_risk_manager is not None else None
            if env:
                env._current_investor_strategy = investor_strategy
            if investor_strategy is not None:
                forecast_gate_passed = forecast_gate_passed or bool(
                    investor_strategy.get('trade_signal', False)
                )

        risk_adj = env.forecast_risk_manager.compute_risk_adjustments(
            z_short=z_short,
            z_medium=z_medium,
            z_long=z_long,
            forecast_trust=forecast_trust,
            position_pnl=unrealized_pnl_total,
            timestep=timestep,
            forecast_deltas=forecast_deltas,
            mape_thresholds=convert_mape_to_price_relative(
                getattr(env, '_mape_thresholds', None) if env else None,
                current_price,
                getattr(config, 'forecast_price_capacity', 6982.0),
                getattr(config, 'minimum_price_floor', 50.0),
            ),
        ) if env and env.forecast_risk_manager is not None else None

        if risk_adj is not None:
            forecast_gate_passed = forecast_gate_passed or bool(risk_adj.get('trade_signal', False))

        forecast_alignment_reward = 0.0
        alignment_score = 0.0
        misalignment_penalty_total = 0.0

        measured_mape_short = _recent_mape('short', n=10)
        measured_mape_medium = _recent_mape('medium', n=10)
        measured_mape_long = _recent_mape('long', n=10)
        mape_thresholds = getattr(env, '_mape_thresholds', {}) or {}
        mape_threshold_short = mape_thresholds.get('short', 0.02)
        mape_threshold_medium = mape_thresholds.get('medium', 0.02)
        mape_threshold_long = mape_thresholds.get('long', 0.02)
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        measured_available = [
            not math.isnan(x)
            for x in (measured_mape_short, measured_mape_medium, measured_mape_long)
        ]
        if all(measured_available):
            avg_mape = (
                weights['short'] * measured_mape_short +
                weights['medium'] * measured_mape_medium +
                weights['long'] * measured_mape_long
            )
        else:
            ms = measured_mape_short if not math.isnan(measured_mape_short) else mape_threshold_short
            mm = measured_mape_medium if not math.isnan(measured_mape_medium) else mape_threshold_medium
            ml = measured_mape_long if not math.isnan(measured_mape_long) else mape_threshold_long
            avg_mape = weights['short'] * ms + weights['medium'] * mm + weights['long'] * ml

        if avg_mape < 0.03:
            quality_mult = 2.0
        elif avg_mape < 0.08:
            quality_mult = 1.0
        else:
            quality_mult = 0.3

        # PHASE 1 FIX: Compute realized_vs_forecast early (needed for error penalties in alignment reward)
        price_return_short = price_returns.get('short') if price_returns else None
        price_return_medium = price_returns.get('medium') if price_returns else None
        price_return_long = price_returns.get('long') if price_returns else None
        price_return_forecast = price_returns.get('forecast') if price_returns else 0.0
        one_step_return = price_returns.get('one_step') if price_returns else 0.0

        if price_return_short is None:
            price_return_short = one_step_return
        if price_return_medium is None:
            price_return_medium = price_return_short
        if price_return_long is None:
            price_return_long = price_return_medium

        realized_vs_forecast = float(one_step_return - (price_return_forecast or 0.0))

        # PHASE 1 FIX: Strengthen forecast confidence gating
        # Only use forecasts when they're highly reliable (addresses reward mismatch)
        forecast_confidence = float(np.clip(1.0 / (1.0 + avg_mape), 0.0, 1.0))
        
        # CRITICAL: Forecast rewards are blocked for fair comparison (observations only)
        # Tier 2: Forecast observations only (no rewards) - agents learn from PnL correlation
        # Tier 3: Forecast observations + FAMC (no forecast rewards) - FAMC handles variance reduction
        # The difference is FAMC (variance reduction), not forecast rewards!
        # Forecast features in observations alone should enable better decision-making through:
        # 1. Predictive information (z-scores indicate future price movements)
        # 2. Implicit learning (agents learn forecast→action mappings from PnL correlation)
        # 3. GNN encoder learning (cross-attention fusion enables forecast→base interactions)
        explicit_gate_passed = False  # Always block for BOTH tiers (fair comparison)
        
        # OR logic: Gate passes if risk manager approves (but no forecast rewards)
        forecast_gate_passed = forecast_gate_passed or explicit_gate_passed

        if forecast_gate_passed:
            alignment_value = position_normalized * forecast_direction_normalized * position_exposure
            alignment_score = float(np.clip(alignment_value, -1.0, 1.0))
            # FIX: Actually use config parameter instead of hardcoded 5.0
            alignment_mult = getattr(config, 'forecast_alignment_multiplier', 10.0)
            # FIX: Moderate cap increase from 5.0 to 10.0 (2x instead of 3x) for balanced learning
            # This provides stronger signal without destabilizing reward distribution
            forecast_alignment_reward = np.clip(alignment_mult * alignment_score * quality_mult, -10.0, 10.0)
            
            # PHASE 1 FIX: Strengthen profitability gate - more aggressive penalties for losses
            # This directly addresses the reward structure mismatch issue
            if mtm_pnl < 0:
                # Progressive penalty: -$1K = 10% reduction, -$5K = 50%, -$10K = 100% (no reward)
                loss_penalty_mult = getattr(config, 'loss_penalty_multiplier', 2.0)  # More aggressive
                loss_penalty = min(1.0, abs(mtm_pnl) / (10000.0 / loss_penalty_mult))  # Scale: $10K loss = 100% penalty
                forecast_alignment_reward *= max(0.0, 1.0 - loss_penalty)
            else:
                # PHASE 1 FIX: Bonus for profitable positions - reward when forecasts lead to profits
                profit_bonus_mult = getattr(config, 'profitability_bonus_multiplier', 1.5)
                profit_bonus = min(1.0, mtm_pnl / 10000.0)  # Scale: $10K profit = 100% bonus
                forecast_alignment_reward *= (1.0 + profit_bonus * profit_bonus_mult * 0.5)  # Up to 50% bonus
            
            # PHASE 1 FIX: Add forecast error penalty - penalize when forecasts are wrong
            # This addresses the credit assignment problem
            forecast_error_penalty_scale = getattr(config, 'forecast_error_penalty_scale', 0.5)
            if abs(realized_vs_forecast) > 0.01:  # Forecast error > 1%
                # Penalize proportionally to forecast error magnitude
                error_penalty = -forecast_error_penalty_scale * abs(realized_vs_forecast) * position_exposure * 10.0
                forecast_alignment_reward += error_penalty
            
            # Weight alignment reward by forecast accuracy (only reward when forecasts are reliable)
            forecast_alignment_reward *= (forecast_trust * forecast_confidence)

            signal_strength = float(investor_strategy.get('signal_strength', 0.5)) if investor_strategy else 0.0
            trade_signal_active = bool(investor_strategy and investor_strategy.get('trade_signal', False))
            hedge_signal_active = bool(investor_strategy and investor_strategy.get('hedge_signal', False))
            direction_conflict = trade_signal_active and position_exposure >= getattr(
                config, 'forecast_direction_conflict_exposure', 0.12
            ) and not agent_followed_forecast

            if trade_signal_active and agent_followed_forecast:
                follow_scale = float(np.clip(position_exposure / getattr(config, 'forecast_follow_target_exposure', 0.12), 0.0, 1.0))
                min_scale = getattr(config, 'forecast_low_exposure_threshold', 0.05) / max(
                    getattr(config, 'forecast_follow_target_exposure', 0.12), 1e-6
                )
                # FIX: Moderate increase from 0.35 to 0.5 to reward following forecasts
                # This encourages forecast usage without over-rewarding
                follow_bonus = 0.5 * signal_strength * quality_mult * max(follow_scale, min_scale)
                forecast_alignment_reward += follow_bonus

            if direction_conflict:
                penalty_scale = float(np.clip((position_exposure - getattr(config, 'forecast_direction_conflict_exposure', 0.12)) / 0.3, 0.0, 1.0))
                # FIX: Increased direction conflict penalty from 0.35 to 1.0 to strongly discourage misalignment
                # This helps agent learn to align position direction with forecast direction
                penalty = getattr(config, 'forecast_direction_penalty_scale', 1.0) * signal_strength * quality_mult * penalty_scale
                forecast_alignment_reward -= penalty
                misalignment_penalty_total += penalty
            elif trade_signal_active and position_exposure < getattr(config, 'forecast_low_exposure_threshold', 0.05):
                entry_gap = float(np.clip(1.0 - (position_exposure / getattr(config, 'forecast_low_exposure_threshold', 0.05)), 0.0, 1.0))
                penalty = getattr(config, 'forecast_entry_penalty_scale', 0.12) * signal_strength * quality_mult * entry_gap
                forecast_alignment_reward -= penalty
                misalignment_penalty_total += penalty

            if not trade_signal_active and not hedge_signal_active:
                neutral_penalty = np.clip(position_exposure - 0.1, 0.0, 1.0)
                neutral_penalty_value = getattr(config, 'forecast_neutral_penalty_scale', 1.2) * neutral_penalty * quality_mult
                forecast_alignment_reward -= neutral_penalty_value
                misalignment_penalty_total += neutral_penalty_value
        else:
            forecast_alignment_reward = 0.0

        if timestep > 0 and timestep < len(env._price_raw):
            prev_price = float(np.clip(env._price_raw[timestep-1], 50.0, 1e9))
            current_price = float(np.clip(env._price_raw[timestep], 50.0, 1e9))
            price_return_raw = (current_price - prev_price) / max(abs(prev_price), 1e-6)
            price_return = np.clip(price_return_raw, -0.1, 0.1)
        else:
            price_return = 0.0

        prev_position = getattr(env, '_prev_total_position', 0.0) if env else 0.0
        prev_position_normalized = normalize_position(prev_position, max_pos)
        pnl_scale = getattr(config, 'forecast_pnl_reward_scale', 500.0)
        pnl_reward = float(np.clip(mtm_pnl / max(pnl_scale, 1.0), -20.0, 20.0))
        if env:
            env._prev_total_position = total_position

        # PHASE 1 FIX: Initialize forecast error penalty (used in signal score)
        # Note: realized_vs_forecast is already computed above
        forecast_error_penalty = 0.0
        if abs(realized_vs_forecast) > 0.01 and position_exposure > 0:  # Forecast error > 1% and position exists
            error_penalty_lambda = getattr(config, 'forecast_error_penalty_lambda', 0.3)
            forecast_error_penalty = -error_penalty_lambda * abs(realized_vs_forecast) * position_exposure * 20.0

        risk_reward = risk_adj.get('risk_reward', 0.0) if risk_adj else 0.0
        extreme_delta_penalty = 0.0
        if forecast_deltas is not None and position_exposure > 0:
            delta_short_raw = abs(forecast_deltas.get('short', 0.0))
            delta_medium_raw = abs(forecast_deltas.get('medium', 0.0))
            delta_long_raw = abs(forecast_deltas.get('long', 0.0))
            # FIX: Only penalize truly extreme forecasts (2.0 z-score = ~2 standard deviations)
            # 0.5 z-score is only ~0.5 standard deviations, which is normal variation
            if delta_short_raw > 2.0 or delta_medium_raw > 2.0 or delta_long_raw > 2.0:
                extreme_delta_penalty = -50.0 * position_exposure  # Less aggressive penalty

        # PHASE 1 FIX: Include forecast error penalty in signal score
        forecast_signal_score = (
            0.60 * forecast_alignment_reward +
            0.30 * pnl_reward +
            0.10 * risk_reward +
            extreme_delta_penalty +
            forecast_error_penalty  # NEW: Penalize forecast errors
        )

        usage_exposure_threshold = getattr(config, 'forecast_usage_exposure_threshold', 0.02)
        forecast_used_flag = bool(
            forecast_gate_passed and agent_followed_forecast and position_exposure > usage_exposure_threshold
        )
        if forecast_gate_passed:
            if forecast_used_flag:
                forecast_usage_reason = 'used'
            elif position_exposure <= usage_exposure_threshold:
                forecast_usage_reason = 'low_exposure'
            elif not agent_followed_forecast:
                forecast_usage_reason = 'direction_mismatch'
            else:
                forecast_usage_reason = 'risk_blocked'
        else:
            forecast_usage_reason = 'gate_blocked'

        if timestep % 500 == 0:
            logger.info(
                "[FORECAST_REWARD] t=%s total=%.2f align=%.2f pnl=%.2f",
                timestep,
                forecast_signal_score,
                forecast_alignment_reward,
                pnl_reward,
            )

        direction_accuracy_short = self._direction_accuracy(z_short, price_return_short)
        direction_accuracy_medium = self._direction_accuracy(z_medium, price_return_medium)
        direction_accuracy_long = self._direction_accuracy(z_long, price_return_long)
        combined_conf_debug = float(getattr(env, '_combined_confidence', 0.5) if env else 0.5)
        reward_weights = getattr(env, 'reward_weights', {}) if env else {}
        adaptive_multiplier_debug = float(
            getattr(env, '_adaptive_forecast_weight', reward_weights.get('forecast', 0.0)) if env else 0.0
        )
        warmup_factor_debug = float(getattr(env, '_forecast_warmup_factor', 1.0) if env else 1.0)

        # FIX #2/#3: Use quality-weighted signal and update dynamic trust
        try:
            mape_short_val = float(measured_mape_short) if np.isfinite(measured_mape_short) else float(mape_threshold_short)
            mape_medium_val = float(measured_mape_medium) if np.isfinite(measured_mape_medium) else float(mape_threshold_medium)
            mape_long_val = float(measured_mape_long) if np.isfinite(measured_mape_long) else float(mape_threshold_long)

            quality_signal = self.compute_quality_weighted_signal(
                float(z_short), float(z_medium), float(z_long),
                mape_short_val, mape_medium_val, mape_long_val,
                confidence_threshold=0.5
            )
            # Override prior forecast_signal_score with quality-weighted version
            forecast_signal_score = quality_signal
            # Update forecast trust dynamically based on MAPE
            self.update_forecast_trust_from_mape(mape_short_val, mape_medium_val, mape_long_val)
            forecast_trust = float(self.forecast_trust)
        except Exception as _e:
            # Keep existing values if quality weighting fails
            pass

        price_return_1step = one_step_return or 0.0
        # CRITICAL FIX: Use normalized position_signed instead of raw total_position
        # total_position is the raw value in DKK, but position_signed is normalized (should be ~-1 to 1)
        # This ensures consistent logging between Tier 1 and Tier 2
        debug_payload = {
            'position_signed': position_signed,  # FIXED: Use normalized value, not raw total_position
            'position_exposure': position_exposure_norm,
            'price_return_1step': price_return_1step,
            'price_return_forecast': price_return_forecast or 0.0,
            'forecast_signal_score': forecast_signal_score,
            'wind_pos_norm': wind_pos_norm,  # FIXED: Use already-calculated normalized values
            'solar_pos_norm': solar_pos_norm,  # FIXED: Use already-calculated normalized values
            'hydro_pos_norm': hydro_pos_norm,  # FIXED: Use already-calculated normalized values
            'z_short': z_short,
            'z_medium': z_medium,
            'z_long': z_long,
            'z_combined': z_combined,
            'forecast_mape': float(avg_mape),
            'forecast_confidence': float(np.clip(1.0 / (1.0 + avg_mape), 0.0, 1.0)),
            'forecast_trust': forecast_trust,
            'trust_scale': forecast_trust,
            'forecast_gate_passed': forecast_gate_passed,
            'forecast_used': forecast_used_flag,
            'forecast_not_used_reason': forecast_usage_reason,
            'forecast_usage_bonus': 0.0,
            'investor_strategy_multiplier': float(getattr(env, '_last_investor_strategy_multiplier', 1.0)),
            'alignment_reward': forecast_alignment_reward,
            'pnl_reward': pnl_reward,
            'base_alignment': alignment_score,
            'profitability_factor': pnl_reward,
            'alignment_multiplier': quality_mult,
            'misalignment_penalty_mult': misalignment_penalty_total,
            'risk_reward': risk_reward,
            'exploration_bonus': 0.0,
            'position_size_bonus': 0.0,
            'signal_gate_multiplier': float(
                risk_adj.get('signal_gate_multiplier', getattr(env.forecast_risk_manager, '_last_gate_multiplier', 0.0)) if env and env.forecast_risk_manager else 0.0
            ) if risk_adj else float(getattr(env.forecast_risk_manager, '_last_gate_multiplier', 0.0) if env and env.forecast_risk_manager else 0.0),
            'adaptive_multiplier': adaptive_multiplier_debug,
            'current_price_dkk': current_price,
            'forecast_price_short_dkk': forecast_price_short,
            'forecast_price_medium_dkk': forecast_price_medium,
            'forecast_price_long_dkk': forecast_price_long,
            'forecast_error_short_pct': forecast_error_short,
            'forecast_error_medium_pct': forecast_error_medium,
            'forecast_error_long_pct': forecast_error_long,
            'forecast_error': realized_vs_forecast,
            'realized_vs_forecast': realized_vs_forecast,
            'mape_short': float(mape_short_val) if 'mape_short_val' in locals() else float(measured_mape_short) if not math.isnan(measured_mape_short) else float(mape_threshold_short),
            'mape_medium': float(mape_medium_val) if 'mape_medium_val' in locals() else float(measured_mape_medium) if not math.isnan(measured_mape_medium) else float(mape_threshold_medium),
            'mape_long': float(mape_long_val) if 'mape_long_val' in locals() else float(measured_mape_long) if not math.isnan(measured_mape_long) else float(mape_threshold_long),
            'forecast_direction': forecast_direction,
            'position_direction': float(np.sign(position_normalized)),
            'agent_followed_forecast': bool(agent_followed_forecast),
            'combined_confidence': combined_conf_debug,
            'combined_forecast_score': forecast_signal_score,
            'warmup_factor': warmup_factor_debug,
            'direction_accuracy_short': direction_accuracy_short,
            'direction_accuracy_medium': direction_accuracy_medium,
            'direction_accuracy_long': direction_accuracy_long,
        }

        if env:
            env._debug_forecast_reward = debug_payload
            env._last_forecast_risk_adj = risk_adj
            env._last_pnl_reward = pnl_reward
            env._last_forecast_alignment_reward = forecast_alignment_reward
            env._last_risk_reward = risk_reward

        return ForecastComputationResult(
            forecast_signal_score=forecast_signal_score,
            forecast_gate_passed=forecast_gate_passed,
            forecast_used_flag=forecast_used_flag,
            forecast_usage_reason=forecast_usage_reason,
            debug_payload=debug_payload,
        )

    def compute_quality_weighted_signal(self, z_short: float, z_medium: float, z_long: float,
                                       mape_short: float, mape_medium: float, mape_long: float,
                                       confidence_threshold: float = 0.5) -> float:
        """
        FIX #2: Compute quality-weighted forecast signal with MAPE-based gating.
        
        Filters raw z-scores through forecast accuracy gates:
        - Short horizon MAPE threshold: 8% (highest confidence)
        - Medium horizon MAPE threshold: 12% (medium confidence)
        - Long horizon MAPE threshold: 20% (lower confidence)
        
        Gates:
        1. MAPE-based quality filtering (prevent trading on poor forecasts)
        2. Confidence weighting (scale signal by accuracy relative to threshold)
        3. Horizon weighting (prioritize accurate short horizon)
        
        Args:
            z_short, z_medium, z_long: Raw z-scores from forecast
            mape_short, mape_medium, mape_long: MAPE accuracy for each horizon
            confidence_threshold: Min confidence required to use signal
            
        Returns:
            Filtered signal in [-1, 1] range, or 0.0 if below confidence threshold
        """
        try:
            # MAPE thresholds (FIX #2 critical parameters)
            mape_thresh_short = 0.08   # 8% - high confidence
            mape_thresh_medium = 0.12  # 12% - medium confidence
            mape_thresh_long = 0.20    # 20% - lower confidence
            
            # Gate 1: Check if horizons meet MAPE thresholds
            short_valid = mape_short < mape_thresh_short
            medium_valid = mape_medium < mape_thresh_medium
            long_valid = mape_long < mape_thresh_long
            
            # Gate 2: Compute confidence weights (1.0 - normalized MAPE)
            # Maps MAPE to confidence: 0% MAPE → 1.0 confidence, at threshold → ~0.2 confidence
            conf_short = max(0.0, (mape_thresh_short - mape_short) / mape_thresh_short) if short_valid else 0.0
            conf_medium = max(0.0, (mape_thresh_medium - mape_medium) / mape_thresh_medium) if medium_valid else 0.0
            conf_long = max(0.0, (mape_thresh_long - mape_long) / mape_thresh_long) if long_valid else 0.0
            
            # Gate 3: Horizon weights (short most reliable, long least)
            weight_short = 0.70   # 70% short horizon
            weight_medium = 0.20  # 20% medium horizon
            weight_long = 0.10    # 10% long horizon (use sparingly)
            
            # Compute weighted signal with confidence multipliers
            weighted_signal = (
                weight_short * z_short * conf_short +
                weight_medium * z_medium * conf_medium +
                weight_long * z_long * conf_long
            )
            
            # Compute overall confidence (weighted average of horizon confidences)
            overall_confidence = (
                weight_short * conf_short +
                weight_medium * conf_medium +
                weight_long * conf_long
            )
            
            # Gate 4: Check if overall confidence meets threshold
            if overall_confidence < confidence_threshold:
                return 0.0  # Return neutral signal if confidence too low
            
            # Clip to [-1, 1] range
            quality_weighted_signal = float(np.clip(weighted_signal, -1.0, 1.0))
            
            # Store for debugging
            if self.env is not None:
                if not hasattr(self.env, '_quality_signal_debug'):
                    self.env._quality_signal_debug = {}
                self.env._quality_signal_debug = {
                    'raw_signal': (z_short + z_medium + z_long) / 3.0,
                    'quality_signal': quality_weighted_signal,
                    'overall_confidence': overall_confidence,
                    'horizons_valid': (short_valid, medium_valid, long_valid),
                    'confidences': (conf_short, conf_medium, conf_long),
                }
            
            return quality_weighted_signal
            
        except Exception as e:
            logger.warning(f"Quality-weighted signal computation failed: {e}")
            return 0.0

    def update_forecast_trust_from_mape(self, mape_short: float, mape_medium: float, 
                                        mape_long: float) -> None:
        """
        FIX #3: Dynamically update forecast trust based on recent MAPE accuracy.
        
        Updates self.forecast_trust from static 0.5 to dynamic [0.2, 1.0] based on
        rolling MAPE accuracy:
        - If MAPE < 5%: trust = 1.0 (excellent)
        - If MAPE < 10%: trust = 0.8 (good)
        - If MAPE < 15%: trust = 0.6 (acceptable)
        - If MAPE >= 15%: trust = 0.3-0.2 (poor)
        
        This allows position sizing to scale with forecast quality.
        """
        try:
            # Compute weighted MAPE (prioritize short horizon)
            weighted_mape = (0.70 * mape_short + 0.20 * mape_medium + 0.10 * mape_long)
            
            # Map MAPE to trust [0.2, 1.0]
            if weighted_mape < 0.05:
                trust = 1.0
            elif weighted_mape < 0.10:
                trust = 0.8
            elif weighted_mape < 0.15:
                trust = 0.6
            elif weighted_mape < 0.20:
                trust = 0.4
            else:
                trust = 0.2
            
            # Update trust with smoothing (70% new, 30% old for stability)
            self.forecast_trust = 0.7 * trust + 0.3 * self.forecast_trust
            
            logger.debug(f"Updated forecast_trust to {self.forecast_trust:.3f} (MAPE: {weighted_mape:.3f})")
            
        except Exception as e:
            logger.warning(f"Forecast trust update failed: {e}")

    def compute_forecast_z_scores(
        self,
        timestep: int,
        price_history: np.ndarray,
        wind_history: np.ndarray,
        solar_history: np.ndarray,
        hydro_history: np.ndarray,
        price_mean: np.ndarray,
        price_std: np.ndarray,
        wind_mean: np.ndarray,
        wind_std: np.ndarray,
        solar_mean: np.ndarray,
        solar_std: np.ndarray,
        hydro_mean: np.ndarray,
        hydro_std: np.ndarray,
        wind_scale: float = None,  # DEPRECATED: kept for backward compatibility
        solar_scale: float = None,  # DEPRECATED: kept for backward compatibility
        hydro_scale: float = None   # DEPRECATED: kept for backward compatibility
    ) -> Dict[str, float]:
        """
        Compute forecast z-scores for all horizons and assets.
        
        Returns dict with z_short_price, z_medium_price, z_long_price, etc.
        Returns empty dict if forecasts disabled.
        """
        if not self.enabled or self.forecast_generator is None:
            return {}
        
        try:
            # Get forecasts for all horizons
            forecasts = self.forecast_generator.predict_all_horizons(timestep=timestep)
            if not isinstance(forecasts, dict):
                return {}
            
            # Get forecast horizons from config
            horizon_short = self.config.forecast_horizons.get('short', 6)
            horizon_medium = self.config.forecast_horizons.get('medium', 24)
            horizon_long = self.config.forecast_horizons.get('long', 144)
            
            # CRITICAL FIX: Use return-based z-scores (consistent with environment.py)
            # Forecast models predict price at t+h, so we compute forecast return: (forecast[t+h] - price[t]) / price[t]
            # This is bias-immune and properly aligned with trading decisions
            current_price_raw = float(price_history[timestep]) if timestep < len(price_history) else 0.0
            
            # Price forecasts (predict price at t+h where h is horizon)
            price_short = self._get_forecast_safe(forecasts, "price", "short", default=current_price_raw)
            price_medium = self._get_forecast_safe(forecasts, "price", "medium", default=current_price_raw)
            price_long = self._get_forecast_safe(forecasts, "price", "long", default=current_price_raw)
            
            # Compute forecast returns: (forecast[t+h] - price[t]) / price[t]
            # This gives the predicted return over the horizon period
            forecast_return_short = (price_short - current_price_raw) / max(abs(current_price_raw), 1.0)
            forecast_return_medium = (price_medium - current_price_raw) / max(abs(current_price_raw), 1.0)
            forecast_return_long = (price_long - current_price_raw) / max(abs(current_price_raw), 1.0)
            
            # Apply tanh with scaling factor (typical returns are ±5%, so scale by 10x)
            # This maps ±5% returns to ±0.46 z-scores, ±10% to ±0.76, ±20% to ±0.96
            # This matches the approach in environment.py _compute_forecast_deltas()
            z_short_price = float(np.tanh(forecast_return_short * 10.0))
            z_medium_price = float(np.tanh(forecast_return_medium * 10.0))
            z_long_price = float(np.tanh(forecast_return_long * 10.0))
            
            # Clip to reasonable range (tanh already bounds to [-1, 1], but clip for safety)
            z_short_price = float(np.clip(z_short_price, -1.0, 1.0))
            z_medium_price = float(np.clip(z_medium_price, -1.0, 1.0))
            z_long_price = float(np.clip(z_long_price, -1.0, 1.0))
            
            # CRITICAL FIX: Generation forecasts use proper z-score formula with TRAINING statistics
            # Use scaler statistics from training models (mean/std from training data), not rolling episode stats
            # This ensures z-scores are normalized with the same distribution as training
            wind_short = self._get_forecast_safe(forecasts, "wind", "short", default=0.0)
            solar_short = self._get_forecast_safe(forecasts, "solar", "short", default=0.0)
            hydro_short = self._get_forecast_safe(forecasts, "hydro", "short", default=0.0)
            
            # Get training statistics from scalers (preferred) or fallback to rolling stats
            # This ensures consistency with training distribution
            wind_model_key = f"wind_{self.config.forecast_horizons.get('short', 'short')}"
            solar_model_key = f"solar_{self.config.forecast_horizons.get('short', 'short')}"
            hydro_model_key = f"hydro_{self.config.forecast_horizons.get('short', 'short')}"
            
            # Try to get training mean/std from scalers first
            if (self.forecast_generator and hasattr(self.forecast_generator, 'scalers') and 
                wind_model_key in self.forecast_generator.scalers and 
                'scaler_y' in self.forecast_generator.scalers[wind_model_key]):
                scaler_y_wind = self.forecast_generator.scalers[wind_model_key]['scaler_y']
                if hasattr(scaler_y_wind, 'mean_') and len(scaler_y_wind.mean_) > 0:
                    wind_mean_val = float(scaler_y_wind.mean_[0])
                else:
                    wind_mean_val = float(wind_mean[timestep]) if timestep < len(wind_mean) else 0.0
                if hasattr(scaler_y_wind, 'scale_') and len(scaler_y_wind.scale_) > 0:
                    wind_std_val = max(float(scaler_y_wind.scale_[0]), 1e-6)
                else:
                    wind_std_val = max(float(wind_std[timestep]), 1e-6) if timestep < len(wind_std) else 1.0
            else:
                # Fallback to rolling stats if scalers not available
                wind_mean_val = float(wind_mean[timestep]) if timestep < len(wind_mean) else 0.0
                wind_std_val = max(float(wind_std[timestep]), 1e-6) if timestep < len(wind_std) else 1.0
            
            if (self.forecast_generator and hasattr(self.forecast_generator, 'scalers') and 
                solar_model_key in self.forecast_generator.scalers and 
                'scaler_y' in self.forecast_generator.scalers[solar_model_key]):
                scaler_y_solar = self.forecast_generator.scalers[solar_model_key]['scaler_y']
                if hasattr(scaler_y_solar, 'mean_') and len(scaler_y_solar.mean_) > 0:
                    solar_mean_val = float(scaler_y_solar.mean_[0])
                else:
                    solar_mean_val = float(solar_mean[timestep]) if timestep < len(solar_mean) else 0.0
                if hasattr(scaler_y_solar, 'scale_') and len(scaler_y_solar.scale_) > 0:
                    solar_std_val = max(float(scaler_y_solar.scale_[0]), 1e-6)
                else:
                    solar_std_val = max(float(solar_std[timestep]), 1e-6) if timestep < len(solar_std) else 1.0
            else:
                solar_mean_val = float(solar_mean[timestep]) if timestep < len(solar_mean) else 0.0
                solar_std_val = max(float(solar_std[timestep]), 1e-6) if timestep < len(solar_std) else 1.0
            
            if (self.forecast_generator and hasattr(self.forecast_generator, 'scalers') and 
                hydro_model_key in self.forecast_generator.scalers and 
                'scaler_y' in self.forecast_generator.scalers[hydro_model_key]):
                scaler_y_hydro = self.forecast_generator.scalers[hydro_model_key]['scaler_y']
                if hasattr(scaler_y_hydro, 'mean_') and len(scaler_y_hydro.mean_) > 0:
                    hydro_mean_val = float(scaler_y_hydro.mean_[0])
                else:
                    hydro_mean_val = float(hydro_mean[timestep]) if timestep < len(hydro_mean) else 0.0
                if hasattr(scaler_y_hydro, 'scale_') and len(scaler_y_hydro.scale_) > 0:
                    hydro_std_val = max(float(scaler_y_hydro.scale_[0]), 1e-6)
                else:
                    hydro_std_val = max(float(hydro_std[timestep]), 1e-6) if timestep < len(hydro_std) else 1.0
            else:
                hydro_mean_val = float(hydro_mean[timestep]) if timestep < len(hydro_mean) else 0.0
                hydro_std_val = max(float(hydro_std[timestep]), 1e-6) if timestep < len(hydro_std) else 1.0
            
            # Proper z-score computation: (value - training_mean) / training_std
            # This ensures z-scores match the training distribution
            z_short_wind = (wind_short - wind_mean_val) / wind_std_val
            z_short_solar = (solar_short - solar_mean_val) / solar_std_val
            z_short_hydro = (hydro_short - hydro_mean_val) / hydro_std_val
            
            # Clip generation z-scores to reasonable range
            z_short_wind = float(np.clip(z_short_wind, -3.0, 3.0))
            z_short_solar = float(np.clip(z_short_solar, -3.0, 3.0))
            z_short_hydro = float(np.clip(z_short_hydro, -3.0, 3.0))
            
            # Update internal state
            self.z_short_price = z_short_price
            self.z_medium_price = z_medium_price
            self.z_long_price = z_long_price
            self.z_short_wind = z_short_wind
            self.z_short_solar = z_short_solar
            self.z_short_hydro = z_short_hydro
            
            # Store in history for reward alignment
            if self.env is not None:
                if not hasattr(self.env, '_z_score_history'):
                    self.env._z_score_history = {}
                self.env._z_score_history[timestep] = {
                    'z_short': z_short_price,
                    'z_medium': z_medium_price,
                    'z_long': z_long_price
                }
            
            return {
                'z_short_price': z_short_price,
                'z_medium_price': z_medium_price,
                'z_long_price': z_long_price,
                'z_short_wind': z_short_wind,
                'z_short_solar': z_short_solar,
                'z_short_hydro': z_short_hydro
            }
            
        except Exception as e:
            logger.warning(f"Forecast z-score computation failed: {e}")
            return {}
    
    def build_observation_features(self) -> ForecastObservationFeatures:
        """
        Build forecast features to add to observations.
        
        Returns ForecastObservationFeatures with arrays for each agent.
        Returns empty features if forecasts disabled.
        
        TEMPORAL ALIGNMENT FIX: Uses lagged z_medium (from t-24) to align with return at t+24.
        This ensures the forecast that predicted current conditions is used.
        """
        if not self.enabled:
            return ForecastObservationFeatures()
        
        try:
            # PHASE 2 FIX: Add forecast error to observations (4D instead of 3D)
            # This allows agent to adapt to changing forecast quality
            env = getattr(self, 'env', None)
            normalized_error = 0.0
            if env is not None:
                try:
                    # Get recent MAPE for short horizon (most relevant for trading)
                    mape_history = getattr(env, '_horizon_mape', {}).get('short', [])
                    if len(mape_history) > 0:
                        recent_mape = float(np.mean(list(mape_history)[-10:]))
                        mape_thresholds = getattr(env, '_mape_thresholds', {})
                        mape_threshold_short = mape_thresholds.get('short', 0.02)
                        # CRITICAL FIX: Normalize error to [0, 1] to match observation space scale
                        # 0.0 = perfect, 1.0 = at threshold or worse (clipped at threshold)
                        # This ensures normalized_error is on same scale as other features (not 2x range)
                        normalized_error = float(np.clip(recent_mape / max(mape_threshold_short, 1e-6), 0.0, 1.0))
                except Exception:
                    pass
            
            # TEMPORAL ALIGNMENT: Get lagged z_medium from t-24 to align with return at t+24
            # The forecast made 24 steps ago predicted what's happening now
            z_medium_lagged = self.z_medium_price  # Default to current if no lag available
            current_timestep = getattr(env, 't', None) if env is not None else None
            horizon_medium = getattr(self.config, 'forecast_horizons', {}).get('medium', 24)
            
            if current_timestep is not None and current_timestep >= horizon_medium:
                lag_timestep = current_timestep - horizon_medium
                if hasattr(env, '_z_score_history') and lag_timestep in env._z_score_history:
                    z_medium_lagged = float(env._z_score_history[lag_timestep].get('z_medium', z_medium_lagged))
            
            # ENGINEERED FEATURES: Direction, Momentum, Strength
            # Direction: sign of z_medium_lagged (-1 for down, +1 for up, 0 for neutral)
            direction = float(np.sign(z_medium_lagged))
            
            # Momentum: change in z_medium (current - previous)
            z_medium_prev = getattr(self, '_z_medium_prev', z_medium_lagged)
            momentum = float(np.clip(z_medium_lagged - z_medium_prev, -1.0, 1.0))
            self._z_medium_prev = z_medium_lagged  # Store for next iteration
            
            # Strength: absolute value of z_medium_lagged (magnitude of signal)
            strength = float(np.clip(abs(z_medium_lagged), 0.0, 1.0))
            
            # TRADE SIGNAL: Composite feature combining direction, strength, and trust
            # Range: [-1, 1] where positive = buy signal, negative = sell signal, magnitude = confidence
            trade_signal = float(np.clip(direction * strength * self.forecast_trust, -1.0, 1.0))
            
            # TIER 22: Full forecast features (8D: z_short, z_medium_lagged, direction, momentum, strength, forecast_trust, normalized_error, trade_signal)
            # All features provide rich context for the agent to learn from
            investor_features = np.array([
                self.z_short_price,        # Short-term price forecast z-score
                z_medium_lagged,           # Medium-term price forecast (temporally aligned)
                direction,                 # Sign of forecast signal (-1, 0, +1)
                momentum,                  # Change in forecast signal
                strength,                  # Absolute magnitude of forecast signal
                self.forecast_trust,       # Forecast trust score
                normalized_error,          # Recent forecast error (reliability indicator)
                trade_signal               # Composite signal (direction * strength * trust)
            ], dtype=np.float32)
            
            # Battery features: 6D (wind, solar, hydro, z_short, z_medium, z_long)
            # CHANGE: Use separate generation signals instead of merged total
            battery_features = np.array([
                float(np.clip(self.z_short_wind, -1.0, 1.0)),
                float(np.clip(self.z_short_solar, -1.0, 1.0)),
                float(np.clip(self.z_short_hydro, -1.0, 1.0)),
                self.z_short_price,
                self.z_medium_price,
                self.z_long_price
            ], dtype=np.float32)
            
            # Risk features: 3D (z_short, vol_forecast, trust)
            forecast_spread = abs(self.z_short_price - self.z_medium_price) + \
                            abs(self.z_medium_price - self.z_long_price) + \
                            abs(self.z_short_price - self.z_long_price)
            price_volatility_forecast = float(np.clip(forecast_spread / 3.0, 0.0, 1.0))
            
            risk_features = np.array([
                self.z_short_price,
                price_volatility_forecast,
                self.forecast_trust
            ], dtype=np.float32)
            
            # Meta features: 2D (trust, expected_return)
            expected_return = (0.7 * self.z_short_price + 
                             0.2 * self.z_medium_price + 
                             0.1 * self.z_long_price) * self.forecast_trust
            meta_features = np.array([
                self.forecast_trust,
                float(np.clip(expected_return, -1.0, 1.0))
            ], dtype=np.float32)
            
            return ForecastObservationFeatures(
                investor_features=investor_features,
                battery_features=battery_features,
                risk_features=risk_features,
                meta_features=meta_features
            )
            
        except Exception as e:
            logger.warning(f"Forecast observation building failed: {e}")
            return ForecastObservationFeatures()
    
    def update_forecast_trust(self, calibration_tracker: Optional[Any] = None):
        """
        Update forecast trust score from calibration tracker.
        
        Args:
            calibration_tracker: Optional calibration tracker for trust calculation
        """
        if not self.enabled:
            self.forecast_trust = 0.5
            return
        
        try:
            if calibration_tracker is not None:
                self.forecast_trust = float(calibration_tracker.get_trust(horizon="short"))
            else:
                # Fallback: use default trust
                self.forecast_trust = getattr(self.config, 'confidence_floor', 0.6)
        except Exception as e:
            logger.warning(f"Forecast trust update failed: {e}")
            self.forecast_trust = 0.5
    
    def update_horizon_correlations(self, correlations: Dict[str, float]):
        """Update horizon correlation tracking."""
        if self.enabled:
            self._horizon_correlations.update(correlations)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current forecast engine state."""
        return {
            'enabled': self.enabled,
            'z_short_price': self.z_short_price,
            'z_medium_price': self.z_medium_price,
            'z_long_price': self.z_long_price,
            'z_short_wind': self.z_short_wind,
            'z_short_solar': self.z_short_solar,
            'z_short_hydro': self.z_short_hydro,
            'forecast_trust': self.forecast_trust,
            'horizon_correlations': self._horizon_correlations.copy()
        }
    
    def _get_forecast_safe(
        self,
        forecasts: Dict[str, Any],
        target: str,
        horizon: str,
        default: float = 0.0
    ) -> float:
        """Safely get forecast value from dict."""
        try:
            key = f"{target}_forecast_{horizon}"
            if key in forecasts:
                val = forecasts[key]
                if isinstance(val, (int, float, np.number)):
                    return float(val)
                elif isinstance(val, np.ndarray):
                    return float(val.flatten()[0])
            return default
        except Exception:
            return default

    @staticmethod
    def _direction_accuracy(pred_signal: float, realized_ret: float, min_abs: float = 1e-4) -> float:
        if pred_signal is None or realized_ret is None:
            return 0.0
        if abs(realized_ret) < min_abs or abs(pred_signal) < min_abs:
            return 0.0
        return 1.0 if np.sign(pred_signal) == np.sign(realized_ret) else 0.0
