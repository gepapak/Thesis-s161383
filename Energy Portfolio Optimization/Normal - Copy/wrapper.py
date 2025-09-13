#!/usr/bin/env python3
"""
Enhanced Multi-Horizon Wrapper (fully patched w/ forecast normalization + horizon alignment)

Adds logging for:
- Env snapshot: budget, capacities, battery_energy, revenue_step (prefers env.last_revenue), actual price/load/wind/solar/hydro
- Episode markers: ep_meta_return, step_in_episode, episode_end, episode_id, seed
- Market regime: market_stress, market_volatility
- Risk vector 6D, plus overall/market quick snapshots
- Reward breakdown + weights (reads env.last_reward_breakdown / env.last_reward_weights if present)
- Ops health: step_time_ms, mem_rss_mb
- Action health: action_clip_frac_* (investor/battery/risk/meta)
- Forecast MAE@1 (price/wind/solar/load/hydro) vs previous logged forecast
- PATCH: Logs true mark-to-market portfolio value/equity and computes performance from it
- NEW (aligned with env economics): generation_revenue, mtm_pnl, distributed_profits,
  cumulative_returns, investment_capital, fund_performance
- NEW (this patch): normalized forecast features + horizon alignment prioritizing env.investment_freq
- NEW logging fields: price_forecast_aligned, price_forecast_norm, forecast_alignment_score

Maintains:
- TOTAL-dimension observation construction (base + forecasts) with strict shape checking
- Buffered CSV writing with safe type coercion (timestamp preserved as string)
"""

from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import csv, os, threading, gc, psutil, logging, pandas as pd, time
from typing import Dict, Any, Tuple, Optional, List, Mapping
from datetime import datetime
from collections import deque, OrderedDict
from contextlib import nullcontext

# Import enhanced monitoring (optional)
try:
    from enhanced_monitoring import EnhancedMetricsMonitor
    _HAS_ENHANCED_MONITORING = True
except Exception:
    _HAS_ENHANCED_MONITORING = False
    EnhancedMetricsMonitor = None

# PRICE_SCALE moved to config - will be imported from environment's config


# =========================
# Memory + Validation Utils
# =========================
class SafeDivision:
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator is None or abs(denominator) < 1e-9:
            return default
        try:
            return float(numerator) / float(denominator)
        except Exception:
            return default

    @staticmethod
    def _safe_volatility(values, default=0.0):
        """Calculate volatility safely, handling edge cases."""
        try:
            if len(values) < 2:
                return default

            # Convert to numpy array and remove NaN/inf values
            arr = np.array(values, dtype=float)
            arr = arr[np.isfinite(arr)]

            if len(arr) < 2:
                return default

            # Calculate returns
            returns = np.diff(arr) / arr[:-1]
            returns = returns[np.isfinite(returns)]

            if len(returns) < 1:
                return default

            # Calculate standard deviation
            volatility = np.std(returns)

            if not np.isfinite(volatility):
                return default

            # Cap volatility at reasonable bounds
            return float(np.clip(volatility, 0.0, 10.0))

        except Exception:
            return default


# =========================
# Enhanced Cache Management
# =========================
class EnhancedLRUCache:
    """Enhanced LRU cache with memory-aware eviction."""
    def __init__(self, max_size: int = 2000, memory_limit_mb: float = 50.0):
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self.cache = OrderedDict()
        self.access_count = 0

    def get(self, key):
        """Get value and move to end (most recently used)."""
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_count += 1
            return value
        return None

    def put(self, key, value):
        """Put value, evicting LRU if necessary."""
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.access_count += 1
        if self.access_count % 500 == 0:  # Reduced frequency for better performance
            self._memory_cleanup()

    def _memory_cleanup(self):
        """Clean up cache if memory usage is high."""
        try:
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            if current_memory > self.memory_limit_mb and len(self.cache) > 100:
                items_to_remove = max(1, len(self.cache) // 3)
                for _ in range(items_to_remove):
                    if self.cache:
                        self.cache.popitem(last=False)
        except Exception:
            pass

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def clear(self):
        self.cache.clear()
        self.access_count = 0


# =========================
# Forecast Post-Processing
# =========================
class ForecastPostProcessor:
    """
    - Normalizes forecasts to match env observation scales:
        price -> price/PRICE_SCALE clipped [-10,10] (same dynamic range as env price_n)
        wind/solar/hydro -> divide by env's p95 scales, clipped [0,1]
        load -> divide by env.load_scale, clipped [0,1]
      Unknown targets are passed through unchanged.

    - Aligns horizons for each agent by ordering target×horizon keys so that the
      horizon closest to env.investment_freq comes first. We DO NOT change total dims;
      we only re-order priority when building the feature vector.
    """
    def __init__(self, env, normalize=True, align_horizons=True):
        self.env = env
        self.normalize = bool(normalize)
        self.align_horizons = bool(align_horizons)

    def normalize_value(self, key: str, val: float) -> float:
        if not self.normalize:
            return float(val) if np.isfinite(val) else 0.0

        v = float(val) if np.isfinite(val) else 0.0
        k = (key or "").lower()

        if "price" in k:
            # FIXED: Use z-score normalization like the environment instead of fixed scale
            try:
                # Get current timestep for normalization parameters
                t = getattr(self.env, 't', 0)
                if hasattr(self.env, '_price_mean') and hasattr(self.env, '_price_std'):
                    if t < len(self.env._price_mean) and t < len(self.env._price_std):
                        mean = float(self.env._price_mean[t])
                        std = float(self.env._price_std[t])
                        std = max(std, 1e-6)  # Avoid division by zero
                        normalized = (v - mean) / std
                        return float(np.clip(normalized, -3.0, 3.0))  # Same bounds as environment

                # Fallback: use reasonable DKK price statistics if normalization params unavailable
                mean = 250.0  # Typical DKK price
                std = 50.0    # Typical DKK price volatility
                normalized = (v - mean) / std
                return float(np.clip(normalized, -3.0, 3.0))
            except Exception:
                # Last resort fallback to old method
                scale = 10.0
                return float(np.clip(v / scale, -10.0, 10.0))

        if "load" in k:
            try:
                scale = max(float(getattr(self.env, "load_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))
            except Exception:
                return float(np.clip(v, 0.0, 1.0))

        try:
            if "wind" in k:
                scale = max(float(getattr(self.env, "wind_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))
            if "solar" in k:
                scale = max(float(getattr(self.env, "solar_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))
            if "hydro" in k:
                scale = max(float(getattr(self.env, "hydro_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))
        except Exception:
            pass

        return v

    def order_keys_by_horizon_alignment(self, agent: str, keys: List[str]) -> List[str]:
        """
        Reorders forecast keys so those whose horizon is closest to env.investment_freq
        appear first. We expect keys like "..._forecast_<h>" where <h> is an integer.
        If parsing fails, returns keys unchanged.
        """
        if not self.align_horizons or not isinstance(keys, list) or len(keys) == 0:
            return keys

        try:
            inv_freq = int(getattr(self.env, "investment_freq", 12))

            def parse_h(k):
                try:
                    suf = str(k).split("_")[-1]
                    return abs(int(suf) - inv_freq), int(suf)
                except Exception:
                    return (10**9, 10**9)

            return sorted(keys, key=parse_h)
        except Exception:
            return keys

    def normalize_and_align(self, agent: str, raw_forecasts: Mapping[str, float],
                            expected_keys: List[str]) -> Dict[str, float]:
        """
        Returns a dict with the same key set as expected_keys, filled with normalized values.
        Keys are supplied in an order that prioritizes alignment; values are pulled from raw
        (with 0.0 fallback), normalized per target.
        """
        if not isinstance(raw_forecasts, Mapping):
            raw_forecasts = {}

        ordered_keys = self.order_keys_by_horizon_alignment(agent, list(expected_keys))
        out = {}
        for k in ordered_keys:
            v = raw_forecasts.get(k, 0.0)
            out[k] = self.normalize_value(k, v)

        final_out = {}
        for k in expected_keys:
            final_out[k] = out.get(k, 0.0)
        return final_out


# =========================
# Memory tracker
# =========================
class EnhancedMemoryTracker:
    """Lightweight memory tracker for caches used in the wrapper."""
    def __init__(self, max_memory_mb=300):
        self.max_memory_mb = max_memory_mb
        self.cleanup_thresholds = {
            'light': max_memory_mb * 0.7,
            'medium': max_memory_mb * 0.85,
            'heavy': max_memory_mb * 0.95,
        }
        self.tracked_caches = []
        self.memory_history = deque(maxlen=200)

    def register_cache(self, cache_obj):
        self.tracked_caches.append(cache_obj)

    def get_memory_usage(self) -> float:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def should_cleanup(self, force=False):
        cur = self.get_memory_usage()
        self.memory_history.append(cur)
        if force:
            return 'heavy', cur
        if cur > self.cleanup_thresholds['heavy']:
            return 'heavy', cur
        if cur > self.cleanup_thresholds['medium']:
            return 'medium', cur
        if cur > self.cleanup_thresholds['light'] or len(self.memory_history) % 200 == 0:
            return 'light', cur
        return None, cur

    def cleanup(self, level='light'):
        before = self.get_memory_usage()
        try:
            if level in ('light', 'medium', 'heavy'):
                for cache in list(self.tracked_caches):
                    try:
                        if hasattr(cache, 'clear'):
                            cache.clear()
                    except Exception:
                        pass
            for _ in range(2):
                gc.collect()
            try:
                import torch  # noqa: F401
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finally:
            after = self.get_memory_usage()
        return max(0.0, before - after)

    def get_memory_stats(self):
        cur = self.get_memory_usage()
        return {
            'current_memory_mb': cur,
            'max_memory_mb': self.max_memory_mb,
            'memory_usage_pct': (cur / self.max_memory_mb) * 100 if self.max_memory_mb else 0.0,
            'tracked_caches': len(self.tracked_caches),
            'memory_history': list(self.memory_history),
        }


# =========================
# Observation Validator
# =========================
class EnhancedObservationValidator:
    """
    Validates base obs from the ENV and appends forecast features to build TOTAL obs.

    IMPORTANT: the ENV returns BASE-only observations. The WRAPPER is responsible for:
      total_dim = base_dim + forecast_dim

    Forecasts are normalized AND keys are prioritized to align horizons (closest to env.investment_freq first).
    """
    def __init__(self, base_env, forecaster, debug=False, postproc: Optional[ForecastPostProcessor] = None):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug
        self.postproc = postproc or ForecastPostProcessor(base_env, normalize=True, align_horizons=True)

        self.agent_observation_specs = {}
        self.validation_errors = deque(maxlen=50)
        self.validation_cache = {}

        self._initialize_observation_specs()

    def _initialize_observation_specs(self):
        for agent in self.base_env.possible_agents:
            try:
                base_dim = self._get_validated_base_dim(agent)
                forecast_dim = self._calculate_forecast_dimension(agent)
                total_dim = base_dim + forecast_dim
                total_low, total_high = self._get_safe_bounds(total_dim)
                self.agent_observation_specs[agent] = {
                    'base_dim': base_dim,
                    'forecast_dim': forecast_dim,
                    'total_dim': total_dim,
                    'forecast_keys': self._get_agent_forecast_keys(agent, forecast_dim),
                    'bounds': (total_low, total_high)
                }
                if self.debug:
                    print(f"✅ {agent}: base={base_dim}, forecast={forecast_dim}, total={total_dim}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to init specs for {agent}: {e}")
                self._create_fallback_spec(agent)

    def _get_validated_base_dim(self, agent: str) -> int:
        try:
            if hasattr(self.base_env, '_get_base_observation_dim'):
                return int(self.base_env._get_base_observation_dim(agent))
            space = self.base_env.observation_space(agent)
            return int(space.shape[0])
        except Exception:
            return {'investor_0': 6, 'battery_operator_0': 4, 'risk_controller_0': 9, 'meta_controller_0': 11}.get(agent, 8)

    def _calculate_forecast_dimension(self, agent: str) -> int:
        try:
            if hasattr(self.forecaster, 'get_agent_forecast_dims'):
                dims = self.forecaster.get_agent_forecast_dims()
                base_dims = int(dims.get(agent, 0))
                # Add 1 for forecast confidence (except risk_controller_0 which doesn't get confidence)
                return base_dims + (1 if agent != "risk_controller_0" else 0)
            if hasattr(self.forecaster, 'agent_horizons') and hasattr(self.forecaster, 'agent_targets'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                base_dims = int(len(targets) * len(horizons))
                # Add 1 for forecast confidence (except risk_controller_0 which doesn't get confidence)
                return base_dims + (1 if agent != "risk_controller_0" else 0)
            # FIXED: Correct forecast dimensions based on actual agent allocations + confidence
            # investor_0: 4 targets × 3 horizons + 1 confidence = 13
            # battery_operator_0: 3 targets × 4 horizons + 1 confidence = 13
            # risk_controller_0: 4 targets × 3 horizons (no confidence) = 12
            # meta_controller_0: 5 targets × 4 horizons + 1 confidence = 21
            return {'investor_0': 13, 'battery_operator_0': 13, 'risk_controller_0': 12, 'meta_controller_0': 21}.get(agent, 0)
        except Exception:
            # FIXED: Correct forecast dimensions based on actual agent allocations + confidence
            return {'investor_0': 13, 'battery_operator_0': 13, 'risk_controller_0': 12, 'meta_controller_0': 21}.get(agent, 0)

    def _get_agent_forecast_keys(self, agent: str, expected_count: int) -> List[str]:
        try:
            if hasattr(self.forecaster, 'agent_targets') and hasattr(self.forecaster, 'agent_horizons'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                raw_keys = [f"{t}_forecast_{h}" for t in targets for h in horizons]
                ordered = self.postproc.order_keys_by_horizon_alignment(agent, raw_keys)
                while len(ordered) < expected_count:
                    ordered.append(f"fallback_forecast_{len(ordered)}")
                return ordered[:expected_count]
        except Exception:
            pass
        return [f"forecast_{i}" for i in range(expected_count)]

    def _get_safe_bounds(self, total_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        low = np.full(total_dim, -100.0, dtype=np.float32)
        high = np.full(total_dim, 1000.0, dtype=np.float32)
        return low, high

    def _create_fallback_spec(self, agent: str):
        base_dim = {'investor_0': 6, 'battery_operator_0': 4, 'risk_controller_0': 9, 'meta_controller_0': 11}.get(agent, 8)
        # FIXED: Correct forecast dimensions based on actual agent allocations + confidence
        forecast_dim = {'investor_0': 13, 'battery_operator_0': 13, 'risk_controller_0': 12, 'meta_controller_0': 21}.get(agent, 0)
        total_dim = base_dim + forecast_dim
        self.agent_observation_specs[agent] = {
            'base_dim': base_dim,
            'forecast_dim': forecast_dim,
            'total_dim': total_dim,
            'forecast_keys': self._get_agent_forecast_keys(agent, forecast_dim),
            'bounds': self._get_safe_bounds(total_dim)
        }

    # ---- in-place assembly ----
    def build_total_observation(self, agent: str, base_obs: np.ndarray,
                                forecasts: Mapping[str, float],
                                out: Optional[np.ndarray] = None) -> np.ndarray:
        if agent not in self.agent_observation_specs:
            return self._create_safe_observation(agent)
        spec = self.agent_observation_specs[agent]
        bd, fd, td = spec['base_dim'], spec['forecast_dim'], spec['total_dim']
        if out is None or not isinstance(out, np.ndarray) or out.shape != (td,):
            out = np.zeros(td, dtype=np.float32)

        base_obs = self._validate_base_observation(agent, base_obs, spec)
        out[:bd] = base_obs

        if fd > 0:
            keys = spec['forecast_keys'][:fd]
            f_proc = self.postproc.normalize_and_align(agent, forecasts, keys)
            out[bd:bd + fd].fill(0.0)
            for j, key in enumerate(keys):
                v = f_proc.get(key, 0.0)
                out[bd + j] = float(v if np.isfinite(v) else 0.0)

        low, high = spec['bounds']
        np.clip(out, low[:td], high[:td], out=out)
        return out

    # ---- helpers used by both paths ----
    def _validate_base_observation(self, agent: str, base_obs: np.ndarray, spec: Dict) -> np.ndarray:
        if not isinstance(base_obs, np.ndarray):
            if base_obs is None:
                base_obs = np.zeros(spec['base_dim'], dtype=np.float32)
            elif isinstance(base_obs, (list, tuple)):
                base_obs = np.array(base_obs, dtype=np.float32)
            else:
                base_obs = np.full(spec['base_dim'], float(base_obs), dtype=np.float32)
        else:
            base_obs = base_obs.astype(np.float32)
        if base_obs.ndim != 1:
            base_obs = base_obs.flatten()
        bd = spec['base_dim']
        if base_obs.size < bd:
            base_obs = np.pad(base_obs, (0, bd - base_obs.size))
        elif base_obs.size > bd:
            base_obs = base_obs[:bd]
        low, high = spec['bounds']
        base_obs = np.nan_to_num(base_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        base_obs = np.clip(base_obs, low[:bd], high[:bd])
        return base_obs

    def _create_safe_observation(self, agent: str) -> np.ndarray:
        if agent in self.agent_observation_specs:
            spec = self.agent_observation_specs[agent]
            expected = spec['total_dim']
            low, high = spec['bounds']
            return ((low[:expected] + high[:expected]) / 2.0).astype(np.float32)
        return np.zeros(20, dtype=np.float32)

    def get_validation_stats(self) -> Dict:
        return {
            'agents_configured': len(self.agent_observation_specs),
            'recent_errors': len(self.validation_errors),
            'cache_size': len(self.validation_cache),
            'specs': {a: {
                'base_dim': s['base_dim'],
                'forecast_dim': s['forecast_dim'],
                'total_dim': s['total_dim'],
            } for a, s in self.agent_observation_specs.items()},
            'recent_error_messages': list(self.validation_errors)[-5:]
        }


# =========================
# Observation Builder
# =========================
class MemoryOptimizedObservationBuilder:
    """Builds total observations (base + normalized & horizon-aligned forecasts) with validation & caching, in-place."""
    def __init__(self, base_env, forecaster, debug=False, normalize_forecasts=True, align_horizons=True):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug

        self.postproc = ForecastPostProcessor(base_env, normalize=normalize_forecasts, align_horizons=align_horizons)
        self.validator = EnhancedObservationValidator(base_env, forecaster, debug, postproc=self.postproc)

        self._obs_out = {
            agent: np.zeros(self.validator.agent_observation_specs[agent]['total_dim'], dtype=np.float32)
            for agent in self.base_env.possible_agents
        }

        # Get cache settings from environment config
        env_config = getattr(self.base_env, 'config', None)
        if env_config:
            forecast_cache_size = env_config.forecast_cache_size
            agent_cache_size = env_config.agent_forecast_cache_size
            memory_limit = env_config.wrapper_memory_mb
        else:
            # Fallback values
            forecast_cache_size = 1000
            agent_cache_size = 2000
            memory_limit = 500.0

        self.forecast_cache = EnhancedLRUCache(max_size=forecast_cache_size, memory_limit_mb=memory_limit)
        self.agent_forecast_cache = EnhancedLRUCache(max_size=agent_cache_size, memory_limit_mb=memory_limit * 1.5)

        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=memory_limit)
        self.memory_tracker.register_cache(self.forecast_cache)
        self.memory_tracker.register_cache(self.agent_forecast_cache)

    def enhance_observations(self, base_obs: Dict[str, np.ndarray],
                             all_forecasts: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        cleanup_level, _ = self.memory_tracker.should_cleanup()
        if cleanup_level:
            self.memory_tracker.cleanup(cleanup_level)

        t = getattr(self.base_env, 't', 0)
        per_agent_possible = hasattr(self.forecaster, 'predict_for_agent')

        for agent in self.base_env.possible_agents:
            try:
                if agent in base_obs:
                    if all_forecasts is not None:
                        fsrc: Mapping[str, float] = all_forecasts
                    elif per_agent_possible:
                        fsrc = self._get_cached_forecasts_for_agent(agent, t)
                    else:
                        fsrc = self._get_cached_forecasts_global(t)

                    self.validator.build_total_observation(
                        agent, base_obs[agent], fsrc, out=self._obs_out[agent]
                    )
                else:
                    self._obs_out[agent][:] = self.validator._create_safe_observation(agent)
            except Exception:
                self._obs_out[agent][:] = self.validator._create_safe_observation(agent)
        return self._obs_out

    # ---- forecast caching ----
    def _get_cached_forecasts_global(self, timestep: int) -> Dict[str, float]:
        key = f"forecasts_{timestep}"
        cached_result = self.forecast_cache.get(key)
        if cached_result is not None:
            return cached_result
        try:
            all_forecasts = self.forecaster.predict_all_horizons(timestep=timestep)
            if not isinstance(all_forecasts, dict):
                all_forecasts = {}
        except Exception:
            all_forecasts = {}
        self.forecast_cache.put(key, all_forecasts)
        return all_forecasts

    def _get_cached_forecasts_for_agent(self, agent: str, timestep: int) -> Dict[str, float]:
        key = (agent, timestep)
        cached_result = self.agent_forecast_cache.get(key)
        if cached_result is not None:
            return cached_result
        try:
            if hasattr(self.forecaster, 'predict_for_agent'):
                f = self.forecaster.predict_for_agent(agent=agent, timestep=timestep)
                if not isinstance(f, dict):
                    f = {}
            else:
                f = self._get_cached_forecasts_global(timestep)
        except Exception:
            f = {}
        self.agent_forecast_cache.put(key, f)
        return f

    def get_diagnostic_info(self) -> Dict:
        return {
            'validator_stats': self.validator.get_validation_stats(),
            'memory_stats': self.memory_tracker.get_memory_stats(),
            'cache_size': {'global': len(self.forecast_cache), 'per_agent': len(self.agent_forecast_cache)},
            'cache_limit': {
                'global': getattr(self.forecast_cache, 'max_size', None),
                'per_agent': getattr(self.agent_forecast_cache, 'max_size', None)
            }
        }


# =========================
# Wrapper Env
# =========================
class MultiHorizonWrapperEnv(ParallelEnv):
    """Wraps a BASE-only env and exposes TOTAL-dim observations (base + normalized & horizon-aligned forecasts)."""
    metadata = {"name": "multi_horizon_wrapper:normalized-aligned-v1"}

    def __init__(self, base_env, multi_horizon_forecaster, log_path=None, max_memory_mb=1500,
                 normalize_forecasts=True, align_horizons=True, total_timesteps=50000, log_last_n=100,
                 disable_csv_logging=False):
        self.env = base_env
        self.forecaster = multi_horizon_forecaster

        # Agents
        self._possible_agents = self.env.possible_agents[:]
        self._agents = self.env.agents[:]

        # Logging control - configurable based on disable_csv_logging
        self.total_timesteps = total_timesteps
        self.log_last_n = log_last_n if not disable_csv_logging else 0
        self.log_start_step = 0
        self.logging_enabled = not disable_csv_logging  # Disable logging if requested
        self.disable_csv_logging = disable_csv_logging

        # Memory & logging infra
        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=max_memory_mb)
        self.log_path = self._setup_logging_path(log_path) if log_path else None
        self.log_buffer = deque(maxlen=256)
        self.log_lock = threading.RLock() if threading else nullcontext()
        self._flush_every_rows = 1000

        # Enhanced monitoring
        self.enhanced_monitor = None
        if _HAS_ENHANCED_MONITORING:
            self.enhanced_monitor = EnhancedMetricsMonitor(
                window_size=1000,
                log_frequency=100
            )

        # Inject forecaster reference into env if empty
        if hasattr(self.env, 'forecast_generator') and getattr(self.env, 'forecast_generator') is None:
            try:
                self.env.forecast_generator = self.forecaster
            except Exception:
                pass

        # Builder / specs
        self.obs_builder = MemoryOptimizedObservationBuilder(
            self.env, self.forecaster, debug=False,
            normalize_forecasts=normalize_forecasts, align_horizons=align_horizons
        )
        self.memory_tracker.register_cache(self.obs_builder.forecast_cache)
        self.memory_tracker.register_cache(self.obs_builder.agent_forecast_cache)
        self.memory_tracker.register_cache(self.log_buffer)

        # Observation spaces (TOTAL dims)
        self._build_wrapper_observation_spaces()

        # counters & episode markers
        self.step_count = 0
        self.episode_count = 0
        # keep errors bounded to avoid log spam
        self.error_count = 0
        self.max_errors = 100  # Increased limit to avoid silent failures

        self._ep_meta_return = 0.0
        self._prev_forecasts_for_error: Dict[str, float] = {}
        self._log_interval = 20
        self._last_episode_seed = None
        self._last_episode_end_flag = 0
        self._last_step_wall_ms = 0.0
        self._last_clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}
        self._last_clip_totals = {"investor": 1, "battery": 1, "risk": 1, "meta": 1}

        # a couple of last-known values for logging
        self._last_price_forecast_norm = 0.0
        self._last_price_forecast_aligned = 0.0

        # logging - initialize
        if self.log_path and not self.disable_csv_logging:
            print(f"📊 Full logging every 20 timesteps from start")
            self._initialize_logging_safe()
        elif self.disable_csv_logging:
            print("⚡ CSV logging disabled - maximum speed mode")
        else:
            print("⚡ No logging configured - maximum speed mode")
        print("✅ Enhanced multi-horizon wrapper initialized (TOTAL-dim observations, normalized + aligned forecasts)")

    # ---- spaces ----
    def _build_wrapper_observation_spaces(self):
        self._obs_spaces = {}
        specs = self.obs_builder.validator.agent_observation_specs
        for agent, spec in specs.items():
            low, high = spec['bounds']
            total_dim = spec['total_dim']
            self._obs_spaces[agent] = spaces.Box(low=low[:total_dim], high=high[:total_dim],
                                                 shape=(total_dim,), dtype=np.float32)

    # ---- logging setup ----
    def _setup_logging_path(self, log_path):
        if log_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/multi_horizon_metrics_{ts}.csv"
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Could not create log directory: {e}")
            ts = datetime.now().strftime("%Y%m%d_%H%%M%S")
            log_path = f"fallback_metrics_{ts}.csv"
        return log_path

    def _should_log_this_step(self) -> bool:
        """Determine if we should log this timestep."""
        # Check if CSV logging is disabled
        if self.disable_csv_logging:
            return False

        # Check if only final results should be logged
        if self.log_last_n == 1:  # final_results_only mode
            # Only log on the very last step
            return self.step_count >= (self.total_timesteps - 1)

        # Always log (normal mode only)
        return True

    def _initialize_logging_safe(self):
        # Initialize logging based on mode
        if not self.log_path:
            return

        # Create log file immediately (normal mode)
        try:
            if not os.path.isfile(self.log_path):
                self._create_log_header()
            print(f"✅ Logging initialized: {self.log_path}")
        except Exception as e:
            print(f"⚠️ Logging initialization failed: {e}")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"emergency_metrics_{ts}.csv"
            try:
                self._create_log_header()
            except Exception:
                self.log_path = None

    def _create_log_header(self):
        if not self.log_path:
            return
        headers = [
            # timestamp + core
            "timestamp", "timestep", "episode", "meta_reward", "investment_freq", "capital_fraction",
            # actions
            "meta_action_0", "meta_action_1", "inv_action_0", "inv_action_1", "inv_action_2",
            "batt_action_0", "risk_action_0",
            # immediate forecasts (raw keys, then normalized/aligned adds)
            "wind_forecast_immediate", "solar_forecast_immediate", "hydro_forecast_immediate", "price_forecast_immediate", "load_forecast_immediate",
            # normalized/aligned extras
            "price_forecast_norm", "price_forecast_aligned",
            # perf & quick risks (financial values in USD for clarity)
            "portfolio_performance", "portfolio_value_usd", "equity_usd", "total_return_nav_usd", "overall_risk", "market_risk",
            # env snapshot (budget in USD, capacities in MW/MWh)
            "budget_usd", "wind_cap", "solar_cap", "hydro_cap", "battery_energy",
            "price_actual", "load_actual", "wind_actual", "solar_actual", "hydro_actual",
            "market_stress", "market_volatility", "revenue_step_usd",
            # NEW economics fields (all financial values in USD)
            "generation_revenue_usd", "mtm_pnl_usd", "distributed_profits_usd", "cumulative_returns_usd", "investment_capital_usd", "fund_performance",
            # episode markers
            "ep_meta_return", "step_in_episode", "episode_end",
            # 6D risk vector
            "risk_market", "risk_gen_var", "risk_portfolio", "risk_liquidity", "risk_stress", "risk_overall",
            # reward components + weights
            "reward_financial", "reward_risk", "reward_sustainability", "reward_efficiency", "reward_diversification",
            "w_financial", "w_risk", "w_sustainability", "w_efficiency", "w_diversification",
            # ops & scaling health
            "episode_id", "seed", "step_time_ms", "mem_rss_mb",
            "action_clip_frac_investor", "action_clip_frac_battery", "action_clip_frac_risk", "action_clip_frac_meta",
            # MAE (1-step abs error vs previous logged forecast)
            "mae_price_1", "mae_wind_1", "mae_solar_1", "mae_load_1", "mae_hydro_1",
            # forecast signal usefulness diagnostic
            "forecast_alignment_score"
        ]
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

    # ---- properties / API ----
    @property
    def possible_agents(self): return self._possible_agents

    @property
    def agents(self): return self._agents

    @agents.setter
    def agents(self, value): self._agents = value[:]

    def observation_space(self, agent): return self._obs_spaces[agent]

    def action_space(self, agent): return self.env.action_space(agent)

    @property
    def observation_spaces(self): return {a: self.observation_space(a) for a in self.possible_agents}

    @property
    def action_spaces(self): return {a: self.action_space(a) for a in self.possible_agents}

    @property
    def t(self): return getattr(self.env, "t", 0)

    @property
    def max_steps(self): return getattr(self.env, "max_steps", 1000)

    # ---- verification methods ----
    def _verify_capacity_consistency(self):
        """Verify wrapper is using correct capacity values from environment"""
        try:
            env_ref = getattr(self, "env", self)
            
            # Check physical capacities exist and are reasonable
            wind_mw = getattr(env_ref, 'wind_capacity_mw', 0.0)
            solar_mw = getattr(env_ref, 'solar_capacity_mw', 0.0)
            hydro_mw = getattr(env_ref, 'hydro_capacity_mw', 0.0)
            
            total_mw = wind_mw + solar_mw + hydro_mw

            # Dynamic capacity check based on budget size
            try:
                budget = getattr(env_ref, 'init_budget', 500_000_000)
                # Expect roughly 1 MW per $2M budget (conservative estimate)
                expected_min_mw = budget / 2_000_000

                if total_mw < expected_min_mw * 0.1:  # Allow very low deployment (10% of expected)
                    logging.warning(f"Very low capacity deployment: {total_mw:.1f}MW (expected ~{expected_min_mw:.0f}MW for ${budget:,.0f} budget)")
                    return False
            except Exception:
                # Fallback to original check for small funds
                if total_mw < 5:  # Minimum 5MW for any meaningful operation
                    logging.warning(f"Insufficient capacity deployment: {total_mw:.1f}MW")
                    return False
                
            return True
        except Exception as e:
            logging.error(f"Capacity verification failed: {e}")
            return False

    def _safe_portfolio_value(self, raw_value, initial_budget):
        """Apply safety bounds to portfolio values"""
        min_portfolio = initial_budget * 0.01  # Minimum 1% of initial fund
        max_portfolio = initial_budget * 3.0   # Maximum 300% of initial fund
        return float(np.clip(raw_value, min_portfolio, max_portfolio))

    # ---- core loop ----
    def reset(self, seed=None, options=None):
        self._cleanup_memory_enhanced(force=True)
        self._last_episode_seed = seed
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.episode_count += 1
        self.error_count = 0
        self._ep_meta_return = 0.0
        self._prev_forecasts_for_error = {}
        self._last_price_forecast_norm = 0.0
        self._last_price_forecast_aligned = 0.0

        # Initialize forecaster history with sufficient data for predictions
        try:
            if hasattr(self.env, "data") and len(self.env.data) > 0:
                if hasattr(self.forecaster, "initialize_history"):
                    self.forecaster.initialize_history(self.env.data, start_idx=0)
                elif hasattr(self.forecaster, "update"):
                    self.forecaster.update(self.env.data.iloc[0])
        except Exception:
            pass

        enhanced = self.obs_builder.enhance_observations(obs)
        validated = self._validate_observations_safe(enhanced)
        return validated, info

    def step(self, actions: Dict[str, Any]):
        t0 = time.perf_counter()

        # validate & track clipping
        actions, clip_counts, clip_totals = self._validate_actions_comprehensive(actions, track_clipping=True)
        self._last_clip_counts = clip_counts
        self._last_clip_totals = clip_totals

        obs, rewards, dones, truncs, infos = self.env.step(actions)
        self._last_episode_end_flag = 1 if any(bool(x) for x in dones.values()) else 0
        self._ep_meta_return += float(rewards.get("meta_controller_0", 0.0) or 0.0)

        # update forecaster
        self._update_forecaster_safe()

        # build enhanced obs with full forecasting
        enhanced = self.obs_builder.enhance_observations(obs)
        validated = self._validate_observations_safe(enhanced)

        self.step_count += 1
        step_time = (time.perf_counter() - t0)
        self._last_step_wall_ms = step_time * 1000.0

        # No fast mode progress display - removed

        # Enhanced monitoring (optional)
        if self.enhanced_monitor:
            self.enhanced_monitor.update_system_metrics(step_time)
            for agent, reward in rewards.items():
                self.enhanced_monitor.update_training_metrics(agent, reward)
            if hasattr(self, '_last_portfolio_value') and self._last_portfolio_value is not None:
                self.enhanced_monitor.update_portfolio_metrics(self._last_portfolio_value)
            self.enhanced_monitor.log_summary()

        # FIXED: Populate forecast arrays for DL overlay labeler
        try:
            if hasattr(self.env, 'populate_forecast_arrays'):
                current_forecasts = self._get_forecasts_for_logging()
                self.env.populate_forecast_arrays(getattr(self.env, 't', 0), current_forecasts)
        except Exception:
            pass  # Don't break main flow if forecast population fails

        # periodic logging
        if self.log_path and self._should_log_this_step() and not self.disable_csv_logging:
            if (self.step_count % self._log_interval == 0 or self._last_episode_end_flag):
                log_forecasts = self._get_forecasts_for_logging()
                self._log_metrics_efficient(actions, rewards, log_forecasts)

        if self._last_episode_end_flag:
            self._prev_forecasts_for_error = {}

        return validated, rewards, dones, truncs, infos

    # ---- helpers ----
    def _validate_observations_safe(self, obs_dict):
        validated = {}
        for agent in self.possible_agents:
            try:
                if agent in obs_dict:
                    obs = obs_dict[agent]
                    expected = self.observation_space(agent).shape
                    if obs.shape != expected:
                        if self.error_count < self.max_errors:
                            print(f"⚠️ Obs shape mismatch for {agent}: {obs.shape} vs {expected}")
                        self.error_count += 1
                        obs = self.obs_builder.validator._create_safe_observation(agent)
                    validated[agent] = obs.astype(np.float32)
                else:
                    validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
            except Exception as e:
                if self.error_count < self.max_errors:
                    print(f"⚠️ Obs validation failed for {agent}: {e}")
                self.error_count += 1
                validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
        return validated

    def _cleanup_memory_enhanced(self, force=False):
        try:
            cleanup_level, _ = self.memory_tracker.should_cleanup(force=force)
            if cleanup_level or force:
                self.memory_tracker.cleanup(cleanup_level or 'medium')
                # Enhanced cache cleanup with LRU
                if hasattr(self.obs_builder, 'forecast_cache') and len(self.obs_builder.forecast_cache) > 500:
                    self.obs_builder.forecast_cache._memory_cleanup()
                if hasattr(self.obs_builder, 'agent_forecast_cache') and len(self.obs_builder.agent_forecast_cache) > 1000:
                    self.obs_builder.agent_forecast_cache._memory_cleanup()
                self._flush_log_buffer()
                if hasattr(self.obs_builder.validator, 'validation_cache'):
                    self.obs_builder.validator.validation_cache.clear()
                gc.collect()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Memory cleanup failed: {e}")
                self.error_count += 1

    def _to_numpy_safe(self, a_in):
        """
        Brings various tensor types to a CPU numpy array without importing heavy libs.
        Handles torch.Tensor, JAX DeviceArray, etc., by duck-typing.
        """
        try:
            # torch: .detach().cpu().numpy()
            if hasattr(a_in, "detach") and callable(a_in.detach):
                try:
                    a_in = a_in.detach()
                except Exception:
                    pass
            if hasattr(a_in, "cpu") and callable(a_in.cpu):
                try:
                    a_in = a_in.cpu()
                except Exception:
                    pass
            if hasattr(a_in, "numpy") and callable(a_in.numpy):
                try:
                    a_in = a_in.numpy()
                except Exception:
                    pass
        except Exception:
            pass
        return a_in

    def _validate_actions_comprehensive(self, actions, track_clipping=False):
        validated = {}
        clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}
        clip_totals = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        for agent in self.possible_agents:
            space = self.env.action_space(agent)
            a_in = actions.get(agent, None)

            # Bring tensors to numpy if needed (CUDA-safe)
            if a_in is not None:
                a_in = self._to_numpy_safe(a_in)

            # Discrete
            if isinstance(space, spaces.Discrete):
                if isinstance(a_in, np.ndarray):
                    val = int(np.atleast_1d(a_in).flatten()[0])
                elif np.isscalar(a_in):
                    val = int(a_in)
                elif isinstance(a_in, (list, tuple)) and len(a_in) > 0:
                    val = int(a_in[0])
                else:
                    val = 0
                val_clipped = int(np.clip(val, 0, space.n - 1))
                validated[agent] = val_clipped
                if track_clipping:
                    key = ("investor" if agent.startswith("investor")
                           else "battery" if agent.startswith("battery")
                           else "risk" if agent.startswith("risk")
                           else "meta")
                    clip_totals[key] += 1
                    clip_counts[key] += int(val != val_clipped)
                continue

            # Box → float array with shape
            if not isinstance(a_in, np.ndarray):
                if np.isscalar(a_in):
                    arr = np.array([a_in], dtype=np.float32)
                elif isinstance(a_in, (list, tuple)):
                    arr = np.array(a_in, dtype=np.float32)
                else:
                    # default to mid-point
                    middle = (space.low + space.high) / 2.0
                    m = float(middle.flatten()[0]) if hasattr(middle, 'flatten') else float(middle)
                    arr = np.array([m], dtype=np.float32)
            else:
                arr = a_in.astype(np.float32)

            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim > 1:
                arr = arr.flatten()

            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            need = int(np.prod(space.shape))
            if arr.size != need:
                if arr.size < need:
                    middle = (space.low + space.high) / 2.0
                    m = float(middle.flatten()[0]) if hasattr(middle, 'flatten') else float(middle)
                    arr = np.concatenate([arr, np.full(need - arr.size, m, dtype=np.float32)])
                else:
                    arr = arr[:need]

            before = arr.copy()
            arr = np.minimum(np.maximum(arr, space.low), space.high)
            validated[agent] = arr.astype(np.float32)

            if track_clipping:
                key = ("investor" if agent.startswith("investor")
                       else "battery" if agent.startswith("battery")
                       else "risk" if agent.startswith("risk")
                       else "meta")
                clip_totals[key] += arr.size
                clip_counts[key] += int(np.sum(np.abs(before - arr) > 1e-12))

        return validated, clip_counts, clip_totals

    def _update_forecaster_safe(self):
        """Update forecaster with current timestep data to avoid forecast lag."""
        try:
            if hasattr(self.env, "data") and self.env.t < len(self.env.data):
                row = self.env.data.iloc[self.env.t]
                if hasattr(self.forecaster, "update"):
                    self.forecaster.update(row)
        except Exception:
            pass

    # ---- logging helpers ----
    def _safe_float(self, v, default=0.0):
        try:
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v).reshape(-1)
                v = v[0] if v.size > 0 else default
            return float(v) if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    def _get_actuals_for_logging(self) -> Dict[str, float]:
        out = {'wind': 0.0, 'solar': 0.0, 'hydro': 0.0, 'price': 0.0, 'load': 0.0}
        try:
            if hasattr(self.env, "data") and isinstance(self.env.data, pd.DataFrame) and len(self.env.data) > 0:
                idx = max(0, min(getattr(self.env, "t", 0) - 1, len(self.env.data) - 1))
                row = self.env.data.iloc[idx]
                for k in out.keys():
                    if k in row:
                        out[k] = self._safe_float(row[k], 0.0)
        except Exception:
            pass
        return out

    def _get_timestamp_for_logging(self) -> str:
        try:
            if hasattr(self.env, "data") and isinstance(self.env.data, pd.DataFrame):
                if "timestamp" in self.env.data.columns and len(self.env.data) > 0:
                    idx = max(0, min(getattr(self.env, "t", 0) - 1, len(self.env.data) - 1))
                    ts = pd.to_datetime(self.env.data.iloc[idx]["timestamp"], errors="coerce")
                    if pd.notna(ts):
                        return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        return ""

    def _get_forecasts_for_logging(self) -> Dict[str, float]:
        try:
            f = self.forecaster.predict_all_horizons(timestep=getattr(self.env, 't', 0))
            return f if isinstance(f, dict) else {}
        except Exception:
            return {}

    def _get_price_forecast_norm_and_aligned(self, all_forecasts: Dict[str, float]) -> Tuple[float, float]:
        """
        Returns (price_forecast_norm, price_forecast_aligned_norm).
        We compute aligned as the forecast with horizon closest to env.investment_freq.
        """
        pf_norm = 0.0
        pf_aligned = 0.0
        try:
            price_items = [(k, all_forecasts[k]) for k in all_forecasts.keys() if "price_forecast_" in str(k).lower()]
            if price_items:
                cand = []
                for k, v in price_items:
                    n = self.obs_builder.postproc.normalize_value(k, v)
                    cand.append((k, n))
                    if pf_norm == 0.0:
                        pf_norm = float(n)

                invf = int(getattr(self.env, "investment_freq", 12))

                def dist(k):
                    try:
                        h = int(str(k).split("_")[-1])
                        return abs(h - invf)
                    except Exception:
                        return 10**9

                k_best, v_best = min(cand, key=lambda kv: dist(kv[0]))
                pf_aligned = float(v_best)
        except Exception:
            pass
        return pf_norm, pf_aligned

    def _get_risk_vector6(self) -> List[float]:
        default = [0.5, 0.2, 0.25, 0.15, 0.35, 0.25]
        try:
            if hasattr(self.env, 'enhanced_risk_controller') and self.env.enhanced_risk_controller:
                arr = np.asarray(self.env.enhanced_risk_controller.get_risk_metrics_for_observation())
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                if arr.size >= 6:
                    return [float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else d for x, d in zip(arr[:6], default)]
        except Exception:
            pass
        return default

    def _calc_perf_and_quick_risks(self) -> List[float]:
        """
        PATCH: Compute performance from true MTM equity (preferred), not capacity proxy.
        Returns [perf, overall_risk, market_risk].
        """
        try:
            initial_budget = float(getattr(self.env, 'init_budget', 1e7))
            initial_budget = max(initial_budget, 1.0)

            if hasattr(self.env, 'equity'):
                portfolio_value = float(self._safe_float(getattr(self.env, 'equity', initial_budget), initial_budget))
            else:
                cash = float(self._safe_float(getattr(self.env, 'budget', initial_budget), initial_budget))
                w_val = float(self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0))
                s_val = float(self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0))
                h_val = float(self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0))
                portfolio_value = cash + w_val + s_val + h_val

            # Apply safety bounds using the new helper method
            portfolio_value = self._safe_portfolio_value(portfolio_value, initial_budget)

            perf = float(SafeDivision._safe_divide(portfolio_value, initial_budget, 1.0))
            perf_clipped = float(np.clip(perf, 0.0, 10.0))
            overall_risk = float(np.clip(getattr(self.env, "overall_risk_snapshot", 0.5), 0.0, 1.0))
            market_risk = float(np.clip(getattr(self.env, "market_risk_snapshot", 0.5), 0.0, 1.0))
            self._last_portfolio_value = portfolio_value
            return [perf_clipped, overall_risk, market_risk]
        except Exception:
            self._last_portfolio_value = None
            return [1.0, 0.5, 0.3]

    def _log_metrics_efficient(self, actions: Dict[str, Any], rewards: Dict[str, float],
                               all_forecasts: Dict[str, float]):
        if not self.log_path:
            return
        try:
            with self.log_lock:
                ts_str = self._get_timestamp_for_logging()
                step_t = getattr(self.env, 't', self.step_count)

                # --- START: CURRENCY & METRIC SETUP ---
                # Get the single source of truth for currency conversion from the environment
                dkk_to_usd_rate = getattr(self.env, '_dkk_to_usd_rate', 0.145)

                # Fetch core financial values from the environment (all are in DKK at this stage)
                # These will be converted to USD before logging.
                budget_dkk = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                generation_revenue_dkk = self._safe_float(getattr(self.env, 'last_generation_revenue', 0.0), 0.0)
                revenue_step_dkk = self._safe_float(getattr(self.env, 'last_revenue', getattr(self.env, 'revenue', 0.0)), 0.0)
                mtm_pnl_dkk = self._safe_float(getattr(self.env, 'last_mtm_pnl', 0.0), 0.0)
                distributed_profits_dkk = self._safe_float(getattr(self.env, 'distributed_profits', 0.0), 0.0)
                cumulative_returns_dkk = self._safe_float(getattr(self.env, 'cumulative_returns', 0.0), 0.0)
                investment_capital_dkk = self._safe_float(getattr(self.env, 'investment_capital', 0.0), 0.0)
                # --- END: CURRENCY & METRIC SETUP ---

                # core (non-monetary)
                meta_reward = self._safe_float(rewards.get("meta_controller_0", 0.0), 0.0)
                inv_freq = int(getattr(self.env, "investment_freq", -1))
                cap_frac = self._safe_float(getattr(self.env, "capital_allocation_fraction", -1), -1.0)

                # actions
                def get_action_vec(agent, n):
                    a = actions.get(agent, None)
                    a = self._to_numpy_safe(a) # convert tensors
                    if isinstance(a, np.ndarray): vec = a.flatten().tolist()
                    elif np.isscalar(a): vec = [float(a)]
                    elif isinstance(a, (list, tuple)): vec = list(a)
                    else: vec = []
                    out = [float(v) if np.isfinite(v) else 0.0 for v in vec]
                    while len(out) < n: out.append(0.0)
                    return out[:n]

                meta_a0, meta_a1 = get_action_vec("meta_controller_0", 2)
                inv_a0, inv_a1, inv_a2 = get_action_vec("investor_0", 3)
                batt_a0 = get_action_vec("battery_operator_0", 1)[0]
                risk_a0 = get_action_vec("risk_controller_0", 1)[0]

                # Forecasts (non-monetary)
                forecast_keys = ["wind_forecast_immediate", "solar_forecast_immediate", "hydro_forecast_immediate", "price_forecast_immediate", "load_forecast_immediate"]
                forecasts_logged = [self._safe_float(all_forecasts.get(k, 0.0), 0.0) for k in forecast_keys]
                pf_norm, pf_aligned = self._get_price_forecast_norm_and_aligned(all_forecasts)
                self._last_price_forecast_norm = pf_norm
                self._last_price_forecast_aligned = pf_aligned

                # perf & quick risks (ratios/scores)
                perf, overall_risk, market_risk = self._calc_perf_and_quick_risks()

                # Get portfolio value/equity (in DKK from env)
                if hasattr(self, '_last_portfolio_value') and self._last_portfolio_value is not None:
                    portfolio_value_dkk = float(self._last_portfolio_value)
                else:
                    cash_dkk = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                    w_val_dkk = self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0)
                    s_val_dkk = self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0)
                    h_val_dkk = self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0)
                    portfolio_value_dkk = cash_dkk + w_val_dkk + s_val_dkk + h_val_dkk
                    initial_budget_dkk = float(getattr(self.env, 'init_budget', 500_000_000))
                    portfolio_value_dkk = self._safe_portfolio_value(portfolio_value_dkk, initial_budget_dkk)
                
                # --- START: CONSISTENT USD CONVERSION FOR LOGGING ---
                # CURRENCY APPROACH: All internal calculations in DKK, convert to USD only for CSV output
                # This provides clear $USD values in reports while maintaining DKK precision internally
                portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
                equity_usd = portfolio_value_usd # Equity is the portfolio value in this context
                distributed_profits_usd = distributed_profits_dkk * dkk_to_usd_rate
                total_return_nav_usd = portfolio_value_usd + distributed_profits_usd
                budget_usd = budget_dkk * dkk_to_usd_rate
                
                generation_revenue_usd = generation_revenue_dkk * dkk_to_usd_rate
                revenue_step_usd = revenue_step_dkk * dkk_to_usd_rate
                mtm_pnl_usd = mtm_pnl_dkk * dkk_to_usd_rate
                cumulative_returns_usd = cumulative_returns_dkk * dkk_to_usd_rate
                investment_capital_usd = investment_capital_dkk * dkk_to_usd_rate
                # fund_performance is a ratio, no conversion needed
                fund_performance = self._safe_float(getattr(self.env, 'fund_performance', 0.0), 0.0)
                # --- END: CONSISTENT USD CONVERSION FOR LOGGING ---

                # env snapshot (physical capacities, non-monetary)
                wind_c = self._safe_float(getattr(self.env, 'wind_capacity_mw', 0.0), 0.0)
                solar_c = self._safe_float(getattr(self.env, 'solar_capacity_mw', 0.0), 0.0)
                hydro_c = self._safe_float(getattr(self.env, 'hydro_capacity_mw', 0.0), 0.0)
                batt_e = self._safe_float(getattr(self.env, 'battery_energy', 0.0), 0.0)
                actuals = self._get_actuals_for_logging()
                price_act = self._safe_float(actuals['price'], 0.0)
                load_act = self._safe_float(actuals['load'], 0.0)
                wind_act = self._safe_float(actuals['wind'], 0.0)
                solar_act = self._safe_float(actuals['solar'], 0.0)
                hydro_act = self._safe_float(actuals['hydro'], 0.0)
                market_stress = self._safe_float(getattr(self.env, 'market_stress', 0.0), 0.0)
                market_vol = self._safe_float(getattr(self.env, 'market_volatility', 0.0), 0.0)

                # episode markers & risk (non-monetary)
                step_in_ep = int(getattr(self.env, 'step_in_episode', self.step_count))
                episode_end = int(self._last_episode_end_flag)
                r6 = self._get_risk_vector6()

                # reward breakdown & weights (scaled scores, non-monetary)
                rb = getattr(self.env, "last_reward_breakdown", {}) or {}
                rw = getattr(self.env, "last_reward_weights", {}) or {}
                r_fin = self._safe_float(rb.get("financial", 0.0), 0.0)
                r_risk = self._safe_float(rb.get("risk_management", 0.0), 0.0)
                r_sus = self._safe_float(rb.get("sustainability", 0.0), 0.0)
                r_eff = self._safe_float(rb.get("efficiency", 0.0), 0.0)
                r_div = self._safe_float(rb.get("diversification", 0.0), 0.0)
                w_fin = self._safe_float(rw.get("financial", 0.0), 0.0)
                w_rsk = self._safe_float(rw.get("risk_management", 0.0), 0.0)
                w_sus = self._safe_float(rw.get("sustainability", 0.0), 0.0)
                w_eff = self._safe_float(rw.get("efficiency", 0.0), 0.0)
                w_div = self._safe_float(rw.get("diversification", 0.0), 0.0)

                # ops & scaling health (non-monetary)
                mem_mb = float(self.memory_tracker.get_memory_usage())
                seed = self._last_episode_seed if self._last_episode_seed is not None else -1
                step_ms = float(self._last_step_wall_ms)
                def clip_frac(key):
                    tot = max(1, self._last_clip_totals.get(key, 1))
                    return float(SafeDivision._safe_divide(self._last_clip_counts.get(key, 0), tot, 0.0))
                clip_inv, clip_bat, clip_rsk, clip_meta = clip_frac("investor"), clip_frac("battery"), clip_frac("risk"), clip_frac("meta")

                # MAE@1 using previous forecast cache (non-monetary)
                def _mae(prev, actual): return abs(self._safe_float(prev, 0.0) - self._safe_float(actual, 0.0))
                mae_price = _mae(self._prev_forecasts_for_error.get("price_forecast_immediate"), price_act)
                mae_wind = _mae(self._prev_forecasts_for_error.get("wind_forecast_immediate"), wind_act)
                mae_solar = _mae(self._prev_forecasts_for_error.get("solar_forecast_immediate"), solar_act)
                mae_load = _mae(self._prev_forecasts_for_error.get("load_forecast_immediate"), load_act)
                mae_hydro = _mae(self._prev_forecasts_for_error.get("hydro_forecast_immediate"), hydro_act)
                self._prev_forecasts_for_error = {k: v for k, v in zip(forecast_keys, forecasts_logged)}

                # Forecast alignment score diagnostic (non-monetary)
                try:
                    price_scale_fallback = getattr(getattr(self.env, 'config', object()), 'price_scale', 10.0)
                    cur_price_norm = float(np.clip(self._safe_float(actuals['price'], 0.0) / price_scale_fallback, -10.0, 10.0))
                    realized_ret_dummy = 0.0 # Placeholder as real return calc is complex here
                    forecast_alignment_score = float(np.sign(self._last_price_forecast_aligned - cur_price_norm) * realized_ret_dummy)
                except Exception:
                    forecast_alignment_score = 0.0

                # --- Assemble the final log row with all monetary values in USD ---
                row = [
                    ts_str, int(step_t), int(self.episode_count), meta_reward, inv_freq, cap_frac,
                    meta_a0, meta_a1, inv_a0, inv_a1, inv_a2, batt_a0, risk_a0,
                    *forecasts_logged,
                    self._last_price_forecast_norm, self._last_price_forecast_aligned,
                    perf, portfolio_value_usd, equity_usd, total_return_nav_usd, overall_risk, market_risk,
                    budget_usd, wind_c, solar_c, hydro_c, batt_e,
                    price_act, load_act, wind_act, solar_act, hydro_act,
                    market_stress, market_vol, revenue_step_usd,
                    generation_revenue_usd, mtm_pnl_usd, distributed_profits_usd, cumulative_returns_usd, investment_capital_usd, fund_performance,
                    self._ep_meta_return, step_in_ep, episode_end,
                    *r6,
                    r_fin, r_risk, r_sus, r_eff, r_div,
                    w_fin, w_rsk, w_sus, w_eff, w_div,
                    self.episode_count, seed, step_ms, mem_mb,
                    clip_inv, clip_bat, clip_rsk, clip_meta,
                    mae_price, mae_wind, mae_solar, mae_load, mae_hydro,
                    forecast_alignment_score
                ]

                self.log_buffer.append(row)
                if len(self.log_buffer) >= 25 or self.step_count % 500 == 0 or len(self.log_buffer) >= self._flush_every_rows or episode_end:
                    self._flush_log_buffer()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Metrics logging failed: {e}")
                self.error_count += 1

    def _flush_log_buffer(self):
        """Safely flushes the in-memory log buffer to the CSV file."""
        if not self.log_path or not self.log_buffer:
            return
        try:
            with self.log_lock, open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                while self.log_buffer:
                    writer.writerow(self.log_buffer.popleft())
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Failed to flush log buffer: {e}")
                self.error_count += 1

    # ---- debug methods ----
    def debug_wrapper_data_sources(self):
        """Debug what data sources the wrapper is using"""
        env_ref = getattr(self, "env", self)
        
        print("Wrapper Data Sources:")
        print(f"  Physical Wind: {getattr(env_ref, 'wind_capacity_mw', 'MISSING')}MW")
        print(f"  Financial Wind: ${getattr(env_ref, 'wind_instrument_value', 'MISSING'):,.0f}")
        print(f"  Budget: ${getattr(env_ref, 'budget', 'MISSING'):,.0f}")
        print(f"  Equity: ${getattr(env_ref, 'equity', 'MISSING'):,.0f}")

    # ---- passthroughs ----
    def render(self):
        try:
            if hasattr(self.env, "render"):
                return self.env.render()
        except Exception:
            return None

    def close(self):
        try:
            # Save final results if CSV logging was disabled but final results are requested
            if self.disable_csv_logging and hasattr(self, 'log_path') and self.log_path:
                self._save_final_results_only()
            else:
                self._flush_log_buffer()
        finally:
            try:
                if hasattr(self.env, "close"):
                    self.env.close()
            except Exception:
                pass

    def _save_final_results_only(self):
        """Save only the final timestep results to CSV when disable_csv_logging=True."""
        if not self.log_path:
            return

        try:
            # Create header if file doesn't exist
            if not os.path.isfile(self.log_path):
                self._create_log_header()

            # Get final state and log it
            final_actions = {}  # Empty actions for final state
            final_rewards = {}  # Empty rewards for final state
            final_forecasts = self._get_forecasts_for_logging()

            print(f"💾 Saving final results to: {self.log_path}")
            self._log_metrics_efficient(final_actions, final_rewards, final_forecasts)
            self._flush_log_buffer()

        except Exception as e:
            print(f"⚠️ Failed to save final results: {e}")

    # alias PettingZoo naming if needed
    def state(self):
        try:
            if hasattr(self.env, "state"):
                return self.env.state()
        except Exception:
            return None


class BaselineCSVWrapper(ParallelEnv):
    """Baseline wrapper that adds enhanced CSV logging without forecasting overhead."""

    def __init__(self, base_env, log_path=None, total_timesteps=50000, log_last_n=100):
        self.env = base_env
        self.total_timesteps = total_timesteps
        self.log_last_n = log_last_n

        # Enhanced logging setup (adapted from MultiHorizonWrapperEnv)
        self.log_path = log_path
        self.log_buffer = deque(maxlen=256)
        self.log_lock = threading.RLock() if threading else nullcontext()
        self._flush_every_rows = 1000
        self.step_count = 0

        # Enhanced metrics tracking (adapted from MultiHorizonWrapperEnv)
        self.episode_count = 0
        self._last_episode_end_flag = False
        self._last_episode_seed = None
        self._last_step_wall_ms = 0.0
        self._last_clip_counts = {}
        self._last_clip_totals = {}
        self._last_portfolio_value = None
        self._ep_meta_return = 0.0
        self.error_count = 0
        self.max_errors = 10

        # Initialize CSV file
        if self.log_path:
            self._initialize_logging_safe()

        # Copy attributes from base environment
        self.metadata = getattr(base_env, 'metadata', {})
        self.possible_agents = base_env.possible_agents
        self.agents = base_env.agents
        self.observation_spaces = base_env.observation_spaces
        self.action_spaces = base_env.action_spaces

    def _initialize_logging_safe(self):
        """Initialize logging based on mode (adapted from MultiHorizonWrapperEnv)."""
        if not self.log_path:
            return

        # Create log file immediately (normal mode)
        try:
            if not os.path.isfile(self.log_path):
                self._create_log_header()
            print(f"✅ Baseline logging initialized: {self.log_path}")
        except Exception as e:
            print(f"⚠️ Baseline logging initialization failed: {e}")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"emergency_baseline_metrics_{ts}.csv"
            try:
                self._create_log_header()
            except Exception:
                self.log_path = None

    def _create_log_header(self):
        """Create CSV header with enhanced metrics (no forecasts)."""
        headers = [
            # timestamp + core
            "timestamp", "timestep", "episode", "meta_reward", "investment_freq", "capital_fraction",
            # actions
            "meta_action_0", "meta_action_1", "inv_action_0", "inv_action_1", "inv_action_2",
            "batt_action_0", "risk_action_0",
            # performance & portfolio (all USD)
            "performance", "portfolio_value_usd", "equity_usd", "total_return_nav_usd", "overall_risk", "market_risk",
            # env snapshot (all USD)
            "budget_usd", "wind_cap", "solar_cap", "hydro_cap", "battery_energy",
            # actuals
            "price_actual", "load_actual", "wind_actual", "solar_actual", "hydro_actual",
            # market regime
            "market_stress", "market_volatility", "revenue_step_usd",
            # financial breakdown (all USD)
            "generation_revenue_usd", "mtm_pnl_usd", "distributed_profits_usd", "cumulative_returns_usd", "investment_capital_usd", "fund_performance",
            # episode markers
            "ep_meta_return", "step_in_episode", "episode_end",
            # risk vector 6D
            "risk_0", "risk_1", "risk_2", "risk_3", "risk_4", "risk_5",
            # reward breakdown
            "reward_financial", "reward_risk", "reward_sustainability", "reward_efficiency", "reward_diversification",
            "w_financial", "w_risk", "w_sustainability", "w_efficiency", "w_diversification",
            # ops & scaling health
            "episode_id", "seed", "step_time_ms", "mem_rss_mb",
            "action_clip_frac_investor", "action_clip_frac_battery", "action_clip_frac_risk", "action_clip_frac_meta"
        ]
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

    # Helper methods adapted from MultiHorizonWrapperEnv
    def _safe_float(self, val, default=0.0):
        """Safely convert value to float."""
        try:
            if val is None or not np.isfinite(float(val)):
                return default
            return float(val)
        except (ValueError, TypeError, OverflowError):
            return default

    def _to_numpy_safe(self, val):
        """Safely convert to numpy array."""
        if hasattr(val, 'detach'):  # PyTorch tensor
            val = val.detach().cpu().numpy()
        elif hasattr(val, 'numpy'):  # TensorFlow tensor
            val = val.numpy()
        return val

    def _get_timestamp_for_logging(self):
        """Get timestamp for logging."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _safe_portfolio_value(self, portfolio_value, initial_budget):
        """Safely calculate portfolio value with bounds checking."""
        try:
            if portfolio_value <= 0 or not np.isfinite(portfolio_value):
                return initial_budget
            # Reasonable bounds check
            if portfolio_value > initial_budget * 100 or portfolio_value < initial_budget * 0.01:
                return initial_budget
            return portfolio_value
        except Exception:
            return initial_budget

    def _calc_perf_and_quick_risks(self):
        """Calculate performance and quick risk metrics using proper equity calculation."""
        try:
            # Get the single source of truth for currency conversion from the environment
            dkk_to_usd_rate = getattr(self.env, '_dkk_to_usd_rate', 0.145)
            initial_budget = float(getattr(self.env, 'init_budget', 500_000_000))

            # Get portfolio value/equity (in DKK from env) - FIXED: Use proper equity calculation
            portfolio_value_dkk = getattr(self.env, 'equity', None)
            if portfolio_value_dkk is None or portfolio_value_dkk == 0:
                # Try to calculate NAV if equity is not available
                if hasattr(self.env, '_calculate_fund_nav'):
                    try:
                        portfolio_value_dkk = self.env._calculate_fund_nav()
                    except Exception:
                        # Fallback to manual calculation
                        cash_dkk = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                        w_val_dkk = self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0)
                        s_val_dkk = self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0)
                        h_val_dkk = self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0)
                        portfolio_value_dkk = cash_dkk + w_val_dkk + s_val_dkk + h_val_dkk
                else:
                    # Fallback to manual calculation
                    cash_dkk = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                    w_val_dkk = self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0)
                    s_val_dkk = self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0)
                    h_val_dkk = self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0)
                    portfolio_value_dkk = cash_dkk + w_val_dkk + s_val_dkk + h_val_dkk

            # Apply safety bounds using the new helper method
            portfolio_value = self._safe_portfolio_value(portfolio_value_dkk, initial_budget)

            perf = float(SafeDivision._safe_divide(portfolio_value, initial_budget, 1.0))
            perf_clipped = float(np.clip(perf, 0.0, 10.0))
            overall_risk = float(np.clip(getattr(self.env, "overall_risk_snapshot", 0.5), 0.0, 1.0))
            market_risk = float(np.clip(getattr(self.env, "market_risk_snapshot", 0.5), 0.0, 1.0))
            self._last_portfolio_value = portfolio_value
            return [perf_clipped, overall_risk, market_risk]
        except Exception:
            self._last_portfolio_value = None
            return [1.0, 0.5, 0.3]

    def _log_metrics_efficient(self, actions: Dict[str, Any], rewards: Dict[str, float]):
        """Enhanced logging method adapted from MultiHorizonWrapperEnv (no forecasts)."""
        if not self.log_path:
            return
        try:
            with self.log_lock:
                ts_str = self._get_timestamp_for_logging()
                step_t = getattr(self.env, 't', self.step_count)

                # --- START: CURRENCY & METRIC SETUP ---
                # Get the single source of truth for currency conversion from the environment
                dkk_to_usd_rate = getattr(self.env, '_dkk_to_usd_rate', 0.145)

                # --- ACTIONS ---
                meta_reward = self._safe_float(rewards.get("meta_agent", 0.0), 0.0)
                inv_freq = self._safe_float(getattr(self.env, "investment_freq", 6), 6)
                cap_frac = self._safe_float(getattr(self.env, "capital_allocation_fraction", 0.1), 0.1)

                # Extract actions safely
                meta_a0 = self._safe_float(actions.get("meta_agent", [0, 0])[0] if isinstance(actions.get("meta_agent"), (list, np.ndarray)) and len(actions.get("meta_agent", [])) > 0 else 0, 0.0)
                meta_a1 = self._safe_float(actions.get("meta_agent", [0, 0])[1] if isinstance(actions.get("meta_agent"), (list, np.ndarray)) and len(actions.get("meta_agent", [])) > 1 else 0, 0.0)
                inv_a0 = self._safe_float(actions.get("investor_agent", [0, 0, 0])[0] if isinstance(actions.get("investor_agent"), (list, np.ndarray)) and len(actions.get("investor_agent", [])) > 0 else 0, 0.0)
                inv_a1 = self._safe_float(actions.get("investor_agent", [0, 0, 0])[1] if isinstance(actions.get("investor_agent"), (list, np.ndarray)) and len(actions.get("investor_agent", [])) > 1 else 0, 0.0)
                inv_a2 = self._safe_float(actions.get("investor_agent", [0, 0, 0])[2] if isinstance(actions.get("investor_agent"), (list, np.ndarray)) and len(actions.get("investor_agent", [])) > 2 else 0, 0.0)
                batt_a0 = self._safe_float(actions.get("battery_agent", [0])[0] if isinstance(actions.get("battery_agent"), (list, np.ndarray)) and len(actions.get("battery_agent", [])) > 0 else 0, 0.0)
                risk_a0 = self._safe_float(actions.get("risk_agent", [0])[0] if isinstance(actions.get("risk_agent"), (list, np.ndarray)) and len(actions.get("risk_agent", [])) > 0 else 0, 0.0)

                # --- PERFORMANCE & PORTFOLIO ---
                perf, overall_risk, market_risk = self._calc_perf_and_quick_risks()

                # Get portfolio value/equity (in DKK from env) - FIXED: Use proper equity calculation
                portfolio_value_dkk = getattr(self.env, 'equity', None)
                if portfolio_value_dkk is None or portfolio_value_dkk == 0:
                    # Try to calculate NAV if equity is not available
                    if hasattr(self.env, '_calculate_fund_nav'):
                        try:
                            portfolio_value_dkk = self.env._calculate_fund_nav()
                        except Exception:
                            # Fallback to manual calculation
                            cash_dkk = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                            w_val_dkk = self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0)
                            s_val_dkk = self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0)
                            h_val_dkk = self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0)
                            portfolio_value_dkk = cash_dkk + w_val_dkk + s_val_dkk + h_val_dkk
                    else:
                        # Fallback to manual calculation
                        cash_dkk = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                        w_val_dkk = self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0)
                        s_val_dkk = self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0)
                        h_val_dkk = self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0)
                        portfolio_value_dkk = cash_dkk + w_val_dkk + s_val_dkk + h_val_dkk

                # Apply safety bounds
                initial_budget_dkk = float(getattr(self.env, 'init_budget', 500_000_000))
                portfolio_value_dkk = self._safe_portfolio_value(portfolio_value_dkk, initial_budget_dkk)
                self._last_portfolio_value = portfolio_value_dkk

                # Convert all DKK values to USD for logging
                portfolio_value_usd = self._safe_float(portfolio_value_dkk * dkk_to_usd_rate, 0.0)
                equity_usd = portfolio_value_usd  # Same as portfolio value
                total_return_nav_usd = self._safe_float(getattr(self.env, 'fund_nav', portfolio_value_dkk) * dkk_to_usd_rate, 0.0)

                # --- ENV SNAPSHOT (convert DKK to USD) ---
                budget_usd = self._safe_float(getattr(self.env, 'budget', 0.0) * dkk_to_usd_rate, 0.0)
                wind_c = self._safe_float(getattr(self.env, 'wind_capacity', 0.0), 0.0)
                solar_c = self._safe_float(getattr(self.env, 'solar_capacity', 0.0), 0.0)
                hydro_c = self._safe_float(getattr(self.env, 'hydro_capacity', 0.0), 0.0)
                batt_e = self._safe_float(getattr(self.env, 'battery_energy', 0.0), 0.0)

                # --- ACTUALS ---
                i = getattr(self.env, 't', 0)
                if i < len(self.env.data):
                    row = self.env.data.iloc[i]
                    price_act = self._safe_float(row.get('price', 0), 0.0)
                    load_act = self._safe_float(row.get('load', 0), 0.0)
                    wind_act = self._safe_float(row.get('wind', 0), 0.0)
                    solar_act = self._safe_float(row.get('solar', 0), 0.0)
                    hydro_act = self._safe_float(row.get('hydro', 0), 0.0)
                else:
                    price_act = load_act = wind_act = solar_act = hydro_act = 0.0

                # --- MARKET REGIME ---
                market_stress = self._safe_float(getattr(self.env, 'market_stress', 0.5), 0.5)
                market_vol = self._safe_float(getattr(self.env, 'market_volatility', 0.0), 0.0)
                revenue_step_usd = self._safe_float(getattr(self.env, 'last_revenue', 0.0) * dkk_to_usd_rate, 0.0)

                # --- FINANCIAL BREAKDOWN (convert DKK to USD) ---
                generation_revenue_usd = self._safe_float(getattr(self.env, 'last_generation_revenue', 0.0) * dkk_to_usd_rate, 0.0)
                mtm_pnl_usd = self._safe_float(getattr(self.env, 'last_mtm_pnl', 0.0) * dkk_to_usd_rate, 0.0)
                distributed_profits_usd = self._safe_float(getattr(self.env, 'distributed_profits', 0.0) * dkk_to_usd_rate, 0.0)
                cumulative_returns_usd = self._safe_float(getattr(self.env, 'cumulative_returns', 0.0) * dkk_to_usd_rate, 0.0)
                investment_capital_usd = self._safe_float(getattr(self.env, 'investment_capital', 0.0) * dkk_to_usd_rate, 0.0)
                fund_performance = self._safe_float(getattr(self.env, 'fund_performance', 1.0), 1.0)

                # --- EPISODE MARKERS ---
                step_in_ep = getattr(self.env, 'step_in_episode', step_t)
                episode_end = any(getattr(self.env, 'dones', {}).values()) if hasattr(self.env, 'dones') else False

                # --- RISK VECTOR 6D ---
                risk_vec = getattr(self.env, 'last_risk_vector', None)
                if risk_vec is not None and len(risk_vec) >= 6:
                    r6 = [self._safe_float(risk_vec[i], 0.5) for i in range(6)]
                else:
                    r6 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

                # --- REWARD BREAKDOWN ---
                reward_breakdown = getattr(self.env, 'last_reward_breakdown', {})
                r_fin = self._safe_float(reward_breakdown.get('financial', 0.0), 0.0)
                r_risk = self._safe_float(reward_breakdown.get('risk', 0.0), 0.0)
                r_sus = self._safe_float(reward_breakdown.get('sustainability', 0.0), 0.0)
                r_eff = self._safe_float(reward_breakdown.get('efficiency', 0.0), 0.0)
                r_div = self._safe_float(reward_breakdown.get('diversification', 0.0), 0.0)

                reward_weights = getattr(self.env, 'last_reward_weights', {})
                w_fin = self._safe_float(reward_weights.get('financial', 0.2), 0.2)
                w_rsk = self._safe_float(reward_weights.get('risk', 0.2), 0.2)
                w_sus = self._safe_float(reward_weights.get('sustainability', 0.2), 0.2)
                w_eff = self._safe_float(reward_weights.get('efficiency', 0.2), 0.2)
                w_div = self._safe_float(reward_weights.get('diversification', 0.2), 0.2)

                # --- OPS & SCALING HEALTH ---
                seed = getattr(self.env, 'seed', 0)
                step_ms = self._safe_float(self._last_step_wall_ms, 0.0)
                mem_mb = self._safe_float(psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0, 0.0)

                # Action clipping fractions
                clip_inv = self._safe_float(self._last_clip_counts.get('investor_agent', 0) / max(self._last_clip_totals.get('investor_agent', 1), 1), 0.0)
                clip_bat = self._safe_float(self._last_clip_counts.get('battery_agent', 0) / max(self._last_clip_totals.get('battery_agent', 1), 1), 0.0)
                clip_rsk = self._safe_float(self._last_clip_counts.get('risk_agent', 0) / max(self._last_clip_totals.get('risk_agent', 1), 1), 0.0)
                clip_meta = self._safe_float(self._last_clip_counts.get('meta_agent', 0) / max(self._last_clip_totals.get('meta_agent', 1), 1), 0.0)

                # --- Assemble the final log row with all monetary values in USD ---
                row = [
                    ts_str, int(step_t), int(self.episode_count), meta_reward, inv_freq, cap_frac,
                    meta_a0, meta_a1, inv_a0, inv_a1, inv_a2, batt_a0, risk_a0,
                    perf, portfolio_value_usd, equity_usd, total_return_nav_usd, overall_risk, market_risk,
                    budget_usd, wind_c, solar_c, hydro_c, batt_e,
                    price_act, load_act, wind_act, solar_act, hydro_act,
                    market_stress, market_vol, revenue_step_usd,
                    generation_revenue_usd, mtm_pnl_usd, distributed_profits_usd, cumulative_returns_usd, investment_capital_usd, fund_performance,
                    self._ep_meta_return, step_in_ep, episode_end,
                    *r6,
                    r_fin, r_risk, r_sus, r_eff, r_div,
                    w_fin, w_rsk, w_sus, w_eff, w_div,
                    self.episode_count, seed, step_ms, mem_mb,
                    clip_inv, clip_bat, clip_rsk, clip_meta
                ]

                self.log_buffer.append(row)
                if len(self.log_buffer) >= 25 or self.step_count % 500 == 0 or len(self.log_buffer) >= self._flush_every_rows or episode_end:
                    self._flush_log_buffer()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Baseline metrics logging failed: {e}")
                self.error_count += 1

    def _flush_log_buffer(self):
        """Safely flushes the in-memory log buffer to the CSV file."""
        if not self.log_path or not self.log_buffer:
            return
        try:
            with self.log_lock, open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                while self.log_buffer:
                    writer.writerow(self.log_buffer.popleft())
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Failed to flush baseline log buffer: {e}")
                self.error_count += 1

    def reset(self, seed=None, options=None):
        """Reset environment and initialize logging."""
        obs, infos = self.env.reset(seed=seed, options=options)
        self.step_count = 0
        self.episode_count += 1
        self._ep_meta_return = 0.0
        return obs, infos

    def step(self, actions):
        """Step environment and log data."""
        start_time = time.time()
        obs, rewards, dones, truncs, infos = self.env.step(actions)

        self.step_count += 1
        self._last_step_wall_ms = (time.time() - start_time) * 1000

        # Track episode meta return
        if "meta_agent" in rewards:
            self._ep_meta_return += rewards["meta_agent"]

        # Log if within logging window
        if self.log_last_n == 0 or self.step_count > (self.total_timesteps - self.log_last_n):
            self._log_metrics_efficient(actions, rewards)

        return obs, rewards, dones, truncs, infos

    def close(self):
        """Close environment and flush any remaining logs."""
        if self.log_path:
            self._flush_log_buffer()
        if hasattr(self.env, 'close'):
            self.env.close()


class UltraFastProgressWrapper(ParallelEnv):
    """Ultra-fast wrapper that only adds progress tracking without any forecasting overhead."""

    def __init__(self, base_env, total_timesteps):
        self.env = base_env
        self.total_timesteps = total_timesteps
        self.step_count = 0
        self.global_step_count = 0  # Track cumulative steps across all intervals
        self.progress_interval = max(1000, total_timesteps // 50)  # Show progress every 2%
        self.last_progress_step = 0
        self.last_portfolio_value = None  # Track for validation
        self.portfolio_history = []  # Track recent values for trend analysis

        # Mark the base environment as being in ultra fast mode (for progress display only)
        self.env.ultra_fast_mode = True

        # Delegate all attributes to base environment
        self.possible_agents = base_env.possible_agents
        self.agents = base_env.agents
        self.observation_spaces = base_env.observation_spaces
        self.action_spaces = base_env.action_spaces
        self.metadata = base_env.metadata

        print(f"[ULTRA FAST] Progress updates every {self.progress_interval:,} steps (~2% intervals)")

    def reset(self, seed=None, options=None):
        # Only reset episode-level counters, keep global tracking
        self.step_count = 0
        # DON'T reset global_step_count - it tracks cumulative progress
        # DON'T reset last_progress_step - it tracks global progress intervals
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, dones, truncs, infos = self.env.step(actions)

        self.step_count += 1
        self.global_step_count += 1

        # Show progress every interval with financial breakdown
        if self.global_step_count - self.last_progress_step >= self.progress_interval:
            # Use global step count for accurate cumulative progress
            progress_pct = (self.global_step_count / self.total_timesteps) * 100

            # Get portfolio value with validation (convert DKK to USD for display)
            # Try multiple sources for portfolio value in DKK
            portfolio_value_dkk = getattr(self.env, 'equity', None)
            if portfolio_value_dkk is None or portfolio_value_dkk == 0:
                # Fallback to fund_nav calculation
                portfolio_value_dkk = getattr(self.env, 'fund_nav', None)
                if portfolio_value_dkk is None:
                    # Calculate NAV if available
                    if hasattr(self.env, '_calculate_fund_nav'):
                        try:
                            portfolio_value_dkk = self.env._calculate_fund_nav()
                        except:
                            # Use correct DKK equivalent of 500M USD
                            portfolio_value_dkk = getattr(self.env, 'init_budget', 3_448_275_862)  # 500M USD in DKK
                    else:
                        # Use correct DKK equivalent of 500M USD
                        portfolio_value_dkk = getattr(self.env, 'init_budget', 3_448_275_862)  # 500M USD in DKK

            # Convert DKK to USD for display
            dkk_to_usd_rate = getattr(self.env.config, 'dkk_to_usd_rate', 0.145)
            portfolio_value = (portfolio_value_dkk * dkk_to_usd_rate) / 1e6  # Convert to USD millions

            # Validate portfolio value for consistency
            if self.last_portfolio_value is not None:
                value_change = portfolio_value - self.last_portfolio_value
                if abs(value_change) > 100:  # Alert for large changes (>$100M)
                    print(f"⚠️  Large portfolio change detected: ${value_change:+.1f}M")

            # Track portfolio history for trend analysis
            self.portfolio_history.append(portfolio_value)
            if len(self.portfolio_history) > 10:
                self.portfolio_history.pop(0)  # Keep last 10 values

            # Get financial breakdown with DKK to USD conversion
            dkk_to_usd_rate = getattr(self.env, '_dkk_to_usd_rate', 0.145)
            ops_revenue_dkk = getattr(self.env, 'last_generation_revenue', 0.0)
            ops_revenue_usd = ops_revenue_dkk * dkk_to_usd_rate / 1e3  # Convert to thousands USD

            # Get trading PnL - use cumulative performance instead of step-by-step change
            trading_pnl_dkk = getattr(self.env, 'cumulative_mtm_pnl', getattr(self.env, 'last_mtm_pnl', 0.0))
            trading_pnl_usd = trading_pnl_dkk * dkk_to_usd_rate / 1e3  # Convert to thousands USD

            # ENHANCED DEBUG OUTPUT - Show detailed trading info at progress intervals
            if self.global_step_count % self.progress_interval == 0:
                investment_freq = getattr(self.env, 'investment_freq', 1)
                trading_enabled = getattr(self.env.reward_calculator, 'trading_enabled', True) if hasattr(self.env, 'reward_calculator') else True
                financial_positions = getattr(self.env, 'financial_positions', {})
                total_financial = sum(abs(v) for v in financial_positions.values()) if financial_positions else 0
                current_step = getattr(self.env, 't', 0)
                steps_since_trade = current_step % investment_freq

                # ENHANCED: Show individual position values for better debugging
                wind_pos = financial_positions.get('wind_instrument_value', 0.0)
                solar_pos = financial_positions.get('solar_instrument_value', 0.0)
                hydro_pos = financial_positions.get('hydro_instrument_value', 0.0)

                # Enhanced debug info for trading issues
                ultra_fast_mode = getattr(self.env, 'ultra_fast_mode', False)
                current_drawdown = getattr(self.env.reward_calculator, 'current_drawdown', 0.0) if hasattr(self.env, 'reward_calculator') else 0.0
                max_drawdown_threshold = getattr(self.env.reward_calculator, 'max_drawdown_threshold', 0.0) if hasattr(self.env, 'reward_calculator') else 0.0
                ultra_fast_override = getattr(self.env.reward_calculator, 'ultra_fast_mode_trading_enabled', False) if hasattr(self.env, 'reward_calculator') else False

                print(f"[TRADING DEBUG] Step {self.global_step_count} (env.t={current_step})")
                print(f"  Investment freq: {investment_freq} (next trade in {investment_freq - steps_since_trade} steps)")
                print(f"  Trading enabled: {trading_enabled}")
                print(f"  Financial positions: {total_financial:.0f} DKK")
                print(f"    Wind: {wind_pos:.0f} DKK | Solar: {solar_pos:.0f} DKK | Hydro: {hydro_pos:.0f} DKK")

                # Show both step and cumulative MTM PnL
                step_mtm = getattr(self.env, 'last_mtm_pnl', 0.0)
                cumulative_mtm = getattr(self.env, 'cumulative_mtm_pnl', 0.0)
                print(f"  MTM PnL (step): {step_mtm:.2f} DKK")
                print(f"  MTM PnL (cumulative): {cumulative_mtm:.2f} DKK")
                print(f"  Ultra fast mode: {ultra_fast_mode}")
                print(f"  Current drawdown: {current_drawdown:.1%}")
                print(f"  Max drawdown threshold: {max_drawdown_threshold:.1%}")
                print(f"  Ultra fast override enabled: {ultra_fast_override}")

                # ENHANCED: Show if trading should occur at this step
                should_trade = (current_step % investment_freq == 0)
                print(f"  Should trade at current step: {should_trade}")
                if not should_trade:
                    print(f"  Next trading step: {current_step + (investment_freq - steps_since_trade)}")

            # Format the breakdown with enhanced info
            ops_str = f"Ops: ${ops_revenue_usd:+.1f}k"
            trading_str = f"Trading: ${trading_pnl_usd:+.1f}k" if trading_pnl_usd != 0 else "Trading: $0.0k"

            # Add trend indicator
            trend_str = ""
            if len(self.portfolio_history) >= 2:
                recent_change = self.portfolio_history[-1] - self.portfolio_history[-2]
                if abs(recent_change) > 1.0:  # Show trend for changes > $1M
                    trend_str = f" | Trend: ${recent_change:+.1f}M"

            print(f"[ULTRA FAST] Progress: {self.global_step_count:,}/{self.total_timesteps:,} ({progress_pct:.1f}%) | Portfolio: ${portfolio_value:.1f}M USD | {ops_str} | {trading_str}{trend_str}")

            self.last_progress_step = self.global_step_count
            self.last_portfolio_value = portfolio_value

        return obs, rewards, dones, truncs, infos

    def close(self):
        return self.env.close()

    def get_progress_summary(self):
        """Get detailed progress summary for debugging."""
        portfolio_range = ""
        if len(self.portfolio_history) >= 2:
            min_val = min(self.portfolio_history)
            max_val = max(self.portfolio_history)
            portfolio_range = f" | Range: ${min_val:.1f}M-${max_val:.1f}M"

        return {
            'global_steps': self.global_step_count,
            'total_steps': self.total_timesteps,
            'progress_pct': (self.global_step_count / self.total_timesteps) * 100,
            'current_portfolio': self.last_portfolio_value,
            'portfolio_history': self.portfolio_history.copy(),
            'summary': f"Progress: {self.global_step_count:,}/{self.total_timesteps:,} ({(self.global_step_count / self.total_timesteps) * 100:.1f}%){portfolio_range}"
        }

    def set_global_step_count(self, count):
        """Allow external setting of global step count for synchronization."""
        self.global_step_count = count
        print(f"🔄 Global step count synchronized to: {count:,}")

    def __getattr__(self, name):
        # Delegate any other attribute access to the base environment
        return getattr(self.env, name)