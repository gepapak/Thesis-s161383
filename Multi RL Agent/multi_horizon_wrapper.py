# multi_horizon_wrapper.py
#!/usr/bin/env python3
"""
FULLY PATCHED Enhanced Multi-Horizon Wrapper

New in this patch (on top of your last file):
- Richer CSV logging:
  • Env snapshot: budget, capacities, battery_energy, revenue, actual price/load
  • Episode markers: cumulative meta return (ep_meta_return), step_in_episode
  • Full 6D risk vector from risk controller
  • 1-step forecast absolute error (MAE) for price/wind/solar/load (vs previous logged forecast)
- Helpers: _safe_float(), _get_actuals_for_logging()
- Tracks previous forecasts for MAE across logging intervals.
- Discrete action spaces handled correctly (ints), alongside Box actions.
- Optional reward analysis (get_reward_analysis) to match main script.
- Defensive observation building & action validation; clear fallbacks.
- Logging is buffered and flushed safely; gracefully disables on persistent I/O errors.
- _calculate_essential_performance_metrics tolerates numpy inputs and uses init_budget=1e7 default,
  clipping portfolio_performance for log stability.

Additional in this patch:
- CSV now includes a leading human-readable `timestamp` column pulled from env.data['timestamp'] (if present).
- Introduces `_flush_every_rows` (default 1000) to force periodic hard flushes by buffer size.
- Flush no longer converts the timestamp into a float; only numeric cells are coerced.

Maintains:
- TOTAL-dimension observation construction (base + forecasts) with strict shape checking.
- Forecast normalization policy (price normalized by PRICE_SCALE; wind/solar/hydro/load in [0,1]).
- Lightweight memory tracker and bounded caches.
"""

from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import csv
import os
import threading
from typing import Dict, Any, Tuple, Optional, List, Mapping
from datetime import datetime
from collections import deque
from contextlib import nullcontext
import pandas as pd
import gc
import psutil
import logging

# Keep price scale consistent with base env (price in [0,10])
PRICE_SCALE = 10.0


# =========================
# Memory + Validation Utils
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
        self.logger = logging.getLogger(__name__)

    def register_cache(self, cache_obj):
        self.tracked_caches.append(cache_obj)

    def get_memory_usage(self):
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
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
                import torch
                if torch.cuda.is_available():
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


class EnhancedObservationValidator:
    """
    Validates base obs from the ENV and appends forecast features to build TOTAL obs.

    IMPORTANT: the ENV returns BASE-only observations. The WRAPPER is responsible for:
      total_dim = base_dim + forecast_dim

    NOTE: Price forecast normalization is unified to divide by PRICE_SCALE (→ [0,10]),
    matching the base env's price feature scale.
    """
    def __init__(self, base_env, forecaster, debug=False):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        self.agent_observation_specs = {}
        self.validation_errors = deque(maxlen=50)
        self.validation_cache = {}

        self._initialize_observation_specs()

    def _initialize_observation_specs(self):
        for agent in self.base_env.possible_agents:
            try:
                base_dim = self._get_validated_base_dim(agent)
                forecast_dim = self._calculate_forecast_dimension(agent)
                total_dim = base_dim + forecast_dim  # authoritative

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
                self.logger.error(f"Failed to initialize specs for {agent}: {e}")
                self._create_fallback_spec(agent)

    def _get_validated_base_dim(self, agent: str) -> int:
        try:
            if hasattr(self.base_env, '_get_base_observation_dim'):
                return int(self.base_env._get_base_observation_dim(agent))
            space = self.base_env.observation_space(agent)
            return int(space.shape[0])
        except Exception:
            return self._estimate_base_dimension(agent)

    def _estimate_base_dimension(self, agent: str) -> int:
        base_dims = {
            'investor_0': 6,
            'battery_operator_0': 4,
            'risk_controller_0': 9,
            'meta_controller_0': 11
        }
        return base_dims.get(agent, 8)

    def _calculate_forecast_dimension(self, agent: str) -> int:
        try:
            if hasattr(self.forecaster, 'get_agent_forecast_dims'):
                dims = self.forecaster.get_agent_forecast_dims()
                return int(dims.get(agent, 0))
            if hasattr(self.forecaster, 'agent_horizons') and hasattr(self.forecaster, 'agent_targets'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                return int(len(targets) * len(horizons))
            return self._get_default_forecast_dim(agent)
        except Exception:
            return self._get_default_forecast_dim(agent)

    def _get_default_forecast_dim(self, agent: str) -> int:
        defaults = {
            'investor_0': 8,
            'battery_operator_0': 4,
            'risk_controller_0': 0,
            'meta_controller_0': 15
        }
        return defaults.get(agent, 0)

    def _get_agent_forecast_keys(self, agent: str, expected_count: int) -> List[str]:
        try:
            if hasattr(self.forecaster, 'agent_targets') and hasattr(self.forecaster, 'agent_horizons'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                keys = [f"{t}_forecast_{h}" for t in targets for h in horizons]
                while len(keys) < expected_count:
                    keys.append(f"fallback_forecast_{len(keys)}")
                return keys[:expected_count]
        except Exception:
            pass
        return [f"forecast_{i}" for i in range(expected_count)]

    def _get_safe_bounds(self, total_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        low = np.full(total_dim, -10.0, dtype=np.float32)
        high = np.full(total_dim, 10.0, dtype=np.float32)
        return low, high

    def _create_fallback_spec(self, agent: str):
        base_dim = self._estimate_base_dimension(agent)
        forecast_dim = self._get_default_forecast_dim(agent)
        total_dim = base_dim + forecast_dim
        self.agent_observation_specs[agent] = {
            'base_dim': base_dim,
            'forecast_dim': forecast_dim,
            'total_dim': total_dim,
            'forecast_keys': self._get_agent_forecast_keys(agent, forecast_dim),
            'bounds': self._get_safe_bounds(total_dim)
        }

    # -------- in-place total observation builder (no concat allocations) --------

    def build_total_observation(self, agent: str, base_obs: np.ndarray,
                                forecasts: Mapping[str, float],
                                out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate + assemble TOTAL observation in-place into `out` if provided.
        Falls back to allocation if `out` is None.
        """
        if agent not in self.agent_observation_specs:
            return self._create_safe_observation(agent)

        spec = self.agent_observation_specs[agent]
        bd, fd, td = spec['base_dim'], spec['forecast_dim'], spec['total_dim']

        # ensure output buffer
        if out is None or not isinstance(out, np.ndarray) or out.shape != (td,):
            out = np.zeros(td, dtype=np.float32)

        # 1) validate base
        fixed_base = self._validate_base_observation(agent, base_obs, spec)
        out[:bd] = fixed_base

        # 2) fill forecast slice directly
        if fd > 0:
            fk = spec['forecast_keys']
            out[bd:bd + fd].fill(0.0)  # initialize
            for j, key in enumerate(fk[:fd]):
                v = forecasts.get(key, 0.0) if isinstance(forecasts, Mapping) else 0.0
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    v = 0.0
                # normalize per feature type
                if 'price' in key:
                    v = np.clip(v / PRICE_SCALE, 0, 10)
                elif any(term in key for term in ['wind', 'solar', 'hydro', 'load']):
                    v = np.clip(v, 0, 1)
                else:
                    v = np.clip(v, -1, 1)
                out[bd + j] = float(v)
        else:
            if td > bd:
                out[bd:td].fill(0.0)

        # 3) final clip against total bounds
        low, high = spec['bounds']
        np.clip(out, low[:td], high[:td], out=out)
        return out

    # -------- legacy allocate path (kept for safety) --------

    def validate_and_fix_observation(self, agent: str, base_obs: np.ndarray,
                                     forecasts: Dict[str, float]) -> np.ndarray:
        if agent not in self.agent_observation_specs:
            return self._create_safe_observation(agent)
        spec = self.agent_observation_specs[agent]

        try:
            obs_key = (agent, str(base_obs.shape),
                       hash(base_obs.tobytes()) if isinstance(base_obs, np.ndarray) else hash(str(base_obs)))
            if obs_key in self.validation_cache:
                return self.validation_cache[obs_key].copy()

            fixed_base = self._validate_base_observation(agent, base_obs, spec)
            fixed_forecasts = self._extract_forecast_component(agent, forecasts, spec)

            combined = fixed_base if spec['forecast_dim'] == 0 else np.concatenate([fixed_base, fixed_forecasts])
            final = self._final_dimension_check(agent, combined, spec)

            if len(self.validation_cache) < 1000:
                self.validation_cache[obs_key] = final.copy()
            return final.astype(np.float32)

        except Exception as e:
            self.validation_errors.append(f"Validation failed for {agent}: {e}")
            return self._create_safe_observation(agent)

    # ---- internals reused by both paths ----

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
        cur = base_obs.shape[0]
        if cur < bd:
            low, high = spec['bounds']
            pad = ((low[:bd] + high[:bd]) / 2.0)[cur:bd]
            base_obs = np.concatenate([base_obs, pad])
        elif cur > bd:
            base_obs = base_obs[:bd]

        base_obs = np.nan_to_num(base_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        low, high = spec['bounds']
        base_obs = np.clip(base_obs, low[:bd], high[:bd])
        return base_obs

    def _extract_forecast_component(self, agent: str, forecasts: Dict[str, float], spec: Dict) -> np.ndarray:
        fdim = spec['forecast_dim']
        if fdim == 0:
            return np.array([], dtype=np.float32)

        keys = spec['forecast_keys']
        vals = []
        for k in keys:
            v = forecasts.get(k, 0.0)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                v = 0.0
            if 'price' in k:
                v = np.clip(v / PRICE_SCALE, 0, 10)
            elif any(term in k for term in ['wind', 'solar', 'hydro', 'load']):
                v = np.clip(v, 0, 1)
            else:
                v = np.clip(v, -1, 1)
            vals.append(float(v))

        while len(vals) < fdim:
            vals.append(0.0)
        return np.array(vals[:fdim], dtype=np.float32)

    def _final_dimension_check(self, agent: str, obs: np.ndarray, spec: Dict) -> np.ndarray:
        expected = spec['total_dim']
        cur = obs.shape[0]
        if cur < expected:
            obs = np.concatenate([obs, np.zeros(expected - cur, dtype=np.float32)])
        elif cur > expected:
            obs = obs[:expected]
        low, high = spec['bounds']
        obs = np.clip(obs, low[:expected], high[:expected])
        return obs

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
    """Builds total observations (base + forecasts) with validation & caching, in-place."""
    def __init__(self, base_env, forecaster, debug=False):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug

        self.validator = EnhancedObservationValidator(base_env, forecaster, debug)

        # Preallocate TOTAL-dim output buffers (one per agent) and reuse them every step
        self._obs_out = {
            agent: np.zeros(self.validator.agent_observation_specs[agent]['total_dim'], dtype=np.float32)
            for agent in self.base_env.possible_agents
        }

        # Forecast caching
        self.forecast_cache: Dict[str, Dict[str, float]] = {}          # global dict cache per timestep
        self.agent_forecast_cache: Dict[Tuple[str, int], Dict[str, float]] = {}  # per-agent cache
        self.cache_size_limit = 500

        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=300)
        self.memory_tracker.register_cache(self.forecast_cache)
        self.memory_tracker.register_cache(self.agent_forecast_cache)

        if debug:
            stats = self.validator.get_validation_stats()
            print("🔍 Enhanced Observation Builder Initialized:")
            for agent, spec in stats['specs'].items():
                print(f"   {agent}: base={spec['base_dim']}, forecast={spec['forecast_dim']}, total={spec['total_dim']}")

    # -------- main entry --------

    def enhance_observations(self, base_obs: Dict[str, np.ndarray],
                             all_forecasts: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Fill TOTAL observations in-place and return the reusable dict of arrays.
        If `all_forecasts` is None and the forecaster supports per-agent prediction,
        forecasts are fetched per agent and cached.
        """
        cleanup_level, _ = self.memory_tracker.should_cleanup()
        if cleanup_level:
            self.memory_tracker.cleanup(cleanup_level)

        t = getattr(self.base_env, 't', 0)
        per_agent_possible = hasattr(self.forecaster, 'predict_for_agent')

        for agent in self.base_env.possible_agents:
            try:
                if agent in base_obs:
                    # choose forecast source (per-agent preferred if available)
                    if all_forecasts is not None:
                        fsrc: Mapping[str, float] = all_forecasts
                    elif per_agent_possible:
                        fsrc = self._get_cached_forecasts_for_agent(agent, t)
                    else:
                        # fallback: global dict (cached)
                        fsrc = self._get_cached_forecasts_global(t)
                    # in-place assembly
                    self.validator.build_total_observation(
                        agent, base_obs[agent], fsrc, out=self._obs_out[agent]
                    )
                else:
                    self._obs_out[agent][:] = self.validator._create_safe_observation(agent)
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Observation building failed for {agent}: {e}")
                self._obs_out[agent][:] = self.validator._create_safe_observation(agent)

        return self._obs_out

    # -------- forecast caching helpers --------

    def _get_cached_forecasts_global(self, timestep: int) -> Dict[str, float]:
        key = f"forecasts_{timestep}"
        if key in self.forecast_cache:
            return self.forecast_cache[key]
        try:
            all_forecasts = self.forecaster.predict_all_horizons(timestep=timestep)
            if not isinstance(all_forecasts, dict):
                all_forecasts = {}
        except Exception:
            all_forecasts = {}
        self._bounded_put(self.forecast_cache, key, all_forecasts)
        return all_forecasts

    def _get_cached_forecasts_for_agent(self, agent: str, timestep: int) -> Dict[str, float]:
        key = (agent, timestep)
        if key in self.agent_forecast_cache:
            return self.agent_forecast_cache[key]
        try:
            if hasattr(self.forecaster, 'predict_for_agent'):
                f = self.forecaster.predict_for_agent(agent=agent, timestep=timestep)
                if not isinstance(f, dict):
                    f = {}
            else:
                # fallback to global and reuse
                f = self._get_cached_forecasts_global(timestep)
        except Exception:
            f = {}
        self._bounded_put(self.agent_forecast_cache, key, f)
        return f

    def _bounded_put(self, d: Dict, key, value):
        if len(d) >= self.cache_size_limit:
            # drop ~60% oldest
            drop = int(len(d) * 0.6)
            for k in list(d.keys())[:drop]:
                del d[k]
        d[key] = value

    # -------- diagnostics --------

    def get_diagnostic_info(self) -> Dict:
        return {
            'validator_stats': self.validator.get_validation_stats(),
            'memory_stats': self.memory_tracker.get_memory_stats(),
            'cache_size': {
                'global': len(self.forecast_cache),
                'per_agent': len(self.agent_forecast_cache),
            },
            'cache_limit': self.cache_size_limit
        }


# =========================
# The Wrapper Env
# =========================

class MultiHorizonWrapperEnv(ParallelEnv):
    """
    Wraps a BASE-only env and exposes TOTAL-dim observations (base + forecasts).
    """
    def __init__(self, base_env, multi_horizon_forecaster, log_path=None, max_memory_mb=1500):
        self.env = base_env
        self.forecaster = multi_horizon_forecaster

        # Agent lists
        self._possible_agents = self.env.possible_agents[:]
        self._agents = self._possible_agents[:]
        self.max_memory_mb = max_memory_mb

        # Memory & logging
        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=max_memory_mb)
        self.memory_check_counter = 0
        self.log_path = self._setup_logging_path(log_path)
        self.log_buffer = deque(maxlen=25)
        self.log_lock = threading.RLock() if threading else nullcontext()

        # NEW: hard flush threshold by buffered rows
        self._flush_every_rows = 1000

        # Do NOT mutate env spaces. Just hold the forecaster reference.
        if hasattr(self.env, 'forecast_generator') and getattr(self.env, 'forecast_generator') is None:
            try:
                self.env.forecast_generator = self.forecaster
            except Exception:
                pass

        # Builder (creates specs)
        self.obs_builder = MemoryOptimizedObservationBuilder(self.env, self.forecaster, debug=False)
        self.memory_tracker.register_cache(self.obs_builder.forecast_cache)
        self.memory_tracker.register_cache(self.obs_builder.agent_forecast_cache)
        self.memory_tracker.register_cache(self.log_buffer)

        # Build wrapper observation spaces (TOTAL dims)
        self._build_wrapper_observation_spaces()

        self._initialize_logging_safe()

        # misc counters
        self.step_count = 0
        self.episode_count = 0
        self.last_log_flush = 0
        self.error_count = 0
        self.max_errors = 50

        # episode markers & MAE helpers
        self._ep_meta_return = 0.0
        self._prev_forecasts_for_error: Dict[str, float] = {}

        # logging-time forecast cache (per timestep)
        self._log_forecasts_step = -1
        self._log_forecasts_cache: Dict[str, float] = {}
        self.memory_tracker.register_cache(self._log_forecasts_cache)

        # log every N steps
        self._log_interval = 20

        print("✅ Enhanced multi-horizon wrapper initialized (TOTAL-dim observations)")

    # ---- wrapper observation spaces (TOTAL dims) ----
    def _build_wrapper_observation_spaces(self):
        self._obs_spaces = {}
        specs = self.obs_builder.validator.agent_observation_specs
        for agent, spec in specs.items():
            low, high = spec['bounds']
            total_dim = spec['total_dim']
            self._obs_spaces[agent] = spaces.Box(low=low[:total_dim], high=high[:total_dim],
                                                 shape=(total_dim,), dtype=np.float32)

    # ------------- logging setup -------------
    def _setup_logging_path(self, log_path):
        if log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/multi_horizon_metrics_{timestamp}.csv"
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Could not create log directory: {e}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"fallback_metrics_{timestamp}.csv"
        return log_path

    def _initialize_logging_safe(self):
        try:
            if self.log_path is not None and not os.path.isfile(self.log_path):
                self._create_log_header()
            print(f"✅ Logging initialized: {self.log_path}")
        except Exception as e:
            print(f"⚠️ Logging initialization failed: {e}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"emergency_metrics_{timestamp}.csv"
            try:
                self._create_log_header()
            except Exception as e2:
                print(f"❌ Emergency logging failed: {e2}")
                self.log_path = None

    def _create_log_header(self):
        if self.log_path is None:
            return
        try:
            headers = [
                # NEW: prepend a real timestamp column
                "timestamp",

                # core
                "timestep", "episode", "meta_reward", "investment_freq", "capital_fraction",
                "meta_action_0", "meta_action_1", "inv_action_0", "inv_action_1", "inv_action_2",
                "batt_action_0", "risk_action_0",

                # immediate forecasts (as returned by forecaster)
                "wind_forecast_immediate", "solar_forecast_immediate",
                "price_forecast_immediate", "load_forecast_immediate",

                # perf & quick risks
                "portfolio_performance", "overall_risk", "market_risk",

                # --- env snapshot ---
                "budget", "wind_cap", "solar_cap", "hydro_cap", "battery_energy",
                "price_actual", "load_actual", "revenue_step",

                # --- episode markers ---
                "ep_meta_return", "step_in_episode",

                # --- full 6D risk vector ---
                "risk_market", "risk_gen_var", "risk_portfolio",
                "risk_liquidity", "risk_stress", "risk_overall",

                # --- 1-step absolute forecast error (MAE) vs previous forecast ---
                "mae_price_1", "mae_wind_1", "mae_solar_1", "mae_load_1"
            ]
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(headers)
        except Exception as e:
            print(f"⚠️ Log header creation failed: {e}")
            self.log_path = None

    # ------------- properties -------------

    @property
    def possible_agents(self):
        return self._possible_agents

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, value):
        self._agents = value[:]

    # ------------- gym/pettingzoo API -------------

    def observation_space(self, agent):
        """Return TOTAL-dim observation space (base + forecasts)."""
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self.env.action_space(agent)

    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    @property
    def t(self):
        return getattr(self.env, 't', 0)

    @property
    def max_steps(self):
        return getattr(self.env, 'max_steps', 1000)

    # ------------- main loop -------------

    def reset(self, seed=None, options=None):
        try:
            self._cleanup_memory_enhanced(force=True)

            obs, info = self.env.reset(seed=seed, options=options)
            self.agents = self.possible_agents[:]
            self.step_count = 0
            self.episode_count += 1
            self.error_count = 0
            self._ep_meta_return = 0.0
            self._prev_forecasts_for_error = {}

            # optional forecaster "first look"
            try:
                if hasattr(self.env, "data") and len(self.env.data) > 0:
                    self.forecaster.update(self.env.data.iloc[0])
            except Exception:
                pass

            enhanced = self.obs_builder.enhance_observations(obs)  # per-agent by default if supported
            validated = self._validate_observations_safe(enhanced)
            return validated, info
        except Exception as e:
            print(f"⚠️ Environment reset failed: {e}")
            return self._create_safe_reset_result()

    def step(self, actions):
        try:
            self.step_count += 1
            if self.step_count % (50 if self.step_count < 1000 else 100) == 0:
                self._cleanup_memory_enhanced()

            actions = self._validate_actions_comprehensive(actions)
            obs, rewards, dones, truncs, infos = self.env.step(actions)

            # accumulate episode meta return
            r_meta = rewards.get("meta_controller_0", 0.0)
            self._ep_meta_return += self._safe_float(r_meta, 0.0)

            # update forecaster with last row
            self._update_forecaster_safe()

            # build enhanced obs (per-agent forecasting used internally if available)
            enhanced = self.obs_builder.enhance_observations(obs)
            validated = self._validate_observations_safe(enhanced)

            # logging (also when we need global forecasts)
            if self.step_count % self._log_interval == 0:
                log_forecasts = self._get_forecasts_for_logging()
                self._log_metrics_efficient(actions, rewards, log_forecasts)

            return validated, rewards, dones, truncs, infos

        except Exception as e:
            self.error_count += 1
            if self.error_count <= self.max_errors:
                print(f"⚠️ Environment step failed: {e}")
            return self._create_safe_step_result()

    # ------------- helpers -------------

    def _validate_observations_safe(self, obs_dict):
        validated = {}
        for agent in self.possible_agents:
            try:
                if agent in obs_dict:
                    obs = obs_dict[agent]
                    expected_shape = self.observation_space(agent).shape  # TOTAL dims
                    if obs.shape != expected_shape:
                        if self.error_count < self.max_errors:
                            print(f"⚠️ Observation shape mismatch for {agent}: got {obs.shape}, expected {expected_shape}")
                            self.error_count += 1
                        obs = self.obs_builder.validator._create_safe_observation(agent)
                    validated[agent] = obs.astype(np.float32)
                else:
                    validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
            except Exception as e:
                if self.error_count < self.max_errors:
                    print(f"⚠️ Observation validation failed for {agent}: {e}")
                    self.error_count += 1
                validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
        return validated

    def _cleanup_memory_enhanced(self, force=False):
        try:
            cleanup_level, _ = self.memory_tracker.should_cleanup(force=force)
            if cleanup_level or force:
                self.memory_tracker.cleanup(cleanup_level or 'medium')

                # wrapper-specific cache trims
                if hasattr(self.obs_builder, 'forecast_cache'):
                    cache_size_before = len(self.obs_builder.forecast_cache)
                    if cache_size_before > 100:
                        items_to_remove = int(cache_size_before * 0.7)
                        old_keys = list(self.obs_builder.forecast_cache.keys())[:items_to_remove]
                        for k in old_keys:
                            del self.obs_builder.forecast_cache[k]

                if hasattr(self.obs_builder, 'agent_forecast_cache'):
                    cache_size_before = len(self.obs_builder.agent_forecast_cache)
                    if cache_size_before > 100:
                        items_to_remove = int(cache_size_before * 0.7)
                        old_keys = list(self.obs_builder.agent_forecast_cache.keys())[:items_to_remove]
                        for k in old_keys:
                            del self.obs_builder.agent_forecast_cache[k]

                self._flush_log_buffer()
                if hasattr(self.obs_builder.validator, 'validation_cache'):
                    self.obs_builder.validator.validation_cache.clear()
                gc.collect()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Enhanced memory cleanup failed: {e}")
                self.error_count += 1

    def _validate_actions_comprehensive(self, actions):
        validated = {}
        for agent in self.possible_agents:
            space = self.env.action_space(agent)
            if agent in actions:
                try:
                    action = actions[agent]
                    if isinstance(space, spaces.Discrete):
                        # Discrete: return an int in [0, n-1]
                        if isinstance(action, np.ndarray):
                            a = int(np.atleast_1d(action).flatten()[0])
                        elif np.isscalar(action):
                            a = int(action)
                        elif isinstance(action, (list, tuple)) and len(action) > 0:
                            a = int(action[0])
                        else:
                            a = 0
                        a = int(np.clip(a, 0, space.n - 1))
                        validated[agent] = a
                    else:
                        # Box: return clipped float array of correct shape
                        if not isinstance(action, np.ndarray):
                            if np.isscalar(action):
                                action = np.array([action], dtype=np.float32)
                            elif isinstance(action, (list, tuple)):
                                action = np.array(action, dtype=np.float32)
                            else:
                                action = np.array([float(action)], dtype=np.float32)
                        else:
                            action = action.astype(np.float32)
                        if action.ndim == 0:
                            action = action.reshape(1)
                        elif action.ndim > 1:
                            action = action.flatten()
                        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
                        if len(action) != space.shape[0]:
                            if len(action) < space.shape[0]:
                                middle = (space.low + space.high) / 2.0
                                m = float(middle[0]) if hasattr(middle, '__len__') else float(middle)
                                padding = np.full(space.shape[0] - len(action), m, dtype=np.float32)
                                action = np.concatenate([action, padding])
                            else:
                                action = action[:space.shape[0]]
                        validated[agent] = np.clip(action, space.low, space.high)
                except Exception as e:
                    if self.error_count < self.max_errors:
                        print(f"⚠️ Action validation failed for {agent}: {e}")
                        self.error_count += 1
                    validated[agent] = self._safe_action_default(space)
            else:
                validated[agent] = self._safe_action_default(space)
        return validated

    @staticmethod
    def _safe_action_default(space):
        if isinstance(space, spaces.Discrete):
            return 0
        return ((space.low + space.high) / 2.0).astype(np.float32)

    def _update_forecaster_safe(self):
        try:
            if hasattr(self.env, "data") and 0 < self.env.t < len(self.env.data):
                row = self.env.data.iloc[self.env.t - 1]
                self.forecaster.update(row)
        except Exception:
            pass

    # ---- helpers for logging ----
    def _safe_float(self, v, default=0.0):
        try:
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v).reshape(-1)
                v = v[0] if v.size > 0 else default
            return float(v) if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    def _get_actuals_for_logging(self) -> Dict[str, float]:
        """Read actual current values from env/data safely (using row t-1)."""
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
        """Resolve a human-readable timestamp string from env.data if present."""
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

    # Only called when logging; avoids extra work during normal steps
    def _get_forecasts_for_logging(self) -> Dict[str, float]:
        t = getattr(self.env, 't', 0)
        if self._log_forecasts_step == t and self._log_forecasts_cache:
            return self._log_forecasts_cache
        try:
            f = self.forecaster.predict_all_horizons(timestep=t)
            if not isinstance(f, dict):
                f = {}
        except Exception:
            f = {}
        self._log_forecasts_cache = f
        self._log_forecasts_step = t
        return f

    def _log_metrics_efficient(self, actions: Dict[str, Any], rewards: Dict[str, float],
                               all_forecasts: Dict[str, float]):
        if self.log_path is None:
            return
        try:
            with self.log_lock:
                t = getattr(self.env, 't', self.step_count)
                ts_str = self._get_timestamp_for_logging()

                r_meta = self._safe_float(rewards.get("meta_controller_0", 0.0), 0.0)
                freq = int(getattr(self.env, "investment_freq", -1))
                cap_frac = self._safe_float(getattr(self.env, "capital_allocation_fraction", -1), -1.0)

                a_meta = self._safe_get_action(actions, "meta_controller_0", 2)
                a_inv  = self._safe_get_action(actions, "investor_0", 3)
                a_batt = self._safe_get_action(actions, "battery_operator_0", 1)
                a_risk = self._safe_get_action(actions, "risk_controller_0", 1)

                # start row with timestamp (string), then numeric fields
                row = [ts_str,
                       int(t), int(self.episode_count), r_meta, freq, cap_frac,
                       *a_meta, *a_inv, a_batt[0], a_risk[0]]

                # forecasts (as-is, same scale your forecaster emits)
                for k in ["wind_forecast_immediate", "solar_forecast_immediate",
                          "price_forecast_immediate", "load_forecast_immediate"]:
                    v = self._safe_float(all_forecasts.get(k, 0.0), 0.0)
                    row.append(v)

                # portfolio perf + quick risks
                row.extend(self._calculate_essential_performance_metrics())

                # ---- env snapshot ----
                budget = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                wind_c = self._safe_float(getattr(self.env, 'wind_capacity', 0.0), 0.0)
                solar_c = self._safe_float(getattr(self.env, 'solar_capacity', 0.0), 0.0)
                hydro_c = self._safe_float(getattr(self.env, 'hydro_capacity', 0.0), 0.0)
                batt_e = self._safe_float(getattr(self.env, 'battery_energy', 0.0), 0.0)
                revenue_step = self._safe_float(getattr(self.env, 'revenue', 0.0), 0.0)

                actuals = self._get_actuals_for_logging()
                price_act = self._safe_float(actuals['price'], 0.0)
                load_act  = self._safe_float(actuals['load'], 0.0)

                row.extend([budget, wind_c, solar_c, hydro_c, batt_e,
                            price_act, load_act, revenue_step])

                # ---- episode/training markers ----
                step_in_ep = int(getattr(self.env, 'step_in_episode', self.step_count))
                row.extend([self._ep_meta_return, step_in_ep])

                # ---- full 6D risk vector ----
                rvec = [0.5, 0.2, 0.25, 0.15, 0.35, 0.25]
                try:
                    if hasattr(self.env, 'enhanced_risk_controller') and self.env.enhanced_risk_controller:
                        rm = self.env.enhanced_risk_controller.get_risk_metrics_for_observation()
                        arr = np.asarray(rm, dtype=np.float32).reshape(-1)
                        if arr.size >= 6:
                            rvec = [float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else d
                                    for x, d in zip(arr[:6], rvec)]
                except Exception:
                    pass
                row.extend(rvec)  # market, gen_var, portfolio, liquidity, stress, overall

                # ---- 1-step MAE vs previous-logged forecast for immediate horizon ----
                def _mae(prev, actual):
                    prev = self._safe_float(prev, 0.0); actual = self._safe_float(actual, 0.0)
                    return abs(prev - actual)

                mae_price = _mae(self._prev_forecasts_for_error.get("price_forecast_immediate"), price_act)
                mae_wind  = _mae(self._prev_forecasts_for_error.get("wind_forecast_immediate"),  actuals['wind'])
                mae_solar = _mae(self._prev_forecasts_for_error.get("solar_forecast_immediate"), actuals['solar'])
                mae_load  = _mae(self._prev_forecasts_for_error.get("load_forecast_immediate"),  load_act)
                row.extend([mae_price, mae_wind, mae_solar, mae_load])

                # update previous-forecast cache for next MAE computation
                self._prev_forecasts_for_error = dict(all_forecasts) if isinstance(all_forecasts, dict) else {}

                # buffer row as-is (first item is timestamp string; others are numeric)
                self.log_buffer.append(row)

                # periodic flush: either step-gated, buffer length, or big buffer threshold
                if (len(self.log_buffer) >= 25 or
                        self.step_count - self.last_log_flush >= 500 or
                        len(self.log_buffer) >= self._flush_every_rows):
                    self._flush_log_buffer()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Metrics logging failed: {e}")
                self.error_count += 1

    def _safe_get_action(self, actions, agent, expected_size):
        try:
            if agent in actions:
                a = actions[agent]
                if isinstance(a, np.ndarray):
                    a = a.flatten().tolist() if a.ndim > 0 else [float(a.item())]
                elif np.isscalar(a):
                    a = [float(a)]
                elif isinstance(a, (list, tuple)):
                    a = list(a)
                else:
                    a = [float(a)]
                out = []
                for v in a:
                    out.append(float(v) if isinstance(v, (int, float)) and np.isfinite(v) else 0.0)
                while len(out) < expected_size:
                    out.append(0.0)
                return out[:expected_size]
            return [0.0] * expected_size
        except Exception:
            return [0.0] * expected_size

    def _calculate_essential_performance_metrics(self) -> list:
        """
        Portfolio performance + risk snapshot for logging.
        - Uses env.default init_budget=1e7 if missing.
        - Accepts numpy arrays from the risk controller without type issues.
        - Clips performance for log stability (does not affect training).
        """
        try:
            # Budget & portfolio value → performance ratio
            initial_budget = float(getattr(self.env, 'init_budget', 1e7))
            initial_budget = max(initial_budget, 1.0)
            current_budget = float(getattr(self.env, 'budget', initial_budget))
            wind = float(getattr(self.env, 'wind_capacity', 0))
            solar = float(getattr(self.env, 'solar_capacity', 0))
            hydro = float(getattr(self.env, 'hydro_capacity', 0))

            portfolio_value = current_budget + (wind + solar + hydro) * 100.0
            portfolio_performance = portfolio_value / initial_budget

            overall_risk = 0.5
            market_risk = 0.5
            if hasattr(self.env, 'enhanced_risk_controller') and self.env.enhanced_risk_controller:
                try:
                    rm = self.env.enhanced_risk_controller.get_risk_metrics_for_observation()
                    arr = np.asarray(rm, dtype=np.float32)  # robust: list/tuple/ndarray all OK
                    if arr.size >= 6:
                        if np.isfinite(arr[0]):
                            market_risk = float(np.clip(arr[0], 0.0, 1.0))
                        if np.isfinite(arr[-1]):
                            overall_risk = float(np.clip(arr[-1], 0.0, 1.0))
                except Exception:
                    pass

            # Clip only for logging stability (does not affect training dynamics)
            portfolio_performance = float(np.clip(portfolio_performance, 0.0, 10.0))
            return [portfolio_performance, overall_risk, market_risk]
        except Exception:
            return [1.0, 0.5, 0.3]

    def _flush_log_buffer(self):
        if not self.log_buffer or self.log_path is None:
            return
        with self.log_lock:
            rows = list(self.log_buffer)
            self.log_buffer.clear()
            self.last_log_flush = self.step_count
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            # Write rows without coercing the first (timestamp) column to float
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                for r in rows:
                    try:
                        out = [r[0]]  # timestamp (string)
                        # numeric columns thereafter
                        for x in r[1:]:
                            if isinstance(x, (int, float)) and np.isfinite(x):
                                out.append(float(x))
                            else:
                                # if someone sneaks in a non-numeric, write 0.0
                                out.append(0.0)
                        w.writerow(out)
                    except Exception:
                        continue
        except Exception:
            # disable logging on persistent failures
            self.log_path = None

    # ------------- analysis (optional) -------------

    def get_reward_analysis(self) -> Optional[Dict[str, float]]:
        """
        Optional helper for the main script: summarizes meta rewards over time from the CSV.
        Returns a small dict of stats (or None if unavailable).
        """
        try:
            if self.log_path is None or not os.path.exists(self.log_path):
                return None
            df = pd.read_csv(self.log_path)
            if 'meta_reward' not in df.columns or len(df) < 10:
                return None
            early = float(df['meta_reward'].head(100).mean()) if len(df) >= 100 else float(df['meta_reward'].mean())
            late = float(df['meta_reward'].tail(100).mean()) if len(df) >= 100 else early
            improvement = late - early
            pct = (improvement / abs(early) * 100.0) if abs(early) > 1e-12 else 0.0
            summary = {
                'rows': int(len(df)),
                'early_meta_reward': early,
                'late_meta_reward': late,
                'improvement': improvement,
                'improvement_pct': pct
            }
            print("📊 Wrapper Reward Analysis:")
            print(f"   Rows: {summary['rows']:,} | Early: {early:.4f} | Late: {late:.4f} | Δ: {improvement:+.4f} ({pct:+.1f}%)")
            return summary
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Reward analysis failed: {e}")
                self.error_count += 1
            return None

    # ------------- safe fallbacks -------------

    def _create_safe_reset_result(self):
        safe = {}
        for agent in self.possible_agents:
            safe[agent] = self.obs_builder.validator._create_safe_observation(agent)
        return safe, {}

    def _create_safe_step_result(self):
        safe = {}
        for agent in self.possible_agents:
            safe[agent] = self.obs_builder.validator._create_safe_observation(agent)
        rewards = {agent: 0.0 for agent in self.possible_agents}
        dones = {agent: False for agent in self.possible_agents}
        truncs = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        return safe, rewards, dones, truncs, infos

    # ------------- diagnostics & cleanup -------------

    def get_wrapper_diagnostics(self):
        try:
            return {
                'wrapper_info': {
                    'step_count': self.step_count,
                    'episode_count': self.episode_count,
                    'error_count': self.error_count,
                    'log_buffer_size': len(self.log_buffer),
                    'log_path': self.log_path
                },
                'observation_diagnostics': self.obs_builder.get_diagnostic_info(),
                'memory_diagnostics': self.memory_tracker.get_memory_stats(),
                'validation_errors': len(self.obs_builder.validator.validation_errors)
            }
        except Exception as e:
            return {'error': f"Diagnostics failed: {e}"}

    def render(self):
        try:
            if hasattr(self.env, "render"):
                return self.env.render()
        except Exception:
            return None

    def close(self):
        """Ensure logs are flushed on env close and delegate to base env."""
        try:
            self._flush_log_buffer()
        except Exception:
            pass
        try:
            self._cleanup_memory_enhanced(force=True)
        except Exception:
            pass
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass

    def __getattr__(self, name):
        try:
            return getattr(self.env, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __del__(self):
        try:
            self._flush_log_buffer()
        except Exception:
            pass
        try:
            self._cleanup_memory_enhanced(force=True)
        except Exception:
            pass
        try:
            if hasattr(self, 'obs_builder') and hasattr(self.obs_builder, 'forecast_cache'):
                self.obs_builder.forecast_cache.clear()
            if hasattr(self, 'obs_builder') and hasattr(self.obs_builder, 'agent_forecast_cache'):
                self.obs_builder.agent_forecast_cache.clear()
        except Exception:
            pass


# =========================
# Quick Test
# =========================

def test_wrapper_integration():
    """Minimal test ensuring wrapper outputs TOTAL-dim obs and handles Discrete actions."""
    print("🧪 Testing Enhanced Multi-Horizon Wrapper Integration...")

    try:
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range("2024-01-01", periods=5, freq="H"),
            'wind': [0.5, 0.6, 0.4, 0.7, 0.3],
            'solar': [0.3, 0.8, 0.2, 0.6, 0.4],
            'hydro': [0.9, 0.7, 0.8, 0.5, 0.6],
            'price': [6.0, 5.5, 7.0, 6.5, 5.8],
            'load': [0.7, 0.6, 0.8, 0.9, 0.5],
            'risk': [0.3, 0.4, 0.2, 0.5, 0.3]
        })

        class DummyForecaster:
            agent_targets = {
                'investor_0': ['wind','solar','price','load'],
                'battery_operator_0': ['price','load'],
                'risk_controller_0': [],
                'meta_controller_0': ['wind','solar','price','load','hydro'],
            }
            agent_horizons = {
                'investor_0': [1,3],
                'battery_operator_0': [1,3],
                'risk_controller_0': [],
                'meta_controller_0': [1,3,6],
            }
            def get_agent_forecast_dims(self):
                return {
                    'investor_0': 8, 'battery_operator_0': 4,
                    'risk_controller_0': 0, 'meta_controller_0': 15
                }
            def predict_all_horizons(self, timestep=0):
                return {
                    "wind_forecast_immediate": 0.5,
                    "solar_forecast_immediate": 0.4,
                    "price_forecast_immediate": 6.0,
                    "load_forecast_immediate": 0.7,
                }
            def predict_for_agent(self, agent, timestep=0):
                if agent == 'investor_0':
                    return {f"{k}_forecast_{h}": 0.1 for k in ['wind','solar','price','load'] for h in [1,3]}
                if agent == 'battery_operator_0':
                    return {f"{k}_forecast_{h}": 0.2 for k in ['price','load'] for h in [1,3]}
                if agent == 'meta_controller_0':
                    return {f"{k}_forecast_{h}": 0.3 for k in ['wind','solar','price','load','hydro'] for h in [1,3,6]}
                return {}
            def update(self, row): pass

        class MockBaseEnv:
            def __init__(self, df):
                self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]
                self.agents = self.possible_agents[:]
                self.t = 0
                self.data = df
                self.init_budget = 1e7
                self.budget = 1e7
                self.wind_capacity = 0.0
                self.solar_capacity = 0.0
                self.hydro_capacity = 0.0
                self.battery_energy = 0.0
                self.revenue = 0.0
            def observation_space(self, agent):
                base_dims = {"investor_0": 6, "battery_operator_0": 4, "risk_controller_0": 9, "meta_controller_0": 11}
                return spaces.Box(low=-10, high=10, shape=(base_dims[agent],), dtype=np.float32)
            def action_space(self, agent):
                if agent == "investor_0":
                    return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                if agent == "battery_operator_0":
                    return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                if agent == "risk_controller_0":
                    return spaces.Discrete(3)
                return spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
            def reset(self, seed=None, options=None):
                self.t = 0
                obs = {a: np.zeros(self.observation_space(a).shape, dtype=np.float32) for a in self.possible_agents}
                return obs, {}
            def step(self, actions):
                self.t += 1
                obs = {a: np.random.random(self.observation_space(a).shape).astype(np.float32) for a in self.possible_agents}
                rewards = {a: 0.1 for a in self.possible_agents}
                dones = {a: self.t >= 5 for a in self.possible_agents}
                truncs = {a: False for a in self.possible_agents}
                infos = {a: {} for a in self.possible_agents}
                return obs, rewards, dones, truncs, infos
            def close(self): pass

        base_env = MockBaseEnv(sample_data)
        wrapper = MultiHorizonWrapperEnv(base_env, DummyForecaster(), log_path="test_metrics.csv")
        print("✅ Wrapper created")

        for agent in wrapper.possible_agents:
            ws = wrapper.observation_space(agent).shape
            print(f"🔎 {agent} total obs shape: {ws}")

        obs, _ = wrapper.reset()
        for agent in wrapper.possible_agents:
            expected = wrapper.observation_space(agent).shape
            assert obs[agent].shape == expected, f"{agent} reset obs shape mismatch {obs[agent].shape}!={expected}"
        print("✅ Reset obs shapes OK")

        for step in range(3):
            actions = {
                "investor_0": wrapper.action_space("investor_0").sample(),
                "battery_operator_0": wrapper.action_space("battery_operator_0").sample(),
                "risk_controller_0": int(1),
                "meta_controller_0": wrapper.action_space("meta_controller_0").sample()
            }
            obs, rewards, dones, truncs, infos = wrapper.step(actions)
            for agent in wrapper.possible_agents:
                expected = wrapper.observation_space(agent).shape
                assert obs[agent].shape == expected, f"{agent} step obs shape mismatch"
        print("🎉 Steps OK, TOTAL-dim observations validated")

        try:
            wrapper.get_reward_analysis()
        except Exception:
            pass

        try:
            if os.path.exists("test_metrics.csv"):
                os.remove("test_metrics.csv")
        except Exception:
            pass
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    test_wrapper_integration()
