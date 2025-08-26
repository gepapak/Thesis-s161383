# multi_horizon_wrapper.py
#!/usr/bin/env python3
"""
Enhanced Multi-Horizon Wrapper (fully patched)

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
from collections import deque
from contextlib import nullcontext

PRICE_SCALE = 10.0  # keep aligned with base env

# =========================
# Memory + Validation Utils
# =========================
class SafeDivision:
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator is None or abs(denominator) < 1e-9:
            return default
        return numerator / denominator

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

# =========================
# Observation Validator
# =========================
class EnhancedObservationValidator:
    """
    Validates base obs from the ENV and appends forecast features to build TOTAL obs.

    IMPORTANT: the ENV returns BASE-only observations. The WRAPPER is responsible for:
      total_dim = base_dim + forecast_dim

    Note: Forecasts are passed in raw units; no normalization is applied here.
    """
    def __init__(self, base_env, forecaster, debug=False):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug

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
            # safe default per agent type
            return {'investor_0': 6, 'battery_operator_0': 4, 'risk_controller_0': 9, 'meta_controller_0': 11}.get(agent, 8)

    def _calculate_forecast_dimension(self, agent: str) -> int:
        try:
            if hasattr(self.forecaster, 'get_agent_forecast_dims'):
                dims = self.forecaster.get_agent_forecast_dims()
                return int(dims.get(agent, 0))
            if hasattr(self.forecaster, 'agent_horizons') and hasattr(self.forecaster, 'agent_targets'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                return int(len(targets) * len(horizons))
            # defaults (kept as-is because your run is stable with these totals)
            return {'investor_0': 8, 'battery_operator_0': 4, 'risk_controller_0': 0, 'meta_controller_0': 15}.get(agent, 0)
        except Exception:
            return {'investor_0': 8, 'battery_operator_0': 4, 'risk_controller_0': 0, 'meta_controller_0': 15}.get(agent, 0)

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
        low = np.full(total_dim, -100.0, dtype=np.float32)
        high = np.full(total_dim, 1000.0, dtype=np.float32)
        return low, high

    def _create_fallback_spec(self, agent: str):
        base_dim = {'investor_0': 6, 'battery_operator_0': 4, 'risk_controller_0': 9, 'meta_controller_0': 11}.get(agent, 8)
        forecast_dim = {'investor_0': 8, 'battery_operator_0': 4, 'risk_controller_0': 0, 'meta_controller_0': 15}.get(agent, 0)
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
            out[bd:bd + fd].fill(0.0)
            for j, key in enumerate(keys):
                v = forecasts.get(key, 0.0) if isinstance(forecasts, Mapping) else 0.0
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    v = 0.0
                out[bd + j] = float(v)

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
    """Builds total observations (base + forecasts) with validation & caching, in-place."""
    def __init__(self, base_env, forecaster, debug=False):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug

        self.validator = EnhancedObservationValidator(base_env, forecaster, debug)
        self._obs_out = {
            agent: np.zeros(self.validator.agent_observation_specs[agent]['total_dim'], dtype=np.float32)
            for agent in self.base_env.possible_agents
        }

        self.forecast_cache: Dict[str, Dict[str, float]] = {}
        self.agent_forecast_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.cache_size_limit = 500

        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=300)
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
                f = self._get_cached_forecasts_global(timestep)
        except Exception:
            f = {}
        self._bounded_put(self.agent_forecast_cache, key, f)
        return f

    def _bounded_put(self, d: Dict, key, value):
        if len(d) >= self.cache_size_limit:
            drop = int(len(d) * 0.6)
            for k in list(d.keys())[:drop]:
                del d[k]
        d[key] = value

    def get_diagnostic_info(self) -> Dict:
        return {
            'validator_stats': self.validator.get_validation_stats(),
            'memory_stats': self.memory_tracker.get_memory_stats(),
            'cache_size': {'global': len(self.forecast_cache), 'per_agent': len(self.agent_forecast_cache)},
            'cache_limit': self.cache_size_limit
        }

# =========================
# Wrapper Env
# =========================
class MultiHorizonWrapperEnv(ParallelEnv):
    """Wraps a BASE-only env and exposes TOTAL-dim observations (base + forecasts)."""
    def __init__(self, base_env, multi_horizon_forecaster, log_path=None, max_memory_mb=1500):
        self.env = base_env
        self.forecaster = multi_horizon_forecaster

        # Agents
        self._possible_agents = self.env.possible_agents[:]
        self._agents = self.env.agents[:]

        # Memory & logging infra
        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=max_memory_mb)
        self.log_path = self._setup_logging_path(log_path)
        self.log_buffer = deque(maxlen=256)
        self.log_lock = threading.RLock() if threading else nullcontext()
        self._flush_every_rows = 1000

        # Inject forecaster reference into env if empty
        if hasattr(self.env, 'forecast_generator') and getattr(self.env, 'forecast_generator') is None:
            try:
                self.env.forecast_generator = self.forecaster
            except Exception:
                pass

        # Builder / specs
        self.obs_builder = MemoryOptimizedObservationBuilder(self.env, self.forecaster, debug=False)
        self.memory_tracker.register_cache(self.obs_builder.forecast_cache)
        self.memory_tracker.register_cache(self.obs_builder.agent_forecast_cache)
        self.memory_tracker.register_cache(self.log_buffer)

        # Observation spaces (TOTAL dims)
        self._build_wrapper_observation_spaces()

        # counters & episode markers
        self.step_count = 0
        self.episode_count = 0
        self.error_count = 0
        self.max_errors = 50
        self._ep_meta_return = 0.0
        self._prev_forecasts_for_error: Dict[str, float] = {}
        self._log_interval = 20
        self._last_episode_seed = None
        self._last_episode_end_flag = 0
        self._last_step_wall_ms = 0.0
        self._last_clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}
        self._last_clip_totals = {"investor": 1, "battery": 1, "risk": 1, "meta": 1}

        # logging
        self._initialize_logging_safe()
        print("✅ Enhanced multi-horizon wrapper initialized (TOTAL-dim observations)")

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
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"fallback_metrics_{ts}.csv"
        return log_path

    def _initialize_logging_safe(self):
        try:
            if self.log_path and not os.path.isfile(self.log_path):
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
            # immediate forecasts
            "wind_forecast_immediate", "solar_forecast_immediate", "hydro_forecast_immediate", "price_forecast_immediate", "load_forecast_immediate",
            # perf & quick risks
            "portfolio_performance", "portfolio_value", "equity", "total_return_nav", "overall_risk", "market_risk",  # <-- PATCH: added value/equity + total return
            # env snapshot
            "budget", "wind_cap", "solar_cap", "hydro_cap", "battery_energy",
            "price_actual", "load_actual", "wind_actual", "solar_actual", "hydro_actual",
            "market_stress", "market_volatility", "revenue_step",
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
            "mae_price_1", "mae_wind_1", "mae_solar_1", "mae_load_1", "mae_hydro_1"
        ]
        with open(self.log_path, "w", newline="") as f:
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

        # optional initial update
        try:
            if hasattr(self.env, "data") and len(self.env.data) > 0 and hasattr(self.forecaster, "update"):
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

        # build enhanced obs
        enhanced = self.obs_builder.enhance_observations(obs)
        validated = self._validate_observations_safe(enhanced)

        self.step_count += 1
        self._last_step_wall_ms = (time.perf_counter() - t0) * 1000.0

        # periodic logging
        if self.log_path and (self.step_count % self._log_interval == 0 or self._last_episode_end_flag):
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
                if hasattr(self.obs_builder, 'forecast_cache') and len(self.obs_builder.forecast_cache) > 100:
                    for k in list(self.obs_builder.forecast_cache.keys())[: int(len(self.obs_builder.forecast_cache)*0.7)]:
                        del self.obs_builder.forecast_cache[k]
                if hasattr(self.obs_builder, 'agent_forecast_cache') and len(self.obs_builder.agent_forecast_cache) > 100:
                    for k in list(self.obs_builder.agent_forecast_cache.keys())[: int(len(self.obs_builder.agent_forecast_cache)*0.7)]:
                        del self.obs_builder.agent_forecast_cache[k]
                self._flush_log_buffer()
                if hasattr(self.obs_builder.validator, 'validation_cache'):
                    self.obs_builder.validator.validation_cache.clear()
                gc.collect()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Memory cleanup failed: {e}")
                self.error_count += 1

    def _validate_actions_comprehensive(self, actions, track_clipping=False):
        validated = {}
        clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}
        clip_totals = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        for agent in self.possible_agents:
            space = self.env.action_space(agent)
            a_in = actions.get(agent, None)

            # Normalize to ndarray/list for uniform handling
            orig = None
            if isinstance(space, spaces.Discrete):
                # Discrete → int
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
                    arr = np.array([(space.low + space.high)[0] / 2.0], dtype=np.float32)
            else:
                arr = a_in.astype(np.float32)

            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim > 1:
                arr = arr.flatten()

            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            if arr.size != space.shape[0]:
                if arr.size < space.shape[0]:
                    middle = (space.low + space.high) / 2.0
                    m = float(middle[0]) if hasattr(middle, '__len__') else float(middle)
                    arr = np.concatenate([arr, np.full(space.shape[0] - arr.size, m, dtype=np.float32)])
                else:
                    arr = arr[: space.shape[0]]

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
        try:
            if hasattr(self.env, "data") and 0 < self.env.t < len(self.env.data):
                row = self.env.data.iloc[self.env.t - 1]
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

            # Prefer env.equity if available; otherwise reconstruct
            if hasattr(self.env, 'equity'):
                portfolio_value = float(self._safe_float(getattr(self.env, 'equity', initial_budget), initial_budget))
            else:
                cash  = float(self._safe_float(getattr(self.env, 'budget', initial_budget), initial_budget))
                w_val = float(self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0))
                s_val = float(self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0))
                h_val = float(self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0))
                portfolio_value = cash + w_val + s_val + h_val

            # Don't clip performance to allow realistic returns (both positive and negative)
            perf = float(SafeDivision._safe_divide(portfolio_value, initial_budget, 1.0))
            perf_clipped = float(np.clip(perf, 0.0, 10.0))  # Keep clipped version for observations
            overall_risk = float(np.clip(getattr(self.env, "overall_risk_snapshot", 0.5), 0.0, 1.0))
            market_risk  = float(np.clip(getattr(self.env, "market_risk_snapshot", 0.5), 0.0, 1.0))
            # Keep portfolio_value on instance for logging reuse
            self._last_portfolio_value = portfolio_value
            return [perf_clipped, overall_risk, market_risk]  # Use clipped version for observations
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

                # core
                meta_reward = self._safe_float(rewards.get("meta_controller_0", 0.0), 0.0)
                inv_freq = int(getattr(self.env, "investment_freq", -1))
                cap_frac = self._safe_float(getattr(self.env, "capital_allocation_fraction", -1), -1.0)

                # actions (0-padded to expected lengths)
                def get_action_vec(agent, n):
                    a = actions.get(agent, None)
                    if isinstance(a, np.ndarray):
                        vec = a.flatten().tolist()
                    elif np.isscalar(a):
                        vec = [float(a)]
                    elif isinstance(a, (list, tuple)):
                        vec = list(a)
                    else:
                        vec = []
                    out = []
                    for v in vec:
                        try:
                            out.append(float(v) if np.isfinite(v) else 0.0)
                        except Exception:
                            out.append(0.0)
                    while len(out) < n:
                        out.append(0.0)
                    return out[:n]

                meta_a0, meta_a1 = get_action_vec("meta_controller_0", 2)
                inv_a0, inv_a1, inv_a2 = get_action_vec("investor_0", 3)
                batt_a0 = get_action_vec("battery_operator_0", 1)[0]
                risk_a0 = get_action_vec("risk_controller_0", 1)[0]

                # Forecasts logged (incl. hydro)
                forecast_keys = ["wind_forecast_immediate", "solar_forecast_immediate", "hydro_forecast_immediate", "price_forecast_immediate", "load_forecast_immediate"]
                forecasts_logged = [self._safe_float(all_forecasts.get(k, 0.0), 0.0) for k in forecast_keys]

                # perf & quick risks (perf based on true equity)
                perf, overall_risk, market_risk = self._calc_perf_and_quick_risks()

                # Compute/log true equity & portfolio_value
                if hasattr(self, '_last_portfolio_value') and self._last_portfolio_value is not None:
                    portfolio_value_logged = float(self._last_portfolio_value)
                else:
                    # reconstruct if needed
                    cash  = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                    w_val = self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0)
                    s_val = self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0)
                    h_val = self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0)
                    portfolio_value_logged = cash + w_val + s_val + h_val
                equity_logged = portfolio_value_logged  # same thing, explicit for analyzer

                # Total return NAV (includes distributed profits)
                distributed_profits = self._safe_float(getattr(self.env, 'distributed_profits', 0.0), 0.0)
                total_return_nav = portfolio_value_logged + distributed_profits

                # env snapshot
                budget = self._safe_float(getattr(self.env, 'budget', 0.0), 0.0)
                wind_c = self._safe_float(getattr(self.env, 'wind_capacity', 0.0), 0.0)
                solar_c = self._safe_float(getattr(self.env, 'solar_capacity', 0.0), 0.0)
                hydro_c = self._safe_float(getattr(self.env, 'hydro_capacity', 0.0), 0.0)
                batt_e = self._safe_float(getattr(self.env, 'battery_energy', 0.0), 0.0)

                actuals = self._get_actuals_for_logging()
                price_act = self._safe_float(actuals['price'], 0.0)
                load_act  = self._safe_float(actuals['load'], 0.0)
                wind_act  = self._safe_float(actuals['wind'], 0.0)
                solar_act = self._safe_float(actuals['solar'], 0.0)
                hydro_act = self._safe_float(actuals['hydro'], 0.0)

                market_stress = self._safe_float(getattr(self.env, 'market_stress', 0.0), 0.0)
                market_vol = self._safe_float(getattr(self.env, 'market_volatility', 0.0), 0.0)

                # prefer last_revenue; fallback to revenue
                revenue_step = self._safe_float(getattr(self.env, 'last_revenue', getattr(self.env, 'revenue', 0.0)), 0.0)

                # episode markers
                step_in_ep = int(getattr(self.env, 'step_in_episode', self.step_count))
                episode_end = int(self._last_episode_end_flag)

                # risk 6-vector
                r6 = self._get_risk_vector6()

                # reward breakdown & weights
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

                # ops & scaling health
                mem_mb = float(self.memory_tracker.get_memory_usage())
                seed = self._last_episode_seed if self._last_episode_seed is not None else -1
                step_ms = float(self._last_step_wall_ms)
                def clip_frac(key):
                    tot = max(1, self._last_clip_totals.get(key, 1))
                    return float(SafeDivision._safe_divide(self._last_clip_counts.get(key, 0), tot, 0.0))
                clip_inv = clip_frac("investor"); clip_bat = clip_frac("battery"); clip_rsk = clip_frac("risk"); clip_meta = clip_frac("meta")

                # MAE@1 using previous forecast cache
                def _mae(prev, actual):
                    return abs(self._safe_float(prev, 0.0) - self._safe_float(actual, 0.0))
                mae_price = _mae(self._prev_forecasts_for_error.get("price_forecast_immediate"), price_act)
                mae_wind  = _mae(self._prev_forecasts_for_error.get("wind_forecast_immediate"),  wind_act)
                mae_solar = _mae(self._prev_forecasts_for_error.get("solar_forecast_immediate"), solar_act)
                mae_load  = _mae(self._prev_forecasts_for_error.get("load_forecast_immediate"),  load_act)
                mae_hydro = _mae(self._prev_forecasts_for_error.get("hydro_forecast_immediate"), hydro_act)
                
                # Update previous forecast cache for next MAE calculation
                self._prev_forecasts_for_error = {k: v for k, v in zip(forecast_keys, forecasts_logged)}

                row = [
                    ts_str, int(step_t), int(self.episode_count), meta_reward, inv_freq, cap_frac,
                    meta_a0, meta_a1, inv_a0, inv_a1, inv_a2, batt_a0, risk_a0,
                    *forecasts_logged,
                    # perf & quick risks (PATCH: include true value/equity + total return)
                    perf, portfolio_value_logged, equity_logged, total_return_nav, overall_risk, market_risk,
                    # env snapshot
                    budget, wind_c, solar_c, hydro_c, batt_e,
                    price_act, load_act, wind_act, solar_act, hydro_act,
                    market_stress, market_vol, revenue_step,
                    self._ep_meta_return, step_in_ep, episode_end,
                    *r6,
                    r_fin, r_risk, r_sus, r_eff, r_div,
                    w_fin, w_rsk, w_sus, w_eff, w_div,
                    self.episode_count, seed, step_ms, mem_mb,
                    clip_inv, clip_bat, clip_rsk, clip_meta,
                    mae_price, mae_wind, mae_solar, mae_load, mae_hydro
                ]

                self.log_buffer.append(row)
                # hard flush triggers
                if (len(self.log_buffer) >= 25 or
                    self.step_count % 500 == 0 or
                    len(self.log_buffer) >= self._flush_every_rows or
                    episode_end):
                    self._flush_log_buffer()
        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"⚠️ Metrics logging failed: {e}")
                self.error_count += 1

    def _flush_log_buffer(self):
        """
        Safely flushes the in-memory log buffer to the CSV file.
        """
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
