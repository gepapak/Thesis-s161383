#!/usr/bin/env python3
"""Observation validation for the cache-only Tier-1 runtime.

Forecast utilization is intentionally not a wrapper feature. The single public
flag, ``enable_forecast_utilization``, loads a cached ANN forecast prior inside
``RenewableMultiAgentEnv`` and applies it only during investor execution.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils import UnifiedObservationValidator


BaseObservationValidator = UnifiedObservationValidator


class ObservationValidatorMixin:
    """Common shape/range helpers used by training and metacontroller code."""

    def fix_observation_shape(self, obs: Any, expected_dim: int) -> np.ndarray:
        strict = bool(getattr(self, "strict_validation", False))
        if not isinstance(obs, np.ndarray):
            if obs is None:
                obs = np.zeros(expected_dim, dtype=np.float32)
            elif isinstance(obs, (list, tuple)):
                obs = np.array(obs, dtype=np.float32)
            else:
                obs = np.full(expected_dim, float(obs), dtype=np.float32)
        else:
            obs = obs.astype(np.float32)

        if obs.ndim != 1:
            obs = obs.flatten()

        if obs.size != expected_dim:
            if strict:
                raise ValueError(
                    f"[OBS_DIM_MISMATCH] expected {expected_dim} dims, got {obs.size}. "
                    "strict_validation=True."
                )
            if obs.size < expected_dim:
                obs = np.pad(obs, (0, expected_dim - obs.size))
            else:
                obs = obs[:expected_dim]
        return obs

    def create_fallback_observation(
        self,
        expected_dim: int,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if low is not None and high is not None:
            return ((low[:expected_dim] + high[:expected_dim]) / 2.0).astype(np.float32)
        return np.zeros(expected_dim, dtype=np.float32)


class EnhancedObservationValidator(BaseObservationValidator, ObservationValidatorMixin):
    """Validate Tier-1 observations without adding forecast features."""

    def __init__(self, base_env, debug: bool = False, strict_validation: bool = True):
        super().__init__(base_env, debug)
        self.base_env = base_env
        self.strict_validation = bool(strict_validation)
        self.agent_observation_specs: Dict[str, Dict[str, Any]] = {}
        self._initialize_observation_specs()
        self.observation_specs = self.agent_observation_specs

    def _initialize_observation_specs(self) -> None:
        for agent in self.base_env.possible_agents:
            try:
                base_dim = self._get_validated_base_dim(agent)
                low, high = self._get_safe_bounds(base_dim)
                self.agent_observation_specs[agent] = {
                    "base_dim": base_dim,
                    "forecast_dim": 0,
                    "total_dim": base_dim,
                    "forecast_keys": [],
                    "low": low,
                    "high": high,
                    "bounds": (low, high),
                }
                if self.debug or agent == "meta_controller_0":
                    print(f"[SPEC] {agent}: base={base_dim}, forecast=0, total={base_dim}")
            except Exception as exc:
                logging.getLogger(__name__).error("Failed to init specs for %s: %s", agent, exc)
                self._create_fallback_spec(agent)

    def _get_validated_base_dim(self, agent: str) -> int:
        try:
            if hasattr(self.base_env, "_get_base_observation_dim"):
                dim = int(self.base_env._get_base_observation_dim(agent))
                if dim <= 0:
                    raise ValueError(f"invalid base dim {dim}")
                return dim

            obs_space = getattr(self.base_env, "observation_space", None)
            if obs_space is None:
                raise ValueError("environment has no observation_space")
            space = obs_space(agent) if callable(obs_space) else obs_space.get(agent)
            if space is None or not hasattr(space, "shape"):
                raise ValueError(f"missing observation space for {agent}")
            dim = int(space.shape[0])
            if dim <= 0:
                raise ValueError(f"invalid observation space dim {dim}")
            return dim
        except Exception as exc:
            raise ValueError(f"Failed to get base observation dimension for {agent}: {exc}") from exc

    def _get_safe_bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(dim, -10.0, dtype=np.float32),
            np.full(dim, 10.0, dtype=np.float32),
        )

    def _create_fallback_spec(self, agent: str) -> None:
        base_dim = self._get_validated_base_dim(agent)
        low, high = self._get_safe_bounds(base_dim)
        self.agent_observation_specs[agent] = {
            "base_dim": base_dim,
            "forecast_dim": 0,
            "total_dim": base_dim,
            "forecast_keys": [],
            "low": low,
            "high": high,
            "bounds": (low, high),
        }

    def build_total_observation(
        self,
        agent: str,
        base_obs: np.ndarray,
        forecasts: Optional[Dict[str, float]] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if forecasts:
            raise ValueError("Forecast observation features are disabled in the cache-only runtime.")
        if agent not in self.agent_observation_specs:
            return self._create_safe_observation(agent)
        spec = self.agent_observation_specs[agent]
        base_dim = int(spec["base_dim"])
        total_dim = int(spec["total_dim"])
        if out is None or not isinstance(out, np.ndarray) or out.shape != (total_dim,):
            out = np.zeros(total_dim, dtype=np.float32)
        out[:base_dim] = self._validate_base_observation(agent, base_obs, spec)
        low, high = spec["bounds"]
        np.clip(out, low[:total_dim], high[:total_dim], out=out)
        return out

    def _validate_base_observation(self, agent: str, base_obs: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
        base_dim = int(spec["base_dim"])
        fixed = self.fix_observation_shape(base_obs, base_dim)
        fixed = np.nan_to_num(fixed, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        low, high = spec["bounds"]
        np.clip(fixed, low[:base_dim], high[:base_dim], out=fixed)
        return fixed

    def _create_safe_observation(self, agent: str) -> np.ndarray:
        spec = self.agent_observation_specs.get(agent)
        if spec is None:
            raise ValueError(f"Cannot create safe observation for unknown agent {agent!r}")
        expected = int(spec["total_dim"])
        low, high = spec["bounds"]
        return ((low[:expected] + high[:expected]) / 2.0).astype(np.float32)

    def fix_observation(self, agent: str, obs: Any) -> np.ndarray:
        spec = self.agent_observation_specs.get(agent)
        if spec is None:
            self._create_fallback_spec(agent)
            spec = self.agent_observation_specs[agent]
        expected = int(spec.get("total_dim", spec.get("base_dim", 1)))
        fixed = self.fix_observation_shape(obs, expected)
        fixed = np.nan_to_num(fixed, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        low, high = spec.get("bounds", (None, None))
        if low is not None and high is not None:
            np.clip(fixed, low[:expected], high[:expected], out=fixed)
        return fixed

    def get_validation_stats(self) -> Dict[str, Any]:
        return {
            "agents_configured": len(self.agent_observation_specs),
            "recent_errors": len(self.validation_errors),
            "cache_size": len(self.validation_cache),
            "specs": {
                agent: {
                    "base_dim": spec["base_dim"],
                    "forecast_dim": 0,
                    "total_dim": spec["total_dim"],
                }
                for agent, spec in self.agent_observation_specs.items()
            },
            "recent_error_messages": list(self.validation_errors)[-5:],
        }
