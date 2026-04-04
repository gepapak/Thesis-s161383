# environment_observations.py
# Observation helper logic extracted from environment.py.

from __future__ import annotations

from typing import Any, Dict
import numpy as np


class ObservationBuilder:
    """Builds per-agent observations from split state dictionaries."""

    def __init__(self, config: Any, observation_manager: Any):
        self.config = config
        self.observation_manager = observation_manager

    def build_observations(self, state: Dict[str, Any], timestep: int) -> Dict[int, np.ndarray]:
        num_agents = int(getattr(self.config, "num_agents", 1))
        observations: Dict[int, np.ndarray] = {}
        for agent_id in range(num_agents):
            observations[agent_id] = self._build_agent_observation(agent_id, state, timestep)
        return observations

    def _build_agent_observation(self, agent_id: int, state: Dict[str, Any], timestep: int) -> np.ndarray:
        _ = (agent_id, state, timestep)
        obs_size = int(getattr(self.config, "observation_space_size", 1))
        obs = np.zeros(obs_size, dtype=np.float32)

        use_stab = bool(getattr(self.config, "use_observation_stabilization", False))
        stabilize = getattr(self.observation_manager, "stabilize_observation", None)
        if use_stab and callable(stabilize):
            try:
                return np.asarray(stabilize(obs), dtype=np.float32)
            except Exception:
                return obs
        return obs

    def get_observation_space_size(self) -> int:
        return int(getattr(self.config, "observation_space_size", 1))
