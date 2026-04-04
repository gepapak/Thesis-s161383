# environment_rewards.py
# Reward helper logic extracted from environment.py.

from __future__ import annotations

from typing import Any, Dict


class SplitRewardCalculator:
    """Split helper for reward computations; does not replace live env reward logic."""

    def __init__(self, config: Any):
        self.config = config

    def calculate_rewards(
        self,
        actions: Dict[int, Dict[str, Any]],
        observations: Dict[int, Dict[str, Any]],
        timestep: int,
    ) -> Dict[int, float]:
        _ = timestep
        rewards: Dict[int, float] = {}
        for agent_id, action in actions.items():
            obs = observations.get(agent_id, {})
            rewards[agent_id] = self._calculate_agent_reward(agent_id, action, obs)
        return rewards

    def _calculate_agent_reward(
        self,
        agent_id: int,
        action: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> float:
        _ = (agent_id, action)
        pnl = float(observation.get("pnl", 0.0))
        risk = float(observation.get("risk", 0.0))
        nav = float(observation.get("nav", 1.0))
        risk_penalty_weight = float(getattr(self.config, "risk_penalty_weight", 0.0))
        return float((pnl / max(abs(nav), 1e-6)) - (risk_penalty_weight * risk))
