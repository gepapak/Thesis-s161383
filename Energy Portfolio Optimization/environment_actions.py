# environment_actions.py
# Action parsing helpers extracted from environment.py.

from __future__ import annotations

from typing import Any, Dict
import numpy as np


class ActionProcessor:
    """Processes and validates agent actions."""

    def __init__(self, config: Any):
        self.config = config

    def process_actions(self, raw_actions: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        processed_actions: Dict[int, Dict[str, Any]] = {}
        for agent_id, action_array in raw_actions.items():
            arr = np.asarray(action_array, dtype=np.float32).reshape(-1)
            clipped = self._clip_action(arr)
            processed_actions[agent_id] = self._array_to_action(clipped)
        return processed_actions

    def _array_to_action(self, action_array: np.ndarray) -> Dict[str, Any]:
        num_assets = int(getattr(self.config, "num_assets", 3))
        num_hedges = int(getattr(self.config, "num_hedges", 0))
        return {
            "asset_allocation": action_array[:num_assets],
            "hedge_positions": action_array[num_assets : num_assets + num_hedges],
            "timestamp": None,
        }

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, -1.0, 1.0)

    def _get_default_action(self) -> Dict[str, Any]:
        num_assets = int(getattr(self.config, "num_assets", 3))
        num_hedges = int(getattr(self.config, "num_hedges", 0))
        return {
            "asset_allocation": np.zeros(num_assets, dtype=np.float32),
            "hedge_positions": np.zeros(num_hedges, dtype=np.float32),
            "timestamp": None,
        }

    def validate_action_bounds(self, action: Dict[str, Any]) -> bool:
        asset_alloc = np.asarray(action.get("asset_allocation", []), dtype=float)
        hedge_alloc = np.asarray(action.get("hedge_positions", []), dtype=float)
        return bool(np.all(asset_alloc <= 1.0) and np.all(asset_alloc >= -1.0) and np.all(hedge_alloc <= 1.0) and np.all(hedge_alloc >= -1.0))
