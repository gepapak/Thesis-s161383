# environment_core.py
# Core state helpers extracted from environment.py.

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


class EnvironmentCore:
    """Lightweight core state helper for environment refactoring."""

    def __init__(self, config: Any):
        self.config = config
        self._rng = np.random.default_rng()
        self._last_seed: Optional[int] = None

    def set_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._rng = np.random.default_rng()
            self._last_seed = None
            return
        self._rng = np.random.default_rng(int(seed))
        self._last_seed = int(seed)

    def basic_state_snapshot(self) -> Dict[str, Any]:
        """Return a normalized initial snapshot used during split migration."""
        init_budget = float(getattr(self.config, "init_budget", 0.0))
        fin_alloc = float(getattr(self.config, "financial_allocation", 0.0))
        return {
            "t": 0,
            "budget": init_budget * fin_alloc,
            "equity": init_budget,
            "physical_assets": {
                "wind_capacity_mw": 0.0,
                "solar_capacity_mw": 0.0,
                "hydro_capacity_mw": 0.0,
                "battery_capacity_mwh": 0.0,
            },
            "financial_positions": {
                "wind_instrument_value": 0.0,
                "solar_instrument_value": 0.0,
                "hydro_instrument_value": 0.0,
            },
            "operational_state": {
                "battery_energy": 0.0,
                "battery_discharge_power": 0.0,
            },
        }
