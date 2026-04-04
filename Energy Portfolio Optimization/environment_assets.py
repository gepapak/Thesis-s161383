# environment_assets.py
# Asset deployment and asset-level helper logic.

from __future__ import annotations

from typing import Any, Dict, Optional


class AssetManager:
    """Manages physical asset helper operations during split migration."""

    def __init__(self, config: Any):
        self.config = config
        self.assets_deployed = False

    def deploy_initial_assets(self, initial_asset_plan: Optional[dict] = None) -> dict:
        """Return the plan to be deployed; deployment remains in environment.py for now."""
        if self.assets_deployed:
            return initial_asset_plan or {}
        if initial_asset_plan is not None:
            plan = initial_asset_plan
        else:
            getter = getattr(self.config, "get_initial_asset_plan", None)
            plan = getter() if callable(getter) else {}
        self.assets_deployed = True
        return plan

    def get_asset_capex(self, currency: str = "DKK") -> Dict[str, float]:
        getter = getattr(self.config, "get_asset_capex", None)
        if not callable(getter):
            return {}
        result = getter(currency)
        return dict(result) if isinstance(result, dict) else {}

    def calculate_depreciation(self, timestep: int) -> Dict[str, float]:
        _ = timestep
        return {}

    def get_operational_costs(self) -> Dict[str, float]:
        return {}
