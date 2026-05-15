"""Runtime contract helpers for train/eval consistency checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


FORECAST_PRIOR_CONTRACT_KEYS = (
    ("forecast_prior_window", 500, int),
    ("forecast_prior_min_samples", 50, int),
    ("forecast_prior_hit_lcb_z", 1.64, float),
    ("forecast_prior_residual_quantile", 0.70, float),
    ("forecast_prior_default_residual", 0.10, float),
    ("forecast_prior_edge_gain", 3.0, float),
    ("forecast_prior_error_hurdle", 0.50, float),
    ("forecast_prior_skill_power", 1.0, float),
    ("forecast_prior_max_abs_exposure", 0.60, float),
    ("forecast_prior_residual_scale", 0.10, float),
    ("forecast_prior_blend", 0.85, float),
    ("forecast_prior_vol_half_life_steps", 288, int),
    ("forecast_prior_vol_target", 0.15, float),
    ("forecast_prior_horizon_steps", 6, int),
    ("forecast_prior_denom_floor", 10.0, float),
)


def forecast_prior_contract_settings(source: Any) -> Dict[str, Any]:
    """Extract the forecast-prior settings that must match between train/eval."""
    out: Dict[str, Any] = {}
    for key, default, caster in FORECAST_PRIOR_CONTRACT_KEYS:
        if key == "forecast_prior_horizon_steps":
            if isinstance(source, dict):
                if key in source:
                    value = source.get(key, default)
                else:
                    horizons = source.get("forecast_horizons", {}) or {}
                    value = horizons.get("short", source.get("short_horizon_steps", default)) if isinstance(horizons, dict) else default
            else:
                value = getattr(source, key, None)
                if value is None:
                    horizons = getattr(source, "forecast_horizons", {}) or {}
                    value = horizons.get("short", getattr(source, "short_horizon_steps", default)) if isinstance(horizons, dict) else default
        elif key == "forecast_prior_denom_floor":
            if isinstance(source, dict):
                value = source.get(key, source.get("minimum_price_filter", default))
            else:
                value = getattr(source, key, getattr(source, "minimum_price_filter", default))
        elif isinstance(source, dict):
            value = source.get(key, default)
        else:
            value = getattr(source, key, default)
        if value is None:
            value = default
        out[key] = caster(value)
    return out


def build_runtime_contract(
    *,
    global_norm_mode: str,
    rolling_past_history_dir: str,
    investment_freq: int,
    meta_freq_min: int,
    meta_freq_max: int,
    enable_forecast_utilization: bool = False,
    forecast_prior_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce the canonical Tier1 train/eval consistency contract."""
    contract = {
        "global_norm_mode": str(global_norm_mode or "rolling_past").strip().lower(),
        "rolling_past_history_dir": str(rolling_past_history_dir or "").strip(),
        "investment_freq": int(investment_freq),
        "meta_freq_min": int(meta_freq_min),
        "meta_freq_max": int(meta_freq_max),
        "enable_forecast_utilization": bool(enable_forecast_utilization),
    }
    if bool(enable_forecast_utilization):
        contract["forecast_prior"] = dict(forecast_prior_settings or {})
    return contract


def runtime_contract_hash(contract: Dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
