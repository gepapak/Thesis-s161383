"""CLI helpers for forecast-prior ablation settings."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from runtime_contract import FORECAST_PRIOR_CONTRACT_KEYS


def add_forecast_prior_override_args(parser) -> None:
    """Expose forecast-prior contract keys as optional CLI overrides."""
    for key, _default, caster in FORECAST_PRIOR_CONTRACT_KEYS:
        parser.add_argument(
            f"--{key}",
            type=caster,
            default=None,
            help=f"Override {key} for forecast-utilization ablations.",
        )


def collect_forecast_prior_overrides(source: Any) -> Dict[str, Any]:
    """Return explicitly supplied forecast-prior override values."""
    overrides: Dict[str, Any] = {}
    for key, _default, caster in FORECAST_PRIOR_CONTRACT_KEYS:
        value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        if value is None:
            continue
        overrides[key] = caster(value)
    return overrides


def apply_forecast_prior_overrides(config: Any, source: Any) -> Dict[str, Any]:
    """Apply explicit forecast-prior overrides to an EnhancedConfig-like object."""
    overrides = collect_forecast_prior_overrides(source)
    for key, value in overrides.items():
        setattr(config, key, value)
        if key == "forecast_prior_horizon_steps":
            horizons = dict(getattr(config, "forecast_horizons", {}) or {})
            horizons["short"] = int(value)
            config.forecast_horizons = horizons
    return overrides


def forecast_prior_override_cli_args(source: Any) -> List[str]:
    """Serialize explicit forecast-prior overrides back into CLI args."""
    args: List[str] = []
    for key, value in collect_forecast_prior_overrides(source).items():
        args.extend([f"--{key}", str(value)])
    return args


def format_forecast_prior_overrides(overrides: Dict[str, Any]) -> str:
    """Stable one-line representation for logs/protocols."""
    return ", ".join(f"{k}={overrides[k]}" for k in sorted(overrides))


__all__ = [
    "add_forecast_prior_override_args",
    "apply_forecast_prior_overrides",
    "collect_forecast_prior_overrides",
    "forecast_prior_override_cli_args",
    "format_forecast_prior_overrides",
]
