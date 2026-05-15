"""Compatibility shim for the SARL baseline.

The SARL baseline must use the current project environment, not a copied
environment snapshot. Keeping this small shim preserves older imports while
making the current root environment the single source of truth.
"""

from pathlib import Path
import importlib.util
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ENV_PATH = PROJECT_ROOT / "environment.py"
_SPEC = importlib.util.spec_from_file_location("_tier1_current_environment", _ENV_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load current environment from {_ENV_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

RenewableMultiAgentEnv = _MODULE.RenewableMultiAgentEnv


__all__ = ["RenewableMultiAgentEnv"]
