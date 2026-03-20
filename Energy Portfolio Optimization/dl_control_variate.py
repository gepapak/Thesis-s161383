#!/usr/bin/env python3
"""Backward-compatible Tier-2 module shim.

The canonical implementation now lives in ``tier2_routed_overlay.py``.
This shim is retained so older checkpoints, scripts, and imports keep working.
"""

from tier2_routed_overlay import *  # noqa: F401,F403
