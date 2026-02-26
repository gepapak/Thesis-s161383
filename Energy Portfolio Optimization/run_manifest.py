#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight run-manifest utilities for reproducibility.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"value": repr(obj)}


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {str(k): _to_jsonable(v) for k, v in vars(value).items()}
    return repr(value)


def _get_git_info(cwd: str) -> Dict[str, Optional[str]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = None

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        branch = None

    return {"commit": commit, "branch": branch}


def write_run_manifest(
    output_dir: str,
    argv: Optional[list] = None,
    args: Any = None,
    config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    filename_prefix: str = "run_manifest",
) -> str:
    """
    Write a reproducibility manifest JSON and return its path.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = _iso_now()
    stamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{stamp_file}.json"
    path = os.path.join(output_dir, filename)

    cwd = os.getcwd()
    payload = {
        "timestamp_utc": timestamp,
        "cwd": cwd,
        "argv": list(argv) if argv is not None else list(sys.argv),
        "args": _to_jsonable(_as_dict(args)),
        "config": _to_jsonable(_as_dict(config)),
        "extra": _to_jsonable(extra or {}),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "git": _get_git_info(cwd),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True)

    return path

