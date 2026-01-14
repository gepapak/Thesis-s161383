#!/usr/bin/env python3
"""
Run Tier 3 ONLY (FGB/FAMC) in a specific order: meta → online → fixed.

Purpose:
- You asked for a Tier 3-only sweep (no Tier 1/2), starting from meta mode.
- Uses the same CLI flags as `main.py` Tier 3:
  --enable_forecast_utilisation --forecast_baseline_enable --fgb_mode {meta,online,fixed}

Notes:
- `main.py` auto-enables `--meta_baseline_enable` when `--fgb_mode meta` is selected.
- This script avoids accidental overwrites: if a save_dir exists and --overwrite is NOT set,
  it appends a timestamp suffix.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd, name: str) -> bool:
    print(f"\n{'=' * 90}")
    print(f"Starting: {name}")
    print(f"{'=' * 90}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 90}\n")

    start = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
        process.wait()

        elapsed = time.time() - start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        ok = (process.returncode == 0)
        print(f"\n{'=' * 90}")
        print(f"{'SUCCESS' if ok else 'FAILED'}: {name} (exit={process.returncode})")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {h}h {m}m {s}s")
        print(f"{'=' * 90}\n")
        return ok
    except KeyboardInterrupt:
        print(f"\n\nInterrupted: {name}")
        try:
            process.terminate()  # type: ignore[name-defined]
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"\n\nError running {name}: {e}")
        return False


def resolve_save_dir(base_dir: str, overwrite: bool) -> str:
    p = Path(base_dir)
    if overwrite or not p.exists():
        return str(p)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(p.parent / f"{p.name}_run{ts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier 3 ONLY: meta → online → fixed (seed 789 by default).")
    parser.add_argument("--seed", type=int, default=789, help="Random seed to pass to main.py (default: 789)")
    parser.add_argument("--start_episode", type=int, default=0, help="Start episode index (default: 0)")
    parser.add_argument("--end_episode", type=int, default=19, help="End episode index (default: 19)")
    parser.add_argument("--cooling_period", type=int, default=1, help="Cooling period between episodes (default: 1)")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Checkpoint frequency (default: 5000)")
    parser.add_argument("--investment_freq", type=int, default=48, help="Investment frequency (default: 48)")
    parser.add_argument(
        "--episode_data_dir",
        type=str,
        default="training_dataset",
        help="Directory containing scenario_*.csv episode files (default: training_dataset)",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="",
        help="Optional prefix for save_dir names (e.g., 'colab_' to avoid collisions).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, reuse save_dir even if it already exists (default: False).",
    )
    args = parser.parse_args()

    py = sys.executable  # robust on Colab (python3) and Windows (python.exe)
    prefix = args.save_prefix or ""

    common_args = [
        "--episode_training",
        "--episode_data_dir", args.episode_data_dir,
        "--start_episode", str(args.start_episode),
        "--end_episode", str(args.end_episode),
        "--cooling_period", str(args.cooling_period),
        "--checkpoint_freq", str(args.checkpoint_freq),
        "--seed", str(args.seed),
        "--investment_freq", str(args.investment_freq),
        "--enable_gnn_encoder",
        "--enable_forecast_utilisation",
        "--forecast_baseline_enable",
    ]

    def tier3_cmd(mode: str):
        base_save_dir = f"{prefix}tier3gnn_seed{args.seed}_fgb_{mode}"
        save_dir = resolve_save_dir(base_save_dir, overwrite=args.overwrite)
        cmd = [py, "main.py"] + common_args + ["--save_dir", save_dir, "--fgb_mode", mode]
        if mode == "meta":
            # Redundant but explicit; main.py also auto-enables this for meta mode.
            cmd += ["--meta_baseline_enable"]
        return cmd, save_dir

    modes = ["meta", "online", "fixed"]
    plan = []
    for mode in modes:
        cmd, save_dir = tier3_cmd(mode)
        plan.append((f"Tier 3 ({mode}) → save_dir={save_dir}", cmd))

    print("\n" + "=" * 90)
    print("TIER 3 ONLY SWEEP (meta → online → fixed)")
    print("=" * 90)
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.start_episode}..{args.end_episode}")
    print(f"Episode data: {args.episode_data_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    for i, (name, _) in enumerate(plan, 1):
        print(f"  {i}. {name}")
    print("=" * 90 + "\n")

    results = {}
    overall_start = time.time()
    for name, cmd in plan:
        ok = run_command(cmd, name)
        results[name] = ok
        if not ok:
            print(f"\nStopping: {name} failed.\n")
            sys.exit(1)

    elapsed = time.time() - overall_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print("\n" + "=" * 90)
    print("TIER 3 SWEEP COMPLETE")
    print("=" * 90)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {h}h {m}m {s}s")
    for name, ok in results.items():
        print(f"  - {name}: {'SUCCESS' if ok else 'FAILED'}")
    print("=" * 90 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()

