#!/usr/bin/env python3
"""
Run Tier1/Tier2 + Tier3 FGB ablations sequentially.

This script is purpose-built for the FGB/FAMC ablation matrix:
1) Tier 1 (baseline, no forecasts)
2) Tier 2 (forecast observations)
3) Tier 3 + FAMC meta    (--forecast_baseline_enable --fgb_mode meta)   [auto-enables meta head in main.py]
4) Tier 3 + FGB online   (--forecast_baseline_enable --fgb_mode online)
5) Tier 3 + FGB fixed    (--forecast_baseline_enable --fgb_mode fixed)

It does NOT run the risk-uplift variants (those are in run_all_tiers.py).
"""

import subprocess
import sys
import time
from datetime import datetime
import argparse


def run_command(cmd, name: str) -> bool:
    print(f"\n{'='*90}")
    print(f"Starting: {name}")
    print(f"{'='*90}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}\n")

    start = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()

        elapsed = time.time() - start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        ok = (process.returncode == 0)
        print(f"\n{'='*90}")
        print(f"{'SUCCESS' if ok else 'FAILED'}: {name} (exit={process.returncode})")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {h}h {m}m {s}s")
        print(f"{'='*90}\n")
        return ok
    except KeyboardInterrupt:
        print(f"\n\nInterrupted: {name}")
        try:
            process.terminate()
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"\n\nError running {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Tier1/Tier2 + Tier3 FGB ablations sequentially.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to pass to main.py (default: 42)")
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
    args = parser.parse_args()

    py = sys.executable  # robust on Colab (python3) and Windows (python.exe)

    # Common args: edit here once, affects all runs.
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
    ]

    seed_idx = common_args.index("--seed") + 1
    seed_value = common_args[seed_idx] if seed_idx < len(common_args) else str(args.seed)

    prefix = args.save_prefix or ""

    tier1 = [py, "main.py"] + common_args + ["--save_dir", f"{prefix}tier1gnn_seed{seed_value}_fgbabl"]

    tier2 = [py, "main.py"] + common_args + [
        "--save_dir", f"{prefix}tier2gnn_seed{seed_value}_fgbabl",
        "--enable_forecast_utilisation",
    ]

    def tier3(mode: str):
        return [py, "main.py"] + common_args + [
            "--save_dir", f"{prefix}tier3gnn_seed{seed_value}_fgb_{mode}",
            "--enable_forecast_utilisation",
            "--forecast_baseline_enable",
            "--fgb_mode", mode,
        ]

    # User-requested order: Tier 1, Tier 2, Tier 3 meta, Tier 3 online, Tier 3 fixed
    plan = [
        ("Tier 1 (baseline)", tier1),
        ("Tier 2 (forecast obs)", tier2),
        ("Tier 3 + FAMC meta", tier3("meta")),
        ("Tier 3 + FGB online", tier3("online")),
        ("Tier 3 + FGB fixed", tier3("fixed")),
    ]

    print("\n" + "=" * 90)
    print("FGB/FAMC ABLATION RUN (Tier 1/2 + Tier 3 meta/online/fixed)")
    print("=" * 90)
    print(f"Seed: {seed_value}")
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
    print("ABLATION RUN COMPLETE")
    print("=" * 90)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {h}h {m}m {s}s")
    for name, ok in results.items():
        print(f"  - {name}: {'SUCCESS' if ok else 'FAILED'}")
    print("=" * 90 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()

