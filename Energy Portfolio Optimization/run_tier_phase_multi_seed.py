#!/usr/bin/env python3
"""
Run canonical tier phases across multiple seeds.

Phase ordering per seed:
- tier1_tier2:
  1) Tier 1 hybrid RL baseline
  2) Tier 2 forecast-guided enhanced MARL baseline
- tier1_enhancer:
  1) Tier 1 hybrid RL baseline
  2) Tier 2 forecast-guided enhanced MARL baseline

For each run: train via main.py, evaluate via evaluation.py --mode tiers.
The runner forwards an explicit tier evaluation mode so suites can validate the
deployed Tier-2 path or stay in paper-only policy evaluation.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime


VARIANT_TRAIN_SPECS = {
    "tier1": {
        "name": "Tier 1 hybrid RL baseline",
        "slug": "tier1",
        "extra": [],
    },
    "tier2": {
        "name": "Tier 2 forecast-guided enhanced MARL baseline",
        "slug": "tier2",
        "extra": [
            "--forecast_baseline_enable",
        ],
    },
    "tier2_ablated": {
        "name": "Tier 2 forecast-ablated enhanced MARL control run",
        "slug": "tier2_ablated",
        "extra": [
            "--forecast_baseline_enable",
            "--tier2_value_ablate_forecast_features",
        ],
    },
}

PHASE_VARIANTS = {
    "tier1_tier2": [
        "tier1",
        "tier2",
    ],
    # Legacy alias kept only for backward compatibility with existing scripts/suite dirs.
    "tier1_enhancer": [
        "tier1",
        "tier2",
    ],
    "tier2_only": ["tier2"],
    "tier1_tier2_full_ablated": [
        "tier1",
        "tier2",
        "tier2_ablated",
    ],
}

EVAL_DIR_ARG_BY_VARIANT = {
    "tier1": "--tier1_dir",
    "tier2": "--tier2_dir",
    "tier2_ablated": "--tier2_ablated_dir",
}

# evaluation.py --tiers_only accepts canonical tier names, with "baseline" as
# a legacy alias for tier1.
EVAL_TIERS_ONLY_BY_VARIANT = {
    "tier1": "tier1",
    "tier2": "tier2",
    "tier2_ablated": "tier2_ablated",
}


def format_cmd(cmd):
    return subprocess.list2cmdline(cmd)


def run_command(cmd, name: str) -> dict:
    started_at = datetime.now()
    start = time.time()
    print(f"\n{'='*100}")
    print(f"Starting: {name}")
    print(f"Command: {format_cmd(cmd)}")
    print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        elapsed = time.time() - start
        ok = process.returncode == 0
        finished_at = datetime.now()
        print(f"\n{'='*100}")
        print(f"{'SUCCESS' if ok else 'FAILED'}: {name} (exit={process.returncode})")
        print(f"Duration: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s")
        print(f"{'='*100}\n")
        return {
            "success": ok,
            "returncode": int(process.returncode),
            "duration_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nError: {e}")
        return {
            "success": False,
            "returncode": -1,
            "duration_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
        }


def resolve_eval_steps(eval_data_path: str, eval_steps: int = None) -> int:
    if eval_steps is not None and eval_steps > 0:
        return int(eval_steps)
    try:
        import pandas as pd
        # Load only first column to get row count (avoids loading full file)
        df = pd.read_csv(eval_data_path, usecols=[0])
        return max(1, len(df) - 1)  # steps = rows - 1 (matches evaluation.py convention)
    except Exception:
        return 1000


def find_latest_file(directory: str, pattern: str) -> str:
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_eval_metrics(json_path: str, canonical_variant: str = None) -> dict:
    out = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return out

        source = data
        if canonical_variant and isinstance(data.get("tiers"), dict):
            tier_entry = data.get("tiers", {}).get(canonical_variant, {})
            if isinstance(tier_entry, dict):
                source = tier_entry

        if isinstance(source, dict):
            for k, v in source.items():
                if isinstance(v, (int, float, str, bool)):
                    out[k] = v
            sleeve = source.get("sleeve_metrics", {})
            if isinstance(sleeve, dict):
                for k, v in sleeve.items():
                    if isinstance(v, (int, float, str, bool)):
                        out[k] = v

        for k in ("tier_report_scope", "tier_comparison_csv", "tier_comparison_md"):
            v = data.get(k)
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
        return out
    except Exception:
        return out


def write_seed_summary_csv(rows: list, csv_path: str):
    if not rows:
        return
    # Stable union of keys across rows (preserve first-seen order)
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_run_plan_csv(rows: list, csv_path: str):
    if not rows:
        return
    fieldnames = [
        "global_run_number",
        "total_planned_runs",
        "seed",
        "seed_run_order",
        "run_name",
        "canonical_variant",
        "save_dir",
        "eval_output_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_existing_summary_csv(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    rows = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except Exception:
        return []
    return rows


def upsert_summary_row(rows: list, row: dict) -> list:
    key = str(row.get("global_run_number", ""))
    if not key:
        rows.append(row)
        return rows
    for i, existing in enumerate(rows):
        if str(existing.get("global_run_number", "")) == key:
            merged = dict(existing)
            merged.update(row)
            rows[i] = merged
            return rows
    rows.append(row)
    return rows


def as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


def _parse_seeds(seed_tokens):
    seeds = []
    for token in seed_tokens:
        for part in str(token).replace(",", " ").split():
            try:
                seeds.append(int(part))
            except ValueError:
                pass
    seen = set()
    return [s for s in seeds if not (s in seen or seen.add(s))]


def _build_suite_dir(output_root: str, phase: str, run_tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"tiers_{phase}_{stamp}"
    if run_tag:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_tag).strip())
        if safe:
            base = f"{base}_{safe}"
    path = os.path.join(output_root, base)
    os.makedirs(path, exist_ok=True)
    return path


def write_phase_protocol_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_existing_protocol(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed tier phases with evaluation.")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seeds, e.g. --seeds 7 42 123 789 2025")
    parser.add_argument("--phase", type=str, default="tier1_tier2_full_ablated",
                        choices=list(PHASE_VARIANTS.keys()),
                        help="Phase: tier1_tier2_full_ablated (3 runs/seed) or tier1_tier2, tier1_enhancer (legacy alias only), tier2_only")
    parser.add_argument("--episode_data_dir", type=str, default="training_dataset")
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--end_episode", type=int, default=19)
    parser.add_argument("--global_norm_mode", type=str, default="global", choices=["rolling_past", "global"],
                        help="Normalization mode forwarded to main.py.")
    parser.add_argument("--rolling_past_history_dir", type=str, default="",
                        help="Optional rolling-past history dir forwarded to main.py when --global_norm_mode rolling_past.")
    parser.add_argument("--investment_freq", type=int, default=6)
    parser.add_argument("--meta_freq_min", type=int, default=6,
                        help="Default live investor cadence minimum forwarded to train/eval. Default pins short horizon.")
    parser.add_argument("--meta_freq_max", type=int, default=6,
                        help="Default live investor cadence maximum forwarded to train/eval. Default pins short horizon.")
    parser.add_argument("--cooling_period", type=int, default=0)
    parser.add_argument("--forecast_training_dataset_dir", type=str, default="forecast_training_dataset")
    parser.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    parser.add_argument("--forecast_cache_dir", type=str, default="forecast_cache")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Pinned PPO learning rate for canonical seed-suite runs.")
    parser.add_argument("--ent_coef", type=float, default=0.03,
                        help="Pinned PPO entropy coefficient for canonical seed-suite runs.")
    parser.add_argument("--tier2_value_lr", type=float, default=1e-3,
                        help="Explicit Tier-2 optimizer learning rate forwarded to main.py for Tier-2 runs.")
    parser.add_argument("--dl_train_every", type=int, default=128,
                        help="Explicit Tier-2 online training frequency forwarded to main.py for Tier-2 runs.")
    parser.add_argument("--ppo_log_std_init", type=float, default=None,
                        help="Pinned PPO log_std_init for canonical seed-suite runs.")
    parser.add_argument("--enable_ppo_use_sde", action="store_true",
                        help="Enable SB3 PPO gSDE. Default suite path stays vanilla without gSDE.")
    parser.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--tiers_eval_mode", type=str, default="deployed", choices=["paper", "deployed"],
                        help="Evaluation mode forwarded to evaluation.py. Default is deployed so Tier-2 runs are checked with Tier-2 weights attached.")
    parser.add_argument("--tiers_precompute_forecasts", action="store_true",
                        help="When tiers_eval_mode=deployed, precompute/load the evaluation forecast cache for speed and determinism.")
    parser.add_argument("--tiers_precompute_batch_size", type=int, default=8192,
                        help="Batch size forwarded to evaluation.py when --tiers_precompute_forecasts is enabled.")
    parser.add_argument("--output_root", type=str, default="batch_tier_phase_runs")
    parser.add_argument("--suite_dir", type=str, default="",
                        help="Optional existing suite directory to resume into. If empty, a new suite dir is created.")
    parser.add_argument("--start_run_number", type=int, default=1,
                        help="Global run number to start from (1-based) for resume after OOM.")
    parser.add_argument("--end_run_number", type=int, default=None,
                        help="Optional global run number to stop at (inclusive).")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--continue_on_error", action="store_true")

    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        print("Error: No valid seeds provided.")
        sys.exit(1)

    phase_variants = PHASE_VARIANTS[args.phase]
    py = sys.executable
    eval_steps = resolve_eval_steps(args.eval_data, args.eval_steps)
    if str(args.suite_dir).strip():
        suite_dir = str(args.suite_dir).strip()
        os.makedirs(suite_dir, exist_ok=True)
    else:
        suite_dir = _build_suite_dir(args.output_root, args.phase, args.run_tag)
    summary_csv = os.path.join(suite_dir, f"{args.phase}_seed_suite_summary.csv")
    run_plan_csv = os.path.join(suite_dir, f"{args.phase}_run_plan.csv")

    common_args = [
        "--episode_training",
        "--episode_data_dir", args.episode_data_dir,
        "--start_episode", str(args.start_episode),
        "--end_episode", str(args.end_episode),
        "--global_norm_mode", str(args.global_norm_mode),
        "--investment_freq", str(args.investment_freq),
        "--meta_freq_min", str(args.meta_freq_min),
        "--meta_freq_max", str(args.meta_freq_max),
        "--cooling_period", str(args.cooling_period),
        "--lr", str(args.lr),
        "--ent_coef", str(args.ent_coef),
    ]
    if args.ppo_log_std_init is not None:
        common_args.extend(["--ppo_log_std_init", str(args.ppo_log_std_init)])
    if str(getattr(args, "rolling_past_history_dir", "")).strip():
        common_args.extend([
            "--rolling_past_history_dir", str(args.rolling_past_history_dir).strip(),
        ])
    if bool(args.enable_ppo_use_sde):
        common_args.append("--ppo_use_sde")
    baseline_policy_args = []
    forecast_args = [
        "--forecast_training_dataset_dir", args.forecast_training_dataset_dir,
        "--forecast_base_dir", args.forecast_base_dir,
        "--forecast_cache_dir", args.forecast_cache_dir,
    ]
    tier2_policy_args = [
        "--tier2_value_lr", str(args.tier2_value_lr),
        "--dl_train_every", str(args.dl_train_every),
    ]
    protocol_path = os.path.join(suite_dir, "phase_protocol.json")
    if str(args.suite_dir).strip():
        existing_protocol = _load_existing_protocol(protocol_path)
        if existing_protocol:
            compatibility_checks = {
                "phase": args.phase,
                "global_norm_mode": str(args.global_norm_mode),
                "investment_freq": int(args.investment_freq),
                "meta_freq_min": int(args.meta_freq_min),
                "meta_freq_max": int(args.meta_freq_max),
                "ppo_use_sde": bool(args.enable_ppo_use_sde),
                "ppo_log_std_init": None if args.ppo_log_std_init is None else float(args.ppo_log_std_init),
                "tier2_value_lr": float(args.tier2_value_lr),
                "dl_train_every": int(args.dl_train_every),
            }
            for key, expected in compatibility_checks.items():
                actual = existing_protocol.get(key)
                if actual != expected:
                    print(
                        f"Error: suite_dir protocol mismatch for '{key}': "
                        f"existing={actual!r}, requested={expected!r}"
                    )
                    sys.exit(1)
    write_phase_protocol_json(
        protocol_path,
        {
            "phase": args.phase,
            "global_norm_mode": str(args.global_norm_mode),
            "rolling_past_history_dir": str(getattr(args, "rolling_past_history_dir", "") or ""),
            "investment_freq": int(args.investment_freq),
            "meta_freq_min": int(args.meta_freq_min),
            "meta_freq_max": int(args.meta_freq_max),
            "ppo_use_sde": bool(args.enable_ppo_use_sde),
            "ppo_log_std_init": None if args.ppo_log_std_init is None else float(args.ppo_log_std_init),
            "tier2_value_lr": float(args.tier2_value_lr),
            "dl_train_every": int(args.dl_train_every),
            "tiers_eval_mode": str(args.tiers_eval_mode),
            "tiers_precompute_forecasts": bool(args.tiers_precompute_forecasts),
            "tiers_precompute_batch_size": int(args.tiers_precompute_batch_size),
            "shared_training_args": common_args,
            "tier1_policy_args": baseline_policy_args,
            "forecast_args_for_tier2": forecast_args,
            "tier2_policy_args": tier2_policy_args,
            "documentation": (
                "Tier 1 uses the hybrid RL baseline. Tier 2 is an independent forecast-guided enhanced MARL "
                "baseline that trains its own RL policies together with its own ANN-primary short-horizon "
                "forecast-conditioned Tier-2 layer. Tier 2 ablated keeps the same training contract but removes "
                "the full Tier-2 forecast-memory block."
            ),
            "tier2_training_contract": "independent_forecast_guided_enhanced_marl_baseline",
        },
    )
    # Build full run plan with deterministic global numbering.
    planned_runs = []
    global_run = 0
    for seed in seeds:
        seed_dir = os.path.join(suite_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        for idx, canonical in enumerate(phase_variants, 1):
            spec = VARIANT_TRAIN_SPECS[canonical]
            save_dir = os.path.join(seed_dir, f"{spec['slug']}_seed{seed}")
            eval_out = os.path.join(seed_dir, "evaluations_2025", canonical)
            global_run += 1
            planned_runs.append({
                "global_run_number": global_run,
                "total_planned_runs": 0,  # filled below
                "seed": seed,
                "seed_run_order": idx,
                "run_name": spec["name"],
                "canonical_variant": canonical,
                "save_dir": save_dir,
                "eval_output_dir": eval_out,
            })
    total_runs = len(planned_runs)
    for r in planned_runs:
        r["total_planned_runs"] = total_runs

    if args.start_run_number < 1:
        print("Error: --start_run_number must be >= 1")
        sys.exit(1)
    if args.start_run_number > total_runs:
        print(f"Error: --start_run_number {args.start_run_number} exceeds total planned runs {total_runs}")
        sys.exit(1)
    end_run = total_runs if args.end_run_number is None else int(args.end_run_number)
    if end_run < args.start_run_number:
        print("Error: --end_run_number must be >= --start_run_number")
        sys.exit(1)
    end_run = min(end_run, total_runs)

    write_run_plan_csv(planned_runs, run_plan_csv)

    print(f"\nPhase: {args.phase}")
    print(f"Seeds: {seeds}")
    print(f"Suite dir: {suite_dir}")
    print(f"Protocol doc: {protocol_path}")
    print(f"Run plan CSV: {run_plan_csv}")
    print(f"Summary CSV: {summary_csv}\n")
    print(f"Run range: {args.start_run_number}..{end_run} / {total_runs}\n")

    summary_rows = read_existing_summary_csv(summary_csv)
    stop = False
    for plan in planned_runs:
        if stop:
            break
        run_no = int(plan["global_run_number"])
        if run_no < args.start_run_number or run_no > end_run:
            continue

        seed = int(plan["seed"])
        idx = int(plan["seed_run_order"])
        canonical = str(plan["canonical_variant"])
        spec = VARIANT_TRAIN_SPECS[canonical]
        save_dir = str(plan["save_dir"])
        eval_out = str(plan["eval_output_dir"])
        current_seed_dir = os.path.join(suite_dir, f"seed{seed}")
        current_tier1_dir = os.path.join(
            current_seed_dir,
            f"{VARIANT_TRAIN_SPECS['tier1']['slug']}_seed{seed}",
        )

        extra = list(spec["extra"])
        train_extra_args = []
        if canonical != "tier1":
            train_extra_args = list(baseline_policy_args) + forecast_args + tier2_policy_args
        else:
            train_extra_args = list(baseline_policy_args)
        label = f"[Run {run_no}/{total_runs}] {spec['name']} (seed={seed}, order={idx})"
        train_cmd = [py, "main.py"] + common_args + ["--seed", str(seed), "--save_dir", save_dir] + train_extra_args + extra
        train_result = run_command(train_cmd, label)

        row = {
            "phase": args.phase,
            "global_run_number": run_no,
            "total_planned_runs": total_runs,
            "seed": seed,
            "run_order": idx,
            "run_name": spec["name"],
            "canonical_variant": canonical,
            "save_dir": save_dir,
            "eval_output_dir": eval_out,
            "training_success": train_result.get("success", False),
            "training_returncode": train_result.get("returncode", -1),
            "training_duration_seconds": train_result.get("duration_seconds", 0),
            "evaluation_success": False,
        }

        if train_result.get("success"):
            eval_dir_arg = EVAL_DIR_ARG_BY_VARIANT[canonical]
            os.makedirs(eval_out, exist_ok=True)
            tiers_only_val = EVAL_TIERS_ONLY_BY_VARIANT[canonical]
            eval_cmd = [
                py, "evaluation.py", "--mode", "tiers", "--tiers_only", tiers_only_val,
                eval_dir_arg, save_dir,
                "--tier1_dir", current_tier1_dir,
                "--eval_data", args.eval_data,
                "--eval_steps", str(eval_steps),
                "--output_dir", eval_out,
                "--investment_freq", str(args.investment_freq),
                "--meta_freq_min", str(args.meta_freq_min),
                "--meta_freq_max", str(args.meta_freq_max),
                "--tiers_eval_mode", str(args.tiers_eval_mode),
                "--forecast_base_dir", args.forecast_base_dir,
                "--forecast_cache_dir", args.forecast_cache_dir,
                "--tiers_precompute_batch_size", str(args.tiers_precompute_batch_size),
            ]
            if bool(args.tiers_precompute_forecasts):
                eval_cmd.append("--tiers_precompute_forecasts")
            eval_label = f"[Run {run_no}/{total_runs}] Eval {canonical} (seed={seed})"
            eval_result = run_command(eval_cmd, eval_label)
            row["evaluation_success"] = eval_result.get("success", False)
            if row["evaluation_success"]:
                eval_json_path = find_latest_file(eval_out, "evaluation_tiers_*.json")
                if eval_json_path:
                    row["evaluation_results_json"] = eval_json_path
                    row.update(extract_eval_metrics(eval_json_path, canonical))

        summary_rows = upsert_summary_row(summary_rows, row)
        write_seed_summary_csv(summary_rows, summary_csv)

        if (not train_result.get("success") or not row.get("evaluation_success", False)) and not args.continue_on_error:
            stop = True
            break

    print(f"\nDone. Summary: {summary_csv}")
    overall_ok = all(as_bool(r.get("training_success")) and as_bool(r.get("evaluation_success")) for r in summary_rows)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
