#!/usr/bin/env python3
"""
Run Tier-1 baseline training and evaluation across multiple seeds.

For each seed: train via main.py, then evaluate via evaluation.py --mode tiers.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from runtime_contract import (
    build_runtime_contract,
    forecast_prior_contract_settings,
    runtime_contract_hash,
)
from forecast_prior_cli import (
    add_forecast_prior_override_args,
    collect_forecast_prior_overrides,
    forecast_prior_override_cli_args,
)


DEFAULT_ROLLING_PAST_HISTORY_DIR = "rolling_past_history_dataset"


def _effective_rolling_past_history_dir(args) -> str:
    explicit = str(getattr(args, "rolling_past_history_dir", "") or "").strip()
    if str(getattr(args, "global_norm_mode", "")).strip().lower() == "rolling_past":
        return explicit or DEFAULT_ROLLING_PAST_HISTORY_DIR
    return explicit


VARIANT_TRAIN_SPECS = {
    "tier1": {
        "name": "Tier 1 hybrid RL baseline",
        "slug": "tier1",
        "extra": [],
        "eval_extra": [],
        "reuse_train_from": None,
    },
    "tier1_forecast_utilization": {
        "name": "Tier 1 + conformal ANN forecast-cache utilization",
        "slug": "tier1_forecast_utilization",
        "extra": ["--enable_forecast_utilization"],
        "eval_extra": ["--enable_forecast_utilization"],
        "reuse_train_from": None,
    },
}

PHASE_VARIANTS = {
    "tier1_only": ["tier1"],
    "tier1_forecast_utilization_only": ["tier1_forecast_utilization"],
    "tier1_forecast_utilization_pair": ["tier1", "tier1_forecast_utilization"],
}

EVAL_DIR_ARG_BY_VARIANT = {
    "tier1": "--tier1_dir",
    "tier1_forecast_utilization": "--tier1_dir",
}

EVAL_TIERS_ONLY_BY_VARIANT = {
    "tier1": "tier1",
    "tier1_forecast_utilization": "tier1",
}

# Variants that require their own evaluation forecast-cache directory.
# Empty string ⇒ use the value of args.forecast_cache_dir (default).
EVAL_FORECAST_CACHE_DIR_OVERRIDE = {
    "tier1": "",
    "tier1_forecast_utilization": "",
}


def format_cmd(cmd):
    return subprocess.list2cmdline(cmd)


def _run_timeout_seconds() -> int:
    try:
        hours = float(os.environ.get("TIER_RUN_TIMEOUT_HOURS", "8"))
        return max(60, int(hours * 3600))
    except Exception:
        return 8 * 3600


def run_command(cmd, name: str) -> dict:
    started_at = datetime.now()
    start = time.time()
    print(f"\n{'='*100}")
    print(f"Starting: {name}")
    print(f"Command: {format_cmd(cmd)}")
    print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    try:
        timeout_s = _run_timeout_seconds()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        timed_out = False
        deadline = time.time() + timeout_s
        for line in process.stdout:
            print(line, end="")
            if time.time() > deadline:
                process.kill()
                timed_out = True
                print(f"\n[TIMEOUT] {name} exceeded {timeout_s}s — process killed.")
                break
        process.wait()
        elapsed = time.time() - start
        ok = process.returncode == 0 and not timed_out
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
        df = pd.read_csv(eval_data_path, usecols=[0])
        return max(1, len(df) - 1)
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
            if not tier_entry and str(canonical_variant).startswith("tier1"):
                tier_entry = data.get("tiers", {}).get("tier1", {})
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

        for k in ("tier_report_scope", "tier_report_csv", "tier_report_md"):
            v = data.get(k)
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
        return out
    except Exception:
        return out


def write_seed_summary_csv(rows: list, csv_path: str):
    if not rows:
        return
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
    parser = argparse.ArgumentParser(description="Run multi-seed Tier-1 training and evaluation.")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seeds, e.g. --seeds 7 42 123 789 2025")
    parser.add_argument("--phase", type=str, default="tier1_only",
                        choices=list(PHASE_VARIANTS.keys()),
                        help="Phase to run.")
    parser.add_argument("--episode_data_dir", type=str, default="training_dataset")
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--end_episode", type=int, default=19)
    parser.add_argument("--global_norm_mode", type=str, default="rolling_past", choices=["rolling_past", "global"])
    parser.add_argument("--rolling_past_history_dir", type=str, default="")
    parser.add_argument("--investment_freq", type=int, default=6)
    parser.add_argument("--meta_freq_min", type=int, default=6)
    parser.add_argument("--meta_freq_max", type=int, default=6)
    parser.add_argument("--cooling_period", type=int, default=0)
    parser.add_argument("--forecast_cache_dir", type=str, default="forecast_cache")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.03)
    parser.add_argument("--ppo_log_std_init", type=float, default=None)
    parser.add_argument("--enable_ppo_use_sde", action="store_true")
    parser.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--output_root", type=str, default="batch_tier_phase_runs")
    parser.add_argument("--suite_dir", type=str, default="",
                        help="Existing suite directory to resume into. If empty, a new one is created.")
    parser.add_argument("--start_run_number", type=int, default=1,
                        help="Global run number to start from (1-based) for resume after OOM.")
    parser.add_argument("--end_run_number", type=int, default=None,
                        help="Optional global run number to stop at (inclusive).")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--continue_on_error", action="store_true")
    add_forecast_prior_override_args(parser)

    args = parser.parse_args()
    if str(args.global_norm_mode).strip().lower() != "rolling_past":
        print("Error: this suite is pinned to rolling_past for consistency. Use --global_norm_mode rolling_past.")
        sys.exit(1)
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
    eff_roll = _effective_rolling_past_history_dir(args)
    if eff_roll:
        common_args.extend(["--rolling_past_history_dir", eff_roll])
    if bool(args.enable_ppo_use_sde):
        common_args.append("--ppo_use_sde")

    forecast_args = [
        "--forecast_cache_dir", args.forecast_cache_dir,
    ]
    forecast_prior_overrides = collect_forecast_prior_overrides(args)
    forecast_prior_args = forecast_prior_override_cli_args(args)

    base_contract = build_runtime_contract(
        global_norm_mode=str(args.global_norm_mode),
        rolling_past_history_dir=eff_roll,
        investment_freq=int(args.investment_freq),
        meta_freq_min=int(args.meta_freq_min),
        meta_freq_max=int(args.meta_freq_max),
        enable_forecast_utilization=False,
    )
    base_contract_hash = runtime_contract_hash(base_contract)
    forecast_contract = build_runtime_contract(
        global_norm_mode=str(args.global_norm_mode),
        rolling_past_history_dir=eff_roll,
        investment_freq=int(args.investment_freq),
        meta_freq_min=int(args.meta_freq_min),
        meta_freq_max=int(args.meta_freq_max),
        enable_forecast_utilization=True,
        forecast_prior_settings=forecast_prior_contract_settings(forecast_prior_overrides),
    )
    runtime_contracts_by_variant = {
        "tier1": base_contract,
        "tier1_forecast_utilization": forecast_contract,
    }
    runtime_contract_hashes_by_variant = {
        key: runtime_contract_hash(value)
        for key, value in runtime_contracts_by_variant.items()
    }
    active_runtime_contract_hashes = {
        key: runtime_contract_hashes_by_variant[key]
        for key in phase_variants
    }

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
                "runtime_contract_hash": base_contract_hash,
                "variant_runtime_contract_hashes": active_runtime_contract_hashes,
            }
            for key, expected in compatibility_checks.items():
                actual = existing_protocol.get(key)
                if actual != expected:
                    print(
                        f"Error: suite_dir protocol mismatch for '{key}': "
                        f"existing={actual!r}, requested={expected!r}"
                    )
                    sys.exit(1)
            if str(args.global_norm_mode).strip().lower() == "rolling_past":
                exp_roll = _effective_rolling_past_history_dir(args)
                act_roll = str(existing_protocol.get("rolling_past_history_dir", "") or "").strip()
                norm_roll = lambda x: x or DEFAULT_ROLLING_PAST_HISTORY_DIR
                if norm_roll(act_roll) != norm_roll(exp_roll):
                    print(
                        "Error: suite_dir protocol mismatch for 'rolling_past_history_dir' "
                        f"(normalized): existing={act_roll!r}, requested={exp_roll!r}"
                    )
                    sys.exit(1)

    write_phase_protocol_json(
        protocol_path,
        {
            "phase": args.phase,
            "phase_variants": list(phase_variants),
            "global_norm_mode": str(args.global_norm_mode),
            "rolling_past_history_dir": _effective_rolling_past_history_dir(args),
            "runtime_contract": base_contract,
            "runtime_contract_hash": base_contract_hash,
            "variant_runtime_contracts": {
                key: runtime_contracts_by_variant[key]
                for key in phase_variants
            },
            "variant_runtime_contract_hashes": active_runtime_contract_hashes,
            "investment_freq": int(args.investment_freq),
            "meta_freq_min": int(args.meta_freq_min),
            "meta_freq_max": int(args.meta_freq_max),
            "ppo_use_sde": bool(args.enable_ppo_use_sde),
            "ppo_log_std_init": None if args.ppo_log_std_init is None else float(args.ppo_log_std_init),
            "shared_training_args": common_args,
            "forecast_args": forecast_args,
            "forecast_prior_overrides": forecast_prior_overrides,
        },
    )

    # Build full run plan with deterministic global numbering.
    planned_runs = []
    global_run = 0
    for seed in seeds:
        seed_dir = os.path.join(suite_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        # Pre-resolve save_dirs by canonical variant within this seed so that
        # variants with reuse_train_from=<other> point at <other>'s save_dir.
        save_dir_by_canonical = {}
        for canonical in phase_variants:
            spec = VARIANT_TRAIN_SPECS[canonical]
            reuse_from = spec.get("reuse_train_from")
            if reuse_from:
                if reuse_from not in save_dir_by_canonical:
                    raise ValueError(
                        f"Variant '{canonical}' has reuse_train_from='{reuse_from}' "
                        f"but '{reuse_from}' has not been planned earlier in phase "
                        f"'{args.phase}'. Reorder PHASE_VARIANTS so the source "
                        f"variant comes first."
                    )
                save_dir_by_canonical[canonical] = save_dir_by_canonical[reuse_from]
            else:
                save_dir_by_canonical[canonical] = os.path.join(
                    seed_dir, f"{spec['slug']}_seed{seed}"
                )
        for idx, canonical in enumerate(phase_variants, 1):
            spec = VARIANT_TRAIN_SPECS[canonical]
            save_dir = save_dir_by_canonical[canonical]
            eval_out = os.path.join(seed_dir, "evaluations_2025", canonical)
            global_run += 1
            planned_runs.append({
                "global_run_number": global_run,
                "total_planned_runs": 0,
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
    print(f"Protocol: {protocol_path}")
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

        label = f"[Run {run_no}/{total_runs}] {spec['name']} (seed={seed}, order={idx})"
        variant_extra = list(spec.get("extra", []) or [])
        reuse_from = spec.get("reuse_train_from")
        if reuse_from:
            print(f"\n[REUSE_TRAIN] {label}: reusing trained checkpoint from variant "
                  f"'{reuse_from}' at {save_dir}")
            if not os.path.isdir(os.path.join(save_dir, "final_models")):
                train_result = {
                    "success": False,
                    "returncode": -2,
                    "duration_seconds": 0,
                    "started_at": datetime.now().isoformat(),
                    "finished_at": datetime.now().isoformat(),
                }
                print(f"[REUSE_TRAIN][ERROR] expected final_models dir not found at "
                      f"{save_dir}; the source variant must have completed training.")
            else:
                train_result = {
                    "success": True,
                    "returncode": 0,
                    "duration_seconds": 0,
                    "started_at": datetime.now().isoformat(),
                    "finished_at": datetime.now().isoformat(),
                    "reused_from": reuse_from,
                }
        else:
            train_cmd = (
                [py, "main.py"]
                + common_args
                + ["--seed", str(seed), "--save_dir", save_dir]
                + forecast_args
                + variant_extra
            )
            if canonical == "tier1_forecast_utilization":
                train_cmd += forecast_prior_args
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
            # Variant-specific eval forecast-cache override (FoCAL needs the
            # episode-20 unseen-data cache; baseline ignores the cache entirely).
            eval_forecast_cache_dir = (
                EVAL_FORECAST_CACHE_DIR_OVERRIDE.get(canonical, "")
                or args.forecast_cache_dir
            )
            eval_cmd = [
                py, "evaluation.py", "--mode", "tiers",
                "--tiers_only", tiers_only_val,
                eval_dir_arg, save_dir,
                "--eval_data", args.eval_data,
                "--eval_steps", str(eval_steps),
                "--output_dir", eval_out,
                "--investment_freq", str(args.investment_freq),
                "--meta_freq_min", str(args.meta_freq_min),
                "--meta_freq_max", str(args.meta_freq_max),
                "--global_norm_mode", str(args.global_norm_mode),
                "--forecast_cache_dir", eval_forecast_cache_dir,
            ]
            # Variant-specific eval extras (e.g., extra eval-only flags).
            eval_cmd.extend(list(spec.get("eval_extra", []) or []))
            if canonical == "tier1_forecast_utilization":
                eval_cmd.extend(forecast_prior_args)
            if eff_roll:
                eval_cmd.extend(["--rolling_past_history_dir", eff_roll])
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
