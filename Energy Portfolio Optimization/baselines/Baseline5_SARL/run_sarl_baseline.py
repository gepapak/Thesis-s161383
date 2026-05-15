"""Current-codebase SARL baseline evaluator.

This runner evaluates a trained single-agent RL policy against the current root
Tier1 environment. It intentionally does not evaluate an untrained SARL policy
unless --allow_untrained is supplied, because ranking a random policy would make
the publication comparison misleading.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


BASELINE_DIR = Path(__file__).resolve().parent
BASELINES_ROOT = BASELINE_DIR.parent
PROJECT_ROOT = BASELINES_ROOT.parent
for _path in (PROJECT_ROOT, BASELINES_ROOT, BASELINE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from baseline_common import (  # noqa: E402
    add_common_summary_fields,
    compute_performance_metrics,
    load_baseline_economic_config,
)
from config import EnhancedConfig  # noqa: E402
from environment import RenewableMultiAgentEnv  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _flatten_obs_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        parts = [_flatten_obs_value(value[k]) for k in sorted(value)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def flatten_observations(obs: Dict[str, Any], agents: Iterable[str]) -> np.ndarray:
    parts = [_flatten_obs_value(obs[a]) for a in agents if a in obs]
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def sarl_action_to_env_actions(action: np.ndarray, env: RenewableMultiAgentEnv) -> Dict[str, Any]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < 4:
        action = np.pad(action, (0, 4 - action.size), constant_values=0.0)
    action = np.clip(action[:4], -1.0, 1.0)

    battery_space = env.action_space("battery_operator_0")
    battery_n = int(getattr(battery_space, "n", 5))
    battery_idx = int(round(((float(action[1]) + 1.0) / 2.0) * max(battery_n - 1, 0)))
    battery_idx = int(np.clip(battery_idx, 0, max(battery_n - 1, 0)))

    return {
        "investor_0": np.array([float(action[0])], dtype=np.float32),
        "battery_operator_0": battery_idx,
        "risk_controller_0": np.array([float(action[2])], dtype=np.float32),
        "meta_controller_0": np.array([float(action[3])], dtype=np.float32),
    }


def _get_total_distributions_dkk(env: RenewableMultiAgentEnv) -> float:
    for attr in ("total_distributions", "distributed_profits"):
        if hasattr(env, attr):
            return float(max(0.0, getattr(env, attr, 0.0)))
    return 0.0


def _write_skipped_summary(output_dir: Path, reason: str) -> Dict[str, Any]:
    econ_cfg = load_baseline_economic_config()
    summary: Dict[str, Any] = {
        "status": "skipped",
        "skip_reason": reason,
        "method": "SARL-DQN Current Environment",
        "strategy_type": "Single-Agent RL Baseline",
        "valid_for_publication_comparison": False,
        "current_codebase_environment": True,
        "initial_portfolio_value": econ_cfg.initial_budget_usd,
        "final_portfolio_value": 0.0,
        "initial_value_usd": econ_cfg.initial_budget_usd,
        "final_value_usd": 0.0,
        "total_return": 0.0,
        "models_loaded": 0,
    }
    add_common_summary_fields(summary, method="SARL-DQN Current Environment", status="skipped")
    _write_json(output_dir / "summary_metrics.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def evaluate_sarl(
    *,
    data_path: Path,
    output_dir: Path,
    timesteps: int,
    seed: int,
    model_path: Path | None,
    allow_untrained: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_path is None and not allow_untrained:
        return _write_skipped_summary(
            output_dir,
            "No current SARL model supplied. Pass --model_path here, or --sarl_model_path via run_current_baselines.py, to include SARL in the ranked comparison.",
        )
    if model_path is not None and not model_path.is_file():
        return _write_skipped_summary(output_dir, f"SARL model not found: {model_path}")

    import torch
    from sarl_dqn_optimizer import SARLConfig, SARLDQNOptimizer

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    data = pd.read_csv(data_path)
    max_steps = min(int(timesteps), len(data))

    cfg = EnhancedConfig()
    cfg.seed = int(seed)
    cfg.enable_forecast_utilization = False
    cfg.enable_forecast_utilisation = False
    cfg.training_global_step = 0

    env = RenewableMultiAgentEnv(
        data.iloc[:max_steps].reset_index(drop=True),
        investment_freq=int(getattr(cfg, "investment_freq", 6)),
        enhanced_risk_controller=True,
        config=cfg,
        log_dir=str(output_dir / "debug_logs"),
    )
    obs, _ = env.reset(seed=seed)
    state = flatten_observations(obs, env.possible_agents)

    sarl_cfg = SARLConfig(state_dim=int(state.size), action_dim=4)
    agent = SARLDQNOptimizer(sarl_cfg, device="cpu", state_dim=int(state.size))
    model_loaded = False
    if model_path is not None:
        agent.load_model(str(model_path))
        model_loaded = True
    else:
        agent.epsilon = 0.0

    econ_cfg = load_baseline_economic_config()
    reported_nav_values = []
    adjusted_nav_values = []
    risk_levels = []
    total_rewards = 0.0
    rows = []
    start = time.time()

    for step in range(max_steps):
        action = agent.select_action(state, training=False)
        actions = sarl_action_to_env_actions(action, env)
        obs, rewards, dones, truncs, infos = env.step(actions)

        reward_sum = float(sum(float(v) for v in rewards.values()))
        total_rewards += reward_sum

        reported_nav_dkk = float(env._calculate_fund_nav())
        total_distributions_usd = _get_total_distributions_dkk(env) * econ_cfg.dkk_to_usd_rate
        reported_nav_usd = reported_nav_dkk * econ_cfg.dkk_to_usd_rate
        adjusted_nav_usd = reported_nav_usd + total_distributions_usd
        reported_nav_values.append(reported_nav_usd)
        adjusted_nav_values.append(adjusted_nav_usd)

        if hasattr(env, "get_risk_level"):
            try:
                risk_levels.append(float(env.get_risk_level()))
            except Exception:
                pass

        rows.append(
            {
                "timestep": step,
                "reported_nav_usd": reported_nav_usd,
                "distribution_adjusted_nav_usd": adjusted_nav_usd,
                "total_distributions_usd": total_distributions_usd,
                "reward_sum": reward_sum,
                "investor_action": float(actions["investor_0"][0]),
                "battery_action": int(actions["battery_operator_0"]),
                "risk_action": float(actions["risk_controller_0"][0]),
                "meta_action": float(actions["meta_controller_0"][0]),
            }
        )

        state = flatten_observations(obs, env.possible_agents)
        if any(dones.values()) or any(truncs.values()):
            break

    pd.DataFrame(rows).to_csv(output_dir / "detailed_results.csv", index=False)

    metrics = compute_performance_metrics(
        reported_nav_values,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=econ_cfg.annual_risk_free_rate,
        total_distributions=(
            adjusted_nav_values[-1] - reported_nav_values[-1]
            if adjusted_nav_values and reported_nav_values
            else 0.0
        ),
        distribution_adjusted_values=adjusted_nav_values,
        value_suffix="usd",
    )
    final_reported = float(reported_nav_values[-1]) if reported_nav_values else 0.0
    final_adjusted = float(adjusted_nav_values[-1]) if adjusted_nav_values else 0.0
    summary: Dict[str, Any] = {
        **metrics,
        "final_value_usd": final_reported,
        "initial_value_usd": econ_cfg.initial_budget_usd,
        "distribution_adjusted_final_value_usd": final_adjusted,
        "total_rewards": float(total_rewards),
        "average_risk": float(np.mean(risk_levels)) if risk_levels else 0.0,
        "max_risk": float(np.max(risk_levels)) if risk_levels else 0.0,
        "min_risk": float(np.min(risk_levels)) if risk_levels else 0.0,
        "evaluation_steps": int(len(rows)),
        "runtime_seconds": float(time.time() - start),
        "models_loaded": 1 if model_loaded else 0,
        "model_path": str(model_path) if model_path else "",
        "current_codebase_environment": True,
        "valid_for_publication_comparison": bool(model_loaded),
        "untrained_policy": not model_loaded,
        "strategy_type": "Single-Agent RL Baseline",
    }
    add_common_summary_fields(summary, method="SARL-DQN Current Environment")
    if not model_loaded:
        summary["status"] = "completed_untrained_not_ranked"
        summary["warning"] = "Untrained SARL was evaluated only because --allow_untrained was supplied."

    _write_json(output_dir / "summary_metrics.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SARL against the current Tier1 environment.")
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "evaluation_dataset" / "unseendata.csv")
    parser.add_argument("--output_dir", type=Path, default=BASELINE_DIR / "results")
    parser.add_argument("--timesteps", type=int, default=39305)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument(
        "--allow_untrained",
        action="store_true",
        help="Evaluate a randomly initialized SARL policy. This is not ranked for publication.",
    )
    args = parser.parse_args()

    summary = evaluate_sarl(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        timesteps=args.timesteps,
        seed=args.seed,
        model_path=args.model_path.resolve() if args.model_path else None,
        allow_untrained=args.allow_untrained,
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
