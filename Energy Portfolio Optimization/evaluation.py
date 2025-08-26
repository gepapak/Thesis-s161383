import argparse
import os
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
from main import load_energy_data  # <-- patched import (no utils)  # noqa: E402
from environment import RenewableMultiAgentEnv
from metacontroller import MultiESGAgent
from generator import MultiHorizonForecastGenerator
from wrapper import MultiHorizonWrapperEnv


class EvaluationConfig:
    """Lightweight config; policies are placeholders (actual params loaded from disk)."""
    def __init__(self):
        self.update_every = 128
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.verbose = 1
        self.seed = 42
        self.multithreading = True
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "PPO"},  # battery_operator_0
            {"mode": "PPO"},  # risk_controller_0
            {"mode": "SAC"},  # meta_controller_0
        ]


def _coerce_action_for_space(action: np.ndarray, action_space):
    """
    Make sure an action predicted by a policy matches the env's action_space.
    Handles both Box and Discrete spaces.
    """
    if hasattr(action_space, "n"):  # Discrete-like
        # action can be [[x]] or [x] or scalar; take a scalar int
        if isinstance(action, (list, tuple, np.ndarray)):
            action = np.array(action).astype(np.int64).flatten()
            return int(action[0])
        return int(action)
    else:
        # Box-like: return as 1D float32 vector with correct shape if possible
        act = np.array(action, dtype=np.float32).squeeze()
        if hasattr(action_space, "shape") and action_space.shape is not None:
            target = int(np.prod(action_space.shape))
            act = act.flatten()
            if act.size != target:
                # Pad/trim defensively
                if act.size < target:
                    act = np.pad(act, (0, target - act.size))
                else:
                    act = act[:target]
        return act


def _calculate_performance_metrics(portfolio_values, episode_rewards, risk_levels, steps):
    """Calculate comprehensive performance metrics for evaluation"""
    metrics = {}

    try:
        if portfolio_values and len(portfolio_values) > 1:
            # Portfolio performance
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0

            # Calculate returns series
            pv_series = np.array(portfolio_values)
            returns = np.diff(pv_series) / pv_series[:-1]
            returns = returns[np.isfinite(returns)]  # Remove inf/nan

            if len(returns) > 0:
                # Annualized metrics (assuming 10-minute intervals)
                steps_per_year = 365 * 24 * 6  # 52,560 steps per year
                annualization_factor = steps_per_year / len(returns) if len(returns) > 0 else 1

                annual_return = (1 + total_return) ** annualization_factor - 1
                volatility = np.std(returns) * np.sqrt(steps_per_year) if len(returns) > 1 else 0
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0

                # Drawdown calculation
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

                # Win rate
                win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

                metrics.update({
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'final_portfolio_value': final_value,
                    'initial_portfolio_value': initial_value
                })

        # Risk metrics
        if risk_levels:
            metrics.update({
                'avg_risk_level': np.mean(risk_levels),
                'max_risk_level': np.max(risk_levels),
                'risk_volatility': np.std(risk_levels)
            })

        # Reward metrics
        if episode_rewards:
            all_rewards = []
            for agent_rewards in episode_rewards.values():
                all_rewards.extend(agent_rewards)

            if all_rewards:
                metrics.update({
                    'avg_reward': np.mean(all_rewards),
                    'reward_volatility': np.std(all_rewards),
                    'total_reward': np.sum(all_rewards)
                })

        # Effectiveness score (0-100)
        effectiveness_score = 0
        if 'sharpe_ratio' in metrics:
            # Sharpe contribution (40 points max)
            sharpe_score = min(metrics['sharpe_ratio'] / 2.0, 1.0) * 40  # Sharpe 2.0 = full points
            effectiveness_score += sharpe_score

        if 'max_drawdown' in metrics:
            # Drawdown contribution (30 points max)
            drawdown_score = max(0, (0.25 - abs(metrics['max_drawdown'])) / 0.25) * 30  # <25% drawdown = full points
            effectiveness_score += drawdown_score

        if 'win_rate' in metrics:
            # Win rate contribution (30 points max)
            win_score = metrics['win_rate'] * 30  # 100% win rate = full points
            effectiveness_score += win_score

        metrics['effectiveness_score'] = min(effectiveness_score, 100)

        # Performance classification
        if effectiveness_score >= 80:
            metrics['performance_class'] = 'EXCELLENT'
        elif effectiveness_score >= 60:
            metrics['performance_class'] = 'GOOD'
        elif effectiveness_score >= 40:
            metrics['performance_class'] = 'MODERATE'
        else:
            metrics['performance_class'] = 'POOR'

    except Exception as e:
        print(f"⚠️ Error calculating performance metrics: {e}")
        metrics['error'] = str(e)

    return metrics


def evaluate_trained_agents(
    data: pd.DataFrame,
    trained_agents_dir: str,
    model_dir: str,
    scaler_dir: str,
    evaluation_steps: int | None = None,
    log_path: str | None = None
) -> Dict[str, Any] | None:
    """
    Evaluate pre-trained agents on new data.
    Returns a dictionary with basic aggregates (for a quick summary),
    or None on failure.
    """
    print("🔍 Evaluating trained agents on new data.")
    print(f"📊 Evaluation data shape: {data.shape}")

    # ---- Forecaster ----
    print("🔮 Loading forecaster.")
    try:
        forecaster = MultiHorizonForecastGenerator(
            model_dir=model_dir,
            scaler_dir=scaler_dir,
            look_back=6,
            verbose=False
        )
    except Exception as e:
        print(f"❌ Failed to load forecaster: {e}")
        return None

    # ---- Base env ----
    print("🏗️ Setting up evaluation environment.")
    base_env = RenewableMultiAgentEnv(
        data,
        investment_freq=144,        # 10-min cadence default; OK for eval
        forecast_generator=forecaster
    )

    # ---- Wrapper (adds logging, forecasted features, etc.) ----
    if log_path is None:
        os.makedirs("evaluation_logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"evaluation_logs/evaluation_metrics_{timestamp}.csv"

    eval_env = MultiHorizonWrapperEnv(
        base_env,
        forecaster,
        log_path=log_path
    )

    # ---- Agent system (loads from disk) ----
    print(f"🤖 Initializing agent system.")
    config = EvaluationConfig()
    agent_system = MultiESGAgent(
        config,
        env=eval_env,
        device="cpu",       # CPU is fine for evaluation
        training=False,     # eval mode
        debug=False
    )

    print(f"📂 Loading trained policies from {trained_agents_dir}.")
    loaded_count = agent_system.load_policies(trained_agents_dir)
    if loaded_count == 0:
        print(f"❌ No policies loaded! Check the path: {trained_agents_dir}")
        return None
    print(f"✅ Loaded {loaded_count} trained policies")

    # ---- Run evaluation loop ----
    print("🚀 Starting evaluation.")
    obs, _ = eval_env.reset()

    # Pick evaluation length
    if evaluation_steps is None:
        evaluation_steps = min(len(data) - 1, 10_000)
    evaluation_steps = int(max(1, evaluation_steps))
    print(f"📏 Evaluating for {evaluation_steps} steps.")

    # Aggregates (kept minimal & robust)
    episode_rewards: Dict[str, list[float]] = {agent: [] for agent in eval_env.possible_agents}
    portfolio_values: list[float] = []
    risk_levels: list[float] = []
    actions_taken: Dict[str, list[Any]] = {agent: [] for agent in eval_env.possible_agents}

    for step in range(evaluation_steps):
        if step % 1000 == 0:
            print(f"📊 Progress: {step}/{evaluation_steps} ({step / evaluation_steps * 100:.1f}%)")

        actions: Dict[str, Any] = {}

        # Deterministic policy actions
        for i, agent in enumerate(eval_env.possible_agents):
            if agent not in obs:
                continue
            agent_obs = np.array(obs[agent], dtype=np.float32).reshape(1, -1)

            policy = agent_system.policies[i]
            if hasattr(policy, "predict"):
                act, _ = policy.predict(agent_obs, deterministic=True)
            else:
                # Fallback sampling if a policy object has no predict()
                act = eval_env.action_space(agent).sample()

            act = _coerce_action_for_space(act, eval_env.action_space(agent))
            actions[agent] = act
            # Keep a copy for analysis
            try:
                actions_taken[agent].append(np.array(act).copy())
            except Exception:
                actions_taken[agent].append(act)

        # Environment step
        obs, rewards, dones, truncs, infos = eval_env.step(actions)

        # Collect simple metrics
        for agent, r in rewards.items():
            try:
                episode_rewards[agent].append(float(r))
            except Exception:
                pass

        # Portfolio proxy (budget + scaled capacity) if available
        pv = None
        try:
            # Prefer wrapper -> base env fields if present
            env_ref = getattr(eval_env, "env", None)
            budget = getattr(eval_env, "budget", None)
            if env_ref is not None:
                budget = getattr(env_ref, "budget", budget)
            wc = getattr(env_ref, "wind_capacity", 0.0)
            sc = getattr(env_ref, "solar_capacity", 0.0)
            hc = getattr(env_ref, "hydro_capacity", 0.0)
            if budget is not None:
                pv = float(budget) + float(wc + sc + hc) * 100.0
        except Exception:
            pv = None
        if pv is not None:
            portfolio_values.append(pv)

        # Risk metric if exposed
        try:
            risk_val = getattr(eval_env, "market_stress", None)
            if risk_val is not None:
                risk_levels.append(float(risk_val))
        except Exception:
            pass

        # Handle termination
        try:
            if any(bool(x) for x in dones.values()):
                obs, _ = eval_env.reset()
        except Exception:
            # If dones is absent or malformed, just continue
            pass

    print("✅ Evaluation loop finished.")

    # Calculate performance metrics
    performance_metrics = _calculate_performance_metrics(
        portfolio_values, episode_rewards, risk_levels, evaluation_steps
    )

    return {
        "steps_evaluated": evaluation_steps,
        "episode_rewards": episode_rewards,
        "portfolio_values": portfolio_values,
        "risk_levels": risk_levels,
        "actions_taken": actions_taken,
        "log_path": log_path,
        "performance_metrics": performance_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents on new data.")
    parser.add_argument("--eval_data", required=True, help="CSV with evaluation data")
    parser.add_argument("--trained_agents", required=True, help="Directory with saved agent policies")
    parser.add_argument("--model_dir", default="models", help="Forecast model directory")
    parser.add_argument("--scaler_dir", default="scalers", help="Forecast scaler directory")
    parser.add_argument("--eval_steps", type=int, default=None, help="Number of timesteps to evaluate")
    parser.add_argument("--output_dir", default="evaluation_logs", help="Where to save logs and summary")
    args = parser.parse_args()

    print(f"📦 Loading evaluation data from: {args.eval_data}")
    try:
        eval_data = load_energy_data(args.eval_data)
        print(f"✅ Loaded evaluation data: {eval_data.shape}")
        if "timestamp" in eval_data.columns and eval_data["timestamp"].notna().any():
            ts = eval_data["timestamp"].dropna()
            print(f"📅 Date range: {ts.iloc[0]} → {ts.iloc[-1]}")
    except Exception as e:
        print(f"❌ Error loading evaluation data: {e}")
        return

    if not os.path.exists(args.trained_agents):
        print(f"❌ Trained agents directory not found: {args.trained_agents}")
        print("💡 Train agents first with your main training script.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"evaluation_metrics_{timestamp}.csv")

    results = evaluate_trained_agents(
        data=eval_data,
        trained_agents_dir=args.trained_agents,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
        evaluation_steps=args.eval_steps,
        log_path=log_path
    )

    if not results:
        print("❌ Evaluation failed or no results produced.")
        return

    print("\n🎉 Evaluation completed successfully!")
    print(f"📈 Detailed metrics CSV: {results['log_path']}")
    print(f"📊 Output directory: {args.output_dir}")

    # Display key performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print(f"\n📊 KEY PERFORMANCE METRICS:")
        print("-" * 40)

        if 'annual_return' in metrics:
            print(f"📈 Annualized Return: {metrics['annual_return']:+.2%}")
        if 'sharpe_ratio' in metrics:
            print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        if 'max_drawdown' in metrics:
            print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        if 'win_rate' in metrics:
            print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        if 'effectiveness_score' in metrics:
            print(f"🏆 Effectiveness Score: {metrics['effectiveness_score']:.1f}/100")
        if 'performance_class' in metrics:
            print(f"🎖️ Performance Class: {metrics['performance_class']}")

        # Quick interpretation
        if 'effectiveness_score' in metrics:
            score = metrics['effectiveness_score']
            if score >= 80:
                print("🚀 EXCELLENT: System is highly effective!")
            elif score >= 60:
                print("✅ GOOD: System shows strong performance!")
            elif score >= 40:
                print("📊 MODERATE: System has potential for improvement.")
            else:
                print("⚠️ POOR: System needs significant optimization.")

    # Write enhanced summary
    summary_path = os.path.join(args.output_dir, f"evaluation_summary_{timestamp}.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation Data: {args.eval_data}\n")
            f.write(f"Trained Agents: {args.trained_agents}\n")
            f.write(f"Steps Evaluated: {results['steps_evaluated']}\n")

            # Performance metrics
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                f.write(f"\nPERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")

                if 'total_return' in metrics:
                    f.write(f"Total Return: {metrics['total_return']:+.2%}\n")
                if 'annual_return' in metrics:
                    f.write(f"Annualized Return: {metrics['annual_return']:+.2%}\n")
                if 'sharpe_ratio' in metrics:
                    f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n")
                if 'max_drawdown' in metrics:
                    f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
                if 'win_rate' in metrics:
                    f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
                if 'volatility' in metrics:
                    f.write(f"Volatility: {metrics['volatility']:.2%}\n")

                if 'avg_risk_level' in metrics:
                    f.write(f"Average Risk Level: {metrics['avg_risk_level']:.3f}\n")
                if 'avg_reward' in metrics:
                    f.write(f"Average Reward: {metrics['avg_reward']:.3f}\n")

                if 'effectiveness_score' in metrics:
                    f.write(f"\nEFFECTIVENESS SCORE: {metrics['effectiveness_score']:.1f}/100\n")
                if 'performance_class' in metrics:
                    f.write(f"PERFORMANCE CLASS: {metrics['performance_class']}\n")

                # Interpretation
                f.write(f"\nINTERPRETATION\n")
                f.write("-" * 30 + "\n")

                if 'sharpe_ratio' in metrics:
                    sharpe = metrics['sharpe_ratio']
                    if sharpe > 1.5:
                        f.write("✅ Excellent risk-adjusted returns (Sharpe > 1.5)\n")
                    elif sharpe > 1.0:
                        f.write("✅ Good risk-adjusted returns (Sharpe > 1.0)\n")
                    elif sharpe > 0.5:
                        f.write("📊 Moderate risk-adjusted returns (Sharpe > 0.5)\n")
                    else:
                        f.write("⚠️ Poor risk-adjusted returns (Sharpe < 0.5)\n")

                if 'max_drawdown' in metrics:
                    dd = abs(metrics['max_drawdown'])
                    if dd < 0.10:
                        f.write("🛡️ Excellent risk control (drawdown < 10%)\n")
                    elif dd < 0.20:
                        f.write("✅ Good risk control (drawdown < 20%)\n")
                    elif dd < 0.30:
                        f.write("⚠️ Moderate risk (drawdown < 30%)\n")
                    else:
                        f.write("🚨 High risk (drawdown > 30%)\n")

            # Legacy metrics for backward compatibility
            pv = results["portfolio_values"]
            f.write(f"\nLEGACY METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Portfolio Values Logged: {len(pv)}\n")
            if pv:
                try:
                    roi = (pv[-1] - pv[0]) / max(1e-9, pv[0]) * 100.0
                    f.write(f"Portfolio ROI (simple): {roi:+.2f}%\n")
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ Failed to write summary: {e}")
    else:
        print(f"📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
