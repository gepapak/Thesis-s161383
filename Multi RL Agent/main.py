#!/usr/bin/env python3
"""
Enhanced Multi-Agent RL with Hyperparameter Optimization and Multi-Objective Rewards
Step 2: Multi-Agent RL with Multi-Horizon Forecasting (Enhanced Version)

Deep Learning Portfolio Allocation Overlay (ONLINE):
- Online self-labeling during RL: generate Markowitz targets from forecasts on the fly.
- Buffer (features, targets) and do tiny Keras fits periodically.
- Maps DL weights -> investor_0 action only (meta/risk/battery untouched).
"""

import argparse
import os
from datetime import datetime
import json
import random
import numpy as np
import pandas as pd
import torch  # for device availability check
from collections import deque

# ---- Optional SB3 bits (callback base) ----
from stable_baselines3.common.callbacks import BaseCallback
from RenewableMultiAgentEnv import RenewableMultiAgentEnv
from multi_horizon_generator import MultiHorizonForecastGenerator
from multi_horizon_wrapper import MultiHorizonWrapperEnv

# DL portfolio model (kept as-is in your repo)
from portfolio_optimization_dl import DeepPortfolioOptimizer

# CVXPY is optional; if unavailable we use a heuristic labeler.
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

try:
    from metacontroller import MultiESGAgent, HyperparameterOptimizer
except Exception:
    from metacontroller import MultiESGAgent
    HyperparameterOptimizer = None


# =====================================================================
# Inlined utilities so utils.py is no longer needed
# =====================================================================

def load_energy_data(csv_path: str) -> pd.DataFrame:
    """
    Load energy time series data from CSV.
    Requires at least: wind, solar, hydro, price, load.
    Keeps extra cols like timestamp, risk, scenario, etc. if present.
    Casts numeric columns to float where possible and parses timestamp when present.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse timestamp if present (or build from date+time)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    # (We keep timestamp as a column; we do NOT set it as index so the env sees all columns.)

    required = ["wind", "solar", "hydro", "price", "load"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Best-effort numeric casting for core cols + common extras
    numeric_extra = [c for c in ["risk", "revenue", "battery_energy", "npv"] if c in df.columns]
    for col in required + numeric_extra:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop obviously bad rows (optional, conservative)
    df = df.dropna(subset=required).reset_index(drop=True)

    return df


# =====================================================================
# Stable-Baselines3-compatible callback (optional)
# =====================================================================

class RewardAdaptationCallback(BaseCallback):
    """
    Periodically asks the base env to adapt reward weights (if available).
    """
    def __init__(self, env, adaptation_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.adaptation_freq = max(1, int(adaptation_freq))
        self._step_count = 0

    def _on_training_start(self) -> None:
        return None

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.adaptation_freq == 0:
            if hasattr(self.env, "adapt_reward_weights"):
                try:
                    self.env.adapt_reward_weights()
                except Exception:
                    # Never crash training on adaptation errors
                    pass
        return True


# =====================================================================
# Config + HPO helpers
# =====================================================================

class EnhancedConfig:
    """Enhanced configuration class with optimization support"""
    def __init__(self, optimized_params=None):
        # Defaults
        self.update_every = 128
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.verbose = 1
        self.seed = 42
        self.multithreading = True

        # PPO-ish bits
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

        # Network
        self.net_arch = [128, 64]
        self.activation_fn = "tanh"

        # Agent policies
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "PPO"},  # battery_operator_0
            {"mode": "PPO"},  # risk_controller_0
            {"mode": "SAC"},  # meta_controller_0
        ]

        if optimized_params:
            self._apply_optimized_params(optimized_params)

    def _apply_optimized_params(self, params):
        print("🎯 Applying optimized hyperparameters...")

        # Accept either 'update_every' or 'n_steps'
        self.update_every = int(params.get('update_every', params.get('n_steps', self.update_every)))

        # Learning parameters
        self.lr = float(params.get('lr', self.lr))
        self.ent_coef = params.get('ent_coef', self.ent_coef)
        self.batch_size = int(params.get('batch_size', self.batch_size))
        self.gamma = float(params.get('gamma', self.gamma))
        self.gae_lambda = float(params.get('gae_lambda', self.gae_lambda))
        self.clip_range = float(params.get('clip_range', self.clip_range))
        self.vf_coef = float(params.get('vf_coef', self.vf_coef))
        self.max_grad_norm = float(params.get('max_grad_norm', self.max_grad_norm))

        # Net arch: take explicit list if provided; otherwise map from size label
        if isinstance(params.get('net_arch'), (list, tuple)):
            self.net_arch = list(params['net_arch'])
        else:
            net_arch_mapping = {
                'small': [64, 32],
                'medium': [128, 64],
                'large': [256, 128, 64]
            }
            self.net_arch = net_arch_mapping.get(params.get('net_arch_size', 'medium'), [128, 64])

        # Activation: accept 'activation' or 'activation_fn'
        self.activation_fn = params.get('activation', params.get('activation_fn', self.activation_fn))

        # Agent modes: if a full list is provided, use it; otherwise accept *_mode aliases
        if isinstance(params.get('agent_policies'), list) and params['agent_policies']:
            self.agent_policies = params['agent_policies']
        else:
            self.agent_policies = [
                {"mode": params.get('investor_mode', 'PPO')},
                {"mode": params.get('battery_mode', 'PPO')},
                {"mode": params.get('risk_mode', 'PPO')},
                {"mode": params.get('meta_mode', 'SAC')},
            ]

        print(f"  📚 Learning rate: {self.lr:.2e}")
        print(f"  🎲 Entropy coefficient: {self.ent_coef}")
        print(f"  🏗️ Network architecture: {self.net_arch}")
        print(f"  🤖 Agent modes: {[p['mode'] for p in self.agent_policies]}")
        print(f"  🔁 Update every: {self.update_every}")
        print(f"  📦 Batch size: {self.batch_size}")


def _perf_to_float(best_performance):
    """Return a printable float if we can, else 0.0."""
    if isinstance(best_performance, dict):
        return float(best_performance.get("heuristic_score", 0.0))
    try:
        return float(best_performance)
    except Exception:
        return 0.0


def run_hyperparameter_optimization(env, device="cpu", n_trials=30, timeout=1800):
    print("🔧 Starting Enhanced Hyperparameter Optimization")
    print("=" * 55)

    if HyperparameterOptimizer is None:
        print("(Skipping HPO: HyperparameterOptimizer not available)")
        return None, None

    optimizer = HyperparameterOptimizer(
        env=env,
        device=device,
        n_trials=n_trials,
        timeout=timeout
    )

    print(f"⚙️ Configuration:")
    print(f"   Number of trials: {n_trials}")
    print(f"   Timeout: {timeout} seconds ({timeout/60:.1f} minutes)")
    print(f"   Device: {device}")
    print(f"   Environment: {env.__class__.__name__}")

    try:
        best_params, best_performance = optimizer.optimize()
        perf_value = _perf_to_float(best_performance)
        print(f"\n🏆 Optimization Results:")
        print(f"   Best heuristic score: {perf_value:.4f}")
        if isinstance(best_performance, dict):
            print(f"   Raw performance: {best_performance}")
        print(f"   Optimization completed successfully!")
        return best_params, best_performance
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("🔄 Falling back to default parameters")
        return None, None


def save_optimization_results(best_params, best_performance, save_dir="optimization_results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    params_file = os.path.join(save_dir, f"best_params_{timestamp}.json")
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)

    results_file = os.path.join(save_dir, f"optimization_summary_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        if isinstance(best_performance, dict):
            f.write(f"Heuristic Score: {best_performance.get('heuristic_score', 0.0):.6f}\n")
            for k, v in best_performance.items():
                if k != "heuristic_score":
                    f.write(f"{k}: {v}\n")
        else:
            try:
                f.write(f"Score: {float(best_performance):.6f}\n")
            except Exception:
                f.write(f"Score: {best_performance}\n")
        f.write(f"Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"💾 Optimization results saved:")
    print(f"   Parameters: {params_file}")
    print(f"   Summary: {results_file}")

    return params_file


def load_previous_optimization(optimization_dir="optimization_results"):
    if not os.path.exists(optimization_dir):
        return None, None

    # FIX: os.path.listdir -> os.listdir
    param_files = [f for f in os.listdir(optimization_dir)
                   if f.startswith("best_params_") and f.endswith(".json")]
    if not param_files:
        return None, None

    latest_file = max(param_files, key=lambda x: os.path.getctime(os.path.join(optimization_dir, x)))

    try:
        with open(os.path.join(optimization_dir, latest_file), 'r') as f:
            best_params = json.load(f)
        print(f"📂 Loaded previous optimization results from: {latest_file}")
        return best_params, latest_file
    except Exception as e:
        print(f"⚠️ Failed to load previous optimization: {e}")
        return None, None


def setup_enhanced_training_monitoring(log_path, save_dir):
    monitoring_dirs = {
        'checkpoints': os.path.join(save_dir, 'checkpoints'),
        'metrics': os.path.join(save_dir, 'metrics'),
        'models': os.path.join(save_dir, 'models')
    }
    for _, dir_path in monitoring_dirs.items():
        os.makedirs(dir_path, exist_ok=True)

    print("📊 Enhanced monitoring setup:")
    print(f"   Metrics log: {log_path}")
    print(f"   Checkpoints: {monitoring_dirs['checkpoints']}")
    print(f"   Model saves: {monitoring_dirs['models']}")
    return monitoring_dirs


def enhanced_training_loop(agent, env, timesteps, checkpoint_freq, monitoring_dirs, callbacks=None):
    """
    Train in intervals, but measure *actual* steps each time.
    Works with a meta-controller whose learn(total_timesteps=N) means "do N more steps".
    """
    print("🚀 Starting Enhanced Training Loop")
    print(f"   Total timesteps: {timesteps:,}")
    print(f"   Checkpoint frequency: {checkpoint_freq:,}")

    total_trained = 0
    checkpoint_count = 0

    try:
        while total_trained < timesteps:
            remaining = timesteps - total_trained
            interval = min(checkpoint_freq, remaining)

            print(f"\n🏃 Training interval {checkpoint_count + 1}")
            print(f"   Steps: {total_trained:,} → {total_trained + interval:,}")
            print(f"   Progress: {total_trained/timesteps*100:.1f}%")

            start_time = datetime.now()
            start_steps = getattr(agent, "total_steps", 0)

            # NOTE: assumes meta-controller treats this as a *relative* budget.
            agent.learn(total_timesteps=interval, callbacks=callbacks)

            end_time = datetime.now()
            end_steps = getattr(agent, "total_steps", start_steps)
            trained_now = max(0, int(end_steps) - int(start_steps))

            # Update totals using the *actual* number of steps collected.
            total_trained += trained_now
            checkpoint_count += 1

            training_time = (end_time - start_time).total_seconds()
            sps = (trained_now / training_time) if training_time > 0 else 0.0
            print(f"   ⏱️ Training time: {training_time:.1f}s ({sps:.1f} steps/s)")
            print(f"   ✅ Actual steps collected this interval: {trained_now:,} (agent.total_steps = {end_steps:,})")

            if trained_now == 0:
                print("⚠️ No steps were collected in this interval. "
                      "Check that the meta-controller's learn() uses a relative budget or increase n_steps.")
                # Avoid infinite loop; break so we can at least save progress.
                break

            # Save checkpoint only if we actually progressed and still have more to do
            if total_trained < timesteps and trained_now > 0:
                checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"checkpoint_{total_trained}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"💾 Saving checkpoint at {total_trained:,} steps...")
                saved_count = agent.save_policies(checkpoint_dir)

                state_file = os.path.join(checkpoint_dir, "training_state.json")
                with open(state_file, 'w') as f:
                    json.dump({
                        'total_trained': int(total_trained),
                        'agent_total_steps': int(end_steps),
                        'checkpoint_count': int(checkpoint_count),
                        'timestamp': datetime.now().isoformat(),
                        'performance_summary': {'steps_per_second': sps, 'training_time': training_time}
                    }, f, indent=2)
                print(f"   ✅ Checkpoint saved: {saved_count} policies")

            # Opportunistic metrics flush to avoid buffer loss if the process stops unexpectedly
            try:
                if hasattr(env, "_flush_log_buffer"):
                    env._flush_log_buffer()
            except Exception:
                pass

        print("\n🎉 Enhanced training completed!")
        print(f"   Total steps (budget accumulation): {total_trained:,}")
        print(f"   Agent-reported total steps: {getattr(agent, 'total_steps', 'unknown')}")
        print(f"   Checkpoints created: {checkpoint_count}")
        return total_trained

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"   Progress: {total_trained:,}/{timesteps:,} ({total_trained/timesteps*100:.1f}%)")
        emergency_dir = os.path.join(monitoring_dirs['checkpoints'], f"emergency_{total_trained}")
        os.makedirs(emergency_dir, exist_ok=True)
        agent.save_policies(emergency_dir)
        print(f"💾 Emergency checkpoint saved to: {emergency_dir}")
        return total_trained

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


def analyze_training_performance(env, log_path, monitoring_dirs):
    print("\n📈 Training Performance Analysis")
    print("=" * 40)
    try:
        # If env exposes a reward analysis method
        if hasattr(env, 'get_reward_analysis'):
            env.get_reward_analysis()

        if os.path.exists(log_path):
            metrics = pd.read_csv(log_path)
            if len(metrics) > 100 and 'meta_reward' in metrics.columns:
                early = metrics['meta_reward'].head(100).mean()
                late = metrics['meta_reward'].tail(100).mean()
                improvement = late - early
                pct = (improvement / abs(early) * 100) if abs(early) > 1e-12 else 0.0
                print("📊 Training Metrics Summary:")
                print(f"   Rows logged (wrapper): {len(metrics):,}")
                print(f"   Early performance: {early:.4f}")
                print(f"   Late performance:  {late:.4f}")
                print(f"   Improvement:       {improvement:+.4f} ({pct:+.1f}%)")

        ckpt_dir = monitoring_dirs.get('checkpoints', '')
        if os.path.exists(ckpt_dir):
            ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith('checkpoint_')]
            print(f"   💾 Checkpoints available: {len(ckpts)}")

    except Exception as e:
        print(f"⚠️ Performance analysis failed: {e}")


# =====================================================================
# Deep Learning Portfolio Allocation Overlay (ONLINE)
# =====================================================================

class PortfolioAdapter:
    """
    Online self-labeling:
      - Build deterministic features each step (same as inference).
      - Every 'label_every' steps, compute Markowitz weights from a rolling window of forecast*price proxies.
      - Push (features, target_weights) into a small buffer.
      - Every 'train_every' steps, fit the DL model on a random minibatch from the buffer (few epochs).
    """
    def __init__(self, base_env, feature_dim=9, buffer_size=2048, label_every=12, train_every=60,
                 window=24, lam=5.0, batch_size=128, epochs=1):
        self.e = base_env
        self.model = DeepPortfolioOptimizer(num_assets=3, num_market_features=feature_dim)

        # Online training hyperparams
        self.buffer = deque(maxlen=buffer_size)
        self.label_every = int(label_every)
        self.train_every = int(train_every)
        self.window = int(window)
        self.lam = float(lam)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)

        # Try to compile Keras model (both direct or inner .model)
        try:
            if hasattr(self.model, "compile"):
                self.model.compile(optimizer="adam", loss="mse")
            elif hasattr(self.model, "model") and hasattr(self.model.model, "compile"):
                self.model.model.compile(optimizer="adam", loss="mse")
        except Exception:
            pass

        # Optional: try load previous online-trained weights to warm-start
        for candidate in ["dl_allocator_online.h5", "dl_allocator_weights.h5"]:
            try:
                if hasattr(self.model, "load_weights"):
                    self.model.load_weights(candidate)
                    print(f"ℹ️ Loaded allocator weights: {candidate}")
                    break
                elif hasattr(self.model, "model") and hasattr(self.model.model, "load_weights"):
                    self.model.model.load_weights(candidate)
                    print(f"ℹ️ Loaded allocator weights: {candidate}")
                    break
            except Exception:
                continue

    # ---------- feature builder ----------
    def _market_state(self, t: int) -> np.ndarray:
        e = self.e
        def get(arr_name, default=0.0):
            try:
                arr = getattr(e, arr_name)
                return float(arr[t]) if (hasattr(arr, "__len__") and t < len(arr)) else float(default)
            except Exception:
                return float(default)

        price = get("_price", 0.0)
        load  = get("_load", 0.0)
        wind  = get("_wind", 0.0)
        solar = get("_solar", 0.0)
        hydro = get("_hydro", 0.0)

        try:
            budget_ratio = float(e.budget / max(1e-6, e.init_budget))
        except Exception:
            budget_ratio = 1.0
        cap_frac = float(getattr(e, "capital_allocation_fraction", 0.5))

        try:
            fmin = float(getattr(e, "META_FREQ_MIN", 1))
            fmax = float(getattr(e, "META_FREQ_MAX", 288))
            freq_norm = (float(getattr(e, "investment_freq", fmin)) - fmin) / max(1e-6, (fmax - fmin))
            freq_norm = float(np.clip(freq_norm, 0.0, 1.0))
        except Exception:
            freq_norm = 0.5

        PRICE_SCALE = 10.0
        feats = np.array([[price/PRICE_SCALE, load, wind, solar, hydro,
                           budget_ratio, cap_frac, freq_norm, 1.0]], dtype=np.float32)
        return feats

    # ---------- position snapshot ----------
    def _positions(self) -> np.ndarray:
        e = self.e
        w = float(getattr(e, "wind_capacity", 0.0))
        s = float(getattr(e, "solar_capacity", 0.0))
        h = float(getattr(e, "hydro_capacity", 0.0))
        return np.array([[w, s, h]], dtype=np.float32)

    # ---------- labeler ----------
    def _target_weights(self, t: int) -> np.ndarray:
        """
        Build expected-return proxy from immediate forecasts * price, over a rolling window.
        Solve mean-variance with non-negativity and sum-to-1 constraints.
        """
        def arr(name_fcast, name_actual):
            a = getattr(self.e, name_fcast, None)
            if a is None:
                a = getattr(self.e, name_actual, None)
            return np.array(a, dtype=float) if a is not None else None

        pf = arr("_price_forecast_immediate", "_price")
        fw = arr("_wind_forecast_immediate", "_wind")
        fs = arr("_solar_forecast_immediate", "_solar")
        fh = arr("_hydro_forecast_immediate", "_hydro")

        if pf is None or fw is None or fs is None:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        start = max(0, t - self.window + 1)
        p = pf[start:t+1]
        R = np.stack([
            fw[start:t+1] * p,
            fs[start:t+1] * p,
            (fh[start:t+1] if fh is not None else np.zeros_like(p))
        ], axis=1)

        if R.shape[0] < 2:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        mu = R.mean(axis=0)
        Sigma = np.cov(R.T) + 1e-6 * np.eye(3)

        if _HAS_CVXPY:
            w = cp.Variable(3)
            objective = cp.Maximize(mu @ w - self.lam * cp.quad_form(w, Sigma))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                if w.value is not None:
                    out = np.maximum(w.value, 0)
                    s = out.sum()
                    return (out / s).astype(np.float32) if s > 1e-8 else np.array([1/3,1/3,1/3], np.float32)
            except Exception:
                pass

        # heuristic fallback: normalized positive means
        out = np.clip(mu, 0, None)
        s = out.sum()
        return (out / s).astype(np.float32) if s > 1e-8 else np.array([1/3,1/3,1/3], np.float32)

    # ---------- inference ----------
    def infer_weights(self, t: int) -> np.ndarray:
        try:
            out = self.model(
                {'market_state': self._market_state(t), 'current_positions': self._positions()},
                training=False
            )
            w = np.asarray(out['weights'].numpy()[0], dtype=np.float32)
            s = float(np.sum(w))
            return w / (s if s > 1e-8 else 1.0)
        except Exception:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    # ---------- map to env action ----------
    def weights_to_action(self, w: np.ndarray) -> np.ndarray:
        e = self.e
        wcap = float(getattr(e, "wind_capacity", 0.0))
        scap = float(getattr(e, "solar_capacity", 0.0))
        hcap = float(getattr(e, "hydro_capacity", 0.0))
        total = max(1e-6, wcap + scap + hcap)
        cur = np.array([wcap, scap, hcap], dtype=np.float32) / total
        delta = np.clip(w - cur, -0.2, 0.2) * 5.0  # -> roughly [-1,1]
        return np.clip(delta.astype(np.float32), -1.0, 1.0)

    # ---------- online training hook ----------
    def maybe_learn(self, t: int):
        # (1) label a sample periodically
        if t % self.label_every == 0:
            X = self._market_state(t)[0]    # (feature_dim,)
            Y = self._target_weights(t)     # (3,)
            self.buffer.append((X, Y))

        # (2) fit a minibatch periodically
        if len(self.buffer) >= self.batch_size and t % self.train_every == 0:
            idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
            Xb = np.stack([self.buffer[i][0] for i in idx], axis=0).astype(np.float32)
            Yb = np.stack([self.buffer[i][1] for i in idx], axis=0).astype(np.float32)
            try:
                if hasattr(self.model, "fit"):
                    self.model.fit(Xb, Yb, epochs=self.epochs, batch_size=self.batch_size, verbose=0, shuffle=True)
                elif hasattr(self.model, "model") and hasattr(self.model.model, "fit"):
                    self.model.model.fit(Xb, Yb, epochs=self.epochs, batch_size=self.batch_size, verbose=0, shuffle=True)
            except Exception:
                pass


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent RL with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, default="sample.csv", help="Path to energy time series data")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cuda", help="Device for RL training (cuda/cpu)")
    parser.add_argument("--investment_freq", type=int, default=144, help="Investor action frequency in steps")
    parser.add_argument("--model_dir", type=str, default="multi_horizon_models", help="Dir with trained forecast models")
    parser.add_argument("--scaler_dir", type=str, default="multi_horizon_scalers", help="Dir with trained scalers")

    # Optimization
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")
    parser.add_argument("--optimization_trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--optimization_timeout", type=int, default=1800, help="Optimization timeout (s)")
    parser.add_argument("--use_previous_optimization", action="store_true", help="Use latest saved optimized params")

    # Training
    parser.add_argument("--save_dir", type=str, default="enhanced_rl_training", help="Where to save outputs")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Save checkpoint every N timesteps")
    parser.add_argument("--validate_env", action="store_true", default=True, help="Validate env setup before training")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # Rewards
    parser.add_argument("--adapt_rewards", action="store_true", default=True, help="Enable adaptive reward weights")
    parser.add_argument("--reward_analysis_freq", type=int, default=2000, help="Analyze rewards every N steps")

    args = parser.parse_args()

    # Safer device selection
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Seed everything for repeatability (uses config.seed after we create it; use a temp seed now too)
    seed_hint = 42
    random.seed(seed_hint); np.random.seed(seed_hint); torch.manual_seed(seed_hint)

    print("🚀 Enhanced Multi-Horizon Energy Investment RL System")
    print("=" * 60)
    print("🎯 Features: Hyperparameter Optimization + Multi-Objective Rewards")

    # Create save dir + metrics subdir
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # 1) Load data
    print(f"\n📦 Loading data from: {args.data_path}")
    try:
        data = load_energy_data(args.data_path)
        print(f"✅ Data loaded: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        if "timestamp" in data.columns and data["timestamp"].notna().any():
            ts = data["timestamp"].dropna()
            print(f"📅 Date range: {ts.iloc[0]} → {ts.iloc[-1]}")
        if len(data) < 1000:
            print(f"⚠️ Limited data ({len(data)} rows). More data → better training stability.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2) Forecaster
    print("\n🔧 Initializing multi-horizon forecaster...")
    try:
        forecaster = MultiHorizonForecastGenerator(
            model_dir=args.model_dir,
            scaler_dir=args.scaler_dir,
            look_back=6,
            verbose=True
        )
        print("✅ Forecaster initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize forecaster: {e}")
        return

    # 3) Environment setup
    print("\n🏗️ Setting up enhanced environment with multi-objective rewards...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(metrics_dir, f"enhanced_metrics_{timestamp}.csv")

    try:
        base_env = RenewableMultiAgentEnv(
            data,
            investment_freq=args.investment_freq,
            forecast_generator=forecaster
        )
        env = MultiHorizonWrapperEnv(base_env, forecaster, log_path=log_path)
        print("✅ Enhanced environment created successfully!")
        print("   Multi-objective rewards: ✅")
        print("   Enhanced risk management: ✅")
        print("   Forecast-augmented observations via wrapper: ✅")
        print(f"   Metrics CSV: {log_path}")
    except Exception as e:
        print(f"❌ Failed to setup environment: {e}")
        return

    # Optional one-time validation (safe)
    if args.validate_env:
        try:
            _obs, _ = env.reset()
            print("✅ Env reset OK for validation.")
        except Exception as e:
            print(f"⚠️ Env validation reset failed (continuing): {e}")

    # === Activate DL allocation overlay (ONLINE self-labeling) ===
    try:
        adapter = PortfolioAdapter(base_env)
        _orig_process = base_env._process_actions_safe

        def _patched_process(actions, t):
            # Online learning step (cheap and infrequent)
            adapter.maybe_learn(t)

            # Inference to guide investor_0
            w = adapter.infer_weights(t)
            actions['investor_0'] = adapter.weights_to_action(w)
            return _orig_process(actions, t)

        base_env._process_actions_safe = _patched_process
        print("✅ DL allocation overlay active (online self-labeling enabled)")
    except Exception as e:
        print(f"⚠️ Failed to activate DL allocation overlay: {e}")
    # === END overlay ===

    # 4) (Optional) HPO
    best_params = None
    if args.use_previous_optimization:
        print("\n🔍 Checking for previous optimization results...")
        best_params, _ = load_previous_optimization(os.path.join(args.save_dir, "optimization_results"))
        if best_params:
            print("✅ Using previous optimization results")
        else:
            print("⚠️ No previous optimization found")

    if args.optimize and not best_params:
        print("\n🎯 Running hyperparameter optimization...")
        opt_data = data.head(min(5000, len(data)))
        opt_base_env = RenewableMultiAgentEnv(opt_data, forecast_generator=forecaster)
        opt_env = MultiHorizonWrapperEnv(opt_base_env, forecaster, log_path=None)

        best_params, best_perf = run_hyperparameter_optimization(
            opt_env,
            device=args.device,
            n_trials=args.optimization_trials,
            timeout=args.optimization_timeout
        )
        if best_params:
            opt_dir = os.path.join(args.save_dir, "optimization_results")
            save_optimization_results(best_params, best_perf, opt_dir)
        del opt_env, opt_base_env

    # 5) Config
    print("\n⚙️ Creating optimized training configuration...")
    config = EnhancedConfig(optimized_params=best_params)

    # Re-seed using config.seed to ensure consistency with agent init
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)

    # 6) Agents
    print("\n🤖 Initializing enhanced multi-agent RL system...")
    try:
        agent = MultiESGAgent(
            config,
            env=env,
            device=args.device,
            training=True,
            debug=args.debug
        )
        print("✅ Enhanced multi-agent system initialized")
        print(f"   Device: {args.device}")
        print(f"   Agents: {env.possible_agents}")
        print(f"   Learning rate: {config.lr:.2e}")
        print(f"   Update frequency: {config.update_every}")
        print("   Multi-objective rewards: ✅")
        print("   Adaptive hyperparameters: ✅")
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        return

    # 7) Monitoring
    monitoring_dirs = setup_enhanced_training_monitoring(log_path, args.save_dir)

    # 8) Training
    print("\n🚀 Starting Enhanced Multi-Objective Training...")
    print(f"   Training timesteps: {args.timesteps:,}")
    print(f"   Checkpoint frequency: {args.checkpoint_freq:,}")
    print(f"   Adaptive rewards: {'✅' if args.adapt_rewards else '❌'}")
    print("   Performance monitoring: ✅")

    try:
        callbacks = None
        if args.adapt_rewards:
            reward_cb = RewardAdaptationCallback(base_env, args.reward_analysis_freq, verbose=0)
            # replicate so multi-agent wrapper/agent sees a list of callbacks (one per policy)
            callbacks = [reward_cb] * len(env.possible_agents)

        total_trained = enhanced_training_loop(
            agent=agent,
            env=env,
            timesteps=args.timesteps,
            checkpoint_freq=args.checkpoint_freq,
            monitoring_dirs=monitoring_dirs,
            callbacks=callbacks
        )
        print("✅ Enhanced training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    # 9) Save final models
    print("\n💾 Saving final trained models...")
    final_dir = os.path.join(args.save_dir, "final_models")
    os.makedirs(final_dir, exist_ok=True)
    saved_count = agent.save_policies(final_dir)

    if saved_count > 0:
        print(f"✅ Saved {saved_count} trained agents to: {final_dir}")
        cfg_file = os.path.join(final_dir, "training_config.json")
        with open(cfg_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data_path': args.data_path,
                'total_timesteps_budgeted': int(total_trained),
                'agent_total_steps': int(getattr(agent, 'total_steps', 0)),
                'device': args.device,
                'optimized_params': best_params,
                'final_config': {
                    'lr': config.lr,
                    'ent_coef': config.ent_coef,
                    'batch_size': config.batch_size,
                    'agent_policies': config.agent_policies,
                    'net_arch': config.net_arch,
                    'activation_fn': config.activation_fn,
                    'update_every': config.update_every
                }
            }, f, indent=2)
        print(f"📋 Training configuration saved to: {cfg_file}")

    # 10) Save the online-trained allocator (if possible)
    try:
        out_weights = os.path.join(args.save_dir, "dl_allocator_online.h5")
        if hasattr(adapter.model, "save_weights"):
            adapter.model.save_weights(out_weights)
            print(f"💾 Saved online-trained DL allocator to: {out_weights}")
        elif hasattr(adapter.model, "model") and hasattr(adapter.model.model, "save_weights"):
            adapter.model.model.save_weights(out_weights)
            print(f"💾 Saved online-trained DL allocator to: {out_weights}")
    except Exception as e:
        print(f"ℹ️ Could not save DL allocator weights: {e}")

    # 11) Force a final log flush (avoid losing buffered rows)
    try:
        if hasattr(env, "_flush_log_buffer"):
            env._flush_log_buffer()
    except Exception:
        pass

    # 12) Quick post-run analysis
    analyze_training_performance(env, log_path, monitoring_dirs)


if __name__ == "__main__":
    main()
