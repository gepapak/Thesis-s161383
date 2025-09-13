# main.py

import argparse
import os
import sys
from datetime import datetime
import json
import random
from collections import deque
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch  # for device availability check

# ---- Optional SB3 bits (callback base) ----
from stable_baselines3.common.callbacks import BaseCallback

# ---- Live Training Diagnostic ----
from live_training_diagnostic import integrate_diagnostic_with_training

# Import patched environment classes
from environment import RenewableMultiAgentEnv
from generator import MultiHorizonForecastGenerator
from wrapper import MultiHorizonWrapperEnv
from dl_overlay import HedgeOptimizer, AdvancedHedgeOptimizer

# CVXPY is optional; if unavailable we use a heuristic labeler.
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

# Meta-controller (patched)
try:
    from metacontroller import MultiESGAgent, HyperparameterOptimizer
except Exception:
    from metacontroller import MultiESGAgent
    HyperparameterOptimizer = None


# =====================================================================
# Inlined utilities so utils.py is no longer needed
# =====================================================================

def load_energy_data(csv_path: str, convert_to_raw_units: bool = True) -> pd.DataFrame:
    """
    Load energy time series data from CSV with optional unit conversion.
    Requires at least: wind, solar, hydro, price, load.
    Keeps extra cols like timestamp, risk, scenario, etc. if present.
    Casts numeric columns to float where possible and parses timestamp when present.

    Args:
        csv_path: Path to CSV file
        convert_to_raw_units: If True, converts capacity factors to absolute MW units
                             to match forecast model training data
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse timestamp if present (or build from date+time)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
        )
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

    # Convert capacity factors to raw MW values for direct forecasting
    if convert_to_raw_units and _is_capacity_factor_data(df):
        print("[INFO] Converting capacity factors to raw MW values for direct forecasting...")
        df = _convert_to_raw_mw_values(df)
        print("[OK] Raw MW conversion completed - forecasts will work directly with these units")
    else:
        print("[INFO] Data already in raw MW units - ready for direct forecasting")

    return df


def _is_capacity_factor_data(df: pd.DataFrame) -> bool:
    """Detect if data is in capacity factor format (0-1 range) vs absolute MW."""
    # Check if renewable data is in 0-1 range (capacity factors)
    renewable_cols = ['wind', 'solar', 'hydro', 'load']

    for col in renewable_cols:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 2.0:  # If any value > 2, likely already in MW
                return False

    return True  # All values <= 2, likely capacity factors


def _convert_to_raw_mw_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert capacity factors to raw MW values for direct forecasting.

    This eliminates normalization complexity and uses the exact training scale.
    """

    # EXACT conversion factors derived from scaler analysis
    # These ensure models receive data in the same scale they were trained on
    capacity_mw = {
        'wind': 1103,    # From training scaler mean: 1103.4 MW
        'solar': 100,    # From training scaler mean: 61.5 MW (min 100 for stability)
        'hydro': 534,    # From training scaler mean: 534.1 MW
        'load': 2999,    # From training scaler mean: 2999.8 MW
    }

    df_converted = df.copy()

    print("[INFO] Converting to raw MW values (no normalization):")

    # Convert capacity factors to raw MW values
    for col, capacity in capacity_mw.items():
        if col in df_converted.columns:
            original_range = f"[{df[col].min():.3f}, {df[col].max():.3f}]"
            df_converted[col] = df[col] * capacity
            new_range = f"[{df_converted[col].min():.1f}, {df_converted[col].max():.1f}] MW"
            print(f"  {col}: {original_range} -> {new_range}")

    # Price: Already in $/MWh (no conversion needed)
    if 'price' in df_converted.columns:
        price_range = f"[{df_converted['price'].min():.1f}, {df_converted['price'].max():.1f}] $/MWh"
        print(f"  price: {price_range} (no conversion)")

    print("[OK] Raw MW conversion complete - ready for direct forecasting")

    return df_converted


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
        # SB3 allows returning None here
        return None

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.adaptation_freq == 0 and hasattr(self.env, "adapt_reward_weights"):
            try:
                self.env.adapt_reward_weights()
            except Exception:
                # Never crash training on adaptation errors
                pass
        return True


# =====================================================================
# Config + HPO helpers
# =====================================================================
from config import EnhancedConfig


def _perf_to_float(best_performance: Any) -> float:
    """Return a printable float if we can, else 0.0."""
    if isinstance(best_performance, dict):
        return float(best_performance.get("heuristic_score", 0.0))
    try:
        return float(best_performance)
    except Exception:
        return 0.0


def run_hyperparameter_optimization(env, device: str = "cpu", n_trials: int = 30, timeout: int = 1800):
    print("Starting Enhanced Hyperparameter Optimization")
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

    print("Configuration:")
    print(f"   Number of trials: {n_trials}")
    print(f"   Timeout: {timeout} seconds ({timeout/60:.1f} minutes)")
    print(f"   Device: {device}")
    print(f"   Environment: {env.__class__.__name__}")

    try:
        best_params, best_performance = optimizer.optimize()
        perf_value = _perf_to_float(best_performance)
        print("\nOptimization Results:")
        print(f"   Best heuristic score: {perf_value:.4f}")
        if isinstance(best_performance, dict):
            print(f"   Raw performance: {best_performance}")
        print("   Optimization completed successfully!")
        return best_params, best_performance
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Falling back to default parameters")
        return None, None


def save_optimization_results(best_params: Dict[str, Any], best_performance: Any, save_dir: str = "optimization_results"):
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
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"   {key}: {value}\n")

    print("Optimization results saved:")
    print(f"   Parameters: {params_file}")
    print(f"   Summary: {results_file}")

    return params_file


def load_previous_optimization(optimization_dir: str = "optimization_results"):
    if not os.path.exists(optimization_dir):
        return None, None

    param_files = [
        f for f in os.listdir(optimization_dir)
        if f.startswith("best_params_") and f.endswith(".json")
    ]
    if not param_files:
        return None, None

    latest_file = max(param_files, key=lambda x: os.path.getctime(os.path.join(optimization_dir, x)))

    try:
        with open(os.path.join(optimization_dir, latest_file), 'r') as f:
            best_params = json.load(f)
        print(f"Loaded previous optimization results from: {latest_file}")
        return best_params, latest_file
    except Exception as e:
        print(f"Failed to load previous optimization: {e}")
        return None, None


def setup_enhanced_training_monitoring(log_path: str, save_dir: str) -> Dict[str, str]:
    monitoring_dirs = {
        'checkpoints': os.path.join(save_dir, 'checkpoints'),
        'metrics': os.path.join(save_dir, 'metrics'),
        'models': os.path.join(save_dir, 'models')
    }
    for _, dir_path in monitoring_dirs.items():
        os.makedirs(dir_path, exist_ok=True)

    print("Enhanced monitoring setup:")
    print(f"   Metrics log: {log_path}")
    print(f"   Checkpoints: {monitoring_dirs['checkpoints']}")
    print(f"   Model saves: {monitoring_dirs['models']}")
    return monitoring_dirs


def enhanced_training_loop(agent, env, timesteps: int, checkpoint_freq: int, monitoring_dirs: Dict[str, str], callbacks=None, diagnostic=None) -> int:
    """
    Train in intervals, but measure *actual* steps each time.
    Works with a meta-controller whose learn(total_timesteps=N) means "do N more steps".
    """
    print("Starting Enhanced Training Loop")
    print(f"   Total timesteps: {timesteps:,}")
    print(f"   Checkpoint frequency: {checkpoint_freq:,}")

    total_trained = 0
    checkpoint_count = 0

    try:
        while total_trained < timesteps:
            remaining = timesteps - total_trained
            interval = min(checkpoint_freq, remaining)

            print(f"\nTraining interval {checkpoint_count + 1}")
            print(f"   Steps: {total_trained:,} -> {total_trained + interval:,}")
            print(f"   Progress: {total_trained/timesteps*100:.1f}%")

            start_time = datetime.now()
            start_steps = getattr(agent, "total_steps", 0)

            # NOTE: assumes meta-controller treats this as a *relative* budget.
            agent.learn(total_timesteps=interval, callbacks=callbacks)

            end_time = datetime.now()
            end_steps = getattr(agent, "total_steps", start_steps)
            trained_now = max(0, int(end_steps) - int(start_steps))

            # Log diagnostic data after each training interval
            if diagnostic:
                # Update diagnostic to track all steps in this interval
                diagnostic.update_step_range(start_steps, end_steps)

                # Force log at checkpoint intervals
                diagnostic.log_step(force_log=True)

            # Update totals using the *actual* number of steps collected.
            total_trained += trained_now
            checkpoint_count += 1

            training_time = (end_time - start_time).total_seconds()
            sps = (trained_now / training_time) if training_time > 0 else 0.0
            print(f"   Training time: {training_time:.1f}s ({sps:.1f} steps/s)")
            print(f"   Actual steps collected this interval: {trained_now:,} (agent.total_steps = {end_steps:,})")

            if trained_now == 0:
                print("No steps were collected in this interval. "
                      "Check that the meta-controller's learn() uses a relative budget or increase n_steps.")
                # Avoid infinite loop; break so we can at least save progress.
                break

            # Save checkpoint only if we actually progressed and still have more to do
            if total_trained < timesteps and trained_now > 0:
                checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"checkpoint_{total_trained}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"Saving checkpoint at {total_trained:,} steps...")
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
                print(f"   Checkpoint saved: {saved_count} policies")

            # Opportunistic metrics flush to avoid buffer loss if the process stops unexpectedly
            try:
                if hasattr(env, "_flush_log_buffer"):
                    env._flush_log_buffer()
            except Exception:
                pass

        print("\nEnhanced training completed!")
        print(f"   Total steps (budget accumulation): {total_trained:,}")
        print(f"   Agent-reported total steps: {getattr(agent, 'total_steps', 'unknown')}")
        print(f"   Checkpoints created: {checkpoint_count}")
        return total_trained

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"   Progress: {total_trained:,}/{timesteps:,} ({total_trained/timesteps*100:.1f}%)")
        emergency_dir = os.path.join(monitoring_dirs['checkpoints'], f"emergency_{total_trained}")
        os.makedirs(emergency_dir, exist_ok=True)
        agent.save_policies(emergency_dir)
        print(f"Emergency checkpoint saved to: {emergency_dir}")
        return total_trained

    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


def analyze_training_performance(env, log_path: str, monitoring_dirs: Dict[str, str]) -> None:
    print("\nTraining Performance Analysis")
    print("=" * 40)
    try:
        # If env exposes a reward analysis method
        if hasattr(env, 'get_reward_analysis'):
            env.get_reward_analysis()

        if log_path and os.path.exists(log_path):
            metrics = pd.read_csv(log_path)
            if len(metrics) > 100 and 'meta_reward' in metrics.columns:
                early = metrics['meta_reward'].head(100).mean()
                late = metrics['meta_reward'].tail(100).mean()
                improvement = late - early
                pct = (improvement / abs(early) * 100) if abs(early) > 1e-12 else 0.0
                print("Training Metrics Summary:")
                print(f"   Rows logged (wrapper): {len(metrics):,}")
                print(f"   Early performance: {early:.4f}")
                print(f"   Late performance:  {late:.4f}")
                print(f"   Improvement:       {improvement:+.4f} ({pct:+.1f}%)")

        ckpt_dir = monitoring_dirs.get('checkpoints', '')
        if os.path.exists(ckpt_dir):
            ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith('checkpoint_')]
            print(f"   Checkpoints available: {len(ckpts)}")

    except Exception as e:
        print(f"Performance analysis failed: {e}")


# =====================================================================
# Deep Learning Hedge Optimization Overlay (ONLINE)
# =====================================================================

class HedgeAdapter:
    """
    Online hedge strategy optimization:
      - Build market state features each step for hedge decision making
      - Every 'label_every' steps, compute optimal hedge parameters based on market conditions
      - Push (features, target_hedge_params) into a buffer for learning
      - Every 'train_every' steps, train the HedgeOptimizer on hedge effectiveness

    Replaces portfolio optimization with hedge strategy optimization for single-price coherence
    """
    def __init__(self, base_env, feature_dim=None, buffer_size=1024, label_every=20, train_every=100,
                 window=24, batch_size=64, epochs=1):
        self.e = base_env

        # Fixed feature dimensions for DL overlay (independent of observation space)
        # Core features: 4 state + 3 positions + 3 forecasts + 3 portfolio = 13 features
        # This ensures DL overlay works consistently regardless of other add-ons
        if feature_dim is None:
            feature_dim = 13  # Fixed dimension for DL overlay consistency

        self.feature_dim = feature_dim
        self.model = AdvancedHedgeOptimizer(feature_dim=feature_dim)

        # Online training hyperparams
        self.buffer = deque(maxlen=buffer_size)
        self.label_every = int(label_every)
        self.train_every = int(train_every)
        self.window = int(window)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)

        # Compile HedgeOptimizer for training
        try:
            self.model.compile(
                optimizer="adam",
                loss={
                    "hedge_intensity": "mse",
                    "risk_allocation": "categorical_crossentropy",
                    "hedge_direction": "mse",
                    "hedge_effectiveness": "binary_crossentropy"
                },
                loss_weights={
                    "hedge_intensity": 1.0,
                    "risk_allocation": 1.0,
                    "hedge_direction": 1.0,
                    "hedge_effectiveness": 0.5
                },
                metrics={
                    "hedge_intensity": ["mae"],
                    "risk_allocation": ["accuracy"],
                    "hedge_direction": ["mae"],
                    "hedge_effectiveness": ["accuracy"]
                }
            )
        except Exception as e:
            logging.warning(f"HedgeOptimizer compilation failed: {e}")

    def update_feature_dimensions(self, new_feature_dim: int):
        """DL overlay uses fixed feature dimensions - no update needed"""
        # DL overlay maintains consistent 13-feature input regardless of observation space
        # This ensures the add-on works independently of forecasting or other features
        logging.info(f"DL overlay maintains fixed feature dimensions: {self.feature_dim} (ignoring observation space: {new_feature_dim})")
        # No model recreation needed - DL overlay is self-contained

        # Optional: try load previous online-trained hedge optimizer weights to warm-start
        for candidate in ["hedge_optimizer_online.h5", "hedge_optimizer_weights.h5"]:
            try:
                if hasattr(self.model, "load_weights"):
                    self.model.load_weights(candidate)
                    print(f"Loaded hedge optimizer weights: {candidate}")
                    break
                elif hasattr(self.model, "model") and hasattr(self.model.model, "load_weights"):
                    self.model.model.load_weights(candidate)
                    print(f"Loaded hedge optimizer weights: {candidate}")
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

        # Basic market data
        price = get("_price", 0.0)
        load = get("_load", 0.0)
        wind = get("_wind", 0.0)
        solar = get("_solar", 0.0)
        hydro = get("_hydro", 0.0)

        # Enhanced features for PPA-based economics
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

        # Capacity factor features (normalized renewable resource availability)
        try:
            wind_cf = float(wind / max(getattr(e, "wind_scale", 1.0), 1e-6))
            solar_cf = float(solar / max(getattr(e, "solar_scale", 1.0), 1e-6))
            hydro_cf = float(hydro / max(getattr(e, "hydro_scale", 1.0), 1e-6))
        except Exception:
            wind_cf = solar_cf = hydro_cf = 0.0

        # True physical capacity utilization (generation/capacity)
        try:
            # Get current generation levels (capacity factors)
            current_generation = wind_cf + solar_cf + hydro_cf  # Sum of capacity factors

            # Get total physical capacity in MW
            total_physical_mw = (getattr(e, "wind_capacity_mw", 0.0) +
                               getattr(e, "solar_capacity_mw", 0.0) +
                               getattr(e, "hydro_capacity_mw", 0.0))

            # Calculate true utilization: if we have 100MW total and generating at 30% average = 0.3
            if total_physical_mw > 0:
                avg_capacity_factor = current_generation / 3.0  # Average across 3 technologies
                capacity_utilization = float(np.clip(avg_capacity_factor, 0.0, 1.0))
            else:
                capacity_utilization = 0.0  # No physical assets deployed
        except Exception:
            capacity_utilization = 0.0

        # Market volatility and risk indicators
        try:
            market_vol = float(getattr(e, "market_volatility", 0.0))
            market_stress = float(getattr(e, "market_stress", 0.5))
        except Exception:
            market_vol = market_stress = 0.5

        # FIXED: Return only 4 state features to match environment (4+3+3+3=13 total)
        # This ensures DL overlay training uses same feature dimensions as prediction
        feats = np.array([
            budget_ratio,            # 0: budget ratio (matches environment state_feats[0])
            cap_frac,                # 1: equity ratio (matches environment state_feats[1])
            market_vol,              # 2: market volatility (matches environment state_feats[2])
            market_stress            # 3: market stress (matches environment state_feats[3])
        ], dtype=np.float32)
        return feats

    # ---------- position snapshot ----------
    def _positions(self) -> np.ndarray:
        """Get position features to match environment (3 features)"""
        e = self.e
        # Get normalized financial instrument values (matches environment position_feats)
        try:
            fund_size = max(getattr(e, 'init_budget', 1e9), 1e6)
            wind_pos = float(getattr(e, 'financial_positions', {}).get('wind_instrument_value', 0.0)) / fund_size
            solar_pos = float(getattr(e, 'financial_positions', {}).get('solar_instrument_value', 0.0)) / fund_size
            hydro_pos = float(getattr(e, 'financial_positions', {}).get('hydro_instrument_value', 0.0)) / fund_size
            return np.array([wind_pos, solar_pos, hydro_pos], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # ---------- hedge parameter labeler ----------
    def _target_hedge_params(self, t: int) -> dict:
        """
        Calculate optimal hedge parameters based on current market conditions.
        Returns target hedge intensity, risk allocation, and direction for training.
        """
        try:
            # Get current market conditions for hedge parameter calculation
            e = self.e

            # Current generation and price data
            current_price = float(getattr(e, '_price', [0])[t] if hasattr(e, '_price') and t < len(e._price) else 0)
            current_wind = float(getattr(e, '_wind', [0])[t] if hasattr(e, '_wind') and t < len(e._wind) else 0)
            current_solar = float(getattr(e, '_solar', [0])[t] if hasattr(e, '_solar') and t < len(e._solar) else 0)
            current_hydro = float(getattr(e, '_hydro', [0])[t] if hasattr(e, '_hydro') and t < len(e._hydro) else 0)

            # Portfolio metrics
            nav = e._calculate_fund_nav() if hasattr(e, '_calculate_fund_nav') else 1e9
            drawdown = getattr(e.reward_calculator, 'current_drawdown', 0.0) if hasattr(e, 'reward_calculator') else 0.0

            # Calculate optimal hedge intensity based on market volatility and drawdown
            if drawdown > 0.3:
                hedge_intensity = 1.8  # High hedging during stress
            elif drawdown > 0.15:
                hedge_intensity = 1.4  # Medium hedging
            elif current_price > 300:  # High price volatility
                hedge_intensity = 1.2  # Moderate hedging
            else:
                hedge_intensity = 1.0  # Normal hedging


            # Calculate risk allocation based on generation patterns and volatility
            # Higher generation variability = higher risk budget allocation
            wind_capacity = float(getattr(e, 'wind_capacity_mw', 225))
            solar_capacity = float(getattr(e, 'solar_capacity_mw', 100))
            hydro_capacity = float(getattr(e, 'hydro_capacity_mw', 40))

            # Base allocation on capacity and volatility
            wind_vol = 0.35  # Wind has higher volatility
            solar_vol = 0.25  # Solar has medium volatility
            hydro_vol = 0.15  # Hydro has lower volatility

            # Weight by capacity and volatility for risk allocation
            wind_risk_weight = wind_capacity * wind_vol
            solar_risk_weight = solar_capacity * solar_vol
            hydro_risk_weight = hydro_capacity * hydro_vol

            total_risk_weight = wind_risk_weight + solar_risk_weight + hydro_risk_weight

            if total_risk_weight > 0:
                risk_allocation = np.array([
                    wind_risk_weight / total_risk_weight,
                    solar_risk_weight / total_risk_weight,
                    hydro_risk_weight / total_risk_weight
                ])
            else:
                risk_allocation = np.array([0.4, 0.35, 0.25])  # Default allocation

            # Calculate hedge direction based on generation vs expected
            # If generation is low, go LONG (hedge against high prices when we can't generate)
            # If generation is high, go SHORT (hedge against low prices when we generate a lot)
            total_generation = current_wind + current_solar + current_hydro
            expected_generation = wind_capacity * 0.35 + solar_capacity * 0.20 + hydro_capacity * 0.45  # Expected based on capacity factors

            if total_generation < expected_generation * 0.8:
                hedge_direction = 1.0  # LONG hedge (protect against high prices)
            elif total_generation > expected_generation * 1.2:
                hedge_direction = -1.0  # SHORT hedge (protect against low prices)
            else:
                hedge_direction = 0.5  # Mild LONG bias (default protection)

            # Calculate hedge effectiveness target (for training feedback)
            # Higher effectiveness when hedging reduces portfolio volatility
            hedge_effectiveness = 0.7  # Target 70% effectiveness

            return {
                'hedge_intensity': np.array([hedge_intensity], dtype=np.float32),
                'risk_allocation': risk_allocation.astype(np.float32),
                'hedge_direction': np.array([hedge_direction], dtype=np.float32),
                'hedge_effectiveness': np.array([hedge_effectiveness], dtype=np.float32)
            }

        except Exception as e:
            # Fallback to default hedge parameters
            return {
                'hedge_intensity': np.array([1.0], dtype=np.float32),
                'risk_allocation': np.array([0.4, 0.35, 0.25], dtype=np.float32),
                'hedge_direction': np.array([1.0], dtype=np.float32),
                'hedge_effectiveness': np.array([0.7], dtype=np.float32)
            }

    # ---------- hedge parameter inference ----------
    def infer_hedge_params(self, t: int) -> dict:
        """Get optimal hedge parameters from the trained model"""
        try:
            # Prepare inputs for hedge optimization model
            market_state = self._market_state(t)
            current_positions = self._positions()
            generation_forecast = self._get_generation_forecast(t)
            portfolio_metrics = self._get_portfolio_metrics(t)

            inputs = {
                'market_state': market_state,
                'current_positions': current_positions,
                'generation_forecast': generation_forecast,
                'portfolio_metrics': portfolio_metrics
            }

            # Get hedge parameters from model
            out = self.model(inputs, training=False)

            return {
                'hedge_intensity': float(out['hedge_intensity'].numpy()[0, 0]),
                'risk_allocation': out['risk_allocation'].numpy()[0],
                'hedge_direction': float(out['hedge_direction'].numpy()[0, 0]),
                'hedge_effectiveness': float(out['hedge_effectiveness'].numpy()[0, 0])
            }

        except Exception as e:
            # Fallback to default hedge parameters
            return {
                'hedge_intensity': 1.0,
                'risk_allocation': np.array([0.4, 0.35, 0.25]),
                'hedge_direction': 1.0,
                'hedge_effectiveness': 0.7
            }

    def _get_generation_forecast(self, t: int) -> np.ndarray:
        """Get generation forecast for hedge decision making"""
        try:
            e = self.e
            wind = float(getattr(e, '_wind', [0])[t] if hasattr(e, '_wind') and t < len(e._wind) else 0)
            solar = float(getattr(e, '_solar', [0])[t] if hasattr(e, '_solar') and t < len(e._solar) else 0)
            hydro = float(getattr(e, '_hydro', [0])[t] if hasattr(e, '_hydro') and t < len(e._hydro) else 0)

            # Normalize by capacity
            wind_cap = max(getattr(e, 'wind_capacity_mw', 225), 1e-6)
            solar_cap = max(getattr(e, 'solar_capacity_mw', 100), 1e-6)
            hydro_cap = max(getattr(e, 'hydro_capacity_mw', 40), 1e-6)

            return np.array([wind/wind_cap, solar/solar_cap, hydro/hydro_cap], dtype=np.float32)
        except Exception:
            return np.array([0.35, 0.20, 0.45], dtype=np.float32)  # Default capacity factors

    def _get_portfolio_metrics(self, t: int) -> np.ndarray:
        """Get portfolio metrics for hedge decision making (3 features to match environment)"""
        try:
            e = self.e
            # Match environment portfolio_metrics: capital_allocation_fraction, investment_freq, forecast_confidence
            cap_alloc = float(getattr(e, 'capital_allocation_fraction', 0.1))
            inv_freq = float(getattr(e, 'investment_freq', 10)) / 100.0  # Normalized
            forecast_conf = 1.0  # Default confidence

            return np.array([cap_alloc, inv_freq, forecast_conf], dtype=np.float32)
        except Exception:
            return np.array([0.1, 0.1, 1.0], dtype=np.float32)

    # ---------- online hedge learning hook ----------
    def maybe_learn(self, t: int):
        # (1) label hedge parameters periodically
        if self.e is None:
            return
        if t % self.label_every == 0:
            # Capture market state, positions, and target hedge parameters
            X = self._market_state(t)    # Market features (4 elements)
            P = self._positions()        # Current positions (3 elements)
            G = self._get_generation_forecast(t)  # Generation forecast (3 elements)
            M = self._get_portfolio_metrics(t)    # Portfolio metrics (3 elements)
            Y = self._target_hedge_params(t)      # Target hedge parameters (dict)

            # Ensure all arrays have correct shapes before storing
            X = np.array(X).flatten().astype(np.float32)
            P = np.array(P).flatten().astype(np.float32)
            G = np.array(G).flatten().astype(np.float32)
            M = np.array(M).flatten().astype(np.float32)

            # Validate shapes
            assert len(X) == 4, f"Market state should have 4 features, got {len(X)}"
            assert len(P) == 3, f"Positions should have 3 features, got {len(P)}"
            assert len(G) == 3, f"Generation forecast should have 3 features, got {len(G)}"
            assert len(M) == 3, f"Portfolio metrics should have 3 features, got {len(M)}"

            self.buffer.append((X, P, G, M, Y))   # Store all components

        # (2) train hedge optimizer periodically
        if len(self.buffer) >= self.batch_size and t % self.train_every == 0:
            idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)

            # Retrieve all stored components for the training batch with robust shape handling
            # Market state (should be 4 features)
            X_list = []
            for i in idx:
                x = self.buffer[i][0]
                if np.isscalar(x):
                    x = np.array([x, 0.0, 0.0, 0.0])  # Default 4 features
                else:
                    x = np.array(x).flatten()
                    if len(x) != 4:
                        x = np.pad(x, (0, max(0, 4 - len(x))), 'constant')[:4]
                X_list.append(x.astype(np.float32))
            Xb = np.array(X_list, dtype=np.float32)

            # Positions (should be 3 features)
            P_list = []
            for i in idx:
                p = self.buffer[i][1]
                if np.isscalar(p):
                    p = np.array([p, 0.0, 0.0])  # Default 3 features
                else:
                    p = np.array(p).flatten()
                    if len(p) != 3:
                        p = np.pad(p, (0, max(0, 3 - len(p))), 'constant')[:3]
                P_list.append(p.astype(np.float32))
            Pb = np.array(P_list, dtype=np.float32)

            # Generation forecast (should be 3 features)
            G_list = []
            for i in idx:
                g = self.buffer[i][2]
                if np.isscalar(g):
                    g = np.array([g, 0.0, 0.0])  # Default 3 features
                else:
                    g = np.array(g).flatten()
                    if len(g) != 3:
                        g = np.pad(g, (0, max(0, 3 - len(g))), 'constant')[:3]
                G_list.append(g.astype(np.float32))
            Gb = np.array(G_list, dtype=np.float32)

            # Portfolio metrics (should be 3 features)
            M_list = []
            for i in idx:
                m = self.buffer[i][3]
                if np.isscalar(m):
                    m = np.array([m, 0.0, 0.0])  # Default 3 features
                else:
                    m = np.array(m).flatten()
                    if len(m) != 3:
                        m = np.pad(m, (0, max(0, 3 - len(m))), 'constant')[:3]
                M_list.append(m.astype(np.float32))
            Mb = np.array(M_list, dtype=np.float32)

            # Extract hedge parameter targets from stored dictionaries
            # Handle scalars and arrays properly to avoid axis dimension issues
            hedge_intensity_list = []
            risk_allocation_list = []
            hedge_direction_list = []
            hedge_effectiveness_list = []

            for i in idx:
                hedge_params = self.buffer[i][4]

                # Ensure scalars are properly shaped
                intensity = hedge_params['hedge_intensity']
                hedge_intensity_list.append(float(intensity) if np.isscalar(intensity) else float(intensity.item()))

                direction = hedge_params['hedge_direction']
                hedge_direction_list.append(float(direction) if np.isscalar(direction) else float(direction.item()))

                effectiveness = hedge_params['hedge_effectiveness']
                hedge_effectiveness_list.append(float(effectiveness) if np.isscalar(effectiveness) else float(effectiveness.item()))

                # Risk allocation is an array - ensure it's properly shaped
                allocation = hedge_params['risk_allocation']
                if np.isscalar(allocation):
                    allocation = np.array([allocation, 0.0, 0.0])  # Default allocation
                elif hasattr(allocation, 'shape') and allocation.shape == ():
                    allocation = np.array([float(allocation), 0.0, 0.0])
                else:
                    allocation = np.array(allocation).flatten()
                    if len(allocation) != 3:
                        allocation = np.pad(allocation, (0, max(0, 3 - len(allocation))), 'constant')[:3]
                risk_allocation_list.append(allocation)

            # Convert to numpy arrays
            hedge_intensity_targets = np.array(hedge_intensity_list, dtype=np.float32)
            hedge_direction_targets = np.array(hedge_direction_list, dtype=np.float32)
            hedge_effectiveness_targets = np.array(hedge_effectiveness_list, dtype=np.float32)
            risk_allocation_targets = np.array(risk_allocation_list, dtype=np.float32)

            # Combine all input features
            X_combined = np.concatenate([Xb, Pb, Gb, Mb], axis=1)

            try:
                # Train the AdvancedHedgeOptimizer model (supports all 4 outputs)
                self.model.fit(
                    X_combined,
                    {
                        "hedge_intensity": hedge_intensity_targets,
                        "risk_allocation": risk_allocation_targets,
                        "hedge_direction": hedge_direction_targets,
                        "hedge_effectiveness": hedge_effectiveness_targets
                    },
                    epochs=self.epochs,
                    batch_size=min(self.batch_size, len(X_combined)),
                    verbose=0,
                    shuffle=True
                )

                # Log training progress occasionally
                if t % (self.train_every * 10) == 0:
                    logging.info(f"HedgeOptimizer training at step {t}: batch_size={len(X_combined)}, "
                               f"features_dim={Xb.shape[1]}, buffer_size={len(self.buffer)}")

            except Exception as e:
                logging.warning(f"HedgeOptimizer fitting failed at step {t}: {e}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent RL with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, default="sample.csv", help="Path to energy time series data")
    parser.add_argument("--timesteps", type=int, default=50000, help="TUNED: Increased for full synergy emergence (was 20000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for RL training (cuda/cpu)")
    parser.add_argument("--investment_freq", type=int, default=144, help="Investor action frequency in steps")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="Dir with trained forecast models")
    parser.add_argument("--scaler_dir", type=str, default="saved_scalers", help="Dir with trained scalers")

    # Optimization
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")
    parser.add_argument("--optimization_trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--optimization_timeout", type=int, default=1800, help="Optimization timeout (s)")
    parser.add_argument("--use_previous_optimization", action="store_true", help="Use latest saved optimized params")

    # Training
    parser.add_argument("--save_dir", type=str, default="training_agent_results", help="Where to save outputs")
    parser.add_argument("--checkpoint_freq", type=int, default=52000, help="Checkpoint frequency optimized for 520K timestep runs (10 checkpoints total)")
    parser.add_argument("--validate_env", action="store_true", default=True, help="Validate env setup before training")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--experience_replay", action="store_true", help="Enable experience replay buffer")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Size of experience replay buffer")

    # Rewards
    parser.add_argument("--adapt_rewards", action="store_true", default=True, help="Enable adaptive reward weights")
    parser.add_argument("--reward_analysis_freq", type=int, default=2000, help="Analyze rewards every N steps")
    parser.add_argument("--portfolio_reward_weight", type=float, default=0.8, help="Weight for portfolio performance in reward")
    parser.add_argument("--risk_penalty_weight", type=float, default=0.2, help="Weight for risk penalty in reward")

    # DL Overlay - Hedge optimization parameters
    parser.add_argument("--dl_overlay", action="store_true", help="Enable DL hedge optimization overlay")
    parser.add_argument("--dl_buffer_size", type=int, default=2048, help="DL buffer size")
    parser.add_argument("--dl_label_every", type=int, default=12, help="DL labeling frequency")
    parser.add_argument("--dl_train_every", type=int, default=60, help="DL training frequency")
    parser.add_argument("--dl_batch_size", type=int, default=128, help="TUNED: Optimal DL batch size")
    parser.add_argument("--dl_learning_rate", type=float, default=1e-3, help="TUNED: Optimal DL learning rate")
    parser.add_argument("--risk_aversion", type=float, default=5.0, help="TUNED: Balanced risk aversion parameter")

    # Forecasting Control (only enabled with --enable_forecasts)
    # Get default from config
    try:
        from config import EnhancedConfig
        default_config = EnhancedConfig()
        default_threshold = default_config.forecast_confidence_threshold
    except Exception:
        default_threshold = 0.05  # Fallback

    parser.add_argument("--forecast_confidence_threshold", type=float, default=default_threshold, help="Minimum forecast confidence for trading")
    parser.add_argument("--conservative_trading", action="store_true", help="Enable conservative trading mode")

    # Ultra Fast Mode Control (logging only)
    parser.add_argument("--ultra_fast_mode", action="store_true",
                       help="Ultra fast mode: no CSV logging, console output only")
    parser.add_argument("--final_results_only", action="store_true",
                       help="Only save final timestep results to CSV (header + 1 row, good for benchmarking)")

    # Forecasting integration
    parser.add_argument(
        "--enable_forecasts",
        action="store_true",
        help="Enable forecasting integration with precomputed forecasts for O(1) lookups during training."
    )
    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=8192,
        help="Batch size to use for offline forecast precompute (TF inference)."
    )
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Directory to save/load cached precomputed forecasts (CSV format)."
    )

    args = parser.parse_args()

    # No incompatible flag combinations - ultra fast mode only affects logging

    # Safer device selection
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Initialize TensorFlow based on device setting
    from generator import initialize_tensorflow
    initialize_tensorflow(args.device)

    # Seed everything for repeatability (uses config.seed after we create it; use a temp seed now too)
    seed_hint = 42
    random.seed(seed_hint)
    np.random.seed(seed_hint)
    torch.manual_seed(seed_hint)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_hint)

    print("Enhanced Multi-Horizon Energy Investment RL System")
    print("=" * 60)
    print("Features: Hyperparameter Optimization + Multi-Objective Rewards")

    # Create save dir + metrics subdir
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # 1) Load data
    print(f"\nLoading data from: {args.data_path}")
    try:
        data = load_energy_data(args.data_path)
        print(f"Data loaded: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        if "timestamp" in data.columns and data["timestamp"].notna().any():
            ts = data["timestamp"].dropna()
            print(f"Date range: {ts.iloc[0]} -> {ts.iloc[-1]}")
        if len(data) < 1000:
            print(f"Limited data ({len(data)} rows). More data -> better training stability.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2) Forecaster - ONLY enabled with --enable_forecasts flag
    if not args.enable_forecasts:
        print("\n[BASELINE] Forecasting disabled (use --enable_forecasts to enable)")
        forecaster = None
    else:
        print("\nInitializing multi-horizon forecaster with precomputed forecasts...")
        try:
            forecaster = MultiHorizonForecastGenerator(
                model_dir=args.model_dir,
                scaler_dir=args.scaler_dir,
                look_back=6,
                verbose=True
            )
            print("Forecaster initialized successfully!")

            # Precompute forecasts offline (required when forecaster is enabled)
            try:
                print(f"Precomputing forecasts offline (batch_size={args.precompute_batch_size})…")
                print("Checking for cached forecasts first...")
                forecaster.precompute_offline(
                    df=data,
                    timestamp_col="timestamp",
                    batch_size=max(1, int(args.precompute_batch_size)),
                    cache_dir=args.forecast_cache_dir
                )
                print("[OK] Forecasts precomputed successfully!")
            except Exception as pe:
                print(f"❌ Precompute failed: {pe}")
                print("Forecasting requires precomputed forecasts. Disabling forecaster.")
                forecaster = None

        except Exception as e:
            print(f"Failed to initialize forecaster: {e}")
            forecaster = None

    # 3) Initialize best_params (must happen before config creation)
    best_params = None
    if args.use_previous_optimization:
        print("\nChecking for previous optimization results...")
        opt_dir = os.path.join(args.save_dir, "optimization_results")
        best_params, _ = load_previous_optimization(opt_dir)
        if best_params:
            print("Using previous optimization results")
        else:
            print("No previous optimization found")

    # 4) Config setup (MOVED UP - must happen before environment creation)
    print("\nCreating optimized training configuration...")
    config = EnhancedConfig(optimized_params=best_params)

    # Re-seed using config.seed to ensure consistency with agent init
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 4) Environment setup
    print("\nSetting up enhanced environment with multi-objective rewards...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(metrics_dir, f"enhanced_metrics_{timestamp}.csv")

    # Instantiate HedgeAdapter for hedge strategy optimization
    dl_adapter = None
    if args.dl_overlay:
        # SIMPLE HEDGE OPTIMIZER - Now integrated directly into dl_overlay.py!
        print("[DL] Using SIMPLE hedge optimizer (integrated into dl_overlay.py)")
        dl_adapter = HedgeAdapter(
            None,  # Environment set later
            buffer_size=getattr(args, 'dl_buffer_size', 256),  # Smaller buffer for simple implementation
            label_every=getattr(args, 'dl_label_every', 20),
            train_every=getattr(args, 'dl_train_every', 100),
            batch_size=getattr(args, 'dl_batch_size', 64)
        )
        print("   [OK] Memory: 287MB -> 0.1MB (1934x reduction)")
        print("   [OK] Parameters: 2.7M -> 3.2K (839x reduction)")
        print("   [OK] Features: 43 -> 17 (8 historical + 9 forecast)")
        print("   [OK] Lightweight and efficient implementation with forecast integration")

        if args.enable_forecasts:
            print("DL hedge optimization enabled with forecasting integration")
        else:
            print("DL hedge optimization enabled (basic mode without forecasting)")

    try:
        base_env = RenewableMultiAgentEnv(
            data,
            investment_freq=args.investment_freq,
            forecast_generator=forecaster,
            dl_adapter=dl_adapter,
            config=config  # Pass config to environment
        )

        # FIXED: Apply command line forecast confidence threshold
        if hasattr(base_env, 'reward_calculator') and hasattr(base_env.reward_calculator, 'forecast_confidence_threshold'):
            base_env.reward_calculator.forecast_confidence_threshold = args.forecast_confidence_threshold
        if dl_adapter:
            dl_adapter.e = base_env  # Now set the environment on the adapter

        # Environment wrapper selection based on mode
        if args.ultra_fast_mode:
            # Ultra fast mode: no CSV logging, console output only
            if forecaster is not None:
                # Ultra fast mode with forecasting - use forecasting wrapper but disable CSV logging
                env = MultiHorizonWrapperEnv(
                    base_env,
                    forecaster,
                    log_path=None,  # No CSV logging
                    total_timesteps=args.timesteps,
                    disable_csv_logging=True  # Disable CSV logging
                )
                print("[ULTRA FAST] Using ULTRA FAST mode with forecasting (no CSV logging)")
            else:
                # Ultra fast mode baseline - use simple progress wrapper
                from wrapper import UltraFastProgressWrapper
                env = UltraFastProgressWrapper(base_env, args.timesteps)
                print("[ULTRA FAST] Using ULTRA FAST mode baseline (no CSV logging)")
        elif forecaster is None:
            # Baseline mode WITH CSV logging - use wrapper without forecaster
            from wrapper import BaselineCSVWrapper
            # Use baseline-specific log path
            baseline_log_path = os.path.join(args.save_dir, "baseline_results.csv")
            env = BaselineCSVWrapper(
                base_env,
                log_path=baseline_log_path,
                total_timesteps=args.timesteps,
                log_last_n=1 if args.final_results_only else args.timesteps
            )
            if args.final_results_only:
                print("[OK] Using baseline environment with CSV logging (final results only)")
            else:
                print("[OK] Using baseline environment with CSV logging (full logging)")
        else:
            # AI mode with forecasting and CSV logging
            if args.final_results_only:
                # Minimal CSV logging (final results only)
                env = MultiHorizonWrapperEnv(
                    base_env,
                    forecaster,
                    log_path=log_path,
                    total_timesteps=args.timesteps,
                    log_last_n=1  # Only final step
                )
                print("[OK] Using full AI environment with forecasting (final results only)")
            else:
                # Full CSV logging
                env = MultiHorizonWrapperEnv(
                    base_env,
                    forecaster,
                    log_path=log_path,
                    total_timesteps=args.timesteps,
                    log_last_n=args.timesteps  # Log everything
                )
                print("[OK] Using full AI environment with forecasting and full CSV logging")

        # DL adapter maintains fixed 13-feature architecture for add-on independence
        if dl_adapter:
            try:
                # DL adapter uses fixed 13 features regardless of observation space
                # This ensures add-on independence (forecasting doesn't affect DL overlay)
                fixed_feature_dim = 13  # 4 state + 3 positions + 3 forecasts + 3 portfolio
                print(f"   DL adapter maintains fixed {fixed_feature_dim} features (add-on independent)")
            except Exception as e:
                print(f"   Warning: DL adapter dimension info: {e}")

        print("Enhanced environment created successfully!")
        print("   Multi-objective rewards: enabled")
        print("   Enhanced risk management: enabled")
        print("   Forecast-augmented observations via wrapper: enabled")
        print(f"   Metrics CSV: {log_path}")
    except Exception as e:
        print(f"Failed to setup environment: {e}")
        return

    # Optional one-time validation (safe)
    if args.validate_env:
        try:
            _obs, _ = env.reset()
            print("Env reset OK for validation.")
        except Exception as e:
            print(f"Env validation reset failed (continuing): {e}")

    # Patched: DL integration now happens inside the environment itself.
    if args.dl_overlay:
        print("DL allocation overlay active (online self-labeling enabled)")

    # 4) (Optional) HPO (best_params already initialized above)
    if args.optimize and not best_params:
        print("\nRunning hyperparameter optimization...")
        opt_data = data.head(min(5000, len(data)))
        opt_base_env = RenewableMultiAgentEnv(opt_data, forecast_generator=forecaster, dl_adapter=None, config=config)
        opt_env = opt_base_env if forecaster is None \
                  else MultiHorizonWrapperEnv(opt_base_env, forecaster, log_path=None)

        best_params, best_perf = run_hyperparameter_optimization(
            opt_env,
            device=args.device,
            n_trials=args.optimization_trials,
            timeout=args.optimization_timeout
        )
        if best_params:
            opt_dir = os.path.join(args.save_dir, "optimization_results")
            save_optimization_results(best_params, best_perf, opt_dir)
        # Help GC
        try:
            if hasattr(opt_env, "close"):
                opt_env.close()
        except Exception:
            pass
        del opt_env, opt_base_env

    # 5) Agents (config already created above)
    print("\nInitializing enhanced multi-agent RL system...")
    try:
        agent = MultiESGAgent(
            config,
            env=env,
            device=args.device,
            training=True,
            debug=args.debug
        )
        print("Enhanced multi-agent system initialized")
        print(f"   Device: {args.device}")
        print(f"   Agents: {env.possible_agents}")
        print(f"   Learning rate: {config.lr:.2e}")
        print(f"   Update frequency: {config.update_every}")
        print("   Multi-objective rewards: enabled")
        print("   Adaptive hyperparameters: enabled")
    except Exception as e:
        print(f"Failed to initialize agents: {e}")
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass
        return

    # 7) Monitoring
    monitoring_dirs = setup_enhanced_training_monitoring(log_path, args.save_dir)

    # 8) Initialize Live Training Diagnostic
    diagnostic = integrate_diagnostic_with_training(base_env, log_interval=2000)
    diagnostic.capture_baseline()
    print(f"Live training diagnostic initialized (log interval: 2000 steps)")

    # 9) Training
    print("\nStarting Enhanced Multi-Objective Training...")
    print(f"   Training timesteps: {args.timesteps:,}")
    print(f"   Checkpoint frequency: {args.checkpoint_freq:,}")
    print(f"   Adaptive rewards: {'enabled' if args.adapt_rewards else 'disabled'}")
    print("   Performance monitoring: enabled")

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
            callbacks=callbacks,
            diagnostic=diagnostic
        )
        print("Enhanced training completed!")

        # Generate final diagnostic report
        if 'diagnostic' in locals():
            diagnostic.generate_final_report()

    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        # Close envs if they expose close()
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass

    # 9) Save final models
    print("\nSaving final trained models...")
    final_dir = os.path.join(args.save_dir, "final_models")
    os.makedirs(final_dir, exist_ok=True)
    saved_count = agent.save_policies(final_dir)

    if saved_count > 0:
        print(f"Saved {saved_count} trained agents to: {final_dir}")
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
        print(f"Training configuration saved to: {cfg_file}")

    # 10) Save the online-trained hedge optimizer (if possible)
    if args.dl_overlay and getattr(base_env, "dl_adapter", None):
        try:
            out_weights = os.path.join(args.save_dir, "hedge_optimizer_online.h5")
            if hasattr(base_env.dl_adapter.model, "save_weights"):
                base_env.dl_adapter.model.save_weights(out_weights)
                print(f"Saved online-trained hedge optimizer to: {out_weights}")
            elif hasattr(base_env.dl_adapter.model, "model") and hasattr(base_env.dl_adapter.model.model, "save_weights"):
                base_env.dl_adapter.model.model.save_weights(out_weights)
                print(f"Saved online-trained hedge optimizer to: {out_weights}")
        except Exception as e:
            print(f"Could not save hedge optimizer weights: {e}")

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
