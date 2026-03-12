# main.py

import argparse
import os
import sys
from datetime import datetime
import json
import random
import csv
import glob
from collections import deque
import logging
import traceback
import gc
from typing import Optional, Tuple, Dict, Any, List

# Determinism-critical environment variables must be set before any TensorFlow import.
# Parse CLI seed early (best effort) so child libs see a stable value at import time.
def _early_determinism_env_bootstrap() -> None:
    seed_str = "42"
    try:
        if "--seed" in sys.argv:
            idx = sys.argv.index("--seed")
            if idx + 1 < len(sys.argv):
                seed_str = str(int(sys.argv[idx + 1]))
    except Exception:
        seed_str = "42"
    os.environ.setdefault("PYTHONHASHSEED", seed_str)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Pin BLAS/math threadpools for reproducibility.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


_early_determinism_env_bootstrap()

import numpy as np
import pandas as pd
import torch  # for device availability check

# ---- Optional SB3 bits (callback base) ----
from stable_baselines3.common.callbacks import BaseCallback

# ===== CENTRALIZED LOGGING (from logger.py) =====
from logger import configure_logging, get_logger, TeeOutput, RewardLogger

# Configure logging (will be reconfigured with file handler in main())
configure_logging()
logger = get_logger(__name__)




# Import patched environment classes
from environment import RenewableMultiAgentEnv
from generator import MultiHorizonForecastGenerator
from utils import clear_tf_session, configure_tf_memory  # UNIFIED: Import from single source of truth
from config import normalize_price, ENHANCER_BASE_FEATURE_DIM  # UNIFIED: Import from single source of truth

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

def load_energy_data(csv_path: str, convert_to_raw_units: bool = True, config=None, mw_scale_overrides=None) -> pd.DataFrame:
    """
    Load energy time series data from CSV with optional unit conversion.
    Requires at least: wind, solar, hydro, price, load.
    Keeps extra cols like timestamp, risk, scenario, etc. if present.
    Casts numeric columns to float where possible and parses timestamp when present.

    Args:
        csv_path: Path to CSV file
        convert_to_raw_units: If True, converts capacity factors to absolute MW units
                             to match forecast model training data
        config: EnhancedConfig object with mw_conversion_scales
        mw_scale_overrides: Dict with CLI override values for MW scales
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
        logger.info("[INFO] Converting capacity factors to raw MW values for direct forecasting...")
        df = _convert_to_raw_mw_values(df, config=config, mw_scale_overrides=mw_scale_overrides)
        logger.info("[OK] Raw MW conversion completed - forecasts will work directly with these units")
    else:
        logger.info("[INFO] Data already in raw MW units - ready for direct forecasting")

    return df


def _is_capacity_factor_data(df: pd.DataFrame) -> bool:
    """Detect if data is in capacity factor format (0-1 range) vs absolute MW."""
    # Check if renewable generation looks like capacity factors (typically 0..1).
    # IMPORTANT: Do NOT use `load` here; many datasets have load already in MW
    # even when generation is expressed as capacity factors.
    renewable_cols = ['wind', 'solar', 'hydro']

    for col in renewable_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            max_val = float(np.nanmax(s.values)) if len(s) else 0.0
            if max_val > 2.0:  # If any renewable > 2, likely already in MW
                return False

    return True  # Renewables all <= 2.0 => likely capacity factors


def _enhancer_weights_filename(feature_dim: int) -> str:
    """Return the canonical enhancer checkpoint filename for the active Tier-2 contract."""
    fd = int(feature_dim)
    return f"dl_enhancer_{fd}d.h5"


def _capture_rolling_past_state(config) -> dict:
    """Persist rolling_past normalization state across episode checkpoints."""
    return {
        "rolling_past_price_state": getattr(config, "rolling_past_price_state", None),
        "rolling_past_wind_scale": getattr(config, "rolling_past_wind_scale", None),
        "rolling_past_solar_scale": getattr(config, "rolling_past_solar_scale", None),
        "rolling_past_hydro_scale": getattr(config, "rolling_past_hydro_scale", None),
        "rolling_past_load_scale": getattr(config, "rolling_past_load_scale", None),
    }


def _restore_rolling_past_state(config, state: Optional[dict]) -> bool:
    """Restore rolling_past normalization state from an episode checkpoint."""
    if not isinstance(state, dict):
        return False

    restored = False
    for key in (
        "rolling_past_price_state",
        "rolling_past_wind_scale",
        "rolling_past_solar_scale",
        "rolling_past_hydro_scale",
        "rolling_past_load_scale",
    ):
        if key in state:
            setattr(config, key, state.get(key))
            restored = True
    return restored


def _convert_to_raw_mw_values(df: pd.DataFrame, config=None, mw_scale_overrides=None) -> pd.DataFrame:
    """Convert capacity factors to raw MW values for direct forecasting.

    This eliminates normalization complexity and uses the exact training scale.

    Args:
        df: DataFrame with capacity factor data
        config: EnhancedConfig object with mw_conversion_scales
        mw_scale_overrides: Dict with CLI override values for MW scales
    """

    # Get MW conversion scales from config or use fallback defaults
    if config and hasattr(config, 'mw_conversion_scales'):
        capacity_mw = config.mw_conversion_scales.copy()
    else:
        # Fallback to original hardcoded values if no config
        capacity_mw = {
            'wind': 1103,    # From training scaler mean: 1103.4 MW
            'solar': 100,    # From training scaler mean: 61.5 MW (min 100 for stability)
            'hydro': 534,    # From training scaler mean: 534.1 MW
            'load': 2999,    # From training scaler mean: 2999.8 MW
        }

    # Apply CLI overrides if provided
    if mw_scale_overrides:
        for key, value in mw_scale_overrides.items():
            if value is not None and key in capacity_mw:
                capacity_mw[key] = float(value)
                logger.info(f"[OVERRIDE] Using CLI override for {key}: {value} MW")

    df_converted = df.copy()

    logger.info("[INFO] Converting to raw MW values (no normalization):")

    # Convert capacity factors to raw MW values.
    # SAFETY: convert per-column only if the column still looks like a capacity factor.
    def _looks_like_capacity_factor(series: pd.Series) -> bool:
        s = pd.to_numeric(series, errors="coerce")
        if len(s) == 0:
            return False
        max_val = float(np.nanmax(s.values))
        min_val = float(np.nanmin(s.values))
        # Allow small numerical negatives and occasional >1 spikes.
        return (max_val <= 2.0) and (min_val >= -0.1)

    for col, capacity in capacity_mw.items():
        if col in df_converted.columns:
            original_range = f"[{df[col].min():.3f}, {df[col].max():.3f}]"
            if _looks_like_capacity_factor(df[col]):
                df_converted[col] = df[col] * capacity
                new_range = f"[{df_converted[col].min():.1f}, {df_converted[col].max():.1f}] MW"
                logger.info(f"  {col}: {original_range} -> {new_range}")
            else:
                logger.info(f"  {col}: {original_range} (kept as-is; already looks like MW)")

    # Price: Already in $/MWh (no conversion needed)
    if 'price' in df_converted.columns:
        price_range = f"[{df_converted['price'].min():.1f}, {df_converted['price'].max():.1f}] $/MWh"
        logger.info(f"  price: {price_range} (no conversion)")

    logger.info("[OK] Raw MW conversion complete - ready for direct forecasting")

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
    logger.info("Starting Enhanced Hyperparameter Optimization")
    logger.info("=" * 55)

    if HyperparameterOptimizer is None:
        logger.warning("(Skipping HPO: HyperparameterOptimizer not available)")
        return None, None

    optimizer = HyperparameterOptimizer(
        env=env,
        device=device,
        n_trials=n_trials,
        timeout=timeout
    )

    logger.info("Configuration:")
    logger.info(f"   Number of trials: {n_trials}")
    logger.info(f"   Timeout: {timeout} seconds ({timeout/60:.1f} minutes)")
    logger.info(f"   Device: {device}")
    logger.info(f"   Environment: {env.__class__.__name__}")

    try:
        best_params, best_performance = optimizer.optimize()
        perf_value = _perf_to_float(best_performance)
        logger.info("\nOptimization Results:")
        logger.info(f"   Best heuristic score: {perf_value:.4f}")
        if isinstance(best_performance, dict):
            logger.info(f"   Raw performance: {best_performance}")
        logger.info("   Optimization completed successfully!")
        return best_params, best_performance
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        logger.warning("Falling back to default parameters")
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

    logger.info("Optimization results saved:")
    logger.info(f"   Parameters: {params_file}")
    logger.info(f"   Summary: {results_file}")

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
        logger.info(f"Loaded previous optimization results from: {latest_file}")
        return best_params, latest_file
    except Exception as e:
        logger.warning(f"Failed to load previous optimization: {e}")
        return None, None


def setup_enhanced_training_monitoring(log_path: str, save_dir: str) -> Dict[str, str]:
    monitoring_dirs = {
        'checkpoints': os.path.join(save_dir, 'checkpoints'),
        'metrics': os.path.join(save_dir, 'metrics'),
        'models': os.path.join(save_dir, 'models'),
        'logs': os.path.join(save_dir, 'logs')
    }
    for _, dir_path in monitoring_dirs.items():
        os.makedirs(dir_path, exist_ok=True)

    logger.info("Enhanced monitoring setup:")
    logger.info(f"   Metrics log: {log_path}")
    logger.info(f"   Checkpoints: {monitoring_dirs['checkpoints']}")
    logger.info(f"   Model saves: {monitoring_dirs['models']}")
    return monitoring_dirs


def load_checkpoint_state(checkpoint_dir: str) -> Dict:
    """Load training state from checkpoint directory"""
    state_file = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}

def print_portfolio_summary(env, step_count):
    """Print a clean portfolio summary"""
    try:
        # Access the underlying environment if wrapped
        base_env = env
        while hasattr(base_env, 'env') and not hasattr(base_env, '_calculate_fund_nav'):
            base_env = base_env.env

        # Get current environment state
        current_timestep = getattr(base_env, 't', 0)

        # Calculate NAV using the environment's method
        fund_nav_dkk = base_env._calculate_fund_nav()
        dkk_to_usd = 0.145  # Approximate conversion rate
        fund_nav_usd = fund_nav_dkk * dkk_to_usd / 1_000_000  # Convert to millions USD

        # Get financial positions
        cash_dkk = getattr(base_env, 'budget', 0)

        # Get MtM positions
        mtm_positions = getattr(base_env, 'cumulative_mtm_pnl', 0)
        mtm_usd = mtm_positions * dkk_to_usd / 1_000  # Convert to thousands USD

        # Get trading gains (from cumulative mark-to-market PnL)
        # FIX: Changed from sum(financial_positions.values()) to cumulative_mtm_pnl
        # Rationale: financial_positions is current position values (misleading), 
        # cumulative_mtm_pnl is actual cumulative trading performance
        trading_gains = getattr(base_env, 'cumulative_mtm_pnl', 0)
        trading_gains_usd = trading_gains * dkk_to_usd / 1_000  # Convert to thousands USD

        # Get operational gains (generation revenue)
        operational_gains = getattr(base_env, 'cumulative_generation_revenue', 0)
        operational_gains_usd = operational_gains * dkk_to_usd / 1_000  # Convert to thousands USD

        logger.info(f"\n[STATS] PORTFOLIO SUMMARY - Step {step_count:,} (Env: {current_timestep:,})")
        logger.info(f"   Portfolio Value: ${fund_nav_usd:.1f}M")
        logger.info(f"   Cash: {cash_dkk:,.0f} DKK")
        logger.info(f"   MtM Positions: ${mtm_usd:+.1f}k")
        logger.info(f"   Trading Gains: ${trading_gains_usd:+.1f}k")
        logger.info(f"   Operating Gains: ${operational_gains_usd:+.1f}k")

    except Exception as e:
        logger.error(f"\n[STATS] PORTFOLIO SUMMARY - Step {step_count:,}")
        logger.error(f"   Error calculating portfolio: {e}")

# Portfolio metrics are now logged at every timestep in the debug CSV (first columns)
# No separate checkpoint summary file needed - all data is in the per-episode CSV files

def enhanced_training_loop(agent, env, timesteps: int, checkpoint_freq: int, monitoring_dirs: Dict[str, str], callbacks=None, resume_from: str = None, dl_train_every: int = 100, preserve_agent_state: bool = False) -> int:
    """
    Train in intervals, but measure *actual* steps each time.
    Works with a meta-controller whose learn(total_timesteps=N) means "do N more steps".
    Enhanced with comprehensive memory monitoring and leak prevention.

    Args:
        agent: MultiESGAgent instance
        env: Environment (with optional enhancer_trainer)
        timesteps: Total timesteps to train
        checkpoint_freq: Frequency of checkpoints
        monitoring_dirs: Monitoring directories
        callbacks: Training callbacks
        resume_from: Resume from checkpoint path
        dl_train_every: Frequency for online enhancer updates inside the environment
        preserve_agent_state: If True, don't reset agent state even on first interval (for episode continuity)
    """
    logger.info("Starting Enhanced Training Loop with Memory Monitoring")
    logger.info(f"   Total timesteps: {timesteps:,}")
    logger.info(f"   Checkpoint frequency: {checkpoint_freq:,}")
    logger.info(f"   DL enhancer training: every {dl_train_every:,} steps (when buffer sufficient)")

    # Initialize memory monitoring
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"   Initial memory usage: {initial_memory:.1f}MB")
        memory_history = []
    except Exception:
        initial_memory = 0
        memory_history = []

    # Handle resume from checkpoint
    total_trained = 0
    checkpoint_count = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f"\n[CYCLE] RESUMING from checkpoint: {resume_from}")

        # Load agent policies
        loaded_count = agent.load_policies(resume_from)
        if loaded_count > 0:
            logger.info(f"[OK] Loaded {loaded_count} agent policies")

            # Load training state
            checkpoint_state = load_checkpoint_state(resume_from)
            if checkpoint_state:
                total_trained = checkpoint_state.get('total_trained', 0)
                checkpoint_count = checkpoint_state.get('checkpoint_count', 0)
                logger.info(f"[OK] Resuming from step {total_trained:,} (checkpoint {checkpoint_count})")

                # Load DL Enhancer weights when resuming
                if hasattr(env, 'enhancer_adapter') and env.enhancer_adapter is not None:
                    clear_tf_session()
                    fd = env.enhancer_adapter.feature_dim
                    enh_path = os.path.join(resume_from, _enhancer_weights_filename(fd))
                    if os.path.exists(enh_path):
                        try:
                            env.enhancer_adapter.model.load_weights(enh_path)
                            logger.info(f"[OK] Loaded DL enhancer weights from {enh_path}")
                        except Exception as e:
                            logger.warning(f"[WARN] Could not load enhancer weights: {e}")
            else:
                logger.warning("[WARN] No training state found, starting from step 0")
        else:
            logger.error("[ERROR] Failed to load policies, starting fresh")
            resume_from = None
    elif resume_from:
        logger.error(f"[ERROR] Resume checkpoint not found: {resume_from}")
        logger.info("Starting fresh training...")
        resume_from = None

    try:
        while total_trained < timesteps:
            remaining = timesteps - total_trained
            interval = min(checkpoint_freq, remaining)

            logger.info(f"\nTraining interval {checkpoint_count + 1}")
            logger.info(f"   Steps: {total_trained:,} -> {total_trained + interval:,}")
            logger.info(f"   Progress: {total_trained/timesteps*100:.1f}%")

            start_time = datetime.now()
            start_steps = getattr(agent, "total_steps", 0)

            # PHASE 5.6 FIX: Don't reset agent state between checkpoint intervals
            # Only reset on the first interval (checkpoint_count == 0)
            # For subsequent intervals, keep agent state to continue training
            # CRITICAL FIX: If preserve_agent_state=True (episode continuity), NEVER reset
            reset_agent_state = (checkpoint_count == 0) and not preserve_agent_state

            agent.learn(total_timesteps=interval, callbacks=callbacks, reset_num_timesteps=reset_agent_state)

            end_time = datetime.now()
            end_steps = getattr(agent, "total_steps", start_steps)
            trained_now = max(0, int(end_steps) - int(start_steps))

            # Enhancer updates are performed online inside env.step() so feature/target
            # alignment matches the realized investor MTM target.

            # Print portfolio summary at each checkpoint
            print_portfolio_summary(agent.env, end_steps)

            # Update totals using the *actual* number of steps collected.
            total_trained += trained_now
            checkpoint_count += 1

            training_time = (end_time - start_time).total_seconds()
            sps = (trained_now / training_time) if training_time > 0 else 0.0

            # Memory monitoring and leak detection
            try:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_history.append(current_memory)
                memory_growth = current_memory - initial_memory

                # Calculate memory growth rate
                if len(memory_history) > 1:
                    recent_growth = memory_history[-1] - memory_history[-2] if len(memory_history) > 1 else 0
                    avg_growth = memory_growth / checkpoint_count if checkpoint_count > 0 else 0
                else:
                    recent_growth = 0
                    avg_growth = 0

                logger.info(f"   Training time: {training_time:.1f}s ({sps:.1f} steps/s)")
                logger.info(f"   Memory: {current_memory:.1f}MB (+{memory_growth:.1f}MB total, +{recent_growth:.1f}MB this interval)")
                logger.info(f"   Actual steps collected this interval: {trained_now:,} (agent.total_steps = {end_steps:,})")
                
                # Log system activity proof: show agents are learning and making decisions
                if hasattr(agent, '_training_metrics'):
                    metrics = agent._training_metrics
                    successful_steps = metrics.get('successful_steps', 0)
                    policy_errors = metrics.get('policy_errors', 0)
                    logger.info(f"   [SYSTEM_PROOF] Training activity: {successful_steps:,} steps collected, "
                               f"{policy_errors} policy errors, {metrics.get('memory_cleanups', 0)} memory cleanups")
                
                # Log NAV change to show trading impact
                if hasattr(env, '_calculate_fund_nav'):
                    try:
                        current_nav = env._calculate_fund_nav()
                        if hasattr(env, '_initial_nav_from_reset') and env._initial_nav_from_reset > 0:
                            nav_change = current_nav - env._initial_nav_from_reset
                            nav_change_pct = (nav_change / env._initial_nav_from_reset) * 100
                            logger.info(f"   [SYSTEM_PROOF] NAV: {current_nav:,.0f} DKK ({nav_change_pct:+.2f}% from start)")
                    except Exception:
                        pass

                # Trigger aggressive cleanup if memory growth is concerning
                if memory_growth > 2000:  # More than 2GB growth
                    logger.warning(f"[WARN]  High memory usage detected ({current_memory:.1f}MB), triggering cleanup...")
                    try:
                        # Force cleanup on environment if it has DL enhancer
                        if hasattr(env, 'enhancer_adapter') and env.enhancer_adapter is not None:
                            pass  # Enhancer has no explicit cleanup

                        # Force cleanup on wrapper
                        if hasattr(env, '_cleanup_memory_enhanced'):
                            env._cleanup_memory_enhanced(force=True)

                        # Force cleanup on agent
                        if hasattr(agent, 'memory_tracker'):
                            agent.memory_tracker.cleanup('heavy')

                        # gc already imported at module level
                        for _ in range(3):
                            gc.collect()

                        # Check memory after cleanup
                        after_cleanup = process.memory_info().rss / 1024 / 1024
                        freed = current_memory - after_cleanup
                        logger.info(f"   Cleanup freed {freed:.1f}MB, new usage: {after_cleanup:.1f}MB")

                    except Exception as cleanup_error:
                        logger.warning(f"   Cleanup failed: {cleanup_error}")

            except Exception:
                logger.info(f"   Training time: {training_time:.1f}s ({sps:.1f} steps/s)")
                logger.info(f"   Actual steps collected this interval: {trained_now:,} (agent.total_steps = {end_steps:,})")

            if trained_now == 0:
                logger.warning("No steps were collected in this interval. "
                      "Check that the meta-controller's learn() uses a relative budget or increase n_steps.")
                # Avoid infinite loop; break so we can at least save progress.
                break

            # Save checkpoint only if we actually progressed and still have more to do
            if total_trained < timesteps and trained_now > 0:
                checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"checkpoint_{total_trained}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.info(f"Saving checkpoint at {total_trained:,} steps...")
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
                logger.info(f"   Checkpoint saved: {saved_count} policies")
                # Note: Portfolio metrics are logged at every timestep in debug CSV - no separate checkpoint summary needed

            # Opportunistic metrics flush to avoid buffer loss if the process stops unexpectedly
            try:
                if hasattr(env, "_flush_log_buffer"):
                    env._flush_log_buffer()
            except Exception:
                pass

        logger.info("\nEnhanced training completed!")
        logger.info(f"   Total steps (budget accumulation): {total_trained:,}")
        logger.info(f"   Agent-reported total steps: {getattr(agent, 'total_steps', 'unknown')}")
        logger.info(f"   Checkpoints created: {checkpoint_count}")
        return total_trained

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info(f"   Progress: {total_trained:,}/{timesteps:,} ({total_trained/timesteps*100:.1f}%)")
        emergency_dir = os.path.join(monitoring_dirs['checkpoints'], f"emergency_{total_trained}")
        os.makedirs(emergency_dir, exist_ok=True)
        agent.save_policies(emergency_dir)
        logger.info(f"Emergency checkpoint saved to: {emergency_dir}")
        return total_trained

    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        raise


def analyze_training_performance(env, log_path: str, monitoring_dirs: Dict[str, str]) -> None:
    logger.info("\nTraining Performance Analysis")
    logger.info("=" * 40)
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
                logger.info("Training Metrics Summary:")
                logger.info(f"   Rows logged (wrapper): {len(metrics):,}")
                logger.info(f"   Early performance: {early:.4f}")
                logger.info(f"   Late performance:  {late:.4f}")
                logger.info(f"   Improvement:       {improvement:+.4f} ({pct:+.1f}%)")

        ckpt_dir = monitoring_dirs.get('checkpoints', '')
        if os.path.exists(ckpt_dir):
            ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith('checkpoint_')]
            logger.info(f"   Checkpoints available: {len(ckpts)}")

    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")


# =====================================================================
# Deep Learning Hedge Optimization Overlay (ONLINE)
# =====================================================================

# =====================================================================
# Memory Management Utilities
# =====================================================================

def get_memory_status():
    """Get current memory status for monitoring"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except Exception:
        return 0.0

def force_memory_cleanup(context="general"):
    """Force comprehensive memory cleanup"""
    try:
        # gc and os already imported at module level

        logger.info(f"   [SWEEP] Force cleanup ({context})...")

        # Multiple garbage collection passes
        collected_total = 0
        for i in range(3):
            collected = gc.collect()
            collected_total += collected

        # Clear TensorFlow sessions
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info(f"   [OK] Cleanup freed {collected_total} objects")
        return collected_total

    except Exception as e:
        logger.warning(f"   [WARN] Cleanup failed: {e}")
        return 0

# =====================================================================
# Episode Training Functions
# =====================================================================

def get_episode_info(episode_num):
    """Get episode information based on episode number (0-19)"""
    year = 2015 + (episode_num // 2)
    half = "H1" if episode_num % 2 == 0 else "H2"

    if half == "H1":
        start_date = f"Jan 1, {year}"
        end_date = f"Jun 30, {year}"
    else:
        start_date = f"Jul 1, {year}"
        end_date = f"Dec 31, {year}"

    return {
        "year": year,
        "half": half,
        "start_date": start_date,
        "end_date": end_date,
        "description": f"{half} {year}"
    }

def load_episode_data(episode_data_dir, episode_num, config=None, mw_scale_overrides=None):
    """Load data for a specific episode with configurable MW conversion"""
    import os
    import pandas as pd

    # Try different possible filename patterns
    possible_files = [
        f"scenario_{episode_num:03d}.csv",  # scenario_000.csv, scenario_001.csv, etc.
        f"episode_{episode_num}.csv",
        f"episode{episode_num}.csv",
        f"{episode_num}.csv",
        f"data_{episode_num}.csv"
    ]

    for filename in possible_files:
        filepath = os.path.join(episode_data_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"   [DATA] Loading episode data: {filepath}")

            # Use load_energy_data for consistent processing including MW conversion
            data = load_energy_data(filepath, convert_to_raw_units=True, config=config, mw_scale_overrides=mw_scale_overrides)

            logger.info(f"   [STATS] Episode {episode_num} data: {len(data)} rows")
            return data

    # If no file found, list available files
    if os.path.exists(episode_data_dir):
        available_files = [f for f in os.listdir(episode_data_dir) if f.endswith('.csv')]
        logger.error(f"   [ERROR] Episode {episode_num} data not found")
        logger.info(f"   [FILES] Available files in {episode_data_dir}: {available_files}")
    else:
        logger.error(f"   [ERROR] Episode data directory not found: {episode_data_dir}")

    raise FileNotFoundError(f"Episode {episode_num} data not found in {episode_data_dir}")


def _robust_p95_scale(values, min_scale: float = 0.1) -> float:
    """Robust positive scale estimate for causal rolling-past bootstrap."""
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            p95_val = float(np.nanpercentile(arr, 95))
            if np.isfinite(p95_val) and p95_val > 0.0:
                return max(p95_val, float(min_scale))
    except Exception:
        pass
    return max(float(min_scale), 1.0)


def _running_moments_from_values(values) -> Optional[Dict[str, float]]:
    """Convert a historical series into the persistent online-state format."""
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        count = float(arr.size)
        mean = float(np.mean(arr))
        var = float(np.var(arr, ddof=0))
        return {
            "count": count,
            "mean": mean,
            "m2": max(var * count, 0.0),
        }
    except Exception:
        return None


def _resolve_rolling_past_history_path(history_dir: str, episode_num: int) -> Optional[str]:
    """Return the causal history file for a MARL training episode."""
    if not history_dir:
        return None
    history_path = os.path.join(history_dir, f"history_{int(episode_num):03d}.csv")
    return history_path if os.path.exists(history_path) else None


def _iter_rolling_past_history_paths(history_dir: str, episode_num: int):
    """
    Yield causal history files from newest to oldest for one MARL episode.

    Example:
    - episode 0 -> history_000 only (2014 bootstrap year)
    - episode 2 -> history_002, history_001, history_000
    """
    ep = int(max(episode_num, 0))
    while ep >= 0:
        path = _resolve_rolling_past_history_path(history_dir, ep)
        if path is None:
            break
        yield path
        ep -= 1


def _select_rolling_past_history_tail(history_df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Tighten rolling_past bootstrap to the trailing causal history window only.

    By default we use the last 365 days from the accumulated causal history
    window. If the assembled history is shorter, or the configured tail length
    is invalid, we fall back to the full available history.
    """
    try:
        tail_days = int(getattr(config, "rolling_past_history_tail_days", 365) or 365)
    except Exception:
        tail_days = 365
    try:
        rows_per_day = int(getattr(config, "rolling_past_history_rows_per_day", 144) or 144)
    except Exception:
        rows_per_day = 144

    tail_days = max(tail_days, 1)
    rows_per_day = max(rows_per_day, 1)
    tail_rows = max(tail_days * rows_per_day, rows_per_day)
    if len(history_df) <= tail_rows:
        return history_df.copy()
    return history_df.iloc[-tail_rows:].copy()


def _load_rolling_past_bootstrap_history(config, history_csv_path: str, mw_scale_overrides=None) -> pd.DataFrame:
    """
    Assemble the causal bootstrap history for rolling_past.

    The current episode should see up to one full previous year of history.
    Because individual history files may only cover one prior MARL episode, we
    walk backwards through the causal history folder and prepend older files
    until the desired trailing window can be formed.
    """
    history_dir = os.path.dirname(history_csv_path)
    history_name = os.path.basename(history_csv_path)
    stem = os.path.splitext(history_name)[0]
    try:
        episode_num = int(stem.split("_")[-1])
    except Exception:
        episode_num = 0

    try:
        tail_days = int(getattr(config, "rolling_past_history_tail_days", 365) or 365)
    except Exception:
        tail_days = 365
    try:
        rows_per_day = int(getattr(config, "rolling_past_history_rows_per_day", 144) or 144)
    except Exception:
        rows_per_day = 144
    target_rows = max(1, tail_days) * max(1, rows_per_day)

    frames = []
    rows_accum = 0
    for path in _iter_rolling_past_history_paths(history_dir, episode_num):
        df = load_energy_data(
            path,
            convert_to_raw_units=True,
            config=config,
            mw_scale_overrides=mw_scale_overrides,
        )
        frames.insert(0, df)
        rows_accum += int(len(df))
        if rows_accum >= target_rows:
            break

    if not frames:
        raise RuntimeError(f"No causal rolling_past history found for bootstrap: {history_csv_path}")
    return pd.concat(frames, axis=0, ignore_index=True)


def _prime_rolling_past_from_history(config, history_csv_path: str, mw_scale_overrides=None) -> Dict[str, float]:
    """
    Seed rolling_past normalization from a causal history file.

    Episode 0 can use 2014 prehistory; later episodes can use the previous MARL
    episode. We intentionally seed from the trailing recent window only, not the
    full file, to keep rolling_past locally adaptive while remaining causal.
    """
    history_df = _load_rolling_past_bootstrap_history(
        config,
        history_csv_path,
        mw_scale_overrides=mw_scale_overrides,
    )
    full_rows = int(len(history_df))
    history_df = _select_rolling_past_history_tail(history_df, config)
    if "price" not in history_df.columns:
        raise RuntimeError(f"History bootstrap missing price column: {history_csv_path}")

    price_state = _running_moments_from_values(history_df["price"].to_numpy())
    if price_state is None:
        raise RuntimeError(f"History bootstrap could not build price state: {history_csv_path}")

    config.rolling_past_price_state = price_state
    config.rolling_past_wind_scale = _robust_p95_scale(history_df["wind"].to_numpy(), min_scale=0.1)
    config.rolling_past_solar_scale = _robust_p95_scale(history_df["solar"].to_numpy(), min_scale=0.1)
    config.rolling_past_hydro_scale = _robust_p95_scale(history_df["hydro"].to_numpy(), min_scale=0.1)
    config.rolling_past_load_scale = _robust_p95_scale(history_df["load"].to_numpy(), min_scale=0.1)

    return {
        "rows_total": float(full_rows),
        "rows_used": float(len(history_df)),
        "price_mean": float(price_state["mean"]),
        "price_std": float(max(np.sqrt(max(price_state["m2"], 0.0) / max(price_state["count"], 1.0)), 0.0)),
        "wind_scale": float(config.rolling_past_wind_scale),
        "solar_scale": float(config.rolling_past_solar_scale),
        "hydro_scale": float(config.rolling_past_hydro_scale),
        "load_scale": float(config.rolling_past_load_scale),
    }

def cooling_period(minutes, episode_num):
    """Thermal cooling period between episodes with enhanced memory monitoring"""
    import time
    import psutil
    import sys
    # gc already imported at module level

    logger.info(f"\n[TEMP] Cooling period after Episode {episode_num}: {minutes} minutes")
    logger.info("   Allowing CPU to cool down and system to stabilize...")

    # Initial memory reading
    try:
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"   [STATS] Initial cooling memory: {initial_memory:.1f}MB")
    except Exception:
        initial_memory = 0

    # Monitor system during cooling
    start_time = time.time()
    while time.time() - start_time < minutes * 60:
        remaining = minutes * 60 - (time.time() - start_time)
        remaining_min = remaining / 60

        # Check system resources with enhanced memory monitoring
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            process_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            # logging.Logger does not support print-style `end=...`; use stdout for a single-line progress indicator.
            sys.stdout.write(
                f"\r   [TIME] {remaining_min:.1f}min | CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Process: {process_memory:.1f}MB"
            )
            sys.stdout.flush()

            # Perform gentle cleanup during cooling if memory is high
            if process_memory > 4000:  # More than 4GB
                gc.collect()

        except Exception:
            sys.stdout.write(f"\r   [TIME] {remaining_min:.1f}min remaining")
            sys.stdout.flush()

        time.sleep(30)  # Check every 30 seconds

    # Final memory reading
    try:
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_change = final_memory - initial_memory
        change_sign = "+" if memory_change > 0 else ""
        logger.info(f"\n   [OK] Cooling complete | Final memory: {final_memory:.1f}MB ({change_sign}{memory_change:.1f}MB)")
    except Exception:
        logger.info(f"\n   [OK] Cooling period complete")

def run_episode_training(agent, base_env, env, args, monitoring_dirs, config, mw_scale_overrides, forecaster=None):
    """Run training across multiple 6-month episodes"""
    import os
    import time
    from datetime import datetime

    logger.info(f"\n[GOAL] Episode Training Configuration:")
    logger.info(f"   Episodes: {args.start_episode} -> {args.end_episode}")
    logger.info(f"   Data directory: {args.episode_data_dir}")
    logger.info(f"   Cooling period: {args.cooling_period} minutes")

    # ------------------------------------------------------------------
    # GLOBAL NORMALIZATION (compute once per run for stable cross-episode learning)
    # ------------------------------------------------------------------
    try:
        # Enable for episode training to avoid distribution shifts at episode boundaries.
        # Respect explicit mode selection:
        # - use_global_normalization=False  -> rolling_past (no global precompute)
        # - use_global_normalization=True   -> global normalization with precomputed/full-dataset stats
        need_global_norm = bool(getattr(args, "episode_training", False))
        use_global_norm_requested = bool(getattr(config, "use_global_normalization", False))
        has_stats = all(
            hasattr(config, k) for k in ("global_price_mean", "global_price_std", "global_wind_scale", "global_solar_scale", "global_hydro_scale", "global_load_scale")
        )
        if need_global_norm and (not use_global_norm_requested):
            logger.info("[GLOBAL_NORM] Using rolling_past mode (global normalization disabled).")
        elif need_global_norm and (not has_stats):
            scenario_paths = sorted(glob.glob(os.path.join(args.episode_data_dir, "scenario_*.csv")))
            if len(scenario_paths) >= 2:
                prices = []
                wind = []
                solar = []
                hydro = []
                load = []
                for sp in scenario_paths:
                    df = pd.read_csv(sp)
                    if "price" in df.columns:
                        prices.append(df["price"].astype(float).to_numpy())
                    if "wind" in df.columns:
                        wind.append(df["wind"].astype(float).to_numpy())
                    if "solar" in df.columns:
                        solar.append(df["solar"].astype(float).to_numpy())
                    if "hydro" in df.columns:
                        hydro.append(df["hydro"].astype(float).to_numpy())
                    if "load" in df.columns:
                        load.append(df["load"].astype(float).to_numpy())

                import numpy as np

                if prices:
                    price_all = np.concatenate(prices)
                    config.global_price_mean = float(np.nanmean(price_all))
                    config.global_price_std = float(max(np.nanstd(price_all), 1e-6))

                def p95(x):
                    if not x:
                        return None
                    arr = np.concatenate(x)
                    return float(max(np.nanpercentile(arr, 95), 1e-6))

                config.global_wind_scale = p95(wind)
                config.global_solar_scale = p95(solar)
                config.global_hydro_scale = p95(hydro)
                config.global_load_scale = p95(load)

                config.use_global_normalization = True
                logger.info(
                    "[GLOBAL_NORM] Enabled global normalization for episode training: "
                    f"price_mean={config.global_price_mean:.3f} price_std={config.global_price_std:.3f} "
                    f"| p95 scales: wind={config.global_wind_scale} solar={config.global_solar_scale} "
                    f"hydro={config.global_hydro_scale} load={config.global_load_scale}"
                )
            else:
                logger.warning("[GLOBAL_NORM] Not enough scenario_*.csv files found to compute global normalization; leaving disabled.")
    except Exception as e:
        logger.error(f"[GLOBAL_NORM_FATAL] Failed to compute global normalization stats: {e}")
        raise RuntimeError("[GLOBAL_NORM_FATAL] Failed to compute global normalization stats") from e

    # Handle episode restart
    if args.resume_episode is not None:
        logger.info(f"   [CYCLE] Resume mode: Starting from Episode {args.resume_episode}")

        # Load checkpoint from specified episode
        episode_checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"episode_{args.resume_episode}")
        if os.path.exists(episode_checkpoint_dir):
            try:
                loaded_count = agent.load_policies(episode_checkpoint_dir)
                if loaded_count > 0:
                    logger.info(f"   [OK] Loaded model from Episode {args.resume_episode} checkpoint")

                    # [ALERT] CRITICAL: Load agent training state to preserve learning continuity
                    agent_state_path = os.path.join(episode_checkpoint_dir, "agent_training_state.json")
                    if os.path.exists(agent_state_path):
                        try:
                            with open(agent_state_path, 'r') as f:
                                agent_state = json.load(f)

                            # Restore agent's training progress
                            if hasattr(agent, 'total_steps'):
                                agent.total_steps = agent_state.get('total_steps', 0)
                                logger.info(f"   [OK] Restored agent.total_steps: {agent.total_steps:,}")

                            # Restore training metrics if available
                            if hasattr(agent, '_training_metrics') and 'training_metrics' in agent_state:
                                agent._training_metrics.update(agent_state['training_metrics'])
                                logger.info(f"   [OK] Restored training metrics")

                            # Restore error tracking
                            if hasattr(agent, '_consecutive_errors'):
                                agent._consecutive_errors = agent_state.get('consecutive_errors', 0)

                            if _restore_rolling_past_state(config, agent_state.get('rolling_past_state')):
                                logger.info("   [OK] Restored rolling_past normalization state")

                            # Store for later use in episode loop
                            restored_total_trained = agent_state.get('total_trained_all_episodes', 0)
                            logger.info(f"   [OK] Restored cumulative training: {restored_total_trained:,} steps")

                            # PHASE 3 PATCH C: Global Training Budget Synchronization
                            # Ensure agent's internal step counter is synchronized with external cumulative counter
                            # This prevents learning rate decay or exploration schedule desync across episodes
                            if hasattr(agent, 'total_steps') and restored_total_trained > 0:
                                # Verify consistency: agent.total_steps should match the cumulative budget
                                if agent.total_steps != restored_total_trained:
                                    logger.info(f"   [PATCH C] Synchronizing agent.total_steps: {agent.total_steps:,} -> {restored_total_trained:,}")
                                    agent.total_steps = restored_total_trained
                                else:
                                    logger.debug(f"   [PATCH C] Agent budget already synchronized: {agent.total_steps:,} steps")

                        except Exception as state_error:
                            logger.warning(f"   [WARN] Could not load agent training state: {state_error}")
                            logger.warning(f"   [WARN] Agent will start with total_steps=0 (learning continuity may be affected)")
                    else:
                        logger.warning(f"   [WARN] No agent training state found - this checkpoint may be from before the fix")
                        logger.warning(f"   [WARN] Agent will start with total_steps=0 (learning continuity may be affected)")

                    # Load DL Enhancer weights from episode checkpoint
                    if args.forecast_baseline_enable and hasattr(base_env, 'enhancer_adapter') and base_env.enhancer_adapter is not None:
                        clear_tf_session()
                        fd = base_env.enhancer_adapter.feature_dim
                        enh_path = os.path.join(episode_checkpoint_dir, _enhancer_weights_filename(fd))
                        if os.path.exists(enh_path):
                            try:
                                base_env.enhancer_adapter.model.load_weights(enh_path)
                                logger.info(f"   [OK] Loaded DL enhancer weights from Episode {args.resume_episode}: {enh_path}")
                            except Exception as e:
                                logger.warning(f"   [WARN] Could not load enhancer weights: {e}")

                    # Adjust start episode to resume from next episode
                    args.start_episode = args.resume_episode + 1
                else:
                    logger.warning(f"   [WARN] Failed to load Episode {args.resume_episode} checkpoint, starting fresh")
            except Exception as e:
                logger.warning(f"   [WARN] Error loading Episode {args.resume_episode} checkpoint: {e}")
                logger.info(f"   Starting fresh from Episode {args.start_episode}")
        else:
            logger.error(f"   [ERROR] Episode {args.resume_episode} checkpoint not found: {episode_checkpoint_dir}")
            logger.info(f"   Starting fresh from Episode {args.start_episode}")

    # Initialize episode tracking (may be overridden by resume)
    total_trained_all_episodes = locals().get('restored_total_trained', 0)
    successful_episodes = 0

    # CRITICAL FIX: Track if we resumed from a checkpoint
    # When resuming from episode N, we've already loaded policies from episode N
    # So the first episode (N+1) should load from episode N, not start fresh
    resumed_from_checkpoint = args.resume_episode is not None

    for episode_num in range(args.start_episode, args.end_episode + 1):
        # CRITICAL FIX: More aggressive cleanup for episodes 10+ to prevent OOM
        if episode_num >= 10:
            logger.warning(f"   [ALERT] Episode {episode_num} - EXTRA AGGRESSIVE CLEANUP")
            # Clear forecast generator caches
            if forecaster is not None and hasattr(forecaster, '_cleanup_memory'):
                forecaster._cleanup_memory()
            # Additional GC passes
            for i in range(10):
                gc.collect()
            # Clear any environment caches
            if hasattr(base_env, '_forecast_history'):
                if hasattr(base_env._forecast_history, 'clear'):
                    base_env._forecast_history.clear()
            if hasattr(base_env, '_z_score_history'):
                base_env._z_score_history.clear()

        episode_info = get_episode_info(episode_num)

        logger.info(f"\n" + "="*70)
        logger.info(f"[LAUNCH] EPISODE {episode_num}: {episode_info['description']}")
        logger.info(f"   Period: {episode_info['start_date']} -> {episode_info['end_date']}")
        logger.info(f"   Progress: Episode {episode_num + 1}/{args.end_episode + 1}")
        logger.info("="*70)

        # [ALERT] CRITICAL: AGGRESSIVE PRE-EPISODE MEMORY MANAGEMENT
        logger.warning(f"   [ALERT] AGGRESSIVE MEMORY CLEANUP BEFORE EPISODE {episode_num}")
        try:
            import psutil

            # Get initial memory
            initial_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.info(f"   [STATS] Pre-episode memory: {initial_mem:.1f}MB")

            # FIX: NUCLEAR CLEANUP - Aggressive memory management between episodes
            # REASON: Multi-agent RL with replay buffers, forecast caches, and TensorFlow graphs
            # can accumulate significant memory across episodes. This cleanup is necessary because:
            # 1. SB3 replay buffers hold large experience batches
            # 2. TensorFlow maintains computation graphs and cached tensors
            # 3. PyTorch maintains CUDA memory pools
            # 4. Python's garbage collector may not immediately release large objects
            # 5. Episode-based training creates new environments each iteration
            #
            # ALTERNATIVE APPROACHES (considered but not sufficient):
            # - Incremental buffer trimming: Insufficient for multi-episode training
            # - Lazy garbage collection: Causes memory pressure during episode
            # - Environment pooling: Complex and error-prone with state management
            #
            # This aggressive cleanup is a pragmatic solution for production stability.

            # 1. Clear TensorFlow completely
            # REASON: TensorFlow maintains computation graphs and cached tensors between episodes
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                # Force TF to release GPU memory
                if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'reset_memory_growth'):
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    for gpu in gpus:
                        tf.config.experimental.reset_memory_growth(gpu)
            except Exception:
                pass

            # 2. Clear PyTorch completely
            # REASON: PyTorch maintains CUDA memory pools and cached tensors
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force clear all cached memory
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            # 3. Force agent cleanup
            # REASON: Agent may hold references to large buffers and cached data
            if hasattr(agent, 'memory_tracker'):
                agent.memory_tracker.cleanup('heavy')

            # 4. Clear any existing environment references
            # REASON: Previous episode's environment may hold large state arrays
            if hasattr(agent, 'env') and agent.env is not None:
                if hasattr(agent.env, '_cleanup_memory_enhanced'):
                    agent.env._cleanup_memory_enhanced(force=True)

            # 5. NUCLEAR GARBAGE COLLECTION (multiple passes)
            # REASON: Python's garbage collector may need multiple passes to break circular references
            # and release large objects held by replay buffers and forecast caches
            for i in range(5):
                collected = gc.collect()
                if i == 0:
                    logger.debug(f"   [CLEANUP] GC Pass {i+1}: {collected} objects")

            # 6. Force OS memory release
            # REASON: Some memory may be held by the OS memory allocator
            try:
                if hasattr(os, 'sync'):
                    os.sync()
            except Exception:
                pass

            # Check memory after cleanup
            after_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            freed = initial_mem - after_mem
            logger.info(f"   [OK] Aggressive cleanup freed {freed:.1f}MB | Current: {after_mem:.1f}MB")

            # ABORT if memory is still too high
            if after_mem > 3500:  # More than 3.5GB
                logger.warning(f"   [ALERT] WARNING: High memory ({after_mem:.1f}MB) before episode start!")
                logger.warning(f"   [ALERT] Performing EMERGENCY cleanup...")

                # Emergency cleanup
                for i in range(10):
                    gc.collect()

                final_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                logger.info(f"   [STATS] Post-emergency memory: {final_mem:.1f}MB")

                if final_mem > 4000:  # Still too high
                    logger.error(f"   [ERROR] CRITICAL: Memory too high ({final_mem:.1f}MB) - SKIPPING EPISODE {episode_num}")
                    logger.warning(f"   [SUGGESTION] Restart the process or reduce episode size")
                    continue

        except Exception as cleanup_error:
            logger.warning(f"   [WARN] Pre-episode cleanup error: {cleanup_error}")
            # Continue anyway but warn
            logger.warning(f"   [WARN] Continuing with episode despite cleanup failure...")

        try:
            # 0. Pre-episode memory check and cleanup
            try:
                import psutil
                pre_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                logger.info(f"   [STATS] Memory before Episode {episode_num}: {pre_memory:.1f}MB")

                # Force cleanup if memory is high before starting episode
                if pre_memory > 3000:  # More than 3GB
                    logger.info(f"   [SWEEP] High memory detected, performing pre-episode cleanup...")
                    # gc already imported at module level
                    for _ in range(2):
                        gc.collect()

                    # Clear any lingering TensorFlow sessions
                    try:
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass

                    after_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    freed = pre_memory - after_memory
                    logger.info(f"   [OK] Pre-episode cleanup freed {freed:.1f}MB")

            except Exception:
                pass

            if not bool(getattr(config, "use_global_normalization", False)):
                history_bootstrap_enabled = bool(getattr(config, "rolling_past_history_enable", True))
                history_dir = getattr(config, "rolling_past_history_dir", None)
                if history_bootstrap_enabled and history_dir:
                    history_path = _resolve_rolling_past_history_path(history_dir, episode_num)
                    if history_path is not None:
                        try:
                            history_stats = _prime_rolling_past_from_history(
                                config,
                                history_path,
                                mw_scale_overrides=mw_scale_overrides,
                            )
                            logger.info(
                                f"   [ROLLING_PAST] Episode {episode_num} bootstrapped from {history_path} "
                                f"(rows_used={int(history_stats['rows_used'])}/{int(history_stats['rows_total'])}, "
                                f"price_mean={history_stats['price_mean']:.3f}, "
                                f"price_std={history_stats['price_std']:.3f})"
                            )
                        except Exception as hist_error:
                            logger.warning(
                                f"   [WARN] rolling_past history bootstrap failed for Episode {episode_num}: {hist_error}"
                            )
                    else:
                        logger.warning(
                            f"   [WARN] No rolling_past history file found for Episode {episode_num} in {history_dir}; "
                            "falling back to configured rolling_past state."
                        )

            # 1. Load episode data
            episode_data = load_episode_data(args.episode_data_dir, episode_num, config=config, mw_scale_overrides=mw_scale_overrides)

            # 1.5. EPISODE-SPECIFIC FORECAST MODEL VALIDATION AND CACHE GENERATION
            # Tier-2 requires prebuilt episode-specific forecast models.
            # We validate that the no-leakage forecast models exist for this episode,
            # then generate/load the forecast cache used by the enhancer feature builder.
            episode_forecasts_required = bool(args.forecast_baseline_enable)
            if episode_forecasts_required:
                from episode_forecast_integration import (
                    ensure_episode_forecasts_ready,
                    check_episode_cache_exists,
                    get_episode_forecast_paths
                )
                
                logger.info(f"\n   [FORECAST] Preparing Episode {episode_num} forecast models...")

                # NO-LEAKAGE FORECAST TRAINING (rolling 1-year windows):
                # - MARL episode 0 (2015H1) uses forecast models trained on forecast_scenario_00 (2014H1+H2)
                # - MARL episode k uses forecast models trained on forecast_scenario_{k:02d}
                # - forecast_scenario_20 (2024H1+H2) is reserved for 2025 evaluation (not used in MARL training)
                import os
                forecast_dataset_dir = getattr(args, "forecast_training_dataset_dir", "forecast_training_dataset")
                forecast_scenario_file = f"forecast_scenario_{episode_num:02d}.csv"
                forecast_data_path = os.path.join(forecast_dataset_dir, forecast_scenario_file)
                if not os.path.exists(forecast_data_path):
                    raise FileNotFoundError(
                        f"[FORECAST] Missing forecast training scenario for MARL episode {episode_num}: {forecast_data_path}\n"
                        f"Expected files: {forecast_dataset_dir}\\forecast_scenario_00.csv .. forecast_scenario_20.csv\n"
                        f"Fix: regenerate forecast_training_dataset or set --forecast_training_dataset_dir correctly."
                    )
                
                forecast_base_dir = getattr(args, 'forecast_base_dir', "forecast_models")
                cache_base_dir = getattr(args, 'forecast_cache_dir', "forecast_cache")
                
                # Step 1: Ensure the prebuilt forecast models exist for this episode.
                success, forecast_paths = ensure_episode_forecasts_ready(
                    episode_num=episode_num,
                    episode_data_path=forecast_data_path,
                    forecast_base_dir=forecast_base_dir,
                    cache_base_dir=cache_base_dir
                )
                
                if not success:
                    logger.error(f"   [ERROR] Missing prebuilt forecast models for Episode {episode_num}")
                    logger.error(f"   [ERROR] Cannot continue - forecast models are required for the Tier-2 DL enhancer")
                    raise RuntimeError(f"Episode {episode_num} forecast model validation failed")
                
                # Step 2: Reinitialize forecaster with episode-specific models
                logger.info(f"   [FORECAST] Reinitializing forecaster with Episode {episode_num} models...")
                try:
                    from generator import MultiHorizonForecastGenerator
                    
                    # CRITICAL: Create new forecaster with episode-specific paths
                    # This overwrites the forecaster variable for this episode
                    # The environment will use this episode-specific forecaster
                    forecaster = MultiHorizonForecastGenerator(
                        model_dir=forecast_paths['model_dir'],
                        scaler_dir=forecast_paths['scaler_dir'],
                        metadata_dir=forecast_paths['metadata_dir'],
                        look_back=24,  # Will be overridden by metadata if available
                        verbose=args.debug,
                        fallback_mode=False  # CRITICAL: Fail fast - no fallback allowed
                    )
                    
                    # Verify models loaded correctly
                    stats = forecaster.get_loading_stats()
                    logger.info(f"   [STATS] Episode {episode_num} models: {stats['models_loaded']}/{stats['models_attempted']} loaded ({stats['success_rate']:.1f}% success)")
                    models_loaded = int(stats.get('models_loaded', 0) or 0)
                    models_attempted = int(stats.get('models_attempted', 0) or 0)
                    scalers_loaded = int(stats.get('scalers_loaded', 0) or 0)
                    scalers_attempted = int(stats.get('scalers_attempted', 0) or 0)

                    if (
                        bool(stats.get('fallback_mode', False))
                        or models_loaded <= 0
                        or models_loaded != models_attempted
                        or scalers_loaded != scalers_attempted
                    ):
                        error_msg = (
                            f"Episode {episode_num} forecast models failed to load!\n"
                            f"   Models loaded: {models_loaded}/{models_attempted}\n"
                            f"   Scalers loaded: {scalers_loaded}/{scalers_attempted}\n"
                            f"   Fallback mode: {bool(stats.get('fallback_mode', False))}\n"
                            f"   Model directory: {forecast_paths['model_dir']}\n"
                            f"   Scaler directory: {forecast_paths['scaler_dir']}\n"
                            f"   Metadata directory: {forecast_paths['metadata_dir']}"
                        )
                        logger.error(f"   [ERROR] {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    logger.info(f"   [OK] Forecaster reinitialized with Episode {episode_num} models ({stats['models_loaded']} models loaded)")
                except Exception as e:
                    error_msg = f"Episode {episode_num} forecaster reinitialization failed: {e} - cannot continue (no fallback)"
                    logger.error(f"   [ERROR] {error_msg}")
                    raise RuntimeError(error_msg)

            # 2. Precompute forecasts for this episode (only when forecasts are required)
            if episode_forecasts_required and forecaster is not None:
                try:
                    # Check if cache already exists
                    cache_base_dir = getattr(args, 'forecast_cache_dir', "forecast_cache")
                    episode_cache_dir = os.path.join(cache_base_dir, f"episode_{episode_num}")
                    
                    cache_exists = check_episode_cache_exists(episode_num, cache_base_dir)
                    
                    if cache_exists:
                        logger.info(f"   [FORECAST] Episode {episode_num} cache already exists - loading...")
                        # Cache will be loaded automatically by precompute_offline if it exists
                        # Just call it and it will load from cache
                        forecaster.precompute_offline(
                            df=episode_data,
                            timestamp_col="timestamp",
                            batch_size=max(1, int(args.precompute_batch_size)),
                            cache_dir=episode_cache_dir
                        )
                        logger.info(f"   [OK] Episode {episode_num} forecast cache loaded")
                    else:
                        logger.info(f"   [FORECAST] Precomputing forecasts for Episode {episode_num}...")
                        forecaster.precompute_offline(
                            df=episode_data,
                            timestamp_col="timestamp",
                            batch_size=max(1, int(args.precompute_batch_size)),
                            cache_dir=episode_cache_dir
                        )
                        logger.info(f"   [OK] Episode {episode_num} forecasts precomputed and cached")
                except Exception as pe:
                    logger.error(f"   [ERROR] Episode {episode_num} forecast precomputation failed: {pe}")
                    raise  # Fail fast - forecasts are mandatory when the Tier-2 enhancer is enabled
            elif episode_forecasts_required and forecaster is None:
                # This should not happen - forecaster should have been initialized above
                logger.error(f"   [ERROR] Forecasts required but forecaster is None!")
                raise RuntimeError(f"Episode {episode_num}: Forecasts required but forecaster not initialized")

            # 3. Determine timesteps for this episode
            episode_timesteps = args.episode_timesteps if args.episode_timesteps else len(episode_data)
            logger.info(f"   [GOAL] Episode timesteps: {episode_timesteps:,}")

            # 4. Create new environment with episode data (memory-aware)
            logger.info(f"   [CYCLE] Creating environment for Episode {episode_num}...")
            logger.debug(f"   [DEBUG] forecaster = {forecaster}")
            logger.debug(f"   [DEBUG] forecaster type = {type(forecaster).__name__ if forecaster else 'None'}")

            # MEMORY FIX: Ensure clean environment creation
            try:
                # Create episode-specific environment with proper initialization
                # NEW: Pass logs as log_dir so debug logs are saved in logs folder
                logs_dir = os.path.join(args.save_dir, 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                episode_base_env = RenewableMultiAgentEnv(
                    episode_data,
                    forecast_generator=forecaster,
                    dl_adapter=None,  # DLAdapter will be set after environment creation
                    config=getattr(base_env, 'config', None),
                    investment_freq=getattr(base_env, 'investment_freq', 12),
                    init_budget=getattr(base_env, 'init_budget', None),
                    enhanced_risk_controller=True,
                    log_dir=logs_dir  # NEW: Save debug logs in logs folder
                )
                
                # CRITICAL FIX: Set episode number for debug logging
                # In episode-based training, each episode creates a new environment
                # so _episode_counter resets. We need to set it from the training loop.
                if hasattr(episode_base_env, '_episode_counter'):
                    episode_base_env._episode_counter = episode_num
                    if hasattr(episode_base_env, 'debug_tracker'):
                        episode_base_env.debug_tracker.start_episode(episode_num)
                        logger.debug(f"   [DEBUG] Set episode counter to {episode_num} for debug logging")

                # CRITICAL FIX: Initialize calibration tracker for episode environment
                # For episode training, CalibrationTracker must be initialized per-episode with episode-specific forecaster
                if args.forecast_baseline_enable and forecaster is not None:
                    try:
                        from dl_enhancer import CalibrationTracker
                        episode_base_env.calibration_tracker = CalibrationTracker(
                            window_size=config.forecast_trust_window,
                            trust_metric=config.forecast_trust_metric,
                            verbose=args.debug,
                            init_budget=config.init_budget,
                            direction_weight=config.forecast_trust_direction_weight,
                            trust_boost=getattr(config, "forecast_trust_boost", 0.0),
                            fail_fast=bool(getattr(config, "fgb_fail_fast", False)),
                        )
                        logger.info(f"   [OK] Calibration tracker initialized for episode environment")
                    except Exception as e:
                        err_msg = f"   [ERROR] Failed to initialize CalibrationTracker for episode: {e}"
                        if bool(getattr(config, "fgb_fail_fast", False)):
                            raise RuntimeError(err_msg) from e
                        logger.error(err_msg)
                        episode_base_env.calibration_tracker = None
                else:
                    episode_base_env.calibration_tracker = None
                    if args.forecast_baseline_enable and forecaster is None:
                        msg = "   [WARN] Tier-2 enhancer enabled but forecaster is None - CalibrationTracker not initialized"
                        if bool(getattr(config, "fgb_fail_fast", False)):
                            raise RuntimeError(msg)
                        logger.warning(msg)

                logger.debug(f"   [DEBUG] episode_base_env.forecast_generator = {episode_base_env.forecast_generator}")
                logger.debug(f"   [DEBUG] Enhancer feature_dim = {getattr(base_env, 'enhancer_feature_dim', 'N/A')}")

                # MEMORY FIX: Explicitly clear episode data from memory after env creation
                # Keep only essential references
                episode_data_size = len(episode_data)
                del episode_data  # Free the DataFrame memory
                # gc already imported at module level
                gc.collect()
                logger.info(f"   [CLEANUP] Freed episode data ({episode_data_size} rows) from memory")

            except Exception as env_error:
                logger.error(f"   [ERROR] Environment creation failed: {env_error}")
                raise

            # Initialize environment properly
            episode_base_env.reset()
            logger.info(f"   [OK] Episode environment initialized")

            # CRITICAL: Initialize Tier-2 enhancer state for the episode environment.
            # FIXED: For episode training, reuse the enhancer from base_env (initialized at startup)
            # This prevents dimension changes per-episode, avoiding policy recreation
            if args.forecast_baseline_enable:
                if args.episode_training:
                    # EPISODE TRAINING MODE: Reuse DL enhancer from base_env (initialized at startup with correct dimensions)
                    # This ensures episode environments have the same enhancer structure, preventing dimension mismatches
                    if hasattr(base_env, 'enhancer_adapter') and base_env.enhancer_adapter is not None:
                        try:
                            episode_base_env.enhancer_adapter = base_env.enhancer_adapter
                            episode_base_env.enhancer_trainer = base_env.enhancer_trainer
                            episode_base_env.enhancer_adapter = getattr(base_env, 'enhancer_adapter', None)
                            episode_base_env.enhancer_trainer = getattr(base_env, 'enhancer_trainer', None)
                            episode_base_env.feature_dim = base_env.feature_dim
                            episode_base_env.enhancer_feature_dim = getattr(base_env, 'enhancer_feature_dim', None)
                            # Attach forecaster to episode environment for Tier-2 enhancer features
                            if forecaster is not None:
                                episode_base_env.forecast_generator = forecaster
                            logger.info(
                                f"   [OK] DL enhancer linked from base_env to episode environment "
                                f"(enhancer_feature_dim={episode_base_env.enhancer_feature_dim})"
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"[ENHANCER_EPISODE_FATAL] Failed to link DL enhancer from base_env to episode {episode_num}: {e}"
                            ) from e
                    else:
                        raise RuntimeError(
                            f"[ENHANCER_EPISODE_FATAL] Forecast baseline enabled but base_env.enhancer_adapter is missing for episode {episode_num}."
                        )
                else:
                    # NON-EPISODE TRAINING MODE: Reuse the same DL enhancer from base_env for consistency
                    if hasattr(base_env, 'enhancer_adapter') and base_env.enhancer_adapter is not None:
                        try:
                            episode_base_env.enhancer_adapter = base_env.enhancer_adapter
                            episode_base_env.enhancer_trainer = base_env.enhancer_trainer
                            episode_base_env.enhancer_adapter = getattr(base_env, 'enhancer_adapter', None)
                            episode_base_env.enhancer_trainer = getattr(base_env, 'enhancer_trainer', None)
                            episode_base_env.feature_dim = base_env.feature_dim
                            episode_base_env.enhancer_feature_dim = getattr(base_env, 'enhancer_feature_dim', None)
                            logger.info(
                                f"   [OK] DL enhancer linked to episode environment "
                                f"(enhancer_feature_dim={episode_base_env.enhancer_feature_dim})"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to link DL enhancer to episode: {e}")

            # Forecasts are never appended to agent observations.
            # Use the base environment as-is (Tier-1 observation spaces remain unchanged).
            episode_env = episode_base_env
            # PHASE 5.6 FIX: Set episode training mode flag to prevent environment reset
            episode_env._episode_training_mode = True
            logger.info(f"   [OK] Episode environment initialized")

            # 5. Update agent's environment reference to use episode data
            logger.info(f"   [CYCLE] Updating agent environment reference...")
            agent.env = episode_env
            
            # ROOT CAUSE FIX: Only reinitialize if observation spaces actually changed.
            # If dimensions are already correct, no recreation is needed.
            # Check if dimensions match before triggering recreation
            needs_reinit = False
            for agent_name in agent.possible_agents:
                env_obs_dim = episode_env.observation_space(agent_name).shape[0]
                agent_obs_dim = agent.observation_spaces[agent_name].shape[0] if agent_name in agent.observation_spaces else None
                if agent_obs_dim is not None and env_obs_dim != agent_obs_dim:
                    logger.info(f"   [OBS_SPACE_CHANGE] {agent_name}: dimension mismatch ({agent_obs_dim}D → {env_obs_dim}D) - will recreate policies")
                    needs_reinit = True
                    break
            
            if needs_reinit:
                agent.reinitialize_observation_spaces()
            else:
                # Dimensions match - just update spaces without recreation
                for agent_name in agent.possible_agents:
                    agent.observation_spaces[agent_name] = episode_env.observation_space(agent_name)
                logger.debug(f"   [OBS_SPACE_OK] Observation spaces match environment - no policy recreation needed")

            # [ALERT] CRITICAL: Load policies from PREVIOUS episode to continue learning (for ALL modes)
            # CRITICAL FIX: When resuming, the first episode after resume should load from the resume episode
            # Example: --resume_episode 8 --start_episode 9 → Episode 9 should load from Episode 8
            should_load_previous = (episode_num > args.start_episode) or (resumed_from_checkpoint and episode_num == args.start_episode)

            if should_load_previous:
                prev_episode_num = episode_num - 1
                prev_episode_checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"episode_{prev_episode_num}")

                if os.path.exists(prev_episode_checkpoint_dir):
                    # CRITICAL: Load agent policies from previous episode
                    logger.info(f"   [LOAD] Loading agent policies from Episode {prev_episode_num}...")
                    loaded_policies = agent.load_policies(prev_episode_checkpoint_dir)
                    if loaded_policies > 0:
                        logger.info(f"   [OK] Loaded {loaded_policies} agent policies from Episode {prev_episode_num}")
                    else:
                        logger.warning(f"   [WARN] No agent policies loaded from Episode {prev_episode_num}")

                    # CRITICAL FIX: Load agent training state to preserve total_steps and learning continuity
                    agent_state_path = os.path.join(prev_episode_checkpoint_dir, "agent_training_state.json")
                    if os.path.exists(agent_state_path):
                        try:
                            with open(agent_state_path, 'r') as f:
                                agent_state = json.load(f)

                            # Restore agent's training progress
                            if hasattr(agent, 'total_steps'):
                                restored_total_steps = agent_state.get('total_steps', 0)
                                agent.total_steps = restored_total_steps
                                logger.info(f"   [OK] Restored agent.total_steps: {agent.total_steps:,}")

                            # Restore training metrics if available
                            if hasattr(agent, '_training_metrics') and 'training_metrics' in agent_state:
                                agent._training_metrics.update(agent_state['training_metrics'])
                                logger.info(f"   [OK] Restored training metrics")

                            # Restore error tracking
                            if hasattr(agent, '_consecutive_errors'):
                                agent._consecutive_errors = agent_state.get('consecutive_errors', 0)
                                logger.info(f"   [OK] Restored consecutive_errors: {agent._consecutive_errors}")

                            if _restore_rolling_past_state(config, agent_state.get('rolling_past_state')):
                                logger.info("   [OK] Restored rolling_past normalization state")
                        except Exception as e:
                            logger.warning(f"   [WARN] Could not restore agent training state: {e}")
                    else:
                        logger.warning(f"   [WARN] No agent training state found at: {agent_state_path}")
                else:
                    logger.warning(f"   [WARN] Previous episode checkpoint not found: {prev_episode_checkpoint_dir}")
            else:
                logger.info(f"   [INFO] Episode {episode_num} is first episode, starting with fresh policies")

            if hasattr(episode_base_env, 'enhancer_adapter') and episode_base_env.enhancer_adapter is not None:
                episode_base_env.feature_dim = ENHANCER_BASE_FEATURE_DIM
                episode_base_env.enhancer_feature_dim = int(episode_base_env.enhancer_adapter.feature_dim)

                # Load DL enhancer weights from PREVIOUS episode to continue learning
                # CRITICAL FIX: When resuming, the first episode after resume should load from the resume episode
                if should_load_previous:
                    prev_episode_num = episode_num - 1
                    prev_episode_checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"episode_{prev_episode_num}")

                    if os.path.exists(prev_episode_checkpoint_dir):
                        # Load enhancer weights from previous episode
                        if hasattr(episode_base_env, 'enhancer_adapter') and episode_base_env.enhancer_adapter is not None:
                            fd = episode_base_env.enhancer_adapter.feature_dim
                            enh_path = os.path.join(prev_episode_checkpoint_dir, _enhancer_weights_filename(fd))
                            if os.path.exists(enh_path):
                                try:
                                    episode_base_env.enhancer_adapter.model.load_weights(enh_path)
                                    logger.info(f"   [OK] Loaded DL enhancer weights ({fd}D) from Episode {prev_episode_num}")
                                except Exception as e:
                                    logger.warning(f"   [WARN] Could not load enhancer weights: {e}")
                        else:
                            logger.info(f"   [INFO] Starting Episode {episode_num} with fresh DL enhancer - no compatible weights from Episode {prev_episode_num}")
                    else:
                        logger.warning(f"   [WARN] Previous episode checkpoint not found: {prev_episode_checkpoint_dir}")
                else:
                    logger.info(f"   [INFO] Episode {episode_num} is first episode, starting with fresh DL weights")

            # CRITICAL FIX: Reset RL agent buffers AND internal state for new episode data
            # FIX: Use comprehensive reset method that handles optimizer states and learning rate schedules
            logger.info(f"   [SWEEP] Resetting RL agent buffers and state for new episode...")
            try:
                # FIX: Call comprehensive reset method that handles all internal state
                agent.reset_for_new_episode()
                logger.info(f"   [OK] All RL agent buffers and state reset for Episode {episode_num}")
            except Exception as buffer_error:
                logger.warning(f"   [WARN] Buffer reset warning: {buffer_error}")

            # PHASE 5.8 FIX: Clear TensorFlow/PyTorch caches to prevent state leakage
            logger.info(f"   [CLEANUP] Clearing TensorFlow/PyTorch caches...")
            try:
                import tensorflow as tf
                import torch

                # Clear TensorFlow session and backend
                tf.keras.backend.clear_session()
                logger.info(f"     [OK] Cleared TensorFlow session")

                # Clear PyTorch CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"     [OK] Cleared PyTorch CUDA cache")

                # Force garbage collection
                # gc already imported at module level
                gc.collect()
                logger.info(f"     [OK] Forced garbage collection")

            except Exception as cache_error:
                logger.warning(f"   [WARN] Cache clearing warning: {cache_error}")

            # 6. Setup callbacks for this episode
            callbacks = None
            if args.adapt_rewards:
                reward_cb = RewardAdaptationCallback(episode_base_env, args.reward_analysis_freq, verbose=0)
                callbacks = [reward_cb] * len(episode_env.possible_agents)

            # 7. Run training for this episode
            episode_start_time = datetime.now()

            # CRITICAL FIX: Preserve agent state across episodes for learning continuity
            # Episode 0: preserve_agent_state=False (fresh start)
            # Episode 1+: preserve_agent_state=True (continue learning from previous episode)
            preserve_state = (episode_num > 0)

            # FIX: If checkpoint_freq is -1 (default), save at end of episode
            checkpoint_freq = args.checkpoint_freq
            if checkpoint_freq == -1:
                checkpoint_freq = episode_timesteps
                logger.info(f"   [CHECKPOINT] Using episode length ({episode_timesteps:,} steps) as checkpoint frequency")

            episode_trained = enhanced_training_loop(
                agent=agent,
                env=episode_env,
                timesteps=episode_timesteps,
                checkpoint_freq=checkpoint_freq,
                monitoring_dirs=monitoring_dirs,
                callbacks=callbacks,
                resume_from=None,  # Each episode starts fresh with data
                dl_train_every=args.dl_train_every,  # Pass enhancer training frequency
                preserve_agent_state=preserve_state  # CRITICAL: Preserve learning continuity across episodes
            )

            episode_end_time = datetime.now()
            episode_duration = (episode_end_time - episode_start_time).total_seconds() / 60

            # 6. Episode completion
            total_trained_all_episodes += episode_trained
            successful_episodes += 1

            logger.info(f"\n[OK] Episode {episode_num} COMPLETED!")
            logger.info(f"   Duration: {episode_duration:.1f} minutes")
            logger.info(f"   Steps trained: {episode_trained:,}")
            logger.info(f"   Total progress: {successful_episodes}/{args.end_episode - args.start_episode + 1} episodes")

            # [ALERT] CRITICAL: Print final NAV at end of episode
            print_portfolio_summary(episode_env, episode_trained)

            # 7. Save episode checkpoint
            episode_checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"episode_{episode_num}")
            os.makedirs(episode_checkpoint_dir, exist_ok=True)
            agent.save_policies(episode_checkpoint_dir)

            # 7.1 CRITICAL: Save DL enhancer weights for episode checkpoint
            enhancer_env = episode_env if hasattr(episode_env, 'enhancer_adapter') and episode_env.enhancer_adapter else (
                episode_base_env if 'episode_base_env' in locals() and hasattr(episode_base_env, 'enhancer_adapter') and episode_base_env.enhancer_adapter else None
            )
            if enhancer_env is not None and enhancer_env.enhancer_adapter is not None:
                try:
                    fd = enhancer_env.enhancer_adapter.feature_dim
                    enh_path = os.path.join(episode_checkpoint_dir, _enhancer_weights_filename(fd))
                    enhancer_env.enhancer_adapter.model.save_weights(enh_path)
                    logger.info(f"   [SAVE] DL enhancer weights ({fd}D) saved: {enh_path}")
                except Exception as e:
                    logger.warning(f"   [WARN] Could not save DL enhancer weights: {e}")

            # 7.2 CRITICAL: Save agent training state (total_steps, etc.)
            try:
                agent_state_path = os.path.join(episode_checkpoint_dir, "agent_training_state.json")
                agent_state = {
                    'total_steps': getattr(agent, 'total_steps', 0),
                    'episode_num': episode_num,
                    'episode_trained': episode_trained,
                    'total_trained_all_episodes': total_trained_all_episodes,
                    'timestamp': datetime.now().isoformat(),
                    'training_metrics': getattr(agent, '_training_metrics', {}),
                    'consecutive_errors': getattr(agent, '_consecutive_errors', 0),
                    'rolling_past_state': _capture_rolling_past_state(config),
                }
                with open(agent_state_path, 'w') as f:
                    json.dump(agent_state, f, indent=2)
                logger.info(f"   [SAVE] Agent training state saved: {agent_state_path}")
            except Exception as e:
                logger.warning(f"   [WARN] Could not save agent training state: {e}")

            logger.info(f"   [SAVE] Episode checkpoint saved: {episode_checkpoint_dir}")

            # Note: Portfolio metrics are logged at every timestep in debug CSV - no separate checkpoint summary needed

            # 8. [ALERT] CRITICAL: NUCLEAR MEMORY CLEANUP BETWEEN EPISODES
            logger.warning(f"   [ALERT] NUCLEAR MEMORY CLEANUP AFTER EPISODE {episode_num}")
            try:
                import psutil
                import shutil
                # gc already imported at module level

                # Get memory before cleanup
                before_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                logger.info(f"   [STATS] Memory before cleanup: {before_mem:.1f}MB")

                # 1. DESTROY episode environment references completely
                if 'episode_env' in locals():
                    try:
                        if hasattr(episode_env, '_cleanup_memory_enhanced'):
                            episode_env._cleanup_memory_enhanced(force=True)
                        if hasattr(episode_env, 'close'):
                            episode_env.close()
                        del episode_env
                        logger.debug(f"   [NUCLEAR] Destroyed episode_env")
                    except Exception as e:
                        logger.warning(f"   [WARN] episode_env cleanup error: {e}")

                if 'episode_base_env' in locals():
                    try:
                        if hasattr(episode_base_env, 'cleanup'):
                            episode_base_env.cleanup()
                        if hasattr(episode_base_env, 'close'):
                            episode_base_env.close()
                        del episode_base_env
                        logger.debug(f"   [NUCLEAR] Destroyed episode_base_env")
                    except Exception as e:
                        logger.warning(f"   [WARN] episode_base_env cleanup error: {e}")

                # 2. FORECASTER MEMORY CLEANUP (PRESERVE DISK CACHE)
                if forecaster is not None:
                    try:
                        # Clear only in-memory forecaster caches (preserve disk cache for reuse)
                        if hasattr(forecaster, '_cleanup_memory'):
                            forecaster._cleanup_memory()
                        # NEW: Heavy end-of-episode cleanup to prevent CPU RAM creep across episodes
                        # (drops precomputed arrays + model/scaler refs; safe because forecaster is recreated per-episode)
                        if hasattr(forecaster, 'cleanup_end_of_episode'):
                            forecaster.cleanup_end_of_episode()
                        if hasattr(forecaster, '_global_cache'):
                            forecaster._global_cache.clear()
                        if hasattr(forecaster, '_agent_cache'):
                            forecaster._agent_cache.clear()

                        # [GOAL] PRESERVE episode forecast cache on disk for reuse across episodes
                        # The cached forecasts can be reused by future episodes with same data
                        episode_cache_dir = os.path.join(args.forecast_cache_dir, f"episode_{episode_num}")
                        if os.path.exists(episode_cache_dir):
                            logger.info(f"   [SAVE] PRESERVED episode {episode_num} forecast cache for reuse")

                    except Exception as e:
                        logger.warning(f"   [WARN] Forecaster cleanup error: {e}")

                    # IMPORTANT: Drop reference so Python can reclaim RAM
                    # (forecaster is recreated at the start of the next episode anyway)
                    try:
                        del forecaster
                    except Exception:
                        pass
                    forecaster = None

                # 3. NUCLEAR AGENT CLEANUP
                try:
                    if hasattr(agent, 'memory_tracker'):
                        agent.memory_tracker.cleanup('heavy')

                    # Clear agent buffers if accessible
                    if hasattr(agent, 'policies'):
                        policies_obj = agent.policies
                        if isinstance(policies_obj, dict):
                            policy_iter = policies_obj.values()
                        elif isinstance(policies_obj, (list, tuple)):
                            policy_iter = policies_obj
                        else:
                            policy_iter = []
                        for policy in policy_iter:
                            if hasattr(policy, 'rollout_buffer') and hasattr(policy.rollout_buffer, 'reset'):
                                policy.rollout_buffer.reset()

                    logger.debug(f"   [NUCLEAR] Agent nuclear cleanup completed")
                except Exception as e:
                    logger.warning(f"   [WARN] Agent cleanup error: {e}")

                # 4. NUCLEAR TENSORFLOW CLEANUP
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                    # Force TF memory release
                    if hasattr(tf.config, 'experimental'):
                        gpus = tf.config.experimental.list_physical_devices('GPU')
                        for gpu in gpus:
                            try:
                                tf.config.experimental.reset_memory_growth(gpu)
                            except Exception:
                                pass
                    logger.debug(f"   [NUCLEAR] TensorFlow nuclear cleanup")
                except Exception as e:
                    logger.warning(f"   [WARN] TensorFlow cleanup error: {e}")

                # 5. NUCLEAR PYTORCH CLEANUP
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                    logger.debug(f"   [NUCLEAR] PyTorch nuclear cleanup")
                except Exception as e:
                    logger.warning(f"   [WARN] PyTorch cleanup error: {e}")

                # 6. NUCLEAR GARBAGE COLLECTION (10 passes!)
                total_collected = 0
                for i in range(10):
                    collected = gc.collect()
                    total_collected += collected
                logger.debug(f"   [NUCLEAR] NUCLEAR GC: {total_collected} objects destroyed")

                # 7. Force OS memory sync
                try:
                    import os
                    if hasattr(os, 'sync'):
                        os.sync()
                except Exception:
                    pass

                # 8. Final memory check
                after_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                freed = before_mem - after_mem
                logger.info(f"   [OK] NUCLEAR CLEANUP: {freed:.1f}MB freed | Final: {after_mem:.1f}MB")

                # CRITICAL CHECK: If memory is still high, FORCE more cleanup
                if after_mem > 3000:  # More than 3GB
                    logger.warning(f"   [ALERT] MEMORY STILL HIGH ({after_mem:.1f}MB) - EMERGENCY MEASURES")
                    for i in range(20):  # 20 more GC passes
                        gc.collect()

                    final_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    logger.info(f"   [STATS] Post-emergency memory: {final_mem:.1f}MB")

            except Exception as cleanup_error:
                logger.error(f"   [ERROR] NUCLEAR CLEANUP FAILED: {cleanup_error}")
                logger.warning(f"   [ALERT] CONTINUING ANYWAY - MONITOR MEMORY CLOSELY")

            # 9. Cooling period (except after last episode)
            if episode_num < args.end_episode:
                cooling_period(args.cooling_period, episode_num)

        except Exception as e:
            logger.error(f"\n[ERROR] Episode {episode_num} FAILED: {e}")
            logger.info(f"   Completed episodes: {successful_episodes}")
            logger.info(f"   Total steps trained: {total_trained_all_episodes:,}")

            # Check if this is an OOM error
            is_oom_error = any(keyword in str(e).lower() for keyword in ['memory', 'oom', 'out of memory', 'allocation'])
            if is_oom_error:
                logger.error(f"   [ALERT] DETECTED: Out of Memory (OOM) Error")

                # Report current memory status
                try:
                    import psutil
                    current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    logger.info(f"   [STATS] Current memory usage: {current_memory:.1f}MB")
                except Exception:
                    pass

                # Perform emergency memory cleanup
                logger.warning(f"   [SWEEP] Performing emergency memory cleanup...")
                try:
                    # Clear any remaining episode references
                    for var_name in ['episode_env', 'episode_base_env', 'episode_data']:
                        if var_name in locals():
                            del locals()[var_name]

                    # Force aggressive cleanup
                    # gc already imported at module level
                    for _ in range(5):
                        gc.collect()

                    # Clear TensorFlow
                    try:
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass

                    # Clear CUDA cache
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    logger.info(f"   [OK] Emergency cleanup completed")

                except Exception as cleanup_error:
                    logger.warning(f"   [WARN] Emergency cleanup failed: {cleanup_error}")

            # Save emergency checkpoint
            emergency_dir = os.path.join(monitoring_dirs['checkpoints'], f"emergency_episode_{episode_num}")
            os.makedirs(emergency_dir, exist_ok=True)
            try:
                agent.save_policies(emergency_dir)

                # Save DL enhancer weights for emergency checkpoint
                enhancer_env = episode_env if 'episode_env' in locals() and hasattr(episode_env, 'enhancer_adapter') and episode_env.enhancer_adapter else (
                    episode_base_env if 'episode_base_env' in locals() and hasattr(episode_base_env, 'enhancer_adapter') and episode_base_env.enhancer_adapter else None
                )
                if enhancer_env is not None and enhancer_env.enhancer_adapter is not None:
                    try:
                        fd = enhancer_env.enhancer_adapter.feature_dim
                        enh_path = os.path.join(emergency_dir, _enhancer_weights_filename(fd))
                        enhancer_env.enhancer_adapter.model.save_weights(enh_path)
                        logger.info(f"   [SAVE] Emergency DL enhancer weights saved: {enh_path}")
                    except Exception as dl_save_error:
                        logger.warning(f"   [WARN] Emergency DL enhancer save failed: {dl_save_error}")

                # Save agent training state for emergency checkpoint
                try:
                    agent_state_path = os.path.join(emergency_dir, "agent_training_state.json")
                    agent_state = {
                        'total_steps': getattr(agent, 'total_steps', 0),
                        'episode_num': episode_num,
                        'episode_trained': locals().get('episode_trained', 0),
                        'total_trained_all_episodes': total_trained_all_episodes,
                        'timestamp': datetime.now().isoformat(),
                        'training_metrics': getattr(agent, '_training_metrics', {}),
                        'consecutive_errors': getattr(agent, '_consecutive_errors', 0),
                        'emergency_save': True,
                        'rolling_past_state': _capture_rolling_past_state(config),
                    }
                    with open(agent_state_path, 'w') as f:
                        json.dump(agent_state, f, indent=2)
                    logger.info(f"   [SAVE] Emergency agent state saved: {agent_state_path}")
                except Exception as state_error:
                    logger.warning(f"   [WARN] Emergency agent state save failed: {state_error}")

                logger.info(f"   [SAVE] Emergency checkpoint saved: {emergency_dir}")
            except Exception as save_error:
                logger.warning(f"   [WARN] Emergency checkpoint save failed: {save_error}")

            if is_oom_error:
                logger.info(f"   [SUGGESTION] OOM: Consider reducing batch size, episode timesteps, or enabling more aggressive memory cleanup")
            raise RuntimeError(
                f"[EPISODE_TRAINING_FATAL] Episode {episode_num} failed; aborting run under fail-fast policy."
            ) from e

    logger.info(f"\n[SUCCESS] EPISODE TRAINING COMPLETED!")
    logger.info(f"   Successful episodes: {successful_episodes}/{args.end_episode - args.start_episode + 1}")
    logger.info(f"   Total steps trained: {total_trained_all_episodes:,}")
    logger.info(f"   Time period covered: {get_episode_info(args.start_episode)['start_date']} -> {get_episode_info(args.end_episode)['end_date']}")

    return total_trained_all_episodes

# =====================================================================
# Main
# =====================================================================

def initialize_dl_enhancer(env: RenewableMultiAgentEnv, config, args) -> bool:
    """
    Initialize Tier-2 DL Enhancer for an environment.

    Called when --forecast_baseline_enable. Only the DL enhancer is used for
    Tier-2 exposure adjustment.
    """
    if not bool(getattr(args, "forecast_baseline_enable", False)):
        return False

    try:
        from config import TIER2_ENHANCER_FEATURE_DIM, TIER2_ENHANCER_ABLATED_FEATURE_DIM
        from dl_enhancer import EnhancerAdapter, EnhancerTrainer
        ablate = bool(getattr(config, "tier2_enhancer_ablate_forecast_features", False))
        feature_dim = int(TIER2_ENHANCER_ABLATED_FEATURE_DIM if ablate else TIER2_ENHANCER_FEATURE_DIM)
        logger.info(f"[DL] Initializing Tier-2 DL Enhancer ({feature_dim}D{' ablated' if ablate else ''})")
        logger.debug(f"[DEBUG] env.forecast_generator = {env.forecast_generator}")

        if env.forecast_generator is None:
            episode_training_mode = getattr(args, 'episode_training', False)
            if not episode_training_mode:
                raise ValueError("DL Enhancer requires forecast_generator. Forecasts are mandatory.")
            logger.info("[EPISODE_TRAINING] DL Enhancer: Initializing at startup without forecaster (will be attached per-episode)")

        delta_max = float(getattr(config, "tier2_enhancer_delta_max", 0.35) or 0.35)
        enhancer_lr = float(getattr(config, "tier2_enhancer_lr", 3e-3) or 3e-3)
        intervention_l1 = float(getattr(config, "tier2_enhancer_intervention_l1", 0.02) or 0.0)
        uncertainty_discount = float(getattr(config, "tier2_enhancer_uncertainty_discount", 1.10) or 1.10)
        overconfidence_penalty = float(
            getattr(config, "tier2_enhancer_overconfidence_penalty", 0.03) or 0.03
        )
        direction_loss_weight = float(
            getattr(config, "tier2_enhancer_direction_loss_weight", 0.08) or 0.0
        )
        sharpe_loss_weight = float(
            getattr(config, "tier2_enhancer_sharpe_loss_weight", 0.30) or 0.0
        )
        seed = int(getattr(args, "seed", 42))
        enhancer_adapter = EnhancerAdapter(
            feature_dim=feature_dim,
            delta_max=delta_max,
            seed=seed,
            uncertainty_discount=uncertainty_discount,
        )
        enhancer_trainer = EnhancerTrainer(
            model=enhancer_adapter.model,
            delta_max=delta_max,
            learning_rate=enhancer_lr,
            seed=seed,
            intervention_l1=intervention_l1,
            overconfidence_penalty=overconfidence_penalty,
            direction_loss_weight=direction_loss_weight,
            sharpe_loss_weight=sharpe_loss_weight,
        )
        env.enhancer_adapter = enhancer_adapter
        env.enhancer_trainer = enhancer_trainer
        env.feature_dim = ENHANCER_BASE_FEATURE_DIM
        env.enhancer_feature_dim = feature_dim
        logger.info(
            f"   [OK] Tier-2 DL Enhancer initialized "
            f"({feature_dim}D, delta_max={delta_max:.3f}, lr={enhancer_lr:.6f}, "
            f"memory_steps={getattr(config, 'tier2_enhancer_memory_steps', 3)}, "
            f"l1={intervention_l1:.3f}, dir_loss={direction_loss_weight:.3f}, "
            f"sharpe_loss={sharpe_loss_weight:.3f}, seed={seed})"
        )
        return True

    except Exception as e:
        logger.error(f"[ENHANCER_INIT_FATAL] Failed to initialize DL enhancer: {e}")
        env.enhancer_adapter = None
        env.enhancer_trainer = None
        env.enhancer_feature_dim = None
        raise RuntimeError("[ENHANCER_INIT_FATAL] DL enhancer initialization failed") from e


def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent RL with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, default="sample.csv", help="Path to energy time series data")
    parser.add_argument("--timesteps", type=int, default=50000, help="TUNED: Increased for full synergy emergence (was 20000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for RL training (cuda/cpu)")
    parser.add_argument("--investment_freq", type=int, default=6, help="Investor action frequency in steps")
    parser.add_argument("--meta_freq_min", type=int, default=None, help="Minimum live investor trade cadence in steps (default: config.meta_freq_min).")
    parser.add_argument("--meta_freq_max", type=int, default=None, help="Maximum live investor trade cadence in steps (default: config.meta_freq_max).")
    parser.add_argument("--model_dir", type=str, default="Forecast_ANN/models", help="Dir with trained forecast models (NEW: Updated to Forecast_ANN structure)")
    parser.add_argument("--scaler_dir", type=str, default="Forecast_ANN/scalers", help="Dir with trained scalers (NEW: Updated to Forecast_ANN structure)")
    parser.add_argument("--forecast_training_data", type=str, default="training_dataset/trainset.csv", help="Path to training dataset for episode-specific forecast training")
    parser.add_argument("--forecast_base_dir", type=str, default="forecast_models", help="Base directory for episode-specific forecast models (episode_N subdirectories)")
    parser.add_argument(
        "--forecast_training_dataset_dir",
        type=str,
        default="forecast_training_dataset",
        help=(
            "Directory containing rolling 1-year forecast training scenarios: "
            "forecast_scenario_00.csv .. forecast_scenario_20.csv. "
            "For MARL episode k, forecast models are trained on forecast_scenario_{k:02d} (no leakage). "
            "forecast_scenario_20 is reserved for 2025 evaluation."
        ),
    )

    # Optimization
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")
    parser.add_argument("--optimization_trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--optimization_timeout", type=int, default=1800, help="Optimization timeout (s)")
    parser.add_argument("--use_previous_optimization", action="store_true", help="Use latest saved optimized params")

    # Training
    parser.add_argument("--save_dir", type=str, default="training_agent_results", help="Where to save outputs")
    parser.add_argument("--checkpoint_freq", type=int, default=-1, help="Checkpoint frequency (-1 = end of each episode, otherwise save every N steps)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint directory")
    parser.add_argument("--validate_env", action="store_true", default=True, help="Validate env setup before training")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # Rewards
    parser.add_argument("--adapt_rewards", action="store_true", default=True, help="Enable adaptive reward weights")
    parser.add_argument("--reward_analysis_freq", type=int, default=2000, help="Analyze rewards every N steps")
    # Use optimized_params to override if needed

    parser.add_argument("--expert_blend_weight", type=float, default=0.0, help="Expert suggestion blend weight (0.0=pure PPO, 1.0=pure expert, 0.3=30%% expert)")
    parser.add_argument("--expert_blend_mode", type=str, default="none", choices=["none", "fixed", "adaptive", "residual"],
                       help="Expert blending mode: none (passive hints), fixed (constant blend), adaptive (confidence-based), residual (PPO learns corrections)")
    parser.add_argument("--dl_train_every", type=int, default=100, help="Tier-2 enhancer online training frequency in environment steps.")
    parser.add_argument("--tier2_enhancer_lr", type=float, default=None,
                       help="Tier-2 enhancer optimizer learning rate (default: config.tier2_enhancer_lr).")

    # Forecast backend (Tier-2 DL enhancer)
    parser.add_argument("--forecast_baseline_enable", action="store_true", default=False,
                       help="Enable the Tier-2 DL enhancer runtime path (does NOT change agent observations).")
    parser.add_argument("--forecast_trust_window", type=int, default=None, help="Forecast backend: rolling window for trust calibration (default: config.forecast_trust_window).")
    # Default to directional hit-rate for trust calibration. Users can still select "combo" (dir + magnitude).
    parser.add_argument("--forecast_trust_metric", type=str, default="hitrate", choices=["combo", "hitrate", "absdir"], help="Forecast backend: trust computation method (default: hitrate for canonical parity)")
    parser.add_argument("--forecast_trust_boost", type=float, default=None, help="Forecast backend: optional trust optimism boost (default: config.forecast_trust_boost).")
    parser.add_argument(
        "--tier2_enhancer_ablate_forecast_features",
        action="store_true",
        default=False,
        help="Tier-2 ablation: DL enhancer uses 5D non-forecast core only (no forecast features).",
    )


    # PPO exploration controls (helps avoid near-constant actions / saturation)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: config.lr).")
    parser.add_argument("--ent_coef", type=float, default=None,
                        help="Override PPO entropy coefficient (default: config.ent_coef). Useful to prevent action collapse.")
    parser.add_argument("--ppo_use_sde", dest="ppo_use_sde", action="store_true", default=None,
                        help="Enable PPO gSDE (state-dependent exploration) for continuous actions.")
    parser.add_argument("--ppo_no_sde", dest="ppo_use_sde", action="store_false",
                        help="Disable PPO gSDE (state-dependent exploration) for continuous actions.")
    # NOTE: This repo uses a custom multi-agent rollout collector (not SB3's).
    # We reset gSDE noise explicitly; default to 1 to avoid long-lived sign bias across episodes.
    parser.add_argument("--ppo_sde_sample_freq", type=int, default=None,
                        help="PPO gSDE: sample a new noise matrix every n steps (default: config.ppo_sde_sample_freq).")
    parser.add_argument("--ppo_log_std_init", type=float, default=None,
                        help="Override PPO policy log_std_init for continuous actions (e.g., -0.3 to increase exploration).")
    parser.add_argument("--ppo_mean_clip", type=float, default=None,
                        help="Override PPO mean-action clip before tanh (default: config.ppo_mean_clip).")

    # Forecasting Control (only used when forecasts are enabled for the Tier-2 DL enhancer)
    parser.add_argument("--confidence_floor", type=float, default=0.6, help="Minimum confidence floor (0.0-1.0, default 0.6). MAPE-based confidence cannot go below this value.")
    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=8192,
        help="Batch size to use for offline forecast precompute (TF inference)."
    )

    # MW Scale Configuration - Fix for capacity-factor->MW conversion
    parser.add_argument("--wind_mw_scale", type=float, default=None,
                       help="Override wind MW scale for capacity factor conversion (default: from config)")
    parser.add_argument("--solar_mw_scale", type=float, default=None,
                       help="Override solar MW scale for capacity factor conversion (default: from config)")
    parser.add_argument("--hydro_mw_scale", type=float, default=None,
                       help="Override hydro MW scale for capacity factor conversion (default: from config)")
    parser.add_argument("--load_mw_scale", type=float, default=None,
                       help="Override load MW scale for capacity factor conversion (default: from config)")

    # NEW: Episode training arguments
    parser.add_argument("--episode_training", action="store_true", help="Enable episode-based training with 6-month datasets")
    parser.add_argument("--episode_data_dir", type=str, default="training_dataset", help="Directory containing episode datasets")
    parser.add_argument("--start_episode", type=int, default=0, help="Starting episode number (0-19)")
    parser.add_argument("--end_episode", type=int, default=19, help="Ending episode number (0-19)")
    parser.add_argument(
        "--global_norm_mode",
        type=str,
        default="global",
        choices=["rolling_past", "global"],
        help="Normalization mode for episode runs: rolling_past (episode rolling stats) or global (full-dataset stats).",
    )
    parser.add_argument(
        "--rolling_past_history_dir",
        type=str,
        default="rolling_past_history_dataset",
        help="Directory with causal rolling-past bootstrap files (history_000.csv, history_001.csv, ...). Used only when --global_norm_mode rolling_past.",
    )
    parser.add_argument("--cooling_period", type=int, default=0, help="Cooling period between episodes (minutes)")
    parser.add_argument("--episode_timesteps", type=int, default=None, help="Override timesteps per episode (auto-detect from data if None)")
    parser.add_argument("--resume_episode", type=int, default=None, help="Resume from specific episode checkpoint (loads model from episode_X directory)")
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Directory to save/load cached precomputed forecasts (CSV format)."
    )

    args = parser.parse_args()

    # === SETUP TERMINAL OUTPUT LOGGING ===
    # Create logs directory in save_dir
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"training_output_{timestamp}.txt")
    
    # Setup TeeOutput to capture both print() and logging
    # Create separate instances for stdout and stderr (both write to same file)
    tee_output_stdout = TeeOutput(log_file_path)
    tee_output_stdout._set_stream_type(is_stdout=True)
    tee_output_stderr = TeeOutput(log_file_path)
    tee_output_stderr._set_stream_type(is_stderr=True)
    sys.stdout = tee_output_stdout
    sys.stderr = tee_output_stderr
    # Store both for cleanup (use stdout as primary reference)
    tee_output = tee_output_stdout
    tee_output._stderr_instance = tee_output_stderr  # Store reference to stderr instance
    
    # Reconfigure logging with file handler (centralized in logger.py)
    configure_logging(level=logging.INFO, log_file=log_file_path, force_reconfigure=True)
    
    logger.info(f"\n[LOG] Terminal output being saved to: {log_file_path}")
    logger.info(f"[LOG] All logging will be captured in this file\n")

    # === AUTO-DEPENDENCY RESOLUTION (PAPER MODES) ===
    def resolve_dependencies(args):
        """
        Automatically resolve configuration dependencies for paper-friendly modes:

        - Baseline (Tier 1): No forecasts, no Tier-2 enhancer (observations unchanged).
        - Tier-2 DL Enhancer: --forecast_baseline_enable with forecast backend enabled.
        """
        changes_made = []

        # Log auto-configuration summary
        if changes_made:
            logger.info("\n" + "="*70)
            logger.info("AUTO-CONFIGURATION: Dependencies Resolved")
            logger.info("="*70)
            for change in changes_made:
                logger.info(f"  ✓ {change}")
            logger.info("="*70 + "\n")

        # Log mode information
        if args.forecast_baseline_enable:
            logger.info("[MODE] Tier-2 DL Enhancer: Tier-1 observations + learned exposure adjustment")
        else:
            logger.info("[MODE] Baseline MARL: no forecasts, no enhancer (Tier-1 observations)")

        return args

    # Apply dependency resolution
    args = resolve_dependencies(args)

    # Seed value first (single source of truth for this run).
    seed_value = int(args.seed)

    # CRITICAL: Set deterministic env vars BEFORE TensorFlow/CUDA initialization.
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # Recommended by PyTorch/CUDA for deterministic GEMM behavior.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Pin math threadpools to 1 for stronger determinism.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Safer device selection
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Initialize TensorFlow based on device setting (after deterministic env setup).
    from generator import initialize_tensorflow
    tf = initialize_tensorflow(args.device)

    # CRITICAL: enforce deterministic algorithms (hard fail on non-deterministic ops).
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Seed all random number generators.
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)

    # Seed TensorFlow after initialization.
    if tf is not None:
        tf.random.set_seed(seed_value)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

    logger.info(f"[SEED] All RNGs seeded with {seed_value} for deterministic training")
    logger.info(f"[SEED] PyTorch deterministic algorithms: ENABLED")
    logger.info(f"[SEED] CuDNN deterministic: ENABLED")
    logger.info("[SEED] Threading pinned: OMP=1 MKL=1 NUMEXPR=1 TORCH=1 TF=1")

    logger.info("Enhanced Multi-Horizon Energy Investment RL System")
    logger.info("=" * 60)
    logger.info("Features: Hyperparameter Optimization + Multi-Objective Rewards")

    # Create save dir + metrics subdir
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # CRITICAL FIX: Create config FIRST before loading data
    # Initialize best_params (must happen before config creation)
    best_params = None
    if args.use_previous_optimization:
        logger.info("\nChecking for previous optimization results...")
        opt_dir = os.path.join(args.save_dir, "optimization_results")
        best_params, _ = load_previous_optimization(opt_dir)
        if best_params:
            logger.info("Using previous optimization results")
        else:
            logger.info("No previous optimization found")

    # Create config
    logger.info("\nCreating optimized training configuration...")
    config = EnhancedConfig(optimized_params=best_params)

    # Override config.seed with command-line argument
    config.seed = args.seed
    logger.info(f"[CONFIG] Using seed: {config.seed}")
    # Determinism policy: always use single-threaded policy training.
    config.multithreading = False
    logger.info("[CONFIG] Forced multithreading=False for deterministic training")

    # === OBSERVATION POLICY (TIER-1 OBSERVATION SPACE) ===
    logger.info("\nObservation configuration...")
    logger.info("[OBS] Investor observations: 6D (Tier-1 base)")
    config.investment_freq = int(args.investment_freq)
    if args.meta_freq_min is not None:
        config.meta_freq_min = int(args.meta_freq_min)
    if args.meta_freq_max is not None:
        config.meta_freq_max = int(args.meta_freq_max)
    logger.info(
        f"[CADENCE] startup_investment_freq={config.investment_freq} "
        f"live_meta_freq=[{config.meta_freq_min}, {config.meta_freq_max}]"
    )

    # PPO exploration overrides (optional)
    if args.lr is not None:
        config.lr = float(args.lr)
        logger.info(f"[PPO] Overriding lr -> {config.lr}")
    if args.ent_coef is not None:
        config.ent_coef = float(args.ent_coef)
        logger.info(f"[PPO] Overriding ent_coef -> {config.ent_coef}")
    if getattr(args, "ppo_use_sde", None) is not None:
        config.ppo_use_sde = bool(args.ppo_use_sde)
        logger.info(f"[PPO] Overriding use_sde -> {config.ppo_use_sde}")
    if getattr(args, "ppo_sde_sample_freq", None) is not None:
        config.ppo_sde_sample_freq = int(args.ppo_sde_sample_freq)
        logger.info(f"[PPO] Overriding sde_sample_freq -> {config.ppo_sde_sample_freq}")
    if getattr(args, "ppo_log_std_init", None) is not None:
        config.ppo_log_std_init = float(args.ppo_log_std_init)
        logger.info(f"[PPO] Overriding log_std_init -> {config.ppo_log_std_init}")
    if getattr(args, "ppo_mean_clip", None) is not None:
        config.ppo_mean_clip = float(args.ppo_mean_clip)
        logger.info(f"[PPO] Overriding ppo_mean_clip -> {config.ppo_mean_clip}")
    if bool(getattr(config, "force_disable_sde", False)):
        if config.ppo_use_sde:
            logger.info("[PPO] force_disable_sde=True: overriding use_sde -> False")
        config.ppo_use_sde = False
    logger.info(
        f"[PPO] use_sde={config.ppo_use_sde} "
        f"sde_sample_freq={config.ppo_sde_sample_freq} "
        f"log_std_init={config.ppo_log_std_init} "
        f"mean_clip={getattr(config, 'ppo_mean_clip', None)}"
    )

    # Apply forecast backend CLI args to config
    logger.info("\nApplying Tier-2 DL enhancer configuration...")
    config.forecast_baseline_enable = args.forecast_baseline_enable
    # The manifest reflects the Tier-2 enhancer runtime directly.
    # initialize_dl_enhancer() will perform concrete adapter construction when forecast_baseline_enable.
    if args.forecast_trust_window is not None:
        config.forecast_trust_window = args.forecast_trust_window
    config.forecast_trust_metric = args.forecast_trust_metric
    if args.forecast_trust_boost is not None:
        config.forecast_trust_boost = float(args.forecast_trust_boost)
    config.tier2_enhancer_ablate_forecast_features = bool(getattr(args, "tier2_enhancer_ablate_forecast_features", False))
    if args.tier2_enhancer_lr is not None:
        config.tier2_enhancer_lr = float(args.tier2_enhancer_lr)
    config.tier2_enhancer_train_every = int(getattr(args, "dl_train_every", 100) or 100)
    if args.global_norm_mode is not None:
        config.use_global_normalization = bool(args.global_norm_mode == "global")
        logger.info(f"[GLOBAL_NORM] global_norm_mode={args.global_norm_mode} -> use_global_normalization={config.use_global_normalization}")
    if getattr(args, "rolling_past_history_dir", None):
        config.rolling_past_history_dir = str(args.rolling_past_history_dir)
        logger.info(f"[ROLLING_PAST] history_dir={config.rolling_past_history_dir}")

    # Forecast backend configuration (Tier-2 DL Enhancer)
    logger.info("\nApplying forecast backend configuration...")
    config.fgb_fail_fast = True
    logger.info(
        f"[FORECAST_BACKEND] trust_window={config.forecast_trust_window} metric={config.forecast_trust_metric} "
        f"trust_boost={getattr(config, 'forecast_trust_boost', 0.0)}"
    )

    # Expert Blending: Apply expert blending CLI args to config
    logger.info("\nApplying expert blending configuration...")
    config.expert_blend_mode = args.expert_blend_mode
    config.expert_blend_weight = args.expert_blend_weight
    logger.info(f"[EXPERT_BLEND] Mode: {config.expert_blend_mode}")
    if config.expert_blend_mode != 'none':
        logger.info(f"[EXPERT_BLEND] Weight: {config.expert_blend_weight}")

    logger.info(f"\n[TIER2_ENHANCER] Enabled: {bool(config.forecast_baseline_enable)}")
    if bool(config.forecast_baseline_enable):
        logger.info(
            f"[TIER2_CONTRACT] live_residual_memory=True "
            f"delta_max={float(getattr(config, 'tier2_enhancer_delta_max', 0.35) or 0.35):.3f}"
        )

    # DEBUG: Verify forecast backend + expert blending config values.
    logger.info("\n" + "="*70)
    logger.info("FORECAST BACKEND & EXPERT BLENDING CONFIGURATION VERIFICATION")
    logger.info("="*70)
    logger.info(f"forecast_baseline_enable:  {config.forecast_baseline_enable}")
    logger.info(f"fgb_fail_fast:             {getattr(config, 'fgb_fail_fast', False)}")
    logger.info("tier2_mode:                DL Enhancer")
    logger.info(f"enhancer_lr:               {getattr(config, 'tier2_enhancer_lr', 3e-3)}")
    logger.info(f"enhancer_train_every:      {getattr(config, 'tier2_enhancer_train_every', 100)}")
    logger.info(f"enhancer_delta_max:        {getattr(config, 'tier2_enhancer_delta_max', 0.35)}")
    logger.info(f"enhancer_memory_steps:     {getattr(config, 'tier2_enhancer_memory_steps', 3)}")
    logger.info(f"enhancer_unit_return_ref:  {getattr(config, 'tier2_enhancer_realized_unit_return_ref', 0.01)}")
    logger.info(f"enhancer_intervention_l1:  {getattr(config, 'tier2_enhancer_intervention_l1', 0.02)}")
    logger.info(f"expert_blend_mode:         {getattr(config, 'expert_blend_mode', 'NOT SET')}")
    logger.info(f"expert_blend_weight:       {getattr(config, 'expert_blend_weight', 'NOT SET')}")
    logger.info("="*70)

    # VALIDATE CONFIGURATION (NEW)
    logger.info("\nValidating configuration...")
    try:
        config.validate_configuration()
        logger.info("[OK] Configuration validation passed")
    except ValueError as e:
        logger.error(f"[ERROR] Configuration validation failed:")
        logger.error(f"  {e}")
        return

    # Validate forecast configuration - forecasts are needed for the Tier-2 DL enhancer.
    logger.info("\nValidating forecast configuration...")
    forecast_required = bool(args.forecast_baseline_enable)

    if forecast_required:
        logger.info(f"Forecast models required (forecast_baseline_enable={bool(args.forecast_baseline_enable)})")
        
        # CRITICAL: For episode-based training, models are created per-episode on-the-fly
        # Don't fail if models don't exist at startup - they'll be trained during episode training
        if args.episode_training:
            logger.info("[EPISODE_TRAINING] Forecast models must already exist per episode (episode-specific models)")
            logger.info(f"[EPISODE_TRAINING] Expected model layout: {getattr(args, 'forecast_base_dir', 'forecast_models')}/episode_N/")
            logger.info("[EPISODE_TRAINING] Skipping pre-startup model validation; strict existence checks happen per episode")
        else:
            # For non-episode training (continuous mode), validate existing models
            if not args.model_dir or not os.path.exists(args.model_dir):
                error_msg = (
                    f"[ERROR] Forecast models required but model_dir not found: {args.model_dir}\n"
                    f"  Reason: Forecasts enabled (forecast_baseline_enable)\n"
                    f"  Action: Train forecast models and save to '{args.model_dir}/' or disable forecast_baseline_enable."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            if not args.scaler_dir or not os.path.exists(args.scaler_dir):
                error_msg = (
                    f"[ERROR] Forecast scalers required but scaler_dir not found: {args.scaler_dir}\n"
                    f"  Reason: Forecasts enabled (forecast_baseline_enable)\n"
                    f"  Action: Train forecast models and save scalers to '{args.scaler_dir}/' or disable forecast_baseline_enable."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            logger.info(f"[OK] Forecast directories validated: model_dir={args.model_dir}, scaler_dir={args.scaler_dir}")
    else:
        logger.info("[TIER 1] Baseline MARL mode - no forecasts required")

    # === RUN MANIFEST (REPRODUCIBILITY) ===
    # Write a lightweight JSON manifest capturing argv/args/final config + git hash (if available).
    # This is research-release hygiene and does not affect algorithm behavior.
    try:
        from run_manifest import write_run_manifest
        manifest_path = write_run_manifest(
            args.save_dir,
            argv=sys.argv,
            args=args,
            config=config,
            extra={
                "entrypoint": "main.py",
                "episode_training": bool(getattr(args, "episode_training", False)),
                "forecast_required": bool(forecast_required),
                "forecast_baseline_enable": bool(getattr(args, "forecast_baseline_enable", False)),
            },
            filename_prefix="train_manifest",
        )
        logger.info(f"[MANIFEST] Wrote run manifest: {manifest_path}")
    except Exception as e:
        logger.warning(f"[MANIFEST] Failed to write run manifest: {e}")

    # Apply MW scale overrides from CLI arguments
    mw_scale_overrides = {
        'wind': args.wind_mw_scale,
        'solar': args.solar_mw_scale,
        'hydro': args.hydro_mw_scale,
        'load': args.load_mw_scale
    }

    # Update config with CLI overrides if provided
    if any(v is not None for v in mw_scale_overrides.values()):
        logger.info("\nApplying MW scale overrides from CLI:")
        for key, value in mw_scale_overrides.items():
            if value is not None:
                config.mw_conversion_scales[key] = float(value)
                logger.info(f"  {key}: {value} MW")

    # 1) Load data - Skip for episode training (data loaded per episode)
    if args.episode_training:
        logger.info(f"\n[EPISODE] Episode Training Mode: Data will be loaded per episode from {args.episode_data_dir}")
        logger.info(f"   Skipping initial data load - using episode datasets instead")
        # Create dummy data for environment initialization
        data = pd.DataFrame({
            'timestamp': pd.date_range('2015-01-01', periods=100, freq='10min'),
            'wind': np.random.rand(100),
            'solar': np.random.rand(100),
            'hydro': np.random.rand(100),
            'load': np.random.rand(100),
            'price': np.random.rand(100)
        })
        logger.info(f"   Created dummy data for environment setup: {data.shape}")
    else:
        logger.info(f"\nLoading data from: {args.data_path}")
        try:
            data = load_energy_data(args.data_path, config=config, mw_scale_overrides=mw_scale_overrides)
            logger.info(f"Data loaded: {data.shape}")
            logger.info(f"Columns: {list(data.columns)}")
            if "timestamp" in data.columns and data["timestamp"].notna().any():
                ts = data["timestamp"].dropna()
                logger.info(f"Date range: {ts.iloc[0]} -> {ts.iloc[-1]}")
            if len(data) < 1000:
                logger.warning(f"Limited data ({len(data)} rows). More data -> better training stability.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return

    # 2) Forecaster
    # Tier 1 (Baseline MARL): No forecasts needed
    # Tier-2 enhancer mode:
    # - forecasts are needed for the short-horizon Tier-2 memory features
    # - forecasts are never injected into policy observations or rewards
    forecaster = None
    forecast_required = bool(args.forecast_baseline_enable)

    if forecast_required:
        feature_description = []
        if args.forecast_baseline_enable:
            feature_description.append("Tier-2 DL Enhancer")
            feature_description.append("29D/5D short-memory enhancer feature builder")

        # CRITICAL: For episode-based training, skip forecaster initialization at startup
        # Forecaster will be initialized per-episode with episode-specific models
        if args.episode_training:
            logger.info(f"\n[EPISODE_TRAINING] Skipping forecaster initialization at startup")
            logger.info(f"[EPISODE_TRAINING] Forecaster will be initialized per-episode with episode-specific models")
            logger.info(f"[EPISODE_TRAINING] Episode-specific models must be prebuilt in: forecast_models/episode_N/")
            logger.info(f"[EPISODE_TRAINING] Forecast precomputation will be done per-episode")
        else:
            logger.info(f"\nInitializing multi-horizon forecaster (required for: {', '.join(feature_description)})...")
            try:
                # Auto-detect metadata directory if using Forecast_ANN structure
                metadata_dir = None
                if "Forecast_ANN" in args.model_dir or os.path.exists(os.path.join(os.path.dirname(args.model_dir), "metadata")):
                    potential_metadata = os.path.join(os.path.dirname(args.model_dir), "metadata")
                    if os.path.exists(potential_metadata):
                        metadata_dir = potential_metadata
                
                forecaster = MultiHorizonForecastGenerator(
                    model_dir=args.model_dir,
                    scaler_dir=args.scaler_dir,
                    metadata_dir=metadata_dir,  # NEW: Auto-detected metadata directory
                    look_back=24,  # IMPROVED: Default to 24 (will be overridden by metadata if available)
                    verbose=True,
                    fallback_mode=False,  # CRITICAL: Disable fallback - we need real models
                    timing_log_path=None  # Timing logging disabled (optional performance monitoring)
                )
                logger.info("Forecaster initialized successfully!")

                # DIAGNOSTIC: Check if models are actually loaded
                stats = forecaster.get_loading_stats()
                logger.info(f"   [STATS] Forecast models: {stats['models_loaded']}/{stats['models_attempted']} loaded ({stats['success_rate']:.1f}% success)")
                models_loaded = int(stats.get('models_loaded', 0) or 0)
                models_attempted = int(stats.get('models_attempted', 0) or 0)
                scalers_loaded = int(stats.get('scalers_loaded', 0) or 0)
                scalers_attempted = int(stats.get('scalers_attempted', 0) or 0)

                # FAIL-FAST: No silent fallback or partial loads allowed when forecasts are required
                if (
                    bool(stats.get('fallback_mode', False))
                    or models_loaded <= 0
                    or models_loaded != models_attempted
                    or scalers_loaded != scalers_attempted
                ):
                    error_msg = (
                        f"\n[CRITICAL ERROR] Forecast models failed to load!\n"
                        f"   Models loaded: {models_loaded}/{models_attempted}\n"
                        f"   Scalers loaded: {scalers_loaded}/{scalers_attempted}\n"
                        f"   Fallback mode: {bool(stats.get('fallback_mode', False))}\n"
                        f"   Required for: {', '.join(feature_description)}\n"
                        f"   Model directory: {args.model_dir}\n"
                        f"   Scaler directory: {args.scaler_dir}\n"
                        f"\n"
                        f"   ACTION REQUIRED:\n"
                        f"   1. Train forecast models using the forecast training script\n"
                        f"   2. Ensure models are saved to '{args.model_dir}/'\n"
                        f"   3. Ensure scalers are saved to '{args.scaler_dir}/'\n"
                        f"   OR\n"
                        f"   4. Disable forecast features (remove --forecast_baseline_enable flag)\n"
                        f"\n"
                        f"   Loading errors: {stats['loading_errors'][:3] if stats['loading_errors'] else 'None'}\n"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    logger.info(f"   [OK] Forecaster using trained models - real predictions enabled")

                # Precompute forecasts offline (required for forecast features)
                try:
                    logger.info(f"Precomputing forecasts offline (batch_size={args.precompute_batch_size})…")
                    logger.info("Checking for cached forecasts first...")
                    forecaster.precompute_offline(
                        df=data,
                        timestamp_col="timestamp",
                        batch_size=max(1, int(args.precompute_batch_size)),
                        cache_dir=args.forecast_cache_dir
                    )
                    logger.info("[OK] Forecasts precomputed successfully!")
                except Exception as pe:
                    error_msg = (
                        f"[ERROR] Forecast precomputation failed: {pe}\n"
                        f"  Forecasts are required for: {', '.join(feature_description)}\n"
                        f"  Cannot continue without precomputed forecasts."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            except Exception as e:
                # FAIL-FAST: Don't silently continue if forecasts are required
                error_msg = (
                    f"[CRITICAL ERROR] Failed to initialize forecaster: {e}\n"
                    f"  Forecasts are required for: {', '.join(feature_description)}\n"
                    f"  Cannot continue without forecast models."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    else:
        logger.info("\n[TIER 1] Baseline MARL mode: Skipping forecaster initialization")
        logger.info("[TIER 1] Investor observations: 6D (wind, solar, hydro, price, load, budget)")
        logger.info("[TIER 1] To enable Tier-2 DL Enhancer, use: --forecast_baseline_enable")

    # Re-seed using config.seed to ensure consistency with agent init
    # (config.seed was already set from args.seed above)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # CRITICAL: Re-seed TensorFlow with config.seed
    if tf is not None:
        tf.random.set_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)

    logger.info(f"[SEED] Re-seeded all RNGs with {config.seed} before environment creation")

    # 4) Environment setup
    logger.info("\nSetting up enhanced environment with multi-objective rewards...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(metrics_dir, f"enhanced_metrics_{timestamp}.csv")

    # CRITICAL FIX: Create base environment first, then DL adapter with proper reference
    try:
        # Step 1: Create base environment without DL adapter
        # NEW: Pass logs as log_dir so debug logs are saved in logs folder
        # Create logs subdirectory inside save_dir
        logs_dir = os.path.join(args.save_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        # IMPORTANT: In episode training, base_env is a bootstrap env (used for init/wiring).
        # It should NOT write episode logs into the main logs directory, otherwise it can clobber
        # tier*_debug_ep0.csv when resuming at a later episode.
        bootstrap_logs_dir = os.path.join(logs_dir, "_bootstrap") if args.episode_training else logs_dir
        os.makedirs(bootstrap_logs_dir, exist_ok=True)
        base_env = RenewableMultiAgentEnv(
            data,
            investment_freq=args.investment_freq,
            forecast_generator=forecaster,
            dl_adapter=None,  # No adapter yet
            config=config,  # Pass config to environment
            log_dir=bootstrap_logs_dir  # Keep bootstrap logs isolated in episode training
        )

        # Step 2: Initialize the Tier-2 DL enhancer (shared runtime adapter + trainer)
        initialize_dl_enhancer(base_env, config, args)

        # Initialize CalibrationTracker for forecast trust computation.
        # CRITICAL: For episode training, CalibrationTracker is initialized per-episode with episode-specific forecaster.
        if args.episode_training:
            logger.info("[EPISODE_TRAINING] Skipping CalibrationTracker initialization at startup (will be initialized per-episode)")
            base_env.calibration_tracker = None
        elif config.forecast_baseline_enable and forecaster is not None:
            try:
                from dl_enhancer import CalibrationTracker
                base_env.calibration_tracker = CalibrationTracker(
                    window_size=config.forecast_trust_window,
                    trust_metric=config.forecast_trust_metric,
                    verbose=args.debug,
                    init_budget=config.init_budget,  # Pass fund NAV for exposure scaling
                    direction_weight=config.forecast_trust_direction_weight,  # Weight for directional accuracy (default: 0.7)
                    trust_boost=getattr(config, "forecast_trust_boost", 0.0),
                    fail_fast=bool(getattr(config, "fgb_fail_fast", False)),
                )
                logger.info(f"[FORECAST_TRUST] CalibrationTracker initialized (window={config.forecast_trust_window}, metric={config.forecast_trust_metric}, dir_weight={config.forecast_trust_direction_weight})")
            except Exception as e:
                # FAIL-FAST: CalibrationTracker is required for forecast utilisation
                error_msg = f"[ERROR] CalibrationTracker initialization failed but is required for forecast utilisation: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            base_env.calibration_tracker = None
            if config.forecast_baseline_enable and forecaster is None:
                logger.warning("[WARN] Tier-2 enhancer enabled but forecaster not available - CalibrationTracker not initialized")

        # Forecasts are never appended to agent observations in the paper setup.
        # Use the base environment as-is (Tier-1 observation spaces remain unchanged).
        env = base_env
        logger.info("[OK] Using base environment (Tier-1 observations; no wrapper)")

        # FIXED: Apply command line forecast confidence threshold to config (single source of truth)
        # Only set and display forecast confidence if forecasting is actually enabled
        if forecaster is not None:
            # Apply confidence floor from command line
            if hasattr(config, 'confidence_floor'):
                config.confidence_floor = args.confidence_floor
                logger.info(f"   Confidence floor set to: {args.confidence_floor}")

        logger.info("Enhanced environment created successfully!")
        logger.info("   Multi-objective rewards: enabled")
        if forecaster is not None:
            logger.info(f"   Forecast backend: enabled (confidence_floor={args.confidence_floor})")
            logger.info("   Observations: Tier-1 base (no forecast features)")
        else:
            # In episode-training mode, the forecaster is intentionally initialized per-episode.
            if getattr(args, "episode_training", False) and forecast_required:
                logger.info("   Forecast backend: pending (initialized per-episode)")
            else:
                logger.info("   Forecast backend: disabled")
            logger.info("   Observations: Tier-1 base (no forecast features)")
        logger.info("   Checkpoint summaries: enabled (saved after each checkpoint)")

        # FORECAST OPTIMIZATION: Add diagnostic logging after environment creation
        if forecast_required and forecaster is not None:
            logger.info("\n" + "="*70)
            logger.info("FORECAST BACKEND DIAGNOSTICS")
            logger.info("="*70)
            logger.info(f"Max position size:              {config.max_position_size * 100:.1f}% of capital")
            logger.info(f"Capital allocation:             {config.capital_allocation_fraction * 100:.1f}% of fund")
            logger.info(f"Effective max position:         ${config.init_budget * config.capital_allocation_fraction * config.max_position_size * config.dkk_to_usd_rate / 1e6:.1f}M USD")
            logger.info(f"Confidence floor:               {config.confidence_floor * 100:.0f}%")
            logger.info(f"Forecast reward weight:         {base_env.reward_calculator.reward_weights.get('forecast', 0.0) * 100:.0f}% of total reward" if hasattr(base_env, 'reward_calculator') and base_env.reward_calculator else "Not initialized yet")
            logger.info("="*70)

        # VALIDATION SUMMARY: Display what forecast features are active
        logger.info("\n" + "="*70)
        logger.info("FORECAST FEATURE VALIDATION SUMMARY")
        logger.info("="*70)
        
        # Determine active evaluation/training variant (paper modes)
        if args.forecast_baseline_enable:
            active_variant = "Tier-2 DL Enhancer"
        else:
            active_variant = "Baseline (Tier-1 MARL)"
        logger.info(f"Active Variant:                 {active_variant}")
        
        if forecaster is not None:
            forecaster_status = "YES"
        else:
            # Avoid confusion: in episode-training, this is expected at startup.
            if getattr(args, "episode_training", False) and forecast_required:
                forecaster_status = "NO (expected: initialized per-episode)"
            else:
                forecaster_status = "NO"
        logger.info(f"Forecaster loaded:              {forecaster_status}")
        if args.forecast_baseline_enable:
            from config import TIER2_ENHANCER_FEATURE_DIM, TIER2_ENHANCER_ABLATED_FEATURE_DIM
            enhancer_dim = (
                TIER2_ENHANCER_ABLATED_FEATURE_DIM
                if bool(getattr(config, "tier2_enhancer_ablate_forecast_features", False))
                else TIER2_ENHANCER_FEATURE_DIM
            )
            logger.info(f"DL Enhancer enabled:            YES ({enhancer_dim}D)")
        else:
            logger.info("DL Enhancer enabled:            NO")

        # Only show Tier-2 enhancer details if enabled
        if args.forecast_baseline_enable:
            logger.info("  └─ Tier-2 DL Enhancer:        YES")
            logger.info("  └─ Mode:                      learned investor exposure adjustment from tight short-horizon forecast features")
        else:
            logger.info("Forecast backend mode:          DISABLED")

        # CRITICAL VALIDATION: Ensure consistency
        validation_errors = []
        # CRITICAL: For episode training, forecaster and CalibrationTracker are initialized per-episode
        # Skip forecaster validation for episode training mode
        if not args.episode_training:
            if args.forecast_baseline_enable and forecaster is None:
                validation_errors.append("Forecast-guided baseline enabled but forecaster not loaded!")
        # Validate enhancer when forecast_baseline_enable
        if not args.episode_training and args.forecast_baseline_enable and (not hasattr(base_env, 'enhancer_adapter') or base_env.enhancer_adapter is None):
            validation_errors.append("Forecast baseline enabled but enhancer not initialized!")

        if validation_errors:
            logger.error("\n" + "!"*70)
            logger.error("CRITICAL VALIDATION ERRORS:")
            for error in validation_errors:
                logger.error(f"  ❌ {error}")
            logger.error("!"*70)
            raise RuntimeError("Forecast feature validation failed. See errors above.")
        else:
            # In episode-training startup, forecaster may be None by design; don't claim forecasts are validated.
            if getattr(args, "episode_training", False) and forecast_required and forecaster is None:
                logger.info("\n✓ Forecast backend schema validated (forecaster will be initialized per-episode)")
            else:
                logger.info("\n✓ All forecast features validated successfully")
        logger.info("="*70 + "\n")
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Optional one-time validation (safe)
    if args.validate_env:
        try:
            _obs, _ = env.reset()
            logger.info("Env reset OK for validation.")
        except Exception as e:
            logger.warning(f"Env validation reset failed (continuing): {e}")

    if args.forecast_baseline_enable:
        logger.info("DL Enhancer active (Tier-2 exposure adjustment)")

    # 4) (Optional) HPO (best_params already initialized above)
    if args.optimize and not best_params:
        logger.info("\nRunning hyperparameter optimization...")
        opt_data = data.head(min(5000, len(data)))
        opt_base_env = RenewableMultiAgentEnv(opt_data, forecast_generator=forecaster, dl_adapter=None, config=config)
        # No wrapper: keep Tier-1 observation spaces fixed for a fair comparison.
        opt_env = opt_base_env

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
    logger.info("\nInitializing enhanced multi-agent RL system...")
    try:
        # Reset environment BEFORE initializing agents so observation spaces and any optional
        # forecast backend state are fully initialized.
        logger.info("   [PREP] Resetting environment to finalize observation spaces...")
        try:
            obs = env.reset()

            # FORECAST OPTIMIZATION: Verify forecasts are working after first reset
            if forecast_required and forecaster is not None:
                logger.info("\n" + "="*70)
                logger.info("FORECAST VERIFICATION (After First Reset)")
                logger.info("="*70)
                
                # Check if forecast attributes exist
                has_z_short = hasattr(base_env, 'z_short_price')
                has_z_medium = hasattr(base_env, 'z_medium_price')
                has_trust = hasattr(base_env, '_forecast_trust')
                
                logger.info(f"Forecast attributes present:    {has_z_short and has_z_medium and has_trust}")
                
                if has_z_short and has_z_medium:
                    z_short = float(getattr(base_env, 'z_short_price', 0.0))
                    z_medium = float(getattr(base_env, 'z_medium_price', 0.0))
                    trust = float(getattr(base_env, '_forecast_trust', 0.0))
                    
                    logger.info(f"  z_short_price:                {z_short:.4f}")
                    logger.info(f"  z_medium_price:               {z_medium:.4f}")
                    logger.info(f"  forecast_trust:               {trust:.4f}")
                    
                    # Check if values are non-zero (good sign)
                    if abs(z_short) > 1e-6 or abs(z_medium) > 1e-6:
                        logger.info("  Status:                       ✅ FORECASTS ARE WORKING!")
                    else:
                        logger.warning("  Status:                       ⚠️  WARNING: Forecast z-scores are ZERO!")
                        logger.warning("  Action:                       Check if forecast models loaded correctly")
                else:
                    logger.error("  Status:                       ❌ ERROR: Forecast attributes missing!")
                    logger.error("  Action:                       Check forecast generator initialization")
                
                # Check investor observations
                if 'investor_0' in obs:
                    inv_obs = obs['investor_0']
                    logger.info(f"\nInvestor observation shape:     {inv_obs.shape}")
                    logger.info("Expected shape:                 (6,) (Tier-1 invariant observation)")
                    # Legacy debug branch removed: investor observation shape is logged above.

                logger.info("="*70 + "\n")
                
        except Exception as e:
            logger.warning(f"   [WARN] Environment reset during prep failed: {e}")

        agent = MultiESGAgent(
            config,
            env=env,
            device=args.device,
            training=True,
            debug=args.debug
        )
        logger.info("Enhanced multi-agent system initialized")
        logger.info(f"   Device: {args.device}")
        logger.info(f"   Agents: {env.possible_agents}")
        logger.info(f"   Learning rate: {config.lr:.2e}")
        logger.info(f"   Update frequency: {config.update_every}")
        logger.info("   Multi-objective rewards: enabled")
        logger.info("   Adaptive hyperparameters: enabled")
        
        # Log system readiness for training (after agent is created)
        logger.info("\n" + "="*70)
        logger.info("SYSTEM READINESS FOR TRAINING")
        logger.info("="*70)
        logger.info(f"Agent system initialized:        {'YES' if agent is not None else 'NO'}")
        logger.info(f"Environment initialized:         {'YES' if env is not None else 'NO'}")
        logger.info(f"Observation spaces:             {len(env.observation_spaces) if env else 0} agents")
        logger.info(f"Action spaces:                  {len(env.action_spaces) if env else 0} agents")
        if hasattr(args, 'timesteps') and args.timesteps:
            logger.info(f"Total timesteps to train:       {args.timesteps:,}")
        elif hasattr(args, 'episode_training') and args.episode_training:
            logger.info(f"Training mode:                  Episode-based ({args.start_episode} to {args.end_episode})")
        if hasattr(args, 'checkpoint_freq') and args.checkpoint_freq > 0:
            logger.info(f"Checkpoint frequency:           {args.checkpoint_freq:,}")
        logger.info("="*70)
        logger.info("Training will log:")
        logger.info("  • [TRAINING_PROOF] Policy updates with loss metrics")
        logger.info("  • [DECISION_PROOF] Action sampling and values")
        logger.info("  • [EXECUTION_PROOF] Trade execution and position changes")
        logger.info("  • [SYSTEM_PROOF] Learning progress and NAV changes")
        logger.info("="*70 + "\n")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass
        return

    # 7) Monitoring
    monitoring_dirs = setup_enhanced_training_monitoring(log_path, args.save_dir)



    # 9) Training - Episode or Continuous
    try:
        if args.episode_training:
            logger.info("\n[TRAINING] Starting Episode-Based Training (6-month periods)...")
            logger.info(f"   Episode data directory: {args.episode_data_dir}")
            logger.info(f"   Episodes: {args.start_episode} -> {args.end_episode}")
            logger.info(f"   Cooling period: {args.cooling_period} minutes")
            logger.info("   Thermal management: enabled")

            total_trained = run_episode_training(
                agent=agent,
                base_env=base_env,
                env=env,
                args=args,
                monitoring_dirs=monitoring_dirs,
                config=config,
                mw_scale_overrides=mw_scale_overrides,
                forecaster=forecaster
            )
        else:
            logger.info("\nStarting Enhanced Multi-Objective Training...")
            logger.info(f"   Training timesteps: {args.timesteps:,}")
            logger.info(f"   Checkpoint frequency: {args.checkpoint_freq:,}")
            logger.info(f"   Adaptive rewards: {'enabled' if args.adapt_rewards else 'disabled'}")
            logger.info("   Performance monitoring: enabled")

            callbacks = None
            if args.adapt_rewards:
                reward_cb = RewardAdaptationCallback(base_env, args.reward_analysis_freq, verbose=0)
                # replicate so multi-agent wrapper/agent sees a list of callbacks (one per policy)
                callbacks = [reward_cb] * len(env.possible_agents)

            # FIX: If checkpoint_freq is -1 (default), use old default of 52000 for regular training
            checkpoint_freq = args.checkpoint_freq
            if checkpoint_freq == -1:
                checkpoint_freq = 52000
                logger.info(f"[CHECKPOINT] Using default checkpoint frequency: {checkpoint_freq:,} steps")

            total_trained = enhanced_training_loop(
                agent=agent,
                env=env,
                timesteps=args.timesteps,
                checkpoint_freq=checkpoint_freq,
                monitoring_dirs=monitoring_dirs,
                callbacks=callbacks,
                resume_from=args.resume_from,
                dl_train_every=args.dl_train_every  # Pass enhancer training frequency
            )

        logger.info("Enhanced training completed!")



    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"[TRAINING_FATAL] {e}") from e
    finally:
        # Close envs if they expose close()
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass
        # Always close log file, even on error
        try:
            logger.info(f"\n[LOG] Closing log file: {log_file_path}")
            # Close both tee outputs
            if hasattr(tee_output, 'close'):
                tee_output.close()
            if hasattr(tee_output, '_stderr_instance') and hasattr(tee_output._stderr_instance, 'close'):
                tee_output._stderr_instance.close()
            # Restore original stdout/stderr
            if hasattr(tee_output, 'stdout'):
                sys.stdout = tee_output.stdout
            if hasattr(tee_output, '_stderr_instance') and hasattr(tee_output._stderr_instance, 'stderr'):
                sys.stderr = tee_output._stderr_instance.stderr
            elif hasattr(tee_output, 'stderr'):
                sys.stderr = tee_output.stderr
        except Exception:
            pass

    # 9) Save final models
    logger.info("\nSaving final trained models...")
    final_dir = os.path.join(args.save_dir, "final_models")
    os.makedirs(final_dir, exist_ok=True)
    saved_count = agent.save_policies(final_dir)

    if saved_count > 0:
        logger.info(f"Saved {saved_count} trained agents to: {final_dir}")
        cfg_file = os.path.join(final_dir, "training_config.json")
        with open(cfg_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data_path': args.data_path,
                'total_timesteps_budgeted': int(total_trained),
                'agent_total_steps': int(getattr(agent, 'total_steps', 0)),
                'device': args.device,
                'flags': {
                    'forecast_baseline_enable': bool(getattr(config, 'forecast_baseline_enable', False)),
                    'tier2_live_overlay': bool(getattr(config, 'forecast_baseline_enable', False)),
                },
                'optimized_params': best_params,
                'enhanced_features': {
                    'forecasting_enabled': args.forecast_baseline_enable,
                    'enhancer_enabled': args.forecast_baseline_enable,
                    'dl_enhancer_enabled': args.forecast_baseline_enable,
                    'has_enhancer_weights': args.forecast_baseline_enable and getattr(base_env, "enhancer_adapter", None) is not None,
                    'short_memory_residual_overlay': bool(args.forecast_baseline_enable),
                },
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
        logger.info(f"Training configuration saved to: {cfg_file}")

    # 10) Save the online-trained DL enhancer model (if possible)
    if args.forecast_baseline_enable and hasattr(base_env, 'enhancer_adapter') and base_env.enhancer_adapter is not None:
        try:
            fd = base_env.enhancer_adapter.feature_dim
            enh_path = os.path.join(final_dir, _enhancer_weights_filename(fd))
            base_env.enhancer_adapter.model.save_weights(enh_path)
            logger.info(f"[SAVE] DL enhancer weights ({fd}D) saved: {enh_path}")
        except Exception as e:
            logger.warning(f"Could not save DL enhancer weights: {e}")

    # 11) Force a final log flush (avoid losing buffered rows)
    try:
        if hasattr(env, "_flush_log_buffer"):
            env._flush_log_buffer()
    except Exception:
        pass

    # 12) Quick post-run analysis
    analyze_training_performance(env, log_path, monitoring_dirs)


def smoke_test():
    """Unit smoke test to ensure no duplicate names and no syntax errors."""
    logger.info("Running smoke test...")

    # Test imports to catch syntax errors and duplicate names
    try:
        import environment
        logger.info("[OK] environment.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] environment.py import failed: {e}")
        return False

    try:
        import wrapper
        logger.info("[OK] wrapper.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] wrapper.py import failed: {e}")
        return False

    try:
        import config
        logger.info("[OK] config.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] config.py import failed: {e}")
        return False

    try:
        import risk
        logger.info("[OK] risk.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] risk.py import failed: {e}")
        return False

    try:
        import generator
        logger.info("[OK] generator.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] generator.py import failed: {e}")
        return False

    try:
        import dl_enhancer
        logger.info("[OK] dl_enhancer.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] dl_enhancer.py import failed: {e}")
        return False

    try:
        import evaluation
        logger.info("[OK] evaluation.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] evaluation.py import failed: {e}")
        return False

    try:
        import metacontroller
        logger.info("[OK] metacontroller.py imported successfully")
    except Exception as e:
        logger.error(f"[FAIL] metacontroller.py import failed: {e}")
        return False

    logger.info("[OK] All modules imported successfully - no duplicate names or syntax errors detected")
    return True


if __name__ == "__main__":
    # Run smoke test first if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        smoke_test()
    else:
        main()
