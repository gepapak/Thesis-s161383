# main.py

import argparse
import os
import sys
from datetime import datetime
import json
import random
import glob
import logging
import gc
from typing import Optional, Dict, Any

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
from logger import configure_logging, get_logger, TeeOutput

# Configure logging (will be reconfigured with file handler in main())
configure_logging()
logger = get_logger(__name__)




# Import patched environment classes
from environment import RenewableMultiAgentEnv
from generator import load_energy_data
from utils import clear_tf_session, configure_tf_memory  # UNIFIED: Import from single source of truth
from runtime_contract import (
    build_runtime_contract,
    forecast_prior_contract_settings,
    runtime_contract_hash,
)
from forecast_prior_cli import (
    add_forecast_prior_override_args,
    apply_forecast_prior_overrides,
    format_forecast_prior_overrides,
)

# Meta-controller (patched)
try:
    from metacontroller import MultiESGAgent, HyperparameterOptimizer
except Exception:
    from metacontroller import MultiESGAgent
    HyperparameterOptimizer = None

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
        'logs': os.path.join(save_dir, 'logs')
    }
    for _, dir_path in monitoring_dirs.items():
        os.makedirs(dir_path, exist_ok=True)

    logger.info("Enhanced monitoring setup:")
    logger.info(f"   Metrics log: {log_path}")
    logger.info(f"   Checkpoints: {monitoring_dirs['checkpoints']}")
    logger.info(f"   Logs:        {monitoring_dirs['logs']}")
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

        current_timestep = getattr(base_env, 't', 0)
        nav_breakdown = base_env.get_fund_nav_breakdown() if hasattr(base_env, "get_fund_nav_breakdown") else {}
        if not nav_breakdown:
            fund_nav_dkk = float(base_env._calculate_fund_nav())
            dkk_to_usd = float(getattr(base_env.config, 'dkk_to_usd_rate', 0.145))
            financial_mtm_dkk = float(
                sum((getattr(base_env, 'financial_mtm_positions', getattr(base_env, 'financial_positions', {})) or {}).values())
            )
            trading_cash_dkk = float(getattr(base_env, 'budget', 0.0))
            nav_breakdown = {
                'fund_nav_dkk': fund_nav_dkk,
                'trading_cash_dkk': trading_cash_dkk,
                'trading_cash_core_dkk': trading_cash_dkk,
                'battery_cash_contribution_dkk': 0.0,
                'trading_sleeve_value_dkk': float(trading_cash_dkk + financial_mtm_dkk),
                'physical_book_value_dkk': 0.0,
                'accumulated_operational_revenue_dkk': float(getattr(base_env, 'accumulated_operational_revenue', 0.0)),
                'financial_mtm_dkk': financial_mtm_dkk,
                'trading_mtm_tracker_dkk': float(getattr(base_env, 'cumulative_mtm_pnl', 0.0)),
                'battery_revenue_tracker_dkk': float(getattr(base_env, 'cumulative_battery_revenue', 0.0)),
            }
        dkk_to_usd = float(getattr(base_env.config, 'dkk_to_usd_rate', 0.145))

        fund_nav_usd_m = float(nav_breakdown.get('fund_nav_dkk', 0.0)) * dkk_to_usd / 1_000_000.0
        cash_usd_m = float(nav_breakdown.get('trading_cash_dkk', 0.0)) * dkk_to_usd / 1_000_000.0
        trading_cash_core_usd_m = float(nav_breakdown.get('trading_cash_core_dkk', nav_breakdown.get('trading_cash_dkk', 0.0))) * dkk_to_usd / 1_000_000.0
        battery_cash_contribution_usd_m = float(nav_breakdown.get('battery_cash_contribution_dkk', 0.0)) * dkk_to_usd / 1_000_000.0
        trading_sleeve_usd_m = float(nav_breakdown.get('trading_sleeve_value_dkk', 0.0)) * dkk_to_usd / 1_000_000.0
        physical_usd_m = float(nav_breakdown.get('physical_book_value_dkk', 0.0)) * dkk_to_usd / 1_000_000.0
        operating_revenue_usd_m = float(nav_breakdown.get('accumulated_operational_revenue_dkk', 0.0)) * dkk_to_usd / 1_000_000.0
        mtm_usd_m = float(nav_breakdown.get('financial_mtm_dkk', 0.0)) * dkk_to_usd / 1_000_000.0

        logger.info(f"\n[STATS] PORTFOLIO SUMMARY - Step {step_count:,} (Env: {current_timestep:,})")
        logger.info(
            f"   NAV: ${fund_nav_usd_m:.2f}M = Cash ${cash_usd_m:.2f}M + "
            f"Physical ${physical_usd_m:.2f}M + Op ${operating_revenue_usd_m:.2f}M + MTM ${mtm_usd_m:.2f}M"
        )
        logger.info(
            f"   Sleeve: Trading ${trading_sleeve_usd_m:.2f}M = Cash ${cash_usd_m:.2f}M + MTM ${mtm_usd_m:.2f}M"
        )
        logger.info(
            f"   Cash: Trading Core ${trading_cash_core_usd_m:.2f}M + "
            f"Battery ${battery_cash_contribution_usd_m:+.2f}M = Cash ${cash_usd_m:.2f}M"
        )

    except Exception as e:
        logger.error(f"\n[STATS] PORTFOLIO SUMMARY - Step {step_count:,}")
        logger.error(f"   Error calculating portfolio: {e}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _export_episode_investor_health_csv(
    episode_env,
    episode_num: int,
    monitoring_dirs: Dict[str, str],
    investor_health_summary: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    DEPRECATED: Health rows are now written incrementally by RewardLogger (one CSV per agent
    per episode).  This stub is kept only so that existing call sites don't crash; it is a
    no-op and will be removed in a future cleanup pass.
    """
    return None
def enhanced_training_loop(agent, env, timesteps: int, checkpoint_freq: int, monitoring_dirs: Dict[str, str], callbacks=None, resume_from: str = None, preserve_agent_state: bool = False) -> int:
    """
    Train in intervals, but measure *actual* steps each time.
    Works with a meta-controller whose learn(total_timesteps=N) means "do N more steps".
    Enhanced with comprehensive memory monitoring and leak prevention.

    Args:
        agent: MultiESGAgent instance
        env: Environment
        timesteps: Total timesteps to train
        checkpoint_freq: Frequency of checkpoints
        monitoring_dirs: Monitoring directories
        callbacks: Training callbacks
        resume_from: Resume from checkpoint path
        preserve_agent_state: If True, don't reset agent state even on first interval (for episode continuity)
    """
    logger.info("Starting Enhanced Training Loop with Memory Monitoring")
    logger.info(f"   Total timesteps: {timesteps:,}")
    logger.info(f"   Checkpoint frequency: {checkpoint_freq:,}")

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
    zero_step_retry_used = False  # NAN_GUARD: allow one zero-step retry per episode call.

    # NAN_GUARD: reset per-episode rollback budget on each policy. Each episode gets one
    # rollback attempt; a second NaN within the same episode fails hard.
    try:
        if hasattr(agent, "policies"):
            for _pol in agent.policies:
                _pol._nan_rollback_count = 0
    except Exception:
        pass

    if resume_from and os.path.exists(resume_from):
        logger.info(f"\n[CYCLE] RESUMING from checkpoint: {resume_from}")

        loaded_count = agent.load_policies(resume_from)
        if loaded_count > 0:
            logger.info(f"[OK] Loaded {loaded_count} agent policies")

            # Load training state
            checkpoint_state = load_checkpoint_state(resume_from)
            if checkpoint_state:
                total_trained = checkpoint_state.get('total_trained', 0)
                checkpoint_count = checkpoint_state.get('checkpoint_count', 0)
                logger.info(f"[OK] Resuming from step {total_trained:,} (checkpoint {checkpoint_count})")

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
                else:
                    recent_growth = 0

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
                # FAIL-HARD: zero-step intervals after warmup are diagnostic of NaN guards
                # firing or environment deadlock. Allow ONE retry (a NaN rollback can leave
                # the buffer empty), then abort — silent skipping of episodes is forbidden.
                if not zero_step_retry_used:
                    zero_step_retry_used = True
                    logger.warning("[FAIL_HARD_RETRY] First zero-step interval - retrying once after recovery.")
                    continue
                raise RuntimeError(
                    f"[FAIL_HARD] Two consecutive zero-step intervals at total_trained={total_trained:,}/"
                    f"{timesteps:,}. Refusing to silently skip remaining {timesteps - total_trained:,} steps. "
                    f"Likely cause: NaN guard tripped or env deadlock. Investigate logs above."
                )
            else:
                # Reset retry counter once we make progress again.
                zero_step_retry_used = False

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

def run_episode_training(agent, base_env, env, args, monitoring_dirs, config, mw_scale_overrides):
    """Run training across multiple 6-month episodes"""
    import os
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
    last_episode_investor_health = {}

    # CRITICAL FIX: Track if we resumed from a checkpoint
    # When resuming from episode N, we've already loaded policies from episode N
    # So the first episode (N+1) should load from episode N, not start fresh
    resumed_from_checkpoint = args.resume_episode is not None

    # Resolve checkpoint frequency and callbacks for the episode training loop
    checkpoint_freq = getattr(args, 'checkpoint_freq', -1)
    if checkpoint_freq <= 0:
        checkpoint_freq = 52000
    callbacks = None

    for episode_num in range(args.start_episode, args.end_episode + 1):
        # CRITICAL FIX: More aggressive cleanup for episodes 10+ to prevent OOM
        if episode_num >= 10:
            logger.warning(f"   [ALERT] Episode {episode_num} - EXTRA AGGRESSIVE CLEANUP")
            # Additional GC passes
            for i in range(10):
                gc.collect()
            # Clear any environment caches
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
                if True:
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
                        if True:
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

            # Forecast utilization is cache-only. When enabled, the environment
            # loads forecast_cache/episode_N directly and fails fast on a miss.

            # 3. Determine timesteps for this episode
            episode_timesteps = args.episode_timesteps if args.episode_timesteps else len(episode_data)
            logger.info(f"   [GOAL] Episode timesteps: {episode_timesteps:,}")

            # 4. Create new environment with episode data (memory-aware)
            logger.info(f"   [CYCLE] Creating environment for Episode {episode_num}...")
            # MEMORY FIX: Ensure clean environment creation
            try:
                # Create episode-specific environment with proper initialization
                # NEW: Pass logs as log_dir so debug logs are saved in logs folder
                logs_dir = os.path.join(args.save_dir, 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                episode_base_env = RenewableMultiAgentEnv(
                    episode_data,
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
                    logger.debug(f"   [DEBUG] Set episode counter to {episode_num} for debug logging")

                episode_base_env.calibration_tracker = None

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

            # Forecasts are never appended to agent observations.
            # Use the base environment as-is (Tier-1 observation spaces remain unchanged).
            episode_env = episode_base_env
            # PHASE 5.6 FIX: Set episode training mode flag to prevent environment reset
            episode_env._episode_training_mode = True
            logger.info(f"   [OK] Episode environment initialized")

            # 5. Update agent's environment reference to use episode data
            logger.info(f"   [CYCLE] Updating agent environment reference...")
            agent.env = episode_env
            try:
                from wrapper import EnhancedObservationValidator

                agent.obs_validator = EnhancedObservationValidator(
                    episode_env,
                    debug=getattr(agent, 'debug', False),
                )
                logger.debug("   [OBS_VALIDATOR_REBIND] Observation validator rebound to episode environment")
            except Exception as obs_validator_error:
                logger.warning(f"   [WARN] Failed to rebind observation validator: {obs_validator_error}")
            
            # ROOT CAUSE FIX: Only reinitialize if observation spaces actually changed.
            # If dimensions are already correct, no recreation is needed.
            # Check if dimensions match before triggering recreation
            needs_reinit = False
            for agent_name in agent.possible_agents:
                def _fd_local(sp):
                    if sp is None: return None
                    shp = getattr(sp, "shape", None)
                    if shp is not None:
                        try: return int(shp[0])
                        except Exception: return None
                    subs = getattr(sp, "spaces", None)
                    if subs is not None:
                        try: return int(sum(int(v.shape[0]) for v in subs.values() if getattr(v, "shape", None) is not None))
                        except Exception: return None
                    return None
                _env_sp = episode_env.observation_space(agent_name)
                _ag_sp = agent.observation_spaces.get(agent_name)
                env_obs_dim = _fd_local(_env_sp)
                agent_obs_dim = _fd_local(_ag_sp)
                if agent_obs_dim is not None and env_obs_dim is not None and env_obs_dim != agent_obs_dim:
                    logger.info(f"   [OBS_SPACE_CHANGE] {agent_name}: dimension mismatch ({agent_obs_dim}D ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ {env_obs_dim}D) - will recreate policies")
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
            # Example: --resume_episode 8 --start_episode 9 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ Episode 9 should load from Episode 8
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
                if episode_num == args.start_episode and args.resume_from:
                    logger.info(
                        f"   [INFO] Episode {episode_num} is first episode in this run; "
                        "policies will be initialized from --resume_from inside the training loop"
                    )
                else:
                    logger.info(f"   [INFO] Episode {episode_num} is first episode, starting with fresh policies")
            initial_resume_from = args.resume_from if episode_num == args.start_episode else None

            episode_start_time = datetime.now()
            episode_trained = enhanced_training_loop(
                agent=agent,
                env=episode_env,
                timesteps=episode_timesteps,
                checkpoint_freq=checkpoint_freq,
                monitoring_dirs=monitoring_dirs,
                callbacks=callbacks,
                resume_from=initial_resume_from,
                preserve_agent_state=(episode_num > args.start_episode)  # CRITICAL: Preserve learning continuity across episodes
            )

            episode_end_time = datetime.now()
            episode_duration = (episode_end_time - episode_start_time).total_seconds() / 60

            # FAIL-HARD: episode must collect at least 95% of expected steps. Anything less
            # indicates a guard tripped or env deadlocked — do NOT silently advance to the
            # next episode and corrupt the training schedule.
            _min_required = int(0.95 * episode_timesteps)
            if episode_trained < _min_required:
                raise RuntimeError(
                    f"[FAIL_HARD] Episode {episode_num} collected only {episode_trained:,}/"
                    f"{episode_timesteps:,} steps ({episode_trained / max(1, episode_timesteps):.1%}); "
                    f"required >= {_min_required:,} (95%). Refusing to proceed to episode "
                    f"{episode_num + 1} with truncated training. Likely NaN guard rollback "
                    f"or environment failure — investigate logs above."
                )

            # 6. Episode completion
            total_trained_all_episodes += episode_trained
            successful_episodes += 1

            logger.info(f"\n[OK] Episode {episode_num} COMPLETED!")
            logger.info(f"   Duration: {episode_duration:.1f} minutes")
            logger.info(f"   Steps trained: {episode_trained:,}")
            logger.info(f"   Total progress: {successful_episodes}/{args.end_episode - args.start_episode + 1} episodes")

            episode_investor_health = {}
            try:
                if hasattr(episode_env, "get_investor_health_summary"):
                    episode_investor_health = dict(episode_env.get_investor_health_summary() or {})
                    last_episode_investor_health = dict(episode_investor_health)
                    logger.info(
                        "   [INV_HEALTH] "
                        f"clip_rate={float(episode_investor_health.get('mean_clip_hit_rate', 0.0)):.3f} "
                        f"mu_abs={float(episode_investor_health.get('mu_abs_roll', 0.0)):.3f} "
                        f"sign_cons={float(episode_investor_health.get('mu_sign_consistency', 0.0)):.3f} "
                        f"sigma={float(episode_investor_health.get('policy_sigma_raw', 0.0)):.3f}"
                    )
            except Exception as e:
                logger.warning(f"   [WARN] Could not capture investor health summary: {e}")

            # Battery / Risk / Meta one-liner health summaries (matches investor pattern)
            try:
                if hasattr(episode_env, "get_battery_health_summary"):
                    bh = dict(episode_env.get_battery_health_summary() or {})
                    logger.info(
                        "   [BAT_HEALTH] "
                        f"last_action={int(bh.get('last_action_idx', 0))} "
                        f"q_max={float(bh.get('last_q_max', 0.0)):.3f} "
                        f"q_chosen={float(bh.get('last_q_chosen', 0.0)):.3f} "
                        f"reward_step={float(bh.get('last_reward_step', 0.0)):.4f} "
                        f"soc={float(bh.get('last_soc', 0.0)):.3f}"
                    )
            except Exception as e:
                logger.warning(f"   [WARN] Could not capture battery health summary: {e}")

            try:
                if hasattr(episode_env, "get_risk_health_summary"):
                    rh = dict(episode_env.get_risk_health_summary() or {})
                    logger.info(
                        "   [RISK_HEALTH] "
                        f"action={float(rh.get('last_action_raw', 0.0)):.3f} "
                        f"cap={float(rh.get('last_risk_cap', 1.0)):.3f} "
                        f"drawdown={float(rh.get('last_drawdown', 0.0)):.4f} "
                        f"value={float(rh.get('last_value', 0.0)):.3f} "
                        f"td_err={float(rh.get('last_td_error', 0.0)):.4f}"
                    )
            except Exception as e:
                logger.warning(f"   [WARN] Could not capture risk health summary: {e}")

            try:
                if hasattr(episode_env, "get_meta_health_summary"):
                    mh = dict(episode_env.get_meta_health_summary() or {})
                    logger.info(
                        "   [META_HEALTH] "
                        f"action={float(mh.get('last_action_raw', 0.0)):.3f} "
                        f"cap_frac={float(mh.get('last_cap_fraction', 0.0)):.3f} "
                        f"budget_n={float(mh.get('last_budget_n', 0.0)):.3f} "
                        f"value={float(mh.get('last_value', 0.0)):.3f} "
                        f"td_err={float(mh.get('last_td_error', 0.0)):.4f}"
                    )
            except Exception as e:
                logger.warning(f"   [WARN] Could not capture meta health summary: {e}")

            # [ALERT] CRITICAL: Print final NAV at end of episode
            print_portfolio_summary(episode_env, episode_trained)

            # 7. Save episode checkpoint
            episode_checkpoint_dir = os.path.join(monitoring_dirs['checkpoints'], f"episode_{episode_num}")
            os.makedirs(episode_checkpoint_dir, exist_ok=True)
            agent.save_policies(episode_checkpoint_dir)

            # 7.1 CRITICAL: Save agent training state (total_steps, etc.)
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
                    'investor_health': episode_investor_health,
                    'rolling_past_state': _capture_rolling_past_state(config),
                }
                with open(agent_state_path, 'w') as f:
                    json.dump(agent_state, f, indent=2)
                logger.info(f"   [SAVE] Agent training state saved: {agent_state_path}")
            except Exception as e:
                logger.warning(f"   [WARN] Could not save agent training state: {e}")

            # Health CSVs are written incrementally during the episode (one per agent).
            # Log the paths so they're visible in the run log.
            try:
                _dt = getattr(episode_env, "debug_tracker", None)
                if _dt is not None:
                    for _ak in ('investor', 'battery', 'risk', 'meta'):
                        _h = _dt.agent_health_handles.get(_ak)
                        if _h is not None and not _h.closed:
                            _h.flush()
                            logger.info(f"   [HEALTH] {_ak} health log: {_h.name}")
            except Exception as e:
                logger.warning(f"   [WARN] Could not flush agent health logs: {e}")

            logger.info(f"   [SAVE] Episode checkpoint saved: {episode_checkpoint_dir}")

            # Note: Portfolio metrics are logged at every timestep in debug CSV - no separate checkpoint summary needed

            # 8. [ALERT] CRITICAL: NUCLEAR MEMORY CLEANUP BETWEEN EPISODES
            logger.warning(f"   [ALERT] NUCLEAR MEMORY CLEANUP AFTER EPISODE {episode_num}")
            try:
                import psutil
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

                # 2. NUCLEAR AGENT CLEANUP
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
                    if True:
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
                    else:
                        logger.debug(f"   [NUCLEAR] Skipped TensorFlow session clear to preserve continuous Tier1 state")
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
                    try:
                        del episode_env
                    except NameError:
                        pass
                    try:
                        del episode_base_env
                    except NameError:
                        pass
                    try:
                        del episode_data
                    except NameError:
                        pass

                    # Force aggressive cleanup
                    # gc already imported at module level
                    for _ in range(5):
                        gc.collect()

                    # Clear TensorFlow only when no persistent Tier1 runtime is active.
                    try:
                        if True:
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



def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent RL with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, default="sample.csv", help="Path to energy time series data")
    parser.add_argument("--timesteps", type=int, default=50000, help="TUNED: Increased for full synergy emergence (was 20000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for RL training (cuda/cpu)")
    parser.add_argument("--investment_freq", type=int, default=6, help="Investor action frequency in steps")
    parser.add_argument("--meta_freq_min", type=int, default=None, help="Minimum live investor trade cadence in steps (default: config.meta_freq_min).")
    parser.add_argument("--meta_freq_max", type=int, default=None, help="Maximum live investor trade cadence in steps (default: config.meta_freq_max).")
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
    parser.add_argument("--expert_blend_mode", type=str, default="none", choices=["none", "fixed", "adaptive"],
                       help="Expert blending mode: none (passive hints), fixed (constant blend), adaptive (confidence-based)")
    parser.set_defaults(tier1_dir=None)
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
        default="rolling_past",
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
    parser.add_argument("--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Directory to save/load cached precomputed forecasts (CSV format)."
    )
    parser.add_argument(
        "--enable_forecast_utilization",
        action="store_true",
        default=False,
        help=(
            "Single switch for ANN forecast-cache utilization. Investor observations "
            "remain 12-D; the PPO action is interpreted as a residual around a "
            "calibrated cache-derived target exposure."
        ),
    )
    add_forecast_prior_override_args(parser)

    args = parser.parse_args()
    args.tier1_dir = None

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

        - Baseline (Tier 1): no forecast influence.
        - Forecast-prior variant: precomputed ANN cache drives the execution prior;
          observations remain unchanged.
        """
        changes_made = []

        # Log auto-configuration summary
        if changes_made:
            logger.info("\n" + "="*70)
            logger.info("AUTO-CONFIGURATION: Dependencies Resolved")
            logger.info("="*70)
            for change in changes_made:
                logger.info(f"  ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ {change}")
            logger.info("="*70 + "\n")

        logger.info("[MODE] Tier-1 hybrid RL baseline")

        return args

    # Apply dependency resolution
    args = resolve_dependencies(args)

    # Seed value first (single source of truth for this run).
    seed_value = int(args.seed)

    tf_thread_budget = 1
    tf_interop_budget = 1
    torch_thread_budget = 1
    torch_interop_budget = 1
    blas_thread_budget = 1

    # CRITICAL: Set deterministic env vars BEFORE TensorFlow/CUDA initialization.
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # Recommended by PyTorch/CUDA for deterministic GEMM behavior.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['OMP_NUM_THREADS'] = str(blas_thread_budget)
    os.environ['MKL_NUM_THREADS'] = str(blas_thread_budget)
    os.environ['NUMEXPR_NUM_THREADS'] = str(blas_thread_budget)

    # Safer device selection
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Initialize TensorFlow based on device setting (after deterministic env setup).
    from generator import initialize_tensorflow
    tf = initialize_tensorflow(args.device)

    # Enable GPU memory growth BEFORE any TF graph operations.
    configure_tf_memory()

    # CRITICAL: enforce deterministic algorithms (hard fail on non-deterministic ops).
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(torch_thread_budget)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(torch_interop_budget)
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
            tf.config.threading.set_intra_op_parallelism_threads(tf_thread_budget)
            tf.config.threading.set_inter_op_parallelism_threads(tf_interop_budget)
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
    logger.info("[OBS] Investor observations: 12D (price_momentum, realized_vol, budget, exposure, mtm_pnl, decision_step, cap_frac, risk_cap, drawdown, step_pnl_return, medium_momentum, intraday_sin)")
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
    if bool(getattr(config, "force_disable_sde", False)):
        if config.ppo_use_sde:
            logger.info("[PPO] force_disable_sde=True: overriding use_sde -> False")
        config.ppo_use_sde = False
    logger.info(
        f"[PPO] use_sde={config.ppo_use_sde} "
        f"sde_sample_freq={config.ppo_sde_sample_freq} "
        f"log_std_init={config.ppo_log_std_init}"
    )
    logger.info(
        f"[PPO] investor_beta_policy={bool(getattr(config, 'investor_use_beta_policy', False))} "
        f"beta_epsilon={float(getattr(config, 'investor_beta_epsilon', 1e-6)):.1e}"
    )

    if args.global_norm_mode is not None:
        config.use_global_normalization = bool(args.global_norm_mode == "global")
        logger.info(f"[GLOBAL_NORM] global_norm_mode={args.global_norm_mode} -> use_global_normalization={config.use_global_normalization}")
    if getattr(args, "rolling_past_history_dir", None):
        config.rolling_past_history_dir = str(args.rolling_past_history_dir)
        logger.info(f"[ROLLING_PAST] history_dir={config.rolling_past_history_dir}")

    # Expert blending config is retained only for compatibility with old CLI
    # surfaces; the active Tier1 path no longer uses expert blending.
    config.expert_blend_mode = args.expert_blend_mode
    config.expert_blend_weight = args.expert_blend_weight

    config.forecast_cache_dir = str(getattr(args, "forecast_cache_dir", "forecast_cache"))
    config.enable_forecast_utilization = bool(getattr(args, "enable_forecast_utilization", False))
    forecast_prior_overrides = apply_forecast_prior_overrides(config, args)

    logger.info("\n[TIER1] Enabled: True")
    logger.info(f"  enable_forecast_utilization = {bool(config.enable_forecast_utilization)}")
    if forecast_prior_overrides:
        logger.info(
            "  forecast_prior_overrides = "
            f"{format_forecast_prior_overrides(forecast_prior_overrides)}"
        )

    runtime_contract = build_runtime_contract(
        global_norm_mode="global" if bool(getattr(config, "use_global_normalization", False)) else "rolling_past",
        rolling_past_history_dir=str(getattr(config, "rolling_past_history_dir", "") or ""),
        investment_freq=int(getattr(config, "investment_freq", 6) or 6),
        meta_freq_min=int(getattr(config, "meta_freq_min", 6) or 6),
        meta_freq_max=int(getattr(config, "meta_freq_max", 6) or 6),
        enable_forecast_utilization=bool(getattr(config, "enable_forecast_utilization", False)),
        forecast_prior_settings=forecast_prior_contract_settings(config),
    )
    runtime_contract_hash_value = runtime_contract_hash(runtime_contract)
    config.runtime_contract = runtime_contract
    logger.info(f"[RUNTIME_CONTRACT] hash={runtime_contract_hash_value} payload={runtime_contract}")

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

    logger.info("\n[TIER 1] Forecast utilization runtime: cache-only execution prior")
    logger.info("[TIER 1] Forecast utilization, when enabled, is cache-only execution-prior logic.")
    logger.info("[TIER 1] Investor observations remain 12D.")

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

    # CRITICAL FIX: Create base environment first with proper references
    try:
        # Step 1: Create base environment
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
            config=config,  # Pass config to environment
            log_dir=bootstrap_logs_dir  # Keep bootstrap logs isolated in episode training
        )

        if args.episode_training:
            base_env.calibration_tracker = None
        else:
            base_env.calibration_tracker = None

        # Forecasts are never appended to agent observations in the paper setup.
        # Use the base environment as-is (Tier-1 observation spaces remain unchanged).
        env = base_env
        logger.info("[OK] Using base environment (Tier-1 fixed observations)")

        logger.info("Enhanced environment created successfully!")
        logger.info("   Multi-objective rewards: enabled")
        logger.info(f"   Forecast utilization: {'enabled' if bool(getattr(config, 'enable_forecast_utilization', False)) else 'disabled'}")
        logger.info("   Forecast integration: cache-only execution prior")
        logger.info("   Observations: Tier-1 base (no forecast features)")
        logger.info("   Checkpoint summaries: enabled (saved after each checkpoint)")

        # VALIDATION SUMMARY: Display what forecast runtime is active.
        logger.info("\n" + "="*70)
        logger.info("FORECAST RUNTIME VALIDATION SUMMARY")
        logger.info("="*70)
        
        # Determine active evaluation/training variant (paper modes)
        if False:
            active_variant = "Tier1"
        else:
            active_variant = "Baseline (Tier-1 hybrid RL)"
        logger.info(f"Active Variant:                 {active_variant}")
        
        logger.info("Forecaster loaded:              NO")
        logger.info("Forecast utilization mode:      CACHE_EXECUTION_PRIOR_ONLY")
        logger.info("Forecast runtime checks passed")
        logger.info("="*70 + "\n")
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e

    # Optional one-time validation (safe)
    if args.validate_env:
        try:
            _obs, _ = env.reset()
            logger.info("Env reset OK for validation.")
        except Exception as e:
            logger.warning(f"Env validation reset failed (continuing): {e}")

    # 4) (Optional) HPO (best_params already initialized above)
    if args.optimize and not best_params:
        logger.info("\nRunning hyperparameter optimization...")
        opt_data = data.head(min(5000, len(data)))
        opt_base_env = RenewableMultiAgentEnv(opt_data, config=config)
        # Keep Tier-1 observation spaces fixed for a fair comparison.
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
        logger.info("  ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ [TRAINING_PROOF] Policy updates with loss metrics")
        logger.info("  ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ [DECISION_PROOF] Action sampling and values")
        logger.info("  ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ [EXECUTION_PROOF] Trade execution and position changes")
        logger.info("  ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ [SYSTEM_PROOF] Learning progress and NAV changes")
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

    final_investor_health = dict(locals().get("last_episode_investor_health", {}) or {})
    if not final_investor_health:
        try:
            if hasattr(base_env, "get_investor_health_summary"):
                final_investor_health = dict(base_env.get_investor_health_summary() or {})
        except Exception as e:
            logger.warning(f"Could not capture final investor health summary: {e}")

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
                    'enable_forecast_utilization': bool(getattr(config, 'enable_forecast_utilization', False)),
                    'one_factor_investor_sleeve': True,
                },
                'runtime_contract': runtime_contract,
                'runtime_contract_hash': runtime_contract_hash_value,
                'final_config': {
                    'lr': config.lr,
                    'ent_coef': config.ent_coef,
                    'batch_size': config.batch_size,
                    'max_position_size': getattr(config, 'max_position_size', None),
                    'transaction_cost_bps': getattr(config, 'transaction_cost_bps', None),
                    'transaction_fixed_cost': getattr(config, 'transaction_fixed_cost', None),
                    'no_trade_threshold': getattr(config, 'no_trade_threshold', None),
                    'capital_allocation_fraction': getattr(config, 'capital_allocation_fraction', None),
                    'financial_allocation': getattr(config, 'financial_allocation', None),
                    'ppo_use_sde': getattr(config, 'ppo_use_sde', None),
                    'ppo_log_std_init': getattr(config, 'ppo_log_std_init', None),
                    'investor_use_beta_policy': getattr(config, 'investor_use_beta_policy', None),
                    'investor_beta_epsilon': getattr(config, 'investor_beta_epsilon', None),
                    'investor_clean_reward_contract': getattr(config, 'investor_clean_reward_contract', None),
                    'investor_exposure_action_mode': getattr(config, 'investor_exposure_action_mode', None),
                    'investor_delta_exposure_scale': getattr(config, 'investor_delta_exposure_scale', None),
                    'investor_exposure_power': getattr(config, 'investor_exposure_power', None),
                    'investor_trading_return_weight': getattr(config, 'investor_trading_return_weight', None),
                    'investor_trading_return_clip': getattr(config, 'investor_trading_return_clip', None),
                    'investor_trading_quality_weight': getattr(config, 'investor_trading_quality_weight', None),
                    'investor_trading_drawdown_weight': getattr(config, 'investor_trading_drawdown_weight', None),
                    'investor_trading_cost_weight': getattr(config, 'investor_trading_cost_weight', None),
                    'investor_active_risk_weight': getattr(config, 'investor_active_risk_weight', None),
                    'investor_active_risk_free_band': getattr(config, 'investor_active_risk_free_band', None),
                    'investor_active_risk_vol_mult': getattr(config, 'investor_active_risk_vol_mult', None),
                    'investor_mean_clip_hit_window': getattr(config, 'investor_mean_clip_hit_window', None),
                    'investor_mean_clip_hit_threshold': getattr(config, 'investor_mean_clip_hit_threshold', None),
                    'meta_cap_min': getattr(config, 'meta_cap_min', None),
                    'meta_cap_max': getattr(config, 'meta_cap_max', None),
                    'capital_allocation_fraction': getattr(config, 'capital_allocation_fraction', None),
                    'meta_local_investor_weight': getattr(config, 'meta_local_investor_weight', None),
                    'meta_local_battery_weight': getattr(config, 'meta_local_battery_weight', None),
                    'meta_local_risk_weight': getattr(config, 'meta_local_risk_weight', None),
                    'meta_local_signal_clip': getattr(config, 'meta_local_signal_clip', None),
                    'meta_capital_alignment_weight': getattr(config, 'meta_capital_alignment_weight', None),
                    'risk_controller_rule_based': getattr(config, 'risk_controller_rule_based', None),
                    'meta_controller_rule_based': getattr(config, 'meta_controller_rule_based', None),
                    'risk_base_reward_weight': getattr(config, 'risk_base_reward_weight', None),
                    'meta_base_reward_weight': getattr(config, 'meta_base_reward_weight', None),
                    'agent_policies': config.agent_policies,
                    'net_arch': config.net_arch,
                    'activation_fn': config.activation_fn,
                    'update_every': config.update_every,
                    'minimum_price_filter': getattr(config, 'minimum_price_filter', None),
                    'forecast_horizons': getattr(config, 'forecast_horizons', None),
                    'forecast_cache_dir': str(getattr(config, 'forecast_cache_dir', 'forecast_cache')),
                    'enable_forecast_utilization': bool(getattr(config, 'enable_forecast_utilization', False)),
                    'forecast_prior_window': getattr(config, 'forecast_prior_window', None),
                    'forecast_prior_hit_lcb_z': getattr(config, 'forecast_prior_hit_lcb_z', None),
                    'forecast_prior_default_residual': getattr(config, 'forecast_prior_default_residual', None),
                    'forecast_prior_residual_scale': getattr(config, 'forecast_prior_residual_scale', None),
                    'forecast_prior_edge_gain': getattr(config, 'forecast_prior_edge_gain', None),
                    'forecast_prior_error_hurdle': getattr(config, 'forecast_prior_error_hurdle', None),
                    'forecast_prior_skill_power': getattr(config, 'forecast_prior_skill_power', None),
                    'forecast_prior_max_abs_exposure': getattr(config, 'forecast_prior_max_abs_exposure', None),
                    'forecast_prior_blend': getattr(config, 'forecast_prior_blend', None),
                    'forecast_prior_residual_quantile': getattr(config, 'forecast_prior_residual_quantile', None),
                    'forecast_prior_min_samples': getattr(config, 'forecast_prior_min_samples', None),
                    'forecast_prior_vol_half_life_steps': getattr(config, 'forecast_prior_vol_half_life_steps', None),
                    'forecast_prior_vol_target': getattr(config, 'forecast_prior_vol_target', None),
                    'forecast_prior_horizon_steps': getattr(config, 'forecast_prior_horizon_steps', None),
                    'forecast_prior_denom_floor': getattr(config, 'forecast_prior_denom_floor', None),
                    'forecast_prior_fail_fast': getattr(config, 'forecast_prior_fail_fast', None),
                }
            }, f, indent=2)
        logger.info(f"Training configuration saved to: {cfg_file}")


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
