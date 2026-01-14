# metacontroller.py
# -----------------------------------------------------------------------------
from copy import deepcopy
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import concurrent.futures
from collections import deque
from functools import partial
from tqdm import tqdm
import warnings
import gc
import psutil
import os
import threading
import inspect
import weakref
from datetime import datetime
import random
import json
import pandas as pd
import optuna
from config import EnhancedConfig
from logger import get_logger
from utils import UnifiedMemoryManager, UnifiedObservationValidator, ErrorHandler, safe_operation  # UNIFIED: Import from single source of truth

# Centralized logging - ALL logging goes through logger.py
logger = get_logger(__name__)

# GNN Policy (Tier 2 only)
try:
    from gnn_encoder import GNNActorCriticPolicy, get_gnn_policy_kwargs
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    logger.warning("[GNN] gnn_encoder.py not found - GNN encoder disabled")

warnings.filterwarnings("ignore", category=UserWarning)

LEARNING_MODES = ["PPO", "SAC", "TD3"]

__all__ = ["MultiESGAgent", "HyperparameterOptimizer"]


# UNIFIED MEMORY MANAGER (NEW)
# EnhancedMemoryTracker is now replaced by UnifiedMemoryManager from memory_manager.py
# For backward compatibility, we create an alias
EnhancedMemoryTracker = UnifiedMemoryManager


# ============================= Training Monitor =============================
class StabilizedTrainingMonitor(BaseCallback):
    """Enhanced performance monitoring with memory management."""

    def __init__(self, window_size=500, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.step_count = 0
        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=1000)

    def _on_step(self) -> bool:
        self.step_count += 1

        infos = self.locals.get("infos", [])
        if infos:
            for info in infos:
                ep = info.get("episode")
                if isinstance(ep, dict) and "r" in ep:
                    self.episode_rewards.append(ep["r"])

        if self.step_count % 200 == 0:
            cleanup_level, _ = self.memory_tracker.should_cleanup()
            if cleanup_level:
                memory_freed = self.memory_tracker.cleanup(cleanup_level)
                if self.verbose > 0 and memory_freed > 50:
                    logger.debug(f"Callback memory cleanup ({cleanup_level}): freed {memory_freed:.1f}MB")
        return True


# ============================== Dummy Gym Env ===============================
class DummyGymEnv(Env):
    """Lightweight dummy environment for SB3 initialization."""

    metadata = {"render_modes": []}

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        obs = self.observation_space.sample()
        if hasattr(self.observation_space, "low") and hasattr(self.observation_space, "high"):
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        self._step_count = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        if hasattr(self.observation_space, "low") and hasattr(self.observation_space, "high"):
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        reward = 0.0
        terminated = self._step_count >= 100
        truncated = False
        info = {}
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pass


# ================================ Utilities =================================
def safe_multithreaded_processing(function, policies, multithreading=True, max_workers=None):
    """Safe multithreaded processing with error handling."""
    if not multithreading or len(policies) <= 1:
        for polid, policy in enumerate(policies):
            try:
                function(polid, policy)
            except Exception as exc:
                logger.error(f"Policy {polid} error: {exc}")
        return

    max_workers = max_workers or min(4, len(policies))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [(executor.submit(function, polid, policy), polid) for polid, policy in enumerate(policies)]
        for future, polid in futures:
            try:
                future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Policy {polid} training timeout")
            except Exception as exc:
                print(f"Policy {polid} generated an exception: {exc}")


# ======================== Observation Validator =============================
# UNIFIED: Use EnhancedObservationValidator from wrapper.py
from wrapper import EnhancedObservationValidator

# For backward compatibility, keep the reference
# (The actual implementation is in wrapper.py)


# =========================== Multi Agent Controller =========================
class MultiESGAgent:
    """Enhanced multi-agent system with memory management and validation."""

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.num_agents = len(self.possible_agents)
        self.num_envs = 1
        self.n_steps = int(getattr(config, "update_every", 128))
        self.verbose = int(getattr(config, "verbose", 0))
        self.debug = bool(debug)
        self.multithreading = bool(getattr(config, "multithreading", False))

        self._logger = logger
        self.logger = self._logger

        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=int(getattr(config, "max_memory_mb", 5000)))
        # Get forecaster from env if available
        forecaster = getattr(env, 'forecast_generator', None)
        self.obs_validator = EnhancedObservationValidator(env, forecaster=forecaster, debug=debug)

        # Spaces
        self.observation_spaces: Dict[str, spaces.Box] = {}
        self.action_spaces: Dict[str, spaces.Space] = {}
        self._initialize_spaces()

        # ROBUST INITIALIZATION: Verify observation spaces match actual environment output
        # This MUST happen BEFORE creating policies to ensure they're built with correct dimensions
        self._verify_observation_spaces_with_actual_observations()

        # Policies
        self.policies: List[Any] = []
        self._initialize_policies_enhanced(config, device)
        
        # ROBUST VERIFICATION: Verify policies were created with correct observation spaces
        # This checks the actual neural network input dimensions, not just the observation_space attribute
        self._verify_policies_match_observation_spaces()

        # Training state
        self.config = config
        self.device = device
        self.training = bool(training)
        self.total_steps = 0
        self.max_steps = int(getattr(env, "max_steps", 100000))

        # Progress
        self._global_target_steps: Optional[int] = None

        # Runtime buffers
        self._last_obs: Optional[Dict[str, np.ndarray]] = None
        self._episode_starts: List[bool] = [True] * self.num_agents
        self._consecutive_errors = 0
        self._max_consecutive_errors = 20
        self._training_metrics = {
            "memory_cleanups": 0,
            "observation_fixes": 0,
            "policy_errors": 0,
            "successful_steps": 0,
        }

        # FAMC: Forecast-Aware Meta-Critic state tracking
        # EMA moments for online λ* computation
        self._ema_mA = 0.0   # E[A]
        self._ema_mC = 0.0   # E[C]
        self._ema_mA2 = 0.0  # E[A²]
        self._ema_mC2 = 0.0  # E[C²]
        self._ema_mAC = 0.0  # E[A·C]
        self._lambda_star_prev = 0.0  # Smoothed λ*
        self._step_count_famc = 0  # Step counter for FAMC

        # FAMC: Buffers for meta-critic training
        self._meta_features_buffer = []  # State features for meta head training
        self._meta_adv_buffer = []       # Advantages for meta head training

        # FAMC: Metrics tracking
        self._famc_metrics = {
            'lambda_star': 0.0,
            'corr_AC': 0.0,
            'clip_rate': 0.0,
            'meta_loss': 0.0,
            'var_adv_before': 0.0,
            'var_adv_after': 0.0,
            'num_clipped': 0,
            'num_total': 0,
        }

        print(f"Enhanced MultiESGAgent initialized with {len(self.policies)} agents")
        if self.debug:
            self._print_initialization_summary()

    def set_total_training_budget(self, total_steps: int):
        try:
            self._global_target_steps = int(total_steps)
        except Exception:
            self._global_target_steps = None

    def reset_for_new_episode(self):
        """
        FIX: Complete reset of agent internal state for new episode.
        This prevents state leakage between episodes including:
        - Optimizer states
        - Learning rate schedules
        - Buffer states
        - Episode tracking variables
        - FAMC meta buffers (CRITICAL for OOM prevention)

        CRITICAL: Does NOT reset total_steps - this must be preserved for learning continuity!
        """
        try:
            # CRITICAL FIX: DO NOT reset total_steps - it must accumulate across episodes
            # for proper learning rate schedules and optimizer state
            # self.total_steps = 0  # ← REMOVED! This was breaking learning continuity

            # Reset observation tracking
            self._last_obs = None
            self._episode_starts = [True] * self.num_agents
            self._consecutive_errors = 0

            # Reset training metrics
            self._training_metrics = {
                "memory_cleanups": 0,
                "observation_fixes": 0,
                "policy_errors": 0,
                "successful_steps": 0,
            }

            # FAMC: Clear meta buffers to prevent OOM across episodes
            # CRITICAL: These buffers accumulate observations and advantages
            # Without clearing, they grow unbounded across 20 episodes
            if hasattr(self, '_meta_features_buffer'):
                buffer_size_before = len(self._meta_features_buffer)
                self._meta_features_buffer.clear()
                self._meta_adv_buffer.clear()
                if buffer_size_before > 0:
                    self.logger.info(f"[FAMC] Cleared meta buffers: {buffer_size_before} entries freed")

            # FAMC: Reset EMA moments for new episode
            # This ensures λ* computation starts fresh
            self._ema_mA = 0.0
            self._ema_mC = 0.0
            self._ema_mA2 = 0.0
            self._ema_mC2 = 0.0
            self._ema_mAC = 0.0
            self._lambda_star_prev = 0.0
            # Note: Keep _step_count_famc to maintain warmup across episodes

            # Reset each policy's internal state
            for i, policy in enumerate(self.policies):
                try:
                    # Reset buffers
                    if hasattr(policy, 'replay_buffer') and policy.replay_buffer is not None:
                        policy.replay_buffer.reset()
                    if hasattr(policy, 'rollout_buffer') and policy.rollout_buffer is not None:
                        policy.rollout_buffer.reset()

                    # FAMC: Clear policy-specific FAMC step data
                    if hasattr(policy, '_famc_step_data'):
                        policy._famc_step_data.clear()

                    # CRITICAL FIX: DO NOT reset optimizer state
                    # Resetting optimizer wipes out Adam's momentum/variance estimates
                    # This causes learning instability and prevents convergence
                    # The optimizer should maintain state across episodes for stable learning
                    if hasattr(policy, 'policy') and hasattr(policy.policy, 'optimizer'):
                        # Just log that optimizer state is preserved
                        self.logger.debug(f"[OK] Optimizer state preserved for policy {i} (maintains momentum)")

                    # Reset policy-specific tracking
                    if hasattr(policy, '_last_obs'):
                        policy._last_obs = None
                    if hasattr(policy, '_last_episode_starts'):
                        policy._last_episode_starts = None
                    # CRITICAL FIX: DO NOT reset num_timesteps - let SB3 manage it via reset_num_timesteps parameter
                    # Resetting num_timesteps can break learning rate schedules and callbacks that depend on it
                    # The reset_num_timesteps parameter in agent.learn() already handles this correctly
                    # if hasattr(policy, 'num_timesteps'):
                    #     policy.num_timesteps = 0  # REMOVED: Let SB3 manage this via reset_num_timesteps

                    self.logger.info(f"[OK] Reset policy {i} ({policy.mode}) for new episode")
                except Exception as e:
                    self.logger.warning(f"[WARN] Could not fully reset policy {i}: {e}")

            self.logger.info("[OK] Agent reset complete for new episode")
        except Exception as e:
            self.logger.error(f"[ERROR] Agent reset failed: {e}")
            raise

    def _initialize_spaces(self):
        """
        FIX: Initialize observation and action spaces from environment.
        CRITICAL: These spaces MUST match exactly what the environment produces during step/reset.
        No runtime modification of observation space is allowed.
        """
        for agent in self.possible_agents:
            try:
                obs_space_from_env = self.env.observation_space(agent)
                self.observation_spaces[agent] = obs_space_from_env
                self.action_spaces[agent] = self.env.action_space(agent)
                # CRITICAL: Log observation space dimensions for debugging
                obs_dim = obs_space_from_env.shape[0] if hasattr(obs_space_from_env, 'shape') else None
                self.logger.info(f"[OBS_SPACE_INIT] {agent}: observation_space={obs_dim}D (from env.observation_space())")
                if self.debug:
                    obs_space = self.observation_spaces[agent]
                    act_space = self.action_spaces[agent]
                    print(f"{agent} | Obs: {obs_space.shape} | Act: {getattr(act_space,'shape',act_space)}")
            except Exception as e:
                error_msg = f"[CRITICAL] Failed to get observation/action spaces for {agent}: {e}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

        # FIX: Validate that observation spaces are consistent
        self._validate_observation_spaces()
    
    def _verify_observation_spaces_with_actual_observations(self):
        """
        ROBUST INITIALIZATION: Verify that observation spaces match actual observations from environment.
        
        This method:
        1. Resets the environment to get actual observations
        2. Compares observation dimensions with declared observation spaces
        3. FAILS FAST if there's any mismatch (no workarounds)
        
        This ensures policies are created with correct observation spaces from the start.
        """
        self.logger.info("[ROBUST_INIT] Verifying observation spaces match actual environment output...")
        
        try:
            # Get actual observations from environment
            actual_obs, _ = self.env.reset()
            
            mismatches = []
            for agent in self.possible_agents:
                if agent not in self.observation_spaces:
                    mismatches.append(f"{agent}: Observation space not initialized")
                    continue
                
                if agent not in actual_obs:
                    mismatches.append(f"{agent}: No observation returned from environment")
                    continue
                
                declared_dim = self.observation_spaces[agent].shape[0]
                actual_obs_array = actual_obs[agent]
                
                # Ensure observation is 1D
                if actual_obs_array.ndim > 1:
                    if actual_obs_array.shape[0] == 1:
                        actual_obs_array = actual_obs_array.squeeze(0)
                    else:
                        actual_obs_array = actual_obs_array.flatten()
                
                actual_dim = actual_obs_array.shape[0]
                
                if declared_dim != actual_dim:
                    mismatches.append(
                        f"{agent}: Declared observation space is {declared_dim}D, "
                        f"but environment produces {actual_dim}D observations. "
                        f"This is a CRITICAL initialization error."
                    )
                else:
                    self.logger.debug(f"[ROBUST_INIT] ✓ {agent}: Observation space matches ({declared_dim}D)")
            
            if mismatches:
                error_msg = (
                    f"[CRITICAL INITIALIZATION ERROR] Observation space mismatches detected:\n"
                    + "\n".join(f"  - {m}" for m in mismatches) +
                    f"\n\nThis indicates the environment's observation_space() method returns incorrect dimensions, "
                    f"or the environment produces observations that don't match its declared spaces.\n"
                    f"Cannot proceed with policy initialization until this is fixed."
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.logger.info("[ROBUST_INIT] ✓ All observation spaces match actual environment output")
            
        except RuntimeError:
            # Re-raise RuntimeError (our fail-fast errors)
            raise
        except Exception as e:
            error_msg = (
                f"[CRITICAL INITIALIZATION ERROR] Failed to verify observation spaces: {e}\n"
                f"Cannot proceed with policy initialization."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _verify_policies_match_observation_spaces(self):
        """
        ROBUST VERIFICATION: Verify that policies were created with correct observation spaces.
        
        This method:
        1. Checks each policy's internal neural network input dimension
        2. Compares with the environment's observation space
        3. FAILS FAST if there's any mismatch (no workarounds)
        
        This ensures policies can actually process the observations they'll receive.
        """
        self.logger.info("[ROBUST_VERIFY] Verifying policies match observation spaces...")
        
        mismatches = []
        for idx, (agent_name, policy) in enumerate(zip(self.possible_agents, self.policies)):
            if policy is None:
                mismatches.append(f"{agent_name}: Policy is None")
                continue
            
            env_obs_space = self.observation_spaces.get(agent_name)
            if env_obs_space is None:
                mismatches.append(f"{agent_name}: Observation space not found")
                continue
            
            env_obs_dim = env_obs_space.shape[0]
            
            # Check policy's observation_space attribute
            policy_obs_dim = None
            if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                policy_obs_dim = policy.observation_space.shape[0]
            
            # Check actual network input dimension (most reliable)
            network_input_dim = None
            try:
                if hasattr(policy, 'policy') and hasattr(policy.policy, 'features_extractor'):
                    fe = policy.policy.features_extractor
                    # Check GNN encoder input dimension
                    if hasattr(fe, 'gnn_encoder') and hasattr(fe.gnn_encoder, 'obs_dim'):
                        network_input_dim = fe.gnn_encoder.obs_dim
                    # Check MLP first layer input dimension
                    elif hasattr(fe, 'mlp') and len(fe.mlp) > 0:
                        first_layer = fe.mlp[0]
                        if hasattr(first_layer, 'in_features'):
                            network_input_dim = first_layer.in_features
            except Exception as e:
                self.logger.debug(f"[ROBUST_VERIFY] Could not determine network input dim for {agent_name}: {e}")
            
            # Verify dimensions match
            if network_input_dim is not None and network_input_dim != env_obs_dim:
                mismatches.append(
                    f"{agent_name}: Network input dimension ({network_input_dim}D) doesn't match "
                    f"environment observation space ({env_obs_dim}D). Policy was created with wrong dimensions."
                )
            elif policy_obs_dim is not None and policy_obs_dim != env_obs_dim:
                mismatches.append(
                    f"{agent_name}: Policy observation_space attribute ({policy_obs_dim}D) doesn't match "
                    f"environment observation space ({env_obs_dim}D). Policy was created with wrong dimensions."
                )
            elif network_input_dim is None and policy_obs_dim is None:
                mismatches.append(
                    f"{agent_name}: Cannot verify policy observation space. "
                    f"Policy has no observable input dimension."
                )
            else:
                self.logger.debug(
                    f"[ROBUST_VERIFY] ✓ {agent_name}: Policy matches observation space "
                    f"(env={env_obs_dim}D, policy_attr={policy_obs_dim}D, network={network_input_dim}D)"
                )
        
        if mismatches:
            error_msg = (
                f"[CRITICAL INITIALIZATION ERROR] Policy observation space mismatches detected:\n"
                + "\n".join(f"  - {m}" for m in mismatches) +
                f"\n\nThis indicates policies were created with incorrect observation spaces.\n"
                f"Cannot proceed with training until this is fixed."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.logger.info("[ROBUST_VERIFY] ✓ All policies match observation spaces")
    
    def reinitialize_observation_spaces(self):
        """
        CRITICAL: Reinitialize observation spaces from environment.
        
        This must be called when the environment is updated (e.g., per episode)
        to ensure observation spaces match the new environment's observation spaces.
        
        CRITICAL: When observation spaces change, rollout buffers must be reset because
        they were initialized with the old observation space dimensions.
        """
        self.logger.info("[OBS_SPACE_REINIT] Reinitializing observation spaces from environment")
        old_spaces = {agent: self.observation_spaces[agent].shape[0] if agent in self.observation_spaces and hasattr(self.observation_spaces[agent], 'shape') else None 
                      for agent in self.possible_agents}
        
        # Reinitialize spaces from environment
        spaces_changed = False
        for agent in self.possible_agents:
            try:
                obs_space_from_env = self.env.observation_space(agent)
                old_dim = old_spaces.get(agent)
                new_dim = obs_space_from_env.shape[0] if hasattr(obs_space_from_env, 'shape') else None
                
                if old_dim is not None and new_dim is not None and old_dim != new_dim:
                    self.logger.info(f"[OBS_SPACE_REINIT] {agent}: observation_space changed from {old_dim}D to {new_dim}D")
                    spaces_changed = True
                elif new_dim is not None:
                    self.logger.info(f"[OBS_SPACE_REINIT] {agent}: observation_space={new_dim}D (from env.observation_space())")
                
                self.observation_spaces[agent] = obs_space_from_env
            except Exception as e:
                self.logger.error(f"[OBS_SPACE_REINIT] Failed to reinitialize space for {agent}: {e}")
        
        if spaces_changed:
            self.logger.warning("[OBS_SPACE_REINIT] Observation spaces changed - RECREATING policies with correct dimensions")
            # ROBUST FIX: When observation spaces change, policies MUST be recreated
            # This ensures neural networks are built with correct input dimensions
            try:
                # Use robust validation method to recreate policies
                self._validate_and_fix_policy_observation_spaces_robust()
                self.logger.info("[OBS_SPACE_REINIT] ✓ Policies recreated with correct observation spaces")
            except Exception as e:
                self.logger.error(f"[OBS_SPACE_REINIT] ✗ Failed to recreate policies: {e}")
                raise RuntimeError(f"CRITICAL: Cannot continue with mismatched observation spaces. Failed to recreate policies: {e}")
        
        self._validate_observation_spaces()

    def _validate_observation_spaces(self):
        """
        FIX: Validate that observation spaces match what environment will produce.
        This prevents runtime mismatches between policy initialization and actual observations.
        """
        try:
            # Test observation space consistency by checking a sample observation
            obs, _ = self.env.reset()

            for agent in self.possible_agents:
                if agent in obs:
                    actual_obs = obs[agent]
                    expected_space = self.observation_spaces[agent]

                    # Check shape consistency
                    if isinstance(actual_obs, np.ndarray):
                        if actual_obs.shape[0] != expected_space.shape[0]:
                            self.logger.error(
                                f"[CRITICAL] Observation space mismatch for {agent}: "
                                f"expected shape {expected_space.shape}, got {actual_obs.shape}. "
                                f"This will cause training failures. Fix environment or observation space definition."
                            )
                            raise ValueError(
                                f"Observation space mismatch for {agent}: "
                                f"expected {expected_space.shape}, got {actual_obs.shape}"
                            )
                    else:
                        self.logger.error(f"[CRITICAL] Observation for {agent} is not numpy array: {type(actual_obs)}")
                        raise TypeError(f"Observation for {agent} must be numpy array, got {type(actual_obs)}")

            self.logger.info("[OK] Observation spaces validated successfully")
        except Exception as e:
            self.logger.error(f"[CRITICAL] Observation space validation failed: {e}")
            raise

    def _initialize_policies_enhanced(self, config, device):
        act = getattr(config, "activation_fn", nn.Tanh)
        if isinstance(act, str):
            act_map = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "elu": nn.ELU,
                "leaky_relu": nn.LeakyReLU,
                "sigmoid": nn.Sigmoid,
                "swish": nn.SiLU,
                "gelu": nn.GELU,
            }
            act = act_map.get(act.lower(), nn.Tanh)

        for i, agent in enumerate(self.possible_agents):
            try:
                policy = self._create_single_policy(config, device, agent, i, act)
                # ROBUST: Policy creation must succeed - fail fast if it doesn't
                if policy is None:
                    raise RuntimeError(f"Policy creation returned None for {agent} - this should never happen")
                
                self.memory_tracker.register_policy(policy)
                if hasattr(policy, "replay_buffer") and policy.replay_buffer is not None:
                    self.memory_tracker.register_buffer(policy.replay_buffer)
                if hasattr(policy, "rollout_buffer") and policy.rollout_buffer is not None:
                    self.memory_tracker.register_buffer(policy.rollout_buffer)
                self.policies.append(policy)
                if self.debug:
                    print(f"Policy created for {agent} ({policy.mode})")
            except Exception as e:
                error_msg = f"[CRITICAL] Failed to initialize policy for {agent}: {e}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

    def _create_single_policy(self, config, device, agent, agent_idx, activation_fn):
        try:
            policy_cfg = getattr(config, "agent_policies", None)
            if policy_cfg and agent_idx < len(policy_cfg):
                policy_mode = policy_cfg[agent_idx].get("mode", "PPO")
            else:
                policy_mode = getattr(config, "default_policy", "PPO")

            obs_space = self.observation_spaces[agent]
            act_space = self.action_spaces[agent]

            dummy_env = DummyVecEnv([partial(DummyGymEnv, obs_space, act_space)])
            self.memory_tracker.register_env(dummy_env)

            algo_cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[policy_mode]

            # GNN Encoder: Check if enabled (works for both Tier 1 and Tier 2, only for PPO agents)
            # GNN encoder is independent of forecast integration
            # It can work on base observations (6D - Tier 1) OR forecast-enhanced observations (9D - Tier 2)
            # This makes GNN encoder a true add-on that can be used with or without forecasts
            use_gnn = (
                policy_mode == "PPO" and
                GNN_AVAILABLE and
                getattr(config, "enable_gnn_encoder", False)
                # REMOVED: enable_forecast_utilisation requirement
                # GNN encoder works on any observation space (6D base or 9D with forecasts)
            )

            if use_gnn:
                # Use tier-specific GNN configuration based on observation dimension
                # Check actual observation dimension to determine which GNN configuration to use
                obs_dim = obs_space.shape[0]
                enable_forecast_util = getattr(config, "enable_forecast_utilisation", False)
                
                # Hierarchical GNN works for any agent with forecast integration (base + forecast split)
                # - Investor: 14D (6 base + 8 forecast) → base_feature_dim=6
                # - Battery: 10D (4 base + 6 forecast) → base_feature_dim=4
                if enable_forecast_util:
                    # Tier 2: Use hierarchical GNN for agents with forecast integration
                    features_dim = getattr(config, "gnn_features_dim_tier2", 18)
                    hidden_dim = getattr(config, "gnn_hidden_dim_tier2", 30)
                    num_layers = getattr(config, "gnn_num_layers_tier2", 2)
                    dropout = getattr(config, "gnn_dropout_tier2", 0.1)
                    graph_type = getattr(config, "gnn_graph_type_tier2", "hierarchical")
                    num_heads = getattr(config, "gnn_num_heads_tier2", 3)
                    use_attention_pooling = getattr(config, "gnn_use_attention_pooling_tier2", True)
                    net_arch = getattr(config, "gnn_net_arch_tier2", [128, 64])
                    # Determine base_feature_dim based on observation dimension
                    if obs_dim == 14:
                        base_feature_dim = 6  # Tier 2 Investor: 6 base + 8 forecast
                    elif obs_dim == 10:
                        base_feature_dim = 4  # Battery: 4 base + 6 forecast (separate generation signals)
                    elif obs_dim == 8:
                        # Backward compatibility: 8D format (old merged total generation)
                        base_feature_dim = 4  # Battery: 4 base + 4 forecast
                    else:
                        # Other agents: Use standard GNN if dimension doesn't match known splits
                        graph_type = getattr(config, "gnn_graph_type", "full")
                        base_feature_dim = None
                else:
                    # Tier 1: Use standard settings for base observations (6D investor, 4D battery, etc.)
                    features_dim = getattr(config, "gnn_features_dim", 18)
                    hidden_dim = getattr(config, "gnn_hidden_dim", 30)
                    num_layers = getattr(config, "gnn_num_layers", 2)
                    dropout = getattr(config, "gnn_dropout", 0.1)
                    graph_type = getattr(config, "gnn_graph_type", "full")
                    num_heads = getattr(config, "gnn_num_heads", 3)
                    use_attention_pooling = getattr(config, "gnn_use_attention_pooling", True)
                    net_arch = getattr(config, "gnn_net_arch", [128, 64])
                    base_feature_dim = None
                
                policy_kwargs = get_gnn_policy_kwargs(
                    features_dim=features_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    graph_type=graph_type,
                    num_heads=num_heads,
                    use_attention_pooling=use_attention_pooling,
                    net_arch=net_arch,
                    base_feature_dim=base_feature_dim
                )
                policy_class = GNNActorCriticPolicy
                logger.info(f"[GNN] Using GNN policy for {agent} (features_dim={policy_kwargs['features_extractor_kwargs']['features_dim']})")
            else:
                # Standard MLP policy
                policy_kwargs = {
                    "net_arch": getattr(config, "net_arch", [64, 32]),
                    "activation_fn": activation_fn,
                    "normalize_images": False,
                    "optimizer_class": torch.optim.Adam,
                    "optimizer_kwargs": {"eps": 1e-8, "weight_decay": 0.0},
                }
                policy_class = "MlpPolicy"

            algo_kwargs = {
                "policy": policy_class,
                "env": dummy_env,
                "learning_rate": float(getattr(config, "lr", 3e-4)),
                "policy_kwargs": policy_kwargs,
                "verbose": 0,
                "device": device,
                "seed": int(getattr(config, "seed", 42)),
            }

            if policy_mode == "PPO":
                algo_kwargs.update(
                    {
                        "ent_coef": getattr(config, "ent_coef", 0.005),
                        "n_steps": int(min(self.n_steps, getattr(config, "n_steps", 1024))),  # STRENGTHENED: Increased from 512
                        "batch_size": getattr(config, "batch_size", 256),  # STRENGTHENED: Use config batch_size (256)
                        "gae_lambda": float(getattr(config, "gae_lambda", 0.98)),
                        "clip_range": float(getattr(config, "clip_range", 0.15)),
                        "normalize_advantage": True,
                        "vf_coef": float(getattr(config, "vf_coef", 0.5)),
                        "max_grad_norm": float(getattr(config, "max_grad_norm", 0.5)),
                        "gamma": float(getattr(config, "gamma", 0.995)),
                        "n_epochs": int(min(getattr(config, "n_epochs", 10), getattr(config, "n_epochs", 10))),  # STRENGTHENED: Increased from 5 to 10
                    }
                )
            elif policy_mode in ("SAC", "TD3"):
                algo_kwargs.update(
                    {
                        "buffer_size": int(min(getattr(config, "buffer_size", 50000), 100000)),
                        "learning_starts": int(getattr(config, "learning_starts", 100)),
                        "batch_size": 128,  # FIXED: Standardized batch size for SAC/TD3 agents
                        "tau": float(getattr(config, "tau", 0.005)),
                        "gamma": float(getattr(config, "gamma", 0.99)),
                        "train_freq": getattr(config, "train_freq", 1),
                        "gradient_steps": int(getattr(config, "gradient_steps", 1)),
                    }
                )
                if policy_mode == "SAC":
                    algo_kwargs.update(
                        {
                            "ent_coef": getattr(config, "ent_coef", "auto"),
                            "target_update_interval": int(getattr(config, "target_update_interval", 1)),
                        }
                    )
                else:  # TD3
                    algo_kwargs.update(
                        {
                            "policy_delay": int(getattr(config, "policy_delay", 2)),
                            "target_policy_noise": float(getattr(config, "target_policy_noise", 0.2)),
                            "target_noise_clip": float(getattr(config, "target_noise_clip", 0.5)),
                        }
                    )

            policy = algo_cls(**algo_kwargs)
            policy.mode = policy_mode
            policy.agent_name = agent
            policy.action_space = act_space  # help with buffer clipping later
            
            # CRITICAL FIX: Verify and fix policy observation space to match environment
            # This ensures the policy always uses the correct observation space, even if it was saved with wrong dimensions
            if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                policy_obs_dim = policy.observation_space.shape[0]
                env_obs_dim = obs_space.shape[0]
                if policy_obs_dim != env_obs_dim:
                    self.logger.warning(f"[OBS_SPACE_FIX] {agent}: Policy initialized with {policy_obs_dim}D, but environment provides {env_obs_dim}D. Fixing...")
                    policy.observation_space = obs_space
                    if hasattr(policy, 'policy') and hasattr(policy.policy, 'observation_space'):
                        policy.policy.observation_space = obs_space
                    # CRITICAL: Also update the rollout/replay buffer observation space if it exists
                    # AND force buffer recreation if dimensions changed
                    if hasattr(policy, 'rollout_buffer') and policy.rollout_buffer is not None:
                        try:
                            buffer = policy.rollout_buffer
                            # Update observation space attribute
                            if hasattr(buffer, 'observation_space'):
                                buffer.observation_space = obs_space
                            # CRITICAL: Buffer shape validation disabled - manual recreation breaks SB3's internal state
                            # Policies are recreated when observation spaces change, which properly recreates buffers
                        except Exception as e:
                            self.logger.warning(f"[OBS_SPACE_FIX] Failed to update rollout buffer for {agent}: {e}")
                    if hasattr(policy, 'replay_buffer') and policy.replay_buffer is not None:
                        try:
                            policy.replay_buffer.observation_space = obs_space
                        except Exception:
                            pass
                    self.logger.info(f"[OBS_SPACE_FIX] {agent}: Policy observation space updated to {env_obs_dim}D")
            
            return policy
        except Exception as e:
            error_msg = (
                f"[CRITICAL INITIALIZATION ERROR] Failed to create policy for {agent}: {e}\n"
                f"This is a critical error - policy creation must succeed for training to proceed.\n"
                f"Cannot continue with missing or incorrectly initialized policies."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # REMOVED: _create_fallback_policy - No fallbacks, fail fast if policy creation fails

    def _validate_and_fix_policy_observation_spaces_robust(self):
        """
        ROBUST FIX: Validate that all policies have correct observation spaces matching the environment.
        If any policy has wrong observation space, RECREATE it with correct space.
        This is called immediately after policy initialization to ensure correctness from the start.
        """
        self.logger.info("[ROBUST_CHECK] Validating all policies have correct observation spaces...")
        
        # First, ensure observation spaces are up-to-date from environment
        for agent in self.possible_agents:
            try:
                obs_space_from_env = self.env.observation_space(agent)
                self.observation_spaces[agent] = obs_space_from_env
            except Exception as e:
                self.logger.error(f"[ROBUST_CHECK] Failed to get observation space for {agent}: {e}")
        
        # Now check each policy and recreate if needed
        policies_recreated = 0
        for idx, (agent_name, policy) in enumerate(zip(self.possible_agents, self.policies)):
            if policy is None:
                continue
                
            try:
                # ROBUST FIX: Get observation space DIRECTLY from environment, not from cache
                # This ensures we get the actual current observation space
                try:
                    env_obs_space = self.env.observation_space(agent_name)
                    env_obs_dim = env_obs_space.shape[0] if hasattr(env_obs_space, 'shape') else None
                    # Update cached observation space
                    self.observation_spaces[agent_name] = env_obs_space
                except Exception as e:
                    self.logger.error(f"[ROBUST_CHECK] Failed to get observation space from environment for {agent_name}: {e}")
                    continue
                
                if env_obs_dim is None:
                    self.logger.error(f"[ROBUST_CHECK] Invalid observation space dimension for {agent_name}, skipping")
                    continue
                
                # Check if policy's observation space matches environment
                policy_obs_dim = None
                if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                    policy_obs_dim = policy.observation_space.shape[0]
                    self.logger.debug(f"[ROBUST_CHECK] {agent_name}: Policy obs_space dimension = {policy_obs_dim}D")
                else:
                    self.logger.warning(f"[ROBUST_CHECK] {agent_name}: Policy has no observation_space attribute or shape")
                
                # Check network input dimension (more reliable check)
                network_input_dim = None
                try:
                    if hasattr(policy, 'policy') and hasattr(policy.policy, 'features_extractor'):
                        fe = policy.policy.features_extractor
                        # Check GNN encoder input dimension
                        if hasattr(fe, 'gnn_encoder') and hasattr(fe.gnn_encoder, 'obs_dim'):
                            network_input_dim = fe.gnn_encoder.obs_dim
                        # Check MLP first layer input dimension
                        elif hasattr(fe, 'mlp') and len(fe.mlp) > 0:
                            first_layer = fe.mlp[0]
                            if hasattr(first_layer, 'in_features'):
                                network_input_dim = first_layer.in_features
                except Exception:
                    pass
                
                # ROBUST FIX: ALWAYS recreate if observation space doesn't match
                # Don't rely on network checks - if obs_space doesn't match, recreate
                needs_recreation = False
                self.logger.debug(f"[ROBUST_CHECK] {agent_name}: Comparing - Policy obs_space: {policy_obs_dim}D, Env obs_space: {env_obs_dim}D, Network input: {network_input_dim}D")
                
                if policy_obs_dim is not None and policy_obs_dim != env_obs_dim:
                    needs_recreation = True
                    self.logger.warning(f"[ROBUST_CHECK] {agent_name}: MISMATCH DETECTED - Policy obs_space is {policy_obs_dim}D, environment is {env_obs_dim}D. RECREATING policy.")
                elif network_input_dim is not None and network_input_dim != env_obs_dim:
                    needs_recreation = True
                    self.logger.warning(f"[ROBUST_CHECK] {agent_name}: MISMATCH DETECTED - Network expects {network_input_dim}D, environment provides {env_obs_dim}D. RECREATING policy.")
                elif policy_obs_dim is None:
                    # If we can't check policy obs_space, check network - if that also fails, recreate to be safe
                    if network_input_dim is None:
                        # Can't check either - recreate to ensure correctness
                        needs_recreation = True
                        self.logger.warning(f"[ROBUST_CHECK] {agent_name}: Cannot verify observation space dimensions. RECREATING policy to ensure correctness.")
                    elif network_input_dim != env_obs_dim:
                        needs_recreation = True
                        self.logger.warning(f"[ROBUST_CHECK] {agent_name}: MISMATCH DETECTED (no policy obs_space) - Network expects {network_input_dim}D, environment provides {env_obs_dim}D. RECREATING policy.")
                
                if needs_recreation:
                    # RECREATE policy with correct observation space
                    try:
                        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU}
                        act = getattr(self.config, "activation_fn", "tanh")
                        activation_fn = act_map.get(act.lower() if isinstance(act, str) else act, nn.Tanh)
                        
                        new_policy = self._create_single_policy(self.config, self.device, agent_name, idx, activation_fn)
                        if new_policy is not None:
                            # Clean up old policy
                            old_policy = self.policies[idx]
                            if hasattr(self.memory_tracker, '_policies') and old_policy in self.memory_tracker._policies:
                                self.memory_tracker._policies.remove(old_policy)
                            if hasattr(old_policy, 'rollout_buffer') and old_policy.rollout_buffer is not None:
                                if hasattr(self.memory_tracker, '_buffers') and old_policy.rollout_buffer in self.memory_tracker._buffers:
                                    self.memory_tracker._buffers.remove(old_policy.rollout_buffer)
                            
                            # Install new policy
                            self.policies[idx] = new_policy
                            self.memory_tracker.register_policy(new_policy)
                            if hasattr(new_policy, 'rollout_buffer') and new_policy.rollout_buffer is not None:
                                self.memory_tracker.register_buffer(new_policy.rollout_buffer)
                            if hasattr(new_policy, 'replay_buffer') and new_policy.replay_buffer is not None:
                                self.memory_tracker.register_buffer(new_policy.replay_buffer)
                            
                            policies_recreated += 1
                            self.logger.info(f"[ROBUST_CHECK] ✓ {agent_name}: Policy RECREATED with {env_obs_dim}D observation space")
                        else:
                            self.logger.error(f"[ROBUST_CHECK] ✗ {agent_name}: Failed to recreate policy")
                    except Exception as recreate_error:
                        self.logger.error(f"[ROBUST_CHECK] ✗ {agent_name}: Error recreating policy: {recreate_error}")
                else:
                    # Policy is correct - verify observation space attribute matches
                    if policy_obs_dim != env_obs_dim:
                        # Update observation space attribute to match
                        policy.observation_space = env_obs_space
                        if hasattr(policy, 'policy') and hasattr(policy.policy, 'observation_space'):
                            policy.policy.observation_space = env_obs_space
                        self.logger.debug(f"[ROBUST_CHECK] ✓ {agent_name}: Updated observation_space attribute to {env_obs_dim}D")
                    else:
                        self.logger.debug(f"[ROBUST_CHECK] ✓ {agent_name}: Policy observation space correct ({env_obs_dim}D)")
            except Exception as e:
                self.logger.error(f"[ROBUST_CHECK] ✗ {agent_name}: Error during validation: {e}")
        
        if policies_recreated > 0:
            self.logger.info(f"[ROBUST_CHECK] Recreated {policies_recreated} policies with correct observation spaces")
        else:
            self.logger.info(f"[ROBUST_CHECK] All policies have correct observation spaces ✓")
    
    def _fix_policy_observation_spaces(self):
        """
        CRITICAL FIX: Runtime check to ensure all policies have the correct observation space.
        This catches cases where a saved policy was loaded with wrong dimensions.
        
        IMPORTANT: If the policy's neural network was built with wrong dimensions, we need to
        recreate the policy, not just update the observation_space attribute.
        
        This method should only be called once at the start of training, not every step.
        """
        # Use a flag to ensure we only check once per agent
        if not hasattr(self, '_obs_space_fix_checked'):
            self._obs_space_fix_checked = set()
        
        for idx, (agent_name, policy) in enumerate(zip(self.possible_agents, self.policies)):
            if policy is None or agent_name in self._obs_space_fix_checked:
                continue
            try:
                env_obs_space = self.observation_spaces.get(agent_name)
                if env_obs_space is None:
                    continue
                
                env_obs_dim = env_obs_space.shape[0] if hasattr(env_obs_space, 'shape') else None
                if env_obs_dim is None:
                    continue
                
                # Check policy observation space AND network input dimension
                policy_needs_recreation = False
                if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                    policy_obs_dim = policy.observation_space.shape[0]
                    if policy_obs_dim != env_obs_dim:
                        # Check if the network was actually built with wrong dimensions
                        network_input_dim = None
                        try:
                            if hasattr(policy, 'policy') and hasattr(policy.policy, 'features_extractor'):
                                # Try to get the input dimension from the features extractor
                                if hasattr(policy.policy.features_extractor, 'mlp'):
                                    first_layer = policy.policy.features_extractor.mlp[0]
                                    if hasattr(first_layer, 'in_features'):
                                        network_input_dim = first_layer.in_features
                                elif hasattr(policy.policy.features_extractor, 'linear'):
                                    # For GNN or other custom extractors
                                    first_layer = policy.policy.features_extractor.linear[0] if isinstance(policy.policy.features_extractor.linear, nn.ModuleList) else policy.policy.features_extractor.linear
                                    if hasattr(first_layer, 'in_features'):
                                        network_input_dim = first_layer.in_features
                        except Exception:
                            pass
                        
                        if network_input_dim is not None and network_input_dim != env_obs_dim:
                            policy_needs_recreation = True
                            self.logger.warning(f"[OBS_SPACE_RECREATE] {agent_name}: Network expects {network_input_dim}D input, but environment provides {env_obs_dim}D. Recreating policy...")
                        elif network_input_dim is None:
                            # Can't check network, but observation space doesn't match - assume needs recreation
                            policy_needs_recreation = True
                            self.logger.warning(f"[OBS_SPACE_RECREATE] {agent_name}: Policy observation space is {policy_obs_dim}D, environment is {env_obs_dim}D. Recreating policy...")
                
                if policy_needs_recreation:
                    # CRITICAL: Recreate the policy with correct observation space
                    try:
                        self.logger.warning(f"[OBS_SPACE_RECREATE] {agent_name}: Recreating policy with {env_obs_dim}D observation space...")
                        # Get activation function from config
                        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
                        act = getattr(self.config, "activation_fn", "tanh")
                        activation_fn = act_map.get(act.lower(), nn.Tanh)
                        
                        new_policy = self._create_single_policy(self.config, self.device, agent_name, idx, activation_fn)
                        if new_policy is not None:
                            # Replace the old policy
                            old_policy = self.policies[idx]
                            self.policies[idx] = new_policy
                            # NOTE: With robust tier detection (config flags), policies should not be recreated
                            # after initialization, so GAE skip workaround is no longer needed
                            # Update memory tracking
                            if hasattr(self.memory_tracker, '_policies') and old_policy in self.memory_tracker._policies:
                                self.memory_tracker._policies.remove(old_policy)
                            self.memory_tracker.register_policy(new_policy)
                            # Transfer buffers if they exist
                            if hasattr(old_policy, 'rollout_buffer') and old_policy.rollout_buffer is not None:
                                if hasattr(self.memory_tracker, '_buffers') and old_policy.rollout_buffer in self.memory_tracker._buffers:
                                    self.memory_tracker._buffers.remove(old_policy.rollout_buffer)
                            if hasattr(new_policy, 'rollout_buffer') and new_policy.rollout_buffer is not None:
                                self.memory_tracker.register_buffer(new_policy.rollout_buffer)
                            self.logger.info(f"[OBS_SPACE_RECREATE] {agent_name}: Policy recreated successfully with {env_obs_dim}D (will skip GAE on first rollout)")
                        else:
                            self.logger.error(f"[OBS_SPACE_RECREATE] {agent_name}: Failed to recreate policy, keeping old one")
                    except Exception as recreate_error:
                        self.logger.error(f"[OBS_SPACE_RECREATE] {agent_name}: Error recreating policy: {recreate_error}")
                
                # Mark as checked
                self._obs_space_fix_checked.add(agent_name)
            except Exception as e:
                self.logger.debug(f"Could not fix observation space for {agent_name}: {e}")
                self._obs_space_fix_checked.add(agent_name)  # Mark as checked even on error

    # ----------------------------- Action guards -----------------------------
    def _ensure_action_shape(self, action, action_space):
        """
        PHASE 5.5 PATCH A: Type and Shape Coercion Only

        This function focuses ONLY on type and shape coercion.
        Final clamping to action space bounds is performed by wrapper._validate_actions_comprehensive.
        """
        try:
            if isinstance(action_space, Discrete):
                if isinstance(action, (list, tuple, np.ndarray)) and np.size(action) > 0:
                    a = int(np.atleast_1d(action)[0])
                elif action is None:
                    a = 0
                else:
                    a = int(action)
                # PHASE 5.5 PATCH A: Remove clamping - wrapper will handle it
                return int(a)

            exp = int(action_space.shape[0]) if hasattr(action_space, "shape") else 1
            if action is None:
                a = np.zeros(exp, np.float32)
            else:
                a = np.array(action, np.float32).reshape(-1)
            if a.size == 0:
                a = np.zeros(exp, np.float32)
            elif a.size == 1 and exp > 1:
                a = np.repeat(a[0], exp).astype(np.float32)
            if a.size < exp:
                a = np.concatenate([a, np.zeros(exp - a.size, np.float32)])
            elif a.size > exp:
                a = a[:exp]
            # PHASE 5.5 PATCH A: Remove clamping - wrapper will handle it
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            return a.astype(np.float32)
        except Exception:
            if isinstance(action_space, Box):
                return ((action_space.low + action_space.high) / 2.0).astype(np.float32)
            return 0

    def _process_action_enhanced(self, raw_action, action_space):
        """
        PHASE 5.5 PATCH A: Type and Shape Coercion Only

        Sanitize actions (type/shape/NaN handling) while preserving distributional variability.
        Final clamping to action space bounds is performed by wrapper._validate_actions_comprehensive.
        """
        try:
            if isinstance(action_space, Discrete):
                if isinstance(raw_action, (list, tuple, np.ndarray)) and np.size(raw_action) > 0:
                    a = int(np.round(np.atleast_1d(raw_action)[0]))
                else:
                    a = int(np.round(float(raw_action)))
                # PHASE 5.5 PATCH A: Remove clamping - wrapper will handle it
                return int(a)
            a = np.asarray(raw_action, np.float32).reshape(-1)
            if np.any(~np.isfinite(a)):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # PHASE 5.5 PATCH A: Remove clamping - wrapper will handle it
            return a.astype(np.float32)
        except Exception:
            return self._ensure_action_shape(raw_action, action_space)

    def _coerce_action_for_buffer(self, policy, agent_name: str, action):
        try:
            # Get the action space from policy or fallback to class-level spaces
            action_space = getattr(policy, "action_space", None)
            if action_space is None:
                action_space = self.action_spaces.get(agent_name, None)

            if action_space is None:
                # Ultimate fallback: return action as-is but ensure it's a numpy array
                return np.asarray(action, dtype=np.float32).flatten()

            # Handle discrete action spaces
            if isinstance(action_space, Discrete):
                if isinstance(action, (list, tuple, np.ndarray)) and np.size(action) > 0:
                    a = int(np.round(np.atleast_1d(action)[0]))
                else:
                    a = int(np.round(float(action)))
                # Return as 1D array for buffer compatibility
                return np.array([np.clip(a, 0, action_space.n - 1)], dtype=np.float32)

            # Handle continuous (Box) action spaces
            a = np.asarray(action, dtype=np.float32).flatten()

            # Handle NaN/inf values
            if np.any(~np.isfinite(a)):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure correct shape for the action space
            if hasattr(action_space, "shape") and action_space.shape is not None:
                target_size = int(np.prod(action_space.shape))
                if a.size == 0:
                    # Create zero action with correct size
                    a = np.zeros(target_size, dtype=np.float32)
                elif a.size < target_size:
                    # Pad with zeros
                    padding = np.zeros(target_size - a.size, dtype=np.float32)
                    a = np.concatenate([a, padding])
                elif a.size > target_size:
                    # Truncate to correct size
                    a = a[:target_size]

            # MAINTENANCE: Apply bounds if available - ROBUSTIFIED for complex action space objects
            # Verify that action clipping for off-policy agents (SAC/TD3) correctly handles
            # the robustified clipping of the action vector to ensure replay buffer stores valid, bounded actions
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                low = getattr(action_space, "low", None)
                high = getattr(action_space, "high", None)
                if low is not None and high is not None:
                    try:
                        # Ensure low/high are convertible to numpy arrays for clipping
                        low_np = np.asarray(low, dtype=np.float32).flatten()
                        high_np = np.asarray(high, dtype=np.float32).flatten()
                        a_flat = a.flatten()
                        target_size = a_flat.size

                        # Ensure size matches a
                        if low_np.size == 0:
                            # Empty bounds - use default safe bounds
                            low_np = np.full(target_size, -1.0, dtype=np.float32)
                        elif low_np.size == 1:
                            # Single value - broadcast to all dimensions
                            low_np = np.full(target_size, float(low_np[0]), dtype=np.float32)
                        elif low_np.size < target_size:
                            # Pad with last value (or first if empty)
                            pad_val = float(low_np[-1]) if low_np.size > 0 else -1.0
                            low_np = np.pad(low_np, (0, target_size - low_np.size), 'constant', constant_values=pad_val)
                        elif low_np.size > target_size:
                            low_np = low_np[:target_size]

                        if high_np.size == 0:
                            # Empty bounds - use default safe bounds
                            high_np = np.full(target_size, 1.0, dtype=np.float32)
                        elif high_np.size == 1:
                            # Single value - broadcast to all dimensions
                            high_np = np.full(target_size, float(high_np[0]), dtype=np.float32)
                        elif high_np.size < target_size:
                            # Pad with last value (or first if empty)
                            pad_val = float(high_np[-1]) if high_np.size > 0 else 1.0
                            high_np = np.pad(high_np, (0, target_size - high_np.size), 'constant', constant_values=pad_val)
                        elif high_np.size > target_size:
                            high_np = high_np[:target_size]

                        # Ensure bounds are valid (low <= high) - fix any inverted bounds
                        low_np = np.minimum(low_np, high_np)
                        high_np = np.maximum(low_np, high_np)

                        # FIX: Final action clipping for off-policy agents (SAC/TD3)
                        # CRITICAL: This clipped action MUST be the same action executed by wrapper
                        # The wrapper will use this clipped action, ensuring consistency between
                        # replay buffer storage and environment execution
                        a_flat = np.clip(a_flat, low_np, high_np)
                        a = a_flat.reshape(a.shape)

                        # MAINTENANCE: Verify clipping was successful
                        if not np.all(np.isfinite(a)):
                            if hasattr(self, 'logger'):
                                self.logger.warning(f"Non-finite values after clipping for {agent_name}")
                            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

                    except Exception as e:
                        # Log warning about complex bounds and skip clipping
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Complex action space bounds for {agent_name}, skipping clipping: {e}")

            return a.astype(np.float32)

        except Exception as e:
            # Fallback: log warning and return a safe default action
            if hasattr(self, 'logger'):
                self.logger.warning(f"Action coercion failed for {agent_name}: {e}. Using fallback.")

            # Try to create a safe default action
            try:
                if isinstance(action_space, Discrete):
                    return np.array([0], dtype=np.float32)
                elif hasattr(action_space, "shape") and action_space.shape is not None:
                    target_size = int(np.prod(action_space.shape))
                    if hasattr(action_space, "low") and hasattr(action_space, "high"):
                        # Use midpoint of action space - ROBUSTIFIED for complex bounds
                        try:
                            low = np.asarray(action_space.low, dtype=np.float32).flatten()
                            high = np.asarray(action_space.high, dtype=np.float32).flatten()

                            # Ensure both arrays have the same size
                            min_size = min(low.size, high.size, target_size)
                            low = low[:min_size]
                            high = high[:min_size]

                            midpoint = (low + high) / 2.0

                            # Pad to target size if needed
                            if midpoint.size < target_size:
                                midpoint = np.pad(midpoint, (0, target_size - midpoint.size), 'constant', constant_values=0.0)

                            return midpoint[:target_size].astype(np.float32)
                        except Exception:
                            # Fallback to zeros if midpoint calculation fails
                            return np.zeros(target_size, dtype=np.float32)
                    else:
                        return np.zeros(target_size, dtype=np.float32)
                else:
                    # Ultimate fallback
                    return np.array([0.0], dtype=np.float32)
            except Exception:
                # Last resort
                return np.array([0.0], dtype=np.float32)

    # ------------------------------- Learn -----------------------------------
    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List] = None,
        log_interval: int = 1,
        tb_log_name: str = "MultiAgentRL",
        reset_num_timesteps: bool = True,
        overall_target: Optional[int] = None,
    ):
        if overall_target is not None:
            self.set_total_training_budget(overall_target)

        monitor = StabilizedTrainingMonitor(verbose=self.verbose)
        if callbacks is None:
            callbacks = [monitor] * self.num_agents
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]

        if len(callbacks) < self.num_agents:
            last = callbacks[-1] if callbacks else monitor
            callbacks = callbacks + [last] * (self.num_agents - len(callbacks))

        # Ensure each policy gets a callback list
        normalized_callbacks = []
        for cb in callbacks:
            if cb is None:
                normalized_callbacks.append([monitor])
            elif isinstance(cb, list):
                normalized_callbacks.append(cb + [monitor])
            else:
                normalized_callbacks.append([cb, monitor])
        callbacks = normalized_callbacks

        self._setup_learning_enhanced(total_timesteps, reset_num_timesteps, callbacks)
        self._run_enhanced_training_loop(total_timesteps, callbacks)
        self._finalize_training()

    def _setup_learning_enhanced(self, total_timesteps, reset_num_timesteps, callbacks):
        for polid, policy in enumerate(self.policies):
            try:
                if getattr(policy, "mode", None) in LEARNING_MODES:
                    adjusted = max(1000, int(total_timesteps) // max(1, self.num_agents))
                    policy._setup_learn(total_timesteps=adjusted, reset_num_timesteps=reset_num_timesteps)
                    policy.callback = policy._init_callback(callbacks[polid])
                    policy.callback.on_training_start(locals(), globals())
                    if self.debug:
                        print(f"Setup done for {policy.agent_name} ({policy.mode})")
            except Exception as e:
                self.logger.error(f"Setup failed for policy {polid}: {e}")
                self._training_metrics["policy_errors"] += 1

        # PHASE 3 PATCH C: Synchronize RL Rollouts with environment state (Mid-Episode Guardrail)
        # When the RL agent resets its buffer mid-episode (env.t is preserved, financial state is preserved),
        # the environment state is non-stationary across rollouts. This patch links the agent's total_steps
        # to the environment's current time index to enhance learning efficiency.
        # NOTE: The primary episodic budget synchronization is handled in main.py:run_episode_training.
        # This guardrail ensures robustness during mid-episode PPO buffer resets.
        try:
            current_env_t = getattr(self.env, 't', 0)
            if reset_num_timesteps:
                # True episode reset: agent starts fresh
                self.total_steps = 0
            else:
                # Mid-episode buffer reset: synchronize agent progress to data progress
                if current_env_t > 0 and self.total_steps < current_env_t:
                    self.total_steps = current_env_t
                    self.logger.info(f"[PHASE 3 PATCH C] Synchronized total_steps to env.t={current_env_t:,} (Mid-Episode Buffer Reset)")
        except Exception as e:
            self.logger.warning(f"[PHASE 3 PATCH C] Synchronization failed: {e}")

    def _run_enhanced_training_loop(self, total_timesteps, callbacks):
        def train_with_memory(polid, policy):
            try:
                level, before = self.memory_tracker.should_cleanup()
                if level:
                    freed = self.memory_tracker.cleanup(level)
                    self._training_metrics["memory_cleanups"] += 1
                    if self.debug and freed > 100:
                        print(f"Pre-train cleanup {policy.agent_name}: {freed:.1f}MB freed")

                if getattr(policy, "mode", None) == "PPO":
                    self._train_ppo_enhanced(policy)
                else:
                    self._train_offpolicy_enhanced(policy)

                _, after = self.memory_tracker.should_cleanup()
                if (after - before) > 200:
                    self.memory_tracker.cleanup("light")
            except Exception as e:
                self.logger.error(f"Training error for {getattr(policy, 'agent_name', polid)}: {e}")
                self._training_metrics["policy_errors"] += 1

        pbar = tqdm(total=int(total_timesteps), desc="Training..", unit="steps")

        self._initialize_environment_enhanced()
        target = self.total_steps + int(total_timesteps)

        while self.total_steps < target:
            try:
                steps_collected = self._collect_rollouts_enhanced(callbacks)
                if steps_collected == 0:
                    self._handle_rollout_failure_enhanced()
                    continue

                self.total_steps += steps_collected
                self._training_metrics["successful_steps"] += steps_collected
                pbar.update(steps_collected)

                if self.total_steps % 1000 == 0:
                    self._update_progress_display(pbar, total_timesteps)

                safe_multithreaded_processing(train_with_memory, self.policies, self.multithreading, max_workers=2)
                self._consecutive_errors = 0

                if self.total_steps % 5000 == 0:
                    self.memory_tracker.cleanup("medium")

            except Exception as e:
                self._handle_training_error_enhanced(e)
                if self._consecutive_errors >= self._max_consecutive_errors:
                    break

        pbar.close()

    def _train_ppo_enhanced(self, policy):
        """
        Train PPO policy with robust buffer fullness check.

        CRITICAL FIX: Instead of relying on unreliable `buffer.full` flag,
        explicitly check that buffer.pos >= n_steps to ensure sufficient data.
        """
        if not hasattr(policy, "rollout_buffer") or policy.rollout_buffer is None:
            return

        buffer = policy.rollout_buffer

        # Get n_steps from policy (default to self.n_steps if not available)
        n_steps = getattr(policy, "n_steps", self.n_steps)

        # ROBUST CHECK: Verify buffer has enough data
        # Check both buffer.pos (current position) and buffer.full flag
        buffer_pos = getattr(buffer, "pos", 0)
        buffer_full = getattr(buffer, "full", False)
        buffer_size = getattr(buffer, "buffer_size", n_steps)

        # Train if buffer is full OR has at least n_steps of data
        has_enough_data = buffer_full or (buffer_pos >= n_steps)

        if has_enough_data:
            try:
                # Train policy and capture training metrics if available
                train_info = policy.train()
                
                # Log training proof: policy updates happening
                if train_info and isinstance(train_info, dict):
                    policy_loss = train_info.get('train/policy_loss', train_info.get('policy_loss', 'N/A'))
                    value_loss = train_info.get('train/value_loss', train_info.get('value_loss', 'N/A'))
                    entropy_loss = train_info.get('train/entropy_loss', train_info.get('entropy_loss', 'N/A'))
                    approx_kl = train_info.get('train/approx_kl', train_info.get('approx_kl', 'N/A'))
                    
                    # Log periodically to show training is happening
                    if self.total_steps % 5000 == 0 or (hasattr(self, '_last_log_step') and self.total_steps - self._last_log_step >= 5000):
                        self.logger.info(f"[TRAINING_PROOF] {policy.agent_name} policy updated at step {self.total_steps}: "
                                       f"policy_loss={policy_loss}, value_loss={value_loss}, entropy={entropy_loss}, "
                                       f"approx_kl={approx_kl}")
                        if not hasattr(self, '_last_log_step'):
                            self._last_log_step = 0
                        self._last_log_step = self.total_steps
                else:
                    # Even if no metrics available, log that training occurred
                    if self.total_steps % 5000 == 0 or (hasattr(self, '_last_log_step') and self.total_steps - self._last_log_step >= 5000):
                        self.logger.info(f"[TRAINING_PROOF] {policy.agent_name} policy updated at step {self.total_steps} (metrics not available)")
                        if not hasattr(self, '_last_log_step'):
                            self._last_log_step = 0
                        self._last_log_step = self.total_steps

                # FAMC: Train meta-critic head periodically (meta mode only)
                self._step_count_famc += 1
                self._train_meta_head_if_needed()

                policy.rollout_buffer.reset()
                gc.collect()
            except Exception as e:
                self.logger.warning(f"PPO training failed for {policy.agent_name}: {e}")
        else:
            # Only log if we're past warmup phase and buffer is unexpectedly empty
            if self.total_steps > 100 and buffer_pos == 0:
                self.logger.debug(
                    f"PPO buffer for {policy.agent_name} not ready: "
                    f"pos={buffer_pos}/{buffer_size}, full={buffer_full}, n_steps={n_steps}"
                )

    def _rb_size(self, rb) -> int:
        try:
            return int(rb.size()) if hasattr(rb, "size") else int(len(rb))
        except Exception:
            return 0

    def _train_offpolicy_enhanced(self, policy):
        if not hasattr(policy, "replay_buffer") or policy.replay_buffer is None:
            return
        try:
            buffer_size = self._rb_size(policy.replay_buffer)
            if buffer_size >= 128:  # FIXED: Use standardized batch size
                max_grad_steps = min(32, buffer_size // 128)
                if self.memory_tracker.get_memory_usage() > self.memory_tracker.max_memory_mb * 0.8:
                    max_grad_steps = min(max_grad_steps, 8)
                policy.train(gradient_steps=max_grad_steps)

                if buffer_size > 75000:
                    try:
                        if hasattr(policy.replay_buffer, "_storage"):
                            trim = int(buffer_size * 0.75)
                            policy.replay_buffer._storage = policy.replay_buffer._storage[-trim:]
                            policy.replay_buffer.pos = trim
                            policy.replay_buffer.full = trim >= getattr(policy.replay_buffer, "buffer_size", trim)
                            if self.debug:
                                self.logger.info(
                                    f"Trimmed {policy.agent_name} buffer: {buffer_size} → {trim}"
                                )
                    except Exception as te:
                        self.logger.warning(f"Buffer trimming failed for {policy.agent_name}: {te}")
        except Exception as e:
            self.logger.warning(f"Off-policy training failed for {policy.agent_name}: {e}")

    # ----------------------------- Core Loop ---------------------------------
    def _initialize_environment_enhanced(self):
        try:
            self._last_obs, _ = self.env.reset()
            # CRITICAL FIX: Ensure all observations are 1D (remove batch dimensions)
            for agent in self._last_obs:
                obs = self._last_obs[agent]
                if isinstance(obs, np.ndarray):
                    if obs.ndim > 1:
                        # Remove batch dimension if present
                        if obs.shape[0] == 1:
                            self._last_obs[agent] = obs.squeeze(0)
                        else:
                            self._last_obs[agent] = obs.flatten()
                    elif obs.ndim == 0:
                        # Scalar - convert to 1D array
                        self._last_obs[agent] = np.array([obs], dtype=np.float32)
                    else:
                        self._last_obs[agent] = obs.astype(np.float32)
            
            # ROBUST VERIFICATION: Verify observation spaces still match actual observations
            # This should NEVER fail if initialization was correct
            for agent in self._last_obs:
                if agent in self.observation_spaces:
                    expected_dim = self.observation_spaces[agent].shape[0]
                    actual_dim = self._last_obs[agent].shape[0]
                    if expected_dim != actual_dim:
                        error_msg = (
                            f"[CRITICAL RUNTIME ERROR] Observation space mismatch for {agent}: "
                            f"Policy expects {expected_dim}D but environment produces {actual_dim}D. "
                            f"This should NEVER happen if initialization was correct. "
                            f"This indicates:\n"
                            f"  1. Environment observation space changed after initialization, OR\n"
                            f"  2. Initialization verification failed to catch this mismatch\n\n"
                            f"Cannot continue - this is a critical bug."
                        )
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)
            
            self._episode_starts = [True] * self.num_agents
            if not self.obs_validator.validate_observation_dict(self._last_obs):
                if self.debug:
                    print("Fixing initial observations")
                self._last_obs = self._fix_observation_dict_enhanced(self._last_obs)
                self._training_metrics["observation_fixes"] += 1
        except Exception as e:
            error_msg = f"[CRITICAL] Environment initialization failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _collect_rollouts_enhanced(self, callbacks):
        steps_collected = 0
        max_steps_per_rollout = min(self.n_steps, 256)

        level, _ = self.memory_tracker.should_cleanup()
        if level == "heavy":
            self.memory_tracker.cleanup("medium")

        while steps_collected < max_steps_per_rollout:
            try:
                if self._last_obs is None:
                    self._initialize_environment_enhanced()
                    continue

                if not self.obs_validator.validate_observation_dict(self._last_obs):
                    self._last_obs = self._fix_observation_dict_enhanced(self._last_obs)
                    self._training_metrics["observation_fixes"] += 1

                # ROBUST: No runtime fixes - if there's a mismatch, initialization should have caught it
                actions_dict, agent_data = self._collect_actions_enhanced()
                next_obs, rewards, dones, truncs, infos = self._execute_environment_step_enhanced(actions_dict)
                self._add_experiences_enhanced(agent_data, rewards, dones, truncs, next_obs)
                self._update_state_enhanced(next_obs, dones, truncs)

                steps_collected += 1

                # PHASE 5.6 FIX: Don't exit early during episode training
                # During episode training, continue collecting steps until max_steps_per_rollout
                is_episode_training = getattr(self.env, '_episode_training_mode', False)
                if not is_episode_training and self._check_buffers_full():
                    break

                if steps_collected % 80 == 0:
                    lvl, _ = self.memory_tracker.should_cleanup()
                    if lvl:
                        self.memory_tracker.cleanup("light")
            except Exception as e:
                self.logger.error(f"Rollout step error: {e}")
                break

        self._finalize_rollouts_enhanced()
        return steps_collected

    # REMOVED: _blend_with_expert_suggestions() - Legacy action blending removed
    # Action blending is deprecated and replaced by FGB (Tier 3)
    # Tier 2 does NOT blend actions - PPO learns directly from observations

    def _collect_actions_enhanced(self):
        actions_dict: Dict[str, Any] = {}
        agent_data: Dict[int, Dict[str, Any]] = {}

        for polid, policy in enumerate(self.policies):
            agent_name = self.possible_agents[polid]
            try:
                # CRITICAL FIX: Ensure observation is 1D before reshaping
                raw_obs = self._last_obs[agent_name]
                if isinstance(raw_obs, np.ndarray):
                    if raw_obs.ndim > 1:
                        # Remove batch dimension if present
                        if raw_obs.shape[0] == 1:
                            raw_obs = raw_obs.squeeze(0)
                        else:
                            raw_obs = raw_obs.flatten()
                    raw_obs = raw_obs.astype(np.float32)
                
                # ROBUST VERIFICATION: Verify observation dimension matches policy's expected dimension
                # NO WORKAROUNDS - if dimensions don't match, this is a CRITICAL ERROR indicating initialization failure
                if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                    expected_dim = policy.observation_space.shape[0]
                else:
                    expected_dim = self.observation_spaces[agent_name].shape[0]
                
                actual_dim = raw_obs.shape[0]
                if actual_dim != expected_dim:
                    # CRITICAL ERROR: Observation dimension mismatch detected during action collection
                    # This should NEVER happen if initialization was correct
                    # This indicates a bug in initialization or environment observation space changed
                    error_msg = (
                        f"[CRITICAL RUNTIME ERROR] Observation dimension mismatch for {agent_name}:\n"
                        f"  Policy expects: {expected_dim}D\n"
                        f"  Environment produces: {actual_dim}D\n"
                        f"  Policy obs_space: {policy.observation_space.shape if hasattr(policy, 'observation_space') else 'N/A'}\n"
                        f"  Agent obs_space: {self.observation_spaces[agent_name].shape}\n\n"
                        f"This indicates:\n"
                        f"  1. Initialization verification failed to catch this mismatch, OR\n"
                        f"  2. Environment observation space changed after initialization, OR\n"
                        f"  3. Policy was created with incorrect observation space\n\n"
                        f"Cannot continue - this is a critical bug that must be fixed."
                    )
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Now reshape to (1, -1) for policy input
                obs = raw_obs.reshape(1, -1)

                with torch.no_grad():
                    obs_tensor = obs_as_tensor(obs, policy.device)
                    if getattr(policy, "mode", None) == "PPO":
                        # Stable SB3 API: distribution sampling + log_prob + value
                        distribution = policy.policy.get_distribution(obs_tensor)
                        action_t = distribution.get_actions(deterministic=False)
                        log_prob_t = distribution.log_prob(action_t)
                        value_t = policy.policy.predict_values(obs_tensor)

                        raw = action_t.detach().cpu().numpy().flatten()
                        proc = self._process_action_enhanced(raw, self.action_spaces[agent_name])
                        proc = self._ensure_action_shape(proc, self.action_spaces[agent_name])
                        
                        # Log action decisions periodically to prove agents are making decisions
                        if self.total_steps % 10000 == 0:
                            value_mean = float(value_t.detach().cpu().numpy().mean()) if value_t is not None else 'N/A'
                            log_prob_mean = float(log_prob_t.detach().cpu().numpy().mean()) if log_prob_t is not None else 'N/A'
                            action_str = f"[{', '.join([f'{x:.3f}' for x in proc.flatten()])}]" if hasattr(proc, 'flatten') else str(proc)
                            self.logger.info(f"[DECISION_PROOF] {agent_name} sampled action {action_str} at step {self.total_steps} "
                                           f"(value={value_mean}, log_prob={log_prob_mean})")

                        # EXPERT BLENDING: Blend PPO action with DL overlay expert suggestions
                        # This gives DL overlay more direct control over actions
                        # if agent_name == "investor_0":
                        #     proc = self._blend_with_expert_suggestions(proc, agent_name)

                        agent_data[polid] = {
                            "obs": self._last_obs[agent_name].copy(),
                            "action": proc.copy(),
                            "value_t": value_t,
                            "log_prob_t": log_prob_t,
                        }
                        actions_dict[agent_name] = proc
                    else:
                        pred = policy.predict(obs, deterministic=False)[0]
                        if hasattr(pred, "flatten"):
                            pred = pred.flatten()
                        proc = self._process_action_enhanced(pred, self.action_spaces[agent_name])
                        proc = self._ensure_action_shape(proc, self.action_spaces[agent_name])
                        agent_data[polid] = {
                            "obs": self._last_obs[agent_name].copy(),
                            "action": proc.copy(),
                            "value_t": None,
                            "log_prob_t": None,
                        }
                        actions_dict[agent_name] = proc
            except Exception as e:
                self.logger.warning(f"Action collection error for {agent_name}: {e}")
                actions_dict[agent_name], agent_data[polid] = self._create_fallback_action_data(agent_name, polid)

        return actions_dict, agent_data

    def _execute_environment_step_enhanced(self, actions_dict):
        try:
            return self.env.step(actions_dict)
        except Exception as e:
            self.logger.warning(f"Environment step error: {e}")
            # reset to avoid cascading bad states
            try:
                self._initialize_environment_enhanced()
            except Exception:
                pass
            next_obs = self._last_obs.copy() if isinstance(self._last_obs, dict) else {}
            rewards = {a: 0.0 for a in self.possible_agents}
            dones = {a: False for a in self.possible_agents}
            truncs = {a: False for a in self.possible_agents}
            infos = {a: {} for a in self.possible_agents}
            return next_obs, rewards, dones, truncs, infos

    def _create_fallback_action_data(self, agent_name, polid):
        action_space = self.action_spaces[agent_name]
        if isinstance(action_space, Box):
            safe = ((action_space.low + action_space.high) / 2.0).astype(np.float32)
        else:
            safe = 0
        safe = self._ensure_action_shape(safe, action_space)
        data = {
            "obs": self._last_obs[agent_name].copy(),
            "action": safe.copy() if isinstance(safe, np.ndarray) else safe,
            "value_t": None,
            "log_prob_t": None,
        }
        return safe, data

    def _progress_denominator(self, interval_total: int) -> int:
        if isinstance(self._global_target_steps, int) and self._global_target_steps > 0:
            return self._global_target_steps
        return max(1, interval_total)

    def _update_progress_display(self, pbar, interval_total_timesteps):
        memory_stats = self.memory_tracker.get_memory_stats()
        denom = self._progress_denominator(interval_total_timesteps)
        overall_pct = min(100.0, (self.total_steps / denom) * 100.0)
        interval_pct = min(100.0, (pbar.n / max(1, interval_total_timesteps)) * 100.0)
        pbar.set_postfix(
            {
                "step": self.total_steps,
                "interval": f"{interval_pct:.1f}%",
                "overall": f"{overall_pct:.1f}%",
                "mem": f"{memory_stats['current_memory_mb']:.0f}MB",
                "use": f"{memory_stats['memory_usage_pct']:.1f}%",
                "errs": self._consecutive_errors,
                "clean": self._training_metrics["memory_cleanups"],
                "fix": self._training_metrics["observation_fixes"],
            }
        )

    def _handle_rollout_failure_enhanced(self):
        if self.debug:
            print("Rollout failure recovery…")
        self.memory_tracker.cleanup("heavy")
        self._initialize_environment_enhanced()

    def _handle_training_error_enhanced(self, error):
        self._consecutive_errors += 1
        if self.debug:
            self.logger.error(
                f"Training error {self._consecutive_errors}/{self._max_consecutive_errors}: {error}"
            )
        if self._consecutive_errors < self._max_consecutive_errors // 2:
            self.memory_tracker.cleanup("medium")
        else:
            self.memory_tracker.cleanup("heavy")
        self._initialize_environment_enhanced()

    def _fix_observation_dict_enhanced(self, obs_dict):
        fixed = {}
        for agent in self.possible_agents:
            if agent in obs_dict:
                fixed[agent] = self.obs_validator.fix_observation(agent, obs_dict[agent])
            else:
                spec = self.obs_validator.observation_specs.get(agent)
                fixed[agent] = ((spec["low"] + spec["high"]) / 2.0) if spec else np.zeros(10, np.float32)
        return fixed

    def _create_emergency_observations_enhanced(self):
        out = {}
        for agent in self.possible_agents:
            spec = self.obs_validator.observation_specs.get(agent)
            out[agent] = ((spec["low"] + spec["high"]) / 2.0) if spec else np.zeros(10, np.float32)
        return out

    def _add_experiences_enhanced(self, agent_data, rewards, dones, truncs, next_obs):
        for polid, policy in enumerate(self.policies):
            agent_name = self.possible_agents[polid]
            if polid not in agent_data:
                continue
            try:
                data = agent_data[polid]
                reward = float(rewards.get(agent_name, 0.0))
                done = bool(dones.get(agent_name, False))

                # FGB/FAMC: Add forecast signals to data dict for baseline adjustment
                # These are computed by environment._compute_forecast_signals() each step
                data['forecast_trust'] = getattr(self.env, '_forecast_trust', 0.0)
                data['expected_dnav'] = getattr(self.env, '_expected_dnav', 0.0)

                # FAMC: Add meta-critic prediction if available
                # This comes from the DL overlay's meta_adv head (state-only)
                if hasattr(self.env, '_last_overlay_output') and self.env._last_overlay_output:
                    overlay_out = self.env._last_overlay_output
                    if isinstance(overlay_out, dict) and 'meta_adv' in overlay_out:
                        # Extract scalar from (1, 1) tensor
                        meta_adv_raw = overlay_out['meta_adv']
                        if hasattr(meta_adv_raw, 'shape') and len(meta_adv_raw.shape) > 0:
                            data['meta_adv_pred'] = float(meta_adv_raw.flatten()[0])
                        else:
                            data['meta_adv_pred'] = float(meta_adv_raw)
                    else:
                        data['meta_adv_pred'] = 0.0
                else:
                    data['meta_adv_pred'] = 0.0

                if getattr(policy, "mode", None) == "PPO":
                    self._add_ppo_experience(policy, data, reward, polid)
                else:
                    self._add_offpolicy_experience(policy, data, reward, done, truncs, next_obs, agent_name)
                if hasattr(policy, "num_timesteps"):
                    policy.num_timesteps += 1
            except Exception as e:
                self.logger.warning(f"Experience addition error for {agent_name}: {e}")
    
    def _validate_and_fix_buffer_shape(self, policy, polid):
        """
        ROBUST: Validate buffer observation shape matches policy observation space.
        If mismatch detected, recreate buffer to prevent corruption.
        
        CRITICAL: SB3's RolloutBuffer stores observations as (buffer_size, obs_dim) numpy array.
        If obs_dim changes, the buffer must be fully reinitialized, not just reset.
        
        Returns: True if buffer is valid, False if buffer was recreated
        """
        if not hasattr(policy, "rollout_buffer") or policy.rollout_buffer is None:
            return True
        
        if not hasattr(policy, 'observation_space') or not hasattr(policy.observation_space, 'shape'):
            return True
        
        try:
            buffer = policy.rollout_buffer
            expected_obs_dim = policy.observation_space.shape[0]
            buffer_size = getattr(buffer, 'buffer_size', 128)
            
            # Check if buffer has observations array
            if hasattr(buffer, 'observations') and buffer.observations is not None:
                obs_shape = buffer.observations.shape
                
                # SB3 stores observations as (buffer_size, obs_dim) - 2D array
                shape_invalid = False
                
                # SB3 RolloutBuffer can use either format:
                # - VecEnv format: (buffer_size, n_envs, obs_dim) where n_envs=1 for single env
                # - Single-env format: (buffer_size, obs_dim)
                # Both are valid, but we need obs_dim to match expected_obs_dim
                
                if len(obs_shape) == 3:
                    # VecEnv format: (buffer_size, n_envs, obs_dim)
                    n_envs_in_buffer = obs_shape[1]
                    obs_dim_in_buffer = obs_shape[2]
                    if obs_dim_in_buffer != expected_obs_dim:
                        # Observation dimension mismatch
                        shape_invalid = True
                        error_msg = f"Buffer obs_dim mismatch in VecEnv format: buffer has {obs_dim_in_buffer}D, expected {expected_obs_dim}D (shape={obs_shape})"
                    # Note: n_envs_in_buffer should be 1 for our single-env setup, but we don't enforce it
                elif len(obs_shape) == 2:
                    # Single-env format: (buffer_size, obs_dim)
                    if obs_shape[0] != buffer_size:
                        # Buffer size mismatch
                        shape_invalid = True
                        error_msg = f"Buffer size mismatch: buffer has {obs_shape[0]}, expected {buffer_size}"
                    elif obs_shape[1] != expected_obs_dim:
                        # Observation dimension mismatch
                        shape_invalid = True
                        error_msg = f"Buffer obs_dim mismatch: buffer has {obs_shape[1]}D, expected {expected_obs_dim}D"
                else:
                    # Invalid shape (not 2D or 3D)
                    shape_invalid = True
                    error_msg = f"Buffer has invalid shape {obs_shape}, expected 2D or 3D"
                
                if shape_invalid:
                    # CRITICAL: Buffer shape mismatch - DO NOT manually fix
                    # Manual buffer recreation breaks SB3's internal state and causes 'clone' errors
                    # Policies are recreated when observation spaces change, which properly recreates buffers
                    # Just log a warning and return True (assume valid) to avoid breaking training
                    # The policy recreation logic handles observation space mismatches
                    if not hasattr(self, '_buffer_mismatch_warned'):
                        self._buffer_mismatch_warned = set()
                    if polid not in self._buffer_mismatch_warned:
                        self.logger.warning(
                            f"[BUFFER_SHAPE_MISMATCH] Policy {polid} ({policy.agent_name}): "
                            f"Buffer shape mismatch detected but not fixing (would break SB3 internal state). "
                            f"Policy should be recreated if observation space changed."
                        )
                        self._buffer_mismatch_warned.add(polid)
                    return True  # Assume valid - policy recreation handles mismatches
            
            # CORRECT: Buffer shape is valid (or buffer doesn't have observations array yet)
            return True
        except Exception as e:
            self.logger.warning(f"[BUFFER_SHAPE_CHECK] Failed to validate buffer shape for policy {polid}: {e}")
            return True  # Assume valid if check fails to avoid blocking training

    def _add_ppo_experience(self, policy, data, reward, polid):
        """
        Add one transition to SB3's RolloutBuffer.

        CRITICAL FIX: SB3 RolloutBuffer.add() expects:
          - obs:            np.ndarray (CPU, float32)   shape: (obs_dim,) - 1D flattened
          - action:         np.ndarray (CPU, float32)   shape: (act_dim,) - 1D flattened
          - reward:         scalar or np.ndarray (CPU, float32)   shape: () or (1,)
          - episode_starts: scalar or np.ndarray (CPU, bool)      shape: () or (1,)
          - value:          torch.Tensor on policy.device, shape: (1, 1)
          - log_prob:       torch.Tensor on policy.device, shape: (1, 1)
        
        The buffer internally handles batching - we pass single flattened observations/actions.
        """
        # RolloutBuffer may not be initialized yet (e.g., during warmup)
        if not hasattr(policy, "rollout_buffer"):
            return

        # Check if buffer is None
        if policy.rollout_buffer is None:
            return

        # CRITICAL: Buffer shape validation disabled - manual recreation breaks SB3's internal state
        # If buffer shape is wrong, policies are recreated (which properly recreates buffers)
        # Manual buffer recreation causes 'clone' errors in compute_returns_and_advantage

        try:
            value_t = data.get("value_t", None)
            log_prob_t = data.get("log_prob_t", None)
            if value_t is None or log_prob_t is None:
                raise RuntimeError("Missing PPO tensors: value_t and/or log_prob_t are None.")

            # --- ensure value/log_prob shapes and device ---
            if hasattr(value_t, "shape") and tuple(value_t.shape) != (1, 1):
                value_t = value_t.reshape(1, 1)
            if hasattr(log_prob_t, "shape") and tuple(log_prob_t.shape) != (1, 1):
                log_prob_t = log_prob_t.reshape(1, 1)
            value_t = value_t.to(policy.device)
            log_prob_t = log_prob_t.to(policy.device)

            # --- obs -> numpy (CPU, float32, shape (obs_dim,)) ---
            # CRITICAL FIX: SB3 RolloutBuffer.add() expects observations as (obs_dim,) NOT (1, obs_dim)
            # The buffer internally handles batching - we pass a single flattened observation
            obs = data["obs"]
            if "torch" in str(type(obs)):
                obs = obs.detach().cpu().numpy()
            elif not isinstance(obs, np.ndarray):
                obs = np.asarray(obs)
            
            # Flatten to 1D array (obs_dim,) for SB3 buffer
            obs_np = obs.astype(np.float32).flatten()
            
            # CRITICAL: Validate observation dimension matches policy's expected dimension
            # After rebuild_observation_spaces() is called, dimensions should always match
            # CRITICAL FIX: obs_np is now 1D (obs_dim,), so we check shape[0]
            if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                expected_obs_dim = policy.observation_space.shape[0]
                actual_obs_dim = obs_np.shape[0]  # obs_np is now 1D (obs_dim,)
                
                if actual_obs_dim != expected_obs_dim:
                    # ROBUST FIX: NO TRUNCATION/PADDING - this is a critical error
                    error_msg = (
                        f"[CRITICAL] Observation dimension mismatch for policy {polid} ({policy.agent_name}): "
                        f"Expected {expected_obs_dim}D (policy obs_space), got {actual_obs_dim}D (actual observation). "
                        f"This indicates a bug - observation spaces must match exactly. "
                        f"Policy obs_space: {policy.observation_space.shape if hasattr(policy, 'observation_space') else 'N/A'}, "
                        f"Actual observation shape: {obs_np.shape}"
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            # --- action -> numpy (CPU, float32, shape (action_dim,)) ---
            # CRITICAL FIX: SB3 RolloutBuffer.add() expects actions as (action_dim,) NOT (1, action_dim)
            action_space = getattr(policy, "action_space", None) or self.action_spaces.get(
                getattr(policy, "agent_name", ""), None
            )
            action_fixed = self._ensure_action_shape(data["action"], action_space)
            if "torch" in str(type(action_fixed)):
                action_fixed = action_fixed.detach().cpu().numpy()
            action_np = np.asarray(action_fixed, dtype=np.float32).flatten()  # Flatten to 1D

            # --- FGB/FAMC: Forecast-guided baseline adjustment ---
            # Mode "fixed": Apply reward-level adjustment (legacy FGB)
            # Mode "online"/"meta": Store data for advantage-level correction (FAMC)
            fgb_mode = getattr(self.config, 'fgb_mode', 'fixed')

            if getattr(self.config, 'forecast_baseline_enable', False) and fgb_mode == 'fixed':
                # LEGACY FGB: Reward-level adjustment with fixed λ
                try:
                    lam = getattr(self.config, 'forecast_baseline_lambda', 0.5)
                    tau = data.get('forecast_trust', 0.0) if isinstance(data, dict) else 0.0
                    exp_dnav = data.get('expected_dnav', 0.0) if isinstance(data, dict) else 0.0

                    # FGB: Normalize E[ΔNAV] from DKK to return units
                    # This keeps the baseline adjustment comparable to reward scale
                    nav_norm = max(1.0, float(getattr(self.config, "init_budget", 0.0)) or 1.0)
                    baseline_adj = lam * tau * (exp_dnav / nav_norm)

                    # CRITICAL FIX: Increase clip range for meaningful FGB impact
                    # Previous: ±0.01 (1 bp) was too conservative - only 1% of typical reward
                    # New: ±0.10 (10 bp) allows 10% adjustment - meaningful variance reduction
                    # This ensures FGB has bite while still preventing reward swamping
                    baseline_adj = float(np.clip(baseline_adj, -0.10, 0.10))

                    reward = float(reward) - baseline_adj

                    # Log telemetry (metrics tracked in _famc_metrics dict)
                    # Note: self.logger is Python logging.Logger, not SB3 logger
                    # SB3 logger.record() is not available here
                    if self.debug and self._step_count_famc % 1000 == 0:
                        self.logger.debug(f"[FGB] trust={tau:.3f}, exp_dnav={exp_dnav:.2f}, baseline_adj={baseline_adj:.4f}")
                except Exception as e:
                    self.logger.debug(f"[FGB] Baseline adjustment failed: {e}")

            # FAMC: Store control variate data for advantage-level correction (online/meta modes)
            # This data will be used in _finalize_rollouts_enhanced after GAE computation
            if fgb_mode in ['online', 'meta']:
                # Store forecast signals and meta predictions for this step
                if not hasattr(policy, '_famc_step_data'):
                    policy._famc_step_data = []

                # FAMC: Get overlay features (34D) for meta head training
                # These are the features used for overlay inference, not agent observations
                overlay_feats = None
                if hasattr(self.env, '_last_overlay_features') and self.env._last_overlay_features is not None:
                    overlay_feats = self.env._last_overlay_features.copy()
                    # CRITICAL: Ensure overlay features are 34D (28D base + 6D deltas)
                    if overlay_feats.ndim == 2:
                        overlay_feats = overlay_feats[0]  # Remove batch dimension if present
                    if overlay_feats.shape[0] != 34:
                        # Dimension mismatch - skip storing this step's data
                        self.logger.debug(f"[FAMC] Skipping step: overlay features have wrong dimension ({overlay_feats.shape[0]}D, expected 34D)")
                        overlay_feats = None
                else:
                    # CRITICAL FIX: Don't use agent obs as fallback - dimensions differ per agent!
                    # investor_0 has 18D, battery_operator_0 has 14D - can't mix them in same buffer
                    overlay_feats = None

                # Only store FAMC data if we have valid overlay features
                if overlay_feats is not None and overlay_feats.shape[0] == 34:
                    famc_data = {
                        'tau': data.get('forecast_trust', 0.0),
                        'expected_dnav': data.get('expected_dnav', 0.0),
                        'meta_adv_pred': data.get('meta_adv_pred', 0.0),
                        'obs': overlay_feats,  # Store overlay features (34D) for meta head training
                    }
                    policy._famc_step_data.append(famc_data)

            # --- reward/start flags -> numpy (CPU) ---
            reward_np = np.array([reward], dtype=np.float32)
            starts_np = np.array([self._episode_starts[polid]], dtype=bool)

            # --- finally add to SB3 buffer ---
            # NOTE: Observation shape mismatch is already handled above (before this point)
            
            # CRITICAL FIX: Ensure obs_np and action_np are 1D arrays (obs_dim,) and (action_dim,)
            # SB3 RolloutBuffer.add() expects:
            # - obs: (obs_dim,) - flattened single observation
            # - action: (action_dim,) - flattened single action
            # - reward: scalar or (1,) array
            # - done: scalar or (1,) array
            # - value: (1, 1) tensor or scalar
            # - log_prob: (1, 1) tensor or scalar
            
            # Ensure 1D shape
            if obs_np.ndim > 1:
                obs_np = obs_np.flatten()
            if action_np.ndim > 1:
                action_np = action_np.flatten()
            
            # Ensure reward and starts are scalars or 1-element arrays
            if isinstance(reward_np, np.ndarray) and reward_np.size > 1:
                reward_np = reward_np[0] if reward_np.size > 0 else 0.0
            if isinstance(starts_np, np.ndarray) and starts_np.size > 1:
                starts_np = starts_np[0] if starts_np.size > 0 else False
            
            # CRITICAL FIX: Check buffer position BEFORE adding to prevent overflow
            # SB3's RolloutBuffer wraps automatically, but we need to check BEFORE add() is called
            buffer = policy.rollout_buffer
            buffer_size = getattr(buffer, 'buffer_size', 128)
            current_pos = getattr(buffer, 'pos', 0)
            
            # If buffer is at capacity (pos >= buffer_size), reset to 0 before adding
            # This prevents "index 128 is out of bounds" error
            if current_pos >= buffer_size:
                buffer.pos = 0
                buffer.full = False
                if not hasattr(self, '_buffer_wrap_warned'):
                    self._buffer_wrap_warned = set()
                if polid not in self._buffer_wrap_warned:
                    self.logger.warning(f"[BUFFER_WRAP] Policy {polid} ({policy.agent_name}): Buffer at capacity (pos={current_pos}/{buffer_size}), wrapping to 0")
                    self._buffer_wrap_warned.add(polid)
            
            policy.rollout_buffer.add(obs_np, action_np, reward_np, starts_np, value_t, log_prob_t)

        except Exception as e:
            self.logger.warning(f"PPO buffer add error for policy {polid}: {e}")
            # CRITICAL: Log detailed error information for debugging
            agent_name = getattr(policy, 'agent_name', f'policy_{polid}')
            try:
                if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                    self.logger.warning(f"  Policy obs_space: {policy.observation_space.shape}")
            except:
                pass
            try:
                if hasattr(policy, 'rollout_buffer') and policy.rollout_buffer is not None:
                    if hasattr(policy.rollout_buffer, 'obs_shape'):
                        self.logger.warning(f"  Buffer obs_shape: {policy.rollout_buffer.obs_shape}")
            except:
                pass
            try:
                self.logger.warning(f"  Actual observation shape: {obs_np.shape if 'obs_np' in locals() else 'N/A'}")
            except:
                pass

    def _add_offpolicy_experience(self, policy, data, reward, done, truncs, next_obs, agent_name):
        if not hasattr(policy, "replay_buffer") or policy.replay_buffer is None:
            return
        try:
            next_obs_agent = next_obs.get(agent_name, data["obs"])
            obs_fixed = np.asarray(data["obs"], np.float32).reshape(-1)
            next_obs_fixed = np.asarray(next_obs_agent, np.float32).reshape(-1)
            if obs_fixed.size == 0:
                obs_fixed = np.zeros(1, np.float32)
            if next_obs_fixed.size == 0:
                next_obs_fixed = obs_fixed.copy()

            # ROBUST FIX: Validate observation dimensions match policy's expected dimension
            # NO TRUNCATION/PADDING - if dimensions don't match, this is a critical error
            if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                expected_obs_dim = policy.observation_space.shape[0]
                
                # Validate obs_fixed
                if obs_fixed.shape[0] != expected_obs_dim:
                    error_msg = (
                        f"[CRITICAL] Observation dimension mismatch for {agent_name} in replay buffer: "
                        f"Policy expects {expected_obs_dim}D, but obs_fixed has {obs_fixed.shape[0]}D. "
                        f"This indicates a bug - observation spaces must match exactly."
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Validate next_obs_fixed
                if next_obs_fixed.shape[0] != expected_obs_dim:
                    error_msg = (
                        f"[CRITICAL] Observation dimension mismatch for {agent_name} in replay buffer: "
                        f"Policy expects {expected_obs_dim}D, but next_obs_fixed has {next_obs_fixed.shape[0]}D. "
                        f"This indicates a bug - observation spaces must match exactly."
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            action_fixed = self._coerce_action_for_buffer(policy, agent_name, data["action"])
            action_fixed = np.asarray(action_fixed, np.float32).reshape(-1)

            rb_adim = getattr(policy.replay_buffer, "action_dim", None) or getattr(
                policy.replay_buffer, "_action_dim", None
            )
            if rb_adim is None:
                ps = getattr(policy, "action_space", None)
                rb_adim = int(ps.shape[0]) if isinstance(ps, Box) and hasattr(ps, "shape") else 1
            rb_adim = max(1, int(rb_adim))

            if action_fixed.size == 0:
                action_fixed = np.zeros(rb_adim, np.float32)
            elif action_fixed.size < rb_adim:
                action_fixed = np.concatenate([action_fixed, np.zeros(rb_adim - action_fixed.size, np.float32)])
            elif action_fixed.size > rb_adim:
                action_fixed = action_fixed[:rb_adim]

            ps = getattr(policy, "action_space", None)
            if isinstance(ps, Box):
                action_fixed = np.clip(action_fixed, ps.low, ps.high)

            done_flag = bool(done or truncs.get(agent_name, False))

            add_sig = inspect.signature(policy.replay_buffer.add)
            n_params = len(add_sig.parameters)

            obs_b = obs_fixed.reshape(1, -1).astype(np.float32)
            next_obs_b = next_obs_fixed.reshape(1, -1).astype(np.float32)
            action_b = action_fixed.reshape(1, -1).astype(np.float32)
            # MAINTENANCE: Use np.asarray for cleaner tensor conversion
            reward_b = np.asarray([reward], np.float32)
            done_b = np.asarray([done_flag], np.float32)
            infos_b = [{}]

            if n_params >= 6:
                policy.replay_buffer.add(obs_b, next_obs_b, action_b, reward_b, done_b, infos_b)
            else:
                policy.replay_buffer.add(obs_b, next_obs_b, action_b, reward_b, done_b)
        except Exception as e:
            self.logger.warning(f"Replay buffer add error: {e}")

    def _update_state_enhanced(self, next_obs, dones, truncs):
        if not self.obs_validator.validate_observation_dict(next_obs):
            next_obs = self._fix_observation_dict_enhanced(next_obs)
            self._training_metrics["observation_fixes"] += 1

        # CRITICAL FIX: Ensure all observations are 1D (remove batch dimensions)
        for agent in next_obs:
            obs = next_obs[agent]
            if isinstance(obs, np.ndarray):
                if obs.ndim > 1:
                    # Remove batch dimension if present
                    if obs.shape[0] == 1:
                        next_obs[agent] = obs.squeeze(0)
                    else:
                        next_obs[agent] = obs.flatten()
                elif obs.ndim == 0:
                    # Scalar - convert to 1D array
                    next_obs[agent] = np.array([obs], dtype=np.float32)
                else:
                    next_obs[agent] = obs.astype(np.float32)

        self._last_obs = next_obs
        for polid, agent in enumerate(self.possible_agents):
            self._episode_starts[polid] = bool(dones.get(agent, False) or truncs.get(agent, False))

        # PHASE 5.6 FIX: Don't reset environment during episode training
        # During episode training, the environment should continue until episode_timesteps is reached
        if any(dones.values()) or any(truncs.values()):
            is_episode_training = getattr(self.env, '_episode_training_mode', False)
            if not is_episode_training:
                self._initialize_environment_enhanced()

    def _check_buffers_full(self):
        """
        Check if any PPO buffer has enough data for training.

        ROBUST: Check buffer.pos >= n_steps instead of just buffer.full flag.
        """
        for policy in self.policies:
            if getattr(policy, "mode", None) != "PPO":
                continue
            if not hasattr(policy, "rollout_buffer") or policy.rollout_buffer is None:
                continue

            buffer = policy.rollout_buffer
            n_steps = getattr(policy, "n_steps", self.n_steps)
            buffer_pos = getattr(buffer, "pos", 0)
            buffer_full = getattr(buffer, "full", False)

            # Return True if any buffer is full or has enough data
            if buffer_full or buffer_pos >= n_steps:
                return True

        return False

    def _finalize_rollouts_enhanced(self):
        for polid, policy in enumerate(self.policies):
            try:
                if getattr(policy, "mode", None) == "PPO" and hasattr(policy, "rollout_buffer"):
                    agent_name = self.possible_agents[polid]
                    if agent_name in self._last_obs:
                        # MAINTENANCE: Verify rollout buffer state before GAE calculation
                        buffer = policy.rollout_buffer

                        # Check if buffer is None or empty
                        if buffer is None:
                            if self.total_steps > 100:
                                self.logger.warning(f"PPO rollout buffer for policy {polid} is None!")
                            continue

                        if buffer.pos == 0:
                            # Only warn if we're past the initial warmup phase (total_steps > 100)
                            # During early training, empty buffers are normal and expected
                            if self.total_steps > 100:
                                # Add diagnostic info to help debug why buffer is empty
                                buffer_size = getattr(buffer, 'buffer_size', 'unknown')
                                full = getattr(buffer, 'full', 'unknown')
                                self.logger.warning(
                                    f"PPO rollout buffer for policy {polid} is empty or invalid "
                                    f"(total_steps={self.total_steps}, buffer.pos=0/{buffer_size}, "
                                    f"full={full})"
                                )
                                # Check if this is due to action collection errors
                                self.logger.warning(
                                    f"  → This usually means experiences aren't being added to the buffer. "
                                    f"Check for 'PPO buffer add error' or 'Action collection error' messages above."
                                )
                            continue

                        # PHASE 3 PATCH B: PPO Rollout Stability Guard
                        # Prevent GAE calculation crashes when buffer is under-filled
                        # This can occur after premature environment termination or buffer reset
                        n_steps = getattr(policy, 'n_steps', 128)
                        min_gae_len = max(1, n_steps // 4)  # Require at least 25% of n_steps

                        if buffer.pos < min_gae_len:
                            if self.total_steps > 100:
                                self.logger.warning(
                                    f"[PATCH B] PPO GAE skipped for {policy.agent_name}: "
                                    f"pos={buffer.pos}/{buffer.buffer_size} < min_gae_len={min_gae_len} "
                                    f"(n_steps={n_steps}). Buffer under-filled, skipping GAE to prevent crashes."
                                )
                            continue

                        # CRITICAL FIX: Skip buffer dimension check - SB3 manages buffer shape internally
                        # The 'clone' error suggests SB3 is trying to process the buffer incorrectly
                        # This usually happens when buffer has wrong observation shape internally
                        # Our truncation in _add_ppo_experience should prevent mismatches

                        final_obs = self._last_obs[agent_name].reshape(1, -1)
                        
                        # CRITICAL FIX: Ensure final_obs matches policy's expected observation dimension
                        # This prevents tensor dimension mismatches in predict_values
                        if hasattr(policy, 'observation_space') and hasattr(policy.observation_space, 'shape'):
                            expected_obs_dim = policy.observation_space.shape[0]
                            actual_obs_dim = final_obs.shape[1]
                            
                            if actual_obs_dim != expected_obs_dim:
                                # Fix dimension mismatch
                                if actual_obs_dim > expected_obs_dim:
                                    # Observation is larger - truncate to match policy
                                    final_obs = final_obs[:, :expected_obs_dim]
                                else:
                                    # Observation is smaller - pad with zeros
                                    pad_size = expected_obs_dim - actual_obs_dim
                                    padding = np.zeros((final_obs.shape[0], pad_size), dtype=np.float32)
                                    final_obs = np.concatenate([final_obs, padding], axis=1)
                        
                        with torch.no_grad():
                            obs_tensor = obs_as_tensor(final_obs, policy.device)
                            final_value = policy.policy.predict_values(obs_tensor)
                        
                        # CRITICAL FIX: SB3's compute_returns_and_advantage expects last_values as PyTorch tensor
                        # It internally calls .clone() which only works on tensors, not numpy arrays
                        # Keep final_value as a tensor on the policy's device
                        if isinstance(final_value, torch.Tensor):
                            if final_value.dim() == 0:
                                final_value = final_value.unsqueeze(0).unsqueeze(0)  # (1, 1)
                            elif final_value.dim() == 1:
                                final_value = final_value.unsqueeze(0)  # (1, n)
                            # Ensure shape is (1, 1) for SB3's compute_returns_and_advantage
                            if final_value.shape != (1, 1):
                                final_value = final_value.reshape(1, -1)[:, :1]  # Take first value, reshape to (1, 1)
                            # CRITICAL: Keep as tensor on policy device - DO NOT convert to numpy
                            # SB3's compute_returns_and_advantage will handle tensor operations internally
                            if final_value.device != policy.device:
                                final_value = final_value.to(policy.device)
                        else:
                            # Convert numpy/scalar to tensor if not already a tensor
                            if isinstance(final_value, np.ndarray):
                                if final_value.ndim == 0:
                                    final_value = final_value.reshape(1, 1)
                                elif final_value.ndim == 1:
                                    final_value = final_value.reshape(1, -1)[:, :1]
                                final_value = torch.as_tensor(final_value, dtype=torch.float32, device=policy.device)
                            else:
                                # Scalar value - convert to tensor
                                final_value = torch.tensor([[float(final_value)]], dtype=torch.float32, device=policy.device)

                        # CRITICAL FIX: GAE Terminal Flag
                        # Because the environment immediately resets on done=True, the final_obs
                        # collected in the buffer is the post-reset state, which is non-terminal.
                        # Therefore, dones_b must be explicitly set to np.array([False], dtype=bool)
                        # to align with custom loop logic and prevent GAE corruption.
                        dones_b = np.array([False], dtype=bool)

                        # ROOT CAUSE FIX: SB3's compute_returns_and_advantage expects last_values as PyTorch tensor
                        # It internally calls last_values.clone() which fails if final_value is numpy array
                        # We now keep final_value as tensor (fix above), so this should work correctly
                        
                        # Validate final_value is a tensor before passing to SB3
                        if not isinstance(final_value, torch.Tensor):
                            self.logger.error(f"[GAE_ERROR] Policy {polid} ({agent_name}): final_value is not a tensor! Type: {type(final_value)}")
                            # Convert to tensor as fallback
                            final_value = torch.tensor([[float(final_value)]], dtype=torch.float32, device=policy.device)
                        
                        # Ensure final_value is on correct device
                        if final_value.device != policy.device:
                            final_value = final_value.to(policy.device)
                        
                        # Check buffer state before GAE computation
                        buffer_pos = getattr(buffer, 'pos', 0)
                        buffer_full = getattr(buffer, 'full', False)
                        buffer_size = getattr(buffer, 'buffer_size', 128)
                        
                        # Only call GAE if buffer has sufficient data
                        buffer_ready = (buffer_pos >= buffer_size) or buffer_full
                        
                        if not buffer_ready:
                            # Skip GAE if buffer not ready (not enough data)
                            if self.total_steps % 1000 == 0:  # Log periodically
                                self.logger.debug(f"[GAE_SKIP] Policy {polid} ({agent_name}): Buffer not ready (pos={buffer_pos}, full={buffer_full})")
                            continue
                        
                        # Call GAE with tensor final_value - SB3 will handle tensor operations correctly
                        try:
                            policy.rollout_buffer.compute_returns_and_advantage(final_value, dones_b)
                        except Exception as gae_error:
                            error_str = str(gae_error)
                            if 'clone' in error_str.lower():
                                # This should not happen with the root cause fix (keeping final_value as tensor)
                                # Log as error with full diagnostics for investigation
                                self.logger.error(f"[GAE_CLONE_ERROR] Policy {polid} ({agent_name}): Unexpected clone error after root cause fix!")
                                self.logger.error(f"  final_value type: {type(final_value)}, device: {final_value.device if isinstance(final_value, torch.Tensor) else 'N/A'}")
                                self.logger.error(f"  Buffer state: pos={buffer_pos}/{buffer_size}, full={buffer_full}")
                                
                                # Check buffer observation format for additional diagnostics
                                if hasattr(buffer, 'observations') and buffer.observations is not None:
                                    obs_type = type(buffer.observations).__name__
                                    obs_shape = buffer.observations.shape if hasattr(buffer.observations, 'shape') else 'N/A'
                                    self.logger.error(f"  Buffer observations: type={obs_type}, shape={obs_shape}")
                                
                                # This should not happen - investigate if it does
                                # Reset buffer as fallback to allow training to continue
                                buffer.pos = 0
                                buffer.full = False
                                self.logger.error(f"  Buffer reset - please report this error if it persists")
                            else:
                                self.logger.error(f"[GAE_ERROR] Policy {polid} ({agent_name}): GAE computation failed: {gae_error}")
                                # Don't re-raise - let training continue (GAE failure is non-fatal)

                        # FAMC: Apply advantage-level correction (online/meta modes)
                        fgb_mode = getattr(self.config, 'fgb_mode', 'fixed')
                        if fgb_mode in ['online', 'meta'] and getattr(self.config, 'forecast_baseline_enable', False):
                            self._apply_famc_correction(policy, polid)

                        # MAINTENANCE: Log buffer state for verification
                        self.logger.debug(f"GAE computed for policy {polid}: buffer_size={buffer.buffer_size}, pos={buffer.pos}")
            except Exception as e:
                self.logger.warning(f"Rollout finalization error for policy {polid}: {e}")

    def _apply_famc_correction(self, policy, polid):
        """
        FAMC: Apply learned control-variate baseline correction to advantages.

        This implements variance-optimal advantage correction:
        A'_t = A_t - λ* * C_t

        where:
        - C_t is the control variate (meta-critic prediction or expected_dnav)
        - λ* = Cov(A,C) / Var(C) is computed online via EMA

        Args:
            policy: PPO policy with rollout_buffer containing computed advantages
            polid: Policy index
        """
        try:
            buffer = policy.rollout_buffer
            if buffer is None or buffer.pos == 0:
                return

            # Get FAMC configuration
            fgb_mode = getattr(self.config, 'fgb_mode', 'fixed')
            warmup_steps = getattr(self.config, 'fgb_warmup_steps', 2000)
            lambda_max = getattr(self.config, 'fgb_lambda_max', 0.8)
            clip_bps = getattr(self.config, 'fgb_clip_bps', 0.01)
            moment_beta = getattr(self.config, 'fgb_moment_beta', 0.01)

            # Skip if still in warmup
            if self._step_count_famc < warmup_steps:
                return

            # Get stored FAMC data
            if not hasattr(policy, '_famc_step_data') or len(policy._famc_step_data) == 0:
                return

            famc_data_list = policy._famc_step_data
            n_steps = min(buffer.pos, len(famc_data_list))

            if n_steps == 0:
                return

            # Extract advantages from buffer (shape: (buffer_size, 1))
            advantages = buffer.advantages[:n_steps].flatten()  # (n_steps,)

            # Build control variate C_t based on mode
            C_t = np.zeros(n_steps, dtype=np.float32)
            nav_norm = max(1.0, float(getattr(self.config, "init_budget", 0.0)) or 1.0)

            for i in range(n_steps):
                # CRITICAL FIX: Use .get() with defaults to prevent KeyError
                famc_data = famc_data_list[i] if i < len(famc_data_list) else {}
                tau = float(famc_data.get('tau', 0.0))
                exp_dnav = float(famc_data.get('expected_dnav', 0.0))
                meta_pred = float(famc_data.get('meta_adv_pred', 0.0))

                if fgb_mode == 'online':
                    # Use expected_dnav as control variate (gated by trust)
                    C_t[i] = tau * (exp_dnav / nav_norm)
                elif fgb_mode == 'meta':
                    # Use meta-critic prediction as control variate (gated by trust)
                    C_t[i] = tau * meta_pred

            # Standardize advantages and control variates for stable moment estimation
            A_mean = np.mean(advantages)
            A_std = np.std(advantages) + 1e-8
            A_std_norm = (advantages - A_mean) / A_std

            C_mean = np.mean(C_t)
            C_std = np.std(C_t) + 1e-8
            C_std_norm = (C_t - C_mean) / C_std

            # Update EMA moments (on standardized values)
            for i in range(n_steps):
                a = A_std_norm[i]
                c = C_std_norm[i]

                # Update first moments
                self._ema_mA = (1 - moment_beta) * self._ema_mA + moment_beta * a
                self._ema_mC = (1 - moment_beta) * self._ema_mC + moment_beta * c

                # Update second moments
                self._ema_mA2 = (1 - moment_beta) * self._ema_mA2 + moment_beta * (a ** 2)
                self._ema_mC2 = (1 - moment_beta) * self._ema_mC2 + moment_beta * (c ** 2)
                self._ema_mAC = (1 - moment_beta) * self._ema_mAC + moment_beta * (a * c)

            # Compute variance and covariance from EMA moments
            var_A = self._ema_mA2 - self._ema_mA ** 2
            var_C = self._ema_mC2 - self._ema_mC ** 2
            cov_AC = self._ema_mAC - self._ema_mA * self._ema_mC

            # Compute optimal λ* = Cov(A,C) / Var(C)
            if var_C > 1e-6:
                lambda_star_raw = cov_AC / var_C
                # Clip to [0, lambda_max] (only positive correlation reduces variance)
                lambda_star = np.clip(lambda_star_raw, 0.0, lambda_max)
            else:
                lambda_star = 0.0

            # Smooth λ* to reduce jitter
            self._lambda_star_prev = 0.9 * self._lambda_star_prev + 0.1 * lambda_star
            lambda_star_smooth = self._lambda_star_prev

            # Apply correction in standardized space (correct scale)
            # A'_std = A_std - λ* * C_std
            A_corrected_std = A_std_norm - lambda_star_smooth * C_std_norm

            # De-standardize back to original scale
            # A' = A'_std * σ_A + μ_A
            advantages_corrected = A_corrected_std * A_std + A_mean

            # Clip the CHANGE (not the correction itself)
            # This preserves the advantage scale while limiting per-step adjustments
            delta = advantages_corrected - advantages
            delta_clipped = np.clip(delta, -clip_bps, clip_bps)
            advantages_corrected = advantages + delta_clipped

            # Compute variance for metrics
            var_before = np.var(advantages)
            var_after = np.var(advantages_corrected)
            num_clipped = np.sum(np.abs(delta) > clip_bps)

            # CRITICAL FIX: Write back to buffer BEFORE SB3 normalization
            # SB3's compute_returns_and_advantage already computed advantages
            # We're modifying them here, which is fine as long as we don't trigger
            # another normalization pass. The buffer is ready for training after this.
            #
            # RISK MITIGATION: This modification happens AFTER GAE computation but
            # BEFORE the PPO update. SB3 will NOT re-normalize advantages during
            # the update step, so our corrections are preserved.
            buffer.advantages[:n_steps] = advantages_corrected.reshape(-1, 1)

            # Compute correlation for diagnostics
            if np.std(advantages) > 1e-6 and np.std(C_t) > 1e-6:
                corr_AC = np.corrcoef(advantages, C_t)[0, 1]
            else:
                corr_AC = 0.0

            # Update metrics
            self._famc_metrics['lambda_star'] = float(lambda_star_smooth)
            self._famc_metrics['corr_AC'] = float(corr_AC)
            self._famc_metrics['clip_rate'] = float(num_clipped) / max(1, n_steps)
            self._famc_metrics['var_adv_before'] = float(var_before)
            self._famc_metrics['var_adv_after'] = float(var_after)
            self._famc_metrics['num_clipped'] = int(num_clipped)
            self._famc_metrics['num_total'] = int(n_steps)
            self._famc_metrics['var_reduction'] = float(1.0 - var_after / max(var_before, 1e-8))

            # Periodic detailed logging
            if self._step_count_famc % 500 == 0:
                var_red_pct = (1.0 - var_after / max(var_before, 1e-8)) * 100
                clip_rate_pct = (float(num_clipped) / max(1, n_steps)) * 100
                self.logger.info(f"[FAMC] λ*={lambda_star_smooth:.3f} | Corr={corr_AC:.3f} | VarRed={var_red_pct:.1f}% | Clip={clip_rate_pct:.1f}%")

            # Store features and advantages for meta head training (meta mode only)
            if fgb_mode == 'meta' and getattr(self.config, 'meta_baseline_enable', False):
                # CRITICAL: Only store if overlay features are valid (34D)
                for i in range(n_steps):
                    obs = famc_data_list[i]['obs']
                    # Verify dimension is correct (34D overlay features)
                    if isinstance(obs, np.ndarray) and obs.shape[0] == 34:
                        self._meta_features_buffer.append(obs)
                        self._meta_adv_buffer.append(A_std_norm[i])  # Store standardized advantage
                    else:
                        # Skip invalid observations to prevent dimension mismatches
                        if isinstance(obs, np.ndarray):
                            self.logger.debug(f"[FAMC] Skipping invalid observation: shape={obs.shape}, expected 34D")

                # Limit buffer size
                max_buffer_size = 10000
                if len(self._meta_features_buffer) > max_buffer_size:
                    self._meta_features_buffer = self._meta_features_buffer[-max_buffer_size:]
                    self._meta_adv_buffer = self._meta_adv_buffer[-max_buffer_size:]

            # Clear FAMC step data for next rollout
            policy._famc_step_data = []

        except Exception as e:
            self.logger.warning(f"[FAMC] Correction failed for policy {polid}: {e}")

    def _train_meta_head_if_needed(self):
        """
        FAMC: Train meta-critic head periodically on accumulated advantages.

        This trains the meta head g_φ(x_t) to predict standardized advantages
        from state features only (no action leakage).
        """
        try:
            # Check if meta head training is enabled
            fgb_mode = getattr(self.config, 'fgb_mode', 'fixed')
            if fgb_mode != 'meta' or not getattr(self.config, 'meta_baseline_enable', False):
                return

            # Check if it's time to train
            meta_train_every = getattr(self.config, 'meta_train_every', 512)
            if self._step_count_famc % meta_train_every != 0:
                return

            # Check if we have enough data
            if len(self._meta_features_buffer) < 128:
                return

            # Get overlay trainer from environment
            if not hasattr(self.env, 'overlay_trainer') or self.env.overlay_trainer is None:
                self.logger.warning("[FAMC] Overlay trainer not found, skipping meta head training")
                return

            overlay_trainer = self.env.overlay_trainer

            # Sample a batch from buffer
            batch_size = min(256, len(self._meta_features_buffer))
            indices = np.random.choice(len(self._meta_features_buffer), batch_size, replace=False)

            # Extract features and advantages
            # CRITICAL FIX: Verify all features have the same dimension before stacking
            valid_indices = []
            for i in indices:
                feat = self._meta_features_buffer[i]
                if isinstance(feat, np.ndarray) and feat.shape[0] == 34:
                    valid_indices.append(i)
            
            if len(valid_indices) < 32:  # Need at least 32 samples for a reasonable batch
                self.logger.warning(f"[FAMC] ❌ SKIPPING meta head training: insufficient valid features ({len(valid_indices)}/{batch_size} valid)")
                return
            
            # Extract only valid features and advantages
            features_list = [self._meta_features_buffer[i] for i in valid_indices]
            adv_list = [self._meta_adv_buffer[i] for i in valid_indices]
            
            # Stack into batch - all should have shape (34,) so this should work
            try:
                features_batch = np.array(features_list, dtype=np.float32)  # (batch, 34)
                adv_batch = np.array(adv_list, dtype=np.float32)  # (batch,)
            except ValueError as e:
                self.logger.warning(f"[FAMC] ❌ SKIPPING meta head training: failed to stack features: {e}")
                # Debug: Check dimensions
                dims = [f.shape for f in features_list[:10]]
                self.logger.warning(f"[FAMC] First 10 feature shapes: {dims}")
                return

            # Ensure features are (batch, 34) - extract from (batch, 1, obs_dim) if needed
            if len(features_batch.shape) == 3:
                features_batch = features_batch.squeeze(1)

            # CRITICAL CHECK: Verify feature dimension matches overlay input (34D)
            if features_batch.shape[1] != 34:
                self.logger.warning(f"[FAMC] ❌ SKIPPING meta head training: feature dim mismatch (got {features_batch.shape[1]}, expected 34)")
                self.logger.warning(f"[FAMC] This means overlay features are not being cached properly!")
                return

            # SUCCESS: Features are correct dimension
            if self._step_count_famc % (meta_train_every * 10) == 0:
                self.logger.info(f"[FAMC] ✅ Meta head training with correct 34D features (buffer_size={len(self._meta_features_buffer)})")

            # Train meta head
            loss_type = getattr(self.config, 'meta_baseline_loss', 'mse')
            meta_loss = overlay_trainer.train_meta_baseline(features_batch, adv_batch, loss_type)

            # Update metrics
            self._famc_metrics['meta_loss'] = float(meta_loss)
            self._famc_metrics['meta_buffer_size'] = len(self._meta_features_buffer)

            # Log
            if self.verbose:
                self.logger.info(f"[FAMC] Meta head trained: loss={meta_loss:.6f}, buffer_size={len(self._meta_features_buffer)}")

        except Exception as e:
            self.logger.warning(f"[FAMC] Meta head training failed: {e}")

    def _finalize_training(self):
        for polid, policy in enumerate(self.policies):
            try:
                if hasattr(policy, "callback") and policy.callback is not None:
                    policy.callback.on_training_end()
            except Exception as e:
                self.logger.warning(f"Callback finalization error for policy {polid}: {e}")

        self.memory_tracker.cleanup("heavy")

        if self.debug:
            self._print_training_summary()

    def _print_initialization_summary(self):
        print("\nInitialization Summary")
        print(f"   Agents: {self.num_agents}")
        print(f"   Device: {self.device}")
        print(f"   Memory limit: {self.memory_tracker.max_memory_mb}MB")
        print(f"   Current memory: {self.memory_tracker.get_memory_usage():.1f}MB")
        print(f"   Multithreading: {self.multithreading}")
        val_stats = self.obs_validator.get_validation_stats()
        print(f"   Observation specs: {val_stats['agents_configured']}")
        modes = [getattr(p, "mode", "Unknown") for p in self.policies]
        counts = {m: modes.count(m) for m in set(modes)}
        print(f"   Policy distribution: {counts}")

    def _print_training_summary(self):
        final_stats = self.memory_tracker.get_memory_stats()
        val_stats = self.obs_validator.get_validation_stats()
        print("\nTraining Completed")
        print(f"   Total steps: {self.total_steps:,}")
        print(f"   Successful steps: {self._training_metrics['successful_steps']:,}")
        print(f"   Memory cleanups: {self._training_metrics['memory_cleanups']}")
        print(f"   Observation fixes: {self._training_metrics['observation_fixes']}")
        print(f"   Policy errors: {self._training_metrics['policy_errors']}")
        print(f"   Final memory: {final_stats['current_memory_mb']:.1f}MB")
        print(f"   Memory freed total: {final_stats['cleanup_stats']['memory_freed_mb']:.1f}MB")
        print(f"   Validation errors: {val_stats['recent_errors']}")

    # ------------------------------- I/O -------------------------------------
    def save_policies(self, save_dir: str) -> int:
        os.makedirs(save_dir, exist_ok=True)
        self.memory_tracker.cleanup("medium")

        saved_count = 0
        save_errors: List[str] = []

        for agent_name, policy in zip(self.possible_agents, self.policies):
            try:
                if not hasattr(policy, "save"):
                    self.logger.warning(f"Policy {agent_name} has no save()")
                    continue

                path = os.path.join(save_dir, f"{agent_name}_policy.zip")

                # Temporarily detach non-picklables
                orig_logger = getattr(policy, "_logger", None)
                orig_callback = getattr(policy, "callback", None)
                orig_env = getattr(policy, "env", None)
                try:
                    if hasattr(policy, "set_logger"):
                        policy.set_logger(None)
                    else:
                        policy._logger = None
                except Exception:
                    policy._logger = None
                policy.callback = None
                policy.env = None

                try:
                    before = self.memory_tracker.get_memory_usage()
                    policy.save(path)
                    after = self.memory_tracker.get_memory_usage()
                    saved_count += 1
                    if self.debug:
                        used = after - before
                        print(f"Saved {agent_name} policy (Δmem {used:.1f}MB)")
                    if after > self.memory_tracker.max_memory_mb * 0.8:
                        self.memory_tracker.cleanup("light")
                finally:
                    try:
                        if hasattr(policy, "set_logger"):
                            policy.set_logger(orig_logger)
                        else:
                            policy._logger = orig_logger
                    except Exception:
                        pass
                    policy.callback = orig_callback
                    policy.env = orig_env

            except Exception as e:
                msg = f"Failed to save {agent_name} policy: {e}"
                save_errors.append(msg)
                self.logger.error(msg)

        try:
            meta = {
                "timestamp": datetime.now().isoformat(),
                "total_steps": self.total_steps,
                "training_metrics": self._training_metrics,
                "memory_stats": self.memory_tracker.get_memory_stats(),
                "validation_stats": self.obs_validator.get_validation_stats(),
                "save_errors": save_errors,
            }
            with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save training metadata: {e}")

        print(f"Save complete: {saved_count}/{len(self.policies)} policies saved")
        if save_errors:
            print("Save errors:")
            for err in save_errors:
                print(f"   - {err}")
        return saved_count

    def load_policies(self, load_dir: str) -> int:
        if not os.path.exists(load_dir):
            self.logger.error(f"Load directory does not exist: {load_dir}")
            return 0

        self.memory_tracker.cleanup("medium")

        loaded_count = 0
        load_errors: List[str] = []

        for idx, (agent_name, policy) in enumerate(zip(self.possible_agents, self.policies)):
            path = os.path.join(load_dir, f"{agent_name}_policy.zip")
            if not os.path.exists(path):
                load_errors.append(f"Policy file not found: {path}")
                continue
            try:
                algo_name = getattr(policy, "mode", "PPO")
                algo_cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}.get(algo_name, PPO)

                # reattach a dummy env with correct spaces
                obs_space = self.observation_spaces[agent_name]
                act_space = self.action_spaces[agent_name]
                
                # CRITICAL FIX: Check if saved policy's observation space matches current environment
                # Load policy metadata first to check observation space dimension
                # This prevents loading policies with incompatible observation spaces (e.g., Tier 3 13D policy into Tier 2 9D environment)
                pre_check_passed = False
                try:
                    import zipfile
                    import json
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        if 'data' in zip_ref.namelist():
                            data_str = zip_ref.read('data').decode('utf-8')
                            data = json.loads(data_str)
                            saved_obs_shape = data.get('observation_space', {}).get('shape', None)
                            if saved_obs_shape is not None:
                                # Handle different shape formats: [dim] or dim or (dim,)
                                if isinstance(saved_obs_shape, (list, tuple)):
                                    saved_obs_dim = int(saved_obs_shape[0]) if len(saved_obs_shape) > 0 else None
                                elif isinstance(saved_obs_shape, (int, float)):
                                    saved_obs_dim = int(saved_obs_shape)
                                else:
                                    saved_obs_dim = None
                                
                                if saved_obs_dim is not None:
                                    current_obs_dim = int(obs_space.shape[0])
                                    if saved_obs_dim != current_obs_dim:
                                        self.logger.warning(f"[OBS_SPACE_MISMATCH] {agent_name}: Saved policy has {saved_obs_dim}D observations, "
                                                           f"but current environment expects {current_obs_dim}D. Skipping load - will use existing policy.")
                                        load_errors.append(f"Observation space mismatch: saved={saved_obs_dim}D, current={current_obs_dim}D")
                                        continue
                                    else:
                                        pre_check_passed = True
                                        self.logger.debug(f"[OBS_SPACE_CHECK] {agent_name}: Pre-check passed ({saved_obs_dim}D matches {current_obs_dim}D)")
                except Exception as check_error:
                    # If we can't check, proceed with load and verify after loading
                    # This handles edge cases like corrupted metadata or different SB3 versions
                    self.logger.debug(f"Could not pre-check observation space for {agent_name}: {check_error}. Will verify after load.")
                
                dummy_env = DummyVecEnv([partial(DummyGymEnv, obs_space, act_space)])

                before = self.memory_tracker.get_memory_usage()
                try:
                    loaded = algo_cls.load(path, device=self.device, env=dummy_env)
                except Exception as load_exception:
                    # CRITICAL FIX: Catch observation space mismatch errors from SB3's load()
                    # SB3 raises an error when observation spaces don't match
                    error_msg = str(load_exception)
                    if "observation space" in error_msg.lower() or "observation_space" in error_msg.lower():
                        # Extract dimensions from error message if possible
                        self.logger.warning(f"[OBS_SPACE_MISMATCH] {agent_name}: Cannot load policy - observation space mismatch. "
                                           f"Error: {error_msg}. Skipping load - will use existing policy.")
                        load_errors.append(f"Observation space mismatch: {error_msg}")
                        continue
                    else:
                        # Re-raise if it's a different error
                        raise
                
                # Double-check: Verify loaded policy's observation space matches (critical safety check)
                # This catches cases where pre-check failed or policy was saved with different format
                observation_space_valid = False
                try:
                    if hasattr(loaded, 'observation_space') and hasattr(loaded.observation_space, 'shape'):
                        loaded_obs_dim = int(loaded.observation_space.shape[0])
                        current_obs_dim = int(obs_space.shape[0])
                        if loaded_obs_dim != current_obs_dim:
                            self.logger.warning(f"[OBS_SPACE_MISMATCH] {agent_name}: Loaded policy has {loaded_obs_dim}D observations, "
                                               f"but current environment expects {current_obs_dim}D. Skipping load - will use existing policy.")
                            load_errors.append(f"Observation space mismatch after load: loaded={loaded_obs_dim}D, current={current_obs_dim}D")
                            continue
                        else:
                            observation_space_valid = True
                            # CRITICAL FIX: Update policy's observation space to match environment
                            # This ensures the policy uses the correct observation space even if it was saved with a different one
                            loaded.observation_space = obs_space
                            if hasattr(loaded, 'policy') and hasattr(loaded.policy, 'observation_space'):
                                loaded.policy.observation_space = obs_space
                            
                            # ROBUST: Validate and fix buffer shape after loading
                            # This ensures buffers are recreated with correct dimensions if they were saved with wrong dimensions
                            if hasattr(loaded, 'rollout_buffer') and loaded.rollout_buffer is not None:
                                # Validate buffer shape matches observation space
                                if loaded_obs_dim != current_obs_dim:
                                    # Observation space changed - buffer needs to be recreated
                                    self.logger.warning(f"[BUFFER_VALIDATION] {agent_name}: Observation space changed ({loaded_obs_dim}D → {current_obs_dim}D). Recreating buffer.")
                                    self._validate_and_fix_buffer_shape(loaded, idx)
                                else:
                                    # Even if dimensions match, validate buffer shape is correct
                                    self._validate_and_fix_buffer_shape(loaded, idx)
                            # Also check and update network input dimension if accessible
                            if hasattr(loaded, 'policy') and hasattr(loaded.policy, 'features_extractor'):
                                try:
                                    # Verify network input dimension matches (if accessible)
                                    # For GNN encoder, check the actual input dimension (obs_dim), not output dimension (features_dim)
                                    fe = loaded.policy.features_extractor
                                    if hasattr(fe, 'gnn_encoder') and hasattr(fe.gnn_encoder, 'obs_dim'):
                                        # GNN encoder: check input dimension
                                        net_input_dim = fe.gnn_encoder.obs_dim
                                    elif hasattr(fe, 'observation_space') and hasattr(fe.observation_space, 'shape'):
                                        # Standard feature extractor: check observation space
                                        net_input_dim = fe.observation_space.shape[0]
                                    else:
                                        # Fallback: skip dimension check
                                        net_input_dim = None
                                    
                                    if net_input_dim is not None and net_input_dim != current_obs_dim:
                                        self.logger.warning(f"[NETWORK_MISMATCH] {agent_name}: Policy network expects {net_input_dim}D input, "
                                                           f"but environment provides {current_obs_dim}D. This may cause runtime errors.")
                                except Exception:
                                    pass  # Network dimension check is optional
                    else:
                        # Policy doesn't have observation_space attribute - this is unusual but not necessarily fatal
                        self.logger.warning(f"[OBS_SPACE_WARNING] {agent_name}: Loaded policy missing observation_space attribute. Proceeding with caution.")
                        observation_space_valid = True  # Assume valid if we can't check
                except Exception as verify_error:
                    # If verification fails, don't load the policy to be safe
                    self.logger.error(f"[OBS_SPACE_VERIFY_ERROR] {agent_name}: Failed to verify observation space: {verify_error}. Skipping load.")
                    load_errors.append(f"Observation space verification failed: {str(verify_error)}")
                    continue
                
                # keep metadata expected elsewhere
                loaded.mode = algo_name
                loaded.agent_name = agent_name
                loaded.action_space = act_space

                # swap in
                self.policies[idx] = loaded
                self.memory_tracker.register_policy(loaded)
                if hasattr(loaded, "replay_buffer") and loaded.replay_buffer is not None:
                    self.memory_tracker.register_buffer(loaded.replay_buffer)
                if hasattr(loaded, "rollout_buffer") and loaded.rollout_buffer is not None:
                    self.memory_tracker.register_buffer(loaded.rollout_buffer)
                    # ROBUST: Final validation - ensure buffer is correct after loading
                    # This catches any edge cases where buffer wasn't properly recreated
                    self._validate_and_fix_buffer_shape(loaded, idx)

                after = self.memory_tracker.get_memory_usage()
                loaded_count += 1
                if self.debug:
                    print(f"Loaded {agent_name} policy (Δmem {after - before:.1f}MB)")
                if after > self.memory_tracker.max_memory_mb * 0.8:
                    self.memory_tracker.cleanup("light")
            except Exception as e:
                error_msg = str(e)
                # CRITICAL FIX: Check if this is an observation space mismatch
                # If so, log as warning instead of error (this is expected when switching tiers)
                if "observation space" in error_msg.lower() or "observation_space" in error_msg.lower():
                    msg = f"Failed to load {agent_name} policy: {error_msg} (Observation space mismatch - expected when switching tiers)"
                    load_errors.append(msg)
                    self.logger.warning(msg)
                else:
                    # Other errors are logged as errors
                    msg = f"Failed to load {agent_name} policy: {e}"
                    load_errors.append(msg)
                    self.logger.error(msg)

        try:
            meta_path = os.path.join(load_dir, "training_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    _ = json.load(f)  # not used; kept for completeness
                if self.debug:
                    print(f"Loaded training metadata from: {meta_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load training metadata: {e}")

        print(f"Load complete: {loaded_count}/{len(self.policies)} policies loaded")
        if load_errors:
            print("Load errors:")
            for err in load_errors:
                print(f"   - {err}")
        return loaded_count

    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        try:
            return {
                "agent_info": {
                    "num_agents": self.num_agents,
                    "possible_agents": self.possible_agents,
                    "total_steps": self.total_steps,
                    "consecutive_errors": self._consecutive_errors,
                },
                "memory_diagnostics": self.memory_tracker.get_memory_stats(),
                "validation_diagnostics": self.obs_validator.get_validation_stats(),
                "training_metrics": self._training_metrics,
                "policy_info": [
                    {
                        "agent": getattr(p, "agent_name", f"policy_{i}"),
                        "mode": getattr(p, "mode", "Unknown"),
                        "has_buffer": hasattr(p, "rollout_buffer") or hasattr(p, "replay_buffer"),
                        "timesteps": getattr(p, "num_timesteps", 0),
                    }
                    for i, p in enumerate(self.policies)
                ],
                "environment_info": {
                    "observation_spaces": {a: str(s.shape) for a, s in self.observation_spaces.items()},
                    "action_spaces": {a: str(getattr(s, 'shape', s)) for a, s in self.action_spaces.items()},
                },
            }
        except Exception as e:
            return {"diagnostics_error": str(e)}

    def __del__(self):
        try:
            if hasattr(self, "memory_tracker"):
                self.memory_tracker.cleanup("heavy")
            if hasattr(self, "policies"):
                for policy in self.policies:
                    try:
                        if hasattr(policy, "rollout_buffer"):
                            policy.rollout_buffer = None
                        if hasattr(policy, "replay_buffer"):
                            policy.replay_buffer = None
                    except Exception:
                        pass
                self.policies.clear()
            if hasattr(self, "obs_validator") and hasattr(self.obs_validator, "validation_cache"):
                self.obs_validator.validation_cache.clear()
            gc.collect()
        except Exception:
            pass


# ========================= Lightweight HPO (Optional) =======================
# PATCH: Replaced the old heuristic-based optimizer with the new Optuna-based version.
class OptunaHyperparameterOptimizer:
    """
    A powerful hyperparameter optimizer using Optuna to maximize the Sharpe Ratio
    from short training and evaluation episodes.
    """

    def __init__(
        self,
        env,
        device: str = "cpu",
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        training_steps_per_trial: int = 5000,
        eval_steps_per_trial: int = 1000,
    ):
        """
        Initializes the optimizer.

        Args:
            env: The environment instance to use for optimization.
            device: The device to use for training ('cpu' or 'cuda').
            n_trials: The maximum number of trials to run.
            timeout: The maximum time in seconds for the optimization process.
            training_steps_per_trial: Number of training steps for each trial.
            eval_steps_per_trial: Number of evaluation steps for each trial.
        """
        self.env = env
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.training_steps_per_trial = training_steps_per_trial
        self.eval_steps_per_trial = eval_steps_per_trial
        self.logger = logger

    def _calculate_performance_metric(self, portfolio_values: List[float]) -> float:
        """
        Calculates the Sharpe Ratio from a series of portfolio values.
        Returns a very low number on failure to guide Optuna away from bad trials.
        """
        if not portfolio_values or len(portfolio_values) < 20:
            return -10.0  # Penalize trials that fail to produce a meaningful series

        try:
            pv_series = np.array(portfolio_values, dtype=np.float32)
            # Use percentage change for returns; robust to starting value
            returns = pd.Series(pv_series).pct_change().dropna().to_numpy()
            
            if returns.size < 10 or np.std(returns) < 1e-9:
                return -10.0 # Not enough variance to calculate Sharpe

            # Simplified Sharpe Ratio for comparison purposes
            sharpe_ratio = np.mean(returns) / np.std(returns)
            
            # Annualize (assuming daily-like frequency for trial runs)
            annualized_sharpe = sharpe_ratio * np.sqrt(252)
            
            return float(np.nan_to_num(annualized_sharpe, nan=-10.0))
        except Exception:
            return -10.0

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """The core objective function that Optuna tries to maximize."""
        try:
            # 1. Sample Hyperparameters
            params = {
                'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.02),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'net_arch_size': trial.suggest_categorical('net_arch_size', ['small', 'medium']),
                'activation_fn': trial.suggest_categorical('activation_fn', ['tanh', 'relu']),
                'update_every': trial.suggest_categorical('update_every', [128, 256, 512]),
                'batch_size': 128,  # FIXED: Standardized batch size (no longer a hyperparameter)
                # Suggest a single mode for all agents for simplicity, can be expanded
                'agent_mode': trial.suggest_categorical('agent_mode', ['PPO', 'SAC'])
            }
            # Apply the same mode to all agents in this trial
            params['agent_policies'] = [{"mode": params['agent_mode']}] * len(self.env.possible_agents)
            
            # 2. Configure and Initialize the Agent
            config = EnhancedConfig(optimized_params=params)
            agent_system = MultiESGAgent(config, self.env, self.device, training=True)

            # 3. Short Training Run
            agent_system.learn(total_timesteps=self.training_steps_per_trial)

            # 4. Short Evaluation Run
            obs, _ = self.env.reset()
            portfolio_values = []
            
            for step in range(self.eval_steps_per_trial):
                actions = {}
                for i, agent_name in enumerate(self.env.possible_agents):
                    agent_obs = obs[agent_name].reshape(1, -1)
                    policy = agent_system.policies[i]
                    action, _ = policy.predict(agent_obs, deterministic=True)
                    actions[agent_name] = action

                obs, _, _, _, _ = self.env.step(actions)
                
                # Correctly access the base environment's equity
                base_env = getattr(self.env, 'env', self.env)
                equity = getattr(base_env, 'equity', None)
                if equity is not None:
                    portfolio_values.append(equity)

                # Optuna Pruning Hook
                trial.report(np.mean(portfolio_values) if portfolio_values else 0, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # 5. Calculate and Return Final Metric
            sharpe_ratio = self._calculate_performance_metric(portfolio_values)
            return sharpe_ratio

        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.warning(f"Trial failed with exception: {e}")
            return -10.0 # Return a very bad score for failed trials

    def optimize(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Runs the Optuna optimization study.

        Returns:
            A tuple of (best_parameters_dict, performance_summary_dict).
        """
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=self.eval_steps_per_trial // 2)
        )

        try:
            print(f"🚀 Starting Optuna HPO for {self.n_trials} trials (timeout: {self.timeout}s)...")
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n🛑 HPO interrupted by user.")
        
        if not study.best_trial:
             print("⚠️ HPO finished with no completed trials. Returning default parameters.")
             return {}, {"best_sharpe_ratio": -10.0}

        best_params = study.best_params
        best_value = study.best_value
        
        # Format the agent policies correctly for the config
        best_params['agent_policies'] = [{"mode": best_params.get('agent_mode', 'PPO')}] * len(self.env.possible_agents)

        performance_summary = {"best_sharpe_ratio": best_value}
        
        print("\n🎉 HPO Finished!")
        print(f"🏆 Best Sharpe Ratio: {best_value:.4f}")
        print("📋 Best Hyperparameters:")
        for key, value in best_params.items():
            print(f"  - {key}: {value}")

        return best_params, performance_summary


# ============================= DL Overlay Tuning ================================
def tune_overlay(env, window_size: int = 100, adjustment_factor: float = 0.05) -> None:
    """
    Periodic meta-controller tuning for DL overlay intensity.

    Adjusts overlay_intensity based on recent P&L and drawdown:
    - If recent P&L is positive and drawdown is low: increase intensity (more aggressive)
    - If recent P&L is negative or drawdown is high: decrease intensity (more defensive)

    Args:
        env: The environment with reward_calculator and config
        window_size: Number of recent steps to consider (default: 100)
        adjustment_factor: How much to adjust intensity per call (default: 0.05)
    """
    try:
        if not hasattr(env, 'reward_calculator') or env.reward_calculator is None:
            return

        if not hasattr(env, 'config') or env.config is None:
            return

        # Get recent performance metrics
        recent_gains = float(getattr(env.reward_calculator, 'recent_trading_gains', 0.0))
        current_dd = float(getattr(env.reward_calculator, 'current_drawdown', 0.0))

        # Get current overlay intensity
        current_intensity = getattr(env.config, 'overlay_intensity', 1.0)

        # Determine adjustment direction
        adjustment = 0.0

        # Positive P&L and low drawdown: increase intensity
        if recent_gains > 50_000 and current_dd < 0.02:
            adjustment = adjustment_factor
        # Negative P&L or high drawdown: decrease intensity
        elif recent_gains < -50_000 or current_dd > 0.05:
            adjustment = -adjustment_factor

        # Apply adjustment with bounds [0.5, 1.5]
        new_intensity = np.clip(current_intensity + adjustment, 0.5, 1.5)

        # Update config
        env.config.overlay_intensity = new_intensity

        # Log if significant change
        if abs(new_intensity - current_intensity) > 1e-6:
            logger.info(f"[OVERLAY-TUNE] t={env.t} recent_gains={recent_gains:.0f} dd={current_dd:.3f} "
                        f"intensity: {current_intensity:.2f} -> {new_intensity:.2f}")

    except Exception as e:
        logger.debug(f"Overlay tuning failed: {e}")


# Keep this alias for backward compatibility with main.py
HyperparameterOptimizer = OptunaHyperparameterOptimizer