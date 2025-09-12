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
import logging
from datetime import datetime
import random
import json
import pandas as pd
import optuna
from config import EnhancedConfig


warnings.filterwarnings("ignore", category=UserWarning)

LEARNING_MODES = ["PPO", "SAC", "TD3"]

__all__ = ["MultiESGAgent", "HyperparameterOptimizer"]


# ============================= Memory Tracker ================================
class EnhancedMemoryTracker:
    """Enhanced memory tracker with SB3-specific cleanup."""

    def __init__(self, max_memory_mb=None, config=None):
        # Get memory limit from config if available
        if config and hasattr(config, 'metacontroller_memory_mb'):
            max_memory_mb = max_memory_mb or config.metacontroller_memory_mb
        else:
            max_memory_mb = max_memory_mb or 6000
        self.max_memory_mb = max_memory_mb
        self.cleanup_counter = 0
        self.memory_history = deque(maxlen=200)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # Track SB3 components that need special cleanup
        self.tracked_policies: List[weakref.ReferenceType] = []
        self.tracked_buffers: List[weakref.ReferenceType] = []
        self.tracked_envs: List[weakref.ReferenceType] = []

        self.cleanup_thresholds = {
            "light": max_memory_mb * 0.70,
            "medium": max_memory_mb * 0.85,
            "heavy": max_memory_mb * 0.95,
        }
        self.cleanup_stats = {
            "light_cleanups": 0,
            "medium_cleanups": 0,
            "heavy_cleanups": 0,
            "memory_freed_mb": 0.0,
        }

    def register_policy(self, policy):
        self.tracked_policies.append(weakref.ref(policy))

    def register_buffer(self, buffer):
        self.tracked_buffers.append(weakref.ref(buffer))

    def register_env(self, env):
        self.tracked_envs.append(weakref.ref(env))

    def get_memory_usage(self) -> float:
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return float(memory_info.rss) / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def should_cleanup(self, force=False) -> Tuple[Optional[str], float]:
        with self.lock:
            current_memory = self.get_memory_usage()
            self.memory_history.append(current_memory)
            self.cleanup_counter += 1

            if force:
                return "heavy", current_memory
            if current_memory > self.cleanup_thresholds["heavy"]:
                return "heavy", current_memory
            if current_memory > self.cleanup_thresholds["medium"]:
                return "medium", current_memory
            if current_memory > self.cleanup_thresholds["light"] or (self.cleanup_counter % 800 == 0):
                return "light", current_memory
            return None, current_memory

    def cleanup(self, level="light") -> float:
        with self.lock:
            memory_before = self.get_memory_usage()

            if level in ("light", "medium", "heavy"):
                self._cleanup_light()
            if level in ("medium", "heavy"):
                self._cleanup_medium()
            if level == "heavy":
                self._cleanup_heavy()

            self._cleanup_basic()

            memory_after = self.get_memory_usage()
            memory_freed = max(0.0, memory_before - memory_after)

            key = f"{level}_cleanups"
            if key in self.cleanup_stats:
                self.cleanup_stats[key] += 1
            self.cleanup_stats["memory_freed_mb"] += memory_freed

            if memory_freed > 50:
                self.logger.info(
                    f"Memory cleanup ({level}): {memory_before:.1f}MB → "
                    f"{memory_after:.1f}MB (freed {memory_freed:.1f}MB)"
                )
            return memory_freed

    def _cleanup_light(self):
        self.tracked_policies = [ref for ref in self.tracked_policies if ref() is not None]
        self.tracked_buffers = [ref for ref in self.tracked_buffers if ref() is not None]
        self.tracked_envs = [ref for ref in self.tracked_envs if ref() is not None]
        gc.collect()

    def _cleanup_medium(self):
        for buffer_ref in self.tracked_buffers:
            buffer = buffer_ref()
            if buffer is None:
                continue
            try:
                if hasattr(buffer, "reset"):
                    buffer.reset()
                elif hasattr(buffer, "clear"):
                    buffer.clear()
            except Exception as e:
                self.logger.warning(f"Failed to clear buffer: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _cleanup_heavy(self):
        for policy_ref in self.tracked_policies:
            policy = policy_ref()
            if policy is None:
                continue
            try:
                self._cleanup_policy_memory(policy)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup policy memory: {e}")

        for env_ref in self.tracked_envs:
            env = env_ref()
            if env is not None and hasattr(env, "reset"):
                try:
                    env.reset()
                except Exception as e:
                    self.logger.warning(f"Failed to reset environment: {e}")

    def _cleanup_policy_memory(self, policy):
        # Optimizer state
        try:
            optimizer = getattr(getattr(policy, "policy", None), "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "state"):
                # Zero out grads and clear optimizer states
                for group in optimizer.param_groups:
                    for p in group.get("params", []):
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                optimizer.state.clear()
        except Exception as e:
            self.logger.warning(f"Failed to clear optimizer: {e}")

        # Replay / rollout buffers
        try:
            if hasattr(policy, "replay_buffer") and policy.replay_buffer is not None:
                if hasattr(policy.replay_buffer, "reset"):
                    policy.replay_buffer.reset()
                elif hasattr(policy.replay_buffer, "_storage"):
                    policy.replay_buffer._storage = []
                    policy.replay_buffer.pos = 0
                    policy.replay_buffer.full = False
        except Exception as e:
            self.logger.warning(f"Failed to clear replay buffer: {e}")

        try:
            if hasattr(policy, "rollout_buffer") and policy.rollout_buffer is not None:
                policy.rollout_buffer.reset()
        except Exception as e:
            self.logger.warning(f"Failed to clear rollout buffer: {e}")

    def _cleanup_basic(self):
        for _ in range(2):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, Any]:
        current_memory = self.get_memory_usage()
        return {
            "current_memory_mb": current_memory,
            "max_memory_mb": self.max_memory_mb,
            "memory_usage_pct": (current_memory / max(1, self.max_memory_mb)) * 100.0,
            "tracked_policies": len([ref for ref in self.tracked_policies if ref() is not None]),
            "tracked_buffers": len([ref for ref in self.tracked_buffers if ref() is not None]),
            "tracked_envs": len([ref for ref in self.tracked_envs if ref() is not None]),
            "cleanup_stats": self.cleanup_stats.copy(),
            "memory_history": list(self.memory_history),
        }


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
                    print(f"Callback memory cleanup ({cleanup_level}): freed {memory_freed:.1f}MB")
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
                print(f"Policy {polid} error: {exc}")
        return

    max_workers = max_workers or min(4, len(policies))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [(executor.submit(function, polid, policy), polid) for polid, policy in enumerate(policies)]
        for future, polid in futures:
            try:
                future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                print(f"Policy {polid} training timeout")
            except Exception as exc:
                print(f"Policy {polid} generated an exception: {exc}")


# ======================== Observation Validator =============================
class EnhancedObservationValidator:
    """Enhanced observation validator with comprehensive dimension checking."""

    def __init__(self, env, debug=False):
        self.env = env
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.observation_specs = self._build_enhanced_specs()
        self.validation_cache: Dict[Any, bool] = {}
        self.validation_errors = deque(maxlen=100)

    def _build_enhanced_specs(self) -> Dict[str, Dict[str, Any]]:
        specs = {}
        for agent in self.env.possible_agents:
            try:
                obs_space = self.env.observation_space(agent)
                expected_dim = int(obs_space.shape[0])
                low = np.clip(obs_space.low.copy(), -1e6, 1e6).astype(np.float32)
                high = np.clip(obs_space.high.copy(), -1e6, 1e6).astype(np.float32)
                if np.any(low >= high):
                    self.logger.warning(f"Invalid bounds for {agent}, using safe defaults")
                    low = np.full(expected_dim, -10.0, np.float32)
                    high = np.full(expected_dim, 10.0, np.float32)
                specs[agent] = {
                    "expected_dim": expected_dim,
                    "low": low,
                    "high": high,
                    "dtype": np.float32,
                    "shape": (expected_dim,),
                    "original_space": obs_space,
                }
                if self.debug:
                    print(
                        f"Enhanced specs for {agent}: dim={expected_dim}, "
                        f"bounds=[{low.min():.2f}, {high.max():.2f}]"
                    )
            except Exception as e:
                self.logger.error(f"Failed to build specs for {agent}: {e}")
                fallback_dim = self._estimate_agent_dimension(agent)
                specs[agent] = {
                    "expected_dim": fallback_dim,
                    "low": np.full(fallback_dim, -10.0, np.float32),
                    "high": np.full(fallback_dim, 10.0, np.float32),
                    "dtype": np.float32,
                    "shape": (fallback_dim,),
                    "original_space": None,
                }
        return specs

    def _estimate_agent_dimension(self, agent: str) -> int:
        # FIXED: Correct total dimensions (base + forecast + confidence)
        # Based on actual agent forecast allocations from generator.py:
        # - investor_0: 6 base + 12 forecast + 1 confidence = 19
        # - battery_operator_0: 4 base + 12 forecast + 1 confidence = 17
        # - risk_controller_0: 9 base + 12 forecast (no confidence) = 21
        # - meta_controller_0: 11 base + 20 forecast + 1 confidence = 32
        estimates = {
            "investor_0": 19,           # UPDATED: +1 for confidence
            "battery_operator_0": 17,   # UPDATED: +1 for confidence
            "risk_controller_0": 21,    # No change (no confidence)
            "meta_controller_0": 32,    # UPDATED: +1 for confidence
        }
        return int(estimates.get(agent, 20))

    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> bool:
        if not isinstance(obs_dict, dict):
            self.validation_errors.append("obs_dict is not a dictionary")
            return False
        for agent in self.env.possible_agents:
            if agent not in obs_dict:
                self.validation_errors.append(f"Missing agent {agent} in observation dictionary")
                return False
            if not self.validate_single_observation(agent, obs_dict[agent]):
                return False
        return True

    def _obs_signature(self, agent: str, obs: Any):
        if isinstance(obs, np.ndarray):
            try:
                return (agent, obs.shape, obs.dtype, hash(obs.tobytes()))
            except Exception:
                return (agent, obs.shape, obs.dtype, None)
        return (agent, type(obs), str(obs)[:64])

    def validate_single_observation(self, agent: str, obs: Any) -> bool:
        sig = self._obs_signature(agent, obs)
        if sig in self.validation_cache:
            return self.validation_cache[sig]

        if agent not in self.observation_specs:
            self.validation_errors.append(f"No specification for agent {agent}")
            self.validation_cache[sig] = False
            return False

        spec = self.observation_specs[agent]
        ok = True
        problems = []

        if not isinstance(obs, np.ndarray):
            problems.append(f"type={type(obs)} expected np.ndarray")
            ok = False
        else:
            if obs.ndim != 1:
                problems.append(f"ndim={obs.ndim} expected 1D")
                ok = False
            elif obs.shape[0] != spec["expected_dim"]:
                problems.append(f"dim={obs.shape[0]} expected {spec['expected_dim']}")
                ok = False
            if obs.dtype != np.float32:
                problems.append(f"dtype={obs.dtype} expected float32")
            if np.any(~np.isfinite(obs)):
                nbad = int(np.sum(~np.isfinite(obs)))
                problems.append(f"{nbad} non-finite")
                ok = False
            if np.any(obs < spec["low"]) or np.any(obs > spec["high"]):
                out = int(np.sum((obs < spec["low"]) | (obs > spec["high"])) )
                problems.append(f"{out} out-of-bounds")

        if problems and self.debug:
            msg = f"Validation issues for {agent}: " + "; ".join(problems)
            self.validation_errors.append(msg)
            self.logger.warning(msg)

        self.validation_cache[sig] = ok
        if len(self.validation_cache) > 2000:
            for k in list(self.validation_cache.keys())[:500]:
                del self.validation_cache[k]
        return ok

    def fix_observation(self, agent: str, obs: Any) -> np.ndarray:
        """Fix obs to expected shape/type/range."""
        if agent not in self.observation_specs:
            return np.zeros(self._estimate_agent_dimension(agent), dtype=np.float32)

        spec = self.observation_specs[agent]
        expected = spec["expected_dim"]

        try:
            if not isinstance(obs, np.ndarray):
                if obs is None:
                    obs = np.zeros(expected, np.float32)
                elif isinstance(obs, (list, tuple)):
                    obs = np.array(obs, np.float32)
                else:
                    obs = np.full(expected, float(obs), np.float32)
            else:
                obs = obs.astype(np.float32)

            if obs.ndim != 1:
                obs = obs.flatten()

            if obs.shape[0] != expected:
                if obs.shape[0] < expected:
                    if obs.size > 0 and np.any(np.isfinite(obs)):
                        fill = float(np.median(obs[np.isfinite(obs)]))
                    else:
                        fill = float((spec["low"][0] + spec["high"][0]) / 2.0)
                    fill = float(np.clip(fill, spec["low"][0], spec["high"][0]))
                    obs = np.concatenate([obs, np.full(expected - obs.shape[0], fill, np.float32)])
                else:
                    obs = obs[:expected]

            invalid = ~np.isfinite(obs)
            if np.any(invalid):
                rep = float((spec["low"][0] + spec["high"][0]) / 2.0)
                obs[invalid] = rep

            obs = np.clip(obs, spec["low"], spec["high"]).astype(np.float32)
            return obs
        except Exception:
            return ((spec["low"] + spec["high"]) / 2.0).astype(np.float32)

    def get_validation_stats(self) -> Dict[str, Any]:
        return {
            "agents_configured": len(self.observation_specs),
            "validation_cache_size": len(self.validation_cache),
            "recent_errors": len(self.validation_errors),
            "error_summary": list(self.validation_errors)[-10:],
            "specs_summary": {
                agent: {
                    "expected_dim": int(spec["expected_dim"]),
                    "bounds_range": [float(spec["low"].min()), float(spec["high"].max())],
                    "has_original_space": spec["original_space"] is not None,
                }
                for agent, spec in self.observation_specs.items()
            },
        }


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

        self._logger = logging.getLogger(__name__)
        self.logger = self._logger

        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=int(getattr(config, "max_memory_mb", 5000)))
        self.obs_validator = EnhancedObservationValidator(env, debug=debug)

        # Spaces
        self.observation_spaces: Dict[str, spaces.Box] = {}
        self.action_spaces: Dict[str, spaces.Space] = {}
        self._initialize_spaces()

        # Policies
        self.policies: List[Any] = []
        self._initialize_policies_enhanced(config, device)

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

        print(f"Enhanced MultiESGAgent initialized with {len(self.policies)} agents")
        if self.debug:
            self._print_initialization_summary()

    def set_total_training_budget(self, total_steps: int):
        try:
            self._global_target_steps = int(total_steps)
        except Exception:
            self._global_target_steps = None

    def _initialize_spaces(self):
        for agent in self.possible_agents:
            try:
                self.observation_spaces[agent] = self.env.observation_space(agent)
                self.action_spaces[agent] = self.env.action_space(agent)
                if self.debug:
                    obs_space = self.observation_spaces[agent]
                    act_space = self.action_spaces[agent]
                    print(f"{agent} | Obs: {obs_space.shape} | Act: {getattr(act_space,'shape',act_space)}")
            except Exception as e:
                self.logger.error(f"Failed to get spaces for {agent}: {e}")
                obs_dim = self.obs_validator._estimate_agent_dimension(agent)
                self.observation_spaces[agent] = spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)
                self.action_spaces[agent] = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                print(f"Created fallback spaces for {agent}")

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
                if policy is not None:
                    self.memory_tracker.register_policy(policy)
                    if hasattr(policy, "replay_buffer") and policy.replay_buffer is not None:
                        self.memory_tracker.register_buffer(policy.replay_buffer)
                    if hasattr(policy, "rollout_buffer") and policy.rollout_buffer is not None:
                        self.memory_tracker.register_buffer(policy.rollout_buffer)
                    self.policies.append(policy)
                    if self.debug:
                        print(f"Policy created for {agent} ({policy.mode})")
            except Exception as e:
                self.logger.error(f"Failed to initialize policy for {agent}: {e}")
                self._create_fallback_policy(agent, device)

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

            policy_kwargs = {
                "net_arch": getattr(config, "net_arch", [64, 32]),
                "activation_fn": activation_fn,
                "normalize_images": False,
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {"eps": 1e-8, "weight_decay": 0.0},
            }

            algo_kwargs = {
                "policy": "MlpPolicy",
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
                        "ent_coef": getattr(config, "ent_coef", 0.01),
                        "n_steps": int(min(self.n_steps, 512)),
                        "batch_size": 128,  # FIXED: Standardized batch size for all agents
                        "gae_lambda": float(getattr(config, "gae_lambda", 0.95)),
                        "clip_range": float(getattr(config, "clip_range", 0.2)),
                        "normalize_advantage": True,
                        "vf_coef": float(getattr(config, "vf_coef", 0.5)),
                        "max_grad_norm": float(getattr(config, "max_grad_norm", 0.5)),
                        "gamma": float(getattr(config, "gamma", 0.99)),
                        "n_epochs": int(min(getattr(config, "n_epochs", 10), 5)),
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
            return policy
        except Exception as e:
            self.logger.error(f"Single policy creation failed for {agent}: {e}")
            return None

    def _create_fallback_policy(self, agent, device):
        try:
            obs_space = self.observation_spaces[agent]
            act_space = self.action_spaces[agent]
            dummy_env = DummyVecEnv([partial(DummyGymEnv, obs_space, act_space)])
            fallback = PPO(
                "MlpPolicy", dummy_env, verbose=0, device=device, n_steps=128, batch_size=128, learning_rate=3e-4  # FIXED: Standardized parameters
            )
            fallback.mode = "PPO"
            fallback.agent_name = agent
            fallback.action_space = act_space
            self.memory_tracker.register_policy(fallback)
            if hasattr(fallback, "rollout_buffer"):
                self.memory_tracker.register_buffer(fallback.rollout_buffer)
            self.policies.append(fallback)
            print(f"Created fallback PPO policy for {agent}")
        except Exception as e:
            self.logger.error(f"Fallback policy creation failed for {agent}: {e}")

    # ----------------------------- Action guards -----------------------------
    def _ensure_action_shape(self, action, action_space):
        try:
            if isinstance(action_space, Discrete):
                if isinstance(action, (list, tuple, np.ndarray)) and np.size(action) > 0:
                    a = int(np.atleast_1d(action)[0])
                elif action is None:
                    a = 0
                else:
                    a = int(action)
                return int(np.clip(a, 0, action_space.n - 1))

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
            low = getattr(action_space, "low", None)
            high = getattr(action_space, "high", None)
            if low is not None and high is not None:
                a = np.clip(a, low, high)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            return a.astype(np.float32)
        except Exception:
            if isinstance(action_space, Box):
                return ((action_space.low + action_space.high) / 2.0).astype(np.float32)
            return 0

    def _process_action_enhanced(self, raw_action, action_space):
        """Clamp/sanitize actions while preserving distributional variability."""
        try:
            if isinstance(action_space, Discrete):
                if isinstance(raw_action, (list, tuple, np.ndarray)) and np.size(raw_action) > 0:
                    a = int(np.round(np.atleast_1d(raw_action)[0]))
                else:
                    a = int(np.round(float(raw_action)))
                return int(np.clip(a, 0, action_space.n - 1))
            a = np.asarray(raw_action, np.float32).reshape(-1)
            if np.any(~np.isfinite(a)):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            low = getattr(action_space, "low", None)
            high = getattr(action_space, "high", None)
            if low is not None and high is not None:
                a = np.clip(a, low, high)
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

            # Apply bounds if available
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                low = getattr(action_space, "low", None)
                high = getattr(action_space, "high", None)
                if low is not None and high is not None:
                    a = np.clip(a, low, high)

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
                        # Use midpoint of action space
                        low = np.asarray(action_space.low).flatten()
                        high = np.asarray(action_space.high).flatten()
                        midpoint = (low + high) / 2.0
                        return midpoint[:target_size].astype(np.float32)
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
        if hasattr(policy, "rollout_buffer") and getattr(policy.rollout_buffer, "full", False):
            try:
                policy.train()
                policy.rollout_buffer.reset()
                gc.collect()
            except Exception as e:
                self.logger.warning(f"PPO training failed for {policy.agent_name}: {e}")

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
            self._episode_starts = [True] * self.num_agents
            if not self.obs_validator.validate_observation_dict(self._last_obs):
                if self.debug:
                    print("Fixing initial observations")
                self._last_obs = self._fix_observation_dict_enhanced(self._last_obs)
                self._training_metrics["observation_fixes"] += 1
        except Exception as e:
            self.logger.error(f"Environment initialization failed: {e}")
            self._last_obs = self._create_emergency_observations_enhanced()
            self._episode_starts = [True] * self.num_agents

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

                actions_dict, agent_data = self._collect_actions_enhanced()
                next_obs, rewards, dones, truncs, infos = self._execute_environment_step_enhanced(actions_dict)
                self._add_experiences_enhanced(agent_data, rewards, dones, truncs, next_obs)
                self._update_state_enhanced(next_obs, dones, truncs)

                steps_collected += 1

                if self._check_buffers_full():
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

    def _collect_actions_enhanced(self):
        actions_dict: Dict[str, Any] = {}
        agent_data: Dict[int, Dict[str, Any]] = {}

        for polid, policy in enumerate(self.policies):
            agent_name = self.possible_agents[polid]
            try:
                obs = self._last_obs[agent_name].reshape(1, -1)
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
                if getattr(policy, "mode", None) == "PPO":
                    self._add_ppo_experience(policy, data, reward, polid)
                else:
                    self._add_offpolicy_experience(policy, data, reward, done, truncs, next_obs, agent_name)
                if hasattr(policy, "num_timesteps"):
                    policy.num_timesteps += 1
            except Exception as e:
                self.logger.warning(f"Experience addition error for {agent_name}: {e}")
    
    def _add_ppo_experience(self, policy, data, reward, polid):
        """
        Add one transition to SB3's RolloutBuffer.

        SB3 expects:
          - obs:            np.ndarray (CPU, float32)   shape: (1, obs_dim)
          - action:         np.ndarray (CPU, float32)   shape: (1, act_dim)
          - reward:         np.ndarray (CPU, float32)   shape: (1,)
          - episode_starts: np.ndarray (CPU, bool)      shape: (1,)
          - value:          torch.Tensor on policy.device, shape: (1, 1)
          - log_prob:       torch.Tensor on policy.device, shape: (1, 1)
        """
        # RolloutBuffer may not be initialized yet (e.g., during warmup)
        if not hasattr(policy, "rollout_buffer"):
            return

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

            # --- obs -> numpy (CPU, float32, shape (1, -1)) ---
            obs = data["obs"]
            if "torch" in str(type(obs)):
                obs = obs.detach().cpu().numpy()
            elif not isinstance(obs, np.ndarray):
                obs = np.asarray(obs)
            obs_np = obs.astype(np.float32).reshape(1, -1)

            # --- action -> numpy (CPU, float32, shape (1, -1)) ---
            action_space = getattr(policy, "action_space", None) or self.action_spaces.get(
                getattr(policy, "agent_name", ""), None
            )
            action_fixed = self._ensure_action_shape(data["action"], action_space)
            if "torch" in str(type(action_fixed)):
                action_fixed = action_fixed.detach().cpu().numpy()
            action_np = np.asarray(action_fixed, dtype=np.float32).reshape(1, -1)

            # --- reward/start flags -> numpy (CPU) ---
            reward_np = np.array([reward], dtype=np.float32)
            starts_np = np.array([self._episode_starts[polid]], dtype=bool)

            # --- finally add to SB3 buffer ---
            policy.rollout_buffer.add(obs_np, action_np, reward_np, starts_np, value_t, log_prob_t)

        except Exception as e:
            # Keep message identical to your logs so it's easy to grep
            self.logger.warning(f"PPO buffer add error: {e}")

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
            reward_b = np.array([reward], np.float32)
            done_b = np.array([done_flag], np.float32)
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

        self._last_obs = next_obs
        for polid, agent in enumerate(self.possible_agents):
            self._episode_starts[polid] = bool(dones.get(agent, False) or truncs.get(agent, False))

        if any(dones.values()) or any(truncs.values()):
            self._initialize_environment_enhanced()

    def _check_buffers_full(self):
        return any(
            getattr(policy, "mode", None) == "PPO"
            and hasattr(policy, "rollout_buffer")
            and getattr(policy.rollout_buffer, "full", False)
            for policy in self.policies
        )

    def _finalize_rollouts_enhanced(self):
        for polid, policy in enumerate(self.policies):
            try:
                if getattr(policy, "mode", None) == "PPO" and hasattr(policy, "rollout_buffer"):
                    agent_name = self.possible_agents[polid]
                    if agent_name in self._last_obs:
                        final_obs = self._last_obs[agent_name].reshape(1, -1)
                        with torch.no_grad():
                            obs_tensor = obs_as_tensor(final_obs, policy.device)
                            final_value = policy.policy.predict_values(obs_tensor)
                        # Use true episode-start (terminal) flag for correct GAE
                        dones_b = np.array([self._episode_starts[polid]], dtype=bool)
                        policy.rollout_buffer.compute_returns_and_advantage(final_value, dones_b)
            except Exception as e:
                self.logger.warning(f"Rollout finalization error for policy {polid}: {e}")

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
                dummy_env = DummyVecEnv([partial(DummyGymEnv, obs_space, act_space)])

                before = self.memory_tracker.get_memory_usage()
                loaded = algo_cls.load(path, device=self.device, env=dummy_env)
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

                after = self.memory_tracker.get_memory_usage()
                loaded_count += 1
                if self.debug:
                    print(f"Loaded {agent_name} policy (Δmem {after - before:.1f}MB)")
                if after > self.memory_tracker.max_memory_mb * 0.8:
                    self.memory_tracker.cleanup("light")
            except Exception as e:
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
        self.logger = logging.getLogger(__name__)

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

# Keep this alias for backward compatibility with main.py
HyperparameterOptimizer = OptunaHyperparameterOptimizer