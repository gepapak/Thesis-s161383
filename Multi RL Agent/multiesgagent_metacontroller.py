from copy import deepcopy
from hardcode_agents import HardCodePolicy
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import configure_logger, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import time
import concurrent.futures
from collections import deque
from functools import partial

LEARNING_MODES = ["PPO"]

def multithreaded_processing(function, policies, multithreading=True):
    if multithreading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(function, polid, policy) for polid, policy in enumerate(policies)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"❌ Policy {future} generated an exception: {exc}")
                    raise exc
    else:
        for polid, policy in enumerate(policies):
            function(polid, policy)

class DummyGymEnv(Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        done = True
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def render(self):
        pass

class MultiESGAgent:
    def __init__(self, config, env, device, training=True, with_expert=None, debug=True):
        self.env = env
        self.possible_agents = env.possible_agents
        self.num_agents = len(self.possible_agents)
        self.num_envs = 1
        self.n_steps = config.update_every
        self.episode = 0
        self.tensorboard_log = None
        self.verbose = config.verbose
        self._logger = None
        self.multithreading = getattr(config, "multithreading", True)
        self.debug = debug

        # ✅ Reset only once to avoid desync
        obs, _ = env.reset()

        # ✅ Get correct obs/action spaces post-wrapper
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=np.inf, shape=(obs[agent].shape[0],), dtype=np.float32)
            for agent in obs
        }
        self.action_spaces = {
            agent: env.action_space(agent) for agent in self.possible_agents
        }

        if self.debug:
            print("🔍 Final Obs Shapes (post-wrapper):")
            for agent, space in self.observation_spaces.items():
                print(f"  - {agent}: shape = {space.shape}")

        class CustomDummyEnv(Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}
            def step(self, action):
                return self.observation_space.sample(), 0.0, True, False, {}
            def render(self): pass

        self.policies = []
        for i, agent in enumerate(self.possible_agents):
            obs_space = self.observation_spaces[agent]
            act_space = self.action_spaces[agent]

            print(f"[INIT] Creating PPO for {agent} with input dim {obs_space.shape[0]}")
            dummy_env = DummyVecEnv([
                partial(CustomDummyEnv, obs_space, act_space)
            ])
            dummy_env.observation_space = obs_space
            dummy_env.action_space = act_space

            policy = PPO(
                policy="MlpPolicy",
                env=dummy_env,
                n_steps=self.n_steps,
                learning_rate=config.lr,
                ent_coef=config.ent_coef,
                policy_kwargs={
                    'net_arch': [256, 128],
                    'activation_fn': nn.Tanh
                },
                verbose=config.verbose,
                device=device,
                seed=config.seed
            )

            policy.mode = config.agent_policies[i]["mode"]
            if policy.mode == "Hardcoded":
                policy.policy = HardCodePolicy(policy.policy, **config.agent_policies[i])
            self.policies.append(policy)

        self.max_steps = env.max_steps
        self.total_steps = 0
        self.config = config
        self.device = device
        self.training = training
        self.with_expert = with_expert
        self.loss = None
        self.buffer = []
        self.model_others = False

    def learn(self, total_timesteps: int, callbacks: Optional[List] = None, log_interval: int = 1, tb_log_name: str = "MultiAgentPPO", reset_num_timesteps: bool = True):
        num_timesteps = 0
        if not callbacks:
            callbacks = [None] * self.num_agents
        self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                policy.start_time = time.time()
                policy.ep_info_buffer = deque(maxlen=100)
                policy.ep_success_buffer = deque(maxlen=100)
                if policy.action_noise:
                    policy.action_noise.reset()
                policy.num_timesteps = 0
                policy._episode_num = 0
                policy._total_timesteps = total_timesteps
                policy._logger = configure_logger(policy.verbose, None, f"policy_{polid}", reset_num_timesteps)
                callbacks[polid] = policy._init_callback(callbacks[polid])
                callbacks[polid].on_training_start(locals(), globals())

        def train_loop(polid, policy):
            if policy.mode in LEARNING_MODES:
                policy._update_current_progress_remaining(policy.num_timesteps, total_timesteps)
                policy.train()

        while num_timesteps < total_timesteps:
            self.collect_rollouts(callbacks)
            num_timesteps += self.num_envs * self.n_steps
            multithreaded_processing(train_loop, self.policies, multithreading=self.multithreading)

        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                callbacks[polid].on_training_end()

    def collect_rollouts(self, callbacks):
        last_obs, info = self.env.reset()
        steps = 0

        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        all_last_episode_starts = [np.ones((self.num_envs,), dtype=bool)] * self.num_agents

        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                policy.policy.set_training_mode(False)
                policy.rollout_buffer.reset()
                callbacks[polid].on_rollout_start()

        while steps < self.n_steps:
            if steps == 0 or True in all_dones[0]:
                last_obs, info = self.env.reset()
            for polid in range(self.num_agents):
                agent = self.possible_agents[polid]
                all_last_obs[polid] = np.array([last_obs[agent]])

            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents

            def evaluate_action(polid, policy):
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    if self.debug:
                        print(f"Agent {self.possible_agents[polid]} | obs_tensor.shape: {obs_tensor.shape}")
                    all_actions[polid], all_values[polid], all_log_probs[polid] = policy.policy.forward(obs_tensor)
                    clipped_actions = all_actions[polid].cpu().numpy()
                    action_space = self.action_spaces[self.possible_agents[polid]]
                    if isinstance(action_space, Box) and policy.mode != "Hardcoded":
                        clipped_actions = (clipped_actions + 1) / 2 * action_space.high
                        clipped_actions = np.clip(clipped_actions, action_space.low, action_space.high)
                    elif isinstance(action_space, Discrete):
                        clipped_actions = np.array([action.item() for action in clipped_actions])
                    all_clipped_actions[polid] = clipped_actions

            multithreaded_processing(evaluate_action, self.policies, multithreading=self.multithreading)

            action_dict = {self.possible_agents[polid]: all_clipped_actions[polid][0] for polid in range(self.num_agents)}
            obs, rewards, dones, truncations, infos = self.env.step(action_dict)
            for polid in range(self.num_agents):
                agent = self.possible_agents[polid]
                all_obs[polid] = np.array([obs[agent]])
                all_rewards[polid] = np.array([rewards[agent]])
                all_dones[polid] = np.array([dones[agent]])
                all_infos[polid] = [infos[agent]]

            for polid, policy in enumerate(self.policies):
                if policy.mode in LEARNING_MODES:
                    policy.rollout_buffer.add(
                        all_last_obs[polid],
                        all_actions[polid].cpu().numpy(),
                        all_rewards[polid],
                        all_last_episode_starts[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    )
                    policy._update_info_buffer(all_infos[polid])
                    policy.num_timesteps += self.num_envs
                    callbacks[polid].on_step()

            all_last_obs = all_obs
            all_last_episode_starts = all_dones
            steps += 1

        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    _, value, _ = policy.policy.forward(obs_tensor)
                    policy.rollout_buffer.compute_returns_and_advantage(value, all_dones[polid])
                callbacks[polid].on_rollout_end()
                policy._last_episode_starts = all_last_episode_starts[polid]
        return obs
