import gym
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
import torch
import wandb
import yaml
from stable_baselines3.common.callbacks import BaseCallback

# ✅ Removed unused import from old ESG code
# from wrapper_env import WrapperInvestESGEnv

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    print(f"Episode reward: {info['episode']['r']}")
        return True

    def _on_training_end(self) -> None:
        avg_reward = np.mean(self.episode_rewards)
        print(f"Average reward over {len(self.episode_rewards)} episodes: {avg_reward}")

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def merge_configs(update, default):
    if isinstance(update,dict) and isinstance(default,dict):
        for k,v in default.items():
            if k not in update:
                update[k] = v
            else:
                update[k] = merge_configs(update[k],v)
    return update

def make_env(config):
    if 'MultiGrid' in config.domain:
        from envs import gym_multigrid
        from envs.gym_multigrid import multigrid_envs
        env = gym.make(config.domain)
    else:
        raise NotImplementedError("Only MultiGrid is supported in this version of utils.py")
    return env

def argmax_2d_index(arr):
    assert len(arr.shape) == 2
    best_2d_index = (arr==torch.max(arr)).nonzero()
    if best_2d_index.shape[0] > 1:
        best_2d_index = best_2d_index[random.randrange(best_2d_index.shape[0]),:]
    return best_2d_index.squeeze()

def process_state(state, observation_shape):
    if len(observation_shape) == 3:
        state = torch.tensor(state)
        state = state.transpose(0, 2).transpose(1, 2)
        state = state.float().unsqueeze(0)
    return state

def generate_parameters(mode, domain, debug=False, seed=None, wandb_project=None):
    os.environ["WANDB_MODE"] = "online"
    config_default = yaml.safe_load(open("config/default.yaml", "r"))
    config_domain = yaml.safe_load(open("config/domain/" + domain + ".yaml", "r"))
    config_mode = yaml.safe_load(open("config/mode/" + mode + ".yaml", "r"))
    if seed:
        config_default['seed'] = seed
    config_default['experiment_name'] = 'MultiGrid'
    config_with_domain = merge_configs(config_domain, config_default)
    config = dotdict(merge_configs(config_mode, config_with_domain))
    if debug:
        print('Debug selected, disabling wandb')
        wandb.init(project = wandb_project + '-' + domain, config=config, mode='disabled')
    else:
        wandb.init(project = wandb_project + '-' + domain, config=config)

    path_configs = {'model_name': config.mode + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'load_model_path': config.load_model_start_path + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'wandb_project': wandb_project + '-' + config.domain}
    wandb.config.update(path_configs)

    print("CONFIG")
    print(wandb.config)

    wandb.define_metric("episode/x_axis")
    wandb.define_metric("step/x_axis")
    wandb.define_metric("episode/*", step_metric="episode/x_axis")
    wandb.define_metric("step/*", step_metric="step/x_axis")

    if not os.path.exists("models/"):
        os.makedirs("models/")
    if not os.path.exists("traj/"):
        os.makedirs("traj/")

    wandb.run.name = config.model_name
    return wandb.config

# Other plotting, video, and helper functions remain unchanged
def load_energy_data(csv_path):
    """
    Load energy time series data from CSV.
    Expects columns: wind, solar, hydro, price, load, risk (optional).
    """
    df = pd.read_csv(csv_path)
    required_cols = ["wind", "solar", "hydro", "price", "load"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in data: {col}")
    return df