import os
import wandb
import argparse
import numpy as np
from datetime import datetime

from utils import load_energy_data
from RenewableMultiAgentEnv import RenewableMultiAgentEnv
from ForecastFeatureGenerator import ForecastFeatureGenerator
from forecast_multi_wrapper_env import ForecastMultiWrapperEnv
from multiesgagent_metacontroller import MultiESGAgent

os.environ["WANDB_API_KEY"] = "9a27acac954ecccc53a820d856e7a6a088487f04"

class Config:
    def __init__(self):
        self.update_every = 128
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.verbose = 1
        self.seed = 42
        self.multithreading = True
        self.agent_policies = [
            {"mode": "PPO"},
            {"mode": "PPO"},
            {"mode": "PPO"},
        ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sample.csv")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="energy-thesis")
    parser.add_argument("--initial_freq", type=int, default=48)
    parser.add_argument("--final_freq", type=int, default=1008)
    parser.add_argument("--phases", type=int, default=3)
    parser.add_argument("--look_back", type=int, default=6)
    args = parser.parse_args()

    data = load_energy_data(args.data_path)
    forecast_gen = ForecastFeatureGenerator("forecast models", look_back=args.look_back, verbose=True)

    phase_timesteps = args.timesteps // args.phases
    run_name = f"curriculum_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name)

    for i in range(args.phases):
        phase_freq = int(np.interp(i, [0, args.phases - 1], [args.initial_freq, args.final_freq]))
        wandb.log({"current_investment_freq": phase_freq})

        base_env = RenewableMultiAgentEnv(data, investment_freq=phase_freq)
        env = ForecastMultiWrapperEnv(base_env, forecast_models=forecast_gen, look_back=args.look_back)

        config = Config()
        agent = MultiESGAgent(config, env=env, device=args.device, training=True, debug=True)
        agent.learn(total_timesteps=phase_timesteps)

    wandb.finish()
