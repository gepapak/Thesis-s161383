import wandb
import argparse
import os
from datetime import datetime

from utils import load_energy_data
from RenewableMultiAgentEnv import RenewableMultiAgentEnv
from multiesgagent_metacontroller import MultiESGAgent
from ForecastFeatureGenerator import ForecastFeatureGenerator
from forecast_multi_wrapper_env import ForecastMultiWrapperEnv

# Optional: Set WandB API key here or use wandb login
os.environ["WANDB_API_KEY"] = "9a27acac954ecccc53a820d856e7a6a088487f04"  # Replace with your actual key

class Config:
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
        ]
        self.company_initial_memory = 0
        self.investor_initial_memory = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sample.csv")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="energy-thesis")
    parser.add_argument("--investment_freq", type=int, default=144, help="Investor action frequency in steps")
    parser.add_argument("--debug", action="store_true", help="Enable observation shape printouts")

    args = parser.parse_args()

    run_name = f"multi_agent_energy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "timesteps": args.timesteps,
            "data_path": args.data_path,
            "device": args.device,
            "investment_freq": args.investment_freq,
            "agents": ["investor_0", "battery_operator_0", "risk_controller_0"]
        }
    )

    # ✅ Load and wrap environment
    data = load_energy_data(args.data_path)
    base_env = RenewableMultiAgentEnv(data, investment_freq=args.investment_freq)  # ← dynamic
    forecast_gen = ForecastFeatureGenerator("forecast models", look_back=6, verbose=True)
    env = ForecastMultiWrapperEnv(base_env, forecast_models=forecast_gen, look_back=6)

    # ✅ Launch multi-agent controller
    config = Config()
    agent = MultiESGAgent(config, env=env, device=args.device, training=True, debug=args.debug)

    # ✅ Begin training
    agent.learn(total_timesteps=args.timesteps)

    wandb.finish()
