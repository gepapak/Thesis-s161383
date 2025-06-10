# main.py — patched to support forecast models & retain WandB

import argparse
import os
import wandb
from datetime import datetime

from data_loader import load_energy_data
from RenewableInvestmentEnv import RenewableInvestmentEnv
from forecast_wrapper_env import ForecastWrapperEnv
from ForecastFeatureGenerator import ForecastFeatureGenerator

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback

# Optional: Set WandB API key here or use wandb login
os.environ["WANDB_API_KEY"] = "9a27acac954ecccc53a820d856e7a6a088487f04"  # Replace with your actual key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sample.csv")
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()

    run_name = f"ppo_run_energy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project="energy-thesis",
        name=run_name,
        config={
            "timesteps": args.timesteps,
            "data_path": args.data_path,
            "algorithm": "PPO"
        }
    )

    # === Load data and wrap with forecast-enhanced environment ===
    data = load_energy_data(args.data_path)
    base_env = RenewableInvestmentEnv(data)

    forecast_gen = ForecastFeatureGenerator(
        model_dir="forecast models",
        look_back=6
    )
    env = ForecastWrapperEnv(base_env, forecast_gen)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path="checkpoints/",
            verbose=1
        )
    )

    model.save("ppo_energy_agent")
    wandb.finish()

if __name__ == "__main__":
    main()
