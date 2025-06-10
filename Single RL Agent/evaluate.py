import wandb
from stable_baselines3 import PPO
from data_loader import load_energy_data
from RenewableInvestmentEnv import RenewableInvestmentEnv
from forecast_wrapper_env import ForecastWrapperEnv
from ForecastFeatureGenerator import ForecastFeatureGenerator

# ✅ Initialize WandB for evaluation logging
wandb.init(
    project="energy-thesis",
    name="ppo_evaluation_run",
    mode="online",  # Use "disabled" to turn off logging
    config={"mode": "evaluation"}
)

# ✅ Load environment and forecast models
data = load_energy_data("sample.csv")
base_env = RenewableInvestmentEnv(data)
forecast_gen = ForecastFeatureGenerator("forecast models")
env = ForecastWrapperEnv(base_env, forecast_gen)

# ✅ Load the trained agent
model = PPO.load("ppo_energy_agent")

obs, _ = env.reset()
done = False
cumulative_reward = 0.0

# ✅ Evaluation loop
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    cumulative_reward += reward

print(f"\n✅ Evaluation complete. Cumulative Reward: {cumulative_reward:.2f}")
wandb.log({"evaluation_cumulative_reward": cumulative_reward})
wandb.finish()
