import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import wandb  # ✅ global wandb

class RenewableInvestmentEnv(gym.Env):
    def __init__(self, data, initial_budget=1e7, wind_capex=1000, solar_capex=800,
                 hydro_capex=1200, battery_capex=400, timestep_minutes=10):
        super(RenewableInvestmentEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.n_steps = len(data)
        self.initial_budget = initial_budget
        self.wind_capex = wind_capex
        self.solar_capex = solar_capex
        self.hydro_capex = hydro_capex
        self.battery_capex = battery_capex
        self.opex_fraction = 0.02
        self.timestep_minutes = timestep_minutes
        self.timesteps_per_year = int(525600 / timestep_minutes)
        self.discount_rate = 0.07

        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)

        self.seed()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.budget = self.initial_budget
        self.wind_capacity = 0.0
        self.solar_capacity = 0.0
        self.hydro_capacity = 0.0
        self.battery_capacity = 0.0
        self.battery_energy = 0.0
        self.cumulative_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.t]
        obs = np.array([
            row['wind'] / 1000,
            row['solar'] / 1000,
            row['hydro'] / 1000,
            row['price'] / 500,
            row['load'] / 10000,
            self.wind_capacity / 1000,
            self.solar_capacity / 1000,
            self.battery_energy / 1000,
        ], dtype=np.float32)
        obs = np.clip(obs, 0, 1.5)
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def step(self, action):
        action = (action + 1.0) / 2.0
        wind_invest, solar_invest, hydro_invest, battery_invest, battery_use = np.clip(action, 0, 1)

        invest_total = wind_invest + solar_invest + hydro_invest + battery_invest
        if invest_total > 0:
            wind_share = wind_invest / invest_total
            solar_share = solar_invest / invest_total
            hydro_share = hydro_invest / invest_total
            battery_share = battery_invest / invest_total

            allocated = self.budget * 0.1

            self.wind_capacity += (allocated * wind_share) / self.wind_capex
            self.solar_capacity += (allocated * solar_share) / self.solar_capex
            self.hydro_capacity += (allocated * hydro_share) / self.hydro_capex
            self.battery_capacity += (allocated * battery_share) / self.battery_capex

            self.budget -= allocated

        row = self.data.iloc[self.t]
        wind_power = self.wind_capacity * row['wind']
        solar_power = self.solar_capacity * row['solar']
        hydro_power = self.hydro_capacity * row['hydro']
        total_gen = wind_power + solar_power + hydro_power
        total_gen = min(total_gen, 1e5)

        price = row['price']

        if self.battery_capacity > 0:
            if battery_use > 0.5:
                discharge_amount = min((battery_use - 0.5) * 2 * self.battery_capacity, self.battery_energy)
                total_gen += discharge_amount
                self.battery_energy -= discharge_amount
            else:
                charge_amount = min((0.5 - battery_use) * 2 * self.battery_capacity, total_gen)
                self.battery_energy += charge_amount
                total_gen -= charge_amount

            self.battery_energy = np.clip(self.battery_energy, 0, self.battery_capacity)

        if not np.isfinite(self.battery_energy):
            self.battery_energy = 0.0

        revenue = total_gen * price * (self.timestep_minutes / 60)
        opex = (self.wind_capacity + self.solar_capacity + self.hydro_capacity) * self.opex_fraction
        net_profit = revenue - opex
        risk_penalty = row.get('risk', 0.0) if 'risk' in row else 0.0
        reward = (net_profit - 0.5 * risk_penalty) / 100000.0

        self.cumulative_reward += reward
        done = self.t >= self.n_steps - 1

        # ✅ Global logging to the active WandB run
        wandb.log({
            "step": self.t,
            "reward": reward,
            "net_profit": net_profit,
            "revenue": revenue,
            "risk_penalty": risk_penalty,
            "wind_capacity": self.wind_capacity,
            "solar_capacity": self.solar_capacity,
            "hydro_capacity": self.hydro_capacity,
            "battery_capacity": self.battery_capacity,
            "battery_energy": self.battery_energy,
            "cumulative_reward": self.cumulative_reward
        })

        self.t += 1

        info = {
            "net_profit": net_profit,
            "revenue": revenue,
            "risk_penalty": risk_penalty
        }

        return self._get_obs(), reward, done, False, info

    def render(self):
        print(f"[Step {self.t}] Budget={self.budget:.2f}, Wind={self.wind_capacity:.2f}MW, Solar={self.solar_capacity:.2f}MW, Hydro={self.hydro_capacity:.2f}MW, Battery={self.battery_capacity:.2f}MWh")

    def seed(self, seed=None):
        np.random.seed(seed)
