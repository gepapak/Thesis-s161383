from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd

class RenewableMultiAgentEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "renewable_marl_v0"}

    def __init__(self, data: pd.DataFrame, investment_freq=144):  # 10min × 144 = 1 day
        self.data = data.reset_index(drop=True)
        self.max_steps = len(data)
        self.t = 0
        self.investment_freq = investment_freq
        self.last_investor_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.investor_reward_buffer = []  # ✅ Buffer for smoothing

        self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0"]
        self.agents = self.possible_agents[:]

        self.init_budget = 1e7
        self.wind_capex = 1000
        self.solar_capex = 800
        self.hydro_capex = 1200
        self.battery_capex = 400

        self.wind_capacity = 0.0
        self.solar_capacity = 0.0
        self.hydro_capacity = 0.0
        self.battery_capacity = 0.0
        self.battery_energy = 0.0
        self.budget = self.init_budget
        self.risk_multiplier = 1.0

        self.observation_spaces = {
            "investor_0": spaces.Box(low=0, high=np.inf, shape=(14,), dtype=np.float32),
            "battery_operator_0": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),
            "risk_controller_0": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        }

        self.action_spaces = {
            "investor_0": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            "battery_operator_0": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "risk_controller_0": spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        self.t = 0
        self.agents = self.possible_agents[:]
        self.wind_capacity = 0.0
        self.solar_capacity = 0.0
        self.hydro_capacity = 0.0
        self.battery_capacity = 0.0
        self.battery_energy = 0.0
        self.budget = self.init_budget
        self.risk_multiplier = 1.0
        self.last_investor_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.investor_reward_buffer = []  # ✅ Reset smoothing buffer
        return self._get_obs(), {}

    def step(self, actions):
        row = self.data.iloc[self.t]
        done = self.t >= self.max_steps - 1

        terminated = {agent: done for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # ✅ Always apply risk controller
        self.risk_multiplier = float(np.clip(actions["risk_controller_0"][0], 0.0, 2.0))

        # 🔁 Investor acts only every `investment_freq` steps
        if self.t % self.investment_freq == 0:
            self.last_investor_action = np.array(actions["investor_0"], dtype=np.float32)

        # Use cached investor action
        invest_action = np.clip((self.last_investor_action + 1.0) / 2.0, 0, 1)
        wind_invest, solar_invest, hydro_invest = invest_action
        total_invest = wind_invest + solar_invest + hydro_invest

        if total_invest > 0:
            allocated = self.budget * 0.1
            wind_share = wind_invest / total_invest
            solar_share = solar_invest / total_invest
            hydro_share = hydro_invest / total_invest

            self.wind_capacity += (allocated * wind_share) / self.wind_capex
            self.solar_capacity += (allocated * solar_share) / self.solar_capex
            self.hydro_capacity += (allocated * hydro_share) / self.hydro_capex
            self.budget -= allocated

        # ✅ Battery operator action every step
        battery_use = np.clip((actions["battery_operator_0"][0] + 1.0) / 2.0, 0, 1)
        if self.battery_capacity == 0:
            self.battery_capacity += (self.init_budget * 0.05) / self.battery_capex

        # Energy generation
        wind_power = self.wind_capacity * row["wind"]
        solar_power = self.solar_capacity * row["solar"]
        hydro_power = self.hydro_capacity * row["hydro"]
        total_gen = wind_power + solar_power + hydro_power

        # Battery operation
        if battery_use > 0.5:
            discharge = min((battery_use - 0.5) * 2 * self.battery_capacity, self.battery_energy)
            total_gen += discharge
            self.battery_energy -= discharge
        else:
            charge = min((0.5 - battery_use) * 2 * self.battery_capacity, total_gen)
            self.battery_energy += charge
            total_gen -= charge

        self.battery_energy = np.clip(self.battery_energy, 0, self.battery_capacity)

        # Financial performance
        price = row["price"]
        revenue = total_gen * price * (10 / 60)
        opex = (self.wind_capacity + self.solar_capacity + self.hydro_capacity) * 0.02
        risk_penalty = row.get("risk", 0.0) * self.risk_multiplier
        net_profit = revenue - opex

        investor_reward = (net_profit - 0.5 * risk_penalty) / 100000.0
        battery_reward = revenue / 100000.0
        risk_reward = -risk_penalty / 100000.0

        # ✅ Smooth investor reward across investment_freq window
        self.investor_reward_buffer.append(investor_reward)
        if len(self.investor_reward_buffer) > self.investment_freq:
            self.investor_reward_buffer.pop(0)
        smoothed_investor_reward = np.mean(self.investor_reward_buffer)

        self.t += 1
        return self._get_obs(), {
            "investor_0": smoothed_investor_reward,
            "battery_operator_0": battery_reward,
            "risk_controller_0": risk_reward
        }, terminated, truncated, infos

    def _get_obs(self):
        row = self.data.iloc[self.t]
        return {
            "investor_0": np.array([
                row['wind'], row['solar'], row['hydro'],
                row['price'], row['load'], self.budget / 1e7
            ], dtype=np.float32),
            "battery_operator_0": np.array([
                row['price'], self.battery_energy / 1000,
                self.battery_capacity / 1000, row['load']
            ], dtype=np.float32),
            "risk_controller_0": np.array([
                row['price'], row.get("risk", 0.0),
                self.budget / 1e7
            ], dtype=np.float32)
        }

    def render(self):
        print(f"[t={self.t}] Wind={self.wind_capacity:.2f}, Solar={self.solar_capacity:.2f}, "
              f"Battery_E={self.battery_energy:.2f}, Budget={self.budget:.2f}, "
              f"RiskMult={self.risk_multiplier:.2f}")
