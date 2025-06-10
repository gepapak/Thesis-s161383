from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import wandb

class ForecastMultiWrapperEnv(ParallelEnv):
    def __init__(self, base_env, forecast_models, look_back=6):
        self.env = base_env
        self.forecast_models = forecast_models
        self.look_back = look_back
        self.possible_agents = self.env.possible_agents
        self.max_steps = self.env.max_steps

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents  # Must track current agents
        self._update_history()
        forecast_obs = self._add_forecasts(obs)
        return forecast_obs, info

    def step(self, actions):
        obs, rewards, dones, truncations, infos = self.env.step(actions)
        self._update_history()
        forecast_obs = self._add_forecasts(obs)

        # ✅ WandB logging
        log_data = {"timestep": self.env.t}
        for agent in self.agents:
            agent_actions = actions[agent] if isinstance(actions[agent], (list, np.ndarray)) else [actions[agent]]
            for i, val in enumerate(agent_actions):
                log_data[f"action_{agent}_{i}"] = float(val)

        forecast_vals = self.forecast_models.predict()  # ✅ FIXED: Use internal model state
        for k, v in forecast_vals.items():
            log_data[f"forecast_{k}"] = float(v)
        wandb.log(log_data)

        return forecast_obs, rewards, dones, truncations, infos

    def _update_history(self):
        """Update forecast generator with the current timestep row."""
        if hasattr(self.env, "data") and self.env.t < len(self.env.data):
            row = self.env.data.iloc[self.env.t]
            self.forecast_models.update(row)

    def _add_forecasts(self, obs):
        forecasted = self.forecast_models.predict()  # ✅ FIXED: Use internal history
        new_obs = {}

        for agent in self.env.agents:
            base_obs = obs[agent]

            if agent == "investor_0":
                additions = np.array([
                    forecasted.get("wind_forecast", 0.0),
                    forecasted.get("solar_forecast", 0.0),
                    forecasted.get("hydro_forecast", 0.0),
                    forecasted.get("price_forecast", 0.0)
                ])
            elif agent == "battery_operator_0":
                additions = np.array([
                    forecasted.get("price_forecast", 0.0),
                    forecasted.get("load_forecast", 0.0)
                ])
            elif agent == "risk_controller_0":
                additions = np.array([
                    forecasted.get("risk_forecast", 0.0)
                ])
            else:
                additions = np.array([])

            new_obs[agent] = np.concatenate([base_obs, additions]).astype(np.float32)
            # print(f"[DEBUG] Agent: {agent}, Obs shape: {new_obs[agent].shape}")

        return new_obs

    def render(self):
        return self.env.render()

    def observation_space(self, agent):
        base_shape = self.env.observation_spaces[agent].shape[0]
        if agent == "investor_0":
            return spaces.Box(low=0, high=np.inf, shape=(base_shape + 4,), dtype=np.float32)
        elif agent == "battery_operator_0":
            return spaces.Box(low=0, high=np.inf, shape=(base_shape + 2,), dtype=np.float32)
        elif agent == "risk_controller_0":
            return spaces.Box(low=0, high=np.inf, shape=(base_shape + 1,), dtype=np.float32)
        else:
            return self.env.observation_spaces[agent]

    def action_space(self, agent):
        return self.env.action_spaces[agent]

    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent) for agent in self.possible_agents}
