import gymnasium as gym
import numpy as np
import tensorflow as tf

class ForecastWrapperEnv(gym.Wrapper):
    def __init__(self, env, forecast_models, look_back=6, forecast_interval=6):
        super().__init__(env)
        self.look_back = look_back
        self.forecast_models = forecast_models  # dict of {'wind': model, 'price': model, ...}
        self.forecast_interval = forecast_interval  # forecast every N steps only
        self.forecast_cache = {}

        # Expand observation space
        extra_features = len(self.forecast_models.models)
        low = np.concatenate([self.env.observation_space.low, np.zeros(extra_features)])
        high = np.concatenate([self.env.observation_space.high, np.ones(extra_features)*np.inf])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        self.forecast_cache = {}
        obs, info = self.env.reset(**kwargs)
        self._update_forecasts()
        return self._concat_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._update_forecasts()
        return self._concat_obs(obs), reward, done, truncated, info

    def _concat_obs(self, obs):
        forecast_values = [self.forecast_cache.get(var, 0.0) for var in self.forecast_models]
        return np.concatenate([obs, np.array(forecast_values, dtype=np.float32)])

    def _update_forecasts(self):
        t = self.env.t
        if t % self.forecast_interval != 0:
            return  # Skip unnecessary steps

        for var, model in self.forecast_models.models.items():
            if t < self.look_back:
                self.forecast_cache[var] = 0.0
                continue

            try:
                window = self.env.data[var].iloc[t - self.look_back:t].values
                norm_input = window.reshape(1, self.look_back, 1)
                forecast = float(model.predict(norm_input)[0])
                self.forecast_cache[var] = forecast
            except Exception as e:
                print(f"[ForecastWrapper] Forecasting failed for {var} at t={t}: {e}")
                self.forecast_cache[var] = 0.0
