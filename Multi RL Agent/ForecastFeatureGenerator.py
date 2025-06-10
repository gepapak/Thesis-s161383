import os
import numpy as np
import tensorflow as tf

class ForecastFeatureGenerator:
    def __init__(self, model_dir, look_back=6, forecast_horizon=1, verbose=True):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.history = {}
        self.verbose = verbose

        for fname in os.listdir(model_dir):
            if fname.endswith(".h5"):
                target = fname.replace(".h5", "")
                path = os.path.join(model_dir, fname)
                if self.verbose:
                    print(f"🔍 Loading model for '{target}' from '{fname}'")
                self.models[target] = tf.keras.models.load_model(path, compile=False)
                self.history[target] = []  # Initialize empty series history

    def update(self, row):
        """Update internal history buffer with new timestep row (a dict or pd.Series)."""
        for target in self.models.keys():
            if target in row:
                self.history[target].append(float(row[target]))

    def predict(self, timestep=None):
        """Predict forecast values from internal history."""
        results = {}

        for target, model in self.models.items():
            series = self.history.get(target, [])
            if len(series) < self.look_back:
                if self.verbose:
                    print(f"⚠️ Not enough history for '{target}' (have {len(series)}, need {self.look_back})")
                # Pad with zeros if not enough data
                padded_series = [0.0] * (self.look_back - len(series)) + series
            else:
                padded_series = series[-self.look_back:]

            try:
                window = np.array(padded_series).reshape(1, self.look_back, 1)
                forecast = model.predict(window, verbose=0)[0]
                forecast_value = forecast[self.forecast_horizon - 1] if len(forecast) >= self.forecast_horizon else forecast[-1]
                results[f"{target}_forecast"] = float(forecast_value)
            except Exception as e:
                print(f"❌ Forecasting failed for '{target}': {e}")
                results[f"{target}_forecast"] = 0.0

        return results

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def items(self):
        return self.models.items()

    def __repr__(self):
        return f"ForecastFeatureGenerator(targets={list(self.models.keys())}, look_back={self.look_back})"
