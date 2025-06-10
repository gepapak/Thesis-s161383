import os
import numpy as np
import tensorflow as tf

class ForecastFeatureGenerator:
    def __init__(self, model_dir, look_back=6, forecast_horizon=1):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.models = {}

        for fname in os.listdir(model_dir):
            if fname.endswith(".h5"):
                target = fname.replace(".h5", "").split("_")[-1]
                path = os.path.join(model_dir, fname)
                print(f"🔍 Loading model for {target} from {fname}")
                self.models[target] = tf.keras.models.load_model(path, compile=False)

    def predict(self, series_dict):
        results = {}
        for target, model in self.models.items():
            if target not in series_dict or len(series_dict[target]) < self.look_back:
                results[f"{target}_forecast"] = 0.0
                continue

            window = np.array(series_dict[target][-self.look_back:]).reshape(1, self.look_back, 1)
            forecast = model.predict(window, verbose=0)[0]
            forecast_value = forecast[self.forecast_horizon - 1] if len(forecast) >= self.forecast_horizon else forecast[-1]
            results[f"{target}_forecast"] = float(forecast_value)

        return results

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def items(self):
        return self.models.items()

    def __repr__(self):
        return f"ForecastFeatureGenerator(targets={list(self.models.keys())}, look_back={self.look_back})"
