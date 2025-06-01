from RenewableInvestmentEnv import RenewableInvestmentEnv
from ForecastWrapperEnv import ForecastWrapperEnv
import tensorflow as tf

# Load base environment
env = RenewableInvestmentEnv(data)

# Load trained models
forecast_models = {
    "hydro": tf.keras.models.load_model("forecast models/ceemdan_lstm_hydro.h5"),
    "wind": tf.keras.models.load_model("forecast models/ceemdan_lstm_wind.h5"),
    # Add more as needed
}

# Wrap with forecast-aware wrapper
env = ForecastWrapperEnv(env, forecast_models=forecast_models, look_back=6)