import os
import pandas as pd
import tensorflow as tf

from lstm import lstm_model
from ceemdan_lstm import ceemdan_lstm_model
from emd_lstm import emd_lstm_model
from eemd_lstm import eemd_lstm_model
from ann import ann_model
from karijadi import karijadi_model

# === CONFIG ===
file_path = "full_training_data.csv"
targets = ["wind", "hydro", "solar", "load", "price"]
i = [1, 2]
look_back = 6
data_partition = 0.7
output_dir = "forecast models"

# === Ensure output folder exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['timestamp'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# === Wrapper to extract Keras model ===
def wrapper(model_func, *args, **kwargs):
    result = model_func(*args, **kwargs)
    if isinstance(result, tuple) and isinstance(result[-1], tf.keras.Model):
        return result[-1]
    else:
        raise ValueError("Returned object is not a Keras model.")

# === Model functions ===
model_funcs = {
    "LSTM": lstm_model,
    "CEEMDAN_LSTM": ceemdan_lstm_model,
    "EMD_LSTM": emd_lstm_model,
    "EEMD_LSTM": eemd_lstm_model,
    "ANN": ann_model,
    "KARIJADI": karijadi_model
}

# === Export Loop ===
for target in targets:
    for name, model_func in model_funcs.items():
        print(f"\n📦 Exporting {name} model for {target}...")
        try:
            new_data = df[["Month", "Year", "Date", target]].copy()
            cap = new_data[target].max()
            model = wrapper(model_func, new_data, i, look_back, data_partition, cap, target)
            model_path = os.path.join(output_dir, f"{target}_{name}.h5")
            model.save(model_path)
            print(f"✅ Saved to {model_path}")
        except Exception as e:
            print(f"❌ Failed to export {name} for {target}: {e}")
