# Test/Validation Set - Model Selection and Hyperparameter Tuning (with CSV options)
import numpy as np
import pandas as pd
import os
import joblib
import json
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from create_dataset import create_dataset

# Import the multi-horizon dataset creation function from ann.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ann import _create_dataset_horizon

# --- CONFIG ---
# Read from environment variables (set by main_training.ipynb)
target = os.environ.get('VAL_TARGET', 'wind')
horizon_label = os.environ.get('VAL_HORIZON_LABEL', 'immediate')
EVAL_SPLIT = os.environ.get('VAL_SPLIT', 'validation')   # 'validation', 'test', or 'external'
USE_BEST_MODEL_IF_AVAILABLE = True

# CSV options (change these as you like)
CSV_OUT_DIR = "forecast_results"
CSV_FILENAME = f"{target}_{horizon_label}_{EVAL_SPLIT}_results.csv"
CSV_SEP = ','               # e.g., ';' for European CSVs
CSV_DECIMAL = '.'           # e.g., ',' for European decimal
CSV_FLOAT_FMT = '%.6f'
CSV_INDEX = False
CSV_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- Load metadata ---
# NEW FORMAT: {target}_{horizon_label}_metadata.json
metadata_path = os.path.join("metadata", f"{target}_{horizon_label}_metadata.json")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"❌ ANN metadata not found at: {metadata_path}")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Extract parameters
cap = metadata['cap']
look_back = metadata['look_back']
horizon_steps = metadata.get('horizon_steps', 1)  # CRITICAL: Get horizon_steps for multi-horizon
train_ratio = metadata['train_ratio']
val_ratio = metadata['val_ratio']
test_ratio = metadata['test_ratio']
model_path_final = metadata['model_path']
model_path_best = metadata.get('model_path_best', None)
scaler_x_path = metadata['scaler_x_path']
scaler_y_path = metadata['scaler_y_path']
original_data_path = metadata['original_data_path']
test_data_path = metadata.get('test_data_path', None)

# Prefer best checkpoint if available
model_path = None
if USE_BEST_MODEL_IF_AVAILABLE and model_path_best and os.path.exists(model_path_best):
    model_path = model_path_best
else:
    model_path = model_path_final

print(f"🔍 Evaluating ANN Model on: {EVAL_SPLIT.upper()}")
print("="*52)
print(f"📊 Target: {target}")
print(f"📊 Horizon: {horizon_label} ({horizon_steps} steps ahead)")
print(f"📊 Split ratios: {train_ratio:.1%} train / {val_ratio:.1%} validation / {test_ratio:.1%} test")
print(f"📁 Using model: {model_path}")

# --- Load data depending on split ---
if EVAL_SPLIT == 'validation':
    # Use original series and carve out the validation region chronologically
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"❌ Original data not found at: {original_data_path}")
    original_df = pd.read_csv(original_data_path)
    series = original_df[target].values.reshape(-1, 1)

    total_samples = len(series)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    print(f"📊 Total samples: {total_samples:,}")
    print(f"📊 Train samples: {train_size:,}")
    print(f"📊 Validation samples: {val_size:,}")

    # Extract validation slice and build windows
    val_data = series[train_size:train_size + val_size]
    X, y_true = _create_dataset_horizon(val_data, look_back, horizon_steps)

    # Timestamps aligned to the first predicted index in validation
    if 'timestamp' in original_df.columns:
        try:
            all_ts = pd.to_datetime(original_df['timestamp'])
            start_idx = train_size + look_back
            timestamps = all_ts.iloc[start_idx:start_idx + len(y_true)].values
            if len(timestamps) != len(y_true):
                print(f"⚠️  Timestamp mismatch: {len(timestamps)} vs {len(y_true)}; using index.")
                timestamps = range(len(y_true))
        except Exception as e:
            print(f"⚠️  Error creating timestamps ({e}); using index.")
            timestamps = range(len(y_true))
    else:
        timestamps = range(len(y_true))

elif EVAL_SPLIT == 'test':
    # Use the persisted, held-out test CSV (true unseen)
    if not test_data_path or not os.path.exists(test_data_path):
        raise FileNotFoundError("❌ Test data CSV not found. Ensure your training run saved it in metadata['test_data_path'].")
    test_df = pd.read_csv(test_data_path)
    series = test_df[target].values.reshape(-1, 1)

    print(f"📊 Test samples (raw): {len(series):,}")

    X, y_true = _create_dataset_horizon(series, look_back, horizon_steps)
    # Timestamps aligned to first predicted index in the test CSV itself
    if 'timestamp' in test_df.columns:
        try:
            ts = pd.to_datetime(test_df['timestamp'])
            start_idx = look_back
            timestamps = ts.iloc[start_idx:start_idx + len(y_true)].values
            if len(timestamps) != len(y_true):
                print(f"⚠️  Timestamp mismatch: {len(timestamps)} vs {len(y_true)}; using index.")
                timestamps = range(len(y_true))
        except Exception as e:
            print(f"⚠️  Error creating timestamps ({e}); using index.")
            timestamps = range(len(y_true))
    else:
        timestamps = range(len(y_true))

elif EVAL_SPLIT == 'external':
    # Use external validation dataset (e.g., 2025 data)
    external_path = os.environ.get('VAL_EXTERNAL_PATH', None)
    if not external_path or not os.path.exists(external_path):
        raise FileNotFoundError(f"❌ External validation data not found at: {external_path}")

    external_df = pd.read_csv(external_path)
    if target not in external_df.columns:
        raise ValueError(f"❌ Target column '{target}' not found in external data")

    series = external_df[target].values.reshape(-1, 1)
    print(f"📊 External samples (raw): {len(series):,}")

    X, y_true = _create_dataset_horizon(series, look_back, horizon_steps)

    # Timestamps aligned to first predicted index in the external CSV
    if 'timestamp' in external_df.columns:
        try:
            ts = pd.to_datetime(external_df['timestamp'])
            start_idx = look_back
            timestamps = ts.iloc[start_idx:start_idx + len(y_true)].values
            if len(timestamps) != len(y_true):
                print(f"⚠️  Timestamp mismatch: {len(timestamps)} vs {len(y_true)}; using index.")
                timestamps = range(len(y_true))
        except Exception as e:
            print(f"⚠️  Error creating timestamps ({e}); using index.")
            timestamps = range(len(y_true))
    else:
        timestamps = range(len(y_true))

else:
    raise ValueError("EVAL_SPLIT must be 'validation', 'test', or 'external'.")

print(f"📊 Features shape: {X.shape}")
print(f"📊 Targets shape: {y_true.shape}")

# --- Load model & scalers ---
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ ANN model not found at: {model_path}")

# Load model without compiling (avoids loss function issues)
# We only need it for inference, not training
model = load_model(model_path, compile=False)
sc_X = joblib.load(scaler_x_path)
sc_y = joblib.load(scaler_y_path)

print(f"\n🔮 Making predictions on {EVAL_SPLIT} set:")
print(f"📊 Look-back: {look_back}")
print(f"📊 Samples: {len(X)}")

# --- Predict ---
X_scaled = sc_X.transform(X)
y_pred_scaled = model.predict(X_scaled, verbose=0).ravel()
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = y_true.reshape(-1)

print(f"📊 Arrays - y_true: {len(y_true)}, y_pred: {len(y_pred)}")

# --- Metrics ---
# Capacity-normalized error (matches training MAPE definition)
cap_mape = np.mean(np.abs(y_true - y_pred) / cap) * 100

# Standard APE% (handle zeros safely: NaN where y_true == 0)
with np.errstate(divide='ignore', invalid='ignore'):
    ape = np.where(np.abs(y_true) > 0, np.abs(y_true - y_pred) / np.abs(y_true) * 100, np.nan)
std_mape = np.nanmean(ape)

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"\n📊 {EVAL_SPLIT.capitalize()} Performance:")
print("="*35)
print(f"✅ Capacity-MAPE: {cap_mape:.2f}%  (|error| / cap)")
print(f"✅ Standard MAPE: {std_mape:.2f}%  (|error| / |y_true|, ignoring zeros)")
print(f"✅ RMSE: {rmse:.6f}")
print(f"✅ MAE:  {mae:.6f}")

# --- Compare with stored training/validation performance (if evaluating validation) ---
if EVAL_SPLIT == 'validation' and 'performance' in metadata:
    train_perf = metadata['performance']['train']
    val_perf = metadata['performance']['validation']
    print(f"\n📈 Stored Performance (from training):")
    print("="*35)
    print(f"🎯 Training Capacity-MAPE: {train_perf['mape']:.2f}%")
    print(f"🔍 Validation Capacity-MAPE: {val_perf['mape']:.2f}%")
    print(f"📊 Difference: {val_perf['mape'] - train_perf['mape']:.2f}%")
    if val_perf['mape'] > train_perf['mape'] * 1.5:
        print("⚠️  Significant overfitting detected!")
    elif val_perf['mape'] > train_perf['mape'] * 1.2:
        print("⚠️  Mild overfitting detected")
    else:
        print("✅ Good generalization")

# --- Save results to CSV (with options) ---
os.makedirs(CSV_OUT_DIR, exist_ok=True)
forecast_output_path = os.path.join(CSV_OUT_DIR, CSV_FILENAME)

# Per-row errors
abs_err = np.abs(y_true - y_pred)
with np.errstate(divide='ignore', invalid='ignore'):
    percent_err_by_true = np.where(np.abs(y_true) > 0, abs_err / np.abs(y_true) * 100, np.nan)
capacity_percent_err = abs_err / cap * 100

out_df = pd.DataFrame({
    'timestamp': timestamps,
    f'{target}_actual': y_true,
    f'{target}_predicted': y_pred,
    'abs_error': abs_err,
    'percent_error': percent_err_by_true,       # standard APE% (NaN where y_true==0)
    'capacity_percent_error': capacity_percent_err,  # |err| / cap * 100
    'split': EVAL_SPLIT
})

out_df.to_csv(
    forecast_output_path,
    index=CSV_INDEX,
    sep=CSV_SEP,
    decimal=CSV_DECIMAL,
    float_format=CSV_FLOAT_FMT,
    date_format=CSV_DATE_FORMAT
)

print(f"\n📁 Results saved to: {forecast_output_path}")

# --- Save metrics JSON (for main_training.ipynb) ---
metrics_json_path = os.path.join(CSV_OUT_DIR, f"{target}_{horizon_label}_{EVAL_SPLIT}_metrics.json")
metrics_dict = {
    'cap_mape': float(cap_mape),
    'std_mape': float(std_mape),
    'rmse': float(rmse),
    'mae': float(mae),
    'split': EVAL_SPLIT,
    'target': target,
    'horizon_label': horizon_label
}
with open(metrics_json_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)
print(f"📁 Metrics JSON saved to: {metrics_json_path}")

# --- Sample prints ---
print(f"\n🎯 Sample predictions vs actual:")
for i in range(min(5, len(y_pred))):
    print(f"  {i+1}: Actual={y_true[i]:.6f}  Pred={y_pred[i]:.6f}  AbsErr={abs_err[i]:.6f}")
