# validation.py — evaluate saved model on: validation | test | external (per horizon)
import os
import json
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import joblib

# --- CONFIG via env ---
target = os.getenv("VAL_TARGET", "wind")
EVAL_SPLIT = os.getenv("VAL_SPLIT", "validation").lower()   # 'validation' | 'test' | 'external'
horizon_label = os.getenv("VAL_HORIZON_LABEL", "immediate")
horizon_steps = int(os.getenv("VAL_HORIZON_STEPS", "1"))
VAL_MONTHS = os.getenv("VAL_MONTHS", None)
VAL_EXTERNAL_PATH = os.getenv("VAL_EXTERNAL_PATH", "validationset.csv")

# CSV options
CSV_OUT_DIR = "forecast_results"
CSV_SEP = ','
CSV_DECIMAL = '.'
CSV_FLOAT_FMT = '%.6f'
CSV_INDEX = False
CSV_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Helpers
def _create_dataset_horizon(dataset, look_back, horizon_steps):
    if horizon_steps == 1:
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i+look_back, 0])
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)
    dataX, dataY = [], []
    N = len(dataset)
    end = N - look_back - (horizon_steps - 1)
    for i in range(end):
        dataX.append(dataset[i:i+look_back, 0])
        dataY.append(dataset[i + look_back + horizon_steps - 1, 0])
    return np.array(dataX), np.array(dataY)

def _std_mape(y_true, y_pred, eps=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(np.abs(y_true), eps)
        ape = np.where(denom > 0, np.abs(y_true - y_pred) / denom * 100.0, np.nan)
    return float(np.nanmean(ape))

# Load metadata (pick file by target + horizon label)
metadata_path = os.path.join("metadata", f"{target}_{horizon_label}_metadata.json")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata not found: {metadata_path}")

with open(metadata_path, "r") as f:
    md = json.load(f)

cap        = md['cap']
look_back  = md['look_back']
train_ratio = md['train_ratio']
val_ratio   = md['val_ratio']
test_ratio  = md['test_ratio']

model_path = md.get('model_path_best') or md['model_path']
scaler_x_path = md['scaler_x_path']
scaler_y_path = md['scaler_y_path']
original_data_path = md['original_data_path']
test_data_path     = md.get('test_data_path', None)

print(f"Evaluating {target} [{horizon_label} | {horizon_steps} steps] on: {EVAL_SPLIT.upper()}")
print("="*70)
print(f"Model: {model_path}")

# Load model & scalers
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
try:
    model = load_model(model_path, compile=False)
    # Recompile the model to ensure compatibility
    model.compile(loss='mse', optimizer='adam')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to recreate model from metadata...")
    # Recreate model architecture as fallback
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(look_back,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    print("WARNING: Using untrained model architecture - predictions will be random!")
sc_X  = joblib.load(scaler_x_path)
sc_y  = joblib.load(scaler_y_path)

# Build X,y and timestamps
timestamps = None

if EVAL_SPLIT == 'validation':
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"Original data not found: {original_data_path}")
    original_df = pd.read_csv(original_data_path)
    series = original_df[target].values.reshape(-1, 1)

    total_samples = len(series)
    train_size = int(total_samples * train_ratio)
    val_size   = int(total_samples * val_ratio)
    print(f"Total: {total_samples:,} | Train: {train_size:,} | Val: {val_size:,}")

    val_data = series[train_size:train_size + val_size]
    X, y_true = _create_dataset_horizon(val_data, look_back, horizon_steps)

    if 'timestamp' in original_df.columns:
        try:
            all_ts = pd.to_datetime(original_df['timestamp'])
            # first predicted index inside validation slice
            start_idx = train_size + look_back + (horizon_steps - 1)
            timestamps = all_ts.iloc[start_idx:start_idx + len(y_true)].values
            if len(timestamps) != len(y_true):
                print("WARNING: Timestamp mismatch; using index.")
                timestamps = range(len(y_true))
        except Exception as e:
            print(f"WARNING: Timestamp error ({e}); using index.")
            timestamps = range(len(y_true))
    else:
        timestamps = range(len(y_true))

elif EVAL_SPLIT == 'test':
    if not test_data_path or not os.path.exists(test_data_path):
        raise FileNotFoundError("Test data CSV not found. Ensure training saved it in metadata['test_data_path'].")
    test_df = pd.read_csv(test_data_path)
    series = test_df[target].values.reshape(-1, 1)
    print(f"Test samples (raw): {len(series):,}")

    X, y_true = _create_dataset_horizon(series, look_back, horizon_steps)

    if 'timestamp' in test_df.columns:
        try:
            ts = pd.to_datetime(test_df['timestamp'])
            start_idx = look_back + (horizon_steps - 1)
            timestamps = ts.iloc[start_idx:start_idx + len(y_true)].values
            if len(timestamps) != len(y_true):
                print("WARNING: Timestamp mismatch; using index.")
                timestamps = range(len(y_true))
        except Exception as e:
            print(f"WARNING: Timestamp error ({e}); using index.")
            timestamps = range(len(y_true))
    else:
        timestamps = range(len(y_true))

elif EVAL_SPLIT == 'external':
    if not os.path.exists(VAL_EXTERNAL_PATH):
        raise FileNotFoundError(f"External dataset not found: {VAL_EXTERNAL_PATH}")
    ext_df = pd.read_csv(VAL_EXTERNAL_PATH)

    if 'Date' not in ext_df.columns and 'timestamp' in ext_df.columns:
        ext_df['Date'] = pd.to_datetime(ext_df['timestamp'])
    elif 'Date' in ext_df.columns:
        ext_df['Date'] = pd.to_datetime(ext_df['Date'])
    else:
        raise ValueError("External CSV must have either 'timestamp' or 'Date' column.")

    if 'Month' not in ext_df.columns:
        ext_df['Month'] = ext_df['Date'].dt.month

    if VAL_MONTHS:
        try:
            months_filter = [int(x) for x in VAL_MONTHS.split(",") if x.strip()]
            ext_df = ext_df.loc[ext_df['Month'].isin(months_filter)]
        except Exception:
            pass

    ext_df = ext_df.dropna(subset=[target])
    series = ext_df[target].values.reshape(-1, 1)

    X, y_true = _create_dataset_horizon(series, look_back, horizon_steps)

    base_ts = ext_df['timestamp'] if 'timestamp' in ext_df.columns else ext_df['Date']
    base_ts = pd.to_datetime(base_ts)
    start_idx = look_back + (horizon_steps - 1)
    timestamps = base_ts.iloc[start_idx:start_idx + len(y_true)].values
else:
    raise ValueError("EVAL_SPLIT must be 'validation', 'test', or 'external'.")

if X.size == 0:
    raise ValueError("Empty windowed dataset (check look_back/horizon and slice lengths).")

# Predict (scale X, inverse y)
X_scaled = sc_X.transform(X).astype(np.float32)
y_pred_scaled = model.predict(X_scaled, verbose=0).ravel()
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = y_true.reshape(-1)

# Metrics
cap_mape = float(np.mean(np.abs(y_true - y_pred) / cap) * 100.0)
std_mape = _std_mape(y_true, y_pred, eps=0.0)
std_mape_eps = _std_mape(y_true, y_pred, eps=1e-8)
rmse = float(sqrt(mean_squared_error(y_true, y_pred)))
mae  = float(mean_absolute_error(y_true, y_pred))

print(f"\n{EVAL_SPLIT.capitalize()} Performance:")
print("="*40)
print(f"Capacity-MAPE: {cap_mape:.3f}%")
print(f"Standard MAPE: {std_mape:.3f}%")
print(f"Std MAPE (eps): {std_mape_eps:.3f}%")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")

# Save outputs
os.makedirs(CSV_OUT_DIR, exist_ok=True)
csv_name    = f"{target}_{horizon_label}_{EVAL_SPLIT}_results.csv"
json_name   = f"{target}_{horizon_label}_{EVAL_SPLIT}_metrics.json"
paired_name = f"{target}_{horizon_label}_{EVAL_SPLIT}_paired.csv"
two_col_name = f"{target}_{horizon_label}_{EVAL_SPLIT}_pred_vs_actual.csv"  # <-- NEW: exactly 2 columns

forecast_output_path = os.path.join(CSV_OUT_DIR, csv_name)
metrics_path         = os.path.join(CSV_OUT_DIR, json_name)
paired_path          = os.path.join(CSV_OUT_DIR, paired_name)
two_col_path         = os.path.join(CSV_OUT_DIR, two_col_name)

# rich row-by-row results
abs_err = np.abs(y_true - y_pred)
with np.errstate(divide='ignore', invalid='ignore'):
    percent_err_by_true = np.where(np.abs(y_true) > 0, abs_err / np.abs(y_true) * 100.0, np.nan)
capacity_percent_err = abs_err / cap * 100.0

out_df = pd.DataFrame({
    'timestamp': timestamps,
    f'{target}_actual': y_true,
    f'{target}_predicted': y_pred,
    'abs_error': abs_err,
    'percent_error': percent_err_by_true,
    'capacity_percent_error': capacity_percent_err,
    'split': EVAL_SPLIT,
    'horizon_label': horizon_label,
    'horizon_steps': horizon_steps
})
out_df.to_csv(
    forecast_output_path,
    index=CSV_INDEX, sep=CSV_SEP, decimal=CSV_DECIMAL,
    float_format=CSV_FLOAT_FMT, date_format=CSV_DATE_FORMAT
)

# aggregate metrics
with open(metrics_path, "w") as f:
    json.dump({
        "target": target,
        "split": EVAL_SPLIT,
        "horizon_label": horizon_label,
        "horizon_steps": int(horizon_steps),
        "n_samples": int(len(y_true)),
        "cap_mape": cap_mape,
        "std_mape": std_mape,
        "std_mape_eps": std_mape_eps,
        "rmse": rmse,
        "mae": mae
    }, f, indent=2)

# paired with timestamp (3 cols)
pd.DataFrame({
    'timestamp': timestamps,
    'forecast_saved_model': y_pred,
    'actual': y_true
}).to_csv(paired_path, index=False)

# NEW: minimal two-column file (exactly forecast & actual)
pd.DataFrame({
    f'{target}_predicted': y_pred,
    f'{target}_actual': y_true
}).to_csv(
    two_col_path,
    index=False, sep=CSV_SEP, decimal=CSV_DECIMAL,
    float_format=CSV_FLOAT_FMT
)

print(f"\nResults saved to: {forecast_output_path}")
print(f"Aggregate metrics saved to: {metrics_path}")
print(f"Paired (with timestamp) saved to: {paired_path}")
print(f"Two-column pred/actual saved to: {two_col_path}")
