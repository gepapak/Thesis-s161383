# train_models.py — train per target×horizon on trainingdata.csv with GPU acceleration
import os
import sys
import json
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ann import ann_model

print("🚀 Starting Multi-Horizon Model Training with TensorFlow 2.10.1 + CUDA")
print("=" * 70)

# --- Config ---
targets = ['wind','hydro','solar','load','price']

HORIZONS = {
    "immediate": 1,      # direct next step
    "short": 6,          # 6 steps ahead
    "medium": 24,        # 24 steps ahead
    "long": 144,         # 144 steps ahead
    "strategic": 1008    # 1008 steps ahead
}

months_filter = list(range(1, 13))
look_back = 6

train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15

epochs   = 200
batch_size = 64
shuffle  = True
patience = 12

TRAIN_PATH = "trainingdata.csv"  # Use your main training data
EXT_VALIDATION_PATH = "unseendata.csv"  # Use your validation data

# --- IO prep ---
for d in ["metadata", "models", "scalers", "feature_matrices", "datasets", "history", "seeds", "forecast_results"]:
    os.makedirs(d, exist_ok=True)

# Load training data (for month filter)
print(f"📊 Loading training data from: {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
df_train['Date']  = pd.to_datetime(df_train['timestamp'])
df_train['Year']  = df_train['Date'].dt.year
df_train['Month'] = df_train['Date'].dt.month

print(f"📊 Training data shape: {df_train.shape}")
print(f"📊 Date range: {df_train['Date'].min()} to {df_train['Date'].max()}")

def read_metrics_json(path):
    with open(path, "r") as f:
        return json.load(f)

rows = []

print(f"\n🎯 Training {len(targets)} targets × {len(HORIZONS)} horizons = {len(targets) * len(HORIZONS)} models")
print("🚀 GPU acceleration enabled!")

with tqdm(total=len(targets), desc="Targets", unit="target") as tbar:
    for target in targets:
        with tqdm(total=len(HORIZONS), desc=f"{target} horizons", unit="hz", leave=False) as hbar:
            for hlabel, hsteps in HORIZONS.items():
                tqdm.write(f"\n📌 Target: {target} | Horizon: {hlabel} ({hsteps} steps)")
                # Prepare training view
                new_data = df_train[['Month', 'Year', 'Date', target]].copy()
                cap = float(new_data[target].max())

                # 1) Train per horizon
                out = ann_model(
                    new_data=new_data,
                    i=months_filter,
                    look_back=look_back,
                    cap=cap,
                    target=target,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    patience=patience,
                    horizon_steps=hsteps,
                    horizon_label=hlabel
                )

                # Read performance from metadata
                metadata_path = out.get('metadata_path') or os.path.join("metadata", f"{target}_{hlabel}_metadata.json")
                with open(metadata_path, "r") as f:
                    md = json.load(f)
                perf = md['performance']
                train_perf = perf['train']

                # Skip validation for now - just focus on training
                tqdm.write(f"✅ {target}_{hlabel}: Train MAPE={train_perf['mape']:.3f}%")

                # Collect row
                rows.append({
                    'target': target,
                    'horizon_label': hlabel,
                    'horizon_steps': hsteps,
                    'train_MAPE': float(train_perf['mape']),
                    'train_STD_MAPE': float(train_perf.get('std_mape', np.nan)),
                    'train_RMSE': float(train_perf['rmse']),
                    'train_MAE' : float(train_perf['mae']),
                    'model_path'     : md.get('model_path'),
                    'model_path_best': md.get('model_path_best'),
                    'metadata_path'  : metadata_path,
                    'history_path'   : md.get('history_path'),
                })
                hbar.update(1)
        tbar.update(1)

# Save combined metrics CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
results_df = pd.DataFrame(rows)
out_csv = f"training_results_{timestamp}.csv"
results_df.to_csv(out_csv, index=False)

print(f"\n🎉 Training Complete!")
print(f"📊 Results saved to: {out_csv}")
print(f"📁 Models saved to: models/")
print(f"📁 Scalers saved to: scalers/")
print("\n📈 Performance Summary:")
print(results_df[['target', 'horizon_label', 'train_MAPE']].to_string(index=False))

print(f"\n🔧 Next steps:")
print(f"1. Copy models: xcopy models\\*.* saved_models\\ /Y")
print(f"2. Copy scalers: xcopy scalers\\*.* saved_scalers\\ /Y")
print(f"3. Test: python main.py --data_path trainingdata.csv --timesteps 10000 --device cuda --precompute_forecasts")
