# main.py — train per target×horizon on trainset.csv, then validate (test + external) via validation.py
import os
import sys
import json
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ann import ann_model

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

TRAIN_PATH = "trainingdata.csv"
EXT_VALIDATION_PATH = "unseendata.csv"

# --- IO prep ---
for d in ["metadata", "saved_models", "saved_scalers", "feature_matrices", "datasets", "history", "seeds", "forecast_results"]:
    os.makedirs(d, exist_ok=True)

# Load training data (for month filter)
df_train = pd.read_csv(TRAIN_PATH)
df_train['Date']  = pd.to_datetime(df_train['timestamp'])
df_train['Year']  = df_train['Date'].dt.year
df_train['Month'] = df_train['Date'].dt.month

def read_metrics_json(path):
    with open(path, "r") as f:
        return json.load(f)

rows = []

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

                # 2) Run validation.py for internal unseen test
                env = os.environ.copy()
                env["VAL_TARGET"] = target
                env["VAL_HORIZON_LABEL"] = hlabel
                env["VAL_HORIZON_STEPS"] = str(hsteps)
                env["VAL_MONTHS"] = ",".join(map(str, months_filter))

                env["VAL_SPLIT"] = "test"
                res = subprocess.run([sys.executable, "validation.py"], env=env, capture_output=True, text=True)
                if res.returncode != 0:
                    print(res.stdout); print(res.stderr)
                    raise RuntimeError(f"validation.py failed (split=test, target={target}, horizon={hlabel})")
                test_metrics_json = os.path.join("forecast_results", f"{target}_{hlabel}_test_metrics.json")
                test_metrics = read_metrics_json(test_metrics_json)

                # 3) Run validation.py for external unseen dataset
                env["VAL_SPLIT"] = "external"
                env["VAL_EXTERNAL_PATH"] = os.path.abspath(EXT_VALIDATION_PATH)
                res = subprocess.run([sys.executable, "validation.py"], env=env, capture_output=True, text=True)
                if res.returncode != 0:
                    print(res.stdout); print(res.stderr)
                    raise RuntimeError(f"validation.py failed (split=external, target={target}, horizon={hlabel})")
                ext_metrics_json = os.path.join("forecast_results", f"{target}_{hlabel}_external_metrics.json")
                ext_metrics = read_metrics_json(ext_metrics_json)

                # 4) Collect one row per target×horizon
                rows.append({
                    'target': target,
                    'horizon_label': hlabel,
                    'horizon_steps': hsteps,

                    # training metrics (capacity & std mape if present)
                    'train_MAPE': float(train_perf['mape']),
                    'train_STD_MAPE': float(train_perf.get('std_mape', np.nan)),
                    'train_RMSE': float(train_perf['rmse']),
                    'train_MAE' : float(train_perf['mae']),

                    # internal test (unseen part of trainset) from validation.py
                    'unseen_train_MAPE': float(test_metrics['cap_mape']),
                    'unseen_train_STD_MAPE': float(test_metrics.get('std_mape', np.nan)),
                    'unseen_train_RMSE': float(test_metrics['rmse']),
                    'unseen_train_MAE' : float(test_metrics['mae']),

                    # external unseen dataset from validation.py
                    'unseen_external_MAPE': float(ext_metrics['cap_mape']),
                    'unseen_external_STD_MAPE': float(ext_metrics.get('std_mape', np.nan)),
                    'unseen_external_RMSE': float(ext_metrics['rmse']),
                    'unseen_external_MAE' : float(ext_metrics['mae']),

                    # paths for traceability
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
out_csv = f"metrics_results_{timestamp}.csv"
results_df.to_csv(out_csv, index=False)
print(f"\n✅ Metrics saved to '{out_csv}'")
