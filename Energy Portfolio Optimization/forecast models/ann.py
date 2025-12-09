# ann.py — ANN with Train/Val/Test + EarlyStopping + Multi-Horizon
import numpy as np
import pandas as pd
import os
import json
import gc
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from math import sqrt
from create_dataset import create_dataset  # used for horizon_steps==1

def _create_dataset_horizon(dataset, look_back, horizon_steps):
    """Window univariate series to predict exactly ``horizon_steps`` ahead."""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")

    if horizon_steps == 1:
        return create_dataset(dataset, look_back)

    dataX, dataY = [], []
    N = len(dataset)
    end = N - look_back - horizon_steps + 1
    if end <= 0:
        return np.empty((0, look_back)), np.empty((0,))

    for i in range(end):
        dataX.append(dataset[i:i + look_back, 0])
        target_idx = i + look_back + horizon_steps - 1
        dataY.append(dataset[target_idx, 0])

    return np.array(dataX), np.array(dataY)

def _std_mape(y_true, y_pred, eps=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(np.abs(y_true), eps)
        ape = np.where(denom > 0, np.abs(y_true - y_pred) / denom * 100.0, np.nan)
    return float(np.nanmean(ape))

def ann_model(
    new_data, i, look_back, cap, target,
    train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
    epochs=200, batch_size=64, shuffle=True, patience=12,
    horizon_steps=1, horizon_label="immediate"
):
    """
    Trains an ANN for a given horizon.
    Artifacts are saved using the pattern: {target}_{horizon_label}_*.*
    Example: hydro_strategic_model.keras

    Returns: dict with metrics, paths, and in-memory model/scalers.
    """

    # ----------------
    # Prepare data
    # ----------------
    data1 = (
        new_data.loc[new_data['Month'].isin(i)]
        .reset_index(drop=True)
        .dropna(subset=[target])
    )
    series = data1[target].values.reshape(-1, 1)
    total_samples = len(series)
    print(f"📊 Total samples: {total_samples:,}")
    print(f"🕒 Horizon label: {horizon_label} | steps ahead: {horizon_steps}")

    if total_samples <= (look_back + horizon_steps):
        raise ValueError(
            f"Not enough samples ({total_samples}) for look_back={look_back} and horizon={horizon_steps}"
        )

    # ----------------
    # Chronological 3-way split
    # ----------------
    train_size = int(total_samples * train_ratio)
    val_size   = int(total_samples * val_ratio)
    test_size  = total_samples - train_size - val_size

    train_data = series[:train_size]
    val_data   = series[train_size:train_size + val_size]
    test_data  = series[train_size + val_size:]

    print(f"📊 Train: {len(train_data):,} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"📊 Validation: {len(val_data):,} ({len(val_data)/total_samples*100:.1f}%)")
    print(f"📊 Test: {len(test_data):,} ({len(test_data)/total_samples*100:.1f}%)")

    # ----------------
    # Windowing (multi-horizon aware)
    # ----------------
    trainX, trainY = _create_dataset_horizon(train_data, look_back, horizon_steps)
    valX,   valY   = _create_dataset_horizon(val_data,   look_back, horizon_steps)
    testX,  testY  = _create_dataset_horizon(test_data,  look_back, horizon_steps)

    if trainX.size == 0 or valX.size == 0 or testX.size == 0:
        raise ValueError("Insufficient windowed samples for given look_back/horizon after splitting.")

    # ----------------
    # Scaling (fit on train only)
    # ----------------
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(trainX)
    y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()

    X_val = sc_X.transform(valX)
    y_val = sc_y.transform(valY.reshape(-1, 1)).ravel()

    X_test = sc_X.transform(testX)
    y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

    # Ensure float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    y_val   = y_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    print(f"📊 Training features: {X_train.shape}")
    print(f"📊 Validation features: {X_val.shape}")
    print(f"📊 Test features: {X_test.shape}")

    # ----------------
    # Seeds & determinism
    # ----------------
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['PYTHONHASHSEED'] = '0'
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    # ----------------
    # Build IMPROVED ANN (Option 3: Fix + Improve)
    # ----------------
    # IMPROVEMENTS:
    # 1. Increased neurons: 128 → 256
    # 2. Added 2nd hidden layer (128 neurons)
    # 3. Added Dropout (0.2) to prevent overfitting
    # 4. Slightly lower learning rate for stability
    from tensorflow.keras.layers import Dropout

    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008))

    # ----------------
    # Output dirs & filenames (target + horizon label)
    # ----------------
    for d in ["models", "scalers", "metadata", "datasets", "history", "seeds", "feature_matrices"]:
        os.makedirs(d, exist_ok=True)

    prefix = f"{target}_{horizon_label}"
    model_path_best         = f"models/{prefix}_model_best.h5"
    model_path              = f"models/{prefix}_model.h5"
    scaler_x_path           = f"scalers/{prefix}_sc_X.pkl"
    scaler_y_path           = f"scalers/{prefix}_sc_y.pkl"
    original_data_path      = f"datasets/{prefix}_original_series.csv"
    test_data_path          = f"datasets/{prefix}_test_data.csv"
    seed_window_raw_path    = f"seeds/{prefix}_seed_window_raw.npy"
    seed_window_scaled_path = f"seeds/{prefix}_seed_window_scaled.npy"
    history_path            = f"history/{prefix}_history.json"
    metadata_path           = f"metadata/{prefix}_metadata.json"
    # Feature matrices paths
    feature_matrix_train_X_path = f"feature_matrices/{prefix}_train_X.npy"
    feature_matrix_train_y_path = f"feature_matrices/{prefix}_train_y.npy"
    feature_matrix_val_X_path   = f"feature_matrices/{prefix}_val_X.npy"
    feature_matrix_val_y_path   = f"feature_matrices/{prefix}_val_y.npy"
    feature_matrix_test_X_path  = f"feature_matrices/{prefix}_test_X.npy"
    feature_matrix_test_y_path  = f"feature_matrices/{prefix}_test_y.npy"

    # ----------------
    # Callbacks
    # ----------------
    ckpt = ModelCheckpoint(
        filepath=model_path_best,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=0
    )
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    rlrop = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(2, patience//3),
        min_lr=1e-5,
        verbose=0
    )

    # ----------------
    # Train
    # ----------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        callbacks=[ckpt, es, rlrop],
        verbose=0
    )

    # ----------------
    # Predictions (inverse transform)
    # ----------------
    y_pred_train = model.predict(X_train, verbose=0).ravel()
    y_pred_val   = model.predict(X_val,   verbose=0).ravel()
    y_pred_test  = model.predict(X_test,  verbose=0).ravel()

    y_pred_train_inv = sc_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
    y_train_inv      = sc_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_pred_val_inv   = sc_y.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
    y_val_inv        = sc_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_test_inv  = sc_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    y_test_inv       = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # ----------------
    # Metrics
    # ----------------
    def cap_mape(y_true, y_pred, cap_):
        return float(np.mean(np.abs(y_true - y_pred) / cap_) * 100.0)

    train_mape = cap_mape(y_train_inv, y_pred_train_inv, cap)
    val_mape   = cap_mape(y_val_inv,   y_pred_val_inv,   cap)
    test_mape  = cap_mape(y_test_inv,  y_pred_test_inv,  cap)

    train_rmse = float(sqrt(mean_squared_error(y_train_inv, y_pred_train_inv)))
    val_rmse   = float(sqrt(mean_squared_error(y_val_inv,   y_pred_val_inv)))
    test_rmse  = float(sqrt(mean_squared_error(y_test_inv,  y_pred_test_inv)))

    train_mae = float(mean_absolute_error(y_train_inv, y_pred_train_inv))
    val_mae   = float(mean_absolute_error(y_val_inv,   y_pred_val_inv))
    test_mae  = float(mean_absolute_error(y_test_inv,  y_pred_test_inv))

    # Standard MAPE (and epsilon-stabilized)
    train_std     = _std_mape(y_train_inv, y_pred_train_inv, eps=0.0)
    val_std       = _std_mape(y_val_inv,   y_pred_val_inv,   eps=0.0)
    test_std      = _std_mape(y_test_inv,  y_pred_test_inv,  eps=0.0)
    train_std_eps = _std_mape(y_train_inv, y_pred_train_inv, eps=1e-8)
    val_std_eps   = _std_mape(y_val_inv,   y_pred_val_inv,   eps=1e-8)
    test_std_eps  = _std_mape(y_test_inv,  y_pred_test_inv,  eps=1e-8)

    print(f"\n📊 Performance Results")
    print("="*50)
    print(f"🎯 Train  - MAPE: {train_mape:.3f}% | StdMAPE: {train_std:.3f}% | StdMAPE(ε): {train_std_eps:.3f}% | RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
    print(f"🔍 Val    - MAPE: {val_mape:.3f}% | StdMAPE: {val_std:.3f}% | StdMAPE(ε): {val_std_eps:.3f}% | RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
    print(f"🚀 Test   - MAPE: {test_mape:.3f}% | StdMAPE: {test_std:.3f}% | StdMAPE(ε): {test_std_eps:.3f}% | RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")

    if val_mape > train_mape * 1.5:
        print("⚠️  Possible overfitting (val >> train).")
    elif abs(val_mape - test_mape) < max(1e-9, train_mape * 0.1):
        print("✅ Good generalization (val ≈ test).")

    # ----------------
    # Save artifacts (compatible format)
    # ----------------
    # Save model in more compatible way
    try:
        # Method 1: Save with explicit options for compatibility
        model.save(model_path, save_format='h5', include_optimizer=False)
        print(f"✅ Model saved (method 1): {model_path}")
    except Exception as e:
        print(f"⚠️ Method 1 failed: {e}")
        try:
            # Method 2: Recreate model and save weights only
            model_json = model.to_json()
            with open(model_path.replace('.h5', '_architecture.json'), 'w') as f:
                f.write(model_json)
            model.save_weights(model_path.replace('.h5', '_weights.h5'))
            print(f"✅ Model saved (method 2): architecture + weights")
        except Exception as e2:
            print(f"❌ Both save methods failed: {e2}")
            # Fallback: save the model anyway
            model.save(model_path)

    joblib.dump(sc_X, scaler_x_path)
    joblib.dump(sc_y, scaler_y_path)

    # Save feature matrices (scaled X and y for train/val/test)
    np.save(feature_matrix_train_X_path, X_train)
    np.save(feature_matrix_train_y_path, y_train)
    np.save(feature_matrix_val_X_path, X_val)
    np.save(feature_matrix_val_y_path, y_val)
    np.save(feature_matrix_test_X_path, X_test)
    np.save(feature_matrix_test_y_path, y_test)

    # original series used for splitting (handy for validation split recreation)
    pd.DataFrame({
        'timestamp': data1['Date'] if 'Date' in data1.columns else range(len(series)),
        target: series.flatten()
    }).to_csv(original_data_path, index=False)

    # persist true test slice (chronological tail)
    test_start_idx = train_size + val_size
    pd.DataFrame({
        'timestamp': data1['Date'].iloc[test_start_idx:] if 'Date' in data1.columns else range(test_start_idx, total_samples),
        target: test_data.flatten(),
        'split': 'test'
    }).to_csv(test_data_path, index=False)

    # seed windows (last look_back of full series) raw + scaled (of last window)
    seed_window_raw = series[-look_back:].astype(np.float32).reshape(1, -1)
    lastX_full, _ = _create_dataset_horizon(series, look_back, horizon_steps)
    # use only the last input window (scale with train scaler)
    seed_window_scaled = StandardScaler().fit_transform(trainX).astype(np.float32)  # not used; keep for compatibility
    # better: scale lastX_full with train scaler:
    seed_window_scaled = sc_X.transform(lastX_full[-1:].astype(np.float32))
    np.save(seed_window_raw_path, seed_window_raw)
    np.save(seed_window_scaled_path, seed_window_scaled)

    # training history
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_val_loss = float(np.min(history.history["val_loss"]))
    best_epoch = int(np.argmin(history.history["val_loss"]))
    with open(history_path, "w") as f:
        json.dump({"history": hist, "best_val_loss": best_val_loss, "best_epoch": best_epoch}, f, indent=2)

    # metadata
    metadata = {
        'target_column': target,
        'horizon_label': horizon_label,
        'horizon_steps': int(horizon_steps),
        'look_back': look_back,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'months_used': i,
        'cap': float(cap),
        'train_size': int(train_size),
        'val_size': int(val_size),
        'test_size': int(test_size),
        'total_samples': int(total_samples),
        'architecture': '256-128 (2 hidden layers with dropout 0.2)',
        'model_type': 'ANN_3way_split_multi_horizon_improved',
        'training_start': str(data1['Date'].iloc[0]) if 'Date' in data1.columns else "unknown",
        'training_end': str(data1['Date'].iloc[-1]) if 'Date' in data1.columns else "unknown",
        'input_features': int(X_train.shape[1]),
        'model_path': model_path,
        'model_path_best': model_path_best,
        'scaler_x_path': scaler_x_path,
        'scaler_y_path': scaler_y_path,
        'original_data_path': original_data_path,
        'test_data_path': test_data_path,
        'seed_window_raw_path': seed_window_raw_path,
        'seed_window_scaled_path': seed_window_scaled_path,
        'history_path': history_path,
        'feature_matrix_train_X_path': feature_matrix_train_X_path,
        'feature_matrix_train_y_path': feature_matrix_train_y_path,
        'feature_matrix_val_X_path': feature_matrix_val_X_path,
        'feature_matrix_val_y_path': feature_matrix_val_y_path,
        'feature_matrix_test_X_path': feature_matrix_test_X_path,
        'feature_matrix_test_y_path': feature_matrix_test_y_path,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'performance': {
            'train': {
                'mape': train_mape, 'std_mape': train_std, 'std_mape_eps': train_std_eps,
                'rmse': train_rmse, 'mae': train_mae
            },
            'validation': {
                'mape': val_mape, 'std_mape': val_std, 'std_mape_eps': val_std_eps,
                'rmse': val_rmse, 'mae': val_mae
            },
            'test': {
                'mape': test_mape, 'std_mape': test_std, 'std_mape_eps': test_std_eps,
                'rmse': test_rmse, 'mae': test_mae
            }
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Model (best) saved to: {model_path_best}")
    print(f"✅ Model (final w/ best weights) saved to: {model_path}")
    print(f"✅ Scalers saved to: {scaler_x_path}, {scaler_y_path}")
    print(f"✅ History saved to: {history_path}")
    print(f"✅ Metadata saved to: {metadata_path}")
    print(f"✅ Seed windows saved to: {seed_window_raw_path}, {seed_window_scaled_path}")
    print(f"✅ Test data saved to: {test_data_path}")
    print(f"✅ Feature matrices saved to: feature_matrices/{prefix}_*_X.npy and feature_matrices/{prefix}_*_y.npy")

    # MEMORY MANAGEMENT: Clear model and scalers from memory after saving
    # They are already saved to disk, so we can safely delete them from memory
    # to prevent memory accumulation when training multiple models
    del model
    del sc_X
    del sc_y
    # Suppress deprecation warning from clear_session
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        tf.keras.backend.clear_session()
    gc.collect()

    return {
        'MAPE': val_mape,
        'RMSE': val_rmse,
        'MAE': val_mae,
        # Don't return model/scalers in memory - they're saved to disk
        # Access them later via: load_model(model_path) and joblib.load(scaler_path)
        'metadata': metadata,
        'model_path': model_path,
        'model_path_best': model_path_best,
        'metadata_path': metadata_path,
        'history_path': history_path
    }
