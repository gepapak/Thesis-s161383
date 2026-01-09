"""
Episode-Specific Forecast Model Training

Trains forecast models per episode using data up to that episode's end date.
Saves models, scalers, and metadata to episode-specific directories.

Usage:
    python train_episode_forecasts.py --episode 0 --data_path training_dataset/trainset.csv
    python train_episode_forecasts.py --episode 0 1 2 --data_path training_dataset/trainset.csv  # Multiple episodes
    python train_episode_forecasts.py --all --data_path training_dataset/trainset.csv  # All episodes 0-19
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
import gc
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from math import sqrt

# Import episode info from main
sys.path.insert(0, os.path.dirname(__file__))
from main import get_episode_info

# =========================
# COPIED FROM Forecast_ANN: Dataset creation (self-contained)
# =========================

def create_dataset(dataset, look_back=1, multivariate=False):
    """
    Turns a series (N,1) or matrix (N,F) into (X, y) with right-aligned windows.
    Univariate target is always the first column.
    """
    dataX, dataY = [], []

    if multivariate:
        # X: [i : i+look_back, :], y: dataset[i+look_back, 0]
        for i in range(len(dataset) - look_back):
            X_slice = dataset[i:(i + look_back), :]
            y_value = dataset[i + look_back, 0]
            dataX.append(X_slice)
            dataY.append(y_value)
    else:
        # X: [i : i+look_back, 0], y: dataset[i+look_back, 0]
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


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
        # Target is horizon_steps ahead of the LAST input value
        # Input window: [dataset[i], dataset[i+1], ..., dataset[i+look_back-1]]
        # Last input index: i + look_back - 1
        # Target index: (i + look_back - 1) + horizon_steps = i + look_back + horizon_steps - 1
        target_idx = i + look_back + horizon_steps - 1
        dataY.append(dataset[target_idx, 0])
    
    return np.array(dataX), np.array(dataY)


def _std_mape(y_true, y_pred, eps=0.0):
    """Standard MAPE with epsilon stabilization"""
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(np.abs(y_true), eps)
        ape = np.where(denom > 0, np.abs(y_true - y_pred) / denom * 100.0, np.nan)
    return float(np.nanmean(ape))


def train_episode_model(
    episode_num,
    target,
    horizon_label,
    horizon_steps,
    data_filtered,
    look_back=24,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=200,
    batch_size=64,
    shuffle=True,
    patience=12,
    output_base_dir="forecast_models"
):
    """
    Train a forecast model FROM SCRATCH for a specific episode, target, and horizon.
    
    IMPORTANT: This function trains models independently for each episode.
    - No loading from previous episodes (unlike RL weights)
    - Uses data filtered up to episode end date
    - Each episode has its own model files in forecast_models/episode_N/
    
    Args:
        episode_num: Episode number (0-19)
        target: Target variable ('price', 'wind', 'solar', 'hydro', 'load')
        horizon_label: Horizon label ('immediate', 'short', 'medium', 'long', 'strategic')
        horizon_steps: Number of steps ahead for this horizon
        data_filtered: DataFrame with data up to episode end date (already filtered)
        look_back: Look-back window size (default 24)
        train_ratio, val_ratio, test_ratio: Train/val/test split ratios
        epochs, batch_size, shuffle, patience: Training hyperparameters
        output_base_dir: Base directory for saving outputs
    
    Returns:
        dict with training results and paths
    """
    
    # Create episode-specific output directories
    episode_dir = os.path.join(output_base_dir, f"episode_{episode_num}")
    for subdir in ["models", "scalers", "metadata", "datasets", "history", "seeds", "feature_matrices"]:
        os.makedirs(os.path.join(episode_dir, subdir), exist_ok=True)
    
    # Prepare data
    if target not in data_filtered.columns:
        raise ValueError(f"Target '{target}' not found in data columns: {list(data_filtered.columns)}")
    
    new_data = data_filtered[['Month', 'Year', 'Date', target]].copy()
    
    # Filter by all months (use all available data up to end date)
    months_filter = list(range(1, 13))
    data1 = (
        new_data.loc[new_data['Month'].isin(months_filter)]
        .reset_index(drop=True)
        .dropna(subset=[target])
    )
    
    series = data1[target].values.reshape(-1, 1)
    total_samples = len(series)
    
    print(f"  Target: {target} | Horizon: {horizon_label} ({horizon_steps} steps) | Samples: {total_samples:,}")
    
    if total_samples <= (look_back + horizon_steps):
        raise ValueError(
            f"Not enough samples ({total_samples}) for look_back={look_back} and horizon={horizon_steps}"
        )
    
    # Cap value (max from training data)
    cap = float(series.max())
    
    # Chronological 3-way split
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    train_data = series[:train_size]
    val_data = series[train_size:train_size + val_size]
    test_data = series[train_size + val_size:]
    
    print(f"    Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    
    # Windowing (multi-horizon aware)
    trainX, trainY = _create_dataset_horizon(train_data, look_back, horizon_steps)
    valX, valY = _create_dataset_horizon(val_data, look_back, horizon_steps)
    testX, testY = _create_dataset_horizon(test_data, look_back, horizon_steps)
    
    if trainX.size == 0 or valX.size == 0 or testX.size == 0:
        raise ValueError("Insufficient windowed samples for given look_back/horizon after splitting.")
    
    # Scaling (fit on train only)
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
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Seeds & determinism
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['PYTHONHASHSEED'] = '0'
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    
    # Build model (same architecture as Forecast_ANN)
    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008))
    
    # Output paths (episode-specific)
    prefix = f"{target}_{horizon_label}"
    model_path_best = os.path.join(episode_dir, "models", f"{prefix}_model_best.h5")
    model_path = os.path.join(episode_dir, "models", f"{prefix}_model.h5")
    scaler_x_path = os.path.join(episode_dir, "scalers", f"{prefix}_sc_X.pkl")
    scaler_y_path = os.path.join(episode_dir, "scalers", f"{prefix}_sc_y.pkl")
    original_data_path = os.path.join(episode_dir, "datasets", f"{prefix}_original_series.csv")
    test_data_path = os.path.join(episode_dir, "datasets", f"{prefix}_test_data.csv")
    seed_window_raw_path = os.path.join(episode_dir, "seeds", f"{prefix}_seed_window_raw.npy")
    seed_window_scaled_path = os.path.join(episode_dir, "seeds", f"{prefix}_seed_window_scaled.npy")
    history_path = os.path.join(episode_dir, "history", f"{prefix}_history.json")
    metadata_path = os.path.join(episode_dir, "metadata", f"{prefix}_metadata.json")
    
    # Callbacks
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
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        callbacks=[ckpt, es, rlrop],
        verbose=0
    )
    
    best_epoch = len(history.history['val_loss'])
    best_val_loss = min(history.history['val_loss'])
    
    # Load best model and save final
    try:
        best_model = tf.keras.models.load_model(model_path_best, compile=False)
        best_model.save(model_path)
    except Exception as e:
        print(f"    WARNING: Could not load best model, saving current: {e}")
        model.save(model_path)
    
    # Predictions (inverse transform)
    y_pred_train = model.predict(X_train, verbose=0).ravel()
    y_pred_val = model.predict(X_val, verbose=0).ravel()
    y_pred_test = model.predict(X_test, verbose=0).ravel()
    
    y_pred_train_inv = sc_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
    y_train_inv = sc_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_pred_val_inv = sc_y.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
    y_val_inv = sc_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_test_inv = sc_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Metrics
    def cap_mape(y_true, y_pred, cap_):
        return float(np.mean(np.abs(y_true - y_pred) / cap_) * 100.0)
    
    train_mape = cap_mape(y_train_inv, y_pred_train_inv, cap)
    val_mape = cap_mape(y_val_inv, y_pred_val_inv, cap)
    test_mape = cap_mape(y_test_inv, y_pred_test_inv, cap)
    
    train_rmse = float(sqrt(mean_squared_error(y_train_inv, y_pred_train_inv)))
    val_rmse = float(sqrt(mean_squared_error(y_val_inv, y_pred_val_inv)))
    test_rmse = float(sqrt(mean_squared_error(y_test_inv, y_pred_test_inv)))
    
    train_mae = float(mean_absolute_error(y_train_inv, y_pred_train_inv))
    val_mae = float(mean_absolute_error(y_val_inv, y_pred_val_inv))
    test_mae = float(mean_absolute_error(y_test_inv, y_pred_test_inv))
    
    train_std = _std_mape(y_train_inv, y_pred_train_inv, eps=0.0)
    val_std = _std_mape(y_val_inv, y_pred_val_inv, eps=0.0)
    test_std = _std_mape(y_test_inv, y_pred_test_inv, eps=0.0)
    
    print(f"    Train MAPE: {train_mape:.3f}% | Val MAPE: {val_mape:.3f}% | Test MAPE: {test_mape:.3f}%")
    
    # Save scalers
    joblib.dump(sc_X, scaler_x_path)
    joblib.dump(sc_y, scaler_y_path)
    
    # Save seed window (last look_back values from train set)
    seed_window_raw = train_data[-look_back:].reshape(1, -1)
    seed_window_scaled = sc_X.transform(seed_window_raw)
    np.save(seed_window_raw_path, seed_window_raw)
    np.save(seed_window_scaled_path, seed_window_scaled)
    
    # Save datasets
    pd.DataFrame(series, columns=[target]).to_csv(original_data_path, index=False)
    pd.DataFrame(test_data, columns=[target]).to_csv(test_data_path, index=False)
    
    # Save history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save metadata
    metadata = {
        'target_column': target,
        'horizon_label': horizon_label,
        'horizon_steps': horizon_steps,
        'look_back': look_back,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'months_used': months_filter,
        'cap': float(cap),
        'train_size': int(train_size),
        'val_size': int(val_size),
        'test_size': int(test_size),
        'total_samples': int(total_samples),
        'architecture': '256-128 (2 hidden layers with dropout 0.2)',
        'model_type': 'ANN_3way_split_multi_horizon_episode_specific',
        'training_start': str(data1['Date'].iloc[0]) if 'Date' in data1.columns else "unknown",
        'training_end': str(data1['Date'].iloc[-1]) if 'Date' in data1.columns else "unknown",
        'episode_num': episode_num,
        'input_features': int(X_train.shape[1]),
        'model_path': os.path.relpath(model_path, output_base_dir),
        'model_path_best': os.path.relpath(model_path_best, output_base_dir),
        'scaler_x_path': os.path.relpath(scaler_x_path, output_base_dir),
        'scaler_y_path': os.path.relpath(scaler_y_path, output_base_dir),
        'original_data_path': os.path.relpath(original_data_path, output_base_dir),
        'test_data_path': os.path.relpath(test_data_path, output_base_dir),
        'seed_window_raw_path': os.path.relpath(seed_window_raw_path, output_base_dir),
        'seed_window_scaled_path': os.path.relpath(seed_window_scaled_path, output_base_dir),
        'history_path': os.path.relpath(history_path, output_base_dir),
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'performance': {
            'train': {
                'mape': train_mape, 'std_mape': train_std,
                'rmse': train_rmse, 'mae': train_mae
            },
            'validation': {
                'mape': val_mape, 'std_mape': val_std,
                'rmse': val_rmse, 'mae': val_mae
            },
            'test': {
                'mape': test_mape, 'std_mape': test_std,
                'rmse': test_rmse, 'mae': test_mae
            }
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clear memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
    return {
        'metadata_path': metadata_path,
        'model_path': model_path,
        'model_path_best': model_path_best,
        'scaler_x_path': scaler_x_path,
        'scaler_y_path': scaler_y_path,
        'test_mape': test_mape,
        'val_mape': val_mape
    }


def train_episode_forecasts(episode_num, episode_data_path, output_base_dir="forecast_models"):
    """
    Train all forecast models FROM SCRATCH for a specific episode.
    
    CRITICAL: Each episode trains ONLY on its own 6-month period (not cumulative!):
    - Episode 0: Uses scenario_000.csv (2015 H1: Jan 1 - Jun 30, 2015)
    - Episode 1: Uses scenario_001.csv (2015 H2: Jul 1 - Dec 31, 2015)
    - Episode 19: Uses scenario_019.csv (2024 H2: Jul 1 - Dec 31, 2024)
    
    This ensures forecast models match the episode's data distribution exactly.
    Models are trained independently - no loading from previous episodes.
    
    Args:
        episode_num: Episode number (0-19)
        episode_data_path: Path to episode-specific scenario file (e.g., scenario_000.csv)
        output_base_dir: Base directory for episode outputs
    
    Returns:
        dict with training summary (successful, failed counts)
    """
    print("="*80)
    print(f"TRAINING FORECAST MODELS FOR EPISODE {episode_num}")
    print("="*80)
    
    # Get episode info
    episode_info = get_episode_info(episode_num)
    print(f"Episode: {episode_info['description']}")
    print(f"Period: {episode_info['start_date']} to {episode_info['end_date']}")
    print(f"Data file: {episode_data_path}")
    
    # Load episode-specific scenario file directly (no filtering needed!)
    # CRITICAL: Use load_energy_data to ensure same data processing as MARL training
    # This applies MW conversion if needed to match forecast model training data format
    
    # Try to import load_energy_data from main.py for consistent data processing
    try:
        from main import load_energy_data
        print(f"Loading episode data from {episode_data_path}...")
        # Use load_energy_data to ensure consistent processing (MW conversion if needed)
        # convert_to_raw_units=True ensures data matches forecast model training format
        data_filtered = load_energy_data(
            episode_data_path, 
            convert_to_raw_units=True, 
            config=None,  # Will use defaults for MW conversion
            mw_scale_overrides=None
        )
        # Ensure Date column exists for filtering
        if 'timestamp' in data_filtered.columns:
            data_filtered['Date'] = pd.to_datetime(data_filtered['timestamp'])
        elif 'Date' not in data_filtered.columns:
            raise ValueError("Data must have 'timestamp' or 'Date' column")
    except (ImportError, AttributeError):
        # Fallback: import pandas and handle conversion manually if load_energy_data unavailable
        print(f"Loading episode data from {episode_data_path} (using fallback method)...")
        data_filtered = pd.read_csv(episode_data_path)
        # Parse date column (handle both 'timestamp' and 'Date' columns)
        if 'timestamp' in data_filtered.columns:
            data_filtered['Date'] = pd.to_datetime(data_filtered['timestamp'])
        elif 'Date' in data_filtered.columns:
            data_filtered['Date'] = pd.to_datetime(data_filtered['Date'])
        else:
            raise ValueError("Data must have 'timestamp' or 'Date' column")
    
    # Add Year and Month columns (needed for training)
    if 'Year' not in data_filtered.columns:
        data_filtered['Year'] = data_filtered['Date'].dt.year
    if 'Month' not in data_filtered.columns:
        data_filtered['Month'] = data_filtered['Date'].dt.month
    
    print(f"  Loaded {len(data_filtered):,} samples (from {data_filtered['Date'].min()} to {data_filtered['Date'].max()})")
    
    # Training configuration
    targets = ['wind', 'hydro', 'solar', 'load', 'price']
    horizons = {
        "immediate": 1,
        "short": 6,
        "medium": 24,
        "long": 144,
        "strategic": 1008
    }
    look_back = 24
    
    # Train all target×horizon combinations
    results = []
    total_models = len(targets) * len(horizons)
    current_model = 0
    
    for target in targets:
        for horizon_label, horizon_steps in horizons.items():
            current_model += 1
            print(f"\n[{current_model}/{total_models}] Training {target} {horizon_label}...")
            
            try:
                result = train_episode_model(
                    episode_num=episode_num,
                    target=target,
                    horizon_label=horizon_label,
                    horizon_steps=horizon_steps,
                    data_filtered=data_filtered,
                    look_back=look_back,
                    output_base_dir=output_base_dir
                )
                results.append({
                    'target': target,
                    'horizon': horizon_label,
                    'success': True,
                    'test_mape': result['test_mape'],
                    'val_mape': result['val_mape']
                })
                print(f"    ✅ Success: Test MAPE={result['test_mape']:.3f}%")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results.append({
                    'target': target,
                    'horizon': horizon_label,
                    'success': False,
                    'error': str(e)
                })
    
    # Summary
    print("\n" + "="*80)
    print(f"EPISODE {episode_num} TRAINING SUMMARY")
    print("="*80)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successfully trained: {len(successful)}/{total_models} models")
    if failed:
        print(f"Failed: {len(failed)} models")
        for f in failed:
            print(f"  - {f['target']} {f['horizon']}: {f.get('error', 'Unknown error')}")
    
    if successful:
        avg_test_mape = np.mean([r['test_mape'] for r in successful])
        avg_val_mape = np.mean([r['val_mape'] for r in successful])
        print(f"\nAverage Test MAPE: {avg_test_mape:.3f}%")
        print(f"Average Val MAPE: {avg_val_mape:.3f}%")
    
    print(f"\nModels saved to: {output_base_dir}/episode_{episode_num}/")
    
    return {
        'episode_num': episode_num,
        'results': results,
        'successful': len(successful),
        'failed': len(failed)
    }


def main():
    parser = argparse.ArgumentParser(description='Train episode-specific forecast models')
    parser.add_argument('--episode', type=int, nargs='+', help='Episode number(s) to train (0-19)')
    parser.add_argument('--all', action='store_true', help='Train all episodes (0-19)')
    parser.add_argument('--data_path', type=str, default='training_dataset/trainset.csv',
                       help='Path to training dataset CSV')
    parser.add_argument('--output_dir', type=str, default='forecast_models',
                       help='Base output directory for episode models')
    
    args = parser.parse_args()
    
    # Determine episodes to train
    if args.all:
        episodes = list(range(20))  # Episodes 0-19
    elif args.episode:
        episodes = args.episode
    else:
        parser.error("Must specify --episode or --all")
    
    # Validate episodes
    for ep in episodes:
        if ep < 0 or ep > 19:
            parser.error(f"Episode number must be 0-19, got {ep}")
    
    # Check data path
    if not os.path.exists(args.data_path):
        parser.error(f"Data file not found: {args.data_path}")
    
    print(f"\nTraining forecast models for episodes: {episodes}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Train each episode
    all_results = []
    for episode_num in episodes:
        try:
            result = train_episode_forecasts(
                episode_num=episode_num,
                data_path=args.data_path,
                output_base_dir=args.output_dir
            )
            all_results.append(result)
            print(f"\n✅ Episode {episode_num} complete\n")
        except Exception as e:
            print(f"\n❌ Episode {episode_num} failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("="*80)
    print("ALL EPISODES TRAINING SUMMARY")
    print("="*80)
    for result in all_results:
        ep = result['episode_num']
        success = result['successful']
        total = len(result['results'])
        print(f"Episode {ep:2d}: {success}/{total} models trained successfully")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

