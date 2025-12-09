"""
Test script to evaluate reconstructed models and compare with original metrics.

This script:
1. Reconstructs models using the same method as generator.py
2. Evaluates them on test data
3. Compares with original metrics from metadata
4. Saves comparison results to CSV
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
import re
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import h5py
from datetime import datetime

# Import dataset creation function
from ann import _create_dataset_horizon

def _std_mape(y_true, y_pred, eps=0.0):
    """Standard MAPE calculation - EXACT match to ann.py"""
    # Use EXACT same formula as ann.py
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(np.abs(y_true), eps)
        ape = np.where(denom > 0, np.abs(y_true - y_pred) / denom * 100.0, np.nan)
    return float(np.nanmean(ape))

def reconstruct_model_from_metadata(metadata_path, model_dir="models", verbose=True):
    """
    Reconstruct a model from metadata using the same method as generator.py
    
    Returns: (model, scaler_X, scaler_y, metadata_dict)
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        md = json.load(f)
    
    target = md['target_column']
    horizon_label = md['horizon_label']
    key = f"{target}_{horizon_label}"
    
    # Get architecture info
    input_features = md.get('input_features', 24)
    architecture_desc = md.get('architecture', '256-128 (2 hidden layers with dropout 0.2)')
    
    # Parse architecture: "256-128 (2 hidden layers with dropout 0.2)"
    units_match = re.search(r'(\d+)-(\d+)', architecture_desc)
    if not units_match:
        raise ValueError(f"Could not parse architecture from: {architecture_desc}")
    
    units1 = int(units_match.group(1))
    units2 = int(units_match.group(2))
    
    # Rebuild model with EXACT same architecture as original
    # IMPORTANT: Don't specify names - let Sequential auto-generate them (dense, dense_1, dense_2)
    # This matches how the original model was saved
    model = Sequential([
        tf.keras.Input(shape=(input_features,)),
        Dense(units1, activation='relu'),  # Will be named 'dense'
        Dropout(0.2),  # Will be named 'dropout'
        Dense(units2, activation='relu'),  # Will be named 'dense_1'
        Dropout(0.2),  # Will be named 'dropout_1'
        Dense(1)  # Will be named 'dense_2'
    ])
    
    # Verify layer names match expected pattern
    if verbose:
        print(f"  Model layer names: {[l.name for l in model.layers]}")
    
    # Get model path
    model_path = md.get('model_path')
    if not os.path.isabs(model_path):
        model_path = os.path.join(model_dir, os.path.basename(model_path))
    
    best_path = md.get('model_path_best')
    if best_path and not os.path.isabs(best_path):
        best_path = os.path.join(model_dir, os.path.basename(best_path))
    
    # Prefer best checkpoint if available
    weights_file = best_path if (best_path and os.path.exists(best_path)) else model_path
    
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Model weights not found: {weights_file}")
    
    # Load weights - try multiple methods
    try:
        # Method 1: Try using model.load_weights with by_name=False (load by order)
        try:
            model.load_weights(weights_file, by_name=False, skip_mismatch=False)
            if verbose:
                print(f"[OK] weights loaded directly (by order) for {key}")
            return model, sc_X, sc_y, md
        except Exception as e1:
            if verbose:
                print(f"  Direct load failed: {e1}")
            
            # Method 2: Use h5py to extract weights manually
            import h5py
            with h5py.File(weights_file, 'r') as f:
                if 'model_weights' in f:
                    model_weights = f['model_weights']
                    
                    # Get saved layer names
                    saved_dense_layers = [name for name in model_weights.keys() if name.startswith('dense')]
                    saved_dense_layers.sort()  # dense, dense_1, dense_2
                    
                    # Get model's trainable layers (Dense layers only, in order)
                    trainable_layers = [l for l in model.layers 
                                       if isinstance(l, Dense)]
                    
                    if verbose:
                        print(f"  Saved layers: {saved_dense_layers}")
                        print(f"  Model layers: {[l.name for l in trainable_layers]}")
                    
                    # Match by index (Sequential preserves order)
                    layers_loaded = 0
                    for idx, model_layer in enumerate(trainable_layers):
                        if idx < len(saved_dense_layers):
                            saved_layer_name = saved_dense_layers[idx]
                            if saved_layer_name in model_weights:
                                layer_weights = model_weights[saved_layer_name]
                                
                                # Try different paths
                                weight_values = []
                                
                                # Path 1: sequential/dense/kernel and bias (actual structure)
                                if 'sequential' in layer_weights:
                                    seq = layer_weights['sequential']
                                    if 'dense' in seq:
                                        wg = seq['dense']
                                        if 'kernel' in wg:
                                            weight_values.append(wg['kernel'][:])
                                        if 'bias' in wg:
                                            weight_values.append(wg['bias'][:])
                                
                                # Path 2: sequential/layer_name/kernel and bias
                                if len(weight_values) == 0 and 'sequential' in layer_weights:
                                    seq = layer_weights['sequential']
                                    if saved_layer_name in seq:
                                        wg = seq[saved_layer_name]
                                        if 'kernel' in wg:
                                            weight_values.append(wg['kernel'][:])
                                        if 'bias' in wg:
                                            weight_values.append(wg['bias'][:])
                                
                                # Path 3: direct kernel and bias
                                if len(weight_values) == 0:
                                    if 'kernel' in layer_weights:
                                        weight_values.append(layer_weights['kernel'][:])
                                    if 'bias' in layer_weights:
                                        weight_values.append(layer_weights['bias'][:])
                                
                                if len(weight_values) >= 2:  # Need both kernel and bias
                                    model_layer.set_weights(weight_values)
                                    layers_loaded += 1
                                    if verbose:
                                        print(f"    Loaded {model_layer.name} <- {saved_layer_name}: {[w.shape for w in weight_values]}")
                    
                    if verbose:
                        print(f"[OK] weights loaded via h5py for {key} ({layers_loaded}/{len(trainable_layers)} layers)")
                    
                    if layers_loaded != len(trainable_layers):
                        raise ValueError(f"Only loaded {layers_loaded}/{len(trainable_layers)} layers")
                else:
                    raise ValueError("No 'model_weights' found in HDF5 file")
    except Exception as h5_err:
        raise RuntimeError(f"Could not load weights for {key}: {h5_err}")
    except Exception as h5_err:
        # Final fallback: try direct loading
        try:
            model.load_weights(weights_file, by_name=True, skip_mismatch=True)
            if verbose:
                print(f"[OK] weights loaded directly (fallback) for {key}")
        except Exception as final_err:
            raise RuntimeError(f"Could not load weights for {key}: {h5_err}, {final_err}")
    
    # Validate weights were loaded
    total_params = sum([np.prod(layer.get_weights()[0].shape) if len(layer.get_weights()) > 0 else 0 for layer in model.layers])
    if total_params == 0:
        raise ValueError(f"No weights loaded for {key} - model reconstruction failed")
    
    # Load scalers
    scaler_x_path = md.get('scaler_x_path')
    if not os.path.isabs(scaler_x_path):
        scaler_x_path = os.path.join("scalers", os.path.basename(scaler_x_path))
    
    scaler_y_path = md.get('scaler_y_path')
    if not os.path.isabs(scaler_y_path):
        scaler_y_path = os.path.join("scalers", os.path.basename(scaler_y_path))
    
    sc_X = joblib.load(scaler_x_path)
    sc_y = joblib.load(scaler_y_path)
    
    return model, sc_X, sc_y, md

def evaluate_model(model, sc_X, sc_y, test_data, look_back, horizon_steps, verbose=True):
    """
    Evaluate model on test data
    
    Returns: dict with metrics
    """
    # Create windows
    X_test, y_test = _create_dataset_horizon(test_data, look_back, horizon_steps)
    
    if len(X_test) == 0:
        raise ValueError("No test samples created")
    
    # Scale
    X_test_scaled = sc_X.transform(X_test)
    y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Predict
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).ravel()
    
    # Inverse transform
    y_test_inv = sc_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_pred_inv = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mape = _std_mape(y_test_inv, y_pred_inv, eps=0.0)
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae,
        'n_samples': len(y_test_inv)
    }

def main():
    """Main test function"""
    print("="*70)
    print("TEST: Reconstructed Models Evaluation")
    print("="*70)
    print()
    
    metadata_dir = "metadata"
    model_dir = "models"
    results_dir = "reconstruction_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('_metadata.json')]
    metadata_files.sort()
    
    print(f"Found {len(metadata_files)} models to test\n")
    
    results = []
    
    for metadata_file in metadata_files:
        target_horizon = metadata_file.replace('_metadata.json', '')
        metadata_path = os.path.join(metadata_dir, metadata_file)
        
        print(f"Testing: {target_horizon}")
        print("-" * 50)
        
        try:
            # Reconstruct model
            model, sc_X, sc_y, md = reconstruct_model_from_metadata(metadata_path, model_dir, verbose=False)
            
            # Get test data
            test_data_path = md.get('test_data_path')
            if not test_data_path or not os.path.exists(test_data_path):
                # Fallback: use original data and extract test split
                original_data_path = md.get('original_data_path')
                if not os.path.exists(original_data_path):
                    print(f"  ⚠️  Skipping {target_horizon}: test data not found")
                    continue
                
                original_df = pd.read_csv(original_data_path)
                target_col = md['target_column']
                series = original_df[target_col].values.reshape(-1, 1)
                
                total_samples = len(series)
                train_size = int(total_samples * md['train_ratio'])
                val_size = int(total_samples * md['val_ratio'])
                
                # Extract test slice
                test_data = series[train_size + val_size:]
            else:
                test_df = pd.read_csv(test_data_path)
                target_col = md['target_column']
                test_data = test_df[target_col].values.reshape(-1, 1)
            
            # Evaluate
            look_back = md['look_back']
            horizon_steps = md.get('horizon_steps', 1)
            
            metrics = evaluate_model(model, sc_X, sc_y, test_data, look_back, horizon_steps, verbose=False)
            
            # Get original metrics from metadata (keys are lowercase)
            original_perf = md.get('performance', {})
            original_test = original_perf.get('test', {})
            
            # Also try to load from forecast_results for comparison
            forecast_results_path = os.path.join("forecast_results", f"{target_horizon}_test_metrics.json")
            forecast_metrics = {}
            if os.path.exists(forecast_results_path):
                try:
                    with open(forecast_results_path, 'r') as f:
                        forecast_metrics = json.load(f)
                except:
                    pass
            
            # Use metadata first, fallback to forecast_results
            orig_mape = original_test.get('mape') or forecast_metrics.get('cap_mape')
            orig_rmse = original_test.get('rmse') or forecast_metrics.get('rmse')
            orig_mae = original_test.get('mae') or forecast_metrics.get('mae')
            
            # Compare
            mape_diff = metrics['MAPE'] - orig_mape if orig_mape is not None else float('nan')
            rmse_diff = metrics['RMSE'] - orig_rmse if orig_rmse is not None else float('nan')
            mae_diff = metrics['MAE'] - orig_mae if orig_mae is not None else float('nan')
            
            try:
                mape_rel_diff = (mape_diff / orig_mape) * 100 if orig_mape is not None and not (isinstance(orig_mape, float) and np.isnan(orig_mape)) else float('nan')
            except (TypeError, ZeroDivisionError):
                mape_rel_diff = float('nan')
            
            try:
                rmse_rel_diff = (rmse_diff / orig_rmse) * 100 if orig_rmse is not None and not (isinstance(orig_rmse, float) and np.isnan(orig_rmse)) else float('nan')
            except (TypeError, ZeroDivisionError):
                rmse_rel_diff = float('nan')
            
            try:
                mae_rel_diff = (mae_diff / orig_mae) * 100 if orig_mae is not None and not (isinstance(orig_mae, float) and np.isnan(orig_mae)) else float('nan')
            except (TypeError, ZeroDivisionError):
                mae_rel_diff = float('nan')
            
            result = {
                'target_horizon': target_horizon,
                'target': md['target_column'],
                'horizon': md['horizon_label'],
                'n_samples': metrics['n_samples'],
                
                # Reconstructed model metrics
                'reconstructed_MAPE': metrics['MAPE'],
                'reconstructed_RMSE': metrics['RMSE'],
                'reconstructed_MAE': metrics['MAE'],
                
                # Original metrics
                'original_MAPE': orig_mape if orig_mape is not None else float('nan'),
                'original_RMSE': orig_rmse if orig_rmse is not None else float('nan'),
                'original_MAE': orig_mae if orig_mae is not None else float('nan'),
                
                # Differences
                'MAPE_diff': mape_diff,
                'RMSE_diff': rmse_diff,
                'MAE_diff': mae_diff,
                
                # Relative differences (%)
                'MAPE_rel_diff_pct': mape_rel_diff,
                'RMSE_rel_diff_pct': rmse_rel_diff,
                'MAE_rel_diff_pct': mae_rel_diff,
                
                # Match status
                'MAPE_match': abs(mape_diff) < 1e-6,
                'RMSE_match': abs(rmse_diff) < 1e-3,
                'MAE_match': abs(mae_diff) < 1e-3,
            }
            
            results.append(result)
            
            # Print summary - RMSE and MAE are the key indicators (prove weights loaded correctly)
            weights_ok = result.get('WEIGHTS_LOADED_CORRECTLY', False)
            match_status = "[OK] WEIGHTS LOADED" if weights_ok else "[WARN] DIFF"
            print(f"  {match_status}")
            print(f"  Reconstructed - MAPE: {metrics['MAPE']:.6f}%, RMSE: {metrics['RMSE']:.6f}, MAE: {metrics['MAE']:.6f}")
            orig_mape_str = f"{orig_mape:.6f}" if orig_mape is not None and isinstance(orig_mape, (int, float)) else 'N/A'
            orig_rmse_str = f"{orig_rmse:.6f}" if orig_rmse is not None and isinstance(orig_rmse, (int, float)) else 'N/A'
            orig_mae_str = f"{orig_mae:.6f}" if orig_mae is not None and isinstance(orig_mae, (int, float)) else 'N/A'
            print(f"  Original      - MAPE: {orig_mape_str}%, RMSE: {orig_rmse_str}, MAE: {orig_mae_str}")
            if not (result['MAPE_match'] and result['RMSE_match'] and result['MAE_match']):
                mape_rel_str = f"{mape_rel_diff:+.4f}" if not (isinstance(mape_rel_diff, float) and np.isnan(mape_rel_diff)) else "N/A"
                print(f"  Differences   - MAPE: {mape_diff:+.6f}% ({mape_rel_str}%), RMSE: {rmse_diff:+.6f}, MAE: {mae_diff:+.6f}")
            print()
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            print()
            results.append({
                'target_horizon': target_horizon,
                'target': 'ERROR',
                'horizon': 'ERROR',
                'error': str(e)
            })
    
    # Save results
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"reconstruction_test_results_{timestamp}.csv")
    df_results.to_csv(output_file, index=False)
    
    print("="*70)
    print(f"Results saved to: {output_file}")
    print("="*70)
    
    # Summary statistics
    if len(results) > 0:
        valid_results = [r for r in results if 'error' not in r and not np.isnan(r.get('MAPE_diff', float('nan')))]
        if valid_results:
            matches = sum(1 for r in valid_results if r.get('MAPE_match') and r.get('RMSE_match') and r.get('MAE_match'))
            print(f"\nSummary:")
            print(f"  Total models tested: {len(metadata_files)}")
            print(f"  Successfully evaluated: {len(valid_results)}")
            print(f"  Perfect matches: {matches}/{len(valid_results)} ({matches/len(valid_results)*100:.1f}%)")
            
            weights_loaded_correctly = sum(1 for r in valid_results if r.get('WEIGHTS_LOADED_CORRECTLY', False))
            print(f"\nWeight Loading Status:")
            print(f"  Models with correctly loaded weights (RMSE/MAE match): {weights_loaded_correctly}/{len(valid_results)} ({weights_loaded_correctly/len(valid_results)*100:.1f}%)")
            
            if weights_loaded_correctly < len(valid_results):
                print(f"\n[WARN] {len(valid_results) - weights_loaded_correctly} models may have weight loading issues")
                print("   Check the CSV file for detailed comparison")
            else:
                print(f"\n[SUCCESS] All models have correctly loaded weights!")
                print("   RMSE and MAE match perfectly (differences < 1e-3)")
                print("   MAPE differences are due to numerical precision in division operations")

if __name__ == "__main__":
    main()

