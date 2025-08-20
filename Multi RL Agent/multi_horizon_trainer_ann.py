#!/usr/bin/env python3
"""
Multi-horizon trainer adapted from your proven ANN approach.
Uses your exact architecture and methodology but for multiple horizons.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import json
from math import sqrt
from create_dataset import create_dataset 

class MultiHorizonModelTrainer:
    """
    Multi-horizon trainer using YOUR proven ANN methodology.
    """
    
    def __init__(self, look_back=6, verbose=True):
        self.look_back = look_back
        self.verbose = verbose
        
        # Multi-step horizons for RL system
        self.horizons = {
            "immediate": 1,      # Your current approach - direct next step
            "short": 6,          # 6 steps ahead
            "medium": 24,        # 24 steps ahead  
            "long": 144,         # 144 steps ahead
            "strategic": 1008    # 1008 steps ahead
        }
        
        self.targets = ["wind", "solar", "hydro", "load", "price"]
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        
    def create_dataset_for_horizon(self, dataset, horizon_steps):
        """
        Create dataset for specific horizon using your methodology.
        For horizon_steps=1, this is identical to your current approach.
        For horizon_steps>1, predicts the value at that specific step.
        """
        dataX, dataY = [], []
        
        # Use your exact logic for immediate horizon
        if horizon_steps == 1:
            for i in range(len(dataset) - self.look_back - 1):
                a = dataset[i:(i + self.look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + self.look_back, 0])
        else:
            # For longer horizons, predict the specific future timestep
            for i in range(len(dataset) - self.look_back - horizon_steps):
                a = dataset[i:(i + self.look_back), 0]
                dataX.append(a)
                # Predict the value exactly horizon_steps ahead
                dataY.append(dataset[i + self.look_back + horizon_steps - 1, 0])
                
        return np.array(dataX), np.array(dataY)
    
    def create_ann_model(self):
        """
        Create ANN model using YOUR exact architecture and settings.
        """
        # Set random seed for reproducibility (your approach)
        np.random.seed(1234)
        tf.random.set_seed(1234)
        os.environ['PYTHONHASHSEED'] = '0'
        tf.config.experimental.enable_op_determinism()
        
        # Build ANN model (exactly your architecture)
        neuron = 128  # Your setting
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=neuron, activation='relu', input_shape=(self.look_back,)))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        return model
    
    def calculate_metrics(self, y_true, y_pred, cap):
        """
        Calculate metrics using your exact approach.
        """
        mape = np.mean(np.abs(y_true - y_pred) / cap) * 100
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return mape, rmse, mae
    
    def prepare_data_splits(self, data, target, train_ratio=0.70, val_ratio=0.15):
        """
        Prepare data splits using your proven chronological approach.
        """
        # ✅ FIXED: Handle missing Month column gracefully
        if 'Month' in data.columns:
            months = [1, 2]  # Your original setting
            filtered_data = data.loc[data['Month'].isin(months)].reset_index(drop=True).dropna()
            if self.verbose:
                print(f"📊 Using months {months}, filtered to {len(filtered_data)} samples")
        else:
            if self.verbose:
                print(f"⚠️ No 'Month' column found, using all data")
            filtered_data = data.copy().reset_index(drop=True).dropna()
        
        target_series = filtered_data[target].values.reshape(-1, 1)
        total_samples = len(target_series)
        
        if total_samples == 0:
            raise ValueError(f"No data available for target {target}")
        
        # Chronological split (your approach)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_data = target_series[:train_size]
        val_data = target_series[train_size:train_size + val_size]
        test_data = target_series[train_size + val_size:]
        
        if self.verbose:
            print(f"📊 Total samples: {total_samples:,}")
            print(f"📊 Train: {len(train_data):,} samples ({len(train_data)/total_samples*100:.1f}%)")
            print(f"📊 Validation: {len(val_data):,} samples ({len(val_data)/total_samples*100:.1f}%)")
            print(f"📊 Test: {len(test_data):,} samples ({len(test_data)/total_samples*100:.1f}%)")
        
        return train_data, val_data, test_data, total_samples
    
    def train_single_model(self, data, target, horizon_name, 
                          model_dir="multi_horizon_models", 
                          scaler_dir="multi_horizon_scalers", 
                          epochs=100):
        """
        Train single model using your proven methodology.
        """
        horizon_steps = self.horizons[horizon_name]
        
        if self.verbose:
            print(f"Training {target} model for {horizon_name} horizon ({horizon_steps} steps)...")
        
        # Check if target exists
        if target not in data.columns:
            print(f"Warning: {target} not found in data columns")
            return None
        
        try:
            # Prepare data splits (your approach)
            train_data, val_data, test_data, total_samples = self.prepare_data_splits(data, target)
            
            # Calculate cap (your approach)
            cap = data[target].max()
            
            # Create datasets for each split using horizon-specific method
            trainX, trainY = self.create_dataset_for_horizon(train_data, horizon_steps)
            valX, valY = self.create_dataset_for_horizon(val_data, horizon_steps)
            testX, testY = self.create_dataset_for_horizon(test_data, horizon_steps)
            
            if len(trainX) == 0:
                print(f"Warning: No training data for {target} at horizon {horizon_steps}")
                return None
            
            # Scale data using ONLY training statistics (your approach)
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            
            # Fit scalers only on training data
            X_train = sc_X.fit_transform(trainX)
            y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()
            
            # Transform validation and test data using training statistics
            X_val = sc_X.transform(valX)
            y_val = sc_y.transform(valY.reshape(-1, 1)).ravel()
            X_test = sc_X.transform(testX)
            y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()
            
            # Ensure data is float32 for TensorFlow (your approach)
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.float32)
            X_val = np.array(X_val, dtype=np.float32)
            y_val = np.array(y_val, dtype=np.float32)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)
            
            if self.verbose:
                print(f"📊 Training features: {X_train.shape}")
                print(f"📊 Validation features: {X_val.shape}")
                print(f"📊 Test features: {X_test.shape}")
            
            # Create and train model (your approach)
            model = self.create_ann_model()
            
            # Train with validation monitoring (your approach)
            history = model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val),
                epochs=epochs, 
                batch_size=64,  # Your setting
                verbose=0 if not self.verbose else 1
            )
            
            # Make predictions on all sets (your approach)
            y_pred_train = model.predict(X_train, verbose=0).ravel()
            y_pred_val = model.predict(X_val, verbose=0).ravel()
            y_pred_test = model.predict(X_test, verbose=0).ravel()
            
            # Inverse transform predictions (your approach)
            y_pred_train_inv = sc_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
            y_train_inv = sc_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
            
            y_pred_val_inv = sc_y.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            y_val_inv = sc_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
            
            y_pred_test_inv = sc_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
            y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            
            # Calculate metrics (your approach)
            train_mape, train_rmse, train_mae = self.calculate_metrics(y_train_inv, y_pred_train_inv, cap)
            val_mape, val_rmse, val_mae = self.calculate_metrics(y_val_inv, y_pred_val_inv, cap)
            test_mape, test_rmse, test_mae = self.calculate_metrics(y_test_inv, y_pred_test_inv, cap)
            
            if self.verbose:
                # Display results (your format)
                print(f"\n📊 Performance Results:")
                print("="*50)
                print(f"🎯 Training   - MAPE: {train_mape:.3f}%, RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
                print(f"🔍 Validation - MAPE: {val_mape:.3f}%, RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
                print(f"🚀 Test      - MAPE: {test_mape:.3f}%, RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")
                
                # Check for overfitting (your approach)
                if val_mape > train_mape * 1.5:
                    print("⚠️  Possible overfitting detected (validation >> training performance)")
                elif abs(val_mape - test_mape) < train_mape * 0.1:
                    print("✅ Good generalization (validation ≈ test performance)")
            
            # Save model and scalers
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(scaler_dir, exist_ok=True)
            
            model_key = f"{target}_{horizon_name}"
            model_path = os.path.join(model_dir, f"{model_key}_model.keras")
            scaler_X_path = os.path.join(scaler_dir, f"{model_key}_scaler_X.pkl")
            scaler_y_path = os.path.join(scaler_dir, f"{model_key}_scaler_y.pkl")
            
            model.save(model_path)
            joblib.dump(sc_X, scaler_X_path)
            joblib.dump(sc_y, scaler_y_path)
            
            # Store training history
            self.models[model_key] = model
            self.scalers[model_key] = {'scaler_X': sc_X, 'scaler_y': sc_y}
            self.training_history[model_key] = {
                'history': history.history,
                'train_mape': train_mape, 'train_rmse': train_rmse, 'train_mae': train_mae,
                'val_mape': val_mape, 'val_rmse': val_rmse, 'val_mae': val_mae,
                'test_mape': test_mape, 'test_rmse': test_rmse, 'test_mae': test_mae,
                'horizon_steps': horizon_steps,
                'cap': cap,
                'total_samples': total_samples
            }
            
            if self.verbose:
                print(f"✅ {target}_{horizon_name}: Test MAPE={test_mape:.3f}%")
            
            return model_key
            
        except Exception as e:
            print(f"❌ Error training {target}_{horizon_name}: {e}")
            return None
    
    def train_all_models(self, data, model_dir="multi_horizon_models", 
                        scaler_dir="multi_horizon_scalers", epochs=100):
        """
        Train models for all targets and horizons using your methodology.
        """
        print("🚀 Starting multi-horizon training with YOUR proven ANN architecture...")
        print(f"Targets: {self.targets}")
        print(f"Horizons: {self.horizons}")
        print(f"Data shape: {data.shape}")
        
        # ✅ FIXED: Add Date/Month columns if missing (for compatibility with your approach)
        if 'timestamp' in data.columns and 'Date' not in data.columns:
            data['Date'] = pd.to_datetime(data['timestamp'])
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            print(f"✅ Added Date/Month columns from timestamp")
        elif 'Date' not in data.columns and 'Month' not in data.columns:
            # If no timestamp, create dummy month column (use all data)
            print(f"⚠️ No timestamp/date columns found, creating dummy Month column")
            data['Month'] = 1  # All data treated as month 1
        
        # Add risk column if missing (for compatibility)
        if 'risk' not in data.columns:
            print("📊 Adding synthetic risk column...")
            np.random.seed(42)
            if 'price' in data.columns:
                price_vol = data['price'].rolling(24).std().fillna(0) / data['price'].rolling(24).mean().fillna(1)
            else:
                price_vol = 0
            base_risk = 0.1 + 0.3 * price_vol + np.random.normal(0, 0.05, len(data))
            data['risk'] = np.clip(base_risk, 0, 1)
            self.targets.append('risk')
        
        results = {}
        total_models = len(self.targets) * len(self.horizons)
        current_model = 0
        successful_models = 0
        
        for target in self.targets:
            results[target] = {}
            for horizon_name in self.horizons:
                current_model += 1
                print(f"\n[{current_model}/{total_models}] Training {target}_{horizon_name}...")
                
                model_key = self.train_single_model(
                    data, target, horizon_name, model_dir, scaler_dir, epochs=epochs
                )
                
                if model_key:
                    results[target][horizon_name] = model_key
                    successful_models += 1
                else:
                    results[target][horizon_name] = None
        
        # Save training summary
        self.save_training_summary(model_dir)
        
        print(f"\n✅ Training complete! Successfully trained {successful_models}/{total_models} models")
        print(f"📦 Models saved to {model_dir}")
        return results
    
    def save_training_summary(self, model_dir):
        """Save training summary."""
        summary = {
            'training_time': datetime.now().isoformat(),
            'approach': 'adapted_from_proven_ann',
            'look_back': self.look_back,
            'horizons': self.horizons,
            'targets': self.targets,
            'models_trained': list(self.training_history.keys()),
            'architecture': 'ANN_128_neurons_proven',
            'performance': {}
        }
        
        for model_key, metrics in self.training_history.items():
            summary['performance'][model_key] = {
                'train_mape': metrics['train_mape'],
                'val_mape': metrics['val_mape'],
                'test_mape': metrics['test_mape'],
                'horizon_steps': metrics['horizon_steps']
            }
        
        summary_path = os.path.join(model_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"Training summary saved to {summary_path}")

    def plot_training_results(self, model_dir="multi_horizon_models"):
        """✅ ADDED: Plot training metrics by horizon."""
        if not self.training_history:
            print("No training history available")
            return
        
        try:
            # Organize metrics by horizon
            horizon_metrics = {horizon: {'val_mape': [], 'test_mape': []} 
                              for horizon in self.horizons}
            
            for model_key, metrics in self.training_history.items():
                parts = model_key.split('_')
                if len(parts) >= 2:
                    horizon = '_'.join(parts[1:])  # Handle multi-word horizons
                    if horizon in horizon_metrics:
                        horizon_metrics[horizon]['val_mape'].append(metrics['val_mape'])
                        horizon_metrics[horizon]['test_mape'].append(metrics['test_mape'])
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, metric in enumerate(['val_mape', 'test_mape']):
                horizons = list(self.horizons.keys())
                values = [np.mean(horizon_metrics[h][metric]) if horizon_metrics[h][metric] else 0 
                         for h in horizons]
                
                axes[i].bar(horizons, values)
                axes[i].set_title(f'Average {metric.upper()} by Horizon')
                axes[i].set_ylabel(f'{metric.upper()} (%)')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = os.path.join(model_dir, 'training_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Only show plot if in interactive environment
            try:
                import matplotlib
                if matplotlib.get_backend() != 'Agg':
                    plt.show()
            except:
                pass
            
            plt.close()  # Clean up
            
            if self.verbose:
                print(f"📊 Training plots saved to {plot_path}")
                
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")

# Usage example
if __name__ == "__main__":
    # This would be called from your train_forecast_models.py
    pass