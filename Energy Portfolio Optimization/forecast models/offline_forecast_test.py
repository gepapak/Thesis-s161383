import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_model_and_scalers():
    """Load the trained model and scalers"""
    print("Loading model and scalers...")

    # Load the model (compile=False to avoid loss function issues)
    model = load_model('saved_models/wind_immediate_model.h5', compile=False)

    # Try loading scalers with different methods
    scaler_X = None
    scaler_y = None

    # Try joblib first
    try:
        scaler_X = joblib.load('saved_scalers/wind_immediate_sc_X.pkl')
        scaler_y = joblib.load('saved_scalers/wind_immediate_sc_y.pkl')
        print("Scalers loaded successfully with joblib!")
    except Exception as e:
        print(f"Joblib loading failed: {e}")

        # Try pickle with different protocols
        try:
            with open('saved_scalers/wind_immediate_sc_X.pkl', 'rb') as f:
                scaler_X = pickle.load(f)
            with open('saved_scalers/wind_immediate_sc_y.pkl', 'rb') as f:
                scaler_y = pickle.load(f)
            print("Scalers loaded successfully with pickle!")
        except Exception as e2:
            print(f"Pickle loading also failed: {e2}")
            print("Will proceed without scalers (using raw data)")
            return model, None, None

    return model, scaler_X, scaler_y

def create_dataset_horizon(dataset, look_back, horizon_steps=1):
    """Create windowed samples that predict exactly ``horizon_steps`` ahead."""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")

    if horizon_steps == 1:
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i + look_back, 0])
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    dataX, dataY = [], []
    N = len(dataset)
    end = N - look_back - horizon_steps + 1
    if end <= 0:
        return np.empty((0, look_back)), np.empty((0,))

    for i in range(end):
        dataX.append(dataset[i:i + look_back, 0])
        dataY.append(dataset[i + look_back + horizon_steps - 1, 0])
    return np.array(dataX), np.array(dataY)

def prepare_features(data, look_back=6, horizon_steps=1):
    """
    Prepare input features for the model using the correct univariate time series approach.
    The model expects sequences of previous wind values, not multivariate features.
    """
    # Extract wind values as a time series
    wind_series = data['wind'].values.reshape(-1, 1)

    print(f"Wind series shape: {wind_series.shape}")
    print(f"Wind series range: {wind_series.min():.2f} to {wind_series.max():.2f}")

    # Create windowed features using the same function as training
    X, y_actual = create_dataset_horizon(wind_series, look_back, horizon_steps)

    print(f"Windowed X shape: {X.shape}")
    print(f"Windowed y shape: {y_actual.shape}")

    return X, y_actual

def generate_forecasts(model, scaler_X, scaler_y, data, look_back=6, horizon_steps=1):
    """Generate forecasts for the given data using the correct time series approach"""
    print("Preparing features...")

    # Prepare input features using the correct univariate time series approach
    X, y_actual = prepare_features(data, look_back, horizon_steps)

    print(f"Feature matrix shape: {X.shape}")

    # Scale the features if scaler is available
    if scaler_X is not None:
        print("Scaling features...")
        X_scaled = scaler_X.transform(X)
    else:
        print("No X scaler available, using raw features")
        # Normalize features manually (simple min-max scaling)
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    print("Generating forecasts...")
    # Generate predictions
    y_pred_scaled = model.predict(X_scaled)

    # Inverse transform to get actual values if scaler is available
    if scaler_y is not None:
        print("Inverse transforming predictions...")
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        print("No y scaler available, using raw predictions")
        # If no scaler, assume predictions are already in correct scale
        y_pred = y_pred_scaled

    return y_pred.flatten(), y_actual.flatten()

def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def main():
    """Main function to run offline forecast testing"""
    print("Starting offline forecast testing...")

    # Load training data
    print("Loading training data...")
    try:
        data = pd.read_csv('trainingdata.csv')
        print(f"Data loaded successfully! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")

        # Print data statistics
        print(f"\nWind data statistics:")
        print(f"Min: {data['wind'].min():.2f}")
        print(f"Max: {data['wind'].max():.2f}")
        print(f"Mean: {data['wind'].mean():.2f}")
        print(f"Std: {data['wind'].std():.2f}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load model and scalers
    try:
        model, scaler_X, scaler_y = load_model_and_scalers()

        # Print scaler information if available
        if scaler_X is not None:
            print(f"\nX Scaler info:")
            print(f"Feature means: {scaler_X.mean_}")
            print(f"Feature scales: {scaler_X.scale_}")

        if scaler_y is not None:
            print(f"\ny Scaler info:")
            print(f"Target mean: {scaler_y.mean_}")
            print(f"Target scale: {scaler_y.scale_}")

    except Exception as e:
        print(f"Error loading model/scalers: {e}")
        return

    # Generate forecasts
    try:
        forecasts, actual_wind = generate_forecasts(model, scaler_X, scaler_y, data, look_back=6, horizon_steps=1)
        print(f"Generated {len(forecasts)} forecasts")

        # Print forecast statistics
        print(f"\nForecast statistics:")
        print(f"Min: {forecasts.min():.2f}")
        print(f"Max: {forecasts.max():.2f}")
        print(f"Mean: {forecasts.mean():.2f}")
        print(f"Std: {forecasts.std():.2f}")

        print(f"\nActual wind statistics (windowed):")
        print(f"Min: {actual_wind.min():.2f}")
        print(f"Max: {actual_wind.max():.2f}")
        print(f"Mean: {actual_wind.mean():.2f}")
        print(f"Std: {actual_wind.std():.2f}")

    except Exception as e:
        print(f"Error generating forecasts: {e}")
        return

    # Calculate metrics (handle division by zero)
    try:
        metrics = calculate_metrics(actual_wind, forecasts)
    except:
        # Calculate metrics manually with better error handling
        mae = mean_absolute_error(actual_wind, forecasts)
        mse = mean_squared_error(actual_wind, forecasts)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_wind, forecasts)

        # Calculate MAPE with zero handling
        non_zero_mask = actual_wind != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual_wind[non_zero_mask] - forecasts[non_zero_mask]) / actual_wind[non_zero_mask])) * 100
        else:
            mape = float('inf')

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

    print("\n" + "="*50)
    print("FORECAST ACCURACY METRICS")
    print("="*50)
    for metric, value in metrics.items():
        if np.isfinite(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Create results dataframe
    # Handle percentage error calculation with zero division
    percentage_errors = []
    for actual, forecast in zip(actual_wind, forecasts):
        if actual != 0:
            pe = np.abs((actual - forecast) / actual) * 100
        else:
            pe = float('inf') if forecast != 0 else 0
        percentage_errors.append(pe)

    # Create timestamps for windowed data
    # The first forecast corresponds to index look_back (6) in the original data
    look_back = 6
    start_idx = look_back
    end_idx = start_idx + len(forecasts)

    # Ensure we don't go beyond the data length
    if end_idx > len(data):
        end_idx = len(data)
        forecasts = forecasts[:end_idx - start_idx]
        actual_wind = actual_wind[:end_idx - start_idx]
        percentage_errors = percentage_errors[:end_idx - start_idx]

    results_df = pd.DataFrame({
        'timestamp': data['timestamp'].iloc[start_idx:end_idx],
        'actual_wind': actual_wind,
        'forecast_wind': forecasts,
        'absolute_error': np.abs(actual_wind - forecasts),
        'percentage_error': percentage_errors
    })

    # Add additional columns from original data for context (aligned with windowed indices)
    results_df['solar'] = data['solar'].iloc[start_idx:end_idx].values
    results_df['hydro'] = data['hydro'].iloc[start_idx:end_idx].values
    results_df['load'] = data['load'].iloc[start_idx:end_idx].values
    results_df['price'] = data['price'].iloc[start_idx:end_idx].values

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"forecast_results_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to: {output_filename}")

    # Create a summary statistics file
    summary_stats = {
        'Total_Samples': len(forecasts),
        'Mean_Actual_Wind': np.mean(actual_wind),
        'Mean_Forecast_Wind': np.mean(forecasts),
        'Std_Actual_Wind': np.std(actual_wind),
        'Std_Forecast_Wind': np.std(forecasts),
        **metrics
    }

    summary_df = pd.DataFrame([summary_stats])
    summary_filename = f"forecast_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to: {summary_filename}")

    # Display sample results
    print("\n" + "="*50)
    print("SAMPLE FORECAST RESULTS (First 10 rows)")
    print("="*50)
    sample_results = results_df[['timestamp', 'actual_wind', 'forecast_wind', 'absolute_error']].head(10)
    print(sample_results.to_string(index=False))

    print(f"\nScript completed successfully!")
    print(f"Check the generated files:")
    print(f"- {output_filename} (detailed results)")
    print(f"- {summary_filename} (summary statistics)")

    # Warning about negative forecasts
    if forecasts.min() < 0:
        print(f"\n⚠️  WARNING: Model is producing negative forecasts!")
        print(f"This suggests a scaling or training issue.")
        print(f"You may need to retrain the model or check the scaler files.")

if __name__ == "__main__":
    main()
