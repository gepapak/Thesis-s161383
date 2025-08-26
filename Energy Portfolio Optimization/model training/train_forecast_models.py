#!/usr/bin/env python3
"""
Step 1: Train Multi-Horizon Forecast Models
Run this BEFORE main_multi_horizon.py
"""

import pandas as pd
import numpy as np
import os
from multi_horizon_trainer_ann import MultiHorizonModelTrainer

def main():
    print("🚀 Step 1: Training Multi-Horizon Forecast Models")
    print("=" * 50)
    
    # Configuration
    data_path = "sample.csv"  # Your energy data
    model_dir = "saved_models"
    scaler_dir = "saved_scalers"
    
    # Training parameters
    look_back = 6  # Use 6 previous timesteps
    epochs = 50   # Reduced for faster testing
    
    # Load your energy data
    print(f"📦 Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path)
        print(f"✅ Data loaded: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # ✅ ADDED: Handle Date/Month columns
        if 'timestamp' in data.columns and 'Date' not in data.columns:
            data['Date'] = pd.to_datetime(data['timestamp'])
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            print(f"✅ Added Date/Month columns from timestamp")
        elif 'Date' not in data.columns and 'Month' not in data.columns:
            # If no timestamp, create dummy month column (use all data)
            print(f"⚠️ No timestamp/date columns found, will use all data")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Verify required columns
    required_cols = ["wind", "solar", "hydro", "load", "price"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"⚠️ Missing required columns: {missing_cols}")
        print("Please ensure your CSV has: wind, solar, hydro, load, price")
        return
    
    # Initialize trainer
    print(f"🔧 Initializing Multi-Horizon Trainer...")
    print(f"   Look-back window: {look_back}")
    print(f"   Training epochs: {epochs}")
    
    trainer = MultiHorizonModelTrainer(look_back=look_back, verbose=True)
    
    # Show what will be trained
    print(f"\n📋 Training Plan:")
    print(f"   Targets: {trainer.targets}")
    print(f"   Horizons: {trainer.horizons}")
    print(f"   Total models: {len(trainer.targets)} × {len(trainer.horizons)} = {len(trainer.targets) * len(trainer.horizons)}")
    
    # Calculate expected training time
    total_models = len(trainer.targets) * len(trainer.horizons)
    estimated_time = total_models * epochs * 0.05  # Reduced estimate
    print(f"   Estimated training time: ~{estimated_time/60:.1f} minutes")
    
    # Ask for confirmation
    response = input(f"\n❓ Proceed with training {total_models} models? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Create output directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"Models will be saved to: {model_dir}")
    print(f"Scalers will be saved to: {scaler_dir}")
    
    try:
        results = trainer.train_all_models(
            data, 
            model_dir=model_dir, 
            scaler_dir=scaler_dir,
            epochs=epochs
        )
        
        print(f"\n✅ Training completed!")
        print(f"📦 Models saved to: {model_dir}")
        print(f"📦 Scalers saved to: {scaler_dir}")
        
        # Show training results
        print(f"\n📊 Training Results Summary:")
        successful_models = 0
        for target, horizons in results.items():
            for horizon, model_key in horizons.items():
                if model_key:
                    successful_models += 1
        
        total_models = len(trainer.targets) * len(trainer.horizons)
        print(f"   Successfully trained: {successful_models}/{total_models} models")
        
        if successful_models > 0:
            # Generate training plots
            print(f"\n📈 Generating training plots...")
            trainer.plot_training_results(model_dir)
            
            # Next steps
            print(f"\n🎯 Next Steps:")
            print(f"   1. ✅ Multi-horizon models trained")
            print(f"   2. ➡️  Run main_multi_horizon.py for RL training")
            print(f"   3. 📊 Models will be automatically loaded from {model_dir}")
            
            print(f"\n💡 Command to run next:")
            print(f"   python main_multi_horizon.py --data_path {data_path}")
        else:
            print(f"\n❌ No models were successfully trained!")
            print(f"   Check the error messages above for details")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    print(f"\n🎉 Step 1 Complete!")

if __name__ == "__main__":
    main()