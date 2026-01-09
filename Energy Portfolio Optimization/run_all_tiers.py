#!/usr/bin/env python3
"""
Run All Tiers Sequentially

This script runs 5 tiers sequentially:
1. Tier 1 (Baseline MARL)
2. Tier 2 (MARL + Forecast Integration)
3. Tier 3 (MARL + Forecast + FAMC)
4. Tier 2 + Risk Uplift
5. Tier 3 + Risk Uplift

Each tier will complete before the next one starts.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_command(cmd, tier_name):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Starting {tier_name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        if process.returncode == 0:
            print(f"\n{'='*80}")
            print(f"✅ {tier_name} completed successfully!")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {hours}h {minutes}m {seconds}s")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"\n{'='*80}")
            print(f"❌ {tier_name} failed with exit code {process.returncode}")
            print(f"Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {hours}h {minutes}m {seconds}s")
            print(f"{'='*80}\n")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  {tier_name} interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n\n❌ Error running {tier_name}: {e}")
        return False

def main():
    """Run all five tiers sequentially."""
    
    # Common arguments
    # NOTE: Using seed 789 for all tiers
    common_args = [
        '--episode_training',
        '--episode_data_dir', 'training_dataset',
        '--start_episode', '0',
        '--end_episode', '19',
        '--cooling_period', '1',
        '--checkpoint_freq', '5000',
        '--seed', '789',  # Seed 789 for all tiers
        '--investment_freq', '48',
        '--enable_gnn_encoder'
    ]
    
    # Get seed value from common_args for directory naming
    seed_idx = common_args.index('--seed') + 1
    seed_value = common_args[seed_idx] if seed_idx < len(common_args) else '789'
    
    # Tier 1 command (Baseline MARL)
    tier1_cmd = ['python', 'main.py'] + common_args + [
        '--save_dir', f'tier1gnn_seed{seed_value}'
    ]
    
    # Tier 2 command (MARL + Forecast Integration)
    tier2_cmd = ['python', 'main.py'] + common_args + [
        '--save_dir', f'tier2gnn_seed{seed_value}',
        '--enable_forecast_utilisation'
    ]
    
    # Tier 3 command (MARL + Forecast + FAMC)
    tier3_cmd = ['python', 'main.py'] + common_args + [
        '--save_dir', f'tier3gnn_seed{seed_value}',
        '--enable_forecast_utilisation',
        '--forecast_baseline_enable',
        '--fgb_mode', 'meta'
    ]
    
    # Tier 2 + Risk Uplift command
    tier2_risk_cmd = ['python', 'main.py'] + common_args + [
        '--save_dir', f'tier2gnn_riskuplift_seed{seed_value}',
        '--enable_forecast_utilisation',
        '--enable_forecast_risk_management'
    ]
    
    # Tier 3 + Risk Uplift command
    tier3_risk_cmd = ['python', 'main.py'] + common_args + [
        '--save_dir', f'tier3gnn_riskuplift_seed{seed_value}',
        '--enable_forecast_utilisation',
        '--forecast_baseline_enable',
        '--fgb_mode', 'meta',
        '--enable_forecast_risk_management'
    ]
    
    print("\n" + "="*80)
    print("MULTI-TIER TRAINING RUN (5 TIERS)")
    print("="*80)
    print(f"Seed: {seed_value}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nTiers to run:")
    print("  1. Tier 1 (Baseline MARL)")
    print("  2. Tier 2 (MARL + Forecast Integration)")
    print("  3. Tier 3 (MARL + Forecast + FAMC)")
    print("  4. Tier 2 + Risk Uplift")
    print("  5. Tier 3 + Risk Uplift")
    print("="*80)
    
    results = {}
    overall_start_time = time.time()
    
    # Run Tier 1
    results['Tier 1'] = run_command(tier1_cmd, 'Tier 1 (Baseline MARL)')
    if not results['Tier 1']:
        print("\n❌ Tier 1 failed. Stopping execution.")
        sys.exit(1)
    
    # Run Tier 2
    results['Tier 2'] = run_command(tier2_cmd, 'Tier 2 (MARL + Forecast Integration)')
    if not results['Tier 2']:
        print("\n❌ Tier 2 failed. Stopping execution.")
        sys.exit(1)
    
    # Run Tier 3
    results['Tier 3'] = run_command(tier3_cmd, 'Tier 3 (MARL + Forecast + FAMC)')
    if not results['Tier 3']:
        print("\n❌ Tier 3 failed. Stopping execution.")
        sys.exit(1)
    
    # Run Tier 2 + Risk Uplift
    results['Tier 2 + Risk Uplift'] = run_command(tier2_risk_cmd, 'Tier 2 + Risk Uplift (MARL + Forecast + Risk Management)')
    if not results['Tier 2 + Risk Uplift']:
        print("\n❌ Tier 2 + Risk Uplift failed. Stopping execution.")
        sys.exit(1)
    
    # Run Tier 3 + Risk Uplift
    results['Tier 3 + Risk Uplift'] = run_command(tier3_risk_cmd, 'Tier 3 + Risk Uplift (MARL + Forecast + FAMC + Risk Management)')
    if not results['Tier 3 + Risk Uplift']:
        print("\n❌ Tier 3 + Risk Uplift failed. Stopping execution.")
        sys.exit(1)
    
    # Summary
    overall_elapsed = time.time() - overall_start_time
    hours = int(overall_elapsed // 3600)
    minutes = int((overall_elapsed % 3600) // 60)
    seconds = int(overall_elapsed % 60)
    
    print("\n" + "="*80)
    print("TRAINING RUN COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {hours}h {minutes}m {seconds}s")
    print("\nResults:")
    for tier, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {tier}: {status}")
    print("="*80 + "\n")
    
    if all(results.values()):
        print("🎉 All 5 tiers completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some tiers failed. Check the output above for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()

