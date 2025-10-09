#!/usr/bin/env python3
"""
Buy-and-Hold Baseline Runner (Baseline3)

This script runs the Buy-and-Hold baseline strategy, which represents the simplest
possible investment approach: earning the risk-free rate through government bonds
or cash equivalents.

This baseline serves as the "floor" for investment performance evaluation.
Any active strategy should be able to beat this risk-free rate to justify
the additional complexity and risk.

Usage:
    python run_buy_and_hold_baseline.py --data_path evaluation_dataset/unseendata.csv
    python run_buy_and_hold_baseline.py --data_path data.csv --output_dir results --timesteps 10000
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server compatibility
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from buy_and_hold_optimizer import BuyAndHoldOptimizer


class BuyAndHoldBaselineRunner:
    """
    Buy-and-Hold Baseline Runner
    
    This class orchestrates the execution of the buy-and-hold strategy,
    providing comprehensive evaluation and reporting capabilities.
    """
    
    def __init__(self, data_path: str, output_dir: str = "Baseline3_BuyAndHold/results", timebase_hours: float = None):
        """
        Initialize runner.
        
        Args:
            data_path: Path to evaluation dataset CSV
            output_dir: Output directory for results
            timebase_hours: Hours per timestep (None=auto-detect, 1.0 for hourly, 0.1667 for 10-min)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.data = self.load_data()
        
        # Auto-detect timebase if not specified
        if timebase_hours is None:
            timebase_hours = self._detect_timebase()
        
        self.timebase_hours = timebase_hours
        
        # Initialize optimizer
        initial_budget_usd = 8e8  # $800M USD
        self.optimizer = BuyAndHoldOptimizer(
            initial_budget_usd=initial_budget_usd,
            timebase_hours=timebase_hours
        )
        
        # Results storage
        self.results = []
    
    def load_data(self):
        """Load and prepare market data."""
        print(f"Loading data from: {self.data_path}")
        data = pd.read_csv(self.data_path)
        
        # Note: Buy-and-hold doesn't need specific columns since it's market-agnostic
        # But we'll check for basic structure
        print(f"Data loaded: {len(data):,} timesteps")
        print(f"Columns: {list(data.columns)}")
        return data
    
    def _detect_timebase(self):
        """Auto-detect timebase from timestamp column if available."""
        if 'timestamp' not in self.data.columns:
            print("Warning: No timestamp column found, defaulting to 10-minute timebase (0.1667 hours)")
            return 0.1667
        
        try:
            # Parse timestamps
            timestamps = pd.to_datetime(self.data['timestamp'])
            
            # Calculate median time delta in minutes
            time_deltas = timestamps.diff().dropna()
            median_delta_minutes = time_deltas.median().total_seconds() / 60
            
            # Convert to hours
            timebase_hours = median_delta_minutes / 60
            
            print(f"Auto-detected timebase: {timebase_hours:.4f} hours ({median_delta_minutes:.1f} minutes)")
            return timebase_hours
        except Exception as e:
            print(f"Warning: Could not auto-detect timebase ({e}), defaulting to 10-minute (0.1667 hours)")
            return 0.1667
    
    def run_optimization(self, max_timesteps: int = None):
        """
        Run the buy-and-hold strategy.
        
        Args:
            max_timesteps: Maximum number of timesteps to run (None = all data)
        """
        print("\n" + "="*70)
        print("Baseline 3: Buy-and-Hold Strategy (Risk-Free Rate)")
        print("="*70)
        
        # Determine timesteps to run
        total_timesteps = len(self.data)
        if max_timesteps is not None:
            total_timesteps = min(max_timesteps, total_timesteps)
        
        print(f"Running buy-and-hold strategy for {total_timesteps:,} timesteps...")
        print(f"Timebase: {self.timebase_hours} hours per step")
        print(f"Initial budget: ${self.optimizer.initial_budget_usd/1e6:.1f}M USD")
        
        # Progress tracking
        progress_interval = max(1, total_timesteps // 20)  # 20 progress updates
        start_time = datetime.now()
        
        # Run strategy (very simple - just compound interest)
        for i in range(total_timesteps):
            # Get data row (not actually used by buy-and-hold)
            data_row = self.data.iloc[i] if i < len(self.data) else pd.Series()
            
            # Execute strategy step
            result = self.optimizer.step(data_row)
            self.results.append(result)
            
            # Progress reporting
            if i % progress_interval == 0 or i == total_timesteps - 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (i + 1) / total_timesteps * 100
                
                print(f"Step {i+1:6,}/{total_timesteps:,} ({progress:5.1f}%) | "
                      f"Portfolio: ${result['portfolio_value_usd']/1e6:6.1f}M | "
                      f"Return: {result['total_return']*100:+6.2f}% | "
                      f"Cash: ${result['cash_usd']/1e6:6.1f}M | "
                      f"Elapsed: {elapsed:.0f}s")
        
        elapsed_total = (datetime.now() - start_time).total_seconds()
        speed = total_timesteps / elapsed_total if elapsed_total > 0 else 0
        
        print(f"\nOptimization completed in {elapsed_total:.1f} seconds")
        print(f"Average speed: {speed:.1f} steps/second")
    
    def save_results(self):
        """Save all results to files."""
        print(f"\nSaving results to: {self.output_dir}")
        
        # 1. Detailed results CSV
        detailed_path = os.path.join(self.output_dir, "detailed_results.csv")
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")
        
        # 2. Summary metrics JSON
        metrics = self.optimizer.get_performance_metrics()
        
        summary_metrics_path = os.path.join(self.output_dir, "summary_metrics.json")
        with open(summary_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Summary metrics saved to: {summary_metrics_path}")
        
        # 3. Summary JSON (for compatibility)
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Summary also saved to: {summary_path}")
        
        # 4. Performance report
        self.generate_report()
        
        return metrics
    
    def generate_report(self):
        """Generate performance visualization report."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Baseline 3: Buy-and-Hold Strategy Performance Report', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            timesteps = [r['timestep'] for r in self.results]
            portfolio_values = [r['portfolio_value_usd'] / 1e6 for r in self.results]  # Convert to millions
            returns = [r['total_return'] * 100 for r in self.results]  # Convert to percentage
            
            # 1. Portfolio Value Over Time
            ax1.plot(timesteps, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Portfolio Value (Million USD)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Cumulative Returns
            ax2.plot(timesteps, returns, 'g-', linewidth=2, label='Cumulative Return')
            ax2.set_title('Cumulative Returns Over Time')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. Risk-Free Rate Visualization
            days = [t * self.timebase_hours / 24 for t in timesteps]
            theoretical_return = [(1 + 0.02) ** (d / 365.25) - 1 for d in days]
            theoretical_return_pct = [r * 100 for r in theoretical_return]
            
            ax3.plot(days, returns, 'g-', linewidth=2, label='Actual Return')
            ax3.plot(days, theoretical_return_pct, 'r--', linewidth=2, label='Theoretical 2% Annual')
            ax3.set_title('Risk-Free Rate Performance')
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Return (%)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. Strategy Summary (text)
            ax4.axis('off')
            metrics = self.optimizer.get_performance_metrics()
            
            summary_text = f"""
Buy-and-Hold Strategy Summary

Strategy Type: Risk-Free Rate Baseline
Market Exposure: 0% (Pure cash/bonds)
Active Decisions: 0 (Passive strategy)

Performance Metrics:
• Total Return: {metrics['total_return']*100:.2f}%
• Annual Return: {metrics['annual_return']*100:.2f}%
• Volatility: {metrics['volatility']*100:.4f}%
• Max Drawdown: {metrics['max_drawdown']*100:.2f}%
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

Portfolio Composition:
• Cash/Bonds: ${metrics['final_cash_usd']/1e6:.1f}M USD (100%)
• Physical Assets: $0.0M USD (0%)
• Market Exposure: None

Risk Profile:
• Risk Level: Zero (Government bonds)
• Volatility: Minimal (Risk-free rate)
• Drawdown Risk: None (Capital preservation)

This baseline represents the minimum return
any active strategy should achieve to justify
additional complexity and risk.
            """
            
            ax4.text(0.05, 0.95, summary_text.strip(), transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            # Save plot
            plt.tight_layout()
            report_path = os.path.join(self.output_dir, "performance_report.png")
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance report saved to: {report_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate performance report: {e}")
    
    def get_final_metrics(self):
        """Get final performance metrics."""
        return self.optimizer.get_performance_metrics()


def main():
    parser = argparse.ArgumentParser(description='Baseline 3: Buy-and-Hold Strategy (Risk-Free Rate)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation dataset CSV')
    parser.add_argument('--timesteps', type=int, default=None, help='Maximum timesteps to run')
    parser.add_argument('--output_dir', type=str, default='Baseline3_BuyAndHold/results', 
                       help='Output directory')
    parser.add_argument('--timebase_hours', type=float, default=None, 
                       help='Hours per timestep (None=auto-detect, 1.0 for hourly, 0.1667 for 10-min)')
    
    args = parser.parse_args()
    
    # Initialize and run
    runner = BuyAndHoldBaselineRunner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        timebase_hours=args.timebase_hours
    )
    
    # Run optimization
    runner.run_optimization(max_timesteps=args.timesteps)
    
    # Save results
    final_metrics = runner.save_results()
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL RESULTS - Baseline 3: Buy-and-Hold Strategy")
    print("="*70)
    print(f"Initial Portfolio Value: ${final_metrics['initial_value_usd']/1e6:.2f}M USD")
    print(f"Final Portfolio Value:   ${final_metrics['final_value_usd']/1e6:.2f}M USD")
    print(f"Total Return:            {final_metrics['total_return']*100:+.2f}%")
    print(f"Annual Return:           {final_metrics['annual_return']*100:+.2f}%")
    print(f"Sharpe Ratio:            {final_metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown:        {final_metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility:              {final_metrics['volatility']*100:.4f}%")
    print(f"\nStrategy Characteristics:")
    print(f"Risk-Free Rate:          {final_metrics['risk_free_rate_annual']*100:.1f}% annual")
    print(f"Market Exposure:         {final_metrics['market_exposure']*100:.1f}%")
    print(f"Active Decisions:        {final_metrics['active_decisions']}")
    print(f"Final Cash Holdings:     ${final_metrics['final_cash_usd']/1e6:.2f}M USD")
    print("="*70)


if __name__ == "__main__":
    main()
