#!/usr/bin/env python3
"""
Baseline 2 Runner: Rule-Based Heuristic Portfolio

Runs the rule-based heuristic baseline for renewable energy portfolio evaluation.
Implements simple domain expert rules without optimization or machine learning.

Key Features:
- Weather-based investment decisions (heuristic)
- Price-based battery arbitrage (heuristic)
- Data-driven operational revenue (generation × price)
- Risk management rules (heuristic)

NO mathematical optimization or machine learning - pure expert system approach.
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import time
from datetime import datetime

# Set matplotlib backend for headless operation (before importing pyplot)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimizer
from rule_based_optimizer import RuleBasedHeuristicOptimizer

class RuleBasedBaselineRunner:
    """Runner for rule-based heuristic baseline."""

    def __init__(self, data_path, output_dir="Baseline2_RuleBasedHeuristic/results", timebase_hours=None):
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
        self.optimizer = RuleBasedHeuristicOptimizer(
            initial_budget_usd=initial_budget_usd,
            timebase_hours=timebase_hours
        )

        # Results storage
        self.results = []
        
    def load_data(self):
        """Load and prepare market data."""
        print(f"Loading data from: {self.data_path}")
        data = pd.read_csv(self.data_path)

        # Ensure required columns exist
        required_cols = ['wind', 'solar', 'hydro', 'price']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")

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
    
    def run_optimization(self, max_timesteps=None):
        """Run rule-based heuristic optimization."""
        print("\n" + "="*70)
        print("Baseline 2: Rule-Based Heuristic Portfolio")
        print("="*70)

        if max_timesteps is None:
            max_timesteps = len(self.data)
        else:
            max_timesteps = min(max_timesteps, len(self.data))

        print(f"Running optimization for {max_timesteps:,} timesteps...")
        print(f"Timebase: {self.timebase_hours} hours per step")
        print(f"Initial budget: ${self.optimizer.initial_budget_usd/1e6:.1f}M USD")

        start_time = time.time()

        for t in range(max_timesteps):
            # Get current data
            data_row = self.data.iloc[t]

            # Execute heuristic decision step
            result = self.optimizer.step(data_row, t)

            self.results.append(result)

            # Progress reporting
            if t % 5000 == 0 or t == max_timesteps - 1:
                elapsed = time.time() - start_time
                progress = (t + 1) / max_timesteps * 100
                portfolio_return = (result['portfolio_value_usd'] - self.optimizer.initial_budget_usd) / self.optimizer.initial_budget_usd * 100

                print(f"Step {t:6,}/{max_timesteps:,} ({progress:5.1f}%) | "
                      f"Portfolio: ${result['portfolio_value_usd']/1e6:6.1f}M | "
                      f"Return: {portfolio_return:+6.2f}% | "
                      f"Cash: ${result['cash_usd']/1e6:5.1f}M | "
                      f"Capacity: ${result['capacity_value_usd']/1e6:5.1f}M | "
                      f"Wind: {result['wind_capacity']:4.0f}MW | "
                      f"Solar: {result['solar_capacity']:4.0f}MW | "
                      f"Hydro: {result['hydro_capacity']:4.0f}MW | "
                      f"Elapsed: {elapsed:.0f}s")

        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.1f} seconds")
        print(f"Average speed: {max_timesteps/total_time:.1f} steps/second")

        # Save results
        self.save_results()
        self.generate_report()

        return self.get_final_metrics()
    
    def save_results(self):
        """Save optimization results."""
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)

        # Flatten investments column (list of tuples)
        df['num_investments'] = df['investments'].apply(len)

        # Save detailed results
        results_path = os.path.join(self.output_dir, "detailed_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to: {results_path}")

        # Save summary metrics
        summary = self.optimizer.get_performance_metrics()

        # Add additional metrics
        battery_actions = df['battery_action'].value_counts().to_dict()
        summary['battery_charge_count'] = battery_actions.get('charge', 0)
        summary['battery_discharge_count'] = battery_actions.get('discharge', 0)
        summary['battery_idle_count'] = battery_actions.get('idle', 0)
        summary['total_investment_decisions'] = int(df['num_investments'].sum())
        summary['total_operational_revenue_usd'] = float(df['operational_revenue'].sum() * self.optimizer.dkk_to_usd_rate)
        summary['total_battery_revenue_usd'] = float(df['battery_revenue'].sum() * self.optimizer.dkk_to_usd_rate)
        summary['total_cash_return_usd'] = float(df['cash_return'].sum() * self.optimizer.dkk_to_usd_rate)

        summary_path = os.path.join(self.output_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary metrics saved to: {summary_path}")

        # Also save as summary.json for consistency with Baseline1
        summary_path2 = os.path.join(self.output_dir, "summary.json")
        with open(summary_path2, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary also saved to: {summary_path2}")
    
    def generate_report(self):
        """Generate performance report with visualizations."""
        df = pd.DataFrame(self.results)

        # Create visualizations
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Baseline 2: Rule-Based Heuristic Portfolio - Performance Report', fontsize=16, fontweight='bold')

        # Portfolio value over time (USD)
        axes[0, 0].plot(df['timestep'], df['portfolio_value_usd'] / 1e6, 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.optimizer.initial_budget_usd/1e6, color='r', linestyle='--', alpha=0.5, label='Initial')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Portfolio Value (Million USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Asset capacity over time
        axes[0, 1].plot(df['timestep'], df['wind_capacity'], label='Wind', linewidth=2)
        axes[0, 1].plot(df['timestep'], df['solar_capacity'], label='Solar', linewidth=2)
        axes[0, 1].plot(df['timestep'], df['hydro_capacity'], label='Hydro', linewidth=2)
        axes[0, 1].set_title('Renewable Capacity Over Time')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Capacity (MW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Cash vs Capacity value (USD)
        axes[1, 0].plot(df['timestep'], df['cash_usd'] / 1e6, 'g-', linewidth=2, label='Cash')
        axes[1, 0].plot(df['timestep'], df['capacity_value_usd'] / 1e6, 'orange', linewidth=2, label='Capacity')
        axes[1, 0].set_title('Cash vs Capacity Value Over Time')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Value (Million USD)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Battery state of charge
        axes[1, 1].plot(df['timestep'], df['battery_soc'], 'purple', linewidth=2)
        axes[1, 1].set_title('Battery State of Charge')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('SOC')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        # Revenue breakdown (cumulative, USD)
        cumulative_operational = (df['operational_revenue'] * self.optimizer.dkk_to_usd_rate).cumsum() / 1e6
        cumulative_battery = (df['battery_revenue'] * self.optimizer.dkk_to_usd_rate).cumsum() / 1e6
        cumulative_cash = (df['cash_return'] * self.optimizer.dkk_to_usd_rate).cumsum() / 1e6

        axes[2, 0].plot(df['timestep'], cumulative_operational, label='Operational Revenue', linewidth=2)
        axes[2, 0].plot(df['timestep'], cumulative_battery, label='Battery Revenue', linewidth=2)
        axes[2, 0].plot(df['timestep'], cumulative_cash, label='Cash Return (Risk-Free)', linewidth=2)
        axes[2, 0].set_title('Cumulative Revenue by Source')
        axes[2, 0].set_xlabel('Timestep')
        axes[2, 0].set_ylabel('Cumulative Revenue (Million USD)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Battery actions distribution
        battery_actions = df['battery_action'].value_counts()
        colors = {'charge': 'blue', 'discharge': 'orange', 'idle': 'gray'}
        axes[2, 1].pie(battery_actions.values, labels=battery_actions.index, autopct='%1.1f%%',
                       colors=[colors.get(action, 'gray') for action in battery_actions.index])
        axes[2, 1].set_title('Battery Action Distribution')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, "performance_report.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance report saved to: {plot_path}")
    
    def get_final_metrics(self):
        """Get final performance metrics for benchmarking."""
        return self.optimizer.get_performance_metrics()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Baseline 2: Rule-Based Heuristic Portfolio')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation dataset CSV')
    parser.add_argument('--timesteps', type=int, default=None, help='Maximum timesteps to run')
    parser.add_argument('--output_dir', type=str, default='Baseline2_RuleBasedHeuristic/results',
                       help='Output directory')
    parser.add_argument('--timebase_hours', type=float, default=None,
                       help='Hours per timestep (None=auto-detect, 1.0 for hourly, 0.1667 for 10-min)')

    args = parser.parse_args()

    # Run baseline
    runner = RuleBasedBaselineRunner(args.data_path, args.output_dir, args.timebase_hours)
    final_metrics = runner.run_optimization(args.timesteps)

    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS - Baseline 2: Rule-Based Heuristic Portfolio")
    print("="*70)
    print(f"Initial Portfolio Value: ${final_metrics['initial_value_usd']/1e6:.2f}M USD")
    print(f"Final Portfolio Value:   ${final_metrics['final_value_usd']/1e6:.2f}M USD")
    print(f"Total Return:            {final_metrics['total_return']*100:+.2f}%")
    print(f"Annual Return:           {final_metrics['annual_return']*100:+.2f}%")
    print(f"Sharpe Ratio:            {final_metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown:        {final_metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility:              {final_metrics['volatility']*100:.2f}%")
    print(f"\nFinal Capacity:")
    print(f"  Wind:  {final_metrics['wind_capacity_mw']:.0f} MW")
    print(f"  Solar: {final_metrics['solar_capacity_mw']:.0f} MW")
    print(f"  Hydro: {final_metrics['hydro_capacity_mw']:.0f} MW")
    print(f"Final Cash: ${final_metrics['final_cash_usd']/1e6:.2f}M USD")
    print(f"\nRule Triggers:")
    for rule, count in final_metrics['rule_triggers'].items():
        print(f"  {rule}: {count}")
    print("="*70)


if __name__ == "__main__":
    main()
