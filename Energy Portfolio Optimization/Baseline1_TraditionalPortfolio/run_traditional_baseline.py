#!/usr/bin/env python3
"""
Baseline 1 Runner: Traditional Portfolio Optimization

Runs the traditional portfolio optimization baseline for IEEE benchmarking.
Implements classical financial optimization techniques for comparison with MARL.
"""

import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from traditional_portfolio_optimizer import TraditionalPortfolioOptimizer, TraditionalBatteryOptimizer

class TraditionalBaselineRunner:
    """Runner for traditional portfolio optimization baseline."""
    
    def __init__(self, data_path, output_dir="baseline1_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        self.data = self.load_data()
        
        # Initialize optimizers
        self.portfolio_optimizer = TraditionalPortfolioOptimizer()
        self.battery_optimizer = TraditionalBatteryOptimizer()
        
        # Results storage
        self.results = []
        self.metrics_log = []
        
    def load_data(self):
        """Load and prepare market data."""
        print(f"Loading data from: {self.data_path}")
        data = pd.read_csv(self.data_path)
        
        # Ensure required columns exist
        required_cols = ['wind', 'solar', 'hydro', 'price', 'load']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        print(f"Data loaded: {data.shape}")
        print(f"Date range: {data.index[0] if 'timestamp' in data.columns else 'No timestamp'}")
        return data
    
    def run_optimization(self, max_timesteps=None):
        """Run traditional portfolio optimization."""
        print("\n" + "="*60)
        print("Traditional Portfolio Optimization Baseline")
        print("="*60)
        
        if max_timesteps is None:
            max_timesteps = len(self.data)
        else:
            max_timesteps = min(max_timesteps, len(self.data))
        
        print(f"Running optimization for {max_timesteps:,} timesteps...")
        
        start_time = time.time()
        
        for t in range(max_timesteps):
            # Portfolio optimization step
            portfolio_result = self.portfolio_optimizer.step(self.data, t)
            
            # Battery optimization step
            current_price = self.data.iloc[t]['price']
            battery_revenue = self.battery_optimizer.optimize_battery(current_price)
            
            # Combine results
            result = {
                'timestep': t,
                'portfolio_value': portfolio_result['portfolio_value'],
                'battery_revenue': battery_revenue,
                'battery_soc': self.battery_optimizer.soc,
                'total_value': portfolio_result['portfolio_value'] + battery_revenue,
                'weights': portfolio_result['weights'],
                'returns': portfolio_result['returns'],
                'price': current_price
            }
            
            # Add portfolio metrics
            result.update(portfolio_result['metrics'])
            
            self.results.append(result)
            
            # Progress reporting
            if t % 10000 == 0 or t == max_timesteps - 1:
                elapsed = time.time() - start_time
                progress = (t + 1) / max_timesteps * 100
                portfolio_return = (portfolio_result['portfolio_value'] - 5e8) / 5e8 * 100
                
                print(f"Progress: {progress:5.1f}% | Step: {t:6,} | "
                      f"Portfolio: ${portfolio_result['portfolio_value']/1e6:.1f}M | "
                      f"Return: {portfolio_return:+5.1f}% | "
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
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "detailed_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Detailed results saved to: {results_path}")
        
        # Save summary metrics
        summary = self.portfolio_optimizer.get_summary()
        summary['battery_total_revenue'] = sum(r['battery_revenue'] for r in self.results)
        summary['final_battery_soc'] = self.battery_optimizer.soc
        
        summary_path = os.path.join(self.output_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary metrics saved to: {summary_path}")
    
    def generate_report(self):
        """Generate performance report with visualizations."""
        df = pd.DataFrame(self.results)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Traditional Portfolio Optimization - Performance Report', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(df['timestep'], df['portfolio_value'] / 1e6)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Portfolio Value (Million DKK)')
        axes[0, 0].grid(True)
        
        # Asset allocation over time
        weights_df = pd.DataFrame(list(df['weights']))
        weights_df.columns = ['Wind', 'Solar', 'Hydro', 'Battery', 'Cash']
        weights_df.index = df['timestep']
        
        for i, asset in enumerate(weights_df.columns):
            axes[0, 1].plot(weights_df.index[::1000], weights_df[asset].iloc[::1000], label=asset)
        axes[0, 1].set_title('Asset Allocation Over Time')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Battery state of charge
        axes[1, 0].plot(df['timestep'], df['battery_soc'])
        axes[1, 0].set_title('Battery State of Charge')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('SOC')
        axes[1, 0].grid(True)
        
        # Returns distribution
        portfolio_returns = df['portfolio_value'].pct_change().dropna()
        axes[1, 1].hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Portfolio Returns Distribution')
        axes[1, 1].set_xlabel('Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "performance_report.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance report saved to: {plot_path}")
    
    def get_final_metrics(self):
        """Get final performance metrics for benchmarking."""
        summary = self.portfolio_optimizer.get_summary()
        
        # Add battery metrics
        battery_revenue = sum(r['battery_revenue'] for r in self.results)
        summary['battery_total_revenue'] = battery_revenue
        summary['battery_contribution'] = battery_revenue / summary['final_value'] if summary['final_value'] > 0 else 0
        
        # Add IEEE benchmarking metrics
        df = pd.DataFrame(self.results)
        returns = df['portfolio_value'].pct_change().dropna()
        
        summary['information_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
        summary['sortino_ratio'] = returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        summary['max_consecutive_losses'] = self._max_consecutive_losses(returns)
        summary['win_rate'] = (returns > 0).mean()
        
        return summary
    
    def _max_consecutive_losses(self, returns):
        """Calculate maximum consecutive losses."""
        losses = returns < 0
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Traditional Portfolio Optimization Baseline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to market data CSV')
    parser.add_argument('--timesteps', type=int, default=None, help='Maximum timesteps to run')
    parser.add_argument('--output_dir', type=str, default='baseline1_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Run baseline
    runner = TraditionalBaselineRunner(args.data_path, args.output_dir)
    final_metrics = runner.run_optimization(args.timesteps)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS - Traditional Portfolio Optimization")
    print("="*60)
    print(f"Final Portfolio Value: ${final_metrics['final_value']/1e6:.2f}M")
    print(f"Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {final_metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility: {final_metrics['volatility']*100:.2f}%")
    print(f"Calmar Ratio: {final_metrics['calmar_ratio']:.3f}")
    print(f"Battery Revenue: ${final_metrics['battery_total_revenue']/1e6:.2f}M")
    print("="*60)


if __name__ == "__main__":
    main()
