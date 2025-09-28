#!/usr/bin/env python3
"""
Baseline 2 Runner: Rule-Based Heuristic System

Runs the rule-based heuristic baseline for IEEE benchmarking.
Implements domain expert knowledge through simple decision rules.

This baseline focuses exclusively on expert system/rule-based approaches:
- Weather-based renewable energy investment decisions
- Price-based battery arbitrage rules
- Risk-based position sizing heuristics
- Domain expert knowledge without optimization algorithms

NO mathematical optimization or machine learning - pure expert system approach.
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for importing main project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main project
from config import EnhancedConfig

# Import local baseline optimizer
from rule_based_optimizer import ExpertSystemEnergyOptimizer

class RuleBasedBaselineRunner:
    """Runner for rule-based heuristic baseline."""
    
    def __init__(self, data_path=None, output_dir="baseline2_results"):
        # Use shared data from root directory if no specific path provided
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainingdata.csv")

        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load main project configuration for consistency
        self.config = EnhancedConfig()

        # Load and prepare data
        self.data = self.load_data()

        # Initialize expert system optimizer
        self.optimizer = ExpertSystemEnergyOptimizer(
            initial_budget=self.config.init_budget
        )

        # Results storage
        self.results = []
        
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
        print(f"Columns: {list(data.columns)}")
        return data
    
    def run_optimization(self, max_timesteps=None):
        """Run rule-based heuristic optimization."""
        print("\n" + "="*60)
        print("Rule-Based Heuristic Energy Optimization Baseline")
        print("="*60)
        
        if max_timesteps is None:
            max_timesteps = len(self.data)
        else:
            max_timesteps = min(max_timesteps, len(self.data))
        
        print(f"Running optimization for {max_timesteps:,} timesteps...")
        
        start_time = time.time()
        
        for t in range(max_timesteps):
            # Get current data
            data_row = self.data.iloc[t]
            
            # Execute optimization step (risk assessment is integrated in the optimizer)
            result = self.optimizer.step(data_row, t)
            
            self.results.append(result)
            
            # Progress reporting
            if t % 10000 == 0 or t == max_timesteps - 1:
                elapsed = time.time() - start_time
                progress = (t + 1) / max_timesteps * 100
                initial_budget = 8e8 / 0.145  # $800M USD in DKK
                portfolio_return = (result['portfolio_value'] - initial_budget) / initial_budget * 100
                
                # Convert to USD for display
                portfolio_value_usd = result['portfolio_value'] * 0.145  # DKK to USD
                cash_usd = result['cash'] * 0.145  # DKK to USD
                print(f"Progress: {progress:5.1f}% | Step: {t:6,} | "
                      f"Portfolio: ${portfolio_value_usd/1e6:.1f}M USD | "
                      f"Return: {portfolio_return:+5.1f}% | "
                      f"Cash: ${cash_usd/1e6:.1f}M USD | "
                      f"Wind: {result['wind_capacity']:.0f}MW | "
                      f"Solar: {result['solar_capacity']:.0f}MW | "
                      f"Hydro: {result['hydro_capacity']:.0f}MW | "
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
        
        # Use dataframe as-is (no nested dictionaries to flatten)
        df_flat = df
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "detailed_results.csv")
        df_flat.to_csv(results_path, index=False)
        print(f"Detailed results saved to: {results_path}")
        
        # Save summary metrics
        summary = self.optimizer.get_performance_metrics()
        
        # Add additional metrics
        battery_actions = [r['battery_action'] for r in self.results]
        summary['battery_charge_count'] = battery_actions.count('charge')
        summary['battery_discharge_count'] = battery_actions.count('discharge')
        summary['battery_idle_count'] = battery_actions.count('idle')
        
        total_investments = sum(len(r['investments']) for r in self.results)
        summary['total_investment_decisions'] = total_investments
        
        summary_path = os.path.join(self.output_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary metrics saved to: {summary_path}")
    
    def generate_report(self):
        """Generate performance report with visualizations."""
        df = pd.DataFrame(self.results)
        
        # Create visualizations
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Rule-Based Heuristic System - Performance Report', fontsize=16)
        
        # Portfolio value over time (convert to USD)
        portfolio_values_usd = df['portfolio_value'] * 0.145  # Convert DKK to USD
        axes[0, 0].plot(df['timestep'], portfolio_values_usd / 1e6, 'b-', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Portfolio Value (Million USD)')
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
        
        # Cash position over time (convert to USD)
        cash_values_usd = df['cash'] * 0.145  # Convert DKK to USD
        axes[1, 0].plot(df['timestep'], cash_values_usd / 1e6, 'g-', linewidth=2)
        axes[1, 0].set_title('Cash Position Over Time')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Cash (Million USD)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Battery state of charge
        axes[1, 1].plot(df['timestep'], df['battery_soc'], 'orange', linewidth=2)
        axes[1, 1].set_title('Battery State of Charge')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('SOC')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Investment decisions over time
        investment_counts = []
        window_size = 1000
        for i in range(0, len(df), window_size):
            window_data = df.iloc[i:i+window_size]
            total_investments = sum(len(inv) for inv in window_data['investments'])
            investment_counts.append(total_investments)
        
        x_windows = range(0, len(df), window_size)
        axes[2, 0].bar(x_windows, investment_counts, width=window_size*0.8, alpha=0.7)
        axes[2, 0].set_title(f'Investment Decisions (per {window_size} timesteps)')
        axes[2, 0].set_xlabel('Timestep')
        axes[2, 0].set_ylabel('Number of Investments')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Battery actions distribution
        battery_actions = [r['battery_action'] for r in self.results]
        action_counts = {action: battery_actions.count(action) for action in ['charge', 'discharge', 'idle']}
        
        axes[2, 1].pie(action_counts.values(), labels=action_counts.keys(), autopct='%1.1f%%')
        axes[2, 1].set_title('Battery Action Distribution')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "performance_report.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance report saved to: {plot_path}")
        
        # Generate decision analysis
        self.generate_decision_analysis()
    
    def generate_decision_analysis(self):
        """Generate detailed decision analysis."""
        df = pd.DataFrame(self.results)
        
        # Investment analysis
        all_investments = []
        for result in self.results:
            for inv in result['investments']:
                all_investments.append({
                    'timestep': result['timestep'],
                    'asset_type': inv[0],
                    'capacity': inv[1],
                    'cost': inv[2],
                    'price': result['price']
                })
        
        if all_investments:
            inv_df = pd.DataFrame(all_investments)
            
            # Investment summary by asset type
            inv_summary = inv_df.groupby('asset_type').agg({
                'capacity': ['sum', 'count', 'mean'],
                'cost': ['sum', 'mean'],
                'price': 'mean'
            }).round(2)
            
            # Save investment analysis
            inv_path = os.path.join(self.output_dir, "investment_analysis.csv")
            inv_summary.to_csv(inv_path)
            print(f"Investment analysis saved to: {inv_path}")
        
        # Battery performance analysis
        battery_df = df[['timestep', 'battery_action', 'battery_result', 'battery_soc', 'price']].copy()
        battery_summary = {
            'total_battery_revenue': df['battery_result'].sum(),
            'average_battery_revenue_per_action': df[df['battery_result'] != 0]['battery_result'].mean(),
            'battery_utilization': (df['battery_action'] != 'idle').mean(),
            'average_soc': df['battery_soc'].mean(),
            'soc_volatility': df['battery_soc'].std()
        }
        
        # Save battery analysis
        battery_path = os.path.join(self.output_dir, "battery_analysis.json")
        with open(battery_path, 'w') as f:
            json.dump(battery_summary, f, indent=2, default=str)
        print(f"Battery analysis saved to: {battery_path}")
    
    def get_final_metrics(self):
        """Get final performance metrics for benchmarking."""
        metrics = self.optimizer.get_performance_metrics()
        
        # Add IEEE benchmarking metrics
        df = pd.DataFrame(self.results)
        
        # Portfolio returns
        portfolio_returns = df['portfolio_value'].pct_change().dropna()
        
        # Additional metrics
        metrics['information_ratio'] = (portfolio_returns.mean() / portfolio_returns.std() 
                                      if portfolio_returns.std() > 0 else 0)
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        metrics['sortino_ratio'] = (portfolio_returns.mean() / downside_returns.std() 
                                   if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)
        
        metrics['win_rate'] = (portfolio_returns > 0).mean()
        metrics['average_win'] = portfolio_returns[portfolio_returns > 0].mean()
        metrics['average_loss'] = portfolio_returns[portfolio_returns < 0].mean()
        
        # Decision metrics
        metrics['total_investments'] = sum(len(r['investments']) for r in self.results)
        metrics['investment_frequency'] = metrics['total_investments'] / len(self.results)
        
        battery_actions = [r['battery_action'] for r in self.results]
        metrics['battery_utilization'] = (np.array(battery_actions) != 'idle').mean()
        
        return metrics


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rule-Based Heuristic Energy Optimization Baseline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to market data CSV')
    parser.add_argument('--timesteps', type=int, default=None, help='Maximum timesteps to run')
    parser.add_argument('--output_dir', type=str, default='baseline2_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Run baseline
    runner = RuleBasedBaselineRunner(args.data_path, args.output_dir)
    final_metrics = runner.run_optimization(args.timesteps)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS - Rule-Based Heuristic System")
    print("="*60)
    print(f"Final Portfolio Value: ${final_metrics['final_value_usd']/1e6:.2f}M USD")
    print(f"Initial Portfolio Value: ${final_metrics['initial_value_usd']/1e6:.2f}M USD")
    print(f"Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {final_metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility: {final_metrics['volatility']*100:.2f}%")
    print(f"Wind Capacity: {final_metrics['wind_capacity_mw']:.0f} MW")
    print(f"Solar Capacity: {final_metrics['solar_capacity_mw']:.0f} MW")
    print(f"Hydro Capacity: {final_metrics['hydro_capacity_mw']:.0f} MW")
    print(f"Final Cash: ${final_metrics['final_cash_usd']/1e6:.2f}M USD")
    print(f"Battery Utilization: {final_metrics.get('battery_utilization', 0)*100:.1f}%")
    print(f"Total Investments: {final_metrics.get('total_investment_decisions', 0)}")
    print("="*60)


if __name__ == "__main__":
    main()
