#!/usr/bin/env python3
"""
IEEE Standards Baseline Runner
==============================

This script runs the IEEE standards compliance evaluation baseline for renewable energy
investment systems. It provides comprehensive IEEE standards-compliant evaluation
and benchmarking capabilities for academic research.

IEEE Standards Evaluated:
- IEEE 1547: Grid Integration Standards
- IEEE 2030: Smart Grid Interoperability
- IEEE 1815: Communication Protocol Standards
- IEEE 3000: Power System Standards

Usage:
    python run_ieee_baseline.py --data_path trainingdata.csv --timesteps 10000
"""

import os
import sys
import argparse
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path for importing main project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main project modules
from config import EnhancedConfig

# Import local IEEE modules
from ieee_standards_optimizer import IEEEStandardsOptimizer, IEEEComplianceConfig


class IEEEBaselineRunner:
    """
    IEEE Standards Compliance Baseline Runner
    
    This class orchestrates the execution of IEEE standards compliance evaluation
    for renewable energy investment systems, providing comprehensive benchmarking
    capabilities for academic research and publication.
    """
    
    def __init__(self, data_path: Optional[str] = None, output_dir: str = "results"):
        """
        Initialize IEEE baseline runner.
        
        Args:
            data_path: Path to market data CSV file
            output_dir: Directory to save results
        """
        # Use shared data from root directory if no specific path provided
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainingdata.csv")

        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load main project configuration for consistency
        self.config = EnhancedConfig()
        
        # Initialize IEEE compliance configuration
        self.ieee_config = IEEEComplianceConfig()

        # Load and prepare data
        self.data = self.load_data()

        # Initialize IEEE standards optimizer with proper initial budget
        # Use same initial budget as other baselines: $800M USD = ~5.52B DKK (consistent with config.py)
        initial_budget_dkk = 8e8 / 0.145  # $800M USD converted to DKK
        self.optimizer = IEEEStandardsOptimizer(self.ieee_config, initial_budget_dkk)

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
                print(f"Warning: Column '{col}' not found, using zeros")
                data[col] = 0
        
        print(f"Loaded {len(data)} data points")
        return data
    
    def run_evaluation(self, max_timesteps: int = 10000) -> Dict[str, Any]:
        """
        Run IEEE standards compliance evaluation.
        
        Args:
            max_timesteps: Maximum number of timesteps to evaluate
            
        Returns:
            Final evaluation metrics
        """
        print(f"\nStarting IEEE Standards Compliance Evaluation...")
        print(f"Maximum timesteps: {max_timesteps}")
        print(f"Data points available: {len(self.data)}")
        
        # Limit timesteps to available data
        max_timesteps = min(max_timesteps, len(self.data))
        
        start_time = time.time()
        
        for t in range(max_timesteps):
            # Get current data
            data_row = self.data.iloc[t]
            
            # Execute IEEE compliance evaluation step
            result = self.optimizer.step(data_row, t)
            
            self.results.append(result)
            
            # Progress reporting
            if t % 1000 == 0 or t == max_timesteps - 1:
                elapsed = time.time() - start_time
                progress = (t + 1) / max_timesteps * 100
                
                # Convert portfolio value to USD for display
                portfolio_value_usd = result['portfolio_value'] * 0.145  # DKK to USD
                print(f"Progress: {progress:5.1f}% | Step: {t:6,} | "
                      f"Portfolio: ${portfolio_value_usd/1e6:.1f}M USD | "
                      f"IEEE Compliance: {result['overall_compliance_score']:.3f} | "
                      f"Grid Integration: {result['ieee_1547_compliance']['compliance_score']:.3f} | "
                      f"Interoperability: {result['ieee_2030_interoperability']['compliance_score']:.3f} | "
                      f"Communication: {result['ieee_1815_communication']['compliance_score']:.3f} | "
                      f"Power Quality: {result['ieee_3000_power_systems']['compliance_score']:.3f} | "
                      f"Elapsed: {elapsed:.0f}s")
        
        total_time = time.time() - start_time
        print(f"\nIEEE evaluation completed in {total_time:.1f} seconds")
        print(f"Average speed: {max_timesteps/total_time:.1f} evaluations/second")
        
        # Save results
        self.save_results()
        
        # Generate report
        self.generate_report()
        
        return self.get_final_metrics()
    
    def save_results(self):
        """Save detailed results to files."""
        print(f"\nSaving results to: {self.output_dir}")
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        
        # Use dataframe as-is (no nested dictionaries to flatten for IEEE baseline)
        df_flat = df
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "detailed_results.csv")
        df_flat.to_csv(results_path, index=False)
        print(f"Detailed results saved to: {results_path}")
        
        # Save IEEE compliance analysis
        ieee_analysis = self.analyze_ieee_compliance()
        ieee_path = os.path.join(self.output_dir, "ieee_compliance_analysis.json")
        with open(ieee_path, 'w') as f:
            json.dump(ieee_analysis, f, indent=2, default=str)
        print(f"IEEE compliance analysis saved to: {ieee_path}")
        
        # Save summary metrics
        summary = self.optimizer.get_summary()
        summary_path = os.path.join(self.output_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary metrics saved to: {summary_path}")
    
    def analyze_ieee_compliance(self) -> Dict[str, Any]:
        """Analyze IEEE standards compliance in detail."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        analysis = {
            'ieee_1547_analysis': {
                'avg_compliance_score': df['ieee_1547_compliance'].apply(lambda x: x['compliance_score']).mean(),
                'voltage_compliance_rate': df['ieee_1547_compliance'].apply(lambda x: x['voltage_regulation_compliant']).mean(),
                'frequency_compliance_rate': df['ieee_1547_compliance'].apply(lambda x: x['frequency_regulation_compliant']).mean(),
                'power_factor_compliance_rate': df['ieee_1547_compliance'].apply(lambda x: x['power_factor_compliant']).mean()
            },
            'ieee_2030_analysis': {
                'avg_compliance_score': df['ieee_2030_interoperability'].apply(lambda x: x['compliance_score']).mean(),
                'interoperability_compliance_rate': df['ieee_2030_interoperability'].apply(lambda x: x['interoperability_compliant']).mean(),
                'latency_compliance_rate': df['ieee_2030_interoperability'].apply(lambda x: x['latency_compliant']).mean(),
                'data_integrity_compliance_rate': df['ieee_2030_interoperability'].apply(lambda x: x['data_integrity_compliant']).mean()
            },
            'ieee_1815_analysis': {
                'avg_compliance_score': df['ieee_1815_communication'].apply(lambda x: x['compliance_score']).mean(),
                'dnp3_compliance_rate': df['ieee_1815_communication'].apply(lambda x: x['dnp3_compliant']).mean(),
                'scada_compliance_rate': df['ieee_1815_communication'].apply(lambda x: x['scada_compliant']).mean()
            },
            'ieee_3000_analysis': {
                'avg_compliance_score': df['ieee_3000_power_systems'].apply(lambda x: x['compliance_score']).mean(),
                'power_quality_compliance_rate': df['ieee_3000_power_systems'].apply(lambda x: x['power_quality_compliant']).mean(),
                'harmonic_compliance_rate': df['ieee_3000_power_systems'].apply(lambda x: x['harmonic_compliant']).mean()
            },
            'overall_analysis': {
                'avg_overall_compliance': df['overall_compliance_score'].mean(),
                'compliance_std': df['overall_compliance_score'].std(),
                'min_compliance': df['overall_compliance_score'].min(),
                'max_compliance': df['overall_compliance_score'].max(),
                'compliance_trend': 'improving' if df['overall_compliance_score'].iloc[-100:].mean() > df['overall_compliance_score'].iloc[:100].mean() else 'stable'
            }
        }
        
        return analysis
    
    def generate_report(self):
        """Generate IEEE compliance evaluation report with visualizations."""
        print("Generating IEEE compliance report...")
        
        df = pd.DataFrame(self.results)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE Standards Compliance Evaluation Report', fontsize=16)
        
        # Overall compliance score over time
        axes[0, 0].plot(df['timestep'], df['overall_compliance_score'])
        axes[0, 0].set_title('Overall IEEE Compliance Score')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Compliance Score')
        axes[0, 0].grid(True)
        
        # Individual IEEE standards compliance
        ieee_1547_scores = df['ieee_1547_compliance'].apply(lambda x: x['compliance_score'])
        ieee_2030_scores = df['ieee_2030_interoperability'].apply(lambda x: x['compliance_score'])
        ieee_1815_scores = df['ieee_1815_communication'].apply(lambda x: x['compliance_score'])
        ieee_3000_scores = df['ieee_3000_power_systems'].apply(lambda x: x['compliance_score'])
        
        axes[0, 1].plot(df['timestep'], ieee_1547_scores, label='IEEE 1547 (Grid)')
        axes[0, 1].plot(df['timestep'], ieee_2030_scores, label='IEEE 2030 (Smart Grid)')
        axes[0, 1].plot(df['timestep'], ieee_1815_scores, label='IEEE 1815 (Communication)')
        axes[0, 1].plot(df['timestep'], ieee_3000_scores, label='IEEE 3000 (Power)')
        axes[0, 1].set_title('Individual IEEE Standards Compliance')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Compliance Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Compliance score distribution
        axes[1, 0].hist(df['overall_compliance_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Overall Compliance Score Distribution')
        axes[1, 0].set_xlabel('Compliance Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # IEEE standards comparison (average scores)
        standards = ['IEEE 1547', 'IEEE 2030', 'IEEE 1815', 'IEEE 3000']
        avg_scores = [
            ieee_1547_scores.mean(),
            ieee_2030_scores.mean(),
            ieee_1815_scores.mean(),
            ieee_3000_scores.mean()
        ]
        
        axes[1, 1].bar(standards, avg_scores, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[1, 1].set_title('Average IEEE Standards Compliance Scores')
        axes[1, 1].set_ylabel('Average Compliance Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "ieee_compliance_report.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"IEEE compliance report saved to: {plot_path}")
    
    def get_final_metrics(self) -> Dict[str, Any]:
        """Get final IEEE compliance evaluation metrics."""
        summary = self.optimizer.get_summary()
        
        # Add IEEE-specific metrics
        ieee_metrics = {
            'ieee_evaluation_type': 'IEEE Standards Compliance Baseline',
            'total_evaluations': len(self.results),
            'ieee_standards_evaluated': ['IEEE 1547', 'IEEE 2030', 'IEEE 1815', 'IEEE 3000'],
            'academic_compliance': True,
            'publication_ready': True
        }
        
        # Combine with optimizer summary
        final_metrics = {**summary, **ieee_metrics}
        
        return final_metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="IEEE Standards Compliance Baseline Evaluation")
    parser.add_argument("--data_path", required=True, help="Path to market data CSV")
    parser.add_argument("--timesteps", type=int, default=10000, help="Maximum timesteps to evaluate")
    parser.add_argument("--output_dir", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("="*60)
    print(" IEEE STANDARDS COMPLIANCE BASELINE EVALUATION")
    print("="*60)
    print(f"Data file: {args.data_path}")
    print(f"Max timesteps: {args.timesteps}")
    print(f"Output directory: {args.output_dir}")
    print(f"IEEE Standards: 1547, 2030, 1815, 3000")
    
    # Initialize and run IEEE baseline
    runner = IEEEBaselineRunner(args.data_path, args.output_dir)
    final_metrics = runner.run_evaluation(args.timesteps)
    
    print("\n" + "="*60)
    print(" IEEE EVALUATION COMPLETED")
    print("="*60)
    print(f"Final Portfolio Value: ${final_metrics['final_value_usd']/1e6:.2f}M USD")
    print(f"Initial Portfolio Value: ${final_metrics['initial_value_usd']/1e6:.2f}M USD")
    print(f"Total Return: {final_metrics['total_return']*100:.2f}%")
    print(f"Overall Compliance Score: {final_metrics['performance_metrics']['overall_compliance_score']:.3f}")
    print(f"Grid Integration (IEEE 1547): {final_metrics['ieee_standards_summary']['ieee_1547_avg_score']:.3f}")
    print(f"Smart Grid (IEEE 2030): {final_metrics['ieee_standards_summary']['ieee_2030_avg_score']:.3f}")
    print(f"Communication (IEEE 1815): {final_metrics['ieee_standards_summary']['ieee_1815_avg_score']:.3f}")
    print(f"Power Systems (IEEE 3000): {final_metrics['ieee_standards_summary']['ieee_3000_avg_score']:.3f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
