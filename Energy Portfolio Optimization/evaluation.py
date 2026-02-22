#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation and Analysis Script
===========================================

Complete evaluation system that combines:
- Checkpoint-based evaluation (Stable Baselines3 models)
- Agent directory evaluation (MultiESGAgent system)
- Portfolio performance analysis
- Risk and economic model analysis
- Statistical confidence assessment
- Comprehensive plotting and reporting

Features:
- Automatic latest checkpoint detection
- Direct Stable Baselines3 model loading
- MultiESGAgent system loading
- Forecasting enable/disable for baseline comparison
- Comprehensive metrics calculation
- Portfolio analysis with plotting
- Flexible data input (defaults to unseen evaluation dataset)
- JSON results and analysis reports

Usage:
    # Evaluate latest checkpoint (automatic unseen data)
    python evaluation.py --mode checkpoint

    # Evaluate specific agent directory
    python evaluation.py --mode agents --trained_agents saved_agents --eval_data "evaluation_dataset/unseendata.csv"

    # Baseline comparison (no forecasting)
    python evaluation.py --mode agents --trained_agents saved_agents --eval_data "evaluation_dataset/unseendata.csv" --no_forecast

    # With comprehensive analysis and plots
    python evaluation.py --mode checkpoint --analyze --plot
"""

import argparse
import os
import sys
import warnings
import glob
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

def setup_console_encoding():
    """
    Best-effort Windows console UTF-8 configuration.

    IMPORTANT: Do NOT detach/replace sys.stdout/sys.stderr at import time.
    Import-time I/O mutation breaks logging handlers in other modules.
    """
    if sys.platform != "win32":
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # Non-fatal; keep defaults.
        pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_progress(message: str, step: int = None, total: int = None):
    """Print progress message with optional step counter."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if step is not None and total is not None:
        progress = f"[{step}/{total}]"
        print(f"[{timestamp}] {progress} {message}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)
    sys.stdout.flush()

# Import project modules (no side effects / no prints at import time).
from main import load_energy_data
from environment import RenewableMultiAgentEnv
from metacontroller import MultiESGAgent
from generator import MultiHorizonForecastGenerator

# Portfolio analysis imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ Matplotlib/Seaborn not available - plotting disabled")


class EvaluationConfig:
    """Lightweight config for evaluation mode."""
    def __init__(self):
        self.update_every = 128
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.verbose = 1
        self.seed = 42
        self.multithreading = True
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "PPO"},  # battery_operator_0
            {"mode": "PPO"},  # risk_controller_0
            {"mode": "SAC"},  # meta_controller_0
        ]


class PortfolioAnalysisConfig:
    """Configuration for portfolio analysis."""
    def __init__(self):
        self.risk_free_annual = 0.02
        self.min_annualization = 252
        self.max_annualization = 52560
        self.equity_cols = ("portfolio_value", "equity", "total_return_nav", "portfolio_performance", "fund_performance")
        self.budget_cols = ("budget", "investment_capital")
        self.plot_title = "Hybrid Renewable Energy Fund - AI Performance Analysis"

        # Get fund parameters from config if available
        try:
            from config import EnhancedConfig
            config = EnhancedConfig()
            self.initial_fund_size = config.init_budget_usd
            self.currency_conversion = config.dkk_to_usd_rate
            self.physical_allocation = config.physical_allocation
            self.target_baseline_return = 0.0376  # 3.76% baseline
            self.target_ai_return = 0.0544        # 5.44% AI-enhanced
        except ImportError:
            self.initial_fund_size = 800_000_000
            self.currency_conversion = 0.15
            self.physical_allocation = 0.88
            self.target_baseline_return = 0.0376
            self.target_ai_return = 0.0544


class SafeDivision:
    """Safe division utility to avoid division by zero."""
    @staticmethod
    def div(numerator, denominator, default=0.0):
        return numerator / denominator if denominator != 0 else default


class PortfolioAnalyzer:
    """Integrated portfolio analyzer for evaluation results."""

    def __init__(self, results_data: Dict[str, Any], log_data: Optional[pd.DataFrame] = None):
        self.results = results_data
        self.log_data = log_data
        self.config = PortfolioAnalysisConfig()
        self.analysis_results = {}

    def analyze_performance(self, make_plots: bool = False, save_plots: bool = False, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Perform comprehensive portfolio performance analysis."""
        print("\n📊 PERFORMING PORTFOLIO ANALYSIS")
        print("=" * 50)

        analysis = {}

        # Basic performance metrics
        analysis['basic_metrics'] = self._analyze_basic_metrics()

        # Risk analysis
        analysis['risk_analysis'] = self._analyze_risk_metrics()

        # Economic model analysis
        analysis['economic_analysis'] = self._analyze_economic_model()

        # AI performance analysis
        analysis['ai_analysis'] = self._analyze_ai_performance()

        # Statistical confidence
        analysis['confidence_analysis'] = self._analyze_statistical_confidence()

        if make_plots and HAS_PLOTTING:
            analysis['plots'] = self._create_performance_plots(save_plots, output_dir)

        # Generate summary
        analysis['summary'] = self._generate_analysis_summary(analysis)

        self.analysis_results = analysis
        print("✅ Portfolio analysis completed")

        return analysis

    def _analyze_basic_metrics(self) -> Dict[str, Any]:
        """Analyze basic performance metrics."""
        metrics = {}

        # Extract key metrics from results
        metrics['total_return'] = self.results.get('total_return', 0.0)
        metrics['sharpe_ratio'] = self.results.get('sharpe_ratio', 0.0)
        metrics['volatility'] = self.results.get('volatility', 0.0)
        metrics['max_drawdown'] = self.results.get('max_drawdown', 0.0)

        # Portfolio values
        initial_pv = self.results.get('initial_portfolio_value', 0.0)
        final_pv = self.results.get('final_portfolio_value', 0.0)

        if initial_pv > 0:
            metrics['absolute_return'] = final_pv - initial_pv
            metrics['return_percentage'] = (final_pv / initial_pv - 1) * 100

        # Performance vs targets
        actual_return = metrics['total_return']
        metrics['vs_baseline_target'] = actual_return - self.config.target_baseline_return
        metrics['vs_ai_target'] = actual_return - self.config.target_ai_return
        metrics['target_achievement_ratio'] = SafeDivision.div(actual_return, self.config.target_ai_return, 0.0)

        return metrics

    def _analyze_risk_metrics(self) -> Dict[str, Any]:
        """Analyze risk-related metrics."""
        risk_analysis = {}

        # Basic risk metrics
        risk_analysis['average_risk'] = self.results.get('average_risk', 0.0)
        risk_analysis['max_risk'] = self.results.get('max_risk', 0.0)
        risk_analysis['min_risk'] = self.results.get('min_risk', 0.0)

        # Risk-adjusted returns
        total_return = self.results.get('total_return', 0.0)
        avg_risk = risk_analysis['average_risk']

        if avg_risk > 0:
            risk_analysis['return_per_unit_risk'] = total_return / avg_risk
            risk_analysis['risk_efficiency'] = 'high' if risk_analysis['return_per_unit_risk'] > 0.1 else 'moderate'

        # Volatility analysis
        volatility = self.results.get('volatility', 0.0)
        risk_analysis['volatility_category'] = (
            'low' if volatility < 0.1 else
            'moderate' if volatility < 0.2 else
            'high'
        )

        return risk_analysis

    def _analyze_economic_model(self) -> Dict[str, Any]:
        """Analyze economic model performance."""
        economic = {}

        # Fund structure analysis
        initial_pv = self.results.get('initial_portfolio_value', 0.0)
        final_pv = self.results.get('final_portfolio_value', 0.0)

        economic['fund_size_initial'] = initial_pv
        economic['fund_size_final'] = final_pv
        economic['fund_growth'] = final_pv - initial_pv if initial_pv > 0 else 0.0
        economic['fund_growth_percentage'] = SafeDivision.div(economic['fund_growth'], initial_pv, 0.0) * 100

        # Asset allocation efficiency
        physical_target = self.config.physical_allocation
        economic['target_physical_allocation'] = physical_target * 100
        economic['target_trading_allocation'] = (1 - physical_target) * 100

        # Revenue analysis
        total_rewards = self.results.get('total_rewards', 0.0)
        economic['total_rewards'] = total_rewards
        economic['reward_efficiency'] = SafeDivision.div(total_rewards, initial_pv, 0.0) if initial_pv > 0 else 0.0

        return economic

    def _analyze_ai_performance(self) -> Dict[str, Any]:
        """Analyze AI-specific performance improvements."""
        ai_analysis = {}

        # Model performance
        if 'prediction_success_rate' in self.results:
            ai_analysis['prediction_success_rate'] = self.results['prediction_success_rate'] * 100
            ai_analysis['prediction_quality'] = (
                'excellent' if self.results['prediction_success_rate'] > 0.9 else
                'good' if self.results['prediction_success_rate'] > 0.7 else
                'moderate'
            )

        # Agent performance
        agent_rewards = {}
        total_agent_rewards = 0
        for key, value in self.results.items():
            if '_total_reward' in key:
                agent_name = key.replace('_total_reward', '')
                agent_rewards[agent_name] = value
                total_agent_rewards += value

        ai_analysis['agent_rewards'] = agent_rewards
        ai_analysis['total_agent_rewards'] = total_agent_rewards

        # AI enhancement assessment
        actual_return = self.results.get('total_return', 0.0)
        baseline_target = self.config.target_baseline_return
        ai_target = self.config.target_ai_return

        if actual_return > baseline_target:
            ai_analysis['ai_enhancement'] = actual_return - baseline_target
            ai_analysis['enhancement_percentage'] = SafeDivision.div(ai_analysis['ai_enhancement'], baseline_target, 0.0) * 100
        else:
            ai_analysis['ai_enhancement'] = 0.0
            ai_analysis['enhancement_percentage'] = 0.0

        ai_analysis['ai_effectiveness'] = (
            'highly_effective' if actual_return > ai_target else
            'effective' if actual_return > baseline_target else
            'needs_improvement'
        )

        return ai_analysis

    def _analyze_statistical_confidence(self) -> Dict[str, Any]:
        """Analyze statistical confidence in results."""
        confidence = {}

        # Evaluation steps and data quality
        eval_steps = self.results.get('evaluation_steps', 0)
        confidence['evaluation_steps'] = eval_steps
        confidence['data_points'] = eval_steps

        # Estimate confidence based on data size
        if eval_steps >= 10000:
            confidence['confidence_level'] = 'high'
            confidence['confidence_score'] = 0.95
        elif eval_steps >= 5000:
            confidence['confidence_level'] = 'moderate'
            confidence['confidence_score'] = 0.80
        elif eval_steps >= 1000:
            confidence['confidence_level'] = 'low'
            confidence['confidence_score'] = 0.65
        else:
            confidence['confidence_level'] = 'very_low'
            confidence['confidence_score'] = 0.50

        # Time period analysis (assuming 10-minute intervals)
        minutes_of_data = eval_steps * 10
        hours_of_data = minutes_of_data / 60
        days_of_data = hours_of_data / 24

        confidence['hours_of_data'] = hours_of_data
        confidence['days_of_data'] = days_of_data
        confidence['months_of_data'] = days_of_data / 30

        return confidence

    def _create_performance_plots(self, save_plots: bool = False, output_dir: str = "evaluation_results") -> Dict[str, str]:
        """Create performance visualization plots."""
        if not HAS_PLOTTING:
            return {'error': 'Plotting libraries not available'}

        plots_created = {}

        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(self.config.plot_title, fontsize=16, fontweight='bold')

            # Plot 1: Performance Summary
            ax1 = axes[0, 0]
            metrics = ['Total Return', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']
            values = [
                self.results.get('total_return', 0) * 100,
                self.results.get('sharpe_ratio', 0),
                self.results.get('volatility', 0) * 100,
                self.results.get('max_drawdown', 0) * 100
            ]

            bars = ax1.bar(metrics, values, color=['green', 'blue', 'orange', 'red'])
            ax1.set_title('Key Performance Metrics')
            ax1.set_ylabel('Value (%)')
            ax1.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')

            # Plot 2: Agent Performance
            ax2 = axes[0, 1]
            agent_rewards = {}
            for key, value in self.results.items():
                if '_total_reward' in key:
                    agent_name = key.replace('_total_reward', '').replace('_0', '')
                    agent_rewards[agent_name] = value

            if agent_rewards:
                agents = list(agent_rewards.keys())
                rewards = list(agent_rewards.values())
                bars = ax2.bar(agents, rewards, color='skyblue')
                ax2.set_title('Agent Performance (Total Rewards)')
                ax2.set_ylabel('Total Reward')
                ax2.tick_params(axis='x', rotation=45)

                # Add value labels
                for bar, reward in zip(bars, rewards):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{reward:.1f}', ha='center', va='bottom')

            # Plot 3: Risk Analysis
            ax3 = axes[1, 0]
            risk_metrics = ['Avg Risk', 'Max Risk', 'Volatility']
            risk_values = [
                self.results.get('average_risk', 0),
                self.results.get('max_risk', 0),
                self.results.get('volatility', 0)
            ]

            bars = ax3.bar(risk_metrics, risk_values, color='coral')
            ax3.set_title('Risk Metrics')
            ax3.set_ylabel('Risk Level')

            # Plot 4: Performance vs Targets
            ax4 = axes[1, 1]
            targets = ['Baseline Target', 'AI Target', 'Actual Return']
            target_values = [
                self.config.target_baseline_return * 100,
                self.config.target_ai_return * 100,
                self.results.get('total_return', 0) * 100
            ]

            colors = ['gray', 'lightblue', 'green']
            bars = ax4.bar(targets, target_values, color=colors)
            ax4.set_title('Performance vs Targets')
            ax4.set_ylabel('Return (%)')
            ax4.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, value in zip(bars, target_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')

            plt.tight_layout()

            if save_plots:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_created['performance_plot'] = plot_path
                print(f"📊 Performance plot saved: {plot_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")
            plots_created['error'] = str(e)

        return plots_created

    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis."""
        summary = {}

        # Overall performance assessment
        total_return = self.results.get('total_return', 0.0)
        ai_target = self.config.target_ai_return

        if total_return >= ai_target:
            summary['overall_assessment'] = 'EXCELLENT'
            summary['performance_grade'] = 'A'
        elif total_return >= self.config.target_baseline_return:
            summary['overall_assessment'] = 'GOOD'
            summary['performance_grade'] = 'B'
        elif total_return >= 0:
            summary['overall_assessment'] = 'MODERATE'
            summary['performance_grade'] = 'C'
        else:
            summary['overall_assessment'] = 'POOR'
            summary['performance_grade'] = 'D'

        # Key highlights
        summary['key_metrics'] = {
            'total_return_pct': f"{total_return * 100:.2f}%",
            'sharpe_ratio': f"{self.results.get('sharpe_ratio', 0):.3f}",
            'max_drawdown_pct': f"{self.results.get('max_drawdown', 0) * 100:.2f}%",
            'confidence_level': analysis.get('confidence_analysis', {}).get('confidence_level', 'unknown')
        }

        # Recommendations
        recommendations = []

        if total_return < ai_target:
            recommendations.append("Consider optimizing AI model parameters")

        if self.results.get('volatility', 0) > 0.2:
            recommendations.append("Implement additional risk management measures")

        if analysis.get('ai_analysis', {}).get('prediction_success_rate', 100) < 80:
            recommendations.append("Improve forecasting model accuracy")

        if not recommendations:
            recommendations.append("Maintain current strategy - performance is satisfactory")

        summary['recommendations'] = recommendations

        return summary

    def _analyze_economic_model(self) -> Dict[str, Any]:
        """Analyze economic model performance."""
        economic = {}

        # Fund structure analysis
        initial_pv = self.results.get('initial_portfolio_value', 0.0)
        final_pv = self.results.get('final_portfolio_value', 0.0)

        economic['fund_size_initial'] = initial_pv
        economic['fund_size_final'] = final_pv
        economic['fund_growth'] = final_pv - initial_pv if initial_pv > 0 else 0.0
        economic['fund_growth_percentage'] = SafeDivision.div(economic['fund_growth'], initial_pv, 0.0) * 100

        # Asset allocation efficiency
        physical_target = self.config.physical_allocation
        economic['target_physical_allocation'] = physical_target * 100
        economic['target_trading_allocation'] = (1 - physical_target) * 100

        # Revenue analysis
        total_rewards = self.results.get('total_rewards', 0.0)
        economic['total_rewards'] = total_rewards
        economic['reward_efficiency'] = SafeDivision.div(total_rewards, initial_pv, 0.0) if initial_pv > 0 else 0.0

        return economic

    def _analyze_ai_performance(self) -> Dict[str, Any]:
        """Analyze AI-specific performance improvements."""
        ai_analysis = {}

        # Model performance
        if 'prediction_success_rate' in self.results:
            ai_analysis['prediction_success_rate'] = self.results['prediction_success_rate'] * 100
            ai_analysis['prediction_quality'] = (
                'excellent' if self.results['prediction_success_rate'] > 0.9 else
                'good' if self.results['prediction_success_rate'] > 0.7 else
                'moderate'
            )

        # Agent performance
        agent_rewards = {}
        total_agent_rewards = 0
        for key, value in self.results.items():
            if '_total_reward' in key:
                agent_name = key.replace('_total_reward', '')
                agent_rewards[agent_name] = value
                total_agent_rewards += value

        ai_analysis['agent_rewards'] = agent_rewards
        ai_analysis['total_agent_rewards'] = total_agent_rewards

        # AI enhancement assessment
        actual_return = self.results.get('total_return', 0.0)
        baseline_target = self.config.target_baseline_return
        ai_target = self.config.target_ai_return

        if actual_return > baseline_target:
            ai_analysis['ai_enhancement'] = actual_return - baseline_target
            ai_analysis['enhancement_percentage'] = SafeDivision.div(ai_analysis['ai_enhancement'], baseline_target, 0.0) * 100
        else:
            ai_analysis['ai_enhancement'] = 0.0
            ai_analysis['enhancement_percentage'] = 0.0

        ai_analysis['ai_effectiveness'] = (
            'highly_effective' if actual_return > ai_target else
            'effective' if actual_return > baseline_target else
            'needs_improvement'
        )

        return ai_analysis

    def _analyze_statistical_confidence(self) -> Dict[str, Any]:
        """Analyze statistical confidence in results."""
        confidence = {}

        # Evaluation steps and data quality
        eval_steps = self.results.get('evaluation_steps', 0)
        confidence['evaluation_steps'] = eval_steps
        confidence['data_points'] = eval_steps

        # Estimate confidence based on data size
        if eval_steps >= 10000:
            confidence['confidence_level'] = 'high'
            confidence['confidence_score'] = 0.95
        elif eval_steps >= 5000:
            confidence['confidence_level'] = 'moderate'
            confidence['confidence_score'] = 0.80
        elif eval_steps >= 1000:
            confidence['confidence_level'] = 'low'
            confidence['confidence_score'] = 0.65
        else:
            confidence['confidence_level'] = 'very_low'
            confidence['confidence_score'] = 0.50

        # Time period analysis (assuming 10-minute intervals)
        minutes_of_data = eval_steps * 10
        hours_of_data = minutes_of_data / 60
        days_of_data = hours_of_data / 24

        confidence['hours_of_data'] = hours_of_data
        confidence['days_of_data'] = days_of_data
        confidence['months_of_data'] = days_of_data / 30

        return confidence

    def _create_performance_plots(self, save_plots: bool = False, output_dir: str = "evaluation_results") -> Dict[str, str]:
        """Create performance visualization plots."""
        if not HAS_PLOTTING:
            return {'error': 'Plotting libraries not available'}

        plots_created = {}

        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(self.config.plot_title, fontsize=16, fontweight='bold')

            # Plot 1: Performance Summary
            ax1 = axes[0, 0]
            metrics = ['Total Return', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']
            values = [
                self.results.get('total_return', 0) * 100,
                self.results.get('sharpe_ratio', 0),
                self.results.get('volatility', 0) * 100,
                self.results.get('max_drawdown', 0) * 100
            ]

            bars = ax1.bar(metrics, values, color=['green', 'blue', 'orange', 'red'])
            ax1.set_title('Key Performance Metrics')
            ax1.set_ylabel('Value (%)')
            ax1.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')

            # Plot 2: Agent Performance
            ax2 = axes[0, 1]
            agent_rewards = {}
            for key, value in self.results.items():
                if '_total_reward' in key:
                    agent_name = key.replace('_total_reward', '').replace('_0', '')
                    agent_rewards[agent_name] = value

            if agent_rewards:
                agents = list(agent_rewards.keys())
                rewards = list(agent_rewards.values())
                bars = ax2.bar(agents, rewards, color='skyblue')
                ax2.set_title('Agent Performance (Total Rewards)')
                ax2.set_ylabel('Total Reward')
                ax2.tick_params(axis='x', rotation=45)

                # Add value labels
                for bar, reward in zip(bars, rewards):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{reward:.1f}', ha='center', va='bottom')

            # Plot 3: Risk Analysis
            ax3 = axes[1, 0]
            risk_metrics = ['Avg Risk', 'Max Risk', 'Volatility']
            risk_values = [
                self.results.get('average_risk', 0),
                self.results.get('max_risk', 0),
                self.results.get('volatility', 0)
            ]

            bars = ax3.bar(risk_metrics, risk_values, color='coral')
            ax3.set_title('Risk Metrics')
            ax3.set_ylabel('Risk Level')

            # Plot 4: Performance vs Targets
            ax4 = axes[1, 1]
            targets = ['Baseline Target', 'AI Target', 'Actual Return']
            target_values = [
                self.config.target_baseline_return * 100,
                self.config.target_ai_return * 100,
                self.results.get('total_return', 0) * 100
            ]

            colors = ['gray', 'lightblue', 'green']
            bars = ax4.bar(targets, target_values, color=colors)
            ax4.set_title('Performance vs Targets')
            ax4.set_ylabel('Return (%)')
            ax4.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, value in zip(bars, target_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')

            plt.tight_layout()

            if save_plots:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_created['performance_plot'] = plot_path
                print(f"📊 Performance plot saved: {plot_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")
            plots_created['error'] = str(e)

        return plots_created

    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis."""
        summary = {}

        # Overall performance assessment
        total_return = self.results.get('total_return', 0.0)
        ai_target = self.config.target_ai_return

        if total_return >= ai_target:
            summary['overall_assessment'] = 'EXCELLENT'
            summary['performance_grade'] = 'A'
        elif total_return >= self.config.target_baseline_return:
            summary['overall_assessment'] = 'GOOD'
            summary['performance_grade'] = 'B'
        elif total_return >= 0:
            summary['overall_assessment'] = 'MODERATE'
            summary['performance_grade'] = 'C'
        else:
            summary['overall_assessment'] = 'POOR'
            summary['performance_grade'] = 'D'

        # Key highlights
        summary['key_metrics'] = {
            'total_return_pct': f"{total_return * 100:.2f}%",
            'sharpe_ratio': f"{self.results.get('sharpe_ratio', 0):.3f}",
            'max_drawdown_pct': f"{self.results.get('max_drawdown', 0) * 100:.2f}%",
            'confidence_level': analysis.get('confidence_analysis', {}).get('confidence_level', 'unknown')
        }

        # Recommendations
        recommendations = []

        if total_return < ai_target:
            recommendations.append("Consider optimizing AI model parameters")

        if self.results.get('volatility', 0) > 0.2:
            recommendations.append("Implement additional risk management measures")

        if analysis.get('ai_analysis', {}).get('prediction_success_rate', 100) < 80:
            recommendations.append("Improve forecasting model accuracy")

        if not recommendations:
            recommendations.append("Maintain current strategy - performance is satisfactory")

        summary['recommendations'] = recommendations

        return summary


def create_buy_and_hold_baseline(eval_data_path: str, timesteps: int) -> Dict[str, Any]:
    """
    Create a realistic Buy-and-Hold renewable energy baseline.

    This baseline:
    1. Starts with $800M and maintains it (no active trading)
    2. Earns conservative risk-free rate (like holding bonds/cash)
    3. Provides realistic passive investment comparison
    4. No renewable energy generation (pure financial holding)
    """
    import pandas as pd
    import numpy as np

    print("🏗️ Creating Buy-and-Hold baseline (passive cash/bonds strategy)...")

    # Load data
    data = pd.read_csv(eval_data_path)
    timesteps = min(timesteps, len(data))

    # Initial portfolio value (same as RL agents)
    initial_value = 800_000_000  # $800M USD
    current_value = initial_value

    # Conservative risk-free rate (Danish government bonds ~2% annually)
    annual_risk_free_rate = 0.02
    # Convert to per-timestep rate (assuming 10-minute intervals, 52,560 per year)
    timestep_rate = annual_risk_free_rate / 52560

    # Track portfolio values
    portfolio_values = [initial_value]

    for t in range(timesteps):
        # Conservative risk-free return (compound interest)
        current_value = current_value * (1 + timestep_rate)
        portfolio_values.append(current_value)

    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value

    # Calculate returns for risk metrics
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
    volatility = np.std(returns) if len(returns) > 1 else 0.0

    # Sharpe ratio (excess return over risk-free rate, but this IS the risk-free rate)
    # So Sharpe ratio should be 0 (no excess return over risk-free rate)
    sharpe_ratio = 0.0

    # Max drawdown (should be 0 for risk-free investment)
    max_drawdown = 0.0

    return {
        'method': 'Buy-and-Hold Strategy',
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'final_value_usd': final_value,
        'initial_value_usd': initial_value,
        'final_portfolio_value': final_value,
        'initial_portfolio_value': initial_value,
        'status': 'completed'
    }


def run_single_baseline(baseline_name: str, baseline_dir: str, eval_data_path: str, timesteps: int, output_dir: str = "evaluation_results") -> Tuple[bool, Optional[Dict]]:
    """Run a single baseline and return success status and results."""
    print(f"🚀 Running {baseline_name}...")

    # Create baseline-specific subdirectory within the main evaluation results folder
    baseline_output = os.path.join(output_dir, f"baseline_{baseline_name.lower().replace(' ', '_')}")
    os.makedirs(baseline_output, exist_ok=True)

    # Determine the script to run
    script_map = {
        "Traditional Portfolio": "run_traditional_baseline.py",
        "Rule-Based Heuristic": "run_rule_based_baseline.py",
        "Buy-and-Hold Strategy": "run_buy_and_hold_baseline.py"
    }

    script_name = script_map.get(baseline_name)
    if not script_name:
        print(f"❌ Unknown baseline: {baseline_name}")
        return False, None

    script_path = os.path.join(baseline_dir, script_name)
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False, None

    cmd = [
        sys.executable, script_path,
        "--data_path", eval_data_path,
        "--timesteps", str(timesteps),
        "--output_dir", baseline_output
    ]

    try:
        # Set timeout based on baseline type
        timeout = 180 if baseline_name == "IEEE Standards" else 300
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='ascii', errors='ignore')

        if result.returncode == 0:
            print(f"✅ {baseline_name} completed successfully")

            # Try to load results
            results_file = os.path.join(baseline_output, "evaluation_results.json")
            summary_file = os.path.join(baseline_output, "summary_metrics.json")

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    return True, json.load(f)
            elif os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    # Convert to expected format
                    return True, {
                        'method': baseline_name,
                        'total_return': data.get('total_return', 0.0),
                        'sharpe_ratio': data.get('sharpe_ratio', 0.0),
                        'max_drawdown': data.get('max_drawdown', 0.0),
                        'volatility': data.get('volatility', 0.0),
                        'final_value_usd': data.get('final_value_usd', 0.0),
                        'initial_value_usd': data.get('initial_value_usd', 0.0),
                        'status': 'completed'
                    }
            else:
                print(f"⚠️ No results found for {baseline_name}")
                return True, {
                    'method': baseline_name,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'status': 'completed_no_results'
                }
        else:
            print(f"❌ {baseline_name} failed: {result.stderr}")
            return False, {'error': result.stderr}

    except subprocess.TimeoutExpired:
        print(f"⏰ {baseline_name} timed out")
        return False, {'error': 'timeout'}
    except Exception as e:
        print(f"❌ {baseline_name} error: {str(e)}")
        return False, {'error': str(e)}


def run_traditional_baselines(eval_data_path: str, timesteps: int = 10000, output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """Run traditional baseline methods and return results."""
    print("🏛️ Running traditional baseline methods...")

    baseline_results = {}

    # Define baselines to run
    baselines = [
        ("Traditional Portfolio", "Baseline1_TraditionalPortfolio"),
        ("Rule-Based Heuristic", "Baseline2_RuleBasedHeuristic"),
        ("Buy-and-Hold Strategy", "Baseline3_BuyAndHold")
    ]

    print("🚀 Executing traditional baselines...")

    for i, (baseline_name, baseline_dir) in enumerate(baselines, 1):
        # Run traditional baseline via subprocess
        success, results = run_single_baseline(baseline_name, baseline_dir, eval_data_path, timesteps, output_dir)

        baseline_key = f"baseline_{i}"
        if success and results:
            baseline_results[baseline_key] = results
            # Report portfolio performance for successful baselines
            if 'final_value_usd' in results and 'initial_value_usd' in results:
                initial_val = results['initial_value_usd']
                final_val = results['final_value_usd']
                return_pct = ((final_val - initial_val) / initial_val * 100) if initial_val > 0 else 0
                print_progress(f"   💰 {baseline_name}: ${final_val/1e6:.1f}M (${initial_val/1e6:.1f}M → {return_pct:+.2f}%)")
            elif 'total_return' in results:
                return_pct = results['total_return']
                print_progress(f"   📈 {baseline_name}: Total return {return_pct:+.2f}%")
        else:
            baseline_results[baseline_key] = {
                'method': baseline_name,
                'status': 'failed',
                'error': results.get('error', 'unknown') if results else 'unknown'
            }

    # Check if any baselines succeeded
    successful_baselines = [k for k, v in baseline_results.items() if v.get('status') != 'failed']
    if successful_baselines:
        print_progress(f"✅ Traditional baselines completed: {len(successful_baselines)}/3 successful")

        # Summary of baseline performance
        print_progress("📊 Baseline Performance Summary:")
        for key, result in baseline_results.items():
            if result.get('status') != 'failed':
                method = result.get('method', 'Unknown')
                if 'final_value_usd' in result:
                    final_val = result['final_value_usd']
                    print_progress(f"   • {method}: ${final_val/1e6:.1f}M final value")
                elif 'total_return' in result:
                    return_pct = result['total_return']
                    print_progress(f"   • {method}: {return_pct:+.2f}% total return")
    else:
        print_progress("❌ All traditional baselines failed")

    return baseline_results


def find_latest_checkpoint(checkpoint_base_dir: str = "normal/checkpoints") -> Optional[str]:
    """Find the latest checkpoint directory or final models."""

    # First, try to find final_models directory (preferred)
    base_dir = os.path.dirname(checkpoint_base_dir) if checkpoint_base_dir.endswith('checkpoints') else checkpoint_base_dir
    final_models_dir = os.path.join(base_dir, "final_models")

    if os.path.exists(final_models_dir):
        # Check if final models contain the required policy files
        required_files = [
            "investor_0_policy.zip",
            "battery_operator_0_policy.zip",
            "risk_controller_0_policy.zip",
            "meta_controller_0_policy.zip"
        ]

        all_files_exist = all(os.path.exists(os.path.join(final_models_dir, f)) for f in required_files)

        if all_files_exist:
            print(f"🎯 Found final models directory: {final_models_dir}")
            return final_models_dir
        else:
            print(f"⚠️ Final models directory exists but missing some policy files")

    # Fallback to checkpoint detection
    if not os.path.exists(checkpoint_base_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_base_dir}")
        return None

    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(checkpoint_base_dir, "checkpoint_*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_base_dir}")
        return None

    # Sort by checkpoint number (extract number from checkpoint_XXXXX)
    def get_checkpoint_number(path):
        try:
            return int(os.path.basename(path).split('_')[1])
        except (IndexError, ValueError):
            return 0

    latest_checkpoint = max(checkpoints, key=get_checkpoint_number)
    checkpoint_num = get_checkpoint_number(latest_checkpoint)

    print(f"🔍 Found latest checkpoint: {os.path.basename(latest_checkpoint)} (step {checkpoint_num})")
    return latest_checkpoint


def load_checkpoint_models(checkpoint_dir: str) -> Dict[str, Any]:
    """Load models from checkpoint using Stable Baselines3."""
    print(f"🔥 Loading models from checkpoint: {checkpoint_dir}")
    
    try:
        # ------------------------------------------------------------------
        # NumPy pickle compatibility shim
        # ------------------------------------------------------------------
        # Some SB3 checkpoints can embed pickled objects referencing internal NumPy module paths
        # like `numpy._core.numeric` (NumPy 2.x). If this runtime uses a different NumPy layout,
        # unpickling can fail with: "No module named 'numpy._core.numeric'".
        #
        # We install a minimal alias so older/newer checkpoints can load on this machine.
        try:
            import sys
            import types
            import numpy as _np  # noqa: F401
            import numpy.core.numeric as _np_core_numeric

            if "numpy._core.numeric" not in sys.modules:
                sys.modules.setdefault("numpy._core", types.ModuleType("numpy._core"))
                sys.modules["numpy._core.numeric"] = _np_core_numeric
                # Expose as attribute for completeness (some loaders inspect numpy._core)
                try:
                    setattr(sys.modules["numpy"], "_core", sys.modules["numpy._core"])
                except Exception as alias_error:
                    print(f"Warning: NumPy shim attribute alias failed: {alias_error}")
        except Exception as shim_error:
            # Never block evaluation due to a best-effort compatibility shim.
            print(f"Warning: NumPy compatibility shim skipped: {shim_error}")

        from stable_baselines3 import PPO, SAC
        print("✅ Successfully imported stable-baselines3")
        
        model_configs = [
            ("investor_0_policy.zip", PPO),
            ("battery_operator_0_policy.zip", PPO),
            ("risk_controller_0_policy.zip", PPO),
            ("meta_controller_0_policy.zip", SAC)
        ]
        
        loaded_models = {}
        
        for model_file, model_class in model_configs:
            model_path = os.path.join(checkpoint_dir, model_file)
            agent_name = model_file.replace('_policy.zip', '')
            
            if os.path.exists(model_path):
                try:
                    print(f"🔧 Loading {model_file} with {model_class.__name__}...")
                    model = model_class.load(model_path)
                    loaded_models[agent_name] = model
                    print(f"✅ SUCCESS: Loaded {agent_name} from checkpoint!")
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
                    loaded_models[agent_name] = None
            else:
                print(f"❌ Model file not found: {model_path}")
                loaded_models[agent_name] = None
        
        return loaded_models
        
    except ImportError as e:
        print(f"❌ Cannot import stable-baselines3: {e}")
        return {}
    except Exception as e:
        print(f"❌ Error loading checkpoint models: {e}")
        return {}


def _load_eval_forecast_paths_episode20(forecast_base_dir: str = "forecast_models") -> Dict[str, str]:
    """
    Evaluation on unseen data uses the evaluation-reserved forecast models (Episode 20).
    No training should happen in evaluation.
    """
    try:
        from episode_forecast_integration import get_evaluation_forecast_paths
        return get_evaluation_forecast_paths(forecast_base_dir=forecast_base_dir)
    except Exception:
        # Fallback to the conventional folder layout
        ep_dir = os.path.join(forecast_base_dir, "episode_20")
        return {
            "model_dir": os.path.join(ep_dir, "models"),
            "scaler_dir": os.path.join(ep_dir, "scalers"),
            "metadata_dir": os.path.join(ep_dir, "metadata"),
        }

def _resolve_tier3_overlay_weights(final_models_dir: str, tier_dir: str) -> Optional[str]:
    """
    Resolve Tier 3 DL overlay weights (short-horizon 18D).
    """
    import re
    from config import OVERLAY_FEATURE_DIM
    dim_str = f"{OVERLAY_FEATURE_DIM}d"
    candidates = [
        os.path.join(final_models_dir, f"dl_overlay_online_{dim_str}.h5"),
        os.path.join(final_models_dir, "dl_overlay_online.h5"),
    ]

    ckpt_root = os.path.join(tier_dir, "checkpoints")
    if os.path.isdir(ckpt_root):
        ep_dirs = []
        try:
            for name in os.listdir(ckpt_root):
                m = re.match(r"episode_(\d+)$", name)
                if m:
                    ep_dirs.append((int(m.group(1)), os.path.join(ckpt_root, name)))
        except Exception:
            ep_dirs = []

        for _, ep_path in sorted(ep_dirs, key=lambda x: x[0], reverse=True):
            candidates.append(os.path.join(ep_path, f"dl_overlay_online_{dim_str}.h5"))
            candidates.append(os.path.join(ep_path, "dl_overlay_online.h5"))

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_overlay_weights_from_models_dir(models_dir: str) -> Optional[str]:
    """
    Resolve overlay weights from a generic models directory.

    Preferred filenames are the current Tier-2/Tier-3 naming.
    Legacy filename is supported only as a last-resort fallback.
    """
    try:
        from config import OVERLAY_FEATURE_DIM
        dim_str = f"{int(OVERLAY_FEATURE_DIM)}d"
    except Exception:
        dim_str = ""

    candidates = []
    roots = [models_dir, os.path.join(models_dir, "final_models")]
    for root in roots:
        if not root:
            continue
        if dim_str:
            candidates.append(os.path.join(root, f"dl_overlay_online_{dim_str}.h5"))
        candidates.append(os.path.join(root, "dl_overlay_online.h5"))
        # Legacy fallback for older runs.
        candidates.append(os.path.join(root, "hedge_optimizer_online.h5"))

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def run_tier_suite_evaluation(eval_data: pd.DataFrame, args) -> Dict[str, Any]:
    """Paper-style tier-suite evaluation on unseen data.

    Compares three training regimes under identical Tier-1 environment dynamics and
    observation spaces (no forecast-derived observations):

      - baseline:   Tier-1 MARL (no FGB/FAMC)
      - fgb_online: Tier-1 MARL trained with Forecast-Guided Baseline (FGB) online
      - fgb_meta:   Tier-1 MARL trained with FGB + meta-critic (FAMC)

    IMPORTANT:
      - Forecasts are NOT enabled during this evaluation suite.
      - FGB/FAMC are training-time variance-reduction techniques; evaluation should
        be the pure environment with the learned policies.
    """
    from config import EnhancedConfig

    results: Dict[str, Any] = {
        "evaluation_mode": "tiers",
        "eval_data": args.eval_data,
        "tiers": {},
    }

    # Default: evaluate the full unseen dataset unless user explicitly sets --eval_steps
    steps = args.eval_steps if args.eval_steps is not None else (len(eval_data) - 1)
    steps = int(max(1, steps))

    def _tier_eval(name: str, run_dir: str) -> Dict[str, Any]:
        tier_out = os.path.join(args.output_dir, name)
        os.makedirs(tier_out, exist_ok=True)
        env_log_dir = os.path.join(tier_out, "env_logs")
        os.makedirs(env_log_dir, exist_ok=True)

        # Load SB3 policies
        final_models_dir = os.path.join(run_dir, "final_models")
        models = load_checkpoint_models(final_models_dir)

        eval_mode = str(getattr(args, "tiers_eval_mode", "paper") or "paper").strip().lower()
        deployed = (eval_mode == "deployed")

        # Force Tier-1 eval config for all variants (fair comparison).
        cfg = EnhancedConfig()
        dl_weights_path = None

        if not deployed:
            cfg.overlay_enabled = False
            cfg.forecast_baseline_enable = False
            cfg.meta_baseline_enable = False
            cfg.fgb_mode = "online"

            env, _ = create_evaluation_environment(
                eval_data,
                enable_forecasting=False,
                output_dir=tier_out,
                investment_freq=args.investment_freq,
                config=cfg,
                env_log_dir=env_log_dir,
                fail_fast=True,
            )
        else:
            # Deployed/system evaluation: enable forecasting (Episode 20 models) + optionally attach DL overlay weights.
            # IMPORTANT: This is NOT paper-fair (runtime forecaster + overlay enabled).
            try:
                forecast_base = str(getattr(args, "forecast_base_dir", "forecast_models") or "forecast_models")
                forecast_paths = _load_eval_forecast_paths_episode20(forecast_base_dir=forecast_base)
            except Exception as e:
                raise RuntimeError(
                    f"Deployed tier evaluation requires valid forecast paths, but loading failed: {e}"
                ) from e

            model_dir = str(forecast_paths.get("model_dir") or "forecast_models/episode_20/models")
            scaler_dir = str(forecast_paths.get("scaler_dir") or "forecast_models/episode_20/scalers")
            metadata_dir = forecast_paths.get("metadata_dir", None)

            # Prefer using the existing evaluation cache directory if present.
            cache_root = str(getattr(args, "forecast_cache_dir", "forecast_cache") or "forecast_cache")
            cache_dir = os.path.join(cache_root, "forecast_cache_eval_episode20_2025", "forecast_cache_eval_episode20_2025-full")
            if not os.path.isdir(cache_dir):
                cache_dir = os.path.join(cache_root, "forecast_cache_eval_episode20_2025")

            # Enable overlay only for the variants that are supposed to ship it.
            wants_overlay = name in ("fgb_online", "fgb_meta")
            if wants_overlay:
                dl_weights_path = _resolve_tier3_overlay_weights(final_models_dir, run_dir)
                if not dl_weights_path:
                    raise RuntimeError(
                        f"Deployed tier evaluation for '{name}' requires overlay weights, but none were found in "
                        f"'{final_models_dir}' or checkpoints under '{run_dir}'."
                    )

            cfg.overlay_enabled = bool(dl_weights_path)
            cfg.forecast_baseline_enable = bool(dl_weights_path)
            cfg.meta_baseline_enable = bool(dl_weights_path) and (name == "fgb_meta")
            cfg.fgb_mode = "meta" if name == "fgb_meta" else "online"

            env, _ = create_evaluation_environment(
                eval_data,
                enable_forecasting=True,
                model_dir=model_dir,
                scaler_dir=scaler_dir,
                metadata_dir=metadata_dir,
                dl_overlay_weights_path=dl_weights_path,
                output_dir=tier_out,
                investment_freq=args.investment_freq,
                config=cfg,
                env_log_dir=env_log_dir,
                forecast_cache_dir=cache_dir,
                precompute_forecasts=bool(getattr(args, "tiers_precompute_forecasts", False)),
                precompute_batch_size=int(getattr(args, "tiers_precompute_batch_size", 8192) or 8192),
                fail_fast=True,
            )

        tier_metrics = run_checkpoint_evaluation(models, env, eval_data, steps)
        if tier_metrics is None:
            return {"status": "failed", "error": "evaluation_failed"}

        # Post-hoc reporting: decompose NAV into operating vs trading sleeves from env logs.
        # This does not change evaluation dynamics; it just makes the report interpretable
        # when physical book value dominates total NAV.
        try:
            debug_log_path = _find_env_debug_log(env_log_dir)
            tier_metrics["env_log_dir"] = env_log_dir
            tier_metrics["env_debug_log"] = debug_log_path or ""
            if debug_log_path:
                tier_metrics["sleeve_metrics"] = _compute_sleeve_metrics_from_env_log(
                    debug_log_path,
                    dkk_to_usd_rate=float(getattr(cfg, "dkk_to_usd_rate", 0.145) or 0.145),
                )
        except Exception as e:
            tier_metrics["sleeve_metrics"] = {"sleeve_metrics_error": str(e)}

        tier_metrics["status"] = "completed"
        tier_metrics["run_dir"] = run_dir
        tier_metrics["final_models_dir"] = final_models_dir
        tier_metrics["evaluation_mode"] = "tiers"
        tier_metrics["variant"] = name
        tier_metrics["tiers_eval_mode"] = eval_mode
        tier_metrics["deployed_enable_forecasting"] = bool(deployed)
        tier_metrics["deployed_dl_overlay_weights_path"] = dl_weights_path or ""
        tier_metrics["deployed_overlay_enabled"] = bool(getattr(cfg, "overlay_enabled", False))
        tier_metrics["deployed_meta_baseline_enable"] = bool(getattr(cfg, "meta_baseline_enable", False))
        return tier_metrics

    print_section_header("Tier Suite Evaluation (Unseen 2025 Full Year)")
    print_progress(f"Evaluating for {steps} steps")

    sel = str(getattr(args, "tiers_only", "all") or "all").strip().lower()

    def _wants(key: str, legacy_key: str) -> bool:
        if sel == "all":
            return True
        return sel in (key, legacy_key)

    # Baseline (legacy: tier1)
    if _wants("baseline", "tier1"):
        results["tiers"]["baseline"] = _tier_eval("baseline", args.tier1_dir)

    # FGB-online (legacy: tier2)
    if _wants("fgb_online", "tier2"):
        results["tiers"]["fgb_online"] = _tier_eval("fgb_online", args.tier2_dir)

    # FGB-meta (legacy: tier3)
    if _wants("fgb_meta", "tier3"):
        results["tiers"]["fgb_meta"] = _tier_eval("fgb_meta", args.tier3_dir)

    # Forecast-ablated variants (explicit labels for downstream aggregation).
    if sel in ("fgb_online_no_forecast", "tier2_no_forecast"):
        results["tiers"]["fgb_online_no_forecast"] = _tier_eval("fgb_online_no_forecast", args.tier2_dir)
    if sel in ("fgb_meta_no_forecast", "tier3_no_forecast"):
        results["tiers"]["fgb_meta_no_forecast"] = _tier_eval("fgb_meta_no_forecast", args.tier3_dir)

    return results


def _pct(x: Any) -> float:
    """Convert a return-like value to percent (supports fraction or percent already)."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    # Heuristic: if magnitude looks like already-percent (e.g. 4.2), keep it.
    if abs(v) > 1.5:
        return v
    return v * 100.0


def _find_env_debug_log(env_log_dir: str) -> Optional[str]:
    """Best-effort lookup of the per-episode debug CSV produced by the env logger."""
    try:
        # Common path in this codebase.
        p = os.path.join(env_log_dir, "tier1_debug_ep0.csv")
        if os.path.isfile(p):
            return p

        # Fallback: any *debug_ep0.csv in that directory.
        candidates = glob.glob(os.path.join(env_log_dir, "*debug_ep0.csv"))
        return candidates[0] if candidates else None
    except Exception:
        return None


def _compute_sleeve_metrics_from_env_log(debug_csv_path: str, dkk_to_usd_rate: float = 0.145) -> Dict[str, Any]:
    """Split total NAV into (operating sleeve) and (trading sleeve) from env debug logs.

    Why this exists:
      Total NAV in this hybrid-fund simulator includes the physical book value of assets.
      That makes total-NAV volatility/drawdown look tiny even when the financial trading
      sleeve takes meaningful risk. For reporting/paper figures, it is often clearer to
      decompose NAV into:
        - operating_sleeve = physical_book_value + accumulated_operational_revenue
        - trading_sleeve   = trading_cash + financial_mtm
    """
    metrics: Dict[str, Any] = {}
    if not debug_csv_path or not os.path.isfile(debug_csv_path):
        return metrics

    required = {
        "fund_nav_dkk",
        "trading_cash_dkk",
        "physical_book_value_dkk",
        "accumulated_operational_revenue_dkk",
        "financial_mtm_dkk",
    }
    optional = {"financial_exposure_dkk", "decision_step"}

    try:
        with open(debug_csv_path, "r", encoding="utf-8") as f:
            header = (f.readline() or "").strip().split(",")
        have = set(h for h in header if h)
    except Exception:
        have = set()

    if not required.issubset(have):
        # If the log schema changes, skip gracefully.
        missing = sorted(list(required - have))
        metrics["sleeve_metrics_error"] = f"missing_required_columns:{','.join(missing)}"
        return metrics

    usecols = sorted(list(required | (optional & have)))

    try:
        df = pd.read_csv(debug_csv_path, usecols=usecols)
    except Exception as e:
        metrics["sleeve_metrics_error"] = f"read_failed:{e}"
        return metrics

    # Convert to USD (all logs are in DKK).
    nav_usd = df["fund_nav_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    trading_usd = (
        (df["trading_cash_dkk"].to_numpy(dtype=float) + df["financial_mtm_dkk"].to_numpy(dtype=float))
        * dkk_to_usd_rate
    )
    operating_usd = (
        (df["physical_book_value_dkk"].to_numpy(dtype=float) + df["accumulated_operational_revenue_dkk"].to_numpy(dtype=float))
        * dkk_to_usd_rate
    )

    if len(nav_usd) == 0:
        return metrics

    total_gain = float(nav_usd[-1] - nav_usd[0])
    trading_gain = float(trading_usd[-1] - trading_usd[0])
    operating_gain = float(operating_usd[-1] - operating_usd[0])

    metrics.update({
        "sleeve_total_initial_usd": float(nav_usd[0]),
        "sleeve_total_final_usd": float(nav_usd[-1]),
        "sleeve_total_gain_usd": total_gain,
        "sleeve_trading_initial_usd": float(trading_usd[0]),
        "sleeve_trading_final_usd": float(trading_usd[-1]),
        "sleeve_trading_gain_usd": trading_gain,
        "sleeve_operating_initial_usd": float(operating_usd[0]),
        "sleeve_operating_final_usd": float(operating_usd[-1]),
        "sleeve_operating_gain_usd": operating_gain,
    })

    if total_gain != 0.0:
        metrics["sleeve_trading_gain_share"] = float(trading_gain / total_gain)
    else:
        metrics["sleeve_trading_gain_share"] = 0.0

    if trading_usd[0] != 0.0:
        metrics["sleeve_trading_return_pct"] = float((trading_usd[-1] / trading_usd[0] - 1.0) * 100.0)
    else:
        metrics["sleeve_trading_return_pct"] = 0.0

    if operating_usd[0] != 0.0:
        metrics["sleeve_operating_return_pct"] = float((operating_usd[-1] / operating_usd[0] - 1.0) * 100.0)
    else:
        metrics["sleeve_operating_return_pct"] = 0.0

    # Trading sleeve "risk" metrics (computed the same way as calculate_performance_metrics()).
    try:
        if len(trading_usd) > 1 and float(trading_usd[0]) != 0.0:
            tr = np.diff(trading_usd) / trading_usd[:-1]
            vol = float(np.std(tr)) if len(tr) > 1 else 0.0
            sharpe = float(np.mean(tr) / np.std(tr)) if vol > 0 else 0.0
            peak = np.maximum.accumulate(trading_usd)
            drawdowns = np.where(peak > 0.0, (peak - trading_usd) / peak, 0.0)
            dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            metrics["sleeve_trading_volatility"] = vol
            metrics["sleeve_trading_sharpe_ratio"] = sharpe
            metrics["sleeve_trading_max_drawdown_pct"] = dd * 100.0
    except Exception:
        pass

    # Exposure diagnostics on decision steps (more interpretable than total-NAV vol when book value dominates).
    try:
        if "financial_exposure_dkk" in df.columns:
            expo = df["financial_exposure_dkk"].to_numpy(dtype=float)
            if "decision_step" in df.columns:
                dec = df["decision_step"].astype(bool).to_numpy()
                expo = expo[dec] if dec.any() else expo
            metrics["sleeve_mean_abs_exposure_dkk"] = float(np.mean(np.abs(expo))) if len(expo) else 0.0
            metrics["sleeve_max_abs_exposure_dkk"] = float(np.max(np.abs(expo))) if len(expo) else 0.0
    except Exception:
        pass

    return metrics


def write_tier_comparison_report(tier_results: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """Write a single, consolidated comparison report (CSV + Markdown).

    Returns: (csv_path, md_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    preferred_order = [
        "baseline",
        "fgb_online",
        "fgb_meta",
        "fgb_online_no_forecast",
        "fgb_meta_no_forecast",
    ]
    variants = [v for v in preferred_order if v in tier_results]
    variants.extend([v for v in tier_results.keys() if v not in variants])
    if not variants:
        variants = preferred_order[:3]

    for variant in variants:
        r = tier_results.get(variant, {}) or {}
        s = r.get("sleeve_metrics", {}) or {}
        rows.append({
            "variant": variant,
            "status": r.get("status", "unknown"),
            "final_portfolio_value_usd": float(r.get("final_portfolio_value", 0.0)),
            "initial_portfolio_value_usd": float(r.get("initial_portfolio_value", 0.0)),
            "total_return_pct": _pct(r.get("total_return", 0.0)),
            "sharpe_ratio": float(r.get("sharpe_ratio", 0.0)),
            "max_drawdown_pct": _pct(r.get("max_drawdown", 0.0)),
            "volatility": float(r.get("volatility", 0.0)),
            "total_rewards": float(r.get("total_rewards", 0.0)),
            "average_risk": float(r.get("average_risk", 0.0)),
            # Sleeve decomposition (from env debug logs)
            "trading_gain_usd": float(s.get("sleeve_trading_gain_usd", 0.0)),
            "operating_gain_usd": float(s.get("sleeve_operating_gain_usd", 0.0)),
            "trading_gain_share": float(s.get("sleeve_trading_gain_share", 0.0)),
            "trading_return_pct": float(s.get("sleeve_trading_return_pct", 0.0)),
            "operating_return_pct": float(s.get("sleeve_operating_return_pct", 0.0)),
            "mean_abs_exposure_dkk": float(s.get("sleeve_mean_abs_exposure_dkk", 0.0)),
            "trading_sharpe": float(s.get("sleeve_trading_sharpe_ratio", 0.0)),
            "trading_max_dd_pct": float(s.get("sleeve_trading_max_drawdown_pct", 0.0)),
            "run_dir": r.get("run_dir", ""),
            "final_models_dir": r.get("final_models_dir", ""),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"tier_comparison_{ts}.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, f"tier_comparison_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Baseline vs FGB-online vs FGB-meta - Unseen Evaluation Report\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Output CSV: `{os.path.basename(csv_path)}`\n\n")

        f.write("### Summary Table\n\n")
        f.write("| Variant | Status | Final (USD) | Return % | Sharpe | Max DD % | Vol |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['variant']} | {row['status']} | {row['final_portfolio_value_usd']:.2f} | "
                f"{row['total_return_pct']:.3f} | {row['sharpe_ratio']:.4f} | "
                f"{row['max_drawdown_pct']:.3f} | {row['volatility']:.6f} |\n"
            )

        f.write("\n### Evaluation Inputs / Artifacts\n\n")
        for _, row in df.iterrows():
            f.write(f"- {row['variant']}:\n")
            f.write(f"  - final_models: `{row['final_models_dir']}`\n")
            if str(row.get('run_dir', '')).strip() != "":
                f.write(f"  - run_dir: `{row['run_dir']}`\n")

        f.write("\n### NAV Sleeve Decomposition (Operating vs Trading)\n\n")
        f.write("Total NAV includes physical book value, which can compress volatility/drawdown.\n")
        f.write("This section splits gains into the operating sleeve (physical + ops revenue) and\n")
        f.write("the trading sleeve (cash + MTM).\n\n")
        f.write("| Variant | Trading Gain (USD) | Operating Gain (USD) | Trading Share | Trading Return % | Mean |Exposure| (DKK) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['variant']} | {row['trading_gain_usd']:.2f} | {row['operating_gain_usd']:.2f} | "
                f"{row['trading_gain_share']:.3f} | {row['trading_return_pct']:.2f} | {row['mean_abs_exposure_dkk']:.2f} |\n"
            )

    return csv_path, md_path


def load_agent_system(trained_agents_dir: str, eval_env, enhanced: bool = False) -> Optional[Any]:
    """Load MultiESGAgent system from directory."""
    model_type = "enhanced" if enhanced else "standard"
    print(f"🤖 Loading {model_type} agent system from: {trained_agents_dir}")

    try:
        config = EvaluationConfig()
        agent_system = MultiESGAgent(
            config,
            env=eval_env,
            device="cpu",
            training=False,
            debug=False
        )

        # Load policies from disk
        loaded_count = agent_system.load_policies(trained_agents_dir)
        print(f"✅ Loaded {loaded_count} agent policies")

        if loaded_count == 0:
            print("❌ No agent policies loaded successfully")
            return None

        return agent_system

    except Exception as e:
        print(f"❌ Error loading agent system: {e}")
        return None


def load_enhanced_agent_system(enhanced_models_dir: str, eval_env) -> Optional[Any]:
    """Load enhanced MultiESGAgent system with forecasting and DL overlay."""
    print(f"🚀 Loading enhanced agent system from: {enhanced_models_dir}")

    # Check if this directory has enhanced features
    config_file = os.path.join(enhanced_models_dir, "training_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            enhanced_features = config.get('enhanced_features', {})
            forecasting_enabled = enhanced_features.get('forecasting_enabled', False)
            dl_overlay_enabled = enhanced_features.get('dl_overlay_enabled', False)
            has_dl_weights = enhanced_features.get('has_dl_weights', False)

            print(f"   📊 Enhanced features detected:")
            print(f"      Forecasting: {'✅' if forecasting_enabled else '❌'}")
            print(f"      DL Overlay: {'✅' if dl_overlay_enabled else '❌'}")
            print(f"      DL Weights: {'✅' if has_dl_weights else '❌'}")

            if has_dl_weights:
                dl_weights_path = _resolve_overlay_weights_from_models_dir(enhanced_models_dir)
                if dl_weights_path and os.path.exists(dl_weights_path):
                    print(f"   💾 DL overlay weights found: {dl_weights_path}")
                else:
                    print("   ⚠️ DL overlay weights missing")

        except Exception as e:
            print(f"   ⚠️ Could not read training config: {e}")

    return load_agent_system(enhanced_models_dir, eval_env, enhanced=True)


def create_evaluation_dl_adapter(base_env, dl_weights_path: str, config=None, strict_load: bool = False):
    """
    Create the Tier 3 DL overlay adapter for evaluation (short-horizon, matches training).
    """
    try:
        import numpy as np
        from dl_overlay import DLAdapter
        from utils import load_overlay_weights

        from config import OVERLAY_FEATURE_DIM
        feature_dim = int(OVERLAY_FEATURE_DIM)  # training contract for Tier 3 overlay
        meta_head_dim = int(getattr(config, "meta_baseline_head_dim", 32)) if config is not None else 32
        enable_meta_head = bool(getattr(config, "meta_baseline_enable", False)) if config is not None else False

        adapter = DLAdapter(
            feature_dim=feature_dim,
            verbose=False,
            meta_head_dim=meta_head_dim,
            enable_meta_head=enable_meta_head,
        )

        # Build weights by forcing one forward pass (TF needs variables created before load)
        _ = adapter.shared_inference(np.zeros((1, feature_dim), dtype=np.float32), training=False)

        ok = load_overlay_weights(adapter, dl_weights_path, feature_dim)
        if ok:
            print_progress("✅ DL overlay weights loaded successfully (Tier 3)")
        else:
            msg = "Failed to load DL overlay weights"
            if strict_load:
                raise RuntimeError(msg)
            print_progress(f"⚠️ {msg} (continuing with untrained overlay)")

        return adapter

    except Exception as e:
        if strict_load:
            raise RuntimeError(f"Failed to create Tier 3 DL adapter: {e}") from e
        print_progress(f"⚠️ Failed to create Tier 3 DL adapter: {e}")
        return None


def create_evaluation_environment(
    data: pd.DataFrame,
    enable_forecasting: bool = True,
    model_dir: str = "saved_models",
    scaler_dir: str = "saved_scalers",
    metadata_dir: Optional[str] = None,
    log_path: Optional[str] = None,
    dl_overlay_weights_path: Optional[str] = None,
    output_dir: str = "evaluation_results",
    investment_freq: int = 6,
    config=None,
    env_log_dir: Optional[str] = None,
    forecast_cache_dir: Optional[str] = None,
    precompute_forecasts: bool = False,
    precompute_batch_size: int = 8192,
    fail_fast: bool = True,
) -> Tuple[Any, Optional[Any]]:
    """Create evaluation environment with optional forecasting (Tier-aligned)."""
    print_progress("🏗️ Setting up evaluation environment...")

    # Setup forecaster
    forecaster = None
    if enable_forecasting:
        print_progress("🔮 Loading forecaster models and scalers...")
        try:
            # Auto-detect metadata directory if caller did not provide it
            if metadata_dir is None:
                if "Forecast_ANN" in model_dir or os.path.exists(os.path.join(os.path.dirname(model_dir), "metadata")):
                    potential_metadata = os.path.join(os.path.dirname(model_dir), "metadata")
                    if os.path.exists(potential_metadata):
                        metadata_dir = potential_metadata
            
            forecaster = MultiHorizonForecastGenerator(
                model_dir=model_dir,
                scaler_dir=scaler_dir,
                metadata_dir=metadata_dir,  # NEW: Auto-detected metadata directory
                look_back=24,  # IMPROVED: Default to 24 (will be overridden by metadata if available)
                verbose=False
            )
            print_progress("✅ Forecaster loaded successfully")

            # Optional: build/load an offline cache for the entire evaluation dataset.
            # This mirrors training-time precompute and makes Tier 2/3 evaluation faster + deterministic.
            if precompute_forecasts:
                try:
                    cache_dir = forecast_cache_dir or os.path.join(output_dir, "forecast_cache_eval")
                    os.makedirs(cache_dir, exist_ok=True)
                    print_progress(f"🧊 Precomputing/loading forecast cache for evaluation (batch_size={int(precompute_batch_size)})...")
                    forecaster.precompute_offline(
                        df=data,
                        timestamp_col="timestamp",
                        batch_size=max(1, int(precompute_batch_size)),
                        cache_dir=cache_dir,
                    )
                    print_progress(f"✅ Evaluation forecast cache ready: {cache_dir}")
                except Exception as e:
                    msg = f"Forecast cache precompute failed: {e}"
                    if fail_fast:
                        raise RuntimeError(msg) from e
                    print_progress(f"⚠️ {msg} (continuing with on-the-fly forecasts)")
        except Exception as e:
            if fail_fast:
                raise RuntimeError(f"Failed to load forecaster: {e}") from e
            print_progress(f"❌ Failed to load forecaster: {e}")
            print_progress("🚫 Continuing without forecasting...")
    else:
        print_progress("🚫 Forecasting disabled (baseline evaluation)")
    
    # Create base environment
    try:
        print_progress("🌍 Creating base environment...")

        # Create base environment without forecaster for training-compatible mode
        if enable_forecasting:
            base_env = RenewableMultiAgentEnv(
                data,
                investment_freq=int(investment_freq),
                forecast_generator=forecaster,
                config=config,
                log_dir=env_log_dir,
            )
            print_progress("✅ Enhanced base environment created")
        else:
            # Create basic environment without forecaster for training compatibility
            base_env = RenewableMultiAgentEnv(
                data,
                investment_freq=int(investment_freq),
                forecast_generator=None,
                config=config,
                log_dir=env_log_dir,
            )
            print_progress("✅ Basic base environment created")

        # Tier-2 eval trust: mirror training by initializing CalibrationTracker when forecasts are enabled.
        if enable_forecasting and config is not None:
            try:
                from dl_overlay import CalibrationTracker

                base_env.calibration_tracker = CalibrationTracker(
                    window_size=getattr(config, "forecast_trust_window", 500),
                    trust_metric=getattr(config, "forecast_trust_metric", "hitrate"),
                    verbose=False,
                    init_budget=getattr(config, "init_budget", None),
                    direction_weight=getattr(config, "forecast_trust_direction_weight", 0.8),
                    trust_boost=getattr(config, "forecast_trust_boost", 0.0),
                    min_exposure_ratio=getattr(config, "fgb_expected_dnav_min_exposure_ratio", 0.0),
                    pred_weight=getattr(config, "fgb_expected_dnav_pred_weight", 0.85),
                    mwdir_weight=getattr(config, "fgb_expected_dnav_mwdir_weight", 0.15),
                    fail_fast=bool(fail_fast),
                )
                print_progress("✅ CalibrationTracker initialized for evaluation (forecast trust)")
            except Exception as e:
                base_env.calibration_tracker = None
                if fail_fast:
                    raise RuntimeError(f"CalibrationTracker init failed: {e}") from e
                print_progress(f"⚠️ CalibrationTracker init failed: {e}")

        # Setup DL overlay if weights are provided
        if dl_overlay_weights_path and os.path.exists(dl_overlay_weights_path):
            try:
                print_progress(f"🚀 Loading DL overlay weights: {os.path.basename(dl_overlay_weights_path)}")

                # Create the real Tier 3 DL adapter and attach it
                dl_adapter = create_evaluation_dl_adapter(
                    base_env,
                    dl_overlay_weights_path,
                    config=config,
                    strict_load=bool(fail_fast),
                )

                if dl_adapter:
                    # Attach DL adapter to base environment (use canonical dl_adapter_overlay naming)
                    base_env.dl_adapter_overlay = dl_adapter
                    print_progress("✅ DL overlay attached to evaluation environment")
                else:
                    if fail_fast:
                        raise RuntimeError("Failed to create DL adapter")
                    print_progress("⚠️ Failed to create DL adapter")

            except Exception as e:
                if fail_fast:
                    raise RuntimeError(f"Failed to load DL overlay weights: {e}") from e
                print_progress(f"⚠️ Failed to load DL overlay weights: {e}")
                print_progress("   Continuing evaluation without DL overlay...")
        elif dl_overlay_weights_path and fail_fast:
            raise FileNotFoundError(f"DL overlay weights path does not exist: {dl_overlay_weights_path}")

        # Wrap with forecasting if enabled
        if forecaster is not None and enable_forecasting:
            print_progress("🔄 Wrapping environment with forecasting...")
            # We rely on the base env's log_dir instead.
            # No wrapper: keep Tier-1 observation spaces fixed for a fair comparison.
            eval_env = base_env
            print_progress("✅ Environment created with forecasting wrapper")
        else:
            eval_env = base_env
            print_progress("✅ Environment created (training-compatible baseline mode)")

        return eval_env, forecaster

    except Exception as e:
        if fail_fast:
            raise
        print_progress(f"❌ Error creating environment: {e}")
        return None, None




def _coerce_action_for_space(action: np.ndarray, action_space):
    """Make sure an action matches the environment's action space."""
    if hasattr(action_space, "n"):  # Discrete
        if isinstance(action, (list, tuple, np.ndarray)):
            action = np.array(action).astype(np.int64).flatten()
            return int(action[0])
        return int(action)
    else:  # Box
        act = np.array(action, dtype=np.float32).squeeze()
        if hasattr(action_space, "shape") and action_space.shape is not None:
            target = int(np.prod(action_space.shape))
            act = act.flatten()
            if act.size != target:
                if act.size < target:
                    act = np.pad(act, (0, target - act.size))
                else:
                    act = act[:target]
        return act


def calculate_performance_metrics(portfolio_values: list, 
                                rewards_by_agent: Dict[str, list],
                                risk_levels: list,
                                evaluation_steps: int) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    try:
        # Portfolio performance
        if portfolio_values:
            pv_array = np.array(portfolio_values)
            returns = np.diff(pv_array) / pv_array[:-1]
            
            metrics['total_return'] = (pv_array[-1] / pv_array[0] - 1) if len(pv_array) > 1 else 0.0
            metrics['volatility'] = np.std(returns) if len(returns) > 1 else 0.0
            metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
            if len(pv_array) > 0:
                peak = np.maximum.accumulate(pv_array)
                drawdowns = np.where(peak > 0.0, (peak - pv_array) / peak, 0.0)
                metrics['max_drawdown'] = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            else:
                metrics['max_drawdown'] = 0.0
            metrics['initial_portfolio_value'] = float(pv_array[0]) if len(pv_array) > 0 else 0.0
            metrics['final_portfolio_value'] = float(pv_array[-1]) if len(pv_array) > 0 else 0.0
        
        # Agent rewards
        total_rewards = 0
        for agent, rewards in rewards_by_agent.items():
            if rewards:
                agent_total = np.sum(rewards)
                metrics[f'{agent}_total_reward'] = float(agent_total)
                metrics[f'{agent}_avg_reward'] = float(np.mean(rewards))
                total_rewards += agent_total
        
        metrics['total_rewards'] = float(total_rewards)
        
        # Risk metrics
        if risk_levels:
            metrics['average_risk'] = float(np.mean(risk_levels))
            metrics['max_risk'] = float(np.max(risk_levels))
            metrics['min_risk'] = float(np.min(risk_levels))
        
        # Evaluation info
        metrics['evaluation_steps'] = evaluation_steps
        metrics['evaluation_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        print(f"⚠️ Error calculating performance metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics


def run_checkpoint_evaluation(models: Dict[str, Any],
                            eval_env,
                            data: pd.DataFrame,
                            evaluation_steps: int = 8000) -> Optional[Dict[str, Any]]:
    """Run evaluation using checkpoint models (Stable Baselines3)."""
    print("🚀 Starting checkpoint model evaluation...")

    if not models or not eval_env:
        print("❌ No models or environment available")
        return None

    # Count loaded models
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"🎯 Checkpoint models loaded: {loaded_count}/4")

    if loaded_count == 0:
        print("❌ No checkpoint models loaded successfully")
        return None

    # Run evaluation
    obs, _ = eval_env.reset()
    steps = min(evaluation_steps, len(data) - 1)

    # Tracking metrics
    portfolio_values = []
    rewards_by_agent = {agent: [] for agent in eval_env.possible_agents}
    risk_levels = []
    successful_predictions = 0
    total_predictions = 0

    print(f"🧪 Running evaluation for {steps} steps...")

    for step in range(steps):
        if step % 1000 == 0:
            current_portfolio = portfolio_values[-1] if portfolio_values else 800_000_000
            portfolio_change = ((current_portfolio / 800_000_000) - 1) * 100
            total_reward = sum(sum(rewards_by_agent[agent]) for agent in rewards_by_agent)
            success_rate = successful_predictions / total_predictions if total_predictions > 0 else 0.0
            print(f"📊 Progress: {step}/{steps} ({step/steps*100:.1f}%) | Portfolio: ${current_portfolio/1e6:.1f}M ({portfolio_change:+.2f}%) | Reward: {total_reward:.1f} | Success: {success_rate*100:.1f}%")

        actions = {}

        # Get actions from checkpoint models
        for agent in eval_env.possible_agents:
            if agent not in obs:
                continue

            total_predictions += 1
            agent_key = agent.replace('_0', '_0')  # Normalize agent name

            if agent_key in models and models[agent_key] is not None:
                try:
                    # Use checkpoint model for prediction
                    action, _ = models[agent_key].predict(obs[agent], deterministic=True)
                    actions[agent] = action
                    successful_predictions += 1
                except Exception as e:
                    print(f"⚠️ Prediction error for {agent}: {e}")
                    actions[agent] = eval_env.action_space(agent).sample()
            else:
                # Fallback to random action
                actions[agent] = eval_env.action_space(agent).sample()

        # Execute step
        try:
            obs, rewards, dones, truncs, infos = eval_env.step(actions)

            # Track metrics
            for agent, reward in rewards.items():
                rewards_by_agent[agent].append(float(reward))

            # Extract portfolio value and risk if available
            portfolio_value_dkk = None
            extraction_method = "unknown"

            # Try multiple methods to get portfolio value (in DKK)
            if hasattr(eval_env, 'get_portfolio_value'):
                portfolio_value_dkk = eval_env.get_portfolio_value()
                extraction_method = "get_portfolio_value"
            elif hasattr(eval_env, '_calculate_fund_nav'):
                # Direct access to environment's NAV calculation
                portfolio_value_dkk = eval_env._calculate_fund_nav()
                extraction_method = "_calculate_fund_nav"
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, '_calculate_fund_nav'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env._calculate_fund_nav()
                extraction_method = "wrapped._calculate_fund_nav"
            elif 'fund_nav' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['fund_nav']
                extraction_method = "infos.fund_nav"
            elif 'portfolio_value' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['portfolio_value']
                extraction_method = "infos.portfolio_value"
            elif hasattr(eval_env, 'equity'):
                portfolio_value_dkk = eval_env.equity
                extraction_method = "equity"
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'equity'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env.equity
                extraction_method = "wrapped.equity"
            else:
                # Fallback to initial value in DKK (800M USD = ~5.52B DKK)
                portfolio_value_dkk = 800_000_000 / 0.145  # Convert USD to DKK
                extraction_method = "fallback"

            # Convert DKK to USD for consistent analysis
            # Get conversion rate from environment or use default
            dkk_to_usd_rate = 0.145  # Default rate
            if hasattr(eval_env, 'config') and hasattr(eval_env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.config.dkk_to_usd_rate
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'config') and hasattr(eval_env.env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.env.config.dkk_to_usd_rate

            # Convert to USD for analysis
            portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
            portfolio_values.append(portfolio_value_usd)

            # Debug output for first step
            if step == 0:
                print(f"🔍 Portfolio value extraction: {portfolio_value_dkk/1e9:.2f}B DKK → ${portfolio_value_usd/1e6:.1f}M USD using method '{extraction_method}'")

            if hasattr(eval_env, 'get_risk_level'):
                risk_levels.append(eval_env.get_risk_level())

            # Handle episode termination
            if any(dones.values()) or any(truncs.values()):
                obs, _ = eval_env.reset()

        except Exception as e:
            print(f"⚠️ Step execution error: {e}")
            break

    print("✅ Checkpoint evaluation completed")

    # Calculate metrics
    success_rate = successful_predictions / total_predictions if total_predictions > 0 else 0.0
    metrics = calculate_performance_metrics(portfolio_values, rewards_by_agent, risk_levels, steps)

    # Add checkpoint-specific metrics
    metrics['prediction_success_rate'] = success_rate
    metrics['models_loaded'] = loaded_count
    metrics['evaluation_mode'] = 'checkpoint'

    return metrics


def run_agent_evaluation(agent_system,
                        eval_env,
                        data: pd.DataFrame,
                        evaluation_steps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Run evaluation using MultiESGAgent system."""
    print_progress("🚀 Starting agent system evaluation...")

    if not agent_system or not eval_env:
        print_progress("❌ No agent system or environment available")
        return None

    # Run evaluation
    print_progress("🔄 Resetting evaluation environment...")
    try:
        obs, _ = eval_env.reset()
        print_progress("✅ Environment reset successful")
    except Exception as e:
        print_progress(f"❌ Environment reset failed: {e}")
        return None

    # Pick evaluation length
    if evaluation_steps is None:
        evaluation_steps = min(len(data) - 1, 10_000)
    evaluation_steps = int(max(1, evaluation_steps))

    print_progress(f"📏 Evaluating for {evaluation_steps} steps")

    # Initialize PPO buffer if needed
    if hasattr(agent_system, 'policies'):
        for policy in agent_system.policies:
            if hasattr(policy, 'policy') and hasattr(policy.policy, 'reset_noise'):
                try:
                    policy.policy.reset_noise()
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    print_progress(
                        f"[PPO] reset_noise skipped for {getattr(policy, 'agent_name', 'unknown')}: {e}"
                    )
    print_progress("[PPO] PPO BUFFER RESET: Preserving financial state at step 0")

    # Tracking metrics
    portfolio_values = []
    rewards_by_agent = {agent: [] for agent in eval_env.possible_agents}
    risk_levels = []
    actions_taken = {agent: [] for agent in eval_env.possible_agents}

    print_progress("🔄 Starting evaluation loop...")

    for step in range(evaluation_steps):
        if step % 1000 == 0:
            current_portfolio = portfolio_values[-1] if portfolio_values else 800_000_000
            portfolio_change = ((current_portfolio / 800_000_000) - 1) * 100
            total_reward = sum(sum(rewards_by_agent[agent]) for agent in rewards_by_agent)
            print_progress(f"📊 Progress: {step}/{evaluation_steps} ({step/evaluation_steps*100:.1f}%) | Portfolio: ${current_portfolio/1e6:.1f}M ({portfolio_change:+.2f}%) | Total Reward: {total_reward:.1f}")

            # Debug: Show portfolio value extraction method used
            if step == 0 and portfolio_values:
                print_progress(f"🔍 Portfolio value extraction method working: ${current_portfolio/1e6:.1f}M")

        if step == 0:
            print_progress("🔄 Processing first step...")

        actions = {}

        if step == 0:
            print_progress("🔄 Getting actions from agents...")

        # Get actions from agent system
        for i, agent in enumerate(eval_env.possible_agents):
            if agent not in obs:
                continue

            try:
                if step == 0:
                    print_progress(f"🔄 Processing agent {agent} (obs shape: {np.array(obs[agent]).shape})")

                agent_obs = np.array(obs[agent], dtype=np.float32).reshape(1, -1)
                policy = agent_system.policies[i]

                if hasattr(policy, "predict"):
                    if step == 0:
                        print_progress(f"🔄 Getting prediction from {agent}...")
                    act, _ = policy.predict(agent_obs, deterministic=True)
                    if step == 0:
                        print_progress(f"✅ Got prediction from {agent}")
                else:
                    act = eval_env.action_space(agent).sample()

                act = _coerce_action_for_space(act, eval_env.action_space(agent))
                actions[agent] = act
                actions_taken[agent].append(np.array(act).copy() if hasattr(act, 'copy') else act)

            except Exception as e:
                print_progress(f"⚠️ Action prediction error for {agent}: {e}")
                actions[agent] = eval_env.action_space(agent).sample()

        if step == 0:
            print_progress("🔄 Executing first environment step...")

        # Execute step
        try:
            obs, rewards, dones, truncs, infos = eval_env.step(actions)
            if step == 0:
                print_progress("✅ First environment step completed")

            # Track metrics
            for agent, reward in rewards.items():
                rewards_by_agent[agent].append(float(reward))

            # Extract portfolio value and risk if available
            portfolio_value_dkk = None

            # Try multiple methods to get portfolio value (in DKK)
            if hasattr(eval_env, 'get_portfolio_value'):
                portfolio_value_dkk = eval_env.get_portfolio_value()
            elif hasattr(eval_env, '_calculate_fund_nav'):
                # Direct access to environment's NAV calculation
                portfolio_value_dkk = eval_env._calculate_fund_nav()
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, '_calculate_fund_nav'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env._calculate_fund_nav()
            elif 'fund_nav' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['fund_nav']
            elif 'portfolio_value' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['portfolio_value']
            elif hasattr(eval_env, 'equity'):
                portfolio_value_dkk = eval_env.equity
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'equity'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env.equity
            else:
                # Fallback to initial value in DKK (800M USD = ~5.52B DKK)
                portfolio_value_dkk = 800_000_000 / 0.145  # Convert USD to DKK

            # Convert DKK to USD for consistent analysis
            # Get conversion rate from environment or use default
            dkk_to_usd_rate = 0.145  # Default rate
            if hasattr(eval_env, 'config') and hasattr(eval_env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.config.dkk_to_usd_rate
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'config') and hasattr(eval_env.env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.env.config.dkk_to_usd_rate

            # Convert to USD for analysis
            portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
            portfolio_values.append(portfolio_value_usd)

            if hasattr(eval_env, 'get_risk_level'):
                risk_levels.append(eval_env.get_risk_level())

            # Handle episode termination
            if any(dones.values()) or any(truncs.values()):
                obs, _ = eval_env.reset()

        except Exception as e:
            print(f"⚠️ Step execution error: {e}")
            break

    print("✅ Agent evaluation completed")

    # Calculate metrics
    metrics = calculate_performance_metrics(portfolio_values, rewards_by_agent, risk_levels, evaluation_steps)
    metrics['evaluation_mode'] = 'agents'

    return metrics


def run_comprehensive_evaluation(eval_data: pd.DataFrame, args) -> Dict[str, Any]:
    """Run comprehensive evaluation comparing all configurations with separate environments."""
    print_section_header("COMPREHENSIVE EVALUATION: All Configurations")
    print_progress("🚀 Starting comprehensive evaluation of 5 systems...")

    comprehensive_results = {
        'evaluation_type': 'comprehensive',
        'timestamp': datetime.now().isoformat(),
        'eval_data_path': args.eval_data,
        'eval_steps': args.eval_steps,
        'configurations': {}
    }

    # 1. Evaluate Baselines (use basic environment)
    print_progress("📊 [1/3] EVALUATING BASELINES...", 1, 3)
    try:
        baseline_results = run_traditional_baselines(args.eval_data, args.eval_steps or 8000, args.output_dir)
        if baseline_results:
            comprehensive_results['configurations']['baselines'] = baseline_results
            print_progress("✅ Baseline evaluation completed")
        else:
            print_progress("❌ Baseline evaluation failed")
            comprehensive_results['configurations']['baselines'] = {'error': 'Baseline evaluation failed'}
    except Exception as e:
        print_progress(f"❌ Baseline evaluation error: {e}")
        comprehensive_results['configurations']['baselines'] = {'error': str(e)}

    # 2. Evaluate Normal Models (no forecasts, no DL overlay) - Create basic environment
    print_progress("🤖 [2/3] EVALUATING NORMAL MODELS (No Forecasts/DL Overlay)...", 2, 3)
    try:
        if os.path.exists(args.normal_models):
            # Create basic environment for normal models (no forecasting)
            print_progress("🏗️ Creating basic environment for normal models...")
            normal_eval_env, _ = create_evaluation_environment(
                eval_data,
                enable_forecasting=False,  # No forecasting for normal models
                model_dir=args.model_dir,
                scaler_dir=args.scaler_dir,
                output_dir=args.output_dir,
                fail_fast=True,
            )

            if normal_eval_env:
                # Load normal models without enhanced features
                print_progress("🔄 Loading normal agent system...")
                normal_agent_system = load_agent_system(args.normal_models, normal_eval_env, enhanced=False)
                if normal_agent_system:
                    print_progress("🔄 Running normal agent evaluation...")
                    normal_results = run_agent_evaluation(normal_agent_system, normal_eval_env, eval_data, args.eval_steps)
                    if normal_results:
                        normal_results['model_type'] = 'normal'
                        normal_results['features'] = {'forecasting': False, 'dl_overlay': False}
                        comprehensive_results['configurations']['normal_agents'] = normal_results
                        print_progress("✅ Normal agent evaluation completed")
                    else:
                        print_progress("❌ Normal agent evaluation failed")
                        comprehensive_results['configurations']['normal_agents'] = {'error': 'Normal agent evaluation failed'}
                else:
                    print_progress("❌ Failed to load normal agent system")
                    comprehensive_results['configurations']['normal_agents'] = {'error': 'Failed to load normal agent system'}
            else:
                print_progress("❌ Failed to create basic evaluation environment")
                comprehensive_results['configurations']['normal_agents'] = {'error': 'Failed to create basic evaluation environment'}
        else:
            print_progress(f"❌ Normal models directory not found: {args.normal_models}")
            comprehensive_results['configurations']['normal_agents'] = {'error': f'Directory not found: {args.normal_models}'}
    except Exception as e:
        print_progress(f"❌ Normal agent evaluation error: {e}")
        comprehensive_results['configurations']['normal_agents'] = {'error': str(e)}

    # 3. Evaluate Full Models (with forecasts and DL overlay) - Create enhanced environment
    print_progress("🚀 [3/3] EVALUATING FULL MODELS (With Forecasts + DL Overlay)...", 3, 3)
    try:
        if os.path.exists(args.full_models):
            # Create enhanced evaluation environment with DL overlay
            print_progress("🏗️ Creating enhanced environment for full models...")
            dl_weights_path = _resolve_overlay_weights_from_models_dir(args.full_models)
            enhanced_eval_env, enhanced_forecaster = create_evaluation_environment(
                eval_data,
                enable_forecasting=True,  # Enable forecasting for full models
                model_dir=args.model_dir,
                scaler_dir=args.scaler_dir,
                dl_overlay_weights_path=dl_weights_path,
                output_dir=args.output_dir,
                fail_fast=True,
            )

            if enhanced_eval_env:
                # Load enhanced models with all features
                print_progress("🔄 Loading enhanced agent system...")
                full_agent_system = load_enhanced_agent_system(args.full_models, enhanced_eval_env)
                if full_agent_system:
                    print_progress("🔄 Running full agent evaluation...")
                    full_results = run_agent_evaluation(full_agent_system, enhanced_eval_env, eval_data, args.eval_steps)
                    if full_results:
                        full_results['model_type'] = 'enhanced'
                        full_results['features'] = {'forecasting': True, 'dl_overlay': True}
                        comprehensive_results['configurations']['full_agents'] = full_results
                        print_progress("✅ Full agent evaluation completed")
                    else:
                        print_progress("❌ Full agent evaluation failed")
                        comprehensive_results['configurations']['full_agents'] = {'error': 'Full agent evaluation failed'}
                else:
                    print_progress("❌ Failed to load full agent system")
                    comprehensive_results['configurations']['full_agents'] = {'error': 'Failed to load full agent system'}
            else:
                print_progress("❌ Failed to create enhanced evaluation environment")
                comprehensive_results['configurations']['full_agents'] = {'error': 'Failed to create enhanced evaluation environment'}
        else:
            print_progress(f"❌ Full models directory not found: {args.full_models}")
            comprehensive_results['configurations']['full_agents'] = {'error': f'Directory not found: {args.full_models}'}
    except Exception as e:
        print_progress(f"❌ Full agent evaluation error: {e}")
        comprehensive_results['configurations']['full_agents'] = {'error': str(e)}

    print_progress("🎉 Comprehensive evaluation completed!")
    return comprehensive_results

    return comprehensive_results


def save_results(results: Dict[str, Any], output_dir: str, mode: str, analysis: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
    """Save evaluation results and analysis to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save main results
    results_file = os.path.join(output_dir, f"evaluation_{mode}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {results_file}")

    # Save analysis if provided
    analysis_file = None
    if analysis:
        analysis_file = os.path.join(output_dir, f"analysis_{mode}_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"📊 Analysis saved to: {analysis_file}")

    return results_file, analysis_file


def analyze_comprehensive_results(comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze comprehensive evaluation results across all configurations."""
    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("="*50)

    analysis = {
        'summary': {},
        'performance_ranking': [],
        'feature_impact': {},
        'statistical_comparison': {}
    }

    configurations = comprehensive_results.get('configurations', {})

    # Extract performance metrics for comparison
    performance_data = {}

    # ========================================
    # 📊 FINAL PORTFOLIO VALUES TABLE
    # ========================================
    print("\n" + "=" * 80)
    print("📊 FINAL PORTFOLIO VALUES - ALL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<4} {'Configuration':<25} {'Final Value':<15} {'Return':<10} {'Sharpe':<8} {'Features':<20}")
    print("-" * 80)

    for config_name, config_results in configurations.items():
        if 'error' in config_results:
            print(f"⚠️ {config_name}: {config_results['error']}")
            continue

        # Extract key metrics
        if config_name == 'baselines':
            # Handle baseline results structure
            for baseline_name, baseline_data in config_results.items():
                if isinstance(baseline_data, dict) and baseline_data.get('status') != 'failed':
                    config_display_name = baseline_data.get('method', baseline_name)

                    # Extract portfolio values correctly
                    final_value = baseline_data.get('final_portfolio_value', baseline_data.get('final_value_usd', 800_000_000))
                    initial_value = baseline_data.get('initial_portfolio_value', baseline_data.get('initial_value_usd', 800_000_000))
                    total_return = baseline_data.get('total_return', 0) * 100  # Convert to percentage

                    performance_data[config_display_name] = {
                        'total_return': total_return,
                        'sharpe_ratio': baseline_data.get('sharpe_ratio', 0),
                        'max_drawdown': baseline_data.get('max_drawdown', 0) * 100,  # Convert to percentage
                        'final_value_usd': final_value,
                        'initial_value_usd': initial_value,
                        'type': 'baseline'
                    }
        else:
            # Handle agent results structure
            if 'total_return' in config_results or 'final_portfolio_value' in config_results:
                features = config_results.get('features', {})
                config_display_name = "Normal Agents" if config_name == 'normal_agents' else "Full Agents"

                # Extract portfolio values correctly
                final_value = config_results.get('final_portfolio_value', 800_000_000)
                initial_value = config_results.get('initial_portfolio_value', 800_000_000)
                total_return = config_results.get('total_return', 0) * 100  # Convert to percentage

                performance_data[config_display_name] = {
                    'total_return': total_return,
                    'sharpe_ratio': config_results.get('sharpe_ratio', 0),
                    'max_drawdown': config_results.get('max_drawdown', 0) * 100,  # Convert to percentage
                    'final_value_usd': final_value,
                    'initial_value_usd': initial_value,
                    'type': 'agent',
                    'forecasting': features.get('forecasting', False),
                    'dl_overlay': features.get('dl_overlay', False)
                }

    # Sort by final portfolio value for the main table
    ranked_by_value = sorted(performance_data.items(), key=lambda x: x[1]['final_value_usd'], reverse=True)

    for i, (config_name, metrics) in enumerate(ranked_by_value, 1):
        final_value = metrics['final_value_usd']
        initial_value = metrics['initial_value_usd']
        return_pct = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0.0

        # Format features
        features_str = ""
        if metrics['type'] == 'agent':
            features = []
            if metrics.get('forecasting'): features.append("Forecasting")
            if metrics.get('dl_overlay'): features.append("DL Overlay")
            features_str = ', '.join(features) if features else 'Basic RL'
        else:
            features_str = 'Traditional'

        # Color coding for top performers
        rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

        print(f"{rank_symbol:<4} {config_name:<25} ${final_value/1e6:>10.1f}M {return_pct:>+7.2f}% {metrics['sharpe_ratio']:>6.3f} {features_str:<20}")

    print("-" * 80)
    print(f"{'Initial Portfolio Value:':<41} $800.0M")
    if ranked_by_value:
        print(f"{'Best Performer:':<41} {ranked_by_value[0][0]} (${ranked_by_value[0][1]['final_value_usd']/1e6:.1f}M)")
        improvement = ((ranked_by_value[0][1]['final_value_usd'] - 800_000_000) / 800_000_000 * 100)
        print(f"{'Best Return:':<41} +{improvement:.2f}%")
    print("=" * 80)

    # Rank by Sharpe ratio for detailed analysis
    ranked_configs = sorted(performance_data.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)

    print("\n🏆 PERFORMANCE RANKING (by Sharpe Ratio):")
    for i, (config_name, metrics) in enumerate(ranked_configs, 1):
        features_str = ""
        if metrics['type'] == 'agent':
            features = []
            if metrics.get('forecasting'): features.append("Forecasting")
            if metrics.get('dl_overlay'): features.append("DL Overlay")
            features_str = f" ({', '.join(features) if features else 'Basic'})"

        print(f"   {i}. {config_name}{features_str}")
        print(f"      Return: {metrics['total_return']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | Drawdown: {metrics['max_drawdown']:.2f}%")

    analysis['performance_ranking'] = ranked_configs

    # Feature impact analysis
    if len([x for x in performance_data.values() if x['type'] == 'agent']) >= 2:
        print("\n🔬 FEATURE IMPACT ANALYSIS:")

        # Find normal vs full agents
        normal_metrics = None
        full_metrics = None

        for config_name, metrics in performance_data.items():
            if metrics['type'] == 'agent':
                if not metrics.get('forecasting') and not metrics.get('dl_overlay'):
                    normal_metrics = metrics
                elif metrics.get('forecasting') and metrics.get('dl_overlay'):
                    full_metrics = metrics

        if normal_metrics and full_metrics:
            return_improvement = full_metrics['total_return'] - normal_metrics['total_return']
            sharpe_improvement = full_metrics['sharpe_ratio'] - normal_metrics['sharpe_ratio']

            print(f"   📈 Enhanced Features Impact:")
            print(f"      Return Improvement: {return_improvement:+.2f}%")
            print(f"      Sharpe Improvement: {sharpe_improvement:+.3f}")

            analysis['feature_impact'] = {
                'return_improvement_pct': return_improvement,
                'sharpe_improvement': sharpe_improvement,
                'normal_performance': normal_metrics,
                'enhanced_performance': full_metrics
            }

    # Best performer summary
    if ranked_configs:
        best_config = ranked_configs[0]
        analysis['summary']['best_performer'] = {
            'name': best_config[0],
            'metrics': best_config[1]
        }

        print(f"\n🥇 BEST PERFORMER: {best_config[0]}")
        print(f"   Sharpe Ratio: {best_config[1]['sharpe_ratio']:.3f}")
        print(f"   Total Return: {best_config[1]['total_return']:.2f}%")

    return analysis


def main():
    """Main evaluation function."""
    setup_console_encoding()
    print_progress("🚀 Starting Comprehensive Evaluation System")
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["checkpoint", "agents", "baselines", "compare", "comprehensive", "tiers"],
        required=True,
        help=(
            "Evaluation mode: "
            "'checkpoint' for latest checkpoint, "
            "'agents' for custom agent directory, "
            "'baselines' for traditional methods, "
            "'compare' for comprehensive comparison, "
            "'comprehensive' for all configurations, "
            "'tiers' to evaluate Baseline vs FGB-online vs FGB-meta (unseen data) using final_models/"
        ),
    )

    # Data and model paths
    parser.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv",
                       help="Path to evaluation data CSV")
    parser.add_argument("--trained_agents", type=str, default=None,
                       help="Directory with saved agent policies (required for 'agents' mode)")
    parser.add_argument("--checkpoint_dir", type=str, default="normal/checkpoints",
                       help="Base directory for checkpoints (for 'checkpoint' mode)")

    # Forecasting options
    parser.add_argument("--model_dir", type=str, default="Forecast_ANN/models",
                       help="Forecast model directory (NEW: Updated to Forecast_ANN structure)")
    parser.add_argument("--scaler_dir", type=str, default="Forecast_ANN/scalers",
                       help="Forecast scaler directory (NEW: Updated to Forecast_ANN structure)")
    parser.add_argument("--no_forecast", action="store_true",
                       help="Disable forecasting for baseline comparison")


    # Enhanced model support
    parser.add_argument("--enhanced_models", type=str, default=None,
                       help="Directory with enhanced models (forecasting + DL overlay enabled)")
    parser.add_argument("--force_forecasting", action="store_true",
                       help="Force enable forecasting for enhanced models")

    # Comprehensive evaluation paths
    parser.add_argument("--normal_models", type=str, default="normal/final_models",
                       help="Directory with normal models (agents without forecasts/DL overlay)")
    parser.add_argument("--full_models", type=str, default="full/final_models",
                       help="Directory with full models (agents with forecasts and DL overlay)")

    # Evaluation options
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="Number of timesteps to evaluate (default: auto)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--investment_freq", type=int, default=6,
                       help="Investor action frequency for evaluation (should match training; default 6)")

    # Tier-suite evaluation (paper modes on unseen data)
    parser.add_argument("--tier1_dir", type=str, default="tier1_seed789",
                       help="Baseline run directory containing final_models/")
    parser.add_argument("--tier2_dir", type=str, default="tier2_seed789",
                       help="FGB-online run directory containing final_models/")
    parser.add_argument("--tier3_dir", type=str, default="tier3_seed789",
                       help="FGB-meta run directory containing final_models/")
    parser.add_argument(
        "--tiers_only",
        type=str,
        default="all",
        choices=[
            "all",
            "baseline",
            "fgb_online",
            "fgb_meta",
            "tier1",
            "tier2",
            "tier3",
            "fgb_online_no_forecast",
            "fgb_meta_no_forecast",
            "tier2_no_forecast",
            "tier3_no_forecast",
        ],
        help=(
            "Run only a subset in --mode tiers "
            "(baseline / fgb_online / fgb_meta / fgb_online_no_forecast / fgb_meta_no_forecast). "
            "Legacy aliases: tier1/2/3 and tier2_no_forecast/tier3_no_forecast."
        ),
    )
    parser.add_argument(
        "--tiers_eval_mode",
        type=str,
        default="paper",
        choices=["paper", "deployed"],
        help=(
            "In --mode tiers: "
            "'paper' evaluates the learned SB3 policies only (forecasts+DL overlay disabled; paper-fair). "
            "'deployed' enables forecasting and loads DL overlay .h5 weights (system-level evaluation; NOT paper-fair). "
            "In deployed mode, Tier-2/Tier-3 now fail fast on forecast/overlay setup errors. "
            "NOTE: With the current implementation, the overlay is a training-time baseline provider and does not "
            "directly override actions, so NAV may be unchanged; this mode is mainly for validating the deployed pipeline."
        ),
    )
    parser.add_argument(
        "--forecast_base_dir",
        type=str,
        default="forecast_models",
        help="Base directory that contains episode-specific forecast models (e.g., forecast_models/episode_20/...). Used in --tiers_eval_mode deployed.",
    )
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Base directory for forecast caches. Used in --tiers_eval_mode deployed.",
    )
    parser.add_argument(
        "--tiers_precompute_forecasts",
        action="store_true",
        help="When --tiers_eval_mode deployed: precompute/load offline forecast cache for the evaluation dataset (recommended for speed/determinism).",
    )
    parser.add_argument(
        "--tiers_precompute_batch_size",
        type=int,
        default=8192,
        help="Batch size for offline forecast precompute when --tiers_precompute_forecasts is set.",
    )

    # Analysis options
    parser.add_argument("--analyze", action="store_true",
                       help="Perform comprehensive portfolio analysis")
    parser.add_argument("--plot", action="store_true",
                       help="Generate performance plots (requires --analyze)")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots to files instead of displaying (requires --plot)")

    args = parser.parse_args()

    # === RUN MANIFEST (REPRODUCIBILITY) ===
    # Write a lightweight JSON manifest capturing argv/args + a config snapshot + git hash (if available).
    # This is research-release hygiene and does not affect evaluation results.
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        from run_manifest import write_run_manifest
        try:
            cfg_snapshot = EnhancedConfig()
            # Reflect key CLI overrides for reproducibility.
            if getattr(args, "investment_freq", None) is not None:
                cfg_snapshot.investment_freq = int(args.investment_freq)
        except Exception as snapshot_error:
            print_progress(
                f"⚠️  Could not build config snapshot for evaluation manifest: {snapshot_error}"
            )
            cfg_snapshot = None
        manifest_path = write_run_manifest(
            args.output_dir,
            argv=sys.argv,
            args=args,
            config=cfg_snapshot,
            extra={"entrypoint": "evaluation.py", "mode": getattr(args, "mode", None)},
            filename_prefix="eval_manifest",
        )
        print_progress(f"🧾 Wrote evaluation run manifest: {manifest_path}")
    except Exception as e:
        print_progress(f"⚠️  Failed to write evaluation run manifest: {e}")

    print_progress("📋 Parsing arguments and validating paths...")

    # Auto-detect trained agents for agents mode if not specified
    if args.mode == "agents" and args.trained_agents is None and args.enhanced_models is None:
        # Try common locations
        candidate_dirs = [
            "normal/final_models",
            "enhanced/final_models",  # New location for enhanced models
            "training_agent_results/final_models",
            "final_models",
            "saved_agents"
        ]

        for candidate in candidate_dirs:
            if os.path.exists(candidate):
                required_files = [
                    "investor_0_policy.zip", "battery_operator_0_policy.zip",
                    "risk_controller_0_policy.zip", "meta_controller_0_policy.zip"
                ]
                if all(os.path.exists(os.path.join(candidate, f)) for f in required_files):
                    # Check if this is an enhanced model directory
                    is_enhanced = False

                    # Method 1: Check directory name
                    if "enhanced" in candidate.lower() or args.force_forecasting:
                        is_enhanced = True

                    # Method 2: Check training config for enhanced features
                    config_file = os.path.join(candidate, "training_config.json")
                    if os.path.exists(config_file):
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            enhanced_features = config.get('enhanced_features', {})
                            if enhanced_features.get('forecasting_enabled') or enhanced_features.get('dl_overlay_enabled'):
                                is_enhanced = True
                        except Exception as e:
                            print(f"⚠️ Could not parse training config for auto-detection ({config_file}): {e}")

                    if is_enhanced:
                        args.enhanced_models = candidate
                        print(f"🎯 Auto-detected enhanced models: {candidate}")
                    else:
                        args.trained_agents = candidate
                        print(f"🎯 Auto-detected trained agents: {candidate}")
                    break

        if args.trained_agents is None and args.enhanced_models is None:
            print("❌ --trained_agents or --enhanced_models is required when using 'agents' mode")
            print("   Searched locations: normal/final_models, enhanced/final_models, training_agent_results/final_models")
            sys.exit(1)

    print(f"🔥 UNIFIED EVALUATION - MODE: {args.mode.upper()}")
    print("=" * 60)

    # Load evaluation data
    print_progress(f"📊 Loading evaluation data from: {args.eval_data}")
    try:
        eval_data = load_energy_data(args.eval_data)
        print_progress(f"✅ Loaded evaluation data: {eval_data.shape}")
        if "timestamp" in eval_data.columns and eval_data["timestamp"].notna().any():
            ts = eval_data["timestamp"].dropna()
            print_progress(f"📅 Date range: {ts.iloc[0]} → {ts.iloc[-1]}")
    except Exception as e:
        print_progress(f"❌ Error loading evaluation data: {e}")
        sys.exit(1)

    # Tier suite mode evaluates baseline vs FGB-online vs FGB-meta on unseen data using final_models/
    if args.mode == "tiers":
        try:
            results = run_tier_suite_evaluation(eval_data, args)
            # Always write a single consolidated report (CSV + Markdown)
            csv_path, md_path = write_tier_comparison_report(results.get("tiers", {}), args.output_dir)
            results["tier_comparison_csv"] = csv_path
            results["tier_comparison_md"] = md_path

            analysis = None
            if args.analyze:
                analysis = analyze_comprehensive_results({"configurations": results.get("tiers", {})})

            save_results(results, args.output_dir, mode="tiers", analysis=analysis)
            print_progress(f"📄 Tier comparison report written: {md_path}")
            print_progress(f"📄 Tier comparison CSV written: {csv_path}")
            return
        except Exception as e:
            print_progress(f"❌ Tier suite evaluation failed: {e}")
            raise

    # Create evaluation environment for other modes
    enable_forecasting = (not args.no_forecast) and (args.mode not in ["agents", "checkpoint", "compare"])

    eval_env, forecaster = create_evaluation_environment(
        eval_data,
        enable_forecasting=enable_forecasting,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
        output_dir=args.output_dir,
        investment_freq=args.investment_freq,
        config=None,
        fail_fast=True,
    )

    if eval_env is None:
        print("❌ Failed to create evaluation environment")
        sys.exit(1)

    # Run evaluation based on mode
    results = None

    if args.mode == "checkpoint":
        # Checkpoint mode: find latest checkpoint and load SB3 models
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint is None:
            print("❌ No checkpoints found")
            sys.exit(1)

        checkpoint_models = load_checkpoint_models(latest_checkpoint)
        if not checkpoint_models or not any(m is not None for m in checkpoint_models.values()):
            print("❌ No checkpoint models loaded successfully")
            sys.exit(1)

        results = run_checkpoint_evaluation(
            checkpoint_models,
            eval_env,
            eval_data,
            args.eval_steps or 8000
        )

        if results:
            results['checkpoint_path'] = latest_checkpoint

    elif args.mode == "agents":
        # Agents mode: load MultiESGAgent system (standard or enhanced)

        # Determine which model type to load
        if args.enhanced_models:
            if not os.path.exists(args.enhanced_models):
                print(f"❌ Enhanced models directory not found: {args.enhanced_models}")
                sys.exit(1)

            print("🚀 Loading enhanced models with forecasting and DL overlay...")

            # Create enhanced evaluation environment with DL overlay
            dl_weights_path = _resolve_overlay_weights_from_models_dir(args.enhanced_models)
            if dl_weights_path and os.path.exists(dl_weights_path):
                enhanced_eval_env, enhanced_forecaster = create_evaluation_environment(
                    eval_data,
                    enable_forecasting=True,
                    dl_overlay_weights_path=dl_weights_path,
                    output_dir=args.output_dir,
                    fail_fast=True,
                )
                eval_env = enhanced_eval_env if enhanced_eval_env else eval_env

            agent_system = load_enhanced_agent_system(args.enhanced_models, eval_env)
            model_path = args.enhanced_models
        else:
            if not os.path.exists(args.trained_agents):
                print(f"❌ Trained agents directory not found: {args.trained_agents}")
                sys.exit(1)

            agent_system = load_agent_system(args.trained_agents, eval_env)
            model_path = args.trained_agents

        if agent_system is None:
            print("❌ Failed to load agent system")
            sys.exit(1)

        results = run_agent_evaluation(
            agent_system,
            eval_env,
            eval_data,
            args.eval_steps
        )

        if results:
            results['trained_agents_path'] = model_path
            results['model_type'] = 'enhanced' if args.enhanced_models else 'standard'

    elif args.mode == "baselines":
        # Baselines mode: run traditional baseline methods
        print("🏛️ Running traditional baseline evaluation...")

        baseline_results = run_traditional_baselines(
            args.eval_data,
            args.eval_steps or 10000,  # Use same default as agents
            args.output_dir
        )

        if baseline_results and 'error' not in baseline_results:
            results = {
                'evaluation_type': 'traditional_baselines',
                'baselines': baseline_results,
                'eval_data_path': args.eval_data,
                'eval_steps': args.eval_steps or 8000
            }
        else:
            print("❌ Traditional baseline evaluation failed")
            sys.exit(1)

    elif args.mode == "compare":
        # Compare mode: run both AI and traditional baselines
        print("🔄 Running comprehensive comparison...")

        # Initialize ai_results
        ai_results = None

        # Run AI evaluation first (enhanced or standard models)
        if args.enhanced_models:
            # Use enhanced models with DL overlay
            dl_weights_path = _resolve_overlay_weights_from_models_dir(args.enhanced_models)
            if dl_weights_path and os.path.exists(dl_weights_path):
                enhanced_eval_env, enhanced_forecaster = create_evaluation_environment(
                    eval_data,
                    enable_forecasting=True,
                    dl_overlay_weights_path=dl_weights_path,
                    output_dir=args.output_dir,
                    fail_fast=True,
                )
                eval_env = enhanced_eval_env if enhanced_eval_env else eval_env

            agent_system = load_enhanced_agent_system(args.enhanced_models, eval_env)
            if agent_system:
                ai_results = run_agent_evaluation(agent_system, eval_env, eval_data, args.eval_steps)
                if ai_results:
                    ai_results['model_type'] = 'enhanced'
                    ai_results['trained_agents_path'] = args.enhanced_models
        elif args.trained_agents:
            # Use standard models
            agent_system = load_agent_system(args.trained_agents, eval_env)
            if agent_system:
                ai_results = run_agent_evaluation(agent_system, eval_env, eval_data, args.eval_steps)
                if ai_results:
                    ai_results['model_type'] = 'standard'
                    ai_results['trained_agents_path'] = args.trained_agents
        else:
            # Auto-detect or use latest checkpoint
            latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
            if latest_checkpoint and "final_models" in latest_checkpoint:
                args.trained_agents = latest_checkpoint
                agent_system = load_agent_system(args.trained_agents, eval_env)
                if agent_system:
                    ai_results = run_agent_evaluation(agent_system, eval_env, eval_data, args.eval_steps)
                    if ai_results:
                        ai_results['model_type'] = 'standard'
                        ai_results['trained_agents_path'] = args.trained_agents
            elif latest_checkpoint:
                checkpoint_models = load_checkpoint_models(latest_checkpoint)
                if checkpoint_models:
                    ai_results = run_checkpoint_evaluation(checkpoint_models, eval_env, eval_data, args.eval_steps or 8000)
            else:
                print("❌ No AI models found for comparison")

        # Run traditional baselines with same steps as AI agents
        baseline_results = run_traditional_baselines(args.eval_data, args.eval_steps or 10000, args.output_dir)

        # Combine results
        if ai_results and baseline_results and 'error' not in baseline_results:
            results = {
                'evaluation_type': 'comprehensive_comparison',
                'ai_results': ai_results,
                'baseline_results': baseline_results,
                'eval_data_path': args.eval_data,
                'eval_steps': args.eval_steps or 8000
            }
        elif ai_results and 'error' in baseline_results:
            print("⚠️ Traditional baselines failed, showing AI results only")
            results = ai_results
            results['baseline_error'] = baseline_results.get('error', 'Unknown error')
        elif baseline_results and 'error' not in baseline_results and not ai_results:
            print("⚠️ AI evaluation failed, showing baseline results only")
            results = {
                'evaluation_type': 'baselines_only',
                'baseline_results': baseline_results,
                'ai_error': 'AI evaluation failed'
            }
        else:
            print("❌ Both AI and baseline evaluations failed")
            sys.exit(1)

    elif args.mode == "comprehensive":
        # Comprehensive mode: evaluate all configurations
        print("🎯 Running comprehensive evaluation of all configurations...")

        results = run_comprehensive_evaluation(eval_data, args)

        if not results or not results.get('configurations'):
            print("❌ Comprehensive evaluation failed")
            sys.exit(1)

    # Save and display results
    if results:
        # Add common metadata
        results['eval_data_path'] = args.eval_data
        results['forecasting_enabled'] = enable_forecasting
        results['eval_steps_requested'] = args.eval_steps

        # Perform analysis if requested
        analysis_results = None
        if args.analyze:
            print("\n📊 PERFORMING COMPREHENSIVE ANALYSIS...")

            # Handle different result types
            if results.get('evaluation_type') == 'comprehensive':
                # For comprehensive results, use specialized analysis
                analysis_results = analyze_comprehensive_results(results)
            elif results.get('evaluation_type') == 'comprehensive_comparison':
                # For comparison results, analyze the AI results part
                ai_results = results.get('ai_results', {})
                if ai_results:
                    analyzer = PortfolioAnalyzer(ai_results)
                    analysis_results = analyzer.analyze_performance(
                        make_plots=args.plot,
                        save_plots=args.save_plots,
                        output_dir=args.output_dir
                    )
                    # Add baseline comparison info to analysis
                    if 'baseline_results' in results:
                        analysis_results['baseline_comparison'] = results['baseline_results']
                else:
                    # Fallback for comparison without AI results
                    analysis_results = {
                        'basic_metrics': {'total_return': 0.0, 'sharpe_ratio': 0.0},
                        'summary': {'overall_assessment': 'MODERATE', 'performance_grade': 'C'}
                    }
            else:
                # For regular results (agents, baselines, checkpoints)
                analyzer = PortfolioAnalyzer(results)
                analysis_results = analyzer.analyze_performance(
                    make_plots=args.plot,
                    save_plots=args.save_plots,
                    output_dir=args.output_dir
                )

        # Save results and analysis
        results_file, analysis_file = save_results(results, args.output_dir, args.mode, analysis_results)

        # Display summary
        print(f"\n🎉 EVALUATION SUCCESS!")

        # Handle different result types for display
        if results.get('evaluation_type') == 'comprehensive':
            # For comprehensive results, show summary from analysis
            if analysis_results and 'summary' in analysis_results:
                best_performer = analysis_results['summary'].get('best_performer', {})
                if best_performer:
                    print(f"🥇 BEST PERFORMER: {best_performer['name']}")
                    metrics = best_performer['metrics']
                    print(f"   📈 Return: {metrics['total_return']:.2f}%")
                    print(f"   ⚡ Sharpe: {metrics['sharpe_ratio']:.3f}")
                    print(f"   📉 Drawdown: {metrics['max_drawdown']:.2f}%")

                if 'feature_impact' in analysis_results:
                    impact = analysis_results['feature_impact']
                    print(f"\n🔬 ENHANCED FEATURES IMPACT:")
                    print(f"   📈 Return Improvement: {impact['return_improvement_pct']:+.2f}%")
                    print(f"   ⚡ Sharpe Improvement: {impact['sharpe_improvement']:+.3f}")
            else:
                print("📊 Comprehensive evaluation completed - see detailed results in JSON file")

        elif results.get('evaluation_type') == 'comprehensive_comparison':
            # For comparison results, show AI results
            ai_results = results.get('ai_results', {})
            baseline_results = results.get('baseline_results', {})

            if ai_results:
                print(f"🤖 AI AGENTS PERFORMANCE:")
                print(f"📈 Final Return: {ai_results.get('total_return', 0):+.2%}")
                print(f"⚡ Sharpe Ratio: {ai_results.get('sharpe_ratio', 0):.3f}")
                print(f"📊 Total Rewards: {ai_results.get('total_rewards', 0):.2f}")

                if 'initial_portfolio_value' in ai_results and 'final_portfolio_value' in ai_results:
                    initial = ai_results['initial_portfolio_value']
                    final = ai_results['final_portfolio_value']
                    print(f"💰 Portfolio: ${initial/1e6:.1f}M → ${final/1e6:.1f}M (${(final-initial)/1e6:+.1f}M)")
                    print(f"📊 Volatility: {ai_results.get('volatility', 0)*100:.2f}%")
                    print(f"📉 Max Drawdown: {ai_results.get('max_drawdown', 0)*100:.2f}%")

            if baseline_results:
                print(f"\n🏛️ BASELINE COMPARISON:")
                for baseline_name, baseline_data in baseline_results.items():
                    if isinstance(baseline_data, dict) and 'total_return' in baseline_data:
                        method = baseline_data.get('method', baseline_name)
                        return_pct = baseline_data.get('total_return', 0) * 100
                        sharpe = baseline_data.get('sharpe_ratio', 0)
                        print(f"   {method}: {return_pct:+.2f}% return, {sharpe:.2f} Sharpe")
        else:
            # For regular results
            print(f"📈 Final Return: {results.get('total_return', 0):+.2%}")
            print(f"⚡ Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
            print(f"📊 Total Rewards: {results.get('total_rewards', 0):.2f}")
            print(f"🎯 Average Risk: {results.get('average_risk', 0):.3f}")

            # Portfolio performance details
            if 'initial_portfolio_value' in results and 'final_portfolio_value' in results:
                initial = results['initial_portfolio_value']
                final = results['final_portfolio_value']
                print(f"💰 Portfolio: ${initial/1e6:.1f}M → ${final/1e6:.1f}M (${(final-initial)/1e6:+.1f}M)")
                print(f"📊 Volatility: {results.get('volatility', 0)*100:.2f}%")
                print(f"📉 Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")

        if 'prediction_success_rate' in results:
            print(f"🎯 Prediction Success: {results['prediction_success_rate']:.1%}")
        if 'models_loaded' in results:
            print(f"🤖 Models Loaded: {results['models_loaded']}/4")

        # Display analysis summary if available
        if analysis_results and 'summary' in analysis_results:
            summary = analysis_results['summary']
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"🏆 Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
            print(f"📝 Performance Grade: {summary.get('performance_grade', 'N/A')}")

            if 'recommendations' in summary:
                print(f"💡 Recommendations:")
                for i, rec in enumerate(summary['recommendations'], 1):
                    print(f"   {i}. {rec}")

        print(f"\n💾 Files saved:")
        print(f"   📄 Results: {results_file}")
        if analysis_file:
            print(f"   📊 Analysis: {analysis_file}")

    else:
        print("❌ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
