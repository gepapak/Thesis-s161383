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
- Cache-only forecast utilization flag alignment
- Comprehensive metrics calculation
- Portfolio analysis with plotting
- Flexible data input (defaults to unseen evaluation dataset)
- JSON results and analysis reports

Usage:
    # Evaluate latest checkpoint (automatic unseen data)
    python evaluation.py --mode checkpoint

    # Evaluate specific agent directory
    python evaluation.py --mode agents --trained_agents saved_agents --eval_data "evaluation_dataset/unseendata.csv"

    # Baseline comparison
    python evaluation.py --mode agents --trained_agents saved_agents --eval_data "evaluation_dataset/unseendata.csv"

    # With comprehensive analysis and plots
    python evaluation.py --mode checkpoint --analyze --plot
"""

import argparse
import builtins as _builtins
import math
import os
import sys
import re
import warnings
import glob
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config import EnhancedConfig
from runtime_contract import (
    build_runtime_contract,
    forecast_prior_contract_settings,
    runtime_contract_hash,
)
from forecast_prior_cli import (
    add_forecast_prior_override_args,
    apply_forecast_prior_overrides,
)

# Keep evaluation aligned with training-time rolling_past bootstrap.
EVAL_ROLLING_PAST_HISTORY_DIR = "rolling_past_history_dataset"

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

# Clean ASCII overrides for the console helpers above.
def _sanitize_console_text(message: str) -> str:
    """Normalize evaluation console output to simple ASCII."""
    text = str(message)
    for _ in range(2):
        if any(ch in text for ch in ("Ã", "Â", "â")):
            try:
                repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                break
            if repaired and repaired != text:
                text = repaired
                continue
        break
    replacements = {
        "âœ…": "[OK]",
        "âš ": "[WARN]",
        "âŒ": "[FAIL]",
        "ðŸ”": "[INFO]",
        "ðŸ“‚": "[LOAD]",
        "ðŸ“": "[LOAD]",
        "ðŸ“Š": "[ANALYZE]",
        "ðŸ“ˆ": "[EVAL]",
        "ðŸ’¾": "[SAVE]",
        "ðŸŽ¯": "[GOAL]",
        "ðŸ”§": "[SETUP]",
        "â€¦": "...",
        "→": "->",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    return " ".join(text.split())


def print(*args, **kwargs):
    """Module-local print wrapper that sanitizes evaluation output."""
    cleaned = [
        _sanitize_console_text(arg) if isinstance(arg, str) else arg
        for arg in args
    ]
    return _builtins.print(*cleaned, **kwargs)


def print_progress(message: str, step: int = None, total: int = None):
    """Print progress message with optional step counter."""
    message = _sanitize_console_text(message)
    timestamp = datetime.now().strftime("%H:%M:%S")
    if step is not None and total is not None:
        progress = f"[{step}/{total}]"
        print(f"[{timestamp}] {progress} {message}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    sys.stdout.flush()


def _print_eval_loop_progress(
    completed_steps: int,
    total_steps: int,
    portfolio_values: list,
    rewards_by_agent: Dict[str, list],
    successful_inference_actions: Optional[int] = None,
    total_inference_attempts: Optional[int] = None,
):
    """Emit a live ASCII evaluation progress line."""
    if total_steps <= 0:
        return

    current_portfolio = portfolio_values[-1] if portfolio_values else 800_000_000
    portfolio_change = ((current_portfolio / 800_000_000) - 1) * 100
    total_reward = sum(sum(rewards_by_agent[agent]) for agent in rewards_by_agent)
    msg = (
        f"Progress: {completed_steps}/{total_steps} "
        f"({completed_steps / total_steps * 100:.1f}%) | "
        f"NAV: ${current_portfolio/1e6:.1f}M ({portfolio_change:+.2f}%) | "
        f"Total Reward: {total_reward:.1f}"
    )
    if successful_inference_actions is not None and total_inference_attempts is not None:
        success_rate = (
            successful_inference_actions / total_inference_attempts
            if total_inference_attempts > 0 else 0.0
        )
        msg += f" | Inference: {success_rate*100:.1f}%"
    print_progress(msg)

# Import project modules (no side effects / no prints at import time).
from environment import RenewableMultiAgentEnv
from metacontroller import MultiESGAgent
from generator import load_energy_data
from utils import configure_tf_memory

# Portfolio analysis imports
try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Matplotlib/Seaborn not available - plotting disabled")


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
            {"mode": "DQN"},  # battery_operator_0
            {"mode": "RULE"},  # risk_controller_0
            {"mode": "RULE"},  # meta_controller_0
        ]
        self.battery_action_mode = "discrete"
        self.battery_discrete_action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.battery_initial_soc = 0.50


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
        print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  PERFORMING PORTFOLIO ANALYSIS")
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
        print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Portfolio analysis completed")

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

    def _create_performance_plots(self, save_plots: bool = False, output_dir: str = "evaluation_results") -> Dict[str, str]:
        """Create performance visualization plots."""
        if not HAS_PLOTTING:
            return {'error': 'Plotting libraries not available'}

        plots_created = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(self.config.plot_title, fontsize=16, fontweight='bold')

            ax1 = axes[0, 0]
            metrics = ['Total Return', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']
            values = [
                self.results.get('total_return', 0) * 100,
                self.results.get('sharpe_ratio', 0),
                self.results.get('volatility', 0) * 100,
                self.results.get('max_drawdown', 0) * 100,
            ]
            bars = ax1.bar(metrics, values, color=['green', 'blue', 'orange', 'red'])
            ax1.set_title('Key Performance Metrics')
            ax1.set_ylabel('Value (%)')
            ax1.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.2f}%', ha='center', va='bottom')

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
                for bar, reward in zip(bars, rewards):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{reward:.1f}', ha='center', va='bottom')

            ax3 = axes[1, 0]
            risk_metrics = ['Avg Risk', 'Max Risk', 'Volatility']
            risk_values = [
                self.results.get('average_risk', 0),
                self.results.get('max_risk', 0),
                self.results.get('volatility', 0),
            ]
            ax3.bar(risk_metrics, risk_values, color='coral')
            ax3.set_title('Risk Metrics')
            ax3.set_ylabel('Risk Level')

            ax4 = axes[1, 1]
            targets = ['Baseline Target', 'AI Target', 'Actual Return']
            target_values = [
                self.config.target_baseline_return * 100,
                self.config.target_ai_return * 100,
                self.results.get('total_return', 0) * 100,
            ]
            bars = ax4.bar(targets, target_values, color=['gray', 'lightblue', 'green'])
            ax4.set_title('Performance vs Targets')
            ax4.set_ylabel('Return (%)')
            ax4.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, target_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.2f}%', ha='center', va='bottom')

            plt.tight_layout()

            if save_plots:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_created['performance_plot'] = plot_path
                print(f"Performance plot saved: {plot_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"Error creating plots: {e}")
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

        inference_rate = analysis.get('ai_analysis', {}).get('action_inference_success_rate', None)
        if inference_rate is not None and inference_rate < 80:
            recommendations.append("Investigate policy inference failures during evaluation")

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

        # Model inference reliability
        inference_rate = _extract_action_inference_success_rate(self.results)
        if inference_rate is not None:
            ai_analysis['action_inference_success_rate'] = inference_rate * 100
            ai_analysis['action_inference_reliability'] = (
                'excellent' if inference_rate > 0.9 else
                'good' if inference_rate > 0.7 else
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


def _annual_rate_to_step_rate(annual_rate: float, periods_per_year: float) -> float:
    annual = float(annual_rate)
    periods = float(max(periods_per_year, 1.0))
    if not np.isfinite(annual) or not np.isfinite(periods):
        return 0.0
    if annual <= -1.0:
        return -1.0
    return float((1.0 + annual) ** (1.0 / periods) - 1.0)


def _infer_periods_per_year(timestamps, default_periods_per_year: float = 52560.0) -> float:
    try:
        if timestamps is None:
            return float(default_periods_per_year)
        ts = pd.Series(timestamps).dropna()
        if ts.empty:
            return float(default_periods_per_year)
        parsed = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
        if parsed.size < 2:
            return float(default_periods_per_year)
        deltas = parsed.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if deltas.size == 0:
            return float(default_periods_per_year)
        median_seconds = float(np.median(deltas))
        if median_seconds <= 0.0:
            return float(default_periods_per_year)
        seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0
        inferred = seconds_per_year / median_seconds
        return float(np.clip(inferred, 1.0, 60.0 * 24.0 * 365.25))
    except Exception:
        return float(default_periods_per_year)


def _compute_annualized_risk_metrics(
    returns,
    *,
    periods_per_year: float,
    annual_risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    vals = np.asarray(returns, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    periods = float(max(periods_per_year, 1.0))
    if vals.size == 0:
        return {
            "step_mean_return": 0.0,
            "step_volatility": 0.0,
            "annualized_volatility": 0.0,
            "annualized_sharpe": 0.0,
            "per_step_risk_free_rate": _annual_rate_to_step_rate(annual_risk_free_rate, periods),
        }

    step_mean = float(np.mean(vals))
    step_vol = float(np.std(vals)) if vals.size > 1 else 0.0
    step_rf = _annual_rate_to_step_rate(annual_risk_free_rate, periods)
    excess_mean = float(np.mean(vals - step_rf))
    annualized_vol = float(step_vol * math.sqrt(periods))
    annualized_sharpe = float((excess_mean / step_vol) * math.sqrt(periods)) if step_vol > 0.0 else 0.0
    return {
        "step_mean_return": step_mean,
        "step_volatility": step_vol,
        "annualized_volatility": annualized_vol,
        "annualized_sharpe": annualized_sharpe,
        "per_step_risk_free_rate": float(step_rf),
    }


def _resolve_eval_risk_free_rate(eval_env, default: float = 0.02) -> float:
    try:
        cfg = getattr(eval_env, "config", None)
        if cfg is not None:
            val = getattr(cfg, "risk_free_rate", None)
            if val is not None:
                return float(val)
        wrapped = getattr(eval_env, "env", None)
        if wrapped is not None:
            cfg = getattr(wrapped, "config", None)
            if cfg is not None:
                val = getattr(cfg, "risk_free_rate", None)
                if val is not None:
                    return float(val)
    except Exception:
        pass
    return float(default)

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

    print("ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬â€ÃƒÂ¯Ã‚Â¸Ã‚Â Creating Buy-and-Hold baseline (passive cash/bonds strategy)...")

    # Load data
    data = pd.read_csv(eval_data_path)
    timesteps = min(timesteps, len(data))

    # Initial portfolio value (same as RL agents)
    initial_value = 800_000_000  # $800M USD
    current_value = initial_value

    # Conservative risk-free rate (Danish government bonds ~2% annually)
    annual_risk_free_rate = 0.02
    periods_per_year = _infer_periods_per_year(data.get("timestamp"))
    timestep_rate = _annual_rate_to_step_rate(annual_risk_free_rate, periods_per_year)

    # Track portfolio values
    portfolio_values = [initial_value]

    for t in range(timesteps):
        # Conservative risk-free return (compound interest)
        current_value = current_value * (1 + timestep_rate)
        portfolio_values.append(current_value)

    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value

    risk_stats = _compute_annualized_risk_metrics(
        np.diff(portfolio_values) / np.array(portfolio_values[:-1]),
        periods_per_year=periods_per_year,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    volatility = float(risk_stats["annualized_volatility"])
    sharpe_ratio = float(risk_stats["annualized_sharpe"])
    if abs(sharpe_ratio) < 1e-12:
        sharpe_ratio = 0.0

    # Max drawdown (should be 0 for risk-free investment)
    max_drawdown = 0.0

    return {
        'method': 'Buy-and-Hold Strategy',
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'step_volatility': float(risk_stats["step_volatility"]),
        'periods_per_year': float(periods_per_year),
        'annual_risk_free_rate': float(annual_risk_free_rate),
        'final_value_usd': final_value,
        'initial_value_usd': initial_value,
        'final_portfolio_value': final_value,
        'initial_portfolio_value': initial_value,
        'status': 'completed'
    }


def run_single_baseline(baseline_name: str, baseline_dir: str, eval_data_path: str, timesteps: int, output_dir: str = "evaluation_results") -> Tuple[bool, Optional[Dict]]:
    """Run a single baseline and return success status and results."""
    print(f"ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Running {baseline_name}...")

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
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Unknown baseline: {baseline_name}")
        return False, None

    script_path = os.path.join(baseline_dir, script_name)
    if not os.path.exists(script_path):
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Script not found: {script_path}")
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
            print(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ {baseline_name} completed successfully")

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
                print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â No results found for {baseline_name}")
                return True, {
                    'method': baseline_name,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'status': 'completed_no_results'
                }
        else:
            print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ {baseline_name} failed: {result.stderr}")
            return False, {'error': result.stderr}

    except subprocess.TimeoutExpired:
        print(f"ÃƒÂ¢Ã‚ÂÃ‚Â° {baseline_name} timed out")
        return False, {'error': 'timeout'}
    except Exception as e:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ {baseline_name} error: {str(e)}")
        return False, {'error': str(e)}


def run_traditional_baselines(eval_data_path: str, timesteps: int = 10000, output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """Run traditional baseline methods and return results."""
    print("ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬ÂºÃƒÂ¯Ã‚Â¸Ã‚Â Running traditional baseline methods...")

    baseline_results = {}

    # Define baselines to run
    baselines = [
        ("Traditional Portfolio", "Baseline1_TraditionalPortfolio"),
        ("Rule-Based Heuristic", "Baseline2_RuleBasedHeuristic"),
        ("Buy-and-Hold Strategy", "Baseline3_BuyAndHold")
    ]

    print("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Executing traditional baselines...")

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
                print_progress(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° {baseline_name}: ${final_val/1e6:.1f}M (${initial_val/1e6:.1f}M ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {return_pct:+.2f}%)")
            elif 'total_return' in results:
                return_pct = results['total_return']
                print_progress(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  {baseline_name}: Total return {return_pct:+.2f}%")
        else:
            baseline_results[baseline_key] = {
                'method': baseline_name,
                'status': 'failed',
                'error': results.get('error', 'unknown') if results else 'unknown'
            }

    # Check if any baselines succeeded
    successful_baselines = [k for k, v in baseline_results.items() if v.get('status') != 'failed']
    if successful_baselines:
        print_progress(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Traditional baselines completed: {len(successful_baselines)}/3 successful")

        # Summary of baseline performance
        print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Baseline Performance Summary:")
        for key, result in baseline_results.items():
            if result.get('status') != 'failed':
                method = result.get('method', 'Unknown')
                if 'final_value_usd' in result:
                    final_val = result['final_value_usd']
                    print_progress(f"   ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ {method}: ${final_val/1e6:.1f}M final value")
                elif 'total_return' in result:
                    return_pct = result['total_return']
                    print_progress(f"   ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ {method}: {return_pct:+.2f}% total return")
    else:
        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ All traditional baselines failed")

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
        ]

        all_files_exist = all(os.path.exists(os.path.join(final_models_dir, f)) for f in required_files)

        if all_files_exist:
            print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Found final models directory: {final_models_dir}")
            return final_models_dir
        else:
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Final models directory exists but missing some policy files")

    # Fallback to checkpoint detection
    if not os.path.exists(checkpoint_base_dir):
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Checkpoint directory not found: {checkpoint_base_dir}")
        return None

    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(checkpoint_base_dir, "checkpoint_*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoints found in {checkpoint_base_dir}")
        return None

    # Sort by checkpoint number (extract number from checkpoint_XXXXX)
    def get_checkpoint_number(path):
        try:
            return int(os.path.basename(path).split('_')[1])
        except (IndexError, ValueError):
            return 0

    latest_checkpoint = max(checkpoints, key=get_checkpoint_number)
    checkpoint_num = get_checkpoint_number(latest_checkpoint)

    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â Found latest checkpoint: {os.path.basename(latest_checkpoint)} (step {checkpoint_num})")
    return latest_checkpoint


def load_checkpoint_models(checkpoint_dir: str) -> Dict[str, Any]:
    """Load models from checkpoint using Stable Baselines3."""
    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¥ Loading models from checkpoint: {checkpoint_dir}")
    
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

        from stable_baselines3 import PPO, SAC, TD3, DQN
        from policy import BetaActorCriticPolicy
        print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Successfully imported stable-baselines3")

        algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DQN": DQN}
        agent_modes = {
            "investor_0": "PPO",
            "battery_operator_0": "DQN",
            "risk_controller_0": "PPO",
            "meta_controller_0": "PPO",
        }
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                final_cfg = saved.get("final_config", {}) or {}
                saved_policies = final_cfg.get("agent_policies", None)
                agent_order = [
                    "investor_0",
                    "battery_operator_0",
                    "risk_controller_0",
                    "meta_controller_0",
                ]
                if isinstance(saved_policies, list):
                    for idx, agent_name in enumerate(agent_order):
                        if idx >= len(saved_policies):
                            break
                        policy_cfg = saved_policies[idx] or {}
                        if isinstance(policy_cfg, dict):
                            agent_modes[agent_name] = str(
                                policy_cfg.get("mode", agent_modes[agent_name])
                            ).upper()
            except Exception as e:
                print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not read training_config.json for algo detection: {e}")

        _ = BetaActorCriticPolicy
        model_configs = [
            ("investor_0_policy.zip", algo_map.get(agent_modes["investor_0"], PPO)),
            ("battery_operator_0_policy.zip", algo_map.get(agent_modes["battery_operator_0"], DQN)),
            ("risk_controller_0_policy.zip", algo_map.get(agent_modes["risk_controller_0"], PPO)),
            ("meta_controller_0_policy.zip", algo_map.get(agent_modes["meta_controller_0"], PPO)),
        ]
        
        loaded_models = {}
        
        for model_file, model_class in model_configs:
            model_path = os.path.join(checkpoint_dir, model_file)
            agent_name = model_file.replace('_policy.zip', '')
            
            if os.path.exists(model_path):
                try:
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â§ Loading {model_file} with {model_class.__name__}...")
                    model = model_class.load(model_path)
                    loaded_models[agent_name] = model
                    print(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ SUCCESS: Loaded {agent_name} from checkpoint!")
                except Exception as e:
                    print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load {model_file}: {e}")
                    loaded_models[agent_name] = None
            else:
                print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Model file not found: {model_path}")
                loaded_models[agent_name] = None
        
        return loaded_models
        
    except ImportError as e:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Cannot import stable-baselines3: {e}")
        return {}
    except Exception as e:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error loading checkpoint models: {e}")
        return {}


def _resolve_reserved_eval_forecast_context(
    args,
    output_dir: Optional[str] = None,
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Resolve the canonical evaluation forecast-cache context."""

    cache_root = str(getattr(args, "forecast_cache_dir", "forecast_cache") or "forecast_cache")
    cache_dir = os.path.join(cache_root, "forecast_cache_eval_episode20_2025", "forecast_cache_eval_episode20_2025-full")
    if not os.path.isdir(cache_dir):
        cache_dir = os.path.join(cache_root, "forecast_cache_eval_episode20_2025")

    return {
        "forecast_cache_dir": cache_dir,
        "output_dir": output_dir,
    }


def _build_eval_config(args):
    from config import EnhancedConfig

    cfg = EnhancedConfig()
    if getattr(args, "investment_freq", None) is not None:
        cfg.investment_freq = int(args.investment_freq)
    if getattr(args, "meta_freq_min", None) is not None:
        cfg.meta_freq_min = int(args.meta_freq_min)
    if getattr(args, "meta_freq_max", None) is not None:
        cfg.meta_freq_max = int(args.meta_freq_max)
    if getattr(args, "global_norm_mode", None) is not None:
        cfg.use_global_normalization = bool(str(args.global_norm_mode).strip().lower() == "global")
    if not bool(getattr(cfg, "use_global_normalization", False)):
        cfg.rolling_past_history_enable = True
        roll_dir = str(getattr(args, "rolling_past_history_dir", "") or "").strip()
        cfg.rolling_past_history_dir = roll_dir or EVAL_ROLLING_PAST_HISTORY_DIR
    # Forecast utilization opt-in: typically inherited from training_config.json via
    # _hydrate_eval_config_from_training_config; CLI override accepted for
    # symmetry with training.
    if bool(getattr(args, "enable_forecast_utilization", False)):
        cfg.enable_forecast_utilization = True
    apply_forecast_prior_overrides(cfg, args)
    return cfg


def _running_moments_from_values(values):
    """Convert a historical series into the persistent online-state format."""
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        count = float(arr.size)
        mean = float(np.mean(arr))
        var = float(np.var(arr, ddof=0))
        return {
            "count": count,
            "mean": mean,
            "m2": max(var * count, 0.0),
        }
    except Exception:
        return None


def _robust_p95_scale(values, min_scale=0.1):
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return max(float(min_scale), 1.0)
        p95_val = float(np.percentile(np.abs(arr), 95))
        if np.isfinite(p95_val) and p95_val > 0.0:
            return max(p95_val, float(min_scale))
    except Exception:
        pass
    return max(float(min_scale), 1.0)


def _prime_eval_rolling_past_from_history(cfg):
    """Prime rolling_past state from the newest available causal history file."""
    history_dir = str(getattr(cfg, "rolling_past_history_dir", "") or "").strip()
    if not history_dir:
        return
    pattern = os.path.join(history_dir, "history_*.csv")
    candidates = []
    for p in glob.glob(pattern):
        name = os.path.basename(str(p))
        m = re.match(r"^history_(\d+)\.csv$", name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        print(f"[ROLLING_PAST][EVAL] No history file found in {history_dir}; continuing without bootstrap.")
        return

    candidates.sort(key=lambda x: x[0])
    history_csv_path = candidates[-1][1]
    history_df = load_energy_data(
        history_csv_path,
        convert_to_raw_units=False,
        config=cfg,
        mw_scale_overrides=None,
    )

    tail_days = int(getattr(cfg, "rolling_past_history_tail_days", 365) or 365)
    rows_per_day = int(getattr(cfg, "rolling_past_history_rows_per_day", 144) or 144)
    tail_rows = max(max(tail_days, 1) * max(rows_per_day, 1), max(rows_per_day, 1))
    if len(history_df) > tail_rows:
        history_df = history_df.iloc[-tail_rows:].copy()

    if "price" not in history_df.columns:
        raise RuntimeError(f"Evaluation rolling_past bootstrap missing price column: {history_csv_path}")

    price_state = _running_moments_from_values(history_df["price"].to_numpy())
    if price_state is None:
        raise RuntimeError(f"Evaluation rolling_past bootstrap could not build price state: {history_csv_path}")

    cfg.rolling_past_price_state = price_state
    cfg.rolling_past_wind_scale = _robust_p95_scale(history_df["wind"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_solar_scale = _robust_p95_scale(history_df["solar"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_hydro_scale = _robust_p95_scale(history_df["hydro"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_load_scale = _robust_p95_scale(history_df["load"].to_numpy(), min_scale=0.1)
    print(
        f"[ROLLING_PAST][EVAL] Bootstrapped from {history_csv_path} "
        f"(rows={len(history_df)}, price_mean={float(price_state['mean']):.3f})"
    )


def _hydrate_eval_config_from_training_config(cfg, final_models_dir: str):
    """Load Tier-specific evaluation overrides from the saved training_config.json."""
    config_path = os.path.join(final_models_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return cfg
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return cfg

    final_cfg = saved.get("final_config", {}) or {}
    for key, value in dict(final_cfg).items():
        if value is None:
            continue
        if hasattr(cfg, str(key)):
            setattr(cfg, str(key), value)

    runtime_contract = saved.get("runtime_contract", {}) or {}
    if isinstance(runtime_contract, dict):
        mode = str(runtime_contract.get("global_norm_mode", "") or "").strip().lower()
        if mode:
            cfg.use_global_normalization = bool(mode == "global")
        for key in ("rolling_past_history_dir", "investment_freq", "meta_freq_min", "meta_freq_max"):
            if key in runtime_contract and hasattr(cfg, key):
                setattr(cfg, key, runtime_contract[key])
        prior_contract = runtime_contract.get("forecast_prior", {}) or {}
        if isinstance(prior_contract, dict):
            for key, value in prior_contract.items():
                if value is None:
                    continue
                if hasattr(cfg, str(key)):
                    setattr(cfg, str(key), value)
            if "forecast_prior_horizon_steps" in prior_contract:
                horizon = int(prior_contract["forecast_prior_horizon_steps"])
                horizons = dict(getattr(cfg, "forecast_horizons", {}) or {})
                horizons["short"] = horizon
                cfg.forecast_horizons = horizons
            if "forecast_prior_denom_floor" in prior_contract:
                cfg.forecast_prior_denom_floor = float(prior_contract["forecast_prior_denom_floor"])

    flags = saved.get("flags", {}) or {}
    if "enable_forecast_utilization" in flags:
        cfg.enable_forecast_utilization = bool(flags.get("enable_forecast_utilization"))

    return cfg


def _load_runtime_contract_hash_from_training_config(final_models_dir: str) -> str:
    config_path = os.path.join(final_models_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return str(saved.get("runtime_contract_hash", "") or "").strip().lower()
    except Exception:
        return ""


def run_tier_suite_evaluation(eval_data: pd.DataFrame, args) -> Dict[str, Any]:
    """Tier1 evaluation on unseen data."""
    results: Dict[str, Any] = {
        "evaluation_mode": "tier1",
        "eval_data": args.eval_data,
        "tiers": {},
    }

    steps = args.eval_steps if args.eval_steps is not None else (len(eval_data) - 1)
    steps = int(max(1, steps))

    tier_out = os.path.join(args.output_dir, "tier1")
    os.makedirs(tier_out, exist_ok=True)
    env_log_dir = os.path.join(tier_out, "env_logs")
    os.makedirs(env_log_dir, exist_ok=True)

    run_dir = args.tier1_dir
    policy_final_models_dir = os.path.join(run_dir, "final_models")
    models = load_checkpoint_models(policy_final_models_dir)
    cfg = _hydrate_eval_config_from_training_config(_build_eval_config(args), policy_final_models_dir)

    eval_runtime_contract = build_runtime_contract(
        global_norm_mode="global" if bool(getattr(cfg, "use_global_normalization", False)) else "rolling_past",
        rolling_past_history_dir=str(getattr(cfg, "rolling_past_history_dir", "") or ""),
        investment_freq=int(getattr(cfg, "investment_freq", getattr(args, "investment_freq", 6)) or 6),
        meta_freq_min=int(getattr(cfg, "meta_freq_min", getattr(args, "meta_freq_min", 6)) or 6),
        meta_freq_max=int(getattr(cfg, "meta_freq_max", getattr(args, "meta_freq_max", 6)) or 6),
        enable_forecast_utilization=bool(getattr(cfg, "enable_forecast_utilization", False)),
        forecast_prior_settings=forecast_prior_contract_settings(cfg),
    )
    eval_runtime_hash = runtime_contract_hash(eval_runtime_contract)
    train_hash = _load_runtime_contract_hash_from_training_config(policy_final_models_dir)
    if train_hash and eval_runtime_hash != train_hash:
        raise RuntimeError(
            f"[EVAL_RUNTIME_CONTRACT] training/eval hash mismatch for tier1: "
            f"train={train_hash}, eval={eval_runtime_hash}"
        )
    if not bool(getattr(cfg, "use_global_normalization", False)):
        _prime_eval_rolling_past_from_history(cfg)

    eval_forecast_cache_dir = args.forecast_cache_dir
    if bool(getattr(cfg, "enable_forecast_utilization", False)):
        forecast_ctx = _resolve_reserved_eval_forecast_context(args, output_dir=tier_out, config=cfg)
        eval_forecast_cache_dir = forecast_ctx["forecast_cache_dir"]

    env = create_evaluation_environment(
        eval_data,
        output_dir=tier_out,
        investment_freq=args.investment_freq,
        config=cfg,
        env_log_dir=env_log_dir,
        forecast_cache_dir=eval_forecast_cache_dir,
        fail_fast=True,
    )

    tier_metrics = run_checkpoint_evaluation(models, env, eval_data, steps)
    if tier_metrics is None:
        tier_metrics = {"status": "failed", "error": "evaluation_failed"}
    else:
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
        tier_metrics.update({
            "status": "completed",
            "run_dir": run_dir,
            "final_models_dir": policy_final_models_dir,
            "evaluation_mode": "tier1",
            "variant": "tier1",
        })
    results["tiers"]["tier1"] = tier_metrics
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


def _extract_action_inference_success_rate(results: Dict[str, Any]) -> Optional[float]:
    """Read the current or compatibility action-inference success metric from result dictionaries."""
    for key in ("action_inference_success_rate", "prediction_success_rate"):
        if key not in results:
            continue
        try:
            return float(results[key])
        except Exception:
            return None
    return None


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
    optional = {
        "financial_exposure_dkk",
        "decision_step",
        "total_distributions_dkk",
        "distribution_adjusted_nav_dkk",
        "distribution_adjusted_trading_sleeve_dkk",
    }

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

    # Convert to USD (all logs are in DKK). Prefer total-wealth series when
    # available; keep raw NAV as explicit reported/ex-distribution diagnostics.
    reported_nav_usd = df["fund_nav_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    if "distribution_adjusted_nav_dkk" in df.columns:
        nav_usd = df["distribution_adjusted_nav_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    else:
        nav_usd = reported_nav_usd

    reported_trading_usd = (
        (df["trading_cash_dkk"].to_numpy(dtype=float) + df["financial_mtm_dkk"].to_numpy(dtype=float))
        * dkk_to_usd_rate
    )
    if "distribution_adjusted_trading_sleeve_dkk" in df.columns:
        trading_usd = df["distribution_adjusted_trading_sleeve_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    else:
        trading_usd = reported_trading_usd
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
        "sleeve_reported_nav_initial_usd": float(reported_nav_usd[0]),
        "sleeve_reported_nav_final_usd": float(reported_nav_usd[-1]),
        "sleeve_trading_initial_usd": float(trading_usd[0]),
        "sleeve_trading_final_usd": float(trading_usd[-1]),
        "sleeve_trading_gain_usd": trading_gain,
        "sleeve_reported_trading_initial_usd": float(reported_trading_usd[0]),
        "sleeve_reported_trading_final_usd": float(reported_trading_usd[-1]),
        "sleeve_operating_initial_usd": float(operating_usd[0]),
        "sleeve_operating_final_usd": float(operating_usd[-1]),
        "sleeve_operating_gain_usd": operating_gain,
    })
    if "total_distributions_dkk" in df.columns:
        metrics["sleeve_total_distributions_usd"] = float(df["total_distributions_dkk"].to_numpy(dtype=float)[-1] * dkk_to_usd_rate)

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
            periods_per_year = _infer_periods_per_year(df["timestamp"] if "timestamp" in df.columns else None)
            annual_risk_free_rate = float(getattr(PortfolioAnalysisConfig(), "risk_free_annual", 0.02))
            risk_stats = _compute_annualized_risk_metrics(
                tr,
                periods_per_year=periods_per_year,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            vol = float(risk_stats["annualized_volatility"])
            sharpe = float(risk_stats["annualized_sharpe"])
            peak = np.maximum.accumulate(trading_usd)
            drawdowns = np.where(peak > 0.0, (peak - trading_usd) / peak, 0.0)
            dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            metrics["sleeve_trading_volatility"] = vol
            metrics["sleeve_trading_step_volatility"] = float(risk_stats["step_volatility"])
            metrics["sleeve_trading_sharpe_ratio"] = sharpe
            metrics["sleeve_trading_max_drawdown_pct"] = dd * 100.0
    except Exception as e:
        metrics["sleeve_trading_risk_metrics_error"] = str(e)

    # Exposure diagnostics on decision steps (more interpretable than total-NAV vol when book value dominates).
    try:
        if "financial_exposure_dkk" in df.columns:
            expo = df["financial_exposure_dkk"].to_numpy(dtype=float)
            if "decision_step" in df.columns:
                dec = df["decision_step"].astype(bool).to_numpy()
                expo = expo[dec] if dec.any() else expo
            metrics["sleeve_mean_abs_exposure_dkk"] = float(np.mean(np.abs(expo))) if len(expo) else 0.0
            metrics["sleeve_max_abs_exposure_dkk"] = float(np.max(np.abs(expo))) if len(expo) else 0.0
    except Exception as e:
        metrics["sleeve_exposure_metrics_error"] = str(e)

    return metrics


def write_tier_report(tier_results: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """Write a consolidated tier report (comparison or single-variant evaluation).

    Returns: (csv_path, md_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    preferred_order = [
        "tier1",
    ]
    variants = [v for v in preferred_order if v in tier_results]
    variants.extend([v for v in tier_results.keys() if v not in variants])
    if not variants:
        variants = preferred_order[:1]

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
            "total_distributions_usd": float(r.get("total_distributions_usd", 0.0)),
            "reported_nav_final_usd": float(r.get("reported_nav_final_portfolio_value", 0.0)),
            "reported_nav_return_pct": _pct(r.get("reported_nav_total_return", 0.0)),
            "reported_nav_sharpe": float(r.get("reported_nav_sharpe_ratio", 0.0)),
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
    report_scope = "single_variant"
    report_prefix = "tier_result"
    report_title = "Tier Evaluation Report"
    mode_values = sorted({"tier1"})
    mode_label = "tier1"

    csv_path = os.path.join(output_dir, f"{report_prefix}_{ts}.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, f"{report_prefix}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"## {report_title}\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Report scope: `{report_scope}`\n")
        f.write(f"- Evaluation mode: `{mode_label}`\n")
        f.write(f"- Output CSV: `{os.path.basename(csv_path)}`\n\n")

        f.write("### Summary Table\n\n")
        f.write("| Variant | Status | Final (USD) | Return % | Sharpe | Max DD % | Trading Sharpe | Trading Max DD % |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['variant']} | {row['status']} | "
                f"{row['final_portfolio_value_usd']:.2f} | {row['total_return_pct']:.3f} | "
                f"{row['sharpe_ratio']:.4f} | {row['max_drawdown_pct']:.3f} | "
                f"{row['trading_sharpe']:.4f} | {row['trading_max_dd_pct']:.3f} |\n"
            )

        f.write("\nTotal-NAV Sharpe/DD can be compressed by the physical operating sleeve.\n")
        f.write("Trading-sleeve Sharpe/DD are shown above for a cleaner baseline trading read.\n")

        f.write("\n### Evaluation Inputs / Runtime\n\n")
        for _, row in df.iterrows():
            f.write(f"- {row['variant']}:\n")
            f.write(f"  - final_models: `{row['final_models_dir']}`\n")
            if str(row.get('run_dir', '')).strip() != "":
                f.write(f"  - run_dir: `{row['run_dir']}`\n")

        f.write("\n### NAV Sleeve Decomposition (Operating vs Trading)\n\n")
        f.write("Total NAV includes physical book value, which can compress volatility/drawdown.\n")
        f.write("This section splits gains into the operating sleeve (physical + ops revenue) and\n")
        f.write("the trading sleeve (cash + MTM).\n\n")
        f.write("| Variant | Trading Gain (USD) | Operating Gain (USD) | Trading Share | Trading Return % | Mean |Exposure| (DKK) | Avg Risk |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['variant']} | {row['trading_gain_usd']:.2f} | {row['operating_gain_usd']:.2f} | "
                f"{row['trading_gain_share']:.3f} | {row['trading_return_pct']:.2f} | "
                f"{row['mean_abs_exposure_dkk']:.2f} | {row['average_risk']:.3f} |\n"
            )

    return csv_path, md_path


def load_agent_system(trained_agents_dir: str, eval_env, enhanced: bool = False) -> Optional[Any]:
    """Load MultiESGAgent system from directory."""
    model_type = "enhanced" if enhanced else "standard"
    print(f"Loading {model_type} agent system from: {trained_agents_dir}")

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
        print(f"Loaded {loaded_count} agent policies")

        if loaded_count == 0:
            print("No agent policies loaded successfully")
            return None

        return agent_system

    except Exception as e:
        print(f"Error loading agent system: {e}")
        return None

def load_enhanced_agent_system(enhanced_models_dir: str, eval_env) -> Optional[Any]:
    """Load enhanced MultiESGAgent system with the fixed Tier1 evaluation runtime."""
    print(f"Loading enhanced agent system from: {enhanced_models_dir}")

    return load_agent_system(enhanced_models_dir, eval_env, enhanced=True)


def create_evaluation_environment(
    data: pd.DataFrame,
    log_path: Optional[str] = None,
    output_dir: str = "evaluation_results",
    investment_freq: int = 6,
    config=None,
    env_log_dir: Optional[str] = None,
    forecast_cache_dir: Optional[str] = None,
    fail_fast: bool = True,
) -> Any:
    """Create evaluation environment. Forecast utilization is cache-only."""
    print_progress("Setting up evaluation environment...")

    try:
        if forecast_cache_dir is not None and bool(getattr(config, "enable_forecast_utilization", False)):
            try:
                config.forecast_cache_dir = str(forecast_cache_dir)
            except Exception:
                pass

        base_env = RenewableMultiAgentEnv(
            data,
            investment_freq=int(investment_freq),
            config=config,
            log_dir=env_log_dir,
        )
        print_progress("Environment created")
        return base_env

    except Exception as e:
        if fail_fast:
            raise
        print_progress(f"Failed to create evaluation environment: {e}")
        return None


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
                                evaluation_steps: int,
                                timestamps=None,
                                annual_risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    try:
        # Portfolio performance
        if portfolio_values:
            pv_array = np.array(portfolio_values)
            returns = np.diff(pv_array) / pv_array[:-1]
            periods_per_year = _infer_periods_per_year(timestamps)
            risk_stats = _compute_annualized_risk_metrics(
                returns,
                periods_per_year=periods_per_year,
                annual_risk_free_rate=annual_risk_free_rate,
            )

            metrics['total_return'] = (pv_array[-1] / pv_array[0] - 1) if len(pv_array) > 1 else 0.0
            metrics['volatility'] = float(risk_stats['annualized_volatility'])
            metrics['step_volatility'] = float(risk_stats['step_volatility'])
            metrics['sharpe_ratio'] = float(risk_stats['annualized_sharpe'])
            metrics['periods_per_year'] = float(periods_per_year)
            metrics['annual_risk_free_rate'] = float(annual_risk_free_rate)
            metrics['per_step_risk_free_rate'] = float(risk_stats['per_step_risk_free_rate'])
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
        print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Error calculating performance metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics


def _iter_env_chain(env):
    """Yield an environment and simple wrapper parents."""
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _get_total_distributions_dkk(env) -> float:
    """Cash distributions are investor wealth and must be added back to NAV."""
    for candidate in _iter_env_chain(env):
        if hasattr(candidate, "total_distributions"):
            try:
                return float(max(0.0, getattr(candidate, "total_distributions", 0.0)))
            except Exception:
                return 0.0
    for candidate in _iter_env_chain(env):
        if hasattr(candidate, "distributed_profits"):
            try:
                return float(max(0.0, getattr(candidate, "distributed_profits", 0.0)))
            except Exception:
                return 0.0
    return 0.0


def _attach_distribution_adjusted_context(
    metrics: Dict[str, Any],
    reported_nav_values: list,
    risk_levels: list,
    evaluation_steps: int,
    timestamps=None,
    annual_risk_free_rate: float = 0.0,
    final_total_distributions_usd: float = 0.0,
) -> Dict[str, Any]:
    """Primary metrics use total wealth; raw NAV metrics stay visible."""
    reported = calculate_performance_metrics(
        reported_nav_values,
        {},
        risk_levels,
        evaluation_steps,
        timestamps=timestamps,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    for key in (
        "total_return",
        "volatility",
        "step_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "initial_portfolio_value",
        "final_portfolio_value",
    ):
        if key in reported:
            metrics[f"reported_nav_{key}"] = reported[key]
    metrics["distribution_adjusted_evaluation"] = True
    metrics["total_distributions_usd"] = float(max(0.0, final_total_distributions_usd))
    return metrics


def run_checkpoint_evaluation(models: Dict[str, Any],
                            eval_env,
                            data: pd.DataFrame,
                            evaluation_steps: int = 8000) -> Optional[Dict[str, Any]]:
    """Run evaluation using checkpoint models (Stable Baselines3)."""
    print("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Starting checkpoint model evaluation...")

    if not models or not eval_env:
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ No models or environment available")
        return None

    # Count loaded models
    loaded_count = sum(1 for m in models.values() if m is not None)
    model_total = max(1, len(models))
    print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Checkpoint models loaded: {loaded_count}/{model_total}")

    if loaded_count == 0:
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoint models loaded successfully")
        return None

    # Run evaluation
    obs, _ = eval_env.reset()
    steps = min(evaluation_steps, len(data) - 1)

    # Tracking metrics
    portfolio_values = []
    reported_nav_values = []
    rewards_by_agent = {agent: [] for agent in eval_env.possible_agents}
    risk_levels = []
    successful_inference_actions = 0
    total_inference_attempts = 0
    model_inference_successes = 0
    model_inference_attempts = 0

    print(f"ÃƒÂ°Ã…Â¸Ã‚Â§Ã‚Âª Running evaluation for {steps} steps...")

    for step in range(steps):
        actions = {}

        # Get actions from checkpoint models
        for agent in eval_env.possible_agents:
            if agent not in obs:
                continue

            total_inference_attempts += 1
            agent_key = agent.replace('_0', '_0')  # Normalize agent name

            if agent_key in models and models[agent_key] is not None:
                model_inference_attempts += 1
                try:
                    # Use checkpoint model for prediction
                    action, _ = models[agent_key].predict(obs[agent], deterministic=True)
                    actions[agent] = action
                    successful_inference_actions += 1
                    model_inference_successes += 1
                except Exception as e:
                    # FAIL-HARD on the investor agent: silent fallback to
                    # rule-based actions would invalidate tier comparisons.
                    if agent in ("investor_0", "investor"):
                        raise RuntimeError(
                            f"[EVAL FAIL-HARD] investor model prediction crashed; "
                            f"refusing to silently fall back to rule-based actions. "
                            f"Underlying error: {type(e).__name__}: {e}"
                        ) from e
                    print(f"[WARN] Prediction error for {agent}: {e}")
                    if hasattr(eval_env, "get_rule_based_agent_action"):
                        actions[agent] = eval_env.get_rule_based_agent_action(agent, obs[agent])
                    else:
                        actions[agent] = eval_env.action_space(agent).sample()
            else:
                if agent in ("investor_0", "investor"):
                    raise RuntimeError(
                        "[EVAL FAIL-HARD] investor model is missing; refusing rule-based/sample fallback."
                    )
                if hasattr(eval_env, "get_rule_based_agent_action"):
                    actions[agent] = eval_env.get_rule_based_agent_action(agent, obs[agent])
                else:
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

            # Convert to USD for analysis. Primary evaluation uses total
            # investor wealth: reported NAV plus cumulative cash distributions.
            reported_portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
            total_distributions_usd = _get_total_distributions_dkk(eval_env) * dkk_to_usd_rate
            portfolio_value_usd = reported_portfolio_value_usd + total_distributions_usd
            reported_nav_values.append(reported_portfolio_value_usd)
            portfolio_values.append(portfolio_value_usd)

            # Debug output for first step
            if step == 0:
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â Portfolio value extraction: {portfolio_value_dkk/1e9:.2f}B DKK ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ${portfolio_value_usd/1e6:.1f}M USD using method '{extraction_method}'")

            if hasattr(eval_env, 'get_risk_level'):
                risk_levels.append(eval_env.get_risk_level())

            completed_steps = step + 1
            if completed_steps % 1000 == 0 or completed_steps == steps:
                _print_eval_loop_progress(
                    completed_steps=completed_steps,
                    total_steps=steps,
                    portfolio_values=portfolio_values,
                    rewards_by_agent=rewards_by_agent,
                    successful_inference_actions=successful_inference_actions,
                    total_inference_attempts=total_inference_attempts,
                )

            # Handle episode termination
            if any(dones.values()) or any(truncs.values()):
                obs, _ = eval_env.reset()

        except Exception as e:
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Step execution error: {e}")
            break

    print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Checkpoint evaluation completed")

    # Calculate metrics
    success_rate = (
        model_inference_successes / model_inference_attempts
        if model_inference_attempts > 0 else 0.0
    )
    success_rate_all_agents = (
        successful_inference_actions / total_inference_attempts
        if total_inference_attempts > 0 else 0.0
    )
    risk_free_rate = _resolve_eval_risk_free_rate(eval_env)
    metrics = calculate_performance_metrics(
        portfolio_values,
        rewards_by_agent,
        risk_levels,
        steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
    )
    metrics = _attach_distribution_adjusted_context(
        metrics,
        reported_nav_values,
        risk_levels,
        steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
        final_total_distributions_usd=(
            float(portfolio_values[-1] - reported_nav_values[-1])
            if portfolio_values and reported_nav_values
            else 0.0
        ),
    )

    # Add checkpoint-specific metrics
    metrics['action_inference_success_rate'] = success_rate
    metrics['action_inference_success_rate_all_agents'] = success_rate_all_agents
    metrics['models_loaded'] = loaded_count
    metrics['evaluation_mode'] = 'checkpoint'

    return metrics


def run_agent_evaluation(agent_system,
                        eval_env,
                        data: pd.DataFrame,
                        evaluation_steps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Run evaluation using MultiESGAgent system."""
    print_progress("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Starting agent system evaluation...")

    if not agent_system or not eval_env:
        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ No agent system or environment available")
        return None

    # Run evaluation
    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Resetting evaluation environment...")
    try:
        obs, _ = eval_env.reset()
        print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Environment reset successful")
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Environment reset failed: {e}")
        return None

    # Pick evaluation length
    if evaluation_steps is None:
        evaluation_steps = min(len(data) - 1, 10_000)
    evaluation_steps = int(max(1, evaluation_steps))

    print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‚Â Evaluating for {evaluation_steps} steps")

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
    reported_nav_values = []
    rewards_by_agent = {agent: [] for agent in eval_env.possible_agents}
    risk_levels = []
    actions_taken = {agent: [] for agent in eval_env.possible_agents}

    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Starting evaluation loop...")

    for step in range(evaluation_steps):
        if step == 0:
            print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Processing first step...")

        actions = {}

        if step == 0:
            print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Getting actions from agents...")

        # Get actions from agent system
        for i, agent in enumerate(eval_env.possible_agents):
            if agent not in obs:
                continue

            try:
                if step == 0:
                    print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Processing agent {agent} (obs shape: {np.array(obs[agent]).shape})")

                agent_obs = np.array(obs[agent], dtype=np.float32).reshape(1, -1)
                policy = agent_system.policies[i]

                if hasattr(policy, "predict"):
                    if step == 0:
                        print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Getting prediction from {agent}...")
                    act, _ = policy.predict(agent_obs, deterministic=True)
                    if step == 0:
                        print_progress(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Got prediction from {agent}")
                else:
                    if agent in ("investor_0", "investor"):
                        raise RuntimeError(
                            "[EVAL FAIL-HARD] investor policy has no predict() method; refusing sample fallback."
                        )
                    act = eval_env.action_space(agent).sample()

                act = _coerce_action_for_space(act, eval_env.action_space(agent))
                actions[agent] = act
                actions_taken[agent].append(np.array(act).copy() if hasattr(act, 'copy') else act)

            except Exception as e:
                print_progress(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Action prediction error for {agent}: {e}")
                if agent in ("investor_0", "investor"):
                    raise RuntimeError(
                        f"[EVAL FAIL-HARD] investor policy prediction crashed in agent evaluation; "
                        f"refusing sample fallback. Underlying error: {type(e).__name__}: {e}"
                    ) from e
                actions[agent] = eval_env.action_space(agent).sample()

        if step == 0:
            print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Executing first environment step...")

        # Execute step
        try:
            obs, rewards, dones, truncs, infos = eval_env.step(actions)
            if step == 0:
                print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ First environment step completed")

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

            # Convert to USD for analysis. Primary evaluation uses total
            # investor wealth: reported NAV plus cumulative cash distributions.
            reported_portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
            total_distributions_usd = _get_total_distributions_dkk(eval_env) * dkk_to_usd_rate
            portfolio_value_usd = reported_portfolio_value_usd + total_distributions_usd
            reported_nav_values.append(reported_portfolio_value_usd)
            portfolio_values.append(portfolio_value_usd)

            if hasattr(eval_env, 'get_risk_level'):
                risk_levels.append(eval_env.get_risk_level())

            completed_steps = step + 1
            if completed_steps % 1000 == 0 or completed_steps == evaluation_steps:
                _print_eval_loop_progress(
                    completed_steps=completed_steps,
                    total_steps=evaluation_steps,
                    portfolio_values=portfolio_values,
                    rewards_by_agent=rewards_by_agent,
                )

            # Handle episode termination
            if any(dones.values()) or any(truncs.values()):
                obs, _ = eval_env.reset()

        except Exception as e:
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Step execution error: {e}")
            break

    print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Agent evaluation completed")

    # Calculate metrics
    risk_free_rate = _resolve_eval_risk_free_rate(eval_env)
    metrics = calculate_performance_metrics(
        portfolio_values,
        rewards_by_agent,
        risk_levels,
        evaluation_steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
    )
    metrics = _attach_distribution_adjusted_context(
        metrics,
        reported_nav_values,
        risk_levels,
        evaluation_steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
        final_total_distributions_usd=(
            float(portfolio_values[-1] - reported_nav_values[-1])
            if portfolio_values and reported_nav_values
            else 0.0
        ),
    )
    metrics['evaluation_mode'] = 'agents'

    return metrics


def run_comprehensive_evaluation(eval_data: pd.DataFrame, args) -> Dict[str, Any]:
    """Run comprehensive evaluation comparing all configurations with separate environments."""
    print_section_header("COMPREHENSIVE EVALUATION: All Configurations")
    print_progress("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Starting comprehensive evaluation of 5 systems...")

    comprehensive_results = {
        'evaluation_type': 'comprehensive',
        'timestamp': datetime.now().isoformat(),
        'eval_data_path': args.eval_data,
        'eval_steps': args.eval_steps,
        'configurations': {}
    }

    # 1. Evaluate Baselines (use basic environment)
    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  [1/3] EVALUATING BASELINES...", 1, 3)
    try:
        baseline_results = run_traditional_baselines(args.eval_data, args.eval_steps or 8000, args.output_dir)
        if baseline_results:
            comprehensive_results['configurations']['baselines'] = baseline_results
            print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Baseline evaluation completed")
        else:
            print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Baseline evaluation failed")
            comprehensive_results['configurations']['baselines'] = {'error': 'Baseline evaluation failed'}
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Baseline evaluation error: {e}")
        comprehensive_results['configurations']['baselines'] = {'error': str(e)}

    # 2. Evaluate Normal Models (no forecasts, no Tier1 runtime) - Create basic environment
    print_progress("[2/3] EVALUATING NORMAL MODELS", 2, 3)
    try:
        if os.path.exists(args.normal_models):
            # Create basic environment for normal models.
            print_progress("ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬â€ÃƒÂ¯Ã‚Â¸Ã‚Â Creating basic environment for normal models...")
            normal_cfg = _build_eval_config(args)
            normal_cfg.enable_forecast_utilization = False
            normal_eval_env = create_evaluation_environment(
                eval_data,
                output_dir=args.output_dir,
                investment_freq=args.investment_freq,
                config=normal_cfg,
                fail_fast=True,
            )

            if normal_eval_env:
                # Load normal models without enhanced features
                print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Loading normal agent system...")
                normal_agent_system = load_agent_system(args.normal_models, normal_eval_env, enhanced=False)
                if normal_agent_system:
                    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Running normal agent evaluation...")
                    normal_results = run_agent_evaluation(normal_agent_system, normal_eval_env, eval_data, args.eval_steps)
                    if normal_results:
                        normal_results['model_type'] = 'normal'
                        normal_results['features'] = {'forecast_utilization': False, 'tier1_reward_shaping': False}
                        comprehensive_results['configurations']['normal_agents'] = normal_results
                        print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Normal agent evaluation completed")
                    else:
                        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Normal agent evaluation failed")
                        comprehensive_results['configurations']['normal_agents'] = {'error': 'Normal agent evaluation failed'}
                else:
                    print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load normal agent system")
                    comprehensive_results['configurations']['normal_agents'] = {'error': 'Failed to load normal agent system'}
            else:
                print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to create basic evaluation environment")
                comprehensive_results['configurations']['normal_agents'] = {'error': 'Failed to create basic evaluation environment'}
        else:
            print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Normal models directory not found: {args.normal_models}")
            comprehensive_results['configurations']['normal_agents'] = {'error': f'Directory not found: {args.normal_models}'}
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Normal agent evaluation error: {e}")
        comprehensive_results['configurations']['normal_agents'] = {'error': str(e)}

    # 3. Evaluate Full Models with the fixed Tier1 runtime.
    print_progress("[3/3] EVALUATING FULL MODELS", 3, 3)
    try:
        if os.path.exists(args.full_models):
            # Create enhanced evaluation environment with the fixed-observation runtime.
            print_progress("Creating enhanced environment for full models...")
            full_cfg = _build_eval_config(args)
            forecast_ctx = None
            if bool(getattr(full_cfg, "enable_forecast_utilization", False)):
                forecast_ctx = _resolve_reserved_eval_forecast_context(args, output_dir=args.output_dir, config=full_cfg)
            enhanced_eval_env = create_evaluation_environment(
                eval_data,
                output_dir=args.output_dir,
                investment_freq=args.investment_freq,
                config=full_cfg,
                forecast_cache_dir=forecast_ctx["forecast_cache_dir"] if forecast_ctx else args.forecast_cache_dir,
                fail_fast=True,
            )

            if enhanced_eval_env:
                # Load enhanced models with all features
                print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Loading enhanced agent system...")
                full_agent_system = load_enhanced_agent_system(args.full_models, enhanced_eval_env)
                if full_agent_system:
                    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Running full agent evaluation...")
                    full_results = run_agent_evaluation(full_agent_system, enhanced_eval_env, eval_data, args.eval_steps)
                    if full_results:
                        full_results['model_type'] = 'enhanced'
                        full_results['features'] = {
                            'forecast_utilization': bool(getattr(full_cfg, "enable_forecast_utilization", False)),
                            'tier1_reward_shaping': False,
                        }
                        comprehensive_results['configurations']['full_agents'] = full_results
                        print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Full agent evaluation completed")
                    else:
                        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Full agent evaluation failed")
                        comprehensive_results['configurations']['full_agents'] = {'error': 'Full agent evaluation failed'}
                else:
                    print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load full agent system")
                    comprehensive_results['configurations']['full_agents'] = {'error': 'Failed to load full agent system'}
            else:
                print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to create enhanced evaluation environment")
                comprehensive_results['configurations']['full_agents'] = {'error': 'Failed to create enhanced evaluation environment'}
        else:
            print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Full models directory not found: {args.full_models}")
            comprehensive_results['configurations']['full_agents'] = {'error': f'Directory not found: {args.full_models}'}
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Full agent evaluation error: {e}")
        comprehensive_results['configurations']['full_agents'] = {'error': str(e)}

    print_progress("ÃƒÂ°Ã…Â¸Ã…Â½Ã¢â‚¬Â° Comprehensive evaluation completed!")
    return comprehensive_results


def save_results(results: Dict[str, Any], output_dir: str, mode: str, analysis: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
    """Save evaluation results and analysis to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save main results
    results_file = os.path.join(output_dir, f"evaluation_{mode}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¾ Results saved to: {results_file}")

    # Save analysis if provided
    analysis_file = None
    if analysis:
        analysis_file = os.path.join(output_dir, f"analysis_{mode}_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Analysis saved to: {analysis_file}")

    return results_file, analysis_file


def analyze_comprehensive_results(comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze comprehensive evaluation results across all configurations."""
    print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  COMPREHENSIVE ANALYSIS")
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
    # ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  FINAL PORTFOLIO VALUES TABLE
    # ========================================
    print("\n" + "=" * 80)
    print("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  FINAL PORTFOLIO VALUES - ALL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<4} {'Configuration':<25} {'Final Value':<15} {'Return':<10} {'Sharpe':<8} {'Features':<20}")
    print("-" * 80)

    for config_name, config_results in configurations.items():
        if 'error' in config_results:
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â {config_name}: {config_results['error']}")
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
                    'forecast_utilization': features.get('forecast_utilization', False),
                    'tier1_reward_shaping': features.get('tier1_reward_shaping', False)
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
            if metrics.get('forecast_utilization'): features.append("Forecast Utilization")
            if metrics.get('tier1_reward_shaping', False): features.append("Tier1 Reward Shaping")
            features_str = ', '.join(features) if features else 'Basic RL'
        else:
            features_str = 'Traditional'

        # Color coding for top performers
        rank_symbol = "ÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â¡" if i == 1 else "ÃƒÂ°Ã…Â¸Ã‚Â¥Ã‹â€ " if i == 2 else "ÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â°" if i == 3 else f"{i}."

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

    print("\nÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬Â  PERFORMANCE RANKING (by Sharpe Ratio):")
    for i, (config_name, metrics) in enumerate(ranked_configs, 1):
        features_str = ""
        if metrics['type'] == 'agent':
            features = []
            if metrics.get('forecast_utilization'): features.append("Forecast Utilization")
            if metrics.get('tier1_reward_shaping', False): features.append("Tier1 Reward Shaping")
            features_str = f" ({', '.join(features) if features else 'Basic'})"

        print(f"   {i}. {config_name}{features_str}")
        print(f"      Return: {metrics['total_return']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | Drawdown: {metrics['max_drawdown']:.2f}%")

    analysis['performance_ranking'] = ranked_configs

    # Feature impact analysis
    if len([x for x in performance_data.values() if x['type'] == 'agent']) >= 2:
        print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬ FEATURE IMPACT ANALYSIS:")

        # Find normal vs full agents
        normal_metrics = None
        full_metrics = None

        for config_name, metrics in performance_data.items():
            if metrics['type'] == 'agent':
                if not metrics.get('forecast_utilization') and not metrics.get('tier1_reward_shaping', False):
                    normal_metrics = metrics
                elif metrics.get('forecast_utilization') and metrics.get('tier1_reward_shaping', False):
                    full_metrics = metrics

        if normal_metrics and full_metrics:
            return_improvement = full_metrics['total_return'] - normal_metrics['total_return']
            sharpe_improvement = full_metrics['sharpe_ratio'] - normal_metrics['sharpe_ratio']

            print("   Enhanced Features Impact:")
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

        print(f"\nÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â¡ BEST PERFORMER: {best_config[0]}")
        print(f"   Sharpe Ratio: {best_config[1]['sharpe_ratio']:.3f}")
        print(f"   Total Return: {best_config[1]['total_return']:.2f}%")

    return analysis


def main():
    """Main evaluation function."""
    setup_console_encoding()
    print_progress("Starting Comprehensive Evaluation System")
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
            "'tiers' to evaluate Tier1/baseline models on unseen data using final_models/"
        ),
    )

    # Data and model paths
    parser.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv",
                       help="Path to evaluation data CSV")
    parser.add_argument("--trained_agents", type=str, default=None,
                       help="Directory with saved agent policies (required for 'agents' mode)")
    parser.add_argument("--checkpoint_dir", type=str, default="normal/checkpoints",
                       help="Base directory for checkpoints (for 'checkpoint' mode)")

    # Enhanced model support
    parser.add_argument("--enhanced_models", type=str, default=None,
                       help="Directory with enhanced models")

    # Comprehensive evaluation paths
    parser.add_argument("--normal_models", type=str, default="normal/final_models",
                       help="Directory with normal agent policies")
    parser.add_argument("--full_models", type=str, default="full/final_models",
                       help="Directory with full agent policies")

    # Evaluation options
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="Number of timesteps to evaluate (default: auto)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--investment_freq", type=int, default=6,
                       help="Investor action frequency for evaluation (should match training; default 6)")
    parser.add_argument("--meta_freq_min", type=int, default=None,
                       help="Minimum live investor trade cadence for evaluation (should match training).")
    parser.add_argument("--meta_freq_max", type=int, default=None,
                       help="Maximum live investor trade cadence for evaluation (should match training).")
    parser.add_argument(
        "--global_norm_mode",
        type=str,
        default="rolling_past",
        choices=["rolling_past", "global"],
        help="Normalization mode for evaluation config; keep aligned with training defaults.",
    )
    parser.add_argument(
        "--rolling_past_history_dir",
        type=str,
        default="",
        help=(
            "Directory with history_*.csv for rolling_past eval bootstrap when global_norm_mode=rolling_past "
            f"(default: {EVAL_ROLLING_PAST_HISTORY_DIR})."
        ),
    )

    # Tier1/baseline evaluation.
    parser.add_argument("--tier1_dir", type=str, default="tier1_seed789",
                       help="Tier1 run directory containing final_models/")
    parser.add_argument(
        "--tiers_only",
        type=str,
        default="tier1",
        choices=[
            "tier1",
            "baseline",
        ],
        help="Run only a subset in --mode tiers.",
    )
    parser.set_defaults(tier1_dir=None)
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Base directory for forecast caches.",
    )
    parser.add_argument(
        "--enable_forecast_utilization",
        action="store_true",
        default=False,
        help=(
            "Single switch for ANN forecast-cache utilization during evaluation. "
            "Usually inherited from training_config.json; exposed here so "
            "per-variant eval CLIs stay symmetric with training."
        ),
    )
    add_forecast_prior_override_args(parser)

    # Analysis options
    parser.add_argument("--analyze", action="store_true",
                       help="Perform comprehensive portfolio analysis")
    parser.add_argument("--plot", action="store_true",
                       help="Generate performance plots (requires --analyze)")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots to files instead of displaying (requires --plot)")

    args = parser.parse_args()

    # Enable GPU memory growth before any TF graph operations.
    configure_tf_memory()

    print_progress("Parsing arguments and validating paths...")

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
                ]
                if all(os.path.exists(os.path.join(candidate, f)) for f in required_files):
                    # Check if this is an enhanced model directory
                    is_enhanced = False

                    # Method 1: Check directory name
                    if "enhanced" in candidate.lower():
                        is_enhanced = True

                    # Method 2: Check training config for enhanced features
                    config_file = os.path.join(candidate, "training_config.json")
                    if os.path.exists(config_file):
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            enhanced_features = config.get('enhanced_features', {})
                            if (
                                enhanced_features.get('tier1_enabled')
                                or enhanced_features.get('tier1_reward_shaping_enabled')
                            ):
                                is_enhanced = True
                        except Exception as e:
                            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not parse training config for auto-detection ({config_file}): {e}")

                    if is_enhanced:
                        args.enhanced_models = candidate
                        print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Auto-detected enhanced models: {candidate}")
                    else:
                        args.trained_agents = candidate
                        print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Auto-detected trained agents: {candidate}")
                    break

        if args.trained_agents is None and args.enhanced_models is None:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ --trained_agents or --enhanced_models is required when using 'agents' mode")
            print("   Searched locations: normal/final_models, enhanced/final_models, training_agent_results/final_models")
            sys.exit(1)

    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¥ UNIFIED EVALUATION - MODE: {args.mode.upper()}")
    print("=" * 60)

    # Load evaluation data
    print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Loading evaluation data from: {args.eval_data}")
    try:
        eval_data = load_energy_data(args.eval_data)
        print_progress(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Loaded evaluation data: {eval_data.shape}")
        if "timestamp" in eval_data.columns and eval_data["timestamp"].notna().any():
            ts = eval_data["timestamp"].dropna()
            print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â¦ Date range: {ts.iloc[0]} ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {ts.iloc[-1]}")
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error loading evaluation data: {e}")
        sys.exit(1)

    # Tier suite mode evaluates one or more tier variants on unseen data using final_models/
    if args.mode == "tiers":
        try:
            results = run_tier_suite_evaluation(eval_data, args)
            # Always write a single consolidated report (CSV + Markdown)
            csv_path, md_path = write_tier_report(results.get("tiers", {}), args.output_dir)
            results["tier_report_csv"] = csv_path
            results["tier_report_md"] = md_path
            results["tier_report_scope"] = "single_variant"

            analysis = None
            if args.analyze:
                analysis = analyze_comprehensive_results({"configurations": results.get("tiers", {})})

            save_results(results, args.output_dir, mode="tiers", analysis=analysis)
            print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Å¾ Tier report written: {md_path}")
            print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Å¾ Tier report CSV written: {csv_path}")
            return
        except Exception as e:
            print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Tier suite evaluation failed: {e}")
            raise

    # Create evaluation environment for other modes. Forecast utilization is
    # cache-prior execution logic only.
    eval_cfg = _build_eval_config(args)
    eval_forecast_cache_dir = args.forecast_cache_dir
    if bool(getattr(eval_cfg, "enable_forecast_utilization", False)):
        forecast_ctx = _resolve_reserved_eval_forecast_context(args, output_dir=args.output_dir, config=eval_cfg)
        eval_forecast_cache_dir = forecast_ctx["forecast_cache_dir"]

    eval_env = create_evaluation_environment(
        eval_data,
        output_dir=args.output_dir,
        investment_freq=args.investment_freq,
        config=eval_cfg,
        forecast_cache_dir=eval_forecast_cache_dir,
        fail_fast=True,
    )

    if eval_env is None:
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to create evaluation environment")
        sys.exit(1)

    # Run evaluation based on mode
    results = None

    if args.mode == "checkpoint":
        # Checkpoint mode: find latest checkpoint and load SB3 models
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint is None:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoints found")
            sys.exit(1)

        checkpoint_models = load_checkpoint_models(latest_checkpoint)
        if not checkpoint_models or not any(m is not None for m in checkpoint_models.values()):
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoint models loaded successfully")
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
                print(f"Enhanced models directory not found: {args.enhanced_models}")
                sys.exit(1)

            print("Loading enhanced models with the Tier1 evaluation runtime...")

            agent_system = load_enhanced_agent_system(args.enhanced_models, eval_env)
            model_path = args.enhanced_models
        else:
            if not os.path.exists(args.trained_agents):
                print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Trained agents directory not found: {args.trained_agents}")
                sys.exit(1)

            agent_system = load_agent_system(args.trained_agents, eval_env)
            model_path = args.trained_agents

        if agent_system is None:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load agent system")
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
        print("ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬ÂºÃƒÂ¯Ã‚Â¸Ã‚Â Running traditional baseline evaluation...")

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
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Traditional baseline evaluation failed")
            sys.exit(1)

    elif args.mode == "compare":
        # Compare mode: run both AI and traditional baselines
        print("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Running comprehensive comparison...")

        # Initialize ai_results
        ai_results = None

        # Run AI evaluation first (enhanced or standard models)
        if args.enhanced_models:
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
                print("ÃƒÂ¢Ã‚ÂÃ…â€™ No AI models found for comparison")

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
            print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Traditional baselines failed, showing AI results only")
            results = ai_results
            results['baseline_error'] = baseline_results.get('error', 'Unknown error')
        elif baseline_results and 'error' not in baseline_results and not ai_results:
            print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â AI evaluation failed, showing baseline results only")
            results = {
                'evaluation_type': 'baselines_only',
                'baseline_results': baseline_results,
                'ai_error': 'AI evaluation failed'
            }
        else:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Both AI and baseline evaluations failed")
            sys.exit(1)

    elif args.mode == "comprehensive":
        # Comprehensive mode: evaluate all configurations
        print("ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Running comprehensive evaluation of all configurations...")

        results = run_comprehensive_evaluation(eval_data, args)

        if not results or not results.get('configurations'):
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Comprehensive evaluation failed")
            sys.exit(1)

    # Save and display results
    if results:
        # Add common metadata
        results['eval_data_path'] = args.eval_data
        results['forecast_utilization_enabled'] = bool(getattr(eval_cfg, "enable_forecast_utilization", False))
        results['eval_steps_requested'] = args.eval_steps

        # Perform analysis if requested
        analysis_results = None
        if args.analyze:
            print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  PERFORMING COMPREHENSIVE ANALYSIS...")

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
        print(f"\nÃƒÂ°Ã…Â¸Ã…Â½Ã¢â‚¬Â° EVALUATION SUCCESS!")

        # Handle different result types for display
        if results.get('evaluation_type') == 'comprehensive':
            # For comprehensive results, show summary from analysis
            if analysis_results and 'summary' in analysis_results:
                best_performer = analysis_results['summary'].get('best_performer', {})
                if best_performer:
                    print(f"ÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â¡ BEST PERFORMER: {best_performer['name']}")
                    metrics = best_performer['metrics']
                    print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Return: {metrics['total_return']:.2f}%")
                    print(f"   ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe: {metrics['sharpe_ratio']:.3f}")
                    print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Drawdown: {metrics['max_drawdown']:.2f}%")

                if 'feature_impact' in analysis_results:
                    impact = analysis_results['feature_impact']
                    print(f"\nÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬ ENHANCED FEATURES IMPACT:")
                    print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Return Improvement: {impact['return_improvement_pct']:+.2f}%")
                    print(f"   ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe Improvement: {impact['sharpe_improvement']:+.3f}")
            else:
                print("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Comprehensive evaluation completed - see detailed results in JSON file")

        elif results.get('evaluation_type') == 'comprehensive_comparison':
            # For comparison results, show AI results
            ai_results = results.get('ai_results', {})
            baseline_results = results.get('baseline_results', {})

            if ai_results:
                print(f"ÃƒÂ°Ã…Â¸Ã‚Â¤Ã¢â‚¬â€œ AI AGENTS PERFORMANCE:")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Final Return: {ai_results.get('total_return', 0):+.2%}")
                print(f"ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe Ratio: {ai_results.get('sharpe_ratio', 0):.3f}")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Total Rewards: {ai_results.get('total_rewards', 0):.2f}")

                if 'initial_portfolio_value' in ai_results and 'final_portfolio_value' in ai_results:
                    initial = ai_results['initial_portfolio_value']
                    final = ai_results['final_portfolio_value']
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Portfolio: ${initial/1e6:.1f}M ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ${final/1e6:.1f}M (${(final-initial)/1e6:+.1f}M)")
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Volatility: {ai_results.get('volatility', 0)*100:.2f}%")
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Max Drawdown: {ai_results.get('max_drawdown', 0)*100:.2f}%")

            if baseline_results:
                print(f"\nÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬ÂºÃƒÂ¯Ã‚Â¸Ã‚Â BASELINE COMPARISON:")
                for baseline_name, baseline_data in baseline_results.items():
                    if isinstance(baseline_data, dict) and 'total_return' in baseline_data:
                        method = baseline_data.get('method', baseline_name)
                        return_pct = baseline_data.get('total_return', 0) * 100
                        sharpe = baseline_data.get('sharpe_ratio', 0)
                        print(f"   {method}: {return_pct:+.2f}% return, {sharpe:.2f} Sharpe")
        else:
            # For regular results
            print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Final Return: {results.get('total_return', 0):+.2%}")
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
            print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Total Rewards: {results.get('total_rewards', 0):.2f}")
            print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Average Risk: {results.get('average_risk', 0):.3f}")

            # Portfolio performance details
            if 'initial_portfolio_value' in results and 'final_portfolio_value' in results:
                initial = results['initial_portfolio_value']
                final = results['final_portfolio_value']
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Portfolio: ${initial/1e6:.1f}M ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ${final/1e6:.1f}M (${(final-initial)/1e6:+.1f}M)")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Volatility: {results.get('volatility', 0)*100:.2f}%")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")

        inference_rate = _extract_action_inference_success_rate(results)
        if inference_rate is not None:
            print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Action Inference Success: {inference_rate:.1%}")
        if 'models_loaded' in results:
            print(f"ÃƒÂ°Ã…Â¸Ã‚Â¤Ã¢â‚¬â€œ Models Loaded: {results['models_loaded']}")

        # Display analysis summary if available
        if analysis_results and 'summary' in analysis_results:
            summary = analysis_results['summary']
            print(f"\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  ANALYSIS SUMMARY:")
            print(f"ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬Â  Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
            print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‚Â Performance Grade: {summary.get('performance_grade', 'N/A')}")

            if 'recommendations' in summary:
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¡ Recommendations:")
                for i, rec in enumerate(summary['recommendations'], 1):
                    print(f"   {i}. {rec}")

        print(f"\nÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¾ Files saved:")
        print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Å¾ Results: {results_file}")
        if analysis_file:
            print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Analysis: {analysis_file}")

    else:
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
