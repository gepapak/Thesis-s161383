#!/usr/bin/env python3
"""
Buy-and-Hold Optimizer for Baseline3

This module implements a simple buy-and-hold strategy that earns the risk-free rate.
It represents the most conservative passive investment approach - holding cash/bonds
and earning a steady risk-free return without any active trading or market exposure.

This baseline serves as the "floor" for investment performance - any active strategy
should be able to beat the risk-free rate to justify the additional complexity and risk.

Key Characteristics:
- No active trading or portfolio rebalancing
- No exposure to renewable energy markets
- Earns Danish government bond rate (~2% annually)
- Zero volatility (risk-free investment)
- Zero drawdown (capital preservation)
- Compound interest growth

This is the simplest possible baseline and represents what an investor could achieve
by simply holding government bonds or high-grade cash equivalents.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import json


class BuyAndHoldOptimizer:
    """
    Buy-and-Hold Strategy for Risk-Free Rate Baseline.
    
    This optimizer implements the simplest possible investment strategy:
    1. Hold initial capital in risk-free assets (government bonds/cash)
    2. Earn steady risk-free rate (Danish government bonds ~2% annually)
    3. Compound returns over time
    4. No active decisions, no market exposure, no volatility
    
    This serves as the "floor" baseline - any active strategy should beat this
    to justify additional complexity and risk.
    """
    
    def __init__(self, initial_budget_usd: float = 8e8, timebase_hours: float = 1/6):
        """
        Initialize Buy-and-Hold optimizer.
        
        Args:
            initial_budget_usd: Initial portfolio value in USD (default: $800M)
            timebase_hours: Hours per timestep (default: 0.1667 for 10-minute intervals)
        """
        # Portfolio tracking
        self.initial_budget_usd = initial_budget_usd
        self.current_value_usd = initial_budget_usd
        self.timebase_hours = timebase_hours
        
        # Risk-free rate parameters
        self.annual_risk_free_rate = 0.02  # 2% annual (Danish government bonds)
        self.steps_per_year = int(8760 / timebase_hours)  # Steps per year
        self.timestep_rate = self.annual_risk_free_rate / self.steps_per_year
        
        # Performance tracking
        self.portfolio_history = [initial_budget_usd]
        self.timestep_count = 0
        
        # Currency conversion (for consistency with other baselines)
        self.dkk_to_usd_rate = 0.145
        
        print(f"Buy-and-Hold Optimizer initialized:")
        print(f"  Initial Budget: ${initial_budget_usd/1e6:.1f}M USD")
        print(f"  Risk-free Rate: {self.annual_risk_free_rate*100:.1f}% annual")
        print(f"  Timebase: {timebase_hours:.4f} hours per step")
        print(f"  Steps per Year: {self.steps_per_year:,}")
        print(f"  Rate per Step: {self.timestep_rate*1e6:.3f} ppm")
    
    def step(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Execute one timestep of buy-and-hold strategy.
        
        Args:
            data_row: Market data for current timestep (not used - risk-free strategy)
        
        Returns:
            Dict with portfolio state and performance metrics
        """
        # Compound interest calculation (risk-free growth)
        self.current_value_usd *= (1 + self.timestep_rate)
        
        # Track portfolio evolution
        self.portfolio_history.append(self.current_value_usd)
        self.timestep_count += 1
        
        # Calculate performance metrics
        total_return = (self.current_value_usd - self.initial_budget_usd) / self.initial_budget_usd
        
        # Return step results
        return {
            'timestep': self.timestep_count,
            'portfolio_value_usd': self.current_value_usd,
            'total_return': total_return,
            'cash_usd': self.current_value_usd,  # All holdings are cash/bonds
            'capacity_value_usd': 0.0,  # No physical assets
            'wind_capacity_mw': 0.0,
            'solar_capacity_mw': 0.0,
            'hydro_capacity_mw': 0.0,
            'battery_soc': 0.0,  # No battery
            'risk_free_return': self.current_value_usd - self.initial_budget_usd,
            'method': 'Buy-and-Hold Strategy'
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dict with all performance metrics for evaluation
        """
        if len(self.portfolio_history) < 2:
            return {'error': 'Insufficient data for metrics calculation'}
        
        # Basic performance
        final_value = self.portfolio_history[-1]
        total_return = (final_value - self.initial_budget_usd) / self.initial_budget_usd
        
        # Time-based metrics
        days_elapsed = (self.timestep_count * self.timebase_hours) / 24
        years_elapsed = days_elapsed / 365.25
        annual_return = (1 + total_return) ** (1 / years_elapsed) - 1 if years_elapsed > 0 else 0
        
        # Risk metrics (should all be zero for risk-free investment)
        returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        max_drawdown = 0.0  # Risk-free investment has no drawdown
        
        # Sharpe ratio (should be 0 - no excess return over risk-free rate)
        sharpe_ratio = 0.0
        
        # Currency conversion for consistency
        final_value_dkk = final_value / self.dkk_to_usd_rate
        initial_value_dkk = self.initial_budget_usd / self.dkk_to_usd_rate
        
        return {
            # Core performance
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            
            # Portfolio values
            'final_value_usd': final_value,
            'initial_value_usd': self.initial_budget_usd,
            'final_value_dkk': final_value_dkk,
            'initial_value_dkk': initial_value_dkk,
            
            # Holdings (all cash/bonds)
            'final_cash_usd': final_value,
            'final_cash_dkk': final_value_dkk,
            'wind_capacity_mw': 0.0,
            'solar_capacity_mw': 0.0,
            'hydro_capacity_mw': 0.0,
            
            # Risk-free specific metrics
            'risk_free_rate_annual': self.annual_risk_free_rate,
            'total_risk_free_return_usd': final_value - self.initial_budget_usd,
            'compound_periods': self.timestep_count,
            
            # Time metrics
            'timesteps': self.timestep_count,
            'days_elapsed': days_elapsed,
            'years_elapsed': years_elapsed,
            
            # Strategy identification
            'method': 'Buy-and-Hold Strategy',
            'strategy_type': 'Risk-Free Rate Baseline',
            'active_decisions': 0,  # No active decisions made
            'market_exposure': 0.0,  # No market exposure
        }
    
    def get_portfolio_evolution(self) -> List[Dict[str, Any]]:
        """
        Get detailed portfolio evolution over time.
        
        Returns:
            List of portfolio states for each timestep
        """
        evolution = []
        
        for i, value in enumerate(self.portfolio_history):
            total_return = (value - self.initial_budget_usd) / self.initial_budget_usd
            
            evolution.append({
                'timestep': i,
                'portfolio_value_usd': value,
                'total_return': total_return,
                'cash_usd': value,
                'capacity_value_usd': 0.0,
                'wind_capacity_mw': 0.0,
                'solar_capacity_mw': 0.0,
                'hydro_capacity_mw': 0.0,
                'battery_soc': 0.0,
                'risk_free_return_usd': value - self.initial_budget_usd,
                'method': 'Buy-and-Hold Strategy'
            })
        
        return evolution
    
    def save_state(self, filepath: str):
        """Save optimizer state to JSON file."""
        state = {
            'initial_budget_usd': self.initial_budget_usd,
            'current_value_usd': self.current_value_usd,
            'timebase_hours': self.timebase_hours,
            'annual_risk_free_rate': self.annual_risk_free_rate,
            'timestep_count': self.timestep_count,
            'portfolio_history': self.portfolio_history,
            'performance_metrics': self.get_performance_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load optimizer state from JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.initial_budget_usd = state['initial_budget_usd']
        self.current_value_usd = state['current_value_usd']
        self.timebase_hours = state['timebase_hours']
        self.annual_risk_free_rate = state['annual_risk_free_rate']
        self.timestep_count = state['timestep_count']
        self.portfolio_history = state['portfolio_history']
        
        # Recalculate derived parameters
        self.steps_per_year = int(8760 / self.timebase_hours)
        self.timestep_rate = self.annual_risk_free_rate / self.steps_per_year


if __name__ == "__main__":
    # Simple test
    optimizer = BuyAndHoldOptimizer(initial_budget_usd=800_000_000, timebase_hours=1/6)
    
    # Simulate a few timesteps
    dummy_data = pd.Series({'wind': 1000, 'solar': 500, 'hydro': 800, 'price': 600})
    
    for i in range(5):
        result = optimizer.step(dummy_data)
        print(f"Step {i+1}: Portfolio = ${result['portfolio_value_usd']/1e6:.2f}M, Return = {result['total_return']*100:.6f}%")
    
    # Get final metrics
    metrics = optimizer.get_performance_metrics()
    print(f"\nFinal Metrics:")
    print(f"  Total Return: {metrics['total_return']*100:.6f}%")
    print(f"  Annual Return: {metrics['annual_return']*100:.4f}%")
    print(f"  Volatility: {metrics['volatility']*100:.6f}%")
