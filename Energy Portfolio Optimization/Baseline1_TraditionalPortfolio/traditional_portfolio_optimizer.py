#!/usr/bin/env python3
"""
Baseline 1: Traditional Portfolio Optimization for IEEE Benchmarking

This baseline implements classical portfolio optimization techniques:
- Modern Portfolio Theory (Markowitz optimization)
- Mean-variance optimization
- Risk parity allocation
- Black-Litterman model
- Traditional financial metrics (Sharpe ratio, VaR, etc.)

For IEEE publication benchmarking against MARL approach.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class TraditionalPortfolioOptimizer:
    """
    Traditional portfolio optimization using Modern Portfolio Theory.
    Implements multiple classical approaches for robust benchmarking.
    """
    
    def __init__(self, initial_budget=5e8, lookback_window=252*24*12, rebalance_freq=24*7):
        """
        Initialize traditional portfolio optimizer.
        
        Args:
            initial_budget: Initial portfolio value (DKK)
            lookback_window: Historical data window for optimization (hours)
            rebalance_freq: Rebalancing frequency (hours)
        """
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.lookback_window = lookback_window
        self.rebalance_freq = rebalance_freq
        
        # Asset universe (matching MARL environment)
        self.assets = ['wind', 'solar', 'hydro', 'battery', 'cash']
        self.n_assets = len(self.assets)
        
        # Portfolio state
        self.weights = np.array([0.2, 0.2, 0.2, 0.1, 0.3])  # Initial allocation
        self.positions = self.weights * self.initial_budget
        self.returns_history = []
        self.portfolio_values = []
        
        # Risk parameters
        self.risk_aversion = 3.0  # Typical institutional investor
        self.max_weight = 0.4     # Maximum single asset weight
        self.min_weight = 0.0     # Minimum weight (no short selling)
        
        # Performance tracking
        self.metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'var_95': 0.0,
            'calmar_ratio': 0.0
        }
        
        # Covariance estimation
        self.cov_estimator = LedoitWolf()
        
    def calculate_asset_returns(self, data, t):
        """Calculate asset returns for portfolio optimization."""
        if t < 24:  # Need at least 24 hours of data
            return np.zeros(self.n_assets)
            
        # Get recent data
        recent_data = data.iloc[max(0, t-24):t+1]
        
        # Calculate returns for each asset type
        returns = np.zeros(self.n_assets)
        
        # FIXED: Calculate asset returns with safety bounds
        if len(recent_data) > 1:
            # FIXED: Safe price return calculation
            price_curr = recent_data['price'].iloc[-1]
            price_prev = recent_data['price'].iloc[-2]

            # Avoid division by zero and extreme values
            if price_prev > 0 and np.isfinite(price_curr) and np.isfinite(price_prev):
                price_return = (price_curr - price_prev) / price_prev
                # FIXED: Bound price returns to realistic range
                price_return = np.clip(price_return, -0.2, 0.2)  # Max ±20% price change
            else:
                price_return = 0.0

            # EXACT MATCH: Use same cost structure as MARL environment
            # MTM price return bounds (from MARL config: ±2% per step)
            mtm_cap_min = -0.02  # Same as config.mtm_price_return_cap_min
            mtm_cap_max = 0.02   # Same as config.mtm_price_return_cap_max
            capped_price_return = np.clip(price_return, mtm_cap_min, mtm_cap_max)

            # Physical asset returns (generation revenue like MARL)
            # MARL uses capacity factors and electricity markup
            electricity_markup = 1.0  # Same as MARL

            # Wind return - generation revenue approach (like MARL)
            wind_cf = np.clip(recent_data['wind'].iloc[-1] / 1500.0, 0.0, 1.0)
            wind_generation_return = wind_cf * 0.0001  # Small generation return
            # Add MTM on financial instruments (like MARL financial positions)
            wind_mtm = capped_price_return * 0.5  # 50% exposure to financial instruments
            returns[0] = wind_generation_return + wind_mtm

            # Solar return - same approach
            solar_cf = np.clip(recent_data['solar'].iloc[-1] / 1000.0, 0.0, 1.0)
            solar_generation_return = solar_cf * 0.0001
            solar_mtm = capped_price_return * 0.5
            returns[1] = solar_generation_return + solar_mtm

            # Hydro return - same approach
            hydro_cf = np.clip(recent_data['hydro'].iloc[-1] / 1000.0, 0.0, 1.0)
            hydro_generation_return = hydro_cf * 0.0001
            hydro_mtm = capped_price_return * 0.5
            returns[2] = hydro_generation_return + hydro_mtm

            # Battery return - arbitrage with exact MARL costs
            if recent_data['price'].mean() > 0:
                # Battery operational costs (same as MARL: 0.0002 * capacity)
                battery_opex_rate = 0.0002 / 100  # Per step, scaled for portfolio weight
                # Simple arbitrage opportunity
                price_volatility = recent_data['price'].std() / recent_data['price'].mean()
                battery_arbitrage = np.clip(price_volatility * 0.0001, 0.0, 0.0005)
                returns[3] = battery_arbitrage - battery_opex_rate
            else:
                returns[3] = -0.0002 / 100  # Just operational costs

            # Cash return - minimal (like MARL cash position)
            returns[4] = 0.0  # No return on cash (like MARL)
            
        return returns
    
    def estimate_covariance_matrix(self, returns_data):
        """Estimate covariance matrix using Ledoit-Wolf shrinkage."""
        if len(returns_data) < 10:
            # Use identity matrix for insufficient data
            return np.eye(self.n_assets) * 0.01
            
        returns_array = np.array(returns_data)
        if returns_array.shape[0] < returns_array.shape[1]:
            # Transpose if needed
            returns_array = returns_array.T
            
        try:
            cov_matrix, _ = self.cov_estimator.fit(returns_array).covariance_, self.cov_estimator.shrinkage_
            return cov_matrix
        except:
            return np.eye(self.n_assets) * 0.01
    
    def markowitz_optimization(self, expected_returns, cov_matrix):
        """
        Markowitz mean-variance optimization.
        Maximize: w^T * mu - (lambda/2) * w^T * Sigma * w
        Subject to: sum(w) = 1, w >= min_weight, w <= max_weight
        """
        w = cp.Variable(self.n_assets)
        
        # Objective: maximize expected return - risk penalty
        portfolio_return = w.T @ expected_returns
        portfolio_risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(portfolio_return - (self.risk_aversion / 2) * portfolio_risk)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= self.min_weight,  # No short selling
            w <= self.max_weight   # Concentration limits
        ]
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            if w.value is not None:
                return np.array(w.value).flatten()
        except:
            pass
            
        # Fallback to equal weights
        return np.ones(self.n_assets) / self.n_assets
    
    def risk_parity_optimization(self, cov_matrix):
        """
        Risk parity optimization - equal risk contribution from each asset.
        """
        def risk_parity_objective(weights, cov_matrix):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]
        bounds = [(self.min_weight, self.max_weight) for _ in range(self.n_assets)]
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        try:
            result = minimize(
                risk_parity_objective,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            if result.success:
                return result.x
        except:
            pass
            
        return x0
    
    def black_litterman_optimization(self, market_caps, expected_returns, cov_matrix):
        """
        Black-Litterman model with market equilibrium.
        """
        # Market equilibrium weights (based on market caps)
        w_market = market_caps / np.sum(market_caps)
        
        # Implied equilibrium returns
        pi = self.risk_aversion * cov_matrix @ w_market
        
        # Black-Litterman formula (simplified - no views)
        tau = 0.025  # Scaling factor
        omega = np.eye(self.n_assets) * 0.01  # Uncertainty in views
        
        # Posterior expected returns
        M1 = np.linalg.inv(tau * cov_matrix)
        M2 = np.linalg.inv(omega)
        M3 = np.linalg.inv(M1 + M2)
        
        mu_bl = M3 @ (M1 @ pi + M2 @ expected_returns)
        
        # Optimize with Black-Litterman returns
        return self.markowitz_optimization(mu_bl, cov_matrix)
    
    def step(self, data, t):
        """
        Execute one optimization step.
        
        Args:
            data: Market data DataFrame
            t: Current timestep
            
        Returns:
            dict: Portfolio metrics and actions
        """
        # Calculate current asset returns
        current_returns = self.calculate_asset_returns(data, t)
        self.returns_history.append(current_returns)
        
        # Rebalance portfolio if needed
        if t % self.rebalance_freq == 0 and len(self.returns_history) >= 10:
            self.rebalance_portfolio()
        
        # Update portfolio value with EXACT MARL cost structure
        portfolio_return = np.sum(self.weights * current_returns)

        # EXACT MATCH: Transaction costs (same as MARL: 0.0005 * trade_amount)
        # Assume 10% of portfolio trades each rebalancing period
        if t % self.rebalance_freq == 0 and t > 0:
            trade_amount_fraction = 0.1  # 10% of portfolio trades
            transaction_costs = 0.0005 * trade_amount_fraction  # Same rate as MARL
            portfolio_return -= transaction_costs

        # EXACT MATCH: No management fees (MARL removed admin costs for better learning)
        # MARL comment: "REMOVED: All admin costs for better agent learning environment"

        # EXACT MATCH: Apply same bounds as MARL MTM (±2% per step)
        portfolio_return = np.clip(portfolio_return, -0.02, 0.02)  # Same as MARL MTM caps

        # Safety check
        if not np.isfinite(portfolio_return):
            portfolio_return = 0.0  # No change if calculation fails

        self.current_budget *= (1 + portfolio_return)

        # EXACT MATCH: Use same bounds as MARL (70%-150% range is reasonable for institutional funds)
        min_value = self.initial_budget * 0.7   # Allow significant losses (like MARL can experience)
        max_value = self.initial_budget * 1.5   # Allow good gains (like MARL can achieve)
        self.current_budget = np.clip(self.current_budget, min_value, max_value)

        self.portfolio_values.append(self.current_budget)
        
        # Update positions
        self.positions = self.weights * self.current_budget
        
        # Calculate performance metrics
        self.update_metrics()
        
        return {
            'portfolio_value': self.current_budget,
            'weights': self.weights.copy(),
            'positions': self.positions.copy(),
            'returns': current_returns,
            'metrics': self.metrics.copy()
        }
    
    def rebalance_portfolio(self):
        """Rebalance portfolio using optimization."""
        if len(self.returns_history) < self.lookback_window:
            returns_data = self.returns_history
        else:
            returns_data = self.returns_history[-self.lookback_window:]
        
        # Estimate parameters
        expected_returns = np.mean(returns_data, axis=0)
        cov_matrix = self.estimate_covariance_matrix(returns_data)
        
        # Market cap weights (proxy)
        market_caps = np.array([0.3, 0.2, 0.2, 0.1, 0.2])  # Renewable energy market
        
        # Try different optimization methods
        try:
            # Primary: Markowitz optimization
            new_weights = self.markowitz_optimization(expected_returns, cov_matrix)
            
            # Fallback: Risk parity if Markowitz fails
            if np.any(np.isnan(new_weights)) or np.sum(new_weights) < 0.95:
                new_weights = self.risk_parity_optimization(cov_matrix)
                
            # Final fallback: Equal weights
            if np.any(np.isnan(new_weights)) or np.sum(new_weights) < 0.95:
                new_weights = np.ones(self.n_assets) / self.n_assets
                
            self.weights = new_weights
            
        except Exception as e:
            print(f"Optimization failed: {e}, using equal weights")
            self.weights = np.ones(self.n_assets) / self.n_assets
    
    def update_metrics(self):
        """Update performance metrics."""
        if len(self.portfolio_values) < 2:
            return
            
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # Total return
        self.metrics['total_return'] = (self.current_budget - self.initial_budget) / self.initial_budget
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            excess_returns = returns - 0.02/(365*24)  # Risk-free rate
            self.metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365*24)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        self.metrics['max_drawdown'] = np.min(drawdown)
        
        # Volatility (annualized)
        if len(returns) > 1:
            self.metrics['volatility'] = np.std(returns) * np.sqrt(365*24)
        
        # Value at Risk (95%)
        if len(returns) >= 20:
            self.metrics['var_95'] = np.percentile(returns, 5)
        
        # Calmar ratio
        if abs(self.metrics['max_drawdown']) > 1e-6:
            self.metrics['calmar_ratio'] = self.metrics['total_return'] / abs(self.metrics['max_drawdown'])
    
    def get_summary(self):
        """Get portfolio summary for reporting."""
        return {
            'final_value': self.current_budget,
            'total_return': self.metrics['total_return'],
            'sharpe_ratio': self.metrics['sharpe_ratio'],
            'max_drawdown': self.metrics['max_drawdown'],
            'volatility': self.metrics['volatility'],
            'var_95': self.metrics['var_95'],
            'calmar_ratio': self.metrics['calmar_ratio'],
            'final_weights': self.weights.copy()
        }


class TraditionalBatteryOptimizer:
    """
    Traditional battery optimization using price forecasting and arbitrage.
    """

    def __init__(self, capacity_mwh=10, efficiency=0.9):
        self.capacity_mwh = capacity_mwh
        self.efficiency = efficiency
        self.soc = 0.5  # State of charge (50% initial)
        self.charge_rate = 0.25  # Max charge/discharge rate (25% per hour)

        # Price forecasting (simple moving average)
        self.price_history = []
        self.forecast_horizon = 24  # 24-hour forecast

    def forecast_prices(self, current_price):
        """Simple price forecasting using moving average."""
        self.price_history.append(current_price)
        if len(self.price_history) > 168:  # Keep 1 week history
            self.price_history.pop(0)

        if len(self.price_history) < 24:
            return current_price

        # Simple trend-following forecast
        recent_avg = np.mean(self.price_history[-24:])
        long_avg = np.mean(self.price_history[-168:]) if len(self.price_history) >= 168 else recent_avg

        # Forecast based on trend
        trend = (recent_avg - long_avg) / long_avg if long_avg > 0 else 0
        forecast = current_price * (1 + trend * 0.1)  # Conservative trend following

        return forecast

    def optimize_battery(self, current_price):
        """Optimize battery operation based on price forecast."""
        forecast_price = self.forecast_prices(current_price)

        # Decision logic
        price_spread = (forecast_price - current_price) / current_price

        if price_spread > 0.05 and self.soc < 0.9:  # Charge if price expected to rise
            charge_amount = min(self.charge_rate, 0.9 - self.soc)
            self.soc += charge_amount
            return -charge_amount * self.capacity_mwh * current_price  # Cost of charging

        elif price_spread < -0.05 and self.soc > 0.1:  # Discharge if price expected to fall
            discharge_amount = min(self.charge_rate, self.soc - 0.1)
            self.soc -= discharge_amount
            return discharge_amount * self.capacity_mwh * current_price * self.efficiency  # Revenue from discharge

        return 0.0  # No action
