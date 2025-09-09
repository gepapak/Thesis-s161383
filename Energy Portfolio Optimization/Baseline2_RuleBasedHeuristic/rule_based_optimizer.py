#!/usr/bin/env python3
"""
Baseline 2: Rule-Based Heuristic System for IEEE Benchmarking

This baseline implements domain expert knowledge through heuristic rules:
- Weather-based renewable energy investment
- Price-based battery arbitrage
- Risk-based position sizing
- Seasonal and time-of-day patterns
- Simple technical indicators

For IEEE publication benchmarking against MARL approach.
"""

import numpy as np
import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class RuleBasedEnergyOptimizer:
    """
    Rule-based energy portfolio optimizer using domain expert knowledge.
    Implements heuristic decision rules for renewable energy investment.
    """
    
    def __init__(self, initial_budget=5e8):
        """
        Initialize rule-based optimizer.
        
        Args:
            initial_budget: Initial portfolio value (DKK)
        """
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.cash = initial_budget
        
        # Asset positions (MW capacity owned)
        self.wind_capacity = 0.0
        self.solar_capacity = 0.0
        self.hydro_capacity = 0.0
        
        # Asset costs (DKK per MW)
        self.wind_cost_per_mw = 1.8e6
        self.solar_cost_per_mw = 1.2e6
        self.hydro_cost_per_mw = 3.0e6
        
        # Battery system
        self.battery_capacity_mwh = 10.0
        self.battery_soc = 0.5  # State of charge
        self.battery_efficiency = 0.9
        
        # Historical data for decision making
        self.price_history = deque(maxlen=168)  # 1 week
        self.wind_history = deque(maxlen=168)
        self.solar_history = deque(maxlen=168)
        self.load_history = deque(maxlen=168)
        
        # Performance tracking
        self.portfolio_values = []
        self.decisions_log = []
        
        # Rule parameters
        self.price_percentile_high = 0.8  # High price threshold
        self.price_percentile_low = 0.2   # Low price threshold
        self.capacity_factor_threshold = 0.3  # Minimum CF for investment
        self.max_single_investment = 0.1  # Max 10% of budget per investment
        self.risk_threshold = 0.7  # Risk level threshold
        
    def update_history(self, data_row):
        """Update historical data for decision making."""
        self.price_history.append(data_row['price'])
        self.wind_history.append(data_row['wind'])
        self.solar_history.append(data_row['solar'])
        self.load_history.append(data_row['load'])
    
    def calculate_capacity_factors(self, data_row):
        """Calculate capacity factors for renewable assets."""
        # Normalize by typical capacity
        wind_cf = data_row['wind'] / 1500.0  # 1500 MW typical wind farm
        solar_cf = data_row['solar'] / 1000.0  # 1000 MW typical solar farm
        hydro_cf = data_row['hydro'] / 1000.0  # 1000 MW typical hydro plant
        
        return {
            'wind': min(wind_cf, 1.0),
            'solar': min(solar_cf, 1.0),
            'hydro': min(hydro_cf, 1.0)
        }
    
    def get_price_signals(self):
        """Get price-based trading signals."""
        if len(self.price_history) < 24:
            return {'trend': 'neutral', 'volatility': 'low', 'level': 'medium'}
        
        prices = np.array(self.price_history)
        
        # Price level (relative to recent history)
        recent_prices = prices[-24:]  # Last 24 hours
        price_percentile = np.percentile(prices, [20, 80])
        
        if recent_prices[-1] > price_percentile[1]:
            level = 'high'
        elif recent_prices[-1] < price_percentile[0]:
            level = 'low'
        else:
            level = 'medium'
        
        # Price trend (short vs long term average)
        if len(prices) >= 48:
            short_avg = np.mean(prices[-24:])
            long_avg = np.mean(prices[-48:])
            trend_ratio = (short_avg - long_avg) / long_avg
            
            if trend_ratio > 0.02:
                trend = 'rising'
            elif trend_ratio < -0.02:
                trend = 'falling'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
        
        # Volatility
        volatility = np.std(recent_prices) / np.mean(recent_prices) if len(recent_prices) > 1 else 0
        vol_level = 'high' if volatility > 0.1 else 'medium' if volatility > 0.05 else 'low'
        
        return {'trend': trend, 'volatility': vol_level, 'level': level}
    
    def get_weather_signals(self, data_row):
        """Get weather-based investment signals."""
        capacity_factors = self.calculate_capacity_factors(data_row)
        
        signals = {}
        
        # Wind investment signal
        if len(self.wind_history) >= 24:
            recent_wind_avg = np.mean(list(self.wind_history)[-24:])
            long_wind_avg = np.mean(list(self.wind_history)) if len(self.wind_history) > 48 else recent_wind_avg
            
            if (capacity_factors['wind'] > self.capacity_factor_threshold and 
                recent_wind_avg > long_wind_avg * 1.1):
                signals['wind'] = 'strong_buy'
            elif capacity_factors['wind'] > self.capacity_factor_threshold:
                signals['wind'] = 'buy'
            else:
                signals['wind'] = 'hold'
        else:
            signals['wind'] = 'hold'
        
        # Solar investment signal
        if len(self.solar_history) >= 24:
            recent_solar_avg = np.mean(list(self.solar_history)[-24:])
            long_solar_avg = np.mean(list(self.solar_history)) if len(self.solar_history) > 48 else recent_solar_avg
            
            if (capacity_factors['solar'] > self.capacity_factor_threshold and 
                recent_solar_avg > long_solar_avg * 1.1):
                signals['solar'] = 'strong_buy'
            elif capacity_factors['solar'] > self.capacity_factor_threshold:
                signals['solar'] = 'buy'
            else:
                signals['solar'] = 'hold'
        else:
            signals['solar'] = 'hold'
        
        # Hydro investment signal (more stable, less weather dependent)
        if capacity_factors['hydro'] > 0.4:  # Higher threshold for hydro
            signals['hydro'] = 'buy'
        else:
            signals['hydro'] = 'hold'
        
        return signals
    
    def battery_arbitrage_decision(self, data_row):
        """Make battery charging/discharging decision."""
        price_signals = self.get_price_signals()
        current_price = data_row['price']
        
        # Simple arbitrage rules
        if price_signals['level'] == 'low' and self.battery_soc < 0.9:
            # Charge when prices are low
            charge_amount = min(0.25, 0.9 - self.battery_soc)  # Max 25% per hour
            self.battery_soc += charge_amount
            cost = charge_amount * self.battery_capacity_mwh * current_price
            return -cost, 'charge'
            
        elif price_signals['level'] == 'high' and self.battery_soc > 0.1:
            # Discharge when prices are high
            discharge_amount = min(0.25, self.battery_soc - 0.1)
            self.battery_soc -= discharge_amount
            revenue = discharge_amount * self.battery_capacity_mwh * current_price * self.battery_efficiency
            return revenue, 'discharge'
        
        return 0.0, 'idle'
    
    def renewable_investment_decision(self, data_row):
        """Make renewable energy investment decisions."""
        weather_signals = self.get_weather_signals(data_row)
        price_signals = self.get_price_signals()
        
        investments = []
        
        # Available cash for investment
        available_cash = self.cash * 0.8  # Keep 20% cash buffer
        max_investment = self.current_budget * self.max_single_investment
        
        # Wind investment
        if (weather_signals['wind'] in ['buy', 'strong_buy'] and 
            price_signals['trend'] in ['rising', 'neutral'] and
            available_cash > self.wind_cost_per_mw):
            
            investment_amount = min(max_investment, available_cash * 0.3)
            capacity_to_buy = investment_amount / self.wind_cost_per_mw
            
            if capacity_to_buy >= 1.0:  # Minimum 1 MW investment
                investments.append(('wind', capacity_to_buy, investment_amount))
        
        # Solar investment
        if (weather_signals['solar'] in ['buy', 'strong_buy'] and 
            price_signals['trend'] in ['rising', 'neutral'] and
            available_cash > self.solar_cost_per_mw):
            
            investment_amount = min(max_investment, available_cash * 0.3)
            capacity_to_buy = investment_amount / self.solar_cost_per_mw
            
            if capacity_to_buy >= 1.0:
                investments.append(('solar', capacity_to_buy, investment_amount))
        
        # Hydro investment (more conservative)
        if (weather_signals['hydro'] == 'buy' and 
            price_signals['volatility'] == 'low' and
            available_cash > self.hydro_cost_per_mw * 2):  # Require more cash for hydro
            
            investment_amount = min(max_investment, available_cash * 0.2)
            capacity_to_buy = investment_amount / self.hydro_cost_per_mw
            
            if capacity_to_buy >= 0.5:  # Smaller minimum for expensive hydro
                investments.append(('hydro', capacity_to_buy, investment_amount))
        
        return investments
    
    def calculate_generation_revenue(self, data_row):
        """Calculate revenue from owned renewable assets."""
        capacity_factors = self.calculate_capacity_factors(data_row)
        current_price = data_row['price']
        
        # Revenue from each asset type
        wind_revenue = (self.wind_capacity * capacity_factors['wind'] * 
                       current_price * 0.95)  # 5% O&M costs
        
        solar_revenue = (self.solar_capacity * capacity_factors['solar'] * 
                        current_price * 0.97)  # 3% O&M costs
        
        hydro_revenue = (self.hydro_capacity * capacity_factors['hydro'] * 
                        current_price * 0.92)  # 8% O&M costs
        
        total_revenue = wind_revenue + solar_revenue + hydro_revenue
        
        return {
            'wind_revenue': wind_revenue,
            'solar_revenue': solar_revenue,
            'hydro_revenue': hydro_revenue,
            'total_revenue': total_revenue
        }
    
    def step(self, data_row, timestep):
        """Execute one optimization step."""
        # Update historical data
        self.update_history(data_row)
        
        # Calculate generation revenue
        generation_revenue = self.calculate_generation_revenue(data_row)
        
        # Battery arbitrage decision
        battery_result, battery_action = self.battery_arbitrage_decision(data_row)
        
        # Investment decisions
        investments = self.renewable_investment_decision(data_row)
        
        # Execute investments
        total_investment = 0
        for asset_type, capacity, cost in investments:
            if self.cash >= cost:
                if asset_type == 'wind':
                    self.wind_capacity += capacity
                elif asset_type == 'solar':
                    self.solar_capacity += capacity
                elif asset_type == 'hydro':
                    self.hydro_capacity += capacity
                
                self.cash -= cost
                total_investment += cost
        
        # Update cash with revenues
        self.cash += generation_revenue['total_revenue'] + battery_result
        
        # Calculate total portfolio value
        asset_value = (self.wind_capacity * self.wind_cost_per_mw * 0.95 +  # Depreciation
                      self.solar_capacity * self.solar_cost_per_mw * 0.97 +
                      self.hydro_capacity * self.hydro_cost_per_mw * 0.98)
        
        self.current_budget = self.cash + asset_value
        self.portfolio_values.append(self.current_budget)
        
        # Log decision
        decision_log = {
            'timestep': timestep,
            'portfolio_value': self.current_budget,
            'cash': self.cash,
            'wind_capacity': self.wind_capacity,
            'solar_capacity': self.solar_capacity,
            'hydro_capacity': self.hydro_capacity,
            'battery_soc': self.battery_soc,
            'battery_action': battery_action,
            'battery_result': battery_result,
            'investments': investments,
            'total_investment': total_investment,
            'generation_revenue': generation_revenue['total_revenue'],
            'price': data_row['price']
        }
        
        self.decisions_log.append(decision_log)
        
        return decision_log

    def get_performance_metrics(self):
        """Calculate performance metrics for benchmarking."""
        if len(self.portfolio_values) < 2:
            return {}

        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        # Basic metrics
        total_return = (self.current_budget - self.initial_budget) / self.initial_budget

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(365 * 24) if len(returns) > 1 else 0

        # Sharpe ratio
        risk_free_rate = 0.02 / (365 * 24)  # 2% annual
        excess_returns = returns - risk_free_rate
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365 * 24)
                       if np.std(excess_returns) > 0 else 0)

        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Asset allocation
        total_asset_value = (self.wind_capacity * self.wind_cost_per_mw * 0.95 +
                           self.solar_capacity * self.solar_cost_per_mw * 0.97 +
                           self.hydro_capacity * self.hydro_cost_per_mw * 0.98)

        allocation = {
            'wind_allocation': (self.wind_capacity * self.wind_cost_per_mw * 0.95) / self.current_budget,
            'solar_allocation': (self.solar_capacity * self.solar_cost_per_mw * 0.97) / self.current_budget,
            'hydro_allocation': (self.hydro_capacity * self.hydro_cost_per_mw * 0.98) / self.current_budget,
            'cash_allocation': self.cash / self.current_budget
        }

        return {
            'final_value': self.current_budget,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'wind_capacity_mw': self.wind_capacity,
            'solar_capacity_mw': self.solar_capacity,
            'hydro_capacity_mw': self.hydro_capacity,
            'final_cash': self.cash,
            'battery_final_soc': self.battery_soc,
            **allocation
        }


class RuleBasedRiskManager:
    """
    Rule-based risk management system.
    Implements simple risk controls and position sizing rules.
    """

    def __init__(self, max_portfolio_risk=0.15, max_single_asset_weight=0.4):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_asset_weight = max_single_asset_weight
        self.risk_history = deque(maxlen=168)  # 1 week of risk data

    def assess_market_risk(self, data_row):
        """Assess current market risk level."""
        # Simple volatility-based risk assessment
        price = data_row['price']
        load = data_row['load']

        # Price volatility risk
        if len(self.risk_history) >= 24:
            recent_prices = [r['price'] for r in list(self.risk_history)[-24:]]
            price_vol = np.std(recent_prices) / np.mean(recent_prices)
        else:
            price_vol = 0.05  # Default moderate volatility

        # Load variability risk
        if 'risk' in data_row:
            market_risk = data_row['risk']
        else:
            market_risk = 0.3  # Default moderate risk

        # Combined risk score
        combined_risk = (price_vol * 0.6 + market_risk * 0.4)

        # Store for history
        self.risk_history.append({
            'price': price,
            'load': load,
            'risk': combined_risk
        })

        return {
            'price_volatility': price_vol,
            'market_risk': market_risk,
            'combined_risk': combined_risk,
            'risk_level': 'high' if combined_risk > 0.6 else 'medium' if combined_risk > 0.3 else 'low'
        }

    def position_sizing_rule(self, signal_strength, available_capital, risk_assessment):
        """Determine position size based on signal strength and risk."""
        base_size = available_capital * 0.1  # Base 10% position

        # Adjust for signal strength
        if signal_strength == 'strong_buy':
            size_multiplier = 1.5
        elif signal_strength == 'buy':
            size_multiplier = 1.0
        else:
            size_multiplier = 0.5

        # Adjust for risk
        if risk_assessment['risk_level'] == 'high':
            risk_multiplier = 0.5
        elif risk_assessment['risk_level'] == 'medium':
            risk_multiplier = 0.8
        else:
            risk_multiplier = 1.0

        final_size = base_size * size_multiplier * risk_multiplier
        return min(final_size, available_capital * self.max_single_asset_weight)
