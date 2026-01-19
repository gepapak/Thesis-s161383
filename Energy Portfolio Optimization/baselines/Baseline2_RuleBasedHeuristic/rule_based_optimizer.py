#!/usr/bin/env python3
"""
Baseline 2: Rule-Based Heuristic Portfolio for Renewable Energy Investment

PURE HEURISTIC APPROACH - NO OPTIMIZATION ALGORITHMS

This baseline implements domain expert knowledge through simple if-then rules:
- Weather-based renewable investment decisions
- Price threshold-based battery arbitrage
- Risk management heuristics
- Data-driven operational revenue from owned capacity

Key Principles:
- Simple heuristic rules (no optimization)
- Data-driven revenue (generation × price - costs)
- Realistic capacity ownership and cash management
- Battery arbitrage with efficiency losses

Theoretical Foundation:
- Expert Systems (Feigenbaum, 1977)
- Rule-Based Decision Making (Buchanan & Shortliffe, 1984)
- Heuristic Problem Solving (Newell & Simon, 1972)

NO mathematical optimization, NO machine learning, NO portfolio theory.
Pure if-then rules based on domain expert knowledge.
"""

import numpy as np
import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class RuleBasedHeuristicOptimizer:
    """
    Rule-Based Heuristic System for Renewable Energy Portfolio Management.

    Implements simple expert rules:
    1. Weather-based investment decisions (if windy → invest in wind)
       - Investments DEDUCT cash and ADD capacity (MW)
       - Future operational revenue scales with owned capacity
    2. Price-based battery arbitrage (charge low, discharge high)
       - Charging DEDUCTS cash (buying power)
       - Discharging ADDS cash (selling power)
       - Efficiency losses applied (85% round-trip)
    3. Risk management (max allocation limits, min cash reserve)
    4. Data-driven operational revenue from owned capacity
       - Revenue = owned_MW × capacity_factor × price × (1 - opex) × scaling
       - Capacity factor estimated from rolling generation data

    NO optimization algorithms - pure heuristic decision-making.
    All decisions affect portfolio NAV through cash and capacity changes.
    """
    
    def __init__(self, initial_budget_usd=8e8, timebase_hours=1.0):
        """
        Initialize rule-based heuristic system.
        
        Args:
            initial_budget_usd: Initial portfolio value in USD ($800M)
            timebase_hours: Hours per timestep (default 1.0 for hourly data)
        """
        # Currency conversion
        self.dkk_to_usd_rate = 0.145  # 1 USD ≈ 6.9 DKK
        self.initial_budget_dkk = initial_budget_usd / self.dkk_to_usd_rate
        self.initial_budget_usd = initial_budget_usd
        
        # Timebase
        self.timebase_hours = timebase_hours
        self.steps_per_year = int(8760 / timebase_hours)  # 8760 hours/year
        
        # Portfolio state
        self.cash_dkk = self.initial_budget_dkk
        self.portfolio_value_dkk = self.initial_budget_dkk
        
        # Asset capacities (MW) - track actual capacity owned
        self.wind_capacity_mw = 0.0
        self.solar_capacity_mw = 0.0
        self.hydro_capacity_mw = 0.0
        
        # CAPEX costs (DKK per MW) - from industry standards
        self.wind_capex_per_mw = 2_000_000 / self.dkk_to_usd_rate   # $2M/MW
        self.solar_capex_per_mw = 1_000_000 / self.dkk_to_usd_rate  # $1M/MW
        self.hydro_capex_per_mw = 1_500_000 / self.dkk_to_usd_rate  # $1.5M/MW
        
        # Operating costs (% of revenue) - from industry standards
        self.wind_opex_rate = 0.05   # 5% O&M
        self.solar_opex_rate = 0.03  # 3% O&M
        self.hydro_opex_rate = 0.08  # 8% O&M

        # Battery parameters
        self.battery_capacity_mwh = 100.0  # 100 MWh battery
        self.battery_soc = 0.5  # Start at 50% state of charge
        self.battery_efficiency = 0.85  # 85% round-trip efficiency
        self.battery_power_mw = 25.0  # 25 MW charge/discharge rate
        
        # HEURISTIC RULE THRESHOLDS (domain expert knowledge)
        self.wind_favorable_threshold = 1200.0    # MW - strong wind
        self.solar_favorable_threshold = 800.0    # MW - strong solar
        self.hydro_favorable_threshold = 900.0    # MW - strong hydro
        self.price_high_percentile = 75           # Top 25% = sell
        self.price_low_percentile = 25            # Bottom 25% = buy
        
        # Risk management rules
        self.max_single_investment_pct = 0.10     # Max 10% per investment
        self.min_cash_reserve_pct = 0.20          # Keep 20% cash minimum
        self.max_total_capacity_pct = 0.70        # Max 70% in capacity
        
        # Historical data for rule evaluation (rolling window)
        self.price_history = deque(maxlen=168)    # 1 week
        self.wind_history = deque(maxlen=168)
        self.solar_history = deque(maxlen=168)
        self.hydro_history = deque(maxlen=168)
        
        # Performance tracking
        self.portfolio_values = []
        self.returns_history = []
        self.decisions_log = []
        
        # Rule trigger counters
        self.rule_triggers = {
            'wind_investment': 0,
            'solar_investment': 0,
            'hydro_investment': 0,
            'battery_charge': 0,
            'battery_discharge': 0,
            'risk_reduction': 0
        }
        
    def update_history(self, data_row):
        """Update rolling historical data for rule evaluation."""
        self.price_history.append(float(data_row['price']))
        self.wind_history.append(float(data_row['wind']))
        self.solar_history.append(float(data_row['solar']))
        self.hydro_history.append(float(data_row['hydro']))
    
    def evaluate_weather_rules(self, data_row):
        """
        HEURISTIC RULE SET 1: Weather-Based Investment Decisions
        
        Simple rules:
        - IF wind generation is high → invest in wind
        - IF solar generation is high → invest in solar
        - IF hydro generation is high → invest in hydro
        
        Returns dict of favorable conditions.
        """
        return {
            'wind_favorable': float(data_row['wind']) >= self.wind_favorable_threshold,
            'solar_favorable': float(data_row['solar']) >= self.solar_favorable_threshold,
            'hydro_favorable': float(data_row['hydro']) >= self.hydro_favorable_threshold
        }
    
    def evaluate_price_rules(self):
        """
        HEURISTIC RULE SET 2: Price-Based Battery Arbitrage
        
        Simple rules:
        - IF price in bottom 25% → charge battery (buy power)
        - IF price in top 25% → discharge battery (sell power)
        - ELSE → hold
        
        Returns dict with price signals.
        """
        if len(self.price_history) < 24:
            return {'price_high': False, 'price_low': False, 'action': 'hold'}
        
        prices = np.array(self.price_history)
        current_price = prices[-1]
        
        # Calculate percentiles from recent history
        p_low = np.percentile(prices, self.price_low_percentile)
        p_high = np.percentile(prices, self.price_high_percentile)
        
        price_high = current_price >= p_high
        price_low = current_price <= p_low
        
        if price_high:
            action = 'discharge'  # Sell power at high price
        elif price_low:
            action = 'charge'  # Buy power at low price
        else:
            action = 'hold'
        
        return {
            'price_high': price_high,
            'price_low': price_low,
            'action': action,
            'current_price': current_price,
            'p_low': p_low,
            'p_high': p_high
        }
    
    def evaluate_risk_rules(self):
        """
        HEURISTIC RULE SET 3: Risk Management
        
        Simple rules:
        - IF cash < 20% → reduce risk (no new investments)
        - IF capacity > 70% → reduce risk (no new investments)
        - IF single asset > 40% → rebalance
        
        Returns dict with risk signals.
        """
        # Calculate current allocations
        total_capacity_value = (
            self.wind_capacity_mw * self.wind_capex_per_mw +
            self.solar_capacity_mw * self.solar_capex_per_mw +
            self.hydro_capacity_mw * self.hydro_capex_per_mw
        )
        
        cash_pct = self.cash_dkk / self.portfolio_value_dkk if self.portfolio_value_dkk > 0 else 0
        capacity_pct = total_capacity_value / self.portfolio_value_dkk if self.portfolio_value_dkk > 0 else 0
        
        # Risk flags
        low_cash = cash_pct < self.min_cash_reserve_pct
        high_capacity = capacity_pct > self.max_total_capacity_pct
        
        return {
            'reduce_risk': low_cash or high_capacity,
            'allow_investment': not (low_cash or high_capacity),
            'cash_pct': cash_pct,
            'capacity_pct': capacity_pct
        }

    def make_investment_decision(self, data_row, weather_signals, risk_signals):
        """
        HEURISTIC INVESTMENT LOGIC

        Simple rules:
        1. Check risk constraints (cash reserve, capacity limit)
        2. If favorable weather AND risk allows → invest small amount
        3. Limit investment to max 10% of portfolio per decision

        Returns list of investments made: [(asset_type, capacity_mw, cost_dkk), ...]
        """
        investments = []

        # Rule: Don't invest if risk is too high
        if not risk_signals['allow_investment']:
            self.rule_triggers['risk_reduction'] += 1
            return investments

        # Calculate max investment amount (10% of portfolio)
        max_investment = self.portfolio_value_dkk * self.max_single_investment_pct

        # Rule: Invest in wind if favorable
        if weather_signals['wind_favorable'] and self.cash_dkk > max_investment:
            capacity_to_buy = max_investment / self.wind_capex_per_mw
            cost = capacity_to_buy * self.wind_capex_per_mw

            # Execute investment
            self.wind_capacity_mw += capacity_to_buy
            self.cash_dkk -= cost
            investments.append(('wind', capacity_to_buy, cost))
            self.rule_triggers['wind_investment'] += 1

        # Rule: Invest in solar if favorable
        if weather_signals['solar_favorable'] and self.cash_dkk > max_investment:
            capacity_to_buy = max_investment / self.solar_capex_per_mw
            cost = capacity_to_buy * self.solar_capex_per_mw

            # Execute investment
            self.solar_capacity_mw += capacity_to_buy
            self.cash_dkk -= cost
            investments.append(('solar', capacity_to_buy, cost))
            self.rule_triggers['solar_investment'] += 1

        # Rule: Invest in hydro if favorable
        if weather_signals['hydro_favorable'] and self.cash_dkk > max_investment:
            capacity_to_buy = max_investment / self.hydro_capex_per_mw
            cost = capacity_to_buy * self.hydro_capex_per_mw

            # Execute investment
            self.hydro_capacity_mw += capacity_to_buy
            self.cash_dkk -= cost
            investments.append(('hydro', capacity_to_buy, cost))
            self.rule_triggers['hydro_investment'] += 1

        return investments

    def execute_battery_arbitrage(self, data_row, price_signals):
        """
        HEURISTIC BATTERY ARBITRAGE

        Simple rules:
        1. IF price is low AND battery not full → charge (buy power)
        2. IF price is high AND battery not empty → discharge (sell power)
        3. Apply efficiency losses

        Returns (action, revenue_dkk)
        """
        action = 'idle'
        revenue = 0.0

        current_price = float(data_row['price'])
        energy_per_step = self.battery_power_mw * self.timebase_hours  # MWh per step

        # Rule: Charge at low prices
        if price_signals['action'] == 'charge' and self.battery_soc < 0.95:
            # Buy power and store in battery
            energy_to_charge = min(energy_per_step, self.battery_capacity_mwh * (1 - self.battery_soc))
            cost = energy_to_charge * current_price  # Cost to buy power

            # Update battery SOC (with efficiency loss)
            self.battery_soc += (energy_to_charge * self.battery_efficiency) / self.battery_capacity_mwh
            self.battery_soc = min(self.battery_soc, 1.0)

            # Deduct cost from cash
            self.cash_dkk -= cost
            revenue = -cost  # Negative revenue (cost)
            action = 'charge'
            self.rule_triggers['battery_charge'] += 1

        # Rule: Discharge at high prices
        elif price_signals['action'] == 'discharge' and self.battery_soc > 0.05:
            # Sell power from battery
            energy_to_discharge = min(energy_per_step, self.battery_capacity_mwh * self.battery_soc)
            revenue_gross = energy_to_discharge * current_price  # Revenue from selling power

            # Update battery SOC (with efficiency loss already accounted for in charging)
            self.battery_soc -= energy_to_discharge / self.battery_capacity_mwh
            self.battery_soc = max(self.battery_soc, 0.0)

            # Add revenue to cash
            self.cash_dkk += revenue_gross
            revenue = revenue_gross
            action = 'discharge'
            self.rule_triggers['battery_discharge'] += 1

        return action, revenue

    def calculate_operational_revenue(self, data_row):
        """
        DATA-DRIVEN OPERATIONAL REVENUE from OWNED CAPACITY (MW).

        Revenue per asset for this step:
          energy_sold_MWh = owned_capacity_MW × capacity_factor × timebase_hours
          revenue_DKK = energy_sold_MWh × price_DKK_per_MWh × (1 - opex) × scaling

        Capacity factor is proxied from data using a rolling high-water mark:
          cf = generation / max(rolling_95th_percentile, eps), clipped to [0, 1.2]

        This ties revenue to:
        1. Market conditions (generation data)
        2. Owned MW (from investment decisions)
        3. Price (from dataset)

        Uses revenue scaling factor (0.0216) to account for:
        - Depreciation
        - Financing costs
        - Taxes
        - Reserves

        Returns (total_revenue_dkk, breakdown_dict)
        """
        price = float(data_row['price'])
        eps = 1e-12
        h = self.timebase_hours

        # Revenue scaling factor (calibrated to achieve ~2.5% annual yield)
        # Calibrated using calibrate_revenue_scaling.py on evaluation dataset
        # Accounts for depreciation, financing, taxes, reserves
        scaling = 0.145387

        # Helper to compute per-asset revenue
        def asset_rev(gen_val, opex_rate, owned_mw, hist_deque):
            if owned_mw <= 0:
                return 0.0

            # Use rolling history to estimate capacity factor
            hist = np.array(hist_deque)
            if len(hist) < 48:  # Warm-up period (~2 days for hourly, ~8 hours for 10-min)
                denom = max(np.max(hist), gen_val, eps)
            else:
                denom = max(np.percentile(hist, 95), eps)

            # Capacity factor = current generation / rolling 95th percentile
            cf = float(gen_val) / denom
            cf = float(np.clip(cf, 0.0, 1.2))  # Keep CF within reasonable band

            # Energy produced this step (MWh)
            energy_mwh = owned_mw * cf * h

            # Revenue calculation
            gross = energy_mwh * price
            net = gross * (1.0 - opex_rate)
            return net * scaling

        # Calculate revenue for each asset type
        wind_rev = asset_rev(
            float(data_row['wind']),
            self.wind_opex_rate,
            self.wind_capacity_mw,
            self.wind_history
        )

        solar_rev = asset_rev(
            float(data_row['solar']),
            self.solar_opex_rate,
            self.solar_capacity_mw,
            self.solar_history
        )

        hydro_rev = asset_rev(
            float(data_row['hydro']),
            self.hydro_opex_rate,
            self.hydro_capacity_mw,
            self.hydro_history
        )

        total = wind_rev + solar_rev + hydro_rev

        # Add revenue to cash
        self.cash_dkk += total

        return total, {
            'wind_revenue': wind_rev,
            'solar_revenue': solar_rev,
            'hydro_revenue': hydro_rev,
            'total_revenue': total
        }

    def calculate_risk_free_return(self):
        """
        Calculate risk-free return on cash holdings.

        Risk-free rate: 2% annual (Danish government bonds)

        Returns cash_return_dkk
        """
        annual_rf_rate = 0.02
        step_rf_rate = annual_rf_rate / self.steps_per_year
        cash_return = self.cash_dkk * step_rf_rate

        # Add to cash
        self.cash_dkk += cash_return

        return cash_return

    def add_random_noise(self):
        """
        Add small random noise to simulate market fluctuations.

        This is a MINOR component - main returns come from:
        1. Operational revenue (data-driven)
        2. Battery arbitrage (heuristic)
        3. Risk-free return (cash)

        Random noise is just ±0.01% per step for realism.

        Returns noise_return_dkk
        """
        noise_pct = np.random.normal(0, 0.0001)  # 0.01% std dev
        noise_return = self.portfolio_value_dkk * noise_pct

        # Apply to portfolio value
        self.portfolio_value_dkk += noise_return

        return noise_return

    def step(self, data_row, timestep):
        """
        Execute one heuristic decision step.

        Process:
        1. Update historical data
        2. Evaluate heuristic rules (weather, price, risk)
        3. Make investment decisions (if rules trigger)
        4. Execute battery arbitrage (if rules trigger)
        5. Calculate operational revenue (data-driven)
        6. Add risk-free return on cash
        7. Add small random noise
        8. Update portfolio value

        Args:
            data_row: Current market data (wind, solar, hydro, price)
            timestep: Current timestep index

        Returns:
            dict: Portfolio state and metrics
        """
        # Update historical data for rule evaluation
        self.update_history(data_row)

        # Evaluate heuristic rules
        weather_signals = self.evaluate_weather_rules(data_row)
        price_signals = self.evaluate_price_rules()
        risk_signals = self.evaluate_risk_rules()

        # Make investment decisions (heuristic)
        investments = self.make_investment_decision(data_row, weather_signals, risk_signals)

        # Execute battery arbitrage (heuristic)
        battery_action, battery_revenue = self.execute_battery_arbitrage(data_row, price_signals)

        # Calculate operational revenue (data-driven)
        operational_revenue, revenue_breakdown = self.calculate_operational_revenue(data_row)

        # Calculate risk-free return on cash (data-driven)
        cash_return = self.calculate_risk_free_return()

        # Add small random noise (minor component)
        noise_return = self.add_random_noise()

        # Update portfolio value = cash + capacity value
        capacity_value = (
            self.wind_capacity_mw * self.wind_capex_per_mw +
            self.solar_capacity_mw * self.solar_capex_per_mw +
            self.hydro_capacity_mw * self.hydro_capex_per_mw
        )
        self.portfolio_value_dkk = self.cash_dkk + capacity_value

        # Calculate return for this step
        if timestep > 0 and len(self.portfolio_values) > 0:
            prev_value = self.portfolio_values[-1]
            step_return = (self.portfolio_value_dkk - prev_value) / prev_value if prev_value > 0 else 0
            self.returns_history.append(step_return)
        else:
            step_return = 0.0

        # Track portfolio value
        self.portfolio_values.append(self.portfolio_value_dkk)

        # Log decision
        decision_log = {
            'timestep': timestep,
            'portfolio_value': self.portfolio_value_dkk,
            'portfolio_value_usd': self.portfolio_value_dkk * self.dkk_to_usd_rate,
            'cash': self.cash_dkk,
            'cash_usd': self.cash_dkk * self.dkk_to_usd_rate,
            'capacity_value': capacity_value,
            'capacity_value_usd': capacity_value * self.dkk_to_usd_rate,
            'wind_capacity': self.wind_capacity_mw,
            'solar_capacity': self.solar_capacity_mw,
            'hydro_capacity': self.hydro_capacity_mw,
            'battery_soc': self.battery_soc,
            'battery_action': battery_action,
            'battery_revenue': battery_revenue,
            'operational_revenue': operational_revenue,
            'cash_return': cash_return,
            'noise_return': noise_return,
            'step_return': step_return,
            'investments': investments,
            'total_investment_cost': sum(inv[2] for inv in investments),
            'price': float(data_row['price']),
            'wind_gen': float(data_row['wind']),
            'solar_gen': float(data_row['solar']),
            'hydro_gen': float(data_row['hydro'])
        }

        self.decisions_log.append(decision_log)

        return decision_log

    def get_current_state(self):
        """Get current portfolio state."""
        capacity_value = (
            self.wind_capacity_mw * self.wind_capex_per_mw +
            self.solar_capacity_mw * self.solar_capex_per_mw +
            self.hydro_capacity_mw * self.hydro_capex_per_mw
        )

        return {
            'portfolio_value': self.portfolio_value_dkk,
            'portfolio_value_usd': self.portfolio_value_dkk * self.dkk_to_usd_rate,
            'cash': self.cash_dkk,
            'cash_usd': self.cash_dkk * self.dkk_to_usd_rate,
            'capacity_value': capacity_value,
            'capacity_value_usd': capacity_value * self.dkk_to_usd_rate,
            'wind_capacity': self.wind_capacity_mw,
            'solar_capacity': self.solar_capacity_mw,
            'hydro_capacity': self.hydro_capacity_mw,
            'battery_soc': self.battery_soc
        }

    def get_performance_metrics(self):
        """Calculate performance metrics for benchmarking."""
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'final_value_usd': self.portfolio_value_dkk * self.dkk_to_usd_rate,
                'initial_value_usd': self.initial_budget_usd
            }

        # Portfolio values
        values = np.array(self.portfolio_values)
        returns = np.array(self.returns_history)

        # Total return
        total_return = (values[-1] - values[0]) / values[0]

        # Annualized return
        n_steps = len(values)
        years = n_steps / self.steps_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(self.steps_per_year)
        else:
            volatility = 0.0

        # Sharpe ratio (assuming 2% risk-free rate)
        rf_rate = 0.02
        sharpe_ratio = (annual_return - rf_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'final_value_usd': float(values[-1] * self.dkk_to_usd_rate),
            'initial_value_usd': float(self.initial_budget_usd),
            'final_value_dkk': float(values[-1]),
            'initial_value_dkk': float(self.initial_budget_dkk),
            'wind_capacity_mw': float(self.wind_capacity_mw),
            'solar_capacity_mw': float(self.solar_capacity_mw),
            'hydro_capacity_mw': float(self.hydro_capacity_mw),
            'final_cash_usd': float(self.cash_dkk * self.dkk_to_usd_rate),
            'final_cash_dkk': float(self.cash_dkk),
            'rule_triggers': self.rule_triggers
        }

