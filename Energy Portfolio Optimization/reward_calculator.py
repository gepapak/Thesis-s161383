#!/usr/bin/env python3
"""
Reward Calculator Module

Handles reward computation for multi-agent reinforcement learning:
- Investor rewards (PnL-based)
- Battery operator rewards (arbitrage revenue)
- Risk controller rewards (risk management)
- Meta controller rewards (overall performance)

Extracted from environment.py to improve code organization and maintainability.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from utils import SafeDivision, safe_clip, ErrorHandler
from config import EnhancedConfig

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculates rewards for all agents in the multi-agent system.
    
    Responsibilities:
    - Calculate investor rewards (trading PnL)
    - Calculate battery operator rewards (arbitrage)
    - Calculate risk controller rewards (risk management)
    - Calculate meta controller rewards (coordination)
    """
    
    def __init__(self, config: EnhancedConfig):
        """
        Initialize reward calculator.
        
        Args:
            config: Enhanced configuration object
        """
        self.config = config
        
        # Reward scaling factors
        self.pnl_scale = 1e-6  # Scale PnL to reasonable reward range
        self.risk_scale = 1.0
        self.battery_scale = 1e-6
        
        logger.info("RewardCalculator initialized")
    
    def calculate_investor_reward(self, pnl: float, position_exposure: float,
                                  risk_penalty: float = 0.0) -> float:
        """
        Calculate reward for investor agent.
        
        Args:
            pnl: Profit & Loss for current step (DKK)
            position_exposure: Total position exposure [0, 1]
            risk_penalty: Risk penalty term
        
        Returns:
            Investor reward
        """
        # Base reward from PnL
        pnl_reward = pnl * self.pnl_scale
        
        # Exploration bonus for taking positions
        exploration_bonus = 0.0
        if position_exposure > 0.001:
            exploration_bonus = position_exposure * 0.5
        
        # Position size bonus for meaningful positions
        position_bonus = 0.0
        if position_exposure > 0.05:
            position_bonus = (position_exposure - 0.05) * 2.0
        
        # Total reward
        reward = pnl_reward + exploration_bonus + position_bonus - risk_penalty
        
        return float(reward)
    
    def calculate_battery_reward(self, battery_revenue: float, soc: float,
                                price_volatility: float = 0.0) -> float:
        """
        Calculate reward for battery operator agent.
        
        Args:
            battery_revenue: Revenue from battery operations (DKK)
            soc: State of charge [0, 1]
            price_volatility: Price volatility measure
        
        Returns:
            Battery operator reward
        """
        # Base reward from arbitrage revenue
        revenue_reward = battery_revenue * self.battery_scale
        
        # Penalty for extreme SoC (encourage staying in middle range)
        soc_penalty = 0.0
        if soc < 0.2 or soc > 0.8:
            soc_penalty = 0.1
        
        # Bonus for high volatility (more arbitrage opportunities)
        volatility_bonus = price_volatility * 0.1
        
        # Total reward
        reward = revenue_reward + volatility_bonus - soc_penalty
        
        return float(reward)
    
    def calculate_risk_reward(self, overall_risk: float, risk_target: float = 0.5,
                             risk_budget_utilization: float = 0.0) -> float:
        """
        Calculate reward for risk controller agent.
        
        Args:
            overall_risk: Current overall risk level [0, 1]
            risk_target: Target risk level [0, 1]
            risk_budget_utilization: How much of risk budget is used [0, 1]
        
        Returns:
            Risk controller reward
        """
        # Reward for keeping risk near target
        risk_deviation = abs(overall_risk - risk_target)
        risk_reward = -risk_deviation * self.risk_scale
        
        # Penalty for excessive risk
        if overall_risk > 0.8:
            risk_reward -= (overall_risk - 0.8) * 2.0
        
        # Bonus for efficient risk budget utilization
        # Encourage using risk budget (not too conservative)
        if 0.3 < risk_budget_utilization < 0.7:
            risk_reward += 0.1
        
        return float(risk_reward)
    
    def calculate_meta_reward(self, nav_change: float, risk_adjusted_return: float,
                             coordination_bonus: float = 0.0) -> float:
        """
        Calculate reward for meta controller agent.
        
        Args:
            nav_change: Change in NAV (DKK)
            risk_adjusted_return: Risk-adjusted return (Sharpe-like)
            coordination_bonus: Bonus for good agent coordination
        
        Returns:
            Meta controller reward
        """
        # Base reward from NAV change
        nav_reward = nav_change * self.pnl_scale
        
        # Risk-adjusted return bonus
        risk_adj_bonus = risk_adjusted_return * 0.5
        
        # Coordination bonus
        coord_bonus = coordination_bonus * 0.2
        
        # Total reward
        reward = nav_reward + risk_adj_bonus + coord_bonus
        
        return float(reward)
    
    def calculate_all_rewards(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate rewards for all agents.
        
        Args:
            state: Dictionary with current environment state
        
        Returns:
            Dictionary with rewards for each agent
        """
        # Extract state variables
        pnl = state.get('step_pnl', 0.0)
        battery_revenue = state.get('battery_revenue', 0.0)
        overall_risk = state.get('overall_risk', 0.5)
        nav_change = state.get('nav_change', 0.0)
        position_exposure = state.get('position_exposure', 0.0)
        battery_soc = state.get('battery_soc', 0.5)
        
        # Calculate individual rewards
        investor_reward = self.calculate_investor_reward(pnl, position_exposure)
        battery_reward = self.calculate_battery_reward(battery_revenue, battery_soc)
        risk_reward = self.calculate_risk_reward(overall_risk)
        meta_reward = self.calculate_meta_reward(nav_change, 0.0)  # TODO: Add risk-adjusted return
        
        return {
            'investor_0': investor_reward,
            'battery_operator_0': battery_reward,
            'risk_controller_0': risk_reward,
            'meta_controller_0': meta_reward,
        }
    
    def get_reward_breakdown(self, agent: str, state: Dict[str, float]) -> Dict[str, float]:
        """
        Get detailed reward breakdown for debugging.
        
        Args:
            agent: Agent name
            state: Current environment state
        
        Returns:
            Dictionary with reward components
        """
        if agent == 'investor_0':
            pnl = state.get('step_pnl', 0.0)
            position_exposure = state.get('position_exposure', 0.0)
            return {
                'pnl_reward': pnl * self.pnl_scale,
                'exploration_bonus': position_exposure * 0.5 if position_exposure > 0.001 else 0.0,
                'position_bonus': (position_exposure - 0.05) * 2.0 if position_exposure > 0.05 else 0.0,
            }
        
        # Add other agents as needed
        return {}

