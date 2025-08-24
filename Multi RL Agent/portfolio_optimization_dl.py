#!/usr/bin/env python3
"""
Deep Learning Portfolio Optimization & Intelligent Asset Allocation
"""

import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from typing import Dict, List, Tuple, Optional
import cvxpy as cp
from collections import deque
import random
import logging

# Check if CVXPY is available, otherwise use a fallback
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


# =====================================================================
# DL Models
# =====================================================================

class DeepPortfolioOptimizer(tf.keras.Model):
    """
    Neural network that learns optimal portfolio allocation strategies
    considering risk, return, and market conditions
    """
    
    def __init__(self, num_assets=5, num_market_features=20):
        super(DeepPortfolioOptimizer, self).__init__()
        
        # Market state encoder
        self.market_encoder = tf.keras.Sequential([
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
        ])
        
        # Asset correlation learning
        self.correlation_net = tf.keras.Sequential([
            Dense(num_assets * num_assets, activation='tanh'),
            Reshape((num_assets, num_assets)),
        ])
        
        # Risk-adjusted return predictor
        self.return_predictor = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(num_assets, activation='linear', name='expected_returns')
        ])
        
        # Portfolio weight generator with constraints
        self.weight_generator = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_assets, activation='softmax', name='portfolio_weights')  # Sum to 1
        ])
        
        # Risk estimator
        self.risk_estimator = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid', name='portfolio_risk')
        ])
        
    def call(self, inputs, training=None):
        market_features = inputs['market_state']
        current_positions = inputs['current_positions']
        
        # Encode market state
        market_encoded = self.market_encoder(market_features, training=training)
        
        # Predict asset correlations
        correlations = self.correlation_net(market_encoded, training=training)
        
        # Predict returns
        expected_returns = self.return_predictor(market_encoded, training=training)
        
        # Generate optimal weights
        combined_input = tf.concat([market_encoded, current_positions], axis=-1)
        weights = self.weight_generator(combined_input, training=training)
        
        # Estimate portfolio risk
        portfolio_input = tf.concat([weights, expected_returns, market_encoded], axis=-1)
        portfolio_risk = self.risk_estimator(portfolio_input, training=training)
        
        return {
            'weights': weights,
            'expected_returns': expected_returns,
            'correlations': correlations,
            'portfolio_risk': portfolio_risk
        }


class SmartExecutionEngine(tf.keras.Model):
    """
    Deep RL model for optimal trade execution considering market microstructure
    """
    
    def __init__(self, action_dim=3):  # buy, sell, hold
        super(SmartExecutionEngine, self).__init__()
        
        # Market microstructure encoder
        self.microstructure_encoder = tf.keras.Sequential([
            Conv1D(32, 3, activation='relu', padding='same'),
            Conv1D(64, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
        ])
        
        # Order book analyzer
        self.orderbook_analyzer = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
        ])
        
        # Execution policy network
        self.policy_net = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(action_dim, activation='softmax')
        ])
        
        # Value network for critic
        self.value_net = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        
    def call(self, inputs, training=None):
        price_history = inputs['price_history']
        orderbook_data = inputs['orderbook']
        market_features = inputs['market_features']
        
        # Encode microstructure
        micro_features = self.microstructure_encoder(price_history, training=training)
        
        # Analyze order book
        orderbook_features = self.orderbook_analyzer(orderbook_data, training=training)
        
        # Combine all features
        combined = tf.concat([micro_features, orderbook_features, market_features], axis=-1)
        
        # Generate action probabilities
        action_probs = self.policy_net(combined, training=training)
        
        # Estimate state value
        state_value = self.value_net(combined, training=training)
        
        return {
            'action_probs': action_probs,
            'state_value': state_value
        }


class ESGScoringNetwork(tf.keras.Model):
    """
    Neural network to score energy investments based on ESG criteria
    """
    
    def __init__(self, num_assets=5):
        super(ESGScoringNetwork, self).__init__()
        
        # Environmental impact analyzer
        self.environmental_net = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid', name='environmental_score')
        ])
        
        # Social impact analyzer
        self.social_net = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid', name='social_score')
        ])
        
        # Governance analyzer
        self.governance_net = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid', name='governance_score')
        ])
        
        # Combined ESG score
        self.esg_combiner = tf.keras.Sequential([
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid', name='overall_esg_score')
        ])
        
    def call(self, inputs, training=None):
        env_features = inputs['environmental_data']
        social_features = inputs['social_data']
        governance_features = inputs['governance_data']
        
        env_score = self.environmental_net(env_features, training=training)
        social_score = self.social_net(social_features, training=training)
        governance_score = self.governance_net(governance_features, training=training)
        
        # Combine scores
        combined_scores = tf.concat([env_score, social_score, governance_score], axis=-1)
        overall_esg = self.esg_combiner(combined_scores, training=training)
        
        return {
            'environmental': env_score,
            'social': social_score,
            'governance': governance_score,
            'overall_esg': overall_esg
        }


class AdaptiveAssetAllocation(tf.keras.Model):
    """
    Learns to dynamically adjust portfolio allocation based on changing market conditions
    Uses attention mechanisms to focus on relevant market signals
    """
    
    def __init__(self, num_assets=5, lookback_window=60):
        super(AdaptiveAssetAllocation, self).__init__()
        
        self.num_assets = num_assets
        self.lookback_window = lookback_window
        
        # Multi-timeframe feature extractors
        self.short_term_conv = Conv1D(64, 5, activation='relu', padding='same')
        self.medium_term_conv = Conv1D(64, 15, activation='relu', padding='same')
        self.long_term_conv = Conv1D(64, 30, activation='relu', padding='same')
        
        # Temporal attention
        self.temporal_attention = MultiHeadAttention(num_heads=8, key_dim=64)
        self.temporal_norm = LayerNormalization()
        
        # Asset-specific encoders
        self.asset_encoders = []
        for i in range(num_assets):
            self.asset_encoders.append(
                tf.keras.Sequential([
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                ], name=f'asset_encoder_{i}')
            )
        
        # Cross-asset correlation learner
        self.correlation_attention = MultiHeadAttention(num_heads=4, key_dim=64)
        
        # Dynamic allocation network
        self.allocation_network = tf.keras.Sequential([
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_assets, activation='softmax')  # Portfolio weights
        ])
        
        # Risk-return trade-off learner
        self.risk_return_net = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(2, name='risk_return')  # [expected_return, expected_risk]
        ])
        
    def call(self, inputs, training=None):
        market_data = inputs['market_history']  # (batch, time, features)
        asset_fundamentals = inputs['asset_fundamentals']  # (batch, assets, fund_features)
        macro_indicators = inputs['macro_indicators']  # (batch, macro_features)
        
        # Multi-timeframe feature extraction
        short_features = self.short_term_conv(market_data, training=training)
        medium_features = self.medium_term_conv(market_data, training=training)
        long_features = self.long_term_conv(market_data, training=training)
        
        # Combine timeframe features
        multi_timeframe = tf.concat([short_features, medium_features, long_features], axis=-1)
        
        # Apply temporal attention
        attended_features = self.temporal_attention(
            multi_timeframe, multi_timeframe, training=training
        )
        attended_features = self.temporal_norm(multi_timeframe + attended_features)
        
        # Global pooling for sequence features
        sequence_features = GlobalAveragePooling1D()(attended_features)
        
        # Asset-specific encoding
        asset_embeddings = []
        for i, encoder in enumerate(self.asset_encoders):
            asset_emb = encoder(asset_fundamentals[:, i, :], training=training)
            asset_embeddings.append(asset_emb)
        
        asset_embeddings = tf.stack(asset_embeddings, axis=1)  # (batch, assets, emb_dim)
        
        # Cross-asset correlation learning
        correlated_assets = self.correlation_attention(
            asset_embeddings, asset_embeddings, training=training
        )
        
        # Combine all features
        asset_summary = GlobalAveragePooling1D()(correlated_assets)
        combined_features = tf.concat([sequence_features, asset_summary, macro_indicators], axis=-1)
        
        # Generate portfolio weights
        portfolio_weights = self.allocation_network(combined_features, training=training)
        
        # Predict risk-return profile
        risk_return = self.risk_return_net(combined_features, training=training)
        
        return {
            'portfolio_weights': portfolio_weights,
            'expected_return': risk_return[:, 0:1],
            'expected_risk': risk_return[:, 1:2],
            'asset_correlations': correlated_assets
        }


class RebalancingAgent(tf.keras.Model):
    """
    Intelligent rebalancing agent that decides when and how to rebalance portfolio
    Considers transaction costs, market impact, and tax implications
    """
    
    def __init__(self, num_assets=5):
        super(RebalancingAgent, self).__init__()
        
        # Portfolio drift analyzer
        self.drift_analyzer = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid', name='drift_magnitude')
        ])
        
        # Market impact predictor
        self.market_impact_net = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(num_assets, activation='linear', name='market_impact')
        ])
        
        # Transaction cost estimator
        self.transaction_cost_net = tf.keras.Sequential([
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid', name='transaction_cost')
        ])
        
        # Rebalancing decision network
        self.rebalancing_decision = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax', name='rebalance_decision')  # [wait, rebalance]
        ])
        
        # Optimal trade size calculator
        self.trade_size_net = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(num_assets, activation='tanh', name='trade_sizes')  # [-1, 1] normalized
        ])
        
    def call(self, inputs, training=None):
        current_weights = inputs['current_weights']
        target_weights = inputs['target_weights']
        market_conditions = inputs['market_conditions']
        portfolio_value = inputs['portfolio_value']
        
        # Calculate portfolio drift
        weight_drift = target_weights - current_weights
        drift_features = tf.concat([weight_drift, current_weights, target_weights], axis=-1)
        drift_magnitude = self.drift_analyzer(drift_features, training=training)
        
        # Predict market impact and costs
        trade_features = tf.concat([weight_drift, market_conditions], axis=-1)
        market_impact = self.market_impact_net(trade_features, training=training)
        transaction_cost = self.transaction_cost_net(trade_features, training=training)
        
        # Make rebalancing decision
        decision_features = tf.concat([
            drift_magnitude, market_impact, transaction_cost, market_conditions
        ], axis=-1)
        rebalance_decision = self.rebalancing_decision(decision_features, training=training)
        
        # Calculate optimal trade sizes
        trade_sizes = self.trade_size_net(decision_features, training=training)
        
        return {
            'rebalance_decision': rebalance_decision,
            'trade_sizes': trade_sizes,
            'expected_costs': transaction_cost,
            'market_impact': market_impact,
            'drift_magnitude': drift_magnitude
        }


class PortfolioOptimizationTrainer:
    """
    Training system for the portfolio optimization networks
    """
    
    def __init__(self, num_assets=5):
        self.num_assets = num_assets
        self.portfolio_optimizer = DeepPortfolioOptimizer(num_assets)
        self.adaptive_allocator = AdaptiveAssetAllocation(num_assets)
        self.rebalancing_agent = RebalancingAgent(num_assets)
        self.execution_engine = SmartExecutionEngine()
        
        # Optimizers
        self.optimizer_portfolio = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_allocation = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_rebalancing = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def portfolio_loss(self, y_true, y_pred):
        """
        Custom loss combining return maximization, risk minimization, and constraint satisfaction
        """
        predicted_weights = y_pred['weights']
        predicted_returns = y_pred['expected_returns']
        predicted_risk = y_pred['portfolio_risk']
        
        actual_returns = y_true['actual_returns']
        actual_risk = y_true['actual_risk']
        
        # Portfolio return loss
        portfolio_return = tf.reduce_sum(predicted_weights * predicted_returns, axis=-1)
        return_loss = -tf.reduce_mean(portfolio_return)  # Maximize returns
        
        # Risk prediction loss
        risk_loss = tf.reduce_mean(tf.square(predicted_risk - actual_risk))
        
        # Weight constraints (sum to 1, non-negative)
        weight_sum_loss = tf.reduce_mean(tf.square(tf.reduce_sum(predicted_weights, axis=-1) - 1.0))
        negative_weight_penalty = tf.reduce_mean(tf.nn.relu(-predicted_weights))
        
        # Diversification bonus (encourage spread)
        concentration_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(predicted_weights), axis=-1))
        
        total_loss = (return_loss + 
                      0.5 * risk_loss + 
                      10.0 * weight_sum_loss + 
                      5.0 * negative_weight_penalty + 
                      0.1 * concentration_penalty)
        
        return total_loss
    
    def train_step(self, batch_data):
        """Single training step for all networks"""
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
            # Portfolio optimization
            portfolio_outputs = self.portfolio_optimizer(batch_data['portfolio_inputs'])
            portfolio_loss = self.portfolio_loss(batch_data['portfolio_targets'], portfolio_outputs)
            
            # Adaptive allocation
            allocation_outputs = self.adaptive_allocator(batch_data['allocation_inputs'])
            allocation_loss = tf.keras.losses.mse(
                batch_data['allocation_targets']['weights'], 
                allocation_outputs['portfolio_weights']
            )
            
            # Rebalancing agent
            rebalancing_outputs = self.rebalancing_agent(batch_data['rebalancing_inputs'])
            rebalancing_loss = tf.keras.losses.sparse_categorical_crossentropy(
                batch_data['rebalancing_targets']['decisions'],
                rebalancing_outputs['rebalance_decision']
            )
        
        # Apply gradients
        gradients1 = tape1.gradient(portfolio_loss, self.portfolio_optimizer.trainable_variables)
        gradients2 = tape2.gradient(allocation_loss, self.adaptive_allocator.trainable_variables)
        gradients3 = tape3.gradient(rebalancing_loss, self.rebalancing_agent.trainable_variables)
        
        self.optimizer_portfolio.apply_gradients(
            zip(gradients1, self.portfolio_optimizer.trainable_variables)
        )
        self.optimizer_allocation.apply_gradients(
            zip(gradients2, self.adaptive_allocator.trainable_variables)
        )
        self.optimizer_rebalancing.apply_gradients(
            zip(gradients3, self.rebalancing_agent.trainable_variables)
        )
        
        return {
            'portfolio_loss': portfolio_loss,
            'allocation_loss': allocation_loss,
            'rebalancing_loss': rebalancing_loss
        }