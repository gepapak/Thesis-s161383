#!/usr/bin/env python3
"""
Deep Learning Hedge Optimization & Intelligent Risk Management

"""

import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging

# Check TensorFlow version for compatibility
try:
    TF_VERSION = tuple(map(int, tf.__version__.split('.')[:2]))
    # Test if tf.diff actually works (some versions have issues)
    test_tensor = tf.constant([[1.0, 2.0, 3.0]])
    _ = tf.diff(test_tensor, axis=-1)
    HAS_TF_DIFF = True
    logging.info(f"TensorFlow {tf.__version__}: tf.diff available")
except Exception:
    HAS_TF_DIFF = False
    logging.info(f"TensorFlow {tf.__version__}: tf.diff not available, using manual diff")

# Check if CVXPY is available, otherwise use a fallback
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False
    logging.warning("CVXPY not found. Some optimization fallback may be used.")


# =====================================================================
# Helper Functions
# =====================================================================

def _get_config_value(attr_name, default_value):
    """Helper to get config values with fallback."""
    try:
        from config import EnhancedConfig
        config = EnhancedConfig()
        return getattr(config, attr_name, default_value)
    except Exception:
        return default_value

# =====================================================================
# DL Models
# =====================================================================

class HedgeOptimizer(tf.keras.Model):
    """
    Deep learning model for hedge strategy optimization with single-price coherence.

    Purpose: Optimize hedging parameters instead of asset allocation
    - Hedge intensity prediction (how much to hedge)
    - Risk allocation optimization (how to distribute hedge across sleeves)
    - Hedge direction prediction (net portfolio hedge direction)
    - Hedge effectiveness estimation (for training feedback)

    Replaces traditional portfolio optimization with hedge strategy optimization
    """

    def __init__(self, num_assets=3, market_dim=15, positions_dim=3, name="HedgeOptimizer"):
        super().__init__(name=name)
        self.num_assets = num_assets
        self.market_dim = market_dim
        self.positions_dim = positions_dim

        # Market state encoder (reuse existing architecture)
        self.market_encoder = tf.keras.Sequential([
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(_get_config_value('portfolio_dropout_rate_1', 0.2)),
            Dense(64, activation='relu'),
            BatchNormalization(),
        ])

        # Hedge intensity predictor (0.5 to 2.0 scaling for portfolio protection)
        self.intensity_predictor = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid', name='hedge_intensity_raw')  # [0,1] -> scale to [0.5,2.0]
        ])

        # Risk allocation predictor (how to distribute hedge across sleeves)
        self.allocation_predictor = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_assets, activation='softmax', name='risk_allocation')  # Sum to 1
        ])

        # Hedge direction predictor (net portfolio hedge direction)
        self.direction_predictor = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='tanh', name='hedge_direction')  # [-1, +1]
        ])

        # Hedge effectiveness estimator (for training feedback)
        self.effectiveness_estimator = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid', name='hedge_effectiveness')
        ])

    def call(self, inputs, training=None):
        """
        Predict optimal hedging parameters for single-price portfolio protection

        Args:
            inputs: Dict with keys:
                - market_state: Market conditions and forecasts
                - current_positions: Current financial positions
                - generation_forecast: Expected generation [wind, solar, hydro]
                - portfolio_metrics: [nav, drawdown, volatility, cash_flow]
        """
        # Handle both dictionary and tensor inputs for flexibility
        if isinstance(inputs, dict):
            market_features = inputs['market_state']
            current_positions = inputs['current_positions']
            generation_forecast = inputs.get('generation_forecast', tf.zeros((tf.shape(market_features)[0], 3)))
            portfolio_metrics = inputs.get('portfolio_metrics', tf.zeros((tf.shape(market_features)[0], 4)))
        else:
            # Tensor input from training loop - slice apart
            market_features = inputs[:, :self.market_dim]
            current_positions = inputs[:, self.market_dim:self.market_dim+3]
            generation_forecast = inputs[:, self.market_dim+3:self.market_dim+6]
            portfolio_metrics = inputs[:, self.market_dim+6:self.market_dim+10]

        # Encode market state
        market_encoded = self.market_encoder(market_features, training=training)

        # Extract forecast signals for hedge decision making
        forecast_features = market_encoded[:, -5:]  # Last 5 features assumed to be forecasts

        # Calculate forecast momentum and volatility for hedge timing
        if HAS_TF_DIFF:
            forecast_momentum = tf.reduce_mean(tf.diff(forecast_features, axis=-1), axis=-1, keepdims=True)
        else:
            forecast_diff = forecast_features[:, 1:] - forecast_features[:, :-1]
            forecast_momentum = tf.reduce_mean(forecast_diff, axis=-1, keepdims=True)
        forecast_volatility = tf.math.reduce_std(forecast_features, axis=-1, keepdims=True)

        # Combine all inputs for hedge parameter prediction
        combined_input = tf.concat([
            market_encoded,
            forecast_momentum,
            forecast_volatility,
            current_positions,
            generation_forecast,
            portfolio_metrics
        ], axis=-1)

        # Predict hedge parameters
        hedge_intensity_raw = self.intensity_predictor(combined_input, training=training)
        hedge_intensity = 0.5 + 1.5 * hedge_intensity_raw  # Scale to [0.5, 2.0]

        risk_allocation = self.allocation_predictor(combined_input, training=training)
        hedge_direction = self.direction_predictor(combined_input, training=training)

        # Predict hedge effectiveness for training feedback
        hedge_effectiveness = self.effectiveness_estimator(combined_input, training=training)

        return {
            'hedge_intensity': hedge_intensity,
            'risk_allocation': risk_allocation,
            'hedge_direction': hedge_direction,
            'hedge_effectiveness': hedge_effectiveness
        }



