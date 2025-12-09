#!/usr/bin/env python3
"""
DL Overlay System - Tier 3 ONLY (Multiple Capabilities)

IMPORTANT: This module is ONLY for Tier 3 (when --dl_overlay flag is set).
Tier 2 (forecast integration) does NOT need this - it uses direct deltas from ForecastGenerator.

TIER 2 (Forecast Integration - NO DL overlay needed):
- ForecastGenerator (ANN/LSTM models) → generates forecasts
- _compute_forecast_deltas() → computes z-scores from forecasts  
- ForecastEngine → processes z-scores into observation features
- PPO learns directly from forecast deltas (z_short, z_medium, trust)

TIER 3 COMPONENTS (Multiple Capabilities - requires --dl_overlay flag):
- OverlaySharedModel: Shared encoder with 7 output heads providing:
  1. bridge_vec: Coordination signals (appended to agent observations)
  2. risk_budget: Position sizing multiplier (scales position sizes)
  3. pred_reward: Predictive reward shaping (adds to reward signal)
  4. strat_immediate/short/medium/long: Multi-horizon trading strategy signals
  5. meta_adv: Meta-critic head for FAMC (Forecast-Aware Meta-Critic)
- DLAdapter: Thin API for environment integration with strict shape contracts
- OverlayExperienceBuffer: Circular buffer for training (TIER 3 ONLY)
- OverlayTrainer: Multi-head loss training with per-horizon metrics (TIER 3 ONLY)
- CalibrationTracker: FGB forecast trust (τₜ) and expected ΔNAV (TIER 3 ONLY)

DL OVERLAY OUTPUTS (All Tier 3):
1. Bridge Vectors (4D): Multi-agent coordination signals
2. Risk Budget (1D): Position sizing multiplier [0.5, 1.5]
3. Predicted Reward (1D): Auxiliary reward signal [-1, 1]
4. Strategy Heads (4×4D): Multi-horizon trading signals (immediate/short/medium/long)
5. FGB Components: Forecast trust (τₜ) and expected ΔNAV for baseline adjustment
6. Meta-Critic: FAMC advantage prediction (Tier 3 with --meta_baseline_enable)

Features:
- 34D mode: 28D base + 6D forecast deltas (short/medium/long for price/total_gen)
- Bridge vectors + Risk budgeting + Multi-horizon strategy guidance
- Strict shape/name invariants enforced at runtime
- Memory efficient (< 1 MB)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from typing import Dict, Any, Optional
from collections import deque
import logging
from config import (
    DL_OVERLAY_LOSS_WEIGHTS,
    DL_OVERLAY_WINDOW_SIZE_DEFAULT,
    DL_OVERLAY_INIT_BUDGET_DEFAULT,
    DL_OVERLAY_DIRECTION_WEIGHT_DEFAULT,
)  # UNIFIED: Import constants from config

# ============================================================================
# DL OVERLAY: SHARED ENCODER + HEADS (28D FORECAST-AWARE MODE ONLY)
# ============================================================================

class OverlaySharedModel(tf.keras.Model):
    """
    Shared encoder for 28D forecast-aware mode only.

    Outputs (strict shape contract):
    - bridge_vec: (batch, bridge_dim) tanh ∈ [-1,1] → appended per agent
    - risk_budget: (batch, 1) in [0.5, 1.5] → env scales position sizes
    - pred_reward: (batch, 1) in [-1, +1] → auxiliary reward shaping
    - strat_immediate: (batch, 4) tanh ∈ [-1,1] → [wind, solar, hydro, price] immediate signals
    - strat_short: (batch, 4) tanh ∈ [-1,1] → [wind, solar, hydro, price] short-horizon (1h)
    - strat_medium: (batch, 4) tanh ∈ [-1,1] → [wind, solar, hydro, price] medium-horizon (4h)
    - strat_long: (batch, 4) tanh ∈ [-1,1] → [wind, solar, hydro, price] long-horizon (24h, risk-only)

    DESIGN: 4 strategy heads for 4 forecast horizons
    - Immediate: Reactive trading (respond to NOW signals) - full forecast
    - Short: Tactical adjustments (1h ahead) - 95% confidence
    - Medium: Strategic shifts (4h ahead) - 90% confidence
    - Long: Positioning (24h ahead) - risk-only (no directional signal)
    """

    def __init__(self, feature_dim: int, bridge_dim: int = 4, meta_head_dim: int = 32, enable_meta_head: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.bridge_dim = bridge_dim
        self.meta_head_dim = meta_head_dim
        self.enable_meta_head = enable_meta_head

        # INCREASED CAPACITY: Deeper, wider network for better pattern recognition
        self.d1 = Dense(256, activation='relu', name='overlay_d1')
        self.do1 = Dropout(0.15, name='overlay_do1')
        self.d2 = Dense(128, activation='relu', name='overlay_d2')
        self.do2 = Dropout(0.15, name='overlay_do2')
        self.d3 = Dense(64, activation='relu', name='overlay_d3')
        self.do3 = Dropout(0.1, name='overlay_do3')
        self.d4 = Dense(32, activation='relu', name='overlay_d4')

        # Output heads
        self.bridge_head = Dense(bridge_dim, activation='tanh', name="bridge_vec")
        self.risk_budget_head = Dense(1, activation='sigmoid', name="risk_budget")
        self.pred_reward_head = Dense(1, activation='tanh', name="pred_reward")

        # 28D strategy heads (4D: wind, solar, hydro, price) - one for each horizon
        # OPTIMAL: 4 heads for 4 horizons = full information utilization
        self.strat_immediate = Dense(4, activation='tanh', name="strat_immediate")
        self.strat_short = Dense(4, activation='tanh', name="strat_short")
        self.strat_medium = Dense(4, activation='tanh', name="strat_medium")
        self.strat_long = Dense(4, activation='tanh', name="strat_long")

        # TIER 3 ONLY: FAMC Meta-critic head g_φ(x_t) for learned control-variate baseline
        # This head predicts a state-only scalar correlated with PPO advantage
        # CRITICAL: State-only (no action leakage) - uses same features as other heads
        # TIER 2: This is NOT used (enable_meta_head=False)
        if enable_meta_head:
            self.meta_head = tf.keras.Sequential([
                Dense(meta_head_dim, activation='elu', name='meta_adv_hidden'),
                Dense(1, activation='linear', name='meta_advantage_head')
            ], name='meta_critic')
            logging.info(f"OverlaySharedModel initialized: {feature_dim} features -> bridge_dim={bridge_dim}, meta_head_dim={meta_head_dim} (TIER 3: FAMC enabled)")
        else:
            self.meta_head = None
            logging.info(f"OverlaySharedModel initialized: {feature_dim} features -> bridge_dim={bridge_dim} (TIER 2: No meta-critic)")

    def call(self, x, training=None):
        """Forward pass through shared encoder and heads"""
        z = self.d1(x)
        z = self.do1(z, training=training)
        z = self.d2(z)
        z = self.do2(z, training=training)
        z = self.d3(z)
        z = self.do3(z, training=training)
        z = self.d4(z)

        # Generate outputs
        bridge_vec = self.bridge_head(z)  # [-1,1]^bridge_dim
        risk_budget = self.risk_budget_head(z) * 1.0 + 0.5  # [0.5,1.5]
        pred_reward = self.pred_reward_head(z)  # [-1,1]

        out = {
            "bridge_vec": bridge_vec,
            "risk_budget": risk_budget,
            "pred_reward": pred_reward,
            "strat_immediate": self.strat_immediate(z),
            "strat_short": self.strat_short(z),
            "strat_medium": self.strat_medium(z),
            "strat_long": self.strat_long(z),
        }

        # TIER 3 ONLY: FAMC meta-critic head output
        if self.enable_meta_head and self.meta_head is not None:
            meta_adv = self.meta_head(z, training=training)  # (batch, 1) - state-only advantage prediction
            out["meta_adv"] = meta_adv

        return out


class DLAdapter:
    """
    Thin adapter API for env/wrapper to call overlay inference.
    Manages OverlaySharedModel and exposes shared_inference method.

    STRICT 34D MODE: feature_dim must be 34 (28D base + 6D deltas). Fails fast if not.
    """

    def __init__(self, feature_dim: int, bridge_dim: int = 4, verbose: bool = False,
                 meta_head_dim: int = 32, enable_meta_head: bool = False):
        # CRITICAL FIX: Remove hard 34D constraint - accept dynamic feature dimensions
        # This makes the system robust to feature additions (e.g., temperature, volatility)
        # The model will adapt to whatever feature_dim is provided by the environment
        if feature_dim < 10:
            error_msg = (
                f"[CRITICAL] DLAdapter requires at least 10 features, got {feature_dim}\n"
                f"REASON: The DL overlay needs minimum market context to function:\n"
                f"  - Market features (wind, solar, hydro, price, etc.)\n"
                f"  - Position features (wind_pos, solar_pos, hydro_pos)\n"
                f"  - Portfolio metrics (nav, cash, efficiency)\n"
                f"SOLUTION: Ensure environment provides sufficient observation features.\n"
                f"If forecasting is disabled, use baseline mode (--dl_overlay not set)."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        self.feature_dim = feature_dim
        self.bridge_dim = bridge_dim
        self.verbose = verbose
        self.enable_meta_head = enable_meta_head
        self.model = OverlaySharedModel(feature_dim, bridge_dim, meta_head_dim, enable_meta_head)
        self._shape_check_done = False  # Track if we've validated shapes

        if enable_meta_head:
            logging.info(f"DLAdapter initialized: feature_dim={feature_dim} (dynamic), bridge_dim={bridge_dim}, meta_head_dim={meta_head_dim} (FAMC enabled)")
        else:
            logging.info(f"DLAdapter initialized: feature_dim={feature_dim} (dynamic), bridge_dim={bridge_dim}")

    def shared_inference(self, feats_np: np.ndarray, training: bool = False) -> Dict[str, Any]:
        """
        Run shared inference on features (dynamic dimension).

        Args:
            feats_np: numpy array of shape (1, feature_dim) or (batch, feature_dim)
            training: whether in training mode

        Returns:
            dict with strict keys and shapes:
            - bridge_vec: (batch, bridge_dim)
            - risk_budget: (batch, 1) in [0.5, 1.5]
            - pred_reward: (batch, 1) in [-1, 1]
            - strat_immediate: (batch, 4)
            - strat_short: (batch, 4)
            - strat_medium: (batch, 4)
            - strat_long: (batch, 4)
        """
        try:
            # Ensure 2D input
            if len(feats_np.shape) == 1:
                feats_np = feats_np.reshape(1, -1)

            # CRITICAL FIX: Verify input shape matches feature_dim (dynamic)
            if feats_np.shape[1] != self.feature_dim:
                error_msg = (
                    f"[CRITICAL] DLAdapter.shared_inference: Expected {self.feature_dim}D features, got {feats_np.shape[1]}D\n"
                    f"REASON: Feature vector dimension must match the adapter's feature_dim.\n"
                    f"SOLUTION: Ensure environment observation space matches adapter initialization.\n"
                    f"  - Adapter initialized with feature_dim={self.feature_dim}\n"
                    f"  - Received features with shape={feats_np.shape}\n"
                    f"  - Check that all forecast models are loaded and generating expected dimensions."
                )
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Convert to tensor and run inference
            feats_tensor = tf.convert_to_tensor(feats_np, dtype=tf.float32)
            outs = self.model(feats_tensor, training=training)

            # Convert tensors → numpy
            result = {}
            for k, v in outs.items():
                if tf.is_tensor(v):
                    result[k] = v.numpy()
                else:
                    result[k] = np.array(v)

            # STRICT SHAPE VALIDATION (first call only)
            if not self._shape_check_done:
                self._validate_output_shapes(result, feats_np.shape[0])
                self._shape_check_done = True

            # DEBUG: Log successful inference
            if self.verbose:
                logging.debug(f"[DLAdapter] Inference successful: input_shape={feats_np.shape}, outputs={list(result.keys())}")

            return result

        except Exception as e:
            logging.error(f"[DLAdapter] shared_inference failed: input_shape={feats_np.shape if feats_np is not None else 'None'}, error={e}")
            raise  # Fail fast - no silent fallbacks

    def _validate_output_shapes(self, result: Dict[str, np.ndarray], batch_size: int):
        """Validate output shapes match strict contract."""
        required_keys = ["bridge_vec", "risk_budget", "pred_reward", "strat_immediate", "strat_short", "strat_medium", "strat_long"]

        for key in required_keys:
            if key not in result:
                raise KeyError(f"Missing required output key: {key}")

        # Validate bridge_vec
        assert result["bridge_vec"].ndim == 2, f"bridge_vec must be 2D, got {result['bridge_vec'].ndim}D"
        assert result["bridge_vec"].shape[0] == batch_size, f"bridge_vec batch mismatch"
        assert result["bridge_vec"].shape[1] == self.bridge_dim, f"bridge_vec dim mismatch"

        # Validate risk_budget
        assert result["risk_budget"].ndim in (1, 2), f"risk_budget must be 1D or 2D, got {result['risk_budget'].ndim}D"
        if result["risk_budget"].ndim == 2:
            assert result["risk_budget"].shape == (batch_size, 1), f"risk_budget shape mismatch"

        # Validate pred_reward
        assert result["pred_reward"].ndim in (1, 2), f"pred_reward must be 1D or 2D, got {result['pred_reward'].ndim}D"

        # Validate strategy heads (all must be (batch, 4))
        for head_name in ["strat_immediate", "strat_short", "strat_medium", "strat_long"]:
            head = result[head_name]
            assert head.ndim == 2, f"{head_name} must be 2D, got {head.ndim}D"
            assert head.shape == (batch_size, 4), f"{head_name} must be (batch, 4), got {head.shape}"

        logging.info(f"[DLAdapter] Output shapes validated: batch_size={batch_size}, all heads OK")


# ============================================================================
# OVERLAY TRAINER: TRAINS THE OVERLAY MODEL ON COLLECTED EXPERIENCE
# ============================================================================

class OverlayExperienceBuffer:
    """Circular buffer for storing overlay training experiences"""

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.validation_split = 0.2

    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample_batch(self, batch_size: int = 64) -> Dict[str, np.ndarray]:
        """Sample random batch from buffer (28D only)"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Stack into arrays (28D: immediate, short, medium, long)
        result = {
            'features': np.stack([b['features'] for b in batch]),
            'bridge_vec_target': np.stack([b['bridge_vec_target'] for b in batch]),
            'risk_budget_target': np.array([b['risk_budget_target'] for b in batch]),
            'pred_reward_target': np.array([b['pred_reward_target'] for b in batch]),
            'strat_immediate_target': np.stack([b['strat_immediate_target'] for b in batch]),
            'strat_short_target': np.stack([b['strat_short_target'] for b in batch]),
            'strat_medium_target': np.stack([b['strat_medium_target'] for b in batch]),
            'strat_long_target': np.stack([b['strat_long_target'] for b in batch]),
        }
        return result

    def get_validation_set(self) -> Dict[str, np.ndarray]:
        """Get validation set (last 20% of buffer, 28D only)"""
        val_size = max(1, int(len(self.buffer) * self.validation_split))
        val_indices = list(range(len(self.buffer) - val_size, len(self.buffer)))
        batch = [self.buffer[i] for i in val_indices]

        result = {
            'features': np.stack([b['features'] for b in batch]),
            'bridge_vec_target': np.stack([b['bridge_vec_target'] for b in batch]),
            'risk_budget_target': np.array([b['risk_budget_target'] for b in batch]),
            'pred_reward_target': np.array([b['pred_reward_target'] for b in batch]),
            'strat_immediate_target': np.stack([b['strat_immediate_target'] for b in batch]),
            'strat_short_target': np.stack([b['strat_short_target'] for b in batch]),
            'strat_medium_target': np.stack([b['strat_medium_target'] for b in batch]),
            'strat_long_target': np.stack([b['strat_long_target'] for b in batch]),
        }
        return result

    def size(self) -> int:
        return len(self.buffer)


class OverlayTrainer:
    """
    TIER 3 ONLY: Trains OverlaySharedModel on collected experience

    This class is used for online training of the DL overlay model.
    Tier 2 does NOT use this - it only uses pre-trained overlay for inference.
    """

    def __init__(self, model: OverlaySharedModel, learning_rate: float = 3e-3, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.buffer = OverlayExperienceBuffer(max_size=10_000)

        # Loss functions for each head (28D only)
        self.loss_fns = {
            'bridge_vec': tf.keras.losses.MeanSquaredError(),
            'risk_budget': tf.keras.losses.MeanSquaredError(),
            'pred_reward': tf.keras.losses.MeanSquaredError(),
            'strat_immediate': tf.keras.losses.MeanSquaredError(),
            'strat_short': tf.keras.losses.MeanSquaredError(),
            'strat_medium': tf.keras.losses.MeanSquaredError(),
            'strat_long': tf.keras.losses.MeanSquaredError(),
        }

        # Loss weights from config (OPTIMAL: Horizon-weighted for 28D)
        # Immediate signals most actionable → highest weight
        # Long-term signals less actionable → lower weight
        # Tuned for: immediate=100%, short=95%, medium=90%, long=risk-only
        self.loss_weights = DL_OVERLAY_LOSS_WEIGHTS.copy()

        # Per-horizon metrics for monitoring
        self.horizon_metrics = {
            'strat_immediate': {'mae': deque(maxlen=100), 'rmse': deque(maxlen=100)},
            'strat_short': {'mae': deque(maxlen=100), 'rmse': deque(maxlen=100)},
            'strat_medium': {'mae': deque(maxlen=100), 'rmse': deque(maxlen=100)},
            'strat_long': {'mae': deque(maxlen=100), 'rmse': deque(maxlen=100)},
        }

        # Metrics
        self.training_history = deque(maxlen=100)
        self.validation_history = deque(maxlen=100)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10

        logging.info(f"OverlayTrainer initialized with learning_rate={learning_rate}")

    def add_experience(self, experience: Dict[str, Any]):
        """Add experience to training buffer"""
        self.buffer.add(experience)

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single training step on batch (28D only)"""
        try:
            features = tf.convert_to_tensor(batch['features'], dtype=tf.float32)

            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self.model(features, training=True)

                # Compute loss for each head
                total_loss = 0.0
                losses = {}

                # Process all heads (28D: immediate, short, medium, long)
                for head_name in ['bridge_vec', 'risk_budget', 'pred_reward', 'strat_immediate', 'strat_short', 'strat_medium', 'strat_long']:
                    target_key = f'{head_name}_target'
                    target = tf.convert_to_tensor(batch[target_key], dtype=tf.float32)
                    pred = predictions[head_name]

                    # Reshape risk_budget and pred_reward targets to match predictions
                    if head_name in ['risk_budget', 'pred_reward']:
                        target = tf.reshape(target, (-1, 1))

                    head_loss = self.loss_fns[head_name](target, pred)
                    weighted_loss = head_loss * self.loss_weights[head_name]
                    total_loss += weighted_loss
                    losses[head_name] = float(head_loss.numpy())

                    # Compute per-horizon metrics for strategy heads
                    if head_name in self.horizon_metrics:
                        mae = float(tf.reduce_mean(tf.abs(target - pred)).numpy())
                        rmse = float(tf.sqrt(tf.reduce_mean(tf.square(target - pred))).numpy())
                        self.horizon_metrics[head_name]['mae'].append(mae)
                        self.horizon_metrics[head_name]['rmse'].append(rmse)

                losses['total'] = float(total_loss.numpy())

            # Backward pass
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return losses

        except Exception as e:
            logging.error(f"Training step failed: {e}")
            raise  # Fail fast

    def train(self, epochs: int = 3, batch_size: int = 64) -> Dict[str, float]:
        """Train model on buffered experiences"""
        if self.buffer.size() < batch_size:
            return {'status': 'insufficient_data'}

        epoch_losses = []

        for epoch in range(epochs):
            batch = self.buffer.sample_batch(batch_size)
            losses = self.train_step(batch)
            epoch_losses.append(losses['total'])

        avg_loss = float(np.mean(epoch_losses))
        self.training_history.append(avg_loss)

        # Validation
        val_loss = self.validate()
        self.validation_history.append(val_loss)

        # Early stopping check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.verbose:
            logging.info(f"[OVERLAY-TRAIN] train_loss={avg_loss:.6f} val_loss={val_loss:.6f} patience={self.patience_counter}")

        return {
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'patience': self.patience_counter,
            'should_stop': self.patience_counter >= self.max_patience
        }

    def train_meta_baseline(self, features_batch: np.ndarray, adv_targets: np.ndarray, loss_type: str = "mse") -> float:
        """
        FAMC: Train meta-critic head g_φ(x_t) to predict advantages.

        This head learns to predict PPO advantages from state features only (no actions).
        The predictions are used as control variates to reduce advantage variance.

        Args:
            features_batch: State features (batch, 34) - overlay input features
            adv_targets: Standardized advantages (batch, 1) - mean=0, std=1
            loss_type: {"mse", "corr"} - loss function

        Returns:
            Loss value (float)

        CRITICAL: This function must ONLY use state features (no action leakage).
        """
        try:
            if not hasattr(self.model, 'meta_head') or self.model.meta_head is None:
                logging.warning("[FAMC] Meta head not initialized, skipping training")
                return 0.0

            # Convert to tensors
            features = tf.convert_to_tensor(features_batch, dtype=tf.float32)
            targets = tf.convert_to_tensor(adv_targets, dtype=tf.float32)

            # Ensure targets are (batch, 1)
            if len(targets.shape) == 1:
                targets = tf.reshape(targets, (-1, 1))

            with tf.GradientTape() as tape:
                # Forward pass through meta head only
                pred = self.model.meta_head(self.model.d4(
                    self.model.do3(self.model.d3(
                        self.model.do2(self.model.d2(
                            self.model.do1(self.model.d1(features), training=True)
                        ), training=True)
                    ), training=True)
                ), training=True)

                if loss_type == "mse":
                    # MSE loss (targets should be standardized outside)
                    loss = tf.reduce_mean(tf.square(pred - targets))
                elif loss_type == "corr":
                    # Negative Pearson correlation (maximize correlation)
                    x = pred - tf.reduce_mean(pred)
                    y = targets - tf.reduce_mean(targets)
                    corr = tf.reduce_sum(x * y) / (
                        tf.sqrt(tf.reduce_sum(x * x)) * tf.sqrt(tf.reduce_sum(y * y)) + 1e-8
                    )
                    loss = -corr  # Negative because we want to maximize correlation
                else:
                    logging.error(f"[FAMC] Unknown loss_type: {loss_type}, using MSE")
                    loss = tf.reduce_mean(tf.square(pred - targets))

            # Backward pass (only update meta head parameters)
            vars = self.model.meta_head.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            return float(loss.numpy())

        except Exception as e:
            logging.error(f"[FAMC] Meta baseline training failed: {e}")
            return 0.0

    def validate(self) -> float:
        """Compute validation loss (28D only)"""
        try:
            if self.buffer.size() < 10:
                return 0.0

            val_batch = self.buffer.get_validation_set()
            features = tf.convert_to_tensor(val_batch['features'], dtype=tf.float32)
            predictions = self.model(features, training=False)

            total_loss = 0.0
            # Validate all heads (28D: immediate, short, medium, long)
            for head_name in ['bridge_vec', 'risk_budget', 'pred_reward', 'strat_immediate', 'strat_short', 'strat_medium', 'strat_long']:
                target_key = f'{head_name}_target'
                target = tf.convert_to_tensor(val_batch[target_key], dtype=tf.float32)
                pred = predictions[head_name]

                if head_name in ['risk_budget', 'pred_reward']:
                    target = tf.reshape(target, (-1, 1))

                head_loss = self.loss_fns[head_name](target, pred)
                total_loss += head_loss * self.loss_weights[head_name]

            return float(total_loss.numpy())

        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise  # Fail fast


# ============================================================================
# TIER 3 ONLY: FGB CALIBRATION TRACKER (Forecast-Guided Baseline)
# ============================================================================

class CalibrationTracker:
    """
    TIER 3 ONLY: FGB - Rolling online calibration for forecast trust (τₜ) and expected ΔNAV.

    This class tracks forecast accuracy over a rolling window and computes:
    1. Trust τₜ ∈ [0,1]: How much to trust the forecast (based on hit-rate, MAE, etc.)
    2. Expected ΔNAV: Forecasted change in NAV for the next step

    The DL overlay remains a forecaster, not a controller.
    PPO remains the action decision-maker; we reduce advantage variance via baseline adjustment.

    TIER 2: This is NOT used (forecast trust is computed differently)
    """

    def __init__(self, window_size: int = None, trust_metric: str = "combo", verbose: bool = False, init_budget: Optional[float] = None, direction_weight: float = None):
        """
        Args:
            window_size: Rolling window size for calibration (default from config: 2016 ≈ 2 weeks @ 10-min steps)
            trust_metric: {"combo", "hitrate", "absdir"} - method for computing trust
            verbose: Enable debug logging
            init_budget: Initial fund NAV (used to scale expected_dnav when positions are flat, default from config)
            direction_weight: Weight for directional accuracy in combo metric (default from config: 0.8 = 80% direction, 20% magnitude)
        """
        self.window_size = window_size if window_size is not None else DL_OVERLAY_WINDOW_SIZE_DEFAULT
        self.trust_metric = trust_metric
        self.verbose = verbose
        self.init_budget = float(init_budget) if init_budget is not None else DL_OVERLAY_INIT_BUDGET_DEFAULT
        self.direction_weight = float(direction_weight) if direction_weight is not None else DL_OVERLAY_DIRECTION_WEIGHT_DEFAULT

        # Rolling history for calibration
        self.forecast_history = deque(maxlen=window_size)  # (forecast, realized) pairs
        self.direction_history = deque(maxlen=window_size)  # Direction hit/miss

        # Per-horizon tracking (optional, for future multi-horizon trust)
        self.horizon_stats = {
            'short': {'hits': 0, 'total': 0, 'mae': deque(maxlen=window_size)},
            'medium': {'hits': 0, 'total': 0, 'mae': deque(maxlen=window_size)},
            'long': {'hits': 0, 'total': 0, 'mae': deque(maxlen=window_size)},
        }

        # Cache for efficiency
        self._trust_cache = None
        self._cache_step = -1

        if self.verbose:
            logging.info(f"[CalibrationTracker] Initialized: window={window_size}, metric={trust_metric}, init_budget={self.init_budget:,.0f}")

    def update(self, forecast: float, realized: float, horizon: str = "short"):
        """
        FGB: Update calibration with a new forecast/realized pair.

        Args:
            forecast: Forecasted value (e.g., pred_reward, mwdir, or ΔNAV proxy)
            realized: Realized value (e.g., actual reward, price change, or ΔNAV)
            horizon: {"short", "medium", "long"} - forecast horizon (optional)
        """
        # Store forecast/realized pair
        self.forecast_history.append((forecast, realized))

        # Track direction accuracy
        if abs(forecast) > 1e-6 and abs(realized) > 1e-6:
            direction_match = np.sign(forecast) == np.sign(realized)
            self.direction_history.append(1.0 if direction_match else 0.0)

        # Update per-horizon stats
        if horizon in self.horizon_stats:
            stats = self.horizon_stats[horizon]
            stats['total'] += 1
            if abs(forecast) > 1e-6 and abs(realized) > 1e-6:
                if np.sign(forecast) == np.sign(realized):
                    stats['hits'] += 1
            stats['mae'].append(abs(forecast - realized))

        # Invalidate cache
        self._trust_cache = None

    def get_trust(self, horizon: str = "short", recent_mape: Optional[float] = None) -> float:
        """
        FGB: Compute forecast trust τₜ ∈ [0,1].

        Trust is computed based on rolling calibration metrics:
        - "hitrate": Direction hit-rate (2 * hit_rate - 1)
        - "absdir": Absolute value of mwdir (smoothed)
        - "combo": Weighted combination of hit-rate and MAE

        FIX: Now includes MAPE-based trust component for responsiveness to forecast quality.

        Args:
            horizon: {"short", "medium", "long"} - forecast horizon
            recent_mape: Optional recent MAPE value to incorporate into trust calculation

        Returns:
            τₜ ∈ [0,1]: Trust score (0 = no trust, 1 = full trust)
        """
        # Return cached value if available (same step)
        if self._trust_cache is not None:
            return self._trust_cache

        # Not enough data yet - buffer is warming up
        buffer_size = len(self.forecast_history)
        if buffer_size < 10:
            # FIX #3: Return neutral trust (0.5) instead of 0.0 during warm-up
            # This prevents forecast rewards from being cut in half during early training
            # With trust=0.5, trust_scale = 0.5 + (1.5-0.5)*0.5 = 1.0 (no penalty)
            # Log warning every 10 samples to inform user about warm-up
            if buffer_size > 0 and buffer_size % 10 == 0:
                logging.warning(
                    f"[CalibrationTracker] Buffer warm-up: {buffer_size}/10 samples collected. "
                    f"Trust will remain 0.5 (neutral) until 10 samples are available. "
                    f"Forecast rewards will use neutral scaling (1.0x) during warm-up."
                )
            return 0.5  # FIX #3: Changed from 0.0 to 0.5 (neutral trust)

        # Compute trust based on selected metric
        if self.trust_metric == "hitrate":
            # Direction hit-rate: 2 * hit_rate - 1 ∈ [-1, 1], then clip to [0, 1]
            if len(self.direction_history) > 0:
                hit_rate = np.mean(list(self.direction_history))
                trust = max(0.0, 2.0 * hit_rate - 1.0)
            else:
                trust = 0.0

        elif self.trust_metric == "absdir":
            # Use absolute value of recent forecasts (proxy for confidence)
            recent_forecasts = [f for f, _ in list(self.forecast_history)[-100:]]
            if len(recent_forecasts) > 0:
                trust = min(1.0, np.mean(np.abs(recent_forecasts)))
            else:
                trust = 0.0

        elif self.trust_metric == "combo":
            # Combo: direction_weight * (2·hit_rate - 1) + (1 - direction_weight) * (1 - norm_mae)
            # Default: 0.8 * direction + 0.2 * magnitude (was 0.5/0.5)
            # This gives more weight to directional accuracy, which is more robust to tanh compression
            # RATIONALE: With 75% directional accuracy, 80/20 weighting yields trust ≈ 0.6 (threshold)
            if len(self.direction_history) > 0:
                hit_rate = np.mean(list(self.direction_history))
                hit_component = max(0.0, 2.0 * hit_rate - 1.0)
            else:
                hit_component = 0.0

            # MAE component (normalized by typical forecast magnitude)
            if len(self.forecast_history) > 0:
                forecasts = [f for f, _ in list(self.forecast_history)]
                realized = [r for _, r in list(self.forecast_history)]
                mae = np.mean(np.abs(np.array(forecasts) - np.array(realized)))
                typical_mag = np.mean(np.abs(realized)) + 1e-6
                norm_mae = min(1.0, mae / typical_mag)
                mae_component = max(0.0, 1.0 - norm_mae)
            else:
                mae_component = 0.0

            # Use configurable weights (default: 80% direction, 20% magnitude)
            dir_weight = getattr(self, 'direction_weight', 0.8)
            mag_weight = 1.0 - dir_weight
            trust = dir_weight * hit_component + mag_weight * mae_component

        else:
            logging.warning(f"[CalibrationTracker] Unknown trust_metric: {self.trust_metric}, defaulting to 0.0")
            trust = 0.0

        # FIX: Incorporate MAPE-based trust component for responsiveness to forecast quality
        # Low MAPE (accurate forecasts) → higher trust
        # High MAPE (inaccurate forecasts) → lower trust
        if recent_mape is not None and recent_mape > 0:
            # Normalize MAPE: typical MAPE is ~1.0, so map [0, 2.0] → [1.0, 0.0]
            # MAPE = 0.0 → mape_factor = 1.0 (perfect)
            # MAPE = 1.0 → mape_factor = 0.5 (moderate)
            # MAPE = 2.0 → mape_factor = 0.0 (poor)
            max_expected_mape = 2.0  # Typical max MAPE for normalization
            mape_factor = max(0.0, 1.0 - min(recent_mape / max_expected_mape, 1.0))
            
            # Combine correlation-based trust (60%) with MAPE-based trust (40%)
            # This makes trust responsive to actual forecast quality
            trust = 0.6 * trust + 0.4 * mape_factor

        # FIX: Apply moderate optimistic boost to trust calculation
        # Current trust (0.438 avg) is too low - moderate boost to encourage forecast usage
        # Boost trust by 10% (cap at 1.0) - balanced approach, not too aggressive
        # This helps agent learn to use forecasts without over-trusting bad forecasts
        trust_boost = 0.1
        trust = min(1.0, trust * (1.0 + trust_boost))

        # Clip to [0, 1]
        trust = float(np.clip(trust, 0.0, 1.0))

        # Cache result
        self._trust_cache = trust

        if self.verbose and len(self.forecast_history) % 500 == 0:
            mape_info = f", MAPE={recent_mape:.4f}" if recent_mape is not None else ""
            logging.info(f"[CalibrationTracker] Trust τₜ={trust:.3f} (metric={self.trust_metric}, n={len(self.forecast_history)}{mape_info})")

        return trust

    def expected_dnav(self, overlay_output: Dict[str, Any], positions: Dict[str, float],
                     costs: Dict[str, float]) -> float:
        """
        FGB: Compute expected ΔNAV for the next step given current state.

        Maps forecast to expected ΔNAV using a linear proxy:
            E[ΔNAV] = exposure * pred_reward * mwdir - est_costs

        Args:
            overlay_output: Dict with keys {"pred_reward", "mwdir", ...}
            positions: Dict with current positions (e.g., {"wind": 1000, "solar": 500, ...})
            costs: Dict with transaction cost params (e.g., {"bps": 5, "fixed": 100})

        Returns:
            Expected ΔNAV (scalar, can be positive or negative)
        """
        try:
            # Extract forecast signals
            pred_reward = float(overlay_output.get("pred_reward", 0.0))
            if isinstance(pred_reward, np.ndarray):
                pred_reward = float(pred_reward.flatten()[0])

            mwdir = float(overlay_output.get("mwdir", 0.0))

            # Compute exposure (total notional of current positions)
            # FGB: Use sum of absolute position values as exposure proxy
            exposure = sum(abs(float(v)) for v in positions.values() if isinstance(v, (int, float)))

            # If no positions, use a small fraction of fund NAV as proxy
            # This ensures FGB has bite early when policy is still flat
            if exposure < 1e-6:
                exposure = 0.01 * self.init_budget  # 1% of initial NAV

            # Expected ΔNAV (linear proxy)
            # pred_reward is in [-1, 1], mwdir is in [-1, 1]
            # Scale by exposure to get DKK units
            exp_dnav_gross = exposure * pred_reward * mwdir

            # Estimate transaction costs (if positions change)
            # FGB: Assume typical trade size is 10% of exposure
            typical_trade_notional = 0.1 * exposure
            bps = costs.get("bps", 5.0)
            fixed = costs.get("fixed", 100.0)
            est_costs = (typical_trade_notional * bps / 10000.0) + fixed

            # Net expected ΔNAV
            exp_dnav = exp_dnav_gross - est_costs

            return float(exp_dnav)

        except Exception as e:
            logging.warning(f"[CalibrationTracker] expected_dnav failed: {e}")
            return 0.0

    def reset(self):
        """FGB: Reset calibration history (e.g., at episode boundaries)."""
        self.forecast_history.clear()
        self.direction_history.clear()
        for stats in self.horizon_stats.values():
            stats['hits'] = 0
            stats['total'] = 0
            stats['mae'].clear()
        self._trust_cache = None

        if self.verbose:
            logging.info("[CalibrationTracker] Reset calibration history")


# ============================================================================
# SUMMARY
# ============================================================================

# Removed startup message - only log when overlay is actually activated
