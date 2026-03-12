#!/usr/bin/env python3
"""
Tier-2 DL Enhancer - Forecast-backed investor exposure adjustment

Uses short-horizon forecast memory plus realized investor outcome to learn a
residual exposure adjustment that improves sizing and risk control.

Streamlined full model (12D):
- 4D core: [proposed_exposure, gross_exposure_ratio, tradeable_capital_ratio,
            realized_volatility_regime]
- 2 x 4 memory channels: [price_short_signal, short_revision, forecast_quality,
                         short_imbalance_signal]

Design: Learns from realized outcomes; GRU over forecast memory; uncertainty-aware
output (sigma scales delta). Minimal feature set for faster training and less overfit.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU
from typing import Dict, Any, Optional
from collections import deque
import logging

from config import (
    ENHANCER_BASE_FEATURE_DIM,
    TIER2_ENHANCER_MEMORY_CHANNELS,
    TIER2_ENHANCER_MEMORY_STEPS,
    TIER2_ENHANCER_FEATURE_DIM,
    TIER2_ENHANCER_ABLATED_FEATURE_DIM,
    FORECAST_BASE_WINDOW_SIZE_DEFAULT,
    FORECAST_BASE_INIT_BUDGET_DEFAULT,
    FORECAST_BASE_DIRECTION_WEIGHT_DEFAULT,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CalibrationTracker - Rolling forecast trust and expected ΔNAV
# (used by get_fgb_trust_for_agent and evaluation)
# ============================================================================

class CalibrationTracker:
    """
    Tier-2 rolling calibration for forecast trust (τₜ) and expected ΔNAV.
    """

    def __init__(
        self,
        window_size: int = None,
        trust_metric: str = "combo",
        verbose: bool = False,
        init_budget: Optional[float] = None,
        direction_weight: float = None,
        trust_boost: float = 0.0,
        min_exposure_ratio: float = 0.0,
        pred_weight: float = 0.85,
        mwdir_weight: float = 0.15,
        fail_fast: bool = False,
    ):
        self.window_size = window_size if window_size is not None else FORECAST_BASE_WINDOW_SIZE_DEFAULT
        self.trust_metric = trust_metric
        self.verbose = verbose
        self.init_budget = float(init_budget) if init_budget is not None else FORECAST_BASE_INIT_BUDGET_DEFAULT
        self.direction_weight = float(direction_weight) if direction_weight is not None else FORECAST_BASE_DIRECTION_WEIGHT_DEFAULT
        self.trust_boost = max(0.0, float(trust_boost or 0.0))
        self.min_exposure_ratio = max(0.0, float(min_exposure_ratio or 0.0))
        self.pred_weight = float(pred_weight)
        self.mwdir_weight = float(mwdir_weight)
        self.fail_fast = bool(fail_fast)
        self.forecast_history = deque(maxlen=self.window_size)
        self.direction_history = deque(maxlen=self.window_size)
        self.horizon_stats = {
            'short': {'hits': 0, 'total': 0, 'mae': deque(maxlen=self.window_size)},
            'medium': {'hits': 0, 'total': 0, 'mae': deque(maxlen=self.window_size)},
            'long': {'hits': 0, 'total': 0, 'mae': deque(maxlen=self.window_size)},
        }
        self._trust_cache: Dict[str, float] = {}

    def update(self, forecast: float, realized: float, horizon: str = "short"):
        self.forecast_history.append((forecast, realized))
        if abs(forecast) > 1e-6 and abs(realized) > 1e-6:
            self.direction_history.append(1.0 if (np.sign(forecast) == np.sign(realized)) else 0.0)
        if horizon in self.horizon_stats:
            stats = self.horizon_stats[horizon]
            stats['total'] += 1
            if abs(forecast) > 1e-6 and abs(realized) > 1e-6 and np.sign(forecast) == np.sign(realized):
                stats['hits'] += 1
            stats['mae'].append(abs(forecast - realized))
        self._trust_cache.clear()

    def get_trust(self, horizon: str = "short", recent_mape: Optional[float] = None) -> float:
        h = str(horizon or "short").strip().lower()
        if h not in self.horizon_stats:
            h = "short"
        stats = self.horizon_stats.get(h, {})
        buffer_size = int(stats.get('total', 0))
        if buffer_size < 10:
            return 0.5
        if self.trust_metric == "hitrate":
            hit_rate = stats.get('hits', 0) / max(1.0, float(buffer_size))
            return max(0.0, 2.0 * float(hit_rate) - 1.0)
        hits = float(stats.get('hits', 0))
        hit_rate = hits / max(1.0, float(buffer_size))
        mae_vals = list(stats.get('mae', []))
        norm_mae = (1.0 - min(np.mean(mae_vals) / 0.5, 1.0)) if mae_vals else 0.5
        return float(np.clip(
            self.direction_weight * (2.0 * hit_rate - 1.0) + (1.0 - self.direction_weight) * norm_mae,
            0.0, 1.0
        ))

    def expected_dnav(self, forecast_base_output: Dict[str, Any], positions: Dict[str, float],
                     costs: Dict[str, float]) -> float:
        """Compute expected ΔNAV. When forecast-base output is absent, return 0."""
        if not forecast_base_output:
            return 0.0
        pred = forecast_base_output.get("pred_reward", 0.0)
        if isinstance(pred, (list, np.ndarray)):
            pred = float(np.ravel(pred)[0]) if np.size(pred) > 0 else 0.0
        mwdir = float(forecast_base_output.get("mwdir", 0.0))
        if isinstance(mwdir, np.ndarray):
            mwdir = float(np.ravel(mwdir)[0]) if np.size(mwdir) > 0 else 0.0
        blend = self.pred_weight * float(pred) + self.mwdir_weight * float(mwdir)
        return float(np.clip(blend, -1.0, 1.0))

    def reset(self):
        self.forecast_history.clear()
        self.direction_history.clear()
        for h in self.horizon_stats:
            self.horizon_stats[h]['hits'] = 0
            self.horizon_stats[h]['total'] = 0
            self.horizon_stats[h]['mae'].clear()
        self._trust_cache.clear()


# ============================================================================
# Tier-2 DL Enhancer model and training
# ============================================================================


class Tier2EnhancerModel(tf.keras.Model):
    """
    DL model for Tier-2 Enhancer: learns a residual signed exposure correction.
    Input: 29D (full) or 5D (ablated: non-forecast core only)
    Output:
    - mean exposure delta in [-1, 1] (later scaled by delta_max)
    - predictive scale (>0) for continuous uncertainty discounting
    Reproducible: uses seed for deterministic initialization.
    """

    def __init__(self, feature_dim: int, seed: int = 42):
        super().__init__()
        self.feature_dim = feature_dim
        valid_dims = (
            TIER2_ENHANCER_FEATURE_DIM,
            TIER2_ENHANCER_ABLATED_FEATURE_DIM,
        )
        if feature_dim not in valid_dims:
            raise ValueError(
                f"Enhancer expects {TIER2_ENHANCER_FEATURE_DIM}D, "
                f"or {TIER2_ENHANCER_ABLATED_FEATURE_DIM}D; "
                f"got {feature_dim}D"
            )
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        self.core_dim = 4
        self.is_full_model = bool(feature_dim == TIER2_ENHANCER_FEATURE_DIM)
        if self.is_full_model:
            self.memory_steps = int(TIER2_ENHANCER_MEMORY_STEPS)
            self.memory_channels = int(TIER2_ENHANCER_MEMORY_CHANNELS)
            if feature_dim != self.core_dim + self.memory_steps * self.memory_channels:
                raise ValueError(
                    f"Full enhancer expects {self.core_dim} + {self.memory_steps}x{self.memory_channels} = "
                    f"{self.core_dim + self.memory_steps * self.memory_channels} features; got {feature_dim}"
                )
            self.core_d1 = Dense(32, activation='relu', name='enhancer_core_d1')
            self.core_d2 = Dense(16, activation='relu', name='enhancer_core_d2')
            self.memory_gru = GRU(16, name='enhancer_memory_gru')
            self.latest_d = Dense(8, activation='relu', name='enhancer_latest_d')
            self.fuse_d1 = Dense(32, activation='relu', name='enhancer_fuse_d1')
            self.do1 = Dropout(0.08, name='enhancer_fuse_do1')
            self.fuse_d2 = Dense(16, activation='relu', name='enhancer_fuse_d2')
        else:
            # Ablated 4D: core only.
            self.d1 = Dense(64, activation='relu', name='enhancer_d1')
            self.do1 = Dropout(0.10, name='enhancer_do1')
            self.d2 = Dense(32, activation='relu', name='enhancer_d2')
            self.do2 = Dropout(0.08, name='enhancer_do2')
            self.d3 = Dense(16, activation='relu', name='enhancer_d3')
        self.delta_head = Dense(1, activation='tanh', name='exposure_delta_raw')
        self.sigma_head = Dense(
            1,
            activation='softplus',
            bias_initializer=tf.keras.initializers.Constant(-2.0),
            name='exposure_delta_sigma_raw',
        )

        logger.info(f"Tier2EnhancerModel initialized: {feature_dim}D features")

    def call(self, x, training=None):
        if self.is_full_model:
            core = x[:, :self.core_dim]
            memory_flat = x[:, self.core_dim:]
            memory = tf.reshape(memory_flat, (-1, self.memory_steps, self.memory_channels))

            core_h = self.core_d1(core)
            core_h = self.core_d2(core_h)
            memory_h = self.memory_gru(memory)
            latest_h = self.latest_d(memory[:, -1, :])
            cross = core_h * memory_h
            fused = tf.concat([core_h, memory_h, latest_h, cross], axis=1)
            z = self.fuse_d1(fused)
            z = self.do1(z, training=training)
            z = self.fuse_d2(z)
        else:
            z = self.d1(x)
            z = self.do1(z, training=training)
            z = self.d2(z)
            z = self.do2(z, training=training)
            z = self.d3(z)
        delta_raw = self.delta_head(z)  # [-1, 1]
        sigma_raw = self.sigma_head(z)  # > 0
        return tf.concat([delta_raw, sigma_raw], axis=1)


class EnhancerAdapter:
    """
    Thin adapter for env to call enhancer inference.
    Manages model and exposes enhancer adjustment inference.
    """

    def __init__(
        self,
        feature_dim: int,
        delta_max: float = 0.35,
        seed: int = 42,
        uncertainty_discount: float = 1.10,
    ):
        fd = int(feature_dim)
        valid_dims = (
            TIER2_ENHANCER_FEATURE_DIM,
            TIER2_ENHANCER_ABLATED_FEATURE_DIM,
        )
        if fd not in valid_dims:
            raise ValueError(
                f"EnhancerAdapter expects {TIER2_ENHANCER_FEATURE_DIM}D, "
                f"or {TIER2_ENHANCER_ABLATED_FEATURE_DIM}D; "
                f"got {feature_dim}D"
            )
        self.feature_dim = feature_dim
        self.delta_max = float(max(0.0, delta_max))
        self.uncertainty_discount = float(max(0.0, uncertainty_discount))
        self.model = Tier2EnhancerModel(feature_dim, seed=seed)

    def predict_adjustment(self, feats_np: np.ndarray, training: bool = False) -> Dict[str, float]:
        """
        Run inference and return exposure adjustment components.
        """
        if len(feats_np.shape) == 1:
            feats_np = feats_np.reshape(1, -1)
        if feats_np.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim}D features, got {feats_np.shape[1]}D")

        feats_tensor = tf.convert_to_tensor(feats_np, dtype=tf.float32)
        out = self.model(feats_tensor, training=training).numpy().reshape(-1)
        delta_raw_val = float(out[0])
        sigma_unit = float(np.clip(out[1], 1e-3, 2.0))
        base_exposure = float(np.clip(feats_np[0, 0], -1.0, 1.0))
        raw_delta = float(np.clip(delta_raw_val * self.delta_max, -self.delta_max, self.delta_max))
        uncertainty_scale = float(np.exp(-self.uncertainty_discount * sigma_unit))
        delta = float(np.clip(raw_delta * uncertainty_scale, -self.delta_max, self.delta_max))
        pred_sigma = float(np.clip(sigma_unit * self.delta_max, 1e-4, self.delta_max))
        target_exposure = float(np.clip(base_exposure + delta, -1.0, 1.0))
        return {
            "delta": delta,
            "raw_delta": raw_delta,
            "pred_sigma": pred_sigma,
            "uncertainty_scale": uncertainty_scale,
            "target_exposure": target_exposure,
        }


class EnhancerExperienceBuffer:
    """Circular buffer for enhancer training. Reproducible sampling when seed is set."""

    def __init__(self, max_size: int = 10_000, seed: Optional[int] = None):
        self.buffer = deque(maxlen=max_size)
        self.validation_split = 0.2
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def add(self, experience: Dict[str, Any]):
        self.buffer.append(experience)

    def sample_batch(self, batch_size: int = 64) -> Dict[str, np.ndarray]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        rng = self._rng if self._rng is not None else np.random
        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return {
            'features': np.stack([b['features'] for b in batch]),
            'exposure_delta_target': np.array([b['exposure_delta_target'] for b in batch], dtype=np.float32),
            'decision_weight': np.array([b.get('decision_weight', 1.0) for b in batch], dtype=np.float32),
            'sharpe_signal': np.array([b.get('sharpe_signal', 0.0) for b in batch], dtype=np.float32),
        }

    def get_validation_set(self) -> Dict[str, np.ndarray]:
        val_size = max(1, int(len(self.buffer) * self.validation_split))
        val_indices = list(range(len(self.buffer) - val_size, len(self.buffer)))
        batch = [self.buffer[i] for i in val_indices]
        return {
            'features': np.stack([b['features'] for b in batch]),
            'exposure_delta_target': np.array([b['exposure_delta_target'] for b in batch], dtype=np.float32),
            'decision_weight': np.array([b.get('decision_weight', 1.0) for b in batch], dtype=np.float32),
            'sharpe_signal': np.array([b.get('sharpe_signal', 0.0) for b in batch], dtype=np.float32),
        }

    def size(self) -> int:
        return len(self.buffer)


class EnhancerTrainer:
    """
    Trains Tier2EnhancerModel on realized investor outcome.
    Target generation happens in the environment; the trainer only fits the
    residual exposure-delta regression.
    """

    def __init__(self, model: Tier2EnhancerModel, delta_max: float,
                 learning_rate: float = 3e-3, seed: int = 42,
                 intervention_l1: float = 0.02,
                 overconfidence_penalty: float = 0.05,
                 direction_loss_weight: float = 0.08,
                 sharpe_loss_weight: float = 0.30):
        self.model = model
        self.delta_max = float(max(0.0, delta_max))
        self.learning_rate = float(learning_rate)
        self.intervention_l1 = float(max(0.0, intervention_l1))
        self.overconfidence_penalty = float(max(0.0, overconfidence_penalty))
        self.direction_loss_weight = float(max(0.0, direction_loss_weight))
        self.sharpe_loss_weight = float(max(0.0, sharpe_loss_weight))
        self.buffer = EnhancerExperienceBuffer(max_size=10_000, seed=seed)

        self.train_vars = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self._build_optimizer()
        logger.info(
            f"EnhancerTrainer initialized: delta_max={self.delta_max:.3f} "
            f"lr={self.learning_rate:.6f} l1={self.intervention_l1:.3f} "
            f"overconf={self.overconfidence_penalty:.3f} "
            f"dir_loss={self.direction_loss_weight:.3f} "
            f"sharpe_loss={self.sharpe_loss_weight:.3f}"
        )

    def _build_optimizer(self):
        if not self.model.built:
            dummy = tf.zeros((1, self.model.feature_dim), dtype=tf.float32)
            _ = self.model(dummy, training=False)
        self.train_vars = list(self.model.trainable_variables)
        try:
            if hasattr(self.optimizer, "build") and callable(self.optimizer.build):
                self.optimizer.build(self.train_vars)
        except Exception as e:
            logger.warning(f"Enhancer optimizer build skipped: {e}")

    def add_experience(self, experience: Dict[str, Any]):
        self.buffer.add(experience)

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = tf.convert_to_tensor(batch['features'], dtype=tf.float32)
        targets = tf.convert_to_tensor(batch['exposure_delta_target'], dtype=tf.float32)
        targets = tf.reshape(targets, (-1, 1))
        decision_weight = tf.convert_to_tensor(
            batch.get('decision_weight', np.ones(len(batch['features']), dtype=np.float32)),
            dtype=tf.float32,
        )
        decision_weight = tf.reshape(tf.clip_by_value(decision_weight, 1e-3, 1.0), (-1, 1))
        sharpe_signal = tf.convert_to_tensor(
            batch.get('sharpe_signal', np.zeros(len(batch['features']), dtype=np.float32)),
            dtype=tf.float32,
        )
        sharpe_signal = tf.reshape(tf.clip_by_value(sharpe_signal, -1.0, 1.0), (-1, 1))
        train_vars = list(self.model.trainable_variables)
        if not train_vars:
            self._build_optimizer()
            train_vars = list(self.model.trainable_variables)

        with tf.GradientTape() as tape:
            out = self.model(features, training=True)
            delta_raw = tf.clip_by_value(out[:, :1], -1.0, 1.0)
            sigma_unit = tf.clip_by_value(out[:, 1:2], 1e-3, 2.0)
            pred = self.delta_max * delta_raw
            pred_sigma = tf.clip_by_value(self.delta_max * sigma_unit, 1e-4, self.delta_max)
            error = targets - pred
            nll = 0.5 * tf.square(error / pred_sigma) + tf.math.log(pred_sigma + 1e-6)
            base_loss = tf.reduce_sum(decision_weight * nll) / (tf.reduce_sum(decision_weight) + 1e-6)
            aux_loss = tf.constant(0.0, dtype=tf.float32)
            reg_loss = tf.constant(0.0, dtype=tf.float32)
            overconfidence_loss = tf.constant(0.0, dtype=tf.float32)
            decision_loss = tf.constant(0.0, dtype=tf.float32)
            sharpe_loss = tf.constant(0.0, dtype=tf.float32)
            loss = base_loss

            if self.intervention_l1 > 0.0:
                reg_loss = tf.reduce_sum(decision_weight * tf.abs(pred)) / (
                    tf.reduce_sum(decision_weight) + 1e-6
                )
                loss = loss + self.intervention_l1 * reg_loss

            if self.overconfidence_penalty > 0.0:
                overconfidence = tf.nn.relu(tf.abs(error) - 1.5 * pred_sigma)
                overconfidence_loss = tf.reduce_sum(decision_weight * overconfidence) / (
                    tf.reduce_sum(decision_weight) + 1e-6
                )
                loss = loss + self.overconfidence_penalty * overconfidence_loss

            if self.direction_loss_weight > 0.0:
                target_sign = tf.sign(targets)
                active_mask = tf.cast(tf.abs(targets) > 1e-4, tf.float32)
                normalized_margin = (target_sign * pred) / max(self.delta_max, 1e-6)
                direction_penalty = tf.nn.softplus(-normalized_margin)
                decision_loss = tf.reduce_sum(decision_weight * active_mask * direction_penalty) / (
                    tf.reduce_sum(decision_weight * tf.maximum(active_mask, 1e-3)) + 1e-6
                )
                loss = loss + self.direction_loss_weight * decision_loss

            if self.sharpe_loss_weight > 0.0:
                sharpe_mask = tf.cast(tf.abs(sharpe_signal) > 1e-4, tf.float32)
                sharpe_margin = (sharpe_signal * pred) / max(self.delta_max, 1e-6)
                sharpe_penalty = tf.nn.softplus(-sharpe_margin)
                sharpe_loss = tf.reduce_sum(decision_weight * sharpe_mask * sharpe_penalty) / (
                    tf.reduce_sum(decision_weight * tf.maximum(sharpe_mask, 1e-3)) + 1e-6
                )
                loss = loss + self.sharpe_loss_weight * sharpe_loss

        grads = tape.gradient(loss, train_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)
        return {
            'loss': float(loss.numpy()),
            'base_loss': float(base_loss.numpy()),
            'aux_loss': float(aux_loss.numpy()),
            'reg_loss': float(reg_loss.numpy()),
            'overconfidence_loss': float(overconfidence_loss.numpy()),
            'decision_loss': float(decision_loss.numpy()),
            'sharpe_loss': float(sharpe_loss.numpy()),
            'sigma_mean': float(tf.reduce_mean(pred_sigma).numpy()),
        }

    def train(self, epochs: int = 3, batch_size: int = 64) -> Dict[str, float]:
        if self.buffer.size() < batch_size:
            return {'status': 'insufficient_data', 'loss': 0.0}

        losses = []
        base_losses = []
        aux_losses = []
        reg_losses = []
        overconfidence_losses = []
        decision_losses = []
        sharpe_losses = []
        sigma_means = []
        target_means = []
        target_stds = []
        for _ in range(epochs):
            batch = self.buffer.sample_batch(batch_size)
            targets = np.asarray(batch['exposure_delta_target'], dtype=np.float32)
            if targets.size > 0:
                target_means.append(float(np.mean(targets)))
                target_stds.append(float(np.std(targets)))
            step_metrics = self.train_step(batch)
            losses.append(step_metrics.get('loss', 0.0))
            base_losses.append(step_metrics.get('base_loss', 0.0))
            aux_losses.append(step_metrics.get('aux_loss', 0.0))
            reg_losses.append(step_metrics.get('reg_loss', 0.0))
            overconfidence_losses.append(step_metrics.get('overconfidence_loss', 0.0))
            decision_losses.append(step_metrics.get('decision_loss', 0.0))
            sharpe_losses.append(step_metrics.get('sharpe_loss', 0.0))
            sigma_means.append(step_metrics.get('sigma_mean', 0.0))

        return {
            'loss': float(np.mean(losses)),
            'base_loss': float(np.mean(base_losses)) if base_losses else 0.0,
            'aux_loss': float(np.mean(aux_losses)) if aux_losses else 0.0,
            'reg_loss': float(np.mean(reg_losses)) if reg_losses else 0.0,
            'overconfidence_loss': float(np.mean(overconfidence_losses)) if overconfidence_losses else 0.0,
            'decision_loss': float(np.mean(decision_losses)) if decision_losses else 0.0,
            'sharpe_loss': float(np.mean(sharpe_losses)) if sharpe_losses else 0.0,
            'sigma_mean': float(np.mean(sigma_means)) if sigma_means else 0.0,
            'status': 'ok',
            'target_mean': float(np.mean(target_means)) if target_means else 0.0,
            'target_std': float(np.mean(target_stds)) if target_stds else 0.0,
        }

