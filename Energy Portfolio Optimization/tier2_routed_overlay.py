#!/usr/bin/env python3
"""
Tier-2 learned forecast layer - short-horizon conformal decision-focused routed expert overlay.

Tier-2 keeps Tier-1 observations intact, but learns an investor-only controller from:
- the Tier-1 investor observation,
- a compact forecast-context core, and
- a short-horizon forecast memory tensor carrying a real price expert bank.

The shared temporal encoder serves one role only:
- runtime short-horizon defer-and-route control for the investor's exposure overlay.

Architecture:
- observation stream
- compact core forecast-context stream
- temporal forecast-memory encoder
- short-horizon routed expert bank over ANN / LSTM / SVR / RF
- per-expert quality and conformal-risk channels carried inside the temporal forecast memory
- abstain-or-route head + normalized override head + intervention-confidence head
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D
from typing import Dict, Any, Optional, List
from collections import deque
import logging

from config import (
    FORECAST_BASE_WINDOW_SIZE_DEFAULT,
    FORECAST_BASE_INIT_BUDGET_DEFAULT,
    FORECAST_BASE_DIRECTION_WEIGHT_DEFAULT,
    TIER2_CV_FEATURE_DIM,
    TIER2_CV_MEMORY_CHANNELS,
    TIER2_CV_MEMORY_STEPS,
)
from tier2_expert_bank import (
    TIER2_ACTIVE_EXPERT_NAMES,
    TIER2_ROUTE_EXPERT_NAMES,
    build_routed_decision_training_bundle,
    compute_short_horizon_window_stats,
    decision_utility_numpy,
    extract_tier2_route_state_from_features,
    tier2_quality_gain,
    tier2_risk_gain,
    tier2_route_probability_gain,
)

logger = logging.getLogger(__name__)

# Canonical Tier-2 routed-overlay feature and weight names.
TIER2_ROUTED_OVERLAY_FEATURE_DIM = int(TIER2_CV_FEATURE_DIM)
TIER2_ROUTED_OVERLAY_WEIGHTS_FILENAME = "tier2_routed_overlay.h5"

# Backward-compatible aliases retained for older code/config paths.
CV_FORECAST_DIM = int(TIER2_ROUTED_OVERLAY_FEATURE_DIM)
CV_WEIGHTS_FILENAME = TIER2_ROUTED_OVERLAY_WEIGHTS_FILENAME


def _extract_base_exposure_from_features(forecast_features: np.ndarray) -> float:
    """
    Base Tier-1 investor exposure anchor encoded in the Tier-2 core features.

    The first core slot is the pre-overlay investor exposure. Tier-2 must learn
    whether and how much to deviate from that baseline, not re-learn the whole
    trading policy from scratch.
    """
    try:
        ff = np.asarray(forecast_features, dtype=np.float32).flatten()
        if ff.size > 0 and np.isfinite(float(ff[0])):
            return float(np.clip(float(ff[0]), -1.0, 1.0))
    except Exception:
        pass
    return 0.0


def _extract_executed_exposure_from_experience(experience: Dict[str, Any]) -> float:
    """
    Executed post-overlay investor exposure for one Tier-2 experience.

    This is the actual inventory that generated the realized short-horizon
    return paired with the experience.
    """
    try:
        executed = experience.get("executed_exposure", None)
        if executed is not None and np.isfinite(float(executed)):
            return float(np.clip(float(executed), -1.0, 1.0))
    except Exception:
        pass

    try:
        return _extract_base_exposure_from_features(experience.get("forecast_features", []))
    except Exception:
        pass

    return 0.0


def _extract_quality_context_from_features(forecast_features: np.ndarray) -> float:
    """
    Extract the latest short-horizon forecast-quality context in [0, 1].
    """
    try:
        return float(
            extract_tier2_route_state_from_features(forecast_features).get("quality_context", 0.5)
        )
    except Exception:
        return 0.5


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
# Tier-2 Learned Control Variate (model, adapter, trainer)
# ============================================================================


class Tier2RoutedOverlayModel(tf.keras.Model):
    """
    Tier-2 short-horizon residual-aware defer-and-route controller.

    Tier-2 does not learn a free-form exposure signal from scratch. Instead it:
    - encodes Tier-1 investor state plus the compact forecast block,
    - constructs a small routed expert bank from the latest short-horizon
      forecast-memory row, and
    - learns when to abstain, which expert to trust, and how large the bounded
      investor override should be.
    """

    def __init__(
        self,
        obs_dim: int,
        forecast_dim: int = CV_FORECAST_DIM,
        memory_steps: int = TIER2_CV_MEMORY_STEPS,
        memory_channels: int = TIER2_CV_MEMORY_CHANNELS,
        seed: int = 42,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)
        self.input_dim = self.obs_dim + self.forecast_dim
        self.memory_steps = max(1, int(memory_steps))
        self.memory_channels = max(1, int(memory_channels))
        self.expected_memory_flat_dim = int(self.memory_steps * self.memory_channels)
        self.memory_flat_dim = int(min(self.expected_memory_flat_dim, self.forecast_dim))
        self.core_dim = int(max(0, self.forecast_dim - self.memory_flat_dim))
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        self.obs_d1 = Dense(64, activation="relu", name="cv_obs_d1")
        self.obs_do1 = Dropout(0.08, name="cv_obs_do1")
        self.obs_d2 = Dense(32, activation="relu", name="cv_obs_d2")
        self.obs_proj = Dense(16, activation="relu", name="cv_obs_proj")

        self.core_d1 = Dense(24, activation="relu", name="cv_core_d1")
        self.core_d2 = Dense(16, activation="relu", name="cv_core_d2")

        self.temporal_conv1 = Conv1D(16, kernel_size=2, padding="causal", activation="relu", name="cv_temporal_conv1")
        self.temporal_conv2 = Conv1D(16, kernel_size=1, padding="same", activation="relu", name="cv_temporal_conv2")
        self.temporal_pool = GlobalAveragePooling1D(name="cv_temporal_pool")
        self.temporal_proj = Dense(16, activation="relu", name="cv_temporal_proj")

        self.route_d1 = Dense(32, activation="relu", name="cv_route_d1")
        self.route_do1 = Dropout(0.08, name="cv_route_do1")
        self.route_out = Dense(len(TIER2_ROUTE_EXPERT_NAMES), name="cv_route_out")

        self.magnitude_d1 = Dense(24, activation="relu", name="cv_magnitude_d1")
        self.magnitude_do1 = Dropout(0.05, name="cv_magnitude_do1")
        self.magnitude_out = Dense(1, activation="sigmoid", name="cv_magnitude_out")

        self.confidence_d1 = Dense(24, activation="relu", name="cv_confidence_d1")
        self.confidence_do1 = Dropout(0.05, name="cv_confidence_do1")
        self.confidence_d2 = Dense(12, activation="relu", name="cv_confidence_d2")
        self.confidence_out = Dense(1, activation="sigmoid", name="cv_confidence_out")

        logger.info(
            "Tier2RoutedOverlayModel: obs_dim=%s, forecast_dim=%s, core_dim=%s, memory=%sx%s",
            obs_dim,
            forecast_dim,
            self.core_dim,
            self.memory_steps,
            self.memory_channels,
        )

    def _split_input(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        obs = x[:, : self.obs_dim]
        forecast = x[:, self.obs_dim : self.obs_dim + self.forecast_dim]
        if self.forecast_dim <= 0:
            batch_size = tf.shape(obs)[0]
            core = tf.zeros((batch_size, 0), dtype=obs.dtype)
            memory = tf.zeros((batch_size, self.memory_steps, self.memory_channels), dtype=obs.dtype)
            return obs, core, memory

        core = forecast[:, : self.core_dim] if self.core_dim > 0 else tf.zeros((tf.shape(obs)[0], 0), dtype=obs.dtype)
        memory_flat = forecast[:, self.core_dim :]
        flat_width = int(memory_flat.shape[-1]) if memory_flat.shape[-1] is not None else self.memory_flat_dim
        if flat_width < self.expected_memory_flat_dim:
            pad_width = self.expected_memory_flat_dim - flat_width
            memory_flat = tf.pad(memory_flat, [[0, 0], [0, pad_width]])
        elif flat_width > self.expected_memory_flat_dim:
            memory_flat = memory_flat[:, : self.expected_memory_flat_dim]
        memory = tf.reshape(memory_flat, (-1, self.memory_steps, self.memory_channels))
        return obs, core, memory

    @staticmethod
    def _build_route_expert_signals(last_step: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Decode the real short-horizon expert bank from the latest memory row."""
        n = len(TIER2_ACTIVE_EXPERT_NAMES)
        has_active_memory = tf.cast(
            tf.reduce_max(tf.abs(last_step), axis=1, keepdims=True) > 1e-8,
            last_step.dtype,
        )
        neutral_quality = tf.fill(tf.shape(last_step[:, 0:1]), tf.cast(0.5, last_step.dtype))
        expert_signals = tf.clip_by_value(last_step[:, 0:n], -1.0, 1.0)
        route_quality = has_active_memory * tf.clip_by_value(last_step[:, n:2*n], 0.0, 1.0) + (1.0 - has_active_memory) * tf.concat([neutral_quality] * n, axis=-1)
        route_risk = has_active_memory * tf.clip_by_value(last_step[:, 2*n:3*n], 0.0, 1.0) + (1.0 - has_active_memory) * tf.concat([neutral_quality] * n, axis=-1)
        consensus_signal = tf.clip_by_value(last_step[:, 3*n:3*n+1], -1.0, 1.0)
        disagreement_signal = tf.clip_by_value(last_step[:, 3*n+1:3*n+2], 0.0, 1.0)
        imbalance_signal = tf.clip_by_value(last_step[:, 3*n+2:3*n+3], -1.0, 1.0)
        quality_context = tf.clip_by_value(
            tf.reduce_mean(route_quality, axis=-1, keepdims=True)
            * (1.0 - 0.25 * tf.reduce_mean(route_risk, axis=-1, keepdims=True))
            * (1.0 - 0.15 * disagreement_signal),
            0.0,
            1.0,
        )
        return expert_signals, route_quality, quality_context, route_risk, consensus_signal, disagreement_signal, imbalance_signal

    def forward_components(self, x: tf.Tensor, training: Optional[bool] = None) -> Dict[str, tf.Tensor]:
        obs, core, memory = self._split_input(x)

        obs_h = self.obs_d1(obs)
        obs_h = self.obs_do1(obs_h, training=training)
        obs_h = self.obs_d2(obs_h)
        obs_proj = self.obs_proj(obs_h)

        if self.core_dim > 0:
            core_h = self.core_d1(core)
            core_h = self.core_d2(core_h)
        else:
            core_h = tf.zeros((tf.shape(obs)[0], 16), dtype=obs.dtype)

        temporal_h = self.temporal_conv1(memory)
        temporal_h = self.temporal_conv2(temporal_h)
        temporal_h = self.temporal_pool(temporal_h)
        temporal_h = self.temporal_proj(temporal_h)

        last_step = memory[:, -1, :]
        expert_signals, route_quality, quality_context, route_risk, consensus_signal, disagreement_signal, imbalance_signal = self._build_route_expert_signals(last_step)
        route_context = tf.concat(
            [
                obs_h,
                obs_proj,
                core_h,
                temporal_h,
                route_quality,
                route_risk,
                expert_signals,
                consensus_signal,
                disagreement_signal,
                imbalance_signal,
            ],
            axis=-1,
        )
        route_logits = self.route_out(self.route_do1(self.route_d1(route_context), training=training))
        route_probs = tf.nn.softmax(route_logits, axis=-1)

        abstain_prob = route_probs[:, 0:1]
        active_route_probs = route_probs[:, 1:]
        selected_expert_idx = tf.argmax(active_route_probs, axis=-1, output_type=tf.int32)
        selected_expert_probability = tf.reduce_max(active_route_probs, axis=-1, keepdims=True)
        selected_expert_signal = tf.expand_dims(
            tf.gather(expert_signals, selected_expert_idx, axis=1, batch_dims=1),
            axis=-1,
        )
        selected_route_quality = tf.expand_dims(
            tf.gather(route_quality, selected_expert_idx, axis=1, batch_dims=1),
            axis=-1,
        )
        selected_route_risk = tf.expand_dims(
            tf.gather(route_risk, selected_expert_idx, axis=1, batch_dims=1),
            axis=-1,
        )
        route_selected_mask = tf.cast(selected_expert_probability > abstain_prob, expert_signals.dtype)
        routed_signal = route_selected_mask * selected_expert_signal

        magnitude = self.magnitude_out(
            self.magnitude_do1(
                self.magnitude_d1(
                    tf.concat(
                        [
                            route_context,
                            routed_signal,
                            selected_expert_probability,
                            selected_route_quality,
                            selected_route_risk,
                        ],
                        axis=-1,
                    )
                ),
                training=training,
            )
        )
        delta = tf.clip_by_value(magnitude * routed_signal, -1.0, 1.0)

        confidence_h = self.confidence_d1(
            tf.concat([route_context, delta, abstain_prob, quality_context, selected_route_risk], axis=-1)
        )
        confidence_h = self.confidence_do1(confidence_h, training=training)
        confidence_h = self.confidence_d2(confidence_h)
        confidence = self.confidence_out(confidence_h)
        route_entropy = -tf.reduce_sum(route_probs * tf.math.log(route_probs + 1e-8), axis=-1, keepdims=True)

        return {
            "delta": delta,
            "confidence": confidence,
            "route_logits": route_logits,
            "route_probs": route_probs,
            "abstain_prob": abstain_prob,
            "expert_signals": expert_signals,
            "route_quality": route_quality,
            "route_risk": route_risk,
            "quality_context": quality_context,
            "magnitude": magnitude,
            "selected_expert_idx": selected_expert_idx,
            "selected_expert_probability": selected_expert_probability,
            "selected_expert_signal": selected_expert_signal,
            "selected_route_quality": selected_route_quality,
            "selected_route_risk": selected_route_risk,
            "route_selected_mask": route_selected_mask,
            "route_entropy": route_entropy,
            "consensus_signal": consensus_signal,
            "disagreement_signal": disagreement_signal,
            "imbalance_signal": imbalance_signal,
        }

    def call(self, x, training=None):
        return self.forward_components(x, training=training)["delta"]


class Tier2RoutedOverlayAdapter:
    """
    Adapter for metacontroller to call Tier-2 routed-overlay inference.
    Supports load_weights with compatibility check (model must be built first).
    """

    def __init__(
        self,
        obs_dim: int,
        forecast_dim: int = CV_FORECAST_DIM,
        memory_steps: int = TIER2_CV_MEMORY_STEPS,
        memory_channels: int = TIER2_CV_MEMORY_CHANNELS,
        seed: int = 42,
    ):
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)
        self.model = Tier2RoutedOverlayModel(
            obs_dim,
            forecast_dim,
            memory_steps=memory_steps,
            memory_channels=memory_channels,
            seed=seed,
        )

    def _ensure_built(self):
        """Ensure model is built (required for load_weights)."""
        if not self.model.built:
            dummy = np.zeros((1, self.obs_dim + self.forecast_dim), dtype=np.float32)
            _ = self.model(tf.convert_to_tensor(dummy), training=False)

    def _prepare_input(self, obs: np.ndarray, forecast_features: np.ndarray) -> np.ndarray:
        """Build a single concatenated input row with the configured obs/forecast dimensions."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if forecast_features.ndim == 1:
            forecast_features = forecast_features.reshape(1, -1)
        if forecast_features.shape[1] != self.forecast_dim:
            f = np.zeros((forecast_features.shape[0], self.forecast_dim), dtype=np.float32)
            n = min(forecast_features.shape[1], self.forecast_dim)
            f[:, :n] = forecast_features[:, :n]
            forecast_features = f
        x = np.concatenate([obs.astype(np.float32), forecast_features.astype(np.float32)], axis=1)
        if x.shape[1] != self.obs_dim + self.forecast_dim:
            x = np.concatenate([
                obs[:, :self.obs_dim].astype(np.float32),
                forecast_features.astype(np.float32),
            ], axis=1)
        return x

    def _run_forward_components(
        self,
        obs: np.ndarray,
        forecast_features: np.ndarray,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Single forward_components call with shared input preparation."""
        x = self._prepare_input(obs, forecast_features)
        inp = tf.convert_to_tensor(x, dtype=tf.float32)
        return self.model.forward_components(inp, training=training)

    @staticmethod
    def _components_to_dict(out: Dict[str, tf.Tensor]) -> Dict[str, float]:
        """Convert routed forward_components() outputs into plain diagnostics."""
        route_probs = np.asarray(out["route_probs"].numpy()[0], dtype=np.float32).flatten()
        active_route_probs = route_probs[1:] if route_probs.size > 1 else np.zeros(len(TIER2_ACTIVE_EXPERT_NAMES), dtype=np.float32)
        expert_signals = np.asarray(out["expert_signals"].numpy()[0], dtype=np.float32).flatten()
        route_quality = np.asarray(out["route_quality"].numpy()[0], dtype=np.float32).flatten()
        route_risk = np.asarray(out["route_risk"].numpy()[0], dtype=np.float32).flatten() if "route_risk" in out else np.full(len(TIER2_ACTIVE_EXPERT_NAMES), 0.5, dtype=np.float32)
        delta_prediction = float(out["delta"].numpy().flatten()[0])
        confidence_prediction = float(out["confidence"].numpy().flatten()[0])
        magnitude_prediction = float(out["magnitude"].numpy().flatten()[0])
        abstain_prob = float(out["abstain_prob"].numpy().flatten()[0]) if "abstain_prob" in out else float(route_probs[0] if route_probs.size > 0 else 1.0)
        route_entropy = float(out["route_entropy"].numpy().flatten()[0]) if "route_entropy" in out else 0.0

        selected_local_idx = int(np.argmax(active_route_probs)) if active_route_probs.size > 0 else -1
        selected_route_idx = selected_local_idx + 1 if selected_local_idx >= 0 else 0
        selected_route_prob = float(active_route_probs[selected_local_idx]) if selected_local_idx >= 0 else 0.0
        selected_expert_signal = float(expert_signals[selected_local_idx]) if selected_local_idx >= 0 and selected_local_idx < expert_signals.size else 0.0
        selected_route_quality = float(route_quality[selected_local_idx]) if selected_local_idx >= 0 and selected_local_idx < route_quality.size else 0.5
        if abstain_prob >= selected_route_prob:
            selected_route_idx = 0
            selected_route_prob = float(np.clip(abstain_prob, 0.0, 1.0))
            selected_expert_signal = 0.0
            selected_route_quality = 0.5
            selected_route_risk = 1.0
        else:
            selected_route_risk = float(route_risk[selected_local_idx]) if selected_local_idx >= 0 and selected_local_idx < route_risk.size else 0.5
        route_quality_mean = float(np.mean(route_quality)) if route_quality.size > 0 else 0.5
        route_risk_mean = float(np.mean(route_risk)) if route_risk.size > 0 else 0.5
        conformal_residual_risk = float(selected_route_risk)
        route_active_scale = float(np.clip(1.0 - abstain_prob, 0.0, 1.0))
        effective_route_active_prob = float(np.clip(selected_route_prob if selected_route_idx > 0 else 0.0, 0.0, 1.0))

        out_dict = {
            "delta_prediction": delta_prediction,
            "delta_signal": float(np.clip(delta_prediction, -1.0, 1.0)),
            "controller_confidence": float(np.clip(confidence_prediction, 0.0, 1.0)),
            "direction_signal": float(np.clip(selected_expert_signal, -1.0, 1.0)),
            "route_entropy": float(route_entropy),
            "route_abstain_prob": float(np.clip(abstain_prob, 0.0, 1.0)),
            "route_active_prob": float(effective_route_active_prob),
            "route_non_abstain_prob": float(np.clip(1.0 - abstain_prob, 0.0, 1.0)),
            "route_active_scale": float(route_active_scale),
            "route_selected_expert_idx": float(selected_route_idx),
            "route_selected_probability": float(np.clip(selected_route_prob, 0.0, 1.0)),
            "route_selected_signal": float(np.clip(selected_expert_signal, -1.0, 1.0)),
            "route_selected_quality": float(np.clip(selected_route_quality, 0.0, 1.0)),
            "route_quality_mean": float(np.clip(route_quality_mean, 0.0, 1.0)),
            "route_risk_mean": float(np.clip(route_risk_mean, 0.0, 1.0)),
            "route_selected_risk": float(np.clip(selected_route_risk, 0.0, 1.0)),
            "conformal_residual_risk": float(np.clip(conformal_residual_risk, 0.0, 1.0)),
            "magnitude_prediction": float(np.clip(magnitude_prediction, 0.0, 1.0)),
            "consensus_signal": float(np.clip(out.get("consensus_signal", tf.zeros((1, 1))).numpy().flatten()[0], -1.0, 1.0)),
            "disagreement_signal": float(np.clip(out.get("disagreement_signal", tf.zeros((1, 1))).numpy().flatten()[0], 0.0, 1.0)),
            "imbalance_signal": float(np.clip(out.get("imbalance_signal", tf.zeros((1, 1))).numpy().flatten()[0], -1.0, 1.0)),
        }
        for idx, name in enumerate(TIER2_ACTIVE_EXPERT_NAMES):
            out_dict[f"{name}_weight"] = float(active_route_probs[idx]) if active_route_probs.size > idx else 0.0
            out_dict[f"{name}_signal"] = float(expert_signals[idx]) if expert_signals.size > idx else 0.0
            out_dict[f"{name}_quality"] = float(route_quality[idx]) if route_quality.size > idx else 0.5
            out_dict[f"{name}_risk"] = float(route_risk[idx]) if route_risk.size > idx else 0.5
        return out_dict

    def load_weights(self, path: str) -> bool:
        """
        Load weights from path. Returns True on success.
        Builds model if needed (Keras requires variables before loading HDF5).
        """
        if not path or not os.path.exists(path):
            return False
        try:
            self._ensure_built()
            self.model.load_weights(path)
            logger.info(f"[TIER2] Loaded routed overlay weights from {path}")
            return True
        except Exception as e:
            logger.warning(f"[TIER2] Could not load routed overlay weights from {path}: {e}")
            return False

    def save_weights(self, path: str) -> bool:
        """Save weights to path. Returns True on success."""
        try:
            self._ensure_built()
            self.model.save_weights(path)
            return True
        except Exception as e:
            logger.warning(f"[TIER2] Could not save routed overlay weights to {path}: {e}")
            return False

    def predict(self, obs: np.ndarray, forecast_features: np.ndarray, training: bool = False) -> float:
        """Backward-compatible scalar prediction alias. Returns delta prediction."""
        out = self.model(self._prepare_input(obs, forecast_features), training=training).numpy().flatten()
        return float(out[0])

    def predict_delta(self, obs: np.ndarray, forecast_features: np.ndarray, training: bool = False) -> float:
        """Return scalar short-horizon delta-exposure prediction."""
        out = self._run_forward_components(obs, forecast_features, training=training)["delta"].numpy().flatten()
        return float(out[0])

    def predict_alpha(self, obs: np.ndarray, forecast_features: np.ndarray, training: bool = False) -> float:
        """Backward-compatible alias for the learned delta-exposure head."""
        return self.predict_delta(obs, forecast_features, training=training)

    def predict_confidence(self, obs: np.ndarray, forecast_features: np.ndarray, training: bool = False) -> float:
        """Return learned intervention confidence in [0, 1]."""
        out = self._run_forward_components(obs, forecast_features, training=training)["confidence"].numpy().flatten()
        return float(out[0])

    def predict_components(self, obs: np.ndarray, forecast_features: np.ndarray) -> Dict[str, float]:
        """
        Return Tier-2 architectural diagnostics from the short-horizon defer-and-route controller.
        """
        return self._components_to_dict(self._run_forward_components(obs, forecast_features, training=False))

    def predict_runtime_bundle(
        self,
        obs: np.ndarray,
        forecast_features: np.ndarray,
        samples: int = 7,
    ) -> Dict[str, Any]:
        """
        Fused runtime inference for Tier-2.

        Returns deterministic structural diagnostics plus an MC-dropout delta
        distribution without repeating separate forward calls.
        """
        deterministic_out = self._run_forward_components(obs, forecast_features, training=False)
        components = self._components_to_dict(deterministic_out)
        confidence_mean = float(components.get("controller_confidence", 0.5))

        sample_count = max(1, int(samples))
        if sample_count == 1:
            delta_mean = float(components["delta_prediction"])
            delta_std = 0.0
        else:
            delta_samples = []
            confidence_samples = []
            for _ in range(sample_count):
                sampled_out = self._run_forward_components(obs, forecast_features, training=True)
                delta_samples.append(float(sampled_out["delta"].numpy().flatten()[0]))
                confidence_samples.append(float(sampled_out["confidence"].numpy().flatten()[0]))
            delta_np = np.asarray(delta_samples, dtype=np.float32)
            delta_mean = float(np.mean(delta_np))
            delta_std = float(np.std(delta_np))
            confidence_mean = float(np.mean(np.asarray(confidence_samples, dtype=np.float32)))

        return {
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
            "confidence_mean": float(np.clip(confidence_mean, 0.0, 1.0)),
            "components": components,
        }

    def predict_distribution(
        self,
        obs: np.ndarray,
        forecast_features: np.ndarray,
        samples: int = 7,
    ) -> tuple[float, float]:
        """
        Monte-Carlo dropout estimate for the delta head.
        """
        return self.predict_delta_distribution(obs, forecast_features, samples=samples)

    def predict_delta_distribution(
        self,
        obs: np.ndarray,
        forecast_features: np.ndarray,
        samples: int = 7,
    ) -> tuple[float, float]:
        """
        Monte-Carlo dropout estimate for the direct delta-exposure head.
        """
        bundle = self.predict_runtime_bundle(obs, forecast_features, samples=samples)
        return float(bundle["delta_mean"]), float(bundle["delta_std"])

    def predict_alpha_distribution(
        self,
        obs: np.ndarray,
        forecast_features: np.ndarray,
        samples: int = 7,
    ) -> tuple[float, float]:
        """Backward-compatible alias for the learned delta-exposure head."""
        return self.predict_delta_distribution(obs, forecast_features, samples=samples)


class Tier2ExperienceBuffer:
    """
    Stores Tier-2 investor experiences during rollout.
    Supports continuous learning:
    - Rolling buffer: keeps last max_size experiences
    - Replay buffer: stores decision-focused training bundles for sampling across rollouts
    """

    def __init__(self, max_size: int = 50_000, replay_size: int = 20_000, seed: Optional[int] = None):
        self.buffer: List[Dict[str, Any]] = []  # Current rollout (no returns yet)
        self.replay: List[Dict[str, Any]] = []  # Persistent (obs, ff, V, return) for continuous learning
        self.max_size = max_size
        self.replay_size = replay_size
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        forecast_features: np.ndarray,
        value: float = 0.0,
        realized_return: float = 0.0,
        executed_exposure: Optional[float] = None,
        rollout_pos: Optional[int] = None,
        env_step: Optional[int] = None,
    ):
        realized_return = float(realized_return) if np.isfinite(realized_return) else 0.0
        exp_dict: Dict[str, Any] = {
            'obs': np.asarray(obs, dtype=np.float32).flatten().copy(),
            'forecast_features': np.asarray(forecast_features, dtype=np.float32).flatten().copy(),
            'value': float(value),
            'realized_return': realized_return,
        }
        if executed_exposure is not None and np.isfinite(float(executed_exposure)):
            exp_dict['executed_exposure'] = float(np.clip(float(executed_exposure), -1.0, 1.0))
        if rollout_pos is not None:
            exp_dict['rollout_pos'] = int(rollout_pos)
        if env_step is not None:
            exp_dict['env_step'] = int(env_step)
        self.buffer.append(exp_dict)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def promote_with_returns(
        self,
        returns: Optional[np.ndarray] = None,
        experiences: Optional[List[Dict[str, Any]]] = None,
        decision_horizon: int = 4,
        decision_scale: float = 0.05,
        decision_downside_weight: float = 0.75,
        decision_vol_weight: float = 0.25,
        decision_stability_penalty: float = 0.12,
        magnitude_grid_points: int = 11,
        delta_limit: float = 0.20,
        conformal_risk_scale: float = 1.0,
        runtime_conformal_margin: float = 0.05,
        runtime_gain: float = 1.0,
    ) -> int:
        """Promote current rollout to replay with decision-focused defer-and-route targets."""
        source = list(experiences) if experiences is not None else list(self.buffer)
        n = len(source)
        if n < 8:
            return 0
        realized_returns = np.asarray(
            [source[i].get('realized_return', 0.0) for i in range(n)],
            dtype=np.float32,
        )
        base_exposures = np.asarray(
            [_extract_base_exposure_from_features(e.get("forecast_features", [])) for e in source[:n]],
            dtype=np.float32,
        )
        executed_exposures = np.asarray(
            [_extract_executed_exposure_from_experience(e) for e in source[:n]],
            dtype=np.float32,
        )
        ff_stack = np.stack([np.asarray(e.get("forecast_features", []), dtype=np.float32).flatten() for e in source[:n]]).astype(np.float32)
        training_bundle = build_routed_decision_training_bundle(
            realized_returns,
            base_exposures,
            executed_exposures,
            ff_stack,
            decision_horizon,
            delta_limit=delta_limit,
            utility_scale=decision_scale,
            downside_weight=decision_downside_weight,
            vol_weight=decision_vol_weight,
            stability_penalty=decision_stability_penalty,
            magnitude_grid_points=magnitude_grid_points,
            conformal_risk_scale=conformal_risk_scale,
            runtime_conformal_margin=runtime_conformal_margin,
            runtime_gain=runtime_gain,
        )
        for i, e in enumerate(source[:n]):
            self.replay.append({
                **e,
                'delta_target': float(training_bundle['delta_targets'][i]) if i < training_bundle['delta_targets'].size else 0.0,
                'confidence_target': float(training_bundle['confidence_targets'][i]) if i < training_bundle['confidence_targets'].size else 0.0,
                'sample_weight': float(training_bundle['sample_weights'][i]) if i < training_bundle['sample_weights'].size else 1.0,
                'base_exposure': float(training_bundle['base_exposures'][i]) if i < training_bundle['base_exposures'].size else 0.0,
                'gross_return': float(training_bundle['gross_returns'][i]) if i < training_bundle['gross_returns'].size else 0.0,
                'downside_return': float(training_bundle['downside_returns'][i]) if i < training_bundle['downside_returns'].size else 0.0,
                'vol_return': float(training_bundle['vol_returns'][i]) if i < training_bundle['vol_returns'].size else 0.0,
                'oracle_improvement': float(training_bundle['oracle_improvement'][i]) if i < training_bundle['oracle_improvement'].size else 0.0,
                'route_target': int(training_bundle['route_targets'][i]) if i < training_bundle['route_targets'].size else 0,
                'route_margin': float(training_bundle['route_margins'][i]) if i < training_bundle['route_margins'].size else 0.0,
                'expert_signal_target': float(training_bundle['expert_signal_targets'][i]) if i < training_bundle['expert_signal_targets'].size else 0.0,
                'certified_improvement': float(training_bundle['certified_improvement'][i]) if i < training_bundle['certified_improvement'].size else 0.0,
            })
        while len(self.replay) > self.replay_size:
            self.replay.pop(0)
        return n

    def get_current_rollout(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def sample_replay(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Sample batch from replay for continuous learning. Returns None if insufficient."""
        if len(self.replay) < batch_size:
            return None
        idx = self._rng.choice(len(self.replay), size=batch_size, replace=False)
        batch = [self.replay[i] for i in idx]
        return {
            'obs': np.stack([b['obs'] for b in batch]),
            'forecast_features': np.stack([b['forecast_features'] for b in batch]),
            'delta_targets': np.array([b.get('delta_target', 0.0) for b in batch], dtype=np.float32),
            'confidence_targets': np.array([b.get('confidence_target', 0.0) for b in batch], dtype=np.float32),
            'route_targets': np.array([b.get('route_target', 0) for b in batch], dtype=np.int32),
            'route_margins': np.array([b.get('route_margin', 0.0) for b in batch], dtype=np.float32),
            'expert_signal_targets': np.array([b.get('expert_signal_target', 0.0) for b in batch], dtype=np.float32),
            'sample_weights': np.array([b.get('sample_weight', 1.0) for b in batch], dtype=np.float32),
            'base_exposures': np.array([b.get('base_exposure', 0.0) for b in batch], dtype=np.float32),
            'gross_returns': np.array([b.get('gross_return', 0.0) for b in batch], dtype=np.float32),
            'downside_returns': np.array([b.get('downside_return', 0.0) for b in batch], dtype=np.float32),
            'vol_returns': np.array([b.get('vol_return', 0.0) for b in batch], dtype=np.float32),
            'oracle_improvement': np.array([b.get('oracle_improvement', 0.0) for b in batch], dtype=np.float32),
            'certified_improvement': np.array([b.get('certified_improvement', 0.0) for b in batch], dtype=np.float32),
        }

    def clear_rollout(self):
        """Clear only current rollout (replay kept for continuous learning)."""
        self.buffer.clear()

    def clear(self):
        self.buffer.clear()
        self.replay.clear()

    def size(self) -> int:
        return len(self.buffer)

    def replay_size_count(self) -> int:
        return len(self.replay)


class Tier2RoutedOverlayTrainer:
    """
    Trains the Tier-2 defer-and-route expert controller.

    The objective is baseline-relative and decision-focused:
    - route head chooses abstain vs short-horizon expert family
    - override head predicts the normalized investor-only deviation from Tier-1
    - confidence head predicts whether intervening is worth trusting
    """

    def __init__(
        self,
        model: Tier2RoutedOverlayModel,
        obs_dim: int,
        forecast_dim: int = CV_FORECAST_DIM,
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-4,
        seed: int = 42,
        replay_batch_size: int = 64,
        train_epochs: int = 1,
        route_loss_weight: float = 0.35,
        override_loss_weight: float = 0.25,
        decision_weight: float = 0.35,
        confidence_weight: float = 0.15,
        decision_horizon: int = 4,
        decision_scale: float = 0.05,
        decision_downside_weight: float = 0.75,
        decision_vol_weight: float = 0.25,
        decision_stability_penalty: float = 0.12,
        magnitude_grid_points: int = 11,
        delta_limit: float = 0.20,
        expert_signal_weight: float = 0.10,
        conformal_risk_scale: float = 1.0,
        runtime_gain: float = 1.0,
        runtime_conformal_margin: float = 0.05,
    ):
        self.model = model
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
        self.l2_reg = float(l2_reg)
        self.replay_batch_size = int(replay_batch_size)
        self.train_epochs = max(1, int(train_epochs))
        self.route_loss_weight = float(max(route_loss_weight, 0.0))
        self.override_loss_weight = float(max(override_loss_weight, 0.0))
        self.decision_weight = float(max(decision_weight, 0.0))
        self.confidence_weight = float(max(confidence_weight, 0.0))
        self.decision_horizon = max(1, int(decision_horizon))
        self.decision_scale = float(max(decision_scale, 1e-6))
        self.decision_downside_weight = float(max(decision_downside_weight, 0.0))
        self.decision_vol_weight = float(max(decision_vol_weight, 0.0))
        self.decision_stability_penalty = float(max(decision_stability_penalty, 0.0))
        self.magnitude_grid_points = max(3, int(magnitude_grid_points))
        self.delta_limit = float(max(delta_limit, 1e-6))
        self.expert_signal_weight = float(max(expert_signal_weight, 0.0))
        self.conformal_risk_scale = float(max(conformal_risk_scale, 0.0))
        self.runtime_gain = float(np.clip(runtime_gain, 0.0, 1.0))
        self.runtime_conformal_margin = float(max(runtime_conformal_margin, 0.0))
        logger.info(
            "Tier2RoutedOverlayTrainer: lr=%s, l2=%s, replay_batch=%s, epochs=%s, route_w=%s, override_w=%s, decision_w=%s, conf_w=%s, horizon=%s, scale=%s, down=%s, vol=%s, stability=%s, grid=%s, delta_limit=%s, conformal_scale=%s, runtime_gain=%s, runtime_conformal_margin=%s",
            learning_rate,
            l2_reg,
            replay_batch_size,
            self.train_epochs,
            self.route_loss_weight,
            self.override_loss_weight,
            self.decision_weight,
            self.confidence_weight,
            self.decision_horizon,
            self.decision_scale,
            self.decision_downside_weight,
            self.decision_vol_weight,
            self.decision_stability_penalty,
            self.magnitude_grid_points,
            self.delta_limit,
            self.conformal_risk_scale,
            self.runtime_gain,
            self.runtime_conformal_margin,
        )

    @staticmethod
    def _decision_utility_tf(
        exposure_t: tf.Tensor,
        base_exposure_t: tf.Tensor,
        gross_t: tf.Tensor,
        downside_t: tf.Tensor,
        vol_t: tf.Tensor,
        downside_weight: float,
        vol_weight: float,
        stability_penalty: float,
    ) -> tf.Tensor:
        return (
            exposure_t * gross_t
            - float(downside_weight) * tf.abs(exposure_t) * downside_t
            - float(vol_weight) * tf.abs(exposure_t) * vol_t
            - float(stability_penalty) * tf.square(exposure_t - base_exposure_t)
        )

    def _train_step(
        self,
        x: np.ndarray,
        route_targets: np.ndarray,
        delta_targets: np.ndarray,
        confidence_targets: np.ndarray,
        base_exposures: np.ndarray,
        gross_returns: np.ndarray,
        downside_returns: np.ndarray,
        vol_returns: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        route_margins: Optional[np.ndarray] = None,
        expert_signal_targets: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Single routed decision-focused training step."""
        route_targets_t = tf.convert_to_tensor(np.asarray(route_targets, dtype=np.int32).reshape(-1), dtype=tf.int32)
        delta_targets_t = tf.convert_to_tensor(delta_targets.reshape(-1, 1), dtype=tf.float32)
        confidence_targets_t = tf.convert_to_tensor(confidence_targets.reshape(-1, 1), dtype=tf.float32)
        base_exposures_t = tf.convert_to_tensor(base_exposures.reshape(-1, 1), dtype=tf.float32)
        gross_returns_t = tf.convert_to_tensor(gross_returns.reshape(-1, 1), dtype=tf.float32)
        downside_returns_t = tf.convert_to_tensor(downside_returns.reshape(-1, 1), dtype=tf.float32)
        vol_returns_t = tf.convert_to_tensor(vol_returns.reshape(-1, 1), dtype=tf.float32)
        if route_margins is None:
            route_margins_np = np.zeros(delta_targets.shape[0], dtype=np.float32)
        else:
            route_margins_np = np.asarray(route_margins, dtype=np.float32).reshape(-1)
            if route_margins_np.size != delta_targets.shape[0]:
                route_margins_np = np.zeros(delta_targets.shape[0], dtype=np.float32)
        route_margins_t = tf.convert_to_tensor(route_margins_np.reshape(-1, 1), dtype=tf.float32)
        if expert_signal_targets is None:
            expert_signal_targets_np = np.sign(np.asarray(delta_targets, dtype=np.float32).reshape(-1))
        else:
            expert_signal_targets_np = np.asarray(expert_signal_targets, dtype=np.float32).reshape(-1)
            if expert_signal_targets_np.size != delta_targets.shape[0]:
                expert_signal_targets_np = np.sign(np.asarray(delta_targets, dtype=np.float32).reshape(-1))
        expert_signal_targets_t = tf.convert_to_tensor(expert_signal_targets_np.reshape(-1, 1), dtype=tf.float32)
        x_t = tf.convert_to_tensor(x, dtype=tf.float32)
        if sample_weights is None:
            weights_np = np.ones(delta_targets.shape[0], dtype=np.float32)
        else:
            weights_np = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
            if weights_np.size != delta_targets.shape[0]:
                weights_np = np.ones(delta_targets.shape[0], dtype=np.float32)
        weights_t = tf.convert_to_tensor(weights_np.reshape(-1, 1), dtype=tf.float32)
        weights_t = weights_t / tf.maximum(tf.reduce_mean(weights_t), 1e-6)
        with tf.GradientTape() as tape:
            out = self.model.forward_components(x_t, training=True)
            delta_pred = out["delta"]
            confidence_pred = out["confidence"]
            route_logits = out["route_logits"]
            route_probs = out["route_probs"]
            abstain_prob = out["abstain_prob"]
            selected_route_prob = out["selected_expert_probability"]
            selected_route_quality = out["selected_route_quality"]
            selected_route_risk = out["selected_route_risk"]
            route_selected_mask = tf.cast(out["route_selected_mask"], tf.float32)
            routed_signal_pred = tf.cast(out["route_selected_mask"], tf.float32) * out["selected_expert_signal"]
            risk_gain = tf.clip_by_value(
                1.0 - 0.35 * tf.maximum(
                    0.0,
                    self.conformal_risk_scale * selected_route_risk - self.runtime_conformal_margin,
                ),
                0.0,
                1.0,
            )
            confidence_gain = tf.clip_by_value(0.50 + 0.50 * confidence_pred, 0.50, 1.0)
            route_gain = route_selected_mask * tf.clip_by_value(0.40 + 0.60 * selected_route_prob, 0.0, 1.0)
            quality_gain = tf.clip_by_value(0.60 + 0.40 * selected_route_quality, 0.60, 1.0)
            effective_overlay_gain = tf.clip_by_value(
                self.runtime_gain * route_gain * quality_gain * confidence_gain * risk_gain,
                0.0,
                1.0,
            )
            predicted_exposure = tf.clip_by_value(
                base_exposures_t + self.delta_limit * effective_overlay_gain * delta_pred,
                -1.0,
                1.0,
            )
            baseline_utility = self._decision_utility_tf(
                base_exposures_t,
                base_exposures_t,
                gross_returns_t,
                downside_returns_t,
                vol_returns_t,
                self.decision_downside_weight,
                self.decision_vol_weight,
                self.decision_stability_penalty,
            )
            predicted_utility = self._decision_utility_tf(
                predicted_exposure,
                base_exposures_t,
                gross_returns_t,
                downside_returns_t,
                vol_returns_t,
                self.decision_downside_weight,
                self.decision_vol_weight,
                self.decision_stability_penalty,
            )
            decision_improvement = predicted_utility - baseline_utility
            decision_score = tf.tanh(decision_improvement / self.decision_scale)
            decision_loss = -tf.reduce_mean(weights_t * decision_score)
            route_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=route_targets_t, logits=route_logits)
            route_weights_t = tf.reshape(weights_t * (1.0 + 0.5 * route_margins_t), (-1,))
            route_loss = tf.reduce_mean(route_weights_t * route_ce)
            delta_mse = tf.reduce_mean(weights_t * tf.square(delta_targets_t - delta_pred))
            confidence_mse = tf.reduce_mean(weights_t * tf.square(confidence_targets_t - confidence_pred))
            expert_signal_mse = tf.reduce_mean(weights_t * tf.square(expert_signal_targets_t - routed_signal_pred))
            l2_loss = sum(tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights)
            loss = (
                self.route_loss_weight * route_loss
                + self.override_loss_weight * delta_mse
                + self.decision_weight * decision_loss
                + self.confidence_weight * confidence_mse
                + self.expert_signal_weight * expert_signal_mse
                + self.l2_reg * l2_loss
            )
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        route_probs = out["route_probs"]
        abstain_prob = route_probs[:, :1]
        active_route_probs = route_probs[:, 1:]
        selected_active_prob = tf.reduce_max(active_route_probs, axis=1, keepdims=True)
        effective_active_prob = tf.where(
            abstain_prob >= selected_active_prob,
            tf.zeros_like(selected_active_prob),
            selected_active_prob,
        )

        return {
            "loss": float(loss.numpy()),
            "route_loss": float(route_loss.numpy()),
            "decision_loss": float(decision_loss.numpy()),
            "decision_improvement_mean": float(tf.reduce_mean(decision_improvement).numpy()),
            "delta_mse": float(delta_mse.numpy()),
            "confidence_mse": float(confidence_mse.numpy()),
            "expert_signal_mse": float(expert_signal_mse.numpy()),
            "route_entropy": float(tf.reduce_mean(out["route_entropy"]).numpy()),
            "route_active_prob": float(tf.reduce_mean(effective_active_prob).numpy()),
            "route_non_abstain_prob": float(tf.reduce_mean(1.0 - out["abstain_prob"]).numpy()),
        }

    def _build_input(self, obs_stack: np.ndarray, ff_stack: np.ndarray) -> np.ndarray:
        if ff_stack.shape[1] != self.forecast_dim:
            ff_fixed = np.zeros((ff_stack.shape[0], self.forecast_dim), dtype=np.float32)
            n = min(ff_stack.shape[1], self.forecast_dim)
            ff_fixed[:, :n] = ff_stack[:, :n]
            ff_stack = ff_fixed
        if obs_stack.shape[1] != self.obs_dim:
            obs_stack = obs_stack[:, :self.obs_dim]
        return np.concatenate([obs_stack, ff_stack], axis=1)

    def train_on_rollout(
        self,
        experiences: List[Dict[str, Any]],
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train on one rollout using a baseline-relative decision-focused objective.
        """
        if len(experiences) < 8:
            return {'status': 'insufficient_data', 'loss': 0.0}

        obs_stack = np.stack([e['obs'] for e in experiences]).astype(np.float32)
        ff_stack = np.stack([e['forecast_features'] for e in experiences]).astype(np.float32)
        realized_returns = np.array([e.get('realized_return', 0.0) for e in experiences], dtype=np.float32)
        base_exposures = np.asarray(
            [_extract_base_exposure_from_features(e.get("forecast_features", [])) for e in experiences],
            dtype=np.float32,
        )
        executed_exposures = np.asarray(
            [_extract_executed_exposure_from_experience(e) for e in experiences],
            dtype=np.float32,
        )
        training_bundle = build_routed_decision_training_bundle(
            realized_returns,
            base_exposures,
            executed_exposures,
            ff_stack,
            self.decision_horizon,
            delta_limit=self.delta_limit,
            utility_scale=self.decision_scale,
            downside_weight=self.decision_downside_weight,
            vol_weight=self.decision_vol_weight,
            stability_penalty=self.decision_stability_penalty,
            magnitude_grid_points=self.magnitude_grid_points,
            conformal_risk_scale=self.conformal_risk_scale,
            runtime_conformal_margin=self.runtime_conformal_margin,
            runtime_gain=self.runtime_gain,
        )

        x = self._build_input(obs_stack, ff_stack)
        metrics = {}
        for _ in range(self.train_epochs):
            metrics = self._train_step(
                x,
                training_bundle["route_targets"],
                training_bundle["delta_targets"],
                training_bundle["confidence_targets"],
                training_bundle["base_exposures"],
                training_bundle["gross_returns"],
                training_bundle["downside_returns"],
                training_bundle["vol_returns"],
                sample_weights=training_bundle["sample_weights"],
                route_margins=training_bundle["route_margins"],
                expert_signal_targets=training_bundle["expert_signal_targets"],
            )
        return {
            'status': 'ok',
            'loss': metrics['loss'],
            'route_loss': metrics.get('route_loss', 0.0),
            'decision_loss': metrics.get('decision_loss', 0.0),
            'decision_improvement_mean': metrics.get('decision_improvement_mean', 0.0),
            'delta_mse': metrics.get('delta_mse', 0.0),
            'confidence_mse': metrics.get('confidence_mse', 0.0),
            'expert_signal_mse': metrics.get('expert_signal_mse', 0.0),
            'route_entropy': metrics.get('route_entropy', 0.0),
            'route_active_prob': metrics.get('route_active_prob', 0.0),
            'route_non_abstain_prob': metrics.get('route_non_abstain_prob', 0.0),
            'delta_target_std': float(np.std(training_bundle["delta_targets"])),
            'confidence_target_mean': float(np.mean(training_bundle["confidence_targets"])),
            'quality_target_mean': float(np.mean(training_bundle["quality_targets"])),
            'route_target_mean': float(np.mean(training_bundle["route_targets"] > 0)),
            'oracle_improvement_mean': float(np.mean(training_bundle["oracle_improvement"])),
            'certified_improvement_mean': float(np.mean(training_bundle["certified_improvement"])),
            'executed_advantage_mean': float(np.mean(training_bundle["executed_advantage"])),
        }

    def train_on_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train on a replay batch with routed decision-focused targets."""
        route_targets = np.asarray(batch.get('route_targets', np.zeros(batch['obs'].shape[0], dtype=np.int32)), dtype=np.int32)
        delta_targets = np.asarray(batch.get('delta_targets', np.zeros(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        confidence_targets = np.asarray(
            batch.get('confidence_targets', np.full(batch['obs'].shape[0], 0.5, dtype=np.float32)),
            dtype=np.float32,
        )
        route_margins = np.asarray(batch.get('route_margins', np.zeros(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        expert_signal_targets = np.asarray(batch.get('expert_signal_targets', np.sign(delta_targets)), dtype=np.float32)
        sample_weights = np.asarray(batch.get('sample_weights', np.ones(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        base_exposures = np.asarray(batch.get('base_exposures', np.zeros(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        gross_returns = np.asarray(batch.get('gross_returns', np.zeros(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        downside_returns = np.asarray(batch.get('downside_returns', np.zeros(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        vol_returns = np.asarray(batch.get('vol_returns', np.zeros(batch['obs'].shape[0], dtype=np.float32)), dtype=np.float32)
        x = self._build_input(batch['obs'], batch['forecast_features'])
        metrics = {}
        for _ in range(self.train_epochs):
            metrics = self._train_step(
                x,
                route_targets,
                delta_targets,
                confidence_targets,
                base_exposures,
                gross_returns,
                downside_returns,
                vol_returns,
                sample_weights=sample_weights,
                route_margins=route_margins,
                expert_signal_targets=expert_signal_targets,
            )
        return metrics

    def train_continuous(
        self,
        experiences: List[Dict[str, Any]],
        returns: Optional[np.ndarray],
        cv_buffer: "Tier2ExperienceBuffer",
    ) -> Dict[str, float]:
        """
        Continuous learning: train on current rollout, promote to replay, then sample from replay.
        """
        result = self.train_on_rollout(experiences, returns)
        if result['status'] != 'ok':
            return result

        promoted = cv_buffer.promote_with_returns(
            returns=None,
            experiences=experiences,
            decision_horizon=self.decision_horizon,
            decision_scale=self.decision_scale,
            decision_downside_weight=self.decision_downside_weight,
            decision_vol_weight=self.decision_vol_weight,
            decision_stability_penalty=self.decision_stability_penalty,
            magnitude_grid_points=self.magnitude_grid_points,
            delta_limit=self.delta_limit,
            conformal_risk_scale=self.conformal_risk_scale,
            runtime_conformal_margin=self.runtime_conformal_margin,
            runtime_gain=self.runtime_gain,
        )
        cv_buffer.clear_rollout()

        # Extra step: sample from replay for continuous learning
        if promoted > 0 and cv_buffer.replay_size_count() >= self.replay_batch_size:
            replay_batch = cv_buffer.sample_replay(self.replay_batch_size)
            if replay_batch is not None:
                extra = self.train_on_batch(replay_batch)
                result['replay_loss'] = extra.get('loss', 0.0)

        return result


# Backward-compatible aliases retained while the rest of the codebase migrates.
ControlVariateModel = Tier2RoutedOverlayModel
ControlVariateAdapter = Tier2RoutedOverlayAdapter
CVExperienceBuffer = Tier2ExperienceBuffer
ControlVariateTrainer = Tier2RoutedOverlayTrainer
