from __future__ import annotations

import logging
import math
import os
import pickle
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    TIER2_VALUE_CORE_DIM,
    TIER2_VALUE_FEATURE_DIM,
    TIER2_VALUE_MEMORY_CHANNELS,
    TIER2_VALUE_MEMORY_STEPS,
)
from utils import _get_tf

logger = logging.getLogger(__name__)

TIER2_PRIMARY_EXPERT_NAME = "ann"
TIER2_ACTIVE_EXPERT_NAMES = (TIER2_PRIMARY_EXPERT_NAME,)
TIER2_INTERNAL_EXPERTS = ("base", "trend", "reversion", "defensive")
TIER2_POLICY_IMPROVEMENT_FEATURE_DIM = int(TIER2_VALUE_FEATURE_DIM)

_CORE_KEYS = (
    "proposed_exposure",
    "abs_proposed_exposure",
    "gross_exposure_ratio",
    "tradeable_capital_ratio",
    "realized_volatility_regime",
    "investor_local_quality",
    "investor_local_quality_delta",
    "investor_local_drawdown",
    # DL self-conditioning: previous-step Tier2 outputs fed back
    "tier2_trust_radius",
    "tier2_advantage_strength",
    "tier2_deployment_scale",
)

_MEMORY_KEYS = (
    f"{TIER2_PRIMARY_EXPERT_NAME}_signal",
    f"{TIER2_PRIMARY_EXPERT_NAME}_quality",
    f"{TIER2_PRIMARY_EXPERT_NAME}_risk",
    "expert_consensus_signal",
    "expert_disagreement",
    "short_imbalance_signal",
    "ann_return_signal",
    "ann_abs_return",
    "ann_price_level_signal",
    "ann_direction_margin",
    "ann_direction_accuracy",
    "ann_latent_0",
    "ann_latent_1",
    "ann_latent_2",
    "ann_latent_3",
    "ann_latent_norm",
    "price_level_signal",
    "price_return_signal",
    "wind_level_signal",
    "solar_level_signal",
    "hydro_level_signal",
    "load_level_signal",
)


def get_tier2_weight_filenames(policy_family: Optional[str] = None) -> Tuple[str, ...]:
    del policy_family
    return ("tier2_policy_improvement.pkl", "tier2_policy_improvement.h5")


def get_primary_tier2_weight_filename(policy_family: Optional[str] = None) -> str:
    return str(get_tier2_weight_filenames(policy_family)[0])


def resolve_no_trade_threshold_dkk(
    threshold_config: Optional[float],
    *,
    max_position_notional: float,
) -> float:
    threshold = float(threshold_config or 0.0)
    max_notional = float(max(max_position_notional, 0.0))
    if threshold <= 0.0:
        return 0.0
    if threshold <= 1.0:
        return float(threshold * max_notional)
    return float(threshold)


def tier2_quality_gain(quality: float) -> float:
    q = float(np.clip(quality, 0.0, 1.0))
    return float(np.clip(0.20 + 0.80 * math.sqrt(q), 0.0, 1.0))


def tier2_risk_gain(
    conformal_risk: float,
    *,
    conformal_risk_scale: float = 1.0,
    runtime_conformal_margin: float = 0.05,
) -> float:
    risk = float(max(conformal_risk, 0.0))
    scale = float(max(conformal_risk_scale, 1e-6))
    margin = float(max(runtime_conformal_margin, 0.0))
    adjusted = max(risk - margin, 0.0) / scale
    return float(np.clip(1.0 / (1.0 + adjusted), 0.0, 1.0))


def tier2_expert_quality_blend(
    *,
    direction_accuracy: float,
    mape_quality: float,
    metadata_skill: float,
    economic_skill: float,
) -> float:
    parts = np.asarray(
        [
            float(np.clip(direction_accuracy, 0.0, 1.0)),
            float(np.clip(mape_quality, 0.0, 1.0)),
            float(np.clip(metadata_skill, 0.0, 1.0)),
            float(np.clip(economic_skill, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    weights = np.asarray([0.30, 0.20, 0.20, 0.30], dtype=np.float32)
    return float(np.clip(np.sum(parts * weights), 0.0, 1.0))


def tier2_short_expert_realized_utility_score(
    *,
    forecast_return: float,
    actual_return: float,
    utility_scale: float,
    delta_limit: float,
    runtime_gain: float,
    downside_weight: float,
) -> float:
    util_scale = float(max(abs(utility_scale), 1e-6))
    delta_cap = float(max(delta_limit, 1e-6))
    gain = float(max(runtime_gain, 1e-6))
    downside = float(max(downside_weight, 0.0))

    f_ret = float(forecast_return)
    a_ret = float(actual_return)
    aligned = 1.0 if (f_ret == 0.0 and a_ret == 0.0) else float(np.sign(f_ret) == np.sign(a_ret))
    magnitude = float(np.clip(abs(a_ret) / util_scale, 0.0, 1.0))
    forecast_conf = float(np.clip(abs(f_ret) / delta_cap, 0.0, 1.0))
    downside_penalty = float(np.clip(max(-a_ret, 0.0) / util_scale, 0.0, 1.0))

    raw = (
        0.55 * (2.0 * aligned - 1.0)
        + 0.20 * magnitude
        + 0.15 * forecast_conf
        - 0.10 * downside * downside_penalty
    )
    return float(np.clip(0.5 + 0.5 * math.tanh(raw * gain), 0.0, 1.0))


def extract_tier2_core_state_from_features(
    tier2_features: Sequence[float],
    *,
    core_dim: int = TIER2_VALUE_CORE_DIM,
) -> Dict[str, float]:
    arr = np.asarray(tier2_features, dtype=np.float32).flatten()
    vals = np.zeros(max(core_dim, len(_CORE_KEYS)), dtype=np.float32)
    use = min(arr.size, core_dim, vals.size)
    if use > 0:
        vals[:use] = arr[:use]
    data = {key: float(vals[idx]) for idx, key in enumerate(_CORE_KEYS[:core_dim])}
    return data


def extract_tier2_memory_state_from_features(
    tier2_features: Sequence[float],
    *,
    core_dim: int = TIER2_VALUE_CORE_DIM,
) -> Dict[str, float]:
    arr = np.asarray(tier2_features, dtype=np.float32).flatten()
    offset = int(max(core_dim, 0))
    memory = arr[offset:]
    latest = np.zeros(TIER2_VALUE_MEMORY_CHANNELS, dtype=np.float32)
    if memory.size > 0:
        latest_slice = memory[-TIER2_VALUE_MEMORY_CHANNELS :]
        latest[-latest_slice.size :] = latest_slice

    data = {key: float(latest[idx]) for idx, key in enumerate(_MEMORY_KEYS)}
    ann_signal = float(data.get(f"{TIER2_PRIMARY_EXPERT_NAME}_signal", 0.0))
    ann_quality = float(np.clip(data.get(f"{TIER2_PRIMARY_EXPERT_NAME}_quality", 0.5), 0.0, 1.0))
    ann_risk = float(np.clip(data.get(f"{TIER2_PRIMARY_EXPERT_NAME}_risk", 0.5), 0.0, 1.0))
    ann_direction_accuracy = float(np.clip(data.get("ann_direction_accuracy", ann_quality), 0.0, 1.0))
    disagreement = float(np.clip(data.get("expert_disagreement", 0.0), 0.0, 1.0))
    latent_norm = float(np.clip(data.get("ann_latent_norm", 0.0), 0.0, 1.0))

    data["primary_signal"] = ann_signal
    data["primary_quality"] = ann_quality
    data["primary_risk"] = ann_risk
    data["ann_direction_accuracy"] = ann_direction_accuracy
    data["primary_quality_gap"] = 0.0
    data["primary_risk_gap"] = 0.0
    data["primary_vs_consensus_gap"] = 0.0
    data["primary_leadership"] = 0.0
    data["quality_context"] = float(np.clip(0.60 * ann_quality + 0.40 * ann_direction_accuracy, 0.0, 1.0))
    data["context_strength"] = float(np.clip(max(abs(ann_signal), latent_norm), 0.0, 1.0))
    data["alignment"] = float(np.clip(ann_signal * (1.0 - disagreement), -1.0, 1.0))
    return data


class CalibrationTracker:
    def __init__(
        self,
        *,
        window_size: int = 500,
        trust_metric: str = "hitrate",
        verbose: bool = False,
        init_budget: float = 0.0,
        direction_weight: float = 0.8,
        trust_boost: float = 0.0,
        fail_fast: bool = False,
        metadata_quality_path: Optional[str] = None,
        metadata_quality_weight: float = 0.3,
    ) -> None:
        del init_budget
        self.window_size = int(max(window_size, 10))
        self.trust_metric = str(trust_metric or "hitrate").strip().lower()
        self.verbose = bool(verbose)
        self.direction_weight = float(np.clip(direction_weight, 0.0, 1.0))
        self.trust_boost = float(max(trust_boost, 0.0))
        self.fail_fast = bool(fail_fast)
        self.metadata_quality_weight = float(np.clip(metadata_quality_weight, 0.0, 1.0))
        self.metadata_quality = self._load_metadata_quality(metadata_quality_path)
        self.forecast_history: Deque[float] = deque(maxlen=self.window_size)
        self.realized_history: Deque[float] = deque(maxlen=self.window_size)
        self._history_by_horizon: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: {
                "forecast": deque(maxlen=self.window_size),
                "realized": deque(maxlen=self.window_size),
                "abs_error": deque(maxlen=self.window_size),
                "direction_hit": deque(maxlen=self.window_size),
            }
        )

    def _load_metadata_quality(self, path: Optional[str]) -> Optional[float]:
        """Load directional accuracy from ANN metadata."""
        if not path:
            return None
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            val_metrics = data.get('val_metrics', {})
            test_metrics = data.get('test_metrics', {})
            # Use test metrics if available, else val
            metrics = test_metrics or val_metrics
            return float(metrics.get('directional_accuracy', 0.5))
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to load directional accuracy from {path}: {e}")
            return None

    def update(self, *, forecast: float, realized: float, horizon: str = "short") -> None:
        try:
            f_val = float(forecast)
            r_val = float(realized)
            name = str(horizon or "short")
            hit = 1.0 if (f_val == 0.0 and r_val == 0.0) else float(np.sign(f_val) == np.sign(r_val))
            abs_error = float(abs(f_val - r_val))

            self.forecast_history.append(f_val)
            self.realized_history.append(r_val)
            bucket = self._history_by_horizon[name]
            bucket["forecast"].append(f_val)
            bucket["realized"].append(r_val)
            bucket["abs_error"].append(abs_error)
            bucket["direction_hit"].append(hit)
        except Exception as exc:
            if self.fail_fast:
                raise
            if self.verbose:
                logger.warning("[TIER2_CALIBRATION_UPDATE] %s", exc)

    def get_trust(self, *, horizon: str = "short", recent_mape: Optional[float] = None) -> float:
        bucket = self._history_by_horizon.get(str(horizon or "short"))
        runtime_trust = 0.5

        if bucket:
            hits = np.asarray(list(bucket["direction_hit"]), dtype=np.float32)
            errs = np.asarray(list(bucket["abs_error"]), dtype=np.float32)
            if hits.size >= 5 and errs.size >= 5:
                hit_score = float(np.clip(np.mean(hits), 0.0, 1.0))
                err_score = float(np.clip(1.0 - min(float(np.mean(errs)) / 2.0, 1.0), 0.0, 1.0))

                if self.trust_metric in {"hitrate", "sign", "direction", "absdir"}:
                    runtime_trust = hit_score
                elif self.trust_metric == "combo":
                    runtime_trust = self.direction_weight * hit_score + (1.0 - self.direction_weight) * err_score
                else:
                    runtime_trust = err_score

        # Incorporate directional accuracy if available
        if self.metadata_quality is not None:
            runtime_trust = (1.0 - self.metadata_quality_weight) * runtime_trust + self.metadata_quality_weight * self.metadata_quality

        if recent_mape is not None and np.isfinite(recent_mape):
            mape_score = float(np.clip(1.0 - min(float(recent_mape) / 0.25, 1.0), 0.0, 1.0))
            runtime_trust = 0.70 * runtime_trust + 0.30 * mape_score

        return float(np.clip(runtime_trust + self.trust_boost, 0.0, 1.0))


class Tier2PolicyImprovementBuffer:
    def __init__(self, *, max_size: int = 50_000, replay_size: int = 20_000, seed: int = 42) -> None:
        self.max_size = int(max(max_size, 128))
        self.replay_size = int(max(replay_size, 128))
        self._rollout: List[Dict[str, Any]] = []
        self._replay: Deque[Dict[str, Any]] = deque(maxlen=self.replay_size)
        self._rng = random.Random(int(seed))

    def add(
        self,
        obs: Sequence[float],
        tier2_features: Sequence[float],
        *,
        realized_return: float,
        executed_exposure: Optional[float],
        rollout_pos: Optional[int],
        env_step: int,
        return_horizon_steps: int,
        decision_step_sampled: bool,
    ) -> None:
        sample = {
            "obs": np.asarray(obs, dtype=np.float32).flatten(),
            "tier2_features": np.asarray(tier2_features, dtype=np.float32).flatten(),
            "realized_return": float(realized_return),
            "executed_exposure": float(executed_exposure) if executed_exposure is not None else None,
            "rollout_pos": int(rollout_pos) if rollout_pos is not None else -1,
            "env_step": int(env_step),
            "return_horizon_steps": int(max(return_horizon_steps, 1)),
            "decision_step_sampled": bool(decision_step_sampled),
        }
        self._rollout.append(sample)
        self._replay.append(sample)
        if len(self._rollout) > self.max_size:
            self._rollout = self._rollout[-self.max_size :]

    def size(self) -> int:
        return int(len(self._rollout) + len(self._replay))

    def get_current_rollout(self) -> List[Dict[str, Any]]:
        return list(self._rollout)

    def clear_rollout(self) -> None:
        self._rollout.clear()

    def sample_replay(self, batch_size: int) -> List[Dict[str, Any]]:
        if len(self._replay) == 0:
            return []
        k = int(min(max(batch_size, 1), len(self._replay)))
        return self._rng.sample(list(self._replay), k=k)


def _safe_softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - np.max(arr, axis=-1, keepdims=True)
    expv = np.exp(arr)
    den = np.sum(expv, axis=-1, keepdims=True)
    den = np.maximum(den, 1e-6)
    return expv / den


class Tier2PolicyImprovementAdapter:
    def __init__(
        self,
        *,
        obs_dim: int,
        forecast_dim: int,
        policy_family: Optional[str] = None,
        memory_steps: int = TIER2_VALUE_MEMORY_STEPS,
        delta_limit: float = 0.10,
        value_target_scale: float = 0.01,
        runtime_min_abs_delta: float = 0.003,
        nav_head_scale: float = 3.0,
        return_floor_head_scale: float = 4.0,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)
        self.policy_family = str(policy_family or "forecast_guided_meta_baseline")
        self.memory_steps = int(max(memory_steps, 1))
        self.delta_limit = float(max(delta_limit, 1e-6))
        self.value_target_scale = float(max(value_target_scale, 1e-6))
        self.runtime_min_abs_delta = float(max(runtime_min_abs_delta, 0.0))
        self.nav_head_scale = float(max(nav_head_scale, 1e-6))
        self.return_floor_head_scale = float(max(return_floor_head_scale, 1e-6))
        self.seed = int(seed)
        self._has_persisted = False
        self._compiled_forward_infer = None
        self._compiled_forward_train = None

        self._tf = _get_tf()
        self.model = None
        if self._tf is None:
            raise RuntimeError("Tier-2 requires TensorFlow for the full DL path")
        self._build_model()

    def _build_model(self) -> None:
        tf = self._tf
        if tf is None:
            return

        try:
            tf.keras.utils.set_random_seed(self.seed)
        except Exception:
            pass

        obs_in = tf.keras.Input(shape=(self.obs_dim,), name="obs")
        feat_in = tf.keras.Input(shape=(self.forecast_dim,), name="features")

        core_in = tf.keras.layers.Lambda(
            lambda x: x[:, :TIER2_VALUE_CORE_DIM],
            name="core_slice",
        )(feat_in)
        memory_flat = tf.keras.layers.Lambda(
            lambda x: x[:, TIER2_VALUE_CORE_DIM:],
            name="memory_slice",
        )(feat_in)
        memory_tokens = tf.keras.layers.Reshape(
            (TIER2_VALUE_MEMORY_STEPS, TIER2_VALUE_MEMORY_CHANNELS),
            name="memory_reshape",
        )(memory_flat)

        obs_x = tf.keras.layers.LayerNormalization(name="obs_norm")(obs_in)
        obs_x = tf.keras.layers.Dense(64, activation="swish", name="obs_proj")(obs_x)

        core_x = tf.keras.layers.LayerNormalization(name="core_norm")(core_in)
        core_x = tf.keras.layers.Dense(64, activation="swish", name="core_proj")(core_x)

        state_ctx = tf.keras.layers.Concatenate(name="state_concat")([obs_x, core_x])
        state_ctx = tf.keras.layers.Dense(128, activation="swish", name="state_dense1")(state_ctx)
        state_ctx = tf.keras.layers.Dense(128, activation="swish", name="state_dense2")(state_ctx)

        mem_x = tf.keras.layers.LayerNormalization(name="memory_norm")(memory_tokens)
        mem_x = tf.keras.layers.Dense(96, activation="swish", name="memory_token_proj")(mem_x)

        # State-conditioned FiLM modulation lets the Tier-1 context query the
        # short-horizon ANN forecast cache directly instead of flattening it.
        film_gamma = tf.keras.layers.Dense(96, activation="tanh", name="film_gamma")(state_ctx)
        film_beta = tf.keras.layers.Dense(96, activation="tanh", name="film_beta")(state_ctx)
        film_gamma = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="film_gamma_expand")(film_gamma)
        film_beta = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="film_beta_expand")(film_beta)
        mem_x = tf.keras.layers.Lambda(
            lambda xs: xs[0] * (1.0 + 0.20 * xs[1]) + 0.20 * xs[2],
            name="film_modulation",
        )([mem_x, film_gamma, film_beta])

        temporal_conv = tf.keras.layers.Conv1D(
            filters=96,
            kernel_size=3,
            padding="causal",
            activation="swish",
            name="memory_temporal_conv",
        )(mem_x)
        mem_x = tf.keras.layers.LayerNormalization(name="memory_conv_norm")(mem_x + temporal_conv)

        self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=24,
            dropout=0.05,
            name="memory_self_attention",
        )(mem_x, mem_x, mem_x)
        mem_x = tf.keras.layers.LayerNormalization(name="memory_self_attn_norm")(mem_x + self_attn)

        mem_ffn = tf.keras.layers.Dense(128, activation="swish", name="memory_ffn1")(mem_x)
        mem_ffn = tf.keras.layers.Dense(96, activation=None, name="memory_ffn2")(mem_ffn)
        mem_x = tf.keras.layers.LayerNormalization(name="memory_ffn_norm")(mem_x + mem_ffn)

        query = tf.keras.layers.Dense(96, activation="swish", name="state_query")(state_ctx)
        query = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="state_query_expand")(query)
        cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=24,
            dropout=0.05,
            name="state_to_memory_attention",
        )(query, mem_x, mem_x)
        forecast_ctx = tf.keras.layers.LayerNormalization(name="forecast_ctx_norm")(query + cross_attn)
        forecast_ctx = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="forecast_ctx_squeeze")(forecast_ctx)

        temporal_logits = tf.keras.layers.Dense(1, activation=None, name="temporal_importance_logits")(mem_x)
        temporal_weights = tf.keras.layers.Softmax(axis=1, name="temporal_importance")(temporal_logits)
        temporal_pool = tf.keras.layers.Lambda(
            lambda xs: tf.reduce_sum(xs[0] * xs[1], axis=1),
            name="temporal_weighted_pool",
        )([mem_x, temporal_weights])

        shared_in = tf.keras.layers.Concatenate(name="fusion_concat")([state_ctx, forecast_ctx, temporal_pool])
        shared_in = tf.keras.layers.Dense(160, activation="swish", name="fusion_dense1")(shared_in)
        shared_in = tf.keras.layers.Dropout(0.08, name="fusion_dropout")(shared_in)
        shared_in = tf.keras.layers.Dense(128, activation="swish", name="fusion_dense2")(shared_in)

        # --- ACRP-v2: Learned DA Embedding (FIXED architecture) ---
        # Extract per-step directional accuracy from memory channel index 10.
        # FIX: Use 1D causal convolution + delta-encoding to capture the DA
        # *trajectory* (improving, degrading, volatile) rather than Dense
        # layers that collapsed to a constant (~1.0).
        # The delta-encoding forces temporal contrast: the encoder must
        # represent changes in DA, not level.
        da_channel_idx = _MEMORY_KEYS.index("ann_direction_accuracy")
        da_per_step = tf.keras.layers.Lambda(
            lambda tokens, idx=da_channel_idx: tokens[:, :, idx],
            name="da_channel_extract",
        )(memory_tokens)  # (batch, memory_steps)
        # Expand to 3D for Conv1D: (batch, steps, 1)
        da_expanded = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name="da_expand",
        )(da_per_step)
        # Temporal convolution captures local DA trends
        da_conv = tf.keras.layers.Conv1D(
            filters=8, kernel_size=3, padding="causal",
            activation="swish", name="da_temporal_conv",
        )(da_expanded)
        da_conv = tf.keras.layers.LayerNormalization(name="da_conv_norm")(da_conv)
        # Global average pool → fixed-size representation
        da_encoded = tf.keras.layers.GlobalAveragePooling1D(name="da_pool")(da_conv)
        da_encoded = tf.keras.layers.Dense(8, activation="swish", name="da_encoder_2")(da_encoded)
        # Learned DA scalar: sigmoid output, but trained with contrastive
        # target (see training changes below) instead of raw DA level.
        da_learned_scalar = tf.keras.layers.Dense(1, activation="sigmoid", name="da_learned_scalar")(da_encoded)

        gate_hidden = tf.keras.layers.Dense(96, activation="swish", name="gate_hidden")(shared_in)
        expert_mix = tf.keras.layers.Dense(
            len(TIER2_INTERNAL_EXPERTS),
            activation="softmax",
            name="expert_mix",
        )(gate_hidden)

        expert_latents = []
        for expert_name in TIER2_INTERNAL_EXPERTS:
            ex = tf.keras.layers.Dense(128, activation="swish", name=f"{expert_name}_dense1")(shared_in)
            ex = tf.keras.layers.Dropout(0.08, name=f"{expert_name}_dropout")(ex)
            ex = tf.keras.layers.Dense(96, activation="swish", name=f"{expert_name}_dense2")(ex)
            expert_latents.append(ex)

        expert_stack = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="expert_stack")(expert_latents)
        mix_expand = tf.keras.layers.Lambda(lambda w: tf.expand_dims(w, axis=-1), name="expert_mix_expand")(expert_mix)
        mixed = tf.keras.layers.Lambda(lambda xs: tf.reduce_sum(xs[0] * xs[1], axis=1), name="expert_weighted_sum")(
            [expert_stack, mix_expand]
        )

        trunk = tf.keras.layers.Dense(96, activation="swish", name="trunk_dense1")(mixed)
        trunk = tf.keras.layers.Dense(64, activation="swish", name="trunk_dense2")(trunk)
        trunk = tf.keras.layers.Dropout(0.05, name="trunk_dropout")(trunk)

        # --- ACRP-v2: Learned Trust Network ---
        # The trust network now receives both the trunk (market context) AND
        # the learned DA embedding (forecast trajectory quality).  This lets
        # it learn trust surfaces conditioned on regime × DA trajectory —
        # e.g. "DA was 0.55 but rising in a trending market → trust more"
        # vs. "DA 0.55 but falling in choppy market → trust less".
        trust_input = tf.keras.layers.Concatenate(name="trust_input_concat")([trunk, da_encoded])
        trust_hidden = tf.keras.layers.Dense(32, activation="swish", name="trust_hidden")(trust_input)
        trust_radius_out = tf.keras.layers.Dense(1, activation="sigmoid", name="trust_radius")(trust_hidden)

        # --- ACRP-v2: Adaptive Advantage Attention ---
        # Diagnostic head: per-sample softmax over [gate, confidence, value×floor].
        # Still supervised (diversity loss) and logged, but deployment is now
        # determined by the end-to-end deployment head below.
        adv_attn_hidden = tf.keras.layers.Dense(32, activation="swish", name="adv_attn_hidden")(trunk)
        adv_attn_weights = tf.keras.layers.Dense(3, activation="softmax", name="adv_attn_weights")(adv_attn_hidden)

        # Individual auxiliary heads (still have their own losses for
        # representation shaping; the deployment head reads their outputs).
        delta_raw_out = tf.keras.layers.Dense(1, activation="tanh", name="delta_raw")(trunk)
        gate_prob_out = tf.keras.layers.Dense(1, activation="sigmoid", name="gate_prob")(trunk)
        uplift_mean_out = tf.keras.layers.Dense(1, activation="tanh", name="uplift_mean")(trunk)
        uplift_sigma_out = tf.keras.layers.Dense(1, activation="softplus", name="uplift_sigma")(trunk)
        nav_pred_out = tf.keras.layers.Dense(1, activation="tanh", name="nav_prediction")(trunk)
        nav_sigma_out = tf.keras.layers.Dense(1, activation="softplus", name="nav_sigma")(trunk)
        floor_pred_out = tf.keras.layers.Dense(1, activation="tanh", name="return_floor_prediction")(trunk)
        floor_sigma_out = tf.keras.layers.Dense(1, activation="softplus", name="return_floor_sigma")(trunk)
        tail_risk_out = tf.keras.layers.Dense(1, activation="sigmoid", name="tail_risk")(trunk)
        factor_opp_out = tf.keras.layers.Dense(1, activation="sigmoid", name="factor_opportunity")(trunk)

        # --- ACRP-v3: End-to-end Learned Deployment Head ---
        # The deployment decision is now FULLY learned.  Instead of a
        # hand-coded formula (advantage × trust × risk_veto), a small MLP
        # receives the trunk latent together with ALL auxiliary head outputs
        # and learns the optimal deployment surface end-to-end through the
        # counterfactual Sharpe actor loss.
        #
        # Input: trunk(64) + gate(1) + uplift_mean(1) + uplift_sigma(1) +
        #        nav(1) + nav_sigma(1) + floor(1) + floor_sigma(1) +
        #        tail_risk(1) + trust_radius(1) + adv_attn(3) +
        #        da_learned(1) = 78D
        #
        # The head can learn any combination function (multiplicative,
        # additive, conditional) from the auxiliary signals, while the
        # auxiliary losses still shape the individual head representations.
        # Stop gradients from the deployment head flowing back into the
        # auxiliary heads.  Without this, the actor loss (which trains
        # deployment_scale for Sharpe maximization) would also train
        # gate_prob, uplift_mean, etc. — exactly the competing-gradient
        # problem Fix 3 was designed to solve.  The auxiliary heads keep
        # their own supervised losses; the deployment head can READ their
        # values but cannot MODIFY their weights.
        deployment_input = tf.keras.layers.Concatenate(name="deployment_input")([
            trunk,              # 64D  - full market/forecast context (shared backbone — gradient OK)
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_gate")(gate_prob_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_uplift_mean")(uplift_mean_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_uplift_sigma")(uplift_sigma_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_nav")(nav_pred_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_nav_sigma")(nav_sigma_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_floor")(floor_pred_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_floor_sigma")(floor_sigma_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_tail")(tail_risk_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_trust")(trust_radius_out),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_adv_attn")(adv_attn_weights),
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="sg_da")(da_learned_scalar),
        ])
        deployment_hidden = tf.keras.layers.Dense(48, activation="swish", name="deployment_hidden1")(deployment_input)
        deployment_hidden = tf.keras.layers.Dense(24, activation="swish", name="deployment_hidden2")(deployment_hidden)
        deployment_scale_out = tf.keras.layers.Dense(1, activation="sigmoid", name="deployment_scale")(deployment_hidden)

        outputs = {
            "delta_raw": delta_raw_out,
            "gate_prob": gate_prob_out,
            "uplift_mean": uplift_mean_out,
            "uplift_sigma": uplift_sigma_out,
            "nav_prediction": nav_pred_out,
            "nav_sigma": nav_sigma_out,
            "return_floor_prediction": floor_pred_out,
            "return_floor_sigma": floor_sigma_out,
            "tail_risk": tail_risk_out,
            "factor_opportunity": factor_opp_out,
            "expert_mix": expert_mix,
            "trust_radius": trust_radius_out,
            "adv_attn_weights": adv_attn_weights,
            "da_learned_scalar": da_learned_scalar,
            "deployment_scale": deployment_scale_out,
        }
        self.model = tf.keras.Model(
            inputs=[obs_in, feat_in],
            outputs=outputs,
            name="tier2_cross_attention_forecast_guided_fgb",
        )
        try:
            # Compile model forward paths to reduce Python overhead in runtime/train loops.
            self._compiled_forward_infer = tf.function(
                lambda obs, feat: self.model([obs, feat], training=False),
                reduce_retracing=True,
            )
            self._compiled_forward_train = tf.function(
                lambda obs, feat: self.model([obs, feat], training=True),
                reduce_retracing=True,
            )
        except Exception:
            self._compiled_forward_infer = None
            self._compiled_forward_train = None

    def has_persisted_weights(self) -> bool:
        return bool(self._has_persisted)

    def save_weights(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "obs_dim": self.obs_dim,
                "forecast_dim": self.forecast_dim,
                "policy_family": self.policy_family,
                "memory_steps": self.memory_steps,
                "delta_limit": self.delta_limit,
                "value_target_scale": self.value_target_scale,
                "runtime_min_abs_delta": self.runtime_min_abs_delta,
                "nav_head_scale": self.nav_head_scale,
                "return_floor_head_scale": self.return_floor_head_scale,
                "seed": self.seed,
                "weights": self.model.get_weights() if self.model is not None else None,
            }
            with open(path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self._has_persisted = True
            return True
        except Exception as exc:
            logger.warning("[TIER2_SAVE] %s", exc)
            return False

    def load_weights(self, path: str) -> bool:
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if int(payload.get("obs_dim", -1)) != self.obs_dim:
                return False
            if int(payload.get("forecast_dim", -1)) != self.forecast_dim:
                return False
            weights = payload.get("weights", None)
            if self.model is not None and weights is not None:
                self.model.set_weights(weights)
            self._has_persisted = True
            return True
        except Exception as exc:
            logger.warning("[TIER2_LOAD] %s", exc)
            return False

    def _predict_dict(self, obs_batch: np.ndarray, feat_batch: np.ndarray, *, training: bool) -> Dict[str, np.ndarray]:
        if self.model is None or self._tf is None:
            raise RuntimeError("TensorFlow model unavailable")

        tf = self._tf
        obs_tensor = tf.convert_to_tensor(np.asarray(obs_batch, dtype=np.float32))
        feat_tensor = tf.convert_to_tensor(np.asarray(feat_batch, dtype=np.float32))
        if training and self._compiled_forward_train is not None:
            outputs = self._compiled_forward_train(obs_tensor, feat_tensor)
        elif (not training) and self._compiled_forward_infer is not None:
            outputs = self._compiled_forward_infer(obs_tensor, feat_tensor)
        else:
            outputs = self.model([obs_tensor, feat_tensor], training=training)
        return {
            key: np.asarray(value.numpy(), dtype=np.float32)
            for key, value in outputs.items()
        }

    def predict_runtime_bundle(
        self,
        investor_obs: Sequence[float],
        tier2_features: Sequence[float],
        *,
        samples: int = 1,
    ) -> Dict[str, Any]:
        obs_np = np.asarray(investor_obs, dtype=np.float32).reshape(1, -1)
        feat_np = np.asarray(tier2_features, dtype=np.float32).reshape(1, -1)
        if obs_np.shape[1] != self.obs_dim or feat_np.shape[1] != self.forecast_dim:
            raise ValueError(
                f"Tier2 runtime input mismatch: obs={obs_np.shape[1]} expected {self.obs_dim}, "
                f"features={feat_np.shape[1]} expected {self.forecast_dim}"
            )

        if self.model is None or self._tf is None:
            raise RuntimeError("Tier-2 runtime requires an initialized TensorFlow model")

        passes = max(int(samples), 1)
        if passes == 1:
            mean_pred = self._predict_dict(obs_np, feat_np, training=False)
            std_pred = {
                key: np.zeros_like(value, dtype=np.float32)
                for key, value in mean_pred.items()
            }
        else:
            pred_list: List[Dict[str, np.ndarray]] = []
            for _ in range(passes):
                pred_list.append(self._predict_dict(obs_np, feat_np, training=True))

            mean_pred: Dict[str, np.ndarray] = {}
            std_pred: Dict[str, np.ndarray] = {}
            for key in pred_list[0].keys():
                stacked = np.stack([pred[key] for pred in pred_list], axis=0)
                mean_pred[key] = np.mean(stacked, axis=0)
                std_pred[key] = np.std(stacked, axis=0)

        core = extract_tier2_core_state_from_features(feat_np[0])
        memory = extract_tier2_memory_state_from_features(feat_np[0])
        base_exposure = float(np.clip(core.get("proposed_exposure", 0.0), -1.0, 1.0))
        gate_prob = float(np.clip(mean_pred["gate_prob"][0, 0], 0.0, 1.0))
        delta_raw = float(np.clip(mean_pred["delta_raw"][0, 0], -1.0, 1.0))
        raw_delta = float(np.clip(delta_raw * self.delta_limit, -self.delta_limit, self.delta_limit))

        uplift_mean = float(mean_pred["uplift_mean"][0, 0]) * self.value_target_scale
        sigma_base = float(mean_pred["uplift_sigma"][0, 0]) * self.value_target_scale
        sigma_mc = float(std_pred["uplift_mean"][0, 0]) * self.value_target_scale
        sigma = float(max(sigma_base + sigma_mc, 1e-6))

        nav_pred = float(mean_pred["nav_prediction"][0, 0]) * self.value_target_scale * self.nav_head_scale
        nav_sigma_base = float(mean_pred["nav_sigma"][0, 0]) * self.value_target_scale * self.nav_head_scale
        nav_sigma_mc = float(std_pred["nav_prediction"][0, 0]) * self.value_target_scale * self.nav_head_scale
        nav_sigma = float(max(nav_sigma_base + nav_sigma_mc, 1e-6))
        floor_pred = float(mean_pred["return_floor_prediction"][0, 0]) * self.value_target_scale * self.return_floor_head_scale
        floor_sigma_base = float(mean_pred["return_floor_sigma"][0, 0]) * self.value_target_scale * self.return_floor_head_scale
        floor_sigma_mc = float(std_pred["return_floor_prediction"][0, 0]) * self.value_target_scale * self.return_floor_head_scale
        floor_sigma = float(max(floor_sigma_base + floor_sigma_mc, 1e-6))
        tail_risk = float(np.clip(mean_pred["tail_risk"][0, 0], 0.0, 1.0))
        factor_pred = float(np.clip(mean_pred["factor_opportunity"][0, 0], 0.0, 1.0))
        expert_mix = np.asarray(mean_pred["expert_mix"][0], dtype=np.float32)

        # --- ACRP-v2: Read learned trust radius, adaptive advantage weights, DA embedding ---
        learned_trust_radius = float(np.clip(mean_pred["trust_radius"][0, 0], 0.0, 1.0))
        adv_weights = np.clip(mean_pred["adv_attn_weights"][0], 1e-6, 1.0)
        adv_weights = adv_weights / adv_weights.sum()  # re-normalize after clip
        da_learned = float(np.clip(mean_pred["da_learned_scalar"][0, 0], 0.0, 1.0))

        # --- ACRP-v3: Read end-to-end deployment scale directly from model ---
        deployment_scale_val = float(np.clip(mean_pred["deployment_scale"][0, 0], 0.0, 1.0))

        signal = float(np.clip(memory.get("primary_signal", 0.0), -1.0, 1.0))
        quality = float(np.clip(memory.get("primary_quality", 0.5), 0.0, 1.0))
        primary_risk = float(np.clip(memory.get("primary_risk", 0.5), 0.0, 1.0))
        direction_accuracy = float(np.clip(memory.get("ann_direction_accuracy", quality), 0.0, 1.0))
        forecast_trust = float(np.clip((0.60 * quality + 0.40 * direction_accuracy) * (1.0 - 0.35 * primary_risk), 0.0, 1.0))
        controller_sigma = float(
            max(
                sigma,
                nav_sigma / max(self.nav_head_scale, 1e-6),
                floor_sigma / max(self.return_floor_head_scale, 1e-6),
            )
        )
        confidence_support = float(
            np.clip(1.0 / (1.0 + controller_sigma / max(self.value_target_scale, 1e-6)), 0.0, 1.0)
        )
        value_lcb = float(uplift_mean - 1.2816 * sigma)
        value_ucb = float(uplift_mean + 1.2816 * sigma)
        nav_lcb = float(nav_pred - 1.2816 * nav_sigma)
        floor_lcb = float(floor_pred - 0.75 * floor_sigma)
        value_support = float(np.clip(1.0 / (1.0 + math.exp(-value_lcb / max(self.value_target_scale, 1e-6))), 0.0, 1.0))
        floor_support = float(np.clip(1.0 / (1.0 + math.exp(-floor_lcb / max(self.value_target_scale, 1e-6))), 0.0, 1.0))
        risk_support = float(np.clip(1.0 - tail_risk, 0.0, 1.0))
        projection_strength = float(np.clip(gate_prob, 0.0, 1.0))
        route_uncertainty_quality = risk_support

        # --- ACRP-v2: Learned Trust + Adaptive Advantage (DIAGNOSTIC ONLY) ---
        # These remain computed for logging / interpretability, but the actual
        # deployment decision is now driven solely by the end-to-end
        # deployment head (ACRP-v3).
        trust_radius = learned_trust_radius

        # Advantage strength: diagnostic blend via learned softmax weights
        advantage_signals = np.array([
            projection_strength,           # signal 0: gate (should I act?)
            confidence_support,            # signal 1: value-head certainty
            value_support * floor_support, # signal 2: positive-outlook × floor safety
        ])
        advantage_strength = float(np.clip(np.dot(adv_weights, advantage_signals), 0.0, 1.0))

        # --- ACRP-v3: Separated gate + deployment scale ---
        # FIX (Flaw 3): gate_prob is the hard should-I-act gate.
        # deployment_scale_val is the continuous sizing signal.
        # Previously both were merged into deployment_scale, causing
        # competing gradients.  Now gate decides IF, scale decides HOW MUCH.
        # Gate threshold: only deploy when gate_prob > 0.5 (learned gate).
        gate_pass = bool(gate_prob >= 0.50)
        policy_scale = deployment_scale_val if gate_pass else 0.0
        delta = float(np.clip(deployment_scale_val * raw_delta, -self.delta_limit, self.delta_limit)) if gate_pass else 0.0
        target_exposure = float(np.clip(base_exposure + delta, -1.0, 1.0))

        clipped_mix = np.clip(expert_mix, 1e-6, 1.0)
        mix_entropy = float(-np.sum(clipped_mix * np.log(clipped_mix)) / math.log(len(TIER2_INTERNAL_EXPERTS)))
        components = {
            "selected_value_prediction": uplift_mean,
            "value_prediction": uplift_mean,
            "uplift_value_prediction": uplift_mean,
            "selected_value_lcb": value_lcb,
            "selected_value_ucb": value_ucb,
            "nav_preservation_prediction": nav_pred,
            "nav_prediction": nav_pred,
            "nav_prediction_std": nav_sigma,
            "nav_preservation_certified_lcb": nav_lcb,
            "nav_preservation_lcb": nav_lcb,
            "return_floor_prediction": floor_pred,
            "return_floor_prediction_std": floor_sigma,
            "return_floor_certified_lcb": floor_lcb,
            "return_floor_lcb": floor_lcb,
            "tail_risk_prediction": tail_risk,
            "factor_opportunity": factor_pred,
            "conformal_residual_risk": float(np.clip(controller_sigma / max(self.value_target_scale, 1e-6), 0.0, 1.0)),
            "projection_strength": projection_strength,
            "overlay_strength": projection_strength,
            "controller_confidence": confidence_support,
            "route_uncertainty_quality": route_uncertainty_quality,
            "forecast_trust": forecast_trust,
            "direction_accuracy": direction_accuracy,
            "direction_signal": signal,
            "magnitude_prediction": float(np.clip(abs(delta) / max(self.delta_limit, 1e-6), 0.0, 1.0)),
            "gate_head_probability": gate_prob,
            "value_support": value_support,
            "floor_support": floor_support,
            "risk_support": risk_support,
            "confidence_support": confidence_support,
            "policy_scale": policy_scale,
            "dominant_expert_weight": float(np.max(expert_mix)),
            "expert_weight_entropy": mix_entropy,
            "internal_base_weight": float(expert_mix[0]),
            "internal_trend_weight": float(expert_mix[1]),
            "internal_reversion_weight": float(expert_mix[2]),
            "internal_defensive_weight": float(expert_mix[3]),
            "full_value_prediction": uplift_mean,
            "ablated_value_prediction": 0.0,
            "learned_gate_score": float(policy_scale),
            "learned_gate_head_probability": float(np.clip(gate_prob, 0.0, 1.0)),
            "quality_anchor": quality,
            "trust_anchor": forecast_trust,
            "acrp_learned_trust_radius": trust_radius,
            "acrp_advantage_strength": advantage_strength,
            "acrp_attn_gate_weight": float(adv_weights[0]),
            "acrp_attn_conf_weight": float(adv_weights[1]),
            "acrp_attn_outlook_weight": float(adv_weights[2]),
            "acrp_da_learned_scalar": da_learned,
            "primary_quality_gap": 0.0,
            "primary_risk_gap": 0.0,
            "primary_vs_consensus_gap": 0.0,
            "primary_leadership": 0.0,
        }
        return {
            "selected_delta": delta,
            "selected_intervene_probability": float(policy_scale),
            "selected_probability": float(policy_scale),
            "selected_gate_probability": float(np.clip(gate_prob, 0.0, 1.0)),
            "selected_value_mean": uplift_mean,
            "selected_value_std": sigma,
            "selected_value_lcb": value_lcb,
            "selected_value_ucb": value_ucb,
            "target_exposure": target_exposure,
            "components": components,
        }


class Tier2PolicyImprovementTrainer:
    def __init__(
        self,
        *,
        model: Any,
        obs_dim: int,
        forecast_dim: int,
        policy_family: Optional[str] = None,
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-4,
        seed: int = 42,
        replay_batch_size: int = 64,
        train_epochs: int = 2,
        delta_loss_weight: float = 1.0,
        decision_weight: float = 1.0,
        confidence_weight: float = 0.0,
        quality_anchor_weight: float = 0.0,
        trust_anchor_weight: float = 0.0,
        gate_calibration_weight: float = 0.10,
        expert_mix_loss_weight: float = 0.0,
        expert_winner_loss_weight: float = 0.0,
        expert_temperature: float = 0.35,
        gate_top_fraction: float = 0.30,
        value_loss_weight: float = 0.75,
        quantile_loss_weight: float = 0.35,
        intervene_loss_weight: float = 0.25,
        value_target_scale: float = 0.01,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        return_floor_margin: float = 5e-5,
        return_floor_interval_penalty: float = 0.10,
        nav_interval_penalty: float = 0.05,
        decision_downside_weight: float = 0.75,
        decision_drawdown_weight: float = 0.60,
        decision_vol_weight: float = 0.75,
        decision_stability_penalty: float = 0.18,
        delta_limit: float = 0.10,
        runtime_min_abs_delta: float = 0.003,
        max_position_size: float = 0.35,
        transaction_cost_bps: float = 0.5,
        transaction_fixed_cost_dkk: float = 0.0,
        init_budget_dkk: float = 1.0,
        factor_loss_weight: float = 0.0,
        tail_risk_budget: float = 0.10,
        nav_actor_weight: float = 0.20,
        nav_lcb_actor_weight: float = 0.25,
        tail_loss_weight: float = 0.50,
        nav_head_scale: float = 3.0,
        return_floor_head_scale: float = 4.0,
        **_: Any,
    ) -> None:
        self.model = model
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)
        self.policy_family = str(policy_family or "forecast_guided_meta_baseline")
        self.learning_rate = float(max(learning_rate, 1e-6))
        self.l2_reg = float(max(l2_reg, 0.0))
        self.seed = int(seed)
        self.replay_batch_size = int(max(replay_batch_size, 8))
        self.train_epochs = int(max(train_epochs, 1))
        self.delta_loss_weight = float(max(delta_loss_weight, 0.0))
        self.decision_weight = float(max(decision_weight, 0.0))
        self.confidence_weight = float(max(confidence_weight, 0.0))
        self.quality_anchor_weight = float(max(quality_anchor_weight, 0.0))
        self.trust_anchor_weight = float(max(trust_anchor_weight, 0.0))
        self.gate_calibration_weight = float(max(gate_calibration_weight, 0.0))
        self.expert_mix_loss_weight = float(max(expert_mix_loss_weight, 0.0))
        self.expert_winner_loss_weight = float(max(expert_winner_loss_weight, 0.0))
        self.expert_temperature = float(max(expert_temperature, 1e-3))
        self.gate_top_fraction = float(np.clip(gate_top_fraction, 0.05, 0.95))
        self.value_loss_weight = float(max(value_loss_weight, 0.0))
        self.quantile_loss_weight = float(max(quantile_loss_weight, 0.0))
        self.intervene_loss_weight = float(max(intervene_loss_weight, 0.0))
        self.value_target_scale = float(max(value_target_scale, 1e-6))
        self.lower_quantile = float(np.clip(lower_quantile, 0.01, 0.49))
        self.upper_quantile = float(np.clip(upper_quantile, 0.51, 0.99))
        self.return_floor_margin = float(return_floor_margin)
        self.return_floor_interval_penalty = float(max(return_floor_interval_penalty, 0.0))
        self.nav_interval_penalty = float(max(nav_interval_penalty, 0.0))
        self.decision_downside_weight = float(max(decision_downside_weight, 0.0))
        self.decision_drawdown_weight = float(max(decision_drawdown_weight, 0.0))
        self.decision_vol_weight = float(max(decision_vol_weight, 0.0))
        self.decision_stability_penalty = float(max(decision_stability_penalty, 0.0))
        self.delta_limit = float(max(delta_limit, 1e-6))
        self.runtime_min_abs_delta = float(max(runtime_min_abs_delta, 0.0))
        self.max_position_size = float(max(max_position_size, 1e-6))
        self.transaction_cost_bps = float(max(transaction_cost_bps, 0.0))
        self.transaction_fixed_cost_dkk = float(max(transaction_fixed_cost_dkk, 0.0))
        self.init_budget_dkk = float(max(init_budget_dkk, 1.0))
        self.factor_loss_weight = float(max(factor_loss_weight, 0.0))
        self.tail_risk_budget = float(np.clip(tail_risk_budget, 0.0, 1.0))
        self.nav_actor_weight = float(max(nav_actor_weight, 0.0))
        self.nav_lcb_actor_weight = float(max(nav_lcb_actor_weight, 0.0))
        self.tail_loss_weight = float(max(tail_loss_weight, 0.0))
        self.nav_head_scale = float(max(nav_head_scale, 1e-6))
        self.return_floor_head_scale = float(max(return_floor_head_scale, 1e-6))

        # Loss scheduling parameters
        self.loss_schedule_start_step = 1000  # Start scheduling after this many steps
        self.loss_schedule_end_step = 10000  # Full schedule after this many steps
        self.auxiliary_loss_decay = 0.5  # How much to decay auxiliary losses

        self._tf = _get_tf()
        self._optimizer = None
        self._compiled_forward_train = None
        if self._tf is not None and self.model is not None:
            self._optimizer = self._tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            try:
                self._compiled_forward_train = self._tf.function(
                    lambda obs, feat: self.model([obs, feat], training=True),
                    reduce_retracing=True,
                )
            except Exception:
                self._compiled_forward_train = None

        random.seed(self.seed)
        np.random.seed(self.seed)

    def _decode_feature_batch(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        feat = np.asarray(features, dtype=np.float32)
        latest = feat[:, -TIER2_VALUE_MEMORY_CHANNELS:]
        return {
            "proposed_exposure": feat[:, 0],
            "realized_volatility_regime": feat[:, 4],
            "local_quality": feat[:, 5],
            "local_quality_delta": feat[:, 6],
            "local_drawdown": feat[:, 7],
            "ann_signal": latest[:, 0],
            "ann_quality": latest[:, 1],
            "ann_risk": latest[:, 2],
            "consensus": latest[:, 3],
            "disagreement": latest[:, 4],
            "imbalance": latest[:, 5],
            "ann_return_signal": latest[:, 6],
            "ann_abs_return": latest[:, 7],
            "ann_direction_accuracy": latest[:, 10],
            "price_return_signal": latest[:, 17],
            "ann_direction_margin": latest[:, 9],
            "ann_latent_norm": latest[:, 15],
        }

    def _build_expert_regime_targets(
        self,
        *,
        ann_signal: Any,
        ann_quality: Any,
        ann_direction_accuracy: Any,
        ann_risk: Any,
        disagreement: Any,
        volatility: Any,
        drawdown: Any,
        quality_delta: Any,
        price_return_signal: Any,
        uplift_target: Any,
        tail_target: Any,
    ) -> Tuple[Any, Any]:
        tf = self._tf
        if tf is None:
            raise RuntimeError("TensorFlow unavailable for expert targets")

        temperature = tf.constant(self.expert_temperature, dtype=tf.float32)
        signal_strength = tf.abs(ann_signal)
        quality_strength = 0.35 + 0.35 * ann_quality + 0.30 * ann_direction_accuracy
        safety_strength = 1.0 - 0.50 * ann_risk
        disagreement_strength = tf.clip_by_value(disagreement, 0.0, 1.0)
        trend_alignment = tf.nn.relu(ann_signal * price_return_signal)
        reversion_alignment = tf.nn.relu(-ann_signal * price_return_signal)
        positive_uplift = tf.nn.relu(uplift_target / tf.maximum(self.value_target_scale, 1e-6))
        negative_uplift = tf.nn.relu(-uplift_target / tf.maximum(self.value_target_scale, 1e-6))

        trend_score = (
            (0.20 + signal_strength)
            * quality_strength
            * (1.0 - disagreement_strength)
            * (0.60 + 0.40 * trend_alignment)
            * safety_strength
        )
        reversion_score = (
            (0.15 + signal_strength)
            * (0.45 + 0.55 * disagreement_strength)
            * (0.60 + 0.40 * reversion_alignment)
            * (0.50 + 0.50 * tf.nn.relu(-quality_delta))
            * safety_strength
        )
        defensive_score = (
            tf.maximum(tf.maximum(ann_risk, volatility), tf.maximum(drawdown, tail_target))
            * (1.0 + negative_uplift)
        )
        base_score = (
            0.25
            + 0.45 * (1.0 - signal_strength)
            + 0.20 * (1.0 - disagreement_strength)
            + 0.10 * tf.nn.relu(quality_delta)
            + 0.10 * positive_uplift
        )

        logits = tf.concat(
            [base_score, trend_score, reversion_score, defensive_score],
            axis=1,
        )
        target_probs = tf.nn.softmax(logits / temperature, axis=1)
        winner = tf.argmax(target_probs, axis=1, output_type=tf.int32)
        return target_probs, winner

    def _quantile_loss(self, target: Any, pred: Any, quantile: float) -> Any:
        tf = self._tf
        err = tf.cast(target, tf.float32) - tf.cast(pred, tf.float32)
        q = tf.cast(quantile, tf.float32)
        return tf.reduce_mean(tf.maximum(q * err, (q - 1.0) * err))

    def _batch_from_samples(self, samples: Sequence[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
        if not samples:
            return None
        obs = []
        features = []
        realized = []
        exposure = []
        for sample in samples:
            obs_i = np.asarray(sample.get("obs", []), dtype=np.float32).flatten()
            feat_i = np.asarray(sample.get("tier2_features", []), dtype=np.float32).flatten()
            if obs_i.size != self.obs_dim or feat_i.size != self.forecast_dim:
                continue
            obs.append(obs_i)
            features.append(feat_i)
            realized.append(float(sample.get("realized_return", 0.0)))
            exposure_val = sample.get("executed_exposure", None)
            exposure.append(float(exposure_val) if exposure_val is not None else float(feat_i[0]))
        if not obs:
            return None
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "features": np.asarray(features, dtype=np.float32),
            "realized_return": np.asarray(realized, dtype=np.float32).reshape(-1),
            "executed_exposure": np.asarray(exposure, dtype=np.float32).reshape(-1),
        }

    def train_continuous(
        self,
        current_rollout: Sequence[Dict[str, Any]],
        _unused: Any,
        replay_buffer: Optional[Tier2PolicyImprovementBuffer],
        training_step: int = 0,
    ) -> Dict[str, Any]:
        if self._tf is None or self.model is None or self._optimizer is None:
            return {"status": "no_tf"}

        live_samples = list(current_rollout)
        if len(live_samples) < 8:
            return {"status": "skip_short_rollout"}

        replay_samples: List[Dict[str, Any]] = []
        if replay_buffer is not None:
            replay_samples = replay_buffer.sample_replay(self.replay_batch_size)

        if replay_samples:
            target_batch_size = max(self.replay_batch_size, 8)
            live_target = min(len(live_samples), max(target_batch_size // 2, 8))
            replay_target = min(len(replay_samples), max(target_batch_size - live_target, 0))
            selected_live = live_samples[-live_target:]
            selected_replay = replay_samples[:replay_target]
            batch_samples = selected_live + selected_replay
            if len(batch_samples) < min(target_batch_size, len(live_samples) + len(replay_samples)):
                deficit = min(target_batch_size, len(live_samples) + len(replay_samples)) - len(batch_samples)
                if deficit > 0 and len(live_samples) > live_target:
                    batch_samples = live_samples[-(live_target + deficit):] + selected_replay
        else:
            batch_samples = live_samples[-max(self.replay_batch_size, 8):]

        batch = self._batch_from_samples(batch_samples)
        if batch is None:
            return {"status": "skip_bad_batch"}

        tf = self._tf
        obs = batch["obs"]
        features = batch["features"]
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        feat_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        feature_dec = self._decode_feature_batch(features)
        proposed_np = np.asarray(feature_dec["proposed_exposure"], dtype=np.float32).reshape(-1, 1)
        quality_np = np.clip(np.asarray(feature_dec["ann_quality"], dtype=np.float32), 0.0, 1.0).reshape(-1, 1)
        risk_np = np.clip(np.asarray(feature_dec["ann_risk"], dtype=np.float32), 0.0, 1.0).reshape(-1, 1)
        vol_np = np.clip(np.asarray(feature_dec["realized_volatility_regime"], dtype=np.float32), 0.0, 1.0).reshape(-1, 1)
        drawdown_np = np.clip(np.asarray(feature_dec["local_drawdown"], dtype=np.float32), 0.0, 1.0).reshape(-1, 1)
        disagreement_np = np.clip(np.asarray(feature_dec["disagreement"], dtype=np.float32), 0.0, 1.0).reshape(-1, 1)
        price_return_np = np.asarray(feature_dec["price_return_signal"], dtype=np.float32).reshape(-1, 1)
        direction_acc_np = np.clip(np.asarray(feature_dec["ann_direction_accuracy"], dtype=np.float32), 0.0, 1.0).reshape(-1, 1)
        quality_delta_np = np.asarray(feature_dec["local_quality_delta"], dtype=np.float32).reshape(-1, 1)
        realized_np = np.asarray(batch["realized_return"], dtype=np.float32).reshape(-1, 1)
        executed_np = np.asarray(batch["executed_exposure"], dtype=np.float32).reshape(-1, 1)
        executed_np = np.where(np.isfinite(executed_np), executed_np, proposed_np)

        proposed_tensor = tf.convert_to_tensor(proposed_np, dtype=tf.float32)
        realized_tensor = tf.convert_to_tensor(realized_np, dtype=tf.float32)
        executed_tensor = tf.convert_to_tensor(executed_np, dtype=tf.float32)
        risk_tensor = tf.convert_to_tensor(risk_np, dtype=tf.float32)
        vol_tensor = tf.convert_to_tensor(vol_np, dtype=tf.float32)
        drawdown_tensor = tf.convert_to_tensor(drawdown_np, dtype=tf.float32)
        disagreement_tensor = tf.convert_to_tensor(disagreement_np, dtype=tf.float32)
        price_return_tensor = tf.convert_to_tensor(price_return_np, dtype=tf.float32)
        direction_acc_tensor = tf.convert_to_tensor(direction_acc_np, dtype=tf.float32)
        quality_delta_tensor = tf.convert_to_tensor(quality_delta_np, dtype=tf.float32)
        value_scale = tf.constant(self.value_target_scale, dtype=tf.float32)
        trade_cost_rate = tf.constant(
            (self.transaction_cost_bps / 10_000.0) * self.max_position_size,
            dtype=tf.float32,
        )
        fixed_cost_rate = tf.constant(
            self.transaction_fixed_cost_dkk / self.init_budget_dkk,
            dtype=tf.float32,
        )

        bce = tf.keras.losses.BinaryCrossentropy()
        huber = tf.keras.losses.Huber(delta=0.5)
        last_metrics: Dict[str, float] = {}

        for _ in range(self.train_epochs):
            with tf.GradientTape() as tape:
                if self._compiled_forward_train is not None:
                    preds = self._compiled_forward_train(obs_tensor, feat_tensor)
                else:
                    preds = self.model([obs_tensor, feat_tensor], training=True)
                delta_pred = preds["delta_raw"] * self.delta_limit
                gate_pred = preds["gate_prob"]
                uplift_pred = preds["uplift_mean"] * self.value_target_scale
                sigma_pred = preds["uplift_sigma"] * self.value_target_scale + 1e-6
                nav_pred = preds["nav_prediction"] * self.value_target_scale * self.nav_head_scale
                nav_sigma_pred = preds["nav_sigma"] * self.value_target_scale * self.nav_head_scale + 1e-6
                floor_pred = preds["return_floor_prediction"] * self.value_target_scale * self.return_floor_head_scale
                floor_sigma_pred = preds["return_floor_sigma"] * self.value_target_scale * self.return_floor_head_scale + 1e-6
                tail_pred = preds["tail_risk"]
                factor_pred = preds["factor_opportunity"]
                mix_pred = preds["expert_mix"]
                trust_radius_pred = preds["trust_radius"]
                adv_attn_pred = preds["adv_attn_weights"]  # (batch, 3) softmax
                da_learned_pred = preds["da_learned_scalar"]  # (batch, 1) from DA encoder
                controller_sigma_pred = tf.maximum(
                    sigma_pred,
                    tf.maximum(
                        nav_sigma_pred / self.nav_head_scale,
                        floor_sigma_pred / self.return_floor_head_scale,
                    ),
                )
                conf_pred = 1.0 / (1.0 + controller_sigma_pred / self.value_target_scale)
                uplift_lcb_pred = uplift_pred - 1.2816 * sigma_pred
                floor_lcb_pred_for_scale = floor_pred - 0.75 * floor_sigma_pred

                # --- ACRP-v3: End-to-end learned deployment ---
                # The deployment decision now comes DIRECTLY from the model's
                # deployment_scale head — no hand-coded formula.  The
                # auxiliary heads (trust, adv_attn, da_learned) still receive
                # their own supervised losses below for representation
                # shaping, but deployment_scale is the single learned surface
                # that drives the effective delta.
                deployment_scale = preds["deployment_scale"]

                effective_delta = deployment_scale * delta_pred
                effective_exposure = tf.clip_by_value(proposed_tensor + effective_delta, -1.0, 1.0)
                effective_delta = effective_exposure - proposed_tensor
                base_exposure = tf.clip_by_value(proposed_tensor, -1.0, 1.0)

                gross_return = effective_exposure * realized_tensor
                base_gross_return = base_exposure * realized_tensor

                trade_cost = trade_cost_rate * tf.abs(effective_delta)
                if self.transaction_fixed_cost_dkk > 0.0:
                    trade_cost += fixed_cost_rate * tf.sigmoid(
                        (tf.abs(effective_delta) - self.runtime_min_abs_delta) * 64.0
                    )

                downside_penalty = self.decision_downside_weight * tf.nn.relu(-gross_return)
                base_downside_penalty = self.decision_downside_weight * tf.nn.relu(-base_gross_return)
                # FIX (Flaw 2): Remove value_scale multiplier from vol/draw
                # penalties.  Previously these were ~11x smaller than the
                # return signal, providing no effective cost for increasing
                # exposure.  Now penalties are in the same units as returns.
                vol_penalty = self.decision_vol_weight * vol_tensor * tf.abs(effective_exposure)
                base_vol_penalty = self.decision_vol_weight * vol_tensor * tf.abs(base_exposure)
                draw_penalty = self.decision_drawdown_weight * drawdown_tensor * tf.abs(effective_exposure)
                base_draw_penalty = self.decision_drawdown_weight * drawdown_tensor * tf.abs(base_exposure)

                # FIX (Flaw 6): Explicit exposure cost — proportional penalty
                # for the absolute delta size.  This creates a direct cost
                # for increasing exposure that the model cannot circumvent.
                # Without this, the model can always find individually
                # "good" trades that inflate cumulative exposure.
                exposure_cost_weight = tf.constant(0.002, dtype=tf.float32)
                exposure_cost = exposure_cost_weight * tf.abs(effective_delta)

                # Uncertainty penalty: penalise large positions when the model
                # itself is unsure (high sigma).  controller_sigma_pred aggregates
                # the worst-case sigma across the value, NAV, and floor heads.
                uncertainty_penalty = (
                    self.decision_stability_penalty
                    * controller_sigma_pred
                    / tf.maximum(value_scale, 1e-6)
                    * tf.abs(effective_exposure)
                )

                direct_objective = (
                    gross_return
                    - trade_cost
                    - downside_penalty
                    - vol_penalty
                    - draw_penalty
                    - exposure_cost
                )
                base_objective = (
                    base_gross_return
                    - base_downside_penalty
                    - base_vol_penalty
                    - base_draw_penalty
                )
                uplift_realized = direct_objective - base_objective

                uplift_target = tf.stop_gradient(
                    tf.clip_by_value(uplift_realized, -6.0 * value_scale, 6.0 * value_scale)
                )
                nav_target = tf.stop_gradient(
                    tf.clip_by_value(gross_return - trade_cost, -8.0 * value_scale, 8.0 * value_scale)
                )
                floor_target = tf.stop_gradient(
                    tf.clip_by_value(gross_return - trade_cost - downside_penalty, -8.0 * value_scale, 8.0 * value_scale)
                )
                # Denominator calibrated for post-Fix-2 penalty magnitudes.
                # Penalties are now in direct return units (~0.01-0.08 typical).
                # 4*value_scale=0.04 was correct when vol/draw had ×value_scale;
                # now we need ~0.20 to keep tail_target in a useful [0.05, 0.5] range.
                tail_penalty_scale = tf.constant(0.20, dtype=tf.float32)
                tail_target = tf.stop_gradient(
                    tf.clip_by_value(
                        (downside_penalty + vol_penalty + draw_penalty + uncertainty_penalty)
                        / tf.maximum(tail_penalty_scale, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                ann_signal_tensor = tf.convert_to_tensor(feature_dec["ann_signal"].reshape(-1, 1), dtype=tf.float32)
                direction_correct = tf.cast((ann_signal_tensor * realized_tensor) > 0.0, tf.float32)
                gate_teacher_score = (
                    uplift_target / tf.maximum(value_scale, 1e-6)
                    + 0.20 * direction_acc_tensor
                    + 0.15 * direction_correct * direction_acc_tensor
                    - 0.20 * tail_target
                )
                flat_teacher_score = tf.reshape(gate_teacher_score, [-1])
                batch_size = tf.shape(flat_teacher_score)[0]
                top_k = tf.maximum(1, tf.cast(tf.round(self.gate_top_fraction * tf.cast(batch_size, tf.float32)), tf.int32))
                teacher_cutoff = tf.math.top_k(flat_teacher_score, k=top_k, sorted=True).values[-1]
                gate_target = tf.stop_gradient(
                    tf.cast(
                        tf.logical_and(gate_teacher_score >= teacher_cutoff, gate_teacher_score > 0.0),
                        tf.float32,
                    )
                )
                confidence_target = tf.stop_gradient(
                    tf.clip_by_value(
                        tf.sigmoid((uplift_target - tail_target * value_scale) / tf.maximum(value_scale, 1e-6)),
                        0.0,
                        1.0,
                    )
                )
                factor_target = tf.stop_gradient(
                    tf.clip_by_value(
                        tf.nn.relu(uplift_target) / tf.maximum(2.0 * value_scale, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                expert_target_probs, expert_winner = self._build_expert_regime_targets(
                    ann_signal=ann_signal_tensor,
                    ann_quality=tf.convert_to_tensor(quality_np, dtype=tf.float32),
                    ann_direction_accuracy=direction_acc_tensor,
                    ann_risk=risk_tensor,
                    disagreement=disagreement_tensor,
                    volatility=vol_tensor,
                    drawdown=drawdown_tensor,
                    quality_delta=quality_delta_tensor,
                    price_return_signal=price_return_tensor,
                    uplift_target=uplift_target,
                    tail_target=tail_target,
                )
                quality_anchor_target = tf.stop_gradient(
                    tf.clip_by_value(
                        0.50 * tf.convert_to_tensor(quality_np, dtype=tf.float32)
                        + 0.25 * (1.0 - risk_tensor)
                        + 0.25 * (1.0 - disagreement_tensor),
                        0.0,
                        1.0,
                    )
                )
                trust_anchor_target = tf.stop_gradient(
                    tf.clip_by_value(
                        0.35 * quality_anchor_target
                        + 0.25 * tf.sigmoid(nav_target / tf.maximum(value_scale, 1e-6))
                        + 0.25 * tf.sigmoid(floor_target / tf.maximum(value_scale, 1e-6))
                        + 0.15 * (1.0 - tail_target),
                        0.0,
                        1.0,
                    )
                )

                uplift_mean_batch = tf.reduce_mean(uplift_realized)
                uplift_std_batch = tf.sqrt(tf.reduce_mean(tf.square(uplift_realized - uplift_mean_batch)) + 1e-6)
                sharpe_surrogate = uplift_mean_batch / (uplift_std_batch + 0.50 * value_scale)

                # --- ACRP: Pure Sharpe actor loss (FIXED) ---
                # FIX (Flaw 1): Removed raw uplift_mean/value_scale and
                # da_bonus/value_scale terms.  These dominated the loss
                # (~10x Sharpe), rewarding larger positions instead of
                # risk-adjusted return.  Now the actor ONLY maximizes the
                # Sharpe ratio of the residual (overlay return / overlay vol).
                # The DA bonus is kept but at Sharpe-proportional scale.
                da_bonus = tf.reduce_mean(da_learned_pred * direction_correct * uplift_realized)
                da_bonus_sharpe = da_bonus / (uplift_std_batch + 0.50 * value_scale)
                actor_loss = -(
                    sharpe_surrogate
                    + 0.25 * da_bonus_sharpe
                )
                delta_loss = tf.reduce_mean(tf.square(effective_delta))
                # FIX (Flaw 3): Separate gate from deployment scale.
                # Previously gate_loss trained deployment_scale as both a
                # binary gate AND a continuous scale, creating competing
                # gradients (actor→maximize, gate→binary).  Now:
                # - gate_loss trains gate_prob (the actual should-I-act head)
                # - deployment_scale is trained only by actor_loss (sizing)
                # - gate_calibration_loss is removed (was a dead signal)
                # At runtime, gate_prob will serve as the hard gate and
                # deployment_scale only controls magnitude.
                gate_loss = bce(gate_target, gate_pred)
                gate_calibration_loss = tf.constant(0.0, dtype=tf.float32)

                # --- ACRP-v2: DA-encoder-weighted value supervision ---
                # Uses the learned DA scalar from the encoder instead of the
                # raw aggregate.  The encoder sees the full 8-step DA trajectory
                # so it produces a more nuanced quality signal.
                da_weight = tf.stop_gradient(0.50 + 0.50 * da_learned_pred)
                uplift_residual = uplift_target - uplift_pred
                value_loss = tf.reduce_mean(da_weight * ((tf.square(uplift_residual) / sigma_pred) + tf.math.log(sigma_pred)))
                q_lo_loss = self._quantile_loss(uplift_target, uplift_pred - sigma_pred, self.lower_quantile)
                q_hi_loss = self._quantile_loss(uplift_target, uplift_pred + sigma_pred, self.upper_quantile)
                quantile_loss = 0.5 * (q_lo_loss + q_hi_loss)
                nav_residual = nav_target - nav_pred
                nav_nll = tf.reduce_mean((tf.square(nav_residual) / nav_sigma_pred) + tf.math.log(nav_sigma_pred))
                nav_lcb_pred = nav_pred - 1.2816 * nav_sigma_pred
                nav_loss = huber(nav_target, nav_pred) + nav_nll + self.nav_interval_penalty * tf.reduce_mean(
                    tf.nn.relu(-nav_lcb_pred)
                )
                floor_residual = floor_target - floor_pred
                floor_nll = tf.reduce_mean((tf.square(floor_residual) / floor_sigma_pred) + tf.math.log(floor_sigma_pred))
                floor_lcb_pred = floor_lcb_pred_for_scale
                floor_loss = huber(floor_target, floor_pred) + floor_nll + self.return_floor_interval_penalty * tf.reduce_mean(
                    tf.nn.relu(self.return_floor_margin - floor_lcb_pred)
                )
                tail_loss = huber(tail_target, tail_pred)
                conf_loss = huber(confidence_target, conf_pred) if self.confidence_weight > 0.0 else tf.constant(0.0, dtype=tf.float32)

                mix_pred_clipped = tf.clip_by_value(mix_pred, 1e-6, 1.0)
                target_probs_clipped = tf.clip_by_value(expert_target_probs, 1e-6, 1.0)
                mix_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        target_probs_clipped * (tf.math.log(target_probs_clipped) - tf.math.log(mix_pred_clipped)),
                        axis=1,
                    )
                )
                winner_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(expert_winner, mix_pred_clipped)
                )
                quality_anchor_loss = huber(quality_anchor_target, gate_pred)
                trust_anchor_loss = huber(trust_anchor_target, gate_pred)

                factor_loss = huber(factor_target, factor_pred) if self.factor_loss_weight > 0.0 else tf.constant(0.0, dtype=tf.float32)

                # --- ACRP-v2: Trust meta-loss ---
                # The trust head should produce high trust when the residual
                # actually improved returns, and low trust otherwise.
                # Ground truth: did trusting the forecast pay off?
                # trust_target = sigmoid(uplift_realized / scale) ∈ [0,1]
                # positive uplift → trust_target ≈ 1, negative → ≈ 0
                trust_meta_target = tf.stop_gradient(
                    tf.sigmoid(uplift_realized / tf.maximum(value_scale, 1e-6))
                )
                trust_meta_loss = tf.reduce_mean(tf.square(trust_radius_pred - trust_meta_target))

                # --- ACRP-v2: Attention diversity loss ---
                # Entropy bonus prevents the adaptive weights from collapsing
                # to a single signal.  max entropy = log(3) ≈ 1.099
                attn_entropy = -tf.reduce_mean(
                    tf.reduce_sum(adv_attn_pred * tf.math.log(adv_attn_pred + 1e-8), axis=-1)
                )
                # We want to MAXIMIZE entropy → minimize negative entropy
                attn_diversity_loss = -attn_entropy

                # --- ACRP-v2: DA encoder supervision (FIXED) ---
                # FIX (Flaw 6): Changed from raw DA reconstruction
                # (da_learned ≈ direction_accuracy → collapsed to ~1.0)
                # to a contrastive target: was the forecast USEFULLY correct
                # in this specific context?  Target = direction_correct
                # (binary: did the ANN predict the right sign?).
                # This forces the encoder to predict whether the DA
                # trajectory indicates reliable forecasting NOW, not just
                # report the historical average.
                da_encoder_loss = tf.reduce_mean(tf.square(da_learned_pred - direction_correct))

                l2_penalty = tf.add_n(
                    [tf.nn.l2_loss(var) for var in self.model.trainable_variables if "bias" not in var.name]
                ) * self.l2_reg

                # Loss scheduling: decay auxiliary losses over training
                schedule_progress = min(1.0, max(0.0, (training_step - self.loss_schedule_start_step) /
                                                (self.loss_schedule_end_step - self.loss_schedule_start_step)))
                auxiliary_decay = 1.0 - schedule_progress * (1.0 - self.auxiliary_loss_decay)

                # Group losses for clarity and scheduling
                actor_losses = (
                    self.decision_weight * actor_loss
                    + self.intervene_loss_weight * self.decision_weight * gate_loss
                )

                value_risk_losses = (
                    self.delta_loss_weight * delta_loss
                    + self.value_loss_weight * value_loss
                    + self.quantile_loss_weight * quantile_loss
                    + self.nav_actor_weight * nav_loss
                    + self.nav_lcb_actor_weight * floor_loss
                    + self.tail_loss_weight * tail_loss
                )

                auxiliary_losses = (
                    self.expert_mix_loss_weight * mix_loss
                    + self.expert_winner_loss_weight * winner_loss
                    + self.confidence_weight * conf_loss * auxiliary_decay
                    + self.quality_anchor_weight * quality_anchor_loss * auxiliary_decay
                    + self.trust_anchor_weight * trust_anchor_loss * auxiliary_decay
                    + self.gate_calibration_weight * gate_calibration_loss
                    + self.factor_loss_weight * factor_loss * auxiliary_decay
                    + 0.15 * trust_meta_loss
                    + 0.02 * attn_diversity_loss * auxiliary_decay
                    + 0.10 * da_encoder_loss
                )

                regularization_losses = l2_penalty

                total_loss = actor_losses + value_risk_losses + auxiliary_losses + regularization_losses

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            valid_pairs = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
            if valid_pairs:
                valid_grads, valid_vars = zip(*valid_pairs)
                clipped_grads, _ = tf.clip_by_global_norm(list(valid_grads), 1.0)
                self._optimizer.apply_gradients(zip(clipped_grads, valid_vars))

            executed_gross = executed_tensor * realized_tensor
            executed_downside = self.decision_downside_weight * tf.nn.relu(-executed_gross)
            executed_vol_penalty = self.decision_vol_weight * vol_tensor * tf.abs(executed_tensor)
            executed_draw_penalty = self.decision_drawdown_weight * drawdown_tensor * tf.abs(executed_tensor)
            executed_objective = executed_gross - executed_downside - executed_vol_penalty - executed_draw_penalty
            mix_np = np.asarray(mix_pred.numpy(), dtype=np.float32)
            effective_delta_np = np.asarray(effective_delta.numpy(), dtype=np.float32)
            uplift_target_np = np.asarray(uplift_target.numpy(), dtype=np.float32)
            gate_target_np = np.asarray(gate_target.numpy(), dtype=np.float32)
            trade_cost_np = np.asarray(trade_cost.numpy(), dtype=np.float32)
            forecast_trust_np = np.clip(quality_np * (1.0 - risk_np), 0.0, 1.0)
            expert_target_np = np.asarray(expert_target_probs.numpy(), dtype=np.float32)
            last_metrics = {
                "loss": float(total_loss.numpy()),
                "actor_loss": float(actor_loss.numpy()),
                "sharpe_surrogate": float(sharpe_surrogate.numpy()),
                "value_loss": float(value_loss.numpy()),
                "gate_probability_mean": float(np.mean(gate_pred.numpy())),
                "deployment_strength_mean": float(np.mean(deployment_scale.numpy())),
                "intervene_loss_weight": float(self.intervene_loss_weight),
                "gate_calibration_loss": float(gate_calibration_loss.numpy()),
                "action_active_prob": float(np.mean(np.abs(effective_delta_np) >= max(self.runtime_min_abs_delta, 1e-9))),
                "confidence_mean": float(np.mean(conf_pred.numpy())),
                "value_prediction_mean": float(np.mean(uplift_pred.numpy())),
                "uplift_prediction_mean": float(np.mean(uplift_pred.numpy())),
                "uplift_certified_lcb_mean": float(np.mean((uplift_pred - 1.2816 * sigma_pred).numpy())),
                "nav_prediction_mean": float(np.mean(nav_pred.numpy())),
                "nav_prediction_sigma_mean": float(np.mean(nav_sigma_pred.numpy())),
                "nav_certified_lcb_mean": float(np.mean((nav_pred - 1.2816 * nav_sigma_pred).numpy())),
                "return_floor_prediction_mean": float(np.mean(floor_pred.numpy())),
                "return_floor_prediction_sigma_mean": float(np.mean(floor_sigma_pred.numpy())),
                "return_floor_certified_lcb_mean": float(np.mean((floor_pred - 0.75 * floor_sigma_pred).numpy())),
                "tail_risk_prediction_mean": float(np.mean(tail_pred.numpy())),
                "factor_opportunity_mean": float(np.mean(factor_pred.numpy())),
                "return_constraint_lambda": float(max(np.mean(np.maximum(self.return_floor_margin - floor_pred.numpy(), 0.0)), 0.0)),
                "tail_risk_constraint_lambda": float(max(np.mean(np.maximum(tail_pred.numpy() - self.tail_risk_budget, 0.0)), 0.0)),
                "nav_return_delta_mean": float(np.mean(nav_pred.numpy() - uplift_pred.numpy())),
                "teacher_advantage_mean": float(np.mean(uplift_target_np)),
                "teacher_gate_target_mean": float(np.mean(gate_target_np)),
                "teacher_positive_rate": float(np.mean(uplift_target_np > 0.0)),
                "ann_quality_mean": float(np.mean(quality_np)),
                "forecast_trust_mean": float(np.mean(forecast_trust_np)),
                "base_internal_weight_mean": float(np.mean(mix_np[:, 0])),
                "trend_internal_weight_mean": float(np.mean(mix_np[:, 1])),
                "reversion_internal_weight_mean": float(np.mean(mix_np[:, 2])),
                "defensive_internal_weight_mean": float(np.mean(mix_np[:, 3])),
                "expert_weight_entropy_mean": float(
                    -np.mean(np.sum(np.clip(mix_np, 1e-6, 1.0) * np.log(np.clip(mix_np, 1e-6, 1.0)), axis=1))
                    / math.log(len(TIER2_INTERNAL_EXPERTS))
                ),
                "quality_anchor_loss": float(quality_anchor_loss.numpy()),
                "trust_anchor_loss": float(trust_anchor_loss.numpy()),
                "expert_mix_loss": float(mix_loss.numpy()),
                "expert_winner_loss": float(winner_loss.numpy()),
                "target_base_weight_mean": float(np.mean(expert_target_np[:, 0])),
                "target_trend_weight_mean": float(np.mean(expert_target_np[:, 1])),
                "target_reversion_weight_mean": float(np.mean(expert_target_np[:, 2])),
                "target_defensive_weight_mean": float(np.mean(expert_target_np[:, 3])),
                "trade_cost_mean": float(np.mean(trade_cost_np)),
                "acrp_trust_radius_mean": float(np.mean(trust_radius_pred.numpy())),
                "acrp_deployment_scale_mean": float(np.mean(deployment_scale.numpy())),
                "acrp_da_bonus": float(da_bonus.numpy()),
                "acrp_direction_accuracy_mean": float(np.mean(direction_acc_tensor.numpy())),
                "acrp_trust_meta_loss": float(trust_meta_loss.numpy()),
                "acrp_attn_diversity_loss": float(attn_diversity_loss.numpy()),
                "acrp_attn_gate_weight_mean": float(np.mean(adv_attn_pred.numpy()[:, 0])),
                "acrp_attn_conf_weight_mean": float(np.mean(adv_attn_pred.numpy()[:, 1])),
                "acrp_attn_outlook_weight_mean": float(np.mean(adv_attn_pred.numpy()[:, 2])),
                "acrp_da_encoder_loss": float(da_encoder_loss.numpy()),
                "acrp_da_learned_scalar_mean": float(np.mean(da_learned_pred.numpy())),
                "replay_loss": float(total_loss.numpy()),
                "action_target_mean": float(np.mean(effective_delta_np)),
                "decision_improvement_mean": float(np.mean(uplift_target_np)),
                "executed_objective_mean": float(np.mean(executed_objective.numpy())),
            }

        last_metrics.update(
            {
                "status": "ok",
            }
        )
        return last_metrics
