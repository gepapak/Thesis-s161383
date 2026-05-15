"""
Forecast-cache exposure prior for the investor agent.

This module converts the per-episode ANN price forecast cache into a
calibrated target exposure.  It is intentionally not an observation augmenter:
the investor policy keeps the same 12D observation contract and learns a
residual around this executable prior.
"""

from __future__ import annotations

import glob
import logging
import os
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


FORECAST_PRIOR_FEATURE_COLS = [
    "price_short_expert_ann_pred_return",
    "price_short_expert_ann_direction_prob",
    "price_short_expert_ann_direction_margin",
    "price_short_expert_ann_uncertainty",
    "price_short_expert_ann_quality",
]

FORECAST_PRIOR_OPTIONAL_COLS = [
    "price_short_expert_ann_latent_norm",
    "price_short_expert_ann_latent_0",
    "price_short_expert_ann_latent_1",
    "price_short_expert_ann_latent_2",
    "price_short_expert_ann_latent_3",
]

FORECAST_PRIOR_ALL_COLS = FORECAST_PRIOR_FEATURE_COLS + FORECAST_PRIOR_OPTIONAL_COLS


def load_forecast_prior_features(cache_dir: str, episode_num: int) -> Optional[np.ndarray]:
    """
    Load ANN forecast-cache columns for one episode.

    Returns an array with columns matching ``FORECAST_PRIOR_ALL_COLS``. Missing
    optional latent columns are zero-filled; missing required columns fail closed
    and return ``None``.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("[FORECAST_PRIOR] pandas is required to read forecast cache")
        return None

    ep_dir = os.path.join(str(cache_dir), f"episode_{int(episode_num)}")
    if not os.path.isdir(ep_dir):
        ep_dir = str(cache_dir)

    csv_paths = sorted(glob.glob(os.path.join(ep_dir, "precomputed_forecasts_*.csv")))
    csv_paths = [p for p in csv_paths if "_metadata" not in os.path.basename(p)]
    if not csv_paths:
        return None

    csv_path = csv_paths[0]
    try:
        header = pd.read_csv(csv_path, nrows=0)
        available = set(str(c) for c in header.columns)
        missing_required = [c for c in FORECAST_PRIOR_FEATURE_COLS if c not in available]
        if missing_required:
            logger.error(
                "[FORECAST_PRIOR] Cache '%s' missing required columns: %s",
                csv_path,
                missing_required,
            )
            return None

        read_cols = [c for c in FORECAST_PRIOR_ALL_COLS if c in available]
        df = pd.read_csv(csv_path, usecols=read_cols)
        for col in FORECAST_PRIOR_ALL_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[FORECAST_PRIOR_ALL_COLS]
    except Exception as e:
        logger.error("[FORECAST_PRIOR] Failed to read '%s': %s", csv_path, e)
        return None

    if df.empty:
        return None

    arr = df.to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        defaults = np.array(
            [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        bad = ~np.isfinite(arr)
        for j in range(arr.shape[1]):
            arr[bad[:, j], j] = defaults[j]
    return arr


class ConformalForecastPrior:
    """
    Stateful online conformal calibrator for ANN forecast-cache signals.

    At time ``t`` it first updates calibration from the forecast made at
    ``t - horizon_steps`` using the now-realized raw DKK price.  It then emits
    a conservative exposure prior for the current cache row.  The prior is
    bounded to [-max_abs_exposure, max_abs_exposure].
    """

    __slots__ = (
        "raw",
        "decision_freq",
        "horizon",
        "window_size",
        "min_samples",
        "hit_lcb_z",
        "residual_quantile",
        "default_residual",
        "edge_gain",
        "error_hurdle",
        "skill_power",
        "max_abs_exposure",
        "denom_floor",
        "vol_half_life_steps",
        "vol_target",
        "_hits",
        "_abs_errors",
        "_prices",
        "_calibrated",
        "_ema_logret2",
        "_last_price",
        "_last_t",
        "_last_signal",
    )

    def __init__(
        self,
        raw_features: Optional[np.ndarray],
        *,
        decision_freq: int = 6,
        horizon_steps: int = 6,
        window_size: int = 500,
        min_samples: int = 50,
        hit_lcb_z: float = 1.64,
        residual_quantile: float = 0.70,
        default_residual: float = 0.10,
        edge_gain: float = 3.0,
        error_hurdle: float = 0.50,
        skill_power: float = 1.0,
        max_abs_exposure: float = 0.60,
        denom_floor: float = 10.0,
        vol_half_life_steps: int = 288,
        vol_target: float = 0.15,
    ) -> None:
        self.raw = raw_features
        self.decision_freq = max(int(decision_freq), 1)
        self.horizon = max(int(horizon_steps), 1)
        self.window_size = max(int(window_size), 1)
        self.min_samples = max(int(min_samples), 1)
        self.hit_lcb_z = float(max(hit_lcb_z, 0.0))
        self.residual_quantile = float(np.clip(residual_quantile, 0.50, 0.99))
        self.default_residual = float(max(default_residual, 1e-6))
        self.edge_gain = float(max(edge_gain, 0.0))
        self.error_hurdle = float(max(error_hurdle, 0.0))
        self.skill_power = float(max(skill_power, 1e-6))
        self.max_abs_exposure = float(np.clip(max_abs_exposure, 0.0, 1.0))
        self.denom_floor = float(max(denom_floor, 1e-6))
        self.vol_half_life_steps = max(int(vol_half_life_steps), 1)
        self.vol_target = float(max(vol_target, 1e-6))

        self._hits = deque(maxlen=self.window_size)
        self._abs_errors = deque(maxlen=self.window_size)
        self._prices: Dict[int, float] = {}
        self._calibrated = set()
        self._ema_logret2 = (1e-3) ** 2
        self._last_price: Optional[float] = None
        self._last_t: Optional[int] = None
        self._last_signal = self._empty_signal(reason="cold")

    @staticmethod
    def _sgn(x: float) -> float:
        if x > 0.0:
            return 1.0
        if x < 0.0:
            return -1.0
        return 0.0

    def _empty_signal(self, *, reason: str, phase: float = 0.0) -> Dict[str, Any]:
        return {
            "step": -1,
            "prior_exposure": 0.0,
            "forecast_sign": 0.0,
            "pred_return": 0.0,
            "direction_margin": 0.0,
            "direction_confidence": 0.5,
            "hit_lcb": 0.5,
            "skill": 0.0,
            "residual_q": self.default_residual,
            "edge_excess": 0.0,
            "magnitude": 0.0,
            "quality": 0.0,
            "uncertainty": 1.0,
            "sigma_h": 0.0,
            "calibration_count": len(self._hits),
            "phase": float(phase),
            "active": False,
            "reason": str(reason),
        }

    def _update_price_state(self, t: int, price: float) -> None:
        if not (np.isfinite(price) and price > 0.0):
            return
        self._prices[int(t)] = float(price)
        if self._last_price is not None and self._last_price > 0.0:
            lr = float(np.log(float(price) / self._last_price))
            alpha = 1.0 - 0.5 ** (1.0 / float(self.vol_half_life_steps))
            self._ema_logret2 = float(
                (1.0 - alpha) * self._ema_logret2 + alpha * (lr * lr)
            )
        self._last_price = float(price)

    def _update_calibration(self, t: int, price: float) -> None:
        k = int(t) - self.horizon
        if k < 0 or k in self._calibrated:
            return
        self._calibrated.add(k)
        if self.raw is None or k >= self.raw.shape[0]:
            return
        past_price = self._prices.get(k)
        if past_price is None or not (np.isfinite(past_price) and past_price > 0.0):
            return
        if not (np.isfinite(price) and price > 0.0):
            return

        realized = float((float(price) - past_price) / max(abs(past_price), self.denom_floor))
        pred_return = float(self.raw[k, 0])
        margin = float(np.clip(self.raw[k, 2], -1.0, 1.0))
        forecast_sign = self._sgn(margin) or self._sgn(pred_return)
        realized_sign = self._sgn(realized)
        if forecast_sign != 0.0 and realized_sign != 0.0:
            self._hits.append(1.0 if forecast_sign == realized_sign else 0.0)
        if np.isfinite(realized) and np.isfinite(pred_return):
            self._abs_errors.append(abs(realized - pred_return))

    def update_and_compute(self, t: int, price: float) -> Dict[str, Any]:
        t = int(t)
        phase = float(t % self.decision_freq) / float(self.decision_freq)
        if self._last_t == t:
            return dict(self._last_signal)

        price_f = float(price) if np.isfinite(price) else 0.0
        self._update_price_state(t, price_f)
        self._update_calibration(t, price_f)

        if self.raw is None or t < 0 or t >= self.raw.shape[0]:
            out = self._empty_signal(reason="no_cache", phase=phase)
            out["step"] = t
            self._last_t = t
            self._last_signal = out
            return dict(out)

        pred_return = float(self.raw[t, 0])
        direction_prob = float(np.clip(self.raw[t, 1], 0.0, 1.0))
        margin = float(np.clip(self.raw[t, 2], -1.0, 1.0))
        uncertainty = float(max(0.0, self.raw[t, 3]))
        quality = float(np.clip(self.raw[t, 4], 0.0, 1.0))

        forecast_sign = self._sgn(margin) or self._sgn(pred_return)
        if forecast_sign > 0.0:
            direction_confidence = direction_prob
        elif forecast_sign < 0.0:
            direction_confidence = 1.0 - direction_prob
        else:
            direction_confidence = 0.5
        direction_confidence = float(np.clip(direction_confidence, 0.0, 1.0))

        if len(self._hits) >= self.min_samples:
            p_hat = float(np.mean(np.asarray(self._hits, dtype=np.float64)))
            se = float(np.sqrt(max(p_hat * (1.0 - p_hat), 1e-6) / max(len(self._hits), 1)))
            hit_lcb = float(np.clip(p_hat - self.hit_lcb_z * se, 0.0, 1.0))
        else:
            # Stay flat until the prior has causal evidence from this episode.
            # Direction-probability alone made the old prior active nearly
            # everywhere and allowed high exposure before error calibration.
            hit_lcb = 0.5

        if len(self._abs_errors) >= self.min_samples:
            residual_q = float(
                np.quantile(np.asarray(self._abs_errors, dtype=np.float64), self.residual_quantile)
            )
        else:
            residual_q = self.default_residual
        residual_q = float(max(residual_q, 1e-6))

        skill_raw = float(max(0.0, 2.0 * hit_lcb - 1.0))
        skill = float(skill_raw ** self.skill_power)
        edge_excess = float(max(0.0, abs(pred_return) - self.error_hurdle * residual_q))
        magnitude = float(edge_excess / (abs(pred_return) + residual_q))
        sigma_step = float(max(self._ema_logret2, 1e-12) ** 0.5)
        sigma_h = float(max(sigma_step * (self.horizon ** 0.5), 1e-6))
        vol_shrink = float(min(1.0, self.vol_target / sigma_h))

        raw_prior = (
            forecast_sign
            * self.edge_gain
            * skill
            * magnitude
            * quality
            * float(np.exp(-uncertainty))
            * vol_shrink
        )
        prior_exposure = float(np.clip(raw_prior, -self.max_abs_exposure, self.max_abs_exposure))

        active = bool(abs(prior_exposure) > 1e-9 and forecast_sign != 0.0 and skill > 0.0)
        out = {
            "step": t,
            "prior_exposure": prior_exposure,
            "forecast_sign": float(forecast_sign),
            "pred_return": pred_return,
            "direction_margin": margin,
            "direction_confidence": direction_confidence,
            "hit_lcb": hit_lcb,
            "skill": skill,
            "residual_q": residual_q,
            "edge_excess": edge_excess,
            "magnitude": magnitude,
            "quality": quality,
            "uncertainty": uncertainty,
            "sigma_h": sigma_h,
            "calibration_count": len(self._hits),
            "phase": phase,
            "active": active,
            "reason": "active" if active else "zero_edge",
        }
        self._last_t = t
        self._last_signal = out
        return dict(out)


__all__ = [
    "FORECAST_PRIOR_FEATURE_COLS",
    "FORECAST_PRIOR_OPTIONAL_COLS",
    "FORECAST_PRIOR_ALL_COLS",
    "load_forecast_prior_features",
    "ConformalForecastPrior",
]
