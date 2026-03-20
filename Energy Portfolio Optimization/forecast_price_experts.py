from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model

try:
    from PyEMD import CEEMDAN
except Exception:
    CEEMDAN = None


logger = logging.getLogger(__name__)

PRICE_SHORT_EXPERT_METHODS: Tuple[str, ...] = (
    "ann",
    "lstm",
    "svr",
    "rf",
    # "ceemdan_lstm",  # DEACTIVATED: slow training; uncomment to re-enable
)

PRICE_SHORT_EXPERT_LABELS: Dict[str, str] = {
    "ann": "ANN",
    "lstm": "LSTM",
    "svr": "SVR",
    "rf": "RF",
    # "ceemdan_lstm": "CEEMDAN_LSTM",  # DEACTIVATED
}

PRICE_SHORT_EXPERT_VERSION = "1.1.0"
PRICE_SHORT_EXPERT_TARGET = "price"
PRICE_SHORT_EXPERT_HORIZON = "short"
PRICE_SHORT_EXPERT_DENOM_FLOOR = 50.0
PRICE_SHORT_EXPERT_CEEMDAN_MAX_IMFS = 4
PRICE_SHORT_EXPERT_CEEMDAN_CHANNELS = PRICE_SHORT_EXPERT_CEEMDAN_MAX_IMFS + 1
PRICE_SHORT_EXPERT_SAMPLE_LIMITS = {
    "ann": (20000, 4000, 4000),
    "lstm": (16000, 3000, 3000),
    "svr": (12000, 2500, 2500),
    "rf": (12000, 2500, 2500),
    # "ceemdan_lstm": (4000, 800, 800),  # DEACTIVATED
}


def _set_deterministic_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> float:
    denom = np.maximum(np.abs(anchors), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.abs(y_true - y_pred) / denom
    return float(np.nanmean(vals) * 100.0)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> float:
    actual_ret = np.asarray(y_true, dtype=np.float32) - np.asarray(anchors, dtype=np.float32)
    pred_ret = np.asarray(y_pred, dtype=np.float32) - np.asarray(anchors, dtype=np.float32)
    mask = np.abs(actual_ret) > 1e-8
    if not np.any(mask):
        return 0.5
    return float(np.mean(np.sign(actual_ret[mask]) == np.sign(pred_ret[mask])))


def _residual_risk(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> float:
    denom = np.maximum(np.abs(anchors), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    actual_ret = (np.asarray(y_true, dtype=np.float32) - anchors) / denom
    pred_ret = (np.asarray(y_pred, dtype=np.float32) - anchors) / denom
    residuals = np.abs(actual_ret - pred_ret)
    if residuals.size <= 0:
        return 0.5
    tail = float(np.quantile(residuals, 0.90))
    return float(np.clip(np.tanh(tail / 0.10), 0.0, 1.0))


def _create_horizon_windows(series: np.ndarray, look_back: int, horizon_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    n = int(values.size)
    end = n - int(look_back) - int(horizon_steps) + 1
    if end <= 0:
        return (
            np.zeros((0, int(look_back)), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    x = np.zeros((end, int(look_back)), dtype=np.float32)
    y = np.zeros(end, dtype=np.float32)
    anchors = np.zeros(end, dtype=np.float32)
    for idx in range(end):
        window = values[idx: idx + int(look_back)]
        target_idx = idx + int(look_back) + int(horizon_steps) - 1
        x[idx, :] = window
        y[idx] = float(values[target_idx])
        anchors[idx] = float(window[-1])
    return x, y, anchors


def _window_returns(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    prev = np.concatenate([values[:, :1], values[:, :-1]], axis=1)
    denom = np.maximum(np.abs(prev), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    returns = (values - prev) / denom
    returns[:, 0] = 0.0
    return returns.astype(np.float32)


def _window_momentum(values: np.ndarray, span: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] <= 1:
        return np.zeros(arr.shape[0], dtype=np.float32)
    span = max(1, min(int(span), arr.shape[1] - 1))
    anchor_idx = max(0, arr.shape[1] - 1 - span)
    anchor = arr[:, anchor_idx]
    denom = np.maximum(np.abs(anchor), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    return ((arr[:, -1] - anchor) / denom).astype(np.float32)


def _build_engineered_window_features(windows: np.ndarray, variant: str) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    look_back = int(values.shape[1]) if values.ndim == 2 else 0
    if look_back <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    returns = _window_returns(values)
    active_returns = returns[:, 1:] if look_back > 1 else returns
    short_span = max(2, look_back // 4)
    medium_span = max(3, look_back // 2)

    last = values[:, -1]
    first = values[:, 0]
    mean_price = np.mean(values, axis=1)
    std_price = np.std(values, axis=1)
    min_price = np.min(values, axis=1)
    max_price = np.max(values, axis=1)
    range_price = np.maximum(max_price - min_price, 1e-6)
    denom_last = np.maximum(np.abs(last), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    denom_first = np.maximum(np.abs(first), PRICE_SHORT_EXPERT_DENOM_FLOOR)

    mean_ret = np.mean(active_returns, axis=1)
    std_ret = np.std(active_returns, axis=1)
    abs_mean_ret = np.mean(np.abs(active_returns), axis=1)
    short_returns = active_returns[:, -short_span:]
    medium_returns = active_returns[:, -medium_span:]
    short_mean_ret = np.mean(short_returns, axis=1)
    short_std_ret = np.std(short_returns, axis=1)
    medium_mean_ret = np.mean(medium_returns, axis=1)
    medium_std_ret = np.std(medium_returns, axis=1)
    sign_balance = np.mean(np.sign(active_returns), axis=1)
    up_fraction = np.mean((active_returns > 0.0).astype(np.float32), axis=1)

    level_bias = (last - mean_price) / denom_last
    range_norm = (max_price - min_price) / denom_last
    price_position = (last - min_price) / range_price
    momentum_full = (last - first) / denom_first
    momentum_3 = _window_momentum(values, 3)
    momentum_6 = _window_momentum(values, 6)
    momentum_12 = _window_momentum(values, 12)
    volatility_ratio = short_std_ret / np.maximum(medium_std_ret, 1e-6)
    mean_reversion_gap = short_mean_ret - medium_mean_ret
    price_std_norm = std_price / denom_last

    if str(variant).strip().lower() == "svr":
        features = np.column_stack(
            [
                level_bias,
                momentum_full,
                momentum_3,
                momentum_6,
                mean_ret,
                std_ret,
                short_mean_ret,
                short_std_ret,
                range_norm,
                price_position,
                sign_balance,
                mean_reversion_gap,
            ]
        )
    else:
        features = np.column_stack(
            [
                level_bias,
                momentum_full,
                momentum_3,
                momentum_6,
                momentum_12,
                mean_ret,
                std_ret,
                abs_mean_ret,
                short_mean_ret,
                short_std_ret,
                medium_mean_ret,
                medium_std_ret,
                range_norm,
                price_position,
                up_fraction,
                sign_balance,
                volatility_ratio,
                price_std_norm,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def _get_method_input_view(method: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(metadata, dict):
        view = str(metadata.get("input_view", "") or "").strip()
        if view:
            return view
        if str(metadata.get("version", "")) != PRICE_SHORT_EXPERT_VERSION:
            return "raw_price_window"
    method_key = str(method).strip().lower()
    default_views = {
        "ann": "raw_price_window",
        "lstm": "return_sequence",
        "svr": "engineered_smooth_regime",
        "rf": "engineered_tree_regime",
        "ceemdan_lstm": "ceemdan_tensor",
    }
    return default_views.get(method_key, "raw_price_window")


def _build_method_input_view(
    windows: np.ndarray,
    method: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    method_key = str(method).strip().lower()
    input_view = _get_method_input_view(method_key, metadata=metadata)
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    if input_view == "return_sequence":
        return _window_returns(values)
    if input_view == "engineered_smooth_regime":
        return _build_engineered_window_features(values, variant="svr")
    if input_view == "engineered_tree_regime":
        return _build_engineered_window_features(values, variant="rf")
    return values.astype(np.float32)


def _split_series_three_way(series: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    n = int(values.size)
    train_size = int(n * float(train_ratio))
    val_size = int(n * float(val_ratio))
    train = values[:train_size]
    val = values[train_size: train_size + val_size]
    test = values[train_size + val_size:]
    return train, val, test


def _subsample_evenly(
    x: np.ndarray,
    y: np.ndarray,
    anchors: np.ndarray,
    max_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or x.shape[0] <= int(max_samples):
        return x, y, anchors
    idx = np.linspace(0, x.shape[0] - 1, int(max_samples), dtype=np.int32)
    return x[idx], y[idx], anchors[idx]


def _fit_standard_scalers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[StandardScaler, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train_scaled = sc_x.fit_transform(x_train)
    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    x_val_scaled = sc_x.transform(x_val)
    y_val_scaled = sc_y.transform(y_val.reshape(-1, 1)).ravel()
    x_test_scaled = sc_x.transform(x_test)
    y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()
    return (
        sc_x,
        sc_y,
        x_train_scaled.astype(np.float32),
        y_train_scaled.astype(np.float32),
        x_val_scaled.astype(np.float32),
        y_val_scaled.astype(np.float32),
        x_test_scaled.astype(np.float32),
        y_test_scaled.astype(np.float32),
    )


def _build_ann_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            tf.keras.Input(shape=(int(input_dim),)),
            Dense(256, activation="relu"),
            Dropout(0.15),
            Dense(128, activation="relu"),
            Dropout(0.10),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4))
    return model


def _build_lstm_model(look_back: int, channels: int = 1) -> Sequential:
    model = Sequential(
        [
            tf.keras.Input(shape=(int(look_back), int(channels))),
            LSTM(64, activation="tanh"),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4))
    return model


def _fit_keras_model(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    history_path: str,
    epochs: int = 120,
    batch_size: int = 64,
    patience: int = 12,
) -> Sequential:
    ckpt = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=0,
    )
    early = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(4, int(patience)),
        restore_best_weights=True,
        verbose=0,
    )
    rlrop = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(2, int(patience) // 3),
        min_lr=1e-5,
        verbose=0,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        shuffle=False,
        callbacks=[ckpt, early, rlrop],
        verbose=0,
    )
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "loss": [float(v) for v in history.history.get("loss", [])],
                "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            },
            fh,
            indent=2,
        )
    best_model = load_model(model_path, compile=False)
    best_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4))
    return best_model


def _decompose_window_ceemdan(window: np.ndarray, max_imfs: int = PRICE_SHORT_EXPERT_CEEMDAN_MAX_IMFS) -> np.ndarray:
    if CEEMDAN is None:
        raise RuntimeError("PyEMD is not installed; cannot build CEEMDAN-LSTM expert.")
    values = np.asarray(window, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        return np.zeros((0, PRICE_SHORT_EXPERT_CEEMDAN_CHANNELS), dtype=np.float32)
    decomposer = CEEMDAN(epsilon=0.05)
    imfs = np.asarray(decomposer(values), dtype=np.float32)
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)
    channels = np.zeros((int(max_imfs) + 1, values.size), dtype=np.float32)
    use_imfs = min(int(max_imfs), int(imfs.shape[0]))
    if use_imfs > 0:
        channels[:use_imfs, :] = imfs[:use_imfs, :]
    residual = values - np.sum(imfs[:use_imfs, :], axis=0, dtype=np.float32)
    channels[int(max_imfs), :] = residual.astype(np.float32)
    return channels.T.astype(np.float32)


def _build_ceemdan_tensor(windows: np.ndarray) -> np.ndarray:
    n, look_back = windows.shape[0], windows.shape[1]
    tensor = np.zeros((n, look_back, PRICE_SHORT_EXPERT_CEEMDAN_CHANNELS), dtype=np.float32)
    for idx in range(n):
        tensor[idx] = _decompose_window_ceemdan(windows[idx])
    return tensor


def _transform_3d_with_feature_scaler(tensor: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
    n, steps, channels = tensor.shape
    flat = tensor.reshape(-1, channels)
    flat_scaled = scaler.fit_transform(flat) if fit else scaler.transform(flat)
    return flat_scaled.reshape(n, steps, channels).astype(np.float32)


def get_price_short_expert_root(episode_dir: str) -> str:
    return os.path.join(str(episode_dir), "price_short_experts")


def get_price_short_expert_paths(episode_dir: str, method: str) -> Dict[str, str]:
    method_key = str(method).strip().lower()
    if method_key not in PRICE_SHORT_EXPERT_LABELS:
        raise ValueError(f"Unknown price-short expert method: {method}")
    root = _ensure_dir(os.path.join(get_price_short_expert_root(episode_dir), PRICE_SHORT_EXPERT_LABELS[method_key]))
    ext = ".keras" if method_key in ("ann", "lstm", "ceemdan_lstm") else ".pkl"
    return {
        "root": root,
        "model_path": os.path.join(root, f"{method_key}_model{ext}"),
        "scaler_x_path": os.path.join(root, f"{method_key}_scaler_x.pkl"),
        "scaler_y_path": os.path.join(root, f"{method_key}_scaler_y.pkl"),
        "feature_scaler_path": os.path.join(root, f"{method_key}_feature_scaler.pkl"),
        "history_path": os.path.join(root, f"{method_key}_history.json"),
        "metadata_path": os.path.join(root, f"{method_key}_metadata.json"),
    }


def price_short_expert_bank_exists(episode_dir: str) -> bool:
    try:
        for method in PRICE_SHORT_EXPERT_METHODS:
            paths = get_price_short_expert_paths(episode_dir, method)
            if not os.path.isfile(paths["metadata_path"]):
                return False
            with open(paths["metadata_path"], "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            if str(metadata.get("version", "")) != PRICE_SHORT_EXPERT_VERSION:
                return False
            if not os.path.isfile(paths["model_path"]):
                return False
            if not os.path.isfile(paths["scaler_x_path"]):
                return False
            if not os.path.isfile(paths["scaler_y_path"]):
                return False
            if method == "ceemdan_lstm" and not os.path.isfile(paths["feature_scaler_path"]):
                return False
        return True
    except Exception:
        return False


def _evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchors: np.ndarray,
) -> Dict[str, float]:
    return {
        "mape": _safe_mape(y_true, y_pred, anchors),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_accuracy": _directional_accuracy(y_true, y_pred, anchors),
        "residual_risk": _residual_risk(y_true, y_pred, anchors),
    }


def _build_metadata_quality(metrics: Dict[str, float]) -> float:
    dir_acc = float(np.clip(metrics.get("directional_accuracy", 0.5), 0.0, 1.0))
    mape = float(max(metrics.get("mape", 100.0), 0.0))
    residual_risk = float(np.clip(metrics.get("residual_risk", 0.5), 0.0, 1.0))
    q_mape = 1.0 / (1.0 + mape / 25.0)
    return float(np.clip(0.50 * dir_acc + 0.30 * q_mape + 0.20 * (1.0 - residual_risk), 0.0, 1.0))


@dataclass
class ExpertArtifacts:
    method: str
    model: Any
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    metadata: Dict[str, Any]
    feature_scaler: Optional[StandardScaler] = None


class PriceShortExpertBank:
    """Real short-horizon price expert bank used by Tier-2 routing."""

    def __init__(
        self,
        episode_dir: str,
        look_back: int,
        horizon_steps: int,
        verbose: bool = False,
        refresh_stride: int = 6,
    ):
        self.episode_dir = str(episode_dir)
        self.look_back = int(look_back)
        self.horizon_steps = int(horizon_steps)
        self.verbose = bool(verbose)
        self.refresh_stride = max(1, int(refresh_stride))
        self.artifacts: Dict[str, ExpertArtifacts] = {}
        self._load()

    def _load(self) -> None:
        self.artifacts.clear()
        for method in PRICE_SHORT_EXPERT_METHODS:
            paths = get_price_short_expert_paths(self.episode_dir, method)
            if not os.path.isfile(paths["metadata_path"]):
                continue
            with open(paths["metadata_path"], "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            if str(metadata.get("version", "")) != PRICE_SHORT_EXPERT_VERSION:
                logger.warning(
                    "Price-short expert %s in %s uses metadata version %s; current version is %s.",
                    method,
                    self.episode_dir,
                    metadata.get("version", "unknown"),
                    PRICE_SHORT_EXPERT_VERSION,
                )
                continue
            scaler_x = joblib.load(paths["scaler_x_path"])
            scaler_y = joblib.load(paths["scaler_y_path"])
            feature_scaler = None
            if method == "ceemdan_lstm" and os.path.isfile(paths["feature_scaler_path"]):
                feature_scaler = joblib.load(paths["feature_scaler_path"])
            if paths["model_path"].endswith(".pkl"):
                model = joblib.load(paths["model_path"])
            else:
                model = load_model(paths["model_path"], compile=False)
            self.artifacts[method] = ExpertArtifacts(
                method=method,
                model=model,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                metadata=dict(metadata),
                feature_scaler=feature_scaler,
            )
        if self.verbose:
            logger.info(
                "Loaded %s/%s price-short experts from %s",
                len(self.artifacts),
                len(PRICE_SHORT_EXPERT_METHODS),
                self.episode_dir,
            )

    def is_complete(self) -> bool:
        return all(method in self.artifacts for method in PRICE_SHORT_EXPERT_METHODS)

    def methods(self) -> Tuple[str, ...]:
        return tuple(method for method in PRICE_SHORT_EXPERT_METHODS if method in self.artifacts)

    def get_metadata_quality(self, method: str) -> float:
        art = self.artifacts.get(str(method).strip().lower())
        if art is None:
            return 0.5
        return float(np.clip(art.metadata.get("metadata_quality", 0.5), 0.0, 1.0))

    def get_metadata_metrics(self, method: str) -> Dict[str, float]:
        art = self.artifacts.get(str(method).strip().lower())
        if art is None:
            return {}
        metrics = art.metadata.get("test_metrics", {}) or {}
        return {str(k): float(v) for k, v in metrics.items() if np.isfinite(v)}

    def get_cache_validation_info(self) -> Dict[str, Dict[str, float | str]]:
        info: Dict[str, Dict[str, float | str]] = {}
        for method in self.methods():
            paths = get_price_short_expert_paths(self.episode_dir, method)
            info[method] = {
                "model_path": paths["model_path"],
                "model_mtime": float(os.path.getmtime(paths["model_path"])) if os.path.isfile(paths["model_path"]) else 0.0,
                "metadata_path": paths["metadata_path"],
                "metadata_mtime": float(os.path.getmtime(paths["metadata_path"])) if os.path.isfile(paths["metadata_path"]) else 0.0,
            }
        return info

    def _predict_non_ceemdan_batch(self, method: str, windows: np.ndarray) -> np.ndarray:
        art = self.artifacts[method]
        model_input = _build_method_input_view(windows, method, metadata=art.metadata)
        x_scaled = art.scaler_x.transform(model_input)
        if method == "lstm":
            model_in = x_scaled.reshape((-1, x_scaled.shape[1], 1)).astype(np.float32)
            preds_scaled = np.ravel(art.model.predict(model_in, verbose=0))
        else:
            if method == "rf" and hasattr(art.model, "n_jobs"):
                try:
                    art.model.n_jobs = 1
                except Exception:
                    pass
            preds_scaled = np.ravel(art.model.predict(x_scaled))
        preds = art.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))[:, 0]
        return np.asarray(preds, dtype=np.float32)

    def _predict_ceemdan_batch(self, windows: np.ndarray) -> np.ndarray:
        art = self.artifacts["ceemdan_lstm"]
        if art.feature_scaler is None:
            raise RuntimeError("CEEMDAN-LSTM feature scaler is missing.")
        tensor = _build_ceemdan_tensor(windows)
        tensor_scaled = _transform_3d_with_feature_scaler(tensor, art.feature_scaler, fit=False)
        preds_scaled = np.ravel(art.model.predict(tensor_scaled, verbose=0))
        preds = art.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))[:, 0]
        return np.asarray(preds, dtype=np.float32)

    def predict_from_window(self, method: str, window: np.ndarray) -> float:
        values = np.asarray(window, dtype=np.float32).reshape(-1)
        if values.size != self.look_back:
            raise ValueError(f"Expected window length {self.look_back}, got {values.size}")
        method_key = str(method).strip().lower()
        if method_key not in self.artifacts:
            raise KeyError(f"Price-short expert not loaded: {method_key}")
        windows = values.reshape(1, -1)
        if method_key == "ceemdan_lstm":
            pred = self._predict_ceemdan_batch(windows)[0]
        else:
            pred = self._predict_non_ceemdan_batch(method_key, windows)[0]
        return float(pred)

    def predict_all_from_window(self, window: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for method in self.methods():
            out[method] = self.predict_from_window(method, window)
        return out

    def precompute_for_series(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        values = np.asarray(series, dtype=np.float32).reshape(-1)
        t_count = int(values.size)
        if t_count <= 0:
            return {method: np.zeros(0, dtype=np.float32) for method in self.methods()}
        windows = np.zeros((t_count, self.look_back), dtype=np.float32)
        mean_val = float(np.mean(values)) if values.size > 0 else 0.0
        for t in range(t_count):
            if t == 0:
                windows[t, :] = mean_val
                continue
            # Include price at t so forecast[t] predicts price[t+horizon_steps] (matches env expectation)
            start = max(0, t + 1 - self.look_back)
            hist = values[start : t + 1]
            if hist.size <= 0:
                windows[t, :] = mean_val
            elif hist.size < self.look_back:
                windows[t, : self.look_back - hist.size] = float(np.mean(hist))
                windows[t, self.look_back - hist.size :] = hist
            else:
                windows[t, :] = hist[-self.look_back :]
        outputs: Dict[str, np.ndarray] = {}
        for method in self.methods():
            if method == "ceemdan_lstm":
                preds = np.zeros(t_count, dtype=np.float32)
                step_values = list(range(0, t_count, self.refresh_stride))
                if step_values[-1] != t_count - 1:
                    step_values.append(t_count - 1)
                last_value = float(values[0]) if t_count > 0 else 0.0
                cursor = 0
                for step in step_values:
                    pred_val = float(self.predict_from_window(method, windows[step]))
                    preds[cursor : step + 1] = last_value
                    preds[step] = pred_val
                    last_value = pred_val
                    cursor = step + 1
                if cursor < t_count:
                    preds[cursor:] = last_value
                outputs[method] = preds
            else:
                outputs[method] = self._predict_non_ceemdan_batch(method, windows)
        return outputs


def _save_metadata(
    metadata_path: str,
    payload: Dict[str, Any],
) -> None:
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _save_standard_artifacts(
    method: str,
    episode_dir: str,
    model: Any,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    test_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    train_count: int,
    val_count: int,
    test_count: int,
    sampled_counts: Dict[str, int],
    look_back: int,
    horizon_steps: int,
    history_path: Optional[str] = None,
    feature_scaler: Optional[StandardScaler] = None,
    input_view: Optional[str] = None,
) -> Dict[str, str]:
    paths = get_price_short_expert_paths(episode_dir, method)
    if paths["model_path"].endswith(".pkl"):
        joblib.dump(model, paths["model_path"])
    else:
        model.save(paths["model_path"], include_optimizer=True)
    joblib.dump(scaler_x, paths["scaler_x_path"])
    joblib.dump(scaler_y, paths["scaler_y_path"])
    if feature_scaler is not None:
        joblib.dump(feature_scaler, paths["feature_scaler_path"])
    metadata = {
        "version": PRICE_SHORT_EXPERT_VERSION,
        "method": method,
        "target": PRICE_SHORT_EXPERT_TARGET,
        "horizon": PRICE_SHORT_EXPERT_HORIZON,
        "look_back": int(look_back),
        "horizon_steps": int(horizon_steps),
        "input_view": str(input_view or _get_method_input_view(method)),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "sampled_counts": {k: int(v) for k, v in sampled_counts.items()},
        "model_path": paths["model_path"],
        "scaler_x_path": paths["scaler_x_path"],
        "scaler_y_path": paths["scaler_y_path"],
        "feature_scaler_path": paths["feature_scaler_path"] if feature_scaler is not None else None,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "metadata_quality": _build_metadata_quality(test_metrics),
    }
    if history_path and os.path.isfile(history_path):
        metadata["history_path"] = history_path
    _save_metadata(paths["metadata_path"], metadata)
    return paths


def _train_ann_expert(
    episode_dir: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    a_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    a_test: np.ndarray,
    look_back: int,
    horizon_steps: int,
    seed: int,
) -> Dict[str, Any]:
    del a_train
    paths = get_price_short_expert_paths(episode_dir, "ann")
    x_train_view = _build_method_input_view(x_train, "ann")
    x_val_view = _build_method_input_view(x_val, "ann")
    x_test_view = _build_method_input_view(x_test, "ann")
    sc_x, sc_y, x_train_s, y_train_s, x_val_s, y_val_s, _, _ = _fit_standard_scalers(
        x_train_view, y_train, x_val_view, y_val, x_test_view, y_test
    )
    x_test_s = sc_x.transform(x_test_view).astype(np.float32)
    _set_deterministic_seed(seed)
    model = _build_ann_model(x_train_view.shape[1])
    best_model = _fit_keras_model(
        model,
        x_train_s,
        y_train_s,
        x_val_s,
        y_val_s,
        paths["model_path"],
        paths["history_path"],
        epochs=100,
        batch_size=64,
        patience=10,
    )
    val_pred = sc_y.inverse_transform(np.ravel(best_model.predict(x_val_s, verbose=0)).reshape(-1, 1))[:, 0]
    test_pred = sc_y.inverse_transform(np.ravel(best_model.predict(x_test_s, verbose=0)).reshape(-1, 1))[:, 0]
    val_metrics = _evaluate_forecast(y_val, val_pred, a_val)
    test_metrics = _evaluate_forecast(y_test, test_pred, a_test)
    _save_standard_artifacts(
        "ann",
        episode_dir,
        best_model,
        sc_x,
        sc_y,
        test_metrics,
        val_metrics,
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
        {"train": x_train.shape[0], "val": x_val.shape[0], "test": x_test.shape[0]},
        look_back,
        horizon_steps,
        history_path=paths["history_path"],
        input_view=_get_method_input_view("ann"),
    )
    return {"method": "ann", "success": True, "test_metrics": test_metrics, "val_metrics": val_metrics}


def _train_lstm_expert(
    episode_dir: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    a_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    a_test: np.ndarray,
    look_back: int,
    horizon_steps: int,
    seed: int,
) -> Dict[str, Any]:
    del a_train
    paths = get_price_short_expert_paths(episode_dir, "lstm")
    x_train_view = _build_method_input_view(x_train, "lstm")
    x_val_view = _build_method_input_view(x_val, "lstm")
    x_test_view = _build_method_input_view(x_test, "lstm")
    sc_x, sc_y, x_train_s, y_train_s, x_val_s, y_val_s, _, _ = _fit_standard_scalers(
        x_train_view, y_train, x_val_view, y_val, x_test_view, y_test
    )
    x_test_s = sc_x.transform(x_test_view).astype(np.float32)
    x_train_seq = x_train_s.reshape((-1, look_back, 1))
    x_val_seq = x_val_s.reshape((-1, look_back, 1))
    x_test_seq = x_test_s.reshape((-1, look_back, 1))
    _set_deterministic_seed(seed)
    model = _build_lstm_model(look_back, channels=1)
    best_model = _fit_keras_model(
        model,
        x_train_seq,
        y_train_s,
        x_val_seq,
        y_val_s,
        paths["model_path"],
        paths["history_path"],
        epochs=120,
        batch_size=64,
        patience=12,
    )
    val_pred = sc_y.inverse_transform(np.ravel(best_model.predict(x_val_seq, verbose=0)).reshape(-1, 1))[:, 0]
    test_pred = sc_y.inverse_transform(np.ravel(best_model.predict(x_test_seq, verbose=0)).reshape(-1, 1))[:, 0]
    val_metrics = _evaluate_forecast(y_val, val_pred, a_val)
    test_metrics = _evaluate_forecast(y_test, test_pred, a_test)
    _save_standard_artifacts(
        "lstm",
        episode_dir,
        best_model,
        sc_x,
        sc_y,
        test_metrics,
        val_metrics,
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
        {"train": x_train.shape[0], "val": x_val.shape[0], "test": x_test.shape[0]},
        look_back,
        horizon_steps,
        history_path=paths["history_path"],
        input_view=_get_method_input_view("lstm"),
    )
    return {"method": "lstm", "success": True, "test_metrics": test_metrics, "val_metrics": val_metrics}


def _train_svr_expert(
    episode_dir: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    a_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    a_test: np.ndarray,
    look_back: int,
    horizon_steps: int,
    seed: int,
) -> Dict[str, Any]:
    del seed, a_train
    x_train_view = _build_method_input_view(x_train, "svr")
    x_val_view = _build_method_input_view(x_val, "svr")
    x_test_view = _build_method_input_view(x_test, "svr")
    sc_x, sc_y, x_train_s, y_train_s, x_val_s, _, x_test_s, _ = _fit_standard_scalers(
        x_train_view, y_train, x_val_view, y_val, x_test_view, y_test
    )
    model = SVR(kernel="rbf", C=3.0, epsilon=0.02, gamma="scale")
    model.fit(x_train_s, y_train_s)
    val_pred_s = model.predict(x_val_s)
    test_pred_s = model.predict(x_test_s)
    val_pred = sc_y.inverse_transform(val_pred_s.reshape(-1, 1))[:, 0]
    test_pred = sc_y.inverse_transform(test_pred_s.reshape(-1, 1))[:, 0]
    val_metrics = _evaluate_forecast(y_val, val_pred, a_val)
    test_metrics = _evaluate_forecast(y_test, test_pred, a_test)
    _save_standard_artifacts(
        "svr",
        episode_dir,
        model,
        sc_x,
        sc_y,
        test_metrics,
        val_metrics,
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
        {"train": x_train.shape[0], "val": x_val.shape[0], "test": x_test.shape[0]},
        look_back,
        horizon_steps,
        input_view=_get_method_input_view("svr"),
    )
    return {"method": "svr", "success": True, "test_metrics": test_metrics, "val_metrics": val_metrics}


def _train_rf_expert(
    episode_dir: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    a_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    a_test: np.ndarray,
    look_back: int,
    horizon_steps: int,
    seed: int,
) -> Dict[str, Any]:
    del a_train
    x_train_view = _build_method_input_view(x_train, "rf")
    x_val_view = _build_method_input_view(x_val, "rf")
    x_test_view = _build_method_input_view(x_test, "rf")
    sc_x, sc_y, x_train_s, y_train_s, x_val_s, _, x_test_s, _ = _fit_standard_scalers(
        x_train_view, y_train, x_val_view, y_val, x_test_view, y_test
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(x_train_s, y_train_s)
    val_pred_s = model.predict(x_val_s)
    test_pred_s = model.predict(x_test_s)
    val_pred = sc_y.inverse_transform(val_pred_s.reshape(-1, 1))[:, 0]
    test_pred = sc_y.inverse_transform(test_pred_s.reshape(-1, 1))[:, 0]
    val_metrics = _evaluate_forecast(y_val, val_pred, a_val)
    test_metrics = _evaluate_forecast(y_test, test_pred, a_test)
    _save_standard_artifacts(
        "rf",
        episode_dir,
        model,
        sc_x,
        sc_y,
        test_metrics,
        val_metrics,
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
        {"train": x_train.shape[0], "val": x_val.shape[0], "test": x_test.shape[0]},
        look_back,
        horizon_steps,
        input_view=_get_method_input_view("rf"),
    )
    return {"method": "rf", "success": True, "test_metrics": test_metrics, "val_metrics": val_metrics}


def _train_ceemdan_lstm_expert(
    episode_dir: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    a_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    a_test: np.ndarray,
    look_back: int,
    horizon_steps: int,
    seed: int,
) -> Dict[str, Any]:
    if CEEMDAN is None:
        raise RuntimeError("PyEMD is not installed; CEEMDAN-LSTM expert cannot be trained.")
    paths = get_price_short_expert_paths(episode_dir, "ceemdan_lstm")
    train_limit, val_limit, test_limit = PRICE_SHORT_EXPERT_SAMPLE_LIMITS["ceemdan_lstm"]
    x_train_sub, y_train_sub, a_train_sub = _subsample_evenly(x_train, y_train, a_train, train_limit)
    x_val_sub, y_val_sub, a_val_sub = _subsample_evenly(x_val, y_val, a_val, val_limit)
    x_test_sub, y_test_sub, a_test_sub = _subsample_evenly(x_test, y_test, a_test, test_limit)

    train_tensor = _build_ceemdan_tensor(x_train_sub)
    val_tensor = _build_ceemdan_tensor(x_val_sub)
    test_tensor = _build_ceemdan_tensor(x_test_sub)

    feat_scaler = StandardScaler()
    train_tensor_s = _transform_3d_with_feature_scaler(train_tensor, feat_scaler, fit=True)
    val_tensor_s = _transform_3d_with_feature_scaler(val_tensor, feat_scaler, fit=False)
    test_tensor_s = _transform_3d_with_feature_scaler(test_tensor, feat_scaler, fit=False)

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    sc_x.fit(x_train_sub)
    y_train_s = sc_y.fit_transform(y_train_sub.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = sc_y.transform(y_val_sub.reshape(-1, 1)).ravel().astype(np.float32)

    _set_deterministic_seed(seed)
    model = _build_lstm_model(look_back, channels=PRICE_SHORT_EXPERT_CEEMDAN_CHANNELS)
    best_model = _fit_keras_model(
        model,
        train_tensor_s,
        y_train_s,
        val_tensor_s,
        y_val_s,
        paths["model_path"],
        paths["history_path"],
        epochs=80,
        batch_size=32,
        patience=10,
    )
    val_pred = sc_y.inverse_transform(np.ravel(best_model.predict(val_tensor_s, verbose=0)).reshape(-1, 1))[:, 0]
    test_pred = sc_y.inverse_transform(np.ravel(best_model.predict(test_tensor_s, verbose=0)).reshape(-1, 1))[:, 0]
    val_metrics = _evaluate_forecast(y_val_sub, val_pred, a_val_sub)
    test_metrics = _evaluate_forecast(y_test_sub, test_pred, a_test_sub)
    _save_standard_artifacts(
        "ceemdan_lstm",
        episode_dir,
        best_model,
        sc_x,
        sc_y,
        test_metrics,
        val_metrics,
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
        {"train": x_train_sub.shape[0], "val": x_val_sub.shape[0], "test": x_test_sub.shape[0]},
        look_back,
        horizon_steps,
        history_path=paths["history_path"],
        feature_scaler=feat_scaler,
    )
    return {"method": "ceemdan_lstm", "success": True, "test_metrics": test_metrics, "val_metrics": val_metrics}


def train_price_short_expert_bank(
    episode_num: int,
    data_filtered: Any,
    output_base_dir: str = "forecast_models",
    look_back: int = 24,
    horizon_steps: int = 6,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 1234,
) -> Dict[str, Any]:
    episode_dir = os.path.join(str(output_base_dir), f"episode_{int(episode_num)}")
    _ensure_dir(get_price_short_expert_root(episode_dir))

    if PRICE_SHORT_EXPERT_TARGET not in data_filtered.columns:
        raise ValueError(f"'{PRICE_SHORT_EXPERT_TARGET}' column not found for price expert bank training.")

    series = np.asarray(data_filtered[PRICE_SHORT_EXPERT_TARGET].astype(np.float32).values, dtype=np.float32)
    if "Date" in data_filtered.columns:
        raw_dates = [str(x) for x in data_filtered["Date"].tolist()]
    elif "timestamp" in data_filtered.columns:
        raw_dates = [str(x) for x in data_filtered["timestamp"].tolist()]
    else:
        raw_dates = []
    train_series, val_series, test_series = _split_series_three_way(series, train_ratio=train_ratio, val_ratio=val_ratio)
    x_train, y_train, a_train = _create_horizon_windows(train_series, look_back, horizon_steps)
    x_val, y_val, a_val = _create_horizon_windows(val_series, look_back, horizon_steps)
    x_test, y_test, a_test = _create_horizon_windows(test_series, look_back, horizon_steps)
    if min(x_train.shape[0], x_val.shape[0], x_test.shape[0]) <= 0:
        raise ValueError("Insufficient price data to train the short-horizon expert bank.")

    results = []
    n_total = len(series)
    train_cut = int(n_total * float(train_ratio))
    val_cut = train_cut + int(n_total * float(val_ratio))
    training_start = raw_dates[0] if raw_dates else None
    training_end = raw_dates[max(0, train_cut - 1)] if raw_dates and train_cut > 0 else None
    validation_end = raw_dates[max(0, min(val_cut, n_total) - 1)] if raw_dates and val_cut > 0 else None
    test_end = raw_dates[-1] if raw_dates else None
    trainers = {
        "ann": _train_ann_expert,
        "lstm": _train_lstm_expert,
        "svr": _train_svr_expert,
        "rf": _train_rf_expert,
        # "ceemdan_lstm": _train_ceemdan_lstm_expert,  # DEACTIVATED
    }
    n_methods = len(PRICE_SHORT_EXPERT_METHODS)
    method_labels = [PRICE_SHORT_EXPERT_LABELS[m] for m in PRICE_SHORT_EXPERT_METHODS]
    print(f"[PRICE_SHORT_EXPERT_BANK] Training {n_methods} experts: {', '.join(method_labels)}")
    for offset, method in enumerate(PRICE_SHORT_EXPERT_METHODS):
        label = PRICE_SHORT_EXPERT_LABELS[method]
        print(f"  [{offset + 1}/{n_methods}] Training {label}...", flush=True)
        try:
            result = trainers[method](
                episode_dir,
                x_train,
                y_train,
                a_train,
                x_val,
                y_val,
                a_val,
                x_test,
                y_test,
                a_test,
                int(look_back),
                int(horizon_steps),
                int(seed) + offset,
            )
        except Exception as exc:
            result = {"method": method, "success": False, "error": str(exc)}
        status = "OK" if result.get("success") else f"FAILED ({result.get('error', 'unknown')})"
        print(f"  [{offset + 1}/{n_methods}] {label}: {status}", flush=True)
        if result.get("success"):
            try:
                paths = get_price_short_expert_paths(episode_dir, method)
                with open(paths["metadata_path"], "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                metadata["training_start"] = training_start
                metadata["training_end"] = training_end
                metadata["validation_end"] = validation_end
                metadata["test_end"] = test_end
                with open(paths["metadata_path"], "w", encoding="utf-8") as fh:
                    json.dump(metadata, fh, indent=2)
            except Exception as exc:
                result = {
                    "method": method,
                    "success": False,
                    "error": f"metadata_update_failed: {exc}",
                }
        results.append(result)
    successful = [r for r in results if r.get("success")]
    return {
        "episode_num": int(episode_num),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "results": results,
    }
