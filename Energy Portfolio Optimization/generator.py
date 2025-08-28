import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: F401 (used when loaded from disk)
import json
from typing import Dict, List, Optional, Tuple, Any, Mapping
import pandas as pd
from collections import deque
import logging

# =========================
# TensorFlow setup (optional)
# =========================

def fix_tensorflow_gpu_setup():
    """Best-effort TF GPU config set *before* importing TF. Safe if TF is absent."""
    try:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        os.environ.setdefault("TF_MEMORY_GROWTH", "true")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        import tensorflow as tf  # noqa
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            try:
                # keep a small cap to play nicely with other frameworks
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
                )
            except Exception:
                pass
        tf.get_logger().setLevel('ERROR')
        return tf
    except Exception:
        return None

tf = fix_tensorflow_gpu_setup()


# =========================
# Errors and Utilities
# =========================

class ModelLoadingError(Exception):
    pass

class ForecastGenerationError(Exception):
    pass

class SafeDivision:
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Robust division with protection against zero-division."""
        if denominator is None or abs(denominator) < 1e-9:
            return default
        try:
            return float(numerator) / float(denominator)
        except (ValueError, TypeError, ZeroDivisionError):
            return default


# =========================
# Forecast Generator
# =========================

class MultiHorizonForecastGenerator:
    """
    Fast, resilient multi-horizon forecaster with **per-agent** prediction and caching.

    Key performance features:
      - Per-agent caching: results are memoized per (agent, timestep)
      - Optional per-agent throttle via `agent_refresh_stride`
      - In-place input buffers per target: no per-call allocations
      - De-pandas hot path: history stored as fixed-len deques
      - Graceful fallback when models/scalers are missing

    The wrapper will:
      - call `predict_for_agent(agent, timestep)` each step for obs building
      - call `predict_all_horizons(timestep)` only for logging checkpoints

    Patched:
      - Anti-saturation for renewables/load forecasts (avoid hard 1.0).
      - Clip diagnostics (hit rates).
      - Bounded caches to prevent memory growth.
      - Updated to provide raw, unnormalized forecasts.
      - Bug fix: corrected handling of non-finite values in `_constrain_target`.
      - Corrected diagnostic helper name; added SafeDivision utility.
      - NEW: Offline precompute cache for O(1) forecast lookups during training.
    """

    def __init__(
        self,
        model_dir: str = "saved_models",
        scaler_dir: str = "saved_scalers",
        look_back: int = 6,
        verbose: bool = True,
        fallback_mode: bool = True,
        # PATCH: Changed the default value to 1 to disable throttling and ensure fresh forecasts.
        agent_refresh_stride: int = 1,
    ):
        self.look_back = int(look_back)
        self.verbose = verbose
        self.fallback_mode = fallback_mode
        self.agent_refresh_stride = max(1, int(agent_refresh_stride))

        # horizons in 10-min steps (names must match wrapper)
        self.horizons: Dict[str, int] = {
            "immediate": 1,     # 10 min
            "short": 6,         # 1 hour
            "medium": 24,       # 4 hours
            "long": 144,        # 24 hours
            "strategic": 1008,  # 1 week
        }

        # forecast targets (no 'risk' models)
        self.targets: List[str] = ["wind", "solar", "hydro", "price", "load"]

        # agent assignments (per wrapper/env design)
        self.agent_horizons: Dict[str, List[str]] = {
            "investor_0": ["immediate", "short"],
            "battery_operator_0": ["immediate", "short"],
            "risk_controller_0": [],  # no forecasts
            "meta_controller_0": ["immediate", "short", "medium"],
        }
        self.agent_targets: Dict[str, List[str]] = {
            "investor_0": ["wind", "solar", "hydro", "price"],
            "battery_operator_0": ["price", "load"],
            "risk_controller_0": [],
            "meta_controller_0": ["wind", "solar", "hydro", "price", "load"],
        }

        # storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Dict[str, Any]] = {}
        self._model_available: Dict[str, bool] = {}
        self.history: Dict[str, deque] = {}               # per-target values
        self._X_buffers: Dict[str, np.ndarray] = {}       # per-target (1, look_back) preallocated inputs

        # caches (bounded)
        self._global_cache: Dict[Tuple[str, str, int], float] = {}      # (target, horizon_name, t) -> value
        self._agent_cache: Dict[Tuple[str, int], Dict[str, float]] = {} # (agent, t) -> {key: val}
        self._last_agent_step: Dict[str, int] = {}
        self._cache_limit_global = 5000
        self._cache_limit_agent = 2000

        # NEW: offline precomputed forecasts (target, horizon) -> np.ndarray of shape (T,)
        self._precomputed: Dict[Tuple[str, str], np.ndarray] = {}

        # stats
        self.loading_stats = {
            "models_attempted": 0,
            "models_loaded": 0,
            "scalers_attempted": 0,
            "scalers_loaded": 0,
            "loading_errors": [],
        }
        # clip diagnostics
        self._clip_stats = {
            t: {"total": 0, "high": 0, "low": 0} for t in self.targets
        }

        # load and init
        try:
            self._load_models_and_scalers(model_dir, scaler_dir)
            self._initialize_history()
            self._preallocate_buffers()
            self._precompute_availability()

        except Exception as e:
            if self.fallback_mode:
                logging.warning(f"⚠️ Forecast generator init fallback: {e}")
                self._initialize_fallback_mode()
            else:
                raise

    # -------- init helpers --------

    def _initialize_fallback_mode(self):
        self.models.clear()
        self.scalers.clear()
        self._model_available.clear()
        self._initialize_history()
        self._preallocate_buffers()
        if self.verbose:
            print("✅ Fallback mode enabled (forecasts will use history/defaults)")

    def _load_models_and_scalers(self, model_dir: str, scaler_dir: str):
        if not os.path.exists(model_dir):
            msg = f"Model dir not found: {model_dir}"
            if self.fallback_mode:
                logging.warning(f"⚠️ {msg} (using fallback)")
                return
            raise ModelLoadingError(msg)
        if not os.path.exists(scaler_dir):
            msg = f"Scaler dir not found: {scaler_dir}"
            if self.fallback_mode:
                logging.warning(f"⚠️ {msg} (using fallback)")
                return
            raise ModelLoadingError(msg)

        # optional training summary
        summary_path = os.path.join(model_dir, "training_summary.json")
        self.training_summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    self.training_summary = json.load(f)
                if self.verbose:
                    print("✅ Loaded training summary")
            except Exception as e:
                logging.warning(f"⚠️ Could not load training summary: {e}")

        # load models/scalers
        for target in self.targets:
            for hname in self.horizons.keys():
                key = f"{target}_{hname}"
                self.loading_stats["models_attempted"] += 1
                model_path = os.path.join(model_dir, f"{key}_model.keras")
                if os.path.exists(model_path) and tf is not None:
                    try:
                        self.models[key] = tf.keras.models.load_model(model_path, compile=False)
                        self.loading_stats["models_loaded"] += 1
                        if self.verbose:
                            print(f"✅ model loaded: {key}")
                    except Exception as e:
                        self.loading_stats["loading_errors"].append(f"model {key}: {e}")
                        if self.verbose:
                            logging.error(f"❌ model load failed: {key} ({e})")
                else:
                    if self.verbose:
                        print(f"⚠️ model missing: {key}")

                # scalers (if present)
                self.loading_stats["scalers_attempted"] += 1
                sx = os.path.join(scaler_dir, f"{key}_scaler_X.pkl")
                sy = os.path.join(scaler_dir, f"{key}_scaler_y.pkl")
                if os.path.exists(sx) and os.path.exists(sy):
                    try:
                        scaler_X = joblib.load(sx)
                        scaler_y = joblib.load(sy)
                        if not hasattr(scaler_X, "transform") or not hasattr(scaler_y, "inverse_transform"):
                            raise ValueError("invalid scaler objects")
                        self.scalers[key] = {"scaler_X": scaler_X, "scaler_y": scaler_y}
                        self.loading_stats["scalers_loaded"] += 1
                        if self.verbose:
                            print(f"✅ scalers loaded: {key}")
                    except Exception as e:
                        self.loading_stats["loading_errors"].append(f"scalers {key}: {e}")
                        if self.verbose:
                            logging.error(f"❌ scalers load failed: {key} ({e})")

        if self.loading_stats["models_loaded"] == 0 and self.fallback_mode:
            print("⚠️ No models loaded; operating in fallback mode.")

    def _initialize_history(self):
        # compact, fast append/pop
        maxlen = self.look_back * 3
        self.history = {t: deque(maxlen=maxlen) for t in self.targets}

    def _preallocate_buffers(self):
        # one (1, look_back) array per target, reused every call
        self._X_buffers = {t: np.zeros((1, self.look_back), dtype=np.float32) for t in self.targets}

    def _precompute_availability(self):
        self._model_available = {
            f"{t}_{h}": (f"{t}_{h}" in self.models) for t in self.targets for h in self.horizons.keys()
        }

    # -------- bounded cache helpers --------

    def _bounded_put(self, d: Dict, key, value, limit: int):
        if len(d) >= limit:
            try:
                keys_to_drop = list(d.keys())[:max(1, int(limit * 0.6))]
                for k in keys_to_drop:
                    try:
                        del d[k]
                    except KeyError:
                        continue
            except Exception:
                d.clear()
        d[key] = value

    # -------- public utils --------

    def update(self, row: Mapping[str, Any]):
        """Feed one new row (dict/Series) to the rolling history. Fast path—no pandas ops."""
        if not isinstance(row, (dict, pd.Series)):
            if self.verbose:
                print(f"⚠️ update: unsupported row type {type(row)}")
            return
        for t in self.targets:
            try:
                if t in row:
                    v = float(row[t])
                    if np.isfinite(v):
                        self.history[t].append(v)
            except (ValueError, TypeError):
                # ignore bad values
                pass

    def reset_history(self):
        for t in self.targets:
            self.history[t].clear()
        if self.verbose:
            print("✅ history cleared")

    # -------- predict (fast paths) --------

    def predict_for_agent(self, agent: str, timestep: Optional[int] = None) -> Dict[str, float]:
        """
        Per-agent forecasts for current step, with caching + optional throttle.
        Returns keys like 'wind_forecast_immediate', ... exactly as the wrapper expects.
        """
        if agent == "risk_controller_0":
            return {}

        t = int(timestep or 0)
        cache_key = (agent, t)
        stride = self.agent_refresh_stride

        # throttle: if not on stride boundary AND we have recent cache, reuse last
        last_t = self._last_agent_step.get(agent, None)
        if last_t is not None and stride > 1 and t - last_t < stride:
            prev_key = (agent, last_t)
            if prev_key in self._agent_cache:
                return self._agent_cache[prev_key]

        # standard per-step cache
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        targets = self.agent_targets.get(agent, [])
        horizons = self.agent_horizons.get(agent, [])
        out: Dict[str, float] = {}

        for target in targets:
            for hname in horizons:
                k = f"{target}_forecast_{hname}"
                out[k] = self._predict_target_horizon(target, hname, t)

        # store cache (bounded)
        self._bounded_put(self._agent_cache, cache_key, out, self._cache_limit_agent)
        self._last_agent_step[agent] = t
        return out

    def predict_all_horizons(self, timestep: Optional[int] = None) -> Dict[str, float]:
        """
        Full forecast dict used by logging. Cached per (target,horizon,timestep).
        """
        t = int(timestep or 0)
        results: Dict[str, float] = {}

        for target in self.targets:
            for hname in self.horizons.keys():
                k = f"{target}_forecast_{hname}"
                results[k] = self._predict_target_horizon(target, hname, t)

        return results

    # legacy compatibility (kept)
    def predict(self, timestep: Optional[int] = None) -> Dict[str, float]:
        all_f = self.predict_all_horizons(timestep)
        compat = {}
        for target in self.targets:
            compat[f"{target}_forecast"] = all_f.get(f"{target}_forecast_immediate", self._fallback_value(target))
        compat.update(all_f)
        # ensure numeric
        for k, v in list(compat.items()):
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                target = k.split("_")[0]
                compat[k] = self._fallback_value(target)
        if self.verbose and (timestep or 0) <= 5:
            # short debug print grouped by horizon
            for h in self.horizons.keys():
                slice_h = {k.split("_")[0]: f"{v:.3f}" for k, v in compat.items() if k.endswith(f"_{h}")}
                if slice_h:
                    print(f"[debug] t={timestep} {h}: {slice_h}")
        return compat

    # -------- internals --------

    def _predict_target_horizon(self, target: str, hname: str, t: int) -> float:
        # 0) offline precomputed fast path (if available)
        arr = self._precomputed.get((target, hname), None)
        if arr is not None and 0 <= t < len(arr):
            val = arr[t]
            if np.isfinite(val):
                return float(val)

        # 1) cache
        gkey = (target, hname, t)
        if gkey in self._global_cache:
            return self._global_cache[gkey]

        model_key = f"{target}_{hname}"
        use_model = self._model_available.get(model_key, False)

        # 2) prepare input buffer in-place
        X = self._prepare_input_buffer(target)

        if use_model:
            try:
                Xs = self._scale_X(model_key, X)
                y_scaled = self.models[model_key].predict(Xs, verbose=0)
                # scalarize
                if isinstance(y_scaled, np.ndarray):
                    if y_scaled.size == 0:
                        raise ValueError("empty model output")
                    y_scaled = float(np.ravel(y_scaled)[0])
                else:
                    y_scaled = float(y_scaled)

                y = self._inverse_scale_y(model_key, y_scaled)
                y = self._constrain_target(target, y)
            except Exception:
                y = self._fallback_value(target)
        else:
            y = self._fallback_value(target)

        # bounded cache put
        self._bounded_put(self._global_cache, gkey, y, self._cache_limit_global)
        return y

    def _prepare_input_buffer(self, target: str) -> np.ndarray:
        """
        Fill self._X_buffers[target][0, :] with the last look_back values (padded).
        Returns the buffer (shape (1, look_back)).
        """
        buf = self._X_buffers[target]
        h = self.history[target]
        lb = self.look_back

        if len(h) == 0:
            buf[0, :].fill(self._default_for_target(target))
            return buf

        if len(h) < lb:
            # pad left with mean, then copy history to the right
            mean_val = float(np.mean(h))
            need = lb - len(h)
            if need > 0:
                buf[0, :need].fill(mean_val)
                lst = list(h)
                buf[0, need:lb] = np.asarray(lst, dtype=np.float32)
            else:
                buf[0, :] = np.asarray(h, dtype=np.float32)[-lb:]
        else:
            # fast path: copy last look_back
            lst = list(h)[-lb:]
            buf[0, :] = np.asarray(lst, dtype=np.float32)

        # clean NaNs
        np.nan_to_num(buf, copy=False, nan=self._default_for_target(target), posinf=1e6, neginf=-1e6)
        return buf

    def _scale_X(self, model_key: str, X: np.ndarray) -> np.ndarray:
        sc = self.scalers.get(model_key, {}).get("scaler_X", None)
        if sc is None:
            return X
        try:
            return sc.transform(X)
        except Exception:
            return X

    def _inverse_scale_y(self, model_key: str, y_scaled: float) -> float:
        scy = self.scalers.get(model_key, {}).get("scaler_y", None)
        if scy is None:
            return float(y_scaled)
        try:
            return float(scy.inverse_transform([[y_scaled]])[0, 0])
        except Exception:
            return float(y_scaled)

    # -------- constraints/fallbacks --------

    def _default_for_target(self, target: str) -> float:
        return {
            "wind": 0.3,
            "solar": 0.2,
            "hydro": 0.5,
            "price": 50.0,
            "load": 0.6,
        }.get(target, 0.0)

    def _constrain_target(self, target: str, val: float) -> float:
        """
        Do not apply any constraints or normalization; return raw value.
        Ensures non-finite values are handled correctly.
        """
        try:
            if not np.isfinite(val):
                return self._default_for_target(target)
            return float(val)
        except (ValueError, TypeError):
            return self._default_for_target(target)

    def _fallback_value(self, target: str) -> float:
        h = self.history.get(target, None)
        if h and len(h) > 0:
            v = float(np.mean(list(h)[-min(10, len(h)) :]))
            return self._constrain_target(target, v)
        return self._default_for_target(target)

    # -------- NEW: offline precompute --------

    def precompute_offline(self, df: pd.DataFrame, timestamp_col: str = "timestamp", batch_size: int = 4096) -> None:
        """
        Precompute forecasts for all targets/horizons across the entire dataframe.
        Stores results in self._precomputed[(target, hname)] as float32 arrays of length T.
        Assumes df has columns matching self.targets (e.g., 'wind','solar','hydro','price','load').

        At runtime, _predict_target_horizon() will return arr[t] (O(1)) if present.
        """
        if df is None or len(df) == 0:
            if self.verbose:
                print("precompute_offline: empty dataframe; skipping.")
            return

        # validate columns
        missing = [c for c in self.targets if c not in df.columns]
        if missing:
            raise ValueError(f"precompute_offline: missing columns in df: {missing}")

        T = len(df)
        look_back = int(self.look_back)

        # build rolling windows per target (T x look_back), mean-padded on left
        def make_windows(series: np.ndarray, lb: int) -> np.ndarray:
            X = np.empty((T, lb), dtype=np.float32)
            mean_val = float(np.nanmean(series)) if series.size else 0.0
            for t in range(T):
                if t + 1 < lb:
                    need = lb - (t + 1)
                    tail = series[: t + 1]
                    if tail.size > 0:
                        X[t, :need] = mean_val
                        X[t, need:lb] = tail.astype(np.float32)
                    else:
                        X[t, :].fill(mean_val)
                else:
                    X[t, :] = series[t + 1 - lb : t + 1].astype(np.float32)
            np.nan_to_num(X, copy=False, nan=mean_val, posinf=1e6, neginf=-1e6)
            return X

        target_series: Dict[str, np.ndarray] = {
            t: df[t].to_numpy(dtype=np.float32, copy=False) for t in self.targets
        }
        target_windows: Dict[str, np.ndarray] = {
            t: make_windows(target_series[t], look_back) for t in self.targets
        }

        # for each (target, horizon), run batched predict (or fast fallback)
        for target in self.targets:
            X_full = target_windows[target]
            for hname in self.horizons.keys():
                key = (target, hname)
                model_key = f"{target}_{hname}"
                use_model = bool(self._model_available.get(model_key, False))

                if not use_model or tf is None or model_key not in self.models:
                    # fallback: simple rolling-mean proxy (consistent with runtime fallback spirit)
                    out = np.nanmean(X_full, axis=1).astype(np.float32)
                    for i in range(T):
                        out[i] = self._constrain_target(target, out[i])
                    self._precomputed[key] = out
                    continue

                scaler_X = self.scalers.get(model_key, {}).get("scaler_X", None)
                scaler_y = self.scalers.get(model_key, {}).get("scaler_y", None)
                model = self.models.get(model_key, None)

                if model is None:
                    out = np.nanmean(X_full, axis=1).astype(np.float32)
                    for i in range(T):
                        out[i] = self._constrain_target(target, out[i])
                    self._precomputed[key] = out
                    continue

                X_scaled = X_full if scaler_X is None else scaler_X.transform(X_full)

                bs = max(1, int(batch_size))
                preds_scaled = np.empty(T, dtype=np.float32)
                i = 0
                while i < T:
                    j = min(T, i + bs)
                    yb = model.predict(X_scaled[i:j, :], verbose=0)
                    # flatten to 1D
                    if isinstance(yb, np.ndarray):
                        if yb.ndim == 2 and yb.shape[1] == 1:
                            yb = yb[:, 0]
                        else:
                            yb = np.ravel(yb)
                    preds_scaled[i:j] = np.asarray(yb, dtype=np.float32)
                    i = j

                if scaler_y is not None:
                    try:
                        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))[:, 0].astype(np.float32)
                    except Exception:
                        preds = preds_scaled
                else:
                    preds = preds_scaled

                # constrain/clean
                for i in range(T):
                    preds[i] = self._constrain_target(target, preds[i])

                self._precomputed[key] = preds

        if self.verbose:
            total_series = len(self.targets) * len(self.horizons)
            print(f"✅ Precompute complete: cached {total_series} series for T={T} timesteps.")

    # -------- diagnostics & metadata --------

    def get_agent_forecast_dims(self) -> Dict[str, int]:
        return {
            agent: len(self.agent_targets.get(agent, [])) * len(self.agent_horizons.get(agent, []))
            for agent in self.agent_horizons
        }

    def get_loading_stats(self) -> Dict[str, Any]:
        return {
            "models_loaded": self.loading_stats["models_loaded"],
            "models_attempted": self.loading_stats["models_attempted"],
            "scalers_loaded": self.loading_stats["scalers_loaded"],
            "scalers_attempted": self.loading_stats["scalers_attempted"],
            "loading_errors": self.loading_stats["loading_errors"],
            "success_rate": (
                (self.loading_stats["models_loaded"] / max(1, self.loading_stats["models_attempted"])) * 100.0
            ),
            "fallback_mode": len(self.models) == 0,
        }

    def get_clip_stats(self) -> Dict[str, Dict[str, int]]:
        """Return pre-clip hit rates per renewable/load target."""
        return {t: dict(v) for t, v in self._clip_stats.items() if t in {"wind", "solar", "hydro", "load"}}

    def get_forecast_summary(self):
        print("\n=== Multi-Horizon Forecast Generator Summary ===")
        print(f"look_back={self.look_back}, horizons={self.horizons}")
        print(f"targets={self.targets}")
        stats = self.get_loading_stats()
        print(f"models: {stats['models_loaded']}/{stats['models_attempted']} (success {stats['success_rate']:.1f}%)")
        print(f"fallback: {'yes' if stats['fallback_mode'] else 'no'}")
        print("\nAgent assignments:")
        for a in self.agent_horizons:
            Ts = self.agent_targets.get(a, [])
            Hs = self.agent_horizons.get(a, [])
            if a == "risk_controller_0":
                print(f"  {a}: no forecasts (enhanced risk)")
            else:
                print(f"  {a}: {Ts} × {Hs} = {len(Ts)*len(Hs)}")
        print("\nModel availability:")
        for t in self.targets:
            avail = [h for h in self.horizons if self._model_available.get(f"{t}_{h}", False)]
            badge = "✅" if avail else "❌"
            print(f"  {t}: {badge} {avail}")
        if self.loading_stats["loading_errors"]:
            print(f"\n⚠️ {len(self.loading_stats['loading_errors'])} loading errors (showing up to 5):")
            for e in self.loading_stats["loading_errors"][:5]:
                print(f"  • {e}")

        # Clip diagnostics summary (if any samples)
        cs = self.get_clip_stats()
        if any(v["total"] > 0 for v in cs.values()):
            print("\nClip diagnostics (renewables/load):")
            for t, d in cs.items():
                tot = max(1, d.get("total", 0))
                hi = SafeDivision._safe_divide(d.get("high", 0) * 100.0, tot)
                lo = SafeDivision._safe_divide(d.get("low", 0) * 100.0, tot)
                print(f"  {t:6s}: total={tot}, high@1.0={hi:.1f}%, low@0.0={lo:.1f}%")

        print("=" * 60 + "\n")

    def get_enhanced_agent_info(self) -> Dict[str, Any]:
        info = {
            "forecast_agents": {},
            "risk_agent": {
                "risk_controller_0": {
                    "forecasts": 0,
                    "enhanced_metrics": 6,
                    "risk_dimensions": [
                        "market_risk", "operational_risk", "portfolio_risk",
                        "liquidity_risk", "regulatory_risk", "overall_risk"
                    ],
                    "description": "Uses comprehensive risk assessment instead of forecasts",
                }
            },
            "loading_stats": self.get_loading_stats(),
        }
        for a in self.agent_horizons:
            if a == "risk_controller_0":
                continue
            Ts = self.agent_targets.get(a, [])
            Hs = self.agent_horizons.get(a, [])
            required = len(Ts) * len(Hs)
            available = sum(1 for t in Ts for h in Hs if self._model_available.get(f"{t}_{h}", False))
            info["forecast_agents"][a] = {
                "targets": Ts,
                "horizons": Hs,
                "total_forecasts": required,
                "available_models": available,
                "model_availability": (available / max(1, required)) * 100.0,
                "forecast_keys": [f"{t}_forecast_{h}" for t in Ts for h in Hs],
            }
        return info

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "tensorflow_available": tf is not None,
            "models_loaded": len(self.models),
            "scalers_loaded": len(self.scalers),
            "targets_tracked": len(self.targets),
            "history_status": {t: len(self.history[t]) for t in self.targets},
            "loading_stats": self.get_loading_stats(),
            "fallback_mode": len(self.models) == 0,
            "agent_forecast_dims": self.get_agent_forecast_dims(),
            "agent_refresh_stride": self.agent_refresh_stride,
            "clip_stats": self.get_clip_stats(),
            "cache_sizes": {
                "global": len(self._global_cache),
                "agent": len(self._agent_cache),
            }
        }

    def validate_system_integrity(self) -> bool:
        issues = []
        if tf is None:
            issues.append("TensorFlow not available")
        if len(self.models) == 0:
            issues.append("No models loaded (fallback mode)")
        for a in self.agent_horizons:
            if a == "risk_controller_0":
                continue
            Ts = self.agent_targets.get(a, [])
            Hs = self.agent_horizons.get(a, [])
            req = len(Ts) * len(Hs)
            have = sum(1 for t in Ts for h in Hs if self._model_available.get(f"{t}_{h}", False))
            if have < req:
                issues.append(f"{a}: {have}/{req} models available")
        if issues:
            print("⚠️ Integrity issues:")
            for m in issues:
                print("  •", m)
            return False
        print("✅ System integrity OK")
        return True

    def __str__(self):
        s = self.get_loading_stats()
        return (f"MultiHorizonForecastGenerator("
                f"models={s['models_loaded']}/{s['models_attempted']}, "
                f"targets={len(self.targets)}, horizons={len(self.horizons)}, "
                f"fallback={'Yes' if s['fallback_mode'] else 'No'}, "
                f"stride={self.agent_refresh_stride})")

    __repr__ = __str__


# =========================
# Quick self-test
# =========================

def test_forecast_generator():
    print("🧪 Testing per-agent forecaster…")
    gen = MultiHorizonForecastGenerator(
        model_dir="non_existent_models",
        scaler_dir="non_existent_scalers",
        look_back=6,
        verbose=True,
        fallback_mode=True,
        agent_refresh_stride=3,  # demonstrate throttle (every 3 steps per agent)
    )

    # feed a few rows
    for _ in range(8):
        gen.update({"wind": 0.5, "solar": 0.3, "hydro": 0.7, "price": 60.0, "load": 0.65})

    # per-agent predictions (should cache + throttle)
    for t in range(6):
        for a in ["investor_0", "battery_operator_0", "meta_controller_0", "risk_controller_0"]:
            out = gen.predict_for_agent(a, timestep=t)
            if a != "risk_controller_0":
                assert len(out) == len(gen.agent_targets[a]) * len(gen.agent_horizons[a])
            else:
                assert out == {}
        # global log forecasts once in a while
        if t % 2 == 0:
            full = gen.predict_all_horizons(timestep=t)
            # ensure immediate keys exist
            for k in ["wind", "solar", "price", "load", "hydro"]:
                assert f"{k}_forecast_immediate" in full

    # check clip stats structure
    cs = gen.get_clip_stats()
    assert isinstance(cs, dict)

    print("🎉 per-agent forecaster OK")
    return True


if __name__ == "__main__":
    test_forecast_generator()
