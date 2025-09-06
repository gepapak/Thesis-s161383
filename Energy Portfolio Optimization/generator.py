import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: F401 (used when loaded from disk)
import json
from typing import Dict, List, Optional, Tuple, Any, Mapping
import pandas as pd
from collections import deque, OrderedDict
import logging

# =========================
# TensorFlow setup (optional)
# =========================

def get_gpu_memory_info():
    """Get available GPU memory information."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None

        # Get memory info for the first GPU
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        if 'device_name' in gpu_details:
            # Try to get memory info from nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    'total_mb': info.total // (1024 * 1024),
                    'free_mb': info.free // (1024 * 1024),
                    'used_mb': info.used // (1024 * 1024)
                }
            except ImportError:
                # Fallback: assume common GPU memory sizes
                return {'total_mb': 8192, 'free_mb': 6144, 'used_mb': 2048}
        return None
    except Exception:
        return None

def fix_tensorflow_gpu_setup():
    """Enhanced TF GPU config with dynamic memory allocation."""
    try:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        os.environ.setdefault("TF_MEMORY_GROWTH", "true")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        import tensorflow as tf  # noqa

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")

            # Get memory information
            memory_info = get_gpu_memory_info()

            for i, gpu in enumerate(gpus):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ Enabled memory growth for GPU {i}")
                except Exception as e:
                    print(f"⚠️ Failed to set memory growth for GPU {i}: {e}")

            try:
                # Dynamic memory limit calculation
                if memory_info:
                    # Use 70% of available memory, minimum 2GB, maximum 8GB
                    available_mb = memory_info['free_mb']
                    memory_limit = max(2048, min(8192, int(available_mb * 0.7)))
                    print(f"GPU Memory - Total: {memory_info['total_mb']}MB, "
                          f"Free: {memory_info['free_mb']}MB, Setting limit: {memory_limit}MB")
                else:
                    # Conservative fallback
                    memory_limit = 3072
                    print(f"⚠️ Could not detect GPU memory, using conservative limit: {memory_limit}MB")

                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
                print(f"✅ Set GPU memory limit to {memory_limit}MB")

            except Exception as e:
                print(f"⚠️ Failed to set memory limit: {e}")
                # Continue without memory limit
        else:
            print("No GPUs found, using CPU")

        tf.get_logger().setLevel('ERROR')
        return tf
    except Exception as e:
        print(f"❌ TensorFlow GPU setup failed: {e}")
        return None

tf = fix_tensorflow_gpu_setup()


# =========================
# Memory Management Utilities
# =========================

class LRUCache:
    """Lightweight LRU cache implementation for forecast caching."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        """Get value and move to end (most recently used)."""
        if key in self.cache:
            # Move to end
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        """Put value, evicting LRU if necessary."""
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Evict LRU (first item)
            self.cache.popitem(last=False)

        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def clear(self):
        self.cache.clear()

    def keys(self):
        return self.cache.keys()


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
# Forecast Validation
# =========================

class ForecastValidator:
    """Validates forecast consistency and tracks accuracy metrics."""

    def __init__(self, tolerance=0.15, window_size=50):
        self.tolerance = tolerance
        self.window_size = window_size
        self.forecast_history = {}  # (target, horizon) -> deque of forecasts
        self.accuracy_history = {}  # (target, horizon) -> deque of errors
        self.validation_errors = []

    def validate_forecast_consistency(self, target: str, horizon: str,
                                    current_forecast: float, timestep: int) -> bool:
        """Check if current forecast is consistent with recent history."""
        key = (target, horizon)

        if key not in self.forecast_history:
            from collections import deque
            self.forecast_history[key] = deque(maxlen=self.window_size)
            self.accuracy_history[key] = deque(maxlen=self.window_size)

        history = self.forecast_history[key]

        if len(history) < 3:  # Need some history to validate
            history.append((timestep, current_forecast))
            return True

        # Check for unrealistic jumps
        recent_forecasts = [f for _, f in list(history)[-3:]]
        recent_mean = np.mean(recent_forecasts)
        recent_std = np.std(recent_forecasts) + 1e-6  # Avoid division by zero

        # Z-score based validation
        z_score = abs(current_forecast - recent_mean) / recent_std
        is_valid = z_score <= 3.0  # 3-sigma rule

        # Relative change validation
        last_forecast = history[-1][1]
        if abs(last_forecast) > 1e-6:
            relative_change = abs(current_forecast - last_forecast) / abs(last_forecast)
            is_valid = is_valid and (relative_change <= self.tolerance)

        if not is_valid:
            self.validation_errors.append({
                'timestep': timestep,
                'target': target,
                'horizon': horizon,
                'forecast': current_forecast,
                'recent_mean': recent_mean,
                'z_score': z_score,
                'relative_change': relative_change if abs(last_forecast) > 1e-6 else 0.0
            })

        history.append((timestep, current_forecast))
        return is_valid

    def update_accuracy(self, target: str, horizon: str, forecast: float, actual: float):
        """Update accuracy tracking for a target/horizon combination."""
        key = (target, horizon)
        if key not in self.accuracy_history:
            from collections import deque
            self.accuracy_history[key] = deque(maxlen=self.window_size)

        error = abs(forecast - actual) / (abs(actual) + 1e-6)  # MAPE
        self.accuracy_history[key].append(error)

    def get_accuracy_stats(self, target: str, horizon: str) -> dict:
        """Get accuracy statistics for a target/horizon combination."""
        key = (target, horizon)
        if key not in self.accuracy_history or len(self.accuracy_history[key]) == 0:
            return {'mean_error': 0.0, 'std_error': 0.0, 'samples': 0}

        errors = list(self.accuracy_history[key])
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'samples': len(errors)
        }


# =========================
# Forecast Generator
# =========================

class MultiHorizonForecastGenerator:
    """
    Simplified multi-horizon forecaster for raw MW units.

    Key features:
      - Direct raw MW forecasting (no normalization complexity)
      - Simple caching for performance
      - Graceful fallback when models unavailable
      - Pre-computed forecasts for training efficiency

    This forecaster expects and produces raw MW values:
      - Wind: 0-1103 MW
      - Solar: 0-100 MW
      - Hydro: 0-534 MW
      - Load: 0-2999 MW
      - Price: $/MWh (no conversion)
    """

    def __init__(
        self,
        model_dir: str = "saved_models",
        scaler_dir: str = "saved_scalers",
        look_back: int = 6,
        verbose: bool = True,
        fallback_mode: bool = True,
        # Simple refresh - no throttling complexity
        agent_refresh_stride: int = 1,
    ):
        self.look_back = int(look_back)
        self.verbose = verbose
        self.fallback_mode = fallback_mode
        self.agent_refresh_stride = max(1, int(agent_refresh_stride))

        # Simplified - no complex validation
        self.enable_validation = False
        self.validator = None

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

        # caches (LRU-based for better memory management)
        self._global_cache = LRUCache(max_size=8000)  # Increased size for better hit rate
        self._agent_cache = LRUCache(max_size=4000)   # Increased size for better hit rate
        self._last_agent_step: Dict[str, int] = {}

        # Memory management
        self._step_counter = 0
        self._cleanup_frequency = 200  # Cleanup every 200 steps

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
                # Try both .h5 and .keras formats
                model_path_h5 = os.path.join(model_dir, f"{key}_model.h5")
                model_path_keras = os.path.join(model_dir, f"{key}_model.keras")
                model_path = model_path_h5 if os.path.exists(model_path_h5) else model_path_keras
                if os.path.exists(model_path) and tf is not None:
                    try:
                        # Try multiple loading strategies for compatibility
                        try:
                            # Method 1: Standard loading
                            self.models[key] = tf.keras.models.load_model(model_path, compile=False)
                        except Exception as e1:
                            # Method 2: Load with custom objects (for compatibility)
                            try:
                                self.models[key] = tf.keras.models.load_model(
                                    model_path,
                                    compile=False,
                                    custom_objects=None
                                )
                            except Exception as e2:
                                # Method 3: Try loading architecture + weights separately
                                arch_path = model_path.replace('.h5', '_architecture.json')
                                weights_path = model_path.replace('.h5', '_weights.h5')
                                if os.path.exists(arch_path) and os.path.exists(weights_path):
                                    with open(arch_path, 'r') as f:
                                        model_json = f.read()
                                    model = tf.keras.models.model_from_json(model_json)
                                    model.load_weights(weights_path)
                                    self.models[key] = model
                                else:
                                    raise e2

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

                # scalers (if present) - try both naming conventions
                self.loading_stats["scalers_attempted"] += 1
                # Try TestForecast naming convention first
                sx = os.path.join(scaler_dir, f"{key}_sc_X.pkl")
                sy = os.path.join(scaler_dir, f"{key}_sc_y.pkl")
                # Fallback to original naming convention
                if not (os.path.exists(sx) and os.path.exists(sy)):
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

    # -------- memory management helpers --------

    def _cleanup_memory(self):
        """Periodic memory cleanup to prevent leaks."""
        try:
            import gc

            # Force garbage collection
            gc.collect()

            # Clear validation errors if too many
            if self.validator and len(self.validator.validation_errors) > 100:
                self.validator.validation_errors = self.validator.validation_errors[-50:]

            if self.verbose and self._step_counter % (self._cleanup_frequency * 10) == 0:
                print(f"🧹 Memory cleanup at step {self._step_counter}: "
                      f"Global cache: {len(self._global_cache)}, "
                      f"Agent cache: {len(self._agent_cache)}")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Memory cleanup failed: {e}")

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

                        # NEW: Update forecast accuracy if validator is enabled
                        if self.validator:
                            # Check if we have recent forecasts to compare against
                            for hname in self.horizons.keys():
                                key = (t, hname)
                                if key in self.validator.forecast_history:
                                    history = self.validator.forecast_history[key]
                                    if len(history) > 0:
                                        # Find the most recent forecast for this target/horizon
                                        recent_forecast = history[-1][1]
                                        self.validator.update_accuracy(t, hname, recent_forecast, v)

                                        # Update enhanced monitoring if available
                                        if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
                                            self.enhanced_monitor.update_forecast_accuracy(t, hname, recent_forecast, v)
            except (ValueError, TypeError):
                # ignore bad values
                pass

    def reset_history(self):
        for t in self.targets:
            self.history[t].clear()
        if self.verbose:
            print("✅ history cleared")

    def initialize_history(self, data: pd.DataFrame, start_idx: int = 0):
        """Initialize history with sufficient data for predictions.

        Args:
            data: DataFrame with historical data
            start_idx: Starting index in data to begin history initialization
        """
        if self.verbose:
            print(f"🔄 Initializing forecaster history from data...")

        # Pre-populate history with look_back worth of data
        end_idx = min(start_idx + self.look_back, len(data))

        for i in range(start_idx, end_idx):
            row = data.iloc[i]
            self.update(row)

        if self.verbose:
            print(f"✅ History initialized with {end_idx - start_idx} data points")
            for target in self.targets:
                if target in self.history:
                    hist_len = len(self.history[target])
                    print(f"   {target}: {hist_len} items")

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
            cached_result = self._agent_cache.get(prev_key)
            if cached_result is not None:
                return cached_result

        # standard per-step cache
        cached_result = self._agent_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        targets = self.agent_targets.get(agent, [])
        horizons = self.agent_horizons.get(agent, [])
        out: Dict[str, float] = {}

        for target in targets:
            for hname in horizons:
                k = f"{target}_forecast_{hname}"
                out[k] = self._predict_target_horizon(target, hname, t)

        # store cache with LRU eviction
        self._agent_cache.put(cache_key, out)
        self._last_agent_step[agent] = t

        # Periodic memory cleanup
        self._step_counter += 1
        if self._step_counter % self._cleanup_frequency == 0:
            self._cleanup_memory()

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
        """
        Simplified prediction that works directly with raw MW units.
        No complex normalization - models trained on raw data, return raw data.
        """
        # 1) Check precomputed cache first (fastest path)
        arr = self._precomputed.get((target, hname), None)
        if arr is not None and 0 <= t < len(arr):
            val = arr[t]
            if np.isfinite(val):
                return float(val)

        # 2) Check runtime cache
        gkey = (target, hname, t)
        cached_value = self._global_cache.get(gkey)
        if cached_value is not None:
            return cached_value

        # 3) Model prediction (simplified)
        model_key = f"{target}_{hname}"
        use_model = self._model_available.get(model_key, False)

        if use_model and model_key in self.models:
            try:
                # Prepare input (raw values)
                X = self._prepare_input_buffer(target)

                # Scale input if scaler exists (but expect raw output)
                Xs = self._scale_X(model_key, X) if self.scalers.get(model_key, {}).get("scaler_X") else X

                # Get prediction
                y_pred = self.models[model_key].predict(Xs, verbose=0)

                # Extract scalar value
                if isinstance(y_pred, np.ndarray):
                    y_pred = float(np.ravel(y_pred)[0]) if y_pred.size > 0 else 0.0
                else:
                    y_pred = float(y_pred)

                # Inverse scale if needed (but models should output raw units)
                y = self._inverse_scale_y(model_key, y_pred)

                # Simple bounds check
                y = self._constrain_target(target, y)

            except Exception as e:
                if self.verbose:
                    print(f"Model prediction failed for {model_key}: {e}")
                y = self._fallback_value(target)
        else:
            y = self._fallback_value(target)

        # Cache result
        self._global_cache.put(gkey, y)
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
        """Default values in raw MW units based on typical capacity."""
        return {
            "wind": 330.0,    # ~30% of 1103 MW capacity
            "solar": 20.0,    # ~20% of 100 MW capacity
            "hydro": 267.0,   # ~50% of 534 MW capacity
            "price": 50.0,    # $/MWh (unchanged)
            "load": 1800.0,   # ~60% of 2999 MW capacity
        }.get(target, 0.0)

    def _constrain_target(self, target: str, val: float) -> float:
        """
        Apply realistic bounds for raw MW values.
        Much simpler than previous normalization approach.
        """
        try:
            if not np.isfinite(val):
                return self._default_for_target(target)

            result = float(val)

            # Apply realistic bounds based on actual data range
            bounds = {
                "wind": (0.0, 1600.0),    # 0 to max observed (1500) + buffer
                "solar": (0.0, 1100.0),   # 0 to max observed (1000) + buffer
                "hydro": (0.0, 1100.0),   # 0 to max observed (1000) + buffer
                "price": (-50.0, 500.0),  # Reasonable price range
                "load": (500.0, 3500.0),  # Reasonable load range
            }

            if target in bounds:
                min_val, max_val = bounds[target]
                result = max(min_val, min(max_val, result))

            return result

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

                for start in range(0, T, bs):
                    end = min(start + bs, T)
                    batch_X = X_scaled[start:end]
                    try:
                        batch_preds = model.predict(batch_X, verbose=0)
                        if isinstance(batch_preds, np.ndarray):
                            batch_preds = np.ravel(batch_preds)
                        preds_scaled[start:end] = batch_preds[:end - start]
                    except Exception:
                        preds_scaled[start:end] = np.nanmean(X_full[start:end], axis=1)

                if scaler_y is not None:
                    try:
                        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))[:, 0].astype(np.float32)
                    except Exception:
                        preds = preds_scaled
                else:
                    preds = preds_scaled

                for i in range(T):
                    preds[i] = self._constrain_target(target, preds[i])

                self._precomputed[key] = preds

        if self.verbose:
            total_forecasts = len(self.targets) * len(self.horizons)
            print(f"Offline precompute complete: {total_forecasts} forecast series cached.")



    def _create_trend_forecasts(self, series: np.ndarray, target: str, hname: str, T: int) -> np.ndarray:
        """Create simple trend-based forecasts as fallback."""
        forecasts = np.zeros(T, dtype=np.float32)

        # Simple moving average with slight trend
        window = min(24, len(series) // 4) if len(series) > 4 else len(series)

        for t in range(T):
            if t == 0:
                forecasts[t] = self._default_for_target(target)
            elif t < window:
                # Use available history
                forecasts[t] = np.mean(series[:t])
            else:
                # Use recent window with slight trend
                recent = series[t-window:t]
                base_forecast = np.mean(recent)

                # Add small trend component
                if len(recent) > 1:
                    trend = (recent[-1] - recent[0]) / len(recent)
                    base_forecast += trend * 0.1  # Damped trend

                forecasts[t] = base_forecast

            # Apply constraints
            forecasts[t] = self._constrain_target(target, forecasts[t])

        return forecasts

    def _create_fallback_precomputed(self, T: int) -> None:
        """Create fallback forecasts when data is missing."""
        for target in self.targets:
            for hname in self.horizons.keys():
                key = (target, hname)
                # Simple constant forecasts
                default_val = self._default_for_target(target)
                self._precomputed[key] = np.full(T, default_val, dtype=np.float32)

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
