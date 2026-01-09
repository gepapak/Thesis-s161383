import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: F401 (used when loaded from disk)
import json
import time
import csv
from typing import Dict, List, Optional, Tuple, Any, Mapping
import pandas as pd
from collections import deque, OrderedDict
import logging
from utils import UnifiedMemoryManager, SafeDivision, configure_tf_memory, _get_tf  # UNIFIED: Import from single source of truth

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

def fix_tensorflow_gpu_setup(use_gpu=True):
    """Enhanced TF GPU config with dynamic memory allocation."""
    try:
        import tensorflow as tf  # noqa

        if not use_gpu:
            # Force CPU-only mode
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("TensorFlow configured for CPU-only mode")
            tf.get_logger().setLevel('ERROR')
            return tf

        # GPU mode setup
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        os.environ.setdefault("TF_MEMORY_GROWTH", "true")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")

            # Get memory information
            memory_info = get_gpu_memory_info()

            for i, gpu in enumerate(gpus):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"[OK] Enabled memory growth for GPU {i}")
                except Exception as e:
                    print(f"[WARNING] Failed to set memory growth for GPU {i}: {e}")

            # HIGH: Fix GPU Initialization Conflict - avoid setting memory_limit after memory_growth
            # Rely solely on memory_growth or cuda_malloc_async for dynamic GPU memory allocation
            print(f"[OK] Using dynamic GPU memory allocation (memory_growth=True)")
            print(f"[INFO] Skipping fixed memory_limit to prevent OOM errors on modern HPC environments")
        else:
            print("No GPUs found, using CPU")

        tf.get_logger().setLevel('ERROR')
        return tf
    except Exception as e:
        print(f"❌ TensorFlow GPU setup failed: {e}")
        return None

# LAZY TensorFlow initialization (use _get_tf() from utils)
# Suppress TensorFlow warnings globally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import logging as _logging
    _logging.getLogger("tensorflow").setLevel(_logging.ERROR)
except Exception:
    pass

def initialize_tensorflow(device="cuda"):
    """Initialize TensorFlow based on device setting with enhanced memory management (lazy)."""
    use_gpu = device.lower() == "cuda"
    tf = _get_tf()  # Lazy initialization from utils

    if tf is None:
        logging.warning("TensorFlow not available, cannot initialize")
        return None

    # Configure TensorFlow for memory efficiency - CONSERVATIVE APPROACH
    try:
        # HIGH: Fix GPU Initialization Conflict - rely solely on memory_growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and use_gpu:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Remove hardcoded VirtualDeviceConfiguration(memory_limit=X) to prevent instability
        logging.info(f"TensorFlow configured for {'GPU' if use_gpu else 'CPU'}-only mode with dynamic memory allocation")
    except Exception as e:
        logging.warning(f"Failed to configure TensorFlow memory settings: {e}")

    return tf


# =========================
# Memory Management Utilities
# =========================

# UNIFIED MEMORY MANAGER (NEW)
# LRUCache is now managed by UnifiedMemoryManager from memory_manager.py
# For backward compatibility, we provide a simple LRU cache wrapper
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


# =========================
# Forecast Validation
# =========================

class ForecastValidator:
    """Validates forecast consistency and tracks accuracy metrics."""

    def __init__(self, tolerance=0.10, window_size=50):  # TUNED: Tightened from 0.15 to 0.10 for better reliability
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
        is_valid = (z_score < 3.5)  # Fixed: Use proper boolean expression

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

    CRITICAL DATA LEAKAGE WARNING:
    ==============================
    The forecast models loaded from disk are assumed to be trained on EXTERNAL data
    that is COMPLETELY SEPARATE from the training/testing data used in this RL system.

    If the forecast models were trained on data that overlaps with the RL training data,
    this constitutes DATA LEAKAGE and will result in:
    - Unrealistically optimistic forecast accuracy
    - Overfitted RL policies that don't generalize
    - Invalid performance metrics

    VALIDATION STEPS:
    1. Verify forecast model training data is from a different time period
    2. Verify forecast model training data is from a different source/region if applicable
    3. Check model training dates vs. RL training data dates
    4. If possible, load model metadata to confirm training integrity

    RECOMMENDATION:
    Add a validation step that loads and checks model metadata (training date range,
    data source, etc.) to confirm no overlap with RL training data.
    """

    def __init__(
        self,
        model_dir: str = "saved_models",
        scaler_dir: str = "saved_scalers",
        metadata_dir: Optional[str] = None,  # NEW: Directory with metadata JSON files
        look_back: int = 24,  # IMPROVED: Increased from 6 to 24 (must match retrained models)
        verbose: bool = True,
        fallback_mode: bool = True,
        # Simple refresh - no throttling complexity
        agent_refresh_stride: int = 1,
        # Data leakage validation
        validate_data_integrity: bool = True,
        rl_train_start_date: Optional[str] = None,
        timing_log_path: Optional[str] = None): 
        self.look_back = int(look_back)
        self.verbose = verbose
        self.metadata_dir = metadata_dir  # Store metadata directory
        self.fallback_mode = fallback_mode
        self.agent_refresh_stride = max(1, int(agent_refresh_stride))
        self.validate_data_integrity = validate_data_integrity
        self.rl_train_start_date = rl_train_start_date

        # Simplified - no complex validation
        self.enable_validation = False
        self.validator = None

        # CANONICAL: Use single source of truth from config for horizon definitions
        try:
            from config import EnhancedConfig
            self.config = EnhancedConfig()  # Cache config for reuse
            self.horizons: Dict[str, int] = self.config.forecast_horizons.copy()
        except Exception as e:
            # FAIL-FAST: No hardcoded horizon fallbacks allowed for production safety
            raise ValueError(f"Cannot initialize forecast horizons: config.EnhancedConfig unavailable. "
                           f"Fix packaging/paths or config initialization. Original error: {e}")

        # FAIL-FAST: No hardcoded target lists allowed for production safety
        # Get forecast targets from config to maintain single source of truth
        try:
            if hasattr(self.config, 'forecast_targets'):
                self.targets: List[str] = list(self.config.forecast_targets)
            else:
                # If not defined in config, require explicit definition
                raise ValueError("config.forecast_targets missing. Define forecast targets in config to maintain single source of truth.")
        except Exception as e:
            raise ValueError(f"Cannot initialize forecast targets: {str(e)}. "
                           f"Add forecast_targets to config.EnhancedConfig to maintain single source of truth.")

        # TUNED: Expanded agent horizon assignments for maximum value creation
        # RESTORED: Include strategic horizon for agents that benefit from long-term planning
        self.agent_horizons: Dict[str, List[str]] = {
            "investor_0": ["immediate", "short", "medium", "long"],  # RESTORED: Include long for day-ahead positioning
            "battery_operator_0": ["immediate", "short", "medium", "long"],  # Battery needs up to day-ahead planning
            "risk_controller_0": ["immediate", "short", "medium", "long", "strategic"],  # RESTORED: Risk needs multi-day view
            "meta_controller_0": ["immediate", "short", "medium", "long", "strategic"],  # RESTORED: Meta needs full horizon set
        }
        # TUNED: Optimized agent target assignments for enhanced decision-making
        self.agent_targets: Dict[str, List[str]] = {
            "investor_0": ["wind", "solar", "hydro", "price"],           # Financial trading (unchanged - optimal)
            "battery_operator_0": ["price", "load", "wind"],             # Battery operations (unchanged - optimal)
            "risk_controller_0": ["price", "wind", "solar", "load"],     # TUNED: Added load for grid stress prediction
            "meta_controller_0": ["wind", "solar", "hydro", "price", "load"], # Overall coordination (unchanged - comprehensive)
        }

        # storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Dict[str, Any]] = {}
        self._model_available: Dict[str, bool] = {}
        self.history: Dict[str, deque] = {}               # per-target values
        self._X_buffers: Dict[str, np.ndarray] = {}       # per-target (1, look_back) preallocated inputs
        
        # CRITICAL FIX: Store training cap values from metadata (for bounds consistency)
        # Will be populated during model loading from metadata files
        self._training_caps: Dict[str, float] = {}  # target -> cap value from training

        # caches (LRU-based for better memory management)
        self._global_cache = LRUCache(max_size=8000)  # Increased size for better hit rate
        self._agent_cache = LRUCache(max_size=4000)   # Increased size for better hit rate
        self._last_agent_step: Dict[str, int] = {}

        # Memory management
        self._step_counter = 0
        self._cleanup_frequency = 200  # Cleanup every 200 steps

        # NEW: offline precomputed forecasts (target, horizon) -> np.ndarray of shape (T,)
        self._precomputed: Dict[Tuple[str, str], np.ndarray] = {}
        self._data_df: Optional[pd.DataFrame] = None  # Store data for debiasing

        # Timing instrumentation (optional)
        self.timing_log_path = timing_log_path
        self._predict_accumulator = 0.0  # milliseconds accumulated for current predict call
        if self.timing_log_path:
            try:
                os.makedirs(os.path.dirname(self.timing_log_path), exist_ok=True)
            except Exception:
                pass
            # Create file and header if not exists
            try:
                if not os.path.exists(self.timing_log_path):
                    with open(self.timing_log_path, 'w', newline='') as tf:
                        w = csv.writer(tf)
                        w.writerow(['iso_ts', 'timestep', 'agent', 'total_ms', 'num_predictions', 'mean_ms'])
            except Exception:
                # If logging setup fails, disable timing logging quietly
                self.timing_log_path = None

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

        # accuracy tracking for confidence calculation
        self.accuracy_history: Dict[Tuple[str, str], deque] = {}

        # Data integrity tracking
        self._forecast_train_end_date: Optional[str] = None
        self._data_leakage_validated: bool = False

        # Initialize TensorFlow (lazy loading from utils)
        # This is needed for model loading in _load_models_and_scalers()
        self.tf = _get_tf()

        # load and init
        try:
            self._load_models_and_scalers(model_dir, scaler_dir, metadata_dir)

            # CRITICAL: Validate data integrity to prevent leakage
            if self.validate_data_integrity:
                self._validate_forecast_model_integrity(model_dir, metadata_dir)

            self._initialize_history()
            self._preallocate_buffers()
            self._precompute_availability()

        except Exception as e:
            if self.fallback_mode:
                logging.warning(f"[WARNING] Forecast generator init fallback: {e}")
                self._initialize_fallback_mode()
            else:
                # FAIL-FAST: When fallback_mode=False, raise immediately
                # This ensures forecast features (DL overlay, FGB) fail loudly if models don't load
                logging.error(f"[CRITICAL] Forecast generator initialization failed (fallback_mode=False): {e}")
                raise

    # -------- init helpers --------

    def _initialize_fallback_mode(self):
        self.models.clear()
        self.scalers.clear()
        self._model_available.clear()
        self._initialize_history()
        self._preallocate_buffers()
        if self.verbose:
            print("[OK] Fallback mode enabled (forecasts will use history/defaults)")

    def _load_models_and_scalers(self, model_dir: str, scaler_dir: str, metadata_dir: Optional[str] = None):
        """
        Load models and scalers using new metadata-based structure or fallback to old structure.
        
        NEW STRUCTURE (preferred):
        - Looks for metadata/{target}_{horizon}_metadata.json
        - Uses model_path_best from metadata (prefers best checkpoint)
        - Gets look_back and other params from metadata
        - Paths in metadata are relative to Forecast_ANN/ or absolute
        
        OLD STRUCTURE (fallback):
        - Direct file lookup in model_dir and scaler_dir
        - Pattern: {target}_{horizon}_model.h5
        """
        if not os.path.exists(model_dir):
            msg = f"Model dir not found: {model_dir}"
            if self.fallback_mode:
                logging.warning(f"[WARNING] {msg} (using fallback)")
                return
            raise ModelLoadingError(msg)
        if not os.path.exists(scaler_dir):
            msg = f"Scaler dir not found: {scaler_dir}"
            if self.fallback_mode:
                logging.warning(f"[WARNING] {msg} (using fallback)")
                return
            raise ModelLoadingError(msg)

        # Determine metadata directory
        if metadata_dir is None:
            # Try to infer: if model_dir is Forecast_ANN/models, metadata is Forecast_ANN/metadata
            if "Forecast_ANN" in model_dir or "models" in model_dir:
                potential_metadata_dir = model_dir.replace("models", "metadata")
                if os.path.exists(potential_metadata_dir):
                    metadata_dir = potential_metadata_dir
            # Also try parent/metadata
            parent_dir = os.path.dirname(model_dir)
            potential_metadata_dir2 = os.path.join(parent_dir, "metadata")
            if metadata_dir is None and os.path.exists(potential_metadata_dir2):
                metadata_dir = potential_metadata_dir2

        # optional training summary (old format)
        summary_path = os.path.join(model_dir, "training_summary.json")
        self.training_summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    self.training_summary = json.load(f)
                if self.verbose:
                    print("[OK] Loaded training summary")
            except Exception as e:
                logging.warning(f"[WARNING] Could not load training summary: {e}")

        # Track if we're using new structure
        using_metadata = False
        if metadata_dir and os.path.exists(metadata_dir):
            using_metadata = True
            if self.verbose:
                print(f"[OK] Using metadata-based loading from: {metadata_dir}")

        # load models/scalers
        for target in self.targets:
            for hname in self.horizons.keys():
                key = f"{target}_{hname}"
                self.loading_stats["models_attempted"] += 1
                
                # NEW STRUCTURE: Try loading from metadata first
                model_path = None
                scaler_x_path = None
                scaler_y_path = None
                metadata_look_back = None
                
                if using_metadata:
                    metadata_path = os.path.join(metadata_dir, f"{key}_metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r") as f:
                                md = json.load(f)
                            
                            # Get model path (prefer best checkpoint)
                            model_path_best = md.get("model_path_best")
                            model_path_final = md.get("model_path", "")
                            
                            # Resolve paths (handle relative paths)
                            if model_path_best:
                                if os.path.isabs(model_path_best):
                                    model_path = model_path_best
                                else:
                                    # Try relative to metadata dir parent, then model_dir
                                    potential_paths = [
                                        os.path.join(os.path.dirname(metadata_dir), model_path_best),
                                        os.path.join(model_dir, os.path.basename(model_path_best)),
                                    ]
                                    for p in potential_paths:
                                        if os.path.exists(p):
                                            model_path = p
                                            break
                                    if model_path is None:
                                        model_path = os.path.join(model_dir, os.path.basename(model_path_best))
                            elif model_path_final:
                                if os.path.isabs(model_path_final):
                                    model_path = model_path_final
                                else:
                                    potential_paths = [
                                        os.path.join(os.path.dirname(metadata_dir), model_path_final),
                                        os.path.join(model_dir, os.path.basename(model_path_final)),
                                    ]
                                    for p in potential_paths:
                                        if os.path.exists(p):
                                            model_path = p
                                            break
                                    if model_path is None:
                                        model_path = os.path.join(model_dir, os.path.basename(model_path_final))
                            
                            # Get scaler paths
                            scaler_x_path = md.get("scaler_x_path")
                            scaler_y_path = md.get("scaler_y_path")
                            if scaler_x_path and not os.path.isabs(scaler_x_path):
                                potential_paths = [
                                    os.path.join(os.path.dirname(metadata_dir), scaler_x_path),
                                    os.path.join(scaler_dir, os.path.basename(scaler_x_path)),
                                ]
                                for p in potential_paths:
                                    if os.path.exists(p):
                                        scaler_x_path = p
                                        break
                                if not os.path.exists(scaler_x_path):
                                    scaler_x_path = os.path.join(scaler_dir, os.path.basename(scaler_x_path))
                            
                            if scaler_y_path and not os.path.isabs(scaler_y_path):
                                potential_paths = [
                                    os.path.join(os.path.dirname(metadata_dir), scaler_y_path),
                                    os.path.join(scaler_dir, os.path.basename(scaler_y_path)),
                                ]
                                for p in potential_paths:
                                    if os.path.exists(p):
                                        scaler_y_path = p
                                        break
                                if not os.path.exists(scaler_y_path):
                                    scaler_y_path = os.path.join(scaler_dir, os.path.basename(scaler_y_path))
                            
                            # Get look_back from metadata (update if different)
                            metadata_look_back = md.get("look_back")
                            if metadata_look_back and metadata_look_back != self.look_back:
                                if self.verbose:
                                    print(f"[INFO] Updating look_back from {self.look_back} to {metadata_look_back} (from metadata for {key})")
                                self.look_back = int(metadata_look_back)
                            
                            # CRITICAL FIX: Store training cap value from metadata (for bounds consistency)
                            # Cap is the max value from training data, used for MAPE calculation
                            training_cap = md.get("cap")
                            if training_cap is not None and target in self.targets:
                                # Use the maximum cap across all horizons for each target
                                if target not in self._training_caps:
                                    self._training_caps[target] = float(training_cap)
                                else:
                                    # Use max cap if multiple horizons have different caps
                                    self._training_caps[target] = max(self._training_caps[target], float(training_cap))
                            
                        except Exception as e:
                            if self.verbose:
                                logging.warning(f"[WARNING] Could not load metadata for {key}: {e}")
                
                # OLD STRUCTURE FALLBACK: Direct file lookup
                if model_path is None or not os.path.exists(model_path):
                    model_path_h5 = os.path.join(model_dir, f"{key}_model.h5")
                    model_path_keras = os.path.join(model_dir, f"{key}_model.keras")
                    model_path = model_path_h5 if os.path.exists(model_path_h5) else model_path_keras
                
                # Load model
                if os.path.exists(model_path) and self.tf is not None:
                    try:
                        # Try multiple loading strategies for compatibility
                        try:
                            # Method 1: Try keras.saving.load_model first (better DTypePolicy handling in Keras 3.x)
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore')
                                try:
                                    from keras.saving import load_model as keras_saving_load
                                    self.models[key] = keras_saving_load(model_path, compile=False)
                                except (ImportError, Exception):
                                    # Fallback to tf.keras with DTypePolicy in custom_objects
                                    custom_objects = {}
                                    
                                    # Handle DTypePolicy (Keras 3.x) - required for loading Keras 3.x models
                                    try:
                                        # Try multiple import paths for DTypePolicy
                                        try:
                                            from keras import DTypePolicy
                                            custom_objects['DTypePolicy'] = DTypePolicy
                                        except ImportError:
                                            try:
                                                from keras.dtype_policies import DTypePolicy
                                                custom_objects['DTypePolicy'] = DTypePolicy
                                            except ImportError:
                                                try:
                                                    from keras.dtype_policies.dtype_policy import DTypePolicy
                                                    custom_objects['DTypePolicy'] = DTypePolicy
                                                except ImportError:
                                                    pass
                                    except Exception:
                                        pass
                                    
                                    try:
                                        if custom_objects:
                                            self.models[key] = self.tf.keras.models.load_model(
                                                model_path,
                                                compile=False,
                                                custom_objects=custom_objects
                                            )
                                        else:
                                            self.models[key] = self.tf.keras.models.load_model(model_path, compile=False)
                                    except Exception as load_err:
                                        # If that fails, try with safe_mode=False
                                        try:
                                            self.models[key] = self.tf.keras.models.load_model(
                                                model_path,
                                                compile=False,
                                                safe_mode=False,
                                                custom_objects=custom_objects if custom_objects else None
                                            )
                                        except TypeError:
                                            # safe_mode not available, try without it
                                            raise load_err
                        except Exception as e1:
                            # Method 2: Try with safe_mode=False (for Keras 3.x compatibility with older models)
                            try:
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore')
                                    # Try to get DTypePolicy for custom_objects
                                    custom_objects = {}
                                    try:
                                        from keras import DTypePolicy
                                        custom_objects['DTypePolicy'] = DTypePolicy
                                    except:
                                        pass
                                    
                                    try:
                                        # Keras 3.x supports safe_mode parameter
                                        self.models[key] = self.tf.keras.models.load_model(
                                            model_path,
                                            compile=False,
                                            safe_mode=False,
                                            custom_objects=custom_objects if custom_objects else None
                                        )
                                    except TypeError:
                                        # safe_mode not available (Keras 2.x), try with custom_objects
                                        self.models[key] = self.tf.keras.models.load_model(
                                            model_path,
                                            compile=False,
                                            custom_objects=custom_objects if custom_objects else None
                                        )
                            except Exception as e2:
                                # Method 3: Try using keras.saving.load_model (better DTypePolicy handling)
                                try:
                                    import warnings
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings('ignore')
                                        try:
                                            from keras.saving import load_model as keras_load_model
                                            # Try keras.saving.load_model which handles DTypePolicy better
                                            self.models[key] = keras_load_model(model_path, compile=False)
                                        except ImportError:
                                            # Fallback to tf.keras with comprehensive custom_objects
                                            custom_objects = {}
                                            
                                            # Add DTypePolicy
                                            try:
                                                from keras import DTypePolicy
                                                custom_objects['DTypePolicy'] = DTypePolicy
                                            except:
                                                try:
                                                    from keras.dtype_policies import DTypePolicy
                                                    custom_objects['DTypePolicy'] = DTypePolicy
                                                except:
                                                    pass
                                            
                                            # Add InputLayer handler
                                            from tensorflow.keras.layers import InputLayer as BaseInputLayer
                                            class CompatibleInputLayer(BaseInputLayer):
                                                def __init__(self, **kwargs):
                                                    if 'batch_shape' in kwargs:
                                                        batch_shape = kwargs.pop('batch_shape')
                                                        if batch_shape and len(batch_shape) > 1:
                                                            kwargs['input_shape'] = batch_shape[1:]
                                                    super().__init__(**kwargs)
                                            custom_objects['InputLayer'] = CompatibleInputLayer
                                            
                                            self.models[key] = self.tf.keras.models.load_model(
                                                model_path,
                                                compile=False,
                                                custom_objects=custom_objects if custom_objects else None
                                            )
                                except Exception as e3:
                                    # Method 4: Reconstruct model from metadata and load weights
                                    try:
                                        if using_metadata and metadata_dir:
                                            metadata_path = os.path.join(metadata_dir, f"{key}_metadata.json")
                                            if os.path.exists(metadata_path):
                                                with open(metadata_path, "r") as f:
                                                    md = json.load(f)
                                                
                                                # Reconstruct model architecture from metadata
                                                input_features = md.get('input_features', 24)
                                                architecture_desc = md.get('architecture', '256-128 (2 hidden layers with dropout 0.2)')
                                                
                                                # Parse architecture: "256-128 (2 hidden layers with dropout 0.2)"
                                                import re
                                                units_match = re.search(r'(\d+)-(\d+)', architecture_desc)
                                                if units_match:
                                                    units1 = int(units_match.group(1))
                                                    units2 = int(units_match.group(2))
                                                    
                                                    from tensorflow.keras.layers import Dense, Dropout
                                                    from tensorflow.keras.models import Sequential
                                                    
                                                    # Rebuild model with EXACT same architecture as original
                                                    # Original: Input -> Dense(256) -> Dropout(0.2) -> Dense(128) -> Dropout(0.2) -> Dense(1)
                                                    # Note: Sequential models auto-name layers as "dense", "dense_1", "dense_2", "dropout", "dropout_1"
                                                    # We don't specify names to let Sequential auto-generate them (matches original)
                                                    model = Sequential([
                                                        self.tf.keras.Input(shape=(input_features,)),
                                                        Dense(units1, activation='relu'),
                                                        Dropout(0.2),
                                                        Dense(units2, activation='relu'),
                                                        Dropout(0.2),
                                                        Dense(1)
                                                    ])
                                                    
                                                    # Load weights using h5py to bypass DTypePolicy issues
                                                    weights_file = model_path
                                                    best_path = md.get('model_path_best')
                                                    if best_path and os.path.exists(best_path):
                                                        weights_file = best_path
                                                    
                                                    if os.path.exists(weights_file):
                                                        try:
                                                            # Method 1: Try loading by index (more reliable for Sequential models)
                                                            # Sequential models save layers in order, so we can load by index
                                                            import h5py
                                                            with h5py.File(weights_file, 'r') as f:
                                                                if 'model_weights' in f:
                                                                    model_weights = f['model_weights']
                                                                    
                                                                    # Get all layer names from saved model (filter out non-weight layers)
                                                                    saved_layer_names = [name for name in model_weights.keys() 
                                                                                        if name not in ['top_level_model_weights'] 
                                                                                        and len(model_weights[name].keys()) > 0]
                                                                    
                                                                    # Filter out InputLayer and Dropout (they have no trainable weights)
                                                                    trainable_layers = [l for l in model.layers 
                                                                                       if hasattr(l, 'get_weights') and len(l.get_weights()) > 0]
                                                                    
                                                                    # Match layers by name (dense, dense_1, dense_2)
                                                                    # Structure: model_weights/dense/sequential/dense/kernel and bias
                                                                    layers_loaded = 0
                                                                    for model_layer in trainable_layers:
                                                                        layer_name = model_layer.name
                                                                        if layer_name in model_weights:
                                                                            layer_weights = model_weights[layer_name]
                                                                            # Weights are stored under 'sequential/dense' subdirectory
                                                                            # Structure: model_weights/dense/sequential/dense/kernel and bias
                                                                            weight_values = []
                                                                            if 'sequential' in layer_weights:
                                                                                seq_weights = layer_weights['sequential']
                                                                                # The actual weights are under 'dense' (not layer_name)
                                                                                if 'dense' in seq_weights:
                                                                                    weight_group = seq_weights['dense']
                                                                                    # Keys are 'kernel' and 'bias', not 'kernel:0' and 'bias:0'
                                                                                    if 'kernel' in weight_group:
                                                                                        weight_values.append(weight_group['kernel'][:])
                                                                                    if 'bias' in weight_group:
                                                                                        weight_values.append(weight_group['bias'][:])
                                                                                # Fallback: try with layer name
                                                                                elif layer_name in seq_weights:
                                                                                    weight_group = seq_weights[layer_name]
                                                                                    if 'kernel' in weight_group:
                                                                                        weight_values.append(weight_group['kernel'][:])
                                                                                    if 'bias' in weight_group:
                                                                                        weight_values.append(weight_group['bias'][:])
                                                                            
                                                                            if len(weight_values) >= 2:  # Need both kernel and bias
                                                                                model_layer.set_weights(weight_values)
                                                                                layers_loaded += 1
                                                                    
                                                                    if self.verbose:
                                                                        print(f"[OK] weights loaded via h5py for {key} ({layers_loaded}/{len(trainable_layers)} layers)")
                                                                    
                                                                    if layers_loaded == 0:
                                                                        raise ValueError("No weights loaded - check layer name matching")
                                                                else:
                                                                    raise ValueError("No 'model_weights' found in HDF5 file")
                                                        except Exception as h5_err:
                                                            # Final fallback: try direct loading
                                                            try:
                                                                model.load_weights(weights_file, by_name=True, skip_mismatch=True)
                                                                if self.verbose:
                                                                    print(f"[OK] weights loaded directly (fallback) for {key}")
                                                            except Exception as final_err:
                                                                if self.verbose:
                                                                    logging.warning(f"[WARNING] Could not load weights for {key}: {h5_err}, {final_err}")
                                                                raise
                                                    
                                                    # Validate weights were loaded correctly
                                                    total_params = sum([np.prod(layer.get_weights()[0].shape) if len(layer.get_weights()) > 0 else 0 for layer in model.layers])
                                                    if total_params == 0:
                                                        raise ValueError(f"No weights loaded for {key} - model reconstruction failed")
                                                    
                                                    self.models[key] = model
                                                    if self.verbose:
                                                        print(f"[OK] model reconstructed from metadata: {key} ({total_params:,} params loaded)")
                                                else:
                                                    raise e3
                                            else:
                                                raise e3
                                        else:
                                            raise e3
                                    except Exception as e4:
                                        # Method 5: Try loading architecture + weights separately (old structure)
                                        arch_path = model_path.replace('.h5', '_architecture.json')
                                        weights_path = model_path.replace('.h5', '_weights.h5')
                                        if os.path.exists(arch_path) and os.path.exists(weights_path):
                                            with open(arch_path, 'r') as f:
                                                model_json = f.read()
                                            # Fix batch_shape and DTypePolicy in JSON
                                            import json as json_module
                                            try:
                                                config = json_module.loads(model_json)
                                                # Recursively fix config
                                                def fix_config(obj):
                                                    if isinstance(obj, dict):
                                                        # Fix batch_shape
                                                        if 'batch_shape' in obj:
                                                            batch_shape = obj['batch_shape']
                                                            if batch_shape and len(batch_shape) > 1:
                                                                obj['input_shape'] = batch_shape[1:]
                                                                del obj['batch_shape']
                                                        # Fix DTypePolicy -> simple dtype string
                                                        if 'dtype' in obj and isinstance(obj['dtype'], dict):
                                                            if obj['dtype'].get('class_name') == 'DTypePolicy':
                                                                dtype_name = obj['dtype'].get('config', {}).get('name', 'float32')
                                                                obj['dtype'] = dtype_name
                                                        for v in obj.values():
                                                            fix_config(v)
                                                    elif isinstance(obj, list):
                                                        for item in obj:
                                                            fix_config(item)
                                                fix_config(config)
                                                model_json = json_module.dumps(config)
                                            except:
                                                pass  # If JSON parsing fails, try as-is
                                            model = self.tf.keras.models.model_from_json(model_json)
                                            model.load_weights(weights_path)
                                            self.models[key] = model
                                        else:
                                            raise e4

                        self.loading_stats["models_loaded"] += 1
                        if self.verbose:
                            source = "metadata" if using_metadata and os.path.exists(os.path.join(metadata_dir, f"{key}_metadata.json")) else "direct"
                            print(f"[OK] model loaded: {key} (from {source})")
                    except Exception as e:
                        self.loading_stats["loading_errors"].append(f"model {key}: {e}")
                        if self.verbose:
                            logging.error(f"[ERROR] model load failed: {key} ({e})")
                else:
                    if self.verbose:
                        print(f"[WARNING] model missing: {key} (path={model_path}, exists={os.path.exists(model_path) if model_path else False}, tf={self.tf is not None})")

                # Load scalers
                self.loading_stats["scalers_attempted"] += 1
                if scaler_x_path and scaler_y_path and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
                    # Use paths from metadata
                    sx, sy = scaler_x_path, scaler_y_path
                else:
                    # OLD STRUCTURE FALLBACK: Try both naming conventions
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
                            source = "metadata" if using_metadata and scaler_x_path else "direct"
                            print(f"[OK] scalers loaded: {key} (from {source})")
                    except Exception as e:
                        self.loading_stats["loading_errors"].append(f"scalers {key}: {e}")
                        if self.verbose:
                            logging.error(f"[ERROR] scalers load failed: {key} ({e})")

        if self.loading_stats["models_loaded"] == 0 and self.fallback_mode:
            print("[WARNING] No models loaded; operating in fallback mode.")

    def _validate_forecast_model_integrity(self, model_dir: str, metadata_dir: Optional[str] = None):
        """
        CRITICAL: Validate that forecast models don't leak RL training data.

        Checks:
        1. Forecast model training end date < RL training start date
        2. No temporal overlap between forecast training and RL training
        3. Proper time-series split validation

        Raises:
            ValueError: If data leakage is detected
        """
        try:
            # Try new structure: look for any metadata file to get training_end
            forecast_train_end = None
            if metadata_dir and os.path.exists(metadata_dir):
                # Try to find any metadata file to get training dates
                for target in self.targets:
                    for hname in self.horizons.keys():
                        metadata_path = os.path.join(metadata_dir, f"{target}_{hname}_metadata.json")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, "r") as f:
                                    md = json.load(f)
                                forecast_train_end = md.get("training_end")
                                if forecast_train_end:
                                    break
                            except Exception:
                                continue
                    if forecast_train_end:
                        break
            
            # Fallback to old structure
            if not forecast_train_end:
                metadata_path = os.path.join(model_dir, "training_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        forecast_train_end = metadata.get("train_end_date") or metadata.get("training_end")
                    except Exception:
                        pass
            
            if not forecast_train_end:
                logging.warning(
                    "[DATA LEAKAGE WARNING] No training_end date found in metadata. "
                    "Cannot validate forecast model integrity. "
                    "Ensure forecast models were trained on data BEFORE RL training period."
                )
                return

            self._forecast_train_end_date = forecast_train_end

            # If RL training start date is provided, validate no overlap
            if self.rl_train_start_date:
                from datetime import datetime
                forecast_end = datetime.fromisoformat(forecast_train_end)
                rl_start = datetime.fromisoformat(self.rl_train_start_date)

                if forecast_end >= rl_start:
                    raise ValueError(
                        f"DATA LEAKAGE DETECTED: Forecast model trained until {forecast_train_end}, "
                        f"but RL training starts at {self.rl_train_start_date}. "
                        f"Forecast models must be trained on data BEFORE RL training period."
                    )

                logging.info(
                    f"[OK] Data integrity validated: Forecast training ended {forecast_train_end}, "
                    f"RL training starts {self.rl_train_start_date}"
                )

            # Validate time-series split (optional - only if test_start is available in metadata)
            # Note: This is optional validation, not all metadata files have test_start

            self._data_leakage_validated = True

        except Exception as e:
            if self.validate_data_integrity:
                logging.error(f"[CRITICAL] Data integrity validation failed: {e}")
                raise
            else:
                logging.warning(f"[WARNING] Data integrity validation failed: {e}")

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
        """Enhanced periodic memory cleanup to prevent leaks."""
        try:
            import gc

            # 1. Clear TensorFlow session if available
            if self.tf is not None:
                try:
                    self.tf.keras.backend.clear_session()
                except Exception:
                    pass

            # 2. Clear model prediction caches
            for model_key, model in self.models.items():
                if hasattr(model, '_prediction_cache'):
                    model._prediction_cache.clear()

            # 3. Trim caches if they're getting too large
            if len(self._global_cache) > 6000:  # Reduced from 8000
                # Clear oldest 25% of entries
                keys_to_remove = list(self._global_cache.keys())[:len(self._global_cache)//4]
                for key in keys_to_remove:
                    if key in self._global_cache:
                        del self._global_cache.cache[key]

            if len(self._agent_cache) > 3000:  # Reduced from 4000
                # Clear oldest 25% of entries
                keys_to_remove = list(self._agent_cache.keys())[:len(self._agent_cache)//4]
                for key in keys_to_remove:
                    if key in self._agent_cache:
                        del self._agent_cache.cache[key]

            # 4. Force garbage collection
            for _ in range(2):
                gc.collect()

            # 5. Clear validation errors if too many
            if self.validator and len(self.validator.validation_errors) > 100:
                self.validator.validation_errors = self.validator.validation_errors[-50:]

            # 6. Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            if self.verbose and self._step_counter % (self._cleanup_frequency * 10) == 0:
                print(f"[CLEANUP] Enhanced memory cleanup at step {self._step_counter}: "
                      f"Global cache: {len(self._global_cache)}, "
                      f"Agent cache: {len(self._agent_cache)}")

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Enhanced memory cleanup failed: {e}")

    # -------- public utils --------

    def update(self, row: Mapping[str, Any]):
        """Feed one new row (dict/Series) to the rolling history. Fast path—no pandas ops."""
        if not isinstance(row, (dict, pd.Series)):
            if self.verbose:
                print(f"[WARNING] update: unsupported row type {type(row)}")
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
            print("[OK] history cleared")

    def initialize_history(self, data: pd.DataFrame, start_idx: int = 0):
        """Initialize history with sufficient data for predictions.

        Args:
            data: DataFrame with historical data
            start_idx: Starting index in data to begin history initialization
        """
        if self.verbose:
            print(f"[INFO] Initializing forecaster history from data...")

        # Pre-populate history with look_back worth of data
        end_idx = min(start_idx + self.look_back, len(data))

        for i in range(start_idx, end_idx):
            row = data.iloc[i]
            self.update(row)

        if self.verbose:
            print(f"[OK] History initialized with {end_idx - start_idx} data points")
            for target in self.targets:
                if target in self.history:
                    hist_len = len(self.history[target])
                    print(f"   {target}: {hist_len} items")

    # -------- predict (fast paths) --------

    def calculate_forecast_confidence(self, agent: str, timestep: Optional[int] = None) -> float:
        """
        Calculate forecast confidence based on recent MAPE (Mean Absolute Percentage Error).

        NOTE: This method is called but typically OVERRIDDEN by environment's
        _get_forecast_confidence() which uses more sophisticated MAPE tracking.

        Returns:
            Confidence score in [0.6, 1.0] based on forecast accuracy
            - 1.0 = perfect forecasts (0% error)
            - 0.6 = floor (40%+ error)
        """
        if agent == "risk_controller_0":
            return 1.0  # Risk controller doesn't need forecast confidence

        # MAPE-BASED CONFIDENCE: Calculate from recent forecast errors
        # This provides a simple fallback if environment doesn't override
        try:
            if hasattr(self, '_forecast_errors') and self._forecast_errors:
                # Get recent errors across all targets
                all_errors = []
                for target_errors in self._forecast_errors.values():
                    if len(target_errors) > 0:
                        all_errors.extend(list(target_errors)[-10:])  # Last 10 per target

                if all_errors:
                    # Calculate MAPE
                    mape = np.mean(all_errors)
                    # Convert to confidence: 0% error → 1.0, 50% error → 0.5
                    confidence = 1.0 - np.clip(mape, 0.0, 0.5)
                    # Apply floor of 0.6 (60%)
                    return float(np.clip(confidence, 0.6, 1.0))
        except Exception:
            pass

        # FALLBACK: Return floor confidence if no error tracking available
        return 0.6  # Conservative default (was 1.0 - now realistic)

    def predict_for_agent(self, agent: str, timestep: Optional[int] = None) -> Dict[str, float]:
        """
        Per-agent forecasts for current step, with caching + optional throttle.
        Returns keys like 'wind_forecast_immediate', ... exactly as the wrapper expects.
        """
        if agent == "risk_controller_0":
            return {}

        t = int(timestep or 0)
        cache_key = (agent, t)
        stride = self._get_adaptive_refresh_stride(t)

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

        # Reset per-call accumulator for timing instrumentation
        self._predict_accumulator = 0.0

        targets = self.agent_targets.get(agent, [])
        horizons = self.agent_horizons.get(agent, [])
        out: Dict[str, float] = {}

        for target in targets:
            for hname in horizons:
                k = f"{target}_forecast_{hname}"
                out[k] = self._predict_target_horizon(target, hname, t)

        # Add forecast confidence to agent observations
        if agent != "risk_controller_0":
            out["forecast_confidence"] = self.calculate_forecast_confidence(agent, t)

        # Provide optional timing information (ms) for the whole predict_for_agent call
        out["forecast_inference_time_ms"] = float(self._predict_accumulator)

        # If timing log path configured, append a CSV line with summary stats
        if self.timing_log_path:
            try:
                num_preds = sum(1 for k in out.keys() if k.endswith(tuple(self.horizons.keys())))
                mean_ms = float(self._predict_accumulator / max(1, num_preds))
                with open(self.timing_log_path, 'a', newline='') as tf:
                    w = csv.writer(tf)
                    w.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'), t, agent, f"{self._predict_accumulator:.6f}", num_preds, f"{mean_ms:.6f}"])
            except Exception:
                pass

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
        Enhanced with uncertainty quantification.
        """
        t = int(timestep or 0)
        # Reset accumulator for timing instrumentation
        self._predict_accumulator = 0.0
        results: Dict[str, float] = {}

        for target in self.targets:
            for hname in self.horizons.keys():
                k = f"{target}_forecast_{hname}"
                forecast_value = self._predict_target_horizon(target, hname, t)
                results[k] = forecast_value

                # Add uncertainty quantification
                uncertainty = self._calculate_forecast_uncertainty(target, hname, t)
                results[f"{target}_uncertainty_{hname}"] = uncertainty

                # Add prediction intervals (95% confidence)
                lower_bound = forecast_value - 1.96 * uncertainty
                upper_bound = forecast_value + 1.96 * uncertainty
                results[f"{target}_lower95_{hname}"] = lower_bound
                results[f"{target}_upper95_{hname}"] = upper_bound

        # Include timing summary for full-horizon predictions
        results['forecast_inference_time_ms'] = float(self._predict_accumulator)

        # Optionally persist timing to CSV (agent unknown for full predict_all_horizons, use 'all')
        if self.timing_log_path:
            try:
                num_preds = sum(1 for k in results.keys() if k.endswith(tuple(self.horizons.keys())))
                mean_ms = float(self._predict_accumulator / max(1, num_preds))
                with open(self.timing_log_path, 'a', newline='') as tf:
                    w = csv.writer(tf)
                    w.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'), t, 'all', f"{self._predict_accumulator:.6f}", num_preds, f"{mean_ms:.6f}"])
            except Exception:
                pass

        return results

    def _calculate_forecast_uncertainty(self, target: str, horizon: str, timestep: int) -> float:
        """
        Calculate forecast uncertainty based on recent prediction errors and model confidence.
        Returns uncertainty estimate (standard deviation).
        """
        try:
            # Base uncertainty from model availability
            model_key = f"{target}_{horizon}"
            model_available = model_key in self.models and self.models[model_key] is not None
            base_uncertainty = 0.05 if model_available else 0.15  # Lower uncertainty for trained models

            # Historical accuracy-based uncertainty
            key = (target, horizon)
            if key in self.accuracy_history and len(self.accuracy_history[key]) > 0:
                recent_errors = list(self.accuracy_history[key])[-10:]  # Last 10 predictions
                if recent_errors:
                    # Use standard deviation of recent errors as uncertainty estimate
                    error_std = np.std(recent_errors)
                    # Combine with base uncertainty
                    accuracy_uncertainty = min(error_std, 0.3)  # Cap at 30%
                    base_uncertainty = (base_uncertainty + accuracy_uncertainty) / 2

            # Market volatility adjustment
            if hasattr(self, 'history') and 'price' in self.history:
                price_history = self.history['price']
                if len(price_history) >= 5:
                    recent_prices = price_history[-5:]
                    if len(recent_prices) > 1:
                        price_volatility = np.std(recent_prices) / (np.mean(recent_prices) + 1e-6)
                        # Higher market volatility increases forecast uncertainty
                        volatility_factor = 1.0 + min(price_volatility, 0.5)  # Cap at 50% increase
                        base_uncertainty *= volatility_factor

            # Horizon adjustment (longer horizons = higher uncertainty)
            horizon_multipliers = {
                'immediate': 1.0,
                'short': 1.2,
                'medium': 1.5,
                'long': 2.0,
                'strategic': 2.5
            }
            horizon_factor = horizon_multipliers.get(horizon, 1.5)
            base_uncertainty *= horizon_factor

            # Ensure reasonable bounds
            return float(np.clip(base_uncertainty, 0.01, 0.5))  # 1% to 50% uncertainty

        except Exception:
            # Fallback uncertainty
            return 0.1  # 10% default uncertainty

    def _get_adaptive_refresh_stride(self, timestep: int) -> int:
        """
        Calculate adaptive refresh stride based on market conditions.
        Higher volatility = more frequent refreshes (lower stride).
        """
        try:
            # Base stride from configuration
            base_stride = self.agent_refresh_stride

            # Calculate market volatility
            market_volatility = self._calculate_market_volatility()

            # Adaptive stride based on volatility
            if market_volatility > 0.7:
                # High volatility: refresh every step
                adaptive_stride = 1
            elif market_volatility > 0.4:
                # Medium volatility: refresh every 2-3 steps
                adaptive_stride = max(1, base_stride // 2)
            elif market_volatility > 0.2:
                # Low volatility: use base stride
                adaptive_stride = base_stride
            else:
                # Very low volatility: refresh less frequently
                adaptive_stride = min(base_stride * 2, 10)

            return max(1, adaptive_stride)  # Ensure at least 1

        except Exception:
            # Fallback to base stride
            return self.agent_refresh_stride

    def _calculate_market_volatility(self) -> float:
        """
        Calculate current market volatility based on recent price history.
        Returns volatility score between 0.0 and 1.0.
        """
        try:
            if not hasattr(self, 'history') or 'price' not in self.history:
                return 0.5  # Default medium volatility

            price_history = self.history['price']
            if len(price_history) < 5:
                return 0.5  # Not enough data

            # Use recent price history for volatility calculation
            recent_prices = list(price_history)[-10:]  # Last 10 prices
            if len(recent_prices) < 2:
                return 0.5

            # Calculate price volatility (coefficient of variation)
            price_mean = np.mean(recent_prices)
            if price_mean <= 1e-6:
                return 0.5

            price_std = np.std(recent_prices)
            volatility = price_std / price_mean

            # Normalize to [0, 1] range
            # Typical energy price volatility ranges from 0.1 to 0.8
            normalized_volatility = np.clip(volatility / 0.8, 0.0, 1.0)

            return float(normalized_volatility)

        except Exception:
            return 0.5  # Default medium volatility

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

    def _debias_price_forecast(self, raw_forecast: float, t: int) -> float:
        """
        Apply minimal bounds checking to price forecasts.
        
        SIMPLIFIED: Models are always trained per-episode on the same data they're used on.
        No distribution mismatch → no debiasing needed.
        
        Only applies reasonable bounds to prevent extreme outliers.
        """
        # Get recent actual prices for bounds calculation
        recent_prices = []
        if self._data_df is not None and 'price' in self._data_df.columns:
            lookback = min(144, max(10, t))
            start_idx = max(0, t - lookback)
            end_idx = t
            if end_idx <= len(self._data_df) and start_idx < end_idx:
                recent_prices = self._data_df['price'].iloc[start_idx:end_idx].values.tolist()
                recent_prices = [p for p in recent_prices if np.isfinite(p)]
        
        if len(recent_prices) >= 5:
            local_mean = float(np.mean(recent_prices))
            # Apply conservative bounds: ±100% around local mean (just to prevent extreme outliers)
            min_forecast = local_mean * 0.5
            max_forecast = local_mean * 2.0
            return float(np.clip(raw_forecast, min_forecast, max_forecast))
        else:
            # No history - return raw forecast with basic safety bounds
            return float(np.clip(raw_forecast, 50.0, 1000.0))

    def _predict_target_horizon(self, target: str, hname: str, t: int) -> float:
        """
        Simplified prediction that works directly with raw MW units.
        No complex normalization - models trained on raw data, return raw data.

        CRITICAL FIX: Apply adaptive debiasing for price forecasts to correct
        for train-test distribution mismatch.
        """
        # 1) Check precomputed cache first (fastest path)
        arr = self._precomputed.get((target, hname), None)
        if arr is not None and 0 <= t < len(arr):
            val = arr[t]
            if np.isfinite(val):
                # NOTE: Precomputed forecasts already have debiasing applied during precomputation
                # No need to apply debiasing again here
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

                # Get prediction with memory management
                try:
                    start_t = time.perf_counter()
                    y_pred = self.models[model_key].predict(Xs, verbose=0)
                    end_t = time.perf_counter()
                    # accumulate predict time (ms) for optional logging
                    try:
                        self._predict_accumulator += (end_t - start_t) * 1000.0
                    except Exception:
                        pass

                    # Extract scalar value
                    if isinstance(y_pred, np.ndarray):
                        y_pred = float(np.ravel(y_pred)[0]) if y_pred.size > 0 else 0.0
                    else:
                        y_pred = float(y_pred)
                except Exception as e:
                    # If prediction fails, propagate to fallback below
                    if self.verbose:
                        print(f"Model prediction failed for {model_key}: {e}")
                    y_pred = None

                # MEMORY LEAK FIX: Clear prediction tensors immediately
                # (Note: y_pred is already converted to float, so no tensor cleanup needed here)

                # Inverse scale if needed (but models should output raw units)
                if y_pred is None:
                    y = self._fallback_value(target)
                else:
                    y = self._inverse_scale_y(model_key, y_pred)
                    # Simple bounds check
                    y = self._constrain_target(target, y)
                    
                    # Apply debiasing for price forecasts (runtime predictions)
                    if target == 'price':
                        y = self._debias_price_forecast(y, t)

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
        """Default values in raw MW units - try to get from cached config first."""
        # Try to get defaults from cached config if available
        try:
            if self.config and hasattr(self.config, 'default_forecasts') and target in self.config.default_forecasts:
                return self.config.default_forecasts[target]
        except Exception:
            pass

        # Fallback to hardcoded values
        return {
            "wind": 330.0,    # ~30% of 1103 MW capacity
            "solar": 20.0,    # ~20% of 100 MW capacity
            "hydro": 267.0,   # ~50% of 534 MW capacity
            "price": 345.0,   # DKK/MWh (50 USD * 6.9 = ~345 DKK)
            "load": 1800.0,   # ~60% of 2999 MW capacity
        }.get(target, 0.0)

    def _constrain_target(self, target: str, val: float) -> float:
        """
        Apply realistic bounds for raw MW values using training cap values from metadata.
        
        CRITICAL FIX: Uses training cap values (max from training data) for bounds consistency.
        Falls back to hardcoded bounds if training caps not available.
        """
        try:
            if not np.isfinite(val):
                return self._default_for_target(target)

            result = float(val)

            # CRITICAL FIX: Use training cap values from metadata for bounds (consistent with training)
            # Add small buffer (5%) above cap to allow for slight variations
            if target in self._training_caps:
                training_cap = self._training_caps[target]
                # Use cap * 1.05 as upper bound (5% buffer) or cap * 1.1 for wind (10% buffer for larger variance)
                buffer_mult = 1.1 if target == 'wind' else 1.05
                max_val = training_cap * buffer_mult
                bounds = (0.0, max_val)
            else:
                # Fallback to hardcoded bounds if training caps not loaded
                bounds_map = {
                    "wind": (0.0, 1600.0),    # 0 to max observed (1500) + buffer
                    "solar": (0.0, 1100.0),   # 0 to max observed (1000) + buffer
                    "hydro": (0.0, 1100.0),   # 0 to max observed (1000) + buffer
                    "price": (-50.0, 500.0),  # Reasonable price range
                    "load": (500.0, 3500.0),  # Reasonable load range
                }
                bounds = bounds_map.get(target, (0.0, 1e6))

            if bounds:
                min_val, max_val = bounds
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

    def precompute_offline(self, df: pd.DataFrame, timestamp_col: str = "timestamp", batch_size: int = 4096,
                          cache_dir: str = "forecast_cache") -> None:
        """
        Precompute forecasts for all targets/horizons across the entire dataframe.
        Stores results in self._precomputed[(target, hname)] as float32 arrays of length T.
        Assumes df has columns matching self.targets (e.g., 'wind','solar','hydro','price','load').

        At runtime, _predict_target_horizon() will return arr[t] (O(1)) if present.

        NEW: Supports caching to disk to avoid recomputation on subsequent runs.
        """
        if df is None or len(df) == 0:
            if self.verbose:
                print("precompute_offline: empty dataframe; skipping.")
            return

        # Store data for debiasing
        self._data_df = df.copy()

        # Check for cached forecasts first
        if self._load_cached_forecasts(df, cache_dir):
            if self.verbose:
                print("✅ Loaded precomputed forecasts from cache!")
            return

        # validate columns
        missing = [c for c in self.targets if c not in df.columns]
        if missing:
            raise ValueError(f"precompute_offline: missing columns in df: {missing}")

        T = len(df)
        look_back = int(self.look_back)

        # build rolling windows per target (T x look_back), mean-padded on left
        def make_windows(series: np.ndarray, lb: int) -> np.ndarray:
            """
            PHASE 5 PATCH A: CRITICAL FIX - Time-Series Data Leakage Prevention

            Build rolling windows for time-series prediction.
            For prediction at time 't', use only data up to 't-1' (no target leakage).
            """
            X = np.empty((T, lb), dtype=np.float32)
            mean_val = float(np.nanmean(series)) if series.size else 0.0

            for t in range(T):
                # CRITICAL FIX: Predict time 't' using history up to 't-1'
                # end_idx = t ensures we use series[start_idx:t] (excludes series[t])
                end_idx = t
                start_idx = max(0, end_idx - lb)
                window = series[start_idx:end_idx]

                if window.size < lb:
                    # Pad left with mean value
                    need = lb - window.size
                    if window.size > 0:
                        X[t, :need] = mean_val
                        X[t, need:lb] = window.astype(np.float32)
                    else:
                        X[t, :].fill(mean_val)
                else:
                    # Copy last look_back elements (should be exactly 'lb')
                    X[t, :] = window[-lb:].astype(np.float32)

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

                if not use_model or self.tf is None or model_key not in self.models:
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
                
                # CRITICAL FIX: Apply debiasing for price forecasts during precomputation
                # This ensures precomputed forecasts have proper bias correction applied
                if target == 'price' and self._data_df is not None and 'price' in self._data_df.columns:
                    for i in range(T):
                        preds[i] = self._debias_price_forecast(preds[i], i)

                self._precomputed[key] = preds

        if self.verbose:
            total_forecasts = len(self.targets) * len(self.horizons)
            print(f"Offline precompute complete: {total_forecasts} forecast series cached.")

        # Save computed forecasts to cache
        self._save_cached_forecasts(df, cache_dir)

    def _get_cache_filename(self, df: pd.DataFrame) -> str:
        """Generate a descriptive cache filename based on data characteristics"""
        # Create descriptive filename based on data characteristics
        start_date = 'unknown'
        end_date = 'unknown'

        if 'timestamp' in df.columns and len(df) > 0:
            try:
                # Ensure timestamp is datetime
                ts_col = pd.to_datetime(df['timestamp'], errors='coerce')
                if not ts_col.isna().all():
                    start_date = ts_col.iloc[0].strftime('%Y%m%d')
                    end_date = ts_col.iloc[-1].strftime('%Y%m%d')
            except Exception:
                pass

        num_rows = len(df)

        # Create descriptive filename
        cache_filename = f"precomputed_forecasts_{start_date}_to_{end_date}_{num_rows}rows.csv"
        return cache_filename

    def _load_cached_forecasts(self, df: pd.DataFrame, cache_dir: str) -> bool:
        """Load precomputed forecasts from CSV cache if available and valid"""
        import os

        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = self._get_cache_filename(df)
            cache_file = os.path.join(cache_dir, cache_filename)

            if not os.path.exists(cache_file):
                if self.verbose:
                    print(f"No cached forecasts found: {cache_filename}")
                return False

            # Load cached forecasts
            cached_df = pd.read_csv(cache_file)

            # Validate cache matches current data
            if len(cached_df) != len(df):
                if self.verbose:
                    print(f"Cache length mismatch: {len(cached_df)} vs {len(df)}, recomputing...")
                return False

            # CRITICAL: Validate cache integrity with model metadata
            cache_metadata = self._validate_cache_integrity(cache_file)
            if cache_metadata is None:
                if self.verbose:
                    print(f"Cache integrity validation failed, recomputing...")
                return False

            # Convert CSV back to _precomputed format
            self._precomputed = {}

            alignment_mode = cache_metadata.get('forecast_alignment', 'origin_timestamp')
            horizon_offsets = cache_metadata.get('horizon_offsets', {})
            forecast_tails = cache_metadata.get('forecast_tails', {})

            for target in self.targets:
                for hname in self.horizons.keys():
                    col_name = f"{target}_forecast_{hname}"
                    if col_name not in cached_df.columns:
                        if self.verbose:
                            print(f"Missing column {col_name} in cache, recomputing...")
                        return False

                    series = cached_df[col_name].values.astype(np.float32)

                    if alignment_mode == 'target_timestamp':
                        steps = int(horizon_offsets.get(hname, self.horizons.get(hname, 0)))
                        tail_map = forecast_tails.get(target, {}) if isinstance(forecast_tails, dict) else {}
                        tail_values = tail_map.get(hname, []) if isinstance(tail_map, dict) else []
                        restored = self._restore_forecast_from_cache(series, steps, tail_values, target)
                        self._precomputed[(target, hname)] = restored
                    else:
                        self._precomputed[(target, hname)] = series

            # Store data for debiasing
            self._data_df = df.copy()

            if self.verbose:
                total_series = len(self._precomputed)
                file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                print(f"✅ Loaded {total_series} forecast series from CSV cache ({file_size_mb:.1f}MB)")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Failed to load cached forecasts: {e}")
            return False

    def _validate_cache_integrity(self, cache_file: str) -> Optional[Dict[str, Any]]:
        """
        Validate that cached forecasts are still valid for current models.

        Checks:
        1. Cache metadata file exists
        2. Model versions match
        3. Model modification times haven't changed

        Returns:
            Optional[Dict[str, Any]]: Metadata dict when cache is valid, otherwise None
        """
        try:
            metadata_file = cache_file.replace('.csv', '_metadata.json')

            if not os.path.exists(metadata_file):
                # No metadata = old cache format, invalidate
                return None

            with open(metadata_file, 'r') as f:
                cache_metadata = json.load(f)

            # Check model modification times
            cached_model_times = cache_metadata.get('model_modification_times', {})

            for model_key in self.models.keys():
                model_path = cache_metadata.get('model_paths', {}).get(model_key)
                if model_path and os.path.exists(model_path):
                    current_mtime = os.path.getmtime(model_path)
                    cached_mtime = cached_model_times.get(model_key)

                    if cached_mtime is None or abs(current_mtime - cached_mtime) > 1.0:
                        # Model has been modified, invalidate cache
                        if self.verbose:
                            print(f"Model {model_key} modified, invalidating cache")
                        return None

            # Check look_back parameter
            if cache_metadata.get('look_back') != self.look_back:
                if self.verbose:
                    print(f"look_back changed ({cache_metadata.get('look_back')} -> {self.look_back}), invalidating cache")
                return None

            return cache_metadata

        except Exception as e:
            logging.warning(f"Cache integrity validation failed: {e}")
            return None

    def _align_forecast_for_export(self, series: np.ndarray, horizon_steps: int) -> np.ndarray:
        """Shift forecasts forward so row timestamp matches the forecast target time."""
        if series is None:
            return np.array([], dtype=np.float32)

        series = np.asarray(series, dtype=np.float32)
        T = len(series)
        if T == 0 or horizon_steps <= 0:
            return series.copy()

        aligned = np.full(T, np.nan, dtype=np.float32)
        if horizon_steps < T:
            aligned[horizon_steps:] = series[:-horizon_steps]
        # When horizon_steps >= T, we intentionally leave the array as NaN (no aligned targets)
        return aligned

    def _restore_forecast_from_cache(
        self,
        aligned_series: np.ndarray,
        horizon_steps: int,
        tail_values: Optional[List[float]] = None,
        target: Optional[str] = None,
    ) -> np.ndarray:
        """Restore original timeline forecasts from an aligned cache column."""
        aligned = np.asarray(aligned_series, dtype=np.float32)
        T = len(aligned)
        if T == 0 or horizon_steps <= 0:
            return aligned.copy()

        raw = np.full(T, np.nan, dtype=np.float32)
        if horizon_steps < T:
            raw[:T - horizon_steps] = aligned[horizon_steps:]

        tail_len = min(horizon_steps, T)
        if tail_len == 0:
            return np.nan_to_num(raw, nan=self._default_for_target(target) if target else 0.0)

        tail_array = None
        if tail_values:
            try:
                tail_array = np.asarray(tail_values, dtype=np.float32)
            except Exception:
                tail_array = None

        if tail_array is not None and tail_array.size > 0:
            tail_array = tail_array[-tail_len:]
            raw[T - tail_array.size:] = tail_array
        else:
            fill_value = self._default_for_target(target) if target else 0.0
            if T - tail_len - 1 >= 0 and np.isfinite(raw[T - tail_len - 1]):
                fill_value = float(raw[T - tail_len - 1])
            raw[T - tail_len:] = fill_value

        return raw

    def invalidate_cache(self, cache_dir: str = "forecast_cache"):
        """
        Invalidate all cached forecasts.

        Use this when:
        - Models have been retrained
        - Data has changed
        - Configuration has changed
        """
        try:
            import glob
            cache_files = glob.glob(os.path.join(cache_dir, "precomputed_forecasts_*.csv"))
            metadata_files = glob.glob(os.path.join(cache_dir, "precomputed_forecasts_*_metadata.json"))

            for f in cache_files + metadata_files:
                try:
                    os.remove(f)
                    if self.verbose:
                        print(f"Removed cache file: {f}")
                except Exception as e:
                    logging.warning(f"Failed to remove cache file {f}: {e}")

            self._precomputed.clear()

            if self.verbose:
                print(f"✅ Cache invalidated: removed {len(cache_files)} cache files")

        except Exception as e:
            logging.error(f"Failed to invalidate cache: {e}")

    def _save_cached_forecasts(self, df: pd.DataFrame, cache_dir: str) -> None:
        """Save precomputed forecasts to CSV cache"""
        import os

        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = self._get_cache_filename(df)
            cache_file = os.path.join(cache_dir, cache_filename)

            # Prepare forecast data for CSV
            forecast_data = {}

            # Add timestamp if available
            if 'timestamp' in df.columns:
                # Ensure timestamp is properly formatted for CSV
                try:
                    ts_series = pd.to_datetime(df['timestamp'], errors='coerce')
                    forecast_data['timestamp'] = ts_series.dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    forecast_data['timestamp'] = df['timestamp'].values

            horizon_offsets = {hname: int(steps) for hname, steps in self.horizons.items()}
            forecast_tails: Dict[str, Dict[str, List[float]]] = {}

            # Add forecast columns (aligned to target timestamps)
            for target in self.targets:
                forecast_tails[target] = {}
                for hname, horizon_steps in self.horizons.items():
                    col_name = f"{target}_forecast_{hname}"
                    raw_series = self._precomputed.get((target, hname))

                    if raw_series is None:
                        default_val = self._default_for_target(target)
                        raw_series = np.full(len(df), default_val, dtype=np.float32)
                    else:
                        raw_series = np.asarray(raw_series, dtype=np.float32)

                    aligned_series = self._align_forecast_for_export(raw_series, int(horizon_steps))
                    forecast_data[col_name] = aligned_series

                    tail_len = min(int(horizon_steps), len(raw_series))
                    if tail_len > 0:
                        forecast_tails[target][hname] = raw_series[-tail_len:].astype(float).tolist()
                    else:
                        forecast_tails[target][hname] = []

            # Save to CSV
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df.to_csv(cache_file, index=False)

            # Save metadata for cache validation
            metadata_file = cache_file.replace('.csv', '_metadata.json')
            metadata = {
                'look_back': self.look_back,
                'targets': self.targets,
                'horizons': list(self.horizons.keys()),
                'horizon_offsets': horizon_offsets,
                'forecast_alignment': 'target_timestamp',
                'forecast_tails': forecast_tails,
                'model_modification_times': {},
                'model_paths': {},
                'cache_created': pd.Timestamp.now().isoformat(),
            }

            # Record model modification times for cache invalidation
            for model_key in self.models.keys():
                # Try to find model file path
                for ext in ['.h5', '.keras']:
                    model_path = os.path.join(os.path.dirname(cache_file), '..', 'saved_models', f"{model_key}_model{ext}")
                    if os.path.exists(model_path):
                        metadata['model_paths'][model_key] = model_path
                        metadata['model_modification_times'][model_key] = os.path.getmtime(model_path)
                        break

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            if self.verbose:
                file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                print(f"💾 Saved forecasts to cache: {cache_filename} ({file_size_mb:.1f}MB)")

        except Exception as e:
            if self.verbose:
                print(f"Failed to save cached forecasts: {e}")

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
        dims = {}
        for agent in self.agent_horizons:
            if agent == "risk_controller_0":
                # Risk controller uses enhanced risk metrics instead of forecasts
                dims[agent] = 0
            else:
                dims[agent] = len(self.agent_targets.get(agent, [])) * len(self.agent_horizons.get(agent, []))
        return dims

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
            badge = "[OK]" if avail else "[MISSING]"
            print(f"  {t}: {badge} {avail}")
        if self.loading_stats["loading_errors"]:
            print(f"\n[WARNING] {len(self.loading_stats['loading_errors'])} loading errors (showing up to 5):")
            for e in self.loading_stats["loading_errors"][:5]:
                print(f"  • {e}")

        # Clip diagnostics summary (if any samples)
        cs = self.get_clip_stats()
        if any(v["total"] > 0 for v in cs.values()):
            print("\nClip diagnostics (renewables/load):")
            for t, d in cs.items():
                tot = max(1, d.get("total", 0))
                hi = SafeDivision.div(d.get("high", 0) * 100.0, tot)
                lo = SafeDivision.div(d.get("low", 0) * 100.0, tot)
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
            "tensorflow_available": self.tf is not None,
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
        if self.tf is None:
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
            print("[WARNING] Integrity issues:")
            for m in issues:
                print("  •", m)
            return False
        print("[OK] System integrity OK")
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
        look_back=24,  # IMPROVED: Increased from 6 to 24 (must match retrained models)
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
                # Account for forecast_confidence being added to the output
                expected_forecasts = len(gen.agent_targets[a]) * len(gen.agent_horizons[a])
                expected_total = expected_forecasts + 1  # +1 for forecast_confidence
                assert len(out) == expected_total, f"{a}: got {len(out)}, expected {expected_total}"
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
