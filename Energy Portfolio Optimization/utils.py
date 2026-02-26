#!/usr/bin/env python3
"""
UNIFIED UTILITIES MODULE (COMPREHENSIVE)

Consolidates all utility functions and classes into a single, organized module:
- safe_utils.py (SafeDivision, safe_clip, safe_mean, safe_std, sanitize_array, ensure_finite)
- utils_overlay.py (load_overlay_weights, get_overlay_feature_dim, clear_tf_session, configure_tf_memory)
- error_handling.py (ErrorHandler, safe_operation, validation functions, ContextualLogger)
- memory_manager.py (UnifiedMemoryManager)
- observation_validator.py (UnifiedObservationValidator)
- dl_overlay.py (OverlaySharedModel, DLAdapter, OverlayExperienceBuffer, OverlayTrainer, CalibrationTracker)

Single source of truth for all utility operations across the entire codebase.
"""

import os
import gc
import logging
import traceback
import functools
import psutil
import threading
import weakref
import numpy as np
from collections import deque, OrderedDict
from typing import Dict, Tuple, Any, Optional, Callable, List, Type

# LAZY TensorFlow initialization (no top-level import to avoid duplicate prints)
_tf_initialized = False
_tf_module = None

def _get_tf():
    """Lazy TensorFlow initialization (single point of initialization)."""
    global _tf_initialized, _tf_module
    if not _tf_initialized:
        try:
            import tensorflow as tf
            _tf_module = tf
            _tf_initialized = True
        except ImportError:
            logging.warning("TensorFlow not available")
            _tf_initialized = True
    return _tf_module

logger = logging.getLogger(__name__)


# ============================================================================
# SAFE OPERATIONS (from safe_utils.py)
# ============================================================================

class SafeDivision:
    """
    UNIFIED SAFE DIVISION (SINGLE SOURCE OF TRUTH)
    
    Provides robust division with protection against zero-division and NaN/Inf.
    Used throughout the codebase to prevent numerical errors.
    """
    
    @staticmethod
    def div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Perform safe division with fallback."""
        try:
            if denominator is None or abs(denominator) < 1e-9:
                return default
            result = float(numerator) / float(denominator)
            if not np.isfinite(result):
                return default
            return result
        except Exception:
            return default


def safe_clip(value: float, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    """Safely clip a value to a range."""
    try:
        v = float(value)
        if not np.isfinite(v):
            return default
        return float(np.clip(v, low, high))
    except Exception:
        return default


def safe_mean(values: list, default: float = 0.0) -> float:
    """Safely compute mean of a list."""
    try:
        if not values or len(values) == 0:
            return default
        arr = np.array([float(v) for v in values if np.isfinite(float(v))])
        if len(arr) == 0:
            return default
        return float(np.mean(arr))
    except Exception:
        return default


def safe_std(values: list, default: float = 0.1) -> float:
    """Safely compute standard deviation of a list."""
    try:
        if not values or len(values) < 2:
            return default
        arr = np.array([float(v) for v in values if np.isfinite(float(v))])
        if len(arr) < 2:
            return default
        return float(np.std(arr))
    except Exception:
        return default


def safe_percentile(values: list, percentile: float = 95.0, default: float = 1.0) -> float:
    """Safely compute percentile of a list."""
    try:
        if not values or len(values) == 0:
            return default
        arr = np.array([float(v) for v in values if np.isfinite(float(v))])
        if len(arr) == 0:
            return default
        result = float(np.percentile(arr, percentile))
        if not np.isfinite(result):
            return default
        return result
    except Exception:
        return default


def sanitize_array(arr: np.ndarray, nan_value: float = 0.0, 
                   posinf_value: float = 1.0, neginf_value: float = -1.0) -> np.ndarray:
    """Sanitize numpy array by replacing NaN and Inf values."""
    try:
        return np.nan_to_num(arr, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
    except Exception as e:
        logger.warning(f"Array sanitization failed: {e}")
        return arr


def ensure_finite(value: float, default: float = 0.0) -> float:
    """Ensure a value is finite (not NaN or Inf)."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ============================================================================
# FORECAST MATH UTILITIES (shared by Tier D forecast integration)
# ============================================================================

def convert_mape_to_price_relative(
    mape_dict: Optional[Dict[str, float]],
    current_price: float,
    price_capacity: float,
    price_floor: float = 50.0
) -> Optional[Dict[str, float]]:
    """Convert capacity-based MAPE values into price-relative percentages."""
    if not mape_dict:
        return None

    try:
        denom = max(abs(float(current_price)), float(price_floor), 1e-6)
    except (TypeError, ValueError):
        denom = max(price_floor, 1e-6)

    try:
        capacity_scale = float(price_capacity)
    except (TypeError, ValueError):
        capacity_scale = 1.0

    scale = capacity_scale / denom if denom else 1.0
    converted: Dict[str, float] = {}

    for horizon, value in mape_dict.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        converted[horizon] = float(val * scale)

    return converted if converted else None


def compute_forecast_direction(
    z_short: float,
    z_medium: float,
    z_long: float,
    weights: Tuple[float, float, float] = (0.35, 0.40, 0.25)
) -> float:
    """Combine horizon-weighted z-scores into a capped directional signal."""
    try:
        w_short, w_medium, w_long = weights
    except Exception:
        w_short, w_medium, w_long = (0.35, 0.40, 0.25)

    direction = (
        w_short * np.sign(float(z_short)) +
        w_medium * np.sign(float(z_medium)) +
        w_long * np.sign(float(z_long))
    )
    return float(np.clip(direction, -1.0, 1.0))


def normalize_position(total_position: float, max_position: float) -> float:
    """Normalize a signed position by the configured max position size."""
    try:
        denom = max(abs(float(max_position)), 1.0)
    except (TypeError, ValueError):
        denom = 1.0
    try:
        return float(total_position) / denom
    except Exception:
        return 0.0


def exposure_with_cap(total_position: float, max_position: float, cap: float = 0.3) -> float:
    """Compute absolute exposure normalized by max position and optionally cap it."""
    normalized = normalize_position(total_position, max_position)
    try:
        return float(np.clip(abs(normalized), 0.0, max(cap, 0.0)))
    except Exception:
        return float(abs(normalized))


# ============================================================================
# DL OVERLAY UTILITIES (from utils_overlay.py)
# ============================================================================

def load_overlay_weights(adapter, weights_path: str, feature_dim: int) -> bool:
    """
    Load DL overlay weights with proper Keras model building.

    CRITICAL: Keras models must be "built" (variables initialized) before
    load_weights() can be called. This function handles that requirement.
    """
    if not os.path.exists(weights_path):
        logging.warning(f"Weights file not found: {weights_path}")
        return False

    try:
        # CRITICAL: Get TensorFlow lazily
        tf = _get_tf()
        if tf is None:
            logging.warning(f"TensorFlow not available, cannot load weights from {weights_path}")
            return False

        # CRITICAL: Build model before loading weights
        dummy_input = tf.random.normal((1, feature_dim), dtype=tf.float32)
        _ = adapter.model(dummy_input, training=False)

        # Now load weights
        if hasattr(adapter.model, "load_weights"):
            adapter.model.load_weights(weights_path)
            logging.info(f"[OK] Loaded DL overlay weights ({feature_dim}D) from {weights_path}")
            return True
        else:
            logging.warning(f"DL overlay model does not have load_weights method")
            return False

    except Exception as e:
        if "incompatible" in str(e).lower() or "shape" in str(e).lower():
            logging.warning(f"Feature dimension mismatch when loading {weights_path}")
        else:
            logging.warning(f"Could not load DL overlay weights from {weights_path}: {e}")
        return False


def get_overlay_feature_dim(env) -> int:
    """Get the feature dimension for DL overlay (strict contract; single source of truth)."""
    try:
        from config import OVERLAY_FEATURE_DIM
        return int(OVERLAY_FEATURE_DIM)
    except Exception:
        return 18  # Fallback for medium-horizon overlay


def clear_tf_session():
    """Clear TensorFlow session to prevent memory/graph conflicts."""
    try:
        tf = _get_tf()
        if tf is not None:
            tf.keras.backend.clear_session()
    except Exception:
        pass


def configure_tf_memory():
    """Configure TensorFlow GPU memory management (lazy initialization)."""
    try:
        tf = _get_tf()
        if tf is None:
            logging.warning("TensorFlow not available, skipping GPU memory configuration")
            return

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logging.debug("No GPUs detected, TensorFlow will use CPU")
            return

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"[OK] TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s)")
    except Exception as e:
        logging.warning(f"Could not configure TensorFlow GPU memory: {e}")


# ============================================================================
# ERROR HANDLING (from error_handling.py)
# ============================================================================

class ErrorHandler:
    """UNIFIED ERROR HANDLING - Provides consistent error handling with proper logging."""
    
    @staticmethod
    def safe_call(func: Callable, *args, default: Any = None, 
                  error_msg: Optional[str] = None, **kwargs) -> Any:
        """Safely call a function with error handling and logging."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = error_msg or f"Error calling {func.__name__}"
            logger.error(f"{msg}: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            return default
    
    @staticmethod
    def safe_call_with_fallback(func: Callable, fallback_func: Callable,
                                *args, **kwargs) -> Any:
        """Safely call a function with fallback."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}, using fallback")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as e2:
                logger.error(f"Fallback function also failed: {e2}")
                raise
    
    @staticmethod
    def log_exception(exc: Exception, context: str = "", level: int = logging.ERROR) -> None:
        """Log an exception with context."""
        msg = f"Exception in {context}: {exc}" if context else f"Exception: {exc}"
        logger.log(level, msg)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())


def safe_operation(default: Any = None, error_msg: Optional[str] = None,
                   log_level: int = logging.ERROR):
    """Decorator for safe operations with error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = error_msg or f"Error in {func.__name__}"
                logger.log(log_level, f"{msg}: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
                return default
        return wrapper
    return decorator


def validate_input(condition: bool, error_msg: str) -> None:
    """Validate input condition and raise ValueError if false."""
    if not condition:
        logger.error(f"Validation failed: {error_msg}")
        raise ValueError(error_msg)


def validate_type(value: Any, expected_type: Type, param_name: str) -> None:
    """Validate that value is of expected type."""
    if not isinstance(value, expected_type):
        msg = f"{param_name} must be {expected_type.__name__}, got {type(value).__name__}"
        logger.error(msg)
        raise TypeError(msg)


def validate_range(value: float, min_val: float, max_val: float, param_name: str) -> None:
    """Validate that value is within expected range."""
    try:
        v = float(value)
        if not (min_val <= v <= max_val):
            msg = f"{param_name} must be in [{min_val}, {max_val}], got {v}"
            logger.error(msg)
            raise ValueError(msg)
    except (TypeError, ValueError) as e:
        msg = f"{param_name} must be a valid number in [{min_val}, {max_val}], got {value}"
        logger.error(msg)
        raise ValueError(msg) from e


def validate_not_none(value: Any, param_name: str) -> None:
    """Validate that value is not None."""
    if value is None:
        msg = f"{param_name} cannot be None"
        logger.error(msg)
        raise ValueError(msg)


def validate_array_shape(array: np.ndarray, expected_shape: Tuple[int, ...], param_name: str) -> None:
    """Validate that array has expected shape."""
    if not isinstance(array, np.ndarray):
        msg = f"{param_name} must be numpy array, got {type(array).__name__}"
        logger.error(msg)
        raise TypeError(msg)

    if array.shape != expected_shape:
        msg = f"{param_name} must have shape {expected_shape}, got {array.shape}"
        logger.error(msg)
        raise ValueError(msg)


def validate_dict_keys(d: Dict, required_keys: List[str], param_name: str) -> None:
    """Validate that dictionary contains all required keys."""
    if not isinstance(d, dict):
        msg = f"{param_name} must be dict, got {type(d).__name__}"
        logger.error(msg)
        raise TypeError(msg)

    missing_keys = set(required_keys) - set(d.keys())
    if missing_keys:
        msg = f"{param_name} missing required keys: {missing_keys}"
        logger.error(msg)
        raise ValueError(msg)


class RetryHandler:
    """Handler for retrying operations with exponential backoff."""

    @staticmethod
    def retry(func: Callable, max_attempts: int = 3, backoff_factor: float = 2.0,
              exceptions: Tuple[Type[Exception], ...] = (Exception,),
              default: Any = None) -> Any:
        """
        Retry a function with exponential backoff.

        Args:
            func: Function to retry
            max_attempts: Maximum number of attempts
            backoff_factor: Multiplier for wait time between attempts
            exceptions: Tuple of exceptions to catch
            default: Default value to return if all attempts fail

        Returns:
            Result of function call or default value
        """
        import time

        for attempt in range(max_attempts):
            try:
                return func()
            except exceptions as e:
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} attempts failed: {e}")
                    return default

                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        return default


def validate_range(value: float, low: float, high: float, param_name: str) -> None:
    """Validate that value is within range."""
    if not (low <= value <= high):
        msg = f"{param_name}={value} not in [{low}, {high}]"
        logger.error(msg)
        raise ValueError(msg)


def validate_not_none(value: Any, param_name: str) -> None:
    """Validate that value is not None."""
    if value is None:
        msg = f"{param_name} cannot be None"
        logger.error(msg)
        raise ValueError(msg)


class ContextualLogger:
    """Logger with context information."""
    
    def __init__(self, name: str, context: str = ""):
        self.logger = logging.getLogger(name)
        self.context = context
    
    def _format_msg(self, msg: str) -> str:
        if self.context:
            return f"[{self.context}] {msg}"
        return msg
    
    def debug(self, msg: str) -> None:
        self.logger.debug(self._format_msg(msg))
    
    def info(self, msg: str) -> None:
        self.logger.info(self._format_msg(msg))
    
    def warning(self, msg: str) -> None:
        self.logger.warning(self._format_msg(msg))
    
    def error(self, msg: str) -> None:
        self.logger.error(self._format_msg(msg))
    
    def critical(self, msg: str) -> None:
        self.logger.critical(self._format_msg(msg))


# ============================================================================
# MEMORY MANAGEMENT (from memory_manager.py)
# ============================================================================

class UnifiedMemoryManager:
    """
    UNIFIED MEMORY MANAGEMENT
    
    Provides consistent memory management across all modules:
    - Memory usage tracking
    - Cleanup at multiple levels (light, medium, heavy)
    - Thread-safe operations
    - Weak reference tracking for policies, buffers, environments
    - LRU cache management
    """
    
    def __init__(self, max_memory_mb: float = 6000.0):
        """Initialize unified memory manager."""
        self.max_memory_mb = float(max_memory_mb)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_history = deque(maxlen=200)
        self.cleanup_counter = 0
        
        # Cleanup thresholds (as fractions of max_memory_mb)
        self.cleanup_thresholds = {
            "light": max_memory_mb * 0.70,
            "medium": max_memory_mb * 0.85,
            "heavy": max_memory_mb * 0.95,
        }
        
        # Cleanup statistics
        self.cleanup_stats = {
            "light_cleanups": 0,
            "medium_cleanups": 0,
            "heavy_cleanups": 0,
            "memory_freed_mb": 0.0,
        }
        
        # Tracked components (weak references to prevent memory leaks)
        self.tracked_policies: List[weakref.ReferenceType] = []
        self.tracked_buffers: List[weakref.ReferenceType] = []
        self.tracked_envs: List[weakref.ReferenceType] = []
        self.tracked_caches: List[Any] = []
    
    def register_policy(self, policy: Any) -> None:
        """Register a policy for memory tracking."""
        self.tracked_policies.append(weakref.ref(policy))
    
    def register_buffer(self, buffer: Any) -> None:
        """Register a buffer for memory tracking."""
        self.tracked_buffers.append(weakref.ref(buffer))
    
    def register_env(self, env: Any) -> None:
        """Register an environment for memory tracking."""
        self.tracked_envs.append(weakref.ref(env))
    
    def register_cache(self, cache: Any) -> None:
        """Register a cache for memory tracking."""
        self.tracked_caches.append(cache)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss) / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def should_cleanup(self, force: bool = False) -> Tuple[Optional[str], float]:
        """Determine if cleanup is needed and at what level."""
        with self.lock:
            current_memory = self.get_memory_usage()
            self.memory_history.append(current_memory)
            self.cleanup_counter += 1
            
            if force:
                return "heavy", current_memory
            if current_memory > self.cleanup_thresholds["heavy"]:
                return "heavy", current_memory
            if current_memory > self.cleanup_thresholds["medium"]:
                return "medium", current_memory
            if current_memory > self.cleanup_thresholds["light"] or (self.cleanup_counter % 100 == 0):
                return "light", current_memory
            return None, current_memory
    
    def cleanup(self, level: str = "light") -> float:
        """Perform cleanup at specified level."""
        with self.lock:
            memory_before = self.get_memory_usage()
            
            try:
                if level in ("light", "medium", "heavy"):
                    self._cleanup_light()
                if level in ("medium", "heavy"):
                    self._cleanup_medium()
                if level == "heavy":
                    self._cleanup_heavy()
                
                self._cleanup_basic()
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")
            
            memory_after = self.get_memory_usage()
            memory_freed = max(0.0, memory_before - memory_after)
            
            # Update statistics
            key = f"{level}_cleanups"
            if key in self.cleanup_stats:
                self.cleanup_stats[key] += 1
            self.cleanup_stats["memory_freed_mb"] += memory_freed
            
            if memory_freed > 50:
                self.logger.info(
                    f"Memory cleanup ({level}): {memory_before:.1f}MB -> "
                    f"{memory_after:.1f}MB (freed {memory_freed:.1f}MB)"
                )
            
            return memory_freed
    
    def _cleanup_light(self) -> None:
        """Light cleanup: Clear caches."""
        for cache in self.tracked_caches:
            try:
                if hasattr(cache, 'clear'):
                    cache.clear()
            except Exception:
                pass
    
    def _cleanup_medium(self) -> None:
        """Medium cleanup: Clear TensorFlow session."""
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    
    def _cleanup_heavy(self) -> None:
        """Heavy cleanup: Force garbage collection and CUDA cache clear."""
        for _ in range(3):
            gc.collect()
        
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _cleanup_basic(self) -> None:
        """Basic cleanup: Garbage collection."""
        gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current = self.get_memory_usage()
        return {
            'current_memory_mb': current,
            'max_memory_mb': self.max_memory_mb,
            'memory_usage_pct': (current / self.max_memory_mb * 100) if self.max_memory_mb else 0.0,
            'tracked_policies': len([p for p in self.tracked_policies if p() is not None]),
            'tracked_buffers': len([b for b in self.tracked_buffers if b() is not None]),
            'tracked_envs': len([e for e in self.tracked_envs if e() is not None]),
            'tracked_caches': len(self.tracked_caches),
            'cleanup_stats': self.cleanup_stats.copy(),
            'memory_history': list(self.memory_history),
        }


# ============================================================================
# OBSERVATION VALIDATION (from observation_validator.py)
# ============================================================================

class UnifiedObservationValidator:
    """
    UNIFIED OBSERVATION VALIDATION
    
    Provides consistent observation validation across all modules.
    Validates shape, dtype, range, and finiteness of observations.
    """
    
    def __init__(self, env: Any = None, debug: bool = False):
        """Initialize observation validator."""
        self.env = env
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.validation_errors = deque(maxlen=100)
        self.validation_cache: Dict[Any, bool] = {}
    
    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> bool:
        """Validate a dictionary of observations."""
        if not isinstance(obs_dict, dict):
            self.logger.error(f"Observation dict is not a dict: {type(obs_dict)}")
            return False
        
        all_valid = True
        for agent_name, obs in obs_dict.items():
            if not self._validate_single_observation(agent_name, obs):
                all_valid = False
        
        return all_valid
    
    def _validate_single_observation(self, agent_name: str, obs: Any) -> bool:
        """Validate a single observation."""
        try:
            if not isinstance(obs, np.ndarray):
                self.logger.warning(f"{agent_name}: Observation is not numpy array: {type(obs)}")
                return False
            
            if len(obs.shape) != 1:
                self.logger.warning(f"{agent_name}: Observation is not 1D: shape={obs.shape}")
                return False
            
            if not np.all(np.isfinite(obs)):
                n_invalid = np.sum(~np.isfinite(obs))
                self.logger.warning(f"{agent_name}: {n_invalid}/{len(obs)} values are not finite")
                return False
            
            if len(obs) == 0:
                self.logger.warning(f"{agent_name}: Observation is empty")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"{agent_name}: Validation failed: {e}")
            return False
    
    def sanitize_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sanitize a dictionary of observations by fixing NaN/Inf values."""
        sanitized = {}
        for agent_name, obs in obs_dict.items():
            try:
                if isinstance(obs, np.ndarray):
                    sanitized[agent_name] = sanitize_array(obs)
                else:
                    sanitized[agent_name] = obs
            except Exception as e:
                self.logger.warning(f"Failed to sanitize {agent_name}: {e}")
                sanitized[agent_name] = obs
        
        return sanitized
    
    def validate_observation_shape(self, agent_name: str, obs: np.ndarray, 
                                   expected_shape: Tuple[int, ...]) -> bool:
        """Validate that observation matches expected shape."""
        if not isinstance(obs, np.ndarray):
            self.logger.warning(f"{agent_name}: Not a numpy array")
            return False
        
        if obs.shape != expected_shape:
            self.logger.warning(
                f"{agent_name}: Shape mismatch. Expected {expected_shape}, got {obs.shape}"
            )
            return False
        
        return True
    
    def validate_observation_range(self, agent_name: str, obs: np.ndarray,
                                   low: float = -100.0, high: float = 1000.0) -> bool:
        """Validate that observation values are within expected range."""
        if not isinstance(obs, np.ndarray):
            return False
        
        out_of_range = np.sum((obs < low) | (obs > high))
        if out_of_range > 0:
            self.logger.warning(
                f"{agent_name}: {out_of_range}/{len(obs)} values out of range [{low}, {high}]"
            )
            return False
        
        return True
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_errors': len(self.validation_errors),
            'recent_errors': list(self.validation_errors)[-10:],
            'cache_size': len(self.validation_cache),
        }


# ============================================================================
# DL OVERLAY SYSTEM (from dl_overlay.py)
# ============================================================================
# NOTE: DL overlay classes (OverlaySharedModel, DLAdapter, etc.) remain in dl_overlay.py
# due to complex TensorFlow keras.Model inheritance requirements.
# Import them from dl_overlay module when needed.


# ============================================================================
# EXPORTS
# ============================================================================

# ============================================================================
# TIER UTILITIES (migrated from tier_utils.py)
# ============================================================================

TIER_1 = "TIER_1"
TIER_2 = "TIER_2"
TIER_3 = "TIER_3"


def determine_tier(
    forecast_baseline_enable: bool,
    meta_baseline_enable: bool,
    dl_overlay_enabled: bool,
) -> str:
    """Return the paper-tier label for a run configuration.

    Tier meanings in the paper refactor:
    - TIER_1: MARL baseline (no forecast baseline/meta; no extra observations)
    - TIER_2: FGB online (forecast-guided baseline; no extra observations)
    - TIER_3: FGB meta (FAMC meta-critic enabled; no extra observations)
    """
    if meta_baseline_enable:
        return TIER_3
    if forecast_baseline_enable or dl_overlay_enabled:
        return TIER_2
    return TIER_1


def get_expected_observation_dims(tier: str, agent_name: str) -> int:
    base_dims = {
        'investor_0': 6,
        'battery_operator_0': 4,
        'risk_controller_0': 9,
        'meta_controller_0': 11,
    }
    # Observation shapes are fixed (Tier-1) across baseline and FGB/FAMC variants.
    # Tier-2 forecast-augmented observations are deprecated.
    return base_dims.get(agent_name, 6)


def get_tier_from_config(config) -> str:
    forecast_baseline_enable = getattr(config, 'forecast_baseline_enable', False)
    meta_baseline_enable = getattr(config, 'meta_baseline_enable', False)
    overlay_enabled = getattr(config, 'overlay_enabled', False)
    return determine_tier(forecast_baseline_enable, meta_baseline_enable, overlay_enabled)


def get_tier_from_env(env) -> str:
    if not hasattr(env, 'config'):
        return TIER_1
    config = env.config
    forecast_baseline_enable = getattr(config, 'forecast_baseline_enable', False)
    meta_baseline_enable = getattr(config, 'meta_baseline_enable', False)
    overlay_enabled = getattr(config, 'overlay_enabled', False)
    has_dl_adapter = hasattr(env, 'dl_adapter_overlay') and env.dl_adapter_overlay is not None
    return determine_tier(forecast_baseline_enable, meta_baseline_enable, overlay_enabled or has_dl_adapter)


def get_tier_description(tier: str) -> str:
    descriptions = {
        TIER_1: "MARL baseline (no forecast baseline/meta; no extra observations)",
        TIER_2: "MARL + FGB online (forecast-guided baseline; no extra observations)",
        TIER_3: "MARL + FGB meta (FAMC meta-critic; no extra observations)",
    }
    return descriptions.get(tier, f"Unknown tier: {tier}")

__all__ = [
    # Safe operations
    'SafeDivision', 'safe_clip', 'safe_mean', 'safe_std', 'safe_percentile',
    'sanitize_array', 'ensure_finite',

    # DL overlay utilities
    'load_overlay_weights', 'get_overlay_feature_dim', 'clear_tf_session', 'configure_tf_memory',

    # Error handling
    'ErrorHandler', 'safe_operation', 'validate_input', 'validate_type', 'validate_range',
    'validate_not_none', 'ContextualLogger',

    # Memory management
    'UnifiedMemoryManager',

    # Observation validation
    'UnifiedObservationValidator',

    # Tier utilities
    'TIER_1', 'TIER_2', 'TIER_3',
    'determine_tier', 'get_expected_observation_dims',
    'get_tier_from_config', 'get_tier_from_env', 'get_tier_description',
]

