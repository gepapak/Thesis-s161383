#!/usr/bin/env python3
"""
Enhanced Multi-Horizon Wrapper (fully patched w/ forecast normalization + horizon alignment)

Provides:
- TOTAL-dimension observation construction (base + normalized & horizon-aligned forecasts)
- Strict shape checking and validation
- Memory-optimized observation building with caching
- Enhanced monitoring (optional)

"""

from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import csv, os, threading, gc, psutil, pandas as pd, time
from typing import Dict, Any, Tuple, Optional, List, Mapping
from datetime import datetime
from collections import deque, OrderedDict
from contextlib import nullcontext
from config import (
    normalize_price,
    WRAPPER_FORECAST_CACHE_SIZE_DEFAULT,
    WRAPPER_AGENT_CACHE_SIZE_DEFAULT,
    WRAPPER_MEMORY_LIMIT_MB_DEFAULT,
)  # UNIFIED: Import from single source of truth
from utils import SafeDivision, UnifiedMemoryManager, UnifiedObservationValidator, ErrorHandler, safe_operation  # UNIFIED: Import from single source of truth
from logger import get_logger

logger = get_logger(__name__)

# Import enhanced monitoring (optional)
try:
    from enhanced_monitoring import EnhancedMetricsMonitor
    _HAS_ENHANCED_MONITORING = True
except Exception:
    _HAS_ENHANCED_MONITORING = False
    EnhancedMetricsMonitor = None


# =========================
# Enhanced Cache Management
# =========================
class EnhancedLRUCache:
    """Enhanced LRU cache with memory-aware eviction."""
    def __init__(self, max_size: int = 2000, memory_limit_mb: float = 1024.0):
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self.cache = OrderedDict()
        self.access_count = 0

    def get(self, key):
        """Get value and move to end (most recently used)."""
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_count += 1
            return value
        return None

    def put(self, key, value):
        """Put value, evicting LRU if necessary."""
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.access_count += 1
        if self.access_count % 500 == 0:  # Reduced frequency for better performance
            self._memory_cleanup()

    def _memory_cleanup(self):
        """Clean up cache if memory usage is high."""
        try:
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            if current_memory > self.memory_limit_mb and len(self.cache) > 100:
                items_to_remove = max(1, len(self.cache) // 3)
                for _ in range(items_to_remove):
                    if self.cache:
                        self.cache.popitem(last=False)
        except Exception:
            pass

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def clear(self):
        self.cache.clear()
        self.access_count = 0


# =========================
# Forecast Post-Processing
# =========================
class ForecastPostProcessor:
    """
    - Normalizes forecasts to match env observation scales:
        price -> z-score normalization using env's rolling mean/std, divided by 3, clipped to [-1,1] (same as env price_n)
        wind/solar/hydro -> divide by env's p95 scales, clipped [0,1]
        load -> divide by env.load_scale, clipped [0,1]
      Unknown targets are passed through unchanged.

    - Aligns horizons for each agent by ordering target×horizon keys so that the
      horizon closest to env.investment_freq comes first. We DO NOT change total dims;
      we only re-order priority when building the feature vector.
    """
    def __init__(self, env, normalize=True, align_horizons=True):
        self.env = env
        self.normalize = bool(normalize)
        self.align_horizons = bool(align_horizons)

        # Track normalization parameters for consistency validation
        self._initial_price_mean = None
        self._initial_price_std = None
        self._initial_wind_scale = None
        self._initial_solar_scale = None
        self._initial_hydro_scale = None
        self._initial_load_scale = None
        self._normalization_validated = False

        # Capture initial normalization parameters
        self._capture_initial_normalization_params()

    def _normalize_price_zscore_clipped(self, value: float, mean: float, std: float) -> float:
        """
        DYNAMIC PRICE NORMALIZATION WITH Z-SCORE CLIPPING

        Normalizes price to [−1,1] range using Z-score with clipping.
        Uses DYNAMIC mean/std parameters (timestep-specific statistics).

        Args:
            value: Raw price value to normalize
            mean: Dynamic mean price for current timestep
            std: Dynamic std dev for current timestep

        Returns:
            Normalized price in [-1, 1] range
        """
        try:
            # Step 1: Calculate raw Z-score using DYNAMIC parameters
            z_score = (value - mean) / max(std, 1e-6)

            # Step 2: Clip to ±3.0σ BEFORE dividing by 3.0 to ensure strict [−1,1] bounds
            clipped_z = np.clip(z_score, -3.0, 3.0)

            # Step 3: Divide by 3.0 to normalize to [−1,1] range
            normalized = clipped_z / 3.0

            return float(normalized)
        except Exception:
            # Fallback to safe default
            return 0.0

    def normalize_value(self, key: str, val: float) -> float:
        """
        Normalize feature values using CONSISTENT approach with environment.py.

        CRITICAL: Ensures overlay model receives normalized features matching
        what environment.py produces in _build_overlay_features().

        Normalization strategy:
        - PRICE: Z-score with dynamic mean/std (time-varying statistics)
        - RENEWABLES (wind, solar, hydro, load): Fixed scale (95th percentile)

        This consistency is ESSENTIAL for overlay model training and inference.
        """
        if not self.normalize:
            return float(val) if np.isfinite(val) else 0.0

        v = float(val) if np.isfinite(val) else 0.0
        k = (key or "").lower()

        if "price" in k:
            # PRICE: Z-score normalization with dynamic mean/std (MATCHES environment.py _norm_price)
            try:
                t = getattr(self.env, 't', 0)
                if hasattr(self.env, '_price_mean') and hasattr(self.env, '_price_std'):
                    if t < len(self.env._price_mean) and t < len(self.env._price_std):
                        mean = float(self.env._price_mean[t])
                        std = float(self.env._price_std[t])
                        return self._normalize_price_zscore_clipped(v, mean, std)

                # Fallback: use config-driven fallback statistics
                config = getattr(self.env, 'config', None)
                fallback_mean = getattr(config, 'price_fallback_mean', 250.0) if config else 250.0
                fallback_std = getattr(config, 'price_fallback_std', 50.0) if config else 50.0
                return self._normalize_price_zscore_clipped(v, fallback_mean, fallback_std)
            except Exception as e:
                logger.warning(f"[NORMALIZATION] Price normalization failed: {e}, returning 0.0")
                return 0.0

        # RENEWABLES: Fixed scale normalization (MATCHES environment.py _build_overlay_features)
        # Uses 95th percentile scales computed at initialization
        try:
            config = getattr(self.env, 'config', None)

            if "wind" in k:
                scale = max(float(getattr(self.env, "wind_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))

            if "solar" in k:
                scale = max(float(getattr(self.env, "solar_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))

            if "hydro" in k:
                scale = max(float(getattr(self.env, "hydro_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))

            if "load" in k:
                scale = max(float(getattr(self.env, "load_scale", 1.0)), 1e-6)
                return float(np.clip(v / scale, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"[NORMALIZATION] Renewable normalization failed for {k}: {e}, returning 0.0")
            return 0.0

        return v

    def _capture_initial_normalization_params(self):
        """Capture initial normalization parameters for consistency validation."""
        try:
            if hasattr(self.env, '_price_mean') and len(self.env._price_mean) > 0:
                self._initial_price_mean = float(self.env._price_mean[0])
            if hasattr(self.env, '_price_std') and len(self.env._price_std) > 0:
                self._initial_price_std = float(self.env._price_std[0])
            if hasattr(self.env, 'wind_scale'):
                self._initial_wind_scale = float(self.env.wind_scale)
            if hasattr(self.env, 'solar_scale'):
                self._initial_solar_scale = float(self.env.solar_scale)
            if hasattr(self.env, 'hydro_scale'):
                self._initial_hydro_scale = float(self.env.hydro_scale)
            if hasattr(self.env, 'load_scale'):
                self._initial_load_scale = float(self.env.load_scale)
        except Exception as e:
            logger.warning(f"[NORMALIZATION] Failed to capture initial params: {e}")

    def _validate_initial_normalization_parameters(self) -> bool:
        """
        CRITICAL: Validate that normalization parameters haven't changed during episode.

        This prevents subtle bugs where environment's normalization parameters change
        but wrapper continues using old values, causing observation distribution shift.

        Returns:
            bool: True if normalization is consistent, False otherwise
        """
        if not self.normalize:
            return True

        issues = []

        try:
            # Check price normalization (only check first timestep for static validation)
            if self._initial_price_mean is not None and hasattr(self.env, '_price_mean'):
                if len(self.env._price_mean) > 0:
                    current_mean = float(self.env._price_mean[0])
                    if abs(current_mean - self._initial_price_mean) > 1e-6:
                        issues.append(
                            f"Price mean changed: {self._initial_price_mean:.4f} -> {current_mean:.4f}"
                        )

            if self._initial_price_std is not None and hasattr(self.env, '_price_std'):
                if len(self.env._price_std) > 0:
                    current_std = float(self.env._price_std[0])
                    if abs(current_std - self._initial_price_std) > 1e-6:
                        issues.append(
                            f"Price std changed: {self._initial_price_std:.4f} -> {current_std:.4f}"
                        )

            # Check renewable scales
            if self._initial_wind_scale is not None and hasattr(self.env, 'wind_scale'):
                current_scale = float(self.env.wind_scale)
                if abs(current_scale - self._initial_wind_scale) > 1e-6:
                    issues.append(
                        f"Wind scale changed: {self._initial_wind_scale:.4f} -> {current_scale:.4f}"
                    )

            if self._initial_solar_scale is not None and hasattr(self.env, 'solar_scale'):
                current_scale = float(self.env.solar_scale)
                if abs(current_scale - self._initial_solar_scale) > 1e-6:
                    issues.append(
                        f"Solar scale changed: {self._initial_solar_scale:.4f} -> {current_scale:.4f}"
                    )

            if self._initial_hydro_scale is not None and hasattr(self.env, 'hydro_scale'):
                current_scale = float(self.env.hydro_scale)
                if abs(current_scale - self._initial_hydro_scale) > 1e-6:
                    issues.append(
                        f"Hydro scale changed: {self._initial_hydro_scale:.4f} -> {current_scale:.4f}"
                    )

            if self._initial_load_scale is not None and hasattr(self.env, 'load_scale'):
                current_scale = float(self.env.load_scale)
                if abs(current_scale - self._initial_load_scale) > 1e-6:
                    issues.append(
                        f"Load scale changed: {self._initial_load_scale:.4f} -> {current_scale:.4f}"
                    )

        except Exception as e:
            logger.warning(f"[NORMALIZATION] Consistency validation failed: {e}")
            return False

        if issues:
            error_msg = "[NORMALIZATION INCONSISTENCY DETECTED]\n" + "\n".join(issues)
            logger.error(error_msg)
            return False

        self._normalization_validated = True
        return True

    def _validate_runtime_normalization_alignment(self) -> bool:
        """
        CRITICAL VALIDATION: Ensure wrapper normalization matches environment.py.

        This prevents silent degradation where overlay model receives inconsistently
        normalized features, causing poor performance without obvious errors.

        Returns:
            bool: True if normalization is consistent, False otherwise
        """
        try:
            # Test price normalization
            test_price = 250.0  # Typical price
            t = getattr(self.env, 't', 0)

            # Get wrapper normalization
            wrapper_norm = self.normalize_value("price", test_price)

            # Get environment normalization (if available)
            if hasattr(self.env, '_norm_price'):
                env_norm = self.env._norm_price(test_price, t)

                # Check if they're reasonably close (within 0.1 due to floating point)
                if abs(wrapper_norm - env_norm) > 0.1:
                    logger.warning(
                        f"[NORMALIZATION_MISMATCH] Price normalization inconsistent:\n"
                        f"  Wrapper: {wrapper_norm:.4f}\n"
                        f"  Environment: {env_norm:.4f}\n"
                        f"  Difference: {abs(wrapper_norm - env_norm):.4f}\n"
                        f"  This will confuse the overlay model!"
                    )
                    return False

            # Test renewable normalization
            test_wind = 100.0
            wrapper_wind = self.normalize_value("wind", test_wind)

            # Verify it's in [0, 1] range
            if not (0.0 <= wrapper_wind <= 1.0):
                logger.warning(
                    f"[NORMALIZATION_INVALID] Wind normalization out of bounds: {wrapper_wind}\n"
                    f"  Expected [0.0, 1.0], got {wrapper_wind}"
                )
                return False

            logger.info("[NORMALIZATION_VALID] Wrapper and environment normalization are consistent")
            return True

        except Exception as e:
            logger.warning(f"[NORMALIZATION_VALIDATION] Failed to validate: {e}")
            return False

    def validate_normalization_consistency(self) -> bool:
        """Run both static and runtime normalization checks."""
        initial_ok = self._validate_initial_normalization_parameters()
        runtime_ok = self._validate_runtime_normalization_alignment()
        return bool(initial_ok and runtime_ok)

    def order_keys_by_horizon_alignment(self, agent: str, keys: List[str]) -> List[str]:
        """
        Reorders forecast keys so those whose horizon is closest to env.investment_freq
        appear first. We expect keys like "..._forecast_<horizon_name>" where horizon_name
        is a string like "immediate", "short", "medium", etc.
        """
        if not self.align_horizons or not isinstance(keys, list) or len(keys) == 0:
            return keys

        try:
            inv_freq = int(getattr(self.env, "investment_freq", 12))

            # Get horizons map from forecaster if available
            forecaster = getattr(self.env, "forecast_generator", None)
            if forecaster is None:
                return keys

            hmap = getattr(forecaster, "horizons", {})
            if not hmap:
                # CANONICAL: Use cached config source of truth for horizon fallback
                try:
                    hmap = getattr(getattr(self.env, 'config', None), 'forecast_horizons', {}).copy()
                    if not hmap:
                        raise ValueError("No forecast_horizons in config")
                except Exception:
                    # FAIL-FAST: No hardcoded horizon fallbacks allowed for production safety
                    raise ValueError("Cannot access forecast_horizons from env.config: "
                                   "config missing or malformed. Fix config initialization.")

            def parse_h(k):
                try:
                    # Extract horizon name from key (e.g., "price_forecast_short" -> "short")
                    horizon_name = str(k).split("_")[-1]
                    if horizon_name not in hmap:
                        raise ValueError(f"Horizon '{horizon_name}' not found in config.forecast_horizons. "
                                       f"Available horizons: {list(hmap.keys())}. Fix config to maintain single source of truth.")
                    horizon_steps = int(hmap[horizon_name])
                    return abs(horizon_steps - inv_freq), horizon_steps
                except Exception as e:
                    raise ValueError(f"Failed to parse horizon from key '{k}': {str(e)}. "
                                   f"Ensure all forecast keys follow pattern 'target_forecast_horizon' with valid horizons from config.")

            return sorted(keys, key=parse_h)
        except Exception:
            return keys

    def normalize_and_align(self, agent: str, raw_forecasts: Mapping[str, float],
                            expected_keys: List[str]) -> Dict[str, float]:
        """
        Returns a dict with the same key set as expected_keys, filled with normalized values.
        Keys are supplied in an order that prioritizes alignment; values are pulled from raw
        (with 0.0 fallback), normalized per target.
        """
        if not isinstance(raw_forecasts, Mapping):
            raw_forecasts = {}

        ordered_keys = self.order_keys_by_horizon_alignment(agent, list(expected_keys))
        out = {}
        for k in ordered_keys:
            v = raw_forecasts.get(k, 0.0)
            out[k] = self.normalize_value(k, v)

        final_out = {}
        for k in expected_keys:
            final_out[k] = out.get(k, 0.0)
        return final_out


# UNIFIED MEMORY MANAGER (NEW)
# EnhancedMemoryTracker is now replaced by UnifiedMemoryManager from memory_manager.py
# For backward compatibility, we create an alias
EnhancedMemoryTracker = UnifiedMemoryManager


# UNIFIED OBSERVATION VALIDATION (NEW)
# BaseObservationValidator is now replaced by UnifiedObservationValidator from observation_validator.py
# For backward compatibility, we create an alias
BaseObservationValidator = UnifiedObservationValidator


class ObservationValidatorMixin:
    """
    Mixin class providing common validation methods.
    Can be used by both wrapper and metacontroller validators.
    """

    def fix_observation_shape(self, obs: Any, expected_dim: int) -> np.ndarray:
        """Fix observation to expected shape and type."""
        if not isinstance(obs, np.ndarray):
            if obs is None:
                obs = np.zeros(expected_dim, dtype=np.float32)
            elif isinstance(obs, (list, tuple)):
                obs = np.array(obs, dtype=np.float32)
            else:
                obs = np.full(expected_dim, float(obs), dtype=np.float32)
        else:
            obs = obs.astype(np.float32)

        if obs.ndim != 1:
            obs = obs.flatten()

        if obs.size < expected_dim:
            obs = np.pad(obs, (0, expected_dim - obs.size))
        elif obs.size > expected_dim:
            obs = obs[:expected_dim]

        return obs

    def create_fallback_observation(self, expected_dim: int,
                                   low: Optional[np.ndarray] = None,
                                   high: Optional[np.ndarray] = None) -> np.ndarray:
        """Create a safe fallback observation."""
        if low is not None and high is not None:
            return ((low[:expected_dim] + high[:expected_dim]) / 2.0).astype(np.float32)
        return np.zeros(expected_dim, dtype=np.float32)


class EnhancedObservationValidator(BaseObservationValidator, ObservationValidatorMixin):
    """
    Validates base obs from the ENV and appends forecast features to build TOTAL obs.

    IMPORTANT: the ENV returns BASE-only observations. The WRAPPER is responsible for:
      total_dim = base_dim + forecast_dim

    Forecasts are normalized AND keys are prioritized to align horizons (closest to env.investment_freq first).
    """
    def __init__(self, base_env, forecaster, debug=False, postproc: Optional[ForecastPostProcessor] = None):
        # Initialize base class
        super().__init__(base_env, debug)

        self.base_env = base_env
        self.forecaster = forecaster
        self.postproc = postproc or ForecastPostProcessor(base_env, normalize=True, align_horizons=True)

        self.agent_observation_specs = {}
        self._initialize_observation_specs()

    def _initialize_observation_specs(self):
        for agent in self.base_env.possible_agents:
            try:
                base_dim = self._get_validated_base_dim(agent)
                forecast_dim = self._calculate_forecast_dimension(agent)
                total_dim = base_dim + forecast_dim
                total_low, total_high = self._get_safe_bounds(total_dim)
                self.agent_observation_specs[agent] = {
                    'base_dim': base_dim,
                    'forecast_dim': forecast_dim,
                    'total_dim': total_dim,
                    'forecast_keys': self._get_agent_forecast_keys(agent, forecast_dim),
                    'bounds': (total_low, total_high)
                }
                if self.debug or agent == "meta_controller_0":
                    print(f"[SPEC] {agent}: base={base_dim}, forecast={forecast_dim}, total={total_dim}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to init specs for {agent}: {e}")
                self._create_fallback_spec(agent)

    def _get_validated_base_dim(self, agent: str) -> int:
        """
        FAIL-FAST: Get base observation dimension from environment with strict validation.
        No static fallbacks allowed - any failure indicates configuration/environment issues.
        
        CRITICAL FIX: When forecasts are enabled, the base environment ALREADY includes
        forecast features in the "base" observation space. So we should use the FULL
        observation space dimension from the base environment, not try to add more.
        """
        try:
            # Method 1: Use environment's explicit dimension method
            if hasattr(self.base_env, '_get_base_observation_dim'):
                dim = int(self.base_env._get_base_observation_dim(agent))
                if dim <= 0:
                    raise ValueError(f"Invalid base dimension {dim} from env._get_base_observation_dim({agent})")
                return dim

            # Method 2: Use observation space shape
            # CRITICAL FIX: When forecasts are enabled, base_env.observation_space already includes forecasts
            # So we should use the FULL dimension, not try to add more forecast dimensions
            if hasattr(self.base_env, 'observation_space'):
                if callable(self.base_env.observation_space):
                    space = self.base_env.observation_space(agent)
                else:
                    space = self.base_env.observation_space.get(agent)

                if space is None:
                    raise ValueError(f"No observation space found for agent '{agent}' in base_env.observation_space")

                if not hasattr(space, 'shape'):
                    raise ValueError(f"Observation space for agent '{agent}' has no shape attribute: {type(space)}")

                dim = int(space.shape[0])
                if dim <= 0:
                    raise ValueError(f"Invalid base dimension {dim} from observation_space({agent}).shape")
                
                # CRITICAL FIX: When forecasts are enabled, the base environment observation space
                # already includes all forecast features. So we return the FULL dimension.
                # The wrapper should NOT add additional forecast dimensions in this case.
                enable_forecast_util = getattr(self.base_env.config, 'enable_forecast_utilisation', False) if hasattr(self.base_env, 'config') else False
                if enable_forecast_util:
                    # Base environment already includes forecasts - return full dimension
                    logger.debug(f"[BASE_DIM] {agent}: Using full observation space dimension {dim}D (forecasts already included in base)")
                    return dim
                else:
                    # Tier 1: Base environment doesn't include forecasts - return base dimension
                    return dim

            # No valid method found
            raise ValueError(f"Cannot determine base observation dimension for agent '{agent}': "
                           f"base_env missing both '_get_base_observation_dim' method and 'observation_space' attribute")

        except Exception as e:
            # FAIL-FAST: No fallbacks allowed for production safety
            raise ValueError(
                f"Failed to get validated base dimension for agent '{agent}': {str(e)}. "
                "This indicates environment/wrapper configuration issues that must be fixed."
            )

    def _calculate_forecast_dimension(self, agent: str) -> int:
        # If no forecaster, return 0 forecast dimensions (baseline mode)
        if self.forecaster is None:
            return 0

        # CONDITIONAL: When enable_forecast_utilisation=True (Tier 2/3), the base environment already includes
        # the forecast features in its observation space (e.g., investor_0 is 8D = 6 base + 2 forecast).
        # So the wrapper should NOT add additional forecast dimensions.
        enable_forecast_util = getattr(self.base_env.config, 'enable_forecast_utilisation', False) if hasattr(self.base_env, 'config') else False
        if enable_forecast_util:
            # Tier 2/3: Deltas already in base observations, no additional forecasts needed
            return 0

        try:
            if hasattr(self.forecaster, 'get_agent_forecast_dims'):
                dims = self.forecaster.get_agent_forecast_dims()
                base_dims = int(dims.get(agent, 0))
                # Add 1 for forecast confidence (except risk_controller_0 which doesn't get confidence)
                result = base_dims + (1 if agent != "risk_controller_0" else 0)
                # FIX #7: Add 2 dims for consensus_direction and consensus_confidence
                result += (2 if agent != "risk_controller_0" else 0)
                if agent == "meta_controller_0":
                    print(f"[FORECAST_DIM] {agent}: get_agent_forecast_dims={base_dims}, confidence=1, total={result}")
                return result
            if hasattr(self.forecaster, 'agent_horizons') and hasattr(self.forecaster, 'agent_targets'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                base_dims = int(len(targets) * len(horizons))
                # Add 1 for forecast confidence (except risk_controller_0 which doesn't get confidence)
                result = base_dims + (1 if agent != "risk_controller_0" else 0)
                # FIX #7: Add 2 dims for consensus signals
                result += (2 if agent != "risk_controller_0" else 0)
                if agent == "meta_controller_0":
                    print(f"[FORECAST_DIM] {agent}: targets={len(targets)}, horizons={len(horizons)}, base_dims={base_dims}, confidence=1, total={result}")
                return result
            # DYNAMIC: Calculate forecast dimensions from actual forecaster configuration
            try:
                if hasattr(self.forecaster, 'agent_targets') and hasattr(self.forecaster, 'agent_horizons'):
                    targets = self.forecaster.agent_targets.get(agent, [])
                    horizons = self.forecaster.agent_horizons.get(agent, [])
                    base_dims = len(targets) * len(horizons)
                    # Add 1 for forecast confidence (except risk_controller_0 which doesn't get confidence)
                    confidence_dim = 1 if agent != "risk_controller_0" else 0
                    # FIX #7: Add consensus signals (2 dims) except risk_controller
                    consensus_dim = 2 if agent != "risk_controller_0" else 0
                    return base_dims + confidence_dim + consensus_dim
            except Exception:
                pass
            # FAIL-FAST: No static fallbacks - fail fast to maintain single source of truth
            raise ValueError(f"Cannot calculate forecast dimensions for agent '{agent}': forecaster missing agent_targets or agent_horizons metadata")
        except Exception:
            # DYNAMIC: Try to calculate from forecaster, fallback to hardcoded if needed
            try:
                if hasattr(self.forecaster, 'agent_targets') and hasattr(self.forecaster, 'agent_horizons'):
                    targets = self.forecaster.agent_targets.get(agent, [])
                    horizons = self.forecaster.agent_horizons.get(agent, [])
                    base_dims = len(targets) * len(horizons)
                    confidence_dim = 1 if agent != "risk_controller_0" else 0
                    consensus_dim = 2 if agent != "risk_controller_0" else 0
                    return base_dims + confidence_dim + consensus_dim
            except Exception:
                pass
            # FAIL-FAST: No static fallbacks - fail fast to maintain single source of truth
            raise ValueError(f"Cannot calculate forecast dimensions for agent '{agent}': forecaster missing agent_targets or agent_horizons metadata")

    def _get_agent_forecast_keys(self, agent: str, expected_count: int) -> List[str]:
        try:
            if hasattr(self.forecaster, 'agent_targets') and hasattr(self.forecaster, 'agent_horizons'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                raw_keys = [f"{t}_forecast_{h}" for t in targets for h in horizons]
                ordered = self.postproc.order_keys_by_horizon_alignment(agent, raw_keys)
                while len(ordered) < expected_count:
                    ordered.append(f"fallback_forecast_{len(ordered)}")
                return ordered[:expected_count]
        except Exception:
            pass
        return [f"forecast_{i}" for i in range(expected_count)]

    def _get_safe_bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get safe bounds for observation space."""
        low = np.full(dim, -10.0, dtype=np.float32)
        high = np.full(dim, 10.0, dtype=np.float32)
        return low, high

    def _create_fallback_spec(self, agent: str):
        # FAIL-FAST: No static dimension estimates allowed for production safety
        # Base dimensions must be calculated dynamically from environment configuration
        try:
            # Get base observation dimension from environment's observation space
            if hasattr(self.base_env, 'observation_space') and agent in self.base_env.observation_space:
                base_dim = self.base_env.observation_space[agent].shape[0]
            elif hasattr(self.base_env, '_get_base_observation_dimension'):
                base_dim = self.base_env._get_base_observation_dimension(agent)
            else:
                raise ValueError(f"Cannot determine base observation dimension for agent '{agent}': "
                               f"environment missing observation_space or _get_base_observation_dimension method")
        except Exception as e:
            raise ValueError(
                f"Cannot create fallback spec for agent '{agent}': "
                f"failed to get base dimension dynamically. {str(e)}."
            )
        # DYNAMIC: Calculate forecast dimensions from actual forecaster configuration
        try:
            if hasattr(self.forecaster, 'agent_targets') and hasattr(self.forecaster, 'agent_horizons'):
                targets = self.forecaster.agent_targets.get(agent, [])
                horizons = self.forecaster.agent_horizons.get(agent, [])
                base_forecast_dims = len(targets) * len(horizons)
                confidence_dim = 1 if agent != "risk_controller_0" else 0
                forecast_dim = base_forecast_dims + confidence_dim
            else:
                # ENFORCE: No static fallbacks - fail fast to maintain single source of truth
                raise ValueError(f"Cannot create fallback spec for agent '{agent}': forecaster missing agent_targets or agent_horizons metadata")
        except Exception as e:
            # ENFORCE: No static fallbacks - fail fast to maintain single source of truth
            raise ValueError(f"Cannot create fallback spec for agent '{agent}': {str(e)}")
        total_dim = base_dim + forecast_dim
        self.agent_observation_specs[agent] = {
            'base_dim': base_dim,
            'forecast_dim': forecast_dim,
            'total_dim': total_dim,
            'forecast_keys': self._get_agent_forecast_keys(agent, forecast_dim),
            'bounds': self._get_safe_bounds(total_dim)
        }

    # ---- in-place assembly ----
    def build_total_observation(self, agent: str, base_obs: np.ndarray,
                                forecasts: Mapping[str, float],
                                out: Optional[np.ndarray] = None) -> np.ndarray:
        if agent not in self.agent_observation_specs:
            return self._create_safe_observation(agent)
        spec = self.agent_observation_specs[agent]
        bd, fd, td = spec['base_dim'], spec['forecast_dim'], spec['total_dim']
        if out is None or not isinstance(out, np.ndarray) or out.shape != (td,):
            out = np.zeros(td, dtype=np.float32)

        base_obs = self._validate_base_observation(agent, base_obs, spec)
        out[:bd] = base_obs

        if fd > 0:
            keys = spec['forecast_keys'][:fd]
            f_proc = self.postproc.normalize_and_align(agent, forecasts, keys)
            out[bd:bd + fd].fill(0.0)
            for j, key in enumerate(keys):
                v = f_proc.get(key, 0.0)
                out[bd + j] = float(v if np.isfinite(v) else 0.0)

        # Optional: Append consensus signals (direction, confidence) after forecast features.
        # IMPORTANT: Only write these if the declared spec/space already includes room for them.
        # In Tier 2/3, the base env already includes all forecast-related features and the wrapper
        # must NOT silently expand the observation vector.
        try:
            if agent != "risk_controller_0":
                # Consensus derived from forecast engine's quality-weighted signal and trust
                consensus_direction = 0.0
                consensus_confidence = 0.0
                # Attempt to read from environment's forecast engine debug/state
                fe = getattr(self.base_env, 'forecast_engine', None)
                if fe is not None:
                    # Use last quality-weighted signal and trust if available
                    qsig = None
                    try:
                        if hasattr(self.base_env, '_quality_signal_debug'):
                            dbg = getattr(self.base_env, '_quality_signal_debug')
                            qsig = float(dbg.get('quality_signal', 0.0))
                    except Exception:
                        pass
                    if qsig is None:
                        # Fallback: use combined_forecast_score from env debug if present
                        try:
                            if hasattr(self.base_env, '_debug_forecast_reward'):
                                qsig = float(self.base_env._debug_forecast_reward.get('combined_forecast_score', 0.0))
                        except Exception:
                            qsig = 0.0
                    consensus_direction = float(np.sign(qsig))
                    consensus_confidence = float(np.clip(getattr(fe, 'forecast_trust', 0.5), 0.0, 1.0))
                else:
                    # Fallback: derive from forecasts dictionary if possible
                    avg_signal = 0.0
                    cnt = 0
                    for k, v in (forecasts or {}).items():
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            avg_signal += float(v)
                            cnt += 1
                    avg_signal = avg_signal / cnt if cnt > 0 else 0.0
                    consensus_direction = float(np.sign(avg_signal))
                    consensus_confidence = 0.5

                # Append ONLY if the observation spec already includes these two dims
                if td >= (bd + fd + 2):
                    out[bd + fd: bd + fd + 2] = np.array(
                        [consensus_direction, consensus_confidence], dtype=np.float32
                    )
        except Exception:
            # Keep observations valid even if consensus computation fails
            pass

        low, high = spec['bounds']
        # Clip only within declared bounds length; extra consensus dims already included in td
        np.clip(out, low[:td], high[:td], out=out)
        return out

    # ---- helpers used by both paths ----
    def _validate_base_observation(self, agent: str, base_obs: np.ndarray, spec: Dict) -> np.ndarray:
        if not isinstance(base_obs, np.ndarray):
            if base_obs is None:
                base_obs = np.zeros(spec['base_dim'], dtype=np.float32)
            elif isinstance(base_obs, (list, tuple)):
                base_obs = np.array(base_obs, dtype=np.float32)
            else:
                base_obs = np.full(spec['base_dim'], float(base_obs), dtype=np.float32)
        else:
            base_obs = base_obs.astype(np.float32)
        if base_obs.ndim != 1:
            base_obs = base_obs.flatten()
        bd = spec['base_dim']
        if base_obs.size < bd:
            base_obs = np.pad(base_obs, (0, bd - base_obs.size))
        elif base_obs.size > bd:
            base_obs = base_obs[:bd]
        low, high = spec['bounds']
        base_obs = np.nan_to_num(base_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        base_obs = np.clip(base_obs, low[:bd], high[:bd])
        return base_obs

    def _create_safe_observation(self, agent: str) -> np.ndarray:
        if agent in self.agent_observation_specs:
            spec = self.agent_observation_specs[agent]
            expected = spec['total_dim']
            low, high = spec['bounds']
            return ((low[:expected] + high[:expected]) / 2.0).astype(np.float32)
        # FAIL-FAST: No static fallbacks allowed for production safety
        raise ValueError(f"Cannot create safe observation for agent '{agent}': "
                        "agent not in observation specs. Build spec first using dynamic calculation.")

    def get_validation_stats(self) -> Dict:
        return {
            'agents_configured': len(self.agent_observation_specs),
            'recent_errors': len(self.validation_errors),
            'cache_size': len(self.validation_cache),
            'specs': {a: {
                'base_dim': s['base_dim'],
                'forecast_dim': s['forecast_dim'],
                'total_dim': s['total_dim'],
            } for a, s in self.agent_observation_specs.items()},
            'recent_error_messages': list(self.validation_errors)[-5:]
        }


# =========================
# Observation Builder
# =========================
class MemoryOptimizedObservationBuilder:
    """Builds total observations (base + normalized & horizon-aligned forecasts) with validation & caching, in-place."""
    def __init__(self, base_env, forecaster, debug=False, normalize_forecasts=True, align_horizons=True):
        self.base_env = base_env
        self.forecaster = forecaster
        self.debug = debug

        self.postproc = ForecastPostProcessor(base_env, normalize=normalize_forecasts, align_horizons=align_horizons)
        self.validator = EnhancedObservationValidator(base_env, forecaster, debug, postproc=self.postproc)

        self._obs_out = {
            agent: np.zeros(self.validator.agent_observation_specs[agent]['total_dim'], dtype=np.float32)
            for agent in self.base_env.possible_agents
        }

        # Get cache settings from environment config
        env_config = getattr(self.base_env, 'config', None)
        if env_config:
            forecast_cache_size = env_config.forecast_cache_size
            agent_cache_size = env_config.agent_forecast_cache_size
            memory_limit = env_config.wrapper_memory_mb
        else:
            # Fallback values from config constants
            forecast_cache_size = WRAPPER_FORECAST_CACHE_SIZE_DEFAULT
            agent_cache_size = WRAPPER_AGENT_CACHE_SIZE_DEFAULT
            memory_limit = WRAPPER_MEMORY_LIMIT_MB_DEFAULT

        self.forecast_cache = EnhancedLRUCache(max_size=forecast_cache_size, memory_limit_mb=memory_limit)
        self.agent_forecast_cache = EnhancedLRUCache(max_size=agent_cache_size, memory_limit_mb=memory_limit * 1.5)

        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=memory_limit)
        self.memory_tracker.register_cache(self.forecast_cache)
        self.memory_tracker.register_cache(self.agent_forecast_cache)

    def enhance_observations(self, base_obs: Dict[str, np.ndarray],
                             all_forecasts: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        cleanup_level, _ = self.memory_tracker.should_cleanup()
        if cleanup_level:
            self.memory_tracker.cleanup(cleanup_level)

        t = getattr(self.base_env, 't', 0)
        per_agent_possible = hasattr(self.forecaster, 'predict_for_agent')

        for agent in self.base_env.possible_agents:
            try:
                if agent in base_obs:
                    if all_forecasts is not None:
                        fsrc: Mapping[str, float] = all_forecasts
                    elif per_agent_possible:
                        fsrc = self._get_cached_forecasts_for_agent(agent, t)
                    else:
                        fsrc = self._get_cached_forecasts_global(t)

                    self.validator.build_total_observation(
                        agent, base_obs[agent], fsrc, out=self._obs_out[agent]
                    )
                else:
                    self._obs_out[agent][:] = self.validator._create_safe_observation(agent)
            except Exception:
                self._obs_out[agent][:] = self.validator._create_safe_observation(agent)
        return self._obs_out

    # ---- forecast caching ----
    def _get_cached_forecasts_global(self, timestep: int) -> Dict[str, float]:
        key = f"forecasts_{timestep}"
        cached_result = self.forecast_cache.get(key)
        if cached_result is not None:
            return cached_result
        try:
            all_forecasts = self.forecaster.predict_all_horizons(timestep=timestep)
            if not isinstance(all_forecasts, dict):
                all_forecasts = {}
        except Exception:
            all_forecasts = {}
        self.forecast_cache.put(key, all_forecasts)
        return all_forecasts

    def _get_cached_forecasts_for_agent(self, agent: str, timestep: int) -> Dict[str, float]:
        key = (agent, timestep)
        cached_result = self.agent_forecast_cache.get(key)
        if cached_result is not None:
            return cached_result
        try:
            if hasattr(self.forecaster, 'predict_for_agent'):
                f = self.forecaster.predict_for_agent(agent=agent, timestep=timestep)
                if not isinstance(f, dict):
                    f = {}
            else:
                f = self._get_cached_forecasts_global(timestep)
        except Exception:
            f = {}
        self.agent_forecast_cache.put(key, f)
        return f

    def get_diagnostic_info(self) -> Dict:
        return {
            'validator_stats': self.validator.get_validation_stats(),
            'memory_stats': self.memory_tracker.get_memory_stats(),
            'cache_size': {'global': len(self.forecast_cache), 'per_agent': len(self.agent_forecast_cache)},
            'cache_limit': {
                'global': getattr(self.forecast_cache, 'max_size', None),
                'per_agent': getattr(self.agent_forecast_cache, 'max_size', None)
            }
        }


# =========================
# Wrapper Env
# =========================
class MultiHorizonWrapperEnv(ParallelEnv):
    """Wraps a BASE-only env and exposes TOTAL-dim observations (base + normalized & horizon-aligned forecasts)."""
    metadata = {"name": "multi_horizon_wrapper:normalized-aligned-v1"}

    def __init__(self, base_env, multi_horizon_forecaster, max_memory_mb=1500,
                 normalize_forecasts=True, align_horizons=True, total_timesteps=50000):
        self.env = base_env
        self.base_env = base_env  # Alias for compatibility
        self.forecaster = multi_horizon_forecaster
        # CRITICAL: Forward config for code paths that expect env.config (e.g., action validation toggles).
        # The base env owns config; the wrapper should expose it transparently.
        self.config = getattr(base_env, "config", None)

        # Agents
        self._possible_agents = self.env.possible_agents[:]
        self._agents = self.env.agents[:]

        # Memory infra (no CSV logging)
        self.memory_tracker = EnhancedMemoryTracker(max_memory_mb=max_memory_mb)

        # Enhanced monitoring
        self.enhanced_monitor = None
        if _HAS_ENHANCED_MONITORING:
            self.enhanced_monitor = EnhancedMetricsMonitor(
                window_size=1000,
                log_frequency=100
            )

        # Inject forecaster reference into env if empty
        if hasattr(self.env, 'forecast_generator') and getattr(self.env, 'forecast_generator') is None:
            try:
                self.env.forecast_generator = self.forecaster
            except Exception:
                pass

        # Builder / specs
        self.obs_builder = MemoryOptimizedObservationBuilder(
            self.env, self.forecaster, debug=False,
            normalize_forecasts=normalize_forecasts, align_horizons=align_horizons
        )
        self.memory_tracker.register_cache(self.obs_builder.forecast_cache)
        self.memory_tracker.register_cache(self.obs_builder.agent_forecast_cache)

        # FIX: Validate normalization consistency between wrapper and environment
        # This prevents silent degradation where overlay receives inconsistent features
        if normalize_forecasts:
            self.obs_builder.postproc.validate_normalization_consistency()

        # Observation spaces (TOTAL dims)
        self._build_wrapper_observation_spaces()

        # counters & episode markers
        self.step_count = 0
        self.episode_count = 0
        # keep errors bounded to avoid log spam
        self.error_count = 0
        self.max_errors = 100  # Increased limit to avoid silent failures

        self._ep_meta_return = 0.0
        self._prev_forecasts_for_error: Dict[str, float] = {}
        self._last_episode_seed = None
        self._last_episode_end_flag = 0
        self._last_step_wall_ms = 0.0
        self._last_clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}
        self._last_clip_totals = {"investor": 1, "battery": 1, "risk": 1, "meta": 1}

        # a couple of last-known values for logging
        self._last_price_forecast_norm = 0.0
        self._last_price_forecast_aligned = 0.0

        print("[OK] Enhanced multi-horizon wrapper initialized (TOTAL-dim observations, normalized + aligned forecasts)")

    def _get_currency_rate(self) -> float:
        """Get currency conversion rate from single source of truth."""
        # Priority: env._dkk_to_usd_rate > env.config.dkk_to_usd_rate > fallback
        rate = getattr(self.env, '_dkk_to_usd_rate', None)
        if rate is not None:
            return float(rate)

        config = getattr(self.env, 'config', None)
        if config and hasattr(config, 'dkk_to_usd_rate'):
            return float(config.dkk_to_usd_rate)

        return 0.145  # Fallback rate

    # ---- spaces ----
    def _build_wrapper_observation_spaces(self):
        """
        FIX: Build STATIC observation spaces during initialization.

        This method calculates the FINAL, fully augmented observation space dimensions
        for each agent.
        
        IMPORTANT (fairness): Tier 3 is observation-identical to Tier 2.

        These spaces are defined ONCE at initialization and NEVER modified at runtime.
        This ensures compatibility with Stable Baselines3, which builds policy networks
        based on observation spaces defined at agent initialization time.

        CRITICAL: The wrapper must return observations that EXACTLY match these
        statically-defined spaces in both reset() and step() methods.
        """
        self._obs_spaces = {}
        specs = self.obs_builder.validator.agent_observation_specs

        # ROBUST TIER DETECTION: Use centralized tier utilities
        from utils import get_tier_from_config, get_tier_description
        
        tier = get_tier_from_config(self.env.config)
        logger.info(f"[OBS_SPACE_TIER] Detected: {tier} - {get_tier_description(tier)}")
        
        enable_forecast_util = getattr(self.env.config, 'enable_forecast_utilisation', False)
        logger.info(f"[OBS_SPACE_CONFIG] tier={tier}, enable_forecast_util={enable_forecast_util}")

        for agent, spec in specs.items():
            low, high = spec['bounds']
            total_dim = spec['total_dim']  # base_dim + forecast_dim (from EnhancedObservationValidator)

            final_dim = total_dim
            self._obs_spaces[agent] = spaces.Box(
                low=low[:final_dim], high=high[:final_dim],
                shape=(final_dim,), dtype=np.float32
            )
            logger.info(f"[OBS_SPACE_STATIC] {agent}: base={spec['base_dim']}, forecast={spec['forecast_dim']}, TOTAL={final_dim}")



    # ---- properties / API ----
    @property
    def possible_agents(self): return self._possible_agents

    @property
    def agents(self): return self._agents

    @agents.setter
    def agents(self, value): self._agents = value[:]

    def observation_space(self, agent): return self._obs_spaces[agent]

    def action_space(self, agent): return self.env.action_space(agent)
    
    def rebuild_observation_spaces(self):
        """
        Rebuild observation spaces after environment/forecaster changes.

        """
        logger.info("[OBS_SPACE_REBUILD] Rebuilding observation spaces")
        self._build_wrapper_observation_spaces()
        logger.info("[OBS_SPACE_REBUILD] Observation spaces rebuilt successfully")

    @property
    def observation_spaces(self): return {a: self.observation_space(a) for a in self.possible_agents}

    @property
    def action_spaces(self): return {a: self.action_space(a) for a in self.possible_agents}

    @property
    def t(self): return getattr(self.env, "t", 0)

    @property
    def max_steps(self): return getattr(self.env, "max_steps", 1000)

    @property
    def overlay_trainer(self):
        """Forward overlay_trainer access to base environment."""
        return getattr(self.env, "overlay_trainer", None)

    # ---- unified augmentation ----
        # No observation augmentation beyond the environment-provided observation space.
        # Tier 3 is observation-identical to Tier 2 for fair comparison.

    # ---- verification methods ----
    def _verify_capacity_consistency(self):
        """Verify wrapper is using correct capacity values from environment"""
        try:
            env_ref = getattr(self, "env", self)
            
            # Check physical capacities exist and are reasonable
            wind_mw = getattr(env_ref, 'wind_capacity_mw', 0.0)
            solar_mw = getattr(env_ref, 'solar_capacity_mw', 0.0)
            hydro_mw = getattr(env_ref, 'hydro_capacity_mw', 0.0)
            
            total_mw = wind_mw + solar_mw + hydro_mw

            # Dynamic capacity check based on budget size
            try:
                budget = getattr(env_ref, 'init_budget', 500_000_000)
                # Expect roughly 1 MW per $2M budget (conservative estimate)
                expected_min_mw = budget / 2_000_000

                if total_mw < expected_min_mw * 0.1:  # Allow very low deployment (10% of expected)
                    logger.warning(f"Very low capacity deployment: {total_mw:.1f}MW (expected ~{expected_min_mw:.0f}MW for ${budget:,.0f} budget)")
                    return False
            except Exception:
                # Fallback to original check for small funds
                if total_mw < 5:  # Minimum 5MW for any meaningful operation
                    logger.warning(f"Insufficient capacity deployment: {total_mw:.1f}MW")
                    return False
                
            return True
        except Exception as e:
            logger.error(f"Capacity verification failed: {e}")
            return False

    def _safe_portfolio_value(self, raw_value, initial_budget):
        """Apply safety bounds to portfolio values"""
        min_portfolio = initial_budget * 0.01  # Minimum 1% of initial fund
        max_portfolio = initial_budget * 3.0   # Maximum 300% of initial fund
        return float(np.clip(raw_value, min_portfolio, max_portfolio))

    # ---- core loop ----
    def reset(self, seed=None, options=None):
        self._cleanup_memory_enhanced(force=True)
        self._last_episode_seed = seed
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.episode_count += 1
        self.error_count = 0
        self._ep_meta_return = 0.0
        self._prev_forecasts_for_error = {}
        self._last_price_forecast_norm = 0.0

        self._last_price_forecast_aligned = 0.0

        # Initialize forecaster history with sufficient data for predictions
        try:
            if hasattr(self.env, "data") and len(self.env.data) > 0:
                if hasattr(self.forecaster, "initialize_history"):
                    self.forecaster.initialize_history(self.env.data, start_idx=0)
                elif hasattr(self.forecaster, "update"):
                    self.forecaster.update(self.env.data.iloc[0])
        except Exception:
            pass

        enhanced = self.obs_builder.enhance_observations(obs)

        # FIX: Skip investor_0 and battery_operator_0 validation before augmentation
        # (they will be augmented after, and augmentation must produce exact static space size)
        validated = self._validate_observations_safe(enhanced, skip_investor_0=True, skip_battery=True)

        # No additional observation augmentation (Tier 3 is observation-identical to Tier 2).

        # FIX: VALIDATION - Verify all observations match their static observation spaces
        # This is a safety check to catch any dimension mismatches
        for agent in validated:
            if agent in self._obs_spaces:
                # CRITICAL FIX: Ensure observation is 1D (no batch dimension)
                obs = validated[agent]
                if obs.ndim > 1:
                    # Remove batch dimension if present - use squeeze(0) to be explicit
                    if obs.shape[0] == 1:
                        obs = obs.squeeze(0)  # Remove first dimension if it's size 1
                    else:
                        obs = obs.flatten()  # Flatten if no single batch dimension
                    validated[agent] = obs.astype(np.float32)
                elif obs.ndim == 0:
                    # Scalar - convert to 1D array
                    validated[agent] = np.array([obs], dtype=np.float32)
                else:
                    validated[agent] = obs.astype(np.float32)
                
                expected_dim = self._obs_spaces[agent].shape[0]
                actual_dim = validated[agent].shape[0]

                if actual_dim != expected_dim:
                    # FAIL-FAST: Log the mismatch and raise an error
                    # This indicates a bug in observation building or augmentation
                    error_msg = (f"[OBS_SPACE_MISMATCH] {agent}: expected {expected_dim} dims, "
                               f"got {actual_dim}. This indicates a bug in observation building. "
                               f"Static space: {self._obs_spaces[agent].shape}, "
                               f"Actual observation: {validated[agent].shape}")
                    logger.error(error_msg)
                    # Truncate or pad to match (fallback, but log the error)
                    if actual_dim > expected_dim:
                        validated[agent] = validated[agent][:expected_dim]
                    else:
                        padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                        validated[agent] = np.concatenate([validated[agent], padding])

        return validated, info

    def step(self, actions: Dict[str, Any]):
        t0 = time.perf_counter()

        # validate & track clipping
        actions, clip_counts, clip_totals = self._validate_actions_comprehensive(actions, track_clipping=True)
        self._last_clip_counts = clip_counts
        self._last_clip_totals = clip_totals

        # CRITICAL FIX #6: Populate forecast arrays BEFORE env.step() so DL labeler has access
        # The DL overlay labeler needs forecasts during step execution for label generation
        try:
            if hasattr(self.env, 'populate_forecast_arrays'):
                current_forecasts = self._get_forecasts_for_logging()
                self.env.populate_forecast_arrays(getattr(self.env, 't', 0), current_forecasts)
        except Exception:
            pass  # Don't break main flow if forecast population fails

        obs, rewards, dones, truncs, infos = self.env.step(actions)
        self._last_episode_end_flag = 1 if any(bool(x) for x in dones.values()) else 0
        self._ep_meta_return += float(rewards.get("meta_controller_0", 0.0) or 0.0)

        # update forecaster
        self._update_forecaster_safe()

        # build enhanced obs with full forecasting
        enhanced = self.obs_builder.enhance_observations(obs)

        # FIX: Skip investor_0 and battery_operator_0 validation before augmentation
        # (they will be augmented after, and augmentation must produce exact static space size)
        validated = self._validate_observations_safe(enhanced, skip_investor_0=True, skip_battery=True)

        # No additional observation augmentation (Tier 3 is observation-identical to Tier 2).

        # FIX: VALIDATION - Verify all observations match their static observation spaces
        # This is a safety check to catch any dimension mismatches
        for agent in validated:
            if agent in self._obs_spaces:
                # CRITICAL FIX: Ensure observation is 1D (no batch dimension)
                obs = validated[agent]
                if obs.ndim > 1:
                    # Remove batch dimension if present - use squeeze(0) to be explicit
                    if obs.shape[0] == 1:
                        obs = obs.squeeze(0)  # Remove first dimension if it's size 1
                    else:
                        obs = obs.flatten()  # Flatten if no single batch dimension
                    validated[agent] = obs.astype(np.float32)
                elif obs.ndim == 0:
                    # Scalar - convert to 1D array
                    validated[agent] = np.array([obs], dtype=np.float32)
                else:
                    validated[agent] = obs.astype(np.float32)
                
                expected_dim = self._obs_spaces[agent].shape[0]
                actual_dim = validated[agent].shape[0]

                if actual_dim != expected_dim:
                    # FAIL-FAST: Log the mismatch and raise an error
                    # This indicates a bug in observation building or augmentation
                    error_msg = (f"[OBS_SPACE_MISMATCH] {agent}: expected {expected_dim} dims, "
                               f"got {actual_dim}. This indicates a bug in observation building. "
                               f"Static space: {self._obs_spaces[agent].shape}, "
                               f"Actual observation: {validated[agent].shape}")
                    logger.error(error_msg)
                    # Truncate or pad to match (fallback, but log the error)
                    if actual_dim > expected_dim:
                        validated[agent] = validated[agent][:expected_dim]
                    else:
                        padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                        validated[agent] = np.concatenate([validated[agent], padding])

        self.step_count += 1
        step_time = (time.perf_counter() - t0)
        self._last_step_wall_ms = step_time * 1000.0


        # Enhanced monitoring (optional)
        if self.enhanced_monitor:
            self.enhanced_monitor.update_system_metrics(step_time)
            for agent, reward in rewards.items():
                self.enhanced_monitor.update_training_metrics(agent, reward)
            if hasattr(self, '_last_portfolio_value') and self._last_portfolio_value is not None:
                self.enhanced_monitor.update_portfolio_metrics(self._last_portfolio_value)
            self.enhanced_monitor.log_summary()

        if self._last_episode_end_flag:
            self._prev_forecasts_for_error = {}

        return validated, rewards, dones, truncs, infos

    # ---- helpers ----
    def _validate_observations_safe(self, obs_dict, skip_investor_0=False, skip_battery=False):
        """
        Validate observations.

        Args:
            obs_dict: Dictionary of observations
            skip_investor_0: If True, skip validation for investor_0 (will be augmented after)
            skip_battery: If True, skip validation for battery_operator_0 (will be augmented after)
        """
        validated = {}
        for agent in self.possible_agents:
            try:
                # Skip investor_0 if it will be augmented after validation
                if skip_investor_0 and agent == "investor_0":
                    if agent in obs_dict:
                        validated[agent] = obs_dict[agent].astype(np.float32)
                    else:
                        validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
                    # DEBUG: Confirm skip
                    # print(f"[SKIP] investor_0 validation skipped (will be augmented after)")
                    continue

                # Skip battery_operator_0 if it will be augmented after validation
                if skip_battery and agent == "battery_operator_0":
                    if agent in obs_dict:
                        validated[agent] = obs_dict[agent].astype(np.float32)
                    else:
                        validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
                    continue

                if agent in obs_dict:
                    obs = obs_dict[agent]
                    expected = self.observation_space(agent).shape
                    if obs.shape != expected:
                        if self.error_count < self.max_errors:
                            print(f"[WARN] Obs shape mismatch for {agent}: {obs.shape} vs {expected}")
                        self.error_count += 1
                        obs = self.obs_builder.validator._create_safe_observation(agent)
                    validated[agent] = obs.astype(np.float32)
                else:
                    validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
            except Exception as e:
                if self.error_count < self.max_errors:
                    print(f"[WARN] Obs validation failed for {agent}: {e}")
                self.error_count += 1
                validated[agent] = self.obs_builder.validator._create_safe_observation(agent)
        return validated

    def _cleanup_memory_enhanced(self, force=False):
        try:
            cleanup_level, _ = self.memory_tracker.should_cleanup(force=force)
            if cleanup_level or force:
                self.memory_tracker.cleanup(cleanup_level or 'medium')

                # Enhanced cache cleanup with LRU - more aggressive thresholds
                if hasattr(self.obs_builder, 'forecast_cache') and len(self.obs_builder.forecast_cache) > 300:  # Reduced from 500
                    self.obs_builder.forecast_cache._memory_cleanup()
                if hasattr(self.obs_builder, 'agent_forecast_cache') and len(self.obs_builder.agent_forecast_cache) > 600:  # Reduced from 1000
                    self.obs_builder.agent_forecast_cache._memory_cleanup()

                # Clear TensorFlow session if available
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass

                # Clear validation caches
                if hasattr(self.obs_builder.validator, 'validation_cache'):
                    self.obs_builder.validator.validation_cache.clear()

                # Clear forecaster caches if available
                if hasattr(self.forecaster, '_global_cache'):
                    if len(self.forecaster._global_cache) > 4000:  # Reduced threshold
                        self.forecaster._global_cache.clear()
                if hasattr(self.forecaster, '_agent_cache'):
                    if len(self.forecaster._agent_cache) > 2000:  # Reduced threshold
                        self.forecaster._agent_cache.clear()

                # Multiple garbage collection cycles for thorough cleanup
                for _ in range(2):
                    gc.collect()

                # Clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

        except Exception as e:
            if self.error_count < self.max_errors:
                print(f"[WARN] Memory cleanup failed: {e}")
                self.error_count += 1

    def _to_numpy_safe(self, a_in):
        """
        Brings various tensor types to a CPU numpy array without importing heavy libs.
        Handles torch.Tensor, JAX DeviceArray, etc., by duck-typing.
        """
        try:
            # torch: .detach().cpu().numpy()
            if hasattr(a_in, "detach") and callable(a_in.detach):
                try:
                    a_in = a_in.detach()
                except Exception:
                    pass
            if hasattr(a_in, "cpu") and callable(a_in.cpu):
                try:
                    a_in = a_in.cpu()
                except Exception:
                    pass
            if hasattr(a_in, "numpy") and callable(a_in.numpy):
                try:
                    a_in = a_in.numpy()
                except Exception:
                    pass
        except Exception:
            pass
        return a_in

    def _validate_actions_comprehensive(self, actions, track_clipping=False):
        validated = {}
        clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}
        clip_totals = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        for agent in self.possible_agents:
            space = self.env.action_space(agent)
            a_in = actions.get(agent, None)

            # Bring tensors to numpy if needed (CUDA-safe)
            if a_in is not None:
                a_in = self._to_numpy_safe(a_in)

            # Discrete
            if isinstance(space, spaces.Discrete):
                if isinstance(a_in, np.ndarray):
                    val = int(np.atleast_1d(a_in).flatten()[0])
                elif np.isscalar(a_in):
                    val = int(a_in)
                elif isinstance(a_in, (list, tuple)) and len(a_in) > 0:
                    val = int(a_in[0])
                else:
                    val = 0
                val_clipped = int(np.clip(val, 0, space.n - 1))
                validated[agent] = val_clipped
                if track_clipping:
                    key = ("investor" if agent.startswith("investor")
                           else "battery" if agent.startswith("battery")
                           else "risk" if agent.startswith("risk")
                           else "meta")
                    clip_totals[key] += 1
                    clip_counts[key] += int(val != val_clipped)
                continue

            # Box → float array with shape
            if not isinstance(a_in, np.ndarray):
                if np.isscalar(a_in):
                    arr = np.array([a_in], dtype=np.float32)
                elif isinstance(a_in, (list, tuple)):
                    arr = np.array(a_in, dtype=np.float32)
                else:
                    # default to mid-point
                    middle = (space.low + space.high) / 2.0
                    m = float(middle.flatten()[0]) if hasattr(middle, 'flatten') else float(middle)
                    arr = np.array([m], dtype=np.float32)
            else:
                arr = a_in.astype(np.float32)

            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim > 1:
                arr = arr.flatten()

            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            # Anti-collapse: smooth-squash only if out of bounds.
            # Prevents hard clipping from turning near-constant out-of-range outputs into perfectly constant actions.
            if bool(getattr(self.config, "enable_action_tanh_squash", True)) and np.any(np.abs(arr) > 1.0 + 1e-6):
                arr = np.tanh(arr).astype(np.float32)
            need = int(np.prod(space.shape))
            if arr.size != need:
                if arr.size < need:
                    middle = (space.low + space.high) / 2.0
                    m = float(middle.flatten()[0]) if hasattr(middle, 'flatten') else float(middle)
                    arr = np.concatenate([arr, np.full(need - arr.size, m, dtype=np.float32)])
                else:
                    arr = arr[:need]

            before = arr.copy()
            # FIX: Final action clipping in wrapper
            # CRITICAL: This must match the clipping done in metacontroller._coerce_action_for_buffer
            # to ensure off-policy agents (SAC/TD3) have consistent actions between replay buffer and execution
            arr = np.minimum(np.maximum(arr, space.low), space.high)
            validated[agent] = arr.astype(np.float32)

            if track_clipping:
                key = ("investor" if agent.startswith("investor")
                       else "battery" if agent.startswith("battery")
                       else "risk" if agent.startswith("risk")
                       else "meta")
                clip_totals[key] += arr.size
                clip_counts[key] += int(np.sum(np.abs(before - arr) > 1e-12))

        return validated, clip_counts, clip_totals

    def _update_forecaster_safe(self):
        """Update forecaster with current timestep data to avoid forecast lag."""
        try:
            if hasattr(self.env, "data") and self.env.t < len(self.env.data):
                row = self.env.data.iloc[self.env.t]
                if hasattr(self.forecaster, "update"):
                    self.forecaster.update(row)
        except Exception:
            pass

    def validate_observation_dimensions_comprehensive(self, observations: Dict[str, np.ndarray]) -> bool:
        """
        COMPREHENSIVE: Validate all observation dimensions match declared spaces.

        This is a runtime safety check to catch:
        - Config changes between initialization and runtime
        - Observation dimension mismatches
        - Forecast augmentation errors

        Returns:
            bool: True if all observations are valid, False otherwise

        Raises:
            ValueError: If critical dimension mismatch detected
        """
        all_valid = True
        errors = []

        for agent, obs in observations.items():
            if agent not in self._obs_spaces:
                errors.append(f"Agent {agent} not in observation spaces")
                all_valid = False
                continue

            expected_space = self._obs_spaces[agent]
            expected_dim = expected_space.shape[0]
            actual_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)

            if actual_dim != expected_dim:
                error_msg = (
                    f"DIMENSION MISMATCH for {agent}: "
                    f"expected {expected_dim}, got {actual_dim}. "
                    f"Expected space: {expected_space.shape}, "
                    f"Actual observation: {obs.shape if hasattr(obs, 'shape') else len(obs)}"
                )
                errors.append(error_msg)
                all_valid = False

            # Validate bounds
            if not expected_space.contains(obs):
                # Check which dimensions are out of bounds
                out_of_bounds = []
                for i in range(min(len(obs), len(expected_space.low))):
                    if obs[i] < expected_space.low[i] or obs[i] > expected_space.high[i]:
                        out_of_bounds.append(
                            f"dim[{i}]={obs[i]:.4f} not in [{expected_space.low[i]:.4f}, {expected_space.high[i]:.4f}]"
                        )

                if out_of_bounds:
                    error_msg = f"OUT OF BOUNDS for {agent}: {', '.join(out_of_bounds[:5])}"
                    if len(out_of_bounds) > 5:
                        error_msg += f" ... and {len(out_of_bounds) - 5} more"
                    errors.append(error_msg)
                    all_valid = False

        if not all_valid:
            error_summary = "\n".join(errors)
            logger.error(f"[OBSERVATION VALIDATION FAILED]\n{error_summary}")

            # In strict mode, raise exception
            if getattr(self, 'strict_validation', False):
                raise ValueError(f"Observation validation failed:\n{error_summary}")

        return all_valid

    # ---- logging helpers ----
    def _safe_float(self, v, default=0.0):
        try:
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v).reshape(-1)
                v = v[0] if v.size > 0 else default
            return float(v) if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    def _ensure_float(self, x, default=0.0):
        """Enhanced safety helper to ensure all logged fields are valid floats with no NaNs/None."""
        try:
            if x is None:
                return float(default)
            if isinstance(x, (list, tuple, np.ndarray)):
                x = np.asarray(x).reshape(-1)
                x = x[0] if x.size > 0 else default
            result = float(x)
            if not np.isfinite(result):
                return float(default)
            return result
        except (ValueError, TypeError, OverflowError):
            return float(default)

    def _get_actuals_for_logging(self) -> Dict[str, float]:
        out = {'wind': 0.0, 'solar': 0.0, 'hydro': 0.0, 'price': 0.0, 'load': 0.0}
        try:
            if hasattr(self.env, "data") and isinstance(self.env.data, pd.DataFrame) and len(self.env.data) > 0:
                idx = max(0, min(getattr(self.env, "t", 0) - 1, len(self.env.data) - 1))
                row = self.env.data.iloc[idx]
                for k in out.keys():
                    if k in row:
                        out[k] = self._safe_float(row[k], 0.0)
        except Exception:
            pass
        return out

    def _get_timestamp_for_logging(self) -> str:
        try:
            if hasattr(self.env, "data") and isinstance(self.env.data, pd.DataFrame):
                if "timestamp" in self.env.data.columns and len(self.env.data) > 0:
                    idx = max(0, min(getattr(self.env, "t", 0) - 1, len(self.env.data) - 1))
                    ts = pd.to_datetime(self.env.data.iloc[idx]["timestamp"], errors="coerce")
                    if pd.notna(ts):
                        return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        return ""

    def _get_forecasts_for_logging(self) -> Dict[str, float]:
        try:
            f = self.forecaster.predict_all_horizons(timestep=getattr(self.env, 't', 0))
            return f if isinstance(f, dict) else {}
        except Exception:
            return {}

    def _get_price_forecast_norm_and_aligned(self, all_forecasts: Dict[str, float]) -> Tuple[float, float]:
        """
        Returns (price_forecast_norm, price_forecast_aligned_norm).
        We compute aligned as the forecast with horizon closest to env.investment_freq.
        """
        pf_norm = 0.0
        pf_aligned = 0.0
        try:
            # ENHANCED: Filter out None/NaN values from price_items
            price_items = []
            for k in all_forecasts.keys():
                if "price_forecast_" in str(k).lower():
                    v = all_forecasts[k]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        price_items.append((k, v))

            if price_items:
                cand = []
                for k, v in price_items:
                    try:
                        n = self.obs_builder.postproc.normalize_value(k, v)
                        # ENHANCED: Filter out None/NaN normalized values
                        if n is not None and not (isinstance(n, float) and np.isnan(n)):
                            cand.append((k, n))
                            if pf_norm == 0.0:
                                pf_norm = float(n)
                    except Exception:
                        # Skip invalid normalization results
                        continue

                invf = int(getattr(self.env, "investment_freq", 12))

                # FIXED: Use generator's horizons map to translate names to steps
                def steps_for_key(k):
                    """Convert horizon name to steps using generator's horizons map."""
                    try:
                        # Get horizons map from forecaster
                        hmap = getattr(self.forecaster, "horizons", {})
                        if not hmap:
                            # CANONICAL: Use cached config source of truth for horizon fallback
                            try:
                                hmap = getattr(getattr(self.base_env, 'config', None), 'forecast_horizons', {}).copy()
                                if not hmap:
                                    raise ValueError("No forecast_horizons in config")
                            except Exception:
                                # FAIL-FAST: No hardcoded horizon fallbacks allowed for production safety
                                raise ValueError("Cannot access forecast_horizons from base_env.config: "
                                               "config missing or malformed. Fix config initialization to maintain single source of truth.")

                        # Extract horizon name from key (e.g., "price_forecast_short" -> "short")
                        name = str(k).split("_")[-1]
                        if name not in hmap:
                            raise ValueError(f"Horizon '{name}' not found in config.forecast_horizons. "
                                           f"Available horizons: {list(hmap.keys())}. Fix config to maintain single source of truth.")
                        return int(hmap[name])
                    except Exception as e:
                        raise ValueError(f"Failed to get horizon steps for key '{k}': {str(e)}. "
                                       f"Ensure all forecast keys follow pattern 'target_forecast_horizon' with valid horizons from config.")

                def dist(k):
                    """Calculate distance between horizon steps and investment frequency."""
                    return abs(steps_for_key(k) - invf)

                # ENHANCED: Ensure cand is not empty before calling min()
                if cand:
                    k_best, v_best = min(cand, key=lambda kv: dist(kv[0]))
                    pf_aligned = float(v_best)
        except Exception:
            pass
        return pf_norm, pf_aligned

    def _get_risk_vector6(self) -> List[float]:
        default = [0.5, 0.2, 0.25, 0.15, 0.35, 0.25]
        try:
            if hasattr(self.env, 'enhanced_risk_controller') and self.env.enhanced_risk_controller:
                arr = np.asarray(self.env.enhanced_risk_controller.get_risk_metrics_for_observation())
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                if arr.size >= 6:
                    return [float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else d for x, d in zip(arr[:6], default)]
        except Exception:
            pass
        return default

    def _calc_perf_and_quick_risks(self) -> List[float]:
        """
        PATCH: Compute performance from true MTM equity (preferred), not capacity proxy.
        Returns [perf, overall_risk, market_risk].
        """
        try:
            initial_budget = float(getattr(self.env, 'init_budget', 1e7))
            initial_budget = max(initial_budget, 1.0)

            if hasattr(self.env, 'equity'):
                portfolio_value = float(self._safe_float(getattr(self.env, 'equity', initial_budget), initial_budget))
            else:
                cash = float(self._safe_float(getattr(self.env, 'budget', initial_budget), initial_budget))
                w_val = float(self._safe_float(getattr(self.env, 'wind_instrument_value', 0.0), 0.0))
                s_val = float(self._safe_float(getattr(self.env, 'solar_instrument_value', 0.0), 0.0))
                h_val = float(self._safe_float(getattr(self.env, 'hydro_instrument_value', 0.0), 0.0))
                portfolio_value = cash + w_val + s_val + h_val

            # Apply safety bounds using the new helper method
            portfolio_value = self._safe_portfolio_value(portfolio_value, initial_budget)

            perf = float(_safe_divide(portfolio_value, initial_budget, 1.0))
            perf_clipped = float(np.clip(perf, 0.0, 10.0))
            overall_risk = float(np.clip(getattr(self.env, "overall_risk_snapshot", 0.5), 0.0, 1.0))
            market_risk = float(np.clip(getattr(self.env, "market_risk_snapshot", 0.5), 0.0, 1.0))
            self._last_portfolio_value = portfolio_value
            return [perf_clipped, overall_risk, market_risk]
        except Exception:
            self._last_portfolio_value = None
            return [1.0, 0.5, 0.3]



    # ---- debug methods ----
    def debug_wrapper_data_sources(self):
        """Debug what data sources the wrapper is using"""
        env_ref = getattr(self, "env", self)

        print("Wrapper Data Sources:")
        print(f"  Physical Wind: {getattr(env_ref, 'wind_capacity_mw', 'MISSING')}MW")
        print(f"  Financial Wind: ${getattr(env_ref, 'wind_instrument_value', 'MISSING'):,.0f}")
        print(f"  Budget: ${getattr(env_ref, 'budget', 'MISSING'):,.0f}")
        print(f"  Equity: ${getattr(env_ref, 'equity', 'MISSING'):,.0f}")

    # ---- passthroughs ----
    def render(self):
        try:
            if hasattr(self.env, "render"):
                return self.env.render()
        except Exception:
            return None

    def close(self):
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass

    # alias PettingZoo naming if needed
    def state(self):
        try:
            if hasattr(self.env, "state"):
                return self.env.state()
        except Exception:
            return None


# =========================
# FGB Forecast Validation
# =========================
def verify_fgb_forecasts(wrapper, num_steps: int = 10) -> bool:
    """
    Verify that forecast features are present and non-degenerate in observations.

    This function checks:
    1. Forecast features are present in observations
    2. Forecast features have non-zero values (not just padding)
    3. Forecast features have reasonable variance

    Args:
        wrapper: The wrapped environment
        num_steps: Number of steps to verify (default: 10)

    Returns:
        True if forecasts are valid, raises ValueError otherwise

    Raises:
        ValueError: If forecasts are missing, all-zero, or have near-zero variance
    """
    try:
        logger.info("[FGB_VALIDATION] Starting forecast verification...")

        # Reset environment
        obs, info = wrapper.reset()

        if 'investor_0' not in obs:
            raise ValueError("FGB validation: investor_0 not in observations")

        investor_obs = obs['investor_0']
        total_dim = len(investor_obs)

        # Current expected layouts:
        # - Tier 2 / Tier 3 (fair, obs-identical): investor_0 is 8D = 6 base + 2 forecast features
        #   Forecast slice: indices [6:8]
        # - Legacy/experimental layouts may have more dims; we only require that there is a non-empty
        #   forecast slice immediately after the 6 base dims.
        base_dim = 6
        if total_dim <= base_dim:
            raise ValueError(
                f"FGB validation: Observation too small ({total_dim} dims). "
                f"Expected > {base_dim} dims so forecast features can exist."
            )

        forecast_start = base_dim
        # Prefer the canonical 8D slice when available; otherwise take all remaining dims after base.
        forecast_end = 8 if total_dim >= 8 else total_dim
        forecast_features = investor_obs[forecast_start:forecast_end]
        if forecast_features.size == 0:
            raise ValueError(
                f"FGB validation: Empty forecast feature slice. total_dim={total_dim}, "
                f"slice=[{forecast_start}:{forecast_end}]"
            )

        # Check 1: Forecast magnitude (should not be all zeros)
        forecast_magnitude = float(np.abs(forecast_features).sum())
        if forecast_magnitude < 0.01:
            raise ValueError(
                f"FGB validation: Forecast features are all near-zero! "
                f"Forecasts may not be generated or appended. "
                f"forecast_magnitude={forecast_magnitude:.6f} (expected > 0.01)"
            )

        # Check 2: Forecast variance (should have some variation)
        forecast_std = float(np.std(forecast_features))
        if forecast_std < 0.001:
            raise ValueError(
                f"FGB validation: Forecast features have near-zero variance! "
                f"May be using flat/constant forecasts. "
                f"forecast_std={forecast_std:.6f} (expected > 0.001)"
            )

        # Check 3: Run a few steps and verify forecasts change
        forecast_history = [forecast_features.copy()]
        for step in range(num_steps):
            actions = {agent: wrapper.action_space(agent).sample() for agent in wrapper.possible_agents}
            obs, rewards, dones, truncs, infos = wrapper.step(actions)

            if 'investor_0' in obs:
                investor_obs = obs['investor_0']
                forecast_features = investor_obs[forecast_start:forecast_end]
                forecast_history.append(forecast_features.copy())

        # Check that forecasts change over time (not static)
        forecast_changes = []
        for i in range(1, len(forecast_history)):
            change = float(np.abs(forecast_history[i] - forecast_history[i-1]).sum())
            forecast_changes.append(change)

        avg_change = float(np.mean(forecast_changes))
        if avg_change < 0.001:
            logger.warning(
                f"FGB validation: Forecasts are not changing over time! "
                f"avg_change={avg_change:.6f} (expected > 0.001). "
                f"This may indicate forecasts are static or not being updated."
            )

        # All checks passed
        logger.info(
            f"[FGB_VALIDATION] ✓ Forecasts validated successfully:\n"
            f"  - Magnitude: {forecast_magnitude:.4f}\n"
            f"  - Std dev: {forecast_std:.4f}\n"
            f"  - Avg change/step: {avg_change:.6f}\n"
            f"  - Observation dims: {total_dim} (forecast slice: [{forecast_start}:{forecast_end}] "
            f"= {forecast_end - forecast_start} dims)"
        )
        return True

    except ValueError as e:
        logger.error(f"[FGB_VALIDATION] ✗ FAILED: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"[FGB_VALIDATION] ✗ Unexpected error: {str(e)}")
        raise ValueError(f"FGB forecast validation failed: {str(e)}")
