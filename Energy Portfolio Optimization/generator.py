import os
import numpy as np
import pandas as pd
import logging
from utils import _get_tf

logger = logging.getLogger(__name__)

# =========================
# TensorFlow setup (optional)
# =========================


def _is_capacity_factor_data(df: pd.DataFrame) -> bool:
    """Detect whether renewable generation columns look like capacity factors."""
    renewable_cols = ["wind", "solar", "hydro"]
    for col in renewable_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            max_val = float(np.nanmax(s.values)) if len(s) else 0.0
            if max_val > 2.0:
                return False
    return True


def _convert_to_raw_mw_values(df: pd.DataFrame, config=None, mw_scale_overrides=None) -> pd.DataFrame:
    """Convert capacity-factor data to raw MW values for direct forecasting."""
    if config and hasattr(config, "mw_conversion_scales"):
        capacity_mw = config.mw_conversion_scales.copy()
    else:
        capacity_mw = {
            "wind": 1103,
            "solar": 100,
            "hydro": 534,
            "load": 2999,
        }

    if mw_scale_overrides:
        for key, value in mw_scale_overrides.items():
            if value is not None and key in capacity_mw:
                capacity_mw[key] = float(value)
                logger.info("[OVERRIDE] Using CLI override for %s: %s MW", key, value)

    df_converted = df.copy()
    logger.info("[INFO] Converting to raw MW values (no normalization):")

    def _looks_like_capacity_factor(series: pd.Series) -> bool:
        s = pd.to_numeric(series, errors="coerce")
        if len(s) == 0:
            return False
        max_val = float(np.nanmax(s.values))
        min_val = float(np.nanmin(s.values))
        return (max_val <= 2.0) and (min_val >= -0.1)

    for col, capacity in capacity_mw.items():
        if col in df_converted.columns:
            original_range = f"[{df[col].min():.3f}, {df[col].max():.3f}]"
            if _looks_like_capacity_factor(df[col]):
                df_converted[col] = df[col] * capacity
                new_range = f"[{df_converted[col].min():.1f}, {df_converted[col].max():.1f}] MW"
                logger.info("  %s: %s -> %s", col, original_range, new_range)
                logger.info("  %s: %s (kept as-is; already looks like MW)", col, original_range)

    if "price" in df_converted.columns:
        price_range = f"[{df_converted['price'].min():.1f}, {df_converted['price'].max():.1f}] $/MWh"
        logger.info("  price: %s (no conversion)", price_range)

    logger.info("[OK] Raw MW conversion complete - ready for direct forecasting")
    return df_converted


def load_energy_data(csv_path: str, convert_to_raw_units: bool = True, config=None, mw_scale_overrides=None) -> pd.DataFrame:
    """
    Load energy time series data from CSV with optional MW conversion.

    Requires at least: wind, solar, hydro, price, load.
    Keeps extra columns if present and parses timestamp when possible.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )

    required = ["wind", "solar", "hydro", "price", "load"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    numeric_extra = [c for c in ["risk", "revenue", "battery_energy", "npv"] if c in df.columns]
    for col in required + numeric_extra:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    if convert_to_raw_units and _is_capacity_factor_data(df):
        logger.info("[INFO] Converting capacity factors to raw MW values for direct forecasting...")
        df = _convert_to_raw_mw_values(df, config=config, mw_scale_overrides=mw_scale_overrides)
        logger.info("[OK] Raw MW conversion completed - forecasts will work directly with these units")
    else:
        logger.info("[INFO] Data already in raw MW units - ready for direct forecasting")

    return df

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

