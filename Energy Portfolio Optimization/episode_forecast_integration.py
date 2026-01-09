"""
Episode-Specific Forecast Integration

Helper functions for training episode-specific forecast models and generating cache.
Integrated into main.py's episode training loop.

CRITICAL BEHAVIOR:
- Each episode trains forecast models FROM SCRATCH on its OWN 6-month period ONLY (not cumulative!)
  * Episode 0: Only 2015 H1 (Jan 1 - Jun 30, 2015)
  * Episode 1: Only 2015 H2 (Jul 1 - Dec 31, 2015)
  * Episode 19: Only 2024 H2 (Jul 1 - Dec 31, 2024)
- Forecast models are NOT loaded from previous episodes (unlike RL weights which are continuous)
- Each episode has its own independent set of forecast models in forecast_models/episode_N/
- The precomputed forecast cache is also episode-specific in forecast_cache/episode_N/
- For evaluation on unseen 2025 data, use Episode 19 models (latest training period: 2024 H2)
"""

import os
import sys
import json
import logging

# Get logger - try to use main logger if available, otherwise create new
try:
    from logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def check_episode_models_exist(episode_num: int, forecast_base_dir: str = "forecast_models") -> bool:
    """
    Check if episode-specific forecast models exist.
    
    Args:
        episode_num: Episode number (0-19)
        forecast_base_dir: Base directory for forecast models
    
    Returns:
        True if models directory exists and contains model files, False otherwise
    """
    episode_dir = os.path.join(forecast_base_dir, f"episode_{episode_num}")
    models_dir = os.path.join(episode_dir, "models")
    scalers_dir = os.path.join(episode_dir, "scalers")
    metadata_dir = os.path.join(episode_dir, "metadata")
    
    # Check if all required directories exist
    if not all(os.path.exists(d) for d in [models_dir, scalers_dir, metadata_dir]):
        return False
    
    # Check if at least some model files exist (expecting ~25 models per episode)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.keras')]
    scaler_files = [f for f in os.listdir(scalers_dir) if f.endswith('.pkl')]
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    
    # Require at least 5 models (one per target for short horizon minimum)
    has_sufficient_models = len(model_files) >= 5 and len(scaler_files) >= 10 and len(metadata_files) >= 5
    
    return has_sufficient_models


def check_episode_cache_exists(episode_num: int, cache_base_dir: str = "forecast_cache") -> bool:
    """
    Check if episode-specific forecast cache exists.
    
    Args:
        episode_num: Episode number (0-19)
        cache_base_dir: Base directory for forecast cache
    
    Returns:
        True if cache directory exists and contains cache files, False otherwise
    """
    episode_cache_dir = os.path.join(cache_base_dir, f"episode_{episode_num}")
    
    if not os.path.exists(episode_cache_dir):
        return False
    
    # Check for cache files (generator.py saves cache in this directory)
    cache_files = [f for f in os.listdir(episode_cache_dir) if f.endswith('.npy') or f.endswith('.pkl')]
    
    return len(cache_files) > 0


def train_episode_forecast_models(
    episode_num: int,
    episode_data_path: str,
    forecast_base_dir: str = "forecast_models"
) -> bool:
    """
    Train forecast models for a specific episode (directly, no subprocess).
    
    Args:
        episode_num: Episode number (0-19)
        episode_data_path: Path to episode-specific scenario file (e.g., scenario_000.csv)
        forecast_base_dir: Base directory for forecast models
    
    Returns:
        True if training succeeded, False otherwise
    
    Raises:
        RuntimeError: If training fails (fail-fast, no fallback)
    """
    logger.info(f"   [FORECAST] Training models for Episode {episode_num}...")
    logger.info(f"   [DATA] Using episode file: {episode_data_path}")
    
    try:
        # Import training function directly (no subprocess)
        from train_episode_forecasts import train_episode_forecasts
        
        # Call training function directly with episode-specific scenario file
        result = train_episode_forecasts(
            episode_num=episode_num,
            episode_data_path=episode_data_path,
            output_base_dir=forecast_base_dir
        )
        
        if result.get('successful', 0) > 0:
            logger.info(f"   [OK] Episode {episode_num} forecast models trained successfully ({result['successful']} models)")
            return True
        else:
            error_msg = f"Forecast training failed for Episode {episode_num}: {result.get('failed', 0)} models failed"
            logger.error(f"   [ERROR] {error_msg}")
            raise RuntimeError(error_msg)
            
    except ImportError as e:
        error_msg = f"Failed to import train_episode_forecasts module: {e}"
        logger.error(f"   [ERROR] {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Forecast training failed for Episode {episode_num}: {e}"
        logger.error(f"   [ERROR] {error_msg}")
        raise RuntimeError(error_msg)


def get_episode_forecast_paths(episode_num: int, forecast_base_dir: str = "forecast_models"):
    """
    Get paths for episode-specific forecast models, scalers, and metadata.
    
    Args:
        episode_num: Episode number (0-19)
        forecast_base_dir: Base directory for forecast models
    
    Returns:
        dict with 'model_dir', 'scaler_dir', 'metadata_dir' keys
    """
    episode_dir = os.path.join(forecast_base_dir, f"episode_{episode_num}")
    return {
        'model_dir': os.path.join(episode_dir, "models"),
        'scaler_dir': os.path.join(episode_dir, "scalers"),
        'metadata_dir': os.path.join(episode_dir, "metadata")
    }


def ensure_episode_forecasts_ready(
    episode_num: int,
    episode_data_path: str,
    forecast_base_dir: str = "forecast_models",
    cache_base_dir: str = "forecast_cache"
) -> tuple[bool, dict]:
    """
    Ensure episode-specific forecast models are trained and ready.
    
    CRITICAL: Each episode trains models FROM SCRATCH on its OWN 6-month period only (not cumulative!):
    - Episode 0: Uses scenario_000.csv (2015 H1: Jan 1 - Jun 30, 2015)
    - Episode 1: Uses scenario_001.csv (2015 H2: Jul 1 - Dec 31, 2015)
    - Episode 19: Uses scenario_019.csv (2024 H2: Jul 1 - Dec 31, 2024)
    
    This ensures forecast models match each episode's data distribution exactly.
    Each episode is independent - no loading from previous episodes (unlike RL weights).
    
    This function:
    1. Checks if models exist for THIS episode
    2. If not, trains them FROM SCRATCH using episode-specific scenario file
    3. Returns paths for forecaster initialization
    
    Args:
        episode_num: Episode number (0-19)
        episode_data_path: Path to episode-specific scenario file (e.g., scenario_000.csv)
        forecast_base_dir: Base directory for forecast models
        cache_base_dir: Base directory for forecast cache
    
    Returns:
        (success: bool, paths: dict) where paths contains model_dir, scaler_dir, metadata_dir
    """
    # Check if models exist
    models_exist = check_episode_models_exist(episode_num, forecast_base_dir)
    
    if not models_exist:
        logger.info(f"   [FORECAST] Episode {episode_num} models not found - training now...")
        success = train_episode_forecast_models(
            episode_num=episode_num,
            episode_data_path=episode_data_path,
            forecast_base_dir=forecast_base_dir
        )
        
        if not success:
            logger.error(f"   [ERROR] Failed to train Episode {episode_num} forecast models")
            return False, {}
        
        # Verify models were created
        if not check_episode_models_exist(episode_num, forecast_base_dir):
            logger.error(f"   [ERROR] Episode {episode_num} models still not found after training")
            return False, {}
    else:
        logger.info(f"   [FORECAST] Episode {episode_num} models already exist - skipping training")
    
    # Get paths
    paths = get_episode_forecast_paths(episode_num, forecast_base_dir)
    
    return True, paths


def get_evaluation_forecast_paths(forecast_base_dir: str = "forecast_models"):
    """
    Get paths for evaluation on unseen 2025 data.
    
    Uses Episode 19 models (latest training period: 2015-2024 H2).
    These are the most recent forecast models trained on the full training dataset.
    
    Args:
        forecast_base_dir: Base directory for forecast models
    
    Returns:
        dict with 'model_dir', 'scaler_dir', 'metadata_dir' keys for Episode 19
    """
    return get_episode_forecast_paths(19, forecast_base_dir)

