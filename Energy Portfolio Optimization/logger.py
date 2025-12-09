"""
Centralized Logging Module

This module provides:
1. Centralized application logging configuration
2. RewardLogger: CSV data logging for Tier 1 vs Tier 2 comparison

ALL logging should go through this module - no print() statements or direct logging.basicConfig() calls.
"""

import csv
import json
import os
import sys
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================

_logging_configured = False
_log_file_handler = None


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    force_reconfigure: bool = False
) -> None:
    """
    Configure centralized logging for the entire application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
        force_reconfigure: If True, reconfigure even if already configured (default: False)
    """
    global _logging_configured, _log_file_handler
    
    if _logging_configured and not force_reconfigure:
        # Already configured, but allow adding file handler if not present
        if log_file and _log_file_handler is None:
            root_logger = logging.getLogger()
            if format_string is None:
                format_string = '%(levelname)s: %(message)s'
            _log_file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            _log_file_handler.setLevel(level)
            _log_file_handler.setFormatter(logging.Formatter(format_string))
            root_logger.addHandler(_log_file_handler)
        return
    
    if format_string is None:
        format_string = '%(levelname)s: %(message)s'
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Close previous file handler if exists
    if _log_file_handler is not None:
        try:
            _log_file_handler.close()
        except Exception:
            pass
        _log_file_handler = None
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        _log_file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        _log_file_handler.setLevel(level)
        _log_file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(_log_file_handler)
    
    root_logger.setLevel(level)
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    if not _logging_configured:
        configure_logging()  # Auto-configure if not done
    return logging.getLogger(name)


# ============================================================================
# TERMINAL OUTPUT TEE (for capturing print statements)
# ============================================================================

class TeeOutput:
    """
    Class to tee output to both console and file.
    Captures both print() statements and logging output.
    Also captures exceptions and tracebacks.
    
    This class can be assigned to both sys.stdout and sys.stderr,
    and will write to the original streams plus the log file.
    """
    def __init__(self, file_path: str):
        self.file = open(file_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.file_path = file_path
        self._is_stdout = False
        self._is_stderr = False
        
    def _set_stream_type(self, is_stdout: bool = False, is_stderr: bool = False):
        """Set which stream type this instance represents"""
        self._is_stdout = is_stdout
        self._is_stderr = is_stderr
        
    def write(self, text: str) -> None:
        """Write to both console and file"""
        try:
            # Determine which original stream to write to
            original_stream = None
            if self._is_stdout or (not self._is_stderr and not self._is_stdout):
                original_stream = self._original_stdout
            elif self._is_stderr:
                original_stream = self._original_stderr
            else:
                original_stream = self._original_stdout  # Default to stdout
            
            # Write to original stream (console)
            if original_stream:
                original_stream.write(text)
                original_stream.flush()
            
            # Always write to file
            if self.file and not self.file.closed:
                self.file.write(text)
                self.file.flush()  # Ensure immediate write
        except Exception as e:
            # If file write fails, at least try to write to console
            try:
                if original_stream:
                    original_stream.write(text)
                    original_stream.flush()
            except Exception:
                pass
        
    def flush(self) -> None:
        """Flush both streams"""
        try:
            if self._is_stdout or (not self._is_stderr and not self._is_stdout):
                self._original_stdout.flush()
            elif self._is_stderr:
                self._original_stderr.flush()
            if self.file and not self.file.closed:
                self.file.flush()
        except Exception:
            pass
        
    def close(self) -> None:
        """Close the file"""
        try:
            if self.file and not self.file.closed:
                self.file.flush()
                self.file.close()
        except Exception:
            pass
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def isatty(self) -> bool:
        """Return False to indicate this is not a TTY (for libraries that check)"""
        return False
    
    def fileno(self) -> int:
        """Return file descriptor for original stdout/stderr"""
        if self._is_stdout or (not self._is_stderr and not self._is_stdout):
            return self._original_stdout.fileno()
        elif self._is_stderr:
            return self._original_stderr.fileno()
        else:
            return self._original_stdout.fileno()
    
    @property
    def stdout(self):
        """Return original stdout for restoration"""
        return self._original_stdout
    
    @property
    def stderr(self):
        """Return original stderr for restoration"""
        return self._original_stderr


# ============================================================================
# REWARD LOGGER (CSV Data Logging)
# ============================================================================

class RewardLogger:
    """Logs detailed reward, forecast, and portfolio information at every timestep"""
    
    def __init__(self, log_dir: str = "debug_logs", tier_name: str = "tier1"):
        self.log_dir = log_dir
        self.tier_name = tier_name
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file for step-by-step logging (will be set per episode)
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_handle = None
        
        # Category-specific CSV writers (portfolio, forecast, rewards, positions)
        self.category_writers = {}
        self.category_handles = {}
        self.category_fields = {
            'portfolio': ['episode', 'timestep', 'portfolio_value_usd_millions', 'cash_dkk',
                          'trading_gains_usd_thousands', 'operating_gains_usd_thousands', 'mtm_pnl',
                          'price_current', 'price_return_1step', 'price_return_forecast'],
            'positions': ['episode', 'timestep', 'position_signed', 'position_exposure',
                          'wind_pos_norm', 'solar_pos_norm', 'hydro_pos_norm',
                          'investor_action', 'battery_action'],
             'forecast': ['episode', 'timestep', 'z_short', 'z_medium', 'z_long', 'z_combined',
                          'forecast_confidence', 'forecast_mape', 'forecast_trust', 'signal_gate_multiplier',
                          'forecast_gate_passed', 'forecast_used', 'forecast_not_used_reason', 'agent_followed_forecast', 'forecast_usage_bonus', 'investor_strategy_multiplier',
                         'mape_short', 'mape_medium', 'mape_long'],
            'rewards': ['episode', 'timestep', 'base_reward', 'investor_reward', 'battery_reward',
                        'alignment_reward', 'pnl_reward', 'forecast_signal_score', 'risk_score',
                        'operational_score']
        }
        
        # Track episode-level aggregates
        self.episode_data = []
        self.current_episode = 0
        
    def start_episode(self, episode_num: int):
        """Initialize logging for a new episode - creates a new CSV file per episode"""
        # Close previous episode's file if open
        if self.csv_file_handle is not None:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None
        
        # Close previous category files
        for handle in self.category_handles.values():
            if handle is not None:
                handle.close()
        self.category_handles.clear()
        self.category_writers.clear()
        
        self.current_episode = episode_num
        
        # Create new CSV file for this episode (renamed from _reward_debug_ to _debug_)
        self.csv_file = os.path.join(self.log_dir, f"{self.tier_name}_debug_ep{episode_num}.csv")
        
        try:
            # Always create new file (write mode) for each episode
            self.csv_file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=self._get_fieldnames())
            self.csv_writer.writeheader()
            self.csv_file_handle.flush()
            logging.info(f"[LOGGER] Created new log file for episode {episode_num}: {self.csv_file}")
            
            # Create category-specific CSV files
            self._create_category_csvs(episode_num)
        except Exception as e:
            logging.exception(f"[LOGGER] ERROR creating log file {self.csv_file}: {e}")
    
    def _create_category_csvs(self, episode_num: int):
        """Create category-specific CSV files for better organization."""
        try:
            for category, fields in self.category_fields.items():
                csv_file = os.path.join(self.log_dir, f"{self.tier_name}_{category}_ep{episode_num}.csv")
                try:
                    handle = open(csv_file, 'w', newline='', encoding='utf-8')
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    handle.flush()
                    self.category_handles[category] = handle
                    self.category_writers[category] = writer
                    logging.debug(f"[LOGGER] Created category CSV: {csv_file}")
                except Exception as e:
                    logging.warning(f"[LOGGER] Could not create {category} CSV: {e}")
        except Exception as e:
            logging.warning(f"[LOGGER] Could not create category CSVs: {e}")
    
    def log_step(self, 
                 timestep: int,
                 # Portfolio metrics (logged at every timestep)
                 portfolio_value_usd_millions: float = 0.0,
                 cash_dkk: float = 0.0,
                 trading_gains_usd_thousands: float = 0.0,
                 operating_gains_usd_thousands: float = 0.0,
                 # Forecast signals
                 z_short: float = 0.0,
                 z_medium: float = 0.0,
                 z_long: float = 0.0,
                 z_combined: float = 0.0,
                 forecast_confidence: float = 0.0,
                 forecast_trust: float = 0.0,
                 signal_gate_multiplier: float = 0.0,
                 # Position info
                 position_signed: float = 0.0,
                 position_exposure: float = 0.0,
                 # Price data (for forward-looking accuracy analysis)
                 price_current: float = 0.0,  # Raw price at current timestep (DKK/MWh)
                 # Price returns
                 price_return_1step: float = 0.0,
                 price_return_forecast: float = 0.0,
                 # Forecast reward components
                 alignment_reward: float = 0.0,
                 pnl_reward: float = 0.0,
                 forecast_signal_score: float = 0.0,
                 generation_forecast_score: float = 0.0,
                 combined_forecast_score: float = 0.0,
                 trust_scale: float = 1.0,
                 warmup_factor: float = 1.0,
                 # Main reward components
                 operational_score: float = 0.0,
                 risk_score: float = 0.0,
                 hedging_score: float = 0.0,
                 nav_stability_score: float = 0.0,
                 forecast_score: float = 0.0,
                 # Reward weights
                 weight_operational: float = 0.0,
                 weight_risk: float = 0.0,
                 weight_hedging: float = 0.0,
                 weight_nav: float = 0.0,
                 weight_forecast: float = 0.0,
                 # Final rewards
                 base_reward: float = 0.0,
                 investor_reward: float = 0.0,
                 battery_reward: float = 0.0,
                 # Financial metrics
                 mtm_pnl: float = 0.0,  # Per-step MTM PnL (not cumulative)
                 # Actions
                 investor_action: Optional[Dict] = None,
                 battery_action: Optional[Dict] = None,
                 # NEW: Enhanced debugging fields
                 # Alignment debugging
                 base_alignment: float = 0.0,
                 profitability_factor: float = 0.0,
                 alignment_multiplier: float = 0.0,
                 misalignment_penalty_mult: float = 0.0,
                 forecast_direction: float = 0.0,
                 position_direction: float = 0.0,
                 is_aligned: float = 0.0,
                 alignment_status: str = '',
                 # Correlation debugging
                 corr_short: float = 0.0,
                 corr_medium: float = 0.0,
                 corr_long: float = 0.0,
                 weight_short: float = 0.0,
                 weight_medium: float = 0.0,
                 weight_long: float = 0.0,
                 use_short: bool = False,
                 use_medium: bool = False,
                 use_long: bool = False,
                 # Reward breakdown debugging
                 unrealized_pnl_norm: float = 0.0,
                 combined_confidence: float = 0.0,
                 adaptive_multiplier: float = 0.0,
                 # Position breakdown
                 wind_pos_norm: float = 0.0,
                 solar_pos_norm: float = 0.0,
                 hydro_pos_norm: float = 0.0,
                 # Forecast quality
                 forecast_error: float = 0.0,
                 forecast_mape: float = 0.0,
                 realized_vs_forecast: float = 0.0,
                 # NEW: Battery and generation forecast debugging
                 z_short_wind: float = 0.0,
                 z_short_solar: float = 0.0,
                 z_short_hydro: float = 0.0,
                 z_short_total_gen: float = 0.0,
                 # NEW: Adaptive forecast weight
                 adaptive_forecast_weight: float = 0.0,
                 # NEW: Forecast utilization flag
                 enable_forecast_util: bool = False,
                 forecast_gate_passed: bool = False,
                 forecast_used: bool = False,
                 forecast_not_used_reason: str = '',
                 # NOVEL: Investor strategy fields
                 investor_strategy: str = 'none',
                 investor_position_scale: float = 0.0,
                 investor_signal_strength: float = 0.0,
                 investor_consensus: bool = False,
                 investor_direction: int = 0,
                 investor_strategy_bonus: float = 0.0,
                 investor_strategy_multiplier: float = 1.0,
                 forecast_usage_bonus: float = 0.0,
                 # NEW: Position alignment diagnostics (Bug #1 fix tracking)
                 position_alignment_status: str = 'no_strategy',
                 investor_position_ratio: float = 0.0,
                 investor_position_direction: int = 0,
                 investor_total_position: float = 0.0,
                 investor_exploration_bonus: float = 0.0,
                 # NEW: Per-horizon MAPE tracking
                 mape_short: float = 0.0,
                 mape_medium: float = 0.0,
                 mape_long: float = 0.0,
                 mape_short_recent: float = 0.0,
                 mape_medium_recent: float = 0.0,
                 mape_long_recent: float = 0.0,
                 # NEW: Per-horizon correlation tracking
                 horizon_corr_short: float = 0.0,
                 horizon_corr_medium: float = 0.0,
                 horizon_corr_long: float = 0.0,
                 # NEW: Per-asset forecast accuracy
                 mape_wind: float = 0.0,
                 mape_solar: float = 0.0,
                 mape_hydro: float = 0.0,
                 mape_price: float = 0.0,
                 mape_load: float = 0.0,
                 # NEW: Forecast deltas (for debugging strategy)
                 forecast_delta_short: float = 0.0,
                 forecast_delta_medium: float = 0.0,
                 forecast_delta_long: float = 0.0,
                 # NEW: MAPE thresholds (for debugging strategy)
                 mape_threshold_short: float = 0.0,
                 mape_threshold_medium: float = 0.0,
                 mape_threshold_long: float = 0.0,
                 # NEW: Price floor diagnostics (Bug #4 fix tracking)
                 price_raw: float = 0.0,
                 price_floor_used: float = 0.0,
                 # NEW: Directional accuracy
                 direction_accuracy_short: float = 0.0,
                 direction_accuracy_medium: float = 0.0,
                 direction_accuracy_long: float = 0.0,
                 # NEW: Battery forecast bonus
                 battery_forecast_bonus: float = 0.0,
                 # NEW: Forecast vs actual comparison (for deep debugging)
                 current_price_dkk: float = 0.0,
                 forecast_price_short_dkk: float = 0.0,
                 forecast_price_medium_dkk: float = 0.0,
                 forecast_price_long_dkk: float = 0.0,
                 forecast_error_short_pct: float = 0.0,
                 forecast_error_medium_pct: float = 0.0,
                 forecast_error_long_pct: float = 0.0,
                 agent_followed_forecast: bool = False):
        """Log detailed step information"""
        # DEBUG to verify log_step is being called
        if timestep % 100 == 0:
            logging.debug(f"[LOGGER] log_step() called at timestep={timestep}")

        # Auto-initialize if not already done (safety check)
        if self.csv_writer is None:
            logging.warning(f"[LOGGER] csv_writer is NONE at timestep={timestep}, initializing...")
            self.start_episode(self.current_episode if self.current_episode > 0 else 0)

        if self.csv_writer is None:
            logging.error(f"[LOGGER] csv_writer STILL NONE after initialization at timestep={timestep}, SKIPPING!")
            return  # Still None after initialization attempt - skip logging

        # DEBUG to verify we're about to write
        if timestep % 100 == 0:
            logging.debug(f"[LOGGER] csv_writer EXISTS, writing row for timestep={timestep}")

        # CRITICAL FIX: Use forecast_direction parameter instead of computing from z_combined
        # The new approach uses forecast_direction (weighted combination of z-scores)
        # instead of z_combined (correlation-based combination)
        forecast_dir = forecast_direction
        position_dir = position_direction
        aligned = is_aligned
        align_status = alignment_status if alignment_status else 'NEUTRAL'

        # Convert boolean to int for CSV (0 or 1)
        investor_consensus_int = 1 if investor_consensus else 0
        
        row = {
            # ===================================================================
            # COMMON COLUMNS (Tier 1 & Tier 2) - First columns
            # ===================================================================
            # Portfolio metrics (logged at every timestep)
            'episode': self.current_episode,
            'timestep': timestep,
            'portfolio_value_usd_millions': portfolio_value_usd_millions,
            'cash_dkk': cash_dkk,
            'trading_gains_usd_thousands': trading_gains_usd_thousands,
            'operating_gains_usd_thousands': operating_gains_usd_thousands,
            # Primary portfolio metrics (mtm_pnl is per-step, not cumulative)
            'mtm_pnl': mtm_pnl,
            # Position info
            'position_signed': position_signed,
            'position_exposure': position_exposure,
            # Price data (for forward-looking accuracy analysis)
            'price_current': price_current,  # Raw price at current timestep (DKK/MWh)
            # Price returns
            'price_return_1step': price_return_1step,
            'price_return_forecast': price_return_forecast,
            # Main reward components
            'operational_score': operational_score,
            'risk_score': risk_score,
            'hedging_score': hedging_score,
            'nav_stability_score': nav_stability_score,
            # Reward weights
            'weight_operational': weight_operational,
            'weight_risk': weight_risk,
            'weight_hedging': weight_hedging,
            'weight_nav': weight_nav,
            # Final rewards
            'base_reward': base_reward,
            'investor_reward': investor_reward,
            'battery_reward': battery_reward,
            # Position breakdown
            'wind_pos_norm': wind_pos_norm,
            'solar_pos_norm': solar_pos_norm,
            'hydro_pos_norm': hydro_pos_norm,
            # Reward breakdown debugging
            'unrealized_pnl_norm': unrealized_pnl_norm,
            'combined_confidence': combined_confidence,
            'adaptive_multiplier': adaptive_multiplier,
            
            # ===================================================================
            # FORECAST-RELATED COLUMNS (Tier 2 only) - At the END
            # ===================================================================
            # Forecast signals
            'z_short': z_short,
            'z_medium': z_medium,
            'z_long': z_long,
            'z_combined': z_combined,
            'forecast_confidence': forecast_confidence,
            'forecast_trust': forecast_trust,
            'signal_gate_multiplier': signal_gate_multiplier,
            # Forecast reward components
            'alignment_reward': alignment_reward,
            'pnl_reward': pnl_reward,
            'forecast_signal_score': forecast_signal_score,
            'generation_forecast_score': generation_forecast_score,
            'combined_forecast_score': combined_forecast_score,
            'trust_scale': trust_scale,
            'warmup_factor': warmup_factor,
            # Forecast score
            'forecast_score': forecast_score,
            # Forecast weight
            'weight_forecast': weight_forecast,
            # Alignment debugging
            'base_alignment': base_alignment,
            'profitability_factor': profitability_factor,
            'alignment_multiplier': alignment_multiplier,
            'misalignment_penalty_mult': misalignment_penalty_mult,
            'forecast_direction': forecast_dir,
            'position_direction': position_dir,
            'is_aligned': aligned,
            'alignment_status': align_status,
            # Correlation debugging
            'corr_short': corr_short,
            'corr_medium': corr_medium,
            'corr_long': corr_long,
            'weight_short': weight_short,
            'weight_medium': weight_medium,
            'weight_long': weight_long,
            'use_short': 1.0 if use_short else 0.0,
            'use_medium': 1.0 if use_medium else 0.0,
            'use_long': 1.0 if use_long else 0.0,
            # Forecast quality
            'forecast_error': forecast_error,
            'forecast_mape': forecast_mape,
            'realized_vs_forecast': realized_vs_forecast,
            # Battery and generation forecast debugging
            'z_short_wind': z_short_wind,
            'z_short_solar': z_short_solar,
            'z_short_hydro': z_short_hydro,
            'z_short_total_gen': z_short_total_gen,
            # Adaptive forecast weight
            'adaptive_forecast_weight': adaptive_forecast_weight,
            # Forecast utilization flag
            'enable_forecast_util': 1.0 if enable_forecast_util else 0.0,
            'forecast_gate_passed': 1.0 if forecast_gate_passed else 0.0,
            'forecast_used': 1.0 if forecast_used else 0.0,
            'forecast_not_used_reason': forecast_not_used_reason,
            'agent_followed_forecast': 1.0 if agent_followed_forecast else 0.0,
            # NOVEL: Investor strategy fields
            'investor_strategy': investor_strategy,
            'investor_position_scale': investor_position_scale,
            'investor_signal_strength': investor_signal_strength,
            'investor_consensus': investor_consensus_int,
            'investor_direction': investor_direction,
            'investor_strategy_bonus': investor_strategy_bonus,
            'investor_strategy_multiplier': investor_strategy_multiplier,
            'forecast_usage_bonus': forecast_usage_bonus,
            # NEW: Position alignment diagnostics (Bug #1 fix tracking)
            'position_alignment_status': position_alignment_status,
            'investor_position_ratio': investor_position_ratio,
            'investor_position_direction': investor_position_direction,
            'investor_total_position': investor_total_position,
            'investor_exploration_bonus': investor_exploration_bonus,
            # NEW: Per-horizon MAPE tracking
            'mape_short': mape_short,
            'mape_medium': mape_medium,
            'mape_long': mape_long,
            'mape_short_recent': mape_short_recent,
            'mape_medium_recent': mape_medium_recent,
            'mape_long_recent': mape_long_recent,
            # NEW: Per-horizon correlation tracking
            'horizon_corr_short': horizon_corr_short,
            'horizon_corr_medium': horizon_corr_medium,
            'horizon_corr_long': horizon_corr_long,
            # NEW: Per-asset forecast accuracy
            'mape_wind': mape_wind,
            'mape_solar': mape_solar,
            'mape_hydro': mape_hydro,
            'mape_price': mape_price,
            'mape_load': mape_load,
            # NEW: Forecast deltas (for debugging strategy)
            'forecast_delta_short': forecast_delta_short,
            'forecast_delta_medium': forecast_delta_medium,
            'forecast_delta_long': forecast_delta_long,
            # NEW: MAPE thresholds (for debugging strategy)
            'mape_threshold_short': mape_threshold_short,
            'mape_threshold_medium': mape_threshold_medium,
            'mape_threshold_long': mape_threshold_long,
            # NEW: Price floor diagnostics (Bug #4 fix tracking)
            'price_raw': price_raw,
            'price_floor_used': price_floor_used,
            # NEW: Directional accuracy
            'direction_accuracy_short': direction_accuracy_short,
            'direction_accuracy_medium': direction_accuracy_medium,
            'direction_accuracy_long': direction_accuracy_long,
            # NEW: Battery forecast bonus
            'battery_forecast_bonus': battery_forecast_bonus,
        }
        
        # Add action info if available
        # CRITICAL FIX: Handle arrays and dicts properly to avoid boolean ambiguity
        def safe_get_action_value(action_dict, key, default=0.0):
            """Safely extract action value, handling arrays and dicts"""
            if action_dict is None:
                return default
            if not isinstance(action_dict, dict):
                return default
            val = action_dict.get(key, default)
            # Handle numpy arrays
            if isinstance(val, np.ndarray):
                return float(val.item() if val.size == 1 else val.flatten()[0])
            return float(val) if val is not None else default

        def serialize_action(action_dict):
            """Serialize an action dict (possibly with numpy values) into JSON for CSV logs."""
            if action_dict is None or not isinstance(action_dict, dict) or len(action_dict) == 0:
                return ''

            def _normalize(value):
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        return float(value.item())
                    return value.astype(float).tolist()
                if isinstance(value, (np.generic,)):
                    return float(value)
                if isinstance(value, dict):
                    return {k: _normalize(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_normalize(v) for v in value]
                return float(value) if isinstance(value, (int, float)) else value

            normalized = {k: _normalize(v) for k, v in action_dict.items()}
            try:
                return json.dumps(normalized, separators=(',', ':'))
            except (TypeError, ValueError):
                return str(normalized)
        
        if investor_action is not None and isinstance(investor_action, dict):
            row['inv_wind'] = safe_get_action_value(investor_action, 'wind', 0.0)
            row['inv_solar'] = safe_get_action_value(investor_action, 'solar', 0.0)
            row['inv_hydro'] = safe_get_action_value(investor_action, 'hydro', 0.0)
        else:
            row['inv_wind'] = 0.0
            row['inv_solar'] = 0.0
            row['inv_hydro'] = 0.0
            
        if battery_action is not None and isinstance(battery_action, dict):
            row['batt_charge'] = safe_get_action_value(battery_action, 'charge', 0.0)
            row['batt_discharge'] = safe_get_action_value(battery_action, 'discharge', 0.0)
        else:
            row['batt_charge'] = 0.0
            row['batt_discharge'] = 0.0

        # Preserve raw JSON actions for category-specific logs
        row['investor_action'] = serialize_action(investor_action)
        row['battery_action'] = serialize_action(battery_action)
        
        row_to_write = row
        try:
            writer_fieldnames = getattr(self.csv_writer, 'fieldnames', None)
            if writer_fieldnames:
                allowed_fields = set(writer_fieldnames)
                extra_fields = [key for key in row.keys() if key not in allowed_fields]
                if extra_fields:
                    logging.warning(f"[LOGGER] Dropping unknown columns: {extra_fields}")
                row_to_write = {field: row.get(field, '') for field in writer_fieldnames}
        except Exception as filter_err:
            logging.debug(f"[LOGGER] Could not align row with fieldnames: {filter_err}")
            row_to_write = row

        try:
            self.csv_writer.writerow(row_to_write)
            self.csv_file_handle.flush()
            # DEBUG to verify write succeeded
            if timestep % 100 == 0:
                logging.debug(f"[LOGGER] Successfully wrote row for timestep={timestep} to {self.csv_file}")
            
            # Write to category-specific CSVs if enabled
            self._write_to_category_csvs(row)
        except Exception as e:
            logging.exception(f"[LOGGER] ERROR writing to log file: {e}")
    
    def _write_to_category_csvs(self, row: dict):
        """Write row data to category-specific CSV files (only the fields defined for each category)."""
        try:
            for category, writer in self.category_writers.items():
                if writer is None:
                    continue

                if not self._is_valid_category_row(category, row):
                    continue

                handle = self.category_handles.get(category)
                if handle is not None:
                    # Extract only fields that belong to this category
                    try:
                        fields = self.category_fields.get(category, [])
                        filtered_row = {k: row.get(k, '') for k in fields if k in row or k in ['episode', 'timestep']}
                        writer.writerow(filtered_row)
                        handle.flush()
                    except Exception as e:
                        logging.debug(f"[LOGGER] Could not write to {category} CSV: {e}")
        except Exception as e:
            logging.debug(f"[LOGGER] Error in _write_to_category_csvs: {e}")

    def _is_valid_category_row(self, category: str, row: dict) -> bool:
        episode = row.get('episode')
        timestep = row.get('timestep')

        if not self._is_valid_nonnegative_number(episode, max_value=100000):
            return False

        if not self._is_valid_nonnegative_number(timestep, max_value=1000000):
            return False

        if category == 'portfolio':
            portfolio_val = row.get('portfolio_value_usd_millions')
            if not self._is_valid_numeric_value(portfolio_val):
                return False

        return True

    @staticmethod
    def _is_valid_nonnegative_number(value, max_value=1e9) -> bool:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False
        return 0.0 <= val <= max_value and np.isfinite(val)

    @staticmethod
    def _is_valid_numeric_value(value) -> bool:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False
        return np.isfinite(val)
    
    def _get_fieldnames(self):
        """Get CSV field names - common columns first, forecast columns at the end"""
        fieldnames = [
            # ===================================================================
            # COMMON COLUMNS (Tier 1 & Tier 2) - First columns
            # ===================================================================
            # Portfolio metrics (logged at every timestep)
            'episode', 'timestep',
            'portfolio_value_usd_millions', 'cash_dkk',
            'trading_gains_usd_thousands', 'operating_gains_usd_thousands',
            # Primary portfolio metrics (mtm_pnl is per-step, not cumulative)
            'mtm_pnl',  # Mark-to-market PnL (per-step, not cumulative)
            # Position info (common to both tiers)
            'position_signed', 'position_exposure',
            # Price data (for forward-looking accuracy analysis)
            'price_current',  # Raw price at current timestep (DKK/MWh)
            # Price returns (common to both tiers)
            'price_return_1step', 'price_return_forecast',
            # Main reward components (common to both tiers)
            'operational_score', 'risk_score', 'hedging_score', 'nav_stability_score',
            # Reward weights (common to both tiers, forecast weight will be 0.0 for Tier 1)
            'weight_operational', 'weight_risk', 'weight_hedging', 'weight_nav',
            # Final rewards (common to both tiers)
            'base_reward', 'investor_reward', 'battery_reward',
            # Actions (common to both tiers)
            'inv_wind', 'inv_solar', 'inv_hydro', 'batt_charge', 'batt_discharge',
            'investor_action', 'battery_action',
            # Position breakdown (common to both tiers)
            'wind_pos_norm', 'solar_pos_norm', 'hydro_pos_norm',
            # Reward breakdown debugging (common to both tiers)
            'unrealized_pnl_norm', 'combined_confidence', 'adaptive_multiplier',
            
            # ===================================================================
            # FORECAST-RELATED COLUMNS (Tier 2 only) - At the END
            # ===================================================================
            # Forecast signals (0.0 for Tier 1, populated for Tier 2)
            'z_short', 'z_medium', 'z_long', 'z_combined', 'forecast_confidence', 'forecast_trust',
            'signal_gate_multiplier',
            # Forecast reward components (0.0 for Tier 1, populated for Tier 2)
            'alignment_reward', 'pnl_reward', 'forecast_signal_score', 'generation_forecast_score',
            'combined_forecast_score', 'trust_scale', 'warmup_factor',
            # Forecast score from reward components (0.0 for Tier 1)
            'forecast_score',
            # Forecast weight from reward weights (0.0 for Tier 1)
            'weight_forecast',
            # Alignment debugging (0.0/NEUTRAL for Tier 1, populated for Tier 2)
            'base_alignment', 'profitability_factor', 'alignment_multiplier', 'misalignment_penalty_mult',
            'forecast_direction', 'position_direction', 'is_aligned', 'alignment_status',
            # Correlation debugging (0.0 for Tier 1, populated for Tier 2)
            'corr_short', 'corr_medium', 'corr_long', 'weight_short', 'weight_medium', 'weight_long',
            'use_short', 'use_medium', 'use_long',
            # Forecast quality (0.0 for Tier 1, populated for Tier 2)
            'forecast_error', 'forecast_mape', 'realized_vs_forecast',
            # Battery and generation forecast debugging (0.0 for Tier 1, populated for Tier 2)
            'z_short_wind', 'z_short_solar', 'z_short_hydro', 'z_short_total_gen',
            # Adaptive forecast weight (0.0 for Tier 1, populated for Tier 2)
            'adaptive_forecast_weight',
            # Forecast utilization flag (0.0 for Tier 1, 1.0 for Tier 2)
            'enable_forecast_util', 'forecast_gate_passed', 'forecast_used', 'forecast_not_used_reason',
            # NOVEL: Investor strategy fields (Tier 2 only)
            'investor_strategy', 'investor_position_scale', 'investor_signal_strength',
            'investor_consensus', 'investor_direction', 'investor_strategy_bonus', 'investor_strategy_multiplier', 'forecast_usage_bonus',
            # NEW: Position alignment diagnostics (Bug #1 fix tracking)
            'position_alignment_status', 'investor_position_ratio', 'investor_position_direction',
            'investor_total_position', 'investor_exploration_bonus',
            # NEW: Per-horizon MAPE tracking (Tier 2 only)
            'mape_short', 'mape_medium', 'mape_long',
            'mape_short_recent', 'mape_medium_recent', 'mape_long_recent',
            # NEW: Per-horizon correlation tracking (Tier 2 only)
            'horizon_corr_short', 'horizon_corr_medium', 'horizon_corr_long',
            # NEW: Per-asset forecast accuracy (Tier 2 only)
            'mape_wind', 'mape_solar', 'mape_hydro', 'mape_price', 'mape_load',
            # NEW: Forecast deltas for debugging strategy (Tier 2 only)
            'forecast_delta_short', 'forecast_delta_medium', 'forecast_delta_long',
            # NEW: MAPE thresholds for debugging strategy (Tier 2 only)
            'mape_threshold_short', 'mape_threshold_medium', 'mape_threshold_long',
            # NEW: Price floor diagnostics (Bug #4 fix tracking)
            'price_raw', 'price_floor_used',
            # NEW: Directional accuracy (Tier 2 only)
            'direction_accuracy_short', 'direction_accuracy_medium', 'direction_accuracy_long',
            # NEW: Battery forecast bonus (Tier 2 only)
            'battery_forecast_bonus',
            # NEW: Forecast vs actual comparison (deep debugging)
            'current_price_dkk', 'forecast_price_short_dkk', 'forecast_price_medium_dkk', 'forecast_price_long_dkk',
            'forecast_error_short_pct', 'forecast_error_medium_pct', 'forecast_error_long_pct',
            'agent_followed_forecast'
        ]

        # Preserve insertion order while removing duplicates
        deduped = []
        seen = set()
        for name in fieldnames:
            if name not in seen:
                deduped.append(name)
                seen.add(name)

        # Guarantee serialized action columns are present even if future edits remove them
        for required in ('investor_action', 'battery_action'):
            if required not in seen:
                deduped.append(required)
                seen.add(required)

        return deduped
    
    def close(self):
        """Close log files"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None

