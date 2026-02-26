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

        # Track unknown-field warnings so we only log each unknown field once
        self._warned_unknown_fields = set()
        
        # CSV file for step-by-step logging (will be set per episode)
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_handle = None
        
        # Category-specific CSV writers (portfolio, forecast, rewards, positions)
        self.category_writers = {}
        self.category_handles = {}
        self.category_fields = {
            'portfolio': [
                'episode', 'timestep',
                # Core portfolio metrics
                'portfolio_value_usd_millions', 'cash_dkk',
                'trading_gains_usd_thousands', 'operating_gains_usd_thousands', 'mtm_pnl',
                # Fund NAV component breakdown (DKK)
                'fund_nav_dkk', 'trading_cash_dkk', 'physical_book_value_dkk',
                'accumulated_operational_revenue_dkk', 'financial_mtm_dkk', 'financial_exposure_dkk',
                'depreciation_ratio', 'years_elapsed',
                # NAV attribution drivers
                'nav_start', 'nav_end', 'pnl_total', 'pnl_battery', 'pnl_generation', 'pnl_hedge', 'cash_delta_ops',
                # Battery dispatch metrics (FIX #5)
                'battery_decision', 'battery_intensity', 'battery_spread', 'battery_adjusted_hurdle', 'battery_volatility_adj',
                # Price & returns
                'price_current', 'price_return_1step', 'price_return_forecast'
            ],
            'positions': [
                'episode', 'timestep',
                'position_signed', 'position_exposure',
                'wind_pos_norm', 'solar_pos_norm', 'hydro_pos_norm',
                'decision_step', 'exposure_exec', 'action_sign',
                'trade_signal_active', 'trade_signal_sign',
                'risk_multiplier', 'vol_brake_mult', 'strategy_multiplier',
                'combined_multiplier', 'tradeable_capital', 'mtm_exit_count',
                'investor_action', 'battery_action',
                'inv_mu_raw', 'inv_sigma_raw', 'inv_a_raw', 'inv_tanh_mu', 'inv_tanh_a',
                'inv_sat_mean', 'inv_sat_sample', 'inv_sat_noise_only',
                'inv_reward_step', 'inv_value', 'inv_value_next', 'inv_td_error'
                , 'probe_r2_base', 'probe_r2_base_plus_signal', 'probe_delta_r2'
            ],
             'forecast': ['episode', 'timestep', 'z_short', 'z_medium', 'z_long', 'z_combined',
                          'forecast_confidence', 'forecast_mape', 'forecast_trust', 'signal_gate_multiplier',
                          'forecast_gate_passed', 'forecast_used', 'forecast_not_used_reason', 'agent_followed_forecast', 'forecast_usage_bonus', 'investor_strategy_multiplier',
                          'mape_short', 'mape_medium', 'mape_long',
                          'obs_trade_signal', 'obs_trade_action_corr', 'obs_trade_exposure_corr', 'obs_trade_delta_exposure_corr'],
            'rewards': ['episode', 'timestep', 'base_reward', 'investor_reward', 'battery_reward',
                        'risk_score', 'operational_score']
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
        # Single canonical file name per episode.
        # Desired behavior: re-running an episode overwrites its prior logs (e.g., after OOM/restart).
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
                # Single canonical file name per episode/category.
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
                 # Fund NAV component breakdown (DKK)
                 fund_nav_dkk: float = 0.0,
                 trading_cash_dkk: float = 0.0,
                 physical_book_value_dkk: float = 0.0,
                 accumulated_operational_revenue_dkk: float = 0.0,
                 financial_mtm_dkk: float = 0.0,
                 financial_exposure_dkk: float = 0.0,
                 depreciation_ratio: float = 0.0,
                 years_elapsed: float = 0.0,
                 # Forecast signals
                 z_short: float = 0.0,
                 z_medium: float = 0.0,
                 z_long: float = 0.0,
                 z_combined: float = 0.0,
                 forecast_confidence: float = 0.0,
                 forecast_trust: float = 0.0,
                 # OBSERVATION-LEVEL forecast signals (what the policy sees after warmup/ablation)
                 obs_z_short: float = 0.0,
                 # Deprecated: kept for backward-compat with older callers; not logged.
                 obs_z_medium: float = 0.0,
                 obs_z_long: float = 0.0,
                 obs_forecast_trust: float = 0.0,
                 obs_normalized_error: float = 0.0,
                 obs_trade_signal: float = 0.0,
                 obs_trade_action_corr: float = 0.0,
                 obs_trade_exposure_corr: float = 0.0,
                 obs_trade_delta_exposure_corr: float = 0.0,
                 signal_gate_multiplier: float = 0.0,
                 # Position info
                 position_signed: float = 0.0,
                 position_exposure: float = 0.0,
                 decision_step: float = 0.0,
                 exposure_exec: float = 0.0,
                 action_sign: float = 0.0,
                 trade_signal_active: float = 0.0,
                 trade_signal_sign: float = 0.0,
                 risk_multiplier: float = 1.0,
                 vol_brake_mult: float = 1.0,
                 strategy_multiplier: float = 1.0,
                 combined_multiplier: float = 1.0,
                 tradeable_capital: float = 0.0,
                 mtm_exit_count: float = 0.0,
                 # Price data (for forward-looking accuracy analysis)
                 price_current: float = 0.0,  # Raw price at current timestep (DKK/MWh)
                 # Price returns
                 price_return_1step: float = 0.0,
                 price_return_forecast: float = 0.0,
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
                 agent_followed_forecast: bool = False,
                 # NEW: NAV attribution drivers (per-step financial breakdown)
                 nav_start: float = 0.0,
                 nav_end: float = 0.0,
                 pnl_total: float = 0.0,
                 pnl_battery: float = 0.0,
                 pnl_generation: float = 0.0,
                 pnl_hedge: float = 0.0,
                 cash_delta_ops: float = 0.0,
                 # NEW: Battery dispatch metrics (FIX #5)
                 battery_decision: str = 'idle',
                 battery_intensity: float = 0.0,
                 battery_spread: float = 0.0,
                 battery_adjusted_hurdle: float = 0.0,
                 battery_volatility_adj: float = 0.0,
                 # NEW: Battery state metrics (BATTERY REWARD FIX)
                 battery_energy: float = 0.0,
                 battery_capacity: float = 0.0,
                 battery_soc: float = 0.0,
                 battery_cash_delta: float = 0.0,
                 battery_throughput: float = 0.0,
                 battery_degradation_cost: float = 0.0,
                 battery_eta_charge: float = 0.0,
                 battery_eta_discharge: float = 0.0,
                 # NEW (anti-collapse verification): penalty diagnostics
                 training_global_step: int = 0,
                 inv_penalty_warm: float = 0.0,
                 inv_pen_boundary: float = 0.0,
                 inv_pen_exposure: float = 0.0,
                 inv_pen_exposure_stuck: float = 0.0,
                 inv_mu_raw: float = 0.0,
                 inv_sigma_raw: float = 0.0,
                 inv_a_raw: float = 0.0,
                 inv_tanh_mu: float = 0.0,
                 inv_tanh_a: float = 0.0,
                 inv_sat_mean: float = 0.0,
                 inv_sat_sample: float = 0.0,
                 inv_sat_noise_only: float = 0.0,
                 inv_reward_step: float = 0.0,
                 inv_value: float = 0.0,
                 inv_value_next: float = 0.0,
                 inv_td_error: float = 0.0,
                 probe_r2_base: float = 0.0,
                 probe_r2_base_plus_signal: float = 0.0,
                 probe_delta_r2: float = 0.0):
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
            # Fund NAV component breakdown (DKK)
            'fund_nav_dkk': fund_nav_dkk,
            'trading_cash_dkk': trading_cash_dkk,
            'physical_book_value_dkk': physical_book_value_dkk,
            'accumulated_operational_revenue_dkk': accumulated_operational_revenue_dkk,
            'financial_mtm_dkk': financial_mtm_dkk,
            'financial_exposure_dkk': financial_exposure_dkk,
            'depreciation_ratio': depreciation_ratio,
            'years_elapsed': years_elapsed,
            # Position info
            'position_signed': position_signed,
            'position_exposure': position_exposure,
            'decision_step': decision_step,
            'exposure_exec': exposure_exec,
            'action_sign': action_sign,
            'trade_signal_active': trade_signal_active,
            'trade_signal_sign': trade_signal_sign,
            'risk_multiplier': risk_multiplier,
            'vol_brake_mult': vol_brake_mult,
            'strategy_multiplier': strategy_multiplier,
            'combined_multiplier': combined_multiplier,
            'tradeable_capital': tradeable_capital,
            'mtm_exit_count': mtm_exit_count,
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
            # Penalty diagnostics (for episode-boundary collapse debugging)
            'training_global_step': int(training_global_step),
            'inv_penalty_warm': float(inv_penalty_warm),
            'inv_pen_boundary': float(inv_pen_boundary),
            'inv_pen_exposure': float(inv_pen_exposure),
            'inv_pen_exposure_stuck': float(inv_pen_exposure_stuck),
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
            # OBSERVATION-LEVEL forecast signals (policy-visible; post-ablation)
            'obs_z_short': obs_z_short,
            'obs_z_long': obs_z_long,
            'obs_forecast_trust': obs_forecast_trust,
            'obs_normalized_error': obs_normalized_error,
            'obs_trade_signal': obs_trade_signal,
            'obs_trade_action_corr': obs_trade_action_corr,
            'obs_trade_exposure_corr': obs_trade_exposure_corr,
            'obs_trade_delta_exposure_corr': obs_trade_delta_exposure_corr,
            'signal_gate_multiplier': signal_gate_multiplier,
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
            # NEW: Forecast vs actual comparison (for deep debugging)
            'current_price_dkk': current_price_dkk,
            'forecast_price_short_dkk': forecast_price_short_dkk,
            'forecast_price_medium_dkk': forecast_price_medium_dkk,
            'forecast_price_long_dkk': forecast_price_long_dkk,
            'forecast_error_short_pct': forecast_error_short_pct,
            'forecast_error_medium_pct': forecast_error_medium_pct,
            'forecast_error_long_pct': forecast_error_long_pct,
            'agent_followed_forecast': int(agent_followed_forecast),
            # NEW: NAV attribution drivers (per-step financial breakdown)
            'nav_start': nav_start,
            'nav_end': nav_end,
            'pnl_total': pnl_total,
            'pnl_battery': pnl_battery,
            'pnl_generation': pnl_generation,
            'pnl_hedge': pnl_hedge,
            'cash_delta_ops': cash_delta_ops,
            # NEW: Battery dispatch metrics (FIX #5)
            'battery_decision': battery_decision,
            'battery_intensity': battery_intensity,
            'battery_spread': battery_spread,
            'battery_adjusted_hurdle': battery_adjusted_hurdle,
            'battery_volatility_adj': battery_volatility_adj,
            # NEW: Battery state metrics (BATTERY REWARD FIX)
            'battery_energy': battery_energy,
            'battery_capacity': battery_capacity,
            'battery_soc': battery_soc,
            'battery_cash_delta': battery_cash_delta,
            'battery_throughput': battery_throughput,
            'battery_degradation_cost': battery_degradation_cost,
            'battery_eta_charge': battery_eta_charge,
            'battery_eta_discharge': battery_eta_discharge,
            # Investor policy distribution diagnostics (pre-tanh / pre-postprocess)
            'inv_mu_raw': inv_mu_raw,
            'inv_sigma_raw': inv_sigma_raw,
            'inv_a_raw': inv_a_raw,
            'inv_tanh_mu': inv_tanh_mu,
            'inv_tanh_a': inv_tanh_a,
            'inv_sat_mean': inv_sat_mean,
            'inv_sat_sample': inv_sat_sample,
            'inv_sat_noise_only': inv_sat_noise_only,
            # Investor TD-error proxy (credit assignment diagnostics)
            'inv_reward_step': inv_reward_step,
            'inv_value': inv_value,
            'inv_value_next': inv_value_next,
            'inv_td_error': inv_td_error,
            # Linear probe (forecast utilization): exposure_exec ~ base_obs (+ trade_signal)
            'probe_r2_base': probe_r2_base,
            'probe_r2_base_plus_signal': probe_r2_base_plus_signal,
            'probe_delta_r2': probe_delta_r2,
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
        
        # NEW: NAV attribution drivers (per-step financial breakdown)
        row['nav_start'] = nav_start
        row['nav_end'] = nav_end
        row['pnl_total'] = pnl_total
        row['pnl_battery'] = pnl_battery
        row['pnl_generation'] = pnl_generation
        row['pnl_hedge'] = pnl_hedge
        row['cash_delta_ops'] = cash_delta_ops
        
        # NEW: Battery dispatch metrics (FIX #5)
        row['battery_decision'] = battery_decision
        row['battery_intensity'] = battery_intensity
        row['battery_spread'] = battery_spread
        row['battery_adjusted_hurdle'] = battery_adjusted_hurdle
        row['battery_volatility_adj'] = battery_volatility_adj
        # NEW: Battery state metrics (BATTERY REWARD FIX)
        row['battery_energy'] = battery_energy
        row['battery_capacity'] = battery_capacity
        row['battery_soc'] = battery_soc
        row['battery_cash_delta'] = battery_cash_delta
        row['battery_throughput'] = battery_throughput
        row['battery_degradation_cost'] = battery_degradation_cost
        row['battery_eta_charge'] = battery_eta_charge
        row['battery_eta_discharge'] = battery_eta_discharge
        
        # NEW: Forecast vs actual comparison
        row['current_price_dkk'] = current_price_dkk
        row['forecast_price_short_dkk'] = forecast_price_short_dkk
        row['forecast_price_medium_dkk'] = forecast_price_medium_dkk
        row['forecast_price_long_dkk'] = forecast_price_long_dkk
        row['forecast_error_short_pct'] = forecast_error_short_pct
        row['forecast_error_medium_pct'] = forecast_error_medium_pct
        row['forecast_error_long_pct'] = forecast_error_long_pct
        row['agent_followed_forecast'] = int(agent_followed_forecast)
        
        row_to_write = row
        try:
            writer_fieldnames = getattr(self.csv_writer, 'fieldnames', None)
            if writer_fieldnames:
                allowed_fields = set(writer_fieldnames)
                extra_fields = [key for key in row.keys() if key not in allowed_fields]
                if extra_fields:
                    new_fields = [f for f in extra_fields if f not in self._warned_unknown_fields]
                    if new_fields:
                        logging.warning(f"[LOGGER] Dropping unknown columns: {new_fields}")
                        self._warned_unknown_fields.update(new_fields)
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
            # Fund NAV component breakdown (DKK)
            'fund_nav_dkk', 'trading_cash_dkk', 'physical_book_value_dkk',
            'accumulated_operational_revenue_dkk', 'financial_mtm_dkk', 'financial_exposure_dkk',
            'depreciation_ratio', 'years_elapsed',
            # Position info (common to both tiers)
            'position_signed', 'position_exposure', 'decision_step', 'exposure_exec', 'action_sign',
            'trade_signal_active', 'trade_signal_sign',
            'risk_multiplier', 'vol_brake_mult', 'strategy_multiplier',
            'combined_multiplier', 'tradeable_capital', 'mtm_exit_count',
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
            # Penalty diagnostics (for episode-boundary collapse debugging)
            'training_global_step', 'inv_penalty_warm', 'inv_pen_boundary', 'inv_pen_exposure', 'inv_pen_exposure_stuck',
            # Investor policy distribution diagnostics (pre-tanh / pre-postprocess)
            'inv_mu_raw', 'inv_sigma_raw', 'inv_a_raw', 'inv_tanh_mu', 'inv_tanh_a',
            'inv_sat_mean', 'inv_sat_sample', 'inv_sat_noise_only',
            # Investor TD-error proxy (credit assignment diagnostics)
            'inv_reward_step', 'inv_value', 'inv_value_next', 'inv_td_error',
            # Linear probe (forecast utilization)
            'probe_r2_base', 'probe_r2_base_plus_signal', 'probe_delta_r2',
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
            # OBSERVATION-LEVEL forecast signals (what the policy actually sees after ablations/warmup)
            # These are critical for proving forecast usage and for causal ablations.
            'obs_z_short', 'obs_z_long',
            'obs_forecast_trust', 'obs_normalized_error', 'obs_trade_signal', 'obs_trade_action_corr',
            'obs_trade_exposure_corr', 'obs_trade_delta_exposure_corr',
            'signal_gate_multiplier',
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
            'agent_followed_forecast',
            # NEW: NAV attribution drivers (per-step financial breakdown)
            'nav_start', 'nav_end', 'pnl_total', 'pnl_battery', 'pnl_generation', 'pnl_hedge', 'cash_delta_ops',
            # NEW: Battery dispatch metrics (FIX #5)
            'battery_decision', 'battery_intensity', 'battery_spread', 'battery_adjusted_hurdle', 'battery_volatility_adj',
            # NEW: Battery state metrics for diagnostics (BATTERY REWARD FIX)
            'battery_energy', 'battery_capacity', 'battery_soc',
            'battery_cash_delta', 'battery_throughput', 'battery_degradation_cost',
            'battery_eta_charge', 'battery_eta_discharge'
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
