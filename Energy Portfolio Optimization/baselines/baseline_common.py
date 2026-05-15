"""Shared baseline utilities aligned with the current Tier1 evaluation contract."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


BASELINES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINES_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class BaselineEconomicConfig:
    initial_budget_usd: float
    dkk_to_usd_rate: float
    annual_risk_free_rate: float
    distribution_rate: float
    target_cash_ratio: float
    min_distribution_threshold_ratio: float
    time_step_hours: float
    owned_wind_capacity_mw: float
    owned_solar_capacity_mw: float
    owned_hydro_capacity_mw: float
    owned_battery_capacity_mwh: float
    wind_capex_per_mw_usd: float
    solar_capex_per_mw_usd: float
    hydro_capex_per_mw_usd: float
    battery_capex_per_mwh_usd: float

    @property
    def initial_budget_dkk(self) -> float:
        return self.initial_budget_usd / self.dkk_to_usd_rate

    @property
    def wind_capex_per_mw_dkk(self) -> float:
        return self.wind_capex_per_mw_usd / self.dkk_to_usd_rate

    @property
    def solar_capex_per_mw_dkk(self) -> float:
        return self.solar_capex_per_mw_usd / self.dkk_to_usd_rate

    @property
    def hydro_capex_per_mw_dkk(self) -> float:
        return self.hydro_capex_per_mw_usd / self.dkk_to_usd_rate

    @property
    def battery_capex_per_mwh_dkk(self) -> float:
        return self.battery_capex_per_mwh_usd / self.dkk_to_usd_rate


def load_baseline_economic_config() -> BaselineEconomicConfig:
    """Read the economic constants from the current project config."""
    try:
        from config import EnhancedConfig

        cfg = EnhancedConfig()
        return BaselineEconomicConfig(
            initial_budget_usd=float(getattr(cfg, "init_budget_usd", 800_000_000.0)),
            dkk_to_usd_rate=float(getattr(cfg, "dkk_to_usd_rate", 0.145)),
            annual_risk_free_rate=float(getattr(cfg, "risk_free_rate", 0.02)),
            distribution_rate=float(getattr(cfg, "distribution_rate", 0.10)),
            target_cash_ratio=float(getattr(cfg, "target_cash_ratio", 0.15)),
            min_distribution_threshold_ratio=float(
                getattr(cfg, "min_distribution_threshold_ratio", 0.01)
            ),
            time_step_hours=float(getattr(cfg, "time_step_hours", 10.0 / 60.0)),
            owned_wind_capacity_mw=float(getattr(cfg, "owned_wind_capacity_mw", 270.0)),
            owned_solar_capacity_mw=float(getattr(cfg, "owned_solar_capacity_mw", 100.0)),
            owned_hydro_capacity_mw=float(getattr(cfg, "owned_hydro_capacity_mw", 40.0)),
            owned_battery_capacity_mwh=float(getattr(cfg, "owned_battery_capacity_mwh", 10.0)),
            wind_capex_per_mw_usd=float(getattr(cfg, "wind_capex_per_mw", 2_000_000.0)),
            solar_capex_per_mw_usd=float(getattr(cfg, "solar_capex_per_mw", 1_000_000.0)),
            hydro_capex_per_mw_usd=float(getattr(cfg, "hydro_capex_per_mw", 1_500_000.0)),
            battery_capex_per_mwh_usd=float(
                getattr(cfg, "battery_capex_per_mwh", 400_000.0)
            ),
        )
    except Exception:
        return BaselineEconomicConfig(
            initial_budget_usd=800_000_000.0,
            dkk_to_usd_rate=0.145,
            annual_risk_free_rate=0.02,
            distribution_rate=0.10,
            target_cash_ratio=0.15,
            min_distribution_threshold_ratio=0.01,
            time_step_hours=10.0 / 60.0,
            owned_wind_capacity_mw=270.0,
            owned_solar_capacity_mw=100.0,
            owned_hydro_capacity_mw=40.0,
            owned_battery_capacity_mwh=10.0,
            wind_capex_per_mw_usd=2_000_000.0,
            solar_capex_per_mw_usd=1_000_000.0,
            hydro_capex_per_mw_usd=1_500_000.0,
            battery_capex_per_mwh_usd=400_000.0,
        )


def detect_timebase_hours(data: pd.DataFrame, default_hours: float = 10.0 / 60.0) -> float:
    """Infer hours per row from a timestamp column, falling back to 10 minutes."""
    if "timestamp" not in data.columns:
        return float(default_hours)
    try:
        parsed = pd.to_datetime(data["timestamp"], errors="coerce").dropna().sort_values()
        if parsed.size < 2:
            return float(default_hours)
        deltas = parsed.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if deltas.size == 0:
            return float(default_hours)
        return float(np.median(deltas) / 3600.0)
    except Exception:
        return float(default_hours)


def periods_per_year_from_hours(timebase_hours: float) -> float:
    return float((365.25 * 24.0) / max(float(timebase_hours), 1e-12))


def infer_periods_per_year(timestamps, default_periods_per_year: float = 52560.0) -> float:
    try:
        if timestamps is None:
            return float(default_periods_per_year)
        ts = pd.Series(timestamps).dropna()
        if ts.empty:
            return float(default_periods_per_year)
        parsed = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
        if parsed.size < 2:
            return float(default_periods_per_year)
        deltas = parsed.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if deltas.size == 0:
            return float(default_periods_per_year)
        seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0
        return float(np.clip(seconds_per_year / float(np.median(deltas)), 1.0, 525960.0))
    except Exception:
        return float(default_periods_per_year)


def annual_rate_to_step_rate(annual_rate: float, periods_per_year: float) -> float:
    annual = float(annual_rate)
    periods = float(max(periods_per_year, 1.0))
    if not np.isfinite(annual) or annual <= -1.0:
        return 0.0
    return float((1.0 + annual) ** (1.0 / periods) - 1.0)


def distribute_excess_cash(cash_value: float, nav_value: float, cfg: BaselineEconomicConfig) -> Tuple[float, float]:
    """Apply the same shareholder cash-distribution rule used by Tier1."""
    cash = float(cash_value)
    nav = float(max(nav_value, 0.0))
    target_cash = nav * float(cfg.target_cash_ratio)
    excess_cash = cash - target_cash
    threshold = nav * float(cfg.min_distribution_threshold_ratio)
    if excess_cash <= threshold:
        return cash, 0.0
    distribution = excess_cash * float(cfg.distribution_rate)
    return cash - distribution, float(max(distribution, 0.0))


def _finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def compute_performance_metrics(
    nav_values: Iterable[float],
    *,
    timestamps=None,
    periods_per_year: Optional[float] = None,
    annual_risk_free_rate: float = 0.02,
    total_distributions: float = 0.0,
    distribution_adjusted_values: Optional[Iterable[float]] = None,
    value_suffix: str = "usd",
) -> Dict[str, Any]:
    """Compute metrics with distribution-adjusted wealth as the primary series."""
    reported = _finite_values(nav_values)
    if distribution_adjusted_values is None:
        primary = reported
    else:
        primary = _finite_values(distribution_adjusted_values)
    if primary.size == 0:
        return {"error": "No finite portfolio values"}

    periods = (
        float(periods_per_year)
        if periods_per_year is not None and np.isfinite(periods_per_year) and periods_per_year > 0.0
        else infer_periods_per_year(timestamps)
    )
    returns = np.diff(primary) / np.clip(primary[:-1], 1e-12, None)
    returns = returns[np.isfinite(returns)]
    step_vol = float(np.std(returns)) if returns.size > 1 else 0.0
    ann_vol = float(step_vol * math.sqrt(periods))
    step_rf = annual_rate_to_step_rate(annual_risk_free_rate, periods)
    excess = returns - step_rf
    sharpe = float((np.mean(excess) / step_vol) * math.sqrt(periods)) if step_vol > 0.0 else 0.0
    peak = np.maximum.accumulate(primary)
    drawdown = np.where(peak > 0.0, (peak - primary) / peak, 0.0)

    out: Dict[str, Any] = {
        "total_return": float(primary[-1] / primary[0] - 1.0) if primary.size > 1 else 0.0,
        "annual_return": (
            float((primary[-1] / primary[0]) ** (periods / max(primary.size - 1, 1)) - 1.0)
            if primary.size > 1 and primary[0] > 0.0 and primary[-1] > 0.0
            else 0.0
        ),
        "volatility": ann_vol,
        "step_volatility": step_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(np.max(drawdown)) if drawdown.size else 0.0,
        "initial_portfolio_value": float(primary[0]),
        "final_portfolio_value": float(primary[-1]),
        "periods_per_year": float(periods),
        "annual_risk_free_rate": float(annual_risk_free_rate),
        "per_step_risk_free_rate": float(step_rf),
        "distribution_adjusted_evaluation": True,
        f"total_distributions_{value_suffix}": float(max(total_distributions, 0.0)),
    }

    if reported.size:
        reported_returns = np.diff(reported) / np.clip(reported[:-1], 1e-12, None)
        reported_returns = reported_returns[np.isfinite(reported_returns)]
        reported_step_vol = float(np.std(reported_returns)) if reported_returns.size > 1 else 0.0
        reported_peak = np.maximum.accumulate(reported)
        reported_dd = np.where(reported_peak > 0.0, (reported_peak - reported) / reported_peak, 0.0)
        reported_excess = reported_returns - step_rf
        out.update(
            {
                "reported_nav_total_return": float(reported[-1] / reported[0] - 1.0)
                if reported.size > 1
                else 0.0,
                "reported_nav_annual_return": (
                    float((reported[-1] / reported[0]) ** (periods / max(reported.size - 1, 1)) - 1.0)
                    if reported.size > 1 and reported[0] > 0.0 and reported[-1] > 0.0
                    else 0.0
                ),
                "reported_nav_volatility": float(reported_step_vol * math.sqrt(periods)),
                "reported_nav_step_volatility": reported_step_vol,
                "reported_nav_sharpe_ratio": float(
                    (np.mean(reported_excess) / reported_step_vol) * math.sqrt(periods)
                )
                if reported_step_vol > 0.0
                else 0.0,
                "reported_nav_max_drawdown": float(np.max(reported_dd)) if reported_dd.size else 0.0,
                "reported_nav_initial_portfolio_value": float(reported[0]),
                "reported_nav_final_portfolio_value": float(reported[-1]),
            }
        )
    return out


def add_common_summary_fields(
    summary: Dict[str, Any],
    *,
    method: str,
    status: str = "completed",
) -> Dict[str, Any]:
    summary["method"] = method
    summary["status"] = status
    summary["evaluation_contract"] = "tier1_2026_distribution_adjusted"
    summary["baseline_source_folder"] = "baselines"
    return summary


class HybridFundLedger:
    """Current-codebase hybrid fund accounting used by publication baselines.

    This mirrors the Tier1 environment accounting contract:
    - fixed physical infrastructure is deployed from the 88% physical sleeve;
    - strategy decisions affect only the 12% trading sleeve;
    - operating revenue, financial MTM, transaction costs, depreciation, battery
      cash flow, and shareholder distributions are tracked separately.
    """

    def __init__(self, data: pd.DataFrame, *, seed: int = 42, timebase_hours: Optional[float] = None):
        from config import EnhancedConfig

        self.config = EnhancedConfig()
        self.config.seed = int(seed)
        self.config.enable_forecast_utilization = False
        self.config.enable_forecast_utilisation = False

        self.econ_cfg = load_baseline_economic_config()
        self.data = data.reset_index(drop=True).copy()
        self.timebase_hours = float(timebase_hours or detect_timebase_hours(self.data, self.econ_cfg.time_step_hours))
        self.periods_per_year = periods_per_year_from_hours(self.timebase_hours)
        self.dkk_to_usd_rate = float(getattr(self.config, "dkk_to_usd_rate", self.econ_cfg.dkk_to_usd_rate))

        self.price = self.data.get("price", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self.wind = self.data.get("wind", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self.solar = self.data.get("solar", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self.hydro = self.data.get("hydro", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()

        self.asset_capex = self.config.get_asset_capex(currency="DKK")
        self.initial_budget_dkk = float(getattr(self.config, "init_budget", self.econ_cfg.initial_budget_dkk))
        self.initial_budget_usd = self.initial_budget_dkk * self.dkk_to_usd_rate
        self.trading_allocation_budget = self.initial_budget_dkk * float(getattr(self.config, "financial_allocation", 0.12))
        self.budget = float(self.trading_allocation_budget)
        self.capital_allocation_fraction = float(getattr(self.config, "capital_allocation_fraction", 0.60))
        self.max_position_size = float(getattr(self.config, "max_position_size", 0.35))
        self.investment_freq = int(max(1, getattr(self.config, "investment_freq", 6)))

        self.physical_assets = {
            "wind_capacity_mw": float(getattr(self.config, "owned_wind_capacity_mw", self.econ_cfg.owned_wind_capacity_mw)),
            "solar_capacity_mw": float(getattr(self.config, "owned_solar_capacity_mw", self.econ_cfg.owned_solar_capacity_mw)),
            "hydro_capacity_mw": float(getattr(self.config, "owned_hydro_capacity_mw", self.econ_cfg.owned_hydro_capacity_mw)),
            "battery_capacity_mwh": float(getattr(self.config, "owned_battery_capacity_mwh", self.econ_cfg.owned_battery_capacity_mwh)),
        }
        self.physical_book_initial = self._physical_book_value(current_timestep=0)

        self.financial_positions = {
            "wind_instrument_value": 0.0,
            "solar_instrument_value": 0.0,
            "hydro_instrument_value": 0.0,
        }
        self.financial_mtm_positions = {
            "wind_instrument_value": 0.0,
            "solar_instrument_value": 0.0,
            "hydro_instrument_value": 0.0,
        }

        init_soc = float(np.clip(
            getattr(self.config, "battery_initial_soc", 0.5),
            getattr(self.config, "batt_soc_min", 0.1),
            getattr(self.config, "batt_soc_max", 0.9),
        ))
        self.battery_energy_mwh = init_soc * self.physical_assets["battery_capacity_mwh"]
        self.battery_discharge_power = 0.0

        self.accumulated_operational_revenue = 0.0
        self.total_distributions = 0.0
        self.cumulative_generation_revenue = 0.0
        self.cumulative_battery_revenue = 0.0
        self.cumulative_mtm_pnl = 0.0
        self.cumulative_transaction_costs = 0.0
        self.last_transaction_cost = 0.0
        self.last_mtm_pnl = 0.0
        self.last_battery_cash_delta = 0.0
        self.last_generation_revenue = 0.0
        self.last_distribution = 0.0

        self.nav_values_usd = []
        self.adjusted_nav_values_usd = []
        self.records = []

    @property
    def battery_soc(self) -> float:
        cap = max(self.physical_assets.get("battery_capacity_mwh", 0.0), 1e-12)
        return float(np.clip(self.battery_energy_mwh / cap, 0.0, 1.0))

    @property
    def current_total_exposure_dkk(self) -> float:
        return float(sum(self.financial_positions.values()))

    @property
    def current_abs_exposure_dkk(self) -> float:
        return float(sum(abs(v) for v in self.financial_positions.values()))

    def _physical_book_value(self, current_timestep: int) -> float:
        years_elapsed = float(current_timestep) / (365.25 * 24.0 * 6.0)
        dep_rate = float(getattr(self.config, "annual_depreciation_rate", 0.02))
        max_dep = float(getattr(self.config, "max_depreciation_ratio", 0.75))
        dep = min(years_elapsed * dep_rate, max_dep)
        return float(
            self.physical_assets["wind_capacity_mw"] * self.asset_capex["wind_mw"] * (1.0 - dep)
            + self.physical_assets["solar_capacity_mw"] * self.asset_capex["solar_mw"] * (1.0 - dep)
            + self.physical_assets["hydro_capacity_mw"] * self.asset_capex["hydro_mw"] * (1.0 - dep)
            + self.physical_assets["battery_capacity_mwh"] * self.asset_capex["battery_mwh"] * (1.0 - dep)
        )

    def fund_nav_dkk(self, current_timestep: int) -> float:
        from financial_engine import FinancialEngine

        return float(FinancialEngine.calculate_fund_nav(
            budget=self.budget,
            physical_assets=self.physical_assets,
            asset_capex=self.asset_capex,
            financial_mtm_values=self.financial_mtm_positions,
            accumulated_operational_revenue=self.accumulated_operational_revenue,
            current_timestep=int(current_timestep),
            config=self.config,
        ))

    def _generation_revenue(self, timestep: int) -> float:
        from financial_engine import FinancialEngine

        price = float(np.clip(self.price[timestep], -1000.0, 1e9))
        return float(FinancialEngine.calculate_generation_revenue(
            timestep=int(timestep),
            price=price,
            wind_data=self.wind,
            solar_data=self.solar,
            hydro_data=self.hydro,
            wind_scale=1.0,
            solar_scale=1.0,
            hydro_scale=1.0,
            physical_assets=self.physical_assets,
            asset_capex=self.asset_capex,
            config=self.config,
            electricity_markup=float(getattr(self.config, "electricity_markup", 1.0)),
            currency_conversion=float(getattr(self.config, "currency_conversion", 1.0)),
        ))

    def _add_mtm_for_step(self, timestep: int) -> float:
        from financial_engine import FinancialEngine

        current_price = float(np.clip(self.price[timestep], -1000.0, 1e9))
        price_return = FinancialEngine.calculate_price_returns(
            timestep=int(timestep),
            current_price=current_price,
            price_history=self.price,
        )["price_return"]
        mtm_pnl, per_asset = FinancialEngine.calculate_mtm_pnl_from_exposure(
            financial_exposures=self.financial_positions,
            price_return=price_return,
            config=self.config,
        )
        for key in self.financial_mtm_positions:
            self.financial_mtm_positions[key] = float(self.financial_mtm_positions.get(key, 0.0) + per_asset.get(key, 0.0))
        self.last_mtm_pnl = float(mtm_pnl)
        self.cumulative_mtm_pnl += float(mtm_pnl)
        return float(mtm_pnl)

    def _risk_budget_weights(self) -> np.ndarray:
        allocation = getattr(self.config, "risk_budget_allocation", None)
        if isinstance(allocation, dict):
            weights = np.array([
                allocation.get("wind", 0.40),
                allocation.get("solar", 0.35),
                allocation.get("hydro", 0.25),
            ], dtype=np.float64)
        else:
            weights = np.array([0.40, 0.35, 0.25], dtype=np.float64)
        denom = float(np.sum(np.abs(weights)))
        if denom <= 0.0:
            weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            denom = 3.0
        return weights / denom

    def _settle_mtm_on_resize(self, old_positions: Dict[str, float], target_positions: Dict[str, float]) -> None:
        for asset in ("wind", "solar", "hydro"):
            key = f"{asset}_instrument_value"
            old_exp = float(old_positions.get(key, 0.0))
            new_exp = float(target_positions.get(key, 0.0))
            old_mtm = float(self.financial_mtm_positions.get(key, 0.0))
            if abs(old_exp) <= 100.0:
                continue

            closed_fraction = 0.0
            if old_exp * new_exp < 0.0:
                closed_fraction = 1.0
            elif abs(new_exp) < abs(old_exp) and old_exp * new_exp >= 0.0:
                closed_fraction = (abs(old_exp) - abs(new_exp)) / max(abs(old_exp), 1e-9)
            if closed_fraction <= 0.0:
                continue

            realized = old_mtm * float(np.clip(closed_fraction, 0.0, 1.0))
            self.budget += realized
            self.financial_mtm_positions[key] = float(old_mtm - realized)

    def rebalance_exposure(self, target_exposure: float, timestep: int) -> float:
        """Set aggregate trading exposure in [-1, 1] of the current max position."""
        if int(timestep) == 0:
            self.last_transaction_cost = 0.0
            return 0.0

        target_exposure = float(np.clip(target_exposure, -1.0, 1.0))
        max_pos_size = float(max(self.budget * self.capital_allocation_fraction * self.max_position_size, 1.0))
        weights = self._risk_budget_weights()
        target_positions = {
            "wind_instrument_value": float(target_exposure * weights[0] * max_pos_size),
            "solar_instrument_value": float(target_exposure * weights[1] * max_pos_size),
            "hydro_instrument_value": float(target_exposure * weights[2] * max_pos_size),
        }
        old_positions = dict(self.financial_positions)
        self._settle_mtm_on_resize(old_positions, target_positions)

        total_traded_notional = float(sum(abs(target_positions[k] - old_positions.get(k, 0.0)) for k in target_positions))
        threshold = float(getattr(self.config, "no_trade_threshold", 0.0)) * max_pos_size
        if total_traded_notional <= threshold:
            self.last_transaction_cost = 0.0
            return 0.0

        txn_cost = (
            total_traded_notional * float(getattr(self.config, "transaction_cost_bps", 0.5)) / 10000.0
            + float(getattr(self.config, "transaction_fixed_cost", 0.0))
        )
        self.budget = max(0.0, self.budget - txn_cost)
        self.cumulative_transaction_costs += float(txn_cost)
        self.last_transaction_cost = float(txn_cost)
        self.financial_positions.update(target_positions)
        return total_traded_notional

    def execute_battery_action(self, action: str | int | float, timestep: int) -> Tuple[float, str]:
        from trading_engine import TradingEngine

        if isinstance(action, str):
            normalized = action.strip().lower()
            if normalized == "charge":
                bat_action = 0
            elif normalized == "discharge":
                bat_action = len(getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0])) - 1
            else:
                bat_action = len(getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0])) // 2
        else:
            bat_action = action
            levels = getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0])
            idx = int(np.clip(int(round(float(np.asarray(action).reshape(-1)[0]))), 0, len(levels) - 1))
            val = float(levels[idx])
            normalized = "charge" if val < 0.0 else "discharge" if val > 0.0 else "idle"

        price = float(np.clip(
            self.price[timestep],
            getattr(self.config, "minimum_price_filter", -1000.0),
            getattr(self.config, "maximum_price_cap", 1e9),
        ))
        cash_delta, state = TradingEngine.execute_battery_operations(
            bat_action=np.array([bat_action]),
            timestep=int(timestep),
            battery_capacity_mwh=self.physical_assets["battery_capacity_mwh"],
            battery_energy=self.battery_energy_mwh,
            price=price,
            batt_power_c_rate=float(getattr(self.config, "batt_power_c_rate", 0.5)),
            batt_eta_charge=float(getattr(self.config, "batt_eta_charge", 0.90)),
            batt_eta_discharge=float(getattr(self.config, "batt_eta_discharge", 0.90)),
            batt_soc_min=float(getattr(self.config, "batt_soc_min", 0.1)),
            batt_soc_max=float(getattr(self.config, "batt_soc_max", 0.9)),
            batt_degradation_cost=float(getattr(self.config, "battery_degradation_cost_mwh", 0.0)),
            battery_dispatch_policy_fn=None,
            config=self.config,
            use_heuristic_dispatch=False,
        )
        self.battery_energy_mwh = float(state.get("battery_energy", self.battery_energy_mwh))
        self.battery_discharge_power = float(state.get("battery_discharge_power", 0.0))
        self.last_battery_cash_delta = float(cash_delta)
        self.cumulative_battery_revenue += float(cash_delta)
        return float(cash_delta), normalized

    def step(
        self,
        timestep: int,
        *,
        target_exposure: Optional[float] = None,
        battery_action: str | int | float = "idle",
    ) -> Dict[str, Any]:
        t = int(timestep)
        self._add_mtm_for_step(t)
        traded_notional = 0.0
        if target_exposure is not None:
            traded_notional = self.rebalance_exposure(float(target_exposure), t)
        battery_cash_delta, battery_action_label = self.execute_battery_action(battery_action, t)

        generation_revenue = self._generation_revenue(t)
        battery_opex = float(getattr(self.config, "battery_opex_rate", 0.0002)) * self.physical_assets["battery_capacity_mwh"]
        operational_revenue = float(generation_revenue - battery_opex)

        self.budget = max(0.0, self.budget + battery_cash_delta)
        nav_before_current_ops = self.fund_nav_dkk(t)

        from financial_engine import FinancialEngine

        self.budget, distribution_amount = FinancialEngine.distribute_excess_cash(
            budget=self.budget,
            current_fund_nav=nav_before_current_ops,
            init_budget=self.initial_budget_dkk,
            config=self.config,
        )
        self.total_distributions += float(distribution_amount)
        self.last_distribution = float(distribution_amount)

        self.accumulated_operational_revenue += operational_revenue
        self.cumulative_generation_revenue += float(generation_revenue)
        self.last_generation_revenue = float(generation_revenue)

        nav_dkk = self.fund_nav_dkk(t)
        nav_usd = nav_dkk * self.dkk_to_usd_rate
        adjusted_usd = nav_usd + self.total_distributions * self.dkk_to_usd_rate
        physical_book = self._physical_book_value(t)
        financial_mtm = float(sum(self.financial_mtm_positions.values()))
        trading_sleeve = self.budget + financial_mtm

        self.nav_values_usd.append(nav_usd)
        self.adjusted_nav_values_usd.append(adjusted_usd)
        record = {
            "timestep": t,
            "portfolio_value": nav_dkk,
            "portfolio_value_usd": nav_usd,
            "distribution_adjusted_value": nav_dkk + self.total_distributions,
            "distribution_adjusted_value_usd": adjusted_usd,
            "cash": self.budget,
            "cash_usd": self.budget * self.dkk_to_usd_rate,
            "trading_cash_dkk": self.budget,
            "trading_cash_usd": self.budget * self.dkk_to_usd_rate,
            "physical_book_value_dkk": physical_book,
            "physical_book_value_usd": physical_book * self.dkk_to_usd_rate,
            "capacity_value": physical_book,
            "capacity_value_usd": physical_book * self.dkk_to_usd_rate,
            "financial_mtm_dkk": financial_mtm,
            "financial_mtm_usd": financial_mtm * self.dkk_to_usd_rate,
            "trading_sleeve_dkk": trading_sleeve,
            "trading_sleeve_usd": trading_sleeve * self.dkk_to_usd_rate,
            "wind_capacity": self.physical_assets["wind_capacity_mw"],
            "solar_capacity": self.physical_assets["solar_capacity_mw"],
            "hydro_capacity": self.physical_assets["hydro_capacity_mw"],
            "battery_capacity_mwh": self.physical_assets["battery_capacity_mwh"],
            "battery_soc": self.battery_soc,
            "battery_action": battery_action_label,
            "battery_revenue": battery_cash_delta,
            "operational_revenue": operational_revenue,
            "generation_revenue": generation_revenue,
            "cash_return": 0.0,
            "mtm_pnl": self.last_mtm_pnl,
            "transaction_cost": self.last_transaction_cost,
            "total_traded_notional": traded_notional,
            "target_exposure": float(target_exposure) if target_exposure is not None else np.nan,
            "current_exposure_dkk": self.current_total_exposure_dkk,
            "current_abs_exposure_dkk": self.current_abs_exposure_dkk,
            "distribution_amount": float(distribution_amount),
            "total_distributions": self.total_distributions,
            "price": float(self.price[t]),
            "wind_gen": float(self.wind[t]),
            "solar_gen": float(self.solar[t]),
            "hydro_gen": float(self.hydro[t]),
        }
        self.records.append(record)
        return record

    def performance_metrics(self) -> Dict[str, Any]:
        metrics = compute_performance_metrics(
            self.nav_values_usd,
            timestamps=self.data.get("timestamp"),
            annual_risk_free_rate=float(getattr(self.config, "risk_free_rate", self.econ_cfg.annual_risk_free_rate)),
            total_distributions=self.total_distributions * self.dkk_to_usd_rate,
            distribution_adjusted_values=self.adjusted_nav_values_usd,
            value_suffix="usd",
        )
        final_nav = float(self.nav_values_usd[-1]) if self.nav_values_usd else self.initial_budget_usd
        final_adjusted = float(self.adjusted_nav_values_usd[-1]) if self.adjusted_nav_values_usd else final_nav
        metrics.update({
            "final_value_usd": final_nav,
            "initial_value_usd": self.initial_budget_usd,
            "distribution_adjusted_final_value_usd": final_adjusted,
            "final_portfolio_value": final_adjusted,
            "initial_portfolio_value": self.initial_budget_usd,
            "total_distributions_usd": float(self.total_distributions * self.dkk_to_usd_rate),
            "hybrid_benchmark_contract": True,
            "physical_sleeve_initial_usd": float(self.physical_book_initial * self.dkk_to_usd_rate),
            "trading_sleeve_initial_usd": float(self.trading_allocation_budget * self.dkk_to_usd_rate),
            "physical_allocation": float(getattr(self.config, "physical_allocation", 0.88)),
            "financial_allocation": float(getattr(self.config, "financial_allocation", 0.12)),
            "final_trading_cash_usd": float(self.budget * self.dkk_to_usd_rate),
            "final_financial_mtm_usd": float(sum(self.financial_mtm_positions.values()) * self.dkk_to_usd_rate),
            "total_generation_revenue_usd": float(self.cumulative_generation_revenue * self.dkk_to_usd_rate),
            "total_operational_revenue_usd": float(self.accumulated_operational_revenue * self.dkk_to_usd_rate),
            "total_battery_revenue_usd": float(self.cumulative_battery_revenue * self.dkk_to_usd_rate),
            "total_mtm_pnl_usd": float(self.cumulative_mtm_pnl * self.dkk_to_usd_rate),
            "total_transaction_costs_usd": float(self.cumulative_transaction_costs * self.dkk_to_usd_rate),
            "wind_capacity_mw": float(self.physical_assets["wind_capacity_mw"]),
            "solar_capacity_mw": float(self.physical_assets["solar_capacity_mw"]),
            "hydro_capacity_mw": float(self.physical_assets["hydro_capacity_mw"]),
            "battery_capacity_mwh": float(self.physical_assets["battery_capacity_mwh"]),
            "final_battery_soc": float(self.battery_soc),
        })
        return metrics
