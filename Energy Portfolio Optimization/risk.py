#!/usr/bin/env python3
"""
Enhanced Risk Controller (fully patched)

Key guarantees:
- Stable public API used by your env:
    * update_risk_history(env_state)
    * calculate_comprehensive_risk(env_state) -> dict with 'overall_risk'
    * get_risk_metrics_for_observation() -> np.float32[6]
- Fixed-size, finite, [0,1]-clipped outputs
- Lightweight memory footprint (bounded deques)
- Robust error handling & safe fallbacks

PATCHED:
- Fixed a bug where array slicing was incorrect, causing empty arrays.
- Replaced raw divisions with a safe division utility to prevent NaNs or Inf.
"""

from typing import Dict, Any
import numpy as np
from collections import deque
import logging
import inspect
from utils import SafeDivision  # UNIFIED: Import from single source of truth
from config import (
    ADAPTIVE_SCALE_SATURATION_THRESHOLD,
    ADAPTIVE_SCALE_MIN_OFFSET,
    ADAPTIVE_SCALE_COMPRESSION_FACTOR,
    REGULATORY_RISK_BASE_STRESS_WEIGHT,
    REGULATORY_RISK_FALLBACK,
    REGULATORY_RISK_SEASON_BASE,
    REGULATORY_RISK_SEASON_AMPLITUDE,
    REGULATORY_RISK_FALLBACK_ERROR,
    RISK_FALLBACK_MARKET,
    RISK_FALLBACK_OPERATIONAL,
    RISK_FALLBACK_PORTFOLIO,
    RISK_FALLBACK_LIQUIDITY,
    RISK_FALLBACK_REGULATORY,
    RISK_FALLBACK_OVERALL,
    RISK_ACTION_MULTIPLIER_DEFAULT,
    RISK_ACTION_MAX_INVESTMENT_DEFAULT,
    RISK_ACTION_CASH_RESERVE_DEFAULT,
    RISK_ACTION_HEDGE_DEFAULT,
    RISK_ACTION_REBALANCE_DEFAULT,
    RISK_ACTION_TOLERANCE_DEFAULT,
    RISK_LOOKBACK_WINDOW_DEFAULT,
    RISK_LOOKBACK_WINDOW_MAX,
)  # UNIFIED: Import constants from config


def _clip01(x: float) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.5

def _adaptive_scale(x: float, min_val: float = 0.0, max_val: float = 1.0,
                   saturation_threshold: float = ADAPTIVE_SCALE_SATURATION_THRESHOLD) -> float:
    """Adaptive scaling to prevent saturation at boundaries."""
    try:
        x = float(x)
        if x <= min_val:
            return min_val + ADAPTIVE_SCALE_MIN_OFFSET
        elif x >= max_val:
            return max_val - ADAPTIVE_SCALE_MIN_OFFSET
        elif x >= max_val * saturation_threshold:
            # Compress values near maximum to prevent saturation
            excess = x - max_val * saturation_threshold
            range_remaining = max_val * (1 - saturation_threshold)
            compressed = max_val * saturation_threshold + range_remaining * (1 - np.exp(-excess * ADAPTIVE_SCALE_COMPRESSION_FACTOR))
            return min(compressed, max_val - ADAPTIVE_SCALE_MIN_OFFSET)
        elif x <= min_val + (max_val - min_val) * (1 - saturation_threshold):
            # Expand values near minimum
            deficit = (min_val + (max_val - min_val) * (1 - saturation_threshold)) - x
            range_available = (max_val - min_val) * (1 - saturation_threshold)
            expanded = min_val + range_available * (1 - np.exp(-deficit * ADAPTIVE_SCALE_COMPRESSION_FACTOR))
            return max(expanded, min_val + ADAPTIVE_SCALE_MIN_OFFSET)
        else:
            return x
    except Exception:
        return (min_val + max_val) / 2


class EnhancedRiskController:
    """
    Lightweight, robust risk controller with bounded history and consistent outputs.
    """

    def __init__(self, lookback_window: int = None, config=None):
        # Import config if not provided
        if config is None:
            try:
                from config import EnhancedConfig
                config = EnhancedConfig()
            except Exception:
                config = None

        # PHASE 5.10 FIX: Store config as instance variable for use in risk calculations
        self.config = config

        # Use config values with fallbacks
        if config and hasattr(config, 'risk_lookback_window'):
            default_lookback = config.risk_lookback_window
        else:
            default_lookback = RISK_LOOKBACK_WINDOW_DEFAULT

        self.lookback_window = int(max(1, min(lookback_window or default_lookback, RISK_LOOKBACK_WINDOW_MAX)))  # guard + cap
        self.logger = logging.getLogger(__name__)

        # Bounded histories to prevent memory growth
        self.price_history = deque(maxlen=self.lookback_window)
        self.generation_history = deque(maxlen=self.lookback_window)   # dicts {'wind','solar','hydro'}
        self.portfolio_history = deque(maxlen=self.lookback_window)    # dicts {'total_value','diversification'}
        self.cash_flow_history = deque(maxlen=self.lookback_window)    # floats (revenue)
        self.market_stress_history = deque(maxlen=self.lookback_window)

        # Risk weights from config (sum ~ 1.0)
        if config and hasattr(config, 'risk_weights'):
            self.risk_weights = config.risk_weights.copy()
        else:
            self.risk_weights = {
                'market': 0.25,
                'operational': 0.20,
                'portfolio': 0.25,
                'liquidity': 0.15,
                'regulatory': 0.15
            }

        # Adaptive thresholds from config
        if config and hasattr(config, 'risk_thresholds'):
            self.adaptive_thresholds = config.risk_thresholds.copy()
        else:
            self.adaptive_thresholds = {
                'market_stress_high': 0.85,      # Reduced from 1.0
                'market_stress_medium': 0.65,    # Reduced from 0.8
                'volatility_high': 0.80,         # Reduced from 1.0
                'volatility_medium': 0.50,       # Reduced from 0.6
                'portfolio_concentration_high': 0.75,  # Reduced from 0.9
                'liquidity_stress_high': 0.80,   # Reduced from 1.0
            }

        # Default metrics for cold start (6 values for obs head)
        self._default_obs_metrics = np.array([0.30, 0.20, 0.25, 0.15, 0.35, 0.25], dtype=np.float32)

    # ----------------------------- Core calculations -----------------------------

    def calculate_market_risk(self, current_price: float) -> float:
        """
        Market risk from recent price volatility + momentum.
        Returns a value in [0, 1].
        """
        try:
            if len(self.price_history) < 5:
                return 0.30

            # FIX: use a slice (trailing colon) so we get an array, not a single element
            prices = np.asarray(list(self.price_history), dtype=np.float64)

            if prices.size > 1:
                rets = np.diff(prices) / (np.asarray(prices[:-1], dtype=np.float64) + 1e-8)
                rets = rets[np.isfinite(rets)]
                vol = float(np.std(rets)) if rets.size > 0 else 0.10
            else:
                vol = 0.10

            window = min(10, len(prices))
            recent_avg = float(np.mean(prices[-min(5, window):]))
            older_avg = float(np.mean(prices[-window:-min(5, window)])) if window > 5 else recent_avg

            momentum = SafeDivision.div(abs(recent_avg - older_avg), abs(older_avg), default=0.10)

            # PHASE 5.10 FIX: Use config parameters for market risk weights
            vol_weight = getattr(self.config, 'market_risk_volatility_weight', 0.6) if self.config else 0.6
            mom_weight = getattr(self.config, 'market_risk_momentum_weight', 0.4) if self.config else 0.4
            mom_factor = getattr(self.config, 'market_risk_momentum_factor', 0.5) if self.config else 0.5
            mrisk = vol_weight * vol + mom_weight * (mom_factor * momentum)
            return _clip01(mrisk)
        except Exception as e:
            self.logger.warning(f"Market risk calculation failed: {e}")
            return 0.30

    def calculate_operational_risk(self) -> float:
        """
        Operational risk from generation volatility + recent drops.
        Returns a value in [0, 1].
        """
        try:
            if len(self.generation_history) < 3:
                return 0.20

            # FIX: use a slice so we iterate the last K dicts
            k = min(10, len(self.generation_history))
            recent = list(self.generation_history)[-k:]
            totals = [sum(d.values()) for d in recent if isinstance(d, dict)]
            if len(totals) > 1:
                mean_gen = float(np.mean(totals))
                vol = SafeDivision.div(float(np.std(totals)), abs(mean_gen), default=0.20)
                vol = min(vol, 1.0)
            else:
                vol = 0.20

            intermittency = 0.10
            if len(totals) >= 3:
                last3 = totals[-3:]
                drops = []
                for prev, curr in zip(last3[:-1], last3[1:]):
                    d = SafeDivision.div(max(0.0, (prev - curr)), abs(prev), default=0.0)
                    drops.append(d)
                if drops:
                    intermittency = min(max(drops), 0.5)

            # PHASE 5.10 FIX: Use config parameters for operational risk weights
            vol_weight = getattr(self.config, 'operational_risk_volatility_weight', 0.7) if self.config else 0.7
            int_weight = getattr(self.config, 'operational_risk_intermittency_weight', 0.3) if self.config else 0.3
            orisk = vol_weight * vol + int_weight * intermittency
            return _clip01(orisk)
        except Exception as e:
            self.logger.warning(f"Operational risk calculation failed: {e}")
            return 0.20

    def calculate_portfolio_risk(self, capacities: Dict[str, float], budget: float, initial_budget: float) -> float:
        """
        Portfolio risk from concentration + capital deployment.
        Returns a value in [0, 1].
        """
        try:
            if not capacities:
                return 0.25

            caps = [float(v) for v in capacities.values() if isinstance(v, (int, float)) and v > 0]
            if len(caps) > 1:
                total = sum(caps)
                if total > 0:
                    shares = [c / total for c in caps]
                    hhi = sum(s * s for s in shares)  # Herfindahl index
                    # Normalize HHI to [0,1] concentration risk (0 best diversified → 1 fully concentrated)
                    min_hhi = SafeDivision.div(1.0, len(shares))
                    conc = SafeDivision.div((hhi - min_hhi), (1.0 - min_hhi), default=0.5)
                else:
                    conc = 0.5
            else:
                conc = 0.8  # single-tech concentration

            if initial_budget > 0:
                deploy = SafeDivision.div(max(0.0, (initial_budget - max(0.0, float(budget)))), float(initial_budget))
                capital = min(1.0, 1.5 * deploy)
            else:
                capital = 0.50

            # PHASE 5.10 FIX: Use config parameters for portfolio risk weights
            conc_weight = getattr(self.config, 'portfolio_risk_concentration_weight', 0.6) if self.config else 0.6
            cap_weight = getattr(self.config, 'portfolio_risk_capital_weight', 0.4) if self.config else 0.4
            prisk = conc_weight * conc + cap_weight * capital
            return _adaptive_scale(prisk, 0.0, 1.0)
        except Exception as e:
            self.logger.warning(f"Portfolio risk calculation failed: {e}")
            return 0.25

    def calculate_liquidity_risk(self, budget: float, initial_budget: float) -> float:
        """
        Liquidity risk from cash buffer + cash flow volatility.
        Returns a value in [0, 1].
        """
        try:
            if initial_budget > 0:
                cash_ratio = SafeDivision.div(float(max(0.0, budget)), float(initial_budget))
                buffer_risk = max(0.0, (0.05 - cash_ratio) * 20.0)  # >0 when cash < 5% of initial
            else:
                buffer_risk = 0.50

            cf_vol = 0.10
            if len(self.cash_flow_history) > 5:
                cfa = np.asarray(list(self.cash_flow_history)[-10:], dtype=np.float64)
                cfa = cfa[np.isfinite(cfa)]
                if cfa.size > 1:
                    mu = float(np.mean(cfa))
                    cf_vol = SafeDivision.div(float(np.std(cfa)), abs(mu), default=0.10) if abs(mu) > 1e-12 else 0.10
                    cf_vol = min(cf_vol, 1.0)

            # PHASE 5.10 FIX: Use config parameters for liquidity risk weights
            buf_weight = getattr(self.config, 'liquidity_risk_buffer_weight', 0.6) if self.config else 0.6
            cf_weight = getattr(self.config, 'liquidity_risk_cashflow_weight', 0.4) if self.config else 0.4
            lrisk = buf_weight * buffer_risk + cf_weight * cf_vol
            return _adaptive_scale(lrisk, 0.0, 1.0)
        except Exception as e:
            self.logger.warning(f"Liquidity risk calculation failed: {e}")
            return 0.15

    def calculate_regulatory_risk(self, timestep: int) -> float:
        """
        Regulatory/policy risk scaled by recent market stress + mild seasonality.
        Returns a value in [0, 1].
        """
        try:
            if len(self.market_stress_history) > 0:
                avg_stress = float(np.mean(list(self.market_stress_history)[-5:]))
                base = REGULATORY_RISK_BASE_STRESS_WEIGHT * avg_stress
            else:
                base = REGULATORY_RISK_FALLBACK

            # Simple annual seasonality; 144 steps/day × 365 days
            season = REGULATORY_RISK_SEASON_BASE + REGULATORY_RISK_SEASON_AMPLITUDE * abs(np.sin(SafeDivision.div((timestep or 0) * 2 * np.pi, (365 * 144))))
            rrisk = base + season
            return _clip01(rrisk)
        except Exception as e:
            self.logger.warning(f"Regulatory risk calculation failed: {e}")
            return REGULATORY_RISK_FALLBACK_ERROR

    def calculate_forecast_uncertainty_risk(self, env_state: Dict) -> float:
        """
        ENHANCEMENT: Calculate risk from forecast uncertainty (spread of quantiles).
        Returns a value in [0, 1].
        """
        try:
            # Extract forecast quantiles if available
            price_forecasts = env_state.get('price_forecasts', [])
            if isinstance(price_forecasts, (list, tuple)) and len(price_forecasts) >= 3:
                # Assume forecasts are [q10, q50, q90] or similar
                forecasts = np.array(price_forecasts, dtype=np.float64)
                if len(forecasts) >= 3:
                    q10, q50, q90 = forecasts[0], forecasts[1], forecasts[2]
                    if q50 > 0:
                        uncertainty = (q90 - q10) / q50  # Relative spread
                        return float(np.clip(uncertainty * 0.5, 0.0, 1.0))  # Scale to [0,1]

            # Fallback: use price volatility as proxy for forecast uncertainty
            if len(self.price_history) >= 5:
                prices = np.array(list(self.price_history)[-5:], dtype=np.float64)
                volatility = np.std(prices) / max(np.mean(prices), 1.0)
                return float(np.clip(volatility * 2.0, 0.0, 1.0))

            return 0.15  # Default moderate uncertainty
        except Exception as e:
            self.logger.warning(f"Forecast uncertainty risk calculation failed: {e}")
            return 0.15

    # ----------------------------- Aggregation & API -----------------------------

    def calculate_comprehensive_risk(self, env_state: Dict) -> Dict[str, float]:
        """
        Combine risk components into a dict with 'overall_risk'.
        All outputs are finite floats in [0, 1].
        """
        try:
            price = float(env_state.get('price', 50.0))
            budget = float(env_state.get('budget', 1e7))
            initial_budget = float(env_state.get('initial_budget', 1e7))
            timestep = int(env_state.get('timestep', 0))

            capacities = {
                'wind': float(env_state.get('wind_capacity_mw', 0.0)),
                'solar': float(env_state.get('solar_capacity_mw', 0.0)),
                'hydro': float(env_state.get('hydro_capacity_mw', 0.0)),
                'battery': float(env_state.get('battery_capacity_mwh', 0.0)),
            }

            # ENHANCEMENT: Add forecast uncertainty risk
            forecast_uncertainty_risk = self.calculate_forecast_uncertainty_risk(env_state)

            comp = {
                'market_risk':      self.calculate_market_risk(price),
                'operational_risk': self.calculate_operational_risk(),
                'portfolio_risk':   self.calculate_portfolio_risk(capacities, budget, initial_budget),
                'liquidity_risk':   self.calculate_liquidity_risk(budget, initial_budget),
                'regulatory_risk':  self.calculate_regulatory_risk(timestep),
                'forecast_uncertainty_risk': forecast_uncertainty_risk,
            }

            overall = (
                self.risk_weights['market']      * comp['market_risk'] +
                self.risk_weights['operational'] * comp['operational_risk'] +
                self.risk_weights['portfolio']   * comp['portfolio_risk'] +
                self.risk_weights['liquidity']   * comp['liquidity_risk'] +
                self.risk_weights['regulatory']  * comp['regulatory_risk'] +
                0.10 * comp['forecast_uncertainty_risk']  # 10% weight for forecast uncertainty
            )

            # Apply adaptive scaling to prevent saturation
            overall = _adaptive_scale(overall, 0.0, 1.0)

            comp['overall_risk'] = _clip01(overall)

            # HIGH: Add internal storage for canonical Overall Risk score
            self._last_overall_risk = comp['overall_risk']

            # Final sanitize (finite & in range)
            for k, v in list(comp.items()):
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    comp[k] = 0.5
                else:
                    comp[k] = _clip01(v)
            return comp

        except Exception as e:
            self.logger.warning(f"Comprehensive risk calculation failed: {e}")
            return {
                'market_risk': RISK_FALLBACK_MARKET,
                'operational_risk': RISK_FALLBACK_OPERATIONAL,
                'portfolio_risk': RISK_FALLBACK_PORTFOLIO,
                'liquidity_risk': RISK_FALLBACK_LIQUIDITY,
                'regulatory_risk': RISK_FALLBACK_REGULATORY,
                'overall_risk': RISK_FALLBACK_OVERALL
            }

    def get_risk_adjusted_actions(self, risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """
        Simple policy suggestions based on current risk. Values in [0, 1] except risk_multiplier in [0.5, 2.0].
        """
        try:
            overall = float(risk_assessment.get('overall_risk', 0.5))
            market = float(risk_assessment.get('market_risk', 0.3))
            portfolio = float(risk_assessment.get('portfolio_risk', 0.25))
            liquidity = float(risk_assessment.get('liquidity_risk', 0.15))

            actions = {
                'risk_multiplier':      float(np.clip(0.5 + 1.5 * overall, 0.5, 2.0)),
                'max_single_investment': float(np.clip(0.5 - portfolio, 0.1, 0.5)),
                'cash_reserve_target':   float(np.clip(0.05 + 0.25 * liquidity, 0.05, 0.30)),
                'hedge_recommendation':  _adaptive_scale(2.0 * market, 0.0, 1.0),
                'rebalance_urgency':     _adaptive_scale(portfolio, 0.0, 1.0),
                'risk_tolerance':        _adaptive_scale(1.0 - overall, 0.1, 1.0),
            }
            return actions
        except Exception as e:
            self.logger.warning(f"Risk actions calculation failed: {e}")
            return {
                'risk_multiplier': RISK_ACTION_MULTIPLIER_DEFAULT,
                'max_single_investment': RISK_ACTION_MAX_INVESTMENT_DEFAULT,
                'cash_reserve_target': RISK_ACTION_CASH_RESERVE_DEFAULT,
                'hedge_recommendation': RISK_ACTION_HEDGE_DEFAULT,
                'rebalance_urgency': RISK_ACTION_REBALANCE_DEFAULT,
                'risk_tolerance': RISK_ACTION_TOLERANCE_DEFAULT,
            }

    def get_risk_metrics_for_observation(self, expected_size: int = 6) -> np.ndarray:
        """
        Return a fixed-length vector (default 6) of normalized risk signals for the env observation.
        Always float32, finite, clipped to [0,1], and exactly expected_size long.
        """
        try:
            n = int(expected_size) if expected_size and expected_size > 0 else 6

            if len(self.price_history) < 3:
                metrics = self._default_obs_metrics.copy()
            else:
                metrics = self._compute_obs_head_metrics()

            if not isinstance(metrics, np.ndarray):
                metrics = np.asarray(metrics, dtype=np.float32)

            # Resize/pad safely
            if metrics.shape[0] != n:
                result = np.full(n, 0.3, dtype=np.float32)
                copy_n = min(metrics.shape[0], n)
                result[:copy_n] = metrics[:copy_n]
                metrics = result

            # Sanitize
            metrics = np.nan_to_num(metrics, nan=0.3, posinf=1.0, neginf=0.0)
            metrics = np.clip(metrics, 0.0, 1.0).astype(np.float32)

            # Final guard
            if metrics.shape[0] != n:
                safe = np.full(n, 0.3, dtype=np.float32)
                safe[:min(n, 6)] = self._default_obs_metrics[:min(n, 6)]
                return safe
            return metrics
        except Exception as e:
            self.logger.warning(f"Risk metrics generation failed: {e}")
            safe = np.full(max(6, expected_size or 6), 0.3, dtype=np.float32)
            return safe[:expected_size or 6]

    # ----------------------------- History management -----------------------------

    def update_risk_history(self, env_state: Dict[str, Any]) -> None:
        """
        Update internal histories from env_state. All updates are optional and guarded.
        """
        try:
            # Price
            p = env_state.get('price', None)
            if isinstance(p, (int, float)) and np.isfinite(p):
                self.price_history.append(float(p))

            # Generation (wind/solar/hydro)
            gen = {}
            for k in ('wind', 'solar', 'hydro'):
                v = env_state.get(k, 0.0)
                gen[k] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else 0.0
            self.generation_history.append(gen)

            # Portfolio value (budget + simple mark of capacities)
            budget = env_state.get('budget', 0.0)
            wind_c = env_state.get('wind_capacity_mw', 0.0)
            solar_c = env_state.get('solar_capacity_mw', 0.0)
            hydro_c = env_state.get('hydro_capacity_mw', 0.0)

            if all(isinstance(x, (int, float)) for x in (budget, wind_c, solar_c, hydro_c)):
                total_value = float(budget) + 100.0 * float(max(0.0, wind_c)) \
                              + 100.0 * float(max(0.0, solar_c)) \
                              + 100.0 * float(max(0.0, hydro_c))
                self.portfolio_history.append({
                    'total_value': total_value,
                    'diversification': self._safe_diversification_index(wind_c, solar_c, hydro_c)
                })

            # Cash flow proxy (revenue)
            rev = env_state.get('revenue', None)
            if isinstance(rev, (int, float)) and np.isfinite(rev):
                self.cash_flow_history.append(float(rev))

            # Market stress
            stress = env_state.get('market_stress', None)
            if isinstance(stress, (int, float)) and np.isfinite(stress):
                self.market_stress_history.append(float(np.clip(stress, 0.0, 1.0)))
        except Exception as e:
            self.logger.warning(f"Risk history update failed: {e}")

    # ----------------------------- Helpers -----------------------------

    def _compute_obs_head_metrics(self) -> np.ndarray:
        """
        Compute 6-element metrics: [mkt_vol, gen_var, port_var, liq_var, stress, overall]
        """
        out = []

        # 1) Market volatility
        try:
            prices = np.asarray(list(self.price_history)[-10:], dtype=np.float64)
            if prices.size > 1:
                rets = np.diff(prices) / (np.asarray(prices[:-1], dtype=np.float64) + 1e-8)
                rets = rets[np.isfinite(rets)]
                mv = float(np.std(rets)) if rets.size > 0 else 0.30
                out.append(min(1.0, mv * 10.0))
            else:
                out.append(0.30)
        except Exception:
            out.append(0.30)

        # 2) Generation variability
        try:
            recent = list(self.generation_history)[-5:]
            totals = [sum(d.values()) for d in recent if isinstance(d, dict)]
            if len(totals) > 1:
                mu = float(np.mean(totals))
                gv = SafeDivision.div(float(np.std(totals)), abs(mu), default=0.20) if mu != 0 else 0.20
                out.append(min(1.0, gv * 5.0))
            else:
                out.append(0.20)
        except Exception:
            out.append(0.20)

        # 3) Portfolio variability (returns on total_value)
        try:
            vals = [p.get('total_value', 0.0) for p in list(self.portfolio_history)[-5:]]
            vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v) and v > 0]
            if len(vals) > 1:
                rets = np.diff(vals) / (np.asarray(vals[:-1], dtype=np.float64) + 1e-8)
                rets = rets[np.isfinite(rets)]
                pv = float(np.std(rets)) if rets.size > 0 else 0.25
                out.append(min(1.0, pv * 20.0))
            else:
                out.append(0.25)
        except Exception:
            out.append(0.25)

        # 4) Liquidity variability (cash-flow volatility)
        try:
            cf = np.asarray(list(self.cash_flow_history)[-10:], dtype=np.float64)
            cf = cf[np.isfinite(cf)]
            if cf.size > 1:
                mu = float(np.mean(cf))
                lv = SafeDivision.div(float(np.std(cf)), abs(mu), default=0.15) if abs(mu) > 1e-12 else 0.15
                out.append(min(1.0, lv))
            else:
                out.append(0.15)
        except Exception:
            out.append(0.15)

        # 5) Market stress
        try:
            if len(self.market_stress_history) > 0:
                stress = float(np.mean(list(self.market_stress_history)[-5:]))
                out.append(_clip01(stress))
            else:
                out.append(0.35)
        except Exception:
            out.append(0.35)

        # 6) Overall indicator - HIGH: Fix Metric 6 to use canonical weighted risk score
        try:
            # Use the canonical, weighted risk score stored in self._last_overall_risk
            if hasattr(self, '_last_overall_risk') and self._last_overall_risk is not None:
                overall = float(self._last_overall_risk)
            else:
                # Fallback to simple average if canonical score not available
                overall = float(np.mean(out[:5])) if len(out) >= 5 else 0.25
            out.append(_clip01(overall))
        except Exception:
            out.append(0.25)

        # Ensure length == 6
        while len(out) < 6:
            out.append(0.30)

        arr = np.asarray(out[:6], dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _safe_diversification_index(wind_c: float, solar_c: float, hydro_c: float) -> float:
        """
        Calculate Shannon entropy-based diversification index.

        FIX: Safe handling of s=0 case using np.where to prevent log(0) errors
        """
        try:
            caps = [c for c in (wind_c, solar_c, hydro_c) if isinstance(c, (int, float)) and c > 0]
            if len(caps) < 2:
                return 0.0
            total = float(sum(caps))
            shares = np.array([SafeDivision.div(c, total) for c in caps], dtype=np.float64)

            # FIX: Shannon entropy with safe handling of s=0
            # Use np.where to ensure s * log(s) = 0 when s = 0
            epsilon = 1e-10
            safe_shares = np.maximum(shares, epsilon)
            entropy_terms = np.where(shares > 0, shares * np.log(safe_shares), 0.0)
            entropy = -np.sum(entropy_terms)

            max_entropy = np.log(len(shares))
            return SafeDivision.div(entropy, max_entropy, default=0.0) if max_entropy > 0 else 0.0
        except Exception:
            return 0.0

    # ----------------------------- Introspection & cleanup -----------------------------

    def get_risk_summary(self) -> Dict[str, Any]:
        try:
            return {
                'lookback_window': self.lookback_window,
                'history_lengths': {
                    'price': len(self.price_history),
                    'generation': len(self.generation_history),
                    'portfolio': len(self.portfolio_history),
                    'cash_flow': len(self.cash_flow_history),
                    'market_stress': len(self.market_stress_history),
                },
                'risk_weights': dict(self.risk_weights),
                'obs_head_len': 6,
            }
        except Exception as e:
            return {'error': str(e)}

    def __del__(self):
        try:
            self.price_history.clear()
            self.generation_history.clear()
            self.portfolio_history.clear()
            self.cash_flow_history.clear()
            self.market_stress_history.clear()
        except Exception:
            pass


# =====================================================================
# FORECAST RISK MANAGER
# =====================================================================
# Uses forecasts for RISK MANAGEMENT instead of directional trading.
# This replaces the broken forecast alignment reward system.

class ForecastRiskManager:
    """
    Manages risk using forecast signals.

    Instead of rewarding agents for following forecast directions,
    this class uses forecasts to:
    - Reduce position sizes when uncertainty is high
    - Exit positions when forecasts change direction
    - Take profits when forecasts are extreme
    - Reward risk-adjusted performance
    - NOVEL: Only trade when forecast delta > MAPE (signal filtering)
    """

    def __init__(self, config):
        """
        Initialize forecast risk manager.

        Args:
            config: Configuration object with risk management parameters
        """
        self.config = config

        # Position scaling thresholds
        self.confidence_high = getattr(config, 'forecast_confidence_high_threshold', 0.90)
        self.confidence_medium = getattr(config, 'forecast_confidence_medium_threshold', 0.70)
        self.confidence_low_scale = getattr(config, 'forecast_confidence_low_scale', 0.5)

        # Volatility scaling thresholds
        self.volatility_high = getattr(config, 'forecast_volatility_high_threshold', 1.5)
        self.volatility_medium = getattr(config, 'forecast_volatility_medium_threshold', 1.0)

        # Exit signal parameters
        self.direction_change_exit = getattr(config, 'forecast_direction_change_exit_fraction', 0.5)
        self.direction_change_min = getattr(config, 'forecast_direction_change_min_strength', 0.5)

        # Profit-taking parameters
        self.extreme_threshold = getattr(config, 'forecast_extreme_threshold', 2.0)
        self.extreme_profit_taking = getattr(config, 'forecast_extreme_profit_taking_fraction', 0.3)

        # Logging
        self.log_decisions = getattr(config, 'log_position_scaling_decisions', False)

        # Signal gate relaxation parameters (defaults keep backwards compatibility)
        base_gate = getattr(config, 'forecast_signal_gate_multiplier', None)
        if base_gate is None:
            base_gate = getattr(config, 'signal_gate_initial_multiplier', None)
        self.signal_gate_initial = max(0.5, float(base_gate if base_gate is not None else 3.0))
        self.signal_gate_min = max(
            0.1,
            float(getattr(config, 'signal_gate_min_multiplier', getattr(config, 'signal_filter_mape_multiplier', 0.8)))
        )
        self.signal_gate_decay_start = max(0, int(getattr(config, 'signal_gate_decay_start', 720)))
        self.signal_gate_decay_duration = max(1, int(getattr(config, 'signal_gate_decay_duration', 2880)))
        self.signal_gate_decay_enabled = bool(getattr(config, 'enable_signal_gate_decay', True))
        self.signal_gate_multiplier = self.signal_gate_initial  # legacy attribute
        self._last_gate_multiplier = self.signal_gate_initial

        # State tracking
        self.prev_z_short = 0.0
        self.prev_forecast_direction = 0

        logging.info("ForecastRiskManager initialized")
        logging.info(f"  Confidence thresholds: high={self.confidence_high}, med={self.confidence_medium}")
        logging.info(f"  Volatility thresholds: high={self.volatility_high}, med={self.volatility_medium}")
        logging.info(
            f"  Signal gate: initial={self.signal_gate_initial:.2f}, min={self.signal_gate_min:.2f}, "
            f"decay_start={self.signal_gate_decay_start}, decay_duration={self.signal_gate_decay_duration}"
        )

    def _current_gate_multiplier(self, timestep: int) -> float:
        """Return the relaxed gate multiplier for the given timestep."""
        if not self.signal_gate_decay_enabled:
            self._last_gate_multiplier = self.signal_gate_initial
            return self.signal_gate_initial

        if timestep <= self.signal_gate_decay_start:
            self._last_gate_multiplier = self.signal_gate_initial
            return self.signal_gate_initial

        progress = min(1.0, (timestep - self.signal_gate_decay_start) / self.signal_gate_decay_duration)
        multiplier_range = self.signal_gate_initial - self.signal_gate_min
        current = self.signal_gate_initial - progress * multiplier_range
        lower = min(self.signal_gate_initial, self.signal_gate_min)
        upper = max(self.signal_gate_initial, self.signal_gate_min)
        current = float(np.clip(current, lower, upper))
        self._last_gate_multiplier = current
        return current

    def compute_risk_adjustments(self,
                                 z_short: float,
                                 z_medium: float,
                                 z_long: float,
                                 forecast_trust: float,
                                 position_pnl: float = 0.0,
                                 timestep: int = 0,
                                 forecast_deltas: Dict[str, float] = None,
                                 mape_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Compute risk management adjustments based on forecasts.

        Args:
            z_short: Short-horizon forecast z-score
            z_medium: Medium-horizon forecast z-score
            z_long: Long-horizon forecast z-score
            forecast_trust: Forecast confidence/trust [0, 1]
            position_pnl: Current position P&L (for profit-taking)
            timestep: Current timestep (for logging)
            forecast_deltas: Dict with actual forecast deltas (price differences)
                            {'short': delta_short, 'medium': delta_medium, 'long': delta_long}
            mape_thresholds: Dict with MAPE thresholds for each horizon
                            {'short': mape_short, 'medium': mape_medium, 'long': mape_long}

        Returns:
            Dict with risk adjustment factors:
            - confidence_scale: Position scale based on confidence [0, 1]
            - volatility_scale: Position scale based on volatility [0, 1]
            - combined_scale: Combined position scale [0, 1]
            - exit_signal: Fraction of position to exit [0, 1]
            - profit_taking: Fraction of position to take profits [0, 1]
            - risk_reward: Reward for good risk management
            - trade_signal: Whether to trade (True) or stay neutral (False)
            - signal_strength: Strength of trading signal [0, 1]
            - active_horizons: List of horizons with significant signals
        """
        # =====================================================================
        # NOVEL MULTI-HORIZON SIGNAL FILTERING
        # =====================================================================
        # Only trade when forecast delta > MAPE threshold (statistically significant)
        # This filters out noise and focuses on high-confidence opportunities

        trade_signal = False
        signal_strength = 0.0
        active_horizons = []
        gate_multiplier = self._current_gate_multiplier(timestep)

        if forecast_deltas is not None and mape_thresholds is not None:
            # Check each horizon for statistically significant signals
            horizon_signals = {}

            for horizon in ['short', 'medium', 'long']:
                delta = forecast_deltas.get(horizon, 0.0)
                mape = mape_thresholds.get(horizon, 0.02)  # Default 2% MAPE
                gate_threshold = abs(mape) * max(gate_multiplier, 0.1)

                # Signal is significant if |delta| exceeds the gated threshold
                is_significant = abs(delta) > gate_threshold

                if is_significant:
                    # Normalize signal strength by how much it exceeds the gated threshold
                    strength = abs(delta) / max(gate_threshold, 1e-6)
                    horizon_signals[horizon] = {
                        'delta': delta,
                        'strength': strength,
                        'direction': np.sign(delta)
                    }
                    active_horizons.append(horizon)

            # Aggregate signals across horizons
            if len(horizon_signals) > 0:
                trade_signal = True

                # Weighted average by horizon (short=0.5, medium=0.3, long=0.2)
                weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
                weighted_strength = 0.0

                for horizon, signal in horizon_signals.items():
                    weighted_strength += signal['strength'] * weights.get(horizon, 0.0)

                # Normalize to [0, 1] range (cap at 3.0 for extreme signals)
                signal_strength = min(weighted_strength / 3.0, 1.0)

        # 1. CONFIDENCE-BASED POSITION SCALING
        if forecast_trust > self.confidence_high:
            confidence_scale = 1.0  # High confidence → normal positions
        elif forecast_trust > self.confidence_medium:
            confidence_scale = 0.7  # Medium confidence → 70% positions
        else:
            confidence_scale = self.confidence_low_scale  # Low confidence → 50% positions

        # NOVEL: If no significant signal, reduce position to zero (stay neutral)
        if not trade_signal:
            confidence_scale *= 0.1  # 10% of normal (near-zero positions)

        # 2. VOLATILITY-BASED POSITION SCALING
        forecast_volatility = float(np.std([z_short, z_medium, z_long]))

        if forecast_volatility > self.volatility_high:
            volatility_scale = 0.5  # High volatility → 50% positions
        elif forecast_volatility > self.volatility_medium:
            volatility_scale = 0.75  # Medium volatility → 75% positions
        else:
            volatility_scale = 1.0  # Low volatility → normal positions

        # 3. DIRECTION CHANGE EXIT SIGNAL
        curr_forecast_direction = int(np.sign(z_short))

        if (self.prev_forecast_direction != 0 and
            curr_forecast_direction != self.prev_forecast_direction and
            abs(self.prev_z_short) > self.direction_change_min):
            # Forecast direction flipped → exit signal
            exit_signal = self.direction_change_exit
        else:
            exit_signal = 0.0

        # Update state
        self.prev_z_short = z_short
        self.prev_forecast_direction = curr_forecast_direction

        # 4. EXTREME FORECAST PROFIT-TAKING
        if abs(z_short) > self.extreme_threshold and position_pnl > 0:
            # Extreme forecast + profitable position → take profits
            profit_taking = self.extreme_profit_taking
        else:
            profit_taking = 0.0

        # 5. COMBINED POSITION SCALE
        combined_scale = confidence_scale * volatility_scale

        # 6. RISK MANAGEMENT REWARD
        # Reward for reducing positions when uncertainty/volatility is high
        # AND for only trading when signals are statistically significant
        risk_reward = 0.0

        # Reward for signal filtering (only trade when delta > MAPE)
        if trade_signal:
            # Reward proportional to signal strength
            risk_reward += 0.4 * signal_strength
        else:
            # Reward for staying neutral when no significant signal
            risk_reward += 0.2

        if forecast_trust < self.confidence_medium:
            # Low confidence → reward conservative behavior
            risk_reward += 0.2 * (1.0 - confidence_scale)

        if forecast_volatility > self.volatility_medium:
            # High volatility → reward conservative behavior
            risk_reward += 0.2 * (1.0 - volatility_scale)

        if exit_signal > 0:
            # Exiting on direction change → reward
            risk_reward += 0.1

        if profit_taking > 0:
            # Taking profits on extremes → reward
            risk_reward += 0.1

        # Logging
        if self.log_decisions and timestep % 1000 == 0:
            logging.info(f"[RISK_MGR] t={timestep}")
            logging.info(f"  Forecast: z_short={z_short:.3f}, z_med={z_medium:.3f}, z_long={z_long:.3f}")
            logging.info(f"  Trust: {forecast_trust:.3f}, Volatility: {forecast_volatility:.3f}")
            logging.info(f"  NOVEL SIGNAL FILTER: trade={trade_signal}, strength={signal_strength:.3f}, active_horizons={active_horizons}")
            logging.info(f"  Gate multiplier: {gate_multiplier:.2f}")
            if forecast_deltas and mape_thresholds:
                for h in ['short', 'medium', 'long']:
                    delta = forecast_deltas.get(h, 0.0)
                    mape = mape_thresholds.get(h, 0.02)
                    logging.info(f"    {h}: delta={delta:.4f}, MAPE={mape:.4f}, significant={abs(delta) > mape}")
            logging.info(f"  Scales: confidence={confidence_scale:.2f}, volatility={volatility_scale:.2f}, combined={combined_scale:.2f}")
            logging.info(f"  Signals: exit={exit_signal:.2f}, profit_taking={profit_taking:.2f}")
            logging.info(f"  Risk Reward: {risk_reward:.3f}")

        return {
            'confidence_scale': float(confidence_scale),
            'volatility_scale': float(volatility_scale),
            'combined_scale': float(combined_scale),
            'exit_signal': float(exit_signal),
            'profit_taking': float(profit_taking),
            'risk_reward': float(risk_reward),
            'forecast_volatility': float(forecast_volatility),
            # NOVEL: Signal filtering outputs
            'trade_signal': bool(trade_signal),
            'signal_strength': float(signal_strength),
            'active_horizons': active_horizons,
            'num_active_horizons': len(active_horizons),
            'signal_gate_multiplier': float(gate_multiplier),
        }

    def reset(self):
        """Reset state for new episode."""
        self.prev_z_short = 0.0
        self.prev_forecast_direction = 0

    def compute_investor_strategy(self,
                                  forecast_deltas: Dict[str, float],
                                  mape_thresholds: Dict[str, float],
                                  current_positions: Dict[str, float],
                                  forecast_trust: float) -> Dict[str, Any]:
        """
        NOVEL: Investor-specific strategy using multi-threshold approach.

        Determines whether to HEDGE, TRADE, or AGGRESSIVE_TRADE based on
        forecast signal strength relative to MAPE thresholds.

        Strategy:
        1. HEDGE (|delta| > 0.5x MAPE): Protect existing positions, small adjustments
        2. TRADE (|delta| > 1.5x MAPE): Take new positions, normal size
        3. AGGRESSIVE_TRADE (|delta| > 2.0x MAPE): Strong conviction, large positions
        4. NEUTRAL (|delta| < 0.5x MAPE): Stay out, noise only

        Multi-horizon consensus:
        - Require at least 2 horizons to agree on direction
        - Weight by horizon (short=50%, medium=30%, long=20%)
        - Check directional consensus (70% threshold)

        Args:
            forecast_deltas: Dict with forecast deltas as PERCENTAGES (e.g., 0.02 = 2%)
                            {'short': 0.015, 'medium': 0.025, 'long': 0.018}
            mape_thresholds: Dict with MAPE thresholds as PERCENTAGES (e.g., 0.02 = 2%)
                            {'short': 0.02, 'medium': 0.025, 'long': 0.03}
            current_positions: Dict with current positions {'wind': 0.5, 'solar': 0.3, ...}
            forecast_trust: Overall forecast confidence [0, 1]

        Returns:
            Dict with:
            - strategy: 'neutral', 'hedge', 'trade', or 'aggressive_trade'
            - position_scale: Recommended position scale [0, 1]
            - direction: +1 (long), -1 (short), or 0 (neutral)
            - signal_strength: Overall signal strength [0, 1]
            - consensus: Whether horizons agree on direction
            - active_horizons: List of horizons with signals
            - hedge_signal: Whether hedging is recommended
            - trade_signal: Whether trading is recommended
        """
        # Get thresholds from config
        # NOVEL ADAPTIVE THRESHOLDS: Use MAPE-based confidence to switch between strategies
        # When MAPE is LOW (good forecasts): Use MAPE-relative thresholds (aggressive trading)
        # When MAPE is HIGH (bad forecasts): Use ABSOLUTE thresholds (conservative trading)
        # This allows Tier 2 to exploit good forecasts while protecting against bad ones!

        # Calculate average MAPE across horizons
        mape_values = [mape_thresholds.get(h, 0.0) for h in ['short', 'medium', 'long'] if mape_thresholds.get(h, 0.0) > 1e-6]
        avg_mape = np.mean(mape_values) if len(mape_values) > 0 else 0.05

        # ADAPTIVE THRESHOLD LOGIC:
        # If MAPE < 3% (excellent forecasts): Use MAPE-relative with LOW multipliers (enable trading)
        # If MAPE 3-8% (good forecasts): Use MAPE-relative with MEDIUM multipliers
        # If MAPE > 8% (poor forecasts): Use ABSOLUTE thresholds (protect capital)

        if avg_mape < 0.03:  # Excellent forecasts (MAPE < 3%)
            # Use aggressive MAPE-relative thresholds
            hedge_mult = 0.3   # Very low - enable more trading
            trade_mult = 0.8   # Low - enable aggressive trading
            aggressive_mult = 1.5  # Medium - enable very aggressive trading
            threshold_mode = 'aggressive'
        elif avg_mape < 0.08:  # Good forecasts (MAPE 3-8%)
            # Use balanced MAPE-relative thresholds
            hedge_mult = 0.5   # Standard
            trade_mult = 1.5   # Standard
            aggressive_mult = 3.0  # Standard
            threshold_mode = 'balanced'
        else:  # Poor forecasts (MAPE > 8%)
            # Use ABSOLUTE thresholds (ignore MAPE, use fixed values)
            # This prevents trading on noise when forecasts are unreliable
            hedge_mult = None  # Will use absolute values below
            trade_mult = None
            aggressive_mult = None
            threshold_mode = 'absolute'

        # Absolute mode fallback thresholds (percent moves)
        abs_hedge_threshold = getattr(self.config, 'investor_absolute_hedge_threshold', 0.01)
        abs_trade_threshold = getattr(self.config, 'investor_absolute_trade_threshold', 0.03)
        abs_aggressive_threshold = getattr(self.config, 'investor_absolute_aggressive_threshold', 0.05)

        # Position scales for each strategy
        hedge_scale = getattr(self.config, 'investor_hedge_position_scale', 0.3)
        trade_scale = getattr(self.config, 'investor_trade_position_scale', 0.7)
        aggressive_scale = getattr(self.config, 'investor_aggressive_position_scale', 1.0)

        # Consensus requirements
        require_consensus = getattr(self.config, 'investor_require_consensus', True)
        min_horizons = getattr(self.config, 'investor_consensus_min_horizons', 2)
        direction_threshold = getattr(self.config, 'investor_consensus_direction_threshold', 0.7)

        # Allow single-strong-horizon overrides when trust is high
        single_horizon_override = getattr(self.config, 'investor_single_horizon_override', True)
        single_horizon_strength = getattr(self.config, 'investor_single_horizon_strength', 0.6)
        single_horizon_trust = getattr(self.config, 'investor_single_horizon_trust', 0.6)
        single_horizon_scale = getattr(self.config, 'investor_single_horizon_scale', 0.8)

        # Analyze each horizon
        horizon_analysis = {}
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}

        for horizon in ['short', 'medium', 'long']:
            delta = forecast_deltas.get(horizon, 0.0)
            mape = mape_thresholds.get(horizon, 0.02)

            # FIX #20: CRITICAL BUG - When MAPE = 0 (not enough data), IGNORE this horizon!
            # Otherwise any delta > 0 triggers aggressive_trade (since thresholds = 0)
            if mape < 1e-6:  # MAPE is effectively zero (not enough data)
                # Skip this horizon - don't include in analysis
                horizon_analysis[horizon] = {
                    'delta': delta,
                    'signal_type': 'neutral',  # Force neutral when no MAPE data
                    'strength': 0.0,
                    'direction': 0,
                    'weight': weights[horizon],
                    'skipped': True  # Mark as skipped due to insufficient data
                }
                continue

            # Compute signal strength relative to different thresholds
            # ADAPTIVE THRESHOLD COMPUTATION
            if threshold_mode == 'absolute':
                # When forecasts are unreliable fall back to fixed absolute thresholds
                abs_delta = abs(delta)
                direction = np.sign(delta)

                if abs_delta > abs_aggressive_threshold:
                    signal_type = 'aggressive_trade'
                    strength = min(abs_delta / max(abs_aggressive_threshold, 1e-6), 2.0)
                elif abs_delta > abs_trade_threshold:
                    signal_type = 'trade'
                    strength = min(abs_delta / max(abs_trade_threshold, 1e-6), 2.0)
                elif abs_delta > abs_hedge_threshold:
                    signal_type = 'hedge'
                    strength = min(abs_delta / max(abs_hedge_threshold, 1e-6), 1.5)
                else:
                    signal_type = 'neutral'
                    strength = 0.0
            else:
                # Use MAPE-relative thresholds when MAPE is low (good forecasts)
                hedge_threshold = hedge_mult * mape
                trade_threshold = trade_mult * mape
                aggressive_threshold = aggressive_mult * mape

                abs_delta = abs(delta)
                direction = np.sign(delta)

                # CRITICAL FIX: Reject extreme deltas regardless of MAPE
                # If forecast error > 50% of price, it's noise - don't trade
                if abs_delta > 0.5:  # 50% absolute cap
                    signal_type = 'neutral'
                    strength = 0.0
                # Classify signal
                elif abs_delta > aggressive_threshold:
                    signal_type = 'aggressive_trade'
                    strength = min(abs_delta / aggressive_threshold, 2.0)  # Cap at 2x
                elif abs_delta > trade_threshold:
                    signal_type = 'trade'
                    strength = min(abs_delta / trade_threshold, 2.0)  # Cap at 2x
                elif abs_delta > hedge_threshold:
                    signal_type = 'hedge'
                    strength = min(abs_delta / hedge_threshold, 2.0)  # Cap at 2x
                else:
                    signal_type = 'neutral'
                    strength = 0.0

            horizon_analysis[horizon] = {
                'delta': delta,
                'signal_type': signal_type,
                'strength': strength,
                'direction': direction,
                'weight': weights[horizon],
                'skipped': False
            }

        # Check for multi-horizon consensus
        active_horizons = [h for h, a in horizon_analysis.items() if a['signal_type'] != 'neutral']

        if len(active_horizons) == 0:
            # No signals, stay neutral
            return {
                'strategy': 'neutral',
                'position_scale': 0.1,
                'direction': 0,
                'signal_strength': 0.0,
                'consensus': False,
                'active_horizons': [],
                'hedge_signal': False,
                'trade_signal': False
            }

        # Compute weighted direction
        weighted_direction = sum(
            horizon_analysis[h]['direction'] * horizon_analysis[h]['weight']
            for h in active_horizons
        )
        total_weight = sum(horizon_analysis[h]['weight'] for h in active_horizons)
        avg_direction = weighted_direction / max(total_weight, 1e-6)

        # Check consensus
        consensus = (
            len(active_horizons) >= min_horizons and
            abs(avg_direction) >= direction_threshold
        )

        # Majority override: if at least two horizons agree on direction, treat as consensus
        pos_horizons = [h for h in active_horizons if horizon_analysis[h]['direction'] > 0]
        neg_horizons = [h for h in active_horizons if horizon_analysis[h]['direction'] < 0]
        pos_weight = sum(horizon_analysis[h]['weight'] for h in pos_horizons)
        neg_weight = sum(horizon_analysis[h]['weight'] for h in neg_horizons)

        if not consensus and (len(pos_horizons) >= 2 or len(neg_horizons) >= 2):
            consensus = True
            if len(pos_horizons) >= len(neg_horizons):
                avg_direction = pos_weight / max(total_weight, 1e-6)
            else:
                avg_direction = -neg_weight / max(total_weight, 1e-6)

        # NOVEL FEATURE: FORECAST QUALITY BOOSTING
        # When forecasts are excellent (MAPE < 3%), BOOST position sizes
        # When forecasts are poor (MAPE > 8%), REDUCE position sizes
        # This allows agent to capitalize on good forecasts while protecting against bad ones

        if avg_mape < 0.03:  # Excellent forecasts
            quality_boost = 1.3  # Increase position sizes by 30%
        elif avg_mape < 0.05:  # Good forecasts
            quality_boost = 1.1  # Increase position sizes by 10%
        elif avg_mape < 0.08:  # Decent forecasts
            quality_boost = 1.0  # No change
        else:  # Poor forecasts
            quality_boost = 0.7  # Reduce position sizes by 30%

        forced_strategy = None
        forced_position_scale = None

        if require_consensus and not consensus and single_horizon_override and len(active_horizons) >= 1:
            strongest_horizon = max(
                active_horizons,
                key=lambda h: horizon_analysis[h]['strength'] * horizon_analysis[h]['weight']
            )
            strongest_strength = horizon_analysis[strongest_horizon]['strength']

            if (
                strongest_strength >= single_horizon_strength and
                forecast_trust >= single_horizon_trust and
                horizon_analysis[strongest_horizon]['signal_type'] != 'neutral'
            ):
                # Promote strongest horizon to act as temporary consensus
                consensus = True
                avg_direction = horizon_analysis[strongest_horizon]['direction']
                forced_strategy = horizon_analysis[strongest_horizon]['signal_type']
                scale_lookup = {
                    'hedge': hedge_scale,
                    'trade': trade_scale,
                    'aggressive_trade': aggressive_scale
                }
                forced_position_scale = scale_lookup.get(forced_strategy, trade_scale) * \
                    forecast_trust * quality_boost * single_horizon_scale

        # If consensus required but not met, downgrade to hedge or neutral
        if require_consensus and not consensus:
            # Check if at least one horizon suggests hedging
            has_hedge = any(horizon_analysis[h]['signal_type'] in ['hedge', 'trade', 'aggressive_trade']
                          for h in active_horizons)
            if has_hedge:
                strategy = 'hedge'
                position_scale = hedge_scale * forecast_trust * quality_boost
            else:
                strategy = 'neutral'
                position_scale = 0.1
        else:
            # Determine overall strategy from strongest signal
            signal_types = [horizon_analysis[h]['signal_type'] for h in active_horizons]

            if forced_strategy is not None:
                strategy = forced_strategy
                position_scale = float(np.clip(forced_position_scale, 0.0, 1.0))
            elif 'aggressive_trade' in signal_types:
                strategy = 'aggressive_trade'
                position_scale = aggressive_scale * forecast_trust * quality_boost
            elif 'trade' in signal_types:
                strategy = 'trade'
                position_scale = trade_scale * forecast_trust * quality_boost
            elif 'hedge' in signal_types:
                strategy = 'hedge'
                position_scale = hedge_scale * forecast_trust * quality_boost
            else:
                strategy = 'neutral'
                position_scale = 0.1

        # Compute overall signal strength
        weighted_strength = sum(
            horizon_analysis[h]['strength'] * horizon_analysis[h]['weight']
            for h in active_horizons
        )
        signal_strength = min(weighted_strength / max(total_weight, 1e-6), 1.0)

        # Final direction
        effective_consensus = consensus or not require_consensus
        final_direction = int(np.sign(avg_direction)) if effective_consensus else 0

        result = {
            'strategy': strategy,
            'position_scale': float(np.clip(position_scale, 0.0, 1.0)),
            'direction': final_direction,
            'signal_strength': float(signal_strength),
            'consensus': bool(consensus),  # Ensure it's a Python bool, not numpy bool
            'active_horizons': active_horizons,
            'hedge_signal': bool(strategy == 'hedge'),
            'trade_signal': bool(strategy in ['trade', 'aggressive_trade']),
            'horizon_analysis': horizon_analysis,  # For debugging
            'single_horizon_override_used': bool(forced_strategy is not None)
        }

        # DIAGNOSTIC LOGGING: Track strategy decisions (sample 1% of calls)
        import random
        if random.random() < 0.01:  # Log 1% of strategy computations
            import logging
            logging.info(f"[INVESTOR_STRATEGY_ADAPTIVE] strategy={strategy} consensus={consensus} "
                       f"direction={final_direction} signal_strength={signal_strength:.3f} "
                       f"active_horizons={active_horizons} position_scale={result['position_scale']:.3f} "
                       f"avg_mape={avg_mape*100:.2f}% threshold_mode={threshold_mode} quality_boost={quality_boost:.2f} "
                       f"deltas={[(h, forecast_deltas.get(h, 0.0)) for h in ['short', 'medium', 'long']]} "
                       f"mapes={[(h, mape_thresholds.get(h, 0.0)) for h in ['short', 'medium', 'long']]}")

        return result