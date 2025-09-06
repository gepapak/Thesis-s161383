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


def _clip01(x: float) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.5

def _adaptive_scale(x: float, min_val: float = 0.0, max_val: float = 1.0,
                   saturation_threshold: float = 0.95) -> float:
    """Adaptive scaling to prevent saturation at boundaries."""
    try:
        x = float(x)
        if x <= min_val:
            return min_val + 0.01  # Small offset from minimum
        elif x >= max_val:
            return max_val - 0.01  # Small offset from maximum
        elif x >= max_val * saturation_threshold:
            # Compress values near maximum to prevent saturation
            excess = x - max_val * saturation_threshold
            range_remaining = max_val * (1 - saturation_threshold)
            compressed = max_val * saturation_threshold + range_remaining * (1 - np.exp(-excess * 5))
            return min(compressed, max_val - 0.01)
        elif x <= min_val + (max_val - min_val) * (1 - saturation_threshold):
            # Expand values near minimum
            deficit = (min_val + (max_val - min_val) * (1 - saturation_threshold)) - x
            range_available = (max_val - min_val) * (1 - saturation_threshold)
            expanded = min_val + range_available * (1 - np.exp(-deficit * 5))
            return max(expanded, min_val + 0.01)
        else:
            return x
    except Exception:
        return (min_val + max_val) / 2

class SafeDivision:
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Robust division with protection against zero-division."""
        if abs(denominator) < 1e-8:
            return default
        return numerator / denominator


class EnhancedRiskController:
    """
    Lightweight, robust risk controller with bounded history and consistent outputs.
    """

    def __init__(self, lookback_window: int = 144):
        self.lookback_window = int(max(1, min(lookback_window, 200)))  # guard + cap
        self.logger = logging.getLogger(__name__)

        # Bounded histories to prevent memory growth
        self.price_history = deque(maxlen=self.lookback_window)
        self.generation_history = deque(maxlen=self.lookback_window)   # dicts {'wind','solar','hydro'}
        self.portfolio_history = deque(maxlen=self.lookback_window)    # dicts {'total_value','diversification'}
        self.cash_flow_history = deque(maxlen=self.lookback_window)    # floats (revenue)
        self.market_stress_history = deque(maxlen=self.lookback_window)

        # Risk weights (sum ~ 1.0)
        self.risk_weights = {
            'market': 0.25,
            'operational': 0.20,
            'portfolio': 0.25,
            'liquidity': 0.15,
            'regulatory': 0.15
        }

        # Adaptive thresholds to prevent saturation
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

            momentum = SafeDivision._safe_divide(abs(recent_avg - older_avg), abs(older_avg), default=0.10)

            mrisk = 0.6 * vol + 0.4 * (0.5 * momentum)
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
                vol = SafeDivision._safe_divide(float(np.std(totals)), abs(mean_gen), default=0.20)
                vol = min(vol, 1.0)
            else:
                vol = 0.20

            intermittency = 0.10
            if len(totals) >= 3:
                last3 = totals[-3:]
                drops = []
                for prev, curr in zip(last3[:-1], last3[1:]):
                    d = SafeDivision._safe_divide(max(0.0, (prev - curr)), abs(prev), default=0.0)
                    drops.append(d)
                if drops:
                    intermittency = min(max(drops), 0.5)

            orisk = 0.7 * vol + 0.3 * intermittency
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
                    min_hhi = SafeDivision._safe_divide(1.0, len(shares))
                    conc = SafeDivision._safe_divide((hhi - min_hhi), (1.0 - min_hhi), default=0.5)
                else:
                    conc = 0.5
            else:
                conc = 0.8  # single-tech concentration

            if initial_budget > 0:
                deploy = SafeDivision._safe_divide(max(0.0, (initial_budget - max(0.0, float(budget)))), float(initial_budget))
                capital = min(1.0, 1.5 * deploy)
            else:
                capital = 0.50

            prisk = 0.6 * conc + 0.4 * capital
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
                cash_ratio = SafeDivision._safe_divide(float(max(0.0, budget)), float(initial_budget))
                buffer_risk = max(0.0, (0.05 - cash_ratio) * 20.0)  # >0 when cash < 5% of initial
            else:
                buffer_risk = 0.50

            cf_vol = 0.10
            if len(self.cash_flow_history) > 5:
                cfa = np.asarray(list(self.cash_flow_history)[-10:], dtype=np.float64)
                cfa = cfa[np.isfinite(cfa)]
                if cfa.size > 1:
                    mu = float(np.mean(cfa))
                    cf_vol = SafeDivision._safe_divide(float(np.std(cfa)), abs(mu), default=0.10) if abs(mu) > 1e-12 else 0.10
                    cf_vol = min(cf_vol, 1.0)

            lrisk = 0.6 * buffer_risk + 0.4 * cf_vol
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
                base = 0.4 * avg_stress
            else:
                base = 0.20

            # Simple annual seasonality; 144 steps/day × 365 days
            season = 0.05 + 0.05 * abs(np.sin(SafeDivision._safe_divide((timestep or 0) * 2 * np.pi, (365 * 144))))
            rrisk = base + season
            return _clip01(rrisk)
        except Exception as e:
            self.logger.warning(f"Regulatory risk calculation failed: {e}")
            return 0.35

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
                'market_risk': 0.30,
                'operational_risk': 0.20,
                'portfolio_risk': 0.25,
                'liquidity_risk': 0.15,
                'regulatory_risk': 0.35,
                'overall_risk': 0.25
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
                'risk_multiplier': 1.0,
                'max_single_investment': 0.30,
                'cash_reserve_target': 0.10,
                'hedge_recommendation': 0.50,
                'rebalance_urgency': 0.30,
                'risk_tolerance': 0.70,
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
                gv = SafeDivision._safe_divide(float(np.std(totals)), abs(mu), default=0.20) if mu != 0 else 0.20
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
                lv = SafeDivision._safe_divide(float(np.std(cf)), abs(mu), default=0.15) if abs(mu) > 1e-12 else 0.15
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

        # 6) Overall indicator
        try:
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
        try:
            caps = [c for c in (wind_c, solar_c, hydro_c) if isinstance(c, (int, float)) and c > 0]
            if len(caps) < 2:
                return 0.0
            total = float(sum(caps))
            shares = [SafeDivision._safe_divide(c, total) for c in caps]
            # Shannon entropy normalized
            entropy = 0.0
            for s in shares:
                if s > 0:
                    entropy -= s * np.log(s + 1e-8)
            max_entropy = np.log(len(shares))
            return SafeDivision._safe_divide(entropy, max_entropy, default=0.0) if max_entropy > 0 else 0.0
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