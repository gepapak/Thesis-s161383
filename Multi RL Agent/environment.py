# environment.py
# Clean, MTM-based environment with cash/equity separation and realistic frictions
# Compatible with your wrapper, analyzer, metacontroller & risk controller

from __future__ import annotations

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from collections import deque
import logging, psutil, os, gc

from risk import EnhancedRiskController


# =============================================================================
# Utilities
# =============================================================================
class SafeDivision:
    @staticmethod
    def div(n: float, d: float, default: float = 0.0) -> float:
        try:
            if d is None or abs(d) < 1e-9:
                return default
            return float(n) / float(d)
        except Exception:
            return default


class LightweightMemoryManager:
    """Very light cache/memory watchdog used by the env."""
    def __init__(self, max_memory_mb: float = 1500.0):
        self.max_memory_mb = max_memory_mb
        self._history = deque(maxlen=256)

    def current_mb(self) -> float:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def should_cleanup(self, force: bool = False) -> bool:
        cur = self.current_mb()
        self._history.append(cur)
        return force or (cur > self.max_memory_mb * 0.9)

    def cleanup_if_needed(self, force: bool = False) -> bool:
        if self.should_cleanup(force):
            gc.collect()
            return True
        return False


# =============================================================================
# Observation specs (BASE-dim only; wrapper appends forecasts)
# =============================================================================
class StabilizedObservationManager:
    def __init__(self, env: 'RenewableMultiAgentEnv'):
        self.env = env
        self.observation_specs = self._build_specs()
        self.base_spaces = self._build_spaces()

    def _build_specs(self) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        # BASE dims are fixed; wrapper will add forecast features to reach model totals
        specs["investor_0"] = {"base": 6}   # wind, solar, hydro, price_n, load, budget_n
        specs["battery_operator_0"] = {"base": 4}  # price_n, batt_energy, batt_capacity, load
        specs["risk_controller_0"] = {"base": 9}   # regime cues + positions + knobs
        specs["meta_controller_0"] = {"base": 11}  # budget, positions, price_n, risks, perf, knobs
        return specs

    def _build_spaces(self) -> Dict[str, spaces.Box]:
        sp: Dict[str, spaces.Box] = {}
        sp["investor_0"]         = spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32)
        sp["battery_operator_0"] = spaces.Box(low=0.0, high=10.0, shape=(4,), dtype=np.float32)
        sp["risk_controller_0"]  = spaces.Box(low=0.0, high=10.0, shape=(9,), dtype=np.float32)
        sp["meta_controller_0"]  = spaces.Box(low=0.0, high=10.0, shape=(11,), dtype=np.float32)
        return sp

    def obs_space(self, agent: str) -> spaces.Box:
        return self.base_spaces[agent]

    def base_dim(self, agent: str) -> int:
        return self.observation_specs[agent]["base"]


# =============================================================================
# Reward calculation (lightweight multi-objective)
# =============================================================================
class MultiObjectiveRewardCalculator:
    def __init__(self, initial_budget: float, max_history_size: int = 500):
        self.initial_budget = float(max(1.0, initial_budget))
        self.max_history_size = max_history_size
        self.reward_weights = {
            'financial': 0.35,
            'risk_management': 0.30,
            'sustainability': 0.10,
            'efficiency': 0.10,
            'diversification': 0.15,
        }
        self.performance_history = {
            'financial_scores': deque(maxlen=max_history_size),
            'risk_scores': deque(maxlen=max_history_size),
            'sustainability_scores': deque(maxlen=max_history_size),
            'efficiency_scores': deque(maxlen=max_history_size),
            'diversification_scores': deque(maxlen=max_history_size),
        }

    @staticmethod
    def _diversification_score(w: float, s: float, h: float) -> float:
        tot = max(1e-9, w + s + h)
        ww, ss, hh = w / tot, s / tot, h / tot
        hhi = ww*ww + ss*ss + hh*hh
        return float(1.0 - hhi)  # max at 1/3,1/3,1/3

    def calculate(self, env_state: Dict[str, float], financial: Dict[str, float], risk: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        # Enhanced financial score incorporating both MTM and generation revenue
        net_profit = financial.get('net_profit', 0.0)
        generation_revenue = financial.get('generation_revenue', 0.0)
        mtm_pnl = financial.get('mtm_pnl', 0.0)

        # Separate scoring for different revenue sources
        fin_total = float(np.clip(SafeDivision.div(net_profit, self.initial_budget * 0.01, 0.0), -1.0, 1.0))
        fin_generation = float(np.clip(SafeDivision.div(generation_revenue, self.initial_budget * 0.005, 0.0), 0.0, 1.0))
        fin_trading = float(np.clip(SafeDivision.div(mtm_pnl, self.initial_budget * 0.01, 0.0), -1.0, 1.0))

        # Combined financial score with emphasis on generation revenue (PPA economics)
        fin = 0.5 * fin_total + 0.3 * fin_generation + 0.2 * fin_trading

        # Risk: comprehensive risk weighting using multiple risk components
        overall_risk = float(np.clip(risk.get('overall_risk', 0.5), 0.0, 1.0))
        market_risk = float(np.clip(risk.get('market_risk', 0.3), 0.0, 1.0))
        portfolio_risk = float(np.clip(risk.get('portfolio_risk', 0.25), 0.0, 1.0))
        liquidity_risk = float(np.clip(risk.get('liquidity_risk', 0.15), 0.0, 1.0))

        # Weighted risk score (1 - risk for reward maximization)
        risk_score = float(1.0 - (0.4 * overall_risk + 0.25 * market_risk + 0.25 * portfolio_risk + 0.1 * liquidity_risk))

        # Enhanced sustainability score based on renewable capacity deployment
        tot_pos = float(max(0.0, env_state.get('wind_pos', 0.0) + env_state.get('solar_pos', 0.0) + env_state.get('hydro_pos', 0.0)))
        capacity_deployment = float(np.clip(SafeDivision.div(tot_pos, self.initial_budget, 0.0), 0.0, 1.0))

        # Generation performance bonus (reward actual power delivery)
        wind_gen = financial.get('wind_generation', 0.0)
        solar_gen = financial.get('solar_generation', 0.0)
        hydro_gen = financial.get('hydro_generation', 0.0)
        total_generation = wind_gen + solar_gen + hydro_gen
        generation_performance = float(np.clip(total_generation / max(1.0, tot_pos * 1e-6), 0.0, 1.0))

        # Combined sustainability score
        sus = 0.6 * capacity_deployment + 0.4 * generation_performance

        # Enhanced efficiency incorporating generation efficiency
        eff = float(np.clip(financial.get('efficiency', 0.0), 0.0, 1.0))

        # Diversification with technology-specific weighting
        div = self._diversification_score(env_state.get('wind_pos',0.0), env_state.get('solar_pos',0.0), env_state.get('hydro_pos',0.0))

        # Adjusted reward weights to emphasize generation performance
        adjusted_weights = {
            'financial': 0.40,      # Increased emphasis on financial returns
            'risk_management': 0.25, # Reduced slightly
            'sustainability': 0.20,  # Increased for renewable deployment
            'efficiency': 0.10,      # Maintained
            'diversification': 0.05, # Reduced slightly
        }

        total = (adjusted_weights['financial']      * fin +
                 adjusted_weights['risk_management']* risk_score +
                 adjusted_weights['sustainability'] * sus +
                 adjusted_weights['efficiency']     * eff +
                 adjusted_weights['diversification']* div)

        # track enhanced metrics
        self.performance_history['financial_scores'].append(fin)
        self.performance_history['risk_scores'].append(risk_score)
        self.performance_history['sustainability_scores'].append(sus)
        self.performance_history['efficiency_scores'].append(eff)
        self.performance_history['diversification_scores'].append(div)

        return float(total), {
            'total': float(total),
            'financial': fin,
            'financial_generation': fin_generation,
            'financial_trading': fin_trading,
            'risk_management': risk_score,
            'sustainability': sus,
            'capacity_deployment': capacity_deployment,
            'generation_performance': generation_performance,
            'efficiency': eff,
            'diversification': div,
        }


# =============================================================================
# Environment
# =============================================================================
class RenewableMultiAgentEnv(ParallelEnv):
    """Multi-agent renewable investment environment (BASE obs only).

    Key corrections:
      - Cash (self.budget) vs Equity (cash + marked-to-market positions)
      - Positions (wind/solar/hydro) are currency notionals
      - MTM P&L from price returns; *no* payout × price multiplication
      - Transaction costs on turnover; battery opex; optional generation revenue (disabled)
    """
    metadata = {"name": "renewable_multi_agent:MTM-v1"}

    # ----- meta/risk knob ranges used by meta controller -----
    META_FREQ_MIN = 6        # every hour if 10-min data
    META_FREQ_MAX = 288      # daily
    META_CAP_MIN  = 0.02
    META_CAP_MAX  = 0.50
    SAT_EPS       = 1e-3

    def __init__(
        self,
        data: pd.DataFrame,
        forecast_generator: Optional[Any] = None,
        dl_adapter: Optional[Any] = None,
        investment_freq: int = 12,
        enhanced_risk_controller: bool = True,
        init_budget: float = 1e7,
        max_memory_mb: float = 1500.0,
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.max_steps = int(len(self.data))
        self.forecast_generator = forecast_generator
        self.dl_adapter = dl_adapter
        self.investment_freq = max(1, int(investment_freq))
        self.init_budget = float(init_budget)

        # vectorized series
        self._wind  = self.data.get('wind',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._solar = self.data.get('solar', pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._hydro = self.data.get('hydro', pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._price = self.data.get('price', pd.Series(50.0, index=self.data.index)).astype(float).to_numpy()
        self._load  = self.data.get('load',  pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self._riskS = self.data.get('risk',  pd.Series(0.3, index=self.data.index)).astype(float).to_numpy()

        # normalization scales (95th)
        def p95(x):
            try:
                return float(np.nanpercentile(np.asarray(x, dtype=float), 95)) or 1.0
            except Exception:
                return 1.0
        self.wind_scale  = p95(self._wind)
        self.solar_scale = p95(self._solar)
        self.hydro_scale = p95(self._hydro)

        # agents
        self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]
        self.agents = self.possible_agents[:]

        # observation manager & spaces (BASE only here)
        self.obs_manager = StabilizedObservationManager(self)
        self.observation_spaces = {a: self.obs_manager.obs_space(a) for a in self.possible_agents}

        # action spaces
        self.action_spaces = {
            "investor_0":         spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "battery_operator_0": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "risk_controller_0":  spaces.Box(low=0.0,  high=2.0, shape=(1,), dtype=np.float32),
            "meta_controller_0":  spaces.Box(low=0.0,  high=1.0, shape=(2,), dtype=np.float32),
        }

        # reward calc & risk controller
        self.reward_calculator = MultiObjectiveRewardCalculator(initial_budget=self.init_budget)
        try:
            self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144) if enhanced_risk_controller else None
        except Exception:
            # fallback if ctor signature differs
            self.enhanced_risk_controller = EnhancedRiskController() if enhanced_risk_controller else None

        # memory manager
        self.memory_manager = LightweightMemoryManager(max_memory_mb=max_memory_mb)

        # buffers
        self._obs_buf: Dict[str, np.ndarray] = {a: np.zeros(self.observation_spaces[a].shape, np.float32) for a in self.possible_agents}
        self._rew_buf: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self._done_buf: Dict[str, bool] = {a: False for a in self.possible_agents}
        self._trunc_buf: Dict[str, bool] = {a: False for a in self.possible_agents}
        self._info_buf: Dict[str, Dict[str, Any]] = {a: {} for a in self.possible_agents}

        # histories
        self.performance_history = {
            'revenue_history': deque(maxlen=512),
            'payout_efficiency': deque(maxlen=256),
        }

        # runtime vars
        self.t = 0
        self._last_saturation_count = 0
        self._clip_counts = {"investor": 0, "battery": 0, "risk": 0, "meta": 0}

        # regime snapshots (wrapper reads these)
        self.market_volatility = 0.0
        self.market_stress = 0.5
        self.overall_risk_snapshot = 0.5
        self.market_risk_snapshot = 0.5

        # meta knobs
        self.capital_allocation_fraction = 0.10

        # tracked finance state (cash/positions)
        self._reset_portfolio_state()

    # ------------------------------------------------------------------
    # SB3/wrapper expect callable per-agent space getters
    # ------------------------------------------------------------------
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    # Wrapper helper for base dims
    def _get_base_observation_dim(self, agent: str) -> int:
        return self.obs_manager.base_dim(agent)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.t = 0
        self._reset_portfolio_state()
        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        self._fill_obs()
        return self._obs_buf, {}

    def step(self, actions: Dict[str, Any]):
        if self.t >= self.max_steps:
            return self._terminal_step()

        try:
            i = self.t
            acts = self._validate_actions(actions)

            # meta & risk knobs
            self._apply_risk_control(acts['risk_controller_0'])
            self._apply_meta_control(acts['meta_controller_0'])

            # investor + battery ops (battery returns realized cash delta)
            trade_amount = self._execute_investor_trades(acts['investor_0'])
            battery_cash_delta = self._execute_battery_ops(acts['battery_operator_0'], i)

            # finance update (MTM, costs, realized rev incl. battery cash)
            financial = self._update_finance(i, trade_amount, battery_cash_delta)

            # regime updates & rewards
            self._update_market_conditions(i)
            self._update_risk_snapshots(i)
            self._assign_rewards(financial)

            # step forward
            self.t += 1
            self._fill_obs()
            self._populate_info(i, financial, acts)

            return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

        except Exception as e:
            logging.error(f"step error at t={self.t}: {e}")
            return self._safe_step()

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _validate_actions(self, actions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for agent in self.possible_agents:
            space = self.action_spaces[agent]
            a = actions.get(agent, None)
            if a is None:
                mid = (space.low + space.high) / 2.0
                out[agent] = np.array(mid, dtype=np.float32).reshape(space.shape)
                continue
            arr = np.array(a, dtype=np.float32).flatten()
            need = int(np.prod(space.shape))
            if arr.size != need:
                if arr.size < need:
                    mid = (space.low + space.high) / 2.0
                    pad = np.array(mid, dtype=np.float32).flatten()[0]
                    arr = np.concatenate([arr, np.full(need - arr.size, pad, np.float32)])
                else:
                    arr = arr[:need]
            arr = np.nan_to_num(arr, nan=0.0)
            arr = np.minimum(np.maximum(arr, space.low), space.high).astype(np.float32)
            out[agent] = arr.reshape(space.shape)
        # clip counts for wrapper diagnostics (0/1 flags)
        self._clip_counts = {
            "investor": int(np.any((out['investor_0'] < self.action_spaces['investor_0'].low) | (out['investor_0'] > self.action_spaces['investor_0'].high))),
            "battery":  int(np.any((out['battery_operator_0'] < self.action_spaces['battery_operator_0'].low) | (out['battery_operator_0'] > self.action_spaces['battery_operator_0'].high))),
            "risk":     int(np.any((out['risk_controller_0'] < self.action_spaces['risk_controller_0'].low) | (out['risk_controller_0'] > self.action_spaces['risk_controller_0'].high))),
            "meta":     int(np.any((out['meta_controller_0'] < self.action_spaces['meta_controller_0'].low) | (out['meta_controller_0'] > self.action_spaces['meta_controller_0'].high))),
        }
        return out

    def _apply_risk_control(self, risk_action: np.ndarray):
        # map 0..2 -> 0.5..2.0
        val = float(np.clip(risk_action.reshape(-1)[0], 0.0, 2.0))
        self.risk_multiplier = 0.5 + 0.75 * val  # 0.5..2.0

    def _apply_meta_control(self, meta_action: np.ndarray):
        a0, a1 = np.array(meta_action, dtype=np.float32).reshape(-1)[:2]
        # map a0 in [0,1] -> [META_CAP_MIN, META_CAP_MAX]
        cap = self.META_CAP_MIN + float(np.clip(a0, 0.0, 1.0)) * (self.META_CAP_MAX - self.META_CAP_MIN)
        self.capital_allocation_fraction = float(np.clip(cap, self.META_CAP_MIN, self.META_CAP_MAX))
        # map a1 in [0,1] to freq range
        freq = int(round(self.META_FREQ_MIN + float(np.clip(a1, 0.0, 1.0)) * (self.META_FREQ_MAX - self.META_FREQ_MIN)))
        self.investment_freq = int(np.clip(freq, self.META_FREQ_MIN, self.META_FREQ_MAX))

    def _execute_investor_trades(self, inv_action: np.ndarray) -> float:
        """Turn investor action into target position values and trade towards them.
        Returns total traded notional (for txn costs).
        """
        # Enforce investment frequency gating
        if self.t % self.investment_freq != 0:
            return 0.0  # No trading allowed this step

        # DL Overlay Integration: Use DL adapter if available
        if self.dl_adapter is not None:
            try:
                # Get DL-optimized weights
                dl_weights = self.dl_adapter.infer_weights(self.t)
                w_w, w_s, w_h = float(dl_weights[0]), float(dl_weights[1]), float(dl_weights[2])

                # Trigger online learning
                self.dl_adapter.maybe_learn(self.t)

                # Optional: blend with RL action (can be disabled by setting blend_factor=0)
                blend_factor = 0.1  # 10% RL, 90% DL
                if blend_factor > 0:
                    # convert RL action to weights in [0,1]
                    rl_a = (np.array(inv_action, dtype=np.float32).reshape(-1)[:3] + 1.0) / 2.0
                    rl_w_w, rl_w_s, rl_w_h = [float(np.clip(x, 0.0, 1.0)) for x in rl_a]
                    rl_w_sum = max(1e-6, rl_w_w + rl_w_s + rl_w_h)
                    rl_w_w, rl_w_s, rl_w_h = rl_w_w / rl_w_sum, rl_w_s / rl_w_sum, rl_w_h / rl_w_sum

                    # Blend DL and RL weights
                    w_w = (1 - blend_factor) * w_w + blend_factor * rl_w_w
                    w_s = (1 - blend_factor) * w_s + blend_factor * rl_w_s
                    w_h = (1 - blend_factor) * w_h + blend_factor * rl_w_h

            except Exception as e:
                # Fallback to RL action if DL fails
                logging.warning(f"DL overlay failed at step {self.t}: {e}")
                a = (np.array(inv_action, dtype=np.float32).reshape(-1)[:3] + 1.0) / 2.0
                w_w, w_s, w_h = [float(np.clip(x, 0.0, 1.0)) for x in a]
                w_sum = max(1e-6, w_w + w_s + w_h)
                w_w, w_s, w_h = w_w / w_sum, w_s / w_sum, w_h / w_sum
        else:
            # Standard RL action processing (no DL overlay)
            a = (np.array(inv_action, dtype=np.float32).reshape(-1)[:3] + 1.0) / 2.0
            w_w, w_s, w_h = [float(np.clip(x, 0.0, 1.0)) for x in a]
            w_sum = max(1e-6, w_w + w_s + w_h)
            w_w, w_s, w_h = w_w / w_sum, w_s / w_sum, w_h / w_sum

        # Investment fund economics: Use fixed investment capital, not growing equity
        # This prevents the exponential feedback loop
        available_capital = self.investment_capital + self.budget  # Fixed capital + retained cash

        # Apply leverage limits (max 1.5x initial capital)
        max_investment = self.init_budget * self.max_leverage
        available_capital = min(available_capital, max_investment)

        # leverage/size dampened by risk multiplier (intended: higher risk -> smaller size)
        size_cap = self.capital_allocation_fraction * SafeDivision.div(1.0, max(0.5, self.risk_multiplier), 1.0)
        size_cap = float(np.clip(size_cap, 0.05, 1.0))  # cap gross at 0.05..1.0 of available capital
        target_gross = available_capital * size_cap

        # current positions
        cur_w, cur_s, cur_h = self.wind_instrument_value, self.solar_instrument_value, self.hydro_instrument_value
        tgt_w, tgt_s, tgt_h = target_gross * w_w, target_gross * w_s, target_gross * w_h

        # deltas (positive = buy, negative = sell)
        d_w, d_s, d_h = tgt_w - cur_w, tgt_s - cur_s, tgt_h - cur_h

        # selling provides cash, buying consumes it; enforce cash >= 0
        buy_need  = sum(max(0.0, x) for x in (d_w, d_s, d_h))
        sell_gain = sum(max(0.0, -x) for x in (d_w, d_s, d_h))
        cash_after_sells = self.budget + sell_gain
        scale = 1.0
        if buy_need > cash_after_sells + 1e-8:
            scale = float((cash_after_sells) / buy_need) if buy_need > 0 else 1.0
        # apply scaled trades
        d_w *= scale; d_s *= scale; d_h *= scale
        # update positions and cash
        self.wind_instrument_value  = max(0.0, cur_w + d_w)
        self.solar_instrument_value = max(0.0, cur_s + d_s)
        self.hydro_instrument_value = max(0.0, cur_h + d_h)
        self.budget = max(0.0, self.budget - max(0.0, d_w) - max(0.0, d_s) - max(0.0, d_h) + max(0.0, -d_w) + max(0.0, -d_s) + max(0.0, -d_h))

        # expose capacity proxies for the wrapper (keeps interface stable)
        self.wind_capacity  = self.wind_instrument_value
        self.solar_capacity = self.solar_instrument_value
        self.hydro_capacity = self.hydro_instrument_value

        return abs(d_w) + abs(d_s) + abs(d_h)

    def _execute_battery_ops(self, bat_action: np.ndarray, i: int) -> float:
        """Return battery cash delta for proper reward accounting."""
        try:
            u = float(np.clip(bat_action.reshape(-1)[0], -1.0, 1.0))
            step_h = 10.0 / 60.0
            battery_cash_delta = 0.0
            
            # init a small battery on-the-fly
            if self.battery_capacity <= 0.0 and self.budget > self.init_budget * 0.01:
                capex_per_mwh = 400.0
                invest = min(self.init_budget * 0.01, self.budget * 0.1)
                self.battery_capacity = invest / capex_per_mwh
                self.budget -= invest
            # operate
            price = float(np.clip(self._price[i], 0.0, 1e6))
            if u > 0.5:
                # discharge
                rate = (u - 0.5) * 2.0
                discharge = min(self.battery_energy, self.battery_capacity * rate * 0.1)
                self.battery_energy -= discharge
                # realized cash revenue (returned to finance update; do NOT mutate budget here)
                battery_cash_delta = discharge * price * step_h
                self.battery_discharge_power = discharge
            else:
                # charge
                rate = (0.5 - u) * 2.0
                charge = min(self.battery_capacity - self.battery_energy, self.battery_capacity * rate * 0.1)
                self.battery_energy += charge
                self.battery_discharge_power = 0.0
            # clamp
            self.battery_energy = float(np.clip(self.battery_energy, 0.0, self.battery_capacity))
            self.battery_capacity = float(np.clip(self.battery_capacity, 0.0, 1e6))
            
            return float(battery_cash_delta)
        except Exception as e:
            logging.error(f"battery ops: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Finance & rewards
    # ------------------------------------------------------------------
    def _update_finance(self, i: int, trade_amount: float, battery_cash_delta: float) -> Dict[str, float]:
        # price return for this step
        price_t = float(np.clip(self._price[i], 0.0, 1e9))
        price_tm1 = float(np.clip(self._price[i-1] if i > 0 else self._price[i], 0.0, 1e9))
        price_ret = 0.0 if price_tm1 <= 0 else (price_t - price_tm1) / price_tm1

        # Correct PPA model: All technologies respond identically to price changes
        # since they all sell electricity at the same market price
        pos_before = self.wind_instrument_value + self.solar_instrument_value + self.hydro_instrument_value
        pnl_mtm = pos_before * price_ret

        # Update instrument values with uniform price sensitivity (PPA contract values)
        # All renewable technologies sell at the same market price
        self.wind_instrument_value  = max(0.0, self.wind_instrument_value * (1.0 + price_ret))
        self.solar_instrument_value = max(0.0, self.solar_instrument_value * (1.0 + price_ret))
        self.hydro_instrument_value = max(0.0, self.hydro_instrument_value * (1.0 + price_ret))

        # Physical revenue generation based on actual power delivery (PPA model)
        revenue_generation = self._calculate_generation_revenue(i, price_t)

        # costs: transaction, battery opex
        txn_rate = 0.0005
        txn_costs = float(txn_rate * float(trade_amount))
        opex_battery = 0.0002 * self.battery_capacity

        # realized cash flow (battery + physical generation revenue) - costs
        realized = float(battery_cash_delta + revenue_generation - txn_costs - opex_battery)

        # Investment fund economics: Separate profits from investment capital
        if realized > 0:
            # Positive returns: 70% distributed to investors, 30% retained for operations
            distributed_amount = realized * 0.7
            retained_amount = realized * 0.3

            self.distributed_profits += distributed_amount
            self.cumulative_returns += realized
            self.budget = max(0.0, self.budget + retained_amount)
        else:
            # Losses come from budget
            self.budget = max(0.0, self.budget + realized)
            self.cumulative_returns += realized

        # equity after MTM (for position valuation only)
        self.equity = float(self.budget + self.wind_instrument_value + self.solar_instrument_value + self.hydro_instrument_value)

        # store last revenue for wrapper (now includes generation revenue)
        self.last_revenue = float(realized)

        # Enhanced efficiency metrics including generation performance
        pos_efficiency = 1.0 if pos_before > 0 else 0.0
        generation_efficiency = self._calculate_generation_efficiency(i)
        batt_eff = 1.0 if self.battery_capacity > 0 else 0.0

        # reward's financial should reflect realized + MTM
        fin = {
            'revenue': realized,
            'generation_revenue': revenue_generation,
            'mtm_pnl': pnl_mtm,
            'net_profit': realized + pnl_mtm,
            'efficiency': 0.4 * pos_efficiency + 0.4 * generation_efficiency + 0.2 * batt_eff,
            'portfolio_value': self.equity,
            'battery_cash_delta': float(battery_cash_delta),
            'wind_generation': self._get_wind_generation(i),
            'solar_generation': self._get_solar_generation(i),
            'hydro_generation': self._get_hydro_generation(i),
            # Investment fund metrics
            'distributed_profits': self.distributed_profits,
            'cumulative_returns': self.cumulative_returns,
            'investment_capital': self.investment_capital,
            'fund_performance': self.cumulative_returns / self.init_budget if self.init_budget > 0 else 0.0,
        }
        # history for regime calc
        self.performance_history['revenue_history'].append(realized)
        return fin

    def _calculate_generation_revenue(self, i: int, price: float) -> float:
        """
        Calculate revenue from actual power generation based on PPA contracts.
        Revenue = Capacity * Capacity Factor * Price * Time Step * Scaling Factor
        """
        try:
            # Time step factor (10-minute intervals = 1/6 hour)
            time_step_hours = 10.0 / 60.0

            # Conservative revenue scaling factor for realistic economics
            # This represents a much smaller capacity per dollar to prevent exponential growth
            revenue_scale = 1e-9  # $1M instrument value = 0.001 MW equivalent capacity

            # Get normalized capacity factors (0-1 range)
            wind_cf = self._get_wind_capacity_factor(i)
            solar_cf = self._get_solar_capacity_factor(i)
            hydro_cf = self._get_hydro_capacity_factor(i)

            # Calculate generation revenue for each technology
            # Revenue = Instrument Value * Revenue Scale * Capacity Factor * Price * Time
            wind_revenue = self.wind_instrument_value * revenue_scale * wind_cf * price * time_step_hours
            solar_revenue = self.solar_instrument_value * revenue_scale * solar_cf * price * time_step_hours
            hydro_revenue = self.hydro_instrument_value * revenue_scale * hydro_cf * price * time_step_hours

            total_revenue = wind_revenue + solar_revenue + hydro_revenue

            # Apply conservative scaling to ensure economic balance
            # This prevents revenue from overwhelming MTM returns while keeping it meaningful
            balanced_revenue = total_revenue * 0.01  # 1% of theoretical full revenue (much more conservative)

            return float(np.clip(balanced_revenue, 0.0, self.init_budget * 0.0001))  # Cap at 0.01% of initial budget per step

        except Exception as e:
            logging.warning(f"Generation revenue calculation failed at step {i}: {e}")
            return 0.0

    def _get_wind_capacity_factor(self, i: int) -> float:
        """Get normalized wind capacity factor (0-1)"""
        try:
            raw_wind = float(self._wind[i]) if i < len(self._wind) else 0.0
            return float(np.clip(raw_wind / max(self.wind_scale, 1e-6), 0.0, 1.0))
        except Exception:
            return 0.0

    def _get_solar_capacity_factor(self, i: int) -> float:
        """Get normalized solar capacity factor (0-1)"""
        try:
            raw_solar = float(self._solar[i]) if i < len(self._solar) else 0.0
            return float(np.clip(raw_solar / max(self.solar_scale, 1e-6), 0.0, 1.0))
        except Exception:
            return 0.0

    def _get_hydro_capacity_factor(self, i: int) -> float:
        """Get normalized hydro capacity factor (0-1)"""
        try:
            raw_hydro = float(self._hydro[i]) if i < len(self._hydro) else 0.0
            return float(np.clip(raw_hydro / max(self.hydro_scale, 1e-6), 0.0, 1.0))
        except Exception:
            return 0.0

    def _get_wind_generation(self, i: int) -> float:
        """Get actual wind generation in MW (using conservative scaling)"""
        try:
            cf = self._get_wind_capacity_factor(i)
            capacity_mw = self.wind_instrument_value * 1e-9  # Convert $ to MW (conservative scaling)
            return float(capacity_mw * cf)
        except Exception:
            return 0.0

    def _get_solar_generation(self, i: int) -> float:
        """Get actual solar generation in MW (using conservative scaling)"""
        try:
            cf = self._get_solar_capacity_factor(i)
            capacity_mw = self.solar_instrument_value * 1e-9  # Convert $ to MW (conservative scaling)
            return float(capacity_mw * cf)
        except Exception:
            return 0.0

    def _get_hydro_generation(self, i: int) -> float:
        """Get actual hydro generation in MW (using conservative scaling)"""
        try:
            cf = self._get_hydro_capacity_factor(i)
            capacity_mw = self.hydro_instrument_value * 1e-9  # Convert $ to MW (conservative scaling)
            return float(capacity_mw * cf)
        except Exception:
            return 0.0

    def _calculate_generation_efficiency(self, i: int) -> float:
        """
        Calculate generation efficiency based on capacity factor utilization.
        Higher efficiency when capacity factors are well-utilized across the portfolio.
        """
        try:
            wind_cf = self._get_wind_capacity_factor(i)
            solar_cf = self._get_solar_capacity_factor(i)
            hydro_cf = self._get_hydro_capacity_factor(i)

            # Weight by instrument values to get portfolio-weighted efficiency
            total_value = self.wind_instrument_value + self.solar_instrument_value + self.hydro_instrument_value
            if total_value <= 0:
                return 0.0

            wind_weight = self.wind_instrument_value / total_value
            solar_weight = self.solar_instrument_value / total_value
            hydro_weight = self.hydro_instrument_value / total_value

            # Portfolio-weighted capacity factor
            portfolio_cf = wind_weight * wind_cf + solar_weight * solar_cf + hydro_weight * hydro_cf

            # Efficiency bonus for diversification (higher when not concentrated in one technology)
            diversification_bonus = 1.0 - (wind_weight**2 + solar_weight**2 + hydro_weight**2)

            # Combined efficiency score
            efficiency = portfolio_cf * (1.0 + 0.2 * diversification_bonus)

            return float(np.clip(efficiency, 0.0, 1.0))

        except Exception:
            return 0.5  # Default moderate efficiency

    def _update_market_conditions(self, i: int):
        try:
            # simple realized revenue vol proxy
            hist = list(self.performance_history['revenue_history'])
            if len(hist) > 4:
                r = np.diff(np.asarray(hist, dtype=float))
                self.market_volatility = float(np.clip(np.std(r) / (np.mean(np.abs(hist)) + 1e-8), 0.0, 1.0))
            self.market_stress = float(np.clip(self._riskS[i], 0.0, 1.0))
        except Exception:
            self.market_volatility = 0.2
            self.market_stress = 0.5

    def _update_risk_snapshots(self, i: int):
        try:
            if self.enhanced_risk_controller is None:
                return

            # Preferred: comprehensive API if present
            if hasattr(self.enhanced_risk_controller, "update_risk_history") and \
               hasattr(self.enhanced_risk_controller, "calculate_comprehensive_risk"):
                env_state = {
                    'price': float(self._price[i]),
                    'budget': self.budget,
                    'initial_budget': self.init_budget,
                    'timestep': self.t,
                    'wind_capacity': self.wind_instrument_value,
                    'solar_capacity': self.solar_instrument_value,
                    'hydro_capacity': self.hydro_instrument_value,
                    'battery_capacity': self.battery_capacity,
                    'wind': float(self._wind[i]) if i < len(self._wind) else 0.0,
                    'solar': float(self._solar[i]) if i < len(self._solar) else 0.0,
                    'hydro': float(self._hydro[i]) if i < len(self._hydro) else 0.0,
                    'revenue': self.last_revenue,
                    'market_stress': self.market_stress,
                }
                self.enhanced_risk_controller.update_risk_history(env_state)
                comp = self.enhanced_risk_controller.calculate_comprehensive_risk(env_state)
                self.overall_risk_snapshot   = float(np.clip(comp.get('overall_risk', 0.5), 0.0, 1.0))
                self.market_risk_snapshot    = float(np.clip(comp.get('market_risk', 0.3), 0.0, 1.0))
                self.portfolio_risk_snapshot = float(np.clip(comp.get('portfolio_risk', 0.25), 0.0, 1.0))
                self.liquidity_risk_snapshot = float(np.clip(comp.get('liquidity_risk', 0.15), 0.0, 1.0))
                return

            # Fallback: quick 6D vector API
            vec = self.enhanced_risk_controller.quick_risk_metrics(
                equity=self.equity,
                budget=self.budget,
                exposures=(self.wind_instrument_value, self.solar_instrument_value, self.hydro_instrument_value),
                price=float(self._price[i]),
                load=float(self._load[i]),
            )
            self.overall_risk_snapshot   = float(np.clip(vec[0], 0.0, 1.0))
            self.market_risk_snapshot    = float(np.clip(vec[1] if len(vec) > 1 else vec[0], 0.0, 1.0))
            self.portfolio_risk_snapshot = float(np.clip(vec[2] if len(vec) > 2 else vec[0], 0.0, 1.0))
            self.liquidity_risk_snapshot = float(np.clip(vec[3] if len(vec) > 3 else 0.15, 0.0, 1.0))

        except Exception as e:
            logging.warning(f"Risk snapshot update failed: {e}")
            # Fallback values
            self.overall_risk_snapshot = 0.5
            self.market_risk_snapshot = 0.3
            self.portfolio_risk_snapshot = 0.25
            self.liquidity_risk_snapshot = 0.15

    def _assign_rewards(self, financial: Dict[str, float]):
        try:
            env_state = {
                'budget': self.budget,
                'wind_pos': self.wind_instrument_value,
                'solar_pos': self.solar_instrument_value,
                'hydro_pos': self.hydro_instrument_value,
            }
            # Use comprehensive risk data (or fallbacks set in _update_risk_snapshots)
            risk = {
                'overall_risk': self.overall_risk_snapshot,
                'market_risk': getattr(self, 'market_risk_snapshot', 0.3),
                'portfolio_risk': getattr(self, 'portfolio_risk_snapshot', 0.25),
                'liquidity_risk': getattr(self, 'liquidity_risk_snapshot', 0.15),
            }
            total, breakdown = self.reward_calculator.calculate(env_state, financial, risk)
            # investor: emphasize financial; battery: efficiency; risk: risk; meta: total
            self._rew_buf['investor_0'] = float(np.clip(0.7 * breakdown['financial'] + 0.3 * total, -10, 10))
            self._rew_buf['battery_operator_0'] = float(np.clip(0.6 * breakdown['efficiency'] + 0.4 * total, -10, 10))
            self._rew_buf['risk_controller_0'] = float(np.clip(0.7 * breakdown['risk_management'] + 0.3 * total, -10, 10))
            self._rew_buf['meta_controller_0'] = float(np.clip(total, -10, 10))
            self.last_reward_breakdown = dict(breakdown)
            self.last_reward_weights = dict(self.reward_calculator.reward_weights)
        except Exception as e:
            logging.error(f"reward assignment: {e}")
            for a in self.possible_agents:
                self._rew_buf[a] = 0.0

    # ------------------------------------------------------------------
    # Observations (BASE only)
    # ------------------------------------------------------------------
    def _fill_obs(self):
        i = min(self.t, self.max_steps - 1)
        # scales
        price_n = float(np.clip(SafeDivision.div(self._price[i], 10.0, 0.0), 0.0, 10.0))
        load_n  = float(np.clip(self._load[i], 0.0, 1.0))
        windf   = float(np.clip(SafeDivision.div(self._wind[i],  self.wind_scale, 0.0), 0.0, 1.0))
        solarf  = float(np.clip(SafeDivision.div(self._solar[i], self.solar_scale, 0.0), 0.0, 1.0))
        hydrof  = float(np.clip(SafeDivision.div(self._hydro[i], self.hydro_scale, 0.0), 0.0, 1.0))
        # budget normalized by init/10 for dynamic range ~0..10
        init_div = self.init_budget / 10.0 if self.init_budget > 0 else 1.0
        budget_n = float(np.clip(SafeDivision.div(self.budget, init_div, 0.0), 0.0, 10.0))

        # investor (6)
        inv = self._obs_buf['investor_0']
        inv[:6] = (windf, solarf, hydrof, price_n, load_n, budget_n)

        # battery (4)
        batt = self._obs_buf['battery_operator_0']
        batt[:4] = (price_n,
                    float(np.clip(self.battery_energy, 0.0, 10.0)),
                    float(np.clip(self.battery_capacity, 0.0, 10.0)),
                    load_n)

        # risk controller (9) – regime cues and normalized positions
        rsk = self._obs_buf['risk_controller_0']
        rsk[:9] = (
            price_n,
            float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
            float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.wind_instrument_value,  self.init_budget), 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.solar_instrument_value, self.init_budget), 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.hydro_instrument_value, self.init_budget), 0.0, 1.0)) * 10.0,
            float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
            float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0)),
            float(np.clip(self.risk_multiplier, 0.0, 2.0)) * 5.0,
        )

        # meta (11)
        perf_ratio = float(np.clip(SafeDivision.div(self.equity, self.init_budget, 1.0), 0.0, 10.0))
        meta = self._obs_buf['meta_controller_0']
        meta[:11] = (
            float(np.clip(SafeDivision.div(self.budget, self.init_budget/10.0, 0.0), 0.0, 10.0)),
            float(np.clip(SafeDivision.div(self.wind_instrument_value,  self.init_budget, 0.0), 0.0, 10.0)),
            float(np.clip(SafeDivision.div(self.solar_instrument_value, self.init_budget, 0.0), 0.0, 10.0)),
            float(np.clip(SafeDivision.div(self.hydro_instrument_value, self.init_budget, 0.0), 0.0, 10.0)),
            price_n,
            float(np.clip(self.overall_risk_snapshot, 0.0, 1.0)) * 10.0,
            perf_ratio,
            float(np.clip(self.market_risk_snapshot, 0.0, 1.0)) * 10.0,
            float(np.clip(self.market_volatility, 0.0, 1.0)) * 10.0,
            float(np.clip(self.market_stress, 0.0, 1.0)) * 10.0,
            float(np.clip(self.capital_allocation_fraction, 0.0, 1.0)) * 10.0,
        )

    # ------------------------------------------------------------------
    # Info & helpers
    # ------------------------------------------------------------------
    def _populate_info(self, i: int, financial: Dict[str, float], acts: Dict[str, np.ndarray]):
        try:
            # expose capacity aliases for wrapper (it reads env.wind_capacity, etc.)
            self.wind_capacity  = self.wind_instrument_value
            self.solar_capacity = self.solar_instrument_value
            self.hydro_capacity = self.hydro_instrument_value

            for a in self.possible_agents:
                self._info_buf[a] = {
                    'wind': float(self._wind[i]),
                    'solar': float(self._solar[i]),
                    'hydro': float(self._hydro[i]),
                    'price': float(self._price[i]),
                    'load':  float(self._load[i]),
                    'wind_capacity': self.wind_instrument_value,
                    'solar_capacity': self.solar_instrument_value,
                    'hydro_capacity': self.hydro_instrument_value,
                    'budget': self.budget,
                    'initial_budget': self.init_budget,
                    'timestep': self.t,
                    'reward_breakdown': dict(self.last_reward_breakdown),
                    'reward_weights': dict(self.last_reward_weights),
                    'last_revenue': float(self.last_revenue),
                    'action_investor': acts['investor_0'].tolist(),
                    'action_battery':  acts['battery_operator_0'].tolist(),
                    'action_risk':     acts['risk_controller_0'].tolist(),
                    'action_meta':     acts['meta_controller_0'].tolist(),
                }
        except Exception as e:
            logging.error(f"info populate: {e}")

    # ------------------------------------------------------------------
    # Terminal / safety
    # ------------------------------------------------------------------
    def _terminal_step(self):
        for a in self.possible_agents:
            self._done_buf[a] = True
            self._trunc_buf[a] = True
            self._rew_buf[a] = 0.0
            self._info_buf[a].clear()
        return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

    def _safe_step(self):
        # neutral observations & zero rewards
        self._fill_obs()
        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

    # ------------------------------------------------------------------
    # State init
    # ------------------------------------------------------------------
    def _reset_portfolio_state(self):
        # finance
        self.budget = float(self.init_budget)  # cash only
        self.wind_instrument_value  = 0.0
        self.solar_instrument_value = 0.0
        self.hydro_instrument_value = 0.0
        self.equity = float(self.budget)  # cash + positions

        # Investment fund economics - separate capital from returns
        self.investment_capital = float(self.init_budget)  # Fixed capital base for investments
        self.distributed_profits = 0.0  # Profits distributed to investors (not reinvested)
        self.cumulative_returns = 0.0   # Total returns generated
        self.max_leverage = 1.5         # Maximum 1.5x leverage allowed

        # battery
        self.battery_capacity = 0.0
        self.battery_energy = 0.0
        self.battery_discharge_power = 0.0

        # knobs
        self.risk_multiplier = 1.0

        # logging helpers
        self.last_revenue = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = dict(self.reward_calculator.reward_weights)
        
        # comprehensive risk storage
        self.comprehensive_risk = {}
        self.portfolio_risk_snapshot = 0.25
        self.liquidity_risk_snapshot = 0.15

    # ------------------------------------------------------------------
    # Del
    # ------------------------------------------------------------------
    def __del__(self):
        try:
            self.memory_manager.cleanup_if_needed(force=True)
        except Exception:
            pass
