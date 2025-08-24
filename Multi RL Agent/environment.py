# environment.py
#!/usr/bin/env python3
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from collections import deque
import gc, psutil, os, logging

from risk import EnhancedRiskController  # keep your risk module

# ============================================================================#
# Utilities                                                                   #
# ============================================================================#
class SafeDivision:
    @staticmethod
    def _safe_divide(numerator: float, denominator: float = 1.0, default: float = 0.0) -> float:
        if denominator is None or abs(denominator) < 1e-8:
            return default
        return numerator / denominator


class MemoryManager:
    """Simple, inexpensive memory hygiene for longer runs."""
    def __init__(self, max_memory_mb=3000):
        self.max_memory_mb = max_memory_mb
        self._counter = 0

    def _rss_mb(self) -> float:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def cleanup_if_needed(self, force=False) -> bool:
        self._counter += 1
        if not force and (self._counter % 200 != 0):
            return False
        cur = self._rss_mb()
        if force or (self.max_memory_mb and cur > self.max_memory_mb):
            gc.collect()
            return True
        return False


# ============================================================================#
# Observations                                                                #
# ============================================================================#
class StabilizedObservationManager:
    """
    Manages fixed BASE observation specs.

    IMPORTANT for wrapper compatibility:
    - Env returns BASE-only observations (these spaces live here).
    - Wrapper appends forecasts → TOTAL = base_dim + forecast_dim.
    """
    def __init__(self, env, forecaster=None):
        self.env = env
        self.forecaster = forecaster
        self.observation_specs = self._build_fixed_specs()
        self.base_spaces = self._build_base_spaces()

    def _build_fixed_specs(self):
        specs = {}

        agent_cfg = {
            "investor_0":         {"base": 6,  "forecast": 8,  "total": 14},
            "battery_operator_0": {"base": 4,  "forecast": 4,  "total": 8},
            "risk_controller_0":  {"base": 9,  "forecast": 0,  "total": 9},
            "meta_controller_0":  {"base": 11, "forecast": 15, "total": 26}
        }

        for agent, cfg in agent_cfg.items():
            total_dim = cfg["total"]
            base_dim = cfg["base"]

            low = np.full(total_dim, -100.0, dtype=np.float32)
            high = np.full(total_dim, 1000.0, dtype=np.float32)

            if agent == "investor_0":
                low[:6]  = [0, 0, 0, 0, 0, 0]          # wind, solar, hydro, price, load, budget
                high[:6] = [1, 1, 1, 10, 1, 10]
            elif agent == "battery_operator_0":
                low[:4]  = [0, 0, 0, 0]                # price, energy, capacity, load
                high[:4] = [10, 10, 10, 1]
            elif agent == "risk_controller_0":
                low[:9]  = [0]*9                       # price, risk, budget + 6 risk metrics
                high[:9] = [10, 1, 10, 1, 1, 1, 1, 1, 1]
            elif agent == "meta_controller_0":
                low[:11]  = [0]*11                     # budget, 4 caps, price, risk, 4 perf metrics
                high[:11] = [10,10,10,10,10,10,1,10,10,10,10]

            specs[agent] = {
                "total_dim": total_dim,
                "base_dim": base_dim,
                "forecast_dim": cfg["forecast"],
                "low": low,
                "high": high,
            }
        return specs

    def _build_base_spaces(self) -> Dict[str, spaces.Box]:
        base_spaces = {}
        for agent, spec in self.observation_specs.items():
            bd = spec["base_dim"]
            low, high = spec["low"][:bd], spec["high"][:bd]
            base_spaces[agent] = spaces.Box(low=low, high=high, shape=(bd,), dtype=np.float32)
        return base_spaces

    def get_observation_space(self, agent: str) -> spaces.Box:
        return self.base_spaces.get(agent, spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32))

    def base_bounds(self, agent: str) -> Tuple[np.ndarray, np.ndarray]:
        spec = self.observation_specs.get(agent)
        if spec is None:
            return np.full(10, -10.0, np.float32), np.full(10, 10.0, np.float32)
        bd = spec["base_dim"]
        return spec["low"][:bd], spec["high"][:bd]


# ============================================================================#
# Reward calculator                                                           #
# ============================================================================#
class MultiObjectiveRewardCalculator:
    """Memory-conscious multi-objective reward."""
    def __init__(self, initial_budget=1e7, max_history_size=500):
        self.initial_budget = initial_budget
        self.performance_history = {
            'financial_scores': deque(maxlen=max_history_size),
            'risk_scores': deque(maxlen=max_history_size),
            'sustainability_scores': deque(maxlen=max_history_size),
            'efficiency_scores': deque(maxlen=max_history_size),
            'diversification_scores': deque(maxlen=max_history_size),
        }
        self.reward_weights = {
            'financial': 0.35,
            'risk_management': 0.25,
            'sustainability': 0.20,
            'efficiency': 0.15,
            'diversification': 0.05,
        }
        self.normalization = {
            'revenue_scale': 100000.0,
            'risk_scale': 1.0,
            'efficiency_scale': 1.0,
            'diversification_scale': 1.0
        }

    def calculate_multi_objective_reward(self, env_state: Dict, financial: Dict, risk: Dict) -> Tuple[float, Dict]:
        try:
            net_profit = float(financial.get('net_profit', 0.0))
            budget = float(env_state.get('budget', self.initial_budget))
            invested_capital = max(0.0, self.initial_budget - budget)

            # Financial: scaled profit + ROI shaping
            profit_reward = SafeDivision._safe_divide(net_profit, self.normalization['revenue_scale'])
            roi_reward = 0.0
            if invested_capital > 0:
                roi = SafeDivision._safe_divide(net_profit, invested_capital, 0.0)
                roi_reward = float(np.tanh(roi * 10) * 0.5)
            financial_reward = float(profit_reward + roi_reward)

            # Risk: lower is better
            overall_risk = float(risk.get('overall_risk', 0.5))
            risk_reward = float(-overall_risk * 2.0)

            # Sustainability: crude proxy by capacity scale
            tot_cap = float(env_state.get('wind_capacity', 0.0) +
                            env_state.get('solar_capacity', 0.0) +
                            env_state.get('hydro_capacity', 0.0))
            sustainability_reward = float(min(1.0, SafeDivision._safe_divide(tot_cap, 1000.0)))

            # Efficiency: pass-through
            efficiency_reward = float(financial.get('capacity_utilization', 0.0))

            # Diversification: 1 - HHI
            diversification_reward = self._div_score(env_state)

            total = (self.reward_weights['financial'] * financial_reward +
                     self.reward_weights['risk_management'] * risk_reward +
                     self.reward_weights['sustainability'] * sustainability_reward +
                     self.reward_weights['efficiency'] * efficiency_reward +
                     self.reward_weights['diversification'] * diversification_reward)

            # track light history
            self.performance_history['financial_scores'].append(financial_reward)
            self.performance_history['risk_scores'].append(risk_reward)
            self.performance_history['sustainability_scores'].append(sustainability_reward)
            self.performance_history['efficiency_scores'].append(efficiency_reward)
            self.performance_history['diversification_scores'].append(diversification_reward)

            breakdown = {
                'total': total,
                'financial': financial_reward,
                'risk_management': risk_reward,
                'sustainability': sustainability_reward,
                'efficiency': efficiency_reward,
                'diversification': diversification_reward
            }
            return float(total), breakdown
        except Exception as e:
            logging.error(f"⚠️ Reward calc error: {e}")
            return 0.0, {'total': 0.0, 'financial': 0.0, 'risk_management': 0.0,
                         'sustainability': 0.0, 'efficiency': 0.0, 'diversification': 0.0}

    @staticmethod
    def _div_score(env_state: Dict) -> float:
        w = float(env_state.get('wind_capacity', 0.0))
        s = float(env_state.get('solar_capacity', 0.0))
        h = float(env_state.get('hydro_capacity', 0.0))
        total = w + s + h
        if total > 0:
            shares = [x / total for x in (w, s, h) if x > 0]
            if len(shares) > 1:
                hhi = sum(x * x for x in shares)
                return float(1.0 - hhi)
        return 0.0


# ============================================================================#
# Main Environment                                                            #
# ============================================================================#
class RenewableMultiAgentEnv(ParallelEnv):
    """
    Parallel multi-agent env returning BASE-only observations.
    Wrapper appends forecasts to form TOTAL-dim observations.
    """
    metadata = {"render_modes": ["human"], "name": "renewable_marl_v0"}

    # meta param smoothing / clamps
    META_FREQ_MIN, META_FREQ_MAX = 24, 1008
    META_CAP_MIN, META_CAP_MAX = 0.01, 0.50
    META_SMOOTH_BETA, SAT_EPS = 0.8, 0.02

    def __init__(self, data: pd.DataFrame, investment_freq=144, forecast_generator=None, dl_adapter=None):
        # housekeeping
        self.memory_manager = MemoryManager(max_memory_mb=3000)

        # core data
        self.data = data.reset_index(drop=True)
        self.max_steps = len(self.data)
        self.t = 0
        self.step_in_episode = 0

        # controls
        self.investment_freq = int(investment_freq)
        self.capital_allocation_fraction = 0.1
        self.forecast_generator = forecast_generator
        self.dl_adapter = dl_adapter

        # vectorized columns (fast access)
        self._wind  = self.data['wind' ].to_numpy(np.float32, copy=False) if 'wind'  in self.data else np.zeros(self.max_steps, np.float32)
        self._solar = self.data['solar'].to_numpy(np.float32, copy=False) if 'solar' in self.data else np.zeros(self.max_steps, np.float32)
        self._hydro = self.data['hydro'].to_numpy(np.float32, copy=False) if 'hydro' in self.data else np.zeros(self.max_steps, np.float32)
        self._price = self.data['price'].to_numpy(np.float32, copy=False) if 'price' in self.data else np.full(self.max_steps, 50.0, np.float32)
        self._load  = self.data['load' ].to_numpy(np.float32, copy=False) if 'load'  in self.data else np.zeros(self.max_steps, np.float32)
        self._risk  = self.data['risk' ].to_numpy(np.float32, copy=False) if 'risk'  in self.data else np.full(self.max_steps, 0.3, np.float32)

        # agents
        self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]
        self.agents = self.possible_agents[:]

        # portfolio & budget
        self.init_budget = 1e7
        self.wind_capex, self.solar_capex, self.hydro_capex, self.battery_capex = 1000, 800, 1200, 400
        self._reset_portfolio_state()

        # observation manager (BASE only)
        self.obs_manager = StabilizedObservationManager(self, forecast_generator)

        # risk controller
        try:
            self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144)
        except Exception as e:
            logging.error(f"⚠️ Risk controller init failed: {e}")
            self.enhanced_risk_controller = None

        # reward calc
        self.reward_calculator = MultiObjectiveRewardCalculator(initial_budget=self.init_budget, max_history_size=500)

        # lightweight perf history
        self.performance_history = {
            'revenue_history': deque(maxlen=500),
            'capacity_utilization': deque(maxlen=200),
            'risk_adjusted_returns': deque(maxlen=200),
            'market_volatility': deque(maxlen=100),
        }

        # market regime snapshots (wrapper reads these)
        self.market_volatility = 0.0
        self.market_stress = 0.5
        self.overall_risk_snapshot = 0.5
        self.market_risk_snapshot = 0.5

        # last-step artifacts for wrapper logging
        self.last_revenue = 0.0
        self.last_reward_breakdown = {}
        self.last_reward_weights = dict(self.reward_calculator.reward_weights)

        # spaces & step buffers
        self.observation_spaces = {a: self.obs_manager.get_observation_space(a) for a in self.possible_agents}
        self.action_spaces = {
            "investor_0": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "battery_operator_0": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "risk_controller_0": spaces.Box(low=0.0,  high=2.0, shape=(1,), dtype=np.float32),
            "meta_controller_0": spaces.Box(low=0.0,  high=1.0, shape=(2,), dtype=np.float32),
        }

        self._obs_buf = {a: np.zeros(self.observation_spaces[a].shape, np.float32) for a in self.possible_agents}
        self._rew_buf = {a: 0.0 for a in self.possible_agents}
        self._done_buf = {a: False for a in self.possible_agents}
        self._trunc_buf = {a: False for a in self.possible_agents}
        self._info_buf = {a: {} for a in self.possible_agents}

        print("✅ RenewableMultiAgentEnv initialized with BASE-only observations")

    # ------------------------ API: Spaces & Reset ------------------------ #
    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.memory_manager.cleanup_if_needed(force=True)
        self.t = 0
        self.step_in_episode = 0
        self.agents = self.possible_agents[:]
        self._reset_portfolio_state()
        self.investor_reward_buffer = deque(maxlen=self.investment_freq or 144)

        for v in self.performance_history.values():
            v.clear()

        if self.enhanced_risk_controller:
            try:
                self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144)
            except Exception as e:
                logging.error(f"⚠️ Risk controller reset failed: {e}")

        self.reward_calculator = MultiObjectiveRewardCalculator(initial_budget=self.init_budget, max_history_size=500)
        self.last_reward_breakdown = {}
        self.last_reward_weights = dict(self.reward_calculator.reward_weights)
        self.last_revenue = 0.0
        self.overall_risk_snapshot = 0.5
        self.market_risk_snapshot = 0.5

        obs = self._get_obs_base_only()
        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        return obs, {}

    # ------------------------ Core Step ------------------------ #
    def step(self, actions):
        try:
            if self.t % 100 == 0:
                self.memory_manager.cleanup_if_needed()

            if self.t >= self.max_steps:
                return self._create_terminal_state()

            i = self.t

            # Optional DL overlay
            if self.dl_adapter is not None:
                actions = self._apply_dl_overlay(actions, i)

            # Apply actions (smoothed meta; clipped)
            self._process_actions_safe(actions, i)

            # Financials & budget
            financial = self._calculate_portfolio_performance_safe(i)
            self.last_revenue = float(financial.get('revenue', 0.0))
            self.budget = max(0.0, float(self.budget) + float(financial.get('net_profit', 0.0)))

            # Market conditions & risk controller updates
            self._update_market_conditions_safe(i)
            self._update_risk_controller_state(i, financial)

            # Perf history
            self._update_performance_history_safe(financial)

            # Rewards (expose breakdown & weights for wrapper logging)
            rewards = self._calculate_agent_rewards_safe(financial, i)

            # time advance
            self.t += 1
            self.step_in_episode += 1

            # Observations (BASE-only)
            obs = self._get_obs_base_only()

            # Dones/truncs
            for a in self.agents:
                self._done_buf[a] = False
                self._trunc_buf[a] = False

            # Meta info (handy for debugging)
            self._info_buf["meta_controller_0"].update({
                "investment_freq": int(self.investment_freq),
                "capital_allocation_fraction": float(self.capital_allocation_fraction),
                "saturation_hits": int(self._last_saturation_count),
            })

            return obs, rewards, self._done_buf, self._trunc_buf, self._info_buf

        except Exception as e:
            logging.error(f"⚠️ Step error at t={self.t}: {e}")
            return self._create_safe_step_result()

    # ------------------------ DL Overlay ------------------------ #
    def _apply_dl_overlay(self, actions: Dict[str, Any], t: int) -> Dict[str, Any]:
        if self.dl_adapter is None:
            return actions
        try:
            self.dl_adapter.maybe_learn(t)
            w = self.dl_adapter.infer_weights(t)
            modified = actions.copy()
            modified['investor_0'] = self.dl_adapter.weights_to_action(w)
            return modified
        except Exception as e:
            logging.warning(f"DL overlay failed: {e}")
            return actions

    # ------------------------ Observations (BASE) ------------------------ #
    def _get_obs_base_only(self) -> Dict[str, np.ndarray]:
        if self.t >= self.max_steps:
            self._fill_safe_observation_buffers()
            return self._obs_buf

        i = self.t
        init_div = (self.init_budget / 10.0) if self.init_budget > 0 else 1.0

        budget_n = float(np.clip(SafeDivision._safe_divide(self.budget, init_div), 0, 10))
        price_n  = float(np.clip(SafeDivision._safe_divide(self._price[i], 10.0), 0, 10))
        windf, solarf, hydrof = [float(np.clip(x, 0, 1)) for x in (self._wind[i], self._solar[i], self._hydro[i])]
        loadf, riskf = float(np.clip(self._load[i], 0, 1)), float(np.clip(self._risk[i], 0, 1))

        # investor (6)
        inv = self._obs_buf["investor_0"]
        inv[:6] = (windf, solarf, hydrof, price_n, loadf, budget_n)
        low, high = self.obs_manager.base_bounds("investor_0")
        np.clip(inv[:6], low, high, out=inv[:6])

        # battery (4)
        batt = self._obs_buf["battery_operator_0"]
        batt[:4] = (price_n,
                    float(np.clip(self.battery_energy,   0, 10)),
                    float(np.clip(self.battery_capacity, 0, 10)),
                    loadf)
        low, high = self.obs_manager.base_bounds("battery_operator_0")
        np.clip(batt[:4], low, high, out=batt[:4])

        # risk (9) : price, risk, budget, 6 metrics
        rm = self._get_enhanced_risk_metrics_safe()
        rsk = self._obs_buf["risk_controller_0"]
        rsk[:3] = (price_n, riskf, budget_n)
        rsk[3:9] = rm[:6].astype(np.float32)
        low, high = self.obs_manager.base_bounds("risk_controller_0")
        np.clip(rsk[:9], low, high, out=rsk[:9])

        # meta (11) : budget, 4 caps, price, risk, 4 perf metrics
        perf = self._get_performance_metrics_safe().astype(np.float32)
        m = self._obs_buf["meta_controller_0"]
        m[:7] = (budget_n,
                 float(np.clip(SafeDivision._safe_divide(self.wind_capacity, 1000.0, 0), 0, 10)),
                 float(np.clip(SafeDivision._safe_divide(self.solar_capacity, 1000.0, 0), 0, 10)),
                 float(np.clip(SafeDivision._safe_divide(self.hydro_capacity, 1000.0, 0), 0, 10)),
                 float(np.clip(SafeDivision._safe_divide(self.battery_capacity, 1000.0, 0), 0, 10)),
                 price_n, riskf)
        m[7:11] = perf[:4]
        low, high = self.obs_manager.base_bounds("meta_controller_0")
        np.clip(m[:11], low, high, out=m[:11])

        return self._obs_buf

    def _get_enhanced_risk_metrics_safe(self) -> np.ndarray:
        try:
            if self.enhanced_risk_controller:
                return np.asarray(self.enhanced_risk_controller.get_risk_metrics_for_observation(), dtype=np.float32)
        except Exception:
            pass
        return np.array([0.5, 0.2, 0.3, 0.1, 0.4, 0.25], dtype=np.float32)

    def _get_performance_metrics_safe(self) -> np.ndarray:
        try:
            portfolio_value = self.budget + (self.wind_capacity + self.solar_capacity + self.hydro_capacity) * 100.0
            perf_ratio = float(np.clip(SafeDivision._safe_divide(portfolio_value, self.init_budget, 1.0), 0, 10))
            util = float(np.clip(np.mean(self.performance_history['capacity_utilization']) if self.performance_history['capacity_utilization'] else 0.0, 0, 1))
            rar  = float(np.clip(np.mean(self.performance_history['risk_adjusted_returns']) if self.performance_history['risk_adjusted_returns'] else 0.0, -10, 10))
            vol  = float(np.clip(np.mean(self.performance_history['market_volatility']) if self.performance_history['market_volatility'] else 0.0, 0, 10))
            return np.array([perf_ratio, util, rar, vol], dtype=np.float32)
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _fill_safe_observation_buffers(self):
        for agent in self.possible_agents:
            low, high = self.obs_manager.base_bounds(agent)
            self._obs_buf[agent][:] = ((low + high) / 2.0).astype(np.float32)

    # ------------------------ Actions & Dynamics ------------------------ #
    def _process_actions_safe(self, actions: Dict[str, Any], i: int):
        try:
            # Validate & clip into arrays
            validated = {}
            for agent in self.possible_agents:
                space = self.action_spaces[agent]
                a = actions.get(agent, None)
                if a is None:
                    # mid action by default
                    mid = (space.low + space.high) / 2.0
                    validated[agent] = mid.astype(np.float32)
                    continue
                arr = np.array(a, dtype=np.float32).flatten()
                if arr.size != space.shape[0]:
                    if arr.size < space.shape[0]:
                        mid = (space.low + space.high) / 2.0
                        padval = float(mid[0]) if hasattr(mid, "__len__") else float(mid)
                        arr = np.concatenate([arr, np.full(space.shape[0] - arr.size, padval, np.float32)])
                    else:
                        arr = arr[: space.shape[0]]
                arr = np.nan_to_num(arr, nan=0.0)
                validated[agent] = np.minimum(np.maximum(arr, space.low), space.high).astype(np.float32)

            # Meta (EMA-smoothed → freq & cap_frac)
            meta = validated.get("meta_controller_0", np.array([0.5, 0.1], np.float32))
            self._last_meta_raw = meta.copy()
            tgt_freq = float(self.META_FREQ_MIN + meta[0] * (self.META_FREQ_MAX - self.META_FREQ_MIN))
            tgt_cap  = float(self.META_CAP_MIN + meta[1] * (self.META_CAP_MAX - self.META_CAP_MIN))
            beta = self.META_SMOOTH_BETA
            self.investment_freq = int(np.clip(beta * self.investment_freq + (1 - beta) * tgt_freq, self.META_FREQ_MIN, self.META_FREQ_MAX))
            self.capital_allocation_fraction = float(np.clip(beta * self.capital_allocation_fraction + (1 - beta) * tgt_cap, self.META_CAP_MIN, self.META_CAP_MAX))

            # Risk (0..2 → risk_multiplier)
            self.risk_multiplier = float(np.clip(validated.get("risk_controller_0", np.array([1.0], np.float32))[0], 0.0, 2.0))

            # Investor allocations at gating
            if self.t % max(1, self.investment_freq) == 0:
                self.last_investor_action = np.clip(validated.get("investor_0", np.array([0.0, 0.0, 0.0], np.float32)), -1.0, 1.0)
            self._execute_investments_safe()

            # Battery ops continuous
            self._execute_battery_operations_safe(float(validated.get("battery_operator_0", np.array([0.0], np.float32))[0]), i)

            # Saturation (for tiny regularization)
            self._last_saturation_count = self._count_saturation_hits(meta, self.last_investor_action)
        except Exception as e:
            logging.error(f"⚠️ Action processing error: {e}")

    def _count_saturation_hits(self, meta_action: np.ndarray, investor_action: np.ndarray) -> int:
        cnt = 0
        if investor_action is not None and investor_action.size > 0:
            cnt += int(np.sum(np.abs(investor_action) >= (1.0 - self.SAT_EPS)))
        cnt += int(meta_action[0] <= self.SAT_EPS or meta_action[0] >= (1.0 - self.SAT_EPS))
        cnt += int(meta_action[1] <= self.SAT_EPS or meta_action[1] >= (1.0 - self.SAT_EPS))
        cnt += int(self.investment_freq <= self.META_FREQ_MIN + 1 or self.investment_freq >= self.META_FREQ_MAX - 1)
        cnt += int(self.capital_allocation_fraction <= self.META_CAP_MIN + 1e-6 or self.capital_allocation_fraction >= self.META_CAP_MAX - 1e-6)
        return int(cnt)

    def _execute_investments_safe(self):
        try:
            invest_action = np.clip((self.last_investor_action + 1.0) / 2.0, 0, 1)
            w_share, s_share, h_share = invest_action
            total_share = w_share + s_share + h_share
            if total_share <= 0 or self.budget <= 0:
                return

            available = max(0.0, self.budget * 0.9)
            base_alloc = min(available * self.capital_allocation_fraction, self.budget * 0.5)

            if self.risk_multiplier >= 1.0:
                alloc = SafeDivision._safe_divide(base_alloc, max(1e-6, self.risk_multiplier))
            else:
                alloc = base_alloc * min(1.25, SafeDivision._safe_divide(1.0, max(self.risk_multiplier, 1e-3)))
            allocated = float(np.clip(alloc, 0.0, self.budget))

            ws = SafeDivision._safe_divide(w_share, total_share, 0.0)
            ss = SafeDivision._safe_divide(s_share, total_share, 0.0)
            hs = SafeDivision._safe_divide(h_share, total_share, 0.0)

            invest_w = min(allocated * ws, self.budget)
            invest_s = min(allocated * ss, max(0.0, self.budget - invest_w))
            invest_h = min(allocated * hs, max(0.0, self.budget - invest_w - invest_s))

            self.wind_capacity  = float(np.clip(self.wind_capacity  + SafeDivision._safe_divide(invest_w, self.wind_capex, 0.0), 0, 10000))
            self.solar_capacity = float(np.clip(self.solar_capacity + SafeDivision._safe_divide(invest_s, self.solar_capex, 0.0), 0, 10000))
            self.hydro_capacity = float(np.clip(self.hydro_capacity + SafeDivision._safe_divide(invest_h, self.hydro_capex, 0.0), 0, 10000))

            self.budget = max(0.0, self.budget - (invest_w + invest_s + invest_h))
        except Exception as e:
            logging.error(f"⚠️ Investment exec error: {e}")

    def _execute_battery_operations_safe(self, battery_action: float, i: int):
        try:
            if self.battery_capacity == 0.0:
                min_invest = min(self.init_budget * 0.01, self.budget * 0.1)
                self.battery_capacity = SafeDivision._safe_divide(min_invest, self.battery_capex, 0.0)

            w = self.wind_capacity  * float(np.clip(self._wind[i],  0, 1))
            s = self.solar_capacity * float(np.clip(self._solar[i], 0, 1))
            h = self.hydro_capacity * float(np.clip(self._hydro[i], 0, 1))
            total_gen = w + s + h

            u = float(np.clip((battery_action + 1.0) / 2.0, 0, 1))
            if u > 0.5:  # discharge
                rate = (u - 0.5) * 2
                discharge = min(rate * self.battery_capacity * 0.1, self.battery_energy)
                self.battery_energy = max(0.0, self.battery_energy - discharge)
                total_gen += discharge
            else:       # charge
                rate = (0.5 - u) * 2
                max_charge = min(rate * self.battery_capacity * 0.1,
                                 total_gen * 0.5,
                                 self.battery_capacity - self.battery_energy)
                charge = max(0.0, max_charge)
                self.battery_energy = min(self.battery_capacity, self.battery_energy + charge)
                total_gen = max(0.0, total_gen - charge)

            self.battery_energy   = float(np.clip(self.battery_energy, 0, self.battery_capacity))
            self.battery_capacity = float(np.clip(self.battery_capacity, 0, 10000))
        except Exception as e:
            logging.error(f"⚠️ Battery ops error: {e}")

    # ------------------------ Finance & Risk ------------------------ #
    def _calculate_portfolio_performance_safe(self, i: int) -> Dict[str, float]:
        try:
            wp = float(np.clip(self.wind_capacity  * self._wind[i],  0, 10000))
            sp = float(np.clip(self.solar_capacity * self._solar[i], 0, 10000))
            hp = float(np.clip(self.hydro_capacity * self._hydro[i], 0, 10000))
            total_gen = wp + sp + hp

            price = float(np.clip(self._price[i], 0, 1000))
            revenue = total_gen * price * (10.0 / 60.0)  # 10-min time step

            total_cap = self.wind_capacity + self.solar_capacity + self.hydro_capacity
            opex = total_cap * 0.02 + self.battery_capacity * 0.01
            net_profit = revenue - opex

            cap_util = float(np.clip(SafeDivision._safe_divide(total_gen, total_cap, 0.0), 0, 1))
            batt_eff = float(np.clip(SafeDivision._safe_divide(self.battery_energy, self.battery_capacity, 0.0), 0, 1))

            return {
                'revenue': revenue,
                'net_profit': net_profit,
                'costs': opex,
                'total_generation': total_gen,
                'renewable_generation': total_gen,
                'capacity_utilization': cap_util,
                'battery_efficiency': batt_eff,
                'portfolio_var': 0.0,
                'market_volatility': self.market_volatility
            }
        except Exception as e:
            logging.error(f"⚠️ Performance calc error: {e}")
            return {'revenue': 0.0, 'net_profit': 0.0, 'costs': 0.0,
                    'total_generation': 0.0, 'renewable_generation': 0.0,
                    'capacity_utilization': 0.0, 'battery_efficiency': 0.0,
                    'portfolio_var': 0.0, 'market_volatility': 0.0}

    def _update_market_conditions_safe(self, i: int):
        try:
            if len(self.performance_history['revenue_history']) > 10:
                r = list(self.performance_history['revenue_history'])[-10:]
                if len(r) > 1:
                    ret = np.diff(r) / (np.array(r[:-1]) + 1e-8)
                    self.market_volatility = float(np.clip(np.std(ret), 0, 1))
            self.market_stress = float(np.clip(self._risk[i], 0, 1))
        except Exception:
            self.market_volatility = 0.2
            self.market_stress = 0.5

    def _update_risk_controller_state(self, i: int, financial: Dict[str, float]):
        try:
            # quick snapshots (used by wrapper's quick risk)
            if self.enhanced_risk_controller:
                metrics = np.asarray(self.enhanced_risk_controller.get_risk_metrics_for_observation(), dtype=np.float32).reshape(-1)
                if metrics.size >= 6:
                    self.market_risk_snapshot = float(np.clip(metrics[0], 0.0, 1.0))
                    self.overall_risk_snapshot = float(np.clip(metrics[-1], 0.0, 1.0))

            if not self.enhanced_risk_controller:
                return

            price = float(self._price[i])
            prev = float(self._price[i - 1]) if i > 0 else price
            price_ret = SafeDivision._safe_divide(price - prev, prev + 1e-8, 0.0)

            payload = {
                'price': price,
                'price_return': float(price_ret),
                'budget': float(self.budget),
                'net_profit': float(financial.get('net_profit', 0.0)),
                'revenue': float(financial.get('revenue', 0.0)),
                'capacity': float(self.wind_capacity + self.solar_capacity + self.hydro_capacity),
                'market_stress': float(self.market_stress),
                'volatility_proxy': float(self.market_volatility),
                'timestep': int(self.t),
            }
            # Be tolerant to controller API shape
            if hasattr(self.enhanced_risk_controller, "update_observation"):
                try: self.enhanced_risk_controller.update_observation(payload)
                except Exception: pass
            if hasattr(self.enhanced_risk_controller, "update_risk_history"):
                try: self.enhanced_risk_controller.update_risk_history(payload)
                except Exception: pass
        except Exception:
            pass

    def _update_performance_history_safe(self, financial: Dict[str, float]):
        try:
            self.performance_history['revenue_history'].append(financial['revenue'])
            self.performance_history['capacity_utilization'].append(financial['capacity_utilization'])
            if self.init_budget > self.budget:
                rar = SafeDivision._safe_divide(financial['net_profit'], (self.init_budget - self.budget))
                self.performance_history['risk_adjusted_returns'].append(float(np.clip(rar, -10, 10)))
            self.performance_history['market_volatility'].append(self.market_volatility)
        except Exception as e:
            logging.error(f"⚠️ Perf history error: {e}")

    def _calculate_agent_rewards_safe(self, financial: Dict[str, float], i: int) -> Dict[str, float]:
        try:
            # scale overall risk by risk_multiplier
            base_risk = self._get_risk_assessment_safe(i)
            scaled_overall = float(np.clip(base_risk.get('overall_risk', 0.5) * self.risk_multiplier, 0.0, 2.0))
            risk_scaled = dict(base_risk); risk_scaled['overall_risk'] = scaled_overall

            env_state = {
                'budget': self.budget,
                'wind_capacity': self.wind_capacity,
                'solar_capacity': self.solar_capacity,
                'hydro_capacity': self.hydro_capacity,
                'battery_capacity': self.battery_capacity,
                'market_volatility': self.market_volatility,
            }

            total, breakdown = self.reward_calculator.calculate_multi_objective_reward(env_state, financial, risk_scaled)

            # expose for wrapper logging
            self.last_reward_breakdown = dict(breakdown)
            self.last_reward_weights = dict(self.reward_calculator.reward_weights)

            # small regularizer for saturation
            sat_pen = -0.001 * float(self._last_saturation_count)

            inv = total * 0.8 + breakdown['financial'] * 0.2 + sat_pen
            self.investor_reward_buffer.append(inv)
            inv_smoothed = float(np.mean(self.investor_reward_buffer)) if self.investor_reward_buffer else 0.0

            batt = (breakdown['efficiency'] * 0.5 + breakdown['financial'] * 0.3 + total * 0.2)
            rsk  = (breakdown['risk_management'] * 0.6 + (1.0 - scaled_overall) * 0.4)
            meta = total + breakdown['diversification'] * 0.2 + sat_pen

            self._rew_buf["investor_0"] = float(np.clip(inv_smoothed, -10, 10))
            self._rew_buf["battery_operator_0"] = float(np.clip(batt, -10, 10))
            self._rew_buf["risk_controller_0"] = float(np.clip(rsk, -10, 10))
            self._rew_buf["meta_controller_0"] = float(np.clip(meta, -10, 10))
            return self._rew_buf
        except Exception as e:
            logging.error(f"⚠️ Reward calc error: {e}")
            return {a: 0.0 for a in self.agents}

    def _get_risk_assessment_safe(self, i: int) -> Dict[str, float]:
        try:
            if self.enhanced_risk_controller:
                state = {
                    'price': float(self._price[i]),
                    'wind': float(self._wind[i]),
                    'solar': float(self._solar[i]),
                    'hydro': float(self._hydro[i]),
                    'wind_capacity': self.wind_capacity,
                    'solar_capacity': self.solar_capacity,
                    'hydro_capacity': self.hydro_capacity,
                    'budget': self.budget,
                    'initial_budget': self.init_budget,
                    'timestep': self.t,
                    'revenue': self.last_revenue,
                    'market_stress': self.market_stress,
                }
                self.enhanced_risk_controller.update_risk_history(state)
                return self.enhanced_risk_controller.calculate_comprehensive_risk(state)
        except Exception:
            pass
        return {'market_risk': 0.3, 'operational_risk': 0.2, 'portfolio_risk': 0.3,
                'liquidity_risk': 0.1, 'regulatory_risk': 0.2, 'overall_risk': 0.25}

    # ------------------------ Adaptation & Analysis ------------------------ #
    def adapt_reward_weights(self):
        """Very light reweighting based on regime."""
        try:
            stress, vol = float(self.market_stress), float(self.market_volatility)
            w = self.reward_calculator.reward_weights
            delta = 0.01
            if stress > 0.6 or vol > 0.4:
                w['risk_management'] = min(0.35, w['risk_management'] + delta)
                w['financial'] = max(0.25, w['financial'] - delta)
            else:
                w['financial'] = min(0.45, w['financial'] + delta)
                w['risk_management'] = max(0.15, w['risk_management'] - delta)
            s = sum(w.values())
            for k in w:
                w[k] = SafeDivision._safe_divide(w[k], s, w[k])
        except Exception:
            pass

    def get_reward_analysis(self):
        """Console summary (uses calculator history)."""
        try:
            w = self.reward_calculator.reward_weights
            hist = self.reward_calculator.performance_history
            print("📊 Reward Weights:", {k: round(v, 3) for k, v in w.items()})
            if hist['financial_scores']:
                print("   Avg financial:", round(float(np.mean(hist['financial_scores'])), 4))
            if hist['risk_scores']:
                print("   Avg risk:", round(float(np.mean(hist['risk_scores'])), 4))
        except Exception:
            pass

    # ------------------------ Terminal / Fallbacks ------------------------ #
    def _create_terminal_state(self):
        self._fill_safe_observation_buffers()
        for a in self.agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = True
            self._trunc_buf[a] = True
            self._info_buf[a].clear()
        return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

    def _create_safe_step_result(self):
        self._fill_safe_observation_buffers()
        for a in self.agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        return self._obs_buf, self._rew_buf, self._done_buf, self._trunc_buf, self._info_buf

    # ------------------------ Wrapper-facing helpers ------------------------ #
    def _reset_portfolio_state(self):
        self.wind_capacity = 0.0
        self.solar_capacity = 0.0
        self.hydro_capacity = 0.0
        self.battery_capacity = 0.0
        self.battery_energy = 0.0
        self.budget = self.init_budget
        self.risk_multiplier = 1.0
        self.last_investor_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.investor_reward_buffer = deque(maxlen=144)

    def _get_base_observation_dim(self, agent: str) -> int:
        return self.obs_manager.observation_specs[agent]['base_dim']

    def _get_forecast_dimension(self, agent: str) -> int:
        return self.obs_manager.observation_specs[agent]['forecast_dim']

    def _get_performance_dimension(self, agent: str) -> int:
        return 0  # performance is part of BASE for meta agent

    def __del__(self):
        try:
            self.memory_manager.cleanup_if_needed(force=True)
        except Exception:
            pass
