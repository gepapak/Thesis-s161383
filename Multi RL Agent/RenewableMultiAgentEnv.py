from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from enhanced_risk_controller import EnhancedRiskController
from collections import deque
import gc
import psutil
import os


class MemoryManager:
    """Memory management utility to prevent memory leaks"""
    def __init__(self, max_memory_mb=4000):
        self.max_memory_mb = max_memory_mb
        self.cleanup_counter = 0

    def check_memory(self):
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except Exception:
            return 0.0

    def cleanup_if_needed(self, force=False):
        # throttle checks a bit (cheap guard)
        self.cleanup_counter += 1
        if not force and (self.cleanup_counter % 200 != 0):
            return False
        current_memory = self.check_memory()
        if force or (self.max_memory_mb and current_memory > self.max_memory_mb):
            gc.collect()
            if force:
                print(f"🧹 Memory cleanup forced. Usage: {current_memory:.1f}MB")
            return True
        return False


class StabilizedObservationManager:
    """
    Manages fixed observation specs.

    IMPORTANT for wrapper compatibility:
    - 'base_dim' is what the ENV will actually emit.
    - 'total_dim' = base_dim + forecast_dim (what the WRAPPER will expose after appending forecasts).
    - Env's observation_space() reflects BASE dims only.
    """
    def __init__(self, env, forecaster=None):
        self.env = env
        self.forecaster = forecaster
        self.observation_specs = self._build_fixed_specs()
        self.base_spaces = self._build_base_spaces()

    def _build_fixed_specs(self):
        """Build observation specs that guarantee consistency"""
        specs = {}

        agent_configs = {
            "investor_0":         {"base": 6,  "forecast": 8,  "total": 14},
            "battery_operator_0": {"base": 4,  "forecast": 4,  "total": 8},
            "risk_controller_0":  {"base": 9,  "forecast": 0,  "total": 9},
            "meta_controller_0":  {"base": 11, "forecast": 15, "total": 26}
        }

        for agent, config in agent_configs.items():
            total_dim = config["total"]
            base_dim = config["base"]
            low  = np.full(total_dim, -10.0, dtype=np.float32)
            high = np.full(total_dim,  10.0, dtype=np.float32)

            if agent == "investor_0":
                low[:6]  = [0, 0, 0, 0, 0, 0]          # wind, solar, hydro, price, load, budget
                high[:6] = [1, 1, 1, 10, 1, 10]
            elif agent == "battery_operator_0":
                low[:4]  = [0, 0, 0, 0]                # price, energy, capacity, load
                high[:4] = [10, 10, 10, 1]
            elif agent == "risk_controller_0":
                low[:9]  = [0]*9                       # price, risk, budget + 6 metrics
                high[:9] = [10, 1, 10, 1, 1, 1, 1, 1, 1]
            elif agent == "meta_controller_0":
                low[:11]  = [0]*11                     # budget, capacities, price, risk, + 4 perf metrics
                high[:11] = [10,10,10,10,10,10,1,10,10,10,10]

            # Forecast bounds (for wrapper)
            if config["forecast"] > 0:
                low[base_dim:]  = 0.0
                high[base_dim:] = 10.0

            specs[agent] = {
                "total_dim": total_dim,
                "base_dim": base_dim,
                "forecast_dim": config["forecast"],
                "low": low,
                "high": high,
            }

        return specs

    def _build_base_spaces(self) -> Dict[str, spaces.Box]:
        """Spaces that match what the ENV returns (base only)."""
        base_spaces = {}
        for agent, spec in self.observation_specs.items():
            bd = spec["base_dim"]
            low  = spec["low"][:bd]
            high = spec["high"][:bd]
            base_spaces[agent] = spaces.Box(low=low, high=high, shape=(bd,), dtype=np.float32)
        return base_spaces

    def get_observation_space(self, agent: str) -> spaces.Box:
        """Return BASE-ONLY space for the env output (wrapper will append forecasts)."""
        if agent in self.base_spaces:
            return self.base_spaces[agent]
        return spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)

    def base_bounds(self, agent: str) -> Tuple[np.ndarray, np.ndarray]:
        spec = self.observation_specs.get(agent)
        if spec is None:
            low  = np.full(10, -10.0, dtype=np.float32)
            high = np.full(10,  10.0, dtype=np.float32)
            return low, high
        bd = spec["base_dim"]
        return spec["low"][:bd], spec["high"][:bd]


class MultiObjectiveRewardCalculator:
    """Memory-efficient multi-objective reward calculation"""
    def __init__(self, initial_budget=1e7, max_history_size=500):
        self.initial_budget = initial_budget
        self.max_history_size = max_history_size
        self.reward_weights = {
            'financial': 0.35,
            'risk_management': 0.25,
            'sustainability': 0.20,
            'efficiency': 0.15,
            'diversification': 0.05
        }
        self.performance_history = {
            'financial_scores': deque(maxlen=max_history_size),
            'risk_scores': deque(maxlen=max_history_size),
            'sustainability_scores': deque(maxlen=max_history_size),
            'efficiency_scores': deque(maxlen=max_history_size),
            'diversification_scores': deque(maxlen=max_history_size)
        }
        self.normalization = {
            'revenue_scale': 100000.0,
            'risk_scale': 1.0,
            'efficiency_scale': 1.0,
            'diversification_scale': 1.0
        }

    def calculate_multi_objective_reward(self, env_state: Dict, financial_metrics: Dict,
                                         risk_assessment: Dict) -> Tuple[float, Dict]:
        try:
            net_profit = financial_metrics.get('net_profit', 0.0)
            budget = env_state.get('budget', self.initial_budget)
            invested_capital = self.initial_budget - budget

            # Financial
            profit_reward = net_profit / self.normalization['revenue_scale']
            roi_reward = 0.0
            if invested_capital > 0:
                roi = net_profit / invested_capital
                roi_reward = np.tanh(roi * 10) * 0.5
            financial_reward = profit_reward + roi_reward

            # Risk
            overall_risk = risk_assessment.get('overall_risk', 0.5)
            risk_reward = -overall_risk * 2.0

            # Sustainability
            total_capacity = sum([
                env_state.get('wind_capacity', 0),
                env_state.get('solar_capacity', 0),
                env_state.get('hydro_capacity', 0)
            ])
            sustainability_reward = min(1.0, total_capacity / 1000.0)

            # Efficiency
            capacity_utilization = financial_metrics.get('capacity_utilization', 0.0)
            efficiency_reward = capacity_utilization

            # Diversification
            diversification_reward = self._calculate_simple_diversification(env_state)

            total_reward = (
                self.reward_weights['financial'] * financial_reward +
                self.reward_weights['risk_management'] * risk_reward +
                self.reward_weights['sustainability'] * sustainability_reward +
                self.reward_weights['efficiency'] * efficiency_reward +
                self.reward_weights['diversification'] * diversification_reward
            )

            # Track history (lightweight)
            self.performance_history['financial_scores'].append(financial_reward)
            self.performance_history['risk_scores'].append(risk_reward)
            self.performance_history['sustainability_scores'].append(sustainability_reward)
            self.performance_history['efficiency_scores'].append(efficiency_reward)
            self.performance_history['diversification_scores'].append(diversification_reward)

            reward_breakdown = {
                'total': total_reward,
                'financial': financial_reward,
                'risk_management': risk_reward,
                'sustainability': sustainability_reward,
                'efficiency': efficiency_reward,
                'diversification': diversification_reward
            }
            return total_reward, reward_breakdown

        except Exception as e:
            print(f"⚠️ Reward calculation error: {e}")
            return 0.0, {'total': 0.0, 'financial': 0.0, 'risk_management': 0.0,
                         'sustainability': 0.0, 'efficiency': 0.0, 'diversification': 0.0}

    def _calculate_simple_diversification(self, env_state: Dict) -> float:
        capacities = [
            env_state.get('wind_capacity', 0),
            env_state.get('solar_capacity', 0),
            env_state.get('hydro_capacity', 0)
        ]
        total = sum(capacities)
        if total > 0:
            shares = [c / total for c in capacities if c > 0]
            if len(shares) > 1:
                hhi = sum(s**2 for s in shares)
                return 1.0 - hhi
        return 0.0


class RenewableMultiAgentEnv(ParallelEnv):
    """
    Parallel multi-agent env for renewable portfolio optimization.

    Meta-controller action semantics (Box[0,1]^2):
      - action[0] in [0,1] → investment_freq in [24, 1008] (10-min steps) with EMA smoothing
      - action[1] in [0,1] → capital_allocation_fraction in [0.01, 0.50] with EMA smoothing
    """
    metadata = {"render_modes": ["human"], "name": "renewable_marl_v0"}

    # ---- meta smoothing / clamp ranges ----
    META_FREQ_MIN = 24
    META_FREQ_MAX = 1008
    META_CAP_MIN = 0.01
    META_CAP_MAX = 0.50
    META_SMOOTH_BETA = 0.8   # higher = smoother (new = beta*old + (1-beta)*target)
    SAT_EPS = 0.02           # how close to bounds counts as "saturated"

    def __init__(self, data: pd.DataFrame, investment_freq=144, forecast_generator=None):
        # Memory management
        self.memory_manager = MemoryManager(max_memory_mb=3000)

        # --- Core environment state ---
        self.data = data.reset_index(drop=True)
        self.max_steps = len(self.data)
        self.t = 0
        self.investment_freq = int(investment_freq)
        self.capital_allocation_fraction = 0.1
        self.forecast_generator = forecast_generator

        # keep last raw meta for diagnostics/penalty
        self._last_meta_raw = np.array([0.5, 0.1], dtype=np.float32)
        self._last_saturation_count = 0

        # --- Convert dataframe columns to fast NumPy views ---
        self._wind  = self.data['wind' ].to_numpy(dtype=np.float32, copy=False) if 'wind'  in self.data else np.zeros(self.max_steps, np.float32)
        self._solar = self.data['solar'].to_numpy(dtype=np.float32, copy=False) if 'solar' in self.data else np.zeros(self.max_steps, np.float32)
        self._hydro = self.data['hydro'].to_numpy(dtype=np.float32, copy=False) if 'hydro' in self.data else np.zeros(self.max_steps, np.float32)
        self._price = self.data['price'].to_numpy(dtype=np.float32, copy=False) if 'price' in self.data else np.full(self.max_steps, 50.0, np.float32)
        self._load  = self.data['load' ].to_numpy(dtype=np.float32, copy=False) if 'load'  in self.data else np.zeros(self.max_steps, np.float32)
        self._risk  = self.data['risk' ].to_numpy(dtype=np.float32, copy=False) if 'risk'  in self.data else np.full(self.max_steps, 0.3, np.float32)

        # Agents
        self.possible_agents = ["investor_0", "battery_operator_0", "risk_controller_0", "meta_controller_0"]
        self.agents = self.possible_agents[:]

        # Portfolio state with bounds checking
        self.init_budget = 1e7
        self.wind_capex = 1000
        self.solar_capex = 800
        self.hydro_capex = 1200
        self.battery_capex = 400

        self._reset_portfolio_state()

        # Observation manager (defines specs; env emits BASE only)
        self.obs_manager = StabilizedObservationManager(self, forecast_generator)

        # Enhanced risk controller
        try:
            self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144)
        except Exception as e:
            print(f"⚠️ Risk controller initialization failed: {e}")
            self.enhanced_risk_controller = None

        # Reward calculator
        self.reward_calculator = MultiObjectiveRewardCalculator(
            initial_budget=self.init_budget, max_history_size=500
        )

        # Limited performance tracking
        self.performance_history = {
            'revenue_history': deque(maxlen=500),
            'capacity_utilization': deque(maxlen=200),
            'risk_adjusted_returns': deque(maxlen=200),
            'market_volatility': deque(maxlen=100)
        }

        # Spaces (ENV exposes BASE dims; wrapper appends forecasts)
        self.observation_spaces = self._create_stable_observation_spaces()
        self.action_spaces = self._create_stable_action_spaces()

        # Market simulation state
        self.market_volatility = 0.0
        self.regulatory_pressure = 0.0
        self.last_revenue = 0.0
        self.market_stress = 0.5

        # --- Preallocate step buffers to avoid per-step allocations ---
        self._obs_buf = {a: np.zeros(self.observation_spaces[a].shape, dtype=np.float32)
                         for a in self.possible_agents}
        self._rew_buf = {a: 0.0 for a in self.possible_agents}
        self._done_buf = {a: False for a in self.possible_agents}
        self._trunc_buf = {a: False for a in self.possible_agents}
        self._info_buf = {a: {} for a in self.possible_agents}

        print(f"✅ RenewableMultiAgentEnv initialized with BASE-only observation spaces")

    def _reset_portfolio_state(self):
        self.wind_capacity = 0.0
        self.solar_capacity = 0.0
        self.hydro_capacity = 0.0
        self.battery_capacity = 0.0
        self.battery_energy = 0.0
        self.budget = self.init_budget
        self.risk_multiplier = 1.0
        self.last_investor_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.investor_reward_buffer = deque(maxlen=144)  # will be resized after reset if needed

    def _create_stable_observation_spaces(self) -> Dict[str, spaces.Box]:
        # BASE-only spaces
        return {agent: self.obs_manager.get_observation_space(agent) for agent in self.possible_agents}

    def _create_stable_action_spaces(self) -> Dict[str, spaces.Box]:
        return {
            "investor_0": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "battery_operator_0": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "risk_controller_0": spaces.Box(low=0.0,  high=2.0, shape=(1,), dtype=np.float32),
            "meta_controller_0": spaces.Box(low=0.0,  high=1.0, shape=(2,), dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        self.memory_manager.cleanup_if_needed(force=True)
        self.t = 0
        self.agents = self.possible_agents[:]
        self._reset_portfolio_state()
        self.investor_reward_buffer = deque(maxlen=self.investment_freq or 144)

        for key in self.performance_history:
            self.performance_history[key].clear()

        if self.enhanced_risk_controller:
            try:
                self.enhanced_risk_controller = EnhancedRiskController(lookback_window=144)
            except Exception as e:
                print(f"⚠️ Risk controller reset failed: {e}")

        self.reward_calculator = MultiObjectiveRewardCalculator(
            initial_budget=self.init_budget, max_history_size=500
        )

        # Fill obs buffer in-place and return it
        obs = self._get_obs_base_only()
        # reset step buffers
        for a in self.possible_agents:
            self._rew_buf[a] = 0.0
            self._done_buf[a] = False
            self._trunc_buf[a] = False
            self._info_buf[a].clear()
        return obs, {}

    def step(self, actions):
        try:
            if self.t % 100 == 0:
                self.memory_manager.cleanup_if_needed()

            if self.t >= self.max_steps:
                return self._create_terminal_state()

            i = self.t  # index into NumPy columns

            # Apply actions (with smoothing & clamps)
            self._process_actions_safe(actions, i)

            # Financials
            financial_metrics = self._calculate_portfolio_performance_safe(i)

            # Budget update
            net_profit = float(financial_metrics.get('net_profit', 0.0))
            try:
                self.budget = max(0.0, float(self.budget) + net_profit)
            except Exception:
                self.budget = max(0.0, float(self.budget))

            # Market conditions
            self._update_market_conditions_safe(i)

            # Update risk controller with latest snapshot (so wrapper can read dynamic risk)
            self._update_risk_controller_state(i, financial_metrics)

            # Performance history
            self._update_performance_history_safe(financial_metrics)

            # Rewards (+ saturation penalties)
            rewards = self._calculate_agent_rewards_safe(financial_metrics, i)

            # Increment time
            self.t += 1

            # Observations (BASE-only) into preallocated buffers
            obs = self._get_obs_base_only()

            # Dones/truncs (no terminal here)
            for a in self.agents:
                self._done_buf[a] = False
                self._trunc_buf[a] = False

            # Populate some infos for diagnostics
            self._info_buf["meta_controller_0"].update({
                "investment_freq": int(self.investment_freq),
                "capital_allocation_fraction": float(self.capital_allocation_fraction),
                "saturation_hits": int(self._last_saturation_count)
            })

            return obs, rewards, self._done_buf, self._trunc_buf, self._info_buf

        except Exception as e:
            print(f"⚠️ Environment step error at t={self.t}: {e}")
            return self._create_safe_step_result()

    # ---------- BASE-ONLY OBS (env returns base; wrapper appends forecasts) ----------

    def _get_obs_base_only(self) -> Dict[str, np.ndarray]:
        """Fill BASE-ONLY observations in-place into self._obs_buf and return it."""
        if self.t >= self.max_steps:
            # use safe mid-bounds
            self._fill_safe_observation_buffers()
            return self._obs_buf

        i = self.t
        # Precompute normalized scalars (single clips only)
        init_div = (self.init_budget / 10.0) if self.init_budget > 0 else 1.0
        normalized_budget = float(np.clip(self.budget / init_div, 0, 10))
        price_n = float(np.clip(self._price[i] / 10.0, 0, 10))
        windf  = float(np.clip(self._wind[i],  0, 1))
        solarf = float(np.clip(self._solar[i], 0, 1))
        hydrof = float(np.clip(self._hydro[i], 0, 1))
        loadf  = float(np.clip(self._load[i],  0, 1))
        riskf  = float(np.clip(self._risk[i],  0, 1))

        # --- investor (6) ---
        inv = self._obs_buf["investor_0"]
        inv[:6] = (windf, solarf, hydrof, price_n, loadf, normalized_budget)
        low, high = self.obs_manager.base_bounds("investor_0")
        np.clip(inv[:6], low, high, out=inv[:6])

        # --- battery (4) ---
        batt = self._obs_buf["battery_operator_0"]
        batt[:4] = (price_n,
                    float(np.clip(self.battery_energy,   0, 10)),
                    float(np.clip(self.battery_capacity, 0, 10)),
                    loadf)
        low, high = self.obs_manager.base_bounds("battery_operator_0")
        np.clip(batt[:4], low, high, out=batt[:4])

        # --- risk (9) ---
        rm = self._get_enhanced_risk_metrics_safe()  # expects 6 floats
        risk_obs = self._obs_buf["risk_controller_0"]
        risk_obs[:3] = (price_n, riskf, normalized_budget)
        risk_obs[3:9] = rm[:6].astype(np.float32)
        low, high = self.obs_manager.base_bounds("risk_controller_0")
        np.clip(risk_obs[:9], low, high, out=risk_obs[:9])

        # --- meta (11) ---
        perf = self._get_performance_metrics_safe().astype(np.float32)  # 4 metrics
        meta = self._obs_buf["meta_controller_0"]
        meta[:7] = (normalized_budget,
                    float(np.clip(self.wind_capacity,    0, 10)),
                    float(np.clip(self.solar_capacity,   0, 10)),
                    float(np.clip(self.hydro_capacity,   0, 10)),
                    float(np.clip(self.battery_capacity, 0, 10)),
                    price_n, riskf)
        meta[7:11] = perf[:4]
        low, high = self.obs_manager.base_bounds("meta_controller_0")
        np.clip(meta[:11], low, high, out=meta[:11])

        return self._obs_buf

    # -------------------------------------------------------------------------------

    def _get_enhanced_risk_metrics_safe(self) -> np.ndarray:
        try:
            if self.enhanced_risk_controller:
                return np.array(self.enhanced_risk_controller.get_risk_metrics_for_observation(),
                                dtype=np.float32)
            else:
                return np.array([0.5, 0.2, 0.3, 0.1, 0.4, 0.25], dtype=np.float32)
        except Exception:
            return np.array([0.5, 0.2, 0.3, 0.1, 0.4, 0.25], dtype=np.float32)

    def _get_performance_metrics_safe(self) -> np.ndarray:
        try:
            portfolio_value = self.budget + (self.wind_capacity + self.solar_capacity + self.hydro_capacity) * 100
            portfolio_performance = float(np.clip(portfolio_value / self.init_budget, 0, 10))
            utilization = (np.mean(list(self.performance_history['capacity_utilization']))
                           if self.performance_history['capacity_utilization'] else 0.0)
            utilization = float(np.clip(utilization, 0, 1))
            risk_adj_return = (np.mean(list(self.performance_history['risk_adjusted_returns']))
                               if self.performance_history['risk_adjusted_returns'] else 0.0)
            risk_adj_return = float(np.clip(risk_adj_return, -10, 10))
            volatility = (np.mean(list(self.performance_history['market_volatility']))
                          if self.performance_history['market_volatility'] else 0.0)
            volatility = float(np.clip(volatility, 0, 10))
            return np.array([portfolio_performance, utilization, risk_adj_return, volatility], dtype=np.float32)
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _fill_safe_observation_buffers(self):
        """Fill obs buffers with safe mid-bounds (BASE dims)"""
        for agent in self.possible_agents:
            low, high = self.obs_manager.base_bounds(agent)
            self._obs_buf[agent][:] = ((low + high) / 2.0).astype(np.float32)

    # ---------- Actions & dynamics ----------

    def _process_actions_safe(self, actions: Dict[str, np.ndarray], i: int):
        try:
            # Validate & clip
            validated_actions = {}
            for agent, action in actions.items():
                if agent in self.action_spaces:
                    action_space = self.action_spaces[agent]
                    arr = np.array(action, dtype=np.float32).flatten()
                    # pad/truncate to expected size
                    if arr.shape[0] != action_space.shape[0]:
                        if arr.shape[0] < action_space.shape[0]:
                            pad_val = ((action_space.low + action_space.high) / 2.0).astype(np.float32)
                            pad = np.full(action_space.shape[0] - arr.shape[0],
                                          pad_val[0] if pad_val.ndim else float(pad_val), dtype=np.float32)
                            arr = np.concatenate([arr, pad])
                        else:
                            arr = arr[:action_space.shape[0]]
                    arr = np.clip(arr, action_space.low, action_space.high)
                    validated_actions[agent] = arr

            # Meta (with smoothing + bounded mapping)
            meta_action = validated_actions.get("meta_controller_0", np.array([0.5, 0.1], dtype=np.float32))
            self._last_meta_raw = meta_action.copy()

            # Map to targets
            target_freq = float(self.META_FREQ_MIN + meta_action[0] * (self.META_FREQ_MAX - self.META_FREQ_MIN))
            target_cap  = float(self.META_CAP_MIN + meta_action[1] * (self.META_CAP_MAX - self.META_CAP_MIN))

            # EMA smoothing
            beta = self.META_SMOOTH_BETA
            self.investment_freq = int(np.clip(beta * self.investment_freq + (1.0 - beta) * target_freq,
                                               self.META_FREQ_MIN, self.META_FREQ_MAX))
            self.capital_allocation_fraction = float(np.clip(
                beta * self.capital_allocation_fraction + (1.0 - beta) * target_cap,
                self.META_CAP_MIN, self.META_CAP_MAX
            ))

            # Risk (0..2)
            risk_action = validated_actions.get("risk_controller_0", np.array([1.0], dtype=np.float32))[0]
            self.risk_multiplier = float(np.clip(risk_action, 0.0, 2.0))

            # Investor (frequency-gated)
            if self.t % max(1, self.investment_freq) == 0:
                investor_action = validated_actions.get("investor_0", np.array([0.0, 0.0, 0.0], dtype=np.float32))
                self.last_investor_action = np.clip(investor_action, -1.0, 1.0)

            self._execute_investments_safe()

            # Battery
            battery_action = validated_actions.get("battery_operator_0", np.array([0.0], dtype=np.float32))[0]
            self._execute_battery_operations_safe(battery_action, i)

            # Track saturation hits for penalty
            self._last_saturation_count = self._count_saturation_hits(meta_action, self.last_investor_action)

        except Exception as e:
            print(f"⚠️ Action processing error: {e}")

    def _count_saturation_hits(self, meta_action: np.ndarray, investor_action: np.ndarray) -> int:
        """Count how many components are near their bounds (for small regularization)."""
        cnt = 0
        # investor in [-1, 1] → near ±1 counts
        if investor_action is not None and investor_action.size > 0:
            cnt += int(np.sum(np.abs(investor_action) >= (1.0 - self.SAT_EPS)))
        # meta raw in [0,1] near 0 or 1
        cnt += int(meta_action[0] <= self.SAT_EPS or meta_action[0] >= (1.0 - self.SAT_EPS))
        cnt += int(meta_action[1] <= self.SAT_EPS or meta_action[1] >= (1.0 - self.SAT_EPS))
        # also if clamped ends reached
        cnt += int(self.investment_freq <= self.META_FREQ_MIN + 1 or self.investment_freq >= self.META_FREQ_MAX - 1)
        cnt += int(self.capital_allocation_fraction <= self.META_CAP_MIN + 1e-6 or
                   self.capital_allocation_fraction >= self.META_CAP_MAX - 1e-6)
        return int(cnt)

    def _execute_investments_safe(self):
        """
        Execute investor allocations. Ties risk_multiplier to leverage:
        - Higher risk_multiplier → more conservative (lower allocation cap).
        """
        try:
            invest_action = np.clip((self.last_investor_action + 1.0) / 2.0, 0, 1)
            wind_invest, solar_invest, hydro_invest = invest_action
            total_invest = wind_invest + solar_invest + hydro_invest

            if total_invest > 0 and self.budget > 0:
                available_budget = max(0.0, self.budget * 0.9)

                base_alloc = available_budget * self.capital_allocation_fraction
                base_alloc = min(base_alloc, self.budget * 0.5)

                # risk scaling
                if self.risk_multiplier >= 1.0:
                    scaled_alloc = base_alloc / max(1e-6, self.risk_multiplier)
                else:
                    scaled_alloc = base_alloc * min(1.25, 1.0 / max(self.risk_multiplier, 1e-3))

                allocated = float(np.clip(scaled_alloc, 0.0, self.budget))

                wind_share  = wind_invest / total_invest
                solar_share = solar_invest / total_invest
                hydro_share = hydro_invest / total_invest

                wind_investment  = min(allocated * wind_share, self.budget)
                solar_investment = min(allocated * solar_share, max(0.0, self.budget - wind_investment))
                hydro_investment = min(allocated * hydro_share,  max(0.0, self.budget - wind_investment - solar_investment))

                self.wind_capacity  += wind_investment  / self.wind_capex
                self.solar_capacity += solar_investment / self.solar_capex
                self.hydro_capacity += hydro_investment / self.hydro_capex

                total_spent = wind_investment + solar_investment + hydro_investment
                self.budget = max(0.0, self.budget - total_spent)

                self.wind_capacity   = float(np.clip(self.wind_capacity,   0, 10000))
                self.solar_capacity  = float(np.clip(self.solar_capacity,  0, 10000))
                self.hydro_capacity  = float(np.clip(self.hydro_capacity,  0, 10000))

        except Exception as e:
            print(f"⚠️ Investment execution error: {e}")

    def _execute_battery_operations_safe(self, battery_action: float, i: int):
        try:
            if self.battery_capacity == 0:
                min_battery_investment = min(self.init_budget * 0.01, self.budget * 0.1)
                self.battery_capacity = min_battery_investment / self.battery_capex

            wind_power  = self.wind_capacity  * float(np.clip(self._wind[i],  0, 1))
            solar_power = self.solar_capacity * float(np.clip(self._solar[i], 0, 1))
            hydro_power = self.hydro_capacity * float(np.clip(self._hydro[i], 0, 1))
            total_gen = wind_power + solar_power + hydro_power

            # normalize action [-1,1] -> [0,1]
            battery_use = float(np.clip((battery_action + 1.0) / 2.0, 0, 1))
            if battery_use > 0.5:
                discharge_rate = (battery_use - 0.5) * 2
                discharge = min(discharge_rate * self.battery_capacity * 0.1, self.battery_energy)
                self.battery_energy = max(0.0, self.battery_energy - discharge)
                total_gen += discharge
            else:
                charge_rate = (0.5 - battery_use) * 2
                max_charge = min(charge_rate * self.battery_capacity * 0.1,
                                 total_gen * 0.5,
                                 self.battery_capacity - self.battery_energy)
                charge = max(0.0, max_charge)
                self.battery_energy = min(self.battery_capacity, self.battery_energy + charge)
                total_gen = max(0.0, total_gen - charge)

            self.battery_energy   = float(np.clip(self.battery_energy, 0, self.battery_capacity))
            self.battery_capacity = float(np.clip(self.battery_capacity, 0, 10000))

        except Exception as e:
            print(f"⚠️ Battery operation error: {e}")

    def _calculate_portfolio_performance_safe(self, i: int) -> Dict[str, float]:
        try:
            wind_power  = float(np.clip(self.wind_capacity  * self._wind[i],  0, 10000))
            solar_power = float(np.clip(self.solar_capacity * self._solar[i], 0, 10000))
            hydro_power = float(np.clip(self.hydro_capacity * self._hydro[i], 0, 10000))
            total_gen = wind_power + solar_power + hydro_power

            price = float(np.clip(self._price[i], 0, 1000))
            revenue = total_gen * price * (10.0 / 60.0)  # 10-minute intervals

            total_capacity = self.wind_capacity + self.solar_capacity + self.hydro_capacity
            base_opex = total_capacity * 0.02
            battery_opex = self.battery_capacity * 0.01
            total_costs = base_opex + battery_opex

            net_profit = revenue - total_costs

            capacity_utilization = (total_gen / (total_capacity + 1e-8)) if total_capacity > 0 else 0.0
            capacity_utilization = float(np.clip(capacity_utilization, 0, 1))

            battery_efficiency = float(np.clip(self.battery_energy / (self.battery_capacity + 1e-8), 0, 1))

            self.last_revenue = revenue

            return {
                'revenue': revenue,
                'net_profit': net_profit,
                'costs': total_costs,
                'total_generation': total_gen,
                'renewable_generation': total_gen,
                'capacity_utilization': capacity_utilization,
                'battery_efficiency': battery_efficiency,
                'portfolio_var': 0.0,
                'market_volatility': self.market_volatility
            }

        except Exception as e:
            print(f"⚠️ Performance calculation error: {e}")
            return {
                'revenue': 0.0, 'net_profit': 0.0, 'costs': 0.0,
                'total_generation': 0.0, 'renewable_generation': 0.0,
                'capacity_utilization': 0.0, 'battery_efficiency': 0.0,
                'portfolio_var': 0.0, 'market_volatility': 0.0
            }

    def _update_market_conditions_safe(self, i: int):
        try:
            if len(self.performance_history['revenue_history']) > 10:
                recent_revenues = list(self.performance_history['revenue_history'])[-10:]
                if len(recent_revenues) > 1:
                    returns = np.diff(recent_revenues) / (np.array(recent_revenues[:-1]) + 1e-8)
                    self.market_volatility = float(np.clip(np.std(returns), 0, 1))
            self.market_stress = float(np.clip(self._risk[i], 0, 1))
        except Exception:
            self.market_volatility = 0.2
            self.market_stress = 0.5

    def _update_risk_controller_state(self, i: int, financial_metrics: Dict[str, float]):
        """Best-effort: push latest snapshot to risk controller for dynamic metrics."""
        try:
            if not self.enhanced_risk_controller:
                return
            price = float(self._price[i])
            price_prev = float(self._price[i - 1]) if i > 0 else price
            price_ret = (price - price_prev) / (price_prev + 1e-8)

            payload = {
                'price': price,
                'price_return': float(price_ret),
                'budget': float(self.budget),
                'net_profit': float(financial_metrics.get('net_profit', 0.0)),
                'revenue': float(financial_metrics.get('revenue', 0.0)),
                'capacity': float(self.wind_capacity + self.solar_capacity + self.hydro_capacity),
                'market_stress': float(self.market_stress),
                'volatility_proxy': float(self.market_volatility),
                'timestep': int(self.t),
            }

            # Be tolerant to controller API shape
            if hasattr(self.enhanced_risk_controller, "update_observation"):
                try:
                    self.enhanced_risk_controller.update_observation(payload)
                except Exception:
                    pass
            if hasattr(self.enhanced_risk_controller, "update_risk_history"):
                try:
                    self.enhanced_risk_controller.update_risk_history(payload)
                except Exception:
                    pass
        except Exception:
            pass

    def _update_performance_history_safe(self, financial_metrics: Dict[str, float]):
        try:
            self.performance_history['revenue_history'].append(financial_metrics['revenue'])
            self.performance_history['capacity_utilization'].append(financial_metrics['capacity_utilization'])
            if self.init_budget > self.budget:
                risk_adj_return = financial_metrics['net_profit'] / (self.init_budget - self.budget)
                self.performance_history['risk_adjusted_returns'].append(float(np.clip(risk_adj_return, -10, 10)))
            self.performance_history['market_volatility'].append(self.market_volatility)
        except Exception as e:
            print(f"⚠️ Performance history update error: {e}")

    def _calculate_agent_rewards_safe(self, financial_metrics: Dict[str, float], i: int) -> Dict[str, float]:
        try:
            # Risk assessment scaled by risk_multiplier
            risk_assessment = self._get_risk_assessment_safe(i)
            scaled_overall = float(np.clip(risk_assessment.get('overall_risk', 0.5) * self.risk_multiplier, 0.0, 2.0))
            risk_assessment_scaled = dict(risk_assessment)
            risk_assessment_scaled['overall_risk'] = scaled_overall

            env_state = {
                'budget': self.budget,
                'wind_capacity': self.wind_capacity,
                'solar_capacity': self.solar_capacity,
                'hydro_capacity': self.hydro_capacity,
                'battery_capacity': self.battery_capacity,
                'market_volatility': self.market_volatility
            }

            base_reward, r = self.reward_calculator.calculate_multi_objective_reward(
                env_state, financial_metrics, risk_assessment_scaled
            )

            # Small regularization to discourage saturation at bounds
            sat_penalty = -0.001 * float(self._last_saturation_count)

            investor_reward = base_reward * 0.8 + r['financial'] * 0.2 + sat_penalty
            self.investor_reward_buffer.append(investor_reward)
            smoothed_investor_reward = (np.mean(list(self.investor_reward_buffer))
                                        if self.investor_reward_buffer else 0.0)

            battery_reward = (r['efficiency'] * 0.5 + r['financial'] * 0.3 + base_reward * 0.2)
            risk_reward = (r['risk_management'] * 0.6 + (1.0 - scaled_overall) * 0.4)
            meta_reward = base_reward + r['diversification'] * 0.2 + sat_penalty

            # write rewards into reusable buffer
            self._rew_buf["investor_0"] = float(np.clip(smoothed_investor_reward, -10, 10))
            self._rew_buf["battery_operator_0"] = float(np.clip(battery_reward, -10, 10))
            self._rew_buf["risk_controller_0"] = float(np.clip(risk_reward, -10, 10))
            self._rew_buf["meta_controller_0"] = float(np.clip(meta_reward, -10, 10))
            return self._rew_buf

        except Exception as e:
            print(f"⚠️ Reward calculation error: {e}")
            return {agent: 0.0 for agent in self.agents}

    def _get_risk_assessment_safe(self, i: int) -> Dict[str, float]:
        try:
            if self.enhanced_risk_controller:
                env_state = {
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
                    'market_stress': self.market_stress
                }
                self.enhanced_risk_controller.update_risk_history(env_state)
                return self.enhanced_risk_controller.calculate_comprehensive_risk(env_state)
            else:
                return {
                    'market_risk': 0.3, 'operational_risk': 0.2, 'portfolio_risk': 0.3,
                    'liquidity_risk': 0.1, 'regulatory_risk': 0.2, 'overall_risk': 0.25
                }
        except Exception:
            return {
                'market_risk': 0.5, 'operational_risk': 0.3, 'portfolio_risk': 0.4,
                'liquidity_risk': 0.2, 'regulatory_risk': 0.3, 'overall_risk': 0.35
            }

    # ---------- Adaptive reward weighting (optional, used by callback) ----------
    def adapt_reward_weights(self):
        """Lightweight adaptation based on market stress/volatility."""
        try:
            stress = float(self.market_stress)
            vol = float(self.market_volatility)
            w = self.reward_calculator.reward_weights

            # tilt towards risk in high stress, towards financial when calm
            delta = 0.01
            if stress > 0.6 or vol > 0.4:
                w['risk_management'] = min(0.35, w['risk_management'] + delta)
                w['financial'] = max(0.25, w['financial'] - delta)
            else:
                w['financial'] = min(0.45, w['financial'] + delta)
                w['risk_management'] = max(0.15, w['risk_management'] - delta)

            # renormalize
            s = sum(w.values())
            for k in w:
                w[k] = float(w[k] / s)
        except Exception:
            pass

    def get_reward_analysis(self):
        """Simple console summary for post-run analysis."""
        try:
            w = self.reward_calculator.reward_weights
            print("📊 Reward Weights:", {k: round(v, 3) for k, v in w.items()})
            if self.performance_history['financial_scores']:
                print("   Avg financial:", round(float(np.mean(self.reward_calculator.performance_history['financial_scores'])), 4))
            if self.performance_history['risk_scores']:
                print("   Avg risk:", round(float(np.mean(self.reward_calculator.performance_history['risk_scores'])), 4))
        except Exception:
            pass

    # ---------- Terminal / error fallbacks ----------

    def _create_terminal_state(self):
        # Fill safe obs midpoints once
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

    # ---------- API helpers expected by the wrapper ----------

    def observation_space(self, agent):
        # BASE-only space (wrapper appends forecasts)
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_base_observation_dim(self, agent: str) -> int:
        return self.obs_manager.observation_specs[agent]['base_dim']

    def _get_forecast_dimension(self, agent: str) -> int:
        return self.obs_manager.observation_specs[agent]['forecast_dim']

    def _get_performance_dimension(self, agent: str) -> int:
        """Wrapper must NOT add performance separately; part of base."""
        return 0

    def __del__(self):
        try:
            self.memory_manager.cleanup_if_needed(force=True)
        except Exception:
            pass


# ============================================================================
# TESTING FUNCTION - verifies BASE-only obs match env spaces (wrapper adds forecasts)
# ============================================================================
def test_observation_dimensions():
    """Test that environment returns BASE-only observations matching env spaces"""
    print("🧪 Testing observation dimensions (BASE-only)...")

    sample_data = pd.DataFrame({
        'wind': [0.5, 0.6, 0.4],
        'solar': [0.3, 0.8, 0.2],
        'hydro': [0.9, 0.7, 0.8],
        'price': [60, 55, 70],
        'load': [0.7, 0.6, 0.8],
        'risk': [0.3, 0.4, 0.2]
    })

    try:
        env = RenewableMultiAgentEnv(sample_data)
        obs, _ = env.reset()

        print("✅ Environment created successfully")

        for agent in env.possible_agents:
            expected_shape = env.observation_space(agent).shape  # BASE dims
            actual_shape = obs[agent].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"❌ {agent}: observation shape mismatch! "
                    f"Expected {expected_shape}, got {actual_shape}"
                )
            else:
                print(f"✅ {agent}: {actual_shape} matches expected {expected_shape}")

        # Test a few steps
        for i in range(3):
            actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
            obs, rewards, dones, truncs, infos = env.step(actions)

            for agent in env.possible_agents:
                expected_shape = env.observation_space(agent).shape
                actual_shape = obs[agent].shape
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"❌ Step {i}, {agent}: observation shape mismatch! "
                        f"Expected {expected_shape}, got {actual_shape}"
                    )

        print("🎉 All BASE-only observation dimensions match across multiple steps!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_observation_dimensions()
