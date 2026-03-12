# Tier1 MARL: Deep Analysis for a Healthy Trading System

## 1. What Does "Healthy Trading" Mean for Tier1?

For a **single-price hedging** MARL system where the investor trades instruments that all follow the same price, a healthy system should:

| Criterion | Meaning |
|-----------|---------|
| **Directional responsiveness** | Policy goes long when price is expected to rise, short when it falls — not stuck in one sign |
| **Risk-adjusted sizing** | Position size adapts to volatility, drawdown, and capital availability |
| **Hedging alignment** | Trading sleeve offsets operational revenue volatility (negative correlation with generation) |
| **Credit assignment** | Rewards flow correctly to the actions that caused them |
| **Regime robustness** | Behavior remains sensible across episodes, price regimes, and market stress |
| **No collapse** | Policy does not saturate at ±1 or degenerate to constant output |

---

## 2. Architecture Overview (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ OBSERVATIONS (6D investor)                                                   │
│ [price_n, budget_n, wind_pos_norm, solar_pos_norm, hydro_pos_norm,         │
│  mtm_pnl_norm]                                                               │
│                                                                              │
│ price_n: z-score clipped / 3 → [-1,1]  (level, NOT momentum)                 │
│ positions: exposure / max_pos → [-1,1]                                       │
│ mtm_pnl_norm: cumulative_mtm / init_budget → [-1,1]                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ POLICY (PPO + ClampedActorCriticPolicy)                                      │
│ - 1D action: exposure_raw ∈ [-1, 1]                                          │
│ - mean_clip=0.60, log_std ∈ [-2.2, -1.2], gSDE, latent_clip=0.15            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPOSURE MAPPING                                                             │
│ exposure = sign(raw) * |raw|^power  (power=1.0 → linear)                     │
│ alloc = exposure × risk_budget_weights  [0.40 wind, 0.35 solar, 0.25 hydro]  │
│ target_pos = alloc × (tradeable_capital × max_position_size)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXECUTION (investment_freq=6, i.e. every 1h)                                │
│ - MTM added BEFORE trades (correct attribution)                              │
│ - Trades executed to reach target positions                                   │
│ - Transaction costs, MTM loss exits (6% threshold)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ REWARDS                                                                      │
│ Base (40% to investor): operational + risk + hedging + nav_stability         │
│ Investor-specific: mtm_delta + trading_return + quality - drawdown - penalties│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Design Strengths (What Is Already Healthy)

### 3.1 Correct MTM and Trade Ordering
- **MTM before trades** (`_add_mtm_for_step` before `_execute_investor_trades`): PnL is attributed to positions held during the price move. This is correct and avoids reward leakage.

### 3.2 Single-Price MTM Model
- All three instruments use the **same** `price_return` for MTM (`financial_engine.calculate_mtm_pnl_from_exposure`). This matches the design: wind/solar/hydro instruments follow the same price. MTM = Σ(exposure_i × price_return).

### 3.3 Realistic Position Sizing
- `tradeable_capital = budget × capital_allocation × risk_mult × vol_brake × strategy_mult`
- `max_pos_size = tradeable_capital × max_position_size` (0.35)
- Volatility brake reduces size when recent vol > 2.5× median.
- Drawdown gates reduce or disable trading (soft/medium/hard thresholds).

### 3.4 Risk Budget Allocation
- Wind 40%, solar 35%, hydro 25% — reflects relative volatility. Exposure is not equally split; it respects asset risk.

### 3.5 Investor Local Rewards
- `investor_trading_return_weight`, `investor_trading_quality_weight`, `investor_trading_drawdown_weight` create a local trading sleeve objective. The investor is not purely a shared-reward hedge; it has direct trading PnL feedback.

### 3.6 Hedging Effectiveness Reward
- `hedging_score = -corr(ops_returns, trading_returns)`. Negative correlation is rewarded. This aligns with the design: trading should offset operational risk.

### 3.7 Transaction Costs and MTM Loss Exits
- Transaction costs and 6% MTM loss exit threshold prevent runaway positions and encourage risk-aware behavior.

### 3.8 Global Normalization
- `use_global_normalization = True` with fixed global mean/std and p95 scales. Observation distribution is stable across episodes, reducing episode-boundary collapse.

---

## 4. Critical Gaps (What Makes It Unhealthy)

### 4.1 **Observation: No Price Direction / Momentum**

**Current:** `price_n` is the **level** (z-score of current price). It does not encode direction.

**Problem:** To decide "long vs short," the policy needs to know whether price is rising or falling. A level of 0.5 could mean:
- Price is high and rising (→ long)
- Price is high and falling (→ short)
- Price is high and flat (→ neutral)

**Impact:** The policy cannot reliably infer direction from level alone. It may latch onto spurious correlations (e.g., "high price → long" in a trending regime) and become one-sided.

**Gap:** Missing `price_return_1step` or `price_momentum` (e.g., 6-step return) in the investor observation.

---

### 4.2 **Reward: Base Reward Dominated by NAV, Not Trading**

**Base reward weights:**
- operational_revenue: 42%
- risk_management: 15%
- hedging_effectiveness: 15%
- nav_stability: 28%

**NAV composition:** NAV = trading_cash + physical_book_value + accumulated_operational_revenue + financial_mtm.

**Problem:** Physical assets are ~88% of the fund. Operational revenue and physical book value dominate NAV changes. The trading sleeve (12% of capital) contributes a small fraction of NAV volatility. So:
- `nav_stability_score` is driven mostly by operational and physical book, not trading.
- The investor's 40% share of base reward is weakly sensitive to its own trading decisions.

**Impact:** The investor may get stronger gradient signal from `investor_trading_return_delta` and `investor_mtm_delta` than from the base reward. If those local terms are noisy or biased early, the policy can drift.

---

### 4.3 **Reward: Sparse and Delayed Trading Feedback**

**Trading cadence:** `investment_freq = 6` (every 1 hour). The investor acts every 6 steps.

**MTM:** `last_realized_investor_dnav_return` is the 1-step return on the previous exposure. It is computed at each step, but the investor only acts every 6 steps.

**Problem:** The reward for "action at t" is the MTM from t to t+1, t+1 to t+2, ... up to the next decision. The policy sees a delayed, diluted signal. Credit assignment over 6 steps is harder than over 1 step.

**Impact:** Early in training, the policy may not learn a clear causal link between action and outcome. Random exploration can get reinforced if the first few outcomes happen to favor one direction.

---

### 4.4 **Hedging Alignment: Correlation vs Direction**

**Hedging score:** Rewards negative correlation between operational and trading returns.

**Single-price design:** All instruments move with the same price. So:
- Long exposure → profit when price rises, loss when it falls.
- Operational revenue (generation × price) → profit when price rises (more revenue), loss when it falls.

**Implication:** Operational and trading returns are **positively** correlated when both are long (both profit from rising price). So the "hedge" requires the investor to go **short** when operational exposure is long (physical generation). The reward correctly encourages negative correlation.

**Gap:** The investor does not observe operational exposure or generation directly. It only sees price, positions, and MTM. The policy must infer "how to hedge" from price and NAV history. Without explicit operational exposure, this is an indirect learning problem.

---

### 4.5 **No Anti-Collapse Penalty (One-Sign Drift)**

**Current:** `investor_mean_collapse_penalty = 0.0`. The penalty for prolonged one-sided mean is disabled.

**Problem:** When the policy mean stays on one side (e.g., always long) for many steps, there is no penalty. PPO gradients can reinforce this if early outcomes favor that direction.

**Impact:** Directly contributes to `inv_mu_sign_consistency = 1.0` — no sign flipping.

---

### 4.6 **Policy Mean Clip Allows Strong Bias**

**Current:** `investor_ppo_mean_clip = 0.60`. The raw mean can be clamped to ±0.6 before tanh.

**Problem:** tanh(0.6) ≈ 0.54. A policy that consistently outputs 0.6 produces a mean of ~0.54. So inv_mu_abs_roll ≈ 0.535 is consistent with a policy that has learned to output near the clip boundary.

**Impact:** The clip does not prevent moderate bias; it just caps the maximum. A biased policy can sit at the clip.

---

### 4.7 **Regime / Data Sensitivity**

**Episode structure:** Each episode is a 6-month data slice. If ep0 has a strong trend (e.g., mostly rising prices), the policy may learn "always long" as a heuristic.

**Global normalization:** Helps but does not remove regime bias. The policy still sees a sequence of observations that may be skewed.

**Gap:** No explicit "regime" or "trend" feature. The policy must infer regime from the 6D observation history, which is limited.

---

### 4.8 **Budget vs Trading Capital Confusion**

**Observation:** `budget_n = budget / init_budget`. The investor sees `budget`, which is trading cash.

**Execution:** `tradeable_capital = budget × capital_allocation × ...`. So the investor's action is scaled by available trading cash.

**Potential issue:** If `budget` is low (e.g., after drawdown or distribution), the investor may still output large exposure_raw. The environment scales it down by `tradeable_capital`, so the executed trade is small. The policy may not "see" that its effective size was reduced — it only observes `budget_n` and position norms. The reward may be small (because the trade was small), but the policy might not learn the causal link clearly.

---

## 5. Reward Flow Deep Dive

### 5.1 Investor Reward Components

```
investor_reward = investor_base
                + investor_mtm_delta
                + investor_trading_return_delta
                + investor_trading_quality_delta
                + investor_strategy_delta
                - investor_action_penalty
                - investor_action_boundary_penalty
                - investor_exposure_penalty
                - investor_exposure_stuck_penalty
                - investor_mean_collapse_penalty
                - investor_trading_drawdown_penalty
```

**Weights (config):**
- `investor_base_reward_weight = 0.40` → 40% of base (NAV-driven)
- `investor_mtm_reward_weight = 0.08`, scale 5000, clip 1.0
- `investor_trading_return_weight = 0.45`, scale 0.002, clip 2.0
- `investor_trading_quality_weight = 0.25`, clip 2.0
- `investor_trading_drawdown_weight = 0.20`, scale 0.05

**All penalties:** Currently 0 (boundary, exposure, stuck, mean_collapse disabled).

### 5.2 Trading Return Delta

```python
realized_investor_return = last_realized_investor_dnav_return  # MTM / exposure
investor_trading_return_delta = 0.45 * clip(realized_investor_return / 0.002, -2, 2)
```

So a 0.2% (0.002) realized return → 0.45 full contribution. A 0.4% return → clipped at 2. The scale is sensitive to small returns.

### 5.3 MTM Delta

```python
investor_mtm_delta = 0.08 * clip(mtm_pnl / 5000, -1, 1)
```

So 5000 DKK MTM → 0.08 contribution. 50000 DKK → clipped at 1. For a ~5B DKK fund, 5000 DKK is 0.0001% of NAV. The MTM term is scaled for small absolute PnL.

### 5.4 Quality (Sharpe-like)

```python
_investor_local_quality = recent_mean / max(recent_vol, vol_floor)
```

Quality is a Sharpe-like ratio. High mean with low vol → positive; high mean with high vol → damped. This encourages risk-adjusted returns.

---

## 6. Health Checklist

| Check | Status | Notes |
|-------|--------|------|
| MTM before trades | ✅ | Correct ordering |
| Single-price MTM | ✅ | All instruments use same price_return |
| Position sizing | ✅ | Risk, vol brake, drawdown gates |
| Risk budget | ✅ | Wind/solar/hydro weighted |
| Transaction costs | ✅ | Applied |
| MTM loss exits | ✅ | 6% threshold |
| Global normalization | ✅ | Stable across episodes |
| Price direction in obs | ❌ | Only level, no momentum |
| Investor base reward sensitivity | ⚠️ | Weak; NAV dominated by physical |
| Trading feedback delay | ⚠️ | 6-step cadence |
| Anti-collapse penalty | ❌ | Disabled |
| Operational exposure in obs | ❌ | Not explicit |
| Regime/trend feature | ❌ | Not present |

---

## 7. Recommendations for a Healthy Tier1

### 7.1 High Priority (Core Health)

1. **Add price momentum to investor obs**  
   Extend to 7D: `price_return_1step` or `(price_t - price_{t-6}) / scale`. Clipped to [-1,1]. This gives the policy a direct directional signal.

2. **Enable mean-collapse penalty**  
   Set `investor_mean_collapse_penalty = 0.15` with existing thresholds. Penalizes prolonged one-sided mean.

3. **Lower mean clip**  
   `investor_ppo_mean_clip = 0.45` to reduce the maximum bias the policy can express early.

### 7.2 Medium Priority (Robustness)

4. **Concave exposure mapping**  
   `investor_exposure_power = 0.7` so interior actions still produce meaningful exposure and the policy is less incentivized to push to ±1.

5. **Optional: Operational exposure proxy**  
   Add a normalized "operational exposure" or "generation vs load" proxy to the investor obs so the policy can learn hedging more directly.

6. **Reward scaling audit**  
   Verify that `investor_trading_return_scale` and `investor_mtm_reward_scale` produce gradients of similar magnitude to the base reward. If local terms dominate too early, consider reducing them or increasing base sensitivity to trading.

### 7.3 Lower Priority (Polish)

7. **Regime feature**  
   Add a simple trend indicator (e.g., sign of 24-step return) so the policy can condition on regime.

8. **Episode shuffling / augmentation**  
   If ep0 data has strong trend bias, consider varying episode order or augmenting with different regimes.

---

## 8. Summary

**Strengths:**  
- Correct MTM and trade ordering, single-price model, realistic sizing, risk budget, drawdown gates, global normalization.  
- Investor has local trading rewards (return, quality, drawdown) and hedging effectiveness in the base reward.

**Gaps:**  
- No price direction in observations; base reward weakly sensitive to trading; 6-step feedback delay; no anti-collapse penalty; mean clip allows moderate bias.  
- These gaps favor one-sign drift and make it harder for the policy to learn true directional hedging.

**Path to health:**  
- Add direction (momentum) to observations, enable mean-collapse penalty, lower mean clip, and optionally use concave exposure mapping.  
- These changes target the root causes of one-sign drift while keeping the design and Tier2 compatibility intact.
