# Tier1/Tier2 Deep Analysis: Should the Investor Have Only Trading Profit Reward?

## Executive Summary

**No.** The investor should **not** have only the trading profit reward. The design requires a **hedging incentive** (negative correlation with operational revenue) that pure profit maximization does not provide. A simplified reward structure is recommended: **trading profit as primary + hedging alignment + risk penalties**, with the shared base reward reduced or replaced by explicit terms.

---

## 1. Current Investor Reward Structure

### 1.1 All Components (environment.py _assign_rewards)

| Component | Weight/Scale | Source | Purpose |
|-----------|--------------|--------|---------|
| **investor_base** | 0.40 × base_reward | ProfitFocusedRewardCalculator | Shared fund objective (NAV, ops, risk, hedging) |
| **investor_mtm_delta** | 0.08 × clip(mtm/5000, ±1) | mtm_pnl | Per-step MTM PnL |
| **investor_trading_profit_delta** | 0.35 × clip(mtm/3000, ±1.5) | mtm_pnl | Same signal, different scale (redundant) |
| **investor_trading_return_delta** | 0.45 × clip(return/0.002, ±2) | last_realized_investor_dnav_return | Rolling realized return |
| **investor_trading_quality_delta** | 0.25 × clip(quality, ±2) | _investor_local_quality (Sharpe-like) | Risk-adjusted return |
| **investor_strategy_delta** | 0 | — | Disabled |
| **investor_trading_drawdown_penalty** | 0.20 × clip(dd/0.05, 0, 2) | _investor_local_drawdown | Sleeve drawdown |
| **investor_mean_collapse_penalty** | 0 | — | Disabled |

### 1.2 Base Reward Composition (ProfitFocusedRewardCalculator)

```
base_reward = 0.42×operational_revenue + 0.15×risk_management + 0.15×hedging_effectiveness + 0.28×nav_stability
```

- **operational_revenue**: Cash flow from physical generation
- **risk_management**: -(volatility_penalty + drawdown_penalty)
- **hedging_effectiveness**: **-corr(ops_returns, trading_returns)** ← critical for design
- **nav_stability**: NAV growth, volatility, drawdown

### 1.3 Hedging Logic (environment.py _calculate_hedging_effectiveness)

```python
hedging_score = -corr(ops_returns_norm, trading_returns_norm)
```

- **Negative correlation** = good (trading offsets operational risk)
- **Positive correlation** = bad (trading amplifies operational risk)

**Implication:** Physical generation profits when price rises (more revenue). To hedge, the investor must go **short** when operational exposure is effectively long. Pure profit maximization would favor going long when price rises—the opposite of hedging.

---

## 2. What Happens If Investor Has ONLY Trading Profit?

### 2.1 Pure Profit Maximization

- **Objective:** Maximize MTM PnL (or realized return).
- **Behavior:** Go long when price expected to rise, short when expected to fall.
- **Problem:** This is **directional trading**, not hedging. When price rises:
  - Operational revenue rises (generation × price)
  - Trading PnL rises if long
  - **Positive correlation** → hedging_score negative → base reward would penalize, but with no base reward the investor has no hedging incentive.

### 2.2 Loss of Hedging Alignment

The fund design is **single-price hedging**: instruments follow the same price as physical generation. The trading sleeve should **offset** operational volatility. Without a hedging term:

- The investor learns to trade for profit only.
- It may consistently go long in rising markets (profitable) but worsen fund-level volatility.
- The 88% physical / 12% trading split means trading can still materially affect NAV when misaligned.

### 2.3 Loss of MARL Coordination

- **Risk controller** reduces position size on drawdown; investor reward is unaffected.
- **Meta controller** allocates capital; it uses `investor_meta_signal` (mtm + return + quality - drawdown). If investor ignores fund health, meta gets a distorted signal.
- **Battery** arbitrages; no direct coupling, but fund-level NAV affects all agents.

### 2.4 Reward Sparsity and Noise

- MTM PnL is very noisy (price moves every step).
- Pure profit can produce sparse, high-variance gradients.
- Quality and drawdown terms provide smoothing and risk-awareness.

---

## 3. Tier2 DL Enhancer Independence

**Tier2 does not depend on Tier1 reward structure.**

- **Tier2** learns from **realized investor sleeve return** over decision intervals.
- Target: `_compute_enhancer_delta_target` uses `interval_score` (return, drawdown, Sharpe).
- The enhancer trains on **outcomes**, not on the Tier1 reward.
- Changing Tier1 investor reward does **not** affect Tier2 training.

---

## 4. Recommended Reward Structure

### 4.1 Option A: Simplified Mixed (Recommended)

Replace the current mix with a **clear, minimal** structure:

| Term | Weight | Purpose |
|------|--------|---------|
| **Trading profit** | 0.50 | Primary: maximize sleeve PnL |
| **Hedging alignment** | 0.25 | Explicit: reward -corr(ops, trading) |
| **Drawdown penalty** | 0.20 | Risk: penalize sleeve drawdown |
| **Quality (Sharpe-like)** | 0.15 | Risk-adjusted return |
| **Base (minimal)** | 0.10 | Light fund-level anchor |

**Implementation:** Add `investor_hedging_reward_weight` and scale hedging_score from the reward calculator. Reduce `investor_base_reward_weight` to 0.10. Remove redundant `investor_mtm_delta` (keep only `investor_trading_profit_delta`).

### 4.2 Option B: Pure Profit + Explicit Hedging Bonus

| Term | Weight | Purpose |
|------|--------|---------|
| **Trading profit** | 0.60 | Primary |
| **Hedging bonus** | 0.30 | Explicit -corr term |
| **Drawdown penalty** | 0.20 | Risk |
| **Base** | 0 | Remove |

**Implementation:** `investor_base_reward_weight = 0`. Add standalone `investor_hedging_bonus = weight * hedging_score` from the calculator.

### 4.3 Option C: Keep Base, Simplify Redundancy

- **Remove** `investor_mtm_delta` (redundant with `investor_trading_profit_delta`).
- **Keep** `investor_base_reward_weight = 0.40` for hedging + coordination.
- **Keep** `investor_trading_profit_delta` as primary profit signal.
- **Keep** return, quality, drawdown for risk-awareness.

---

## 5. Redundancy in Current Implementation

**investor_mtm_delta** and **investor_trading_profit_delta** both use `mtm_pnl`:

- `investor_mtm_delta` = 0.08 × clip(mtm_pnl/5000, ±1)
- `investor_trading_profit_delta` = 0.35 × clip(mtm_pnl/3000, ±1.5)

They are the **same signal** with different scaling. This doubles the gradient from MTM and can over-emphasize short-term noise. **Recommendation:** Keep only `investor_trading_profit_delta`; remove `investor_mtm_delta`.

---

## 6. Summary Table

| Question | Answer |
|----------|--------|
| Should investor have ONLY trading profit? | **No** — loses hedging incentive |
| Is base reward necessary? | **Partially** — hedging_effectiveness is critical; rest can be reduced |
| Redundancy? | **Yes** — mtm_delta and trading_profit_delta duplicate |
| Tier2 impact? | **None** — Tier2 trains on realized outcomes, not Tier1 reward |

---

## 7. Recommended Changes (Concrete)

1. **Remove** `investor_mtm_delta` (redundant with trading_profit_delta).
2. **Add** explicit `investor_hedging_reward_weight` (e.g. 0.25) scaling `hedging_score` from the calculator.
3. **Reduce** `investor_base_reward_weight` from 0.40 to 0.15 (keep light coordination).
4. **Keep** `investor_trading_profit_delta` as primary profit signal (weight 0.50).
5. **Keep** return, quality, drawdown terms for risk-awareness.

This yields: **profit (primary) + hedging (explicit) + risk penalties + minimal base**.
