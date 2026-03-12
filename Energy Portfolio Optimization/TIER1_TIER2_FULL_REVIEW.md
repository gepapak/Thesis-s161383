# Tier1 & Tier2 Full System Review

**Review date:** Post profitable-trading refactor  
**Design intent:** Investor optimizes for profitable trading (not hedging)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TIER1: MARL (4 agents)                                                           │
│ • investor_0 (PPO): exposure scalar [-1,1] → wind/solar/hydro allocation        │
│ • battery_operator_0 (PPO): charge/discharge                                    │
│ • risk_controller_0 (PPO): risk multiplier                                     │
│ • meta_controller_0 (SAC): capital allocation + investment_freq                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TIER2 (optional): DL Enhancer                                                    │
│ • forecast_baseline_enable=True                                                  │
│ • Learns residual exposure delta on top of Tier1 proposal                        │
│ • 29D features (5D core + 3×8 forecast memory) or 5D ablated                      │
│ • Trains on realized investor sleeve return                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tier1: Current State

### 2.1 Investor Observations (6D)

| Index | Feature | Source | Purpose |
|-------|---------|--------|---------|
| 0 | price_momentum | (price_t - price_{t-6}) / scale, clip to [-1,1] | Directional signal: rising vs falling |
| 1 | budget_n | budget / init_budget | Capital availability |
| 2 | wind_pos_norm | exposure / max_pos | Current wind exposure |
| 3 | solar_pos_norm | exposure / max_pos | Current solar exposure |
| 4 | hydro_pos_norm | exposure / max_pos | Current hydro exposure |
| 5 | mtm_pnl_norm | cumulative_mtm / init_budget | Trading PnL state |

**Momentum scale:** `investor_price_momentum_scale = 0.08` → 8% return over 6 steps maps to ±1.

### 2.2 Investor Reward (Profitable Trading Focus)

| Component | Weight | Status |
|-----------|--------|--------|
| investor_base | 0.10 | Minimal fund anchor |
| investor_mtm_delta | 0 | Removed (redundant) |
| investor_trading_profit_delta | 0.60 | **Primary** profit signal |
| investor_hedging_delta | 0 | Disabled (profitable trading, not hedging) |
| investor_trading_return_delta | 0.45 | Rolling realized return |
| investor_trading_quality_delta | 0.25 | Sharpe-like |
| investor_trading_drawdown_penalty | 0.20 | Sleeve drawdown |

### 2.3 Action Flow

1. **Policy** outputs exposure_raw ∈ [-1, 1]
2. **Exposure mapping:** `exposure = sign(raw) * |raw|^power` (power=1.0)
3. **Tier2** (if enabled): `exposure += enhancer_delta` (delta ∈ [-0.35, 0.35])
4. **Allocation:** exposure × risk_budget_weights [0.40 wind, 0.35 solar, 0.25 hydro]
5. **Sizing:** target_pos = alloc × (tradeable_capital × max_position_size 0.35)
6. **Execution:** every `investment_freq`=6 steps (1h)

### 2.4 MTM & Execution Order

- **MTM before trades** (correct): `_add_mtm_for_step` → `_execute_investor_trades`
- **Single price:** All instruments use same `price_return` (FinancialEngine)
- **MTM loss exit:** 6% threshold forces exit on large unrealized loss

### 2.5 Risk Controls

- **Drawdown gates:** soft/medium/hard → reduce or disable trading
- **Volatility brake:** 2.5× median vol → 20% size reduction
- **Risk controller:** multiplier on tradeable_capital
- **Transaction costs:** fixed + bps

---

## 3. Tier2: Current State

### 3.1 When Active

- `forecast_baseline_enable=True` (CLI: `--forecast_baseline_enable`)
- Requires forecast generator (trained models or cache)

### 3.2 Enhancer Features (Streamlined)

**Full (12D):**
- 4D core: proposed_exposure, gross_exposure_ratio, tradeable_capital_ratio, volatility_regime
- 2×4 forecast memory: price_short_signal, short_revision, forecast_quality, short_imbalance_signal

**Ablated (4D):** Core only, no forecast features.

### 3.3 Enhancer Training

- **Target:** `_compute_enhancer_delta_target` from interval_score (return, drawdown, Sharpe)
- **Source:** Realized investor sleeve return over decision intervals
- **Independent of Tier1 reward:** Tier2 does not use investor_reward; it uses outcomes

### 3.4 Enhancer Output

- `exposure_delta` ∈ [-0.35, 0.35]
- `exposure_final = clip(exposure_tier1 + delta, -1, 1)`

---

## 4. Consistency Check

| Aspect | Tier1 | Tier2 | Consistent? |
|--------|-------|-------|-------------|
| Observation | 6D, price_momentum | N/A (enhancer has own features) | ✓ |
| Investor objective | Profitable trading | Improves realized return | ✓ |
| Hedging | Disabled | N/A | ✓ |
| Policy obs | No forecast | Enhancer uses forecast | ✓ (by design) |

---

## 5. Potential Issues

### 5.1 Stale Comment in environment.py

Line ~5473: "Critical for single-price design; pure profit alone would not encourage hedging" — hedging is now disabled. Comment is outdated but harmless.

### 5.2 Base Reward Still Contains Hedging

`ProfitFocusedRewardCalculator` base_reward includes `hedging_effectiveness` (15% weight). With `investor_base_reward_weight=0.10`, the investor gets 10% × 15% = 1.5% of total reward from hedging. Minor; could remove if desired.

### 5.3 Meta Controller Signal

`investor_meta_signal` includes `investor_hedging_delta` (0 when disabled). Meta uses this for capital allocation. With hedging off, meta signal = profit + return + quality - drawdown. Coherent.

### 5.4 Risk Budget Allocation

Config still uses "SINGLE-PRICE HEDGING" and "risk budget allocation" language. The allocation (40/35/25) is just exposure distribution across instruments; it doesn't imply hedging. No change needed.

---

## 6. Summary: Current Design

| Layer | Role |
|-------|------|
| **Tier1 investor** | Profitable trading: momentum obs, profit reward, return/quality/drawdown |
| **Tier1 other agents** | Battery arbitrage, risk control, capital allocation |
| **Tier2 enhancer** | Forecast-backed residual adjustment on Tier1 exposure; trains on realized return |

**Design is coherent** for profitable-trading focus. Tier1 and Tier2 are aligned; no conflicts.
