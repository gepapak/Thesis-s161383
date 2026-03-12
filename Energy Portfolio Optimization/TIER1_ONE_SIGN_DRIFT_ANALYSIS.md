# Tier1 MARL One-Sign Drift Analysis

## Executive Summary

Your Tier1 MARL system shows **early one-sign drift** in episode 0, indicated by:
- `inv_mu_abs_roll ~= 0.535` — policy mean consistently biased toward one direction
- `inv_mu_sign_consistency = 1.0` — **all non-zero actions share the same sign** (always long or always short)
- `inv_sigma_raw tail mean ~= 0.276` — exploration std is reasonable
- Low saturation (~3.5% raw, ~1.6% sample, ~0.8% exposure) — not yet at boundary collapse

This analysis traces root causes through the codebase and proposes mitigations.

---

## Architecture Recap

### Tier1 (MARL)
- **Investor**: PPO with exposure-only action (1D scalar in [-1, 1])
- **Instruments**: wind, solar, hydro — all follow the **same price** (single-price hedging)
- **Allocation**: Risk budget weights (40% wind, 35% solar, 25% hydro) distribute exposure
- **Observations**: 6D — `[price_n, budget_n, wind_pos_norm, solar_pos_norm, hydro_pos_norm, mtm_pnl_norm]`

### Tier2 (DL Enhancer)
- Forecast-backed residual adjustment on top of Tier1 exposure
- Not active in Tier1-only training

---

## Root Cause Analysis

### 1. **Policy Mean Bias (inv_mu_abs_roll ≈ 0.535)**

**Where it comes from:**
- `environment.py` lines 5555–5564: `_investor_mean_abs_rolling` = mean of `|tanh(mu)|` over a 256-step window
- `metacontroller.py` 1753–1769: `mu0` = policy distribution mean (pre-tanh), `tanh_mu0` = post-tanh

**Interpretation:** The policy mean is consistently ~0.54 in magnitude. With `investor_ppo_mean_clip = 0.60`, the raw mean is clamped before tanh. A mean near 0.55–0.60 → tanh ≈ 0.50–0.54. So the policy is **structurally biased** toward one side.

**Likely drivers:**
- **ReLU / network init**: First layer biases or ReLU can produce asymmetric outputs
- **Observation regime**: If ep0 data has a dominant regime (e.g., rising prices), the policy can lock onto “long” or “short”
- **Reward early signal**: First few MTM/trading returns may favor one direction and reinforce it via PPO gradients

### 2. **Perfect Sign Consistency (inv_mu_sign_consistency = 1.0)**

**Definition** (`environment.py` 5559–5564):
```python
active = mu_vals[np.abs(mu_vals) > 0.05]
inv_mu_sign_consistency = |mean(sign(active))|
```
- 1.0 means every non-tiny action has the **same sign** — no flipping

**Why this is bad:**
- For single-price hedging, the investor should go long when price is expected to rise and short when it falls
- A policy that never flips cannot hedge; it behaves like a static directional bet
- PPO gradients from one-sided outcomes can further reinforce the bias

### 3. **inv_sigma_raw ~ 0.276**

- Investor: `log_std_init = -1.20` → std ≈ exp(-1.2) ≈ 0.30
- `inv_sigma_raw` is the distribution std; 0.276 is in line with config
- So exploration noise is present, but the **mean** is biased, so samples cluster on one side

### 4. **Low Saturation (Good So Far)**

- Raw/sample/exposure saturation are low — the policy is not yet pegged at ±1
- The concern is **directional** (one-sign) rather than magnitude saturation

---

## Code Paths Involved

| Component | File | Role |
|-----------|------|------|
| Policy creation | `metacontroller.py` 636–670 | ClampedActorCriticPolicy, log_std, mean_clip, sde_latent_clip |
| Action sampling | `metacontroller.py` 1700–1775 | Samples action, sets `_inv_mu_raw`, `_inv_sigma_raw`, `_inv_tanh_mu`, saturation flags |
| Mean tracking | `environment.py` 5549–5580 | `_investor_mean_history`, `_investor_mean_abs_rolling`, `_investor_mean_sign_consistency` |
| Penalties | `environment.py` 5575–5580 | `investor_mean_collapse_penalty` — **currently 0.0** (disabled) |
| Exposure mapping | `environment.py` 2840–2879 | `exposure_raw` → residual power → risk-budget allocation |

---

## Recommended Mitigations

### A. Enable Mean-Collapse Penalty (High Impact)

**Current:** `investor_mean_collapse_penalty = 0.0` (config.py 567)

**Change:** Turn on the existing penalty so prolonged one-sided means are penalized:

```python
# config.py
self.investor_mean_collapse_penalty = 0.15   # was 0.0
self.investor_mean_collapse_window = 256
self.investor_mean_collapse_warmup_steps = 4000
self.investor_mean_collapse_abs_mean_threshold = 0.25   # penalize if |mean| > 0.25
self.investor_mean_collapse_sign_threshold = 0.80      # penalize if sign consistency > 80%
```

This penalizes the investor when both (a) mean magnitude is high and (b) sign consistency is high.

### B. Observation Augmentation for Directional Balance

**Issue:** 6D obs may not give a strong “price direction” signal. Add a simple feature:

- `price_return_1step` or `price_momentum` (e.g., (price_t - price_{t-6}) / scale)
- This helps the policy distinguish “price rising” vs “price falling” and encourages sign flipping

**Location:** `observation_builder.py` `build_investor_observations` — extend obs to 7D and add momentum.

### C. Reduce Policy Mean Clip (Softer Bias)

**Current:** `investor_ppo_mean_clip = 0.60` (config.py 525)

**Change:** Lower to 0.45–0.50 so the policy cannot output as strong a one-sided mean early on:

```python
self.investor_ppo_mean_clip = 0.45
```

### D. Increase Entropy / Exploration Early

**Current:** `ent_coef = 0.030`, `investor_ppo_log_std_init = -1.20`

**Options:**
- Raise `ent_coef` to 0.04–0.05 for the first 50k–100k steps
- Slightly increase `investor_ppo_log_std_init` to -1.0 (std ≈ 0.37) for more exploration

### E. Concave Exposure Mapping

**Current:** `investor_exposure_power = 1.0` (linear)

**Change:** Use a concave map so interior actions still produce meaningful exposure:

```python
self.investor_exposure_power = 0.7   # or 0.8
```

Then `exposure = sign(x) * |x|^0.7` — smaller raw actions map to proportionally larger exposure, reducing the need to push toward ±1.

### F. Reward: Penalize One-Sided Exposure

Add a small penalty term when `inv_mu_sign_consistency` exceeds a threshold (e.g., 0.9) for several steps. This can be implemented in `_assign_rewards` alongside the existing collapse penalty logic.

### G. Data / Regime Check

- Inspect ep0 price series: is there a strong trend (e.g., mostly rising)?
- If so, consider:
  - Shuffling or augmenting episodes
  - Using `use_global_normalization = True` (already set) to avoid episode-specific scaling
  - Adding a “regime” or “trend” feature so the policy can condition on it

---

## Quick Diagnostic Commands

To inspect the debug CSV:

```bash
# Check inv_mu and sign consistency over time
python -c "
import pandas as pd
df = pd.read_csv('path/to/tier1_debug_ep0.csv')
print('inv_mu_abs_roll:', df['inv_mu_abs_roll'].describe())
print('inv_mu_sign_consistency:', df['inv_mu_sign_consistency'].describe())
print('inv_tanh_mu tail:', df['inv_tanh_mu'].tail(100).values)
"
```

---

## Summary Table

| Symptom | Cause | Mitigation |
|---------|-------|------------|
| inv_mu_abs_roll ≈ 0.535 | Policy mean biased | Enable mean-collapse penalty, lower mean_clip |
| inv_mu_sign_consistency = 1.0 | No sign flipping | Observation momentum, penalty, more entropy |
| inv_sigma_raw ~ 0.28 | OK | Keep as is |
| Low saturation | OK for now | Monitor; concave exposure mapping helps |

**Suggested order of changes:**
1. Enable `investor_mean_collapse_penalty` (A)
2. Lower `investor_ppo_mean_clip` (C)
3. Add price momentum to investor obs (B)
4. Consider concave `investor_exposure_power` (E)

These should reduce early one-sign drift while preserving the Tier1 MARL design and compatibility with Tier2.
