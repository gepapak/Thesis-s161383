# Deep Codebase Review: Tier1 MARL, Tier2 Expert Bank, Forecast Models & Evaluation

**Reviewer:** AI Engineer  
**Date:** March 16, 2025  
**Scope:** Tier1 MARL system, Tier2 expert bank, forecast model training/existence checks, legacy cleanup, evaluation flow

---

## Executive Summary

| Component | Status | Notes |
|-----------|--------|------|
| **Tier1 MARL** | ✅ Correct | Investor (PPO), Battery (DQN); Meta & Risk are rule-based |
| **Tier2 Expert Bank** | ✅ Correct | 5 experts: ANN, LSTM, SVR, RF, CEEMDAN-LSTM; price + short horizon only |
| **Forecast Model Existence** | ✅ Working | `check_episode_models_exist` → `ensure_episode_forecasts_ready` triggers training |
| **Folder Structure** | ✅ Correct | `forecast_models/episode_N/price_short_experts/{ANN,LSTM,SVR,RF,CEEMDAN_LSTM}/` |
| **Legacy Code** | ⚠️ Present | Several legacy paths for wind/solar/hydro/load, medium/long horizons |
| **Evaluation** | ✅ Working | Episode 20 reserved; auto-trains if missing; Tier comparison supported |

---

## 1. Tier1: MARL System (Meta & Risk Rule-Based)

### 1.1 Architecture

- **`environment.py`**: `RenewableMultiAgentEnv` with 4 agents:
  - `investor_0` → PPO (RL)
  - `battery_operator_0` → DQN (RL)
  - `risk_controller_0` → RULE
  - `meta_controller_0` → RULE

- **`config.py`** (lines 519–520):
  ```python
  self.risk_controller_rule_based = True
  self.meta_controller_rule_based = True
  ```

- **`metacontroller.py`**: `MultiESGAgent` trains PPO/DQN for investor/battery; meta and risk use rule-based logic.

**Verdict:** Tier1 is correctly implemented as MARL with rule-based meta and risk.

---

## 2. Tier2: Expert Bank (Price, Short Horizon Only)

### 2.1 Expert Bank Design

- **`forecast_price_experts.py`**:
  - `PRICE_SHORT_EXPERT_METHODS`: `("ann", "lstm", "svr", "rf", "ceemdan_lstm")`
  - Target: `price` only
  - Horizon: `short` (6 steps)
  - Look-back: 24 steps

- **`tier2_expert_bank.py`**: Builds 40D feature vector (4D core + 2×18 memory channels) for defer-and-route controller.

- **`tier2_routed_overlay.py`**: Short-horizon conformal decision-focused controller using the expert bank.

**Verdict:** Tier2 is correctly scoped to price and short horizon with a multi-expert bank.

---

## 3. Forecast Model Existence Check & Training Trigger

### 3.1 Existence Check

1. **`price_short_expert_bank_exists(episode_dir)`** (`forecast_price_experts.py` 316–330)
   - Requires for each method: `metadata_path`, `model_path`, `scaler_x_path`, `scaler_y_path`
   - CEEMDAN-LSTM also needs `feature_scaler_path`

2. **`check_episode_models_exist(episode_num, forecast_base_dir)`** (`forecast_engine.py` 495–500)
   - Builds `MultiHorizonForecastGenerator` via `_build_forecaster_for_episode`
   - Returns `forecaster.is_complete_stack()`
   - On exception (e.g. dirs missing): returns `False`

3. **`is_complete_stack()`** (`generator.py` 1759–1780)
   - Expert-only mode: all 5 experts loaded and `has_price_short_expert_bank()`

### 3.2 Training Trigger

1. **`ensure_episode_forecasts_ready()`** (`forecast_engine.py` 518–553)
   - If `check_episode_models_exist()` is False → calls `train_episode_forecasts_for_episode()`

2. **`main.py`** (Tier2 mode with `--forecast_baseline_enable`)
   - Calls `ensure_episode_forecasts_ready()` before each episode

3. **`evaluation.py`** (`_load_eval_forecast_paths_episode20`, lines 942–949)
   - If `check_episode_models_exist(20)` is False → `train_evaluation_forecasts(episode_num=20)`

**Verdict:** Existence check and training trigger logic are correct.

---

## 4. Forecast Model Folder Structure

### 4.1 Actual Structure

```
forecast_models/
  episode_N/
    price_short_experts/
      ANN/          # ann_model.keras, ann_scaler_x.pkl, ann_scaler_y.pkl, ann_metadata.json
      LSTM/         # lstm_model.keras, ...
      SVR/          # svr_model.pkl, ...
      RF/           # rf_model.pkl, ...
      CEEMDAN_LSTM/ # ceemdan_lstm_model.keras, ceemdan_lstm_feature_scaler.pkl, ...
    models/         # Empty in expert-only mode (legacy)
    scalers/        # Empty in expert-only mode (legacy)
    metadata/       # Empty in expert-only mode (legacy)
```

- **`get_price_short_expert_paths()`** (`forecast_price_experts.py` 299–313): Uses `PRICE_SHORT_EXPERT_LABELS` to map `ann` → `ANN`, etc.
- **`_load_price_short_expert_bank()`** (`generator.py` 622–624): Derives `episode_dir` from `model_dir` parent and loads from `episode_dir/price_short_experts/`.

**Verdict:** Folder structure matches design (ANN, RF, CEEMDAN, LSTM, SVR).

---

## 5. Legacy Code to Remove or Deprecate

### 5.1 High Priority (Remove or Clearly Deprecate)

| Location | Description | Recommendation |
|----------|-------------|----------------|
| **`forecast_engine.train_episode_model()`** (lines 120–369) | Generic target/horizon ANN for arbitrary targets/horizons | **Remove** – Not called by `train_episode_forecasts()`; only price-short expert bank is trained |
| **`generator.py` non-expert-only branch** (lines 488–502) | `agent_horizons` / `agent_targets` for wind, solar, hydro, load; immediate/medium/long/strategic | **Simplify** – When `forecast_price_short_expert_only=True` (default), this branch is unused; consider removing or guarding with `if not price_short_expert_only` |
| ~~`environment.py` legacy forecast deltas~~ | ~~wind/solar/hydro/load, medium/long~~ | **Done** – Removed 584 lines dead code; pre-alloc now price-only |
| ~~`OtherForecastModels/myfunctions_france.py`~~ | *(User deleted)* | Removed |

### 5.2 Medium Priority (Document or Simplify)

| Location | Description | Recommendation |
|----------|-------------|----------------|
| **`environment.py`** `_horizon_forecast_pairs` for medium/long | MAPE tracking for medium/long horizons | Keep for backward compatibility or remove if not used in paper setup |
| **`environment.py`** `z_medium_price`, `z_long_price`, `ema_std_medium`, `ema_std_long` | Multi-horizon z-scores and EMA | When expert-only, these are fed from fallbacks (current price); consider simplifying |
| **`forecast_engine`** `get_episode_forecast_dirs` | Creates `models/`, `scalers/`, `metadata/` | Required for `_validate_forecast_dirs`; keep but document that they are empty in expert-only mode |

### 5.3 Config Already Narrowed

- `config.forecast_targets = ["price"]`
- `config.required_forecast_horizons = ["short"]`
- `config.forecast_price_short_expert_only = True`

---

## 6. Evaluation Flow

### 6.1 Forecast Path Resolution

- **`_load_eval_forecast_paths_episode20()`** (`evaluation.py` 921–965):
  - Uses Episode 20 for evaluation (unseen 2025 data)
  - If `check_episode_models_exist(20)` is False → `train_evaluation_forecasts(episode_num=20)`
  - Returns `model_dir`, `scaler_dir`, `metadata_dir` for Episode 20

### 6.2 Tier Comparison

- **`evaluation.py`** supports `--mode tiers` for Tier1 baseline vs Tier2 vs Tier2 ablated
- **`_resolve_cv_weights()`**: Looks for `tier2_routed_overlay.h5` or `dl_control_variate.h5` in `final_models/` and `checkpoints/episode_*/`

### 6.3 Potential Issue

- **`_validate_forecast_paths`** requires `model_dir`, `scaler_dir`, `metadata_dir` to exist as directories.
- For expert-only mode these are created by `train_episode_forecasts` but remain empty.
- The generator loads experts from `episode_dir/price_short_experts/`; the empty dirs are only for validation.
- **Recommendation:** Consider validating `price_short_expert_bank_exists(episode_dir)` directly when in expert-only mode, to avoid depending on empty legacy dirs.

---

## 7. Recommendations Summary

### 7.1 Immediate

1. **Remove `forecast_engine.train_episode_model()`** – Dead code; only `train_price_short_expert_bank` is used.
2. **Add explicit check** – In expert-only mode, optionally call `price_short_expert_bank_exists(episode_dir)` before `_validate_forecast_dirs` to fail fast with a clear message when experts are missing.

### 7.2 Short Term

3. **Refactor `environment.py`** – Reduce or isolate wind/solar/hydro/load and medium/long horizon logic when `forecast_price_short_expert_only=True`.
4. **Clean `generator.py`** – Remove or guard the non-expert-only `agent_horizons`/`agent_targets` branch.
5. ~~**Remove or isolate `OtherForecastModels/myfunctions_france.py`**~~ – User deleted.

### 7.3 Documentation

6. **Add README section** – Document that Tier2 uses only price + short horizon and the expert bank folder layout.
7. **Document Episode 20** – Evaluation uses Episode 20; training uses Episodes 0–19 (or rolling past).

---

## 8. Verification Checklist

- [x] Tier1: MARL (PPO/DQN) for investor/battery; rule-based meta and risk
- [x] Tier2: Expert bank with ANN, LSTM, SVR, RF, CEEMDAN-LSTM
- [x] Target: price only
- [x] Horizon: short only (6 steps)
- [x] Forecast models checked before use
- [x] Training triggered when models missing
- [x] Folder structure: `price_short_experts/{ANN,LSTM,SVR,RF,CEEMDAN_LSTM}/`
- [x] Evaluation uses Episode 20; auto-trains if missing
- [ ] Legacy code removed or clearly deprecated (pending)
