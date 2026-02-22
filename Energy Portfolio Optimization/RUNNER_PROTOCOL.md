# Runner Protocol (Tier Phase Multi-Seed)

This project uses `run_tier_phase_multi_seed.py` as the canonical multi-seed runner.

## What Is Shared vs Method-Specific

- Shared across all runs in a suite:
  - core optimizer/training args (episodes, investment frequency, normalization mode)
  - evaluation path and unseen-data evaluation flow
- Shared across all forecast-enabled variants (Tier 2 / Tier 3 / forecast-ablated):
  - `--fgb_lambda_nonnegative`
  - `--fgb_clip_adv 0.15`
  - `--forecast_trust_window 500`
  - `--forecast_trust_metric hitrate`
  - `--forecast_trust_boost 0.0`
- Method-specific:
  - Tier 2 vs Tier 3 mode selection (`--fgb_mode online|meta`)
  - Tier 3 meta head enablement (`--meta_baseline_enable`)
  - forecast ablation switch (`--fgb_ablate_forecasts`)

## Reproducibility Artifact

Each suite writes `phase_protocol.json` in the suite directory.  
That file records the exact shared arguments and the protocol note used for the run.

## Fairness Interpretation

Using shared `fgb_shared_args` across all forecast-enabled variants is a protocol choice.
It is not a per-tier hidden override.  
If strict equal-hyperparameter ablations are needed for a paper section, report that as a separate experiment block.
