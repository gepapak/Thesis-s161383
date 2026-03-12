# Runner Protocol (Tier Phase Multi-Seed)

This project uses `run_tier_phase_multi_seed.py` as the canonical multi-seed runner.

## What Is Shared vs Method-Specific

- Shared across all runs in a suite:
  - core optimizer/training args (episodes, investment frequency, normalization mode)
  - evaluation path and unseen-data evaluation flow
- Shared across all Tier-2 variants:
  - forecast model/cache directories
  - PPO settings pinned by the runner for canonical seed suites
  - deployed-vs-paper tier evaluation mode written into `phase_protocol.json`
- Method-specific:
  - Tier 1 baseline: no enhancer flags
  - Tier 2 full: `--forecast_baseline_enable`
  - Tier 2 feature ablation: `--forecast_baseline_enable --tier2_enhancer_ablate_forecast_features`

## Reproducibility Artifact

Each suite writes `phase_protocol.json` in the suite directory.  
That file records the exact shared arguments and the protocol note used for the run.

## Fairness Interpretation

Using shared runner args across Tier-2 variants is a protocol choice.
It is not a per-tier hidden override.  
If strict equal-hyperparameter ablations are needed for a paper section, report that as a separate experiment block.
