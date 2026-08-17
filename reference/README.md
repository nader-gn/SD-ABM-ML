# Frozen reference outputs

This directory contains expected machine-readable outputs from the fixed manuscript run. It is **verification-only**: production, analysis, and figure-generation scripts do not read from this directory, and `reproduce.py` never writes into it.

The directory mirrors the relevant generated-output categories:

- `core/outputs/` — SC0–SC11 trajectories and derived decision/KPI outputs
- `core/figure_inputs/` — upstream-generated data consumed by later plotting steps
- `core/verification/` — validation, structural, and logic-check outputs
- `supplementary/validation/` — rolling-origin and residual-scale outputs
- `supplementary/uncertainty/` — Gaussian and residual-block propagation outputs
- `supplementary/mechanisms/` — ablation and mechanism-sensitivity outputs
- `supplementary/hard_budget/` — hard-budget stress outputs
- `supplementary/exogenous_stress/` — exogenous-stress scoring outputs
- `supplementary/calibration/` — recovery, historical-fit, and downstream-validation outputs

A name such as `figure_inputs` is stage-relative: it is an input to a plotting script but an output of an upstream computational step. Fresh equivalents are generated under `reproduced/workspace/` and compared here only after the run is complete.
