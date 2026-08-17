# Supplementary analyses

This directory contains the paper's supplementary validation and robustness analyses. None of these scripts overwrites the central SC0–SC11 configuration.

The workflow includes:

- rolling-origin ML validation and residual-scale estimation;
- 32-draw Gaussian ML-prior propagation over SC0–SC11 (384 complete closed-loop runs);
- 8 empirical/sign-reversed residual-block stresses over SC0–SC11 (96 complete closed-loop runs);
- ABM mediation ablation and SC5/SC6 mechanism sensitivities;
- SC10 hard-budget stress;
- exogenous demand/fiscal stress and decision-score summaries;
- independent central-freeze replay;
- Supplementary Figure S8 and its machine-readable panel data.

`ensemble_engine.py` uses persistent worker processes so fitted ML state is loaded once per worker while preserving the same numerical task definitions.
