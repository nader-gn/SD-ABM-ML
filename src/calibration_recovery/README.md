# Calibration recovery analysis

This directory contains two complementary reproducibility layers: (i) a parameterization-evidence replay for historical fitting, calibration, anchoring, and state-alignment rules, and (ii) the local identifiability and recovery/consistency analysis reported in the supplementary evidence.

`reproduce_parameterization_evidence.py` reconstructs the parameterization architecture directly from the repository code and data. It checks all 55 mapped historical input agents and all 16 stock agents; reproduces the 2012–2023 population-flow calibration; replays the four inflation/fare accounting indices; verifies network target tracking and public-fleet stock-flow reconstruction; checks historical-series/projection locks; and exactly reconstructs the bus and metro projection fare bases. The population migration coefficient is recovered from the observed log population-growth rate after accounting for the configured birth rate and baseline demographic mortality; the replay value (0.00555739) rounds to the configured 0.00556. Quantities without a direct historical stock target in the repository are explicitly classified rather than pseudo-calibrated.

`reproduce_full_calibration_evidence.py` then:

1. rebuilds the 2013–2021 historical objective;
2. reruns the reported one-at-a-time ±10% identifiability screens and an expanded check of plausible upstream calibration scalars;
3. re-evaluates the stored TPE and independent-random candidate parameter vectors through `historical_objective.py`;
4. recomputes lambda/gamma sensitivity paths; and
5. verifies recovery of the configured values under the reported regularization setting.

`calibration_candidate_design.csv` stores candidate parameter vectors only. Objective values are recomputed during reproduction. The recovery analysis is separate from the parameterization routes used to construct the configured central specification.

Most agents are not independently calibrated. They are deterministic or stock-flow descendants of fitted ML priors, historical inputs/anchors, calibrated upstream parameters, and fixed structural relations. Per-agent HPO would double-use historical information, increase equifinality, and obscure dependency ownership; it is therefore not part of the deployed methodology. HPO/recovery is reserved for free scalar quantities that are historically informative and not already observed, algebraically reconstructible, mapped by annual series, structurally redundant, or verification-only.

## ML-propagated outcome validation

`summarize_ml_propagation_validation.py` parses the executable dependency graph, identifies observed outputs downstream of the 12 fitted ML modal prior agents, and adds a second validation layer beyond the final modal shares. It reports 2013-2021 historical reconstruction and a conditional 2022-2023 terminal holdout for 17 additional endogenous downstream consequences. Two same-concept empirical anchors (`modal_share_bik` and `pop_r12`) are identified but excluded from independent propagation-accuracy claims. The analysis assesses the propagated hybrid pathway; it is not causal attribution to ML alone.
