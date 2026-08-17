# Source tree

This directory contains only executable inputs and analysis code required by `reproduce.py`.

- `core/`: central SD–ABM–ML model, harmonized executable input, scenarios, KPI/decision logic, and manuscript Figures 4–12.
- `supplementary_analyses/`: rolling-origin validation, ML-error propagation, mechanism/feasibility stresses, and Supplementary Figure S8.
- `calibration_recovery/`: supplementary identifiability and recovery/consistency evidence based on a fixed candidate design with objective values recomputed during reproduction.

Generated results never belong in this tree; they are written under `reproduced/`.
