# Tehran SD–ABM–ML Urban Mobility Model

Computational repository for the study **“A Unified Hybrid SD–ABM–ML Simulation Framework for Sustainable Urban Mobility: Policy Experimentation for Tehran.”**

The repository contains the shared-state simulator, SC0–SC11 scenario definitions, calibration and validation workflows, robustness analyses, figure-generation code, a saved snapshot of the paper-facing computational outputs, and fixed machine-readable reference outputs used only for verification.

## Study scope

- Historical record: 2012–2023
- Calibration window: 2013–2021
- Terminal validation: 2022–2023
- Projection horizon: 2024–2030
- Model inventory: 706 typed computational agents (237 input, 400 SD, 57 ABM, 12 ML)
- Policy portfolio: one baseline, seven atomic interventions, and four integrated packages
- Computational figures: manuscript Figures 4–12 and Supplementary Figure S8

## Reproduce the computational results

Tested environment: Exact package versions are pinned in `requirements.txt` and `environment.yml`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/check_environment.py
python reproduce.py --all --workers 4
```

A complete run executes SC0–SC11, rebuilds validation and decision outputs, reruns the supplementary uncertainty and robustness analyses, reruns the local calibration-recovery evidence, and regenerates the paper-facing figures and source-data tables.

Fresh outputs are written only under `reproduced/`. Production and figure-generation scripts do not read from `reference/`.

## Repository structure

- `src/core/` — executable model, harmonized model input, fixed configuration, scenario overlays, KPI definitions, and figure generators
- `src/supplementary_analyses/` — uncertainty, sensitivity, ablation, hard-budget, and exogenous-stress analyses
- `src/calibration_recovery/` — parameterization evidence and local recovery workflow
- `paper_outputs/` — saved paper-facing output snapshot: Figures 4–12, Figure S8, and their machine-readable source tables
- `reference/` — frozen expected outputs used only by the independent verifier after a fresh run has completed
- `tools/` — environment checks, paper-result checks, artifact export, and independent reference comparison

The full run regenerates the same `paper_outputs/` layout under `reproduced/paper_outputs/`. The verifier compares the fresh run with `reference/`, while the saved publication snapshot can be compared path-for-path with `reproduced/paper_outputs/`.

## Input, derived data, and reference outputs

Names such as `figure_inputs/` refer to a file's role within a later plotting stage. Those files are still generated outputs of upstream model steps. The only primary model inputs are the harmonized data, model configuration, scenario definitions, and analysis specifications stored under `src/`.

`reference/` is never a computational input. It stores expected outputs from the fixed manuscript run so that a fresh reproduction can be checked independently.

## Verification

The complete workflow checks the numerical manuscript claims, validates the required publication artifacts, and compares regenerated outputs with the frozen references. A nonzero exit code indicates a failed check.

## Publication

Article DOI / URL: **[TO BE ADDED AFTER PUBLICATION]**

## Data and license

Data provenance and reuse notes are provided in `DATA_AND_LICENSE.md`.
