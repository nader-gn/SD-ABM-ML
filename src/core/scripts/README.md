# Core script entry points

The recommended entry point is the repository-level `reproduce.py`.

- `run_all_scenarios.py` — executes one of SC0–SC11 from the fixed configuration
- `extract_manuscript_results.py` — rebuilds reported KPI summaries
- `generate_decision_architecture.py` — rebuilds decision-score inputs and results
- `generate_truth_validation.py` — rebuilds historical and terminal validation evidence
- `run_figures.py` — regenerates manuscript Figures 4–12
- `audit_logic.py` and `run_model_consistency_checks.py` — executable model-consistency checks

The paper-facing figure/source-data mapping is written to `paper_outputs/FIGURE_DATA_MAP.csv` and regenerated under `reproduced/paper_outputs/FIGURE_DATA_MAP.csv`.
