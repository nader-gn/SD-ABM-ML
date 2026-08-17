# Central model

`config/BASE_CONFIG.yaml`, `config/system_core.py`, and `config/DATA_clean.csv` define the executable central model and harmonized input used by the paper workflow. Scenario overlays SC1–SC11 are under `scenarios/`; SC0 is the unmodified baseline configuration. `scenarios/scenario_registry.csv` provides the scenario metadata used in the paper.

`reproduce.py` executes each scenario in an isolated process and then rebuilds reported KPI tables, decision outputs, validation evidence, logic checks, and Figures 4–12.

### Executable ontology and connectivity verification

`tables/functional_role_registry.csv` explicitly declares the 57 policy-mediation nodes used for the ABM functional view. During reproduction, `scripts/export_execution_audits.py` combines that registry with `BASE_CONFIG.yaml`, the engine dependency graph, and documented external-support bridges to generate:

- `verification/agent_role_mapping.csv`
- `verification/agent_role_summary.csv`
- `verification/execution_graph_audit.csv`
- `verification/agent_connectivity_audit.csv`
- `verification/agent_connectivity_summary.csv`

The connectivity verification confirms 706 retained typed agents: 237 input, 400 SD, 57 ABM, and 12 ML functional roles. Structurally, the same inventory contains 237 inputs, 441 expressions, 16 stocks, and 12 ML surrogates. All 706 execution groups are singletons, with zero multi-node groups, zero self-loops, and 44 delayed nodes.

The connectivity verification requires zero unexplained unused configured nodes. One engine-isolated input (`job_r12`) is retained intentionally because it supplies the lagged R12 jobs feature used by the ML surrogates through the training/serving feature bridge. The TFP trip subgraph is retained because it is consumed by decision-architecture post-processing. These documented support roles are therefore not classified as unused.

### Reproduction scope

The central engine implements the reported model runtime. It implements fixed ML serving, the shared-state annual runner, topological/SCC safeguards, delayed-node handling, declared constraints, and scenario-overlay execution. Validation, uncertainty, sensitivity, stress testing, and calibration recovery are invoked by the dedicated reproduction scripts. Runtime feature selection remains disabled in the repository configuration.
