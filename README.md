# Tehran SD–ABM–ML Urban Mobility Model

A hybrid simulation framework for exploring how urban mobility policies interact with travel behavior, congestion, public transport, emissions, health, and public finance in Tehran.

This repository accompanies the study **“A Unified Hybrid SD–ABM–ML Simulation Framework for Sustainable Urban Mobility: Policy Experimentation for Tehran.”** The associated manuscript is not yet published; the final citation and DOI will be added here when available.

The project is intended as a **policy experimentation model**, not a real-time traffic system, traveler-level microsimulation, or unconditional forecasting tool. It asks a practical question: *how do mobility policies behave once behavioral adaptation, system feedback, environmental effects, and implementation constraints are allowed to interact over time?*

---

## What the model does

The framework combines three modeling roles inside one annual shared state:

- **System Dynamics (SD)** represents aggregate stocks, flows, delays, accounting relationships, and feedbacks across transport, environment, health, and finance.
- **Agent-Based Modeling (ABM)** represents policy mediation at a meso level: exposure to policies, access friction, compliance/convenience effects, mode competition, and feasibility before outcomes are aggregated.
- **Machine Learning (ML)** supplies nonlinear modal priors from temporally admissible lagged information. These priors are inputs to the hybrid model; they are not the final modal-share predictions on their own.

The central idea is not to run SD, ABM, and ML as three independent models and pass files between them. They operate on a **shared simulation state**, with explicit update order and annual state commitment. At the end of each simulated year, the resolved state is committed and becomes the endogenous memory used in the next annual step. This makes delayed effects, staged policies, and feedback responses part of the same trajectory.

---

## Conceptual structure

The model is organized around several interacting feedback channels:

1. **Congestion-mediated adjustment** — higher activity and motorized loading increase congestion and travel time, which can moderate demand and private-motorized use.
2. **Modal substitution** — changes in travel cost, access, parking, or service quality alter the relative attractiveness of transport modes.
3. **Public-transport reinforcement** — service, fleet, network, speed, access, and reliability improvements affect the attractiveness and operating requirements of public transport.
4. **Fiscal feasibility** — policy and service commitments create operating and capital requirements that are tracked alongside physical outcomes.
5. **Revenue recycling** — access-pricing revenue can contribute to transport financing without being treated as a direct welfare benefit.
6. **Environment and health response** — vehicle activity, technology, energy use, and emission factors propagate to pollutant and health-burden indicators.

These mechanisms are evaluated together because a policy that improves one outcome can create a trade-off elsewhere. Electrification, for example, can reduce tailpipe emissions without relieving congestion; service expansion can shift mode share while increasing operating requirements; and access restrictions can reduce car pressure without automatically improving local particulate emissions.

---

## Spatial and temporal scope

The Tehran application uses nested policy scales rather than treating the city as spatially uniform:

- **Tehran** — metropolitan-scale transport, environmental, social, and fiscal dynamics.
- **Region 12 (R12)** — the policy-intensive central district, where access regulation, daytime activity, congestion, and local exposure are more concentrated.
- **Traffic Plan / TFP layer** — the policy channel linking access-sensitive trips, regulatory friction, charging, and potential revenue recycling.

The main model timeline is:

| Period | Role |
|---|---|
| 2012–2023 | Harmonized historical record and historical reconstruction |
| 2013–2021 | Main calibration window |
| 2022–2023 | Terminal holdout / validation window |
| 2024–2030 | Conditional policy-experiment horizon |

The 2024–2030 trajectories should be interpreted as **scenario-conditional responses under the model assumptions**, not as unconditional forecasts of Tehran’s future.

---

## ML, ABM, and SD roles in practice

### Machine-learning layer

The current model contains **12 modal-prior surrogates**: six transport modes for Tehran and the same six for Region 12. The modeled modes are motorcycle, car, taxi, bus, metro, and other.

The surrogates use gradient-boosted decision trees on a logit scale and rely on admissible lagged features such as previous modal shares, cost/access conditions, public-transport service variables, infrastructure/fleet conditions, parking, and macro-demographic descriptors. Fiscal accounting variables are kept outside the modal-prior feature matrix.

ML outputs are bounded priors. Final modal shares are resolved only after ABM mediation and modal closure.

### ABM layer

The ABM component is **meso-level**, not a synthetic population of individual travelers. It captures mode-, region-, and policy-specific differences in response, access, exposure, convenience, and competition.

In this repository, the word **agent** means a typed computational unit in the dependency graph. The configured model contains **706 typed computational agents** across input, SD, ABM, and ML roles; this should not be interpreted as 706 traveler agents.

### System-dynamics layer

After modal closure, SD relationships propagate the consequences of the resolved mobility state through travel demand, network loading, congestion, fleet and energy use, emissions, health burden, service requirements, and fiscal accounting. Stocks, delays, and committed state memory carry these effects across years.

---

## Policy experiments

The repository includes one baseline, seven atomic interventions, and four integrated policy packages. Scenario schedules are fixed experiment definitions rather than optimized policy intensities.

| Scenario | Type | Main idea |
|---|---|---|
| **SC0** | Baseline | Closed-loop continuation without a new policy perturbation |
| **SC1** | Atomic | Demand smoothing through flex-time, selective telework, and staggered scheduling |
| **SC2** | Atomic | Staged access charging in the traffic-plan zone |
| **SC3** | Atomic | Parking and curb-management restraint in Region 12 with Tehran-scale spillover |
| **SC4** | Atomic | Targeted support for bus and metro fares |
| **SC5** | Atomic | Public-transport speed, access, network, fleet, and reliability improvement |
| **SC6** | Atomic | Local pollutant-intensity reduction for motorcycles, taxis, and buses |
| **SC7** | Atomic | Taxi/bus electrification and fleet renewal |
| **SC8** | Package | Balanced combination of demand smoothing, light parking restraint, fare support, and pollutant cleanup |
| **SC9** | Package | Access-led package combining charging, parking, fare/service support, and cleanup |
| **SC10** | Package | Public-transport-first clean package with service, fare, cleanup, electrification, and light parking measures |
| **SC11** | Package | Broad package combining demand, access, parking, PT service, cleanup, and fleet transition |

Scenario definitions are stored in [`src/core/scenarios/`](src/core/scenarios/). The machine-readable registry is [`scenario_registry.csv`](src/core/scenarios/scenario_registry.csv).

---

## What comes out of a run

Each scenario produces an annual system trajectory rather than a single score. Depending on the analysis stage, outputs include:

- modal shares and trips by mode;
- congestion, travel time, network loading, and vehicle-kilometers traveled;
- public-transport service and operating indicators;
- fuel and electricity use;
- CO₂, NOₓ, PM₂.₅, and related environmental indicators;
- health-burden and social indicators;
- policy revenue, OPEX/CAPEX, subsidy, and fiscal-pressure indicators;
- baseline-relative scenario comparisons;
- four-domain outcome summaries for Transportation, Environmental, Social, and Economic performance;
- a separate implementation screen so outcome desirability is not collapsed into implementation burden.

The saved manuscript-facing figures and source tables are available under [`paper_outputs/`](paper_outputs/). They are useful for browsing the current analysis without rerunning the complete workflow.

---

## Repository map

```text
.
├── README.md
├── DATA_AND_LICENSE.md
├── requirements.txt
├── environment.yml
├── reproduce.py
│
├── src/
│   ├── core/
│   │   ├── config/                  # central model, harmonized input, analysis settings
│   │   ├── scenarios/               # SC1–SC11 policy overlays and scenario registry
│   │   ├── scripts/                 # scenario runner, validation, KPI and figure workflows
│   │   └── tables/                  # model-role and KPI registries
│   │
│   ├── supplementary_analyses/      # uncertainty, sensitivity, ablation and stress analyses
│   └── calibration_recovery/        # parameterization and local recovery evidence
│
├── paper_outputs/                   # current manuscript-facing figures and source data
├── reference/                       # frozen outputs used only for verification
└── tools/                           # environment, export and consistency checks
```

For most users, `src/core/` is the important part of the repository. The supplementary and calibration directories contain the additional analyses used to examine model sensitivity, uncertainty, mechanism dependence, and parameterization evidence.

`reference/` is **not** used as a model input. It exists only for independent checking of a regenerated run.

---

## Getting started

### 1. Create a Python environment

The tested environment uses Python 3.13 and the package versions listed in `requirements.txt`.

Use Python 3.13 in your preferred virtual or Conda environment, then install the project dependencies:

```console
python -m pip install -r requirements.txt
```

A Conda environment definition is also provided in `environment.yml`.

### 2. Run the central model workflow

To execute the baseline and policy scenarios together with the main validation, KPI, decision, and figure routines:

```console
python reproduce.py --core
```

This creates a fresh working area under `reproduced/` and leaves the source model unchanged.

### 3. Run a single scenario

For quick model exploration, an individual scenario can be executed directly:

```console
python src/core/scripts/run_all_scenarios.py --root src/core --scenario SC8
```

Replace `SC8` with any scenario from `SC0` to `SC11`. The resulting trajectory is written to:

```text
src/core/outputs/simulation_data_SC8.csv
```

### 4. Run the full analysis suite when needed

The complete workflow additionally runs the supplementary uncertainty, robustness, calibration-recovery, export, and verification steps:

```console
python reproduce.py --all --workers 4
```

This is mainly useful when rebuilding the complete set of supplementary analyses, exported artifacts, and verification outputs associated with the project.

---

## Working with scenarios

Policy schedules are implemented as YAML overlays in `src/core/scenarios/`. SC0 uses the unmodified baseline configuration; SC1–SC11 change only their declared policy channels.

The central configuration is in:

```text
src/core/config/BASE_CONFIG.yaml
```

The main executable model is in:

```text
src/core/config/system_core.py
```

The harmonized model input is in:

```text
src/core/config/DATA_clean.csv
```

If you create a new experimental scenario, keep its policy assumptions explicit and separate from the baseline configuration. This makes it easier to interpret whether an outcome is driven by demand management, access regulation, public-transport support, fleet transition, pollutant control, or a combination of channels.

---

## Reading model results

A few interpretation rules are important:

- **Scenario effects are comparative.** SC0 is the common reference for SC1–SC11.
- **Shares and levels are different.** A mode share can increase even when absolute trips fall if total travel demand contracts.
- **Environmental and congestion effects are not interchangeable.** A clean-fleet policy can improve tailpipe emissions while leaving road-space pressure almost unchanged.
- **Implementation is evaluated separately from outcomes.** A strong environmental or transport package may also require larger CAPEX, OPEX, subsidies, or metropolitan financing.
- **Tehran and Region 12 should not be interpreted as scaled copies of each other.** The central district has different activity intensity, regulation exposure, congestion, and local-pollutant dynamics.
- **Annual resolution limits the claims.** The framework is designed for comparative policy pathways and feedback reasoning, not minute-by-minute traffic operations or traveler-level distributional analysis.

---

## Current manuscript and project status

This repository is associated with an unpublished manuscript. Until a final journal version is available, please treat the repository title and manuscript title as the preferred project identifiers.

**Manuscript:**  
*A Unified Hybrid SD–ABM–ML Simulation Framework for Sustainable Urban Mobility: Policy Experimentation for Tehran*

**Citation / DOI:** to be added after publication.

---

License and Attribution

Source code in this repository is licensed under the Apache License 2.0. Repository-owned non-software research materials are made available under CC BY 4.0, unless otherwise stated.

Third-party and upstream data remain subject to the terms of their original providers.

For full licensing and attribution terms, see [`LICENSE`](LICENSE) and [`DATA_AND_LICENSE.md`](DATA_AND_LICENSE.md).
---

## Short project summary

In one sentence: **this project uses a shared-state SD–ABM–ML model to test how Tehran mobility policies propagate through behavior, congestion, public transport, emissions, health, and finance over time, and to compare those pathways without reducing the system to a single objective.**
